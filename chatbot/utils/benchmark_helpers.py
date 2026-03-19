"""Shared helpers for parsing XLSX benchmark files and scoring IR results."""

import re

# ── Document-type normalisation map ──────────────────────────────────────────
# Maps raw evidence-text prefix → normalised prefix matching database doc_ids.
# User-confirmed mappings:
#   PERMEN PUPR → PERMENPUPR   | SK DIRJEN BK → SKDIRJENBK
#   PERMEN PPN  → PERMENPPN    | PERATURAN BPS → PERBANBPS
#   PERMEN PERDAGANGAN → PERMENDAG
JENIS_MAP = {
    "UU": "UU", "PP": "PP", "PERPPU": "PERPPU",
    "PERMEN PUPR": "PERMENPUPR", "PERMEN PPN": "PERMENPPN",
    "PERMEN PERDAGANGAN": "PERMENDAG", "PERMEN": "PERMEN",
    "PERMENKES": "PERMENKES", "PERMENDAGRI": "PERMENDAGRI",
    "PERDA": "PERDA", "PERGUB": "PERGUB", "PERWAKO": "PERWAKO",
    "SE": "SE",
    "SK DIRJEN BK": "SKDIRJENBK", "KEPDIRJEN": "KEPDIRJEN",
    "PERATURAN BPS": "PERBANBPS",
    "PERATURAN MENTERI PERENCANAAN PEMBANGUNAN": "PERMENPPN",
    "PERATURAN MENTERI": "PERMEN",
}

# Known cross-system doc_id prefix aliases.
# Neo4j may store "PERMEN-NASIONAL-8-2022" while VDB has "PERMENPUPR-NASIONAL-8-2022".
_ALIAS_PREFIXES = {
    "PERMENPUPR": "PERMEN",
    "PERMENPPN": "PERMEN",
    "PERMENDAG": "PERMEN",
    "PERBANBPS": "PERMEN",
    "SKDIRJENBK": "KEPDIRJEN",
}


def extract_documents(evidence: str) -> list[dict]:
    """Parse document references from evidence text.

    Returns list of ``{raw_ref, jenis, nomor, tahun, candidates}``.
    ``candidates`` contains all plausible doc_id forms for cross-system matching.
    """
    if not evidence:
        return []
    docs: list[dict] = []
    seen: set[tuple] = set()
    patterns = [
        r'(SK\s+Dirjen\s+BK)\s+(\d{4})',
        r'(Peraturan\s+(?:Menteri|BPS|Gubernur|Wali\s+Kota)(?:\s+[A-Za-z]+)*)(?:/[^0-9]+)?\s*(\d+/\d{4})',
        r'(Permen\s+[A-Za-z]+)\s+(\d+/\d{4})',
        r'(Permenkes|Permendagri)\s+(\d+/\d{4})',
        r'(PERGUB|Pergub)\s+(\d+/\d{4})',
        r'(Perppu|PERPPU)\s+(\d+/\d{4})',
        r'(Perda\s+[A-Za-z\s]+)\s+(\d+/\d{4})',
        r'(UU|PP)\s+(\d+/\d{4})',
    ]
    for pat in patterns:
        for match in re.finditer(pat, evidence, re.IGNORECASE):
            jenis_raw = match.group(1).strip()
            num_part = match.group(2).strip()
            if "/" in num_part:
                nomor, tahun = num_part.split("/", 1)
            else:
                nomor, tahun = "", num_part
            key = (jenis_raw.upper(), nomor, tahun)
            if key in seen:
                continue
            seen.add(key)
            jenis_upper = jenis_raw.upper().strip()
            jenis_norm = None
            for prefix, mapped in JENIS_MAP.items():
                if jenis_upper.startswith(prefix.upper()):
                    jenis_norm = mapped
                    break
            if not jenis_norm:
                jenis_norm = jenis_upper.split()[0] if jenis_upper else "UNKNOWN"

            # SK Dirjen BK has implicit nomor=12.1 (user-provided)
            if jenis_norm == "SKDIRJENBK" and not nomor:
                nomor = "12.1"

            candidates: list[str] = []
            if nomor:
                scope = "PROVINSI" if jenis_norm == "PERGUB" else "NASIONAL"
                # Primary canonical form
                candidates.append(f"{jenis_norm}-{scope}-{nomor}-{tahun}")
                # Cross-system alias (e.g. PERMENPUPR → PERMEN)
                alias_prefix = _ALIAS_PREFIXES.get(jenis_norm)
                if alias_prefix:
                    candidates.append(f"{alias_prefix}-{scope}-{nomor}-{tahun}")
                # SK DIRJEN BK underscore format used in VDB
                if jenis_norm == "SKDIRJENBK":
                    candidates.append(
                        f"SK_DIRJEN_BK_TAHUN_{tahun}-Penetapan_Jabker_dan_Konversi_Jabker_Eksisting"
                    )
                # For bare PERMEN, also try specific ministry prefixes
                if jenis_norm == "PERMEN":
                    for specific in ("PERMENPUPR", "PERMENPPN", "PERMENDAG", "PERBANBPS"):
                        candidates.append(f"{specific}-{scope}-{nomor}-{tahun}")
                # Scopeless form (some VDB entries)
                if jenis_norm in ("PERMENKES", "PERMENKEU"):
                    candidates.append(f"{jenis_norm}-{nomor}-{tahun}")
            docs.append({
                "raw_ref": f"{jenis_raw} {num_part}",
                "jenis": jenis_norm,
                "nomor": nomor,
                "tahun": tahun,
                "candidates": candidates,
            })
    return docs


def get_correct_doc_id(doc: dict) -> str:
    """Generate the canonical ``doc_id`` from parsed document info."""
    jenis = doc["jenis"]
    nomor = doc["nomor"]
    tahun = doc["tahun"]
    if not nomor:
        return ""
    scope = "PROVINSI" if jenis == "PERGUB" else "NASIONAL"
    return f"{jenis}-{scope}-{nomor}-{tahun}"


def get_unique_doc_ids(results: list[dict], max_docs: int) -> list[str]:
    """Extract unique doc_ids from VDB results, preserving relevance order."""
    unique: list[str] = []
    seen: set[str] = set()
    for r in results:
        did = r.get("doc_id", "")
        if did and did not in seen:
            seen.add(did)
            unique.append(did)
            if len(unique) >= max_docs:
                break
    return unique


# ── Question-text extraction (for v3 benchmark) ─────────────────────────────

def extract_doc_ids_from_question(question: str) -> set[str]:
    """Parse explicit regulation references from question text.

    E.g. "menurut PP 34/2021" → {"PP-NASIONAL-34-2021"}
    Returns all plausible doc_id forms (canonical + aliases).
    """
    doc_ids: set[str] = set()
    patterns = [
        (r'(?:Permen\s+PUPR|PERMEN\s+PUPR)\s+(\d+)/(\d{4})', "PERMENPUPR"),
        (r'(?:Permen\s+PPN|PERMEN\s+PPN)\s+(\d+)/(\d{4})', "PERMENPPN"),
        (r'(?:Permen\s+Perdagangan|PERMEN\s+PERDAGANGAN)\s+(\d+)/(\d{4})', "PERMENDAG"),
        (r'(?:Peraturan\s+BPS)\s+(\d+)/(\d{4})', "PERBANBPS"),
        (r'(?:Permenkes|PERMENKES)\s+(\d+)/(\d{4})', "PERMENKES"),
        (r'(?:Permendagri|PERMENDAGRI)\s+(\d+)/(\d{4})', "PERMENDAGRI"),
        (r'(?:Pergub|PERGUB)\s+(\d+)/(\d{4})', "PERGUB"),
        (r'(?:Perppu|PERPPU)\s+(\d+)/(\d{4})', "PERPPU"),
        (r'\b(?:UU)\s+(\d+)/(\d{4})', "UU"),
        (r'\b(?:PP)\s+(\d+)/(\d{4})', "PP"),
    ]
    for pat, jenis in patterns:
        for m in re.finditer(pat, question, re.IGNORECASE):
            nomor = m.group(1)
            tahun = m.group(2)
            scope = "PROVINSI" if jenis == "PERGUB" else "NASIONAL"
            primary = f"{jenis}-{scope}-{nomor}-{tahun}"
            doc_ids.add(primary)
            alias_prefix = _ALIAS_PREFIXES.get(jenis)
            if alias_prefix:
                doc_ids.add(f"{alias_prefix}-{scope}-{nomor}-{tahun}")
    return doc_ids


# ── Alias matching (for v3 benchmark) ────────────────────────────────────────

def build_doc_id_aliases(doc_ids: set[str]) -> dict[str, set[str]]:
    """Build a mapping from each doc_id to all equivalent aliases.

    E.g. "PERMEN-NASIONAL-8-2022" ↔ "PERMENPUPR-NASIONAL-8-2022"
    """
    aliases: dict[str, set[str]] = {}
    for did in doc_ids:
        group = {did}
        parts = did.split("-", 1)
        prefix = parts[0]
        suffix = parts[1] if len(parts) > 1 else ""

        if prefix in _ALIAS_PREFIXES and suffix:
            group.add(f"{_ALIAS_PREFIXES[prefix]}-{suffix}")
        elif prefix == "PERMEN" and suffix:
            for specific in ("PERMENPUPR", "PERMENPPN", "PERMENDAG", "PERBANBPS"):
                group.add(f"{specific}-{suffix}")
        elif prefix == "KEPDIRJEN" and suffix:
            group.add(f"SKDIRJENBK-{suffix}")

        if prefix == "SKDIRJENBK" and suffix:
            group.add(f"KEPDIRJEN-{suffix}")
            tahun_match = re.search(r'-(\d{4})$', suffix)
            if tahun_match:
                tahun = tahun_match.group(1)
                group.add(
                    f"SK_DIRJEN_BK_TAHUN_{tahun}-Penetapan_Jabker_dan_Konversi_Jabker_Eksisting"
                )

        for d in group:
            if d in aliases:
                aliases[d] = aliases[d] | group
            else:
                aliases[d] = group.copy()

    # Ensure symmetry
    for did, group in list(aliases.items()):
        for d in group:
            if d not in aliases:
                aliases[d] = group.copy()
            else:
                aliases[d] = aliases[d] | group
    return aliases


def match_with_aliases(retrieved: set[str], gt: set[str],
                       aliases: dict[str, set[str]]) -> set[str]:
    """Find GT doc_ids matched by retrieved docs, considering aliases."""
    matched_gt: set[str] = set()
    for gt_id in gt:
        if gt_id in retrieved:
            matched_gt.add(gt_id)
            continue
        gt_aliases = aliases.get(gt_id, {gt_id})
        if retrieved & gt_aliases:
            matched_gt.add(gt_id)
    return matched_gt
