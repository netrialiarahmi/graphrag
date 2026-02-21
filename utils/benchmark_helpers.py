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
  