"""Knowledge-graph section: parse DASAR_HUKUM, detect conflicts, build agraph."""

import os, sys, re, json

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from streamlit_agraph import Node, Edge
from shared import neo4j_client, llm_stance


# ── Hierarchy helpers (reused from graph_viz) ─────────────────────────────────
_HIERARCHY_LEVELS = {
    "UUD": 0, "TAP_MPR": 1, "TAPMPR": 1, "UU": 2, "PERPPU": 3, "PP": 4,
    "PERPRES": 5, "PERMEN": 6, "PERMENDAG": 6, "PERMENKES": 6,
    "PERMENPUPR": 6, "PERMENPPN": 6, "PERMENDAGRI": 6, "PERMENKEU": 6,
    "PERMENLU": 6, "PERMENLHK": 6, "PERMENAKER": 6, "PERMENHUB": 6,
    "PERMENRISTEKDIKTI": 6, "PERBAN": 7, "PERBANBPS": 7, "PERATURAN": 7,
    "PERDA_PROV": 8, "PERGUB": 8, "PERDA_KAB": 9, "PERWAL": 9, "PERBUP": 9,
    "PERKADA": 10, "KEP": 11, "KEPDIRJEN": 11, "KEPMEN": 11, "KEPPRES": 11,
    "SK": 12, "SK_DIRJEN": 12, "SKDIRJENBK": 12, "INPRES": 13,
    "INSTRUKSI": 13, "SE": 14,
}


def _get_hierarchy_level(doc_id: str) -> int:
    doc_upper = doc_id.upper()
    if doc_upper.startswith("SK_DIRJEN") or doc_upper.startswith("SKDIRJEN"):
        return 12
    prefix = re.split(r"[-]", doc_upper)[0] if doc_upper else ""
    if prefix in _HIERARCHY_LEVELS:
        return _HIERARCHY_LEVELS[prefix]
    for known, level in _HIERARCHY_LEVELS.items():
        if doc_upper.startswith(known):
            return level
    return 6


def _short_label(doc_id: str) -> str:
    """Abbreviated display label from a doc_id."""
    parts = doc_id.split("-")
    if len(parts) < 3:
        return doc_id
    jenis_raw = parts[0]
    _display = {
        "UU": "UU", "PP": "PP", "PERPPU": "Perppu", "PERPRES": "Perpres",
        "PERMEN": "Permen", "PERMENDAG": "Permen Dag", "PERMENKES": "Permen Kes",
        "PERMENPUPR": "Permen PUPR", "PERMENPPN": "Permen PPN",
        "PERMENDAGRI": "Permen Dagri", "PERMENKEU": "Permen Keu",
        "PERGUB": "Pergub", "PERDA_KAB": "Perda Kab", "PERDA_PROV": "Perda Prov",
        "KEPDIRJEN": "Kepdirjen", "KEPMEN": "Kepmen", "KEPPRES": "Keppres",
        "PERWAL": "Perwal", "PERBUP": "Perbup", "SE": "SE", "INPRES": "Inpres",
    }
    jenis_display = _display.get(jenis_raw.upper(), jenis_raw.replace("_", " ").title())
    remaining = [p for p in parts[1:] if p.upper() not in ("NASIONAL", "PROVINSI", "KABUPATEN", "KOTA")]
    if len(remaining) >= 2:
        return f"{jenis_display} {remaining[0]}/{remaining[1]}"
    elif remaining:
        return f"{jenis_display} {remaining[0]}"
    return jenis_display


# ── Core functions ────────────────────────────────────────────────────────────

def parse_dasar_hukum(answer_text: str) -> list[str]:
    """Extract doc_ids from the DASAR_HUKUM footer in the LLM answer."""
    if not answer_text:
        return []
    m = re.search(r"DASAR_HUKUM:\s*\[([^\]]*)\]", answer_text, re.IGNORECASE)
    if not m:
        return []
    raw = m.group(1)
    ids = [tok.strip().strip("'\"") for tok in raw.split(",") if tok.strip()]
    return ids


def detect_conflicts(answer_text: str, doc_ids: list[str]) -> list[dict]:
    """Ask LLM to detect conflicting regulations from the answer text.

    Returns list of dicts: [{source, target, label}]
    """
    if not answer_text or len(doc_ids) < 2:
        return []

    ids_str = ", ".join(doc_ids)
    prompt = (
        "Anda adalah ahli hukum tata negara Indonesia.\n"
        "Berikut adalah analisis hukum yang menyebutkan dokumen-dokumen regulasi ini:\n"
        f"Dokumen: {ids_str}\n\n"
        "Analisis:\n"
        f"{answer_text[:3000]}\n\n"
        "Tugas: Identifikasi pasangan dokumen yang BERTENTANGAN (konflik normatif) "
        "berdasarkan analisis di atas. Hanya laporan konflik nyata.\n"
        "Jika tidak ada konflik, kembalikan array kosong.\n\n"
        "Kembalikan HANYA JSON array (tanpa markdown):\n"
        '[{"source": "DOC_ID_1", "target": "DOC_ID_2", "label": "alasan singkat"}]\n'
    )

    client = llm_stance.get_llm_client()
    try:
        resp = client.chat.completions.create(
            model=os.getenv("LLM_ROUTER_MODEL", llm_stance.LLM_MODEL),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.1,
        )
        content = (resp.choices[0].message.content or "").strip()
        content = content.replace("```json", "").replace("```", "").strip()
        results = json.loads(content)
        if not isinstance(results, list):
            return []
        # Validate entries
        valid = []
        for r in results:
            src = r.get("source", "")
            tgt = r.get("target", "")
            lbl = r.get("label", "")
            if src in doc_ids and tgt in doc_ids and src != tgt:
                valid.append({"source": src, "target": tgt, "label": str(lbl)[:80]})
        return valid
    except Exception:
        return []


def build_answer_graph(
    doc_ids: list[str],
    neo4j_edges: list[dict],
    conflicts: list[dict],
) -> tuple[list[Node], list[Edge]]:
    """Build agraph Nodes and Edges for the knowledge-map section.

    Args:
        doc_ids: document IDs from DASAR_HUKUM
        neo4j_edges: edges from neo4j_client.get_edges_between()
        conflicts: conflict pairs from detect_conflicts()
    Returns:
        (nodes, edges) for streamlit-agraph
    """
    nodes: list[Node] = []
    edges: list[Edge] = []
    seen_ids: set[str] = set()

    # Nodes
    for doc_id in doc_ids:
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)
        level = _get_hierarchy_level(doc_id)
        label = _short_label(doc_id)
        nodes.append(Node(
            id=doc_id,
            label=label,
            size=35,
            color="#1e3a5f",
            title=doc_id,
            shape="box",
            level=level,
            font={"color": "#ffffff", "size": 13, "face": "Inter, sans-serif", "bold": True},
            borderWidth=0,
            borderWidthSelected=3,
            shapeProperties={"borderRadius": 6},
        ))

    # Neo4j factual edges (CITES / HIGHER)
    for e in neo4j_edges:
        src = e.get("source_id", "")
        tgt = e.get("target_id", "")
        rel = e.get("type", "")
        if not src or not tgt:
            continue
        if src not in seen_ids or tgt not in seen_ids:
            continue
        color = "#2563eb" if rel == "CITES" else "#94a3b8"
        dashes = rel == "HIGHER"
        width = 1.5
        edges.append(Edge(
            source=src, target=tgt, label="",
            color=color, width=width, dashes=dashes,
        ))

    # Conflict edges (LLM-detected)
    for c in conflicts:
        src = c.get("source", "")
        tgt = c.get("target", "")
        lbl = c.get("label", "KONFLIK")
        if src not in seen_ids or tgt not in seen_ids:
            continue
        edges.append(Edge(
            source=src, target=tgt,
            label=lbl,
            color="#dc2626",
            width=4,
            dashes=False,
            font={"color": "#dc2626", "size": 11, "face": "Inter, sans-serif", "bold": True},
        ))

    return nodes, edges
