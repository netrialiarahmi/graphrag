"""Knowledge-graph section: parse DASAR_HUKUM, detect conflicts, build agraph."""

import os, sys, re, json

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from streamlit_agraph import Node, Edge
from shared import neo4j_client, llm_stance


# ── Visual hierarchy: 11-level system matching Indonesia's regulation tiers ──
# Each entry: (visual_level, display_name, prefixes[], color)
VISUAL_LEVELS: list[tuple[int, str, list[str], str]] = [
    (1,  "Undang-Undang Dasar",     ["UUD"],                                          "#0f172a"),
    (2,  "Ketetapan MPR",           ["TAP_MPR", "TAPMPR"],                            "#1e3a5f"),
    (3,  "UU / Perppu",             ["UU", "PERPPU"],                                 "#1d4ed8"),
    (4,  "Peraturan Pemerintah",    ["PP"],                                           "#2563eb"),
    (5,  "Peraturan Presiden",      ["PERPRES"],                                      "#0891b2"),
    (6,  "Keputusan Presiden",      ["KEPPRES", "KEP"],                               "#0d9488"),
    (7,  "Instruksi Presiden",      ["INPRES", "INSTRUKSI"],                          "#059669"),
    (8,  "Peraturan Menteri",       [
        "PERMEN", "PERMENDAG", "PERMENKES", "PERMENPUPR", "PERMENPPN",
        "PERMENDAGRI", "PERMENKEU", "PERMENLU", "PERMENLHK", "PERMENAKER",
        "PERMENHUB", "PERMENRISTEKDIKTI", "PERBAN", "PERBANBPS", "PERATURAN",
        "KEPMEN", "KEPDIRJEN", "SE",
    ],                                                                                "#7c3aed"),
    (9,  "Peraturan Daerah",        ["PERDA_PROV", "PERDA_KAB"],                      "#c026d3"),
    (10, "Peraturan Kepala Daerah", ["PERGUB", "PERBUP", "PERWAL", "PERKADA"],        "#e11d48"),
    (11, "Regulasi Lainnya",        ["SK", "SK_DIRJEN", "SKDIRJENBK"],                "#6b7280"),
]

# Build fast lookup: prefix → (visual_level, color)
_PREFIX_MAP: dict[str, tuple[int, str]] = {}
for _vl, _name, _prefixes, _color in VISUAL_LEVELS:
    for _pfx in _prefixes:
        _PREFIX_MAP[_pfx] = (_vl, _color)

# Level metadata
_LEVEL_NAMES: dict[int, str] = {vl: name for vl, name, _, _ in VISUAL_LEVELS}
_LEVEL_COLORS: dict[int, str] = {vl: color for vl, _, _, color in VISUAL_LEVELS}


def _get_visual_level(doc_id: str) -> tuple[int, str]:
    """Return (visual_level, color) for a doc_id."""
    doc_upper = doc_id.upper()
    # Special cases
    if doc_upper.startswith("SK_DIRJEN") or doc_upper.startswith("SKDIRJEN"):
        return _PREFIX_MAP.get("SK_DIRJEN", (11, "#6b7280"))
    prefix = re.split(r"[-]", doc_upper)[0] if doc_upper else ""
    if prefix in _PREFIX_MAP:
        return _PREFIX_MAP[prefix]
    # Fallback: try startswith
    for pfx, val in _PREFIX_MAP.items():
        if doc_upper.startswith(pfx):
            return val
    return (11, "#6b7280")


def _extract_year(doc_id: str) -> str:
    """Extract year from doc_id (typically last 4-digit segment)."""
    parts = doc_id.split("-")
    for p in reversed(parts):
        if re.match(r"^\d{4}$", p):
            return p
    return ""


_JENIS_DISPLAY = {
    "UU": "UU", "PP": "PP", "PERPPU": "Perppu", "PERPRES": "Perpres",
    "PERMEN": "Permen", "PERMENDAG": "Permen Dag", "PERMENKES": "Permen Kes",
    "PERMENPUPR": "Permen PUPR", "PERMENPPN": "Permen PPN",
    "PERMENDAGRI": "Permen Dagri", "PERMENKEU": "Permen Keu",
    "PERGUB": "Pergub", "PERDA_KAB": "Perda Kab", "PERDA_PROV": "Perda Prov",
    "KEPDIRJEN": "Kepdirjen", "KEPMEN": "Kepmen", "KEPPRES": "Keppres",
    "PERWAL": "Perwal", "PERBUP": "Perbup", "SE": "SE", "INPRES": "Inpres",
}


def _short_label(doc_id: str) -> str:
    """Abbreviated display label including year, e.g. 'UU 5/2014'."""
    parts = doc_id.split("-")
    if len(parts) < 3:
        return doc_id
    jenis_raw = parts[0]
    jenis_display = _JENIS_DISPLAY.get(jenis_raw.upper(), jenis_raw.replace("_", " ").title())
    remaining = [p for p in parts[1:] if p.upper() not in ("NASIONAL", "PROVINSI", "KABUPATEN", "KOTA")]
    year = _extract_year(doc_id)
    nomor_parts = [p for p in remaining if p != year]
    if nomor_parts and year:
        return f"{jenis_display} {nomor_parts[0]}/{year}"
    elif nomor_parts:
        return f"{jenis_display} {nomor_parts[0]}"
    elif year:
        return f"{jenis_display} /{year}"
    return jenis_display


def get_level_legend(doc_ids: list[str]) -> list[dict]:
    """Return legend entries only for hierarchy levels present in doc_ids.

    Returns list of {level, name, color} sorted by level.
    """
    present: set[int] = set()
    for did in doc_ids:
        vl, _ = _get_visual_level(did)
        present.add(vl)
    legend = []
    for vl in sorted(present):
        legend.append({
            "level": vl,
            "name": _LEVEL_NAMES.get(vl, "Lainnya"),
            "color": _LEVEL_COLORS.get(vl, "#6b7280"),
        })
    return legend


def get_node_color(doc_id: str) -> str:
    """Return the hierarchy color for a given doc_id."""
    _, color = _get_visual_level(doc_id)
    return color


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

    # Nodes — color-coded by hierarchy level
    for doc_id in doc_ids:
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)
        vl, color = _get_visual_level(doc_id)
        label = _short_label(doc_id)
        year = _extract_year(doc_id)
        level_name = _LEVEL_NAMES.get(vl, "Lainnya")
        tooltip = f"{doc_id}\n{level_name}" + (f" ({year})" if year else "")
        nodes.append(Node(
            id=doc_id,
            label=label,
            size=35,
            color=color,
            title=tooltip,
            shape="box",
            level=vl,
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
            font={"color": "#dc2626", "size": 10, "bold": True},
        ))

    return nodes, edges
