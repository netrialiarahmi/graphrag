"""Graph visualization utilities using streamlit-agraph.

Hierarchical layout based on Indonesian legal regulation hierarchy:
  Level 0: UUD 1945
  Level 1: Ketetapan MPR
  Level 2: Undang-Undang (UU)
  Level 3: Peraturan Pemerintah Pengganti UU (PERPPU)
  Level 4: Peraturan Pemerintah (PP)
  Level 5: Peraturan Presiden (PERPRES)
  Level 6: Peraturan Menteri (PERMEN, PERMENDAG, PERMENKES, PERMENPUPR, etc.)
  Level 7: Peraturan Lembaga / Badan
  Level 8: Peraturan Daerah Provinsi (PERDA_PROV) / Pergub
  Level 9: Peraturan Daerah Kabupaten/Kota (PERDA_KAB)
  Level 10: Peraturan Kepala Daerah
  Level 11: Keputusan / Keputusan Dirjen (KEPDIRJEN)
  Level 12: Surat Keputusan (SK)
  Level 13: Instruksi
  Level 14: Surat Edaran (SE)

Horizontal positioning: newer regulations (higher year) appear further right.
"""

import re
from streamlit_agraph import agraph, Node, Edge, Config

# -- Navy Color Palette --------------------------------------------------------
NAVY_PRIMARY = "#1e3a5f"       # Dark navy
NAVY_ACCENT = "#2563eb"        # Blue accent
NAVY_LIGHT = "#3b82f6"         # Light blue
NAVY_SKY = "#60a5fa"           # Sky blue
SLATE_BORDER = "#e2e8f0"       # Slate border
WHITE = "#ffffff"

NODE_COLORS = {
    "Document": NAVY_PRIMARY,
    "Pasal": "#d97706",
    "Ayat": "#059669",
    "Diktum": "#dc2626",
    "default": "#64748b",
}

NODE_SIZES = {
    "Document": 35,
    "Pasal": 22,
    "Ayat": 18,
    "Diktum": 22,
    "default": 20,
}

EDGE_COLORS = {
    "CITES": NAVY_ACCENT,
    "HIGHER": "#94a3b8",
    "HAS_PASAL": SLATE_BORDER,
    "HAS_AYAT": SLATE_BORDER,
    "HAS_DIKTUM": SLATE_BORDER,
    "default": SLATE_BORDER,
}

# Stance colors (semantic -- not changed to navy)
STANCE_COLORS = {
    "MENDUKUNG": "#059669",
    "MENENTANG": "#dc2626",
    "NETRAL": "#64748b",
}

STANCE_LABELS = {
    "MENDUKUNG": "SUPPORTS",
    "MENENTANG": "CONTRADICTS",
    "NETRAL": "NEUTRAL",
}

STANCE_BADGE_CLASS = {
    "MENDUKUNG": "stance-supports",
    "MENENTANG": "stance-contradicts",
    "NETRAL": "stance-neutral",
}


# -- Hierarchy Level Map -------------------------------------------------------
_HIERARCHY_LEVELS = {
    "UUD":          0,
    "TAP_MPR":      1,
    "TAPMPR":       1,
    "UU":           2,
    "PERPPU":       3,
    "PP":           4,
    "PERPRES":      5,
    "PERMEN":       6,
    "PERMENDAG":    6,
    "PERMENKES":    6,
    "PERMENPUPR":   6,
    "PERMENPPN":    6,
    "PERMENDAGRI":  6,
    "PERMENKEU":    6,
    "PERMENLU":     6,
    "PERMENLHK":    6,
    "PERMENAKER":   6,
    "PERMENHUB":    6,
    "PERMENRISTEKDIKTI": 6,
    "PERBAN":       7,
    "PERBANBPS":    7,
    "PERATURAN":    7,
    "PERDA_PROV":   8,
    "PERGUB":       8,
    "PERDA_KAB":    9,
    "PERWAL":       9,
    "PERBUP":       9,
    "PERKADA":      10,
    "KEP":          11,
    "KEPDIRJEN":    11,
    "KEPMEN":       11,
    "KEPPRES":      11,
    "SK":           12,
    "SK_DIRJEN":    12,
    "SKDIRJENBK":   12,
    "INPRES":       13,
    "INSTRUKSI":    13,
    "SE":           14,
}

HIERARCHY_LEVEL_NAMES = {
    0: "UUD 1945",
    1: "Ketetapan MPR",
    2: "Undang-Undang",
    3: "Perppu",
    4: "Peraturan Pemerintah",
    5: "Peraturan Presiden",
    6: "Peraturan Menteri",
    7: "Peraturan Lembaga/Badan",
    8: "Perda Provinsi / Pergub",
    9: "Perda Kab/Kota",
    10: "Peraturan Kepala Daerah",
    11: "Keputusan",
    12: "Surat Keputusan",
    13: "Instruksi",
    14: "Surat Edaran",
}


def _get_hierarchy_level(doc_id: str, jenis: str = "") -> int:
    """Determine hierarchy level from doc_id prefix or jenis property."""
    if jenis:
        jenis_upper = jenis.strip().upper().replace(" ", "_")
        if jenis_upper in _HIERARCHY_LEVELS:
            return _HIERARCHY_LEVELS[jenis_upper]

    doc_upper = doc_id.upper()

    # Handle special SK format
    if doc_upper.startswith("SK_DIRJEN") or doc_upper.startswith("SKDIRJEN"):
        return 12

    # Extract prefix before first hyphen
    prefix = re.split(r'[-]', doc_upper)[0] if doc_upper else ""

    if prefix in _HIERARCHY_LEVELS:
        return _HIERARCHY_LEVELS[prefix]

    # Try longer prefixes (e.g. PERDA_KAB, PERDA_PROV)
    for known_prefix, level in _HIERARCHY_LEVELS.items():
        if doc_upper.startswith(known_prefix):
            return level

    return 6  # Default: Permen level


def _get_year_from_doc_id(doc_id: str, tahun=None) -> int:
    """Extract year from tahun property or doc_id for horizontal ordering."""
    if tahun:
        try:
            return int(tahun)
        except (ValueError, TypeError):
            pass
    match = re.search(r'(19|20)\d{2}', doc_id)
    if match:
        return int(match.group())
    return 2020


def _get_short_label(doc_id: str) -> str:
    """Create a clean, short display label from a doc_id.

    Examples:
        UU-NASIONAL-11-2020       -> UU 11/2020
        PP-NASIONAL-16-2021       -> PP 16/2021
        PERMENPUPR-NASIONAL-8-2022 -> Permen PUPR 8/2022
        PERGUB-PROVINSI-20-2024   -> Pergub 20/2024
        PERDA_KAB-KABUPATEN-5-2015 -> Perda Kab 5/2015
        KEPDIRJEN-NASIONAL-12.1-2022 -> Kepdirjen 12.1/2022
        SK_DIRJEN_BK_TAHUN_2022-... -> SK Dirjen BK 2022
    """
    doc_upper = doc_id.upper()

    # Special: SK_DIRJEN_BK...
    if doc_upper.startswith("SK_DIRJEN_BK"):
        year_match = re.search(r'(\d{4})', doc_id)
        return f"SK Dirjen BK {year_match.group(1) if year_match else ''}".strip()

    # General pattern: JENIS-LEVEL-NOMOR-TAHUN
    parts = doc_id.split("-")
    if len(parts) < 3:
        return doc_id

    jenis_raw = parts[0]

    _jenis_display = {
        "UU": "UU",
        "PP": "PP",
        "PERPPU": "Perppu",
        "PERPRES": "Perpres",
        "PERMEN": "Permen",
        "PERMENDAG": "Permen Dag",
        "PERMENKES": "Permen Kes",
        "PERMENPUPR": "Permen PUPR",
        "PERMENPPN": "Permen PPN",
        "PERMENDAGRI": "Permen Dagri",
        "PERMENKEU": "Permen Keu",
        "PERGUB": "Pergub",
        "PERDA_KAB": "Perda Kab",
        "PERDA_PROV": "Perda Prov",
        "KEPDIRJEN": "Kepdirjen",
        "KEPMEN": "Kepmen",
        "KEPPRES": "Keppres",
        "PERWAL": "Perwal",
        "PERBUP": "Perbup",
        "SE": "SE",
        "INPRES": "Inpres",
    }

    jenis_display = _jenis_display.get(
        jenis_raw.upper(),
        jenis_raw.replace("_", " ").title(),
    )

    # Skip level segments and get nomor+tahun
    remaining = parts[1:]
    _level_words = {"NASIONAL", "PROVINSI", "KABUPATEN", "KOTA"}
    remaining = [p for p in remaining if p.upper() not in _level_words]

    if len(remaining) >= 2:
        return f"{jenis_display} {remaining[0]}/{remaining[1]}"
    elif len(remaining) == 1:
        return f"{jenis_display} {remaining[0]}"
    else:
        return jenis_display


# -- Public API ----------------------------------------------------------------

def stance_badge_html(stance: str) -> str:
    """Return an HTML badge span for a stance label."""
    label = STANCE_LABELS.get(stance, stance)
    css_class = STANCE_BADGE_CLASS.get(stance, "stance-neutral")
    return f'<span class="{css_class}">{label}</span>'


def get_node_label(node: dict) -> str:
    """Generate a display label for a node."""
    labels = node.get("_labels", node.get("labels", []))
    if "Document" in labels:
        return _get_short_label(node.get("doc_id", "?"))
    elif "Pasal" in labels:
        return node.get("name", "Pasal ?")
    elif "Ayat" in labels:
        return node.get("name", "Ayat ?")
    elif "Diktum" in labels:
        return node.get("name", "Diktum ?")
    else:
        return node.get("name", node.get("doc_id", "?"))


def get_node_type(node: dict) -> str:
    """Determine primary node type from labels."""
    labels = node.get("_labels", node.get("labels", []))
    for t in ["Document", "Pasal", "Ayat", "Diktum"]:
        if t in labels:
            return t
    return "default"


def build_agraph_nodes(nodes: list[dict]) -> list[Node]:
    """Convert graph data nodes to streamlit-agraph Node objects (deduplicated)."""
    agraph_nodes = []
    seen_ids: set[str] = set()
    for node in nodes:
        node_type = get_node_type(node)
        node_id = node.get("_elementId", node.get("doc_id", node.get("id", str(id(node)))))
        if node_id in seen_ids:
            continue
        seen_ids.add(node_id)
        label = get_node_label(node)
        agraph_nodes.append(Node(
            id=node_id,
            label=label,
            size=NODE_SIZES.get(node_type, NODE_SIZES["default"]),
            color=NODE_COLORS.get(node_type, NODE_COLORS["default"]),
            title=f"{node_type}: {label}\n{_node_tooltip(node)}",
            shape="dot" if node_type != "Document" else "box",
        ))
    return agraph_nodes


def build_agraph_edges(edges: list[dict], stance_map: dict = None) -> list[Edge]:
    """Convert graph data edges to streamlit-agraph Edge objects."""
    agraph_edges = []
    stance_map = stance_map or {}
    for edge in edges:
        source = edge.get("source", edge.get("source_id", ""))
        target = edge.get("target", edge.get("target_id", ""))
        rel_type = edge.get("type", "")
        if not source or not target:
            continue
        color = EDGE_COLORS.get(rel_type, EDGE_COLORS["default"])
        label = ""
        width = 1
        stance_key = (source, target)
        stance_key_rev = (target, source)
        stance_result = stance_map.get(stance_key) or stance_map.get(stance_key_rev)
        if stance_result and rel_type in ("CITES", "HIGHER"):
            stance = stance_result.get("stance", "NETRAL")
            color = STANCE_COLORS.get(stance, STANCE_COLORS["NETRAL"])
            width = 3
        dashes = rel_type == "HIGHER"
        agraph_edges.append(Edge(
            source=source, target=target, label=label,
            color=color, width=width, dashes=dashes,
        ))
    return agraph_edges


def render_graph(
    nodes: list[dict],
    edges: list[dict],
    stance_map: dict = None,
    height: int = 500,
    physics: bool = True,
) -> str:
    """Render an interactive graph visualization (general, non-hierarchical)."""
    if not nodes:
        return None
    agraph_nodes = build_agraph_nodes(nodes)
    agraph_edges = build_agraph_edges(edges, stance_map)
    config = Config(
        width="100%",
        height=height,
        directed=True,
        physics=physics,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor=NAVY_ACCENT,
        collapsible=False,
        node={"labelProperty": "label", "renderLabel": True},
        link={"labelProperty": "label", "renderLabel": False},
    )
    selected = agraph(nodes=agraph_nodes, edges=agraph_edges, config=config)
    return selected


def render_document_graph(
    doc_nodes: list[dict],
    doc_edges: list[dict],
    stance_map: dict = None,
    height: int = 600,
) -> str:
    """Render a hierarchical document graph.

    Vertical axis: legal hierarchy (UU at top, SK/SE at bottom).
    Horizontal axis: newer regulations further right.
    Edge styles: solid = CITES, dashed = HIGHER (hierarchy).
    No edge text labels -- arrow direction conveys the relationship.
    """
    agraph_nodes = []
    seen_ids: set[str] = set()

    for doc in doc_nodes:
        doc_id = doc.get("doc_id", "?")
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)

        jenis = doc.get("jenis", "")
        tahun = doc.get("tahun", None)
        judul = doc.get("judul", "")
        level = _get_hierarchy_level(doc_id, jenis)
        label = _get_short_label(doc_id)

        # Tooltip
        tooltip_parts = [f"ID: {doc_id}"]
        if judul:
            tooltip_parts.append(f"Judul: {judul}")
        level_name = HIERARCHY_LEVEL_NAMES.get(level, f"Level {level}")
        tooltip_parts.append(f"Hierarki: {level_name}")
        if tahun:
            tooltip_parts.append(f"Tahun: {tahun}")

        agraph_nodes.append(Node(
            id=doc_id,
            label=label,
            size=35,
            color=NAVY_PRIMARY,
            title="\n".join(tooltip_parts),
            shape="box",
            level=level,
            font={"color": WHITE, "size": 13, "face": "Inter, sans-serif", "bold": True},
            borderWidth=0,
            borderWidthSelected=3,
            shapeProperties={"borderRadius": 6},
        ))

    agraph_edges = []
    stance_map = stance_map or {}

    for edge in doc_edges:
        source = edge.get("source_id", "")
        target = edge.get("target_id", "")
        rel_type = edge.get("type", "")
        if not source or not target:
            continue

        color = EDGE_COLORS.get(rel_type, EDGE_COLORS["default"])
        width = 2

        stance_key = (source, target)
        stance_key_rev = (target, source)
        stance_result = stance_map.get(stance_key) or stance_map.get(stance_key_rev)
        if stance_result:
            stance = stance_result.get("stance", "NETRAL")
            color = STANCE_COLORS.get(stance, STANCE_COLORS["NETRAL"])
            width = 4

        agraph_edges.append(Edge(
            source=source,
            target=target,
            label="",
            color=color,
            width=width,
            dashes=(rel_type == "HIGHER"),
        ))

    config = Config(
        width="100%",
        height=height,
        directed=True,
        physics=False,
        hierarchical=True,
        direction="UD",
        sortMethod="directed",
        levelSeparation=150,
        nodeSpacing=100,
        treeSpacing=200,
        nodeHighlightBehavior=True,
        highlightColor=NAVY_ACCENT,
    )

    selected = agraph(
        nodes=agraph_nodes,
        edges=agraph_edges,
        config=config,
    )

    return selected


def _node_tooltip(node: dict) -> str:
    """Build a tooltip string from node properties."""
    skip_keys = {"_labels", "_elementId", "labels", "elementId", "embedding"}
    parts = []
    for k, v in node.items():
        if k in skip_keys:
            continue
        val_str = str(v)
        if len(val_str) > 100:
            val_str = val_str[:100] + "..."
        parts.append(f"{k}: {val_str}")
    return "\n".join(parts)
