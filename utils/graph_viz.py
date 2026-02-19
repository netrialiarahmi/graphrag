"""Graph visualization utilities using streamlit-agraph."""

from streamlit_agraph import agraph, Node, Edge, Config

# ── Color Palette (Professional Light Theme) ─────────────────────────────────
NODE_COLORS = {
    "Document": "#4f46e5",     # Indigo
    "Pasal": "#d97706",        # Amber
    "Ayat": "#059669",         # Emerald
    "Diktum": "#dc2626",       # Red
    "default": "#6b7280",      # Gray
}

NODE_SIZES = {
    "Document": 30,
    "Pasal": 22,
    "Ayat": 18,
    "Diktum": 22,
    "default": 20,
}

EDGE_COLORS = {
    "CITES": "#4f46e5",        # Indigo
    "HIGHER": "#7c3aed",       # Violet
    "HAS_PASAL": "#e2e8f0",
    "HAS_AYAT": "#e2e8f0",
    "HAS_DIKTUM": "#e2e8f0",
    "default": "#e2e8f0",
}

# Stance colors
STANCE_COLORS = {
    "MENDUKUNG": "#059669",    # Emerald
    "MENENTANG": "#dc2626",    # Red
    "NETRAL": "#6b7280",       # Gray
}

# English labels — no emojis
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


def stance_badge_html(stance: str) -> str:
    """Return an HTML badge span for a stance label."""
    label = STANCE_LABELS.get(stance, stance)
    css_class = STANCE_BADGE_CLASS.get(stance, "stance-neutral")
    return f'<span class="{css_class}">{label}</span>'


def get_node_label(node: dict) -> str:
    """Generate a display label for a node."""
    labels = node.get("_labels", node.get("labels", []))

    if "Document" in labels:
        doc_id = node.get("doc_id", "")
        # Shorten for display: e.g. "UU-NASIONAL-31-2002" -> "UU-31-2002"
        return doc_id if len(doc_id) <= 25 else doc_id.replace("NASIONAL-", "")
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
    """
    Convert graph data edges to streamlit-agraph Edge objects.

    Args:
        edges: List of edge dicts with source, target, type keys.
        stance_map: Optional dict mapping (source_id, target_id) -> stance result.
    """
    agraph_edges = []
    stance_map = stance_map or {}

    for edge in edges:
        source = edge.get("source", edge.get("source_id", ""))
        target = edge.get("target", edge.get("target_id", ""))
        rel_type = edge.get("type", "")

        if not source or not target:
            continue

        # Determine color based on stance (for CITES/HIGHER edges)
        color = EDGE_COLORS.get(rel_type, EDGE_COLORS["default"])
        label = rel_type
        width = 1

        # Check for stance classification
        stance_key = (source, target)
        stance_key_rev = (target, source)
        stance_result = stance_map.get(stance_key) or stance_map.get(stance_key_rev)

        if stance_result and rel_type in ("CITES", "HIGHER"):
            stance = stance_result.get("stance", "NETRAL")
            color = STANCE_COLORS.get(stance, STANCE_COLORS["NETRAL"])
            eng_label = STANCE_LABELS.get(stance, stance)
            label = f"{rel_type} ({eng_label})"
            width = 3

        dashes = rel_type == "HIGHER"

        agraph_edges.append(Edge(
            source=source,
            target=target,
            label=label,
            color=color,
            width=width,
            dashes=dashes,
        ))

    return agraph_edges


def render_graph(
    nodes: list[dict],
    edges: list[dict],
    stance_map: dict = None,
    height: int = 500,
    physics: bool = True,
) -> str:
    """
    Render an interactive graph visualization.

    Args:
        nodes: List of node dicts from Neo4j.
        edges: List of edge dicts from Neo4j.
        stance_map: Optional stance classifications for edges.
        height: Graph canvas height in pixels.
        physics: Enable physics simulation.

    Returns:
        Selected node ID (if user clicked a node), or None.
    """
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
        highlightColor="#6366f1",
        collapsible=False,
        node={
            "labelProperty": "label",
            "renderLabel": True,
        },
        link={
            "labelProperty": "label",
            "renderLabel": True,
        },
    )

    selected = agraph(
        nodes=agraph_nodes,
        edges=agraph_edges,
        config=config,
    )

    return selected


def render_document_graph(
    doc_nodes: list[dict],
    doc_edges: list[dict],
    stance_map: dict = None,
    height: int = 450,
) -> str:
    """
    Render a document-level graph (Documents + CITES/HIGHER edges).
    Simplified view without Pasal/Ayat/Diktum nodes.
    """
    agraph_nodes = []
    seen_ids: set[str] = set()
    for doc in doc_nodes:
        doc_id = doc.get("doc_id", "?")
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)
        agraph_nodes.append(Node(
            id=doc_id,
            label=doc_id,
            size=30,
            color=NODE_COLORS["Document"],
            title=f"Document: {doc_id}\n{doc.get('judul', '')}",
            shape="box",
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
        label = rel_type
        width = 2

        stance_key = (source, target)
        stance_key_rev = (target, source)
        stance_result = stance_map.get(stance_key) or stance_map.get(stance_key_rev)

        if stance_result:
            stance = stance_result.get("stance", "NETRAL")
            color = STANCE_COLORS.get(stance, STANCE_COLORS["NETRAL"])
            eng_label = STANCE_LABELS.get(stance, stance)
            label = f"{rel_type} / {eng_label}"
            width = 4

        agraph_edges.append(Edge(
            source=source,
            target=target,
            label=label,
            color=color,
            width=width,
            dashes=(rel_type == "HIGHER"),
        ))

    config = Config(
        width="100%",
        height=height,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#6366f1",
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
