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
import json
import uuid
import streamlit.components.v1 as components
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


# -------------------- D3 embedded HTML renderer (from lexport) -----------------
RELATIONSHIP_COLORS = {
        "CITES": "#22c55e",
        "AMENDS": "#ef4444",
        "REVOKS": "#f59e0b",
}


def build_d3_html(graph_payload: dict, selected_doc_id: str | None, label_mode: str, charge: int, link_distance: int) -> str:
        """Build the D3 HTML/JS string for embedding via Streamlit components.

        Expects `graph_payload` shaped like: {"nodes": [...], "edges": [...], "meta": {...}}
        Nodes must include `doc_id`, `judul`, `jenis`, `nomor`, `tahun`, `degree`.
        """
        nodes_json = json.dumps(graph_payload.get("nodes", []), ensure_ascii=False)
        edges_json = json.dumps(graph_payload.get("edges", []), ensure_ascii=False)
        edge_colors_json = json.dumps(RELATIONSHIP_COLORS)
        label_mode_json = json.dumps(label_mode)
        selected_doc_id_json = json.dumps(selected_doc_id or "")
        component_suffix = uuid.uuid4().hex
        root_id = f"graph-root-{component_suffix}"
        fit_btn_id = f"fit-btn-{component_suffix}"
        svg_id = f"graph-svg-{component_suffix}"
        root_id_json = json.dumps(root_id)
        fit_btn_id_json = json.dumps(fit_btn_id)
        svg_id_json = json.dumps(svg_id)

        return f"""
        <div id="{root_id}" style="width:100%;height:440px;background:#06111f;border:1px solid rgba(148,163,184,0.18);border-radius:18px;overflow:hidden;position:relative;">
            <div id="graph-toolbar" style="position:absolute;left:16px;top:16px;z-index:5;display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
                <button id="{fit_btn_id}" style="background:#1d4ed8;color:white;border:none;border-radius:999px;padding:8px 14px;font:600 12px sans-serif;cursor:pointer;">Fit Graph</button>
                <span style="color:#94a3b8;font:600 12px sans-serif;">Drag nodes or zoom to navigate.</span>
            </div>
            <svg id="{svg_id}" width="100%" height="100%"></svg>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
        <script>
        (function() {{
            const rootElement = document.getElementById({root_id_json});
            const fitButton = document.getElementById({fit_btn_id_json});
            if (!rootElement || !fitButton || typeof d3 === "undefined") {{
                if (rootElement) {{
                    rootElement.innerHTML = '<div style="padding:24px;color:#cbd5e1;font:600 13px sans-serif;">D3 visualization could not be initialized. Please check browser access to the D3 CDN.</div>';
                }}
                return;
            }}
            const nodes = {nodes_json}.map(node => ({{ ...node }}));
            const links = {edges_json}.map((edge, index) => ({{ ...edge, id: `edge-${{index}}` }}));
            const edgeColors = {edge_colors_json};
            const labelMode = {label_mode_json};
            const selectedDocId = {selected_doc_id_json};
            const width = rootElement.clientWidth;
            const height = rootElement.clientHeight;

            const svg = d3.select("#" + {svg_id_json});
            const root = svg.append("g");
            const zoom = d3.zoom().scaleExtent([0.1, 4]).on("zoom", (event) => {{ root.attr("transform", event.transform); }});
            svg.call(zoom);

            const NODE_RADIUS = 12;
            const jenisDomain = [...new Set(nodes.map(node => node.jenis || "Unknown"))].sort(d3.ascending);
            const color = d3.scaleOrdinal(jenisDomain, d3.schemeTableau10.concat(d3.schemeSet3));

            const simulation = d3.forceSimulation(nodes)
                .force("link", d3.forceLink(links).id(d => d.doc_id).distance({link_distance}).strength(0.2))
                .force("charge", d3.forceManyBody().strength({charge}))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("collision", d3.forceCollide().radius(NODE_RADIUS + 6));

            // Draw links
            const link = root.append("g")
                .attr("stroke-linecap", "round")
                .selectAll("line")
                .data(links)
                .join("line")
                .attr("stroke", d => d.color || edgeColors[d.type] || "#94a3b8")
                .attr("stroke-width", 2)
                .attr("opacity", 0.95);

            // Draw nodes
            const node = root.append("g")
                .selectAll("g")
                .data(nodes)
                .join("g")
                .call(d3.drag()
                    .on("start", (event, d) => {{ if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }})
                    .on("drag", (event, d) => {{ d.fx = event.x; d.fy = event.y; }})
                    .on("end", (event, d) => {{ if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }}));

            node.append("circle")
                .attr("r", NODE_RADIUS)
                .attr("fill", d => color(d.jenis || "Unknown"))
                .attr("stroke", d => d.doc_id === selectedDocId ? "#facc15" : "#ffffff")
                .attr("stroke-width", d => d.doc_id === selectedDocId ? 4 : 2);

            node.append("text")
                .text(d => labelMode === "Title" && d.judul ? d.judul : d.doc_id)
                .attr("fill", "#e5eefb")
                .attr("font-family", "Inter, Segoe UI, sans-serif")
                .attr("font-size", 11)
                .attr("text-anchor", "middle")
                .attr("dy", -(NODE_RADIUS + 10));

            simulation.on("tick", () => {{
                link
                    .attr("x1", d => (d.source.x))
                    .attr("y1", d => (d.source.y))
                    .attr("x2", d => (d.target.x))
                    .attr("y2", d => (d.target.y));

                node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
            }});

            function fitGraph() {{
                const bounds = root.node().getBBox();
                if (!bounds.width || !bounds.height) return;
                const fullWidth = width, fullHeight = height;
                const midX = bounds.x + bounds.width / 2;
                const midY = bounds.y + bounds.height / 2;
                const scale = Math.max(0.1, Math.min(2.5, 0.9 / Math.max(bounds.width / fullWidth, bounds.height / fullHeight)));
                const translate = [fullWidth / 2 - scale * midX, fullHeight / 2 - scale * midY];
                svg.transition().duration(600).call(zoom.transform, d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale));
            }}

            function fitGraphWhenReady(attemptsLeft = 12) {{
                const bounds = root.node().getBBox();
                if (bounds.width && bounds.height) {{
                    fitGraph();
                    return;
                }}
                if (attemptsLeft > 0) {{
                    setTimeout(() => fitGraphWhenReady(attemptsLeft - 1), 150);
                }}
            }}

            fitButton.addEventListener("click", fitGraph);
            simulation.on("end", fitGraph);
            setTimeout(fitGraphWhenReady, 250);
            setTimeout(fitGraph, 1200);
        }})();
        </script>
        """


def render_d3_network(graph_payload: dict, selected_doc_id: str | None = None, label_mode: str = "Doc ID", charge: int = -320, link_distance: int = 90, height: int = 460):
        """Render the D3 graph in Streamlit using components.html."""
        html = build_d3_html(graph_payload, selected_doc_id, label_mode, charge, link_distance)
        components.html(html, height=height, scrolling=False)


def merge_graph_payload(existing: dict, new: dict) -> dict:
        """Merge two graph payloads (nodes/edges) without duplicates.

        Keys: nodes identified by `doc_id`; edges identified by (source,target,type,raw).
        Returns a new merged payload.
        """
        existing_nodes = {n["doc_id"]: n for n in existing.get("nodes", [])}
        for n in new.get("nodes", []):
                existing_nodes.setdefault(n["doc_id"], n)

        seen_edges = set()
        merged_edges = []
        for e in existing.get("edges", []) + new.get("edges", []):
                key = (e.get("source"), e.get("target"), e.get("type"), json.dumps(e.get("raw", ""), ensure_ascii=False))
                if key in seen_edges:
                        continue
                seen_edges.add(key)
                merged_edges.append(e)

        merged = {
                "nodes": list(existing_nodes.values()),
                "edges": merged_edges,
                "meta": new.get("meta") or existing.get("meta") or {},
        }
        return merged
