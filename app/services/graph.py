"""Service layer for building D3.js visualization payloads.

Extracts document graph structure from context_docs and relationship_context,
and returns JSON-serializable nodes/edges for client-side D3 rendering.
"""
from typing import Dict, List, Any, Set
from utils.graph_viz import (
    _get_hierarchy_level, _get_year_from_doc_id, _get_short_label,
    NODE_COLORS, NODE_SIZES, EDGE_COLORS, STANCE_COLORS, HIERARCHY_LEVEL_NAMES
)


def build_d3_payload(
    context_docs: Dict[str, Any],
    relationship_context: str = "",
    doc_metadata: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Build a D3.js-compatible graph payload from context documents.
    
    Args:
        context_docs: Dict of {doc_id: {chunks: [...], source: "..."}}
                     from the agent pipeline.
        relationship_context: String describing relationships between documents.
                             Usually formatted as "- DOC_A --[REL_TYPE]--> DOC_B".
        doc_metadata: Optional dict of {doc_id: {jenis, judul, tahun, ...}}
                     If provided, used for node labels and tooltips.
    
    Returns:
        Dict with keys:
        - nodes: List of node objects {id, label, size, color, title, shape, ...}
        - edges: List of edge objects {source, target, label, color, type, ...}
        - meta: Metadata {node_count, edge_count, primary_doc_ids, ...}
    """
    if not context_docs:
        return {"nodes": [], "edges": [], "meta": {}}
    
    # Initialize doc_metadata if not provided
    if doc_metadata is None:
        doc_metadata = {}
    
    nodes = []
    seen_node_ids: Set[str] = set()
    
    # Build nodes from context_docs
    for doc_id, doc_info in context_docs.items():
        if doc_id in seen_node_ids:
            continue
        seen_node_ids.add(doc_id)
        
        # Get metadata for this doc
        meta = doc_metadata.get(doc_id, {})
        jenis = meta.get("jenis", "")
        judul = meta.get("judul", "")
        tahun = meta.get("tahun", None)
        
        # Determine hierarchy level for vertical positioning (if used)
        level = _get_hierarchy_level(doc_id, jenis)
        
        # Generate label and tooltip
        label = _get_short_label(doc_id)
        tooltip_parts = [f"ID: {doc_id}"]
        if judul:
            tooltip_parts.append(f"Judul: {judul}")
        level_name = HIERARCHY_LEVEL_NAMES.get(level, f"Level {level}")
        tooltip_parts.append(f"Hierarki: {level_name}")
        if tahun:
            tooltip_parts.append(f"Tahun: {tahun}")
        
        # Count chunks to show data availability
        chunk_count = len(doc_info.get("chunks", []))
        if chunk_count > 0:
            tooltip_parts.append(f"Chunks: {chunk_count}")
        
        source = doc_info.get("source", "Unknown")
        tooltip_parts.append(f"Source: {source}")
        
        node = {
            "id": doc_id,
            "label": label,
            "title": "\n".join(tooltip_parts),
            "size": NODE_SIZES.get("Document", 35),
            "color": NODE_COLORS.get("Document", "#1e3a5f"),
            "shape": "box",
            "level": level,  # For hierarchical layout
            "font": {
                "color": "#ffffff",
                "size": 13,
                "face": "Inter, sans-serif",
                "bold": True,
            },
            "borderWidth": 0,
            "borderWidthSelected": 3,
            "shapeProperties": {"borderRadius": 6},
        }
        nodes.append(node)
    
    # Parse edges from relationship_context
    edges = []
    seen_edges: Set[tuple] = set()
    
    if relationship_context:
        for line in relationship_context.splitlines():
            line = line.strip()
            # Expected format: "- SOURCE_ID --[EDGE_TYPE]--> TARGET_ID"
            if not line.startswith("-"):
                continue
            
            line = line.lstrip("- ").strip()
            
            # Parse source, type, target using regex
            import re
            match = re.match(r"(.+?)\s+--\[(.+?)\]-->\s+(.+)", line)
            if not match:
                continue
            
            source_id = match.group(1).strip()
            rel_type = match.group(2).strip()
            target_id = match.group(3).strip()
            
            # Check for duplicates
            edge_key = (source_id, target_id, rel_type)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            
            # Get edge color based on relationship type
            color = EDGE_COLORS.get(rel_type, EDGE_COLORS.get("default", "#94a3b8"))
            
            # Dashed edges for HIGHER (hierarchical) relationships
            dashes = rel_type == "HIGHER"
            
            edge = {
                "source": source_id,
                "target": target_id,
                "label": "",  # No labels on edges for clarity
                "color": color,
                "type": rel_type,
                "width": 4 if rel_type in ("CITES", "AMENDS", "REVOKS") else 2,
                "dashes": dashes,
            }
            edges.append(edge)
    
    # Build metadata
    primary_doc_ids = list(context_docs.keys())
    meta = {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "primary_doc_ids": primary_doc_ids,
        "relationship_types": list(set(e.get("type", "") for e in edges)),
    }
    
    return {
        "nodes": nodes,
        "edges": edges,
        "meta": meta,
    }


def build_d3_html(
    d3_payload: Dict[str, Any],
    selected_doc_id: str = None,
    label_mode: str = "Doc ID",
    charge: int = -320,
    link_distance: int = 90,
) -> str:
    """Build an HTML/JS string for embedding D3 visualization via Streamlit components.
    
    This is a convenience wrapper for server-side rendering of the D3 visualization.
    For most use cases, the frontend should render the d3_payload JSON directly.
    
    Args:
        d3_payload: Output from build_d3_payload().
        selected_doc_id: Optional document ID to highlight.
        label_mode: "Doc ID" or other label mode.
        charge: D3 force simulation charge parameter.
        link_distance: D3 force simulation link distance.
    
    Returns:
        HTML string with embedded D3.js code.
    """
    # Delegate to existing graph_viz function if HTML rendering is needed
    from utils.graph_viz import build_d3_html as _orig_build_d3_html
    
    return _orig_build_d3_html(
        graph_payload=d3_payload,
        selected_doc_id=selected_doc_id,
        label_mode=label_mode,
        charge=charge,
        link_distance=link_distance,
    )
