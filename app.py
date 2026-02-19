"""
GraphRAG -- Legal Document Relationship Explorer
Streamlit app for visualizing and analyzing relationships between Indonesian legal documents.
"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from utils import neo4j_client, pinecone_client, llm_stance, graph_viz

# -- Page Config ---------------------------------------------------------------
st.set_page_config(
    page_title="GraphRAG",
    page_icon="G",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -- Modern Light Theme CSS ----------------------------------------------------
st.markdown("""
<style>
    /* ── Reset & hide Streamlit chrome ────────────────────────────────────── */
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent !important; }

    /* ── Typography ──────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #1a1a2e;
    }
    h1 { font-weight: 800; letter-spacing: -0.5px; color: #0f172a; }
    h2 { font-weight: 700; letter-spacing: -0.3px; font-size: 1.35rem !important; color: #0f172a; }
    h3 { font-weight: 600; font-size: 1.1rem !important; color: #1e293b; }

    /* ── Brand Header ────────────────────────────────────────────────────── */
    .brand-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #334155 100%);
        padding: 1.25rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 24px rgba(15, 23, 42, 0.15);
        position: relative;
        overflow: hidden;
    }
    .brand-header::before {
        content: "";
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 70%);
        pointer-events: none;
    }
    .brand-header::after {
        content: "";
        position: absolute;
        bottom: -60%;
        left: 10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(139,92,246,0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .brand-text {
        position: relative;
        z-index: 1;
    }
    .brand-title {
        font-size: 1.65rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }
    .brand-title span {
        background: linear-gradient(135deg, #818cf8, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .brand-subtitle {
        font-size: 0.82rem;
        color: rgba(255,255,255,0.55);
        margin-top: 4px;
        font-weight: 400;
        letter-spacing: 0.2px;
    }
    .conn-badges {
        display: flex;
        gap: 12px;
        position: relative;
        z-index: 1;
    }
    .conn-badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 0.72rem;
        font-weight: 500;
        color: rgba(255,255,255,0.7);
        letter-spacing: 0.3px;
    }
    .conn-indicator {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .conn-indicator.ok {
        background: #34d399;
        box-shadow: 0 0 6px rgba(52,211,153,0.6);
        animation: pulse-green 2s ease-in-out infinite;
    }
    .conn-indicator.err {
        background: #f87171;
        box-shadow: 0 0 6px rgba(248,113,113,0.6);
        animation: pulse-red 2s ease-in-out infinite;
    }
    @keyframes pulse-green {
        0%, 100% { box-shadow: 0 0 4px rgba(52,211,153,0.4); }
        50% { box-shadow: 0 0 10px rgba(52,211,153,0.8); }
    }
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 4px rgba(248,113,113,0.4); }
        50% { box-shadow: 0 0 10px rgba(248,113,113,0.8); }
    }

    /* ── Tab Styling ─────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #f8fafc;
        border-radius: 12px;
        padding: 4px;
        border: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 28px;
        font-weight: 500;
        font-size: 0.88rem;
        color: #64748b;
        border-radius: 8px;
        border-bottom: none !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #334155;
        background: rgba(99,102,241,0.05);
    }
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(99,102,241,0.3);
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    /* ── Buttons ─────────────────────────────────────────────────────────── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        border: none !important;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 0.55rem 1.6rem;
        box-shadow: 0 2px 12px rgba(99,102,241,0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        letter-spacing: 0.2px;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 4px 20px rgba(99,102,241,0.45);
        transform: translateY(-1px);
    }
    .stButton > button[kind="secondary"],
    .stButton > button:not([kind="primary"]) {
        border-radius: 10px;
        font-weight: 500;
        border: 1px solid #e2e8f0 !important;
        transition: all 0.25s ease;
    }
    .stButton > button:not([kind="primary"]):hover {
        border-color: #6366f1 !important;
        color: #4f46e5 !important;
    }

    /* ── Inputs ──────────────────────────────────────────────────────────── */
    .stTextInput > div > div > input {
        border-radius: 10px !important;
        border: 1.5px solid #e2e8f0 !important;
        padding: 0.6rem 1rem !important;
        font-size: 0.9rem !important;
        transition: all 0.25s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
    }
    .stSelectbox > div > div {
        border-radius: 10px !important;
    }
    .stNumberInput > div > div > input {
        border-radius: 10px !important;
    }

    /* ── Section Dividers ────────────────────────────────────────────────── */
    .section-divider {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 2rem 0 1rem;
    }
    .section-divider::before,
    .section-divider::after {
        content: "";
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
    }
    .section-divider-text {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #94a3b8;
        font-weight: 600;
        white-space: nowrap;
    }

    /* ── Result Cards ────────────────────────────────────────────────────── */
    .result-card {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.15rem 1.25rem;
        margin: 0.5rem 0;
        background: #ffffff;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .result-card::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 3px;
        background: linear-gradient(180deg, #6366f1, #a78bfa);
        border-radius: 3px 0 0 3px;
    }
    .result-card:hover {
        border-color: #c7d2fe;
        box-shadow: 0 4px 16px rgba(99,102,241,0.08);
        transform: translateY(-2px);
    }
    .result-card-title {
        font-weight: 600;
        font-size: 0.92rem;
        color: #0f172a;
        margin-bottom: 4px;
    }
    .result-card-meta {
        font-size: 0.75rem;
        color: #94a3b8;
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
    }
    .result-card-meta span {
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    .result-card-content {
        font-size: 0.83rem;
        color: #475569;
        line-height: 1.55;
        margin-top: 10px;
        padding-top: 10px;
        border-top: 1px solid #f1f5f9;
    }
    .result-card.scope-pasal::before { background: linear-gradient(180deg, #d97706, #f59e0b); }
    .result-card.scope-ayat::before { background: linear-gradient(180deg, #059669, #34d399); }
    .result-card.scope-diktum::before { background: linear-gradient(180deg, #dc2626, #f87171); }

    /* ── Stance Badges ───────────────────────────────────────────────────── */
    .stance-supports {
        display: inline-block;
        background: linear-gradient(135deg, #059669, #34d399);
        color: #fff;
        font-size: 0.68rem;
        font-weight: 600;
        padding: 4px 14px;
        border-radius: 20px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        box-shadow: 0 2px 8px rgba(5,150,105,0.25);
    }
    .stance-contradicts {
        display: inline-block;
        background: linear-gradient(135deg, #dc2626, #f87171);
        color: #fff;
        font-size: 0.68rem;
        font-weight: 600;
        padding: 4px 14px;
        border-radius: 20px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        box-shadow: 0 2px 8px rgba(220,38,38,0.25);
    }
    .stance-neutral {
        display: inline-block;
        background: linear-gradient(135deg, #6b7280, #9ca3af);
        color: #fff;
        font-size: 0.68rem;
        font-weight: 600;
        padding: 4px 14px;
        border-radius: 20px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        box-shadow: 0 2px 8px rgba(107,114,128,0.2);
    }

    /* ── Metric Cards ────────────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8fafc, #f1f5f9);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 16px;
        transition: all 0.25s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: #c7d2fe;
        box-shadow: 0 2px 12px rgba(99,102,241,0.06);
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        color: #0f172a;
        font-weight: 700;
    }

    /* ── Expanders ────────────────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        color: #334155 !important;
        background: #f8fafc !important;
        border-radius: 10px !important;
        border: 1px solid #e2e8f0 !important;
        transition: all 0.2s ease;
    }
    .streamlit-expanderHeader:hover {
        border-color: #c7d2fe !important;
        background: #eef2ff !important;
    }

    /* ── Info / Warning / Error ───────────────────────────────────────────── */
    .stAlert > div {
        border-radius: 10px !important;
        font-size: 0.85rem;
        border: none !important;
    }

    /* ── Graph container ─────────────────────────────────────────────────── */
    .graph-container {
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        overflow: hidden;
        background: #fafbfe;
        box-shadow: 0 1px 8px rgba(0,0,0,0.04);
    }

    /* ── Stat pills ──────────────────────────────────────────────────────── */
    .stat-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 0.75rem;
        font-weight: 500;
        color: #475569;
    }
    .stat-pill-count {
        background: linear-gradient(135deg, #4f46e5, #6366f1);
        color: #fff;
        font-weight: 700;
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 10px;
    }

    /* ── Stance Row ──────────────────────────────────────────────────────── */
    .stance-row {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 12px 16px;
        border: 1px solid #f1f5f9;
        border-radius: 10px;
        margin: 6px 0;
        background: #fff;
        transition: all 0.2s ease;
    }
    .stance-row:hover {
        background: #fafbfe;
        border-color: #e2e8f0;
    }
    .stance-arrow {
        color: #94a3b8;
        font-size: 0.85rem;
    }
    .stance-doc {
        font-weight: 600;
        font-size: 0.82rem;
        color: #1e293b;
    }
    .stance-reason {
        font-size: 0.78rem;
        color: #64748b;
        flex: 1;
    }

    /* ── Footer ──────────────────────────────────────────────────────────── */
    .app-footer {
        text-align: center;
        padding: 1.5rem 0 1rem;
        margin-top: 3rem;
        font-size: 0.73rem;
        color: #94a3b8;
        border-top: 1px solid #e2e8f0;
        letter-spacing: 0.2px;
    }

    /* ── Scrollbar ────────────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

    /* ── Checkbox label ──────────────────────────────────────────────────── */
    .stCheckbox label span {
        font-size: 0.85rem !important;
        color: #475569 !important;
    }

    /* ── Spinner ─────────────────────────────────────────────────────────── */
    .stSpinner > div {
        border-top-color: #6366f1 !important;
    }
</style>
""", unsafe_allow_html=True)


# -- Session State Init --------------------------------------------------------
if "stance_cache" not in st.session_state:
    st.session_state.stance_cache = {}
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "search_doc_ids" not in st.session_state:
    st.session_state.search_doc_ids = []
if "selected_node" not in st.session_state:
    st.session_state.selected_node = None

# -- Connection checks (cached per session) ------------------------------------
neo4j_ok = neo4j_client.test_connection()
pinecone_ok = pinecone_client.test_connection()

# -- Brand Header --------------------------------------------------------------
neo4j_cls = "ok" if neo4j_ok else "err"
neo4j_lbl = "Connected" if neo4j_ok else "Offline"
pine_cls = "ok" if pinecone_ok else "err"
pine_lbl = "Connected" if pinecone_ok else "Offline"

st.markdown(f"""
<div class="brand-header">
    <div class="brand-text">
        <div class="brand-title">Graph<span>RAG</span></div>
        <div class="brand-subtitle">Legal Document Relationship Explorer</div>
    </div>
    <div class="conn-badges">
        <div class="conn-badge">
            <span class="conn-indicator {neo4j_cls}"></span>Neo4j {neo4j_lbl}
        </div>
        <div class="conn-badge">
            <span class="conn-indicator {pine_cls}"></span>Pinecone {pine_lbl}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# -- Helpers -------------------------------------------------------------------
def section_divider(text: str):
    """Render a centered section divider with gradient lines."""
    st.markdown(
        f'<div class="section-divider"><span class="section-divider-text">{text}</span></div>',
        unsafe_allow_html=True,
    )


def render_result_card(doc_id: str, scope: str, content: str, score: float = None, article_id: str = ""):
    """Render a single result as a styled card."""
    scope_class = f"scope-{scope}" if scope in ("pasal", "ayat", "diktum") else ""
    score_html = f'<span>Score: {score:.4f}</span>' if score else ""
    article_display = article_id[:28] if article_id else ""
    article_html = f'<span>Art: {article_display}</span>' if article_display else ""
    preview = content[:280].replace("\n", " ") + ("..." if len(content) > 280 else "")
    st.markdown(f"""
    <div class="result-card {scope_class}">
        <div class="result-card-title">{doc_id}</div>
        <div class="result-card-meta">
            <span>{scope}</span>
            {article_html}
            {score_html}
        </div>
        <div class="result-card-content">{preview}</div>
    </div>
    """, unsafe_allow_html=True)


def render_stance_row(src: str, tgt: str, stance_result: dict):
    """Render a stance row with badge."""
    badge = graph_viz.stance_badge_html(stance_result["stance"])
    reason = stance_result.get("reason", "")
    confidence = stance_result.get("confidence", 0)
    st.markdown(f"""
    <div class="stance-row">
        <span class="stance-doc">{src}</span>
        <span class="stance-arrow">&rarr;</span>
        <span class="stance-doc">{tgt}</span>
        {badge}
        <span class="stance-reason">{reason}</span>
        <span class="stat-pill">{confidence:.0%}</span>
    </div>
    """, unsafe_allow_html=True)


# -- Main Navigation Tabs ------------------------------------------------------
tab_search, tab_browse, tab_compare = st.tabs(["Search & Discover", "Browse Graph", "Compare Documents"])


# ==============================================================================
# TAB 1: Search & Discover
# ==============================================================================
with tab_search:
    section_divider("Search")

    # Inline settings row
    s_col1, s_col2, s_col3, s_col4 = st.columns([5, 1, 1.5, 1.5])
    with s_col1:
        query = st.text_input(
            "Search query",
            placeholder="Search Indonesian legal documents ...",
            label_visibility="collapsed",
        )
    with s_col2:
        top_k = st.number_input("Results", min_value=5, max_value=50, value=10, key="search_topk")
    with s_col3:
        scope_filter = st.selectbox("Scope", ["all", "pasal", "ayat", "diktum"], key="search_scope")
    with s_col4:
        enable_stance = st.checkbox("Stance detection", value=True, key="search_stance")

    search_btn = st.button("Search", type="primary", key="search_btn")

    if query and search_btn:
        with st.spinner("Searching relevant documents ..."):
            try:
                query_embedding = llm_stance.get_embedding(query)
                results = pinecone_client.semantic_search(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    scope_filter=scope_filter if scope_filter != "all" else None,
                )
                st.session_state.search_results = results
            except Exception:
                st.info("Embedding model unavailable. Falling back to metadata search.")
                try:
                    all_docs = neo4j_client.get_all_documents()
                    query_lower = query.lower()
                    matching_docs = [
                        d for d in all_docs
                        if query_lower in str(d.get("judul", "")).lower()
                        or query_lower in str(d.get("doc_id", "")).lower()
                        or query_lower in str(d.get("jenis", "")).lower()
                    ]
                    results = []
                    for doc in matching_docs[:5]:
                        chunks = pinecone_client.fetch_by_doc_id(doc["doc_id"], top_k=5)
                        results.extend(chunks)
                    if not results:
                        for doc in all_docs[:3]:
                            chunks = pinecone_client.fetch_by_doc_id(doc["doc_id"], top_k=3)
                            results.extend(chunks)
                    st.session_state.search_results = results
                except Exception as e2:
                    st.error(f"Search failed: {e2}")
                    st.session_state.search_results = []

            # Extract top 5 unique doc_ids (preserving relevance order)
            raw = st.session_state.search_results
            unique_ids: list[str] = list(dict.fromkeys(r["doc_id"] for r in raw if r.get("doc_id")))[:5]
            st.session_state.search_doc_ids = unique_ids

    # Display search results
    results = st.session_state.search_results
    doc_ids = st.session_state.search_doc_ids

    if results and doc_ids:
        # Stats bar
        st.markdown(
            f'<div style="display:flex;gap:10px;margin:0.5rem 0 0.75rem;">'
            f'<span class="stat-pill"><span class="stat-pill-count">{len(doc_ids)}</span> documents</span>'
            f'<span class="stat-pill"><span class="stat-pill-count">{len(results)}</span> chunks</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Build a doc_id -> judul lookup from Neo4j
        all_docs_cache = neo4j_client.get_all_documents() if neo4j_ok else []
        judul_map = {d.get("doc_id", ""): d.get("judul", "") for d in all_docs_cache}

        # Show document-level cards (grouped by doc_id)
        for did in doc_ids:
            judul = judul_map.get(did, "")
            chunks_for_doc = [r for r in results if r.get("doc_id") == did]
            best = chunks_for_doc[0] if chunks_for_doc else {}
            score = best.get("score")

            title_display = f"{did}"
            if judul:
                title_display += f" -- {judul[:80]}"

            with st.expander(title_display, expanded=False):
                if score:
                    st.caption(f"Top score: {score:.4f} | {len(chunks_for_doc)} chunks matched")
                for i, ch in enumerate(chunks_for_doc[:5]):
                    render_result_card(
                        doc_id=ch["doc_id"],
                        scope=ch.get("scope", "?"),
                        content=ch.get("content", ""),
                        score=ch.get("score"),
                        article_id=ch.get("article_id", ""),
                    )

        # --- Document relationship graph (only the 5 docs + edges between them) ---
        if len(doc_ids) >= 2 and neo4j_ok:
            section_divider("Document Relationships")

            graph_data = neo4j_client.get_edges_between(doc_ids)
            graph_nodes = graph_data["nodes"]
            graph_edges = graph_data["edges"]

            if graph_nodes:
                st.markdown(
                    f'<div style="display:flex;gap:10px;margin-bottom:0.5rem;">'
                    f'<span class="stat-pill"><span class="stat-pill-count">{len(graph_nodes)}</span> documents</span>'
                    f'<span class="stat-pill"><span class="stat-pill-count">{len(graph_edges)}</span> relationships</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Stance detection on edges
                stance_map = {}
                if enable_stance and graph_edges:
                    with st.spinner("Analyzing document relationships ..."):
                        for edge in graph_edges:
                            if edge.get("type") in ("CITES", "HIGHER"):
                                src_id = edge.get("source_id", "")
                                tgt_id = edge.get("target_id", "")
                                cache_key = (src_id, tgt_id)

                                if cache_key not in st.session_state.stance_cache:
                                    src_chunks = [r for r in results if r.get("doc_id") == src_id]
                                    tgt_chunks = [r for r in results if r.get("doc_id") == tgt_id]
                                    if not src_chunks:
                                        src_chunks = pinecone_client.fetch_by_doc_id(src_id, top_k=3)
                                    if not tgt_chunks:
                                        tgt_chunks = pinecone_client.fetch_by_doc_id(tgt_id, top_k=3)

                                    text_a = "\n".join(c.get("content", "") for c in src_chunks[:3])
                                    text_b = "\n".join(c.get("content", "") for c in tgt_chunks[:3])

                                    if text_a and text_b:
                                        sr = llm_stance.classify_stance(text_a, text_b, src_id, tgt_id)
                                        st.session_state.stance_cache[cache_key] = sr

                                if cache_key in st.session_state.stance_cache:
                                    stance_map[cache_key] = st.session_state.stance_cache[cache_key]

                st.markdown('<div class="graph-container">', unsafe_allow_html=True)
                selected = graph_viz.render_document_graph(
                    doc_nodes=graph_nodes,
                    doc_edges=graph_edges,
                    stance_map=stance_map,
                    height=450,
                )
                st.markdown('</div>', unsafe_allow_html=True)

                if selected:
                    st.session_state.selected_node = selected

                # Stance summary
                if stance_map:
                    section_divider("Stance Analysis")
                    for (src, tgt), sr in stance_map.items():
                        render_stance_row(src, tgt, sr)

        # Q&A section
        section_divider("Q&A")
        qa_query = st.text_input(
            "Ask a question about the documents found",
            key="qa_input",
            placeholder="What are the key provisions regarding ...",
        )
        if qa_query and st.button("Submit", key="qa_btn"):
            with st.spinner("Generating answer ..."):
                answer = llm_stance.ask_about_documents(qa_query, results[:5])
                st.markdown(answer)


# ==============================================================================
# TAB 2: Browse Graph
# ==============================================================================
with tab_browse:
    if not neo4j_ok:
        st.error("Neo4j is not connected. Check your configuration.")
    else:
        all_docs = neo4j_client.get_all_documents()

        if not all_docs:
            st.warning("No documents found in the database.")
        else:
            doc_options = [d.get("doc_id", "?") for d in all_docs]
            doc_labels = {
                d.get("doc_id", "?"): f"{d.get('doc_id', '?')} -- {d.get('judul', '')[:60]}"
                for d in all_docs
            }

            # Inline settings row
            b_col1, b_col2, b_col3 = st.columns([5, 1.2, 2])
            with b_col1:
                selected_doc = st.selectbox(
                    "Select document",
                    doc_options,
                    format_func=lambda x: doc_labels.get(x, x),
                    key="browse_doc",
                    label_visibility="collapsed",
                )
            with b_col2:
                k_hops = st.number_input("Hops", min_value=1, max_value=4, value=2, key="browse_hops")
            with b_col3:
                b_stance = st.checkbox("Enable stance detection", value=True, key="browse_stance_chk")

            if selected_doc:
                tab_sub, tab_detail = st.tabs(["Subgraph", "Document Detail"])

                with tab_sub:
                    with st.spinner("Loading subgraph ..."):
                        subgraph = neo4j_client.get_citing_documents(selected_doc, hops=k_hops)

                    if subgraph["nodes"]:
                        st.markdown(
                            f'<div style="display:flex;gap:10px;margin:0.5rem 0;">'
                            f'<span class="stat-pill"><span class="stat-pill-count">{len(subgraph["nodes"])}</span> documents</span>'
                            f'<span class="stat-pill"><span class="stat-pill-count">{len(subgraph["edges"])}</span> relationships</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        stance_map = {}

                        # On-demand stance analysis
                        if b_stance and subgraph["edges"]:
                            if st.button("Analyze Stance", key="browse_analyze"):
                                with st.spinner("Analyzing stances ..."):
                                    for edge in subgraph["edges"]:
                                        if edge.get("type") in ("CITES", "HIGHER"):
                                            src_id = edge.get("source_id", "")
                                            tgt_id = edge.get("target_id", "")
                                            cache_key = (src_id, tgt_id)

                                            if cache_key not in st.session_state.stance_cache:
                                                src_chunks = pinecone_client.fetch_by_doc_id(src_id, top_k=3)
                                                tgt_chunks = pinecone_client.fetch_by_doc_id(tgt_id, top_k=3)
                                                text_a = "\n".join(c.get("content", "") for c in src_chunks[:3])
                                                text_b = "\n".join(c.get("content", "") for c in tgt_chunks[:3])
                                                if text_a and text_b:
                                                    r = llm_stance.classify_stance(text_a, text_b, src_id, tgt_id)
                                                    st.session_state.stance_cache[cache_key] = r

                                            if cache_key in st.session_state.stance_cache:
                                                stance_map[cache_key] = st.session_state.stance_cache[cache_key]

                        # Use cached stances
                        for edge in subgraph.get("edges", []):
                            ckey = (edge.get("source_id", ""), edge.get("target_id", ""))
                            if ckey in st.session_state.stance_cache:
                                stance_map[ckey] = st.session_state.stance_cache[ckey]

                        st.markdown('<div class="graph-container">', unsafe_allow_html=True)
                        selected = graph_viz.render_document_graph(
                            doc_nodes=subgraph["nodes"],
                            doc_edges=subgraph["edges"],
                            stance_map=stance_map,
                            height=500,
                        )
                        st.markdown('</div>', unsafe_allow_html=True)

                        if selected:
                            st.session_state.selected_node = selected

                        # Stance results
                        if stance_map:
                            section_divider("Stance Results")
                            for (src, tgt), sr in stance_map.items():
                                render_stance_row(src, tgt, sr)

                    else:
                        st.info("This document has no CITES/HIGHER relationships with other documents.")

                with tab_detail:
                    with st.spinner("Loading document details ..."):
                        detail = neo4j_client.get_document_detail(selected_doc)

                    if detail and detail.get("document"):
                        doc = detail["document"]
                        st.markdown(f"### {doc.get('doc_id', selected_doc)}")
                        if doc.get("judul"):
                            st.markdown(f"**{doc['judul']}**")

                        # Metadata grid
                        meta_fields = [
                            ("jenis", "Type"), ("tahun", "Year"),
                            ("nomor", "Number"), ("pembentuk", "Issuer"),
                        ]
                        cols = st.columns(len(meta_fields))
                        for col, (key, label) in zip(cols, meta_fields):
                            val = doc.get(key)
                            if val:
                                col.metric(label, str(val))

                        # Konsideran
                        if doc.get("konsideran_menimbang"):
                            with st.expander("Considering (Menimbang)"):
                                items = doc["konsideran_menimbang"]
                                if isinstance(items, list):
                                    for item in items:
                                        st.markdown(f"- {item}")
                                else:
                                    st.text(str(items))

                        if doc.get("konsideran_mengingat"):
                            with st.expander("Referring to (Mengingat)"):
                                items = doc["konsideran_mengingat"]
                                if isinstance(items, list):
                                    for item in items:
                                        st.markdown(f"- {item}")
                                else:
                                    st.text(str(items))

                        # Pasals
                        if detail.get("pasals"):
                            section_divider(f"Articles / Pasal ({len(detail['pasals'])})")
                            for p in detail["pasals"]:
                                name = p.get("name", "?")
                                content = p.get("content", "")
                                bab = p.get("bab_title", "")
                                lbl = name
                                if bab:
                                    lbl += f" -- {bab}"
                                with st.expander(lbl):
                                    if content:
                                        st.text(content)
                                    if p.get("penjelasan") and p["penjelasan"] != "Tidak ada Penjelasan":
                                        st.caption(f"Explanation: {p['penjelasan']}")

                        # Ayats
                        if detail.get("ayats"):
                            section_divider(f"Verses / Ayat ({len(detail['ayats'])})")
                            for a in detail["ayats"]:
                                name = a.get("name", "?")
                                pasal = a.get("pasal_name", "")
                                content = a.get("content", "")
                                with st.expander(f"{pasal} > {name}" if pasal else name):
                                    if content:
                                        st.text(content)

                        # Diktums
                        if detail.get("diktums"):
                            section_divider(f"Diktum ({len(detail['diktums'])})")
                            for dk in detail["diktums"]:
                                name = dk.get("name", "?")
                                content = dk.get("content", "")
                                with st.expander(name):
                                    if content:
                                        st.text(content)
                    else:
                        st.warning("Document details not found.")

        # Full graph overview
        st.markdown("---")
        if st.checkbox("Show full document graph", key="full_graph"):
            with st.spinner("Loading graph ..."):
                overview = neo4j_client.get_graph_overview()
            if overview["nodes"]:
                st.markdown(
                    f'<div style="display:flex;gap:10px;margin-bottom:0.5rem;">'
                    f'<span class="stat-pill"><span class="stat-pill-count">{len(overview["nodes"])}</span> documents</span>'
                    f'<span class="stat-pill"><span class="stat-pill-count">{len(overview["edges"])}</span> relationships</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.markdown('<div class="graph-container">', unsafe_allow_html=True)
                graph_viz.render_document_graph(
                    doc_nodes=overview["nodes"],
                    doc_edges=overview["edges"],
                    height=600,
                )
                st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# TAB 3: Compare Documents
# ==============================================================================
with tab_compare:
    if not neo4j_ok:
        st.error("Neo4j is not connected. Check your configuration.")
    else:
        all_docs_cmp = neo4j_client.get_all_documents()
        doc_options_cmp = [d.get("doc_id", "?") for d in all_docs_cmp]
        doc_labels_cmp = {
            d.get("doc_id", "?"): f"{d.get('doc_id', '?')} -- {d.get('judul', '')[:60]}"
            for d in all_docs_cmp
        }

        if len(doc_options_cmp) < 2:
            st.warning("At least 2 documents are required for comparison.")
        else:
            cmp_col1, cmp_col2, cmp_col3 = st.columns([4, 4, 2])

            with cmp_col1:
                doc_a = st.selectbox(
                    "Document A", doc_options_cmp, index=0, key="compare_a",
                    format_func=lambda x: doc_labels_cmp.get(x, x),
                )
            with cmp_col2:
                default_b = 1 if len(doc_options_cmp) > 1 else 0
                doc_b = st.selectbox(
                    "Document B", doc_options_cmp, index=default_b, key="compare_b",
                    format_func=lambda x: doc_labels_cmp.get(x, x),
                )
            with cmp_col3:
                st.markdown("<br>", unsafe_allow_html=True)
                compare_btn = st.button("Compare", type="primary", key="compare_btn")

            if doc_a and doc_b and doc_a != doc_b:
                if compare_btn:
                    col_left, col_right = st.columns(2)

                    with st.spinner("Loading documents ..."):
                        detail_a = neo4j_client.get_document_detail(doc_a)
                        detail_b = neo4j_client.get_document_detail(doc_b)

                    with st.spinner("Fetching content from Pinecone ..."):
                        chunks_a = pinecone_client.fetch_by_doc_id(doc_a, top_k=20)
                        chunks_b = pinecone_client.fetch_by_doc_id(doc_b, top_k=20)

                    with col_left:
                        st.markdown(f"### {doc_a}")
                        if detail_a and detail_a.get("document"):
                            d = detail_a["document"]
                            if d.get("judul"):
                                st.markdown(f"**{d['judul']}**")
                            st.caption(f"{d.get('jenis', '')} | Year {d.get('tahun', '')} | No. {d.get('nomor', '')}")

                        st.markdown(
                            f'<span class="stat-pill"><span class="stat-pill-count">{len(chunks_a)}</span> chunks</span>',
                            unsafe_allow_html=True,
                        )
                        for c in chunks_a[:5]:
                            render_result_card(
                                doc_id=c.get("doc_id", ""),
                                scope=c.get("scope", "?"),
                                content=c.get("content", ""),
                                article_id=c.get("article_id", ""),
                            )

                    with col_right:
                        st.markdown(f"### {doc_b}")
                        if detail_b and detail_b.get("document"):
                            d = detail_b["document"]
                            if d.get("judul"):
                                st.markdown(f"**{d['judul']}**")
                            st.caption(f"{d.get('jenis', '')} | Year {d.get('tahun', '')} | No. {d.get('nomor', '')}")

                        st.markdown(
                            f'<span class="stat-pill"><span class="stat-pill-count">{len(chunks_b)}</span> chunks</span>',
                            unsafe_allow_html=True,
                        )
                        for c in chunks_b[:5]:
                            render_result_card(
                                doc_id=c.get("doc_id", ""),
                                scope=c.get("scope", "?"),
                                content=c.get("content", ""),
                                article_id=c.get("article_id", ""),
                            )

                    # Overall stance analysis
                    section_divider("Relationship Analysis")

                    with st.spinner("Analyzing relationship ..."):
                        text_a = "\n\n".join(c.get("content", "") for c in chunks_a[:5])
                        text_b = "\n\n".join(c.get("content", "") for c in chunks_b[:5])

                        if text_a and text_b:
                            overall = llm_stance.classify_stance(text_a, text_b, doc_a, doc_b)
                            stance = overall["stance"]
                            eng_label = graph_viz.STANCE_LABELS.get(stance, stance)

                            r_col1, r_col2, r_col3 = st.columns([1.5, 1, 3.5])
                            with r_col1:
                                st.metric("Stance", eng_label)
                            with r_col2:
                                st.metric("Confidence", f"{overall.get('confidence', 0):.0%}")
                            with r_col3:
                                st.info(overall.get("reason", ""))

                            # Per-section comparison
                            section_divider("Section-by-Section Comparison")
                            pairs_to_compare = []
                            for ca in chunks_a[:3]:
                                for cb in chunks_b[:3]:
                                    pairs_to_compare.append({
                                        "text_a": ca.get("content", ""),
                                        "text_b": cb.get("content", ""),
                                        "doc_a_id": f"{doc_a}/{ca.get('scope', '')}",
                                        "doc_b_id": f"{doc_b}/{cb.get('scope', '')}",
                                        "label_a": ca.get("scope", "?"),
                                        "label_b": cb.get("scope", "?"),
                                    })

                            section_results = llm_stance.batch_classify(pairs_to_compare)

                            for pair, result in zip(pairs_to_compare, section_results):
                                s_badge = graph_viz.stance_badge_html(result["stance"])
                                c1, c2, c3, c4 = st.columns([2, 2, 1.2, 3])
                                with c1:
                                    st.text(f"{doc_a}: {pair['label_a']}")
                                with c2:
                                    st.text(f"{doc_b}: {pair['label_b']}")
                                with c3:
                                    st.markdown(s_badge, unsafe_allow_html=True)
                                with c4:
                                    st.caption(result.get("reason", ""))
                        else:
                            st.warning("Insufficient document content for analysis.")

            elif doc_a == doc_b:
                st.warning("Please select two different documents.")


# -- Footer -------------------------------------------------------------------
st.markdown(
    '<div class="app-footer">'
    'GraphRAG &mdash; Legal Document Relationship Explorer'
    ' &nbsp;&middot;&nbsp; Neo4j + Pinecone + OpenRouter'
    '</div>',
    unsafe_allow_html=True,
)
