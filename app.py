"""
GraphRAG -- Legal Document Relationship Explorer
Streamlit app for visualizing and analyzing relationships between Indonesian legal documents.
"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import os, re, glob
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
if "search_answer" not in st.session_state:
    st.session_state.search_answer = None
if "search_context_docs" not in st.session_state:
    st.session_state.search_context_docs = {}
# (benchmark_results and causality_results are now stored as CSV files in output/)

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


# ── Benchmark helpers — imported from shared module ──────────────────────────
from utils.benchmark_helpers import (
    extract_documents as _extract_documents,
    get_correct_doc_id as _get_correct_doc_id,
    get_unique_doc_ids as _get_unique_doc_ids,
)


# -- Main Navigation Tabs (4 tabs) --------------------------------------------
tab_search, tab_browse, tab_compare, tab_kausalitas = st.tabs(
    ["Search & Discover", "Browse Graph", "Compare Documents", "Kausalitas"]
)


# ==============================================================================
# TAB 1: Search & Discover
# ==============================================================================
with tab_search:
    section_divider("Tanya Jawab Regulasi")

    query = st.text_input(
        "Pertanyaan",
        placeholder="Ketik pertanyaan tentang regulasi Indonesia ...",
        label_visibility="collapsed",
        key="search_query",
    )

    search_btn = st.button("Cari & Jawab", type="primary", key="search_btn")

    if query and search_btn:
        with st.spinner("Mencari dokumen relevan ..."):
            try:
                # 1. Embed query via HuggingFace Indo-LegalBERT
                query_embedding = llm_stance.get_embedding(query)

                # 2. Search VDB with high top_k to get 3 unique doc_ids
                raw_results = pinecone_client.semantic_search(
                    query_embedding=query_embedding,
                    top_k=30,
                )

                # 3. Get 3 unique doc_ids
                primary_doc_ids = _get_unique_doc_ids(raw_results, 3)

                if not primary_doc_ids:
                    st.warning("Tidak ditemukan dokumen yang relevan.")
                    st.stop()

                st.session_state.search_doc_ids = primary_doc_ids
                st.session_state.search_results = raw_results

                # 4. Fetch ALL chunks for each primary doc
                all_context_chunks = []
                context_docs = {}  # doc_id -> {source: "VDB", chunks: [...]}

                for did in primary_doc_ids:
                    chunks = pinecone_client.fetch_by_doc_id(did, top_k=100)
                    context_docs[did] = {"source": "VDB (Primary)", "chunks": chunks}
                    all_context_chunks.extend(chunks)

                # 5. For each primary doc, find up to 3 related docs in Neo4j
                related_doc_ids = []
                if neo4j_ok:
                    for did in primary_doc_ids:
                        related = neo4j_client.get_related_documents(did, limit=3)
                        for rdoc in related:
                            rdid = rdoc.get("doc_id", "")
                            if rdid and rdid not in primary_doc_ids and rdid not in related_doc_ids:
                                related_doc_ids.append(rdid)

                # 6. Fetch content for related docs
                for rdid in related_doc_ids:
                    chunks = pinecone_client.fetch_by_doc_id(rdid, top_k=50)
                    if chunks:
                        context_docs[rdid] = {"source": "Neo4j (Related)", "chunks": chunks}
                        all_context_chunks.extend(chunks)

                st.session_state.search_context_docs = context_docs

                # 7. Build combined context and generate answer
                with st.spinner("Menghasilkan jawaban dengan GPT ..."):
                    answer = llm_stance.ask_about_documents(query, all_context_chunks[:20])
                    st.session_state.search_answer = answer

            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.search_answer = None

    # Display answer
    if st.session_state.search_answer:
        section_divider("Jawaban")
        st.markdown(st.session_state.search_answer)

    # Display source documents
    context_docs = st.session_state.search_context_docs
    if context_docs:
        section_divider("Dokumen Sumber")

        # Stats
        n_primary = sum(1 for v in context_docs.values() if "Primary" in v["source"])
        n_related = sum(1 for v in context_docs.values() if "Related" in v["source"])
        total_chunks = sum(len(v["chunks"]) for v in context_docs.values())

        st.markdown(
            f'<div style="display:flex;gap:10px;margin:0.5rem 0 0.75rem;">'
            f'<span class="stat-pill"><span class="stat-pill-count">{n_primary}</span> dokumen VDB</span>'
            f'<span class="stat-pill"><span class="stat-pill-count">{n_related}</span> dokumen terkait (Neo4j)</span>'
            f'<span class="stat-pill"><span class="stat-pill-count">{total_chunks}</span> total chunks</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        for did, info in context_docs.items():
            with st.expander(f"{did}  [{info['source']}] — {len(info['chunks'])} chunks"):
                for ch in info["chunks"][:5]:
                    render_result_card(
                        doc_id=ch.get("doc_id", ""),
                        scope=ch.get("scope", "?"),
                        content=ch.get("content", ""),
                        score=ch.get("score"),
                        article_id=ch.get("article_id", ""),
                    )

        # Show relationship graph between source docs
        all_source_ids = list(context_docs.keys())
        if len(all_source_ids) >= 2 and neo4j_ok:
            section_divider("Hubungan Antar Dokumen")
            graph_data = neo4j_client.get_edges_between(all_source_ids)
            if graph_data["nodes"]:
                st.markdown('<div class="graph-container">', unsafe_allow_html=True)
                graph_viz.render_document_graph(
                    doc_nodes=graph_data["nodes"],
                    doc_edges=graph_data["edges"],
                    height=400,
                )
                st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# TAB 2: Browse Graph
# ==============================================================================
with tab_browse:
    if not neo4j_ok:
        st.error("Neo4j tidak terhubung. Periksa konfigurasi.")
    else:
        section_divider("Graph Dokumen")

        with st.spinner("Memuat graph ..."):
            overview = neo4j_client.get_graph_overview()

        if overview["nodes"]:
            st.markdown(
                f'<div style="display:flex;gap:10px;margin:0.5rem 0;">'
                f'<span class="stat-pill"><span class="stat-pill-count">{len(overview["nodes"])}</span> dokumen</span>'
                f'<span class="stat-pill"><span class="stat-pill-count">{len(overview["edges"])}</span> relasi</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown('<div class="graph-container">', unsafe_allow_html=True)
            selected = graph_viz.render_document_graph(
                doc_nodes=overview["nodes"],
                doc_edges=overview["edges"],
                height=600,
            )
            st.markdown('</div>', unsafe_allow_html=True)

            if selected:
                st.session_state.selected_node = selected

            # Show selected node detail
            if st.session_state.selected_node:
                sel_id = st.session_state.selected_node
                section_divider(f"Detail: {sel_id}")
                detail = neo4j_client.get_document_detail(sel_id)
                if detail and detail.get("document"):
                    doc = detail["document"]
                    if doc.get("judul"):
                        st.markdown(f"**{doc['judul']}**")
                    meta_fields = [
                        ("jenis", "Jenis"), ("tahun", "Tahun"),
                        ("nomor", "Nomor"), ("pembentuk", "Pembentuk"),
                    ]
                    cols = st.columns(len(meta_fields))
                    for col, (key, label) in zip(cols, meta_fields):
                        val = doc.get(key)
                        if val:
                            col.metric(label, str(val))
        else:
            st.info("Tidak ada dokumen dalam database Neo4j.")


# ==============================================================================
# TAB 3: Compare Documents (IR Benchmark) — read-only from output/retrieval/
# ==============================================================================
with tab_compare:
    section_divider("Information Retrieval Benchmark")

    import pandas as pd

    _detail_dir = os.path.join(os.path.dirname(__file__), "output", "retrieval", "detailed retrieval")
    _metrics_dir = os.path.join(os.path.dirname(__file__), "output", "retrieval", "metrics")
    _benchmark_csvs = sorted(glob.glob(os.path.join(_detail_dir, "*_v3.csv"))) if os.path.isdir(_detail_dir) else []

    if not _benchmark_csvs:
        st.info(
            "Belum ada hasil benchmark.  "
            "Jalankan di terminal:  \n"
            "```bash\npython run_benchmark_v3.py\n```"
        )
    else:
        _bm_labels = {f: os.path.basename(f) for f in _benchmark_csvs}
        _selected_bm = st.selectbox(
            "Pilih hasil benchmark",
            _benchmark_csvs,
            format_func=lambda x: _bm_labels[x],
            key="benchmark_csv_select",
        )

        if _selected_bm:
            df_bm = pd.read_csv(_selected_bm)

            # Load summary from metrics/ directory
            _base = os.path.basename(_selected_bm).replace("_v3.csv", "_v3_summary.csv")
            _summary_path = os.path.join(_metrics_dir, _base)
            if os.path.isfile(_summary_path):
                df_summary = pd.read_csv(_summary_path)
                summary_dict = dict(zip(df_summary["Metric"], df_summary["Value"]))

                section_divider("Rata-rata Skor")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Avg Recall GraphRAG", f"{float(summary_dict.get('Avg_Recall_GraphRAG', 0)):.2%}")
                m2.metric("Avg Precision GraphRAG", f"{float(summary_dict.get('Avg_Precision_GraphRAG', 0)):.2%}")
                m3.metric("Avg Recall VDB", f"{float(summary_dict.get('Avg_Recall_VDB', 0)):.2%}")
                m4.metric("Avg Precision VDB", f"{float(summary_dict.get('Avg_Precision_VDB', 0)):.2%}")

                total_q = summary_dict.get("Total_Questions", "?")
                scored_q = summary_dict.get("Scored_Questions", "")
                skipped_q = summary_dict.get("Skipped_Questions", "")
                elapsed = summary_dict.get("Elapsed_Seconds", "?")
                if scored_q:
                    st.caption(f"{total_q} pertanyaan ({scored_q} scored, {skipped_q} skipped)  ·  {elapsed}s")
                elif "Total_Questions" in summary_dict:
                    st.caption(f"{total_q} pertanyaan  ·  {elapsed}s")

            section_divider("Hasil Per Pertanyaan")

            # Format numeric columns as percentages for display
            display_df = df_bm.copy()
            for col in ["Recall_GraphRAG", "Precision_GraphRAG", "Recall_VDB", "Precision_VDB"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(
                        lambda v: f"{float(v):.2%}" if pd.notna(v) else "—"
                    )

            st.dataframe(display_df, use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 4: Kausalitas — read-only from output/kausalitas/
# ==============================================================================
with tab_kausalitas:
    section_divider("Analisis Kausalitas Antar Dokumen")

    _kausalitas_result_path = os.path.join(os.path.dirname(__file__), "output", "kausalitas", "kausalitas_results.csv")
    _kausalitas_summary_path = os.path.join(os.path.dirname(__file__), "output", "kausalitas", "kausalitas_summary.csv")

    if not os.path.isfile(_kausalitas_result_path):
        st.info(
            "Belum ada hasil analisis kausalitas.  "
            "Jalankan di terminal:  \n"
            "```bash\npython run_kausalitas.py\n```"
        )
    else:
        st.markdown(
            "Menampilkan hasil analisis kausalitas (ENTAILMENT / CONTRADICTION / NEUTRAL) "
            "untuk semua pasangan dokumen yang terhubung di Neo4j."
        )

        df_kaus = pd.read_csv(_kausalitas_result_path)

        # Summary metrics
        if os.path.isfile(_kausalitas_summary_path):
            df_ks = pd.read_csv(_kausalitas_summary_path)
            ks_dict = dict(zip(df_ks["Metric"], df_ks["Value"]))
        else:
            # Compute from data
            ks_dict = {
                "CONTRADICTION": int((df_kaus["Kausalitas"] == "CONTRADICTION").sum()),
                "ENTAILMENT": int((df_kaus["Kausalitas"] == "ENTAILMENT").sum()),
                "NEUTRAL": int((df_kaus["Kausalitas"] == "NEUTRAL").sum()),
            }

        section_divider("Ringkasan")
        c1, c2, c3 = st.columns(3)
        c1.metric("CONTRADICTION", ks_dict.get("CONTRADICTION", 0))
        c2.metric("ENTAILMENT", ks_dict.get("ENTAILMENT", 0))
        c3.metric("NEUTRAL", ks_dict.get("NEUTRAL", 0))

        st.caption(f"{len(df_kaus)} pasangan dokumen")

        section_divider("Hasil Analisis")

        def color_kausalitas(val):
            colors = {
                "CONTRADICTION": "background-color: #fee2e2; color: #dc2626;",
                "ENTAILMENT": "background-color: #d1fae5; color: #059669;",
                "NEUTRAL": "background-color: #f1f5f9; color: #6b7280;",
                "Error": "background-color: #fef3c7; color: #d97706;",
            }
            return colors.get(val, "")

        styled = df_kaus.style.map(color_kausalitas, subset=["Kausalitas"])
        st.dataframe(styled, use_container_width=True, hide_index=True)


# -- Footer -------------------------------------------------------------------
st.markdown(
    '<div class="app-footer">'
    'GraphRAG &mdash; Legal Document Relationship Explorer'
    ' &nbsp;&middot;&nbsp; Neo4j + Pinecone + HuggingFace + OpenRouter'
    '</div>',
    unsafe_allow_html=True,
)
