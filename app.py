"""
GraphRAG -- Legal Document Relationship Explorer
Streamlit app for visualizing and analyzing relationships between Indonesian legal documents.
"""

import streamlit as st
import os

# ── Streamlit Cloud: inject secrets into env vars ─────────────────────────────
# On Streamlit Cloud there is no .env file. Secrets entered in the dashboard
# (TOML format) are available via st.secrets.  We copy them into os.environ
# so that every os.getenv() call in utils/* keeps working unchanged.
if hasattr(st, "secrets") and len(st.secrets):
    for key, value in st.secrets.items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)

from dotenv import load_dotenv

load_dotenv()  # local .env still wins for local dev (already set → setdefault is no-op)

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
        background: rgba(30,58,95,0.05);
    }
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        background: linear-gradient(135deg, #1e3a5f, #2563eb) !important;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(30,58,95,0.3);
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    /* ── Buttons ─────────────────────────────────────────────────────────── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1e3a5f, #2563eb) !important;
        border: none !important;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 0.55rem 1.6rem;
        box-shadow: 0 2px 12px rgba(30,58,95,0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        letter-spacing: 0.2px;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 4px 20px rgba(30,58,95,0.45);
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
        border-color: #2563eb !important;
        color: #1e3a5f !important;
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
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.15) !important;
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
        background: linear-gradient(180deg, #1e3a5f, #3b82f6);
        border-radius: 3px 0 0 3px;
    }
    .result-card:hover {
        border-color: #bfdbfe;
        box-shadow: 0 4px 16px rgba(30,58,95,0.08);
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
        border-color: #bfdbfe;
        box-shadow: 0 2px 12px rgba(30,58,95,0.06);
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
        border-color: #bfdbfe !important;
        background: #eff6ff !important;
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
        background: linear-gradient(135deg, #1e3a5f, #2563eb);
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
        border-top-color: #2563eb !important;
    }

    /* ── Kausalitas Dashboard ─────────────────────────────────────────────── */
    .kpi-row {
        display: flex;
        gap: 16px;
        margin-bottom: 1.2rem;
    }
    .kpi-card {
        flex: 1;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 22px;
        position: relative;
        overflow: hidden;
        transition: all 0.25s ease;
    }
    .kpi-card::before {
        content: "";
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 4px;
        border-radius: 4px 0 0 4px;
    }
    .kpi-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        transform: translateY(-1px);
    }
    .kpi-card.kpi-entailment::before { background: linear-gradient(180deg, #059669, #34d399); }
    .kpi-card.kpi-contradiction::before { background: linear-gradient(180deg, #dc2626, #f87171); }
    .kpi-card.kpi-neutral::before { background: linear-gradient(180deg, #64748b, #94a3b8); }
    .kpi-card.kpi-total::before { background: linear-gradient(180deg, #1e3a5f, #3b82f6); }
    .kpi-label {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1;
    }
    .kpi-card.kpi-entailment .kpi-value { color: #059669; }
    .kpi-card.kpi-contradiction .kpi-value { color: #dc2626; }
    .kpi-card.kpi-neutral .kpi-value { color: #64748b; }
    .kpi-card.kpi-total .kpi-value { color: #1e3a5f; }

    mark.pasal-ref {
        background: linear-gradient(135deg, #eff6ff, #dbeafe);
        color: #1e3a5f;
        font-weight: 600;
        padding: 1px 6px;
        border-radius: 4px;
        font-size: 0.92em;
        border: 1px solid #bfdbfe;
    }

    .detail-card {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 6px;
        background: #fafbfe;
    }
    .detail-card-header {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 10px;
    }
    .detail-card-docs {
        font-weight: 600;
        font-size: 0.85rem;
        color: #1e293b;
    }
    .detail-card-arrow {
        color: #94a3b8;
        font-size: 0.85rem;
    }
    .detail-card-rel {
        font-size: 0.72rem;
        color: #64748b;
        background: #f1f5f9;
        padding: 2px 10px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        font-weight: 500;
    }
    .detail-card-reason {
        font-size: 0.84rem;
        color: #475569;
        line-height: 1.6;
        margin-top: 8px;
        padding-top: 10px;
        border-top: 1px solid #f1f5f9;
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
if "search_edges" not in st.session_state:
    st.session_state.search_edges = None
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
    extract_doc_ids_from_question as _extract_doc_ids_from_question,
)


def _build_interleaved_context(
    primary_doc_ids: list[str],
    related_doc_ids: list[str],
    context_docs: dict,
    max_chunks: int = 30,
    max_chars: int = 12000,
) -> list[dict]:
    """Build LLM context by round-robin interleaving chunks from all docs.

    Ensures the LLM sees content from EVERY relevant document, not just
    the first one.  Primary docs get priority; semantic-scored chunks first.
    """
    result: list[dict] = []
    total_chars = 0

    # Prepare per-doc queues: semantic-scored chunks first, then the rest
    doc_queues: dict[str, list[dict]] = {}
    for did in primary_doc_ids + related_doc_ids:
        info = context_docs.get(did)
        if not info:
            continue
        chunks = list(info["chunks"])
        scored = sorted(
            [c for c in chunks if c.get("score") is not None],
            key=lambda c: c.get("score", 0),
            reverse=True,
        )
        unscored = [c for c in chunks if c.get("score") is None]
        doc_queues[did] = scored + unscored

    if not doc_queues:
        return []

    # Round-robin: take one chunk from each doc in turn
    seen_ids: set[str] = set()
    doc_keys = list(doc_queues.keys())
    idx_map = {did: 0 for did in doc_keys}
    exhausted: set[str] = set()

    while (
        len(result) < max_chunks
        and total_chars < max_chars
        and len(exhausted) < len(doc_keys)
    ):
        for did in doc_keys:
            if did in exhausted:
                continue
            queue = doc_queues[did]
            idx = idx_map[did]
            if idx >= len(queue):
                exhausted.add(did)
                continue
            chunk = queue[idx]
            idx_map[did] = idx + 1
            cid = chunk.get("id", "")
            if cid and cid in seen_ids:
                continue
            if cid:
                seen_ids.add(cid)
            content = chunk.get("content", "")
            total_chars += len(content)
            result.append(chunk)
            if len(result) >= max_chunks or total_chars >= max_chars:
                break

    return result


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
        "Masukkan pertanyaan hukum Anda",
        placeholder="Contoh: Apakah perusahaan dapat membagikan dividen interim?",
        key="search_query",
    )

    search_btn = st.button("Cari & Jawab", type="primary", key="search_btn")

    if query and search_btn:
        progress_container = st.container()
        try:
            # ════════════════════════════════════════════════════════════
            # GRAPHRAG-FIRST RETRIEVAL PIPELINE (with early-exit gates)
            # ════════════════════════════════════════════════════════════

            # ── Phase A: Query Analysis ──────────────────────────────
            with progress_container.status("🔍 **Menganalisis pertanyaan …**", expanded=True) as status_a:
                # A1. Parse explicit doc references from query text
                regex_doc_ids = _extract_doc_ids_from_question(query)

                # A2. LLM query expansion
                expanded_terms = llm_stance.expand_query(query)
                status_a.update(label="🔍 **Analisis pertanyaan** — selesai", state="complete", expanded=False)

            # ── Phase B: GraphRAG Document Discovery ─────────────────
            with progress_container.status("📚 **Mencari dokumen di Knowledge Graph …**", expanded=True) as status_b:
                graph_doc_ids: list[str] = []
                if neo4j_ok:
                    all_neo4j_docs = neo4j_client.get_all_documents()
                    graph_doc_ids = llm_stance.smart_doc_lookup(query, all_neo4j_docs)
                status_b.update(label=f"📚 **Knowledge Graph** — {len(graph_doc_ids)} dokumen ditemukan", state="complete", expanded=False)

            # ── Gate 1: Sufficiency check (regex + GraphRAG) ─────────
            with progress_container.status("⚖️ **Mengevaluasi kecukupan dokumen (Gate 1) …**", expanded=True) as status_g1:
                gate1_doc_ids = list(dict.fromkeys(
                    list(graph_doc_ids) + list(regex_doc_ids)
                ))  # Deduplicated, order preserved

                early_exit = False
                raw_results: list[dict] = []
                semantic_chunks_by_doc: dict[str, list[dict]] = {}
                graph_expanded_ids: list[str] = []

                if gate1_doc_ids and neo4j_ok:
                    # Build summaries from Neo4j for sufficiency check
                    gate1_summaries: dict[str, str] = {}
                    for did in gate1_doc_ids[:10]:
                        try:
                            detail = neo4j_client.get_document_detail(did)
                            pasals = detail.get("pasals", [])
                            if pasals:
                                texts = [p.get("content", "") or p.get("name", "") for p in pasals[:3]]
                                gate1_summaries[did] = " ".join(t for t in texts if t)[:500]
                            elif detail.get("document", {}).get("judul"):
                                gate1_summaries[did] = detail["document"]["judul"]
                            else:
                                gate1_summaries[did] = ""
                        except Exception:
                            gate1_summaries[did] = ""

                    if llm_stance.judge_sufficiency(query, gate1_doc_ids, gate1_summaries):
                        # Early exit: skip C, D, E, F → jump to G
                        early_exit = True
                        primary_doc_ids = gate1_doc_ids[:7]
                        merged_doc_ids = gate1_doc_ids
                        added = set(gate1_doc_ids)

                if early_exit:
                    status_g1.update(label="⚖️ **Gate 1** — cukup, lanjut ke jawaban", state="complete", expanded=False)
                else:
                    status_g1.update(label="⚖️ **Gate 1** — perlu pencarian lebih lanjut", state="complete", expanded=False)

            if not early_exit:
                # ── Phase C (lite): VDB Semantic Search ──────────────
                with progress_container.status("🔎 **Pencarian semantik (Vector DB) …**", expanded=True) as status_c:
                    query_embedding = llm_stance.get_embedding(query)
                    raw_results = pinecone_client.semantic_search(
                        query_embedding=query_embedding,
                        top_k=30,
                    )

                    # Group VDB semantic hits by doc (needed for summaries)
                    for hit in raw_results:
                        did = hit.get("doc_id", "")
                        if did:
                            semantic_chunks_by_doc.setdefault(did, []).append(hit)

                    vdb_doc_ids = _get_unique_doc_ids(raw_results, 10)
                    status_c.update(label=f"🔎 **Vector DB** — {len(vdb_doc_ids)} dokumen", state="complete", expanded=False)

                # ── Phase D: Merge all doc sources ───────────────────
                with progress_container.status("🔗 **Menggabungkan sumber dokumen …**", expanded=True) as status_d:
                    merged_doc_ids: list[str] = []
                    added: set[str] = set()

                    for did in graph_doc_ids:
                        if did not in added:
                            merged_doc_ids.append(did)
                            added.add(did)
                    for did in regex_doc_ids:
                        if did not in added:
                            merged_doc_ids.append(did)
                            added.add(did)
                    for did in vdb_doc_ids:
                        if did not in added:
                            merged_doc_ids.append(did)
                            added.add(did)

                    if not merged_doc_ids:
                        st.warning("Tidak ditemukan dokumen yang relevan.")
                        st.stop()
                    status_d.update(label=f"🔗 **Merge** — {len(merged_doc_ids)} dokumen kandidat", state="complete", expanded=False)

                # ── Gate 2: Sufficiency check (merged candidates) ────
                with progress_container.status("⚖️ **Mengevaluasi kecukupan dokumen (Gate 2) …**", expanded=True) as status_g2:
                    gate2_summaries: dict[str, str] = {}
                    for did in merged_doc_ids[:10]:
                        if did in semantic_chunks_by_doc:
                            gate2_summaries[did] = semantic_chunks_by_doc[did][0].get("content", "")[:500]
                        else:
                            gate2_summaries[did] = ""

                    gate2_passed = llm_stance.judge_sufficiency(query, merged_doc_ids, gate2_summaries)

                    if gate2_passed:
                        # Skip E, F → jump to G
                        primary_doc_ids = merged_doc_ids[:7]
                        # Ensure graph + regex picks are always included
                        for did in list(graph_doc_ids[:5]) + list(regex_doc_ids):
                            if did not in primary_doc_ids:
                                primary_doc_ids.append(did)
                        status_g2.update(label="⚖️ **Gate 2** — cukup, lanjut ke jawaban", state="complete", expanded=False)
                    else:
                        status_g2.update(label="⚖️ **Gate 2** — perlu pencarian mendalam", state="complete", expanded=False)

                if not gate2_passed:
                    # ── Phase C (full): Expand VDB search ────────────
                    with progress_container.status("🔎 **Memperluas pencarian semantik …**", expanded=True) as status_cf:
                        if len(raw_results) < 60:
                            more_results = pinecone_client.semantic_search(
                                query_embedding=query_embedding,
                                top_k=100,
                            )
                            seen_ids: set[str] = {h.get("id", "") for h in raw_results}
                            for hit in more_results:
                                hid = hit.get("id", "")
                                if hid and hid not in seen_ids:
                                    raw_results.append(hit)
                                    seen_ids.add(hid)
                                    did = hit.get("doc_id", "")
                                    if did:
                                        semantic_chunks_by_doc.setdefault(did, []).append(hit)

                        # Also search with expanded terms
                        seen_ids_exp: set[str] = {h.get("id", "") for h in raw_results}
                        for term in expanded_terms[:3]:
                            try:
                                term_emb = llm_stance.get_embedding(term)
                                term_results = pinecone_client.semantic_search(
                                    query_embedding=term_emb, top_k=30,
                                )
                                for hit in term_results:
                                    hid = hit.get("id", "")
                                    if hid and hid not in seen_ids_exp:
                                        raw_results.append(hit)
                                        seen_ids_exp.add(hid)
                                        did = hit.get("doc_id", "")
                                        if did:
                                            semantic_chunks_by_doc.setdefault(did, []).append(hit)
                            except Exception:
                                pass

                        # Re-merge with expanded VDB results
                        vdb_doc_ids_full = _get_unique_doc_ids(raw_results, 20)
                        for did in vdb_doc_ids_full:
                            if did not in added:
                                merged_doc_ids.append(did)
                                added.add(did)
                        status_cf.update(label=f"🔎 **VDB diperluas** — {len(raw_results)} chunks", state="complete", expanded=False)

                    # ── Phase E: Deep Graph Traversal ────────────────
                    with progress_container.status("🕸️ **Menelusuri relasi antar-dokumen …**", expanded=True) as status_e:
                        if neo4j_ok:
                            for did in merged_doc_ids[:5]:
                                try:
                                    subgraph = neo4j_client.get_citing_documents(did, hops=2)
                                    for node in subgraph.get("nodes", []):
                                        ndid = node.get("doc_id", "")
                                        if ndid and ndid not in added:
                                            graph_expanded_ids.append(ndid)
                                            added.add(ndid)
                                except Exception:
                                    pass
                        status_e.update(label=f"🕸️ **Graph traversal** — {len(graph_expanded_ids)} dokumen terkait", state="complete", expanded=False)

                    # ── Phase F: Re-ranking ALL candidates ───────────
                    with progress_container.status("🏆 **Re-ranking dokumen …**", expanded=True) as status_f:
                        all_candidate_ids = merged_doc_ids + graph_expanded_ids
                        doc_summaries: dict[str, str] = {}

                        for did in all_candidate_ids:
                            summary = ""
                            if did in semantic_chunks_by_doc:
                                summary = semantic_chunks_by_doc[did][0].get("content", "")
                            if not summary:
                                vdb_chunks = pinecone_client.fetch_by_doc_id(did, top_k=3)
                                if vdb_chunks:
                                    summary = vdb_chunks[0].get("content", "")
                            if not summary and neo4j_ok:
                                try:
                                    detail = neo4j_client.get_document_detail(did)
                                    pasals = detail.get("pasals", [])
                                    if pasals:
                                        texts = [p.get("content", "") or p.get("name", "") for p in pasals[:3]]
                                        summary = " ".join(t for t in texts if t)
                                except Exception:
                                    pass
                            doc_summaries[did] = summary[:500] if summary else ""

                        ranked = llm_stance.rerank_documents(query, doc_summaries)

                        # Keep docs scoring >= 3, max 7
                        primary_doc_ids = [did for did, score in ranked if score >= 3.0][:7]

                        # Ensure graph + regex picks are always included
                        for did in list(graph_doc_ids[:5]) + list(regex_doc_ids):
                            if did not in primary_doc_ids:
                                primary_doc_ids.append(did)

                        if not primary_doc_ids:
                            primary_doc_ids = merged_doc_ids[:5]
                        status_f.update(label=f"🏆 **Re-rank** — top {len(primary_doc_ids)} dokumen", state="complete", expanded=False)

            st.session_state.search_doc_ids = primary_doc_ids
            st.session_state.search_results = raw_results

            # ── Phase G: Content Assembly (VDB + Neo4j) ──────────────
            with progress_container.status("📖 **Mengumpulkan konten dokumen …**", expanded=True) as status_g:
                context_docs: dict[str, dict] = {}
                seen_chunk_ids: set[str] = set()

                for did in primary_doc_ids:
                    doc_chunks: list[dict] = []

                    # G1. VDB semantic hits first
                    sem_chunks = semantic_chunks_by_doc.get(did, [])
                    for ch in sem_chunks:
                        doc_chunks.append(ch)
                        seen_chunk_ids.add(ch.get("id", ""))

                    # G2. VDB fetch_by_doc_id supplement
                    extra = pinecone_client.fetch_by_doc_id(did, top_k=80)
                    for ch in extra:
                        cid = ch.get("id", "")
                        if cid not in seen_chunk_ids:
                            doc_chunks.append(ch)
                            seen_chunk_ids.add(cid)

                    # G3. Neo4j Pasal/Ayat content (always for primary docs)
                    if neo4j_ok:
                        try:
                            detail = neo4j_client.get_document_detail(did)
                            for pasal in detail.get("pasals", []):
                                p_content = pasal.get("content", "")
                                p_name = pasal.get("name", "")
                                neo4j_id = f"neo4j-{did}-{p_name}"
                                if p_content and len(p_content) > 20 and neo4j_id not in seen_chunk_ids:
                                    doc_chunks.append({
                                        "id": neo4j_id,
                                        "doc_id": did,
                                        "article_id": p_name,
                                        "content": p_content,
                                        "scope": "neo4j-pasal",
                                    })
                                    seen_chunk_ids.add(neo4j_id)
                            for ayat in detail.get("ayats", []):
                                a_content = ayat.get("content", "")
                                a_name = ayat.get("name", "")
                                p_name = ayat.get("pasal_name", "")
                                neo4j_id = f"neo4j-{did}-{p_name}-{a_name}"
                                if a_content and len(a_content) > 20 and neo4j_id not in seen_chunk_ids:
                                    doc_chunks.append({
                                        "id": neo4j_id,
                                        "doc_id": did,
                                        "article_id": f"{p_name} {a_name}",
                                        "content": a_content,
                                        "scope": "neo4j-ayat",
                                    })
                                    seen_chunk_ids.add(neo4j_id)
                        except Exception:
                            pass

                    # Determine source label
                    if did in regex_doc_ids:
                        src = "Regex (Explicit)"
                    elif did in graph_doc_ids:
                        src = "GraphRAG (LLM Catalog)"
                    elif did in graph_expanded_ids:
                        src = "GraphRAG (2-hop)"
                    else:
                        src = "VDB (Semantic)"
                    context_docs[did] = {"source": src, "chunks": doc_chunks}

                # G4. Neo4j direct neighbors as supplementary
                related_doc_ids: list[str] = []
                if neo4j_ok:
                    for did in primary_doc_ids[:3]:
                        related = neo4j_client.get_related_documents(did, limit=2)
                        for rdoc in related:
                            rdid = rdoc.get("doc_id", "")
                            if rdid and rdid not in context_docs and rdid not in related_doc_ids:
                                related_doc_ids.append(rdid)

                for rdid in related_doc_ids:
                    chunks = pinecone_client.fetch_by_doc_id(rdid, top_k=30)
                    if chunks:
                        context_docs[rdid] = {"source": "Neo4j (Related)", "chunks": chunks}

                st.session_state.search_context_docs = context_docs
                status_g.update(label=f"📖 **Konten** — {len(context_docs)} dokumen dikumpulkan", state="complete", expanded=False)

            # ── Phase H: Context Building & Answer ───────────────────
            with progress_container.status("💡 **Menghasilkan jawaban dengan GPT …**", expanded=True) as status_h:
                # Build relationship context from Neo4j graph edges
                relationship_context = ""
                edges_data = {"nodes": [], "edges": []}
                if neo4j_ok:
                    try:
                        all_doc_ids = list(context_docs.keys())
                        edges_data = neo4j_client.get_edges_between(all_doc_ids)
                        rel_lines = []
                        for edge in edges_data.get("edges", []):
                            src = edge.get("source_id", "")
                            tgt = edge.get("target_id", "")
                            rel_type = edge.get("type", "")
                            if src and tgt and rel_type:
                                rel_lines.append(f"- {src} --[{rel_type}]--> {tgt}")
                        if rel_lines:
                            relationship_context = "\n".join(rel_lines)
                    except Exception:
                        pass
                st.session_state.search_edges = edges_data

                # G.5: Round-robin interleave ensures every doc is
                # represented in the LLM context (max 40 chunks / 16k chars)

                llm_chunks = _build_interleaved_context(
                    primary_doc_ids=primary_doc_ids,
                    related_doc_ids=related_doc_ids,
                    context_docs=context_docs,
                    max_chunks=40,
                    max_chars=16000,
                )

                answer = llm_stance.ask_about_documents(
                    query, llm_chunks,
                    relationship_context=relationship_context,
                )
                st.session_state.search_answer = answer
                status_h.update(label="💡 **Jawaban** — selesai", state="complete", expanded=False)

        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.search_answer = None
            st.session_state.search_edges = None

    # Display answer
    if st.session_state.search_answer:
        section_divider("Jawaban")
        st.markdown(st.session_state.search_answer)


# ==============================================================================
# TAB 2: Browse Graph
# ==============================================================================
with tab_browse:
    if not neo4j_ok:
        st.error("Neo4j tidak terhubung. Periksa konfigurasi.")
    else:
        section_divider("Hierarki Regulasi Indonesia")

        with st.spinner("Memuat graph ..."):
            overview = neo4j_client.get_graph_overview()

        if overview["nodes"]:
            # Stat pills
            st.markdown(
                f'<div style="display:flex;gap:10px;margin:0.5rem 0 1rem;">'
                f'<span class="stat-pill"><span class="stat-pill-count">{len(overview["nodes"])}</span> dokumen</span>'
                f'<span class="stat-pill"><span class="stat-pill-count">{len(overview["edges"])}</span> relasi</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Legend
            st.markdown(
                '<div style="display:flex;gap:24px;align-items:center;margin-bottom:12px;'
                'font-size:0.78rem;color:#64748b;">'
                '  <span style="display:flex;align-items:center;gap:6px;">'
                '    <svg width="32" height="2"><line x1="0" y1="1" x2="32" y2="1"'
                '     stroke="#2563eb" stroke-width="2"/></svg>'
                '    Mengutip (CITES)'
                '  </span>'
                '  <span style="display:flex;align-items:center;gap:6px;">'
                '    <svg width="32" height="2"><line x1="0" y1="1" x2="32" y2="1"'
                '     stroke="#94a3b8" stroke-width="2" stroke-dasharray="4,3"/></svg>'
                '    Hierarki (HIGHER)'
                '  </span>'
                '  <span style="display:flex;align-items:center;gap:6px;">'
                '    <span style="font-size:0.7rem;">&#8593;</span>'
                '    Lebih tinggi kedudukannya'
                '  </span>'
                '  <span style="display:flex;align-items:center;gap:6px;">'
                '    <span style="font-size:0.7rem;">&#8594;</span>'
                '    Lebih baru (tahun)'
                '  </span>'
                '</div>',
                unsafe_allow_html=True,
            )

            st.markdown('<div class="graph-container">', unsafe_allow_html=True)
            selected = graph_viz.render_document_graph(
                doc_nodes=overview["nodes"],
                doc_edges=overview["edges"],
                height=650,
            )
            st.markdown('</div>', unsafe_allow_html=True)

            if selected:
                st.session_state.selected_node = selected

            # Show selected node detail
            if st.session_state.selected_node:
                sel_id = st.session_state.selected_node
                section_divider(f"Detail: {graph_viz._get_short_label(sel_id)}")
                detail = neo4j_client.get_document_detail(sel_id)
                if detail and detail.get("document"):
                    doc = detail["document"]
                    if doc.get("judul"):
                        st.markdown(f"**{doc['judul']}**")
                    st.caption(f"ID: {sel_id}")
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
# TAB 4: Kausalitas — Corporate Dashboard
# ==============================================================================
with tab_kausalitas:
    import altair as alt

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
        # --- Methodology ---
        with st.expander("Metodologi Klasifikasi", expanded=False):
            st.markdown("""
Setiap pasangan dokumen yang terhubung di Neo4j (melalui relasi **CITES** atau **HIGHER**)
dianalisis oleh LLM (GPT-4.1) menggunakan kerangka hukum tata negara Indonesia.

| Label | Deskripsi | Kriteria |
|-------|-----------|----------|
| **ENTAILMENT** | Regulasi saling mendukung / komplementer | Hubungan delegasi/atribusi, konsistensi substansial, keselarasan asas (UU &rarr; PP &rarr; Permen &rarr; SK) |
| **CONTRADICTION** | Regulasi bertentangan / disharmoni | Benturan kewenangan, pertentangan hak & kewajiban, inkonsistensi terminologi, pelanggaran hierarki (Lex Superior / Lex Specialis / Lex Posterior) |
| **NEUTRAL** | Tidak ada hubungan substantif | Yurisdiksi terpisah, substansi eksklusif, tidak ada persinggungan normatif |

Setiap klasifikasi **wajib menyebutkan Pasal dan Ayat spesifik** dari masing-masing dokumen
sebagai dasar penentuan label.
            """)

        df_kaus = pd.read_csv(_kausalitas_result_path)

        # --- Compute counts ---
        n_entailment = int((df_kaus["Kausalitas"] == "ENTAILMENT").sum())
        n_contradiction = int((df_kaus["Kausalitas"] == "CONTRADICTION").sum())
        n_neutral = int((df_kaus["Kausalitas"] == "NEUTRAL").sum())
        n_error = int((df_kaus["Kausalitas"] == "Error").sum())
        n_total = len(df_kaus)

        # --- KPI Cards (HTML) ---
        section_divider("Ringkasan")
        st.markdown(f"""
        <div class="kpi-row">
            <div class="kpi-card kpi-entailment">
                <div class="kpi-label">Entailment</div>
                <div class="kpi-value">{n_entailment}</div>
            </div>
            <div class="kpi-card kpi-contradiction">
                <div class="kpi-label">Contradiction</div>
                <div class="kpi-value">{n_contradiction}</div>
            </div>
            <div class="kpi-card kpi-neutral">
                <div class="kpi-label">Neutral</div>
                <div class="kpi-value">{n_neutral}</div>
            </div>
            <div class="kpi-card kpi-total">
                <div class="kpi-label">Total Pasangan</div>
                <div class="kpi-value">{n_total}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- Distribution Chart (Altair) ---
        _label_order = ["CONTRADICTION", "ENTAILMENT", "NEUTRAL"]
        _label_colors = ["#dc2626", "#059669", "#94a3b8"]
        _chart_data = pd.DataFrame({
            "Label": ["CONTRADICTION", "ENTAILMENT", "NEUTRAL"],
            "Count": [n_contradiction, n_entailment, n_neutral],
        })

        _bar_chart = (
            alt.Chart(_chart_data)
            .mark_bar(cornerRadiusEnd=6, height=28)
            .encode(
                x=alt.X("Count:Q", title="Jumlah Pasangan", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("Label:N", title=None, sort=_label_order,
                         axis=alt.Axis(labelFontSize=12, labelFontWeight="bold")),
                color=alt.Color("Label:N", scale=alt.Scale(
                    domain=_label_order, range=_label_colors
                ), legend=None),
                tooltip=[alt.Tooltip("Label:N"), alt.Tooltip("Count:Q", title="Jumlah")],
            )
            .properties(height=140)
            .configure_view(strokeWidth=0)
            .configure_axis(grid=False)
        )
        st.altair_chart(_bar_chart, use_container_width=True)

        # --- Filter Bar ---
        section_divider("Hasil Analisis")

        _filter_col1, _filter_col2 = st.columns([1, 2])
        with _filter_col1:
            _label_options = ["Semua", "ENTAILMENT", "CONTRADICTION", "NEUTRAL", "Error"]
            _selected_label = st.selectbox("Filter label", _label_options, index=0, label_visibility="collapsed")
        with _filter_col2:
            _search_query = st.text_input("Cari dokumen", placeholder="Ketik nama dokumen ...", label_visibility="collapsed")

        # Apply filters
        df_display = df_kaus.copy()
        if _selected_label != "Semua":
            df_display = df_display[df_display["Kausalitas"] == _selected_label]
        if _search_query.strip():
            _q = _search_query.strip().lower()
            df_display = df_display[
                df_display["Dokumen_Sumber"].str.lower().str.contains(_q, na=False)
                | df_display["Dokumen_Pembanding"].str.lower().str.contains(_q, na=False)
            ]

        st.caption(f"Menampilkan {len(df_display)} dari {n_total} pasangan dokumen")

        # --- Color-coded table ---
        def _color_kausalitas(val):
            _colors = {
                "CONTRADICTION": "background-color: #fee2e2; color: #dc2626; font-weight: 600;",
                "ENTAILMENT": "background-color: #d1fae5; color: #059669; font-weight: 600;",
                "NEUTRAL": "background-color: #f1f5f9; color: #64748b; font-weight: 600;",
                "Error": "background-color: #fef3c7; color: #d97706; font-weight: 600;",
            }
            return _colors.get(val, "")

        # Sort: CONTRADICTION first, then ENTAILMENT, then NEUTRAL, then Error
        _sort_map = {"CONTRADICTION": 0, "ENTAILMENT": 1, "NEUTRAL": 2, "Error": 3}
        df_display = df_display.copy()
        df_display["_sort"] = df_display["Kausalitas"].map(_sort_map).fillna(9)
        df_display = df_display.sort_values("_sort").drop(columns=["_sort"])

        df_show = df_display.rename(columns={
            "Dokumen_Sumber": "Dokumen Sumber",
            "Dokumen_Pembanding": "Dokumen Pembanding",
            "Tipe_Relasi": "Tipe Relasi",
        })

        styled = df_show.style.map(_color_kausalitas, subset=["Kausalitas"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # --- Download filtered CSV ---
        _csv_export = df_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download hasil (CSV)",
            data=_csv_export,
            file_name="kausalitas_filtered.csv",
            mime="text/csv",
        )

        # --- Detail cards with reasoning + Pasal highlighting ---
        section_divider("Detail Alasan per Pasangan")

        _badge_html_map = {
            "ENTAILMENT": '<span class="stance-supports">ENTAILMENT</span>',
            "CONTRADICTION": '<span class="stance-contradicts">CONTRADICTION</span>',
            "NEUTRAL": '<span class="stance-neutral">NEUTRAL</span>',
            "Error": '<span class="stance-neutral">ERROR</span>',
        }

        def _highlight_pasal(text: str) -> str:
            """Wrap Pasal/Ayat references in <mark> tags for visual emphasis."""
            if not isinstance(text, str):
                return str(text)
            return re.sub(
                r'(Pasal\s+\d+[A-Za-z]*(?:\s+ayat\s*\(\d+\))?)',
                r'<mark class="pasal-ref">\1</mark>',
                text,
                flags=re.IGNORECASE,
            )

        for _idx, _row in df_display.iterrows():
            _label = _row["Kausalitas"]
            _src = _row["Dokumen_Sumber"]
            _tgt = _row["Dokumen_Pembanding"]
            _rel = _row.get("Tipe_Relasi", "")
            _alasan = _row.get("Alasan", "")

            with st.expander(f"{_label}  |  {_src}  \u2192  {_tgt}", expanded=False):
                _badge = _badge_html_map.get(_label, _badge_html_map["Error"])
                st.markdown(
                    f'<div class="detail-card">'
                    f'  <div class="detail-card-header">'
                    f'    <span class="detail-card-docs">{_src}</span>'
                    f'    <span class="detail-card-arrow">\u2192</span>'
                    f'    <span class="detail-card-docs">{_tgt}</span>'
                    f'    {_badge}'
                    f'    <span class="detail-card-rel">{_rel}</span>'
                    f'  </div>'
                    f'  <div class="detail-card-reason">{_highlight_pasal(_alasan)}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# -- Footer -------------------------------------------------------------------
st.markdown(
    '<div class="app-footer">'
    'GraphRAG &mdash; Legal Document Relationship Explorer'
    ' &nbsp;&middot;&nbsp; Neo4j + Pinecone + HuggingFace + OpenRouter'
    '</div>',
    unsafe_allow_html=True,
)
