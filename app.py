"""
GraphRAG -- Legal AI Chatbot
ChatGPT-style interface for Indonesian legal document analysis.
"""

import streamlit as st
import os, time, re, uuid, logging
from datetime import datetime
from typing import Any, cast

# ── Streamlit Cloud: inject secrets into env vars ─────────────────────────────
try:
    if hasattr(st, "secrets") and len(st.secrets):
        for key, value in st.secrets.items():
            if isinstance(value, str):
                os.environ.setdefault(key, value)
except Exception:
    pass

from dotenv import load_dotenv
load_dotenv()

import boto3
from shared.debug_logger import new_trace_id, log_verbose_event
from utils import neo4j_client, pinecone_client, llm_stance, graph_viz
from utils.langsmith_config import init_langsmith
from utils.memory import SemanticMemory
from utils.conflict_logger import (
    is_conflict_related_question, append_conflict_rows, clear_conflict_output_csv,
    VISUALIZE_CSV,
)
from utils.timeline_html import build_timeline_html
from utils.benchmark_helpers import extract_doc_ids_from_question as _extract_doc_ids_from_question

# ── LangSmith tracing (graceful degradation) ─────────────────────────────────
init_langsmith()

# ── File logger ───────────────────────────────────────────────────────────────
_LOG_DIR = os.path.join(os.path.dirname(__file__), "output", "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_PATH = os.path.join(_LOG_DIR, "app.log")
_file_logger = logging.getLogger("graphrag.app_file")
if not _file_logger.handlers:
    _file_logger.setLevel(logging.INFO)
    _fh = logging.FileHandler(_LOG_PATH, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    _file_logger.addHandler(_fh)
    _file_logger.propagate = False


def _write_log(lines: list[str] | None, query: str = "", latency: float = 0.0):
    """Append agent debug logs to output/logs/app.log."""
    if not lines:
        return
    _file_logger.info("query=%s | latency=%.1fs", query, latency)
    for line in lines:
        _file_logger.info("  %s", line)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}

# ── Persistent memory ────────────────────────────────────────────────────────
_MEMORY_DB = os.path.join(os.path.dirname(__file__), "graphrag_memory.db")
semantic_memory = SemanticMemory(_MEMORY_DB)

# ── LangGraph checkpointer ───────────────────────────────────────────────────
import sys
_checkpointer = None

# Detect deployment environment (Streamlit Cloud, Docker, Production)
_is_deployed = any([
    "STREAMLIT_SERVER_RUNDIR" in os.environ,
    os.environ.get("ENVIRONMENT") == "production",
    os.path.exists("/.dockerenv"),
])

if _is_deployed:
    # 🌐 Streamlit Cloud: Use InMemorySaver (no file I/O)
    try:
        from langgraph.checkpoint.memory import InMemorySaver
        _checkpointer = InMemorySaver()
        print("[CHECKPOINTER] ✅ InMemorySaver initialized for deployed environment", file=sys.stderr)
    except ImportError:
        print("[CHECKPOINTER] ⚠️  InMemorySaver not available. Checkpointer disabled.", file=sys.stderr)
        _checkpointer = None
    except Exception as e:
        print(f"[CHECKPOINTER] ⚠️  Failed to init InMemorySaver: {e}. Checkpointer disabled.", file=sys.stderr)
        _checkpointer = None
else:
    # 💻 Local development: Use SqliteSaver (persistent)
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        import sqlite3
        _CHECKPOINT_DB = os.path.join(os.path.dirname(__file__), "checkpointer.db")
        _conn = sqlite3.connect(_CHECKPOINT_DB, check_same_thread=False, timeout=30)
        _conn.execute("PRAGMA journal_mode=WAL")
        _checkpointer = SqliteSaver(_conn)
        print(f"[CHECKPOINTER] ✅ SqliteSaver initialized at {_CHECKPOINT_DB}", file=sys.stderr)
    except ImportError:
        print("[CHECKPOINTER] ⚠️  SqliteSaver not available. Checkpointer disabled.", file=sys.stderr)
        _checkpointer = None
    except Exception as e:
        print(f"[CHECKPOINTER] ⚠️  Failed to init SqliteSaver: {e}. Checkpointer disabled.", file=sys.stderr)
        _checkpointer = None

print(f"[CHECKPOINTER] Final: {type(_checkpointer).__name__ if _checkpointer else 'None'}", file=sys.stderr)

# -- Page Config ---------------------------------------------------------------
st.set_page_config(
    page_title="GraphRAG",
    page_icon="\u2696\uFE0F",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
_CSS = """
<style>
    /* ── Font ─────────────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* ── Hide Streamlit chrome ────────────────────────────────────────────── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent !important; }

    /* ── Main content — centered like ChatGPT ────────────────────────────── */
    .block-container {
        max-width: 820px !important;
        margin: 0 auto !important;
        padding: 1rem 1.5rem 6rem !important;
    }

    /* ── Sidebar — dark navy (ChatGPT-style) ─────────────────────────────── */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        background-color: #0f172a !important;
    }
    [data-testid="stSidebarContent"] {
        background-color: #0f172a !important;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stCaption p {
        color: #94a3b8 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #1e293b !important;
    }

    /* Sidebar — New Chat button (primary) */
    section[data-testid="stSidebar"] .stButton > button[kind="primary"],
    section[data-testid="stSidebar"] .stButton > button[data-testid*="primary"] {
        background: linear-gradient(135deg, #1e3a5f, #2563eb) !important;
        border: none !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 0.55rem 1rem !important;
        letter-spacing: 0.2px;
        transition: all 0.3s ease;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
        box-shadow: 0 4px 20px rgba(37,99,235,0.35) !important;
        transform: translateY(-1px);
    }

    /* Sidebar — History buttons */
    section[data-testid="stSidebar"] .stButton > button:not([kind="primary"]) {
        background: transparent !important;
        border: none !important;
        color: #cbd5e1 !important;
        border-radius: 8px !important;
        font-size: 0.82rem !important;
        font-weight: 400 !important;
        text-align: left !important;
        padding: 8px 12px !important;
        transition: all 0.2s ease;
        justify-content: flex-start !important;
    }
    section[data-testid="stSidebar"] .stButton > button:not([kind="primary"]):hover {
        background: #1e293b !important;
        color: #f1f5f9 !important;
    }

    /* ── Chat greeting ────────────────────────────────────────────────────── */
    .chat-greeting {
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        min-height: 55vh; text-align: center;
        padding: 2rem 1rem;
    }
    .greeting-icon {
        font-size: 3rem; margin-bottom: 1rem; opacity: 0.7;
    }
    .greeting-title {
        font-size: 1.8rem; font-weight: 800; color: #1a1a2e;
        letter-spacing: -0.5px; margin-bottom: 0.3rem;
    }
    .greeting-title span {
        background: linear-gradient(135deg, #2563eb, #818cf8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .greeting-subtitle {
        font-size: 0.95rem; color: #64748b; max-width: 420px;
        line-height: 1.6;
    }

    /* ── Chat messages ────────────────────────────────────────────────────── */
    [data-testid="stChatMessage"] {
        padding: 1rem 0 !important;
        border-bottom: 1px solid #f1f5f9;
    }
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3,
    [data-testid="stChatMessage"] h4,
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li {
        text-align: left !important;
    }
    [data-testid="stChatMessage"]:last-of-type {
        border-bottom: none;
    }

    /* ── Chat input ──────────────────────────────────────────────────────── */
    [data-testid="stChatInput"] {
        border-radius: 16px !important;
    }
    [data-testid="stChatInput"] textarea {
        border-radius: 16px !important;
        font-size: 0.92rem !important;
        font-family: "Inter", sans-serif !important;
    }

    /* ── Connection dots in sidebar ──────────────────────────────────────── */
    .sidebar-conn {
        display: flex; gap: 14px; padding: 4px 2px;
    }
    .sidebar-conn-item {
        display: inline-flex; align-items: center; gap: 6px;
        font-size: 0.7rem; color: rgba(255,255,255,0.5);
        font-weight: 500; letter-spacing: 0.2px;
    }
    .sidebar-dot {
        width: 6px; height: 6px; border-radius: 50%;
    }
    .sidebar-dot.ok { background: #34d399; box-shadow: 0 0 5px rgba(52,211,153,0.5); }
    .sidebar-dot.err { background: #f87171; box-shadow: 0 0 5px rgba(248,113,113,0.5); }

    /* ── Doc cards ────────────────────────────────────────────────────────── */
    .doc-card {
        border: 1px solid #e2e8f0; border-radius: 10px;
        padding: 14px 16px; margin-bottom: 8px;
        background: #ffffff; position: relative; overflow: hidden;
        transition: all 0.2s ease;
    }
    .doc-card::before {
        content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
        border-radius: 3px 0 0 3px;
    }
    .doc-card.doc-uu::before { background: #2563eb; }
    .doc-card.doc-pp::before { background: #059669; }
    .doc-card.doc-permen::before { background: #d97706; }
    .doc-card.doc-perpres::before { background: #7c3aed; }
    .doc-card.doc-other::before { background: #64748b; }
    .doc-card:hover { border-color: #bfdbfe; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
    .doc-card-header {
        display: flex; align-items: center; gap: 10px; margin-bottom: 8px;
    }
    .doc-card-icon {
        width: 32px; height: 32px; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.65rem; font-weight: 700; color: #fff; flex-shrink: 0;
    }
    .doc-card-icon.icon-uu { background: #2563eb; }
    .doc-card-icon.icon-pp { background: #059669; }
    .doc-card-icon.icon-permen { background: #d97706; }
    .doc-card-icon.icon-perpres { background: #7c3aed; }
    .doc-card-icon.icon-other { background: #64748b; }
    .doc-card-title {
        font-size: 0.84rem; font-weight: 600; color: #1a1a2e; line-height: 1.3;
    }
    .doc-card-actions { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .doc-card-actions a, .doc-card-actions button {
        display: inline-flex; align-items: center; gap: 3px;
        font-size: 0.73rem; font-weight: 500; color: #2563eb;
        text-decoration: none; padding: 3px 8px;
        border: 1px solid #e2e8f0; border-radius: 6px;
        background: transparent; cursor: pointer; font-family: inherit;
        transition: all 0.2s ease;
    }
    .doc-card-actions a:hover, .doc-card-actions button:hover {
        background: #eff6ff; border-color: #2563eb;
    }
    .badge-unavailable {
        display: inline-flex; align-items: center; gap: 3px;
        font-size: 0.68rem; font-weight: 500; padding: 3px 8px;
        border-radius: 6px; background: #fef3c7; color: #92400e; border: 1px solid #fde68a;
    }

    /* ── Expanders ────────────────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        font-weight: 500 !important; font-size: 0.85rem !important;
        color: #475569 !important; background: #f8fafc !important;
        border-radius: 8px !important; border: 1px solid #e2e8f0 !important;
    }

    /* ── Scrollbar ────────────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

    /* ── Disclaimer ──────────────────────────────────────────────────────── */
    .legal-disclaimer {
        font-size: 0.72rem; color: #94a3b8; font-style: italic;
        margin-top: 8px; padding: 6px 12px;
        background: #f8fafc; border-radius: 8px;
        border-left: 3px solid #e2e8f0;
    }

    /* ── Dark mode ────────────────────────────────────────────────────────── */
    @media (prefers-color-scheme: dark) {
        .stApp, [data-testid="stAppViewContainer"],
        section[data-testid="stMain"] {
            background-color: #0c1322 !important;
        }
        .stMarkdown, .stMarkdown p, .stMarkdown li,
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
        .stMarkdown h4, .stMarkdown strong {
            color: #e2e8f0 !important;
        }
        .greeting-title { color: #f1f5f9; }
        .greeting-subtitle { color: #94a3b8; }
        [data-testid="stChatMessage"] { border-bottom-color: #1e293b; }
        .doc-card {
            background: #1e293b; border-color: #334155;
            color: #e2e8f0;
        }
        .doc-card:hover { border-color: #475569; }
        .doc-card-title { color: #f1f5f9; }
        .doc-card-actions a, .doc-card-actions button {
            color: #60a5fa; border-color: #334155;
        }
        .doc-card-actions a:hover, .doc-card-actions button:hover {
            background: #253449; border-color: #60a5fa;
        }
        .badge-unavailable { background: #451a03; color: #fbbf24; border-color: #78350f; }
        .streamlit-expanderHeader {
            background: #1e293b !important; color: #94a3b8 !important;
            border-color: #334155 !important;
        }
        .legal-disclaimer { background: #1e293b; border-color: #334155; color: #64748b; }
        .stCodeBlock, code { background: #1e293b !important; color: #cbd5e1 !important; }
    }

    /* ── Pasal ref highlight ─────────────────────────────────────────────── */
    mark.pasal-ref {
        background: linear-gradient(135deg, #eff6ff, #dbeafe);
        color: #1e3a5f; font-weight: 600;
        padding: 1px 6px; border-radius: 4px;
        font-size: 0.92em; border: 1px solid #bfdbfe;
    }
    @media (prefers-color-scheme: dark) {
        mark.pasal-ref {
            background: linear-gradient(135deg, #1e3a5f, #1e40af);
            color: #93c5fd; border-color: #1d4ed8;
        }
    }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_defaults = {
    "messages": [],            # current chat [{role, content, doc_ids?, latency?, logs?}]
    "active_conv_id": str(uuid.uuid4()),
    "search_context_docs": {},
    "feedback_given": set(),
    "chat_history": [],        # LangGraph conversation memory
    "summary": "",             # Condensed older conversation
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ══════════════════════════════════════════════════════════════════════════════
# CONNECTION CHECKS
# ══════════════════════════════════════════════════════════════════════════════
neo4j_ok = neo4j_client.test_connection()
pinecone_ok = pinecone_client.test_connection()


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def _s3_object_exists(bucket: str, key: str) -> bool:
    try:
        region = os.getenv("AWS_REGION", "ap-southeast-3")
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=region,
            endpoint_url=f"https://s3.{region}.amazonaws.com",
        )
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def _get_s3_presigned_url(doc_id: str, expiry: int = 3600) -> str | None:
    try:
        region = os.getenv("AWS_REGION", "ap-southeast-3")
        bucket = os.getenv("S3_BUCKET", "s3-lexport-dev-v1")
        directory = os.getenv("S3_DIRECTORY", "neo4j-dev")
        key = f"{directory}/{doc_id}.pdf"
        if not _s3_object_exists(bucket, key):
            return None
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=region,
            endpoint_url=f"https://s3.{region}.amazonaws.com",
        )
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expiry,
        )
    except Exception:
        return None


def _get_doc_type(doc_id: str) -> str:
    prefix = doc_id.split("-")[0].upper() if "-" in doc_id else ""
    if prefix == "UU":
        return "uu"
    elif prefix == "PP":
        return "pp"
    elif prefix.startswith("PERMEN"):
        return "permen"
    elif prefix in ("PERPRES", "KEPPRES"):
        return "perpres"
    return "other"


_DOC_TYPE_LABELS = {"uu": "UU", "pp": "PP", "permen": "PM", "perpres": "PR", "other": "REG"}


def _add_section_icons(md: str) -> str:
    return md


def _render_doc_card(doc_id: str):
    """Render a single document reference card."""
    dtype = _get_doc_type(doc_id)
    dlabel = _DOC_TYPE_LABELS.get(dtype, "REG")
    s3_url = _get_s3_presigned_url(doc_id)

    if s3_url:
        actions_html = f'<a href="{s3_url}" target="_blank">&#128196; Lihat PDF</a>'
    else:
        actions_html = '<span class="badge-unavailable">&#9888; Tidak tersedia di S3</span>'

    parts = doc_id.split("-")
    cite = f"{parts[0]} No. {parts[2]} Tahun {parts[3]}" if len(parts) >= 4 else doc_id
    copy_html = (
        '<button onclick="'
        f"navigator.clipboard.writeText('{cite}');"
        "this.textContent='Tersalin!';"
        "setTimeout(()=>this.textContent='Salin Sitasi',2000);"
        '">Salin Sitasi</button>'
    )

    st.markdown(
        f'<div class="doc-card doc-{dtype}">'
        f'<div class="doc-card-header">'
        f'<div class="doc-card-icon icon-{dtype}">{dlabel}</div>'
        f'<div class="doc-card-title">{doc_id}</div>'
        f"</div>"
        f'<div class="doc-card-actions">{actions_html}{copy_html}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_assistant_msg(content: str, doc_ids: list | None = None,
                          latency: float | None = None, logs: list | None = None,
                          msg_idx: int = 0):
    """Render assistant response: answer + toolbar + docs + feedback."""
    _display = _add_section_icons(content)
    st.markdown(_display)

    # Latency + disclaimer
    if latency:
        st.caption(f"\u23F1 {latency}s")

    st.markdown(
        '<div class="legal-disclaimer">'
        "Jawaban dihasilkan oleh AI berdasarkan regulasi di database. "
        "Bukan nasihat hukum resmi."
        "</div>",
        unsafe_allow_html=True,
    )

    # Doc cards
    if doc_ids:
        with st.expander(f"\U0001F4DA Dokumen Referensi ({len(doc_ids)})"):
            for did in doc_ids:
                _render_doc_card(did)

    # Feedback
    _fb_key = hash(content[:100]) if content else msg_idx
    if _fb_key not in st.session_state.feedback_given:
        _c1, _c2, _c3 = st.columns([8, 1, 1])
        with _c2:
            if st.button("\U0001F44D", key=f"fb_up_{msg_idx}"):
                st.session_state.feedback_given.add(_fb_key)
                st.toast("Terima kasih!")
                st.rerun()
        with _c3:
            if st.button("\U0001F44E", key=f"fb_dn_{msg_idx}"):
                st.session_state.feedback_given.add(_fb_key)
                st.toast("Terima kasih atas masukan Anda.")
                st.rerun()


def _save_current_conv():
    """Save current conversation title to SQLite semantic memory."""
    if not st.session_state.messages:
        return
    cid = st.session_state.active_conv_id
    first_user = next((m["content"] for m in st.session_state.messages if m["role"] == "user"), "")
    title = first_user[:60] if first_user else "Obrolan"
    semantic_memory.save_conversation_title(cid, title)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Brand
    st.markdown(
        '<div style="padding:0.75rem 0 0.5rem;">'
        '<div style="font-size:1.4rem;font-weight:800;color:#ffffff;letter-spacing:-0.3px;">'
        'Graph<span style="background:linear-gradient(135deg,#818cf8,#a78bfa);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
        'background-clip:text;">RAG</span></div>'
        '<div style="font-size:0.72rem;color:rgba(255,255,255,0.4);margin-top:2px;">'
        "Legal AI Assistant</div></div>",
        unsafe_allow_html=True,
    )

    # New Chat button
    if st.button("\u2795  Obrolan Baru", use_container_width=True, type="primary"):
        _save_current_conv()
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.summary = ""
        st.session_state.active_conv_id = str(uuid.uuid4())
        st.rerun()

    st.divider()

    # Connection status
    _neo_cls = "ok" if neo4j_ok else "err"
    _neo_lbl = "Neo4j" if neo4j_ok else "Neo4j \u2717"
    _pine_cls = "ok" if pinecone_ok else "err"
    _pine_lbl = "Pinecone" if pinecone_ok else "Pinecone \u2717"
    st.markdown(
        f'<div class="sidebar-conn">'
        f'<div class="sidebar-conn-item"><span class="sidebar-dot {_neo_cls}"></span>{_neo_lbl}</div>'
        f'<div class="sidebar-conn-item"><span class="sidebar-dot {_pine_cls}"></span>{_pine_lbl}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # Chat history (from SQLite)
    _saved_convs = semantic_memory.get_all_conversation_titles()
    if _saved_convs:
        st.divider()
        st.caption("RIWAYAT")
        for _conv in _saved_convs[:20]:
            _title_trunc = _conv["title"][:45] + ("..." if len(_conv["title"]) > 45 else "")
            _is_active = _conv["id"] == st.session_state.active_conv_id
            if st.button(
                _title_trunc,
                key=f"conv_{_conv['id']}",
                use_container_width=True,
                disabled=_is_active,
            ):
                _save_current_conv()
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.session_state.summary = ""
                st.session_state.active_conv_id = _conv["id"]
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — CHAT AREA
# ══════════════════════════════════════════════════════════════════════════════

# Greeting when no messages
if not st.session_state.messages:
    st.markdown(
        '<div class="chat-greeting">'
        '<div class="greeting-icon">\u2696\uFE0F</div>'
        '<div class="greeting-title">Graph<span>RAG</span></div>'
        '<div class="greeting-subtitle">'
        "Tanyakan apa saja tentang regulasi Indonesia. "
        "Saya akan menganalisis dan menjawab berdasarkan dokumen hukum yang tersedia."
        "</div></div>",
        unsafe_allow_html=True,
    )

# Render existing messages from session state
for _i, _msg in enumerate(st.session_state.messages):
    if _msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(_msg["content"])
    else:
        with st.chat_message("assistant", avatar="\u2696\uFE0F"):
            _render_assistant_msg(
                content=_msg.get("content", ""),
                doc_ids=_msg.get("doc_ids"),
                latency=_msg.get("latency"),
                msg_idx=_i,
            )

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Tanyakan sesuatu tentang regulasi..."):
    # Add & show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant", avatar="\u2696\uFE0F"):
        _t_start = time.time()
        try:
            from utils.langgraph_agent import create_agent

            with st.status("Menganalisis pertanyaan hukum...", expanded=True) as status:
                _safe_checkpointer = _checkpointer
                try:
                    from langgraph.checkpoint.base import BaseCheckpointSaver
                    if _safe_checkpointer not in (None, True, False) and not isinstance(_safe_checkpointer, BaseCheckpointSaver):
                        print(
                            f"[CHECKPOINTER] Invalid type before create_agent: {type(_safe_checkpointer).__name__}. Fallback to None.",
                            file=sys.stderr,
                        )
                        _safe_checkpointer = None
                except Exception:
                    _safe_checkpointer = None
                print(
                    f"[CHECKPOINTER] Passing into create_agent: {type(_safe_checkpointer).__name__ if _safe_checkpointer is not None else 'None'}",
                    file=sys.stderr,
                )
                agent = create_agent(checkpointer=_safe_checkpointer)

                # Build initial state with memory context
                _user_ctx = semantic_memory.get_user_context_prompt()
                _init_state = {
                    "query": prompt,
                    "logs": [],
                    "narratives": [],
                    "primary_doc_ids": [],
                    "trace_id": new_trace_id(),
                    "verbose_debug": _env_bool("GRAPHRAG_VERBOSE_DEBUG", False),
                    "chat_history": list(st.session_state.chat_history),
                    "summary": st.session_state.summary,
                    "user_context": _user_ctx,
                }
                _init_state = cast(dict[str, Any], _init_state)

                # Thread config for checkpointer
                _thread_config = {"configurable": {"thread_id": st.session_state.active_conv_id}}
                _thread_config = cast(dict[str, Any], _thread_config)

                final_state = {
                    "logs": [], "narratives": [], "primary_doc_ids": [],
                    "context_docs": {}, "answer": "",
                }
                _seen_narr = 0

                for event in agent.stream(
                    cast(Any, _init_state),
                    config=cast(Any, _thread_config),
                ):
                    for _node, _update in event.items():
                        final_state.update(_update)
                        if _init_state["verbose_debug"]:
                            log_verbose_event(
                                route=final_state.get("route", "unknown"),
                                stage="agent_stream",
                                event="node_update",
                                message=f"Node update from {_node}",
                                trace_id=_init_state["trace_id"],
                                payload={
                                    "node": _node,
                                    "route": final_state.get("route", "unknown"),
                                    "primary_doc_ids": final_state.get("primary_doc_ids", []),
                                    "log_tail": (_update.get("logs") or [])[-12:],
                                    "narrative_tail": (_update.get("narratives") or [])[-6:],
                                },
                            )
                        _narrs = _update.get("narratives", [])
                        if len(_narrs) > _seen_narr:
                            for _n in _narrs[_seen_narr:]:
                                st.markdown(f"*{_n}*")
                            _seen_narr = len(_narrs)

                status.update(label="Analisis selesai", state="complete", expanded=False)

            _latency = round(time.time() - _t_start, 1)
            _answer_raw = final_state.get("answer", "")

            # Parse DASAR_HUKUM footer
            _dasar_match = re.search(r"DASAR_HUKUM:\s*(.+)", _answer_raw)
            if _dasar_match:
                _cited_ids = [d.strip() for d in _dasar_match.group(1).split(",") if d.strip()]
                _clean_answer = _answer_raw[: _dasar_match.start()].rstrip()
            else:
                _cited_ids = []
                _clean_answer = _answer_raw

            # Render
            _new_idx = len(st.session_state.messages)
            # Write debug logs to file
            _write_log(final_state.get("logs"), query=prompt, latency=_latency)

            _render_assistant_msg(
                content=_clean_answer,
                doc_ids=_cited_ids if _cited_ids else None,
                latency=_latency,
                msg_idx=_new_idx,
            )

            # Save to messages
            st.session_state.messages.append({
                "role": "assistant",
                "content": _clean_answer,
                "doc_ids": _cited_ids,
                "latency": _latency,
            })

            # Auto-save conversation
            _save_current_conv()

            # Update in-memory chat history for next turn
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.session_state.chat_history.append({"role": "assistant", "content": _clean_answer[:500]})
            # Persist summary if agent updated it
            _new_summary = final_state.get("summary", "")
            if _new_summary:
                st.session_state.summary = _new_summary

            # ── Visualization: document relation timeline ─────────────
            try:
                clear_conflict_output_csv()

                _primary_ids = final_state.get("primary_doc_ids", []) or []
                _context_ids = list((final_state.get("context_docs", {}) or {}).keys())
                _query_ids = list(_extract_doc_ids_from_question(prompt or ""))
                _answer_ids = list(_extract_doc_ids_from_question(_clean_answer or ""))
                _text_ids_raw = list(dict.fromkeys([*_query_ids, *_answer_ids]))

                # Validate text-extracted IDs against Neo4j
                _text_ids = []
                if neo4j_ok:
                    for _did in _text_ids_raw:
                        try:
                            if neo4j_client.get_document_detail(_did):
                                _text_ids.append(_did)
                        except Exception:
                            pass

                _grounded_ids = list(dict.fromkeys([*_primary_ids, *_context_ids]))
                # Prioritize cited IDs from "Dasar Hukum" section
                if _cited_ids and len(_cited_ids) >= 2:
                    _paired_ids = list(dict.fromkeys(_cited_ids))
                elif len(_grounded_ids) >= 2:
                    _paired_ids = _grounded_ids
                elif len(_text_ids) >= 2:
                    _paired_ids = _text_ids
                else:
                    _paired_ids = list(dict.fromkeys([*_grounded_ids, *_text_ids]))

                if len(_paired_ids) >= 2:
                    if is_conflict_related_question(prompt):
                        _conflict_result = llm_stance.detect_conflict_inference(prompt, _clean_answer)
                    else:
                        _conflict_result = {
                            "is_conflict": False,
                            "label": "NO_CONFLICT",
                            "reason": "non_conflict_query_guardrail",
                            "confidence": 1.0,
                        }

                    _rel_ctx = final_state.get("relationship_context", "")
                    append_conflict_rows(
                        conflict_result=_conflict_result,
                        primary_doc_ids=_paired_ids,
                        relationship_context=_rel_ctx,
                        question=prompt,
                        reasoning=_conflict_result.get("reason", ""),
                    )

                    if os.path.isfile(VISUALIZE_CSV):
                        _viz = build_timeline_html(VISUALIZE_CSV)
                        if _viz:
                            _html, _height = _viz
                            import streamlit.components.v1 as components
                            components.html(_html, height=_height, scrolling=True)
            except Exception:
                pass

            # Log to semantic memory
            try:
                _route = final_state.get("route", "")
                semantic_memory.log_query(
                    query=prompt,
                    doc_ids=_cited_ids,
                    route=_route,
                    latency=_latency,
                )
            except Exception:
                pass

        except ConnectionError:
            _err_msg = "Koneksi ke server gagal. Periksa koneksi internet Anda."
            st.error(_err_msg)
            st.session_state.messages.append({"role": "assistant", "content": _err_msg})
        except TimeoutError:
            _err_msg = "Waktu permintaan habis. Server mungkin sedang sibuk."
            st.error(_err_msg)
            st.session_state.messages.append({"role": "assistant", "content": _err_msg})
        except Exception as e:
            _err_str = str(e).lower()
            if "neo4j" in _err_str or "database" in _err_str:
                _err_msg = f"Kesalahan database Neo4j: {e}"
            elif "pinecone" in _err_str or "vector" in _err_str:
                _err_msg = f"Kesalahan Pinecone: {e}"
            elif "openrouter" in _err_str or "api" in _err_str or "rate" in _err_str:
                _err_msg = f"Kesalahan LLM API: {e}"
            else:
                _err_msg = f"Terjadi kesalahan: {e}"
            st.error(_err_msg)
            st.session_state.messages.append({"role": "assistant", "content": _err_msg})
