"""
GraphRAG -- Legal AI Chatbot
ChatGPT-style interface for Indonesian legal document analysis.
"""

import streamlit as st
import os, time, re, uuid
import atexit
from datetime import datetime

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
from utils import neo4j_client, pinecone_client, llm_stance, graph_viz
from utils.langsmith_config import init_langsmith
from utils.memory import SemanticMemory
from utils.conflict_logger import is_conflict_related_question, append_conflict_rows, clear_conflict_output_csv
from utils.timeline_html import build_timeline_html
from utils.benchmark_helpers import extract_doc_ids_from_question as _extract_doc_ids_from_question
from utils.logging_config import setup_logging, log_event, get_logging_config, get_log_paths

setup_logging()

# ── LangSmith tracing (graceful degradation) ─────────────────────────────────
init_langsmith()

# ── Persistent memory ────────────────────────────────────────────────────────
_MEMORY_DB = os.path.join(os.path.dirname(__file__), "graphrag_memory.db")
semantic_memory = SemanticMemory(_MEMORY_DB)

# ── LangGraph checkpointer ───────────────────────────────────────────────────
_checkpointer = None
_checkpointer_cm = None
try:
    from langgraph.checkpoint.sqlite import SqliteSaver

    _candidate = SqliteSaver.from_conn_string(_MEMORY_DB)
    if hasattr(_candidate, "__enter__") and hasattr(_candidate, "__exit__"):
        _checkpointer_cm = _candidate
        _checkpointer = _checkpointer_cm.__enter__()
    else:
        _checkpointer = _candidate
except Exception:
    try:
        from langgraph.checkpoint.memory import InMemorySaver
        _checkpointer = InMemorySaver()
    except Exception:
        _checkpointer = None


def _close_checkpointer() -> None:
    if _checkpointer_cm is not None:
        try:
            _checkpointer_cm.__exit__(None, None, None)
        except Exception:
            pass


atexit.register(_close_checkpointer)

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

    # Debug logs
    if logs:
        with st.expander("\U0001F527 Catatan Teknis"):
            for log in logs:
                st.code(log, language="bash")

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
                logs=_msg.get("logs"),
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
        _trace_id = str(uuid.uuid4())
        try:
            from utils.langgraph_agent import create_agent

            log_event(
                "graphrag.app",
                "Chat request started",
                trace_id=_trace_id,
                route="entry",
                stage="chat",
                event="query_start",
                payload={"query": prompt},
            )

            with st.status("Menganalisis pertanyaan hukum...", expanded=True) as status:
                agent = create_agent(checkpointer=_checkpointer)

                # Build initial state with memory context
                _user_ctx = semantic_memory.get_user_context_prompt()
                _init_state = {
                    "query": prompt,
                    "logs": [],
                    "narratives": [],
                    "primary_doc_ids": [],
                    "chat_history": list(st.session_state.chat_history),
                    "summary": st.session_state.summary,
                    "user_context": _user_ctx,
                }

                # Thread config for checkpointer
                _thread_config = {"configurable": {"thread_id": st.session_state.active_conv_id}}

                final_state = {
                    "logs": [], "narratives": [], "primary_doc_ids": [],
                    "context_docs": {}, "answer": "",
                }
                _seen_narr = 0

                for event in agent.stream(
                    _init_state,
                    config=_thread_config,
                ):
                    for _node, _update in event.items():
                        final_state.update(_update)
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

            # Save conflict/entailment rows for timeline visualization.
            try:
                clear_conflict_output_csv()
                if is_conflict_related_question(prompt):
                    _conflict_result = llm_stance.detect_conflict_inference(prompt, _clean_answer)
                else:
                    _conflict_result = {
                        "is_conflict": False,
                        "label": "NO_CONFLICT",
                        "reason": "non_conflict_query_guardrail",
                        "confidence": 1.0,
                    }

                _primary_ids = final_state.get("primary_doc_ids", []) or []
                _context_ids = list((final_state.get("context_docs", {}) or {}).keys())
                _query_ids = list(_extract_doc_ids_from_question(prompt or ""))
                _answer_ids = list(_extract_doc_ids_from_question(_clean_answer or ""))
                _text_ids = list(dict.fromkeys([*(_query_ids), *(_answer_ids)]))
                _all_ids = list(dict.fromkeys([*(_primary_ids), *(_context_ids), *(_text_ids)]))

                _grounded_ids = list(dict.fromkeys([*(_primary_ids), *(_context_ids)]))
                if len(_grounded_ids) >= 2:
                    _paired_ids = _grounded_ids
                elif len(_text_ids) >= 2:
                    _paired_ids = _text_ids
                else:
                    _paired_ids = _all_ids

                _saved_rows = append_conflict_rows(
                    conflict_result=_conflict_result,
                    primary_doc_ids=_paired_ids,
                    relationship_context=final_state.get("relationship_context", ""),
                    question=prompt,
                    reasoning=_conflict_result.get("reason", ""),
                )
            except Exception:
                _saved_rows = 0

            # Render
            _new_idx = len(st.session_state.messages)
            _render_assistant_msg(
                content=_clean_answer,
                doc_ids=_cited_ids if _cited_ids else None,
                latency=_latency,
                logs=final_state.get("logs"),
                msg_idx=_new_idx,
            )

            if _saved_rows:
                st.caption(f"Tersimpan {_saved_rows} relasi ke output/conflict/visualize_potential_conflict.csv")

            _out_csv = os.path.join(os.path.dirname(__file__), "output", "conflict", "visualize_potential_conflict.csv")
            if os.path.isfile(_out_csv):
                _timeline = build_timeline_html(_out_csv)
                if _timeline:
                    _html, _height = _timeline
                    st.markdown("### Relasi Dokumen - Visualisasi")
                    import streamlit.components.v1 as components
                    components.html(_html, height=_height, scrolling=True)

            # Save to messages
            st.session_state.messages.append({
                "role": "assistant",
                "content": _clean_answer,
                "doc_ids": _cited_ids,
                "latency": _latency,
                "logs": final_state.get("logs", []),
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

            log_event(
                "graphrag.app",
                "Chat request completed",
                trace_id=_trace_id,
                route=final_state.get("route", "unknown"),
                stage="chat",
                event="query_complete",
                payload={
                    "saved_conflict_rows": _saved_rows,
                    "log_paths": get_log_paths(),
                    "verbose_logging": get_logging_config().get("verbose", False),
                },
            )

        except ConnectionError:
            _err_msg = "Koneksi ke server gagal. Periksa koneksi internet Anda."
            st.error(_err_msg)
            st.session_state.messages.append({"role": "assistant", "content": _err_msg})
        except TimeoutError:
            _err_msg = "Waktu permintaan habis. Server mungkin sedang sibuk."
            st.error(_err_msg)
            st.session_state.messages.append({"role": "assistant", "content": _err_msg})
        except Exception as e:
            log_event(
                "graphrag.error",
                "Chat request failed",
                trace_id=_trace_id,
                route="chat",
                stage="chat",
                event="query_error",
                payload={"error": str(e)},
                level=40,
            )
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
