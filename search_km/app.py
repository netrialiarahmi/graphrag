"""
GraphRAG -- Legal Document Relationship Explorer + Knowledge Map
Streamlit app combining search pipeline with knowledge map visualization.
"""

import streamlit as st
import os, sys

# ── Resolve project root so shared/ is importable ─────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

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

import os, re, glob
import pandas as pd
import altair as alt
import boto3
from shared import neo4j_client, pinecone_client, llm_stance
from utils import graph_viz
from utils.knowledge_graph import parse_dasar_hukum, detect_conflicts, build_answer_graph, get_level_legend, get_node_color
from streamlit_agraph import agraph, Config

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
    .brand-text { position: relative; z-index: 1; }
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
    .conn-badges { display: flex; gap: 12px; position: relative; z-index: 1; }
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
    .conn-indicator { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
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
    .stTabs [data-baseweb="tab-border"] { display: none !important; }
    .stTabs [data-baseweb="tab-highlight"] { display: none !important; }

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
    .stSelectbox > div > div { border-radius: 10px !important; }
    .stNumberInput > div > div > input { border-radius: 10px !important; }

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
        left: 0; top: 0; bottom: 0;
        width: 3px;
        background: linear-gradient(180deg, #1e3a5f, #3b82f6);
        border-radius: 3px 0 0 3px;
    }
    .result-card:hover {
        border-color: #bfdbfe;
        box-shadow: 0 4px 16px rgba(30,58,95,0.08);
        transform: translateY(-2px);
    }
    .result-card-title { font-weight: 600; font-size: 0.92rem; color: #0f172a; margin-bottom: 4px; }
    .result-card-meta { font-size: 0.75rem; color: #94a3b8; display: flex; gap: 12px; flex-wrap: wrap; }
    .result-card-meta span { display: inline-flex; align-items: center; gap: 4px; }
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
        color: #fff; font-size: 0.68rem; font-weight: 600;
        padding: 4px 14px; border-radius: 20px;
        letter-spacing: 0.5px; text-transform: uppercase;
        box-shadow: 0 2px 8px rgba(5,150,105,0.25);
    }
    .stance-contradicts {
        display: inline-block;
        background: linear-gradient(135deg, #dc2626, #f87171);
        color: #fff; font-size: 0.68rem; font-weight: 600;
        padding: 4px 14px; border-radius: 20px;
        letter-spacing: 0.5px; text-transform: uppercase;
        box-shadow: 0 2px 8px rgba(220,38,38,0.25);
    }
    .stance-neutral {
        display: inline-block;
        background: linear-gradient(135deg, #6b7280, #9ca3af);
        color: #fff; font-size: 0.68rem; font-weight: 600;
        padding: 4px 14px; border-radius: 20px;
        letter-spacing: 0.5px; text-transform: uppercase;
        box-shadow: 0 2px 8px rgba(107,114,128,0.2);
    }

    /* ── Metric Cards ────────────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8fafc, #f1f5f9);
        border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 14px 16px; transition: all 0.25s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: #bfdbfe;
        box-shadow: 0 2px 12px rgba(30,58,95,0.06);
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8; font-size: 0.7rem;
        text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600;
    }
    [data-testid="stMetricValue"] { color: #0f172a; font-weight: 700; }

    /* ── Expanders ────────────────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        font-weight: 500 !important; font-size: 0.88rem !important;
        color: #334155 !important; background: #f8fafc !important;
        border-radius: 10px !important; border: 1px solid #e2e8f0 !important;
        transition: all 0.2s ease;
    }
    .streamlit-expanderHeader:hover {
        border-color: #bfdbfe !important;
        background: #eff6ff !important;
    }

    /* ── Graph container ─────────────────────────────────────────────────── */
    .graph-container {
        border: 1px solid #e2e8f0; border-radius: 16px;
        overflow: hidden; background: #fafbfe;
        box-shadow: 0 1px 8px rgba(0,0,0,0.04);
    }

    /* ── Stat pills ──────────────────────────────────────────────────────── */
    .stat-pill {
        display: inline-flex; align-items: center; gap: 6px;
        background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 20px;
        padding: 5px 14px; font-size: 0.75rem; font-weight: 500; color: #475569;
    }
    .stat-pill-count {
        background: linear-gradient(135deg, #1e3a5f, #2563eb);
        color: #fff; font-weight: 700; font-size: 0.7rem;
        padding: 2px 8px; border-radius: 10px;
    }

    /* ── Stance Row ──────────────────────────────────────────────────────── */
    .stance-row {
        display: flex; align-items: center; gap: 16px;
        padding: 12px 16px; border: 1px solid #f1f5f9; border-radius: 10px;
        margin: 6px 0; background: #fff; transition: all 0.2s ease;
    }
    .stance-row:hover { background: #fafbfe; border-color: #e2e8f0; }
    .stance-arrow { color: #94a3b8; font-size: 0.85rem; }
    .stance-doc { font-weight: 600; font-size: 0.82rem; color: #1e293b; }
    .stance-reason { font-size: 0.78rem; color: #64748b; flex: 1; }

    /* ── Knowledge Map: Wrapper ─────────────────────────────────────────── */
    .km-wrapper {
        background: #fafbfe;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin-top: 0.5rem;
    }

    /* ── Knowledge Map: Stat Bar ──────────────────────────────────────────── */
    .km-stat-bar {
        display: flex; gap: 10px; flex-wrap: wrap;
        margin-bottom: 14px;
    }
    .km-stat-item {
        display: inline-flex; align-items: center; gap: 6px;
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 20px;
        padding: 5px 14px; font-size: 0.75rem; font-weight: 500; color: #475569;
    }
    .km-stat-count {
        font-weight: 700; font-size: 0.72rem;
        padding: 2px 8px; border-radius: 10px;
        color: #fff;
    }

    /* ── Knowledge Map: Color Legend ──────────────────────────────────────── */
    .km-color-legend {
        display: flex; gap: 10px; flex-wrap: wrap; align-items: center;
        padding: 10px 14px; background: #ffffff;
        border: 1px solid #e2e8f0; border-radius: 10px;
        margin-bottom: 10px;
    }
    .km-color-legend-title {
        font-size: 0.68rem; text-transform: uppercase; letter-spacing: 1px;
        font-weight: 700; color: #94a3b8; margin-right: 4px;
    }
    .km-color-item {
        display: inline-flex; align-items: center; gap: 5px;
        font-size: 0.72rem; color: #475569; font-weight: 500;
    }
    .km-color-swatch {
        width: 14px; height: 14px; border-radius: 3px; flex-shrink: 0;
    }

    /* ── Knowledge Map: Edge Legend ───────────────────────────────────────── */
    .km-edge-legend {
        display: flex; gap: 16px; align-items: center;
        padding: 8px 14px; background: #ffffff;
        border: 1px solid #e2e8f0; border-radius: 10px;
        margin-bottom: 12px;
    }
    .km-edge-item {
        display: inline-flex; align-items: center; gap: 6px;
        font-size: 0.72rem; color: #475569; font-weight: 500;
    }
    .km-edge-line { width: 24px; height: 0; }
    .km-edge-line.cites { border-top: 2px solid #2563eb; }
    .km-edge-line.higher { border-top: 2px dashed #94a3b8; }
    .km-edge-line.conflict { border-top: 3px solid #dc2626; }

    /* ── Knowledge Map: Doc Detail Card ──────────────────────────────────── */
    .km-doc-detail {
        border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 16px 18px; background: #ffffff;
        position: relative; overflow: hidden;
    }
    .km-doc-detail::before {
        content: "";
        position: absolute; left: 0; top: 0; bottom: 0;
        width: 4px; border-radius: 4px 0 0 4px;
    }
    .km-doc-header {
        display: flex; align-items: center; gap: 10px; margin-bottom: 8px;
    }
    .km-doc-title {
        font-weight: 700; font-size: 1rem; color: #0f172a;
    }
    .km-doc-type-badge {
        font-size: 0.65rem; font-weight: 600; color: #fff;
        padding: 3px 10px; border-radius: 12px;
        text-transform: uppercase; letter-spacing: 0.5px;
    }
    .km-doc-id {
        font-size: 0.73rem; color: #94a3b8; font-family: monospace;
        margin-bottom: 10px;
    }
    .km-meta-row {
        display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px;
    }
    .km-meta-pill {
        display: inline-flex; align-items: center; gap: 4px;
        background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 8px;
        padding: 4px 12px; font-size: 0.75rem; color: #475569;
    }
    .km-meta-pill strong { color: #1e293b; font-weight: 600; }

    /* ── Knowledge Map: Chunk Card ────────────────────────────────────────── */
    .km-chunk-card {
        border: 1px solid #e2e8f0; border-left: 3px solid #2563eb;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px; margin: 6px 0;
        background: #ffffff; font-size: 0.82rem;
        color: #334155; line-height: 1.55;
    }
    .km-chunk-scope {
        font-size: 0.68rem; font-weight: 600; color: #2563eb;
        background: #eff6ff; padding: 2px 8px; border-radius: 8px;
        display: inline-block; margin-bottom: 6px;
    }

    /* ── Knowledge Map: Relation Row ─────────────────────────────────────── */
    .km-relation-row {
        display: flex; align-items: center; gap: 8px;
        padding: 8px 12px; border: 1px solid #f1f5f9; border-radius: 8px;
        margin: 4px 0; background: #fff; transition: all 0.2s ease;
        font-size: 0.82rem;
    }
    .km-relation-row:hover { background: #fafbfe; border-color: #e2e8f0; }
    .km-relation-arrow { color: #94a3b8; font-size: 0.85rem; flex-shrink: 0; }
    .km-relation-doc { font-weight: 600; color: #1e293b; }
    .km-relation-type {
        font-size: 0.68rem; color: #64748b; background: #f1f5f9;
        padding: 2px 8px; border-radius: 10px; font-weight: 500;
    }
    .km-relation-type.conflict {
        color: #dc2626; background: #fef2f2;
    }

    /* ── Knowledge Map: Doc List Item ─────────────────────────────────────── */
    .km-doc-list-item {
        display: flex; align-items: center; gap: 10px;
        padding: 10px 14px; border: 1px solid #e2e8f0; border-radius: 10px;
        margin: 5px 0; background: #ffffff; transition: all 0.2s ease;
    }
    .km-doc-list-item:hover {
        border-color: #bfdbfe; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .km-doc-dot {
        width: 10px; height: 10px; border-radius: 3px; flex-shrink: 0;
    }
    .km-doc-list-label {
        font-weight: 600; font-size: 0.85rem; color: #0f172a; flex: 1;
    }
    .km-doc-list-year {
        font-size: 0.72rem; color: #94a3b8; font-weight: 500;
    }

    /* ── Footer ──────────────────────────────────────────────────────────── */
    .app-footer {
        text-align: center; padding: 1.5rem 0 1rem; margin-top: 3rem;
        font-size: 0.73rem; color: #94a3b8; border-top: 1px solid #e2e8f0;
        letter-spacing: 0.2px;
    }

    /* ── Scrollbar ────────────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

    /* ── Checkbox label ──────────────────────────────────────────────────── */
    .stCheckbox label span { font-size: 0.85rem !important; color: #475569 !important; }

    /* ── Spinner ─────────────────────────────────────────────────────────── */
    .stSpinner > div { border-top-color: #2563eb !important; }

    /* ── Kausalitas Dashboard ─────────────────────────────────────────────── */
    .kpi-row { display: flex; gap: 16px; margin-bottom: 1.2rem; }
    .kpi-card {
        flex: 1; background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 20px 22px;
        position: relative; overflow: hidden; transition: all 0.25s ease;
    }
    .kpi-card::before {
        content: ""; position: absolute; left: 0; top: 0; bottom: 0;
        width: 4px; border-radius: 4px 0 0 4px;
    }
    .kpi-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.06); transform: translateY(-1px); }
    .kpi-card.kpi-entailment::before { background: linear-gradient(180deg, #059669, #34d399); }
    .kpi-card.kpi-contradiction::before { background: linear-gradient(180deg, #dc2626, #f87171); }
    .kpi-card.kpi-neutral::before { background: linear-gradient(180deg, #64748b, #94a3b8); }
    .kpi-card.kpi-total::before { background: linear-gradient(180deg, #1e3a5f, #3b82f6); }
    .kpi-label {
        font-size: 0.68rem; text-transform: uppercase; letter-spacing: 1px;
        font-weight: 600; color: #94a3b8; margin-bottom: 6px;
    }
    .kpi-value { font-size: 1.8rem; font-weight: 700; color: #0f172a; line-height: 1; }
    .kpi-card.kpi-entailment .kpi-value { color: #059669; }
    .kpi-card.kpi-contradiction .kpi-value { color: #dc2626; }
    .kpi-card.kpi-neutral .kpi-value { color: #64748b; }
    .kpi-card.kpi-total .kpi-value { color: #1e3a5f; }

    mark.pasal-ref {
        background: linear-gradient(135deg, #eff6ff, #dbeafe);
        color: #1e3a5f; font-weight: 600; padding: 1px 6px;
        border-radius: 4px; font-size: 0.92em; border: 1px solid #bfdbfe;
    }

    .detail-card {
        border: 1px solid #e2e8f0; border-radius: 10px;
        padding: 16px 20px; margin-bottom: 6px; background: #fafbfe;
    }
    .detail-card-header {
        display: flex; align-items: center; gap: 10px;
        flex-wrap: wrap; margin-bottom: 10px;
    }
    .detail-card-docs { font-weight: 600; font-size: 0.85rem; color: #1e293b; }
    .detail-card-arrow { color: #94a3b8; font-size: 0.85rem; }
    .detail-card-rel {
        font-size: 0.72rem; color: #64748b; background: #f1f5f9;
        padding: 2px 10px; border-radius: 12px; border: 1px solid #e2e8f0; font-weight: 500;
    }
    .detail-card-reason {
        font-size: 0.84rem; color: #475569; line-height: 1.6;
        margin-top: 8px; padding-top: 10px; border-top: 1px solid #f1f5f9;
    }

    /* ── Status icon classes (no-emoji replacements) ──────────────────────── */
    .status-icon {
        display: inline-flex; align-items: center; justify-content: center;
        width: 20px; height: 20px; border-radius: 50%;
        font-size: 0.7rem; font-weight: 700; margin-right: 6px;
        flex-shrink: 0;
    }
    .status-icon.processing {
        background: linear-gradient(135deg, #1e3a5f, #2563eb);
        color: #fff;
    }
    .status-icon.done {
        background: linear-gradient(135deg, #059669, #34d399);
        color: #fff;
    }
    .status-icon.thought {
        background: #f1f5f9;
        color: #64748b;
        border: 1px solid #e2e8f0;
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
if "km_neo4j_edges" not in st.session_state:
    st.session_state.km_neo4j_edges = []
if "km_conflicts" not in st.session_state:
    st.session_state.km_conflicts = []
if "km_doc_ids" not in st.session_state:
    st.session_state.km_doc_ids = []
if "km_selected_node" not in st.session_state:
    st.session_state.km_selected_node = None

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
        <div class="brand-subtitle">Legal Document Relationship Explorer + Knowledge Map</div>
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
    st.markdown(
        f'<div class="section-divider"><span class="section-divider-text">{text}</span></div>',
        unsafe_allow_html=True,
    )


def render_result_card(doc_id: str, scope: str, content: str, score: float = None, article_id: str = ""):
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


@st.cache_data(ttl=300, show_spinner=False)
def _s3_presigned_url(doc_id: str) -> str | None:
    try:
        region = os.getenv("AWS_REGION", "ap-southeast-3")
        bucket = os.getenv("S3_BUCKET", "s3-lexport-dev-v1")
        directory = os.getenv("S3_DIRECTORY", "neo4j-dev")
        key = f"{directory}/{doc_id}.pdf"
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=region,
            endpoint_url=f"https://s3.{region}.amazonaws.com",
        )
        s3.head_object(Bucket=bucket, Key=key)
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=3600,
        )
    except Exception:
        return None


def _doc_short_label(doc_id: str) -> str:
    parts = doc_id.split("-")
    if len(parts) >= 4:
        return f"{parts[0]} {parts[2]}/{parts[3]}"
    return doc_id[:30]


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
    result: list[dict] = []
    total_chars = 0
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
# TAB 1: Search & Discover  +  Knowledge Map
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
            from utils.langgraph_agent import create_agent

            with progress_container.status(
                "Memutar Strategi Penelusuran Hukum...",
                expanded=True,
            ) as status:
                agent = create_agent()
                final_state = {"logs": [], "narratives": [], "primary_doc_ids": [], "context_docs": {}, "answer": ""}
                seen_narratives = 0

                for event in agent.stream({"query": query, "logs": [], "narratives": [], "primary_doc_ids": []}):
                    for node_name, state_update in event.items():
                        final_state.update(state_update)
                        curr_narr = state_update.get("narratives", [])
                        if len(curr_narr) > seen_narratives:
                            for nar in curr_narr[seen_narratives:]:
                                st.markdown(f"--- *{nar}*")
                            seen_narratives = len(curr_narr)

                status.update(
                    label="Analisis Hukum Selesai",
                    state="complete",
                    expanded=False,
                )

                st.session_state.search_doc_ids = final_state.get("primary_doc_ids", [])
                st.session_state.search_context_docs = final_state.get("context_docs", {})
                st.session_state.search_answer = final_state.get("answer", "")
                st.session_state.search_edges = {"edges": []}
                st.session_state.km_selected_node = None

                # ── Knowledge Map: parse & build ─────────────────────────────
                answer_text = st.session_state.search_answer or ""
                km_ids = parse_dasar_hukum(answer_text)
                if not km_ids:
                    km_ids = list(st.session_state.search_doc_ids)[:8]

                km_neo4j_edges = []
                if neo4j_ok and len(km_ids) >= 2:
                    try:
                        edge_data = neo4j_client.get_edges_between(km_ids)
                        km_neo4j_edges = edge_data.get("edges", [])
                    except Exception:
                        pass

                km_conflicts = []
                if answer_text and len(km_ids) >= 2:
                    km_conflicts = detect_conflicts(answer_text, km_ids)

                st.session_state.km_doc_ids = km_ids
                st.session_state.km_neo4j_edges = km_neo4j_edges
                st.session_state.km_conflicts = km_conflicts

            with st.expander("System Debug Logs", expanded=False):
                for log in final_state.get("logs", []):
                    st.code(log, language="bash")

        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.search_answer = None
            st.session_state.search_edges = None

    # ── Section 2: Display answer ─────────────────────────────────────────────
    if st.session_state.search_answer:
        section_divider("Jawaban")
        st.markdown(st.session_state.search_answer)

    # ── Section 3: Knowledge Map ──────────────────────────────────────────────
    if st.session_state.get("km_doc_ids"):
        section_divider("Peta Regulasi")

        km_ids = st.session_state.km_doc_ids
        km_neo4j_edges = st.session_state.km_neo4j_edges
        km_conflicts = st.session_state.km_conflicts

        # Stat bar
        n_cites = sum(1 for e in km_neo4j_edges if e.get("type") == "CITES")
        n_higher = sum(1 for e in km_neo4j_edges if e.get("type") == "HIGHER")
        n_conflict = len(km_conflicts)
        st.markdown(
            f'<div class="km-stat-bar">'
            f'<span class="km-stat-item"><span class="km-stat-count" style="background:#1e3a5f">{len(km_ids)}</span> Dokumen</span>'
            f'<span class="km-stat-item"><span class="km-stat-count" style="background:#2563eb">{n_cites}</span> Mengutip</span>'
            f'<span class="km-stat-item"><span class="km-stat-count" style="background:#94a3b8">{n_higher}</span> Hierarki</span>'
            f'<span class="km-stat-item"><span class="km-stat-count" style="background:#dc2626">{n_conflict}</span> Konflik</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Color-coded hierarchy legend (dynamic — only levels present)
        legend_items = get_level_legend(km_ids)
        color_html = "".join(
            f'<span class="km-color-item">'
            f'<span class="km-color-swatch" style="background:{it["color"]}"></span>'
            f'{it["name"]}'
            f'</span>'
            for it in legend_items
        )
        st.markdown(
            f'<div class="km-color-legend">'
            f'<span class="km-color-legend-title">Hierarki</span>'
            f'{color_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Edge type legend
        st.markdown(
            '<div class="km-edge-legend">'
            '<span class="km-edge-item"><span class="km-edge-line cites"></span> Mengutip</span>'
            '<span class="km-edge-item"><span class="km-edge-line higher"></span> Hierarki</span>'
            '<span class="km-edge-item"><span class="km-edge-line conflict"></span> Konflik</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        col_graph, col_detail = st.columns([3, 2])

        with col_graph:
            # Build and render graph
            nodes, edges = build_answer_graph(km_ids, km_neo4j_edges, km_conflicts)

            if nodes:
                config = Config(
                    width="100%",
                    height=550,
                    directed=True,
                    physics=False,
                    hierarchical=True,
                    direction="UD",
                    sortMethod="directed",
                    levelSeparation=150,
                    nodeSpacing=120,
                    treeSpacing=220,
                    nodeHighlightBehavior=True,
                    highlightColor="#2563eb",
                )
                selected_km_node = agraph(nodes=nodes, edges=edges, config=config)

                if selected_km_node:
                    st.session_state.km_selected_node = selected_km_node

                st.caption("Klik node untuk melihat detail dokumen, kutipan relevan, dan PDF.")
            else:
                st.info("Tidak ada dokumen untuk divisualisasikan.")

        with col_detail:
            selected = st.session_state.km_selected_node

            if selected and selected in km_ids:
                # ── Detail card with hierarchy color accent ──
                node_color = get_node_color(selected)
                label = _doc_short_label(selected)

                doc_info = {}
                try:
                    detail = neo4j_client.get_document_detail(selected)
                    doc_info = detail.get("document", {})
                except Exception:
                    pass

                jenis = doc_info.get("jenis", "-")
                tahun = doc_info.get("tahun", "-")
                nomor = doc_info.get("nomor", "-")
                judul = doc_info.get("judul", "")

                st.markdown(
                    f'<div class="km-doc-detail" style="border-left:4px solid {node_color};">'
                    f'<div class="km-doc-header">'
                    f'<span class="km-doc-title">{label}</span>'
                    f'<span class="km-doc-type-badge" style="background:{node_color}">{jenis}</span>'
                    f'</div>'
                    f'<div class="km-doc-id">{selected}</div>'
                    f'<div class="km-meta-row">'
                    f'<span class="km-meta-pill"><strong>Tahun</strong> {tahun}</span>'
                    f'<span class="km-meta-pill"><strong>Nomor</strong> {nomor}</span>'
                    f'</div>'
                    + (f'<div style="font-size:0.85rem;color:#475569;font-style:italic;line-height:1.5;">{judul}</div>' if judul else "")
                    + f'</div>',
                    unsafe_allow_html=True,
                )

                # PDF button
                url = _s3_presigned_url(selected)
                if url:
                    st.link_button("Buka PDF", url, use_container_width=True, type="primary")
                else:
                    st.caption("PDF tidak tersedia di S3")

                # ── Kutipan Relevan ──
                ctx = st.session_state.search_context_docs.get(selected, {})
                chunks = ctx.get("chunks", []) if isinstance(ctx, dict) else []
                if chunks:
                    st.markdown(f"**Kutipan Relevan** ({len(chunks)} fragmen)")
                    for i, chunk in enumerate(chunks[:3]):
                        content = chunk.get("content", "")[:350]
                        scope = chunk.get("scope", "")
                        article = chunk.get("article_id", "")
                        scope_label = article if article else (scope if scope else f"Fragmen {i+1}")
                        with st.expander(scope_label, expanded=(i == 0)):
                            st.markdown(
                                f'<div class="km-chunk-card">'
                                + (f'<span class="km-chunk-scope">{scope_label}</span>' if scope_label else "")
                                + f'<div>{content}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                # ── Relasi Hukum ──
                related = []
                for edge in km_neo4j_edges:
                    src = edge.get("source_id", "")
                    tgt = edge.get("target_id", "")
                    rel = edge.get("type", "")
                    if src == selected:
                        related.append((tgt, _doc_short_label(tgt), rel, "\u2192"))
                    elif tgt == selected:
                        related.append((src, _doc_short_label(src), rel, "\u2190"))
                for conf in km_conflicts:
                    cs = conf.get("source", "")
                    ct = conf.get("target", "")
                    cl = conf.get("label", "KONFLIK")
                    if cs == selected:
                        related.append((ct, _doc_short_label(ct), f"KONFLIK: {cl}", "\u2192"))
                    elif ct == selected:
                        related.append((cs, _doc_short_label(cs), f"KONFLIK: {cl}", "\u2190"))

                if related:
                    st.markdown(f"**Relasi Hukum** ({len(related)})")
                    for rdoc_id, rlabel, rel, direction in related:
                        is_conflict = "KONFLIK" in rel
                        type_cls = "conflict" if is_conflict else ""
                        st.markdown(
                            f'<div class="km-relation-row">'
                            f'<span class="km-relation-arrow">{direction}</span>'
                            f'<span class="km-relation-doc">{rlabel}</span>'
                            f'<span class="km-relation-type {type_cls}">{rel}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        if st.button(f"Lihat {rlabel}", key=f"nav_{rdoc_id}", use_container_width=True):
                            st.session_state.km_selected_node = rdoc_id
                            st.rerun()
            else:
                # ── No selection — compact doc list ──
                st.markdown("**Dokumen Terkait**")
                st.caption("Klik node di peta untuk melihat detail dan kutipan relevan.")
                for did in km_ids:
                    label = _doc_short_label(did)
                    color = get_node_color(did)
                    from utils.knowledge_graph import _extract_year
                    year = _extract_year(did)
                    st.markdown(
                        f'<div class="km-doc-list-item">'
                        f'<span class="km-doc-dot" style="background:{color}"></span>'
                        f'<span class="km-doc-list-label">{label}</span>'
                        + (f'<span class="km-doc-list-year">{year}</span>' if year else "")
                        + f'</div>',
                        unsafe_allow_html=True,
                    )
                    url = _s3_presigned_url(did)
                    if url:
                        st.link_button("Buka PDF", url, key=f"pdf_{did}", use_container_width=True)


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
            doc_nodes = overview.get("nodes", [])
            doc_edges = overview.get("edges", [])

        if not doc_nodes:
            st.info("Belum ada data regulasi di Neo4j.")
        else:
            # SVG Legend
            st.markdown("""
            <div style="display:flex;gap:20px;align-items:center;margin-bottom:10px">
                <div style="display:flex;align-items:center;gap:6px">
                    <svg width="30" height="10"><line x1="0" y1="5" x2="30" y2="5"
                        stroke="#2563eb" stroke-width="2"/></svg>
                    <span style="font-size:0.75rem;color:#475569">CITES</span>
                </div>
                <div style="display:flex;align-items:center;gap:6px">
                    <svg width="30" height="10"><line x1="0" y1="5" x2="30" y2="5"
                        stroke="#94a3b8" stroke-width="2" stroke-dasharray="4,3"/></svg>
                    <span style="font-size:0.75rem;color:#475569">HIGHER</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            selected = graph_viz.render_document_graph(
                doc_nodes, doc_edges,
                stance_map=st.session_state.stance_cache,
                height=700,
            )

            if selected and selected != st.session_state.selected_node:
                st.session_state.selected_node = selected

            if st.session_state.selected_node:
                node_id = st.session_state.selected_node
                section_divider(f"Detail: {node_id}")
                try:
                    detail = neo4j_client.get_document_detail(node_id)
                    doc_info = detail.get("document", {})
                    if doc_info:
                        st.markdown(f"**Judul:** {doc_info.get('judul', '-')}")
                        st.markdown(f"**Jenis:** {doc_info.get('jenis', '-')} | **Tahun:** {doc_info.get('tahun', '-')} | **Nomor:** {doc_info.get('nomor', '-')}")
                    pasals = detail.get("pasals", [])
                    if pasals:
                        with st.expander(f"Pasal ({len(pasals)})", expanded=False):
                            for p in pasals[:20]:
                                st.markdown(f"**{p.get('name', '?')}**: {p.get('content', '')[:300]}")
                except Exception as e:
                    st.warning(f"Gagal memuat detail: {e}")

            # Stats
            st.markdown(
                f'<div style="display:flex;gap:12px;margin-top:10px">'
                f'<span class="stat-pill"><span class="stat-pill-count">{len(doc_nodes)}</span> Dokumen</span>'
                f'<span class="stat-pill"><span class="stat-pill-count">{len(doc_edges)}</span> Relasi</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ==============================================================================
# TAB 3: Compare Documents (Benchmark)
# ==============================================================================
with tab_compare:
    section_divider("Benchmark Retrieval")

    _benchmark_dir = os.path.join(_PROJECT_ROOT, "output", "retrieval")
    _recap_dir = os.path.join(_benchmark_dir, "recap")

    if not os.path.isdir(_recap_dir):
        st.info("Belum ada hasil benchmark. Jalankan `run_benchmark_v6.py` terlebih dahulu.")
    else:
        csv_files = sorted(glob.glob(os.path.join(_recap_dir, "*.csv")))
        if not csv_files:
            st.info("Tidak ada file CSV benchmark ditemukan.")
        else:
            file_labels = [os.path.basename(f).replace(".csv", "") for f in csv_files]
            selected_file = st.selectbox("Pilih skenario benchmark", file_labels)
            idx = file_labels.index(selected_file)
            df = pd.read_csv(csv_files[idx])
            st.dataframe(df, use_container_width=True, hide_index=True)

            _detail_dir = os.path.join(_benchmark_dir, "detail", "csv")
            _detail_name = selected_file + "-metrics.csv"
            _detail_path = os.path.join(_detail_dir, _detail_name)
            if os.path.isfile(_detail_path):
                with st.expander("Detail per-pertanyaan"):
                    df_detail = pd.read_csv(_detail_path)
                    st.dataframe(df_detail, use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 4: Kausalitas
# ==============================================================================
with tab_kausalitas:
    section_divider("Analisis Kausalitas Regulasi")

    _kausalitas_result_path = os.path.join(_PROJECT_ROOT, "output", "kausalitas", "kausalitas_results.csv")

    if not os.path.isfile(_kausalitas_result_path):
        st.info(
            "Belum ada hasil analisis kausalitas.  "
            "Jalankan di terminal:  \n"
            "```bash\npython run_kausalitas.py\n```"
        )
    else:
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

        n_entailment = int((df_kaus["Kausalitas"] == "ENTAILMENT").sum())
        n_contradiction = int((df_kaus["Kausalitas"] == "CONTRADICTION").sum())
        n_neutral = int((df_kaus["Kausalitas"] == "NEUTRAL").sum())
        n_error = int((df_kaus["Kausalitas"] == "Error").sum())
        n_total = len(df_kaus)

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

        section_divider("Hasil Analisis")

        _filter_col1, _filter_col2 = st.columns([1, 2])
        with _filter_col1:
            _label_options = ["Semua", "ENTAILMENT", "CONTRADICTION", "NEUTRAL", "Error"]
            _selected_label = st.selectbox("Filter label", _label_options, index=0, label_visibility="collapsed")
        with _filter_col2:
            _search_query = st.text_input("Cari dokumen", placeholder="Ketik nama dokumen ...", label_visibility="collapsed")

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

        def _color_kausalitas(val):
            _colors = {
                "CONTRADICTION": "background-color: #fee2e2; color: #dc2626; font-weight: 600;",
                "ENTAILMENT": "background-color: #d1fae5; color: #059669; font-weight: 600;",
                "NEUTRAL": "background-color: #f1f5f9; color: #64748b; font-weight: 600;",
                "Error": "background-color: #fef3c7; color: #d97706; font-weight: 600;",
            }
            return _colors.get(val, "")

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

        _csv_export = df_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download hasil (CSV)",
            data=_csv_export,
            file_name="kausalitas_filtered.csv",
            mime="text/csv",
        )

        section_divider("Detail Alasan per Pasangan")

        _badge_html_map = {
            "ENTAILMENT": '<span class="stance-supports">ENTAILMENT</span>',
            "CONTRADICTION": '<span class="stance-contradicts">CONTRADICTION</span>',
            "NEUTRAL": '<span class="stance-neutral">NEUTRAL</span>',
            "Error": '<span class="stance-neutral">ERROR</span>',
        }

        def _highlight_pasal(text: str) -> str:
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
    'GraphRAG &mdash; Legal Document Relationship Explorer + Knowledge Map'
    ' &nbsp;&middot;&nbsp; Neo4j + Pinecone + HuggingFace + OpenRouter'
    '</div>',
    unsafe_allow_html=True,
)
