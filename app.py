"""
app.py
────────────────────────────────────────────────────────────────────────────────
Streamlit Dashboard — Confidence-Aware Adaptive RAG System
────────────────────────────────────────────────────────────────────────────────
Run with:
    streamlit run app.py

Author : Your Name
Paper  : Confidence-Aware Adaptive Retrieval-Augmented Generation
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import streamlit as st

# ── Page config (MUST be first st call) ─────────────────────────────────────
st.set_page_config(
    page_title="CA-RAG Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Make project root importable ─────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.config_loader import load_config, get_retrieval_config
from utils.logger import logger

# ════════════════════════════════════════════════════════════════════════════
# Custom CSS — Premium AI Product Aesthetic
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ──────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
* { font-family: 'Inter', system-ui, -apple-system, sans-serif !important; }

/* ── App Shell ────────────────────────────────────────────────────────── */
.stApp {
    background: #080c14;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(99,102,241,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(16,185,129,0.06) 0%, transparent 50%);
    color: #e2e8f0;
    min-height: 100vh;
}

/* ── Sidebar ──────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: rgba(10, 14, 26, 0.95) !important;
    border-right: 1px solid rgba(99,102,241,0.15) !important;
    backdrop-filter: blur(20px);
}
section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

/* ── Sidebar brand strip ──────────────────────────────────────────────── */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 16px;
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(16,185,129,0.08));
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    margin-bottom: 20px;
}
.sidebar-brand-icon { font-size: 1.6rem; }
.sidebar-brand-text { font-size: 0.85rem; font-weight: 700; color: #a5b4fc; letter-spacing: 0.02em; }
.sidebar-brand-sub  { font-size: 0.68rem; color: #64748b; }

/* ── Section label ────────────────────────────────────────────────────── */
.section-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4b5563;
    margin: 20px 0 8px;
}

/* ── Hero Header ──────────────────────────────────────────────────────── */
.hero-wrap {
    padding: 36px 0 20px;
    position: relative;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.15;
    background: linear-gradient(135deg, #818cf8 0%, #34d399 60%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 10px;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.72rem;
    font-weight: 600;
    color: #818cf8;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 14px;
}
.hero-sub {
    color: #64748b;
    font-size: 1.02rem;
    line-height: 1.6;
    max-width: 680px;
    margin-bottom: 0;
}
.hero-divider {
    height: 1px;
    background: linear-gradient(90deg, rgba(99,102,241,0.3), rgba(16,185,129,0.2), transparent);
    margin: 24px 0;
    border: none;
}

/* ── Tabs ─────────────────────────────────────────────────────────────── */
div[data-testid="stTabs"] button {
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: #4b5563 !important;
    padding: 10px 20px !important;
    border-radius: 8px 8px 0 0 !important;
    transition: color 0.2s, background 0.2s !important;
}
div[data-testid="stTabs"] button:hover { color: #94a3b8 !important; }
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #818cf8 !important;
    background: rgba(99,102,241,0.07) !important;
}

/* ── Cards ────────────────────────────────────────────────────────────── */
.rag-card {
    background: rgba(15, 20, 35, 0.8);
    border: 1px solid rgba(99,102,241,0.12);
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 14px;
    backdrop-filter: blur(10px);
    transition: border-color 0.25s, box-shadow 0.25s;
}
.rag-card:hover {
    border-color: rgba(99,102,241,0.25);
    box-shadow: 0 0 24px rgba(99,102,241,0.07);
}
.rag-card-success {
    border-left: 3px solid #34d399;
    background: linear-gradient(135deg, rgba(16,185,129,0.06) 0%, rgba(15,20,35,0.8) 60%);
}
.rag-card-warning {
    border-left: 3px solid #fbbf24;
    background: linear-gradient(135deg, rgba(251,191,36,0.05) 0%, rgba(15,20,35,0.8) 60%);
}
.rag-card-danger {
    border-left: 3px solid #f87171;
    background: linear-gradient(135deg, rgba(248,113,113,0.05) 0%, rgba(15,20,35,0.8) 60%);
}
.rag-card-info {
    border-left: 3px solid #818cf8;
    background: linear-gradient(135deg, rgba(99,102,241,0.07) 0%, rgba(15,20,35,0.8) 60%);
}

/* ── Round badge ──────────────────────────────────────────────────────── */
.round-badge {
    display: inline-flex;
    align-items: center;
    background: rgba(99,102,241,0.2);
    color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 999px;
    padding: 2px 12px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-right: 10px;
}

/* ── Action badge ─────────────────────────────────────────────────────── */
.action-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    color: #fff;
    border-radius: 999px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}

/* ── Signal meters ────────────────────────────────────────────────────── */
.signal-block {
    background: rgba(10,14,26,0.6);
    border: 1px solid rgba(99,102,241,0.1);
    border-radius: 12px;
    padding: 14px 16px;
    text-align: center;
    transition: border-color 0.2s;
}
.signal-block:hover { border-color: rgba(99,102,241,0.25); }
.signal-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #4b5563;
    margin-bottom: 6px;
}
.signal-value { font-size: 1.75rem; font-weight: 800; font-variant-numeric: tabular-nums; line-height: 1; }
.signal-ok   { color: #34d399; }
.signal-warn { color: #fbbf24; }
.signal-bad  { color: #f87171; }

/* ── Progress bar ─────────────────────────────────────────────────────── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #6366f1, #34d399) !important;
    border-radius: 999px !important;
}

/* ── Answer box ───────────────────────────────────────────────────────── */
.answer-box {
    background: linear-gradient(135deg,
        rgba(16,185,129,0.07) 0%,
        rgba(99,102,241,0.06) 50%,
        rgba(10,14,26,0.9) 100%);
    border: 1px solid rgba(52,211,153,0.25);
    border-radius: 16px;
    padding: 24px 28px;
    font-size: 1.06rem;
    line-height: 1.75;
    color: #e2e8f0;
    position: relative;
    box-shadow: 0 0 40px rgba(52,211,153,0.06);
}
.answer-box::before {
    content: '"';
    position: absolute;
    top: -12px; left: 20px;
    font-size: 3rem;
    color: rgba(52,211,153,0.3);
    font-family: Georgia, serif;
    line-height: 1;
}

/* ── Summary metrics ──────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: rgba(15, 20, 35, 0.8) !important;
    border: 1px solid rgba(99,102,241,0.14) !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    backdrop-filter: blur(10px) !important;
    transition: border-color 0.2s !important;
}
[data-testid="metric-container"]:hover {
    border-color: rgba(99,102,241,0.3) !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.45rem !important;
    font-weight: 800 !important;
    color: #e2e8f0 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #4b5563 !important;
}

/* ── Text input / area ────────────────────────────────────────────────── */
.stTextInput input,
.stTextArea textarea {
    background: rgba(15,20,35,0.9) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: rgba(99,102,241,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
    outline: none !important;
}
.stTextInput input[type="password"] { font-family: 'JetBrains Mono', monospace !important; }

/* ── Selects ──────────────────────────────────────────────────────────── */
.stSelectbox > div > div {
    background: rgba(15,20,35,0.9) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ── Sliders ──────────────────────────────────────────────────────────── */
.stSlider > div > div > div > div { background: #6366f1 !important; }

/* ── Run button ───────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 13px 28px !important;
    font-weight: 700 !important;
    font-size: 0.96rem !important;
    letter-spacing: 0.02em !important;
    transition: transform 0.15s, box-shadow 0.15s, filter 0.15s !important;
    box-shadow: 0 4px 24px rgba(99,102,241,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(99,102,241,0.4) !important;
    filter: brightness(1.08) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Expanders ────────────────────────────────────────────────────────── */
details {
    background: rgba(10,14,26,0.6) !important;
    border: 1px solid rgba(99,102,241,0.12) !important;
    border-radius: 10px !important;
    padding: 2px 12px !important;
    margin-bottom: 8px !important;
}
summary {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    cursor: pointer;
}

/* ── Sample question buttons ──────────────────────────────────────────── */
div[data-testid="stExpanderDetails"] .stButton > button {
    background: rgba(99,102,241,0.08) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    padding: 8px 14px !important;
    text-align: left !important;
    box-shadow: none !important;
    transition: all 0.15s !important;
    margin-bottom: 4px !important;
}
div[data-testid="stExpanderDetails"] .stButton > button:hover {
    background: rgba(99,102,241,0.16) !important;
    color: #e2e8f0 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Info / warning / error ───────────────────────────────────────────── */
.stAlert {
    border-radius: 12px !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    background: rgba(15,20,35,0.8) !important;
}

/* ── Hide Streamlit chrome ────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; }

/* ── Scrollbar ────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.25); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.45); }
</style>
""", unsafe_allow_html=True)



# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def signal_color(val: float, threshold: float = 0.5) -> str:
    if val >= 0.7:   return "signal-ok"
    if val >= threshold: return "signal-warn"
    return "signal-bad"


def conf_color_card(c: float) -> str:
    if c >= 0.7: return "rag-card-success"
    if c >= 0.5: return "rag-card-warning"
    return "rag-card-danger"


def action_badge_html(action: str) -> str:
    color_map = {
        "expand_k"  : "#1f6feb",
        "reformulate": "#388bfd",
        "resample"  : "#8957e5",
        "stop"      : "#3fb950",
    }
    labels = {
        "expand_k"  : "📚 Expanded k",
        "reformulate": "✏️ Reformulated Query",
        "resample"  : "🔄 Re-sampled",
        "stop"      : "✅ Converged",
    }
    color = color_map.get(action, "#58a6ff")
    label = labels.get(action, action)
    return f'<span class="action-badge" style="background:{color}">{label}</span>'


# ════════════════════════════════════════════════════════════════════════════
# System initialisation (cached to survive reruns)
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="🔧 Loading retrieval indexes…")
def load_system(groq_key: str, model_backend: str, model_id: str,
                tau: float, max_rounds: int, top_k: int):
    """Load all heavy objects once and cache."""

    os.environ["GROQ_API_KEY"] = groq_key

    # ── Config ──────────────────────────────────────────────────────────
    cfg = load_config(str(ROOT / "configs" / "config.yaml"))
    cfg["generation"]["model"]    = model_backend
    cfg["generation"]["model_id"] = model_id
    cfg["adaptive"]["threshold_tau"] = tau
    cfg["adaptive"]["max_rounds"]    = max_rounds
    cfg["retrieval"]["top_k"]        = top_k

    r_cfg = get_retrieval_config(cfg)

    # ── Dense retriever ──────────────────────────────────────────────────
    from retrieval.dense_retriever  import DenseRetriever
    from retrieval.bm25_retriever   import BM25Retriever
    from retrieval.hybrid_retriever import HybridRetriever
    from generation.llm_wrapper     import get_llm_backend
    from generation.generator       import RAGGenerator
    from adaptive.controller        import AdaptiveController

    dense = DenseRetriever(
        model_name = r_cfg.get("model", "all-mpnet-base-v2"),
        device     = r_cfg.get("device", "cpu"),
    )
    dense.load_index(r_cfg["index_path"])
    dense.load_corpus_map(r_cfg["id_map_path"])

    bm25 = None
    retriever = dense
    if r_cfg.get("build_bm25", True) and Path(r_cfg["bm25_path"]).exists():
        bm25 = BM25Retriever()
        bm25.load(r_cfg["bm25_path"])
        retriever = HybridRetriever(
            dense  = dense,
            bm25   = bm25,
            method = r_cfg.get("hybrid_method", "rrf"),
        )

    llm  = get_llm_backend(cfg)
    gen  = RAGGenerator(llm=llm, config=cfg)
    ctrl = AdaptiveController(retriever=retriever, generator=gen, config=cfg)

    return ctrl, cfg


# ════════════════════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # Brand strip
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">🧠</div>
        <div>
            <div class="sidebar-brand-text">CA-RAG System</div>
            <div class="sidebar-brand-sub">Journal Research Project · v1.0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">🔐 Authentication</div>', unsafe_allow_html=True)
    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Free key from https://console.groq.com",
        label_visibility="collapsed",
        placeholder="gsk_••••••••••••••••••••••",
    )

    st.markdown('<div class="section-label">🤖 Model Settings</div>', unsafe_allow_html=True)
    model_backend = st.selectbox(
        "LLM Backend",
        options=["groq", "mock"],
        index=0,
        help="'groq' = real LLM, 'mock' = simulator (no key needed)",
    )

    model_id = st.selectbox(
        "Model",
        options=[
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        index=0,
        disabled=(model_backend == "mock"),
    )

    st.markdown('<div class="section-label">🎛️ Adaptive Parameters</div>', unsafe_allow_html=True)
    tau        = st.slider("Confidence Threshold (τ)", 0.3, 0.95, 0.70, 0.05)
    max_rounds = st.slider("Max Retrieval Rounds",     1,   5,    3,    1)
    top_k      = st.slider("Initial Top-K Chunks",    3,   15,   5,    1)

    st.markdown('<div class="section-label">📂 Corpus</div>', unsafe_allow_html=True)
    stats_path = ROOT / "data" / "corpus_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        c1s, c2s = st.columns(2)
        c1s.metric("Chunks",   f"{stats.get('total_chunks', 0):,}")
        c2s.metric("Sources",  f"{stats.get('unique_source_docs', 0):,}")
        st.metric("Avg Words", f"{stats.get('avg_chunk_words', 0)}")
    else:
        st.warning("corpus_stats.json not found.")


# ════════════════════════════════════════════════════════════════════════════
# Header
# ════════════════════════════════════════════════════════════════════════════

# Hero
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">⚡ Adaptive · Self-Correcting · Confidence-Aware</div>
    <div class="hero-title">Confidence-Aware<br>Adaptive RAG</div>
    <div class="hero-sub">
        Diagnostic retrieval-augmented generation that reasons about its own uncertainty.
        Watch the system diagnose failure modes and self-correct in real time.
    </div>
</div>
<hr class="hero-divider">
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# Tabs
# ════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs(["🔍  Live Query", "📊  History", "📂  Past Results"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Query
# ════════════════════════════════════════════════════════════════════════════

with tab1:
    # ── Sample queries ───────────────────────────────────────────────────
    SAMPLES = [
        "Were Scott Derrickson and Ed Wood of the same nationality?",
        "What government position was held by the woman who portrayed Corliss Archer?",
        "The arena where the Lewiston Maineiacs played can seat how many people?",
        "Are Local H and For Against both from the United States?",
        "Who is older, Annie Morton or Terry Richardson?",
    ]
    with st.expander("💡 Sample HotpotQA questions", expanded=False):
        for s in SAMPLES:
            if st.button(s, key=f"btn_{s[:20]}"):
                st.session_state["prefill"] = s

    query = st.text_area(
        "Enter your question:",
        value=st.session_state.get("prefill", ""),
        height=80,
        placeholder="e.g. Who was older, Einstein or Bohr?",
        key="query_box",
    )

    run_btn = st.button("🚀 Run Adaptive RAG", use_container_width=True)

    # ── Prerequisite checks ──────────────────────────────────────────────
    if run_btn:
        if not query.strip():
            st.warning("Please enter a question first.")
            st.stop()

        if model_backend == "groq" and not groq_key.strip():
            st.error("⚠️ Please enter your Groq API Key in the sidebar, or switch to 'mock' backend.")
            st.stop()

        index_ok = (ROOT / "data" / "faiss_index.bin").exists()
        if not index_ok:
            st.error("❌ FAISS index not found. Run `python step2_indexing.py` first.")
            st.stop()

        # ── Load system ──────────────────────────────────────────────────
        with st.spinner("🔧 Initialising system…"):
            ctrl, cfg = load_system(
                groq_key       = groq_key,
                model_backend  = model_backend,
                model_id       = model_id,
                tau            = tau,
                max_rounds     = max_rounds,
                top_k          = top_k,
            )

        st.markdown("---")
        st.markdown(f"**📨 Query:** `{query}`")

        # ── Progress placeholder ─────────────────────────────────────────
        prog_bar  = st.progress(0, text="Starting adaptive loop…")
        log_area  = st.empty()
        rounds_ph = []

        # ── Monkey-patch the controller to yield round-by-round results ──*
        result = {"done": False, "data": None}

        # We run the adaptive loop and intercept per-round state
        from confidence.signals   import compute_Sr, compute_Sl, compute_Sc
        from confidence.calibrator import Calibrator

        calibrator = Calibrator(cfg)
        r_cfg_live  = get_retrieval_config(cfg)
        a_cfg_live  = cfg.get("adaptive", {})

        current_q   = query
        k           = top_k
        max_r       = max_rounds
        epsilon     = a_cfg_live.get("early_stop_epsilon", 0.01)
        delta_k     = a_cfg_live.get("delta_k", 5)

        t_sr  = a_cfg_live.get("threshold_sr", 0.5)
        t_sl  = a_cfg_live.get("threshold_sl", 0.5)
        t_sc  = a_cfg_live.get("threshold_sc", 0.5)

        all_rounds  = []
        prev_c      = -1.0
        final_round = {}

        for r in range(1, max_r + 1):
            prog_bar.progress(r / max_r, text=f"Round {r} / {max_r}  — retrieving…")

            # 1. Retrieve
            docs, scores = ctrl.retriever.retrieve(current_q, top_k=k)

            # 2. Generate
            gen_out = ctrl.generator.generate_answer(query, docs)
            samples = gen_out["all_samples"]
            answers = [s["answer"] for s in samples]

            # 3. Signals
            Sr = compute_Sr(scores)
            Sl = compute_Sl(samples)
            Sc = compute_Sc(answers, ctrl.retriever.dense.model)
            z, C = calibrator.get_calibrated_confidence(Sr, Sl, Sc)

            # 4. Diagnose
            if Sr < t_sr:
                action = "expand_k"
            elif Sl < t_sl:
                action = "reformulate"
            elif Sc < t_sc:
                action = "resample"
            else:
                action = "stop"

            rdata = {
                "round": r, "k": k, "query_used": current_q,
                "Sr": Sr, "Sl": Sl, "Sc": Sc, "z": z, "C": C,
                "action": action,
                "answer": gen_out["final_answer"],
                "docs"  : docs,
            }
            all_rounds.append(rdata)
            final_round = rdata

            # ── Render round card ─────────────────────────────────────────
            card_cls = conf_color_card(C)
            prog_bar.progress(r / max_r, text=f"Round {r} done — C = {C:.3f}")

            with st.container():
                st.markdown(
                    f'<div class="rag-card {card_cls}">'
                    f'<span class="round-badge">Round {r}</span> '
                    f'k={k} &nbsp;|&nbsp; '
                    f'<b>C = {C:.3f}</b> &nbsp;'
                    f'{action_badge_html(action)}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                c1, c2, c3, c4 = st.columns(4)
                def _sig_metric(col, label, val, thr):
                    cls = signal_color(val, thr)
                    col.markdown(
                        f'<div class="signal-label">{label}</div>'
                        f'<div class="signal-value {cls}">{val:.3f}</div>',
                        unsafe_allow_html=True,
                    )
                    col.progress(min(val, 1.0))

                _sig_metric(c1, "📡 Retrieval  Sᵣ", Sr, t_sr)
                _sig_metric(c2, "🤖 LLM Conf.  Sₗ", Sl, t_sl)
                _sig_metric(c3, "🔄 Consistency Sc", Sc, t_sc)
                _sig_metric(c4, "🎯 Calibrated  C",  C,  tau)

                if docs:
                    with st.expander(f"📄 Retrieved Docs (k={k})", expanded=False):
                        for i, doc in enumerate(docs[:3], 1):
                            st.markdown(
                                f"**[Doc {i}]** `{doc.get('source','?')}` — "
                                f"{doc.get('text','')[:300]}…"
                            )

            # ── Stop conditions ───────────────────────────────────────────
            if C >= tau:
                break
            if r > 1 and abs(C - prev_c) < epsilon:
                break
            if r == max_r:
                break

            # ── Adapt ─────────────────────────────────────────────────────
            if action == "expand_k":
                k += delta_k
            elif action == "reformulate":
                current_q = ctrl.reformulate_query(query, gen_out["context_used"])
            elif action == "resample":
                k += delta_k

            prev_c = C

        prog_bar.progress(1.0, text="✅ Done!")

        # ── Final Answer ─────────────────────────────────────────────────
        st.markdown("---")
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown("### 💡 Final Answer")
            st.markdown(
                f'<div class="answer-box">{final_round.get("answer", "—")}</div>',
                unsafe_allow_html=True,
            )
        with col_b:
            st.markdown("### 📈 Summary")
            conf_f = final_round.get("C", 0.0)
            conf_emoji = "🟢" if conf_f >= tau else "🟡" if conf_f >= 0.5 else "🔴"
            st.metric("Final Confidence", f"{conf_emoji} {conf_f:.3f}")
            st.metric("Total Rounds",     len(all_rounds))
            st.metric("Final k",          final_round.get("k", top_k))

        # ── Confidence evolution chart ────────────────────────────────────
        if len(all_rounds) > 1:
            import json
            st.markdown("### 📊 Confidence Evolution Across Rounds")
            try:
                import plotly.graph_objects as go
                rounds_x = [r["round"] for r in all_rounds]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=rounds_x, y=[r["C"]  for r in all_rounds], name="C (Calibrated)",  line=dict(color="#58a6ff", width=3), mode="lines+markers"))
                fig.add_trace(go.Scatter(x=rounds_x, y=[r["Sr"] for r in all_rounds], name="Sᵣ (Retrieval)",  line=dict(color="#f0883e", width=2, dash="dot")))
                fig.add_trace(go.Scatter(x=rounds_x, y=[r["Sl"] for r in all_rounds], name="Sₗ (LLM)",        line=dict(color="#d2a8ff", width=2, dash="dot")))
                fig.add_trace(go.Scatter(x=rounds_x, y=[r["Sc"] for r in all_rounds], name="Sc (Consistency)",line=dict(color="#3fb950", width=2, dash="dot")))
                fig.add_hline(y=tau, line_dash="dash", line_color="#f85149", annotation_text=f"τ={tau}", annotation_position="bottom right")
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0d1117",
                    plot_bgcolor="#0d1117",
                    font=dict(color="#e6edf3"),
                    legend=dict(bgcolor="#161b22"),
                    xaxis=dict(title="Round", gridcolor="#21262d"),
                    yaxis=dict(title="Score", range=[0, 1.05], gridcolor="#21262d"),
                    height=340,
                    margin=dict(l=0, r=0, t=20, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.info("Install plotly for charts: `pip install plotly`")

        # ── Store in history ──────────────────────────────────────────────
        if "history" not in st.session_state:
            st.session_state["history"] = []
        st.session_state["history"].append({
            "query"  : query,
            "answer" : final_round.get("answer", ""),
            "conf"   : final_round.get("C", 0.0),
            "rounds" : len(all_rounds),
        })
        st.session_state.pop("prefill", None)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Session History
# ════════════════════════════════════════════════════════════════════════════

with tab2:
    history = st.session_state.get("history", [])
    if not history:
        st.info("No queries yet this session. Run a query first.")
    else:
        st.markdown(f"**{len(history)} queries this session**")
        for i, h in enumerate(reversed(history), 1):
            c = h["conf"]
            color = "#3fb950" if c >= tau else "#d29922" if c >= 0.5 else "#f85149"
            st.markdown(
                f'<div class="rag-card">'
                f'<b>#{len(history)-i+1}</b> — {h["query"]}<br>'
                f'<small style="color:#8b949e">Rounds: {h["rounds"]} &nbsp;|&nbsp; '
                f'Confidence: <span style="color:{color}">{c:.3f}</span></small>'
                f'</div>',
                unsafe_allow_html=True,
            )
            with st.expander("Answer", expanded=False):
                st.write(h["answer"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Past Results (adaptive_results.json)
# ════════════════════════════════════════════════════════════════════════════

with tab3:
    results_path = ROOT / "data" / "adaptive_results.json"
    if not results_path.exists():
        st.info("No saved results found. Run `python step4_adaptive_rag.py` first.")
    else:
        data = json.loads(results_path.read_text(encoding="utf-8"))
        st.markdown(f"**{len(data)} saved results** from `adaptive_results.json`")

        for i, item in enumerate(data):
            conf = item.get("final_confidence", 0.0)
            rds  = item.get("total_rounds", 1)
            cc   = "#3fb950" if conf >= 0.7 else "#d29922" if conf >= 0.5 else "#f85149"

            with st.expander(f"[{i+1}] {item['query'][:80]}…  — C={conf:.3f}  ({rds} round{'s' if rds>1 else ''})", expanded=False):
                st.markdown(f"**📨 Query:** {item['query']}")
                st.markdown(f"**💡 Final Answer:** {item.get('final_answer', '—')}")
                st.markdown(f"**✅ Gold Answer:** `{item.get('gold_answer', '—')}`")
                st.markdown(f"**🎯 Confidence:** `{conf:.4f}`")

                # Per-round breakdown
                if "rounds" in item and len(item["rounds"]) > 1:
                    st.markdown("**Round-by-Round:**")
                    for r in item["rounds"]:
                        st.markdown(
                            f"&nbsp;&nbsp;Round {r['round']} — "
                            f"k={r['k']}  Sr={r['Sr']:.3f}  Sl={r['Sl']:.3f}  "
                            f"Sc={r['Sc']:.3f}  **C={r['C']:.3f}**"
                        )
