"""AutoSpec AI — An Evidence-Grounded Automotive Specification Intelligence Dashboard."""

from __future__ import annotations

import os
import time
from pathlib import Path

import streamlit as st

# Configure Streamlit page settings
st.set_page_config(
    page_title="AutoSpec AI — Automotive Specification Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Engineering CSS
st.markdown(
    """
    <style>
    /* Global Styles & Variables */
    :root {
        --primary-accent: #0284c7;
        --secondary-accent: #0ea5e9;
        --bg-surface: #0f172a;
        --card-bg: rgba(30, 41, 59, 0.7);
        --border-color: rgba(51, 65, 85, 0.8);
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(51, 65, 85, 0.8);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .spec-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }

    .badge-verified {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-ambiguous {
        background-color: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid #f59e0b;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-unsupported {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-method {
        background-color: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        border: 1px solid #6366f1;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 8px;
    }

    .disclaimer-banner {
        background: rgba(234, 179, 8, 0.1);
        border-left: 4px solid #eab308;
        padding: 12px 16px;
        border-radius: 4px;
        color: #fef08a;
        font-size: 0.88rem;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Imports after page config
from autospec.config import settings
from autospec.pipeline import AutoSpecPipeline
from autospec.schemas import VerificationStatus


@st.cache_resource(show_spinner="Initializing AutoSpec Intelligence Engine...")
def load_pipeline() -> AutoSpecPipeline:
    """Initialize shared pipeline singleton."""
    pipeline = AutoSpecPipeline()
    # Check if pre-cached index exists
    index_dir = settings.get_index_dir()
    manifests = list(index_dir.glob("*_manifest.json"))
    if manifests:
        manifests.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        fp = manifests[0].name.replace("_manifest.json", "")
        if pipeline.vector_store.is_fresh(fp):
            pipeline.vector_store.load_index(fp)
            pipeline.all_spec_chunks = pipeline.vector_store.chunks
            bm25_p = index_dir / f"{fp}.bm25"
            if bm25_p.exists():
                pipeline.keyword_retriever.load(bm25_p, pipeline.all_spec_chunks)
    return pipeline


pipeline = load_pipeline()

# Session State for tracking query history and stats
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "verified_extractions" not in st.session_state:
    st.session_state.verified_extractions = 0
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "query_results" not in st.session_state:
    st.session_state.query_results = None

# Sidebar Navigation
with st.sidebar:
    st.markdown("## 🚗 AutoSpec AI")
    st.markdown("Evidence-Grounded Technical Intelligence")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "🔍 Ask AutoSpec",
            "📑 Specification Explorer",
            "📂 Document Management",
            "🎯 Quality & Evaluation",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    doc_info = pipeline.get_document_info()
    if doc_info:
        st.success(f"**Indexed Document**:\n`{doc_info.filename}`")
        st.caption(f"Pages: {doc_info.total_pages} | Chunks: {doc_info.indexed_chunks}")
        st.caption(f"Fingerprint: `{doc_info.fingerprint[:12]}...`")
    else:
        st.warning("No document currently indexed. Please load a manual in Document Management.")

    st.markdown("### 🤖 Provider Mode")
    if settings.llm_provider == "none":
        st.info("Active: **Deterministic Fallback (100% Offline)**")
    else:
        st.info(f"Active: **{settings.llm_provider.upper()} ({settings.llm_model})**")

    st.markdown("---")
    st.caption("AutoSpec AI v0.1.0 • Grounded Automotive RAG")


# ==============================================================================
# VIEW 1: DASHBOARD
# ==============================================================================
if page == "📊 Dashboard":
    st.markdown("<div class='main-title'>AutoSpec Intelligence Dashboard</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Real-time specification extraction, verification metrics, and manual status</div>",
        unsafe_allow_html=True,
    )

    # Safety disclaimer
    st.markdown(
        """
        <div class='disclaimer-banner'>
            ⚠️ <b>Professional Workshop Safety Notice</b>: All extracted torque values, fluid capacities, 
            and clearances are evidence-grounded directly from workshop manuals. Verify against original physical manuals 
            prior to vehicle service or assembly.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # High-level metrics row
    doc_info = pipeline.get_document_info()
    col1, col2, col3, col4 = st.columns(4)

    total_chunks = len(pipeline.all_spec_chunks)
    total_pages = doc_info.total_pages if doc_info else 0
    doc_name = doc_info.filename if doc_info else "None Loaded"
    freshness = doc_info.freshness if doc_info else "Not Indexed"

    with col1:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-val'>{total_pages}</div>
                <div class='metric-lbl'>Manual Pages</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-val'>{total_chunks}</div>
                <div class='metric-lbl'>Indexed Chunks</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-val'>{st.session_state.query_count}</div>
                <div class='metric-lbl'>Queries Executed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-val'>{st.session_state.verified_extractions}</div>
                <div class='metric-lbl'>Verified Specs Found</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 📋 Document & Index Status")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Active Service Manual")
        st.write(f"- **Filename**: `{doc_name}`")
        st.write(f"- **Fingerprint (SHA-256)**: `{doc_info.fingerprint if doc_info else 'N/A'}`")
        st.write(f"- **Index Freshness**: `{freshness}`")
        st.write(f"- **Created At**: `{doc_info.created_at if doc_info else 'N/A'}`")

    with c2:
        st.markdown("#### Engine & Search Configuration")
        st.write(f"- **Embedding Model**: `{settings.embedding_model}` (384-d)")
        st.write(f"- **Hybrid Fusion**: Reciprocal Rank Fusion ($k={settings.rrf_k}$)")
        st.write(f"- **Lexical Retrieval**: BM25Plus (Automotive Tokenizer)")
        st.write(f"- **Extraction Engine**: `{settings.llm_provider}` (Fallback active)")

    st.markdown("### 🛠️ Quick Action Shortcuts")
    s1, s2, s3 = st.columns(3)
    with s1:
        if st.button("🔍 Run Test Query: Front Caliper Anchor Bolts", use_container_width=True):
            st.session_state.selected_example = "Torque for front brake caliper anchor plate bolts"
            st.info("Switch to 'Ask AutoSpec' in the sidebar to view the results!")
    with s2:
        if st.button("🔍 Run Test Query: Rear Guide Pin Bolts", use_container_width=True):
            st.session_state.selected_example = "Torque for rear brake caliper guide pin bolts"
            st.info("Switch to 'Ask AutoSpec' in the sidebar to view the results!")
    with s3:
        if st.button("🔍 Run Test Query: Brake Fluid Capacity", use_container_width=True):
            st.session_state.selected_example = "What is the brake fluid fill capacity?"
            st.info("Switch to 'Ask AutoSpec' in the sidebar to view the results!")


# ==============================================================================
# VIEW 2: ASK AUTOSPEC
# ==============================================================================
elif page == "🔍 Ask AutoSpec":
    st.markdown("<div class='main-title'>Ask AutoSpec</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Ask natural language questions to retrieve verified technical specifications</div>",
        unsafe_allow_html=True,
    )

    # Example question chips
    st.markdown("**Suggested Engineering Questions:**")
    ex_cols = st.columns(3)
    examples = [
        "Torque for front brake caliper anchor plate bolts",
        "Torque for front brake caliper guide pin bolts",
        "Torque for rear brake caliper guide pin bolts",
        "What is the brake fluid fill capacity?",
        "Brake disc minimum thickness for front brakes",
        "Differential backlash between ring gear teeth and pinion teeth",
    ]

    selected_query = ""
    for idx, ex in enumerate(examples):
        with ex_cols[idx % 3]:
            if st.button(f"📌 {ex}", key=f"chip_{idx}", use_container_width=True):
                selected_query = ex

    # Search bar input
    default_text = selected_query or getattr(st.session_state, "selected_example", "")
    query_input = st.text_input(
        "Enter technical question or component name:",
        value=default_text,
        placeholder="e.g. Torque for brake caliper anchor plate bolts, or brake fluid capacity",
    )

    col_btn, col_k, col_sub = st.columns([2, 1, 1.5])
    with col_btn:
        search_clicked = st.button("🚀 Retrieve & Extract Specifications", type="primary", use_container_width=True)
    with col_k:
        top_k = st.slider("Top Results", min_value=1, max_value=10, value=3)
    with col_sub:
        sub_filter = st.selectbox(
            "Filter Subsystem",
            ["All Subsystems", "front_brakes", "rear_brakes", "front_suspension_rwd", "rear_axle_88"],
            index=0,
        )

    effective_sub = None if sub_filter == "All Subsystems" else sub_filter

    if search_clicked and query_input.strip():
        st.session_state.last_query = query_input.strip()
        with st.spinner("Executing hybrid retrieval and grounding verification..."):
            t0 = time.perf_counter()
            resp = pipeline.query(query_input.strip(), top_k=top_k, subsystem_filter=effective_sub)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            st.session_state.query_results = resp
            st.session_state.query_count += 1

    # Display results
    if st.session_state.query_results:
        res = st.session_state.query_results
        st.markdown(f"#### Results for: *\"{res.query}\"*")
        st.caption(f"Processed in **{res.processing_time_ms:.1f} ms** • Engine: `{res.extraction_method}`")

        if not res.results:
            st.warning("🔍 **No verified specification found.** The manual does not contain an unambiguous specification matching your query.")
        else:
            for s in res.results:
                if s.verification_status == VerificationStatus.VERIFIED.value:
                    st.session_state.verified_extractions += 1

                badge_class = (
                    "badge-verified"
                    if s.verification_status == "verified"
                    else ("badge-ambiguous" if s.verification_status == "ambiguous" else "badge-unsupported")
                )
                alt_txt = f" / {s.alternate_value} {s.alternate_unit}" if s.alternate_value else ""

                st.markdown(
                    f"""
                    <div class='spec-card'>
                        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
                            <h3 style='margin: 0; color: #f8fafc;'>🔧 {s.component}</h3>
                            <div>
                                <span class='{badge_class}'>● {s.verification_status.upper()}</span>
                                <span class='badge-method'>{s.extraction_method}</span>
                            </div>
                        </div>
                        <div style='font-size: 1.4rem; font-weight: 700; color: #38bdf8; margin: 10px 0;'>
                            {s.value} {s.unit} <span style='font-size: 1.05rem; color: #94a3b8; font-weight: 400;'>{alt_txt}</span>
                        </div>
                        <div style='font-size: 0.9rem; color: #cbd5e1; margin-bottom: 12px;'>
                            📍 <b>{s.section}</b> (Subsystem: <code>{s.subsystem or 'general'}</code>) • Page <b>{s.page_number}</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander(f"📖 View Source Evidence Excerpt (Page {s.page_number})"):
                    st.markdown(f"**Verbatim Source Quote:**\n> \"{s.source_quote}\"")
                    if s.verification_notes:
                        st.info(f"**Grounding Verification Diagnostic:** {s.verification_notes}")
                    if s.part_number:
                        st.write(f"- **Part Number**: `{s.part_number}`")
                    if s.material:
                        st.write(f"- **Specification Standard**: `{s.material}`")
                    st.caption(f"Source Document: {s.source_document} | Retrieval Score: {s.retrieval_score:.4f}")

# Fallback view placeholders for pages 3, 4, 5 (to be populated in Commit 16)
elif page == "📑 Specification Explorer":
    st.info("Specification Explorer view is initialized. Full interactive table and export controls in next commit.")
elif page == "📂 Document Management":
    st.info("Document Management view is initialized. Local path and upload controls in next commit.")
elif page == "🎯 Quality & Evaluation":
    st.info("Quality & Evaluation view is initialized. Benchmark report in next commit.")
