"""
DocAI Frontend — Streamlit dashboard for visual document extraction.

Run: streamlit run frontend/app.py --server.port 8501
"""

import streamlit as st
import requests
import json
import time
from pathlib import Path

API_BASE = "http://localhost:8000/api/v1"

# ── Page Config ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DocAI — Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #718096;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .metric-card {
        background: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a365d;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #718096;
    }
    .extraction-json {
        background: #1a202c;
        color: #68d391;
        border-radius: 8px;
        padding: 16px;
        font-family: monospace;
        font-size: 13px;
        overflow-x: auto;
    }
    .pipeline-step {
        background: #ebf8ff;
        border-left: 4px solid #3182ce;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
    }
    .warning-box {
        background: #fffbeb;
        border-left: 4px solid #d69e2e;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Settings")

    # Health check
    try:
        health = requests.get(f"{API_BASE}/health", timeout=3).json()
        st.success(f"API Connected")
        st.caption(f"OCR: {health.get('ocr_engine', 'N/A')}")
        st.caption(f"LLM: {health.get('llm_provider', 'N/A')}")
        st.caption(f"MLflow: {'Connected' if health.get('mlflow_connected') else 'Offline'}")
        st.caption(f"ChromaDB: {'Connected' if health.get('chroma_connected') else 'Offline'}")
    except Exception:
        st.error("API not running. Start with:\n`uvicorn app.main:app --reload`")

    st.divider()

    doc_type = st.selectbox(
        "Document type",
        ["unknown (auto-detect)", "invoice", "contract", "report", "receipt"],
    )
    doc_type_value = doc_type.split(" ")[0]

    store_vector = st.checkbox("Store in ChromaDB", value=True)

    prompt_version = st.text_input(
        "Prompt version (A/B test)", placeholder="e.g., v1 or v2", value=""
    )

    st.divider()
    st.markdown("### Pipeline")
    st.markdown("""
    <div class="pipeline-step">1. Upload document</div>
    <div class="pipeline-step">2. OCR text extraction</div>
    <div class="pipeline-step">3. Auto-classification</div>
    <div class="pipeline-step">4. LLM structured extraction</div>
    <div class="pipeline-step">5. ChromaDB storage</div>
    <div class="pipeline-step">6. MLflow experiment tracking</div>
    """, unsafe_allow_html=True)


# ── Main Content ─────────────────────────────────────────────────────────

st.markdown('<p class="main-header">DocAI — Document Intelligence Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload invoices, contracts, or reports → get structured JSON in seconds</p>', unsafe_allow_html=True)

# Tabs
tab_extract, tab_search, tab_monitor = st.tabs([
    "Extract Document", "Semantic Search", "Monitoring"
])


# ── Tab 1: Extract ───────────────────────────────────────────────────────

with tab_extract:
    col_upload, col_result = st.columns([1, 1])

    with col_upload:
        st.markdown("### Upload document")

        uploaded_file = st.file_uploader(
            "Choose a PDF or image file",
            type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "webp"],
            help="Supported: PDF, PNG, JPG, TIFF, BMP, WebP (max 50MB)",
        )

        if uploaded_file:
            # Show preview
            if uploaded_file.type and "image" in uploaded_file.type:
                st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
            else:
                st.info(f"PDF uploaded: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

            if st.button("Extract Data", type="primary", use_container_width=True):
                with st.spinner("Running extraction pipeline..."):
                    start = time.time()

                    # Build request
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    params = {
                        "document_type": doc_type_value,
                        "store_in_vectordb": store_vector,
                    }
                    if prompt_version.strip():
                        params["prompt_version"] = prompt_version.strip()

                    try:
                        response = requests.post(
                            f"{API_BASE}/extract",
                            files=files,
                            params=params,
                            timeout=120,
                        )
                        elapsed = time.time() - start

                        if response.status_code == 200:
                            result = response.json()
                            st.session_state["last_result"] = result
                            st.session_state["elapsed"] = elapsed
                            st.success(f"Extraction complete in {elapsed:.1f}s")
                        else:
                            st.error(f"Error {response.status_code}: {response.json().get('detail', 'Unknown error')}")

                    except requests.ConnectionError:
                        st.error("Cannot connect to API. Make sure the server is running.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    with col_result:
        st.markdown("### Extraction result")

        if "last_result" in st.session_state:
            result = st.session_state["last_result"]
            elapsed = st.session_state.get("elapsed", 0)

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                confidence = result.get("confidence_score", 0)
                color = "green" if confidence > 0.8 else "orange" if confidence > 0.6 else "red"
                st.metric("Confidence", f"{confidence:.0%}")
            with m2:
                st.metric("Doc type", result.get("document_type", "N/A"))
            with m3:
                ocr = result.get("ocr_result", {})
                st.metric("OCR engine", ocr.get("engine_used", "N/A"))
            with m4:
                st.metric("Time", f"{result.get('processing_time_ms', 0):.0f}ms")

            # Warnings
            warnings = result.get("warnings", [])
            if warnings:
                for w in warnings:
                    st.warning(w)

            # Extracted data
            st.markdown("#### Structured data")
            extracted = result.get("extracted_data", {})
            st.json(extracted)

            # OCR text
            with st.expander("Raw OCR text"):
                ocr_text = result.get("ocr_result", {}).get("raw_text", "")
                st.text(ocr_text[:2000])

            # MLflow link
            run_id = result.get("mlflow_run_id")
            if run_id:
                st.caption(f"MLflow run: `{run_id}` — [View in MLflow](http://localhost:5000)")

            # Document ID
            st.caption(f"Document ID: `{result.get('document_id')}`")

        else:
            st.info("Upload a document and click 'Extract Data' to see results here.")


# ── Tab 2: Search ────────────────────────────────────────────────────────

with tab_search:
    st.markdown("### Semantic search across processed documents")

    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        query = st.text_input("Search query", placeholder="e.g., 'invoices over $5000' or 'contracts with termination clauses'")
    with search_col2:
        n_results = st.number_input("Max results", min_value=1, max_value=20, value=5)

    if query:
        try:
            response = requests.get(
                f"{API_BASE}/search",
                params={"query": query, "n_results": n_results},
                timeout=10,
            )
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                st.caption(f"Found {len(results)} results")

                for r in results:
                    with st.expander(f"Document: {r.get('id', 'N/A')[:20]}..."):
                        st.markdown(f"**Type:** {r.get('metadata', {}).get('document_type', 'N/A')}")
                        st.markdown(f"**Preview:** {r.get('text_preview', 'N/A')}")
                        if r.get("distance") is not None:
                            st.caption(f"Similarity distance: {r['distance']:.4f}")
            else:
                st.error("Search failed")
        except requests.ConnectionError:
            st.error("API not running.")


# ── Tab 3: Monitoring ────────────────────────────────────────────────────

with tab_monitor:
    st.markdown("### Pipeline monitoring")

    mon_col1, mon_col2 = st.columns(2)

    with mon_col1:
        st.markdown("#### Drift detection")
        drift_type = st.selectbox("Document type", ["invoice", "contract", "report"], key="drift_type")

        if st.button("Check for drift"):
            try:
                response = requests.get(
                    f"{API_BASE}/monitoring/drift",
                    params={"document_type": drift_type},
                    timeout=10,
                )
                if response.status_code == 200:
                    drift = response.json()
                    status = drift.get("status", "unknown")

                    if status == "stable":
                        st.success("No drift detected — extraction quality is stable")
                    elif status == "drift_detected":
                        st.warning(f"Drift detected! Magnitude: {drift.get('drift_magnitude', 0):.4f}")
                    elif status == "insufficient_data":
                        st.info(f"Not enough data yet ({drift.get('runs', 0)} runs). Need 50+ for drift analysis.")
                    else:
                        st.info(f"Status: {status}")

                    st.json(drift)
            except requests.ConnectionError:
                st.error("API not running.")

    with mon_col2:
        st.markdown("#### Prompt A/B comparison")
        ab_type = st.selectbox("Document type", ["invoice", "contract", "report"], key="ab_type")

        if st.button("Compare prompts"):
            try:
                response = requests.get(
                    f"{API_BASE}/prompts/compare",
                    params={"document_type": ab_type},
                    timeout=10,
                )
                if response.status_code == 200:
                    data = response.json()
                    versions = data.get("versions", {})
                    if versions:
                        st.json(versions)
                    else:
                        st.info("No prompt comparison data yet. Process some documents first!")
            except requests.ConnectionError:
                st.error("API not running.")

    st.divider()
    st.markdown("#### Quick links")
    link_col1, link_col2, link_col3 = st.columns(3)
    with link_col1:
        st.markdown("[MLflow Dashboard](http://localhost:5000)")
    with link_col2:
        st.markdown("[API Docs (Swagger)](http://localhost:8000/docs)")
    with link_col3:
        st.markdown("[Prometheus Metrics](http://localhost:8000/metrics)")
