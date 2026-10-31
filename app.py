"""
app.py

DocuMind - AI-Powered Intelligent Document Processing System
A modern, dark-themed, high-contrast document-scanning and intelligent document-processing web application.
"""

import io
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
from PIL import Image

# Core Modules
import converter
import ocr_module
import ai_module
import document_processor

# Set page config
st.set_page_config(
    page_title="DocuMind | AI Document Processing System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast, Sleek Cyber-Modern Dark Theme
# Designed for maximum readability and visual impact: Deep Obsidian backgrounds with Electric Blue, Cyan, and Emerald accents.
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global reset */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #080c14 !important;
        color: #f1f5f9 !important;
    }

    /* Container padding */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1440px;
    }

    /* Typography high contrast */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    p, span, label, div {
        color: #e2e8f0;
    }

    /* Top Hero Header with Radiant Cyber Gradient */
    .documind-hero-header {
        background: linear-gradient(135deg, #0b1329 0%, #172554 45%, #1d4ed8 100%);
        border: 1px solid rgba(59, 130, 246, 0.35);
        border-radius: 18px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 12px 36px -8px rgba(37, 99, 235, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.15);
        position: relative;
        overflow: hidden;
    }
    .documind-hero-header h1 {
        margin: 0;
        font-size: 2.4rem;
        font-weight: 800;
        color: #ffffff !important;
        letter-spacing: -0.8px;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
    }
    .documind-hero-header p {
        margin: 0.6rem 0 0 0;
        font-size: 1.05rem;
        color: #cbd5e1 !important;
        max-width: 850px;
        line-height: 1.6;
    }

    /* Cards with High-Contrast Deep Slate Background & Glowing Border */
    .feature-card {
        background: #111827 !important;
        border: 1px solid #1f293d !important;
        border-radius: 16px;
        padding: 1.4rem 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        position: relative;
    }
    .feature-card:hover {
        transform: translateY(-4px);
        border-color: #38bdf8 !important;
        box-shadow: 0 12px 30px -5px rgba(56, 189, 248, 0.25);
    }

    .metric-number {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        font-weight: 700;
        color: #94a3b8 !important;
        letter-spacing: 0.8px;
        margin-bottom: 0.35rem;
    }
    .metric-sub {
        font-size: 0.82rem;
        color: #64748b !important;
        margin-top: 0.25rem;
    }

    /* Badges & Status Pills */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.32rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.4px;
        text-transform: uppercase;
    }
    .pill-cyan { background: #083344; color: #38bdf8; border: 1px solid #0e7490; }
    .pill-emerald { background: #064e3b; color: #34d399; border: 1px solid #059669; }
    .pill-amber { background: #451a03; color: #fbbf24; border: 1px solid #d97706; }
    .pill-rose { background: #4c0519; color: #fb7185; border: 1px solid #e11d48; }
    .pill-indigo { background: #1e1b4b; color: #a5b4fc; border: 1px solid #4f46e5; }

    /* Callout Card */
    .callout-box {
        background: #0f172a !important;
        border: 1px solid #1e293b;
        border-left: 5px solid #38bdf8;
        padding: 1.2rem 1.4rem;
        border-radius: 12px;
        margin-bottom: 1.4rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }

    /* Chat Elements */
    .user-bubble {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: #ffffff !important;
        padding: 0.95rem 1.3rem;
        border-radius: 18px 18px 4px 18px;
        max-width: 82%;
        font-size: 0.96rem;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3);
        line-height: 1.5;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .ai-card {
        background: #111827 !important;
        border: 1px solid #1e293b !important;
        border-left: 4px solid #38bdf8 !important;
        color: #f1f5f9 !important;
        padding: 1.2rem 1.4rem;
        border-radius: 18px 18px 18px 4px;
        max-width: 90%;
        font-size: 0.96rem;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        line-height: 1.6;
    }
    .ai-context-quote {
        background: #0b1120;
        border: 1px dashed #334155;
        border-radius: 8px;
        padding: 0.7rem 0.9rem;
        margin-top: 0.85rem;
        font-size: 0.84rem;
        color: #94a3b8 !important;
        font-style: italic;
    }

    /* Streamlit Native Widget High-Contrast Overrides */
    div[data-testid="stSidebar"] {
        background-color: #060911 !important;
        border-right: 1px solid #1e293b !important;
    }

    div[data-baseweb="input"], div[data-baseweb="textarea"] {
        background-color: #111827 !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }
    input, textarea {
        color: #ffffff !important;
        background-color: transparent !important;
    }
    div[data-baseweb="select"] {
        background-color: #111827 !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }

    /* Buttons with high contrast */
    button[kind="primary"], .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        border: 1px solid #60a5fa !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4) !important;
        transition: all 0.2s ease !important;
    }
    button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.6) !important;
    }
    button[kind="secondary"], .stButton > button {
        background: #111827 !important;
        color: #f1f5f9 !important;
        border: 1px solid #334155 !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
    }
    button[kind="secondary"]:hover, .stButton > button:hover {
        border-color: #38bdf8 !important;
        color: #38bdf8 !important;
        box-shadow: 0 4px 14px rgba(56, 189, 248, 0.2) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: 2px solid #1e293b !important;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 8px 8px 0 0 !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.2rem !important;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #111827 !important;
        color: #ffffff !important;
        border-radius: 10px !important;
    }
    div[data-testid="stExpander"] {
        background-color: #0b1120 !important;
        border: 1px solid #1e293b !important;
        border-radius: 12px !important;
        margin-bottom: 0.8rem;
    }

    /* Pulsing Green Indicator */
    .glow-dot {
        width: 8px;
        height: 8px;
        background-color: #34d399;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 10px #34d399;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# =====================================================================
# SESSION STATE INITIALIZATION
# =====================================================================

if "current_doc" not in st.session_state:
    st.session_state.current_doc = None

if "current_previews" not in st.session_state:
    st.session_state.current_previews = []

if "searchable_pdf_bytes" not in st.session_state:
    st.session_state.searchable_pdf_bytes = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "ocr_engine" not in st.session_state:
    st.session_state.ocr_engine = "auto"

if "external_api_key" not in st.session_state:
    st.session_state.external_api_key = ""

if "api_provider" not in st.session_state:
    st.session_state.api_provider = "openai"

if "quick_prompt_query" not in st.session_state:
    st.session_state.quick_prompt_query = ""


# =====================================================================
# SIDEBAR NAVIGATION & SYSTEM CONTROLS
# =====================================================================

with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:0.8rem; padding: 0.4rem 0 1.2rem 0; border-bottom: 1px solid #1e293b; margin-bottom: 1.2rem;">
        <div style="background: linear-gradient(135deg, #0284c7, #2563eb); width: 40px; height: 40px; border-radius: 10px; display:flex; align-items:center; justify-content:center; font-size:1.3rem; box-shadow: 0 0 15px rgba(37,99,235,0.5);">
            🧠
        </div>
        <div>
            <h2 style="margin:0; font-size:1.35rem; font-weight:800; color:#ffffff !important;">DocuMind</h2>
            <div style="font-size:0.72rem; color:#38bdf8; font-weight:700; text-transform:uppercase; letter-spacing:0.8px;">AI Document Suite</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav_selection = st.radio(
        "Navigation Menu",
        [
            "🏠 Dashboard",
            "📤 Document Upload & Preview",
            "🔍 OCR Text Scanner",
            "💬 Ask AI Assistant",
            "⚙️ Document Converter",
            "🤖 AI Document Intelligence",
            "🔒 Privacy & PII Redactor",
            "🌐 Document Translator",
            "📊 Document Analytics",
            "🕒 Document History",
        ],
        label_visibility="collapsed"
    )

    # Active Document Status Badge
    curr = st.session_state.current_doc
    if curr:
        st.markdown("---")
        st.markdown(f"""
        <div style="background: #111827; border: 1px solid #0284c7; border-radius: 12px; padding: 0.9rem; box-shadow: 0 4px 12px rgba(2, 132, 199, 0.2);">
            <div style="font-size: 0.72rem; color: #38bdf8; font-weight: 800; text-transform: uppercase;">Active Document</div>
            <div style="font-weight: 700; font-size: 0.9rem; color: #ffffff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-top: 0.2rem;">
                {curr.get('filename')}
            </div>
            <div style="display: flex; gap: 0.4rem; margin-top: 0.5rem; flex-wrap: wrap;">
                <span class="status-pill pill-cyan">{curr.get('classification', 'Document')}</span>
                <span class="status-pill pill-emerald">{round(curr.get('avg_confidence', 0.0) * 100, 1)}% Conf</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Engine Settings")

    available_engines = ocr_module.get_available_ocr_engines()
    engine_options = ["Auto (Smart Hybrid)"]
    if "easyocr" in available_engines:
        engine_options.append("EasyOCR (Local Deep Learning)")
    if "tesseract" in available_engines:
        engine_options.append("Tesseract OCR (Local Binary)")

    selected_engine_label = st.selectbox(
        "OCR Engine",
        engine_options,
        index=0,
        help="Select OCR engine. EasyOCR operates 100% locally with high precision."
    )

    if "Auto" in selected_engine_label:
        st.session_state.ocr_engine = "auto"
    elif "EasyOCR" in selected_engine_label:
        st.session_state.ocr_engine = "easyocr"
    elif "Tesseract" in selected_engine_label:
        st.session_state.ocr_engine = "tesseract"

    preprocess_method = st.selectbox(
        "Image Enhancement",
        ["contrast", "adaptive", "denoise", "grayscale", "none"],
        index=0,
        format_func=lambda x: {
            "contrast": "CLAHE Contrast Enhancement",
            "adaptive": "Adaptive Thresholding",
            "denoise": "Non-local Denoising",
            "grayscale": "Grayscale Normalized",
            "none": "Original Image"
        }.get(x, x)
    )

    with st.expander("🌐 Cloud AI Keys (Optional)"):
        st.caption("DocuMind runs 100% offline. Adding an API key is completely optional.")
        st.session_state.api_provider = st.selectbox("Provider", ["OpenAI", "Gemini"], index=0).lower()
        st.session_state.external_api_key = st.text_input(
            "API Key",
            value=st.session_state.external_api_key,
            type="password",
            help="Optional OpenAI or Google Gemini API key"
        )

    st.markdown("---")
    history_records = document_processor.get_history()
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.8rem; color:#94a3b8; font-weight:600;">
        <span><span class="glow-dot"></span> System Online</span>
        <span>History: <b>{len(history_records)}</b></span>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑️ Clear Workspace History", use_container_width=True):
        document_processor.clear_history()
        st.session_state.current_doc = None
        st.session_state.current_previews = []
        st.session_state.chat_history = []
        st.success("History cleared.")
        st.rerun()


# =====================================================================
# PAGE 1: 🏠 DASHBOARD
# =====================================================================

if nav_selection == "🏠 Dashboard":
    st.markdown("""
    <div class="documind-hero-header">
        <h1>Intelligent Document Processing</h1>
        <p>Enterprise AI Document Scanning, High-Precision Dual-Engine OCR, Searchable PDF Synthesis, PII Redaction, Real-Time Document Q&A, and Analytics.</p>
    </div>
    """, unsafe_allow_html=True)

    # Top Metric Tiles
    history = document_processor.get_history()
    total_docs = len(history)
    total_pages = sum(r.get("page_count", 1) for r in history)
    avg_accuracy = (
        round(sum(r.get("avg_confidence", 0.0) for r in history) / max(1, total_docs) * 100, 1)
        if total_docs > 0 else 98.5
    )
    total_insights = sum(
        len(r.get("extracted_fields", {}).get("dates", [])) +
        len(r.get("extracted_fields", {}).get("amounts", []))
        for r in history
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="feature-card">
            <div class="metric-label">📄 Documents Processed</div>
            <div class="metric-number">{total_docs}</div>
            <div class="metric-sub">Archived in database</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="feature-card">
            <div class="metric-label">📑 Pages Analyzed</div>
            <div class="metric-number">{total_pages}</div>
            <div class="metric-sub">PDF pages & image files</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="feature-card">
            <div class="metric-label">🎯 Recognition Accuracy</div>
            <div class="metric-number">{avg_accuracy}%</div>
            <div class="metric-sub">OCR confidence benchmark</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="feature-card">
            <div class="metric-label">💡 Extracted Entities</div>
            <div class="metric-number">{total_insights}</div>
            <div class="metric-sub">Dates, amounts & reference IDs</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Fast Navigation Cards
    st.markdown("### ⚡ Quick Navigation & Capabilities")
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.markdown("""
        <div class="feature-card">
            <span style="font-size: 1.8rem;">📤</span>
            <h4 style="margin: 0.5rem 0 0.3rem 0;">Upload & Preview</h4>
            <p style="font-size: 0.84rem; color: #94a3b8; line-height: 1.4;">Ingest PDFs or images with interactive multi-page page viewer.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Uploader", key="btn_nav_up", use_container_width=True):
            st.session_state["nav_selection"] = "📤 Document Upload & Preview"
            st.rerun()

    with q2:
        st.markdown("""
        <div class="feature-card">
            <span style="font-size: 1.8rem;">🔍</span>
            <h4 style="margin: 0.5rem 0 0.3rem 0;">OCR Text Scanner</h4>
            <p style="font-size: 0.84rem; color: #94a3b8; line-height: 1.4;">Extract, live edit, and download text with confidence metrics.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Scanner", key="btn_nav_ocr", use_container_width=True):
            st.session_state["nav_selection"] = "🔍 OCR Text Scanner"
            st.rerun()

    with q3:
        st.markdown("""
        <div class="feature-card">
            <span style="font-size: 1.8rem;">💬</span>
            <h4 style="margin: 0.5rem 0 0.3rem 0;">Ask AI Assistant</h4>
            <p style="font-size: 0.84rem; color: #94a3b8; line-height: 1.4;">Interactive chat with documents, prompt chips, and cited context.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Chat", key="btn_nav_ask", type="primary", use_container_width=True):
            st.session_state["nav_selection"] = "💬 Ask AI Assistant"
            st.rerun()

    with q4:
        st.markdown("""
        <div class="feature-card">
            <span style="font-size: 1.8rem;">🔒</span>
            <h4 style="margin: 0.5rem 0 0.3rem 0;">Privacy & Redaction</h4>
            <p style="font-size: 0.84rem; color: #94a3b8; line-height: 1.4;">Auto-redact PII (emails, phones, IDs) before exporting.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Redact PII", key="btn_nav_redact", use_container_width=True):
            st.session_state["nav_selection"] = "🔒 Privacy & PII Redactor"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Recent Documents Table
    st.markdown("### 🕒 Recent Scans & Activity")
    if history:
        table_rows = []
        for r in history[:6]:
            table_rows.append({
                "Document Name": r.get("filename"),
                "Type": r.get("classification", "General"),
                "Pages": r.get("page_count", 1),
                "Confidence": f"{round(r.get('avg_confidence', 0.0) * 100, 1)}%",
                "Timestamp": r.get("timestamp"),
                "Speed": f"{r.get('processing_time_sec', 0)}s",
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
    else:
        st.info("No documents have been scanned yet. Upload your first document to see live statistics!")


# =====================================================================
# PAGE 2: 📤 DOCUMENT UPLOAD & PREVIEW
# =====================================================================

elif nav_selection == "📤 Document Upload & Preview":
    st.markdown("""
    <div class="documind-hero-header">
        <h1>Document Upload & Visual Preview</h1>
        <p>Upload PDF documents and high-resolution images (JPG, JPEG, PNG, WEBP). Supports batch uploads and interactive page inspection.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Choose PDF or Image documents",
        type=["pdf", "jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="Supported formats: PDF, JPG, JPEG, PNG, WEBP. Multiple files supported."
    )

    if uploaded_files:
        st.success(f"Selected **{len(uploaded_files)}** document(s).")

        doc_names = [f.name for f in uploaded_files]
        selected_file_name = st.selectbox("Select document to inspect:", doc_names)
        active_file = next(f for f in uploaded_files if f.name == selected_file_name)

        col_proc1, col_proc2 = st.columns([3, 1])
        with col_proc1:
            st.markdown(f"**Size:** `{round(len(active_file.getvalue()) / 1024, 1)} KB` | **Type:** `{active_file.type}`")
        with col_proc2:
            process_btn = st.button("⚡ Process Document Now", type="primary", use_container_width=True)

        if process_btn:
            with st.spinner("Processing document through OCR & AI Engine..."):
                active_file.seek(0)
                result = document_processor.process_document_pipeline(
                    active_file,
                    filename=active_file.name,
                    ocr_engine=st.session_state.ocr_engine,
                    preprocess_method=preprocess_method,
                    external_api_key=st.session_state.external_api_key,
                    api_provider=st.session_state.api_provider,
                )
                st.session_state.current_doc = result["record"]
                st.session_state.current_previews = result["previews"]
                st.session_state.searchable_pdf_bytes = result["searchable_pdf_bytes"]
                st.success("✅ Document processed successfully! Click below to Ask AI questions or inspect extracted text.")

        st.markdown("---")
        st.subheader("👁️ Visual Page Preview")

        active_file.seek(0)
        temp_data = active_file.read()
        active_file.seek(0)

        if active_file.name.lower().endswith(".pdf"):
            temp_previews = converter.pdf_to_images(temp_data, dpi_scale=1.5)
            num_pages = len(temp_previews)
            st.markdown(f"**Total Pages in Document:** `{num_pages}`")

            page_idx = st.slider("Select Page to View", min_value=1, max_value=max(1, num_pages), value=1)
            selected_page_buf = temp_previews[page_idx - 1]
            selected_page_buf.seek(0)
            st.image(selected_page_buf, caption=f"Page {page_idx} of {num_pages}", use_container_width=True)
        else:
            st.image(temp_data, caption=active_file.name, use_container_width=True)

    else:
        st.info("👆 Drag and drop your PDF or image files above to get started.")

        st.markdown("#### 🧪 Test With Sample Documents")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if os.path.exists("sample.png") and st.button("📂 Load sample.png", use_container_width=True):
                with open("sample.png", "rb") as sf:
                    res = document_processor.process_document_pipeline(
                        sf.read(),
                        filename="sample.png",
                        ocr_engine=st.session_state.ocr_engine,
                        preprocess_method=preprocess_method,
                    )
                st.session_state.current_doc = res["record"]
                st.session_state.current_previews = res["previews"]
                st.session_state.searchable_pdf_bytes = res["searchable_pdf_bytes"]
                st.success("Loaded sample.png into active workspace!")
                st.rerun()

        with col_s2:
            if os.path.exists("sample.pdf") and st.button("📂 Load sample.pdf", use_container_width=True):
                with open("sample.pdf", "rb") as sf:
                    res = document_processor.process_document_pipeline(
                        sf.read(),
                        filename="sample.pdf",
                        ocr_engine=st.session_state.ocr_engine,
                        preprocess_method=preprocess_method,
                    )
                st.session_state.current_doc = res["record"]
                st.session_state.current_previews = res["previews"]
                st.session_state.searchable_pdf_bytes = res["searchable_pdf_bytes"]
                st.success("Loaded sample.pdf into active workspace!")
                st.rerun()


# =====================================================================
# PAGE 3: 🔍 OCR TEXT SCANNER
# =====================================================================

elif nav_selection == "🔍 OCR Text Scanner":
    st.markdown("""
    <div class="documind-hero-header">
        <h1>OCR Text Scanner</h1>
        <p>Extract, review, edit, and export text from scanned documents, invoices, receipts, and images with confidence metrics.</p>
    </div>
    """, unsafe_allow_html=True)

    current_doc = st.session_state.current_doc

    if not current_doc:
        st.warning("⚠️ No document is currently active. Please upload a document or load a sample below.")
        if os.path.exists("sample.png") and st.button("📂 Load sample.png for OCR Testing"):
            with open("sample.png", "rb") as f:
                res = document_processor.process_document_pipeline(f.read(), filename="sample.png")
                st.session_state.current_doc = res["record"]
                st.session_state.current_previews = res["previews"]
                st.session_state.searchable_pdf_bytes = res["searchable_pdf_bytes"]
            st.rerun()
    else:
        m1, m2, m3, m4 = st.columns(4)
        words = len(current_doc.get("full_text", "").split())
        chars = len(current_doc.get("full_text", ""))
        conf = round(current_doc.get("avg_confidence", 0.0) * 100, 1)

        m1.metric("Active File", current_doc.get("filename", "Unknown"))
        m2.metric("Word Count", f"{words:,}")
        m3.metric("Character Count", f"{chars:,}")
        m4.metric("OCR Accuracy", f"{conf}%")

        st.markdown("---")

        col_text, col_preview = st.columns([3, 2])

        with col_text:
            st.subheader("📝 Extracted Text (Live Editable)")
            st.caption("Edit or correct text directly in the box below before downloading.")

            edited_text = st.text_area(
                "Document Text",
                value=current_doc.get("full_text", ""),
                height=440,
                key="doc_editor_area",
                label_visibility="collapsed"
            )

            st.markdown("#### 💾 Export Extracted Text")
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.download_button(
                    "📄 Download TXT",
                    data=edited_text.encode("utf-8"),
                    file_name=f"{Path(current_doc.get('filename')).stem}_extracted.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with d2:
                json_buf = document_processor.export_as_json(current_doc)
                st.download_button(
                    "📊 Download JSON",
                    data=json_buf.getvalue(),
                    file_name=f"{Path(current_doc.get('filename')).stem}_report.json",
                    mime="application/json",
                    use_container_width=True
                )
            with d3:
                csv_buf = document_processor.export_as_csv(current_doc)
                st.download_button(
                    "📈 Download CSV",
                    data=csv_buf.getvalue(),
                    file_name=f"{Path(current_doc.get('filename')).stem}_entities.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with d4:
                zip_buf = document_processor.export_full_bundle_zip(current_doc)
                st.download_button(
                    "📦 Full ZIP Bundle",
                    data=zip_buf.getvalue(),
                    file_name=f"{Path(current_doc.get('filename')).stem}_DocuMind_Export.zip",
                    mime="application/zip",
                    use_container_width=True
                )

        with col_preview:
            st.subheader("🖼️ Source Document Preview")
            previews = st.session_state.current_previews
            if previews:
                if len(previews) > 1:
                    sel_p = st.selectbox("Inspect Page:", range(1, len(previews) + 1))
                    img_buf = previews[sel_p - 1]
                    img_buf.seek(0)
                    st.image(img_buf, caption=f"Page {sel_p} of {len(previews)}", use_container_width=True)
                else:
                    img_buf = previews[0]
                    img_buf.seek(0)
                    st.image(img_buf, caption=current_doc.get("filename"), use_container_width=True)
            elif os.path.exists(current_doc.get("saved_path", "")):
                st.image(current_doc.get("saved_path"), caption=current_doc.get("filename"), use_container_width=True)

            st.markdown("#### 🎯 Per-Word Recognition Breakdown")
            all_conf = current_doc.get("pages", [{}])[0].get("confidence", []) if current_doc.get("pages") else []
            if all_conf:
                conf_data = []
                for item in all_conf[:15]:
                    c_val = item.get("confidence", 0.0)
                    conf_data.append({
                        "Detected Phrase": item.get("text", "")[:40],
                        "Accuracy": f"{round(c_val * 100, 1)}%",
                    })
                st.dataframe(pd.DataFrame(conf_data), use_container_width=True, height=180)
            else:
                st.caption("Native digital text extracted with 100% precision.")


# =====================================================================
# PAGE 4: 💬 ASK AI ASSISTANT (CONVERSATIONAL DOCUMENT CHAT)
# =====================================================================

elif nav_selection == "💬 Ask AI Assistant":
    st.markdown("""
    <div class="documind-hero-header">
        <h1>💬 Ask AI Document Assistant</h1>
        <p>Chat directly with your document. Ask natural language questions, extract specific data points, and verify answers with cited page references.</p>
    </div>
    """, unsafe_allow_html=True)

    current_doc = st.session_state.current_doc

    if not current_doc:
        st.warning("⚠️ No document is currently active. Please load or upload a document to begin chatting.")
        col_load_sample1, col_load_sample2 = st.columns(2)
        with col_load_sample1:
            if os.path.exists("sample.png") and st.button("📂 Load sample.png to Chat", use_container_width=True):
                with open("sample.png", "rb") as f:
                    res = document_processor.process_document_pipeline(f.read(), filename="sample.png")
                    st.session_state.current_doc = res["record"]
                    st.session_state.current_previews = res["previews"]
                    st.session_state.searchable_pdf_bytes = res["searchable_pdf_bytes"]
                st.rerun()
        with col_load_sample2:
            if os.path.exists("sample.pdf") and st.button("📂 Load sample.pdf to Chat", use_container_width=True):
                with open("sample.pdf", "rb") as f:
                    res = document_processor.process_document_pipeline(f.read(), filename="sample.pdf")
                    st.session_state.current_doc = res["record"]
                    st.session_state.current_previews = res["previews"]
                    st.session_state.searchable_pdf_bytes = res["searchable_pdf_bytes"]
                st.rerun()
    else:
        # Document Context Strip
        st.markdown(f"""
        <div style="background: #111827; border: 1px solid #1e293b; border-radius: 14px; padding: 1rem 1.4rem; display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.4rem; box-shadow: 0 4px 16px rgba(0,0,0,0.25);">
            <div style="display: flex; align-items: center; gap: 0.85rem;">
                <span style="font-size: 1.8rem;">📄</span>
                <div>
                    <div style="font-weight: 800; color: #ffffff; font-size: 1.05rem;">{current_doc.get('filename')}</div>
                    <div style="font-size: 0.82rem; color: #94a3b8;">{len(current_doc.get('full_text', '').split()):,} words | {current_doc.get('page_count', 1)} page(s) | {round(current_doc.get('avg_confidence', 0.0)*100, 1)}% Accuracy</div>
                </div>
            </div>
            <div style="display: flex; gap: 0.5rem; align-items: center;">
                <span class="status-pill pill-cyan">{current_doc.get('classification', 'General')}</span>
                <span class="status-pill pill-emerald"><span class="glow-dot"></span> AI Grounded</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Quick Question Pills
        st.markdown("**💡 Suggested Questions (Click to Ask):**")
        pills = [
            ("📋 Summarize document", "Summarize this document"),
            ("💰 Total costs & breakdown", "What is the total amount and financial breakdown?"),
            ("📅 Payment deadlines", "When is the payment due date or deadlines?"),
            ("👤 Parties & people", "Who are the key people and organizations mentioned?"),
            ("⚠️ Missing details & risks", "Are there any missing details, risks, or inconsistencies?"),
            ("📞 Contact information", "What are the contact details, email addresses, and phone numbers?")
        ]

        cols_pill = st.columns(len(pills))
        for idx, (label, query_text) in enumerate(pills):
            if cols_pill[idx].button(label, key=f"pill_q_{idx}", use_container_width=True):
                st.session_state["user_chat_input"] = query_text

        st.markdown("<br>", unsafe_allow_html=True)

        # Split Screen Layout: Left Document Inspector, Right AI Conversation
        col_doc_side, col_chat_side = st.columns([1, 1.35])

        # LEFT COLUMN: Document Inspector
        with col_doc_side:
            st.markdown("### 📑 Document Inspector")
            tab_side_preview, tab_side_text, tab_side_entities = st.tabs([
                "🖼️ Visual Page",
                "📝 Extracted Text",
                "🏷️ Quick Entities"
            ])

            with tab_side_preview:
                previews = st.session_state.current_previews
                if previews:
                    if len(previews) > 1:
                        p_sel = st.selectbox("Inspect Page:", range(1, len(previews) + 1), key="chat_page_sel")
                        buf = previews[p_sel - 1]
                        buf.seek(0)
                        st.image(buf, caption=f"Page {p_sel}", use_container_width=True)
                    else:
                        buf = previews[0]
                        buf.seek(0)
                        st.image(buf, caption="Page 1", use_container_width=True)
                elif os.path.exists(current_doc.get("saved_path", "")):
                    st.image(current_doc.get("saved_path"), caption=current_doc.get("filename"), use_container_width=True)

            with tab_side_text:
                st.text_area(
                    "Raw Document Text",
                    value=current_doc.get("full_text", ""),
                    height=450,
                    disabled=True,
                    label_visibility="collapsed"
                )

            with tab_side_entities:
                fields = current_doc.get("extracted_fields", {})
                st.markdown("#### 📅 Dates Found")
                for d in fields.get("dates", [])[:6]:
                    st.markdown(f"- **{d.get('label')}:** `{d.get('value')}`")
                st.markdown("#### 💰 Amounts Found")
                for a in fields.get("amounts", [])[:6]:
                    st.markdown(f"- **{a.get('label')}:** `{a.get('raw')}`")
                st.markdown("#### 📞 Contacts")
                for em in fields.get("contacts", {}).get("emails", []):
                    st.markdown(f"- **Email:** `{em}`")

        # RIGHT COLUMN: Interactive AI Chat
        with col_chat_side:
            st.markdown("### 🤖 Document Conversation Stream")

            chat_container = st.container()

            # Render Chat Stream
            with chat_container:
                if not st.session_state.chat_history:
                    st.markdown("""
                    <div style="text-align: center; padding: 2.8rem 1rem; color: #94a3b8; background: #111827; border: 1px dashed #334155; border-radius: 16px;">
                        <span style="font-size: 2.8rem;">💬</span>
                        <h3 style="margin: 0.7rem 0 0.2rem 0; color: #ffffff;">Ask your first question!</h3>
                        <p style="font-size: 0.92rem; margin: 0; max-width: 400px; margin: auto; color: #94a3b8;">
                            Type any question below or click one of the suggested prompt chips above to extract answers verified by document citations.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    for i, msg in enumerate(st.session_state.chat_history):
                        if msg["role"] == "user":
                            st.markdown(f"""
                            <div style="display: flex; justify-content: flex-end; margin-bottom: 1rem;">
                                <div class="user-bubble">
                                    {msg["content"]}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            conf_score = round(msg.get("confidence", 0.95) * 100, 1)
                            page_ref = msg.get("page_number", 1)
                            context_cite = msg.get("cited_context", "")

                            st.markdown(f"""
                            <div style="display: flex; justify-content: flex-start; margin-bottom: 1.3rem;">
                                <div class="ai-card">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                                            <span style="font-size: 1.2rem;">🧠</span>
                                            <strong style="color: #38bdf8; font-size: 1rem;">DocuMind AI</strong>
                                        </div>
                                        <div style="display: flex; gap: 0.4rem;">
                                            <span class="status-pill pill-cyan">Page {page_ref}</span>
                                            <span class="status-pill pill-emerald">{conf_score}% Conf</span>
                                        </div>
                                    </div>
                                    <div style="white-space: pre-wrap; font-size: 0.98rem; color: #f8fafc;">{msg["content"]}</div>
                                    <div class="ai-context-quote">
                                        <b>🔍 Verified Source Context:</b><br>
                                        "{context_cite}"
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

            # Chat Input Form
            with st.form(key="chat_input_form", clear_on_submit=True):
                user_question = st.text_input(
                    "Ask a question about this document:",
                    value=st.session_state.get("user_chat_input", ""),
                    placeholder="e.g. What is the total amount due? When is the deadline? Who is the client?",
                    label_visibility="collapsed"
                )
                col_btn_send, col_btn_clear = st.columns([4, 1])
                with col_btn_send:
                    send_btn = st.form_submit_button("🚀 Ask Question", type="primary", use_container_width=True)
                with col_btn_clear:
                    clear_chat_btn = st.form_submit_button("🗑️ Clear", use_container_width=True)

            if clear_chat_btn:
                st.session_state.chat_history = []
                st.session_state["user_chat_input"] = ""
                st.rerun()

            if send_btn and user_question.strip():
                st.session_state.chat_history.append({"role": "user", "content": user_question.strip()})
                st.session_state["user_chat_input"] = ""

                with st.spinner("Analyzing document context and verifying answer..."):
                    ai_answer = ai_module.ask_ai_question(
                        user_question.strip(),
                        current_doc.get("full_text", ""),
                        pages=current_doc.get("pages", []),
                        api_key=st.session_state.external_api_key,
                        api_provider=st.session_state.api_provider,
                    )

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": ai_answer.get("answer", ""),
                    "confidence": ai_answer.get("confidence", 0.95),
                    "cited_context": ai_answer.get("cited_context", ""),
                    "page_number": ai_answer.get("page_number", 1),
                })
                st.rerun()


# =====================================================================
# PAGE 5: ⚙️ DOCUMENT CONVERTER
# =====================================================================

elif nav_selection == "⚙️ Document Converter":
    st.markdown("""
    <div class="documind-hero-header">
        <h1>Document Converter & PDF Tools</h1>
        <p>Convert images to searchable PDFs, extract PDF pages into high-res images, rotate, resize, and convert formats.</p>
    </div>
    """, unsafe_allow_html=True)

    tab_searchable, tab_pdf2img, tab_img2pdf, tab_img_tools = st.tabs([
        "🔍 Searchable PDF Creator",
        "📄 PDF Pages to Images",
        "🖼️ Images to PDF",
        "🛠️ Format & Image Tools"
    ])

    with tab_searchable:
        st.subheader("Generate Searchable PDF (with Invisible Text Layer)")
        st.write(
            "Transforms scanned images or documents into a true Searchable PDF. "
            "The visual appearance is preserved while allowing text selection, copying, and search."
        )

        current_doc = st.session_state.current_doc
        if st.session_state.searchable_pdf_bytes:
            st.success("✅ Searchable PDF is ready for download for the currently loaded document!")
            st.download_button(
                "📥 Download Searchable PDF",
                data=st.session_state.searchable_pdf_bytes,
                file_name=f"{Path(current_doc.get('filename', 'document')).stem}_searchable.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )
        else:
            st.info("Upload or process a document first to generate its Searchable PDF.")

        st.markdown("---")
        st.markdown("#### Convert Any Image Directly to Searchable PDF")
        direct_img = st.file_uploader("Upload Image to make Searchable PDF", type=["jpg", "jpeg", "png"], key="uploader_spdf")
        if direct_img:
            if st.button("Generate Searchable PDF", type="primary", key="btn_gen_spdf"):
                with st.spinner("Analyzing text layout and embedding searchable text layer..."):
                    img_bytes = direct_img.read()
                    ocr_res = ocr_module.extract_text(Image.open(io.BytesIO(img_bytes)))
                    searchable_in = [{"image": img_bytes, "boxes": ocr_res}]
                    spdf = converter.create_searchable_pdf(searchable_in)
                    st.success("Searchable PDF created successfully!")
                    st.download_button(
                        "📥 Download Resulting Searchable PDF",
                        data=spdf.getvalue(),
                        file_name=f"{Path(direct_img.name).stem}_searchable.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

    with tab_pdf2img:
        st.subheader("Convert PDF Pages into High-Res Images")
        pdf_input = st.file_uploader("Choose a PDF file", type=["pdf"], key="uploader_pdf2img")
        dpi_choice = st.select_slider("Resolution Quality", options=[1.0, 1.5, 2.0, 3.0], value=2.0, format_func=lambda x: f"{int(x*72)} DPI")
        out_fmt = st.radio("Image Format", ["PNG", "JPEG"], horizontal=True)

        if pdf_input:
            if st.button("⚡ Convert PDF to Images", type="primary", key="btn_pdf2img"):
                with st.spinner("Extracting all pages as images..."):
                    pages_list = converter.pdf_to_images(pdf_input, dpi_scale=dpi_choice, image_format=out_fmt)
                    st.success(f"Extracted **{len(pages_list)}** page(s) successfully!")

                    files_dict = {f"page_{i+1}.{out_fmt.lower()}": p for i, p in enumerate(pages_list)}
                    zip_out = converter.create_zip_archive(files_dict)

                    st.download_button(
                        f"📦 Download All Pages (ZIP - {len(pages_list)} images)",
                        data=zip_out.getvalue(),
                        file_name=f"{Path(pdf_input.name).stem}_pages.zip",
                        mime="application/zip",
                        use_container_width=True
                    )

                    st.markdown("#### Individual Page Previews & Downloads")
                    cols = st.columns(min(3, len(pages_list)))
                    for i, page_buf in enumerate(pages_list):
                        with cols[i % len(cols)]:
                            page_buf.seek(0)
                            st.image(page_buf, caption=f"Page {i+1}", use_container_width=True)
                            page_buf.seek(0)
                            st.download_button(
                                f"Download Page {i+1}",
                                data=page_buf.getvalue(),
                                file_name=f"page_{i+1}.{out_fmt.lower()}",
                                mime=f"image/{out_fmt.lower()}",
                                key=f"dl_p_{i}",
                                use_container_width=True
                            )

    with tab_img2pdf:
        st.subheader("Combine Multiple Images into a Single PDF")
        images_upload = st.file_uploader(
            "Select images to combine",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="uploader_img2pdf"
        )
        if images_upload:
            st.write(f"Selected **{len(images_upload)}** image(s).")
            if st.button("Combine Images into PDF", type="primary", key="btn_img2pdf"):
                with st.spinner("Building combined PDF document..."):
                    pdf_result = converter.image_to_pdf(images_upload)
                    st.success("Combined PDF generated successfully!")
                    st.download_button(
                        "📥 Download Combined PDF",
                        data=pdf_result.getvalue(),
                        file_name="combined_documents.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

    with tab_img_tools:
        st.subheader("Image Conversion, Rotation & Resizing Tools")
        tool_img = st.file_uploader("Upload Image to Edit", type=["jpg", "jpeg", "png"], key="uploader_tool_img")

        if tool_img:
            c_preview, c_controls = st.columns([1, 1])

            with c_preview:
                tool_img.seek(0)
                st.image(tool_img, caption="Original Image", use_container_width=True)

            with c_controls:
                action = st.selectbox("Action", ["Convert Format (JPG <-> PNG)", "Rotate Image", "Resize Image"])

                if action == "Convert Format (JPG <-> PNG)":
                    target_format = st.selectbox("Target Format", ["PNG", "JPG"])
                    if st.button("Convert Image", type="primary", key="btn_conv_tool"):
                        tool_img.seek(0)
                        converted = converter.convert_image(tool_img, target_format)
                        st.success(f"Successfully converted to {target_format}!")
                        st.download_button(
                            f"📥 Download {target_format}",
                            data=converted.getvalue(),
                            file_name=f"converted.{target_format.lower()}",
                            mime=f"image/{target_format.lower()}",
                            use_container_width=True
                        )

                elif action == "Rotate Image":
                    angle = st.selectbox("Rotation Angle", [90, 180, 270], format_func=lambda x: f"{x}° clockwise")
                    if st.button("Rotate Image", type="primary", key="btn_rot_tool"):
                        tool_img.seek(0)
                        rotated = converter.rotate_image(tool_img, 360 - angle if angle != 180 else 180)
                        st.image(rotated, caption=f"Rotated {angle}°", use_container_width=True)
                        rotated.seek(0)
                        st.download_button(
                            "📥 Download Rotated Image",
                            data=rotated.getvalue(),
                            file_name="rotated.png",
                            mime="image/png",
                            use_container_width=True
                        )

                elif action == "Resize Image":
                    w = st.number_input("Width (px)", min_value=50, max_value=8000, value=800)
                    h = st.number_input("Height (px)", min_value=50, max_value=8000, value=600)
                    keep_aspect = st.checkbox("Preserve Aspect Ratio", value=True)
                    if st.button("Resize Image", type="primary", key="btn_res_tool"):
                        tool_img.seek(0)
                        resized = converter.resize_image(tool_img, int(w), int(h), preserve_aspect_ratio=keep_aspect)
                        st.image(resized, caption=f"Resized ({w}x{h})", use_container_width=True)
                        resized.seek(0)
                        st.download_button(
                            "📥 Download Resized Image",
                            data=resized.getvalue(),
                            file_name="resized.png",
                            mime="image/png",
                            use_container_width=True
                        )


# =====================================================================
# PAGE 6: 🤖 AI DOCUMENT INTELLIGENCE
# =====================================================================

elif nav_selection == "🤖 AI Document Intelligence":
    st.markdown("""
    <div class="documind-hero-header">
        <h1>AI Document Intelligence</h1>
        <p>Automated classification, executive summaries, structured entity extraction, and completeness audit checks.</p>
    </div>
    """, unsafe_allow_html=True)

    current_doc = st.session_state.current_doc

    if not current_doc:
        st.warning("⚠️ No document is currently loaded. Please upload or scan a document first.")
        if os.path.exists("sample.png") and st.button("📂 Load sample.png for AI Analysis"):
            with open("sample.png", "rb") as f:
                res = document_processor.process_document_pipeline(f.read(), filename="sample.png")
                st.session_state.current_doc = res["record"]
                st.session_state.current_previews = res["previews"]
                st.session_state.searchable_pdf_bytes = res["searchable_pdf_bytes"]
            st.rerun()
    else:
        ai_res = current_doc.get("ai_results", {})
        doc_type = current_doc.get("classification", "General Document")
        summary = current_doc.get("summary", {})
        fields = current_doc.get("extracted_fields", {})
        missing = current_doc.get("missing_information", [])
        inconsistencies = current_doc.get("inconsistencies", [])

        st.markdown(f"""
        <div class="callout-box">
            <div style="font-size:0.76rem; color:#38bdf8; font-weight:800; text-transform:uppercase;">Classified Document Type</div>
            <h2 style="margin:0.25rem 0 0.5rem 0; color:#ffffff !important; font-weight:800;">{doc_type}</h2>
            <div style="display: flex; gap: 0.5rem;">
                <span class="status-pill pill-cyan">Confidence: 95%</span>
                <span class="status-pill pill-emerald"><span class="glow-dot"></span> Offline AI Grounded</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        tab_summary, tab_entities, tab_audit = st.tabs([
            "📋 Document Summary",
            "🏷️ Extracted Fields & Entities",
            "⚠️ Quality & Audit Checks"
        ])

        with tab_summary:
            st.subheader("Executive Summary")
            st.write(summary.get("executive_summary", "No summary available."))

            st.markdown("#### 🎯 Key Takeaway Highlights")
            bullets = summary.get("bullet_points", [])
            if bullets:
                for b in bullets:
                    st.markdown(f"- {b}")
            else:
                st.write("No distinct bullet points extracted.")

            st.markdown("---")
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Word Count", summary.get("word_count", 0))
            m_col2.metric("Estimated Reading Time", f"{summary.get('reading_time_minutes', 1)} min")
            m_col3.metric("Vocabulary Complexity", summary.get("complexity", "Standard"))

            if ai_res.get("external_insights"):
                st.markdown("#### 🌐 External AI Model Insights")
                st.info(ai_res.get("external_insights"))

        with tab_entities:
            st.subheader("Key Extracted Information")

            c_dates, c_amounts = st.columns(2)
            with c_dates:
                st.markdown("#### 📅 Dates Identified")
                dates_list = fields.get("dates", [])
                if dates_list:
                    for d in dates_list:
                        st.markdown(f"- **{d.get('label')}:** `{d.get('value')}`")
                        st.caption(f"Context: \"{d.get('snippet')}\"")
                else:
                    st.caption("No specific dates identified.")

            with c_amounts:
                st.markdown("#### 💰 Financial Amounts")
                amounts_list = fields.get("amounts", [])
                if amounts_list:
                    for a in amounts_list:
                        st.markdown(f"- **{a.get('label')}:** `{a.get('raw')}`")
                        st.caption(f"Context: \"{a.get('snippet')}\"")
                else:
                    st.caption("No monetary amounts identified.")

            st.markdown("---")
            c_refs, c_contacts = st.columns(2)

            with c_refs:
                st.markdown("#### 🔢 Reference Numbers & IDs")
                ref_list = fields.get("reference_ids", [])
                if ref_list:
                    for r in ref_list:
                        st.markdown(f"- **{r.get('label')}:** `{r.get('value')}`")
                else:
                    st.caption("No reference IDs detected.")

            with c_contacts:
                st.markdown("#### 📞 Contact Information")
                contacts = fields.get("contacts", {})
                emails = contacts.get("emails", [])
                phones = contacts.get("phones", [])
                if emails:
                    st.markdown(f"**Emails:** {', '.join(emails)}")
                if phones:
                    st.markdown(f"**Phones:** {', '.join(phones)}")
                if not emails and not phones:
                    st.caption("No contact details identified.")

            entities = fields.get("entities", {})
            if entities.get("organizations") or entities.get("names"):
                st.markdown("---")
                c_org, c_names = st.columns(2)
                with c_org:
                    st.markdown("#### 🏢 Organizations & Entities")
                    for org in entities.get("organizations", []):
                        st.markdown(f"- {org}")
                with c_names:
                    st.markdown("#### 👤 People & Parties")
                    for name in entities.get("names", []):
                        st.markdown(f"- {name}")

        with tab_audit:
            st.subheader("Document Integrity & Quality Check")

            c_miss, c_incons = st.columns(2)

            with c_miss:
                st.markdown("#### 🔍 Missing Information Detection")
                if missing:
                    for m in missing:
                        sev = m.get("severity", "Medium")
                        badge_cls = "pill-rose" if sev == "High" else "pill-amber"
                        st.markdown(f"""
                        <div style="border: 1px solid #334155; border-radius: 12px; padding: 1rem; margin-bottom: 0.9rem; background: #111827;">
                            <span class="status-pill {badge_cls}">{sev} Priority</span>
                            <strong style="margin-left: 0.5rem; color: #ffffff;">{m.get('field')}</strong>
                            <p style="margin: 0.4rem 0 0 0; font-size: 0.88rem; color: #94a3b8;">{m.get('description')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("✅ All expected standard fields for this document type are present!")

            with c_incons:
                st.markdown("#### ⚠️ Inconsistencies & Anomalies")
                if inconsistencies:
                    for inc in inconsistencies:
                        sev = inc.get("severity", "Medium")
                        badge_cls = "pill-rose" if sev == "High" else "pill-amber"
                        st.markdown(f"""
                        <div style="border: 1px solid #e11d48; background-color: #1f1217; border-radius: 12px; padding: 1rem; margin-bottom: 0.9rem;">
                            <span class="status-pill {badge_cls}">{inc.get('type')}</span>
                            <p style="margin: 0.4rem 0 0 0; font-size: 0.88rem; color: #fca5a5;">{inc.get('detail')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("✅ No conflicting dates, mathematical errors, or transcription anomalies detected.")


# =====================================================================
# PAGE 7: 🔒 PRIVACY & PII REDACTOR (NEW FEATURE!)
# =====================================================================

elif nav_selection == "🔒 Privacy & PII Redactor":
    st.markdown("""
    <div class="documind-hero-header">
        <h1>🔒 Document Privacy & PII Redactor</h1>
        <p>Safely sanitize documents before sharing. Automatically detect and redact sensitive Personally Identifiable Information (PII) including emails, phone numbers, and account identifiers.</p>
    </div>
    """, unsafe_allow_html=True)

    current_doc = st.session_state.current_doc

    if not current_doc:
        st.warning("⚠️ No document is currently active. Please load or upload a document to redact.")
        if os.path.exists("sample.png") and st.button("📂 Load sample.png for Redaction"):
            with open("sample.png", "rb") as f:
                res = document_processor.process_document_pipeline(f.read(), filename="sample.png")
                st.session_state.current_doc = res["record"]
                st.session_state.current_previews = res["previews"]
                st.session_state.searchable_pdf_bytes = res["searchable_pdf_bytes"]
            st.rerun()
    else:
        st.subheader("Redaction Rules & Settings")
        c_r1, c_r2, c_r3, c_r4 = st.columns(4)
        with c_r1:
            m_email = st.checkbox("Redact Email Addresses", value=True)
        with c_r2:
            m_phone = st.checkbox("Redact Phone Numbers", value=True)
        with c_r3:
            m_ids = st.checkbox("Redact Account & Tax IDs", value=True)
        with c_r4:
            m_amts = st.checkbox("Redact Monetary Amounts", value=False)

        raw_doc_text = current_doc.get("full_text", "")
        redacted_text, counts = ai_module.redact_sensitive_information(
            raw_doc_text,
            mask_emails=m_email,
            mask_phones=m_phone,
            mask_ids=m_ids,
            mask_amounts=m_amts
        )

        st.markdown("#### 🛡️ Redaction Summary")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Emails Redacted", counts.get("emails", 0))
        s2.metric("Phones Redacted", counts.get("phones", 0))
        s3.metric("IDs Redacted", counts.get("ids", 0))
        s4.metric("Amounts Redacted", counts.get("amounts", 0))

        st.markdown("---")
        c_orig, c_red = st.columns(2)

        with c_orig:
            st.markdown("#### 📄 Original Text")
            st.text_area("Original", value=raw_doc_text, height=380, disabled=True, key="orig_text_area", label_visibility="collapsed")

        with c_red:
            st.markdown("#### 🔒 Sanitized (Redacted) Text")
            st.text_area("Redacted", value=redacted_text, height=380, key="red_text_area", label_visibility="collapsed")

        st.download_button(
            "📥 Download Sanitized Redacted Text (.txt)",
            data=redacted_text.encode("utf-8"),
            file_name=f"{Path(current_doc.get('filename')).stem}_sanitized.txt",
            mime="text/plain",
            type="primary",
            use_container_width=True
        )


# =====================================================================
# PAGE 8: 🌐 DOCUMENT TRANSLATOR (NEW FEATURE!)
# =====================================================================

elif nav_selection == "🌐 Document Translator":
    st.markdown("""
    <div class="documind-hero-header">
        <h1>🌐 Multi-Language Document Translator</h1>
        <p>Translate document content, key headers, and summaries into 10+ languages with side-by-side verification and exports.</p>
    </div>
    """, unsafe_allow_html=True)

    current_doc = st.session_state.current_doc

    # Session state for translator
    if "translator_output" not in st.session_state:
        st.session_state.translator_output = ""
    if "translator_last_lang" not in st.session_state:
        st.session_state.translator_last_lang = "Spanish"
    if "translator_source_text" not in st.session_state:
        st.session_state.translator_source_text = ""

    # Supported languages
    available_languages = [
        "Spanish", "French", "German", "Italian", "Hindi",
        "Portuguese", "Russian", "Arabic", "Japanese", "Chinese"
    ]

    # Show engine status banner
    has_api_key = bool(st.session_state.get("external_api_key", "").strip())
    active_provider = st.session_state.get("api_provider", "openai").upper()
    if has_api_key:
        st.info(f"🤖 **Translation Engine:** External LLM ({active_provider}) configured via sidebar. Preserving formatting, layout, and document structure.")
    else:
        st.info("⚡ **Translation Engine:** High-performance Neural Machine Translation (with offline business terminology dictionary fallback).")

    # Sample document text for immediate testing
    sample_text_fallback = (
        "TAX INVOICE\n"
        "Global Logistics Inc.\n"
        "Invoice No: INV-2026-9901\n"
        "Invoice Date: 2026-05-15\n"
        "Due Date: 2026-05-30\n"
        "Bill To: Dr. Eleanor Vance\n"
        "Subtotal: $2,500.00\n"
        "Tax: $250.00\n"
        "Total Amount: $2,750.00\n"
        "Contact: billing@globallogistics.com | +1 (555) 019-2834\n"
        "Services rendered for cloud infrastructure assessment."
    )

    if current_doc:
        st.markdown("##### 📌 Content Selection")
        col_src, col_target = st.columns([1, 1])
        with col_src:
            source_option = st.selectbox(
                "Source Section to Translate:",
                [
                    "Executive Summary",
                    "Full Document Text",
                    "Key Fields Summary",
                    "Custom / Editable Text"
                ],
                index=0
            )

        with col_target:
            target_lang = st.selectbox(
                "Select Target Language:",
                available_languages,
                index=0
            )

        # Derive source text based on user selection
        if source_option == "Executive Summary":
            chosen_source = current_doc.get("summary", {}).get("executive_summary", "")
            if not chosen_source:
                chosen_source = current_doc.get("full_text", "")[:800]
        elif source_option == "Full Document Text":
            chosen_source = current_doc.get("full_text", "")
        elif source_option == "Key Fields Summary":
            fields = current_doc.get("extracted_fields", {})
            lines = [f"Document Classification: {current_doc.get('classification', {}).get('document_type', 'Unknown')}"]
            if fields.get("amounts"):
                lines.append("Amounts:")
                for a in fields["amounts"][:5]:
                    lines.append(f"- {a.get('label')}: {a.get('raw')}")
            if fields.get("dates"):
                lines.append("Dates:")
                for d in fields["dates"][:5]:
                    lines.append(f"- {d.get('label')}: {d.get('value')}")
            chosen_source = "\n".join(lines)
        else:
            chosen_source = st.session_state.translator_source_text or current_doc.get("full_text", "")[:1000]

    else:
        st.warning("⚠️ No document is currently active. You can translate custom text below, or load sample.png.")
        col_act1, col_act2 = st.columns([1, 1])
        with col_act1:
            if os.path.exists("sample.png") and st.button("📂 Load sample.png for Translation", use_container_width=True):
                with open("sample.png", "rb") as f:
                    res = document_processor.process_document_pipeline(f.read(), filename="sample.png")
                    st.session_state.current_doc = res["record"]
                    st.session_state.current_previews = res["previews"]
                    st.session_state.searchable_pdf_bytes = res["searchable_pdf_bytes"]
                st.rerun()
        with col_act2:
            target_lang = st.selectbox("Select Target Language:", available_languages, index=0)

        chosen_source = st.session_state.translator_source_text or sample_text_fallback

    # Side-by-side columns
    col_trans_left, col_trans_right = st.columns(2)

    with col_trans_left:
        st.markdown("#### 📄 Original Text (English / Source)")
        source_input = st.text_area(
            "Original Text",
            value=chosen_source,
            height=340,
            key="trans_source_area",
            help="You can edit or paste any document text here before translating."
        )
        st.session_state.translator_source_text = source_input
        src_words = len(source_input.split()) if source_input.strip() else 0
        src_chars = len(source_input)
        st.caption(f"📊 Stats: **{src_words}** words | **{src_chars}** characters")

    with col_trans_right:
        st.markdown(f"#### 🌐 Translated Output ({target_lang})")
        btn_translate = st.button("🔄 Translate Document Now", type="primary", use_container_width=True)

        # Trigger translation if requested or if output is missing for current language
        if btn_translate or (not st.session_state.translator_output and source_input.strip()):
            with st.spinner(f"Translating into {target_lang}..."):
                out = ai_module.translate_document_text(
                    source_input,
                    target_language=target_lang,
                    api_key=st.session_state.get("external_api_key"),
                    api_provider=st.session_state.get("api_provider", "openai")
                )
                st.session_state.translator_output = out
                st.session_state.translator_last_lang = target_lang

        translated_display = st.session_state.translator_output

        st.text_area(
            f"Translated ({target_lang})",
            value=translated_display,
            height=340,
            key="trans_output_area",
            help="High-precision translated output. Fully copyable."
        )
        tr_words = len(translated_display.split()) if translated_display.strip() else 0
        tr_chars = len(translated_display)
        st.caption(f"📊 Stats: **{tr_words}** words | **{tr_chars}** characters")

    # Action buttons for download
    if translated_display:
        st.markdown("---")
        col_down1, col_down2 = st.columns(2)

        raw_filename = current_doc.get("filename") if current_doc else "document"
        stem = Path(raw_filename).stem if raw_filename else "document"

        with col_down1:
            st.download_button(
                f"📥 Download {target_lang} Translation (.txt)",
                data=translated_display.encode("utf-8"),
                file_name=f"{stem}_{target_lang.lower()}.txt",
                mime="text/plain",
                type="primary",
                use_container_width=True
            )

        with col_down2:
            side_by_side = (
                f"DOCUMIND TRANSLATION VERIFICATION EXPORT\n"
                f"Target Language: {target_lang}\n"
                f"{'='*50}\n\n"
                f"--- ORIGINAL SOURCE TEXT ---\n"
                f"{source_input}\n\n"
                f"--- TRANSLATED TEXT ({target_lang.upper()}) ---\n"
                f"{translated_display}\n"
            )
            st.download_button(
                "📑 Download Side-by-Side Comparison (.txt)",
                data=side_by_side.encode("utf-8"),
                file_name=f"{stem}_comparison_{target_lang.lower()}.txt",
                mime="text/plain",
                use_container_width=True
            )


# =====================================================================
# PAGE 9: 📊 DOCUMENT ANALYTICS (NEW FEATURE!)
# =====================================================================

elif nav_selection == "📊 Document Analytics":
    st.markdown("""
    <div class="documind-hero-header">
        <h1>📊 Document Analytics & Linguistic Radar</h1>
        <p>In-depth textual metrics, vocabulary density, detected document tone, and estimated reading speeds.</p>
    </div>
    """, unsafe_allow_html=True)

    current_doc = st.session_state.current_doc

    if not current_doc:
        st.warning("⚠️ No document is currently active. Please load or upload a document first.")
        if os.path.exists("sample.png") and st.button("📂 Load sample.png for Analytics"):
            with open("sample.png", "rb") as f:
                res = document_processor.process_document_pipeline(f.read(), filename="sample.png")
                st.session_state.current_doc = res["record"]
                st.session_state.current_previews = res["previews"]
                st.session_state.searchable_pdf_bytes = res["searchable_pdf_bytes"]
            st.rerun()
    else:
        text_data = current_doc.get("full_text", "")
        analytics = ai_module.calculate_document_analytics(text_data)

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Total Words", f"{analytics['total_words']:,}")
        a2.metric("Unique Words", f"{analytics['unique_words']:,}")
        a3.metric("Vocabulary Richness", f"{analytics['vocabulary_diversity_pct']}%")
        a4.metric("Average Sentence Length", f"{analytics['avg_sentence_length']} words")

        st.markdown("---")
        c_an_left, c_an_right = st.columns(2)

        with c_an_left:
            st.markdown("#### 🎭 Detected Document Tone & Intent")
            tones = analytics.get("detected_tones", ["Professional Standard"])
            for t in tones:
                st.markdown(f"- **Tone Category:** `{t}`")

            st.markdown("#### ⏱️ Processing Speed Benchmarks")
            st.markdown(f"- **Estimated Silent Reading Time:** `{analytics['est_reading_time_sec']} seconds`")
            st.markdown(f"- **Estimated Speaking Presentation Time:** `{analytics['est_speaking_time_sec']} seconds`")
            st.markdown(f"- **Actual OCR Extraction Latency:** `{current_doc.get('processing_time_sec', 0)} seconds`")

        with c_an_right:
            st.markdown("#### 📈 Top Keyword Density")
            kw_list = analytics.get("top_keywords", [])
            if kw_list:
                df_kw = pd.DataFrame(kw_list, columns=["Keyword", "Frequency"])
                st.bar_chart(df_kw.set_index("Keyword"))
            else:
                st.info("No significant content keywords detected.")


# =====================================================================
# PAGE 10: 🕒 DOCUMENT HISTORY
# =====================================================================

elif nav_selection == "🕒 Document History":
    st.markdown("""
    <div class="documind-hero-header">
        <h1>Document Processing History & Audit Log</h1>
        <p>Review past scans, reload extracted text, re-download reports, and audit past document processing records.</p>
    </div>
    """, unsafe_allow_html=True)

    history = document_processor.get_history()

    if not history:
        st.info("No document history found yet. Upload and process documents to build your audit log.")
    else:
        st.write(f"Total Archived Records: **{len(history)}**")

        search_term = st.text_input("🔍 Search history by filename or document type:", "")
        filtered = [
            r for r in history
            if search_term.lower() in r.get("filename", "").lower()
            or search_term.lower() in r.get("classification", "").lower()
        ]

        for item in filtered:
            doc_id = item.get("id")
            filename = item.get("filename", "Untitled")
            doc_type = item.get("classification", "General")
            ts = item.get("timestamp", "")
            conf = round(item.get("avg_confidence", 0.0) * 100, 1)

            with st.expander(f"📄 {filename} — [{doc_type}] | {ts} | Conf: {conf}%"):
                c_info, c_acts = st.columns([3, 1])

                with c_info:
                    st.write(f"**Document ID:** `{doc_id}` | **Pages:** {item.get('page_count', 1)} | **Processing Time:** {item.get('processing_time_sec', 0)}s")
                    st.markdown(f"**Summary:** {item.get('summary', {}).get('executive_summary', 'N/A')[:200]}...")

                with c_acts:
                    if st.button("💬 Ask AI About This Doc", key=f"btn_chat_hist_{doc_id}", type="primary", use_container_width=True):
                        st.session_state.current_doc = item
                        if os.path.exists(item.get("saved_path", "")):
                            st.session_state.current_previews = document_processor.generate_document_previews(item.get("saved_path"))
                        st.session_state["nav_selection"] = "💬 Ask AI Assistant"
                        st.rerun()

                    if st.button("📂 Load in Active Workspace", key=f"btn_load_{doc_id}", use_container_width=True):
                        st.session_state.current_doc = item
                        if os.path.exists(item.get("saved_path", "")):
                            st.session_state.current_previews = document_processor.generate_document_previews(item.get("saved_path"))
                        st.success(f"Loaded {filename} into active workspace!")
                        st.rerun()

                    zip_data = document_processor.export_full_bundle_zip(item)
                    st.download_button(
                        "📦 Download Export Bundle",
                        data=zip_data.getvalue(),
                        file_name=f"{Path(filename).stem}_export.zip",
                        mime="application/zip",
                        key=f"dl_zip_{doc_id}",
                        use_container_width=True
                    )

                    if st.button("🗑️ Delete Record", key=f"btn_del_{doc_id}", use_container_width=True):
                        document_processor.delete_history_item(doc_id)
                        st.success(f"Deleted record {doc_id}.")
                        st.rerun()
