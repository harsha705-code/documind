import re
import io
from datetime import datetime

import fitz
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from rapidocr_onnxruntime import RapidOCR


st.set_page_config(
    page_title="DocuMind | Intelligent Document Scanner",
    page_icon="📄",
    layout="wide",
)

st.markdown("""
<style>
    .main {background-color: #f7f9fc;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.4rem 1.6rem; border-radius: 16px;
        background: linear-gradient(120deg, #14213d, #315b96);
        color: white; margin-bottom: 1.2rem;
    }
    .hero h1 {color: white; margin: 0;}
    .hero p {color: #e7eefc; margin: .35rem 0 0 0;}
    .small-note {color: #64748b; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>📄 DocuMind</h1>
  <p>Intelligent document scanning, OCR, field extraction and document insights</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_ocr():
    return RapidOCR()

def image_to_text(image, engine):
    """Run OCR on a PIL image and return readable text."""
    rgb = image.convert("RGB")
    array = np.asarray(rgb)
    result, _ = engine(array)
    if not result:
        return ""
    lines = []
    for item in result:
        if len(item) >= 2 and str(item[1]).strip():
            lines.append(str(item[1]).strip())
    return "\n".join(lines)

def pdf_to_text(file_bytes, engine):
    """Extract embedded PDF text first; OCR scanned pages when needed."""
    document = fitz.open(stream=file_bytes, filetype="pdf")
    page_texts = []
    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text").strip()
        if len(text) < 25:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            page_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = image_to_text(page_image, engine)
        page_texts.append(f"--- Page {page_number} ---\n{text}".strip())
    document.close()
    return "\n\n".join(page_texts)

def extract_fields(text):
    patterns = {
        "Email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "Phone": r"(?<!\d)(?:\+?\d{1,3}[- ]?)?[6-9]\d{9}(?!\d)",
        "Date": r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b",
        "Invoice number": r"(?i)\b(?:invoice\s*(?:no|number|#)|inv\s*#?)\s*[:\-]?\s*([A-Z0-9/-]+)",
        "GSTIN": r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b",
        "PAN": r"\b[A-Z]{5}\d{4}[A-Z]\b",
        "Amount": r"(?i)\b(?:grand\s*total|total\s*amount|amount\s*due|total)\s*[:\-]?\s*(?:₹|rs\.?|inr)?\s*([0-9,]+(?:\.\d{1,2})?)",
    }
    found = {}
    for label, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            first = matches[0]
            found[label] = first if isinstance(first, str) else str(first)
    return found

def detect_issues(text, fields, document_type):
    issues = []
    lowered = text.lower()

    required = {
        "Invoice": ["Invoice number", "Date", "Amount"],
        "Certificate": ["Date"],
        "Form": ["Date"],
        "Report": [],
        "General document": [],
    }
    for field in required.get(document_type, []):
        if field not in fields:
            issues.append(f"Could not detect the {field.lower()}. It may be missing or unreadable.")

    if len(text.strip()) < 40:
        issues.append("Very little text was detected. Try a clearer scan or a higher-resolution image.")

    dates = re.findall(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b", text)
    if len(set(dates)) > 1:
        issues.append("More than one date was found. Check whether the dates are consistent.")

    amounts = re.findall(r"(?i)(?:₹|rs\.?|inr)\s*([0-9,]+(?:\.\d{1,2})?)", text)
    if len(set(amounts)) > 1:
        issues.append("Multiple currency amounts were found. Verify the totals manually.")

    if "signature" in lowered and not re.search(r"(?i)signature.{0,30}\S", text):
        issues.append("A signature label appears, but no signature text could be read by OCR.")

    return issues

def make_summary(text, max_sentences=4):
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return "No readable text was detected."
    parts = re.split(r"(?<=[.!?])\s+", clean)
    useful = [part for part in parts if len(part.strip()) > 20]
    if not useful:
        return clean[:650] + ("..." if len(clean) > 650 else "")
    summary = " ".join(useful[:max_sentences])
    return summary[:900] + ("..." if len(summary) > 900 else "")

def process_upload(uploaded_file, engine):
    data = uploaded_file.getvalue()
    name = uploaded_file.name
    suffix = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    if suffix == "pdf":
        text = pdf_to_text(data, engine)
        pages = len(fitz.open(stream=data, filetype="pdf"))
    elif suffix in ("png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"):
        image = Image.open(io.BytesIO(data))
        text = image_to_text(image, engine)
        pages = 1
    else:
        raise ValueError("Unsupported file type. Upload a PDF or an image.")
    return text, pages

with st.sidebar:
    st.header("⚙️ Document settings")
    document_type = st.selectbox(
        "Document type",
        ["General document", "Invoice", "Certificate", "Form", "Report"],
    )
    st.caption("DocuMind uses local OCR. Your uploaded files are processed by this app and are not sent to a Gemini API.")
    st.divider()
    st.caption("Supported: PDF, PNG, JPG, JPEG, BMP, TIFF and WEBP")

uploaded_files = st.file_uploader(
    "Upload document(s)",
    type=["pdf", "png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
    accept_multiple_files=True,
)

if uploaded_files:
    try:
        with st.spinner("Loading OCR engine for the first time..."):
            ocr_engine = load_ocr()

        all_rows = []
        for uploaded_file in uploaded_files:
            st.divider()
            st.subheader(f"📑 {uploaded_file.name}")
            try:
                with st.spinner(f"Scanning {uploaded_file.name}..."):
                    extracted_text, page_count = process_upload(uploaded_file, ocr_engine)

                fields = extract_fields(extracted_text)
                issues = detect_issues(extracted_text, fields, document_type)
                summary = make_summary(extracted_text)

                col1, col2, col3 = st.columns(3)
                col1.metric("Pages", page_count)
                col2.metric("Characters extracted", len(extracted_text))
                col3.metric("Fields detected", len(fields))

                tab1, tab2, tab3, tab4 = st.tabs(
                    ["📃 Extracted text", "🔎 Key information", "⚠️ Checks", "📝 Summary"]
                )

                with tab1:
                    if extracted_text.strip():
                        st.text_area("OCR / extracted content", extracted_text, height=300, key=f"text_{uploaded_file.name}")
                        st.download_button(
                            "Download extracted text",
                            data=extracted_text,
                            file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_text.txt",
                            mime="text/plain",
                            key=f"txt_{uploaded_file.name}",
                        )
                    else:
                        st.warning("No readable text found in this file.")

                with tab2:
                    if fields:
                        field_df = pd.DataFrame(
                            [{"Field": key, "Detected value": value} for key, value in fields.items()]
                        )
                        st.dataframe(field_df, use_container_width=True, hide_index=True)
                        st.download_button(
                            "Download extracted fields as CSV",
                            data=field_df.to_csv(index=False).encode("utf-8"),
                            file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_fields.csv",
                            mime="text/csv",
                            key=f"csv_{uploaded_file.name}",
                        )
                    else:
                        st.info("No standard fields were detected. The complete extracted text is available in the first tab.")

                with tab3:
                    if issues:
                        for issue in issues:
                            st.warning(issue)
                    else:
                        st.success("No automatic checks were flagged.")
                    st.caption("These are automated checks, not proof that the document is correct. Verify important details manually.")

                with tab4:
                    st.write(summary)

                all_rows.append({
                    "File": uploaded_file.name,
                    "Pages": page_count,
                    "Characters": len(extracted_text),
                    "Fields detected": len(fields),
                    "Issues": len(issues),
                    "Summary": summary,
                })

            except Exception as file_error:
                st.error(f"Could not process {uploaded_file.name}: {file_error}")

        if all_rows:
            st.divider()
            st.header("📊 Batch processing report")
            report = pd.DataFrame(all_rows)
            st.dataframe(report, use_container_width=True, hide_index=True)
            st.download_button(
                "Download batch report",
                data=report.to_csv(index=False).encode("utf-8"),
                file_name="documind_batch_report.csv",
                mime="text/csv",
            )
    except Exception as error:
        st.error(f"DocuMind could not start: {error}")
        st.info("Check that all packages in requirements.txt are installed in the same Python environment used to run Streamlit.")
else:
    st.info("Upload one or more PDF or image files to start scanning.")
    st.markdown("""
    **Prototype features**
    - OCR for scanned images and scanned PDF pages
    - Direct text extraction from normal (selectable-text) PDFs
    - Automatic detection of common fields such as dates, email, phone, GSTIN, PAN and invoice amount
    - Basic missing-field and consistency checks
    - Short document summary
    - TXT, CSV and batch-report downloads
    """)
