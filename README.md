# DocuMind – AI-Powered Intelligent Document Processing System

DocuMind is an enterprise-grade document-scanning and intelligent document-processing web application built with Python, Streamlit, PyMuPDF, EasyOCR, Tesseract OCR, and a robust local AI intelligence engine.

---

## 🌟 Key Features

### 1. 📤 Document Ingestion & Multi-Format Previews
- **Format Support:** PDF documents, scanned documents, and images (JPG, JPEG, PNG, WEBP).
- **Batch Uploads:** Upload multiple documents simultaneously.
- **Interactive Multi-Page Previews:** Real-time page selector, high-resolution rendering, and visual inspection.

### 2. 🔍 Dual-Engine OCR Text Extraction
- **Smart Hybrid Extraction:** Instantly extracts native text from digital PDFs and automatically engages deep-learning OCR for scanned pages or images.
- **Dual Engine Options:**
  - **EasyOCR:** Bundled local deep-learning OCR requiring zero external system binaries.
  - **Tesseract OCR:** Industry-standard binary integration via `pytesseract` with automatic Windows path discovery.
- **Image Preprocessing:** CLAHE contrast enhancement, adaptive thresholding, non-local means denoising, and grayscale normalization.
- **Interactive Editor:** Extracted text is displayed in an editable area with character count, word count, and per-line confidence scores.
- **Multi-Format Export:** One-click download as Plain Text (`.txt`), Structured JSON (`.json`), Entity Table (`.csv`), or complete ZIP packages.

### 3. ⚙️ Document Conversion & Searchable PDF Creation
- **True Searchable PDF Generation:** Overlays an invisible, selectable text layer onto scanned images or PDFs (PDF specification `render_mode=3`), preserving 100% visual fidelity while enabling cursor selection and search.
- **PDF Pages to Images:** Render high-DPI individual PNG/JPG pages or download complete ZIP bundles.
- **Images to PDF:** Merge multiple JPG/PNG images into a single clean PDF document.
- **Image Optimization & Editing:** In-browser format conversion (JPG <-> PNG), 90°/180°/270° rotation, and aspect-ratio-preserving resizing.

### 4. 💬 Ask AI Document Assistant (Interactive Chat)
- **Conversational Document Dialogue:** Ask natural language questions in an interactive chat interface grounded directly in your uploaded document.
- **One-Click Quick Question Chips:** Instant prompts for summaries, financial breakdowns, deadlines, party identification, missing details, and contact info.
- **Split Workspace View:** Side-by-side view featuring an interactive document inspector (visual page viewer, raw text viewer, detected entities) alongside the live AI chat stream.
- **Evidence-Based Context Citations:** Every AI answer includes exact source quotes and page references for auditability and verification.
- **Multi-Turn Chat History:** Chat history is tracked in session state with one-click clear and copy options.

### 5. 🤖 Local AI Document Intelligence
- **100% Offline by Default:** All AI features run locally with zero mandatory external API keys.
- **Document Classification:** Automatically classifies documents into Invoices, Receipts, Contracts, Resumes, Bank Statements, Medical Records, or General Documents.
- **Executive Summarization:** Generates executive summaries, bulleted key takeaways, reading time estimates, and complexity scores.
- **Structured Field Extraction:**
  - Dates (Issue Date, Due Date, Effective Date, Expiration Date).
  - Monetary Amounts (Total Due, Subtotal, Tax/VAT, Discounts).
  - Reference Numbers (Invoice #, PO #, Order #, Tax ID, Account #).
  - Contact Information (Emails, Phone numbers, URLs).
  - Entities (People, Parties, Organizations).
- **Missing Information Detection:** Automatically audits documents against standard templates and flags missing mandatory fields with severity levels.
- **Inconsistency & Anomaly Detection:** Identifies chronological contradictions (e.g., due date before invoice date), mathematical mismatches (subtotal > total), and OCR noise.
- **Optional Cloud AI:** Seamlessly integrates with OpenAI (`gpt-4o-mini`) or Google Gemini (`gemini-1.5-flash`) if an optional API key is provided.

### 6. 🔒 Document Privacy & PII Redactor
- **Automated PII Detection:** Scans text for sensitive Personally Identifiable Information: emails, phone numbers, tax IDs, account numbers, and monetary figures.
- **Selective Redaction:** Toggle rules on/off to selectively mask specific PII categories with standard clean tokens (`[REDACTED EMAIL]`, `[REDACTED PHONE]`, etc.).
- **Sanitized Export:** Download sanitized text files ready for safe public or internal sharing.

### 7. 🌐 Multi-Language Document Translator
- **Multi-Language Translation:** Translate document content, executive summaries, full texts, or custom excerpts into Spanish, French, German, Italian, Hindi, Portuguese, Russian, Arabic, Japanese, and Chinese.
- **Side-by-Side Verification:** Compare original source text against the translated text in real time with side-by-side verification and one-click TXT export.

### 8. 📊 Document Analytics & Linguistic Radar
- **Textual Metrics:** Vocabulary richness percentage, total/unique word counts, average sentence length.
- **Detected Intent & Tone:** Automatically classifies tone (Financial/Commercial, Formal Legal, Urgent/Notice, Medical/Clinical, Professional Standard).
- **Keyword Density Chart:** Interactive frequency bar chart of top document keywords.
- **Speed Benchmarks:** Estimates silent reading time and presentation speaking time.

### 9. 🕒 Document History & Audit Log
- **Persistent Storage:** Scanned documents, OCR outputs, and AI intelligence reports are automatically saved in `outputs/history.json`.
- **Search & Reload:** Search history by filename or document type, reload any past document back into the active workspace, or download archived reports.

---

## 📁 Project Structure

```
DocuMind/
├── app.py                  # Main Streamlit web application & modern dashboard
├── ocr_module.py           # Dual-engine OCR (EasyOCR + Tesseract) & PDF text extractor
├── converter.py            # Image/PDF converters, transformations & Searchable PDF generator
├── converter_module.py     # Backward-compatible adapter forwarding to converter.py
├── ai_module.py            # Local document intelligence, classification, summary & Q&A
├── document_processor.py   # Unified orchestrator, history management & file exports
├── test_converter.py       # Standalone test suite for converter routines
├── test_system.py          # Complete end-to-end verification test suite
├── requirements.txt        # Production dependency specifications
├── README.md               # Complete project documentation
├── uploads/                # Directory for ingested user documents
└── outputs/                # Directory for generated searchable PDFs and history.json
```

---

## 🚀 Quick Start Guide (Windows)

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed.

### 1. Clone or Open Project
```powershell
cd c:\github\docmind
```

### 2. Activate Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run System Verification Tests
```powershell
python test_system.py
```

### 5. Launch the DocuMind Application
```powershell
streamlit run app.py
```
DocuMind will automatically launch in your default web browser at `http://localhost:8501`.

---

## 🛠️ Error Fixes & Technical Improvements Implemented

1. **Empty `requirements.txt` Fixed:** Added complete dependency specifications for Streamlit, PyMuPDF, EasyOCR, Pillow, OpenCV, Pandas, and Pytesseract.
2. **Missing `sample.pdf` in `test_converter.py` Fixed:** Resolved `FileNotFoundError` by creating `sample.pdf` and adding self-healing fallback mechanisms in test scripts.
3. **PyMuPDF Legacy Deprecation Warning Fixed:** Replaced deprecated `import fitz` statements with `import pymupdf as fitz` and targeted warning filters to eliminate terminal pollution.
4. **CP1252 Windows Unicode Encoding Error Fixed:** Replaced raw unicode symbols in test printouts with clean ASCII indicators `[PASS]`, preventing `UnicodeEncodeError` on Windows consoles.
5. **EasyOCR Startup Latency Resolved:** Converted eager top-level model initialization into lazy, cached singleton loading, ensuring instant script execution.
6. **Searchable PDF Support Added:** Implemented `create_searchable_pdf` using PDF standard invisible text layer (`render_mode=3`), creating fully selectable and searchable PDFs while maintaining original visual fidelity.
7. **Intent Classification Refined:** Fixed token collision in Q&A intent classification so that queries like `"payment due date"` correctly resolve to date extraction rather than amount queries.
8. **Streamlit ScriptRunContext Safety:** Structured session state variables and explicit component keys to prevent state desynchronization and rerun loops.

---

## 📄 License
MIT License. Built for enterprise document scanning and intelligent processing.
