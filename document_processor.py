"""
document_processor.py

DocuMind - Unified Document Processing & Management Engine

Responsibilities:
1. Ingestion, safe saving, and file validation into uploads/ directory.
2. Pipeline orchestration:
   - Page rendering & preview generation
   - OCR & hybrid text extraction
   - AI document intelligence (classification, summary, entities, anomalies, Q&A)
   - Searchable PDF generation
3. Document history persistence in outputs/history.json.
4. Export generation: Plain Text, Structured JSON, CSV, Searchable PDF, and ZIP bundles.
"""

import os
import io
import json
import time
import uuid
import hashlib
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd

from converter import (
    convert_image,
    image_to_pdf,
    pdf_to_images,
    rotate_image,
    resize_image,
    create_searchable_pdf,
    create_zip_archive,
    _read_file,
    _open_image,
)
from ocr_module import (
    process_document as ocr_process_document,
    get_available_ocr_engines,
    enhance_image,
)
from ai_module import (
    analyze_document,
    answer_question_local,
    classify_document,
    generate_summary,
)

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
HISTORY_FILE = OUTPUTS_DIR / "history.json"

# Ensure directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# 1. HISTORY MANAGEMENT
# =====================================================================

def _load_history() -> List[Dict[str, Any]]:
    """Load history records from disk."""
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_history(records: List[Dict[str, Any]]) -> None:
    """Save history records to disk."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
    except Exception as err:
        print(f"Warning: Failed to save history: {err}")


def get_history() -> List[Dict[str, Any]]:
    """Return all historical document processing records."""
    return _load_history()


def get_document_by_id(doc_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single document record by unique ID."""
    history = _load_history()
    for rec in history:
        if rec.get("id") == doc_id:
            return rec
    return None


def delete_history_item(doc_id: str) -> bool:
    """Delete a document record and associated cached files."""
    history = _load_history()
    updated = [r for r in history if r.get("id") != doc_id]
    if len(updated) != len(history):
        _save_history(updated)
        return True
    return False


def clear_history() -> None:
    """Clear all stored history records."""
    _save_history([])


# =====================================================================
# 2. FILE INGESTION & STORAGE
# =====================================================================

def save_uploaded_file(file_obj: Any) -> Tuple[str, str, int]:
    """
    Safely write an uploaded file into uploads/ and return (doc_id, file_path, file_size).
    """
    doc_id = str(uuid.uuid4())[:8]
    orig_name = getattr(file_obj, "name", f"doc_{doc_id}.pdf")
    # Clean filename
    safe_name = "".join(c for c in orig_name if c.isalnum() or c in "._- ")
    filename = f"{doc_id}_{safe_name}"
    target_path = UPLOADS_DIR / filename

    data = _read_file(file_obj)
    with open(target_path, "wb") as f:
        f.write(data)

    file_size = len(data)
    return doc_id, str(target_path), file_size


# =====================================================================
# 3. PREVIEW GENERATION
# =====================================================================

def generate_document_previews(
    file_path: Union[str, Path],
    max_pages: int = 10
) -> List[io.BytesIO]:
    """
    Generate image previews for display in UI (works for both images and multi-page PDFs).
    """
    path_str = str(file_path).lower()
    previews = []

    if path_str.endswith(".pdf"):
        with open(file_path, "rb") as f:
            all_pages = pdf_to_images(f, dpi_scale=1.5, image_format="PNG")
            previews = all_pages[:max_pages]
    else:
        # Single image
        with open(file_path, "rb") as f:
            buf = io.BytesIO(f.read())
            buf.seek(0)
            previews.append(buf)

    return previews


# =====================================================================
# 4. MASTER PROCESSING PIPELINE
# =====================================================================

def process_document_pipeline(
    file_input: Any,
    filename: Optional[str] = None,
    ocr_engine: str = "auto",
    preprocess_method: str = "contrast",
    pdf_mode: str = "auto",
    external_api_key: Optional[str] = None,
    api_provider: str = "openai"
) -> Dict[str, Any]:
    """
    Run the complete intelligent document pipeline:
    1. Ingestion & storage
    2. OCR & text extraction
    3. Document Preview generation
    4. AI document intelligence & field extraction
    5. Searchable PDF generation
    6. Record persistence in history
    """
    start_time = time.time()
    raw_bytes = _read_file(file_input)

    # Derive filename
    if filename is None:
        filename = getattr(file_input, "name", "document.pdf")

    doc_id = str(uuid.uuid4())[:8]
    safe_name = "".join(c for c in filename if c.isalnum() or c in "._- ")
    saved_filename = f"{doc_id}_{safe_name}"
    saved_file_path = UPLOADS_DIR / saved_filename

    with open(saved_file_path, "wb") as f:
        f.write(raw_bytes)

    # 1. Run OCR / Text Extraction
    ocr_results = ocr_process_document(
        raw_bytes,
        engine=ocr_engine,
        preprocess=preprocess_method,
        pdf_mode=pdf_mode,
    )

    full_text = ocr_results.get("full_text", "")
    pages = ocr_results.get("pages", [])
    avg_conf = ocr_results.get("avg_confidence", 0.0)

    # 2. Generate Image Previews
    previews = generate_document_previews(saved_file_path, max_pages=15)

    # 3. Run AI Document Intelligence
    ai_results = analyze_document(
        full_text,
        pages=pages,
        external_api_key=external_api_key,
        api_provider=api_provider,
    )

    # 4. Prepare Searchable PDF
    searchable_pdf_bytes = None
    searchable_pdf_path = None
    try:
        # Build images_with_ocr list
        images_with_ocr = []
        is_pdf = filename.lower().endswith(".pdf") or raw_bytes.startswith(b"%PDF")

        if is_pdf:
            # For PDF, use rendered preview images matched with each page's OCR boxes
            for p_idx, p_data in enumerate(pages):
                if p_idx < len(previews):
                    img_buf = previews[p_idx]
                    img_buf.seek(0)
                    images_with_ocr.append({
                        "image": img_buf,
                        "boxes": p_data.get("confidence", []),
                        "full_text": p_data.get("text", "")
                    })
        else:
            # Single image
            images_with_ocr.append({
                "image": raw_bytes,
                "boxes": pages[0].get("confidence", []) if pages else [],
                "full_text": full_text
            })

        if images_with_ocr:
            spdf_buf = create_searchable_pdf(images_with_ocr)
            searchable_pdf_bytes = spdf_buf.getvalue()
            searchable_filename = f"searchable_{doc_id}_{Path(filename).stem}.pdf"
            searchable_pdf_path = str(OUTPUTS_DIR / searchable_filename)
            with open(searchable_pdf_path, "wb") as sf:
                sf.write(searchable_pdf_bytes)
    except Exception as err:
        print(f"Notice: Searchable PDF creation skipped: {err}")

    processing_time = round(time.time() - start_time, 2)

    # Build Master Record
    record = {
        "id": doc_id,
        "filename": filename,
        "saved_path": str(saved_file_path),
        "searchable_pdf_path": searchable_pdf_path,
        "file_size": len(raw_bytes),
        "page_count": len(pages),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "processing_time_sec": processing_time,
        "ocr_engine": ocr_engine,
        "avg_confidence": avg_conf,
        "full_text": full_text,
        "pages": pages,
        "ai_results": ai_results,
        "classification": ai_results.get("classification", {}).get("document_type", "General Document"),
        "summary": ai_results.get("summary", {}),
        "extracted_fields": ai_results.get("extracted_fields", {}),
        "missing_information": ai_results.get("missing_information", []),
        "inconsistencies": ai_results.get("inconsistencies", []),
    }

    # Save to history
    existing_history = _load_history()
    # Prepend newest record
    existing_history.insert(0, record)
    # Keep max 50 recent records
    _save_history(existing_history[:50])

    return {
        "record": record,
        "previews": previews,
        "searchable_pdf_bytes": searchable_pdf_bytes,
    }


# =====================================================================
# 5. EXPORT GENERATION
# =====================================================================

def export_as_txt(full_text: str) -> io.BytesIO:
    """Generate a clean TXT buffer."""
    buf = io.BytesIO()
    buf.write(full_text.encode("utf-8"))
    buf.seek(0)
    return buf


def export_as_json(record: Dict[str, Any]) -> io.BytesIO:
    """Export complete document intelligence record as JSON."""
    export_payload = {
        "document_id": record.get("id"),
        "filename": record.get("filename"),
        "processed_at": record.get("timestamp"),
        "classification": record.get("classification"),
        "metrics": {
            "page_count": record.get("page_count"),
            "word_count": len(record.get("full_text", "").split()),
            "character_count": len(record.get("full_text", "")),
            "confidence_score": record.get("avg_confidence"),
            "processing_time_sec": record.get("processing_time_sec"),
        },
        "summary": record.get("summary"),
        "extracted_fields": record.get("extracted_fields"),
        "missing_information": record.get("missing_information"),
        "inconsistencies": record.get("inconsistencies"),
        "full_text": record.get("full_text"),
    }
    buf = io.BytesIO()
    buf.write(json.dumps(export_payload, indent=2, ensure_ascii=False).encode("utf-8"))
    buf.seek(0)
    return buf


def export_as_csv(record: Dict[str, Any]) -> io.BytesIO:
    """Export extracted fields as a structured CSV table."""
    rows = []
    fields = record.get("extracted_fields", {})

    # Dates
    for d in fields.get("dates", []):
        rows.append({"Category": "Date", "Field": d.get("label"), "Value": d.get("value"), "Context": d.get("snippet")})

    # Amounts
    for a in fields.get("amounts", []):
        rows.append({"Category": "Amount", "Field": a.get("label"), "Value": a.get("raw"), "Context": a.get("snippet")})

    # Reference IDs
    for r in fields.get("reference_ids", []):
        rows.append({"Category": "Reference ID", "Field": r.get("label"), "Value": r.get("value"), "Context": ""})

    # Contacts
    contacts = fields.get("contacts", {})
    for email in contacts.get("emails", []):
        rows.append({"Category": "Contact", "Field": "Email", "Value": email, "Context": ""})
    for phone in contacts.get("phones", []):
        rows.append({"Category": "Contact", "Field": "Phone", "Value": phone, "Context": ""})

    # Entities
    entities = fields.get("entities", {})
    for name in entities.get("names", []):
        rows.append({"Category": "Entity", "Field": "Name / Person", "Value": name, "Context": ""})
    for org in entities.get("organizations", []):
        rows.append({"Category": "Entity", "Field": "Organization", "Value": org, "Context": ""})

    if not rows:
        rows.append({"Category": "Document", "Field": "Full Text", "Value": record.get("full_text", "")[:200], "Context": ""})

    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


def export_full_bundle_zip(record: Dict[str, Any]) -> io.BytesIO:
    """Export complete analysis package as a ZIP archive."""
    base_name = Path(record.get("filename", "doc")).stem
    files = {
        f"{base_name}_extracted_text.txt": export_as_txt(record.get("full_text", "")),
        f"{base_name}_analysis_report.json": export_as_json(record),
        f"{base_name}_extracted_entities.csv": export_as_csv(record),
    }

    # If searchable PDF exists, add it to ZIP
    spdf_path = record.get("searchable_pdf_path")
    if spdf_path and os.path.exists(spdf_path):
        with open(spdf_path, "rb") as f:
            files[f"{base_name}_searchable.pdf"] = f.read()

    return create_zip_archive(files)


if __name__ == "__main__":
    print("DocuMind Document Processor Engine ready.")
    print(f"Uploads directory: {UPLOADS_DIR}")
    print(f"Outputs directory: {OUTPUTS_DIR}")
    print(f"Current history records: {len(get_history())}")
