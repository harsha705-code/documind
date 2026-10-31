"""
test_system.py

Comprehensive System Verification Suite for DocuMind
Tests:
1. converter.py & converter_module.py
2. ocr_module.py (image OCR, PDF text extraction, enhancement)
3. ai_module.py (classification, summarization, entity extraction, missing info, anomalies, Q&A)
4. document_processor.py (full pipeline, history persistence, exports: TXT, JSON, CSV, Searchable PDF, ZIP)
"""

import os
import io
import json
from pathlib import Path
from PIL import Image

import converter
import converter_module
import ocr_module
import ai_module
import document_processor


def test_converter():
    print("Testing converter module...")
    assert os.path.exists("sample.png"), "sample.png must exist"
    with open("sample.png", "rb") as f:
        jpg_buf = converter.convert_image(f, "JPG")
    assert jpg_buf.getvalue()[:2] == b"\xff\xd8", "JPG conversion failed"

    # Images to PDF
    pdf_buf = converter.image_to_pdf(["sample.png", "sample.png"])
    assert pdf_buf.getvalue().startswith(b"%PDF"), "PDF generation failed"

    # PDF to images
    pages = converter.pdf_to_images(pdf_buf)
    assert len(pages) == 2, f"Expected 2 pages, got {len(pages)}"

    # Searchable PDF
    spdf = converter.create_searchable_pdf([
        {"image": "sample.png", "boxes": [{"text": "DocuMind Test", "box": [[10, 10], [50, 10], [50, 30], [10, 30]]}]}
    ])
    assert spdf.getvalue().startswith(b"%PDF"), "Searchable PDF generation failed"
    print("[PASS] Converter tests passed!")


def test_ocr():
    print("Testing OCR module...")
    assert os.path.exists("sample.png")
    res = ocr_module.process_document("sample.png")
    assert res["status"] == "success", f"OCR failed: {res.get('error')}"
    assert len(res["full_text"]) > 20, "OCR extracted too little text"
    assert res["avg_confidence"] > 0.5, "Confidence score lower than expected"
    print(f"[PASS] OCR extracted {len(res['full_text'])} characters with avg confidence {res['avg_confidence']}.")


def test_ai():
    print("Testing AI module...")
    invoice_sample = (
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

    analysis = ai_module.analyze_document(invoice_sample)
    assert analysis["status"] == "success"
    assert analysis["classification"]["document_type"] == "Invoice"
    assert len(analysis["extracted_fields"]["amounts"]) >= 2
    assert len(analysis["extracted_fields"]["dates"]) >= 2
    assert "billing@globallogistics.com" in analysis["extracted_fields"]["contacts"]["emails"]

    # Q&A test
    qa_amt = ai_module.answer_question_local("What is the total amount?", invoice_sample)
    assert "$2,750.00" in qa_amt["answer"]

    qa_due = ai_module.answer_question_local("When is the payment due date?", invoice_sample)
    assert "2026-05-30" in qa_due["answer"]

    # Test master ask_ai_question dispatcher
    qa_sum = ai_module.ask_ai_question("Summarize this document", invoice_sample)
    assert "Invoice" in qa_sum["answer"]
    assert len(qa_sum.get("suggested_followups", [])) > 0

    qa_parties = ai_module.ask_ai_question("Who are the parties involved?", invoice_sample)
    assert "Eleanor Vance" in qa_parties["answer"] or "Global Logistics" in qa_parties["answer"]

    qa_audit = ai_module.ask_ai_question("Are there any missing items or risks?", invoice_sample)
    assert "Missing" in qa_audit["answer"] or "detected" in qa_audit["answer"]

    # Redaction test
    redacted_sample, counts = ai_module.redact_sensitive_information(invoice_sample)
    assert "[REDACTED EMAIL]" in redacted_sample
    assert counts["emails"] >= 1

    # Translation test
    trans_sample = ai_module.translate_document_text("Invoice Total Amount Due Date", "Spanish")
    assert "Factura" in trans_sample or "factura" in trans_sample.lower()

    # Analytics test
    analytics = ai_module.calculate_document_analytics(invoice_sample)
    assert analytics["total_words"] > 10
    assert "Financial / Commercial" in analytics["detected_tones"]

    print("[PASS] AI analysis, Ask AI, Redaction, Translation & Analytics tests passed!")


def test_document_processor():
    print("Testing unified document processor pipeline...")
    with open("sample.png", "rb") as f:
        pipeline_res = document_processor.process_document_pipeline(f.read(), filename="test_sample.png")

    rec = pipeline_res["record"]
    assert rec["id"] is not None
    assert rec["avg_confidence"] > 0
    assert len(pipeline_res["previews"]) >= 1
    assert pipeline_res["searchable_pdf_bytes"] is not None

    # Test exports
    txt_exp = document_processor.export_as_txt(rec["full_text"])
    assert len(txt_exp.getvalue()) > 0

    json_exp = document_processor.export_as_json(rec)
    parsed = json.loads(json_exp.getvalue().decode("utf-8"))
    assert parsed["document_id"] == rec["id"]

    csv_exp = document_processor.export_as_csv(rec)
    assert b"Category,Field,Value,Context" in csv_exp.getvalue()

    zip_exp = document_processor.export_full_bundle_zip(rec)
    assert zip_exp.getvalue()[:4] == b"PK\x03\x04"

    # Test history retrieval
    hist = document_processor.get_history()
    assert len(hist) > 0
    assert hist[0]["id"] == rec["id"]

    print("[PASS] Document Processor pipeline and exports passed!")


if __name__ == "__main__":
    print("========================================")
    print("Running Complete DocuMind Test Suite...")
    print("========================================")
    test_converter()
    test_ocr()
    test_ai()
    test_document_processor()
    print("========================================")
    print("ALL SYSTEM VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("========================================")
