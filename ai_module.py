"""
ai_module.py

DocuMind - Intelligent Document Analysis Engine

Features:
1. Document Type Classification (Invoice, Receipt, Contract, Resume, Bank Statement,
   Report, ID Document, Generic Document).
2. Document Summarization (Executive summary, bulleted key takeaways, reading metrics).
3. Structured Field & Entity Extraction:
   - Dates (formatted, labeled by role: issue date, due date, effective date)
   - Amounts & Currencies (totals, sub-totals, tax, monetary values)
   - Names (parties, individuals, vendors, clients)
   - Organizations & Companies
   - Identifiers (Invoice #, PO #, Tax ID, Account #, Phone, Email, URLs)
4. Missing Information Detection (document-type-specific completeness checks).
5. Inconsistency & Anomaly Detection (date chronological contradictions, math mismatches,
   unusual value flags, low-confidence transcription risks).
6. Local Document Q&A Engine (instant semantic similarity matching with exact context citation).
7. Optional External API Integration (OpenAI / Claude / Gemini API support when keys are provided).
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from datetime import datetime


# =====================================================================
# 1. DOCUMENT CLASSIFICATION & STRUCTURAL ANALYSIS
# =====================================================================

DOC_TYPE_SIGNATURES = {
    "Invoice": [
        "invoice", "bill to", "invoice date", "due date", "subtotal", "total amount",
        "tax invoice", "balance due", "item description", "qty", "unit price", "remit to"
    ],
    "Receipt": [
        "receipt", "cashier", "merchant", "store", "tax", "change due", "subtotal",
        "visa", "mastercard", "card #", "payment method", "register", "thank you for shopping"
    ],
    "Contract / Agreement": [
        "agreement", "contract", "parties", "hereby", "terms and conditions",
        "whereas", "governing law", "termination", "confidentiality", "indemnification",
        "signed by", "witness", "in witness whereof"
    ],
    "Resume / CV": [
        "curriculum vitae", "resume", "experience", "education", "skills",
        "work experience", "employment", "bachelor", "master", "university",
        "certifications", "projects", "profile", "summary"
    ],
    "Bank / Financial Statement": [
        "statement of account", "bank statement", "account number", "opening balance",
        "closing balance", "deposits", "withdrawals", "debit", "credit", "transaction date",
        "interest earned", "account summary"
    ],
    "Medical / Healthcare": [
        "patient", "diagnosis", "doctor", "physician", "prescription", "rx",
        "hospital", "clinic", "medical record", "dosage", "treatment", "lab results"
    ],
    "Academic / Research": [
        "abstract", "introduction", "methodology", "references", "conclusion",
        "journal", "proceedings", "doi", "bibliography", "authors"
    ],
}


def classify_document(text: str) -> Dict[str, Any]:
    """
    Classify the document type based on structural keyword frequencies.
    """
    lower_text = text.lower()
    scores = {}

    for doc_type, keywords in DOC_TYPE_SIGNATURES.items():
        score = sum(1 for kw in keywords if kw in lower_text)
        scores[doc_type] = score

    best_match = max(scores, key=scores.get)
    max_score = scores[best_match]

    if max_score >= 2:
        confidence = min(0.95, round(0.5 + (max_score * 0.08), 2))
        assigned_type = best_match
    else:
        assigned_type = "General Document"
        confidence = 0.50

    return {
        "document_type": assigned_type,
        "confidence": confidence,
        "type_scores": scores,
    }


# =====================================================================
# 2. DOCUMENT SUMMARIZATION
# =====================================================================

def generate_summary(text: str, max_sentences: int = 4) -> Dict[str, Any]:
    """
    Generate an executive summary and key highlight bullets using extractive
    text ranking and frequency scoring.
    """
    cleaned = text.strip()
    if not cleaned:
        return {
            "executive_summary": "No text content available to summarize.",
            "bullet_points": [],
            "word_count": 0,
            "reading_time_minutes": 0,
            "complexity": "N/A"
        }

    # Split into sentences
    raw_sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 15]

    if not sentences:
        sentences = [cleaned[:250]]

    word_count = len(cleaned.split())
    reading_time = max(1, math.ceil(word_count / 200))

    # Determine vocabulary complexity
    words = re.findall(r"\b[a-zA-Z]{3,}\b", cleaned.lower())
    stop_words = {
        "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
        "have", "has", "had", "they", "their", "will", "would", "about", "there",
        "which", "when", "what", "where", "who", "your", "into", "more", "other"
    }
    filtered_words = [w for w in words if w not in stop_words]
    word_freq = Counter(filtered_words)

    # Score sentences based on keyword prominence and position
    sentence_scores = []
    for idx, sent in enumerate(sentences):
        sent_words = re.findall(r"\b[a-zA-Z]{3,}\b", sent.lower())
        score = sum(word_freq.get(w, 0) for w in sent_words)
        # Position boost for opening and concluding sentences
        if idx == 0:
            score *= 1.3
        elif idx == len(sentences) - 1:
            score *= 1.15
        sentence_scores.append((score, idx, sent))

    sentence_scores.sort(key=lambda x: x[0], reverse=True)
    top_picks = sorted(sentence_scores[:max_sentences], key=lambda x: x[1])

    summary_text = " ".join(item[2] for item in top_picks)

    # Extract distinct bullet highlights
    bullets = []
    for item in top_picks:
        s = item[2].strip()
        if s and not s.endswith("."):
            s += "."
        bullets.append(s)

    # Vocabulary complexity assessment
    avg_word_length = sum(len(w) for w in words) / max(1, len(words))
    if avg_word_length > 6.0:
        complexity = "Technical / Complex"
    elif avg_word_length > 4.8:
        complexity = "Moderate"
    else:
        complexity = "Standard / Simple"

    return {
        "executive_summary": summary_text,
        "bullet_points": bullets,
        "word_count": word_count,
        "reading_time_minutes": reading_time,
        "complexity": complexity,
    }


# =====================================================================
# 3. STRUCTURED ENTITY & FIELD EXTRACTION
# =====================================================================

def extract_dates(text: str) -> List[Dict[str, Any]]:
    """
    Extract dates in multiple formats and tag potential labels (e.g., due date, invoice date).
    """
    date_patterns = [
        # YYYY-MM-DD or YYYY/MM/DD
        (r"\b(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})\b", "%Y-%m-%d"),
        # DD-MM-YYYY or DD/MM/YYYY
        (r"\b(\d{1,2}[-/.]\d{1,2}[-/.]\d{4})\b", "%d-%m-%Y"),
        # Month Day, Year (e.g. January 15, 2025 or Jan 15 2025)
        (r"\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4})\b", "%B %d %Y"),
        # Day Month Year (e.g. 15 January 2025 or 15 Jan 2025)
        (r"\b(\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\b", "%d %B %Y"),
    ]

    found = []
    seen = set()

    lines = text.split("\n")
    for line in lines:
        for pat, _ in date_patterns:
            matches = re.finditer(pat, line, re.IGNORECASE)
            for m in matches:
                raw_val = m.group(1).strip()
                if raw_val in seen:
                    continue
                seen.add(raw_val)

                # Determine context label
                context_label = "Date"
                lower_line = line.lower()
                if "due" in lower_line:
                    context_label = "Due Date"
                elif "invoice" in lower_line or "bill" in lower_line:
                    context_label = "Invoice Date"
                elif "effective" in lower_line:
                    context_label = "Effective Date"
                elif "expire" in lower_line or "expiration" in lower_line:
                    context_label = "Expiration Date"
                elif "issue" in lower_line or "issued" in lower_line:
                    context_label = "Issue Date"
                elif "delivery" in lower_line:
                    context_label = "Delivery Date"
                elif "birth" in lower_line or "dob" in lower_line:
                    context_label = "Date of Birth"

                found.append({
                    "value": raw_val,
                    "label": context_label,
                    "snippet": line.strip()[:100]
                })

    return found


def extract_amounts(text: str) -> List[Dict[str, Any]]:
    """
    Extract monetary values with currency symbols and contextual labeling.
    """
    amount_patterns = [
        # Currency symbol followed by digits: $1,234.56, €50.00, £99, ₹1,500
        r"([$€£₹¥]\s*[\d,]+(?:\.\d{2})?)",
        # Digits followed by currency code or word: 1200.00 USD, 500 EUR, 25000 INR
        r"(\b[\d,]+(?:\.\d{2})?\s*(?:USD|EUR|GBP|INR|CAD|AUD)\b)",
    ]

    found = []
    seen = set()
    lines = text.split("\n")

    for line in lines:
        for pat in amount_patterns:
            matches = re.finditer(pat, line, re.IGNORECASE)
            for m in matches:
                raw_val = m.group(1).strip()
                # Clean numeric value
                num_match = re.search(r"[\d,]+(?:\.\d{2})?", raw_val)
                if not num_match:
                    continue
                num_str = num_match.group(0).replace(",", "")
                try:
                    num_val = float(num_str)
                except ValueError:
                    continue

                if raw_val in seen or num_val == 0:
                    continue
                seen.add(raw_val)

                # Label detection
                lower_line = line.lower()
                label = "Amount"
                if "total" in lower_line and "sub" not in lower_line:
                    label = "Total Amount"
                elif "subtotal" in lower_line or "sub-total" in lower_line:
                    label = "Subtotal"
                elif "tax" in lower_line or "vat" in lower_line or "gst" in lower_line:
                    label = "Tax / VAT"
                elif "discount" in lower_line:
                    label = "Discount"
                elif "balance" in lower_line:
                    label = "Balance Due"
                elif "shipping" in lower_line or "freight" in lower_line:
                    label = "Shipping / Freight"

                found.append({
                    "raw": raw_val,
                    "numeric_value": num_val,
                    "label": label,
                    "snippet": line.strip()[:100]
                })

    # Sort so Total Amount is prominent
    found.sort(key=lambda x: (x["label"] != "Total Amount", -x["numeric_value"]))
    return found


def extract_contact_info(text: str) -> Dict[str, List[str]]:
    """
    Extract email addresses, telephone numbers, and web URLs.
    """
    # Emails
    emails = list(set(re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b", text)))

    # Phone numbers
    phone_candidates = re.findall(
        r"(?:(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4})",
        text
    )
    clean_phones = []
    for p in phone_candidates:
        digits_only = re.sub(r"\D", "", p)
        if 8 <= len(digits_only) <= 15:
            clean_phones.append(p.strip())
    clean_phones = list(set(clean_phones))

    # URLs
    urls = list(set(re.findall(r"\bhttps?://[^\s/$.?#].[^\s]*|www\.[^\s/$.?#].[^\s]*", text)))

    return {
        "emails": emails,
        "phones": clean_phones,
        "urls": urls,
    }


def extract_reference_ids(text: str) -> List[Dict[str, str]]:
    """
    Extract reference IDs, invoice numbers, purchase order numbers, etc.
    """
    patterns = [
        (r"(?:invoice\s*(?:no\.?|number|#)?\s*[:#-]?\s*([A-Za-z0-9\-_]{3,20}))", "Invoice Number"),
        (r"(?:po\s*(?:no\.?|number|#)?\s*[:#-]?\s*([A-Za-z0-9\-_]{3,20}))", "Purchase Order"),
        (r"(?:account\s*(?:no\.?|number|#)?\s*[:#-]?\s*([A-Za-z0-9\-_]{4,24}))", "Account Number"),
        (r"(?:order\s*(?:no\.?|number|#)?\s*[:#-]?\s*([A-Za-z0-9\-_]{3,20}))", "Order Number"),
        (r"(?:tax\s*(?:id|number|no\.?)?\s*[:#-]?\s*([A-Za-z0-9\-_]{5,20}))", "Tax ID / VAT / GST"),
        (r"(?:gstin|gst\s*no\.?)\s*[:#-]?\s*([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1})", "GSTIN"),
        (r"(?:ref(?:erence)?\s*(?:no\.?|number|#)?\s*[:#-]?\s*([A-Za-z0-9\-_]{3,20}))", "Reference Number"),
    ]

    results = []
    seen = set()

    for pat, label in patterns:
        matches = re.finditer(pat, text, re.IGNORECASE)
        for m in matches:
            val = m.group(1).strip()
            if val and val not in seen and len(val) >= 3:
                seen.add(val)
                results.append({"label": label, "value": val})

    return results


def extract_entities_and_names(text: str) -> Dict[str, List[str]]:
    """
    Extract plausible human names and organization names from text cues.
    """
    lines = text.split("\n")
    organizations = []
    person_names = []

    org_suffixes = [
        "inc", "inc.", "llc", "corp", "corporation", "ltd", "limited",
        "company", "co.", "enterprises", "solutions", "technologies",
        "group", "bank", "hospital", "institute", "university"
    ]

    for line in lines:
        trimmed = line.strip()
        if not trimmed or len(trimmed) > 80:
            continue

        lower = trimmed.lower()

        # Check for organization pattern
        for sfx in org_suffixes:
            if re.search(rf"\b{re.escape(sfx)}\b", lower):
                organizations.append(trimmed)
                break

        # Check for Name prefix
        name_match = re.search(
            r"\b(?:Mr\.|Ms\.|Mrs\.|Dr\.|Prof\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})",
            trimmed
        )
        if name_match:
            person_names.append(name_match.group(0))

        # Check for explicit "Bill To:", "From:", "Name:", "Client:"
        field_match = re.search(
            r"^(?:Bill\s*To|Customer|Client|Name|Attention|Attn|Sold\s*To)\s*[:]\s*([A-Za-z\s.,&'-]{3,40})",
            trimmed,
            re.IGNORECASE
        )
        if field_match:
            val = field_match.group(1).strip()
            if val and not any(kw in val.lower() for kw in ["invoice", "date", "address", "phone"]):
                person_names.append(val)

    # De-duplicate
    clean_orgs = list(dict.fromkeys(organizations))[:8]
    clean_names = list(dict.fromkeys(person_names))[:8]

    return {
        "organizations": clean_orgs,
        "names": clean_names,
    }


# =====================================================================
# 4. MISSING INFORMATION DETECTION
# =====================================================================

def detect_missing_information(
    text: str,
    doc_type: str,
    extracted_fields: Dict[str, Any]
) -> List[Dict[str, str]]:
    """
    Audit document completeness based on expected fields for its document type.
    """
    missing = []
    dates = extracted_fields.get("dates", [])
    amounts = extracted_fields.get("amounts", [])
    contacts = extracted_fields.get("contacts", {})
    ref_ids = extracted_fields.get("reference_ids", [])
    entities = extracted_fields.get("entities", {})

    lower_text = text.lower()

    if doc_type == "Invoice":
        # Check Invoice Number
        if not any("Invoice" in r.get("label", "") for r in ref_ids):
            missing.append({
                "field": "Invoice Number",
                "severity": "High",
                "description": "No explicit Invoice Number or Reference ID was detected."
            })
        # Check Dates
        if not dates:
            missing.append({
                "field": "Invoice / Due Date",
                "severity": "High",
                "description": "No invoice date or payment due date found."
            })
        elif not any("Due" in d.get("label", "") for d in dates):
            missing.append({
                "field": "Payment Due Date",
                "severity": "Medium",
                "description": "Due date is missing or not explicitly labeled."
            })
        # Check Total Amount
        if not amounts:
            missing.append({
                "field": "Total Amount Due",
                "severity": "High",
                "description": "No monetary amounts or balance total could be identified."
            })
        # Check Vendor/Contact
        if not contacts.get("emails") and not contacts.get("phones"):
            missing.append({
                "field": "Contact Information",
                "severity": "Medium",
                "description": "No vendor email address or telephone number detected."
            })

    elif doc_type == "Receipt":
        if not amounts:
            missing.append({
                "field": "Total Paid Amount",
                "severity": "High",
                "description": "Receipt does not show a clear total transaction amount."
            })
        if not dates:
            missing.append({
                "field": "Transaction Date",
                "severity": "High",
                "description": "Transaction date is missing."
            })
        if "payment" not in lower_text and "cash" not in lower_text and "card" not in lower_text:
            missing.append({
                "field": "Payment Method",
                "severity": "Low",
                "description": "Method of payment (Cash, Card, Transfer) is unspecified."
            })

    elif doc_type == "Contract / Agreement":
        if not any("Effective" in d.get("label", "") for d in dates) and not dates:
            missing.append({
                "field": "Effective Date",
                "severity": "High",
                "description": "No formal effective or execution date identified."
            })
        if "signature" not in lower_text and "signed" not in lower_text:
            missing.append({
                "field": "Signature Block",
                "severity": "High",
                "description": "Document may lack a formal execution or signature line."
            })
        if "governing law" not in lower_text and "jurisdiction" not in lower_text:
            missing.append({
                "field": "Governing Law / Jurisdiction",
                "severity": "Medium",
                "description": "Standard governing law clause was not found."
            })

    elif doc_type == "Resume / CV":
        if not contacts.get("emails"):
            missing.append({
                "field": "Email Address",
                "severity": "High",
                "description": "Resume does not contain a valid contact email."
            })
        if not contacts.get("phones"):
            missing.append({
                "field": "Phone Number",
                "severity": "Medium",
                "description": "No contact telephone number found."
            })
        if "education" not in lower_text and "degree" not in lower_text and "university" not in lower_text:
            missing.append({
                "field": "Education Section",
                "severity": "Medium",
                "description": "No standard education or academic qualification detected."
            })

    else:
        # Generic document checks
        if not dates:
            missing.append({
                "field": "Document Date",
                "severity": "Low",
                "description": "No date was recorded in this document."
            })
        if not entities.get("names") and not contacts.get("emails"):
            missing.append({
                "field": "Author / Contact Information",
                "severity": "Low",
                "description": "No author name or contact detail is present."
            })

    return missing


# =====================================================================
# 5. INCONSISTENCY & ANOMALY DETECTION
# =====================================================================

def detect_inconsistencies(
    text: str,
    extracted_fields: Dict[str, Any]
) -> List[Dict[str, str]]:
    """
    Detect logical contradictions, chronological errors, and arithmetic discrepancies.
    """
    inconsistencies = []
    dates = extracted_fields.get("dates", [])
    amounts = extracted_fields.get("amounts", [])

    # 1. Chronological date checks (e.g., Due Date before Invoice Date)
    invoice_date_val = None
    due_date_val = None

    for d in dates:
        lbl = d.get("label", "")
        raw = d.get("value", "")
        # Attempt to parse date
        parsed_dt = _try_parse_date(raw)
        if parsed_dt:
            if "Invoice" in lbl or "Issue" in lbl:
                invoice_date_val = parsed_dt
            elif "Due" in lbl:
                due_date_val = parsed_dt

    if invoice_date_val and due_date_val:
        if due_date_val < invoice_date_val:
            inconsistencies.append({
                "type": "Chronological Inconsistency",
                "severity": "High",
                "detail": f"Payment Due Date ({due_date_val.strftime('%Y-%m-%d')}) precedes Invoice Date ({invoice_date_val.strftime('%Y-%m-%d')})."
            })

    # 2. Arithmetic checks in amounts (Subtotal > Total Amount)
    subtotal_val = None
    total_val = None

    for a in amounts:
        lbl = a.get("label", "")
        val = a.get("numeric_value", 0.0)
        if lbl == "Subtotal":
            subtotal_val = val
        elif lbl == "Total Amount":
            total_val = val

    if subtotal_val is not None and total_val is not None:
        # If subtotal is greater than total without a discount
        if subtotal_val > total_val and "discount" not in text.lower():
            inconsistencies.append({
                "type": "Financial Discrepancy",
                "severity": "High",
                "detail": f"Subtotal ({subtotal_val:,.2f}) exceeds the stated Total Amount ({total_val:,.2f}) without recorded discount."
            })

    # 3. Currency symbol mixture
    currencies_found = set(re.findall(r"[$€£₹¥]", text))
    if len(currencies_found) > 1:
        inconsistencies.append({
            "type": "Mixed Currency Symbols",
            "severity": "Medium",
            "detail": f"Multiple currency symbols ({', '.join(currencies_found)}) detected within the same document."
        })

    # 4. Transcription or character anomaly check
    garbage_chars = len(re.findall(r"[^\x00-\x7F\s\w.,!?:;@&%$#()/\-]", text))
    if garbage_chars > 20:
        inconsistencies.append({
            "type": "Potential OCR Noise",
            "severity": "Low",
            "detail": f"Detected {garbage_chars} unusual symbols which may indicate low-quality scan or OCR artifacts."
        })

    return inconsistencies


def _try_parse_date(date_str: str) -> Optional[datetime]:
    """Helper to parse varied date strings."""
    date_str = date_str.strip()
    formats = [
        "%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y",
        "%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y",
        "%d %B %Y", "%d %b %Y"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            pass
    return None


# =====================================================================
# 6. LOCAL DOCUMENT Q&A ENGINE
# =====================================================================

def answer_question_local(
    question: str,
    text: str,
    pages: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Intelligent local question-answering over document text with semantic context citation.
    """
    q_clean = question.strip()
    if not q_clean or not text.strip():
        return {
            "answer": "Please provide a valid question and ensure document text is loaded.",
            "confidence": 0.0,
            "cited_context": "",
            "page_number": 1,
        }

    q_lower = q_clean.lower()
    q_words = re.findall(r"\b\w{3,}\b", q_lower)
    q_tokens = set(q_words)

    # Segment document into sentences and paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 10]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 10]

    # Pre-compiled fast lookups for common question archetypes:
    # 1. Document Overview / Summary / "What is this document?"
    if any(phrase in q_lower for phrase in [
        "what is this document", "summarize", "summary", "explain this document",
        "overview", "what does this document", "about this document", "tell me about",
        "key takeaways", "main point"
    ]):
        doc_class = classify_document(text)
        summary_info = generate_summary(text)
        bullets_str = "\n".join(f"• {b}" for b in summary_info.get("bullet_points", [])[:4])
        ans_text = (
            f"This document is classified as a **{doc_class['document_type']}**.\n\n"
            f"**Executive Summary:**\n{summary_info['executive_summary']}\n\n"
            f"**Key Highlights:**\n{bullets_str}"
        )
        return {
            "answer": ans_text,
            "confidence": 0.95,
            "cited_context": summary_info["executive_summary"][:200],
            "page_number": 1,
            "suggested_followups": [
                "What is the total amount?",
                "When is the payment due date?",
                "Are there any missing items or risks?",
                "Who are the parties involved?"
            ]
        }

    # 2. Date / Deadline queries (prioritize when asking 'when' or 'date')
    if any(w in q_lower for w in ["date", "when", "deadline", "due date", "expiry", "expiration", "timeline"]):
        date_matches = extract_dates(text)
        if date_matches:
            if "due" in q_lower or "deadline" in q_lower:
                for d in date_matches:
                    if "Due" in d["label"]:
                        return {
                            "answer": f"The due date is **{d['value']}**.",
                            "confidence": 0.92,
                            "cited_context": d["snippet"],
                            "page_number": _find_page_for_text(d["snippet"], pages),
                            "suggested_followups": [
                                "What is the total amount due?",
                                "What is the invoice date?",
                                "Who is the payment to be remitted to?"
                            ]
                        }
            if "invoice date" in q_lower or "issue date" in q_lower:
                for d in date_matches:
                    if "Invoice" in d["label"] or "Issue" in d["label"]:
                        return {
                            "answer": f"The invoice date is **{d['value']}**.",
                            "confidence": 0.92,
                            "cited_context": d["snippet"],
                            "page_number": _find_page_for_text(d["snippet"], pages),
                            "suggested_followups": [
                                "When is the due date?",
                                "What is the total amount?"
                            ]
                        }
            all_dates_str = "\n".join(f"• **{d['label']}:** {d['value']} *(Context: \"{d['snippet'][:60]}...\")*" for d in date_matches[:5])
            top_d = date_matches[0]
            return {
                "answer": f"Dates identified in the document:\n{all_dates_str}",
                "confidence": 0.89,
                "cited_context": top_d["snippet"],
                "page_number": _find_page_for_text(top_d["snippet"], pages),
                "suggested_followups": [
                    "What are the payment deadlines?",
                    "What is the total amount?"
                ]
            }

    # 3. Total / Amount queries
    if any(w in q_lower for w in ["total", "amount", "cost", "price", "how much", "subtotal", "balance", "fee", "tax"]):
        amount_matches = extract_amounts(text)
        if amount_matches:
            # If asking specifically about tax or subtotal
            if "tax" in q_lower:
                for a in amount_matches:
                    if "Tax" in a["label"] or "VAT" in a["label"]:
                        return {
                            "answer": f"The identified tax/VAT is **{a['raw']}**.",
                            "confidence": 0.93,
                            "cited_context": a["snippet"],
                            "page_number": _find_page_for_text(a["snippet"], pages),
                            "suggested_followups": ["What is the total amount with tax?", "What is the subtotal?"]
                        }
            if "subtotal" in q_lower:
                for a in amount_matches:
                    if "Subtotal" in a["label"]:
                        return {
                            "answer": f"The subtotal is **{a['raw']}**.",
                            "confidence": 0.93,
                            "cited_context": a["snippet"],
                            "page_number": _find_page_for_text(a["snippet"], pages),
                            "suggested_followups": ["What is the final total?", "When is it due?"]
                        }
            top_amt = amount_matches[0]
            all_amts = "\n".join(f"• **{a['label']}:** {a['raw']}" for a in amount_matches[:5])
            return {
                "answer": f"The primary {top_amt['label']} is **{top_amt['raw']}**.\n\n**Financial Breakdown:**\n{all_amts}",
                "confidence": 0.94,
                "cited_context": top_amt["snippet"],
                "page_number": _find_page_for_text(top_amt["snippet"], pages),
                "suggested_followups": [
                    "When is the payment due date?",
                    "What is the invoice number?",
                    "Who is the client or bill-to party?"
                ]
            }

    # 4. People / Parties / Organizations
    if any(w in q_lower for w in [
        "who is", "who are", "parties", "people", "doctor", "client", "customer",
        "vendor", "recipient", "person", "organization", "company", "bill to", "seller", "buyer"
    ]):
        entities_data = extract_entities_and_names(text)
        orgs = entities_data.get("organizations", [])
        names = entities_data.get("names", [])
        parts = []
        if orgs:
            parts.append(f"**Organizations / Vendors:** {', '.join(orgs)}")
        if names:
            parts.append(f"**Individuals / Clients / Parties:** {', '.join(names)}")
        if parts:
            return {
                "answer": "\n\n".join(parts),
                "confidence": 0.91,
                "cited_context": text[:180],
                "page_number": 1,
                "suggested_followups": [
                    "What are their contact details?",
                    "What services or goods were provided?",
                    "What is the total amount?"
                ]
            }

    # 5. Missing Information & Risk / Audit Checks
    if any(w in q_lower for w in ["missing", "risk", "inconsistent", "error", "problem", "audit", "incomplete", "issue", "discrepancy"]):
        classification = classify_document(text)
        fields = {
            "dates": extract_dates(text),
            "amounts": extract_amounts(text),
            "contacts": extract_contact_info(text),
            "reference_ids": extract_reference_ids(text),
            "entities": extract_entities_and_names(text),
        }
        missing_items = detect_missing_information(text, classification["document_type"], fields)
        inconsistencies = detect_inconsistencies(text, fields)

        audit_res = []
        if missing_items:
            audit_res.append("**Missing Standard Information:**")
            for m in missing_items:
                audit_res.append(f"• **{m['field']}** ({m['severity']} Priority): {m['description']}")
        else:
            audit_res.append("✅ All standard fields for this document type were detected.")

        if inconsistencies:
            audit_res.append("\n**Potential Inconsistencies / Warnings:**")
            for inc in inconsistencies:
                audit_res.append(f"• **{inc['type']}**: {inc['detail']}")
        else:
            audit_res.append("✅ No logical, chronological, or mathematical contradictions found.")

        return {
            "answer": "\n".join(audit_res),
            "confidence": 0.95,
            "cited_context": "Automated Document Quality & Completeness Audit",
            "page_number": 1,
            "suggested_followups": [
                "Summarize the document",
                "What is the total amount?",
                "What are the contact details?"
            ]
        }

    # 6. Invoice / Reference Number queries
    if any(w in q_lower for w in ["invoice number", "reference", "account number", "order number", "po number", "tax id"]):
        ref_matches = extract_reference_ids(text)
        if ref_matches:
            for r in ref_matches:
                if any(w in q_lower for w in r["label"].lower().split()):
                    return {
                        "answer": f"The **{r['label']}** is `{r['value']}`.",
                        "confidence": 0.95,
                        "cited_context": f"{r['label']}: {r['value']}",
                        "page_number": 1,
                        "suggested_followups": [
                            "What is the invoice date?",
                            "What is the total amount due?"
                        ]
                    }
            all_refs = "\n".join(f"• **{r['label']}:** `{r['value']}`" for r in ref_matches)
            return {
                "answer": f"Identified Reference Identifiers:\n{all_refs}",
                "confidence": 0.92,
                "cited_context": f"{ref_matches[0]['label']}: {ref_matches[0]['value']}",
                "page_number": 1,
                "suggested_followups": [
                    "What is the due date?",
                    "What is the total amount?"
                ]
            }

    # 7. Contact / Email queries
    if any(w in q_lower for w in ["email", "contact", "phone", "call", "address", "website", "reach"]):
        contacts = extract_contact_info(text)
        c_parts = []
        if contacts["emails"]:
            c_parts.append(f"**Email Address(es):** {', '.join(contacts['emails'])}")
        if contacts["phones"]:
            c_parts.append(f"**Telephone Number(s):** {', '.join(contacts['phones'])}")
        if contacts["urls"]:
            c_parts.append(f"**Website(s):** {', '.join(contacts['urls'])}")
        if c_parts:
            return {
                "answer": "\n\n".join(c_parts),
                "confidence": 0.95,
                "cited_context": f"Emails: {', '.join(contacts['emails'])}",
                "page_number": 1,
                "suggested_followups": [
                    "Who are the key organizations?",
                    "What is the total amount?"
                ]
            }

    # General semantic scoring over sentences and paragraphs
    scored_sentences = []
    for sent in sentences:
        s_words = re.findall(r"\b\w{3,}\b", sent.lower())
        overlap = sum(1 for w in s_words if w in q_tokens)
        if overlap > 0:
            score = overlap / (len(q_tokens) + 1.0)
            scored_sentences.append((score, sent))

    if scored_sentences:
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        best_score, best_sent = scored_sentences[0]
        confidence = min(0.92, round(0.40 + best_score * 0.6, 2))
        return {
            "answer": best_sent,
            "confidence": confidence,
            "cited_context": best_sent,
            "page_number": _find_page_for_text(best_sent, pages),
            "suggested_followups": [
                "Summarize this document",
                "What is the total amount?",
                "When is the due date?"
            ]
        }

    # Fallback if no direct keyword match
    return {
        "answer": "Could not find a specific answer for this question in the current document text. Try asking about the summary, dates, amounts, people, or reference numbers.",
        "confidence": 0.25,
        "cited_context": text[:150] + "...",
        "page_number": 1,
        "suggested_followups": [
            "Summarize this document",
            "What is the total amount?",
            "What are the key dates mentioned?"
        ]
    }


def ask_ai_question(
    question: str,
    text: str,
    pages: Optional[List[Dict[str, Any]]] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    api_key: Optional[str] = None,
    api_provider: str = "openai"
) -> Dict[str, Any]:
    """
    Master AI Question-Answering dispatcher.
    Uses external LLM (OpenAI/Gemini) if API key is provided,
    otherwise uses the robust, fast local NLP semantic engine.
    """
    if not question.strip():
        return {
            "answer": "Please ask a question.",
            "confidence": 0.0,
            "cited_context": "",
            "page_number": 1,
            "engine": "local_nlp"
        }

    # Check if external API key is configured
    if api_key and api_key.strip():
        try:
            sys_prompt = (
                "You are DocuMind AI, an intelligent document analysis assistant. "
                "Answer the user's question clearly and accurately using ONLY the provided document text. "
                "If the document text does not contain the answer, say so politely. "
                "Format your answer with markdown formatting."
            )
            context_snippet = text[:6000]
            user_prompt = f"Document Content:\n\"\"\"\n{context_snippet}\n\"\"\"\n\nUser Question: {question}"

            external_ans = query_external_llm(
                user_prompt,
                sys_prompt,
                provider=api_provider,
                api_key=api_key
            )
            if external_ans:
                return {
                    "answer": external_ans,
                    "confidence": 0.98,
                    "cited_context": context_snippet[:150] + "...",
                    "page_number": 1,
                    "engine": api_provider,
                    "suggested_followups": [
                        "Can you summarize the main points?",
                        "What are the financial implications?",
                        "What should I do next?"
                    ]
                }
        except Exception:
            pass  # Fallback to local engine

    # Local Engine
    local_res = answer_question_local(question, text, pages=pages)
    local_res["engine"] = "local_nlp"
    return local_res


def _find_page_for_text(snippet: str, pages: Optional[List[Dict[str, Any]]]) -> int:
    """Helper to locate which page contains snippet text."""
    if not pages:
        return 1
    snippet_core = snippet.strip()[:40].lower()
    for p in pages:
        if snippet_core in p.get("text", "").lower():
            return p.get("page_number", 1)
    return 1


# =====================================================================
# 7. OPTIONAL EXTERNAL API INTEGRATION
# =====================================================================

def query_external_llm(
    prompt: str,
    system_prompt: str,
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None
) -> Optional[str]:
    """
    Optional bridge to call OpenAI, Anthropic, or Gemini when API key is provided.
    Falls back gracefully if network or credentials fail.
    """
    if not api_key:
        return None

    import requests

    try:
        provider = provider.lower().strip()
        if provider == "openai":
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model or "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2
            }
            res = requests.post(url, headers=headers, json=payload, timeout=25)
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"]

        elif provider == "gemini":
            model_name = model or "gemini-1.5-flash"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": f"{system_prompt}\n\n{prompt}"}]}]
            }
            res = requests.post(url, headers=headers, json=payload, timeout=25)
            if res.status_code == 200:
                data = res.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]

    except Exception:
        return None

    return None


# =====================================================================
# 8. MASTER ANALYSIS PIPELINE
# =====================================================================

def analyze_document(
    text: str,
    pages: Optional[List[Dict[str, Any]]] = None,
    external_api_key: Optional[str] = None,
    api_provider: str = "openai"
) -> Dict[str, Any]:
    """
    Run full end-to-end intelligent document analysis.
    """
    if not text.strip():
        return {
            "status": "empty_document",
            "message": "No text detected in document to analyze."
        }

    # 1. Classification
    classification = classify_document(text)
    doc_type = classification["document_type"]

    # 2. Summarization
    summary = generate_summary(text)

    # 3. Entity & Field Extraction
    dates = extract_dates(text)
    amounts = extract_amounts(text)
    contacts = extract_contact_info(text)
    ref_ids = extract_reference_ids(text)
    entities = extract_entities_and_names(text)

    extracted_fields = {
        "dates": dates,
        "amounts": amounts,
        "contacts": contacts,
        "reference_ids": ref_ids,
        "entities": entities,
    }

    # 4. Missing Information Detection
    missing_info = detect_missing_information(text, doc_type, extracted_fields)

    # 5. Inconsistency & Anomaly Detection
    inconsistencies = detect_inconsistencies(text, extracted_fields)

    # 6. Optional External LLM Enhancement
    external_insights = None
    if external_api_key:
        sys_p = "You are an expert AI document analyst. Provide concise, professional insights."
        user_p = f"Analyze this {doc_type} text and provide 3 executive observations:\n\n{text[:3000]}"
        external_insights = query_external_llm(user_p, sys_p, provider=api_provider, api_key=external_api_key)

    return {
        "status": "success",
        "classification": classification,
        "summary": summary,
        "extracted_fields": extracted_fields,
        "missing_information": missing_info,
        "inconsistencies": inconsistencies,
        "external_insights": external_insights,
    }


# =====================================================================
# 9. ADVANCED FEATURES: REDACTION, TRANSLATION & ANALYTICS
# =====================================================================

def redact_sensitive_information(
    text: str,
    mask_emails: bool = True,
    mask_phones: bool = True,
    mask_ids: bool = True,
    mask_amounts: bool = False
) -> Tuple[str, Dict[str, int]]:
    """
    Mask personally identifiable information (PII) and sensitive numbers.
    """
    redacted = text
    counts = {"emails": 0, "phones": 0, "ids": 0, "amounts": 0}

    if mask_emails:
        emails = list(set(re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b", redacted)))
        for em in emails:
            redacted = redacted.replace(em, "[REDACTED EMAIL]")
            counts["emails"] += 1

    if mask_phones:
        phone_matches = re.findall(r"(?:(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4})", redacted)
        for ph in set(phone_matches):
            if len(re.sub(r"\D", "", ph)) >= 8:
                redacted = redacted.replace(ph, "[REDACTED PHONE]")
                counts["phones"] += 1

    if mask_ids:
        # Mask invoice / account / tax IDs
        id_patterns = [
            r"(?:invoice\s*(?:no\.?|number|#)?\s*[:#-]?\s*)([A-Za-z0-9\-_]{4,20})",
            r"(?:account\s*(?:no\.?|number|#)?\s*[:#-]?\s*)([A-Za-z0-9\-_]{4,24})",
            r"(?:tax\s*(?:id|number|no\.?)?\s*[:#-]?\s*)([A-Za-z0-9\-_]{5,20})"
        ]
        for pat in id_patterns:
            for m in re.finditer(pat, redacted, re.IGNORECASE):
                val = m.group(1).strip()
                if val:
                    redacted = redacted.replace(val, "[REDACTED ID]")
                    counts["ids"] += 1

    if mask_amounts:
        amount_matches = re.findall(r"([$€£₹¥]\s*[\d,]+(?:\.\d{2})?)", redacted)
        for amt in set(amount_matches):
            redacted = redacted.replace(amt, "[REDACTED AMOUNT]")
            counts["amounts"] += 1

    return redacted, counts


TRANSLATION_DICTIONARIES = {
    "Spanish": {
        "tax invoice": "factura de impuestos", "invoice": "factura",
        "date": "fecha", "due date": "fecha de vencimiento", "issue date": "fecha de emisión",
        "total amount": "monto total", "total": "total", "subtotal": "subtotal",
        "tax": "impuesto", "balance": "saldo", "balance due": "saldo adeudado",
        "summary": "resumen", "executive summary": "resumen ejecutivo",
        "description": "descripción", "contract": "contrato", "agreement": "acuerdo",
        "payment": "pago", "client": "cliente", "customer": "cliente",
        "vendor": "proveedor", "supplier": "proveedor", "terms": "términos y condiciones",
        "terms and conditions": "términos y condiciones", "receipt": "recibo",
        "statement": "estado de cuenta", "amount": "monto", "services": "servicios",
        "contact": "contacto", "phone": "teléfono", "email": "correo electrónico"
    },
    "French": {
        "tax invoice": "facture fiscale", "invoice": "facture",
        "date": "date", "due date": "date d'échéance", "issue date": "date d'émission",
        "total amount": "montant total", "total": "total", "subtotal": "sous-total",
        "tax": "taxe / tva", "balance": "solde", "balance due": "solde dû",
        "summary": "résumé", "executive summary": "résumé exécutif",
        "description": "description", "contract": "contrat", "agreement": "accord",
        "payment": "paiement", "client": "client", "customer": "client",
        "vendor": "fournisseur", "supplier": "fournisseur", "terms": "termes et conditions",
        "terms and conditions": "conditions générales", "receipt": "reçu",
        "statement": "relevé", "amount": "montant", "services": "services",
        "contact": "contact", "phone": "téléphone", "email": "e-mail"
    },
    "German": {
        "tax invoice": "steuerrechnung", "invoice": "rechnung",
        "date": "datum", "due date": "fälligkeitsdatum", "issue date": "ausstellungsdatum",
        "total amount": "gesamtbetrag", "total": "gesamt", "subtotal": "zwischensumme",
        "tax": "steuer / mwst", "balance": "saldo", "balance due": "fälliger betrag",
        "summary": "zusammenfassung", "executive summary": "leitende zusammenfassung",
        "description": "beschreibung", "contract": "vertrag", "agreement": "vereinbarung",
        "payment": "zahlung", "client": "kunde", "customer": "kunde",
        "vendor": "lieferant", "supplier": "lieferant", "terms": "allgemeine geschäftsbedingungen",
        "terms and conditions": "allgemeine geschäftsbedingungen", "receipt": "quittung",
        "statement": "kontoauszug", "amount": "betrag", "services": "dienstleistungen",
        "contact": "kontakt", "phone": "telefon", "email": "e-mail"
    },
    "Italian": {
        "tax invoice": "fattura fiscale", "invoice": "fattura",
        "date": "data", "due date": "data di scadenza", "issue date": "data di emissione",
        "total amount": "importo totale", "total": "totale", "subtotal": "totale parziale",
        "tax": "imposta / iva", "balance": "saldo", "balance due": "saldo dovuto",
        "summary": "sommario", "executive summary": "sintesi esecutiva",
        "description": "descrizione", "contract": "contratto", "agreement": "accordo",
        "payment": "pagamento", "client": "cliente", "customer": "cliente",
        "vendor": "fornitore", "supplier": "fornitore", "terms": "termini e condizioni",
        "terms and conditions": "termini e condizioni", "receipt": "ricevuta",
        "statement": "estratto conto", "amount": "importo", "services": "servizi",
        "contact": "contatto", "phone": "telefono", "email": "e-mail"
    },
    "Hindi": {
        "tax invoice": "कर चालान (Tax Invoice)", "invoice": "चालान (Invoice)",
        "date": "दिनांक (Date)", "due date": "अंतिम तिथि (Due Date)", "issue date": "जारी करने की तिथि (Issue Date)",
        "total amount": "कुल राशि (Total Amount)", "total": "कुल (Total)", "subtotal": "उप-कुल (Subtotal)",
        "tax": "कर (Tax)", "balance": "शेष राशि (Balance)", "balance due": "देय शेष (Balance Due)",
        "summary": "सारांश (Summary)", "executive summary": "मुख्य सारांश (Executive Summary)",
        "description": "विवरण (Description)", "contract": "अनुबंध (Contract)", "agreement": "समझौता (Agreement)",
        "payment": "भुगतान (Payment)", "client": "ग्राहक (Client)", "customer": "ग्राहक (Customer)",
        "vendor": "विक्रेता (Vendor)", "supplier": "आपूर्तिकर्ता (Supplier)", "terms": "नियम और शर्तें (Terms)",
        "terms and conditions": "नियम और शर्तें (Terms and Conditions)", "receipt": "रसीद (Receipt)",
        "statement": "विवरण (Statement)", "amount": "राशि (Amount)", "services": "सेवाएं (Services)",
        "contact": "संपर्क (Contact)", "phone": "फोन (Phone)", "email": "ईमेल (Email)"
    },
    "Portuguese": {
        "tax invoice": "nota fiscal", "invoice": "fatura",
        "date": "data", "due date": "data de vencimento", "issue date": "data de emissão",
        "total amount": "valor total", "total": "total", "subtotal": "subtotal",
        "tax": "imposto", "balance": "saldo", "balance due": "saldo a pagar",
        "summary": "resumo", "executive summary": "resumo executivo",
        "description": "descrição", "contract": "contrato", "agreement": "acordo",
        "payment": "pagamento", "client": "cliente", "customer": "cliente",
        "vendor": "fornecedor", "supplier": "fornecedor", "terms": "termos e condições",
        "terms and conditions": "termos e condições", "receipt": "recibo",
        "statement": "extrato", "amount": "valor", "services": "serviços",
        "contact": "contato", "phone": "telefone", "email": "e-mail"
    },
    "Russian": {
        "tax invoice": "налоговый счет", "invoice": "счет-фактура",
        "date": "дата", "due date": "срок оплаты", "issue date": "дата выдачи",
        "total amount": "общая сумма", "total": "итого", "subtotal": "промежуточный итог",
        "tax": "налог", "balance": "баланс", "balance due": "к оплате",
        "summary": "резюме", "executive summary": "исполнительное резюме",
        "description": "описание", "contract": "договор", "agreement": "соглашение",
        "payment": "оплата", "client": "клиент", "customer": "заказчик",
        "vendor": "поставщик", "supplier": "поставщик", "terms": "условия",
        "terms and conditions": "правила и условия", "receipt": "квитанция",
        "statement": "выписка", "amount": "сумма", "services": "услуги",
        "contact": "контакт", "phone": "телефон", "email": "эл. почта"
    },
    "Arabic": {
        "tax invoice": "فاتورة ضريبية", "invoice": "فاتورة",
        "date": "تاريخ", "due date": "تاريخ الاستحقاق", "issue date": "تاريخ الإصدار",
        "total amount": "المبلغ الإجمالي", "total": "المجموع", "subtotal": "المجموع الفرعي",
        "tax": "ضريبة", "balance": "الرصيد", "balance due": "الرصيد المستحق",
        "summary": "ملخص", "executive summary": "ملخص تنفيذي",
        "description": "الوصف", "contract": "عقد", "agreement": "اتفاقية",
        "payment": "دفع", "client": "عميل", "customer": "عميل",
        "vendor": "مورد", "supplier": "مزود", "terms": "الشروط",
        "terms and conditions": "الشروط والأحكام", "receipt": "إيصال",
        "statement": "كشف حساب", "amount": "المبلغ", "services": "خدمات",
        "contact": "جهة الاتصال", "phone": "هاتف", "email": "بريد إلكتروني"
    },
    "Japanese": {
        "tax invoice": "適格請求書", "invoice": "請求書",
        "date": "日付", "due date": "支払期日", "issue date": "発行日",
        "total amount": "合計金額", "total": "合計", "subtotal": "小計",
        "tax": "税金 / 消費税", "balance": "残高", "balance due": "差引請求残高",
        "summary": "要約", "executive summary": "エグゼクティブサマリー",
        "description": "説明 / 明細", "contract": "契約書", "agreement": "合意書",
        "payment": "支払い", "client": "顧客", "customer": "お客様",
        "vendor": "業者", "supplier": "納入業者", "terms": "利用規約",
        "terms and conditions": "取引条件", "receipt": "領収書",
        "statement": "明細書", "amount": "金額", "services": "サービス",
        "contact": "連絡先", "phone": "電話番号", "email": "メールアドレス"
    },
    "Chinese": {
        "tax invoice": "税务发票", "invoice": "发票",
        "date": "日期", "due date": "到期日", "issue date": "签发日期",
        "total amount": "总金额", "total": "合计", "subtotal": "小计",
        "tax": "税额", "balance": "余额", "balance due": "应付余额",
        "summary": "摘要", "executive summary": "执行摘要",
        "description": "描述 / 规格", "contract": "合同", "agreement": "协议",
        "payment": "付款", "client": "客户", "customer": "客户",
        "vendor": "供应商", "supplier": "供货商", "terms": "条款",
        "terms and conditions": "条款和条件", "receipt": "收据",
        "statement": "对账单", "amount": "金额", "services": "服务",
        "contact": "联系方式", "phone": "电话", "email": "电子邮件"
    }
}


LANGUAGE_LOCALE_MAP = {
    "Spanish": "es-ES",
    "French": "fr-FR",
    "German": "de-DE",
    "Italian": "it-IT",
    "Hindi": "hi-IN",
    "Portuguese": "pt-PT",
    "Russian": "ru-RU",
    "Arabic": "ar-SA",
    "Japanese": "ja-JP",
    "Chinese": "zh-CN",
}


def _align_term_casing(translated_text: str, source_text: str, target_language: str) -> str:
    """
    Ensure translated terms match the casing (Title Case, UPPERCASE) of the source text.
    """
    dict_map = TRANSLATION_DICTIONARIES.get(target_language, {})
    if not dict_map:
        return translated_text

    result = translated_text
    for en_w, foreign_w in sorted(dict_map.items(), key=lambda x: len(x[0]), reverse=True):
        if not foreign_w or len(foreign_w) < 2:
            continue
        # Check casing in source
        if en_w.upper() in source_text:
            pat = re.compile(rf"\b{re.escape(foreign_w)}\b", re.IGNORECASE)
            result = pat.sub(foreign_w.upper(), result)
        elif en_w.title() in source_text or en_w.capitalize() in source_text:
            pat = re.compile(rf"\b{re.escape(foreign_w)}\b", re.IGNORECASE)
            result = pat.sub(foreign_w.title(), result)

    return result


def _chunk_text_for_translation(text: str, max_chars: int = 350) -> List[str]:
    """
    Split text into chunks of at most max_chars without breaking mid-word.
    """
    lines = text.split("\n")
    chunks = []
    current: List[str] = []
    curr_len = 0

    for line in lines:
        if not line.strip():
            if current:
                chunks.append("\n".join(current))
                current = []
                curr_len = 0
            chunks.append("")
            continue

        if len(line) <= max_chars:
            if curr_len + len(line) + 1 <= max_chars:
                current.append(line)
                curr_len += len(line) + 1
            else:
                if current:
                    chunks.append("\n".join(current))
                current = [line]
                curr_len = len(line)
        else:
            if current:
                chunks.append("\n".join(current))
                current = []
                curr_len = 0
            words = line.split(" ")
            sub: List[str] = []
            sub_len = 0
            for w in words:
                if sub_len + len(w) + 1 > max_chars:
                    if sub:
                        chunks.append(" ".join(sub))
                    sub = [w]
                    sub_len = len(w)
                else:
                    sub.append(w)
                    sub_len += len(w) + 1
            if sub:
                chunks.append(" ".join(sub))

    if current:
        chunks.append("\n".join(current))

    return chunks


def translate_document_text(
    text: str,
    target_language: str = "Spanish",
    api_key: Optional[str] = None,
    api_provider: str = "openai"
) -> str:
    """
    Translate full document text, summaries, or selected sections into the target language.
    Uses external LLM when configured, MyMemory neural machine translation as reliable local engine,
    and terminology mapping as an offline fallback.
    """
    if not text or not text.strip():
        return ""

    # 1. External LLM Translation if API key is provided
    if api_key and api_key.strip():
        try:
            sys_p = (
                f"You are a professional multilingual document translator. "
                f"Translate the provided text accurately and fluently into {target_language}. "
                "Preserve all formatting, numerical values, currencies, dates, bullet points, and section layout."
            )
            llm_trans = query_external_llm(text[:5000], sys_p, provider=api_provider, api_key=api_key)
            if llm_trans and len(llm_trans.strip()) > 10:
                return _align_term_casing(llm_trans.strip(), text, target_language)
        except Exception:
            pass

    # 2. Real Machine Translation using MyMemoryTranslator
    target_code = LANGUAGE_LOCALE_MAP.get(target_language, "es-ES")
    try:
        from deep_translator import MyMemoryTranslator
        translator = MyMemoryTranslator(source="en-GB", target=target_code)
        chunks = _chunk_text_for_translation(text, max_chars=350)
        translated_chunks = []
        for chunk in chunks:
            if not chunk.strip():
                translated_chunks.append("")
                continue
            translated = translator.translate(chunk)
            if not translated or any(err in translated.upper() for err in ["MYMEMORY WARNING", "QUERY LENGTH LIMIT", "INVALID SOURCE LANGUAGE"]):
                raise ValueError(f"MyMemory error: {translated}")
            translated_chunks.append(translated)

        result = "\n".join(translated_chunks)
        if result and len(result.strip()) > 0:
            return _align_term_casing(result, text, target_language)
    except Exception:
        pass

    # 3. Offline Dictionary Fallback
    dict_map = TRANSLATION_DICTIONARIES.get(target_language)
    if not dict_map:
        return text

    translated = text
    for en_word, foreign_word in sorted(dict_map.items(), key=lambda x: len(x[0]), reverse=True):
        def repl(m: re.Match) -> str:
            matched = m.group(0)
            if matched.isupper():
                return foreign_w_case.upper()
            elif matched[0].isupper():
                return foreign_w_case.title()
            return foreign_w_case.lower()

        foreign_w_case = foreign_word
        pattern = re.compile(rf"\b{re.escape(en_word)}\b", re.IGNORECASE)
        translated = pattern.sub(repl, translated)

    return _align_term_casing(translated, text, target_language)


def calculate_document_analytics(text: str) -> Dict[str, Any]:
    """
    Compute comprehensive document text analytics, vocabulary density,
    estimated reading/speaking speed, and detected tone.
    """
    words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
    total_words = len(words)
    unique_words = len(set(words))
    diversity = round((unique_words / max(1, total_words)) * 100, 1)

    sentences = [s for s in re.split(r"[.!?]+", text) if len(s.strip()) > 5]
    total_sentences = max(1, len(sentences))
    avg_sentence_len = round(total_words / total_sentences, 1)

    # Tone detection
    lower_t = text.lower()
    tones = []
    if any(w in lower_t for w in ["invoice", "due", "balance", "subtotal", "payment", "bank", "account"]):
        tones.append("Financial / Commercial")
    if any(w in lower_t for w in ["hereby", "agreement", "party", "witness", "jurisdiction", "shall"]):
        tones.append("Formal Legal")
    if any(w in lower_t for w in ["urgent", "immediately", "overdue", "notice", "warning"]):
        tones.append("Urgent / Notice")
    if any(w in lower_t for w in ["patient", "doctor", "clinical", "diagnosis", "dosage"]):
        tones.append("Medical / Clinical")
    if not tones:
        tones.append("Professional Standard")

    # Keyword frequency
    stop_words = {"the", "and", "for", "that", "this", "with", "from", "are", "was", "have", "has", "your", "will"}
    content_words = [w for w in words if w not in stop_words and len(w) > 3]
    top_keywords = Counter(content_words).most_common(8)

    return {
        "total_words": total_words,
        "unique_words": unique_words,
        "vocabulary_diversity_pct": diversity,
        "total_sentences": total_sentences,
        "avg_sentence_length": avg_sentence_len,
        "detected_tones": tones,
        "top_keywords": top_keywords,
        "est_reading_time_sec": round(total_words / 3.3, 1),
        "est_speaking_time_sec": round(total_words / 2.2, 1),
    }


if __name__ == "__main__":
    print("DocuMind AI Module Test")
    sample_text = (
        "INVOICE\n"
        "Acme Global Technologies Inc.\n"
        "Invoice No: INV-2025-089\n"
        "Invoice Date: 2025-04-10\n"
        "Due Date: 2025-04-25\n"
        "Bill To: Dr. Robert Harrison\n"
        "Subtotal: $4,500.00\n"
        "Tax (10%): $450.00\n"
        "Total Amount: $4,950.00\n"
        "Email: billing@acmeglobal.com\n"
        "Phone: +1 (555) 234-5678\n"
        "Terms: Payment due within 15 days."
    )
    res = analyze_document(sample_text)
    print(f"Classification: {res['classification']['document_type']}")
    print(f"Executive Summary: {res['summary']['executive_summary']}")
    print(f"Amounts found: {[a['raw'] for a in res['extracted_fields']['amounts']]}")
    print(f"Dates found: {[d['value'] for d in res['extracted_fields']['dates']]}")
    print(f"Missing items: {len(res['missing_information'])}")
    print(f"Inconsistencies: {len(res['inconsistencies'])}")

    qa = answer_question_local("What is the total amount?", sample_text)
    print(f"Q&A Test: {qa['answer']}")

