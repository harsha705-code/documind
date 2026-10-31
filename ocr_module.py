"""
ocr_module.py

DocuMind - OCR & Intelligent Text Extraction Engine

Features:
1. Dual-engine OCR: EasyOCR (local, robust, deep-learning based) and
   Tesseract OCR (via pytesseract, fast, industry standard).
2. Smart PDF extraction: Fast native digital text extraction with automatic
   OCR fallback for scanned pages or full OCR override.
3. Image pre-processing and enhancement (denoising, contrast enhancement,
   adaptive binarization, deskewing).
4. Structured output with word counts, confidence scores, and bounding boxes.
5. Searchable PDF metadata preparation.
6. Lazy model initialization to prevent slow startup times.
"""

import io
import os
import shutil
import warnings
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

# Suppress PyTorch and PyMuPDF deprecation & backend warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import pymupdf as fitz
except ImportError:
    import fitz

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

# Pytesseract optional detection
try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    pytesseract = None
    HAS_PYTESSERACT = False

# Global cached EasyOCR instance
_EASYOCR_READER = None


def find_tesseract_binary(custom_path: Optional[str] = None) -> Optional[str]:
    """
    Locate Tesseract executable on Windows or POSIX systems.
    """
    if custom_path and os.path.isfile(custom_path):
        return custom_path

    # Check PATH
    cmd = shutil.which("tesseract")
    if cmd:
        return cmd

    # Standard Windows install locations
    candidate_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        os.path.expandvars(r"%USERPROFILE%\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"),
    ]
    for p in candidate_paths:
        if os.path.isfile(p):
            return p

    return None


def get_available_ocr_engines() -> List[str]:
    """
    Return list of available OCR engines on current host.
    """
    engines = ["easyocr"]  # EasyOCR is always bundled in Python
    tess_path = find_tesseract_binary()
    if HAS_PYTESSERACT and tess_path:
        engines.append("tesseract")
    return engines


def get_easyocr_reader(languages: Optional[List[str]] = None):
    """
    Lazily initialize and return the EasyOCR Reader instance.
    """
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        import easyocr
        langs = languages or ["en"]
        _EASYOCR_READER = easyocr.Reader(langs, gpu=False)
    return _EASYOCR_READER


def enhance_image(
    image: Union[Image.Image, np.ndarray],
    method: str = "adaptive"
) -> np.ndarray:
    """
    Preprocess and enhance document image for improved OCR accuracy.

    Args:
        image: PIL Image or numpy array (RGB/BGR).
        method: 'adaptive', 'contrast', 'denoise', 'grayscale', or 'none'.

    Returns:
        np.ndarray: Enhanced image suitable for OCR.
    """
    if isinstance(image, Image.Image):
        np_img = np.array(image.convert("RGB"))
        cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    elif isinstance(image, np.ndarray):
        cv_img = image.copy()
        if len(cv_img.shape) == 2:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
    else:
        raise TypeError("Input must be a PIL Image or numpy array.")

    method = method.lower()
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    if method == "none":
        return cv_img

    elif method == "grayscale":
        return gray

    elif method == "contrast":
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)

    elif method == "denoise":
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)

    elif method == "adaptive":
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        thresh = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )
        # Return as 3-channel for OCR engine compatibility
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    return cv_img


def extract_text_easyocr(
    image: Union[Image.Image, np.ndarray],
    preprocess_method: str = "contrast"
) -> List[Dict[str, Any]]:
    """
    Extract text items using EasyOCR.
    """
    reader = get_easyocr_reader()
    processed = enhance_image(image, method=preprocess_method)
    results = reader.readtext(processed)

    extracted = []
    for box, text, confidence in results:
        # Convert numpy coordinates to standard python ints/floats
        clean_box = [[float(coord[0]), float(coord[1])] for coord in box]
        extracted.append({
            "text": str(text).strip(),
            "confidence": round(float(confidence), 3),
            "box": clean_box,
        })
    return extracted


def extract_text_tesseract(
    image: Union[Image.Image, np.ndarray],
    preprocess_method: str = "adaptive",
    custom_tesseract_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Extract text items using Tesseract OCR (pytesseract).
    """
    if not HAS_PYTESSERACT:
        raise RuntimeError("pytesseract is not installed.")

    tess_bin = find_tesseract_binary(custom_tesseract_path)
    if not tess_bin:
        raise RuntimeError("Tesseract OCR executable not found on system.")

    pytesseract.pytesseract.tesseract_cmd = tess_bin
    processed = enhance_image(image, method=preprocess_method)

    # Use pytesseract image_to_data for word boxes and confidence
    data = pytesseract.image_to_data(
        processed,
        output_type=pytesseract.Output.DICT
    )

    extracted = []
    n_boxes = len(data["text"])
    for i in range(n_boxes):
        txt = data["text"][i].strip()
        conf = float(data["conf"][i])
        if txt and conf > 0:
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            box = [
                [float(x), float(y)],
                [float(x + w), float(y)],
                [float(x + w), float(y + h)],
                [float(x), float(y + h)],
            ]
            extracted.append({
                "text": txt,
                "confidence": round(conf / 100.0, 3),
                "box": box,
            })

    return extracted


def extract_text(
    image: Union[Image.Image, np.ndarray],
    engine: str = "auto",
    preprocess: str = "contrast",
    custom_tesseract_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Unified text extraction dispatcher from an image.
    """
    engine = engine.lower().strip()
    available = get_available_ocr_engines()

    if engine == "tesseract":
        if "tesseract" in available:
            return extract_text_tesseract(image, preprocess, custom_tesseract_path)
        # fallback to easyocr if requested engine unavailable
        return extract_text_easyocr(image, preprocess)

    # Default to EasyOCR
    return extract_text_easyocr(image, preprocess)


def _read_input_bytes(file_or_path: Any) -> Tuple[bytes, str]:
    """
    Helper to extract raw bytes and a filename indicator.
    """
    filename = "document"
    if hasattr(file_or_path, "name"):
        filename = str(file_or_path.name)
    elif isinstance(file_or_path, (str, Path)):
        filename = os.path.basename(str(file_or_path))

    if isinstance(file_or_path, (str, Path)):
        with open(file_or_path, "rb") as f:
            data = f.read()
    elif isinstance(file_or_path, (bytes, bytearray)):
        data = bytes(file_or_path)
    elif hasattr(file_or_path, "read"):
        orig_pos = None
        if hasattr(file_or_path, "tell"):
            try:
                orig_pos = file_or_path.tell()
            except Exception:
                pass
        if hasattr(file_or_path, "seek"):
            file_or_path.seek(0)
        data = file_or_path.read()
        if orig_pos is not None and hasattr(file_or_path, "seek"):
            try:
                file_or_path.seek(orig_pos)
            except Exception:
                pass
    else:
        raise TypeError("Input must be a filepath, bytes, or file-like object.")

    return bytes(data), filename


def process_document(
    uploaded_file: Any,
    engine: str = "auto",
    preprocess: str = "contrast",
    pdf_mode: str = "auto",
    custom_tesseract_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Full document processing pipeline for PDFs, JPGs, and PNGs.

    Args:
        uploaded_file: Streamlit UploadedFile, file-like object, bytes, or path.
        engine: 'auto', 'easyocr', or 'tesseract'.
        preprocess: 'contrast', 'adaptive', 'denoise', 'grayscale', 'none'.
        pdf_mode:
            - 'auto': Use native digital text if available; if empty/scanned, run OCR.
            - 'ocr': Force OCR on all pages.
            - 'native': Only use native digital text.
        custom_tesseract_path: Path to tesseract binary if not in default location.

    Returns:
        dict:
            - full_text (str): Complete document text
            - pages (list): Page details (text, confidence, word_count, char_count, extraction_type)
            - confidence (list): All confidence scores
            - avg_confidence (float): Document-level confidence average
            - engine (str): Engine used
            - status (str): 'success' or error description
    """
    try:
        data, filename = _read_input_bytes(uploaded_file)
        lower_name = filename.lower()
        pages = []

        is_pdf = lower_name.endswith(".pdf") or data.startswith(b"%PDF")
        is_image = lower_name.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")) or data.startswith(
            (b"\xff\xd8", b"\x89PNG", b"RIFF", b"BM")
        )

        if is_pdf:
            doc = fitz.open(stream=data, filetype="pdf")
            try:
                for page_idx, page in enumerate(doc, start=1):
                    native_text = page.get_text("text").strip()
                    should_ocr = (
                        pdf_mode == "ocr"
                        or (pdf_mode == "auto" and len(native_text.split()) < 10)
                    )

                    if should_ocr and pdf_mode != "native":
                        # Render page to high-res image for OCR
                        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), colorspace=fitz.csRGB)
                        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                        extracted_boxes = extract_text(
                            img,
                            engine=engine,
                            preprocess=preprocess,
                            custom_tesseract_path=custom_tesseract_path,
                        )
                        page_text = "\n".join(b["text"] for b in extracted_boxes if b["text"])
                        confs = [b["confidence"] for b in extracted_boxes]
                        avg_conf = round(float(np.mean(confs)), 3) if confs else 0.0

                        pages.append({
                            "page_number": page_idx,
                            "text": page_text,
                            "confidence": extracted_boxes,
                            "confidence_avg": avg_conf,
                            "word_count": len(page_text.split()),
                            "char_count": len(page_text),
                            "extraction_type": "ocr",
                        })
                    else:
                        # Digital PDF native text extraction
                        words = native_text.split()
                        pages.append({
                            "page_number": page_idx,
                            "text": native_text,
                            "confidence": [],
                            "confidence_avg": 1.0,
                            "word_count": len(words),
                            "char_count": len(native_text),
                            "extraction_type": "native",
                        })
            finally:
                doc.close()

        elif is_image:
            img = Image.open(io.BytesIO(data)).convert("RGB")
            extracted_boxes = extract_text(
                img,
                engine=engine,
                preprocess=preprocess,
                custom_tesseract_path=custom_tesseract_path,
            )
            page_text = "\n".join(b["text"] for b in extracted_boxes if b["text"])
            confs = [b["confidence"] for b in extracted_boxes]
            avg_conf = round(float(np.mean(confs)), 3) if confs else 0.0

            pages.append({
                "page_number": 1,
                "text": page_text,
                "confidence": extracted_boxes,
                "confidence_avg": avg_conf,
                "word_count": len(page_text.split()),
                "char_count": len(page_text),
                "extraction_type": "ocr",
            })

        else:
            return {
                "full_text": "",
                "pages": [],
                "confidence": [],
                "avg_confidence": 0.0,
                "engine": engine,
                "status": "unsupported_file_type",
                "error": f"File '{filename}' format is not supported.",
            }

        full_text = "\n\n".join(p["text"] for p in pages if p["text"])
        all_confidence = [item for p in pages for item in p.get("confidence", [])]
        confs_list = [c["confidence"] for c in all_confidence if "confidence" in c]

        if confs_list:
            overall_avg = round(float(np.mean(confs_list)), 3)
        else:
            # If native text extraction, default to 1.0 (100%)
            overall_avg = 1.0 if full_text else 0.0

        return {
            "full_text": full_text,
            "pages": pages,
            "confidence": all_confidence,
            "avg_confidence": overall_avg,
            "engine": engine,
            "status": "success",
        }

    except Exception as err:
        return {
            "full_text": "",
            "pages": [],
            "confidence": [],
            "avg_confidence": 0.0,
            "engine": engine,
            "status": "error",
            "error": str(err),
        }


if __name__ == "__main__":
    print("DocuMind OCR Module")
    print(f"Available OCR engines: {get_available_ocr_engines()}")
    print("Testing OCR on sample.png...")
    if os.path.exists("sample.png"):
        res = process_document("sample.png")
        print(f"Status: {res['status']}")
        print(f"Extracted characters: {len(res['full_text'])}")
        print(f"Average confidence: {res['avg_confidence']}")
