"""
converter.py

DocuMind - Document Conversion & Processing Module

Features:
1. JPG/JPEG to PNG conversion
2. PNG to JPG/JPEG conversion
3. Multiple images to a single PDF
4. PDF pages to PNG/JPEG images (individual or ZIP bundle)
5. Searchable PDF generation (overlaying searchable text onto images)
6. Image rotation (90, 180, 270 degrees)
7. Image resizing with optional aspect ratio preservation
8. Document enhancement and optimization

Dependencies:
    Pillow, PyMuPDF (pymupdf)
"""

import io
import os
import zipfile
import warnings
from pathlib import Path
from typing import List, Union, Tuple, Optional, Dict, Any

# Suppress PyMuPDF legacy deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="fitz")
try:
    import pymupdf as fitz
except ImportError:
    import fitz

from PIL import Image, ImageOps, ImageEnhance, UnidentifiedImageError

# Supported image formats
SUPPORTED_IMAGE_FORMATS = ("JPEG", "JPG", "PNG", "WEBP", "BMP", "TIFF")

# Supported output formats
SUPPORTED_OUTPUT_FORMATS = {
    "JPG": "JPEG",
    "JPEG": "JPEG",
    "PNG": "PNG",
    "WEBP": "WEBP",
}


def _read_file(file: Any) -> bytes:
    """
    Read uploaded files or compatible file-like objects into bytes.

    Returns:
        bytes: Raw file contents.
    """
    if file is None:
        raise ValueError("No file was provided.")

    if isinstance(file, bytes):
        data = file
    elif isinstance(file, bytearray):
        data = bytes(file)
    elif isinstance(file, (str, Path)):
        with open(file, "rb") as f:
            data = f.read()
    elif hasattr(file, "read"):
        orig_pos = None
        try:
            if hasattr(file, "tell"):
                try:
                    orig_pos = file.tell()
                except Exception:
                    orig_pos = None
            if hasattr(file, "seek"):
                file.seek(0)
            data = file.read()
        finally:
            if orig_pos is not None and hasattr(file, "seek"):
                try:
                    file.seek(orig_pos)
                except Exception:
                    pass
    else:
        raise TypeError("Input must be a file path, bytes, or file-like object.")

    if not data:
        raise ValueError("The provided document file is empty.")

    return bytes(data)


def _open_image(image_file: Any) -> Image.Image:
    """
    Open and validate an uploaded image.

    Returns:
        PIL.Image.Image: Loaded image object with EXIF orientation handled.
    """
    data = _read_file(image_file)
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        # Normalize EXIF rotation if present
        return ImageOps.exif_transpose(image)
    except UnidentifiedImageError:
        raise ValueError("Invalid or unsupported image file format.")
    except OSError as err:
        raise ValueError(f"Unable to decode image: {err}")


def convert_image(input_file: Any, output_format: str, quality: int = 95) -> io.BytesIO:
    """
    Convert an image between JPG, JPEG, and PNG formats.

    Args:
        input_file: Uploaded image, bytes, or file-like object.
        output_format (str): 'JPG', 'JPEG', or 'PNG'.
        quality (int): Output compression quality (1-100) for JPEG/WEBP.

    Returns:
        io.BytesIO: Converted image stream.
    """
    output_format = str(output_format).upper().strip()
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(f"Invalid format '{output_format}'. Choose JPG, JPEG, or PNG.")

    image = _open_image(input_file)
    target_format = SUPPORTED_OUTPUT_FORMATS[output_format]
    output = io.BytesIO()

    try:
        if target_format == "JPEG":
            # Flatten transparency to white background
            if image.mode in ("RGBA", "LA") or ("transparency" in image.info):
                rgba_image = image.convert("RGBA")
                background = Image.new("RGB", rgba_image.size, (255, 255, 255))
                background.paste(rgba_image, mask=rgba_image.getchannel("A"))
                image = background
            elif image.mode != "RGB":
                image = image.convert("RGB")

            image.save(
                output,
                format="JPEG",
                quality=quality,
                optimize=True,
                subsampling=0,
            )
        else:
            image.save(output, format="PNG", optimize=True)

        output.seek(0)
        return output
    except OSError as err:
        raise ValueError(f"Image conversion failed: {err}")


def image_to_pdf(images: Union[List[Any], Any], output_path: Optional[str] = None) -> io.BytesIO:
    """
    Combine one or more images into a single high-quality PDF.

    Args:
        images: List of uploaded images or single image.
        output_path: Optional path to save the PDF directly to disk.

    Returns:
        io.BytesIO: In-memory PDF buffer.
    """
    if not images:
        raise ValueError("No images provided for PDF generation.")

    if not isinstance(images, (list, tuple)):
        images = [images]

    pil_images: List[Image.Image] = []
    try:
        for img_item in images:
            img = _open_image(img_item)
            if img.mode != "RGB":
                img = img.convert("RGB")
            pil_images.append(img)

        output = io.BytesIO()
        first_img = pil_images[0]
        rest = pil_images[1:]

        first_img.save(
            output,
            format="PDF",
            save_all=True,
            append_images=rest,
            resolution=150.0,
            quality=95,
        )
        output.seek(0)

        if output_path:
            with open(output_path, "wb") as f:
                f.write(output.getvalue())

        return output
    except Exception as err:
        raise ValueError(f"Image to PDF conversion failed: {err}")
    finally:
        for img in pil_images:
            try:
                img.close()
            except Exception:
                pass


def pdf_to_images(
    pdf_file: Any,
    dpi_scale: float = 2.0,
    image_format: str = "PNG"
) -> List[io.BytesIO]:
    """
    Convert every page of a PDF document into high-resolution images.

    Args:
        pdf_file: Uploaded PDF file, bytes, or file-like object.
        dpi_scale: Zoom scaling (2.0 = ~144 DPI, 3.0 = ~216 DPI).
        image_format: 'PNG' or 'JPEG'.

    Returns:
        List[io.BytesIO]: List of rendered image streams.
    """
    pdf_bytes = _read_file(pdf_file)
    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError("The provided file does not appear to be a valid PDF document.")

    images: List[io.BytesIO] = []
    doc = None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if doc.page_count == 0:
            raise ValueError("The PDF document contains 0 pages.")

        format_str = "png" if image_format.upper() == "PNG" else "jpeg"
        matrix = fitz.Matrix(dpi_scale, dpi_scale)

        for page in doc:
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
            img_bytes = pix.tobytes(format_str)
            buf = io.BytesIO(img_bytes)
            buf.seek(0)
            images.append(buf)

        return images
    except fitz.FileDataError:
        raise ValueError("The PDF file is corrupted or could not be read.")
    except Exception as err:
        if isinstance(err, ValueError):
            raise
        raise ValueError(f"PDF to images conversion failed: {err}")
    finally:
        if doc is not None:
            doc.close()


def rotate_image(image_file: Any, angle: int) -> io.BytesIO:
    """
    Rotate an image by 90, 180, or 270 degrees clockwise or counterclockwise.

    Args:
        image_file: Uploaded image, bytes, or path.
        angle (int): 90, 180, 270 degrees.

    Returns:
        io.BytesIO: Rotated image as PNG.
    """
    valid_angles = (90, 180, 270)
    if angle not in valid_angles:
        raise ValueError(f"Invalid angle {angle}. Choose from {valid_angles} degrees.")

    img = _open_image(image_file)
    try:
        # PIL rotates counterclockwise for positive angles.
        # In UI clockwise 90 = -90 or 270.
        rotated = img.rotate(angle, expand=True)
        out = io.BytesIO()
        if rotated.mode not in ("RGB", "RGBA"):
            rotated = rotated.convert("RGBA")
        rotated.save(out, format="PNG", optimize=True)
        out.seek(0)
        return out
    except Exception as err:
        raise ValueError(f"Image rotation failed: {err}")


def resize_image(
    image_file: Any,
    width: int,
    height: int,
    preserve_aspect_ratio: bool = True
) -> io.BytesIO:
    """
    Resize image to target dimensions.
    """
    if not isinstance(width, int) or not isinstance(height, int):
        raise TypeError("Width and height must be integers.")
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive numbers.")

    img = _open_image(image_file)
    try:
        if preserve_aspect_ratio:
            resized = ImageOps.contain(img, (width, height), method=Image.Resampling.LANCZOS)
        else:
            resized = img.resize((width, height), method=Image.Resampling.LANCZOS)

        out = io.BytesIO()
        if resized.mode not in ("RGB", "RGBA"):
            resized = resized.convert("RGBA")
        resized.save(out, format="PNG", optimize=True)
        out.seek(0)
        return out
    except Exception as err:
        raise ValueError(f"Image resizing failed: {err}")


def create_searchable_pdf(
    images_with_ocr: List[Dict[str, Any]],
    output_path: Optional[str] = None
) -> io.BytesIO:
    """
    Generate a Searchable PDF where scanned image pages are backed by an
    invisible searchable text layer.

    Args:
        images_with_ocr: List of dicts, each with:
            - 'image': PIL Image, bytes, or path
            - 'boxes': list of dicts with 'text', 'box' [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
                       or simply list of items from ocr_module
            - 'full_text': optional fallback plain text
        output_path: Optional destination path.

    Returns:
        io.BytesIO: Searchable PDF document buffer.
    """
    if not images_with_ocr:
        raise ValueError("No images provided to create searchable PDF.")

    doc = fitz.open()
    try:
        for item in images_with_ocr:
            img_input = item.get("image")
            img = _open_image(img_input)
            w, h = img.size

            # Create PDF page matching pixel dimensions (72 points per inch baseline)
            page = doc.new_page(width=float(w), height=float(h))

            # Save image to buffer to insert onto PDF page
            img_buf = io.BytesIO()
            img.save(img_buf, format="PNG")
            img_buf.seek(0)

            # Insert original high-res image
            rect = fitz.Rect(0, 0, float(w), float(h))
            page.insert_image(rect, stream=img_buf.getvalue())

            # Insert searchable text layer
            boxes = item.get("boxes", [])
            if boxes:
                for b in boxes:
                    txt = b.get("text", "").strip()
                    if not txt:
                        continue
                    coords = b.get("box")
                    if coords and len(coords) >= 4:
                        # coords is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                        xs = [p[0] for p in coords]
                        ys = [p[1] for p in coords]
                        min_x, max_x = min(xs), max(xs)
                        min_y, max_y = min(ys), max(ys)
                        box_h = max(8.0, max_y - min_y)
                        font_size = max(6.0, box_h * 0.75)
                        point = fitz.Point(min_x, max_y - 2)
                    else:
                        font_size = 10.0
                        point = fitz.Point(20, 20)

                    # render_mode=3 is PDF specification invisible text (searchable, selectable)
                    try:
                        page.insert_text(
                            point,
                            txt,
                            fontsize=font_size,
                            render_mode=3,
                            color=(0, 0, 0)
                        )
                    except Exception:
                        pass
            elif item.get("full_text"):
                # Fallback: insert text block invisibly across the page
                try:
                    page.insert_textbox(
                        rect,
                        item.get("full_text", ""),
                        fontsize=9,
                        render_mode=3,
                        color=(0, 0, 0)
                    )
                except Exception:
                    pass

        out = io.BytesIO()
        doc.save(out)
        out.seek(0)

        if output_path:
            with open(output_path, "wb") as f:
                f.write(out.getvalue())

        return out
    finally:
        doc.close()


def create_zip_archive(files_dict: Dict[str, Union[bytes, io.BytesIO]]) -> io.BytesIO:
    """
    Create an in-memory ZIP archive from a dictionary of filename -> content.

    Args:
        files_dict: mapping of filename string to bytes or BytesIO.

    Returns:
        io.BytesIO: ZIP file stream ready for download.
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for fname, data in files_dict.items():
            if hasattr(data, "getvalue"):
                content = data.getvalue()
            elif isinstance(data, (bytes, bytearray)):
                content = bytes(data)
            else:
                content = str(data).encode("utf-8")
            zip_file.writestr(fname, content)

    zip_buffer.seek(0)
    return zip_buffer


# Test runner
if __name__ == "__main__":
    print("DocuMind converter module initialized successfully.")
