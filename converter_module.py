"""
converter_module.py

DocuMind - Converter Module Adapter
This module maintains full backward compatibility by re-exporting all functions
from converter.py.
"""

from converter import (
    convert_image,
    image_to_pdf,
    pdf_to_images,
    rotate_image,
    resize_image,
    create_searchable_pdf,
    create_zip_archive,
    SUPPORTED_IMAGE_FORMATS,
    SUPPORTED_OUTPUT_FORMATS,
    _read_file,
    _open_image,
)

__all__ = [
    "convert_image",
    "image_to_pdf",
    "pdf_to_images",
    "rotate_image",
    "resize_image",
    "create_searchable_pdf",
    "create_zip_archive",
    "SUPPORTED_IMAGE_FORMATS",
    "SUPPORTED_OUTPUT_FORMATS",
]

if __name__ == "__main__":
    print("DocuMind Converter Module (Aliased to converter.py)")
    print("All conversion routines ready.")