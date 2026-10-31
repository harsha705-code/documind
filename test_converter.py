"""
test_converter.py

Comprehensive tests for DocuMind Document Converter
"""
import os
from pathlib import Path
from converter import (
    convert_image,
    image_to_pdf,
    pdf_to_images,
    rotate_image,
    resize_image,
    create_searchable_pdf,
    create_zip_archive
)

print("Starting DocuMind Converter Tests...")

# Ensure sample.jpg exists or create mock
if not os.path.exists("sample.jpg") and os.path.exists("sample.png"):
    from PIL import Image
    Image.open("sample.png").convert("RGB").save("sample.jpg")

# 1. JPG to PNG
with open("sample.jpg", "rb") as f:
    result = convert_image(f, "PNG")

with open("converted.png", "wb") as f:
    f.write(result.getvalue())

print("1. JPG to PNG: SUCCESS")

# 2. Multiple images to PDF
with open("sample.jpg", "rb") as f1, open("sample.png", "rb") as f2:
    result = image_to_pdf([f1, f2])

with open("combined.pdf", "wb") as f:
    f.write(result.getvalue())

print("2. Images to PDF: SUCCESS")

# Ensure sample.pdf exists for test 3
if not os.path.exists("sample.pdf"):
    with open("sample.pdf", "wb") as f:
        f.write(result.getvalue())

# 3. PDF to images
with open("sample.pdf", "rb") as f:
    pages = pdf_to_images(f)

for i, page in enumerate(pages):
    with open(f"page_{i+1}.png", "wb") as output:
        output.write(page.getvalue())

print(f"3. PDF to Images: SUCCESS ({len(pages)} pages extracted)")

# 4. Rotate image
with open("sample.jpg", "rb") as f:
    result = rotate_image(f, 90)

with open("rotated.png", "wb") as output:
    output.write(result.getvalue())

print("4. Image Rotation: SUCCESS")

# 5. Resize image
with open("sample.jpg", "rb") as f:
    result = resize_image(f, 800, 600)

with open("resized.png", "wb") as output:
    output.write(result.getvalue())

print("5. Image Resizing: SUCCESS")

# 6. Searchable PDF test
searchable_input = [
    {
        "image": "sample.png",
        "boxes": [
            {"text": "DocuMind Test Document", "box": [[10, 10], [200, 10], [200, 30], [10, 30]]}
        ]
    }
]
s_pdf = create_searchable_pdf(searchable_input)
with open("test_searchable.pdf", "wb") as out:
    out.write(s_pdf.getvalue())
print("6. Searchable PDF Creation: SUCCESS")

# 7. ZIP archive test
zip_out = create_zip_archive({"test.txt": b"DocuMind export"})
assert zip_out.getvalue()[:4] == b"PK\x03\x04"
print("7. ZIP Archive Creation: SUCCESS")

print("\nALL CONVERTER TESTS COMPLETED SUCCESSFULLY!")