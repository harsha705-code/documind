# DocuMind - Single-file working prototype

## Requirements
- Windows 10/11
- Python 3.10, 3.11 or 3.12 recommended

## Setup (PowerShell, from the project folder)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run this once in the same terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Then open the Local URL printed by Streamlit, usually http://localhost:8501.

## Notes
- The OCR engine downloads its model files the first time it runs, so the first launch needs an internet connection.
- Normal PDFs are read directly. Scanned PDFs and image files are processed with local OCR.
- No Gemini API key or Tesseract installation is required.
- Automatic summaries and checks are lightweight prototype features. They are not legal, financial, or compliance verification.
