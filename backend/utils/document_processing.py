"""Document processing utilities for extracting text from various formats."""

import os
from io import BytesIO
from typing import Union

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import pypdf
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

from fastapi import UploadFile


def extract_text_from_pdf(file_data: bytes) -> str:
    """Extract text from a PDF file."""
    if not PDF_AVAILABLE:
        raise RuntimeError("pypdf is not installed")
    reader = pypdf.PdfReader(BytesIO(file_data))
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts)


def extract_text_from_docx(file_data: bytes) -> str:
    """Extract text from a DOCX file."""
    if not DOCX_AVAILABLE:
        raise RuntimeError("python-docx is not installed")
    from docx import Document as WordDocument
    import tempfile
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_data)
        tmp.flush()
        doc = WordDocument(tmp.name)
        text = "\n".join([p.text for p in doc.paragraphs])
        os.unlink(tmp.name)
    return text


def read_file_to_text(file: UploadFile) -> str:
    """
    Read an uploaded file and extract its text.
    
    Supports PDF, DOCX, and plain text files.
    """
    data = file.file.read()
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == ".pdf":
        return extract_text_from_pdf(data)
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(data)
    else:
        # Plain text or unknown formats - try UTF-8 first, then latin1
        try:
            return data.decode("utf-8")
        except Exception:
            return data.decode("latin1", errors="ignore")

