import io
import os
from typing import BinaryIO

from docx import Document as DocxDocument
from pypdf import PdfReader

from app.utils.errors import ValidationError

try:
    import pytesseract
    from pdf2image import convert_from_bytes
except Exception:  # noqa: BLE001
    pytesseract = None
    convert_from_bytes = None


ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".doc"}


def parse_uploaded_file(file_stream: BinaryIO, filename: str) -> str:
    ext = os.path.splitext((filename or "").lower())[1]
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError("Unsupported file type. Use TXT, MD, PDF, DOCX, or DOC.")

    file_bytes = file_stream.read()
    if not file_bytes:
        raise ValidationError("Uploaded file is empty")

    if ext in {".txt", ".md"}:
        return _decode_text(file_bytes)

    if ext == ".pdf":
        return _extract_pdf_text(file_bytes)

    if ext == ".docx":
        return _extract_docx_text(file_bytes)

    if ext == ".doc":
        # Legacy .doc parsing without external system binaries is limited.
        return _decode_text(file_bytes)

    raise ValidationError("Unable to parse uploaded file")


def _decode_text(file_bytes: bytes) -> str:
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return file_bytes.decode(encoding, errors="ignore").strip()
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="ignore").strip()


def _extract_pdf_text(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages).strip()
        if not text:
            text = _extract_pdf_text_with_ocr(file_bytes)
        if not text:
            raise ValidationError("No readable text found in PDF file")
        return text
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(f"Failed to parse PDF: {exc}") from exc


def _extract_docx_text(file_bytes: bytes) -> str:
    try:
        document = DocxDocument(io.BytesIO(file_bytes))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs).strip()
        if not text:
            raise ValidationError("No readable text found in DOCX file")
        return text
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(f"Failed to parse DOCX: {exc}") from exc


def _extract_pdf_text_with_ocr(file_bytes: bytes) -> str:
    if pytesseract is None or convert_from_bytes is None:
        return ""
    try:
        images = convert_from_bytes(file_bytes)
        extracted = []
        for image in images:
            extracted.append(pytesseract.image_to_string(image) or "")
        return "\n".join(extracted).strip()
    except Exception:  # noqa: BLE001
        return ""
