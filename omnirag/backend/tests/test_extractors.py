"""
Extractor tests.

Uses real generated files (reportlab for PDF, python-docx for DOCX) rather
than mocking the underlying parser libraries — a mock would just assert
"pypdf.PdfReader was called," not that extraction actually recovers the
text that was in the file.
"""

from io import BytesIO

import docx
import pytest
from pypdf import PdfWriter
from reportlab.pdfgen import canvas

from app.ingestion.extractors import ExtractionError, extract_text


def _make_pdf_with_text(text: str) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer)
    c.drawString(72, 750, text)
    c.save()
    return buffer.getvalue()


def _make_encrypted_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.encrypt(user_password="secret", owner_password="secret")
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _make_docx_with_text(paragraphs: list[str]) -> bytes:
    document = docx.Document()
    for p in paragraphs:
        document.add_paragraph(p)
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_extract_plain_text():
    result = extract_text(b"Gradient descent minimizes a loss function.", ".txt")
    assert "Gradient descent" in result.text
    assert result.page_count is None


def test_extract_markdown_uses_same_path_as_txt():
    result = extract_text(b"# Heading\n\nSome content.", ".md")
    assert "Heading" in result.text


def test_extract_plain_text_rejects_non_utf8():
    with pytest.raises(ExtractionError):
        extract_text(b"\xff\xfe\x00\x01invalid", ".txt")


def test_extract_plain_text_rejects_empty_file():
    with pytest.raises(ExtractionError):
        extract_text(b"   \n  ", ".txt")


def test_extract_pdf_recovers_real_text():
    pdf_bytes = _make_pdf_with_text("RAG combines retrieval with generation.")
    result = extract_text(pdf_bytes, ".pdf")
    assert "RAG combines retrieval" in result.text
    assert result.page_count == 1


def test_extract_pdf_rejects_encrypted_file():
    with pytest.raises(ExtractionError, match="password-protected"):
        extract_text(_make_encrypted_pdf(), ".pdf")


def test_extract_pdf_rejects_garbage_bytes():
    with pytest.raises(ExtractionError):
        extract_text(b"this is not a pdf at all", ".pdf")


def test_extract_docx_recovers_real_text():
    docx_bytes = _make_docx_with_text(["Introduction to embeddings.", "They map text to vectors."])
    result = extract_text(docx_bytes, ".docx")
    assert "Introduction to embeddings" in result.text
    assert "map text to vectors" in result.text
    assert result.page_count is None


def test_extract_docx_rejects_empty_document():
    empty_docx = _make_docx_with_text([])
    with pytest.raises(ExtractionError):
        extract_text(empty_docx, ".docx")


def test_extract_unsupported_extension():
    with pytest.raises(ExtractionError):
        extract_text(b"whatever", ".exe")
