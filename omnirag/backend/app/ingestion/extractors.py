"""
Text extraction.

One function per format, dispatched by extension. Kept as plain functions
(not a class hierarchy) because there's no shared state or behavior between
extractors beyond "take bytes, return text" — an ExtractorRegistry/ABC would
be the "unnecessary abstraction" the project brief explicitly warns against.
If URL ingestion (Phase 12) or OCR needs shared retry/cleanup logic later,
that's the point to introduce one, not before.

Every extractor raises `ExtractionError` on failure (corrupt file, wrong
magic bytes despite the extension, etc.) rather than letting the underlying
library's exception propagate — callers (the Celery task) only need to know
"extraction failed, here's why," not which library threw what.
"""

from dataclasses import dataclass
from io import BytesIO

import docx
from pypdf import PdfReader
from pypdf.errors import PdfReadError


class ExtractionError(Exception):
    pass


@dataclass
class ExtractionResult:
    text: str
    page_count: int | None


def extract_pdf(content: bytes) -> ExtractionResult:
    try:
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            # pypdf can sometimes still read encrypted PDFs with an empty
            # password, but treating "requires a password" as a hard failure
            # is the honest behavior — silently returning blank/garbled text
            # would be worse than a clear error the user can act on.
            raise ExtractionError("PDF is password-protected.")
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages).strip()
        if not text:
            raise ExtractionError(
                "No extractable text found (the PDF may be scanned/image-only; "
                "OCR support is a planned future improvement, not yet implemented)."
            )
        return ExtractionResult(text=text, page_count=len(reader.pages))
    except PdfReadError as exc:
        raise ExtractionError(f"Could not parse PDF: {exc}") from exc


def extract_docx(content: bytes) -> ExtractionResult:
    try:
        document = docx.Document(BytesIO(content))
    except Exception as exc:  # python-docx raises varied/undocumented exceptions on bad input
        raise ExtractionError(f"Could not parse DOCX: {exc}") from exc

    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    text = "\n\n".join(paragraphs).strip()
    if not text:
        raise ExtractionError("DOCX contains no extractable text.")
    return ExtractionResult(text=text, page_count=None)  # DOCX has no fixed "pages"


def extract_plain_text(content: bytes) -> ExtractionResult:
    try:
        text = content.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise ExtractionError(f"File is not valid UTF-8 text: {exc}") from exc

    if not text:
        raise ExtractionError("File is empty.")
    return ExtractionResult(text=text, page_count=None)


_EXTRACTORS = {
    ".pdf": extract_pdf,
    ".docx": extract_docx,
    ".txt": extract_plain_text,
    ".md": extract_plain_text,
}


def extract_text(content: bytes, extension: str) -> ExtractionResult:
    extractor = _EXTRACTORS.get(extension.lower())
    if extractor is None:
        raise ExtractionError(f"No extractor registered for extension '{extension}'.")
    return extractor(content)
