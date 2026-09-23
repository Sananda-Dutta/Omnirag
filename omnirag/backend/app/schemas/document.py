import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.document import DocumentStatus


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    knowledge_base_id: uuid.UUID
    filename: str
    content_type: str
    # Nullable, not int: Phase 12 URL-sourced documents have no uploaded
    # file at all (no bytes were ever read into this process), so there is
    # no honest byte count to report until the worker fetches the page.
    # Coercing this to 0 would be a fabricated value, not a missing one.
    file_size_bytes: int | None
    # Phase 12: set for URL-sourced documents (the URL the content was
    # fetched from), null for uploaded files. Exposed so a client can show
    # "from https://..." instead of guessing from `filename` alone.
    source_url: str | None = None
    status: DocumentStatus
    error_message: str | None
    page_count: int | None
    char_count: int | None
    created_at: datetime
    updated_at: datetime


class DocumentDetail(DocumentRead):
    """Includes extracted_text — only returned from the single-document
    GET endpoint, not from list endpoints, so listing a knowledge base with
    50 documents doesn't ship 50 full-text blobs in one response."""

    extracted_text: str | None
