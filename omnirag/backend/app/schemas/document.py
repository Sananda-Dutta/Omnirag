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
    file_size_bytes: int
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
