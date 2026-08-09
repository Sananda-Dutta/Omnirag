"""
Document table.

Status is modeled as an explicit enum rather than a free-text string or a
boolean "is_processed" flag, because ingestion genuinely has more than two
states a client needs to distinguish: a document sitting in the upload
queue looks different from one that's actively being parsed, and both look
very different from one that failed and needs the user to see *why*
(`error_message`) rather than silently vanishing.

`extracted_text` lives directly on this row (not a separate table) for
Phase 4. Phase 5 (chunking) reads from this column to produce
`document_chunks` rows — keeping raw extracted text and derived chunks in
separate tables mirrors the ingestion pipeline's own stages: extraction
produces text, chunking is a distinct, later transformation of it.
"""

import enum
import uuid

from sqlalchemy import Enum, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class DocumentStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "documents"

    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    content_type: Mapped[str] = mapped_column(String(128), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    # Path on disk (Phase 4) — swappable for an S3 key/URL later without
    # touching any other column, since callers only ever read the file back
    # through the StorageBackend abstraction, never this path directly.
    storage_path: Mapped[str] = mapped_column(String(1024), nullable=False)

    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus, name="document_status"),
        default=DocumentStatus.PENDING,
        nullable=False,
        index=True,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    extracted_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    knowledge_base: Mapped["KnowledgeBase"] = relationship(back_populates="documents")

    def __repr__(self) -> str:
        return f"<Document id={self.id} filename={self.filename!r} status={self.status}>"
