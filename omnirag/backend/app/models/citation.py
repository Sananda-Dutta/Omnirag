"""
Citation table.

Deferred from Phase 9 (which built the citation *data* per-request but had
nothing to attach it to durably — no Message existed yet). Now that
messages exist, each assistant Message's citations get persisted here.

Deliberately denormalized rather than a bare FK to document_chunks:
`chunk_id` is kept (nullable, ON DELETE SET NULL) for as long as the source
chunk exists, but `document_id`, `document_filename`, `chunk_index`,
`page_number`, `score`, and `text_snippet` are stored as plain values, not
foreign keys. A citation is a historical record of what an answer was
grounded in at the time it was generated — if the source document is later
deleted or the document is reprocessed (regenerating its chunks with new
IDs), the citation should still show what it originally cited, not go
blank or point at nothing. This is the same reasoning as denormalizing
owner_id/knowledge_base_id onto DocumentChunk in Phase 5, applied one level
up: a record whose whole purpose is durability shouldn't be structurally
dependent on the thing it's recording surviving.
"""

import uuid

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Citation(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "citations"

    message_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_chunks.id", ondelete="SET NULL"), nullable=True
    )

    # Denormalized historical record — see module docstring for why these
    # are plain values, not FKs, to document_id/etc.
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    document_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    text_snippet: Mapped[str] = mapped_column(Text, nullable=False)

    message: Mapped["Message"] = relationship(back_populates="citations")

    def __repr__(self) -> str:
        return f"<Citation id={self.id} message_id={self.message_id} document={self.document_filename!r}>"
