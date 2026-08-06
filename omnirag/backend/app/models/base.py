"""
Base model class and shared mixins.

Why UUID primary keys instead of auto-increment integers:
    - Documents/knowledge-bases will eventually be referenced in URLs
      (GET /documents/{id}) and in vector DB payloads (Qdrant point IDs).
      UUIDs mean those IDs are non-guessable (an integer ID lets someone
      enumerate /documents/1, /documents/2, ... — a real issue given the
      per-user isolation requirement) and safely generatable client-side or
      in background workers without a round-trip to get the next sequence
      value.
    - Cost: UUIDs are 16 bytes vs 4-8 for ints, and don't index quite as
      compactly. For this project's scale, the security property is worth
      more than the marginal storage/index difference.

Why a TimestampMixin instead of repeating created_at/updated_at on every
model: every table in this schema (users, knowledge_bases, documents,
messages, ...) needs both. Defining it once means one place to fix if the
convention ever changes (e.g. switching to timezone-naive by mistake is the
kind of bug you want to be impossible, not just avoided by convention).
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class UUIDPrimaryKeyMixin:
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
