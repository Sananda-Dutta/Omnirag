"""
Message table.

One row per turn — both user questions and assistant answers are Messages,
distinguished by `role`. Storing both (not just assistant answers) is what
lets `GET /conversations/{id}` reconstruct the full back-and-forth for
display, and is also the raw material app/rag/memory.py reads to build the
`history` passed into LLMProvider.generate().

`model_used` / `context_found` are nullable and only ever set on assistant
messages — a user's question doesn't have a "model" that produced it. Kept
on Message rather than a separate table because they're small, per-message
facts with no independent lifecycle of their own (unlike citations, which
get their own table since a message can have many).
"""

import uuid
from typing import Literal

from sqlalchemy import Boolean, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

MessageRole = Literal["user", "assistant"]


class Message(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "messages"

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Assistant-only fields — null for role="user".
    model_used: Mapped[str | None] = mapped_column(String(128), nullable=True)
    context_found: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")
    citations: Mapped[list["Citation"]] = relationship(
        back_populates="message", cascade="all, delete-orphan", order_by="Citation.created_at"
    )

    def __repr__(self) -> str:
        return f"<Message id={self.id} role={self.role} conversation_id={self.conversation_id}>"
