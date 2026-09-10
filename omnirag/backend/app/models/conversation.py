"""
Conversation table.

`summary` (Phase 10): a running compaction of everything before the most
recent message window (see app/rag/memory.py), so a long conversation
doesn't require sending its entire history to the LLM on every turn. Null
until the conversation actually exceeds CONVERSATION_RECENT_MESSAGE_WINDOW
messages — most conversations never need one.

`knowledge_base_id` is nullable and set once, at creation, from whatever
scope the first message used — a conversation doesn't change which
knowledge base it searches partway through. This mirrors how `/chat`
already accepted an optional `knowledge_base_id` in Phase 7; Phase 10 just
gives that choice somewhere durable to live instead of being re-specified
(and potentially changed) on every single message.
"""

import uuid

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Conversation(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "conversations"

    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    knowledge_base_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_bases.id", ondelete="SET NULL"), nullable=True
    )
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    # How many of this conversation's oldest messages (in chronological
    # order) have already been folded into `summary`. Without this, naively
    # recomputing "everything before the last N messages" on every turn
    # would re-summarize the same already-summarized messages again each
    # time the window is recalculated — this counter is what makes
    # summarization incremental (each message folded in exactly once) 
    # instead of repeated. See app/rag/memory.py.
    summarized_message_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at"
    )

    def __repr__(self) -> str:
        return f"<Conversation id={self.id} owner_id={self.owner_id}>"
