"""
KnowledgeBase table.

A user can create multiple knowledge bases (e.g. "ML coursework" vs "work
research"). Every document, chunk, and conversation eventually scopes to a
knowledge_base_id, which scopes to a user_id — that two-level chain is what
"different users cannot access each other's documents" (and, within a user,
"documents don't bleed across unrelated knowledge bases") actually means at
the schema level. Phase 3's auth dependency + Phase 4's document endpoints
will enforce this at the query level (always filter by
`owner_id == current_user.id`), but the FK relationship is what makes that
enforceable in the first place.
"""

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class KnowledgeBase(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "knowledge_bases"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    owner_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    owner: Mapped["User"] = relationship(back_populates="knowledge_bases")
    documents: Mapped[list["Document"]] = relationship(
        back_populates="knowledge_base", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<KnowledgeBase id={self.id} name={self.name!r} owner_id={self.owner_id}>"
