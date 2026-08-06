"""
User table.

Scope note: this is the User *table*, not auth. Phase 3 adds password
hashing, JWT issuance, and the register/login endpoints. Defining the table
now (rather than in Phase 3) is deliberate: `knowledge_bases`, `documents`,
and every other table in this schema has a foreign key to `users.id` for
per-user isolation, so the table needs to exist before anything that
references it.

`hashed_password` is a plain string column here — Phase 3 owns the actual
hashing (bcrypt/argon2) and never stores plaintext. Storing it as `str` here
just describes the column type, not the security guarantee.
"""

from sqlalchemy import Boolean, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class User(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    knowledge_bases: Mapped[list["KnowledgeBase"]] = relationship(
        back_populates="owner", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email}>"
