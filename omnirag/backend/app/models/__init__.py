"""
Importing every model here means `from app.models import Base` gives Alembic
(and anything else that needs the full metadata) a complete picture. Models
that only import each other, without being imported somewhere central, are a
classic way to get silently-incomplete Alembic autogenerate migrations.
"""

from app.models.base import Base
from app.models.document import Document, DocumentStatus
from app.models.knowledge_base import KnowledgeBase
from app.models.user import User

__all__ = ["Base", "User", "KnowledgeBase", "Document", "DocumentStatus"]
