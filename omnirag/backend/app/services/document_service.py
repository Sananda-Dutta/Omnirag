"""
Document service.

Validation order matters here: extension check first (free, no I/O), then
size check (already-known from the upload, no I/O), and only then do we
write anything to disk or the DB. Rejecting a bad upload should never leave
a half-written file or an orphan DB row behind.
"""

import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.ingestion.storage import get_storage_backend
from app.models.document import Document
from app.services.knowledge_base_service import get_knowledge_base


class UnsupportedFileTypeError(Exception):
    pass


class FileTooLargeError(Exception):
    pass


class DocumentNotFoundError(Exception):
    pass


def validate_upload(filename: str, size_bytes: int) -> str:
    """Returns the validated lowercase extension, or raises."""
    extension = Path(filename).suffix.lower()
    if extension not in settings.ALLOWED_UPLOAD_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"'{extension or '(no extension)'}' is not supported. "
            f"Allowed types: {', '.join(settings.ALLOWED_UPLOAD_EXTENSIONS)}"
        )
    if size_bytes > settings.max_upload_size_bytes:
        raise FileTooLargeError(
            f"File is {size_bytes / (1024 * 1024):.1f}MB, "
            f"which exceeds the {settings.MAX_UPLOAD_SIZE_MB}MB limit."
        )
    return extension


async def upload_document(
    db: AsyncSession,
    owner_id: uuid.UUID,
    knowledge_base_id: uuid.UUID,
    filename: str,
    content_type: str,
    content: bytes,
) -> Document:
    # Raises KnowledgeBaseNotFoundError (-> 404) if this KB isn't the
    # caller's — this is what stops a user from uploading into someone
    # else's knowledge base by guessing/brute-forcing a UUID.
    await get_knowledge_base(db, owner_id, knowledge_base_id)

    extension = validate_upload(filename, len(content))

    storage = get_storage_backend()
    storage_path = await storage.save(content, extension)

    document = Document(
        knowledge_base_id=knowledge_base_id,
        filename=filename,
        content_type=content_type,
        file_size_bytes=len(content),
        storage_path=storage_path,
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)

    # Imported here, not at module load: importing app.workers.tasks eagerly
    # pulls in the Celery app (which reads settings.REDIS_URL and, if a
    # broker connection is attempted at import time in some Celery configs,
    # can slow down or fail app startup in environments with no Redis
    # reachable yet, e.g. a bare `pip install` sanity check). Deferring the
    # import to the one call site that needs it keeps that coupling local.
    from app.workers.tasks import process_document_task

    process_document_task.delay(str(document.id))

    return document


async def list_documents(
    db: AsyncSession, owner_id: uuid.UUID, knowledge_base_id: uuid.UUID
) -> list[Document]:
    await get_knowledge_base(db, owner_id, knowledge_base_id)  # ownership check
    result = await db.execute(
        select(Document)
        .where(Document.knowledge_base_id == knowledge_base_id)
        .order_by(Document.created_at.desc())
    )
    return list(result.scalars().all())


async def get_document(db: AsyncSession, owner_id: uuid.UUID, document_id: uuid.UUID) -> Document:
    # Joins through KnowledgeBase to enforce ownership in the query itself,
    # same reasoning as knowledge_base_service — no fetch-then-check.
    from app.models.knowledge_base import KnowledgeBase

    result = await db.execute(
        select(Document)
        .join(KnowledgeBase, Document.knowledge_base_id == KnowledgeBase.id)
        .where(Document.id == document_id, KnowledgeBase.owner_id == owner_id)
    )
    document = result.scalar_one_or_none()
    if document is None:
        raise DocumentNotFoundError(str(document_id))
    return document


async def delete_document(db: AsyncSession, owner_id: uuid.UUID, document_id: uuid.UUID) -> None:
    document = await get_document(db, owner_id, document_id)
    get_storage_backend().delete(document.storage_path)
    await db.delete(document)
    await db.commit()


async def list_chunks(
    db: AsyncSession, owner_id: uuid.UUID, document_id: uuid.UUID
) -> list["DocumentChunk"]:
    from app.models.document_chunk import DocumentChunk

    await get_document(db, owner_id, document_id)  # ownership check; raises DocumentNotFoundError
    result = await db.execute(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index)
    )
    return list(result.scalars().all())
