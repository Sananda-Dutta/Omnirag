"""
Document processing task.

Celery tasks are synchronous by design (that's how the worker pool model
works), but our DB layer and storage backend are async. Rather than
maintaining a second, sync SQLAlchemy stack just for the worker — real
duplication for no real benefit — this task wraps a single async function
with `asyncio.run()`. Each task invocation gets its own event loop, which is
exactly the isolation a worker process handling one task at a time wants.

Status transitions this task is responsible for:
    PENDING (set by the upload endpoint)
      -> PROCESSING (set here, first thing, so a client polling the
         document sees it moved off the queue)
      -> COMPLETED (extraction succeeded AND chunking+embedding succeeded —
         "completed" means the document is actually searchable, not merely
         that text was recovered from the file)
      -> FAILED (extraction, chunking, or embedding raised; error_message
         set to something a user can act on where possible, e.g. "PDF is
         password-protected")

Any exception NOT anticipated by the extractors (a bug, not a bad file) is
still caught at the top level and recorded as FAILED with a generic message
— a document should never get stuck in PROCESSING forever because of an
unhandled exception, and internal error details shouldn't be handed back to
the API response.

Reprocessing is idempotent: existing chunks for a document are cleared
before new ones are inserted (via `document.chunks.clear()`, relying on the
model's `cascade="all, delete-orphan"`), so re-running this task for the
same document — e.g. after a chunking bug fix — doesn't accumulate
duplicate chunks.
"""

import asyncio
from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.logging import get_logger
from app.embeddings.factory import get_embedding_provider
from app.ingestion.chunking import chunk_text
from app.ingestion.extractors import ExtractionError, extract_text
from app.ingestion.storage import get_storage_backend
from app.models.document import Document, DocumentStatus
from app.models.document_chunk import DocumentChunk
from app.workers.celery_app import celery_app

logger = get_logger(__name__)


async def _process_document(document_id: str) -> None:
    # A dedicated engine per task invocation (not the app's shared `engine`
    # from app.database.session) — the worker process is not the API
    # process, and giving it its own small pool avoids fighting the API's
    # pool for connections under load.
    engine = create_async_engine(settings.DATABASE_URL, pool_size=2, max_overflow=0)
    session_local = async_sessionmaker(bind=engine, expire_on_commit=False)

    try:
        async with session_local() as db:
            await _run(db, UUID(document_id))
    finally:
        await engine.dispose()


async def _run(db: AsyncSession, document_id: UUID) -> None:
    document = (
        await db.execute(
            select(Document)
            .options(selectinload(Document.knowledge_base), selectinload(Document.chunks))
            .where(Document.id == document_id)
        )
    ).scalar_one_or_none()

    if document is None:
        logger.warning("document_not_found", extra={"context": {"document_id": str(document_id)}})
        return

    document.status = DocumentStatus.PROCESSING
    await db.commit()

    try:
        storage = get_storage_backend()
        content = await storage.read(document.storage_path)
        extension = Path(document.filename).suffix

        result = extract_text(content, extension)

        document.extracted_text = result.text
        document.page_count = result.page_count
        document.char_count = len(result.text)

        await _chunk_and_embed(db, document)

        document.status = DocumentStatus.COMPLETED
        document.error_message = None

    except ExtractionError as exc:
        document.status = DocumentStatus.FAILED
        document.error_message = str(exc)
        logger.info(
            "document_extraction_failed",
            extra={"context": {"document_id": str(document_id), "reason": str(exc)}},
        )

    except Exception as exc:  # noqa: BLE001 — intentional catch-all, see module docstring
        document.status = DocumentStatus.FAILED
        document.error_message = "An unexpected error occurred while processing this document."
        logger.error(
            "document_processing_unexpected_error",
            extra={"context": {"document_id": str(document_id)}},
            exc_info=exc,
        )

    await db.commit()


async def _chunk_and_embed(db: AsyncSession, document: Document) -> None:
    """Splits document.extracted_text into chunks, embeds each one, and
    (re)persists them as DocumentChunk rows. Any exception here propagates
    to _run's except block and marks the document FAILED — a document whose
    text extracted fine but couldn't be chunked/embedded is not actually
    usable for retrieval, so COMPLETED must not be reported for it."""
    pieces = chunk_text(
        document.extracted_text or "",
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    # Idempotent reprocessing: clear existing chunks before inserting new
    # ones (cascade="all, delete-orphan" on Document.chunks issues the
    # deletes on flush/commit). Safe to run this task more than once for the
    # same document without accumulating duplicates.
    document.chunks.clear()

    if not pieces:
        # Legitimate outcome (e.g. a file that extracted to only whitespace)
        # — zero chunks, not an error. The document is still COMPLETED;
        # there's simply nothing to retrieve from it.
        return

    provider = get_embedding_provider()
    vectors = await provider.embed_texts([piece.text for piece in pieces])

    owner_id = document.knowledge_base.owner_id
    for piece, vector in zip(pieces, vectors):
        document.chunks.append(
            DocumentChunk(
                knowledge_base_id=document.knowledge_base_id,
                owner_id=owner_id,
                chunk_index=piece.index,
                text=piece.text,
                char_count=piece.char_count,
                embedding=vector,
                embedding_model=provider.model_name,
                embedding_dimension=provider.dimension,
            )
        )


@celery_app.task(name="process_document")
def process_document_task(document_id: str) -> None:
    asyncio.run(_process_document(document_id))
