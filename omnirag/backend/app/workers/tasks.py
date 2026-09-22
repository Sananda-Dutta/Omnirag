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
      -> COMPLETED (extraction succeeded AND chunking+embedding+vector-indexing
         succeeded — "completed" means the document is actually searchable
         via the vector store, not merely that text was recovered from the
         file)
      -> FAILED (extraction, chunking, embedding, or vector indexing raised;
         error_message set to something a user can act on where possible,
         e.g. "PDF is password-protected")

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

Phase 12 (`process_url_document_task`): a separate task for URL-sourced
documents rather than overloading `process_document_task` with a branch.
The two have genuinely different failure domains — file extraction
(corrupt/encrypted files) vs network fetch (SSRF guarding, unreachable
hosts, unsupported content types) — so keeping them as separate tasks
keeps each one's except-block honest about what it's actually handling,
same reasoning as web_fetcher.py's URLFetchError being distinct from
ExtractionError. Both converge on the same `_chunk_and_embed` — the
part of the pipeline that doesn't care where the text came from.
"""

import asyncio
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.logging import get_logger
from app.embeddings.factory import get_embedding_provider
from app.ingestion.chunking import chunk_pages, chunk_text
from app.ingestion.extractors import ExtractionError, extract_text
from app.ingestion.storage import get_storage_backend
from app.ingestion.web_fetcher import URLFetchError, fetch_and_extract
from app.models.document import Document, DocumentStatus
from app.models.document_chunk import DocumentChunk
from app.retrieval.factory import get_vector_store
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

        await _chunk_and_embed(db, document, pages=result.pages)

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


async def _process_url_document(document_id: str) -> None:
    """Same isolation pattern as `_process_document`: dedicated engine per
    task invocation, disposed in `finally`."""
    engine = create_async_engine(settings.DATABASE_URL, pool_size=2, max_overflow=0)
    session_local = async_sessionmaker(bind=engine, expire_on_commit=False)

    try:
        async with session_local() as db:
            await _run_url(db, UUID(document_id))
    finally:
        await engine.dispose()


async def _run_url(db: AsyncSession, document_id: UUID) -> None:
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
        page = await fetch_and_extract(document.source_url)

        document.extracted_text = page.text
        document.page_count = None
        document.char_count = len(page.text)
        if page.title:
            document.filename = page.title[:255]

        # URL-sourced text has no per-page structure (unlike PDFs), so this
        # goes through the same flat chunk_text path as DOCX/TXT uploads —
        # pages=None, matching _run's behavior for those file types.
        await _chunk_and_embed(db, document, pages=None)

        document.status = DocumentStatus.COMPLETED
        document.error_message = None

    except URLFetchError as exc:
        document.status = DocumentStatus.FAILED
        document.error_message = str(exc)
        logger.info(
            "url_document_fetch_failed",
            extra={"context": {"document_id": str(document_id), "reason": str(exc)}},
        )

    except Exception as exc:  # noqa: BLE001 — intentional catch-all, see module docstring
        document.status = DocumentStatus.FAILED
        document.error_message = "An unexpected error occurred while processing this URL."
        logger.error(
            "url_document_processing_unexpected_error",
            extra={"context": {"document_id": str(document_id)}},
            exc_info=exc,
        )

    await db.commit()


async def _chunk_and_embed(
    db: AsyncSession, document: Document, pages: list[str] | None = None
) -> None:
    """Splits document.extracted_text into chunks, embeds each one,
    (re)persists them as DocumentChunk rows, and indexes them into the
    vector store. Any exception here propagates to the caller's except
    block and marks the document FAILED — a document whose text extracted
    fine but couldn't be chunked/embedded/indexed is not actually usable
    for retrieval, so COMPLETED must not be reported for it.

    `pages`: per-page text from the extractor (PDF only — None for
    DOCX/TXT/MD and URL-sourced documents, which have no real page
    concept). When present, chunking runs per-page (chunk_pages) so every
    chunk carries an accurate page_number for citations; otherwise falls
    back to the original flat chunk_text over the whole document."""
    if pages is not None:
        pieces = chunk_pages(pages, chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    else:
        pieces = chunk_text(
            document.extracted_text or "",
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

    # Idempotent reprocessing: clear existing chunks before inserting new
    # ones (cascade="all, delete-orphan" on Document.chunks issues the
    # deletes on flush/commit). Safe to run this task more than once for the
    # same document without accumulating duplicates. Qdrant upserts below
    # are separately idempotent too (same point IDs overwrite).
    document.chunks.clear()

    if not pieces:
        # Legitimate outcome (e.g. a file that extracted to only whitespace)
        # — zero chunks, not an error. The document is still COMPLETED;
        # there's simply nothing to retrieve from it.
        return

    provider = get_embedding_provider()
    vectors = await provider.embed_texts([piece.text for piece in pieces])

    owner_id = document.knowledge_base.owner_id
    chunk_ids = [uuid4() for _ in pieces]

    for piece, vector, chunk_id in zip(pieces, vectors, chunk_ids):
        document.chunks.append(
            DocumentChunk(
                id=chunk_id,  # set explicitly (not left to the column default)
                # so it's known now, before flush, for the Qdrant upsert below.
                knowledge_base_id=document.knowledge_base_id,
                owner_id=owner_id,
                chunk_index=piece.index,
                text=piece.text,
                char_count=piece.char_count,
                page_number=piece.page_number,
                embedding=vector,
                embedding_model=provider.model_name,
                embedding_dimension=provider.dimension,
            )
        )

    vector_store = get_vector_store()
    await vector_store.ensure_collection(dimension=provider.dimension)
    await vector_store.upsert_chunks(
        chunk_ids=chunk_ids,
        vectors=vectors,
        owner_id=owner_id,
        knowledge_base_id=document.knowledge_base_id,
        document_id=document.id,
    )


@celery_app.task(name="process_document")
def process_document_task(document_id: str) -> None:
    asyncio.run(_process_document(document_id))


@celery_app.task(name="process_url_document")
def process_url_document_task(document_id: str) -> None:
    asyncio.run(_process_url_document(document_id))