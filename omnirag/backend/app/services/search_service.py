"""
Search service.

Two-step retrieval, deliberately: the vector store returns chunk IDs and
scores (that's what it's good at — approximate nearest neighbor search),
then a Postgres lookup fetches the authoritative chunk text and document
filename for those IDs. The vector store is not treated as a second source
of truth for chunk content — Postgres is, always. This also means a chunk's
text is never stale relative to what's in the vector index by construction:
there's only one place chunk text lives.

Result ordering: Qdrant returns results in relevance order, but a Postgres
`WHERE id IN (...)` query does NOT preserve that order — SQL makes no such
guarantee. Results are explicitly re-sorted to match the vector store's
order after the fetch, or the ranking Qdrant computed would be silently
discarded.
"""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.embeddings.factory import get_embedding_provider
from app.models.document_chunk import DocumentChunk
from app.retrieval.factory import get_vector_store
from app.schemas.search import SearchResultItem
from app.services.knowledge_base_service import get_knowledge_base


async def search(
    db: AsyncSession,
    owner_id: uuid.UUID,
    query: str,
    knowledge_base_id: uuid.UUID | None,
    top_k: int | None,
) -> list[SearchResultItem]:
    if knowledge_base_id is not None:
        await get_knowledge_base(db, owner_id, knowledge_base_id)  # ownership check; 404 if not found/not owned

    provider = get_embedding_provider()
    [query_vector] = await provider.embed_texts([query])

    vector_store = get_vector_store()
    matches = await vector_store.search(
        query_vector=query_vector,
        owner_id=owner_id,
        knowledge_base_id=knowledge_base_id,
        top_k=top_k or settings.DEFAULT_SEARCH_TOP_K,
    )
    if not matches:
        return []

    chunk_ids = [m.chunk_id for m in matches]
    result = await db.execute(
        select(DocumentChunk)
        .options(selectinload(DocumentChunk.document))
        .where(DocumentChunk.id.in_(chunk_ids))
    )
    chunks_by_id = {chunk.id: chunk for chunk in result.scalars().all()}

    items = []
    for match in matches:
        chunk = chunks_by_id.get(match.chunk_id)
        if chunk is None:
            # The vector store and Postgres briefly disagreeing is possible
            # (e.g. a delete that removed the Postgres row but hasn't yet —
            # or ever, if it failed — removed the Qdrant point). Skip rather
            # than error: a slightly-short result list beats a 500 for the
            # whole search because of one stale vector.
            continue
        items.append(
            SearchResultItem(
                chunk_id=chunk.id,
                score=match.score,
                text=chunk.text,
                chunk_index=chunk.chunk_index,
                document_id=chunk.document_id,
                document_filename=chunk.document.filename,
            )
        )
    return items
