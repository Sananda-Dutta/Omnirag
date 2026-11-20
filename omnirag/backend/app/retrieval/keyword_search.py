"""
Keyword (full-text) search via Postgres.

Queries the `text_search_vector` generated column (see
app/models/document_chunk.py for why it's a generated column and why
Postgres FTS instead of a separate search engine). Isolation is enforced
the same way as everywhere else in this project: `owner_id` is a required
filter, never optional, on every query — see app/retrieval/vector_store.py
for why that's a hard requirement rather than a convention.

`plainto_tsquery` (not `to_tsquery`) is used deliberately: it takes plain
user text and handles tokenization/stemming itself, safely ignoring
special tsquery operator syntax a user might type. `to_tsquery` expects the
caller to have already built valid tsquery syntax (`cat & dog`, etc.) —
using it directly on raw user input is both a poor UX (a query containing
an unescaped `&` or `:` breaks) and, depending on how it's constructed, a
potential injection-adjacent footgun.
"""

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document_chunk import DocumentChunk


async def keyword_search(
    db: AsyncSession,
    owner_id: uuid.UUID,
    query: str,
    knowledge_base_id: uuid.UUID | None,
    top_k: int,
) -> list[uuid.UUID]:
    """Returns chunk IDs ranked by Postgres's ts_rank_cd, best first."""
    tsquery = func.plainto_tsquery("english", query)
    rank = func.ts_rank_cd(DocumentChunk.text_search_vector, tsquery)

    stmt = (
        select(DocumentChunk.id)
        .where(DocumentChunk.owner_id == owner_id)
        .where(DocumentChunk.text_search_vector.op("@@")(tsquery))
        .order_by(rank.desc())
        .limit(top_k)
    )
    if knowledge_base_id is not None:
        stmt = stmt.where(DocumentChunk.knowledge_base_id == knowledge_base_id)

    result = await db.execute(stmt)
    return list(result.scalars().all())
