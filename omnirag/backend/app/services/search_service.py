"""
Search service — hybrid retrieval pipeline (Phase 8).

    query -> dense search (Qdrant)          --\
          -> keyword search (Postgres FTS)   ---> RRF fusion -> fetch chunk
                                                    text from Postgres
                                                    -> lexical rerank -> top_k

Both ENABLE_KEYWORD_SEARCH and ENABLE_RERANKING can be turned off
independently (falling back toward Phase 6's dense-only behavior), mainly
so Phase 15 (RAG evaluation) can A/B compare configurations against the
same test question set rather than only ever seeing the fully-hybrid result.

Each retrieval method over-fetches (HYBRID_CANDIDATE_MULTIPLIER * top_k)
before fusion/reranking narrows back down — fusion and reranking need a
wider candidate pool to actually change the final ranking, not just
re-sort a list that's already been cut down to its final size.

`score` on the returned SearchResultItem reflects whichever stage actually
determined the final order: the lexical reranker's score if reranking ran,
otherwise the RRF fusion score if hybrid search ran, otherwise Qdrant's raw
cosine similarity in dense-only mode. It is NOT always a cosine similarity
— documented on the schema field itself, not just here.

Two-step retrieval (IDs first, then a Postgres fetch), and chunk-text
authority living only in Postgres, are unchanged from Phase 6 — see the
reasoning that used to live in this docstring, now still true, just
extended with two more retrieval methods feeding the same fetch step.
"""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.embeddings.factory import get_embedding_provider
from app.models.document_chunk import DocumentChunk
from app.retrieval.factory import get_vector_store
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.keyword_search import keyword_search
from app.retrieval.reranker import LocalLexicalReranker, RerankCandidate
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

    final_top_k = top_k or settings.DEFAULT_SEARCH_TOP_K
    candidate_k = final_top_k * settings.HYBRID_CANDIDATE_MULTIPLIER

    provider = get_embedding_provider()
    [query_vector] = await provider.embed_texts([query])

    vector_store = get_vector_store()
    dense_matches = await vector_store.search(
        query_vector=query_vector,
        owner_id=owner_id,
        knowledge_base_id=knowledge_base_id,
        top_k=candidate_k,
    )
    dense_ids = [m.chunk_id for m in dense_matches]
    dense_scores = {m.chunk_id: m.score for m in dense_matches}

    if settings.ENABLE_KEYWORD_SEARCH:
        keyword_ids = await keyword_search(
            db, owner_id, query=query, knowledge_base_id=knowledge_base_id, top_k=candidate_k
        )
        fused = reciprocal_rank_fusion([dense_ids, keyword_ids], k=settings.RRF_K)
        candidate_ids_ranked = [chunk_id for chunk_id, _ in fused]
        candidate_scores: dict[uuid.UUID, float] = dict(fused)
    else:
        candidate_ids_ranked = dense_ids
        candidate_scores = dense_scores

    if not candidate_ids_ranked:
        return []

    result = await db.execute(
        select(DocumentChunk)
        .options(selectinload(DocumentChunk.document))
        .where(DocumentChunk.id.in_(candidate_ids_ranked))
    )
    chunks_by_id = {chunk.id: chunk for chunk in result.scalars().all()}

    # Preserve fused/dense rank order, dropping any ID the vector store or
    # keyword search returned but that no longer exists in Postgres (e.g. a
    # delete that removed the row but — for whatever reason — not yet, or
    # not successfully, the Qdrant point). A slightly-short result list
    # beats a 500 for the whole search because of one stale reference.
    ordered_ids = [cid for cid in candidate_ids_ranked if cid in chunks_by_id]
    if not ordered_ids:
        return []

    if settings.ENABLE_RERANKING:
        candidates = [
            RerankCandidate(chunk_id=cid, text=chunks_by_id[cid].text) for cid in ordered_ids
        ]
        reranked = LocalLexicalReranker().rerank(query, candidates)
        final_ids_scores = reranked[:final_top_k]
    else:
        final_ids_scores = [
            (cid, candidate_scores.get(cid, 0.0)) for cid in ordered_ids[:final_top_k]
        ]

    items = []
    for chunk_id, score in final_ids_scores:
        chunk = chunks_by_id[chunk_id]
        items.append(
            SearchResultItem(
                chunk_id=chunk.id,
                score=score,
                text=chunk.text,
                chunk_index=chunk.chunk_index,
                document_id=chunk.document_id,
                document_filename=chunk.document.filename,
            )
        )
    return items
