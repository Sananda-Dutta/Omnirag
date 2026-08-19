"""
Basic RAG pipeline.

    question -> vector search (Phase 6) -> build grounded context
             -> LLM generation -> citations back to specific chunks

Hallucination-control decision worth calling out: if search returns zero
results, the LLM is never called at all — the pipeline returns a fixed "not
found" answer immediately. This isn't just a nice-to-have; it's the
difference between "the model was never given a chance to guess" and
"the model was told to only use context, given none, and might still try
anyway." Not calling the LLM is a stronger guarantee than prompting it not
to hallucinate.
"""

import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.llm.factory import get_llm_provider
from app.rag.prompts import build_system_prompt
from app.schemas.search import SearchResultItem
from app.services.search_service import search as run_search


@dataclass(frozen=True)
class Citation:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_filename: str
    chunk_index: int
    score: float
    text_snippet: str


@dataclass(frozen=True)
class RAGAnswer:
    answer: str
    citations: list[Citation]
    model_used: str
    context_found: bool


async def answer_question(
    db: AsyncSession,
    owner_id: uuid.UUID,
    question: str,
    knowledge_base_id: uuid.UUID | None,
    top_k: int | None = None,
) -> RAGAnswer:
    results = await run_search(
        db,
        owner_id,
        query=question,
        knowledge_base_id=knowledge_base_id,
        top_k=top_k or settings.RAG_TOP_K,
    )

    if not results:
        return RAGAnswer(
            answer="I couldn't find anything in your documents related to that question.",
            citations=[],
            model_used="none",
            context_found=False,
        )

    context, used_results = _build_context(results)

    system_prompt = build_system_prompt(context)
    provider = get_llm_provider()
    response = await provider.generate(system=system_prompt, user_message=question)

    citations = [
        Citation(
            chunk_id=r.chunk_id,
            document_id=r.document_id,
            document_filename=r.document_filename,
            chunk_index=r.chunk_index,
            score=r.score,
            text_snippet=r.text[:200],
        )
        for r in used_results
    ]

    return RAGAnswer(
        answer=response.text,
        citations=citations,
        model_used=response.model,
        context_found=True,
    )


def _build_context(results: list[SearchResultItem]) -> tuple[str, list[SearchResultItem]]:
    """Concatenates retrieved chunks into one labeled context block, capped
    at RAG_MAX_CONTEXT_CHARS. Stops adding chunks once the cap would be
    exceeded rather than truncating mid-chunk — a half-sentence of context
    is worse than one fewer whole chunk. Always includes at least the first
    (highest-scoring) chunk even if it alone exceeds the cap, so a single
    oversized chunk can't reduce the answer to "no context at all"."""
    parts: list[str] = []
    used: list[SearchResultItem] = []
    total = 0
    for r in results:
        block = f"[Source: {r.document_filename}, chunk {r.chunk_index}]\n{r.text}"
        if used and total + len(block) > settings.RAG_MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        used.append(r)
        total += len(block)
    return "\n\n---\n\n".join(parts), used
