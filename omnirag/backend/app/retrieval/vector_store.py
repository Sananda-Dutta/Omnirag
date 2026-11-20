"""
VectorStore abstraction.

Same reasoning as EmbeddingProvider (app/embeddings/base.py): callers depend
on this interface, never on `qdrant_client` directly, so swapping vector
databases (Qdrant -> pgvector -> Pinecone -> ...) is a factory change, not a
rewrite scattered across services and tasks.

The isolation design choice worth calling out explicitly: `owner_id` is a
*required* parameter on `search` and `delete_by_document`, not optional. A
per-user isolation bug in a vector store is unusually dangerous — RAG
answers get generated directly from whatever a similarity search returns,
so a filtering mistake here doesn't just leak a list item, it can leak
another user's document content straight into an LLM-generated answer shown
to the wrong person. Making `owner_id` required at the type level means a
caller has to actively go out of their way to search without it, rather
than isolation being something a caller has to remember to add.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class VectorSearchResult:
    chunk_id: UUID
    score: float


class VectorStore(ABC):
    @abstractmethod
    async def ensure_collection(self, dimension: int) -> None:
        """Creates the collection if it doesn't exist. If it already exists
        with a different vector dimension, raises rather than silently
        proceeding — see qdrant_store.py's DimensionMismatchError for why
        that specific failure mode matters."""
        ...

    @abstractmethod
    async def upsert_chunks(
        self,
        *,
        chunk_ids: list[UUID],
        vectors: list[list[float]],
        owner_id: UUID,
        knowledge_base_id: UUID,
        document_id: UUID,
    ) -> None:
        ...

    @abstractmethod
    async def search(
        self,
        *,
        query_vector: list[float],
        owner_id: UUID,
        knowledge_base_id: UUID | None = None,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        """knowledge_base_id narrows to one knowledge base; omitted, search
        spans every knowledge base the owner has — owner_id itself is never
        optional (see module docstring)."""
        ...

    @abstractmethod
    async def delete_by_document(self, *, document_id: UUID, owner_id: UUID) -> None:
        ...
