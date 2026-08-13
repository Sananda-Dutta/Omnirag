"""
Qdrant-backed VectorStore.

Why one collection for every user/knowledge-base rather than a collection
per user: Qdrant collections carry real per-collection overhead (each one
gets its own HNSW index, segment files, and background optimizer threads).
Thousands of users would mean thousands of collections — a scalability
problem, not a security feature. Isolation is enforced instead through a
mandatory payload filter on every query (`owner_id`, optionally narrowed by
`knowledge_base_id`), matching how the Postgres tables in this project
already do isolation via a `WHERE owner_id = ...` clause rather than a
separate database per user. Same pattern, same reasoning, applied to the
vector store.

Point IDs are the chunk's own UUID (as a string — Qdrant accepts UUID
strings directly as point IDs). This makes upserts naturally idempotent:
reprocessing a document re-embeds and re-upserts the same chunk IDs, which
overwrites rather than duplicates.
"""

from uuid import UUID

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.retrieval.vector_store import VectorSearchResult, VectorStore


class DimensionMismatchError(Exception):
    """Raised when the configured embedding dimension doesn't match an
    existing collection's actual vector size. Deliberately not
    auto-resolved (e.g. by silently recreating the collection) — that would
    silently delete every existing embedding. Changing embedding
    dimension/provider is a migration a human should decide to run, not
    something that happens as a side effect of the app starting up."""


class QdrantVectorStore(VectorStore):
    def __init__(self, url: str, collection_name: str):
        self._client = AsyncQdrantClient(url=url)
        self._collection = collection_name

    async def ensure_collection(self, dimension: int) -> None:
        collections = await self._client.get_collections()
        exists = any(c.name == self._collection for c in collections.collections)

        if not exists:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
            return

        info = await self._client.get_collection(self._collection)
        existing_size = info.config.params.vectors.size
        if existing_size != dimension:
            raise DimensionMismatchError(
                f"Collection {self._collection!r} was created with vector size "
                f"{existing_size}, but the configured embedding provider produces "
                f"{dimension}-dimensional vectors. This usually means "
                f"EMBEDDING_PROVIDER or EMBEDDING_DIMENSION changed after chunks "
                f"were already indexed. Existing vectors are not comparable to "
                f"the new dimension — re-embed and re-index into a new "
                f"collection (or recreate this one) rather than mixing dimensions."
            )

    async def upsert_chunks(
        self,
        *,
        chunk_ids: list[UUID],
        vectors: list[list[float]],
        owner_id: UUID,
        knowledge_base_id: UUID,
        document_id: UUID,
    ) -> None:
        if not chunk_ids:
            return

        points = [
            PointStruct(
                id=str(chunk_id),
                vector=vector,
                payload={
                    "owner_id": str(owner_id),
                    "knowledge_base_id": str(knowledge_base_id),
                    "document_id": str(document_id),
                },
            )
            for chunk_id, vector in zip(chunk_ids, vectors)
        ]
        await self._client.upsert(collection_name=self._collection, points=points)

    async def search(
        self,
        *,
        query_vector: list[float],
        owner_id: UUID,
        knowledge_base_id: UUID | None = None,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        must = [FieldCondition(key="owner_id", match=MatchValue(value=str(owner_id)))]
        if knowledge_base_id is not None:
            must.append(
                FieldCondition(
                    key="knowledge_base_id", match=MatchValue(value=str(knowledge_base_id))
                )
            )

        response = await self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            query_filter=Filter(must=must),
            limit=top_k,
        )
        return [
            VectorSearchResult(chunk_id=UUID(point.id), score=point.score)
            for point in response.points
        ]

    async def delete_by_document(self, *, document_id: UUID, owner_id: UUID) -> None:
        # owner_id included in the filter (not just document_id) so this
        # can never be used to delete another user's points even if a
        # document_id were somehow guessed/forged — belt and suspenders
        # with the ownership check that already happens before this is
        # called (see document_service.delete_document).
        await self._client.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(key="document_id", match=MatchValue(value=str(document_id))),
                        FieldCondition(key="owner_id", match=MatchValue(value=str(owner_id))),
                    ]
                )
            ),
        )
