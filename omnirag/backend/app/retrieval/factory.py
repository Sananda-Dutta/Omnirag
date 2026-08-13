"""
VectorStore factory. Same role as app/embeddings/factory.py: the only place
that knows about concrete implementations, so switching vector databases
is a config change, not a code change.

Currently only Qdrant is implemented. There's no `VECTOR_DB_PROVIDER`
setting yet (unlike EMBEDDING_PROVIDER) because there's only one real
option right now — adding a pgvector implementation later is exactly when
that setting would earn its place, not before.
"""

from functools import lru_cache

from app.core.config import settings
from app.retrieval.qdrant_store import QdrantVectorStore
from app.retrieval.vector_store import VectorStore


@lru_cache
def get_vector_store() -> VectorStore:
    return QdrantVectorStore(
        url=settings.VECTOR_DB_URL, collection_name=settings.VECTOR_DB_COLLECTION
    )
