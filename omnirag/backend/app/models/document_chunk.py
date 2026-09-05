"""
DocumentChunk table.

Denormalization note: `knowledge_base_id` and `owner_id` are duplicated here
rather than reached only via `document.knowledge_base.owner_id`. This is
deliberate, not an oversight — retrieval (Phase 7+) needs to filter chunks
by knowledge base and by owner on every single query, and the whole point of
per-user isolation is that this filter can never be forgotten or joined
around. Having the columns directly on the row that gets searched means the
filter is a plain `WHERE owner_id = :user_id` on the table actually being
queried, not a join that a future query could accidentally omit.

Where the embedding vector lives: as a plain Postgres `float8[]` array for
now, not a specialized vector type. Phase 5's job is "produce and persist
embeddings," not "build an ANN index" — that's explicitly Phase 6 (vector
database). Storing raw arrays here means Phase 6 has real vectors to
migrate into Qdrant (or convert this column to pgvector) rather than
building the vector-DB integration and the embedding generation at the same
time as one big untestable step. Similarity search against this column
would currently require a full table scan — acceptable for Phase 5's scope,
explicitly not for production, which is exactly what Phase 6 fixes.

`embedding_model` records which provider/model produced the vector. This
matters because it's a correctness issue, not bookkeeping: vectors from two
different embedding models are not comparable to each other (different
models place semantically similar text in entirely different vector
spaces), so if `EMBEDDING_PROVIDER` ever changes, chunks embedded under the
old provider must be distinguishable from — and re-embedded, not mixed
with — chunks under the new one.

`text_search_vector` (Phase 8): a Postgres-native `tsvector`, declared as a
generated column (`GENERATED ALWAYS AS ... STORED`) so it's computed and
kept in sync by Postgres itself on every insert/update — there's no
application code path that can forget to update it, unlike a
manually-maintained derived column. Backed by a GIN index, this is what
keyword/full-text search (app/retrieval/keyword_search.py) queries against.

Why Postgres full-text search instead of a dedicated BM25 library or a
separate search engine (Elasticsearch/OpenSearch): this project already
runs Postgres as the system of record, and `tsvector`/`ts_rank_cd` is a
real, production-grade keyword ranking capability built into it — adding
Elasticsearch would mean operating a second stateful service (like the
choice not to build a second sync DB stack for the Celery worker in Phase
4) for a capability Postgres already provides adequately at this project's
scale. `ts_rank_cd` isn't literally the BM25 formula, but it's the same
family of technique (term-frequency-based lexical ranking) and is what the
spec's "Keyword/BM25 search" pipeline stage is actually for: complementing
dense vector search with exact-term matching dense embeddings can miss.
"""

import uuid

from sqlalchemy import ARRAY, Computed, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class DocumentChunk(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "document_chunks"
    __table_args__ = (
        Index("ix_document_chunks_text_search_vector", "text_search_vector", postgresql_using="gin"),
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)

    embedding: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(128), nullable=False)
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False)

    text_search_vector: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', text)", persisted=True),
        nullable=True,
    )

    document: Mapped["Document"] = relationship(back_populates="chunks")

    def __repr__(self) -> str:
        return (
            f"<DocumentChunk id={self.id} document_id={self.document_id} "
            f"index={self.chunk_index}>"
        )
