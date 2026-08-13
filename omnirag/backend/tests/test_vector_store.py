"""
QdrantVectorStore tests.

Run against a genuinely running Qdrant server (see the project README for
how it's started; docker-compose runs it as the `qdrant` service). Each
test uses its own uniquely-named collection so tests can't interfere with
each other or with the application's real `document_chunks` collection, and
cleans up after itself.
"""

import uuid

import pytest

from app.retrieval.qdrant_store import DimensionMismatchError, QdrantVectorStore

QDRANT_URL = "http://localhost:6333"


@pytest.fixture
async def store():
    collection_name = f"test_{uuid.uuid4().hex}"
    instance = QdrantVectorStore(url=QDRANT_URL, collection_name=collection_name)
    yield instance
    await instance._client.delete_collection(collection_name)


@pytest.mark.asyncio
async def test_ensure_collection_is_idempotent(store):
    await store.ensure_collection(dimension=8)
    await store.ensure_collection(dimension=8)  # should not raise the second time


@pytest.mark.asyncio
async def test_ensure_collection_rejects_dimension_change(store):
    await store.ensure_collection(dimension=8)
    with pytest.raises(DimensionMismatchError):
        await store.ensure_collection(dimension=16)


@pytest.mark.asyncio
async def test_search_returns_nearest_vector_first(store):
    await store.ensure_collection(dimension=3)
    owner_id, kb_id, doc_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    near_id, far_id = uuid.uuid4(), uuid.uuid4()

    await store.upsert_chunks(
        chunk_ids=[near_id, far_id],
        vectors=[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        owner_id=owner_id,
        knowledge_base_id=kb_id,
        document_id=doc_id,
    )

    results = await store.search(query_vector=[1.0, 0.0, 0.0], owner_id=owner_id, top_k=2)
    assert results[0].chunk_id == near_id
    assert results[0].score > results[1].score


@pytest.mark.asyncio
async def test_search_isolates_by_owner(store):
    await store.ensure_collection(dimension=3)
    owner_a, owner_b = uuid.uuid4(), uuid.uuid4()
    kb_id, doc_id = uuid.uuid4(), uuid.uuid4()
    chunk_id = uuid.uuid4()

    await store.upsert_chunks(
        chunk_ids=[chunk_id],
        vectors=[[1.0, 0.0, 0.0]],
        owner_id=owner_a,
        knowledge_base_id=kb_id,
        document_id=doc_id,
    )

    # owner_a can find it
    results_a = await store.search(query_vector=[1.0, 0.0, 0.0], owner_id=owner_a, top_k=5)
    assert len(results_a) == 1

    # owner_b — a completely different user — cannot, even with an identical query
    results_b = await store.search(query_vector=[1.0, 0.0, 0.0], owner_id=owner_b, top_k=5)
    assert results_b == []


@pytest.mark.asyncio
async def test_search_can_narrow_to_one_knowledge_base(store):
    await store.ensure_collection(dimension=3)
    owner_id = uuid.uuid4()
    kb_1, kb_2 = uuid.uuid4(), uuid.uuid4()
    doc_1, doc_2 = uuid.uuid4(), uuid.uuid4()
    chunk_1, chunk_2 = uuid.uuid4(), uuid.uuid4()

    await store.upsert_chunks(
        chunk_ids=[chunk_1], vectors=[[1.0, 0.0, 0.0]],
        owner_id=owner_id, knowledge_base_id=kb_1, document_id=doc_1,
    )
    await store.upsert_chunks(
        chunk_ids=[chunk_2], vectors=[[1.0, 0.0, 0.0]],
        owner_id=owner_id, knowledge_base_id=kb_2, document_id=doc_2,
    )

    # No KB filter: both are visible to their owner
    all_results = await store.search(query_vector=[1.0, 0.0, 0.0], owner_id=owner_id, top_k=5)
    assert {r.chunk_id for r in all_results} == {chunk_1, chunk_2}

    # Narrowed to kb_1: only chunk_1
    kb1_results = await store.search(
        query_vector=[1.0, 0.0, 0.0], owner_id=owner_id, knowledge_base_id=kb_1, top_k=5
    )
    assert {r.chunk_id for r in kb1_results} == {chunk_1}


@pytest.mark.asyncio
async def test_delete_by_document_removes_only_that_documents_points(store):
    await store.ensure_collection(dimension=3)
    owner_id, kb_id = uuid.uuid4(), uuid.uuid4()
    doc_1, doc_2 = uuid.uuid4(), uuid.uuid4()
    chunk_1, chunk_2 = uuid.uuid4(), uuid.uuid4()

    await store.upsert_chunks(
        chunk_ids=[chunk_1], vectors=[[1.0, 0.0, 0.0]],
        owner_id=owner_id, knowledge_base_id=kb_id, document_id=doc_1,
    )
    await store.upsert_chunks(
        chunk_ids=[chunk_2], vectors=[[0.0, 1.0, 0.0]],
        owner_id=owner_id, knowledge_base_id=kb_id, document_id=doc_2,
    )

    await store.delete_by_document(document_id=doc_1, owner_id=owner_id)

    remaining = await store.search(query_vector=[1.0, 0.0, 0.0], owner_id=owner_id, top_k=5)
    assert {r.chunk_id for r in remaining} == {chunk_2}


@pytest.mark.asyncio
async def test_upsert_is_idempotent_on_chunk_id(store):
    await store.ensure_collection(dimension=3)
    owner_id, kb_id, doc_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    chunk_id = uuid.uuid4()

    await store.upsert_chunks(
        chunk_ids=[chunk_id], vectors=[[1.0, 0.0, 0.0]],
        owner_id=owner_id, knowledge_base_id=kb_id, document_id=doc_id,
    )
    # Re-upsert the same chunk_id with a different vector — simulates
    # reprocessing a document after a chunking/embedding change.
    await store.upsert_chunks(
        chunk_ids=[chunk_id], vectors=[[0.0, 1.0, 0.0]],
        owner_id=owner_id, knowledge_base_id=kb_id, document_id=doc_id,
    )

    results = await store.search(query_vector=[0.0, 1.0, 0.0], owner_id=owner_id, top_k=5)
    assert len(results) == 1  # overwritten, not duplicated
    assert results[0].score > 0.99  # matches the second (overwriting) vector


@pytest.mark.asyncio
async def test_upsert_with_empty_list_is_a_noop(store):
    await store.ensure_collection(dimension=3)
    await store.upsert_chunks(
        chunk_ids=[], vectors=[], owner_id=uuid.uuid4(), knowledge_base_id=uuid.uuid4(), document_id=uuid.uuid4()
    )  # should not raise
