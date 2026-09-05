"""
Keyword search tests — against the real Postgres test database, exercising
the actual generated tsvector column and GIN index, not a mock.
"""

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentStatus
from app.models.document_chunk import DocumentChunk
from app.models.knowledge_base import KnowledgeBase
from app.models.user import User
from app.retrieval.keyword_search import keyword_search


async def _make_chunk(
    db: AsyncSession, owner: User, kb: KnowledgeBase, doc: Document, text: str, index: int = 0
) -> DocumentChunk:
    chunk = DocumentChunk(
        document_id=doc.id,
        knowledge_base_id=kb.id,
        owner_id=owner.id,
        chunk_index=index,
        text=text,
        char_count=len(text),
        embedding=[0.0] * 8,
        embedding_model="test",
        embedding_dimension=8,
    )
    db.add(chunk)
    await db.commit()
    await db.refresh(chunk)
    return chunk


async def _setup_owner_kb_doc(db: AsyncSession, email: str) -> tuple[User, KnowledgeBase, Document]:
    user = User(email=email, hashed_password="x")
    db.add(user)
    await db.commit()
    await db.refresh(user)

    kb = KnowledgeBase(name="kb", owner_id=user.id)
    db.add(kb)
    await db.commit()
    await db.refresh(kb)

    doc = Document(
        knowledge_base_id=kb.id,
        filename="f.txt",
        content_type="text/plain",
        file_size_bytes=10,
        storage_path="irrelevant",
        status=DocumentStatus.COMPLETED,
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    return user, kb, doc


@pytest.mark.asyncio
async def test_keyword_search_finds_exact_term_match(db_session: AsyncSession):
    user, kb, doc = await _setup_owner_kb_doc(db_session, "kwsearch1@example.com")
    matching = await _make_chunk(db_session, user, kb, doc, "The mitochondria is the powerhouse of the cell.")
    other = await _make_chunk(db_session, user, kb, doc, "Photosynthesis converts light into chemical energy.", index=1)

    results = await keyword_search(db_session, user.id, query="mitochondria", knowledge_base_id=None, top_k=5)

    assert matching.id in results
    assert other.id not in results


@pytest.mark.asyncio
async def test_keyword_search_matches_across_word_stems(db_session: AsyncSession):
    # Postgres FTS stems by default (to_tsvector('english', ...)) — a query
    # for "run" should match a chunk containing "running", not just an
    # exact substring. Asserts the actual behavior rather than just noting
    # it exists in a comment.
    user, kb, doc = await _setup_owner_kb_doc(db_session, "kwstem@example.com")
    chunk = await _make_chunk(db_session, user, kb, doc, "The model is running inference on new data.")

    results = await keyword_search(db_session, user.id, query="run", knowledge_base_id=None, top_k=5)
    assert chunk.id in results


@pytest.mark.asyncio
async def test_keyword_search_isolates_by_owner(db_session: AsyncSession):
    user_a, kb_a, doc_a = await _setup_owner_kb_doc(db_session, "kwowner@example.com")
    user_b, _, _ = await _setup_owner_kb_doc(db_session, "kwstranger@example.com")
    chunk = await _make_chunk(db_session, user_a, kb_a, doc_a, "Confidential quarterly revenue figures.")

    results_owner = await keyword_search(db_session, user_a.id, query="quarterly revenue", knowledge_base_id=None, top_k=5)
    results_stranger = await keyword_search(db_session, user_b.id, query="quarterly revenue", knowledge_base_id=None, top_k=5)

    assert chunk.id in results_owner
    assert results_stranger == []


@pytest.mark.asyncio
async def test_keyword_search_can_be_scoped_to_one_knowledge_base(db_session: AsyncSession):
    user = User(email="kwscoped@example.com", hashed_password="x")
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    kb1 = KnowledgeBase(name="kb1", owner_id=user.id)
    kb2 = KnowledgeBase(name="kb2", owner_id=user.id)
    db_session.add_all([kb1, kb2])
    await db_session.commit()
    await db_session.refresh(kb1)
    await db_session.refresh(kb2)

    doc1 = Document(knowledge_base_id=kb1.id, filename="a.txt", content_type="text/plain", file_size_bytes=1, storage_path="x", status=DocumentStatus.COMPLETED)
    doc2 = Document(knowledge_base_id=kb2.id, filename="b.txt", content_type="text/plain", file_size_bytes=1, storage_path="x", status=DocumentStatus.COMPLETED)
    db_session.add_all([doc1, doc2])
    await db_session.commit()
    await db_session.refresh(doc1)
    await db_session.refresh(doc2)

    chunk1 = await _make_chunk(db_session, user, kb1, doc1, "backpropagation neural network training")
    chunk2 = await _make_chunk(db_session, user, kb2, doc2, "backpropagation is not covered in this cookbook")

    results = await keyword_search(db_session, user.id, query="backpropagation", knowledge_base_id=kb1.id, top_k=5)
    assert results == [chunk1.id]


@pytest.mark.asyncio
async def test_keyword_search_returns_nothing_for_no_match(db_session: AsyncSession):
    user, kb, doc = await _setup_owner_kb_doc(db_session, "kwnomatch@example.com")
    await _make_chunk(db_session, user, kb, doc, "This document is entirely about gardening.")

    results = await keyword_search(db_session, user.id, query="quantum entanglement", knowledge_base_id=None, top_k=5)
    assert results == []
