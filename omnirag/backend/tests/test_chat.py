"""
End-to-end chat (RAG) tests.

Uses the real pipeline against the local-extractive LLM provider (the
default, and the only one this sandbox can actually execute without an API
key) — real upload, real Celery worker, real Qdrant search, real context
construction, real (if crude) extractive generation. Nothing mocked here;
the LLM provider mocking lives entirely in test_llm.py, scoped to the
providers that genuinely require network access this environment doesn't have.
"""

import asyncio

import pytest
from httpx import AsyncClient


async def _register_and_login(client: AsyncClient, email: str) -> str:
    await client.post(
        "/api/v1/auth/register", json={"email": email, "password": "correct-horse-battery"}
    )
    login = await client.post(
        "/api/v1/auth/login", data={"username": email, "password": "correct-horse-battery"}
    )
    return login.json()["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


async def _create_kb(client: AsyncClient, token: str, name: str = "test kb") -> str:
    resp = await client.post("/api/v1/knowledge-bases", json={"name": name}, headers=_auth(token))
    return resp.json()["id"]


async def _upload_and_wait(client: AsyncClient, token: str, kb_id: str, filename: str, content: bytes) -> str:
    files = {"file": (filename, content, "text/plain")}
    upload = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents", files=files, headers=_auth(token)
    )
    doc_id = upload.json()["id"]

    deadline = asyncio.get_event_loop().time() + 15.0
    while asyncio.get_event_loop().time() < deadline:
        resp = await client.get(f"/api/v1/documents/{doc_id}", headers=_auth(token))
        if resp.json()["status"] in ("completed", "failed"):
            assert resp.json()["status"] == "completed", resp.json()
            return doc_id
        await asyncio.sleep(0.3)
    raise TimeoutError(f"document {doc_id} did not finish processing in time")


@pytest.mark.asyncio
async def test_chat_answers_from_retrieved_context(client: AsyncClient, celery_worker):
    token = await _register_and_login(client, "chat-basic@example.com")
    kb_id = await _create_kb(client, token)

    await _upload_and_wait(
        client, token, kb_id, "ml.txt",
        b"Retrieval augmented generation combines a retriever with a language model to ground answers in real documents.",
    )

    response = await client.post(
        "/api/v1/chat",
        json={"question": "What does retrieval augmented generation combine?"},
        headers=_auth(token),
    )
    assert response.status_code == 200
    body = response.json()

    assert body["context_found"] is True
    assert body["model_used"] == "local-extractive"
    assert len(body["citations"]) > 0
    assert body["citations"][0]["document_filename"] == "ml.txt"
    assert "retriever" in body["answer"].lower() or "language model" in body["answer"].lower()


@pytest.mark.asyncio
async def test_chat_with_empty_knowledge_base_does_not_call_llm(client: AsyncClient):
    token = await _register_and_login(client, "chat-empty@example.com")
    await _create_kb(client, token)  # KB exists but has no documents

    response = await client.post(
        "/api/v1/chat", json={"question": "Anything at all?"}, headers=_auth(token)
    )
    assert response.status_code == 200
    body = response.json()

    assert body["context_found"] is False
    assert body["citations"] == []
    assert body["model_used"] == "none"
    assert "couldn't find" in body["answer"].lower()


@pytest.mark.asyncio
async def test_chat_requires_auth(client: AsyncClient):
    response = await client.post("/api/v1/chat", json={"question": "anything"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_chat_does_not_leak_across_users(client: AsyncClient, celery_worker):
    token_a = await _register_and_login(client, "chat-owner@example.com")
    token_b = await _register_and_login(client, "chat-stranger@example.com")
    kb_id = await _create_kb(client, token_a)

    await _upload_and_wait(
        client, token_a, kb_id, "secret.txt",
        b"The acquisition price was forty million dollars, confidential until the public announcement.",
    )

    response = await client.post(
        "/api/v1/chat",
        json={"question": "What was the acquisition price?"},
        headers=_auth(token_b),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["context_found"] is False
    assert body["citations"] == []


@pytest.mark.asyncio
async def test_chat_rejects_nonexistent_or_foreign_knowledge_base(client: AsyncClient):
    token_a = await _register_and_login(client, "chat-kbowner@example.com")
    token_b = await _register_and_login(client, "chat-kbstranger@example.com")
    kb_id = await _create_kb(client, token_a)

    response = await client.post(
        "/api/v1/chat",
        json={"question": "anything", "knowledge_base_id": kb_id},
        headers=_auth(token_b),
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_chat_citation_fields_are_populated(client: AsyncClient, celery_worker):
    token = await _register_and_login(client, "chat-citations@example.com")
    kb_id = await _create_kb(client, token)

    await _upload_and_wait(
        client, token, kb_id, "notes.txt",
        b"Hybrid search combines dense vector retrieval with sparse keyword matching for better recall.",
    )

    response = await client.post(
        "/api/v1/chat", json={"question": "What does hybrid search combine?"}, headers=_auth(token)
    )
    citation = response.json()["citations"][0]

    assert citation["document_filename"] == "notes.txt"
    assert citation["chunk_index"] == 0
    # Score semantics depend on which retrieval stage set them (raw cosine
    # similarity, RRF fusion score, or the lexical reranker's score — see
    # SearchResultItem.score's docstring) — only non-negative is guaranteed
    # across all of them.
    assert citation["score"] >= 0.0
    assert len(citation["text_snippet"]) > 0
