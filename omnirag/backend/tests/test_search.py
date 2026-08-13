"""
End-to-end search tests.

Full real path: upload -> real Celery worker chunks/embeds/indexes into a
real Qdrant -> POST /search embeds the query with the same provider and
queries Qdrant -> results are joined back to Postgres for display. Nothing
mocked.
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
async def test_search_finds_relevant_chunk(client: AsyncClient, celery_worker):
    token = await _register_and_login(client, "search-basic@example.com")
    kb_id = await _create_kb(client, token)

    await _upload_and_wait(
        client, token, kb_id, "ml.txt",
        b"Gradient descent is an optimization algorithm used to minimize a loss function.",
    )
    await _upload_and_wait(
        client, token, kb_id, "cooking.txt",
        b"Preheat the oven to 350 degrees and bake the bread for forty minutes.",
    )

    response = await client.post(
        "/api/v1/search",
        json={"query": "how does gradient descent minimize loss"},
        headers=_auth(token),
    )
    assert response.status_code == 200
    results = response.json()

    assert len(results) > 0
    assert "Gradient descent" in results[0]["text"]
    assert results[0]["document_filename"] == "ml.txt"


@pytest.mark.asyncio
async def test_search_requires_auth(client: AsyncClient):
    response = await client.post("/api/v1/search", json={"query": "anything"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_search_does_not_leak_across_users(client: AsyncClient, celery_worker):
    token_a = await _register_and_login(client, "search-owner@example.com")
    token_b = await _register_and_login(client, "search-stranger@example.com")
    kb_id = await _create_kb(client, token_a)

    await _upload_and_wait(
        client, token_a, kb_id, "secret.txt",
        b"The quarterly revenue figures are confidential and only for internal review.",
    )

    response = await client.post(
        "/api/v1/search",
        json={"query": "quarterly revenue figures"},
        headers=_auth(token_b),
    )
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_search_can_be_scoped_to_one_knowledge_base(client: AsyncClient, celery_worker):
    token = await _register_and_login(client, "search-scoped@example.com")
    kb_ml = await _create_kb(client, token, name="ML notes")
    kb_cooking = await _create_kb(client, token, name="Recipes")

    await _upload_and_wait(
        client, token, kb_ml, "ml.txt",
        b"Backpropagation computes gradients for training neural networks.",
    )
    await _upload_and_wait(
        client, token, kb_cooking, "recipe.txt",
        b"Backpropagation is not a cooking term, this document is about bread.",
    )

    response = await client.post(
        "/api/v1/search",
        json={"query": "backpropagation neural networks", "knowledge_base_id": kb_ml},
        headers=_auth(token),
    )
    results = response.json()
    assert len(results) == 1
    assert results[0]["document_filename"] == "ml.txt"


@pytest.mark.asyncio
async def test_search_rejects_nonexistent_or_foreign_knowledge_base(client: AsyncClient):
    token_a = await _register_and_login(client, "search-kbowner@example.com")
    token_b = await _register_and_login(client, "search-kbstranger@example.com")
    kb_id = await _create_kb(client, token_a)

    response = await client.post(
        "/api/v1/search",
        json={"query": "anything", "knowledge_base_id": kb_id},
        headers=_auth(token_b),
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_deleting_document_removes_it_from_search(client: AsyncClient, celery_worker):
    token = await _register_and_login(client, "search-delete@example.com")
    kb_id = await _create_kb(client, token)

    doc_id = await _upload_and_wait(
        client, token, kb_id, "temp.txt",
        b"Transformers use self-attention to process sequences in parallel.",
    )

    before = await client.post(
        "/api/v1/search", json={"query": "self-attention transformers"}, headers=_auth(token)
    )
    assert len(before.json()) > 0

    delete_resp = await client.delete(f"/api/v1/documents/{doc_id}", headers=_auth(token))
    assert delete_resp.status_code == 204

    after = await client.post(
        "/api/v1/search", json={"query": "self-attention transformers"}, headers=_auth(token)
    )
    assert after.json() == []
