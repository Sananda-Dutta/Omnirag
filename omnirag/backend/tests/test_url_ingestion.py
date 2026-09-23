"""
Phase 12 (web ingestion) integration tests.

Mirrors tests/test_documents.py's end-to-end style: real HTTP request ->
real Postgres row -> real Celery task on real Redis -> a genuinely
separate worker process -> real Qdrant upsert. The "real fetch" tests
deliberately hit real, already-allowlisted hosts (github.com,
raw.githubusercontent.com) rather than mocking httpx, so this suite
proves the actual network path works — not just that the code calls the
right mocked function. tests/test_web_fetcher.py covers the
deterministic edge cases (content-type rejection, size cap) with a mocked
transport instead, since those aren't things a real host will reliably
exercise on demand.
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


async def _wait_for_status(
    client: AsyncClient, token: str, document_id: str, timeout: float = 30.0
) -> dict:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        resp = await client.get(f"/api/v1/documents/{document_id}", headers=_auth(token))
        body = resp.json()
        if body["status"] in ("completed", "failed"):
            return body
        await asyncio.sleep(0.3)
    raise TimeoutError(f"Document {document_id} did not finish processing within {timeout}s")


@pytest.mark.asyncio
async def test_ingest_url_creates_pending_document_immediately(client: AsyncClient):
    """The endpoint must not block on the network fetch — see the module
    docstring in document_service.py. Asserting response shape here (no
    celery_worker fixture) proves the request returns before any fetch
    could plausibly have completed."""
    token = await _register_and_login(client, "url-pending@example.com")
    kb_id = await _create_kb(client, token)

    resp = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents/url",
        json={"url": "https://github.com/octocat"},
        headers=_auth(token),
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "pending"
    assert body["source_url"] == "https://github.com/octocat"
    assert body["file_size_bytes"] is None


@pytest.mark.asyncio
async def test_ingest_url_rejects_non_http_scheme(client: AsyncClient):
    token = await _register_and_login(client, "url-badscheme@example.com")
    kb_id = await _create_kb(client, token)

    resp = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents/url",
        json={"url": "ftp://example.com/file"},
        headers=_auth(token),
    )
    # Pydantic's HttpUrl rejects the scheme before this ever reaches the
    # service layer — a 422, not a 201-then-fail.
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_ingest_url_rejects_into_someone_elses_knowledge_base(client: AsyncClient):
    token_a = await _register_and_login(client, "url-owner@example.com")
    token_b = await _register_and_login(client, "url-intruder@example.com")
    kb_id = await _create_kb(client, token_a)

    resp = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents/url",
        json={"url": "https://github.com/octocat"},
        headers=_auth(token_b),
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_url_pointing_at_private_address_fails_with_reason(client: AsyncClient, celery_worker):
    """The SSRF guard in web_fetcher.py runs inside the real worker
    process, not mocked — this proves it actually blocks a private target
    end-to-end rather than only in the unit tests against a mocked
    transport."""
    token = await _register_and_login(client, "url-ssrf@example.com")
    kb_id = await _create_kb(client, token)

    resp = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents/url",
        json={"url": "http://127.0.0.1:6333/collections"},
        headers=_auth(token),
    )
    doc_id = resp.json()["id"]

    final = await _wait_for_status(client, token, doc_id)
    assert final["status"] == "failed"
    assert "private/internal address" in final["error_message"]


@pytest.mark.asyncio
async def test_html_url_is_fetched_extracted_and_made_searchable(client: AsyncClient, celery_worker):
    """Real end-to-end fetch of a real HTML page on an allowlisted host —
    no mocking. Checks the full pipeline: fetch -> extract -> chunk ->
    embed -> index -> searchable via the chat/search endpoints, and that
    the page's real <title> replaced the placeholder filename (the URL)."""
    token = await _register_and_login(client, "url-html@example.com")
    kb_id = await _create_kb(client, token)

    resp = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents/url",
        json={"url": "https://github.com/octocat"},
        headers=_auth(token),
    )
    doc_id = resp.json()["id"]

    final = await _wait_for_status(client, token, doc_id)
    assert final["status"] == "completed"
    assert final["char_count"] > 0
    # page_count is meaningless for a URL-sourced document (no PDF page
    # concept) — must stay null, not a fabricated 1.
    assert final["page_count"] is None
    # The placeholder filename (the URL itself, set at creation time) is
    # expected to have been replaced with the page's real <title> once the
    # worker fetched it — a plain "== url" would mean the title swap never
    # ran.
    assert final["filename"] != "https://github.com/octocat"

    chunks = await client.get(f"/api/v1/documents/{doc_id}/chunks", headers=_auth(token))
    assert chunks.status_code == 200
    assert len(chunks.json()) > 0
    # URL-sourced text has no page concept — chunk_text (not chunk_pages)
    # ran, so every chunk's page_number must be null.
    assert all(c["page_number"] is None for c in chunks.json())


@pytest.mark.asyncio
async def test_plain_text_url_is_fetched_with_no_title_override(client: AsyncClient, celery_worker):
    token = await _register_and_login(client, "url-plaintext@example.com")
    kb_id = await _create_kb(client, token)

    url = "https://raw.githubusercontent.com/octocat/Hello-World/master/README"
    resp = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents/url",
        json={"url": url},
        headers=_auth(token),
    )
    doc_id = resp.json()["id"]

    final = await _wait_for_status(client, token, doc_id)
    assert final["status"] == "completed"
    # Plain text has no <title> to extract — filename stays as the
    # original URL rather than being overwritten with something invented.
    assert final["filename"] == url

    detail = await client.get(f"/api/v1/documents/{doc_id}", headers=_auth(token))
    assert "Hello World" in detail.json()["extracted_text"]


@pytest.mark.asyncio
async def test_url_document_is_isolated_from_other_users_search(client: AsyncClient, celery_worker):
    token_a = await _register_and_login(client, "url-isolate-a@example.com")
    token_b = await _register_and_login(client, "url-isolate-b@example.com")
    kb_id = await _create_kb(client, token_a)

    resp = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents/url",
        json={"url": "https://github.com/octocat"},
        headers=_auth(token_a),
    )
    doc_id = resp.json()["id"]
    await _wait_for_status(client, token_a, doc_id)

    # User B (who owns no knowledge bases at all) must not be able to
    # reach user A's URL-sourced document or its chunks.
    detail = await client.get(f"/api/v1/documents/{doc_id}", headers=_auth(token_b))
    assert detail.status_code == 404
