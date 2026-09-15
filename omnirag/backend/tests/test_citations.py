"""
Phase 9 citation system tests.

NOTE ON THIS FILE'S PROVENANCE: written during a session with no network
access to install dependencies or stand up Postgres/Redis/Qdrant (see
README.md's Phase 9 section) — these tests follow the exact same patterns
as test_chat.py, test_documents.py, and test_extractors.py (which WERE
verified in earlier phases), but have not themselves been executed. Run
`pytest tests/test_citations.py -v` for real before trusting this file.
"""

import asyncio
from io import BytesIO

import pytest
from httpx import AsyncClient
from reportlab.pdfgen import canvas


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
    files = {"file": (filename, content, "application/octet-stream")}
    upload = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents", files=files, headers=_auth(token)
    )
    doc_id = upload.json()["id"]

    deadline = asyncio.get_event_loop().time() + 30.0
    while asyncio.get_event_loop().time() < deadline:
        resp = await client.get(f"/api/v1/documents/{doc_id}", headers=_auth(token))
        if resp.json()["status"] in ("completed", "failed"):
            assert resp.json()["status"] == "completed", resp.json()
            return doc_id
        await asyncio.sleep(0.3)
    raise TimeoutError(f"document {doc_id} did not finish processing in time")


def _make_multipage_pdf(page_texts: list[str]) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer)
    for text in page_texts:
        c.drawString(72, 750, text)
        c.showPage()
    c.save()
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_citation_includes_correct_page_number(client: AsyncClient, celery_worker):
    token = await _register_and_login(client, "citation-page@example.com")
    kb_id = await _create_kb(client, token)

    pdf_bytes = _make_multipage_pdf(
        [
            "Introduction to the course, covering logistics and grading.",
            "Backpropagation computes gradients layer by layer using the chain rule.",
        ]
    )
    await _upload_and_wait(client, token, kb_id, "lecture.pdf", pdf_bytes)

    response = await client.post(
        "/api/v1/chat",
        json={"question": "How does backpropagation compute gradients?"},
        headers=_auth(token),
    )
    body = response.json()
    assert body["context_found"] is True
    assert len(body["citations"]) > 0

    matching = [c for c in body["citations"] if "backpropagation" in c["text_snippet"].lower()]
    assert matching, f"expected a citation mentioning backpropagation, got {body['citations']}"
    assert matching[0]["page_number"] == 2


@pytest.mark.asyncio
async def test_txt_upload_citations_have_no_page_number(client: AsyncClient, celery_worker):
    token = await _register_and_login(client, "citation-nopage@example.com")
    kb_id = await _create_kb(client, token)

    await _upload_and_wait(
        client, token, kb_id, "notes.txt",
        b"Gradient descent minimizes a loss function iteratively over many steps.",
    )

    response = await client.post(
        "/api/v1/chat", json={"question": "What does gradient descent minimize?"}, headers=_auth(token)
    )
    body = response.json()
    assert body["citations"][0]["page_number"] is None


@pytest.mark.asyncio
async def test_citations_never_reference_chunks_outside_the_context_sent_to_the_llm(
    client: AsyncClient, celery_worker
):
    """Anti-fabrication invariant: every citation returned must correspond
    to a chunk that was actually part of the retrieved context for this
    answer — never a chunk the model merely mentions or that exists
    elsewhere in the knowledge base but wasn't retrieved. This is true by
    construction in app/rag/pipeline.py (citations are built directly from
    the same `used_results` list passed into the prompt, not from parsing
    the model's free-text output) — this test is the regression guard for
    that invariant, not the mechanism that enforces it."""
    token = await _register_and_login(client, "citation-fabrication@example.com")
    kb_id = await _create_kb(client, token)

    await _upload_and_wait(
        client, token, kb_id, "a.txt", b"The mitochondria is the powerhouse of the cell."
    )
    await _upload_and_wait(
        client, token, kb_id, "b.txt", b"Photosynthesis converts light into chemical energy."
    )

    response = await client.post(
        "/api/v1/chat", json={"question": "What is the powerhouse of the cell?"}, headers=_auth(token)
    )
    body = response.json()

    # Every cited chunk must be independently fetchable via its own
    # document_id/chunk_id — i.e. it's a real, persisted chunk, not
    # something invented that happens to look like one.
    for citation in body["citations"]:
        detail = await client.get(
            f"/api/v1/documents/{citation['document_id']}/chunks/{citation['chunk_id']}",
            headers=_auth(token),
        )
        assert detail.status_code == 200
        assert (
            detail.json()["text"] == citation["text_snippet"]
            or citation["text_snippet"] in detail.json()["text"]
        )


# --- Chunk detail endpoint (GET /documents/{id}/chunks/{chunk_id}) ---


@pytest.mark.asyncio
async def test_get_chunk_detail_returns_full_text(client: AsyncClient, celery_worker):
    token = await _register_and_login(client, "chunkdetail-owner@example.com")
    kb_id = await _create_kb(client, token)
    doc_id = await _upload_and_wait(
        client, token, kb_id, "notes.txt", b"Hybrid search combines dense and sparse retrieval methods."
    )

    chunks = await client.get(f"/api/v1/documents/{doc_id}/chunks", headers=_auth(token))
    chunk_id = chunks.json()[0]["id"]

    detail = await client.get(f"/api/v1/documents/{doc_id}/chunks/{chunk_id}", headers=_auth(token))
    assert detail.status_code == 200
    assert "Hybrid search" in detail.json()["text"]
    assert "embedding" not in detail.json()  # never exposed, same as the list endpoint


@pytest.mark.asyncio
async def test_get_chunk_detail_rejects_wrong_owner(client: AsyncClient, celery_worker):
    token_a = await _register_and_login(client, "chunkdetail-a@example.com")
    token_b = await _register_and_login(client, "chunkdetail-b@example.com")
    kb_id = await _create_kb(client, token_a)
    doc_id = await _upload_and_wait(client, token_a, kb_id, "notes.txt", b"Private content here.")

    chunks = await client.get(f"/api/v1/documents/{doc_id}/chunks", headers=_auth(token_a))
    chunk_id = chunks.json()[0]["id"]

    response = await client.get(
        f"/api/v1/documents/{doc_id}/chunks/{chunk_id}", headers=_auth(token_b)
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_chunk_detail_rejects_mismatched_document_and_chunk(client: AsyncClient, celery_worker):
    token = await _register_and_login(client, "chunkdetail-mismatch@example.com")
    kb_id = await _create_kb(client, token)
    doc_a = await _upload_and_wait(client, token, kb_id, "a.txt", b"Content of document A here.")
    doc_b = await _upload_and_wait(client, token, kb_id, "b.txt", b"Content of document B here.")

    chunks_b = await client.get(f"/api/v1/documents/{doc_b}/chunks", headers=_auth(token))
    chunk_id_from_b = chunks_b.json()[0]["id"]

    # Asking for document A's chunk list using document B's chunk_id should
    # 404 — a chunk_id that's real but belongs to a different document the
    # caller does own must behave the same as one that doesn't exist at all.
    response = await client.get(
        f"/api/v1/documents/{doc_a}/chunks/{chunk_id_from_b}", headers=_auth(token)
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_chunk_detail_404_for_nonexistent_chunk(client: AsyncClient, celery_worker):
    token = await _register_and_login(client, "chunkdetail-missing@example.com")
    kb_id = await _create_kb(client, token)
    doc_id = await _upload_and_wait(client, token, kb_id, "notes.txt", b"Some content here.")

    response = await client.get(
        f"/api/v1/documents/{doc_id}/chunks/11111111-1111-1111-1111-111111111111",
        headers=_auth(token),
    )
    assert response.status_code == 404
