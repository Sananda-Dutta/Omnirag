"""
End-to-end document ingestion tests.

These go through the full real path: HTTP upload -> validation -> disk
storage -> DB row -> Celery task enqueued on Redis -> a genuinely separate
worker process picks it up -> extracts text -> writes the result back to
Postgres -> the test polls the API until the status transitions.

Nothing here is mocked. This is intentionally the slowest test file in the
suite (real subprocess, real broker round-trip) — that cost buys confidence
that the actual production pipeline works, not just each piece in isolation.
"""

import asyncio
import io

import docx
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


async def _wait_for_status(
    client: AsyncClient, token: str, document_id: str, timeout: float = 15.0
) -> dict:
    """Polls GET /documents/{id} until status leaves PENDING/PROCESSING."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        resp = await client.get(f"/api/v1/documents/{document_id}", headers=_auth(token))
        body = resp.json()
        if body["status"] in ("completed", "failed"):
            return body
        await asyncio.sleep(0.3)
    raise TimeoutError(f"Document {document_id} did not finish processing within {timeout}s")


def _make_pdf(text: str) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer)
    c.drawString(72, 750, text)
    c.save()
    return buffer.getvalue()


def _make_docx(text: str) -> bytes:
    document = docx.Document()
    document.add_paragraph(text)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_txt_upload_is_processed_end_to_end(client: AsyncClient, celery_worker):
    token = await _register_and_login(client, "ingest-txt@example.com")
    kb_id = await _create_kb(client, token)

    files = {"file": ("notes.txt", b"Retrieval augmented generation grounds LLM output.", "text/plain")}
    upload = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents", files=files, headers=_auth(token)
    )
    assert upload.status_code == 201
    doc_id = upload.json()["id"]
    assert upload.json()["status"] == "pending"

    final = await _wait_for_status(client, token, doc_id)
    assert final["status"] == "completed"
    assert final["char_count"] > 0

    detail = await client.get(f"/api/v1/documents/{doc_id}", headers=_auth(token))
    assert "Retrieval augmented generation" in detail.json()["extracted_text"]


@pytest.mark.asyncio
async def test_pdf_upload_is_processed_end_to_end(client: AsyncClient, celery_worker):
    token = await _register_and_login(client, "ingest-pdf@example.com")
    kb_id = await _create_kb(client, token)

    pdf_bytes = _make_pdf("Chunking splits documents before embedding.")
    files = {"file": ("paper.pdf", pdf_bytes, "application/pdf")}
    upload = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents", files=files, headers=_auth(token)
    )
    assert upload.status_code == 201
    doc_id = upload.json()["id"]

    final = await _wait_for_status(client, token, doc_id)
    assert final["status"] == "completed"
    assert final["page_count"] == 1

    detail = await client.get(f"/api/v1/documents/{doc_id}", headers=_auth(token))
    assert "Chunking splits documents" in detail.json()["extracted_text"]


@pytest.mark.asyncio
async def test_docx_upload_is_processed_end_to_end(client: AsyncClient, celery_worker):
    token = await _register_and_login(client, "ingest-docx@example.com")
    kb_id = await _create_kb(client, token)

    docx_bytes = _make_docx("Hybrid search merges dense and sparse retrieval.")
    files = {
        "file": (
            "lecture.docx",
            docx_bytes,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    }
    upload = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents", files=files, headers=_auth(token)
    )
    assert upload.status_code == 201
    doc_id = upload.json()["id"]

    final = await _wait_for_status(client, token, doc_id)
    assert final["status"] == "completed"

    detail = await client.get(f"/api/v1/documents/{doc_id}", headers=_auth(token))
    assert "Hybrid search merges" in detail.json()["extracted_text"]


@pytest.mark.asyncio
async def test_encrypted_pdf_ends_in_failed_status_with_reason(client: AsyncClient, celery_worker):
    from pypdf import PdfWriter

    token = await _register_and_login(client, "ingest-encrypted@example.com")
    kb_id = await _create_kb(client, token)

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.encrypt(user_password="secret", owner_password="secret")
    buffer = io.BytesIO()
    writer.write(buffer)

    files = {"file": ("locked.pdf", buffer.getvalue(), "application/pdf")}
    upload = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents", files=files, headers=_auth(token)
    )
    doc_id = upload.json()["id"]

    final = await _wait_for_status(client, token, doc_id)
    assert final["status"] == "failed"
    assert "password-protected" in final["error_message"]


@pytest.mark.asyncio
async def test_upload_rejects_unsupported_extension(client: AsyncClient):
    token = await _register_and_login(client, "ingest-badtype@example.com")
    kb_id = await _create_kb(client, token)

    files = {"file": ("virus.exe", b"not really an exe", "application/octet-stream")}
    upload = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents", files=files, headers=_auth(token)
    )
    assert upload.status_code == 415


@pytest.mark.asyncio
async def test_upload_rejects_oversized_file(client: AsyncClient):
    from app.core.config import settings

    token = await _register_and_login(client, "ingest-toobig@example.com")
    kb_id = await _create_kb(client, token)

    oversized = b"x" * (settings.max_upload_size_bytes + 1)
    files = {"file": ("huge.txt", oversized, "text/plain")}
    upload = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents", files=files, headers=_auth(token)
    )
    assert upload.status_code == 413


@pytest.mark.asyncio
async def test_upload_rejects_into_someone_elses_knowledge_base(client: AsyncClient):
    token_a = await _register_and_login(client, "ingest-owner@example.com")
    token_b = await _register_and_login(client, "ingest-intruder@example.com")
    kb_id = await _create_kb(client, token_a)

    files = {"file": ("notes.txt", b"some content", "text/plain")}
    upload = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents", files=files, headers=_auth(token_b)
    )
    assert upload.status_code == 404


@pytest.mark.asyncio
async def test_chunks_endpoint_returns_real_chunks_with_embeddings(client: AsyncClient, celery_worker):
    token = await _register_and_login(client, "ingest-chunks@example.com")
    kb_id = await _create_kb(client, token)

    # Long enough to guarantee more than one chunk at the default 1000-char
    # chunk size, so this test actually exercises multi-chunk behavior.
    long_text = "Hybrid search merges dense and sparse retrieval. " * 60
    files = {"file": ("long.txt", long_text.encode(), "text/plain")}
    upload = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents", files=files, headers=_auth(token)
    )
    doc_id = upload.json()["id"]
    await _wait_for_status(client, token, doc_id)

    response = await client.get(f"/api/v1/documents/{doc_id}/chunks", headers=_auth(token))
    assert response.status_code == 200
    chunks = response.json()

    assert len(chunks) > 1
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert c["char_count"] > 0
        assert c["embedding_model"] == "local-hashing-384d"
        assert c["embedding_dimension"] == 384
        assert "embedding" not in c  # never exposed over the API — see schemas/chunk.py


@pytest.mark.asyncio
async def test_user_cannot_list_another_users_document_chunks(client: AsyncClient, celery_worker):
    token_a = await _register_and_login(client, "ingest-chunkowner@example.com")
    token_b = await _register_and_login(client, "ingest-chunkintruder@example.com")
    kb_id = await _create_kb(client, token_a)

    files = {"file": ("notes.txt", b"some private content to chunk", "text/plain")}
    upload = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents", files=files, headers=_auth(token_a)
    )
    doc_id = upload.json()["id"]
    await _wait_for_status(client, token_a, doc_id)

    response = await client.get(f"/api/v1/documents/{doc_id}/chunks", headers=_auth(token_b))
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_user_cannot_fetch_another_users_document(client: AsyncClient, celery_worker):
    token_a = await _register_and_login(client, "ingest-docowner@example.com")
    token_b = await _register_and_login(client, "ingest-docintruder@example.com")
    kb_id = await _create_kb(client, token_a)

    files = {"file": ("notes.txt", b"private content here", "text/plain")}
    upload = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents", files=files, headers=_auth(token_a)
    )
    doc_id = upload.json()["id"]

    response = await client.get(f"/api/v1/documents/{doc_id}", headers=_auth(token_b))
    assert response.status_code == 404

    # Let background processing finish before the test (and its table-drop
    # teardown) ends — a still-in-flight worker transaction racing the
    # teardown's DROP TABLE against the same tables is a genuine deadlock,
    # not a hypothetical one (caught during Phase 5 development).
    await _wait_for_status(client, token_a, doc_id)
