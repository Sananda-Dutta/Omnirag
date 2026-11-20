"""
Phase 15 end-to-end: real upload -> real Celery processing -> real
/evaluation/run request, which itself runs the real /chat pipeline
per question. Same "nothing mocked" style as test_documents.py and
test_url_ingestion.py.
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


async def _create_kb(client: AsyncClient, token: str, name: str = "eval kb") -> str:
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
async def test_evaluation_run_scores_retrieval_and_generation(client: AsyncClient, celery_worker):
    token = await _register_and_login(client, "eval-run@example.com")
    kb_id = await _create_kb(client, token)

    files = {
        "file": (
            "rag_facts.txt",
            b"Retrieval augmented generation grounds LLM output in retrieved documents. "
            b"Hybrid search merges dense vector search with keyword search.",
            "text/plain",
        )
    }
    upload = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents", files=files, headers=_auth(token)
    )
    doc_id = upload.json()["id"]
    await _wait_for_status(client, token, doc_id)

    resp = await client.post(
        "/api/v1/evaluation/run",
        json={
            "knowledge_base_id": kb_id,
            "questions": [
                {
                    "question": "What does retrieval augmented generation do?",
                    "expected_answer": "Retrieval augmented generation grounds LLM output in retrieved documents.",
                    "expected_document_filenames": ["rag_facts.txt"],
                }
            ],
        },
        headers=_auth(token),
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["eval_provider"] == "local-heuristic"
    assert len(body["results"]) == 1

    row = body["results"][0]
    assert row["context_found"] is True
    assert "rag_facts.txt" in row["retrieved_document_filenames"]

    # Retrieval metrics: the only document in the KB is the expected one,
    # so a correctly-working pipeline should find it.
    assert row["retrieval"] is not None
    assert row["retrieval"]["hit_rate"] == 1.0
    assert row["retrieval"]["recall_at_k"] == 1.0

    # Generation metrics: the local extractive LLM returns text pulled
    # verbatim from the context, so faithfulness should score high, and a
    # ground-truth answer was supplied so correctness must be computed.
    assert row["generation"]["faithfulness"] is not None
    assert row["generation"]["faithfulness"] > 0.5
    assert row["generation"]["answer_correctness"] is not None

    summary = body["summary"]
    assert summary["question_count"] == 1
    assert summary["context_found_rate"] == 1.0
    assert summary["avg_hit_rate"] == 1.0


@pytest.mark.asyncio
async def test_evaluation_run_without_expected_sources_skips_retrieval_metrics(
    client: AsyncClient, celery_worker
):
    token = await _register_and_login(client, "eval-norefs@example.com")
    kb_id = await _create_kb(client, token)

    files = {"file": ("notes.txt", b"Chunking splits documents before embedding.", "text/plain")}
    upload = await client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents", files=files, headers=_auth(token)
    )
    doc_id = upload.json()["id"]
    await _wait_for_status(client, token, doc_id)

    resp = await client.post(
        "/api/v1/evaluation/run",
        json={"knowledge_base_id": kb_id, "questions": [{"question": "What splits documents?"}]},
        headers=_auth(token),
    )
    assert resp.status_code == 200
    row = resp.json()["results"][0]
    assert row["retrieval"] is None
    # No expected_answer either -> correctness not computed for this row.
    assert row["generation"]["answer_correctness"] is None
    assert resp.json()["summary"]["avg_answer_correctness"] is None


@pytest.mark.asyncio
async def test_evaluation_run_on_empty_knowledge_base_reports_no_context_found(
    client: AsyncClient,
):
    token = await _register_and_login(client, "eval-empty@example.com")
    kb_id = await _create_kb(client, token)

    resp = await client.post(
        "/api/v1/evaluation/run",
        json={"knowledge_base_id": kb_id, "questions": [{"question": "Anything in here?"}]},
        headers=_auth(token),
    )
    assert resp.status_code == 200
    row = resp.json()["results"][0]
    assert row["context_found"] is False
    # No chunks were ever retrieved -> nothing to score faithfulness against.
    assert row["generation"]["faithfulness"] is None
    assert resp.json()["summary"]["context_found_rate"] == 0.0


@pytest.mark.asyncio
async def test_evaluation_run_rejects_someone_elses_knowledge_base(client: AsyncClient):
    token_a = await _register_and_login(client, "eval-owner@example.com")
    token_b = await _register_and_login(client, "eval-intruder@example.com")
    kb_id = await _create_kb(client, token_a)

    resp = await client.post(
        "/api/v1/evaluation/run",
        json={"knowledge_base_id": kb_id, "questions": [{"question": "hello?"}]},
        headers=_auth(token_b),
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_evaluation_run_rejects_empty_question_list(client: AsyncClient):
    token = await _register_and_login(client, "eval-emptylist@example.com")

    resp = await client.post(
        "/api/v1/evaluation/run", json={"questions": []}, headers=_auth(token)
    )
    assert resp.status_code == 422
