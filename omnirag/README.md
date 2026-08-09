# OmniRAG — Multi-Modal AI Knowledge Assistant

A production-oriented, portfolio-grade RAG system: upload PDFs, DOCX, images,
and URLs, then ask questions and get citation-backed answers.

This repo is being built **in phases**. Each phase is a real, working
increment — nothing is stubbed out and left broken. See `PHASES.md` (added
once we're past a couple of phases) for what's implemented vs planned.

## Current status: Phase 4 — Document Ingestion

What exists right now:
- **Phase 1**: FastAPI skeleton, structured logging, config, package layout, Docker.
- **Phase 2**: Async SQLAlchemy, `User`/`KnowledgeBase` models, Alembic, DB-aware health check.
- **Phase 3**: bcrypt + JWT auth, `get_current_user`, timing side-channel fix.
- **Phase 4** (new):
  - `KnowledgeBase` CRUD (`POST/GET/DELETE /api/v1/knowledge-bases`) and
    `Document` upload/list/get/delete, all enforcing per-user ownership at
    the query level (404, never 403 — see `knowledge_base_service.py`)
  - File upload with real validation (extension allowlist, size limit) —
    `413`/`415` on rejection, checked *before* anything touches disk or the DB
  - `StorageBackend` abstraction (`app/ingestion/storage.py`) — local disk
    now, swappable for S3 later without touching callers; on-disk filenames
    are server-generated, never derived from client input (path-traversal
    protection)
  - Text extraction for PDF/DOCX/TXT/MD (`app/ingestion/extractors.py`),
    each format raising a clean `ExtractionError` a user can act on
    (e.g. "PDF is password-protected") instead of leaking a library traceback
  - **Real background processing**: Celery + Redis. Upload returns
    immediately with `status: pending`; a separate worker process extracts
    the text and updates the row through `processing` → `completed`/`failed`
  - `docker-compose.yml` now runs `postgres`, `redis`, `backend`, and a
    dedicated `worker` service

**Bug found and fixed this phase**: `/knowledge-bases/{id}` and related
routes originally typed the path param as `str`, so a malformed UUID
reached the database layer and crashed with a raw, unhandled 500 (full
asyncpg stack trace in the response). Fixed by typing path params as
`uuid.UUID` so FastAPI validates and rejects with a clean 422 before the
request ever reaches a query.

**Testing**: 39 tests total. Ingestion is tested genuinely end-to-end —
real generated PDFs (via reportlab) and DOCX files (via python-docx), a
real Celery worker subprocess consuming from a real Redis broker and
writing to the real test Postgres database, polled through the actual HTTP
API until processing completes. Not mocked at any layer.

What's deliberately **not** here yet: chunking and embeddings (Phase 5),
vector search (Phase 6), LLM calls, the agent layer, the frontend. OCR for
scanned/image-only PDFs is explicitly out of scope for now — extraction
fails cleanly with a message saying so, rather than returning blank text.

## Architecture (why the folders look like this)

```
backend/app/
├── api/          # HTTP routes only — no business logic
├── core/         # config, logging, security primitives
├── models/       # SQLAlchemy ORM models          (Phase 2)
├── schemas/      # Pydantic request/response models (Phase 2)
├── services/     # orchestration logic that routes call
├── rag/          # the RAG pipeline itself           (Phase 7+)
├── embeddings/   # EmbeddingProvider abstraction      (Phase 5)
├── llm/          # LLMProvider abstraction             (Phase 7)
├── agents/       # tool-calling agent layer           (Phase 13)
├── ingestion/     # document parsing/chunking pipeline  (Phase 4-5)
├── retrieval/    # hybrid search + re-ranking          (Phase 8)
├── evaluation/   # RAG eval harness                    (Phase 15)
├── database/     # SQLAlchemy engine/session mgmt      (Phase 2)
└── utils/        # small shared helpers
```

Routers stay thin; all real logic lives in `services/` or the domain
packages (`rag/`, `llm/`, etc.). This is what lets us swap a vector DB or add
a new LLM provider later without touching the API layer.

## Running Phase 2

### Option A: Docker (simplest — Postgres + backend + migrations, all wired up)

```bash
cp backend/.env.example backend/.env
docker compose up --build
```

The backend container runs `alembic upgrade head` automatically before
starting the server.

### Option B: Local Postgres

```bash
# 1. Create the dev + test databases
createuser omnirag -P            # password: omnirag
createdb omnirag -O omnirag
createdb omnirag_test -O omnirag

# 2. Set up the backend
cd backend
python3 -m venv .venv
. .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # defaults match the DB created above

# 3. Apply migrations
alembic upgrade head

# 4. Run
uvicorn app.main:app --reload
```

Then:
- `GET http://localhost:8000/api/v1/health` → `{"status": "ok", "checks": {"database": "ok"}, ...}`
  (returns HTTP 503 + `"status": "degraded"` if Postgres is unreachable)
- `GET http://localhost:8000/docs` → interactive Swagger UI — click
  "Authorize" and log in with a registered user to try protected endpoints
  directly from the docs

Try the full flow from the command line:

```bash
curl -X POST localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","password":"a-real-password"}'

TOKEN=$(curl -s -X POST localhost:8000/api/v1/auth/login \
  -d "username=you@example.com&password=a-real-password" | python3 -c "import sys,json;print(json.load(sys.stdin)['access_token'])")

KB_ID=$(curl -s -X POST localhost:8000/api/v1/knowledge-bases \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"name":"ML coursework"}' | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")

curl -X POST "localhost:8000/api/v1/knowledge-bases/$KB_ID/documents" \
  -H "Authorization: Bearer $TOKEN" -F "file=@/path/to/some.pdf"
# -> {"status": "pending", ...} — poll GET /api/v1/documents/{id} to watch
# it move to "processing" then "completed" (or "failed" with a reason)
```

### Running the background worker (required for uploads to actually process)

```bash
cd backend
. .venv/bin/activate
celery -A app.workers.celery_app worker --loglevel=info
```

Without a worker running, uploads succeed and sit in `status: pending`
forever — the API and the worker are deliberately decoupled, so this is
expected, not a bug (Redis holds the queued task until something consumes it).

### Tests

Tests run against the separate `omnirag_test` database (never the dev one).
The document-ingestion tests spin up a real Celery worker subprocess
automatically — no manual worker needed to run the suite:

```bash
cd backend
. .venv/bin/activate
pytest tests/ -v
```

### Working with migrations

```bash
# After changing/adding a model in app/models/:
alembic revision --autogenerate -m "describe the change"
alembic upgrade head
```

Always review an autogenerated migration before applying it — Alembic is
good but not infallible, especially around column type changes and renames
(it sees a rename as a drop + add unless you edit the migration by hand).

## Next phase

**Phase 5: Chunking + embeddings** — splitting `extracted_text` into
overlapping chunks, a configurable `EmbeddingProvider` abstraction (so the
embedding model isn't hard-coded to one vendor), and a `document_chunks`
table. This is the last phase before Phase 6 gives those chunks somewhere
to actually live for retrieval (a vector database).
