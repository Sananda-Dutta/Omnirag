# OmniRAG — Multi-Modal AI Knowledge Assistant

A production-oriented, portfolio-grade RAG system: upload PDFs, DOCX, images,
and URLs, then ask questions and get citation-backed answers.

This repo is being built **in phases**. Each phase is a real, working
increment — nothing is stubbed out and left broken. See `PHASES.md` (added
once we're past a couple of phases) for what's implemented vs planned.

## Current status: Phase 5 — Chunking + Embeddings

What exists right now:
- **Phase 1**: FastAPI skeleton, structured logging, config, package layout, Docker.
- **Phase 2**: Async SQLAlchemy, `User`/`KnowledgeBase` models, Alembic, DB-aware health check.
- **Phase 3**: bcrypt + JWT auth, `get_current_user`, timing side-channel fix.
- **Phase 4**: Upload/validate/store/extract pipeline, real Celery + Redis
  background processing, `413`/`415`/`404` handled correctly.
- **Phase 5** (new):
  - Boundary-aware recursive text chunker (`app/ingestion/chunking.py`) —
    tries paragraph, then sentence, then word breaks before ever falling
    back to a hard character cut; configurable `CHUNK_SIZE`/`CHUNK_OVERLAP`
  - `EmbeddingProvider` interface (`app/embeddings/base.py`) with two real
    implementations: `LocalHashingEmbeddingProvider` (feature hashing —
    deterministic, no network/API key, the sandbox-runnable default) and
    `OpenAIEmbeddingProvider` (real implementation of OpenAI's actual API
    shape, but only testable here against a mocked HTTP layer — this
    environment has no egress to `api.openai.com`, stated plainly rather
    than glossed over)
  - `document_chunks` table storing chunk text + a raw float-array
    embedding + which model produced it — a plain Postgres array for now,
    not yet an ANN index (that's explicitly Phase 6's job)
  - The Celery task now chunks and embeds after extraction; `status:
    completed` means the document is actually searchable, not merely that
    text was recovered from the file
  - `GET /api/v1/documents/{id}/chunks` to inspect chunks (embeddings
    themselves are never returned over the API — no client use case for
    shipping a 384-element float array)

**Two real bugs found and fixed this phase, not just in theory:**
1. `sqlalchemy.exc.MissingGreenlet` crash on every upload — `document.chunks.clear()`
   (used for idempotent reprocessing) needed the `chunks` relationship
   already loaded; async SQLAlchemy can't perform an implicit lazy-load
   outside a greenlet context. Fixed by eager-loading it alongside
   `knowledge_base`, verified live: task went from crashing to completing
   in ~0.2s.
2. A genuine Postgres deadlock between a test's `DROP TABLE` teardown and a
   still-in-flight Celery worker transaction on the same tables. Fixed by
   waiting for terminal status before the test ends; confirmed stable
   across repeated runs, not just one lucky pass.

**Testing**: 62 tests total. The local embedding provider is tested for
real, including a semantic-similarity check (texts sharing vocabulary embed
closer together than unrelated ones) — the actual property that matters for
retrieval, not just "does it return a vector."

What's deliberately **not** here yet: a real ANN vector index (Phase 6 —
similarity search against the current `document_chunks.embedding` column
would require a full table scan), LLM calls, the agent layer, the frontend.

## Architecture (why the folders look like this)

```
backend/app/
├── api/          # HTTP routes only — no business logic
├── core/         # config, logging, security primitives
├── models/       # SQLAlchemy ORM models          (Phase 2)
├── schemas/      # Pydantic request/response models (Phase 2)
├── services/     # orchestration logic that routes call
├── rag/          # the RAG pipeline itself           (Phase 7+)
├── embeddings/   # EmbeddingProvider abstraction      (Phase 5 — done)
├── llm/          # LLMProvider abstraction             (Phase 7)
├── agents/       # tool-calling agent layer           (Phase 13)
├── ingestion/     # document parsing/chunking pipeline  (Phase 4-5 — done)
├── retrieval/    # hybrid search + re-ranking          (Phase 8)
├── evaluation/   # RAG eval harness                    (Phase 15)
├── database/     # SQLAlchemy engine/session mgmt      (Phase 2)
└── utils/        # small shared helpers
```

Routers stay thin; all real logic lives in `services/` or the domain
packages (`rag/`, `llm/`, etc.). This is what lets us swap a vector DB or add
a new LLM provider later without touching the API layer.

## Running the project

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

### Switching embedding providers

`EMBEDDING_PROVIDER=local` (the default) needs nothing extra and is what
lets this whole pipeline run without any API key. To use real semantic
embeddings instead:

```bash
# in backend/.env
EMBEDDING_PROVIDER=openai
EMBEDDING_DIMENSION=1536
OPENAI_API_KEY=sk-...
```

Note this hasn't been exercised against the live OpenAI API as part of
building this project — this sandboxed dev environment has no network
access to `api.openai.com`. The implementation follows OpenAI's documented
embeddings API exactly and is unit-tested against a mocked HTTP layer
(`tests/test_embeddings.py`), but "should work" and "verified live" are
different claims, and only the first one is true right now.

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

**Phase 6: Vector database** — standing up Qdrant (or pgvector), migrating
`document_chunks.embedding` into a real ANN index, and a similarity-search
service that queries it with per-user/per-knowledge-base filtering. This is
what turns "we have embeddings sitting in a Postgres array" into "we can
actually retrieve the right chunk for a query in production," and is the
last piece needed before Phase 7 (basic RAG) can generate a grounded answer.
