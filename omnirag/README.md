# OmniRAG — Multi-Modal AI Knowledge Assistant

A production-oriented, portfolio-grade RAG system: upload PDFs, DOCX, images,
and URLs, then ask questions and get citation-backed answers.

This repo is being built **in phases**. Each phase is a real, working
increment — nothing is stubbed out and left broken. See `PHASES.md` (added
once we're past a couple of phases) for what's implemented vs planned.

## Current status: Phase 6 — Vector Database

What exists right now:
- **Phase 1**: FastAPI skeleton, structured logging, config, package layout, Docker.
- **Phase 2**: Async SQLAlchemy, `User`/`KnowledgeBase` models, Alembic, DB-aware health check.
- **Phase 3**: bcrypt + JWT auth, `get_current_user`, timing side-channel fix.
- **Phase 4**: Upload/validate/store/extract pipeline, real Celery + Redis
  background processing, `413`/`415`/`404` handled correctly.
- **Phase 5**: Boundary-aware chunking, `EmbeddingProvider` abstraction
  (local hashing default + real OpenAI implementation), embeddings
  persisted alongside chunks.
- **Phase 6** (new):
  - `VectorStore` interface (`app/retrieval/vector_store.py`) with a real
    Qdrant implementation. `owner_id` is a *required* parameter on every
    search/delete call — not optional — because a filtering bug in a vector
    store doesn't just leak a list item, it can feed another user's
    document content straight into a generated answer shown to the wrong
    person
  - One Qdrant collection for everyone, isolation enforced via a mandatory
    payload filter (`owner_id`, optionally narrowed by
    `knowledge_base_id`) — same pattern the Postgres tables already use,
    not a collection-per-user design (real per-collection overhead in
    Qdrant at any meaningful scale)
  - The Celery task now indexes chunks into Qdrant right after embedding;
    `status: completed` now means the document is actually searchable end
    to end, including the ANN index — not just that a row exists in Postgres
  - `POST /api/v1/search` — embeds the query, searches Qdrant, joins results
    back to Postgres for the authoritative chunk text (the vector store is
    never treated as a second source of truth for content)
  - `DimensionMismatchError` — the app refuses to silently mix vectors from
    different embedding dimensions/models in one collection if
    `EMBEDDING_PROVIDER` ever changes

**Real bug found and fixed this phase**: deleting a document cascaded its
chunks in Postgres but left the corresponding Qdrant vectors behind —
orphaned points whose `chunk_id` no longer existed anywhere, silently
corrupting future search results with dead references. Fixed by deleting
from the vector store *before* the Postgres delete (so a Qdrant failure
leaves the document intact and retryable, rather than deleted-but-still-
searchable — the worse of the two failure orderings). Covered by
`test_deleting_document_removes_it_from_search`.

**On infrastructure**: this development sandbox has no Docker access, so
rather than fake vector search, the real Qdrant 1.12.4 server binary was
downloaded from its GitHub releases and run directly — every test in this
phase runs against an actual Qdrant instance doing real HNSW search, not a
mock or an in-memory stand-in. `docker-compose.yml` runs the official
`qdrant/qdrant` image for anyone running this normally.

**Testing**: 76 tests total. Vector store tests cover nearest-neighbor
ordering, owner isolation, knowledge-base scoping, idempotent upsert, and
deletion — all against the real server. End-to-end search tests go through
the full real path (upload → Celery worker → Qdrant → search endpoint) and
include a genuine relevance check: a "gradient descent" query correctly
ranks a machine-learning chunk above unrelated cooking/recipe content.

What's deliberately **not** here yet: LLM-generated answers (Phase 7 — this
phase only returns raw matching chunks, no generation), hybrid/BM25 search
and re-ranking (Phase 8), the agent layer, the frontend.

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
├── retrieval/    # VectorStore + hybrid search/re-ranking (Phase 6 — VectorStore done; hybrid search Phase 8)
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

### Option B: Local Postgres + Redis + Qdrant

```bash
# 1. Create the dev + test databases
createuser omnirag -P            # password: omnirag
createdb omnirag -O omnirag
createdb omnirag_test -O omnirag

# 2. Run Qdrant (needs Docker, or download the binary directly — see note below)
docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant:v1.12.4

# 3. Set up the backend
cd backend
python3 -m venv .venv
. .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # defaults match the DB/Qdrant created above

# 4. Apply migrations
alembic upgrade head

# 5. Run
uvicorn app.main:app --reload
```

No Docker available? Qdrant ships as a single self-contained binary —
download the release for your platform from
[github.com/qdrant/qdrant/releases](https://github.com/qdrant/qdrant/releases)
and run it directly:

```bash
tar xzf qdrant-x86_64-unknown-linux-gnu.tar.gz
QDRANT__STORAGE__STORAGE_PATH=./qdrant_storage ./qdrant
```

This is exactly how Qdrant was run while building this phase — this dev
sandbox has no Docker access either.

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

curl -X POST localhost:8000/api/v1/search \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"query":"what does this document say about X?"}'
# -> ranked chunks with score, text, and which document each came from
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

**Phase 7: Basic RAG** — an `LLMProvider` abstraction (mirroring
`EmbeddingProvider`'s pattern, so the LLM backend isn't hard-coded to one
vendor), and wiring it to `/search`'s results: retrieve chunks, construct a
grounded prompt, generate an answer, and return it with citations back to
the specific chunks/documents used. This is what turns "we can find the
right chunks" into "we can actually answer the user's question."
