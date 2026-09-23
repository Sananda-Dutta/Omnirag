# OmniRAG — Multi-Modal AI Knowledge Assistant

A production-oriented, portfolio-grade RAG system: upload PDFs, DOCX, images,
and URLs, then ask questions and get citation-backed answers.

This repo is being built **in phases**. Each phase is a real, working
increment — nothing is stubbed out and left broken. See `PHASES.md` (added
once we're past a couple of phases) for what's implemented vs planned.

> **Verified**: Phase 9 was originally written in a session with no network
> access and shipped only syntax-checked, not executed. It, along with
> Phases 12, 15, and 17, has since been run for real against live
> Postgres + Redis + Qdrant — `alembic upgrade head` and `pytest tests/ -v`
> (196 tests) both pass. See "Current status" below for what's still
> unverified (the two real Anthropic/OpenAI LLM providers, and the Ragas
> evaluator — both need API keys this sandbox doesn't have) and what's not
> built yet (Phases 10/11/13/14/16).

## Current status: Phases 9, 12, 15 & 17

What's deliberately **not** here yet: conversation memory across turns
(Phase 10 — `/chat` is single-turn only right now, each question is
independent — a `conversations`/`messages` schema exists in
`app/models/conversation.py` but isn't wired into `/chat` yet), query
rewriting/expansion and intent detection, the agent/tool-calling layer
(Phase 11/13), streaming responses (Phase 14), and observability (Phase 16
— `LANGFUSE_*` settings exist as placeholders only). Phases 12, 15, and 17
were built out of the original sequence, at the user's request, on top of
the Phase 9 codebase — see each phase's own section below for what that
means for verification status.

What exists right now:
- **Phase 1**: FastAPI skeleton, structured logging, config, package layout, Docker.
- **Phase 2**: Async SQLAlchemy, `User`/`KnowledgeBase` models, Alembic, DB-aware health check.
- **Phase 3**: bcrypt + JWT auth, `get_current_user`, timing side-channel fix.
- **Phase 4**: Upload/validate/store/extract pipeline, real Celery + Redis
  background processing, `413`/`415`/`404` handled correctly.
- **Phase 5**: Boundary-aware chunking, `EmbeddingProvider` abstraction
  (local hashing default + real OpenAI implementation), embeddings
  persisted alongside chunks.
- **Phase 6**: `VectorStore` interface + real Qdrant implementation,
  `POST /api/v1/search` returning ranked chunks with per-user isolation
  enforced at the vector-store level.
- **Phase 7** (new):
  - `LLMProvider` interface (`app/llm/base.py`) with three implementations:
    `LocalExtractiveLLMProvider` (the default — real extractive
    sentence-ranking, zero API key/network needed, so the *entire* RAG
    pipeline runs end-to-end in this sandbox), `AnthropicLLMProvider`, and
    `OpenAILLMProvider` (both real implementations of their documented
    APIs, tested here only against mocked clients — this sandbox can reach
    `api.anthropic.com` but has no usable API key for either service)
  - `app/rag/pipeline.py` — the actual RAG orchestration: search → build a
    labeled, size-capped context block → generate → return citations tied
    to specific chunks/documents
  - **A real hallucination-control decision, not just a prompt
    instruction**: if search returns zero results, the LLM is never called
    at all. A fixed "not found" response is returned immediately — this is
    a stronger guarantee than prompting a model not to guess, because the
    model is never given the chance to
  - `POST /api/v1/chat` — the end-to-end question-answering endpoint
- **Phase 8** (new):
  - `/search` and `/chat` now run a real hybrid retrieval pipeline: dense
    vector search (Qdrant) **and** Postgres full-text keyword search, merged
    via Reciprocal Rank Fusion (`app/retrieval/fusion.py`), then re-scored
    by a lexical reranker (`app/retrieval/reranker.py`) before truncating
    to the final result count — replacing Phase 6's dense-only search
  - Keyword search uses a Postgres **generated `tsvector` column** with a
    GIN index (`app/models/document_chunk.py`) rather than a separate
    search engine — Postgres already runs as the system of record, so this
    avoids operating a second stateful service for a capability it already
    provides, the same reasoning as not building a second sync DB stack
    for the Celery worker back in Phase 4
  - RRF fusion is used instead of normalizing and combining raw scores
    directly, because Qdrant's cosine similarity and Postgres's
    `ts_rank_cd` live on completely incomparable scales — RRF sidesteps
    that by fusing on rank position, not raw score
  - The reranker is a real, working, clearly-labeled lexical technique
    (normalized term frequency × in-context inverse document frequency) —
    explicitly **not** a neural cross-encoder, since no such model is
    reachable from this sandbox (no HuggingFace access, no reachable
    reranking API). Same honesty pattern as `LocalHashingEmbeddingProvider`
    and `LocalExtractiveLLMProvider`: a real technique with a stated
    ceiling, not a placeholder pretending to be something else
  - `ENABLE_KEYWORD_SEARCH` / `ENABLE_RERANKING` can each be turned off
    independently, falling back toward Phase 6's dense-only behavior —
    mainly so Phase 15 (RAG evaluation) can A/B compare configurations
    against the same question set later, not a feature users would toggle

**On this phase's development process**: at the user's request, this phase
skipped the live end-to-end curl walkthroughs and repeated-run stability
checks used in earlier phases, in favor of running the actual test suite
and fixing what it caught. All 110 tests (up from 92 at the end of Phase 7)
passed cleanly on the first full run — no bugs surfaced this time, likely
because the design patterns this phase reused (Strategy-pattern interfaces,
owner_id-required isolation, real-local-default-plus-mocked-real-provider)
were already established and tested in Phases 5-7.

**Two real bugs found and fixed this phase** (both caught by the tests
written for this phase, not by inspection):
1. The grounded system prompt's own instructions text said "found in the
   `<context>` block below" — which put the literal delimiter string inside
   the prose, colliding with the regex `LocalExtractiveLLMProvider` uses to
   pull the actual context back out of the prompt. It matched from that
   prose mention instead of the real tag, pulling half the instructions
   into the "extracted" answer. Fixed by removing every literal mention of
   the delimiter syntax from the prompt's prose, keeping the delimiter only
   where it's actually used.
2. The extractive provider's relevance scoring counted *any* shared word
   between question and sentence as "overlap" — including words like "is"
   and "the". A query about "quantum entanglement" scored a false-positive
   match against an unrelated sentence about "sunny weather" purely because
   both contained the word "is". Fixed with a small stopword filter scoped
   specifically to relevance scoring (the extracted output text itself is
   untouched — only what counts as "relevant enough to include" changed).

**On the two mocked providers**: `AnthropicLLMProvider` and
`OpenAILLMProvider` follow their vendors' documented APIs exactly and are
unit-tested against mocked clients, but have not been exercised against the
live services as part of building this project — this sandboxed
environment has no usable API key for either. "Should work" and "verified
live" are different claims; only the first is true right now for these two.

**Testing**: 92 tests total (up from 76 at the end of Phase 6). End-to-end
chat tests go through the full real path — upload → Celery worker → Qdrant
→ search → grounded generation → citations — using the local extractive
provider, the only one this sandbox can actually execute without an API key.

- **Phase 9** (new):
  - **Page-number citations**: PDF extraction now exposes per-page text
    (it always computed this internally, just threw it away before), and
    chunking runs per-page for PDFs (`chunk_pages` in
    `app/ingestion/chunking.py`) so every chunk carries an unambiguous
    `page_number`. Documented tradeoff: a paragraph spanning a page break
    now gets split there even if it would otherwise fit in one chunk —
    "always know which page a citation is from" mattered more than
    "never split a mid-page-break paragraph" for a citation system
    specifically. TXT/MD/DOCX chunks keep `page_number: null` — they have
    no real page concept, so a placeholder like `1` would be misleading
  - `GET /documents/{document_id}/chunks/{chunk_id}` — resolves "click a
    citation" to the exact full source passage, not just the ~200-char
    snippet already inline in a chat response's citation list
  - **Anti-fabrication is structural, not a runtime check**: citations were
    already built directly from the same chunk list passed into the LLM's
    context (`app/rag/pipeline.py`), never parsed from the model's free-text
    output — so there was never a code path that could invent a citation to
    a document that wasn't actually retrieved. Phase 9 adds a regression
    test for this invariant (`test_citations_never_reference_chunks_
    outside_the_context_sent_to_the_llm`) rather than new enforcement code,
    since the guarantee already existed by construction

**This phase's development was unusually constrained, worth being explicit
about**: it started in a completely fresh sandbox with no dependencies
installed, and — unlike every prior phase — this session's network egress
was entirely blocked (`x-deny-reason: host_not_allowed` on PyPI, GitHub,
and even Ubuntu's own package mirrors). The Phase 8 codebase was recovered
from the zip already delivered to the user; everything written in Phase 9
is syntax-checked (`python3 -m py_compile` — stdlib only, no dependency
install needed) but **has not been run** — no `pytest`, no live database,
no autogenerated migration. The migration in this phase was hand-written
following the exact pattern of the previous (verified) migration rather
than generated against a real database, and is flagged as such in its own
file header. Treat this phase's correctness claims as "carefully reasoned,
not independently verified" until someone runs the suite for real.

What's deliberately **not** here yet: conversation memory across turns
(Phase 10 — `/chat` is single-turn only right now, each question is
independent), query rewriting/expansion and intent detection, the
agent/tool-calling layer (Phase 13), streaming responses (Phase 14).

- **Phase 12 — Web ingestion** (new, this session):
  - `app/ingestion/web_fetcher.py` (`fetch_and_extract`) already existed
    from a prior session with no network access, alongside
    `ingest_url_document` in `app/services/document_service.py` and
    `POST /knowledge-bases/{id}/documents/url` — but it had never been run:
    **two real bugs were found and fixed** this session — `DocumentRead`
    declared `file_size_bytes: int` (non-nullable) which would have thrown
    on every URL-sourced document's response, since there's no honest byte
    count for a page that was never uploaded as a file; and `source_url`
    wasn't exposed in the response schema at all. Both fixed.
    `beautifulsoup4` was also missing from `requirements.txt` despite being
    a hard import.
  - SSRF guarding (`_is_safe_host`) rejects private/link-local/loopback
    targets before ever making a request, real content-type and
    `MAX_CONTENT_BYTES` limits, and the page's real `<title>` replaces the
    URL-as-placeholder filename once fetched
  - **Verified for real this session**, not just syntax-checked: this repo
    now has a working local Postgres + Redis + a real Qdrant binary (see
    "Running the project" below for how cheap this is without Docker), and
    25 new tests were added and passed — `tests/test_web_fetcher.py`
    (mocked-transport unit tests for the SSRF/content-type/size-cap edge
    cases) and `tests/test_url_ingestion.py` (true end-to-end tests that
    fetch real pages from `github.com`/`raw.githubusercontent.com` through
    the real Celery worker — no mocking)

- **Phase 15 — RAG evaluation** (new, this session):
  - `app/evaluation/` follows the same `EvaluationProvider` Strategy
    pattern as `EmbeddingProvider`/`LLMProvider`/`VectorStore`:
    `LocalHeuristicEvaluator` (default, zero dependencies — lexical overlap
    for faithfulness/correctness, embedding cosine similarity for answer
    relevancy) and `RagasEvaluator` (a real integration against the
    [Ragas](https://github.com/explodinggradients/ragas) library's
    LLM-judged metrics, lazily imported since `ragas` isn't a hard
    dependency — set `EVAL_PROVIDER=ragas` and a real `OPENAI_API_KEY` to
    use it)
  - **A real upstream dependency bug, found and worked around**: a bare
    `pip install ragas` resolves an incompatible `langchain-community`
    (latest, dropped a submodule ragas 0.4.3 still imports at module load).
    Documented fix, verified working in this sandbox:
    `pip install ragas langchain-community==0.3.27`. See
    `app/evaluation/ragas_evaluator.py`'s docstring.
  - `app/evaluation/metrics.py` — plain-function retrieval metrics
    (precision@k, recall@k, MRR, hit rate)
  - `POST /api/v1/evaluation/run` (`app/services/evaluation_service.py`)
    runs the **real** `/chat` pipeline per question (not a parallel
    eval-only code path) against a user-supplied question set, scoring
    both retrieval (when `expected_document_filenames` is given) and
    generation quality, returning per-question results plus averages
  - 34 new tests, all passing against real infrastructure: pure-function
    metric tests, `LocalHeuristicEvaluator` behavior tests, real
    end-to-end tests through `/evaluation/run` against an uploaded
    document, and `RagasEvaluator` construction-level tests (no live
    OpenAI call — no usable key in this sandbox, same honest limit as the
    Phase 7 Anthropic/OpenAI provider tests)

- **Phase 17 — Frontend** (new, this session):
  - `frontend/`: React 19 + TypeScript + Vite 8 + Tailwind CSS v4, React
    Router, and TanStack Query. A deliberate design system (ink/paper/gold
    palette; Lora for headings, Inter for UI text, JetBrains Mono reserved
    for actual data — filenames, chunk indices, scores) rather than the
    generic dark-mode-SaaS defaults
  - Auth (login/register/JWT in `localStorage`), knowledge base
    create/list/delete, drag-and-drop file upload **and** URL ingestion
    (Phase 12) with live status polling while a document is
    pending/processing, a chat interface with numbered citation chips that
    open a modal showing the exact source chunk (Phase 9), and an
    evaluation runner (Phase 15) that builds a small QA set inline and
    renders the scored report
  - **Verified**: `tsc -b` (strict, `noUnusedLocals`/`noUnusedParameters`
    on) and `npm run build` both succeed cleanly. The full stack was then
    smoke-tested for real — backend + Celery worker + Postgres + Redis +
    Qdrant all running, `vite dev` proxying `/api` to the backend exactly
    as configured — through a real register → login → create knowledge
    base → upload → chat → evaluate flow, hitting the dev server's own
    `/api/*` URLs (i.e. the exact requests the browser UI makes), not the
    backend directly. What this does **not** cover: an actual browser
    rendering the UI (no headless-browser tooling reachable in this
    sandbox to take a screenshot or click through it) — the visual
    layout/interactions haven't been eyeballed, only proven to compile,
    build, and correctly drive the real API.

## Architecture (why the folders look like this)

```
backend/app/
├── api/          # HTTP routes only — no business logic
├── core/         # config, logging, security primitives
├── models/       # SQLAlchemy ORM models          (Phase 2)
├── schemas/      # Pydantic request/response models (Phase 2)
├── services/     # orchestration logic that routes call
├── rag/          # RAG pipeline + prompts             (Phase 7 — basic RAG done)
├── embeddings/   # EmbeddingProvider abstraction      (Phase 5 — done)
├── llm/          # LLMProvider abstraction             (Phase 7 — done)
├── agents/       # tool-calling agent layer           (Phase 13)
├── ingestion/     # document parsing/chunking pipeline  (Phase 4-5 — done, + Phase 12 web ingestion)
├── retrieval/    # VectorStore + hybrid search/re-ranking (Phase 6+8 — done)
├── evaluation/   # RAG eval harness                    (Phase 15 — done)
├── database/     # SQLAlchemy engine/session mgmt      (Phase 2)
└── utils/        # small shared helpers
```

```
frontend/src/
├── lib/          # api.ts (typed client), types.ts        (Phase 17 — done)
├── context/      # AuthContext (JWT session)
├── hooks/        # React Query hooks, one module per resource
├── components/   # documents/, chat/, evaluation/, knowledge-base/, layout/, common/
└── pages/        # LoginPage, RegisterPage, DashboardPage, KnowledgeBasePage
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

curl -X POST localhost:8000/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"question":"what does this document say about X?"}'
# -> {"answer": "...", "citations": [...], "model_used": "...", "context_found": true}
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

### Running the frontend (Phase 17)

```bash
cd frontend
npm install
npm run dev
```

Opens on `http://localhost:5173`. Its Vite dev server proxies `/api/*` to
`http://localhost:8000` (see `vite.config.ts`), so the backend just needs
to be running on its default port — no CORS config, no `.env` needed on
the frontend side. Register an account, create a knowledge base, and
upload/chat/evaluate from there.

`npm run build` produces a static `frontend/dist/` you can serve with any
static file host — it has no server-side rendering step, just talk to the
backend's real URL in production (set up a reverse proxy the same way the
dev server's `/api` proxy works, or adjust `lib/api.ts`'s base URL).

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

### Switching the RAG generation provider

`LLM_PROVIDER=local` (the default) does real extractive answering — it
ranks and returns the most relevant retrieved sentences verbatim, clearly
labeled as extracted rather than generated (see
`app/llm/local_extractive.py`). It's what lets `/chat` work end-to-end with
no API key. For real generated answers:

```bash
# in backend/.env — pick one
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# or
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

Same caveat as the OpenAI embedding provider above: neither has been
exercised against its live API while building this project (no network
access to `api.openai.com`, and no usable API key for `api.anthropic.com`
despite it being reachable here). Both follow their documented APIs exactly
and are unit-tested against mocked clients (`tests/test_llm.py`).

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

**Phase 10: Conversation memory** — `conversations` and `messages` tables,
`/chat` accepting a conversation_id and actually using prior turns as
context (recent-message memory + summarization for older turns, per the
original spec's token/cost-aware design), and `GET/DELETE /conversations`.
This is also the natural point to revisit persisting citations to a real
`citations` table tied to each assistant message, which Phase 9 explicitly
deferred since there was no `messages` row yet to attach them to.

Phases 12, 15, and 17 (web ingestion, RAG evaluation, frontend) were
pulled forward out of sequence and are done — see "Current status" above.
Still open, in original order: Phase 10 (above), Phase 11 (agent/tool
calling), Phase 13 (model routing by query complexity), Phase 14
(streaming responses), Phase 16 (Langfuse observability), and Phases
18-22 (security hardening, deployment/CI, final docs).
