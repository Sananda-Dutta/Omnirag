# OmniRAG — Multi-Modal AI Knowledge Assistant

A production-grade, multi-modal Retrieval-Augmented Generation (RAG) platform. Upload documents, images and web pages, then chat with your knowledge base and get **citation-backed answers** you can trace to the exact source.

OmniRAG is built as a real system, not a thin wrapper around a chat API: pluggable LLM and embedding providers, hybrid retrieval with re-ranking, background ingestion, evaluation, observability and containerised deployment.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688)
![React](https://img.shields.io/badge/React-TypeScript-61dafb)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ed)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Project Status & Roadmap](#project-status--roadmap)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Features

**Multi-modal ingestion**
- PDF, DOCX, TXT and Markdown documents
- Images
- Web URLs (page fetching and cleaning)
- Raw user-provided text
- Optional audio sources

**Retrieval**
- Hybrid search combining dense vector retrieval and BM25
- Re-ranking of candidate chunks
- Query rewriting for better recall
- Conversation memory for multi-turn context

**Generation**
- Citation-backed answers linked to source chunks
- Agent / tool-calling support
- Model routing by query complexity
- Provider-agnostic LLM and embedding abstractions (no hard-coding to a single vendor)

**Engineering**
- Asynchronous background ingestion workers
- RAG evaluation with Ragas / DeepEval
- Observability and tracing with Langfuse
- Dockerised stack with CI via GitHub Actions
- Typed React + TypeScript frontend

---

## Architecture

```
                ┌──────────────────────────┐
                │  React + TypeScript UI   │
                └────────────┬─────────────┘
                             │ REST
                ┌────────────▼─────────────┐
                │      FastAPI Backend     │
                │ auth · chat · ingestion  │
                └───┬──────────┬───────┬───┘
                    │          │       │
        ┌───────────▼──┐  ┌────▼─────┐ │  ┌──────────────────┐
        │ PostgreSQL   │  │ Vector DB│ │  │  Langfuse        │
        │ (metadata,   │  │ (Qdrant /│ │  │  (traces, evals) │
        │  chats)      │  │ pgvector)│ │  └──────────────────┘
        └──────────────┘  └──────────┘ │
                                       │ enqueue
                              ┌────────▼────────┐
                              │ Redis + Workers │
                              │ parse · chunk · │
                              │ embed · index   │
                              └─────────────────┘
```

**Query flow:** rewrite query → hybrid retrieve (dense + BM25) → re-rank → build context → route to an LLM → generate answer with citations → trace and log.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, Pydantic, SQLAlchemy |
| Database | PostgreSQL |
| Vector store | Qdrant or pgvector |
| Background jobs | Celery + Redis (or RQ) |
| Frontend | React, TypeScript, Tailwind CSS |
| Evaluation | Ragas, DeepEval |
| Observability | Langfuse |
| Deployment | Docker, Docker Compose, GitHub Actions |

---

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ and Node.js 18+ (for local, non-Docker development)
- An API key for at least one supported LLM / embedding provider

### Run with Docker Compose

```bash
git clone https://github.com/<your-username>/omnirag.git
cd omnirag

cp .env.example .env
# edit .env and add your provider keys

docker compose up --build
```

Then open:

- Frontend: http://localhost:3000
- API docs (Swagger): http://localhost:8000/docs

### Local development

**Backend**

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

---

## Configuration

All configuration is environment-driven. Example `.env`:

```env
# Providers
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your-key

# Data stores
DATABASE_URL=postgresql://user:password@postgres:5432/omnirag
REDIS_URL=redis://redis:6379/0
VECTOR_STORE=qdrant
QDRANT_URL=http://qdrant:6333

# Observability (optional)
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=
```

Swapping providers or vector stores is a configuration change, not a code change.

---

## Usage

1. **Add knowledge** — upload files, paste text, or submit a URL from the UI. Ingestion runs in the background; progress is shown per source.
2. **Ask questions** — chat against your knowledge base. Each answer includes citations pointing to the source documents and passages.
3. **Follow up** — conversation memory keeps multi-turn context.

Example API call:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarise the key findings in the uploaded report"}'
```

> Endpoint paths may differ slightly; see `/docs` for the live OpenAPI schema.

---

## Evaluation

OmniRAG includes a RAG evaluation pipeline to measure quality rather than guess at it, covering metrics such as:

- Faithfulness (is the answer grounded in the retrieved context?)
- Answer relevancy
- Context precision and recall

Run evaluations against a curated question set to compare retrieval and prompting changes.

---

## Testing

```bash
# Backend
cd backend
pytest

# Frontend
cd frontend
npm run build      # type-check + production build
```

The backend currently ships with a suite of **196 passing tests**, and the frontend builds cleanly under strict TypeScript.

---

## Project Status & Roadmap

OmniRAG is developed in 22 phases, from architecture setup to final documentation and demo.

- [x] Core architecture and backend foundations
- [x] Web URL ingestion
- [x] RAG evaluation pipeline
- [x] React + TypeScript frontend
- [ ] Remaining phases in progress (retrieval enhancements, agents/tools, observability, CI/CD, deployment hardening, final docs and demo)

Update this checklist as phases land.

---

## Project Structure

```
omnirag/
├── backend/
│   ├── app/
│   │   ├── api/            # FastAPI routes
│   │   ├── core/           # config, settings
│   │   ├── models/         # SQLAlchemy models
│   │   ├── schemas/        # Pydantic schemas
│   │   ├── services/       # ingestion, retrieval, generation
│   │   └── workers/        # background tasks
│   └── tests/
├── frontend/
│   └── src/
├── docker-compose.yml
├── .env.example
└── README.md
```

> Adjust the tree above to match your actual folder layout.

---

## Contributing

Issues and pull requests are welcome. Please open an issue first to discuss any significant change, and make sure tests pass before submitting.

---

## License

Distributed under the MIT License. See `LICENSE` for details.

---

## Author

**Sananda Dutta** — [GitHub](https://github.com/<SanandaDutta>) 
