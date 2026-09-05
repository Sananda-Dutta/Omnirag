"""
Application configuration.

Why this exists:
    Scattering `os.getenv("SOME_KEY")` calls across the codebase is how projects
    end up with silent misconfiguration in production (a typo'd env var name
    just returns None and fails somewhere weird, three layers deep).

    Pydantic's BaseSettings gives us:
      - a single source of truth for every config value the app needs
      - validation at startup (the app refuses to boot with bad config,
        instead of failing on the first request that touches the bad value)
      - type coercion (e.g. "8000" from the environment becomes int 8000)
      - a documented contract: this file + .env.example together tell any
        new contributor exactly what the app needs to run

This is intentionally the ONLY place in the codebase allowed to call
`os.environ` / read `.env` directly. Every other module imports `settings`
from here.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # --- App metadata ---
    APP_NAME: str = "OmniRAG"
    APP_ENV: Literal["local", "development", "staging", "production"] = "local"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"

    # --- Server ---
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # --- Database (wired up in Phase 2) ---
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://omnirag:omnirag@localhost:5432/omnirag",
        description="Async SQLAlchemy connection string for PostgreSQL.",
    )

    # --- Vector DB (wired up in Phase 6) ---
    VECTOR_DB_URL: str = Field(default="http://localhost:6333")
    VECTOR_DB_COLLECTION: str = Field(
        default="document_chunks",
        description="Qdrant collection name. Single collection for all "
        "users/knowledge bases — isolation is enforced per-query via a "
        "mandatory payload filter (owner_id, and optionally "
        "knowledge_base_id), not by separate collections per user. See "
        "app/retrieval/qdrant_store.py for why.",
    )
    DEFAULT_SEARCH_TOP_K: int = Field(default=5)

    # --- Redis / background jobs (used starting Phase 4) ---
    REDIS_URL: str = Field(default="redis://localhost:6379/0")

    # --- File storage / ingestion (Phase 4) ---
    UPLOAD_DIR: str = Field(default="./uploads")
    MAX_UPLOAD_SIZE_MB: int = Field(default=25)
    ALLOWED_UPLOAD_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx", ".txt", ".md")

    # --- Chunking (Phase 5) ---
    CHUNK_SIZE: int = Field(
        default=1000,
        description="Target chunk size in characters. Sized for typical "
        "embedding-model input limits and retrieval precision — see "
        "app/ingestion/chunking.py for the full tradeoff explanation.",
    )
    CHUNK_OVERLAP: int = Field(
        default=150,
        description="Characters of overlap between consecutive chunks, so a "
        "sentence split across a chunk boundary is still fully readable in "
        "at least one chunk.",
    )

    # --- Embeddings (Phase 5) ---
    EMBEDDING_PROVIDER: Literal["local", "openai"] = Field(
        default="local",
        description="'local' needs no network/API key and is what this repo "
        "can actually run end-to-end in a sandboxed environment. 'openai' is "
        "a real implementation but needs OPENAI_API_KEY and network access "
        "to api.openai.com.",
    )
    EMBEDDING_DIMENSION: int = Field(
        default=384,
        description="Must match the chosen provider's actual output "
        "dimension (384 for the local provider's default config; 1536 for "
        "OpenAI text-embedding-3-small). Changing providers without "
        "updating this — and re-embedding existing chunks — silently "
        "produces vectors of the wrong shape.",
    )
    OPENAI_EMBEDDING_MODEL: str = Field(default="text-embedding-3-small")

    # --- Auth (wired up in Phase 3) ---
    SECRET_KEY: str = Field(
        default="CHANGE_ME_IN_PRODUCTION",
        description="Used to sign JWTs. Must be overridden outside local dev.",
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # --- LLM / embedding provider API keys ---
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None

    # --- RAG generation (Phase 7) ---
    LLM_PROVIDER: Literal["local", "anthropic", "openai"] = Field(
        default="local",
        description="'local' needs no network/API key and is what lets this "
        "repo run the full RAG pipeline end-to-end in a sandboxed "
        "environment — see app/llm/local_extractive.py for exactly what it "
        "does and doesn't do. 'anthropic' and 'openai' are real "
        "implementations but need their respective API keys and network access.",
    )
    ANTHROPIC_MODEL: str = Field(default="claude-sonnet-4-5-20250929")
    OPENAI_CHAT_MODEL: str = Field(default="gpt-4o-mini")
    RAG_TOP_K: int = Field(
        default=5, description="How many chunks are retrieved as context per question."
    )
    RAG_MAX_CONTEXT_CHARS: int = Field(
        default=8000,
        description="Hard cap on total retrieved-context size sent to the "
        "LLM, regardless of RAG_TOP_K — a safety bound against a pathological "
        "case (very large chunks) blowing the model's context window or "
        "running up cost on a single request.",
    )

    # --- Hybrid retrieval + re-ranking (Phase 8) ---
    ENABLE_KEYWORD_SEARCH: bool = Field(
        default=True,
        description="Adds Postgres full-text search alongside Qdrant dense "
        "search, merged via Reciprocal Rank Fusion. Off falls back to "
        "dense-only search (Phase 6 behavior) — useful for A/B comparison "
        "once Phase 15 (evaluation) exists.",
    )
    ENABLE_RERANKING: bool = Field(
        default=True,
        description="Re-scores the fused candidate set with a lexical "
        "reranker before truncating to RAG_TOP_K. See "
        "app/retrieval/reranker.py for what this does and doesn't do.",
    )
    HYBRID_CANDIDATE_MULTIPLIER: int = Field(
        default=4,
        description="Each retrieval method (dense, keyword) over-fetches "
        "top_k * this many candidates before fusion/reranking narrows back "
        "down to top_k — fusion and reranking need a wider candidate pool "
        "to actually change the final ranking, not just re-sort a list "
        "that's already been cut down to the final size.",
    )
    RRF_K: int = Field(
        default=60,
        description="Reciprocal Rank Fusion damping constant — see "
        "app/retrieval/fusion.py for what it controls. 60 is the value "
        "used in the original RRF paper and most production systems that "
        "cite it; not tuned specifically for this project.",
    )

    # --- Observability (wired up in Phase 16) ---
    LANGFUSE_PUBLIC_KEY: str | None = None
    LANGFUSE_SECRET_KEY: str | None = None

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def max_upload_size_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings accessor.

    Why lru_cache: Settings() re-reads and re-validates the environment every
    time it's constructed. We want that to happen exactly once per process,
    not once per request. FastAPI's dependency-injection system will call
    this via `Depends(get_settings)` later, and lru_cache makes every call
    after the first one free.
    """
    return Settings()


settings = get_settings()
