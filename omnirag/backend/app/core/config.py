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

    # --- Redis / background jobs (used starting Phase 4) ---
    REDIS_URL: str = Field(default="redis://localhost:6379/0")

    # --- File storage / ingestion (Phase 4) ---
    UPLOAD_DIR: str = Field(default="./uploads")
    MAX_UPLOAD_SIZE_MB: int = Field(default=25)
    ALLOWED_UPLOAD_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx", ".txt", ".md")

    # --- Auth (wired up in Phase 3) ---
    SECRET_KEY: str = Field(
        default="CHANGE_ME_IN_PRODUCTION",
        description="Used to sign JWTs. Must be overridden outside local dev.",
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # --- LLM / embedding providers (wired up in Phase 5/7) ---
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None

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
