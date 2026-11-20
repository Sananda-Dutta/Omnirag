"""
Embedding provider factory.

This is the one place in the codebase allowed to know about concrete
provider classes (LocalHashingEmbeddingProvider, OpenAIEmbeddingProvider,
...). Every caller — the Celery task, any future service — depends on
`EmbeddingProvider` (the interface) and gets a configured instance from
`get_embedding_provider()`. Switching providers is changing
`EMBEDDING_PROVIDER` in `.env`, not editing code.
"""

from functools import lru_cache

from app.core.config import Settings, settings
from app.embeddings.base import EmbeddingProvider
from app.embeddings.local_hashing import LocalHashingEmbeddingProvider
from app.embeddings.openai_provider import OpenAIEmbeddingProvider


def _build_provider(cfg: Settings) -> EmbeddingProvider:
    if cfg.EMBEDDING_PROVIDER == "local":
        return LocalHashingEmbeddingProvider(dimension=cfg.EMBEDDING_DIMENSION)
    if cfg.EMBEDDING_PROVIDER == "openai":
        return OpenAIEmbeddingProvider(
            api_key=cfg.OPENAI_API_KEY or "",
            model=cfg.OPENAI_EMBEDDING_MODEL,
            dimension=cfg.EMBEDDING_DIMENSION,
        )
    # Unreachable given Settings.EMBEDDING_PROVIDER's Literal type (pydantic
    # rejects any other value at startup) — kept as a defensive guard so a
    # future new provider value can't silently fall through to None.
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {cfg.EMBEDDING_PROVIDER!r}")


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    return _build_provider(settings)
