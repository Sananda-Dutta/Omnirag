"""
LLMProvider factory. Same role as app/embeddings/factory.py and
app/retrieval/factory.py: the only place that knows about concrete
implementations.
"""

from functools import lru_cache

from app.core.config import Settings, settings
from app.llm.anthropic_provider import AnthropicLLMProvider
from app.llm.base import LLMProvider
from app.llm.local_extractive import LocalExtractiveLLMProvider
from app.llm.openai_provider import OpenAILLMProvider


def _build_provider(cfg: Settings) -> LLMProvider:
    if cfg.LLM_PROVIDER == "local":
        return LocalExtractiveLLMProvider()
    if cfg.LLM_PROVIDER == "anthropic":
        return AnthropicLLMProvider(api_key=cfg.ANTHROPIC_API_KEY or "", model=cfg.ANTHROPIC_MODEL)
    if cfg.LLM_PROVIDER == "openai":
        return OpenAILLMProvider(api_key=cfg.OPENAI_API_KEY or "", model=cfg.OPENAI_CHAT_MODEL)
    # Unreachable given Settings.LLM_PROVIDER's Literal type — defensive
    # guard, same reasoning as app/embeddings/factory.py.
    raise ValueError(f"Unknown LLM_PROVIDER: {cfg.LLM_PROVIDER!r}")


@lru_cache
def get_llm_provider() -> LLMProvider:
    return _build_provider(settings)
