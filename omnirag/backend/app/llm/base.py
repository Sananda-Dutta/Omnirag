"""
LLMProvider abstraction.

Same reasoning as EmbeddingProvider and VectorStore: callers (the RAG
pipeline) depend on this interface, never on `anthropic` or `openai`
directly. Switching the generation backend is a config change
(`LLM_PROVIDER=...`), not a rewrite.

The interface is deliberately narrow — a system prompt plus a single user
message in, one response out. No streaming (Phase 14), no multi-turn
history (Phase 10), no tool calling (Phase 13). Building those in now, before
anything in this codebase actually needs them, is exactly the kind of
speculative complexity this project's own ground rules ("no unnecessary
abstractions") warn against — each gets added in the phase that introduces
a real caller for it.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMProvider(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    async def generate(self, *, system: str, user_message: str) -> LLMResponse: ...
