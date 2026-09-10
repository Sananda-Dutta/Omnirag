"""
LLMProvider abstraction.

Same reasoning as EmbeddingProvider and VectorStore: callers (the RAG
pipeline) depend on this interface, never on `anthropic` or `openai`
directly. Switching the generation backend is a config change
(`LLM_PROVIDER=...`), not a rewrite.

Phase 10 change: `generate()` now takes structured `history` (a list of
prior turns) instead of just a single `user_message`. This is deliberately
NOT done by stuffing prior turns as text into the `system` string — that's
exactly the pattern that caused a real bug in Phase 7 (the grounded
prompt's own instructions happened to contain the literal `<context>`
delimiter substring, which broke the regex parsing it back out). Passing
history as actual structured data instead of embedded text avoids that
entire category of collision, and matches how Anthropic's and OpenAI's
APIs natively represent multi-turn conversations anyway — no translation
layer needed for the two real providers, only for the local fallback (see
local_extractive.py's docstring for why it can't really use history).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ConversationTurn:
    role: Literal["user", "assistant"]
    content: str


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
    async def generate(
        self, *, system: str, history: list[ConversationTurn], user_message: str
    ) -> LLMResponse:
        """`history` is prior turns in this conversation, oldest first, NOT
        including `user_message` itself. Pass an empty list for a
        single-turn (no-history) call — existing Phase 7 behavior."""
        ...
