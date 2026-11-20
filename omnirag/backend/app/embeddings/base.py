"""
EmbeddingProvider abstraction.

Why an interface instead of calling one vendor's SDK directly from
services/tasks: the spec requires the embedding backend to be swappable via
configuration, not hard-coded. Concretely, that means:
  - `app/services` and `app/workers/tasks.py` depend on this interface, never
    on `openai` or any specific vendor's client
  - swapping providers is a one-line config change (`EMBEDDING_PROVIDER=...`),
    not a code change scattered across the codebase
  - a provider can be swapped in tests (e.g. a fake that returns fixed
    vectors) without touching business logic

This is the standard Strategy pattern: the interface defines *what* an
embedding provider does, each implementation defines *how*.
"""

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Length of every vector this provider returns. Callers (and the
        DB schema) need this to stay constant for a given provider — mixing
        vectors of different dimensions in one vector index is invalid."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier recorded alongside each stored embedding, so a later
        provider/model change doesn't silently mix incompatible vectors —
        see DocumentChunk.embedding_model."""
        ...

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Returns one embedding vector per input text, same order as input.
        Implementations should batch internally where the underlying API
        supports it, rather than requiring callers to chunk their own calls."""
        ...
