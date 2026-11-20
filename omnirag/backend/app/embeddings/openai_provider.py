"""
OpenAI embedding provider.

This is a real, complete implementation of the interface — not a stub. It's
tested in this repo only against a mocked HTTP layer (see
tests/test_embeddings.py), because this sandboxed environment's network
egress allowlist doesn't include api.openai.com — there's no way to prove
this works against the live API from here. Said plainly rather than glossed
over: if you run this with a real OPENAI_API_KEY outside this sandbox, it
should work as written (it follows OpenAI's documented embeddings API
exactly), but it has not been exercised against the real service as part of
building this project.
"""

from openai import AsyncOpenAI

from app.embeddings.base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model: str, dimension: int):
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai"
            )
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        # OpenAI's embeddings endpoint accepts a batch (list) input directly
        # in one request — no manual batching needed at this scale. A
        # production system embedding very large corpora would still want
        # request-size-aware batching and retry/backoff on rate limits;
        # noted here as a Phase 21 (cost/performance optimization) concern
        # rather than solved speculatively now.
        response = await self._client.embeddings.create(model=self._model, input=texts)

        # Defensive: the API documents results in input order, but sorting
        # by the returned index costs nothing and removes the assumption.
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]
