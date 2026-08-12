"""
Embedding provider tests.

LocalHashingEmbeddingProvider is tested completely for real — no mocking,
because it needs none (no network, no external service). It's exercised for
its actual documented properties: deterministic, correctly-dimensioned,
L2-normalized, and — the property that actually matters for retrieval —
texts sharing vocabulary embed closer together than texts that don't.

OpenAIEmbeddingProvider is tested against a mocked AsyncOpenAI client. This
is a deliberate exception to "don't mock" elsewhere in this project: it's an
external paid API this sandboxed environment cannot reach (no egress to
api.openai.com), so there is no way to test it for real here. The mock
proves the request is built correctly and the response is parsed and
ordered correctly — it does NOT prove OpenAI's API behaves as documented.
"""

import math
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.embeddings.factory import _build_provider
from app.embeddings.local_hashing import LocalHashingEmbeddingProvider
from app.embeddings.openai_provider import OpenAIEmbeddingProvider


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


# --- LocalHashingEmbeddingProvider: tested for real ---


@pytest.mark.asyncio
async def test_local_provider_returns_correct_dimension():
    provider = LocalHashingEmbeddingProvider(dimension=128)
    [vector] = await provider.embed_texts(["gradient descent"])
    assert len(vector) == 128


@pytest.mark.asyncio
async def test_local_provider_is_deterministic():
    provider = LocalHashingEmbeddingProvider(dimension=128)
    [v1] = await provider.embed_texts(["retrieval augmented generation"])
    [v2] = await provider.embed_texts(["retrieval augmented generation"])
    assert v1 == v2


@pytest.mark.asyncio
async def test_local_provider_vectors_are_l2_normalized():
    provider = LocalHashingEmbeddingProvider(dimension=128)
    [vector] = await provider.embed_texts(["a reasonably long sentence about databases"])
    norm = math.sqrt(sum(v * v for v in vector))
    assert abs(norm - 1.0) < 1e-6


@pytest.mark.asyncio
async def test_local_provider_empty_string_returns_zero_vector_without_crashing():
    provider = LocalHashingEmbeddingProvider(dimension=64)
    [vector] = await provider.embed_texts([""])
    assert vector == [0.0] * 64


@pytest.mark.asyncio
async def test_local_provider_preserves_input_order():
    provider = LocalHashingEmbeddingProvider(dimension=64)
    vectors = await provider.embed_texts(["alpha text here", "beta text here", "gamma text here"])
    assert len(vectors) == 3
    assert vectors[0] != vectors[1] != vectors[2]


@pytest.mark.asyncio
async def test_local_provider_similar_texts_are_more_similar_than_dissimilar_ones():
    # The property that actually matters for retrieval: shared vocabulary
    # should pull cosine similarity up. This is the "real, if crude, bag of
    # words behavior" claim made in local_hashing.py's module docstring —
    # this test is what backs that claim rather than just asserting it.
    provider = LocalHashingEmbeddingProvider(dimension=256)
    a, b, c = await provider.embed_texts(
        [
            "Gradient descent minimizes a loss function during training.",
            "The loss function is minimized using gradient descent.",
            "The recipe calls for two cups of flour and a pinch of salt.",
        ]
    )
    sim_related = _cosine_similarity(a, b)
    sim_unrelated = _cosine_similarity(a, c)
    assert sim_related > sim_unrelated


# --- OpenAIEmbeddingProvider: tested against a mocked HTTP boundary ---


def _mock_embedding_item(index: int, vector: list[float]) -> MagicMock:
    item = MagicMock()
    item.index = index
    item.embedding = vector
    return item


@pytest.mark.asyncio
async def test_openai_provider_parses_and_orders_response(monkeypatch):
    provider = OpenAIEmbeddingProvider(api_key="fake-key-for-test", model="text-embedding-3-small", dimension=3)

    # Simulate the API returning results out of order — the provider must
    # sort by `index`, not trust response order.
    mock_response = MagicMock()
    mock_response.data = [
        _mock_embedding_item(index=1, vector=[0.4, 0.5, 0.6]),
        _mock_embedding_item(index=0, vector=[0.1, 0.2, 0.3]),
    ]
    provider._client.embeddings.create = AsyncMock(return_value=mock_response)

    result = await provider.embed_texts(["first text", "second text"])

    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    provider._client.embeddings.create.assert_awaited_once_with(
        model="text-embedding-3-small", input=["first text", "second text"]
    )


@pytest.mark.asyncio
async def test_openai_provider_empty_input_skips_the_api_call():
    provider = OpenAIEmbeddingProvider(api_key="fake-key-for-test", model="text-embedding-3-small", dimension=3)
    provider._client.embeddings.create = AsyncMock()

    result = await provider.embed_texts([])

    assert result == []
    provider._client.embeddings.create.assert_not_called()


def test_openai_provider_requires_api_key():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIEmbeddingProvider(api_key="", model="text-embedding-3-small", dimension=1536)


# --- Factory ---


def test_factory_builds_local_provider_by_default(monkeypatch):
    from app.core.config import Settings

    cfg = Settings(EMBEDDING_PROVIDER="local", EMBEDDING_DIMENSION=256)
    provider = _build_provider(cfg)
    assert isinstance(provider, LocalHashingEmbeddingProvider)
    assert provider.dimension == 256


def test_factory_builds_openai_provider_when_configured():
    from app.core.config import Settings

    cfg = Settings(
        EMBEDDING_PROVIDER="openai",
        EMBEDDING_DIMENSION=1536,
        OPENAI_API_KEY="fake-key-for-test",
        OPENAI_EMBEDDING_MODEL="text-embedding-3-small",
    )
    provider = _build_provider(cfg)
    assert isinstance(provider, OpenAIEmbeddingProvider)
    assert provider.model_name == "text-embedding-3-small"
