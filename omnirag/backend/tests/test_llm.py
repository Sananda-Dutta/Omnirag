"""
LLMProvider tests.

LocalExtractiveLLMProvider is tested for real — no mocking, it needs none.
AnthropicLLMProvider and OpenAILLMProvider are tested against mocked
clients, for the same reason as OpenAIEmbeddingProvider in Phase 5: this
sandbox has no usable API key for either service.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.llm.anthropic_provider import AnthropicLLMProvider
from app.llm.base import ConversationTurn
from app.llm.factory import _build_provider
from app.llm.local_extractive import LocalExtractiveLLMProvider
from app.llm.openai_provider import OpenAILLMProvider
from app.rag.prompts import build_system_prompt

# --- LocalExtractiveLLMProvider: tested for real ---


@pytest.mark.asyncio
async def test_local_extractive_extracts_relevant_sentence():
    provider = LocalExtractiveLLMProvider()
    context = (
        "Gradient descent is an optimization algorithm. "
        "It minimizes a loss function iteratively. "
        "Bread should be baked at 350 degrees for forty minutes."
    )
    system = build_system_prompt(context)

    response = await provider.generate(system=system, history=[], user_message="What does gradient descent do?")

    assert "gradient descent" in response.text.lower() or "minimizes" in response.text.lower()
    assert "bread" not in response.text.lower()
    assert response.model == "local-extractive"


@pytest.mark.asyncio
async def test_local_extractive_handles_empty_context():
    provider = LocalExtractiveLLMProvider()
    system = build_system_prompt("")
    response = await provider.generate(system=system, history=[], user_message="Anything?")
    assert "no context" in response.text.lower()


@pytest.mark.asyncio
async def test_local_extractive_handles_no_overlap():
    provider = LocalExtractiveLLMProvider()
    system = build_system_prompt("The weather today is sunny with a light breeze.")
    response = await provider.generate(system=system, history=[], user_message="What is quantum entanglement?")
    assert "nothing" in response.text.lower() or "overlap" in response.text.lower()


@pytest.mark.asyncio
async def test_local_extractive_respects_max_sentences():
    provider = LocalExtractiveLLMProvider(max_sentences=1)
    context = "Machine learning models learn patterns. Machine learning models generalize. Machine learning models predict."
    system = build_system_prompt(context)
    response = await provider.generate(system=system, history=[], user_message="Tell me about machine learning models")
    # Body (excluding the disclaimer line) should be a single sentence.
    body = response.text.split("\n\n", 1)[1]
    assert body.count(". ") <= 1


# --- AnthropicLLMProvider: tested against a mocked client ---


@pytest.mark.asyncio
async def test_anthropic_provider_parses_response():
    provider = AnthropicLLMProvider(api_key="fake-key-for-test", model="claude-sonnet-4-5-20250929")

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "This is the generated answer."
    mock_response = MagicMock()
    mock_response.content = [text_block]
    mock_response.usage.input_tokens = 120
    mock_response.usage.output_tokens = 40
    provider._client.messages.create = AsyncMock(return_value=mock_response)

    result = await provider.generate(system="a system prompt", history=[], user_message="a question")

    assert result.text == "This is the generated answer."
    assert result.input_tokens == 120
    assert result.output_tokens == 40
    provider._client.messages.create.assert_awaited_once_with(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        system="a system prompt",
        messages=[{"role": "user", "content": "a question"}],
    )


def test_anthropic_provider_requires_api_key():
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicLLMProvider(api_key="", model="claude-sonnet-4-5-20250929")


@pytest.mark.asyncio
async def test_anthropic_provider_includes_history_in_messages():
    """Phase 10: proves history is actually threaded into the request, not
    just accepted and silently dropped — the risk with any interface change
    that adds a parameter without a caller-visible effect if unused."""
    provider = AnthropicLLMProvider(api_key="fake-key-for-test", model="claude-sonnet-4-5-20250929")

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Following up on that."
    mock_response = MagicMock()
    mock_response.content = [text_block]
    mock_response.usage.input_tokens = 50
    mock_response.usage.output_tokens = 10
    provider._client.messages.create = AsyncMock(return_value=mock_response)

    history = [
        ConversationTurn(role="user", content="What is RAG?"),
        ConversationTurn(role="assistant", content="RAG combines retrieval with generation."),
    ]
    await provider.generate(system="a system prompt", history=history, user_message="Say more about that.")

    provider._client.messages.create.assert_awaited_once_with(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        system="a system prompt",
        messages=[
            {"role": "user", "content": "What is RAG?"},
            {"role": "assistant", "content": "RAG combines retrieval with generation."},
            {"role": "user", "content": "Say more about that."},
        ],
    )


# --- OpenAILLMProvider: tested against a mocked client ---


@pytest.mark.asyncio
async def test_openai_llm_provider_parses_response():
    provider = OpenAILLMProvider(api_key="fake-key-for-test", model="gpt-4o-mini")

    message = MagicMock()
    message.content = "This is the generated answer."
    choice = MagicMock()
    choice.message = message
    mock_response = MagicMock()
    mock_response.choices = [choice]
    mock_response.usage.prompt_tokens = 90
    mock_response.usage.completion_tokens = 30
    provider._client.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await provider.generate(system="a system prompt", history=[], user_message="a question")

    assert result.text == "This is the generated answer."
    assert result.input_tokens == 90
    assert result.output_tokens == 30


@pytest.mark.asyncio
async def test_openai_provider_includes_history_in_messages():
    provider = OpenAILLMProvider(api_key="fake-key-for-test", model="gpt-4o-mini")

    message = MagicMock()
    message.content = "Following up."
    choice = MagicMock()
    choice.message = message
    mock_response = MagicMock()
    mock_response.choices = [choice]
    mock_response.usage = None
    provider._client.chat.completions.create = AsyncMock(return_value=mock_response)

    history = [ConversationTurn(role="user", content="What is RAG?")]
    await provider.generate(system="sys", history=history, user_message="Say more.")

    provider._client.chat.completions.create.assert_awaited_once_with(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "What is RAG?"},
            {"role": "user", "content": "Say more."},
        ],
    )


def test_openai_llm_provider_requires_api_key():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAILLMProvider(api_key="", model="gpt-4o-mini")


@pytest.mark.asyncio
async def test_local_extractive_ignores_history_without_erroring():
    """History is accepted for interface compliance (see module docstring)
    but has no scoring effect — this just proves passing a non-empty
    history doesn't crash the local provider."""
    provider = LocalExtractiveLLMProvider()
    system = build_system_prompt("Gradient descent minimizes a loss function.")
    history = [ConversationTurn(role="user", content="Something unrelated entirely.")]
    response = await provider.generate(system=system, history=history, user_message="What does gradient descent minimize?")
    assert "loss function" in response.text.lower() or "minimizes" in response.text.lower()


# --- Factory ---


def test_factory_builds_local_provider_by_default():
    from app.core.config import Settings

    cfg = Settings(LLM_PROVIDER="local")
    provider = _build_provider(cfg)
    assert isinstance(provider, LocalExtractiveLLMProvider)


def test_factory_builds_anthropic_provider_when_configured():
    from app.core.config import Settings

    cfg = Settings(LLM_PROVIDER="anthropic", ANTHROPIC_API_KEY="fake-key-for-test")
    provider = _build_provider(cfg)
    assert isinstance(provider, AnthropicLLMProvider)
