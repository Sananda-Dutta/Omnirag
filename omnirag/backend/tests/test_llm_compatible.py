"""
Free-LLM support: OpenAILLMProvider against any OpenAI-compatible endpoint.

Uses the REAL `openai` SDK talking over HTTP to a fake OpenAI-compatible
server (httpx.MockTransport), so these tests exercise the actual request
URL, headers, JSON body, and error mapping — not a mocked SDK method.
What they can't prove: that Groq/Gemini/OpenRouter accept it today. That's
what `python -m app.llm.check` is for.
"""

import json

import httpx
import openai
import pytest

from app.llm import openai_provider
from app.llm.base import ConversationTurn, LLMProviderError
from app.llm.openai_provider import OpenAILLMProvider


def _install_fake_server(monkeypatch, handler):
    """Make every AsyncOpenAI the provider builds talk to `handler`."""
    real = openai.AsyncOpenAI

    def factory(**kwargs):
        return real(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    monkeypatch.setattr(openai_provider, "AsyncOpenAI", factory)


def _ok_body(text="Here are the projects: A and B."):
    return {
        "id": "x", "object": "chat.completion", "created": 0, "model": "fake",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


@pytest.mark.asyncio
async def test_base_url_model_key_and_history_reach_the_server(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_ok_body())

    _install_fake_server(monkeypatch, handler)
    provider = OpenAILLMProvider(
        api_key="test-key", model="llama-3.3-70b-versatile",
        base_url="https://api.groq.com/openai/v1",
    )
    result = await provider.generate(
        system="sys", user_message="and now?",
        history=[ConversationTurn(role="user", content="first")],
    )

    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["model"] == "llama-3.3-70b-versatile"
    assert [m["role"] for m in seen["body"]["messages"]] == ["system", "user", "user"]
    assert result.text == "Here are the projects: A and B."
    assert (result.input_tokens, result.output_tokens) == (11, 7)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,expected",
    [(401, "key was rejected"), (404, "OPENAI_CHAT_MODEL"), (429, "rate limit")],
)
async def test_http_errors_become_readable_llm_provider_errors(monkeypatch, status, expected):
    def handler(request):
        return httpx.Response(status, json={"error": {"message": "upstream says no"}})

    _install_fake_server(monkeypatch, handler)
    provider = OpenAILLMProvider(api_key="test-key-SECRET", model="m", base_url="http://x/v1")

    with pytest.raises(LLMProviderError) as exc_info:
        await provider.generate(system="s", user_message="q", history=[])

    message = str(exc_info.value)
    assert str(status) in message
    assert expected in message
    assert "upstream says no" in message
    assert "test-key-SECRET" not in message  # the key must never leak into the error


@pytest.mark.asyncio
async def test_unreachable_server_is_a_readable_error(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("boom")

    _install_fake_server(monkeypatch, handler)
    provider = OpenAILLMProvider(api_key="k", model="m", base_url="http://nowhere/v1")

    with pytest.raises(LLMProviderError, match="Could not reach"):
        await provider.generate(system="s", user_message="q", history=[])


@pytest.mark.asyncio
async def test_empty_model_reply_is_an_error_not_a_blank_answer(monkeypatch):
    _install_fake_server(monkeypatch, lambda r: httpx.Response(200, json=_ok_body(text="")))
    provider = OpenAILLMProvider(api_key="k", model="m", base_url="http://x/v1")

    with pytest.raises(LLMProviderError, match="empty answer"):
        await provider.generate(system="s", user_message="q", history=[])


def test_factory_passes_base_url_through():
    from app.core.config import Settings
    from app.llm.factory import _build_provider

    cfg = Settings(
        LLM_PROVIDER="openai", OPENAI_API_KEY="k", OPENAI_CHAT_MODEL="m",
        OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    provider = _build_provider(cfg)
    assert str(provider._client.base_url).startswith("https://generativelanguage.googleapis.com")


def test_base_url_is_read_from_env_file_not_just_the_shell(tmp_path, monkeypatch):
    """The bug that made the first Gemini attempt fragile: .env values are
    read by the app's own settings object, which previously had no such field."""
    from app.core.config import Settings

    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_BASE_URL=http://localhost:11434/v1\n")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert Settings(_env_file=str(env_file)).OPENAI_BASE_URL == "http://localhost:11434/v1"
