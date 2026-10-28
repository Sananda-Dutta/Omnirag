"""
OpenAI-compatible LLM provider (chat completions).

Works with OpenAI itself and with any service that speaks the same API —
set OPENAI_BASE_URL to use one. That's how a FREE hosted model plugs into
this project without a new provider class: Groq, Google Gemini, OpenRouter
and a local Ollama all expose an OpenAI-compatible endpoint.

Verified here against a fake OpenAI-compatible HTTP server (the real wire
format, real `openai` SDK, real base_url handling — see tests/test_llm.py).
NOT verified against any live third-party service from this sandbox, which
can't reach them; the first real question you ask is the live test, and
`python -m app.llm.check` runs exactly that without the UI.
"""

import openai
from openai import AsyncOpenAI

from app.llm.base import ConversationTurn, LLMProvider, LLMProviderError, LLMResponse


def _describe_error(exc: Exception) -> str:
    """One readable line for the user. Built only from the status code and
    the provider's own message — never from the request, so the API key
    can't leak into it."""
    if isinstance(exc, openai.APIStatusError):
        hint = {
            401: "the API key was rejected — check OPENAI_API_KEY",
            403: "access denied — check the key's permissions",
            404: "model or endpoint not found — check OPENAI_CHAT_MODEL and OPENAI_BASE_URL",
            429: "rate limit or quota reached — wait a minute and retry",
        }.get(exc.status_code, "the provider returned an error")
        return f"LLM provider error ({exc.status_code}): {hint}. Provider said: {exc.message}"
    if isinstance(exc, openai.APIConnectionError):
        return (
            "Could not reach the LLM provider — check OPENAI_BASE_URL and your "
            "internet connection."
        )
    return f"LLM provider error: {exc}"


class OpenAILLMProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        *,
        system: str,
        user_message: str,
        history: list[ConversationTurn] | None = None,
    ) -> LLMResponse:
        history = history or []
        # OpenAI's Chat Completions API takes turns as a flat
        # role-alternating list (system first) — history maps directly on.
        messages = [{"role": "system", "content": system}]
        messages += [{"role": turn.role, "content": turn.content} for turn in history]
        messages.append({"role": "user", "content": user_message})

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
            )
        except openai.OpenAIError as exc:
            raise LLMProviderError(_describe_error(exc)) from exc

        choice = response.choices[0]
        text = choice.message.content or ""
        if not text.strip():
            raise LLMProviderError(
                "The model returned an empty answer "
                f"(finish_reason={getattr(choice, 'finish_reason', None)!r}). "
                "Try again, or try a different OPENAI_CHAT_MODEL."
            )
        usage = response.usage
        return LLMResponse(
            text=text,
            model=self._model,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )
