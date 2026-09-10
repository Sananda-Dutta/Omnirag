"""
Anthropic LLM provider.

A real, complete implementation — not a stub. It follows Anthropic's
documented Messages API exactly. This sandboxed environment can reach
api.anthropic.com over the network (unlike api.openai.com), but has no
ANTHROPIC_API_KEY available to actually call it with, so — same as
OpenAIEmbeddingProvider in Phase 5 — this is tested here only against a
mocked client. Said plainly: "follows the documented API" and "verified
against the live service" are different claims, and only the first one is
true right now.
"""

from anthropic import AsyncAnthropic

from app.llm.base import ConversationTurn, LLMProvider, LLMResponse


class AnthropicLLMProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, max_tokens: int = 1024):
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self, *, system: str, history: list[ConversationTurn], user_message: str
    ) -> LLMResponse:
        # Anthropic's Messages API takes conversation turns natively as a
        # role-alternating list — history maps directly onto it, no string
        # embedding needed.
        messages = [{"role": turn.role, "content": turn.content} for turn in history]
        messages.append({"role": "user", "content": user_message})

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=messages,
        )
        # response.content is a list of content blocks (text, tool_use, ...);
        # a plain-text request produces exactly one text block.
        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMResponse(
            text=text,
            model=self._model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
