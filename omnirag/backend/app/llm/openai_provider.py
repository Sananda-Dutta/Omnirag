"""
OpenAI LLM provider (chat completion). Same status as
app/embeddings/openai_provider.py and app/llm/anthropic_provider.py: a real
implementation of OpenAI's documented Chat Completions API, tested here only
against a mocked client since this sandbox has no OPENAI_API_KEY or network
access to api.openai.com.
"""

from openai import AsyncOpenAI

from app.llm.base import LLMProvider, LLMResponse


class OpenAILLMProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(self, *, system: str, user_message: str) -> LLMResponse:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
        )
        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            text=choice.message.content or "",
            model=self._model,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )
