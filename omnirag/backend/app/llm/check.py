"""
Quick check that your configured LLM works, without the UI:

    cd backend && python -m app.llm.check

Reads backend/.env exactly like the app does, sends one tiny question, and
prints the answer — or the same readable error the chat UI would show.
"""

import asyncio
import sys

from app.core.config import settings
from app.llm.base import LLMProviderError
from app.llm.factory import get_llm_provider


async def main() -> int:
    print(f"LLM_PROVIDER={settings.LLM_PROVIDER}")
    if settings.LLM_PROVIDER == "openai":
        print(f"OPENAI_BASE_URL={settings.OPENAI_BASE_URL or '(default: api.openai.com)'}")
        print(f"OPENAI_CHAT_MODEL={settings.OPENAI_CHAT_MODEL}")
    try:
        provider = get_llm_provider()
        result = await provider.generate(
            system="You are a helpful assistant. Reply in one short sentence.",
            history=[],
            user_message="Say hello and name the model you are.",
        )
    except (LLMProviderError, ValueError) as exc:
        print(f"\nFAILED: {exc}")
        return 1
    print(f"\nOK ({result.model}): {result.text}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
