"""
Tests for app/rag/memory.py's window/summarization logic.

These are pure-logic tests against in-memory Message objects (constructed
directly, never persisted) and a lightweight fake LLM provider — no
database or network needed, so unlike most of this project's tests, these
were written under the same environment constraints as the rest of Phase 10
but don't depend on infrastructure this session couldn't stand up. Still
worth running for real, but the logic itself is exercised more directly
here than in the (unverified) end-to-end tests elsewhere in Phase 10.
"""

import pytest

from app.llm.base import ConversationTurn, LLMProvider, LLMResponse
from app.llm.local_extractive import LocalExtractiveLLMProvider
from app.models.conversation import Conversation
from app.models.message import Message
from app.rag.memory import messages_to_turns, summarize_aged_out_messages, update_conversation_memory


class FakeLLMProvider(LLMProvider):
    """A minimal real implementation of LLMProvider for testing memory.py
    in isolation from any real network provider — returns a fixed,
    inspectable response rather than mocking at the HTTP layer, since
    there's no HTTP layer here to mock."""

    def __init__(self):
        self.calls: list[dict] = []

    @property
    def model_name(self) -> str:
        return "fake-test-model"

    async def generate(self, *, system, history, user_message) -> LLMResponse:
        self.calls.append({"system": system, "history": history, "user_message": user_message})
        return LLMResponse(text=f"SUMMARY_OF[{user_message[:30]}]", model=self.model_name)


def _msg(role: str, content: str) -> Message:
    return Message(role=role, content=content)


def test_messages_to_turns_preserves_role_and_content():
    messages = [_msg("user", "hi"), _msg("assistant", "hello")]
    turns = messages_to_turns(messages)
    assert turns == [
        ConversationTurn(role="user", content="hi"),
        ConversationTurn(role="assistant", content="hello"),
    ]


@pytest.mark.asyncio
async def test_no_summarization_when_under_window():
    conversation = Conversation(owner_id=None, summarized_message_count=0)
    messages = [_msg("user", "one"), _msg("assistant", "two")]
    provider = FakeLLMProvider()

    history = await update_conversation_memory(provider, conversation, messages)

    assert conversation.summary is None
    assert conversation.summarized_message_count == 0
    assert len(history) == 2
    assert provider.calls == []  # no summarization call made


@pytest.mark.asyncio
async def test_summarization_triggers_once_window_exceeded():
    conversation = Conversation(owner_id=None, summarized_message_count=0)
    # window default is 6; 8 messages means 2 should age out
    messages = [_msg("user", f"msg{i}") for i in range(8)]
    provider = FakeLLMProvider()

    history = await update_conversation_memory(provider, conversation, messages)

    assert conversation.summarized_message_count == 2
    assert conversation.summary is not None
    assert len(history) == 6  # the window
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_summarization_does_not_repeat_for_already_summarized_messages():
    """The core correctness property of the incremental design: calling
    update_conversation_memory twice with the SAME message list should only
    summarize once, not repeat the summarization call."""
    conversation = Conversation(owner_id=None, summarized_message_count=0)
    messages = [_msg("user", f"msg{i}") for i in range(8)]
    provider = FakeLLMProvider()

    await update_conversation_memory(provider, conversation, messages)
    assert len(provider.calls) == 1

    # Same messages, called again (simulating a second turn where nothing
    # new has aged out yet because only 8 messages still exist)
    await update_conversation_memory(provider, conversation, messages)
    assert len(provider.calls) == 1  # still just the one call — no re-summarization


@pytest.mark.asyncio
async def test_summarization_advances_incrementally_as_conversation_grows():
    conversation = Conversation(owner_id=None, summarized_message_count=0)
    provider = FakeLLMProvider()

    messages = [_msg("user", f"msg{i}") for i in range(8)]
    await update_conversation_memory(provider, conversation, messages)
    assert conversation.summarized_message_count == 2
    first_call_input = provider.calls[0]["user_message"]

    # Two more messages arrive (10 total) — only messages 2-3 (the newly
    # aged-out ones) should be summarized this time, not messages 0-3 again.
    messages = messages + [_msg("user", "msg8"), _msg("assistant", "msg9")]
    await update_conversation_memory(provider, conversation, messages)
    assert conversation.summarized_message_count == 4
    assert len(provider.calls) == 2
    second_call_input = provider.calls[1]["user_message"]
    assert second_call_input != first_call_input


@pytest.mark.asyncio
async def test_local_provider_uses_truncated_concatenation_not_llm_call():
    provider = LocalExtractiveLLMProvider()
    aged_out = [_msg("user", "a" * 300), _msg("assistant", "b" * 300)]

    summary = await summarize_aged_out_messages(provider, None, aged_out)

    assert "not summarized by an LLM" in summary
    # truncated to 150 chars per message, not the full 300
    assert "a" * 200 not in summary


@pytest.mark.asyncio
async def test_real_provider_path_calls_generate_with_existing_summary_included():
    provider = FakeLLMProvider()
    aged_out = [_msg("user", "What is RAG?")]

    summary = await summarize_aged_out_messages(provider, "Prior summary text.", aged_out)

    assert len(provider.calls) == 1
    assert "Prior summary text." in provider.calls[0]["user_message"]
    assert "What is RAG?" in provider.calls[0]["user_message"]
    assert summary.startswith("SUMMARY_OF[")


@pytest.mark.asyncio
async def test_empty_aged_out_returns_existing_summary_unchanged():
    provider = FakeLLMProvider()
    result = await summarize_aged_out_messages(provider, "unchanged", [])
    assert result == "unchanged"
    assert provider.calls == []
