"""
Conversation memory: recent-message window + summarization for older turns.

Why not just send the whole conversation every time: token cost scales
with history length, and past a certain point a long verbatim history
mostly adds noise (old, now-irrelevant turns) rather than useful context.
The standard mitigation — and what's implemented here — is a hybrid: the
most recent CONVERSATION_RECENT_MESSAGE_WINDOW messages go in verbatim
(recent turns are usually the most relevant to a follow-up question),
while everything older is compacted into a running summary instead of
being dropped entirely or sent in full.

Why incremental, not "resummarize everything older on every turn":
`Conversation.summarized_message_count` tracks how many of the oldest
messages have already been folded into `summary`. Each turn only
summarizes the messages that just aged out of the window since last time
(recomputing "everything older than the window" from scratch every turn
would re-summarize the same already-summarized messages repeatedly — an
O(n^2) token cost over a long conversation's lifetime instead of O(n)).

Why summarization is provider-aware: see summarize_aged_out_messages's
docstring — the short version is that LocalExtractiveLLMProvider has no
mechanism for actually producing an abstractive summary, so pretending it
does would be exactly the kind of fake implementation this project avoids.

Cost note (explicitly requested by the original spec): with the default
window of 6, a conversation costs one extra summarization LLM call every 6
messages, each summarizing only the newly-aged-out slice — not the whole
conversation restated from scratch each time.

Explicitly out of scope for this phase: "relevant historical message
retrieval" (semantically searching further back than the summary covers).
That would mean embedding messages into the vector store too, which is a
real, separate feature — worth doing once there's a concrete case the
recent-window-plus-summary approach falls short on, not spun up now on the
assumption it will be needed.
"""

from app.core.config import settings
from app.llm.base import ConversationTurn, LLMProvider
from app.models.conversation import Conversation
from app.models.message import Message

_SUMMARIZATION_SYSTEM_PROMPT = (
    "Summarize the following conversation excerpt in 2-4 sentences, "
    "preserving concrete facts, names, and figures a reader would need to "
    "follow later messages. Do not add commentary, and do not mention that "
    "you are summarizing — just produce the summary text itself."
)


def messages_to_turns(messages: list[Message]) -> list[ConversationTurn]:
    return [ConversationTurn(role=m.role, content=m.content) for m in messages]


async def summarize_aged_out_messages(
    provider: LLMProvider, existing_summary: str | None, aged_out: list[Message]
) -> str:
    """Folds `aged_out` messages into `existing_summary`, producing a new
    running summary.

    Provider-aware: LocalExtractiveLLMProvider is built to extract relevant
    sentences from retrieved context against a specific query — it has no
    mechanism for open-ended abstractive summarization of arbitrary prior
    conversation. Calling its normal generate() for "summarization" would
    just run lexical-overlap extraction against whatever got passed in,
    which isn't a summary, it just superficially resembles one. So for the
    local provider, aged-out messages are truncated and concatenated
    instead — clearly labeled as such, not claimed to be an LLM summary
    that didn't actually happen. Anthropic/OpenAI get a real abstractive
    summary via one extra generate() call.
    """
    if not aged_out:
        return existing_summary or ""

    if provider.model_name == "local-extractive":
        pieces = [f"{m.role}: {m.content[:150]}" for m in aged_out]
        addition = " | ".join(pieces)
        prefix = existing_summary or (
            "[not summarized by an LLM — local-extractive provider "
            "concatenates truncated excerpts instead of summarizing]"
        )
        return f"{prefix} | {addition}"

    excerpt = "\n".join(f"{m.role}: {m.content}" for m in aged_out)
    prompt_input = (
        f"Existing summary of earlier conversation (may be empty):\n"
        f"{existing_summary or '(none)'}\n\n"
        f"New messages to fold in:\n{excerpt}"
    )
    response = await provider.generate(
        system=_SUMMARIZATION_SYSTEM_PROMPT, history=[], user_message=prompt_input
    )
    return response.text.strip()


async def update_conversation_memory(
    provider: LLMProvider, conversation: Conversation, all_messages: list[Message]
) -> list[ConversationTurn]:
    """Given every existing message in a conversation (oldest first, NOT
    including the message currently being answered), advances
    conversation.summary if the window has moved since last time, and
    returns the ConversationTurn history to pass into
    LLMProvider.generate() for this turn.

    Mutates conversation.summary and conversation.summarized_message_count
    in place but does NOT commit — the caller (app/rag/pipeline.py) persists
    them alongside the new messages in the same transaction, so a summary
    update is never persisted without the messages it was computed from.
    """
    window = settings.CONVERSATION_RECENT_MESSAGE_WINDOW
    boundary = max(0, len(all_messages) - window)

    newly_aged_out = all_messages[conversation.summarized_message_count : boundary]
    if newly_aged_out:
        conversation.summary = await summarize_aged_out_messages(
            provider, conversation.summary, newly_aged_out
        )
        conversation.summarized_message_count = boundary

    recent = all_messages[boundary:]
    return messages_to_turns(recent)
