"""
Prompt construction for grounded generation.

The `<context>...</context>` delimiter is a deliberate, documented contract
with app/llm/local_extractive.py (see that module's docstring) — it's the
one provider that actually parses this back out of the system prompt. Real
LLM providers (Anthropic/OpenAI) just receive the whole string; they don't
need to know the delimiter exists, but keeping it consistent means the same
prompt-building code works unmodified against every provider.

Hallucination-control instructions are deliberately explicit and repeated
(both "use only the context" and "say so if it's not there") rather than
implied — this is the actual mechanism by which this project tries to keep
generated answers grounded, not just documentation of an intention.
"""

SYSTEM_PROMPT_TEMPLATE = """You are a knowledgeable assistant answering questions using ONLY the retrieved passages below, sourced from the user's own uploaded documents.

Rules:
- Answer using only information found in the retrieved passages. Do not use outside knowledge, even if you know the answer.
- If the retrieved passages do not contain enough information to answer the question, say so plainly (e.g. "The provided documents don't contain information about that") instead of guessing or filling gaps from general knowledge.
- When you state a fact from the retrieved passages, keep it traceable to the source material — don't blend it with invented specifics.
- Be concise and direct.

{context_open}
{context}
{context_close}"""

# The delimiter literals are kept out of the prose above and interpolated
# only here — see local_extractive.py's docstring for why this exact
# collision (an instruction that happens to mention the delimiter syntax)
# broke the regex-based parser during development: it matched from the
# prose mention instead of the real tag, pulling the rest of the
# instructions into the "extracted" answer.
_CONTEXT_OPEN = "<context>"
_CONTEXT_CLOSE = "</context>"


def build_system_prompt(context: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        context=context, context_open=_CONTEXT_OPEN, context_close=_CONTEXT_CLOSE
    )
