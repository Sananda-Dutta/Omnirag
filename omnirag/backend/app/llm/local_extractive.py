"""
Local extractive "LLM" provider.

Said plainly: this is not a language model. There is no local free/open
generation model available in this sandbox (no network access to
HuggingFace to download one, and no meaningful CPU-only generation quality
even if there were). Pretending otherwise — e.g. hand-writing a canned
response — would be exactly the "fake implementation" this project's own
ground rules rule out.

What this actually does, and why it's still useful: it's a real, if crude,
extractive answering algorithm. It ranks the sentences in the retrieved
context by word overlap with the question (same bag-of-words spirit as
LocalHashingEmbeddingProvider in Phase 5) and returns the most relevant
ones verbatim, clearly labeled as extracted rather than generated. This is
a legitimate, decades-old IR technique (this is literally how early
open-domain QA systems worked before neural generation), not a placeholder
masquerading as one — but its ceiling is "find and surface the right
sentence," not "synthesize an answer across multiple sentences" or "explain
in different words." That's the honest tradeoff for a provider that needs
zero network access and zero API key.

Coupling worth being explicit about: this provider parses the retrieved
context back out of the `system` prompt via a `<context>...</context>`
delimiter, rather than receiving it as a separate structured argument. That
delimiter is a real contract with app/rag/prompts.py's template — documented
there too. This coupling is specific to this fallback provider; a real LLM
provider just receives the whole system string as-is and never parses it.
"""

import re

from app.llm.base import LLMProvider, LLMResponse

_CONTEXT_RE = re.compile(r"<context>(.*?)</context>", re.DOTALL)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")

# Filtered out of relevance scoring (not from the extracted output text
# itself — only from what counts as "overlap"). Without this, a word as
# generic as "is" or "the" being present in both the question and a
# completely unrelated sentence is enough to mark that sentence "relevant"
# — caught during development: "What is quantum entanglement?" scored a
# false-positive match against "The weather today is sunny" purely on the
# shared word "is". Small, deliberately short list — this is a relevance
# *filter*, not a linguistic stopword corpus; it only needs to cover the
# words common enough to cause false positives in short sentences.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "by", "for", "with", "as", "and", "or",
    "but", "not", "this", "that", "these", "those", "it", "its", "what",
    "which", "who", "whom", "when", "where", "how", "why", "do", "does",
    "did", "can", "could", "will", "would", "should", "has", "have", "had",
}


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _tokenize_for_scoring(text: str) -> set[str]:
    return _tokenize(text) - _STOPWORDS


class LocalExtractiveLLMProvider(LLMProvider):
    def __init__(self, max_sentences: int = 3):
        self._max_sentences = max_sentences

    @property
    def model_name(self) -> str:
        return "local-extractive"

    async def generate(self, *, system: str, user_message: str) -> LLMResponse:
        match = _CONTEXT_RE.search(system)
        context = match.group(1).strip() if match else ""

        if not context:
            text = (
                "[local-extractive provider — no LLM was called]\n"
                "No context was provided to extract an answer from."
            )
            return LLMResponse(text=text, model=self.model_name)

        sentences = [s.strip() for s in _SENTENCE_RE.split(context) if s.strip()]
        question_words = _tokenize_for_scoring(user_message)

        scored = [
            (len(question_words & _tokenize_for_scoring(sentence)), i, sentence)
            for i, sentence in enumerate(sentences)
        ]
        scored_relevant = [(score, i, s) for score, i, s in scored if score > 0]

        if not scored_relevant:
            text = (
                "[local-extractive provider — no LLM was called]\n"
                "Nothing in the retrieved context appears to overlap with the "
                "question closely enough to extract an answer."
            )
            return LLMResponse(text=text, model=self.model_name)

        # Rank by relevance score (descending), trim to max_sentences, then
        # restore original reading order — a jumbled-order excerpt is
        # harder to read than a lower-recall one kept in document order.
        ranked = sorted(scored_relevant, key=lambda t: (-t[0], t[1]))[: self._max_sentences]
        ordered = [s for _, _, s in sorted(ranked, key=lambda t: t[1])]

        text = (
            "[local-extractive provider — no LLM was called; sentences below "
            "are extracted verbatim from retrieved context, not generated]\n\n"
            + " ".join(ordered)
        )
        return LLMResponse(text=text, model=self.model_name)
