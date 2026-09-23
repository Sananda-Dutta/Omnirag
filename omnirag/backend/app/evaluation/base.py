"""
EvaluationProvider abstraction.

Same reasoning as EmbeddingProvider (app/embeddings/base.py) and LLMProvider
(app/llm/base.py): callers (app/services/evaluation_service.py) depend on
this interface, never on a concrete scoring implementation. Swapping the
evaluation backend — a fast, free, local heuristic vs. a real LLM-judged
Ragas run — is a config change (`EVAL_PROVIDER=...`), not a rewrite.

Why generation quality needs its own provider abstraction at all, when
metrics.py (retrieval precision/recall/MRR) is just plain functions: those
retrieval metrics are unambiguous set arithmetic — given "retrieved" and
"expected" document IDs, there's exactly one correct precision@k. Scoring
"is this answer faithful to its context" has no single correct algorithm —
a cheap heuristic (LocalHeuristicEvaluator) and an LLM-as-judge (Ragas)
give genuinely different numbers for genuinely different cost/accuracy
tradeoffs, which is exactly when this codebase reaches for the Strategy
pattern (see EmbeddingProvider's docstring for the same argument).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationScores:
    """All scores are in [0.0, 1.0], higher is better, and None when the
    example didn't provide what's needed to compute that particular score
    (e.g. answer_correctness needs a ground-truth answer; faithfulness and
    answer_relevancy don't)."""

    faithfulness: float | None
    """How much of the answer is actually supported by the retrieved
    context, vs. asserted with nothing behind it. The single most
    important RAG-specific metric — a fluent, well-cited-looking answer
    that isn't actually grounded in the context is the core failure mode
    RAG exists to prevent."""

    answer_relevancy: float | None
    """How on-topic the answer is for the question asked — catches a
    well-grounded answer that nonetheless doesn't address what was asked
    (e.g. answering a narrower or broader question than the one given)."""

    answer_correctness: float | None
    """How well the answer matches a known-correct reference answer, when
    one was supplied for this example. Unlike faithfulness/relevancy, this
    requires a labeled golden dataset — most real eval sets won't have one
    for every question, hence Optional."""


class EvaluationProvider(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Recorded on the eval report so a report is self-describing about
        which scoring backend produced it — the same reasoning as
        DocumentChunk.embedding_model existing at all: two reports using
        different providers are not directly comparable, and a report
        should say which one it is rather than leave that implicit."""
        ...

    @abstractmethod
    async def score(
        self,
        *,
        question: str,
        answer: str,
        context: str,
        ground_truth_answer: str | None,
    ) -> GenerationScores:
        """`context` is the exact labeled context block the RAG pipeline
        handed to the LLM (RAGAnswer.context) — scoring against anything
        else (e.g. reconstructing context from citation snippets) would be
        scoring faithfulness to an approximation of what the model saw,
        not to what it actually saw."""
        ...
