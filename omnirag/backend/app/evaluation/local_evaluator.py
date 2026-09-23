"""
Local heuristic evaluator.

Same honesty pattern as LocalExtractiveLLMProvider (app/llm/local_extractive.py)
and LocalHashingEmbeddingProvider (app/embeddings/local_hashing.py): this is
not what Ragas or DeepEval do internally (they typically prompt a real LLM
to judge faithfulness/relevancy in natural language). This is a real,
working, deliberately simpler technique that needs zero API key and zero
network access, so the evaluation harness — like the rest of this project —
runs end-to-end out of the box. Its ceiling is honestly lower than an
LLM-judged score: it can't detect a fluent paraphrase that changes meaning,
or credit an answer phrased in totally different words than its context.
`RagasEvaluator` exists for when that ceiling actually matters.

What each score actually measures here:
  - faithfulness: fraction of the answer's own content words that also
    appear in the context. A cheap, real proxy for "is this asserted, or
    grounded" — an answer that invents specifics not present anywhere in
    the context will score low, almost by definition.
  - answer_relevancy: cosine similarity between the question's embedding
    and the answer's embedding, using this project's already-configured
    EmbeddingProvider (whichever one EMBEDDING_PROVIDER points at) rather
    than a second, separate embedding call — consistent config, and no new
    dependency.
  - answer_correctness: token-level F1 between the generated answer and a
    supplied ground-truth answer (only computed when one is given) — the
    same spirit as the ROUGE-style overlap metrics classic QA evaluation
    has used for years, not a claim of semantic equivalence.
"""

import math
import re
from collections import Counter

from app.embeddings.factory import get_embedding_provider
from app.evaluation.base import EvaluationProvider, GenerationScores

_WORD_RE = re.compile(r"[a-z0-9]+")

# Same short, deliberately narrow relevance-scoring stopword list as
# app/llm/local_extractive.py — filters the handful of words common enough
# to cause false-positive "overlap" in short text, not a full linguistic
# stopword corpus. Duplicated rather than imported: local_extractive.py's
# list is scoped to its own module's docstring-documented purpose, and
# evaluation is a genuinely separate concern that happens to want the same
# small list right now — importing across those two would couple two
# things that should be free to diverge later for unrelated reasons.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "by", "for", "with", "as", "and", "or",
    "but", "not", "this", "that", "these", "those", "it", "its", "what",
    "which", "who", "whom", "when", "where", "how", "why", "do", "does",
    "did", "can", "could", "will", "would", "should", "has", "have", "had",
}


def _content_words(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    # Clamp: floating-point error can push a mathematically-in-range cosine
    # a hair outside [-1, 1] (e.g. 1.0000000000000002), which would make a
    # downstream "clip to [0, 1]" below look like a real out-of-range value
    # instead of the rounding noise it actually is.
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


def _token_f1(predicted: str, reference: str) -> float:
    pred_tokens = _WORD_RE.findall(predicted.lower())
    ref_tokens = _WORD_RE.findall(reference.lower())
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


class LocalHeuristicEvaluator(EvaluationProvider):
    @property
    def provider_name(self) -> str:
        return "local-heuristic"

    async def score(
        self,
        *,
        question: str,
        answer: str,
        context: str,
        ground_truth_answer: str | None,
    ) -> GenerationScores:
        faithfulness = self._faithfulness(answer, context)
        relevancy = await self._answer_relevancy(question, answer)
        correctness = (
            _token_f1(answer, ground_truth_answer) if ground_truth_answer else None
        )
        return GenerationScores(
            faithfulness=faithfulness,
            answer_relevancy=relevancy,
            answer_correctness=correctness,
        )

    def _faithfulness(self, answer: str, context: str) -> float | None:
        if not context.strip():
            # No context was ever retrieved (context_found=False) — there's
            # nothing for the answer to be faithful *to*. None (not 0.0),
            # since 0.0 would misreport "definitely unfaithful" when the
            # real situation is "not applicable".
            return None
        answer_words = _content_words(answer)
        if not answer_words:
            return 0.0
        context_words = set(_content_words(context))
        supported = sum(1 for w in answer_words if w in context_words)
        return supported / len(answer_words)

    async def _answer_relevancy(self, question: str, answer: str) -> float | None:
        if not answer.strip():
            return 0.0
        provider = get_embedding_provider()
        question_vec, answer_vec = await provider.embed_texts([question, answer])
        similarity = _cosine_similarity(question_vec, answer_vec)
        # Cosine similarity ranges [-1, 1]; GenerationScores documents every
        # field as [0, 1], so rescale rather than silently violating that
        # contract for embedding providers whose vectors aren't
        # already non-negative.
        return (similarity + 1.0) / 2.0
