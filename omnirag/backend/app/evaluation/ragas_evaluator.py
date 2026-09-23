"""
Ragas-backed evaluator.

Real integration against Ragas' modern (`ragas.metrics.collections`) API —
not a placeholder. Two things make this different from every other
provider in this codebase (AnthropicLLMProvider, OpenAIEmbeddingProvider):

1. `ragas` (and its own dependency chain — langchain-core, instructor,
   datasets, ...) is NOT in requirements.txt. It's a genuinely heavy
   dependency for something most of this project needs to run without,
   so it's imported lazily, inside `_load_ragas()`, only when
   `EVAL_PROVIDER=ragas` is actually selected. `ModuleNotFoundError` is
   caught and re-raised as `RagasNotInstalledError` with the exact pip
   command to fix it — a config problem should look like one, not like a
   stack trace three frames into someone else's package.

2. Verified against a real, pinned, working dependency set in this
   project's own dev sandbox — worth being explicit about, because it
   wasn't a given: `pip install ragas` alone resolves an incompatible
   `langchain-community` (latest, 0.4.x, dropped a submodule ragas 0.4.3
   still imports at module load — an upstream ragas bug, not a mistake in
   this integration). The fix, confirmed working here, is pinning
   `langchain-community==0.3.27` alongside `ragas`:

       pip install ragas langchain-community==0.3.27

   This evaluator's *code* has NOT been exercised against a live OpenAI
   API call in this sandbox (no usable OPENAI_API_KEY here) — same honest
   "should work, not verified live" caveat as AnthropicLLMProvider and
   OpenAILLMProvider, documented in their own docstrings and the project
   README, applies here too.

Why this needs a real OpenAI API key specifically, even if `LLM_PROVIDER`
is set to `anthropic` or `local`: Ragas' faithfulness/answer-relevancy
metrics work by prompting a judge model in a structured way (via the
`instructor` library) and need real embeddings for the relevancy metric.
`ragas.llms.llm_factory`'s built-in providers are OpenAI-shaped; wiring an
Anthropic judge through would mean building and maintaining a second
integration path with no local-sandbox way to verify it either. If you
need an Anthropic-judged eval, that's a real, scoped follow-up — not
silently faked here.
"""

from app.core.config import settings
from app.evaluation.base import EvaluationProvider, GenerationScores


class RagasNotInstalledError(Exception):
    def __init__(self) -> None:
        super().__init__(
            "EVAL_PROVIDER=ragas needs the ragas package, which is not "
            "installed (it's intentionally not in requirements.txt — see "
            "app/evaluation/ragas_evaluator.py's docstring for why). Install it with:\n\n"
            "    pip install ragas langchain-community==0.3.27\n\n"
            "(the langchain-community pin works around an upstream ragas "
            "import bug against langchain-community>=0.4 — confirmed "
            "while building this integration)."
        )


class RagasMissingAPIKeyError(Exception):
    def __init__(self) -> None:
        super().__init__(
            "EVAL_PROVIDER=ragas needs a real OPENAI_API_KEY — Ragas' "
            "faithfulness/answer-relevancy/answer-correctness metrics judge "
            "each answer by prompting a real model, there's no local, "
            "no-API-key option (that's what EVAL_PROVIDER=local is for). "
            "Set OPENAI_API_KEY in your .env."
        )


def _load_ragas():
    try:
        from ragas.metrics.collections import AnswerCorrectness, AnswerRelevancy, Faithfulness
        from ragas.llms import llm_factory
    except ModuleNotFoundError as exc:
        raise RagasNotInstalledError() from exc
    return Faithfulness, AnswerRelevancy, AnswerCorrectness, llm_factory


class RagasEvaluator(EvaluationProvider):
    """Lazily builds its Ragas metric objects on first use (not in
    `__init__`) so constructing a RagasEvaluator — which the factory does
    unconditionally whenever EVAL_PROVIDER=ragas, even before any
    evaluation actually runs — doesn't itself require ragas to already be
    importable and configured. The error only surfaces once you actually
    try to score something, same as how AnthropicLLMProvider doesn't fail
    to construct without an API key, only to `generate()`."""

    def __init__(self, openai_api_key: str | None, judge_model: str) -> None:
        self._api_key = openai_api_key
        self._judge_model = judge_model
        self._faithfulness = None
        self._answer_relevancy = None
        self._answer_correctness = None

    @property
    def provider_name(self) -> str:
        return f"ragas ({self._judge_model})"

    def _ensure_metrics(self) -> None:
        if self._faithfulness is not None:
            return
        if not self._api_key:
            raise RagasMissingAPIKeyError()

        Faithfulness, AnswerRelevancy, AnswerCorrectness, llm_factory = _load_ragas()

        from openai import AsyncOpenAI
        from ragas.embeddings import OpenAIEmbeddings

        client = AsyncOpenAI(api_key=self._api_key)
        judge_llm = llm_factory(model=self._judge_model, provider="openai", client=client)
        judge_embeddings = OpenAIEmbeddings(client=client, model="text-embedding-3-small")

        self._faithfulness = Faithfulness(llm=judge_llm)
        self._answer_relevancy = AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings)
        self._answer_correctness = AnswerCorrectness(llm=judge_llm, embeddings=judge_embeddings)

    async def score(
        self,
        *,
        question: str,
        answer: str,
        context: str,
        ground_truth_answer: str | None,
    ) -> GenerationScores:
        self._ensure_metrics()

        if not context.strip():
            # Same "not applicable, not zero" reasoning as
            # LocalHeuristicEvaluator — no context was ever retrieved, so
            # there's nothing to score faithfulness against, and asking
            # Ragas to judge relevancy of an answer that's already known to
            # be the fixed "nothing found" response would just be paying
            # for an API call with a foregone conclusion.
            return GenerationScores(faithfulness=None, answer_relevancy=None, answer_correctness=None)

        faithfulness_result = await self._faithfulness.ascore(
            user_input=question, response=answer, retrieved_contexts=[context]
        )
        relevancy_result = await self._answer_relevancy.ascore(user_input=question, response=answer)

        correctness = None
        if ground_truth_answer:
            correctness_result = await self._answer_correctness.ascore(
                user_input=question, response=answer, reference=ground_truth_answer
            )
            correctness = float(correctness_result.value)

        return GenerationScores(
            faithfulness=float(faithfulness_result.value),
            answer_relevancy=float(relevancy_result.value),
            answer_correctness=correctness,
        )


def build_ragas_evaluator() -> RagasEvaluator:
    return RagasEvaluator(openai_api_key=settings.OPENAI_API_KEY, judge_model=settings.EVAL_JUDGE_MODEL)
