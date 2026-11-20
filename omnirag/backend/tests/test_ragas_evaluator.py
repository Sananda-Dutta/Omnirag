"""
Tests for app/evaluation/ragas_evaluator.py.

Deliberately does NOT call a real OpenAI API (no usable key in this
sandbox — same honest limit as tests/test_llm.py's Anthropic/OpenAI
provider tests). What's tested instead:
  - the "no API key configured" error path (doesn't need ragas installed
    or any network call — checked before either)
  - the empty-context short-circuit (doesn't need a network call: no
    metric is scored when there's no context to be faithful to)
  - that the lazy `ragas` import + object construction (AsyncOpenAI client,
    judge_llm, judge_embeddings) actually succeeds against the real,
    installed ragas package — this is what would break first if the
    documented `pip install ragas langchain-community==0.3.27` combination
    stopped working
"""

import pytest

from app.evaluation.factory import _build_evaluator
from app.evaluation.ragas_evaluator import RagasEvaluator, RagasMissingAPIKeyError

ragas_installed = True
try:
    import ragas  # noqa: F401
except ModuleNotFoundError:
    ragas_installed = False


def test_provider_name_includes_judge_model():
    evaluator = RagasEvaluator(openai_api_key="sk-fake", judge_model="gpt-4o-mini")
    assert evaluator.provider_name == "ragas (gpt-4o-mini)"


@pytest.mark.asyncio
async def test_raises_clear_error_without_api_key():
    evaluator = RagasEvaluator(openai_api_key=None, judge_model="gpt-4o-mini")
    with pytest.raises(RagasMissingAPIKeyError):
        await evaluator.score(
            question="q", answer="a", context="some real context", ground_truth_answer=None
        )


@pytest.mark.asyncio
async def test_empty_context_short_circuits_without_needing_a_real_call():
    """No context was ever retrieved — this must return before touching
    the network, so it works even with an obviously-fake key."""
    evaluator = RagasEvaluator(openai_api_key="sk-fake-not-a-real-key", judge_model="gpt-4o-mini")
    scores = await evaluator.score(question="q", answer="a", context="", ground_truth_answer=None)
    assert scores.faithfulness is None
    assert scores.answer_relevancy is None
    assert scores.answer_correctness is None


@pytest.mark.skipif(not ragas_installed, reason="ragas not installed in this environment")
def test_ensure_metrics_builds_real_ragas_objects_without_a_network_call():
    """Constructing the judge LLM/embeddings/metric objects is pure object
    construction (no request is sent until .ascore() is actually called),
    so this can run for real against the installed ragas package without
    needing a usable API key."""
    evaluator = RagasEvaluator(openai_api_key="sk-fake-not-a-real-key", judge_model="gpt-4o-mini")
    evaluator._ensure_metrics()
    assert evaluator._faithfulness is not None
    assert evaluator._answer_relevancy is not None
    assert evaluator._answer_correctness is not None


def test_local_provider_is_default_and_needs_no_setup():
    from app.core.config import Settings

    cfg = Settings(EVAL_PROVIDER="local")
    evaluator = _build_evaluator(cfg)
    assert evaluator.provider_name == "local-heuristic"


@pytest.mark.skipif(not ragas_installed, reason="ragas not installed in this environment")
def test_factory_builds_ragas_evaluator_when_configured():
    from app.core.config import Settings

    cfg = Settings(EVAL_PROVIDER="ragas", OPENAI_API_KEY="sk-fake", EVAL_JUDGE_MODEL="gpt-4o-mini")
    evaluator = _build_evaluator(cfg)
    assert isinstance(evaluator, RagasEvaluator)
    assert evaluator.provider_name == "ragas (gpt-4o-mini)"
