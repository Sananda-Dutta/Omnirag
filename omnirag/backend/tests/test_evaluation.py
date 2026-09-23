"""Unit tests for app/evaluation/metrics.py and app/evaluation/local_evaluator.py.

Deliberately no DB/network here (mirrors test_web_fetcher.py's split from
test_url_ingestion.py) — deterministic math and heuristic scoring tested in
isolation; the full pipeline (retrieval + generation together, through a
real knowledge base) is covered end-to-end in test_evaluation_e2e.py.
"""

import pytest

from app.evaluation import metrics
from app.evaluation.local_evaluator import LocalHeuristicEvaluator

# --- precision_at_k ----------------------------------------------------


def test_precision_at_k_all_relevant():
    assert metrics.precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0


def test_precision_at_k_partial():
    assert metrics.precision_at_k(["a", "x", "b"], {"a", "b"}, k=3) == pytest.approx(2 / 3)


def test_precision_at_k_none_relevant():
    assert metrics.precision_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0


def test_precision_at_k_empty_retrieved_is_zero():
    assert metrics.precision_at_k([], {"a"}, k=5) == 0.0


def test_precision_at_k_only_considers_top_k():
    # "b" is relevant but sits outside the top-1 window.
    assert metrics.precision_at_k(["x", "b"], {"b"}, k=1) == 0.0


# --- recall_at_k ---------------------------------------------------------


def test_recall_at_k_finds_all():
    assert metrics.recall_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0


def test_recall_at_k_finds_some():
    assert metrics.recall_at_k(["a", "x"], {"a", "b"}, k=2) == 0.5


def test_recall_at_k_empty_expected_is_vacuously_perfect():
    assert metrics.recall_at_k(["a", "b"], set(), k=2) == 1.0


def test_recall_at_k_nothing_retrieved():
    assert metrics.recall_at_k([], {"a"}, k=5) == 0.0


# --- mean_reciprocal_rank --------------------------------------------------


def test_mrr_first_result_relevant():
    assert metrics.mean_reciprocal_rank(["a", "x"], {"a"}) == 1.0


def test_mrr_third_result_relevant():
    assert metrics.mean_reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)


def test_mrr_no_relevant_result():
    assert metrics.mean_reciprocal_rank(["x", "y"], {"a"}) == 0.0


# --- hit_rate --------------------------------------------------------------


def test_hit_rate_hit_within_k():
    assert metrics.hit_rate(["x", "a"], {"a"}, k=2) == 1.0


def test_hit_rate_miss_outside_k():
    assert metrics.hit_rate(["a", "x"], {"a"}, k=1) == 1.0
    assert metrics.hit_rate(["x", "a"], {"a"}, k=1) == 0.0


def test_hit_rate_no_relevant_at_all():
    assert metrics.hit_rate(["x", "y"], {"a"}, k=2) == 0.0


# --- LocalHeuristicEvaluator -------------------------------------------


@pytest.mark.asyncio
async def test_faithfulness_high_when_answer_fully_grounded_in_context():
    evaluator = LocalHeuristicEvaluator()
    scores = await evaluator.score(
        question="What does RAG stand for?",
        answer="Retrieval augmented generation grounds answers in retrieved context.",
        context="[Source: doc.txt, chunk 0]\nRetrieval augmented generation grounds answers in retrieved context.",
        ground_truth_answer=None,
    )
    assert scores.faithfulness == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_faithfulness_low_when_answer_invents_content():
    evaluator = LocalHeuristicEvaluator()
    scores = await evaluator.score(
        question="What does RAG stand for?",
        answer="Bananas are a great source of potassium and fiber.",
        context="[Source: doc.txt, chunk 0]\nRetrieval augmented generation grounds answers in retrieved context.",
        ground_truth_answer=None,
    )
    assert scores.faithfulness is not None
    assert scores.faithfulness < 0.3


@pytest.mark.asyncio
async def test_faithfulness_is_none_when_no_context_was_retrieved():
    evaluator = LocalHeuristicEvaluator()
    scores = await evaluator.score(
        question="What does RAG stand for?",
        answer="I couldn't find anything in your documents related to that question.",
        context="",
        ground_truth_answer=None,
    )
    assert scores.faithfulness is None


@pytest.mark.asyncio
async def test_answer_relevancy_is_in_valid_range():
    evaluator = LocalHeuristicEvaluator()
    scores = await evaluator.score(
        question="What does RAG stand for?",
        answer="Retrieval augmented generation.",
        context="[Source: doc.txt, chunk 0]\nRetrieval augmented generation is a technique.",
        ground_truth_answer=None,
    )
    assert scores.answer_relevancy is not None
    assert 0.0 <= scores.answer_relevancy <= 1.0


@pytest.mark.asyncio
async def test_answer_correctness_none_without_ground_truth():
    evaluator = LocalHeuristicEvaluator()
    scores = await evaluator.score(
        question="q", answer="some answer", context="some context", ground_truth_answer=None
    )
    assert scores.answer_correctness is None


@pytest.mark.asyncio
async def test_answer_correctness_high_for_matching_answer():
    evaluator = LocalHeuristicEvaluator()
    scores = await evaluator.score(
        question="q",
        answer="Paris is the capital of France.",
        context="irrelevant",
        ground_truth_answer="Paris is the capital of France.",
    )
    assert scores.answer_correctness == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_answer_correctness_low_for_unrelated_answer():
    evaluator = LocalHeuristicEvaluator()
    scores = await evaluator.score(
        question="q",
        answer="Bananas are yellow.",
        context="irrelevant",
        ground_truth_answer="Paris is the capital of France.",
    )
    assert scores.answer_correctness is not None
    assert scores.answer_correctness < 0.2


def test_provider_name_is_descriptive():
    assert LocalHeuristicEvaluator().provider_name == "local-heuristic"
