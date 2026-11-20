"""
Evaluation service (Phase 15).

    for each question:
        run the REAL RAG pipeline (rag/pipeline.py — same code path /chat
        uses, not a separate "eval mode")
            -> retrieval metrics (metrics.py) from citations vs.
               expected_document_filenames, when given
            -> generation metrics (EvaluationProvider) from the answer vs.
               the exact context the LLM saw

Running the actual pipeline rather than a parallel eval-only code path
matters: a RAG eval whose retrieval/generation logic secretly diverges from
what users actually hit isn't measuring what it claims to. The only
eval-specific things added here are scoring, on top of a real RAGAnswer.
"""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.evaluation import metrics
from app.evaluation.factory import get_evaluator
from app.rag.pipeline import answer_question
from app.schemas.evaluation import (
    EvalQuestion,
    EvalQuestionResult,
    EvalReport,
    EvalRunRequest,
    EvalSummary,
    GenerationScoresOut,
    RetrievalScores,
)


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


async def _score_one(
    db: AsyncSession,
    owner_id: uuid.UUID,
    question: EvalQuestion,
    knowledge_base_id: uuid.UUID | None,
    top_k: int,
) -> EvalQuestionResult:
    result = await answer_question(
        db,
        owner_id=owner_id,
        question=question.question,
        knowledge_base_id=knowledge_base_id,
        top_k=top_k,
    )

    retrieved_filenames = [c.document_filename for c in result.citations]

    retrieval_scores = None
    if question.expected_document_filenames:
        expected = set(question.expected_document_filenames)
        retrieval_scores = RetrievalScores(
            precision_at_k=metrics.precision_at_k(retrieved_filenames, expected, top_k),
            recall_at_k=metrics.recall_at_k(retrieved_filenames, expected, top_k),
            mrr=metrics.mean_reciprocal_rank(retrieved_filenames, expected),
            hit_rate=metrics.hit_rate(retrieved_filenames, expected, top_k),
        )

    evaluator = get_evaluator()
    generation_scores = await evaluator.score(
        question=question.question,
        answer=result.answer,
        context=result.context,
        ground_truth_answer=question.expected_answer,
    )

    return EvalQuestionResult(
        question=question.question,
        answer=result.answer,
        model_used=result.model_used,
        context_found=result.context_found,
        retrieved_document_filenames=retrieved_filenames,
        retrieval=retrieval_scores,
        generation=GenerationScoresOut(
            faithfulness=generation_scores.faithfulness,
            answer_relevancy=generation_scores.answer_relevancy,
            answer_correctness=generation_scores.answer_correctness,
        ),
    )


def _summarize(results: list[EvalQuestionResult]) -> EvalSummary:
    context_found_count = sum(1 for r in results if r.context_found)
    retrieval_rows = [r.retrieval for r in results if r.retrieval is not None]

    return EvalSummary(
        question_count=len(results),
        context_found_rate=context_found_count / len(results) if results else 0.0,
        avg_precision_at_k=_average([r.precision_at_k for r in retrieval_rows]),
        avg_recall_at_k=_average([r.recall_at_k for r in retrieval_rows]),
        avg_mrr=_average([r.mrr for r in retrieval_rows]),
        avg_hit_rate=_average([r.hit_rate for r in retrieval_rows]),
        avg_faithfulness=_average(
            [r.generation.faithfulness for r in results if r.generation.faithfulness is not None]
        ),
        avg_answer_relevancy=_average(
            [r.generation.answer_relevancy for r in results if r.generation.answer_relevancy is not None]
        ),
        avg_answer_correctness=_average(
            [r.generation.answer_correctness for r in results if r.generation.answer_correctness is not None]
        ),
    )


async def run_evaluation(db: AsyncSession, owner_id: uuid.UUID, request: EvalRunRequest) -> EvalReport:
    top_k = request.top_k or settings.RAG_TOP_K

    # Sequential, not gathered concurrently: each question goes through the
    # same DB session (db_session fixture / request-scoped session is not
    # safe for concurrent use — SQLAlchemy's AsyncSession is not
    # concurrency-safe across simultaneous awaits), and evaluation is
    # explicitly an offline/batch operation, not a latency-sensitive one —
    # correctness over throughput here.
    results = [
        await _score_one(db, owner_id, question, request.knowledge_base_id, top_k)
        for question in request.questions
    ]

    evaluator = get_evaluator()
    return EvalReport(
        eval_provider=evaluator.provider_name,
        top_k=top_k,
        results=results,
        summary=_summarize(results),
    )
