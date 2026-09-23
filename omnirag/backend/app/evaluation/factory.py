"""
Evaluation provider factory. Same role as app/embeddings/factory.py and
app/llm/factory.py: the only place that knows about concrete evaluator
implementations.
"""

from functools import lru_cache

from app.core.config import Settings, settings
from app.evaluation.base import EvaluationProvider
from app.evaluation.local_evaluator import LocalHeuristicEvaluator


def _build_evaluator(cfg: Settings) -> EvaluationProvider:
    if cfg.EVAL_PROVIDER == "local":
        return LocalHeuristicEvaluator()
    if cfg.EVAL_PROVIDER == "ragas":
        # Imported here, not at module load: importing ragas_evaluator
        # eagerly would mean *its* lazy ragas import (see that module's
        # docstring) still runs at app-startup import time if something
        # ever imported this factory module before settings were read —
        # deferred the same way app/services/document_service.py defers
        # importing app.workers.tasks.
        from app.evaluation.ragas_evaluator import build_ragas_evaluator

        return build_ragas_evaluator()
    # Unreachable given Settings.EVAL_PROVIDER's Literal type — defensive
    # guard, same reasoning as the other provider factories.
    raise ValueError(f"Unknown EVAL_PROVIDER: {cfg.EVAL_PROVIDER!r}")


@lru_cache
def get_evaluator() -> EvaluationProvider:
    return _build_evaluator(settings)
