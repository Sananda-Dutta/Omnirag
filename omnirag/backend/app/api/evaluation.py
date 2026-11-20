from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.database.session import get_db
from app.llm.base import LLMProviderError
from app.models.user import User
from app.schemas.evaluation import EvalReport, EvalRunRequest
from app.services.evaluation_service import run_evaluation
from app.services.knowledge_base_service import KnowledgeBaseNotFoundError

router = APIRouter(tags=["evaluation"])


@router.post("/evaluation/run", response_model=EvalReport)
async def run(
    payload: EvalRunRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Runs the real RAG pipeline (same code /chat uses) against a
    user-supplied set of questions, scores retrieval and generation
    quality, and returns per-question results plus aggregated averages.
    See app/services/evaluation_service.py and app/evaluation/ for what
    each score means and app/core/config.py's EVAL_PROVIDER for switching
    the scoring backend."""
    try:
        return await run_evaluation(db, owner_id=current_user.id, request=payload)
    except KnowledgeBaseNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from exc
    except LLMProviderError as exc:
        # The upstream model service failed, not this API — 502, with the
        # provider-error text (never contains the key) so the UI can show it.
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
