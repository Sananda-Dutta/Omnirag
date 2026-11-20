from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.database.session import get_db
from app.models.user import User
from app.schemas.search import SearchRequest, SearchResultItem
from app.services.knowledge_base_service import KnowledgeBaseNotFoundError
from app.services.search_service import search as run_search

router = APIRouter(tags=["search"])


@router.post("/search", response_model=list[SearchResultItem])
async def search(
    payload: SearchRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        return await run_search(
            db,
            owner_id=current_user.id,
            query=payload.query,
            knowledge_base_id=payload.knowledge_base_id,
            top_k=payload.top_k,
        )
    except KnowledgeBaseNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from exc
