from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.database.session import get_db
from app.models.user import User
from app.rag.pipeline import answer_question
from app.schemas.chat import ChatRequest, ChatResponse, CitationItem
from app.services.knowledge_base_service import KnowledgeBaseNotFoundError

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        result = await answer_question(
            db,
            owner_id=current_user.id,
            question=payload.question,
            knowledge_base_id=payload.knowledge_base_id,
        )
    except KnowledgeBaseNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from exc

    return ChatResponse(
        answer=result.answer,
        model_used=result.model_used,
        context_found=result.context_found,
        citations=[
            CitationItem(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                document_filename=c.document_filename,
                chunk_index=c.chunk_index,
                score=c.score,
                text_snippet=c.text_snippet,
            )
            for c in result.citations
        ],
    )
