"""
Document routes.

Upload takes the file as multipart/form-data (FastAPI's UploadFile) rather
than base64-in-JSON — base64 inflates payload size by ~33% and forces the
whole file to sit in memory as a string before FastAPI even sees it as
bytes. UploadFile streams from a spooled temp file, which matters once
someone uploads a file close to MAX_UPLOAD_SIZE_MB.

Reading the whole file into memory with `await file.read()` here is a
known, deliberate limit for Phase 4 (fine up to MAX_UPLOAD_SIZE_MB=20MB) —
true streaming validation (rejecting an oversized file before it's fully
buffered) is a Phase 19 (security/hardening) concern, noted there rather
than solved here.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.database.session import get_db
from app.models.user import User
from app.schemas.document import DocumentDetail, DocumentRead
from app.services.document_service import (
    DocumentNotFoundError,
    FileTooLargeError,
    UnsupportedFileTypeError,
    delete_document,
    get_document,
    list_documents,
    upload_document,
)
from app.services.knowledge_base_service import KnowledgeBaseNotFoundError

router = APIRouter(tags=["documents"])


@router.post(
    "/knowledge-bases/{knowledge_base_id}/documents",
    response_model=DocumentRead,
    status_code=status.HTTP_201_CREATED,
)
async def upload(
    knowledge_base_id: uuid.UUID,
    file: UploadFile,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    content = await file.read()
    try:
        return await upload_document(
            db,
            owner_id=current_user.id,
            knowledge_base_id=knowledge_base_id,
            filename=file.filename or "unnamed",
            content_type=file.content_type or "application/octet-stream",
            content=content,
        )
    except KnowledgeBaseNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from exc
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)) from exc
    except FileTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)
        ) from exc


@router.get("/knowledge-bases/{knowledge_base_id}/documents", response_model=list[DocumentRead])
async def list_for_knowledge_base(
    knowledge_base_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        return await list_documents(db, current_user.id, knowledge_base_id)
    except KnowledgeBaseNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from exc


@router.get("/documents/{document_id}", response_model=DocumentDetail)
async def get_one(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        return await get_document(db, current_user.id, document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from exc


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        await delete_document(db, current_user.id, document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from exc
