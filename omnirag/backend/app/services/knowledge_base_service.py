"""
Knowledge base service.

Every function here takes `owner_id` and filters by it — there is no
function that fetches a KnowledgeBase by ID alone. That's deliberate: it
makes "can this user see this row" a property of the query itself, not
something the caller has to remember to check afterward. The alternative
(fetch by ID, then `if kb.owner_id != current_user.id: raise 403`) works
right up until someone adds a new call site and forgets the check.
"""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge_base import KnowledgeBase


class KnowledgeBaseNotFoundError(Exception):
    pass


async def create_knowledge_base(
    db: AsyncSession, owner_id: uuid.UUID, name: str, description: str | None
) -> KnowledgeBase:
    kb = KnowledgeBase(name=name, description=description, owner_id=owner_id)
    db.add(kb)
    await db.commit()
    await db.refresh(kb)
    return kb


async def list_knowledge_bases(db: AsyncSession, owner_id: uuid.UUID) -> list[KnowledgeBase]:
    result = await db.execute(
        select(KnowledgeBase)
        .where(KnowledgeBase.owner_id == owner_id)
        .order_by(KnowledgeBase.created_at.desc())
    )
    return list(result.scalars().all())


async def get_knowledge_base(
    db: AsyncSession, owner_id: uuid.UUID, knowledge_base_id: uuid.UUID
) -> KnowledgeBase:
    result = await db.execute(
        select(KnowledgeBase).where(
            KnowledgeBase.id == knowledge_base_id, KnowledgeBase.owner_id == owner_id
        )
    )
    kb = result.scalar_one_or_none()
    if kb is None:
        # Deliberately the same error for "doesn't exist" and "belongs to
        # someone else" — the router turns this into a 404, never a 403, so
        # a user can't distinguish "not yours" from "doesn't exist" and use
        # that to enumerate other users' knowledge base IDs.
        raise KnowledgeBaseNotFoundError(str(knowledge_base_id))
    return kb


async def delete_knowledge_base(
    db: AsyncSession, owner_id: uuid.UUID, knowledge_base_id: uuid.UUID
) -> None:
    kb = await get_knowledge_base(db, owner_id, knowledge_base_id)
    await db.delete(kb)
    await db.commit()
