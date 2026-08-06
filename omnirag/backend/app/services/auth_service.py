"""
Auth service: registration and credential verification.

Kept separate from app/api/auth.py on purpose — the router's job is HTTP
(status codes, request parsing), this module's job is "what does it mean to
register a user" / "what does it mean to authenticate one," independent of
whether that's triggered by a REST call, a CLI seed script, or a future
admin tool.
"""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password, verify_password
from app.models.user import User

# A precomputed hash of a value nobody will ever type, used to burn the same
# bcrypt cost when the email doesn't exist at all. Without this,
# authenticate_user returns near-instantly for unknown emails (no bcrypt call)
# but takes ~270ms for known emails with a wrong password — a response-time
# side channel that lets an attacker enumerate registered emails without ever
# seeing an error message. Computed once at import time, not per call.
_DUMMY_HASH = hash_password("this-value-is-never-a-real-password-000000")


class EmailAlreadyRegisteredError(Exception):
    pass


async def get_user_by_email(db: AsyncSession, email: str) -> User | None:
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def get_user_by_id(db: AsyncSession, user_id: uuid.UUID) -> User | None:
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def register_user(db: AsyncSession, email: str, password: str) -> User:
    existing = await get_user_by_email(db, email)
    if existing is not None:
        raise EmailAlreadyRegisteredError(email)

    user = User(email=email, hashed_password=hash_password(password))
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def authenticate_user(db: AsyncSession, email: str, password: str) -> User | None:
    """Returns the User if email+password are valid and the account is
    active, otherwise None. Deliberately doesn't distinguish "no such user"
    from "wrong password" in its return value — that distinction belongs in
    logs/metrics, never in what an API client can observe, or you've built
    a user-enumeration oracle.

    Also always performs a bcrypt comparison, even for an email that doesn't
    exist (against `_DUMMY_HASH`), so the two failure cases take the same
    amount of time — see the module-level comment on `_DUMMY_HASH`."""
    user = await get_user_by_email(db, email)

    if user is None:
        verify_password(password, _DUMMY_HASH)
        return None

    password_ok = verify_password(password, user.hashed_password)
    if not user.is_active or not password_ok:
        return None
    return user
