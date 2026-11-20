"""
Current-user dependency.

This is the single chokepoint every user-scoped endpoint (knowledge bases in
this phase's tests, documents/chat/conversations from Phase 4 onward) will
depend on. Centralizing it here means "how do we know who's making this
request" is answered in exactly one place — critical for the per-user
isolation requirement, since every later query that filters by
`owner_id == current_user.id` is only as trustworthy as this function.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import InvalidTokenError, decode_access_token
from app.database.session import get_db
from app.models.user import User
from app.services.auth_service import get_user_by_id

# tokenUrl is where Swagger's "Authorize" button will POST to try a login —
# purely for the interactive docs UI, not used by our own code path.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        user_id = decode_access_token(token)
    except InvalidTokenError as exc:
        raise credentials_error from exc

    user = await get_user_by_id(db, user_id)
    if user is None or not user.is_active:
        raise credentials_error

    return user
