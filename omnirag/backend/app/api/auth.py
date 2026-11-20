"""
Auth routes.

Why /auth/login takes OAuth2PasswordRequestForm (form-encoded username/password)
instead of a JSON body: this is the shape FastAPI's own security utilities
(OAuth2PasswordBearer, and the "Authorize" button in /docs) expect. Using it
means Swagger's interactive auth flow works out of the box — paste a
username/password into the docs UI and every other endpoint's "try it out"
carries the resulting token automatically. `username` is used as the email
field; OAuth2's spec calls it username, we just treat it as the email.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.core.security import create_access_token
from app.database.session import get_db
from app.models.user import User
from app.schemas.auth import Token, UserCreate, UserRead
from app.services.auth_service import (
    EmailAlreadyRegisteredError,
    authenticate_user,
    register_user,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def register(payload: UserCreate, db: AsyncSession = Depends(get_db)) -> User:
    try:
        return await register_user(db, email=payload.email, password=payload.password)
    except EmailAlreadyRegisteredError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email is already registered.",
        ) from exc


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
) -> Token:
    user = await authenticate_user(db, email=form_data.username, password=form_data.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return Token(access_token=create_access_token(user.id))


@router.get("/me", response_model=UserRead)
async def read_current_user(current_user: User = Depends(get_current_user)) -> User:
    return current_user
