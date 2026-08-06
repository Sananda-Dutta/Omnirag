"""
Password hashing and JWT handling.

Why `bcrypt` directly instead of `passlib`:
    passlib's bcrypt backend has been broken against bcrypt>=4.1 for a while
    (it probes for a `__about__.__version__` attribute that recent bcrypt
    releases removed, so passlib either warns loudly or hard-fails depending
    on version combinations) and passlib itself has had no real release in
    years. Calling `bcrypt` directly is fewer moving parts and no
    compatibility landmine. It costs us passlib's multi-algorithm abstraction,
    but this project only ever needs bcrypt.

Why bcrypt over argon2 here:
    argon2id is the stronger modern choice, but bcrypt is still considered
    secure, is what most FastAPI production codebases actually run, and
    needs no extra native dependencies beyond the `bcrypt` wheel. Worth
    revisiting if this were a greenfield security-critical product; not
    worth the extra dependency for a portfolio project explaining its
    tradeoffs is more valuable here than chasing the theoretical best choice.

Why PyJWT instead of python-jose:
    python-jose has had unresolved CVEs (algorithm-confusion issues in its
    JWS handling) and is more loosely maintained. PyJWT is the library
    FastAPI's own docs now point to, and it does exactly the one thing we
    need: sign and verify HS256 tokens.

Token design: we deliberately keep the JWT payload minimal (`sub`, `exp`,
`iat`) — no email, no roles yet. Anything inside the token is visible to
anyone holding it (JWTs are signed, not encrypted), so this is a place to
default to "as little as possible" rather than "whatever's convenient."
"""

from datetime import datetime, timedelta, timezone
from uuid import UUID

import bcrypt
import jwt

from app.core.config import settings

ALGORITHM = "HS256"


def hash_password(plain_password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(plain_password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except ValueError:
        # Malformed hash (shouldn't happen outside corrupted data) — treat as
        # "does not match" rather than raising, so a bad row can't 500 login.
        return False


def create_access_token(user_id: UUID) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=ALGORITHM)


class InvalidTokenError(Exception):
    pass


def decode_access_token(token: str) -> UUID:
    """Returns the user ID encoded in the token, or raises InvalidTokenError."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.PyJWTError as exc:
        raise InvalidTokenError(str(exc)) from exc

    sub = payload.get("sub")
    if sub is None:
        raise InvalidTokenError("token missing 'sub' claim")

    try:
        return UUID(sub)
    except ValueError as exc:
        raise InvalidTokenError("token 'sub' claim is not a valid UUID") from exc
