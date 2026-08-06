"""
Auth tests.

Deliberately includes negative cases, not just "does registration work":
duplicate email, wrong password, malformed token, and an actually-expired
token (not just "assume the exp check works because the library is
trustworthy" — we construct one by hand and confirm it's rejected).
"""

from datetime import datetime, timedelta, timezone

import jwt
import pytest
from httpx import AsyncClient

from app.core.config import settings
from app.core.security import ALGORITHM


@pytest.mark.asyncio
async def test_register_creates_user(client: AsyncClient):
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": "new@example.com", "password": "correct-horse-battery"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["email"] == "new@example.com"
    assert body["is_active"] is True
    assert "password" not in body
    assert "hashed_password" not in body


@pytest.mark.asyncio
async def test_register_rejects_duplicate_email(client: AsyncClient):
    payload = {"email": "dup@example.com", "password": "correct-horse-battery"}
    first = await client.post("/api/v1/auth/register", json=payload)
    assert first.status_code == 201

    second = await client.post("/api/v1/auth/register", json=payload)
    assert second.status_code == 400


@pytest.mark.asyncio
async def test_register_rejects_short_password(client: AsyncClient):
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": "short@example.com", "password": "short"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_login_succeeds_with_correct_credentials(client: AsyncClient):
    await client.post(
        "/api/v1/auth/register",
        json={"email": "login@example.com", "password": "correct-horse-battery"},
    )
    response = await client.post(
        "/api/v1/auth/login",
        data={"username": "login@example.com", "password": "correct-horse-battery"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert len(body["access_token"]) > 0


@pytest.mark.asyncio
async def test_login_fails_with_wrong_password(client: AsyncClient):
    await client.post(
        "/api/v1/auth/register",
        json={"email": "wrongpw@example.com", "password": "correct-horse-battery"},
    )
    response = await client.post(
        "/api/v1/auth/login",
        data={"username": "wrongpw@example.com", "password": "totally-wrong"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_login_fails_for_unknown_email(client: AsyncClient):
    response = await client.post(
        "/api/v1/auth/login",
        data={"username": "nobody@example.com", "password": "whatever"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_me_returns_current_user_with_valid_token(client: AsyncClient):
    await client.post(
        "/api/v1/auth/register",
        json={"email": "me@example.com", "password": "correct-horse-battery"},
    )
    login = await client.post(
        "/api/v1/auth/login",
        data={"username": "me@example.com", "password": "correct-horse-battery"},
    )
    token = login.json()["access_token"]

    response = await client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["email"] == "me@example.com"


@pytest.mark.asyncio
async def test_me_rejects_missing_token(client: AsyncClient):
    response = await client.get("/api/v1/auth/me")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_me_rejects_malformed_token(client: AsyncClient):
    response = await client.get(
        "/api/v1/auth/me", headers={"Authorization": "Bearer not-a-real-token"}
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_me_rejects_expired_token(client: AsyncClient):
    # Construct an already-expired token by hand, bypassing create_access_token,
    # so this test proves the *verification* path checks `exp` rather than
    # trusting that tokens we generate are always fresh.
    expired_payload = {
        "sub": "11111111-1111-1111-1111-111111111111",
        "iat": datetime.now(timezone.utc) - timedelta(minutes=120),
        "exp": datetime.now(timezone.utc) - timedelta(minutes=60),
    }
    expired_token = jwt.encode(expired_payload, settings.SECRET_KEY, algorithm=ALGORITHM)

    response = await client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {expired_token}"}
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_me_rejects_token_signed_with_wrong_secret(client: AsyncClient):
    forged_payload = {
        "sub": "11111111-1111-1111-1111-111111111111",
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=30),
    }
    forged_token = jwt.encode(forged_payload, "wrong-secret-key", algorithm=ALGORITHM)

    response = await client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {forged_token}"}
    )
    assert response.status_code == 401
