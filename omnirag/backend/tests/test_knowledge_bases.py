import pytest
from httpx import AsyncClient


async def _register_and_login(client: AsyncClient, email: str) -> str:
    await client.post("/api/v1/auth/register", json={"email": email, "password": "correct-horse-battery"})
    login = await client.post(
        "/api/v1/auth/login", data={"username": email, "password": "correct-horse-battery"}
    )
    return login.json()["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_create_and_list_knowledge_base(client: AsyncClient):
    token = await _register_and_login(client, "kbowner@example.com")

    create = await client.post(
        "/api/v1/knowledge-bases",
        json={"name": "ML coursework", "description": "lecture notes and papers"},
        headers=_auth(token),
    )
    assert create.status_code == 201
    kb_id = create.json()["id"]

    listing = await client.get("/api/v1/knowledge-bases", headers=_auth(token))
    assert listing.status_code == 200
    assert any(kb["id"] == kb_id for kb in listing.json())


@pytest.mark.asyncio
async def test_knowledge_base_requires_auth(client: AsyncClient):
    response = await client.get("/api/v1/knowledge-bases")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_user_cannot_see_another_users_knowledge_base(client: AsyncClient):
    token_a = await _register_and_login(client, "isoowner@example.com")
    token_b = await _register_and_login(client, "isointruder@example.com")

    create = await client.post(
        "/api/v1/knowledge-bases", json={"name": "private KB"}, headers=_auth(token_a)
    )
    kb_id = create.json()["id"]

    # User B tries to fetch user A's KB directly by ID
    response = await client.get(f"/api/v1/knowledge-bases/{kb_id}", headers=_auth(token_b))
    assert response.status_code == 404  # not 403 — see knowledge_base_service docstring

    # And user B's own list doesn't include it
    listing = await client.get("/api/v1/knowledge-bases", headers=_auth(token_b))
    assert all(kb["id"] != kb_id for kb in listing.json())


@pytest.mark.asyncio
async def test_delete_knowledge_base(client: AsyncClient):
    token = await _register_and_login(client, "kbdelete@example.com")
    create = await client.post(
        "/api/v1/knowledge-bases", json={"name": "temp"}, headers=_auth(token)
    )
    kb_id = create.json()["id"]

    delete = await client.delete(f"/api/v1/knowledge-bases/{kb_id}", headers=_auth(token))
    assert delete.status_code == 204

    get_after = await client.get(f"/api/v1/knowledge-bases/{kb_id}", headers=_auth(token))
    assert get_after.status_code == 404
