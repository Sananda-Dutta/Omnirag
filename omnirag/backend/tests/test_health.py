"""
Health check test — now covers the DB-connectivity branch added in Phase 2.
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check_reports_ok_when_db_reachable(client: AsyncClient):
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["checks"]["database"] == "ok"
