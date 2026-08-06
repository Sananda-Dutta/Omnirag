"""
Health check endpoint.

Why a dedicated router for this: /health is what Docker healthchecks, load
balancers, and Kubernetes readiness probes hit in Phase 18-20. It should
never depend on business logic — just proof that the process is alive and
(later) that it can reach its critical dependencies (DB, vector store).

For Phase 1, it only proves the API process itself is up. In Phase 2+ we'll
extend it to check DB connectivity without turning it into a slow endpoint.
"""

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.database.session import check_db_connection

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> JSONResponse:
    db_ok = await check_db_connection()

    body = {
        "status": "ok" if db_ok else "degraded",
        "app": settings.APP_NAME,
        "env": settings.APP_ENV,
        "checks": {"database": "ok" if db_ok else "unreachable"},
    }
    # 200 when everything's healthy, 503 when a dependency is down — this is
    # what lets Docker/Kubernetes healthchecks and load balancers actually
    # act on the result instead of just reading a body they may not parse.
    status_code = status.HTTP_200_OK if db_ok else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(content=body, status_code=status_code)
