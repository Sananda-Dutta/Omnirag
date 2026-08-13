"""
Application entrypoint.

Kept deliberately thin: this file wires together config, logging, and
routers. It should never contain business logic — that's the whole point of
the app/services, app/rag, app/llm split. If this file grows past ~100 lines,
that's a signal something belongs in a service instead.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth import router as auth_router
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.knowledge_bases import router as knowledge_bases_router
from app.api.search import router as search_router
from app.core.config import settings
from app.core.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.is_production and settings.SECRET_KEY == "CHANGE_ME_IN_PRODUCTION":
        # Booting with the placeholder secret in production means anyone who
        # reads this open-source repo can forge a valid JWT for any user ID.
        # Fail loudly at startup rather than serving traffic insecurely.
        raise RuntimeError(
            "SECRET_KEY is still the default placeholder. Set a real, random "
            "SECRET_KEY before running in production."
        )

    logger.info(
        "app_startup",
        extra={"context": {"env": settings.APP_ENV, "debug": settings.DEBUG}},
    )
    yield
    logger.info("app_shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        description="Multi-Modal AI Knowledge Assistant — RAG over PDFs, DOCX, images, and URLs.",
        version="0.1.0",
        debug=settings.DEBUG,
        lifespan=lifespan,
    )

    # CORS is wide open for local dev only. This gets locked down to specific
    # origins in Phase 19 (security) before anything touches production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router, prefix=settings.API_V1_PREFIX)
    app.include_router(auth_router, prefix=settings.API_V1_PREFIX)
    app.include_router(knowledge_bases_router, prefix=settings.API_V1_PREFIX)
    app.include_router(documents_router, prefix=settings.API_V1_PREFIX)
    app.include_router(search_router, prefix=settings.API_V1_PREFIX)

    return app


app = create_app()
