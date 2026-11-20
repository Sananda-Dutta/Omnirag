"""
Database engine and session management.

Why async SQLAlchemy instead of sync:
    FastAPI's whole performance model relies on not blocking the event loop.
    If we used sync SQLAlchemy + psycopg2, every DB call would block a worker
    thread, and under load (e.g. someone hammering the /chat endpoint while
    background ingestion is also hitting the DB) that becomes the bottleneck
    long before the LLM calls do. asyncpg + SQLAlchemy's async engine let DB
    I/O yield the event loop like everything else in the app (LLM calls,
    vector search, S3/file I/O later).

Why a single module owns the engine:
    The engine holds a connection pool. Creating more than one engine per
    process means more than one pool fighting over the same DB connection
    limit — an easy way to exhaust `max_connections` in production without
    realizing why. Every other module gets sessions through `get_db()`,
    never by constructing its own engine.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_pre_ping=True,  # detects stale connections (e.g. after DB restart) before using them
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,  # let us still read attributes after commit, without a re-fetch
    autoflush=False,
    autocommit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a request-scoped session.

    Usage:
        @router.get("/documents")
        async def list_documents(db: AsyncSession = Depends(get_db)):
            ...

    The session is closed automatically when the request ends, whether it
    succeeded or raised — that's what the try/finally is for.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def check_db_connection() -> bool:
    """Used by the health endpoint. Cheap: SELECT 1, nothing more."""
    from sqlalchemy import text

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
