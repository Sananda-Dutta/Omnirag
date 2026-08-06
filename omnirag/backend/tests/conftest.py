"""
Test fixtures for database-backed tests.

Why a separate test database instead of mocking the DB:
    Mocking SQLAlchemy sessions gives false confidence — it tests that your
    code calls `.add()` and `.commit()`, not that the actual SQL is valid,
    that the foreign key/unique constraints hold, or that the migration
    that created the schema is correct. For a project whose entire point is
    "production-grade," the tests should hit a real Postgres. `omnirag_test`
    (created alongside the dev DB) is that database — never the dev one, so
    tests can never corrupt data you're looking at during development.

Why create/drop tables per test session via metadata instead of running
Alembic in tests: Alembic migrations are tested by actually running them
(we did that manually against the dev DB, and CI will do it again in
Phase 19). For unit/integration tests, `Base.metadata.create_all` against
the test DB is faster and keeps the test suite decoupled from migration
history — what matters here is "do the models + queries behave correctly,"
not "does the migration chain apply cleanly" (that's a separate, explicit
check).
"""

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.database.session import get_db
from app.main import app
from app.models import Base

TEST_DATABASE_URL = "postgresql+asyncpg://omnirag:omnirag@localhost:5432/omnirag_test"


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    # NullPool + a fresh engine per test: pytest-asyncio gives each test
    # function its own event loop, but asyncpg connections are bound to the
    # loop they were created on. A module-level engine (with its pooled,
    # reused connections) ends up handing test #2 a connection that belongs
    # to test #1's already-closed loop -> "another operation is in progress".
    # Creating (and disposing) the engine inside the fixture keeps engine
    # lifetime == event loop lifetime.
    engine = create_async_engine(TEST_DATABASE_URL, poolclass=NullPool)
    session_local = async_sessionmaker(bind=engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_local() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
