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
from pathlib import Path

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


@pytest.fixture(scope="module")
def celery_worker():
    """
    A real Celery worker subprocess, for tests that need to prove ingestion
    actually works end-to-end (Redis broker -> worker process -> DB write),
    not just that the task function is correct in isolation.

    Started with DATABASE_URL pointed at the test DB explicitly: a subprocess
    has no access to the in-process `client` fixture's dependency override,
    so it needs its own, matching env var to read/write the same rows the
    test's API calls created.
    """
    import os
    import subprocess
    import time

    env = os.environ.copy()
    env["DATABASE_URL"] = TEST_DATABASE_URL
    # REDIS_URL deliberately left as-is: the worker must share the same
    # broker this test process enqueues tasks on via the API.

    proc = subprocess.Popen(
        [
            "celery",
            "-A",
            "app.workers.celery_app",
            "worker",
            "--loglevel=info",
            "--concurrency=1",
            "--without-heartbeat",
            "--without-gossip",
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    started = time.time()
    ready = False
    lines: list[str] = []
    while time.time() - started < 20:
        line = proc.stdout.readline()
        if not line:
            continue
        lines.append(line)
        if "ready" in line.lower():
            ready = True
            break

    if not ready:
        proc.terminate()
        raise RuntimeError("Celery worker did not become ready in time:\n" + "".join(lines))

    yield proc

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
