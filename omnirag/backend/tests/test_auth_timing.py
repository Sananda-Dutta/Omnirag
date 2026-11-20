"""
Timing side-channel regression test.

Guards against re-introducing the bug caught during Phase 2 review: if
`authenticate_user` for an unknown email returns without ever calling
bcrypt, response time leaks whether an email is registered. This measures
both paths directly against the service function (not over HTTP, to avoid
network/ASGI overhead swamping the signal) and asserts they're close.
"""

import time

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.auth_service import authenticate_user, register_user


@pytest.mark.asyncio
async def test_login_timing_does_not_leak_account_existence(db_session: AsyncSession):
    await register_user(db_session, email="exists@example.com", password="correct-horse-battery")

    async def timed(email: str, password: str) -> float:
        start = time.perf_counter()
        await authenticate_user(db_session, email=email, password=password)
        return time.perf_counter() - start

    # Known email, wrong password
    known_wrong = await timed("exists@example.com", "wrong-password-here")
    # Unknown email entirely
    unknown = await timed("nobody-at-all@example.com", "wrong-password-here")

    # Both paths run one bcrypt comparison, so they should be in the same
    # ballpark. A wide gap (e.g. 10x) would indicate the dummy-hash
    # comparison isn't actually happening on the unknown-email path.
    ratio = max(known_wrong, unknown) / min(known_wrong, unknown)
    assert ratio < 3, (
        f"timing gap too large (known_wrong={known_wrong:.4f}s, "
        f"unknown={unknown:.4f}s, ratio={ratio:.2f}) — "
        "unknown-email path may be skipping the bcrypt comparison"
    )
