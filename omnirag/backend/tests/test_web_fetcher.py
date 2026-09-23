"""
Unit tests for app/ingestion/web_fetcher.py (Phase 12).

These are deliberately fast and deterministic: httpx.MockTransport swaps
out the actual network call, so scheme/host validation, content-type
rejection, and the size cap are all tested without depending on any real
site staying up or reachable. Real-network behavior (an actual page
fetched over the wire) is covered separately in tests/test_url_ingestion.py,
which is explicit about depending on real, allowlisted hosts.
"""

import httpx
import pytest

from app.ingestion.web_fetcher import (
    MAX_CONTENT_BYTES,
    URLFetchError,
    _is_safe_host,
    fetch_and_extract,
)


def _patch_transport(monkeypatch, handler) -> None:
    """Makes fetch_and_extract's internal httpx.AsyncClient(...) construct
    a client wired to `handler` instead of hitting the real network."""
    real_async_client = httpx.AsyncClient

    def fake_async_client(*args, **kwargs):
        kwargs.pop("timeout", None)
        return real_async_client(
            transport=httpx.MockTransport(handler), follow_redirects=kwargs.get("follow_redirects", True)
        )

    monkeypatch.setattr("app.ingestion.web_fetcher.httpx.AsyncClient", fake_async_client)


# --- Scheme / host validation ---------------------------------------------


@pytest.mark.asyncio
async def test_rejects_non_http_scheme():
    with pytest.raises(URLFetchError, match="scheme"):
        await fetch_and_extract("ftp://example.com/file.txt")


@pytest.mark.asyncio
async def test_rejects_javascript_scheme():
    with pytest.raises(URLFetchError, match="scheme"):
        await fetch_and_extract("javascript:alert(1)")


@pytest.mark.parametrize(
    "hostname",
    ["127.0.0.1", "localhost", "169.254.169.254", "10.0.0.5", "192.168.1.1", "::1"],
)
def test_is_safe_host_blocks_private_and_link_local(hostname):
    assert _is_safe_host(hostname) is False


def test_is_safe_host_allows_public_address():
    # 8.8.8.8 (Google DNS) resolves via socket.getaddrinfo without a real
    # network round-trip for the literal-IP case, and is unambiguously
    # public — a stable positive control for the "not blocked" path.
    assert _is_safe_host("8.8.8.8") is True


@pytest.mark.asyncio
async def test_refuses_to_fetch_private_address(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should never reach the network for a private host")

    _patch_transport(monkeypatch, handler)
    with pytest.raises(URLFetchError, match="private/internal address"):
        await fetch_and_extract("http://127.0.0.1:6333/")


@pytest.mark.asyncio
async def test_unresolvable_host_raises_url_fetch_error(monkeypatch):
    with pytest.raises(URLFetchError, match="Could not resolve host"):
        await fetch_and_extract("http://this-host-does-not-exist.invalid/")


# --- Content-type / size handling ------------------------------------------


@pytest.mark.asyncio
async def test_rejects_unsupported_content_type(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.4")

    _patch_transport(monkeypatch, handler)
    with pytest.raises(URLFetchError, match="Unsupported content type"):
        await fetch_and_extract("http://8.8.8.8/file.pdf")


@pytest.mark.asyncio
async def test_rejects_page_over_size_cap(monkeypatch):
    oversized = b"x" * (MAX_CONTENT_BYTES + 1)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/plain"}, content=oversized)

    _patch_transport(monkeypatch, handler)
    with pytest.raises(URLFetchError, match="exceeds"):
        await fetch_and_extract("http://8.8.8.8/huge.txt")


@pytest.mark.asyncio
async def test_raises_on_http_error_status(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, headers={"content-type": "text/html"}, content=b"<html>gone</html>")

    _patch_transport(monkeypatch, handler)
    with pytest.raises(URLFetchError, match="HTTP 404"):
        await fetch_and_extract("http://8.8.8.8/missing")


@pytest.mark.asyncio
async def test_raises_when_extracted_text_is_empty(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=b"<html><head><script>var x=1;</script></head><body>   </body></html>",
        )

    _patch_transport(monkeypatch, handler)
    with pytest.raises(URLFetchError, match="No extractable text"):
        await fetch_and_extract("http://8.8.8.8/empty")


# --- Successful extraction ---------------------------------------------------


@pytest.mark.asyncio
async def test_extracts_title_and_visible_text_from_html(monkeypatch):
    html = b"""
    <html>
      <head><title>Retrieval Augmented Generation</title>
      <style>.x{color:red}</style></head>
      <body>
        <nav>skip this nav link</nav>
        <script>skip this script text</script>
        <p>RAG grounds LLM output in retrieved documents.</p>
        <footer>skip this footer</footer>
      </body>
    </html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, content=html)

    _patch_transport(monkeypatch, handler)
    page = await fetch_and_extract("http://8.8.8.8/article")

    assert page.title == "Retrieval Augmented Generation"
    assert "RAG grounds LLM output" in page.text
    assert "skip this nav link" not in page.text
    assert "skip this script text" not in page.text
    assert "skip this footer" not in page.text


@pytest.mark.asyncio
async def test_extracts_plain_text_content_with_no_title(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/plain"}, content=b"Hybrid search merges dense and sparse."
        )

    _patch_transport(monkeypatch, handler)
    page = await fetch_and_extract("http://8.8.8.8/notes.txt")

    assert page.title is None
    assert page.text == "Hybrid search merges dense and sparse."


@pytest.mark.asyncio
async def test_final_url_reflects_redirect_target(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(302, headers={"location": "http://8.8.8.8/new"})
        return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"moved content")

    _patch_transport(monkeypatch, handler)
    page = await fetch_and_extract("http://8.8.8.8/old")

    assert page.final_url == "http://8.8.8.8/new"
    assert page.text == "moved content"
