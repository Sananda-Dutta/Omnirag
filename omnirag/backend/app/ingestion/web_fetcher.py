"""
Web page fetcher for Phase 12 URL ingestion.

Security note: this fetches user-supplied URLs server-side, which is a
classic SSRF vector (a user could point this at http://169.254.169.254/
or an internal service). This rejects non-http(s) schemes and blocks
loopback/link-local/private IP ranges by resolving the hostname before
fetching. Not a complete SSRF defense (doesn't stop DNS rebinding between
the check and the request), but a real first line of defense, documented
as a known gap for Phase 19 hardening — not a placeholder pretending to
be complete.
"""

import ipaddress
import socket
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup

MAX_CONTENT_BYTES = 10 * 1024 * 1024  # 10 MB, same spirit as file upload limits
FETCH_TIMEOUT_SECONDS = 15.0


class URLFetchError(Exception):
    """'Could not safely or successfully retrieve usable text from this
    URL' — blocked scheme, blocked target, network failure, unsupported
    content type, or empty extracted text. Caught in the worker the same
    way ExtractionError is caught for file uploads."""


@dataclass
class FetchedPage:
    text: str
    title: str | None
    final_url: str


def _is_safe_host(hostname: str) -> bool:
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise URLFetchError(f"Could not resolve host: {hostname}") from exc

    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


async def fetch_and_extract(url: str) -> FetchedPage:
    parsed = httpx.URL(url)
    if parsed.scheme not in ("http", "https"):
        raise URLFetchError(f"Unsupported URL scheme: {parsed.scheme!r} — only http/https allowed.")

    if not _is_safe_host(parsed.host):
        raise URLFetchError(f"Refusing to fetch from a private/internal address: {parsed.host}")

    content_type = ""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=FETCH_TIMEOUT_SECONDS) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if "text/html" not in content_type and "text/plain" not in content_type:
                    raise URLFetchError(
                        f"Unsupported content type {content_type!r} — only HTML/plain text supported."
                    )

                chunks, total = [], 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > MAX_CONTENT_BYTES:
                        raise URLFetchError(f"Page exceeds the {MAX_CONTENT_BYTES // (1024*1024)}MB fetch limit.")
                    chunks.append(chunk)
                raw = b"".join(chunks)
                final_url = str(response.url)
    except httpx.HTTPStatusError as exc:
        raise URLFetchError(f"Fetch failed with HTTP {exc.response.status_code}") from exc
    except httpx.RequestError as exc:
        raise URLFetchError(f"Network error fetching URL: {exc}") from exc

    if "text/html" in content_type:
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        title = soup.title.string.strip() if soup.title and soup.title.string else None
    else:
        text = raw.decode("utf-8", errors="replace")
        title = None

    if not text.strip():
        raise URLFetchError("No extractable text content found at this URL.")

    return FetchedPage(text=text, title=title, final_url=final_url)