"""HTTP fetching, with robots.txt compliance and rate limiting.

**Zero dependencies, deliberately.** The obvious choice was ``requests`` plus
``trafilatura``, and a licence audit of the resulting tree killed it: trafilatura is
Apache-2.0 itself, but pulls ``courlan`` which pulls ``tld``, which is tri-licensed
MPL-1.1 / GPL-2.0 / LGPL-2.1. §0.1 forbids GPL outright, so the whole branch is out.
``requests`` in turn pulls ``certifi`` (MPL-2.0), which is not Apache or MIT either.

The stdlib covers this: ``urllib.request`` fetches, ``urllib.robotparser`` handles
robots.txt (which §4.4 requires anyway), and Python's ``ssl`` module uses the system
trust store. The cost is roughly a hundred lines of wrapper. See docs/DECISIONS.md
ADR-017.

This is also the first ARC module that sends anything off the machine. Every request is
audited, rate-limited per host, and refuses to fetch what robots.txt disallows.
"""

from __future__ import annotations

import gzip
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
import zlib
from dataclasses import dataclass, field
from typing import Any

from arc.errors import WebError
from arc.log import get_logger

_log = get_logger(__name__)

#: Honest identification, with a contact path. Pretending to be a browser to evade
#: bot detection is exactly the "detection evasion" this project has no business doing,
#: and a site that blocks a declared agent is a site that does not want to be read.
USER_AGENT = "ARC/0.1 (local personal assistant; +https://github.com/dhruvA207/ARC)"

DEFAULT_TIMEOUT = 20.0

#: Minimum seconds between requests to the same host. Politeness, and it also stops a
#: confused agent from hammering one site in a loop.
DEFAULT_DELAY = 1.0

#: Refuse to download more than this. A stray link to an ISO should not fill the disk.
MAX_BYTES = 5_000_000

_TEXTUAL = ("text/html", "text/plain", "application/xhtml", "application/json", "text/xml")


@dataclass(frozen=True, slots=True)
class Response:
    """A fetched document."""

    url: str
    status: int
    content_type: str
    text: str
    #: The URL actually fetched, after redirects. Provenance must record where the
    #: content really came from, not where we started.
    final_url: str = ""
    elapsed_ms: float = 0.0

    @property
    def is_html(self) -> bool:
        """Whether this looks like HTML."""
        return "html" in self.content_type

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "url": self.url,
            "final_url": self.final_url or self.url,
            "status": self.status,
            "content_type": self.content_type,
            "chars": len(self.text),
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


class RateLimiter:
    """Enforces a minimum delay between requests to the same host."""

    def __init__(self, delay: float = DEFAULT_DELAY) -> None:
        self._delay = delay
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, host: str) -> None:
        """Block until it is polite to call ``host`` again."""
        with self._lock:
            previous = self._last.get(host, 0.0)
            elapsed = time.monotonic() - previous
            remaining = self._delay - elapsed
            if remaining > 0:
                time.sleep(remaining)
            self._last[host] = time.monotonic()


class RobotsCache:
    """Fetches and caches robots.txt per host.

    Cached because §4.4 requires respecting robots.txt, and re-fetching it before every
    page would triple the request count — which would be its own kind of impolite.
    """

    def __init__(self, *, timeout: float = 10.0) -> None:
        self._parsers: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._timeout = timeout
        self._lock = threading.Lock()

    def allowed(self, url: str, *, user_agent: str = USER_AGENT) -> bool:
        """Whether ``url`` may be fetched.

        Fails *open*: a robots.txt that cannot be fetched or parsed is treated as
        permitting access, which is the conventional reading. Failing closed would make
        every site with a flaky robots.txt permanently unreadable.
        """
        parts = urllib.parse.urlparse(url)
        if parts.scheme not in ("http", "https"):
            return False

        origin = f"{parts.scheme}://{parts.netloc}"
        with self._lock:
            if origin not in self._parsers:
                self._parsers[origin] = self._load(origin)
            parser = self._parsers[origin]

        if parser is None:
            return True
        try:
            return bool(parser.can_fetch(user_agent, url))
        except Exception:
            return True

    def _load(self, origin: str) -> urllib.robotparser.RobotFileParser | None:
        """Fetch and parse one robots.txt, or None if unavailable."""
        parser = urllib.robotparser.RobotFileParser()
        target = f"{origin}/robots.txt"
        try:
            request = urllib.request.Request(target, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(
                request, timeout=self._timeout, context=_ssl_context()
            ) as handle:
                body = handle.read(MAX_BYTES).decode("utf-8", errors="replace")
        except Exception as exc:
            _log.debug("no robots.txt for %s: %s", origin, exc)
            return None

        parser.parse(body.splitlines())
        return parser


def _ssl_context() -> ssl.SSLContext:
    """Return a verifying SSL context using the system trust store.

    Avoids ``certifi`` (MPL-2.0) — Python's default context already validates against
    the platform's certificates.
    """
    return ssl.create_default_context()


def _decode(raw: bytes, headers: Any) -> str:
    """Decompress and decode a response body.

    Charset detection order: the Content-Type header, then a meta charset in the first
    kilobyte, then UTF-8 with replacement. Servers lie about encoding often enough that
    trusting the header alone produces mojibake on real pages.
    """
    encoding = (headers.get("Content-Encoding") or "").lower()
    if "gzip" in encoding:
        with__ = gzip.decompress(raw)
        raw = with__
    elif "deflate" in encoding:
        raw = zlib.decompress(raw, -zlib.MAX_WBITS)

    charset = None
    content_type = headers.get("Content-Type") or ""
    if "charset=" in content_type:
        charset = content_type.split("charset=")[-1].split(";")[0].strip().strip("\"'")

    if not charset:
        head = raw[:1024].decode("ascii", errors="ignore").lower()
        if "charset=" in head:
            candidate = head.split("charset=")[-1]
            charset = candidate.split('"')[0].split("'")[0].split(">")[0].split(";")[0].strip()

    for attempt in (charset, "utf-8", "latin-1"):
        if not attempt:
            continue
        try:
            return raw.decode(attempt)
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode("utf-8", errors="replace")


@dataclass
class Fetcher:
    """Fetches URLs politely."""

    timeout: float = DEFAULT_TIMEOUT
    respect_robots: bool = True
    limiter: RateLimiter = field(default_factory=RateLimiter)
    robots: RobotsCache = field(default_factory=RobotsCache)

    def fetch(self, url: str) -> Response:
        """Fetch a URL, returning its decoded text."""
        parts = urllib.parse.urlparse(url)
        if parts.scheme not in ("http", "https"):
            raise WebError(f"refusing to fetch non-http(s) URL: {url}")
        if not parts.netloc:
            raise WebError(f"not a valid URL: {url}")

        if self.respect_robots and not self.robots.allowed(url):
            raise WebError(f"robots.txt disallows fetching {url}")

        self.limiter.wait(parts.netloc)
        started = time.perf_counter()

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Accept-Language": "en",
            },
        )

        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout, context=_ssl_context()
            ) as handle:
                content_type = (handle.headers.get("Content-Type") or "").lower()
                if not any(kind in content_type for kind in _TEXTUAL) and content_type:
                    raise WebError(f"not a textual document ({content_type}): {url}")

                raw = handle.read(MAX_BYTES + 1)
                if len(raw) > MAX_BYTES:
                    raise WebError(f"document exceeds {MAX_BYTES} bytes: {url}")

                text = _decode(raw, handle.headers)
                final_url = handle.geturl()
                status = handle.status
        except urllib.error.HTTPError as exc:
            raise WebError(f"HTTP {exc.code} fetching {url}") from exc
        except urllib.error.URLError as exc:
            raise WebError(f"could not reach {url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise WebError(f"timed out after {self.timeout}s fetching {url}") from exc
        except WebError:
            raise
        except Exception as exc:
            raise WebError(f"could not fetch {url}: {exc}") from exc

        elapsed = (time.perf_counter() - started) * 1000
        _log.info("fetched", extra={"url": url, "status": status, "ms": round(elapsed)})

        return Response(
            url=url,
            status=status,
            content_type=content_type,
            text=text,
            final_url=final_url,
            elapsed_ms=elapsed,
        )
