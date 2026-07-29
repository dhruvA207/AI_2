"""Web search.

Three backends, tried in the order ``config/default.yaml`` lists them. Falling through
on an empty result matters more than it sounds: a backend that returns nothing is
indistinguishable, from the caller's side, from a query with no answers — so without
fallthrough a broken backend silently makes ARC look ignorant rather than broken.

**Measured behaviour of each, on 2026-07-29:**

| Backend | robots.txt | Works without a browser? |
|---|---|---|
| ``google`` | disallows ``/search`` | **No.** Serves a JS shell; 3 links, no results. |
| ``bing`` | disallows ``/search`` | Yes. Results wrapped in ``bing.com/ck/a?...&u=a1<base64>``. |
| ``duckduckgo`` | **allows** ``lite/`` | Yes. |

Google and Bing are only attempted when ``web.respect_robots`` is turned off in config,
since both disallow ``/search``. That switch is the user's to throw — §0.3 is explicit
that ARC does not negotiate access on their behalf — but it defaults to on, and ARC
never impersonates a browser to evade bot detection. A site that blocks a
self-identifying agent is one that does not want to be read by one.
"""

from __future__ import annotations

import base64
import binascii
import html as html_module
import re
import urllib.parse
from dataclasses import dataclass
from typing import Any

from arc.errors import WebError
from arc.log import get_logger
from arc.web.fetch import Fetcher

_log = get_logger(__name__)

ENDPOINTS = {
    "google": "https://www.google.com/search",
    "bing": "https://www.bing.com/search",
    "duckduckgo": "https://lite.duckduckgo.com/lite/",
}

#: Backends that need robots.txt compliance disabled, because their robots.txt
#: disallows the search path outright.
REQUIRES_ROBOTS_OVERRIDE = frozenset({"google", "bing"})

DEFAULT_BACKENDS = ("google", "bing", "duckduckgo")

#: Anchors are matched in two passes rather than one pattern. DuckDuckGo's markup uses
#: single quotes for class and double for href, and puts href *before* class — a
#: pattern fixing either the quote style or the attribute order matches nothing.
_ANCHOR = re.compile(r"<a\s([^>]*)>(.*?)</a>", re.DOTALL | re.IGNORECASE)
_HREF = re.compile(r"""href\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
_CLASS = re.compile(r"""class\s*=\s*["']([^"']*)["']""", re.IGNORECASE)

_DDG_SNIPPET = re.compile(
    r"""<td[^>]*class\s*=\s*["'][^"']*result-snippet[^"']*["'][^>]*>(.*?)</td>""",
    re.DOTALL | re.IGNORECASE,
)
_BING_RESULT = re.compile(r"<h2[^>]*>\s*<a[^>]+href=\"([^\"]+)\"[^>]*>(.*?)</a>", re.DOTALL)
_TAGS = re.compile(r"<[^>]+>")


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One search hit."""

    title: str
    url: str
    snippet: str = ""
    #: Which backend produced it, so a caller can tell where an answer came from.
    backend: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "backend": self.backend,
        }


def _strip(markup: str) -> str:
    """Turn a fragment of result HTML into plain text."""
    return html_module.unescape(_TAGS.sub("", markup)).strip()


def _unwrap(href: str) -> str:
    """Recover the destination from a search engine's redirect wrapper.

    DuckDuckGo percent-encodes it in ``uddg``; Bing base64url-encodes it in ``u`` behind
    an ``a1`` prefix. Both would otherwise leave every result pointing at the search
    engine rather than the page.
    """
    if href.startswith("//"):
        href = "https:" + href
    href = html_module.unescape(href)

    parsed = urllib.parse.urlparse(href)
    params = urllib.parse.parse_qs(parsed.query)

    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = params.get("uddg", [""])[0]
        if target:
            return urllib.parse.unquote(target)

    if "bing.com" in parsed.netloc and parsed.path.startswith("/ck/"):
        encoded = params.get("u", [""])[0]
        if encoded.startswith("a1"):
            payload = encoded[2:]
            try:
                # Restore the padding base64url encoders strip.
                return base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)).decode("utf-8")
            except (binascii.Error, UnicodeDecodeError, ValueError):
                return href

    return href


def _parse_duckduckgo(html: str, limit: int) -> list[SearchResult]:
    """Parse DuckDuckGo's lite results."""
    snippets = [_strip(s) for s in _DDG_SNIPPET.findall(html)]
    results: list[SearchResult] = []
    index = 0

    for attributes, label in _ANCHOR.findall(html):
        classes = _CLASS.search(attributes)
        if not classes or "result-link" not in classes.group(1):
            continue
        href = _HREF.search(attributes)
        if not href:
            continue

        target = _unwrap(href.group(1))
        title = _strip(label)
        if not target.startswith(("http://", "https://")) or not title:
            continue

        results.append(
            SearchResult(
                title=title,
                url=target,
                snippet=snippets[index] if index < len(snippets) else "",
                backend="duckduckgo",
            )
        )
        index += 1
        if len(results) >= limit:
            break
    return results


def _parse_bing(html: str, limit: int) -> list[SearchResult]:
    """Parse Bing's results, which live in ``<h2><a>`` blocks."""
    results: list[SearchResult] = []
    for href, label in _BING_RESULT.findall(html):
        target = _unwrap(href)
        title = _strip(label)
        if not target.startswith(("http://", "https://")) or not title:
            continue
        if "bing.com" in urllib.parse.urlparse(target).netloc:
            continue  # An internal link that did not decode; not a result.
        results.append(SearchResult(title=title, url=target, backend="bing"))
        if len(results) >= limit:
            break
    return results


def _parse_google(html: str, limit: int) -> list[SearchResult]:
    """Parse Google's results.

    Kept deliberately simple because as of 2026-07-29 it has nothing to parse: Google
    serves a JavaScript shell to any client that is not a browser, and the response
    contains three links, none of them results. Extracting anything would require
    impersonating a browser and executing JavaScript, which this project does not do.

    Left in so the backend is selectable and its failure is visible rather than
    mysterious — and so it starts working by itself if Google ever serves plain HTML.
    """
    results: list[SearchResult] = []
    for attributes, label in _ANCHOR.findall(html):
        href = _HREF.search(attributes)
        if not href:
            continue
        target = href.group(1)
        if target.startswith("/url?"):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(target).query)
            target = query.get("q", [""])[0]
        if not target.startswith(("http://", "https://")):
            continue
        host = urllib.parse.urlparse(target).netloc
        if "google." in host or "gstatic" in host or "googleusercontent" in host:
            continue
        title = _strip(label)
        if not title:
            continue
        results.append(SearchResult(title=title, url=target, backend="google"))
        if len(results) >= limit:
            break
    return results


_PARSERS = {
    "google": _parse_google,
    "bing": _parse_bing,
    "duckduckgo": _parse_duckduckgo,
}


def search_with(
    backend: str, query: str, *, limit: int = 8, fetcher: Fetcher | None = None
) -> list[SearchResult]:
    """Search using one named backend."""
    if backend not in ENDPOINTS:
        raise WebError(f"unknown search backend {backend!r}; expected one of {sorted(ENDPOINTS)}")

    client = fetcher or Fetcher()
    if backend in REQUIRES_ROBOTS_OVERRIDE and client.respect_robots:
        raise WebError(
            f"{backend} disallows /search in its robots.txt. Set web.respect_robots "
            "to false in config to query it anyway."
        )

    url = ENDPOINTS[backend] + "?" + urllib.parse.urlencode({"q": query})
    response = client.fetch(url)
    return _PARSERS[backend](response.text, limit)


def search(
    query: str,
    *,
    limit: int = 8,
    fetcher: Fetcher | None = None,
    backends: tuple[str, ...] | list[str] | None = None,
) -> list[SearchResult]:
    """Search the web, trying each configured backend until one returns results.

    Falls through on an empty result as well as on an error. A backend that returns
    nothing looks exactly like a query with no answers, so without fallthrough a broken
    backend would make ARC appear ignorant instead of appearing broken — and the second
    is far easier to diagnose.
    """
    if not query.strip():
        raise WebError("empty search query")

    client = fetcher or Fetcher()
    order = list(backends or DEFAULT_BACKENDS)
    failures: list[str] = []

    for backend in order:
        try:
            results = search_with(backend, query, limit=limit, fetcher=client)
        except WebError as exc:
            failures.append(f"{backend}: {exc}")
            _log.info("search backend %s unavailable: %s", backend, exc)
            continue

        if results:
            _log.info("searched", extra={"backend": backend, "results": len(results)})
            return results

        failures.append(f"{backend}: returned no parseable results")
        _log.info("search backend %s returned nothing, trying the next", backend)

    raise WebError("no search backend returned results — " + "; ".join(failures))
