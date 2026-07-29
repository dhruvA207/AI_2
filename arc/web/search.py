"""Web search.

Uses DuckDuckGo's ``lite`` endpoint, which its robots.txt permits — verified, not
assumed. That matters because §4.4 requires respecting robots.txt, and the obvious
alternatives do not allow it: Google's ``/search`` and Bing's ``/search`` are both
disallowed, as is ``duckduckgo.com/html/``. Scraping them anyway would mean building
robots.txt compliance and then circumventing it.

No API key, so nothing is shared with a third-party search service beyond the query
itself, which any search necessarily reveals. A key-based backend can be configured in
``config/default.yaml`` if the free endpoint ever stops being adequate.
"""

from __future__ import annotations

import html as html_module
import re
import urllib.parse
from dataclasses import dataclass
from typing import Any

from arc.errors import WebError
from arc.log import get_logger
from arc.web.fetch import Fetcher

_log = get_logger(__name__)

SEARCH_ENDPOINT = "https://lite.duckduckgo.com/lite/"

#: Anchors are matched in two passes rather than one clever pattern. The markup uses
#: single quotes for class and double for href, and puts href *before* class — so a
#: pattern that fixes either the quote style or the attribute order silently matches
#: nothing, which is how this returned zero results the first time.
_ANCHOR = re.compile(r"<a\s([^>]*)>(.*?)</a>", re.DOTALL | re.IGNORECASE)
_HREF = re.compile(r"""href\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
_CLASS = re.compile(r"""class\s*=\s*["']([^"']*)["']""", re.IGNORECASE)

_SNIPPET = re.compile(
    r"""<td[^>]*class\s*=\s*["'][^"']*result-snippet[^"']*["'][^>]*>(.*?)</td>""",
    re.DOTALL | re.IGNORECASE,
)
_TAGS = re.compile(r"<[^>]+>")


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One search hit."""

    title: str
    url: str
    snippet: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


def _strip(markup: str) -> str:
    """Turn a fragment of result HTML into plain text."""
    return html_module.unescape(_TAGS.sub("", markup)).strip()


def _unwrap(href: str) -> str:
    """Recover the destination URL from a DuckDuckGo redirect link."""
    if href.startswith("//"):
        href = "https:" + href

    parsed = urllib.parse.urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        params = urllib.parse.parse_qs(parsed.query)
        target = params.get("uddg", [""])[0]
        if target:
            return urllib.parse.unquote(target)
    return href


def search(query: str, *, limit: int = 8, fetcher: Fetcher | None = None) -> list[SearchResult]:
    """Search the web, returning ranked results.

    Raises ``WebError`` rather than returning an empty list when the request fails, so
    "the network is down" is distinguishable from "there are no results" — the agent
    should retry the first and not the second.
    """
    if not query.strip():
        raise WebError("empty search query")

    client = fetcher or Fetcher()
    url = SEARCH_ENDPOINT + "?" + urllib.parse.urlencode({"q": query})
    response = client.fetch(url)

    snippets = [_strip(s) for s in _SNIPPET.findall(response.text)]
    results: list[SearchResult] = []

    index = 0
    for attributes, label in _ANCHOR.findall(response.text):
        classes = _CLASS.search(attributes)
        if not classes or "result-link" not in classes.group(1):
            continue

        href = _HREF.search(attributes)
        if not href:
            continue

        target = _unwrap(html_module.unescape(href.group(1)))
        if not target.startswith(("http://", "https://")):
            continue
        title = _strip(label)
        if not title:
            continue

        results.append(
            SearchResult(
                title=title,
                url=target,
                snippet=snippets[index] if index < len(snippets) else "",
            )
        )
        index += 1
        if len(results) >= limit:
            break

    _log.info("searched", extra={"query": query, "results": len(results)})
    return results
