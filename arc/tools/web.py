"""Web tools for the agent.

Thin wrappers over ``arc.web``. They exist so the agent can search and read pages the
same way it reads files — one registry, one dispatch path, one audit trail.

``web_research`` is the one that writes back to memory. The others are read-only, so
they still run under ``--dry-run``: looking something up changes nothing.
"""

from __future__ import annotations

from typing import Any

from arc.errors import ToolError, WebError
from arc.log import get_logger
from arc.tools.registry import tool
from arc.web.extract import extract
from arc.web.fetch import Fetcher, RateLimiter
from arc.web.research import preview
from arc.web.search import search as run_search

_log = get_logger(__name__)

#: One fetcher for the process, so the rate limiter and robots.txt cache are shared.
#: A fresh fetcher per call would re-download robots.txt every time and defeat the
#: per-host delay entirely.
_fetcher: Fetcher | None = None
_backends: list[str] | None = None
_condense: bool = True


def configure(config: Any) -> None:
    """Apply the ``web`` config section.

    Called once at startup. Until it is, the defaults apply — robots.txt respected and
    the standard backend order — so a caller that forgets to configure gets the
    conservative behaviour rather than an unconfigured one.
    """
    global _fetcher, _backends

    section = config.section("web")
    search_config = section.get("search") or {}

    _fetcher = Fetcher(
        timeout=float(section.get("timeout", 20.0)),
        respect_robots=bool(section.get("respect_robots", True)),
        limiter=RateLimiter(float(section.get("rate_limit_delay", 1.0))),
    )
    _backends = list(search_config.get("backends") or [])
    global _condense
    _condense = bool(search_config.get("condense_queries", True))

    if not _fetcher.respect_robots:
        # Loud, once, at startup. Silently ignoring robots.txt is the kind of thing
        # that should never be a surprise when reading the audit log later.
        _log.warning(
            "robots.txt compliance is DISABLED (web.respect_robots=false); "
            "google and bing search are reachable"
        )


def _client() -> Fetcher:
    """Return the shared fetcher, building a default one if unconfigured."""
    global _fetcher
    if _fetcher is None:
        _fetcher = Fetcher()
    return _fetcher


@tool(category="web")
def web_search(query: str, limit: int = 6) -> str:
    """Search the web and return ranked results with titles, URLs, and snippets.

    Args:
        query: What to search for.
        limit: How many results to return.
    """
    try:
        results = run_search(
            query, limit=limit, fetcher=_client(), backends=_backends, keywords=_condense
        )
    except WebError as exc:
        raise ToolError(str(exc)) from exc

    if not results:
        return f"no results for {query!r}"
    return preview(results)


@tool(category="web")
def web_fetch(url: str, max_chars: int = 8000) -> str:
    """Fetch a web page and return its main content as clean text.

    Args:
        url: Page to read.
        max_chars: Truncate the extracted text at this length.
    """
    try:
        response = _client().fetch(url)
    except WebError as exc:
        raise ToolError(str(exc)) from exc

    document = extract(response.text, response.final_url or url)

    if document.looks_like_js_shell:
        raise ToolError(
            f"{url} renders its content with JavaScript, which a plain fetch cannot "
            "see. Try a different source."
        )

    body = document.text
    if len(body) > max_chars:
        body = body[:max_chars] + f"\n\n[truncated at {max_chars} of {len(body)} chars]"

    header = f"# {document.title}\n{response.final_url or url}\n\n" if document.title else ""
    return header + body


@tool(category="web")
def web_links(url: str, limit: int = 30) -> str:
    """List the outbound links in a page's main content.

    Args:
        url: Page to read.
        limit: How many links to return.
    """
    try:
        response = _client().fetch(url)
    except WebError as exc:
        raise ToolError(str(exc)) from exc

    links = extract(response.text, response.final_url or url).links[:limit]
    if not links:
        return f"no outbound links found in {url}"
    return "\n".join(links)
