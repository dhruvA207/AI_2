"""Web tools for the agent.

Thin wrappers over ``arc.web``. They exist so the agent can search and read pages the
same way it reads files — one registry, one dispatch path, one audit trail.

``web_research`` is the one that writes back to memory. The others are read-only, so
they still run under ``--dry-run``: looking something up changes nothing.
"""

from __future__ import annotations

from arc.errors import ToolError, WebError
from arc.log import get_logger
from arc.tools.registry import tool
from arc.web.extract import extract
from arc.web.fetch import Fetcher
from arc.web.research import preview
from arc.web.search import search as run_search

_log = get_logger(__name__)

#: One fetcher for the process, so the rate limiter and robots.txt cache are shared.
#: A fresh fetcher per call would re-download robots.txt every time and defeat the
#: per-host delay entirely.
_fetcher = Fetcher()


@tool(category="web")
def web_search(query: str, limit: int = 6) -> str:
    """Search the web and return ranked results with titles, URLs, and snippets.

    Args:
        query: What to search for.
        limit: How many results to return.
    """
    try:
        results = run_search(query, limit=limit, fetcher=_fetcher)
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
        response = _fetcher.fetch(url)
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
        response = _fetcher.fetch(url)
    except WebError as exc:
        raise ToolError(str(exc)) from exc

    links = extract(response.text, response.final_url or url).links[:limit]
    if not links:
        return f"no outbound links found in {url}"
    return "\n".join(links)
