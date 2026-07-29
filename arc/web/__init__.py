"""Web research.

The first part of ARC that sends anything off the machine. Everything else is local by
construction; this fetches pages, which necessarily reveals to the host what is being
read. So: robots.txt is respected, requests are rate-limited per host, the user agent
identifies itself honestly, and every fetch is audited.

Zero third-party dependencies, which was not the plan. A licence audit of the obvious
stack (requests + trafilatura) found GPL and MPL packages pulled in transitively, which
§0.1 forbids — so fetching, robots.txt, and content extraction are all built on the
standard library. See docs/DECISIONS.md ADR-017.
"""

from arc.web.extract import Document, chunk, extract
from arc.web.fetch import Fetcher, RateLimiter, Response, RobotsCache
from arc.web.research import Researcher, ResearchResult
from arc.web.search import SearchResult, search

__all__ = [
    "Document",
    "Fetcher",
    "RateLimiter",
    "ResearchResult",
    "Researcher",
    "Response",
    "RobotsCache",
    "SearchResult",
    "chunk",
    "extract",
    "search",
]
