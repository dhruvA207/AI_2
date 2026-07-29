"""Research: search, read, distil, and remember.

§4.4's pipeline end to end — fetch, extract main content, strip boilerplate, chunk,
summarise against the query, then write the result into semantic memory *with
provenance* so the same question does not hit the network twice.

The provenance requirement is not decoration. A fact ARC cannot attribute is a fact it
cannot defend when questioned, and one it cannot re-verify when it goes stale. Every
memory written here carries its source URL, the date it was retrieved, and a confidence
below 1.0 — a web page is weaker evidence than something the user said directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from arc.errors import WebError
from arc.log import get_logger
from arc.memory.service import MemoryService
from arc.model.base import LanguageModel, Message
from arc.web.extract import Document, chunk, extract
from arc.web.fetch import Fetcher
from arc.web.search import SearchResult, search

_log = get_logger(__name__)

#: How long a web-sourced fact is trusted before it is worth re-checking. Facts do not
#: all age alike, so the categories below override this.
DEFAULT_TTL_DAYS = 90

#: Time-sensitive topics, re-verified rather than trusted forever (§4.4). Keyed on
#: words that appear in the *query*, since that is what reveals the intent — "latest
#: version of X" ages in days, "who invented X" does not age at all.
_VOLATILE_HINTS = (
    "latest",
    "current",
    "today",
    "now",
    "recent",
    "news",
    "price",
    "version",
    "release",
    "update",
    "stock",
    "weather",
    "score",
    "who is the",
)
_VOLATILE_TTL_DAYS = 1

_SUMMARY_PROMPT = (
    "Extract only the facts from this page that answer the question. Write them as "
    "short, self-contained statements — each must make sense on its own, months from "
    "now, without the page in front of you. Do not add anything the page does not say. "
    "If the page does not answer the question, reply exactly: NO ANSWER."
)


@dataclass
class ResearchResult:
    """What one research run learned."""

    query: str
    summary: str
    sources: list[str] = field(default_factory=list)
    pages_read: int = 0
    memory_ids: list[int] = field(default_factory=list)
    #: True when the answer came from memory rather than the network.
    from_memory: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "query": self.query,
            "summary": self.summary,
            "sources": self.sources,
            "pages_read": self.pages_read,
            "memory_ids": self.memory_ids,
            "from_memory": self.from_memory,
        }


def ttl_for(query: str) -> int:
    """How many days a fact about ``query`` stays trustworthy."""
    lowered = query.lower()
    if any(hint in lowered for hint in _VOLATILE_HINTS):
        return _VOLATILE_TTL_DAYS
    return DEFAULT_TTL_DAYS


def is_stale(retrieved_at: str, ttl_days: int) -> bool:
    """Whether a memory retrieved at ``retrieved_at`` has aged out."""
    try:
        when = datetime.fromisoformat(retrieved_at)
    except (TypeError, ValueError):
        # An unparseable timestamp means we cannot vouch for its age, so re-verify.
        return True
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return datetime.now(UTC) - when > timedelta(days=ttl_days)


class Researcher:
    """Runs the search-read-distil-remember pipeline."""

    def __init__(
        self,
        model: LanguageModel,
        memory: MemoryService | None = None,
        *,
        fetcher: Fetcher | None = None,
        max_pages: int = 3,
        max_chunk_chars: int = 6000,
        backends: list[str] | None = None,
    ) -> None:
        self._model = model
        self._memory = memory
        self._fetcher = fetcher or Fetcher()
        self._backends = backends
        self._max_pages = max_pages
        self._max_chunk_chars = max_chunk_chars

    # ── Memory first ────────────────────────────────────────────────────────────

    def check_memory(self, query: str) -> ResearchResult | None:
        """Answer from memory if a fresh enough fact is already stored.

        The point of write-back: asking the same question twice should not cost a
        network round trip. Staleness is checked per fact, because a memory that was
        true in March is not evidence about today's release version.
        """
        if self._memory is None:
            return None

        hits = self._memory.recall(query, limit=5)
        ttl = ttl_for(query)

        fresh = []
        for hit in hits:
            record = hit.record
            if record.source != "web" or not record.source_url:
                continue
            retrieved = (record.metadata or {}).get("retrieved_at", record.created_at)
            if is_stale(str(retrieved), ttl):
                _log.info("memory is stale, re-researching", extra={"id": record.id})
                continue
            fresh.append(record)

        if not fresh:
            return None

        return ResearchResult(
            query=query,
            summary="\n".join(f"- {r.content}" for r in fresh),
            sources=[r.source_url for r in fresh if r.source_url],
            memory_ids=[r.id for r in fresh],
            from_memory=True,
        )

    # ── The pipeline ────────────────────────────────────────────────────────────

    def read(self, url: str) -> Document:
        """Fetch and extract one page."""
        response = self._fetcher.fetch(url)
        document = extract(response.text, response.final_url or url)

        if document.looks_like_js_shell:
            # §4.4 asks for a headless browser here. Reported rather than silently
            # returning an empty page, so the caller knows the difference between "no
            # content" and "content this fetcher cannot see".
            raise WebError(
                f"{url} renders its content with JavaScript, which a plain fetch "
                "cannot see. A headless browser backend is not installed."
            )
        if document.word_count < 20:
            raise WebError(f"{url} yielded almost no readable text")

        return document

    def distil(self, query: str, document: Document) -> str:
        """Summarise a page against the query, keeping only what answers it."""
        pieces = chunk(document.text, size=self._max_chunk_chars)
        if not pieces:
            return ""

        # Only the first chunks are read. A 40-page document would otherwise cost
        # dozens of model calls, and the answer to a specific question is very rarely
        # buried past the opening sections.
        summaries: list[str] = []
        for piece in pieces[:3]:
            completion = self._model.generate(
                [
                    Message(role="system", content=_SUMMARY_PROMPT),
                    Message(
                        role="user",
                        content=f"Question: {query}\n\nPage: {document.title}\n\n{piece}",
                    ),
                ],
                max_tokens=400,
                temperature=0.2,
            )
            text = completion.text.strip()
            if text and "NO ANSWER" not in text.upper():
                summaries.append(text)

        return "\n".join(summaries)

    def remember(self, query: str, summary: str, url: str, title: str) -> list[int]:
        """Write distilled facts into semantic memory with provenance."""
        if self._memory is None or not summary.strip():
            return []

        stored: list[int] = []
        for line in summary.splitlines():
            fact = line.strip().lstrip("-*•").strip()
            # Skip fragments: a two-word line is not a self-contained fact and will
            # only pollute retrieval later.
            if len(fact.split()) < 5:
                continue

            memory_id = self._memory.semantic.add_fact(
                fact,
                confidence=0.7,
                source="web",
                source_url=url,
                metadata={
                    "retrieved_at": datetime.now(UTC).isoformat(),
                    "query": query,
                    "page_title": title,
                    "ttl_days": ttl_for(query),
                },
            )
            stored.append(memory_id)

        return stored

    def research(self, query: str, *, use_memory: bool = True) -> ResearchResult:
        """Answer a question, from memory if possible and the web otherwise."""
        if use_memory:
            cached = self.check_memory(query)
            if cached is not None:
                _log.info("answered from memory", extra={"query": query})
                return cached

        results = search(
            query, limit=self._max_pages * 2, fetcher=self._fetcher, backends=self._backends
        )
        if not results:
            return ResearchResult(query=query, summary="No search results found.")

        summaries: list[str] = []
        sources: list[str] = []
        memory_ids: list[int] = []
        pages = 0

        for result in results:
            if pages >= self._max_pages:
                break
            try:
                document = self.read(result.url)
            except WebError as exc:
                # One unreadable page must not end the research; try the next result.
                _log.info("skipping %s: %s", result.url, exc)
                continue

            distilled = self.distil(query, document)
            if not distilled:
                continue

            pages += 1
            summaries.append(f"From {document.title or result.url}:\n{distilled}")
            sources.append(result.url)
            memory_ids.extend(self.remember(query, distilled, result.url, document.title))

        if not summaries:
            return ResearchResult(
                query=query,
                summary="Found search results but could not read anything useful from them.",
                sources=[r.url for r in results[:3]],
            )

        return ResearchResult(
            query=query,
            summary="\n\n".join(summaries),
            sources=sources,
            pages_read=pages,
            memory_ids=memory_ids,
        )


def preview(results: list[SearchResult]) -> str:
    """Render search results as text for the model or the CLI."""
    lines = []
    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. {result.title}\n   {result.url}")
        if result.snippet:
            lines.append(f"   {result.snippet[:200]}")
    return "\n".join(lines)
