"""Deep research: multi-query, multi-source, corroborated.

The shallow pipeline in ``research.py`` runs one search, reads three pages, and
summarises. That is fine for a lookup and inadequate for a question worth researching:
one search reflects one phrasing, and a fact appearing on one page is one page's claim,
not a finding.

This module adds the parts that make the difference:

1. **Query decomposition.** The model turns a question into several sub-queries, so
   retrieval is not hostage to one phrasing.
2. **Wide gathering.** Every sub-query is searched, results are pooled and deduped by
   URL *and* by domain, so one prolific site cannot dominate the evidence.
3. **Corroboration.** Extracted claims are clustered by embedding similarity, and
   confidence scales with the number of *independent domains* asserting them. This is
   the accuracy mechanism: a claim two unrelated sites agree on is worth more than one
   a single site states confidently.
4. **Gap-filling.** After the first round the model names what is still unanswered,
   and those become the next round's queries.
5. **Grounded synthesis.** The final answer is written only from the corroborated
   findings, each carrying its sources.

Disagreement is surfaced rather than averaged away. Two sources contradicting each
other is information, and silently picking one would be the single most damaging thing
this module could do.
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from arc.errors import WebError
from arc.log import get_logger
from arc.memory.service import MemoryService
from arc.model.base import LanguageModel, Message
from arc.web.extract import Document, chunk, extract
from arc.web.fetch import Fetcher
from arc.web.research import ttl_for
from arc.web.search import SearchResult, search

_log = get_logger(__name__)

#: Cosine similarity above which two claims are treated as the same assertion. Lower
#: than the memory deduplication threshold (0.97) on purpose: two sources phrase the
#: same fact differently, and demanding near-identity would count genuine
#: corroboration as two separate single-source claims.
CORROBORATION_THRESHOLD = 0.86

#: Confidence for a claim only one domain asserts. Deliberately below the 0.7 that
#: shallow research assigns, because a deep run had the opportunity to corroborate and
#: did not — that is weaker evidence, not equal evidence.
_SINGLE_SOURCE_CONFIDENCE = 0.55
_CORROBORATED_CONFIDENCE = 0.8
_STRONGLY_CORROBORATED_CONFIDENCE = 0.92

_PLAN_PROMPT = """\
Break this research question into {count} distinct web search queries that together \
would answer it. Cover different angles: definitions, mechanisms, comparisons, and \
concrete specifics.

Rules:
- One query per line, nothing else. No numbering, no commentary.
- Each query is 3-8 words, phrased as a search, not a question.
- Make them genuinely different from each other.
"""

_EXTRACT_PROMPT = """\
Extract the factual claims from this page that bear on the question.

Rules:
- One claim per line, starting with "- ".
- Each claim must be self-contained and understandable months from now without this \
page in front of you.
- Only state what the page actually says. Do not infer, extrapolate, or add outside \
knowledge.
- Skip anything that is opinion, marketing, or navigation text.
- If the page says nothing relevant, reply exactly: NO CLAIMS.
"""

_GAP_PROMPT = """\
Given the original question and what has been learned so far, what is still \
unanswered?

Reply with up to {count} web search queries that would close the gaps, one per line, \
3-8 words each, nothing else. If the question is fully answered, reply exactly: DONE.
"""

_SYNTHESIS_PROMPT = """\
Answer the question using only the findings below.

Rules:
- Do not add anything the findings do not support.
- Where findings disagree, say so explicitly rather than choosing one.
- Note where support is thin (a single source).
- Be direct and concise. No preamble.
"""


@dataclass
class Finding:
    """One claim and the independent sources asserting it."""

    claim: str
    sources: list[str] = field(default_factory=list)
    embedding: list[float] | None = None

    @property
    def domains(self) -> set[str]:
        """Distinct domains backing this claim.

        Domains, not URLs: three pages on one site are one site's position, and
        counting them as three would manufacture corroboration that does not exist.
        """
        return {urllib.parse.urlparse(s).netloc.removeprefix("www.") for s in self.sources}

    @property
    def support(self) -> int:
        """How many independent domains assert this."""
        return len(self.domains)

    @property
    def confidence(self) -> float:
        """Confidence, scaled by independent corroboration."""
        if self.support >= 3:
            return _STRONGLY_CORROBORATED_CONFIDENCE
        if self.support == 2:
            return _CORROBORATED_CONFIDENCE
        return _SINGLE_SOURCE_CONFIDENCE

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "claim": self.claim,
            "sources": self.sources,
            "domains": sorted(self.domains),
            "support": self.support,
            "confidence": round(self.confidence, 2),
        }


@dataclass
class DeepResult:
    """The outcome of a deep research run."""

    question: str
    answer: str
    findings: list[Finding] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    pages_read: int = 0
    rounds: int = 0
    memory_ids: list[int] = field(default_factory=list)

    @property
    def sources(self) -> list[str]:
        """Every distinct source consulted, most-corroborating first."""
        seen: list[str] = []
        for finding in sorted(self.findings, key=lambda f: -f.support):
            for source in finding.sources:
                if source not in seen:
                    seen.append(source)
        return seen

    @property
    def corroborated(self) -> list[Finding]:
        """Findings backed by more than one independent domain."""
        return [f for f in self.findings if f.support > 1]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "question": self.question,
            "answer": self.answer,
            "queries": self.queries,
            "pages_read": self.pages_read,
            "rounds": self.rounds,
            "findings": [f.to_dict() for f in self.findings],
            "corroborated_count": len(self.corroborated),
            "sources": self.sources,
            "memory_ids": self.memory_ids,
        }


def _lines(text: str, *, limit: int) -> list[str]:
    """Pull clean lines out of a model reply, dropping numbering and bullets."""
    out: list[str] = []
    for raw in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", raw).strip().strip('"')
        if not line or line.upper().startswith(("DONE", "NO CLAIMS", "NONE")):
            continue
        if len(line.split()) < 2:
            continue
        out.append(line)
        if len(out) >= limit:
            break
    return out


class DeepResearcher:
    """Multi-round, multi-source research with corroboration."""

    def __init__(
        self,
        model: LanguageModel,
        memory: MemoryService | None = None,
        *,
        fetcher: Fetcher | None = None,
        backends: list[str] | None = None,
        queries_per_round: int = 3,
        pages_per_round: int = 4,
        rounds: int = 2,
    ) -> None:
        self._model = model
        self._memory = memory
        self._fetcher = fetcher or Fetcher()
        self._backends = backends
        self._queries_per_round = queries_per_round
        self._pages_per_round = pages_per_round
        self._rounds = rounds

    # ── Planning ────────────────────────────────────────────────────────────────

    def plan(self, question: str) -> list[str]:
        """Decompose a question into distinct search queries.

        Falls back to the question itself if the model produces nothing usable — a
        degraded search beats no search.
        """
        completion = self._model.generate(
            [
                Message(
                    role="system",
                    content=_PLAN_PROMPT.format(count=self._queries_per_round),
                ),
                Message(role="user", content=question),
            ],
            max_tokens=200,
            temperature=0.4,
        )
        queries = _lines(completion.text, limit=self._queries_per_round)
        return queries or [question]

    def gaps(self, question: str, findings: list[Finding]) -> list[str]:
        """Ask what is still unanswered, and turn it into follow-up queries."""
        if not findings:
            return []

        known = "\n".join(f"- {f.claim}" for f in findings[:25])
        completion = self._model.generate(
            [
                Message(role="system", content=_GAP_PROMPT.format(count=2)),
                Message(role="user", content=f"Question: {question}\n\nLearned so far:\n{known}"),
            ],
            max_tokens=150,
            temperature=0.4,
        )
        if "DONE" in completion.text.upper()[:40]:
            return []
        return _lines(completion.text, limit=2)

    # ── Gathering ───────────────────────────────────────────────────────────────

    def gather(self, queries: list[str], *, exclude: set[str]) -> list[SearchResult]:
        """Search every query and pool the results, deduped by URL and domain.

        Capped at two pages per domain. Without that, a site that ranks well for every
        sub-query supplies most of the evidence, and its claims then look corroborated
        when they are one source repeated.
        """
        pooled: list[SearchResult] = []
        seen_urls = set(exclude)
        per_domain: dict[str, int] = {}

        for query in queries:
            try:
                results = search(
                    query,
                    limit=self._pages_per_round,
                    fetcher=self._fetcher,
                    backends=self._backends,
                    # Sub-queries are already short and specific; condensing them
                    # again strips the subject.
                    keywords=False,
                )
            except WebError as exc:
                _log.info("search failed for %r: %s", query, exc)
                continue

            for result in results:
                if result.url in seen_urls:
                    continue
                domain = urllib.parse.urlparse(result.url).netloc.removeprefix("www.")
                if per_domain.get(domain, 0) >= 2:
                    continue
                seen_urls.add(result.url)
                per_domain[domain] = per_domain.get(domain, 0) + 1
                pooled.append(result)

        return pooled

    def read(self, url: str) -> Document | None:
        """Fetch and extract a page, or None if it cannot be read."""
        try:
            response = self._fetcher.fetch(url)
        except WebError as exc:
            _log.info("could not fetch %s: %s", url, exc)
            return None

        document = extract(response.text, response.final_url or url)
        if document.looks_like_js_shell or document.word_count < 30:
            return None
        return document

    def claims(self, question: str, document: Document) -> list[str]:
        """Extract grounded factual claims from one page."""
        pieces = chunk(document.text, size=6000)
        if not pieces:
            return []

        found: list[str] = []
        # One chunk per page. On a laptop-sized local model the extraction budget is
        # the dominant cost of a deep run — a second chunk roughly doubles wall-clock
        # for claims that are usually restatements of the first.
        for piece in pieces[:1]:
            completion = self._model.generate(
                [
                    Message(role="system", content=_EXTRACT_PROMPT),
                    Message(
                        role="user",
                        content=f"Question: {question}\n\nPage: {document.title}\n\n{piece}",
                    ),
                ],
                max_tokens=256,
                temperature=0.2,
            )
            found.extend(_lines(completion.text, limit=6))
        return found

    # ── Corroboration ───────────────────────────────────────────────────────────

    def corroborate(self, claims: list[tuple[str, str]]) -> list[Finding]:
        """Cluster equivalent claims and attribute each to its sources.

        Clustering is by embedding similarity, which is what lets "the borrow checker
        prevents data races" and "data races are impossible under the borrow checker"
        count as one corroborated finding rather than two isolated ones.

        Without an embedder this degrades to exact-text matching, which still works —
        it just corroborates less.
        """
        findings: list[Finding] = []
        embedder = self._memory.embedder if self._memory is not None else None

        vectors: list[list[float] | None] = [None] * len(claims)
        if embedder is not None and claims:
            try:
                vectors = list(embedder.embed_batch([c for c, _ in claims]))
            except Exception as exc:
                _log.warning("could not embed claims, falling back to text match: %s", exc)

        for (claim, source), vector in zip(claims, vectors, strict=False):
            match = self._nearest(findings, claim, vector)
            if match is not None:
                if source not in match.sources:
                    match.sources.append(source)
            else:
                findings.append(Finding(claim=claim, sources=[source], embedding=vector))

        # Best-supported first, so synthesis and storage both see the strongest
        # evidence at the top.
        findings.sort(key=lambda f: (-f.support, len(f.claim)))
        return findings

    @staticmethod
    def _nearest(findings: list[Finding], claim: str, vector: list[float] | None) -> Finding | None:
        """Return an existing finding asserting the same thing, if any."""
        normalized = claim.lower().strip(" .")

        if vector is None:
            for finding in findings:
                if finding.claim.lower().strip(" .") == normalized:
                    return finding
            return None

        best: Finding | None = None
        best_score = CORROBORATION_THRESHOLD
        for finding in findings:
            if finding.embedding is None:
                continue
            score = sum(a * b for a, b in zip(vector, finding.embedding, strict=False))
            if score > best_score:
                best, best_score = finding, score
        return best

    # ── Synthesis and storage ───────────────────────────────────────────────────

    def synthesize(self, question: str, findings: list[Finding]) -> str:
        """Write the answer from the findings, flagging thin and conflicting support."""
        if not findings:
            return "No usable information was found."

        rendered = "\n".join(
            f"- {f.claim} [{f.support} source{'s' if f.support != 1 else ''}: "
            f"{', '.join(sorted(f.domains))}]"
            for f in findings[:30]
        )
        completion = self._model.generate(
            [
                Message(role="system", content=_SYNTHESIS_PROMPT),
                Message(role="user", content=f"Question: {question}\n\nFindings:\n{rendered}"),
            ],
            max_tokens=600,
            temperature=0.3,
        )
        return completion.text.strip()

    def remember(self, question: str, findings: list[Finding]) -> list[int]:
        """Store findings, with confidence reflecting how well corroborated they are."""
        if self._memory is None:
            return []

        stored: list[int] = []
        for finding in findings:
            if len(finding.claim.split()) < 5:
                continue
            memory_id = self._memory.semantic.add_fact(
                finding.claim,
                confidence=finding.confidence,
                source="web",
                source_url=finding.sources[0],
                metadata={
                    "retrieved_at": datetime.now(UTC).isoformat(),
                    "query": question,
                    "ttl_days": ttl_for(question),
                    "corroborating_sources": finding.sources,
                    "support": finding.support,
                },
            )
            stored.append(memory_id)
        return stored

    # ── The run ─────────────────────────────────────────────────────────────────

    def research(self, question: str, *, on_progress: Any = None, store: bool = True) -> DeepResult:
        """Research a question across several queries, rounds, and sources."""

        def report(message: str) -> None:
            if on_progress is not None:
                on_progress(message)

        all_claims: list[tuple[str, str]] = []
        all_queries: list[str] = []
        seen_urls: set[str] = set()
        pages_read = 0
        findings: list[Finding] = []
        rounds_done = 0

        queries = self.plan(question)
        report(f"planned {len(queries)} queries: {'; '.join(queries)}")

        for round_number in range(1, self._rounds + 1):
            if not queries:
                break
            rounds_done = round_number
            all_queries.extend(queries)

            results = self.gather(queries, exclude=seen_urls)
            report(f"round {round_number}: {len(results)} new pages")

            for result in results[: self._pages_per_round]:
                seen_urls.add(result.url)
                document = self.read(result.url)
                if document is None:
                    continue
                extracted = self.claims(question, document)
                if not extracted:
                    continue
                pages_read += 1
                all_claims.extend((claim, result.url) for claim in extracted)
                report(f"  read {result.url[:60]} — {len(extracted)} claims")

            findings = self.corroborate(all_claims)
            report(
                f"round {round_number}: {len(findings)} findings, "
                f"{sum(1 for f in findings if f.support > 1)} corroborated"
            )

            if round_number < self._rounds:
                queries = self.gaps(question, findings)
                if queries:
                    report(f"following up: {'; '.join(queries)}")

        answer = self.synthesize(question, findings)
        memory_ids = self.remember(question, findings) if store else []

        return DeepResult(
            question=question,
            answer=answer,
            findings=findings,
            queries=all_queries,
            pages_read=pages_read,
            rounds=rounds_done,
            memory_ids=memory_ids,
        )
