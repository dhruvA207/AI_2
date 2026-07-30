"""Web search.

Three backends, tried in the order ``config/default.yaml`` lists them. Falling through
on an empty result matters more than it sounds: a backend that returns nothing is
indistinguishable, from the caller's side, from a query with no answers — so without
fallthrough a broken backend silently makes ARC look ignorant rather than broken.

**Measured behaviour of each, on 2026-07-29:**

| Backend | robots.txt | Works without a browser? |
|---|---|---|
| ``google`` | disallows ``/search`` | **No.** Serves a JS shell; 3 links, no results. |
| ``bing`` | disallows ``/search`` | **Not reliably** — see below. |
| ``duckduckgo`` | **allows** ``lite/`` | Yes, across every query shape tested. |

Bing deserves the detail because it was requested as the default and tested hard for
it: ~60 requests across four parser revisions, advert filtering, and query
condensation. Once it classifies a client as a bot it returns HTTP 200 with arbitrary
pages and paid placements presented as organic results — "python TypeError unhashable
type list" returned literotica.com, "unhashable type list" returned foxnews.com. That
failure mode is the worst kind: structurally valid, semantically wrong, and impossible
to distinguish from a real answer without knowing the answer already. Adverts are now
filtered on their tracking parameters, which helps but does not rescue it.

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

DEFAULT_BACKENDS = ("duckduckgo", "bing")

#: Words carrying no search signal, stripped by ``condense``. Deliberately small: an
#: aggressive list mangles technical queries, where words like "not" and "in" can be
#: part of the thing being searched for.
_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "between",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "explain",
        "for",
        "from",
        "get",
        "had",
        "has",
        "have",
        "here",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "me",
        "might",
        "must",
        "my",
        "of",
        "on",
        "or",
        "please",
        "shall",
        "should",
        "some",
        "tell",
        "that",
        "the",
        "then",
        "there",
        "these",
        "this",
        "those",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)

#: Bing's practical ceiling before it starts entity-matching the first keyword instead
#: of searching. Measured, not guessed — see the module docstring.
KEYWORD_LIMIT = 3

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
#: Bing's organic results each sit in an ``<li class="b_algo">``. Scoping to that
#: container is essential, not tidiness: matching ``<h2><a>`` across the whole page
#: instead picks up the knowledge panel (``tpmeta``/``b_attribution``) that Bing renders
#: *above* the results. Searching "Barbara Liskov substitution principle" that way
#: returned Wikipedia's "Barbara (given name)", and "capital of Mongolia" returned
#: capitalone.com — plausible-looking results that had nothing to do with the query.
_BING_BLOCK = re.compile(r'<li[^>]+class="[^"]*\bb_algo\b[^"]*"[^>]*>(.*?)</li>', re.DOTALL)
_BING_TITLE = re.compile(r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL)
_BING_SNIPPET = re.compile(r"<p[^>]*>(.*?)</p>", re.DOTALL)
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


def condense(query: str, *, limit: int = KEYWORD_LIMIT) -> str:
    """Reduce a natural-language question to search keywords.

    Bing degrades badly past about three content words: instead of searching, it
    entity-matches the first keyword. "what is the capital of Mongolia" returns
    capitalone.com; "capital of Mongolia" returns Ulaanbaatar. Measured, not guessed.

    Stopwords are dropped and the most specific terms kept. Word *order* is preserved
    rather than reordered by any cleverness, because for technical queries the original
    order usually already puts the specific term last where it belongs.

    Falls back to the original words when stripping would leave nothing — a query that
    is entirely stopwords is better sent as-is than sent empty.
    """
    words = re.findall(r"[\w+#.\-]+", query)
    if not words:
        return query.strip()

    kept = [w for w in words if w.lower() not in _STOPWORDS]
    if not kept:
        kept = words

    # Prefer capitalised and non-lowercase terms — proper nouns and identifiers carry
    # the most search signal. Truncating blindly to the tail dropped "Rust" from
    # "Rust borrow checker definition and purpose", leaving "checker definition
    # purpose" and returning dictionary entries.
    if len(kept) > limit:
        distinctive = [w for w in kept if not w.islower()]
        rest = [w for w in kept if w.islower()]
        ordered = distinctive + rest
        chosen = set(ordered[:limit])
        kept = [w for w in kept if w in chosen][:limit]

    return " ".join(kept)


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


#: Query parameters Microsoft attaches to *paid* placements. Their presence is the one
#: reliable way to tell a Bing ad from an organic result — without this filter, a
#: search for "unhashable type list" returns foxnews.com with HTTP 200 and no error,
#: which is structurally indistinguishable from a real result and so defeats the
#: fallthrough to another backend entirely.
_AD_MARKERS = ("msockid=", "msclkid=", "syndicatedsearch")


def _is_advert(url: str) -> bool:
    """Whether a result URL is a paid placement rather than an organic hit."""
    lowered = url.lower()
    return any(marker in lowered for marker in _AD_MARKERS)


def _parse_bing(html: str, limit: int) -> list[SearchResult]:
    """Parse Bing's organic results from their ``b_algo`` containers.

    Advertising placements are dropped. Bing serves them *as* organic results once it
    decides the client is a bot, and returning them would poison memory with pages
    that have nothing to do with the query.
    """
    results: list[SearchResult] = []
    seen: set[str] = set()

    for block in _BING_BLOCK.findall(html):
        title_match = _BING_TITLE.search(block)
        if not title_match:
            continue

        target = _unwrap(title_match.group(1))
        title = _strip(title_match.group(2))
        if not target.startswith(("http://", "https://")) or not title:
            continue
        if "bing.com" in urllib.parse.urlparse(target).netloc:
            continue  # An internal link that did not decode; not a result.
        if _is_advert(target):
            _log.debug("dropping bing advert: %s", target[:80])
            continue
        if target in seen:
            continue
        seen.add(target)

        snippet_match = _BING_SNIPPET.search(block)
        results.append(
            SearchResult(
                title=title,
                url=target,
                snippet=_strip(snippet_match.group(1))[:300] if snippet_match else "",
                backend="bing",
            )
        )
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
    keywords: bool = False,
) -> list[SearchResult]:
    """Search the web, trying each configured backend until one returns results.

    ``keywords`` condenses the query to search terms first — **off by default**. It was
    built to rescue Bing and did not, and it actively harms good backends: condensing
    "Rust borrow checker definition and purpose" to its last three content words gives
    "checker definition purpose", which drops the subject and returns dictionary
    entries. Opt in only where a backend is known to need it.

    Falls through on an empty result as well as on an error. A backend that returns
    nothing looks exactly like a query with no answers, so without fallthrough a broken
    backend would make ARC appear ignorant instead of appearing broken — and the second
    is far easier to diagnose.
    """
    if not query.strip():
        raise WebError("empty search query")

    client = fetcher or Fetcher()
    order = list(backends or DEFAULT_BACKENDS)
    terms = condense(query) if keywords else query
    if terms != query:
        _log.info("condensed query", extra={"from": query, "to": terms})
    failures: list[str] = []

    for backend in order:
        try:
            results = search_with(backend, terms, limit=limit, fetcher=client)
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
