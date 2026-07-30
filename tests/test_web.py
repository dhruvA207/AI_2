"""Tests for fetching, extraction, search parsing, and the staleness policy.

Nothing here touches the network. Fetching is tested through its guards and its
decoder; extraction and search parsing run against fixture HTML. A test suite that
needs the internet is a test suite that fails on a train.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from arc.errors import WebError
from arc.web.extract import chunk, extract
from arc.web.fetch import Fetcher, RateLimiter, _decode
from arc.web.research import is_stale, ttl_for
from arc.web.search import (
    SearchResult,
    _is_advert,
    _parse_bing,
    _parse_google,
    _unwrap,
    condense,
    search_with,
)

PAGE = """<html><head><title>  Rust Ownership  </title></head><body>
<nav class="site-nav"><a href="/">Home</a><a href="/about">About</a></nav>
<div id="cookie-banner">We use cookies. Accept or decline?</div>
<main><article class="post-content">
<h1>Rust Ownership</h1>
<p>Ownership is Rust's most distinctive feature, and it enables memory safety
guarantees without needing a garbage collector at runtime.</p>
<p>Each value has one owner, and when that owner goes out of scope, the value is
dropped automatically by the compiler.</p>
<p>Read the <a href="https://doc.rust-lang.org/book/">official book</a> for detail.</p>
<script>var tracker = 1;</script>
</article></main>
<aside class="related"><a href="/other">Another post</a></aside>
<footer class="site-footer">Copyright 2026. All rights reserved.</footer>
</body></html>"""


# ── Extraction ──────────────────────────────────────────────────────────────────


def test_title_is_extracted_and_trimmed() -> None:
    assert extract(PAGE).title == "Rust Ownership"


def test_main_content_survives() -> None:
    text = extract(PAGE).text
    assert "most distinctive feature" in text
    assert "one owner" in text


@pytest.mark.parametrize("boilerplate", ["cookies", "Copyright", "Another post", "Home", "tracker"])
def test_boilerplate_is_stripped(boilerplate: str) -> None:
    """Boilerplate stored as a fact is worse than no fact: it retrieves later and
    looks authoritative."""
    assert boilerplate not in extract(PAGE).text


def test_script_and_style_never_appear() -> None:
    html = "<html><body><p>Real text here for the body.</p>"
    html += "<style>.x{color:red}</style><script>alert(1)</script></body></html>"
    text = extract(html).text
    assert "color:red" not in text
    assert "alert" not in text


def test_content_links_are_collected() -> None:
    assert "https://doc.rust-lang.org/book/" in extract(PAGE).links


def test_boilerplate_links_are_not_collected() -> None:
    assert not any("/other" in link for link in extract(PAGE).links)


def test_semantic_tags_beat_the_wrapping_body() -> None:
    """Regression: the outermost element always wins on raw text length, so <body>
    was chosen over the <article> it contained and boilerplate leaked through."""
    document = extract(PAGE)
    assert "Copyright" not in document.text
    assert document.word_count < 80


def test_js_shell_is_detected() -> None:
    """§4.4 wants a headless browser for these; detecting them is the prerequisite."""
    assert extract('<html><body><div id="root"></div></body></html>').looks_like_js_shell


def test_real_content_is_not_flagged_as_a_shell() -> None:
    assert not extract(PAGE).looks_like_js_shell


def test_malformed_html_is_salvaged() -> None:
    """Unclosed tags are the norm on the real web, not the exception."""
    html = "<html><body><div><p>Some genuine content that should survive parsing."
    assert "genuine content" in extract(html).text


def test_empty_input() -> None:
    document = extract("")
    assert document.text == ""
    assert document.title == ""


def test_entities_are_decoded() -> None:
    html = (
        "<html><body><article><p>Caf&eacute; &amp; croissants "
        "for everyone here.</p></article></body></html>"
    )
    assert "Café & croissants" in extract(html).text


# ── Chunking ────────────────────────────────────────────────────────────────────


def test_short_text_is_one_chunk() -> None:
    assert chunk("short text") == ["short text"]


def test_empty_text_produces_no_chunks() -> None:
    assert chunk("") == []


def test_long_text_is_split() -> None:
    text = "\n\n".join(f"Paragraph {i} with several words in it." for i in range(80))
    chunks = chunk(text, size=500, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) < 700 for c in chunks)


def test_chunks_overlap() -> None:
    """A fact straddling a boundary would otherwise be lost to both chunks."""
    text = "\n\n".join(f"Paragraph {i} here." for i in range(60))
    chunks = chunk(text, size=300, overlap=100)
    assert any(chunks[0].split("\n\n")[-1] in chunks[1] for _ in [0]), (
        "expected the tail of one chunk to open the next"
    )


# ── Fetching guards ─────────────────────────────────────────────────────────────


def test_non_http_schemes_are_refused() -> None:
    """file:// through the web tool would be an unaudited local file read."""
    fetcher = Fetcher(respect_robots=False)
    for url in ("file:///etc/passwd", "ftp://example.com/x", "javascript:alert(1)"):
        with pytest.raises(WebError, match="non-http"):
            fetcher.fetch(url)


def test_malformed_urls_are_refused() -> None:
    with pytest.raises(WebError, match="not a valid URL"):
        Fetcher(respect_robots=False).fetch("https://")


def test_rate_limiter_delays_the_same_host() -> None:
    import time

    limiter = RateLimiter(delay=0.15)
    limiter.wait("example.com")
    started = time.monotonic()
    limiter.wait("example.com")
    assert time.monotonic() - started >= 0.1


def test_rate_limiter_does_not_delay_different_hosts() -> None:
    import time

    limiter = RateLimiter(delay=0.5)
    limiter.wait("a.com")
    started = time.monotonic()
    limiter.wait("b.com")
    assert time.monotonic() - started < 0.1


class _Headers(dict[str, str]):
    """Stands in for an HTTPMessage."""


def test_decode_uses_the_header_charset() -> None:
    body = "café".encode("latin-1")
    headers = _Headers({"Content-Type": "text/html; charset=latin-1"})
    assert _decode(body, headers) == "café"


def test_decode_falls_back_to_a_meta_charset() -> None:
    """Servers lie about encoding often enough that the header alone gives mojibake."""
    body = b'<meta charset="latin-1"><p>caf\xe9</p>'
    assert "café" in _decode(body, _Headers({"Content-Type": "text/html"}))


def test_decode_handles_gzip() -> None:
    import gzip

    body = gzip.compress(b"hello")
    headers = _Headers({"Content-Encoding": "gzip", "Content-Type": "text/html"})
    assert _decode(body, headers) == "hello"


def test_decode_never_raises_on_bad_bytes() -> None:
    assert _decode(b"\xff\xfe\x00bad", _Headers({"Content-Type": "text/html"}))


# ── Search result parsing ───────────────────────────────────────────────────────


def test_redirect_wrapper_is_unwrapped() -> None:
    """Results are wrapped in //duckduckgo.com/l/?uddg=... and would otherwise all
    point at the search engine rather than the page."""
    href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fdoc.rust-lang.org%2Fbook%2F&rut=abc"
    assert _unwrap(href) == "https://doc.rust-lang.org/book/"


def test_direct_urls_pass_through() -> None:
    assert _unwrap("https://example.com/x") == "https://example.com/x"


def test_protocol_relative_urls_get_a_scheme() -> None:
    assert _unwrap("//example.com/x") == "https://example.com/x"


# ── Staleness policy ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "query",
    ["latest version of Python", "current weather", "recent news", "todays price"],
)
def test_volatile_queries_get_a_short_ttl(query: str) -> None:
    """§4.4: time-sensitive categories are re-verified, not trusted forever."""
    assert ttl_for(query) == 1


@pytest.mark.parametrize("query", ["who invented the transistor", "how does photosynthesis work"])
def test_stable_queries_get_a_long_ttl(query: str) -> None:
    assert ttl_for(query) > 30


def test_fresh_memory_is_not_stale() -> None:
    assert not is_stale(datetime.now(UTC).isoformat(), 90)


def test_old_memory_is_stale() -> None:
    old = (datetime.now(UTC) - timedelta(days=100)).isoformat()
    assert is_stale(old, 90)


def test_the_same_age_differs_by_category() -> None:
    """A fact true in March is not evidence about today's release version."""
    age = (datetime.now(UTC) - timedelta(days=5)).isoformat()
    assert is_stale(age, ttl_for("latest version of Python"))
    assert not is_stale(age, ttl_for("who invented the transistor"))


def test_unparseable_timestamps_are_treated_as_stale() -> None:
    """If we cannot vouch for a fact's age, re-verify rather than assume."""
    assert is_stale("not a date", 90)
    assert is_stale("", 90)


def test_naive_timestamps_are_assumed_utc() -> None:
    naive = (datetime.now(UTC) - timedelta(days=1)).replace(tzinfo=None).isoformat()
    assert not is_stale(naive, 90)


# ── Search backends ─────────────────────────────────────────────────────────────


def test_bing_redirect_is_base64_decoded() -> None:
    """Bing wraps results in ck/a?...&u=a1<base64url>; without decoding, every result
    would point at bing.com rather than the page."""
    href = (
        "https://www.bing.com/ck/a?!&&p=abc&u=a1"
        "aHR0cHM6Ly9ibG9nLmxvZ3JvY2tldC5jb20vaW50cm9kdWNpbmctcnVzdC1ib3Jyb3ctY2hlY2tlci8&ntb=1"
    )
    assert _unwrap(href) == "https://blog.logrocket.com/introducing-rust-borrow-checker/"


def test_malformed_bing_payload_does_not_raise() -> None:
    assert _unwrap("https://www.bing.com/ck/a?u=a1!!!notbase64!!!")


def test_bing_results_are_parsed() -> None:
    # Must sit inside a b_algo container: results outside one are knowledge-panel or
    # sidebar entries, not organic results.
    html = (
        '<li class="b_algo"><h2><a href="https://www.bing.com/ck/a?u=a1'
        'aHR0cHM6Ly9leGFtcGxlLmNvbS9wYWdl&ntb=1">Example Page Title</a></h2></li>'
    )
    results = _parse_bing(html, limit=5)
    assert len(results) == 1
    assert results[0].url == "https://example.com/page"
    assert results[0].backend == "bing"


def test_bing_internal_links_are_not_results() -> None:
    html = '<li class="b_algo"><h2><a href="https://www.bing.com/images/search?q=x">I</a></h2></li>'
    assert _parse_bing(html, limit=5) == []


def test_google_backends_require_the_robots_override() -> None:
    """Both disallow /search, so they are unreachable while compliance is on."""
    polite = Fetcher(respect_robots=True)
    for backend in ("google", "bing"):
        with pytest.raises(WebError, match=r"robots\.txt"):
            search_with(backend, "anything", fetcher=polite)


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(WebError, match="unknown search backend"):
        search_with("altavista", "anything")


def test_google_parser_survives_a_javascript_shell() -> None:
    """Measured 2026-07-29: Google serves non-browsers a shell with no results. The
    parser must return nothing rather than inventing results from chrome links."""
    shell = '<html><body><div id="main"><a href="https://support.google.com/x">Help</a>'
    shell += '<a href="/preferences">Settings</a></div></body></html>'
    assert _parse_google(shell, limit=5) == []


def test_search_falls_through_an_empty_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """A backend returning nothing looks identical to a query with no answers, so
    without fallthrough a broken backend makes ARC look ignorant, not broken."""
    # `import arc.web.search` returns the *function*, not the module: arc/web/__init__
    # re-exports `search`, which rebinds the name on the package. importlib reaches the
    # submodule regardless.
    import importlib

    search_module = importlib.import_module("arc.web.search")
    calls: list[str] = []

    def fake(backend: str, query: str, **_: object) -> list:
        calls.append(backend)
        if backend == "working":
            return [SearchResult(title="t", url="https://x.dev", backend=backend)]
        return []

    monkeypatch.setattr(search_module, "search_with", fake)
    results = search_module.search("q", backends=["empty", "working"])
    assert calls == ["empty", "working"]
    assert results[0].backend == "working"


def test_search_reports_when_every_backend_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    search_module = importlib.import_module("arc.web.search")
    monkeypatch.setattr(search_module, "search_with", lambda *a, **k: [])
    with pytest.raises(WebError, match="no search backend returned results"):
        search_module.search("q", backends=["a", "b"])


def test_empty_query_is_rejected() -> None:
    from arc.web.search import search as run

    with pytest.raises(WebError, match="empty search query"):
        run("   ")


def test_bing_adverts_are_dropped() -> None:
    """Bing serves paid placements *as* organic results once it decides the client is
    a bot. Returning them would poison memory with pages unrelated to the query, and
    they are structurally identical to real results — the tracking parameter is the
    only reliable tell."""
    import base64

    def block(url: str) -> str:
        encoded = base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")
        return f'<li class="b_algo"><h2><a href="https://www.bing.com/ck/a?u=a1{encoded}">T</a></h2></li>'

    html = block("https://example.com/real") + block("https://ads.example/x?msockid=abc123")
    results = _parse_bing(html, limit=5)
    assert [r.url for r in results] == ["https://example.com/real"]


@pytest.mark.parametrize("marker", ["msockid=abc", "msclkid=def", "syndicatedsearch.x/y"])
def test_advert_markers_are_recognised(marker: str) -> None:
    assert _is_advert(f"https://example.com/page?{marker}")


def test_organic_urls_are_not_flagged_as_adverts() -> None:
    assert not _is_advert("https://stackoverflow.com/questions/13675296/x")


def test_bing_results_come_from_organic_containers_only() -> None:
    """Regression: matching <h2><a> across the page picked up the knowledge panel Bing
    renders above the results, so "Barbara Liskov substitution principle" returned
    Wikipedia's "Barbara (given name)"."""
    html = (
        '<div class="tpmeta"><h2><a href="https://en.wikipedia.org/wiki/Wrong">Panel</a></h2></div>'
        '<li class="b_algo"><h2><a href="https://example.com/right">Right</a></h2></li>'
    )
    assert [r.url for r in _parse_bing(html, limit=5)] == ["https://example.com/right"]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("what is the capital of Mongolia", "capital Mongolia"),
        ("rust borrow checker", "rust borrow checker"),
        ("Rust borrow checker definition and purpose", "Rust borrow checker"),
    ],
)
def test_condense_keeps_the_distinctive_terms(query: str, expected: str) -> None:
    """Regression: truncating to the last N words dropped the subject. "Rust borrow
    checker definition and purpose" became "checker definition purpose", which returned
    dictionary entries. Proper nouns and identifiers carry the search signal."""
    assert condense(query) == expected


def test_condense_preserves_word_order() -> None:
    """Reordering terms changes what a search engine matches on."""
    assert condense("how do I fix a segfault in C using malloc").split()[0] == "fix"


def test_condense_falls_back_when_everything_is_a_stopword() -> None:
    """An all-stopword query is better sent as-is than sent empty."""
    assert condense("what is the") == "what is the"


def test_condense_on_empty_input() -> None:
    assert condense("   ") == ""
