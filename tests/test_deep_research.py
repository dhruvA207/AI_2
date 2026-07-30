"""Tests for Research Mode: corroboration, domain independence, and confidence.

The accuracy mechanism is that confidence scales with how many *independent* domains
assert a claim. These tests pin that down, because getting it subtly wrong would make
ARC confidently repeat one site's opinion as established fact.
"""

from __future__ import annotations

import pytest

from arc.web.deep import (
    CORROBORATION_THRESHOLD,
    DeepResearcher,
    DeepResult,
    Finding,
    _lines,
)
from arc.web.extract import _is_boilerplate, extract
from tests.fakes import FakeModel

# ── Finding and corroboration arithmetic ────────────────────────────────────────


def test_single_source_is_low_confidence() -> None:
    finding = Finding(claim="x", sources=["https://a.com/1"])
    assert finding.support == 1
    assert finding.confidence < 0.6


def test_two_domains_raise_confidence() -> None:
    finding = Finding(claim="x", sources=["https://a.com/1", "https://b.com/2"])
    assert finding.support == 2
    assert finding.confidence > 0.7


def test_three_domains_raise_it_further() -> None:
    finding = Finding(claim="x", sources=["https://a.com/1", "https://b.com/2", "https://c.com/3"])
    assert finding.support == 3
    assert finding.confidence > 0.9


def test_multiple_pages_on_one_site_are_one_source() -> None:
    """Three pages on one domain are one site's position. Counting them separately
    would manufacture corroboration that does not exist."""
    finding = Finding(
        claim="x",
        sources=["https://a.com/1", "https://a.com/2", "https://a.com/3"],
    )
    assert finding.support == 1
    assert finding.confidence < 0.6


def test_www_prefix_does_not_split_a_domain() -> None:
    finding = Finding(claim="x", sources=["https://www.a.com/1", "https://a.com/2"])
    assert finding.support == 1


def test_confidence_never_reaches_certainty() -> None:
    """Nothing learned from the web is as good as something the user said."""
    finding = Finding(claim="x", sources=[f"https://s{i}.com/" for i in range(10)])
    assert finding.confidence < 1.0


# ── Clustering ──────────────────────────────────────────────────────────────────


@pytest.fixture
def researcher() -> DeepResearcher:
    return DeepResearcher(FakeModel(), None)


def test_identical_claims_merge_without_an_embedder(researcher: DeepResearcher) -> None:
    """Without an embedder this degrades to exact matching, which still corroborates."""
    findings = researcher.corroborate(
        [("The sky is blue.", "https://a.com/1"), ("the sky is blue", "https://b.com/2")]
    )
    assert len(findings) == 1
    assert findings[0].support == 2


def test_different_claims_stay_separate(researcher: DeepResearcher) -> None:
    findings = researcher.corroborate(
        [("Cats are soft.", "https://a.com/1"), ("Rust prevents data races.", "https://b.com/2")]
    )
    assert len(findings) == 2


def test_the_same_source_is_not_counted_twice(researcher: DeepResearcher) -> None:
    findings = researcher.corroborate(
        [("Same claim.", "https://a.com/1"), ("Same claim.", "https://a.com/1")]
    )
    assert findings[0].sources == ["https://a.com/1"]
    assert findings[0].support == 1


def test_findings_are_ordered_by_support(researcher: DeepResearcher) -> None:
    """Synthesis and storage should both see the strongest evidence first."""
    findings = researcher.corroborate(
        [
            ("Weak claim.", "https://a.com/1"),
            ("Strong claim.", "https://b.com/1"),
            ("Strong claim.", "https://c.com/1"),
        ]
    )
    assert findings[0].claim == "Strong claim."
    assert findings[0].support == 2


def test_corroborate_on_no_claims(researcher: DeepResearcher) -> None:
    assert researcher.corroborate([]) == []


def test_threshold_is_below_memory_dedupe() -> None:
    """Two sources phrase the same fact differently; demanding near-identity would
    count real corroboration as two isolated single-source claims."""
    assert CORROBORATION_THRESHOLD < 0.97


# ── Result reporting ────────────────────────────────────────────────────────────


def test_corroborated_findings_are_identified() -> None:
    result = DeepResult(
        question="q",
        answer="a",
        findings=[
            Finding("one", ["https://a.com/1"]),
            Finding("two", ["https://a.com/1", "https://b.com/2"]),
        ],
    )
    assert len(result.corroborated) == 1
    assert result.corroborated[0].claim == "two"


def test_sources_are_ordered_by_corroboration() -> None:
    result = DeepResult(
        question="q",
        answer="a",
        findings=[
            Finding("weak", ["https://weak.com/1"]),
            Finding("strong", ["https://strong.com/1", "https://other.com/2"]),
        ],
    )
    assert result.sources[0].startswith("https://strong.com")


def test_result_serializes() -> None:
    result = DeepResult(
        question="q", answer="a", findings=[Finding("c", ["https://a.com/1"])], rounds=2
    )
    payload = result.to_dict()
    assert payload["rounds"] == 2
    assert payload["findings"][0]["support"] == 1


# ── Model output parsing ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw",
    ["- first claim here\n- second claim here", "1. first claim here\n2. second claim here"],
)
def test_lines_strips_bullets_and_numbering(raw: str) -> None:
    assert _lines(raw, limit=5) == ["first claim here", "second claim here"]


def test_lines_drops_sentinels() -> None:
    assert _lines("NO CLAIMS", limit=5) == []
    assert _lines("DONE", limit=5) == []


def test_lines_drops_fragments() -> None:
    """A one-word line is not a claim and would only pollute retrieval."""
    assert _lines("- ok\n- a real claim here", limit=5) == ["a real claim here"]


def test_lines_respects_the_limit() -> None:
    raw = "\n".join(f"- claim number {i}" for i in range(20))
    assert len(_lines(raw, limit=3)) == 3


# ── The extraction bug that was silently discarding the best sources ────────────


def test_root_elements_are_never_boilerplate() -> None:
    """Regression: mdBook puts class="sidebar-visible" on <html> itself. Substring
    matching "sidebar" flagged the whole document as furniture and pruned it to
    nothing, so ARC silently discarded doc.rust-lang.org and the rustc dev guide while
    happily keeping blog posts."""
    html = (
        '<html class="light sidebar-visible"><body>'
        "<main><p>Real documentation content that must survive extraction.</p></main>"
        "</body></html>"
    )
    document = extract(html)
    assert "Real documentation content" in document.text
    assert document.word_count > 5


def test_state_classes_on_body_do_not_prune_the_page() -> None:
    html = '<body class="nav-open"><main><p>Content that should survive here.</p></main></body>'
    assert "should survive" in extract(html).text


@pytest.mark.parametrize("tag", ["html", "body", "main", "article"])
def test_structural_tags_are_exempt(tag: str) -> None:
    from arc.web.extract import _Node

    node = _Node(tag, {"class": "sidebar navigation footer"})
    assert not _is_boilerplate(node)


def test_actual_boilerplate_is_still_removed() -> None:
    """The exemption must not disable the filter it sits inside."""
    html = (
        "<body><main><p>The real article content lives here.</p></main>"
        '<div class="sidebar"><p>Related links and promotional junk.</p></div></body>'
    )
    text = extract(html).text
    assert "real article content" in text
    assert "promotional junk" not in text
