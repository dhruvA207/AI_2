"""Main-content extraction.

Turns a web page into the text a human would actually read: no navigation, no cookie
banner, no footer, no sidebar of related links. §4.4 calls this "extract main content,
strip boilerplate", and doing it badly poisons memory — boilerplate stored as a fact is
worse than no fact, because it retrieves later and looks authoritative.

Written on stdlib ``html.parser`` rather than trafilatura or readability-lxml, because
a licence audit of those trees found GPL and MPL dependencies that §0.1 forbids
(docs/DECISIONS.md ADR-017).

The heuristic is the readability idea in miniature: discard elements that are
structurally never content, score the rest by text density, and keep the best subtree.
It is not as good as trafilatura. It is good enough to feed a summariser, and it has no
licence attached.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

from arc.log import get_logger

_log = get_logger(__name__)

#: Tags whose contents are never readable page content.
_DISCARD = frozenset(
    {"script", "style", "noscript", "svg", "canvas", "template", "iframe", "form", "button"}
)

#: Tags that almost always wrap boilerplate rather than the article.
_BOILERPLATE_TAGS = frozenset({"nav", "header", "footer", "aside", "menu"})

#: class/id substrings that mark boilerplate. Crude, and effective: these conventions
#: are near-universal because CSS frameworks and CMSes converged on them.
_BOILERPLATE_HINTS = (
    "nav",
    "menu",
    "sidebar",
    "footer",
    "header",
    "banner",
    "advert",
    "-ad",
    "ad-",
    "cookie",
    "consent",
    "popup",
    "modal",
    "share",
    "social",
    "comment",
    "related",
    "recommend",
    "newsletter",
    "subscribe",
    "breadcrumb",
    "pagination",
    "skip-link",
)

#: Substrings that mark the main article, overriding the hints above.
_CONTENT_HINTS = ("article", "content", "post", "entry", "story", "main", "body-text")

#: Block-level tags, which become paragraph breaks in the extracted text.
_BLOCK = frozenset(
    {
        "p",
        "div",
        "section",
        "article",
        "main",
        "br",
        "li",
        "tr",
        "blockquote",
        "pre",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "figcaption",
        "td",
        "dd",
        "dt",
    }
)

_WHITESPACE = re.compile(r"[ \t\x0b\f\r]+")
_BLANK_LINES = re.compile(r"\n{3,}")


@dataclass(frozen=True, slots=True)
class Document:
    """Extracted page content."""

    title: str
    text: str
    url: str = ""
    #: Links found in the main content, for follow-up research.
    links: list[str] = field(default_factory=list)
    #: True when the page returned almost no text, which usually means a JavaScript
    #: shell that renders client-side. §4.4 wants a headless browser for these.
    looks_like_js_shell: bool = False

    @property
    def word_count(self) -> int:
        """Roughly how much text was recovered."""
        return len(self.text.split())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "title": self.title,
            "url": self.url,
            "words": self.word_count,
            "links": len(self.links),
            "looks_like_js_shell": self.looks_like_js_shell,
        }


@dataclass
class _Node:
    """One element in the simplified tree."""

    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list[_Node | str] = field(default_factory=list)
    parent: _Node | None = None

    def text(self) -> str:
        """Concatenate this subtree's text, with block tags becoming line breaks."""
        parts: list[str] = []
        for child in self.children:
            if isinstance(child, str):
                parts.append(child)
            else:
                inner = child.text()
                if child.tag in _BLOCK:
                    parts.append(f"\n{inner}\n")
                else:
                    parts.append(inner)
        return "".join(parts)

    def marker(self) -> str:
        """The class and id attributes, lowercased, for hint matching."""
        return f"{self.attrs.get('class', '')} {self.attrs.get('id', '')}".lower()


class _TreeBuilder(HTMLParser):
    """Builds a simplified element tree, dropping non-content tags as it goes."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Node("root")
        self._current = self.root
        self._skip_depth = 0
        self._skipping: str | None = None
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._skipping:
            if tag == self._skipping:
                self._skip_depth += 1
            return

        if tag in _DISCARD:
            self._skipping = tag
            self._skip_depth = 1
            return

        if tag == "title":
            self._in_title = True
            return

        node = _Node(tag, {k: (v or "") for k, v in attrs}, parent=self._current)
        self._current.children.append(node)
        # Void elements never close, so descending into them would misparent
        # everything that follows.
        if tag not in {"br", "img", "hr", "input", "meta", "link", "source"}:
            self._current = node

    def handle_endtag(self, tag: str) -> None:
        if self._skipping:
            if tag == self._skipping:
                self._skip_depth -= 1
                if self._skip_depth <= 0:
                    self._skipping = None
            return

        if tag == "title":
            self._in_title = False
            return

        # Walk up to the matching open tag. Real HTML has unclosed tags constantly, so
        # closing blindly would corrupt the tree from the first stray </div>.
        node: _Node | None = self._current
        while node is not None and node.tag != tag:
            node = node.parent
        if node is not None and node.parent is not None:
            self._current = node.parent

    def handle_data(self, data: str) -> None:
        if self._skipping:
            return
        if self._in_title:
            self.title += data
            return
        if data.strip():
            self._current.children.append(data)


def _is_boilerplate(node: _Node) -> bool:
    """Whether a node looks like page furniture rather than content."""
    if node.tag in _BOILERPLATE_TAGS:
        return True
    marker = node.marker()
    if not marker.strip():
        return False
    # A content hint wins: plenty of article wrappers are called "main-header".
    if any(hint in marker for hint in _CONTENT_HINTS):
        return False
    return any(hint in marker for hint in _BOILERPLATE_HINTS)


def _score(node: _Node) -> float:
    """Score a node by how much readable prose it holds.

    Text length times the proportion that sits in paragraphs. A navigation block has
    plenty of text but almost none of it in ``<p>``, which is exactly the signal that
    separates a menu from an article.
    """
    text = node.text().strip()
    if not text:
        return 0.0

    paragraphs = _collect(node, "p")
    paragraph_chars = sum(len(p.text().strip()) for p in paragraphs)
    density = paragraph_chars / max(len(text), 1)

    # Commas track prose better than word count: menus and tag lists have almost none.
    commas = text.count(",")
    return len(text) * (0.3 + 0.7 * density) + commas * 20


def _collect(node: _Node, tag: str) -> list[_Node]:
    """Return every descendant with the given tag."""
    found: list[_Node] = []
    stack: list[_Node] = [node]
    while stack:
        current = stack.pop()
        for child in current.children:
            if isinstance(child, _Node):
                if child.tag == tag:
                    found.append(child)
                stack.append(child)
    return found


def _prune(root: _Node) -> None:
    """Delete boilerplate subtrees from the tree, in place.

    Pruning before scoring rather than skipping during the walk. Skipping only stopped
    the search *descending* into a nav or footer — but an ancestor's ``text()`` still
    concatenated every child, so ``<body>`` scored higher than the article it contained
    and the cookie banner came out in the extracted text.
    """
    stack: list[_Node] = [root]
    while stack:
        current = stack.pop()
        kept: list[_Node | str] = []
        for child in current.children:
            if isinstance(child, _Node) and _is_boilerplate(child):
                continue
            kept.append(child)
            if isinstance(child, _Node):
                stack.append(child)
        current.children = kept


#: Tags that explicitly declare themselves the main content. Given a scoring bonus so
#: that a semantically-correct <article> beats the <body> that wraps it — otherwise the
#: outermost element always wins on raw length alone.
_SEMANTIC_BONUS = {"article": 1.4, "main": 1.3, "section": 1.05}


def _candidates(root: _Node) -> list[_Node]:
    """Return plausible content containers."""
    found: list[_Node] = []
    stack: list[_Node] = [root]
    while stack:
        current = stack.pop()
        for child in current.children:
            if not isinstance(child, _Node):
                continue
            if child.tag in {"article", "main", "div", "section", "body"}:
                found.append(child)
            stack.append(child)
    return found


def _clean(text: str) -> str:
    """Normalise whitespace without destroying paragraph structure."""
    text = _WHITESPACE.sub(" ", text)
    lines = [line.strip() for line in text.split("\n")]
    # Single-word lines are almost always menu remnants that survived the filters.
    kept = [line for line in lines if not line or len(line.split()) > 1 or line.endswith(".")]
    return _BLANK_LINES.sub("\n\n", "\n".join(kept)).strip()


def extract(html: str, url: str = "") -> Document:
    """Pull the main content out of an HTML page."""
    builder = _TreeBuilder()
    try:
        builder.feed(html)
        builder.close()
    except Exception as exc:
        # Malformed HTML is the norm, not the exception. Salvage what parsed.
        _log.debug("HTML parse ended early for %s: %s", url or "input", exc)

    title = _clean(builder.title)[:300]

    _prune(builder.root)

    best: _Node | None = None
    best_score = 0.0
    for node in _candidates(builder.root):
        score = _score(node) * _SEMANTIC_BONUS.get(node.tag, 1.0)
        if score > best_score:
            best, best_score = node, score

    body = _clean(best.text()) if best is not None else _clean(builder.root.text())

    links: list[str] = []
    if best is not None:
        for anchor in _collect(best, "a"):
            href = anchor.attrs.get("href", "").strip()
            if href.startswith(("http://", "https://")) and href not in links:
                links.append(href)

    # A page that parsed fine but yielded almost nothing is usually a client-rendered
    # shell. Reported rather than guessed at, so the caller can decide whether to try a
    # headless browser (§4.4).
    shell = len(body.split()) < 40 and "<div" in html.lower()

    return Document(title=title, text=body, url=url, links=links[:50], looks_like_js_shell=shell)


def chunk(text: str, *, size: int = 2000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks on paragraph boundaries.

    §4.4 wants chunking before summarisation. Splitting on paragraphs rather than a
    fixed character count keeps sentences intact, and the overlap stops a fact that
    straddles a boundary from being lost to both chunks.
    """
    if len(text) <= size:
        return [text] if text.strip() else []

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for paragraph in paragraphs:
        if length + len(paragraph) > size and current:
            chunks.append("\n\n".join(current))
            # Carry the tail forward as context for the next chunk.
            tail: list[str] = []
            carried = 0
            for previous in reversed(current):
                if carried + len(previous) > overlap:
                    break
                tail.insert(0, previous)
                carried += len(previous)
            current = tail
            length = carried
        current.append(paragraph)
        length += len(paragraph)

    if current:
        chunks.append("\n\n".join(current))
    return [c for c in chunks if c.strip()]
