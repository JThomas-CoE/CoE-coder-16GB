"""Deterministic syntax validation for post-reintegration artifacts."""
from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser

from .ts_backends import get_parser


_LANGUAGE_ALIASES = {
    "html": "html",
    "htm": "html",
    "web": "html",
    "javascript": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "ts": "typescript",
    "css": "css",
    "python": "python",
    "py": "python",
    "cpp": "cpp",
    "c++": "cpp",
    "cxx": "cpp",
    "c": "c",
    "rust": "rust",
    "rs": "rust",
    "go": "go",
    "golang": "go",
    "java": "java",
    "kotlin": "kotlin",
    "kt": "kotlin",
    "scala": "scala",
    "sql": "sql",
    "bash": "bash",
    "sh": "bash",
}

_SCRIPT_RE = re.compile(
    r"<script\b(?P<attrs>[^>]*)>(?P<body>.*?)</script\s*>",
    re.IGNORECASE | re.DOTALL,
)
_STYLE_RE = re.compile(
    r"<style\b[^>]*>(?P<body>.*?)</style\s*>",
    re.IGNORECASE | re.DOTALL,
)
_JSON_SCRIPT_TYPES = re.compile(
    r"(?:application|text)/(?:json|ld\+json)|importmap",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SyntaxValidationResult:
    """Result of deterministic post-reintegration syntax validation."""

    checked: bool
    errors: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.errors


def _error_nodes(node) -> list[object]:
    errors: list[object] = []
    if node.type == "ERROR" or getattr(node, "is_missing", False):
        errors.append(node)
    for child in node.children:
        errors.extend(_error_nodes(child))
    return errors


def _node_location(node) -> str:
    point = getattr(node, "start_point", None)
    if point is None:
        return "unknown location"
    try:
        return f"line {point[0] + 1}, column {point[1] + 1}"
    except (IndexError, TypeError):
        return "unknown location"


def _parse_language(source: str, grammar: str, label: str) -> list[str]:
    parser = get_parser(grammar)
    if parser is None:
        return [f"{label}: tree-sitter parser unavailable for {grammar}"]
    try:
        tree = parser.parse(source.encode("utf-8"))
    except Exception as exc:
        return [f"{label}: tree-sitter parse failed: {type(exc).__name__}: {exc}"]
    errors = _error_nodes(tree.root_node)
    if not errors:
        return []
    first = errors[0]
    return [
        f"{label}: tree-sitter reported {len(errors)} syntax error(s); "
        f"first {first.type} at {_node_location(first)}"
    ]


class _HtmlStructureChecker(HTMLParser):
    """Conservative HTML structure check matching the web contract."""

    _VOID_ELEMENTS = frozenset({
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    })

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.error: str | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() not in self._VOID_ELEMENTS:
            self.stack.append(tag.lower())

    def handle_startendtag(self, tag: str, attrs) -> None:
        return

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._VOID_ELEMENTS:
            return
        if not self.stack:
            self.error = f"unmatched closing tag </{tag}>"
            return
        if self.stack[-1] == tag:
            self.stack.pop()
            return
        if tag in self.stack:
            while self.stack and self.stack[-1] != tag:
                self.stack.pop()
            if self.stack:
                self.stack.pop()
            return
        self.error = f"unexpected closing tag </{tag}>"


def _parse_html_structure(source: str) -> list[str]:
    checker = _HtmlStructureChecker()
    try:
        checker.feed(source)
        checker.close()
    except Exception as exc:
        return [f"html: parse failed: {type(exc).__name__}: {exc}"]
    if checker.error:
        return [f"html: {checker.error}"]
    if checker.stack:
        return ["html: unclosed tags: " + ", ".join(f"<{tag}>" for tag in checker.stack)]
    return []


def validate_artifact_syntax(
    content: str,
    language: str | None,
) -> SyntaxValidationResult:
    """Validate a complete candidate artifact before it is committed.

    HTML is parsed as a document and its inline JavaScript and CSS blocks
    are parsed independently because the HTML grammar does not guarantee
    that embedded language bodies are syntactically valid.
    """
    grammar = _LANGUAGE_ALIASES.get((language or "").lower())
    if grammar is None:
        return SyntaxValidationResult(checked=False)

    errors = _parse_html_structure(content) if grammar == "html" else _parse_language(content, grammar, grammar)
    if grammar == "html":
        for index, match in enumerate(_SCRIPT_RE.finditer(content), start=1):
            attrs = match.group("attrs") or ""
            if _JSON_SCRIPT_TYPES.search(attrs):
                continue
            errors.extend(
                _parse_language(match.group("body"), "javascript", f"script[{index}]")
            )
        for index, match in enumerate(_STYLE_RE.finditer(content), start=1):
            errors.extend(
                _parse_language(match.group("body"), "css", f"style[{index}]")
            )
    return SyntaxValidationResult(checked=True, errors=tuple(errors))
