"""Tree-sitter backend shims for the CoE_coder codebase.

This module is the *single source of truth* for the JSON / JSONL
tree-sitter integrations.  Before this module existed, the same
``tree_sitter_language_pack`` import + ``_TS_LANGUAGE_MAP`` were
duplicated in ``chunker.py`` and ``runtime.py``.  JSON was
unparseable through either copy because the language pack 1.13.3
does not include the JSON grammar.

This module:

* Routes ``json`` / ``jsonl`` language requests to the
  separately-installed ``tree-sitter-json`` binding.
* Exposes a unified :data:`LANGUAGE_MAP` for the rest of the
  languages the codebase already supports.
* Provides :func:`parse_jsonl_sections` which splits a JSONL
  payload on newlines and returns one chunk per non-empty line.

All other tree-sitter call sites (``chunker.py`` and ``runtime.py``)
import :func:`get_parser` and :data:`LANGUAGE_MAP` from this module
instead of importing ``tree_sitter_language_pack`` directly.
"""
from __future__ import annotations

import re

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Backend imports
# ---------------------------------------------------------------------------
# The language pack 1.13.3 is the project's primary tree-sitter
# binding today; it carries 15 grammars (python, c, cpp, css, bash,
# go, html, java, javascript, kotlin, rust, scala, sql,
# typescript).  It does NOT include the JSON grammar.
try:
    from tree_sitter_language_pack import get_parser as _pack_get_parser
    _HAS_LANG_PACK = True
except ImportError:  # pragma: no cover
    _pack_get_parser = None  # type: ignore[assignment]
    _HAS_LANG_PACK = False

# JSON is installed separately (option β in the install plan).  We
# construct a parser from the binding's Language object via the
# upstream ``tree_sitter.Parser`` API.
try:
    import tree_sitter as _tree_sitter
    import tree_sitter_json as _ts_json_binding
    _HAS_JSON = True
except ImportError:  # pragma: no cover
    _tree_sitter = None  # type: ignore[assignment]
    _ts_json_binding = None  # type: ignore[assignment]
    _HAS_JSON = False


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

# Unified language alias map.  This consolidates the two maps that
# previously lived in ``chunker.py`` and ``runtime.py`` (the latter
# was a strict superset of the former).  Both call sites now read
# from here.
LANGUAGE_MAP: dict[str, str] = {
    "python": "python", "py": "python",
    "cpp": "cpp", "c++": "cpp", "cxx": "cpp",
    "c": "c",
    "rust": "rust", "rs": "rust",
    "go": "go", "golang": "go",
    "java": "java",
    "kotlin": "kotlin", "kt": "kotlin",
    "scala": "scala",
    "javascript": "javascript", "js": "javascript",
    "typescript": "typescript", "ts": "typescript",
    "html": "html",
    "css": "css",
    "sql": "sql",
    "bash": "bash", "sh": "bash", "shell": "bash",
    # Track A + B addition: structured data with a tree-sitter grammar
    # that is NOT in the language pack.  Routed through this module's
    # JSON shim (see ``get_parser`` below).
    "json": "json",
    # JSONL reuses the JSON grammar.  ``parse_jsonl_sections`` performs
    # the per-line pre-pass the spec requires.
    "jsonl": "jsonl",
}


@dataclass(frozen=True)
class JsonlSection:
    """A single line from a JSONL artifact, parsed as one section."""

    name: str          # "line0001", "line0002", ...
    kind: str          # always "json_value"
    text: str          # the raw line content (one JSON value)
    char_start: int    # byte offset of the line in the original text
    char_end: int      # byte offset one past the line's end
    line_index: int    # 1-based line number (excluding blank lines)
    error_count: int   # tree-sitter ERROR nodes inside the line


def get_parser(language: str):
    """Return a tree-sitter parser configured for ``language``.

    Returns ``None`` if the requested grammar is unavailable.  The
    caller is responsible for falling back to text-mode chunking.

    For ``language == "json"`` and ``language == "jsonl"`` the
    parser is built from the ``tree-sitter-json`` binding; this is
    the case the language pack does not handle.  For all other
    languages the request is forwarded to
    ``tree_sitter_language_pack.get_parser``.
    """
    if language in ("json", "jsonl"):
        if not _HAS_JSON:
            return None
        lang = _ts_json_binding.language()
        return _tree_sitter.Parser(_tree_sitter.Language(lang))
    if not _HAS_LANG_PACK:
        return None
    try:
        return _pack_get_parser(language)
    except Exception:
        return None


def _count_errors(node) -> int:
    """Recursive ERROR / MISSING node counter for a tree-sitter tree."""
    n = 0
    if node.type == "ERROR" or getattr(node, "is_missing", False):
        n += 1
    for child in node.children:
        n += _count_errors(child)
    return n


def parse_jsonl_sections(text: str) -> list[dict[str, Any]]:
    """Split a JSONL payload on newlines and parse each line.

    Returns a list of dicts with the shape used by
    ``_ts_extract_code_chunks`` (text, start_byte, end_byte, language,
    symbol) so the chunker can ingest them uniformly.  Each line is
    parsed with the JSON grammar; lines whose parse produces zero
    ERROR nodes are still returned (a valid JSON value), and the
    ``error_count`` field surfaces any parse problems for the
    malformed-input contract the spec requires.

    Returns ``[]`` if the JSON parser is unavailable.
    """
    parser = get_parser("json")
    if parser is None:
        return []

    sections: list[dict[str, Any]] = []
    cursor = 0
    line_index = 0
    for line in text.splitlines():
        if not line.strip():
            cursor += len(line) + 1  # +1 for the newline itself
            continue
        line_index += 1
        start = cursor
        end = cursor + len(line)
        tree = parser.parse(line.encode("utf-8"))
        errs = _count_errors(tree.root_node)
        sections.append({
            "text": line,
            "start_byte": start,
            "end_byte": end,
            "language": "jsonl",
            "symbol": f"line{line_index:04d}",
            "error_count": errs,
        })
        cursor = end + 1
    return sections


def parse_json_object_sections(text: str) -> list[dict[str, Any]]:
    """Walk a JSON document with the JSON grammar; expose top-level
    keys (object) or array elements (array) as sections.

    Tree-sitter's JSON grammar reports ``document`` as the root, an
    ``object`` or ``array`` as the only named child, then ``pair``
    nodes for object keys or unnamed ``,`` separators for arrays.  This
    helper returns one section per top-level container entry; it does
    not recurse into nested objects (Track A scope).

    Returns a list of dicts with keys: text, start_byte, end_byte,
    language, symbol, error_count.  ``symbol`` is the JSON key string
    for objects and ``"entry0001"``-style for arrays.  Returns ``[]``
    if the JSON parser is unavailable or the document fails to parse.
    """
    parser = get_parser("json")
    if parser is None:
        return []
    tree = parser.parse(text.encode("utf-8"))
    errs = _count_errors(tree.root_node)
    if errs > 0:
        # Per the spec, a malformed artifact is not rejected; we
        # surface the error count on each section so the malformed-
        # input contract can flag the artifact as broken.
        return [{
            "text": text,
            "start_byte": 0,
            "end_byte": len(text.encode("utf-8")),
            "language": "json",
            "symbol": "<malformed>",
            "error_count": errs,
        }]

    sections: list[dict[str, Any]] = []
    root = tree.root_node
    # root.type == "document" with one named child that is the
    # actual object/array.
    body = next((c for c in root.children if c.is_named), None)
    if body is None:
        return sections
    if body.type == "object":
        for child in body.children:
            if child.type != "pair":
                continue
            key_node = child.child(0)
            if key_node is None:
                continue
            # key_node.text includes the surrounding double quotes that
            # the JSON grammar places around every key string literal.
            # Strip them so the section symbol is a clean identifier
            # ("users"), not a quoted literal ("\"users\"").
            raw_key = key_node.text.decode("utf-8")
            if len(raw_key) >= 2 and raw_key[0] == '"' and raw_key[-1] == '"':
                clean_key = raw_key[1:-1]
            else:
                clean_key = raw_key
            sections.append({
                "text": text.encode("utf-8")[child.start_byte:child.end_byte].decode("utf-8", errors="replace"),
                "start_byte": child.start_byte,
                "end_byte": child.end_byte,
                "language": "json",
                "symbol": clean_key,
                "error_count": 0,
            })
    elif body.type == "array":
        # ``enumerate`` over ``body.children`` would also count
        # unnamed separator tokens (commas, brackets), so we keep
        # a separate counter that only increments for named child
        # nodes (the JSON values inside the array).
        entry_index = 0
        for child in body.children:
            if not child.is_named:
                continue
            entry_index += 1
            sections.append({
                "text": text.encode("utf-8")[child.start_byte:child.end_byte].decode("utf-8", errors="replace"),
                "start_byte": child.start_byte,
                "end_byte": child.end_byte,
                "language": "json",
                "symbol": f"entry{entry_index:04d}",
                "error_count": 0,
            })
    return sections


# ===========================================================================
# WEB SYNTAX CHECK (denovo web generation)
# ===========================================================================
# Closes the gap documented in PLAN_web_syntax_check.md: `web_new_code_contract`
# used to verify HTML tag balance only, and explicitly SKIPPED the parse check
# for CSS/JS/TS ("tree-sitter (step 3) will add proper syntax checking").
# That step is this.
#
# WHY INLINE EXTRACTION IS REQUIRED (the plan's central design point):
# `tree-sitter-html` treats <script> and <style> bodies as RAW TEXT, so the
# html grammar reports ZERO error nodes for broken inline JavaScript:
#
#     HTML with BROKEN inline JS  -> html grammar:       0 ERROR nodes
#     BROKEN JS standalone        -> javascript grammar: 1 ERROR node
#
# So parsing the artifact with the html grammar would NOT close the gap.  Each
# inline <script> / <style> body must be extracted and parsed with its OWN
# grammar.  That is what `_iter_inline_blocks` below does.
#
# WHAT THIS CANNOT CATCH (stated so it is never oversold):
#   unclosed/mismatched HTML tags        -> caught (pre-existing checker)
#   JS/CSS structural syntax errors      -> caught (this check)
#   a function declared but never called -> NOT caught; valid syntax
#   a loop that never advances state     -> NOT caught
#   wrong constant / colour / behaviour  -> NOT caught
# A parser sees structure, not intent.  The session's motivating defect
# (`animate()` declared, never invoked) is VALID JavaScript with zero ERROR
# nodes and no grammar can see it — that needs a call-graph check, a different
# feature.
#
# FAILURE POLICY: fails SOFT and SILENT.  No tree-sitter, unknown grammar, or
# any exception returns None (pass).  This check must never turn a working
# generation into a failure — a conservative parse that reports nothing is
# strictly better than a false fail, because a false fail consumes a tier.
# ---------------------------------------------------------------------------

# <script ...> ... </script>  /  <style ...> ... </style>, case-insensitive.
# The body group is non-greedy: it stops at the FIRST close tag.  A literal
# "</script>" inside a JS string will therefore truncate the body — accepted
# and documented, because the alternative (a real HTML tokenizer) buys nothing
# here and a truncated body still parses as valid JS far more often than it
# produces a false ERROR.
_INLINE_BLOCK_RE = re.compile(
    r"<(script|style)\b([^>]*)>(.*?)</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
# A <script> with a `src` attribute has no meaningful inline body.
_SCRIPT_SRC_RE = re.compile(r"\bsrc\s*=", re.IGNORECASE)
# Legacy CDATA / HTML-comment wrappers, stripped before parsing.
_CDATA_WRAP_RE = re.compile(r"^\s*//\s*<!\[CDATA\[|\]\]>\s*//\s*$")
_HTML_COMMENT_WRAP_RE = re.compile(r"^\s*<!--|-->\s*$")


def _strip_script_wrappers(body: str) -> str:
    """Remove CDATA and HTML-comment wrappers around an inline body.

    Old-style XHTML artifacts wrap inline script in a CDATA section or an HTML
    comment to hide it from non-script-aware parsers.  Both are noise to a JS
    parser and would otherwise register as ERROR nodes.
    """
    text = body
    # Only strip when the wrapper spans the WHOLE body: a partial match means
    # the content legitimately contains comment-like text.
    stripped = _CDATA_WRAP_RE.sub("", text, count=1)
    if stripped != text:
        text = stripped
    else:
        lead = _HTML_COMMENT_WRAP_RE.match(text)
        if lead:
            text = _HTML_COMMENT_WRAP_RE.sub("", text, count=1)
    return text


def iter_inline_web_blocks(html: str) -> list[tuple[str, str]]:
    """Yield ``(language, body)`` for each inline <script> / <style> in ``html``.

    ``language`` is a tree-sitter language name (``javascript`` / ``css``).
    Bodies that are empty, whitespace-only, or ``<script src=...>`` are skipped.
    """
    blocks: list[tuple[str, str]] = []
    for m in _INLINE_BLOCK_RE.finditer(html or ""):
        tag = (m.group(1) or "").lower()
        attrs = m.group(2) or ""
        body = m.group(3) or ""
        if tag == "script":
            if _SCRIPT_SRC_RE.search(attrs):
                continue          # external script: no inline body to parse
            lang = "javascript"
            body = _strip_script_wrappers(body)
        else:
            lang = "css"
        if body.strip():
            blocks.append((lang, body))
    return blocks


def _first_error_location(node, body_bytes: bytes) -> tuple[int, int]:
    """1-based line / col of a tree-sitter node, computed on the body bytes."""
    line = body_bytes[:node.start_byte].count(b"\n") + 1
    col = node.start_byte - body_bytes.rfind(b"\n", 0, node.start_byte)
    return line, col


def check_web_syntax(content: str, language: str) -> str | None:
    """Structural syntax check for web content.  ``None`` = pass.

    ``language`` is the detected fence language (``html``/``css``/
    ``javascript``/``typescript``/``jsx``/``tsx``).

    * ``html``  -- parse each inline <script>/<style> body with its own grammar
                   (the html grammar alone cannot see inside them).
    * ``css``/``javascript``/``typescript`` -- parse the whole body.

    Returns ``"syntax_error: <detail> (line N, col M)"`` on the first ERROR
    node, or ``None`` when clean, unavailable, or on any exception.
    """
    try:
        if language in ("html", "htm"):
            # Tag balance is checked by the caller (`_TagBalanceChecker`);
            # only the inline bodies are in scope here.
            for sub_lang, body in iter_inline_web_blocks(content):
                verdict = _parse_ts_body(body, sub_lang)
                if verdict:
                    return f"syntax_error: inline <{sub_lang}> block: {verdict}"
            return None
        if language in ("jsx", "tsx"):
            # No JSX/TSX grammar in the language pack; parsing them as plain
            # javascript/typescript would false-fail on every tag literal.
            return None
        if language in ("css", "javascript", "typescript"):
            return _parse_ts_body(content, language)
        return None
    except Exception:
        # Fail soft: this check must never break a working generation.
        return None


def _parse_ts_body(body: str, language: str) -> str | None:
    """Parse one body with ``language``.  Returns a detail string or ``None``."""
    grammar = LANGUAGE_MAP.get(language)
    if not grammar:
        return None                      # unknown language: no verdict
    try:
        parser = get_parser(grammar)
    except Exception:
        return None
    if parser is None:
        return None                      # grammar unavailable: no verdict
    body_bytes = body.encode("utf-8")
    try:
        tree = parser.parse(body_bytes)
    except Exception:
        return None                      # parse failure is not a verdict
    for node in _walk_errors(tree.root_node):
        line, col = _first_error_location(node, body_bytes)
        return f"ERROR node at line {line}, col {col}"
    return None


def _walk_errors(node):
    """Yield ERROR / missing nodes depth-first."""
    if node.type == "ERROR" or getattr(node, "is_missing", False):
        yield node
    for child in node.children:
        yield from _walk_errors(child)
