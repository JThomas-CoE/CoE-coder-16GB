"""Chunker for session memory V2.

Splits a model's output into retrieval chunks with boundary
context padding.  Three content kinds are handled:

* **code** — tree-sitter semantic nodes (function/class level)
* **prose** — sentence-level splitting with packing to ~500 tokens
* **diff** — unified-diff hunk splitting at ``@@`` markers

Each chunk carries:
* ``text`` — the chunk body (~500 tokens target, 800 hard cap)
* ``prev_context`` — ~100 tokens from the trailing edge of the
  preceding chunk (for continuity at injection time)
* ``next_context`` — ~100 tokens from the leading edge of the
  following chunk

The boundary context is carried *by* the chunk but is NOT the
primary search target — the BM25 index weights ``text`` at 1.0
and ``prev_context`` / ``next_context`` at 0.3.

Chunk sizing:
* Target: 500 tokens (estimated at 4 chars/token)
* Hard cap: 800 tokens — a single tree-sitter node larger than
  this is kept whole rather than artificially split
* Boundary padding: 100 tokens per side (~400 chars)

Token estimation uses the same ``len(text) // 4`` heuristic as
``clean_session_memory.py``.  This is good enough for sizing;
the real tokenizer can replace it later without API changes.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .session_memory import estimate_tokens

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_TOKENS = 500       # soft target for chunk body
HARD_CAP_TOKENS = 800     # never split a single semantic unit above this
BOUNDARY_TOKENS = 100     # padding on each side for prev/next context
BOUNDARY_CHARS = BOUNDARY_TOKENS * 4  # ~400 chars

# Code fence regex: ```lang ... ```
_CODE_FENCE_RE = re.compile(
    r"^```[ \t]*[\w+#-]*[ \t]*\n(.*?)^```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)

# Unified-diff hunk header: @@ -A,B +C,D @@
_DIFF_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@.*$", re.MULTILINE)

# Sentence boundary: split on [.!?] followed by whitespace or end.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Identifier extraction for symbol names.
_SYMBOL_RE = re.compile(
    r"\b(?:def|class|function|func|fn|func|pub fn|pub fn|defn|defn-|fun|method|sub|proc|procedure)\s+([A-Za-z_][A-Za-z0-9_]*)"
)

# Soft import of tree-sitter for code chunking.
try:
    from tree_sitter_language_pack import get_parser as _ts_get_parser
    _HAS_TREE_SITTER = True
except ImportError:  # pragma: no cover
    _ts_get_parser = None  # type: ignore[assignment]
    _HAS_TREE_SITTER = False

# Languages we can tree-sitter chunk.  Same map as clean_runtime.py.
_TS_LANGUAGE_MAP: dict[str, str] = {
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
}

# ---------------------------------------------------------------------------
# Language detection (distinct from CONTENT-KIND detection -- see below)
# ---------------------------------------------------------------------------
# HTML structural tags.  Used to answer "is this an HTML document?" without
# depending on the document containing a `function` keyword.  The COUNT
# thresholds are load-bearing: prose can mention a tag ("I used <div> here")
# without being HTML.
_HTML_TAG = (
    r"html|head|body|script|style|div|span|canvas|svg|section|main|header|"
    r"footer|nav|form|table|ul|ol|li|p|h[1-6]|button|input|select|textarea|a|img"
)
_HTML_OPEN_RE = re.compile(rf"<\s*(?:{_HTML_TAG})\b", re.IGNORECASE)
_HTML_CLOSE_RE = re.compile(rf"</\s*(?:{_HTML_TAG})\s*>", re.IGNORECASE)
# A doctype preceded only by a provenance comment still counts as the document
# head (a stripped/re-attached artifact puts the header between them).
_DOCTYPE_HTML_RE = re.compile(
    r"^\s*(?:<!--.*?-->\s*)*<!doctype\s+html", re.IGNORECASE | re.DOTALL
)
# Language TAGS that mean "an HTML document".  NOTE: `web` here is a FORMAT
# concern only.  The `domain_web` NAME also happens to be spelled `web`, but
# that is a routing concept and lives in a different table (server.py
# _ARTIFACT_LANG_TO_DOMAIN); this set must not be read as that.
_HTML_LANGUAGE_TAGS = frozenset({"html", "htm", "xhtml", "web", "markup"})
# A tree-sitter parse is only trusted as DECISIVE below this error ratio.
# The old code accepted <0.10, which let a real HTML document pass as
# `python` (0.092) when the correct answer scores 0.003.
_LANG_PARSE_MAX_ERROR_RATIO = 0.03
# ...and only above this node count.  MEASURED, not guessed: the `html` grammar
# parses ANY input at 0 error nodes in 2 nodes, so a low threshold classifies
# "hello world" as HTML.  At 12 the whole prose corpus below returns None while
# a one-function Python body still resolves; at 20 both that body and a small
# C++ translation unit fail.
#
# MEASURED FALSE POSITIVE this guards against: the CSS grammar accepts a plain
# English paragraph at error ratio 0.0, so "The analysis here is thorough and
# covers many aspects of the problem." came back as `css`.  Hence the explicit
# declaration-block requirement for css below.
_LANG_PARSE_MIN_NODES = 12
# Candidates for the parse.  `html` is deliberately ABSENT: it is settled by the
# structural rule above, which is precise, whereas the html grammar's 0-error
# acceptance of arbitrary text makes it useless as a disambiguator here.
_LANG_PARSE_CANDIDATES = (
    "css", "python", "javascript", "typescript",
    "cpp", "c", "rust", "go", "java",
)


def detect_language(text: str, language: str | None = None) -> str | None:
    """Detect the LANGUAGE of a body of text; ``None`` when not confident.

    Deliberately separate from :func:`detect_content_kind`.  They answer
    different questions and their answers are NOT interchangeable:

    * ``detect_content_kind`` answers *"how should this be split?"* and
      returns a **content kind** -- ``code`` / ``diff`` / ``prose`` /
      ``mixed``.  Those are SPLITTER SELECTORS, not languages.
    * ``detect_language`` answers *"what FILE is this?"* and returns a
      **language tag** that ``artifact_store._EXTENSION_TABLE`` and
      ``provenance._FRAMING`` can both key on.

    Conflating the two is what made an attached HTML document load as
    ``prose`` (hence ``.md``): the kind ``"prose"`` was written into the
    ``language`` field, ``_extension_for("prose")`` returned ``".md"``, and
    the provenance header -- framed ``<!-- -->`` for ``.html`` -- could no
    longer be stripped, so the artifact silently lost its origin prompt
    while the inert header stayed in the body as content.

    Returning ``None`` rather than a guess is intentional.  A caller that
    receives ``None`` should fall back to a declared default: a WRONG
    language tag is worse than an honest absence, because it produces a
    wrong extension, a wrong provenance framing, and a wrong domain pin.

    Evidence, most to least decisive:
      1. an explicit hint from the caller;
      2. an HTML declaration or structure (cheap and unambiguous);
      3. the fence tag the model itself emitted;
      4. a tree-sitter parse, HTML tried FIRST and trusted only at a strict
         error ratio.
    """
    if not text or not text.strip():
        return None

    hint = (language or "").strip().lower()
    if hint in _HTML_LANGUAGE_TAGS:
        return "html"

    # 1. HTML: decisive, and cheap enough to check before anything else.
    if _DOCTYPE_HTML_RE.search(text):
        return "html"
    if (len(_HTML_OPEN_RE.findall(text)) >= 3
            and len(_HTML_CLOSE_RE.findall(text)) >= 2):
        return "html"

    # 1b. JSON, checked BEFORE any parse.  A JSON object is also valid Python
    #     (`{"a": 1}` parses), so without this a JSON document is classified
    #     `python` and saved as `.py` -- verified on the real side-car files in
    #     artifacts_saved.  A full `json.loads` is used rather than a brace
    #     heuristic, so it cannot fire on code that merely CONTAINS an object
    #     literal; note this also correctly leaves a bare `"string"` or `123`
    #     alone, since those are valid JSON only by the scalar rules that would
    #     make the test useless as a discriminator.
    _stripped = text.lstrip()
    if _stripped.startswith(("{", "[")):
        try:
            json.loads(text)
        except Exception:
            pass
        else:
            return "json"

    # 2. A fence tag is the model's OWN declaration of what it wrote.
    fence_tags = [
        m.lower() for m in
        re.findall(r"^```[ \t]*([\w+#.-]+)[ \t]*\n", text, re.MULTILINE)
    ]
    if fence_tags:
        commonest = max(dict.fromkeys(fence_tags), key=fence_tags.count)
        mapped = _TS_LANGUAGE_MAP.get(commonest)
        if mapped:
            return mapped

    # 3. tree-sitter, tried only when the text shows structural code markers.
    #    `html` is not a candidate: it is settled above, and the html grammar
    #    accepts arbitrary text at 0 errors, which makes it worthless as a
    #    disambiguator.  `css` needs an explicit declaration block for the same
    #    reason -- measured, its grammar returns 0 errors for a plain English
    #    paragraph.
    if _HAS_TREE_SITTER:
        has_def = bool(re.search(
            r"\b(?:def|class|function|fn|func|struct|enum|interface)\b\s+\w+", text
        ))
        has_indent = any(line.startswith((" ", "\t")) for line in text.splitlines())
        has_brackets = bool(re.search(r"[{};:]", text))
        # A matched brace PAIR is itself sufficient.  Flat CSS and a flat
        # C/C++ translation unit show neither a `def`-family keyword nor
        # indentation, so the other clauses miss them; measured, adding this
        # admits both while a 10-item prose corpus containing "{a, b, c}" and
        # "{x}" still yields 0 false positives.
        has_brace_pair = "{" in text and "}" in text
        if has_def or (has_indent and has_brackets) or has_brace_pair:
            raw = text.encode("utf-8")
            for candidate in _LANG_PARSE_CANDIDATES:
                if candidate == "css" and not ("{" in text and "}" in text):
                    continue
                try:
                    tree = _ts_get_parser(candidate).parse(raw)
                    total = _ts_count_nodes(tree.root_node)
                    errors = len(_ts_find_errors(tree.root_node))
                except Exception:
                    continue
                if (total >= _LANG_PARSE_MIN_NODES
                        and errors / max(total, 1) < _LANG_PARSE_MAX_ERROR_RATIO):
                    return candidate

    return None


# Tree-sitter node types that represent semantic code units
# (functions, classes, methods, etc.).
_CODE_CHUNK_TYPES = frozenset({
    "function_definition", "function_declaration",
    "class_definition", "class_declaration", "class_specifier",
    "struct_specifier", "struct_declaration",
    "method_definition", "method_declaration",
    "interface_declaration", "enum_declaration",
    "module_definition", "namespace_definition",
    "decorated_definition",  # Python: @decorator + def/class
    "export_statement",       # JS/TS: export function ...
    "lexical_declaration",    # JS/TS: let/const
    "expression_statement",   # top-level call or assignment
    "if_statement",           # top-level guard
    "for_statement",
    "while_statement",
    "try_statement",
    "with_statement",
    "import_statement",
    "import_from_statement",
})


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single retrieval unit in the session memory chunk index.

    Attributes
    ----------
    chunk_id
        Unique identifier: ``{turn_id}:{kind}:{seq}``
    turn_id
        Foreign key to the parent TurnArtifact.
    seq
        Chronological position within the turn (0-based).
    kind
        Content kind: ``"code"``, ``"prose"``, ``"diff_hunk"``.
    text
        The chunk body (~500 tokens target).
    prev_context
        ~100 tokens from the trailing edge of the preceding chunk.
    next_context
        ~100 tokens from the leading edge of the following chunk.
    char_start
        Byte offset into the parent TurnArtifact.model_output.
    char_end
        Byte offset into the parent TurnArtifact.model_output (exclusive).
    language
        Detected language for code chunks, ``None`` otherwise.
    symbol
        Function/class name extracted from code, ``None`` otherwise.
    """

    chunk_id: str
    turn_id: str
    seq: int
    kind: str
    text: str
    prev_context: str = ""
    next_context: str = ""
    char_start: int = 0
    char_end: int = 0
    language: str | None = None
    symbol: str | None = None

    @property
    def token_estimate(self) -> int:
        """Rough token count of the chunk body only (no context)."""
        return estimate_tokens(self.text)

    @property
    def total_token_estimate(self) -> int:
        """Rough token count including boundary context."""
        return (
            estimate_tokens(self.text)
            + estimate_tokens(self.prev_context)
            + estimate_tokens(self.next_context)
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for debug/logging."""
        return {
            "chunk_id": self.chunk_id,
            "turn_id": self.turn_id,
            "seq": self.seq,
            "kind": self.kind,
            "text": self.text[:200] + "..." if len(self.text) > 200 else self.text,
            "prev_context": self.prev_context[:80] + "..." if len(self.prev_context) > 80 else self.prev_context,
            "next_context": self.next_context[:80] + "..." if len(self.next_context) > 80 else self.next_context,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "language": self.language,
            "symbol": self.symbol,
            "token_estimate": self.token_estimate,
        }


# ---------------------------------------------------------------------------
# Content-kind detection -- a SPLITTER SELECTOR, not a language.  For the
# language of a body of text use :func:`detect_language` above.
# ---------------------------------------------------------------------------

def detect_content_kind(text: str, language: str | None = None) -> str:
    """Detect the dominant content kind of a model output.

    Returns one of:
    * ``"code"`` — output is predominantly fenced or raw code
    * ``"diff"`` — output is predominantly a unified diff
    * ``"prose"`` — output is predominantly natural language
    * ``"mixed"`` — no single kind dominates

    The detection uses a combination of cheap regex heuristics
    and optional tree-sitter parse (when available) to detect
    raw code without fences.  The result determines which
    splitter is used.
    """
    if not text or not text.strip():
        return "prose"

    lines = text.splitlines()
    total_lines = len(lines)

    # Diff detection: count lines starting with +, -, @@
    diff_markers = sum(
        1 for line in lines
        if line.startswith(("+++", "---", "@@")) or
        (line.startswith(("+", "-")) and not line.startswith(("+++", "---")))
    )
    diff_ratio = diff_markers / max(total_lines, 1)
    if diff_ratio > 0.5:
        return "diff"

    # Code fence detection: extract fenced blocks
    fence_blocks = _CODE_FENCE_RE.findall(text)
    code_chars = sum(len(block) for block in fence_blocks)
    total_chars = len(text)
    code_ratio = code_chars / max(total_chars, 1)

    # If there are code fences, check whether the non-code
    # (prose) part is substantial.  If prose between fences
    # is >15% of the text, it's mixed even at high code ratio.
    if fence_blocks:
        non_code_chars = total_chars - code_chars
        non_code_ratio = non_code_chars / max(total_chars, 1)
        if non_code_ratio > 0.15:
            return "mixed"
        if code_ratio > 0.6:
            return "code"

    # Raw code detection (no fences): try tree-sitter parse.
    # If the text parses with a low error ratio AND has
    # structural code markers (colons after def/class,
    # indentation, brackets), it's code.
    if _HAS_TREE_SITTER and not fence_blocks:
        # A `language=` hint is ADDITIVE, not restrictive: it is PREPENDED to
        # the default sweep so it is tried first when it fits, but it never
        # REMOVES a language from consideration.  Both directions matter and
        # both were measured:
        #
        #   * Restrictive HURTS.  With the hint replacing the list, a wrong
        #     hint produces a false negative:
        #         python text, language=None  -> 'code'   (correct)
        #         python text, language='cpp' -> 'prose'  (wrong)
        #   * Additive still HELPS.  bash / kotlin / scala / sql / css / html
        #     are valid grammars in _TS_LANGUAGE_MAP but are absent from the
        #     default sweep, so a hint is the only way they are ever tried:
        #         shell script, language=None   -> 'prose'  (missed)
        #         shell script, language='bash' -> 'code'
        #     Prepending preserves that, because the hint is still reached
        #     first when the content really is that language.
        ts_lang = _TS_LANGUAGE_MAP.get(language.lower()) if language else None
        candidates = ([ts_lang] if ts_lang else []) + [
            "python", "cpp", "rust", "go", "java", "javascript", "typescript", "c"
        ]
        code_bytes = text.encode("utf-8")
        # Pre-filter: require at least some structural markers
        # to avoid misclassifying short prose as code.
        has_def = bool(re.search(r"\b(?:def|class|function|fn|func|struct|enum|interface)\b\s+\w+", text))
        has_indent = any(line.startswith((" ", "\t")) for line in lines)
        has_brackets = bool(re.search(r"[{};:]", text))
        if has_def or (has_indent and has_brackets):
            for lang in candidates:
                if not lang:
                    continue
                try:
                    parser = _ts_get_parser(lang)
                    tree = parser.parse(code_bytes)
                    errors = _ts_find_errors(tree.root_node)
                    total = _ts_count_nodes(tree.root_node)
                    ratio = len(errors) / max(total, 1)
                    if ratio < 0.1:
                        return "code"
                except Exception:
                    continue

    # Keyword heuristic: if the text has many code keywords and
    # few prose markers, treat it as code.
    code_keywords = re.findall(
        r"\b(?:def|class|function|return|import|from|if|else|for|while|try|except|with|fn|func|pub|let|const|var|void|int|float|string|struct|enum|interface)\b",
        text,
    )
    prose_markers = re.findall(r"\b(?:the|a|an|is|are|was|were|this|that|with|for|and|or|but|in|on|to|of)\b", text, re.IGNORECASE)
    if len(code_keywords) > 5 and len(code_keywords) > len(prose_markers) * 0.3:
        return "code"

    return "prose"


def _ts_find_errors(node) -> list:
    """Recursively find all ERROR and missing nodes in a tree-sitter tree."""
    errors = []
    if node.type == "ERROR" or getattr(node, "is_missing", False):
        errors.append(node)
    for child in node.children:
        errors.extend(_ts_find_errors(child))
    return errors


def _ts_count_nodes(node) -> int:
    """Count total nodes in a tree-sitter tree."""
    count = 1
    for child in node.children:
        count += _ts_count_nodes(child)
    return count


# ---------------------------------------------------------------------------
# Code chunker (tree-sitter)
# ---------------------------------------------------------------------------

def _ts_extract_code_chunks(
    code: str,
    language: str | None,
) -> list[dict[str, Any]]:
    """Extract semantic code chunks using tree-sitter.

    Returns a list of dicts with keys:
        text, start_byte, end_byte, language, symbol

    Falls back to line-based splitting if tree-sitter is
    unavailable or the language is not supported.
    """
    if not _HAS_TREE_SITTER or not language:
        return _line_based_code_chunks(code, language)

    ts_lang = _TS_LANGUAGE_MAP.get(language.lower())
    if not ts_lang:
        return _line_based_code_chunks(code, language)

    # Structured-data short-circuits (Track A + B for JSON/JSONL).
    # JSON uses top-level keys as section names; JSONL splits on
    # newlines and emits one section per line.  Both return chunks
    # in the same shape so downstream callers do not have to special
    # case either format.
    if ts_lang == "json":
        from .ts_backends import parse_json_object_sections
        return parse_json_object_sections(code)
    if ts_lang == "jsonl":
        from .ts_backends import parse_jsonl_sections
        return parse_jsonl_sections(code)

    try:
        parser = _ts_get_parser(ts_lang)
        code_bytes = code.encode("utf-8")
        tree = parser.parse(code_bytes)
    except Exception:
        return _line_based_code_chunks(code, language)

    chunks: list[dict[str, Any]] = []
    root = tree.root_node

    def _walk(node) -> None:
        if node.type in _CODE_CHUNK_TYPES:
            text = code_bytes[node.start_byte:node.end_byte].decode(errors="replace")
            symbol = _extract_symbol_name(text, node, code_bytes)
            chunks.append({
                "text": text,
                "start_byte": node.start_byte,
                "end_byte": node.end_byte,
                "language": ts_lang,
                "symbol": symbol,
            })
            # Don't recurse into chunk children
            return
        for child in node.children:
            _walk(child)

    _walk(root)

    # If tree-sitter found no semantic chunks (e.g. the code is
    # just expressions or comments), fall back to line-based.
    if not chunks:
        return _line_based_code_chunks(code, language)

    # Post-pass: split any oversized chunks at line boundaries.
    # Rule: a chunk over HARD_CAP_TOKENS is split into
    # ceil(tokens / HARD_CAP) sub-chunks, each as evenly sized
    # as possible, at natural line boundaries.  The sub-chunks
    # all carry the same symbol name so they can be
    # reintegrated into the whole semantic entity if needed.
    result: list[dict[str, Any]] = []
    for chunk in chunks:
        token_est = estimate_tokens(chunk["text"])
        if token_est <= HARD_CAP_TOKENS:
            result.append(chunk)
        else:
            result.extend(_split_oversized_node(chunk))

    return result


# ---------------------------------------------------------------------------
# HTML with inline <script> chunker (tree-sitter native)
# ---------------------------------------------------------------------------

def _looks_like_html(text: str) -> bool:
    """Check if text is HTML (has <html> or <!DOCTYPE or <body> tags)."""
    return bool(re.search(r"<(?:html|body|head|script|div)\b", text, re.IGNORECASE))


# 2026-09-10 (user-ruled, paragraph principle): body elements shorter than
# this are too small to stand alone as an outline section; they merge with
# their next sibling (previous sibling at block end) rather than being
# emitted as sentence-level sections.
_BODY_ELEMENT_MIN_CHARS = 150

# JS top-level statements shorter than this merge with their next neighbor
# (same exemption rule as body elements: threshold governs the pieces, the
# merged result is exempt).
_JS_STATEMENT_MIN_CHARS = 150


def _find_element_by_tag(root_node, tag: str, code_bytes: bytes):
    """Locate the first ``element`` node whose tag_name equals ``tag``."""
    if root_node.type == "element":
        for child in root_node.children:
            if child.type == "start_tag":
                for sub in child.children:
                    if sub.type == "tag_name":
                        name = code_bytes[sub.start_byte:sub.end_byte].decode(errors="replace").strip().lower()
                        if name == tag:
                            return root_node
    for child in root_node.children:
        found = _find_element_by_tag(child, tag, code_bytes)
        if found is not None:
            return found
    return None


def _element_tag_of(node, code_bytes: bytes) -> str:
    """Return the lowercased tag name for an ``element`` node, or ''."""
    for child in node.children:
        if child.type == "start_tag":
            for sub in child.children:
                if sub.type == "tag_name":
                    return code_bytes[sub.start_byte:sub.end_byte].decode(errors="replace").strip().lower()
    return ""


def _element_section_name(node, code_bytes: bytes) -> str | None:
    """Derive the outline name for a body child element.

    ``{tag}#{id}`` wins over ``{tag}.{class}`` (ids are unique by
    contract); both fall back to the bare tag; no identification at
    all -> ``None`` (caller applies the anonymous grammar).
    """
    tag = _element_tag_of(node, code_bytes)
    if not tag:
        return None
    # id wins
    for child in node.children:
        if child.type != "start_tag":
            continue
        for attr in child.children:
            if attr.type != "attribute":
                continue
            attr_text = code_bytes[attr.start_byte:attr.end_byte].decode(errors="replace")
            m = re.match(r"^([a-zA-Z_:][-a-zA-Z0-9_:.]*)\s*=\s*[\"']([^\"']*)[\"']", attr_text.strip())
            if m and m.group(1).lower() == "id" and m.group(2).strip():
                return f"{tag}#{m.group(2).strip()}"
    # first class value
    for child in node.children:
        if child.type != "start_tag":
            continue
        for attr in child.children:
            if attr.type != "attribute":
                continue
            attr_text = code_bytes[attr.start_byte:attr.end_byte].decode(errors="replace")
            m = re.match(r"^([a-zA-Z_:][-a-zA-Z0-9_.:]*)\s*=\s*[\"']([^\"']*)[\"']", attr_text.strip())
            if m and m.group(1).lower() == "class":
                value = m.group(2).strip()
                if value:
                    first = value.split()[0]
                    if first:
                        return f"{tag}.{first}"
    return f"{tag}"


def _extract_body_element_sections(root_node, code_bytes: bytes) -> list[dict[str, Any]]:
    """Extract the <body> element's DIRECT element children as sections.

    2026-09-10 (user-ruled): tree-sitter parses the complete document, but
    the old extractor kept only script/style content, so body markup had no
    outline sections and modify requests targeting body markup (add a field
    to the modal) had no valid anchor. This emits ONE section per direct
    child element of <body> — paragraph-sized, semantically coherent
    (a component is one nameable thing), NOT a recursive dump of every div.

    Naming: ``{tag}#{id}`` or ``{tag}.{class}`` when present (e.g.
    ``div#modal-overlay``), falling back to the anonymous grammar
    (``anonymous:element:{start}-{end}``) otherwise.
    """
    body_el = _find_element_by_tag(root_node, "body", code_bytes)
    if body_el is None:
        return []
    chunks: list[dict[str, Any]] = []
    for child in body_el.children:
        if child.type != "element":
            continue
        # script/style elements are handled by their own extractors with
        # deeper (function/rule) granularity — skip here to avoid overlap.
        if _element_tag_of(child, code_bytes) in ("script", "style"):
            continue
        name = _element_section_name(child, code_bytes)
        if name is None:
            name = f"anonymous:element:{child.start_byte}-{child.end_byte}"
        chunks.append({
            "text": code_bytes[child.start_byte:child.end_byte].decode(errors="replace"),
            "start_byte": child.start_byte,
            "end_byte": child.end_byte,
            "language": "html",
            "symbol": name,
        })
    return chunks


def _merge_micro_statements(
    raw_chunks: list[dict[str, Any]],
    code_bytes: bytes,
) -> list[dict[str, Any]]:
    """Collapse consecutive sub-threshold top-level JS statements into one.

    2026-09-10 (user-ruled): the JS extractor emits sentence-level sections
    for tiny top-level statements (``let editingId = null;`` ~21 chars).
    Paragraph principle: merge them with the NEXT neighbor (previous
    neighbor at block end) into ONE section. The threshold governs the
    PIECES; the merged result is EXEMPT (semantic coherence, not
    size-maximality). Merging requires a pure-whitespace gap between the
    chunks (same contiguous script text); functions/classes never chain.
    """
    if not raw_chunks:
        return raw_chunks
    merged: list[dict[str, Any]] = []
    i = 0
    n = len(raw_chunks)
    # 2026-09-10 fix 1: the micro-merge applies ONLY to JavaScript chunks.
    # CSS rules are their own paragraph unit (user ruling) and must never be
    # chained; HTML body-element sections are paragraph-sized by design.
    _is_js = lambda rc: (rc.get("language") or "").lower() in ("javascript", "js")
    # 2026-09-10 fix 2: functions/classes/exports NEVER participate in the
    # micro-merge chain (as chain members OR as merge absorbers). A chunk
    # whose text starts a declaration of one of these is passed through
    # untouched, no matter its size.
    _FUNC_START = re.compile(r"^\s*(?:async\s+)?function\b|^\s*class\b|^\s*export\b")
    while i < n:
        cur = raw_chunks[i]
        cur_size = cur["end_byte"] - cur["start_byte"]
        if (not _is_js(cur)) or cur_size >= _JS_STATEMENT_MIN_CHARS or _FUNC_START.match(cur["text"]):
            merged.append(cur)
            i += 1
            continue
        # below threshold: grow forward while neighbors are also small,
        # separated only by whitespace, and none is a function/class
        start, end = cur["start_byte"], cur["end_byte"]
        j = i
        while j + 1 < n:
            nxt = raw_chunks[j + 1]
            if not _is_js(nxt):
                break
            gap = code_bytes[end:nxt["start_byte"]].decode(errors="replace")
            if gap.strip():
                break
            if (nxt["end_byte"] - nxt["start_byte"]) >= _JS_STATEMENT_MIN_CHARS:
                break
            if _FUNC_START.match(nxt["text"]):
                break
            end = nxt["end_byte"]
            j += 1
        if j > i:
            merged.append({
                "text": code_bytes[start:end].decode(errors="replace"),
                "start_byte": start,
                "end_byte": end,
                "language": cur.get("language"),
                "symbol": None,  # merged: anonymous naming downstream
            })
            i = j + 1
        else:
            merged.append(cur)
            i += 1
    return merged


def _chunk_html_with_scripts(
    html_text: str,
    turn_id: str,
) -> list[Chunk]:
    """Chunk an HTML document with inline <script> and <style> tags.

    Uses the HTML tree-sitter parser to find script_element and
    style_element nodes.  Script content is parsed with the JS
    parser to extract function-level chunks.  Style content is
    split into CSS rules (each rule becomes a section named by
    its selector).

    Non-script/non-style HTML content is chunked as prose.

    Returns a list of Chunk objects with correct byte offsets
    into the original HTML text.
    """
    if not _HAS_TREE_SITTER:
        # Fallback: treat the whole thing as one prose chunk
        return [Chunk(
            chunk_id=f"{turn_id}:prose:0", turn_id=turn_id, seq=0,
            kind="prose", text=html_text,
            char_start=0, char_end=len(html_text.encode("utf-8")),
        )]

    try:
        parser = _ts_get_parser("html")
        tree = parser.parse(html_text.encode("utf-8"))
    except Exception:
        return [Chunk(
            chunk_id=f"{turn_id}:prose:0", turn_id=turn_id, seq=0,
            kind="prose", text=html_text,
            char_start=0, char_end=len(html_text.encode("utf-8")),
        )]

    # Collect (start_byte, end_byte, content) for each <script> and <style>
    code_bytes = html_text.encode("utf-8")
    script_blocks: list[tuple[int, int, str]] = []
    style_blocks: list[tuple[int, int, str]] = []

    def _find_blocks(node) -> None:
        if node.type == "script_element":
            for child in node.children:
                if child.type in ("raw_text", "text"):
                    script_content = code_bytes[child.start_byte:child.end_byte].decode(errors="replace")
                    script_blocks.append((child.start_byte, child.end_byte, script_content))
                    break
        elif node.type == "style_element":
            for child in node.children:
                if child.type in ("raw_text", "text"):
                    style_content = code_bytes[child.start_byte:child.end_byte].decode(errors="replace")
                    style_blocks.append((child.start_byte, child.end_byte, style_content))
                    break
        for child in node.children:
            _find_blocks(child)

    _find_blocks(tree.root_node)

    raw_chunks: list[dict[str, Any]] = []

    # Parse each script block with the JS parser
    for script_start, script_end, script_content in script_blocks:
        js_chunks = _ts_extract_code_chunks(script_content, "javascript")
        for jc in js_chunks:
            raw_chunks.append({
                "text": jc["text"],
                "start_byte": script_start + jc["start_byte"],
                "end_byte": script_start + jc["end_byte"],
                "language": jc.get("language") or "javascript",
                "symbol": jc.get("symbol"),
            })

    # Parse each style block — extract CSS rules as sections
    for style_start, style_end, style_content in style_blocks:
        css_chunks = _extract_css_rules(style_content, style_start)
        raw_chunks.extend(css_chunks)

    # 2026-09-10 (user-ruled, "paragraphs not sentences"): section the BODY
    # markup. Tree-sitter already parses the complete document, but only
    # script/style content was extracted, leaving <body> markup with NO
    # sections — models had no valid anchor for the most common modify
    # requests (add a field to the modal, tweak the header). Fix: extract
    # the <body> element's DIRECT element children as one section each —
    # paragraph granularity, NOT a sentence-level or whole-file dump.
    # Naming: tag#id or tag.class when present, anonymous grammar otherwise.
    body_chunks = _extract_body_element_sections(tree.root_node, code_bytes)
    raw_chunks.extend(body_chunks)

    # 2026-09-10 (user-ruled micro-statement merge): consecutive top-level
    # JS micro-statements below the paragraph minimum collapse into ONE
    # section. Threshold triggers the merge; the merged result itself is
    # EXEMPT from the threshold (semantic coherence, not size-maximality).
    raw_chunks = _merge_micro_statements(raw_chunks, code_bytes)

    # If no JS functions were found in any script block, fall back
    # to treating the whole HTML as one prose chunk
    if not raw_chunks:
        return [Chunk(
            chunk_id=f"{turn_id}:prose:0", turn_id=turn_id, seq=0,
            kind="prose", text=html_text,
            char_start=0, char_end=len(html_text.encode("utf-8")),
        )]

    # Assign boundary context
    _assign_boundary_context(raw_chunks, html_text)

    # Convert to Chunk objects
    chunks: list[Chunk] = []
    for i, rc in enumerate(raw_chunks):
        chunks.append(Chunk(
            chunk_id=f"{turn_id}:code:{i}",
            turn_id=turn_id,
            seq=i,
            kind="code",
            text=rc["text"],
            prev_context=rc.get("prev_context", ""),
            next_context=rc.get("next_context", ""),
            char_start=rc["start_byte"],
            char_end=rc["end_byte"],
            language=rc.get("language"),
            symbol=rc.get("symbol"),
        ))

    return chunks


def _extract_css_rules(
    css_content: str,
    style_start: int,
) -> list[dict[str, Any]]:
    """Extract individual CSS rules from a style block.

    Each CSS rule (like `#acid { background-color: red; }` or
    `.box { ... }`) becomes a chunk.  The symbol is the
    selector so the reintegration can target it by name.

    Returns a list of raw chunk dicts (text, start_byte,
    end_byte, language, symbol).
    """
    chunks: list[dict[str, Any]] = []
    pos = 0
    while pos < len(css_content):
        # Find the next `{`
        brace_idx = css_content.find('{', pos)
        if brace_idx == -1:
            break
        raw_selector = css_content[pos:brace_idx].strip()
        # Strip leading CSS comments so the canonical selector
        # is what the model will reference (e.g. ".indicator-box"
        # not "/* foo */ .indicator-box").  Walk left-to-right,
        # dropping /* ... */ blocks and their surrounding whitespace
        # until we find real selector text.
        selector = raw_selector
        while True:
            stripped = selector.lstrip()
            if stripped.startswith("/*"):
                close = stripped.find("*/")
                if close == -1:
                    break  # malformed — leave as-is
                selector = stripped[close + 2 :].strip()
                continue
            break
        # Find the matching `}`
        end_idx = css_content.find('}', brace_idx)
        if end_idx == -1:
            break
        body = css_content[brace_idx:end_idx + 1]
        # Skip @-rules like @media, @import that wrap other rules
        if selector.startswith('@'):
            pos = end_idx + 1
            continue
        # Build the full rule text: selector + body
        rule_text = selector + ' ' + body
        # Compute byte offsets within the full HTML
        # style_start is the byte offset of the raw_text child
        # of the style_element, so we add it to local positions
        local_start = pos
        local_end = end_idx + 1
        chunks.append({
            "text": rule_text,
            "start_byte": style_start + local_start,
            "end_byte": style_start + local_end,
            "language": "css",
            "symbol": selector if selector else None,
        })
        pos = end_idx + 1
    return chunks


def _split_oversized_node(chunk: dict[str, Any]) -> list[dict[str, Any]]:
    """Split an oversized tree-sitter node into sub-chunks.

    Splits into ``ceil(tokens / HARD_CAP)`` sub-chunks at line
    boundaries, targeting even sizes.  Each sub-chunk carries
    the same symbol name.  The byte offsets are adjusted to
    refer to the original code, not the node-local text.

    Example: a 1200-token function splits into 2 sub-chunks
    of ~600 tokens each.  A 2000-token function splits into
    3 sub-chunks of ~667 tokens each.
    """
    import math

    text = chunk["text"]
    base_offset = chunk["start_byte"]
    symbol = chunk.get("symbol")
    language = chunk.get("language")

    token_est = estimate_tokens(text)
    n_splits = max(2, math.ceil(token_est / HARD_CAP_TOKENS))

    lines = text.splitlines(keepends=True)
    if len(lines) < n_splits:
        # Can't split into more pieces than there are lines.
        # Fall back to returning the whole chunk (can't do better).
        return [chunk]

    # Compute target tokens per sub-chunk
    target_per = token_est / n_splits

    # Greedy line assignment: accumulate lines until reaching
    # the target for this sub-chunk, then start the next one.
    sub_chunks: list[dict[str, Any]] = []
    current_lines: list[str] = []
    current_tokens = 0
    current_offset = base_offset
    pos = base_offset
    sub_idx = 0
    line_idx = 0

    while line_idx < len(lines):
        line = lines[line_idx]
        line_bytes = len(line.encode("utf-8"))
        line_tokens = estimate_tokens(line)
        current_lines.append(line)
        current_tokens += line_tokens
        pos += line_bytes
        line_idx += 1

        # Check if we've reached the target for this sub-chunk
        # and there are still enough lines remaining for the
        # remaining sub-chunks.
        remaining_subs = n_splits - sub_idx - 1
        remaining_lines = len(lines) - line_idx

        if (
            current_tokens >= target_per
            and sub_idx < n_splits - 1
            and remaining_lines >= remaining_subs
        ):
            sub_text = "".join(current_lines)
            sub_chunks.append({
                "text": sub_text,
                "start_byte": current_offset,
                "end_byte": pos,
                "language": language,
                "symbol": symbol,
            })
            current_lines = []
            current_tokens = 0
            current_offset = pos
            sub_idx += 1

    # Final sub-chunk gets whatever remains
    if current_lines:
        sub_text = "".join(current_lines)
        sub_chunks.append({
            "text": sub_text,
            "start_byte": current_offset,
            "end_byte": pos,
            "language": language,
            "symbol": symbol,
        })

    return sub_chunks if sub_chunks else [chunk]


def _extract_symbol_name(text: str, node, code_bytes: bytes) -> str | None:
    """Extract the function/class name from a tree-sitter node."""
    # Try to find an identifier child
    for child in node.children:
        if child.type == "identifier":
            return code_bytes[child.start_byte:child.end_byte].decode(errors="replace")
        # For decorated definitions, the name is on the inner node
        for grandchild in child.children:
            if grandchild.type == "identifier":
                return code_bytes[grandchild.start_byte:grandchild.end_byte].decode(errors="replace")
    # Fallback: regex on the text
    m = _SYMBOL_RE.search(text)
    return m.group(1) if m else None


def _line_based_code_chunks(
    code: str,
    language: str | None,
) -> list[dict[str, Any]]:
    """Fallback: split code by blank lines or indentation boundaries.

    Groups consecutive non-blank lines into chunks, targeting
    ~500 tokens.  Each group becomes one chunk.
    """
    lines = code.splitlines(keepends=True)
    if not lines:
        return []

    chunks: list[dict[str, Any]] = []
    current: list[str] = []
    current_start_byte = 0
    pos = 0

    for line in lines:
        line_bytes = len(line.encode("utf-8"))
        current.append(line)
        current_text = "".join(current)

        if estimate_tokens(current_text) >= TARGET_TOKENS:
            # Check if this is at a natural boundary (blank line next)
            chunks.append({
                "text": current_text,
                "start_byte": current_start_byte,
                "end_byte": pos + line_bytes,
                "language": language,
                "symbol": _extract_symbol_regex(current_text),
            })
            current = []
            current_start_byte = pos + line_bytes
        pos += line_bytes

    if current:
        chunks.append({
            "text": "".join(current),
            "start_byte": current_start_byte,
            "end_byte": pos,
            "language": language,
            "symbol": _extract_symbol_regex("".join(current)),
        })

    return chunks


def _extract_symbol_regex(text: str) -> str | None:
    """Extract a symbol name from code text using regex."""
    m = _SYMBOL_RE.search(text)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Prose chunker (sentence-level)
# ---------------------------------------------------------------------------

def _locate_piece(text: str, piece: str, start: int = 0) -> int:
    """Whitespace-tolerant locate of ``piece`` within ``text``.

    ``_split_prose`` normalizes whitespace (``text.strip()`` and
    ``" ".join``), so a returned piece is not always an exact
    substring of the original ``text`` (e.g. ``\n\n`` between
    paragraphs becomes a single space).  A naive ``text.index(piece)``
    then raises ``ValueError: substring not found``, which crashes
    ``chunk_model_output`` and aborts session-memory indexing.

    This collapses runs of whitespace in both strings to a single
    space before searching, so a normalized piece still locates its
    original span.  Returns the byte offset of the first match, or
    ``start`` if no match is found (caller falls back gracefully).
    """
    if not piece:
        return start
    _ws = re.compile(r"\s+")
    norm_text = _ws.sub(" ", text)
    norm_piece = _ws.sub(" ", piece)
    idx = norm_text.find(norm_piece, start)
    if idx < 0:
        return start
    # Map the normalized index back to the original text by counting
    # non-whitespace characters up to ``idx``.
    n_ws = 0
    for i, ch in enumerate(text):
        if n_ws == idx:
            return i
        if not ch.isspace():
            n_ws += 1
    return start


def _split_prose(text: str) -> list[str]:
    """Split prose into sentence-level chunks.

    Splits on sentence boundaries, then packs sentences into
    chunks targeting ~500 tokens.  Each chunk is a group of
    consecutive sentences whose combined token estimate is
    closest to TARGET_TOKENS without exceeding HARD_CAP.
    """
    if not text or not text.strip():
        return []

    # Split into sentences
    raw_sentences = _SENTENCE_RE.split(text.strip())
    sentences = [s for s in raw_sentences if s.strip()]

    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)

        # If a single sentence exceeds the hard cap, split it
        # at a natural boundary (comma, semicolon, or just hard cut).
        if sentence_tokens > HARD_CAP_TOKENS:
            # Flush current buffer first
            if current:
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0
            # Split the oversized sentence
            for piece in _split_long_sentence(sentence):
                chunks.append(piece)
            continue

        if current_tokens + sentence_tokens > HARD_CAP_TOKENS and current:
            chunks.append(" ".join(current))
            current = [sentence]
            current_tokens = sentence_tokens
        else:
            current.append(sentence)
            current_tokens += sentence_tokens

            if current_tokens >= TARGET_TOKENS:
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0

    if current:
        chunks.append(" ".join(current))

    return chunks


def _split_long_sentence(sentence: str) -> list[str]:
    """Split a sentence that exceeds the hard cap.

    Tries comma/semicolon boundaries first, then falls back
    to hard character cuts at the hard cap.
    """
    parts = re.split(r"([,;]\s+)", sentence)
    chunks: list[str] = []
    current = ""
    current_tokens = 0

    i = 0
    while i < len(parts):
        piece = parts[i]
        # Include the delimiter that follows
        if i + 1 < len(parts) and re.match(r"^[,;]\s+", parts[i + 1]):
            piece += parts[i + 1]
            i += 2
        else:
            i += 1

        piece_tokens = estimate_tokens(piece)
        if current_tokens + piece_tokens > HARD_CAP_TOKENS and current:
            chunks.append(current)
            current = piece
            current_tokens = piece_tokens
        else:
            current += piece
            current_tokens += piece_tokens

    if current:
        chunks.append(current)

    return chunks


# ---------------------------------------------------------------------------
# Diff chunker (hunk-level)
# ---------------------------------------------------------------------------

def _split_diff(text: str) -> list[dict[str, Any]]:
    """Split a unified diff into hunk-level chunks.

    Each hunk (starting with ``@@ -A,B +C,D @@``) becomes one
    chunk.  If there's a preamble (file header, ``---``/``+++``
    lines), it's attached to the first hunk.
    """
    if not text or not text.strip():
        return []

    # Find all hunk header positions
    hunk_starts = [m.start() for m in _DIFF_HUNK_RE.finditer(text)]

    if not hunk_starts:
        # No hunk headers — treat the whole thing as one chunk
        return [{
            "text": text,
            "start_byte": 0,
            "end_byte": len(text.encode("utf-8")),
            "language": None,
            "symbol": _extract_diff_symbol(text),
        }]

    chunks: list[dict[str, Any]] = []
    # Preamble (everything before the first @@) goes with the first hunk
    preamble_end = hunk_starts[0]

    for i, start in enumerate(hunk_starts):
        end = hunk_starts[i + 1] if i + 1 < len(hunk_starts) else len(text)
        hunk_text = text[start:end]

        # Include preamble on the first hunk
        if i == 0 and preamble_end > 0:
            hunk_text = text[:start] + hunk_text
            chunk_start = 0
        else:
            chunk_start = start

        # If this hunk is huge (> HARD_CAP), split by context lines
        if estimate_tokens(hunk_text) > HARD_CAP_TOKENS:
            for sub in _split_large_hunk(hunk_text, chunk_start):
                chunks.append(sub)
        else:
            chunks.append({
                "text": hunk_text,
                "start_byte": chunk_start,
                "end_byte": chunk_start + len(hunk_text.encode("utf-8")),
                "language": None,
                "symbol": _extract_diff_symbol(hunk_text),
            })

    return chunks


def _split_large_hunk(hunk_text: str, base_offset: int) -> list[dict[str, Any]]:
    """Split a large diff hunk at blank context lines.

    Groups consecutive changed lines (+ or -) into sub-hunks,
    splitting at context lines (leading space) when the group
    exceeds TARGET_TOKENS.
    """
    lines = hunk_text.splitlines(keepends=True)
    chunks: list[dict[str, Any]] = []
    current: list[str] = []
    current_tokens = 0
    current_offset = base_offset
    pos = base_offset

    for line in lines:
        line_bytes = len(line.encode("utf-8"))
        is_context = line.startswith(" ") or line.startswith("\\") or line.startswith("@@")

        current.append(line)
        current_tokens += estimate_tokens(line)
        pos += line_bytes

        # Split at a context line boundary if we've exceeded the target
        if is_context and current_tokens >= TARGET_TOKENS and len(current) > 1:
            chunks.append({
                "text": "".join(current),
                "start_byte": current_offset,
                "end_byte": pos,
                "language": None,
                "symbol": _extract_diff_symbol("".join(current)),
            })
            current = []
            current_tokens = 0
            current_offset = pos

    if current:
        chunks.append({
            "text": "".join(current),
            "start_byte": current_offset,
            "end_byte": pos,
            "language": None,
            "symbol": _extract_diff_symbol("".join(current)),
        })

    return chunks


def _extract_diff_symbol(text: str) -> str | None:
    """Try to extract a function name from a diff hunk.

    Looks for ``def X``, ``class X``, ``function X`` in the
    added/removed lines.
    """
    # Search added lines first (starting with +)
    for line in text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            m = _SYMBOL_RE.search(line[1:])
            if m:
                return m.group(1)
    # Fall back to context lines
    m = _SYMBOL_RE.search(text)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Mixed-content chunker
# ---------------------------------------------------------------------------

def _split_mixed(text: str, language: str | None = None) -> list[dict[str, Any]]:
    """Split mixed content (code + prose) into chunks.

    Extracts code fences and processes them as code chunks.
    The prose between fences is processed as prose chunks.
    All chunks are ordered by their position in the original text.
    """
    if not text or not text.strip():
        return []

    chunks: list[dict[str, Any]] = []
    pos = 0

    # Find all code fence blocks
    for m in _CODE_FENCE_RE.finditer(text):
        # Prose before this fence
        prose_before = text[pos:m.start()]
        if prose_before.strip():
            for piece in _split_prose(prose_before):
                piece_start = _locate_piece(text, piece, pos)
                chunks.append({
                    "text": piece,
                    "start_byte": piece_start,
                    "end_byte": piece_start + len(piece.encode("utf-8")),
                    "language": None,
                    "symbol": None,
                })

        # Code inside the fence
        code_text = m.group(1)
        # Try to detect language from the fence tag
        fence_lang_match = re.match(r"^```[ \t]*([\w+#-]*)", text[m.start():m.start()+20])
        fence_lang = fence_lang_match.group(1) if fence_lang_match else language

        code_chunks = _ts_extract_code_chunks(code_text, fence_lang)
        code_offset = m.start(1)  # offset of the captured group

        for cc in code_chunks:
            chunks.append({
                "text": cc["text"],
                "start_byte": code_offset + cc["start_byte"],
                "end_byte": code_offset + cc["end_byte"],
                "language": cc.get("language"),
                "symbol": cc.get("symbol"),
            })

        pos = m.end()

    # Trailing prose after the last fence
    if pos < len(text):
        prose_after = text[pos:]
        if prose_after.strip():
            for piece in _split_prose(prose_after):
                piece_start = _locate_piece(text, piece, pos)
                chunks.append({
                    "text": piece,
                    "start_byte": piece_start,
                    "end_byte": piece_start + len(piece.encode("utf-8")),
                    "language": None,
                    "symbol": None,
                })

    # If no fences were found, treat the whole thing as prose
    if not chunks:
        for piece in _split_prose(text):
            piece_start = _locate_piece(text, piece)
            chunks.append({
                "text": piece,
                "start_byte": piece_start,
                "end_byte": piece_start + len(piece.encode("utf-8")),
                "language": None,
                "symbol": None,
            })

    # Sort by position
    chunks.sort(key=lambda c: c["start_byte"])
    return chunks


# ---------------------------------------------------------------------------
# Boundary context assignment
# ---------------------------------------------------------------------------

def _assign_boundary_context(chunks: list[dict[str, Any]], full_text: str) -> None:
    """Assign prev_context and next_context to each chunk.

    For each chunk, the boundary context is drawn from the
    trailing/leading edges of the *adjacent* chunks.  The
    boundary text is ~100 tokens (BOUNDARY_CHARS chars).

    For the first chunk, prev_context is empty.
    For the last chunk, next_context is empty.
    """
    n = len(chunks)
    for i, chunk in enumerate(chunks):
        # prev_context: last ~100 tokens of the preceding chunk
        if i > 0:
            prev_chunk = chunks[i - 1]
            prev_text = prev_chunk["text"]
            if len(prev_text) > BOUNDARY_CHARS:
                chunk["prev_context"] = prev_text[-BOUNDARY_CHARS:]
            else:
                chunk["prev_context"] = prev_text
        else:
            chunk["prev_context"] = ""

        # next_context: first ~100 tokens of the following chunk
        if i < n - 1:
            next_chunk = chunks[i + 1]
            next_text = next_chunk["text"]
            if len(next_text) > BOUNDARY_CHARS:
                chunk["next_context"] = next_text[:BOUNDARY_CHARS]
            else:
                chunk["next_context"] = next_text
        else:
            chunk["next_context"] = ""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def chunk_model_output(
    model_output: str,
    turn_id: str,
    language: str | None = None,
) -> list[Chunk]:
    """Split a model's output into retrieval chunks.

    This is the main entry point for the chunker.  It:

    1. Detects the content kind (code, prose, diff, mixed)
    2. Splits using the appropriate splitter
    3. Assigns boundary context from adjacent chunks
    4. Returns Chunk objects with all metadata

    Parameters
    ----------
    model_output
        The full model output text from a completed turn.
    turn_id
        The ID of the parent turn (used for chunk_id generation).
    language
        Optional language hint for code chunking.  If not
        provided, the chunker will try auto-detection.

    Returns
    -------
    list[Chunk]
        Ordered list of chunks, positioned by their offset
        in ``model_output``.
    """
    if not model_output or not model_output.strip():
        return []

    # Structured-data short-circuit.  ``detect_content_kind`` below
    # falls back to "prose" for any text without code fences or
    # code-shape markers, so a JSON artifact never reaches
    # ``_ts_extract_code_chunks``'s language-aware branch.  This
    # check moves the JSON / JSONL dispatch up one level so the
    # language argument wins regardless of how the kind detector
    # classifies the text.
    if language and language.lower() in ("json", "jsonl"):
        from .ts_backends import parse_json_object_sections, parse_jsonl_sections
        if language.lower() == "json":
            raw_chunks = parse_json_object_sections(model_output)
        else:
            raw_chunks = parse_jsonl_sections(model_output)
        out: list[Chunk] = []
        for raw in raw_chunks:
            out.append(Chunk(
                chunk_id=f"{turn_id}:json:{raw['symbol']}",
                turn_id=turn_id,
                seq=len(out),
                kind="json_value",
                symbol=raw["symbol"],
                text=raw["text"],
                char_start=raw["start_byte"],
                char_end=raw["end_byte"],
            ))
        return out

    kind = detect_content_kind(model_output)

    # If the content is HTML with <script> tags, use the HTML
    # tree-sitter parser to find script elements, then parse each
    # script's content with the JS parser.  This is the clean
    # tree-sitter-native approach — no regex hacks.
    if kind == "code" and _HAS_TREE_SITTER:
        if _looks_like_html(model_output):
            return _chunk_html_with_scripts(model_output, turn_id)

    if kind == "diff":
        raw_chunks = _split_diff(model_output)
    elif kind == "code":
        raw_chunks = _ts_extract_code_chunks(model_output, language)
    elif kind == "mixed":
        raw_chunks = _split_mixed(model_output, language)
    else:  # prose
        raw_chunks = []
        for piece in _split_prose(model_output):
            piece_start = _locate_piece(model_output, piece)
            raw_chunks.append({
                "text": piece,
                "start_byte": piece_start,
                "end_byte": piece_start + len(piece.encode("utf-8")),
                "language": None,
                "symbol": None,
            })

    if not raw_chunks:
        return []

    # Assign boundary context
    _assign_boundary_context(raw_chunks, model_output)

    # Convert to Chunk objects
    chunks: list[Chunk] = []
    for i, rc in enumerate(raw_chunks):
        chunk_kind = "diff_hunk" if kind == "diff" else ("code" if rc.get("language") else "prose")
        chunks.append(Chunk(
            chunk_id=f"{turn_id}:{chunk_kind}:{i}",
            turn_id=turn_id,
            seq=i,
            kind=chunk_kind,
            text=rc["text"],
            prev_context=rc.get("prev_context", ""),
            next_context=rc.get("next_context", ""),
            char_start=rc["start_byte"],
            char_end=rc["end_byte"],
            language=rc.get("language"),
            symbol=rc.get("symbol"),
        ))

    return chunks