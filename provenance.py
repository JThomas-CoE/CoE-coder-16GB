"""Origin-prompt provenance for the origin-gated modify pipeline.

Implements `CoE_coder_spec_modify_pipeline_DRAFT.md`.  Kept in its own module so
the whole feature is reverse-safe: reverting the pipeline is deleting this file
and the (few) call sites that import it.

This module is intentionally PURE where possible -- no I/O, no store access -- so
every rule here is unit-testable without a running server.

Contents
--------
* ``synthesize_combo_prompt`` -- origin + prior modifications + current request
  (the local-regeneration input).
* ``cloud_configured`` / ``modify_paths_available`` -- capability predicate for
  the PATH C message.

(Header framing lands in the next phase.)
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Combo prompt labels.  OURS, never the user's text, so they cannot be confused
# with content.  The labels deliberately read as a build request plus applied
# deltas, because the structured form is what avoids the "growing patch log"
# register that measurably degraded long-spec generations.
# ---------------------------------------------------------------------------
_LBL_ORIGIN = "[ORIGINAL BUILD REQUEST]"
_LBL_MODS = "[SUBSEQUENT MODIFICATIONS ALREADY APPLIED — preserve these behaviours]"
_LBL_CURRENT = "[CURRENT REQUEST — this is the operative change]"


def synthesize_combo_prompt(
    origin_prompt: str | None,
    modifications: list[dict[str, Any]] | None,
    current_request: str,
    *,
    mode: str = "concat",
) -> str | None:
    """Build the prompt for a local regeneration of this lineage.

    Returns the combined prompt, or ``None`` when there is no originating
    prompt.  ``None`` is the structural signal that local regeneration is
    unavailable -- the caller must fall through to the cloud change order.
    Because the unavailability falls out of this function, no separate
    "exclude artifacts with no origin prompt" rule is needed anywhere.

    Parameters
    ----------
    origin_prompt
        The lineage's originating request.  ``None`` => returns ``None``.
    modifications
        The ordered modification log.  Each entry contributes its ``request``
        field.  Entries missing ``request`` are skipped rather than raising, so
        a malformed historical record degrades to "one fewer delta" instead of
        breaking every future modification of that artifact.
    current_request
        The present modification request; always stated last so it reads as the
        operative instruction rather than one more delta.
    mode
        ``"concat"`` (default) -- mechanical, reproducible concatenation.

    Notes
    -----
    The first modification of an artifact with no prior log produces
    ``origin + current`` with the ``[SUBSEQUENT ...]`` block omitted, which is
    exactly the shape measured at 75% reliability in the capability study.
    """
    if not origin_prompt or not origin_prompt.strip():
        return None

    parts: list[str] = [f"{_LBL_ORIGIN}\n{origin_prompt.strip()}"]

    deltas: list[str] = []
    for entry in modifications or []:
        text = (entry or {}).get("request")
        if isinstance(text, str) and text.strip():
            deltas.append(text.strip())
    if deltas:
        # Number the RENDERED deltas contiguously (1..N), not by raw log index:
        # a skipped malformed entry would otherwise leave a confusing gap such
        # as "2. keep me" with no "1.", which reads as a missing instruction to
        # the model.  The prompt is for the model; log positions are for us.
        parts.append(
            f"{_LBL_MODS}\n"
            + "\n".join(f"{i}. {d}" for i, d in enumerate(deltas, start=1))
        )

    current = (current_request or "").strip()
    if current:
        parts.append(f"{_LBL_CURRENT}\n{current}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Artifact file header -- payload format
# ---------------------------------------------------------------------------
# The header carries the prompt history.  It MUST round-trip byte-exact
# regardless of what the user typed, because it is the reproducibility key for
# local regeneration.  Hence LENGTH-PREFIXED, not delimiter-only: any in-band
# delimiter can occur inside a user prompt, whereas a length prefix cannot
# collide with content.
_HEADER_OPEN = "===COE-ARTIFACT-HEADER v1==="
_HEADER_CONTENT = "===COE-ARTIFACT-CONTENT==="
_FRAME_PREFIX = "[FRAME] "

# The [FRAME] value is SYMBOLIC, and that is deliberate.  The draft's literal
# form -- ``[FRAME] block:<open>|<close>`` -- cannot be written for the very
# cases it exists to serve: an HTML block payload containing ``<!--|-->`` is
# itself hit by the ``--`` escape, so the frame line would be rewritten to
# ``<!-\u200b-|-\u200b->`` and could no longer be read WITHOUT first knowing the
# escape -- circular.  Same for CSS and ``*/``.  Verified by measurement.
# Symbolic tokens contain no trigger character at all, so the declaration is
# always written and read verbatim, whatever the framing does to the payload.
#
# ``class`` is the primary key (it says which framing was applied); the optional
# detail after the colon is advisory, because the token tables below are
# authoritative and a stale detail must never defeat a reader.
_FRAME_SYMBOLS: dict[str, str] = {
    "#": "hash", "//": "slashes", "--": "dashes", ";": "semis", "%": "percent",
    "<!--": "html", "/*": "css",
}


def _frame_descriptor(ext: str) -> tuple[str, str]:
    """Declare how a payload for ``ext`` was framed: ``(class, symbol)``.

    Write-side helper for :func:`build_payload`.  It is intentionally derived
    from the SAME ``framing_for_extension`` call that :func:`attach_header` uses
    to frame the bytes, so the declaration cannot drift from what was actually
    applied.

    The symbol is EMPTY for ``plain`` and ``none``: those classes have no open
    token, so naming one would be meaningless -- ``[FRAME] none:plain`` reads as
    "no framing, symbolically plain" and is simply wrong.  The class alone is
    the complete declaration for both.
    """
    cls, open_tok, _close_tok = framing_for_extension(ext)
    return cls, _FRAME_SYMBOLS.get(open_tok, "")


def build_payload(
    origin_prompt: str | None,
    modifications: list[dict[str, Any]] | None,
    frame: tuple[str, str] | None = None,
) -> str:
    """Render the header payload as a single length-prefixed text blob.

    LENGTH IS IN CHARACTERS, and the text FOLLOWS the newline that ends the
    length field.  So parsing reads a length, consumes one newline, then takes
    exactly that many characters -- which cannot collide with any content, so
    a prompt containing the sentinel, ``[MOD]``, or blank lines is still exact.

    ``frame`` is the ``(class, symbol)`` pair from :func:`_frame_descriptor`.
    When it is omitted the payload carries NO ``[FRAME]`` line, which every
    reader tolerates -- so existing callers, existing unit tests, and every
    payload already on disk stay valid and byte-identical.
    """
    out = [_HEADER_OPEN]
    if frame is not None:
        _fcls, _fsym = frame
        # ``line`` and ``block`` NEED their token symbol -- it is what tells a
        # reader which comment syntax framed the payload.  ``plain`` and
        # ``none`` have no token, so they are declared as the class alone.  If a
        # caller ever hands back a `plain:`+symbol pair, the class alone is
        # still emitted rather than a misleading `plain:plain`.
        if _fsym and _fcls in ("line", "block"):
            out.append(f"{_FRAME_PREFIX}{_fcls}:{_fsym}")
        else:
            out.append(f"{_FRAME_PREFIX}{_fcls}")
    if origin_prompt:
        out.append(f"[ORIGIN] {len(origin_prompt)}")
        out.append(origin_prompt)
    for entry in modifications or []:
        text = (entry or {}).get("request")
        if not isinstance(text, str) or not text:
            continue
        method = str((entry or {}).get("method") or "")
        model = str((entry or {}).get("model") or "")
        ts = str((entry or {}).get("ts") or "")
        out.append(f"[MOD] {method} | {model} | {ts} | {len(text)}")
        out.append(text)
    out.append(_HEADER_CONTENT)
    return "\n".join(out)


def parse_payload(blob: str) -> dict[str, Any]:
    """Parse a payload blob.  Character-length driven; never delimiter-searched."""
    origin_prompt: str | None = None
    modifications: list[dict[str, Any]] = []
    frame: tuple[str, str] | None = None
    pos = 0

    # skip the opening sentinel line
    nl = blob.find("\n")
    if nl == -1:
        return {"origin_prompt": None, "modifications": []}
    if blob[:nl].strip() != _HEADER_OPEN:
        return {"origin_prompt": None, "modifications": []}
    pos = nl + 1

    while pos < len(blob):
        nl = blob.find("\n", pos)
        if nl == -1:
            break
        header = blob[pos:nl]
        body_start = nl + 1

        if header.startswith("[ORIGIN] "):
            try:
                n = int(header[len("[ORIGIN] "):].strip())
            except ValueError:
                pos = body_start
                continue
            origin_prompt = blob[body_start:body_start + n]
            pos = body_start + n
            if pos < len(blob) and blob[pos] == "\n":
                pos += 1
            continue

        if header.startswith(_FRAME_PREFIX):
            # ``[FRAME] <class>`` or ``[FRAME] <class>:<symbol>``.  ADVISORY:
            # a malformed or unknown value is recorded as-is and never fatal,
            # because the reader can always fall back on the caller's
            # extension.  This is also why the class is parsed leniently rather
            # than validated against a closed set here.
            _decl = header[len(_FRAME_PREFIX):].strip().lower()
            _dcls, _sep, _dsym = _decl.partition(":")
            frame = (_dcls.strip(), _dsym.strip() if _sep else "")
            pos = body_start
            continue

        if header.startswith("[MOD] "):
            meta = header[len("[MOD] "):]
            parts = [p.strip() for p in meta.split("|")]
            try:
                n = int(parts[-1])
            except (IndexError, ValueError):
                pos = body_start
                continue
            text = blob[body_start:body_start + n]
            modifications.append({
                "request": text,
                "method": parts[0] if len(parts) > 0 else "",
                "model": parts[1] if len(parts) > 1 else "",
                "ts": parts[2] if len(parts) > 2 else "",
            })
            pos = body_start + n
            if pos < len(blob) and blob[pos] == "\n":
                pos += 1
            continue

        if header.strip() == _HEADER_CONTENT:
            break
        pos = body_start

    # `frame` is None for every payload written before this field existed, so
    # callers must treat it as optional.  It is deliberately NOT defaulted to a
    # tuple: "no declaration" and "declared plain" are different facts.
    return {"origin_prompt": origin_prompt, "modifications": modifications,
            "frame": frame}


# ---------------------------------------------------------------------------
# Framing classes (D6): make the header invisible to the language's parser.
# ---------------------------------------------------------------------------
# Selected by FILE EXTENSION.  See the draft's section 2.4.4 / 2.4.4.1.
#
#   line:<token>      payload VERBATIM, each line token-prefixed.
#                     A line comment has NO terminator, so arbitrary user text
#                     (including --, -->, */) is harmless.  Best case, and it
#                     covers every compiled language.
#   block:<o>|<c>     HTML <!--|--> and CSS /*|*/.  These DO have terminators,
#                     so the payload is escaped only when a trigger is present.
#   plain             md/txt/prose: verbatim, deliberately VISIBLE (allowed).
#   none              no comment syntax (json/csv/unknown): header NOT written
#                     to the file; the side-car carries it.
#
# ESCAPING (block only, conditional on the trigger):
#   html trigger "--"  (covers -->, --!>, and the spec's ban on "--" in text)
#   css  trigger "*/"
# A payload with no trigger is written verbatim, readable and valid -- the
# common case.  Base64 is used only for long payloads, where the per-line
# escape would roughly double the length.
_BLOCK_ESCAPE_THRESHOLD = 400
_ZERO_WIDTH = "\u200b"

# extension -> (class, open, close)
_FRAMING: dict[str, tuple[str, str, str]] = {
    ".py": ("line", "#", ""), ".sh": ("line", "#", ""), ".yaml": ("line", "#", ""),
    ".yml": ("line", "#", ""), ".rb": ("line", "#", ""), ".pl": ("line", "#", ""),
    ".js": ("line", "//", ""), ".ts": ("line", "//", ""), ".jsx": ("line", "//", ""),
    ".tsx": ("line", "//", ""), ".c": ("line", "//", ""), ".cpp": ("line", "//", ""),
    ".h": ("line", "//", ""), ".hpp": ("line", "//", ""), ".rs": ("line", "//", ""),
    ".go": ("line", "//", ""), ".java": ("line", "//", ""), ".kt": ("line", "//", ""),
    ".scala": ("line", "//", ""), ".swift": ("line", "//", ""),
    ".sql": ("line", "--", ""), ".lua": ("line", "--", ""), ".hs": ("line", "--", ""),
    ".lisp": ("line", ";", ""), ".asm": ("line", ";", ""), ".ini": ("line", ";", ""),
    ".tex": ("line", "%", ""), ".m": ("line", "%", ""), ".erl": ("line", "%", ""),
    ".html": ("block", "<!--", "-->"), ".htm": ("block", "<!--", "-->"),
    ".xml": ("block", "<!--", "-->"), ".svg": ("block", "<!--", "-->"),
    ".css": ("block", "/*", "*/"),
    ".md": ("plain", "", ""), ".markdown": ("plain", "", ""),
    ".txt": ("plain", "", ""), ".rst": ("plain", "", ""),
    ".json": ("none", "", ""), ".csv": ("none", "", ""),
}

# Preamble lines the header must NOT precede.  A header placed before these
# breaks the file in ways that look like unrelated bugs: a shell script stops
# executing, an XML file fails to parse, a Go build constraint is neutered.
_PREAMBLE_PATTERNS = (
    "#!",                      # shebang (shell, python, perl, ruby)
    "# -*-",                   # python encoding declaration
    "# coding:",               # python encoding declaration (PEP 263)
    "//go:build",              # Go build constraint
    "// +build",               # Go build constraint (legacy)
    "<?xml",                   # XML declaration: a comment before it is illegal
    "<!DOCTYPE",               # html doctype
    "<!doctype",
)


def framing_for_extension(ext: str) -> tuple[str, str, str]:
    """Return (class, open, close) for a file extension.  Unknown -> 'none'."""
    return _FRAMING.get((ext or "").lower(), ("none", "", ""))


def _escape_block(text: str, close: str) -> tuple[str, str]:
    """Escape ``text`` so it cannot terminate a block comment.

    Returns (encoded_text, encoding) where encoding is
    ``verbatim`` | ``zwsp`` | ``base64``.
    """
    trigger = "--" if close == "-->" else "*/"
    if trigger not in text:
        return text, "verbatim"
    if len(text) > _BLOCK_ESCAPE_THRESHOLD:
        import base64
        return base64.b64encode(text.encode("utf-8")).decode("ascii"), "base64"
    # Break the run with a zero-width space: the parser never sees the trigger,
    # while the text still DISPLAYS as a normal hyphen/slash run.  Chosen over
    # an ASCII substitute because it does not invent visible characters.
    if trigger == "--":
        return text.replace("--", "-" + _ZERO_WIDTH + "-"), "zwsp"
    return text.replace("*/", "*" + _ZERO_WIDTH + "/"), "zwsp"


def _unescape_block(text: str, encoding: str) -> str:
    if encoding == "base64":
        import base64
        return base64.b64decode(text.encode("ascii")).decode("utf-8")
    if encoding == "zwsp":
        return text.replace(_ZERO_WIDTH, "")
    return text


def _preamble_len(body: str) -> int:
    """Number of leading characters to keep BEFORE the header.

    A header placed ahead of these breaks the file in ways that look like
    unrelated bugs, so they must stay first:

    * ``#!``            shebang -- if not line 1 the script will not execute
    * ``# -*-``/``# coding:``  Python encoding declaration (must be line 1-2)
    * ``<?xml``         a comment BEFORE the XML declaration is illegal
    * ``<!DOCTYPE``     historically triggers quirks mode
    * ``//go:build``/``// +build``  Go build constraint.  Go requires the
      constraint to be followed by a BLANK LINE, so we keep the constraint AND
      the blank line that terminates it before inserting the header.
    """
    lines = body.split("\n")
    if not lines:
        return 0
    first = lines[0].strip()

    def consumed(count: int) -> int:
        # length of the first `count` lines plus their newlines
        return sum(len(lines[i]) + 1 for i in range(min(count, len(lines))))

    if first.startswith("#!"):
        idx = 1
        # keep a following Python encoding declaration too (line 2)
        if len(lines) > idx:
            second = lines[idx].strip()
            if second.startswith("# -*-") or second.startswith("# coding:"):
                idx += 1
        return consumed(idx)
    if first.startswith("# -*-") or first.startswith("# coding:"):
        return consumed(1)
    if first.startswith("<?xml") or first.startswith("<!DOCTYPE") \
            or first.startswith("<!doctype"):
        return consumed(1)
    if first.startswith("//go:build") or first.startswith("// +build"):
        idx = 1
        # the constraint block ends at the first blank line, which is REQUIRED
        # after it; consume that blank line with the constraint.
        while idx < len(lines) and lines[idx].strip() != "":
            idx += 1
        if idx < len(lines):
            idx += 1
        return consumed(idx)
    return 0


def attach_header(
    body: str,
    origin_prompt: str | None,
    modifications: list[dict[str, Any]] | None,
    ext: str,
) -> tuple[str, dict[str, Any]]:
    """Return (file_text, info) with the header framed into ``body``.

    ``info`` reports what happened so a caller can surface defects:
      ``framed``/``class``/``encoding``/``reason``, plus ``payload_bytes``.
    """
    payload = build_payload(origin_prompt, modifications, _frame_descriptor(ext))
    cls, open_tok, close_tok = framing_for_extension(ext)
    info: dict[str, Any] = {"framed": False, "class": cls, "encoding": "n/a",
                            "reason": "", "payload_bytes": len(payload)}

    if cls == "none":
        # No comment syntax (json/csv/unknown): the header is NOT written to
        # the file.  This is a DEFECT SIGNAL, not a normal state -- it means
        # language detection already failed or the artifact was loaded without
        # a language.  The caller must surface it rather than absorb it.
        info["reason"] = f"no comment syntax for extension {ext!r}"
        return body, info

    # ---- frame the WHOLE payload, content sentinel included, so nothing
    # ---- escapes the comment and the stripped body is byte-exact.
    if cls == "line":
        # A line comment has NO terminator, so verbatim payload is always safe.
        # This is the readable-and-invisible case, and it covers every compiled
        # language.
        framed = "\n".join(f"{open_tok} {ln}" if ln else open_tok
                           for ln in payload.split("\n"))
        info["encoding"] = "verbatim"
    elif cls == "block":
        # Block comments DO have terminators (`-->`, `*/`), so the payload is
        # escaped only when a trigger is present.
        text, enc = _escape_block(payload, close_tok)
        info["encoding"] = enc
        framed = f"{open_tok}\n{text}\n{close_tok}"
    else:  # plain -- md/txt/prose: verbatim, deliberately VISIBLE (allowed)
        framed = payload
        info["encoding"] = "verbatim"

    keep = _preamble_len(body)
    prefix, rest = body[:keep], body[keep:]
    info["framed"] = True
    return prefix + framed + "\n" + rest, info


def strip_header(file_text: str, ext: str) -> tuple[str, dict[str, Any]]:
    """Remove the header, returning (body, parsed_provenance).

    The header must never reach section indexing or a model call: if it did, the
    cloud could anchor on or edit a provenance section and the outline would
    gain a section that is not part of the artifact.
    """
    cls, open_tok, close_tok = framing_for_extension(ext)
    if cls == "none":
        return file_text, {"origin_prompt": None, "modifications": []}

    keep = _preamble_len(file_text)
    prefix, rest = file_text[:keep], file_text[keep:]
    payload = ""
    body = rest

    if cls == "line":
        pre = f"{open_tok} "
        if rest.startswith(pre + _HEADER_OPEN):
            collected = []
            for ln in rest.split("\n"):
                if ln.startswith(pre):
                    collected.append(ln[len(pre):])
                    continue
                if ln == open_tok:          # blank payload line
                    collected.append("")
                    continue
                break
            payload = "\n".join(collected)
            consumed = len("\n".join(
                (pre + c) if c else open_tok for c in collected)) + 1
            body = rest[consumed:]
    elif cls == "block":
        if rest.startswith(open_tok + "\n"):
            end = rest.find("\n" + close_tok)
            if end != -1:
                raw = rest[len(open_tok) + 1:end]
                payload = _decode_block(raw)
                body = rest[end + len(close_tok) + 2:]
    else:  # plain
        if rest.startswith(_HEADER_OPEN + "\n"):
            end = rest.find("\n" + _HEADER_CONTENT)
            if end != -1:
                payload = rest[:end]
                body = rest[end + len(_HEADER_CONTENT) + 2:]

    parsed = (parse_payload(payload) if payload
              else {"origin_prompt": None, "modifications": []})
    return prefix + body, parsed


def _decode_block(raw: str) -> str:
    """Reverse `_escape_block` without needing the encoding recorded.

    Order matters.  The zero-width-space strip is tried FIRST: an escaped
    payload still contains the sentinel (the sentinel itself has no `--`), so
    testing "does this look like a payload" before un-escaping returns the
    escaped text intact and silently leaves U+200B in every prompt.
    """
    stripped = raw.replace(_ZERO_WIDTH, "")
    if stripped.startswith(_HEADER_OPEN):
        return stripped
    if raw.startswith(_HEADER_OPEN):
        return raw
    import base64 as _b64
    try:
        return _b64.b64decode(raw.encode("ascii")).decode("utf-8")
    except Exception:
        return raw
def cloud_configured(backends_cfg: dict[str, Any] | None) -> bool:
    """True if at least one cloud model is ACCESSIBLE PER CONFIGURATION.

    Deliberately model- and provider-AGNOSTIC: this must not name a specific
    model id, nor assume any particular provider or transport.  It only asks
    whether the configuration makes at least one cloud model reachable, so
    changing the model or the provider does not affect the answer.

    Reads the configured ``backends.cloud`` block.  It must NOT read
    ``DEFAULT_DIRECT_CLOUD_CONFIG`` -- that is a code-level T3 default which
    exists regardless of configuration, so it would report "available" for an
    unconfigured deployment.
    """
    cloud = (backends_cfg or {}).get("cloud") or {}
    if not cloud.get("enabled"):
        return False
    if not cloud.get("url"):
        return False
    # A bare ``model_id`` satisfies "at least one cloud model".  ``vision_model_id``
    # is an ADDITIONAL capability, not a substitute, but either one being present
    # means a cloud model is reachable.
    return bool(cloud.get("model_id") or cloud.get("vision_model_id"))


def modify_paths_available(
    origin_prompt: str | None,
    backends_cfg: dict[str, Any] | None,
) -> dict[str, bool]:
    """Which modify routes are available for this artifact.

    ``origin_regen``  -- local regeneration is possible (origin prompt known).
    ``cloud_order``   -- cloud change order is possible (cloud configured).

    Both False means modify is unavailable and the caller must emit the
    capability message rather than attempting any model call.
    """
    return {
        "origin_regen": bool(origin_prompt and origin_prompt.strip()),
        "cloud_order": cloud_configured(backends_cfg),
    }


def modify_capability_message(available: dict[str, bool]) -> str | None:
    """The PATH C message, or None when modify is actually available.

    Capability-based rather than cloud-based, and it names the REMEDY, so an
    "unavailable" answer is actionable rather than a dead end.  Returned only
    in the neither-available case: when either route works the caller proceeds
    silently.
    """
    if available.get("origin_regen") or available.get("cloud_order"):
        return None
    return (
        "Modification is unavailable for this artifact.\n\n"
        "This artifact carries no originating prompt, so it cannot be modified "
        "by local regeneration; and no cloud model is configured, so it cannot "
        "be modified by a cloud change order either.\n\n"
        "To enable modification, either:\n"
        "  * supply the artifact's originating prompt, or\n"
        "  * configure a cloud model (backends.cloud)."
    )


# ---------------------------------------------------------------------------
# Drift checks (section 6 of the draft)
# ---------------------------------------------------------------------------
# THREE separate mechanisms with DIFFERENT dispositions.  The asymmetry is the
# design: a CERTAIN, cheap signal is automated; a HEURISTIC, costly one only
# informs.  Automating the heuristic would produce false escalations, because a
# legitimate modification is allowed to change a constant.
#
#   (a) GATE       requested modification absent      -> AUTO-ESCALATE to cloud
#   (b) FLAG       base values/capabilities differ    -> report, NEVER escalate
#   (c) DISCLOSURE parent has >=1 cloud change order  -> show BEFORE generation

_NUM_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])")


def extract_constants(text: str) -> dict[str, int]:
    """Multiset of numeric literals in ``text`` (Tier-1 drift input).

    Deliberately crude and cheap: it needs no parsing and no execution, and it
    catches the class of failure that actually occurred in the capability study
    (a defining parameter silently changing value, e.g. 500 K -> 600 K).
    """
    out: dict[str, int] = {}
    for m in _NUM_RE.finditer(text or ""):
        out[m.group(1)] = out.get(m.group(1), 0) + 1
    return out


def tier1_drift(parent_text: str, child_text: str) -> dict[str, Any]:
    """Cheap literal diff between parent and child.

    REPORT ONLY -- never escalates (see the module comment above).  A parent
    literal that has vanished from the child is the interesting signal; new
    literals are reported separately because adding a constant is usually
    exactly what the modification asked for.
    """
    p = extract_constants(parent_text)
    c = extract_constants(child_text)
    vanished = {k: v for k, v in p.items() if k not in c}
    added = {k: v for k, v in c.items() if k not in p}
    changed_count = {k: (p[k], c[k]) for k in p if k in c and p[k] != c[k]}
    return {
        "vanished": vanished,
        "added": added,
        "count_changed": changed_count,
        "drift": bool(vanished or changed_count),
        "action": "review",          # never "escalate"
    }


def modification_present(artifact_text: str, request_text: str,
                         *, min_hit_ratio: float = 0.34) -> bool:
    """Cheap presence test for the requested modification (the (a) GATE).

    NOT a semantic check -- it looks for the distinctive tokens of the request
    appearing in the produced artifact.

    The threshold is PROPORTIONAL (a third of the distinctive words) rather than
    an absolute count.  An absolute count of 2 is far too strict for a short
    request: "add linear drag resistance" has only three distinctive words, so
    demanding two hits fails a document that already says "drag".  A ratio
    adapts to request length instead of penalising brevity.

    Bias is toward "present": a false negative triggers a cloud escalation
    (safe, but it costs a call), while a false positive would silently accept an
    artifact missing the requested change.  Given the escalation is the trusted
    route, the ratio is set low enough that ordinary wording overlap passes.
    """
    words = [w for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", (request_text or "").lower())
             if w not in _STOPWORDS]
    distinct = set(words)
    if not distinct:
        return True                    # nothing distinctive to look for
    low = (artifact_text or "").lower()
    hits = sum(1 for w in distinct if w in low)
    need = max(1, int(len(distinct) * min_hit_ratio))
    return hits >= need


# KNOWN LIMITATION (documented, not hidden): this gate is LEXICAL, so a request
# phrased with synonyms of the artifact's own wording can read as "absent" even
# when the modification was made.  Example: a request saying "air resistance"
# against an artifact that calls it "drag" scores zero hits.
#
# The failure direction is deliberately SAFE: a false negative triggers a cloud
# escalation, which is the route the pipeline already trusts, whereas a false
# positive would silently accept an artifact missing the requested change.  The
# caller must therefore treat "absent" as ESCALATE, never as "reject".
# Improving this would mean a semantic check (embedding similarity or a model
# call) -- explicitly out of scope, because the study's conclusion is that
# paying for judgement is what the cloud change order already does better.


_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "should", "must", "when",
    "then", "also", "have", "has", "are", "was", "were", "you", "your",
    "please", "make", "add", "new", "use", "using", "all", "any", "each",
    "from", "into", "onto", "not", "but", "can", "will", "would", "it",
    "its", "they", "them", "their", "there", "here", "which", "what", "who",
    "how", "why", "where", "been", "being", "does", "did", "doing", "done",
    "get", "got", "let", "put", "set", "now", "only", "more", "most", "some",
    "same", "other", "such", "very", "just", "like",
}


def cloud_modification_count(modifications: list[dict[str, Any]] | None) -> int:
    """How many entries in the log were served by a cloud change order."""
    return sum(1 for m in (modifications or [])
               if (m or {}).get("method") == "cloud_change_order")


def basis_label(origin_prompt: str | None,
                modifications: list[dict[str, Any]] | None) -> str:
    """``origin`` | ``origin+mods(N cloud)`` | ``none``.

    ``origin+mods(N cloud)`` means the regeneration basis UNDER-DESCRIBES the
    parent, because each stored entry is a request rather than a record of what
    the cloud actually changed.  Surfaced as the (c) DISCLOSURE.
    """
    if not origin_prompt:
        return "none"
    n = cloud_modification_count(modifications)
    return f"origin+mods({n} cloud)" if n else "origin"


def approximation_disclosure(origin_prompt: str | None,
                             modifications: list[dict[str, Any]] | None) -> str | None:
    """The (c) DISCLOSURE text, or None when no caveat is owed.

    Free and factual -- a property of the INPUT, shown before generation, so it
    cannot fail and never escalates.
    """
    n = cloud_modification_count(modifications)
    if not origin_prompt or n == 0:
        return None
    return (
        f"Regenerating from the originating prompt and {n} prior cloud "
        f"modification(s). Each stored modification is a REQUEST, not a record "
        f"of what the cloud actually changed, so this result approximates the "
        f"cloud-edited parent rather than reproducing it."
    )


def record_modification(
    modifications: list[dict[str, Any]] | None,
    request_text: str,
    method: str,
    model: str = "",
    *,
    ts: str | None = None,
) -> list[dict[str, Any]]:
    """Return a NEW log with this modification APPENDED (never replaces).

    Append-only is what keeps a cloud-edited artifact locally modifiable: the
    origin prompt survives, so a later modification still has a basis to
    regenerate from.  Replacing instead would permanently downgrade the
    artifact to cloud-only.
    """
    import datetime as _dt
    entry = {
        "request": request_text,
        "method": method,
        "model": model,
        "ts": ts or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return list(modifications or []) + [entry]
