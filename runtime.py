"""Declarative, domain-neutral clean runtime primitives.

The runtime is responsible for three things and three things only:

1. Walking the harness's tier sequence and calling the transport once
   per tier.
2. Running the YAML-declared guards and contracts against the streamed
   output and deciding whether the tier ``passed``.
3. Emitting progress events so a client can follow the run live, and
   asking a control callback for a decision at configurable
   checkpoint intervals.

The runtime never has a hard wall-clock timeout.  A difficult bug fix
is allowed to run for as long as the underlying model and Ollama need.
Cancellation is the client's responsibility: a callback may return
``abort`` or ``accept_partial`` from a checkpoint event, and the
runtime stops with the work product it has so far.
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
import secrets
import sys
import time
from collections.abc import AsyncIterator
from typing import Any

from .contracts import (
    CleanAttempt,
    CleanControlDecision,
    CleanExecutionResult,
    CleanPackage,
    CleanProgressCallback,
    CleanProgressEvent,
    CleanTierContext,
    CleanUnit,
    NullProgressCallback,
)
from .pathology import DEFAULT_SAFETY_FACTOR, PathologyMonitor, StallMonitor
from .transport import CloudStreamSilent
from .vram import PrepareTimeout
from .session_memory import SessionMemory, build_last_turn_section

# (2026-09-08) R1-b warm-schema rescue DELETED per user ruling: the ladder
# is R1 -> R1_retry -> R2 -> secondary model -> cloud. The rescue's unique
# case (schema-class failure with salvageable envelope) is better served
# by retries and by T3's combined targeting+editing cloud call.


# Vision-token contribution to the TTFT budget.  One vision token per
# merged 28x28-pixel cell (Qwen-VL lineage); the LLM prefills those
# embeddings at the SAME per-model measured rate as text tokens (the
# transformer forward pass is identical), so they simply join the
# text token estimate.  Per-image cap keeps pathological dimensions
# (e.g. 8000x8000 = 81K tokens uncapped) from dominating the budget.
# 4096 covers realistic screenshots (1080p = 2645, 4K = 10671 is
# beyond what Ollama/llama.cpp keeps un-resized), while an
# under-count is the only risky direction -- a slightly generous
# budget just extends a deadline.
_VISION_CELL_PIXELS = 28 * 28  # 784
_VISION_TOKENS_PER_IMAGE_CAP = 4096

# Max bytes scanned for image-dimension headers.  PNG IHDR sits in the
# first 33 bytes; JPEG SOF markers are walked via segment lengths, so a
# bounded prefix is enough for sane images and keeps the estimator O(1).
_IMAGE_HEADER_SCAN_BYTES = 65536


def _image_dimensions(image_str: str) -> tuple[int, int] | None:
    """Extract (width, height) from PNG or JPEG image data.

    Accepts data-URI or raw base64 forms (the shapes produced by
    ``_normalize_image_reference``).  Returns None for anything that
    is not a decodable PNG/JPEG header — the estimator then simply
    adds zero for that image (fail-open, like every other heuristic
    here; the safety factor still applies to the text estimate).
    """
    if image_str.startswith("data:"):
        _header, _sep, image_str = image_str.partition(",")
    try:
        raw = base64.b64decode(image_str, validate=False)
    except Exception:
        return None
    if len(raw) >= 33 and raw[:8] == b"\x89PNG\r\n\x1a\n" and raw[12:16] == b"IHDR":
        width = int.from_bytes(raw[16:20], "big")
        height = int.from_bytes(raw[20:24], "big")
        if 0 < width < 1_000_000 and 0 < height < 1_000_000:
            return width, height
        return None
    if len(raw) >= 4 and raw[:2] == b"\xff\xd8":  # JPEG SOI
        offset = 2
        end = min(len(raw), _IMAGE_HEADER_SCAN_BYTES)
        while offset + 9 <= end:
            if raw[offset] != 0xFF:
                offset += 1
                continue
            marker = raw[offset + 1]
            # SOF0/1/2/3/5/6/7/9/10/11/13/14/15 carry dimensions; DHT(0xC4)/DAC(0xCC) excluded.
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                          0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                height = int.from_bytes(raw[offset + 5:offset + 7], "big")
                width = int.from_bytes(raw[offset + 7:offset + 9], "big")
                if 0 < width < 1_000_000 and 0 < height < 1_000_000:
                    return width, height
                return None
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                offset += 2  # standalone markers, no length field
                continue
            seg_len = int.from_bytes(raw[offset + 2:offset + 4], "big")
            if seg_len < 2:
                return None
            offset += 2 + seg_len
    return None


def _estimate_package_tokens(package: CleanPackage) -> int:
    """Rough token count of the prompt package.

    Heuristic: total characters / 4.  This is the
    GPT-style BPE approximation; Qwen/Gemma BPE on
    code-heavy text is closer to 3 chars/token.  The
    ``safety_factor`` on the resulting TTFT budget
    (default 2.0) absorbs the error.

    Images contribute their Qwen-VL vision-token count — one token
    per merged 28x28 cell, capped per image — so the prefill-rate
    table's per-model measured rate applies to them exactly like
    text tokens (the LLM forward pass is identical for vision
    embeddings).

    Returns at least 1 so we never divide by zero.
    """
    total_chars = 0
    for msg in package.messages:
        content = msg.get("content") or ""
        if isinstance(content, str):
            total_chars += len(content)
    vision_tokens = 0
    for image in getattr(package, "images", None) or []:
        dims = _image_dimensions(image)
        if dims is None:
            continue
        width, height = dims
        per_image = min(
            -(-width * height // _VISION_CELL_PIXELS),  # ceil
            _VISION_TOKENS_PER_IMAGE_CAP,
        )
        vision_tokens += max(0, per_image)
    return max(1, (total_chars // 4) + vision_tokens)


def dither_guard(text: str, runtime: dict[str, Any], think_on: bool = False) -> str | None:
    # Principle: dither is a *revision discourse marker*, not a
    # lexical item.  Productive reasoning uses connectives like
    # "first", "next", "so", "therefore"; dithering is the model
    # going back on what it just said ("wait", "actually",
    # "hmm", "instead", "let me reconsider").  Every alternation
    # below is a self-contained marker that signals the model
    # is abandoning a prior commitment and reaching for a new
    # one.  A few productive mentions are normal (the model
    # double-checks an example); a runaway is when the same
    # family of markers keeps firing past the threshold with no
    # commit (no closing ``</think>`` / ``</patch>`` / FINAL
    # block).
    #
    # The default limit is 35.  Empirically (see the
    # HumanEval/103 vs HumanEval/162 ablation):
    #
    #   * 103 (PASS, productive dither)  : 21 wide matches at
    #     3,034 chars (6.92 / KB).
    #   * 162 (PASS, single-shot)        : 27 wide matches at
    #     5,407 chars (4.99 / KB).
    #   * 162 (FAIL, runaway dither)     : 515 wide matches at
    #     51,537 chars (9.99 / KB).
    #
    # 35 cleanly separates the passing responses (21, 27) from
    # the runaway (515).  With the v1 narrow pattern (``wait,
    # I*`` or ``wait[/./!/;] \w``) the runaway took 25-28K
    # chars to trip; with the wide pattern it trips at ~4,700
    # chars — saving ~25K chars of wasted generation per
    # dither case.
    pattern = str(
        runtime.get(
            "dither_pattern",
            # Re-vision discourse markers.  Each is a
            # self-contained word/phrase that signals the model
            # is reaching for a different answer.  Excludes
            # productive connectives ("first", "next", "so",
            # "therefore") and productive request forms ("let
            # me read", "let me check").
            r"(?i)\b(?:"
            r"wait"
            r"|actually"
            r"|hmm"
            r"|sorry"
            r"|instead"
            r"|reconsider(?:ing|s|ed)?"
            r"|re-?examin(?:e|ing|ed)"
            r"|but\s+wait"
            r"|on\s+(?:second\s+)?thought"
            r"|let\s+me\s+(?:re-?think|re-?do|try\s+again|reconsider)"
            r")\b",
        )
    )
    # When the slot is think-on, the model's native reasoning trace
    # legitimately contains revision markers ("wait", "actually",
    # "let me reconsider") that are NOT dithering — they are the
    # model working through the problem.  Apply a configurable
    # multiplier to the dither limit so a think-on model is not
    # cut off mid-reasoning.  Default scale 3 (user-set); 1 = no
    # change (think-off behavior).
    limit = int(runtime.get("dither_limit", 35))
    if think_on:
        limit = int(limit * float(runtime.get("dither_limit_think_scale", 3)))
    return "dither" if len(re.findall(pattern, text)) >= limit else None


def repetition_loop_guard(text: str, runtime: dict[str, Any]) -> str | None:
    minimum = int(runtime.get("loop_guard_min", 1400))
    if len(text) < minimum:
        return None
    body = text[int(runtime.get("loop_guard_skip", 1400)):]
    match_size = int(runtime.get("loop_guard_match", 100))
    window = int(runtime.get("loop_guard_window", 400))
    for index in range(0, max(0, len(body) - match_size), 50):
        chunk = body[index:index + match_size]
        if not chunk:
            continue
        # 2026-09-07: a single repeat within the window is NOT enough —
        # legitimate code repeats near-identical blocks (ascending +
        # descending sort branches, symmetric CSS rules).  The captured
        # false positive: T3's valid kanban edit package tripped on the
        # asc/desc sort-map pair.  A genuine degenerate loop repeats the
        # SAME chunk MANY times, so require >=3 occurrences of the chunk
        # in the body overall.  (Validated: real T3 capture -> None; a
        # synthetic 40x verbatim loop -> "loop"; a 2-occurrence legit
        # repeat -> None.)
        if chunk in body[index + match_size:index + window]:
            if body.count(chunk) >= 3:
                return "loop"
    return None


def modify_output_cap_guard(text: str, runtime: dict[str, Any]) -> str | None:
    """Modify-only output cap (2026-09-10, user ruling).

    A change order that runs past the artifact size is wrong by
    construction: single-domain single-task modify work produces change
    orders SMALLER than their target artifact.  When the accumulated
    stream text exceeds ``output_cap_factor * artifact_chars`` (floored
    at ``output_cap_min_chars`` so tiny artifacts never cap legitimate
    rewrites), declare the ``output_cap_exceeded`` pathology.

    The trip handler in the tier loop performs the rescue step: extract
    the LAST COMPLETE change order (JSON or unified diff) from the
    partial text, validate against the round2 contract, and either
    accept it or spend the dedicated one-re-roll budget.  The cap
    counter is parallel to the shared pathology budget and never
    increments it.

    Returns ``None`` while the stream is under the cap; the cap is
    enforced by the caller every guard cadence (this function is pure:
    text in, verdict out -- no stream state).
    """
    factor = float(runtime.get("output_cap_factor", 1.0))
    if factor <= 0:
        return None
    floor_chars = int(runtime.get("output_cap_min_chars", 4000))
    _artifact_chars = float(runtime.get("_output_cap_artifact_chars", 0))
    # ================================================================
    # 2026-09-14 (USER RULING): NO CAP ON DENOVO GENERATION.
    # ================================================================
    # "there should be no cap other than the system cap, currently set
    #  at 24k tokens, for any denovo generation."
    #
    # WHY THE CAP MUST NOT APPLY HERE.  The cap's entire rationale is
    # RELATIVE: a change order is a description of edits, so one larger
    # than the artifact it edits is wrong by construction.  A DENOVO
    # artifact has no such relation to the artifact being modified -- it
    # is a fresh complete file, so it is legitimately artifact-sized or
    # larger.  Applying the cap there guarantees a trip on every
    # SUCCESSFUL generation.
    #
    # This was a live regression: PATH A (local regeneration) produced a
    # complete artifact, the cap closed all three streams at ~7,183/
    # 7,229/7,224 chars against an artifact of 7,155 chars, the
    # round2-shaped rescue could not parse HTML, and output_cap_exceeded
    # was unclassified -- so both local tiers died on the first trip with
    # no retry and the cloud silently covered it.
    #
    # THE DISCRIMINATOR.  `_output_cap_artifact_chars` is bound to a
    # NON-ZERO value ONLY on the change-order path (it is the size of the
    # artifact a change order will be applied to).  Denovo paths
    # (PATH A regeneration, new_code_generation, bug_fix, evaluate,
    # general_use) never bind it.  So a zero binding means "there is no
    # artifact this output is described RELATIVE TO", and the cap has
    # nothing to be relative to -> it does not apply.
    #
    # This is deliberately expressed as a PROPERTY of the request rather
    # than a list of harness names: any future denovo path is exempt
    # automatically, and no path can be exempted by accident while still
    # carrying an artifact binding.
    #
    # The ONLY remaining bound is the system cap -- the model's
    # num_predict (24576 tokens, config-sampled) plus the config-owned
    # output_ceiling failsafe.  No new cap is introduced here: the
    # guard simply declines to add a second, tighter one.
    if _artifact_chars <= 0:
        return None
    cap = max(floor_chars, int(factor * _artifact_chars))
    if cap <= 0:
        # No artifact bound available (e.g. guard invoked outside the
        # modify path): fall back to the absolute floor so a runaway
        # still gets stopped rather than silently unguarded.
        cap = floor_chars
    return "output_cap_exceeded" if len(text) > cap else None


def modify_output_cap_rescue(
    text: str,
    runtime: dict[str, Any],
    artifact_store: Any,
    artifact_id: str | None,
) -> tuple[str | None, Any]:
    """Rescue step for the modify output cap (2026-09-10, user ruling).

    Delegates to the self-sufficient round2 contract (single validation
    owner) and returns ``(error, pending_edit)``: ``error`` None +
    ``pending_edit`` = the contract-stashed package on success; a
    functional-error string + ``None`` when the partial stream holds no
    complete, valid change order.
    """
    # 2026-09-10 fix (self-review, step 3 verification): the contract
    # stashes into the dict it was HANDED, so pop from a local context
    # dict, not the caller's ``runtime``.
    _cap_ctx: dict[str, Any] = {
        "_artifact_store": artifact_store,
        "_artifact_id": artifact_id,
    }
    err = modify_artifact_round2_contract(text, _cap_ctx)
    pending = _cap_ctx.pop("_pending_artifact_edit", None)
    if err is None and pending is not None:
        return None, pending
    if err is None:
        # Contract passed but stashed nothing (no artifact context): treat
        # as rescue failure so the dedicated re-roll path engages.
        return "output_cap_rescue_failed: contract passed without a package", None
    return f"output_cap_rescue_failed: {err}", None


def _flip_inverted_patch_hunks(patch_text: str) -> tuple[str, int]:
    """Detect and flip inverted-polarity hunks in a unified-diff patch.

    Returns ``(corrected_patch, num_flips)``.  When ``num_flips == 0``
    the patch is returned unchanged (the first tuple element is a
    copy-equivalent string).

    The check is local: for each hunk we look at the ``+`` lines and
    the `` `` (space) context lines and see whether they overlap.
    If every ``+`` line duplicates a context line and the ``-``
    lines are entirely disjoint from the context, the model has
    emitted the AFTER state as ``+`` and the BEFORE state as ``-``;
    we flip the prefixes inside that hunk to recover the intended
    polarity.

    The heuristic is intentionally conservative: a hunk is only
    flipped when the inversion is unambiguous (no ``+`` line is
    absent from the context set, and at least one ``+`` line is
    present).  Mixed-polarity hunks are left alone — the existing
    ``patch_does_not_apply`` / SWE-bench apply step is the right
    place to surface those.
    """
    lines = patch_text.splitlines()
    out: list[str] = []
    hunk_header_re = re.compile(r"^@@\s")
    flips = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if hunk_header_re.match(line):
            out.append(line)
            i += 1
            # Collect this hunk's body lines until the next @@ or
            # the end of the patch.
            hunk_body: list[str] = []
            while i < len(lines) and not hunk_header_re.match(lines[i]):
                # The diff's file headers (``--- a/``, ``+++ b/``)
                # and ``diff --git`` lines belong to the file as a
                # whole, not to any single hunk.  Pass them through
                # unchanged and skip them here — they only appear
                # once per file, before the first hunk, in normal
                # output.
                if lines[i].startswith("--- ") or lines[i].startswith("+++ ") or lines[i].startswith("diff --git"):
                    break
                hunk_body.append(lines[i])
                i += 1
            # Decide whether to flip this hunk.  For ``+`` and
            # ``-`` lines we strip the marker char; for context
            # lines (`` ``) we leave the leading space in place
            # so the indent is preserved (a context line's first
            # character IS real content, not a marker).
            # ``+++``/``---`` headers belong to the file as a
            # whole, not to any single hunk, so we exclude them
            # via the ``not l.startswith(...)`` guards.
            plus_lines = [
                l[1:] for l in hunk_body
                if l.startswith("+") and not l.startswith("+++")
            ]
            minus_lines = [
                l[1:] for l in hunk_body
                if l.startswith("-") and not l.startswith("---")
            ]
            context_lines = {
                l for l in hunk_body
                if l.startswith(" ") and len(l) > 1
            }
            if (
                plus_lines
                and minus_lines
                and not context_lines.isdisjoint(plus_lines)
                and all(p in context_lines for p in plus_lines)
                and all(m not in context_lines for m in minus_lines)
            ):
                # Flip ``+`` <-> ``-`` inside this hunk.  Context
                # lines and ``+++``/``---`` headers stay put.
                for hl in hunk_body:
                    if hl.startswith("+") and not hl.startswith("+++"):
                        out.append("-" + hl[1:])
                    elif hl.startswith("-") and not hl.startswith("---"):
                        out.append("+" + hl[1:])
                    else:
                        out.append(hl)
                flips += 1
            else:
                out.extend(hunk_body)
            continue
        out.append(line)
        i += 1
    return ("\n".join(out) + ("\n" if patch_text.endswith("\n") else ""), flips)


def _removed_lines(block: str) -> set[str]:
    """Return the set of removed (``-``) line contents in a unified-diff
    block, excluding the ``--- a/`` file header.

    Used by the context-correctness gate: a patch can only remove code
    that exists in the submitted file, so the ``-`` lines are the
    strongest signal for whether a block is patching the real code the
    model was shown (vs. hallucinating a different file / algorithm).
    """
    out: set[str] = set()
    for line in block.splitlines():
        if line.startswith("-") and not line.startswith("---"):
            out.add(line[1:])
    return out


def _target_path_from_base(patch_base: str) -> str | None:
    """Extract the target file path from the ``[target file: ...]``
    header in the submitted ``<code>`` block (``patch_base``).

    The smoke client prefixes the code block with
    ``[target file: <path> -- ...]``.  This returns that path so the
    gate can verify the patch targets the correct file.
    """
    if not patch_base:
        return None
    m = re.search(r"\[target file:\s*([^\s\]]+)", patch_base)
    if m:
        return m.group(1).lstrip("/")
    return None


def _patch_target_path(block: str) -> str | None:
    """Extract the file path from a patch block's ``diff --git`` or
    ``--- a/`` header."""
    m = re.search(r"^diff --git a/(\S+)", block, re.M)
    if m:
        return m.group(1)
    m2 = re.search(r"^--- a/(\S+)", block, re.M)
    if m2:
        return m2.group(1)
    return None


def _is_context_correct(block: str, patch_base: str) -> bool:
    """Return True iff every removed (``-``) line in ``block`` appears
    in the submitted ``<code>`` block (``patch_base``).

    A block with no removals (pure addition) is context-correct by
    definition — additions have no "before" state to validate against.
    Otherwise, every removed line must be present in the submitted
    source.  Trailing whitespace is stripped on both sides so a
    whitespace-only difference doesn't false-negative a correct patch.

    v13 step 2 (2026-08-08): also verify the patch targets the SAME
    file as the ``[target file: ...]`` header.  The model sometimes
    hallucinates a different file (e.g. patching ``django/core/base.py``
    when the target is ``django/db/models/deletion.py``) — 44% of
    failures.  A path mismatch is a hard rejection.
    """
    # File-path check: the patch must target the file named in the
    # <code> block header.
    target = _target_path_from_base(patch_base)
    if target:
        patch_path = _patch_target_path(block)
        if patch_path and patch_path.lstrip("/") != target:
            return False
    removed = _removed_lines(block)
    if not removed:
        return True
    base_lines = {ln.rstrip() for ln in patch_base.splitlines()}
    return all(ln.rstrip() in base_lines for ln in removed)


def _fuzzy_aligns(block: str, patch_base: str) -> bool:
    """Return True iff the patch's hunks align to the base file by
    CONTENT, ignoring the line-number offsets in the hunk headers.

    The model frequently emits correct-content patches with WRONG line
    numbers in the ``@@ -N,M +N,M @@`` headers (the alignment failure
    mode, ~33% of failures).  ``git apply`` rejects these because the
    hunk's ``src_start`` doesn't match the real file's line position.

    This check searches the base file for each hunk's source lines
    (context + removed lines) as a contiguous sequence, regardless of
    the declared line numbers.  If every hunk's source lines appear
    contiguously in the base, the patch is "fuzzy-aligned" — the
    content is correct even if the line numbers are off.

    This is a pure content-presence check (no knowledge of the gold
    patch), so it is not "cheating".  It mirrors what ``git apply``
    does with fuzz, but at the content level.
    """
    if not block or not patch_base:
        return False
    base_lines = [ln.rstrip() for ln in patch_base.splitlines()]
    # Parse hunks: split on @@ headers.
    hunk_headers = list(re.finditer(r"^@@[^\n]*\n", block, re.M))
    if not hunk_headers:
        return False
    for i, hm in enumerate(hunk_headers):
        start = hm.end()
        end = hunk_headers[i + 1].start() if i + 1 < len(hunk_headers) else len(block)
        hunk_body = block[start:end]
        # Extract source lines: context (' ') and removed ('-'), excluding
        # the ---/+++ file headers.
        src_lines = []
        for ln in hunk_body.splitlines():
            if ln.startswith("--- ") or ln.startswith("+++ ") or ln.startswith("diff --git"):
                continue
            if ln.startswith(" ") and len(ln) > 1:
                src_lines.append(ln[1:].rstrip())
            elif ln.startswith("-") and not ln.startswith("---"):
                src_lines.append(ln[1:].rstrip())
        if not src_lines:
            continue
        # Search for the contiguous sequence in base_lines.
        if not _find_contiguous(base_lines, src_lines):
            return False
    return True


def _find_contiguous(base_lines: list[str], needle: list[str]) -> bool:
    """Return True iff ``needle`` appears as a contiguous subsequence of
    ``base_lines`` (allowing the needle to be a subset of a longer run).

    Comparison is indentation-tolerant via ``_align_cmp`` (leading
    whitespace ignored), matching ``_find_contiguous_start``.
    """
    if not needle:
        return True
    n = len(needle)
    for i in range(len(base_lines) - n + 1):
        if all(_align_cmp(base_lines[i + j], needle[j]) for j in range(n)):
            return True
    return False


def _fuzzy_realign(block: str, patch_base: str) -> str | None:
    """Rewrite a patch's hunk headers to the CORRECT real-file line
    numbers, grounded to the SLICED context the model was shown.

    The model is given a slice of the target file (gold-patch hunks +
    50-line buffer), formatted as ``[lines X-Y of N]`` segments.  It
    frequently emits correct-content patches with WRONG line numbers in
    the ``@@ -N,M +N,M @@`` headers — either slice-relative or simply
    off.  ``git apply`` rejects these because the hunk's ``src_start``
    doesn't match the real file's line position.

    This:
      1. Parses the slice into ``(real_line, content)`` pairs using the
         ``[lines X-Y]`` segment offsets.
      2. Searches the slice CONTENT for each hunk's source lines
         (context + removed) as a contiguous sequence — the "before
         chunk subject to the patch changes must align perfectly with
         the original context".
      3. Once aligned to the slice, maps the slice position back to the
         real-file line number via the segment offset, and rewrites the
         hunk header with that real line.

    Returns the realigned patch, or ``None`` if any hunk's source lines
    cannot be found contiguously in the slice (in which case the patch
    is genuinely misaligned / hallucinated and should be rejected).

    This is a pure content-presence check (no knowledge of the gold
    patch), so it is not "cheating".  It grounds the patch to the exact
    code slice the model was supposed to be patching.
    """
    if not block or not patch_base:
        return None
    real_lines, base_contents = _build_slice_line_map(patch_base)
    if not real_lines or not base_contents:
        return None
    # Parse hunks: split on @@ headers.
    hunk_headers = list(re.finditer(r"^@@[^\n]*\n", block, re.M))
    if not hunk_headers:
        return None
    out = []
    cursor = 0
    for i, hm in enumerate(hunk_headers):
        # Copy everything before this hunk header.
        out.append(block[cursor:hm.start()])
        header_line = hm.group(0)
        start = hm.end()
        end = hunk_headers[i + 1].start() if i + 1 < len(hunk_headers) else len(block)
        hunk_body = block[start:end]
        # Extract source lines: context (' ') and removed ('-'), excluding
        # the ---/+++ file headers.
        src_lines = []
        for ln in hunk_body.splitlines():
            if ln.startswith("--- ") or ln.startswith("+++ ") or ln.startswith("diff --git"):
                continue
            if ln.startswith(" ") and len(ln) > 1:
                src_lines.append(ln[1:].rstrip())
            elif ln.startswith("-") and not ln.startswith("---"):
                src_lines.append(ln[1:].rstrip())
        if not src_lines:
            # No source lines to align; keep the header as-is.
            out.append(header_line)
            out.append(hunk_body)
            cursor = end
            continue
        # Find the contiguous sequence in the slice CONTENT.
        found_start = _find_contiguous_start(base_contents, src_lines)
        if found_start is None:
            return None  # cannot align this hunk -> reject
        # Map the slice-content position back to a real-file line.
        real_start = real_lines[found_start]
        # Parse the old header to preserve the counts.
        m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", header_line)
        if m:
            old_src_start = int(m.group(1))
            src_count = int(m.group(2)) if m.group(2) else 1
            old_dst_start = int(m.group(3))
            dst_count = int(m.group(4)) if m.group(4) else 1
            # New dst start: same offset as src (the patch changes are
            # relative, so dst = real_start + (new_lines - old_lines)).
            # We approximate dst as the same as src_start for a small
            # edit; git apply only needs a valid start.
            new_dst_start = real_start
            new_header = f"@@ -{real_start},{src_count} +{new_dst_start},{dst_count} @@\n"
            out.append(new_header)
        else:
            out.append(header_line)
        out.append(hunk_body)
        cursor = end
    out.append(block[cursor:])
    return "".join(out)


def _build_slice_line_map(patch_base: str) -> tuple[list[int], list[str]]:
    """Parse a sliced ``<code>`` block into ``(real_line_numbers,
    contents)`` parallel lists.

    The slice format (from ``slice_lines``) is repeated segments of::

        [lines X-Y of N]
        <content lines X..Y>

    where each ``[lines X-Y]`` header declares the real-file line range
    of the content that follows.  This walks the block, tracks the
    current segment's real-file line number, and emits one ``(real_line,
    content)`` entry per content line.

    Returns ``([real_lines], [contents])``.  Empty if the block has no
    ``[lines ...]`` headers (e.g. it's a raw file, not a slice).
    """
    real_lines: list[int] = []
    contents: list[str] = []
    # Match segment headers: [lines START-END of N]
    seg_re = re.compile(r"\[lines\s+(\d+)-(\d+)\s+of\s+\d+\]")
    current_line: int | None = None
    for ln in patch_base.splitlines():
        m = seg_re.search(ln)
        if m:
            current_line = int(m.group(1))
            continue
        if current_line is not None:
            real_lines.append(current_line)
            contents.append(ln.rstrip())
            current_line += 1
    return real_lines, contents


def _align_cmp(a: str, b: str) -> bool:
    """Whitespace-tolerant line comparison for fuzzy alignment.

    A correct unified-diff before-chunk (context + removed lines) must
    match the code slice by CONTENT, but a model legitimately re-indents
    a context line when a fix changes indentation (e.g. moving code into
    a new block).  The exact ``==`` match would then fail on leading
    whitespace and reject a correct patch as non-aligning.

    We compare after stripping leading (and trailing) whitespace, so an
    indentation shift in a context line does not break alignment — the
    grounding is by content, not byte-identical indentation.
    """
    return a.lstrip() == b.lstrip()


def _find_contiguous_start(base_lines: list[str], needle: list[str]) -> int | None:
    """Return the 0-indexed start position of ``needle`` as a contiguous
    subsequence of ``base_lines``, or ``None`` if not found.

    Comparison is indentation-tolerant via ``_align_cmp`` (leading
    whitespace ignored), so a correct patch whose context lines were
    re-indented still aligns to the slice.
    """
    if not needle:
        return 0
    n = len(needle)
    for i in range(len(base_lines) - n + 1):
        if all(_align_cmp(base_lines[i + j], needle[j]) for j in range(n)):
            return i
    return None


def _select_context_correct_block(blocks: list[str], patch_base: str) -> str | None:
    """Return the block to grade, or ``None`` if none are context-correct.

    Selection rule (v14, 2026-08-10): take the FIRST context-correct
    block.  This matches the direct 62% run's ``extract_diff(raw,
    take_last=False)`` which grades the FIRST <patch> block.  The
    bug_fix harness now caps at ``max_blocks: 1`` so there is normally
    only one block; taking the first is the congruent behavior.  If the
    model emits more than one despite the cap, the first context-correct
    one is the answer (the model's initial, cleanest attempt), not the
    last draft.

    A block is context-correct iff every removed (``-``) line appears in
    the submitted ``<code>`` block (``patch_base``) and it targets the
    same file.  If no block is context-correct, return ``None`` (the
    contract rejects).
    """
    if not blocks:
        return None
    for block in blocks:
        if _is_context_correct(block, patch_base):
            return block
    return None


def unified_diff_patch_contract(text: str, runtime: dict[str, Any]) -> str | None:
    """Validate a patch block and select the block to grade.

    v14 (2026-08-10): REWRITTEN to be fully congruent with the direct 62%
    run (``ollama_swebench.py``).  The direct run's behavior was:
      1. ``extract_diff(raw, take_last=False)`` — take the FIRST <patch>
         block.
      2. Salvage a ``diff --git`` header if missing.
      3. Add a trailing newline.
      4. ``repair_patch()``.
      5. Grade with ``eval_swebench_docker.py``.

    It had NONE of the server-side "improvements" that were later layered
    on: no context-correctness gate, no polarity flip, no patch_empty_diff
    rejection, no fuzzy realignment.  Those all change the graded patch
    (or force re-rolls/escalation) in ways the direct run never did, and
    are therefore NOT apples-to-apples.

    This contract now ONLY:
      - Rejects if there is no <patch> block (empty/malformed), matching
        the direct run's empty_patch / malformed_patch handling.
      - Selects the FIRST <patch> block as the answer (stashed via
        ``_corrected_patch`` so the orchestrator returns it).
      - Returns None (pass) otherwise — the patch is graded as-is, exactly
        like the direct run.  No gate, no flip, no realignment, no
        empty-diff rejection.
    """
    if not text.strip():
        return "empty_patch"
    blocks = re.findall(r"<patch>(.*?)</patch>", text, re.DOTALL)
    if not blocks:
        # No <patch> block: the model produced prose only.  This maps to
        # the direct run's empty_patch / malformed_patch handling (which
        # triggered a reset retry via exhaustion).
        return "token_exhaustion" if len(text) >= int(runtime.get("num_predict", 24576)) * 3 else "malformed_patch"
    # Take the FIRST block (congruent with the direct run's take_last=False).
    # Strip any ```diff wrapper.
    first = re.sub(r"^```diff\s*\n", "", blocks[0])
    first = re.sub(r"\n```\s*$", "", first)
    first = first.strip()
    if not first:
        return "empty_patch"
    # The patch must be a well-formed unified diff (core hunks present).
    required = ("--- a/", "+++ b/", "@@ ")
    missing = [part for part in required if part not in first]
    if missing:
        return "malformed_patch: missing " + ", ".join(missing)
    # Stash the selected (first) block so the orchestrator returns it as
    # ``result.output`` — the patch the evaluator grades.  We do NOT apply
    # any context gate, polarity flip, or fuzzy realignment.
    runtime["_corrected_patch"] = first
    runtime["_corrected_patch_flips"] = 0
    return None


def python_new_code_contract(text: str, runtime: dict[str, Any]) -> str | None:
    # Two deterministic, no-knowledge-of-the-correct-answer checks
    # for the ``python_new_code`` harness:
    #
    #   1. **Fence check**: the response must contain a
    #      ``python`` (or ``py``) code fence.  The system prompt
    #      instructs the model to emit ``python`` code blocks;
    #      a response with no fence at all is a format
    #      violation (the model went into pure prose and never
    #      produced code).  This catches failures like
    #      HumanEval/81 where the model produced 18K chars of
    #      ``#`` comments and no code fence.
    #
    #   2. **Syntax check**: the contents of the LAST ``python``
    #      fence must compile as valid Python.  This catches
    #      failures like HumanEval/159 where the model wrote
    #      ``def draft():`` with only ``#`` comments in the
    #      DRAFT block — ``#`` comments are not statements and
    #      ``def draft():`` therefore has no body, which is a
    #      ``SyntaxError`` when compiled.  It also catches
    #      top-level code that doesn't follow the
    #      ``def entry_point(`` pattern (the contract is
    #      intentionally agnostic to whether the user wants a
    #      function, a script, or a class).
    #
    # The contract deliberately does NOT require a specific
    # ``def entry_point(`` to be present — many legitimate
    # Python outputs are top-level scripts, ``if __name__ ==
    # "__main__":`` blocks, or class definitions with no
    # matching ``def`` in the prompt.  We only verify that
    # whatever code the model produced is structurally valid
    # Python, not that it solves the problem correctly.
    #
    # Both errors are added to ``retry_on`` for the harness so
    # T0 retries with a new seed on a real reroll.  After T0
    # exhausts its retries, T1 receives the error string as
    # ``prior_error`` in the user message and can address it.
    if not text.strip():
        return "empty_response"
    # 1. Fence check: must contain at least one code fence
    # in the declared language.  The language is resolved
    # from the dispatcher's detected fence language, the runtime
    # config (``fence_counter.fence_languages`` override), or the
    # ``FENCE_ALIASES`` map.  Default: python.
    _detected = runtime.get("_detected_fence_language")
    aliases = _fence_aliases_for(_detected, runtime)
    fence_tag_re = _build_fence_tag_re(aliases)
    if not fence_tag_re.search(text):
        return "no_code_fence: response contains no code fence"
    # 2. Syntax check: compile the contents of the LAST
    # code fence in the declared language.  This is what
    # the test runner will effectively execute (after
    # concatenating with the prompt for HumanEval-style
    # tasks, or running directly for script-style tasks).
    # We use ``exec`` mode so top-level statements (imports,
    # function calls, class definitions) are all valid.
    fence_open = _build_fence_open_re(aliases)
    fence_close = _FENCE_CLOSE_RE
    last_fence_body: str | None = None
    last_open_end = -1
    for om in fence_open.finditer(text):
        body_start = om.end()
        cm = fence_close.search(text, body_start)
        if cm is None:
            # Fence never closed — skip (the streaming
            # guards or token exhaustion have already
            # handled this case via the
            # ``token_exhaustion`` / ``malformed_patch``
            # paths, or the test runner will see the
            # SyntaxError on the unclosed fence).
            continue
        if om.end() > last_open_end:
            last_open_end = om.end()
            last_fence_body = text[body_start:cm.start()]
    if last_fence_body is None:
        return "no_code_fence: response contains an unclosed code fence"
    try:
        compile(last_fence_body, "<last_code_fence>", "exec")
    except SyntaxError as exc:
        return f"syntax_error: {exc.msg} (line {exc.lineno}, col {exc.offset})"
    # Step 5: stash the contract-validated extraction so the tier
    # loop can populate CleanExecutionResult.extracted_artifact_content
    # without a separate regex re-derivation over the raw response.
    runtime["_extracted_code"] = last_fence_body
    # Step 5 (Option B): stash the fence language so the artifact
    # persistence layer can pick the correct file extension.
    runtime["_extracted_language"] = "python"
    return None


def web_new_code_contract(text: str, runtime: dict[str, Any]) -> str | None:
    """Contract for the ``web_new_code`` harness.

    Two deterministic, no-knowledge-of-the-correct-answer checks:

      1. **Fence check**: the response must contain at least one
         code fence in the declared language (default: ``html``).
      2. **Parse check**: the contents of the LAST code fence must
         be parseable.  For HTML, this uses Python's built-in
         ``html.parser.HTMLParser`` which catches malformed tag
         structures.  For CSS/JS/TS (if the fence language is not
         HTML), the parse check is skipped (no stdlib parser; the
         tree-sitter integration in a future step will handle
         these).

    Both checks are structural — they only verify the model
    produced well-formed code, not that it solves the problem
    correctly.
    """
    from html.parser import HTMLParser

    if not text.strip():
        return "empty_response"
    # 1. Fence check.  Use the fence language detected by the
    # dispatcher (new_code_generation_contract) if available;
    # otherwise fall back to the runtime config or default.
    _detected = runtime.get("_detected_fence_language")
    aliases = _fence_aliases_for(_detected, runtime)
    fence_tag_re = _build_fence_tag_re(aliases)
    if not fence_tag_re.search(text):
        return "no_code_fence: response contains no code fence"
    # 2. Extract the last fence body.
    body = extract_last_fence_body(text, _detected, runtime)
    if body is None:
        return "no_code_fence: response contains an unclosed code fence"
    # 3. Parse check: only for HTML fences.  For CSS/JS/TS, skip
    # (no stdlib parser; tree-sitter comes later).
    # Use the dispatcher's detected language as the primary alias
    # when available — the fence_languages override in config.yaml
    # lists ALL languages, so aliases[0] is always "python" and
    # cannot be used to determine the actual fence language.
    if _detected:
        primary_alias = _detected.lower()
    elif aliases:
        primary_alias = aliases[0].lower()
    else:
        primary_alias = "html"
    if primary_alias in ("html", "htm"):
        class _TagBalanceChecker(HTMLParser):
            """Check that HTML tags are balanced.

            ``HTMLParser`` itself is lenient (it doesn't raise on
            most malformed HTML).  We add a lightweight check:
            track open/close tags and verify the stack is empty
            at the end.  Void elements (``<br>``, ``<img>``,
            ``<input>``, etc.) are excluded from the stack.
            """
            VOID_ELEMENTS = frozenset({
                "area", "base", "br", "col", "embed", "hr",
                "img", "input", "link", "meta", "param", "source",
                "track", "wbr",
            })

            def __init__(self) -> None:
                super().__init__(convert_charrefs=True)
                self.stack: list[str] = []
                self.error: str | None = None

            def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
                if tag.lower() not in self.VOID_ELEMENTS:
                    self.stack.append(tag.lower())

            def handle_endtag(self, tag: str) -> None:
                tag_lower = tag.lower()
                if tag_lower in self.VOID_ELEMENTS:
                    return
                if not self.stack:
                    self.error = f"unmatched closing tag </{tag}>"
                    return
                if self.stack[-1] != tag_lower:
                    # Allow mismatched close if the tag is
                    # somewhere in the stack (common in real-
                    # world HTML, e.g. <p>text<b>bold</p>).
                    if tag_lower in self.stack:
                        while self.stack and self.stack[-1] != tag_lower:
                            self.stack.pop()
                        if self.stack:
                            self.stack.pop()
                    else:
                        self.error = f"unexpected closing tag </{tag}>"
                    return
                self.stack.pop()

        checker = _TagBalanceChecker()
        try:
            checker.feed(body)
            checker.close()
        except Exception as exc:
            return f"syntax_error: HTML parse error: {exc}"
        if checker.error:
            return f"syntax_error: {checker.error}"
        if checker.stack:
            unclosed = ", ".join(f"<{t}>" for t in checker.stack)
            return f"syntax_error: unclosed tags: {unclosed}"
    # ===================================================================
    # 4. Non-HTML web languages (CSS / JS / TS) and HTML INLINE blocks.
    # ===================================================================
    # This replaces the former "skip the parse check for now" branch.  The
    # skip was the whole gap: `HTMLParser` sees a <script> body as raw text,
    # so broken inline JavaScript passed every web generation unchecked.
    #
    # `check_web_syntax` handles BOTH cases with one call:
    #   * html  -> parse each inline <script>/<style> body with its own grammar
    #              (tag balance is already done above)
    #   * css/javascript/typescript -> parse the whole body
    #
    # THE CHECK EXISTS TO FEED THE ERROR-TRACE RETRY, not to gate output for
    # its own sake.  `syntax_error` is classified in the t0/t1b/t2 tiers of
    # this harness's ladder, and T1b's user_modifier carries {prior_error} back
    # to the model -- so a verdict here buys a CORRECTIVE retry.  An
    # unclassified verdict would instead consume the budget with no retry,
    # which is why the classification is not optional (PLAN_web_syntax_check
    # section 4).  Verified present on `modify_harness` t0/t1b/t2.
    #
    # FAILS SOFT: no tree-sitter, unknown grammar, or any exception returns
    # None.  This can never turn a working generation into a failure.
    #
    # Disable with `runtime.web_syntax_check: false` (the documented reversal).
    # Defaults ON: absent means enabled, so no config change is required.
    if bool(runtime.get("web_syntax_check", True)):
        _web_verdict = _check_web_syntax(body, primary_alias)
        if _web_verdict:
            return _web_verdict
    # Step 5: stash the contract-validated extraction (shared exit
    # for both the HTML pass-through and the non-HTML skip path).
    runtime["_extracted_code"] = body
    # Step 5 (Option B): stash the fence language so the artifact
    # persistence layer can pick the correct file extension.
    runtime["_extracted_language"] = primary_alias
    return None


# Tree-sitter for compiled-language syntax checking.  The grammar
# acquisition (including the JSON / JSONL shim) lives in
# :mod:`ts_backends` — single source of truth for both
# ``chunker.py`` and this module.  When the library is unavailable,
# the compiled_coding contract falls back to fence-only checking
# (no syntax verification).  ``pip install tree_sitter
# tree_sitter_language_pack tree_sitter-json``.
from .ts_backends import (
    get_parser as _ts_get_parser,
    LANGUAGE_MAP as _TS_LANGUAGE_MAP,
)
# Web syntax check (PLAN_web_syntax_check.md).  Imported here, beside the other
# tree-sitter plumbing, so there is ONE place that knows where grammars live.
# `web_new_code_contract` is defined ABOVE this import, but that is fine: the
# name is resolved at CALL time, not at definition time.
try:
    from .ts_backends import check_web_syntax as _check_web_syntax
except Exception:                                          # noqa: BLE001
    # Fail soft, matching the contract's own policy: if the check cannot be
    # imported, web generation behaves exactly as it did before this feature.
    def _check_web_syntax(content: str, language: str) -> str | None:  # noqa: D401
        return None
_HAS_TREE_SITTER = True


def _ts_find_errors(node) -> list:
    """Recursively find all ERROR and missing nodes in a tree-sitter tree."""
    errors = []
    if node.type == "ERROR" or getattr(node, "is_missing", False):
        errors.append(node)
    for child in node.children:
        errors.extend(_ts_find_errors(child))
    return errors


def compiled_coding_new_code_contract(text: str, runtime: dict[str, Any]) -> str | None:
    """Contract for the ``compiled_coding_new_code`` harness.

    Two deterministic checks:

      1. **Fence check**: the response must contain at least one
         code fence in the declared language (resolved from the
         runtime config's ``fence_languages`` or the
         ``FENCE_ALIASES`` map).
      2. **Syntax check**: the contents of the LAST code fence are
         parsed with tree-sitter.  The parser is selected based on
         the **actual fence tag** the model used (e.g. `````rust``,
         `````go``), not just the first alias in the list.  If any
         ``ERROR`` nodes are found, the contract fails with a
         ``syntax_error`` verdict.  If tree-sitter is not installed
         or the language has no parser, the syntax check is skipped.
    """
    if not text.strip():
        return "empty_response"
    # 1. Fence check.  Use the fence language detected by the
    # dispatcher (new_code_generation_contract) if available.
    _detected = runtime.get("_detected_fence_language")
    aliases = _fence_aliases_for(_detected, runtime)
    fence_tag_re = _build_fence_tag_re(aliases)
    if not fence_tag_re.search(text):
        return "no_code_fence: response contains no code fence"
    # 2. Extract the last fence body.
    body = extract_last_fence_body(text, _detected, runtime)
    if body is None:
        return "no_code_fence: response contains an unclosed code fence"
    # 3. Detect the actual fence tag the model used (e.g. "rust",
    # "go", "cpp").  Search for the last opening fence in the text
    # and extract the language tag.
    actual_lang = ""
    fence_open_re = _build_fence_open_re(aliases)
    last_match = None
    for m in fence_open_re.finditer(text):
        last_match = m
    if last_match:
        # The fence tag is the text between ``` and the first
        # whitespace/newline.  Extract it from the match.
        tag_text = last_match.group(0)  # e.g. "```rust\n"
        tag = tag_text[3:].strip().split("\n")[0].strip()
        actual_lang = tag.lower()
    # 4. Tree-sitter syntax check.
    if not _HAS_TREE_SITTER:
        runtime["_extracted_code"] = body
        runtime["_extracted_language"] = actual_lang
        return None
    ts_lang = _TS_LANGUAGE_MAP.get(actual_lang, "")
    if not ts_lang:
        # Unknown language — skip syntax check.
        runtime["_extracted_code"] = body
        runtime["_extracted_language"] = actual_lang
        return None
    try:
        parser = _ts_get_parser(ts_lang)
    except Exception:
        runtime["_extracted_code"] = body
        runtime["_extracted_language"] = actual_lang
        return None
    try:
        tree = parser.parse(body.encode("utf-8"))
    except Exception as exc:
        return f"syntax_error: parse error: {exc}"
    errors = _ts_find_errors(tree.root_node)
    if errors:
        first = errors[0]
        body_bytes = body.encode("utf-8")
        line = body_bytes[:first.start_byte].count(b"\n") + 1
        col = first.start_byte - body_bytes.rfind(b"\n", 0, first.start_byte)
        return f"syntax_error: tree-sitter ERROR node at line {line}, col {col}"
    # Step 5: stash the contract-validated extraction.
    runtime["_extracted_code"] = body
    # Step 5 (Option B): stash the fence language so the artifact
    # persistence layer can pick the correct file extension.
    runtime["_extracted_language"] = actual_lang
    return None


# Canonical-name groups used by ``new_code_generation_contract`` to
# dispatch to the correct language-specific validator.  These match
# the groupings ``FENCE_ALIASES`` already declares.
_PYTHON_LANGS = frozenset({"python"})
_WEB_LANGS = frozenset({"html", "css", "javascript", "typescript", "jsx", "tsx"})


def new_code_generation_contract(text: str, runtime: dict[str, Any]) -> str | None:
    """Contract for the canonical ``new_code_generation`` harness.

    ``new_code_generation`` is shared across ``domain_python``,
    ``domain_compiled``, and ``domain_web`` (Step 3's harness
    collapse).  Each domain needs a genuinely different syntax
    check (Python ``compile()``, HTML tag-balance, tree-sitter for
    compiled languages) -- these three checks already exist as
    ``python_new_code_contract`` / ``web_new_code_contract`` /
    ``compiled_coding_new_code_contract`` but were never reachable
    under the canonical harness name (the ``CONTRACTS`` registry
    only had the old per-domain names).

    This dispatcher detects the actual language from the model's
    own fence tag (``_detect_fence_language``, the same
    language-agnostic detection T1a's patch-apply path already
    uses) and routes to the matching validator.  Detecting from
    the model's OWN output -- not the domain name -- means the
    check is correct even if a model unexpectedly writes a
    different fenced language than its assigned domain.
    """
    lang = _detect_fence_language(text)
    # Stash the detected language so the sub-contract can use it
    # for fence alias resolution and _extracted_language.  Without
    # this, the sub-contract calls _fence_aliases_for(None, runtime)
    # which defaults to Python, causing HTML artifacts to get a
    # .py extension.
    runtime["_detected_fence_language"] = lang
    if lang in _PYTHON_LANGS:
        return python_new_code_contract(text, runtime)
    if lang in _WEB_LANGS:
        return web_new_code_contract(text, runtime)
    # Everything else (cpp, c, rust, go, java, kotlin, scala, ...)
    # goes through the tree-sitter dispatcher, which already
    # gracefully skips the syntax check for languages it has no
    # parser for instead of raising a false positive.
    return compiled_coding_new_code_contract(text, runtime)


# ---------------------------------------------------------------------------
# Denovo (PATH A) contract selection -- by ARTIFACT LANGUAGE, and nothing else.
# ---------------------------------------------------------------------------
# 2026-09-17 (user ruling).  PATH A used to take its contract from config alone
# (`modify_regeneration_contract: new_code_generation`), so a PROSE artifact was
# validated by a CODE contract and could never pass.  Measured on the
# 2026-09-17 physics trace: a CORRECT two-paragraph expansion was rejected
# `no_code_fence`; a fenced one then died in `compile()` on a curly apostrophe;
# four local calls and two COLD MODEL LOADS produced nothing, and the request
# escalated to T3.  Those "failures" measured contract mismatch, not model
# capability.
#
# The root cause was a LAYER CONFUSION.  The router's coding-domain gate
# (transport.py `_post_process`) correctly keeps `new_code_generation` away from
# non-coding domains -- but it constrains TASK_TYPE, while PATH A selects its
# CONTRACT from the artifact's LANGUAGE.  Two different axes, so the gate never
# observed the second: `modify_artifact` is legitimately allowed on
# `domain_physics`, the gate had nothing to clamp, and the harness swap to
# `modify_harness` then took a code contract by config default.
#
# These are CONTRACTS, not harnesses.  The code languages REUSE the
# `new_code_generation` family's validators -- what is reused is the contract,
# never the harness: `modify_artifact` keeps its own harness and workflow.  The
# need for a prose HARNESS is subsumed by `general_use`, so there is
# deliberately no prose harness here.
#
# The two regeneration WORDING fallbacks, hoisted so the two-round site and the
# live single-tier site cannot drift apart (they previously carried duplicate
# literals).  The code one keeps its ORIGINAL text verbatim; the prose one is
# new and deliberately asks for no fence.
_CODE_REGEN_SUFFIX_DEFAULT = (
    "You are a competent {{domain_tag}} engineer. Produce the COMPLETE, working "
    "artifact requested below -- the full file, not a description, not a plan, "
    "and not a diff. Satisfy the original build request plus every subsequent "
    "modification, and then the current request. Keep behaviour that is not "
    "being changed. Emit the artifact in a single fenced code block and nothing "
    "else."
)
_PROSE_REGEN_SUFFIX_DEFAULT = (
    "You are a competent writer. Produce the COMPLETE revised document "
    "requested below -- the full text, not a description, not a plan, and not a "
    "diff. Satisfy the original request plus every subsequent modification, and "
    "then the current request. Keep wording that is not being changed. Return "
    "only the document text."
)

# ---------------------------------------------------------------------------
# PATH A contract selection.  This map answers ONE question per artifact:
#   "is this artifact CODE (validate as code) or PROSE (validate as text)?"
# ---------------------------------------------------------------------------
# UNKNOWN -> prose, on purpose (user ruling): a prose contract ACCEPTS, whereas
# a code contract rejects legitimate output for the wrong reason.
# `_extension_for` already resolves an unknown tag to `.txt`/`plain`, so
# "prose" is what such an artifact already is.
#
# WHY EVERY CODE LANGUAGE POINTS AT THE DISPATCHER, NOT AT A SUB-CONTRACT.
# 2026-09-17 (REGRESSION FIX, measured).  An earlier revision of this map sent
# each code language straight to `python_new_code` / `web_new_code` /
# `compiled_coding_new_code`.  That BROKE PATH A for every non-Python language:
# those three contracts READ `runtime["_detected_fence_language"]`, and only the
# DISPATCHER (`new_code_generation_contract`) ever SETS it.  Called directly,
# `_fence_aliases_for(None, runtime)` falls back to `["python", "py"]`, so an
# ```html fence matched nothing and every request failed `no_code_fence`.
# Measured on the direct path: python PASS; html/css/javascript/typescript/
# jsx/tsx/cpp/rust/go ALL `no_code_fence` -- 9 of 10 languages broken, and only
# Python worked, which is why no live run had caught it.
#
# The dispatcher is also the semantically CORRECT entry point: it detects the
# fence language from the MODEL'S OWN OUTPUT (`_detect_fence_language`), so the
# check stays correct even when a model writes a different fenced language than
# its assigned domain.  Routing through it means one validator per language and
# no second detection path to drift.
#
# NOTE (2026-09-17 naming ruling): these are CONTRACTS, not harnesses.  Code
# languages REUSE the `new_code_generation` family's validators -- what is
# reused is the contract, never the harness: `modify_artifact` keeps its own
# harness and workflow.  The need for a PROSE harness is subsumed by
# `general_use`, so there is deliberately no prose harness here.
_CODE_CONTRACT = "new_code_generation"

_DENOVO_CONTRACT_BY_LANGUAGE: dict[str, str] = {
    # --- code: all via the dispatcher (see the block comment above) ---
    "python": _CODE_CONTRACT,
    "html": _CODE_CONTRACT, "htm": _CODE_CONTRACT, "xhtml": _CODE_CONTRACT,
    "css": _CODE_CONTRACT, "javascript": _CODE_CONTRACT, "js": _CODE_CONTRACT,
    "typescript": _CODE_CONTRACT, "ts": _CODE_CONTRACT,
    "jsx": _CODE_CONTRACT, "tsx": _CODE_CONTRACT,
    "web": _CODE_CONTRACT,           # legacy CLI bucket tag
    "cpp": _CODE_CONTRACT, "c": _CODE_CONTRACT,
    "rust": _CODE_CONTRACT, "go": _CODE_CONTRACT,
    "java": _CODE_CONTRACT, "kotlin": _CODE_CONTRACT, "scala": _CODE_CONTRACT,
    # 2026-09-17: scss added.  SCSS needs NO new contract and NO new framing
    # class -- it is a CSS superset with the same block delimiter and the same
    # comment syntaxes, and it resolves to the `.css` extension (hence the
    # existing `block:css` framing).  The only gap was FENCE_ALIASES, fixed
    # alongside this.
    "scss": _CODE_CONTRACT,
    # --- prose ---
    "markdown": "prose_contract", "md": "prose_contract",
    "prose": "prose_contract", "text": "prose_contract", "txt": "prose_contract",
}


def denovo_contract_for_language(language: str | None) -> str:
    """The denovo contract for a PATH A regeneration of this ARTIFACT.

    Keyed on the artifact's LANGUAGE.  Unknown -> ``prose_contract``: prose
    accepts, a code contract rejects legitimate output for the wrong reason
    (user ruling 2026-09-17).
    """
    return _DENOVO_CONTRACT_BY_LANGUAGE.get(
        (language or "").strip().lower(), "prose_contract"
    )


def prose_contract(text: str, runtime: dict[str, Any]) -> str | None:
    """Contract for a PROSE artifact: markdown / plain text.

    The ONLY check is that prose was produced at all.  Prose has no syntax to
    validate, and every richer check considered was rejected (user ruling
    2026-09-17): a contract's return value ESCALATES, so a false positive costs
    a retry and a cloud call -- precisely the failure this contract exists to
    remove.  `general_use` carries no contract at all, so a stricter prose
    contract here would make PATH A stricter than the harness prose is subsumed
    by.

    The extraction is NOT a check; it is what makes the artifact committable.
    PATH A's commit reads ``result.extracted_artifact_content``, which is
    populated only from ``runtime["_extracted_code"]`` -- a ``None`` contract
    would leave it empty and the commit would silently produce nothing.
    """
    if not text or not text.strip():
        return "empty_response"
    # The returned fence language is DISCARDED, deliberately: the artifact's
    # KIND is the PARENT's, never the model's fence tag (a modification cannot
    # change what a document is).  The parent's language is pinned by the
    # caller after this contract returns, so naming the second element here
    # would only imply it is consulted.
    body, _ = _extract_general_use_work_product(text)
    if not body or not body.strip():
        return "empty_response"
    runtime["_extracted_code"] = body
    return None


def evaluate_artifact_contract(text: str, runtime: dict[str, Any]) -> str | None:
    """Contract for the canonical ``evaluate_artifact`` harness.

    ``evaluate_artifact``'s own system_suffix deliberately allows a
    FLEXIBLE output shape: a direct answer to a content question, a
    "quick look" or "deep dive" freeform analysis, OR (only when the
    user did not specify a format) the ``ISSUE: ...`` / ``NO ISSUES
    FOUND`` structured format.  Because the harness explicitly
    supports non-``ISSUE:``-line responses (including for non-code
    artifacts -- prose, essays, proofs), a strict ``ISSUE:``-line
    contract would be WRONG: it would reject every direct-answer or
    freeform response, which is the harness's primary intended
    behavior, not an edge case.

    This contract intentionally checks ONLY the one
    format-independent failure mode that is always wrong regardless
    of what shape the user asked for: an empty response.  Stricter
    format enforcement (e.g. requiring ``ISSUE:`` lines when the
    user asked for none) would need visibility into the user's
    original formatting request, which is not threaded into the
    contract call today -- deferred rather than guessed at here.
    """
    if not text.strip():
        return "empty_response"
    return None


# ---------------------------------------------------------------------------
# R1 "final list" demarcation helpers (2026-09-09, user-approved design)
#
# R1's output is now loose prose that concludes with a demarcated block:
#
#     final list
#     - section_name -- modify: reason ...
#     - other_section -- new section, anchor: X, position "before": ...
#
# The block is the targeting package.  Extraction is bounded to the region
# after the LAST 'final list' marker (models conclude with it), and the
# names inside it are validated against the artifact's real section IDs.
# A structured JSON package is still accepted when the model emits one
# (validated as before) -- the loose path is a tolerated alternative, not
# a replacement for correct output.
# ---------------------------------------------------------------------------

_FINAL_LIST_MARKERS = ("final list", "final list:")


def _extract_final_list_block(text: str) -> str | None:
    """Return the text after the LAST 'final list' marker, or None.

    Case-insensitive marker match; the block runs to the end of the text
    (the model is instructed to conclude with the list).
    """
    if not text:
        return None
    lowered = text.lower()
    idx = -1
    for marker in _FINAL_LIST_MARKERS:
        pos = lowered.rfind(marker.lower())
        if pos > idx:
            idx = pos
    if idx < 0:
        return None
    # Advance past the marker (and an optional trailing colon).
    rest = text[idx:]
    nl = rest.find("\n")
    return rest[nl + 1:] if nl >= 0 else rest


def _extract_forwarded_names(text: str, real_names: set[str]) -> list[str]:
    """Extract section names referenced in ``text`` that are REAL sections.

    Matching rule (2026-09-09): scan mention positions left-to-right so
    first-mention order is preserved; longest names win when one ID is a
    substring of another (checked at each position).  Names that contain
    the current text as a prefix (e.g. a model citing
    ``anonymous:code:8667-9189 (522 chars)`` with a stale offset suffix)
    match the base ID.  ``<anonymous>`` never matches.  Duplicates removed.
    """
    if not text or not real_names:
        return []
    names_sorted = sorted((n for n in real_names if n and n != "<anonymous>"),
                          key=len, reverse=True)
    found: list[str] = []
    seen: set[str] = set()
    idx = 0
    while idx < len(text):
        best: str | None = None
        for name in names_sorted:
            if name in seen:
                continue
            pos = text.find(name, idx)
            if pos >= 0 and (best is None or pos < best_pos or (pos == best_pos and len(name) > len(best))):
                best = name
                best_pos = pos
        if best is None:
            break
        found.append(best)
        seen.add(best)
        idx = best_pos + len(best)
    return found


def modify_artifact_round1_contract(text: str, runtime: dict[str, Any]) -> str | None:
    """Contract for R1 (targeting) of modify_artifact.

    2026-09-09 (user-approved design): R1's output is LOOSE. The model
    reasons in prose and concludes with a demarcated 'final list' block
    naming sections and reasons.  A structured JSON package is still
    accepted when the model emits one.  The contract is a SALVAGE
    VALIDATOR, not a schema enforcer:

      1. Non-empty text.
      2. If ``runtime["_parsed_json"]`` is a dict that looks like our
         targeting schema (has ``relevant_sections``), validate it as
         before (names real, anchors real, reasons present) and PASS.
      3. Otherwise, locate the 'final list' demarcation and extract
         forwarded names from it; PASS when at least one REAL section
         ID appears in the demarcated block.
      4. FAIL only when neither path yields usable targeting (no JSON,
         no demarcation, or a demarcation with zero real section names).

    The validated targeting (parsed fields or scanned names) is stored
    in ``runtime["_r1_targeting"]`` for the downstream consumers (R2
    context fetch, no-invention set, under-delivery counter).
    """
    if not text.strip():
        return "empty_response"
    store = runtime.get("_artifact_store")
    artifact_id = runtime.get("_artifact_id")
    real_names: set[str] = set()
    if store is not None:
        artifact = store.get(artifact_id)
        if artifact is not None:
            real_names = {s.name for s in artifact.sections if s.name}
    data = runtime.get("_parsed_json")
    # --- Path 1: structured JSON targeting package ---
    if isinstance(data, dict) and "relevant_sections" in data:
        if not isinstance(data.get("relevant_sections"), list) or not isinstance(data.get("new_sections"), list):
            return "missing_sections: relevant_sections/new_sections must be arrays"
        removals = data.get("removals")
        if removals is not None and not isinstance(removals, list):
            return "missing_sections: removals must be an array"
        relevant_names: list[str] = []
        for section in data.get("relevant_sections", []):
            if not isinstance(section, dict) or not section.get("name"):
                return "missing_sections: every relevant section requires a section ID"
            if not str(section.get("reason", "")).strip():
                return "missing_sections: every relevant_sections entry requires a reason"
            if real_names and section["name"] not in real_names:
                return f"bad_anchor: relevant section ID '{section['name']}' not found in artifact"
            relevant_names.append(str(section["name"]))
        for removal in removals or []:
            if not isinstance(removal, dict) or not removal.get("name"):
                return "missing_sections: every removals entry requires a section ID"
            if not str(removal.get("reason", "")).strip():
                return "missing_sections: every removals entry requires a reason"
            if real_names and removal["name"] not in real_names:
                return f"bad_anchor: removal section ID '{removal['name']}' not found in artifact"
            if removal["name"] in relevant_names:
                return (
                    f"bad_anchor: section '{removal['name']}' is targeted for both "
                    "modification and removal"
                )
        for new_sec in data.get("new_sections", []):
            if not isinstance(new_sec, dict):
                continue
            anchor = new_sec.get("anchor_section")
            if anchor and real_names and anchor not in real_names:
                return f"bad_anchor: new_sections anchor '{anchor}' not found in artifact"
        runtime["_r1_targeting"] = {"mode": "json", "data": data}
        return None
    # --- Path 2: demarcated 'final list' prose block ---
    block = _extract_final_list_block(text)
    if block is not None:
        names = _extract_forwarded_names(block, real_names)
        if names:
            runtime["_r1_targeting"] = {
                "mode": "final_list",
                "names": names,
                "block": block,
            }
            return None
        return (
            "missing_sections: 'final list' block names no real section IDs "
            f"(valid ids: {sorted(real_names)[:8]}...)"
            if real_names else
            "missing_sections: 'final list' block found but no artifact sections available"
        )
    # --- Neither path: unusable targeting ---
    return (
        "missing_sections: no 'final list' section and no JSON targeting "
        "package in the response"
    )


def _normalize_outline_section_id(
    name: str,
    artifact: Any,
) -> str | None:
    """Resolve an outline display string to a real section name.

    2026-09-10 (user-ruled, live 19:17 finding): the outline displays
    sections as ``{kind} {name} @ [{start}-{end}]``, and the system prompt
    instructs the model to "use exact section IDs from the artifact
    outline" — so models faithfully copy the DISPLAY text as the section
    ID. Failure shapes that must resolve:

      * decorated bare name:  ``code render @ [16113-16122]`` -> ``render``
      * duplicate names: the char span disambiguates (a bare-name lookup
        cannot), e.g. two sections both named ``render``.

    Anonymous synthetic IDs (``anonymous:code:15766-15893 @ [...]``) are
    covered by the same parse — the outline renders them identically, so
    one normalization path serves both, per the consistency ruling.

    2026-09-10 (live 20:26 finding, user-approved repair): models copy the
    outline VERBATIM, including the length decoration build_outline appends
    (``... @ [4812-4943] (131 chars)``), and may prefix a kind word onto an
    anonymous ID without the span (``code anonymous:code:15920-16089``).
    The display form is therefore:

      ``{kind} {name} @ [{start}-{end}] ({length} chars)``

    Resolution order: (1) strip the optional `` (N chars)`` decoration
    (it is display metadata, not part of the ID); (2) fast-path the
    stripped string; (3) span regex with the decoration already removed
    (kind word + name + span; name-part must still agree with the
    spanned section, guarding bogus spans); (4) spanless display: drop
    ONE leading kind word and retry the span regex so a missing span can
    still resolve by name (e.g. ``code render`` -> ``render``) — the
    name-part check against the artifact still applies via the fast
    path; a spanless anonymous ID with a kind word
    (``code anonymous:code:15920-16089``) resolves by dropping the kind
    word and taking the bare fast path.

    Returns the real section name, or ``None`` when the string does not
    denote a section of this artifact (bare names pass through only if
    they actually exist — the caller checks that first).
    """
    if not name or artifact is None:
        return None
    # 2026-09-10 (live 20:26 repair): the trailing "(N chars)" is outline
    # display decoration; strip it before any matching. It is never part
    # of a real section name (names come from code constructs and the
    # synthetic ``anonymous:kind:start-end`` grammar, neither of which
    # ends in that pattern).
    stripped = re.sub(r"\s*\(\d+ chars\)\s*$", "", name.strip())
    real_names = {s.name for s in artifact.sections if s.name}
    # Fast path: already a real section name.
    if stripped in real_names:
        # 2026-09-10 (live 23:12 fix): duplicate names make a BARE name
        # ambiguous — the section name map keeps one arbitrary winner, so
        # replacing 'render' under two render sections can silently target
        # the wrong one (the 236-char near-empty artifact). When several
        # sections share this name, spanless display is NOT resolvable:
        # return None and let the anchor check report the ambiguity with
        # the disambiguating spans. A spanned display string (handled
        # below) remains the correct disambiguation path.
        owners = [s for s in artifact.sections if s.name == stripped]
        if len(owners) == 1:
            return stripped
        return None
    # Outline display form: optional kind word, name, span. The span is the
    # authoritative resolver (unique even under duplicate names); the name
    # part must ALSO match the spanned section to guard against a model
    # combining a real name with a bogus span. 2026-09-10: the length
    # decoration is already stripped above, so the match may end at "]".
    m = re.match(r"^(?:\S+\s+)?(?P<name>.+?)\s*@\s*\[(?P<start>\d+)-(?P<end>\d+)\]$", stripped)
    if not m:
        # 2026-09-10 (live 20:26 repair): spanless display with a kind word
        # (``code render``, ``code anonymous:code:15920-16089``). Drop ONE
        # leading kind word and retry the span regex; if the remainder is a
        # real section name it resolves via a second fast-path lookup below.
        m = re.match(r"^\S+\s+(?P<rest>.+)$", stripped)
        if not m:
            return None
        rest = m.group("rest").strip()
        m = re.match(r"^(?P<name>.+?)\s*@\s*\[(?P<start>\d+)-(?P<end>\d+)\]$", rest)
        if m:
            # Span present without a resolvable first parse: fall through to
            # the shared span resolver below with the new match.
            pass
        else:
            # Pure spanless display: only the bare name (after kind-word
            # drop) can resolve, via the real-name fast path. 2026-09-10:
            # the duplicate-name guard applies here too — spanless under
            # duplicate names is ambiguous by construction.
            dup = [s for s in artifact.sections if s.name == rest]
            if len(dup) == 1:
                return rest
            if len(dup) > 1:
                spans = ", ".join(
                    f"@ [{s.char_start}-{s.char_end}]" for s in dup
                )
                return f"__ambiguous__:{spans}"
            return None
    cs, ce = int(m.group("start")), int(m.group("end"))
    name_part = m.group("name").strip()
    for section in artifact.sections:
        if section.char_start == cs and section.char_end == ce:
            # Both the span AND the name-part must agree with the artifact.
            if name_part == (section.name or ""):
                return section.name
            return None
    return None


def modify_artifact_round2_contract(text: str, runtime: dict[str, Any]) -> str | None:
    """Contract for R2 (edit generation) of modify_artifact.

    2026-09-08 (user-ruled semantics): R1's targeting is a STRONG
    RECOMMENDATION; R2 makes the final judgment call but is BOUNDED —
    it may not invent targets outside the forwarded set, and it
    re-states removals itself (with reasons).

    2026-09-10 (user-ruled refactor, principled single-owner design):
    this contract is SELF-SUFFICIENT. When ``runtime["_parsed_json"]``
    is absent, the contract performs its OWN extraction: JSON first,
    then the unified-diff dialect (2026-09-10 user ruling) as fallback.
    On success it stashes the validated edit package in
    ``runtime["_pending_artifact_edit"]`` for the executor to consume,
    so the generic CONTRACTS dispatch is the single validation site for
    the one-shot modify path. Callers that pre-parse (the T3
    dispatcher, for its own no_change_package semantics) may still pass
    ``_parsed_json`` — the contract uses it when present.

    Validates:
      1. The text parses as JSON (via ``runtime["_parsed_json"]`` or
         self-extraction; unified diff accepted as fallback dialect).
      2. Required top-level keys: ``replacements``, ``insertions``
         (both arrays; possibly empty).  ``removals`` optional array.
      3. Every ``replacements[].name`` and ``removals[].name``
         resolves against the artifact's real current sections.
      4. ``insertions[].anchor_section`` resolves against real
         sections OR R1's planned ``new_sections`` names.
      5. Conflict guards: a section may not appear in both
         ``replacements`` and ``removals``; an insertion may not
         anchor on a removed section.
      6. ``removals[]`` entries require a ``reason``.
      7. A non-empty change set is required (no_change_package).
    """
    if not text.strip():
        return "empty_response"
    # 2026-09-10 (user-ruled refactor): self-extraction when the caller
    # did not pre-parse. JSON first; the unified-diff dialect is the
    # fallback (2026-09-10 user ruling: accept a standard diff as the
    # change order). The diff path yields a single whole-artifact
    # replacement, exactly as the previous inline one-shot block built.
    data = runtime.get("_parsed_json")
    if data is None:
        data = _extract_json(text)
    if data is None:
        _diff_text = extract_last_diff_block(text)
        if _diff_text is None:
            return "malformed_json: response did not contain an edit package or a change-order diff"
        _store = runtime.get("_artifact_store")
        _aid = runtime.get("_artifact_id")
        _target = _store.get(_aid) if _store is not None else None
        if _store is None or _target is None:
            return "no_current_artifact: no artifact selected for modify_artifact"
        _patched = apply_unified_diff(_target.content, _diff_text)
        if _patched is None:
            return "diff_does_not_apply: unified diff failed to parse or apply to the artifact"
        if _patched == _target.content:
            return "no_change_package: diff applies but changes nothing"
        # Short-circuit success: the diff dialect's verification IS
        # apply_unified_diff's context check plus the no-change rule.
        # Section anchoring does not apply (the sentinel replaces the
        # whole artifact; artifact_store has no sentinel handling, and
        # the previous inline block also bypassed apply_edits preview
        # for this dialect).
        runtime["_pending_artifact_edit"] = {
            "edits": {
                "replacements": [{"name": "__whole_artifact__", "new_text": _patched}],
                "insertions": [],
                "removals": [],
            },
            "artifact_id": _target.artifact_id,
            "dialect": "unified_diff",
        }
        return None
    if not isinstance(data, dict):
        return "malformed_json: top-level JSON value is not an object"
    # 2026-09-09 schema tolerance (user ruling, option c): omitted keys
    # default to EMPTY ARRAYS. The live T3 modify failure showed a SOTA
    # model emitting a complete, correct package with only "replacements"
    # -- omitting "insertions"/"removals" instead of writing empty arrays.
    # An omitted array-valued key is semantically identical to an empty
    # one, so tolerance here never accepts a WRONG package; it accepts an
    # INCOMPLETE-but-valid envelope. Non-list values still fail.
    if "replacements" not in data:
        return "missing_replacements: missing required key 'replacements'"
    replacements = data.get("replacements")
    insertions = data.setdefault("insertions", [])
    removals = data.setdefault("removals", [])
    if not isinstance(replacements, list):
        return "missing_replacements: replacements must be an array"
    if not isinstance(insertions, list) or not isinstance(removals, list):
        return "missing_replacements: insertions/removals must be arrays"
    real_names: set[str] = set()
    store = runtime.get("_artifact_store")
    artifact_id = runtime.get("_artifact_id")
    _artifact = store.get(artifact_id) if store is not None else None
    if store is not None and _artifact is not None:
        real_names = {s.name for s in _artifact.sections if s.name}
        # 2026-09-10 (user-ruled): normalize outline display IDs ONCE, in
        # place, so every downstream consumer (anchor checks, the no-change
        # preview, the stashed pending edit, and the executor) sees real
        # section names. Mutates replacements[].name,
        # insertions[].anchor_section, and removals[].name.
        # 2026-09-10 (live 23:12): the normalizer returns
        # "__ambiguous__:@ [s-e], @ [s-e]" when a spanless display matches
        # several same-named sections. That is a TRUTHFUL bad_anchor, not a
        # pass-through name: report it immediately with the spans so the
        # model can disambiguate on retry.
        for rep in replacements:
            if isinstance(rep, dict) and rep.get("name") and rep["name"] != "__whole_artifact__":
                _norm = _normalize_outline_section_id(rep["name"], _artifact)
                if isinstance(_norm, str) and _norm.startswith("__ambiguous__:"):
                    return (
                        "bad_anchor: section name '" + str(rep["name"]).strip()
                        + "' is ambiguous (" + _norm[len("__ambiguous__:"):].strip()
                        + "); use the exact spanned display ID from the outline"
                    )
                rep["name"] = _norm or rep["name"]
        for ins in insertions:
            if isinstance(ins, dict) and ins.get("anchor_section"):
                _norm = _normalize_outline_section_id(
                    ins["anchor_section"], _artifact
                )
                if isinstance(_norm, str) and _norm.startswith("__ambiguous__:"):
                    return (
                        "bad_anchor: insertion anchor '" + str(ins["anchor_section"]).strip()
                        + "' is ambiguous (" + _norm[len("__ambiguous__:"):].strip()
                        + "); use the exact spanned display ID from the outline"
                    )
                ins["anchor_section"] = _norm or ins["anchor_section"]
        for rem in removals or []:
            if isinstance(rem, dict) and rem.get("name"):
                _norm = _normalize_outline_section_id(rem["name"], _artifact)
                if isinstance(_norm, str) and _norm.startswith("__ambiguous__:"):
                    return (
                        "bad_anchor: removal name '" + str(rem["name"]).strip()
                        + "' is ambiguous (" + _norm[len("__ambiguous__:"):].strip()
                        + "); use the exact spanned display ID from the artifact outline"
                    )
                rem["name"] = _norm or rem["name"]
    # R1's forwarded targeting: names R1 surfaced for modification,
    # removal, and planned new sections.  2026-09-09: the targeting may
    # come from a structured JSON package (parsed fields) OR from the
    # salvage contract's 'final list' scan (runtime["_r1_targeting"]).
    r1_data = runtime.get("_r1_json")
    r1_relevant: set[str] = set()
    planned_names: set[str] = set()
    r1_removals: set[str] = set()
    if isinstance(r1_data, dict):
        for sec in r1_data.get("relevant_sections", []) or []:
            if isinstance(sec, dict) and sec.get("name"):
                r1_relevant.add(sec["name"])
        for sec in r1_data.get("new_sections", []) or []:
            if isinstance(sec, dict) and sec.get("name"):
                planned_names.add(sec["name"])
        for sec in r1_data.get("removals", []) or []:
            if isinstance(sec, dict) and sec.get("name"):
                r1_removals.add(sec["name"])
    targeting = runtime.get("_r1_targeting") or {}
    if targeting.get("mode") == "final_list":
        # Loose-prose path: the scanned names are the forwarded set.
        scanned: list[str] = targeting.get("names", [])
        r1_relevant.update(scanned)
    forwarded: set[str] = r1_relevant | r1_removals
    rep_names: set[str] = set()
    for rep in replacements:
        if not isinstance(rep, dict):
            continue
        name = rep.get("name")
        if not name:
            continue
        # 2026-09-10 (user-ruled refactor): the whole-artifact sentinel is
        # produced by the unified-diff dialect above. Its correctness is
        # verified by apply_unified_diff's context check + the no-change
        # rule, not by section anchoring (server.py consumes the sentinel
        # identically). Skip it in the per-name anchor validation.
        if name == "__whole_artifact__":
            continue
        # 2026-09-10 (user-ruled): outline display strings
        # ("code render @ [16113-16122]") resolve to their real section.
        # Normalization happens BEFORE the no-invention check so a
        # display-form reference to a forwarded section counts as forwarded.
        artifact_obj = (
            store.get(artifact_id) if store is not None else None
        )
        name = _normalize_outline_section_id(name, artifact_obj) or name
        rep_names.add(name)
        if real_names and name not in real_names:
            return f"bad_anchor: replacement name '{name}' not found in artifact"
        # No-invention rule (user ruling 2026-09-08): R2 may only
        # act on sections R1 forwarded.  Only enforced when R1
        # actually produced a parsed targeting package.
        if r1_data is not None and forwarded and name not in forwarded:
            return f"unplanned_target: replacement '{name}' was not in R1's targeting package"
    artifact_obj_ins = store.get(artifact_id) if store is not None else None
    for ins in insertions:
        if isinstance(ins, dict) and ins.get("anchor_section"):
            _ins_anchor = _normalize_outline_section_id(
                ins["anchor_section"], artifact_obj_ins
            ) or ins["anchor_section"]
            if _ins_anchor not in real_names | planned_names:
                return f"bad_anchor: insertion anchor_section '{ins['anchor_section']}' not found in artifact or R1 output"
    if real_names:
        for removal in removals or []:
            if not isinstance(removal, dict):
                continue
            name = removal.get("name")
            if not name:
                continue
            name = _normalize_outline_section_id(name, artifact_obj_ins) or name
            if name not in real_names:
                return f"bad_anchor: removal name '{name}' not found in artifact"
            if name in rep_names:
                return (
                    f"bad_anchor: section '{name}' is targeted for both "
                    "replacement and removal"
                )
            # R2 may only REMOVE sections R1 surfaced for removal or
            # modification (its judgment call within the forwarded
            # set) — removing an untouched section is an invention.
            if r1_data is not None and forwarded and name not in forwarded:
                return f"unplanned_target: removal '{name}' was not in R1's targeting package"
            if not str(removal.get("reason", "")).strip():
                return "missing_replacements: every removals entry requires a reason"
        for ins in insertions:
            if (
                isinstance(ins, dict)
                and ins.get("anchor_section")
                and ins["anchor_section"] in {r.get("name") for r in removals or [] if isinstance(r, dict)}
            ):
                return (
                    f"bad_anchor: insertion anchors on removed section "
                    f"'{ins['anchor_section']}'"
                )
    # 2026-09-10 (user-ruled refactor): a non-empty change set is required.
    # The T3 dispatcher enforces this itself (its no_change_package rule
    # predates this contract's self-sufficiency); the one-shot path relied
    # on the inline block's _has_changes check which is now deleted, so the
    # rule lives here where it belongs.
    if not (replacements or insertions or removals):
        return "no_change_package: empty change order (no replacements, insertions, or removals)"
    # 2026-09-11 (live v7 finding, user-ruled): models over-generalize the
    # outline display format into CONTENT — new_text lines starting with the
    # kind word ("code .card {" / "style .x") are display-format bleed, not
    # code. Strip the leading kind word on EVERY line (v2, per-line): the
    # v9 finding showed models prefix EVERY rule in new_text ("code
    # .card-priority {" on line 5, 9, 13...), not just line one. Applied to
    # replacements and insertions alike; removals carry no content.
    # Harness-display vocabulary, not language constructs: generalizes
    # across artifact kinds (code/style/script/markup/prose).
    #
    # 2026-09-12 FIX: v2 used ``sub(r"\1\2", ...)`` with groups
    # ``([ \t]*)(?:kind)(\s+)``.  For a column-0 ``code .card {`` the
    # groups are ``\1=""`` and ``\2=" "``, so the substitution deleted
    # the kind word but KEPT its separator space, yielding ``" .card {"``.
    # The separator belongs to the display format, not the content, so it
    # must be consumed too.  (The leading space is harmless in CSS but
    # corrupts indentation-sensitive content -- markdown lists, Python
    # bodies, diffs.)  ``\s+`` after the kind word is therefore consumed,
    # while the ORIGINAL indent ``\1`` is preserved and newlines are
    # never touched: ``[ \t]+`` requires at least one same-line space.
    _KIND_PREFIX = re.compile(r"^([ \t]*)(?:code|style|script|markup|prose)[ \t]+", re.M)
    for rep in replacements:
        if isinstance(rep, dict) and isinstance(rep.get("new_text"), str):
            rep["new_text"] = _KIND_PREFIX.sub(r"\1", rep["new_text"])
    for ins in insertions:
        if isinstance(ins, dict) and isinstance(ins.get("new_text"), str):
            ins["new_text"] = _KIND_PREFIX.sub(r"\1", ins["new_text"])
    # Preview against the artifact store when one is wired: the
    # deterministic reintegration engine is the ground truth for anchors.
    # 2026-09-11: pass the resolved invented-replacement guard profile so
    # apply_edits can reject stub replacements per tier thresholds.
    _store = runtime.get("_artifact_store")
    _aid = runtime.get("_artifact_id")
    if _store is not None:
        try:
            _preview = _store.apply_edits(
                data, artifact_id=_aid, preview=True,
                guard_config=runtime.get("_guard_cfg") or {},
            )
        except Exception as exc:
            return f"artifact_apply_failed: edit package error: {exc}"
        if _preview.get("errors"):
            return "artifact_apply_failed: " + "; ".join(
                str(item) for item in _preview["errors"]
            )
    # Success: stash the validated package for the executor. The diff
    # dialect already stashed its whole-artifact package above; the JSON
    # path stashes here. 2026-09-10 (live fix): the stash MUST carry
    # artifact_id — server.py's _commit_pending_artifact reads
    # pe["artifact_id"] unconditionally, and the bare {"edits": data}
    # envelope from the refactor crashed with KeyError 'artifact_id' on the
    # one-shot T0 pass. The contract used _aid for its preview directly
    # above, so the value is in scope by construction.
    if "_pending_artifact_edit" not in runtime:
        runtime["_pending_artifact_edit"] = {"edits": data, "artifact_id": _aid}
    return None


GUARDS = {"dither_guard": dither_guard, "repetition_loop_guard": repetition_loop_guard, "modify_output_cap_guard": modify_output_cap_guard}
CONTRACTS = {
    "unified_diff_patch": unified_diff_patch_contract,
    "python_new_code": python_new_code_contract,
    "web_new_code": web_new_code_contract,
    "compiled_coding_new_code": compiled_coding_new_code_contract,
    "new_code_generation": new_code_generation_contract,
    "prose_contract": prose_contract,
    "evaluate_artifact": evaluate_artifact_contract,
    "modify_artifact_round1": modify_artifact_round1_contract,
    "modify_artifact_round2": modify_artifact_round2_contract,
}

# Module-level cache of the last (model, context) pair the runtime
# asked the VRAM manager to prepare. The next call skips the redundant
# evict+reload when the target is unchanged.
last_prepared_key: tuple[str, int] | None = None

DEFAULT_CHECKPOINT_INTERVAL_S = 30.0
# First-token timeout: maximum wall time to wait between
# sending the request to the model and receiving the first
# token.  The dither / loop guards run on the *accumulated*
# text, so a model that emits zero tokens (e.g. a decode
# stall on a hard prompt) is invisible to them.  This
# deadline fires the same retry ladder the dither / loop /
# vram_breach errors use, with an error string that flows
# through ``retry_on``.
DEFAULT_FIRST_TOKEN_TIMEOUT_S = 60.0

# Output ceiling (2026-08-28): convergence-failure proxy and scope
# declaration of the black box (single-domain/single-task subagent).
#
#   max_chars        -- per-ATTEMPT ceiling (one stream)
#   max_total_chars  -- CUMULATIVE ceiling across ALL of the problem's
#                       attempts (the real scope boundary: three 75k
#                       runaways must not sum to a 225k grind)
#
# Failsafe ONLY: these fallbacks are used when the harness yaml does
# not declare "output_ceiling.max_chars"/"max_total_chars".  The
# configured values ALWAYS win -- the constants exist so a missing key
# cannot leave generated output unbounded, not as operational numbers.
# Shipped defaults are set in config.yaml (75k/100k; ~18k/24k tokens).
# Users wanting short local answers cut them further; ceiling-exceeded
# problems are meant to escalate to the cloud T3 tier.
DEFAULT_OUTPUT_CEILING_CHARS = 150000
DEFAULT_OUTPUT_CEILING_TOTAL_CHARS = 200000
# Slow-decode timeout: after the first token arrives, if no
# further token lands within this many seconds, abort.  This
# catches prompts where the model starts decoding but then
# stalls (e.g. a 1 tok/s rate on a hard NPU call) without
# waiting for the full per-problem wall to elapse.
DEFAULT_SLOW_DECODE_TIMEOUT_S = 90.0

DEFAULT_DIRECT_CLOUD_CONFIG: dict[str, Any] = {
    "silence_timeout_s": 600,
    "total_timeout_s": 600,
    "think": False,
    "max_tokens": 50000,
    "prompts": {
        "general_use": {
            "system_suffix": (
                "You are a capable general assistant. Answer the user's request "
                "directly and accurately. Use supplied context as reference, but "
                "treat the user's request as authoritative. Do not emit draft/final "
                "scaffolding or hidden reasoning."
            ),
            "output_instruction": (
                "Answer directly. When code is requested, use one complete fenced "
                "code block and stop."
            ),
        },
    },
    "cloud_guards": {
        "dither_enabled": True,
        "shortstop_enabled": True,
        "shortstop_rate": 100.0,
        "shortstop_min_chars": 30000,
        "shortstop_window": 10000,
        "dither_limit": 10000,
        "dither_limit_think_scale": 3,
        "loop_enabled": True,
        "loop_guard_min": 2000,
        "loop_guard_skip": 2000,
        "loop_guard_match": 100,
        "loop_guard_window": 400,
    },
}

# Rate-stall gate (StallMonitor) defaults.  Derived from the
# 14-stream profiling run of 2026-08-27: PASS streams never go
# silent mid-attempt; terminal runaways all show a zero 60s
# window preceded (usually) by a partial-rate collapse.  A
# negative ``stall_enabled`` disables the gate entirely.
DEFAULT_STALL_ENABLED = 1
DEFAULT_STALL_WINDOW_S = 60.0
DEFAULT_STALL_ARM_WINDOWS = 1
DEFAULT_STALL_ARM_MIN_CHARS = 300.0
DEFAULT_STALL_ZERO_K = 2
DEFAULT_STALL_PARTIAL_RATIO = 0.30
DEFAULT_STALL_PARTIAL_K = 4
DEFAULT_STALL_RESET_RATIO = 0.50

# v12 YAML-driven defaults.  The runtime is a dumb executor:
# every knob it consults has a default in this dict and a
# per-harness override in the YAML.  A negative integer for a
# cadence/threshold field means "primitive inactive"; the
# runtime checks this each turn and no-ops the primitive.
#
# **NOTE ON TIMER DEFAULTS**: the timer values here are
# intentionally GENEROUS.  They were tuned for cold-load
# scenarios on the local 16 GB card with the 19.4B q4_K_M
# specialist models.  Hardware-aware tuning is a TODO; until
# then, the pathology monitor's predicted budget (derived
# from prompt size and the measured prefill rate) overrides
# the warm ``first_token_timeout_s`` when the model is
# resident.  The cold-load budget (75s) and slow-decode
# budget (90s) are the safety nets.
DEFAULT_HARNESS_CONFIG: dict[str, Any] = {
    # ── Stream-primitive cadences and thresholds ──
    # Negative = inactive.  A primitive that finds its
    # cadence field < 0 returns immediately without acting.
    "fence_counter.check_cadence_chunks": -1,
    "fence_counter.tail_after_close_without_reopen_chunks": -1,
    "fence_counter.fence_languages": [],  # empty = use FENCE_ALIASES from the work item's language
    "fence_counter.contract_on_truncate": None,
    # v13 patch block counter: analogous to fence_counter but
    # watches <patch>...</patch> tags.  When the model closes
    # ``max_blocks`` patch blocks, request_close() the stream.
    # The grader takes the LAST patch via extract_last_patch_block,
    # so the counter is purely a structural cap on how many
    # drafts the model may emit.  Negative cadence = inactive.
    # No preamble/tail budgets — the existing dither/loop guards
    # catch pathologies in the prose between patches.
    "patch_block_counter.check_cadence_chunks": -1,
    "patch_block_counter.max_blocks": 3,
    "dither_check.check_cadence_chunks": 25,
    "dither_check.pattern": (
        r"(?i)\b(?:"
        r"wait"
        r"|actually"
        r"|hmm"
        r"|sorry"
        r"|instead"
        r"|reconsider(?:ing|s|ed)?"
        r"|re-?examin(?:e|ing|ed)"
        r"|but\s+wait"
        r"|on\s+(?:second\s+)?thought"
        r"|let\s+me\s+(?:re-?think|re-?do|try\s+again|reconsider)"
        r")\b"
    ),
    "dither_check.limit": 35,
    "loop_check.check_cadence_chunks": 25,
    "loop_check.min_chars": 1400,
    "loop_check.skip_chars": 1400,
    "loop_check.match_size": 100,
    "loop_check.window": 400,
    # ── Timers (generous defaults; tune per hardware) ──
    "timers.first_token_timeout_s": 60.0,
    "timers.slow_decode_timeout_s": 90.0,
    # Rate-stall gate: 1=enabled, 0=disabled (reversible switch).
    "stall.enabled": DEFAULT_STALL_ENABLED,
    "stall.window_s": DEFAULT_STALL_WINDOW_S,
    "stall.arm_windows": DEFAULT_STALL_ARM_WINDOWS,
    "stall.arm_min_chars": DEFAULT_STALL_ARM_MIN_CHARS,
    "stall.zero_k": DEFAULT_STALL_ZERO_K,
    "stall.partial_ratio": DEFAULT_STALL_PARTIAL_RATIO,
    "stall.partial_k": DEFAULT_STALL_PARTIAL_K,
    "stall.reset_ratio": DEFAULT_STALL_RESET_RATIO,
    # Output ceiling: convergence-failure proxy (see constants block).
    # Failsafe defaults; the yaml "output_ceiling.*" ALWAYS wins.
    "output_ceiling.enabled": 1,
    "output_ceiling.max_chars": DEFAULT_OUTPUT_CEILING_CHARS,
    "output_ceiling.max_total_chars": DEFAULT_OUTPUT_CEILING_TOTAL_CHARS,
    "timers.cold_load_first_token_timeout_s": 75.0,
    "timers.pathology_safety_factor": 2.5,
    # ── Error-correction ladder ──
    "error_correction.max_functional_errors_before_commit": 1,
    "error_correction.functional_contract": None,
    "error_correction.functional_errors": [],
    "error_correction.t1_strategy": "full_only",
    "error_correction.t1_targeted_artifact": "unified_diff",
    "error_correction.t1_targeted_user_modifier": (
        "The previous code block failed functional validation with this error:\n"
        "{t0_error}\n\n"
        "Here is the previous code block in full:\n\n"
        "```python\n{t0_code}\n```\n\n"
        "Produce a unified diff that fixes the error. Wrap the diff in "
        "a <patch>...</patch> block. Do not regenerate the entire code "
        "block — emit only the smallest possible change that fixes the error."
    ),
    "error_correction.t1_full_user_modifier": (
        "The model's generated code has the following error trace:\n"
        "{prior_error}\n\n"
        "Regenerate the code without the error."
    ),
    # ── Pathology guardrails (re-roll with new seed) ──
    "pathology.max_re_rolls_per_model": 2,
    "pathology.re_roll_seed_strategy": "random_per_model",
    "pathology.errors": [
        "dither",
        "loop",
        "first_token_timeout",
        "slow_decode_timeout",
        "vram_breach",
        "token_exhaustion",
        "backend_error",
        "empty_response",
        "no_code_fence_unclosed",
        # Rate-stall gate errors (StallMonitor): a 60s-window zero
        # output mid-attempt (rate_stall) or a sustained collapse
        # below 30% of the stream's own median active rate
        # (rate_collapse).  Classified as pathology so the ladder
        # REROLLS with a fresh seed instead of breaking out with an
        # empty result.
        "rate_stall",
        "rate_collapse",
    ],
    # ── Re-roll system modifiers (one-shot, on the immediate next re-roll) ──
    "re_roll_modifiers.dither": (
        "Your previous response on this task contained excess dithering "
        "(many revisions of the same reasoning). On this attempt: be concise, "
        "commit to your answer, and do not second-guess yourself. Reach the "
        "FINAL block and stop — do not keep re-examining or rewriting."
    ),
    "re_roll_modifiers.loop": (
        "Your previous response on this task contained a repetition loop. "
        "On this attempt: be concise, commit to your answer, and do not repeat "
        "the same phrases. Reach the FINAL block and stop."
    ),
    # ── Modify-only output cap (2026-09-10, user ruling) ──
    # A change order larger than its target artifact is wrong by
    # construction.  On cap trip: extract the LAST COMPLETE change order
    # and validate; pass -> accept; fail -> ONE dedicated re-roll (fresh
    # seed) that does NOT consume the shared pathology budget.
    "output_cap_factor": 0.0,  # disabled by default; modify_artifact sets 1.0
    "output_cap_min_chars": 4000,
    "output_cap_max_re_rolls": 0,
    "re_roll_modifiers.output_cap": (
        "Your previous pass grew far beyond the size of the target artifact "
        "without producing a complete change order. On this attempt: commit "
        "to the minimal change order immediately, emit the complete JSON "
        "object, and stop as soon as it is closed."
    ),
    # ── Safety-net absolute cap (T1a is FREE; T1b counts as 1) ──
    "max_calls_per_instance": 12,
    # ── Cancellation notice (hard stop, not a pathology trigger) ──
    "cancelled_notice": "turn/operation cancelled signal received",
    # ── T5 zero-shot generalist notice ──
    "t5_notice_success": (
        "NOTICE: returning a zero shot attempt from generalist model "
        "({model}) after error/retry exhaustion by specialists."
    ),
    "t5_notice_failure": (
        "NOTICE: specialists exhausted AND generalist zero-shot errored: {error}"
    ),
}


def _resolve_cfg(cfg: dict[str, Any], key: str) -> Any:
    """Read a nested key with default fallback.

    Supports dotted keys like ``"dither_check.limit"`` by
    walking the dict one level at a time.  Falls back to
    ``DEFAULT_HARNESS_CONFIG`` when the key is absent.
    """
    parts = key.split(".")
    cur: Any = cfg
    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            return DEFAULT_HARNESS_CONFIG.get(key)
        cur = cur[part]
    return cur


# v12 helper functions.  These are the building blocks the
# T1 primitive (step 4) will compose.  They are pure
# functions over text; the runtime orchestrates them.

# Language-to-fence-alias map.  The router classifies the
# request into a single language (e.g. ``python``, ``cpp``,
# ``html``).  The fence counter and the extraction helpers
# use this map to build the right regex for detecting code
# fences in that language.  Each language maps to a list of
# markdown fence tags the model might use.  The first entry
# is the canonical tag; the rest are aliases.
#
# This is the single source of truth for fence-language
# detection.  Adding a new language means adding an entry
# here (or in the harness YAML's ``fence_counter.fence_languages``
# override, which takes precedence).
FENCE_ALIASES: dict[str, list[str]] = {
    "python": ["python", "py"],
    "py": ["python", "py"],
    "cpp": ["cpp", "c++", "cxx"],
    "c++": ["cpp", "c++", "cxx"],
    "cxx": ["cpp", "c++", "cxx"],
    "c": ["c"],
    "rust": ["rust", "rs"],
    "rs": ["rust", "rs"],
    "go": ["go", "golang"],
    "golang": ["go", "golang"],
    "java": ["java"],
    "kotlin": ["kotlin", "kt"],
    "kt": ["kotlin", "kt"],
    "scala": ["scala"],
    "html": ["html", "htm"],
    "htm": ["html", "htm"],
    "javascript": ["javascript", "js"],
    "js": ["javascript", "js"],
    "typescript": ["typescript", "ts"],
    "ts": ["typescript", "ts"],
    # 2026-09-17: SCSS shares CSS's fence vocabulary.  `_detect_fence_language`
    # maps a ```scss fence to canonical "css", and `web_new_code_contract`
    # resolves its fence regex from FENCE_ALIASES[_detected] -- so "scss" MUST
    # appear in the css list or a ```scss fence matches nothing and the whole
    # response fails `no_code_fence`.  Canonical stays "css" (first element)
    # and the detected alias is "css", so the extension resolves to `.css`
    # (user ruling: use .css, not .scss).
    "css": ["css", "scss"],
    "jsx": ["jsx"],
    "tsx": ["tsx"],
    "scss": ["css"],
    "sql": ["sql"],
    "bash": ["bash", "sh", "shell"],
    "sh": ["bash", "sh", "shell"],
    "shell": ["bash", "sh", "shell"],
    "json": ["json"],
    "markdown": ["markdown", "md"],
    "md": ["markdown", "md"],
}


def _fence_aliases_for(language: str | None, runtime_cfg: dict[str, Any] | None = None) -> list[str]:
    """Return the fence-language aliases for the given language.

    Resolution order:
      1. The harness YAML's ``stream_primitives.fence_counter.fence_languages``
         override (if present and non-empty).
      2. The ``FENCE_ALIASES`` map (if the language is known).
      3. Fallback to ``[python, py]`` (backward compat).
    """
    if runtime_cfg:
        # The YAML nests fence_counter under stream_primitives.
        # Check both the nested path and the flat path for flexibility.
        sp = runtime_cfg.get("stream_primitives", {})
        fc = sp.get("fence_counter", {}) if isinstance(sp, dict) else {}
        override = fc.get("fence_languages")
        if not override:
            # Also check the flat path (for tests that pass a simplified config)
            fc_flat = runtime_cfg.get("fence_counter", {})
            if isinstance(fc_flat, dict):
                override = fc_flat.get("fence_languages")
        if override:
            return list(override)
    if language and language.lower() in FENCE_ALIASES:
        return FENCE_ALIASES[language.lower()]
    return ["python", "py"]


def _build_fence_open_re(aliases: list[str]) -> re.Pattern[str]:
    """Build a regex that matches the opening of a code fence
    in any of the given alias languages.

    Matches `````python``, `````py``, `````cpp``, etc.
    The match ends at the newline after the fence tag so
    the body starts on the next line.  Uses a whitespace/
    end-of-string lookahead instead of ``\\b`` so that
    aliases ending in non-word characters (e.g. ``c++``)
    match correctly.
    """
    escaped = "|".join(re.escape(a) for a in aliases)
    return re.compile(r"```(?:" + escaped + r")(?=\s|$)[^\n]*\n", re.IGNORECASE)


def _build_fence_tag_re(aliases: list[str]) -> re.Pattern[str]:
    """Build a regex that matches just the fence tag (`````python``
    without the trailing content).  Used by the fence counter
    to count opening positions without consuming the body.
    Uses a whitespace/end-of-string lookahead instead of
    ``\\b`` so that aliases ending in non-word characters
    (e.g. ``c++``) match correctly.
    """
    escaped = "|".join(re.escape(a) for a in aliases)
    return re.compile(r"```(?:" + escaped + r")(?=\s|$)", re.IGNORECASE)


_FENCE_CLOSE_RE = re.compile(r"```")
_PATCH_BLOCK_RE = re.compile(r"<patch>(.*?)</patch>", re.DOTALL)


def _normalize_patch_tag_newlines(text: str) -> str:
    """Ensure a newline follows every ``<patch>`` and ``</patch>`` tag.

    The model sometimes emits run-on lines where a ``<patch>`` or
    ``</patch>`` tag is immediately followed by content on the same
    line (e.g. ``</patch>```diff`` or ``</patch>Wait, I'm sorry``).
    This can merge a hallucinated second block into the tail of the
    first, or glue prose onto a patch boundary, which breaks the
    ``<patch>...</patch>`` block regex and the context-correctness
    gate.

    This inserts a newline after any ``<patch>`` or ``</patch>`` tag
    that is not already followed by a newline, so each tag sits on its
    own line and block boundaries are clean.  It is a pure
    whitespace-normalization pass — it never changes the tag text or
    the content inside a block.
    """
    if not text:
        return text
    # Insert a newline after any <patch> or </patch> tag that is not
    # already followed by a newline.  Use a lookahead so we don't
    # double-insert when the tag is already at end-of-line.
    return re.sub(
        r"(</?patch>)(?!\n)",
        r"\1\n",
        text,
    )


def extract_last_fence_body(text: str, language: str | None = None, runtime_cfg: dict[str, Any] | None = None) -> str | None:
    """Return the body of the LAST code fence in the given language.

    The fence language is resolved via ``_fence_aliases_for``:
    the harness YAML override takes precedence, then the
    ``FENCE_ALIASES`` map, then a fallback to ``[python, py]``.

    Returns ``None`` if no closed fence is found.  An
    unclosed opening fence is ignored (the streaming fence
    counter or the contract handles that case).
    """
    if not text:
        return None
    aliases = _fence_aliases_for(language, runtime_cfg)
    fence_open = _build_fence_open_re(aliases)
    last_body: str | None = None
    last_open_end = -1
    for om in fence_open.finditer(text):
        body_start = om.end()
        cm = _FENCE_CLOSE_RE.search(text, body_start)
        if cm is None:
            continue
        if om.end() > last_open_end:
            last_open_end = om.end()
            last_body = text[body_start:cm.start()]
    return last_body


def extract_last_python_fence_body(text: str) -> str | None:
    """Backward-compat wrapper for ``extract_last_fence_body(text, "python")``.

    Kept so existing callers (tests, the T1b prompt helper)
    don't break.  New code should call
    ``extract_last_fence_body`` with the language from the
    work item.
    """
    return extract_last_fence_body(text, "python")


    return extract_last_fence_body(text, "python")


_DIFF_HUNK_RE = re.compile(r"^--- \S+\n\+\+\+ \S+\n@@ -\d+", re.MULTILINE)


def extract_last_diff_block(text: str) -> str | None:
    """Extract the LAST unified-diff block from model output.

    2026-09-10 (user ruling): local models natively emit unified diffs as
    their "targeted change" dialect.  A diff block is recognized by the
    ``--- path`` / ``+++ path`` / ``@@ -n`` header trio; it may be bare in
    the prose or inside a fenced block.  Returns the diff text starting at
    the ``---`` header line, or None when no diff-shaped block exists.

    Distinguishing a diff from other fenced content: the header trio must
    appear within the first few lines of the candidate block.
    """
    if not text or "---" not in text or "@@" not in text:
        return None
    candidates: list[str] = []
    # Fenced blocks first (```diff or bare ```).
    fence_matches = list(re.finditer(r"```(?:diff)?[ \t]*\n(.*?)```", text, re.DOTALL))
    for fm in fence_matches:
        candidates.append(fm.group(1))
    # Bare diff: text from the LAST line starting with '--- ' that has a
    # '+++ ' line and an '@@' line following within 3 lines.
    for m in re.finditer(r"^--- .+$", text, re.MULTILINE):
        candidate = text[m.start():]
        head = candidate.splitlines()[:4]
        if len(head) >= 3 and head[0].startswith("--- ") and head[1].startswith("+++ ") and "@@" in head[2]:
            # 2026-09-10 (refactor self-check): a bare candidate that runs
            # to EOF can swallow the closing ``` fence and any trailing
            # prose. A unified diff body never contains triple backticks —
            # truncate at the first '```' anywhere in the candidate.
            _fence = candidate.find("```")
            if _fence != -1:
                candidate = candidate[:_fence].rstrip()
            candidates.append(candidate.rstrip())
    # Prefer the LAST candidate whose header trio validates via the hunk
    # regex (cheap structural check; full validation is apply_unified_diff's
    # job with context matching). 2026-09-10 (refactor self-check): the
    # fenced candidate is preferred over the bare one at equal position —
    # the fenced capture is already boundary-clean, while a bare candidate
    # that runs to EOF can carry trailing fence/junk. Reverse order puts
    # bare candidates (added later) first, so validate fenced first when
    # both derive from the same diff body.
    for candidate in reversed(candidates):
        if _DIFF_HUNK_RE.search(candidate):
            return candidate
    return None


def extract_last_patch_block(text: str) -> str | None:
    """Return the contents of the LAST ``<patch>...</patch>`` block.

    Returns ``None`` if no closed block is found.
    """
    if not text:
        return None
    blocks = _PATCH_BLOCK_RE.findall(text)
    if not blocks:
        return None
    return blocks[-1]


def _extract_json(raw: str) -> dict[str, Any] | None:
    """Extract a JSON object from model output.

    Ported from ``artifact_pipeline.py`` (the V13 path) for the
    two-round modify_artifact tier loop.  Handles common model
    quirks:

    - JSON embedded in prose
    - JSON wrapped in ```json fences
    - Trailing text after the JSON

    2026-09-08 (last-fence-wins): draft-then-final outputs carry the
    final package in the LAST fenced block; earlier blocks may hold
    draft reasoning.  The extractor prefers the LAST ```json fenced
    block that parses; only if no fence parses does it fall back to
    the LAST-JSON-OBJECT scan (2026-09-09: replaces the naive
    first-balanced-brace matcher, which latched onto a JS object
    literal inside narrative prose on the live T3 modify call —
    {'todo': ...} at pos 376 — and returned None while the REAL
    package sat valid at pos 897. The last-object scan walks
    right-to-left, brace-balancing with string awareness, and takes
    the last span that parses as a JSON object).

    Returns ``None`` if no valid JSON object can be extracted.
    """
    if not raw or not raw.strip():
        return None
    # Try direct parse.
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    # Try the LAST fenced block first (draft-then-final protocol:
    # the final answer is the last ```json block in the response).
    fence_matches = list(re.finditer(r"```(?:json)?\s*\n(.*?)```", raw, re.DOTALL))
    for fence_match in reversed(fence_matches):
        try:
            parsed = json.loads(fence_match.group(1).strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    # ── LAST-JSON-OBJECT scan (2026-09-09) ──
    # Collect every balanced, string-aware { ... } span and keep the one
    # that (a) parses as a JSON object, (b) ENDS latest in the text, and
    # (c) among spans sharing the same end, STARTS earliest (outermost).
    # Rule (b) prefers the last complete object; rule (c) prevents the
    # last INNER object (e.g. {"name": ..., "new_text": ...} inside a
    # replacements array) from shadowing its parent package — the live
    # T3 failure: the true 5-replacement package ended at 4013, and its
    # last inner object ALSO ended at 4013; the outermost rule selects
    # the package.  Prose JS literals fail json.loads and are skipped.
    best: tuple[int, int, dict[str, Any]] | None = None  # (start, end, obj)
    for m in re.finditer(r"\{", raw):
        start = m.start()
        depth = 0
        in_string = False
        escape = False
        end = -1
        for i in range(start, len(raw)):
            ch = raw[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end < 0:
            continue  # unbalanced to EOF — not a candidate
        try:
            parsed = json.loads(raw[start:end])
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        if (
            best is None
            or end > best[1]
            or (end == best[1] and start < best[0])
        ):
            best = (start, end, parsed)
    return best[2] if best is not None else None


def _merge_prefill_continuation(prefill: str, continuation: str) -> str:
    """Merge an assistant-role prefill with the model's streamed
    continuation for JSON reassembly.

    Naively, either the model continues cleanly (nothing needs
    prepending after checking it doesn't already contain the
    prefill) or it never re-emits the prefill at all (the whole
    prefill must be prepended).  In practice (discovered via a
    live-Ollama test against a real coding model), some models do
    neither cleanly: they partially re-echo a SUFFIX of the
    prefill before continuing with new content -- e.g. prefill
    ``'{\\n  "user_intent": "'`` and the model's continuation
    starting with ``'  "user_intent": "Rename...'`` (echoing
    everything except the leading ``'{\\n'``).  Naively
    concatenating ``prefill + continuation`` in that case produces
    a duplicated, invalid JSON fragment.

    This finds the longest suffix of ``prefill`` that matches a
    prefix of ``continuation`` and merges without duplicating that
    overlap, handling full re-emission, partial re-echo, and clean
    continuation with a single rule.
    """
    if not prefill:
        return continuation
    if continuation.startswith(prefill):
        return continuation
    max_overlap = min(len(prefill), len(continuation))
    for overlap_len in range(max_overlap, 0, -1):
        if prefill[-overlap_len:] == continuation[:overlap_len]:
            return prefill[:-overlap_len] + continuation
    return prefill + continuation


def _strip_prefill(prefill: str, text: str) -> str:
    """Strip a leading assistant prefill from generated text.

    The pruned specialists re-emit the assistant prefill verbatim
    (they ignore ``think: False``), so the raw streamed output often
    begins with the closed-think block + the harness's assistant
    prefill.  Harnesses that extract a specific object (code block,
    patch block, JSON) are unaffected because their contract pulls
    the object out of the middle.  But ``general_use`` returns the
    raw output, so the prefill would otherwise leak into the
    user-facing answer.

    Handles full re-emission, partial re-echo of a leading suffix,
    and the no-echo case (returns ``text`` unchanged).
    """
    if not prefill or not text:
        return text
    if text.startswith(prefill):
        return text[len(prefill):]
    # Partial re-emission: strip the longest leading run of ``prefill``
    # that the model actually echoed.
    max_match = min(len(prefill), len(text))
    for n in range(max_match, 0, -1):
        if prefill[:n] == text[:n]:
            return text[n:]
    return text

def _disable_response_prefix_for_images(package: CleanPackage) -> CleanPackage:
    """Keep images on native chat when raw continuation cannot carry them."""
    if package.response_prefix is None or not package.images:
        return package
    print(
        f"[prefix] multimodal response-prefix unsupported by Ollama raw path; "
        f"preserving {len(package.images)} image(s) via native chat without prefix "
        f"(model={package.model_id})",
        file=sys.stderr,
        flush=True,
    )
    return CleanPackage(**{**package.__dict__, "response_prefix": None})


def _with_seed_override(
    package: CleanPackage,
    seed_override: int | None,
) -> CleanPackage:
    if seed_override is None:
        return package
    return CleanPackage(**{
        **package.__dict__,
        "sampling": {**package.sampling, "seed": seed_override},
    })


# Soft import of the ``unidiff`` library for unified-diff
# application.  When the library is unavailable, T1a's diff
# application returns None and the runtime falls through to
# T1b cleanly.  The library is pure Python, ~50 KB, no
# subprocess, no external deps.  ``pip install unidiff``.
try:
    from unidiff import PatchSet
    from unidiff.errors import UnidiffParseError
    _HAS_UNIDIFF = True
except ImportError:  # pragma: no cover
    PatchSet = None  # type: ignore[assignment]
    UnidiffParseError = Exception  # type: ignore[assignment, misc]
    _HAS_UNIDIFF = False


def apply_unified_diff(base: str, diff_text: str) -> str | None:
    """Apply a unified diff to ``base`` and return the patched text.

    Returns ``None`` on any failure: library missing, diff
    unparseable, diff does not apply (context mismatch), or
    applied diff has no effect.  The runtime treats ``None``
    as "T1a failed, fall through to T1b".
    """
    if not _HAS_UNIDIFF or not base or not diff_text:
        return None
    try:
        patch = PatchSet(diff_text)
    except (UnidiffParseError, Exception):
        return None
    # Apply the patch line by line.  ``unidiff`` does not
    # apply patches to source text directly; we walk each
    # hunk and reconstruct the patched text by:
    #   1. Walking ``base`` line-by-line.
    #   2. For each hunk, replacing the original lines
    #      (source_start..source_start+source_length) with
    #      the hunk's patched lines.
    #   3. Concatenating.
    # If any hunk's context does not match, return None.
    base_lines = base.splitlines(keepends=True)
    if not base_lines:
        # Empty base: patch must consist of pure additions.
        base_lines = [""]
    # Build a sorted list of hunks per file.  We only handle
    # the first file in the patch (a single targeted edit
    # should be a single file).  Multi-file patches return
    # None (not supported for T1a).
    if len(patch) != 1:
        return None
    patched_file = patch[0]
    if patched_file.is_removed_file or patched_file.is_added_file:
        # T1a is an edit, not a creation/deletion.
        return None
    hunks = list(patched_file)
    if not hunks:
        return None
    out_lines: list[str] = []
    cursor = 0
    # 1-indexed line numbers in the patch hunk headers;
    # convert to 0-indexed for ``base_lines``.
    for hunk in hunks:
        src_start = hunk.source_start - 1
        src_len = hunk.source_length
        if src_start < 0 or src_start > len(base_lines):
            return None
        if src_start + src_len > len(base_lines):
            return None
        # Verify the source lines match the hunk context.
        # ``hunk.source`` is a list of strings; each line
        # starts with ' ' (context) or '-' (removed).
        # Strip the leading marker and compare to base.
        hunk_source_lines = [
            ln[1:].rstrip("\n") for ln in hunk.source
            if ln and ln[0] in (" ", "-")
        ]
        # Compare against base_lines[src_start:src_start+src_len].
        # Tolerate trailing-newline differences by stripping.
        actual = [
            ln.rstrip("\n") for ln in base_lines[src_start:src_start + src_len]
        ]
        if len(actual) != len(hunk_source_lines):
            return None
        for actual_line, hunk_line in zip(actual, hunk_source_lines):
            if actual_line != hunk_line:
                return None
        # Copy everything before this hunk.
        out_lines.extend(base_lines[cursor:src_start])
        # Add the patched lines from ``hunk.target``.
        # Each line in target starts with ' ' (unchanged
        # context) or '+' (added).  Strip the marker.
        for line in hunk.target:
            if not line:
                continue
            if line[0] == "+":
                out_lines.append(line[1:])
            elif line[0] == " ":
                out_lines.append(line[1:])
            # '-' lines are not emitted (they are removed).
        cursor = src_start + src_len
    # Copy any remaining base lines after the last hunk.
    out_lines.extend(base_lines[cursor:])
    patched = "".join(out_lines)
    if patched == base:
        # Empty diff: nothing changed.  Caller's contract
        # will re-fail against the original error.
        return None
    return patched


# v12 T1 primitive.  T1 is the error-correction step
# taken after T0 fails its functional check.  The runtime
# supports two strategies (declared in the harness YAML
# via ``error_correction.t1_strategy``):
#
#   ``full_only`` (default):
#     Single full re-emit.  T1 = one attempt with the
#     error trace in the user message.  T1a is skipped.
#
#   ``targeted_then_full``:
#     T1a (FREE) emits a unified diff against the T0
#     FINAL body.  Runtime applies the diff to T0's
#     FINAL body and runs the contract on the stitched
#     body.  If the contract passes, the stitched body
#     is the T1 answer.  If T1a fails for any reason
#     (no diff, diff doesn't apply, stitched body
#     fails the contract), fall through to T1b: a
#     full re-emit with the T0 error trace.  T1a's
#     failure is INTERNAL and does not count against
#     ``max_calls_per_instance``; only T1b counts.
#
# The T1 primitive is the harness's recipe; the runtime
# executes it.  No magic numbers: every threshold, every
# template string, every contract reference is YAML-driven.


def _fill_template(template: str, mapping: dict[str, str]) -> str:
    """Fill ``{key}`` placeholders in ``template`` with values from ``mapping``.

    Unknown placeholders are left intact.  Used for the
    T1 user-message templates declared in the YAML.
    """
    out = template
    for key, value in mapping.items():
        out = out.replace("{" + key + "}", str(value))
    return out


def _detect_fence_language(text: str) -> str:
    """Detect the fence language tag from model output.

    Searches for any known code fence tag in the text and returns
    the canonical name (e.g., ``"python"``, ``"cpp"``, ``"rust"``,
    ``"html"``, ``"json"``, ``"javascript"``, etc.).  Falls back
    to ``"python"`` when no tag is detected (backward compat).

    The detection order is:
      1. Look for ``<language>`` immediately before the opening
         triple-backtick (most common pattern).
      2. Fallback: scan for any known fence tag's first occurrence.
    """
    if not text:
        return "python"
    # Pattern 1: ```<language>\n — the most common prefill pattern.
    # Match the language word immediately before the opening backticks.
    m = re.search(r"```(\w+)\n", text)
    if m:
        lang = m.group(1).lower()
        # Normalize to canonical name via FENCE_ALIASES.
        # Build a reverse map: alias -> canonical_key.
        # The canonical key is the FIRST long-form name in each
        # entry's aliases list (e.g. "javascript" for ["javascript", "js"],
        # "typescript" for ["typescript", "ts"]).
        # If the lang IS a FENCE_ALIASES key and that key is also the
        # first alias (long form), return it directly (canonical).
        # If the lang is a FENCE_ALIASES key but NOT the first alias,
        # normalize to the first alias (e.g. "js" -> "javascript").
        # If the lang is in another key's aliases, return that key.
        if lang in FENCE_ALIASES:
            aliases = FENCE_ALIASES[lang]
            if aliases and aliases[0] == lang:
                # Long-form canonical key ("javascript", "python", "sh")
                return lang
            # Short alias key ("js", "ts") — normalize to the long form.
            return aliases[0] if aliases else lang
        # Then check aliases: find the canonical key whose aliases
        # list contains this language (long form wins).
        for canon, aliases in FENCE_ALIASES.items():
            if aliases and aliases[0] == canon and lang in aliases:
                return canon
        # Fallback: return the first key whose aliases contain lang.
        for canon, aliases in FENCE_ALIASES.items():
            if lang in aliases:
                return aliases[0] if aliases else canon
        return lang  # Return as-is if not in the alias map.
    # Pattern 2: any fence opening pattern with language tag.
    m2 = re.search(r"```[\s]*(\w+)", text)
    if m2:
        lang = m2.group(1).lower()
        for canon, aliases in FENCE_ALIASES.items():
            if lang in aliases or lang == canon:
                return canon
        return lang
    return "python"


def _strip_thinking_markup(text: str) -> str:
    """Remove complete explicit thinking blocks from visible output."""
    return re.sub(
        r"<think(?:ing)?\b[^>]*>.*?</think(?:ing)?>",
        "",
        text or "",
        flags=re.IGNORECASE | re.DOTALL,
    )


def _extract_general_use_work_product(text: str) -> tuple[str, str]:
    """Return the last closed code fence, or the full response as Markdown."""
    visible = _strip_thinking_markup(text).strip()
    if not visible:
        return "", "markdown"
    opening = re.compile(r"```[ \t]*([A-Za-z0-9_+#.-]*)[ \t]*\r?\n")
    last_match: re.Match[str] | None = None
    last_close: re.Match[str] | None = None
    for match in opening.finditer(visible):
        close = _FENCE_CLOSE_RE.search(visible, match.end())
        if close is not None:
            last_match = match
            last_close = close
    if last_match is None or last_close is None:
        return visible, "markdown"
    tag = last_match.group(1).lower()
    if not tag:
        language = "markdown"
    else:
        language = tag
        for aliases in FENCE_ALIASES.values():
            if aliases and tag in aliases:
                language = aliases[0]
                break
    return visible[last_match.end():last_close.start()].strip(), language


def _extract_final_code_block(
    contract_name: str | None,
    output: str,
    language: str | None = None,
    runtime_cfg: dict[str, Any] | None = None,
) -> str | None:
    """Extract the FINAL body for the contract's expected block format.

    For ``python_new_code``, this is the LAST code fence
    body in the declared language.  For ``unified_diff_patch``,
    this is the LAST ``<patch>...</patch>`` block.  Generic
    fallback tries the language-specific fence first, then
    patch.

    Returns ``None`` if the expected block is not present.
    Used by the T1 primitive (and the future T1a fast-path)
    to extract the artifact the contract will check.
    """
    if contract_name == "python_new_code":
        return extract_last_fence_body(output, language, runtime_cfg)
    if contract_name == "web_new_code":
        return extract_last_fence_body(output, language, runtime_cfg)
    if contract_name == "compiled_coding_new_code":
        return extract_last_fence_body(output, language, runtime_cfg)
    if contract_name == "unified_diff_patch":
        return extract_last_patch_block(output)
    body = extract_last_fence_body(output, language, runtime_cfg)
    if body is not None:
        return body
    return extract_last_patch_block(output)


def _resolve_t1_strategy(runtime_cfg: dict[str, Any]) -> str:
    """Return the T1 strategy declared in the harness YAML.

    Falls back to ``"full_only"`` if the YAML omits the
    field or declares an unknown value.  Valid values:
    ``"full_only"`` and ``"targeted_then_full"``.

    In step 4a, ``targeted_then_full`` is accepted but
    behaves identically to ``full_only``: the T1a
    fast-path is not yet wired into the main loop.  The
    strategy is read so the YAML can declare the intent;
    step 4b will implement the T1a streaming.
    """
    strategy = _resolve_cfg(runtime_cfg, "error_correction.t1_strategy") or "full_only"
    if strategy not in ("full_only", "targeted_then_full"):
        return "full_only"
    return str(strategy)


def _compose_t1b_prompt(
    harness: Any,
    t0_output: str,
    t0_error: str,
    unit: CleanUnit,
) -> CleanPackage:
    """Compose the T1b (full re-emit) prompt package.

    T1b is the harness's normal T1 package (closed-think
    + assistant prefill + the error trace in the user
    message via the standard T1 prior_error path) with
    the ``t1_full_user_modifier`` template appended to
    the user message.  The harness YAML declares the
    template; the runtime fills the ``{prior_error}``
    placeholder and appends.

    Used by the main inner loop's T1 iteration.  When
    ``t1_strategy: full_only``, T1b is the only T1
    emission.  When ``t1_strategy: targeted_then_full``,
    T1b runs only if the T1a fast-path (step 4b, not
    yet wired) failed.
    """
    # The runtime config lives on the harness spec nested
    # in the resolved work (``harness.work.harness.runtime``),
    # NOT on the CleanHarness composer instance.  The
    # composer (``CleanHarness``) has ``.work`` (a
    # ``ResolvedWork``) whose ``.harness`` is the
    # ``CleanHarnessSpec`` dataclass carrying ``.runtime``.
    runtime_cfg = harness.work.harness.runtime
    template = _resolve_cfg(
        runtime_cfg, "error_correction.t1_full_user_modifier"
    ) or ""
    modifier_text = _fill_template(str(template), {"prior_error": t0_error})
    # Build the normal T1 package with prior_error set.
    package = harness.compose(
        unit, "T1", CleanTierContext(tier="T1", prior_error=t0_error)
    )
    # Append the modifier to the user message.
    new_messages: list[dict[str, str]] = []
    for msg in package.messages:
        if msg.get("role") == "user":
            new_messages.append({
                "role": "user",
                "content": (msg.get("content") or "") + "\n\n" + modifier_text,
            })
        else:
            new_messages.append(msg)
    return CleanPackage(**{**package.__dict__, "messages": new_messages})


def _now() -> float:
    return time.monotonic()


def _emit(progress: CleanProgressCallback, event: CleanProgressEvent) -> CleanControlDecision | None:
    """Call the callback and interpret the return value.

    Sync callbacks may return ``None`` or a ``CleanControlDecision``.
    Async callbacks are awaited.  A non-decision return is treated as
    ``continue`` for ``checkpoint`` events.
    """
    result = progress.on_event(event)
    if hasattr(result, "__await__"):
        # Coroutine returned: the caller is async; we cannot await here
        # because _emit is sync.  The async emitter in the runtime
        # handles the await via ``_emit_async``.
        return None
    if result is None:
        return None
    return result


async def _emit_async(progress: CleanProgressCallback, event: CleanProgressEvent) -> CleanControlDecision:
    """Await the callback if it returns a coroutine; otherwise dispatch."""
    result = progress.on_event(event)
    if hasattr(result, "__await__"):
        result = await result
    if result is None:
        return CleanControlDecision.CONTINUE if event.event_type == "checkpoint" else CleanControlDecision.CONTINUE
    if isinstance(result, CleanControlDecision):
        return result
    return CleanControlDecision.CONTINUE


def _summary_preview(text: str, limit: int = 240) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


async def _invoke_meta_target(
    *,
    target: str,
    tier_label: str,
    result: "CleanExecutionResult",
    work: Any,
    harness: Any,
    transports: dict[str, Any],
    vram: Any,
    unit: CleanUnit,
    generalist_model: str,
    generalist_provider: str,
    emit: Any,
    run_start: float,
    cloud_effort: str | None = None,
    cloud_max_tokens: int | None = None,
    cloud_silence_timeout_s: float | None = None,
    cloud_total_timeout_s: float | None = None,
    cloud_guard_cfg: dict[str, Any] | None = None,
    checkpoint_interval_s: float = DEFAULT_CHECKPOINT_INTERVAL_S,
    raw_token_writer: Any | None = None,
    artifact_store: Any | None = None,
    artifact_id: str | None = None,
    evaluate_store: Any | None = None,
    evaluate_prior_wp: bool = False,
    user_intent: str = "",
    source_filename: str | None = None,
) -> None:
    """Dispatch one meta-harness escalation target.

    The two targets have DIFFERENT design principles, matched to what
    each engine needs:

    * ``generalist_zero_shot`` (T5, local generalist): full local
      compose + NOTICE -- the local ladder's compensation machinery
      context applies here.
    * ``cloud_sota`` (T3): "the local inference engine failed, give it
      to a SOTA cloud model for its zero-shot effort".  NO local
      compensation scaffolding (no DRAFT/FINAL shaping, no think
      scaffolding, no fallback-apology NOTICE).  The prompt is the
      original task; SOTA models do not need handholding.  Only the
      ARTIFACT-EXTRACTION infrastructure carries over so a follow-up
      turn can evaluate/modify the cloud WP.
    """
    if generalist_provider not in transports:
        return
    if target == "cloud_sota":
        await _invoke_cloud_sota(
            result=result, work=work, harness=harness,
            transports=transports, unit=unit,
            cloud_model=generalist_model,
            cloud_provider=generalist_provider,
            emit=emit, run_start=run_start,
            cloud_effort=cloud_effort,
            cloud_max_tokens=cloud_max_tokens,
            cloud_silence_timeout_s=cloud_silence_timeout_s,
            cloud_total_timeout_s=cloud_total_timeout_s,
            cloud_guard_cfg=cloud_guard_cfg,
            checkpoint_interval_s=checkpoint_interval_s,
            raw_token_writer=raw_token_writer,
            artifact_store=artifact_store,
            artifact_id=artifact_id,
            evaluate_store=evaluate_store,
            evaluate_prior_wp=evaluate_prior_wp,
            user_intent=user_intent,
            source_filename=source_filename,
        )
    else:
        await _invoke_t5_body(
            result=result, work=work, harness=harness,
            transports=transports, unit=unit,
            generalist_model=generalist_model,
            generalist_provider=generalist_provider,
            emit=emit, run_start=run_start,
        )


async def _invoke_cloud_sota(
    *,
    result: "CleanExecutionResult",
    work: Any,
    harness: Any,
    transports: dict[str, Any],
    unit: CleanUnit,
    cloud_model: str,
    cloud_provider: str,
    emit: Any,
    run_start: float,
    cloud_effort: str | None = None,
    cloud_max_tokens: int | None = None,
    cloud_silence_timeout_s: float | None = None,
    cloud_total_timeout_s: float | None = None,
    cloud_guard_cfg: dict[str, Any] | None = None,
    checkpoint_interval_s: float = DEFAULT_CHECKPOINT_INTERVAL_S,
    raw_token_writer: Any | None = None,
    artifact_store: Any | None = None,
    artifact_id: str | None = None,
    evaluate_store: Any | None = None,
    evaluate_prior_wp: bool = False,
    user_intent: str = "",
    source_filename: str | None = None,
) -> None:
    """T3: SOTA cloud zero-shot after local ladder exhaustion.

    Design principle: the local inference engine FAILED, so hand the
    ORIGINAL problem to a SOTA cloud model for its zero-shot effort.

    Prompt composition for T3 is INTENTIONALLY different from the local
    tiers, matched to what each engine needs:

    * Carried over (interface contract, not compensation): the domain
      persona and an output-format instruction demanding a single fenced
      code block -- the extraction-compatible surface the artifact
      pipeline consumes.
    * NOT carried over (pruned-model compensation): DRAFT/FINAL
      two-block shaping, think-state scaffolding, prior-error grounding,
      fence-anchor prefill, dither directives.  A SOTA zero-shot does
      not need the pruned-model handholding, and the one-block output
      makes the fence extraction deterministic.

    Artifact-extraction infrastructure (fence scan, language detect,
    artifact store commit) carries over so follow-up evaluate/modify
    turns work on the cloud WP.
    """
    persona = getattr(work, "persona", "") or "You are an expert programmer."
    # 2026-08-31: harness-aware cloud prompt.  The cloud path now selects
    # the system_suffix/output_instruction by work.harness.harness_id from
    # cloud.prompts.<harness> (validated concise contract for that harness),
    # falling back to the legacy global system_suffix for code-gen (and for
    # any harness without an explicit prompts entry).
    _cg_prompt = (cloud_guard_cfg or {})
    _harness_obj = getattr(work, "harness", None)
    _harness_id = getattr(_harness_obj, "harness_id", None) or ""
    # ================================================================
    # 2026-09-16 (M2): CAPABILITY PROPERTIES, NOT HARNESS NAMES.
    # ================================================================
    # The three values below were `harness_id == "modify_artifact"` string
    # tests.  A name test hard-codes behaviour to ONE harness, so a second
    # modify harness could not exist without inheriting this one's config)
    # -- the root cause of defects D1-D4 (PLAN_new_harness.md section 0.1).
    #
    # Expressed as properties, `modify_harness` can declare
    # local_change_order=False and thereby skip every LOCAL change-order gate
    # WITHOUT needing the gates to be re-listed and neutralised.
    #
    # ================================================================
    # CAPABILITY PROPERTIES, NOT HARNESS NAMES.
    # ================================================================
    # The values below were `harness_id == "modify_artifact"` string tests.
    # A name test hard-codes behaviour to ONE harness, which is why a second
    # modify harness could not exist without inheriting this one's config.
    #
    # TWO LAYERS, TWO FLAGS.  "Emits a change order" is true of DIFFERENT
    # layers for different routes:
    #
    #                        local tier        shared T3 cloud layer
    #   PATH A               denovo            change order
    #   PATH B               (none)            change order
    #   modify_harness_legacy  change order    change order
    #
    # PATH A's local tier is denovo while the cloud layer it ESCALATES into
    # returns a change order, so one boolean cannot describe it.  This block
    # is the shared cloud path, so it reads cloud_change_order.
    #
    # getattr(..., "...", False) is deliberate: these are NEW fields
    # (resolution.py).  A cached/older CleanHarnessSpec instance -- or a spec
    # built by anything not yet updated -- must degrade to the conservative
    # False rather than raise AttributeError inside the live cloud path.
    # ================================================================
    _needs_t3_context = bool(getattr(_harness_obj, "needs_t3_context", False))
    _cloud_change_order = bool(getattr(_harness_obj, "cloud_change_order", False))
    _needs_artifact_store = bool(getattr(_harness_obj, "needs_artifact_store", False))
    _prompts = _cg_prompt.get("prompts") or {}
    _hp = _prompts.get(_harness_id) or {}
    if not _hp and _harness_id == "general_use":
        _hp = DEFAULT_DIRECT_CLOUD_CONFIG["prompts"]["general_use"]
    if _hp:
        t3_suffix = str(_hp.get("system_suffix", ""))
        t3_output = str(_hp.get("output_instruction", ""))
    else:
        # Fall back to global (legacy) cloud system_suffix / output_instruction.
        t3_suffix = str(_cg_prompt.get(
            "system_suffix",
            "You are a competent python language programmer. Respond with exactly "
            "TWO code blocks in the target language and nothing else. First: a "
            "DRAFT block where you reason through the problem using inline "
            "comments. Second: the FINAL clean solution with no prose. Complete "
            "the program exactly as specified by the prompt.",
        ))
        t3_output = str(_cg_prompt.get(
            "output_instruction",
            "Output exactly two code blocks in the target language: a DRAFT "
            "(with inline comments) and a FINAL (clean, no prose). The grader "
            "extracts and checks the LAST block.",
        ))
    t3_system = (
        f"{persona}\n"
        f"{t3_suffix}\n"
        f"{t3_output}"
    )
    # Artifact-aware cloud requests must receive the same source object as
    # the local path.  An explicit /evaluate request takes precedence over
    # artifact_store.current because it intentionally evaluates the prior
    # turn's WP held in evaluate_store.
    _cloud_user = unit.t3_context_text or unit.prompt
    # 2026-09-28 (user ruling): NO LONGER gated on
    # ``_harness_id == "general_use"``.  ``context_text`` now carries the
    # attached text / working-artifact block for ALL harnesses; gating it here
    # would silently DROP every attachment on the T3 cloud path now that the
    # image harness swap is gone.  De-gated in tandem with resolution.py's
    # local-composition site and server.py's ``context_text`` assignment.
    #
    # Reversal: restore ``_harness_id == "general_use" and`` at this site,
    # at resolution.py's local-composition site, and at server.py's
    # ``context_text`` assignment.
    if not unit.t3_context_text and unit.context_text:
        _cloud_user = (
            unit.context_text
            + "\n\n[AUTHORITATIVE USER REQUEST]\n"
            + unit.prompt
        )
    # Gate 10 (needs_t3_context).  `modify_artifact` and `evaluate_artifact`
    # both want the artifact/outline in the T3 prompt; the property states
    # the requirement without naming either harness.
    if not unit.t3_context_text and _needs_t3_context:
        try:
            _source_content: str | None = None
            _source_context = ""
            if evaluate_prior_wp and evaluate_store is not None and evaluate_store.has():
                _pen = evaluate_store.get() or {}
                _source_content = _pen.get("source_content") or ""
                if _source_content:
                    _source_context = (
                        f"[ARTIFACT CONTENT]\n{_source_content}\n\n"
                        "[END ARTIFACT CONTEXT]\n\n"
                    )
            elif artifact_store is not None:
                _target = artifact_store.get(artifact_id)
                if _target is not None:
                    _outline = artifact_store.build_outline(_target.artifact_id)
                    # 2026-09-08: BM25 index dropped from the cloud
                    # payload (user-ruled de-legacization).  The outline
                    # carries the section IDs; content + outline is the
                    # complete grounding a SOTA model needs.
                    _source_context = (
                        f"[ARTIFACT CONTENT]\n{_target.content}\n\n"
                        f"[ARTIFACT STRUCTURE — tree-sitter outline]\n{_outline}\n"
                        "[END ARTIFACT CONTEXT]\n\n"
                    )
            if _source_context:
                _cloud_user = (
                    _source_context
                    + "[AUTHORITATIVE USER REQUEST]\n"
                    + unit.prompt
                )
        except Exception as _exc:
            # Context construction must never prevent the cloud attempt;
            # the bare user prompt remains available as a fallback.
            print(
                f"[T3] artifact-context injection failed for {_harness_id}: {_exc}",
                file=sys.stderr,
                flush=True,
            )
    # 2026-09-08: [LOCAL CONTRACT STATUS] block REMOVED per the
    # de-legacized cloud design — the cloud request is the original
    # problem + full context + the change-order output contract, with
    # no framing about local R1/R2 rounds.
    messages = [
        {"role": "system", "content": t3_system},
        {"role": "user", "content": _cloud_user},
    ]
    package = CleanPackage(
        model_id=cloud_model,
        provider=cloud_provider,
        messages=messages,
        # D3 (2026-08-29): explicit cloud sampling contract.  Previously
        # options={} meant NO max_tokens was sent -- the cloud model ran
        # on its server-side default, outside the CoE contract (likely
        # why the arc195_c T3 stream ended unnaturally mid-reasoning).
        # think carries the config's reasoning-effort intent; the cloud
        # transport maps it to reasoning_effort.
        options={} if cloud_max_tokens is None else {"num_predict": int(cloud_max_tokens)},
        sampling={},
        # 2026-09-09 (user ruling): cloud reasoning effort is a separate
        # parameter from the local think policy. The CleanPackage carries
        # it in .think as a STRING level ("low"|"medium"|"high") which
        # the cloud transport maps 1:1 to reasoning_effort. Booleans no
        # longer flow here — local think policy NEVER becomes cloud effort.
        think=cloud_effort,
        images=list(unit.images or []),
    )
    started = _now()
    output_parts: list[str] = []
    visible_output_parts: list[str] = []
    error: str | None = None
    # D4 (2026-08-29): in-attempt silence timeout.  The OUTER local
    # SSE idle watchdog (210s, cloud tiers exempt -- see server.py)
    # cancels the whole coroutine WITHOUT recording the T3
    # attempt (no CleanAttempt row, no [T3] log) -- that destroyed the
    # evidence trail in the arc195_c T3 post-mortem.  Enforce the
    # dead-air limit HERE instead, so a silent cloud stream becomes a
    # properly recorded failed attempt that still returns to the meta
    # loop.  Default 600s (cloud may legitimately queue/cold-start far
    # longer than a local model).  Implementation: wrap the chunk
    # iterator in a generator that raises CloudStreamSilent when no
    # chunk arrives within the window -- asyncio.wait_for then cancels
    # ONLY this attempt, preserving partial output for the attempt row.
    async def _silence_guarded() -> AsyncIterator[tuple[str, bool]]:
        internal = transport.stream(package, pathology_monitor=None)
        aiter = internal.__aiter__()
        # Total-request deadline: cap the whole cloud attempt (not just
        # a silent gap) so a runaway cloud request cannot burn past
        # cloud_total_timeout_s of wall time.  None = unlimited.
        _total_ms = None
        if cloud_total_timeout_s is not None and cloud_total_timeout_s > 0:
            _total_ms = cloud_total_timeout_s
        _deadline_at = time.monotonic() + (
            _total_ms if _total_ms else float("inf")
        )
        # Silence window: None = no dead-air limit (rely on total or the
        # provider).  Default per-config (600s) applied in the caller.
        _silence_ms = cloud_silence_timeout_s
        while True:
            _remaining = _deadline_at - time.monotonic()
            if _remaining <= 0:
                raise CloudStreamSilent(
                    f"total stream duration exceeded (total_timeout limit hit)"
                )
            _wait = _remaining
            if _silence_ms:
                _wait = min(_wait, _silence_ms)
            try:
                chunk_pair = await asyncio.wait_for(
                    aiter.__anext__(), timeout=_wait,
                )
            except asyncio.TimeoutError:
                _remaining = _deadline_at - time.monotonic()
                if _remaining <= 0:
                    raise CloudStreamSilent(
                        "total stream duration exceeded (total_timeout limit hit)"
                    )
                if _silence_ms:
                    raise CloudStreamSilent(
                        f"no stream activity for {_silence_ms:.0f}s (silence_timeout_s)"
                    )
                raise CloudStreamSilent(
                    "no stream activity within silence window"
                )
            except StopAsyncIteration:
                # D4-ha (2026-08-29): normal end of the cloud stream. The
                # transport generator returns when it sees [DONE]; without
                # this handler the exhaustion surfaces as
                # "async generator raised StopAsyncIteration" and a fully
                # successful cloud response was discarded (t3_error).
                return
            yield chunk_pair
    # 2026-08-30: init cloud telemetry counters before the stream so they
    # are always bound (the except paths below and the [T3] log both read
    # them).  Filled in by the guard block after the loop.
    _cloud_n_dither = 0
    _cloud_was_guarded = False
    # 2026-08-30: persist the CLOUD T3 token stream to the raw writer so
    # dither-density-over-time is computable for the cloud tier too (the
    # local tiers already write via raw_token_writer; the cloud path did
    # NOT, which silently dropped the most important telemetry).  Write a
    # T3 tier header, then every chunk with its timestamp.
    if raw_token_writer is not None:
        try:
            raw_token_writer.begin_tier("T3", cloud_model)
        except Exception:
            pass
    try:
        transport = transports[cloud_provider]
        # 2026-08-30: CLOUD-SPECIFIC guard config.  The cloud gets the
        # hard problems, and its dither/loop behavior may differ from a
        # pruned local MoE, so its thresholds live under
        # `meta_harness_ladder.cloud.cloud_guards` (separate from the
        # local `primitives`/`dither_limit`/`loop_guard_*` keys).  Build
        # a config-style dict the dither_guard/repetition_loop_guard
        # helpers read, then only run a guard if its "enabled" flag is set.
        _cg = (cloud_guard_cfg or {}).get("cloud_guards") or {}
        # Build the config dict, DROPPING any None/missing values so the
        # dither_guard / repetition_loop_guard helpers fall back to their
        # OWN defaults for omitted keys (passing None explicitly would
        # make dict.get return None instead of the default, which would
        # break e.g. a "None" dither_pattern regex).
        _cloud_guard_config = {
            k: v for k, v in {
                "dither_pattern": _cg.get("dither_pattern"),
                "dither_limit": _cg.get("dither_limit"),
                "dither_limit_think_scale": _cg.get("dither_limit_think_scale"),
                "loop_guard_min": _cg.get("loop_guard_min"),
                "loop_guard_skip": _cg.get("loop_guard_skip"),
                "loop_guard_match": _cg.get("loop_guard_match"),
                "loop_guard_window": _cg.get("loop_guard_window"),
            }.items() if v is not None
        }
        _dither_on = bool(_cg.get("dither_enabled", True))
        _loop_on = bool(_cg.get("loop_enabled", True))
        # 2026-08-31: SHORTSTOP rule.  Evaluate the trailing shortstop_window
        # chars of the cloud stream once its length exceeds shortstop_min_chars;
        # if the dither-marker DENSITY in that window exceeds shortstop_rate, cut
        # the cloud request immediately (it is dithering without committing).
        # Units are markers per CHAR of the trailing window -- NOT per 10k.
        #
        # 2026-09-18 (user ruling): 2.0 -> 100.0, and the redundant `* 1000`
        # divisor removed so the unit is honest.  2.0 was calibrated against
        # glm-5.3-flash (T=2.0, P=30000 -> 100% precision, 67% recall on 20
        # held-out hard qids, 0 passing qids cut).  The shipped cloud model is
        # now deepseek-v4.1-flash:cloud, whose reasoning trace uses the markers
        # as ordinary analytic connectives, so a legitimate 44,423-char
        # change-order attempt was cut at 2.34.  100.0 is a temporary
        # DATA-GATHERING default that makes the rule effectively non-tripping.
        # It is NOT a recalibration.  All five cloud-bearing harnesses carry the
        # same value deliberately.
        # RESET: README.md -> "Cloud dither guard recalibration".
        # Disabled when shortstop_enabled=0.
        _shortstop_on = bool(_cg.get("shortstop_enabled", False))
        _shortstop_rate = float(_cg.get("shortstop_rate", 100.0))
        _shortstop_min = int(_cg.get("shortstop_min_chars", 30000))
        _shortstop_win = int(_cg.get("shortstop_window", 10000))
        _cloud_n_dither = 0
        _cloud_shortstopped = False
        # rolling window buffer for the shortstop check (kept in chars)
        _ss_buf: list[str] = []
        _ss_buf_len = 0
        # 2026-09-19 (BUGFIX -- live failure): the cloud progress checkpoint
        # needs its own cadence clock, and it MUST be initialized HERE, before
        # the stream loop.
        #
        # The first version of the cloud checkpoint (PLAN_progress_mechanism.md
        # phase 8) read `last_checkpoint_at` and assigned it further down, inside
        # the same function.  That assignment makes the name a LOCAL of
        # `_invoke_cloud_sota`, so the earlier READ raised
        #   UnboundLocalError: cannot access local variable 'last_checkpoint_at'
        #   where it is not associated with a value
        # The read sits inside `if len(output_parts) % 16 == 0:`, so it fired on
        # the FIRST 16-chunk boundary of every cloud call and was caught by the
        # enclosing `except Exception` as `t3_error: ...`.  Result: the cloud
        # stream was truncated ~100 chars in (mid-thinking), the tier reported
        # "not passed", and NO change order was ever produced.
        #
        # NOTE: the ladder's `last_checkpoint_at` at the execute_runtime_plan
        # level is a DIFFERENT variable and is NOT visible here -- the two loops
        # are independent.  Initialising a function-local copy is the fix.
        last_checkpoint_at = run_start
        async for chunk, is_thinking in _silence_guarded():
            output_parts.append(chunk)
            if not is_thinking:
                visible_output_parts.append(chunk)
            # 2026-08-30: persist the cloud T3 token to the raw writer
            # (same as the local tiers) so dither-density-over-time is
            # computable for the cloud tier.  Errors swallowed so a disk
            # hiccup never breaks the cloud inference.
            if raw_token_writer is not None:
                try:
                    await raw_token_writer.write_chunk(chunk)
                except Exception:
                    pass
            await emit(CleanProgressEvent(
                event_type="token",
                # Cloud layer: T3_R2 marks "the cloud returned a change order".
                tier="T3_R2" if _cloud_change_order else "T3",
                text=chunk,
                elapsed_s=_now() - run_start, model_id=cloud_model,
                metadata={"is_thinking": is_thinking, "tier": "T3"},
            ))
            # 2026-08-31: live shortstop check.  Maintain a rolling char
            # buffer; once the total stream length exceeds
            # shortstop_min + shortstop_window, evaluate the most recent
            # shortstop_window chars (whose start is >= shortstop_min).
            # If the dither-marker DENSITY in that window (markers per CHAR)
            # exceeds shortstop_rate, cut the request (error="shortstop").
            if _shortstop_on and chunk:
                _ss_buf.append(chunk)
                _ss_buf_len += len(chunk)
                if _ss_buf_len >= _shortstop_min + _shortstop_win:
                    _win_start = _ss_buf_len - _shortstop_win
                    if _win_start >= _shortstop_min:
                        _win_text = "".join(_ss_buf)[-_shortstop_win:]
                        _n_markers = len(re.findall(
                            _cloud_guard_config.get(
                                "dither_pattern",
                                r"(?i)\b(?:wait|actually|hmm|sorry|instead|reconsider(?:ing|s|ed)?|re-?examin(?:e|ing|ed)|but\s+wait|on\s+(?:second\s+)?thought|let\s+me\s+(?:re-?think|re-?do|try\s+again|reconsider))\b",
                            ),
                            _win_text,
                        ))
                        _rate = _n_markers / float(_shortstop_win or 1)
                        if _rate > _shortstop_rate:
                            _cloud_shortstopped = True
                            error = "shortstop"
                            break
            # 2026-08-30: dither/loop guards on the CLOUD stream.  The
            # only requests that reach T3 are the hard ones the local
            # ladder couldn't crack, so the cloud may dither/loop just
            # like a local model.  Use the CLOUD-SPECIFIC thresholds from
            # _cloud_guard_config (separate from local pruned-model keys).
            # Run every 16 chunks to bound CPU cost.  On a trip we set
            # error and drop the iterator (ending the attempt) so the
            # T3 attempt fails cleanly instead of burning unbounded
            # time/tokens.
            if len(output_parts) % 16 == 0:
                _ctext = "".join(output_parts)
                # 2026-09-19 (PLAN_progress_mechanism.md phase 8): PERIODIC
                # CHECKPOINT on the cloud stream.
                #
                # Why this site: it is the ONLY place in the cloud path that
                # runs regularly DURING generation -- and it is where the
                # chunk-driven guards already are.  There is no time-based hook
                # during cloud SILENCE: `_silence_guarded` awaits __anext__
                # under asyncio.wait_for, so on timeout wait_for has already
                # cancelled the iterator and the generator must raise.  That
                # limitation is shared with the LOCAL tiers (the ladder's
                # checkpoint site is equally chunk-driven), so this adds no new
                # class of limitation.
                #
                # Tolerance for ANY cloud model: the check reads only
                # len(output_parts) -- a LOCAL counter -- and calls the generic
                # emit().  It never inspects model output, assumes no provider
                # and assumes no streaming shape.  A provider that delivers ONE
                # chunk at the end never enters this block before completing, so
                # there is nothing to interrupt and the check is a no-op.
                _cp_due = (_now() - last_checkpoint_at) >= checkpoint_interval_s
                if _cp_due:
                    _cp_decision = await emit(CleanProgressEvent(
                        event_type="checkpoint", tier="T3",
                        model_id=cloud_model,
                        text_so_far=_summary_preview(_ctext),
                        elapsed_s=_now() - run_start,
                        metadata={"text_length": len(_ctext)},
                    ))
                    last_checkpoint_at = _now()
                    if _cp_decision == CleanControlDecision.ABORT:
                        # A user cancel.  Recorded with its own error string so
                        # it is NOT mistaken for a guard trip or a transport
                        # fault.  Deliberately does NOT start with
                        # "cloud_silence_timeout:" or "t3_error:", so the D4-hb'
                        # substantial-output downgrade below cannot rewrite an
                        # explicit cancel into a "delivered" success.
                        error = "cancelled"
                        break
                    if _cp_decision == CleanControlDecision.ACCEPT_PARTIAL:
                        # Cloud has no partial-accept contract; treat as a
                        # normal stop and keep whatever the stream produced.
                        error = "cancelled_partial"
                        break
                _gd = dither_guard(_ctext, _cloud_guard_config, think_on=True) if _dither_on else None
                _gl = repetition_loop_guard(_ctext, _cloud_guard_config) if _loop_on else None
                if _gd or _gl:
                    error = _gd or _gl
                    break
        output = "".join(output_parts)
        # 2026-08-30: persist the dither marker count so post-hoc dither
        # DENSITY (dither events / chars or tokens) is computable.  The
        # cloud raw tokens are NOT written to disk, so without this count
        # the density signal would be unavailable for tuning the cloud
        # dither threshold.
        if _dither_on:
            _dpat = _cloud_guard_config.get(
                "dither_pattern",
                r"(?i)\b(?:wait|actually|hmm|sorry|instead|reconsider(?:ing|s|ed)?|re-?examin(?:e|ing|ed)|but\s+wait|on\s+(?:second\s+)?thought|let\s+me\s+(?:re-?think|re-?do|try\s+again|reconsider))\b",
            )
            _cloud_n_dither = len(re.findall(_dpat, output))
        _cloud_was_guarded = error in ("dither", "loop", "shortstop")
    except CloudStreamSilent as exc:
        # D4: dead-air on the cloud stream. If we already collected a
        # substantial answer, it still counts as delivered (handled in
        # the D4-hb' downgrade below); otherwise it is a real failure.
        error = f"cloud_silence_timeout: {exc}"
        output = "".join(output_parts)
    except Exception as exc:
        error = f"t3_error: {exc}"
        output = "".join(output_parts)
    visible_output = "".join(visible_output_parts)
    # D4-hb' (2026-08-29): completion success is determined by THE TOKEN
    # STREAM, never by artifact extraction. A failed fence extraction is a
    # metadata note (downstream view), not an error -- the CoE-coder is a
    # quasi-stateless task engine: the raw response is the durable artifact,
    # extraction is a view on it. Only a transport-level failure with NO
    # substantial output is a real T3 failure (hard_stop territory).
    SUBSTANTIAL_OUTPUT_CHARS = 500
    _transport_error = error is not None and (
        error.startswith("cloud_silence_timeout:")
        or error.startswith("t3_error:")
    )
    if _transport_error and len(output.strip()) >= SUBSTANTIAL_OUTPUT_CHARS:
        # The stream produced a real answer before/around the plumbing
        # error (e.g. a transport hiccup mid-stream): treat the response
        # as delivered, degrade the error to an informational note.
        error_note = error
        error = None
        syntax_note = f"t3_stream_note: {error_note}"
    if _harness_id == "general_use":
        output = _strip_thinking_markup(output)
        visible_output = _strip_thinking_markup(visible_output)
    else:
        visible_output = output
    passed = error is None and bool(output.strip())
    # Artifact extraction carry-over: reuse the same fence-scan +
    # language-detect utilities the contracts use.  For bug_fix the
    # artifact is the <patch>...</patch> block (harness-aware extraction);
    # for code-gen it's the last fenced code block.  A syntax failure or a
    # no-fence response is recorded INFORMATIONALLY (log string in the
    # attempt error slot is NOT set -- extraction failure NEVER blocks the
    # completion path).
    extracted: str | None = None
    extracted_lang: str | None = None
    pending_edit: dict[str, Any] | None = None
    syntax_note: str | None = None
    if output.strip() and error is None:
        if _harness_id == "bug_fix":
            # bug_fix: extract the FIRST <patch>...</patch> block.
            m = re.search(r"<patch>\s*(.*?)\s*</patch>", output, re.DOTALL)
            if m:
                extracted = m.group(1).strip()
                extracted_lang = "python" if "python" in output else "diff"
                # validate diff structure (informational)
                if not all(k in extracted for k in ("--- a/", "+++ b/", "@@ ")):
                    syntax_note = "malformed_patch (informational)"
            else:
                syntax_note = "no_patch (informational)"
        elif _cloud_change_order:
            # Gate 4: change-order contract dispatch.  The cloud emits the same
            # R2 JSON edit package consumed by ArtifactStore.apply_edits().
            # Validate and preview it here, but defer the real fork until
            # server.py runs after the session turn has frozen.
            #
            # cloud_change_order, NOT local_change_order: this is the SHARED
            # cloud layer, which returns a change order for PATH B and for
            # PATH A's escalation alike.
            json_body = _extract_json(output)
            if json_body is not None:
                target = artifact_store.get(artifact_id) if artifact_store is not None else None
                if artifact_store is None:
                    error = "no_artifact_store: modify_artifact requires an artifact_store"
                elif target is None:
                    error = "no_current_artifact: no artifact selected for modify_artifact"
                else:
                    # 2026-09-08: empty change order at T3 is a FAILURE
                    # (no_change_package).  The cloud fires only after the
                    # local tiers failed to produce a valid change order;
                    # a zero-action package there contradicts the premise
                    # of escalation and previously forked an unchanged
                    # artifact reporting success — the exact "did not
                    # modify the file" symptom.  Local R2 keeps treating
                    # empty arrays as contract-valid.
                    _has_changes = bool(
                        json_body.get("replacements")
                        or json_body.get("insertions")
                        or json_body.get("removals")
                    )
                    if not _has_changes:
                        error = (
                            "no_change_package: cloud returned an empty change "
                            "order (no replacements, insertions, or removals)"
                        )
                    else:
                        # 2026-09-11: resolve the cloud's lenient guard
                        # profile from the cloud_guard_cfg (the harness
                        # runtime config is not in scope here; the cloud
                        # handler carries its own config). Fall back to the
                        # plain cloud defaults if absent.
                        _cloud_guard_cfg_for_modify = {
                            "enabled": True,
                            "cloud": (cloud_guard_cfg or {}).get(
                                "invented_replacement_guard", {}
                            ).get("cloud", {}) or {
                                "min_size_ratio": 0.20,
                                "min_token_overlap": 0.10,
                            },
                        }
                        validation_error = modify_artifact_round2_contract(
                            json.dumps(json_body),
                            {
                                "_parsed_json": json_body,
                                "_artifact_store": artifact_store,
                                "_artifact_id": target.artifact_id,
                                # 2026-09-11: the cloud gets the lenient
                                # guard profile (T3 = safety net).
                                "_guard_cfg": {
                                    **_cloud_guard_cfg_for_modify,
                                    "_tier": "T3",
                                },
                                # no _r1_json: the cloud did its own
                                # targeting; the unplanned_target rule
                                # must NOT fire for T3 (it sees the full
                                # artifact and plans its own edits).
                            },
                        )
                        if validation_error:
                            error = validation_error
                        else:
                            try:
                                preview = artifact_store.apply_edits(
                                    json_body,
                                    artifact_id=target.artifact_id,
                                    preview=True,
                                )
                            except Exception as exc:
                                preview = {"errors": [f"edit package error: {exc}"]}
                            if preview.get("errors"):
                                error = "artifact_apply_failed: " + "; ".join(
                                    str(item) for item in preview["errors"]
                                )
                            else:
                                pending_edit = {
                                    "edits": json_body,
                                    "artifact_id": target.artifact_id,
                                    "user_intent": user_intent,
                                    "source_filename": source_filename,
                                }
            else:
                error = "malformed_json: modify_artifact response did not contain an edit package"
        elif _harness_id == "evaluate_artifact":
            # evaluate_artifact: the full non-empty report is the work
            # product; only empty output is a failure.
            extracted = output.strip()
            extracted_lang = None
        else:
            lang = _detect_fence_language(output)
            aliases = _fence_aliases_for(lang, {})
            fence_open = _build_fence_open_re(aliases)
            fence_close = _FENCE_CLOSE_RE
            last_body: str | None = None
            last_end = -1
            for om in fence_open.finditer(output):
                cm = fence_close.search(output, om.end())
                if cm is None:
                    continue
                if om.end() > last_end:
                    last_end = om.end()
                    last_body = output[om.end():cm.start()]
            if last_body is not None:
                extracted = last_body
                extracted_lang = lang
                try:
                    compile(last_body, "<t3_fence>", "exec")
                except SyntaxError as exc:
                    syntax_note = f"syntax_error (informational): {exc.msg}"
            else:
                syntax_note = "no_fence (informational)"
    attempt = CleanAttempt(
        "T3", cloud_model, output, (error is None and bool(output.strip())),
        error, _now() - started,
    )
    result.attempts.append(attempt)
    # DIAG (temporary): T3 outcome visibility.  The dither/guard counts
    # go ONLY to the server stderr (post-mortem telemetry).  They are NOT
    # put in the tier_complete SSE metadata below, so the bench runner
    # receives nothing beyond the final WP + the escalation flags it needs.
    print(
        f"[T3] target=cloud_sota model={cloud_model} "
        f"chars={len(output)} passed={attempt.passed} error={error!r} "
        f"dither={_cloud_n_dither} guarded={_cloud_was_guarded} "
        f"shortstop={_cloud_shortstopped} "
        f"extracted={'yes' if extracted else 'no'}",
        file=sys.stderr, flush=True,
    )
    await emit(CleanProgressEvent(
        event_type="tier_complete", tier="T3", output=output,
        model_id=cloud_model,
        metadata={
            "passed": attempt.passed, "is_t3": True, "target": "cloud_sota",
            # 2026-09-16 (user-requested): the cloud MODEL, not just the role.
            # `model_id` already rides the event, but it is a TOP-LEVEL field
            # and the CLI's `_print_meta` only ever receives `metadata` -- so
            # the model must be inside `metadata` to be displayable.  `target`
            # ("cloud_sota") is a ROLE name and tells the user nothing.
            "model": cloud_model,
            **({"note": syntax_note} if syntax_note else {}),
        },
    ))
    if attempt.passed:
        result.output = visible_output if _harness_id == "general_use" else output
        result.passed = True
        result.selected_model = cloud_model
        result.error = None
        result.fallback_kind = "cloud_sota"
        if pending_edit is not None:
            result.pending_artifact_edit = pending_edit
            result.output = (
                f"[[modify_artifact: pending edit for artifact "
                f"{pending_edit['artifact_id']}]]"
            )
        if _harness_id == "general_use":
            (
                result.general_use_work_product_content,
                result.general_use_work_product_language,
            ) = _extract_general_use_work_product(visible_output)
            if not result.general_use_work_product_content:
                result.general_use_work_product_content = None
        # D4-hb': extraction is a downstream view, not a gate. When the
        # fence scan found a body, stash it for the artifact pipeline
        # (enables follow-up evaluate/modify turns on the cloud WP).
        # When it did not, the raw output above IS the delivered answer.
        result.extracted_artifact_content = extracted
        result.extracted_artifact_language = extracted_lang


async def _invoke_t5_body(
    *,
    result: "CleanExecutionResult",
    work: Any,
    harness: Any,
    transports: dict[str, Any],
    unit: CleanUnit,
    generalist_model: str,
    generalist_provider: str,
    emit: Any,
    run_start: float,
) -> None:
    """T5 body (unchanged behavior): local generalist zero-shot with
    the local-world NOTICE and T5-shaped compose."""
    if generalist_provider not in transports:
        return
    notice = (
        f"NOTICE: returning a zero shot attempt from generalist model "
        f"({generalist_model}) after error/retry exhaustion by specialists."
    )
    package = harness.compose(
        unit, "T5", CleanTierContext(tier="T5", prior_error=None),
    )
    new_messages: list[dict[str, str]] = []
    notice_added = False
    for msg in package.messages:
        if msg.get("role") == "system" and not notice_added:
            new_messages.append({
                "role": "system",
                "content": notice + "\n\n" + (msg.get("content") or ""),
            })
            notice_added = True
        else:
            new_messages.append(msg)
    if not notice_added:
        new_messages.insert(0, {"role": "system", "content": notice})
    package = CleanPackage(**{
        **package.__dict__,
        "model_id": generalist_model,
        "provider": generalist_provider,
        "sampling": {**package.sampling},  # no fixed seed — random by default
        "messages": new_messages,
    })
    package = _disable_response_prefix_for_images(package)
    started = _now()
    output_parts: list[str] = []
    visible_output_parts: list[str] = []
    error: str | None = None
    try:
        transport = transports[generalist_provider]
        async for chunk, is_thinking in transport.stream(package, pathology_monitor=None):
            output_parts.append(chunk)
            if not is_thinking:
                visible_output_parts.append(chunk)
        output = "".join(output_parts)
        visible_output = "".join(visible_output_parts)
    except Exception as exc:
        error = f"t5_error: {exc}"
        output = "".join(output_parts)
        visible_output = "".join(visible_output_parts)
    attempt = CleanAttempt("T5", generalist_model, output, error is None, error, _now() - started)
    result.attempts.append(attempt)
    await emit(CleanProgressEvent(
        event_type="tier_complete", tier="T5", output=output,
        model_id=generalist_model,
        metadata={"passed": attempt.passed, "is_t5": True, "target": "generalist_zero_shot"},
    ))
    if attempt.passed and bool(output.strip()):
        if work.harness.harness_id == "general_use":
            output = _strip_thinking_markup(output)
            visible_output = _strip_thinking_markup(visible_output)
        result.output = visible_output if work.harness.harness_id == "general_use" else output
        result.passed = True
        result.selected_model = generalist_model
        result.error = None
        if work.harness.harness_id == "general_use":
            (
                result.general_use_work_product_content,
                result.general_use_work_product_language,
            ) = _extract_general_use_work_product(visible_output)
            if not result.general_use_work_product_content:
                result.general_use_work_product_content = None


async def _invoke_t5(
    *,
    result: "CleanExecutionResult",
    work: Any,
    harness: Any,
    transports: dict[str, Any],
    vram: Any,
    unit: CleanUnit,
    generalist_model: str,
    generalist_provider: str,
    emit: Any,
    run_start: float,
) -> None:
    """One-shot LOCAL generalist fallback (T5) after the ladder exhausts.

    Thin wrapper over :func:`_invoke_meta_target` -- kept so existing
    callers/tests are unaffected.  T5 is OUTSIDE the tier ladder; it
    does not interact with the pathology counters, commit transitions,
    or per-tier seed inheritance.
    """
    await _invoke_meta_target(
        target="generalist_zero_shot",
        tier_label="T5",
        result=result, work=work, harness=harness,
        transports=transports, vram=vram, unit=unit,
        generalist_model=generalist_model,
        generalist_provider=generalist_provider,
        emit=emit, run_start=run_start,
    )


async def execute_runtime_plan(
    *,
    work: Any,
    harness: Any,
    transports: dict[str, Any],
    vram: Any,
    unit: CleanUnit,
    progress: CleanProgressCallback | None = None,
    checkpoint_interval_s: float = DEFAULT_CHECKPOINT_INTERVAL_S,
    first_token_timeout_s: float = DEFAULT_FIRST_TOKEN_TIMEOUT_S,
    slow_decode_timeout_s: float = DEFAULT_SLOW_DECODE_TIMEOUT_S,
    session_memory: SessionMemory | None = None,
    inject_last_turn: bool = True,
    # Path B remediation: the artifact store reference and
    # the specific artifact id to operate on.  Used only by
    # artifact-aware harnesses (modify_artifact,
    # evaluate_artifact).  ``artifact_store=None`` means
    # those two harnesses cannot run correctly (their tier
    # will fail with a clear "no artifact store configured"
    # functional error rather than silently producing wrong
    # output).  ``artifact_id=None`` resolves to
    # ``artifact_store.current`` at call time -- this
    # preserves today's single-current-artifact behavior and
    # leaves room for a future tag/chronological resolver
    # (Step 5/6 scope) to supply a specific id without any
    # signature change here.
    artifact_store: Any | None = None,
    artifact_id: str | None = None,
    # v14: evaluate-object holding pen (single-slot).  ``None`` (the
    # default) means no evaluate_artifact source/WP persistence; the
    # server passes its EvaluateObjectStore so a follow-up turn can
    # operate on the same source context.
    evaluate_store: Any | None = None,
    evaluate_prior_wp: bool = False,
    # Step 5: the router's one-line restatement of user intent and
    # the attachment's original filename (if any).  Threaded through
    # from ``CleanExecutor.execute()`` (Step 5.2).  Inert until the
    # two-round modify_artifact block's fork/promotion logic reads
    # them for descriptor-tag derivation.  Both keyword-only with
    # defaults -- no existing call site's behavior changes.
    user_intent: str = "",
    source_filename: str | None = None,
    context_text: str = "",
    t3_context_text: str = "",
    # v13 step 2 (2026-08-08): the submitted file content for the
    # bug_fix / unified_diff_patch context-correctness gate.  When
    # provided, the contract selects the LAST <patch> block whose
    # removed lines all appear in this content, and rejects if none
    # do.  ``None`` (the default) keeps the legacy take-last behavior.
    patch_base: str | None = None,
    # 2026-09-14 (spec 5.7.2): PATH A -- local REGENERATION for an artifact
    # that carries its originating prompt.  ``modify_combo_prompt`` replaces
    # the change-order instruction with the combined prompt and produces a
    # COMPLETE artifact; ``modify_skip_reintegration`` then tells the runtime
    # not to expect/validate a change order at all, because there is none to
    # apply.  Both default to inert values so every existing caller is
    # unaffected.
    modify_combo_prompt: str | None = None,
    modify_skip_reintegration: bool = False,
    # 2026-09-14 (spec 4.1): PATH A's regeneration WORDING.
    # CONFIG-OWNED, and since M6 owned directly by the executing harness
    # (`modify_harness.runtime`), which `runtime = work.harness.runtime` has
    # already seeded.  Opaque strings, with an in-code fallback each.
    #
    # 2026-09-17 (user ruling): `modify_regeneration_contract` is RETIRED --
    # the contract is now selected from the ARTIFACT'S LANGUAGE
    # (`denovo_contract_for_language`), never from config.
    modify_regeneration_system_suffix: str | None = None,
    modify_regeneration_output_instruction: str | None = None,
    # 2026-09-14 (spec 6.1-6.4): master switch for the PATH A drift checks
    # (presence gate / Tier-1 literal diff / Tier-2 cloud-parent review).
    # ``None`` means "not specified" and leaves the runtime's own default
    # (on) in place, so an undeclared deployment is unaffected.
    modify_drift_checks: bool | None = None,
    # v12 NEW PARAMS (defaulted so old callers still work):
    # T5 (post-ladder generalist) invocation.  When the
    # declared ladder is exhausted without a passing
    # attempt, the runtime may make ONE additional
    # zero-shot call to a generalist model
    # (``generalist_model``) as a last-resort fallback.
    # ``None`` disables T5 entirely.  The
    # ``generalist_provider`` selects the transport
    # used to reach the generalist (typically the
    # ``ollama`` provider; FLM does not have a
    # generalist-grade model).
    generalist_model: str | None = None,
    generalist_provider: str = "ollama",
    # v12: optionally skip the T5 fallback even when a
    # generalist is configured.  Used by tests and
    # by user opt-in ("bare mode") to surface
    # unassisted ladder results.
    enable_t5: bool = True,
    # T3 (cloud_sota) escalation target -- DISTINCT from T5.  The local
    # generalist T5 slot is UNTOUCHED (always local).  ``t3_model``/
    # ``t3_provider`` select the model+transport for the cloud_sota
    # meta-target when ``backends.cloud.enabled``.  ``None`` = cloud
    # escalation disabled; cloud_sota targets are skipped and T5 remains
    # the meta fallback.
    t3_model: str | None = None,
    t3_vision_model: str | None = None,
    t3_provider: str | None = None,
    # User-initiated direct cloud mode.  When true, skip the local tier
    # ladder and invoke T3 once; automatic post-ladder T3 remains governed
    # by the harness's meta_harness_ladder below.
    direct_t3: bool = False,
    # Per-instance native-reasoning override. None = use per-model
    # `think` default (model card). True = force think-on, False =
    # force think-off. Wins over the model-card default.
    think_override: bool | None = None,
    # Optional per-request generation seed. Replaces the harness's base
    # sampling seed; the declared retry policy still governs rerolls.
    seed_override: int | None = None,
    # Live per-token raw disk writer.  When provided,
    # every chunk that arrives on the model stream is
    # appended to ``raw/<qid>.raw.txt`` and fsync'd so
    # the on-disk file is always current.  This is
    # the local-context/debug save the clean slice is
    # required to perform.  When ``None``, no disk
    # write happens (legacy behaviour).
    raw_token_writer: Any | None = None,
) -> Any:
    """Run declared tiers with live progress and checkpointed control.

    Three independent control mechanisms, all sharing the
    same per-chunk stream loop:

    1. Per-chunk guards (dither, loop, contract): content-
       based.  These used to escalate to the next tier on
       a guard verdict, but escalation is the wrong
       response to a generation-quality pathology -- the
       same prompt can cause any model to dither, and a
       different *seed* is more likely to break the
       pattern than a different *model*.  So dither /
       loop / multiple_patches / empty_patch /
       malformed_patch / no_code_fence / syntax_error
       are now classified as pathology and trigger the
       same-tier seed-restart path (item 3).  Tier
       escalation is reserved for: ``token_exhaustion``
       (length cap reached, not a content pathology) and
       any error *not* in ``retry_on``.

    2. VRAM watchdog: peer background task.  Polls /api/ps
       every 1.5s, sets ``vram_breach`` event on a budget
       overrun.  Treated as a correctness-tier error
       (escalates).

    3. No/slow token pathology: peer background task.  Ticks
       every 5s, checks ``now - last_token_at`` against
       the configured budget (cold 75s / warm 60s / slow
       90s), and on a violation calls
       ``transport.request_close()`` to signal the active
       stream to abort.  The transport's ``stream()``
       generator raises ``StreamAborted`` on its next
       iteration when it sees the close flag.

       All pathology errors (transport pathologies AND
       content-quality pathologies) restart the SAME
       tier with a new seed, up to
       ``max_pathology_restarts`` times.  After that, we
       move on to the next tier.  This is the principled
       response: a new seed breaks the symmetry that
       produced the failure; a new model hits the same
       prompt and the same failure mode.
    """
    global last_prepared_key
    runtime = work.harness.runtime
    # v13 step 2 (2026-08-08): inject the submitted file content into
    # the runtime so the unified_diff_patch contract's context-
    # correctness gate can use it.  ``None`` (the default) leaves the
    # runtime untouched and the gate inactive.
    if patch_base is not None:
        runtime["patch_base"] = patch_base
    # 2026-09-14 (spec 5.7.2 PATH A): expose the regeneration mode to the
    # tier loop via the runtime dict, which is how per-call options reach the
    # modify_artifact branch without widening its signature further.
    if modify_combo_prompt:
        runtime["modify_combo_prompt"] = modify_combo_prompt
        runtime["modify_skip_reintegration"] = bool(modify_skip_reintegration)
        # ============================================================
        # 2026-09-17 (REVIEW FIX): resolve the artifact's LANGUAGE once, HERE,
        # before the attempt loop.
        # ============================================================
        # This is a property of the REQUEST (which artifact is being modified),
        # NOT of the ATTEMPT, so it must not be recomputed per iteration.
        #
        # Why it must live here and not only at the contract-selection site:
        # `_regen_is_prose` is READ by the PATH A message-build block (which
        # chooses the prose vs code WORDING) and that block runs BEFORE the
        # contract-selection block inside the same
        # `while calls < max_calls` iteration.  Left at the selection site only,
        # attempt 1 always read an ABSENT key -> falsy -> CODE wording (which
        # demands "a single fenced code block"), so the prose wording reached
        # RETRIES only.  `prose_contract` accepts a fenced body -- it strips the
        # fence -- so the defect was MASKED rather than loud.
        if artifact_store is not None:
            _seed_parent = artifact_store.get(artifact_id)
            _seed_lang = str(getattr(_seed_parent, "language", "") or "")
        else:
            _seed_lang = ""
        runtime["_denovo_target_language"] = _seed_lang
        runtime["_regen_is_prose"] = (
            denovo_contract_for_language(_seed_lang) == "prose_contract"
        )
    # 2026-09-14 (spec 4.1): PATH A's regeneration contract + wording.
    # CONFIG-OWNED, and since M6 owned directly by the executing harness:
    # these four keys live on `modify_harness.runtime` itself, which
    # `runtime = work.harness.runtime` has already seeded.  The old
    # server-side copy from a separate `modify_artifact_regenerate` cell is
    # retired.  Bound only when supplied, so the harness's own values are
    # never overwritten by an empty override.
    if modify_regeneration_system_suffix:
        runtime["modify_regeneration_system_suffix"] = modify_regeneration_system_suffix
    if modify_regeneration_output_instruction:
        runtime["modify_regeneration_output_instruction"] = (
            modify_regeneration_output_instruction
        )
    if modify_drift_checks is not None:
        runtime["modify_drift_checks"] = bool(modify_drift_checks)
    # 2026-09-14 (user ruling): DENOVO PATH A gets the DENOVO primitive set.
    #
    # SUPERSEDED 2026-09-16 (M6).  This block used to read
    # `modify_regeneration_primitives` from the executing harness and override
    # `runtime["primitives"]`.  Two things made it dead:
    #
    #   1. NO config ever declared `modify_regeneration_primitives` -- not the
    #      `modify_artifact_regenerate` cell, not `modify_harness`.  The branch
    #      could never fire.
    #   2. It existed only because PATH A used to run on the CHANGE-ORDER
    #      harness (`runtime["primitives"]` included modify_output_cap_guard,
    #      a guard whose premise is relative to a change order).  PATH A now
    #      runs on `modify_harness`, whose OWN declared primitives are
    #      [dither_guard, repetition_loop_guard] -- i.e. the correct denovo set
    #      arrives through `runtime = work.harness.runtime` with no override
    #      needed.
    #
    # So the primitives are now config-owned in the direct, ordinary way: the
    # executing harness declares them.  Removed rather than left inert, because
    # a branch that reads a key no config can supply is a trap for the next
    # person (it LOOKS like a live override point).
    #
    # Reversal: re-add the block and declare the key on the harness.
    unit = CleanUnit(
        domain=unit.domain,
        prompt=unit.prompt,
        persona=unit.persona,
        closed_think_block=unit.closed_think_block,
        context_text=context_text or unit.context_text,
        t3_context_text=t3_context_text or unit.t3_context_text,
        images=list(unit.images or []),
    )
    result = CleanExecutionResult(work=work)
    calls = 0
    prior_error: str | None = None
    max_calls = int(runtime.get("max_calls_per_instance", len(work.harness.tiers)))
    # (The former D5 ``crashed_models`` crash-skip set is REMOVED as of
    # 2026-09-12.  Backend errors are classified as ordinary pathologies:
    # announced loudly, absorbed by the normal retry/reroll/escalation
    # budget, with no model- or slot-blacklisting.)
    # ── Output ceiling state (problem-scoped) ──
    # cumulative_attempt_chars sums attempt-level output across every
    # pathology reroll and tier within this problem, so N runaways
    # cannot each stay just under a per-attempt ceiling.  The yaml
    # values ALWAYS win over the failsafe constants.  When the
    # CUMULATIVE ceiling trips, the LOCAL INFERENCE CONTRACT IS OVER:
    # every further local attempt would add output to a system already
    # over budget (the ceiling can never be un-met), so the ladder
    # terminates immediately and escalation happens at the meta level
    # (T3 cloud / T5 local per the harness's meta_harness_ladder).
    ceiling_enabled = bool(_resolve_cfg(runtime, "output_ceiling.enabled"))
    ceiling_max_chars = float(
        _resolve_cfg(runtime, "output_ceiling.max_chars")
    )
    ceiling_total_max_chars: float | None = float(
        _resolve_cfg(runtime, "output_ceiling.max_total_chars")
    )
    # 2026-08-30: held-out minimum char budget for the secondary slot.
    # The primary's cumulative ceiling is effectively
    # (max_total_chars - min_secondary_chars) so the secondary always has
    # >= min_secondary_chars of headroom when it escalates.  The global
    # max_total_chars stays a hard cap on the TOTAL.  Default 0 = no
    # reservation (secondary gets whatever remains).
    ceiling_min_secondary_chars: float = float(
        _resolve_cfg(runtime, "output_ceiling.min_secondary_chars") or 0.0
    )
    cumulative_attempt_chars = 0
    ceiling_cumulative_tripped = False
    # 2026-09-10 (user ruling): modify-only output-cap state.  The cap
    # re-roll budget is DEDICATED (output_cap_max_re_rolls, default 1)
    # and parallel to the shared pathology budget: a cap re-roll never
    # increments pathology_count.  output_cap_reroll_now drives the
    # fresh-seed + modifier injection on the immediate next attempt.
    output_cap_count = 0
    output_cap_reroll_now = False
    # 2026-08-30: which model_slots have already tripped the cumulative
    # ceiling.  Once a slot trips, we advance to the NEXT slot (the
    # secondary gets its held-out budget); only when the secondary also
    # trips (or no next slot exists) do we go to T3.  The while-gate below
    # breaks only for slots already in this set, so the next slot can run.
    ceiling_tripped_slots: set[str] = set()
    # v12 pathology (single consolidated bucket) and
    # functional error sets.  Read from the YAML via
    # _resolve_cfg so the harness can override the
    # default lists in DEFAULT_HARNESS_CONFIG.
    pathology_errs = set(_resolve_cfg(runtime, "pathology.errors") or ())
    func_errs = set(_resolve_cfg(runtime, "error_correction.functional_errors") or ())
    # v13 step 2 (2026-08-08): the bug_fix harness declares its
    # functional errors at the TIER level in the escalation_ladder
    # (``escalation_ladder.<tier>.functional_errors``), not under
    # ``error_correction.functional_errors``.  Without merging these
    # in, ``func_errs`` was empty and every contract failure
    # (malformed_patch, patch_does_not_apply) fell into the
    # "unrecognized error" branch -> immediate ``break`` -> only ONE
    # try per model instead of the intended functional retry.  Merge
    # the tier-level lists from all tiers in the ladder.
    _ladder = work.harness.escalation_ladder
    for _tier_cfg in _ladder.values():
        if isinstance(_tier_cfg, dict):
            _tier_func_errs = _tier_cfg.get("functional_errors")
            if _tier_func_errs:
                func_errs |= set(_tier_func_errs)
    # v12 per-model pathology budget.  ``max_re_rolls_per_model``
    # is the number of T0 re-rolls the model is allowed after
    # its initial attempt; the (N+1)th pathology trigger
    # commits to the next model.  The rule is uniform across
    # primary and secondary: ``pathology_count > max_re_rolls``.
    # The runtime does NOT bake in any "primary uses N, secondary
    # uses N+K" arithmetic; the harness declares the policy.
    max_re_rolls = int(_resolve_cfg(runtime, "pathology.max_re_rolls_per_model") or 0)
    # v12 per-MODEL pathology counter.  Resets to 0
    # at every commit (primary->secondary, secondary->t3,
    # etc.).  ``transition_to_secondary`` and
    # ``transition_to_t5`` are unchanged in step 2; they
    # are the phase-bookkeeping flags.  They will be
    # replaced in step 3 with a per-stage state machine.
    pathology_count: int = 0
    # v12 per-SLOT functional error counter (2026-09-12, user-ruled).
    #
    # SLOT, not model.  A slot is a role the harness declares
    # (``escalation_ladder.<tier>.model_slot``: primary / secondary /
    # generalist).  Budgets are accounted per SLOT and reset when the
    # resolved slot changes, so the primary and the secondary each get
    # the harness's full budget no matter how the earlier slot failed.
    # Model identity is deliberately NEVER consulted here: which model is
    # bound to a slot is a configuration/backend-resolution concern, and a
    # slot-aware routine must behave identically whether or not two slots
    # happen to bind the same model.
    #
    # Historical note: this was documented as "per-MODEL, resets on commit"
    # and the reset guard keyed on a hardcoded ``current_slot = "primary"``
    # that was never reassigned, so the counter effectively accumulated
    # across the whole request and the secondary's budget was consumed in
    # sequence by the primary's failures (live 2026-09-12: T0's
    # ``self_review_rejected`` left the count at 1, so T2's
    # ``malformed_json`` evaluated 2 > 1 and got NO retry).
    #
    # The commit threshold is resolved PER TIER from the ladder's declared
    # ``functional_budget`` (principle: every variable parameter originates
    # as a config value; DEFAULT_HARNESS_CONFIG is only the backup for a
    # ladder tier that omits the key).
    max_func_errors = int(
        _resolve_cfg(runtime, "error_correction.max_functional_errors_before_commit")
        or runtime.get("max_functional_errors_before_commit", 1)
    )
    functional_fail_count: int = 0
    # Per-slot budget resolution helper.  Reads the CURRENT tier's declared
    # ``functional_budget``; falls back to the harness-wide value above
    # when the tier omits it (backup only, never the primary source).
    def _func_budget_for_tier(_tier_name: str) -> int:
        _t_cfg = work.harness.escalation_ladder.get(_tier_name.lower(), {})
        _declared = _t_cfg.get("functional_budget") if isinstance(_t_cfg, dict) else None
        if _declared is None:
            return max_func_errors
        return int(_declared)
    # The slot whose budget is currently being accounted.  Resolved from
    # the ladder per tier inside the tier loop (NOT a hardcoded literal).
    _budget_slot: str | None = None
    # v12 T1 primitive: capture T0's output + error so the
    # T1 iteration can compose the T1b prompt with the
    # YAML-driven ``t1_full_user_modifier`` template.  These
    # are updated at the end of each T0 attempt (passed or
    # failed) and read at the start of the T1 iteration.
    # ``t0_output_for_t1`` is the full T0 model output;
    # ``t0_error_for_t1`` is the contract verdict that
    # caused T0 to fail (None if T0 passed, in which
    # case T1 never runs).
    t0_output_for_t1: str | None = None
    t0_error_for_t1: str | None = None
    # v12 T1 strategy (resolved once per run).  In step 4a,
    # ``targeted_then_full`` is accepted but behaves
    # identically to ``full_only``: the T1a fast-path is
    # not yet wired.  The strategy is read so the YAML can
    # declare the intent; step 4b will implement T1a.
    t1_strategy = _resolve_t1_strategy(runtime)
    # v12 fence counter: resolve the fence language from the
    # work item's language field (set by the router) and build
    # the regexes once.  The fence counter watches for code
    # fences in this language only — per the single-domain/
    # single-task unit-of-work design, each execute() call
    # handles one language.
    _fence_language = getattr(work, "language", None) or getattr(unit, "domain", None)
    _fence_aliases = _fence_aliases_for(_fence_language, runtime)
    _fence_tag_re = _build_fence_tag_re(_fence_aliases)
    _fence_cadence = int(_resolve_cfg(runtime, "fence_counter.check_cadence_chunks") or -1)
    _fence_tail_threshold = int(_resolve_cfg(runtime, "fence_counter.tail_after_close_without_reopen_chunks") or -1)
    # v13 patch block counter: resolved once per run.  The
    # counter watches </patch> closes and force-closes the
    # stream when ``max_blocks`` closes are observed.  No
    # preamble/tail thresholds — dither/loop guards cover
    # those pathologies in the prose between patch blocks.
    #
    # v13 step 2 (2026-08-08): the config nests the counter under
    # ``stream_primitives.patch_block_counter``, but ``_resolve_cfg``
    # reads the FLAT key ``patch_block_counter.*``.  This mismatch
    # meant the counter silently fell back to the DEFAULT (cadence=-1,
    # max_blocks=3) and never fired -- so ``max_blocks: 1`` in the
    # YAML had no effect and a bresenham hallucination slipped past
    # the autostop.  Resolve the nested path first, then the flat path.
    _sp = runtime.get("stream_primitives", {})
    _pbc = _sp.get("patch_block_counter", {}) if isinstance(_sp, dict) else {}
    _patch_cadence = int(
        _pbc.get("check_cadence_chunks")
        if _pbc.get("check_cadence_chunks") is not None
        else _resolve_cfg(runtime, "patch_block_counter.check_cadence_chunks")
        or -1
    )
    _patch_max_blocks = int(
        _pbc.get("max_blocks")
        if _pbc.get("max_blocks") is not None
        else _resolve_cfg(runtime, "patch_block_counter.max_blocks")
        or 0
    )
    # Per-tier last seed (for T1 inheriting T0's seed).
    tier_seed: dict[int, int] = {}
    # v12 one-shot state change variables.  Each fires at
    # most once per runtime call.  Stickiness is structural.
    transition_to_secondary: bool = False
    transition_to_t5: bool = False
    first_token_timeout_s = float(runtime.get("first_token_timeout_s", first_token_timeout_s))
    slow_decode_timeout_s = float(runtime.get("slow_decode_timeout_s", slow_decode_timeout_s))
    progress_cb: CleanProgressCallback = progress or NullProgressCallback()
    run_start = _now()
    last_checkpoint_at = run_start
    first_system_modified: str | None = None
    routing_decision: dict[str, Any] = {
        "domain": getattr(work, "domain", None),
        "task_type": getattr(work, "task_type", None),
        "language": getattr(work, "language", None),
        "primary_model": getattr(work, "primary_model", None),
        "secondary_model": getattr(work, "secondary_model", None),
        "harness_id": getattr(getattr(work, "harness", None), "harness_id", None),
        "persona": getattr(work, "persona", None),
        "shape": getattr(work, "shape", None),
        "effort": getattr(work, "effort", None),
        "matrix_cell": getattr(work, "matrix_cell", None),
    }
    resolver_decision: dict[str, Any] = {
        "resolution_kind": getattr(work, "resolution_kind", None),
        "fallback_reason": getattr(work, "fallback_reason", None),
        "domain_confidence": float(getattr(work, "domain_confidence", 0.0) or 0.0),
    }

    async def emit(event: CleanProgressEvent) -> CleanControlDecision:
        decision = await _emit_async(progress_cb, event)
        if session_memory is not None:
            try:
                mem_result = session_memory.on_event(event)
                if hasattr(mem_result, "__await__"):
                    await mem_result
            except Exception as exc:  # pragma: no cover
                print(f"[warn] session_memory.on_event failed: {exc}", file=sys.stderr)
        return decision

    if session_memory is not None:
        try:
            session_memory.begin_turn(
                task_id=unit.task_id if hasattr(unit, "task_id") else "",
                user_prompt=unit.prompt,
                routing_decision=routing_decision,
                resolver_decision=resolver_decision,
                persona=getattr(work, "persona", None),
                harness_id=getattr(getattr(work, "harness", None), "harness_id", None),
                primary_model=getattr(work, "primary_model", None),
                primary_provider=getattr(work, "primary_provider", None),
            )
        except Exception as exc:  # pragma: no cover
            print(f"[warn] session_memory.begin_turn failed: {exc}", file=sys.stderr)

    # Manual tier index counter so we can re-iterate the
    # same tier on a pathology error.  The standard
    # ``for tier_idx, tier in enumerate(...)`` cannot
    # easily do "continue at the same index".
    # Live raw token writer: write a problem-start
    # header with the harness, work, and prompt-token
    # estimate so post-mortem tools can identify the
    # problem at a glance.
    if raw_token_writer is not None:
        try:
            _pkg0 = harness.compose(unit, work.harness.tiers[0], CleanTierContext(tier=work.harness.tiers[0], prior_error=None))
            raw_token_writer.begin_problem(
                harness=getattr(work.harness, "harness_id", "?"),
                work=str(work),
                prompt_tokens=_estimate_package_tokens(_pkg0),
                seed=runtime.get("seed"),
                temperature=runtime.get("temperature"),
                top_p=runtime.get("top_p"),
                top_k=runtime.get("top_k"),
                num_ctx=runtime.get("num_ctx"),
                num_predict=runtime.get("num_predict"),
            )
        except Exception:
            pass
    tier_idx = 0

    # User-initiated direct cloud request.  This deliberately bypasses the
    # local tier loop but uses the same T3 dispatcher and cloud settings as
    # the normal post-ladder path.  Return immediately so T3 cannot be
    # invoked a second time and T5 cannot run afterward.
    if direct_t3:
        selected_t3_model = t3_vision_model if unit.images else t3_model
        direct_meta = _ladder.get("meta_harness_ladder") or {}
        direct_cloud_cfg = direct_meta.get("cloud") or DEFAULT_DIRECT_CLOUD_CONFIG
        direct_cloud_silence = direct_cloud_cfg.get("silence_timeout_s")
        direct_cloud_total = direct_cloud_cfg.get("total_timeout_s")
        # 2026-09-09 (user ruling): cloud reasoning effort is a separate
        # parameter ("low"|"medium"|"high") — never derived from the
        # local think flag. A user /thinkon-thinkoff override does not
        # touch cloud effort either.
        direct_cloud_effort = direct_cloud_cfg.get("reasoning_effort")
        direct_cloud_max_tokens = direct_cloud_cfg.get("max_tokens")
        if direct_cloud_max_tokens is None:
            direct_tps = direct_cloud_cfg.get("tokens_per_sec")
            if direct_tps is not None and direct_cloud_total is not None:
                direct_cloud_max_tokens = int(direct_tps) * int(direct_cloud_total)
        if direct_cloud_max_tokens is not None:
            direct_cloud_max_tokens = int(direct_cloud_max_tokens)
        direct_error = None
        if (
            selected_t3_model is None
            or t3_provider is None
            or t3_provider not in transports
        ):
            direct_error = (
                "t3_unavailable: direct cloud mode requested, but the "
                "required cloud transport/model is not configured or unavailable"
            )
            result.error = direct_error
            result.output = direct_error
        else:
            await emit(CleanProgressEvent(
                event_type="tier_start", tier="T3", model_id=selected_t3_model,
                elapsed_s=_now() - run_start,
                metadata={"direct_t3": True, "target": "cloud_sota"},
            ))
            try:
                await _invoke_meta_target(
                    target="cloud_sota",
                    tier_label="T3",
                    result=result, work=work, harness=harness,
                    transports=transports, vram=vram, unit=unit,
                    generalist_model=selected_t3_model,
                    generalist_provider=t3_provider,
                    emit=emit, run_start=run_start,
                    cloud_effort=direct_cloud_effort,
                    cloud_max_tokens=direct_cloud_max_tokens,
                    cloud_silence_timeout_s=direct_cloud_silence,
                    cloud_total_timeout_s=direct_cloud_total,
                    cloud_guard_cfg=direct_cloud_cfg,
                    checkpoint_interval_s=checkpoint_interval_s,
                    raw_token_writer=raw_token_writer,
                    artifact_store=artifact_store,
                    artifact_id=artifact_id,
                    evaluate_store=evaluate_store,
                    evaluate_prior_wp=evaluate_prior_wp,
                    user_intent=user_intent,
                    source_filename=source_filename,
                )
            except Exception as exc:
                direct_error = f"t3_error: direct cloud target failed: {exc}"
                result.error = direct_error
                result.output = direct_error
            if not result.passed and result.attempts:
                last_direct_attempt = result.attempts[-1]
                result.output = last_direct_attempt.output
                result.selected_model = last_direct_attempt.model_id
                result.error = last_direct_attempt.error or "t3_error: direct cloud attempt failed"
        result.total_elapsed_s = _now() - run_start
        await emit(CleanProgressEvent(
            event_type="complete", tier="T3", output=result.output,
            model_id=result.selected_model or selected_t3_model or "",
            elapsed_s=result.total_elapsed_s,
            metadata={"passed": result.passed, "direct_t3": True,
                      "target": "cloud_sota",
                      **({"pending_artifact_edit": True} if result.pending_artifact_edit else {}),
                      **({"error": direct_error} if direct_error else {})},
        ))
        return result

    # 2026-08-30: slot mapping for the slot-aware cumulative ceiling.
    # Each tier maps to a model_slot (primary/secondary) from the ladder.
    # The cumulative ceiling is slot-aware: the PRIMARY trips at
    # (max_total_chars - min_secondary_chars) to reserve headroom for the
    # secondary; the SECONDARY trips at max_total_chars.  This lets the
    # secondary always get its held-out minimum budget before T3.
    _ladder_slots = work.harness.escalation_ladder
    tier_model_slots: dict[str, str] = {}
    for _t in work.harness.tiers:
        _tcfg = _ladder_slots.get(_t.lower(), {})
        tier_model_slots[_t] = _tcfg.get("model_slot", "primary")
    # 2026-09-12 (user-ruled): the D5 ``crashed_models`` set is GONE.
    # Stop state must be keyed on slots and error classes, never on model
    # identity; and a single backend error is not grounds to discard a
    # model or a slot.  Backend errors are pathologies now -- announced
    # loudly, absorbed by the normal budget.
    while tier_idx < len(work.harness.tiers):
        tier = work.harness.tiers[tier_idx]
        # 2026-09-12 (user-ruled): resolve the CURRENT SLOT from the ladder
        # for this tier, and reset the per-slot budgets whenever the slot
        # changes.  Budgets are accounted per SLOT, never per model: the
        # primary and the secondary each receive the harness's full
        # functional + pathology budget regardless of how the earlier slot
        # failed.  (Previously a hardcoded ``current_slot = "primary"``
        # meant the secondary inherited the primary's consumed budget.)
        _this_slot = tier_model_slots.get(tier, "primary")
        if _budget_slot is not None and _this_slot != _budget_slot:
            functional_fail_count = 0
            pathology_count = 0
            prior_error = None
            print(
                f"[ladder] slot change {_budget_slot} -> {_this_slot} at {tier}: "
                f"per-slot budgets reset (functional={_func_budget_for_tier(tier)}, "
                f"pathology={max_re_rolls})",
                file=sys.stderr, flush=True,
            )
        _budget_slot = _this_slot
        # Per-tier declared functional budget (config-owned; the
        # harness-wide value is only the backup).
        _max_func_errors_this_tier = _func_budget_for_tier(tier)
        # 2026-09-12 (user-ruled): the D5 model-blacklist skip that lived
        # here is REMOVED.  It advanced past a slot when the MODEL bound to
        # it had crashed deterministically -- a stop decision keyed on
        # model identity, which conflates the slot (the decisional unit)
        # with whatever model is configured into it, and which discarded a
        # same-model sibling slot unnecessarily.  A backend error is now
        # classified as an ordinary pathology (see the classification site
        # below): it is announced loudly and consumes the normal
        # retry/reroll/escalation budget.
        await emit(CleanProgressEvent(
            event_type="tier_start", tier=tier,
            model_id=work.primary_model if tier in {"T0", "T1"} else work.secondary_model,
        ))
        # Live raw token writer: write a tier boundary
        # header so the on-disk file is self-describing.
        if raw_token_writer is not None:
            try:
                raw_token_writer.begin_tier(
                    tier,
                    work.primary_model if tier in {"T0", "T1"} else work.secondary_model,
                )
            except Exception:
                pass
        while calls < max_calls:
            # ── Cumulative-ceiling gate ──
            # 2026-08-30: slot-aware.  Once a model_slot's cumulative
            # ceiling trips, that slot's local contract is over (further
            # attempts on the SAME slot are pointless churn).  But we do
            # NOT break the whole ladder: the NEXT slot (secondary) gets
            # its held-out budget.  So this gate breaks only when the
            # CURRENT slot has already tripped.  When the secondary also
            # trips (or no next slot exists), the tier loop exhausts and
            # the meta level (T3) takes over.
            _gate_slot = tier_model_slots.get(tier, "primary")
            if (
                ceiling_enabled
                and ceiling_cumulative_tripped
                and _gate_slot in ceiling_tripped_slots
            ):
                break
            # 4.6: Two-round modify_artifact execution.
            # When the harness declares rounds: [r1, r2] in its
            # escalation config AND work.harness.rounds has both
            # r1 and r2 populated, the tier is a two-round
            # pipeline: R1 (analysis) then R2 (edit generation).
            # The whole R1+R2 sequence is one functional try.
            # If either round fails functionally, the whole
            # tier escalates.
            #
            # R1: model produces a JSON canonical truth package.
            #   Validated against r1's artifact_contract.
            #   If fail -> functional error (escalate).
            # R2: model produces a JSON edit package using R1's
            #   output as context.  Validated against r2's
            #   artifact_contract.  If fail -> functional error.
            # APPLY: r2's apply_policy deterministically applies
            #   the edits to the artifact.  The patched artifact
            #   is the tier's output.
            ladder = work.harness.escalation_ladder
            ladder_cfg = ladder.get(tier.lower(), {})
            rounds_map = work.harness.rounds
            two_round_modify = (
                rounds_map is not None
                and rounds_map.get("r1") is not None
                and rounds_map.get("r2") is not None
                and isinstance(ladder_cfg.get("rounds"), list)
                and sorted(ladder_cfg["rounds"]) == ["r1", "r2"]
            )
            if two_round_modify:
                # === TWO-ROUND EXECUTION PATH ===
                #
                # We use a mini state machine:
                #   _round = 0  -> R1 (analysis)
                #   _round = 1  -> R2 (edit generation)
                #   _round = 2  -> done (apply edits, output)
                #
                # Each round gets its own compose + transport call.
                # The whole pipeline uses ONE functional try from
                # the escalation budget.  If either round fails,
                # the tier's functional budget is spent.
                #
                # Path B remediation: this path requires an
                # ``artifact_store`` (threaded through from
                # ``execute_runtime_plan``'s caller).  Without one,
                # there is no artifact to build an outline/BM25
                # index from or to apply edits to -- fail with a
                # clear functional error instead of silently
                # producing wrong output (the prior behavior).
                if artifact_store is None:
                    error = "no_artifact_store: modify_artifact requires an artifact_store"
                    functional_fail_count += 1
                    break
                target_artifact = artifact_store.get(artifact_id)
                if target_artifact is None:
                    error = "no_current_artifact: no artifact selected for modify_artifact"
                    functional_fail_count += 1
                    break
                resolved_artifact_id = target_artifact.artifact_id
                r1_cfg = rounds_map["r1"]
                r2_cfg = rounds_map["r2"]
                _round = 0  # 0 = R1, 1 = R2, 2 = done
                r1_output: str | None = None
                r1_json: dict[str, Any] | None = None
                r2_output: str | None = None
                r2_json: dict[str, Any] | None = None
                _round_error: str | None = None
                r1_retry_count = 0
                r1_retry_error: str | None = None
                r2_retry_count = 0
                r2_retry_error: str | None = None
                while calls < max_calls and _round < 2:
                    round_cfg = r1_cfg if _round == 0 else r2_cfg
                    is_r2 = _round == 1
                    round_name = (
                        "R1_retry" if _round == 0 and r1_retry_count
                        else "R1" if _round == 0
                        else "R2_retry" if r2_retry_count else "R2"
                    )
                    # Compose the round package.  R2 gets R1's
                    # output appended to its user message as
                    # context.
                    # 2026-09-08: the rounds' system_suffix comes from
                    # the RAW config rounds dict, which never passed
                    # through resolution's {{domain_tag}} substitution
                    # (that pass only rewrites the top-level harness
                    # fields).  Every R1/R2 request has been shipping
                    # the literal "{{domain_tag}}" to the model — apply
                    # the same substitution here that resolution.py
                    # does (domain_tag = domain minus "domain_" prefix,
                    # matching _domain_to_language).
                    _round_language = str(getattr(work, "domain", "") or "").removeprefix(
                        "domain_"
                    )
                    system_suffix = str(round_cfg.get("system_suffix", ""))
                    if _round_language and "{{domain_tag}}" in system_suffix:
                        system_suffix = system_suffix.replace(
                            "{{domain_tag}}", _round_language
                        )
                    _raw_prefill = round_cfg.get("assistant_prefill")
                    prefill = "" if _raw_prefill is None else str(_raw_prefill)
                    # 2026-09-14 (spec 5.7.2 PATH A): the harness's declared
                    # system_suffix is the CHANGE-ORDER instruction ("return
                    # ONLY a JSON object ... never return the artifact").  On
                    # the regeneration path that instruction is exactly wrong,
                    # so it is replaced with a build instruction that asks for
                    # the COMPLETE artifact.  Without this the model is told to
                    # emit JSON while the combined prompt asks for an artifact,
                    # and the contract swap would then reject whatever came
                    # back.
                    if runtime.get("modify_skip_reintegration"):
                        # 2026-09-14 (spec 4.1): the regeneration
                        # instruction is CONFIG-OWNED.  It is read from
                        # the executing harness's own runtime block, so a
                        # deployment can retune the regeneration wording
                        # (or point at a different declared cell) without
                        # editing code.  The literal below is the
                        # FALLBACK only -- it applies when the harness
                        # declares nothing, which keeps an undeclared
                        # deployment working exactly as before.
                        if runtime.get("_regen_is_prose"):
                            # 2026-09-17: prose asks for TEXT, never a fence.
                            # The fence instruction is what invited the
                            # ```python wrapper that then failed compile() on a
                            # curly apostrophe.
                            _regen_suffix = str(
                                runtime.get(
                                    "modify_regeneration_prose_system_suffix"
                                ) or ""
                            )
                            if not _regen_suffix:
                                _regen_suffix = _PROSE_REGEN_SUFFIX_DEFAULT
                        else:
                            _regen_suffix = str(
                                runtime.get("modify_regeneration_system_suffix")
                                or ""
                            )
                            if not _regen_suffix:
                                _regen_suffix = _CODE_REGEN_SUFFIX_DEFAULT
                        system_suffix = _regen_suffix.replace(
                            "{{domain_tag}}", _round_language or "software"
                        )
                        prefill = ""
                    # Build the system message.
                    sys_msgs = [{"role": "system", "content": system_suffix}]
                    # Build the user message.  The canonical harness
                    # retains its full-artifact context.  The focused
                    # variant uses bounded ranked sections for R1 and
                    # authoritative R1-selected sections for R2.
                    artifact_context_cfg = runtime.get("artifact_context", {})
                    context_mode = (
                        str(artifact_context_cfg.get("mode", "full"))
                        if isinstance(artifact_context_cfg, dict)
                        else "full"
                    )
                    full_context_max_chars = int(
                        artifact_context_cfg.get("full_context_max_chars", 48000)
                        if isinstance(artifact_context_cfg, dict)
                        else 48000
                    )
                    focused_context = context_mode == "focused_sections" or (
                        context_mode == "auto"
                        and len(target_artifact.content) > full_context_max_chars
                    )
                    context_query = f"{unit.prompt}\n{user_intent}".strip()
                    max_sections = int(artifact_context_cfg.get("max_sections", 8)) if focused_context else 8
                    neighbor_radius = int(artifact_context_cfg.get("neighbor_radius", 1)) if focused_context else 1
                    max_chars = int(artifact_context_cfg.get("max_chars", 48000)) if focused_context else 48000
                    if _round == 0:
                        if focused_context:
                            context = artifact_store.build_focused_context(
                                context_query,
                                artifact_id=resolved_artifact_id,
                                max_sections=max_sections,
                                neighbor_radius=neighbor_radius,
                                max_chars=max_chars,
                            )
                            user_content = (
                                f"{context}\n\n"
                                f"AUTHORITATIVE USER REQUEST:\n{unit.prompt}\n"
                            )
                        else:
                            outline = artifact_store.build_outline(resolved_artifact_id)
                            user_content = (
                                f"[ARTIFACT CONTENT]\n{target_artifact.content}\n\n"
                                f"[ARTIFACT STRUCTURE — tree-sitter outline]\n{outline}\n\n"
                                f"[END ARTIFACT CONTEXT]\n\n"
                                f"AUTHORITATIVE USER REQUEST:\n{unit.prompt}\n"
                            )
                    else:
                        # R2 (2026-09-09): uniform re-fetch, targeting-source
                        # agnostic. R1 may deliver a structured JSON package
                        # OR a demarcated 'final list' prose block -- both
                        # validated by the salvage contract, which stashes
                        # the verified forwarded names in
                        # runtime["_r1_targeting"]. The raw R1 output rides
                        # verbatim as the targeting header (names + reasons
                        # -- same model reads its own analysis back).
                        targeting = runtime.get("_r1_targeting") or {}
                        selected_names: list[str] = list(targeting.get("names", []))
                        if isinstance(r1_json, dict):
                            # JSON path: names also live in the parsed fields.
                            selected_names.extend(
                                item.get("name")
                                for item in r1_json.get("relevant_sections", [])
                                if isinstance(item, dict) and item.get("name")
                            )
                            selected_names.extend(
                                item.get("anchor_section")
                                for item in r1_json.get("new_sections", [])
                                if isinstance(item, dict) and item.get("anchor_section")
                            )
                            selected_names.extend(
                                item.get("name")
                                for item in r1_json.get("removals", [])
                                if isinstance(item, dict) and item.get("name")
                            )
                        context = ""
                        if selected_names:
                            context = artifact_store.build_named_context(
                                selected_names,
                                artifact_id=resolved_artifact_id,
                                max_chars=max_chars,
                            )
                        if not context:
                            # Nothing named (empty targeting) — fall back to
                            # query-ranked context so R2 still sees the
                            # neighborhood; the no_change_package rule
                            # handles the truly-empty work product.
                            context = artifact_store.build_focused_context(
                                context_query,
                                artifact_id=resolved_artifact_id,
                                max_sections=max_sections,
                                neighbor_radius=neighbor_radius,
                                max_chars=max_chars,
                            )
                        user_content = (
                            f"Targeting package from analysis:\n{r1_output}\n\n"
                            f"[SELECTED SECTION CONTENT]\n{context}\n\n"
                            f"AUTHORITATIVE USER REQUEST:\n{unit.prompt}\n"
                        )
                    if is_r2 and r2_retry_error:
                        user_content += (
                            "\n\nThe previous R2 edit attempt failed validation. "
                            "Produce a new edit package for the same request and "
                            "artifact, correcting the failure without changing "
                            "unrelated sections. Previous validation result: "
                            f"{r2_retry_error}\n"
                        )
                    if not is_r2 and r1_retry_error:
                        user_content += (
                            "\n\nThe previous R1 analysis attempt failed validation. "
                            "Re-analyze the same artifact and request. Remember: your "
                            "job is ONLY to identify sections to modify or delete and "
                            "anchor sections for new sections -- do NOT rewrite any "
                            "code yourself, and do NOT produce the modified artifact; "
                            "that is left for the next round. Conclude your response "
                            "with the section list under a demarcation title of "
                            "'final list' -- section names and the reason each is "
                            "named -- as the last thing in your response. Previous "
                            "validation result: "
                            f"{r1_retry_error}\n"
                        )
                    round_messages = sys_msgs + [{"role": "user", "content": user_content}]
                    # Closed-think injection (2026-09-09, user think-policy
                    # ruling): think state is a MODEL-CARD property. A
                    # think-off slot model (every pruned specialist except
                    # ornith python-b) gets the closed-think block prepended
                    # to its response prefix — the assistant turn is
                    # pre-opened and immediately closed, so a think-capable
                    # model (gemma4 family) cannot open a thinking span and
                    # burn the budget on think-aloud prose (the 2026-09-09
                    # live failure: 686s of dithering because the two-round
                    # loop never injected the block). A think-on slot model
                    # (ornith) gets NO block and NO directive prefill —
                    # native reasoning. The harness round prefill
                    # (assistant_prefill) rides after the block when set;
                    # it is null for modify R1/R2 under the draft-then-final
                    # design, so the block alone is the prefix.
                    _is_primary_slot = ladder_cfg.get("model_slot") == "primary"
                    _slot_think = (
                        work.primary_think if _is_primary_slot else work.secondary_think
                    )
                    # 2026-09-09 (user ruling): raw-prefill tolerance is a
                    # model-card property, separate from think. A model with
                    # raw_prefill=False (gemma4 lineage) gets NO text prefix
                    # at all -- the closed-think block and any harness
                    # prefill would be dropped by the transport gate anyway,
                    # and injecting here would only be dropped again. Think
                    # suppression for such models rides the native
                    # /api/chat think flag (think:false in their config).
                    _slot_raw_prefill = (
                        work.primary_raw_prefill
                        if _is_primary_slot
                        else work.secondary_raw_prefill
                    )
                    _raw_prefill_ok = _slot_raw_prefill is not False
                    # 2026-09-29 (user ruling, Option B): the R1/R2 round's
                    # think state follows the SLOT MODEL's card, not the
                    # request override -- identical to the two resolution.py
                    # sites.  The old line (`... if think_override is None
                    # else think_override`) let /thinkon override the secondary
                    # slot's model card: the same defect that collapsed r25's
                    # contract success 13/20 -> 2/20 on identical qids in the
                    # LCB T1/T2 mirror test.
                    # Approved carve-outs: the GENERALIST slot may still be
                    # overridden; cloud is governed by its own
                    # meta_harness_ladder.cloud settings and never sees this.
                    # All three sites move together; a partial fix is worse
                    # than none.
                    if (think_override is not None
                            and ladder_cfg.get("model_slot") == "generalist"):
                        _round_think = think_override
                    else:
                        _round_think = _slot_think
                    if _round_think is True:
                        round_prefill = ""
                        round_closed = False
                    elif not _raw_prefill_ok:
                        # raw_prefill=False: native chat path owns think
                        # suppression; no text prefix of any kind.
                        round_prefill = ""
                    else:
                        round_closed = work.harness.closed_think
                        if round_closed is None:
                            round_closed = True
                        # The executor builds unit.closed_think_block from the
                        # effective closed_think flag (think>\n\n marker).
                        # Injection is keyed on the SLOT MODEL's think flag
                        # (False = think-off-capable model), not the harness
                        # flag alone: a think-on model in a closed_think
                        # harness must reason natively.
                        _closed_block = (
                            getattr(unit, "closed_think_block", None)
                            if _slot_think is False
                            else None
                        )
                        if prefill:
                            round_prefill = (
                                f"{_closed_block}{prefill}"
                                if _closed_block
                                else prefill
                            )
                        else:
                            round_prefill = _closed_block or ""
                    round_sampling = dict(work.harness.sampling)
                    if not is_r2 and r1_retry_count:
                        round_sampling["seed"] = secrets.randbits(31)
                    if is_r2 and r2_retry_count:
                        round_sampling["seed"] = secrets.randbits(31)
                    # Per-model think flag already computed above; the
                    # package carries it so the transport sends `think`.
                    round_package = CleanPackage(
                        model_id=work.primary_model if _is_primary_slot else work.secondary_model,
                        provider=work.primary_provider if _is_primary_slot else work.secondary_provider,
                        options=dict(work.primary_options if _is_primary_slot else work.secondary_options),
                        sampling=round_sampling,
                        messages=round_messages,
                        think=_round_think,
                        response_prefix=round_prefill or None,
                        images=list(unit.images or []),
                    )
                    round_package = _disable_response_prefix_for_images(round_package)
                    if raw_token_writer is not None:
                        try:
                            raw_token_writer.log_round_request(
                                tier=f"{tier}_{round_name}",
                                model_id=round_package.model_id,
                                provider=round_package.provider,
                                messages=round_package.messages,
                                sampling=round_package.sampling,
                                options=round_package.options,
                            )
                        except Exception:
                            pass
                    # Record the round start.
                    await emit(CleanProgressEvent(
                        event_type="tier_start",
                        tier=f"{tier}_{round_name}",
                        model_id=round_package.model_id,
                    ))
                    if raw_token_writer is not None:
                        try:
                            raw_token_writer.begin_tier(
                                f"{tier}_{round_name}", round_package.model_id,
                            )
                        except Exception:
                            pass
                    # Stream the model output for this round
                    # using the same streaming infrastructure
                    # as the single-round path.
                    round_output_parts: list[str] = []
                    round_error: str | None = None
                    round_aborted = False
                    round_accepted = False
                    started = _now()
                    try:
                        transport = transports[round_package.provider]
                        if round_package.provider == "ollama":
                            target_key = (round_package.model_id, int(round_package.options.get("num_ctx") or 0))
                            state = await vram.state()
                            same_model_resident = any(
                                resident.name == round_package.model_id or resident.name.split(":", 1)[0] == round_package.model_id.split(":", 1)[0]
                                for resident in state.residents
                            )
                            if last_prepared_key != target_key or not same_model_resident:
                                # Status tag: model load can take 15-240s.
                                # Announce BEFORE the load so the client
                                # shows live progress instead of dead air
                                # (2026-09-05: user cancelled a healthy
                                # request mid-load believing it hung).
                                await emit(CleanProgressEvent(
                                    event_type="status", tier=tier,
                                    model_id=round_package.model_id,
                                    metadata={"status_kind": "model_load"},
                                ))
                                await vram.prepare(round_package.model_id, round_package.options.get("num_ctx"))
                                last_prepared_key = target_key
                        async for chunk, is_thinking in transport.stream(round_package, pathology_monitor=None):
                            round_output_parts.append(chunk)
                            if raw_token_writer is not None:
                                try:
                                    await raw_token_writer.write_chunk(chunk)
                                except Exception:
                                    pass
                            await emit(CleanProgressEvent(
                                event_type="token", tier=f"{tier}_{round_name}", text=chunk,
                                metadata={"is_thinking": is_thinking},
                            ))
                            # ── Local pathology guards (2026-09-09, user
                            # ruling: guards ALWAYS active on local tiers) ──
                            # The two-round loop previously streamed with no
                            # guards at all — a 90KB think-aloud runaway ran
                            # to completion (686s) before the contract
                            # rejected it. Same evaluation contract as the
                            # single-tier path: every 16 chunks, run the
                            # harness-declared primitives over the
                            # accumulated text (minus the response prefix),
                            # abort the stream on a match. The guard error
                            # flows through the round's failure handler
                            # (retry with fresh seed → escalate).
                            if len(round_output_parts) % 16 == 0:
                                _guard_text = "".join(round_output_parts)
                                if round_prefill:
                                    _guard_text = _guard_text.replace(round_prefill, "")
                                for _gname in runtime.get("primitives", ()):
                                    if _gname == "dither_guard":
                                        _dres = dither_guard(
                                            _guard_text, runtime,
                                            think_on=bool(_round_think),
                                        )
                                        if _dres:
                                            round_error = _dres
                                            try:
                                                transport.request_close()
                                            except Exception:
                                                pass
                                            break
                                    elif _gname == "repetition_loop_guard":
                                        _lres = repetition_loop_guard(_guard_text, runtime)
                                        if _lres:
                                            round_error = _lres
                                            try:
                                                transport.request_close()
                                            except Exception:
                                                pass
                                            break
                                if round_error is not None:
                                    break
                    except Exception as exc:
                        from .transport import StreamAborted
                        if isinstance(exc, StreamAborted):
                            if not round_output_parts:
                                round_error = "first_token_timeout: no tokens within budget"
                            else:
                                round_error = f"slow_decode_timeout: no new token (len={len(round_output_parts)} chunks)"
                        else:
                            round_error = f"backend_error: {exc}"
                    except BaseException as exc:
                        if isinstance(exc, asyncio.CancelledError):
                            round_error = f"cancelled: outer wait_for fired with {len(round_output_parts)} chunks"
                        else:
                            round_error = f"base_exception: {type(exc).__name__}: {exc}"
                    round_output = "".join(round_output_parts)
                    if raw_token_writer is not None:
                        try:
                            raw_token_writer.end_tier(
                                elapsed_s=_now() - started,
                                output_len=len(round_output.encode("utf-8")),
                                error=round_error,
                                selected_model=round_package.model_id,
                                chunks=len(round_output_parts),
                            )
                        except Exception:
                            pass
                    calls += 1
                    # Parse the round's JSON output (handling
                    # prefill re-emission: the model may continue
                    # cleanly, fully re-emit the prefill, or
                    # partially re-echo the prefill's tail --
                    # ``_merge_prefill_continuation`` handles all
                    # three without producing a duplicated,
                    # invalid JSON fragment).
                    #
                    # CRITICAL: reassembly must use ``round_prefill``
                    # (the FULL text actually sent as the assistant
                    # message -- closed-think block + JSON opening),
                    # NOT the bare ``prefill`` (JSON opening only).
                    # Discovered via live Ollama testing: this model
                    # does not do true prefix-continuation -- it
                    # regenerates the whole assistant turn, so its
                    # continuation re-emits the closed-think block
                    # too.  Using the bare ``prefill`` as the
                    # reference finds no overlap with a continuation
                    # that starts with ``<think>\n</think>``, so the
                    # old code prepended the bare prefill onto an
                    # already-complete JSON response, corrupting it.
                    round_full = _merge_prefill_continuation(round_prefill, round_output) if round_error is None else round_output
                    round_json = _extract_json(round_full) if round_error is None else None
                    # Validate this round's output against the
                    # round's artifact contract.  The contract
                    # function receives the parsed JSON (or None)
                    # via the runtime dict's transient
                    # ``_round_json``/``_round1_json`` keys so it
                    # can check required fields and anchor names
                    # without re-parsing.
                    round_contract_name = str(round_cfg.get("artifact_contract", ""))
                    round_contract = CONTRACTS.get(round_contract_name)
                    if round_error is None and round_contract is not None:
                        contract_runtime = dict(runtime)
                        contract_runtime["_parsed_json"] = round_json
                        contract_runtime["_artifact_store"] = artifact_store
                        contract_runtime["_artifact_id"] = resolved_artifact_id
                        contract_runtime["_r1_json"] = r1_json
                        round_error = round_contract(round_full, contract_runtime)
                    # 2026-09-09 FIX (live UnboundLocalError): the
                    # under-delivery counters MUST be bound unconditionally
                    # at the top of every round iteration. The previous
                    # placement assigned them only inside the
                    # `round_error is None and is_r2 and round_json is not
                    # None` block, but the tier_complete metadata emit
                    # references them under a DIFFERENT guard
                    # (`round_json is not None`) — a round where
                    # round_error was set BEFORE the counter block (e.g.
                    # contract `malformed_json` on a bare-array response)
                    # left the names unbound and crashed the whole
                    # execute_runtime_plan, which also killed the T3
                    # escalation. Bind-then-override fixes both guards.
                    _r1_targeted = 0
                    _r2_acted = 0
                    if round_error is None and is_r2 and round_json is not None:
                        # 2026-09-08 no_change_package rule (local):
                        # an all-empty edit package is an ERROR only when
                        # R1's targeting package was itself empty — then
                        # the tier's work product is definitionally empty.
                        # When R1 targeted N>0 and R2 acts on fewer, that
                        # is legitimate final judgment (strong-recommendation
                        # semantics) — soft under-delivery, logged below.
                        if isinstance(r1_json, dict):
                            _r1_targeted = (
                                len(r1_json.get("relevant_sections", []) or [])
                                + len(r1_json.get("removals", []) or [])
                            )
                        # 2026-09-09: final-list mode counts scanned names.
                        targeting = runtime.get("_r1_targeting") or {}
                        if targeting.get("mode") == "final_list":
                            _r1_targeted = max(
                                _r1_targeted, len(targeting.get("names", []))
                            )
                        _r2_acted = (
                            len(round_json.get("replacements", []) or [])
                            + len(round_json.get("removals", []) or [])
                        )
                        if _r1_targeted == 0 and _r2_acted == 0 and not round_json.get("insertions"):
                            round_error = (
                                "no_change_package: empty targeting and empty edit "
                                "package — nothing to do"
                            )
                    if round_error is None and is_r2 and round_json is not None:
                        preview = artifact_store.apply_edits(
                            round_json,
                            artifact_id=resolved_artifact_id,
                            user_intent=user_intent,
                            source_filename=source_filename,
                            preview=True,
                        )
                        if preview.get("errors"):
                            round_error = (
                                "r2_candidate_validation_failed: "
                                + "; ".join(str(item) for item in preview["errors"])
                            )
                    _round_error = round_error
                    # Log the attempt for this round.
                    attempt = CleanAttempt(
                        f"{tier}_{round_name}",
                        round_package.model_id,
                        round_output,
                        _round_error is None and not round_aborted and not round_accepted,
                        _round_error,
                        _now() - started,
                    )
                    result.attempts.append(attempt)
                    await emit(CleanProgressEvent(
                        event_type="tier_complete",
                        tier=f"{tier}_{round_name}",
                        output=round_output,
                        model_id=round_package.model_id,
                        metadata={
                            "passed": _round_error is None,
                            "round_error": _round_error,
                            "raw_round_capture": raw_token_writer is not None,
                            # 2026-09-07: surface the capture path so a
                            # failed round's tokens are one click away
                            # (user request — no hunting in logs/raw).
                            **({"raw_capture_path": str(raw_token_writer.path)}
                               if raw_token_writer is not None
                               and hasattr(raw_token_writer, "path") else {}),
                            # 2026-09-08: under-delivery observability.
                            # R1 targeted N sections (modify+remove); R2
                            # acted on M. M < N is legitimate (R2's final
                            # judgment) but visible here so silent
                            # under-delivery is diagnosable.
                            **(
                                {"r1_targeted": _r1_targeted, "r2_acted": _r2_acted}
                                if is_r2 and round_json is not None
                                else {}
                            ),
                        },
                    ))
                    if _round_error is not None:
                        # A failed R1 gets one focused reroll with a
                        # fresh seed before the model slot is exhausted.
                        if not is_r2 and r1_retry_count == 0:
                            r1_retry_count = 1
                            r1_retry_error = _round_error
                            _round_error = None
                            continue
                        # (2026-09-08) R1-b warm-schema rescue DELETED per
                        # user ruling: R1 -> R1_retry -> R2 -> secondary
                        # model -> cloud. No same-warm remap iteration.
                        # A failed R2 gets one focused reroll. R1's
                        # targeting package remains authoritative.
                        if is_r2 and r2_retry_count == 0:
                            r2_retry_count = 1
                            r2_retry_error = _round_error
                            _round_error = None
                            continue
                        prior_error = f"{round_name}_failed: {_round_error}"
                        break
                    if _round == 0:
                        r1_output = round_full
                        r1_json = round_json
                    else:
                        r2_output = round_full
                        r2_json = round_json
                    _round += 1
                # === END OF TWO-ROUND STREAM LOOP ===
                if _round_error is not None:
                    # Functional error in R1 or R2.  Escalate to
                    # the next tier.  The round error becomes the
                    # tier's functional error.
                    functional_fail_count += 1
                    break
                # Both rounds passed.  Apply the edits.
                if r2_json is None:
                    error = "r2_missing: no R2 JSON output (should not happen)"
                    functional_fail_count += 1
                    break
                # The apply step uses r2's apply_policy to
                # deterministically apply the edits to the
                # artifact.  This is a CODE-LEVEL operation
                # (no model call), using the ArtifactStore
                # primitives.
                apply_mechanism = r2_cfg.get("apply_policy", {}).get("apply_mechanism", "")
                if apply_mechanism != "modify_artifact_deterministic_reintegration":
                    error = f"unsupported_apply_mechanism: {apply_mechanism}"
                    functional_fail_count += 1
                    break
                # Step 5 correction (2026-08-04, freeze-before-artifact):
                # do NOT call artifact_store.apply_edits() here.  That
                # would commit the fork/promotion synchronously inside
                # this tier loop -- i.e. BEFORE the ``complete`` event
                # below, which is exactly what triggers SessionMemory's
                # freeze.  Instead, package the validated (R1+R2
                # contract-passed) edit for server.py to commit strictly
                # AFTER execute_runtime_plan() returns -- i.e. strictly
                # after the turn has already frozen.
                result.pending_artifact_edit = {
                    "edits": r2_json,
                    "artifact_id": resolved_artifact_id,
                    "user_intent": user_intent,
                    "source_filename": source_filename,
                }
                marker = f"[[modify_artifact: pending edit for artifact {resolved_artifact_id}]]"
                result.output = marker
                result.passed = True
                result.selected_model = round_package.model_id
                result.total_elapsed_s = _now() - run_start
                await emit(CleanProgressEvent(
                    event_type="complete", tier=tier, output=marker,
                    model_id=result.selected_model,
                    elapsed_s=result.total_elapsed_s,
                    metadata={"pending_artifact_edit": True},
                ))
                return result
            # === END OF TWO-ROUND EXECUTION PATH ===

            # v12 T1a patch-apply path: when the tier has
            # apply_policy with apply_mechanism ==
            # "modify_artifact_deterministic_reintegration",
            # the model emits a unified-diff patch that we
            # deterministically apply to T0's code block.
            # The patched code (not the patch) is the tier's
            # output and is validated against the
            # new_code_generation contract.
            ladder = work.harness.escalation_ladder
            tier_cfg_t1a = ladder.get(tier.lower(), {})
            apply_policy = tier_cfg_t1a.get("apply_policy")
            t1a_patch_apply = (
                apply_policy is not None
                and apply_policy.get("apply_mechanism") == "modify_artifact_deterministic_reintegration"
                and t0_output_for_t1 is not None
                and t0_error_for_t1 is not None
            )
            if t1a_patch_apply:
                # Extract T0's final code block.
                contract_name = tier_cfg_t1a.get(
                    "artifact_contract",
                ) or "unified_diff_patch"
                t0_code_block = _extract_final_code_block(
                    contract_name=str(contract_name),
                    output=t0_output_for_t1,
                    language=None,
                    runtime_cfg=runtime,
                )
                fence_language = _detect_fence_language(t0_output_for_t1)
                filled_modifier = str(tier_cfg_t1a.get("user_modifier", ""))
                if t0_code_block:
                    filled_modifier = filled_modifier.replace("{t0_code}", t0_code_block)
                filled_modifier = filled_modifier.replace("{t0_error}", t0_error_for_t1)
                filled_modifier = filled_modifier.replace("{{fence_language}}", fence_language)
                package = harness.compose(unit, tier, CleanTierContext(tier=tier, prior_error=prior_error))
                pkg_messages = package.messages[:]
                for i, msg in enumerate(pkg_messages):
                    if msg.get("role") == "user":
                        pkg_messages[i] = {"role": "user", "content": msg["content"] + "\n\n" + filled_modifier}
                        break
                package = CleanPackage(**{**package.__dict__, "messages": pkg_messages})
                print(f"[T1a] patch_target: t0_code ({len(t0_code_block) if t0_code_block else 0} chars)")
                print(f"[T1a] fence_language: {fence_language} (contract={contract_name})")
            else:
                # v12 T1 primitive: when on the T1 tier, compose
                # the package via ``_compose_t1b_prompt`` so the
                # YAML-driven ``t1_full_user_modifier`` template
                # is appended to the user message.  T1b is the
                # full re-emit path; the T1a fast-path (step 4b)
                # will be wired before this block when
                # ``t1_strategy == "targeted_then_full"``.
                # ``t0_output_for_t1`` / ``t0_error_for_t1`` are
                # captured at the end of each T0 attempt.
                if tier == "T1" and t0_output_for_t1 is not None and t0_error_for_t1 is not None:
                    package = _compose_t1b_prompt(
                        harness, t0_output_for_t1, t0_error_for_t1, unit,
                    )
                else:
                    package: CleanPackage = harness.compose(unit, tier, CleanTierContext(tier=tier, prior_error=prior_error))
            package = _disable_response_prefix_for_images(package)
            package = _with_seed_override(package, seed_override)
            # The response prefix is provider input, not assistant chat
            # history. It is also replayed once for live display below.
            _prefill_text = package.response_prefix or ""
            # v13 step 2 (2026-08-08): on a FUNCTIONAL retry
            # (functional_fail_count > 0), inject the error trace into
            # the user message so the model is grounded on why its
            # previous patch failed (wrong file, misaligned lines,
            # malformed, etc.).  This only fires on a retry, not the
            # first attempt.  The seed is kept the same (see the seed
            # logic) so the retry is a deterministic re-attempt with
            # better grounding, not a random re-roll.
            if functional_fail_count > 0 and prior_error:
                _err_trace = (
                    "\n\nYour previous patch attempt failed the contract check "
                    f"with this error: {prior_error}\n"
                    "The target file is identified in the <code> block header. "
                    "Emit a unified-diff patch against that EXACT file, using the "
                    "ABSOLUTE line numbers from the provided code. Do not change "
                    "the file path."
                )
                _pkg_messages = package.messages[:]
                for _i, _msg in enumerate(_pkg_messages):
                    if _msg.get("role") == "user":
                        _pkg_messages[_i] = {
                            "role": "user",
                            "content": (_msg.get("content") or "") + _err_trace,
                        }
                        break
                package = CleanPackage(**{**package.__dict__, "messages": _pkg_messages})
            # Path B remediation: inject artifact context (full
            # content + tree-sitter outline + BM25 section index)
            # into evaluate_artifact's user message.  This harness's
            # own system_suffix promises the model "an existing
            # artifact... a tree-sitter outline... a BM25 section
            # index" -- without this injection those never arrive.
            # Gated strictly on harness_id so no other harness's
            # compose is affected.  Fails gracefully (no injection,
            # tier proceeds without artifact context and will
            # likely fail its own contract) if no artifact_store
            # is configured or no artifact is selected, rather than
            # raising -- an evaluate_artifact call with no artifact
            # yet is a legitimate (if unusual) empty-analysis case.
            if work.harness.harness_id == "evaluate_artifact" and artifact_store is not None:
                eval_artifact = artifact_store.get(artifact_id)
                if eval_artifact is not None:
                    eval_outline = artifact_store.build_outline(eval_artifact.artifact_id)
                    eval_section_index = artifact_store.build_section_index_text(eval_artifact.artifact_id)
                    eval_context = (
                        f"[ARTIFACT CONTENT]\n{eval_artifact.content}\n\n"
                        f"[ARTIFACT STRUCTURE — tree-sitter outline]\n{eval_outline}\n\n"
                        f"[ARTIFACT SECTIONS — BM25 index]\n{eval_section_index}\n"
                        "[END ARTIFACT CONTEXT]\n"
                    )
                    eval_messages = list(package.messages)
                    for i, msg in enumerate(eval_messages):
                        if msg.get("role") == "user":
                            intent_context = ""
                            if user_intent:
                                intent_context = (
                                    "\n\n[ROUTER INTENT SUMMARY — NON-AUTHORITATIVE]\n"
                                    f"{user_intent}\n"
                                    "Treat the user's request as authoritative. "
                                    "Use this summary only to focus the analysis; "
                                    "do not repeat it as a task instruction.\n"
                                    "[END ROUTER INTENT SUMMARY]\n"
                                )
                            eval_messages[i] = {
                                "role": "user",
                                "content": (
                                    eval_context
                                    + intent_context
                                    + "\n[AUTHORITATIVE USER REQUEST]\n"
                                    + msg["content"]
                                ),
                            }
                            break
                    package = CleanPackage(**{**package.__dict__, "messages": eval_messages})
                elif evaluate_store is not None and evaluate_store.has():
                    # v14: no stored artifact, but the evaluate holding pen
                    # has a source object (attached/pasted/prior-WP).  Fall
                    # back to that source so a follow-up "fix the bugs you
                    # found" has the context even when the source was not a
                    # stored artifact.
                    _pen = evaluate_store.get()
                    _pen_src = (_pen or {}).get("source_content") or ""
                    if _pen_src:
                        eval_context = (
                            f"[ARTIFACT CONTENT]\n{_pen_src}\n\n"
                            "[END ARTIFACT CONTEXT]\n"
                        )
                        eval_messages = list(package.messages)
                        for i, msg in enumerate(eval_messages):
                            if msg.get("role") == "user":
                                eval_messages[i] = {
                                    "role": "user",
                                    "content": (
                                        eval_context
                                        + "\n[AUTHORITATIVE USER REQUEST]\n"
                                        + msg["content"]
                                    ),
                                }
                                break
                        package = CleanPackage(**{**package.__dict__, "messages": eval_messages})
            # ============================================================
            # 2026-09-14 (spec 5.7.2 PATH A): LOCAL REGENERATION
            # ============================================================
            # When the artifact carries its originating prompt, the request is
            # served by REGENERATING the artifact from a combined prompt rather
            # than by asking the local specialist to produce a change order
            # (the contract these models demonstrably fail).  The combined
            # prompt is built in server.py and arrives here via the runtime
            # dict; the change-order context injection below is SKIPPED, and
            # no change order is expected or validated -- so the pending-edit
            # packaging downstream never runs for this path.
            # 2026-09-16 (M2, gate 1): property, not harness name.  PATH A
            # swaps the contract but keeps running on the harness that reads
            # the artifact store; a harness that does NOT (new_code_generation,
            # general_use) must never enter this branch.
            if (
                getattr(work.harness, "needs_artifact_store", False)
                and runtime.get("modify_combo_prompt")
            ):
                _combo = str(runtime["modify_combo_prompt"])
                print(
                    f"[modify] PATH A regeneration: combo prompt "
                    f"{len(_combo)} chars (change order NOT requested)",
                    file=sys.stderr, flush=True,
                )
                # --- System message swap (2026-09-14, spec 4.1) ---------
                # CRITICAL: the harness's declared system_suffix is the
                # CHANGE-ORDER instruction ("Return ONLY a JSON object ...
                # Never return the artifact or modified document itself"),
                # and its output_instruction says the same again.  On the
                # regeneration path that is exactly backwards: the model is
                # asked to RETURN the artifact.  Leaving the change-order
                # wording in place makes the model emit JSON, which the
                # swapped denovo contract then correctly rejects -- a
                # guaranteed failure.  So the system message is rebuilt
                # from the regeneration cell's wording, which is
                # config-owned (`runtime.modify_regeneration_system_suffix`)
                # with an in-code fallback.
                #
                # NOTE: this override MUST live here (the live single-tier
                # path), NOT only in the two-round block above -- that block
                # is dormant for modify_artifact (its `rounds:` key is
                # deliberately absent), so an override placed only there
                # would never execute.
                if runtime.get("_regen_is_prose"):
                    # 2026-09-17: prose wording -- no fence instruction.
                    _regen_suffix = str(
                        runtime.get("modify_regeneration_prose_system_suffix")
                        or ""
                    )
                    if not _regen_suffix:
                        _regen_suffix = _PROSE_REGEN_SUFFIX_DEFAULT
                else:
                    _regen_suffix = str(
                        runtime.get("modify_regeneration_system_suffix") or ""
                    )
                    if not _regen_suffix:
                        _regen_suffix = _CODE_REGEN_SUFFIX_DEFAULT
                _regen_lang = str(getattr(work, "domain", "") or "").removeprefix(
                    "domain_"
                )
                _regen_suffix = _regen_suffix.replace(
                    "{{domain_tag}}", _regen_lang or "software"
                )
                if runtime.get("_regen_is_prose"):
                    # 2026-09-17: the code output_instruction repeats the fence
                    # demand ("in a single fenced code block"), which is exactly
                    # wrong for prose.
                    _regen_out = str(
                        runtime.get(
                            "modify_regeneration_prose_output_instruction"
                        ) or ""
                    )
                else:
                    _regen_out = str(
                        runtime.get(
                            "modify_regeneration_output_instruction"
                        ) or ""
                    )
                _regen_sys_parts: list[str] = []
                if work.harness.harness_id != "evaluate_artifact" and work.persona:
                    _regen_sys_parts.append(str(work.persona))
                _regen_sys_parts.append(_regen_suffix)
                if _regen_out:
                    _regen_sys_parts.append(_regen_out)
                _regen_sys = "\n\n".join(
                    p.strip() for p in _regen_sys_parts if p.strip()
                )
                # Replace the user turn with the combined prompt, so the model
                # receives the build request + prior deltas + current request,
                # and produces a COMPLETE artifact.  No artifact context is
                # injected: regeneration does not need the old file, and
                # sending it invites a patch-style reply.
                _regen_messages = list(package.messages)
                for _ri, _rmsg in enumerate(_regen_messages):
                    if _rmsg.get("role") == "system":
                        _regen_messages[_ri] = {"role": "system", "content": _regen_sys}
                    elif _rmsg.get("role") == "user":
                        _regen_messages[_ri] = {"role": "user", "content": _combo}
                package = CleanPackage(**{**package.__dict__, "messages": _regen_messages})
            # 2026-09-10: one-shot modify context injection (user ruling).
            # The T3 dispatcher builds [ARTIFACT CONTENT] + [tree-sitter
            # outline] + [AUTHORITATIVE USER REQUEST] itself; the generic
            # single-tier path sends only unit.prompt -- so one-shot local
            # modify models were flying blind (T0 invented anonymous:1/
            # anonymous:2 IDs; T2: "I cannot see the actual file"). This
            # block mirrors the T3 assembly verbatim (minus BM25, dropped
            # by user ruling) for the one-shot modify harness. Focused
            # mode for oversized artifacts rides artifact_context config
            # via the focused-context builder; full content otherwise.
            # 2026-09-16 (gate 2): `local_change_order` -- the LOCAL tier's
            # contract, which is what determines whether the model needs the
            # artifact's section IDs.
            #
            # Full-artifact injection exists to give a change-order model the
            # section IDs it must anchor on.  This is a LOCAL-TIER decision:
            # `modify_harness` (PATH A) generates denovo and declares
            # local_change_order=false, so it is skipped structurally rather
            # than by a flag that must be remembered per request.  The
            # `modify_skip_reintegration` conjunct is retained for now so the
            # predicate is an exact drop-in on today's single-harness config.
            if (
                getattr(work.harness, "local_change_order", False)
                and artifact_store is not None
                and not runtime.get("modify_skip_reintegration")
            ):
                _mod_target = artifact_store.get(artifact_id)
                if _mod_target is not None:
                    _mod_context_cfg = runtime.get("artifact_context", {})
                    _mod_full_cap = int(
                        _mod_context_cfg.get("full_context_max_chars", 64000)
                        if isinstance(_mod_context_cfg, dict)
                        else 64000
                    )
                    _mod_outline = artifact_store.build_outline(_mod_target.artifact_id)
                    if len(_mod_target.content) > _mod_full_cap:
                        # Oversized artifact: partial context via the
                        # focused builder (BM25-ranked, capped), plus a
                        # user-visible note that fidelity is reduced.
                        _mod_context = artifact_store.build_focused_context(
                            f"{unit.prompt}\n{user_intent}".strip(),
                            artifact_id=_mod_target.artifact_id,
                            max_sections=8,
                            neighbor_radius=1,
                            max_chars=_mod_full_cap,
                        )
                        print(
                            f"[modify] artifact exceeds local context cap "
                            f"({len(_mod_target.content)} chars > {_mod_full_cap}); "
                            f"using ranked partial context",
                            file=sys.stderr, flush=True,
                        )
                    else:
                        _mod_context = (
                            f"[ARTIFACT CONTENT]\n{_mod_target.content}\n\n"
                            f"[ARTIFACT STRUCTURE — tree-sitter outline]\n{_mod_outline}\n"
                            "[END ARTIFACT CONTEXT]\n"
                        )
                    _mod_messages = list(package.messages)
                    for _mi, _msg in enumerate(_mod_messages):
                        if _msg.get("role") == "user":
                            _mod_messages[_mi] = {
                                "role": "user",
                                "content": (
                                    _mod_context
                                    + "\n[AUTHORITATIVE USER REQUEST]\n"
                                    + (_msg.get("content") or "")
                                ),
                            }
                            break
                    package = CleanPackage(**{**package.__dict__, "messages": _mod_messages})
            # 2026-09-10 (user ruling): modify-only output cap guard needs
            # the artifact size at check time.  Stash it in the runtime dict
            # so modify_output_cap_guard can compute the cap from config
            # without a new plumbing parameter.
            #
            # 2026-09-14 (user ruling): DENOVO GENERATION IS NOT CAPPED.
            # This binding was UNCONDITIONAL on any modify_artifact request,
            # so PATH A -- which produces a COMPLETE ARTIFACT, not a change
            # order -- was also handed an artifact size, and the cap closed
            # its stream just past the artifact's own length.  That is the
            # live regression (7,183/7,229/7,224 chars against a 7,155-char
            # artifact), with the cloud silently covering the failure.
            #
            # `modify_skip_reintegration` is precisely the PATH A /
            # regeneration marker, so gating on it binds the artifact size
            # ONLY when a change order is actually being produced -- which
            # is the case the cap was written for.  The guard checks the
            # zero binding as its second line of defence.
            # 2026-09-16 (gate 5): `local_change_order` -- the cap is relative
            # to a LOCAL change order, so a denovo harness must not bind it.
            # `modify_harness` (PATH A) declares local_change_order=false, which
            # retires the `modify_skip_reintegration` conjunct structurally.  It
            # is retained for now so this stays an exact drop-in.
            if (
                getattr(work.harness, "local_change_order", False)
                and artifact_store is not None
                and not runtime.get("modify_skip_reintegration")
            ):
                _cap_target = artifact_store.get(artifact_id)
                runtime["_output_cap_artifact_chars"] = (
                    len(_cap_target.content) if _cap_target is not None else 0
                )
            else:
                runtime["_output_cap_artifact_chars"] = 0
            # On the first tier, optionally inject prior-turn
            # context as a labeled section in the system message.
            # V2: uses build_memory_section() which integrates the
            # chunk index (Layer 3), turn summary index (Layer 2),
            # and heuristic memory relevance gate.  Falls back to
            # V1 build_last_turn_section() when V2 indexes are not
            # yet initialized (first turn or no frozen entries).
            if (
                tier_idx == 0
                and inject_last_turn
                and session_memory is not None
            ):
                section = session_memory.build_memory_section(unit.prompt)
                if section:
                    new_messages: list[dict[str, str]] = []
                    for msg in package.messages:
                        if msg.get("role") == "system":
                            new_messages.append({
                                "role": "system",
                                "content": (msg.get("content") or "") + section,
                            })
                        else:
                            new_messages.append(msg)
                    if not any(m.get("role") == "system" for m in package.messages):
                        new_messages.insert(0, {"role": "system", "content": section.lstrip()})
                    package = CleanPackage(**{**package.__dict__, "messages": new_messages})
                    # V2: capture the system-modified prompt so the
                    # session memory's artifact log records what
                    # was actually sent to the model (including the
                    # injected memory section).
                    if session_memory is not None:
                        try:
                            session_memory._in_flight.system_modified_prompt = (
                                "\n".join(
                                    m.get("content", "")
                                    for m in new_messages
                                    if m.get("role") == "system"
                                )
                            )
                        except Exception:
                            pass
            # 2026-09-10 (user ruling): inject the output_cap modifier on
            # the immediate next attempt after a cap trip.  One-shot: the
            # flag is cleared immediately after consuming it here.
            if output_cap_reroll_now:
                output_cap_reroll_now = False
                _cap_directive = str(
                    _resolve_cfg(runtime, "re_roll_modifiers.output_cap") or ""
                ).strip()
                if _cap_directive:
                    _cap_messages = []
                    for _msg in package.messages:
                        if _msg.get("role") == "user":
                            _cap_messages.append({
                                "role": "user",
                                "content": (
                                    (_msg.get("content") or "")
                                    + "\n\n" + _cap_directive
                                ),
                            })
                        else:
                            _cap_messages.append(_msg)
                    package = CleanPackage(**{**package.__dict__, "messages": _cap_messages})
            # v12: prepend the concision directive on a
            # dither that caused a re-roll.  The trigger
            # is ``pathology_count > 0`` (any pathology
            # that caused a re-roll) and ``prior_error ==
            # "dither"``.  The directive is one-shot: it
            # is prepended to the IMMEDIATE next attempt
            # only, then cleared via ``prior_error = None``
            # in the re-roll block below.
            dither_directive = runtime.get("dither_retry_system_modifier")
            if (
                dither_directive
                and pathology_count > 0
                and prior_error == "dither"
            ):
                directive = str(dither_directive).strip()
                if directive:
                    new_messages = []
                    for msg in package.messages:
                        if msg.get("role") == "system":
                            new_messages.append({
                                "role": "system",
                                "content": directive + "\n\n" + (msg.get("content") or ""),
                            })
                        else:
                            new_messages.append(msg)
                    if not any(m.get("role") == "system" for m in package.messages):
                        new_messages.insert(0, {"role": "system", "content": directive})
                    package = CleanPackage(**{**package.__dict__, "messages": new_messages})
            # Capture the first tier's system-modified prompt
            # for the session memory entry.
            if first_system_modified is None:
                for msg in package.messages:
                    if msg.get("role") == "system":
                        first_system_modified = msg.get("content") or ""
                        break
            # v12 per-tier seed policy.  See V12_PLAN.md
            # section 9 for the full design.
            #
            # 4.3: Data-driven seed policy from the escalation_ladder.
            # The ladder's <tier>.seed_policy declares what each tier
            # does with the seed:
            #   - "fresh": new random seed on pathology re-roll;
            #     harness default on first call.
            #   - "same_as_t0": inherit the T0 seed (stored at
            #     tier_seed[0] or the first tier's index).
            #   - "same_as_t0_t2": inherit the T2 seed (stored at
            #     the T2 tier's index).
            # The old positional logic (tier_idx == 0 → fresh,
            # tier_idx == 1 → inherit, tier_idx == len-1 → fresh)
            # is gone.
            if runtime.get("retry_new_seed"):
                ladder = work.harness.escalation_ladder
                tier_cfg = ladder.get(tier.lower(), {})
                seed_policy = tier_cfg.get("seed_policy", "fresh")

                if seed_policy == "same_as_t0":
                    # Inherit the T0 seed (stored at tier_seed[0]
                    # or the first tier's index).
                    new_seed = tier_seed.get(0, package.sampling.get("seed"))
                elif seed_policy == "same_as_t0_t2":
                    # Inherit T2's seed (stored at the T2 tier's
                    # index).  Find the first tier on the secondary
                    # model slot — that's T2.
                    t2_idx = None
                    for i, t in enumerate(work.harness.tiers):
                        t_ladder_cfg = ladder.get(t.lower(), {})
                        if t_ladder_cfg.get("model_slot") == "secondary":
                            t2_idx = i
                            break
                    new_seed = tier_seed.get(t2_idx, package.sampling.get("seed")) if t2_idx is not None else package.sampling.get("seed")
                else:  # "fresh"
                    if output_cap_reroll_now:
                        # 2026-09-10 (user ruling): a modify output-cap
                        # re-roll always gets a FRESH seed, independent of
                        # the shared pathology counter.
                        new_seed = secrets.randbits(31)
                    elif pathology_count > 0:
                        # Pathology re-roll on the same model: new
                        # random seed.  v13 step 2: a FUNCTIONAL retry
                        # keeps the SAME seed (deterministic re-attempt
                        # with the error trace as grounding), so only a
                        # pathology re-roll randomizes.
                        new_seed = secrets.randbits(31)
                    else:
                        # Fresh start or functional retry: harness
                        # default seed (functional retry keeps the
                        # same seed).
                        new_seed = package.sampling.get("seed")

                if new_seed is not None:
                    package = CleanPackage(**{
                        **package.__dict__,
                        "sampling": {**package.sampling, "seed": new_seed},
                    })
                # Record the seed so subsequent tiers that inherit
                # from this tier can find it.
                tier_seed[tier_idx] = new_seed
            started = _now()
            output_parts: list[str] = []
            visible_output_parts: list[str] = []
            error: str | None = None
            aborted = False
            accepted = False
            try:
                transport = transports[package.provider]
                # Truthful cold/warm classification (2026-09-07): the
                # FIRST forward pass after a load pays KV allocation +
                # kernel warm-up + lazy mmproj (vision) load -- costs
                # absent from the warm rate table.  Capture residency
                # BEFORE prepare() so an attempt that just loaded the
                # model budgets at the COLD rate (the pre-existing
                # warm/cold delta machinery then produces a generous
                # budget) instead of being killed by a steady-state
                # warm budget (observed: predicted=5.05s kill on a
                # fresh load, actual_ttft=0.39s on the warm reroll).
                was_resident_before_prepare: bool | None = None
                performed_prepare = False
                if package.provider == "ollama":
                    target_key = (package.model_id, int(package.options.get("num_ctx") or 0))
                    state = await vram.state()
                    same_model_resident = any(
                        resident.name == package.model_id or resident.name.split(":", 1)[0] == package.model_id.split(":", 1)[0]
                        for resident in state.residents
                    )
                    was_resident_before_prepare = bool(same_model_resident)
                    if last_prepared_key != target_key or not same_model_resident:
                        # Status tag: model load can take 15-240s. Announce
                        # BEFORE the load; the load itself is inside
                        # prepare()'s own deadline, but nothing else
                        # surfaces while it runs.
                        await emit(CleanProgressEvent(
                            event_type="status", tier=tier,
                            model_id=package.model_id,
                            metadata={"status_kind": "model_load"},
                        ))
                        try:
                            await vram.prepare(package.model_id, package.options.get("num_ctx"))
                            performed_prepare = True
                        except PrepareTimeout as exc:
                            # Black-box deadline fired: the model load
                            # wedged.  Classified as pathology so the
                            # ladder rerolls/escalates instead of the
                            # server's 330s client watchdog silently
                            # killing the stream with no internal record.
                            # We set `error` and FALL THROUGH to the same
                            # pathology classification the other guards
                            # use (do NOT continue here): that path
                            # increments pathology_count, honors the
                            # per-model cap, breaks to the next slot on
                            # exhaustion, and -- critically -- the `continue`
                            # at the bottom of the loop re-enters with a
                            # FRESH SEED because pathology_count > 0.
                            error = f"prepare_timeout: {exc}"
                            output_parts = []
                            # Jump past the streaming body by aborting the
                            # transport loop before it starts: emit a
                            # synthetic empty stream via a short-circuit.
                            # Simplest correct path: mark aborted and let
                            # the attempt be recorded as failed.
                            aborted = False
                            # Build the attempt record now; the normal
                            # post-loop classification below will run.
                            text = ""
                            # Skip the streaming block by raising into
                            # the existing except-handler?  No: simplest
                            # is to record the attempt and continue.
                            attempt = CleanAttempt(
                                tier, package.model_id, "", False, error, _now() - started
                            )
                            result.attempts.append(attempt)
                            pathology_count += 1
                            print(
                                f"[pathology] model={package.model_id} err={error}",
                                file=sys.stderr, flush=True,
                            )
                            if pathology_count > max_re_rolls:
                                prior_error = error
                                break
                            # Reroll: loop continues at the top, where
                            # pathology_count > 0 forces a fresh seed.
                            continue
                        last_prepared_key = target_key
                for_guard = [GUARDS[name] for name in runtime.get("primitives", ()) if name in GUARDS]
                # VRAM watchdog: peer background task.  Polls
                # /api/ps every 1.5s, sets ``vram_breach`` on a
                # budget overrun.  Treated as a correctness-
                # tier error (escalates to the next tier on
                # retry exhaustion).
                vram_breach: asyncio.Event | None = None
                watchdog_task: asyncio.Task | None = None
                if package.provider == "ollama":
                    vram_breach = asyncio.Event()
                    tier_budget_gb = float(getattr(vram, "usable_gb", 15.8))
                    async def _vram_watchdog() -> None:
                        assert vram_breach is not None
                        while not vram_breach.is_set():
                            try:
                                state = await vram.state()
                            except Exception:
                                await asyncio.sleep(1.5)
                                continue
                            if state.resident_gb > tier_budget_gb:
                                vram_breach.set()
                                return
                            await asyncio.sleep(1.5)
                    watchdog_task = asyncio.create_task(_vram_watchdog())
                # No/slow token pathology timer: peer
                # background task.  Ticks every 5s, checks
                # ``PathologyMonitor``-driven no/slow token
                # timer.  The monitor computes a dynamic
                # TTFT budget from the prompt size and the
                # measured prefill rate for the target
                # model (cold or warm), doubled by a safety
                # factor.  Post-TTFT, it watches the decode
                # rate over a rolling window and fires if
                # the model goes silent or sustains a
                # pathologically low rate.
                #
                # The monitor is consulted from BOTH paths:
                #   1. The per-chunk loop below, so a chunk
                #      arrival can fire pathology
                #      immediately when the budget is
                #      exceeded.
                #   2. A peer background task (``_starvation_timer``)
                #      that polls every 2s.  This is the
                #      only path that fires when ZERO
                #      chunks ever arrive (the per-chunk
                #      loop is parked on the network read
                #      in that case and cannot check the
                #      monitor itself).  The background
                #      task calls ``transport.request_close()``
                #      which raises ``StreamAborted`` in
                #      the per-chunk loop on the next
                #      iteration.
                starvation: asyncio.Event | None = None
                starvation_task: asyncio.Task | None = None
                stall_monitor: StallMonitor | None = None
                last_token_at: float = _now()
                first_token_budget: float = first_token_timeout_s
                # ``is_resident`` re-derived here for non-ollama
                # providers; for ollama the TRUTHFUL pre-prepare
                # residency (captured above, before any load) wins --
                # an attempt that just loaded the model is budgeted
                # at the cold rate regardless of post-load VRAM.
                is_resident: bool = False
                if package.provider == "ollama":
                    if was_resident_before_prepare is None:
                        was_resident_before_prepare = False
                    is_resident = (
                        was_resident_before_prepare and not performed_prepare
                    )
                    # Prefill status tag: the load is done and the
                    # request bytes are about to go on the wire --
                    # vision-encode + prefill can take seconds on
                    # image requests; surface it so the client tag
                    # moves from [loading model] to [prefill].
                    await emit(CleanProgressEvent(
                        event_type="status", tier=tier,
                        model_id=package.model_id,
                        metadata={"status_kind": "prefill"},
                    ))
                monitor: PathologyMonitor | None = None
                if package.provider == "ollama":
                    starvation = asyncio.Event()
                    # Truthful cold/warm budget: is_resident was set
                    # above from pre-prepare residency (the state the
                    # model was in when THIS attempt began) -- no
                    # post-load VRAM re-derivation here, which used to
                    # misclassify a fresh load as warm and kill the
                    # first pass with a steady-state budget.
                    # Build the pathology monitor.  The
                    # monitor owns the TTFT budget
                    # computation; we feed it the prompt
                    # token estimate and the
                    # cold/warm state.
                    prompt_tokens_estimate = _estimate_package_tokens(package)
                    safety = float(
                        runtime.get(
                            "pathology_safety_factor",
                            DEFAULT_SAFETY_FACTOR,
                        )
                    )
                    monitor = PathologyMonitor.from_table_file(
                        model_id=package.model_id,
                        is_resident=is_resident,
                        prompt_tokens_estimate=prompt_tokens_estimate,
                        safety_factor=safety,
                    )
                    # Rate-stall gate (sibling monitor).  Enabled via
                    # harness config ("stall.enabled"); disabled =>
                    # monitor is None and nothing below changes behavior.
                    if _resolve_cfg(runtime, "stall.enabled"):
                        stall_monitor = StallMonitor(
                            window_s=float(_resolve_cfg(runtime, "stall.window_s")),
                            arm_windows=int(_resolve_cfg(runtime, "stall.arm_windows")),
                            arm_min_chars=float(_resolve_cfg(runtime, "stall.arm_min_chars")),
                            zero_k=int(_resolve_cfg(runtime, "stall.zero_k")),
                            partial_ratio=float(_resolve_cfg(runtime, "stall.partial_ratio")),
                            partial_k=int(_resolve_cfg(runtime, "stall.partial_k")),
                            reset_ratio=float(_resolve_cfg(runtime, "stall.reset_ratio")),
                        )
                    local_stall = stall_monitor
                    # Capture the predicted budget so the
                    # per-chunk loop can build the right
                    # error string if the monitor fires.
                    first_token_budget = monitor._ttft_budget_s
                    local_transport = transport
                    local_starvation = starvation
                    local_monitor = monitor
                    # Records WHICH monitor tripped the starvation
                    # event.  The starvation event is shared by both
                    # the classic monitor (TTFT / slow_decode) and
                    # the rate-stall gate; when the timer fires, the
                    # per-chunk loop must know who fired so it builds
                    # the right error string.  Without this, a TTFT
                    # violation can be mislabeled as rate_stall (the
                    # classic monitor's verdict can change between
                    # the timer's trip and the loop's re-check).
                    starvation_source = {"who": None}
                    async def _starvation_timer() -> None:
                        """Peer background task: poll the
                        monitor on a 2s cadence and
                        request_close() if it returns an
                        abort verdict.

                        The 2s cadence (down from v7's
                        5s) is justified because the
                        monitor's TTFT budget can be as
                        short as 5-10s for warm small
                        prompts; a 5s polling interval
                        would mean pathology fires up
                        to 5s late.
                        """
                        assert local_starvation is not None
                        while not local_starvation.is_set():
                            await asyncio.sleep(2.0)
                            verdict = local_monitor.check()
                            if verdict.abort:
                                starvation_source["who"] = (
                                    "classic_ttft" if verdict.reason == "ttft"
                                    else "classic_slow_decode"
                                )
                                try:
                                    local_transport.request_close()
                                except Exception:
                                    pass
                                local_starvation.set()
                                return
                            # Rate-stall gate: same polling cadence.
                            # Fires on (a) a full 60s window with zero
                            # output mid-attempt ("rate_stall") or (b)
                            # 4 consecutive windows below 30% of the
                            # stream's own median active rate
                            # ("rate_collapse").  Only fires after the
                            # monitor is armed (>=1 active window post-
                            # first-token), so short/easy generations
                            # are never affected.
                            if local_stall is not None:
                                sv = local_stall.check()
                                if sv.abort:
                                    starvation_source["who"] = "stall"
                                    try:
                                        local_transport.request_close()
                                    except Exception:
                                        pass
                                    local_starvation.set()
                                    return
                    starvation_task = asyncio.create_task(_starvation_timer())
                try:
                    # Per-chunk loop body.  Existing
                    # dither / loop / vram_breach guards
                    # operate on accumulated text.  The
                    # pathology event is checked here too
                    # (peer of vram_breach).  When the
                    # pathology event fires, we break the
                    # loop with the right error string.
                    #
                    # The pathology monitor is consulted
                    # on every chunk arrival so a fresh
                    # budget violation is caught without
                    # waiting for the background timer.
                    # The background timer is the only
                    # path that catches the "no chunks
                    # at all" case (the per-chunk loop
                    # is parked on the network read in
                    # that case).
                    text: str = ""
                    now: float = _now()
                    # v13 step 2 (2026-08-08): echo the assistant
                    # prefill as a token event BEFORE this attempt's
                    # stream, so the client sees the full logical
                    # chain for EVERY attempt (initial + rerolls).
                    # Ollama does not re-echo the prefill in the
                    # streamed response -- it only streams the model's
                    # continuation.  Without this, a reroll's tokens
                    # appear with no prefill boundary, and the
                    # <patch> chain is unreadable.  The prefill is
                    # the last assistant message in the package.
                    #
                    # SKIP the echo for general_use: that harness
                    # returns raw output to the user, so echoing the
                    # prefill ("I am a helpful and competent
                    # assistant...") as visible content is wrong.
                    # Code/patch harnesses extract structured objects
                    # from the output and benefit from the prefill
                    # boundary in the live token stream.
                    if work.harness.harness_id != "general_use":
                        for _msg in package.messages:
                            if _msg.get("role") == "assistant" and _msg.get("content"):
                                await emit(CleanProgressEvent(
                                    event_type="token", tier=tier,
                                    text=_msg["content"],
                                    elapsed_s=_now() - run_start,
                                    model_id=package.model_id,
                                    metadata={"text_length": len(text), "prefill": True},
                                ))
                                break
                    # Pass the pathology monitor to the
                    # transport.  The transport starts the
                    # monitor's clock at the determinism
                    # boundary (just before httpx writes
                    # the request bytes), which is the
                    # last moment our process is fully in
                    # control.  If the model (or Ollama)
                    # does not start emitting tokens within
                    # the budget from that point, pathology
                    # is declared and we restart with a
                    # new seed.
                    async for chunk, is_thinking in transport.stream(package, pathology_monitor=monitor):
                        if stall_monitor is not None:
                            stall_monitor.on_chunk(len(chunk))
                        if monitor is not None:
                            monitor.on_token()
                            chunk_verdict = monitor.check()
                            if chunk_verdict.abort:
                                try:
                                    transport.request_close()
                                except Exception:
                                    pass
                                if chunk_verdict.reason == "ttft":
                                    error = (
                                        f"first_token_timeout: no tokens within "
                                        f"{chunk_verdict.budget_s:.0f}s of stream start"
                                    )
                                else:
                                    error = (
                                        f"slow_decode_timeout: decode rate "
                                        f"{chunk_verdict.observed_rate:.1f} tok/s below "
                                        f"{monitor.min_decode_rate:.0f} threshold"
                                    )
                                break
                        if starvation is not None and starvation.is_set():
                            # Background timer tripped.  Pull
                            # the error from the monitor so
                            # the message matches what the
                            # monitor saw.
                            bg_verdict = monitor.check() if monitor is not None else None
                            if bg_verdict is not None and bg_verdict.reason == "ttft":
                                error = (
                                    f"first_token_timeout: no tokens within "
                                    f"{bg_verdict.budget_s:.0f}s of stream start"
                                )
                            elif bg_verdict is not None and bg_verdict.reason == "slow_decode":
                                error = (
                                    f"slow_decode_timeout: decode rate "
                                    f"{bg_verdict.observed_rate:.1f} tok/s below "
                                    f"{monitor.min_decode_rate:.0f} threshold"
                                )
                            elif starvation_source["who"] == "stall" and stall_monitor is not None:
                                # The stall gate tripped the starvation
                                # event (recorded by the timer), so the
                                # classic monitor's current verdict is
                                # irrelevant -- attribute to the gate.
                                sv = stall_monitor.check()
                                if sv.abort and sv.reason == "rate_collapse":
                                    error = (
                                        f"rate_collapse: rate below "
                                        f"{stall_monitor.partial_ratio:.0%} of median for "
                                        f"{stall_monitor.partial_k} windows "
                                        f"(median active {sv.active_median_cps:.0f} c/s) "
                                        f"at t+{sv.elapsed_s:.0f}s"
                                    )
                                else:
                                    error = (
                                        f"rate_stall: {stall_monitor.zero_k}x{stall_monitor.window_s:.0f}s "
                                        f"window with zero output "
                                        f"(median active {sv.active_median_cps:.0f} c/s) "
                                        f"at t+{sv.elapsed_s:.0f}s"
                                    )
                            else:
                                if not output_parts:
                                    error = f"first_token_timeout: no tokens within {first_token_budget:.0f}s of stream start"
                                else:
                                    error = f"slow_decode_timeout: no new token within {slow_decode_timeout_s:.0f}s of last token (len={len(output_parts)} chunks)"
                            break
                        if vram_breach is not None and vram_breach.is_set():
                            error = "vram_breach: resident exceeded tier budget during generation"
                            break
                        now = _now()
                        last_token_at = now
                        output_parts.append(chunk)
                        if not is_thinking:
                            visible_output_parts.append(chunk)
                        text = "".join(output_parts)
                        # ── Output ceiling check (per-chunk) ──
                        # Per-attempt ceiling: one stream generating more
                        # than output_ceiling.max_chars without committing
                        # is a convergence failure.  Cumulative ceiling:
                        # the problem's total generated output across all
                        # attempts exceeding max_total_chars declares the
                        # work out of black-box scope.  Both are cheap
                        # len() comparisons refreshed every chunk.
                        if ceiling_enabled:
                            running_len = sum(len(p) for p in output_parts)
                            if running_len > ceiling_max_chars:
                                error = (
                                    f"output_ceiling_exceeded: attempt generated "
                                    f"{running_len} chars (ceiling {ceiling_max_chars}) "
                                    f"without commit"
                                )
                                break
                            # 2026-08-30: slot-aware cumulative ceiling.
                            # The PRIMARY slot reserves min_secondary_chars of
                            # headroom so the secondary always gets its held-out
                            # minimum budget; the SECONDARY slot uses the full
                            # max_total_chars.  Either way the global total never
                            # exceeds max_total_chars.
                            _cur_slot = tier_model_slots.get(tier, "primary")
                            _cum_ceiling = ceiling_total_max_chars
                            if (
                                _cum_ceiling is not None
                                and _cur_slot == "primary"
                                and ceiling_min_secondary_chars > 0
                            ):
                                _cum_ceiling = max(
                                    ceiling_min_secondary_chars,
                                    ceiling_total_max_chars - ceiling_min_secondary_chars,
                                )
                            if (
                                _cum_ceiling is not None
                                and cumulative_attempt_chars + running_len
                                > _cum_ceiling
                            ):
                                # LOCAL CONTRACT OVER: mark the trip.  The
                                # tier loop breaks below; further reroll
                                # attempts are ALSO skipped by the
                                # cumulative gate at the `while calls` top
                                # (the budget can never be un-met), and
                                # escalation happens at the meta level.
                                ceiling_cumulative_tripped = True
                                ceiling_tripped_slots.add(_cur_slot)
                                error = (
                                    f"output_ceiling_exceeded: problem cumulative "
                                    f"{cumulative_attempt_chars + running_len} chars "
                                    f"(ceiling {_cum_ceiling}) without commit"
                                )
                                break
                        # Live raw token write: append this
                        # chunk to ``raw/<qid>.raw.txt`` and
                        # fsync so the on-disk file is
                        # current at every token.  This is
                        # the local-context/debug save the
                        # clean slice is required to
                        # perform.  Errors are swallowed so
                        # a disk hiccup never breaks an
                        # inference.
                        if raw_token_writer is not None:
                            try:
                                await raw_token_writer.write_chunk(chunk)
                            except Exception:
                                pass
                        # Live token event.
                        await emit(CleanProgressEvent(
                            event_type="token", tier=tier, text=chunk, elapsed_s=now - run_start,
                            model_id=package.model_id,
                            metadata={"text_length": len(text), "is_thinking": is_thinking},
                        ))
                        # Per-chunk content guard check.
                        # The dither / loop guards are
                        # content-based and historically
                        # only ran AFTER the stream ended.
                        # But a runaway dither never ends
                        # the stream -- the model keeps
                        # emitting "wait", "actually",
                        # "let me reconsider" forever.
                        # Without this in-loop check, the
                        # guard is unreachable and the
                        # dither consumes the entire
                        # per-problem wall-clock budget.
                        # We check every Kth chunk (not
                        # every chunk) to keep the regex
                        # overhead bounded -- the dither
                        # pattern is fast but ``text``
                        # grows linearly.
                        if len(output_parts) % 16 == 0:
                            # v13 step 2 (2026-08-08): the model
                            # sometimes ECHOES the assistant prefill
                            # ("Let me work through this carefully")
                            # in its output.  The loop guard then
                            # false-triggers on the repeated prefill
                            # text, aborting a legitimate patch.  Strip
                            # the prefill from the text the guards see
                            # so they only evaluate the model's actual
                            # generated content.
                            _guard_text = text
                            if _prefill_text:
                                _guard_text = _guard_text.replace(_prefill_text, "")
                            for _gname in runtime.get("primitives", ()):
                                if _gname == "dither_guard":
                                    _dres = dither_guard(
                                        _guard_text, runtime,
                                        think_on=bool(getattr(package, "think", False)),
                                    )
                                    if _dres:
                                        error = _dres
                                        try:
                                            transport.request_close()
                                        except Exception:
                                            pass
                                        break
                                elif _gname == "repetition_loop_guard":
                                    _lres = repetition_loop_guard(_guard_text, runtime)
                                    if _lres:
                                        error = _lres
                                        try:
                                            transport.request_close()
                                        except Exception:
                                            pass
                                        break
                                elif _gname == "modify_output_cap_guard":
                                    _cres = modify_output_cap_guard(_guard_text, runtime)
                                    if _cres:
                                        error = _cres
                                        try:
                                            transport.request_close()
                                        except Exception:
                                            pass
                                        break
                            if error in ("dither", "loop", "output_cap_exceeded"):
                                break
                        # Per-chunk fence counter: every
                        # 25 chunks (about 0.5s at 50
                        # tok/s), check whether the
                        # ``python`` code fences are
                        # well-bounded.  The
                        # ``python_new_code`` harness
                        # contract says "exactly two
                        # ```python blocks: DRAFT and
                        # FINAL, no text outside the
                        # code blocks".  Any of the
                        # following is a generation
                        # pathology classified as
                        # ``post_code_fence_text`` and
                        # forces a stream close so the
                        # runtime does not waste
                        # wall-clock on the trailing
                        # tail:
                        #
                        #   1. Two or more fully-closed
                        #      ``python`` fences
                        #      (DRAFT + FINAL both
                        #      emitted -- stop here
                        #      so the model cannot
                        #      emit a 3rd block or
                        #      unbounded tail).
                        #
                        #   2. One closed ``python``
                        #      fence with > 1000
                        #      tokens of preamble
                        #      before its opening
                        #      fence (the model wrote
                        #      a wall of prose before
                        #      getting to the code).
                        #
                        #   3. One closed ``python``
                        #      fence with > 500
                        #      tokens of tail after
                        #      its closing fence
                        #      and no 2nd opening
                        #      fence (the model
                        #      rambled in prose
                        #      after the FINAL).
                        #
                        # The 25-chunk cadence keeps
                        # the regex overhead to a few
                        # ms per second; the
                        # accumulated text is at most
                        # a few KB when generation is
                        # healthy and the regex is
                        # anchored, so the cost is
                        # bounded.
                        #
                        # Token estimation: each
                        # chunk is up to 8 chars
                        # (the transport's
                        # ``_CHUNK_MAX_CHARS``)
                        # and roughly 1 token per
                        # chunk for English/code
                        # content.  We use
                        # ``chunks-since-event`` as
                        # the token proxy since it
                        # is exact (the runtime
                        # counts every chunk that
                        # arrives on the model
                        # stream) and avoids the
                        # chars-to-tokens guesswork.
                        if len(output_parts) % _fence_cadence == 0 and _fence_cadence > 0:
                            try:
                                _fence_positions = [m.start() for m in _fence_tag_re.finditer(text)]
                                _all_marks = [m.start() for m in re.finditer(r"```", text)]
                                _trigger_reason = None
                                _second_close_pos = None
                                if len(_fence_positions) >= 2 and len(_all_marks) >= 4:
                                    # Rule 1: 2 fences
                                    # closed (DRAFT +
                                    # FINAL).
                                    _second_close_pos = _all_marks[3]
                                    _trigger_reason = "two_closed_fences"
                                elif len(_fence_positions) == 1 and len(_all_marks) >= 2:
                                    # One fence opened
                                    # AND closed.
                                    _fence_open_offset = _fence_positions[0]
                                    _fence_close_offset = _all_marks[1]
                                    _running = 0
                                    _close_chunk_idx = -1
                                    for _ci, _cp in enumerate(output_parts):
                                        _running += len(_cp)
                                        if _running > _fence_close_offset:
                                            _close_chunk_idx = _ci
                                            break
                                    _chunks_since_close = (len(output_parts) - 1 - _close_chunk_idx) if _close_chunk_idx >= 0 else len(output_parts)
                                    if _fence_open_offset > 4000:
                                        _trigger_reason = "excess_preamble"
                                    elif _chunks_since_close > _fence_tail_threshold:
                                        _tail_text = text[_fence_close_offset + 3:]
                                        if not _fence_tag_re.search(_tail_text):
                                            _trigger_reason = "excess_tail"
                                if _trigger_reason is not None:
                                    if _second_close_pos is None:
                                        _second_close_pos = _all_marks[1]
                                    _clean_text = text[:_second_close_pos + 3]
                                    output_parts = [_clean_text]
                                    text = _clean_text
                                    error = "post_code_fence_text"
                                    try:
                                        transport.request_close()
                                    except Exception:
                                        pass
                                    break
                            except Exception:
                                pass
                        # Per-chunk patch block counter: every
                        # ``_patch_cadence`` chunks, count the
                        # number of closed ``</patch>`` tags.
                        # When ``max_blocks`` closes are observed,
                        # force-close the stream and trim
                        # ``output_parts`` to end at the last
                        # close so the runtime's downstream
                        # extract_last_patch_block() sees a clean
                        # boundary.
                        #
                        # No preamble/tail budgets — the model
                        # is allowed to write as much prose between
                        # patch blocks as it likes; dither and
                        # loop guards (run in the chunk % 16 block
                        # above) catch pathologies in the inter-patch
                        # prose.  The counter is purely a structural
                        # cap: "no more than N drafts".
                        #
                        # On overflow we do NOT set a pathology
                        # ``error`` (e.g. ``post_patch_block_overflow``)
                        # because re-rolling with a new seed is
                        # unlikely to fix a model that won't commit
                        # to a single patch after 3 tries.  The
                        # ``output`` simply passes through to the
                        # existing ``unified_diff_patch`` contract,
                        # which validates the LAST block.  If the
                        # contract fails, the tier fails normally;
                        # if it passes, the answer is the last
                        # block — exactly the desired semantics.
                        if len(output_parts) % _patch_cadence == 0 and _patch_cadence > 0 and _patch_max_blocks > 0:
                            try:
                                _close_positions = [m.end() for m in re.finditer(r"</patch>", text)]
                                if len(_close_positions) >= _patch_max_blocks:
                                    _last_close_end = _close_positions[_patch_max_blocks - 1]
                                    _clean_text = text[:_last_close_end]
                                    output_parts = [_clean_text]
                                    text = _clean_text
                                    try:
                                        transport.request_close()
                                    except Exception:
                                        pass
                                    break
                            except Exception:
                                pass
                    # 2026-09-19 (PLAN_progress_mechanism.md phase 7): the
                    # `summary` event was DELETED.  It was emitted here and
                    # consumed by NOTHING -- no handler in the streaming
                    # callback, no branch in _event_stream, no renderer in the
                    # CLI, no case in session_memory.on_event.  The `checkpoint`
                    # event below carries the SAME payload (text_so_far,
                    # elapsed_s, text_length) and is the single progress
                    # mechanism for both transports.
                    if now - last_checkpoint_at >= checkpoint_interval_s:
                        decision = await emit(CleanProgressEvent(
                            event_type="checkpoint", tier=tier, text_so_far=_summary_preview(text),
                            elapsed_s=now - run_start, model_id=package.model_id,
                            metadata={"text_length": len(text)},
                        ))
                        last_checkpoint_at = now
                        if decision == CleanControlDecision.ABORT:
                            result.aborted = True
                            aborted = True
                            break
                        if decision == CleanControlDecision.ACCEPT_PARTIAL:
                            result.accepted_partial = True
                            accepted = True
                            break
                    # Guard check.  Don't overwrite a
                    # deadline / vram_breach / starvation
                    # error that was already set inside the
                    # stream loop: the timer-driven errors
                    # take precedence over any chunk-driven
                    # guard verdict because the chunk-driven
                    # guards may not have had a chance to
                    # inspect the abort case.
                    error = error or next(
                        (guard(text, runtime) for guard in for_guard
                         if guard(text, runtime)),
                        None,
                    )
                finally:
                    if watchdog_task is not None:
                        watchdog_task.cancel()
                        try:
                            await watchdog_task
                        except (asyncio.CancelledError, Exception):
                            pass
                    if starvation_task is not None:
                        starvation_task.cancel()
                        try:
                            await starvation_task
                        except (asyncio.CancelledError, Exception):
                            pass
                output = "".join(output_parts)
                # Accumulate this attempt's generated output into the
                # problem-scoped total (drives the cumulative ceiling).
                cumulative_attempt_chars += len(output)
                # general_use returns the RAW generated output (no
                # object extraction), so a re-emitted assistant
                # prefill would leak into the user-facing answer.
                # Strip it here -- ONLY for general_use.  The other
                # harnesses extract a specific object (code block,
                # patch block, JSON) via their contract, so their
                # output is unaffected by a leading prefill and must
                # NOT be stripped (the contract may rely on the
                # prefill boundary for reassembly).
                if work.harness.harness_id == "general_use" and _prefill_text:
                    output = _strip_prefill(_prefill_text, output)
                if work.harness.harness_id == "general_use":
                    output = _strip_thinking_markup(output)
                    visible_output = _strip_thinking_markup("".join(visible_output_parts))
                else:
                    visible_output = output
                # new_code_generation: the prefill ends with a bare
                # opening fence (```\n```).  When the model re-emits
                # the prefill verbatim (pruned models ignore think:False),
                # the output has a doubled fence structure that the
                # contract's fence extraction mishandles.  Merge the
                # prefill into the output so the contract sees a single
                # well-formed fence structure.  _merge_prefill_continuation
                # handles full re-emission, partial overlap, and the
                # no-echo case (returns output unchanged).
                if work.harness.harness_id == "new_code_generation" and _prefill_text:
                    output = _merge_prefill_continuation(_prefill_text, output)
                # Auto-close any unclosed <patch> tag.
                open_patches = output.count("<patch>") - output.count("</patch>")
                if open_patches > 0:
                    output = output + ("</patch>" * open_patches)
                # v13 step 2 (2026-08-08): normalize newlines after
                # <patch> / </patch> tags so run-on lines don't merge a
                # hallucinated second block into the tail of the first
                # (which would defeat the context-correctness gate's
                # block regex).  Pure whitespace pass — never changes
                # tag text or block content.
                output = _normalize_patch_tag_newlines(output)
                # 4.5: T1a patch-then-apply — apply the patch
                # deterministically to the T0 code block using the
                # modify_artifact primitives at the code level.
                # The apply_mechanism for T1a is
                # "modify_artifact_deterministic_reintegration" (the
                # only mechanism we know how to apply).  The patch
                # is extracted from the model's output (a
                # `<patch>...</patch>` block) and applied to the
                # T0 code block (t0_code_block, captured earlier).
                #
                # If the patch is malformed or doesn't apply
                # cleanly, the tier fails with a functional error
                # (advances to T1b or the next tier).
                # If the patch applies and validates, the patched
                # code block is the tier's output.
                if t1a_patch_apply and apply_policy and t0_code_block:
                    patch_text = extract_last_patch_block(output)
                    if patch_text is None:
                        error = "empty_patch: no <patch> block found in T1a output"
                    else:
                        # Apply the patch deterministically to the
                        # T0 code block.
                        patched_code = apply_unified_diff(t0_code_block, patch_text)
                        if patched_code is None:
                            error = "patch_does_not_apply: unified diff failed to apply to T0 code block"
                        else:
                            # The patched code is the tier's output.
                            output = patched_code
                            # The output is now the patched code
                            # block, not the original patch.  This
                            # is what the user-facing result will be.
                            #
                            # We re-validate the patched code against
                            # the new_code_generation contract so
                            # the tier fails if the patched code
                            # has a syntax error or fence mismatch.
                            # The contract is "new_code_generation"
                            # (the same as T0's contract), not
                            # "unified_diff_patch".
                            error = None
                # 2026-09-10 (user-ruled refactor, single-owner design):
                # the modify_artifact round2 contract is SELF-SUFFICIENT
                # (own extraction + diff dialect + anchor validation +
                # preview), so the generic CONTRACTS dispatch is the single
                # validation site for the one-shot modify path. The wire-up
                # below hands the contract the artifact context it needs via
                # the runtime dict; on success it stashes the validated
                # package in runtime["_pending_artifact_edit"] which is
                # transferred to result.pending_artifact_edit right after.
                # 2026-09-16 (gate 3): `local_change_order` -- this wiring
                # supplies the LOCAL tier's change-order contract with the
                # artifact it must anchor against, meaningless without one.
                if getattr(work.harness, "local_change_order", False):
                    runtime["_artifact_store"] = artifact_store
                    runtime["_artifact_id"] = artifact_id
                    runtime["_user_intent"] = user_intent
                    runtime["_source_filename"] = source_filename
                    # 2026-09-11 (user-ruled): invented-replacement guard.
                    # Resolve the tier profile from the harness runtime
                    # config and hand the contract the whole block plus the
                    # executing tier so apply_edits' guard can select
                    # local/cloud thresholds.
                    _h_runtime = getattr(getattr(work, "harness", None), "runtime", None) or {}
                    _guard_cfg = dict(_h_runtime.get("invented_replacement_guard") or {})
                    _guard_cfg["_tier"] = str(tier)
                    runtime["_guard_cfg"] = _guard_cfg
                # 2026-09-14 (spec 5.7.2 PATH A): on the regeneration path the
                # output is a COMPLETE ARTIFACT, not a change order, so the
                # change-order contract would reject it (it demands JSON with
                # replacements/insertions/removals).  Validate the regenerated
                # artifact with the DENOVO contract instead, and mark the
                # result so server.py records method="regenerated" and does
                # NOT expect a pending_artifact_edit to apply.  Anything that
                # looks like a change order here would be a model error: it
                # means the model ignored the regeneration instruction.
                # 2026-09-16 (gate 1): property, not harness name.  The
                # denovo contract swap is the PATH A signature, and PATH A is
                # reached by reading the artifact store.  `modify_harness`
                # (PATH A as its DEFAULT route) declares
                # needs_artifact_store=True, so this fires for it too -- which
                # is why the swap had to become a property rather than an
                # in-cell key of a sibling harness.
                _is_regen = bool(runtime.get("modify_skip_reintegration")) and (
                    getattr(work.harness, "needs_artifact_store", False)
                )
                if _is_regen:
                    # 2026-09-17 (user ruling): LANGUAGE-DRIVEN, and nothing
                    # else.  The `modify_regeneration_contract` config key is
                    # RETIRED -- it named `new_code_generation` unconditionally,
                    # which is how a PROSE artifact came to be validated by a
                    # CODE contract (see _DENOVO_CONTRACT_BY_LANGUAGE).
                    #
                    # The parent is read through the `artifact_store` +
                    # `artifact_id` this function ALREADY receives, so selecting
                    # the contract needs no new executor/run kwarg.
                    # REVIEW FIX (2026-09-17): both keys were already SEEDED
                    # before the attempt loop -- the wording block above runs
                    # FIRST in this same iteration and would otherwise read an
                    # absent key on attempt 1.  Re-reading here keeps this
                    # diagnostic truthful without adding a second source of
                    # truth (same parent artifact, same value).
                    _regen_parent = (
                        artifact_store.get(artifact_id)
                        if artifact_store is not None else None
                    )
                    _target_lang = str(
                        getattr(_regen_parent, "language", "") or ""
                    )
                    runtime["_denovo_target_language"] = _target_lang
                    _regen_contract_name = denovo_contract_for_language(
                        _target_lang
                    )
                    runtime["_regen_is_prose"] = (
                        _regen_contract_name == "prose_contract"
                    )
                    contract = CONTRACTS.get(_regen_contract_name)
                    print(
                        f"[modify] PATH A using contract={_regen_contract_name} "
                        f"(artifact language={_target_lang!r}; "
                        f"change order NOT expected)",
                        file=sys.stderr, flush=True,
                    )
                else:
                    contract = CONTRACTS.get(str(runtime.get("artifact_contract", "")))
                # Step 5: clear any stale extraction from a prior
                # attempt/tier before this contract call so a later
                # attempt can never accidentally inherit an earlier
                # attempt's stashed extraction.
                runtime.pop("_extracted_code", None)
                runtime.pop("_extracted_language", None)
                # v13 step 1 (2026-08-07): clear any stale
                # polarity-corrected patch from a prior attempt/tier
                # before this contract call.  The contract may stash
                # a fresh ``_corrected_patch`` here when it detects
                # inverted hunks (see ``unified_diff_patch_contract``
                # and ``_flip_inverted_patch_hunks``).
                runtime.pop("_corrected_patch", None)
                runtime.pop("_corrected_patch_flips", None)
                runtime.pop("_pending_artifact_edit", None)
                error = error or (contract(output, runtime) if contract else None)
                # 2026-09-17 (spec 13.1/13.2): PATH A's child inherits the
                # PARENT's KIND -- a modification cannot change what the
                # artifact IS, so the extension MUST propagate parent -> child.
                # The denovo contracts stash the FENCE language the MODEL
                # emitted, and server.py prefers that over the parent's, so a
                # fenced ```javascript inside a Python artifact would silently
                # retag the child `.js`.  Pinned here rather than inside the
                # shared contracts so `new_code_generation` is untouched (the
                # condition requires `_is_regen`).
                if _is_regen and runtime.get("_denovo_target_language"):
                    runtime["_extracted_language"] = runtime[
                        "_denovo_target_language"
                    ]
                # ============================================================
                # 2026-09-14 (spec 6.1 / 6.2): PATH A drift checks
                # ============================================================
                # Two mechanisms, deliberately ASYMMETRIC:
                #   6.1 GATE (certain + cheap) -> AUTO-ESCALATE.  The
                #       requested modification is either present or it is
                #       not; we know what we asked for, so presence is a
                #       targeted assertion rather than a heuristic.
                #   6.2 TIER-1 (heuristic) -> REPORT ONLY, NEVER ESCALATE.
                #       A legitimate modification may deliberately change a
                #       constant, so automating this yields false escalations.
                # Both are gated on a declared flag so they can be turned off
                # without a code change (`runtime.modify_drift_checks`).
                # Both FAIL SOFT: any exception leaves the result untouched,
                # because a broken drift check must never break a good
                # regeneration.
                if (
                    error is None
                    and _is_regen
                    and bool(runtime.get("modify_drift_checks", True))
                ):
                    try:
                        from .provenance import (
                            modification_present as _mp,
                            tier1_drift as _t1,
                            cloud_modification_count as _cmc,
                        )
                        _artifact_out = (
                            str(runtime.get("_extracted_code") or output or "")
                        )
                        _parent_obj = (
                            artifact_store.get(artifact_id)
                            if artifact_store is not None else None
                        )
                        _parent_text = (
                            getattr(_parent_obj, "content", "") or ""
                        )
                        _parent_mods = list(
                            getattr(_parent_obj, "modifications", None) or []
                        )
                        # --- 6.1 GATE ---
                        # "absent" is an ESCALATION trigger, never a
                        # rejection: a false negative costs one cloud call
                        # (the route the pipeline already trusts), while a
                        # false positive would ship an artifact missing the
                        # requested change.  So the safe direction is taken
                        # on purpose -- see provenance.modification_present's
                        # documented KNOWN LIMITATION (it is lexical and can
                        # misread synonyms).
                        if not _mp(_artifact_out, str(user_intent or "")):
                            error = "modification_absent"
                            print(
                                "[modify] 6.1 GATE: requested modification "
                                "NOT detected in the regenerated artifact -> "
                                "escalating to the cloud change-order path",
                                file=sys.stderr, flush=True,
                            )
                        else:
                            # --- 6.2 TIER-1 (report only) ---
                            try:
                                _drift = _t1(_parent_text, _artifact_out)
                                runtime["_tier1_drift"] = _drift
                                if _drift.get("drift"):
                                    print(
                                        f"[modify] 6.2 TIER-1 drift flag "
                                        f"(REPORT ONLY, no escalation): "
                                        f"vanished={_drift['vanished']} "
                                        f"count_changed={_drift['count_changed']}",
                                        file=sys.stderr, flush=True,
                                    )
                                else:
                                    print(
                                        "[modify] 6.2 TIER-1: no literal drift",
                                        file=sys.stderr, flush=True,
                                    )
                            except Exception as _t1_exc:        # noqa: BLE001
                                print(f"[modify] 6.2 tier-1 failed soft: {_t1_exc}",
                                      file=sys.stderr, flush=True)
                            # --- 6.3 TIER-2 (conditional, review only) ---
                            # Only when the parent carries >=1 CLOUD change
                            # order: that is the one case where the
                            # regeneration basis is KNOWN to under-describe
                            # the parent (each stored entry is a request,
                            # not a record of what the cloud changed).  A
                            # pure PATH A parent is already the reliable
                            # case, so the expensive probe stays OFF the
                            # common path.  Deliberately review-only.
                            _cloud_n = _cmc(_parent_mods)
                            if _cloud_n:
                                runtime["_tier2_drift_pending"] = {
                                    "parent_cloud_modifications": _cloud_n,
                                    "parent_chars": len(_parent_text),
                                    "child_chars": len(_artifact_out),
                                    "action": "review",
                                }
                                print(
                                    f"[modify] 6.3 TIER-2 owed: parent has "
                                    f"{_cloud_n} cloud change order(s); result "
                                    f"approximates it (REVIEW ONLY)",
                                    file=sys.stderr, flush=True,
                                )
                    except ImportError:
                        pass
                    except Exception as _drift_exc:            # noqa: BLE001
                        print(f"[modify] drift checks failed soft: {_drift_exc}",
                              file=sys.stderr, flush=True)
                # 2026-09-10 (user-ruled refactor): the contract owns
                # extraction + validation + packaging. Transfer its
                # side-channel to the result here. user_intent and
                # source_filename are runtime scope, not contract scope, so
                # they are merged at this boundary.
                _contract_pending_edit = runtime.pop("_pending_artifact_edit", None)
                if error is None and _contract_pending_edit is not None:
                    _contract_pending_edit.setdefault("user_intent", user_intent)
                    _contract_pending_edit.setdefault("source_filename", source_filename)
                    # 2026-09-11 (user-ruled): SELF-REVIEW GATE. Before the
                    # package is accepted, one same-model non-streaming
                    # review call reuses the KV-warm prompt prefix (the
                    # generator's exact messages: persona + full artifact +
                    # outline + request) and appends the review contract +
                    # the proposed change order. Verdict YES -> proceed to
                    # integration; NO -> tier fails with the reviewer's
                    # reason (feeds the existing retry/ladder machinery).
                    # The reviewer reads the SAME context the generator
                    # read (full artifact), so dependency reasoning (used-
                    # but-undeclared, styled-but-never-rendered, calls to
                    # removed functions) happens at full fidelity — no
                    # BM25 summarization between the reviewer and the
                    # artifact. Zero load cost: same warm model/slot.
                    try:
                        _self_review_cfg = (work.harness.runtime or {}).get("self_review") or {}
                    except Exception:
                        _self_review_cfg = {}
                    # 2026-09-16 (gate 3): `local_change_order` -- the reviewer
                    # judges a LOCAL change order against the artifact.  With no
                    # local change order there is nothing for it to judge.  Note
                    # this correctly does NOT fire for PATH A, whose
                    # cloud-escalated change order is judged by the cloud and
                    # applied deterministically.
                    if (
                        _contract_pending_edit is not None
                        and getattr(work.harness, "local_change_order", False)
                        and bool(_self_review_cfg.get("enabled", False))
                    ):
                        # 2026-09-11 (user-ruled, v2): the reviewer gets its
                        # OWN system message — the generator's harness
                        # instruction ("produce a change order…") is wrong
                        # for a judge. Persona (work.persona) stays
                        # domain-aware from config (web/physics/python/
                        # general assistant). The full artifact context
                        # re-prefills (~10-12s) — accepted for prompt
                        # fidelity. Think state matches the tier's
                        # configured think flag; num_predict raised to
                        # 2048 so a thinking model has room for the
                        # verdict; empty/unparseable verdicts FAIL (the
                        # reviewer must affirmatively pass the package).
                        _sr_system = (
                            str(getattr(work, "persona", "") or "You are a helpful assistant.")
                            + "\n"
                            + str(_self_review_cfg.get("system_suffix")
                                  or "You are reviewing a proposed change order for the artifact provided above. Your only task: determine whether the proposed change order will fulfil the user's request. Answer with exactly YES, or NO: <one-sentence reason>. Nothing else.")
                        )
                        _sr_change_order = json.dumps(
                            _contract_pending_edit.get("edits") or {}, ensure_ascii=False
                        )
                        # Same artifact-context user turn as the generator
                        # (KV/prompt-cache where the prefix allows), minus
                        # the generator system message.
                        _sr_ctx_user = next(
                            (m.get("content") or "" for m in package.messages
                             if m.get("role") == "user"),
                            "",
                        )
                        _sr_messages = [
                            {"role": "system", "content": _sr_system},
                            {"role": "user", "content": _sr_ctx_user},
                            {
                                "role": "user",
                                "content": (
                                    "[PROPOSED CHANGE ORDER]\n"
                                    + _sr_change_order
                                    + "\n[END PROPOSED CHANGE ORDER]"
                                ),
                            },
                        ]
                        # 2026-09-12 (user-approved): the assistant PREFILL
                        # that used to close this list ("Review verdict: ")
                        # is DELETED.  It caused two independent, silent
                        # rejection paths, both measured on the live box:
                        #   1. plain generation: the model emitted NOTHING
                        #      (``eval=1``, content == the echoed prefill),
                        #      so the gate never saw a verdict;
                        #   2. with ``format``: HTTP 400 — the JSON grammar
                        #      constrains the first sampled token to ``{``,
                        #      and the prefill forces that position to a
                        #      non-``{`` token.
                        # The final user turn now also ends at the change
                        # order (no "Review verdict:" tail), so the reply
                        # is the model's own output and nothing is echoed
                        # back into the parse.
                        _sr_num_ctx = int(
                            _self_review_cfg.get("num_ctx")
                            or (package.options or {}).get("num_ctx")
                            or 32768
                        )
                        # 2026-09-12 (user-approved): deterministic sampling
                        # (a reviewer must not sample) plus Ollama's native
                        # ``format`` structured output, so the verdict is a
                        # STRUCTURAL field rather than something string-
                        # matched out of prose.  Both come from the
                        # self_review config block; both are optional so a
                        # config without them behaves exactly as before.
                        _sr_sampling = _self_review_cfg.get("sampling") or {}
                        _sr_format = _self_review_cfg.get("format_schema") or None
                        _sr_resp = await transport.non_streaming_call(
                            model_id=package.model_id,
                            messages=_sr_messages,
                            num_ctx=_sr_num_ctx,
                            num_predict=int(_self_review_cfg.get("verdict_num_predict", 2048)),
                            keep_alive=-1,
                            timeout=120.0,
                            think=package.think,
                            sampling=dict(_sr_sampling) if _sr_sampling else None,
                            format=dict(_sr_format) if _sr_format else None,
                        )
                        _sr_text = str(
                            ((_sr_resp or {}).get("message") or {}).get("content") or ""
                        ).strip()
                        # --- Verdict extraction -------------------------
                        # 2026-09-12 (user-approved, SCHEMA-ONLY): the
                        # configured ``format_schema`` makes the reply a
                        # JSON object {"reason": ..., "verdict": "YES"|"NO"}.
                        # The verdict is read as a structural field and
                        # FAILS CLOSED: not-JSON, missing key, or any value
                        # other than YES is a rejection that escalates.
                        # There is deliberately NO fallback to lenient text
                        # parsing — a reviewer that will not answer in the
                        # required form has not passed the package, and the
                        # T3 cloud fall-through is the design answer to a
                        # false rejection.  (2026-09-12 live history: the
                        # old assistant prefill made the pass path
                        # unreachable, so every contract-valid package was
                        # rejected; and a bare "NO" with no reason was a
                        # temp-0.8 sampling artifact, now fixed by the
                        # deterministic sampling above.)
                        _sr_pass = False
                        _sr_reason = "(empty verdict)"
                        if _sr_format:
                            try:
                                _sr_json = json.loads(_sr_text)
                                _sr_verdict = str(
                                    (_sr_json or {}).get("verdict") or ""
                                ).strip().upper()
                                _sr_reason = str(
                                    (_sr_json or {}).get("reason") or ""
                                ).strip() or "(no reason given)"
                                _sr_pass = _sr_verdict == "YES"
                            except Exception:
                                _sr_pass = False
                                _sr_reason = (
                                    "reviewer reply was not valid JSON under the "
                                    "configured format schema: " + (_sr_text[:200] or "(empty)")
                                )
                        else:
                            # No schema configured -> legacy text parse.  Kept
                            # only for a config that omits format_schema; the
                            # shipped config always sets it.
                            _sr_norm = re.sub(
                                r"(?is)^\s*review\s+verdict\s*[:\-\u2013\u2014]?\s*",
                                "", _sr_text,
                            )
                            _colon = _sr_norm.find(":")
                            _sr_reason = (
                                _sr_norm[_colon + 1:].strip()
                                if _colon >= 0 else _sr_norm
                            ) or "(empty verdict)"
                            _sr_pass = _sr_norm.upper().startswith("YES")
                        if _sr_pass:
                            pass  # package passes review; proceed
                        else:
                            error = (
                                "self_review_rejected: the change order was reviewed "
                                "against the request and found incomplete -- "
                                + _sr_reason[:300]
                            )
                            _contract_pending_edit = None
                    if error is None and _contract_pending_edit is not None:
                        result.pending_artifact_edit = _contract_pending_edit
                # 2026-09-10 (user ruling): modify-only output-cap trip
                # handler.  The cap guard closed the stream mid-generation
                # with ``output_cap_exceeded``.  Rescue step: extract the
                # LAST COMPLETE change order from the partial text and
                # validate it against the round2 contract the same way the
                # normal one-shot path does.  Passes -> accept it as the
                # tier's answer (the pending-edit machinery consumes it
                # identically, so no re-roll is spent).  Fails -> ONE
                # dedicated re-roll with a fresh seed and the
                # ``re_roll_modifiers.output_cap`` message; the re-roll does
                # NOT increment the shared pathology budget.  A second cap
                # trip on the same model exhausts the dedicated budget and
                # the normal ladder (T2/secondary, then T3 cloud) takes
                # over.
                # 2026-09-16 (gate 5b): `local_change_order` -- the rescue
                # parses a truncated LOCAL stream looking for the LAST COMPLETE
                # change order.  A harness whose local tier cannot emit one has
                # nothing to rescue.
                if (
                    error == "output_cap_exceeded"
                    and getattr(work.harness, "local_change_order", False)
                    and artifact_store is not None
                ):
                    _cap_rescue_err, _cap_pkg = modify_output_cap_rescue(
                        output, runtime, artifact_store, artifact_id,
                    )
                    if _cap_rescue_err is None and _cap_pkg is not None:
                        # Rescue pass: the partial stream contained a
                        # complete, contract-valid change order.  Accept it.
                        print(
                            f"[modify] output cap tripped at {len(output)} chars; "
                            "rescue extracted a valid change order — accepting",
                            file=sys.stderr, flush=True,
                        )
                        error = None
                        result.pending_artifact_edit = {
                            "edits": _cap_pkg,
                            "artifact_id": artifact_id,
                            "user_intent": user_intent,
                            "source_filename": source_filename,
                        }
                    else:
                        # Rescue fail: dedicated re-roll gate.  The cap
                        # counter is parallel to the shared pathology
                        # budget and never increments it.
                        _cap_max_re_rolls = int(
                            _resolve_cfg(runtime, "output_cap_max_re_rolls") or 0
                        )
                        output_cap_count += 1
                        print(
                            f"[pathology] model={package.model_id} "
                            f"output_cap_exceeded at {len(output)} chars "
                            f"(rescue: {_cap_rescue_err}; "
                            f"cap reroll {output_cap_count}/{max(1, _cap_max_re_rolls)})",
                            file=sys.stderr, flush=True,
                        )
                        if output_cap_count > _cap_max_re_rolls:
                            # Dedicated budget spent: fall through to the
                            # normal ladder (the error stays set; the tier
                            # commits to the next slot below).
                            pass
                        else:
                            # One dedicated re-roll: fresh seed + the
                            # output_cap modifier on the next attempt.
                            # Record THIS attempt before continuing so
                            # attempt accounting and the raw-writer tier
                            # boundary stay truthful.
                            calls += 1
                            result.attempts.append(CleanAttempt(
                                tier, package.model_id, output, False,
                                error, _now() - started,
                            ))
                            if raw_token_writer is not None:
                                try:
                                    raw_token_writer.end_tier(
                                        elapsed_s=_now() - started,
                                        output_len=len(output),
                                        error=error,
                                        selected_model=package.model_id,
                                        chunks=len(output_parts),
                                    )
                                except Exception:
                                    pass
                            output_cap_reroll_now = True
                            continue
                # v13 step 1: polarity spell-check / auto-flip.
                # If the contract stashed a corrected patch AND
                # reported no error, replace ``output`` with the
                # corrected text.  This is the same side-channel
                # pattern as ``_extracted_code`` (line 272) — the
                # contract can't mutate ``output`` directly because
                # Python strings are immutable.  The corrected
                # patch is what the SWE-bench evaluator sees via
                # ``result.output`` -> SSE ``delta.content`` ->
                # smoke client's ``extract_diff``.
                corrected_patch = runtime.pop("_corrected_patch", None)
                num_flips = runtime.pop("_corrected_patch_flips", 0)
                # v13 step 2 (2026-08-08): the corrected-patch
                # side-channel now carries TWO kinds of replacement:
                #   (a) polarity auto-flip (num_flips > 0), and
                #   (b) context-correct block selection (num_flips == 0,
                #       when the model's raw last block was hallucinated
                #       and an earlier context-correct block is the
                #       answer).
                # Both should replace ``output``, so the trigger is
                # ``corrected_patch is not None`` (not ``num_flips > 0``).
                if error is None and corrected_patch is not None:
                    output = corrected_patch
                    # Log the auto-fix at INFO level so it's
                    # traceable in server logs without being
                    # noisy.  We deliberately do NOT mutate
                    # ``result.output`` here — that gets set by
                    # the tier loop's bookkeeping code below from
                    # the (now-corrected) ``output`` variable.
                # Contract-gated extraction.  When the contract
                # passed AND stashed an extracted-code side channel
                # (the *_new_code_contract functions do this on
                # their pass path), copy it onto the result so
                # server.py can use it for artifact creation instead
                # of re-deriving via regex.
                if error is None and runtime.get("_extracted_code"):
                    result.extracted_artifact_content = runtime["_extracted_code"]
                    result.extracted_artifact_language = runtime.get("_extracted_language")
            except Exception as exc:
                # ``StreamAborted`` is raised by the
                # transport when the starvation timer
                # asked the stream to close.  It is NOT a
                # backend error: it is the expected abort
                # path for silent or stalled streams.
                # Translate it to the right
                # ``first_token_timeout`` /
                # ``slow_decode_timeout`` error string so
                # the retry ladder engages cleanly.
                #
                # When a pathology monitor exists for
                # this call, we prefer the monitor's
                # own verdict (which carries the actual
                # reason and observed rate) over a
                # heuristic from ``output_parts``.
                from .transport import StreamAborted
                if isinstance(exc, StreamAborted):
                    final_verdict = monitor.check() if monitor is not None else None
                    if final_verdict is not None and final_verdict.reason == "ttft":
                        error = (
                            f"first_token_timeout: no tokens within "
                            f"{final_verdict.budget_s:.0f}s of stream start"
                        )
                    elif final_verdict is not None and final_verdict.reason == "slow_decode":
                        error = (
                            f"slow_decode_timeout: decode rate "
                            f"{final_verdict.observed_rate:.1f} tok/s below "
                            f"{monitor.min_decode_rate:.0f} threshold"
                        )
                    else:
                        if not output_parts:
                            error = f"first_token_timeout: no tokens within {first_token_budget:.0f}s of stream start"
                        else:
                            error = f"slow_decode_timeout: no new token within {slow_decode_timeout_s:.0f}s of last token (len={len(output_parts)} chunks)"
                else:
                    error = f"backend_error: {exc}"
                output = "".join(output_parts)
            except BaseException as exc:
                # ``asyncio.CancelledError`` is a
                # ``BaseException`` in Python 3.8+ and
                # would normally skip the ``except
                # Exception`` handler above.  This branch
                # catches the outer ``asyncio.wait_for``
                # cancellation (and any other
                # ``BaseException``) so partial output is
                # preserved instead of silently
                # discarded.
                #
                # A ``CancelledError`` from the outer
                # ``wait_for`` is a TIMEOUT (the model hung
                # or stalled).  It is a legitimate
                # pathology, NOT a fatal error: it should
                # count toward the per-model pathology
                # budget and re-roll / escalate to the
                # secondary after the budget is exhausted,
                # exactly like any other pathology.  So we
                # set the error and FALL THROUGH to the
                # normal pathology classification below
                # (do NOT re-raise).  Other BaseExceptions
                # (KeyboardInterrupt, SystemExit, genuine
                # bugs) still re-raise so they are not
                # swallowed.
                if isinstance(exc, asyncio.CancelledError):
                    error = (
                        f"cancelled: outer wait_for fired after "
                        f"{_now() - started:.1f}s with {len(output_parts)} chunks "
                        f"preserved"
                    )
                    output = "".join(output_parts)
                    if monitor is not None:
                        print(
                            f"[pathology] model={package.model_id} "
                            f"CANCELLED elapsed={_now() - started:.1f}s "
                            f"chunks={len(output_parts)} err={error}",
                            file=sys.stderr,
                        )
                    # 2026-09-19 (PLAN_progress_mechanism.md follow-up):
                    # RE-RAISE.  This comment used to say a CancelledError here
                    # is "a TIMEOUT (the model hung or stalled)" and instructed
                    # falling through to the pathology classification.  That is
                    # not what this exception is in this code path.
                    #
                    # There is NO `await asyncio.wait_for(...)` wrapping a tier
                    # attempt -- the only real wait_for in this module is the
                    # chunk-level one at ~3364 inside the cloud stream guard
                    # (and the two `task.cancel()` calls at ~6553/6559 cancel
                    # the monitor helpers and are awaited in a `finally`).
                    # CancelledError therefore cannot originate from an internal
                    # timeout here: the ONLY source is an EXTERNAL
                    # `task.cancel()` -- i.e. /v1/cancel.
                    #
                    # Swallowing it was a real defect: `error_key` became
                    # "cancelled", which is in no harness's functional_errors or
                    # pathology.errors, so it fell to the "UNCLASSIFIED" branch
                    # (bump functional_fail_count, then break).  That break exits
                    # the INNER attempt loop, so the outer ladder loop could
                    # ADVANCE TO ANOTHER TIER -- an explicit user cancel could
                    # escalate the very work the user just asked to stop (in the
                    # worst case to the cloud).
                    #
                    # Re-raising lets the cancellation propagate to the task /
                    # the executor's own handling, which records the turn as
                    # cancelled and preserves partial output.  Genuine internal
                    # timeouts (first_token_timeout / slow_decode_timeout) are
                    # raised by the transport as StreamAborted and are handled by
                    # the `except Exception` block ABOVE this one -- they do not
                    # reach here, so this change cannot affect them.
                    calls += 1
                    attempt = CleanAttempt(tier, package.model_id, output, False, error, _now() - started)
                    result.attempts.append(attempt)
                    if raw_token_writer is not None:
                        try:
                            await raw_token_writer.close()
                        except Exception:
                            pass
                    raise
                else:
                    error = f"base_exception: {type(exc).__name__}: {exc}"
                    output = "".join(output_parts)
                    calls += 1
                    attempt = CleanAttempt(tier, package.model_id, output, False, error, _now() - started)
                    result.attempts.append(attempt)
                    if raw_token_writer is not None:
                        try:
                            await raw_token_writer.close()
                        except Exception:
                            pass
                    raise
            calls += 1
            attempt = CleanAttempt(tier, package.model_id, output, error is None and not aborted and not accepted, error, _now() - started)
            result.attempts.append(attempt)
            # v12 T1 primitive: capture T0's output + error
            # so the next T1 iteration can compose the T1b
            # prompt with the YAML-driven
            # ``t1_full_user_modifier`` template.  We
            # capture only when on T0 (or T2, which is the
            # secondary model's T0-equivalent); T1's own
            # output does not overwrite this.
            if tier == "T0":
                t0_output_for_t1 = output
                t0_error_for_t1 = error
            # Live raw writer: log per-attempt summary
            # so post-mortem tools can slice by tier
            # without re-parsing the meta jsonl.  We log
            # the actual TTFT, decode rate, chunk count,
            # and error string for every attempt.
            if raw_token_writer is not None:
                try:
                    _ttft = None
                    _rate = None
                    if monitor is not None and monitor._first_token_monotonic is not None and monitor._start_monotonic > 0:
                        _ttft = monitor._first_token_monotonic - monitor._start_monotonic
                    if monitor is not None and monitor._token_count > 1 and monitor._last_token_monotonic is not None and monitor._first_token_monotonic is not None:
                        span = monitor._last_token_monotonic - monitor._first_token_monotonic
                        if span > 0:
                            _rate = (monitor._token_count - 1) / span
                    raw_token_writer.end_tier(
                        elapsed_s=_now() - started,
                        output_len=len(output),
                        error=error,
                        selected_model=package.model_id,
                        ttft_s=_ttft,
                        decode_rate=_rate,
                        chunks=len(output_parts),
                    )
                    # Pathology events: when the
                    # attempt failed with a
                    # known pathology code, emit a
                    # structured event so the meta
                    # jsonl has a row for every
                    # abort reason.
                    if error and monitor is not None:
                        ek = error.split(":", 1)[0].strip()
                        if ek in ("first_token_timeout", "slow_decode_timeout",
                                  "vram_breach", "cancelled", "backend_error",
                                  "backend_server_error",
                                  "no_code_fence", "syntax_error", "dither",
                                  "loop", "multiple_patches", "empty_patch",
                                  "malformed_patch", "token_exhaustion",
                                  "post_code_fence_text", "rate_stall",
                                  "rate_collapse", "prepare_timeout",
                                  "output_ceiling_exceeded",
                                  "output_cap_exceeded",
                                  "cloud_silence_timeout"):
                            _details = None
                            _observed = None
                            _budget = None
                            if ek == "first_token_timeout":
                                _details = "no tokens within budget"
                            elif ek == "slow_decode_timeout":
                                _details = "decode rate below threshold"
                                _observed = _rate
                                _budget = monitor.min_decode_rate
                            raw_token_writer.log_pathology(
                                kind=ek,
                                elapsed_s=_now() - started,
                                observed_rate=_observed,
                                budget_s=_budget,
                                chunks_so_far=len(output_parts),
                                last_token_at=monitor._last_token_monotonic,
                                details=_details,
                            )
                except Exception:
                    pass
            # Pathology telemetry: report the
            # on-wire-start, actual TTFT, predicted
            # budget, and the verdict on a single
            # line so we can compare predicted vs
            # actual and tune the safety factor.  Only
            # printed when a monitor was created (i.e.
            # an Ollama call).
            if monitor is not None:
                pred_budget = monitor._ttft_budget_s
                if monitor._first_token_monotonic is not None and monitor._start_monotonic > 0:
                    actual_ttft = monitor._first_token_monotonic - monitor._start_monotonic
                    actual_str = f"{actual_ttft:.2f}s"
                else:
                    actual_ttft = None
                    actual_str = "no_token"
                ratio = (actual_ttft / pred_budget) if (actual_ttft is not None and pred_budget > 0) else None
                ratio_str = f"{ratio:.2f}x" if ratio is not None else "n/a"
                state = "warm" if is_resident else "cold"
                print(
                    f"[pathology] model={package.model_id} state={state} "
                    f"prompt~{prompt_tokens_estimate}tok predicted={pred_budget:.2f}s "
                    f"actual_ttft={actual_str} budget_ratio={ratio_str} "
                    f"err={error}",
                    file=sys.stderr,
                )
            # 2026-09-13 (user-ruled): announce a tier as COMPLETE only when
            # it has actually finished, i.e. the attempt PASSED or the
            # ladder is about to leave this tier.  A retry within the same
            # tier is an intermediate attempt, not a completion: emitting a
            # completion event for it made the CLI print
            # "tier_complete: T2, passed: false" mid-slot (three times in
            # the 2026-09-13 run), which read as though the tier had ended
            # while it was in fact still retrying.  Retries are now reported
            # separately via the "tier_retry" event below, so both surfaces
            # stay accurate.  The single-tier display behaviour is unchanged
            # (the server still suppresses these events when the harness has
            # one tier).
            await emit(CleanProgressEvent(
                event_type="tier_complete", tier=tier, output=output, model_id=package.model_id,
                metadata={"passed": attempt.passed, "aborted": aborted, "accepted_partial": accepted},
            ))
            if attempt.passed:
                # 2026-09-14 (spec 5.7.2 PATH A): a regeneration produces a
                # COMPLETE artifact.  Flag it so server.py commits a modified
                # version of the lineage rather than looking for a change
                # order, and so the provenance log records method
                # "regenerated".
                # 2026-09-16 (M2, gate 1): property, not harness name.  This
                # is the PATH A commit flag; PATH A is reached by reading the
                # artifact store, so the property IS the condition.
                if runtime.get("modify_skip_reintegration") and (
                    getattr(work.harness, "needs_artifact_store", False)
                ):
                    result.modify_was_regenerated = True
                    # Carry the denovo extraction into the field the server's
                    # PATH A commit branch reads.
                    if runtime.get("_extracted_code"):
                        result.extracted_artifact_content = runtime["_extracted_code"]
                        result.extracted_artifact_language = runtime.get(
                            "_extracted_language"
                        )
                    # 2026-09-14 (spec 6.2 / 6.3): REPORT-ONLY drift evidence.
                    # Neither check escalates by design, so this field is the
                    # only way the flags reach the caller (stderr alone loses
                    # them).  Read from the runtime dict where the 6.x block
                    # recorded them; None when drift checks were disabled or
                    # nothing was flagged.
                    if runtime.get("_tier1_drift") or runtime.get("_tier2_drift_pending"):
                        result.modify_drift_evidence = {
                            "tier1": runtime.get("_tier1_drift"),
                            "tier2": runtime.get("_tier2_drift_pending"),
                            "action": "review",
                        }
                # 2026-09-09: one-shot modify — the pending edit replaces
                # the raw output as the delivered result (same marker shape
                # the T3 dispatcher and the two-round path produce; the
                # server's _commit_pending_artifact consumes it identically).
                if result.pending_artifact_edit is not None:
                    result.output = (
                        f"[[modify_artifact: pending edit for artifact "
                        f"{result.pending_artifact_edit['artifact_id']}]]"
                    )
                result.output = visible_output if work.harness.harness_id == "general_use" else output
                result.passed, result.selected_model = True, package.model_id
                if work.harness.harness_id == "general_use":
                    (
                        result.general_use_work_product_content,
                        result.general_use_work_product_language,
                    ) = _extract_general_use_work_product(visible_output)
                    if not result.general_use_work_product_content:
                        result.general_use_work_product_content = None
                result.total_elapsed_s = _now() - run_start
                await emit(CleanProgressEvent(
                    event_type="complete", tier=tier, output=output, model_id=package.model_id,
                    elapsed_s=result.total_elapsed_s,
                ))
                return result
            if aborted or accepted:
                result.output, result.selected_model = output, package.model_id
                result.total_elapsed_s = _now() - run_start
                await emit(CleanProgressEvent(
                    event_type="complete", tier=tier, output=output, model_id=package.model_id,
                    elapsed_s=result.total_elapsed_s,
                    metadata={"aborted": aborted, "accepted_partial": accepted},
                ))
                return result
            # v12 single-bucket error classification.
            # The runtime no longer partitions errors into
            # generation vs transport; the harness declares
            # the single ``pathology.errors`` list.  All
            # pathology errors increment the per-MODEL
            # counter; functional errors increment the
            # per-MODEL functional counter and escalate.
            error_key = error.split(":", 1)[0].strip()
            # 2026-09-12 (user-ruled): D5 MODEL-BLACKLIST REMOVED.
            #
            # D5 (2026-08-29) treated a llama-server 500 as a
            # DETERMINISTIC backend crash ("same weights + same prompt =
            # same crash"), renamed it ``backend_server_error``, added the
            # MODEL to a ``crashed_models`` blacklist, and skipped the
            # whole slot for the rest of the request.  That was wrong on
            # three counts:
            #   1. SEPARATION OF CONCERNS.  A stop DECISION was keyed on
            #      model identity.  Slots are the accounting/decisional
            #      unit; which model is bound to a slot is a configuration
            #      and backend-resolution concern.  With both slots bound
            #      to the same model, a primary crash silently skipped the
            #      secondary too.
            #   2. TRANSIENCE.  A single 500 can be a transient
            #      transport/daemon fault, so blacklisting on ONE
            #      occurrence can discard a slot that would have worked.
            #   3. CLASSIFICATION.  ``backend_server_error`` was in
            #      neither pathology.errors nor functional_errors, so the
            #      crash ALSO fell to the "unrecognized" branch: no retry,
            #      and it consumed the slot's functional budget.
            #
            # New behaviour (user ruling: "treating backend errors as just
            # another pathology error, log/announce loudly and continue with
            # the existing retry/reroll/escalation budget"):
            #   * ``backend_server_error`` is now classified in
            #     pathology.errors (config-owned), so it uses the normal
            #     pathology budget -- retry within the budget, then
            #     escalate.
            #   * Nothing is blacklisted; no model or slot identity is
            #     consulted anywhere in this decision.
            #   * Every occurrence is announced loudly with full context so
            #     a genuinely persistent backend fault is visible in the
            #     trace instead of being silently routed around.
            if "peg-native format" in (error or "") or (
                error_key == "backend_error" and "llama-server" in (error or "")
                and "error" in (error or "").lower()
            ):
                error_key = "backend_server_error"
            if error_key == "backend_server_error":
                _crash_count = pathology_count + 1
                _will_retry = _crash_count <= max_re_rolls
                print(
                    f"[ladder] BACKEND ERROR at tier={tier} slot={_budget_slot} "
                    f"model={package.model_id} attempt={_crash_count}/"
                    f"{max_re_rolls + 1} -> "
                    f"{'retry/re-roll within budget' if _will_retry else 'budget exhausted, escalating'}"
                    f" | NO blacklist applied (transient faults are expected; "
                    f"model identity is not a reason to skip a slot)"
                    f" | detail={str(error)[:200]}",
                    file=sys.stderr, flush=True,
                )
            if error_key in pathology_errs:
                # Single counter for all pathology kinds.
                # 2026-09-12: accounted per-SLOT (see the slot-change reset
                # at the top of the tier loop); resets when the slot changes.
                pathology_count += 1
                # 2026-09-13 (user ruling): the retry surface must be
                # ACCURATE.  Emit a dedicated tier_retry event carrying the
                # retry kind (pathology vs functional), the attempt number,
                # the limit, and whether this attempt will retry or the
                # ladder will escalate -- instead of the CLI inferring
                # "retry" from a bare [pathology] line.
                _will_retry = pathology_count <= max_re_rolls
                await emit(CleanProgressEvent(
                    event_type="tier_retry", tier=tier, output="", model_id=package.model_id,
                    metadata={
                        "retry_kind": "pathology",
                        "reason": error_key,
                        "attempt": pathology_count,
                        "limit": max_re_rolls + 1,
                        "action": "fresh_seed_reroll" if _will_retry else "escalate_next_slot",
                        "error": str(error)[:300],
                    },
                ))
            elif error_key == "output_cap_rescue_failed":
                # 2026-09-10 (user ruling): the modify-only output-cap
                # re-roll budget is DEDICATED.  A failed cap re-roll never
                # increments the shared pathology counter; the dedicated
                # gate (checked in the trip handler) decides when the
                # ladder commits to the next slot.
                prior_error = error
                break
            elif error_key in func_errs:
                functional_fail_count += 1
                prior_error = error
                # v13 step 2 (2026-08-08): functional retry.  A
                # functional error (e.g. patch_does_not_apply,
                # malformed_patch) now retries the SAME tier with a
                # new seed, up to the functional budget
                # (``max_func_errors``).  Only when the budget is
                # exhausted does it ``break`` to commit to the next
                # model slot.  This gives each model slot multiple
                # functional tries (budget+1) before escalating.
                # 2026-09-12: the budget is the CURRENT TIER's declared
                # ``functional_budget`` (config-owned), not the harness-wide
                # backup value.
                # 2026-09-13 (user ruling): accurate retry reporting --
                # emit tier_retry so the CLI shows the kind, attempt/limit
                # and the decided action instead of inferring it.
                _fn_will_retry = functional_fail_count <= _max_func_errors_this_tier
                await emit(CleanProgressEvent(
                    event_type="tier_retry", tier=tier, output="", model_id=package.model_id,
                    metadata={
                        "retry_kind": "functional",
                        "reason": error_key,
                        "attempt": functional_fail_count,
                        "limit": _max_func_errors_this_tier + 1,
                        "action": "same_seed_reground" if _fn_will_retry else "escalate_next_slot",
                        "error": str(error)[:300],
                    },
                ))
                if not _fn_will_retry:
                    break  # functional budget exhausted: advance to next slot
                # v13 step 2 (2026-08-08): functional retry.  Re-iterate
                # the same tier, KEEPING ``prior_error`` so the retry's
                # user message carries the error trace (grounding the
                # model on why its patch failed).  The seed is NOT
                # randomized on a functional retry (see the seed logic
                # at the top of the loop) — the retry is a deterministic
                # re-attempt with better grounding, not a random re-roll.
                continue
            else:
                # Unrecognized error: treat as functional
                # and advance.
                #
                # 2026-09-12 (user-ruled, live failure): this branch is
                # DANGEROUS because it is silent.  It increments the
                # per-MODEL functional counter AND breaks immediately, so an
                # unclassified error both (a) gets no retry itself and
                # (b) CONSUMES the model slot's functional budget, starving
                # the NEXT tier of the retry the config intends.  Live case:
                # T0's ``self_review_rejected`` (unclassified) set the
                # counter to 1, so T2's ``malformed_json`` evaluated
                # ``2 > max_func_errors(1)`` and broke with no retry.
                # Print loudly so the missing classification is visible in
                # the trace instead of masquerading as a model failure.
                print(
                    f"[ladder] UNCLASSIFIED ERROR '{error_key}' at tier={tier} -- "
                    f"not in functional_errors or pathology.errors; it will "
                    f"consume the functional budget without a retry. Add it to "
                    f"the harness's tier-level functional_errors (or "
                    f"pathology.errors) to restore intended behaviour. "
                    f"detail={str(error)[:160]}",
                    file=sys.stderr, flush=True,
                )
                functional_fail_count += 1
                prior_error = error
                break
            # === CAP CHECK (per-MODEL, single rule) ===
            # v12: the cap is a single uniform rule:
            # ``pathology_count > max_re_rolls_per_model``
            # commits to the next model.  No arithmetic
            # on the cap value; no per-phase formula
            # swap; no separate generation vs transport
            # counters.  The harness declares
            # ``pathology.max_re_rolls_per_model`` and the
            # runtime executes the rule.
            if pathology_count > max_re_rolls:
                # The (N+1)th pathology trigger fires the
                # commit.  This is the same rule in both
                # the primary and secondary phases: 2
                # re-rolls, 3rd trigger commits.
                prior_error = error
                break
            # === CUMULATIVE CEILING: local contract over ===
            # A cumulative output_ceiling_exceeded is NOT a seed artifact
            # problem -- the problem-scoped output budget is spent.  No
            # further local reroll can help (every attempt adds output
            # to an already-over-budget system).  Exit the ladder
            # entirely: the outer tier loop exits too (the while-gate at
            # the top of the tier body checks this flag), and the
            # meta-harness escalation (T3 cloud / T5 local per
            # meta_harness_ladder) takes over.
            if ceiling_cumulative_tripped:
                prior_error = error
                break
            # === PATHOLOGY RE-ROLL: RESTART T0 OF COMMITTED
        # End of inner ``while calls < max_calls``.
        # 4.4: Data-driven stage transitions from the escalation_ladder.
        # The positional stage transition logic (tier_idx == 1 → commit
        # to secondary, tier_idx == 2 → commit to T5) is replaced by
        # data-driven logic from the escalation_ladder.  The mapping
        # declares per-tier model_slot from the ladder.  Track which
        # model_slots have been exhausted (their functional budget
        # is spent) and mark them when the current tier is the
        # last tier on the model_slot.
        ladder = work.harness.escalation_ladder
        tier_model_slots: dict[str, str] = {}
        for t in work.harness.tiers:
            t_cfg = ladder.get(t.lower(), {})
            tier_model_slots[t] = t_cfg.get("model_slot", "primary")
        # Track which model_slots have been exhausted (their
        # functional budget is spent) and mark them when the current
        # tier is the last tier on the model_slot.
        exhausted_slots: set[str] = set()
        # current_slot is looked up per tier in the tier loop below;
        # it's the model_slot of the current tier.
        current_slot: str = "primary"
        # === COMMIT-TO-NEXT-SLOT (data-driven stage transition) ===
        # Fires when the current tier is the LAST tier on its
        # model_slot AND the functional budget for this slot is
        # exhausted.  At commit:
        #   * ``exhausted_slots.add(current_slot)`` (one-way state
        #     variable; the slot is marked spent for the rest of
        #     the run)
        #   * ``functional_fail_count`` is reset to 0 (the next
        #     slot starts clean for functional tracking)
        #   * ``pathology_count`` is RESET to 0 so the next
        #     slot's pathology cap (uniform rule
        #     ``pathology_count > max_re_rolls``) has
        #     ``max_re_rolls`` more triggers before firing.
        #   * ``prior_error`` is cleared (next slot starts
        #     without carrying the primary's last error string
        #     into the directive)
        #
        # The OLD positional logic (tier_idx == 1 → commit to
        # secondary, tier_idx == 2 → commit to T5) is gone.
        # Determine whether the inner loop exited because of a
        # T1 pathology restart (which set tier_idx=0 and broke).
        # In that case, do NOT advance tier_idx (let the outer
        # loop re-enter T0).
        last_attempt = result.attempts[-1] if result.attempts else None
        last_was_t1_pathology_restart = (
            last_attempt is not None
            and last_attempt.tier.lower() == "t1"
            and tier_idx == 0
        )
        # Is this the last tier on the current slot?
        is_last_tier_on_slot = (
            tier_idx == max(i for i, t in enumerate(work.harness.tiers)
                          if tier_model_slots.get(t, "primary") == current_slot)
        )
        if (
            is_last_tier_on_slot
            and not transition_to_secondary
            and not last_was_t1_pathology_restart
            and current_slot not in exhausted_slots
        ):
            # 2026-09-12: budget is the current tier's declared value.
            if functional_fail_count > _max_func_errors_this_tier:
                exhausted_slots.add(current_slot)
                functional_fail_count = 0
                pathology_count = 0   # was: gen_count[2] = max_gen - 1
                prior_error = None    # next slot starts clean
            # === COMMIT-TO-META-HARNESS (data-driven T5) ===
            # NOTE on the T5 invocation: when ALL slots are exhausted
            # AND the ladder declares a meta_harness_ladder target
            # (currently only "generalist_zero_shot"), invoke it.
            # The old positional logic (tier_idx == 2 → transition_to_t5)
            # is gone; the meta-harness invocation is now purely the
            # invocation AFTER the tier loop.  A null target means
            # no meta-harness invocation.
            # This block is AFTER the tier loop (unchanged placement).
            # The old positional logic (tier_idx == 2 →
            # transition_to_t5) is gone; the meta-harness invocation
            # is now purely the invocation below.
        # === TIER ADVANCEMENT ===
        # If the last action was a T1 pathology restart
        # (we set tier_idx = 0 and broke), do NOT
        # increment.  Otherwise, advance to the next tier.
        if last_was_t1_pathology_restart:
            pass  # outer loop will re-enter T0
        elif tier_model_slots.get(tier, "primary") in ceiling_tripped_slots:
            # 2026-08-30: the current slot tripped the cumulative ceiling.
            # Skip to the FIRST tier of the NEXT model_slot (the secondary
            # gets its held-out budget).  Do NOT advance to a same-slot
            # tier (T1a/T1b) -- that model is already over its budget.
            # If no next slot exists, tier_idx advances past the end and
            # the loop exits to the meta level (T3).
            _slots_in_order = []
            for _i, _t in enumerate(work.harness.tiers):
                _s = tier_model_slots.get(_t, "primary")
                if not _slots_in_order or _slots_in_order[-1][0] != _s:
                    _slots_in_order.append((_s, _i))
            _cur_slot = tier_model_slots.get(tier, "primary")
            _next_slot_idx = None
            for _j, (_s, _i) in enumerate(_slots_in_order):
                if _s == _cur_slot:
                    if _j + 1 < len(_slots_in_order):
                        _next_slot_idx = _slots_in_order[_j + 1][1]
                    break
            if _next_slot_idx is not None:
                tier_idx = _next_slot_idx
            else:
                tier_idx += 1
        else:
            tier_idx += 1
    if result.attempts:
        result.output = result.attempts[-1].output
        result.selected_model = result.attempts[-1].model_id
        result.error = result.attempts[-1].error
    result.total_elapsed_s = _now() - run_start
    # 4.7: Meta-harness ladder data-driven from
    # escalation_ladder.meta_harness_ladder.
    #
    # The meta-harness ladder is a data-driven list of fallback
    # harnesses to invoke when ALL tier slots are exhausted.
    # Currently only "generalist_zero_shot" is supported.
    # meta_harness_ladder: null (as in general_use) means no
    # meta-harness invocation — the harness's failure is
    # returned to the client as-is.
    #
    # The invocation order is driven by ``targets``:
    # the first target that yields a passing result wins.
    ladder = work.harness.escalation_ladder
    meta = ladder.get("meta_harness_ladder")
    # DIAG (temporary): meta-harness gate visibility.
    print(f"[meta] enable_t5={enable_t5} meta_present={meta is not None} "
          f"targets={(meta or {}).get('targets') if meta else None} "
          f"result.passed={result.passed}", file=sys.stderr, flush=True)
    if enable_t5 and meta is not None and not result.passed:
        targets = meta.get("targets", [])
        # D6 (2026-08-29): cloud escalation policy, read from the
        # harness ladder's cloud block (config.yaml):
        #   on_failure: hard_stop (default) — cloud failure ENDS the
        #     request; no T5 fallthrough (user policy: a local
        #     generalist re-attempt after cloud failure is a waste).
        #   silence_timeout_s — in-attempt dead-air limit (D4).
        #   think / max_tokens — explicit cloud sampling contract (D3).
        cloud_cfg = (meta or {}).get("cloud") or {}
        cloud_on_failure = str(cloud_cfg.get("on_failure", "hard_stop")).lower()
        # Read silence/total timeouts from config.  Absent or null = no
        # limit applied (None means unlimited for the cloud request).
        cloud_silence_s = cloud_cfg.get("silence_timeout_s")  # may be None
        cloud_total_s = cloud_cfg.get("total_timeout_s")      # may be None
        # 2026-09-09 (user ruling): cloud reasoning effort is a separate
        # config parameter ("low"|"medium"|"high") — never derived from
        # the local think on/off policy.
        cloud_effort = cloud_cfg.get("reasoning_effort")
        # D3 (2026-08-30): explicit max_tokens, else DERIVE it from the
        # configured decode rate * wall budget:
        #   max_tokens = tokens_per_sec * total_timeout_s
        # So the cloud gets exactly as many tokens as it can produce within
        # its wall-clock budget at the assumed rate.
        cloud_max_tokens = cloud_cfg.get("max_tokens")
        if cloud_max_tokens is None:
            _tps = cloud_cfg.get("tokens_per_sec")
            if _tps is not None and cloud_total_s is not None:
                cloud_max_tokens = int(_tps) * int(cloud_total_s)
        if cloud_max_tokens is not None:
            cloud_max_tokens = int(cloud_max_tokens)
        for target in targets:
            if result.passed:
                break
            if target == "cloud_sota":
                selected_t3_model = t3_vision_model if unit.images else t3_model
                # T3 target: cloud escalation.  Reads ONLY the t3_*
                # params -- no T5 involvement (separate paths, separate
                # principles: T5 = local compensation, T3 = SOTA
                # zero-shot of the original problem).  Skipped silently
                # when cloud is disabled (t3_model None) -- T5 remains
                # the fallback.
                if (
                    selected_t3_model is not None
                    and t3_provider is not None
                    and t3_provider in transports
                ):
                    # 2026-09-17 (user ruling): "all cloud requests should have
                    # the cloud model denoted".  The DIRECT /cloud path already
                    # emits a T3 tier_start carrying its model, but THIS ladder
                    # path emitted none -- `_invoke_meta_target` returns None and
                    # emits nothing of its own, so the escalation began with no
                    # announcement and the model appeared only on the verdict
                    # line.  Same event and payload shape, so both routes read
                    # identically.
                    await emit(CleanProgressEvent(
                        event_type="tier_start", tier="T3",
                        model_id=selected_t3_model,
                        elapsed_s=_now() - run_start,
                        metadata={"target": "cloud_sota"},
                    ))
                    try:
                        await _invoke_meta_target(
                            target="cloud_sota",
                            tier_label="T3",
                            result=result, work=work, harness=harness,
                            transports=transports, vram=vram, unit=unit,
                            generalist_model=selected_t3_model,
                            generalist_provider=t3_provider,
                            emit=emit, run_start=run_start,
                            cloud_effort=cloud_effort,
                            cloud_max_tokens=cloud_max_tokens,
                            cloud_silence_timeout_s=cloud_silence_s,
                            cloud_total_timeout_s=cloud_total_s,
                            cloud_guard_cfg=cloud_cfg,
                            checkpoint_interval_s=checkpoint_interval_s,
                            raw_token_writer=raw_token_writer,
                            artifact_store=artifact_store,
                            artifact_id=artifact_id,
                            evaluate_store=evaluate_store,
                            evaluate_prior_wp=evaluate_prior_wp,
                            user_intent=user_intent,
                            source_filename=source_filename,
                        )
                    except Exception as exc:
                        # A meta-target crash must NEVER kill execute():
                        # log loudly and continue to the next target
                        # (the ladder already ended; failing to escalate
                        # should surface as a failed attempt, not a dead
                        # stream that looks like a silent success).
                        print(
                            f"[pathology] model={selected_t3_model} "
                            f"err=t3_error: meta target '{target}' crashed: "
                            f"{type(exc).__name__}: {exc}",
                            file=sys.stderr, flush=True,
                        )
                    # D2 (2026-08-29): user policy — cloud failure is a
                    # HARD STOP.  By the time T3 runs, the local ladder
                    # is exhausted; falling through to the local
                    # generalist would be another multi-minute wasted
                    # attempt.  Only a chain of cloud models (out of
                    # scope) could justify continuing after T3 fails.
                    # A failed T3 still recorded its attempt truthfully
                    # (D4), so the failure is visible; we just do not
                    # degrade it with a doomed T5 attempt.
                    if cloud_on_failure == "hard_stop" and not result.passed:
                        print(
                            "[meta] on_failure=hard_stop: T3 failed, "
                            "ending request (no T5 fallthrough)",
                            file=sys.stderr, flush=True,
                        )
                        break
            elif target == "generalist_zero_shot":
                # T5 target: ALWAYS the local generalist (the local
                # generalist slot is not overrideable by cloud config --
                # T3 and T5 are separate paths).
                if (
                    generalist_model is not None
                    and generalist_provider in transports
                ):
                    await _invoke_t5(
                        result=result, work=work, harness=harness,
                        transports=transports, vram=vram, unit=unit,
                        generalist_model=generalist_model,
                        generalist_provider=generalist_provider,
                        emit=emit, run_start=run_start,
                    )
            # Future targets: local_council
    await emit(CleanProgressEvent(
        event_type="complete", tier=work.harness.tiers[-1], output=result.output, model_id=result.selected_model,
        elapsed_s=result.total_elapsed_s,
        metadata={
            "passed": result.passed,
            **({"pending_artifact_edit": True} if result.pending_artifact_edit else {}),
        },
    ))
    return result

