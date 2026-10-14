#!/usr/bin/env python3
r"""LiveCodeBench v6 comparison runner — THE canonical runner.

Runs a QID set selected by manifest. The default is the frozen 75-QID manifest:

  * benchmarks\LiveCodeBench\data\lcb75_prior_manifest.json (DEFAULT_MANIFEST)
  * an explicit subset of that manifest via --qids

Backends: ``coe`` (the CoE OpenAI-compatible server), ``ollama``/``ornith``
(direct, no CoE harness).  Direct Ollama runs use the serialized LCB prompt as
a raw prompt; the CoE mode sends the same prompt text through the server. Both
are graded against public and private tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import requests

WORKSPACE = Path(__file__).resolve().parent.parent
LCB_DIR = WORKSPACE / "benchmarks" / "LiveCodeBench"
DEFAULT_MANIFEST = LCB_DIR / "data" / "lcb75_prior_manifest.json"
OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_COE_URL = "http://127.0.0.1:8001/v1/chat/completions"
SEED = 42
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
REPEAT_PENALTY = 1.0
PRUNED_ORNITH_MODEL = "CoE-Ornith1.5-python-n1-55-19b-a3b-q4_k_m"
NUM_CTX = 32768
NUM_PREDICT = 24576
DIRECT_READ_TIMEOUT_S = 900
COE_READ_TIMEOUT_S = 900
LOOP_MATCH = 200
LOOP_WINDOW = 2000
LOOP_MIN = 2400

sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(LCB_DIR))
os.chdir(LCB_DIR)

from lcb_runner.lm_styles import LMStyle
from lcb_runner.prompts.code_generation import (
    format_prompt_generation,
    get_generic_question_template_answer,
)
from lcb_runner.utils.extraction_utils import extract_code
from lcb_runner.evaluation.compute_code_generation_metrics import (
    check_correctness as lcb_check_correctness,
)
from _run_lcb_naked import build_lcb_prompt as build_historical_direct_prompt


def load_problems(
    cache_path: Path,
    qids: list[str],
) -> tuple[list[SimpleNamespace], dict[str, Any]]:
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    by_id = {item["question_id"]: item for item in data["problems"]}
    missing = [qid for qid in qids if qid not in by_id]
    if missing:
        raise ValueError(f"Manifest QIDs missing from {cache_path}: {missing[:5]}")
    source_items = [by_id[qid] for qid in qids]
    problems = []
    for item in source_items:
        problems.append(SimpleNamespace(
            question_id=item["question_id"],
            question_title=item["question_title"],
            question_content=item["question_content"],
            difficulty=SimpleNamespace(value=item["difficulty"]),
            platform=SimpleNamespace(value=item["platform"]),
            starter_code=item.get("starter_code", ""),
            metadata=item.get("metadata", {}),
            public_test_cases=[SimpleNamespace(**test) for test in item["public_test_cases"]],
            private_test_cases=[SimpleNamespace(**test) for test in item["private_test_cases"]],
        ))
    meta = data.get("meta", {})
    if len(problems) != len(qids) or len(set(qids)) != len(qids):
        raise ValueError("Manifest must contain unique QIDs matching the loaded problems")
    if meta.get("release") not in (None, "release_v6"):
        raise ValueError(f"Unexpected LCB cache release metadata: {meta}")
    return problems, meta


CONTENT_FIELDS = (
    "question_id",
    "question_title",
    "question_content",
    "difficulty",
    "platform",
    "starter_code",
    "metadata",
    "public_test_cases",
    "private_test_cases",
)


def cache_content_sha256(cache_path: Path, qids: list[str]) -> str:
    """Hash the manifest QIDs' records, independent of file bytes and key order.

    Used when a rebuilt cache differs byte-for-byte from the pinned file but
    holds the same problem records.
    """
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    by_id = {item["question_id"]: item for item in data["problems"]}
    missing = [qid for qid in qids if qid not in by_id]
    if missing:
        raise ValueError(f"Manifest QIDs missing from {cache_path}: {missing[:5]}")
    records = [{field: by_id[qid].get(field) for field in CONTENT_FIELDS} for qid in qids]
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_manifest_problems(manifest_path: Path):
    manifest_path = manifest_path if manifest_path.is_absolute() else WORKSPACE / manifest_path
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache_path = Path(manifest["problem_cache"])
    if not cache_path.is_absolute():
        cache_path = WORKSPACE / cache_path
    cache_path = cache_path.resolve()
    cache_hash = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    qids = manifest["ordered_question_ids"]
    expected_hash = manifest.get("problem_cache_sha256")
    if expected_hash and cache_hash != expected_hash:
        expected_content = manifest.get("problem_cache_content_sha256")
        if not expected_content:
            raise ValueError(f"Problem cache hash mismatch for {cache_path}")
        content_hash = cache_content_sha256(cache_path, qids)
        if content_hash != expected_content:
            raise ValueError(
                f"Problem cache content mismatch for {cache_path}: "
                f"got {content_hash}, manifest expects {expected_content}"
            )
        print(
            "[cache] byte hash differs from manifest; the manifest QID records "
            "match problem_cache_content_sha256 (content-equivalent cache accepted)",
            flush=True,
        )

    problems, cache_meta = load_problems(cache_path, qids)
    counts = Counter(problem.difficulty.value for problem in problems)
    if dict(counts) != manifest.get("tiers"):
        raise ValueError(f"Manifest tier counts disagree with cache: {dict(counts)}")
    if manifest.get("release") != "release_v6":
        raise ValueError(f"Unsupported manifest release: {manifest.get('release')}")
    cache_meta = {**cache_meta, "release": "release_v6", "seed": manifest.get("request_seed")}
    return problems, cache_meta, manifest_path, cache_path, manifest


def select_problems_by_qid(
    problems: list[SimpleNamespace], requested_qids: list[str] | None
) -> list[SimpleNamespace]:
    if requested_qids is None:
        return problems
    if len(requested_qids) != len(set(requested_qids)):
        raise ValueError("--qids must not contain duplicates")
    available = {problem.question_id for problem in problems}
    missing = [qid for qid in requested_qids if qid not in available]
    if missing:
        raise ValueError(f"QIDs are not present in the selected problem set: {missing}")
    requested = set(requested_qids)
    return [problem for problem in problems if problem.question_id in requested]


def check_loop(text: str) -> bool:
    if len(text) < LOOP_MIN:
        return False
    tail = text[-LOOP_MATCH:]
    window = text[-(LOOP_MATCH + LOOP_WINDOW):-LOOP_MATCH]
    return bool(tail and tail in window)


def check_loop_tuned(text: str, *, match: int, window: int, minimum: int,
                     skip: int | None = None, min_occurrences: int = 3) -> bool:
    """Loop detector with caller-supplied thresholds.

    ALGORITHM COPIED FROM THE VALIDATED RUNTIME GUARD
    (``CoE_coder_16GB_final_0_5/runtime.py::repetition_loop_guard``), because a
    naive "does the trailing chunk appear once in the window?" test is WRONG and
    was already fixed there.  Its comment records the captured false positive:

        "a single repeat within the window is NOT enough -- legitimate code
         repeats near-identical blocks (ascending + descending sort branches,
         symmetric CSS rules). ... A genuine degenerate loop repeats the SAME
         chunk MANY times, so require >=3 occurrences of the chunk in the body
         overall."

    2026-09-30: my first version of this function required only ONE repeat and
    reproduced exactly that bug -- in the Tiel parent think-on run it flagged 33
    rows, **22 of which had fully correct code** (e.g. `2953` grade=102/102,
    `2730` grade=14/14).  Those are legitimate programs that happen to repeat a
    construct, not degenerate loops.  Requiring >=3 occurrences in the WHOLE
    body, and scanning the body rather than only the tail, is what makes the
    detector usable on real code.
    """
    if len(text) < minimum:
        return False
    body = text[(skip if skip is not None else minimum):]
    for index in range(0, max(0, len(body) - match), 50):
        chunk = body[index:index + match]
        if not chunk:
            continue
        if chunk in body[index + match:index + window]:
            if body.count(chunk) >= min_occurrences:
                return True
    return False


# ---------------------------------------------------------------------------
# 2026-09-30: DEGENERATE-GENERATION GUARD for NAKED (direct Ollama) RUNS ONLY.
#
# USER RULING (verbatim): "the guards must not be active during a CoE-coder
# benchmark run as the internal guards must be allowed to operate."
#
# Therefore this guard is:
#   * OPT-IN            -- `--guard` (default OFF), so existing behaviour is
#                          unchanged and prior runs stay reproducible.
#   * DIRECT-ONLY       -- `coe_call` has no guard and never will.  The CoE
#                          server owns its own pathology/loop/dither/ceiling
#                          machinery; a client-side guard would duplicate it
#                          and mask the server's own behaviour.
#   * FAILURE-COUNTING  -- a trip is recorded in `failure_flags` AND excludes
#                          the row from `effective_pass`, so a runaway
#                          generation is a FAIL rather than a timeout/abort.
#
# Guarded conditions (ANY trip => fail):
#   degenerated_loop     -- repeated trailing text (check_loop_tuned)
#   degenerated_length   -- cumulative output exceeded --guard-max-chars
#   degenerated_timeout  -- wall clock exceeded --guard-timeout
# ---------------------------------------------------------------------------
DEFAULT_GUARD_MAX_CHARS = 120000
DEFAULT_GUARD_TIMEOUT_S = 900
# Thresholds mirror the runtime's validated new_code_generation ladder values
# (loop_guard_min 1400 / skip 1400 / match 100 / window 400) rather than the
# old trailing-tail constants, so the naked guard and the server guard agree.
LOOP_GUARD_MATCH = 100
LOOP_GUARD_WINDOW = 400
LOOP_GUARD_MIN = 1400
LOOP_GUARD_SKIP = 1400
LOOP_GUARD_MIN_OCCURRENCES = 3


def is_pruned_ornith_model(model: str | None) -> bool:
    return bool(model) and model.split(":", 1)[0] == PRUNED_ORNITH_MODEL


def build_direct_prompt(problem: SimpleNamespace, think: bool) -> str:
    return build_historical_direct_prompt(
        {
            "question_content": problem.question_content,
            "starter_code": problem.starter_code,
        },
        closed_think=not think,
    )


def extract_final_python_block(text: str) -> str:
    """Extract the final Python block despite non-code prefill pseudo-fences."""
    candidates = [text.rfind("```python"), text.rfind("```Python")]
    opening = max(candidates)
    if opening >= 0:
        fence_end = text.find("\n", opening)
        if fence_end < 0:
            return ""
        content_start = fence_end + 1
        closing = text.find("```", content_start)
        if closing < 0:
            return text[content_start:].strip()
        return text[content_start:closing].strip()

    # CoE's assistant prefill ends with an open ``` fence. The provider
    # returns its continuation starting at the language label, not the fence.
    continued = text.lstrip()
    if continued.startswith("python\n"):
        code = continued[len("python\n"):]
        closing = code.rfind("```")
        if closing >= 0:
            code = code[:closing]
        return code.strip()
    return ""


def extract_direct_code(text: str) -> str:
    """Prefer the latest complete Python block, even if a later rewrite is truncated."""
    blocks = re.findall(
        r"```[ \t]*(?:python|py)[ \t]*\r?\n(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    complete = next((block.strip() for block in reversed(blocks) if block.strip()), "")
    if complete:
        return complete

    openings = list(re.finditer(
        r"```[ \t]*(?:python|py)[ \t]*\r?\n", text, flags=re.IGNORECASE
    ))
    if openings and text.rfind("```") == openings[-1].start():
        partial = text[openings[-1].end():].strip()
        if partial:
            return partial

    return extract_code(text, LMStyle.CodeQwenInstruct) or ""


def has_nonempty_think_block(text: str) -> bool:
    return any(content.strip() for content in re.findall(
        r"<think>(.*?)(?:</think>|$)", text, flags=re.DOTALL | re.IGNORECASE
    ))


def build_direct_request(model: str, prompt: str, think: bool) -> dict[str, Any]:
    body = {
        "model": model,
        "prompt": prompt,
        "raw": True,
        "stream": True,
        "options": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "repeat_penalty": REPEAT_PENALTY,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
            "seed": SEED,
            "stop": ["<|im_end|>", "<|endoftext|>"],
        },
    }
    if think:
        body["think"] = True
    return body


def direct_call(model: str, prompt: str, think: bool, *,
                guard: bool = False,
                guard_max_chars: int = DEFAULT_GUARD_MAX_CHARS,
                guard_timeout_s: int = DEFAULT_GUARD_TIMEOUT_S) -> dict[str, Any]:
    """Direct Ollama /api/generate call.

    ``guard`` enables the degenerate-generation guard (naked runs only -- see
    the module note above).  When on, a runaway counts as a FAILURE via the
    ``guard_trip`` field, and the stream is abandoned early so a pathological
    generation cannot burn the full read timeout.
    """
    body = build_direct_request(model, prompt, think)
    response_parts: list[str] = []
    thinking_parts: list[str] = []
    loop_detected = False
    guard_trip: str | None = None
    done_reason = None
    eval_count = None
    start = time.monotonic()
    try:
        with requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=body,
            stream=True,
            timeout=(15, DIRECT_READ_TIMEOUT_S),
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("response"):
                    response_parts.append(chunk["response"])
                if chunk.get("thinking"):
                    thinking_parts.append(chunk["thinking"])
                if is_pruned_ornith_model(model) and check_loop(
                    "".join(thinking_parts + response_parts)
                ):
                    loop_detected = True
                    break
                if guard:
                    assembled = "".join(thinking_parts + response_parts)
                    if check_loop_tuned(assembled, match=LOOP_GUARD_MATCH,
                                        window=LOOP_GUARD_WINDOW,
                                        minimum=LOOP_GUARD_MIN,
                                        skip=LOOP_GUARD_SKIP,
                                        min_occurrences=LOOP_GUARD_MIN_OCCURRENCES):
                        guard_trip = "degenerated_loop"
                        break
                    if len(assembled) > guard_max_chars:
                        guard_trip = "degenerated_length"
                        break
                    if time.monotonic() - start > guard_timeout_s:
                        guard_trip = "degenerated_timeout"
                        break
                if chunk.get("done"):
                    done_reason = chunk.get("done_reason")
                    eval_count = chunk.get("eval_count")
                    break
    except requests.RequestException as exc:
        return {
            "raw_output": "".join(response_parts),
            "thinking": "".join(thinking_parts),
            "loop_detected": loop_detected,
            "guard_trip": guard_trip,
            "done_reason": done_reason,
            "eval_count": eval_count,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "raw_output": "".join(response_parts),
        "thinking": "".join(thinking_parts),
        "loop_detected": loop_detected,
        "guard_trip": guard_trip,
        "done_reason": done_reason,
        "eval_count": eval_count,
        "error": None,
    }


def coe_call(url: str, prompt: str, think: bool, *, live: bool = False,
             raw_fh: Any = None) -> dict[str, Any]:
    """POST to the CoE server and return the assembled call record.

    ``live`` (2026-09-28) is DISPLAY-ONLY: it echoes content tokens in
    1-second batches and prints tier transitions as they arrive, so a long
    benchmark can be watched.  It changes NOTHING about accumulation: the
    same ``parts`` list is built either way, so ``raw_output``, extraction,
    grading, and every ``results.jsonl`` field are byte-identical with or
    without it.  ``raw_fh`` additionally persists the same 1-second batches
    with timestamps, for dither-density analysis.
    """
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "routing_override": {
            "domain": "domain_python",
            "task_type": "new_code_generation",
            "think": think,
        },
        "metadata": {"clear_session": True, "think": think, "seed": SEED},
        "stream": True,
    }
    parts: list[str] = []
    tiers: list[str] = []
    model_ids: list[str] = []
    retries: list[dict[str, Any]] = []
    tier_details: list[dict[str, Any]] = []
    finish_reason = None
    api_error = None
    # --- display-only live state (never feeds ``parts``) ---
    live_batch: list[str] = []
    live_tier = "T0"
    live_last_flush = time.monotonic()

    def _live_flush(force: bool = False) -> None:
        if not live_batch:
            return
        now = time.monotonic()
        if not force and now - live_last_flush < 1.0:
            return
        text = "".join(live_batch)
        live_batch.clear()
        elapsed = time.monotonic() - start
        if live:
            sys.stdout.write(f"\n[{live_tier} {elapsed:6.1f}s] {text}")
            sys.stdout.flush()
        if raw_fh is not None:
            raw_fh.write(f"[{live_tier} {elapsed:.0f}s] {text}\n")
            raw_fh.flush()

    start = time.monotonic()
    try:
        with requests.post(url, json=body, stream=True, timeout=(15, COE_READ_TIMEOUT_S)) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    metadata = delta.get("metadata") or {}
                    observed_model = choice.get("model") or metadata.get("model_id")
                    if observed_model and observed_model not in model_ids:
                        model_ids.append(str(observed_model))
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    if content:
                        parts.append(content)
                        if live or raw_fh is not None:
                            live_batch.append(content)
                            _live_flush()
                    for key in ("tier", "tier_start", "tier_complete"):
                        value = metadata.get(key)
                        if value and value not in tiers:
                            tiers.append(str(value))
                            if live:
                                _live_flush(force=True)
                                print(f"\n[tier] {value}", flush=True)
                                live_tier = str(value)
                    if metadata.get("tier_retry"):
                        retries.append(dict(metadata["tier_retry"]))
                    if metadata.get("tier_details"):
                        tier_details.append(dict(metadata["tier_details"]))
                    if metadata.get("error"):
                        api_error = str(metadata["error"])
                        if live:
                            _live_flush(force=True)
                            print(f"\n[pathology] {api_error}", flush=True)
                if event.get("error"):
                    api_error = str(event["error"])
                if finish_reason == "error" and api_error is None:
                    api_error = "API stream finished with error"
    except requests.RequestException as exc:
        api_error = f"{type(exc).__name__}: {exc}"
    _live_flush(force=True)
    if live:
        print(f"\n[call done {time.monotonic() - start:.1f}s tiers={tiers}]", flush=True)
    return {
        "raw_output": "".join(parts),
        "tiers": tiers,
        "model_ids": model_ids,
        "retries": retries,
        "tier_details": tier_details,
        "finish_reason": finish_reason,
        "error": api_error,
    }


def ornith_call(model: str, problem: SimpleNamespace, think: bool, *,
                guard: bool = False,
                guard_max_chars: int = DEFAULT_GUARD_MAX_CHARS,
                guard_timeout_s: int = DEFAULT_GUARD_TIMEOUT_S) -> dict[str, Any]:
    body = build_ornith_request(model, problem, think)
    content_parts: list[str] = []
    thinking_parts: list[str] = []
    assembled_parts: list[str] = []
    loop_detected = False
    guard_trip: str | None = None
    done_reason = None
    eval_count = None
    api_error = None
    start = time.monotonic()
    try:
        with requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=body,
            stream=True,
            timeout=(15, DIRECT_READ_TIMEOUT_S),
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                message = event.get("message") or {}
                content = message.get("content") or ""
                thinking = message.get("thinking") or message.get("reasoning") or ""
                if content:
                    content_parts.append(content)
                    assembled_parts.append(content)
                if thinking:
                    thinking_parts.append(thinking)
                    assembled_parts.append(thinking)
                if is_pruned_ornith_model(model) and check_loop("".join(assembled_parts)):
                    loop_detected = True
                    break
                if guard:
                    assembled = "".join(assembled_parts)
                    if check_loop_tuned(assembled, match=LOOP_GUARD_MATCH,
                                        window=LOOP_GUARD_WINDOW,
                                        minimum=LOOP_GUARD_MIN,
                                        skip=LOOP_GUARD_SKIP,
                                        min_occurrences=LOOP_GUARD_MIN_OCCURRENCES):
                        guard_trip = "degenerated_loop"
                        break
                    if len(assembled) > guard_max_chars:
                        guard_trip = "degenerated_length"
                        break
                    if time.monotonic() - start > guard_timeout_s:
                        guard_trip = "degenerated_timeout"
                        break
                if event.get("done"):
                    done_reason = event.get("done_reason")
                    eval_count = event.get("eval_count")
                    break
    except requests.RequestException as exc:
        api_error = f"{type(exc).__name__}: {exc}"
    return {
        "raw_output": "".join(assembled_parts),
        "thinking": "".join(thinking_parts),
        "loop_detected": loop_detected,
        "guard_trip": guard_trip,
        "done_reason": done_reason,
        "eval_count": eval_count,
        "error": api_error,
    }


def build_ornith_request(
    model: str, problem: SimpleNamespace, think: bool
) -> dict[str, Any]:
    system = (
        "You are an expert Python competitive programmer. "
        "Respond with exactly TWO ```python code blocks and nothing else. "
        "First: a DRAFT block with inline # comments for your working. "
        "Second: a FINAL clean solution."
    )
    user = (
        "You will be given a question (problem specification) and will generate "
        "a correct Python program that matches the specification and passes all tests. "
        "Respond with exactly two ```python code blocks: first a DRAFT with inline # comments, "
        "then a FINAL clean solution. Nothing outside the code blocks.\n\n"
        f"Question: {problem.question_content}\n\n"
    )
    if problem.starter_code:
        user += (
            "Use the provided function signature. Generate Python3 code and wrap "
            "your entire response in a single ```python code block.\n"
            f"```python\n{problem.starter_code}\n```\n"
        )
    else:
        user += "Generate Python3 code. Wrap your entire response in a single ```python code block.\n"

    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": True,
        "think": think,
        "options": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "repeat_penalty": 1.05 if is_pruned_ornith_model(model) else REPEAT_PENALTY,
            "repeat_last_n": 1024,
            "num_ctx": 32768,
            "num_predict": 24576,
            "seed": SEED,
        },
    }
def run_config(
    args: argparse.Namespace,
    cache_path: Path,
    manifest_path: Path | None = None,
    problem_count: int = 150,
) -> dict[str, Any]:
    coe_mode = args.backend == "coe"
    ornith_mode = args.backend == "ornith"
    direct_mode = args.backend in {"ollama", "ornith"}
    return {
        "backend": args.backend,
        "model": args.model if direct_mode else None,
        "think": args.think,
        "coe_url": args.coe_url if args.backend == "coe" else None,
        "cache": str(cache_path),
        "cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        "manifest": str(manifest_path) if manifest_path else None,
        "manifest_sha256": (
            hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            if manifest_path else None
        ),
        "release": "release_v6",
        "n_per_tier": problem_count // 3,
        "problem_count": problem_count,
        "seed": SEED,
        "prompt_style": (
            "historical naked-LCB code-only prompt"
            if args.backend == "ollama"
            else "Ornith native-chat think-on coding prompt"
            if ornith_mode
            else "historical LCB CodeQwenInstruct prompt as user message; CoE harness prompt"
        ),
        "think_off_prefill": args.backend == "ollama" and args.think == "off",
        "thinking_enabled": args.think == "on",
        "loop_guard_enabled": direct_mode and is_pruned_ornith_model(args.model),
        # 2026-09-30: the degenerate-generation guard is DIRECT-ONLY by design.
        # It is forced False for --backend coe so a CoE run can never have its
        # internal guard behaviour masked by a client-side trip.
        "degenerate_guard_enabled": bool(getattr(args, "guard", False)) and direct_mode,
        "degenerate_guard_max_chars": (
            getattr(args, "guard_max_chars", DEFAULT_GUARD_MAX_CHARS)
            if direct_mode else None
        ),
        "degenerate_guard_timeout_s": (
            getattr(args, "guard_timeout", DEFAULT_GUARD_TIMEOUT_S)
            if direct_mode else None
        ),
        "request_read_timeout_s": (
            COE_READ_TIMEOUT_S if coe_mode else DIRECT_READ_TIMEOUT_S
        ),
        "code_extractor": (
            "latest complete Python block; truncated final-block fallback; LCB fallback"
            if direct_mode else "last python fenced block"
        ),
        "grader": "lcb_runner.evaluation.compute_code_generation_metrics.check_correctness",
        "allow_t3_escalation": args.backend == "coe",
        "temperature": 0.6 if ornith_mode or coe_mode else TEMPERATURE,
        "top_p": 1.0 if coe_mode else 0.95 if ornith_mode else TOP_P,
        "top_k": -1 if coe_mode else 20 if ornith_mode else TOP_K,
        "repeat_penalty": (
            1.05
            if coe_mode or (
                ornith_mode
                and is_pruned_ornith_model(args.model)
            )
            else REPEAT_PENALTY
        ),
        "repeat_last_n": 1024 if coe_mode or ornith_mode else None,
        "num_ctx": 32768 if ornith_mode or coe_mode else NUM_CTX,
        "num_predict": 24576 if ornith_mode or coe_mode else NUM_PREDICT,
    }


def evaluate_record(problem: SimpleNamespace, call: dict[str, Any], backend: str,
                    elapsed_s: float) -> dict[str, Any]:
    raw_output = call.get("raw_output", "")
    if backend == "coe":
        code = extract_final_python_block(raw_output)
        extractor = "last_python_fence"
    else:
        code = extract_direct_code(raw_output)
        extractor = "lcb_codeqwen_with_closed_python_fallback"
    passed_tests, total_tests, grade_error = grade_code(code, problem)
    grade_pass = bool(total_tests and passed_tests == total_tests and grade_error is None)
    flags = []
    if call.get("error"):
        flags.append("request_error")
    if backend in {"ollama", "ornith"}:
        if call.get("loop_detected"):
            flags.append("loop_detected")
        # 2026-09-30: degenerate-generation guard (naked runs only).  A trip is
        # a FAILURE, not an abort -- that is the whole point of the guard.
        if call.get("guard_trip"):
            flags.append(str(call["guard_trip"]))
        if call.get("done_reason") in {"length", "max_tokens"}:
            flags.append("token_limit")
        if call.get("eval_count") is not None and call["eval_count"] >= NUM_PREDICT:
            flags.append("token_limit")
    if not code.strip():
        flags.append("no_code_extracted")
    effective_pass = grade_pass and not any(
        flag in flags for flag in (
            "request_error", "no_code_extracted",
            # Guard trips disqualify the row outright.
            "degenerated_loop", "degenerated_length", "degenerated_timeout",
        )
    )
    observed_models = call.get("model_ids", [])
    t3_used = backend == "coe" and (
        "T3" in call.get("tiers", [])
        or any(":cloud" in model.lower() or "deepseek-v4.1-flash" in model.lower()
               for model in observed_models)
    )
    return {
        "question_id": problem.question_id,
        "difficulty": problem.difficulty.value,
        "backend": backend,
        "elapsed_s": round(elapsed_s, 3),
        "passed_tests": passed_tests,
        "total_tests": total_tests,
        "grade_pass": grade_pass,
        "effective_pass": effective_pass,
        "extractor": extractor,
        "failure_flags": sorted(set(flags)),
        "grade_error": grade_error,
        "loop_detected": bool(call.get("loop_detected", False)),
        "done_reason": call.get("done_reason"),
        "eval_count": call.get("eval_count"),
        "tiers": call.get("tiers", []),
        "model_ids": observed_models,
        "t3_used": t3_used,
        "retries": call.get("retries", []),
        "tier_details": call.get("tier_details", []),
        "finish_reason": call.get("finish_reason"),
        "request_error": call.get("error"),
        "extracted_code": code,
        "raw_output": raw_output,
        "thinking": call.get("thinking", ""),
        "think_text_leak": has_nonempty_think_block(raw_output),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_tier: dict[str, dict[str, int]] = {}
    for row in rows:
        tier = row["difficulty"]
        stats = by_tier.setdefault(tier, {"passed": 0, "total": 0})
        stats["total"] += 1
        stats["passed"] += int(row["effective_pass"])
    passed = sum(item["passed"] for item in by_tier.values())
    total = sum(item["total"] for item in by_tier.values())
    return {
        "passed": passed,
        "total": total,
        "pass_rate_pct": round(100 * passed / total, 2) if total else 0,
        "by_difficulty": by_tier,
    }


def grade_code(code: str, problem: SimpleNamespace) -> tuple[int, int, str | None]:
    tests = problem.public_test_cases + problem.private_test_cases
    total = len(tests)
    if not code.strip():
        return 0, total, "EmptyCode"
    if not tests:
        return 0, 0, "NoTests"
    sample = {
        "input_output": json.dumps({
            "inputs": [test.input for test in tests],
            "outputs": [test.output for test in tests],
            "fn_name": problem.metadata.get("func_name"),
        })
    }
    # 2026-09-30: the upstream grader can CRASH rather than return a verdict.
    # `lcb_check_correctness` does `return result[0], metadata_list[0]`, and
    # `metadata_list` comes back EMPTY for some inputs, raising
    # `IndexError: list index out of range` from inside the multiprocessing
    # manager.  Unhandled, that kills the whole run: it terminated 3 of the 4
    # naked runs on 2026-09-30 (`1/4 EXIT=1`, `3/4 EXIT=1`) after hours of
    # compute, losing every row in them.
    #
    # A grader crash is an INFRASTRUCTURE failure, not a model failure, so it is
    # recorded as a distinct `grader_crash` error and the row is marked failed
    # (0/N) rather than aborting the process.  The distinction matters: a
    # grader_crash row must NOT be read as "the model got it wrong".  The flag
    # surfaces in `failure_flags` via `grade_error`, and the run continues so
    # the remaining 74 problems are not sacrificed.
    try:
        results, metadata = lcb_check_correctness(sample, code, timeout=6, debug=False)
    except (IndexError, ValueError, TypeError, KeyError) as exc:
        return 0, total, f"grader_crash: {type(exc).__name__}: {exc}"[:500]
    except Exception as exc:  # manager/transport-level faults
        return 0, total, f"grader_crash: {type(exc).__name__}: {exc}"[:500]
    if results is None:
        return 0, total, "grader_crash: no results returned"
    passed = sum(result is True for result in results)
    is_full_pass = len(results) == total and passed == total
    return passed, total, None if is_full_pass else str(metadata)[:500]


def regrade_run(run_dir: Path) -> None:
    """Regrade a saved run with upstream LCB checks, preserving original files."""
    source_path = run_dir / "results.jsonl"
    metadata_path = run_dir / "metadata.json"
    if not source_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Expected results.jsonl and metadata.json in {run_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = metadata.get("config", metadata)
    if not config.get("manifest"):
        raise ValueError(f"{run_dir} has no manifest; it cannot be regraded with this runner")
    problems, _, _, _, _ = load_manifest_problems(Path(config["manifest"]))
    problem_by_id = {problem.question_id: problem for problem in problems}
    rows = [
        json.loads(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    corrected: list[dict[str, Any]] = []
    for row in rows:
        problem = problem_by_id[row["question_id"]]
        call = {
            "raw_output": row.get("raw_output", ""),
            "thinking": row.get("thinking", ""),
            "tiers": row.get("tiers", []),
            "model_ids": row.get("model_ids", []),
            "retries": row.get("retries", []),
            "tier_details": row.get("tier_details", []),
            "finish_reason": row.get("finish_reason"),
            "error": row.get("request_error", row.get("error")),
            "loop_detected": row.get("loop_detected", False),
            "done_reason": row.get("done_reason"),
            "eval_count": row.get("eval_count"),
        }
        updated = evaluate_record(
            problem, call, config.get("backend", row.get("backend", "ollama")),
            float(row.get("elapsed_s", 0.0)),
        )
        for key in ("model", "think"):
            if key in row:
                updated[key] = row[key]
        corrected.append(updated)

    result_path = run_dir / "results_official.jsonl"
    with result_path.open("w", encoding="utf-8", buffering=1) as handle:
        for row in corrected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(corrected)
    summary["think_text_leak_rows"] = sum(bool(row["think_text_leak"]) for row in corrected)
    summary["grader"] = "lcb_runner.evaluation.compute_code_generation_metrics.check_correctness"
    if config.get("backend") == "coe":
        summary["t3_split"] = {
            "local_only": summarize([row for row in corrected if not row["t3_used"]]),
            "t3_involved": summarize([row for row in corrected if row["t3_used"]]),
        }
    summary["rows"] = len(corrected)
    summary["regraded_from"] = str(source_path)
    summary["config"] = config
    score_path = run_dir / "score_official.json"
    score_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[official regrade] {summary['passed']}/{summary['total']} ({summary['pass_rate_pct']}%)")
    for tier in ("easy", "medium", "hard"):
        if tier in summary["by_difficulty"]:
            result = summary["by_difficulty"][tier]
            print(f"  {tier}: {result['passed']}/{result['total']}")
    for label, result in summary.get("t3_split", {}).items():
        print(f"  {label}: {result['passed']}/{result['total']}")
    if config.get("backend") == "coe":
        print(f"  responses with visible think text: {summary['think_text_leak_rows']}/{len(corrected)}")
    print(f"[official regrade] {result_path}")
    print(f"[score] {score_path}")


def self_test() -> None:
    problems, _, _, _, manifest = load_manifest_problems(DEFAULT_MANIFEST)
    api_prompt = get_generic_question_template_answer(problems[0])
    assert "<|im_start|>" not in api_prompt
    assert problems[0].question_content in api_prompt
    coe_prompt = format_prompt_generation(problems[0], LMStyle.CodeQwenInstruct)
    assert problems[0].question_content in coe_prompt
    assert "Two-Pass protocol" in coe_prompt
    prior_problems = problems
    assert [problem.question_id for problem in prior_problems] == manifest["ordered_question_ids"]
    assert dict(Counter(problem.difficulty.value for problem in prior_problems)) == {
        "easy": 25, "medium": 25, "hard": 25,
    }
    selected_problems = select_problems_by_qid(
        prior_problems, [prior_problems[3].question_id, prior_problems[0].question_id]
    )
    assert [problem.question_id for problem in selected_problems] == [
        prior_problems[0].question_id, prior_problems[3].question_id,
    ]
    try:
        select_problems_by_qid(prior_problems, [prior_problems[0].question_id] * 2)
    except ValueError:
        pass
    else:
        raise AssertionError("QID selector must reject duplicates")
    direct_off_prompt = build_direct_prompt(problems[0], think=False)
    direct_on_prompt = build_direct_prompt(problems[0], think=True)
    assert direct_off_prompt.endswith("<think>\n</think>\n\n")
    assert not direct_on_prompt.endswith("<think>\n</think>\n\n")
    assert "Two-Pass protocol" not in direct_off_prompt
    ornith_problem = SimpleNamespace(
        question_content="sample problem", starter_code="", 
    )
    ornith_body = build_ornith_request(PRUNED_ORNITH_MODEL, ornith_problem, think=True)
    ornith_parent_body = build_ornith_request("ornith-parent-q4:latest", ornith_problem, think=True)
    assert ornith_body["think"] is True
    assert all(message["role"] in {"system", "user"} for message in ornith_body["messages"])
    assert ornith_body["options"]["num_ctx"] == 32768
    assert ornith_body["options"]["num_predict"] == 24576
    assert ornith_body["options"]["top_k"] == 20
    assert ornith_body["options"]["repeat_penalty"] == 1.05
    assert ornith_body["options"]["repeat_last_n"] == 1024
    assert ornith_parent_body["options"]["repeat_penalty"] == 1.0
    assert is_pruned_ornith_model(PRUNED_ORNITH_MODEL)
    assert is_pruned_ornith_model(f"{PRUNED_ORNITH_MODEL}:latest")
    assert not is_pruned_ornith_model("ornith-parent-q4:latest")
    direct_off_body = build_direct_request("model", "prompt", think=False)
    direct_on_body = build_direct_request("model", "prompt", think=True)
    assert "think" not in direct_off_body
    assert direct_on_body["think"] is True
    assert direct_off_body["options"] == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "repeat_penalty": 1.0,
        "num_ctx": 32768,
        "num_predict": 24576,
        "seed": 42,
        "stop": ["<|im_end|>", "<|endoftext|>"],
    }
    fake_call_problem = SimpleNamespace(
        question_id="self-test",
        difficulty=SimpleNamespace(value="easy"),
        public_test_cases=[SimpleNamespace(input="1", output="2")],
        private_test_cases=[],
        metadata={"func_name": "f"},
    )
    assert grade_code("def f(x): return x + 1", fake_call_problem) == (1, 1, None)
    assert grade_code("def f(x): return x", fake_call_problem)[:2] == (0, 1)
    loop_flagged_pass = evaluate_record(
        fake_call_problem,
        {"raw_output": "```python\ndef f(x): return x + 1\n```", "loop_detected": True},
        "ornith",
        0.0,
    )
    assert loop_flagged_pass["grade_pass"] and loop_flagged_pass["effective_pass"]
    assert "loop_detected" in loop_flagged_pass["failure_flags"]
    decimal_stdio_problem = SimpleNamespace(
        public_test_cases=[SimpleNamespace(input="1 2\n", output="3\n")],
        private_test_cases=[],
        metadata={},
    )
    assert grade_code("print(3.0)", decimal_stdio_problem) == (1, 1, None)
    large_stdio_problem = SimpleNamespace(
        public_test_cases=[SimpleNamespace(input="1" * 50001, output="50001\n")],
        private_test_cases=[],
        metadata={},
    )
    assert grade_code(
        "import sys\nprint(len(sys.stdin.read()))", large_stdio_problem
    ) == (1, 1, None)
    two_pass = "```python\ndef f(x): return 0\n```\n```python\ndef f(x): return 1\n```"
    assert "return 1" in (extract_code(two_pass, LMStyle.CodeQwenInstruct) or "")
    extra_close = "```python\ndef f(x): return 3\n```\n```"
    assert "return 3" in extract_direct_code(extra_close)
    completed_then_truncated = (
        "```python\ndef f(x): return 4\n```\n"
        "Checking the answer again.\n```python\ndef f(x): return"
    )
    assert extract_direct_code(completed_then_truncated) == "def f(x): return 4"
    truncated_only = "```python\ndef f(x): return 5"
    assert extract_direct_code(truncated_only) == "def f(x): return 5"
    completed_rewrites = (
        "```python\ndef f(x): return 6\n```\n"
        "```python\ndef f(x): return 7\n```"
    )
    assert extract_direct_code(completed_rewrites) == "def f(x): return 7"
    assert extract_direct_code("reasoning text only with no Python block") == ""
    coe_stream = (
        "draft\n```python\ndef f(x): return 0\n```\n"
        "```<think>\nnon-code prefill\n```\n```python\ndef f(x): return 1\n```"
    )
    final_block = extract_final_python_block(coe_stream)
    assert "return 1" in final_block and "return 0" not in final_block
    open_prefill = "python\ndef f(x): return 2\n```"
    assert "return 2" in extract_final_python_block(open_prefill)
    assert not has_nonempty_think_block("<think>\n\n</think>")
    assert has_nonempty_think_block("<think>\nreasoning\n</think>")
    assert check_loop("a" * 2400 + "b" * 200 + "a" * 200)
    print(f"SELF-TEST PASS: default {len(problems)}-QID and prior {len(prior_problems)}-QID manifests, prompt/extraction/grading/loop checks")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--backend", choices=("ollama", "ornith", "coe"), default="ollama")
    parser.add_argument("--model", default="qwen3.6-parent-humaneval:latest")
    parser.add_argument("--think", choices=("off", "on"), default="off")
    parser.add_argument("--coe-url", default=os.environ.get("COE_CODER_URL", DEFAULT_COE_URL))
    parser.add_argument("--out", type=Path, required=False)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="ordered QID manifest and its pinned problem cache (default: 75-QID manifest)")
    parser.add_argument("--qids", nargs="+", default=None,
                        help="run only these QIDs from the selected cache/manifest")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--live", action="store_true",
                        help="echo tokens in 1s batches + tier transitions live "
                             "(DISPLAY ONLY; results.jsonl is byte-identical)")
    parser.add_argument("--guard", action="store_true",
                        help="NAKED (direct Ollama) runs only: enable the "
                             "degenerate-generation guard.  A loop/length/timeout "
                             "trip counts as a FAILURE.  NEVER applies to --backend coe: "
                             "the CoE server's own internal guards must operate.")
    parser.add_argument("--guard-max-chars", type=int, default=DEFAULT_GUARD_MAX_CHARS,
                        help=f"guard: abandon a generation past this many chars "
                             f"(default {DEFAULT_GUARD_MAX_CHARS})")
    parser.add_argument("--guard-timeout", type=int, default=DEFAULT_GUARD_TIMEOUT_S,
                        help=f"guard: abandon a generation past this many seconds "
                             f"(default {DEFAULT_GUARD_TIMEOUT_S})")
    parser.add_argument("--raw-out", type=Path, default=None,
                        help="persist the same 1s live batches with timestamps")
    parser.add_argument("--regrade-run", type=Path, default=None,
                        help="offline regrade any saved run with upstream LCB checking")
    args = parser.parse_args()

    if args.regrade_run is not None:
        regrade_run(args.regrade_run if args.regrade_run.is_absolute() else WORKSPACE / args.regrade_run)
        return

    if args.self_test:
        self_test()
        return
    if args.out is None:
        parser.error("--out is required unless --self-test is used")

    problems, cache_meta, manifest_path, cache_path, _ = load_manifest_problems(args.manifest)
    try:
        problems = select_problems_by_qid(problems, args.qids)
    except ValueError as exc:
        parser.error(str(exc))
    out_dir = args.out if args.out.is_absolute() else WORKSPACE / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    metadata_path = out_dir / "metadata.json"
    config = run_config(args, cache_path, manifest_path, len(problems))
    if args.qids is not None:
        config["qid_subset"] = [problem.question_id for problem in problems]
        config["qid_subset_tier_counts"] = dict(Counter(
            problem.difficulty.value for problem in problems
        ))
        config["n_per_tier"] = None
    if results_path.exists():
        if not args.resume:
            parser.error(f"{results_path} exists; use --resume only for the same run")
        if not metadata_path.exists() or json.loads(metadata_path.read_text(encoding="utf-8")) != config:
            parser.error("existing run metadata differs; refusing unsafe resume")
    else:
        metadata_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    rows_by_qid: dict[str, dict[str, Any]] = {}
    if args.resume and results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows_by_qid[row["question_id"]] = row
    remaining = [p for p in problems if p.question_id not in rows_by_qid]
    print(f"[cache] {len(problems)} QIDs ({cache_meta.get('release')} seed={cache_meta.get('seed')})")
    direct_mode = args.backend in {"ollama", "ornith"}
    print(f"[run] backend={args.backend} model={args.model if direct_mode else args.coe_url} think={args.think} remaining={len(remaining)}")

    raw_fh = None
    # 2026-09-30: the degenerate-generation guard is DIRECT-ONLY.  Forcing it
    # off for --backend coe guarantees the CoE server's internal guards are the
    # only guards in play during a CoE-coder benchmark run.
    _guard_on = bool(args.guard) and direct_mode
    if args.guard and not direct_mode:
        print("[guard] IGNORED for --backend coe: the server's internal guards "
              "must operate unimpeded")
    elif _guard_on:
        print(f"[guard] degenerate-generation guard ON (naked run): "
              f"loop={LOOP_GUARD_MATCH}/{LOOP_GUARD_WINDOW}/{LOOP_GUARD_MIN} "
              f"max_chars={args.guard_max_chars} timeout={args.guard_timeout}s "
              f"-> trips count as FAIL")
    if args.raw_out is not None:
        raw_path = args.raw_out if args.raw_out.is_absolute() else WORKSPACE / args.raw_out
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_fh = raw_path.open("a", encoding="utf-8", buffering=1)
        print(f"[live] raw token stream -> {raw_path}")
    if args.live:
        print("[live] token echo ON (display only; results.jsonl unchanged)")

    with results_path.open("a", encoding="utf-8", buffering=1) as output:
        for index, problem in enumerate(remaining, 1):
            prompt = (
                build_direct_prompt(problem, args.think == "on")
                if args.backend == "ollama"
                else format_prompt_generation(problem, LMStyle.CodeQwenInstruct)
            )
            started = time.monotonic()
            if args.live or raw_fh is not None:
                print(f"\n[{index}/{len(remaining)}] {problem.question_id} "
                      f"({problem.difficulty.value}) ...", flush=True)
            if args.backend == "ollama":
                call = direct_call(args.model, prompt, args.think == "on",
                                   guard=_guard_on,
                                   guard_max_chars=args.guard_max_chars,
                                   guard_timeout_s=args.guard_timeout)
            elif args.backend == "ornith":
                call = ornith_call(args.model, problem, args.think == "on",
                                   guard=_guard_on,
                                   guard_max_chars=args.guard_max_chars,
                                   guard_timeout_s=args.guard_timeout)
            else:
                # --backend coe: NO guard.  The CoE server owns its pathology /
                # loop / dither / ceiling machinery and it must run unimpeded
                # (user ruling 2026-09-30).
                call = coe_call(args.coe_url, prompt, args.think == "on",
                                live=args.live, raw_fh=raw_fh)
            row = evaluate_record(problem, call, args.backend, time.monotonic() - started)
            row["model"] = args.model if direct_mode else None
            row["think"] = args.think
            rows_by_qid[problem.question_id] = row
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"[{index:3d}/{len(remaining)}] {problem.question_id:18s} "
                f"{problem.difficulty.value:6s} "
                f"{'PASS' if row['effective_pass'] else 'FAIL'} "
                f"grade={row['passed_tests']}/{row['total_tests']} "
                f"flags={','.join(row['failure_flags']) or '-'} "
                f"elapsed={row['elapsed_s']:.1f}s",
                flush=True,
            )

    ordered_rows = [rows_by_qid[p.question_id] for p in problems if p.question_id in rows_by_qid]
    summary = summarize(ordered_rows)
    if args.backend == "coe":
        summary["t3_split"] = {
            "local_only": summarize([row for row in ordered_rows if not row["t3_used"]]),
            "t3_involved": summarize([row for row in ordered_rows if row["t3_used"]]),
        }
    summary.update({"config": config, "rows": len(ordered_rows)})
    (out_dir / "score.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[score] {summary['passed']}/{summary['total']} ({summary['pass_rate_pct']}%)")
    for tier in ("easy", "medium", "hard"):
        if tier in summary["by_difficulty"]:
            item = summary["by_difficulty"][tier]
            print(f"  {tier}: {item['passed']}/{item['total']}")
    if args.backend == "coe":
        for label, item in summary["t3_split"].items():
            print(f"  {label}: {item['passed']}/{item['total']}")
    print(f"[out] {out_dir}")
    if raw_fh is not None:
        raw_fh.close()
        print("[live] raw token stream closed")


if __name__ == "__main__":
    main()