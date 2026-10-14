"""
HumanEval cross-domain benchmark — uses the established LCB two-block
draft+final pattern (qwen3.6/lcb_bench.py) so the model:

  * has thinking disabled (think=False) and a closed-think assistant
    prefill injected via /api/chat (so Qwen3.6 doesn't burn tokens on
    a multi-thousand-token reasoning trace);
  * is told to output exactly TWO ```` ```python ```` blocks: a DRAFT
    where it works through the problem with inline ``#`` comments, and a
    FINAL clean solution.  We extract the LAST block (the FINAL) — this
    is robust whether the model wrote just the body or the whole ``def``;
  * is graded by executing `check(candidate)` in a sandboxed subprocess
    with the captured test code from HumanEval.jsonl.

Usage:
    python run_humaneval.py --model qwen3.6:35b-a3b-q4_K_M --n 164
    python run_humaneval.py --model qwen3.6:35b-a3b-q4_K_M --n 164 --stride 4   # every 4th
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = WORKSPACE_ROOT / "benchmarks" / "data" / "HumanEval.jsonl"
OLLAMA_URL = "http://localhost:11434"
OPENAI_BASE_URL = "http://localhost:8001/v1"
DEFAULT_DIRECT_MODEL = "JThomas-CoE/coe-qwen3.6-hc-coding_python-r25-19b-a3b-q4_k_m"
COE_DOMAIN = "domain_python"
COE_TASK_TYPE = "new_code_generation"
ASSISTANT_PREFILL = "<think>\n</think>\n\n```python\n"   # closed-think + open first code fence
THINK_ON_SYSTEM_PROMPT = (
    "You are a Python coding assistant. Provide correct, idiomatic, maintainable Python solutions.\n\n"
    "You are a competent Python language programmer. Reason through the problem in your native thinking, "
    "then output exactly ONE clean fenced code block in that language containing the complete solution. "
    "No prose outside the code block.\n\n"
    "Output exactly ONE clean fenced code block, in the language the request specifies, with the complete "
    "solution. No prose, no draft block -- only the final code."
)
LEAN_SYSTEM_PROMPT = (
    "You are a Python coding assistant. Solve the task and return only the complete "
    "Python implementation in one ```python code block. Do not include analysis or explanation."
)
LOOP_GUARD_MATCH  = 200
LOOP_GUARD_WINDOW = 2000
LOOP_GUARD_MIN    = 2400
STREAM_TIMEOUT_S  = 120
RUN_SCHEMA = "coe-humaneval-flex-v1"

SYSTEM_PROMPT = (
    "You are an expert Python competitive programmer. "
    "Respond with exactly TWO ```python code blocks and nothing else. "
    "First: a DRAFT block where you reason through the problem using inline # comments "
    "and work through edge cases. "
    "Second: the FINAL clean solution with no prose. "
    "No text outside the code blocks. "
    "Complete the function exactly as specified by the prompt — do not rename, "
    "do not add a wrapper, do not print, just return the value."
)

_FENCE_RE      = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_THINK_RE      = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_OPEN_THINK_RE = re.compile(r"<think>.*", re.DOTALL)

# ---------------------------------------------------------------------------
# Loop guard
# ---------------------------------------------------------------------------
def _check_loop(s: str) -> bool:
    if len(s) < LOOP_GUARD_MIN:
        return False
    tail = s[-LOOP_GUARD_MATCH:]
    window = s[-(LOOP_GUARD_MATCH + LOOP_GUARD_WINDOW):-LOOP_GUARD_MATCH]
    return tail in window


def build_user_content(prompt: str) -> str:
    return (
        "Complete the following Python function. The function body is what "
        "you need to write. Respond with exactly two ```python code blocks: "
        "first a DRAFT (working through the approach with inline # comments), "
        "then a FINAL clean solution. Nothing outside the code blocks.\n\n"
        f"```python\n{prompt}\n```"
    )


def build_direct_messages(
    user_content: str,
    *,
    think: bool,
    prompt_profile: str = "canonical",
) -> list[dict[str, str]]:
    if prompt_profile == "lean":
        system_prompt = LEAN_SYSTEM_PROMPT
    elif prompt_profile == "canonical":
        system_prompt = THINK_ON_SYSTEM_PROMPT if think else SYSTEM_PROMPT
    else:
        raise ValueError(f"Unknown direct prompt profile: {prompt_profile}")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    if prompt_profile == "canonical" and not think:
        messages.append({"role": "assistant", "content": ASSISTANT_PREFILL})
    return messages


def build_ollama_payload(
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    seed: int,
    num_ctx: int,
    top_k: int,
    top_p: float,
    think: bool,
    prompt_profile: str = "canonical",
) -> dict[str, Any]:
    options = {
        "temperature": temperature,
        "num_predict": max_tokens,
        "repeat_penalty": 1.05,
        "seed": seed,
        "num_threads": 28,
        "num_ctx": num_ctx,
        "top_k": top_k,
        "top_p": top_p,
    }
    if think or prompt_profile == "lean":
        options["repeat_last_n"] = 1024
    return {
        "model": model,
        "messages": messages,
        "stream": True,
        "think": think,
        "options": options,
    }


def _openai_v1_url(base_url: str, endpoint: str) -> str:
    base = base_url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return f"{base}/{endpoint.lstrip('/')}"


def build_openai_payload(
    prompt: str,
    *,
    openai_model: str | None,
    coe_domain: str | None,
    coe_task_type: str | None,
    think: bool,
    temperature: float,
    max_tokens: int,
    seed: int,
) -> tuple[dict[str, Any], str]:
    """Build either a generic OpenAI request or CoE's native harness request."""
    if bool(coe_domain) != bool(coe_task_type):
        raise ValueError("CoE mode requires both --coe-domain and --coe-task-type")

    if coe_domain:
        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "metadata": {
                "think": think,
                "clear_session": True,
                "no_inject": True,
                "seed": seed,
            },
            "routing_override": {
                "domain": coe_domain,
                "task_type": coe_task_type,
            },
        }
        profile = f"coe:{coe_domain}/{coe_task_type}:think={'on' if think else 'off'}"
        return payload, profile

    if not openai_model:
        raise ValueError("--openai-model is required for generic OpenAI-compatible mode")
    payload = {
        "model": openai_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_content(prompt)},
        ],
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
    }
    return payload, "openai:canonical-prompt:no-prefill"


def parse_openai_chunk(chunk: dict[str, Any]) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    """Extract visible deltas and CoE routing/tier metadata from one SSE chunk."""
    content_parts: list[str] = []
    route: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    for choice in chunk.get("choices", []) or []:
        delta = choice.get("delta") or {}
        text = delta.get("content")
        if isinstance(text, str) and text:
            content_parts.append(text)
        delta_metadata = delta.get("metadata") or {}
        if isinstance(delta_metadata, dict):
            maybe_route = delta_metadata.get("coe_routing")
            if isinstance(maybe_route, dict):
                route.update(maybe_route)
            for key in ("tier_start", "tier_complete", "status", "error", "checkpoint"):
                if key in delta_metadata:
                    value = delta_metadata[key]
                    event = value if isinstance(value, dict) else {"tier": value}
                    event["event_type"] = key
                    event.setdefault("model_id", delta.get("model") or chunk.get("model", ""))
                    events.append(event)
        message = choice.get("message") or {}
        text = message.get("content")
        if isinstance(text, str) and text:
            content_parts.append(text)
    top_metadata = chunk.get("metadata") or {}
    if isinstance(top_metadata, dict):
        maybe_route = top_metadata.get("coe_routing") or chunk.get("coe_routing") or chunk.get("routing")
        if isinstance(maybe_route, dict):
            route.update(maybe_route)
        events.extend(event for event in chunk.get("events", []) if isinstance(event, dict))
        if top_metadata.get("event_type"):
            events.append(dict(top_metadata))
    elif isinstance(chunk.get("coe_routing") or chunk.get("routing"), dict):
        route.update(chunk.get("coe_routing") or chunk.get("routing") or {})
    if not isinstance(events, list):
        events = []
    return "".join(content_parts), route, events


def normalize_coe_stream_output(text: str) -> str:
    """Restore CoE's prefix-stripped Python opening fence for shared extraction."""
    match = re.match(r"^(?:<think>.*?</think>\s*)?python\s*\r?\n", text, re.DOTALL)
    if match:
        return "```python\n" + text[match.end():]
    return text


def _cancel_coe_turn(base_url: str, request_id: str, headers: dict[str, str]) -> str:
    try:
        response = requests.post(
            _openai_v1_url(base_url, "cancel"),
            json={"request_id": request_id},
            headers=headers,
            timeout=20,
        )
        if response.status_code == 200:
            return str(response.json().get("mode", "ok"))
        return f"http_{response.status_code}"
    except Exception as exc:  # noqa: BLE001
        return f"cancel_failed:{type(exc).__name__}"


def stream_openai(
    base_url: str,
    payload: dict[str, Any],
    *,
    api_key: str | None,
    request_id: str | None,
    read_timeout_s: float,
    cancel_on_abort: bool,
) -> tuple[str, bool, dict[str, Any]]:
    headers = {"Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if request_id:
        headers["X-Request-Id"] = request_id

    text: list[str] = []
    routing: dict[str, Any] = {}
    loop_detected = False
    try:
        with requests.post(
            _openai_v1_url(base_url, "chat/completions"),
            json=payload,
            headers=headers,
            stream=True,
            timeout=(10, read_timeout_s),
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta, chunk_routing, events = parse_openai_chunk(chunk)
                if chunk_routing:
                    routing.update(chunk_routing)
                for event in events:
                    event_type = str(event.get("event_type", ""))
                    if event_type == "tier_start":
                        tier = event.get("tier", "")
                        model_id = event.get("model_id", "")
                        print(f"\n[{tier}] {model_id}", flush=True)
                if delta:
                    text.append(delta)
                    print(delta, end="", flush=True)
                    if _check_loop("".join(text)):
                        loop_detected = True
                        print("\n[LOOP-ABORT]", flush=True)
                        break
    except KeyboardInterrupt:
        if cancel_on_abort and request_id:
            print(f"\n[CANCEL] {_cancel_coe_turn(base_url, request_id, headers)}", flush=True)
        raise
    except Exception:
        if cancel_on_abort and request_id:
            print(f"\n[CANCEL] {_cancel_coe_turn(base_url, request_id, headers)}", flush=True)
        raise
    finally:
        print()

    if loop_detected and cancel_on_abort and request_id:
        print(f"[CANCEL] {_cancel_coe_turn(base_url, request_id, headers)}", flush=True)
    return "".join(text), loop_detected, routing

# ---------------------------------------------------------------------------
# Ollama streaming — mirrors qwen3.6/lcb_bench.py stream_ollama()
# ---------------------------------------------------------------------------
def stream_ollama(model: str, messages: list, temperature=0.6, max_tokens=8192,
                  seed: int = 42, base_url=OLLAMA_URL,
                  read_timeout_s: float = STREAM_TIMEOUT_S,
                  num_ctx: int = 32768, top_k: int = -1,
                  top_p: float = 1.0, think: bool = False,
                  prompt_profile: str = "canonical") -> tuple[str, bool]:
    """Stream from Ollama /api/chat using the selected native think state.
    Returns (text, loop_detected). text does NOT include ASSISTANT_PREFILL.
    """
    resp = requests.post(
        f"{base_url}/api/chat",
        json=build_ollama_payload(
            model, messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            num_ctx=num_ctx,
            top_k=top_k,
            top_p=top_p,
            think=think,
            prompt_profile=prompt_profile,
        ),
        stream=True, timeout=(10, read_timeout_s),
    )
    resp.raise_for_status()
    text = []
    loop_detected = False
    for line in resp.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        token = chunk.get("message", {}).get("content", "")
        if token:
            text.append(token)
            print(token, end="", flush=True)
            assembled = "".join(text)
            if _check_loop(assembled):
                loop_detected = True
                print("\n[LOOP-ABORT]", flush=True)
                break
        if chunk.get("done"):
            break
    print()
    return "".join(text), loop_detected


def call_ollama(model: str, user_content: str, temperature: float, max_tokens: int,
                seed: int, retry_on_loop: bool, base_url: str,
                read_timeout_s: float = STREAM_TIMEOUT_S,
                num_ctx: int = 32768, top_k: int = -1,
                top_p: float = 1.0, think: bool = False,
                prompt_profile: str = "canonical") -> tuple[str, bool]:
    """Primary call uses assistant prefill; fallback retries without prefill.

    Some model/template combinations may terminate immediately on a prefilled
    assistant turn. A no-prefill retry preserves benchmark continuity.
    """
    messages = build_direct_messages(
        user_content, think=think, prompt_profile=prompt_profile,
    )
    has_prefill = len(messages) > 2
    if has_prefill:
        print(ASSISTANT_PREFILL, end="", flush=True)
    raw_output, loop_detected = stream_ollama(
        model, messages,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        base_url=base_url,
        read_timeout_s=read_timeout_s,
        num_ctx=num_ctx,
        top_k=top_k,
        top_p=top_p,
        think=think,
        prompt_profile=prompt_profile,
    )
    # Protocol: on loop detection, retry once with a fresh seed.
    if loop_detected and retry_on_loop:
        retry_seed = seed + 1000
        print(f"[LOOP-RETRY] retrying once with seed={retry_seed}", flush=True)
        raw_retry, loop_retry = stream_ollama(
            model, messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=retry_seed,
            base_url=base_url,
            read_timeout_s=read_timeout_s,
            num_ctx=num_ctx,
            top_k=top_k,
            top_p=top_p,
            think=think,
            prompt_profile=prompt_profile,
        )
        if raw_retry.strip():
            full_retry = ASSISTANT_PREFILL + raw_retry if has_prefill else raw_retry
            return full_retry, (loop_detected or loop_retry)

    if raw_output.strip():
        full_output = ASSISTANT_PREFILL + raw_output if has_prefill else raw_output
        return full_output, loop_detected

    if not has_prefill:
        return raw_output, loop_detected
    messages_no_prefill = messages[:2]
    raw_retry, loop_retry = stream_ollama(
        model, messages_no_prefill,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        base_url=base_url,
        read_timeout_s=read_timeout_s,
        num_ctx=num_ctx,
        top_k=top_k,
        top_p=top_p,
        think=think,
        prompt_profile=prompt_profile,
    )
    return raw_retry, (loop_detected or loop_retry)

# ---------------------------------------------------------------------------
# Code extraction — LAST closed ```python block is the FINAL clean solution
# ---------------------------------------------------------------------------
def extract_final_code(raw: str) -> str:
    """Return FINAL code using the proven Gemma4 fence extraction pattern."""
    text = raw
    continued_open_fence = text.startswith(ASSISTANT_PREFILL)
    if continued_open_fence:
        text = text[len(ASSISTANT_PREFILL):]
    text = _THINK_RE.sub("", text)
    text = _OPEN_THINK_RE.sub("", text).strip()

    blocks = _FENCE_RE.findall(text)
    if blocks:
        return blocks[-1].strip()

    if continued_open_fence:
        lines = text.splitlines()
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    # Fallback for unclosed trailing fence.
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip().startswith("```")]
    if len(starts) >= 2:
        return "\n".join(lines[starts[-2] + 1: starts[-1]]).strip()
    if starts:
        return "\n".join(lines[starts[-1] + 1:]).strip()
    return text.strip()

# ---------------------------------------------------------------------------
# Grader — sandboxed subprocess running HumanEval check(candidate)
# ---------------------------------------------------------------------------
def grade(prompt: str, code: str, entry_point: str, test_code: str,
          timeout: int = 15) -> tuple[str, str, str]:
    """Run HumanEval test harness on extracted code in a subprocess.
    Returns (grade, note, code_used).
    grade: 'PASS' | 'FAIL' | 'NO_CODE'
    note:  short error string for FAIL, '' for PASS
    """
    code = code.rstrip() + "\n"
    if not code.strip():
        return "NO_CODE", "empty code", ""

    # Extract prompt's import lines so they're always available.
    prompt_imports = "\n".join(
        line for line in prompt.splitlines()
        if line.startswith("import ") or line.startswith("from ")
    )

    # If the FINAL block defines the entry_point function, use it as-is
    # (with prompt imports prepended).  Otherwise the FINAL is a body —
    # splice it into the prompt stub indented to function-body level.
    if re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", code, re.MULTILINE):
        full_code = (prompt_imports + "\n\n" + code) if prompt_imports else code
    else:
        body = code
        if body and not body.startswith("    "):
            body = "\n".join(
                ("    " + line) if line.strip() else line
                for line in body.splitlines()
            )
        full_code = prompt.rstrip() + "\n" + body

    # Alias injection: if model used a different function name, alias it
    # to the entry_point.  Handles the rare wrapper/rename case.
    model_fns = re.findall(r"^def\s+(\w+)\s*\(", full_code, re.MULTILINE)
    alias_block = ""
    if entry_point not in model_fns and model_fns:
        best = model_fns[-1]
        alias_block = f"\n{entry_point} = {best}\n"

    script = (
        full_code
        + "\n\n"
        + alias_block
        + test_code
        + "\n\n"
        + f"check({entry_point})\n"
    )

    fd, tmp = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(script)
        try:
            proc = subprocess.run(
                [sys.executable, tmp],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return "FAIL", "Timeout", code
        if proc.returncode == 0:
            return "PASS", "", code
        err = (proc.stderr or proc.stdout)[-400:].strip()
        return "FAIL", err, code
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------
def _fingerprint(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_existing_results(
    out_path: Path,
    expected_fingerprint: str,
) -> tuple[set[str], int, int, int]:
    completed: dict[str, str] = {}
    malformed = 0
    header: dict[str, Any] | None = None
    with out_path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if header is None:
                if rec.get("_type") != "run_metadata":
                    raise SystemExit(
                        f"Cannot safely resume legacy/unfingerprinted output {out_path}; "
                        "choose a new --out or pass --overwrite."
                    )
                header = rec
                if rec.get("fingerprint") != expected_fingerprint:
                    raise SystemExit(
                        f"Run configuration mismatch in {out_path}; use a new --out "
                        "or pass --overwrite."
                    )
                continue
            task_id = rec.get("task_id")
            grade_value = rec.get("grade")
            if isinstance(task_id, str) and isinstance(grade_value, str):
                completed[task_id] = grade_value

    if header is None:
        raise SystemExit(f"Output file has no valid run metadata header: {out_path}")
    pass_count = sum(1 for result in completed.values() if result == "PASS")
    fail_count = sum(1 for result in completed.values() if result != "PASS")
    return set(completed), pass_count, fail_count, malformed


def self_test() -> None:
    echoed_prefill = (
        ASSISTANT_PREFILL
        + "<think>\n</think>\n\n```python\nvalue = 42\n```"
    )
    continuation = ASSISTANT_PREFILL + "value = 42\n```"
    echoed_code = extract_final_code(echoed_prefill)
    continuation_code = extract_final_code(continuation)
    assert echoed_code == "value = 42", repr(echoed_code)
    assert continuation_code == "value = 42", repr(continuation_code)

    coe_stream = "python\ndef candidate(x):\n    return x + 1\n```"
    normalized_coe_stream = normalize_coe_stream_output(coe_stream)
    assert normalized_coe_stream.startswith("```python\n")
    assert extract_final_code(normalized_coe_stream) == "def candidate(x):\n    return x + 1"

    direct_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_content("def candidate(x):\n    pass")},
        {"role": "assistant", "content": ASSISTANT_PREFILL},
    ]
    assert direct_messages[-1]["content"] == ASSISTANT_PREFILL
    assert "DRAFT" in direct_messages[1]["content"]
    think_on_messages = build_direct_messages("def candidate(x):\n    pass", think=True)
    assert len(think_on_messages) == 2
    assert think_on_messages[0]["content"] == THINK_ON_SYSTEM_PROMPT
    assert "native thinking" in THINK_ON_SYSTEM_PROMPT
    assert "ONE clean fenced code block" in THINK_ON_SYSTEM_PROMPT
    think_on_payload = build_ollama_payload(
        "ornith", think_on_messages, temperature=0.6, max_tokens=24576,
        seed=42, num_ctx=32768, top_k=-1, top_p=1.0, think=True,
    )
    assert think_on_payload["think"] is True
    assert len(think_on_payload["messages"]) == 2
    assert think_on_payload["options"]["seed"] == 42
    assert think_on_payload["options"]["repeat_last_n"] == 1024
    think_off_payload = build_ollama_payload(
        "qwen", direct_messages, temperature=0.6, max_tokens=8192,
        seed=42, num_ctx=32768, top_k=-1, top_p=1.0, think=False,
    )
    assert "repeat_last_n" not in think_off_payload["options"]
    lean_prompt = "def candidate(x):\n    pass"
    for lean_think in (False, True):
        lean_messages = build_direct_messages(
            lean_prompt, think=lean_think, prompt_profile="lean",
        )
        assert len(lean_messages) == 2
        assert lean_messages[0]["content"] == LEAN_SYSTEM_PROMPT
        assert lean_messages[1]["content"] == lean_prompt
        assert all(message["role"] != "assistant" for message in lean_messages)
        lean_payload = build_ollama_payload(
            "parent", lean_messages, temperature=0.6, max_tokens=24576,
            seed=42, num_ctx=32768, top_k=-1, top_p=1.0, think=lean_think,
            prompt_profile="lean",
        )
        assert lean_payload["think"] is lean_think
        assert lean_payload["options"]["repeat_last_n"] == 1024

    coe_payload, profile = build_openai_payload(
        "def candidate(x):\n    pass",
        openai_model=None,
        coe_domain=COE_DOMAIN,
        coe_task_type=COE_TASK_TYPE,
        think=False,
        temperature=0.6,
        max_tokens=8192,
        seed=42,
    )
    assert profile == "coe:domain_python/new_code_generation:think=off"
    assert coe_payload["messages"] == [
        {"role": "user", "content": "def candidate(x):\n    pass"}
    ]
    assert coe_payload["routing_override"] == {
        "domain": "domain_python",
        "task_type": "new_code_generation",
    }
    assert coe_payload["metadata"] == {
        "think": False,
        "clear_session": True,
        "no_inject": True,
        "seed": 42,
    }
    assert "model" not in coe_payload and "assistant" not in str(coe_payload["messages"])

    test_prompt = "def candidate(x):\n    pass"
    test_code = "def check(candidate):\n    assert candidate(2) == 3\n"
    good = grade(test_prompt, "return x + 1", "candidate", test_code)
    bad = grade(test_prompt, "return x", "candidate", test_code)
    assert good[0] == "PASS", good
    assert bad[0] == "FAIL" and "AssertionError" in bad[1], bad

    delta, route, events = parse_openai_chunk({
        "model": "qwen",
        "choices": [{"delta": {
            "content": "token",
            "metadata": {
                "coe_routing": {"domain": "domain_python"},
                "tier_start": "T0",
                "model_id": "qwen",
            },
        }}],
    })
    assert delta == "token" and route["domain"] == "domain_python"
    assert events[0]["tier"] == "T0"
    print("Self-test passed: request profiles, SSE parsing, and HumanEval grading.")


def run_eval(
    *,
    backend: str,
    model: str,
    openai_base_url: str,
    openai_model: str | None,
    api_key_env: str,
    coe_domain: str | None,
    coe_task_type: str | None,
    think: bool,
    expected_coe_model: str,
    data_path: Path,
    n_tasks: int,
    only_ids: list[str] | None,
    out_path: Path,
    stride: int,
    temperature: float,
    max_tokens: int,
    seed: int,
    prompt_profile: str,
    num_ctx: int,
    top_k: int,
    top_p: float,
    read_timeout_s: float,
    retry_on_loop: bool,
    resume: bool,
    overwrite: bool,
) -> None:
    if not data_path.is_file():
        raise SystemExit(f"HumanEval dataset not found: {data_path}")
    if not 1 <= n_tasks <= 164:
        raise SystemExit("--n must be between 1 and 164")
    if stride < 1:
        raise SystemExit("--stride must be at least 1")
    if backend == "ollama" and (coe_domain or coe_task_type):
        raise SystemExit("CoE routing overrides require --backend openai.")
    if backend == "openai" and bool(coe_domain) != bool(coe_task_type):
        raise SystemExit("CoE mode requires both --coe-domain and --coe-task-type")
    if backend == "openai" and not coe_domain and think:
        raise SystemExit("--think on generic OpenAI mode is unsupported; use Ollama or CoE mode.")
    if backend != "ollama" and prompt_profile != "canonical":
        raise SystemExit("--prompt-profile lean is available only for direct Ollama runs.")

    with data_path.open(encoding="utf-8") as fh:
        all_tasks = [json.loads(line) for line in fh if line.strip()]
    if len(all_tasks) < n_tasks:
        raise SystemExit(f"Dataset has {len(all_tasks)} tasks, fewer than requested --n {n_tasks}")
    tasks = all_tasks[:n_tasks]
    if stride > 1:
        tasks = tasks[::stride]
    if only_ids:
        requested_ids = set(only_ids)
        known_ids = {task["task_id"] for task in all_tasks}
        unknown_ids = requested_ids - known_ids
        if unknown_ids:
            raise SystemExit(f"Unknown HumanEval task IDs: {', '.join(sorted(unknown_ids))}")
        tasks = [task for task in tasks if task["task_id"] in requested_ids]

    api_key = os.environ.get(api_key_env) if api_key_env else None
    coe_mode = backend == "openai" and bool(coe_domain)
    if coe_mode:
        profile = f"coe:{coe_domain}/{coe_task_type}:think={'on' if think else 'off'}"
    elif backend == "ollama" and prompt_profile == "lean":
        profile = f"ollama:lean-single-code-no-prefill:think={'on' if think else 'off'}"
    elif backend == "ollama" and think:
        profile = "ollama:python-new-code-thinkon-single-final-no-prefill"
    elif backend == "ollama":
        profile = "ollama:canonical-two-block-open-python-fence"
    else:
        profile = "openai:canonical-prompt:no-prefill"
    run_config: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "backend": backend,
        "prompt_profile": profile,
        "direct_model": model if backend == "ollama" else None,
        "openai_model": None if coe_mode or backend == "ollama" else openai_model,
        "endpoint": OLLAMA_URL if backend == "ollama" else openai_base_url.rstrip("/"),
        "api_key_env": api_key_env if backend == "openai" else None,
        "api_key_present": bool(api_key) if backend == "openai" else False,
        "coe_domain": coe_domain if coe_mode else None,
        "coe_task_type": coe_task_type if coe_mode else None,
        "think": think if (coe_mode or backend == "ollama") else False,
        "expected_coe_model": expected_coe_model if coe_mode else None,
        "data_path": str(data_path.resolve()),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "n_tasks": n_tasks,
        "stride": stride,
        "only_ids": sorted(only_ids) if only_ids else None,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "num_ctx": num_ctx if backend == "ollama" else None,
        "top_k": top_k if backend == "ollama" else None,
        "top_p": top_p if backend == "ollama" else None,
        "read_timeout_s": read_timeout_s,
        "retry_on_loop": retry_on_loop,
        "coe_clear_session": coe_mode,
        "coe_no_inject": coe_mode,
    }
    fingerprint = _fingerprint(run_config)
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nHumanEval | profile={profile} | tasks={len(tasks)} | stride={stride}")
    print(f"Output -> {out_path}")
    print(f"Run fingerprint -> {fingerprint}\n")

    completed_ids: set[str] = set()
    pass_count = 0
    fail_count = 0
    malformed = 0
    if overwrite and out_path.exists():
        out_path.unlink()
    if out_path.exists() and resume and not overwrite:
        completed_ids, pass_count, fail_count, malformed = _load_existing_results(
            out_path, fingerprint,
        )
        if malformed:
            print(f"[RESUME] ignored {malformed} malformed JSONL lines")
        print(f"[RESUME] {len(completed_ids)} tasks already completed")
    elif out_path.exists() and not resume and not overwrite:
        raise SystemExit(f"Output exists: {out_path}. Use --overwrite or choose another --out.")

    out_mode = "a" if out_path.exists() and resume and not overwrite else "w"
    with out_path.open(out_mode, encoding="utf-8") as out_fh:
        if out_mode == "w":
            header = {
                "_type": "run_metadata",
                "schema": RUN_SCHEMA,
                "fingerprint": fingerprint,
                "config": run_config,
            }
            out_fh.write(json.dumps(header, ensure_ascii=False) + "\n")
            out_fh.flush()

        for index, task in enumerate(tasks, 1):
            task_id = task["task_id"]
            prompt = task["prompt"]
            entry_point = task["entry_point"]
            test_code = task["test"]
            if task_id in completed_ids:
                continue

            print(f"\n[{index:3d}/{len(tasks)}] {task_id} | entry: {entry_point}")
            request_id = str(uuid.uuid4()) if coe_mode else None
            started = time.time()
            routing: dict[str, Any] = {}
            loop_detected = False
            full_output = ""
            try:
                if backend == "ollama":
                    full_output, loop_detected = call_ollama(
                        model,
                        prompt if think or prompt_profile == "lean" else build_user_content(prompt),
                        temperature=temperature,
                        max_tokens=max_tokens,
                        seed=seed,
                        retry_on_loop=retry_on_loop,
                        base_url=OLLAMA_URL,
                        read_timeout_s=read_timeout_s,
                        num_ctx=num_ctx,
                        top_k=top_k,
                        top_p=top_p,
                        think=think,
                        prompt_profile=prompt_profile,
                    )
                else:
                    payload, _ = build_openai_payload(
                        prompt,
                        openai_model=openai_model,
                        coe_domain=coe_domain,
                        coe_task_type=coe_task_type,
                        think=think,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        seed=seed,
                    )
                    full_output, loop_detected, routing = stream_openai(
                        openai_base_url,
                        payload,
                        api_key=api_key,
                        request_id=request_id,
                        read_timeout_s=read_timeout_s,
                        cancel_on_abort=coe_mode,
                    )
                    if coe_mode:
                        full_output = normalize_coe_stream_output(full_output)
                    if loop_detected and retry_on_loop:
                        print("[LOOP-RETRY] issuing one fresh request", flush=True)
                        if not coe_mode:
                            payload["seed"] = seed + 1000
                        retry_id = str(uuid.uuid4()) if coe_mode else None
                        retry_output, retry_loop, retry_routing = stream_openai(
                            openai_base_url,
                            payload,
                            api_key=api_key,
                            request_id=retry_id,
                            read_timeout_s=read_timeout_s,
                            cancel_on_abort=coe_mode,
                        )
                        loop_detected = loop_detected or retry_loop
                        if coe_mode:
                            retry_output = normalize_coe_stream_output(retry_output)
                        if retry_output.strip():
                            full_output = retry_output
                            routing = retry_routing or routing
                            request_id = retry_id
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"[ERROR] {backend} call failed: {type(exc).__name__}: {exc}")
                full_output = ""
                note = f"api_error: {type(exc).__name__}: {exc}"
                grade_result = "FAIL"
                code_used = ""
            else:
                code = extract_final_code(full_output)
                grade_result, note, code_used = grade(prompt, code, entry_point, test_code)

            elapsed = time.time() - started
            if grade_result == "PASS":
                pass_count += 1
                status = "PASS"
            else:
                fail_count += 1
                status = f"FAIL | {note[:140]}"
            if loop_detected:
                status += " [LOOP-ABORTED]"
            if coe_mode:
                actual_primary = routing.get("primary_model")
                if actual_primary:
                    matches_expected = str(actual_primary).casefold() == expected_coe_model.casefold()
                    routing["expected_primary_model"] = expected_coe_model
                    routing["primary_model_match"] = matches_expected
                    if not matches_expected:
                        status += f" [ROUTING-MISMATCH primary={actual_primary}]"
                else:
                    routing["expected_primary_model"] = expected_coe_model
                    routing["primary_model_match"] = None
            print(
                f"raw_chars={len(full_output):5d}  code_chars={len(code_used):5d}  "
                f"{elapsed:5.1f}s  {status}"
            )

            record = {
                "task_id": task_id,
                "entry_point": entry_point,
                "grade": grade_result,
                "loop_detected": loop_detected,
                "note": note[:300] if note else "",
                "elapsed_s": round(elapsed, 2),
                "raw_chars": len(full_output),
                "code_chars": len(code_used),
                "prompt": prompt[:500],
                "response": full_output[:4000],
                "code_used": code_used[:1500] if code_used else "",
                "backend": backend,
                "request_id": request_id,
                "routing": routing,
                "run_fingerprint": fingerprint,
            }
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_fh.flush()

    done = pass_count + fail_count
    percentage = pass_count / done * 100 if done else 0
    print(f"\n{'='*72}")
    print(f"DONE  {pass_count}/{done} PASS ({percentage:.1f}%)")
    print(f"Results -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HumanEval canonical runner: direct Ollama or OpenAI-compatible API"
    )
    parser.add_argument("--self-test", action="store_true", help="run offline request/grader checks and exit")
    parser.add_argument("--backend", choices=("ollama", "openai"), default="ollama")
    parser.add_argument("--model", default=DEFAULT_DIRECT_MODEL,
                        help="direct Ollama model tag (think-off Python Qwen by default)")
    parser.add_argument("--openai-base-url", default=OPENAI_BASE_URL,
                        help="OpenAI-compatible server base URL, with or without /v1")
    parser.add_argument("--openai-model", default=None,
                        help="model name for generic OpenAI mode; omitted for CoE mode")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY",
                        help="environment variable holding API key (never written to results)")
    parser.add_argument("--coe-domain", default=None,
                        help="pin CoE routing domain (use domain_python for HumanEval)")
    parser.add_argument("--coe-task-type", default=None,
                        help="pin CoE task type (use new_code_generation for HumanEval)")
    parser.add_argument("--think", choices=("off", "on"), default="off",
                        help="native Ollama think state or CoE think branch; default off")
    parser.add_argument("--prompt-profile", choices=("canonical", "lean"), default="canonical",
                        help="direct Ollama prompt profile; lean uses raw task, one solution block, and no prefill")
    parser.add_argument("--expected-coe-model", default=DEFAULT_DIRECT_MODEL,
                        help="expected CoE primary model ID, recorded and checked when routing telemetry is present")
    parser.add_argument("--read-timeout", type=float, default=STREAM_TIMEOUT_S,
                        help="stream idle timeout in seconds")
    parser.add_argument("--data", type=Path, default=DATA_FILE)
    parser.add_argument("--n", type=int, default=164,
                        help="number of HumanEval tasks (maximum 164)")
    parser.add_argument("--stride", type=int, default=1,
                        help="take every Nth task (default 1 = consecutive)")
    parser.add_argument("--only", default=None,
                        help="comma-separated HumanEval task IDs; useful for failed-only retry rounds")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="direct Ollama/generic OpenAI token budget; CoE owns its harness budget")
    parser.add_argument("--num-ctx", type=int, default=32768,
                        help="direct Ollama context size (aligned with CoE default)")
    parser.add_argument("--top-k", type=int, default=-1,
                        help="direct Ollama top_k (CoE default -1)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="direct Ollama top_p (CoE default 1.0)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-loop-retry", action="store_true",
                        help="disable the single retry when a repetition loop is detected")
    parser.add_argument("--no-resume", action="store_true",
                        help="refuse to resume an existing output file")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace the selected output file")
    parser.add_argument("--out", type=Path, default=Path("humaneval_flexible_v1.jsonl"))
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    coe_mode = bool(args.coe_domain or args.coe_task_type)
    if args.backend == "ollama" and coe_mode:
        parser.error("--coe-domain/--coe-task-type require --backend openai")
    if args.backend == "openai" and not coe_mode and not args.openai_model:
        parser.error("generic --backend openai requires --openai-model")
    if args.backend == "openai" and coe_mode and (args.coe_domain, args.coe_task_type) != (COE_DOMAIN, COE_TASK_TYPE):
        parser.error("HumanEval CoE runs must pin domain_python/new_code_generation")

    run_eval(
        backend=args.backend,
        model=args.model,
        openai_base_url=args.openai_base_url,
        openai_model=args.openai_model,
        api_key_env=args.api_key_env,
        coe_domain=args.coe_domain,
        coe_task_type=args.coe_task_type,
        think=args.think == "on",
        expected_coe_model=args.expected_coe_model,
        data_path=args.data,
        n_tasks=args.n,
        only_ids=[item.strip() for item in args.only.split(",") if item.strip()] if args.only else None,
        out_path=args.out,
        stride=args.stride,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        prompt_profile=args.prompt_profile,
        num_ctx=args.num_ctx,
        top_k=args.top_k,
        top_p=args.top_p,
        read_timeout_s=args.read_timeout,
        retry_on_loop=not args.no_loop_retry,
        resume=not args.no_resume,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
