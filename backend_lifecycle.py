"""Local inference backend lifecycle: probe, start if absent, wait-ready.

**Provider-agnostic.**  CoE-coder talks to its local inference backend over an
OpenAI-compatible HTTP surface, so nothing here names Ollama, llama.cpp or
LM Studio.  The backend is described entirely by its ``backends.<name>`` config
block; this module generalises the four primitives that
``flm_lifecycle.ensure_flm`` established for FLM:

  1. Resolve the launcher binary (config ``binary_path`` first, then a
     PATH search over ``binary_names``).
  2. Spawn the configured command, detached, logs merged to a file.
  3. Poll the configured URL until it responds (or a hard timeout elapses).
  4. Return a status dict so the caller can decide its own policy.

**Policy is the CALLER's, and it differs per backend — deliberately:**

* FLM is the optional NPU *router* accelerator.  It has a fallback (the same
  model on Ollama), so ``server.py`` treats ``up=False`` as warn-and-continue.
* The configured inference backend has **no fallback**.  ``server.py`` treats
  ``up=False`` as FATAL: it prints a loud remedy message and aborts startup,
  rather than serving an instance whose every request would fail.

**There is no ``stop_*`` here, and that is by design, not an omission.**
CoE-coder owns the *start* decision (a missing backend is a broken install) but
never the *stop* decision.  On the shipped reference configuration
``backends.cloud.url`` is the SAME endpoint as ``backends.ollama.url`` -- the
local daemon proxies the cloud provider -- so the daemon serves both local
inference AND the cloud tier.  Stopping it would silently take out cloud
escalation as a side effect.  ``stop_on_exit`` is therefore accepted in config,
documented, and **ignored** here; the key exists so the intent is explicit
rather than implicit.

This is the structural form of the standing rule: *never kill the daemon that
serves the cloud path -- start it if it is missing, leave it alone otherwise.*
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
# OpenAI-compatible readiness probe.  Every backend this package supports
# (Ollama, llama.cpp's server, LM Studio, FLM) answers GET <base>/v1/models
# with 200 once it is accepting requests, so one probe covers them all.
DEFAULT_STARTUP_TIMEOUT_S = 60.0
DEFAULT_POLL_S = 1.0


def resolve_backend_binary(
    backend_name: str,
    configured_path: str | None = None,
    binary_names: list[str] | None = None,
) -> str | None:
    """Find the launcher executable for a backend.

    Resolution order:
      1. ``configured_path`` if it points to an existing file (config
         ``binary_path`` -- the explicit override).
      2. ``shutil.which()`` over ``binary_names``, defaulting to
         ``[backend_name]`` so ``backends.ollama`` finds ``ollama``.

    Returns the absolute path, or ``None`` if the backend is not installed
    in a discoverable location.
    """
    if configured_path:
        p = Path(configured_path)
        if p.is_file():
            return str(p.resolve())
    for name in (binary_names or [backend_name]):
        found = shutil.which(name)
        if found:
            return found
    return None


def is_backend_up(url: str, timeout: float = 2.0) -> bool:
    """Probe the backend's OpenAI-compatible ``/v1/models`` endpoint.

    ``True`` iff a 200 is received within ``timeout`` seconds.  Any httpx
    error counts as down.  The ``/v1`` suffix is stripped before appending
    the path, so both ``http://host:port`` and ``http://host:port/v1``
    forms work.
    """
    base = str(url).rstrip("/").removesuffix("/v1")
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(f"{base}/v1/models")
            return response.status_code == 200
    except httpx.HTTPError:
        return False


def start_backend(
    *,
    binary: str,
    args: list[str] | None,
    log_path: Path,
) -> subprocess.Popen | None:
    """Launch the backend command as a detached child process.

    ``args`` is the full argument list AFTER the binary (e.g.
    ``["serve"]``).  ``stdout``/``stderr`` are merged into ``log_path`` so
    all backend output is captured for diagnosis.

    Detached + new process group, so closing the parent terminal does not
    kill the daemon -- the backend is a long-lived service, not a child of
    CoE-coder.  Returns the ``Popen`` handle, or ``None`` on failure.
    """
    cmd = [binary] + list(args or [])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        log_file = log_path.open("ab")
    except OSError:
        return None
    flags = 0
    if hasattr(subprocess, "DETACHED_PROCESS"):
        flags |= getattr(subprocess, "DETACHED_PROCESS")
    if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP")
    try:
        return subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=flags,
        )
    except OSError:
        try:
            log_file.close()
        except OSError:
            pass
        return None


def wait_for_backend(
    url: str,
    *,
    timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
    poll_s: float = DEFAULT_POLL_S,
) -> bool:
    """Poll the backend URL until it responds or the timeout elapses."""
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if is_backend_up(url, timeout=1.0):
            return True
        time.sleep(poll_s)
    return False


def ensure_backend(
    *,
    backend_name: str,
    cfg: dict[str, Any] | None,
    log_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Ensure a configured local inference backend is up, starting it if not.

    ``cfg`` is that backend's ``backends.<name>`` block.  Recognised keys:

      ``url``               (required) OpenAI-compatible base URL.
      ``auto_start``        (default ``True``) spawn it if it is not up.
      ``binary_path``       (default ``None``) explicit launcher; else PATH.
      ``binary_names``      (default ``[backend_name]``) PATH search names.
      ``serve_args``        (default ``["serve"]``) args after the binary.
      ``startup_timeout_s`` (default ``60``) readiness deadline after spawn.
      ``log_dir``           (default ``<package>/logs``) child log location.
      ``stop_on_exit``      ACCEPTED AND IGNORED -- see the module docstring.

    Returns a status dict with keys ``up``, ``started_by_us``, ``pid``,
    ``log_path``, ``binary``, ``error``, ``backend``.  Never raises: the
    CALLER decides whether an unreachable backend is fatal (the inference
    backend) or a warn-and-continue (FLM, which has a fallback).
    """
    cfg = cfg or {}
    result: dict[str, Any] = {
        "backend": backend_name,
        "up": False,
        "started_by_us": False,
        "pid": None,
        "log_path": None,
        "binary": None,
        "error": None,
    }

    url = str(cfg.get("url") or "").strip()
    if not url:
        result["error"] = f"backends.{backend_name}.url is not configured"
        return result

    # Already up: the common case, and a strict no-op.
    if is_backend_up(url):
        result["up"] = True
        return result

    if not bool(cfg.get("auto_start", True)):
        result["error"] = (
            f"backend '{backend_name}' is not responding at {url} and "
            f"backends.{backend_name}.auto_start is false"
        )
        return result

    binary = resolve_backend_binary(
        backend_name,
        cfg.get("binary_path"),
        cfg.get("binary_names"),
    )
    if not binary:
        result["error"] = (
            f"launcher binary for '{backend_name}' not found on PATH "
            f"(and no backends.{backend_name}.binary_path override)"
        )
        return result
    result["binary"] = binary

    log_root = cfg.get("log_dir") or log_dir
    if log_root is None:
        # Package-relative by default, so a relocated package keeps its
        # logs self-contained regardless of CWD.
        log_root = Path(__file__).resolve().parent / "logs"
    log_path = Path(log_root) / f"{backend_name}-{int(time.time())}.log"

    proc = start_backend(
        binary=binary,
        args=cfg.get("serve_args") or ["serve"],
        log_path=log_path,
    )
    if proc is None:
        result["log_path"] = str(log_path)
        result["error"] = f"subprocess.Popen failed for {binary}"
        return result

    result["started_by_us"] = True
    result["pid"] = int(proc.pid)
    result["log_path"] = str(log_path)

    timeout_s = float(cfg.get("startup_timeout_s", DEFAULT_STARTUP_TIMEOUT_S))
    if wait_for_backend(url, timeout_s=timeout_s):
        result["up"] = True
        return result

    # Startup timeout.  NOTE: we do NOT terminate the child.  A backend that
    # is slow to bind may still come up, and terminating it would be exactly
    # the stop-that-breaks-cloud this module refuses to do.  Report it.
    result["error"] = (
        f"backend '{backend_name}' did not respond at {url} within "
        f"{timeout_s}s after spawn (pid {proc.pid}; log {log_path})"
    )
    return result
