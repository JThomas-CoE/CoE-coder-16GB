"""FLM (FastFlowLM) server lifecycle: probe, start, wait-ready.

The clean slice treats FLM as the primary router backend when
``backends.flm`` is configured.  If the configured URL is not
responding at boot, the runner calls ``ensure_flm`` to:

  1. Resolve the ``flm`` binary (config override first, then
     ``shutil.which('flm')`` so a PATH install is picked up).
  2. Launch ``flm serve <router_model>`` as a detached child
     process with stdout/stderr captured to a log file under
     ``clean_slice_runs/flm-<timestamp>.log``.
  3. Poll the configured URL until ``/v1/models`` responds
     (or a hard timeout elapses).
  4. Return a small status dict so the caller can fall back to
     Ollama if startup fails (warn-and-continue policy).

If FLM is already up, the helper is a no-op and reports
``started_by_us=False``.  If startup fails for any reason, the
helper returns ``up=False`` with an error string; the caller
logs a warning and continues with the Ollama router fallback.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx


def resolve_flm_binary(configured_path: str | None) -> str | None:
    """Find the ``flm`` executable.

    Resolution order:
      1. ``configured_path`` if it points to an existing file.
      2. ``shutil.which('flm')`` — search the current PATH.
    Returns the absolute path or ``None`` if FLM is not
    installed in a discoverable location.
    """
    if configured_path:
        p = Path(configured_path)
        if p.is_file():
            return str(p.resolve())
    return shutil.which("flm")


def is_flm_up(url: str, timeout: float = 2.0) -> bool:
    """Probe the FLM OpenAI-compatible ``/v1/models`` endpoint.

    Returns ``True`` iff a 200 response is received within
    ``timeout`` seconds.  Any httpx error counts as down.
    """
    base = str(url).rstrip("/").removesuffix("/v1")
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(f"{base}/v1/models")
            return response.status_code == 200
    except httpx.HTTPError:
        return False


def start_flm(
    *,
    binary: str,
    model_id: str,
    log_path: Path,
    extra_args: list[str] | None = None,
) -> subprocess.Popen | None:
    """Launch ``flm serve <model>`` as a detached child process.

    The child runs detached so closing the parent terminal will
    not kill it.  ``stdout`` and ``stderr`` are merged into
    ``log_path`` so all FLM output is captured for debugging.
    Returns the ``Popen`` handle on success, ``None`` on failure.
    """
    cmd = [binary, "serve", model_id]
    if extra_args:
        cmd.extend(extra_args)
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
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=flags,
        )
        return proc
    except OSError:
        try:
            log_file.close()
        except OSError:
            pass
        return None


def wait_for_flm(url: str, *, timeout_s: float = 60.0, poll_s: float = 1.0) -> bool:
    """Poll the FLM URL until it responds or the timeout elapses."""
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if is_flm_up(url, timeout=1.0):
            return True
        time.sleep(poll_s)
    return False


def ensure_flm(
    *,
    flm_url: str,
    router_model_id: str,
    binary_override: str | None = None,
    log_dir: str | Path | None = None,
    startup_timeout_s: float = 60.0,
    extra_serve_args: list[str] | None = None,
) -> dict[str, Any]:
    """Top-level helper: ensure FLM is up, starting it if not.

    Returns a dict with keys:
      ``up`` (bool) — whether FLM is reachable after this call.
      ``started_by_us`` (bool) — whether we spawned a process.
      ``pid`` (int or None) — the child PID if we started one.
      ``log_path`` (str or None) — log file we redirected to.
      ``binary`` (str or None) — the resolved binary path.
      ``error`` (str or None) — human-readable failure reason.

    When ``log_dir`` is ``None`` (the default), the FLM
    child process log is written to ``<package>/logs/``
    so the new package's output is self-contained
    regardless of CWD.  Pass an explicit path to override.

    Policy: warn-and-continue.  If FLM cannot be started, the
    runner falls back to Ollama for routing and the user is
    notified via a single boot log line.
    """
    # Already up: no-op.
    if is_flm_up(flm_url):
        return {
            "up": True,
            "started_by_us": False,
            "pid": None,
            "log_path": None,
            "binary": None,
            "error": None,
        }
    # Try to find the binary.
    binary = resolve_flm_binary(binary_override)
    if not binary:
        return {
            "up": False,
            "started_by_us": False,
            "pid": None,
            "log_path": None,
            "binary": None,
            "error": "flm binary not found on PATH (and no binary_path override)",
        }
    # Spawn flm serve.
    if log_dir is None:
        # Default: package-relative, so the new package's
        # FLM child process log is self-contained next to
        # the package source rather than in the workspace
        # root or CWD.
        log_dir = Path(__file__).resolve().parent / "logs"
    log_path = Path(log_dir) / f"flm-{int(time.time())}.log"
    proc = start_flm(
        binary=binary,
        model_id=router_model_id,
        log_path=log_path,
        extra_args=extra_serve_args,
    )
    if proc is None:
        return {
            "up": False,
            "started_by_us": False,
            "pid": None,
            "log_path": str(log_path),
            "binary": binary,
            "error": "subprocess.Popen failed for flm serve",
        }
    # Wait for ready.
    if wait_for_flm(flm_url, timeout_s=startup_timeout_s):
        return {
            "up": True,
            "started_by_us": True,
            "pid": int(proc.pid),
            "log_path": str(log_path),
            "binary": binary,
            "error": None,
        }
    # Startup timeout: kill the child so we don't leak it.
    try:
        proc.terminate()
    except Exception:
        pass
    return {
        "up": False,
        "started_by_us": True,
        "pid": int(proc.pid),
        "log_path": str(log_path),
        "binary": binary,
        "error": f"FLM did not respond within {startup_timeout_s}s after spawn",
    }


def stop_flm(pid: int, *, timeout_s: float = 5.0) -> bool:
    """Terminate a FLM process we previously started.

    Only call this for processes returned by ``ensure_flm``
    with ``started_by_us=True``.  The runner respects the
    ``stop_on_exit`` config flag; this helper is provided for
    callers that do want the shutdown behavior.
    """
    try:
        import psutil  # type: ignore
    except ImportError:
        psutil = None  # type: ignore
    if psutil is not None:
        try:
            p = psutil.Process(pid)
            p.terminate()
            try:
                p.wait(timeout=timeout_s)
            except psutil.TimeoutExpired:
                p.kill()
            return True
        except psutil.NoSuchProcess:
            return True
        except Exception:
            return False
    # Fallback without psutil: use Windows taskkill.
    try:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            timeout=timeout_s,
        )
        return True
    except Exception:
        return False
