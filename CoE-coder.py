#!/usr/bin/env python3
"""CoE-coder — launcher + interactive CLI for the CoE-coder-16GB black box.

Launch:
  CoE-coder                 start/reuse the server (OpenAI surface), block
  CoE-coder -cli            start/reuse server + run interactive CLI
  CoE-coder -port 8003      bind the server on a specific port (default 8001)
  CoE-coder -h              help

The SERVER is the shared durable resource: external agentic wrappers (and the
CLI, with -cli) connect to its OpenAI-compatible surface.  The port is set when
uvicorn opens the server (-port); there is NO runtime /url command.  Server
lifecycle is manual and simple: /bye exits the CLI only; Ctrl+C after /bye
stops the server.  A healthy running server is reused by the next launch
(never spawn a second uvicorn on the same port).

CANCELLING A TURN (2026-09-07): Ctrl+C WHILE A TURN IS RUNNING cancels only
that turn — the server keeps running and the system returns to the prompt
with state exactly as before the turn (nothing committed, session memory and
artifact store untouched).  The server runs in its own process group, so
console Ctrl+C reaches only the CLI, which cancels the turn via
POST /v1/cancel.  See CoE_coder_spec.md "Cancelling a turn (CLI)".
"""
import argparse
import base64
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import requests

DEFAULT_PORT = 8001

# The tier currently in flight, tracked across metadata chunks so the phase
# (`[T0 prefill]`) and verdict (`[tier] T0 -> passed`) lines can name it.
# `_print_meta` is stateless per chunk, so this has to live at module level.
# Set by `tier_start`; read by the `status` and `tier_complete` branches.
_CURRENT_TIER: str = ""
# Read timeout for one CLI request (2026-09-09, ladder-aware).  The old
# flat 600s fired mid-turn while the SERVER-side ladder was legitimately
# still working (live: T2_R1_retry ran 662s, T3 started at 1419s — the
# CLI was already dead at 600s so the user never saw the retry/T3 tier
# lines).  The server owns intra-ladder timeouts (210s SSE idle
# watchdog, 600s cloud silence/total, per-round first-token/slow-decode
# timers); the CLI only needs to outlive the longest LEGITIMATE turn.
# 45 min covers: cold load (2 min) + T0 R1/R2 (+retries) + T2 R1/R2
# (+retries) on a slow specialist + cloud T3 (10 min budget) with
# headroom.  A truly wedged stream still surfaces: the server's own
# watchdogs close the connection, which raises here immediately.
REQUEST_TIMEOUT_S = 2700
current_pending_artifact = None

# --- CLI commands (handled locally, NOT sent to server) ---
CLI_COMMANDS = {
    "/?": "show help",
    "/help": "show help",
    "/bye": "quit",
    "/exit": "quit",
    "/streamon": "enable streaming (default)",
    "/streamoff": "disable streaming",
    "/thinkshow": "show thinking tokens in streamed output",
    "/thinkhide": "hide thinking tokens (default)",
    "/attach": "attach a file for the next prompt",
    "/clear": "clear session",
}

# --- Server-side slash commands (routing/think/evaluate) ---
# Task-type override accepts /task-<name> as the canonical display form;
# /task_type-<name> and /tasktype-<name> remain accepted for compatibility.
# The internal field name stays task_type everywhere (display change only).
SERVER_SLASH = re.compile(r"^/(domain|task_type|tasktype|task)-(\S+)\s*")
THINK_SLASH = re.compile(r"^/(thinkon|thinkoff)\s*")
EVALUATE_SLASH = re.compile(r"^/evaluate\s*")
CLOUD_SLASH = re.compile(r"^/cloud\s*")

DOMAIN_ALIASES = {
    "python": "domain_python", "web": "domain_web", "math": "domain_math",
    "compiled": "domain_compiled", "physics": "domain_physics", "other": "domain_other",
}
TASK_TYPE_ALIASES = {
    "newcode": "new_code_generation", "new_code": "new_code_generation",
    "genuse": "general_use", "general_use": "general_use",
    "modify": "modify_artifact", "modify_artifact": "modify_artifact",
    "bugfix": "bug_fix", "bug_fix": "bug_fix",
    "evaluate": "evaluate_artifact", "evaluate_artifact": "evaluate_artifact",
    "userdefined": "user_defined", "user_defined": "user_defined",
}


def help_text() -> str:
    lines = [
        "CoE-coder CLI — commands:",
        "",
        "  /domain-<name>      routing: python, web, math, compiled, physics, other",
        "  /task-<name>        routing: newcode, genuse, modify, bugfix, evaluate, userdefined",
        "                      (legacy forms /task_type-<name> and /tasktype-<name> still work)",
        "  /thinkon            think-ON ordering: in matrix cells that declare",
        "                      think_on/think_off branches (today: domain_python",
        "                      new_code_generation) the cell's think-enabled model",
        "                      (ornith) becomes primary. Pruned models are NOT made",
        "                      to think by this command — see spec Step 3.",
        "  /thinkoff           think-OFF ordering (default): think-enabled models go",
        "                      secondary; the think_off branch ordering applies",
        "  /cloud              send this request directly to T3 cloud",
        "  /evaluate           feed prior WP to evaluate_artifact",
        "  /attach-<path>      attach a file for the next prompt (a space instead of",
        "                      '-' works too: /attach <path>; quote paths with spaces)",
        "  /clear              clear session",
        "  /streamon           streaming ON (default)",
        "  /streamoff          streaming OFF",
        "  /thinkshow          show thinking tokens while streaming",
        "  /thinkhide          hide thinking tokens (default)",
        "  /?  /help           show this help",
        "  /bye  /exit         quit",
        "",
        "Prompts can mix slash commands, /attach, and text in one block:",
        "  /domain-python /task-modify /attach-bug.py add docstrings",
        "",
        "(The server port is set once at startup via `CoE-coder -port <N>`; ",
        "there is no /url runtime command.)",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------
def _health(base: str) -> bool:
    try:
        return requests.get(f"{base}/health", timeout=2).status_code == 200
    except Exception:
        return False


def _chat_url(base: str) -> str:
    return f"{base}/v1/chat/completions"


def _wait_ready(base: str, timeout: int = 90) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if _health(base):
            return True
        time.sleep(1)
    return False


def start_server(base: str, port: int):
    """Reuse a healthy server, else spawn uvicorn. Return (proc_or_none)."""
    if _health(base):
        print(f"CoE-coder server already running on :{port} (reused)")
        return None
    print(f"[launch] starting CoE-coder server on :{port} ...")
    # ================================================================
    # 2026-09-21: RELOCATABLE LAUNCH PATH.
    # ================================================================
    # This block used to assume the developer's TWO-LEVEL workspace
    # wrapper:
    #     workspace = server_dir.parent.parent.resolve()   # college of experts
    #     module    = "CoE_agent.CoE_coder_16GB.server:app"
    # and ran uvicorn with cwd=workspace.  That hardwired the folder to
    # THIS tree: copy the package anywhere and `-m uvicorn ...` fails
    # unless a directory literally named `CoE_agent` happens to contain
    # it.  Nothing else in the package cares -- every other path is
    # already `__file__`-relative (config.py, save_artifact_to_disk,
    # executor.raw_log_dir, flm_lifecycle/backend_lifecycle log dirs) --
    # so this was the only thing preventing "move the folder and run it".
    #
    # THE CONSTRAINT THAT MATTERS.  `server.py` uses RELATIVE imports
    # (`from .vram import ...`), so it must be imported as part of a
    # PACKAGE -- it cannot be a top-level module.  That fixes the shape:
    #     module = "<package folder name>.server:app"
    #     cwd    = the PACKAGE FOLDER'S PARENT   (what puts it on sys.path)
    # Getting cwd wrong by one level is a `ModuleNotFoundError` for the
    # package name, which is exactly what the first attempt produced.
    #
    # TWO LAYOUTS ARE SUPPORTED, detected at runtime by where `server.py`
    # lives relative to this script:
    #   (1) flat  -- `server.py` sits NEXT TO this file.  This folder IS
    #                the package, so its NAME becomes the module prefix.
    #   (2) nested -- `server.py` is in a SINGLE SUBFOLDER.  The repo
    #                root holds this script and the package folder keeps
    #                its own name.
    # Either way, move the tree anywhere and it runs.
    #
    # [FLAGGED FALSE as of 2026-09-22]  This block previously asserted that
    # the folder name "must be a valid Python identifier (hence hyphens
    # would not do)", and that the nested layout existed because
    # "CoE-Coder-16GB is not a valid module name".  Quoted so the correction
    # is auditable.  **Both claims are false, and measured so:**
    # `uvicorn/importer.py` resolves the app with `importlib.import_module`,
    # NOT the `import` statement, so the module string is never parsed as
    # Python source.  Hyphens and spaces in the folder name therefore work
    # fine.  **The ONLY character that breaks it is a DOT**, because that is
    # what `import_module` splits on: a folder named `CoE.v2` yields
    # `CoE.v2.server:app`, which resolves to a nonexistent `CoE` package.
    # The nested layout is still worth KEEPING -- it lets the repo root be
    # named freely, e.g. `CoE-Coder-16GB`, which is the published repo name.
    # The practical rule is simply: no dots in the payload folder name.
    _here = Path(__file__).parent.resolve()
    pkg_root: Path | None = None
    if (_here / "server.py").is_file():
        pkg_root = _here                                   # layout (1) flat
    else:
        _cands = sorted(
            p for p in _here.iterdir()
            if p.is_dir() and (p / "server.py").is_file()
        )
        if len(_cands) == 1:
            pkg_root = _cands[0]                           # layout (2) nested
    if pkg_root is None:
        raise SystemExit(
            f"[launch] cannot locate server.py next to, or one level under, "
            f"{_here}.  Keep the package contents together and retry."
        )
    server_dir = pkg_root                                  # the PACKAGE folder
    workspace = server_dir.parent                          # cwd -> sys.path
    module = f"{server_dir.name}.server:app"
    # 2026-09-08: the server runs in its OWN console window.  Sharing
    # the CLI's terminal caused (a) uvicorn access logs to interleave
    # with the CLI prompt — desyncing the line-editing cursor from the
    # rendered rows — and (b) server log spam mid-typing.  A dedicated
    # `start` window gives the server live, visible logs (user wants
    # them observable) while the CLI terminal stays clean.
    log_path = server_dir / "logs" / f"server-{port}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Window title carries the port + pid for identification.
    title = f"CoE-coder server :{port}"
    cmd = (
        f'start "{title}" /MIN cmd /k "'
        f'title {title} && '
        f'"{sys.executable}" -m uvicorn {module} '
        f'--host 127.0.0.1 --port {port} '
        f'> "{log_path}" 2>&1"'
    )
    proc = subprocess.Popen(
        cmd,
        cwd=str(workspace),
        shell=True,
        # Cancel-turn isolation (2026-09-07): the server must NOT share
        # the CLI's console Ctrl+C.  A new process group keeps console
        # CTRL_C_EVENTS from reaching uvicorn, so Ctrl+C during a turn
        # cancels the TURN (CLI -> /v1/cancel) instead of killing the
        # whole instance.  The CLI stops the server itself after /bye
        # (proc.terminate() on its own Ctrl+C) preserving the old
        # "Ctrl+C stops the server" muscle memory.
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    if not _wait_ready(base):
        print(f"[launch] ERROR: server :{port} did not become ready in time.")
        _stop_server_tree(proc)
        return None
    print(f"[launch] CoE-coder server ready on :{port} {chr(40)}pid {proc.pid}{chr(41)}")
    return proc


def _stop_server_tree(proc) -> None:
    """Stop the separate-window server started by start_server().

    2026-09-08: the server now runs as `cmd /k uvicorn ...` in its own
    console window.  proc.terminate() only kills the `start` front-end;
    the `cmd /k` shell and the uvicorn child SURVIVE it (observed live:
    health stayed 200 after terminate).  The reliable stop is a tree-kill
    of the wrapper (taskkill /T), followed by a port-based fallback that
    sweeps any surviving uvicorn LISTENING on the port — preserving the
    post-/bye Ctrl+C muscle-memory stop.
    """
    try:
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            capture_output=True, timeout=10,
        )
    except Exception:
        pass
    # Fallback: sweep any process still LISTENING on the port (covers
    # the case where the wrapper already exited but uvicorn didn't, or
    # taskkill could not reach the detached tree).
    try:
        out = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        pids: set[int] = set()
        for ln in out.splitlines():
            parts = ln.split()
            if (
                len(parts) >= 5
                and parts[1].endswith(f":{DEFAULT_PORT}")
                and parts[3] == "LISTENING"
            ):
                try:
                    pids.add(int(parts[4]))
                except ValueError:
                    continue
        for pid in pids:
            if pid != os.getpid():
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(pid), "/T", "/F"],
                        capture_output=True, timeout=10,
                    )
                except Exception:
                    continue
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def parse_block(text: str):
    """Parse a one-block input into (routing_override, metadata, artifact, prompt)."""
    routing = {}
    metadata = {}
    artifact = None
    rest = text.strip()

    while True:
        m = SERVER_SLASH.match(rest)
        if m:
            kind, value = m.group(1), m.group(2)
            if kind in ("task_type", "tasktype", "task"):
                routing["task_type"] = TASK_TYPE_ALIASES.get(value, value)
            else:
                routing["domain"] = DOMAIN_ALIASES.get(value, f"domain_{value}")
            rest = rest[m.end():].lstrip()
            continue
        mt = THINK_SLASH.match(rest)
        if mt:
            routing["think"] = (mt.group(1) == "thinkon")
            rest = rest[mt.end():].lstrip()
            continue
        me = EVALUATE_SLASH.match(rest)
        if me:
            routing["task_type"] = "evaluate_artifact"
            routing["evaluate_prior_wp"] = True
            rest = rest[me.end():].lstrip()
            continue
        mc = CLOUD_SLASH.match(rest)
        if mc:
            metadata["cloud_mode"] = "direct"
            rest = rest[mc.end():].lstrip()
            continue
        # /attach-<path> or /attach <path> — dash or space separator both
        # accepted (regularized alongside /domain-/ /task-); quoted paths
        # with spaces supported.
        mq = re.match(r"^/attach[ -]+(\"[^\"]+\"|'[^']+'|\S+)\s*", rest)
        if mq:
            path = mq.group(1)
            artifact = load_artifact(path)
            rest = rest[mq.end():].lstrip()
            continue
        mclear = re.match(r"^/clear\s*", rest)
        if mclear:
            metadata["clear_session"] = True
            rest = rest[mclear.end():].lstrip()
            continue
        mshow = re.match(r"^/think(show|hide)\s*", rest)
        if mshow:
            metadata["show_thinking"] = (mshow.group(1) == "show")
            rest = rest[mshow.end():].lstrip()
            continue
        break

    return (routing or None), metadata, artifact, rest


# Mirrors artifact_store._EXTENSION_TABLE for the tags this CLI can emit, so
# the attach notice can name the file the artifact WILL be saved as without
# importing the server package.  Kept minimal on purpose: only the tags the
# loader can produce appear here.
_CLI_EXT_FOR_LANG = {
    "python": ".py", "javascript": ".js", "jsx": ".jsx",
    "typescript": ".ts", "tsx": ".tsx", "html": ".html", "css": ".css",
    "json": ".json", "markdown": ".md", "prose": ".md",
}


def _extension_for_lang(language: str) -> str:
    """Extension this language tag will be saved with (CLI-side mirror)."""
    return _CLI_EXT_FOR_LANG.get((language or "").lower(), ".txt")


def load_artifact(path: str):
    """Load a file as an artifact attachment.  Images are base64-encoded."""
    # Strip surrounding quotes in case the path was pasted or passed quoted
    # (e.g. <"C:\...\file.html"> from a shell copy of a path).
    path = path.strip()
    if len(path) >= 2 and path[0] in "\"'" and path[-1] == path[0]:
        path = path[1:-1].strip()
    p = Path(path)
    if not p.exists():
        print(f"  [attach] file not found: {path}")
        return None
    content = p.read_text(encoding="utf-8", errors="replace")
    suffix = p.suffix.lower()
    # 2026-09-16 FIX: the map used a single lossy "web" tag for .html/.js/.css.
    # `_EXTENSION_TABLE` can only map a tag to ONE extension, so that bucket
    # made a .js or .css attachment save as .html -- correct only for .html,
    # which is why this went unnoticed.  Now each suffix carries its REAL
    # language, so the round-trip is exact:
    #
    #     .html/.htm/.xhtml -> "html"        -> .html
    #     .js               -> "javascript"  -> .js
    #     .css              -> "css"         -> .css
    #
    # `.htm` matters beyond tidiness: Windows' 8.3 short names render
    # `..._v1_...html` as `13_MOD~1.HTM`, which is what a user copying a path
    # from Explorer actually pastes.  Before this, `.htm` missed the map,
    # fell to the "prose" default, and the artifact was saved as `.md`.
    #
    # SVG is deliberately ABSENT: it is an image format, and the only web tag
    # available maps to .html, so an .svg attachment would save as .html.
    # It needs a proper "svg" -> ".svg" entry in _EXTENSION_TABLE first.
    _ATTACH_LANG = {
        ".py": "python",
        ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
        ".jsx": "jsx",
        ".ts": "typescript", ".tsx": "tsx",
        ".html": "html", ".htm": "html", ".xhtml": "html",
        ".css": "css", ".scss": "css",
        ".json": "json", ".md": "markdown", ".txt": "prose",
    }
    if suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        print(f"  [attach] {p.name} (image, {p.stat().st_size} bytes)")
        return {"kind": "image", "content": b64, "mime": f"image/{suffix.strip('.')}",
                "filename": p.name}
    # 2026-09-16 (user-requested): SURFACE A MISS instead of silently guessing.
    # An unrecognised suffix previously became "prose" with no signal, so a
    # genuine web file could be saved as .md and nothing said why.  Now the
    # miss is reported and the fallback is NAMED, so the consequence is
    # visible at the moment it is decided.
    _mapped = suffix in _ATTACH_LANG
    if _mapped:
        lang = _ATTACH_LANG[suffix]
        print(f"  [attach] {p.name} ({len(content)} chars, {lang})")
    else:
        lang = "prose"
        print(f"  [attach] {p.name} ({len(content)} chars)")
        print(f"  [attach] NOTE: extension {suffix or '(none)'!r} not recognised -- "
              f"treating as {lang!r}.  The artifact will be saved as "
              f"{_extension_for_lang(lang)}.  Rename it with a known extension "
              f"(e.g. .html) if that is wrong.")
    return {"kind": "code", "content": content, "language": lang,
            "origin": "pasted", "filename": p.name}


def _print_meta(meta: dict) -> None:
    """Print a server metadata event in a user-visible form.

    Phase 3 (no 200-with-silence): metadata events carry the server's routing,
    clarification, alert, and error surfaces.  Dropping them made server-side
    decisions invisible in the CLI.

    NOTE: this function is called once per metadata chunk and holds NO state of
    its own, so anything that must survive between chunks lives in module-level
    `_CURRENT_TIER` (set on `tier_start`, read by the phase and verdict lines).
    Do not introduce a local "current tier" variable here -- there is no
    surrounding loop to hold it.
    """
    global _CURRENT_TIER
    if not isinstance(meta, dict):
        return
    if meta.get("needs_clarification"):
        nc = meta["needs_clarification"]
        print(f"\n  [clarification] {nc.get('question', '')}", flush=True)
        for cand in (nc.get("candidates") or [])[:10]:
            print(f"    - {cand}", flush=True)
        return
    if meta.get("rejection"):
        print(f"\n  [rejected] {meta['rejection'].get('code')}: {meta['rejection'].get('message', '')[:300]}", flush=True)
        return
    if meta.get("alert"):
        alert = meta["alert"]
        print(f"\n  [alert] {alert.get('code')}: {alert.get('message', '')[:300]}", flush=True)
        return
    if meta.get("error"):
        print(f"\n  [server-error] {meta['error']}", flush=True)
        return
    if meta.get("tier_start"):
        _CURRENT_TIER = meta["tier_start"]
        # 2026-09-16 (user-requested): name the MODEL on the tier line for
        # CLOUD tiers only.  `model_id` already rides the tier_start event
        # (server.py forwards it as metadata), but for LOCAL tiers it is a
        # slot designation, which the user does not want shown.  T3 is shown
        # because "which cloud model did it call" is otherwise invisible.
        _ts_model = ""
        if str(meta["tier_start"]).upper().startswith("T3"):
            _ts_model = meta.get("model_id") or ""
        _ts_tag = f" -> {_ts_model}" if _ts_model else ""
        print(f"\n  [tier] {meta['tier_start']}{_ts_tag}", end="", flush=True)
        return
    if meta.get("checkpoint"):
        # 2026-09-19 (PLAN_progress_mechanism.md phase 3): periodic progress.
        # ONE mechanism -- the runtime's single `checkpoint` event, carried as
        # SSE metadata while streaming and read from the turn record for the
        # non-streaming poll endpoint.  This replaces the dead `summary` event,
        # which was emitted and consumed by nothing.
        #
        # Placement matters: this branch sits BEFORE `tier_retry` and `status`
        # because it is the highest-frequency metadata event (one per interval),
        # and the ordering above is a first-match chain.
        cp = meta["checkpoint"]
        _cp_tier = str(cp.get("tier") or _CURRENT_TIER or "?")
        _cp_elapsed = cp.get("elapsed_s")
        _cp_len = cp.get("text_length")
        _cp_prev = (cp.get("preview") or "").replace("\n", " ").strip()
        _cp_el = f"{float(_cp_elapsed):.0f}s" if _cp_elapsed is not None else "?s"
        _cp_ch = f"{int(_cp_len):,} chars" if _cp_len is not None else "? chars"
        # Indent past any partial token line so the progress line is legible
        # without the leading newline the other metadata lines use (a
        # checkpoint can arrive mid-line, and a blank line each time would
        # shred a live token stream).
        print(f"\n  [progress] {_cp_tier} . {_cp_el} . {_cp_ch}", end="", flush=True)
        if _cp_prev:
            print(f" -- {_cp_prev[-120:]}", flush=True)
        else:
            print("", flush=True)
        return
    if meta.get("tier_retry"):
        # 2026-09-13: accurate retry reporting. The runtime tells us the
        # retry kind, which attempt this is, the limit, and what the
        # ladder decided to do next -- so the CLI never has to guess.
        tr = meta["tier_retry"]
        kind = str(tr.get("retry_kind") or "?")
        reason = str(tr.get("reason") or "?")
        attempt = tr.get("attempt")
        limit = tr.get("limit")
        action = str(tr.get("action") or "")
        action_label = {
            "fresh_seed_reroll": "retrying this tier with a FRESH SEED",
            "same_seed_reground": "retrying this tier with error feedback",
            "escalate_next_slot": "budget spent for this slot -> escalating",
        }.get(action, action or "retrying")
        print(
            f"\n  [retry] {_CURRENT_TIER} {kind} ({reason}) attempt {attempt}/{limit}"
            f" -- {action_label}",
            flush=True,
        )
        return
        sf = meta["slot_fallback"]
        print(f"\n  [slot-fallback] {sf.get('role')}: {sf.get('slot_model')} -> {sf.get('fallback')} ({sf.get('reason')})", flush=True)
        return
    if meta.get("status"):
        # Long-silence phase tag (2026-09-05): the server announces model
        # loads and attachment reads so dead air is never mistaken for a
        # hang. One line, no newline — the following tokens continue the
        # line visually.
        st = meta["status"]
        kind = st.get("kind", "")
        detail = st.get("detail", "")
        label = {
            "model_load": "loading model",
            "reading_attachment": "reading attachment",
            "prefill": "prefill",
            "thinking": "thinking",
        }.get(kind, kind or "working")
        suffix = f" ({detail})" if detail else ""
        # 2026-09-16 (user-requested): label the phase line with the CURRENT
        # tier.  A retry generates in the SAME tier index but a NEW attempt,
        # and `tier_start` is not re-emitted for it -- so the retry's prefill
        # previously appeared with no tier context at all and read as "the
        # same tier generated a second time".
        tier_tag = f"{_CURRENT_TIER} " if _CURRENT_TIER else ""
        print(f"\n  [{tier_tag}{label}{suffix}] ...", end="", flush=True)
        return
    if meta.get("tier_complete"):
        # 2026-09-13: a tier that has actually FINISHED (passed, or the
        # ladder is leaving it).  In-tier retries are reported via
        # [retry] above, so this line is now unambiguous.  Rendered in a
        # readable form rather than dumped as raw JSON.
        #
        # 2026-09-16 (user-requested): NAME THE TIER on this line.  It used
        # to be a bare continuation of the preceding `[tier] <name>` line,
        # which is misleading whenever a RETRY happened in between: the
        # `[tier]` line names the tier that FAILED (T0), then the retry runs
        # in a DIFFERENT tier (T1b) whose `[tier]` announcement never prints,
        # so the verdict appeared to belong to T0.  Observed live as:
        #
        #     [tier] T0
        #     [prefill] ... -> not passed      <- actually T0
        #     [prefill] ... -> passed          <- actually T1b, unlabelled
        #
        # Reading "T0 ... passed" made the ladder look like it continued
        # after a pass.  The tier name here comes from the event itself, so
        # it is always the tier that actually produced this verdict.
        td = meta.get("tier_details") or {}
        tier_name = td.get("tier") or meta.get("tier_complete") or "?"
        verdict = "passed" if td.get("passed") else "not passed"
        extra = []
        if td.get("is_t3"):
            # 2026-09-16 (user-requested): show the cloud MODEL in place of
            # the role name.  `target` is "cloud_sota" -- a role, not a model.
            # Fall back to `target` so this still says something if a caller
            # emits a T3 with no model recorded.
            extra.append(f"cloud ({td.get('model') or td.get('target', '')})")
        if td.get("aborted"):
            extra.append("aborted")
        if td.get("accepted_partial"):
            extra.append("accepted partial")
        tail = f" [{', '.join(extra)}]" if extra else ""
        # Leading newline so the verdict starts its own line instead of
        # continuing the [tier]/[prefill] line above it, now it carries the
        # tier name and must be readable on its own.
        print(f"\n  [tier] {tier_name} -> {verdict}{tail}", flush=True)
        return
    if meta.get("heartbeat"):
        return  # connection keep-alive ping; not user-relevant
    if meta.get("coe_routing"):
        return  # routing telemetry: too noisy per-token, printed once via [tier]
    if set(meta.keys()) <= {"is_thinking", "text_length", "prefill"}:
        return  # per-token stream telemetry: belongs to the token, not an event
    # Unknown metadata: show it rather than swallow it.
    print(f"\n  [meta] {json.dumps(meta)[:200]}", flush=True)


def _latex_to_unicode(text: str) -> str:
    """Convert common LaTeX math to readable Unicode text.

    Lightweight, stdlib-only, display-only: the console gets a
    legible approximation while the artifact keeps raw LaTeX (the
    gold source for VS Code preview / GitHub rendering).
    """
    out = text
    # Display math delimiters first.
    out = re.sub(r"\$\$(.+?)\$\$", _convert_latex_span, out, flags=re.DOTALL)
    out = re.sub(r"\$([^$]+?)\$", _convert_latex_span, out)
    return out


def _read_braced(span: str, i: int) -> tuple[str, int]:
    """Read a {...} group starting at span[i] == '{'.

    Returns (contents, next_index). Handles nesting.
    """
    depth = 0
    start = i + 1
    j = i
    while j < len(span):
        if span[j] == '{':
            depth += 1
        elif span[j] == '}':
            depth -= 1
            if depth == 0:
                return span[start:j], j + 1
        j += 1
    return span[start:], len(span)


_LATEX_SYMBOLS = {
    "\\pm": "±", "\\times": "×", "\\cdot": "·",
    "\\Delta": "Δ", "\\delta": "δ", "\\rho": "ρ",
    "\\theta": "θ", "\\alpha": "α", "\\beta": "β",
    "\\pi": "π", "\\lambda": "λ", "\\mu": "μ",
    "\\sigma": "σ", "\\omega": "ω", "\\Omega": "Ω",
    "\\phi": "φ", "\\nabla": "∇", "\\infty": "∞",
    "\\sum": "∑", "\\int": "∫", "\\neq": "≠",
    "\\leq": "≤", "\\geq": "≥", "\\approx": "≈",
    "\\rightarrow": "→", "\\to": "→", "\\left": "",
    "\\right": "", "\\,": " ", "\\;": " ", "\\ ": " ",
    "\\quad": "  ", "\\qquad": "    ", "\\partial": "∂",
}


def _convert_latex_span(m: "re.Match[str]") -> str:
    return _convert_latex_text(m.group(1))


def _convert_latex_text(span: str) -> str:
    """Recursive brace-aware LaTeX fragment conversion."""
    out: list[str] = []
    i = 0
    n = len(span)
    while i < n:
        ch = span[i]
        if ch == "\\":
            matched = False
            for cmd, rep in _LATEX_SYMBOLS.items():
                if span.startswith(cmd, i):
                    out.append(rep)
                    i += len(cmd)
                    matched = True
                    break
            if matched:
                continue
            if span.startswith("\\frac", i) and i + 5 < n and span[i + 5] == "{":
                a, j = _read_braced(span, i + 5)
                b, j2 = _read_braced(span, j)
                out.append("(" + _convert_latex_text(a) + ")/(" + _convert_latex_text(b) + ")")
                i = j2
                continue
            if span.startswith("\\sqrt", i) and i + 5 < n and span[i + 5] == "{":
                x, j = _read_braced(span, i + 5)
                out.append("√(" + _convert_latex_text(x) + ")")
                i = j
                continue
            if span.startswith("\\text", i) and i + 5 < n and span[i + 5] == "{":
                x, j = _read_braced(span, i + 5)
                out.append(x)
                i = j
                continue
            if span.startswith("\\mathbf", i) and i + 7 < n and span[i + 7] == "{":
                x, j = _read_braced(span, i + 7)
                out.append(_convert_latex_text(x))
                i = j
                continue
            # Unknown command: drop the backslash and letters.
            j = i + 1
            while j < n and span[j].isalpha():
                j += 1
            if j == i + 1:
                out.append(ch)
                i += 1
            else:
                i = j
            continue
        if ch in "^_" and i + 1 < n:
            nxt = span[i + 1]
            if nxt == "{":
                body, j = _read_braced(span, i + 1)
            else:
                body = nxt
                j = i + 2
            if ch == "^":
                out.append(body.translate(str.maketrans("0123456789+-=()ni", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿᵢ")))
            else:
                out.append(body.translate(str.maketrans("0123456789+-=()aeiou", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑᵢₒᵤ")))
            i = j
            continue
        out.append(ch)
        i += 1
    return "".join(out).strip()


def _sup_group(m: "re.Match[str]") -> str:
    body = m.group(1)
    table = str.maketrans("0123456789+-=()ni", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿᵢ")
    return body.translate(table)


def _sub_group(m: "re.Match[str]") -> str:
    body = m.group(1)
    table = str.maketrans("0123456789+-=()aeiou", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑᵢₒᵤ")
    return body.translate(table)


def _sup_group(m: "re.Match[str]") -> str:
    body = m.group(1)
    table = str.maketrans("0123456789+-=()ni", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ")
    return body.translate(table)


def _sub_group(m: "re.Match[str]") -> str:
    body = m.group(1)
    table = str.maketrans("0123456789+-=()aeiou", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑᵢₒᵤ")
    return body.translate(table)


def _cancel_turn(base_url: str, request_id: str) -> bool:
    """Best-effort cancel of an in-flight turn.  Bare message: the user
    just returns to the prompt; session/artifact state is preserved by
    the server (attempt recorded as cancelled).  Returns True if the
    server accepted the cancel."""
    try:
        r = requests.post(f"{base_url}/v1/cancel",
                          json={"request_id": request_id}, timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def stream_request(url, payload, base_url: str = "", request_id: str = ""):
    """POST with streaming; print tokens as they arrive.

    Tokens are buffered per line, LaTeX-converted to Unicode math for
    display, then printed. The artifact keeps raw LaTeX (display-only
    conversion).
    """
    line_buf = ""
    headers = {"X-Request-Id": request_id} if request_id else {}
    try:
        with requests.post(url, json=payload, stream=True,
                           timeout=REQUEST_TIMEOUT_S, headers=headers) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                content = delta.get("content")
                meta = delta.get("metadata") or {}
                if content:
                    line_buf += content
                    # Flush complete lines (converted); keep the
                    # trailing partial line buffered.
                    while "\n" in line_buf:
                        complete, line_buf = line_buf.split("\n", 1)
                        print(_latex_to_unicode(complete), flush=True)
                elif meta and line_buf:
                    # Metadata events arrive between content chunks;
                    # flush the buffer first so nothing is visually
                    # lost or interleaved out of order.
                    print(_latex_to_unicode(line_buf), flush=True)
                    line_buf = ""
                if meta:
                    _print_meta(meta)
            # Flush any trailing partial line.
            if line_buf:
                print(_latex_to_unicode(line_buf), flush=True)
                line_buf = ""
        # Clear separation between generated content and trailing
        # metadata/event output (two newlines).
        print("\n\n", flush=True)
    except KeyboardInterrupt:
        # Cancel-turn (2026-09-07): Ctrl+C cancels ONLY this turn.
        # The server (in its own process group) never sees the signal.
        # Server-side state is preserved: the attempt is recorded as
        # cancelled, session memory + artifact store untouched.
        if base_url and request_id:
            _cancel_turn(base_url, request_id)
        print("\n  [cancelled] turn cancelled (state preserved)")
        print("\n\n", flush=True)
    except Exception as e:
        print(f"\n  [error] {type(e).__name__}: {e}")


def plain_request(url, payload, base_url: str = "", request_id: str = "") -> None:
    """POST without streaming; print the answer or visible errors.

    2026-09-19 (PLAN_progress_mechanism.md phase 9): non-streaming mode now has
    BOTH progress and a working cancel, via the turn record rather than the SSE
    channel.

      * X-Request-Id is sent, so the server registers the turn under an id the
        CLI already knows (it would otherwise MINT one the CLI cannot name).
      * A background thread polls GET /v1/turn/{id}/progress and prints a
        `[progress]` line, so a long non-streaming turn is no longer a silent
        black box.
      * On Ctrl+C the CLI now ACTUALLY calls /v1/cancel.  Previously the
        handler printed "turn cancelled (state preserved)" while sending
        nothing at all -- the server kept working to completion, holding VRAM
        and committing the artifact into a session whose client had moved on.

    Cancellation here is COOPERATIVE (the server stops at its next progress
    checkpoint), not immediate, because the HTTP response is still in flight on
    this thread.  The message says so.
    """
    headers = {"X-Request-Id": request_id} if request_id else {}
    stop = threading.Event()

    def _poll() -> None:
        last = 0
        while not stop.wait(5.0):
            try:
                rp = requests.get(f"{base_url}/v1/turn/{request_id}/progress",
                                  timeout=5)
                if rp.status_code != 200:
                    continue
                d = rp.json()
                if d.get("status") != "running":
                    return
                since = d.get("since_last_event_s")
                # Only print when something moved, or when the turn has gone
                # quiet long enough to look stalled -- otherwise a 5s poll on a
                # silent load would spam identical lines.
                moved = d.get("next_cursor", 0) != last
                stalled = since is not None and since > 60
                if moved or stalled:
                    last = d.get("next_cursor", 0)
                    tag = "no output for %.0fs" % since if stalled and not moved else \
                          "%s chars" % f"{int(d.get('text_length') or 0):,}"
                    print(f"\n  [progress] {d.get('tier') or '?'} . "
                          f"{float(d.get('elapsed_s') or 0):.0f}s . {tag}",
                          flush=True)
            except Exception:
                return

    poller = None
    if base_url and request_id:
        poller = threading.Thread(target=_poll, daemon=True)
        poller.start()
    try:
        r = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_S,
                          headers=headers)
        r.raise_for_status()
        data = r.json()
        if r.status_code >= 400:
            print(f"\n  [error {r.status_code}] {json.dumps(data)[:300]}", flush=True)
            return
        msg = (data.get("choices") or [{}])[0].get("message", {})
        print(msg.get("content") or data.get("content") or "")
        ct = data.get("coe_turn") or {}
        if ct:
            print(f"\n  [turn] {ct.get('request_id')} status={ct.get('status')}"
                  f" aborted={ct.get('aborted')}", flush=True)
    except KeyboardInterrupt:
        # A REAL cancel request, then wait for the server to honour it at its
        # next checkpoint.  The poll thread reports when the turn leaves
        # "running"; we do not pretend the work stopped instantly.
        if base_url and request_id:
            _cancel_turn(base_url, request_id)
            print("\n  [cancelling] requested -- the server stops at the next "
                  "progress checkpoint (state preserved)", flush=True)
            for _ in range(60):
                time.sleep(1)
                try:
                    rp = requests.get(
                        f"{base_url}/v1/turn/{request_id}/progress", timeout=5)
                    if rp.status_code != 200 or rp.json().get("status") != "running":
                        print("  [cancelled] turn stopped (state preserved)",
                              flush=True)
                        break
                except Exception:
                    break
        else:
            print("\n  [cancelled] request abandoned locally (no request id)",
                  flush=True)
    except Exception as e:
        print(f"  [error] {type(e).__name__}: {e}")
    finally:
        stop.set()


def run_cli(base: str):
    global current_pending_artifact
    url = _chat_url(base)
    streaming = True
    # 2026-09-09: banner renders ONLY after start_server() has fully
    # returned (main() calls run_cli after _wait_ready passed).  The old
    # torn-render artifact — `server: htt/` — was the banner printing
    # during the cold-start handoff window while the server console
    # window was spawning: conhost redrew mid-paint and clipped the row.
    # By the time run_cli runs, the window spawn is DONE, so nothing
    # competes with these two lines.  Also: single trailing newline
    # (the old "\n" inside the f-string + print's own newline made an
    # empty row that amplified the desync).
    print("CoE-coder CLI — /? for help, /bye to quit")
    print(f"server: {url}")
    print()

    while True:
        try:
            line = input("CoE-coder> ").strip()
        except EOFError:
            print("\nbye")
            break
        except KeyboardInterrupt:
            # Ctrl+C at the PROMPT (no turn in flight) is a no-op:
            # redisplay the prompt.  /bye remains the way out.  This
            # mirrors the during-turn contract (Ctrl+C = cancel current
            # op only, never the session or the server).
            print("\n  (nothing to cancel — /bye to quit)")
            continue
        if not line:
            continue

        if line in ("/?", "/help"):
            print(help_text())
            continue
        if line in ("/bye", "/exit"):
            print("bye")
            break
        if line == "/streamon":
            streaming = True
            print("  streaming ON")
            continue
        if line == "/streamoff":
            streaming = False
            print("  streaming OFF")
            continue
        if line.startswith("/"):
            if not (SERVER_SLASH.match(line) or THINK_SLASH.match(line)
                    or EVALUATE_SLASH.match(line) or CLOUD_SLASH.match(line)
                    or re.match(r"^/attach[ -]", line)
                    or re.match(r"^/think(show|hide)\s*", line)):
                print(f"  unknown command: {line.split()[0]}  (/? for help)")
                continue

        routing, metadata, artifact, prompt = parse_block(line)

        # /thinkshow or /thinkhide with no trailing prompt: toggle only,
        # nothing to send this turn.
        if not prompt and not artifact and ("show_thinking" in (metadata or {})):
            print(f"  thinking tokens: {'SHOW' if metadata['show_thinking'] else 'HIDE'} (for this and following prompts)")
            continue

        # /attach with no trailing prompt = pending attachment held for
        # the next prompt; a single-turn send always requires a prompt.
        if artifact is not None and not prompt:
            current_pending_artifact = artifact
            continue
        if not prompt and line.lstrip().startswith("/attach"):
            # Attachment load failed (load_artifact already reported);
            # nothing to send this turn.
            continue

        if current_pending_artifact and artifact is None:
            artifact = current_pending_artifact
            current_pending_artifact = None

        payload = {"messages": [{"role": "user", "content": prompt or ""}],
                   "metadata": metadata or {}, "stream": streaming}
        if routing:
            payload["routing_override"] = routing
        if artifact:
            if artifact.get("kind") == "image":
                payload.setdefault("context_attachments", []).append(
                    {"type": "image", "content": artifact["content"],
                     "mime": artifact["mime"], "language": None})
            else:
                payload["artifact"] = {"content": artifact["content"],
                                       "language": artifact["language"],
                                       "origin": artifact["origin"],
                                       "filename": artifact["filename"]}

        # Cancel-turn: one request id per turn; sent as a header so
        # /v1/cancel can address THIS turn, and Ctrl+C cancels it.
        request_id = uuid.uuid4().hex
        if streaming:
            try:
                stream_request(url, payload, base_url=base, request_id=request_id)
            except KeyboardInterrupt:
                # TRUE HERE, and only here.  stream_request's own handler
                # already POSTed /v1/cancel, which cancels the executor TASK
                # outright -- the attempt is recorded as cancelled and session
                # state is preserved.  Contrast the non-streaming branch below,
                # where the claim this line used to make was FALSE because
                # nothing was ever sent.
                print("\n  [cancelled] turn cancelled (state preserved)")
        else:
            try:
                plain_request(url, payload, base_url=base, request_id=request_id)
            except KeyboardInterrupt:
                # plain_request handles its own KeyboardInterrupt (it performs
                # the real cancel and waits for the server).  This is only a
                # backstop; it must NOT claim the turn was cancelled, because
                # at this point we cannot know that.
                print("\n  [interrupted] request abandoned locally; "
                      "server state unknown", flush=True)


# ---------------------------------------------------------------------------
# Main launcher
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        prog="CoE-coder",
        description="CoE-coder-16GB launcher: OpenAI-compatible server + optional interactive CLI.",
    )
    ap.add_argument("-cli", action="store_true", help="run the interactive CLI loop instead of server-only mode")
    ap.add_argument("-port", type=int, default=DEFAULT_PORT,
                    help=f"server port (default {DEFAULT_PORT})")
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    proc = start_server(base, args.port)
    if proc is None and not _health(base):
        sys.exit("[launch] server failed to start; aborting.")

    if args.cli:
        try:
            run_cli(base)
        finally:
            # /bye returns here; the SERVER stays running (shared resource).
            # Cancel-turn isolation means console Ctrl+C no longer reaches
            # the server, so the CLI now stops it explicitly to preserve
            # the "Ctrl+C stops the server" muscle memory after /bye.
            if proc is not None:
                print(f"\n[note] server still running on :{args.port} (pid {proc.pid}) — Ctrl+C to stop it")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n[note] stopping CoE-coder server")
                    _stop_server_tree(proc)
    else:
        print(f"[launch] CoE-coder server active — OpenAI-compatible surface at {base}")
        print("[launch] Press Ctrl+C to stop the server.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[launch] stopping CoE-coder server")


if __name__ == "__main__":
    main()
