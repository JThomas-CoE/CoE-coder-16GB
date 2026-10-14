"""Public-API transports for the clean slice.

The Ollama transport uses Ollama's **native** ``/api/chat`` endpoint
because the OpenAI-compatible ``/v1/chat/completions`` endpoint
silently ignores ``options.num_ctx`` in the per-request payload and
always allocates the Modelfile's declared context (262,144 by
default).  Native ``/api/chat`` honours ``options.num_ctx`` per call,
which is what the compact-tier budget requires.

External clients still see a normal OpenAI-shaped surface — the
``clean_server`` module wraps the NDJSON stream from ``/api/chat``
into OpenAI SSE on the way out.  No legacy backend classes are
reused here.

**Universal chokepoint.**  ``CleanOllamaTransport`` is the *only*
code path that issues an Ollama request on behalf of the clean
slice.  Every payload built for Ollama passes through
``_enforce_bounded_options`` which clamps ``num_ctx`` to the
tier budget and refuses to build a payload that would exceed the
budget under any worst-case KV allocation.  This makes a runaway
``256k`` context (the Modelfile default) impossible by
construction: the only way to talk to Ollama is through this
method, and the method rejects anything that would blow the
compact-tier VRAM cap.
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from dataclasses import replace
from typing import Any, AsyncIterator

import httpx
from jinja2 import Environment

from .contracts import (
    AttachmentMeta,
    CleanPackage,
    CleanRouterDecision,
    NeedsClarification,
    RouterFailure,
    RouterTaskUnsupported,
)
from .resolution import (
    CANONICAL_TASK_SCOPES,
    CANONICAL_TASK_TYPES,
    DOMAIN_FALLBACK,
    ROUTER_EMIT_TASK_TYPES,
)
from .pathology import PathologyMonitor


class StreamAborted(Exception):
    """Raised by ``CleanOllamaTransport.stream()`` when a caller
    has called ``request_close()`` to interrupt an in-flight
    request.  This is the abort path the runtime's starvation
    timer uses when the first-token or slow-decode deadline
    fires: the timer sets a flag on the transport, the
    ``stream()`` generator raises ``StreamAborted`` on its
    next iteration, the runtime's broad ``except Exception``
    catches it, and the runtime's per-attempt error path
    translates the message into the right
    ``first_token_timeout`` / ``slow_decode_timeout`` error
    string the retry ladder expects.

    Why a custom exception instead of cancelling the
    generator: cancelling the ``stream()`` generator mid-
    ``aiter_lines()`` causes httpx's ``async with`` block to
    receive a ``GeneratorExit`` and then ``athrow()`` the
    cancellation back into the response object, which is
    already in the middle of a read.  The result is
    ``RuntimeError: athrow(): asynchronous generator is
    already running`` from inside the httpx context
    manager, which surfaces to the runtime as
    ``backend_error: ...`` (wrong error code) and causes
    the runtime to fall back to the secondary model.  By
    raising ``StreamAborted`` from a clean yield-point
    instead, the generator unwinds normally, the
    ``async with`` block exits, the ``async for`` in the
    runtime catches the exception, and the error string
    is what we want.
    """


class CloudStreamSilent(Exception):
    """D4 (2026-08-29): raised by the runtime's silence-guarded T3
    chunk iterator when a cloud stream produces NO lines (not even
    SSE keep-alives) for ``cloud_silence_timeout_s``.  Distinct from
    ``StreamAborted`` (local-only, caller-requested close): this one
    means the REMOTE stream went dead.  The T3 invoker catches it and
    records the attempt as ``cloud_silence_timeout`` so the failure is
    visible in the attempt log instead of being cancelled away by the
    outer SSE watchdog."""


# Universal chokepoint constants — single source of truth for the
# bounded context.  The clean slice behaves as if running on a
# dedicated 16 GB card, so the maximum per-request context is
# 32768 tokens.  Any code that wants to talk to Ollama goes
# through ``CleanOllamaTransport`` which enforces this value.
BOUNDED_NUM_CTX: int = 32768
BOUNDED_NUM_PREDICT: int = 24576

# 2026-09-12 (user-approved): the ONLY option keys a caller may inject via
# ``non_streaming_call(sampling=...)``.  This whitelist exists so that a
# sampling dict can never carry ``num_ctx`` or ``num_predict`` and thereby
# bypass ``_enforce_bounded_options`` — the bounded-context chokepoint owns
# those two keys exclusively.  Used by the self-review reviewer call, which
# must be deterministic (temperature 0.0, top_k 1, fixed seed) instead of
# running at Ollama's defaults.
_SAMPLING_OPTION_KEYS: frozenset[str] = frozenset({
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "seed",
    "repeat_penalty",
    "repeat_last_n",
    "stop",
})


# ---------------------------------------------------------------------------
#  Pathology monitor lifecycle helpers
# ---------------------------------------------------------------------------

def pathology_mark_stopped(monitor: PathologyMonitor) -> None:
    """Mark a monitor's stream as finished.

    The runtime's background timer (the starvation timer)
    polls the monitor every 2s.  When the stream finishes
    normally, the timer should stop polling; otherwise it
    continues until the monitor is GC'd.  We set a flag
    on the monitor that ``check()`` honors.

    Safe to call when ``monitor`` is ``None`` (no-op).
    Safe to call multiple times.
    """
    if monitor is None:
        return
    # We don't add a new public attribute to the
    # ``PathologyMonitor`` dataclass; instead we set the
    # existing ``_started`` flag to ``False`` and let
    # ``check()`` early-out.  This is a private field
    # mutation but the transport is a privileged caller
    # (the runtime constructs the monitor and gives it to
    # the transport; the transport knows the monitor's
    # lifecycle).
    monitor._started = False


# ---------------------------------------------------------------------------
#  Helpers shared by the router and the Ollama transport
# ---------------------------------------------------------------------------

# Keys that must always be sent to Ollama as ``options.<key>``.
_OLLAMA_OPTION_KEYS = frozenset({
    "num_ctx", "num_predict", "num_gpu", "num_thread", "num_batch",
    "keep_alive", "temperature", "top_k", "top_p", "repeat_penalty",
    "repeat_last_n", "seed", "stop", "tfs_z", "typical_p", "mirostat",
    "mirostat_eta", "mirostat_tau",
})

# Cap on the number of characters emitted per yielded chunk.  Ollama
# (and FLM) deliver the entire ``content`` field of each NDJSON/SSE
# line as a single payload, which often coalesces dozens of tokens
# into one chunk.  The clean slice wants token-granular live
# streaming so the recorder flushes often.  We split each payload on
# whitespace, keeping each whitespace run attached to the preceding
# word, so a 200-character payload becomes roughly 20-40 small
# chunks rather than one giant block.
_CHUNK_MAX_CHARS = 8


def _split_token_chunks(text: str) -> list[str]:
    """Break a large text payload into small whitespace-attached pieces.

    The input is split on whitespace runs; each run is attached to
    the word that precedes it so visible spacing is preserved.  Empty
    pieces (e.g. leading whitespace) are skipped.
    """
    if not text:
        return []
    if len(text) <= _CHUNK_MAX_CHARS:
        return [text]
    pieces: list[str] = []
    buf: list[str] = []
    buf_chars = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        buf.append(ch)
        buf_chars += 1
        i += 1
        if ch.isspace():
            # Flush when we hit a whitespace boundary once the buffer
            # is at least the chunk size, so we don't emit tiny
            # single-character chunks when input is mostly spaces.
            if buf_chars >= _CHUNK_MAX_CHARS:
                pieces.append("".join(buf))
                buf = []
                buf_chars = 0
        elif buf_chars >= _CHUNK_MAX_CHARS:
            pieces.append("".join(buf))
            buf = []
            buf_chars = 0
    if buf:
        pieces.append("".join(buf))
    return pieces


def _image_base64_value(image: str) -> str:
    """Return the raw base64 payload used by Ollama native messages."""
    if image.startswith("data:"):
        _header, _separator, payload = image.partition(",")
        return payload
    return image


def _attach_ollama_images(
    messages: list[dict[str, Any]], images: list[str],
) -> list[dict[str, Any]]:
    """Attach images to the last user message in Ollama's native shape."""
    adapted = list(messages)
    for index in range(len(adapted) - 1, -1, -1):
        if adapted[index].get("role") == "user":
            adapted[index] = {
                **adapted[index],
                "images": [_image_base64_value(image) for image in images],
            }
            break
    return adapted


def _attach_openai_images(
    messages: list[dict[str, Any]], images: list[str],
) -> list[dict[str, Any]]:
    """Append images to the last user message as OpenAI content parts."""
    adapted = list(messages)
    for index in range(len(adapted) - 1, -1, -1):
        if adapted[index].get("role") != "user":
            continue
        current = adapted[index].get("content")
        if isinstance(current, list):
            parts = list(current)
        else:
            parts = ([{"type": "text", "text": current}] if current else [])
        parts.extend(
            {"type": "image_url", "image_url": {"url": image}}
            for image in images
        )
        adapted[index] = {**adapted[index], "content": parts}
        break
    return adapted


def _enforce_bounded_options(options: dict[str, Any], *, num_predict: int | None) -> dict[str, Any]:
    """Universal chokepoint: clamp Ollama options to the tier budget.

    Every Ollama request that goes through the clean slice must
    pass through this function.  The function refuses to return a
    payload that would let Ollama allocate more than the bounded
    context.  On a real 16 GB card a 256k context load would
    OOM-kill the host before any watchdog could react, so the
    only safe approach is to never send the dangerous request in
    the first place.

    Returns a new dict with ``num_ctx`` and ``num_predict`` clamped
    to the budget.  Raises ``RuntimeError`` if the caller tries to
    disable the cap (``num_ctx <= 0``).
    """
    bounded: dict[str, Any] = dict(options)
    raw_ctx = bounded.get("num_ctx")
    if raw_ctx is None:
        raise RuntimeError(
            f"Ollama payload missing num_ctx; the clean slice requires every request to declare its bounded context (BOUNDED_NUM_CTX={BOUNDED_NUM_CTX})"
        )
    try:
        ctx = int(raw_ctx)
    except (TypeError, ValueError):
        raise RuntimeError(f"Ollama num_ctx must be an integer, got {raw_ctx!r}")
    if ctx <= 0:
        raise RuntimeError("Ollama num_ctx must be positive; the clean slice refuses to send a 0/negative context request")
    if ctx > BOUNDED_NUM_CTX:
        # Clamp silently — callers sometimes pass the Modelfile
        # default (256k) without realising it will blow the
        # budget.  We rewrite it to the bounded value rather
        # than refuse, so the call still works.
        bounded["num_ctx"] = BOUNDED_NUM_CTX
    else:
        bounded["num_ctx"] = ctx
    # Clamp num_predict too — a 256k context + 8k num_predict
    # would still need KV for the worst-case prompt + output
    # which is what blew past the 16 GB budget.  Cap the predict
    # so the worst-case KV is bounded.
    if num_predict is not None:
        try:
            pred = int(num_predict)
        except (TypeError, ValueError):
            pred = BOUNDED_NUM_PREDICT
        bounded["num_predict"] = min(max(1, pred), BOUNDED_NUM_PREDICT)
    return bounded


def _build_ollama_payload(package: CleanPackage, *, default_num_ctx: int | None = None) -> dict[str, Any]:
    """Build the body for Ollama's native ``/api/chat`` endpoint.

    The transport accepts ``num_ctx`` either as a top-level key in
    ``package.options`` or under ``package.sampling``.  In every
    case the field is moved into ``options`` so the per-request
    bounded context is honoured.  The payload is then passed
    through the universal chokepoint ``_enforce_bounded_options``
    which clamps anything that would exceed the tier budget.
    """
    raw_options: dict[str, Any] = {}
    for source in (package.options, package.sampling):
        for key, value in source.items():
            if key in _OLLAMA_OPTION_KEYS:
                raw_options[key] = value
    if "num_ctx" not in raw_options and default_num_ctx is not None:
        raw_options["num_ctx"] = int(default_num_ctx)
    num_predict = raw_options.get("num_predict")
    options = _enforce_bounded_options(raw_options, num_predict=num_predict)
    if "num_ctx" not in options:
        raise RuntimeError("Ollama payload missing num_ctx; bounded context is mandatory in the clean slice")
    if int(options["num_ctx"]) <= 0:
        raise RuntimeError("Ollama payload num_ctx must be positive")
    payload: dict[str, Any] = {
        "model": package.model_id,
        "messages": package.messages,
        "stream": True,
    }
    if package.images:
        payload["messages"] = _attach_ollama_images(payload["messages"], package.images)
    # Per-model native-reasoning flag (model-card default). Sent at the
    # top level of the /api/chat body, which is where Ollama expects it.
    if package.think is not None:
        payload["think"] = bool(package.think)
    for key, value in package.sampling.items():
        if key not in _OLLAMA_OPTION_KEYS:
            payload[key] = value
    payload["options"] = options
    return payload


# ---------------------------------------------------------------------------
#  Ollama native transport
# ---------------------------------------------------------------------------

class CleanOllamaTransport:
    """Ollama worker transport that talks to the native ``/api/chat``.

    The transport is OpenAI-incompatible by design: it uses NDJSON
    streaming and ``options.num_ctx`` so the bounded context is
    honoured per call.  External OpenAI-shaped clients are served by
    the clean server, which wraps this transport's stream.
    """

    def __init__(self, url: str, api_key: str = "ollama", timeout: float = 900.0) -> None:
        self.base = str(url).rstrip("/").removesuffix("/v1")
        self._api_key = api_key
        self.timeout = float(timeout)
        # The default fallback context is always the bounded
        # value.  This is part of the universal chokepoint: if
        # a caller forgets to set a num_ctx on a package, the
        # default that ships is the budget-safe value, never
        # anything larger.
        self._default_num_ctx: int = BOUNDED_NUM_CTX
        # Active-stream state.  ``_close_requested`` is set
        # by ``close()`` to signal the active ``stream()``
        # generator to raise ``StreamAborted`` on its next
        # iteration.  ``_active_stream`` is set by ``stream()``
        # on entry and cleared on exit so ``close()`` can
        # find the right generator to flag.  The runtime's
        # starvation timer calls ``close()`` when the
        # first-token or slow-decode deadline fires, and
        # the resulting ``StreamAborted`` is translated
        # into the ``first_token_timeout`` / ``slow_decode_timeout``
        # error strings the retry ladder already knows.
        self._close_requested: bool = False
        self._active_stream: AsyncIterator[tuple[str, bool]] | None = None
        # The in-flight ``httpx.Response`` for the active
        # stream.  ``request_close()`` calls
        # ``response.aclose()`` on this so the
        # ``aiter_lines()`` loop unblocks even when no
        # chunk ever arrives.  This is the missing piece
        # v9 had: setting ``_close_requested`` alone does
        # nothing if the per-chunk loop is parked on the
        # socket read.
        self._active_response: httpx.Response | None = None
        self._template_cache: dict[str, str] = {}

    async def _response_prefix_prompt(self, package: CleanPackage) -> str:
        """Render Ollama's chat template, then append a true response prefix."""
        template = self._template_cache.get(package.model_id)
        if template is None:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base}/api/show",
                    headers=self._headers(),
                    json={"name": package.model_id},
                )
                response.raise_for_status()
                template = str(response.json().get("template") or "")
            if not template:
                raise RuntimeError(
                    f"Ollama model '{package.model_id}' has no chat template; "
                    "cannot apply response_prefix safely"
                )
            self._template_cache[package.model_id] = template
        environment = Environment()
        environment.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(RuntimeError(message))
        rendered = environment.from_string(template).render(
            messages=package.messages,
            add_generation_prompt=True,
            enable_thinking=package.think is not False,
            tools=None,
            add_vision_id=False,
        )
        return rendered + (package.response_prefix or "")

    def request_close(self) -> None:
        """Signal the active ``stream()`` generator to abort.

        Two things happen here, in order:

        1. ``_close_requested`` is set.  The
           ``stream()`` generator's per-line check
           sees this on the next ``aiter_lines()``
           iteration and raises ``StreamAborted``.
           This is the cooperative abort path.

        2. ``self._active_response.aclose()`` is
           scheduled via ``asyncio.create_task``.  This
           is the *forceful* abort path: it cancels the
           in-flight read so ``aiter_lines()`` unblocks
           with an exception even when no chunk has
           ever arrived.  Without this, a stream that
           is genuinely hung at the socket layer (Ollama
           accepting the request but never emitting a
           first line) cannot be interrupted: the
           per-chunk loop is parked on the network read
           and never gets to check ``_close_requested``.

        Both paths are necessary.  The cooperative one
        handles the case where chunks are arriving but
        the model is producing garbage; the forceful
        one handles the case where no chunks ever
        arrive (the "hang" condition the pathology
        monitor is designed to catch).
        """
        self._close_requested = True
        response = self._active_response
        if response is not None:
            try:
                # ``aclose()`` returns a coroutine; we
                # schedule it on the running loop without
                # awaiting so the call returns immediately
                # and the close happens asynchronously in
                # the background.  Wrapped in a try/except
                # because a closed loop or other transient
                # error should not raise from this path
                # (the close-flag path is still active).
                import asyncio
                asyncio.create_task(response.aclose())
            except Exception:
                # Best-effort close; the flag path is
                # still active as a fallback.
                pass

    def set_default_num_ctx(self, value: int) -> None:
        # Clamp the default to the bounded context.  Callers
        # that try to widen the default get the budget value
        # instead — a real 16 GB card cannot afford a 256k
        # context even for a warmup ``/api/chat`` ping.
        requested = int(value)
        if requested > BOUNDED_NUM_CTX:
            self._default_num_ctx = BOUNDED_NUM_CTX
        elif requested <= 0:
            raise RuntimeError("set_default_num_ctx must be positive; the clean slice refuses a non-positive default")
        else:
            self._default_num_ctx = requested

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def build_payload(self, package: CleanPackage) -> dict[str, Any]:
        return _build_ollama_payload(package, default_num_ctx=self._default_num_ctx)

    async def stream(
        self,
        package: CleanPackage,
        pathology_monitor: PathologyMonitor | None = None,
    ) -> AsyncIterator[tuple[str, bool]]:
        in_thinking = False
        # 2026-09-09 (user ruling): raw-prefill tolerance is a model-card
        # property. A model with raw_prefill=False (gemma4 lineage) does
        # NOT tolerate the raw rendered prompt+prefix stream -- its strict
        # server-side parser rejects it (the live 'unexpected .' backend
        # error). For such models the text prefill is DROPPED here and the
        # request takes the native /api/chat path, where think suppression
        # rides the model-card think flag.
        use_response_prefix = (
            package.response_prefix is not None
            and package.raw_prefill is not False
        )
        payload = self.build_payload(package)
        endpoint = "/api/chat"
        if use_response_prefix:
            if package.images:
                raise RuntimeError("Ollama response_prefix path does not support images yet")
            payload = {
                "model": package.model_id,
                "prompt": await self._response_prefix_prompt(package),
                "stream": True,
                "raw": True,
                "options": payload.get("options", {}),
            }
            if "think" in self.build_payload(package):
                payload["think"] = self.build_payload(package)["think"]
            endpoint = "/api/generate"
        # keep_alive policy: the package sets its own keep_alive
        # value.  The default (when the package doesn't set one)
        # is -1 ("stay loaded until a different model is needed").
        # Per-request overrides flow through metadata.keep_alive
        # in the request, which the artifact pipeline reads and
        # passes into the package via build_round{1,2}_package.
        payload.setdefault("options", {})
        payload["options"].setdefault("keep_alive", -1)
        # Mark this generator as the active stream so
        # ``request_close()`` can find it.  Clear the
        # close flag at entry so a stale flag from a
        # previous stream does not abort the new one.
        self._close_requested = False
        # Start the pathology monitor's clock at the
        # *determinism boundary*.  The moment we hand the
        # request to httpx is the last moment where our
        # process is fully in control: the bytes will be
        # written to the socket in the next few ms, the
        # body will be sent, Ollama will receive the
        # prompt and start prefill.  None of those steps
        # are under our control any more.  If the model
        # has not started emitting tokens by
        # ``budget`` seconds after this point, the model
        # (or Ollama) is hung and the right move is to
        # pull the plug and start over with a new seed.
        # We do NOT wait for httpx to receive a response
        # status line: that signal may never come if
        # Ollama itself is the thing that is hung, which
        # is exactly the condition we are guarding
        # against.
        if pathology_monitor is not None:
            pathology_monitor.start()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base}{endpoint}",
                    headers=self._headers(),
                    json=payload,
                ) as response:
                    # Save the response so ``request_close()``
                    # can ``aclose()`` it from outside this
                    # coroutine.  Without this, a hung stream
                    # cannot be interrupted: the per-chunk
                    # loop is parked on ``aiter_lines()`` and
                    # never sees ``_close_requested``.
                    self._active_response = response
                    try:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            # Check the close flag BEFORE parsing
                            # the line.  This is the abort path
                            # the runtime's starvation timer uses
                            # when the first-token or slow-decode
                            # deadline fires.  We raise
                            # ``StreamAborted`` (a regular
                            # ``Exception``) so the runtime's
                            # ``except Exception as exc:`` catches
                            # it without needing a separate handler.
                            if self._close_requested:
                                raise StreamAborted(
                                    f"stream aborted by request_close() "
                                    f"(model={package.model_id})"
                                )
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if obj.get("error"):
                                raise RuntimeError(f"Ollama /api/chat error: {obj['error']}")
                            msg = obj.get("message") or {}
                            # Emit every chunk the model sends in the
                            # exact order it sends them, with no
                            # category-based filtering or tagging.  Some
                            # models stream internal ``reasoning`` in
                            # ``message.thinking``; others stream it in
                            # ``message.content`` with ``<think>`` tag
                            # delimiters; either way we surface every
                            # byte of the response as the user explicitly
                            # requires.  The ``thinking`` boolean is
                            # retained only to keep the transport
                            # signature stable; it is always False.
                            text = (obj.get("response") if use_response_prefix else msg.get("content")) or ""
                            thinking_text = (obj.get("thinking") if use_response_prefix else msg.get("thinking")) or ""
                            if not text and not thinking_text and not obj.get("done"):
                                continue
                            if thinking_text:
                                for sub in _split_token_chunks(thinking_text):
                                    yield sub, True
                            if text:
                                for sub in _split_token_chunks(text):
                                    yield sub, False
                            if obj.get("done"):
                                return
                    finally:
                        # Always clear the response handle on
                        # stream exit so a subsequent call's
                        # ``request_close()`` cannot close
                        # the wrong response.  This runs
                        # whether the stream completed
                        # normally, raised an exception, or
                        # was closed by ``request_close()``.
                        self._active_response = None
        finally:
            # Defensive: if the transport's own exception
            # broke us out of the ``async with`` blocks
            # without going through the inner ``finally``,
            # still clear the handle.  Also, if a
            # pathology monitor was given, mark it
            # ``stopped`` so the runtime's background
            # timer knows the stream is over.
            self._active_response = None
            if pathology_monitor is not None:
                pathology_mark_stopped(pathology_monitor)

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base}/api/tags")
                return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def model_available(self, model_id: str) -> bool:
        """Return True if ``model_id`` is a registered, loadable model.

        PROVIDER-AGNOSTIC (2026-09-21).  The primary probe is the
        OpenAI-compatible ``GET <base>/v1/models`` listing -- the SAME
        contract ``backend_lifecycle`` and ``flm_lifecycle`` use to decide a
        backend is up.  That covers Ollama, llama.cpp's ``llama-server`` and
        LM Studio uniformly.

        WHY THIS CHANGED.  The probe used to POST Ollama's ``/api/show``
        unconditionally.  On a non-Ollama backend the backend probe passed
        while this one 404'd, so every specialist was silently substituted by
        the generalist -- a capability collapse hidden behind a green health
        check.  Ollama's ``/api/show`` is now only the FALLBACK, used when the
        listing endpoint yields nothing usable.

        Contract preserved: an unregistered model, an unknown listing shape,
        or any transport error returns False, so the caller's graceful
        generalist substitution is unchanged.
        """
        if not model_id:
            return False
        base = self.base.rstrip("/").removesuffix("/v1")
        # --- primary: the OpenAI-compatible listing -------------------------
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    f"{base}/v1/models",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
                if response.status_code == 200:
                    try:
                        payload = response.json()
                    except Exception:
                        payload = None
                    names = self._model_names_from_listing(payload)
                    if names:
                        return self._listing_has_model(names, model_id)
        except httpx.HTTPError:
            pass
        # --- fallback: Ollama's /api/show -----------------------------------
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    f"{self.base}/api/show",
                    json={"name": model_id},
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
                return response.status_code == 200
        except httpx.HTTPError:
            return False

    @staticmethod
    def _model_names_from_listing(payload: Any) -> set[str]:
        """Extract model ids from an OpenAI-style ``/v1/models`` payload.

        Tolerant by design: accepts ``{"data": [{"id": ...}]}`` (OpenAI /
        llama.cpp / LM Studio), ``{"models": [{"name"|"model"|"id": ...}]}``
        (Ollama's own ``/api/tags`` shape), and a bare list.  Returns an empty
        set when the shape is unrecognised, so the caller falls back rather
        than concluding "absent".
        """
        items: Any = None
        if isinstance(payload, dict):
            for key in ("data", "models"):
                if isinstance(payload.get(key), list):
                    items = payload[key]
                    break
        elif isinstance(payload, list):
            items = payload
        if not isinstance(items, list):
            return set()
        names: set[str] = set()
        for item in items:
            if isinstance(item, str):
                names.add(item)
            elif isinstance(item, dict):
                for key in ("id", "name", "model"):
                    value = item.get(key)
                    if isinstance(value, str) and value:
                        names.add(value)
                        break
        return names

    def _listing_has_model(self, names: set[str], model_id: str) -> bool:
        """True if ``model_id`` appears in a listing, tolerant of a trailing
        ``:latest`` and of an ``owner/name`` vs bare ``name`` difference.

        Deliberately NARROW: a real tag difference (``:q4_K_M`` vs ``:q8_0``)
        still means a different model and must not be conflated.
        """
        if model_id in names:
            return True
        want = model_id
        if want.endswith(":latest"):
            want = want[: -len(":latest")]
        for name in names:
            have = name
            if have.endswith(":latest"):
                have = have[: -len(":latest")]
            if have == want:
                return True
            # owner/name == name (e.g. 'JThomas-CoE/x' vs 'x')
            if have.split("/")[-1] == want.split("/")[-1]:
                return True
        return False

    async def registry_sizes(self) -> dict[str, float]:
        """Return ``{model_id: size_gb}`` from the backend's own listing.

        PROVIDER-AGNOSTIC and best-effort: used by the boot-time footprint
        verification.  Returns an EMPTY dict when the backend exposes no size
        information (most OpenAI-compatible listings do not), so the caller
        reports "cannot verify" instead of inventing a number.

        Ollama's ``/api/tags`` is tried first because it is the only listing
        here that carries a per-model ``size``; ``/v1/models`` is used only to
        enumerate names when ``size`` is absent.
        """
        base = self.base.rstrip("/").removesuffix("/v1")
        sizes: dict[str, float] = {}
        # Ollama /api/tags carries `size` per model.
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(f"{base}/api/tags")
                if response.status_code == 200:
                    for item in (response.json() or {}).get("models", []) or []:
                        name = item.get("name")
                        size = item.get("size")
                        if isinstance(name, str) and name and isinstance(size, (int, float)) and size > 0:
                            sizes[name] = float(size) / 1e9
        except Exception:
            pass
        if sizes:
            return sizes
        # Fallback: enumerate names only (no size information available).
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    f"{base}/v1/models",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
                if response.status_code == 200:
                    for name in self._model_names_from_listing(response.json()):
                        sizes.setdefault(name, 0.0)
        except Exception:
            pass
        return sizes

    async def non_streaming_call(
        self,
        *,
        model_id: str,
        messages: list[dict[str, Any]],
        num_ctx: int | None = None,
        num_predict: int | None = None,
        keep_alive: int | None = None,
        timeout: float | None = None,
        think: bool | None = None,
        sampling: dict[str, Any] | None = None,
        format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Single non-streaming Ollama call routed through the chokepoint.

        Used by the VRAM manager for warmup loads, evictions, and
        other lifecycle calls that must go through Ollama.  The
        payload is built via the same ``build_payload`` path as
        the streaming transport so the bounded-context guarantee
        is uniform across the entire clean slice.

        2026-09-11 (user-ruled): ``think`` passthrough added so the
        self-review reviewer call matches the model's configured think
        state (the 2026-09-11 empty-verdict failure: the think-capable
        web-r25 reviewer burned its whole num_predict budget inside its
        thinking span because the review payload carried no think flag).
        ``think=None`` leaves the payload untouched (model default),
        mirroring the streaming path's semantics.

        2026-09-12 (user-approved): ``sampling`` and ``format`` added for
        the self-review reviewer.

        * ``sampling`` merges caller-supplied sampling keys into the
          request options through a WHITELIST.  The whitelist is the
          point: without it, a caller could pass ``num_ctx`` or
          ``num_predict`` here and bypass the bounded-context chokepoint
          that every other Ollama request respects.  A reviewer must not
          sample (temperature 0.0, top_k 1, fixed seed), which is why
          this exists at all — previously the reviewer call passed no
          options and ran at Ollama's defaults (temperature 0.8).
        * ``format`` is Ollama's native structured-output parameter: a
          JSON schema the reply is constrained to.  It mirrors the
          ``think`` passthrough (``None`` leaves the payload untouched).
          A backend that cannot honour it raises an HTTP error rather
          than silently degrading — the self-review gate has no text
          fallback by design.
        """
        if num_ctx is None:
            num_ctx = self._default_num_ctx
        options: dict[str, Any] = {"num_ctx": int(num_ctx)}
        if num_predict is not None:
            options["num_predict"] = int(num_predict)
        if keep_alive is not None:
            options["keep_alive"] = int(keep_alive)
        # Sampling keys are merged BEFORE the bounded chokepoint, but only
        # from the whitelist — the chokepoint owns num_ctx/num_predict.
        if sampling:
            for _key, _value in sampling.items():
                if _key in _SAMPLING_OPTION_KEYS:
                    options[_key] = _value
        options = _enforce_bounded_options(options, num_predict=options.get("num_predict"))
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if keep_alive is not None:
            payload["keep_alive"] = int(keep_alive)
        if think is not None:
            payload["think"] = bool(think)
        if format is not None:
            payload["format"] = format
        async with httpx.AsyncClient(timeout=timeout or self.timeout) as client:
            response = await client.post(
                f"{self.base}/api/chat",
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()
        return response.json()


# ---------------------------------------------------------------------------
#  Router-side prompt shape tagger
# ---------------------------------------------------------------------------

# Build_prompt / SWE-bench templates embed blocks of unrelated content
# (full problem statements, target-file source, unified-diff examples).
# The classifier only needs to know whether such a block is present
# and how large it is -- it doesn't read the content.  Sending the
# full block to the NPU router confuses small models (qwen3.5:4b)
# because the embedded ``<patch>`` syntax in PATCH_EXAMPLE looks like
# content the classifier might want to emit.
#
# This helper replaces the four known SWE-bench / CoE template blocks
# with one-line shape tags.  Other prose passes through verbatim.
# The specialist/generalist models still receive the FULL prompt --
# only the router sees the shape-only version.

_ROUTER_SHAPE_TAGS = ("issue", "code", "patch", "readmes")
_ROUTER_SHAPE_RE = re.compile(
    r"<(" + "|".join(_ROUTER_SHAPE_TAGS) + r")>(.*?)</\1>",
    re.DOTALL,
)


def shape_only_prompt(prompt: str) -> str:
    """Reduce a prompt to its structural shape for router classification.

    Replaces ``<issue>...</issue>``, ``<code>...</code>``,
    ``<patch>...</patch>``, ``<readmes>...</readmes>`` blocks
    with one-line tags like ``[patch: 412 chars]`` so the router
    knows the shape of the prompt without seeing confusing content
    (e.g. unified-diff syntax in PATCH_EXAMPLE).  Other prose
    (the routing-signals block the router itself emits, plain
    user messages without template blocks, etc.) passes through
    unchanged.
    """
    def _sub(m: re.Match[str]) -> str:
        tag = m.group(1)
        inner = m.group(2)
        return f"[{tag}: {len(inner)} chars]"
    return _ROUTER_SHAPE_RE.sub(_sub, prompt)


# ---------------------------------------------------------------------------
#  FLM (NPU) router transport — uses the OpenAI-compatible endpoint
# ---------------------------------------------------------------------------

class CleanFlmRouterTransport:
    """OpenAI-compatible transport that targets the FLM NPU service."""

    def __init__(self, url: str, api_key: str = "flm", timeout: float = 60.0) -> None:
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _payload(self, package: CleanPackage) -> dict:
        options = dict(package.options)
        options.pop("num_ctx", None)
        options.pop("keep_alive", None)
        # Hoist num_predict to top-level max_tokens — FLM's
        # OpenAI-compatible endpoint ignores options.num_predict
        # and uses its default max.  This was a silent bug: every
        # FLM streaming call ran until the model's default limit,
        # not the configured num_predict.
        num_predict = options.pop("num_predict", None)
        messages = package.messages
        if package.images:
            messages = _attach_openai_images(messages, package.images)
        payload: dict = {
            "model": package.model_id,
            "messages": messages,
            "stream": True,
            **({"temperature": package.sampling["temperature"]} if "temperature" in package.sampling else {}),
            "options": options,
        }
        if num_predict is not None:
            payload["max_tokens"] = int(num_predict)
        if package.think is not None:
            payload["think"] = bool(package.think)
        return payload

    async def stream(
        self,
        package: CleanPackage,
        pathology_monitor: Any | None = None,
    ) -> AsyncIterator[tuple[str, bool]]:
        """Stream from the FLM endpoint.

        ``pathology_monitor`` is accepted (and ignored) so the unified
        tier-dispatch path can call every transport with the same
        signature; FLM attempts are short trivial/router calls and are
        not locally stall-gated.
        """
        in_thinking = False
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.url}/chat/completions",
                headers=self._headers(),
                json=self._payload(package),
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    text = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "") or ""
                    if "<think>" in text:
                        in_thinking = True
                        text = text.replace("<think>", "")
                    if "</think>" in text:
                        text = text.replace("</think>", "")
                        was_thinking = in_thinking
                        in_thinking = False
                        if text:
                            for sub in _split_token_chunks(text):
                                yield sub, was_thinking
                        continue
                    for sub in _split_token_chunks(text):
                        yield sub, in_thinking

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.url}/models", headers=self._headers())
                return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def non_streaming_call(
        self,
        *,
        model_id: str,
        messages: list[dict[str, Any]],
        num_predict: int | None = None,
        temperature: float = 0.0,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Single non-streaming FLM (OpenAI-compatible) call.

        Used by the router probe, which only needs a short draft
        response (~100 tokens) and does NOT need to stream.  A
        non-streaming request is faster (no SSE parsing) and allows
        an independent short timeout so a hung probe can't block
        the full transport timeout.

        ``max_tokens`` is at the top level (not inside ``options``)
        because FLM's OpenAI-compatible endpoint ignores
        ``options.num_predict`` — see ``_payload`` for the same fix
        on the streaming path.
        """
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
        }
        if num_predict is not None:
            payload["max_tokens"] = int(num_predict)
        async with httpx.AsyncClient(timeout=timeout or self.timeout) as client:
            response = await client.post(
                f"{self.url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()
        return response.json()


# ---------------------------------------------------------------------------
#  Cloud (Ollama cloud / OpenAI-compatible) transport — T3 escalation only
# ---------------------------------------------------------------------------

class CleanCloudTransport:
    """OpenAI-compatible transport for a cloud endpoint (e.g. Ollama cloud).

    Used ONLY for the T3 zero-shot escalation of last resort.  It is
    deliberately NOT wired into any specialist tier: the local ladder
    stays VRAM-bounded and offline.  The endpoint is OpenAI-compatible
    (``/v1/chat/completions``, SSE), which is what Ollama cloud and most
    hosted model APIs expose.

    ``num_ctx`` / ``keep_alive`` / Ollama-native ``options`` are stripped
    (cloud endpoints reject them); ``num_predict`` is hoisted to the
    OpenAI ``max_tokens`` field.  If the endpoint is unreachable or the
    key is invalid, ``stream()`` raises, and the runtime's T5 error path
    surfaces the failure (the local generalist is NOT retried — T5 is a
    one-shot last resort by design).
    """

    def __init__(self, url: str, api_key: str = "", timeout: float = 120.0) -> None:
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, package: CleanPackage) -> dict:
        options = dict(package.options)
        # Cloud endpoints reject Ollama-native options.
        options.pop("num_ctx", None)
        options.pop("keep_alive", None)
        options.pop("think", None)
        num_predict = options.pop("num_predict", None)
        messages = package.messages
        if package.images:
            messages = _attach_openai_images(messages, package.images)
        payload: dict = {
            "model": package.model_id,
            "messages": messages,
            "stream": True,
            **({"temperature": package.sampling["temperature"]} if "temperature" in package.sampling else {}),
        }
        if num_predict is not None:
            payload["max_tokens"] = int(num_predict)
        # 2026-09-09 (user ruling): cloud reasoning effort is a SEPARATE
        # config parameter from the local think on/off policy. The T3
        # package now carries a STRING level ("low"|"medium"|"high") in
        # .think which maps 1:1 to reasoning_effort. Booleans no longer
        # arrive here from the modify ladder (the config drives it), but
        # the legacy bool mapping is retained for any direct-cloud caller
        # that still passes one: True->high, False->low.
        think_cfg = package.think
        if think_cfg is not None:
            if isinstance(think_cfg, str):
                payload["reasoning_effort"] = think_cfg
            elif think_cfg is True:
                payload["reasoning_effort"] = "high"
            else:
                payload["reasoning_effort"] = "low"
        return payload

    async def stream(
        self,
        package: CleanPackage,
        pathology_monitor: Any | None = None,
    ) -> AsyncIterator[tuple[str, bool]]:
        """Stream from the cloud endpoint.

        ``pathology_monitor`` is accepted (and ignored) so the
        meta-harness invoker can call every transport with the same
        signature; cloud attempts are one-shot and are not locally
        stall-gated (the remote provider governs its own stream).
        """
        t_started = time.monotonic()
        n_lines = 0
        n_bytes = 0
        n_reasoning_chars = 0
        finish_reason: str | None = None
        t_last_line = t_started
        max_gap_s = 0.0
        in_thinking = False
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.url}/chat/completions",
                headers=self._headers(),
                json=self._payload(package),
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    now = time.monotonic()
                    gap = now - t_last_line
                    if gap > max_gap_s:
                        max_gap_s = gap
                    t_last_line = now
                    n_lines += 1
                    n_bytes += len(line) + 1
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        print(
                            f"[cloud-transport] done model={package.model_id} "
                            f"lines={n_lines} bytes={n_bytes} "
                            f"reasoning_chars={n_reasoning_chars} "
                            f"finish={finish_reason or 'unknown'} "
                            f"elapsed={now - t_started:.1f}s max_line_gap={max_gap_s:.1f}s",
                            file=sys.stderr, flush=True,
                        )
                        return
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    # D1 (2026-08-29): think/content split determined by
                    # STRUCTURED FIELDS, never by sniffing model output
                    # text. The previous " thinking"/" response" string
                    # matching was mojibake (never matched anything) and
                    # silently reclassified GLM's reasoning tokens as
                    # content -- a confound for all downstream rate and
                    # thinking analysis. OpenAI-reasoning-style endpoints
                    # stream reasoning in delta.reasoning /
                    # delta.reasoning_content; regular output in
                    # delta.content.
                    delta_obj = (chunk.get("choices") or [{}])[0].get("delta") or {}
                    reasoning = (
                        delta_obj.get("reasoning")
                        or delta_obj.get("reasoning_content")
                        or ""
                    )
                    text = delta_obj.get("content") or ""
                    # D1-hc (2026-08-29): capture finish_reason for stream
                    # health observability ('stop' = complete, 'length' =
                    # truncated by max_tokens). Never gates success.
                    fr = (chunk.get("choices") or [{}])[0].get("finish_reason")
                    if fr:
                        finish_reason = fr
                    if reasoning:
                        n_reasoning_chars += len(reasoning)
                        for sub in _split_token_chunks(reasoning):
                            yield sub, True
                    if text:
                        for sub in _split_token_chunks(text):
                            yield sub, False
        # Reached on stream end without [DONE] (connection dropped,
        # server closed early, generator torn down by cancellation).
        print(
            f"[cloud-transport] ABNORMAL end model={package.model_id} "
            f"lines={n_lines} bytes={n_bytes} "
            f"reasoning_chars={n_reasoning_chars} "
            f"elapsed={time.monotonic() - t_started:.1f}s max_line_gap={max_gap_s:.1f}s "
            f"(no [DONE] sentinel)",
            file=sys.stderr, flush=True,
        )

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.url}/models", headers=self._headers())
                return response.status_code == 200
        except httpx.HTTPError:
            return False


# ---------------------------------------------------------------------------
#  Heuristic-first router (heuristic, then model confirmation)
# ---------------------------------------------------------------------------
# NOTE (2026-09-21): a module-level ``LANGUAGE_KEYWORDS: dict[str, str]``
# keyword->domain map previously sat here.  It was DELETED, for two
# reasons, either independently sufficient:
#
#   1. It was DEAD.  An AST scan of the whole package found exactly one
#      occurrence of the name -- its own definition -- in both the dev and
#      release trees, and no ``globals()`` / ``getattr()`` / ``eval()``
#      dynamic access.  It was never read by anything.
#   2. It was a keyword->DOMAIN map, which is structurally the mechanism
#      this router must not contain: the heuristic is INFORMATIONAL ONLY
#      and must never set the domain (see ``_heuristic``, which returns
#      ``domain=""`` in every branch).  Domain classification belongs to
#      the routing model.  Had it ever been wired in, it would also have
#      assigned bare tags (``python``, ``web``, ``math``) rather than the
#      ``domain_*`` form, and it named domains with no ``domain_rows``
#      entry (humanities, chemistry, biology, law, business, economics).
#
# The live language signal is ``LANGUAGE_PATTERNS`` below, which feeds
# ``detected_languages`` for diagnostics and does not touch the domain.

# Per-language regex patterns.  Each entry maps a language
# bucket to a list of patterns that, when *any* of them
# match, is enough to commit that language at high
# confidence.  These are unambiguous code patterns — they
# cannot appear in non-coding prompts.  Patterns are
# evaluated left-to-right in declaration order; the first
# match wins.  Python gets checked first because ``def`` /
# ``import`` / ``class`` are the most reliable indicators
# in the LCB-style competitive programming corpus.
LANGUAGE_PATTERNS: tuple[tuple[str, tuple[re.Pattern[str], ...], float], ...] = (
    ("python",
     (
         re.compile(r"\bdef\s+\w+\s*\(", re.IGNORECASE),
         re.compile(r"\bimport\s+\w+\b|\bfrom\s+\w+\s+import\b", re.IGNORECASE),
         re.compile(r"\bclass\s+\w+\s*\(?\s*[A-Z]\w*\s*\)?\s*:", re.IGNORECASE),
         # Tagged python fence, e.g. ``\`\`\`python``.
         re.compile(r"```\s*(python|py)\b", re.IGNORECASE),
     ),
     0.95),
    ("web",
     (
         re.compile(r"<\s*(html|head|body|div|script|template)\b", re.IGNORECASE),
         re.compile(r"\bfunction\s+\w+\s*\(", re.IGNORECASE),
         re.compile(r"```\s*(html|css|js|javascript|typescript|ts)\b", re.IGNORECASE),
         # Prose mentions of web technologies
         re.compile(r"\b(?:html|css|javascript|js|typescript|ts)\s+(?:file|page|app|application|script|program|component)\b", re.IGNORECASE),
         re.compile(r"\b(?:single|self-contained|complete|standalone)\s+(?:html|javascript|typescript)\b", re.IGNORECASE),
     ),
     0.90),
    ("compiled_coding",
     (
         re.compile(r"#include\s*<\w+/?\w*>", re.IGNORECASE),
         re.compile(r"\bstd::\w+", re.IGNORECASE),
         re.compile(r"\bfn\s+\w+\s*\(", re.IGNORECASE),
         re.compile(r"\bpackage\s+\w+|\bfunc\s+\w+", re.IGNORECASE),
         re.compile(r"\bpublic\s+(static\s+)?(void|class|int|String)\b", re.IGNORECASE),
         re.compile(r"```\s*(cpp|c\+\+|c#|java|go|rust|rb|ruby|php|swift|kotlin|sql|bash|sh|matlab)\b", re.IGNORECASE),
     ),
     0.90),
)


# Heuristic "is this a coding prompt?" detector.  Returns
# True if any unambiguous code pattern matches anywhere in
# the prompt.  Used to set domain_confidence.
#
# The first ten are the same as the per-language patterns
# in ``LANGUAGE_PATTERNS`` (an explicit code block in the
# user's prompt).  The remaining patterns are
# competitive-programming / "write a function that ..."
# signals that appear in LCB-style prose-heavy problem
# statements.  These have lower *per-pattern* reliability
# but are decisive when combined with the imperative
# verb signals below.
_CODING_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Explicit code block markers
    re.compile(r"\bdef\s+\w+\s*\(", re.IGNORECASE),
    re.compile(r"\bimport\s+\w+\b|\bfrom\s+\w+\s+import\b", re.IGNORECASE),
    re.compile(r"\bclass\s+\w+\s*\(?\s*[A-Z]\w*\s*\)?\s*:", re.IGNORECASE),
    re.compile(r"<\s*(html|head|body|div|script|template)\b", re.IGNORECASE),
    re.compile(r"\bfunction\s+\w+\s*\(", re.IGNORECASE),
    re.compile(r"#include\s*<\w+/?\w*>", re.IGNORECASE),
    re.compile(r"\bstd::\w+", re.IGNORECASE),
    re.compile(r"\bfn\s+\w+\s*\(", re.IGNORECASE),
    re.compile(r"\bpackage\s+\w+|\bfunc\s+\w+", re.IGNORECASE),
    re.compile(r"\bpublic\s+(static\s+)?(void|class|int|String)\b", re.IGNORECASE),
    # LCB / competitive-programming signals
    re.compile(r"\b(Example|Input|Output|Constraints|Explanation)\s*:\s", re.IGNORECASE),
    re.compile(r"\b(nums|arr|s|grid|tree|root|head|node)\s*[\[\(]", re.IGNORECASE),
    re.compile(r"\boverflow\b|\bbitwise\b|\bO\(n\)\b|\bO\(n\s*log\s*n\)\b", re.IGNORECASE),
    re.compile(r"\breturn\s+(the\s+)?(maximum|minimum|count|sum|length|result|answer|index|value|number|product)\b", re.IGNORECASE),
    re.compile(r"\b(write|implement|create|define|complete)\s+(a\s+)?(function|method|class|program|algorithm|solution)\b", re.IGNORECASE),
    # Prose language mentions: "HTML file", "Python script", "JavaScript app", etc.
    re.compile(r"\b(?:html|css|javascript|js|typescript|ts|python|py|cpp|c\+\+|rust|go|java|kotlin|swift|ruby|php)\s+(?:file|page|app|application|script|program|component|module)\b", re.IGNORECASE),
    # "single HTML file", "self-contained HTML", etc.
    re.compile(r"\b(?:single|self-contained|complete|standalone)\s+(?:html|python|javascript|typescript)\b", re.IGNORECASE),
    # Tagged code fence (``\`\`\`python``, ``\`\`\`cpp``, ...).
    # LCB's "no starter code" branch always emits
    # ``\`\`\`python\n# YOUR CODE HERE\n\`\`\``, so the fence
    # tag alone is a near-certain signal of a coding task.
    re.compile(r"```\s*(python|py|cpp|c\+\+|c#|java|js|javascript|typescript|ts|go|rust|rb|ruby|php|swift|kotlin|sql|html|css|bash|sh|matlab)\b", re.IGNORECASE),
)


def detect_language(prompt: str) -> tuple[str, float] | tuple[None, float]:
    """Walk the LANGUAGE_PATTERNS table in order.  Return the
    first matching language and its confidence, or
    ``(None, 0.0)`` if no code pattern matches.
    """
    for language, patterns, confidence in LANGUAGE_PATTERNS:
        for pattern in patterns:
            if pattern.search(prompt):
                return language, confidence
    return None, 0.0


def is_coding_prompt(prompt: str) -> bool:
    """Return True if any unambiguous code pattern appears
    anywhere in the prompt.  Used by the heuristic to set
    ``domain_confidence`` to 0.95 (coding) or 0.1 (non-coding).
    """
    for pattern in _CODING_PATTERNS:
        if pattern.search(prompt):
            return True
    return False


class CleanHeuristicFirstRouter:
    """Model-primary router with heuristic context.

    The router is independent of the matrix. The matrix is a consumer
    of the router's work product. The router's internal mechanisms
    (probe, identity clarification, post-processing) never involve
    the matrix.
    """

    HEURISTIC_CONFIDENCE = 0.95

    PROBE_SYSTEM_PROMPT = (
        "You are a probe generator. Given a user request, produce "
        "a brief (100-200 token) response aimed at understanding the "
        "request's intent. Do not classify or route — just generate "
        "a short draft response that a classifier could use to "
        "determine the correct domain, task type, and task scope. "
        "Be concise and focused on the user's actual intent."
    )

    def __init__(
        self,
        flm_transport: CleanFlmRouterTransport | None,
        ollama_router_transport: CleanOllamaTransport,
        router_model_id: str,
        router_vision_capable: bool = False,
        vram_for_ollama: "object | None" = None,
        artifact_store: "object | None" = None,
        closed_domains: set[str] | None = None,
        closed_task_types_router_emit: set[str] | None = None,
        closed_task_types_reach: set[str] | None = None,
        closed_task_scopes: set[str] | None = None,
        # Step 5/6: per-domain semantic anchors (description +
        # include/exclude topics) injected into the router system
        # prompt so the classifier can ground its domain choice.
        # Mapping {domain_name: {"description": str,
        # "include_topics": [str], "exclude_topics": [str]}}.  Defaults
        # to {} — the router still works without anchors (they are
        # steering only).
        domain_anchors: dict[str, dict[str, list[str]]] | None = None,
        # Coding-domain set derived from the matrix (domains whose rows
        # declare a new_code_generation harness node).  Domains NOT in
        # this set may not emit new_code_generation or bug_fix; such
        # emissions are clamped to general_use at post-process.
        coding_domains: set[str] | None = None,
        # Router probe policy (from clean_resolution.router_probe).
        # Defaults applied internally when None.
        router_policy: dict[str, Any] | None = None,
    ) -> None:
        self.flm_transport = flm_transport
        self.ollama_transport = ollama_router_transport
        self.router_model_id = router_model_id
        self.router_vision_capable = bool(router_vision_capable)
        self.vram = vram_for_ollama
        self.artifact_store = artifact_store
        self._domain_anchors: dict[str, dict[str, list[str]]] = domain_anchors or {}
        # Coding-domain gate (user ruling 2026-09-04).  None = no gate
        # (router not matrix-wired); when wired, non-coding domains may
        # not emit new_code_generation / bug_fix.
        self._coding_domains: set[str] | None = coding_domains
        self._flm_known_up: bool | None = None
        self.last_router_source: str = "heuristic"
        # Vocabularies.  Without caller-supplied sets the router
        # still works — the canonical fallbacks match v12 behavior.
        # The fallbacks are the explicit, audited default of the
        # platform.  When the matrix is wired in, it overrides.
        # The 19-domain fallback is reserved for infrastructure
        # failures (matrix file unreadable).  The 5-, 6-, and
        # 4-value fallbacks are the platform invariants.
        self._closed_domains: set[str] = closed_domains or set(DOMAIN_FALLBACK)
        # _closed_task_types is the 5-value set the router emits.
        # The 6-value reach set is used for post-process validation
        # (so `user_defined` from a user override is accepted).
        self._closed_task_types: set[str] = closed_task_types_router_emit or set(ROUTER_EMIT_TASK_TYPES)
        self._closed_task_scopes: set[str] = closed_task_scopes or set(CANONICAL_TASK_SCOPES)
        # Override path: post-process validator uses the 6-value
        # reach set when merging user overrides.  Router emit
        # validation uses the 5-value emit set above.
        self._closed_task_types_reach: set[str] = closed_task_types_reach or set(CANONICAL_TASK_TYPES)
        # Router probe policy (user ruling 2026-09-04): per-field scoping,
        # config-owned. Defaults match clean_resolution.router_probe.
        rp = (router_policy or {})
        self._probe_lock_threshold = float(rp.get("lock_threshold", 0.80))
        self._probe_band_low = float(rp.get("band_low", 0.65))
        self._probe_override_min_gain = float(rp.get("override_min_gain", 0.05))
        self._probe_trivial_threshold = float(rp.get("trivial_threshold", 0.80))
        # Domain confidence floor (min_domain_confidence, config-owned).
        # Consumed in _post_process; previously dead config.
        self._domain_floor = float(rp.get("domain_floor", 0.80))

    def _build_system_prompt(self) -> str:
        """Render the router's system prompt from the runtime vocab sets.

        The prompt's `domain` list is sourced from the matrix
        (``clean_resolution.domain_rows`` keys) or the
        ``DOMAIN_FALLBACK`` set if the matrix is unreadable.  The
        ``task_type`` list is the 5-value router-emit set (never
        ``user_defined``; that value is reserved for user override
        only).  The ``task_scope`` list is the 4-value canonical.

        The prompt is rebuilt on every access.  The router fires
        once per request; the cost is negligible.
        """
        domain_list = ", ".join(sorted(self._closed_domains))
        task_type_list = ", ".join(sorted(self._closed_task_types))
        task_scope_list = ", ".join(sorted(self._closed_task_scopes))
        # Step 5/6: build per-domain semantic anchors (description +
        # include/exclude topics) injected as steering guidance.  These
        # are NOT gates — they nudge the classifier into the correct
        # semantic neighborhood; forced selection from the closed
        # domain set does the rest.  Only render domains that have any
        # anchor content, to keep the prompt tight.
        anchor_lines: list[str] = []
        for dom in sorted(self._closed_domains):
            a = self._domain_anchors.get(dom, {})
            desc = a.get("description", "")
            inc = a.get("include_topics", [])
            exc = a.get("exclude_topics", [])
            if not desc and not inc and not exc:
                continue
            bits = []
            if desc:
                bits.append(desc)
            if inc:
                bits.append("includes: " + ", ".join(inc))
            if exc:
                bits.append("NOT: " + ", ".join(exc))
            anchor_lines.append(f"  {dom} -> " + " | ".join(bits))
        anchor_block = ""
        if anchor_lines:
            anchor_block = (
                "\nDomain meanings (for grounding your choice; steer by "
                "these, then pick the single best domain from the closed "
                "list above):\n" + "\n".join(anchor_lines) + "\n\n"
            )
        return (
            "You are a routing classifier. Given a user request, output a "
            "single line of JSON with keys: domain, task_type, task_scope, "
            "trivial, domain_confidence, task_type_confidence, "
            "task_scope_confidence, trivial_confidence, user_intent.\n\n"
            "Allowed values:\n"
            f"  domain in {{{domain_list}}};\n"
            f"  task_type in {{{task_type_list}}};\n"
            f"  task_scope in {{{task_scope_list}}};\n"
            "  trivial is true or false;\n"
            "  confidences are floats in [0, 1].\n\n"
            + anchor_block
            + "The user_intent field is the model's concise assessment of the "
            "overall gist of the user's request. Summarize the central "
            "outcome the user wants, including the primary action and "
            "target when clear. Write it in your own words as a short "
            "verb-led phrase or one sentence.\n\n"
            "Do not quote the user, reproduce requirements as a checklist, "
            "explain your reasoning, justify the routing decision, mention "
            "heuristic signals, or infer goals the user did not express. "
            "Omit secondary implementation details when they are not "
            "necessary to convey the central intent. Keep it preferably "
            "5-20 words and no more than one sentence. Base it on the "
            "user's request itself, not merely on the existence of an "
            "artifact or the heuristic classification. Write the result "
            "directly into the user_intent JSON field. Do not add prose "
            "outside the JSON.\n\n"
            "Set any undetermined numeric field to 0.0. If the overall "
            "user intent cannot be inferred with reasonable confidence, "
            "set user_intent to \"\". Do not write anything other than the "
            "JSON.\n\n"
            "Select task_type by this decision procedure. The task type is "
            "a pure function of the USER'S INTENT as expressed in the "
            "prompt. The mere existence of artifacts in the store is NOT a "
            "signal — an artifact only matters when the prompt explicitly "
            "identifies an existing object, such as an existing, loaded, "
            "previous, or named artifact/file, or uses a modification or "
            "evaluation verb aimed at a concrete existing target. Ordinary "
            "pronouns in requirements, such as 'it must switch themes', "
            "and behavior verbs, such as 'update the task status', describe "
            "the requested output and are not artifact references.\n"
            "Apply this precedence before selecting task_type: first "
            "determine whether the request operates on an existing working "
            "artifact. If it does, any request to repair, correct, fix, "
            "change, or enhance that artifact is modify_artifact, even if "
            "the user calls the problem a bug or asks to fix it. This "
            "existing-artifact precedence overrides bug_fix. Use bug_fix "
            "only when no working artifact is referenced and the user has "
            "attached or pasted code with a specific error trace or bug.\n"
            "1. bug_fix — use ONLY when ALL of these hold: (a) there is NO "
            "referenced existing artifact, (b) a code block or code file is "
            "attached/pasted, AND (c) a specific error trace or bug is "
            "given, with or without an explicit 'fix' instruction. The "
            "code-block-plus-error-trace pair implies the intent is to fix "
            "the cause of the error. SPECIAL CASE: an error trace pasted "
            "ALONE (no code block) also implies fix-the-cause intent only "
            "when no existing working artifact is referenced; otherwise "
            "the request is modify_artifact, relative to the current "
            "artifact.\n"
            "2. modify_artifact — use when the prompt explicitly references "
            "an existing artifact or clearly asks to change a concrete "
            "target in that artifact, OR the user pastes/attaches code or "
            "an artifact and wants to change it, AND it is NOT the bug_fix "
            "pattern above (no code-block-plus-error-trace pair).\n"
            "3. evaluate_artifact — use when the prompt references an "
            "artifact (or attaches one) and the clear intent is "
            "explanation/analysis only (review, analyze, evaluate, check, "
            "explain) — not a change.\n"
            "4. new_code_generation — use when the prompt references no "
            "artifact, attaches no code, and the user wants something new "
            "written from scratch (write/create/build/make a new thing).\n"
            "5. general_use — use for anything that is not a coding or "
            "artifact task.\n\n"
            "HARD LINE: an error trace pasted alone implies fix-the-cause "
            "intent (relative to the most recent output). A code block "
            "PLUS an error trace routes to bug_fix. Otherwise, a "
            "referenced/attached artifact routes to modify_artifact or "
            "evaluate_artifact. A from-scratch request with no reference "
            "and no attachment routes to new_code_generation.\n\n"
            "Set trivial=true ONLY if you have very high confidence the "
            "prompt is simple and can be answered in 50 tokens or fewer "
            "without thinking. When in doubt, set trivial=false and route "
            "to a specialist. For an image attachment, trivial=true is "
            "allowed only for a simple image description or identification "
            "request; visual analysis, OCR, comparison, debugging, and "
            "image-based coding are not trivial.\n\n"
            "If the prompt references a working artifact but you cannot "
            "determine which artifact to good or substantial confidence, "
            "emit task_type=general_use and trivial=false; the server will "
            "initiate an identity clarification side-trip.\n\n"
            "ATTACHED IMAGES: an image is CONTEXT, never an artifact. "
            "Describing, explaining, or analyzing an attached image is "
            "task_type=general_use (keep the domain). Artifact tasks "
            "(modify_artifact / evaluate_artifact / bug_fix) require a "
            "TEXT artifact or code attachment -- never route a task to "
            "them because the prompt says \"the attached image\"."
        )

    @property
    def SYSTEM_PROMPT(self) -> str:
        return self._build_system_prompt()

    async def classify(
        self,
        prompt: str,
        *,
        attachments: list[AttachmentMeta] | None = None,
        hints: dict | None = None,
        artifact_section_names: list[str] | None = None,
        prior_turn_produced_artifact: bool = False,
        routing_override: Any | None = None,
        artifact_identity_resolution: str | None = None,
    ) -> CleanRouterDecision:
        """Classify a prompt into (domain, task_type, task_scope)."""

        # --- routing_override handling ---
        # A routing_override pins the domain (and optionally the
        # task_type).  It must NOT collapse an unspecified task_type
        # to "general_use" -- that is conceptually wrong: overriding
        # the domain should not force the task type.  So:
        #   * If the override specifies BOTH domain and task_type,
        #     short-circuit to _synthesize_override (fully pinned).
        #   * If the override specifies ONLY the domain, run the
        #     normal classification (heuristic + model) to determine
        #     the task_type, then pin the domain on the result.
        override_domain = getattr(routing_override, "domain", None) if routing_override else None
        override_task_type = getattr(routing_override, "task_type", None) if routing_override else None
        if routing_override and override_task_type:
            return self._synthesize_override(routing_override, prompt)
        if routing_override and override_domain and not override_task_type:
            # Pin the domain, classify the task_type normally.
            decision = await self._classify_task_type(
                prompt, attachments=attachments, hints=hints,
                artifact_section_names=artifact_section_names,
                prior_turn_produced_artifact=prior_turn_produced_artifact,
                artifact_identity_resolution=artifact_identity_resolution,
            )
            decision = self._apply_confidence_gates(decision)
            return replace(decision, domain=override_domain, domain_confidence=1.0)

        # --- Artifact identity resolution (from clarification side-trip) ---
        if artifact_identity_resolution is not None:
            decision = await self._classify_task_type(
                prompt, attachments=attachments, hints=hints,
                artifact_section_names=artifact_section_names,
                prior_turn_produced_artifact=prior_turn_produced_artifact,
                artifact_identity_resolution=artifact_identity_resolution,
            )
            decision = self._apply_confidence_gates(decision)
            return decision

        # --- Normal path: heuristic + model ---
        decision = await self._classify_task_type(
            prompt, attachments=attachments, hints=hints,
            artifact_section_names=artifact_section_names,
            prior_turn_produced_artifact=prior_turn_produced_artifact,
            artifact_identity_resolution=None,
        )

        # --- Image-task policy (2026-09-07, user-approved) ---
        # Rule: an attached IMAGE is context, never an artifact.  When
        # the ONLY attachment is an image, an artifact-oriented task
        # verdict can only be the classifier misreading "the attached
        # image" as an artifact reference (the artifact store is
        # text-only).  Gate keys on the ATTACHMENTS of THIS request;
        # the stale artifact_store selection never counts, and the
        # override paths above return before this block (user-explicit
        # routing always wins).
        decision = self._apply_image_task_policy(decision, attachments)

        # Explicit attach wins: a text artifact attached THIS TURN makes
        # the referent unambiguous (see _needs_clarification docstring).
        _explicit_artifact = bool(attachments) and any(
            a.kind == "working_artifact" for a in attachments
        )
        if self._needs_clarification(decision, artifact_section_names,
                                     explicit_artifact_attached=_explicit_artifact):
            raise NeedsClarification(
                question="Which artifact did you mean?",
                candidates=self._artifact_candidates(),
            )

        return decision

    async def _classify_task_type(
        self,
        prompt: str,
        *,
        attachments: list[AttachmentMeta] | None = None,
        hints: dict | None = None,
        artifact_section_names: list[str] | None = None,
        prior_turn_produced_artifact: bool = False,
        artifact_identity_resolution: str | None = None,
    ) -> CleanRouterDecision:
        """Run the normal heuristic + model classification (no override)."""
        if artifact_identity_resolution is not None:
            if artifact_identity_resolution == "none":
                heuristic = self._heuristic(
                    prompt, hints=hints, attachments=attachments,
                    artifact_section_names=None,
                    prior_turn_produced_artifact=False,
                )
                resolved_sections = None
            else:
                resolved_sections = (
                    self.artifact_store.get_section_names()
                    if self.artifact_store and self.artifact_store.current
                    else None
                )
                heuristic = self._heuristic(
                    prompt, hints=hints, attachments=attachments,
                    artifact_section_names=resolved_sections,
                    prior_turn_produced_artifact=True,
                )
        else:
            # Normal path: the model only sees artifact context when the
            # prompt actually REFERENCES an artifact.  The mere existence
            # of artifacts in the store is NOT a routing signal -- a
            # from-scratch prompt must not be biased toward
            # modify_artifact just because an artifact happens to exist.
            # Reference detection (name / deictic / modify-evaluate verb)
            # gates whether artifact section names are surfaced to the
            # classifier.
            _refs = self._prompt_references_artifact(prompt)
            resolved_sections = artifact_section_names if _refs else None
            heuristic = self._heuristic(
                prompt,
                hints=hints,
                attachments=attachments,
                artifact_section_names=resolved_sections,
                prior_turn_produced_artifact=prior_turn_produced_artifact and _refs,
            )
        trivial_allowed = self.trivial_eligible(attachments or [])

        # Gate the prior-artifact signal the same way as the
        # heuristic: only tell the model about a prior artifact when
        # the prompt actually references one.  Without this, the
        # model would see "Prior turn produced a working artifact" in
        # the user content even for a from-scratch prompt, biasing it
        # toward modify_artifact.
        _effective_prior_artifact = prior_turn_produced_artifact
        if artifact_identity_resolution is None:
            _effective_prior_artifact = prior_turn_produced_artifact and _refs

        decision = await self._run_model(
            prompt, heuristic, attachments, resolved_sections,
            _effective_prior_artifact, trivial_allowed,
        )

        if self._needs_probe(decision):
            decision = await self._run_probe(prompt, decision)

        decision = self._post_process(decision)
        decision = self._apply_confidence_gates(decision)

        if decision.trivial and not trivial_allowed:
            print(
                "[router] trivial decision vetoed by attachment policy "
                "(text/artifact attachment or image height > 960/unknown); "
                "routing normally",
                file=sys.stderr,
                flush=True,
            )
            decision = replace(decision, trivial=False)

        return decision

    def is_vision_capable(self) -> bool:
        """Return whether the configured router model accepts images."""
        return self.router_vision_capable

    def _apply_confidence_gates(self, decision: CleanRouterDecision) -> CleanRouterDecision:
        """Apply the config-owned confidence gates to a decision.

        Called from EVERY classify() return path (normal, override-
        pinned, artifact-resolution) so no path can bypass them:
          * domain floor (min_domain_confidence): a domain below the
            floor falls to domain_other.
          * trivial threshold: trivial=true iff trivial_confidence >
            threshold (user ruling 2026-09-04); trivial is never
            probe-eligible and has no other knob.
        """
        domain_floor = self._domain_floor
        if (decision.domain != "domain_other"
                and decision.domain_confidence < domain_floor):
            print(
                f"[router] domain floor: {decision.domain} at "
                f"{decision.domain_confidence:.2f} < {domain_floor:.2f} "
                f"-> domain_other",
                file=sys.stderr,
                flush=True,
            )
            decision = replace(decision, domain="domain_other")
        if decision.trivial and not (
            decision.trivial_confidence > self._probe_trivial_threshold
        ):
            print(
                f"[router] trivial downgraded: confidence "
                f"{decision.trivial_confidence:.2f} <= "
                f"{self._probe_trivial_threshold:.2f} -> non-trivial",
                file=sys.stderr,
                flush=True,
            )
            decision = replace(decision, trivial=False)
        return decision

    # 2026-09-03 unification: answer_trivial() was DELETED. The trivial
    # answer is no longer the router class's private pipeline: the server
    # composes the config-declared trivial_answer harness through
    # CleanHarness.compose() and dispatches via the executor's tier
    # machinery, with the configured router_fallback as fallback.

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _synthesize_override(self, override: Any, prompt: str) -> CleanRouterDecision:
        domain = getattr(override, "domain", None) or "domain_other"
        if domain not in self._closed_domains:
            domain = "domain_other"
        task_type = getattr(override, "task_type", None) or "general_use"
        if task_type not in self._closed_task_types:
            task_type = "general_use"
        return CleanRouterDecision(
            domain=domain, task_type=task_type,
            task_scope="single_domain_single_task",
            trivial=False, detected_languages=[], probe_invoked=False,
            prompt=prompt,
            domain_confidence=1.0, task_type_confidence=1.0,
            task_scope_confidence=1.0, trivial_confidence=1.0,
            confidence=1.0,
        )

    def trivial_eligible(self, attachments: list[AttachmentMeta]) -> bool:
        """Attachment-policy gate for the trivial-answer path.

        This is the SINGLE source of eligibility: pure attachment
        policy (no text/working-artifact attachments; every image
        must have a known native height <= 960 px). Prompt shape and
        length are deliberately NOT gates here — the router model is
        the only judge of whether a request is trivially answerable.
        """
        for a in attachments:
            if a.kind in ("text", "working_artifact"):
                return False
            if a.kind == "image" and (a.height is None or a.height > 960):
                return False
        return True

    async def _run_model(
        self, prompt: str, heuristic: CleanRouterDecision,
        attachments: list[AttachmentMeta] | None,
        artifact_section_names: list[str] | None,
        prior_turn_produced_artifact: bool,
        trivial_allowed: bool,
    ) -> CleanRouterDecision:
        """Run the router model and return only a model decision."""
        pkg = self._router_package(
            prompt, heuristic,
            attachments=attachments,
            artifact_section_names=artifact_section_names,
            prior_turn_produced_artifact=prior_turn_produced_artifact,
            trivial_allowed=trivial_allowed,
        )
        failure_reasons: list[str] = []
        flm_in_use = self.flm_transport is not None and self._flm_known_up is not False
        if flm_in_use:
            try:
                text = await self._stream_through(self.flm_transport, pkg)
                decision = self._parse(text)
                if decision is not None:
                    self._flm_known_up = True
                    self.last_router_source = "flm"
                    # Observability (Step 5/6): log the raw router JSON
                    # so the model's own restated user_intent line is
                    # visible (it is otherwise only consumed silently
                    # for descriptor derivation).
                    print(f"[router] FLM raw: {text!r}", flush=True)
                    print(f"[router] FLM parsed user_intent={decision.user_intent!r}", flush=True)
                    return replace(decision, prompt=prompt)
                print(f"[router] FLM parse failed on text: {text[:200]!r}", flush=True)
                self._flm_known_up = False
                failure_reasons.append("FLM returned an invalid classification")
            except Exception as exc:
                print(f"[router] FLM call failed: {type(exc).__name__}: {exc}", flush=True)
                self._flm_known_up = False
                failure_reasons.append(f"FLM {type(exc).__name__}: {exc}")
        if self.ollama_transport is not None:
            try:
                text = await self._stream_through(self.ollama_transport, pkg)
                decision = self._parse(text)
                if decision is not None:
                    self.last_router_source = "ollama"
                    # Observability: same raw JSON + user_intent logging
                    # for the Ollama router fallback path.
                    print(f"[router] Ollama raw: {text!r}", flush=True)
                    print(f"[router] Ollama parsed user_intent={decision.user_intent!r}", flush=True)
                    return replace(decision, prompt=prompt)
                print(f"[router] Ollama parse failed on text: {text[:200]!r}", flush=True)
                failure_reasons.append("Ollama returned an invalid classification")
            except Exception as exc:
                print(f"[router] Ollama call failed: {type(exc).__name__}: {exc}", flush=True)
                failure_reasons.append(f"Ollama {type(exc).__name__}: {exc}")
            self.last_router_source = "router_failure"
            reason = "; ".join(failure_reasons) or "No router model transport is available"
            raise RouterFailure("model_unavailable", reason)

    def _needs_probe(self, decision: CleanRouterDecision) -> bool:
        """Probe fires ONLY when at least one field is probe-eligible.

        Probe-eligible band (config: clean_resolution.router_probe):
        band_low <= confidence < lock_threshold.  Fields at or above
        lock_threshold are locked; fields below band_low fall to their
        configured defaults — neither group is worth a probe.
        """
        for conf in (decision.domain_confidence,
                     decision.task_type_confidence,
                     decision.task_scope_confidence):
            if self._probe_band_low <= conf < self._probe_lock_threshold:
                return True
        return False

    def _merge_probe_result(self, original: CleanRouterDecision,
                            reeval: CleanRouterDecision) -> CleanRouterDecision:
        """Merge a probe re-evaluation into the original decision.

        Per-field scoping (user ruling 2026-09-04):
          * Fields at or above lock_threshold NEVER change.
          * Fields in the probe-eligible band change only when the
            re-eval's confidence for that field is >= original +
            override_min_gain.
          * Fields below band_low are NOT probe-eligible: they fall to
            their configured defaults (domain_other / general_use /
            SDST) — the probe does not rescue them.
        """
        lock = self._probe_lock_threshold
        low = self._probe_band_low
        gain = self._probe_override_min_gain
        merged = original
        notes: list[str] = []

        # --- domain ---
        oc, rc = original.domain_confidence, reeval.domain_confidence
        if oc < low:
            # Sub-band original: falls to default domain_other. The
            # probe does not rescue it (checked BEFORE re-eval value).
            if original.domain != "domain_other":
                notes.append(f"domain: {original.domain}({oc:.2f}) -> domain_other [below band]")
            merged = replace(merged, domain="domain_other", domain_confidence=oc)
        elif rc >= lock:
            if (reeval.domain != original.domain
                    and (oc < lock or rc > oc)):
                notes.append(f"domain: {original.domain}({oc:.2f}) -> {reeval.domain}({rc:.2f}) [re-eval locked]")
                merged = replace(merged, domain=reeval.domain, domain_confidence=rc)
            elif reeval.domain != original.domain:
                notes.append(f"domain: original {original.domain}({oc:.2f}) stands (re-eval equal/locked)")
        elif low <= oc < lock:
            # Probe-eligible: adopt the re-eval value only on a
            # sufficient confidence gain.
            if (reeval.domain != original.domain
                    and rc >= oc + gain
                    and reeval.domain_confidence < lock):
                notes.append(f"domain: {original.domain}({oc:.2f}) -> {reeval.domain}({rc:.2f}) [gain {rc - oc:.2f}]")
                merged = replace(merged, domain=reeval.domain, domain_confidence=rc)
            elif rc >= lock and reeval.domain != original.domain:
                # Re-eval names a different domain at >= lock: adopt it.
                notes.append(f"domain: {original.domain}({oc:.2f}) -> {reeval.domain}({rc:.2f}) [re-eval locked]")
                merged = replace(merged, domain=reeval.domain, domain_confidence=rc)
            else:
                notes.append(f"domain: original {original.domain}({oc:.2f}) stands")

        # --- task_type ---
        oc, rc = original.task_type_confidence, reeval.task_type_confidence
        if oc < low:
            if original.task_type != "general_use":
                notes.append(f"task_type: {original.task_type}({oc:.2f}) -> general_use [below band]")
            merged = replace(merged, task_type="general_use", task_type_confidence=oc)
        elif rc >= lock:
            if (reeval.task_type != original.task_type
                    and (oc < lock or rc > oc)):
                notes.append(f"task_type: {original.task_type}({oc:.2f}) -> {reeval.task_type}({rc:.2f}) [re-eval locked]")
                merged = replace(merged, task_type=reeval.task_type, task_type_confidence=rc)
            elif reeval.task_type != original.task_type:
                notes.append(f"task_type: original {original.task_type}({oc:.2f}) stands (re-eval equal/locked)")
        elif low <= oc < lock:
            if (reeval.task_type != original.task_type
                    and rc >= oc + gain
                    and rc < lock):
                notes.append(f"task_type: {original.task_type}({oc:.2f}) -> {reeval.task_type}({rc:.2f}) [gain {rc - oc:.2f}]")
                merged = replace(merged, task_type=reeval.task_type, task_type_confidence=rc)
            elif rc >= lock and reeval.task_type != original.task_type:
                notes.append(f"task_type: {original.task_type}({oc:.2f}) -> {reeval.task_type}({rc:.2f}) [re-eval locked]")
                merged = replace(merged, task_type=reeval.task_type, task_type_confidence=rc)
            else:
                notes.append(f"task_type: original {original.task_type}({oc:.2f}) stands")

        # --- task_scope ---
        oc, rc = original.task_scope_confidence, reeval.task_scope_confidence
        if oc < low:
            if original.task_scope != "single_domain_single_task":
                notes.append(f"task_scope: {original.task_scope}({oc:.2f}) -> SDST [below band]")
            merged = replace(merged, task_scope="single_domain_single_task", task_scope_confidence=oc)
        elif rc >= lock:
            if (reeval.task_scope != original.task_scope
                    and (oc < lock or rc > oc)):
                notes.append(f"task_scope: {original.task_scope}({oc:.2f}) -> {reeval.task_scope}({rc:.2f}) [re-eval locked]")
                merged = replace(merged, task_scope=reeval.task_scope, task_scope_confidence=rc)
            else:
                notes.append(f"task_scope: original {original.task_scope}({oc:.2f}) stands")
        elif low <= oc < lock:
            if (reeval.task_scope != original.task_scope
                    and rc >= oc + gain
                    and rc < lock):
                notes.append(f"task_scope: {original.task_scope}({oc:.2f}) -> {reeval.task_scope}({rc:.2f}) [gain {rc - oc:.2f}]")
                merged = replace(merged, task_scope=reeval.task_scope, task_scope_confidence=rc)
            elif rc >= lock and reeval.task_scope != original.task_scope:
                notes.append(f"task_scope: {original.task_scope}({oc:.2f}) -> {reeval.task_scope}({rc:.2f}) [re-eval locked]")
                merged = replace(merged, task_scope=reeval.task_scope, task_scope_confidence=rc)
            else:
                notes.append(f"task_scope: original {original.task_scope}({oc:.2f}) stands")

        for note in notes:
            print(f"[router] probe merge: {note}", flush=True)
        return merged

    async def _run_probe(self, prompt: str, decision: CleanRouterDecision) -> CleanRouterDecision:
        transport = self.flm_transport if self.flm_transport is not None else self.ollama_transport
        if transport is None:
            return replace(decision, probe_invoked=True)
        print(f"[router] probe invoked (re-classify for domain certainty)", flush=True)
        # Probe generation: a brief (~100 token) draft response that
        # helps the classifier understand the prompt's intent.  This
        # is NOT a classification call — it's a trial generation whose
        # output is fed back to the router model alongside the
        # original prompt for re-classification.  100 tokens and a
        # 15s timeout are sufficient: the probe only needs to reveal
        # the domain intent, not produce a full answer.
        probe_messages = [
            {"role": "system", "content": self.PROBE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            if hasattr(transport, "non_streaming_call"):
                resp = await transport.non_streaming_call(
                    model_id=self.router_model_id,
                    messages=probe_messages,
                    num_predict=100,
                    temperature=0.0,
                    timeout=15.0,
                )
                # Normalize response format: FLM (OpenAI-compatible)
                # returns choices[0].message.content; Ollama returns
                # message.content.  Handle both.
                if isinstance(resp, dict):
                    choices = resp.get("choices")
                    if choices:
                        probe_output = (
                            choices[0].get("message", {}).get("content", "")
                            or ""
                        )
                    else:
                        probe_output = (
                            resp.get("message", {}).get("content", "")
                            or ""
                        )
                else:
                    probe_output = ""
            else:
                # Fallback: stream if non_streaming_call is unavailable.
                probe_pkg = CleanPackage(
                    model_id=self.router_model_id,
                    provider="flm" if self.flm_transport is not None else "ollama",
                    messages=probe_messages,
                    sampling={"temperature": 0.0, "stream": True},
                    options={"num_ctx": 4096, "num_predict": 100, "keep_alive": -1},
                    diagnostics={"role": "probe"},
                )
                parts: list[str] = []
                async for chunk_text, is_thinking in transport.stream(probe_pkg):
                    if not is_thinking:
                        parts.append(chunk_text)
                probe_output = "".join(parts)
        except Exception as exc:
            print(f"[router] probe failed: {type(exc).__name__}: {exc}", flush=True)
            return replace(decision, probe_invoked=True)
        # Re-run classifier with probe output as additional context
        reeval_pkg = self._router_package_with_probe(prompt, decision, probe_output)
        try:
            text = await self._stream_through(transport, reeval_pkg)
            # Observability (Step 5/6): log the probe re-evaluation's raw
            # JSON so the FINAL domain decision (post-probe) is visible,
            # not just the pre-probe first pass.
            print(f"[router] probe re-eval raw: {text!r}", flush=True)
            reeval = self._parse(text)
            if reeval is not None:
                print(f"[router] probe re-eval parsed domain={reeval.domain!r} "
                      f"user_intent={reeval.user_intent!r}", flush=True)
                # Phase B (user ruling 2026-09-04): per-field merge with
                # lock thresholds and gain rule — the re-eval no longer
                # wholesale-replaces the original decision.
                merged = self._merge_probe_result(decision, reeval)
                return replace(merged, probe_invoked=True, prompt=prompt)
        except Exception:
            pass
        return replace(decision, probe_invoked=True)

    def _router_package_with_probe(self, prompt: str, decision: CleanRouterDecision, probe_output: str) -> CleanPackage:
        # Router sees the shape of the prompt (no embedded
        # ``<patch>`` syntax).  Specialist/generalist models
        # still get the full prompt -- only the classifier
        # uses the shape-only view.  See shape_only_prompt()
        # for the rationale.
        router_prompt = shape_only_prompt(prompt)
        user_content = (
            f"Previous classification (you may override):\n"
            f"  domain={decision.domain} (conf={decision.domain_confidence:.2f})\n"
            f"  task_type={decision.task_type} (conf={decision.task_type_confidence:.2f})\n"
            f"  task_scope={decision.task_scope} (conf={decision.task_scope_confidence:.2f})\n\n"
            f"Probe output (additional context):\n{probe_output}\n\n"
            f"User request:\n{router_prompt}"
        )
        return CleanPackage(
            model_id=self.router_model_id,
            provider="flm" if self.flm_transport is not None else "ollama",
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            sampling={"temperature": 0.0, "stream": True},
            options={"num_ctx": 4096, "num_predict": 600, "keep_alive": -1},
            diagnostics={"role": "clean_router_reeval"},
        )

    def _post_process(self, decision: CleanRouterDecision) -> CleanRouterDecision:
        domain = decision.domain
        if domain not in self._closed_domains:
            # Domain fall-through (2026-09-21).  The closed set is derived from
            # the matrix's `domain_rows` keys, so a domain the router can name
            # but the matrix does not configure (e.g. `domain_math` before a
            # math row exists) lands here.  Behaviour is unchanged -- it falls
            # to `domain_other`, exactly as before -- but the event is now
            # STATED rather than silently absorbed, so a later audit can read a
            # domain-use distribution from the log.  The raw router JSON
            # (`[router] FLM raw: ...`) already carries the model's un-coerced
            # domain; this line records that a coercion happened and to what.
            # Print-only: it cannot affect the returned decision.
            print(
                f"[router] domain fall-through: {decision.domain!r} not in closed set "
                f"({len(self._closed_domains)} domains) -> domain_other "
                f"(not configured)",
                file=sys.stderr,
                flush=True,
            )
            domain = "domain_other"
        task_type = decision.task_type
        if task_type not in self._closed_task_types:
            task_type = "general_use"
        # Coding-domain gate (user ruling 2026-09-04): non-coding
        # domains (matrix-derived) may not emit new_code_generation or
        # bug_fix; those emissions clamp to general_use.  The matrix
        # keeps its own default-harness fallback as a second line of
        # defense; modify_artifact / evaluate_artifact / general_use
        # remain reachable for all domains.
        if (
            self._coding_domains is not None
            and domain not in self._coding_domains
            and task_type in ("new_code_generation", "bug_fix")
        ):
            print(
                f"[router] task_type gate: {task_type} not reachable for "
                f"non-coding domain '{domain}' -> general_use",
                file=sys.stderr,
                flush=True,
            )
            task_type = "general_use"
        task_scope = decision.task_scope
        if task_scope not in self._closed_task_scopes:
            task_scope = "single_domain_single_task"
        trivial = decision.trivial if isinstance(decision.trivial, bool) else False
        return replace(decision, domain=domain, task_type=task_type, task_scope=task_scope, trivial=trivial)

    def _prompt_references_artifact(self, prompt: str) -> bool:
        """Detect whether the prompt references an existing artifact.

        An artifact is "in play" only when the prompt explicitly
        references one: by an explicit existing-object phrase (for
        example, "the existing file" or "this artifact"), by a
        modification/evaluation verb aimed at an existing object or
        concrete artifact target, or by a filename reference.
        The mere existence of artifacts in the store is NOT a
        reference — a "write a new kanban app" prompt references
        nothing, even though "kanban" matches an artifact name.
        Ordinary pronouns and action words inside a from-scratch
        specification (for example, "it must" or "update the status")
        are requirements for the requested output, not artifact
        references.
        """
        if not prompt:
            return False
        low = prompt.lower()
        # Explicit existing-object references.  A bare "it", "this", or
        # "that" is deliberately insufficient: generation specifications
        # routinely use those words for the requested output.
        if re.search(
            r"\b(?:the\s+(?:existing|original|previous|prior|current|attached|loaded|given|provided)\s+"
            r"(?:artifact|file|code|document|app|application|page|html|css|javascript|script)|"
            r"(?:this|that)\s+(?:artifact|file|code|document|app|application|page|html|css|javascript|script)|"
            r"(?:loaded|attached|previous|current)\s+(?:artifact|file|code|document|app|application|page|html|css|javascript|script)|"
            r"\w[\w.-]*\.(?:html|py|js|ts|css|jsx|tsx|cpp|c|rs|go|java|kt|scala))\b",
            low, re.IGNORECASE,
        ):
            return True
        # A modification/evaluation verb counts only when it is aimed at
        # a concrete artifact target.  This preserves requests such as
        # "change the delete button" while excluding "filters update
        # immediately" and "it must switch themes".
        if re.search(
            r"\b(?:modify|change|update|edit|refactor|rewrite|improve|review|analyze|evaluate|check|fix)\s+"
            r"(?:this|that|it|the\s+(?:(?:existing|original|previous|prior|current|attached|loaded|given|provided)\s+)?"
            r"(?:(?:\w+\s+){0,4})(?:artifact|file|code|document|app|application|page|html|css|javascript|script|button|control|rule|style|bug|error|exception))\b",
            low, re.IGNORECASE,
        ):
            return True
        return False

    def _apply_image_task_policy(
        self,
        decision: CleanRouterDecision,
        attachments: list[AttachmentMeta] | None,
    ) -> CleanRouterDecision:
        """Image-attachment task policy (2026-09-07, user-approved).

        Line: an image itself is PROHIBITED as an artifact, but it is
        legitimate CONTEXT (e.g. an example screenshot guiding an HTML
        modification -- the V13 flow attaches artifact + image together
        and is untouched by this policy).

        * image-only + evaluate_artifact  -> general_use (domain kept):
          describing/explaining an image is a general visual task; the
          tree-sitter/BM25 evaluate machinery would run against a stale
          text artifact.
        * image-only + bug_fix            -> HARD STOP: "bug_fix is text
          only at this time" (no code attached to patch).
        * image-only + modify_artifact    -> HARD STOP: "image to image
          and text to image is not supported at this time".  Left
          uncorrected this would silently modify the STALE selected
          artifact; hard-stopping wastes none of the user's time.
        * image + text artifact attached  -> unchanged (context case).
        * routing_override paths          -> unchanged (never reach here;
          user-explicit wins, and a future img2img harness wires in via
          config without touching this code).
        """
        if not attachments:
            return decision
        has_image = any(a.kind == "image" for a in attachments)
        has_text = any(
            a.kind in ("text", "working_artifact") for a in attachments
        )
        if not has_image or has_text:
            return decision  # no image, or image-as-context: normal routing
        task = decision.task_type
        if task == "evaluate_artifact":
            print(
                "[router] image-only attachment: evaluate_artifact -> "
                f"general_use (domain {decision.domain} retained); an "
                "image is context, never an artifact",
                file=sys.stderr, flush=True,
            )
            return replace(decision, task_type="general_use")
        if task == "bug_fix":
            raise RouterTaskUnsupported(
                "bug_fix is text only at this time",
                task_type=task,
            )
        if task == "modify_artifact":
            raise RouterTaskUnsupported(
                "image to image and text to image is not supported at this time",
                task_type=task,
            )
        return decision

    def _needs_clarification(self, decision: CleanRouterDecision, artifact_section_names: list[str] | None,
                             explicit_artifact_attached: bool = False) -> bool:
        # Clarification is a USER side-channel for genuine identity
        # ambiguity.  It fires ONLY when the prompt actually references
        # an artifact (by name / deictic / modify-evaluate verb) but the
        # reference does not uniquely resolve to a single artifact.  The
        # mere existence of artifacts is NOT a reason to clarify -- a
        # from-scratch prompt that happens to mention a word matching an
        # artifact name must not trigger a spurious "which artifact?"
        # question.
        #
        # Explicit attach wins (2026-09-07): when the request carries a
        # text artifact attachment THIS TURN, the referent is that file
        # by definition -- the server has already made it the store's
        # `current`.  Name collisions with old artifacts (whose
        # descriptors embed past prompt phrases like "the attached")
        # must not hijack the turn into "Which artifact did you mean?".
        if explicit_artifact_attached:
            return False
        if decision.task_type not in ("modify_artifact", "evaluate_artifact", "bug_fix"):
            return False
        prompt = decision.prompt or ""
        if not self._prompt_references_artifact(prompt):
            return False
        # The prompt references an artifact.  Clarify only if the
        # reference is ambiguous (multiple name matches) or the model
        # is unsure which artifact / task type.
        if self.artifact_store is not None:
            matches = self.artifact_store.resolve_by_name(prompt)
            if len(matches) > 1:
                return True
            if self.artifact_store.current is None and self.artifact_store.artifact_candidates():
                return True
        if decision.task_type_confidence < 0.7:
            return True
        return False

    def _artifact_candidates(self) -> list[str]:
        """Return human-readable artifact labels for the clarification
        side-channel, so the user can pick meaningfully (not bare IDs).

        Each label is ``<descriptor> (<source_filename>)`` when a
        filename exists, else ``<descriptor>``, else the artifact id.
        The current artifact is listed first.
        """
        if not self.artifact_store:
            return []
        candidates = self.artifact_store.artifact_candidates()
        labels: list[str] = []
        for candidate in candidates:
            descriptor = candidate.get("descriptor") or ""
            source_filename = candidate.get("source_filename")
            if descriptor:
                label = descriptor
                if source_filename:
                    label += f" ({source_filename})"
            elif source_filename:
                label = source_filename
            else:
                label = candidate["artifact_id"]
            label += (
                f" [session {candidate.get('session_number', 0)}, "
                f"version {candidate.get('version', 0)}; "
                f"id {candidate['artifact_id']}]"
            )
            labels.append(label)
        return labels

    # ------------------------------------------------------------------
    # Router package builder
    # ------------------------------------------------------------------

    def _router_package(
        self,
        prompt: str,
        heuristic: CleanRouterDecision,
        *,
        attachments: list[AttachmentMeta] | None = None,
        artifact_section_names: list[str] | None = None,
        prior_turn_produced_artifact: bool = False,
        trivial_allowed: bool = False,
    ) -> CleanPackage:
        # Router sees the shape of the prompt (no embedded
        # ``<patch>`` syntax).  Specialist/generalist models
        # still get the full prompt -- only the classifier
        # uses the shape-only view.  See shape_only_prompt()
        # for the rationale.
        router_prompt = shape_only_prompt(prompt)
        user_content = (
            f"Non-authoritative routing context (heuristic):\n"
            f"  Use these observations as context only; decide domain and task_type independently from the user's request.\n"
            f"  domain={heuristic.domain} (conf={heuristic.domain_confidence:.2f})\n"
            f"  task_type={heuristic.task_type} (conf={heuristic.task_type_confidence:.2f})\n"
            f"  task_scope={heuristic.task_scope} (conf={heuristic.task_scope_confidence:.2f})\n"
            f"  trivial_allowed={trivial_allowed}\n"
            f"  detected_languages={heuristic.detected_languages}\n\n"
            f"User request:\n{router_prompt}"
        )
        if artifact_section_names:
            user_content += (
                f"\n\nExisting artifact sections:\n"
                + "\n".join(f"  - {name}" for name in artifact_section_names)
            )
        if attachments:
            user_content += "\n\nAttachments:\n"
            for a in attachments:
                user_content += (
                    f"  - kind={a.kind}, mime={a.mime}, "
                    f"language={a.language}, size={a.size_bytes}, "
                    f"width={a.width}, height={a.height}\n"
                )
        if prior_turn_produced_artifact:
            user_content += "\nPrior turn produced a working artifact.\n"
        return CleanPackage(
            model_id=self.router_model_id,
            provider="flm" if self.flm_transport is not None else "ollama",
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            sampling={"temperature": 0.0, "stream": True},
            options={"num_ctx": 4096, "num_predict": 600, "keep_alive": -1},
            diagnostics={"role": "clean_router"},
        )

    # ------------------------------------------------------------------
    # Heuristic (4-bifurcation)
    # ------------------------------------------------------------------

    def _heuristic(
        self,
        prompt: str,
        *,
        hints: dict | None = None,
        attachments: list[AttachmentMeta] | None = None,
        artifact_section_names: list[str] | None = None,
        prior_turn_produced_artifact: bool = False,
    ) -> CleanRouterDecision:
        coding = is_coding_prompt(prompt)

        # Bifurcation 2: working-artifact-referenced vs denovo.
        # Use the same _prompt_references_artifact() method that
        # _classify_task_type uses, so the detection is consistent
        # across the heuristic and the model path.  The inline regex
        # that was here previously was a simpler, less precise version
        # that could false-positive on "the function" etc.
        artifact_referenced = False
        if prior_turn_produced_artifact or (artifact_section_names and len(artifact_section_names) > 0):
            artifact_referenced = self._prompt_references_artifact(prompt)

        # A pasted/attached code object (text attachment with a code
        # language, or a working_artifact) counts as "code present" for
        # routing, but is NOT a pre-existing returned artifact.  It
        # drives the bug_fix / modify_artifact / new_code_generation
        # intent decision below.
        code_attachment = False
        if attachments:
            for a in attachments:
                if a.kind in ("text", "working_artifact") and a.language:
                    code_attachment = True
                    break

        # Detected languages (informational, all matches)
        detected_languages: list[str] = []
        if coding:
            for language, patterns, _ in LANGUAGE_PATTERNS:
                for pattern in patterns:
                    if pattern.search(prompt):
                        if language not in detected_languages:
                            detected_languages.append(language)

        # Emit decision based on intent + code presence.
        # The domain is intentionally left EMPTY ("") with zero
        # confidence in every branch: the heuristic is only ever
        # informational and must never anchor the model to a
        # domain.  In particular, an attached/working artifact's
        # source language must NOT drive the domain -- the user
        # may want a cross-language conversion (e.g. HTML -> React)
        # where the TARGET domain is what matters.  Leaving the
        # domain to the model also avoids the "domain_other at 0.4"
        # signal that previously forced an unnecessary probe.
        #
        # The heuristic fires ONLY on decisive, unambiguous intent
        # signals.  Ambiguous cases are left to the model classifier
        # (and, if needed, the user clarification side-channel) -- the
        # heuristic must not try to cover every edge case.  A neutral
        # low-confidence signal is emitted when no decisive signal is
        # found, so the model decides.
        task_type: str | None = None
        conf = 0.0
        low = prompt.lower()
        # Decisive: explicit from-scratch generation ("write/build/make
        # a new X") with no artifact reference and no attached code.
        if re.search(r"\b(?:write|create|build|generate|make)\b", low, re.IGNORECASE) and not artifact_referenced and not code_attachment:
            task_type = "new_code_generation"
            conf = 0.9
        # Decisive: explicit review/analyze/evaluate of a referenced
        # artifact.
        elif artifact_referenced and re.search(r"\b(?:review|analyze|evaluate|check|look\s+at)\b", low, re.IGNORECASE):
            task_type = "evaluate_artifact"
            conf = 0.9
        # Decisive: explicit modify/change/update of a referenced
        # artifact.
        elif artifact_referenced and re.search(r"\b(?:modify|change|update|edit|refactor|rewrite|improve)\b", low, re.IGNORECASE):
            task_type = "modify_artifact"
            conf = 0.9
        # Decisive: code block + error trace (bug_fix pattern).
        elif code_attachment and re.search(r"\b(?:error|traceback|exception|bug|fails?|broken|wrong)\b", low, re.IGNORECASE):
            task_type = "bug_fix"
            conf = 0.9
        # No decisive signal: defer to the model with a neutral,
        # low-confidence task_type.
        if task_type is None:
            task_type = "general_use"
            conf = 0.1
        return CleanRouterDecision(
            domain="",
            task_type=task_type,
            task_scope="single_domain_single_task",
            trivial=False,
            detected_languages=detected_languages,
            prompt=prompt,
            domain_confidence=0.0,
            task_type_confidence=conf,
            task_scope_confidence=0.0,
            trivial_confidence=0.0,
        )

    # ------------------------------------------------------------------
    # Stream + parse helpers
    # ------------------------------------------------------------------

    async def _stream_through(self, transport, package: CleanPackage) -> str:
        text_parts: list[str] = []
        async for chunk, thinking in transport.stream(package):
            if thinking:
                continue
            text_parts.append(chunk)
        return "".join(text_parts).strip()

    def _parse(self, raw: str) -> CleanRouterDecision | None:
        candidate = raw
        if "{" in raw and "}" in raw:
            candidate = raw[raw.find("{"): raw.rfind("}") + 1]
        obj: dict | None = None
        try:
            obj = json.loads(candidate)
        except Exception:
            try:
                obj = self._parse_lenient(candidate)
            except Exception:
                # FLM may emit the router's free-text user_intent quotes
                # escaped at the JSON envelope level.  Repair only that
                # field's opening/closing quotes; valid JSON already
                # returned above and arbitrary malformed fields stay rejected.
                escaped_intent = re.sub(
                    r'("user_intent"\s*:\s*)\\"', r'\1"', candidate,
                )
                escaped_intent = re.sub(
                    r'(\S)\\"(\s*[,}])', r'\1"\2', escaped_intent,
                )
                if escaped_intent != candidate:
                    try:
                        obj = json.loads(escaped_intent)
                    except Exception:
                        obj = None
                if isinstance(obj, dict):
                    pass
                elif not candidate.lstrip().startswith("{"):
                    return None
                else:
                    # FLM can terminate a tiny classifier response after the
                    # final scalar while omitting only the closing object
                    # brace. Recover that bounded envelope; never invent
                    # missing field values or repair arbitrary malformed JSON.
                    missing_braces = candidate.count("{") - candidate.count("}")
                    if missing_braces != 1:
                        return None
                    try:
                        obj = json.loads(candidate.rstrip() + "}")
                    except Exception:
                        return None
        if not isinstance(obj, dict):
            return None
        try:
            domain = str(obj.get("domain", "domain_other"))
            task_type = str(obj.get("task_type", "general_use"))
            task_scope = str(obj.get("task_scope", "single_domain_single_task"))
            trivial = bool(obj.get("trivial", False))
            legacy = float(obj.get("confidence", 0.0))
            dc = float(obj.get("domain_confidence", legacy))
            tc = float(obj.get("task_type_confidence", legacy))
            tsc = float(obj.get("task_scope_confidence", legacy))
            trc = float(obj.get("trivial_confidence", legacy))
            user_intent = str(obj.get("user_intent", ""))
            return CleanRouterDecision(
                domain=domain,
                task_type=task_type,
                task_scope=task_scope,
                trivial=trivial,
                prompt="",  # set by caller via replace()
                domain_confidence=dc,
                task_type_confidence=tc,
                task_scope_confidence=tsc,
                trivial_confidence=trc,
                user_intent=user_intent,
                confidence=legacy,
            )
        except Exception:
            return None

    @staticmethod
    def _parse_lenient(candidate: str) -> dict | None:
        enum_keys = ("domain", "task_type", "task_scope", "trivial")
        fixed = candidate
        for key in enum_keys:
            pattern = re.compile(
                rf'("{key}"\s*:\s*)([A-Za-z_][A-Za-z0-9_]*|"[^"]*"|null|None|true|True|false|False)'
            )
            def _quote(m: re.Match[str]) -> str:
                v = m.group(2)
                if v in ("null", "None"):
                    return m.group(1) + "null"
                if v in ("true", "True"):
                    return m.group(1) + "true"
                if v in ("false", "False"):
                    return m.group(1) + "false"
                if v.startswith('"') and v.endswith('"'):
                    return m.group(1) + v
                return m.group(1) + f'"{v}"'
            fixed = pattern.sub(_quote, fixed)
        return json.loads(fixed)

# Note: ``CleanKeywordRouter`` was the pre-2026-07 deterministic
# fallback.  It has been removed in favor of the heuristic-first
# path inside ``CleanHeuristicFirstRouter._heuristic`` /
# ``detect_language`` / ``is_coding_prompt``.
