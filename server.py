"""Clean-only FastAPI composition root — unified model API.

The server presents an OpenAI-compatible chat completions endpoint
that acts as a single unified model.  Internally it:

1. Classifies each prompt via the heuristic-first router (FLM/NPU
   if available, Ollama fallback)
2. Resolves the classification into a harness + model pair
3. Executes via the clean runtime with V2 session memory injection
4. Returns the response in OpenAI chat completion format

The client (bench runner, IDE, user) sends prompts and receives
answers.  Routing, model swap, harness selection, session memory,
VRAM management, and NPU offload are all internal — the client
never knows which model ran or whether memory was injected.

Optional ``metadata`` fields in the request:
* (corrected 2026-10-12: ``num_ctx`` was listed here as a context-window override; no code reads ``metadata.num_ctx``; the context window travels per-harness in the sampling block)
* ``closed_think`` — force closed-think block on/off (null = harness default)
* ``clear_session`` — clear session memory before this request
* ``no_inject`` — disable session memory injection for this request

NOTE (2026-09-19): a ``timeout_s`` per-problem wall-clock cap was documented
here but has NEVER been plumbed — no code in this module reads it, and runtime.py
states explicitly that "the runtime never has a hard wall-clock timeout".  The
documentation was therefore removed rather than left promising a cap that does
not exist.  The REAL bounds on a running turn are:

  * per-attempt, local : ``first_token_timeout_s`` (60s, no tokens) and
    ``slow_decode_timeout_s`` (90s, quiet after tokens), owned per harness
    under ``escalation_ladder.*.runtime.timers``;
  * per-attempt, cloud : ``silence_timeout_s`` (240s) and ``total_timeout_s``
    (600s) under the cloud block;
  * client-facing      : the SSE heartbeat / idle watchdog in this module.

There is deliberately no whole-turn cap; see README "Turn progress and
cancellation" for how a long turn is observed and cancelled.
* ``cloud_mode: "direct"`` — user-initiated direct T3 cloud request; skips
    local specialist execution and T5, while automatic post-ladder T3 remains
    governed by the harness configuration

Run with::

    uvicorn server:app --port 8001

[FLAGGED STALE as of 2026-09-21 -- superseded, retained deliberately.]
SUPERSEDED TEXT (was line 44 in this docstring):

    uvicorn CoE_agent.clean_slice.clean_server:app --port 8001

That line named the PRE-CLEAN-SLICE module (`clean_slice.clean_server`),
which has not been the server entrypoint since the package moved to
`CoE_coder_16GB/server.py`.  It was wrong in every tree.  The corrected
form is `server:app` -- the module basename IS `server`, so `server:app`
resolves under any package name (the package itself is not part of the
module path).
"""
from __future__ import annotations

import asyncio
import base64
import binascii
from copy import deepcopy
import json as _json
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .config import load_config
from .contracts import (
    AttachmentMeta,
    CleanControlDecision,
    CleanProgressCallback,
    CleanProgressEvent,
    NeedsClarification,
    RouterFailure,
    RouterTaskUnsupported,
)
from .executor import CleanExecutor
from .resolution import CleanResolution
from .session_memory import SessionMemory
from .flm_lifecycle import ensure_flm
from .transport import (
    CleanCloudTransport,
    CleanFlmRouterTransport,
    CleanHeuristicFirstRouter,
    CleanOllamaTransport,
)
from .vram import CleanVramManager, verify_footprints
from .backend_lifecycle import ensure_backend
from .artifact_store import ArtifactStore, _slugify, build_diff_response, save_artifact_to_disk
from .evaluate_store import EvaluateObjectStore


# Map an artifact's language tag to the CoE domain that should own it.
# When a working artifact is present, its language (derived from the
# file extension at load time) pins the router's domain -- an HTML
# artifact is always domain_web, a Python artifact domain_python, etc.
# This prevents the router from misclassifying the domain based on
# prompt wording (e.g. a descriptive "fix the drag and drop" prompt
# must not route an HTML artifact to domain_python).
_ARTIFACT_LANG_TO_DOMAIN: dict[str, str] = {
    "html": "domain_web",
    "web": "domain_web",
    "javascript": "domain_web",
    "typescript": "domain_web",
    "css": "domain_web",
    "jsx": "domain_web",
    "tsx": "domain_web",
    "python": "domain_python",
    "py": "domain_python",
    "cpp": "domain_compiled",
    "c": "domain_compiled",
    "rust": "domain_compiled",
    "go": "domain_compiled",
    "java": "domain_compiled",
    "kotlin": "domain_compiled",
    "scala": "domain_compiled",
}


def _artifact_domain_override(artifact_language: str | None) -> str | None:
    """Derive a domain override from an artifact's language tag.

    Returns ``None`` when the language is unknown or not a coding
    language (the router classifies the domain normally).
    """
    if not artifact_language:
        return None
    return _ARTIFACT_LANG_TO_DOMAIN.get(artifact_language.lower())


# 2026-09-28 (user ruling): the image -> general_use HARNESS SWAP was
# REMOVED, and the gate that drove it (``_should_use_general_image_harness``)
# was DELETED with it, so no dead caller can be resurrected by accident.
#
# Ruling: "domain plus task type route to a harness but image attachment
# should be able to be processed by any of the specialist models because
# they should all be vision capable."
#
# The swap did NOT change the model -- the model always came from the domain
# row (``resolve()`` sets ``primary_model`` from ``task_cell`` -> row).  What
# it DISCARDED was the harness that domain+task_type had selected, costing:
#   * the harness contract (e.g. ``new_code_generation``'s code-fence
#     extraction of the generated artifact),
#   * its persona + system suffix + output instruction,
#   * the task cell's think-state ordering (``think_on`` / ``assistant_prefill``),
#   * and the ENTIRE escalation ladder -- ``general_use`` is t0-only, so
#     T1a/T1b/T2 and ``meta_harness_ladder`` (cloud_sota + generalist_zero_shot)
#     were unreachable on ANY request carrying an image.
#
# The correct guard for images is the vision gate in ``chat_completions``:
#   ``if images and image_target_model not in state.vision_models:
#     return _emit_vision_unavailable_response(...)``
# which refuses a non-vision selected model LOUDLY instead of silently
# re-harnessing the request.  A vision-capable specialist now keeps its
# harness.  NOTE: ``general_use`` itself is unchanged and remains a valid
# harness for domains that select it.
#
# Reversal: restore the helper above, then in ``chat_completions`` restore
#   ``if _should_use_general_image_harness(images, decision): work = replace(
#       work, harness=state.resolution.compose_harness_variant(
#           "general_use", domain=work.domain))``
# and re-gate the three attached-context sites (see the notes there).


def _extract_code_block(prompt: str) -> str | None:
    """Extract the submitted file content from a prompt's
    ``<code>...</code>`` block, if present.

    The SWE-bench / bug_fix harness embeds the target file's source in
    a ``<code>...</code>`` block so the model can patch it.  This
    returns that content verbatim (minus the ``<code>``/``</code>``
    tags) so the runtime's ``unified_diff_patch`` contract can enforce
    the context-correctness gate: a patch may only remove code that
    actually appears in this block.

    Returns ``None`` when the prompt has no ``<code>`` block (e.g. a
    bare issue with no file attached) — the gate stays inactive and
    the legacy take-last behavior applies.
    """
    if not prompt:
        return None
    m = re.search(r"<code>(.*?)</code>", prompt, re.DOTALL)
    if not m:
        return None
    return m.group(1)


MAX_IMAGE_BYTES = 10 * 1024 * 1024
_IMAGE_MIME_RE = re.compile(r"^[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+$")


# =====================================================================
# 2026-09-16 (S1): LEGACY DEVIATION MARKER (PLAN_legacy_restore.md)
# =====================================================================
# Names every KNOWN, REVIEWED divergence that the origin-prompt work left in
# the LEGACY modify route.  Reported on `coe_routing.legacy_deviations` ONLY
# when legacy_route is true, so a capability-gate run can never silently
# measure modified infrastructure.
#
# These three entries are ACCEPTED by ruling, not defects -- each is
# route-independent and therefore belongs to a shared layer rather than to
# the legacy local ladder (PLAN_legacy_restore.md section 0):
#   1. clarification resolver  -- accepts more answer forms (shared input fix)
#   2. T3 cloud layer          -- carries no repetition loop guard (ONE cloud)
#   3. reintegration machinery -- writes header + side-car (shared by design)
#
# Anything NOT on this list that changes legacy behaviour is an unreviewed
# deviation and must be either fixed or added here deliberately.
_LEGACY_ACCEPTED_DEVIATIONS: list[str] = [
    "clarification resolver accepts additional answer forms (shared fix)",
    "T3 cloud layer runs shortstop only, no repetition loop guard (one cloud layer)",
    "reintegration machinery writes provenance header + side-car (shared by design)",
]


def _image_url_value(value: Any) -> str:
    """Extract an image URL from OpenAI string or object input."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(value.get("url", "")).strip()
    return str(getattr(value, "url", "") or "").strip()


def _canonical_data_uri(data: bytes, mime: str | None) -> str | None:
    """Return bounded bytes as a canonical base64 data URI."""
    if not data or len(data) > MAX_IMAGE_BYTES:
        return None
    normalized_mime = (mime or "").split(";", 1)[0].strip().lower()
    if not normalized_mime.startswith("image/") or not _IMAGE_MIME_RE.fullmatch(normalized_mime):
        return None
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{normalized_mime};base64,{encoded}"


def _image_dimensions(data: bytes) -> tuple[int, int] | None:
    """Read native dimensions from common image headers without rescaling."""
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        return (width, height) if width > 0 and height > 0 else None
    if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
        width = int.from_bytes(data[6:8], "little")
        height = int.from_bytes(data[8:10], "little")
        return (width, height) if width > 0 and height > 0 else None
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP" and len(data) >= 30:
        if data[12:16] == b"VP8X":
            width = 1 + int.from_bytes(data[24:27], "little")
            height = 1 + int.from_bytes(data[27:30], "little")
            return (width, height) if width > 0 and height > 0 else None
        if data[12:16] == b"VP8L" and len(data) >= 25 and data[20] == 0x2F:
            width = 1 + ((data[22] & 0x3F) << 8 | data[21])
            height = 1 + ((data[24] & 0x0F) << 10 | data[23] << 2 | data[22] >> 6)
            return (width, height) if width > 0 and height > 0 else None
        if data[12:16] == b"VP8 ":
            frame_start = data.find(b"\x9D\x01\x2A", 20)
            if frame_start >= 0 and frame_start + 7 <= len(data):
                width = int.from_bytes(data[frame_start + 3:frame_start + 5], "little") & 0x3FFF
                height = int.from_bytes(data[frame_start + 5:frame_start + 7], "little") & 0x3FFF
                return (width, height) if width > 0 and height > 0 else None
    if data.startswith(b"\xff\xd8"):
        offset = 2
        sof_markers = {
            0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
            0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
        }
        while offset + 4 <= len(data):
            if data[offset] != 0xFF:
                offset += 1
                continue
            while offset < len(data) and data[offset] == 0xFF:
                offset += 1
            if offset >= len(data):
                break
            marker = data[offset]
            offset += 1
            if marker in (0xD8, 0xD9, 0xDA):
                break
            segment_length = int.from_bytes(data[offset:offset + 2], "big")
            if segment_length < 2 or offset + segment_length > len(data):
                break
            if marker in sof_markers and segment_length >= 7:
                height = int.from_bytes(data[offset + 3:offset + 5], "big")
                width = int.from_bytes(data[offset + 5:offset + 7], "big")
                return (width, height) if width > 0 and height > 0 else None
            offset += segment_length
    return None


def _canonical_image_details(data_uri: str) -> tuple[str, int, int | None, int | None]:
    """Return MIME, byte count, and native dimensions for a data URI."""
    header, separator, encoded = data_uri.partition(",")
    if not separator:
        return "", 0, None, None
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        return "", 0, None, None
    mime = header[5:].split(";", 1)[0].strip().lower() if header.startswith("data:") else ""
    dimensions = _image_dimensions(data)
    return mime, len(data), dimensions[0] if dimensions else None, dimensions[1] if dimensions else None


async def _normalize_image_reference(value: Any, mime_hint: str | None = None) -> str | None:
    """Normalize a data URI, raw base64 value, or bounded public URL."""
    reference = _image_url_value(value)
    if not reference:
        return None
    if reference.startswith("data:"):
        header, separator, encoded = reference.partition(",")
        if not separator or ";base64" not in header.lower():
            return None
        mime = header[5:].split(";", 1)[0].strip().lower()
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            return None
        return _canonical_data_uri(data, mime)
    parsed = urlparse(reference)
    if parsed.scheme in ("http", "https") and parsed.netloc:
        import httpx
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(10.0, connect=5.0),
                follow_redirects=False,
            ) as client:
                async with client.stream("GET", reference) as response:
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", mime_hint or "")
                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > MAX_IMAGE_BYTES:
                            return None
                        chunks.append(chunk)
                    return _canonical_data_uri(b"".join(chunks), content_type)
        except (httpx.HTTPError, OSError):
            return None
    try:
        data = base64.b64decode(reference, validate=True)
    except (binascii.Error, ValueError):
        return None
    return _canonical_data_uri(data, mime_hint)


def _build_attached_context(request: CleanCompletionRequest, state: CleanState) -> str:
    """Build labeled text context from the request's attachments.

    2026-09-28 (user ruling): renamed from ``_build_general_use_context``.
    The block is attached CONTEXT for whichever harness runs, not a
    general_use-specific feature -- see the matching de-gating at the
    ``context_text`` assignment and in ``resolution.py`` / ``runtime.py``.

    Collects the labeled ``[ATTACHED TEXT CONTEXT n]`` sections and the
    ``[ATTACHED WORKING ARTIFACT]`` section.  IMAGE attachments are
    deliberately NOT inlined here: they travel as image bytes on the package
    (``unit.images``) so the model receives them natively at full resolution.
    """
    sections: list[str] = []
    for index, attachment in enumerate(request.context_attachments, start=1):
        if attachment.type != "text" or not attachment.content:
            continue
        label = f"[ATTACHED TEXT CONTEXT {index}"
        if attachment.language:
            label += f" language={attachment.language}"
        if attachment.mime:
            label += f" mime={attachment.mime}"
        sections.append(
            f"{label}]\n{attachment.content}\n[END ATTACHED TEXT CONTEXT {index}]"
        )
    if request.artifact:
        artifact = state.artifact_store.current
        if artifact is not None:
            artifact_context = state.artifact_store.build_context_package(
                include_content=True,
                # 2026-09-18 (user ruling): BM25 is NOT involved in general_use.
                # Nothing in the general-use prompt asks for a section index;
                # it is a lexical retrieval aid for change-order anchoring.
                include_section_index=False,
                artifact_id=artifact.artifact_id,
            )
            if artifact_context:
                sections.append(
                    "[ATTACHED WORKING ARTIFACT]\n"
                    + artifact_context
                    + "\n[END ATTACHED WORKING ARTIFACT]"
                )
    return "\n\n".join(sections)


def _build_t3_context(
    request: CleanCompletionRequest,
    state: CleanState,
    prompt: str,
    task_type: str,
    image_details: list[tuple[str, str, int, int | None, int | None]],
) -> str:
    """Build the complete substantive request context reserved for T3."""
    sections = ["[ORIGINAL USER REQUEST]\n" + prompt]
    for index, attachment in enumerate(request.context_attachments, start=1):
        if attachment.type != "text" or not attachment.content:
            continue
        label = f"[PASTED/TEXT ATTACHMENT {index}"
        if attachment.language:
            label += f" language={attachment.language}"
        if attachment.mime:
            label += f" mime={attachment.mime}"
        sections.append(
            f"{label}]\n{attachment.content}\n[END PASTED/TEXT ATTACHMENT {index}]"
        )
    if image_details:
        image_lines = ["[IMAGE ATTACHMENTS]"]
        for index, (_image, mime, size_bytes, width, height) in enumerate(image_details, start=1):
            image_lines.append(
                f"image_{index}: mime={mime or 'unknown'} bytes={size_bytes} "
                f"width={width} height={height}"
            )
        image_lines.append(
            "Image bytes are attached separately to the user message at original resolution."
        )
        image_lines.append("[END IMAGE ATTACHMENTS]")
        sections.append("\n".join(image_lines))

    evaluate_prior = bool(
        request.routing_override is not None
        and getattr(request.routing_override, "evaluate_prior_wp", False)
    )
    artifact = None
    if evaluate_prior and state.evaluate_store is not None and state.evaluate_store.has():
        pen = state.evaluate_store.get() or {}
        source_content = pen.get("source_content") or ""
        if source_content:
            sections.append(
                "[EVALUATE SOURCE ARTIFACT]\n"
                f"filename={pen.get('source_filename') or ''} "
                f"language={pen.get('source_language') or ''}\n"
                f"{source_content}\n[END EVALUATE SOURCE ARTIFACT]"
            )
    elif request.artifact or task_type in {
        "bug_fix", "modify_artifact", "evaluate_artifact",
    }:
        artifact = state.artifact_store.current
    if artifact is not None:
        # 2026-09-18 (user ruling): the BM25 section index is emitted ONLY for
        # the one harness whose cloud prompt promises it.  Every other harness
        # receives [ARTIFACT CONTENT] + [tree-sitter outline], which is the
        # complete grounding a change order needs -- and is exactly what
        # runtime.py's own (previously unreachable) T3 context builder already
        # produced per the 2026-09-08 de-legacization.
        artifact_context = state.artifact_store.build_context_package(
            include_content=True,
            include_section_index=(task_type == "evaluate_artifact"),
            artifact_id=artifact.artifact_id,
        )
        if artifact_context:
            sections.append(
                "[WORKING ARTIFACT]\n"
                f"filename={getattr(artifact, 'source_filename', '') or ''} "
                f"language={artifact.language or ''}\n"
                f"{artifact_context}\n[END WORKING ARTIFACT]"
            )
    if state.session_memory is not None:
        try:
            memory_context = state.session_memory.build_memory_section(prompt)
        except Exception:
            memory_context = ""
        if memory_context:
            sections.append("[RELEVANT SESSION CONTEXT]\n" + memory_context)
    return "\n\n".join(sections)


# 2026-09-18 (B3b): unknown-field policy.
#
# By default pydantic IGNORES unknown fields, so a client that misspells or
# misnames a field gets a 200 and its input is silently discarded.  That is the
# exact failure mode B3a hit: /v1/artifact/load accepted `origin_prompt`,
# dropped it, and returned an artifact with no origin prompt -- no error, no
# signal, and the caller had no way to tell.
#
# Policy is SCOPED BY MODEL FAMILY because the two families have different
# contracts:
#
#   CoE-authored models (ArtifactAttachment, ContextAttachment,
#   RoutingOverride, CleanCompletionRequest, LoadArtifactRequest): every field
#   is ours, so an unknown one is unambiguously a caller/typo bug -> FORBID.
#
#   OpenAI-surface mirrors (ContentPart, CleanMessage): their field names come
#   from the OpenAI spec, and real clients (the OpenAI SDK among them) send
#   fields we do not model -- `name` on a message, newer content-part types.
#   Forbidding there would turn normal forward-compatible traffic into a 422,
#   which is a compatibility break rather than a bug fix -> STAY TOLERANT.
#
# Rejection surfaces as a normal FastAPI 422 naming the offending field, so the
# mishandling becomes APPARENT instead of silent.
def _make_strict_model():
    """Return a BaseModel subclass that REJECTS unknown fields (extra="forbid").

    Scoped deliberately: see the comment below for why only the CoE-authored
    models use this and the OpenAI-surface mirrors do not.
    """
    return type(
        "CoEStrictModel",
        (BaseModel,),
        {"model_config": ConfigDict(extra="forbid")},
    )


CoEStrictModel = _make_strict_model()


class ContentPart(BaseModel):
    type: str  # "text" | "image_url"
    text: str | None = None
    image_url: str | dict[str, Any] | None = None


class CleanMessage(BaseModel):
    role: str
    content: str | list[ContentPart]


class ArtifactAttachment(CoEStrictModel):
    content: str
    language: str | None = None
    origin: str = "user-supplied"
    # Step 5 (spec Step 1 addendum): optional original filename of the
    # attached artifact.  Used for descriptor-tag derivation when the
    # artifact is later modified (see `source_filename` threading into
    # executor.execute()).  None for pasted content with no file.
    filename: str | None = None
    # 2026-09-14 (spec 2.3): optional ORIGINATING PROMPT for an externally
    # supplied artifact.  Additive and optional -- absent means None means the
    # artifact has no known origin, so local regeneration is structurally
    # unavailable and modification routes through the cloud change order.
    # This is the ONLY way an externally-loaded artifact can acquire
    # local-regeneration capability.
    origin_prompt: str | None = None


class ContextAttachment(CoEStrictModel):
    type: str  # "text" | "image"
    content: str | None = None
    mime: str | None = None
    language: str | None = None


class RoutingOverride(CoEStrictModel):
    domain: str | None = None
    task_type: str | None = None
    think: bool | None = None
    # 2026-09-01: /evaluate marker.  When True, the server seeds the
    # evaluate_object_store with the immediately prior turn's WP as the
    # object to evaluate (general_use WP -> evaluate_artifact).
    evaluate_prior_wp: bool | None = None
    # 2026-09-14: /task-<name>-legacy marker.  Selects the LEGACY route for
    # that harness -- i.e. the behaviour that existed before a replacement
    # route was introduced.  Only modify_artifact consults it today (its
    # origin-gated routing is the replacement); other harnesses ignore it, so
    # the flag is harmless when unused.  Mirrored programmatically via
    # routing_override.legacy_route or metadata.legacy_route (same precedence
    # chain as `think`), so a scripted driver can reproduce CLI behaviour
    # exactly.  See CoE_coder_spec_modify_pipeline_DRAFT.md 4.2.
    legacy_route: bool | None = None


class CleanCompletionRequest(CoEStrictModel):
    messages: list[CleanMessage]
    artifact: ArtifactAttachment | None = None
    context_attachments: list[ContextAttachment] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    routing_override: RoutingOverride | None = None
    artifact_identity_resolution: str | None = None
    stream: bool = False


class CleanState:
    """Server-wide shared state, created at startup."""

    cfg: dict[str, Any]
    resolution: CleanResolution
    executor: CleanExecutor
    vram: CleanVramManager
    router: CleanHeuristicFirstRouter
    session_memory: SessionMemory
    artifact_store: Any  # ArtifactStore
    evaluate_store: Any  # EvaluateObjectStore (single-slot holding pen)
    vision_models: set[str]
    pending_clarification: dict[str, Any] | None = None


state: CleanState | None = None


def _configured_vision_models(cfg: dict[str, Any]) -> set[str]:
    """Collect explicitly declared vision-capable model IDs."""
    models: set[str] = set()
    for entry in (cfg.get("models") or {}).values():
        if isinstance(entry, dict) and entry.get("vision") and entry.get("model_id"):
            models.add(str(entry["model_id"]))
    resolution = cfg.get("clean_resolution") or {}
    for entry in (resolution.get("model_refs") or {}).values():
        if isinstance(entry, dict) and entry.get("vision") and entry.get("model_id"):
            models.add(str(entry["model_id"]))
    cloud = (cfg.get("backends") or {}).get("cloud") or {}
    if cloud.get("vision") and cloud.get("model_id"):
        models.add(str(cloud["model_id"]))
    if cloud.get("vision_model_id"):
        models.add(str(cloud["vision_model_id"]))
    return models


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize the clean slice as a unified model server.

    This creates all internal components:
    * Ollama transport (primary model backend)
    * FLM transport + heuristic-first router (NPU if available)
    * VRAM manager
    * Clean resolution (domain → harness → model pair)
    * Session memory (V2: chunk index + turn summary index)
    * Clean executor (owns the runtime)
    """
    global state
    cfg = load_config()

    # --- Local inference backend: PROBE, START IF ABSENT, and FAIL HARD ---
    # The backend has NO fallback, so unlike FLM (below) an unreachable
    # backend is FATAL: aborting boot is strictly better than serving an
    # instance whose every request would fail against a dead port.  This is
    # also the point where the "never stop the daemon" rule is enforced by
    # construction -- backend_lifecycle.py deliberately has no stop path,
    # because backends.cloud.url is the SAME endpoint on the reference
    # configuration (the local daemon proxies the cloud provider), so an
    # app-initiated stop would take out cloud escalation too.
    backend_status = ensure_backend(
        backend_name="ollama",
        cfg=cfg.get("backends", {}).get("ollama") or {},
    )
    if not backend_status.get("up"):
        _bk_url = str((cfg.get("backends", {}).get("ollama") or {}).get("url", "<unset>"))
        print("", file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        print("[boot] FATAL: local inference backend is NOT AVAILABLE.", file=sys.stderr)
        print(f"         backend : ollama", file=sys.stderr)
        print(f"         url     : {_bk_url}", file=sys.stderr)
        print(f"         reason  : {backend_status.get('error')}", file=sys.stderr)
        if backend_status.get("started_by_us"):
            print(f"         spawned : pid {backend_status.get('pid')}", file=sys.stderr)
        if backend_status.get("log_path"):
            print(f"         log     : {backend_status['log_path']}", file=sys.stderr)
        print("", file=sys.stderr)
        print("         The backend has NO fallback -- every request would fail,", file=sys.stderr)
        print("         so CoE-coder is refusing to start rather than serve a dead", file=sys.stderr)
        print("         instance.  Remedies, in order of likelihood:", file=sys.stderr)
        print("           1. start it and retry:   ollama serve", file=sys.stderr)
        print("           2. if it is installed elsewhere, set", file=sys.stderr)
        print("              backends.ollama.binary_path in config.yaml", file=sys.stderr)
        print("           3. if it listens on another port, set", file=sys.stderr)
        print("              backends.ollama.url in config.yaml", file=sys.stderr)
        print("           4. to run a different backend entirely (llama.cpp, LM", file=sys.stderr)
        print("              Studio), point backends.ollama.url at it and set", file=sys.stderr)
        print("              binary_name / serve_args -- nothing here is", file=sys.stderr)
        print("              Ollama-specific.", file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        raise SystemExit(2)
    if backend_status.get("started_by_us"):
        print(
            f"[boot] Backend: auto-started (pid {backend_status['pid']}; "
            f"log {backend_status['log_path']})"
        )
    else:
        print("[boot] Backend: up (already running; not spawned by CoE-coder)")

    # --- Transports ---
    ollama_cfg = cfg["backends"]["ollama"]
    ollama_transport = CleanOllamaTransport(
        str(ollama_cfg["url"]),
        str(ollama_cfg.get("api_key", "ollama")),
        float(ollama_cfg.get("timeout", 900)),
    )

    # FLM (NPU router) — optional.  If FLM is down, the
    # router falls back to Ollama for model confirmation.
    flm_cfg = cfg.get("backends", {}).get("flm", {})
    flm_transport: CleanFlmRouterTransport | None = None
    flm_status = {"up": False, "reason": "not configured"}
    if flm_cfg:
        # Auto-start lifecycle: the config's ``auto_start`` flag
        # (and its ``binary_path``/``router_model_id``/
        # ``startup_timeout_s``/``log_dir`` siblings) drives whether
        # the server spawns ``flm serve <router_model>`` at boot when
        # the configured URL is not already responding.  This is the
        # same ``ensure_flm`` lifecycle the clean-slice runner used;
        # it was not previously wired into this server entrypoint.
        # warn-and-continue: on any failure we fall back to Ollama.
        if bool(flm_cfg.get("auto_start", False)):
            ensure = ensure_flm(
                flm_url=str(flm_cfg["url"]),
                router_model_id=str(flm_cfg.get("router_model_id", "qwen3.5:4b")),
                binary_override=flm_cfg.get("binary_path"),
                log_dir=flm_cfg.get("log_dir"),
                startup_timeout_s=float(flm_cfg.get("startup_timeout_s", 60)),
                extra_serve_args=[
                    "-r", str(int(flm_cfg.get("img_pre_resize", 1))),
                ],
            )
            if ensure.get("error"):
                print(f"[boot] FLM auto-start failed: {ensure['error']}")
        try:
            flm_transport = CleanFlmRouterTransport(
                str(flm_cfg["url"]),
                str(flm_cfg.get("api_key", "flm")),
                float(flm_cfg.get("timeout", 60)),
            )
            # Probe FLM health — use /v1/models (OpenAI-compatible)
            # not /health (which FLM doesn't expose).
            import httpx
            flm_url = str(flm_cfg["url"]).rstrip("/")
            # Strip trailing /v1 if present, then add /v1/models
            base = flm_url.removesuffix("/v1")
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{base}/v1/models",
                    headers={"Authorization": f"Bearer {str(flm_cfg.get('api_key', 'flm'))}"},
                )
                if resp.status_code == 200:
                    flm_status = {"up": True, "reason": "ok"}
                else:
                    flm_status = {"up": False, "reason": f"HTTP {resp.status_code}"}
        except Exception as exc:
            flm_transport = None
            flm_status = {"up": False, "reason": str(exc)[:120]}

    if flm_status["up"]:
        print(f"[boot] FLM: up (router on NPU)")
    else:
        print(f"[boot] FLM: down ({flm_status['reason']}); router will use Ollama")

    # --- VRAM ---
    vram = CleanVramManager(cfg, transport=ollama_transport)
    initial = await vram.state()
    print(
        f"[boot] VRAM: total={initial.total_vram_gb:.2f}GB "
        f"usable={initial.usable_gb:.2f}GB "
        f"resident={initial.resident_gb:.2f}GB"
    )
    # Boot eviction policy (vram_tiers.evict_on_boot, default true).  true =
    # unload EVERYTHING resident, whoever loaded it -- the clean-slate
    # guarantee for a dedicated 16 GB appliance.  false = unload only models
    # CoE itself can spawn, leaving another tool's model resident; the user
    # must then leave >= 16 GB free or admission will fail on first request.
    _evict_all_boot = bool(
        ((cfg.get("vram_tiers", {}) or {}).get("evict_on_boot", True))
    )
    _evicted = await vram.evict_all(only_roster=not _evict_all_boot)
    print(
        f"[boot] VRAM: evict_all done ({_evicted} unloaded; ledger-first; "
        + (
            "non-resident models skipped)"
            if _evict_all_boot
            else "evict_on_boot=false: non-CoE models left resident)"
        )
    )

    # --- Boot-time footprint verification (2026-09-21) -------------------
    # WARN ONLY -- this never fails the boot and never changes admission.
    # Compares the DECLARED weight footprints (vram_tiers.model_footprints_gb)
    # against the sizes the backend itself reports, so a stale declaration is
    # visible BEFORE the first request rather than surfacing as a puzzling
    # admission result later.  A stale declaration already has a defined
    # non-fatal policy (the per-slot generalist fallback), and failing a boot
    # on a small drift would be a false error -- hence warn-only.
    #
    # Provider-agnostic and best-effort: `registry_sizes()` returns {} when the
    # backend exposes no size information, in which case we say so and move on.
    # CLOUD IS EXCLUDED: cloud ids never enter the local admission path (user
    # ruling 2026-09-21, "no cloud in local matters"), so they are filtered out
    # of both sides of the comparison.
    try:
        _registered = await ollama_transport.registry_sizes()
        _footprints = dict((cfg.get("vram_tiers", {}) or {}).get("model_footprints_gb", {}) or {})
        # SCOPE TO THE CoE ROSTER.  The user's registry may hold many models
        # that have nothing to do with CoE (a probe model, an unrelated pull).
        # Reporting those as "missing declaration" would be a FALSE ERROR and
        # would bury the real findings, so only roster members are compared.
        _roster: set[str] = set()
        _cloud_ids: set[str] = set()
        for _src in (
            list((cfg.get("models", {}) or {}).values()),
            list(((cfg.get("clean_resolution", {}) or {}).get("model_refs", {}) or {}).values()),
        ):
            for _entry in _src:
                if not isinstance(_entry, dict):
                    continue
                _mid = _entry.get("model_id")
                if not _mid:
                    continue
                _mid = str(_mid)
                if _entry.get("provider") == "cloud" or "cloud" in _mid:
                    _cloud_ids.add(_mid)
                    continue
                _roster.add(_mid)
        _norm = lambda s: s[: -len(":latest")] if s.endswith(":latest") else s
        _roster_norm = {_norm(r) for r in _roster}
        _registered = {
            k: v for k, v in _registered.items()
            if _norm(k) in _roster_norm
        }
        if not _registered:
            print("[boot] Footprints: not verified (backend reports no model sizes)")
        else:
            _findings = verify_footprints(_registered, _footprints, tolerance=0.05)
            # Orphan reporting is also roster-scoped: a footprint key for a
            # model CoE no longer references is a real finding, but only among
            # keys that look like CoE entries.  Every key in the table IS
            # declared by CoE, so this needs no further filter.
            _drift = [f for f in _findings if f["kind"] == "drift"]
            _missing = [f for f in _findings if f["kind"] == "missing"]
            _orphan = [f for f in _findings if f["kind"] == "orphan"]
            print(
                f"[boot] Footprints: verified {len(_registered)} roster model(s) "
                f"at 5% tolerance -- "
                f"{len(_drift)} drift, {len(_missing)} missing, {len(_orphan)} orphan"
            )
            for _f in _drift:
                print(
                    f"[boot]   DRIFT  {_f['model']}: declared {_f['declared']:.2f} GB "
                    f"vs backend {_f['reported']:.2f} GB ({_f['delta_pct']:+.1f}%)"
                )
            for _f in _missing:
                print(
                    f"[boot]   MISSING declaration for {_f['model']} "
                    f"(backend reports {_f['reported']:.2f} GB) -- admission will "
                    f"use the params*0.8 estimate"
                )
            for _f in _orphan:
                print(
                    f"[boot]   ORPHAN declared footprint for {_f['model']} "
                    f"({_f['declared']:.2f} GB) -- matches no registered model"
                )
    except Exception as _fp_exc:  # never let a diagnostic break the boot
        print(f"[boot] Footprints: verification skipped ({type(_fp_exc).__name__}: {_fp_exc})")

    # --- Resolution ---
    # Constructed before the router so the router can read the
    # matrix-derived vocabulary sets (domains, task_types,
    # task_scopes) at init.  CleanResolution only depends on `cfg`.
    resolution = CleanResolution(cfg)

    # --- Artifact Store (V13) ---
    # Must be constructed BEFORE the router, which holds a reference
    # to it (artifact_section_names / artifact_identity_resolution
    # hints).  Previously it was created after the router, causing an
    # UnboundLocalError at startup (fixed 2026-08-04).
    artifact_store = ArtifactStore()
    # --- Evaluate-object holding pen (single-slot) ---
    # Persists the source object + WP of the last evaluate_artifact turn so a
    # follow-up (e.g. "fix the bugs you found") has the source context.
    evaluate_store = EvaluateObjectStore()

    # --- Router ---
    router_model = cfg["models"].get("router_fallback", {}).get("model_id", "qwen3.5:4b")
    router_vocab = resolution.closed_vocabulary()
    router = CleanHeuristicFirstRouter(
        flm_transport=flm_transport if flm_status["up"] else None,
        ollama_router_transport=ollama_transport,
        router_model_id=router_model,
        router_vision_capable=bool(
            cfg["models"].get("router", {}).get("vision", False)
            if flm_status["up"]
            else cfg["models"].get("router_fallback", {}).get("vision", False)
        ),
        router_policy=(
            lambda rp: {**rp, "domain_floor": cfg.get("clean_resolution", {}).get("min_domain_confidence", 0.80)}
        )((cfg.get("clean_resolution", {}) or {}).get("router_probe") or {}),
        vram_for_ollama=vram,
        artifact_store=artifact_store,
        closed_domains=router_vocab["domains"],
        closed_task_types_router_emit=router_vocab["task_types_router_emit"],
        closed_task_types_reach=router_vocab["task_types_reach"],
        closed_task_scopes=router_vocab["task_scopes"],
        # Step 5/6: inject the per-domain semantic anchors (description +
        # include/exclude topics) into the router system prompt so the
        # classifier can ground its domain choice.
        domain_anchors=resolution.domain_anchors(),
        coding_domains=resolution.coding_domains(),
    )

    # --- Session Memory (V2) ---
    session_memory = SessionMemory()

    # --- Generalist fallback ---
    generalist_model = str(cfg["models"]["general_fallback"]["model_id"])
    generalist_provider = str(cfg["models"]["general_fallback"].get("provider", "ollama"))

    # --- Cloud transport (T3 cloud_sota meta-target, optional) ---
    # When ``backends.cloud.enabled`` is true and a URL/model are set,
    # register a cloud transport and expose T3 model/provider params to
    # the runtime.  T3 (cloud_sota) is a DISTINCT escalation path from
    # T5 (local generalist zero-shot) with different design principles:
    # T5 compensates for pruned-model failures; T3 hands the original
    # problem to a SOTA cloud model when the local contract is over
    # budget.  The local generalist (T5) is UNCHANGED and still serves
    # domains with no wired specialist.  If cloud is disabled,
    # cloud_sota targets are silently skipped and T5 stays the fallback.
    transports: dict[str, Any] = {"ollama": ollama_transport}
    if flm_status["up"]:
        transports["flm"] = flm_transport
    t3_model: str | None = None
    t3_vision_model: str | None = None
    t3_provider: str | None = None
    cloud_cfg = cfg.get("backends", {}).get("cloud", {})
    if bool(cloud_cfg.get("enabled", False)):
        cloud_url = str(cloud_cfg.get("url", "")).strip()
        cloud_model = str(cloud_cfg.get("model_id", "")).strip()
        cloud_vision_model = str(cloud_cfg.get("vision_model_id", "")).strip()
        if cloud_url and cloud_model:
            cloud_transport = CleanCloudTransport(
                cloud_url,
                str(cloud_cfg.get("api_key", "")),
                float(cloud_cfg.get("timeout", 120)),
            )
            transports["cloud"] = cloud_transport
            t3_model = cloud_model
            t3_vision_model = cloud_vision_model or None
            t3_provider = "cloud"
            if t3_vision_model:
                print(
                    f"[boot] Cloud T3 escalation enabled (model={cloud_model}, "
                    f"vision_model={t3_vision_model})"
                )
            else:
                print(f"[boot] Cloud T3 escalation enabled (model={cloud_model})")
        else:
            print("[boot] backends.cloud.enabled=true but url/model_id unset; T3 disabled")
    else:
        print("[boot] Cloud T3 escalation disabled; T5 uses local generalist")

    # --- Executor ---
    executor = CleanExecutor(
        transports,
        vram,
        session_memory=session_memory,
        inject_last_turn=True,
        generalist_model=generalist_model,
        generalist_provider=generalist_provider,
        t3_model=t3_model,
        t3_vision_model=t3_vision_model,
        t3_provider=t3_provider,
        # Path B remediation: the executor must own the artifact store
        # reference so artifact-aware harnesses (modify_artifact,
        # evaluate_artifact) can build R1's artifact context and apply
        # edits.  Without this, execute_runtime_plan() gets
        # artifact_store=None and the two-round modify path fails with
        # "no_artifact_store" before R1 ever sees the artifact.
        artifact_store=artifact_store,
        evaluate_store=evaluate_store,
    )

    # --- Assemble state ---
    app_state = CleanState()
    app_state.cfg = cfg
    app_state.resolution = resolution
    app_state.executor = executor
    app_state.vram = vram
    app_state.router = router
    app_state.session_memory = session_memory
    app_state.artifact_store = artifact_store
    app_state.evaluate_store = evaluate_store
    app_state.vision_models = _configured_vision_models(cfg)
    app_state.pending_clarification = None
    state = app_state

    print(f"[boot] Clean slice server ready (generalist={generalist_model})")
    yield
    state = None


app = FastAPI(
    title="College of Experts Clean Slice",
    version="2.0.0-clean",
    description="Unified model API with internal routing and session memory",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Check server and backend health."""
    if state is None:
        raise HTTPException(status_code=503, detail="Clean slice is not initialized")
    reachable = await state.executor.transports["ollama"].health()
    # 2026-09-14: modify capability is a ROUTING-RELEVANT property, so it is
    # discoverable here rather than only discovered per-request.  "cloud_order"
    # false means externally-loaded artifacts (which have no originating prompt)
    # cannot be modified at all, so a caller learns that before submitting work
    # rather than from a failure mid-request.
    _cap = {"origin_regen": True, "cloud_order": False}
    try:
        from .provenance import cloud_configured as _cc
        _cap = {"origin_regen": True,
                "cloud_order": _cc((state.cfg or {}).get("backends"))}
    except ImportError:
        pass
    return {
        "status": "ok" if reachable else "degraded",
        "lane": "clean",
        "session_turns": len(state.session_memory),
        "session_chunks": len(state.session_memory.chunk_index) if state.session_memory.chunk_index else 0,
        "modify_pipeline": {
            "combo_mode": (state.cfg or {}).get("modify_combo_mode", "concat"),
            "capability": _cap,
        },
        # 2026-09-19 (PLAN_progress_mechanism.md phase 6): the progress cadence
        # is discoverable BEFORE a request, so a caller knows how often to poll
        # GET /v1/turn/{id}/progress and, for a non-streaming turn, the
        # worst-case latency of a cooperative cancel.
        "progress": {
            "interval_s": (state.cfg or {}).get("progress_interval_s", 30),
            "turn_record_keep": (state.cfg or {}).get("turn_record_keep", _TURN_KEEP),
            "in_flight_turns": len(_ACTIVE_TURNS),
            "endpoint": "GET /v1/turn/{request_id}/progress",
        },
        "timestamp": int(time.time()),
    }


@app.get("/v1/vram")
async def vram_state() -> dict[str, Any]:
    """Return current VRAM state."""
    if state is None:
        raise HTTPException(status_code=503, detail="Clean slice is not initialized")
    return (await state.vram.state()).to_dict()


@app.post("/v1/session/clear")
async def clear_session() -> dict[str, Any]:
    """Clear session memory (all turns, chunks, summaries).

    Bench runners can call this between independent runs to
    ensure no cross-problem context injection.  For interactive
    sessions, this starts a fresh conversation.
    """
    if state is None:
        raise HTTPException(status_code=503, detail="Clean slice is not initialized")
    state.session_memory.clear()
    return {"status": "ok", "session_turns": 0}


@app.post("/v1/cancel")
async def cancel_turn(request: Request) -> dict[str, Any]:
    """Cancel an in-flight turn (2026-09-07, user-approved).

    Body: {"request_id": "<X-Request-Id the turn was sent with>"}.
    Cancels the executor task; the runtime records the attempt as
    `cancelled` and preserves session/artifact state.  The SSE stream
    for that turn drains and closes, so the client returns to its
    prompt cleanly.  Unknown/finished request ids are a 404 (no-op).
    """
    body = await request.json()
    request_id = str(body.get("request_id") or "")
    if not request_id:
        raise HTTPException(status_code=400, detail="request_id is required")
    # 2026-09-19 (PLAN_progress_mechanism.md phase 5): TWO cancel mechanisms,
    # because the two transports stop in fundamentally different ways.
    #
    #   record.cancel_requested = True  COOPERATIVE.  The next runtime
    #       `checkpoint` returns ABORT and the turn ends cleanly.  Works for
    #       BOTH transports, and is the ONLY option for a non-streaming turn.
    #   task.cancel()                   ABRUPT.  Streaming only.  Kills the
    #       executor task outright; the SSE loop drains and closes.
    #
    # Why the record is consulted FIRST: before this change, cancel was
    # reachable only when a task was registered, and registration happened
    # only inside _event_stream.  A non-streaming turn therefore 404'd here
    # and was genuinely uncancellable while it ran.
    #
    # A non-streaming turn deliberately does NOT register a task: an abrupt
    # task.cancel() would tear down the handler mid-response.  So `mode`
    # reports which path was taken, and a 404 now means NEITHER a record NOR a
    # task exists -- i.e. the id was never in flight.
    rec_t = _ACTIVE_TURNS.get(request_id)
    task = _ACTIVE_REQUEST_TASKS.get(request_id)
    task_live = task is not None and not task.done()
    if rec_t is None and not task_live:
        raise HTTPException(status_code=404, detail=f"no in-flight turn '{request_id}'")
    modes: list[str] = []
    if rec_t is not None:
        rec_t.cancel_requested = True
        modes.append("cooperative")
    if task_live and task is not None:
        task.cancel()
        modes.append("task")
    return {
        "status": "cancelling",
        "request_id": request_id,
        "mode": "+".join(modes),
    }


@app.get("/v1/turn/{request_id}/progress")
async def turn_progress(request_id: str, since: int = 0) -> dict[str, Any]:
    """Live progress for an in-flight (or recent) turn.

    2026-09-19 (PLAN_progress_mechanism.md phase 5).  This is the surface that
    makes a NON-STREAMING turn observable.  It is transport-agnostic: it reads
    the turn record, never the runtime, so it answers from local state and works
    identically whether the turn is streaming or not.

    ``since`` is a cursor over the record's event sequence.  ``next_cursor`` in
    the response is the value to pass on the next poll; ``seq`` is ABSOLUTE and
    survives event-list trimming, so a cursor older than the retained window
    simply observes a gap rather than restarting at 1.

    ``since_last_event_s`` is the hang/silence signal, computed at REQUEST time
    -- a stalled turn is therefore visible the moment it is polled, with no
    dependency on any emission cadence.

    Unknown ids are a 404, matching /v1/cancel's existing contract.  Note a
    FINISHED turn keeps its record (the registry is bounded by insertion order,
    not by completion), so a client can still read the final state.
    """
    if state is None:
        raise HTTPException(status_code=503, detail="Clean slice is not initialized")
    rec = _ACTIVE_TURNS.get(request_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"no turn '{request_id}'")
    return rec.snapshot(since=max(0, int(since or 0)))


@app.get("/v1/session/stats")
async def session_stats() -> dict[str, Any]:
    """Return session memory statistics for debugging."""
    if state is None:
        raise HTTPException(status_code=503, detail="Clean slice is not initialized")
    sm = state.session_memory
    return {
        "turns": len(sm),
        "total_tokens": sm.total_tokens,
        "max_tokens": sm.max_tokens,
        "chunk_index_size": len(sm.chunk_index) if sm.chunk_index else 0,
        "summary_index_size": len(sm.summary_index) if sm.summary_index else 0,
        "chunk_stats": sm.chunk_index.stats() if sm.chunk_index else {},
        "summary_stats": sm.summary_index.stats() if sm.summary_index else {},
        "artifact_store": state.artifact_store.stats() if state.artifact_store else {},
    }


@app.get("/v1/artifact")
async def get_artifact() -> dict[str, Any]:
    """Return the current artifact content (debug endpoint)."""
    if state is None:
        raise HTTPException(status_code=503, detail="Clean slice is not initialized")
    art = state.artifact_store.current
    if not art:
        return {"has_artifact": False}
    return {
        "has_artifact": True,
        "artifact_id": art.artifact_id,
        "language": art.language,
        "produced_by_model": art.produced_by_model,
        "content_length": len(art.content),
        "section_count": len(art.sections),
        "section_names": [s.name for s in art.sections if s.name],
        "content": art.content,
    }


class LoadArtifactRequest(CoEStrictModel):
    content: str
    language: str | None = None
    origin: str = "loaded"
    # 2026-09-18 (B3a FIX): both fields were ABSENT from this model while the
    # endpoint body ALREADY read `getattr(request, "origin_prompt", None)` and
    # passed it to create().  Pydantic drops unknown fields silently, so a
    # caller that supplied an origin prompt got a 200 and an artifact with
    # NO origin prompt -- i.e. PATH A was unreachable through the API and the
    # caller had no signal that its input had been discarded.
    #
    # `origin_prompt` is spec 2.3 (the only way an externally-loaded artifact
    # acquires local-regeneration capability).  `filename` feeds
    # `source_filename`, which the shared naming path uses for the
    # `modify-<file>-<intent>` descriptor on promotion.
    # User ruling: everything must be reachable via the API.
    origin_prompt: str | None = None
    filename: str | None = None


@app.post("/v1/artifact/load")
async def load_artifact(request: LoadArtifactRequest) -> dict[str, Any]:
    """Load an external artifact (from disk) into the store.

    This is the entry point for the "external artifact" use case:
    a file the user already has becomes the working artifact,
    and subsequent modify/evaluate turns operate on it.

    The artifact is parsed into sections (tree-sitter HTML +
    CSS rule extraction + JS function extraction) and indexed
    in BM25 so the modify pipeline can target individual sections.
    """
    if state is None:
        raise HTTPException(status_code=503, detail="Clean slice is not initialized")
    artifact_id = state.artifact_store.create(
        request.content,
        language=request.language,
        origin=request.origin,
        produced_by_model=None,  # external — no specific producing model
        # Spec 2.3: optional.  If supplied, this externally-loaded artifact
        # gains local-regeneration capability; if absent it is
        # cloud-change-order only (pre-existing behaviour).
        origin_prompt=request.origin_prompt,
        # 2026-09-18 (B3a): the original filename, so the promotion descriptor
        # can be `modify-<file>-<intent>` rather than `modify-<intent>`.
        source_filename=request.filename,
    )
    art = state.artifact_store.current
    return {
        "status": "loaded",
        "artifact_id": artifact_id,
        "content_length": len(art.content),
        "section_count": len(art.sections),
        "section_names": [s.name for s in art.sections if s.name],
    }


@app.get("/v1/session/debug-injection")
async def debug_injection(prompt: str = "") -> dict[str, Any]:
    """Debug endpoint: show what the memory injection would produce.

    Query param: ``prompt`` — the prompt to test.
    Returns the relevance verdict, the injected section text,
    and the chunks/summaries that matched.
    """
    if state is None:
        raise HTTPException(status_code=503, detail="Clean slice is not initialized")
    sm = state.session_memory
    if not prompt:
        return {"error": "provide a ?prompt= query parameter"}

    from .turn_summary import heuristic_memory_relevance
    from .memory_inject import build_memory_section

    recent_summaries = sm.summary_index.get_recent(3) if sm.summary_index else []
    recent_chunks = []
    for tid in sm.get_recent_turn_ids(3):
        if sm.chunk_index:
            recent_chunks.extend(sm.chunk_index.get_chunks_for_turn(tid))

    relevance = heuristic_memory_relevance(prompt, recent_summaries, recent_chunks)
    section = build_memory_section(
        relevance, prompt,
        sm.summary_index, sm.chunk_index,
        recent_turn_ids=sm.get_recent_turn_ids(2),
    ) if sm.chunk_index and sm.summary_index else ""

    return {
        "relevance": relevance,
        "section_length": len(section),
        "section": section[:2000] if section else "",
        "recent_summaries": [s.to_dict() for s in recent_summaries],
        "recent_chunk_count": len(recent_chunks),
    }


@app.get("/v1/session/turns")
async def session_turns(*, last: int = 0) -> dict[str, Any]:
    """Return the raw artifact log entries (ground truth).

    Each entry contains the full user_prompt, system_modified_prompt,
    model_output, routing decision, resolver decision, and timing
    for a completed turn.  This is the ground-truth record — what
    actually happened, for reconstruction and debugging.

    Query params:
    * ``last`` — return only the last N turns (0 = all)
    """
    if state is None:
        raise HTTPException(status_code=503, detail="Clean slice is not initialized")
    sm = state.session_memory
    if last > 0:
        entries = sm.get_recent(last)
    else:
        entries = list(sm)
    return {
        "turns": [
            {
                "task_id": e.task_id,
                "timestamp": e.timestamp,
                "user_prompt": e.user_prompt,
                "system_modified_prompt": e.system_modified_prompt,
                "model_output": e.model_output,
                "routing_decision": e.routing_decision,
                "resolver_decision": e.resolver_decision,
                "persona": e.persona,
                "harness_id": e.harness_id,
                "primary_model": e.primary_model,
                "primary_provider": e.primary_provider,
                "elapsed_s": e.elapsed_s,
                "token_count": e.token_count,
            }
            for e in entries
        ],
        "total_turns": len(sm),
    }


class StreamingProgressCallback(CleanProgressCallback):
    """Forwards token events to an asyncio.Queue for SSE streaming.

    The runtime calls ``on_event`` synchronously from its
    async event loop.  We push token text onto a queue so
    the SSE response can consume it asynchronously.  Tier
    start/complete and summary events are forwarded as
    SSE metadata chunks (not content) so the client can
    see routing decisions and tier transitions live.
    """

    def __init__(self, queue: asyncio.Queue) -> None:
        self._queue = queue

    def on_event(self, event: CleanProgressEvent) -> CleanControlDecision | None:
        try:
            if event.event_type == "token":
                text = event.text or ""
                if text:
                    # Step 6.3: include the tier so _event_stream can
                    # suppress R1/R2 internal JSON (two-round
                    # modify_artifact) from ever reaching the client
                    # as visible content.  Also carry the transport's
                    # is_thinking flag (from event.metadata) so the
                    # client can separate thinking tokens from
                    # content tokens.
                    is_thinking = bool(event.metadata and event.metadata.get("is_thinking"))
                    self._queue.put_nowait(("token", text, event.model_id or "", event.tier or "", is_thinking))
            elif event.event_type == "tier_start":
                self._queue.put_nowait(("tier_start", event.tier or "", event.model_id or ""))
            elif event.event_type == "status":
                # Long-silence phase announcer (2026-09-05): model load /
                # attachment reading.  Forwarded as SSE metadata so the
                # client can show a live status tag instead of dead air
                # (the user cancelled a turn mid model-load believing
                # nothing was happening).
                self._queue.put_nowait((
                    "status",
                    str((event.metadata or {}).get("status_kind") or ""),
                    str(event.text or ""),
                    event.model_id or "",
                ))
            elif event.event_type == "tier_complete":
                self._queue.put_nowait((
                    "tier_complete", event.tier or "", event.model_id or "",
                    event.metadata or {},
                ))
            elif event.event_type == "complete":
                output = event.output or ""
                # Step 5.9/6.3: the two-round modify_artifact block's
                # `complete` event carries a short internal marker
                # ("[[modify_artifact: pending edit for artifact ...]]"),
                # not user-visible content -- the real content (a diff
                # or full body) is only known after server.py commits
                # the edit post-freeze.  Flag it so _event_stream never
                # forwards that marker as visible content.
                is_pending_edit = bool(event.metadata and event.metadata.get("pending_artifact_edit"))
                self._queue.put_nowait(("complete", output, event.model_id or "", is_pending_edit))
            elif event.event_type == "tier_retry":
                # 2026-09-18 (B4a FIX): this producer was MISSING.  The runtime
                # emits tier_retry (runtime.py:7471 pathology / :7508 functional)
                # carrying retry_kind, reason, attempt, limit, action and error;
                # _event_stream consumed it and the CLI had a renderer for it --
                # but on_event never queued it, so the event died at the
                # transport and no client ever saw a retry.  The consumer
                # unpacks a 4-tuple, so this producer matches tier_complete's
                # shape exactly.
                self._queue.put_nowait((
                    "tier_retry", event.tier or "", event.model_id or "",
                    event.metadata or {},
                ))
            elif event.event_type == "checkpoint":
                # 2026-09-19 (PLAN_progress_mechanism.md phase 2): QUEUED, not
                # dropped.  This branch previously read `return
                # CleanControlDecision.CONTINUE` and nothing else, so the
                # runtime's periodic progress event -- the ONLY event carrying
                # `text_so_far` and `elapsed_s` -- was discarded at the
                # transport.  _event_stream had no consumer for it either, so
                # no client ever saw progress mid-turn.
                #
                # The payload is a 6-TUPLE, a NEW shape.  Every other event
                # uses 2-, 3-, 4- or 5-tuples, so a stale consumer cannot
                # silently mis-unpack this one -- it would raise on arity
                # rather than read the wrong fields.
                self._queue.put_nowait((
                    "checkpoint",
                    event.tier or "",
                    event.model_id or "",
                    event.metadata or {},
                    event.text_so_far or "",
                    float(event.elapsed_s or 0.0),
                ))
                return CleanControlDecision.CONTINUE
        except Exception:
            # Never let the streaming callback crash the runtime.
            # If the queue is full or the event loop is in a bad
            # state, just drop the event and continue.
            pass
        return None


# ---------------------------------------------------------------------------
# Turn progress record (2026-09-19) -- PLAN_progress_mechanism.md phases 1-5
# ---------------------------------------------------------------------------
# ONE progress mechanism for the whole service.  The runtime emits a single
# ``checkpoint`` event type on a timer; this registry is the transport-agnostic
# place it lands, so a NON-STREAMING caller -- which has no SSE channel at all
# -- can read live progress and cancel an in-flight turn.
#
# Why this exists.  Progress delivery and cancel registration were both
# properties of the SSE TRANSPORT rather than of the TURN: the callback only ran
# in ``_stream_completion``, and ``_ACTIVE_REQUEST_TASKS`` was only populated
# inside ``_event_stream``.  With ``stream: false`` the executor was called with
# no ``progress=`` argument whatsoever, so every event type was discarded at the
# source -- no status, no tier transitions, no cancel, and the CLI printed
# "[cancelled] turn cancelled (state preserved)" while the server kept working.
#
# The same ``checkpoint`` event feeds BOTH surfaces:
#     streaming      -> SSE metadata (live progress line in the CLI)
#     non-streaming  -> this record, read via GET /v1/turn/{id}/progress
#
# Cancellation is one flag.  Streaming cancels the task outright (unchanged
# behaviour); non-streaming cancels COOPERATIVELY, because the runtime only
# READS the callback's return value at the ``checkpoint`` emit sites -- an
# abrupt ``task.cancel()`` mid-handler would kill the HTTP response itself.
# ---------------------------------------------------------------------------

_TURN_EVENT_CAP = 200      # per-turn retained events; ``seq`` stays absolute
_TURN_KEEP = 64            # turns retained before eviction (config override: phase 6)


@dataclass
class _TurnRecord:
    """Live progress + cancel state for ONE in-flight turn.

    Written by :class:`_RecordingProgress` (which wraps whichever callback the
    transport uses) and read by the progress endpoint.  Deliberately dumb: it
    holds state and derives ``since_last_event_s``; it never calls a model, a
    transport, or the executor.
    """

    request_id: str
    started_at: float
    last_event_at: float
    status: str = "running"          # running | done | error
    tier: str = ""
    model_id: str = ""
    text_length: int = 0
    text_preview: str = ""
    cancel_requested: bool = False
    seq: int = 0                     # ABSOLUTE; survives event-list trimming
    finished_at: float | None = None
    error: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def record(self, event: CleanProgressEvent) -> None:
        """Fold one runtime progress event into this record."""
        self.last_event_at = time.monotonic()
        if event.tier:
            self.tier = event.tier
        if event.model_id:
            self.model_id = event.model_id
        et = event.event_type
        if et == "token":
            # Token text is a CHUNK, not the whole output: accumulate length
            # and keep a rolling tail as the preview.
            txt = event.text or ""
            if txt:
                self.text_length += len(txt)
                self.text_preview = (self.text_preview + txt)[-240:]
        else:
            # ``text_so_far`` is already truncated to 240 chars by the
            # runtime's _summary_preview, so it can only ever RAISE the
            # length (via max) -- it never undercuts the token count.
            tsf = event.text_so_far or event.output or ""
            if tsf:
                self.text_length = max(self.text_length, len(tsf))
                self.text_preview = tsf[-240:]
        self.seq += 1
        # Only lifecycle events are retained; the token stream is high-volume
        # and its aggregate (text_length) is already tracked above.
        if et in ("checkpoint", "status", "tier_start",
                  "tier_complete", "tier_retry", "complete"):
            self.events.append({
                "seq": self.seq,
                "event_type": et,
                "tier": event.tier or "",
                "model_id": event.model_id or "",
                "elapsed_s": round(float(event.elapsed_s or 0.0), 3),
                "text_length": self.text_length,
                "status_kind": str((event.metadata or {}).get("status_kind", "")),
            })
            if len(self.events) > _TURN_EVENT_CAP:
                # Drop the oldest, but keep ``seq`` absolute so a caller's
                # cursor stays meaningful -- it simply observes a gap.
                del self.events[:len(self.events) - _TURN_EVENT_CAP]

    def snapshot(self, since: int = 0) -> dict[str, Any]:
        """Serialisable view for the progress endpoint.

        ``since_last_event_s`` is the hang/silence signal, and it is computed
        HERE (at request time) rather than emitted by the runtime, so a stalled
        turn is visible the moment it is polled -- no waiting for a cadence.
        """
        now = time.monotonic()
        return {
            "request_id": self.request_id,
            "status": self.status,
            "elapsed_s": round(now - self.started_at, 3),
            "since_last_event_s": round(now - self.last_event_at, 3),
            "tier": self.tier,
            "model_id": self.model_id,
            "text_length": self.text_length,
            "text_preview": self.text_preview,
            "cancel_requested": self.cancel_requested,
            "error": self.error,
            "events": [e for e in self.events if e["seq"] > since],
            "next_cursor": self.seq,
        }


class _RecordingProgress:
    """Wraps any progress callback: records the event, then delegates.

    Also the SINGLE place cooperative cancellation is decided.  Returning
    ``ABORT`` from a ``checkpoint`` is what stops a non-streaming turn, so this
    must wrap EVERY callback (streaming included) -- otherwise a cancel issued
    against a streaming turn would be recorded but never honoured by the
    runtime's own control path.

    ``inner`` is the transport's callback (``StreamingProgressCallback``) or
    ``None`` for the non-streaming path, where there is nothing to forward to.
    """

    def __init__(self, inner: CleanProgressCallback | None, record: _TurnRecord) -> None:
        self._inner = inner
        self._record = record

    def on_event(self, event: CleanProgressEvent) -> CleanControlDecision | None:
        try:
            self._record.record(event)
        except Exception:
            pass          # never let bookkeeping kill a live turn
        decision: CleanControlDecision | None = None
        if self._inner is not None:
            decision = self._inner.on_event(event)
        if event.event_type == "checkpoint" and self._record.cancel_requested:
            # Cooperative stop.  ONLY ``checkpoint`` carries a control return
            # path: the token emit sites (runtime.py:3469, :6249) discard the
            # callback's return value, so an ABORT sent there is ignored.
            return CleanControlDecision.ABORT
        return decision


# In-flight turn records keyed by the client's X-Request-Id header (or a
# server-minted id when the client supplies none).  Distinct from
# ``_ACTIVE_REQUEST_TASKS``: that registry holds a cancellable asyncio TASK and
# is populated only by the streaming path, whereas this one is populated by
# BOTH paths and is what makes cancel reachable without a stream.
_ACTIVE_TURNS: dict[str, _TurnRecord] = {}


def _register_turn(request_id: str) -> _TurnRecord:
    """Create the record for a turn, evicting old ones."""
    now = time.monotonic()
    rec = _TurnRecord(request_id=request_id, started_at=now, last_event_at=now)
    _ACTIVE_TURNS[request_id] = rec
    _evict_turns()
    return rec


def _evict_turns(keep: int | None = None) -> None:
    """Bound the registry, oldest-first, by INSERTION order.

    Eviction is by insertion, not by completion: a long-running turn must not
    be dropped merely because many short turns finished after it started.
    Dict insertion order gives this for free on 3.7+.
    """
    limit = _TURN_KEEP if keep is None else keep
    while len(_ACTIVE_TURNS) > limit:
        oldest = next(iter(_ACTIVE_TURNS))
        _ACTIVE_TURNS.pop(oldest, None)


def _sse_chunk(
    content: str | None = None,
    *,
    model: str = "",
    role: str | None = None,
    finish_reason: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Format a single SSE chunk in OpenAI streaming format."""
    delta: dict[str, Any] = {}
    if role:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if metadata:
        delta["metadata"] = metadata

    chunk = {
        "id": f"coe-clean-{int(time.time() * 1000)}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {_json.dumps(chunk)}\n\n"


def _parse_slash_command(messages: list[CleanMessage]) -> tuple[RoutingOverride | None, list[CleanMessage]]:
    """Parse /domain-*, /task-* (/task_type-*, /tasktype-*), /thinkon, or
    /thinkoff from the start of the first user message. Returns (override,
    cleaned_messages). ``task`` is the canonical display form; the internal
    field name remains task_type."""
    import re as _re
    if not messages:
        return None, messages
    first = messages[0]
    if first.role != "user":
        return None, messages
    text = first.content if isinstance(first.content, str) else ""
    m = _re.match(r"^/(domain|task_type|tasktype|task)-(\S+)\s*(.*)", text, _re.DOTALL)
    if m:
        kind, value, rest = m.group(1), m.group(2), m.group(3)
        override = RoutingOverride()
        if kind == "domain":
            override.domain = f"domain_{value}"
        else:
            # 2026-09-14: the generic `-legacy` suffix.  It MUST be stripped
            # BEFORE alias resolution: the `(\S+)` capture takes the whole
            # token, so `/task-modify-legacy` would otherwise miss the alias
            # table and fall through to value.replace("-", "_") -- yielding
            # the INVALID task_type "modify_legacy" and a bogus
            # invalid-override rejection.  Stripping first keeps `-legacy`
            # orthogonal to the task vocabulary instead of adding a
            # pseudo-task, and any harness can adopt it later.
            legacy_route = False
            lowered = value.lower()
            for suffix in ("-legacy", "_legacy"):
                if lowered.endswith(suffix) and len(lowered) > len(suffix):
                    value = value[: -len(suffix)]
                    legacy_route = True
                    break
            if legacy_route and not value:
                # e.g. "/task--legacy" -- nothing left to resolve.
                return None, messages
            # Canonical slash vocabulary uses underscore-free short names
            # (newcode, genuse, modify, bugfix, evaluate, userdefined) and
            # legacy underscored/dashed full names; normalize both to the
            # canonical task_type tags the resolver validates against.
            _TASK_DISPLAY_ALIASES = {
                "newcode": "new_code_generation",
                "new_code": "new_code_generation",
                "genuse": "general_use",
                "general_use": "general_use",
                "modify": "modify_artifact",
                "modify_artifact": "modify_artifact",
                "bugfix": "bug_fix",
                "bug_fix": "bug_fix",
                "evaluate": "evaluate_artifact",
                "evaluate_artifact": "evaluate_artifact",
                "userdefined": "user_defined",
                "user_defined": "user_defined",
            }
            override.task_type = _TASK_DISPLAY_ALIASES.get(value, value.replace("-", "_"))
            if legacy_route:
                override.legacy_route = True
        cleaned = list(messages)
        cleaned[0] = CleanMessage(role=first.role, content=rest.strip() if rest.strip() else first.content)
        return override, cleaned
    mt = _re.match(r"^/(thinkon|thinkoff)\s*(.*)", text, _re.DOTALL)
    if mt:
        directive, rest = mt.group(1), mt.group(2)
        override = RoutingOverride(think=(directive == "thinkon"))
        cleaned = list(messages)
        cleaned[0] = CleanMessage(role=first.role, content=rest.strip() if rest.strip() else first.content)
        return override, cleaned
    # 2026-09-01: /evaluate — override to evaluate_artifact, pointing at the
    # immediately prior WP as the object to evaluate.  Sets task_type and a
    # special marker (evaluate_prior_wp) so the server seeds the evaluate
    # object store with the prior turn's work product.
    me = _re.match(r"^/evaluate\s*(.*)", text, _re.DOTALL)
    if me:
        rest = me.group(1)
        override = RoutingOverride(task_type="evaluate_artifact")
        override.evaluate_prior_wp = True  # type: ignore[attr-defined]
        cleaned = list(messages)
        cleaned[0] = CleanMessage(role=first.role, content=rest.strip() if rest.strip() else first.content)
        return override, cleaned
    return None, messages


def _emit_clarification_response(exc: NeedsClarification, stream: bool):
    """Emit a clarification question to the user (matrix not involved)."""
    if stream:
        async def _clarify_stream():
            yield _sse_chunk(
                content=exc.question,
                metadata={"needs_clarification": {
                    "question": exc.question,
                    "candidates": exc.candidates,
                    "pending": True,
                }},
            )
            yield "data: [DONE]\n\n"
        return StreamingResponse(_clarify_stream(), media_type="text/event-stream")
    else:
        return JSONResponse(content={
            "needs_clarification": {
                "question": exc.question,
                "candidates": exc.candidates,
                "pending": True,
            },
        })


def _request_user_text(messages: list[CleanMessage]) -> str:
    """Extract plain user text for a pending clarification answer."""
    parts: list[str] = []
    for message in messages:
        if message.role != "user":
            continue
        if isinstance(message.content, str):
            parts.append(message.content)
        else:
            parts.extend(
                part.text for part in message.content
                if part.type == "text" and part.text
            )
    return "\n".join(parts).strip()


def _resolve_clarification_answer(
    answer: str,
    candidates: list[str],
    artifact_store: Any,
) -> str | None:
    """Resolve an ordinal, exact filename, artifact ID, or path fragment.

    Hardened 2026-09-07: the user's answer may be a FULL PATH (e.g. a
    re-pasted /attach path).  A stored candidate filename that appears
    as a SUBSTRING of the answer resolves to that candidate — exact
    equality is no longer required.  The raw text (before the path) is
    also matched, so "3" / "abort" ordinals still work.
    """
    text = answer.strip()
    lowered = text.lower()
    if lowered in {"abort", "cancel", "stop"}:
        return "abort"
    catalog = {
        item["artifact_id"]: item
        for item in artifact_store.artifact_candidates()
    }
    if text in catalog:
        return text
    for artifact_id, item in catalog.items():
        filename = str(item.get("source_filename") or "")
        if filename and lowered == filename.lower():
            return artifact_id
    # 2026-09-14 FIX (live defect): the prompt offers each candidate as a
    # DISPLAY LABEL, e.g.
    #     generate-a-standalone-html-file-...  [session 9, version 1; id artifact_1]
    # and the live evidence is that users PASTE THAT LABEL BACK VERBATIM.
    # It matched nothing here -- the label's descriptor is a truncated
    # (100-char _slugify) prefix, so it is not the source_filename, and the
    # trailing " [session ...]" means it is not an id either.  The result was
    # a CLARIFICATION LOOP: the identical question returned forever, with a
    # byte-identical candidate list, because the answer could never resolve.
    #
    # Two additions: (a) match the label's own `; id <artifact_id>]` marker,
    # and (b) the substring test below now runs in BOTH directions, so a
    # pasted label resolves through its descriptor prefix.
    _id_match = re.search(r";\s*id\s+([^\];]+?)\s*\]", text)
    if _id_match:
        _cand_id = _id_match.group(1).strip()
        if _cand_id in catalog:
            return _cand_id
    # Substring path match (2026-09-07): full pasted paths contain the
    # stored filename.  Longest filename wins to avoid ambiguity.
    # 2026-09-14: the REVERSE direction is also matched.  A pasted candidate
    # LABEL carries a TRUNCATED descriptor (100-char _slugify prefix), so the
    # stored filename never appears as a substring of the answer; instead the
    # answer begins with the descriptor head, which IS a prefix of the stored
    # filename.  The 12-char floor stops a short generic word from resolving
    # everything.
    best = None
    for artifact_id, item in catalog.items():
        filename = str(item.get("source_filename") or "")
        desc = str(item.get("descriptor") or "")
        if filename and filename.lower() in lowered:
            if best is None or len(filename) > len(best[1]):
                best = (artifact_id, filename)
            continue
        # Reverse: answer contains this candidate's descriptor head.
        for _probe in (desc.lower(),):
            if len(_probe) >= 12 and _probe in lowered:
                _key = filename or desc
                if best is None or len(_key) > len(best[1]):
                    best = (artifact_id, _key)
                break
    if best:
        return best[0]
    if text.isdigit():
        index = int(text) - 1
        if 0 <= index < len(candidates):
            label = candidates[index]
            for artifact_id, item in catalog.items():
                if artifact_id in label:
                    return artifact_id
    return None


def _make_override_invalid_work(
    domain: str, task_type: str, bad: str, allowed: list[str],
) -> Any:
    """Build a ResolvedWork for the override_invalid path."""
    # 2026-09-18 (B1 FIX): was `from .contracts import ResolvedWork`, but that
    # class is defined in resolution.py -- contracts.py holds CleanUnit and
    # CleanPackage.  The ImportError fired INSIDE the error-reporting helper,
    # so every override_invalid path surfaced to the client as an opaque
    # HTTP 500 instead of the intended actionable message.  Reproduced on:
    # any invalid slash task name, an invalid routing_override domain or
    # task_type, and an invalid metadata.cloud_mode on a non-trivial request.
    from .resolution import ResolvedWork
    return ResolvedWork(
        domain=domain,
        task_type=task_type,
        task_scope="single_domain_single_task",
        resolution_kind="override_invalid",
        fallback_reason=f"override_invalid: {bad}",
        matrix_cell=None,
        rejection_message={
            "code": "override_invalid",
            "severity": "error",
            "bad_override": bad,
            "message": (
                f"Unsupported override: '{bad}'. "
                f"Allowed values are: {allowed}."
            ),
        },
    )


def _emit_rejection_response(work: Any, stream: bool):
    """Emit a rejection SSE event and end the turn. No model invoked."""
    if stream:
        async def _rejection_stream():
            yield _sse_chunk(
                content="",
                metadata={"rejection": work.rejection_message},
            )
            yield "data: [DONE]\n\n"
        return StreamingResponse(_rejection_stream(), media_type="text/event-stream")
    else:
        return JSONResponse(content={"rejection": work.rejection_message})


def _emit_user_defined_unavailable_response(work: Any, stream: bool):
    """Emit a user_defined_unavailable alert and end the turn. The model
    is NOT invoked; the user is informed that the override didn't have
    a registered harness."""
    if stream:
        async def _ud_stream():
            yield _sse_chunk(
                content="",
                metadata={"alert": work.user_defined_message},
            )
            yield "data: [DONE]\n\n"
        return StreamingResponse(_ud_stream(), media_type="text/event-stream")
    else:
        return JSONResponse(content={"alert": work.user_defined_message})


def _emit_override_invalid_response(work: Any, stream: bool):
    """Emit an override_invalid alert. Currently the server produces this
    BEFORE the resolver is called (in the override-validation step);
    this helper exists for symmetry and to support future paths that
    need to surface an override_invalid alert from the resolver."""
    if stream:
        async def _oi_stream():
            yield _sse_chunk(
                content="",
                metadata={"alert": work.rejection_message},
            )
            yield "data: [DONE]\n\n"
        return StreamingResponse(_oi_stream(), media_type="text/event-stream")
    else:
        return JSONResponse(content={"alert": work.rejection_message})


def _emit_modify_unavailable_response(message: str, available: dict, stream: bool):
    """PATH C: modify is unavailable for this artifact.

    Emitted with NO model call.  Both routes are impossible -- the artifact has
    no originating prompt, so local regeneration cannot rebuild it, and no cloud
    model is configured, so a change order cannot be requested.  The message
    names the remedy, so "unavailable" is actionable rather than a dead end.
    """
    payload = {"message": message, "available": available}
    if stream:
        async def _mu_stream():
            yield _sse_chunk(content="", metadata={"modify_unavailable": payload})
            yield "data: [DONE]\n\n"
        return StreamingResponse(_mu_stream(), media_type="text/event-stream")
    return JSONResponse(content={"modify_unavailable": payload})


def _synthesize_modify_combo(artifact: Any, current_request: str) -> str | None:
    """Build the combined prompt for a local regeneration (PATH A).

    Returns None when the artifact carries no originating prompt, which cannot
    happen on PATH A but is guarded anyway so a race (the current artifact
    changing underneath us) degrades to "no combo" rather than a crash.
    """
    try:
        from .provenance import synthesize_combo_prompt as _scp
    except ImportError:
        return None
    if artifact is None:
        return None
    return _scp(
        getattr(artifact, "origin_prompt", None),
        getattr(artifact, "modifications", None),
        current_request or "",
    )


def _emit_vision_unavailable_response(work: Any, model_id: str, stream: bool):
    """Report that the selected target is not declared vision-capable."""
    alert = {
        "code": "vision_unavailable",
        "severity": "error",
        "model_id": model_id,
        "message": (
            f"The selected model '{model_id}' is not declared vision-capable. "
            "The image was not sent to a text-only model."
        ),
        "domain": getattr(work, "domain", None),
        "task_type": getattr(work, "task_type", None),
    }
    if stream:
        async def _vision_stream():
            yield _sse_chunk(content="", metadata={"alert": alert})
            yield "data: [DONE]\n\n"
        return StreamingResponse(_vision_stream(), media_type="text/event-stream")
    return JSONResponse(status_code=422, content={"alert": alert})


async def _stream_trivial_answer(app_state: CleanState, answer: str, decision: Any):
    """Stream a trivial-prompt direct answer as SSE."""
    async def _trivial_stream():
        yield _sse_chunk(None, model=app_state.router.router_model_id, role="assistant",
                         metadata={"coe_routing": {
                             "trivial": True,
                             "task_type": decision.task_type,
                             "domain": decision.domain,
                             "stage": "trivial_start",
                         }})
        yield _sse_chunk(answer, model=app_state.router.router_model_id)
        yield _sse_chunk(None, model=app_state.router.router_model_id, finish_reason="stop",
                         metadata={"coe_routing": {
                             "domain": decision.domain,
                             "task_type": decision.task_type,
                             "task_scope": decision.task_scope,
                             "trivial": True,
                         }})
        yield "data: [DONE]\n\n"
    return StreamingResponse(
        _trivial_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


async def _stream_trivial_error(app_state: CleanState, error: str, decision: Any):
    """Stream a trivial-answer failure as a visible SSE terminal event."""
    async def _trivial_error_stream():
        yield _sse_chunk(None, model=app_state.router.router_model_id, role="assistant",
                         metadata={"coe_routing": {
                             "trivial": True,
                             "domain": decision.domain,
                             "task_type": decision.task_type,
                             "stage": "trivial_failed",
                             "error": error,
                         }})
        yield _sse_chunk(f"[trivial answer failed: {error}]",
                         model=app_state.router.router_model_id)
        yield _sse_chunk(None, model=app_state.router.router_model_id, finish_reason="stop",
                         metadata={"coe_routing": {
                             "trivial": True,
                             "stage": "trivial_failed",
                             "error": error,
                         }})
        yield "data: [DONE]\n\n"
    return StreamingResponse(
        _trivial_error_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Step 5.10/6.1/6.2/6.4: post-freeze artifact commit
# ---------------------------------------------------------------------------

def _build_coe_artifact_dict(artifact: Any) -> dict[str, Any]:
    """Render an Artifact as the `coe_artifact` response block (6.1)."""
    return {
        "artifact_id": artifact.artifact_id,
        "session_number": artifact.session_number,
        "version": artifact.version,
        "parent_id": artifact.parent_id,
        "descriptor": artifact.descriptor,
        "language": artifact.language,
        "origin": artifact.origin,
        "saved_path": artifact.saved_path,
    }


def _commit_pending_artifact(
    state: CleanState, work: Any, decision: Any, result: Any,
    prompt: str = "",
    evaluate_prior_wp: bool = False,
) -> tuple[str, dict[str, Any] | None]:
    """Commit any artifact this turn produced, strictly after
    executor.execute() has returned -- i.e. strictly after
    SessionMemory has already frozen this turn on the runtime's
    `complete` event (freeze-before-artifact, Step 5.9/5.10).

    Shared by both the non-streaming and streaming response paths.
    Returns ``(output_text, coe_artifact_dict_or_None)``.
    """
    output_text = result.output
    coe_artifact: dict[str, Any] | None = None

    if result.passed and getattr(result, "modify_was_regenerated", False) \
            and result.extracted_artifact_content:
        # ============================================================
        # PATH A commit (spec 5.7.2): LOCAL REGENERATION
        # ============================================================
        # A regeneration produces a COMPLETE artifact rather than a change
        # order, so there is no pending_artifact_edit to apply and no
        # deterministic re-integration step.  It is still a MODIFICATION of an
        # existing lineage, so:
        #   * the parent's origin_prompt is INHERITED UNCHANGED (never
        #     rewritten -- a daughter holding the combined prompt would drop
        #     every earlier modification on the next regeneration), and
        #   * this request is APPENDED to the modifications log.
        # Naming keeps the lineage hierarchy (version bumps within the same
        # session/descriptor) because this IS the same artifact, modified.
        _parent = state.artifact_store.current
        _reg_lang = result.extracted_artifact_language or (
            getattr(_parent, "language", None) if _parent else None
        )
        _reg_desc = (
            getattr(_parent, "descriptor", "")
            or _slugify(getattr(decision, "user_intent", "") or prompt or "untitled")
        )
        # append this change to the inherited log (method="regenerated")
        _reg_mods = list(getattr(_parent, "modifications", None) or [])
        try:
            from .provenance import record_modification as _rm
            _reg_mods = _rm(
                _reg_mods, prompt or "", "regenerated",
                result.selected_model or getattr(work, "primary_model", "") or "",
            )
        except ImportError:
            pass
        new_id = state.artifact_store.create(
            result.extracted_artifact_content,
            language=_reg_lang, origin="generated",
            produced_by_model=result.selected_model or work.primary_model,
            produced_by_domain=getattr(work, "domain", None),
            descriptor=_reg_desc,
            # origin_prompt INHERITED UNCHANGED (never the combined prompt):
            # a daughter holding the combo would drop every earlier
            # modification on the next regeneration.
            origin_prompt=(getattr(_parent, "origin_prompt", None) if _parent else None),
            modifications=_reg_mods,
            descendant_of=_parent,
            # Matches apply_edits(): only used to build the descriptor when
            # the parent is an unpromoted staging entry.
            user_intent=str(getattr(decision, "user_intent", "") or ""),
        )
        child = state.artifact_store.get(new_id)
        save_artifact_to_disk(child)
        coe_artifact = _build_coe_artifact_dict(child)
        state.session_memory.attach_artifact_to_last_turn(new_id)
        output_text = child.content
    elif result.passed and result.pending_artifact_edit:
        # modify_artifact: runtime.py only validated and packaged the
        # edit (Step 5.9); the real ArtifactStore commit happens here.
        pe = result.pending_artifact_edit
        # Spec 5.7.3: record HOW this modification was served, so the fork can
        # append it to the lineage log.  The origin prompt itself is INHERITED
        # unchanged by apply_edits -- this call must never rewrite it.
        _mod_method = ("regenerated"
                       if getattr(result, "modify_was_regenerated", False)
                       else "cloud_change_order")
        edit_metadata = state.artifact_store.apply_edits(
            pe["edits"], artifact_id=pe["artifact_id"],
            user_intent=pe.get("user_intent", ""),
            source_filename=pe.get("source_filename"),
            modification_request=prompt or None,
            modification_method=_mod_method,
            modification_model=(result.selected_model
                                or getattr(work, "primary_model", None)),
        )
        if edit_metadata.get("errors"):
            # A functional failure discovered only at commit time
            # (should be rare -- R1/R2 contracts already validated
            # anchors -- but apply_edits() is the deterministic
            # ground truth).
            result.passed = False
            result.error = f"bad_anchor: reintegration errors: {edit_metadata['errors']}"
            output_text = result.error
        else:
            new_id = edit_metadata["new_artifact_id"]
            child = state.artifact_store.get(new_id)
            # fork-not-mutate: the parent is still exactly the
            # pre-edit content, even after the commit above.
            parent = state.artifact_store.get(pe["artifact_id"])
            output_text = (
                build_diff_response(parent, child, edit_metadata)
                if child.saved_path else child.content
            )
            coe_artifact = _build_coe_artifact_dict(child)
            state.session_memory.attach_artifact_to_last_turn(new_id)
    elif result.passed and result.extracted_artifact_content:
        # new_code_generation: use the harness's own contract-
        # validated extraction (Step 5.7), not a regex re-derivation.
        # The language is the fence language the contract detected
        # (Option B), so the artifact gets the correct extension
        # (e.g. python -> .py, html -> .html).  Fall back to the
        # domain-derived tag if the contract didn't stash one.
        language = result.extracted_artifact_language or (
            getattr(work, "domain", "").removeprefix("domain_") or None
        )
        # Feature 5 (2026-09-01): descriptor falls back to the raw user
        # prompt when the router returns an empty user_intent (e.g. a
        # full routing_override bypasses the router, so no intent is
        # synthesized).  ``untitled`` is the last resort.
        descriptor = _slugify(
            getattr(decision, "user_intent", "") or prompt or "untitled"
        )
        new_id = state.artifact_store.create(
            result.extracted_artifact_content,
            language=language, origin="generated",
            produced_by_model=result.selected_model or work.primary_model,
            produced_by_domain=getattr(work, "domain", None),
            descriptor=descriptor,
            # Spec 5.7.3: the ORIGIN PROMPT is the user's VERBATIM request
            # (not the router's restated user_intent, which is a paraphrase
            # and can drop the specific parameters that define the artifact).
            # Recorded once here and inherited unchanged by every descendant.
            origin_prompt=prompt or None,
        )
        child = state.artifact_store.get(new_id)
        save_artifact_to_disk(child)  # create()'s "generated" branch doesn't persist itself
        coe_artifact = _build_coe_artifact_dict(child)
        state.session_memory.attach_artifact_to_last_turn(new_id)
        # new_code_generation always returns full content -- no diff
        # format for this path (only modify_artifact gets a diff).
    elif result.passed and result.general_use_work_product_content is not None:
        # general_use persists the visible response as Markdown unless
        # the response contains a complete code fence, in which case the
        # last closed fence is the work product.
        language = result.general_use_work_product_language or "markdown"
        descriptor = _slugify(
            getattr(decision, "user_intent", "") or prompt or "untitled"
        )
        new_id = state.artifact_store.create(
            result.general_use_work_product_content,
            language=language, origin="generated",
            produced_by_model=result.selected_model or work.primary_model,
            produced_by_domain=getattr(work, "domain", None),
            descriptor=descriptor,
            # Spec 5.7.3: verbatim user request is the origin prompt.
            origin_prompt=prompt or None,
        )
        child = state.artifact_store.get(new_id)
        save_artifact_to_disk(child)
        coe_artifact = _build_coe_artifact_dict(child)
        state.session_memory.attach_artifact_to_last_turn(new_id)

    # 6.4: evaluate_artifact holding pen (2026-09-01).  Persist the source
    # object + WP to the single-slot evaluate_object_store so a follow-up
    # turn ("fix the bugs you found") has the source context.  The source is
    # whatever was analyzed (attached file / pasted content / stored
    # artifact); the WP is the evaluate report.  Only a NEW evaluate_artifact
    # task replaces the pen.
    if (
        result.passed
        and getattr(work, "harness", None) is not None
        and getattr(work.harness, "harness_id", None) == "evaluate_artifact"
    ):
        try:
            _src = None
            _src_lang = None
            _src_fn = None
            if evaluate_prior_wp and state.evaluate_store is not None and state.evaluate_store.has():
                _pen = state.evaluate_store.get() or {}
                _src = _pen.get("source_content")
                _src_lang = _pen.get("source_language")
                _src_fn = _pen.get("source_filename")
            else:
                _cur = state.artifact_store.get(None) if state.artifact_store else None
                if _cur is not None:
                    _src = _cur.content
                    _src_lang = _cur.language
                    _src_fn = getattr(_cur, "source_filename", None)
            if _src is None:
                _src = result.output or ""
            if state.evaluate_store is not None:
                state.evaluate_store.put(
                    source_content=_src,
                    source_language=_src_lang,
                    source_filename=_src_fn,
                    wp_content=result.output or "",
                    wp_language=None,
                )
                print(f"[evaluate_store] persisted source ({len(_src)}c) + WP ({len(result.output or '')}c)",
                      file=sys.stderr, flush=True)
        except Exception as _exc:
            print(f"[evaluate_store] put failed: {_exc}", file=sys.stderr, flush=True)

    # 6.4: generalist-fallback notice.  Keyed on result.fallback_kind
    # (set only by runtime.py's _invoke_t5() on a passing attempt),
    # NOT on work.resolution_kind == "generalist" -- that value also
    # fires for ordinary direct routing to a generalist-only domain,
    # which is not a fallback event.
    if result.fallback_kind == "generalist_zero_shot" and result.passed:
        output_text = (
            "_Note: no specialist registered for this domain/task; the "
            "generalist model answered._\n\n"
        ) + output_text

    return output_text, coe_artifact


async def _resolve_slot_fallbacks(app_state: Any, work: Any, transport: Any) -> tuple[Any, list[dict[str, str]]]:
    """Proactively check the resolved work's slots and substitute the generalist.

    Returns ``(work, fallbacks)`` where ``work`` is a possibly-replaced
    ResolvedWork (frozen dataclass -> built via ``replace``) and
    ``fallbacks`` is a list of ``slot_fallback`` notification dicts (one
    per slot whose model is not registered/available).  For each such
    slot, the model is replaced with the generalist so the harness still
    executes (harness-preserving) using the generalist model through the
    same harness framework the missing model would have used.

    The router and the generalist themselves are HARD-STOP roles: they
    are not substituted here.
    """
    fallbacks: list[dict[str, str]] = []
    gen_model = getattr(work, "generalist_model", None)
    if not gen_model:
        gen_model = app_state.cfg["models"]["general_fallback"]["model_id"]
    gen_provider = getattr(work, "generalist_provider", None) or "ollama"

    updates: dict[str, Any] = {}
    slots = [
        ("primary", "primary_model", "primary_provider", "primary_options"),
        ("secondary", "secondary_model", "secondary_provider", "secondary_options"),
    ]
    for role, model_attr, prov_attr, opt_attr in slots:
        model_id = getattr(work, model_attr, None)
        if not model_id:
            continue
        if model_id == gen_model:
            continue
        available = await transport.model_available(model_id)
        if available:
            continue
        # Slot unavailable -> substitute generalist (harness-preserving).
        updates[model_attr] = gen_model
        updates[prov_attr] = gen_provider
        if opt_attr and hasattr(work, opt_attr):
            updates[opt_attr] = dict(getattr(work, opt_attr) or {})
        fallbacks.append({
            "role": role,
            "slot_model": model_id,
            "reason": "model_not_registered",
            "fallback": gen_model,
        })
    if updates:
        from dataclasses import replace
        work = replace(work, **updates)
    return work, fallbacks


# --- Cancel-turn registry (2026-09-07) ---
# In-flight executor tasks keyed by the client's X-Request-Id header.
# /v1/cancel cancels the task; the runtime records the attempt as
# `cancelled` and preserves session/artifact state (observed semantics:
# "[pathology] ... err=cancelled ... preserved").  The SSE stream drains
# and closes via the queue, so the client prompt returns cleanly.
_ACTIVE_REQUEST_TASKS: dict[str, asyncio.Task] = {}


def _resolve_legacy_route(request: Any, metadata: dict[str, Any]) -> bool:
    """Effective `legacy_route` for this request, honouring the documented
    precedence chain (same shape as `think`):

      1. routing_override.legacy_route  (slash form OR body-level override)
      2. metadata.legacy_route          (per-request JSON override)
      3. False                          (origin-gated route is the default)

    2026-09-18 (B2 FIX): extracted into a FUNCTION because it has to be
    evaluated TWICE.  It was originally inlined only near the top of
    chat_completions (so the attachment handler and the routing metadata
    could read it), but `_parse_slash_command` runs LATER and is what sets
    `override.legacy_route` for the `/task-<name>-legacy` suffix.  The result
    was that the SLASH form never populated the variable while BOTH
    programmatic forms did -- precisely inverted from spec V13, which
    requires `/task-modify-legacy` and
    `routing_override.legacy_route=true` to produce identical decisions:
    the CLI command silently ran the ORIGIN-GATED route instead of legacy.

    Idempotent by construction, so calling it again after the slash merge is
    safe; the same precedence means the earlier call's value is unchanged
    whenever no slash directive was present.
    """
    if (getattr(request, "routing_override", None) is not None
            and request.routing_override.legacy_route is not None):
        return bool(request.routing_override.legacy_route)
    if metadata.get("legacy_route") is not None:
        return bool(metadata["legacy_route"])
    return False


@app.post("/v1/chat/completions")
async def chat_completions(
    request: CleanCompletionRequest,
    raw_request: Request,
) -> JSONResponse:
    """OpenAI-compatible chat completions endpoint.

    Accepts a standard OpenAI chat completion request with an
    optional ``metadata`` field for runtime overrides:

    * (corrected 2026-10-12: ``num_ctx`` was listed here as a context-window override; no code reads ``metadata.num_ctx``; the context window travels per-harness in the sampling block)
    * ``closed_think`` — force closed-think on/off
    * ``clear_session`` — clear memory before this request
    * ``no_inject`` — disable memory injection for this request

    (A ``timeout_s`` per-problem wall-clock cap used to be documented here.
    It has never been implemented and is NOT read anywhere; the removal is
    deliberate rather than an omission -- see the module docstring.)

    Returns a standard OpenAI chat completion response.  The
    ``coe_routing`` field in the choice carries diagnostic
    routing information (which domain, harness, and model were
    selected internally).
    """
    if state is None:
        raise HTTPException(status_code=503, detail="Clean slice is not initialized")
    if not request.messages:
        raise HTTPException(status_code=400, detail="At least one message is required")

    # A clarification is a persistent side-channel interaction.  The
    # next request supplies only the user's artifact choice; restore the
    # original request and continue it after resolving that choice.
    pending = state.pending_clarification
    if pending is not None:
        answer = _request_user_text(request.messages)
        selection = request.artifact_identity_resolution or _resolve_clarification_answer(
            answer,
            pending["candidates"],
            state.artifact_store,
        )
        if selection == "abort":
            state.pending_clarification = None
            return JSONResponse(
                status_code=200,
                content={"status": "aborted", "reason": "user requested abort"},
            )
        if selection is None:
            return _emit_clarification_response(
                NeedsClarification(
                    question=pending["question"],
                    candidates=pending["candidates"],
                ),
                request.stream,
            )
        restored = deepcopy(pending["request"])
        restored.artifact_identity_resolution = selection
        state.pending_clarification = None
        request = restored

    metadata = request.metadata or {}
    _seed_override = metadata.get("seed")
    if _seed_override is not None and (
        isinstance(_seed_override, bool) or not isinstance(_seed_override, int)
    ):
        return JSONResponse(
            status_code=422,
            content={"error": "metadata.seed must be an integer"},
        )

    # 2026-09-14: effective `legacy_route` for this request -- the API mirror
    # of the /task-<name>-legacy suffix.  Same precedence chain as `think`:
    #   1. routing_override.legacy_route (slash form OR body-level override)
    #   2. metadata.legacy_route (per-request JSON override)
    #   3. default False (the origin-gated route is the default)
    # Exposed so a programmatic driver can reproduce CLI behaviour exactly.
    #
    # DEFINED HERE, and it MUST stay above every use.  It was originally
    # resolved further down, next to the origin-gated routing block; the
    # attachment handler (S3) and the routing metadata both read it, so an
    # earlier revision that defined it lower was a NameError on EVERY
    # request -- which surfaces to the client as an opaque HTTP 500.
    # Rule: this resolution must precede the attachment block.
    _legacy_route: bool = _resolve_legacy_route(request, metadata)

    # Optional: clear session memory before this request
    if metadata.get("clear_session"):
        state.session_memory.clear()

    # --- Step 1: artifact attachment (if present) ---
    if request.artifact:
        # 2026-09-16 (DIAGNOSTIC): log the language AS RECEIVED.  `create()`
        # stores the supplied tag verbatim (its auto-detect only runs when
        # language is None), and `apply_edits` later inherits it, so a wrong
        # tag here becomes a wrong file extension at the end of the chain
        # with no error anywhere in between.  This prints the value at the
        # boundary so a mis-tag is visible in the trace.
        print(
            f"[artifact] attach: language={request.artifact.language!r} "
            f"origin={request.artifact.origin!r} "
            f"filename={getattr(request.artifact, 'filename', None)!r} "
            f"chars={len(request.artifact.content or '')}",
            file=sys.stderr, flush=True,
        )
        state.artifact_store.create(
            request.artifact.content,
            language=request.artifact.language,
            origin=request.artifact.origin,
            produced_by_model=None,
            # Spec 2.3: an externally supplied artifact may carry its
            # originating prompt.  Absent -> None -> cloud-change-order route
            # only, which is the pre-existing behaviour.
            origin_prompt=getattr(request.artifact, "origin_prompt", None),
            # 2026-09-14 FIX: the attachment's original filename was NOT being
            # recorded, even though ArtifactAttachment has carried `filename`
            # since Step 5 and `create()` accepts `source_filename`.  The
            # consequence was a silently degraded descriptor: promotion
            # (apply_edits) builds `modify-<filename-slug>-<intent-slug>` but
            # with no filename recorded it fell through to
            # `modify-<intent-slug>`, so every attached file produced an
            # indistinguishable `modify-untitled` lineage instead of
            # `modify-sim`.  The downstream threading (execute_kwargs
            # ["source_filename"] at the modify dispatch, and apply_edits'
            # `eff_filename = source_filename or artifact.source_filename`)
            # was already in place and waiting for this value.
            #
            # 2026-09-16 (S3) WITHHOLDING -- RETIRED 2026-09-18.
            # The legacy route previously withheld this filename HERE AND at
            # the execute_kwargs thread below, restoring the pre-session
            # descriptor `modify-<intent-slug>`.  It had to be both sites:
            # apply_edits resolves
            #   `eff_filename = source_filename or artifact.source_filename`
            # so withholding only the thread left the fallback reading the
            # value stored here, and the legacy name was NOT restored.
            #
            # Retired because naming is a POST-SHARED-WORKFLOW function --
            # artifact_store.create() Step 5 -- not a property of the route
            # that produced the daughter.  A denovo regeneration and a cloud
            # change order already share it; the legacy change-order route
            # makes the same product and must share it too, or an identical
            # modification of an identical file lands in two different
            # lineages depending only on which route served it.
            #
            # Comparability of the reference instrument is preserved a
            # different way: /task-modify-legacy is still the single-tier
            # change-order path (no `rounds:` key), so what is comparable over
            # time is the GENERATION CONTRACT, not its output's filename.
            # Route identity remains available as `coe_routing.harness_id`.
            #
            # Reversal: restore the `not _legacy_route` condition HERE and at
            # the execute_kwargs thread below -- both, never one.
            source_filename=request.artifact.filename,
        )

    # --- Slash command parsing ---
    # Slash directives (/domain-*, /task_type-*, /thinkon, /thinkoff) and a
    # body-level routing_override are EQUIVALENT override paths, so both
    # should work the same.  Previously a slash override was dropped
    # entirely whenever a body routing_override existed
    # (`if slash_override and not request.routing_override`), which made
    # /thinkon silently no-op whenever a client sent
    # routing_override={domain,task_type}.  Fix: merge field-by-field —
    # the slash directive fills only the fields the body override left unset,
    # so /thinkon works alongside domain/task_type overrides (and the
    # explicit body override wins for any field it did set).
    slash_override, cleaned_messages = _parse_slash_command(request.messages)
    if slash_override is not None:
        if request.routing_override is None:
            request.routing_override = slash_override
            request.messages = cleaned_messages
        else:
            merged = request.routing_override
            if merged.think is None and slash_override.think is not None:
                merged.think = slash_override.think
            if merged.domain is None and slash_override.domain is not None:
                merged.domain = slash_override.domain
            if merged.task_type is None and slash_override.task_type is not None:
                merged.task_type = slash_override.task_type
            # 2026-09-14: the /task-<name>-legacy marker merges by the same
            # rule (slash fills only what the body override left unset).
            if (merged.legacy_route is None
                    and slash_override.legacy_route is not None):
                merged.legacy_route = slash_override.legacy_route
            request.routing_override = merged
            request.messages = cleaned_messages
        # 2026-09-18 (B2 FIX): RE-RESOLVE.  `_legacy_route` was computed at the
        # top of this function, before the slash merge above, so a slash-only
        # `/task-<name>-legacy` never reached it.  Same precedence, so this is
        # a no-op whenever no slash directive carried the flag.
        _legacy_route = _resolve_legacy_route(request, metadata)

    # 2026-09-01: /evaluate — seed the evaluate_object_store with the
    # immediately prior turn's WP as the object to evaluate.  This lets a
    # general_use WP (or any prior harness WP except evaluate_artifact) be
    # fed to the evaluate_artifact harness in the next turn.  If the prior
    # turn was itself an evaluate_artifact, reject with a clear message.
    if (
        request.routing_override is not None
        and getattr(request.routing_override, "evaluate_prior_wp", False)
    ):
        _prior = state.session_memory.get_last_turn()
        _prior_tt = getattr(_prior, "task_type", None) if _prior else None
        if _prior_tt == "evaluate_artifact":
            return _emit_override_invalid_response(
                _make_override_invalid_work(
                    domain=request.routing_override.domain or "domain_other",
                    task_type="evaluate_artifact",
                    bad="evaluate_prior_wp",
                    allowed=["any non-evaluate_artifact prior turn"],
                ),
                request.stream,
            )
        _prior_wp = getattr(_prior, "model_output", "") if _prior else ""
        if _prior_wp and state.evaluate_store is not None:
            state.evaluate_store.put(
                source_content=_prior_wp,
                source_language=None,
                source_filename=None,
                wp_content="",  # WP filled by the evaluate turn itself
                wp_language=None,
            )
            print(f"[evaluate] /evaluate seeded store with prior WP ({len(_prior_wp)} chars)",
                  file=sys.stderr, flush=True)

    # --- Override validation (before the resolver is called) ---
    # Validates the user's override against the closed vocabularies.
    # An unsupported override blocks the turn with a generic echo.
    if request.routing_override:
        vocab = state.resolution.closed_vocabulary()
        if request.routing_override.domain and request.routing_override.domain not in vocab["domains"]:
            return _emit_override_invalid_response(
                _make_override_invalid_work(
                    domain=request.routing_override.domain,
                    task_type=request.routing_override.task_type or "general_use",
                    bad=f"domain={request.routing_override.domain}",
                    allowed=sorted(vocab["domains"]),
                ),
                request.stream,
            )
        if request.routing_override.task_type and request.routing_override.task_type not in vocab["task_types_reach"]:
            return _emit_override_invalid_response(
                _make_override_invalid_work(
                    domain=request.routing_override.domain or "domain_other",
                    task_type=request.routing_override.task_type,
                    bad=f"task_type={request.routing_override.task_type}",
                    allowed=sorted(vocab["task_types_reach"]),
                ),
                request.stream,
            )

    # --- Prompt extraction (multi-content aware) ---
    def _extract_text(content: str | list[ContentPart]) -> str:
        if isinstance(content, str):
            return content
        return " ".join(p.text for p in content if p.type == "text" and p.text)

    prompt = "\n".join(
        _extract_text(m.content) for m in request.messages if m.role == "user"
    ) or _extract_text(request.messages[-1].content)

    # NOTE: `_legacy_route` is resolved ABOVE, immediately after `metadata`,
    # because the artifact-attachment handler also reads it.  Do not re-declare
    # it here -- a second definition below the first use is exactly the
    # NameError that caused an HTTP 500 on every request.

    # Normalize image bytes once, outside the router package.  The first
    # classifier call receives only the metadata below; images are threaded
    # into the selected answer path after classification.
    image_references: list[tuple[Any, str | None]] = []
    for message in request.messages:
        if isinstance(message.content, list):
            for part in message.content:
                if part.type == "image_url" and part.image_url:
                    image_references.append((part.image_url, None))
    for attachment in request.context_attachments:
        if attachment.type == "image" and attachment.content:
            image_references.append((attachment.content, attachment.mime))
    images: list[str] = []
    image_details: list[tuple[str, str, int, int | None, int | None]] = []
    for reference, mime_hint in image_references:
        normalized = await _normalize_image_reference(reference, mime_hint)
        if normalized is None:
            raise HTTPException(
                status_code=400,
                detail="invalid_image_attachment: expected a valid image data URI, base64 value, or URL under 10 MiB",
            )
        images.append(normalized)
        mime, size_bytes, width, height = _canonical_image_details(normalized)
        image_details.append((normalized, mime, size_bytes, width, height))

    # --- Build attachment metadata for the router ---
    router_attachments: list[AttachmentMeta] = []
    for _image, mime, size_bytes, width, height in image_details:
        router_attachments.append(AttachmentMeta(
            kind="image", mime=mime or "image/*", size_bytes=size_bytes,
            width=width, height=height,
        ))
    for ca in request.context_attachments:
        if ca.type == "image":
            continue
        router_attachments.append(AttachmentMeta(
            kind=ca.type, mime=ca.mime, language=ca.language,
            size_bytes=len(ca.content) if ca.content else 0,
        ))
    if request.artifact:
        art = state.artifact_store.current
        router_attachments.append(AttachmentMeta(
            kind="working_artifact", language=request.artifact.language,
            size_bytes=len(request.artifact.content),
            section_names=state.artifact_store.get_section_names() if art else [],
        ))

    # --- Artifact identity resolution (from clarification side-trip) ---
    artifact_identity_resolution = request.artifact_identity_resolution
    if artifact_identity_resolution == "abort":
        return JSONResponse(
            status_code=200,
            content={"status": "aborted", "reason": "user requested abort"},
        )
    if artifact_identity_resolution == "none":
        artifact_identity_for_router = "none"
    elif artifact_identity_resolution:
        state.artifact_store.switch(artifact_identity_resolution)
        artifact_identity_for_router = artifact_identity_resolution
    else:
        artifact_identity_for_router = None

    # --- Prior-turn artifact status flag ---
    prior_turn_produced_artifact = False
    last_turn = state.session_memory.get_last_turn()
    if last_turn:
        prior_turn_produced_artifact = bool(
            getattr(last_turn, "produced_artifact", False)
        )

    # --- Session memory hints ---
    routing_hints: dict[str, Any] = {}
    if last_turn and last_turn.routing_decision:
        prior_domain = last_turn.routing_decision.get("domain")
        prior_task_type = last_turn.routing_decision.get("task_type")
        if prior_domain:
            routing_hints["prior_domain"] = prior_domain
        if prior_task_type:
            routing_hints["prior_task_type"] = prior_task_type

    # --- Name-based artifact resolution (reverse-safe) ---
    # The user may NAME the artifact they want to modify (e.g. "the
    # original kanban file", "kanban_demo.html").  When the prompt
    # uniquely resolves to a single artifact by name, switch to it as
    # the working artifact so the modify/evaluate harness operates on
    # the right one.  If the name is ambiguous (multiple matches), the
    # router's clarification side-channel fires later with the list.
    # This runs only when the user did NOT already attach an artifact
    # in this request (an explicit attachment wins).
    if not request.artifact and not artifact_identity_resolution:
        _name_matches = state.artifact_store.resolve_by_name(prompt)
        if len(_name_matches) == 1:
            state.artifact_store.switch(_name_matches[0])
            print(
                f"[artifact] resolved by name -> {_name_matches[0]}",
                file=sys.stderr,
            )

    _effective_override = request.routing_override

    # --- Router call with failure handling ---
    try:
        decision = await state.router.classify(
            prompt,
            attachments=router_attachments or None,
            hints=routing_hints or None,
            artifact_section_names=state.artifact_store.get_section_names() if state.artifact_store.current else None,
            prior_turn_produced_artifact=prior_turn_produced_artifact,
            routing_override=_effective_override,
            artifact_identity_resolution=artifact_identity_for_router,
        )
    except RouterFailure as exc:
        return JSONResponse(
            status_code=503,
            content={
                "router_failure": {"kind": exc.kind, "reason": str(exc)},
                "routing_options": ["routing_override", "abort"],
                "valid_domains": list(state.resolution.rows.keys()),
                "valid_task_types": [
                    "new_code_generation", "bug_fix",
                    "modify_artifact", "evaluate_artifact",
                    "general_use", "user_defined",
                ],
            },
        )
    except RouterTaskUnsupported as exc:
        # Policy hard stop (2026-09-07): the classified task cannot run
        # on the attached inputs (e.g. bug_fix / modify_artifact with an
        # image-only attachment).  Surface the reason verbatim instead
        # of burning the user's time on a doomed run.
        print(
            f"[router] task unsupported: {exc.message} (task={exc.task_type})",
            file=sys.stderr, flush=True,
        )
        if request.stream:
            async def _unsupported_stream():
                yield _sse_chunk(
                    content=exc.message,
                    metadata={"task_unsupported": {
                        "message": exc.message,
                        "task_type": exc.task_type,
                    }},
                )
                yield "data: [DONE]\n\n"
            return StreamingResponse(_unsupported_stream(), media_type="text/event-stream")
        return JSONResponse(content={
            "task_unsupported": {
                "message": exc.message,
                "task_type": exc.task_type,
            },
        })
    except NeedsClarification as exc:
        state.pending_clarification = {
            "request": deepcopy(request),
            "question": exc.question,
            "candidates": list(exc.candidates),
        }
        return _emit_clarification_response(exc, request.stream)

    # --- Artifact-language domain pin (Option 1: modify/evaluate only) ---
    # When modifying or evaluating an existing artifact, pin the domain
    # to the artifact's provenance unless the user provided an explicit
    # domain override. Unrelated tasks (new_code_generation, general_use,
    # bug_fix) preserve the router's classified domain.
    #
    # Precedence (2026-09-07, user-approved):
    #   1. produced_by_domain (recorded provenance of a WIRED domain) --
    #      strictly more informative than the language map; closes the
    #      case where a domain_physics markdown artifact was reclassified
    #      domain_other because markdown has no language->domain row.
    #   2. artifact language map (existing heuristic, unchanged).
    #   3. classified domain (no pin).
    # "domain_other" provenance NEVER pins (generalist-produced artifacts
    # must not force domain_other on later modifies); artifacts with no
    # provenance (pre-2026-09-07) fall through to the language map.
    _cur_artifact = state.artifact_store.current
    if (
        _cur_artifact is not None
        and getattr(decision, "task_type", None) in ("modify_artifact", "evaluate_artifact")
        and not (request.routing_override and request.routing_override.domain)
    ):
        _provenance_domain = getattr(_cur_artifact, "produced_by_domain", None)
        _artifact_domain = (
            _provenance_domain
            if (_provenance_domain and _provenance_domain != "domain_other"
                and _provenance_domain in state.resolution.rows)
            else _artifact_domain_override(_cur_artifact.language)
        )
        if _artifact_domain and _artifact_domain != decision.domain:
            print(
                f"[router] artifact domain pin: {decision.domain} -> "
                f"{_artifact_domain} (provenance="
                f"{_provenance_domain or 'none'}, language="
                f"{_cur_artifact.language})",
                file=sys.stderr, flush=True,
            )
            decision = replace(decision, domain=_artifact_domain)

    # --- Routing-change session-context reset (reverse-safe) ---
    # When either the newly-classified domain or task type differs
    # from the prior turn, the ephemeral session context (KV cache /
    # conversation memory) is stale and must be cleared.  A same-
    # domain transition from modify_artifact to evaluate_artifact is
    # significant too: the evaluator must not inherit modification
    # intent or edit markers from the prior turn.
    #
    # The ARTIFACT STORE is NEVER cleared here.  Artifacts are
    # user-owned working state (explicitly loaded via
    # /v1/artifact/load) and are cleared only by an explicit user
    # action.  Auto-clearing them would nuke a freshly-loaded artifact
    # the user intends to modify, leaving the model nothing to work on.
    _new_domain = getattr(decision, "domain", None)
    _new_task_type = getattr(decision, "task_type", None)
    _prior_domain = routing_hints.get("prior_domain")
    _prior_task_type = routing_hints.get("prior_task_type")
    _domain_changed = bool(
        _new_domain and _prior_domain and _new_domain != _prior_domain
    )
    _task_type_changed = bool(
        _new_task_type
        and _prior_task_type
        and _new_task_type != _prior_task_type
    )
    if _domain_changed or _task_type_changed:
        state.session_memory.clear()
        print(
            f"[context] routing change "
            f"{_prior_domain}/{_prior_task_type} -> "
            f"{_new_domain}/{_new_task_type}: cleared session memory "
            f"(KV cache); artifact store preserved",
            file=sys.stderr,
        )

    # --- Turn record + progress (2026-09-19, PLAN_progress_mechanism.md
    # phase 4) -----------------------------------------------------------
    # Resolved ONCE, before the transport split, so the streaming and
    # non-streaming paths agree on the id and on the record.  A client-supplied
    # X-Request-Id wins; otherwise the server MINTS one and returns it in the
    # response's `coe_turn` block, so a caller that did not supply a header can
    # still poll GET /v1/turn/{id}/progress and cancel via /v1/cancel.
    #
    # PLACEMENT IS LOAD-BEARING: this sits ABOVE the trivial short-circuit
    # below, because that branch RETURNS EARLY and previously created no record
    # at all -- a trivial turn was therefore invisible to /v1/turn/.../progress
    # and unpollable.  Caught live: a non-streaming "Say pong." returned 200
    # with no coe_turn and a 404 on the progress endpoint.
    _turn_id = str(raw_request.headers.get("x-request-id") or "") or uuid.uuid4().hex
    _turn_rec = _register_turn(_turn_id)
    # Phase 6: the retained-turn bound is configurable; phase 1's default (64)
    # holds when the key is absent or unusable.
    try:
        _keep = (state.cfg or {}).get("turn_record_keep")
        if _keep is not None and int(_keep) > 0:
            _evict_turns(keep=int(_keep))
    except (TypeError, ValueError):
        pass
    # A trivial turn finishes in one call, so mark it terminal as soon as its
    # answer is resolved (below); the record is created here so it EXISTS even
    # if that branch returns first.

    # --- Trivial-prompt direct answer (unified compose/dispatch) ---
    # The trivial decision composes its package through the SAME
    # CleanHarness.compose()/tier machinery as every other path. The
    # config-declared trivial_answer harness owns the policy; this
    # branch only orchestrates dispatch, fallback, and observability.
    if decision.trivial:
        if not images or state.router.is_vision_capable():
            async def _run_trivial(work_for_trivial: Any) -> tuple[bool, str, str]:
                """Dispatch one trivial attempt through the executor.

                Returns (ok, answer, error). An empty visible output is
                NOT a success: it triggers the configured fallback.
                """
                try:
                    result = await state.executor.execute(
                        work_for_trivial,
                        prompt,
                        # Phase 4 follow-up (caught live): the trivial path had
                        # NO progress callback, so a trivial turn produced a
                        # record with no events.  Wrapped like the main
                        # non-streaming path so trivial turns are observable and
                        # cooperatively cancellable too.  inner=None: this call
                        # has no SSE transport to forward to.
                        progress=_RecordingProgress(None, _turn_rec),
                        images=images,
                        closed_think_override=False,
                        inject_last_turn_override=False,
                    )
                    answer = (result.output or "").strip()
                    if not result.passed or not answer:
                        return False, "", (result.error or "trivial attempt returned empty output")
                    return True, answer, ""
                except Exception as _exc:
                    return False, "", f"{type(_exc).__name__}: {_exc}"

            _triv_error: str | None = None
            _triv_answer: str = ""
            _primary = state.resolution.trivial_work()
            _ok, _triv_answer, _primary_err = await _run_trivial(_primary)
            if not _ok:
                print(
                    f"[router] trivial primary attempt failed ({_primary.primary_model}): "
                    f"{_primary_err}; trying fallback",
                    file=sys.stderr,
                    flush=True,
                )
                _fb = state.resolution.trivial_fallback_work()
                if _fb is not None:
                    _fb_ok, _fb_answer, _fb_err = await _run_trivial(_fb)
                    if _fb_ok:
                        _triv_answer = _fb_answer
                    else:
                        _triv_error = f"primary: {_primary_err}; fallback: {_fb_err}"
                else:
                    _triv_error = _primary_err
            if _triv_error is not None:
                print(f"[router] trivial answer failed: {_triv_error}", file=sys.stderr, flush=True)
                if request.stream:
                    return await _stream_trivial_error(state, _triv_error, decision)
                return JSONResponse(status_code=503, content={
                    "error": "trivial_answer_failed",
                    "detail": _triv_error,
                    "coe_routing": {
                        "domain": decision.domain,
                        "task_type": decision.task_type,
                        "trivial": True,
                    },
                })
            if request.stream:
                return await _stream_trivial_answer(state, _triv_answer, decision)
            else:
                _turn_rec.status = "done"
                _turn_rec.finished_at = time.monotonic()
                return JSONResponse(content={
                    "id": f"coe-clean-{int(time.time() * 1000)}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": _primary.primary_model,
                    "coe_turn": {"request_id": _turn_id, "status": "done",
                                 "aborted": False},
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": _triv_answer},
                        "finish_reason": "stop",
                        "coe_routing": {
                            "domain": decision.domain,
                            "task_type": decision.task_type,
                            "task_scope": decision.task_scope,
                            "trivial": True,
                        },
                    }],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                })
        # A non-VL router must not receive image bytes.  Continuing here lets
        # normal resolution handle the request through the selected harness.

    # Branch-selector think state (per-request, ONLY for choosing the
    # think-state-dependent role ordering).  Precedence: routing_override.think
    # (/thinkon|/thinkoff slash) > metadata.think > metadata.closed_think
    # (legacy; closed_think=False == think-on).  This NEVER changes how a
    # model behaves -- a model's think mode is fixed by its model-card default.
    _branch_think: bool | None = None
    if request.routing_override is not None and request.routing_override.think is not None:
        _branch_think = request.routing_override.think
    elif "think" in metadata and metadata["think"] is not None:
        _branch_think = bool(metadata["think"])
    elif "closed_think" in metadata and metadata["closed_think"] is not None:
        _branch_think = not bool(metadata["closed_think"])

    work = state.resolution.resolve(decision, think_override=_branch_think)

    # --- Proactive slot fallback (graceful, harness-preserving) ---
    # Check the resolved work's model slots against the backend before
    # execution.  Any slot whose model is not registered is substituted
    # with the generalist (same harness framework, not skipped), and a
    # slot_fallback notification is surfaced to the user so they can
    # cancel if they don't want the generalist to answer.  This is the
    # graceful-fallback that replaces the old 404 -> UnboundLocalError
    # crash path.
    _slot_fallbacks: list[dict[str, str]] = []
    if work.resolution_kind in ("specialist", "generalist"):
        try:
            work, _slot_fallbacks = await _resolve_slot_fallbacks(
                state, work, state.executor.transports.get("ollama")
            )
        except Exception as _sf_exc:
            # Never let the fallback check itself break a request.
            print(f"[slot_fallback] check failed (non-fatal): {_sf_exc}", file=sys.stderr)

    # Opt-in A/B harness variant.  The canonical modify_artifact
    # resolution and every other harness remain unchanged unless the
    # caller explicitly requests a registered variant.
    harness_variant = metadata.get("harness_variant")
    if harness_variant is not None:
        if (
            decision.task_type != "modify_artifact"
            or work.resolution_kind != "specialist"
            or not isinstance(harness_variant, str)
            or harness_variant not in state.resolution.harnesses
        ):
            return JSONResponse(
                status_code=400,
                content={
                    "error": "invalid_harness_variant",
                    "harness_variant": harness_variant,
                    # M3: `modify_artifact_focused` was retired during the
                    # modify-harness split (it was an `extends`-based A/B
                    # variant with no domain row using it).
                    # 2026-09-20 correction: the split's first half re-pointed
                    # the row to `modify_harness_legacy`; M4 (applied
                    # 2026-09-20) re-pointed it to `modify_harness`.  The
                    # retirement of `modify_artifact_focused` is unaffected.
                    # No registered variant remains, so the list is empty; the
                    # `harness_variant not in state.resolution.harnesses` check
                    # above is the real gate either way.
                    "allowed": [],
                },
            )
        work = replace(
            work,
            harness=state.resolution.compose_harness_variant(
                harness_variant, domain=work.domain,
            ),
        )

    # 2026-09-28 (user ruling): the image -> general_use HARNESS SWAP used to
    # live here.  It has been REMOVED -- see the full rationale at the top of
    # this module.  An image no longer changes the harness; the domain+task_type
    # harness runs, and its model receives the image bytes.  A model that cannot
    # see is refused by the vision gate further below, not silently
    # re-harnessed.

    # --- general_use override provenance note (spec 8.2, 2026-09-14) ---
    # An explicit /task-genuse escape hatch on an artifact is NOT a
    # modification: it is a rebuild WITHOUT modify lineage, so it deliberately
    # forfeits the naming hierarchy (a denovo artifact must never be mistaken
    # for a modification of the target).  It forfeits the strong guarantee, not
    # the KNOWLEDGE of where the file came from, so a soft one-line note is
    # retained.  Incidental benefit: the denovo artifact does carry the supplied
    # prompt as its origin, so it becomes an ordinary origin-bearing artifact
    # going forward and is locally modifiable with fresh lineage.
    _genuse_override_note: str | None = None
    if (decision.task_type == "general_use"
            and request.routing_override is not None
            and request.routing_override.task_type == "general_use"
            and state.artifact_store.current is not None):
        _src = state.artifact_store.current
        _genuse_override_note = (
            f"derived from artifact {getattr(_src, 'artifact_id', '?')} "
            f"via general_use override; this is a denovo build, not a "
            f"modification, so modify lineage naming does not apply"
        )
        print(f"[modify] general_use override on artifact "
              f"{getattr(_src, 'artifact_id', '?')} -- denovo, no lineage",
              file=sys.stderr, flush=True)

    # --- Step 3 dispatch on resolution_kind ---
    # Three terminal-state kinds short-circuit before execution:
    #   - rejection: scope not SDST (server-side message, no model)
    #   - user_defined_unavailable: user_defined override missing (BLOCKING)
    #   - override_invalid: user supplied an unsupported override (BLOCKING)
    # The remaining two kinds (specialist, generalist) proceed normally.
    if work.resolution_kind == "rejection":
        return _emit_rejection_response(work, request.stream)
    if work.resolution_kind == "user_defined_unavailable":
        return _emit_user_defined_unavailable_response(work, request.stream)
    if work.resolution_kind == "override_invalid":
        return _emit_override_invalid_response(work, request.stream)

    # --- Execute with optional overrides ---
    # 2026-09-14 FIX (live integration test, spec 5.1): `execute_kwargs` MUST
    # be declared BEFORE the origin-gated routing block below, because that
    # block WRITES the PATH A plumbing into it
    # (modify_combo_prompt / modify_skip_reintegration / the declared
    # regeneration cell's contract and wording).
    #
    # It was previously declared AFTER the block.  Because the block is wrapped
    # in a broad `except Exception` (a deliberate fail-soft guard), the
    # resulting UnboundLocalError was CAUGHT AND DISCARDED -- so the routing
    # metadata still reported `modify_path="regenerated"` while PATH A never
    # actually ran.  A silent no-op that looks correct in telemetry is the
    # worst possible failure mode: every unit test of the individual pieces
    # passes, and only an end-to-end request through the real route exposes it.
    # This is the second defect of this exact class in this feature (the first
    # was `_legacy_route` being read before assignment, which produced an HTTP
    # 500 on every request); both were declaration-order faults.
    execute_kwargs: dict[str, Any] = {}

    # =================================================================
    # 2026-09-19: ROUTING OVERRIDES ARE RESOLVED UPSTREAM OF ROUTING.
    # =================================================================
    # Binding resolution order:
    #
    #     task domain -> task type -> cloud override (yes/no)
    #                                   |
    #                     yes: skip the local inference layer(s)/tier(s)
    #                     no : the local layer runs (ladders, PATH A/B/C)
    #
    # An override that changes routing DOWNSTREAM must be evaluated UPSTREAM of
    # it, or the local layer is asked a question the override has already
    # answered.  This block previously sat ~290 lines BELOW the origin-gated
    # routing block, which is why a direct cloud modify request still had the
    # local PATH A/B/C question asked against it: on an origin-bearing artifact
    # that selected PATH A and wrote `modify_combo_prompt` +
    # `modify_skip_reintegration`, i.e. a DENOVO regeneration prompt was handed
    # to a T3 call whose change-order contract expects a change order.
    #
    # A cloud override means the request never walks the local layer at all --
    # no PATH A, no PATH B, no PATH C, no legacy.  It uses the SELECTED
    # harness's own meta-layer cloud entry, so the prompt package
    # (`_prompts.get(_harness_id)`) and the post-generation handling
    # (`cloud_change_order`) are shaped by the task type, exactly as if the
    # local layer had escalated through to the meta layer.
    #
    # Direct cloud is an explicit per-request user/agent override.  It is
    # intentionally separate from the automatic cloud_sota target, which
    # remains available after local ladder exhaustion.  Reject unknown modes
    # rather than silently changing the requested execution path.
    cloud_mode = metadata.get("cloud_mode")
    if cloud_mode is not None:
        if not isinstance(cloud_mode, str) or cloud_mode.strip().lower() != "direct":
            return _emit_override_invalid_response(
                _make_override_invalid_work(
                    domain=decision.domain,
                    task_type=decision.task_type,
                    bad=f"cloud_mode={cloud_mode!r}",
                    allowed=["metadata.cloud_mode='direct'"],
                ),
                request.stream,
            )
        execute_kwargs["direct_t3"] = True

    # =================================================================
    # 2026-09-20: `modify_harness_legacy` is OVERRIDE-ONLY (spec 15.9.7).
    # =================================================================
    # No `task_harnesses` row resolves to it any more (M4 is finished), so the
    # `-legacy` override is the ONLY selector.  Before this block existed the
    # row still named the legacy harness, so the override "worked" only because
    # config happened to point there -- and every path that was NOT PATH A
    # inherited it in NORMAL operation, which is precisely what 15.9.7 forbids.
    #
    # The two harnesses are NOT interchangeable by name:
    #   modify_harness         artifact_contract=new_code_generation (denovo),
    #                          local_change_order=false, 4 modify_regeneration_* keys
    #   modify_harness_legacy  artifact_contract=modify_artifact_round2 (change
    #                          order), local_change_order=true, artifact_context,
    #                          output_cap_factor, fence_languages=['json']
    # so selecting the wrong one is a contract mismatch, not a cosmetic choice.
    #
    # Gated on NOT direct cloud: a cloud override has already chosen the meta
    # layer, so the local override must not compete with it.  `/cloud` wins over
    # `-legacy` -- both produce the same cloud prompt and the same
    # post-generation handling, so the local question does not arise.
    #
    # 2026-09-20 (DECLARATION-ORDER FIX): this block MUST sit BELOW the
    # `execute_kwargs` declaration above and below the cloud-override block.
    # It first read `execute_kwargs.get("direct_t3")` while `execute_kwargs` was
    # not yet bound, which raised
    # `UnboundLocalError: cannot access local variable 'execute_kwargs'` -> HTTP
    # 500 -- but ONLY when `_legacy_route` was True, because that is the first
    # conjunct and `and` short-circuits.  A plain `/task-modify` (legacy False)
    # never evaluated it, so three of four live routing probes PASSED while two
    # routes 500'd.  Third instance of this exact fault class in this function
    # (see the `_legacy_route` 500 and the PATH A `execute_kwargs` notes above).
    # Do not hoist this block above `execute_kwargs`.
    if (_legacy_route
            and not execute_kwargs.get("direct_t3")
            and decision.task_type == "modify_artifact"
            and work.resolution_kind in ("specialist", "generalist")
            and "modify_harness_legacy" in state.resolution.harnesses):
        work = replace(
            work,
            harness=state.resolution.compose_harness_variant(
                "modify_harness_legacy", domain=work.domain,
            ),
        )
        print(
            "[modify] legacy override harness <- modify_harness_legacy",
            file=sys.stderr, flush=True,
        )

    # =================================================================
    # 2026-09-14: origin-gated modify routing
    # (CoE_coder_spec_modify_pipeline_DRAFT.md section 1)
    # =================================================================
    # Three paths, selected by whether the target artifact carries its
    # originating prompt.  /task-modify-legacy (or routing_override /
    # metadata legacy_route) forces the pre-existing behaviour instead.
    #
    #   PATH A  origin prompt present  -> local regeneration from a
    #           synthesized combined prompt; deterministic re-integration
    #           is SKIPPED (there is no change order).
    #   PATH B  no origin prompt       -> cloud change order, applied by
    #           the LOCAL deterministic re-integration (unchanged).
    #   PATH C  no origin, no cloud    -> capability message, NO model call.
    #
    # Reverse safety: the whole block is bypassed when _legacy_route is
    # True, when a direct cloud override is present, and when provenance.py
    # is absent the import fails soft.
    #
    # 2026-09-19: `_direct_cloud_route` -- the cloud override is resolved ABOVE
    # (the routing-override block sits immediately after `execute_kwargs` is
    # declared), so the local layer's origin-prompt question is never asked for
    # a direct cloud request.  PATH A on this route set `modify_combo_prompt` +
    # `modify_skip_reintegration` and swapped in the denovo regeneration
    # harness, none of which a cloud change-order call can use.
    _direct_cloud_route: bool = bool(execute_kwargs.get("direct_t3"))
    if (decision.task_type == "modify_artifact"
            and not _legacy_route
            and not _direct_cloud_route
            and work.resolution_kind in ("specialist", "generalist")):
        try:
            from .provenance import (
                modify_capability_message as _mcm,
                modify_paths_available as _mpa,
                approximation_disclosure as _adisc,
            )
            _target = state.artifact_store.current
            _origin = getattr(_target, "origin_prompt", None)
            _backends = (state.cfg or {}).get("backends") or {}
            _avail = _mpa(_origin, _backends)
            # Captured BEFORE any PATH A harness swap so the log and the
            # routing metadata report what the router selected, not what the
            # swap made it.
            _mod_harness_before = getattr(
                getattr(work, "harness", None), "harness_id", None
            )
            _mod_path = ("regenerated" if _avail["origin_regen"]
                         else "change_order" if _avail["cloud_order"]
                         else "unavailable")
            _mod_basis = (
                f"origin+mods({sum(1 for m in (getattr(_target, 'modifications', []) or [])
                                   if (m or {}).get('method') == 'cloud_change_order')} cloud)"
                if _avail["origin_regen"] and getattr(_target, "modifications", None)
                else "origin" if _avail["origin_regen"] else "none"
            )
            _cap_msg = _mcm(_avail)
            # 2026-09-14 (spec 6.4): the DISCLOSURE.  Free and factual -- a
            # property of the INPUT, not the output, so it cannot fail and
            # never escalates.  Surfaced BEFORE generation because it
            # qualifies what the user is about to receive.  Rides the
            # `coe_routing` metadata chunk, which server.py emits as the
            # FIRST chunk of the response (before the executor runs).
            _mod_disclosure = _adisc(
                _origin, getattr(_target, "modifications", None)
            ) if _target is not None else None
            print(
                f"[modify] origin-gated: path={_mod_path} "
                f"origin_prompt={'yes' if _origin else 'no'} "
                f"cloud={'yes' if _avail['cloud_order'] else 'no'} "
                f"artifact={getattr(_target, 'artifact_id', None)}"
                + (f" disclosure={_mod_disclosure!r}" if _mod_disclosure else ""),
                file=sys.stderr, flush=True,
            )
            if _cap_msg is not None:
                # PATH C.  No model is called -- an unavailable capability
                # must not be dressed up as an attempt that failed.
                return _emit_modify_unavailable_response(
                    _cap_msg, _avail, request.stream
                )
            if _avail["origin_regen"]:
                # PATH A: hand the runtime the combined prompt and tell it
                # to skip re-integration.  The domain has already been
                # pinned to the artifact's provenance above, so the
                # regeneration runs on the SAME domain/model that produced
                # the artifact rather than being re-classified from the
                # modification text.
                _combo = _synthesize_modify_combo(_target, prompt)
                if _combo is not None:
                    # ====================================================
                    # 2026-09-16 (M4): SWAP THE EXECUTING HARNESS FOR PATH A.
                    # ====================================================
                    # PATH A is a DENOVO regeneration, but until now it ran on
                    # the CHANGE-ORDER harness while swapping only the CONTRACT.
                    # The runtime block was not swapped, so change-order
                    # settings leaked into a denovo path -- most visibly:
                    #
                    #   stream_primitives.fence_counter.fence_languages: [json]
                    #
                    # `_fence_aliases_for` returns that override UNCONDITIONALLY,
                    # so an HTML artifact resolved to ['json'], the denovo fence
                    # check could not match the model's own ```html output, and
                    # EVERY regeneration failed with `no_code_fence`.  That
                    # string is unclassified in the change-order tiers, so it
                    # took the 'unrecognized' branch: no retry, budget consumed.
                    # Observed live as `UNCLASSIFIED ERROR 'no_code_fence'`.
                    #
                    # `modify_harness` declares no fence override, no
                    # output_cap_* and no artifact_context, and sets
                    # local_change_order=false.  So the leak is unreachable by
                    # construction rather than neutralised key by key.
                    #
                    # This swap is SAFE to do here because M1/M2 turned the
                    # eleven harness-name gates into capability properties.
                    # Before that, changing harness_id would have silently
                    # disarmed every one of them (the reason this was formerly
                    # done by swapping a config CELL instead).
                    #
                    # [FLAGGED FALSE as of 2026-09-20 -- superseded.]
                    # SUPERSEDED TEXT (was lines 3112-3115 on 2026-09-16):
                    #   # PATH B (no origin prompt) never reaches this branch, so
                    #   # it keeps running `modify_harness_legacy` -- the
                    #   # change-order ladder with its JSON fence and output cap --
                    #   # exactly as before.
                    #
                    # 2026-09-20: "PATH B never reaches this branch" remains
                    # TRUE.  "it keeps running `modify_harness_legacy`" is now
                    # FALSE: that was only ever true because the
                    # `task_harnesses` row named the legacy harness.  The row
                    # now names `modify_harness`, so PATH B runs the harness the
                    # router selected; `modify_harness_legacy` is reached only by
                    # the `-legacy` override, which is gated on NOT direct cloud.
                    # Both routes still reach the SAME shared T3 layer via
                    # cloud_change_order.
                    try:
                        if "modify_harness" in state.resolution.harnesses:
                            work = replace(
                                work,
                                harness=state.resolution.compose_harness_variant(
                                    "modify_harness", domain=work.domain,
                                ),
                            )
                            print(
                                f"[modify] PATH A harness <- modify_harness "
                                f"(was {_mod_harness_before})",
                                file=sys.stderr, flush=True,
                            )
                        else:
                            # Fail-soft: an older deployment without the new
                            # harness keeps working exactly as it does today
                            # (defect and all) rather than failing the request.
                            print(
                                "[modify] PATH A: modify_harness not declared; "
                                "staying on the change-order harness",
                                file=sys.stderr, flush=True,
                            )
                    except Exception as _mh_exc:              # noqa: BLE001
                        print(f"[modify] PATH A harness swap failed soft: {_mh_exc}",
                              file=sys.stderr, flush=True)
                    execute_kwargs["modify_combo_prompt"] = _combo
                    execute_kwargs["modify_skip_reintegration"] = True
                    # ====================================================
                    # 2026-09-16 (M6): the `modify_artifact_regenerate`
                    # cell is RETIRED.  This block used to read that cell's
                    # runtime block by NAME and copy four keys into
                    # execute_kwargs.
                    # ====================================================
                    # Those four keys now live on `modify_harness` itself --
                    # the harness that actually EXECUTES PATH A -- because
                    # `runtime = work.harness.runtime` (runtime.py:3835)
                    # already seeds them.  The copy loop was therefore
                    # redundant, and it was also the LAST thing keeping the
                    # retired cell alive.
                    #
                    # Precedence is unchanged by its removal.  The write-back
                    # in execute_runtime_plan is explicitly guarded ("bound
                    # only when supplied, so the harness's own values are not
                    # overwritten by an empty override"), so a value already
                    # present on the harness wins either way -- and the copy
                    # supplied exactly the same values from the cell.
                    #
                    # To retune PATH A's contract or wording, edit
                    # `harnesses.modify_harness.runtime` directly.
                    #
                    # ====================================================
                    # 2026-09-16 REMOVED: the `modify_basis_note` write.
                    # ====================================================
                    # This block used to do:
                    #
                    #     if _mod_basis.endswith("cloud)"):
                    #         execute_kwargs["modify_basis_note"] = (
                    #             "regenerated from the originating prompt and ...")
                    #
                    # It was a latent CRASH, not a working feature:
                    #   * `execute_kwargs` is splatted into
                    #     `state.executor.execute(work, prompt, **execute_kwargs)`,
                    #     and `executor.execute()` has NEVER accepted
                    #     `modify_basis_note` -- so any request that reached this
                    #     line died with
                    #     `TypeError: CleanExecutor.execute() got an unexpected
                    #     keyword argument 'modify_basis_note'`.
                    #   * No reader existed.  A package-wide search found this
                    #     assignment as the ONLY non-comment occurrence, so even
                    #     had it been accepted the value was never consumed.
                    #
                    # WHY IT WAS LATENT FOR A DAY.  The guard is
                    # `_mod_basis.endswith("cloud)")`, which requires the loaded
                    # artifact to expose provenance (an `origin_prompt` AND a
                    # cloud change order in its log).  Before the extension fix,
                    # an attached `.html` could be tagged `prose`, which made
                    # `_strip_embedded_header` fail to match the header (the
                    # header declares `html`; stripping was given `.md`), so the
                    # artifact loaded with NO provenance, `_mod_basis` was
                    # `"none"`, and this line was never reached.
                    #
                    # Fixing the extension restored the provenance -- correct,
                    # intended behaviour -- which made this branch reachable and
                    # exposed the crash.  The extension fix did not CAUSE it; it
                    # made a dead path live.
                    #
                    # `modify_basis` itself is unaffected: the routing metadata
                    # still reports it.  Reversal: re-add the write AND add the
                    # matching parameter to `executor.execute()` +
                    # `execute_runtime_plan()` if the note is ever wanted.
                    # ====================================================
        except ImportError:
            # provenance.py unavailable -> behave exactly as before.
            pass
        except Exception as _mg_exc:                       # noqa: BLE001
            print(f"[modify] origin-gated routing failed soft: {_mg_exc}",
                  file=sys.stderr, flush=True)

    # --- Routing metadata for response ---
    routing = {
        "domain": getattr(decision, "domain", None),
        "task_type": getattr(decision, "task_type", None),
        "task_scope": getattr(decision, "task_scope", None),
        "trivial": getattr(decision, "trivial", False),
        "domain_confidence": float(getattr(decision, "domain_confidence", 0.0) or 0.0),
        "task_type_confidence": float(getattr(decision, "task_type_confidence", 0.0) or 0.0),
        "task_scope_confidence": float(getattr(decision, "task_scope_confidence", 0.0) or 0.0),
        "trivial_confidence": float(getattr(decision, "trivial_confidence", 0.0) or 0.0),
        "detected_languages": getattr(decision, "detected_languages", []),
        "router_source": getattr(state.router, "last_router_source", "heuristic"),
        "harness_id": getattr(getattr(work, "harness", None), "harness_id", None),
        "primary_model": getattr(work, "primary_model", None),
        "matrix_cell": getattr(work, "matrix_cell", None),
        # Observability (Step 5/6): surface the router's own one-line
        # restatement of user intent in the streamed coe_routing
        # metadata so it is visible per-request without scraping logs.
        "user_intent": getattr(decision, "user_intent", ""),
        # Observability (Step 5/6): expose whether the router's probe
        # (re-classification for domain certainty) was invoked on this
        # request, so the domain decision path is fully transparent.
        "probe_invoked": getattr(decision, "probe_invoked", False),
        # 2026-09-14 (spec 7.7): which modify route was taken, and on what
        # provenance basis.  Two different things are called "modify" --
        # regenerated (local, denovo) vs change_order (cloud + deterministic
        # reintegration) -- with different harnesses, contracts and failure
        # modes.  Without this they are indistinguishable after the fact,
        # which makes a failure unfalsifiable.
        "modify_path": locals().get("_mod_path"),
        "modify_basis": locals().get("_mod_basis"),
        "modify_disclosure": locals().get("_mod_disclosure"),
        "legacy_route": _legacy_route,
        # ============================================================
        # 2026-09-16 (S1): LEGACY DEVIATION MARKER.
        # ============================================================
        # /task-modify-legacy is the NEW-MODEL CAPABILITY GATE, so its value
        # depends on its behaviour being comparable over time.  During the
        # origin-prompt work, four edits changed legacy behaviour with no
        # route guard, and NOTHING REPORTED IT -- a benchmark run would have
        # silently measured modified infrastructure.
        #
        # This marker makes any remaining deviation VISIBLE IN THE RESPONSE
        # instead of silent.  It is deliberately declared as an explicit list
        # rather than computed: each entry names a KNOWN, ACCEPTED divergence,
        # so the list being non-empty is a reviewed state and the list being
        # empty is a checkable claim.
        #
        # Currently-empty entries are the accepted, route-independent ones
        # (listed for auditability, not as defects):
        #   - the clarification resolver accepts more answer forms (shared fix)
        #   - the T3 cloud layer carries no loop guard (one cloud layer)
        #   - the reintegration machinery writes header + side-car (shared)
        # See PLAN_legacy_restore.md sections 0 and 2.
        "legacy_deviations": _LEGACY_ACCEPTED_DEVIATIONS if _legacy_route else [],
        "provenance_note": locals().get("_genuse_override_note"),
        # Slot fallback notifications (graceful fallback).  Populated by
        # _resolve_slot_fallbacks() when a slot's model is unregistered and
        # was substituted with the generalist.  User-visible, non-silent.
        "slot_fallbacks": _slot_fallbacks,
    }

    # --- Runtime overrides from metadata ---
    # NOTE: `execute_kwargs` is ALREADY declared above (immediately before the
    # origin-gated routing block) and is deliberately NOT re-initialised here.
    # Re-declaring it would WIPE the PATH A plumbing the routing block wrote
    # into it (modify_combo_prompt / modify_skip_reintegration / the
    # regeneration cell's contract + wording), silently turning PATH A back
    # into an ordinary change-order modify.  Add to it; never reset it.
    #
    # NOTE: `_legacy_route` is resolved EARLIER (immediately after prompt
    # extraction) because the origin-gated routing block and the routing
    # metadata both read it.  Do not re-declare it here -- a second definition
    # below the first use is exactly the NameError that caused an HTTP 500 on
    # every request.

    # think / closed_think override.  2026-10-13 correction: an earlier
    # revision of this header claimed the effective per-instance override
    # "wins over the model-card default in the runtime".  Per the 2026-09-29
    # user ruling (Option B), it does NOT: the bound slot model's card
    # `think` governs its own think state and prompt variant, and this
    # override selects ROLE ORDERING only (plus the generalist-slot
    # carve-out) -- see the three Option-B sites in resolution.py and
    # runtime.py. Sources, highest precedence first:
    #   1. routing_override.think (from a /thinkon|/thinkoff slash command)
    #   2. metadata.think (per-request JSON override)
    #   3. metadata.closed_think (legacy alias; closed_think=False == think-on)
    _think_override: bool | None = None
    if request.routing_override is not None and request.routing_override.think is not None:
        _think_override = request.routing_override.think
    elif "think" in metadata and metadata["think"] is not None:
        _think_override = bool(metadata["think"])
    elif "closed_think" in metadata and metadata["closed_think"] is not None:
        # closed_think True = think-off; False = think-on.
        _think_override = not bool(metadata["closed_think"])
    if _think_override is not None:
        execute_kwargs["think_override"] = _think_override
    if _seed_override is not None:
        execute_kwargs["seed_override"] = _seed_override

    # Ordinary image requests must reach a vision-capable selected target.
    # Do not silently downgrade them to text-only inference.
    image_target_model = (
        state.executor.t3_vision_model
        if execute_kwargs.get("direct_t3") and state.executor.t3_vision_model
        else state.executor.t3_model
        if execute_kwargs.get("direct_t3")
        else work.primary_model
    )
    if images and image_target_model and image_target_model not in state.vision_models:
        return _emit_vision_unavailable_response(
            work, image_target_model, request.stream,
        )

    if (
        request.routing_override is not None
        and getattr(request.routing_override, "evaluate_prior_wp", False)
    ):
        execute_kwargs["evaluate_prior_wp"] = True

    # NOTE (2026-09-19): this comment used to claim that "timeout_s and num_ctx
    # are handled by the runtime's existing parameters".  Neither is:
    # `num_ctx` travels per-harness in the sampling block, and `timeout_s` is
    # read NOWHERE at all (runtime.py states explicitly that the runtime has no
    # hard wall-clock timeout).  The stale claim is corrected rather than
    # deleted so the absence is recorded as intentional.
    # 2026-09-19 (PLAN_progress_mechanism.md phase 7): the deleted `summary`
    # event's metadata pass-through was REMOVED with it.  That event was emitted
    # by the runtime and consumed by nothing at any layer.
    # NOTE: `metadata` is dict[str, Any], so a client still sending the old key
    # is NOT rejected -- it is simply ignored now.  That is deliberate:
    # `metadata` is an open dict by design and is not subject to the strict
    # extra="forbid" models.
    if "checkpoint_interval_s" in metadata:
        execute_kwargs["checkpoint_interval_s"] = float(metadata["checkpoint_interval_s"])
    else:
        # 2026-09-19 (PLAN_progress_mechanism.md phase 6): the GLOBAL cadence
        # from config.yaml, sibling of modify_combo_mode.  Precedence is
        #   metadata.checkpoint_interval_s  >  config progress_interval_s  >  30.0
        # and the code default (runtime.DEFAULT_CHECKPOINT_INTERVAL_S) is the
        # final fallback, so an absent or null key is always safe.
        # Read defensively: a malformed value must degrade to the default
        # rather than fail the whole request.
        try:
            _pi = (state.cfg or {}).get("progress_interval_s")
            if _pi is not None and float(_pi) > 0:
                execute_kwargs["checkpoint_interval_s"] = float(_pi)
        except (TypeError, ValueError):
            pass

    # Step 5 (#1 fix): forward the router's restated user intent and the
    # attachment's source filename into the executor so the two-round
    # modify_artifact block can derive a meaningful descriptor tag
    # (instead of the fallback "modify-untitled").  These are threaded
    # through execute() -> execute_runtime_plan() -> the two-round
    # block's apply_edits()/pending_artifact_edit packaging.
    execute_kwargs["user_intent"] = (
        getattr(decision, "user_intent", "") or prompt
    )
    # 2026-09-18 (user ruling): ROUTE-SPECIFIC NAMING RETIRED.
    #
    # The attachment's filename is now forwarded on EVERY route, including
    # /task-modify-legacy.  Naming is a POST-SHARED-WORKFLOW function: this
    # value feeds artifact_store.create() Step 5, which is the SAME naming
    # code for a regenerated daughter and a change-ordered one.  Withholding
    # it from the legacy route made an identical modification of an identical
    # file land in a different lineage purely because of which route served
    # it -- the exact divergence create()'s Step 5 comment exists to prevent.
    #
    # The original concern (comparability of the reference instrument) still
    # holds and is met elsewhere: /task-modify-legacy remains the single-tier
    # change-order path, so the GENERATION CONTRACT is what is compared over
    # time.  Route identity is carried by `coe_routing.harness_id`.
    #
    # Reversal: restore the `not _legacy_route` condition here AND at the
    # staging-artifact create() above -- both, never one (see the S3 note).
    execute_kwargs["source_filename"] = (
        request.artifact.filename if request.artifact else None
    )
    execute_kwargs["images"] = images
    # 2026-09-28 (user ruling): NO LONGER gated on ``harness_id == "general_use"``.
    # The attached text / working-artifact block is a prompt-COMPOSITION
    # feature, not a general_use privilege.  With the image harness swap
    # removed, a non-general_use harness would otherwise SILENTLY LOSE its
    # attached context (a text or artifact attachment dropped with no error).
    # The matching composition gates in ``resolution.py`` (local prompt) and
    # ``runtime.py`` (T3 cloud prompt) were de-gated in the same change --
    # all three must move together or the context is dropped downstream.
    #
    # Reversal: restore the ``if ...harness_id == "general_use":`` guard here
    # AND at resolution.py's local-composition site AND runtime.py's
    # ``_cloud_user`` site.
    execute_kwargs["context_text"] = _build_attached_context(request, state)

    # v13 step 2 (2026-08-08): extract the submitted file content from
    # the prompt's <code>...</code> block and pass it as ``patch_base``
    # so the bug_fix / unified_diff_patch contract can enforce the
    # context-correctness gate (only return a patch that removes code
    # actually present in the submitted file).  When the prompt has no
    # <code> block (e.g. a bare issue with no file attached), this is
    # None and the gate stays inactive -- legacy take-last behavior.
    execute_kwargs["patch_base"] = _extract_code_block(prompt)
    execute_kwargs["t3_context_text"] = _build_t3_context(
        request,
        state,
        prompt,
        decision.task_type,
        image_details,
    )

    # Disable memory injection for this request if requested
    if metadata.get("no_inject"):
        old_inject = state.executor.inject_last_turn
        state.executor.inject_last_turn = False
    else:
        old_inject = None

    show_thinking = bool(metadata.get("show_thinking", False))
    keep_alive: int | None = metadata.get("keep_alive")

    # --- Turn record + progress (2026-09-19, PLAN_progress_mechanism.md
    # phase 4) -----------------------------------------------------------
    # Resolved ONCE, before the transport split, so the streaming and
    # non-streaming paths agree on the id and on the record.  A client-supplied
    # X-Request-Id wins; otherwise the server MINTS one and returns it in the
    # response's `coe_turn` block, so a caller that did not supply a header can
    # still poll GET /v1/turn/{id}/progress and cancel via /v1/cancel.
    #
    # Why here and not inside each branch: the record must exist for BOTH
    # transports.  Previously the cancel registry was populated only inside
    # _event_stream, so a non-streaming turn could neither be observed nor
    # cancelled while it ran.
    #
    # NOTE: the turn record for this request was ALREADY created above the
    # trivial short-circuit (see the phase-4 block), specifically so a trivial
    # turn is pollable too.  Do NOT create a second record here.

    if request.stream:
        # Cancel-turn: the client's X-Request-Id addresses this turn
        # for /v1/cancel.  Passed as an explicit parameter — a ContextVar
        # set here is invisible to the stream generator (Starlette
        # iterates it outside the endpoint's context; caught live).
        return await _stream_completion(
            state, work, decision, prompt, routing, execute_kwargs, old_inject,
            _slot_fallbacks, show_thinking=show_thinking,
            images=images,
            request_id=_turn_id,
            record=_turn_rec,
        )

    # --- Non-streaming path: full response ---
    # `progress=` is passed EXPLICITLY, never via execute_kwargs: the streaming
    # branch also passes progress= and then **execute_kwargs, so a `progress`
    # key in that dict would be a duplicate keyword argument -> TypeError.
    #
    # _RecordingProgress(None, ...) -- inner=None because there is no SSE
    # transport to forward to.  This is the ONLY channel by which a
    # non-streaming turn learns that it has been cancelled, and it also feeds
    # the poll endpoint with live progress.
    #
    # NOTE: the task is deliberately NOT registered in _ACTIVE_REQUEST_TASKS.
    # An abrupt task.cancel() from /v1/cancel would tear down this handler
    # mid-response; non-streaming cancels COOPERATIVELY instead (the next
    # runtime `checkpoint` returns ABORT).
    try:
        result = await state.executor.execute(
            work, prompt, progress=_RecordingProgress(None, _turn_rec),
            **execute_kwargs,
        )
        _turn_rec.status = "done"
        _turn_rec.finished_at = time.monotonic()
    except Exception as _exec_exc:
        # Phase 3 (no 200-with-silence): an execution failure must
        # surface as a visible 5xx JSON response, never as an
        # empty/handled-away response.
        _turn_rec.status = "error"
        _turn_rec.error = f"{type(_exec_exc).__name__}: {_exec_exc}"
        _turn_rec.finished_at = time.monotonic()
        print(
            f"[execution] non-streaming failure: {type(_exec_exc).__name__}: {_exec_exc}",
            file=sys.stderr, flush=True,
        )
        return JSONResponse(status_code=503, content={
            "error": "execution_failed",
            "detail": f"{type(_exec_exc).__name__}: {_exec_exc}",
            "coe_routing": routing,
            "coe_turn": {"request_id": _turn_id, "status": "error",
                         "aborted": False},
        })
    finally:
        if old_inject is not None:
            state.executor.inject_last_turn = old_inject

    # Step 5.10: post-freeze artifact commit.  This runs strictly
    # after execute() returned, i.e. strictly after SessionMemory
    # already froze this turn -- see _commit_pending_artifact()'s
    # docstring for why the commit cannot happen any earlier.
    output_text, coe_artifact = _commit_pending_artifact(
        state, work, decision, result, prompt=prompt,
        evaluate_prior_wp=bool(
            request.routing_override is not None
            and getattr(request.routing_override, "evaluate_prior_wp", False)
        ),
    )

    # 2026-09-14 (spec 6.2 / 6.3): pull the drift evidence the runtime
    # recorded on the PATH A branch.  Both keys are absent on every other
    # path, so this stays None and existing responses are unchanged.
    _mod_drift: dict[str, Any] | None = getattr(
        result, "modify_drift_evidence", None
    )

    return JSONResponse(content={
        "id": f"coe-clean-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result.selected_model or work.primary_model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": output_text},
            "finish_reason": "stop" if result.passed else "error",
            "coe_routing": routing,
            "coe_artifact": coe_artifact,
            # 2026-09-14 (spec 6.2 / 6.3): drift-flag observability.  Tier-1
            # literals and the Tier-2 "owe a review" notice are REPORT-ONLY
            # (6.2/6.3 never escalate), so surfacing them here is what makes
            # them observable at all -- otherwise a drift flag is written to
            # stderr and lost.  Both are None on paths that produce no drift
            # evidence (i.e. every non-PATH-A request), so existing clients
            # are unaffected.
            "coe_drift": _mod_drift,
            "coe_attempts": [
                {
                    "tier": a.tier,
                    "model_id": a.model_id,
                    "passed": a.passed,
                    "error": a.error,
                    "latency_s": a.latency_s,
                }
                for a in result.attempts
            ],
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    })


# ---------------------------------------------------------------------------
# Streaming response generator
# ---------------------------------------------------------------------------

async def _stream_completion(
    app_state: CleanState,
    work: Any,
    decision: Any,
    prompt: str,
    routing: dict[str, Any],
    execute_kwargs: dict[str, Any],
    old_inject: bool | None,
    slot_fallbacks: list[dict[str, str]] | None = None,
    show_thinking: bool = False,
    images: list[str] | None = None,
    request_id: str | None = None,
    record: _TurnRecord | None = None,
) -> StreamingResponse:
    """Execute the request and stream token events as SSE.

    Emits OpenAI-compatible SSE chunks:
    * First chunk: role="assistant" (no content)
    * Token chunks: content=<token text>, model=<model_id>
    * Tier transitions: metadata chunk with tier info
    * ``coe_artifact`` metadata chunk, when the turn committed an
      artifact (Step 6.1), pushed before the final content chunk
    * Final content chunk: the post-commit diff/full/notice text
      (Step 5.10/6.2/6.4), pushed only when it differs from what was
      already streamed live -- avoids duplicating content for turns
      that don't touch the artifact store
    * Final chunk: finish_reason="stop" or "error"
    * Final data: [DONE] marker (OpenAI convention)

    Step 6.3: R1/R2 internal JSON (two-round modify_artifact) is
    never forwarded as visible content -- suppressed by tier suffix
    (``_R1``/``_R2``, case-insensitive) -- and the two-round block's
    internal ``complete`` marker is never forwarded either (flagged
    via ``pending_artifact_edit`` metadata). The real content for a
    modify_artifact turn is the diff/full text pushed after the
    post-freeze commit below.
    """
    queue: asyncio.Queue = asyncio.Queue()
    # 2026-09-19 (phase 4): wrap the transport callback so the SAME checkpoint
    # event feeds both surfaces -- the SSE queue (live CLI line) and the turn
    # record (poll endpoint).  Wrapping here rather than extending
    # StreamingProgressCallback keeps that class's tested transport behaviour
    # untouched, and it is also the single place a cooperative ABORT is decided
    # for the streaming path.
    callback = _RecordingProgress(StreamingProgressCallback(queue), record)
    model_name = work.primary_model or "unknown"

    # Send routing info as the first metadata chunk
    await queue.put(("routing", "", model_name))

    # Cancel-turn registry (2026-09-07): the executor runs as a
    # background task keyed by the client's X-Request-Id header.
    # /v1/cancel cancels that task; the runtime's own cancellation
    # handling records the attempt, preserves session state, and the
    # SSE loop below drains the queue and closes the stream cleanly.
    # (request_id is passed as an explicit parameter — a ContextVar set
    # in the endpoint is invisible to the generator, which Starlette
    # iterates outside the endpoint's context.  Caught live 2026-09-07.)
    task_ref: dict[str, asyncio.Task] = {}

    async def _run_execution() -> None:
        """Run the executor in a background task."""
        try:
            result = await app_state.executor.execute(
                work, prompt, progress=callback, **execute_kwargs
            )
            # Step 5.10: post-freeze artifact commit.  Runs strictly
            # after execute() returned, i.e. strictly after
            # SessionMemory already froze this turn.
            output_text, coe_artifact = _commit_pending_artifact(
                app_state, work, decision, result, prompt=prompt,
                evaluate_prior_wp=bool(execute_kwargs.get("evaluate_prior_wp")),
            )
            if coe_artifact:
                await queue.put(("artifact", _json.dumps(coe_artifact), model_name))
            # v13 step 2: emit the CHOSEN patch object -- exactly what
            # the post-processing layer / bench evaluator will grade.
            # For bug_fix this is result.output (the context-correct
            # selected block, after the polarity flip / corrected-patch
            # side-channel).  The client can compare this against the
            # live stream to see which block was selected.
            await queue.put(("selected_patch", result.output or "", model_name))
            # Only push a new content chunk when the post-commit text
            # actually differs from what the live token/complete
            # events already showed (the diff, the notice-prefixed
            # text, or a commit-time error) -- otherwise this would
            # duplicate content for harnesses that never touch the
            # artifact store.
            if output_text and output_text != result.output:
                await queue.put(("final_content", output_text, model_name))
            elif output_text and result.pending_artifact_edit and result.passed:
                # modify_artifact success but output_text happens to
                # equal result.output (e.g. an empty diff edge case)
                # -- the live stream showed nothing (R1/R2 suppressed
                # by 6.3), so this IS new content and must be sent.
                await queue.put(("final_content", output_text, model_name))
            # Signal completion with pass/fail
            await queue.put(("result", "", model_name))
            # Store result for the final chunk
            await queue.put(("finish", "stop" if result.passed else "error", model_name))
        except Exception as exc:
            await queue.put(("error", str(exc)[:200], model_name))
        finally:
            if old_inject is not None:
                app_state.executor.inject_last_turn = old_inject
            await queue.put(("done", "", ""))

    async def _event_stream() -> AsyncIterator[str]:
        """Generate SSE events from the queue."""
        # Start execution in background
        task = asyncio.create_task(_run_execution())
        # Register for /v1/cancel (client-side turn cancellation).
        if request_id:
            _ACTIVE_REQUEST_TASKS[request_id] = task
            task_ref["task"] = task
        # First chunk: role announcement
        yield _sse_chunk(None, model=model_name, role="assistant")

        # Graceful slot fallback notifications.  Emitted as dedicated
        # SSE metadata events so the client can surface them to the user
        # (who may cancel the turn if they don't want the generalist to
        # answer).  Non-silent.
        for _sf in (slot_fallbacks or []):
            yield _sse_chunk(
                None, model=model_name,
                metadata={"slot_fallback": _sf},
            )

        # Step 6.3: tracks the most recent tier_start/tier_complete
        # tier name so `token` events from the two-round modify_artifact
        # R1/R2 rounds can be suppressed (their tier strings end in
        # "_R1"/"_R2" -- see runtime.py's `f"{tier}_{round_name}"`).
        current_tier = ""

        # Track whether any user-visible content tokens were
        # streamed.  When tokens were streamed, the `complete` event
        # is suppressed as content (it would double-emit the answer).
        # The `complete` event is only a fallback for harnesses that
        # don't stream tokens (or when the stream produced zero
        # visible chunks).
        streamed_content = False

        # 2026-09-05 status tags: one-shot latches so the [thinking] tag
        # is emitted only on the FIRST hidden reasoning token (not
        # per-token), and the [reading attachment] tag fires once when
        # the request carries images.
        _thinking_tag_sent = False
        if images:
            yield _sse_chunk(
                None, model=model_name,
                metadata={"status": {"kind": "reading_attachment",
                                      "detail": f"{len(images)} image(s)"}},
            )

        # SSE heartbeat: while the queue is empty (model load, ladder
        # transitions), emit a metadata ping every 30s so the CLIENT
        # connection stays alive through legitimately slow internal
        # phases.  After the first event, a 210s idle watchdog is the
        # backstop for a WEDGED runtime coroutine: every model-side
        # wedge is owned by a closer deadline (vram.prepare 120s,
        # TTFT budget, slow-decode 90s), so the watchdog only fires
        # when the coroutine is stuck without raising.  Cloud tiers
        # are EXEMPT: minutes of queue/cold-start silence are
        # legitimate there and owned by the runtime's cloud
        # silence_timeout_s / total_timeout_s (config.yaml).
        HEARTBEAT_INTERVAL_S = 30.0
        LOCAL_IDLE_WATCHDOG_S = 210.0
        saw_first_event = False

        def _is_cloud_tier(tier_name: str | None) -> bool:
            name = (tier_name or "").upper()
            return "T3" in name or "CLOUD" in name

        while True:
            try:
                # Wait for the next event with a timeout so the
                # SSE connection doesn't hang forever if something
                # goes wrong.  Heartbeats reset the client's idle
                # clock; the idle watchdog still trips if even the
                # heartbeat loop dies (genuine server death).
                wait_s = (
                    LOCAL_IDLE_WATCHDOG_S if saw_first_event
                    else HEARTBEAT_INTERVAL_S
                )
                item = await asyncio.wait_for(queue.get(), timeout=wait_s)
            except asyncio.TimeoutError:
                if not saw_first_event or _is_cloud_tier(current_tier):
                    # No event yet (the prepare deadline owns the
                    # load) or inside a cloud tier (the runtime's
                    # cloud guards own silence): heartbeat and keep
                    # waiting -- the local watchdog does not fire.
                    yield _sse_chunk(
                        None, model=model_name,
                        metadata={"heartbeat": True},
                    )
                    continue
                yield _sse_chunk(
                    "[timeout]", model=model_name,
                    finish_reason="error",
                    metadata={"error": "stream_timeout"},
                )
                yield "data: [DONE]\n\n"
                break

            event_type = item[0]
            # Any real event (routing metadata included) marks the
            # runtime as live: switch from heartbeat mode to the
            # 210s idle watchdog.
            if not saw_first_event:
                saw_first_event = True

            if event_type == "routing":
                # Emit routing metadata as a non-content chunk
                _, text, model_id = item
                yield _sse_chunk(
                    None, model=model_id,
                    metadata={"coe_routing": routing},
                )
            elif event_type == "token":
                _, text, model_id, event_tier, is_thinking = item
                current_tier = event_tier or current_tier
                # R1/R2 round JSON is internal — never user-visible.
                # Containment (not endswith) so retry variants like
                # "T0_R1_retry" / "T2_R2_retry" are suppressed too:
                # endswith(("_R1","_R2")) missed them and streamed the
                # raw contract JSON to the CLI (live bug 2026-09-07).
                tier_u = current_tier.upper()
                if "_R1" in tier_u or "_R2" in tier_u:
                    continue
                if is_thinking and not show_thinking:
                    # Reasoning tokens are suppressed as CONTENT, but the
                    # client must still know the model is alive (a native-
                    # think model can reason for minutes with zero visible
                    # output).  One status tag on the first hidden token
                    # only -- a per-token ping would spam the stream.
                    if not _thinking_tag_sent:
                        _thinking_tag_sent = True
                        yield _sse_chunk(
                            None, model=model_id,
                            metadata={"status": {"kind": "thinking", "detail": ""}},
                        )
                    continue
                streamed_content = True
                yield _sse_chunk(text, model=model_id, metadata={"is_thinking": is_thinking})
            elif event_type == "checkpoint":
                # 2026-09-19 (PLAN_progress_mechanism.md phase 3): the matching
                # consumer for the 6-tuple queued in StreamingProgressCallback.
                # Phase 2 made the producer queue it; without THIS branch the
                # event would sit in the queue unconsumed and the tuple would
                # never reach a client -- the same "produced but not
                # transported" gap as B4a.
                #
                # Emitted as a metadata chunk, NOT content: a progress ping
                # must never be mistaken for response text.  The CLI renders
                # it as a `[progress]` line (CoE-coder.py::_print_meta).
                _, _cp_tier, _cp_model, _cp_meta, _cp_sofar, _cp_elapsed = item
                yield _sse_chunk(
                    None, model=_cp_model or model_name,
                    metadata={"checkpoint": {
                        "elapsed_s": _cp_elapsed,
                        "text_length": int((_cp_meta or {}).get("text_length", 0) or 0),
                        "tier": _cp_tier,
                        "preview": _cp_sofar,
                    }},
                )
            elif event_type == "tier_start":
                _, text, model_id = item
                current_tier = text or current_tier
                # Single-tier harness (e.g. general_use, trivial_answer):
                # LOCAL tier announcements are noise — there is no ladder
                # to show. Internal meta logs still record the tier.
                #
                # 2026-09-18 (user ruling): a CLOUD tier is exempt.  Standing
                # rule — "any cloud request should list the model the request
                # is going to".  A single-tier harness has no ladder, but it
                # can still reach the cloud via a DIRECT request
                # (`metadata.cloud_mode='direct'` -> execute_kwargs
                # ["direct_t3"]), and runtime.py emits a T3 tier_start with
                # the resolved model at every such call.  The tier count is a
                # LOCAL-ladder property and cannot see that, so the cloud
                # announcement was being emitted and then silently dropped:
                # /cloud on general_use reached the model with no model
                # named.  `_is_cloud_tier` is the same helper the SSE
                # watchdog already uses, so this adds no new notion of
                # "cloud".  Local single-tier announcements stay suppressed.
                if len(work.harness.tiers) > 1 or _is_cloud_tier(text):
                    yield _sse_chunk(
                        None, model=model_id,
                        metadata={"tier_start": text, "model_id": model_id},
                    )
            elif event_type == "status":
                # Long-silence phase tag (model load / attachment read).
                # Always surfaced regardless of harness tier count: this
                # is exactly the window where users previously saw dead
                # air and cancelled a healthy request.
                _, kind, text, model_id = item
                yield _sse_chunk(
                    None, model=model_id or model_name,
                    metadata={"status": {"kind": kind, "detail": text}},
                )
            elif event_type == "tier_retry":
                # 2026-09-13 (user ruling): accurate retry reporting.  The
                # runtime emits this for every in-tier retry (pathology
                # re-roll or functional re-attempt) BEFORE the next attempt
                # begins, carrying the retry kind, the attempt number, the
                # limit, and the action the ladder decided.  tier_complete
                # is now reserved for a tier that actually finished (passed
                # or escalated), so the two surfaces no longer contradict
                # each other.
                _, _, model_id, event_metadata = item
                # 2026-09-18 (B4b FIX): the `len(work.harness.tiers) > 1` gate
                # that used to wrap this was REMOVED.  A retry only ever occurs
                # INSIDE a ladder, so the gate could never add information --
                # and it would silently suppress the event on any future
                # single-tier harness that retried.  tier_complete keeps its
                # own gate; this one does not need it.
                yield _sse_chunk(
                    None, model=model_id,
                    metadata={"tier_retry": event_metadata or {}},
                )
            elif event_type == "tier_complete":
                _, text, model_id, event_metadata = item
                current_tier = text or current_tier
                if len(work.harness.tiers) > 1:
                    tier_metadata = {"tier_complete": text}
                    if event_metadata:
                        tier_metadata["tier_details"] = event_metadata
                    yield _sse_chunk(
                        None, model=model_id,
                        metadata=tier_metadata,
                    )
            elif event_type == "complete":
                # The runtime's own complete event.  Suppressed
                # entirely when it's the two-round block's internal
                # pending-edit marker (Step 5.9/6.3) -- the real
                # content is pushed separately as `final_content`
                # after the post-freeze commit.  For all other
                # harnesses this remains the existing fallback: if no
                # tokens were streamed, emit the final output here.
                # When tokens WERE streamed, the content is already
                # in the client's buffer -- re-emitting it here would
                # double the output.
                _, text, model_id, is_pending_edit = item
                if text and not is_pending_edit and not streamed_content:
                    yield _sse_chunk(text, model=model_id)
            elif event_type == "artifact":
                # Step 6.1: coe_artifact metadata, pushed before the
                # final content chunk.
                _, text, model_id = item
                yield _sse_chunk(None, model=model_id, metadata={"coe_artifact": _json.loads(text)})
            elif event_type == "selected_patch":
                # v13 step 2: the CHOSEN patch object -- exactly what
                # the post-processing layer / bench evaluator grades.
                # Emitted as metadata so the client can compare it
                # against the live stream without it being mistaken
                # for more streamed content.
                _, text, model_id = item
                yield _sse_chunk(
                    None, model=model_id,
                    metadata={"selected_patch": text},
                )
            elif event_type == "final_content":
                # Step 5.10/6.2/6.4: the post-commit diff/full/notice
                # text, or a commit-time error message.
                _, text, model_id = item
                yield _sse_chunk(text, model=model_id)
            elif event_type == "result":
                # Execution finished, waiting for finish signal
                pass
            elif event_type == "finish":
                _, text, model_id = item
                yield _sse_chunk(None, model=model_name, finish_reason=text)
            elif event_type == "error":
                _, text, model_id = item
                yield _sse_chunk(
                    None, model=model_name, finish_reason="error",
                    metadata={"error": text},
                )
            elif event_type == "done":
                yield "data: [DONE]\n\n"
                break

        # Deregister from the cancel registry on ANY exit path
        # (normal finish, error, timeout, client disconnect).
        if request_id:
            _ACTIVE_REQUEST_TASKS.pop(request_id, None)
        # Ensure the background task is done
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# V13: Artifact pipeline streaming handlers
# ---------------------------------------------------------------------------

# Step 5.7 correction (2026-08-04): the old `_extract_fenced_code()` regex
# re-derivation over `result.output` has been removed. new_code_generation's
# artifact content now comes from `CleanExecutionResult.extracted_artifact_content`
# -- the harness's own contract-validated extraction (see runtime.py's
# *_new_code_contract functions and _commit_pending_artifact() above).


