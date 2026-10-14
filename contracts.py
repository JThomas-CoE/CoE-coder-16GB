"""Clean-slice contracts shared by resolution, execution, and transport."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Protocol


@dataclass(frozen=True)
class CleanUnit:
    """Immutable task data supplied to a clean harness."""

    domain: str
    prompt: str
    persona: str = ""
    closed_think_block: str | None = None
    context_text: str = ""
    # Complete substantive request context reserved for T3. Local harness
    # composition continues to use context_text only.
    t3_context_text: str = ""
    # Provider-neutral base64 data URIs.  Classification leaves this empty;
    # only the selected answer path carries the image bytes.
    images: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CleanTierContext:
    """Diagnostic context permitted for a single execution tier."""

    tier: str
    prior_error: str | None = None


@dataclass(frozen=True)
class CleanPackage:
    """Fully composed provider-neutral generation request."""

    model_id: str
    provider: str
    messages: list[dict[str, Any]]
    sampling: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    # Native-reasoning flag (model-card default). None = unset (harness
    # closed_think governs). True = think-on, False = think-off.
    think: bool | None = None
    # 2026-09-09 (user ruling): whether the model tolerates RAW prompt
    # continuation (assistant response-prefix injection via
    # /api/generate raw). Model-card property, separate from the think
    # policy. False = the transport MUST NOT use the raw-prefix path;
    # text prefills are dropped and think suppression rides the native
    # /api/chat think flag. None = tolerated (legacy default).
    raw_prefill: bool | None = None
    # Provider-neutral text already present at the beginning of the assistant
    # response. Transports must not represent this as assistant chat history.
    response_prefix: str | None = None
    # Provider-neutral base64 data URIs.  Each transport adapts these to its
    # native multimodal message format.
    images: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CleanAttempt:
    tier: str
    model_id: str
    output: str
    passed: bool
    error: str | None
    latency_s: float


class CleanControlDecision(str, Enum):
    """A checkpoint decision returned to the runtime by the control callback."""

    CONTINUE = "continue"
    ABORT = "abort"
    ACCEPT_PARTIAL = "accept_partial"


@dataclass(frozen=True)
class CleanProgressEvent:
    """A single runtime progress notification.

    ``event_type`` is one of:

    * ``token``         - a token chunk was produced; ``text``
                         carries it.  Every chunk the transport
                         emits is surfaced as a ``token`` event
                         with no category-based filtering.
    * ``tier_start``    - the runtime entered a new tier.
    * ``tier_complete`` - a tier completed.
    * ``checkpoint``    - the runtime is asking the control callback for
                         a decision; ``elapsed_s`` and ``text_so_far`` are
                         provided.  This is the ONE periodic progress event:
                         it is the only event whose callback return value the
                         runtime READS, so it is also the only channel for a
                         cooperative cancellation.  It feeds both the SSE
                         stream and the non-streaming turn record.
    * ``complete``      - the run finished; ``output`` carries the final
                         result.

    A ``summary`` event type was REMOVED (2026-09-19): it was emitted on a
    timer and consumed by nothing at any layer -- not the streaming callback,
    not the SSE consumer, not the CLI, not session memory.  ``checkpoint``
    carries the same payload and supersedes it.
    """

    event_type: str
    tier: str = ""
    text: str = ""
    output: str = ""
    elapsed_s: float = 0.0
    text_so_far: str = ""
    model_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class CleanProgressCallback(Protocol):
    """Observer that the runtime notifies of every progress event.

    ``on_event`` is the only required method.  The runtime always calls
    it; the callback decides whether to print, stream, log, or ignore.
    A callback may also be a coroutine that returns a
    ``CleanControlDecision`` for ``checkpoint`` events; the default
    decision is ``continue`` when the callback is synchronous or
    returns ``None``.
    """

    def on_event(self, event: CleanProgressEvent) -> Any:
        ...


class NullProgressCallback:
    """Drop-in default that returns CONTINUE for every checkpoint."""

    def on_event(self, event: CleanProgressEvent) -> CleanControlDecision | None:
        if event.event_type == "checkpoint":
            return CleanControlDecision.CONTINUE
        return None


@dataclass
class CleanExecutionResult:
    work: Any
    output: str = ""
    passed: bool = False
    selected_model: str = ""
    attempts: list[CleanAttempt] = field(default_factory=list)
    error: str | None = None
    accepted_partial: bool = False
    aborted: bool = False
    total_elapsed_s: float = 0.0

    # --- Step 5 correction (2026-08-04): freeze-before-artifact ---
    # The harness's own contract-validated extracted artifact content
    # for new_code_generation (NOT re-derived from a regex over
    # result.output).  None for harnesses that don't produce/validate
    # an artifact body.  Set by the tier loop's post-contract check
    # when the contract stashes ``runtime["_extracted_code"]`` on a
    # pass (see runtime.py's *_new_code_contract functions).
    extracted_artifact_content: str | None = None
    # Step 5 (Option B): the fence language of the extracted artifact
    # (e.g. "python", "html", "rust"), stashed by the *_new_code_contract
    # functions alongside ``_extracted_code``.  Used by server.py's
    # artifact commit to pick the correct file extension.  None when no
    # artifact body was produced.
    extracted_artifact_language: str | None = None
    # General-use work product, normalized after generation.  A complete
    # fenced block is stored as code with its detected language; otherwise
    # the full visible response is stored as Markdown.
    general_use_work_product_content: str | None = None
    general_use_work_product_language: str | None = None
    # The modify_artifact two-round block's validated-but-not-yet-
    # committed edit package.  runtime.py fills this in on R1+R2
    # success instead of calling ArtifactStore.apply_edits() itself,
    # so the actual commit can happen in server.py strictly after
    # SessionMemory has already frozen this turn.
    # {"edits": r2_json, "artifact_id": ..., "user_intent": ...,
    #  "source_filename": ...} or None.
    pending_artifact_edit: dict[str, Any] | None = None
    # 2026-09-14 (spec 5.7.2 PATH A): True when a modify_artifact request was
    # served by LOCAL REGENERATION rather than a change order.  On this path
    # the output is a COMPLETE artifact (carried in
    # ``extracted_artifact_content``) and ``pending_artifact_edit`` is None
    # because there is nothing to re-integrate.  server.py keys off this to
    # commit a modified VERSION of the lineage (inheriting origin_prompt,
    # appending the modification log) instead of forking via apply_edits, and
    # to record method="regenerated" in the provenance log.
    modify_was_regenerated: bool = False
    # 2026-09-14 (spec 6.2 / 6.3): REPORT-ONLY drift evidence for a PATH A
    # regeneration.  Neither check escalates, so this field is the only way
    # the flags become observable (stderr alone loses them).  Populated by
    # runtime.py from ``_tier1_drift`` (literal diff vs the parent) and
    # ``_tier2_drift_pending`` ("parent has cloud change orders, so the
    # regeneration baseline under-describes it").  None on every path that
    # produces no drift evidence, so existing callers are unaffected.
    modify_drift_evidence: dict[str, Any] | None = None
    # Set ONLY by _invoke_t5() when its one-shot post-ladder
    # generalist attempt actually wins.  Distinct from
    # ResolvedWork.resolution_kind == "generalist", which fires for
    # ANY domain routed directly to the generalist (not a fallback
    # event) and is NOT the condition the 6.4 notice should key on.
    fallback_kind: str | None = None


@dataclass(frozen=True)
class CleanRouterDecision:
    """The router's classification output.

    The matrix sees only (domain, task_type, task_scope) — three fields.
    Everything else is router-internal: response envelope telemetry,
    executor/harness context, or router log diagnostics.
    """
    # --- Matrix-bound fields (the only fields the matrix sees) ---
    domain: str = "domain_other"
    task_type: str = "general_use"
    task_scope: str = "single_domain_single_task"

    # --- Router-internal fields (response envelope, log, executor context) ---
    trivial: bool = False
    detected_languages: list[str] = field(default_factory=list)
    probe_invoked: bool = False
    prompt: str = ""

    # --- Per-field confidences (response envelope only; not on matrix wire) ---
    domain_confidence: float = 0.0
    task_type_confidence: float = 0.0
    task_scope_confidence: float = 0.0
    trivial_confidence: float = 0.0

    # --- Step 5: descriptor-tag source (added 2026-08-04) ---
    # A short, one-line restatement (a few words to one sentence) of
    # what the user wants, in the router's own words.  Router-internal
    # telemetry, exactly like trivial_confidence -- not matrix-bound,
    # does not affect routing.  Consumed only by the artifact
    # descriptor-tag derivation.  Defaults to "" everywhere it isn't
    # explicitly set (heuristic path, override synthesis, probe path).
    user_intent: str = ""

    # --- Legacy (deprecated; may be removed) ---
    confidence: float = 0.0  # min of per-field scores


@dataclass(frozen=True)
class AttachmentMeta:
    """Metadata for an attachment as seen by the router."""
    kind: str  # "text" | "image" | "working_artifact"
    mime: str | None = None
    language: str | None = None
    size_bytes: int = 0
    width: int | None = None
    height: int | None = None
    section_names: list[str] = field(default_factory=list)


class CleanRouter(Protocol):
    """Router boundary; implementations may be heuristic, FLM, or Ollama-based."""

    async def classify(
        self,
        prompt: str,
        *,
        attachments: list[AttachmentMeta] | None = None,
        hints: dict[str, Any] | None = None,
        artifact_section_names: list[str] | None = None,
        prior_turn_produced_artifact: bool = False,
        routing_override: Any | None = None,
        artifact_identity_resolution: str | None = None,
    ) -> CleanRouterDecision:
        ...


class RouterFailure(Exception):
    """Raised when the router model fails to respond."""
    def __init__(self, kind: str, reason: str):
        self.kind = kind  # "timeout" | "network" | "parse_error"
        self.reason = reason
        super().__init__(f"{kind}: {reason}")


class NeedsClarification(Exception):
    """Raised by the router when artifact identity cannot be resolved."""
    def __init__(self, question: str, candidates: list[str]):
        self.question = question
        self.candidates = candidates
        super().__init__(question)


class RouterTaskUnsupported(Exception):
    """Raised when the classified task cannot run on the attached inputs.

    Distinct from RouterFailure (router model failed) and
    NeedsClarification (identity ambiguity): this is a POLICY hard stop
    with a user-facing message -- e.g. a bug_fix/modify_artifact request
    whose only attachment is an image, when no image-capable harness is
    wired.  The server surfaces ``message`` verbatim to the user.
    """
    def __init__(self, message: str, task_type: str = "", has_override: bool = False):
        self.message = message
        self.task_type = task_type
        self.has_override = has_override
        super().__init__(message)


class CleanTransport(Protocol):
    """Clean transport boundary; implementations use public provider APIs."""

    async def stream(self, package: CleanPackage) -> AsyncIterator[tuple[str, bool]]:
        """Yield final or thinking text chunks as ``(text, is_thinking)``."""
        ...

    async def health(self) -> bool:
        """Return whether the provider is reachable."""
        ...
