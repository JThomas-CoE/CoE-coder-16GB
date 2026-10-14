"""Clean execution composition root."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import CleanProgressCallback, CleanUnit
from .resolution import CleanHarness, ResolvedWork
from .runtime import execute_runtime_plan
from .session_memory import SessionMemory


# Package root, resolved at import time.  Used to anchor
# package-internal log directories (see ``raw_log_dir``
# default in ``CleanExecutor.__init__``) so the executor
# writes to a directory next to the package source rather
# than the process's CWD.  This keeps the new package's
# output self-contained regardless of where it is launched
# from.
_PACKAGE_ROOT = Path(__file__).resolve().parent


class CleanExecutor:
    """Execute resolved clean work without legacy agent, ladder, or backend code.

    The executor owns the tier-preparation cache.  Callers can inject a
    ``progress`` callback to receive token, summary, checkpoint, and
    completion events.  There is no wall-clock timeout at the executor
    level; cancellation is the callback's responsibility.

    A caller may also inject a ``session_memory`` (the rolling
    in-memory state of past turns) and an ``inject_last_turn`` flag.
    When present, the runtime forwards every progress event to the
    memory (in addition to the primary ``progress`` callback) and
    prepends the prior turn's user prompt + final output to the
    first tier's system message.  Benchmarks that treat each task
    as a single-instance scope should pass ``inject_last_turn=False``.
    """

    def __init__(
        self,
        transports: dict[str, Any],
        vram: Any,
        session_memory: SessionMemory | None = None,
        inject_last_turn: bool = True,
        # Path B remediation: the artifact store reference.
        # ``None`` (the default) means no artifact-aware
        # harness (modify_artifact/evaluate_artifact) can be
        # executed correctly -- callers that need those
        # harnesses to work must pass the session's
        # ArtifactStore instance.  The executor never
        # constructs its own store; it only reads/mutates
        # the one it is given.
        artifact_store: Any | None = None,
        # v14: evaluate-object holding pen (single-slot).  ``None`` (the
        # default) means no evaluate_artifact source/WP persistence; the
        # server passes its EvaluateObjectStore so a follow-up turn can
        # operate on the same source context.
        evaluate_store: Any | None = None,
        # v12: post-ladder T5 (generalist) parameters.
        # When the declared ladder is exhausted without
        # a passing attempt, the runtime may make ONE
        # final call to a generalist model.  Pass
        # ``None`` (the default) to disable T5.
        generalist_model: str | None = None,
        generalist_provider: str = "ollama",
        enable_t5: bool = True,
        # T3 (cloud_sota) escalation target, decoupled from T5 (local
        # generalist).  ``t3_model``/``t3_provider`` select the model+
        # transport for the cloud_sota meta-target.  ``None`` = cloud
        # escalation disabled (cloud_sota targets are skipped; T5 stays
        # the fallback).
        t3_model: str | None = None,
        t3_vision_model: str | None = None,
        t3_provider: str | None = None,
        # User-initiated direct cloud request.  False preserves the normal
        # local ladder and automatic post-ladder T3 behavior.
        direct_t3: bool = False,
        # Live per-token raw disk writer config.  The
        # clean slice is self-contained: it owns its
        # own internal raw token log directory.  The
        # runner does NOT need to know about this.
        #
        # ``raw_log_dir`` is the directory under which
        # the executor creates ``<task_id>.live.txt``
        # and ``<task_id>.live.meta.jsonl`` for every
        # task.  When ``None``, the executor derives
        # the path from ``cfg.log_dir`` (the FLM
        # log_dir in ``clean_config.yaml``) by
        # appending ``raw``.  Pass an explicit path
        # to override.
        #
        # ``raw_log_enabled`` is the master switch
        # (default True).  Set to False to disable
        # the live log entirely (e.g. in a test).
        raw_log_dir: str | Any | None = None,
        raw_log_enabled: bool = True,
    ) -> None:
        self.transports = transports
        self.vram = vram
        self.session_memory = session_memory
        self.inject_last_turn = inject_last_turn
        self.artifact_store = artifact_store
        self.evaluate_store = evaluate_store
        self.generalist_model = generalist_model
        self.generalist_provider = generalist_provider
        self.enable_t5 = enable_t5
        # T3 target: cloud model/provider (None = cloud disabled).
        self.t3_model = t3_model
        self.t3_vision_model = t3_vision_model
        self.t3_provider = t3_provider
        # The clean slice owns its own internal log
        # directory.  Default is ``<package>/logs/raw``
        # which lives inside the package.  This
        # path is slice-internal: the runner does
        # not need to know or care where the
        # per-token stream lives.
        if raw_log_dir is None:
            raw_log_dir = str(_PACKAGE_ROOT / "logs" / "raw")
        self.raw_log_dir = str(raw_log_dir)
        self.raw_log_enabled = bool(raw_log_enabled)
        # Process-wide call counter for self-generated
        # internal call ids.  Threading lock so the
        # counter is monotonic across concurrent calls
        # (the executor is async-only; a Lock is
        # sufficient).
        import threading
        self._id_lock = threading.Lock()
        self._id_counter = 0

    async def execute(
        self,
        work: ResolvedWork,
        prompt: str,
        progress: CleanProgressCallback | None = None,
        checkpoint_interval_s: float = 30.0,
        first_token_timeout_s: float = 60.0,
        slow_decode_timeout_s: float = 90.0,
        # v12: per-instance closed-think override.  When
        # ``None`` (the default), the harness's declared
        # ``closed_think`` flag is used (True for
        # general_use, python_new_code, python_bug_fix).
        # When ``True``, force the closed-think block on
        # regardless of the harness.  When ``False`",
        # force it off regardless of the harness.  This
        # is the per-instance/per-session override: a
        # caller that wants to disable closed-think for
        # an entire session passes ``False`` on every
        # ``execute()`` call in that session.
        closed_think_override: bool | None = None,
        # v15: per-instance native-reasoning override.  ``None`` (default)
        # uses the model-card ``think`` default.  ``True`` forces think-on,
        # ``False`` forces think-off.  Wins over the model-card default.
        think_override: bool | None = None,
        seed_override: int | None = None,
        # Per-call session-injection override. ``None`` (default) uses
        # the executor's construction-level ``inject_last_turn`` flag.
        inject_last_turn_override: bool | None = None,        # User-initiated direct cloud request.  Skips local execution and
        # invokes the configured T3 target once.
        direct_t3: bool = False,
        # True when /evaluate explicitly selected the immediately prior
        # work product as the source object.
        evaluate_prior_wp: bool = False,
        # Path B remediation: which artifact (by id) the
        # artifact-aware harnesses (modify_artifact,
        # evaluate_artifact) should operate on.  ``None``
        # (the default) resolves to ``self.artifact_store.current``
        # -- whichever artifact the session already has
        # selected (via the existing switch()/
        # artifact_identity_resolution mechanism upstream
        # in server.py).  This parameter exists so a future
        # tag/chronological resolver (Step 5/6 scope) can
        # supply a specific id without any change to this
        # method's signature semantics.
        artifact_id: str | None = None,
        # Step 5: the router's one-line restatement of what the
        # user wants (``CleanRouterDecision.user_intent``) and the
        # attachment's original filename (if any), threaded through
        # to the two-round modify_artifact block for descriptor-tag
        # derivation.  Both default to inert values -- no existing
        # caller's behavior changes unless it explicitly opts in.
        user_intent: str = "",
        source_filename: str | None = None,
        # v13 step 2 (2026-08-08): the submitted file content for the
        # bug_fix / unified_diff_patch context-correctness gate.  When
        # provided, the contract selects the LAST <patch> block whose
        # removed lines all appear in this content, and rejects if none
        # do.  ``None`` (the default) keeps the legacy take-last
        # behavior -- harnesses that don't provide the file (e.g. the
        # LCB runner) are unaffected.
        patch_base: str | None = None,
        context_text: str = "",
        images: list[str] | None = None,
        t3_context_text: str = "",
        # 2026-09-14 (spec 5.7.2 PATH A): local REGENERATION for an artifact
        # that carries its originating prompt.  When ``modify_combo_prompt``
        # is supplied, the runtime replaces the change-order instruction with
        # this combined prompt and produces a COMPLETE artifact; the
        # deterministic re-integration step is then skipped entirely, since
        # there is no change order to apply.  Both default to inert values,
        # so every existing caller is unaffected.
        modify_combo_prompt: str | None = None,
        modify_skip_reintegration: bool = False,
        # 2026-09-14 (spec 4.1): PATH A's regeneration WORDING, declared on
        # the executing harness and seeded into the runtime dict.  Passed
        # through as opaque strings so the executor needs no knowledge of
        # the registry.  Both are optional, so an undeclared deployment
        # behaves exactly as before.
        #
        # 2026-09-17 (user ruling): `modify_regeneration_contract` is
        # RETIRED -- the contract is selected from the ARTIFACT'S LANGUAGE
        # (`denovo_contract_for_language`), never from config.
        modify_regeneration_system_suffix: str | None = None,
        modify_regeneration_output_instruction: str | None = None,
        # 2026-09-14 (spec 6.1-6.4): master switch for the PATH A drift
        # checks.  ``None`` = not specified (runtime default applies).
        modify_drift_checks: bool | None = None,
    ) -> Any:
        """Execute the runtime plan for a single task.

        The clean slice is self-contained: it owns the
        live per-token raw writer.  The caller (e.g. the
        LCB bench runner) does NOT need to pass any
        writer-related arguments; the executor
        auto-constructs a ``RawTokenWriter`` from its
        own ``raw_log_dir`` and a self-generated
        internal call id, and ensures the writer is
        closed on every exit path.

        The internal call id is a slice-local
        monotonic counter + wall-clock timestamp, e.g.
        ``call-000123-2026-07-18T17-30-45``.  The
        runner's question_id is irrelevant to the
        clean slice and is not used.

        ``closed_think_override``: per-instance override
        for the closed-think block injection.  ``None``
        uses the harness's declared value; ``True`` forces
        it on; ``False`` forces it off.  See the YAML's
        ``closed_think`` field docstring for the default
        behavior.
        """
        harness = CleanHarness(work, think_override=think_override)
        # Resolve the effective closed-think flag.  The
        # override (if not None) wins over the harness.
        # The harness's value is ``None`` (defaults to True
        # in the composer at clean_resolution.py:270-271),
        # ``True``, or ``False``; we normalize ``None`` to
        # True so the override comparison is straightforward.
        harness_closed_think = work.harness.closed_think
        if harness_closed_think is None:
            harness_closed_think = True
        effective_closed_think = (
            bool(closed_think_override)
            if closed_think_override is not None
            else bool(harness_closed_think)
        )
        unit = CleanUnit(
            domain=work.domain,
            prompt=prompt,
            persona=work.persona,
            closed_think_block="<think>\n</think>" if effective_closed_think else None,
            context_text=context_text,
            t3_context_text=t3_context_text,
            images=list(images or []),
        )
        if work.primary_provider not in self.transports:
            raise RuntimeError(f"No clean transport configured for provider '{work.primary_provider}'")
        # Self-manage a RawTokenWriter with an
        # internally generated call id.  The runner
        # never sees this id; the file lives in the
        # clean slice's own log directory.
        writer: Any = None
        if self.raw_log_enabled:
            try:
                from .raw_writer import RawTokenWriter
                call_id = self._next_call_id()
                writer = RawTokenWriter(self.raw_log_dir, call_id)
            except Exception as exc:
                # Writer init failure must NEVER break
                # the inference; fall through to None.
                import sys
                print(f"[warn] CleanExecutor RawTokenWriter init failed: {exc}", file=sys.stderr)
                writer = None
        try:
            return await execute_runtime_plan(
                work=work,
                harness=harness,
                transports=self.transports,
                vram=self.vram,
                unit=unit,
                progress=progress,
                checkpoint_interval_s=checkpoint_interval_s,
                first_token_timeout_s=first_token_timeout_s,
                slow_decode_timeout_s=slow_decode_timeout_s,
                session_memory=self.session_memory,
                inject_last_turn=(
                    inject_last_turn_override
                    if inject_last_turn_override is not None
                    else self.inject_last_turn
                ),
                artifact_store=self.artifact_store,
                artifact_id=artifact_id,
                evaluate_store=self.evaluate_store,
                user_intent=user_intent,
                source_filename=source_filename,
                patch_base=patch_base,
                context_text=context_text,
                t3_context_text=t3_context_text,
                modify_combo_prompt=modify_combo_prompt,
                modify_skip_reintegration=modify_skip_reintegration,
                modify_regeneration_system_suffix=modify_regeneration_system_suffix,
                modify_regeneration_output_instruction=modify_regeneration_output_instruction,
                modify_drift_checks=modify_drift_checks,
                generalist_model=self.generalist_model,
                generalist_provider=self.generalist_provider,
                enable_t5=self.enable_t5,
                t3_model=self.t3_model,
                t3_vision_model=self.t3_vision_model,
                t3_provider=self.t3_provider,
                direct_t3=direct_t3,
                evaluate_prior_wp=evaluate_prior_wp,
                raw_token_writer=writer,
                think_override=think_override,
                seed_override=seed_override,
            )
        finally:
            # Close the writer on every exit path
            # (success, exception, asyncio.CancelledError
            # from an outer wait_for).  The finally
            # block runs even on cancellation because
            # Python guarantees that.
            if writer is not None:
                try:
                    await writer.close()
                except Exception:
                    pass

    def _next_call_id(self) -> str:
        """Generate a slice-internal unique call id.

        Format: ``call-NNNNNN-YYYYMMDDTHHMMSS`` where
        NNNNNN is a process-wide monotonic counter.
        The id is intentionally NOT derived from the
        runner's question_id: the clean slice does
        not know or care what the runner calls its
        tasks.
        """
        with self._id_lock:
            self._id_counter += 1
            n = self._id_counter
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        return f"call-{n:06d}-{ts}"

    def reset_for_next_probe(self) -> None:
        """Clear the runtime's tier-preparation cache."""
        execute_runtime_plan.last_prepared_key = None
