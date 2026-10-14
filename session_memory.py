"""Rolling session memory for the clean slice runtime.

The session memory is a system-internal resource that maintains
recent context across turns.  It serves two purposes:

1. **Past-turn recall**: ``get_last_turn()`` returns the most
   recently completed turn as a ``SessionMemoryEntry``.  The
   runtime uses this to enhance the next prompt with a labeled
   section describing the prior user prompt and model output,
   so follow-up queries like "do x with the code you just
   wrote" work without re-stating context.

2. **Live consumption of the streaming event surface**: the
   session memory is itself a ``CleanProgressCallback``
   subscriber.  The runtime hands it every event as it streams
   (token, summary, tier_start, tier_complete, checkpoint,
   complete).  Other consumers (bench recorders, the
   user-facing live streamer) can subscribe to the same event
   stream in parallel without conflict.

Turn boundary
-------------
A "turn" is bounded by the runtime's existing event markers:

* the **first** ``tier_start`` event of a run is the start of a
  new turn
* the **final** ``complete`` event of a run is the end of a turn

When the runtime emits the final ``complete`` event the session
memory **freezes** the in-flight buffer (tokens, summaries,
decisions) into a new ``SessionMemoryEntry``, appends it to
the persistent list, and runs the eviction policy.

Eviction policy
---------------
The memory has a configurable token cap (default 1,000,000).
When the persistent list exceeds the cap, the **oldest complete
turn** is evicted.  Eviction never crosses a turn boundary —
the current turn is always recorded in full even if its content
pushes the total above the cap (the next turn will trigger
eviction).  The current 16k per-turn context cap makes this
trivially safe.

Token estimation
----------------
Tokens are estimated with a rough ``len(text) // 4`` rule for
now.  This is good enough for the cap check; the search /
chunking V2 will need a real tokenizer.

V1 last-turn injection format
------------------------------
The minimal header is appended to the system prompt as a
labeled section:

    ---

    The following is the user prompt and model output for the
    prior turn.  It is given purely for context and may or may
    not be relevant to the current task.

    Prior user prompt:
    ```
    <user prompt>
    ```

    Prior model output:
    ```
    <model output>
    ```

    ---

V2 will route the prior turn through a router/parser model
for a relevance judgment before injection, with a richer
header that includes the judgment.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from .contracts import CleanProgressCallback, CleanProgressEvent


# Default cap is 1M tokens.  Trivial to fit in memory; the
# per-turn context cap (16k) means we can hold ~60 turns at
# full size before eviction kicks in.  Override via the
# ``max_tokens`` constructor arg.
DEFAULT_MAX_TOKENS = 1_000_000


def estimate_tokens(text: str) -> int:
    """Rough token estimate: 4 chars per token on average.

    Good enough for cap enforcement.  A real tokenizer (V2)
    can replace this without changing the API.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass
class SessionMemoryEntry:
    """A frozen record of one completed turn.

    Captured at the moment the runtime emits the final
    ``complete`` event.  Contains the user prompt (original,
    unmodified), the system-modified prompt (what actually
    went to the model), the streamed model output, and
    routing/resolver decisions.

    The full in-flight event stream is *not* stored per
    entry — only the final state.  Live consumers (v2
    router, live streamer) can subscribe to the runtime's
    event surface in parallel for real-time data.
    """
    task_id: str
    timestamp: float
    user_prompt: str
    system_modified_prompt: str
    model_output: str
    routing_decision: dict[str, Any] = field(default_factory=dict)
    resolver_decision: dict[str, Any] = field(default_factory=dict)
    persona: str | None = None
    harness_id: str | None = None
    primary_model: str | None = None
    primary_provider: str | None = None
    elapsed_s: float = 0.0
    token_count: int = 0  # estimated total tokens for this entry

    # --- Step 5 (2026-08-04): artifact provenance for this turn ---
    # Set AFTER this entry has already been frozen and appended --
    # see SessionMemory.attach_artifact_to_last_turn().  Freeze
    # happens strictly before artifact commit (the runtime's
    # ``complete`` event, which triggers the freeze, fires before
    # server.py's post-execute() ArtifactStore commit), so these
    # two fields cannot be populated at freeze time itself.
    produced_artifact: bool = False
    artifact_id: str | None = None

    def estimated_tokens(self) -> int:
        """Re-estimate the token count from current text content."""
        total = (
            estimate_tokens(self.user_prompt)
            + estimate_tokens(self.system_modified_prompt)
            + estimate_tokens(self.model_output)
        )
        # Routing/resolver dicts are tiny; estimate generously.
        total += estimate_tokens(str(self.routing_decision))
        total += estimate_tokens(str(self.resolver_decision))
        return total


@dataclass
class _InFlightBuffer:
    """The in-flight (not yet frozen) state of the current turn.

    Receives live events from the runtime.  When the final
    ``complete`` event arrives, the buffer is frozen into a
    ``SessionMemoryEntry`` and cleared for the next turn.
    """
    user_prompt: str = ""
    system_modified_prompt: str = ""
    tokens: list[str] = field(default_factory=list)
    routing_decision: dict[str, Any] = field(default_factory=dict)
    resolver_decision: dict[str, Any] = field(default_factory=dict)
    persona: str | None = None
    harness_id: str | None = None
    primary_model: str | None = None
    primary_provider: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    last_tier: str = ""

    def append_token(self, text: str) -> None:
        if text:
            self.tokens.append(text)

    @property
    def model_output(self) -> str:
        return "".join(self.tokens)

    def reset(self) -> None:
        self.user_prompt = ""
        self.system_modified_prompt = ""
        self.tokens = []
        self.routing_decision = {}
        self.resolver_decision = {}
        self.persona = None
        self.harness_id = None
        self.primary_model = None
        self.primary_provider = None
        self.started_at = 0.0
        self.finished_at = 0.0
        self.last_tier = ""


class SessionMemory(CleanProgressCallback):
    """Rolling session memory for the clean slice.

    Implements ``CleanProgressCallback`` so the runtime can
    hand it events like any other consumer.  Also exposes
    retrieval methods (``get_last_turn``, ``get_recent``,
    ``get_live_in_progress``) for the runtime's prompt
    enhancement and for future consumers (v2 router, live
    streamer).

    Parameters
    ----------
    max_tokens
        Cap on the total estimated tokens across all frozen
        entries.  When the cap is exceeded, the oldest
        complete turn is evicted (never the in-flight one).
        Default: ``DEFAULT_MAX_TOKENS`` (1,000,000).
    """

    def __init__(self, *, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        self._entries: list[SessionMemoryEntry] = []
        self._max_tokens = int(max_tokens)
        self._in_flight = _InFlightBuffer()
        self._current_task_id: str = ""
        # V2: chunk index (Layer 3) and turn summary index (Layer 2).
        # Lazily imported to avoid circular dependency (clean_chunker
        # imports estimate_tokens from this module).
        self._chunk_index = None
        self._summary_index = None
        self._turn_counter: int = 0

    def _ensure_v2_indexes(self) -> None:
        """Lazy-initialize the V2 chunk and summary indexes.

        Defers the import to break the circular dependency:
        ``clean_chunker`` imports ``estimate_tokens`` from this
        module, so we can't import ``clean_chunker`` at the top.
        """
        if self._chunk_index is not None:
            return
        from .chunk_index import BM25Index
        from .turn_summary import TurnSummaryIndex
        self._chunk_index = BM25Index()
        self._summary_index = TurnSummaryIndex()

    # ------------------------------------------------------------------
    # CleanProgressCallback implementation
    # ------------------------------------------------------------------

    def on_event(self, event: CleanProgressEvent) -> Any:
        et = event.event_type
        if et == "tier_start":
            # A new turn begins at the first tier_start of a run.
            # If the in-flight buffer is already populated, we
            # assume this is a sub-tier of the same turn and
            # don't reset.
            if not self._in_flight.started_at:
                self._in_flight.started_at = time.monotonic()
            self._in_flight.last_tier = event.tier or self._in_flight.last_tier
        elif et == "token":
            self._in_flight.append_token(event.text or "")
        elif et == "tier_complete":
            # The runtime may emit tier_complete with the
            # partial output for that tier; we let the final
            # ``complete`` event do the freeze.  Nothing to
            # do here other than remember the tier name.
            self._in_flight.last_tier = event.tier or self._in_flight.last_tier
        elif et == "complete":
            self._in_flight.finished_at = time.monotonic()
            # Prefer the runtime's authoritative final output
            # whenever provided -- not just when tokens is empty.
            # For modify_artifact this is the short internal marker
            # set by runtime.py's two-round block (Step 5.9), not the
            # raw accumulated R1/R2 token stream, which would
            # otherwise leak internal JSON into the turn's recorded
            # model_output (a session-memory quality concern distinct
            # from, but related to, the client-facing SSE leak fixed
            # in Step 6.3).
            if event.output:
                self._in_flight.tokens = [event.output]
            self._freeze_and_record()
        return None

    def _freeze_and_record(self) -> None:
        """Convert the in-flight buffer into a frozen entry and
        append to the persistent list.  Run eviction if over cap.
        """
        inflight = self._in_flight
        if not self._current_task_id and not inflight.user_prompt and not inflight.tokens:
            # Nothing was ever recorded for this turn; skip.
            return
        entry = SessionMemoryEntry(
            task_id=self._current_task_id or "unknown",
            timestamp=inflight.finished_at or time.time(),
            user_prompt=inflight.user_prompt,
            system_modified_prompt=inflight.system_modified_prompt,
            model_output=inflight.model_output,
            routing_decision=dict(inflight.routing_decision),
            resolver_decision=dict(inflight.resolver_decision),
            persona=inflight.persona,
            harness_id=inflight.harness_id,
            primary_model=inflight.primary_model,
            primary_provider=inflight.primary_provider,
            elapsed_s=(inflight.finished_at - inflight.started_at) if inflight.started_at and inflight.finished_at else 0.0,
        )
        entry.token_count = entry.estimated_tokens()
        self._entries.append(entry)
        # V2: chunk the model output and index it
        self._v2_freeze_chunks(entry)
        self._evict_to_cap()
        # Reset for the next turn.
        inflight.reset()
        self._current_task_id = ""

    def attach_artifact_to_last_turn(self, artifact_id: str) -> None:
        """Update the last FROZEN entry's artifact fields, after the fact.

        Used by server.py because artifact commit now happens
        strictly after ``execute()`` returns -- i.e. after the turn's
        ``complete`` event already froze this entry (Step 5.9/5.10's
        freeze-before-artifact ordering). This mutates
        ``self._entries[-1]`` directly, which is exactly the entry
        frozen for the turn that just completed. No-op if no turn has
        frozen yet (e.g. called before any turn has run).

        Note: an earlier draft of this mechanism
        (``mark_produced_artifact()``) mutated ``self._in_flight``
        instead -- but ``_freeze_and_record()`` already resets
        ``_in_flight`` before ``execute()`` returns, so that approach
        would silently corrupt the NEXT turn's in-flight buffer
        rather than annotate the turn that just completed. This
        method mutates the frozen entry directly and has no such bug.
        """
        if not self._entries:
            return
        self._entries[-1].produced_artifact = True
        self._entries[-1].artifact_id = artifact_id

    def _v2_freeze_chunks(self, entry: SessionMemoryEntry) -> None:
        """Chunk the model output and insert into V2 indexes.

        Called after a turn is frozen.  Builds chunks via
        ``clean_chunker.chunk_model_output()``, inserts them
        into the BM25 chunk index, and builds a deterministic
        ``TurnSummary`` for the turn summary index.
        """
        self._ensure_v2_indexes()
        if not entry.model_output or not entry.model_output.strip():
            # Still record a minimal summary for chronological completeness
            from .turn_summary import deterministic_extract
            summary = deterministic_extract(
                turn_id=entry.task_id,
                seq=self._turn_counter,
                user_prompt=entry.user_prompt,
                model_output="",
                chunks=[],
                routing_decision=entry.routing_decision,
                timestamp=entry.timestamp,
            )
            self._summary_index.insert(summary)
            self._turn_counter += 1
            return

        from .chunker import chunk_model_output
        from .turn_summary import deterministic_extract

        # Detect language from routing decision or harness_id
        language = entry.routing_decision.get("domain") if entry.routing_decision else None

        # Chunk the model output
        chunks = chunk_model_output(entry.model_output, entry.task_id, language)

        # Insert chunks into BM25 index
        self._chunk_index.insert_many(chunks)

        # Build and insert turn summary
        summary = deterministic_extract(
            turn_id=entry.task_id,
            seq=self._turn_counter,
            user_prompt=entry.user_prompt,
            model_output=entry.model_output,
            chunks=chunks,
            routing_decision=entry.routing_decision,
            timestamp=entry.timestamp,
        )
        self._summary_index.insert(summary)
        self._turn_counter += 1

    def _evict_to_cap(self) -> None:
        """Drop oldest complete turns until total <= cap.

        Eviction never crosses a turn boundary — the most
        recent (and any in-flight) entry is preserved.
        V2: also evicts the evicted turn's chunks and summary.
        """
        while self._entries and self._total_tokens() > self._max_tokens:
            evicted = self._entries.pop(0)
            # V2: clean up chunk and summary indexes
            if self._chunk_index is not None:
                self._chunk_index.remove_turn(evicted.task_id)
            if self._summary_index is not None:
                self._summary_index.remove(evicted.task_id)

    def _total_tokens(self) -> int:
        return sum(e.token_count for e in self._entries)

    # ------------------------------------------------------------------
    # Configuration setters (called by the runtime before/at
    # the start of a turn so the in-flight buffer captures the
    # user prompt, system prompt, and resolver decisions).
    # ------------------------------------------------------------------

    def begin_turn(
        self,
        *,
        task_id: str,
        user_prompt: str,
        system_modified_prompt: str = "",
        routing_decision: dict[str, Any] | None = None,
        resolver_decision: dict[str, Any] | None = None,
        persona: str | None = None,
        harness_id: str | None = None,
        primary_model: str | None = None,
        primary_provider: str | None = None,
    ) -> None:
        """Mark the start of a new turn and seed the in-flight
        buffer with the prompt-side metadata.

        The runtime calls this once per turn, *before* any
        token events are emitted, so the in-flight buffer
        already has the user prompt and system-modified prompt
        when the first token arrives.  If the buffer is
        non-empty from a prior turn (the freeze didn't run),
        discard it — only one in-flight turn is supported.
        """
        self._in_flight.reset()
        self._current_task_id = task_id
        self._in_flight.user_prompt = user_prompt
        self._in_flight.system_modified_prompt = system_modified_prompt
        if routing_decision is not None:
            self._in_flight.routing_decision = dict(routing_decision)
        if resolver_decision is not None:
            self._in_flight.resolver_decision = dict(resolver_decision)
        if persona is not None:
            self._in_flight.persona = persona
        if harness_id is not None:
            self._in_flight.harness_id = harness_id
        if primary_model is not None:
            self._in_flight.primary_model = primary_model
        if primary_provider is not None:
            self._in_flight.primary_provider = primary_provider

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_last_turn(self) -> SessionMemoryEntry | None:
        """Return the most recent frozen entry, or None."""
        return self._entries[-1] if self._entries else None

    def get_recent(self, n: int) -> list[SessionMemoryEntry]:
        """Return the last ``n`` frozen entries (most-recent-last)."""
        if n <= 0:
            return []
        return list(self._entries[-n:])

    def get_live_in_progress(self) -> _InFlightBuffer | None:
        """Return the current in-flight buffer (if any), or None.

        V1 does not use this; V2 router/parser can use it to
        inspect the model output as it streams.
        """
        if not self._in_flight.tokens and not self._in_flight.user_prompt:
            return None
        return self._in_flight

    def search(self, query: str, *, k: int = 5) -> list[SessionMemoryEntry]:
        """V2 stub.  Returns ``[]`` until chunking/indexing ships.

        The contract is: given a query string (e.g. an
        embedding or a substring), return the top ``k``
        entries ranked by relevance.  V1 returns nothing.
        """
        return []

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Discard all frozen entries and the in-flight buffer.

        Used by tests and by operators who want a fresh
        session.  Does not affect the runtime's event
        subscription — the memory is still in the callback
        chain, just empty.
        """
        self._entries.clear()
        self._in_flight.reset()
        self._current_task_id = ""
        # V2: clear chunk and summary indexes
        if self._chunk_index is not None:
            self._chunk_index.clear()
        if self._summary_index is not None:
            self._summary_index.clear()
        self._turn_counter = 0

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[SessionMemoryEntry]:
        return iter(self._entries)

    @property
    def total_tokens(self) -> int:
        return self._total_tokens()

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def in_flight_tokens(self) -> int:
        return estimate_tokens(self._in_flight.model_output)

    # ------------------------------------------------------------------
    # V2 retrieval methods
    # ------------------------------------------------------------------

    @property
    def chunk_index(self):
        """Access the V2 BM25 chunk index (Layer 3).

        Returns None if no turns have been frozen yet.
        """
        return self._chunk_index

    @property
    def summary_index(self):
        """Access the V2 turn summary index (Layer 2).

        Returns None if no turns have been frozen yet.
        """
        return self._summary_index

    def get_recent_turn_ids(self, n: int = 2) -> list[str]:
        """Return the turn IDs of the last N frozen entries."""
        if n <= 0:
            return []
        return [e.task_id for e in self._entries[-n:]]

    def build_memory_section(
        self,
        current_prompt: str,
        *, budget_tokens: int = 4000,
    ) -> str:
        """Build a V2 memory injection section for the current prompt.

        This is the main V2 entry point.  It:
        1. Runs the heuristic memory relevance gate
        2. Assembles the labeled memory section from
           chronological + semantic layers
        3. Returns the section (or empty string if no injection)

        Falls back to the V1 ``build_last_turn_section()`` if
        the V2 indexes are not yet initialized.
        """
        if self._chunk_index is None or self._summary_index is None:
            # V1 fallback
            last = self.get_last_turn()
            if last is None:
                return ""
            return build_last_turn_section(last)

        from .turn_summary import heuristic_memory_relevance
        from .memory_inject import build_memory_section as _build

        recent_summaries = self._summary_index.get_recent(3)
        recent_chunks = []
        for tid in self.get_recent_turn_ids(3):
            recent_chunks.extend(self._chunk_index.get_chunks_for_turn(tid))

        relevance = heuristic_memory_relevance(
            current_prompt, recent_summaries, recent_chunks,
        )

        return _build(
            relevance, current_prompt,
            self._summary_index, self._chunk_index,
            budget_tokens=budget_tokens,
            recent_turn_ids=self.get_recent_turn_ids(2),
        )


# ---------------------------------------------------------------------------
# Last-turn injection helper
# ---------------------------------------------------------------------------

LAST_TURN_HEADER = (
    "\n\n---\n"
    "The following is the user prompt and model output for the prior turn. "
    "It is given purely for context and may or may not be relevant to the current task.\n\n"
)


def build_last_turn_section(entry: SessionMemoryEntry) -> str:
    """Render a ``SessionMemoryEntry`` as a labeled section for
    system-prompt injection.

    The section is appended to the system message of the next
    turn so the model sees the prior turn's user prompt and
    final output.  The header describes the section so the
    model can use it for context without confusing it for the
    current task.

    Returns an empty string if the entry has no model output
    (the section would be useless).
    """
    if not entry.model_output and not entry.user_prompt:
        return ""
    parts = [LAST_TURN_HEADER]
    if entry.user_prompt:
        parts.append("Prior user prompt:\n")
        parts.append("```\n")
        parts.append(entry.user_prompt)
        if not entry.user_prompt.endswith("\n"):
            parts.append("\n")
        parts.append("```\n\n")
    if entry.model_output:
        parts.append("Prior model output:\n")
        parts.append("```\n")
        parts.append(entry.model_output)
        if not entry.model_output.endswith("\n"):
            parts.append("\n")
        parts.append("```\n")
    parts.append("---\n")
    return "".join(parts)
