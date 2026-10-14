"""Pathology monitor for clean-slice runtime.

The monitor is the "no/slow token" timer that the runtime
consults during a streaming response.  It owns:

  * A prefill-rate table (loaded from ``prefill_rates.json``
    at construction time).
  * A clock that starts when the prompt package goes on the
    wire (``start()``), and stops at first token
    (``on_first_token()``).
  * A decode-rate check that fires when generation stalls
    below a configurable threshold (default 5 tok/s
    sustained over a 30s window).
  * A lookup function that returns a dynamic TTFT budget
    for a given (model, cold/warm, prompt size).

The monitor is **synchronous** and **stateless across calls**
except for the clock state during one call.  The runtime
creates a fresh instance per stream, calls ``start()`` when
the request bytes go out, ``on_token()`` on every received
chunk, ``on_first_token()`` once, and ``check()`` from the
per-chunk loop to ask "should I abort?".

Design choices:

  * **No tokenizer dependency.**  The prompt token
    estimate comes from the runtime, which already has
    access to the chat message list.  We use a rough
    ``len(text)//4`` heuristic for v1; the 2x safety
    factor absorbs the error.

  * **Hardware-portable.**  The prefill-rate table is a
    per-machine artifact.  If the table is missing a model,
    the monitor falls back to a conservative default rate
    so the runtime still works on a fresh machine.

  * **Cold vs warm is decided at call start, not at
    monitor construction.**  The runtime queries
    ``vram.state()`` once per call and passes the
    thermal-state to the monitor.
"""
from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any


# Where the prefill-rate table is loaded from by default.
DEFAULT_TABLE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "prefill_rates.json",
)

# Safety factor on top of the predicted prefill time.
# 2.0 means "wait at most twice the predicted TTFT before
# declaring a pathology".  1.0 would be too tight (any
# jitter trips pathology).  3.0 would be too loose (we
# wait 3x the predicted TTFT, defeating the purpose).
DEFAULT_SAFETY_FACTOR = 2.0

# Hard ceiling on the TTFT budget regardless of the
# formula.  Even a tiny prompt shouldn't be allowed to
# wait forever; a 30GB model on a slow disk can take
# 2-3 minutes cold.  600s = 10 minutes is a generous
# outer cap that still fires before the 5-minute
# per-problem wall in the LCB harness.
DEFAULT_TTFT_CEILING_S = 600.0

# Minimum TTFT budget floor.  Even the smallest prompt
# needs SOME time to make a network round-trip and start
# prefill.  5s prevents pathological over-aborts on fast
# models.
DEFAULT_TTFT_FLOOR_S = 5.0

# Decode-rate threshold: sustained below this rate (tok/s)
# over a window, declare slow_decode pathology.
DEFAULT_MIN_DECODE_RATE_TOK_PER_S = 5.0

# Window over which to compute the decode rate.
DEFAULT_DECODE_WINDOW_S = 30.0

# Conservative fallback rate when the table has no entry
# for the (model, state, size) combo.  30 tok/s is slow
# enough that the safety factor produces a generous
# budget, so missing entries do not over-abort.
FALLBACK_PREFILL_RATE_TOK_PER_S = 30.0

# Bucket sizes for prefill-rate lookup, in tokens.  The
# runtime rounds its prompt-token estimate to the nearest
# bucket and uses that bucket's measured rate.  This is
# the same set used by ``_measure_prefill.py`` so the
# lookup never misses by more than a factor of 2-3.
BUCKET_SIZES = [2000, 5000, 12500]


@dataclass
class PathologyVerdict:
    """Result of a single ``check()`` call.

    Attributes:
        abort: True if the runtime should pull the plug.
        reason: One of "ttft", "slow_decode", or None.
        budget_s: The TTFT budget that was in effect
            (seconds from ``start()``).
        elapsed_s: Seconds since ``start()``.
        observed_rate: Tokens/sec over the decode window
            (None before first token).
    """
    abort: bool
    reason: str | None
    budget_s: float
    elapsed_s: float
    observed_rate: float | None


@dataclass
class PathologyMonitor:
    """Per-stream pathology monitor.

    Construct one per call.  ``start()`` is called when
    the request bytes go on the wire.  ``on_token()`` is
    called for every received chunk.  ``check()`` is
    polled from the per-chunk loop and returns a verdict.
    """
    model_id: str
    is_resident: bool
    prefill_table: dict[str, Any] = field(default_factory=dict)
    safety_factor: float = DEFAULT_SAFETY_FACTOR
    min_decode_rate: float = DEFAULT_MIN_DECODE_RATE_TOK_PER_S
    decode_window_s: float = DEFAULT_DECODE_WINDOW_S
    ttft_ceiling_s: float = DEFAULT_TTFT_CEILING_S
    ttft_floor_s: float = DEFAULT_TTFT_FLOOR_S

    # State set by start()
    _start_monotonic: float = 0.0
    _started: bool = False
    _first_token_monotonic: float | None = None
    _token_count: int = 0
    _token_times: list[float] = field(default_factory=list)
    _last_token_monotonic: float | None = None
    # Cached prefill rate and budget.
    _prefill_rate: float = FALLBACK_PREFILL_RATE_TOK_PER_S
    _ttft_budget_s: float = 0.0

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def from_table_file(
        cls,
        *,
        model_id: str,
        is_resident: bool,
        prompt_tokens_estimate: int,
        path: str = DEFAULT_TABLE_PATH,
        safety_factor: float = DEFAULT_SAFETY_FACTOR,
    ) -> "PathologyMonitor":
        """Build a monitor with a prefill budget already computed.

        The monitor is returned in the "ready to start" state
        but ``start()`` must still be called when the bytes
        go on the wire so the clock is anchored correctly.
        """
        table: dict[str, Any] = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    table = json.load(f)
            except Exception:
                table = {}
        m = cls(
            model_id=model_id,
            is_resident=is_resident,
            prefill_table=table,
            safety_factor=safety_factor,
        )
        # Compute the rate and budget up-front using the
        # caller's prompt size estimate.  This is the
        # principled prefill time = size / rate, doubled
        # for safety, clamped to [floor, ceiling].
        rate = m.lookup_prefill_rate(model_id, is_resident, prompt_tokens_estimate)
        m._prefill_rate = rate
        budget = (prompt_tokens_estimate / rate) * safety_factor if rate > 0 else DEFAULT_TTFT_CEILING_S
        m._ttft_budget_s = max(m.ttft_floor_s, min(m.ttft_ceiling_s, budget))
        return m

    # ------------------------------------------------------------------ #
    # Prefill rate lookup
    # ------------------------------------------------------------------ #

    @staticmethod
    def _round_to_bucket(tokens: int) -> int:
        """Round a token count to the nearest prefill-rate bucket."""
        if not BUCKET_SIZES:
            return tokens
        return min(BUCKET_SIZES, key=lambda b: abs(b - tokens))

    def lookup_prefill_rate(
        self, model_id: str, is_resident: bool, prompt_tokens: int
    ) -> float:
        """Return the measured prefill rate for (model, state, size).

        The rate is in tokens/sec.  Falls back to a
        conservative default if the model is not in the
        table or the bucket is missing.
        """
        state = "warm" if is_resident else "cold"
        bucket = str(self._round_to_bucket(prompt_tokens))
        model_entry = self.prefill_table.get(model_id) or {}
        state_entry = model_entry.get(state) or {}
        bucket_entry = state_entry.get(bucket)
        if isinstance(bucket_entry, dict):
            rate = bucket_entry.get("median_rate_tok_per_s")
            if isinstance(rate, (int, float)) and rate > 0:
                return float(rate)
        elif isinstance(bucket_entry, (int, float)) and bucket_entry > 0:
            # Tolerate the older flat-rate format.
            return float(bucket_entry)
        # Try a same-family match: many pruned variants
        # share the parent model's prefill rate.  Walk
        # upward in the basename and try again.
        basename = model_id.split(":", 1)[0]
        for try_id in list(self.prefill_table.keys()):
            try_base = try_id.split(":", 1)[0]
            # Match if the try_id basename is a prefix of
            # our model's basename (e.g. "qwen3.6" matches
            # "qwen3.6:35b-a3b-q4_K_M") OR vice versa.
            if (try_base.startswith(basename.split("-", 1)[0])
                    and len(try_base) <= len(basename) + 8):
                state_entry = self.prefill_table.get(try_id, {}).get(state) or {}
                bucket_entry = state_entry.get(bucket)
                if isinstance(bucket_entry, dict):
                    rate = bucket_entry.get("median_rate_tok_per_s")
                    if isinstance(rate, (int, float)) and rate > 0:
                        return float(rate)
        return FALLBACK_PREFILL_RATE_TOK_PER_S

    # ------------------------------------------------------------------ #
    # Stream lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Anchor the clock to "now" (request bytes on the wire)."""
        self._start_monotonic = time.monotonic()
        self._started = True
        self._first_token_monotonic = None
        self._token_count = 0
        self._token_times = []
        self._last_token_monotonic = None

    def on_first_token(self) -> None:
        if self._first_token_monotonic is None:
            self._first_token_monotonic = time.monotonic()

    def on_token(self) -> None:
        """Record a chunk arrival.  Caller is responsible for
        calling ``on_first_token()`` before the first
        ``on_token()`` if it wants pre-TTFT stats, but
        ``on_token()`` will also self-heal by stamping the
        first-token time if it has not been set yet.
        """
        if not self._started:
            self.start()
        now = time.monotonic()
        if self._first_token_monotonic is None:
            self._first_token_monotonic = now
        self._last_token_monotonic = now
        self._token_count += 1
        self._token_times.append(now)
        # Trim the rolling window to keep memory bounded.
        # We only need the last ``decode_window_s`` of timestamps.
        cutoff = now - self.decode_window_s
        while self._token_times and self._token_times[0] < cutoff:
            self._token_times.pop(0)

    # ------------------------------------------------------------------ #
    # Per-chunk check
    # ------------------------------------------------------------------ #

    def check(self) -> PathologyVerdict:
        """Ask "should I abort now?".

        Returns a verdict.  The runtime should call this
        on every chunk and at every checkpoint interval
        (e.g. every 1-2 seconds of wall time) so the abort
        fires as soon as the budget is exceeded.

        If the monitor has not been started yet (or has
        been marked stopped by the transport's lifecycle
        helpers), ``check()`` returns a benign verdict
        with ``abort=False`` and ``reason=None``.  This
        is so the runtime's background timer can keep
        polling without firing on a stale monitor.
        """
        if not self._started:
            return PathologyVerdict(
                abort=False, reason=None,
                budget_s=self._ttft_budget_s, elapsed_s=0.0,
                observed_rate=None,
            )
        now = time.monotonic()
        elapsed = now - self._start_monotonic
        if self._first_token_monotonic is None:
            # Still waiting for first token.  Check TTFT budget.
            if elapsed > self._ttft_budget_s:
                return PathologyVerdict(
                    abort=True, reason="ttft",
                    budget_s=self._ttft_budget_s, elapsed_s=elapsed,
                    observed_rate=None,
                )
            return PathologyVerdict(
                abort=False, reason=None,
                budget_s=self._ttft_budget_s, elapsed_s=elapsed,
                observed_rate=None,
            )
        # Post-TTFT: check sustained decode rate.
        observed_rate = self._current_decode_rate()
        # Need at least 2 tokens to call a rate; we use the
        # rolling window so a brief stall after the first
        # token does not immediately trip the monitor.
        if (observed_rate is not None
                and len(self._token_times) >= 2
                and observed_rate < self.min_decode_rate):
            return PathologyVerdict(
                abort=True, reason="slow_decode",
                budget_s=self._ttft_budget_s, elapsed_s=elapsed,
                observed_rate=observed_rate,
            )
        return PathologyVerdict(
            abort=False, reason=None,
            budget_s=self._ttft_budget_s, elapsed_s=elapsed,
            observed_rate=observed_rate,
        )

    def _current_decode_rate(self) -> float | None:
        """Tokens/sec over the rolling decode window.

        Returns None if there is only one token in the
        window (rate undefined).
        """
        if len(self._token_times) < 2:
            return None
        # The window is bounded by ``decode_window_s``; we
        # already trim on each ``on_token()`` call.  Compute
        # rate as the count divided by the time span.
        span = self._token_times[-1] - self._token_times[0]
        if span <= 0:
            return None
        # tokens within the window = len(times) - 1 (the
        # gaps between consecutive tokens), but a simpler
        # and equivalent measure is len(times) / span for
        # the "token arrival frequency" interpretation.
        return len(self._token_times) / span

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def summary(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "is_resident": self.is_resident,
            "prefill_rate_tok_per_s": self._prefill_rate,
            "ttft_budget_s": self._ttft_budget_s,
            "tokens_seen": self._token_count,
            "first_token_at_s": (
                (self._first_token_monotonic - self._start_monotonic)
                if self._first_token_monotonic is not None else None
            ),
            "current_decode_rate_tok_per_s": self._current_decode_rate(),
        }


# ---------------------------------------------------------------------------
#  Rate-stall monitor (sibling of PathologyMonitor)
# ---------------------------------------------------------------------------
# Data-driven gate derived from the 14-stream profiling run
# (lcb_clean75_coe_20260827_152115, 2026-08-27).  Findings:
#
#   * PASS streams never stall mid-attempt: 0 hard-zero 60s windows
#     and no window below 49% of the stream's own median active rate.
#     Long-thinking successes (3770, 3715, 3033) sustained 120-223 c/s
#     for their entire stream.
#   * Every terminal runaway stalls: abc355_e (169 -> 32 -> 0 x4),
#     abc359_e (126 -> 65 -> 0 x5), abc363_f (74/68 -> 0, fake recovery
#     115/161, -> 0 x6), arc188_b (~1900 bucket-chars x13 -> 0 x33).
#     arc186_c is a pure TTFT hang (already covered by the TTFT guard).
#
#   => A mid-attempt 60s window with ZERO chars is unambiguous
#      (transport-level; client-level zeros were ladder transitions).
#   => A partial stall (< stall_partial_ratio of the stream's own
#      median active rate) sustained for stall_partial_k consecutive
#      windows is the early-warning signature of the same failure.
#
# The monitor is self-calibrating per attempt: the reference rate is
# the median of ACTIVE windows so far, so no persistent per-model
# baseline is needed and thermal/state drift cannot stale it out.


@dataclass
class StallVerdict:
    """Result of a single ``StallMonitor.check()`` call."""
    abort: bool
    reason: str | None          # "rate_stall" | "rate_collapse" | None
    active_median_cps: float    # median chars/s of active windows so far
    elapsed_s: float            # seconds since stream start


class StallMonitor:
    """Mid-attempt rate-stall detector.

    Tracks characters per 60s window (rolling buckets) within ONE
    transport attempt.  Arms after ``stall_arm_windows`` fully-active
    windows following the first token, then:

      * trips ``rate_stall``    on any window with zero chars;
      * trips ``rate_collapse`` on ``stall_partial_k`` consecutive
        windows below ``stall_partial_ratio`` x (active median rate).

    A window >= ``stall_reset_ratio`` x active median clears the
    partial streak (abc363_f's fake recovery taught us a single
    sub-median window must not count as recovery).
    """

    def __init__(
        self,
        *,
        window_s: float = 60.0,
        arm_windows: int = 1,
        arm_min_chars: float = 300.0,
        zero_k: int = 2,
        partial_ratio: float = 0.30,
        partial_k: int = 4,
        reset_ratio: float = 0.50,
    ) -> None:
        self.window_s = float(window_s)
        self.arm_windows = int(arm_windows)
        self.arm_min_chars = float(arm_min_chars)
        self.zero_k = int(zero_k)
        self.partial_ratio = float(partial_ratio)
        self.partial_k = int(partial_k)
        self.reset_ratio = float(reset_ratio)
        # Bucketed char counts.  Buckets are window_s wide.
        self._buckets: list[int] = []
        self._bucket0_monotonic: float | None = None
        self._last_activity_monotonic: float | None = None
        self._armed = False
        self._first_token_seen = False
        self._stopped = False

    # -- lifecycle -------------------------------------------------------

    def on_first_token(self) -> None:
        self._first_token_seen = True

    def mark_stopped(self) -> None:
        self._stopped = True

    # -- per-chunk feed ----------------------------------------------------

    def on_chunk(self, n_chars: int) -> None:
        """Record ``n_chars`` of decoded output at the current time."""
        if self._stopped or n_chars <= 0:
            return
        # Self-heal first-token: the runtime feeds on_chunk() per chunk
        # but has no separate first-token hook for this monitor (the
        # classic monitor's on_token() self-heals the same way).
        # Without this the monitor never arms and the gate is inert.
        if not self._first_token_seen:
            self._first_token_seen = True
        now = time.monotonic()
        if self._bucket0_monotonic is None:
            self._bucket0_monotonic = now
        self._last_activity_monotonic = now
        idx = int((now - self._bucket0_monotonic) // self.window_s)
        while len(self._buckets) <= idx:
            self._buckets.append(0)
        self._buckets[idx] += n_chars

    # -- periodic check ----------------------------------------------------

    def check(self) -> StallVerdict:
        """Evaluate the stall rules.  Call periodically (the runtime's
        2s starvation timer already runs at the right cadence; this
        monitor piggybacks on it).  Cheap: O(windows) per call with
        windows bounded by stream length / window_s."""
        now = time.monotonic()
        elapsed = (
            (now - self._bucket0_monotonic)
            if self._bucket0_monotonic is not None else 0.0
        )
        benign = StallVerdict(
            abort=False, reason=None,
            active_median_cps=self._active_median(), elapsed_s=elapsed,
        )
        if self._stopped or not self._first_token_seen or not self._buckets:
            return benign
        cur_idx = int((now - self._bucket0_monotonic) // self.window_s)
        complete = self._buckets[:cur_idx]
        # Zero-stall: time-since-last-chunk >= zero_k * window_s means
        # ``zero_k`` consecutive 60s windows with zero output.  K is
        # configurable: K=1 trips fastest (kills 3033/3715-style
        # recoveries); K=2 spares the 80s recovery in the profiling
        # data while still catching every runaway at <=260s dead air.
        if (self._last_activity_monotonic is not None
                and (now - self._last_activity_monotonic) >= self.zero_k * self.window_s):
            return StallVerdict(
                abort=True, reason="rate_stall",
                active_median_cps=self._active_median(), elapsed_s=elapsed,
            )
        complete_cps = [c / self.window_s for c in complete]
        active_cps = [r for r in complete_cps if r > 0]
        # Arm: first token seen + arm_windows complete active windows
        # with enough chars (filters prefill tail / TTFT-adjacent noise
        # and guarantees the reference median is meaningful).  Short
        # streams that finish inside the first window never arm.
        if not self._armed:
            if len(active_cps) >= self.arm_windows:
                self._armed = True
            else:
                return benign
        med = self._active_median()
        if med <= 0:
            return benign
        # Partial-stall streak over COMPLETE windows: consecutive
        # trailing windows below partial_ratio x median.
        partial_streak = 0
        for r in reversed(complete_cps):
            if r < self.partial_ratio * med:
                partial_streak += 1
            else:
                break
        if partial_streak >= self.partial_k:
            return StallVerdict(
                abort=True, reason="rate_collapse",
                active_median_cps=med, elapsed_s=elapsed,
            )
        return benign

    def _active_median(self) -> float:
        """Median chars/s over all COMPLETE active windows so far.

        Recomputed from the raw buckets each call -- stateless, so
        polling cannot double-count.  Zero windows (stalls) are
        excluded from the reference so a terminal stall cannot drag
        the median down and un-trip the gate."""
        if not self._buckets:
            return 0.0
        now = time.monotonic()
        if self._bucket0_monotonic is None:
            return 0.0
        cur_idx = int((now - self._bucket0_monotonic) // self.window_s)
        complete = self._buckets[:cur_idx]
        rates = [c / self.window_s for c in complete if c > 0]
        if not rates:
            return 0.0
        return statistics.median(rates)

    def summary(self) -> dict[str, Any]:
        return {
            "armed": self._armed,
            "active_median_cps": self._active_median(),
            "buckets": len(self._buckets),
            "chars": sum(self._buckets),
        }
