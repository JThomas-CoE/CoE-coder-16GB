"""Tier-aware Ollama residency manager for the clean slice.

The manager observes ``/api/ps`` and uses public Ollama lifecycle
requests. It never starts, stops, or otherwise controls the Ollama
service itself.

Compact tier policy (dedicated 16 GB card):

* Total 16 GB; OS / driver reserve 0.2 GB; usable budget 15.8 GB.
* Exactly **one** local Ollama resident at a time.  When a second
  model is detected, the manager evicts it before proceeding.
* The KV cache is part of the resident footprint.  The manager reads
  ``/api/ps`` after every load and refuses to declare a model
  resident when its reported ``context_length`` exceeds the bounded
  cap or when its ``size_vram`` exceeds the usable budget.  When
  Ollama loaded the model with its Modelfile default 262,144
  context window, the manager forces a reload with the bounded
  context until the budget is honoured.
* The router model lives on the FLM NPU when reachable.  When the
  FLM service is down, the same router model is loaded on Ollama as
  a fallback and counts toward the one-resident policy.

The manager exposes a ``probe()`` helper that returns the live
``/api/ps`` state along with the **estimated KV cache footprint**
per resident (total VRAM minus the declared model weight) and the
**per-token KV headroom** implied by the resident's context window.
This makes the KV cache observable in human-readable terms.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .transport import BOUNDED_NUM_CTX


log = logging.getLogger(__name__)

# Deadline for one ``prepare()`` (evict + load + re-probe, up to 3
# attempts).  This is the black box's own clock on its slowest
# internal operation.  Without it, a hung ollama load blocks the
# request until the server's 210s local SSE idle watchdog fires --
# which (cloud tiers exempt) is a CLIENT-facing timeout, not an
# internal one: the runtime never learns the load failed, no
# pathology is counted, and the ladder cannot reroll/escalate.
# With this deadline the runtime receives ``PrepareTimeout`` and
# classifies it exactly like any other pathology.  Healthy loads
# are <=25s on this card; 120s covers the worst legitimate cycle
# (evict + cold load + shrink-reload) with ~2x margin; a genuinely
# wedged load cannot take longer than that without ollama itself
# being dead.
DEFAULT_PREPARE_DEADLINE_S = 120.0


class PrepareTimeout(RuntimeError):
    """Raised when ``prepare()`` exceeds ``prepare_deadline_s``.

    Classified as a pathology by the runtime (reroll with fresh
    seed / escalate after budget exhaustion), like TTFT and
    slow_decode timeouts."""


@dataclass(frozen=True)
class CleanResidentModel:
    name: str
    context_length: int
    size_vram_gb: float


@dataclass(frozen=True)
class CleanVramState:
    tier: str
    total_vram_gb: float
    usable_gb: float
    resident_gb: float
    available_gb: float
    residents: tuple[CleanResidentModel, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "total_vram_gb": round(self.total_vram_gb, 2),
            "usable_gb": round(self.usable_gb, 2),
            "resident_gb": round(self.resident_gb, 2),
            "available_gb": round(self.available_gb, 2),
            "residents": [
                {"name": item.name, "context_length": item.context_length, "size_vram_gb": round(item.size_vram_gb, 2)}
                for item in self.residents
            ],
        }


@dataclass(frozen=True)
class CleanKvProbe:
    """Per-resident breakdown of weight vs KV cache footprint.

    Ollama's ``/api/ps`` reports the *total* resident VRAM
    (``size_vram``) which includes the model weight and the
    allocated KV cache.  The KV cache size is therefore estimated
    as ``size_vram - declared_weight``.  When Ollama allocated a
    large Modelfile default context (262,144) the KV cache
    dominates the difference; the manager uses this signal to
    decide whether to force a reload.
    """

    name: str
    context_length: int
    total_vram_gb: float
    declared_weight_gb: float
    estimated_kv_gb: float
    estimated_kv_per_token_bytes: float
    bounded_context_length: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "context_length": self.context_length,
            "total_vram_gb": round(self.total_vram_gb, 3),
            "declared_weight_gb": round(self.declared_weight_gb, 3),
            "estimated_kv_gb": round(self.estimated_kv_gb, 3),
            "estimated_kv_per_token_bytes": round(self.estimated_kv_per_token_bytes, 1),
            "bounded_context_length": self.bounded_context_length,
        }


def _extract_param_billions(model_id: str) -> float:
    match = re.search(r"(\d+)\s*b", model_id.lower())
    return float(match.group(1)) if match else 0.0


def _normalize_model_key(model_id: str) -> str:
    """Strip a TRAILING ``:latest`` so a tag/registry mismatch cannot miss.

    The registry reports most models as ``<name>:latest`` while
    ``model_refs`` carries the bare ``<name>`` (and vice versa).  Both
    name the SAME model, so a lookup must treat them as equal.

    Deliberately NARROW: only a trailing ``:latest`` is removed.  A real
    tag difference (``:q4_K_M`` vs ``:q8_0``) selects a different blob and
    must NOT be conflated.  This is also why the normalisation is not
    ``split(":", 1)[0]`` -- that would erase a distinguishing quant tag.
    """
    if model_id.endswith(":latest"):
        return model_id[: -len(":latest")]
    return model_id


def _footprint_lookup(model_id: str, footprints: dict[str, float]) -> float | None:
    """Exact hit, else a trailing-``:latest``-tolerant hit, else ``None``.

    Tolerant in BOTH directions, because the mismatch is observed both ways:
    the live ``model_refs`` ids are BARE while the configured table keys
    carry ``:latest``; the registry (``/api/ps``, ``/api/tags``) reports the
    ``:latest`` form while some callers pass the bare id.  Four cases:

        table ``X``          query ``X``          -> exact (1st test)
        table ``X:latest``   query ``X:latest``   -> exact (1st test)
        table ``X``          query ``X:latest``   -> 2nd test
        table ``X:latest``   query ``X``          -> 3rd test

    At most three dict probes, no per-call dict construction, and NO
    normalisation of the stored table (so the config keeps whatever spelling
    the user wrote).
    """
    if model_id in footprints:
        return float(footprints[model_id])
    key = _normalize_model_key(model_id)
    if key != model_id and key in footprints:
        return float(footprints[key])
    with_latest = key + ":latest"
    if with_latest in footprints:
        return float(footprints[with_latest])
    return None


def _declared_weight_gb(model_id: str, footprints: dict[str, float]) -> float:
    declared = _footprint_lookup(model_id, footprints)
    if declared is not None:
        return declared
    billions = _extract_param_billions(model_id)
    if billions > 0:
        return round(billions * 0.8, 3)
    return 0.0


def _footprint_lookup(model_id: str, footprints: dict[str, float]) -> float | None:
    """Exact hit, else a hit tolerant of a TRAILING ``:latest`` on either side.

    Returns ``None`` when the model is genuinely undeclared, so the caller can
    apply its own fallback.  Never raises.
    """
    probe = _normalize_model_key(model_id)
    for key, value in footprints.items():
        if _normalize_model_key(key) == probe:
            return float(value)
    return None


def verify_footprints(
    registered: dict[str, float],
    footprints: dict[str, float],
    tolerance: float = 0.05,
) -> list[dict[str, Any]]:
    """Compare DECLARED footprints against BACKEND-REPORTED sizes.

    Pure.  ``registered`` is ``{model_id: size_gb}`` as the backend reports
    it; ``footprints`` is ``vram_tiers.model_footprints_gb``.  Returns one
    dict per finding, each with a ``kind``:

      ``missing``    declared nowhere -- falls back to the regex guess
      ``drift``      declared value differs from the reported size by more
                     than ``tolerance`` (relative)
      ``orphan``     declared in the table but on no registered model
      ``unverifiable``  registered but the backend reported no usable size

    INTENT: this is a WARN-only diagnostic.  It never raises and never
    changes admission -- a stale declaration already has a defined non-fatal
    policy (the slot fallback), and failing a boot on a small drift would be
    a false-error.  The caller decides how to surface the list.

    Tolerance (default 5%) is chosen to clear the gemma4 projector layer
    (~2.4% of a manifest total) while still catching the 5-30% class of
    stale-declaration errors this exists to find.
    """
    findings: list[dict[str, Any]] = []
    # --- declared vs reported -------------------------------------------
    for model_id, size_gb in sorted(registered.items()):
        declared = _footprint_lookup(model_id, footprints)
        if declared is None:
            findings.append({
                "kind": "missing", "model": model_id,
                "declared": None, "reported": round(float(size_gb), 3),
            })
            continue
        if not size_gb or size_gb <= 0:
            findings.append({
                "kind": "unverifiable", "model": model_id,
                "declared": declared, "reported": None,
            })
            continue
        delta = abs(declared - float(size_gb)) / float(size_gb)
        if delta > tolerance:
            findings.append({
                "kind": "drift", "model": model_id,
                "declared": declared, "reported": round(float(size_gb), 3),
                "delta_pct": round(delta * 100.0, 1),
            })
    # --- declared but not registered anywhere ---------------------------
    for key in sorted(footprints.keys()):
        if not any(
            _normalize_model_key(key) == _normalize_model_key(r)
            for r in registered
        ):
            findings.append({
                "kind": "orphan", "model": key,
                "declared": float(footprints[key]), "reported": None,
            })
    return findings


class CleanVramManager:
    """Enforce the clean tier's single-resident, bounded-VRAM policy.

    All Ollama lifecycle calls (evict, load) flow through the
    ``transport`` reference, which is the universal chokepoint
    that clamps ``num_ctx`` to the bounded budget.  The manager
    no longer opens its own ``httpx`` connection to ``/api/chat``;
    that was a leak that could let a runaway 256k context load
    through on a real 16 GB card.  Set ``transport`` to an
    instance of ``CleanOllamaTransport`` before calling
    ``prepare`` or ``evict``.
    """

    def __init__(self, cfg: dict[str, Any], transport: "Any | None" = None) -> None:
        self.cfg = cfg
        self.transport = transport
        backend = cfg["backends"]["ollama"]
        self.base_url = str(backend["url"]).rstrip("/").removesuffix("/v1")
        self.timeout = float(backend.get("timeout", 900))
        tiers = cfg["vram_tiers"]
        self.tier_name = str(tiers["active"])
        self.tier = dict(tiers["defined"][self.tier_name])
        # Black-box deadline for one prepare() cycle (evict+load+probe).
        # Configurable via vram_tiers.defined.<tier>.prepare_deadline_s.
        self.prepare_deadline_s = float(
            self.tier.get("prepare_deadline_s", DEFAULT_PREPARE_DEADLINE_S)
        )
        self._lock = asyncio.Lock()
        self._footprints = dict(tiers.get("model_footprints_gb", {}))
        self._model_defaults = self._build_model_defaults()
        self._max_residents = int(self.tier.get("max_resident_local_models", 1))
        self._bounded_ctx = int(self.tier["max_num_ctx"])

    def attach_transport(self, transport: "Any") -> None:
        """Bind the universal chokepoint transport after construction.

        The manager refuses to talk to Ollama until a transport
        is attached, so a misconfigured wiring cannot bypass the
        bounded-context chokepoint.
        """
        self.transport = transport

    def _build_model_defaults(self) -> dict[str, dict[str, Any]]:
        defaults: dict[str, dict[str, Any]] = {}
        for entry in self.cfg.get("models", {}).values():
            if entry.get("model_id"):
                defaults[str(entry["model_id"])] = dict(entry)
        for entry in self.cfg["clean_resolution"].get("model_refs", {}).values():
            if entry.get("model_id"):
                defaults.setdefault(str(entry["model_id"]), {}).update(entry)
        return defaults

    @property
    def total_vram_gb(self) -> float:
        return float(self.tier["total_vram_gb"])

    @property
    def usable_gb(self) -> float:
        return float(self.tier["total_vram_gb"]) - float(self.tier.get("os_reserved_gb", 0))

    @property
    def bounded_context(self) -> int:
        return self._bounded_ctx

    def model_defaults(self, model_id: str) -> dict[str, Any]:
        return dict(self._model_defaults.get(model_id, {}))

    async def probe(self) -> list[CleanKvProbe]:
        """Return a per-resident breakdown of weight vs KV cache."""
        async with self._lock:
            state = await self._state_unlocked()
        probes: list[CleanKvProbe] = []
        for resident in state.residents:
            weight = _declared_weight_gb(resident.name, self._footprints)
            kv = max(0.0, resident.size_vram_gb - weight)
            per_token_bytes = (kv * 1e9) / max(1, resident.context_length)
            probes.append(CleanKvProbe(
                name=resident.name,
                context_length=resident.context_length,
                total_vram_gb=resident.size_vram_gb,
                declared_weight_gb=weight,
                estimated_kv_gb=kv,
                estimated_kv_per_token_bytes=per_token_bytes,
                bounded_context_length=self._bounded_ctx,
            ))
        return probes

    async def state(self) -> CleanVramState:
        async with self._lock:
            return await self._state_unlocked()

    async def _state_unlocked(self) -> CleanVramState:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/api/ps")
            response.raise_for_status()
        payload = response.json()
        residents: list[CleanResidentModel] = []
        for item in payload.get("models", []):
            residents.append(CleanResidentModel(
                name=str(item.get("name", "")),
                context_length=int(item.get("context_length", 0) or 0),
                size_vram_gb=float(item.get("size_vram", 0) or 0) / 1e9,
            ))
        used = sum(item.size_vram_gb for item in residents)
        return CleanVramState(
            self.tier_name, self.total_vram_gb, self.usable_gb, used,
            max(0.0, self.usable_gb - used), tuple(residents),
        )

    def _same_model(self, first: str, second: str) -> bool:
        return first == second or first.split(":", 1)[0] == second.split(":", 1)[0]

    def _bounded_context_for(self, model_id: str, requested: int | None) -> int:
        declared = self._model_defaults.get(model_id, {}).get("num_ctx")
        if requested is None:
            requested = declared or self.tier["default_num_ctx"]
        return min(int(requested), self._bounded_ctx)

    def _keep_alive(self, model_id: str) -> int:
        # Always return -1 so the loaded model stays resident
        # in Ollama's VRAM indefinitely.  Once a model is
        # loaded, it stays loaded until a different model is
        # called for, at which point ``prepare()`` evicts the
        # resident (via ``_evict`` which sends ``keep_alive: 0``)
        # and loads the new one.  The harness config's
        # ``keep_alive`` value is intentionally overridden here:
        # a finite timeout would cause Ollama to unload the
        # model after the configured idle period, forcing a
        # 15-25s cold-load on the next request.  See the
        # streaming transport's ``stream()`` for the matching
        # default on the inference path.
        return -1

    def _footprint(self, model_id: str) -> float:
        # Tolerant lookup (2026-09-21): the registry reports most models as
        # ``<name>:latest`` while ``model_refs`` carries the bare ``<name>``,
        # so a strict dict hit MISSED for every live model and fell through
        # to the regex estimate below.  ``_footprint_lookup`` treats a
        # trailing ``:latest`` difference as the same model; a real tag
        # difference (a different quant) still misses, by design.
        declared = _footprint_lookup(model_id, self._footprints)
        if declared is not None:
            return declared
        billions = _extract_param_billions(model_id)
        if billions > 0:
            return round(billions * 0.8, 2)
        raise RuntimeError(f"No verified VRAM footprint configured for clean model '{model_id}'")

    def _require_transport(self) -> "Any":
        if self.transport is None:
            raise RuntimeError(
                "CleanVramManager has no transport bound; call attach_transport(transport) before any Ollama lifecycle call"
            )
        return self.transport

    async def _evict(self, model_id: str) -> None:
        # Ledger-first eviction (2026-09-07, Option A): query /api/ps
        # BEFORE touching the model.  Ollama has no unload-only API --
        # the only way to honor keep_alive: 0 is a generate request,
        # which LOADS the model first if it is not resident.  On a
        # boot with a stale ledger entry (e.g. the user killed the
        # prior llama-server runner, or a request raced the daemon's
        # accounting), the old unconditional evict re-loaded a 12GB
        # specialist just to unload it -- pure GPU compute (prefill of
        # the 1-token probe) with NO Copy engine activity, wasting
        # boot time and adding a wedge surface.  If the model is not
        # actually resident, there is nothing to unload.
        try:
            current = await self._state_unlocked()
        except Exception as exc:
            # Ledger unreadable: fall through to the legacy unconditional
            # evict rather than skipping an unload that might be needed.
            log.debug("Ledger probe before evict of %s failed: %s", model_id, exc)
            current = None
        if current is not None and not any(
            self._same_model(resident.name, model_id)
            for resident in current.residents
        ):
            log.info(
                "Skipping evict of '%s': not resident per ollama ledger "
                "(no keep_alive:0 probe sent; avoids load-first unload)",
                model_id,
            )
            return
        # Route the evict through the universal chokepoint so the
        # bounded-context guarantee applies even for lifecycle
        # calls.  ``non_streaming_call`` always enforces the
        # ``BOUNDED_NUM_CTX`` ceiling.
        #
        # num_predict=1 (2026-09-12).  Ollama has no unload-only API,
        # so this generate IS the unload.  Uncapped, the one-space
        # user message is a *degenerate generation*: the model is free
        # to emit special tokens for up to num_ctx steps.  Measured
        # with gemma4-web-hc, the uncapped call produced 10 bytes of
        # ``<unused25>`` instead of stopping, and one instance ran
        # until the 60s timeout here.  Capping makes the unload
        # deterministic: ~0.1s, done_reason=length, eval=1, on both
        # model families.
        #
        # Honest scope of the evidence: in a controlled A/B with a
        # verified-clean start before each trial, the uncapped evict
        # was ALSO 3/3 clean -- so this cap is harm-removal and
        # determinism, NOT the sole cause of the ``<unused6226>``
        # garbage.  That garbage correlates with a long/degenerate
        # lifecycle generation having run, which this cap prevents.
        transport = self._require_transport()
        try:
            await transport.non_streaming_call(
                model_id=model_id,
                messages=[{"role": "user", "content": " "}],
                num_ctx=BOUNDED_NUM_CTX,
                num_predict=1,
                keep_alive=0,
                timeout=60.0,
            )
        except Exception as exc:
            log.debug("Ollama unload for %s raised: %s", model_id, exc)

    async def _load(self, model_id: str, num_ctx: int, keep_alive: int) -> None:
        # Refuse to load a model with a context window larger than
        # the bounded chokepoint.  This catches a misconfigured
        # caller that tries to pass 256k in here — the only legal
        # values are <= BOUNDED_NUM_CTX.
        bounded = min(int(num_ctx), BOUNDED_NUM_CTX)
        if bounded <= 0:
            raise RuntimeError(f"_load refused for {model_id}: num_ctx must be positive, got {num_ctx}")
        # num_predict=1 (2026-09-12): this call exists only to warm the
        # model into VRAM (keep_alive=-1), never to generate.  Without
        # a cap the one-space user message is an uncapped degenerate
        # generation, and ``_enforce_bounded_options`` will NOT clamp
        # it -- a missing num_predict means "no cap sent".  The prompt
        # package that follows carries the real num_predict from
        # config.yaml, so capping here costs nothing and makes
        # residency deterministic on both model families.
        transport = self._require_transport()
        await transport.non_streaming_call(
            model_id=model_id,
            messages=[{"role": "user", "content": " "}],
            num_ctx=bounded,
            num_predict=1,
            keep_alive=keep_alive,
            timeout=self.timeout,
        )

    async def enforce_single_resident(self) -> CleanVramState:
        """Evict every resident except the first. Idempotent."""
        async with self._lock:
            state = await self._state_unlocked()
            for resident in state.residents[1:]:
                await self._evict(resident.name)
            return await self._state_unlocked()

    async def prepare(self, model_id: str, requested_num_ctx: int | None = None) -> CleanVramState:
        """Make ``model_id`` the only local resident under the budget.

        Steps:

        1. Pre-flight admission on the declared footprint.
        2. Evict every resident that is not the target.
        3. Load the target with the bounded context.  After each
           load, re-probe ``/api/ps``; if the resident's reported
           ``context_length`` exceeds the bounded cap, force a
           reload until Ollama honours the cap.
        4. Verify the post-load state honours the single-resident
           and budget policies.  On violation, evict and raise.

        The whole cycle runs under ``prepare_deadline_s``: a hung
        ollama load raises ``PrepareTimeout`` (a pathology the
        runtime's ladder can reroll/escalate on) instead of parking
        the request until the server's 210s local SSE idle watchdog
        (cloud tiers exempt) fires with no internal record.
        """
        t0 = time.monotonic()
        try:
            async with asyncio.timeout(self.prepare_deadline_s):
                return await self._prepare_locked(model_id, requested_num_ctx)
        except TimeoutError:
            raise PrepareTimeout(
                f"prepare('{model_id}') exceeded {self.prepare_deadline_s:.0f}s deadline "
                f"(evict+load+probe); ollama load is wedged or pathologically slow"
            )
        except asyncio.CancelledError:
            elapsed = time.monotonic() - t0
            if elapsed >= self.prepare_deadline_s:
                raise PrepareTimeout(
                    f"prepare('{model_id}') exceeded {self.prepare_deadline_s:.0f}s deadline "
                    f"(elapsed {elapsed:.0f}s); ollama load is wedged"
                )
            raise

    async def _prepare_locked(self, model_id: str, requested_num_ctx: int | None = None) -> CleanVramState:
        context = self._bounded_context_for(model_id, requested_num_ctx)
        keep_alive = self._keep_alive(model_id)
        estimate = self._footprint(model_id)
        if estimate > self.usable_gb:
            # NOTE (2026-09-21): the value compared here is the DECLARED
            # WEIGHT only (model_footprints_gb holds the weight term; the KV
            # cache is accounted separately as size_vram - weight).  This
            # compares a weight against a weights+KV budget, which is a unit
            # mismatch that is SAFE BY DIRECTION: KV >= 0 implies
            # size_vram >= weight, so this rejects a strict SUBSET of what the
            # post-load check rejects and can never admit a model that will
            # not fit.  It is retained as a cheap guard that avoids paying for
            # an evict+load cycle on a model that cannot fit.  The message
            # therefore says "weight", not "footprint".
            raise RuntimeError(
                f"Model '{model_id}' declared weight {estimate:.2f} GB "
                f"exceeds {self.tier_name} usable budget {self.usable_gb:.2f} GB"
            )
        async with self._lock:
            before = await self._state_unlocked()
            for resident in before.residents:
                if not self._same_model(resident.name, model_id):
                    await self._evict(resident.name)
            for attempt in range(3):
                current = await self._state_unlocked()
                target = next((item for item in current.residents if self._same_model(item.name, model_id)), None)
                if target is None:
                    await self._load(model_id, context, keep_alive)
                elif target.context_length > context or target.size_vram_gb > self.usable_gb:
                    log.info(
                        "Reloading '%s' to shrink KV cache: ctx=%d > %d or size=%.2fGB > %.2fGB",
                        model_id, target.context_length, context, target.size_vram_gb, self.usable_gb,
                    )
                    await self._evict(model_id)
                    await self._load(model_id, context, keep_alive)
                current = await self._state_unlocked()
                target = next((item for item in current.residents if self._same_model(item.name, model_id)), None)
                if target is not None and target.context_length <= context and target.size_vram_gb <= self.usable_gb:
                    break
            after = await self._state_unlocked()
            if after.resident_gb > after.usable_gb:
                for resident in after.residents:
                    await self._evict(resident.name)
                raise RuntimeError(
                    f"Clean VRAM tier budget exceeded after load: "
                    f"{after.resident_gb:.2f} GB > {after.usable_gb:.2f} GB usable"
                )
            if len(after.residents) > self._max_residents:
                for resident in after.residents[1:]:
                    await self._evict(resident.name)
                after = await self._state_unlocked()
            target = next((item for item in after.residents if self._same_model(item.name, model_id)), None)
            if target is None:
                raise RuntimeError(f"Ollama did not report '{model_id}' as resident after prepare")
            if target.context_length > context:
                raise RuntimeError(
                    f"Ollama resident context {target.context_length} exceeds clean cap {context} for '{model_id}'"
                )
            if target.size_vram_gb > self.usable_gb:
                raise RuntimeError(
                    f"Ollama resident size {target.size_vram_gb:.2f} GB exceeds clean budget {self.usable_gb:.2f} GB for '{model_id}'"
                )
            log.info(
                "Loaded '%s' at ctx=%d (kv=%.2fGB total=%.2fGB budget=%.2fGB)",
                model_id, target.context_length,
                max(0.0, target.size_vram_gb - _declared_weight_gb(model_id, self._footprints)),
                target.size_vram_gb, self.usable_gb,
            )
            return after

    async def evict(self, model_id: str) -> None:
        async with self._lock:
            await self._evict(model_id)

    async def evict_all(self, only_roster: bool = False) -> int:
        """Unload resident models.  Returns the count actually evicted.

        ``only_roster=False`` (default, and the historical behaviour) evicts
        EVERY resident model, whoever loaded it.  ``True`` evicts only models
        on the CoE roster -- ``cfg.models`` plus
        ``clean_resolution.model_refs``, the same set ``_build_model_defaults``
        collects -- so a model loaded by another tool is left resident.

        Driven by ``vram_tiers.evict_on_boot`` at the boot call site; the
        opt-out exists for a card with excess VRAM.  A caller choosing it must
        leave >= 16 GB free, because the admission control still measures
        against ``total_vram_gb`` (see the config comment and the README).
        """
        async with self._lock:
            current = await self._state_unlocked()
            count = 0
            for resident in current.residents:
                if only_roster and not self._on_roster(resident.name):
                    log.info(
                        "evict_all: leaving '%s' resident (not a CoE model; "
                        "evict_on_boot is false)",
                        resident.name,
                    )
                    continue
                await self._evict(resident.name)
                count += 1
            return count

    def _on_roster(self, model_id: str) -> bool:
        """True iff ``model_id`` is a model CoE-coder itself can spawn.

        Roster = the keys ``_build_model_defaults`` collected from
        ``cfg.models`` and ``cfg.clean_resolution.model_refs``.  Matching is
        by ``_same_model`` so a registry tag/`:latest` mismatch cannot make a
        CoE model look foreign (which would leave it resident and break the
        clean-boot guarantee).
        """
        for known in self._model_defaults:
            if self._same_model(known, model_id):
                return True
        return False
