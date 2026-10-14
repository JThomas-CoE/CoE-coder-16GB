"""Domain-first, dependency-free resolution for the clean slice.

Resolution order:
  1. The router emits a ``(raw_domain, confidence, language, task_type)`` tuple.
  2. ``resolve`` canonicalises the language first. The language picks the
     coding bucket: ``python`` for Python; ``web`` for HTML/CSS/JS/TS; and
     ``compiled_coding`` for C++/Rust/Go/Java/Scala/Kotlin.
  3. The canonical language is required to match a row. A bare ``coding``
     signal without a language descriptor falls through to the generalist
     because the clean slice has no single ``coding`` domain — the spec
     requires an explicit language.
  4. The row selects a task harness and ranked model roles. The model
     choice is not made by the harness.
"""
from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .contracts import CleanPackage, CleanRouterDecision, CleanTierContext, CleanUnit

log = logging.getLogger(__name__)


class MatrixConfigError(Exception):
    """Raised at matrix load when the config is invalid."""


# The canonical task_type set is platform-invariant.  The router
# model is instructed to emit one of the 5 router-emit values;
# `user_defined` is reserved for user override only.
CANONICAL_TASK_TYPES = frozenset({
    "new_code_generation",
    "bug_fix",
    "modify_artifact",
    "evaluate_artifact",
    "general_use",
    "user_defined",
})

# The router's emit set is the 5-value subset — never `user_defined`.
ROUTER_EMIT_TASK_TYPES = frozenset({
    "new_code_generation",
    "bug_fix",
    "modify_artifact",
    "evaluate_artifact",
    "general_use",
})

# The canonical task_scope set is platform-invariant.  All four
# values reach the matrix; only single_domain_single_task
# currently passes the matrix's scope filter.
CANONICAL_TASK_SCOPES = frozenset({
    "single_domain_single_task",
    "single_domain_multi_task",
    "multi_domain_single_task",
    "multi_domain_multi_task",
})

# Mapping from non-SDST task_scope values to the rejection template
# keys in `clean_resolution.rejection_messages:`.  Currently the
# mapping is identity (the template keys match the task_scope values
# verbatim), but kept as a dict so future scope rewrites can decouple
# the wire name from the key.
SCOPE_REJECTION: dict[str, str] = {
    "single_domain_multi_task": "single_domain_multi_task",
    "multi_domain_single_task": "multi_domain_single_task",
    "multi_domain_multi_task": "multi_domain_multi_task",
}

_UNSET = object()


def _domain_to_language(domain: str) -> str:
    """Map a `domain_<name>` key to a language tag for harness prefill.

    Examples:
        domain_python -> python
        domain_web -> web
        domain_compiled -> compiled
        domain_other -> other

    The language tag is substituted into the harness's
    `assistant_prefill` template at compose-time, replacing any
    ``{{language}}`` placeholders.  The `languages` field (a
    list of detected coding languages from the user's prompt)
    is a separate concept and is passed by the executor at a
    different point.
    """
    return domain.removeprefix("domain_")


def _deep_merge_with_clamp(
    base: dict[str, Any],
    overrides: dict[str, Any],
    ranges: dict[str, list[float]],
    _path: str = "",
) -> dict[str, Any]:
    """Deep-merge cell overrides into the harness's runtime block.

    `ranges` is keyed by dotted path (e.g.,
    "pathology.max_re_rolls_per_model").  Out-of-range values are
    clamped to the nearest boundary.  Clamped values are logged at
    info level.

    Keys in `overrides` not present in `ranges` are dropped with a
    logged warning (surface-validation).
    """
    result = copy.deepcopy(base)
    for key, override_value in overrides.items():
        full_path = f"{_path}.{key}" if _path else key
        if key not in result:
            log.warning(
                "matrix runtime_override dropped: parameter '%s' not in "
                "harness's exposed surface", full_path,
            )
            continue
        if isinstance(override_value, dict) and isinstance(result[key], dict):
            result[key] = _deep_merge_with_clamp(
                result[key], override_value, ranges, full_path,
            )
        else:
            if full_path in ranges:
                lo, hi = ranges[full_path]
                clamped = max(lo, min(hi, override_value))
                if clamped != override_value:
                    log.info(
                        "matrix runtime_override clamped: %s=%s -> %s "
                        "(range [%s, %s])",
                        full_path, override_value, clamped, lo, hi,
                    )
                result[key] = clamped
            else:
                result[key] = override_value
    return result


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge declarative harness configuration dictionaries."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result

# The 19-domain fallback is reserved for infrastructure failures
# (file missing, YAML parse error).  It is not a config fallback.
DOMAIN_FALLBACK = frozenset({
    "domain_python", "domain_web", "domain_compiled",
    "domain_physics", "domain_chemistry", "domain_math",
    "domain_biology", "domain_law", "domain_business",
    "domain_economics", "domain_humanities",
    "domain_engineering", "domain_computer_science",
    "domain_medical", "domain_psychology",
    "domain_vocational", "domain_history",
    "domain_puzzles", "domain_other",
})


LANGUAGE_BUCKETS: dict[str, str] = {
    "python": "python",
    "py": "python",
    "web": "web",
    "html": "web",
    "css": "web",
    "javascript": "web",
    "js": "web",
    "typescript": "web",
    "ts": "web",
    "compiled_coding": "compiled_coding",
    "cpp": "compiled_coding",
    "c++": "compiled_coding",
    "cxx": "compiled_coding",
    "rust": "compiled_coding",
    "go": "compiled_coding",
    "java": "compiled_coding",
    "scala": "compiled_coding",
    "kotlin": "compiled_coding",
}


_LANG_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bdef\s+\w+\s*\(", re.IGNORECASE), "python"),
    (re.compile(r"\bimport\s+\w+\b|\bfrom\s+\w+\s+import\b", re.IGNORECASE), "python"),
    (re.compile(r"\bclass\s+\w+\s*\(?\s*[A-Z]\w*\s*\)?\s*:", re.IGNORECASE), "python"),
    (re.compile(r"<\s*(html|head|body|div|script|template)\b", re.IGNORECASE), "web"),
    (re.compile(r"\b(html|css|javascript|typescript|react|vue|angular|node\.?js)\b", re.IGNORECASE), "web"),
    (re.compile(r"#include\s*<\w+/?\w*>", re.IGNORECASE), "compiled_coding"),
    (re.compile(r"\bstd::\w+", re.IGNORECASE), "compiled_coding"),
    (re.compile(r"\bfn\s+\w+\s*\(", re.IGNORECASE), "rust"),
    (re.compile(r"\bpackage\s+\w+|\bfunc\s+\w+", re.IGNORECASE), "go"),
    (re.compile(r"\bpublic\s+(static\s+)?(void|class|int|String)\b", re.IGNORECASE), "java"),
)


def detect_language(prompt: str, hint: str | None = None) -> str | None:
    """Return a canonical language bucket or ``None`` when undetermined."""
    if hint:
        key = hint.strip().lower()
        if key in LANGUAGE_BUCKETS:
            return LANGUAGE_BUCKETS[key]
    for pattern, language in _LANG_PATTERNS:
        if pattern.search(prompt):
            return LANGUAGE_BUCKETS[language]
    return None


@dataclass(frozen=True)
class CleanHarnessSpec:
    harness_id: str
    family: str = "general_use"
    tiers: tuple[str, ...] = ("T0",)   # minimal default; YAML explicit overrides
    # ================================================================
    # HARNESS CAPABILITY PROPERTIES.
    # ================================================================
    # These replace `harness_id == "modify_artifact"` string tests scattered
    # through runtime.py.  A NAME test hard-codes behaviour to one harness,
    # which is why splitting the modify route required inheriting the
    # change-order harness instead of declaring a second one.
    #
    # THERE ARE TWO DISTINCT LAYERS, and conflating them was a design error
    # (2026-09-16).  "Emits a change order" is TRUE OF DIFFERENT LAYERS for
    # different routes:
    #
    #                          local tier        shared T3 cloud layer
    #   PATH A                 denovo            change order
    #   PATH B                 (none)            change order
    #   modify_harness_legacy  change order      change order
    #
    # So ONE boolean cannot describe PATH A, whose local tier is denovo while
    # the cloud layer it escalates into returns a change order.  Hence two
    # flags.  Key point from the design: the T3 layer is SHARED -- PATH A's
    # escalation and the legacy ladder reach the SAME configured layer, so the
    # cloud-return-type flag is true for both harnesses.
    #
    #   local_change_order  -- THIS harness's LOCAL tier contract is a change
    #                          order (JSON replacements/insertions/removals)
    #                          rather than a complete artifact.  True only for
    #                          the legacy ladder harness.
    #   cloud_change_order  -- the SHARED T3 layer returns a change order that
    #                          local deterministic re-integration applies.
    #                          True for BOTH modify harnesses.
    #   needs_artifact_store -- reads the target artifact from the store.
    #   needs_t3_context     -- wants the artifact context in the T3 prompt.
    #
    # All four default to False, so an undeclared harness behaves exactly as it
    # did before these existed.
    local_change_order: bool = False
    cloud_change_order: bool = False
    needs_artifact_store: bool = False
    needs_t3_context: bool = False
    system_suffix: str = ""
    output_instruction: str = ""
    system_suffix_thinkon: str | None = None   # think-state-aware suffix (None = use system_suffix)
    output_instruction_thinkon: str | None = None  # think-state-aware output instruction
    closed_think: bool | None = None
    assistant_prefill: str | None = None
    sampling: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    escalation_ladder: dict[str, Any] = field(default_factory=dict)
    rounds: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedWork:
    """The matrix's resolve()-time output.

    The matrix sees only `(domain, task_type, task_scope)` from the
    router. The resolver combines those with the matrix's cell
    configuration to produce a `ResolvedWork` that is the input to
    the executor.

    Five resolution_kind discriminator values are supported:

    * ``"rejection"`` — task_scope is not SDST. The resolver returns
      a `rejection_message` and the harness is `None`. The server
      emits the rejection and ends the turn.
    * ``"override_invalid"`` — the user supplied an unsupported
      override. The resolver returns a `rejection_message` (re-used
      for alert payload) and the harness is `None`.
    * ``"user_defined_unavailable"`` — the user override requested
      a `user_defined` harness that the cell does not have. The
      resolver returns a `user_defined_message` and the harness is
      `None`. The turn is BLOCKED.
    * ``"specialist"`` or ``"generalist"`` — the normal flow. The
      harness is composed and the executor runs the model.
    """
    # --- Matrix-bound fields (router-emitted) ---
    domain: str
    task_type: str
    task_scope: str

    # --- Resolution outcome ---
    resolution_kind: str = "specialist"  # "rejection" | "override_invalid" | "user_defined_unavailable" | "specialist" | "generalist"
    fallback_reason: str | None = None
    matrix_cell: str | None = None

    # --- Model slots (None for terminal paths) ---
    primary_model: str | None = None
    secondary_model: str | None = None
    generalist_model: str | None = None
    primary_provider: str | None = None
    secondary_provider: str | None = None
    generalist_provider: str | None = None
    primary_options: dict[str, Any] = field(default_factory=dict)
    secondary_options: dict[str, Any] = field(default_factory=dict)
    generalist_options: dict[str, Any] = field(default_factory=dict)

    # Per-model think (model-card default). None = fall back to harness
    # closed_think. True = think-on (native reasoning), False = think-off.
    primary_think: bool | None = None
    secondary_think: bool | None = None
    generalist_think: bool | None = None

    # 2026-09-09 (user ruling): per-model raw-prefill tolerance
    # (model-card property, separate from the think policy). False = the
    # transport MUST NOT use the /api/generate raw response-prefix path.
    primary_raw_prefill: bool | None = None
    secondary_raw_prefill: bool | None = None
    generalist_raw_prefill: bool | None = None

    # --- Persona (None for terminal paths) ---
    persona: str | None = None

    # --- Harness (None for terminal paths) ---
    harness: CleanHarnessSpec | None = None

    # --- Terminal-path payloads ---
    rejection_message: dict[str, Any] | None = None
    user_defined_message: dict[str, Any] | None = None

    def routing_metadata(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "task_type": self.task_type,
            "task_scope": self.task_scope,
            "resolution_kind": self.resolution_kind,
            "harness_id": self.harness.harness_id if self.harness else None,
            "harness_family": self.harness.family if self.harness else None,
            "fallback_reason": self.fallback_reason,
            "matrix_cell": self.matrix_cell,
            "primary_model": self.primary_model,
            "secondary_model": self.secondary_model,
            "generalist_model": self.generalist_model,
            "rejection_message": self.rejection_message,
            "user_defined_message": self.user_defined_message,
        }


class CleanResolution:
    """Resolve model roles first, then select a task harness."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.policy = cfg["clean_resolution"]
        self.model_refs = self.policy["model_refs"]
        self.rows = self.policy["domain_rows"]
        if not self.rows:
            raise MatrixConfigError(
                "matrix must declare at least one domain row in "
                "clean_resolution.domain_rows"
            )
        self.harnesses = self.policy["harnesses"]
        self.personas = self.policy.get("personas", {})
        # Load rejection templates.  All 3 non-SDST scopes must be
        # present (the resolver uses these for the rejection path).
        self.rejection_messages: dict[str, dict[str, str]] = self.policy.get(
            "rejection_messages", {}
        )
        required_scopes = set(SCOPE_REJECTION.keys())
        missing_scopes = required_scopes - set(self.rejection_messages.keys())
        if missing_scopes:
            raise MatrixConfigError(
                f"matrix is missing rejection_messages for these task scopes: "
                f"{sorted(missing_scopes)}"
            )
        self.minimum_confidence = float(self.policy.get("min_domain_confidence", 0.85))
        fallback = cfg["models"]["general_fallback"]
        self.generalist_model = str(fallback["model_id"])
        self.generalist_provider = str(fallback.get("provider", "ollama"))
        # Load-time validation: every task_harnesses key must be in
        # the canonical 6-value task_type set.  This is a config
        # error, not a runtime fallback.
        invalid_keys: set[str] = set()
        for domain_key, row in self.rows.items():
            for task_key in row.get("task_harnesses", {}).keys():
                if task_key not in CANONICAL_TASK_TYPES:
                    invalid_keys.add(f"{domain_key}.task_harnesses.{task_key}")
        if invalid_keys:
            raise MatrixConfigError(
                f"matrix declares task_harnesses keys outside the canonical "
                f"6-value task_type set: {sorted(invalid_keys)}"
            )

    def closed_vocabulary(self) -> dict[str, frozenset[str]]:
        """Return the closed vocabularies used by the routing layer.

        The domain set is derived from the matrix's domain_rows keys,
        with `domain_other` auto-injected if missing.  The task_type
        and task_scope sets are platform-invariants defined at the
        top of this module.

        The matrix never gets to redefine the task_type or task_scope
        vocabularies — those are platform constants.  The matrix
        maps (domain, task_type) -> cell, it does not enumerate
        the task_types.

        Two task_type sets are reported:
          - "task_types_router_emit": the 5-value set the router
            model is permitted to emit.  Excludes `user_defined`,
            which is reserved for user override only.
          - "task_types_reach": the 6-value set the matrix accepts
            from any source (router emit + user override).

        Returns a dict with four keys:
          - "domains": frozenset of domain tags
          - "task_types_router_emit": frozenset of 5 emit-allowed tags
          - "task_types_reach": frozenset of 6 reach-allowed tags
          - "task_scopes": frozenset of 4 task_scope tags
        """
        domains = set(self.rows.keys())
        if "domain_other" not in domains:
            domains.add("domain_other")
        return {
            "domains": frozenset(domains),
            "task_types_router_emit": ROUTER_EMIT_TASK_TYPES,
            "task_types_reach": CANONICAL_TASK_TYPES,
            "task_scopes": CANONICAL_TASK_SCOPES,
        }

    def coding_domains(self) -> set[str]:
        """Derive the coding-domain set from the matrix.

        Matrix-defined truth (user rulings 2026-09-04): a domain is a
        CODING domain iff its row maps ``new_code_generation`` to a
        DISTINCT harness (not a ``general_use`` alias).  Alias entries
        like ``new_code_generation: general_use`` describe fallback
        behavior, not domain semantics.  Non-coding domains (e.g.
        domain_math, domain_physics, domain_other) are restricted to
        general_use / modify_artifact / evaluate_artifact (+
        user_defined via override); the router clamps
        new_code_generation and bug_fix emissions for them.

        NOTE: domain vs model is orthogonal — the generalist model may
        still be assigned to coding-domain slots (T5/slot fallback);
        the gate is about which requests a DOMAIN may receive, never
        which model a slot runs.
        """
        coding: set[str] = set()
        for domain_key, row in self.rows.items():
            mapping = (row.get("task_harnesses", {}) or {}).get("new_code_generation")
            if mapping is not None and str(mapping) != "general_use":
                coding.add(domain_key)
        return coding

    def domain_anchors(self) -> dict[str, dict[str, list[str]]]:
        """Return per-domain semantic anchors for the router prompt.

        Each domain row may declare optional ``description``,
        ``include_topics``, and ``exclude_topics`` fields (see the
        ``personas:`` block's "HOW TO REGISTER A NEW DOMAIN" docstring
        in ``config.yaml``).  These are STEERING signals: they nudge the
        classifier into the correct semantic neighborhood for each
        domain, and the forced selection from the closed domain set does
        the rest.  They are not hard gates.

        Returns a mapping ``{domain_name: {"description": str,
        "include_topics": [str], "exclude_topics": [str]}}``.  A domain
        row with no anchor fields yields empty lists / empty string, so
        callers can render them unconditionally.
        """
        anchors: dict[str, dict[str, list[str]]] = {}
        for domain_key, row in self.rows.items():
            anchors[domain_key] = {
                "description": str(row.get("description", "")),
                "include_topics": list(row.get("include_topics", []) or []),
                "exclude_topics": list(row.get("exclude_topics", []) or []),
            }
        return anchors

    def resolve(self, decision: CleanRouterDecision, think_override: bool | None = None) -> ResolvedWork:
        """Resolve a router decision into a model-aware, harness-aware work item.

        The router emits (domain, task_type, task_scope) directly.
        The resolver looks up the domain row, picks the harness for
        the task_type, and selects the model pair.

        ``think_override`` (the per-request effective think state, from a
        /thinkon|/thinkoff slash command) selects which THINK-STATE-DEPENDENT
        role ordering the matrix node uses, IF the node declares one.  It
        never changes how a model behaves -- the model's think mode is fixed
        by its model-card default.  ``think_override`` only picks the order
        (e.g. think-on -> ornith primary / qwen secondary; think-off default
        -> qwen primary / ornith secondary).
        """
        domain = decision.domain
        task_type = decision.task_type
        task_scope = decision.task_scope

        # --- Task scope rejection ---
        # --- Task scope rejection ---
        # Three non-SDST scopes route to rejection templates defined
        # in `clean_resolution.rejection_messages:`.  No model is
        # invoked; the harness is None; the rejection_message carries
        # the template content for the server to emit.
        if task_scope != "single_domain_single_task":
            template = self.rejection_messages.get(task_scope)
            if template is None:
                # Should never happen — __init__ validated all 3 scopes.
                raise MatrixConfigError(
                    f"missing rejection template for task_scope: {task_scope}"
                )
            return ResolvedWork(
                domain=domain,
                task_type=task_type,
                task_scope=task_scope,
                resolution_kind="rejection",
                fallback_reason=f"task_scope_not_supported: {task_scope}",
                matrix_cell=f"{domain}/{task_scope}",
                rejection_message={
                    "code": template["code"],
                    "severity": template["severity"],
                    "task_scope": task_scope,
                    "message": template["message"],
                },
            )

        # --- Domain row lookup ---
        row = self.rows.get(domain, {})
        if not row:
            # Domain not configured — fall through to domain_other
            row = self.rows.get("domain_other", {})
            if not row:
                # No domain_other row either — use generalist directly
                return self._generalist(domain, task_type, task_scope, "domain_not_configured")
            domain = "domain_other"

        # --- user_defined_unavailable path ---
        # When task_type=user_defined is set via user override, the
        # cell must have a user_defined task_harness entry.  If it
        # does not, the turn is BLOCKED (the user is informed; the
        # model is not invoked; no half-measure fallback).
        if task_type == "user_defined":
            if "user_defined" not in row.get("task_harnesses", {}):
                return ResolvedWork(
                    domain=domain,
                    task_type=task_type,
                    task_scope=task_scope,
                    resolution_kind="user_defined_unavailable",
                    fallback_reason="user_defined_harness_not_registered",
                    matrix_cell=f"{domain}/{task_type}",
                    user_defined_message={
                        "code": "user_defined_unavailable",
                        "severity": "error",
                        "domain": domain,
                        "message": (
                            f"User override requested task_type=user_defined "
                            f"for domain '{domain}', but no user_defined "
                            f"harness is registered for that domain. "
                            f"Falling back is disabled. Submit again with "
                            f"a domain override that has a user_defined "
                            f"harness, or register a user_defined harness "
                            f"for '{domain}'."
                        ),
                    },
                )

        task_cell = row.get("task_cells", {}).get(task_type, {})
        # Think-state-dependent role ordering.  A task_cell may declare a
        # per-think-state branch: {"think_on": {...}, "think_off": {...}}.
        # The request's effective think state (think_override) selects the
        # branch; default (None / think_off) uses the think_off ordering.
        # If no branch structure exists, fall back to flat primary/secondary.
        if "think_on" in task_cell or "think_off" in task_cell:
            branch_key = "think_on" if think_override is True else "think_off"
            branch = task_cell.get(branch_key) or task_cell.get("think_off") or {}
            primary_ref = branch.get("primary_model_ref", row.get("primary_model_ref", "generalist"))
            secondary_ref = branch.get("secondary_model_ref", row.get("secondary_model_ref", "generalist"))
        else:
            primary_ref = task_cell.get("primary_model_ref", row.get("primary_model_ref", "generalist"))
            secondary_ref = task_cell.get("secondary_model_ref", row.get("secondary_model_ref", "generalist"))
        primary, primary_provider = self._model(primary_ref)
        secondary, secondary_provider = self._model(secondary_ref)
        harness_id = row.get("task_harnesses", {}).get(task_type, "general_use")
        task_matches = task_type in row.get("task_harnesses", {})
        persona_key = row.get("persona", domain)
        # Compose the canonical harness with the cell's per-node
        # runtime overrides (deep merge + range clamping).
        # The `language` tag is derived from the domain name and
        # substituted into the harness's `assistant_prefill` template.
        cell_overrides = task_cell.get("runtime_overrides")
        assistant_prefill_override = (
            task_cell["assistant_prefill"]
            if "assistant_prefill" in task_cell else _UNSET
        )
        language = _domain_to_language(domain)
        harness = self._compose_harness(
            harness_id, cell_overrides, language=language,
            domain_tag=language,
            assistant_prefill_override=assistant_prefill_override,
        )
        return ResolvedWork(
            domain=domain, task_type=task_type, task_scope=task_scope,
            primary_model=primary, secondary_model=secondary,
            primary_provider=primary_provider, secondary_provider=secondary_provider,
            generalist_model=self.generalist_model,
            generalist_provider=self.generalist_provider,
            persona=str(self.personas.get(persona_key, self.personas.get("general", "You are a helpful assistant."))),
            harness=harness, resolution_kind="specialist",
            fallback_reason=None if task_matches else "task_type_no_match",
            matrix_cell=f"{domain}/{task_type}" if task_matches else f"{domain}/default",
            primary_options=self._model_options(primary), secondary_options=self._model_options(secondary),
            generalist_options=self._model_options(self.generalist_model),
            primary_think=self._model_think(primary), secondary_think=self._model_think(secondary),
            generalist_think=self._model_think(self.generalist_model),
            primary_raw_prefill=self._model_raw_prefill(primary), secondary_raw_prefill=self._model_raw_prefill(secondary),
            generalist_raw_prefill=self._model_raw_prefill(self.generalist_model),
        )

    # ------------------------------------------------------------------
    # Trivial-answer work resolution (2026-09-03 unification)
    # ------------------------------------------------------------------

    def trivial_work(self) -> ResolvedWork:
        """Build the ResolvedWork for a trivial-answer turn.

        The model and provider come from ``models.router`` (config-
        owned); the harness is the config-declared ``trivial_answer``
        harness composed through the SAME machinery as every other
        harness. No hand-built packages anywhere downstream.
        """
        router_cfg = self.cfg["models"]["router"]
        model_id = str(router_cfg["model_id"])
        provider = str(router_cfg.get("provider", "flm"))
        harness = self._compose_harness("trivial_answer", None)
        return ResolvedWork(
            domain="domain_other",
            task_type="trivial_answer",
            task_scope="single_domain_single_task",
            primary_model=model_id,
            secondary_model=model_id,
            primary_provider=provider,
            secondary_provider=provider,
            generalist_model=model_id,
            generalist_provider=provider,
            persona="",  # trivial harness system_suffix carries the voice
            harness=harness,
            resolution_kind="specialist",
            fallback_reason=None,
            matrix_cell="trivial/T0",
            primary_options=self._model_options(model_id),
            secondary_options=self._model_options(model_id),
            generalist_options=self._model_options(model_id),
            primary_think=self._model_think(model_id),
            secondary_think=self._model_think(model_id),
            generalist_think=self._model_think(model_id),
            primary_raw_prefill=self._model_raw_prefill(model_id),
            secondary_raw_prefill=self._model_raw_prefill(model_id),
            generalist_raw_prefill=self._model_raw_prefill(model_id),
        )

    def trivial_fallback_work(self) -> ResolvedWork | None:
        """Trivial-answer fallback work using models.router_fallback.

        Returns None when no fallback router is configured — the caller
        then surfaces a visible error instead of a silent downgrade.
        """
        fb = self.cfg["models"].get("router_fallback")
        if not fb or not fb.get("model_id"):
            return None
        model_id = str(fb["model_id"])
        provider = str(fb.get("provider", "ollama"))
        harness = self._compose_harness("trivial_answer", None)
        return ResolvedWork(
            domain="domain_other",
            task_type="trivial_answer",
            task_scope="single_domain_single_task",
            primary_model=model_id,
            secondary_model=model_id,
            primary_provider=provider,
            secondary_provider=provider,
            generalist_model=model_id,
            generalist_provider=provider,
            persona="",
            harness=harness,
            resolution_kind="specialist",
            fallback_reason="trivial_fallback",
            matrix_cell="trivial/T0_fallback",
            primary_options=self._model_options(model_id),
            secondary_options=self._model_options(model_id),
            generalist_options=self._model_options(model_id),
            primary_think=self._model_think(model_id),
            secondary_think=self._model_think(model_id),
            generalist_think=self._model_think(model_id),
            primary_raw_prefill=self._model_raw_prefill(model_id),
            secondary_raw_prefill=self._model_raw_prefill(model_id),
            generalist_raw_prefill=self._model_raw_prefill(model_id),
        )

    def _generalist(self, domain: str, task_type: str, task_scope: str, reason: str) -> ResolvedWork:
        return ResolvedWork(
            domain=domain, task_type=task_type, task_scope=task_scope,
            primary_model=self.generalist_model, secondary_model=self.generalist_model,
            primary_provider=self.generalist_provider, secondary_provider=self.generalist_provider,
            persona=str(self.personas.get("general_assistant", "You are a helpful assistant.")),
            harness=self._harness("general_use"), resolution_kind="generalist",
            fallback_reason=reason, matrix_cell=None,
            primary_options=self._model_options(self.generalist_model),
            secondary_options=self._model_options(self.generalist_model),
            generalist_options=self._model_options(self.generalist_model),
            primary_think=self._model_think(self.generalist_model),
            secondary_think=self._model_think(self.generalist_model),
            generalist_think=self._model_think(self.generalist_model),
            primary_raw_prefill=self._model_raw_prefill(self.generalist_model),
            secondary_raw_prefill=self._model_raw_prefill(self.generalist_model),
            generalist_raw_prefill=self._model_raw_prefill(self.generalist_model),
        )

    def _model(self, ref: str) -> tuple[str, str]:
        if ref == "generalist":
            return self.generalist_model, self.generalist_provider
        key = ref.removeprefix("model:")
        model = self.model_refs[key]
        return str(model["model_id"]), str(model.get("provider", "ollama"))

    def _model_options(self, model_id: str) -> dict[str, Any]:
        for item in self.cfg["models"].values():
            if item.get("model_id") == model_id:
                return {k: item[k] for k in ("num_ctx", "num_predict", "keep_alive", "think") if k in item}
        for item in self.model_refs.values():
            if item.get("model_id") == model_id:
                return {k: item[k] for k in ("num_ctx", "num_predict", "keep_alive", "think") if k in item}
        return {}

    def _model_think(self, model_id: str) -> bool | None:
        """Return the model-card `think` default (True/False), or None if unset."""
        for item in self.cfg["models"].values():
            if item.get("model_id") == model_id and "think" in item:
                return bool(item["think"])
        for item in self.model_refs.values():
            if item.get("model_id") == model_id and "think" in item:
                return bool(item["think"])
        return None

    def _model_raw_prefill(self, model_id: str) -> bool | None:
        """Return the model-card `raw_prefill` tolerance (True/False), or None if unset.

        2026-09-09 (user ruling): a model-card property, separate from the
        think policy. False = the model does NOT tolerate raw prompt
        continuation (assistant response-prefix injection via
        /api/generate raw); the transport must use the native /api/chat
        path and drop any text prefill.
        """
        for item in self.cfg["models"].values():
            if item.get("model_id") == model_id and "raw_prefill" in item:
                return bool(item["raw_prefill"])
        for item in self.model_refs.values():
            if item.get("model_id") == model_id and "raw_prefill" in item:
                return bool(item["raw_prefill"])
        return None

    def compose_harness_variant(self, harness_id: str, *, domain: str) -> CleanHarnessSpec:
        """Compose an explicitly selected harness variant.

        This is used by opt-in A/B experiments.  Model selection and
        task routing still come from the resolved work; only the
        harness definition changes.
        """
        language = _domain_to_language(domain)
        return self._compose_harness(
            harness_id, None, language=language, domain_tag=language,
        )

    def _harness_raw(self, harness_id: str, stack: tuple[str, ...] = ()) -> dict[str, Any]:
        """Resolve declarative harness inheritance without runtime branching."""
        if harness_id in stack:
            raise MatrixConfigError(
                f"harness inheritance cycle: {' -> '.join((*stack, harness_id))}"
            )
        raw = self.harnesses[harness_id]
        parent_id = raw.get("extends")
        if not parent_id:
            return copy.deepcopy(raw)
        base = self._harness_raw(str(parent_id), (*stack, harness_id))
        child = {key: value for key, value in raw.items() if key != "extends"}
        return _merge_dicts(base, child)

    def _harness(self, harness_id: str, language: str = "", domain_tag: str = "") -> CleanHarnessSpec:
        raw = self._harness_raw(harness_id)
        prefill = raw.get("assistant_prefill")
        if prefill and language:
            prefill = prefill.replace("{{language}}", language)
        system_suffix = str(raw.get("system_suffix", ""))
        if system_suffix and domain_tag:
            system_suffix = system_suffix.replace("{{domain_tag}}", domain_tag)
        system_suffix_thinkon = raw.get("system_suffix_thinkon")
        if system_suffix_thinkon and domain_tag:
            system_suffix_thinkon = str(system_suffix_thinkon).replace("{{domain_tag}}", domain_tag)
        output_instruction_thinkon = raw.get("output_instruction_thinkon")
        if output_instruction_thinkon and domain_tag:
            output_instruction_thinkon = str(output_instruction_thinkon).replace("{{domain_tag}}", domain_tag)
        return CleanHarnessSpec(
            harness_id=harness_id, family=str(raw.get("family", "general_use")),
            tiers=tuple(raw.get("tiers", ("T0",))),
            local_change_order=bool(raw.get("local_change_order", False)),
            cloud_change_order=bool(raw.get("cloud_change_order", False)),
            needs_artifact_store=bool(raw.get("needs_artifact_store", False)),
            needs_t3_context=bool(raw.get("needs_t3_context", False)),
            system_suffix=system_suffix,
            output_instruction=str(raw.get("output_instruction", "")),
            system_suffix_thinkon=system_suffix_thinkon,
            output_instruction_thinkon=output_instruction_thinkon,
            closed_think=raw.get("closed_think"), assistant_prefill=prefill,
            sampling=dict(raw.get("sampling", {})), runtime=dict(raw.get("runtime", {})),
            escalation_ladder=dict(raw.get("escalation_ladder", {})),
            rounds=dict(raw.get("rounds", {})),
        )

    def _compose_harness(
        self, harness_id: str, cell_overrides: dict[str, Any] | None,
        language: str = "",
        domain_tag: str = "",
        assistant_prefill_override: Any = _UNSET,
    ) -> CleanHarnessSpec:
        """Compose the canonical harness with the cell's runtime overrides.

        The canonical harness's `runtime:` block carries the
        defaults.  The cell's `runtime_overrides:` is deep-merged
        into the defaults.  Out-of-range values are clamped to
        the harness's declared range (per the harness's
        `runtime.parameter_ranges:` block).  Surface-validation
        drops unknown parameters with a logged warning.

        The `language` parameter is substituted into the
        harness's `assistant_prefill` template (replacing any
        ``{{language}}`` placeholders).  The `domain_tag`
        parameter is substituted into the harness's
        `system_suffix` (replacing any ``{{domain_tag}}``
        placeholders).
        """
        base = self._harness(harness_id, language=language, domain_tag=domain_tag)
        if not cell_overrides and assistant_prefill_override is _UNSET:
            return base
        ranges = base.runtime.get("parameter_ranges", {})
        merged_runtime = (
            _deep_merge_with_clamp(base.runtime, cell_overrides, ranges)
            if cell_overrides else copy.deepcopy(base.runtime)
        )
        assistant_prefill = (
            base.assistant_prefill
            if assistant_prefill_override is _UNSET
            else assistant_prefill_override
        )
        return CleanHarnessSpec(
            harness_id=base.harness_id,
            family=base.family,
            tiers=base.tiers,
            local_change_order=base.local_change_order,
            cloud_change_order=base.cloud_change_order,
            needs_artifact_store=base.needs_artifact_store,
            needs_t3_context=base.needs_t3_context,
            system_suffix=base.system_suffix,
            output_instruction=base.output_instruction,
            system_suffix_thinkon=base.system_suffix_thinkon,
            output_instruction_thinkon=base.output_instruction_thinkon,
            closed_think=base.closed_think,
            assistant_prefill=assistant_prefill,
            sampling=base.sampling,
            runtime=merged_runtime,
            escalation_ladder=base.escalation_ladder,
            rounds=base.rounds,
        )


class CleanHarness:
    """Compose a resolved work item without selecting model identities."""

    def __init__(self, work: ResolvedWork, think_override: bool | None = None) -> None:
        self.work = work
        self.think_override = think_override

    def compose(self, unit: CleanUnit, tier: str, context: CleanTierContext) -> CleanPackage:
        persona = "" if self.work.harness.harness_id == "evaluate_artifact" else self.work.persona
        # Per-model think-on: if the slot's model-card default is think-on,
        # use the think-aware system suffix (which does NOT ask for a DRAFT
        # block with inline reasoning -- think-on already reasons natively).
        ladder = self.work.harness.escalation_ladder
        tier_cfg = ladder.get(tier.lower(), {})
        slot_think = {
            "primary": self.work.primary_think,
            "secondary": self.work.secondary_think,
            "generalist": self.work.generalist_think,
        }.get(tier_cfg.get("model_slot", "primary"))
        # 2026-09-29 (user ruling): the PROMPT VARIANT follows the BOUND
        # MODEL's card.  It is NOT a request property.
        #
        # `think_override` (the per-request /thinkon|/thinkoff state) selects
        # ROLE ORDERING only -- which model stands in the primary slot.  That
        # is what this class's own docstring and the comment above the old
        # line both claimed; the old line contradicted them and overwrote the
        # slot's model-card value unconditionally.
        #
        # MEASURED DEFECT IT CAUSED (LCB 75-QID, 2026-09-29): in a think-on
        # run the secondary slot (t2 -> model_slot="secondary" -> r25, card
        # think:false) was handed the THINKON variant ("output exactly ONE
        # clean fenced code block ... No prose, no draft block") while r25
        # still emitted its trained DRAFT+FINAL.  The contract wants the LAST
        # of two blocks and found a single block -> reject -> escalate.  On
        # the SAME 20 qids the SAME model parsed 13/20 as primary but only
        # 2/20 as secondary, and 18/20 went to cloud.  That broke the T1/T2
        # mirror: T2 is supposed to be an exact slot-level mirror of T1.
        #
        # TWO APPROVED CARVE-OUTS:
        #  * generalist -- MAY be overridden by the request.  A general-purpose
        #    model has no per-run tuned contract and the user wants /thinkon
        #    to control it.  (No ladder tier declares model_slot="generalist"
        #    today, so this branch is currently unreachable; the carve-out
        #    preserves the documented ability for when one does.)
        #  * cloud -- UNAFFECTED here and by design: cloud runs its own
        #    `meta_harness_ladder.cloud` settings (reasoning_effort / max_tokens
        #    / think), read directly in runtime.py, and `think_override` never
        #    reaches it.
        if self.think_override is not None \
                and tier_cfg.get("model_slot", "primary") == "generalist":
            slot_think = self.think_override
        # NOTE: T5 (the post-ladder generalist attempt) has NO ladder entry, so
        # the lookup above falls back to model_slot="primary" and T5 would take
        # the PRIMARY model's think.  `_invoke_t5_body` composes with tier
        # "T5"; it therefore resolves the generalist slot explicitly there
        # (see runtime.py / the "T5" branch below in compose()).
        if str(tier).upper() == "T5":
            slot_think = self.work.generalist_think
            if self.think_override is not None:
                slot_think = self.think_override
        thinkon_suffix = self.work.harness.system_suffix_thinkon
        system_suffix = thinkon_suffix if (slot_think is True and thinkon_suffix) else self.work.harness.system_suffix
        thinkon_out = self.work.harness.output_instruction_thinkon
        output_instruction = thinkon_out if (slot_think is True and thinkon_out) else self.work.harness.output_instruction
        system_parts = [persona, system_suffix, output_instruction]
        user = unit.prompt
        # 2026-09-28 (user ruling): NO LONGER gated on
        # ``harness_id == "general_use"``.  ``context_text`` now carries the
        # attached text / working-artifact block for ALL harnesses; gating it
        # here would silently DROP every attachment for any other harness now
        # that the image harness swap is gone.  The producer side
        # (server.py's ``context_text`` assignment) was de-gated in tandem.
        #
        # Reversal: restore ``harness_id == "general_use" and`` at this site,
        # at server.py's ``context_text`` assignment, and at runtime.py's
        # ``_cloud_user`` assignment.
        if unit.context_text:
            user = f"{unit.context_text}\n\n[AUTHORITATIVE USER REQUEST]\n{unit.prompt}"
        # Data-driven user_modifier from the escalation_ladder block.
        # Replaces the previous hardcoded T1 prior_error message.
        # The ladder keys are lowercase (t0, t1a, t1b, t2); the tier
        # names are uppercase (T0, T1a, T1b, T2).  Normalize to lowercase.
        user_modifier = tier_cfg.get("user_modifier")
        if user_modifier:
            filled = str(user_modifier)
            if context.prior_error:
                filled = filled.replace("{prior_error}", context.prior_error)
                filled = filled.replace("{t0_error}", context.prior_error)
            user = f"{user}\n\n{filled}"
        messages = [{"role": "system", "content": "\n\n".join(part.strip() for part in system_parts if part.strip())}, {"role": "user", "content": user}]
        if slot_think is True:
            response_prefix = None
            closed = False
        else:
            closed = self.work.harness.closed_think
            if closed is None:
                closed = True
            # v14 (2026-08-10): per-tier prefill override.  The escalation_ladder
            # tier block may declare its own ``assistant_prefill``, which takes
            # precedence over the harness-level default.  This lets the bug_fix
            # harness give the PRIMARY no `<patch>` anchor (matching the direct
            # 62% run) while the SECONDARY keeps the `<patch>` anchor (r25's
            # prose-only dither fix).  Falls back to the harness default when the
            # tier declares none.
            prefill = tier_cfg.get("assistant_prefill")
            if prefill is None:
                prefill = self.work.harness.assistant_prefill
            # A null harness prefill is an explicit no-prefix mode. The
            # provider's chat template owns any native think-off markers.
            response_prefix = prefill if prefill is not None else None
        # 2026-09-09 (user ruling): raw-prefill tolerance is a model-card
        # property. A model with raw_prefill=False (gemma4 lineage) does
        # NOT tolerate raw prompt continuation -- drop any text prefill
        # and let the native /api/chat path own think suppression.
        slot_raw_prefill = {
            "primary": self.work.primary_raw_prefill,
            "secondary": self.work.secondary_raw_prefill,
            "generalist": self.work.generalist_raw_prefill,
        }.get(tier_cfg.get("model_slot", "primary"))
        if slot_raw_prefill is False:
            response_prefix = None
        # Data-driven model_slot from the escalation_ladder block.
        # Replaces the previous hardcoded `primary = tier in {"T0", "T1"}`.
        model_slot = tier_cfg.get("model_slot", "primary")
        if model_slot == "secondary":
            model_id = self.work.secondary_model
            provider = self.work.secondary_provider
            options = dict(self.work.secondary_options)
            model_think = self.work.secondary_think
        elif model_slot == "generalist":
            model_id = self.work.generalist_model
            provider = self.work.generalist_provider
            options = dict(self.work.generalist_options)
            model_think = self.work.generalist_think
        else:  # primary (default)
            model_id = self.work.primary_model
            provider = self.work.primary_provider
            options = dict(self.work.primary_options)
            model_think = self.work.primary_think
        # 2026-09-29 (user ruling): the slot's think FLAG is its model-card
        # property, not a request property -- same ruling and same carve-outs
        # as the prompt-variant site above.  Forcing /thinkon onto r25 (card
        # think:false) asked a model that does not reason natively to behave
        # as though it did, and its output shape then disagreed with its
        # prompt variant.  Sites 1/2/3 must move together; a partial fix is
        # worse than none.
        _slot_name = tier_cfg.get("model_slot", "primary")
        if str(tier).upper() == "T5":
            _slot_name = "generalist"
        if self.think_override is not None and _slot_name == "generalist":
            model_think = self.think_override
        # Package-level raw-prefill tolerance for the transport gate.
        package_raw_prefill = slot_raw_prefill
        options: dict[str, Any] = options
        # Make sure bounded context is always present, in case the
        # model row did not declare one explicitly.  The transport
        # nests every Ollama option under ``options`` so ``num_ctx``
        # must be present here.
        if "num_ctx" not in options:
            options["num_ctx"] = 32768
        for key in ("num_ctx", "num_predict", "keep_alive"):
            if key in self.work.harness.sampling:
                options.setdefault(key, self.work.harness.sampling[key])
        sampling = {key: value for key, value in self.work.harness.sampling.items() if key not in {"num_ctx", "num_predict", "keep_alive"}}
        return CleanPackage(
            model_id=model_id,
            provider=provider,
            messages=messages, sampling=sampling, options=options,
            diagnostics={**self.work.routing_metadata(), "tier": tier},
            think=model_think,
            raw_prefill=package_raw_prefill,
            response_prefix=response_prefix,
            images=list(getattr(unit, "images", []) or []),
        )
