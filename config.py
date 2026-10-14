"""Configuration loading and validation for the clean slice only.

`config.yaml` is the SINGLE SOURCE OF TRUTH for the tag -> model binding.  This
module is therefore the single point at which that binding is checked: see
`validate_config`, whose binding-integrity section enforces the tag/reference
rules that the retired `model_assignment.yaml` merge used to supply implicitly.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


class ConfigError(ValueError):
    """Raised when a clean configuration cannot support safe execution."""


def load_config(
    path: str | Path | None = None,
    *,
    strict_bindings: bool = True,
    binding_notices: bool = False,
) -> dict[str, Any]:
    """Load the isolated clean configuration without legacy helpers.

    `config.yaml` is the sole source of truth for the tag -> model binding;
    there is no second binding file and no merge.

    `strict_bindings=False` downgrades binding-integrity findings from errors to
    warnings, for deliberately partial configs (e.g. adding a domain before its
    models are bound).

    `binding_notices=True` additionally emits the INFORMATIONAL findings
    (unreferenced `model_refs` entries, duplicate bindings).  These are off by
    default because they are legitimate states -- `cloud_text`/`cloud_vision`
    are deliberately unreferenced mirrors of `backends.cloud` -- and a shipped
    boot must not warn about a correct configuration.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    validate_config(
        cfg, strict_bindings=strict_bindings, binding_notices=binding_notices
    )
    return cfg


def collect_referenced_tags(cfg: dict[str, Any]) -> dict[str, list[str]]:
    """Map every referenced tag -> the site(s) that reference it.

    Sites are human-readable strings so a ConfigError can name exactly where a
    bad reference lives.  This reads the SAME shapes `resolution.py` resolves, so
    validation cannot drift from runtime semantics:

      * row-level      `domain_rows[d].primary_model_ref` / `.secondary_model_ref`
      * task-cell      `domain_rows[d].task_cells[t].<same keys>`
      * think-branch   `domain_rows[d].task_cells[t].{think_on,think_off}.<same>`

    Only `model:<tag>` values are collected.  The literal `generalist` (and an
    absent key) are deliberate "no specialist" markers, not tag references --
    `resolution.py` returns the generalist for both.
    """
    sites: dict[str, list[str]] = {}
    rows = (cfg.get("clean_resolution") or {}).get("domain_rows") or {}
    keys = ("primary_model_ref", "secondary_model_ref")

    def _add(value: Any, where: str) -> None:
        if isinstance(value, str) and value.startswith("model:"):
            sites.setdefault(value.removeprefix("model:"), []).append(where)

    for domain, row in rows.items():
        if not isinstance(row, dict):
            continue
        for key in keys:
            _add(row.get(key), f"domain_rows.{domain}.{key}")
        cells = row.get("task_cells") or {}
        if not isinstance(cells, dict):
            continue
        for task, cell in cells.items():
            if not isinstance(cell, dict):
                continue
            if "think_on" in cell or "think_off" in cell:
                for branch in ("think_on", "think_off"):
                    block = cell.get(branch) or {}
                    if isinstance(block, dict):
                        for key in keys:
                            _add(
                                block.get(key),
                                f"domain_rows.{domain}.task_cells.{task}.{branch}.{key}",
                            )
            else:
                for key in keys:
                    _add(cell.get(key), f"domain_rows.{domain}.task_cells.{task}.{key}")
    return sites


def validate_config(
    cfg: dict[str, Any],
    *,
    strict_bindings: bool = True,
    binding_notices: bool = False,
) -> None:
    """Validate only the configuration required by the clean lane.

    Three sections, in order:

    1. **structure** -- required sections, VRAM tier, generalist fallback.
    2. **binding shape** -- every `model_refs` entry carries a usable `model_id`.
    3. **binding integrity** -- every referenced tag exists; duplicates warn.

    Sections 2 and 3 are the single point-source-of-truth check on the tag ->
    model binding.  They matter because `resolution.py::_model()` does
    `self.model_refs[key]` (a bare `KeyError` on an unknown tag) and
    `str(model["model_id"])`, which stringifies `None` to the literal `"None"`
    and would hand that to the backend as a model name.
    """
    if not isinstance(cfg, dict):
        raise ConfigError("Clean configuration must be a mapping")
    for key in ("backends", "models", "vram_tiers", "clean_resolution"):
        if key not in cfg:
            raise ConfigError(f"Missing required clean configuration section: {key}")
    tiers = cfg["vram_tiers"]
    active = tiers.get("active")
    if active not in tiers.get("defined", {}):
        raise ConfigError(f"Active VRAM tier '{active}' is not defined")
    compact = tiers["defined"][active]
    if int(compact.get("max_resident_local_models", 1)) < 1:
        raise ConfigError("A VRAM tier must permit at least one local resident model")
    if int(compact.get("max_num_ctx", 0)) <= 0:
        raise ConfigError("A VRAM tier requires a positive max_num_ctx")
    fallback = cfg["models"].get("general_fallback", {})
    if not fallback.get("model_id"):
        raise ConfigError("models.general_fallback.model_id is required")

    _validate_binding(
        cfg, strict_bindings=strict_bindings, binding_notices=binding_notices
    )


def _validate_binding(
    cfg: dict[str, Any], *, strict_bindings: bool, binding_notices: bool
) -> None:
    """Single point of truth for tag -> model binding integrity.

    Raises ConfigError for a broken binding, or warns when `strict_bindings`
    is False.  The informational findings are emitted only when
    `binding_notices` is True.
    """
    findings: list[tuple[str, str]] = []

    refs = (cfg.get("clean_resolution") or {}).get("model_refs")
    if isinstance(refs, dict):
        for tag, entry in refs.items():
            if not isinstance(entry, dict):
                findings.append(("shape", f"model_refs.{tag} is not a mapping"))
                continue
            if "model_id" not in entry:
                findings.append(("shape", f"model_refs.{tag} has no 'model_id'"))
                continue
            model_id = entry.get("model_id")
            if not isinstance(model_id, str) or not model_id.strip():
                findings.append(
                    (
                        "shape",
                        f"model_refs.{tag}.model_id is empty -- the runtime would "
                        f"resolve the literal {str(model_id)!r} as a model name",
                    )
                )
            elif str(model_id).strip() != str(model_id):
                findings.append(
                    ("shape", f"model_refs.{tag}.model_id has surrounding whitespace")
                )

    sites = collect_referenced_tags(cfg)
    known = set(refs) if isinstance(refs, dict) else set()
    for tag in sorted(set(sites) - known):
        where = ", ".join(sorted(set(sites[tag]))[:4])
        more = f" (+{len(sites[tag]) - 4} more)" if len(sites[tag]) > 4 else ""
        findings.append(
            (
                "missing",
                f"tag '{tag}' is referenced but has no model_refs entry "
                f"-- referenced at: {where}{more}",
            )
        )

    # An unreferenced model_refs entry is INFORMATIONAL, never an error:
    # cloud_text / cloud_vision are mirrors of backends.cloud, referenced by
    # nothing on purpose.
    if binding_notices:
        for tag in sorted(known - set(sites)):
            warnings.warn(
                f"clean_resolution.model_refs.{tag} is not referenced by any "
                f"domain_rows entry (correct if it mirrors backends.cloud)",
                stacklevel=3,
            )

    # Duplicate bindings are legitimate aliases (an -a/-b pair pointed at one
    # model on purpose), so this is informational and never fails.
    by_model: dict[str, list[str]] = {}
    if isinstance(refs, dict):
        for tag, entry in refs.items():
            if isinstance(entry, dict):
                model_id = entry.get("model_id")
                if isinstance(model_id, str) and model_id.strip():
                    by_model.setdefault(model_id, []).append(tag)
    if binding_notices:
        for model_id, tags in sorted(by_model.items()):
            if len(tags) > 1:
                warnings.warn(
                    f"model_refs tags {sorted(tags)} all bind to the same model "
                    f"'{model_id}' (fine for a deliberate alias, a mistake otherwise)",
                    stacklevel=3,
                )

    if not findings:
        return
    detail = "\n  ".join(f"[{kind}] {msg}" for kind, msg in findings)
    report = f"config.yaml tag -> model binding check failed:\n  {detail}"
    if strict_bindings:
        raise ConfigError(report)
    warnings.warn(report, stacklevel=3)
