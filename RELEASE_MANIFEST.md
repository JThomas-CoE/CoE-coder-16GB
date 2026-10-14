# Release Inclusion Manifest

Status: pending owner review. Repository root: this directory (`CoE_coder_16GB_final_0_5`).

## Include

- All top-level runtime and CLI Python modules (`*.py`), including `CoE-coder.py`.
- `config.yaml` and `prefill_rates.json`.
- `README.md`, `requirements.txt`, `LICENSE`, `LICENSE-COMMERCIAL.md`, `.gitignore`, and this manifest.
- `benchmarks/`: the cache builder, the `lcb_runner` evaluation, prompt, and utility packages (the `lcb_runner/runner/` package is excluded, see below), the frozen 75-question manifest, the pass@3 summary JSONs `benchmarks/LiveCodeBench/output/release_pass3_campaign_summary_20261005.json` and `benchmarks/LiveCodeBench/output/qwen38_gemma4_pass3_summary_20261005.json` (the qwen3.8-27b-q3 and gemma4:12b comparison cited in the README's "Default generalist" section), and `data/HumanEval.jsonl`.
- `_HASHES.txt` as a source-payload integrity snapshot covering every shipped file; regenerate it after any shipped file changes.

The model files are not included. `config.yaml` names their expected IDs and remains the authoritative model-binding file.

## Exclude from Git

The following are local/runtime/generated material. `.gitignore` excludes them without deleting or moving them from this working folder:

- `artifacts_saved/**` (generated artifacts and metadata sidecars)
- `evaluate_objects/**` (local evaluation objects)
- `logs/**` and `*.log` (runtime and benchmark logs)
- `_checkpoints/**` (local pre/post images and recovery checkpoints)
- `__pycache__/**`, `*.py[cod]`, `.venv/**`, and `.pytest_cache/**`
- `humaneval_flexible_v1_*.jsonl`, `humaneval_*.json`, and `humaneval_*.jsonl` (benchmark result data)
- `.env*`, local editor settings, and operating-system metadata

## Pruned from this payload

Generated outputs, raw benchmark results, the local virtual environment, the `bench/` tier file, and the LiveCodeBench output ledger were removed from the working tree.

Excluded from the release by owner decision: the LiveCodeBench runner package `benchmarks/LiveCodeBench/lcb_runner/runner/` and `benchmarks/LiveCodeBench/pyproject.toml`. They were moved out of this tree to `..\_LCB_excluded_from_release\` (kept locally, not deleted). The release's cache builder and canonical runner do not import the runner package. The two smoke logs `_smoke_health.log` and `_smoke_health.err.log` remain on disk at the owner's direction. They are not part of the release and are excluded from `_HASHES.txt`. Review `git status --ignored` before staging.

## Licensing

- All College of Experts work is PolyForm Noncommercial 1.0.0 (`LICENSE`), with commercial licensing via `LICENSE-COMMERCIAL.md`. This covers runtime code, activation data, utility ranking and expert-selection criteria, pruning masks, pathology guards, and CoE escalation ladders. The README section "Scope of the PolyForm licence" states this.
- Upstream components keep their own terms: parent-model tensors (Ornith-1.5 MIT, Qwen3.6 Apache-2.0), gemma4:12b (Apache-2.0), and the bundled LiveCodeBench harness.
- The Tiel model release (staged separately on Hugging Face) is PolyForm as the primary licence for its weights and masks, with MIT inherited for the upstream tensors.

## Pre-publication checks

- Push to GitHub as public `CoE-coder-16GB` only after explicit owner approval.
- Confirm the licensing text and attribution notice with the project owner.
- Rebuild `_HASHES.txt` after any runtime source change and verify the package tree against the development source.
- Install `requirements.txt` in a clean Python 3.12 environment and run a server health smoke test.
- Confirm that no credentials, model weights, generated artifacts, or local logs are staged.
- Add the release test suite or clearly record the test coverage available in the selected payload before tagging.

Test coverage record (closes the item above). This payload ships no unit-test suite; per-module regression tests live in the development tree and are not part of the release. The validation available in this payload is: the clean-install health check (a fresh Python 3.12 venv built from `requirements.txt`, server start, `GET /health` returning `ok`; see the README Install section); `benchmarks/run_lcb_canonical.py --self-test`, which validates the LiveCodeBench runner against the pinned problem cache (run it only after the cache is obtained or rebuilt); and the bundled benchmark campaign summaries (HumanEval best-of-3 and LiveCodeBench v6 best-of-3), which evidence end-to-end runtime behavior — routing, harness selection, execution, validation, and escalation — for the shipped configuration. These checks exercise the system as shipped; they are not a substitute for a per-module test suite, which remains planned work.
