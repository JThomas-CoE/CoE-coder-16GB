# CoE-coder-16GB

This is the standalone CoE-coder runtime repository for a 16 GB GPU budget; model weights are not included. Model creation, activation profiling, pruning strategies, and model releases live in the [College of Experts AI research repository](https://github.com/JThomas-CoE/College-of-Experts-AI/tree/main).

The runtime implements the separability hypothesis — that domain capability in MoE models is concentrated in expert subsets that can be surgically extracted without post-surgery training, with unrelated capability measurably excised. The hypothesis and its demonstrations are documented in:

- J. Thomas, *College of Experts: Decomposing Mixture-of-Experts Architectures into Disk-Resident Specialist Networks for Consumer Hardware* (technical report v0.9, February 2026). [doi:10.5281/zenodo.23403227](https://doi.org/10.5281/zenodo.23403227)
- J. Thomas, *Separability of Intelligence in Mixture-of-Experts: Slicing Qwen3-Coder into Independent Domain Specialists* (whitepaper, March 2026). [doi:10.5281/zenodo.23403231](https://doi.org/10.5281/zenodo.23403231)
- J. Thomas, *Gemma4 College of Experts MMLU-Pro Domain Specialists: Automated Profiling Pipeline and Near/Far Transfer Analysis* (model card documentation, May 2026). [doi:10.5281/zenodo.23403233](https://doi.org/10.5281/zenodo.23403233)

## What it is

CoE-coder-16GB exposes an OpenAI-compatible chat API and an interactive CLI over memory constrained local specialist models. A domain/task matrix selects a specialist and its harness; bounded execution, validation, retry/escalation policies, artifact versioning, and session memory are managed by the runtime. In this instance the specialist models are MoE models that have had their ffn experts pruned by 50% resulting in about a 40 to 45% reduction in memory footprint excluding KV cache. None of these models has had post-pruning training. On the 75-question LiveCodeBench set, direct runs give Qwen 52/75 against 61/75 for the parent and Tiel 53/75 against 62/75; the benchmark sections below give the full python coding comparison.

The math and physics scores below come from the Hugging Face model cards. Each row compares a shipped specialist with its unpruned parent under the protocol that card reports. The cards differ in protocol and sample size, so read the gap within a row, not across rows.

| Domain | Pruned specialist | Parent | Benchmark and protocol | Pruned | Parent | Gap |
|---|---|---|---|---:|---:|---:|
| math | `math-a`, coe-gemma4-math-hc-14b-a4b-q4_k_m | gemma-4-26b-it | MMLU-Pro math, 1,351 questions, pass@1, think-off | 92.9% | 93.3% | -0.4 pp |
| math | `math-a`, same model | gemma-4-26b-it | MATH-500, think-off | 96.4% | 97.0% | -0.6 pp |
| math | `math-a`, same model | gemma-4-26b-it | AIME 2026, 30 problems, 32K context | 63.3% | 73.3% | -10.0 pp |
| math | `math-b`, coe-qwen3.6-mmlu_pro-math-r31-19b-a3b-q4_k_m | Qwen3.6-35B-A3B | MMLU-Pro math, 135 questions, 8K context, think-off | 88.15% | 91.85% | -3.70 pp |
| physics | `physics-a`, coe-qwen3.6-mmlu_pro-physics-r31-19b-a3b-q4_k_m | Qwen3.6-35B-A3B | MMLU-Pro physics, 130 questions, 8K context, think-off | 86.92% | 86.92% | 0.00 pp |
| physics | `physics-b`, coe-gemma4-physics-mmlu_pro-14b-a4b-q4 | gemma-4-26b-it | MMLU-Pro physics, off5 split, about 130 questions, think-off | 84.0% | 87.3% | -3.3 pp |
| physics | `physics-b`, same model | gemma-4-26b-it | MMLU-Pro physics, off4 split, 130 questions, think-off | 90.8% | 88.5% | +2.3 pp |

On the released MMLU-Pro sets, the pruned math and physics specialists sit within about 4 pp of their parents: math-a -0.4, math-b -3.7, physics-a 0.0, physics-b -3.3. The larger gaps come from harder or smaller tests: AIME 2026 for math-a (-10.0 pp on 30 problems) and the 92-question custom physics benchmark for physics-b (-9.2 pp, reported on its card and not in the table). The physics-b card reports +2.3 pp on its off4 split, so the sign of that gap depends on the split. The Gemma cards put the binomial error at about ±4 to 5 pp for 80 to 135 questions. These gaps have not been re-measured in this repository, and the card harnesses (8K to 32K context) are not the CoE-coder runtime settings.

**The result is smaller specialist models without post-surgery retraining.** Activation-directed surgery keeps half of the routed experts per layer. At the same Q4 quantization, the Qwen3.6 parent is about 23 GB and its pruned coding specialist is 13.2 GB (about 43% smaller); the Ornith/Tiel parent is also about 23 GB on disk and its pruned Tiel specialist is 13.4 GB (about 41% smaller). Halving the expert pool does not halve the whole model: attention, embeddings, shared weights, and other tensors remain. These are GGUF file sizes, not measurements of total inference VRAM including KV cache and runtime overhead; measured resident sizes are given in the benchmark sections.

The `rN` appearing in specialist names is **not** a retention or size percentage. Every mask here keeps **50%** of the routed experts (eg, 128 out of 256 for the qwen based models); `rN` is the swap percentage applied over rank-0 candidates within that 50%. See [Pruning methodology](#pruning-methodology) below for the algorithm and its sources.

The package is designed for a 16 GB VRAM tier and admits one local model at a time. It uses Ollama as the inference backend in the shipped configuration. FastFlowLM (FLM) is an optional NPU router; if it is unavailable, routing falls back to the configured local router model. A cloud escalation target is configurable and can be disabled.

### Why the footprint matters

The current GGUF surgery and backend-compatible expert layout use the 50% reduction step; this is a constraint of this implementation, not a proof that every GGUF model or future kernel would be optimal at a 2x expert reduction. Expert removal reduces the number of stored routed-expert weights while retaining the original selected weight values. It is independent of quantization in principle: pruning removes weights, while quantization uses fewer bits per retained weight. Their storage savings can compound, but the benchmarked sizes above compare models at the **same** Q4 precision, not a measured second quantization step.

Speculative, not measured here: smaller specialists might let a finer-grained domain router (for example, approaches such as JEV or CLM) share a fixed memory budget among more specialist agents. This release does not test that. It loads one local model at a time and does not measure concurrent residency, routing latency, or parent-equivalent quality across every domain. When no domain or task override is given, the router model (`qwen3.5:4b` in the shipped config, or FLM when available) selects the domain and task type. This is also a different approach than some recent schemes that keep hot experts in VRAM and offload the remaining into system RAM; here the total memory footprint is constrained (excluding the optional NPU memory use) and the different domains need not all be from the same parent model or model family. In principle a purpose built and trained model, either dense or MoE, could be developed for a given domain, or the pruned models could undergo SFT to improve their performance, so that collectively the group of specialist models at a given individual memory footprint constraint would be expected to exceed the capabilities of a single general use model at the same footprint at the cost of requiring a larger on-disk memory needed to store all the specialist models. In principle this also allows domain iteration and improvement without risk of regression over the rest of the intelligence stack.

## The 16 GB model set

This is the 16 GB variant of CoE-coder. Three things define it: the tier budget in `config.yaml`, the model tags and domain matrix that use that budget, and the model set below. The runtime code is the same for every variant.

- **Budget.** `vram_tiers.defined.compact` sets `total_vram_gb: 16` and `os_reserved_gb: 0.2`, which leaves 15.8 GB usable. `max_resident_local_models: 1` means one local model is loaded at a time.
- **Models.** Each tag in `clean_resolution.model_refs` and `models.general_fallback` must fit the budget with its KV cache at `num_ctx` 32768.
- **Declared weights.** Each model's weight is declared in `vram_tiers.model_footprints_gb`.

| Domain | Tag | Ollama model ID | Declared weight (GB) | Role | `think` | `raw_prefill` |
|---|---|---|---:|---|---|---|
| python | `python-a` | `JThomas-CoE/coe-qwen3.6-hc-coding_python-r25-19b-a3b-q4_k_m` | 13.19 | Primary (think-off); secondary (think-on) | false | true |
| python | `python-b` | `CoE-Tiel-Coder-python-n1-55-19b-a3b-q4_k_xl` | 13.39 | Secondary (think-off); primary (think-on); `bug_fix` secondary | true | true |
| python | `python-c` | `JThomas-CoE/coe-gemma4-python-hc-14b-a4b-q4_k_m` | 10.40 | `bug_fix` primary | false | false |
| web | `web-a` | `JThomas-CoE/coe-qwen3.6-hc-coding_web-r25-19b-a3b-q4_k_m` | 13.19 | Primary | false | true |
| web | `web-b` | `JThomas-CoE/coe-gemma4-web-hc-14b-a4b-q4_k_m` | 10.40 | Secondary | false | false |
| compiled | `compiled-a` | `JThomas-CoE/coe-gemma4-coding-hc-14b-a4b-q4_k_m` | 10.40 | Primary and secondary (one model) | false | false |
| math | `math-a` | `JThomas-CoE/coe-gemma4-math-hc-14b-a4b-q4_k_m` | 10.40 | Primary (`general_use`) | false | false |
| math | `math-b` | `JThomas-CoE/coe-qwen3.6-mmlu_pro-math-r31-19b-a3b-q4_k_m` | 13.19 | Secondary (`general_use`) | false | true |
| physics | `physics-a` | `JThomas-CoE/coe-qwen3.6-mmlu_pro-physics-r31-19b-a3b-q4_k_m` | 13.19 | Primary (`general_use`) | false | true |
| physics | `physics-b` | `JThomas-CoE/coe-gemma4-physics-mmlu_pro-14b-a4b-q4` | 10.40 | Secondary (`general_use`) | false | false |
| other | `generalist` (`models.general_fallback`) | `gemma4:12b` | 7.38 | `domain_other` catch-all; T5 zero-shot fallback | not set | not set |
| router | `router` (`models.router`) | `qwen3.5:4b` | 3.39 | Domain and task routing; FLM on the NPU when available, otherwise the Ollama fallback | not set | not set |

The largest declared weight is 13.39 GB (`python-b`). That leaves about 2.4 GB of the 15.8 GB usable budget for KV cache and runtime overhead at `num_ctx` 32768. Measured residency is in [LiveCodeBench v6](#livecodebench-v6-footprint-accuracy-and-time).

### Other tiers

A 24, 32, or 48 GB variant would have a different model set, not a setting in this one. It needs a new `total_vram_gb`, new tags in `model_refs` and `models.general_fallback` with their footprints, and a residency check on the target card. No such variant is part of this release, and none has been tested. Raising `total_vram_gb` alone does not create a larger variant: the tags and models are still this set.

## Requirements

- Windows is the reference environment; Python 3.12 was used for the current validation runs. Python 3.10 or newer is required by the source syntax.
- The target configuration is an AMD Ryzen Max+ 395 system with 128 GB shared memory and a software-defined 16 GB inference-memory budget. The NPU router (FLM) is not counted in that budget. Other systems may work but are not validated by this release.
- Ollama version 0.34.0 installed and available on `PATH`. Later versions of Ollama may or may not work.
- The model tags referenced by `config.yaml` pulled or otherwise registered in Ollama. Model weights are not included in this repository; each model retains its own license and usage terms.
- Optional: [FastFlowLM](https://fastflowlm.com/) for NPU-backed routing. The runtime uses the configured Ollama router fallback when FLM is unavailable.
- Optional: authenticated Ollama cloud access if cloud escalation is enabled in `config.yaml`.

The exact model bindings and bounded context settings live in [`config.yaml`](config.yaml). Before first use, review the model IDs and pull the models you intend to use. At minimum, the reference configuration names a local router fallback and a generalist fallback, in addition to specialist models.

## Install

From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install Ollama separately. Pull the model IDs required by the domains you will use, as listed in `config.yaml`. If a model is not available directly from Ollama, download its GGUF file and register it locally; see [Using the pruned models through Ollama](#using-the-pruned-models-through-ollama). If cloud escalation is enabled, authenticate the Ollama installation for its cloud model service before using the runtime. No API key is stored in the shipped configuration.

Clean-install check: in a fresh Python 3.12 venv built from `requirements.txt`, starting the server should return `ok` from `GET /health`.

## Run

Start the interactive CLI (which starts or reuses the API server):

```powershell
python .\CoE-coder.py -cli
```

The default server port is `8001`. To select a different port, use, for example `8003`:

```powershell
python .\CoE-coder.py -port 8003 -cli
```

The API exposes `GET /health` and the OpenAI-compatible `POST /v1/chat/completions` endpoint on the selected local port. The CLI supports routing overrides such as `/domain-python`, `/task-newcode`, think-state selection, artifact attachment, and streaming controls; enter `/help` for its command list. The programmatic surface is documented in the next section; note that the shorthand used in slash commands (eg, `/task-newcode` instead of `/task_type-new_code_generation`) is supported for convenience of use in the CLI.

## Using the API (the programmatic branch)

The CLI and the HTTP API drive the same runtime; this section covers the API. The endpoint is `POST /v1/chat/completions` on the server port (default `8001`), with `GET /health` for readiness. Use plain HTTP rather than an OpenAI client library: the request schema is strict (`extra="forbid"`), so any top-level field not listed below — including the `model` field an OpenAI client sends by default — is rejected with a 422. The client never picks a model; routing is internal and the model that ran is reported in the response.

### A minimal call

```python
import requests

BASE = "http://localhost:8001"

resp = requests.post(f"{BASE}/v1/chat/completions", json={
    "messages": [
        {"role": "user", "content": "Write a Python function that reverses a linked list."},
    ],
})
resp.raise_for_status()
out = resp.json()

print(out["choices"][0]["message"]["content"])
# What actually ran -- domain, task type, harness, model, matrix cell:
print(out["choices"][0]["coe_routing"])
```

The router classifies the prompt and resolves the specialist. The `coe_routing` block reports the decision: `domain`, `task_type`, `harness_id`, `primary_model`, `matrix_cell`, `router_source`, the router's restated `user_intent`, and per-field confidences.

### Overrides

```python
resp = requests.post(f"{BASE}/v1/chat/completions", json={
    "messages": [{"role": "user", "content": "Fix the off-by-one in this loop: ..."}],
    "routing_override": {
        "domain": "domain_python",
        "task_type": "bug_fix",
    },
    "metadata": {"seed": 42},
})
```

(`think` would swap primary and secondary only on cells that define `think_on`/`think_off` branches, such as `domain_python`'s `new_code_generation`; it is inert on `bug_fix`, whose cell is flat.)

Semantics, as implemented in the router:

- Both `domain` and `task_type`: the router is bypassed entirely; the decision is synthesized with confidence 1.0.
- `domain` only: the task type is still classified normally; only the domain is pinned.
- `task_type` only: the decision is synthesized, with the domain defaulting to `domain_other`.
- Validation happens before resolution. A `domain` outside the closed set, or a `task_type` outside the canonical six, returns an `override_invalid` alert (HTTP 200, `{"alert": {...}}`): `Unsupported override: '<bad>'. Allowed values are: [...]`. Overrides are not domain-specific, so the message stays general.
- A valid task type that a domain does not expose (for example `bug_fix` on `domain_math`) is not an error: the domain runs and the harness falls back to `general_use`.
- `user_defined` passes validation, but no domain in this release registers a harness for it, so it returns an alert rather than running.

The closed sets in this release:

| Set | Values |
|---|---|
| `domain` | `domain_python`, `domain_compiled`, `domain_web`, `domain_math`, `domain_physics`, `domain_other` |
| `task_type` | `new_code_generation`, `bug_fix`, `modify_artifact`, `evaluate_artifact`, `general_use`, `user_defined` |

Coding domains expose all five non-`user_defined` task types; `domain_math` and `domain_physics` expose `general_use` only.

`think` scope and use. `think` selects role ordering, not model behavior. A bound specialist's model-card `think` flag governs how that model runs; this override does not change it (the generalist fallback slot is the one exception). What it does select is think-state-dependent role ordering where a domain defines one (`task_cells` with `think_on`/`think_off` branches — `domain_python`'s `new_code_generation` swaps primary and secondary). Use it to pin which model stands primary for an A/B comparison; day-to-day requests do not need it. Precedence: `routing_override.think` > `metadata.think` > `metadata.closed_think` (legacy inverted alias: `closed_think: false` means think-on).

### Request fields and `metadata`

Top-level request fields: `messages` (required), `stream`, `metadata`, `routing_override`, `artifact` (an artifact attachment: `content`, `language`, `origin`, `filename`, `origin_prompt`), `context_attachments`, and `artifact_identity_resolution` (used to answer a clarification). `metadata` is an open dictionary: unknown metadata keys are silently ignored — the strict-schema rule applies to the request body, not to `metadata`.

Keys read by this release:

| Key | Effect |
|---|---|
| `seed` | Per-request sampling seed override. Must be an integer; anything else is a 422. |
| `think` / `closed_think` | Role-ordering selection (see `think` scope above), lower precedence than `routing_override.think`. Does not change a bound specialist's model-card `think` flag. |
| `clear_session` | Clear session memory before this request. |
| `no_inject` | Disable session-memory injection for this request only. |
| `show_thinking` | Streaming only: emit reasoning tokens as visible content. Off by default; hidden reasoning produces a single `status: thinking` tag instead. |
| `checkpoint_interval_s` | Progress-ping cadence for long turns. Precedence: request value > config `progress_interval_s` > 30 s. |
| `cloud_mode` | `"direct"` only: user-initiated direct T3 cloud request, skipping local specialist execution. Requires a configured cloud T3 block in `config.yaml` (`backends.cloud`); `GET /health` reports availability as `modify_pipeline.capability.cloud_order`. Any other value returns the `override_invalid` alert. |
| `harness_variant` | Opt-in A/B harness swap, gated to `modify_artifact` requests resolved to a specialist. The value must name a harness registered in `config.yaml`; anything else is HTTP 400 `invalid_harness_variant`. No A/B variant is registered in this release — note the gate checks the full harness registry, not a variant list, so a registered harness name (e.g. `new_code_generation`) would pass; the key is intended for A/B experiments and is best left unused here. |
| `keep_alive` | Read at the request boundary but not consumed on this release's chat path — model residency is owned by the VRAM manager. |

### Streaming

Set `"stream": true`. The response is SSE in OpenAI chunk format: each event is a `data:` line carrying `{"choices": [{"delta": ...}]}`, terminated by `data: [DONE]`. Answer text arrives in `delta.content`; everything else arrives in `delta.metadata`.

```python
import json

with requests.post(f"{BASE}/v1/chat/completions", json={
    "messages": [{"role": "user", "content": "Draft a FastAPI service with JWT auth."}],
    "stream": True,
}, stream=True) as resp:
    for line in resp.iter_lines():
        if not line.startswith(b"data: "):
            continue
        data = line[6:]
        if data == b"[DONE]":
            break
        chunk = json.loads(data)
        delta = chunk["choices"][0]["delta"]
        if delta.get("content"):
            print(delta["content"], end="", flush=True)
        meta = delta.get("metadata") or {}
        for key in ("coe_routing", "tier_start", "tier_retry", "tier_complete",
                    "checkpoint", "status", "alert", "error"):
            if key in meta:
                print(f"\n[{key}] {json.dumps(meta[key])[:200]}")
```

Metadata event types:

| Event | Meaning |
|---|---|
| `coe_routing` | First event: the routing decision for this turn. |
| `status` | Phase tags: model load, attachment read, `kind: thinking` while reasoning is hidden. |
| `tier_start` / `tier_retry` / `tier_complete` | Ladder progress (multi-tier harnesses; a cloud tier is always announced with its model). |
| `checkpoint` | Progress ping: elapsed seconds, text length, tier, preview. |
| `heartbeat` | 30 s keepalive while the queue is quiet. |
| `coe_artifact`, `selected_patch` | Artifact payloads, emitted before the final content. |
| `alert`, `rejection`, `needs_clarification`, `modify_unavailable` | Terminal side-channels (see below); the stream then ends with `[DONE]`. |
| `error` | Runtime failure; `finish_reason` is `error` on the same chunk. |

Reasoning tokens are never mixed into content unless `show_thinking` is true.

### Response shape (non-streaming)

```json
{
  "id": "coe-clean-...",
  "object": "chat.completion",
  "created": 1760000000,
  "model": "the model that actually ran",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "stop",
    "coe_routing": {"domain": "...", "task_type": "...", "harness_id": "...",
                     "primary_model": "...", "matrix_cell": "...", "router_source": "...",
                     "user_intent": "..."},
    "coe_artifact": null,
    "coe_drift": null,
    "coe_attempts": [{"tier": "T1", "model_id": "...", "passed": true, "error": null, "latency_s": 12.3}]
  }],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
}
```

`usage` is not token-accounted (zeros). `coe_attempts` lists every tier attempt the ladder made, so escalations and retries are visible after the fact. `finish_reason` is `stop` when the turn passed and `error` when it failed.

### Errors and alerts

| Condition | Response |
|---|---|
| Unknown top-level request field, or malformed body | HTTP 422 (strict schema) |
| `metadata.seed` not an integer | HTTP 422 `{"error": "metadata.seed must be an integer"}` |
| Override outside the closed vocabulary | HTTP 200 `{"alert": {"code": "override_invalid", "message": "Unsupported override: '<bad>'. Allowed values are: [...]"}}` |
| Multi-scope prompt rejection | HTTP 200 `{"rejection": {...}}` |
| Ambiguous artifact reference | HTTP 200 `{"needs_clarification": {...}}` — answer the question in the next request, optionally with `artifact_identity_resolution` |
| Modify requested on an artifact with no origin and no cloud configured | HTTP 200 `{"modify_unavailable": {...}}` |
| `harness_variant` outside its gate | HTTP 400 `{"error": "invalid_harness_variant", ...}` |
| Runtime failure | HTTP 503 `{"error": "execution_failed", ...}` |
| Server not initialized | HTTP 503 on any endpoint |

In streaming mode the same payloads arrive as `delta.metadata` chunks (`alert`, `rejection`, `needs_clarification`, `modify_unavailable`) followed by `data: [DONE]`.

### Turn progress and cancellation

Send an `X-Request-Id` header to choose your turn id; otherwise the server mints one. `GET /v1/turn/{request_id}/progress` polls an in-flight or recent turn — streaming or not — with a cursor (`since`/`next_cursor`) and a `since_last_event_s` staleness signal. `POST /v1/cancel` with `{"request_id": ...}` cancels it; a non-streaming turn cancels cooperatively at the next checkpoint. There is deliberately no whole-turn wall-clock cap; per-attempt timers are configured per harness in `config.yaml`.

### Auxiliary endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Backend reachability, session depth, modify capability (`origin_regen`, `cloud_order`), progress cadence. |
| `GET /v1/vram` | Current VRAM state. |
| `POST /v1/session/clear` | Clear session memory — bench runners call this between independent problems. |
| `GET /v1/session/stats`, `GET /v1/session/turns` | Session introspection. |
| `GET /v1/artifact`, `POST /v1/artifact/load` | Inspect and load artifacts (`content`, `language`, `origin`, `origin_prompt`, `filename`). |
| `GET /v1/session/debug-injection` | Show what session memory would inject. |

## HumanEval results (best-of-3)

The 164-task HumanEval set, scored with the official `check(candidate)` tests. best-of-3 is the seed-42 baseline plus two failed-only recovery rounds with fresh seeds; a task counts once if any round passed it. "First try" is the seed-42 baseline alone. This is a progressive retry total, not the unbiased pass@k estimator. Latency is the mean wall-clock seconds per attempt across all rounds, with the stage-1 mean in parentheses.

| Configuration | Model | Think | first try | best-of-3 | Latency s, mean (stage 1) |
|---|---|---|---:|---:|---:|
| Qwen3.6 parent, direct Ollama | qwen3.6-parent-humaneval | Off | 154/164 | 159/164 (97.0%) | 6.76 (6.33) |
| Qwen3.6 pruned r25, direct Ollama | coe-qwen3.6-hc-coding_python-r25-19b-a3b-q4_k_m | Off | 145/164 | 153/164 (93.3%) | 5.61 (5.12) |
| CoE-coder-16GB, Ollama | python-a primary, python-b secondary | Off | 144/164 | 151/164 (92.1%) | 5.69 (5.41) |
| Tiel parent, direct Ollama | tiel-coder-35b-a3b-ud-q4-k-xl | On | 159/164 | 162/164 (98.8%) | 16.72 (15.57) |
| Tiel pruned r55, direct Ollama | CoE-Tiel-Coder-python-n1-55-19b-a3b-q4_k_xl | On | 139/164 | 157/164 (95.7%) | 31.89 (21.29) |
| CoE-coder-16GB, Ollama | python-b primary, python-a secondary | On | 150/164 | 160/164 (97.6%) | 28.92 (25.11) |

Direct rows send the lean, no-prefill Ollama prompt; CoE rows run through the routed `domain_python` / `new_code_generation` path. Each recovery round retries only the prior round's failures. Pass counts are unique QIDs, so retry request counts are not added together. Raw per-run HumanEval outputs are not bundled.

For direct Ollama runs only (parent and pruned naked-prompt rows), any attempt marked `loop_detected` counts as a failure even if the runner's in-request loop retry obtained a passing answer; that task carries to the next outer seed round. The CoE loop guards are integral to that system, so CoE rows retain their ordinary runtime grading and are not reclassified by this direct-run rule. CoE execution may perform configured internal retries or escalation within a request. Progressive totals use a failed-only retry protocol, **not** the conventional model pass@5 estimator that samples five candidates for every task. These are system-level results, not a controlled model-only comparison.

## LiveCodeBench v6: footprint, accuracy, and time

This comparison uses the frozen LiveCodeBench v6 set of 75 questions (25 easy, 25 medium, 25 hard), scored with the upstream LiveCodeBench checker. The LCB runs use seed 42, `num_ctx` 32768, `num_predict` 24576, temperature 0.6, top_p 0.95, top_k 20, repeat_penalty 1.0, and repeat_last_n 1024, as recorded in each run's `metadata.json`. Each row is best-of-3 under the same progressive protocol as HumanEval. Direct model prompts and the CoE harness differ, so these rows compare operating configurations, not pruning in isolation.

### Pruning methodology

The surgery and mask scripts are in the [College of Experts AI research repository](https://github.com/JThomas-CoE/College-of-Experts-AI/tree/main) and are not bundled with this release. The Qwen3.6 and Ornith/Tiel families are 40 layers x 256 routed experts/layer with native top-8 routing, and both are pruned **256 -> 128, i.e. 50% of the routed-expert pool per layer** (a power-of-two GGUF constraint); the gemma4 derived models have a different overall architecture but also take a 50% reduction in the expert pool.

**`rN` is not a retention percentage and not a size fraction.** It is a **swap percentage over rank-0 candidates**: the fraction of the swap-eligible set force-included by evicting the lowest-utility incumbents.

- **Qwen models** - Keep a cosine similarity defined top-128 expert set per layer, subtract the backbone (the cross domain always-kept experts) to get the swap-eligible set, set the swap budget per layer to `round(R_FRAC x |swap-eligible|)`, rank candidates by **rank-0 weight mean**, and inject them in rank order, each time evicting the lowest-cosine non-backbone expert.
- **Ornith/Tiel models** - A simpler approach was tried here. The parent is profiled twice, at native top-8 and at a top-1 reduction, giving two independent utility rankings; each is cut to its **top 128 of 256 (50%)**. Experts in the n1 top-128 but **not** the n8 top-128 form the **n1-only candidate pool**. The **swap ratio r** is the percentage of that pool reranked by descending n1 utility and swapped into the n8 baseline, evicting the corresponding lowest-n8-utility incumbents; the budget stays at exactly 128 per layer. `r` is therefore **a percentage of the candidate pool, not of the 128 slots and not a retention fraction**. The Tiel r55 mask was calibrated on a SWE and Python corpus. Its candidate pool has **973** experts, and **533 of them are swapped in** (r55). The per-layer budget stays at 128, so the mask keeps **5,120 `(layer, expert)` tuples** in total (128 x 40 layers).
- **Tiel is a re-quantisation of Ornith-1.5**. Its base weights come from Ornith-1.5, but its activation profiling and masks are CoE work. The Tiel specialist measured below is the r55 mask described above.
- **Gemma4 models**. Gemma pruned models derive from a 128 routed plus one shared 30 layer architecture and are pruned to 64 routed experts but the essence of the surgery is the same.

**Why the pruned Qwen and Gemma models run think-off (working hypothesis).** The surgery selects experts most active on domain content and reasoning. A side effect may be that the loop-suppression experts, which fire on metacognitive closure signals at the end of a thinking block, have no concentrated domain-specific activation signature and were disproportionately pruned. If so, think-on can loop and exhaust its token budget without emitting an answer. This was observed for the pruned Qwen models, so the release runs them think-off. The effect was less pronounced for the Ornith-1.5/Tiel model, so the pruned Tiel model is measured think-on (see the results below). The hypothesis has not been tested directly. Think mode is set by each model's own `think` flag in `config.yaml`; CoE-coder's think-on setting does not change that flag. It only changes which model fills the python primary and secondary positions: the primary answers first, and the secondary runs only on contract failure.

**Provenance.** All release LCB runs used one runner, one manifest (`lcb75_prior_manifest.json`, sha256 `de48db5c...`) and one problem cache (`_lcb_v6_slim.json`, sha256 `68a45408...`). The per-row figures and the CoE tier split are in the release summary: CoE rows are under `lcb_coe`, and direct model rows are under `lcb_isolated`. Raw per-run result files are not bundled. Model tags: Ornith/Tiel parent `tiel-coder-35b-a3b-ud-q4-k-xl` (21.86 GB); Ornith/Tiel pruned `CoE-Tiel-Coder-python-n1-55-19b-a3b-q4_k_xl` (13.4 GB, r55 mask); Qwen3.6 parent `qwen3.6-parent-humaneval` (about 23 GB, 36.0B); Qwen3.6 pruned `coe-qwen3.6-hc-coding_python-r25-19b-a3b-q4_k_m` (13.2 GB, 19.4B, r25 K128).

`pass@3 local-only` counts a QID as passed only when the local tiers passed it. `best-of-3 incl. T3` also counts passes obtained through the cloud tier (T3, `deepseek-v4.1-flash:cloud` in the shipped config). The local-only figure is the headline. Both use the same 75 QIDs. Latency for CoE rows is split by tier.

| Configuration | Think | best-of-3 local-only | best-of-3 incl. T3 | Latency s per attempt, mean (stage 1) |
|---|---|---:|---:|---:|
| Qwen3.6 parent, direct Ollama | Off | 61/75 (81.3%) | - | 68.38 (53.45) |
| Qwen3.6 pruned r25, direct Ollama | Off | 52/75 (69.3%) | - | 102.76 (57.76) |
| **CoE-coder-16GB, Ollama** | Off | **54/75 (72.0%)** | 57/75 (76.0%) | 109.60 (63.73); local 65.9, cloud 846.7 |
| Tiel parent, direct Ollama | On | 62/75 (82.7%) | - | 227.15 (155.54) |
| Tiel pruned r55, direct Ollama | On | 53/75 (70.7%) | - | 207.12 (160.97) |
| **CoE-coder-16GB, Ollama** | On | **57/75 (76.0%)** | 63/75 (84.0%) | 385.08 (264.15); local 247.7, cloud 1088.0 |

Cloud (T3) was involved in 7 of 125 CoE Qwen primary attempts and 17 of 104 CoE Tiel primary attempts.

Footprint: **CoE-coder resident 13.14 GB** at `num_ctx` 32768 (usable 15.8 GB, headroom 2.66 GB). The **Ornith/Tiel parent is 22.76 GB** at the same context, so it does not fit a 16 GB budget; the CoE configuration does.

**Think-on in the pruned models.** On the release set, the pruned Tiel model keeps most of its think-on accuracy within CoE: best-of-3 is 57/75 against 62/75 for its unpruned parent. The pruned Qwen model runs think-off.

**CoE routing effect.** Routed local-only results beat each pruned model's own direct run: Qwen r25 54 vs 52 (+2, within noise), Tiel r55 57 vs 53 (+4). Neither reaches its unpruned parent in direct runs (61 for Qwen, 62 for Tiel). Neither unpruned parent fits the 16 GB budget as shipped: the Ornith/Tiel parent is 22.76 GB resident at `num_ctx` 32768, and the Qwen parent is 23 GB on disk.

**Latency.** For LCB the CoE-routed Tiel averages 247.7 s per local attempt, against 207 to 227 s for direct Tiel runs. The CoE Qwen row averages 65.9 s local, against 68 to 103 s for the direct Qwen runs. Latency was measured on Ollama 0.34.0 on Windows with ROCm, with one resident model at a time.

Every figure in the two sections above is taken from the release best-of-3 summary bundled at `benchmarks/LiveCodeBench/output/release_pass3_campaign_summary_20261005.json`. The frozen manifest is at `benchmarks/LiveCodeBench/data/lcb75_prior_manifest.json`. Raw per-run output directories are not bundled.

## Default generalist

`models.general_fallback` in [`config.yaml`](config.yaml) is the general-purpose model. It is the model for `domain_other` (the domain with a general-assistant persona). In this release it is `gemma4:12b` (Ollama's default Q4_K_M tag, 11.9B parameters). It was chosen because it accepts image input: its `/api/show` capabilities include `vision`, and an image request at `num_ctx` 32768 succeeded. It runs think-off under the `general_use` harness. A more performant candidate, `qwen3.8-27b-q3_k_xl`, was dropped because its registered tag reports no vision capability and rejected an image request, and adding a vision projector would have pushed the memory constraint; for text-only capability it can be configured at the user's discretion.

Direct Ollama results with the same protocol (seed 42, `num_ctx` 32768), from `benchmarks/LiveCodeBench/output/qwen38_gemma4_pass3_summary_20261005.json`:

| Model | Think | LCB first try | LCB best-of-3 | HumanEval first try | HumanEval best-of-3 | LCB stage-1 mean latency s |
|---|---|---:|---:|---:|---:|---:|
| `gemma4:12b` (shipped generalist) | Off | 48/75 | 59/75 (78.7%) | 157/164 | 162/164 (98.8%) | 283.4 |
| `qwen3.8-27b-q3_k_xl` (not shipped) | Off | 54/75 | 63/75 (84.0%) | 152/164 | 161/164 (98.2%) | 202.7 |
| `qwen3.8-27b-q3_k_xl` (not shipped) | Medium | 54/75 | 62/75 (82.7%) | 161/164 | 162/164 (98.8%) | 358.3 |

On LiveCodeBench, the unshipped Qwen3.8 tag leads `gemma4:12b` by 4 questions best-of-3 and runs about 81 s faster per attempt at stage 1. The generalist was chosen for vision support, not for these scores. On HumanEval the two are level (161 against 162 of 164 for Qwen3.8; 162 for gemma4). The publicly released `gemma4:12b` LiveCodeBench figures are not verified in this release and are not cited here.

Footprint: declared weight 7.38 GB (model layer 7,381,382,048 bytes; the separate 0.175 GB vision projector is not counted in the declared weight). Measured through `/api/ps` at `num_ctx` 32768 after an image request: 8.35 GB resident, all on GPU, leaving about 7.4 GB below the 15.8 GB usable budget. `/api/show` lists `vision` for `gemma4:12b`, so no separate projector download or registration step is needed.

To change the generalist, edit `models.general_fallback` in `config.yaml`: set `model_id`, `num_ctx`, and `vision` (set `true` only for a model registered with a vision projector). Declare its footprint in `model_footprints_gb` (see below).

## Domains and model configuration

The domain matrix is `clean_resolution.domain_rows` in `config.yaml`. The router picks one domain per request. Each domain names a persona and a primary and secondary model reference (`model_refs` tags). Each `task_harnesses` entry maps a task to a harness. Current domains:

| Domain | Persona | Primary / secondary (`model_refs`) | Harnesses and overrides |
|---|---|---|---|
| `domain_python` | `python_coder` | `python-a` / `python-b` | All five. `new_code_generation`: think-off primary `python-a`; think-on primary `python-b`. `bug_fix`: primary `python-c`, secondary `python-b`. |
| `domain_web` | `web_coder` | `web-a` / `web-b` | All five; same pair for each task. |
| `domain_compiled` | `compiled_coder` | `compiled-a` / `compiled-a` | All five; one model fills both roles. |
| `domain_math` | `mathematics_assistant` | `math-a` / `math-b` | `general_use` only. |
| `domain_physics` | `physics_assistant` | `physics-a` / `physics-b` | `general_use` only. |
| `domain_other` | `general_assistant` | `generalist` / `generalist` | `general_use` for code tasks; `modify_harness` for artifact edits; `evaluate_artifact` for evaluation. |

Tag bindings, declared footprints, and the `think` and `raw_prefill` flags are listed in [The 16 GB model set](#the-16-gb-model-set). All tags use `provider: ollama`, `num_ctx` 32768, `num_predict` 24576, `keep_alive` 300, and `vision` true.

The `think` flag belongs to each model and is fixed in `config.yaml`. Think-on and think-off at the CoE level only change the role order in a domain, as described in the pruning section.

### Using any model that fits

Any model can be used for a domain, as long as it fits the memory budget and is configured correctly. Requirements:

- Ollama can run it, registered under the `model_id` that `config.yaml` names (for example `ollama cp <source> <model_id>`, or a Modelfile with `FROM`).
- Only one local model is resident at a time (`max_resident_local_models: 1`). Its weights plus KV cache must fit the usable budget of 15.8 GB (16 GB minus the 0.2 GB OS reserve). The context is bounded by `num_ctx` and the tier's `max_num_ctx` (32768).
- Declare its weight in `model_footprints_gb` under `vram_tiers`, using the `model_id` with or without `:latest`. Undeclared models fall back to an estimate from the parameter count, so declare every live model.
- At load time the runtime checks the declared weight against the budget, then checks Ollama's reported `size_vram` after load. A model that exceeds the budget is evicted and the request fails.
- Check residency with `/api/ps`. When `size` equals `size_vram`, the model is fully on GPU.
- To fit a larger model, reduce its `num_ctx`. The KV cache grows with context length.

Worked example: the shipped generalist `gemma4:12b` has a declared weight of 7.38 GB. At `num_ctx` 32768 with an image request, `/api/ps` reports 8.35 GB resident. The remaining headroom is KV cache plus runtime overhead. It fits within the 15.8 GB usable budget.

### Adding a domain

1. **Models.** Add tags under `clean_resolution.model_refs`, for example `legal-a: { model_id: "<model-id>", provider: ollama, num_ctx: 32768, num_predict: 24576, keep_alive: 300, think: false, raw_prefill: false, vision: false }`. Register each `model_id` in Ollama and declare its footprint in `model_footprints_gb`.
2. **Persona.** Add a role text under `clean_resolution.personas`, for example `legal_assistant: "You are a legal information assistant..."`.
3. **Domain row.** Add `clean_resolution.domain_rows.domain_<name>` with `persona`, `description`, `include_topics`, `exclude_topics`, `primary_model_ref`, `secondary_model_ref`, and `task_harnesses`. Coding domains expose all five harnesses. Non-coding domains expose `modify_artifact`, `evaluate_artifact`, and `general_use`. Optional `task_cells` set a different pair per task and per think state.
4. **Restart the server after any `config.yaml` change** (domain, model binding, or harness). The config is read once at startup — the domain matrix, the router's allowed-domain list, and the override-validation vocabulary are all built then and not refreshed. Until the restart, a new domain is not routable and an override naming it returns the `override_invalid` alert. Changes to `CoE-coder.py` need no server restart; the launcher/CLI starts fresh each run.

To use a different generalist, edit `models.general_fallback` as described above.

## Using the pruned models through Ollama

The pruned models are published on Hugging Face under `JThomas-CoE`. The Tiel model used by `python-b` is the vision-capable build (`vision: true` in `config.yaml`), so it needs two files from [`JThomas-CoE/coe-tiel-coder-python-n1-55-19b-a3b-q4_k_xl`](https://huggingface.co/JThomas-CoE/coe-tiel-coder-python-n1-55-19b-a3b-q4_k_xl): `CoE-Tiel-Coder-python-n1-55-19b-a3b-q4_k_xl.gguf` and `mmproj-BF16.gguf`. Download both into one folder, then create a `Modelfile` beside them:

```text
FROM .\CoE-Tiel-Coder-python-n1-55-19b-a3b-q4_k_xl.gguf
FROM .\mmproj-BF16.gguf

TEMPLATE {{ .Prompt }}
PARAMETER temperature 0.6
PARAMETER top_k 64
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.05
PARAMETER repeat_last_n 1024
PARAMETER num_ctx 32768
PARAMETER num_predict -1
PARAMETER num_gpu 99
```

Then create the local tag the configuration expects (Ollama 0.34 or later, for the projector):

```powershell
ollama create CoE-Tiel-Coder-python-n1-55-19b-a3b-q4_k_xl -f .\Modelfile
```

A single `FROM hf.co/...` pull loads the language model without the vision projector.

The Qwen3.6 r25 specialist is `hf.co/JThomas-CoE/coe-qwen3.6-hc-coding_python-r25-19b-a3b-q4_k_m`.

## Benchmark reproduction (LiveCodeBench v6)

The LiveCodeBench problems come from the public Hugging Face dataset `livecodebench/code_generation_lite`, release `release_v6`: https://huggingface.co/datasets/livecodebench/code_generation_lite

The 75-question release set uses these files:

| File | Purpose | SHA-256 |
|---|---|---|
| `benchmarks/LiveCodeBench/data/lcb75_prior_manifest.json` | 75 question IDs in run order, with tiers (25 easy, 25 medium, 25 hard) and the pins below | `de48db5ca0654b964c68a3dfc17122d4b184b1da79d9005aef3745c8466c5fde` |
| `_lcb_v6_slim.json` | Problem cache expected at the workspace root (about 650 MB) | `68a454088074f666e078e87414d9074630e141b6c31325b4dbf68b5258da7e50` |

Verify the cache after obtaining it:

```powershell
Get-FileHash -Algorithm SHA256 .\_lcb_v6_slim.json
```

The output must equal `68a454088074f666e078e87414d9074630e141b6c31325b4dbf68b5258da7e50`. A different hash means the file is not the pinned cache, and its results will not match the release figures.

The cache is a 75-problem subset of the dataset above and is not bundled. The runner accepts a cache whose byte SHA-256 matches the pin above, or whose content digest matches `problem_cache_content_sha256` (`4c523bcfb9b608de469e9a84cf5f3a1fc51e914ccb717b662de6578e30a05672`) in the manifest. It refuses to run otherwise.

### Rebuilding the cache from the dataset

To rebuild the cache from the dataset, run `benchmarks/build_lcb_cache.py` with network access in an environment that has `datasets` 3.6.0, `pebble` 5.2.3, and `tqdm` 4.70.1. `datasets` 3.6.0 is required because the LiveCodeBench loader uses a dataset script that `datasets` 4.x no longer runs. The script keeps the 75 manifest QIDs in manifest order, checks the content digest against `problem_cache_content_sha256`, and exits with an error on mismatch. Its byte SHA-256 is printed for information only, because a rebuilt file usually differs in byte order from the pinned cache.

## Runners

Examples (run from the repository root; `--help` lists every option):

```powershell
# HumanEval, direct Ollama model
python .\run_humaneval_flexible_v1.py --backend ollama --model qwen3.8-27b-q3_k_xl:latest --think off --num-ctx 32768 --seed 42 --out humaneval_qwen38.jsonl

# LiveCodeBench, direct Ollama model
python .\benchmarks\run_lcb_canonical.py --backend ollama --model qwen3.8-27b-q3_k_xl:latest --think off --manifest .\benchmarks\LiveCodeBench\data\lcb75_prior_manifest.json --out <output-path>

# LiveCodeBench, through a running CoE-coder server (default port 8001)
python .\benchmarks\run_lcb_canonical.py --backend coe --coe-url http://127.0.0.1:8001 --manifest .\benchmarks\LiveCodeBench\data\lcb75_prior_manifest.json --out <output-path>
```

For HumanEval through CoE routing, use `--backend openai` against the local server with `--coe-domain` and `--coe-task-type`; check `--help` for the accepted values.

- `benchmarks/run_lcb_canonical.py` is the LiveCodeBench runner. It reads `benchmarks/LiveCodeBench/data/lcb75_prior_manifest.json` and the problem cache at the workspace root. It requires `requests` and the bundled `benchmarks/LiveCodeBench/lcb_runner` package. The LCB dependencies (`datasets` 3.6.0, `pebble` 5.2.3, `tqdm` 4.70.1) are pinned in `requirements.txt`. `--self-test` reads the problem cache, so run it only after the cache is in place (obtained or rebuilt); otherwise it fails with `FileNotFoundError` on `_lcb_v6_slim.json`.
- `benchmarks/build_lcb_cache.py` rebuilds the problem cache from the dataset (see [Rebuilding the cache](#rebuilding-the-cache-from-the-dataset)).
- `_run_lcb_naked.py` provides the prompt builder used by the runner.
- `run_humaneval_flexible_v1.py` is the HumanEval runner.

## Configuration and data

- [`config.yaml`](config.yaml) is the single source of truth for model bindings, domain/task routing, timeouts, VRAM limits, and optional cloud escalation.
- [`prefill_rates.json`](prefill_rates.json) supplies model-specific prefill timing calibration.
- `artifacts_saved/`, `evaluate_objects/`, runtime logs, checkpoints, and benchmark result files are generated locally. They are not part of the source release and are excluded by `.gitignore`.
- The HumanEval task file (164 tasks) is bundled at `benchmarks/data/HumanEval.jsonl`, SHA-256 `1d49078ba3e2b196b9344535bef34a43021f038fad9561d6ee7c53450609a6a2`. `run_humaneval_flexible_v1.py` reads it from there. The LiveCodeBench problem cache is not bundled; see [Benchmark reproduction](#benchmark-reproduction-livecodebench-v6).

## Known limits

- This release does not bundle models, Ollama, or FLM. Model quality and availability depend on the selected model IDs and their separate terms.
- The one-local-model residency policy and configured context limits are intended to keep execution within the 16 GB budget; other VRAM capacities and hardware backends need separate validation.
- Cloud escalation is optional, but an enabled cloud target requires the corresponding Ollama cloud access. Disable `backends.cloud.enabled` in `config.yaml` when cloud use is not desired.
- The HumanEval and LiveCodeBench figures are single-run best-of-3 totals (progressive retry) and can vary with model versions, backend versions, and runtime configuration. Differences of a few questions are within run-to-run noise.
- The LiveCodeBench harness is bundled from a locally modified checkout of upstream commit 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24, not the unmodified upstream tree. The problem cache must be obtained as described in [Benchmark reproduction](#benchmark-reproduction-livecodebench-v6).
- This release should be considered a beta release suitable for research and experimentation only and is intended as a proof of principle that significantly (ie, by 50% in this case) pruned MoE models can serve as reasonably performant specialist models within their designated domain when RAM footprint is constrained. Post-surgery retraining to recover or exceed the parent's performance in the designated domain has not been demonstrated yet and is a planned next step.

## Project and licensing

This standalone repository packages the CoE-coder runtime; the [College of Experts AI research repository](https://github.com/JThomas-CoE/College-of-Experts-AI/tree/main) documents model creation, activation profiling, and expert pruning. All College of Experts work is offered under the [PolyForm Noncommercial License 1.0.0](LICENSE), and commercial use requires a separate commercial license (see [`LICENSE-COMMERCIAL.md`](LICENSE-COMMERCIAL.md)). That covers the runtime code and configuration, activation data and utility rankings, expert-selection criteria and pruning masks, pathology guards, and the CoE escalation ladders. Base models, datasets, and third-party components keep their own terms, listed below.

**Scope of the PolyForm licence.** The PolyForm licence covers all College of Experts work, not only code. That includes the activation data, the utility ranking and other expert-selection criteria, the pruning masks, the pathology guards, and the CoE escalation ladders. The surgery code cannot run without its mask, and the mask cannot be produced without the activation data and ranking code. Together these form one code object that produces a pruned model from an MIT-licensed parent, so the resulting model is CoE work under PolyForm. The parent's own components keep their MIT terms. The Qwen3.6-35B-A3B base model is Apache-2.0 according to its Hugging Face model card. The default generalist `gemma4:12b` is pulled from the Ollama library and is Apache-2.0. The Tiel weights are a CoE work product: the parent's tensors carry Ornith-1.5's MIT licence in detail, while the set of weights, the masks, and the selection criteria are CoE work under PolyForm. The bundled LiveCodeBench harness is licensed under [`benchmarks/LiveCodeBench/LICENSE`](benchmarks/LiveCodeBench/LICENSE). Other base models and third-party dependencies have their own licenses and terms.
