# Native speculation evaluation

`scripts/ci/evaluate_native_speculation.py` applies the cost-guided part of the
SpecPilot methodology to **existing native AR and MTP paths**. It is an offline,
model-free consumer of `llaminar.benchmark.v1` JSON, not a new runner, optimizer,
tree verifier, controller, or certification path.

The first implementation deliberately provides:

- separately named AR, current-MTP (`stock`), and candidate comparisons;
- externally pinned evaluation intent and content-hashed input artifacts;
- exact output-token agreement, including repeated seeded sampling;
- complete paired-block coverage with disjoint development/held-out prompts;
- whole-decode cost per useful emitted token and amortized verifier cost;
- separately reported, **nonadditive** PerfStats timing observations;
- explicit unknowns instead of fabricated acceptance/kernel headroom.

It does **not** prove serial-row logits or state bytes, authenticate a collector,
seal a filesystem, run correctness suites, calculate statistical significance,
automatically promote changes, or certify a model/image. Exit zero means a
complete diagnostic comparison, even when the candidate is slower.

## Ownership and prerequisites

An independent evaluator must own the evaluator code, plan, pinned plan digest,
held-out requests, and artifact collection. Keep these outside candidate write
permissions (a separate account/container or CI authority). Read-only flags and
hashes stored beside candidate-writable data are not an isolation boundary.
This tool detects changed bytes relative to a **trusted** digest; it cannot
detect fabricated timings or provenance from a dishonest collector.

Use the existing procedures, not another model/topology list:

1. Select actual supported cells/configuration through the canonical planning
   and test authorities. Start with production defaults and retain the resolved
   deployment. Explicitly freeze any subsequent policy override.
2. Run the canonical Unit/`ProductionTestPreflight` prerequisites before model
   admission. Retain their build-bound receipts. See
   [the testing skill](../.agents/llaminar-testing/SKILL.md).
3. Use the existing Release `llaminar2 benchmark` frontend, or
   `scripts/ci/run_model_parity_benchmarks.py` where its canonical inventory and
   production-default policy match the experiment. Do not modify that runner's
   certification policy to accommodate a diagnostic.
4. Keep model shards, tokenizer, weight format, hardware, resolved placement,
   context, prefix policy, movement policy, compiler/runtime, request bytes and
   sampling matched. AR and stock use the identical build. Candidate source and
   binary may differ; compiler, CPU ISA and runtime libraries may not in this
   comparison version.
5. Retain the native `benchmark.json` from each run. Do not retype rounded log
   rates, reuse one run as several blocks, or use profiler-instrumented timings.

The input schema currently supports one logical request at a time. It excludes
one terminal-prefill output token from the decode numerator. Request batching,
continuous serving, state-byte comparison and transaction-trace headroom need
their own observation contracts; they are not inferred from existing counters.

## Freeze a plan

The plan has schema `llaminar.native-speculation-plan.v1` and exactly these
fields:

| Field | Contents |
|---|---|
| `deployment` | `model`, `hardware`, `execution`, each a JSON artifact reference |
| `implementations` | Exactly `ar`, `stock`, `candidate`, each containing `build` and `config` |
| `requests` | Nonempty list of `{id, split, workload}` |
| `pairs` | At least three unique block IDs, for example `["block-1", "block-2", "block-3"]` |

Every artifact reference is `{"path": "relative/file.json", "sha256": "<64 lowercase hex digits>"}`.
Paths resolve relative to the document containing the reference, not the shell
working directory. The digest covers the exact file bytes, including whitespace.
There are no defaults for missing identities.

Deployment documents are evaluator-owned, nonempty JSON objects. The tool pins
their bytes; it does not reinterpret them as a new runtime configuration:

- `model`: the complete canonical model-file inventory and immutable content
  identities, including shards, format, and tokenizer provenance;
- `hardware`: the collector's actual physical device/topology identities;
- `execution`: the saved resolved apply document plus relevant environment,
  affinity, worker geometry and production-path policy.

Collection must verify these declared identities against the actual run. They
are **not runtime attestations merely because the evaluator hashed them**.
Do not hash or download multi-gigabyte models just to inspect this tool; use the
trusted acquisition/build authority's existing immutable identity records.

A build artifact contains at least:

```json
{
  "build_type": "Release",
  "source_revision": "<40 lowercase Git commit hex digits>",
  "binary_sha256": "<64 lowercase hex digits>",
  "cpu_isa": "AVX512",
  "compiler": {"cxx": "<complete compiler identity and relevant flags>"},
  "runtime_libraries": {"closure": "<authenticated native library identities>"}
}
```

`AVX2` is also supported. Extra build provenance is retained through the artifact
hash. Both AR and stock must reference byte-identical build metadata; candidate
metadata must have the same `compiler`, `cpu_isa` and `runtime_libraries` values.
Metadata comparisons retain JSON types: a boolean is not the same compiler or
execution-policy value as an integer. This is evidence binding, not independent
proof that a binary matches a Git SHA.

For each role's `config`, copy the complete native benchmark JSON `config`,
removing only `model_path`, `benchmark_json_output_path`, `sampling`, and
`n_predict`. The last two come from each frozen workload; paths are covered by
the deployment/collector identities. Include every remaining resolved field,
including future native fields: omissions or additions at evaluation fail.
Only `mtp_*` settings may differ between role configurations. `mtp_enabled`
must be false for AR and true for stock/candidate. `benchmark_mode` must be true.
A setup/discovery observation can supply this configuration; it is not a timed
held-out result.

Each request uses the existing canonical workload schema, for example:

```json
{
  "id": "held-out-01",
  "split": "held_out",
  "workload": {
    "schema": 1,
    "prompt_unit": "A previously unseen, evaluator-owned request.",
    "prompt_repetitions": 1,
    "decode_tokens": 256,
    "warmup_iterations": 1,
    "measurement_iterations": 3,
    "seed": 42,
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.95
  }
}
```

Both `development` and `held_out` splits are required. Their final prompt bytes
must be disjoint, even if seeds or prompt-unit factorization differ. Repeated
identical request workloads under new names are rejected; different seeds in
one split are permitted. A full output budget is required; early EOS is not a
shorter successful timing sample. Do not suppress EOS or search for passing
seeds inside evaluation. Request counts, seeded sampler integers and token IDs
must fit the native signed 32-bit representation; counters and timer totals must
fit unsigned 64-bit storage. Arbitrarily wide JSON integers are not native evidence.

Freeze the candidate implementation and plan before collecting held-out data.
Keep the plan SHA-256 in independent evaluator/CI state, not just in the plan's
directory. Then inspect it without loading a model:

```bash
python3 scripts/ci/evaluate_native_speculation.py inspect \
  --plan /evaluation/plan.json --plan-sha256 "$FROZEN_PLAN_SHA256"
```

The output supplies `deployment_sha256` and each role's `build_sha256` for the
collector. Recomputing `FROZEN_PLAN_SHA256` after candidate changes defeats the
freeze. This tool cannot police evaluator permissions or prevent repeated
adaptive use of a spent holdout; that remains an external workflow obligation.
The Python `FrozenPlan` retains immutable storage and returns independent views;
editing a returned request/configuration/build map cannot alter its validated
intent while retaining the original digest.

## Collect and compare

For a local admitted deployment, the [frozen collection adapter](native-speculation-collection.md)
uses saved native apply documents and the canonical launcher to produce the
complete hash-bound evidence below automatically. It validates existing
prerequisite receipts and never downloads/stages models or bypasses gates.
The manual evidence schema remains useful for independent collectors.

Collect all three roles for every request and paired block, balancing or
rotating their run order. Keep device load and timing boundaries matched.
Each native run performs the frozen warmups and at least three measured
iterations. Blocks are separate collections, not renamed native iterations.
Within-run repeats are reduced to a median before forming paired ratios;
they are not presented as independent statistical replications.

Record an evidence document:

```json
{
  "schema": "llaminar.native-speculation-evidence.v1",
  "plan_sha256": "<externally frozen plan digest>",
  "observations": [
    {
      "request": "held-out-01",
      "pair": "block-1",
      "role": "candidate",
      "deployment_sha256": "<inspect output>",
      "build_sha256": "<candidate build digest from inspect>",
      "measurement": "unprofiled",
      "benchmark": {"path": "held-out-01/block-1/candidate/benchmark.json", "sha256": "<file digest>"}
    }
  ]
}
```

The example observation is only a template; an actual input must contain the
entire frozen cross product. The collector, not the candidate, vouches for the
bindings and `unprofiled` assertion. Failed/missing runs cannot be omitted and
still yield a comparison. Keep failed collection logs alongside the evidence;
never retry a different execution policy to make a block pass.

```bash
python3 scripts/ci/evaluate_native_speculation.py evaluate \
  --plan /evaluation/plan.json --plan-sha256 "$FROZEN_PLAN_SHA256" \
  --evidence /evaluation/evidence.json \
  --report /evaluation/comparison-001.json
```

The report path must be new; existing files are never overwritten. Invalid
input exits nonzero and publishes no successful report. Retain these local
artifacts outside source control. `status: compared` and exit zero are **not**
correctness certification, a statistical win, or approval to install a policy.

## Interpret costs without inventing headroom

- `stock_vs_ar`, `candidate_vs_ar`, and `candidate_vs_stock` are separate ratios;
  greater than one means the numerator baseline took longer.
- Decode ratios use the median native decode wall time in each block. Prefill
  and combined prefill-plus-decode ratios expose shifted costs. Combined time
  excludes model startup, HTTP queuing and other unobserved serving overhead.
- `decode_ms_per_emitted_token` uses summed decode wall time divided by actual
  decode outputs **after** the terminal-prefill token. It does not divide by
  proposed drafts, padded rows, accepted-state rows, or node capacity.
- `whole_decode_ms_per_verifier_invocation` amortizes all measured decode work
  over actual verifier calls. It is not individual transaction latency and can
  include ordinary decode. `emitted_tokens_per_verifier_invocation` likewise
  includes all useful decode outputs, not just accepted draft tokens.
- `accepted_fraction` uses accepted/(accepted+rejected) counters. No proposals
  means unknown (`null`), not perfect acceptance or a zero-cost opportunity.
- The ten largest native timers remain separately identified by domain, phase,
  device and tags. Parent/child and cross-device timers can overlap. Do not add
  them or subtract their sum from wall time to obtain an overhead budget.
- Request-level block medians/ranges and separate development/held-out geometric
  means are descriptive only. They are not confidence intervals, a noise model,
  or a paper-equivalent two-GPU swapped-role statistical evaluator.

The report always marks acceptance and kernel/overhead headroom unobserved:
live per-transaction proposal/commit traces, disjoint critical-path timing, and
measured bytes/bandwidth are not supplied by this contract. Likewise exact
emitted IDs do not prove logits, KV/GDN state, complete captured execution or
sampling distribution correctness. Existing numerical, replay, graph,
collective, prefix, stochastic and production-preflight gates still apply
before any code or policy can be qualified.

## Model-free regressions

```bash
python3 tests/v2/unit/scripts/test_native_speculation_evaluation.py
ctest --test-dir build_v2_integration --output-on-failure -FA '.*' \
  -R '^V2_(Unit_NativeSpeculationEvaluation|Integration_NativeSpeculationEvidenceContract)$'
```

The focused evidence contract is explicitly registered in
`ProductionTestPreflight`. Fixtures are synthetic JSON: these are contract
tests, not recorded inference replay, real-model correctness or speedup proof.
