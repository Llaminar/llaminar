# Frozen native speculation collection

`scripts/ci/collect_native_speculation.py` fills the collection seam of the
[native evaluator](native-speculation-evaluation.md). It runs existing native
AR, stock-MTP and candidate benchmarks; it implements no decoding mechanism.
The canonical benchmark runner still owns workload arguments, measured/warmup
iteration settings, the exact-cell watchdog and local process retirement.

This version supports **local executables and saved native apply documents**.
It does not support image collection, automatic placement search, cross-host
launch, continuous serving or request batching. Unsupported inputs fail; there
is no substitute execution policy, fixture/model fetch, retry or gate bypass.

## Independent setup authority

An independent evaluator must own the frozen plan, its external digest, the
collector code, acquisition/build provenance, environment and output storage.
Do not give candidate code write access to those artifacts or held-out prompts.
A hash is not a sandbox or signature. The adapter checks current files against
trusted declarations; it cannot prove an evaluator's honesty, prevent an
external profiler from attaching, or authenticate a declared runtime-library
closure merely by hashing its metadata.

Begin with the existing evaluator plan and externally retained plan SHA-256.
Its build metadata and resolved native benchmark configurations remain exactly
the evaluator's schema. The collector adds these requirements to the deployment
artifacts already referenced by that plan:

### Model

The model artifact must contain `files` (the complete, unique ordered list of
canonical absolute model/shard paths, primary first) and `stat_identity` (the
map returned by `production_artifacts.model_identities` for that inventory).
Additional acquisition/content provenance may remain in the document.

The adapter stats existing files. It **never opens model payloads for hashing,
downloads weights or infers a shard manifest from filenames**. File tuples
bind device/inode/size/mtime/ctime, not immutable weight content. The trusted
canonical acquisition/inventory authority must supply the complete list and
any required content identities. A matching size alone is not admission.

### Hardware

Use the canonical `run_model_parity_benchmarks.hardware_identity()` result.
This is passive CPU/PCI topology metadata, not per-GPU UUID, clock, thermal or
utilization telemetry. Freeze additionally records the collector's CPU
scheduling affinity. Stable metadata does not guarantee an idle device or
identical clocks. The evaluator remains responsible for an exclusive, matched
timing window.

### Execution

The execution artifact has exactly `schema`, `environment` and `roles`:

```json
{
  "schema": "llaminar.native-speculation-local-execution.v1",
  "environment": {"PATH": "/usr/bin:/bin"},
  "roles": {
    "ar": {
      "binary": "/builds/stock-release/llaminar2",
      "configuration": {"path": "ar-apply.json", "sha256": "<file digest>"},
      "preflight_build_directory": "/builds/stock-integration",
      "prerequisites": {"path": "stock-prerequisites.json", "sha256": "<file digest>"}
    },
    "stock": {
      "binary": "/builds/stock-release/llaminar2",
      "configuration": {"path": "stock-apply.json", "sha256": "<file digest>"},
      "preflight_build_directory": "/builds/stock-integration",
      "prerequisites": {"path": "stock-prerequisites.json", "sha256": "<file digest>"}
    },
    "candidate": {
      "binary": "/builds/candidate-release/llaminar2",
      "configuration": {"path": "candidate-apply.json", "sha256": "<file digest>"},
      "preflight_build_directory": "/builds/candidate-integration",
      "prerequisites": {"path": "candidate-prerequisites.json", "sha256": "<file digest>"}
    }
  }
}
```

This is a schema illustration, not a runnable deployment. Replace every path
and digest with actual admitted artifacts. References resolve relative to the
execution document. The binaries must be executable regular files next to
Release `CMakeCache.txt` files and match the evaluator's binary SHA-256 and ISA.
Each Release and prerequisite build must belong to the same clean source
commit/checkout, with matching ISA, enabled backends and compiler selections.

Prerequisites are existing complete canonical Unit/ProductionTestPreflight
receipts, not hand-authored pass flags. The adapter uses the existing source
freshness and unchanged-inventory validators; it does not synthesize receipts,
rebuild tests, run gates or waive missing hardware. A receipt for another
checkout, stale source/build/inventory, or another backend/ISA is rejected.
These are local receipt reuse requirements, not a new qualification policy.

Configurations must be complete **lossless native JSON apply documents** from
`OrchestrationConfigDocument`, not hand-maintained Python placement records.
Native parsing/admission remains authoritative. The collector requires explicit
`Apply`, no hostfile, ordinary MPI bootstrap, nondeterministic optimized kernel
dispatch, the declared primary model and single-request MTP capacity. Here
“nondeterministic” means `deterministic: false` in the diagnostic-dispatch setting;
seeded request output still must reproduce AR exactly. AR must explicitly disable
MTP, stock/candidate must explicitly enable it. All saved fields other than
`mtp` and `mtp_activation_policy` must match, preserving physical placement,
precision, context, prefix and movement policy.

`environment` is the **complete child environment**, not an overlay on the
collector's shell. Include required runtime/library/affinity settings explicitly;
never include credentials or secrets. Profiling, diagnostic deterministic,
external-MPI, preload and benchmark-iteration overrides are rejected. Workload
iteration settings come only from each frozen request and must fit the native
1..100 environment limit. The adapter does not certify that every possible
instrumentation mechanism is absent; protect the collector and timing window.

## Freeze, then collect

Both output directories must be new and outside every source checkout. Keep
results outside source control. No model download occurs in either phase.

```bash
python3 scripts/ci/collect_native_speculation.py freeze \
  --plan /evaluation/plan.json --plan-sha256 "$FROZEN_PLAN_SHA256" \
  --output /evaluation/frozen-001
```

Freeze validates existing local setup, receipt freshness and input identities;
it launches **no benchmark**. Retain its printed collection SHA-256 outside
candidate write authority. It records the fixed role schedule and stat guards
for metadata, model files, executables and generated build boundaries.

```bash
python3 scripts/ci/collect_native_speculation.py collect \
  --collection /evaluation/frozen-001/collection.json \
  --collection-sha256 "$FROZEN_COLLECTION_SHA256" \
  --output /evaluation/collection-001
```

Collection executes the entire planned request/pair/role cross product once,
rotating role order deterministically and keeping held-out requests last.
There is no selector, resume, retry, seed search or shorter-output mode. Each
run uses the canonical benchmark frontend with `--config` plus the frozen
prompt, sampler, output budget and warmup/measurement counts. Binary hashes and
canonical receipt checks bracket the collection; input stat guards bracket
each launch. Current CPU affinity is also checked before every launch.

Artifacts:

- `status.json`: collecting/failed/complete, planned count, completed observations,
  active run on failure, failure detail and collector wall time;
- `observation-NNNN/`: canonical invocation, raw log and native `benchmark.json`;
- `evidence.json`: emitted only when all planned observations pass the actual
  evaluator, with automatically bound plan/deployment/build/artifact identities;
- `comparison.json`: the evaluator's diagnostic comparison, never a certificate.

Benchmark references in completed evidence are relative to the collection
folder. Preserve all observation files when relocating it. Frozen setup still
names the original local deployment paths and requires the original pinned
plan. Failed collections retain progress and raw artifacts, not a successful
`evidence.json`; a rejected final evaluation may retain `evidence.pending.json`
for diagnosis. Native failure, interruption, short output, token drift, missing
JSON or changed identity stops immediately without trying another policy.

Exit zero means the diagnostic collection is complete, even for a slower
candidate. Exact tokens do not prove serial-row logits, KV/GDN bytes, production
capture, seeded-distribution correctness or end-to-end serving performance.
Every stronger production gate from the evaluator documentation remains intact.

## Model-free contracts

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tests/v2/unit/scripts/test_native_speculation_collection.py
ctest --test-dir build_v2_integration -FA '.*' --output-on-failure \
  -R '^V2_(Unit_NativeSpeculationCollection|Integration_NativeSpeculationCollectionContract)$'
df -h /
```

The tests exercise actual collector and launcher code against **mocked process,
output and prerequisite-admission fixtures**. Real binaries, models, workers,
backend servers and inference are not observed. Retirement is mocked too: no
kill operation or destructive container entrypoint is executed by these tests.
The focused collection contract is registered in ProductionTestPreflight.
