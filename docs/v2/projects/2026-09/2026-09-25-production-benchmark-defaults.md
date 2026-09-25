# Canonical benchmark policy: production defaults, not correctness stress

## Why the CPU results disagreed

The first post-driver-upgrade standard benchmark measured Qwen 3.6 MoE 35B
at 16.816 tok/s on two CPU sockets. The preceding tuning checkpoint recorded
approximately 41–42 tok/s. The saved artifacts prove identical model path,
512-token prompt bytes, 256-token output length and terminal token stream;
both are unprofiled Release AVX512 runs. They did **not** use the same execution
policy.

| Per measured request | Tuning/default policy | Former canonical benchmark |
|---|---:|---:|
| Final adaptive depth | 4 | 14 |
| Draft steps | 224 | 811 |
| Accepted draft tokens | 195 | 196 |
| Grouped verifier rows | 284 | 870 |
| Completed expert moves, three requests | 10 / 20 / 20 | 140 / 160 / 160 |
| Initial requested depth | Production default | 15 |
| Depth observation window | Production default (16) | 1 |
| Verifier policy | Production default (greedy) | Speculative sampling |

The runner directly reused `e2e.server_args`, including maximum-depth
initialization and frequent forced movement. The observed depth controller's
last decision was `budget_limited`; a shortened transaction is deliberately
not treated as a full-depth efficiency sample. Regardless of how much each
override contributes individually, timing that combination does not measure
ordinary production-default service. The user requested that benchmarks stop
using these artificial overrides.

The old run was interrupted after eight of fourteen selected cells completed;
it is not a passing complete suite. Its original JSON, per-cell logs and clean
driver receipt remain intact. One three-tier cell was 10.95% below its old
release decode score and was flagged before interruption, not erased by the
policy correction. Last-release values cannot be relabeled as measurements of
the corrected policy.

## Authority and implementation

`ModelParityRuntimeExport.h` now exports a separate versioned `benchmark`
projection for every E2E-tagged case. The canonical model and topology remain
single-source. HTTP and benchmark projections share automatic placement
constraints, but only correctness retains short-trace stress policy.

The benchmark projection retains context, activation/KV precision, backend
counts, strategy, owner order and selected MTP/movement modes. It omits depth
initialization/window overrides, graph-capacity overrides, forced economics,
transfer-slot quotas, prefill segment overrides and collectives/cache tuning.
The live production runtime resolves these defaults. It does not remove flags
from HTTP argv through a blacklist or maintain another model/topology table.

The Python runner requires the typed projection and rejects stale exports
before staging or device work. Workload remains 512 repetitions of ` test`,
256 output tokens, seed 42, temperature zero, one warmup and three measured
requests; each phase uses its median, never a best retry. High-water identities
include the new policy, preserving old marks as a distinct historical series.

Explicit diagnostic image runs may measure an older immutable implementation
against current policy. Evidence separately records the matrix source revision
and runtime image revision. Same-revision/full-E2E admission is unchanged for
certification; diagnostic runs remain non-certifying.

## Functional evidence

- The two Python policy regressions first failed against the former runner.
- All **216 pipeline-harness tests** pass after the fix.
- All **671 Unit registrations** pass in **74.33 seconds**.
- The complete model-definition unit, automatic E2E projection, benchmark
  runtime identity and new `V2_Integration_BenchmarkProductionDefaults`
  registrations pass **4/4** in **1.26 seconds**. Both benchmark regressions
  are explicitly part of `ProductionTestPreflight`.
- All **19** exported E2E/runtime/model records are exactly unchanged except
  for the new benchmark field. No correctness workload was weakened.
- The inference binary is unchanged from the preceding complete **671 Unit +
  392 preflight** transaction and the **11/11 ROCm-containing HTTP cells**.
  Those device checks remain amortized; only benchmark harness/metadata/docs
  changed afterward.

Evidence is under `/tmp/llaminar-driver-upgrade.Qq9WtW/`. In particular,
`benchmark-vs-release.json` preserves the interrupted stress-policy results,
`production-defaults-manifest.json` contains the corrected 19-cell inventory,
and `benchmark-defaults-*` records the producer/consumer regression gates.
Timing results must be reported separately for the former stress series and
the new production-default series. No checked-in high-water values have been
changed during this local diagnostic work.

## Corrected current-build measurements

With the unchanged current Release binary, the new canonical Qwen 3.6 CPU2
cell measures **342.264 tok/s prefill and 41.117 tok/s decode** (decode samples
41.117/41.096/41.177). It settles at depth 4, completes 10/20/20 expert moves,
uses 228 drafts and 289 verifier rows per 256-token request, and preserves the
saved tuning checkpoint's terminal token stream. This reproduces the accepted
approximately 42-tok/s class; the former 16.8-tok/s result is a policy mismatch,
not evidence of a twofold implementation regression.

The Ornith CPU2 cell measures **422.516 / 46.687 tok/s**. All **14 selected
non-CUDA-only cells** complete in **969.18 seconds**. Each raw record proves
PerfStats disabled and three complete 512-token prefill / 256-token generation
requests after warmup. The whole-run kernel observer has **zero new records
and zero driver findings**. The five CUDA-only candidates are excluded; mixed
CUDA/CPU and CUDA/ROCm candidates remain in scope.

These local AVX512 results do not certify an image or establish an AVX2
performance result. Their initial ratchet reports `new_baseline`, because the
corrected benchmark policy must not borrow the old stress-policy marks.

## Matched release comparison method

The original `2026-09-24.1` release's immutable AVX512 image is
`sha256:a68444c62bfdbe2740f5bb12e19e7c070ab3319d2556d512d13a4c44a4c3629e`,
source revision `ff61316f3184a2846201445833c6fd4687d4142e`. Remeasure that exact
image with the corrected manifest and workload on the same host, sequentially
after the current-build run. Reuse the existing tmpfs without copying model
payloads, keep all three samples, and require exact benchmark-identity equality
before comparing each phase. This measures each implementation's production
defaults under matched intent, not the causal contribution of an individual
kernel or policy change. No checked-in marks are advanced by this diagnostic.

The old image reproduces IH2 ring-overflow and deferred-SVM-worker warnings on
the upgraded host driver. Its driver-health result must therefore remain red
even if all benchmark requests complete. Do not confuse those findings with
the current build's clean E2E and benchmark intervals, or change the old image's
registration policy to manufacture a healthy release result. Preserve its raw
timings as a qualified comparison, not a passing correctness certificate.

Release comparison artifacts are under
`parity-results/driver-upgrade-benchmark-defaults-20260925/`; the independent
driver checkpoint/report use the `release-benchmark-defaults-run-` prefix in
`/tmp/llaminar-driver-upgrade.Qq9WtW/`.

## Completed matched comparison

All **14/14** release-image benchmark cells complete in **1,092.50 seconds**.
Every cell has an exactly matching measurement identity, ISA and workload;
all raw samples in both runs are unprofiled and complete. Comparing their
medians through the existing ratchet gives **28/28 passing phase checks** at
the unchanged 10% tolerance. The largest decrease is **1.30%**. The current
runtime is the local Release AVX512 binary with this branch's uncommitted
driver fixes; the reference is the exact published container, not a rebuild.

Rates below are tokens/second. `CPU2` means both physical CPU sockets/NUMA
endpoints. All rows use dynamic MTP and each cell's production-default
movement policy. Models, precision and declared context remain unchanged.

| Model / topology | Release prefill | Current prefill | Change | Release decode | Current decode | Change |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 3.6 35B / CPU2 | 57.24 | 342.26 | +497.94% | 24.45 | 41.12 | +68.19% |
| Ornith 1.5 35B / CPU2 | 113.27 | 422.52 | +273.02% | 28.31 | 46.69 | +64.94% |
| Qwen 3.5 122B / CUDA2 + CPU2 | 146.45 | 199.36 | +36.13% | 15.80 | 16.99 | +7.56% |
| Qwen 3.5 122B / CUDA2 + ROCm4 + CPU2 | 109.12 | 165.59 | +51.75% | 21.30 | 21.39 | +0.43% |
| Qwen 3.5 122B / ROCm2 + CPU2 | 136.70 | 225.75 | +65.14% | 24.81 | 26.95 | +8.66% |
| Qwen 3.5 122B / ROCm4 + CPU2 | 155.22 | 247.77 | +59.62% | 26.13 | 26.99 | +3.30% |
| Qwen 3.5 122B / CUDA2 + ROCm4 | 191.16 | 192.70 | +0.81% | 25.65 | 25.70 | +0.18% |
| Qwen 3.8 27B / CUDA2 + ROCm2 TP/PP | 506.82 | 503.22 | -0.71% | 49.76 | 49.93 | +0.35% |
| Ornith 1.5 35B / ROCm2 | 1,056.94 | 1,062.87 | +0.56% | 69.30 | 70.37 | +1.55% |
| Qwen 3.6 35B / ROCm1 | 1,161.83 | 1,157.98 | -0.33% | 85.90 | 86.32 | +0.49% |
| Ornith 1.5 35B / ROCm1 | 1,161.38 | 1,165.31 | +0.34% | 98.15 | 98.02 | -0.13% |
| Qwen 3.8 27B / ROCm2 TP | 371.55 | 377.96 | +1.72% | 45.34 | 46.11 | +1.71% |
| Qwen 3.8 27B / ROCm2 PP | 273.95 | 275.28 | +0.49% | 39.13 | 39.26 | +0.33% |
| Qwen 3.8 27B / ROCm1, 32K context | 280.36 | 276.73 | -1.30% | 52.09 | 52.34 | +0.49% |

The old image's separate driver gate **fails with 21 findings**: thirteen IH2
overflows, six deferred-SVM-worker warnings and two SVM-restore-worker warnings.
The wrapper consequently exits nonzero even though all timing requests finish.
There is no GPU reset or fault in that interval. These are qualified timings
from an unhealthy old implementation, not a clean release re-certification.
The current build's entire benchmark interval remains clean, as do all eleven
preceding ROCm-containing HTTP cells. The upgraded host driver alone does not
replace the source-level host-registration fix.

`matched-release-comparison.json` retains all 28 ratchet comparisons, all three
samples per cell, links to original raw artifacts, both driver receipts and
the old published stress-policy scores marked **not comparable**. It does not
mint a certificate, update Git high-water marks, or erase the interrupted
stress-policy run. No additional inference tuning was performed during this
benchmark-policy slice.
