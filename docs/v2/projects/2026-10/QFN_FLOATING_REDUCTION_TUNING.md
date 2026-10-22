# QFN floating projection reduction tuning

Updated: 2026-10-22. Candidate96 preserves the original floating arithmetic
while reducing block barriers from nine to three. The focused native gates,
complete Unit suite and frozen A1 accuracy panel pass. Four clean matched
Release arms qualify **L96: 796.370 prefill / 36.750 decode tok/s**, versus
Automatic94 at **791.395 / 36.654** in the same comparison. Prefill improves
0.629%; the 0.260% decode difference is small with overlapping ranges.

## Arithmetic and scope

The three existing ROCm projection families retain their 256 original lane
accumulators, K iteration order, FP32 FMA expressions and original FP32/FP16/BF16
weight conversions. The reduction keeps the original shared-memory edges at
128 and 64, then performs the same additions with a wave64 exchange at
32/16/8/4/2/1. Only the original owning lanes add. Source formats, row admission,
launch geometry, streams, scratch ownership and public launch APIs are unchanged.

This covers FP32 small-N, FP32-activation/native-half-weight, and floating fused
SwiGLU-down projections. Actual-half packed projections already have their own
fixed-tree implementation and are unchanged. The complete configured HIP target
set for this change is gfx906; this is not a wave32 implementation or a CUDA
kernel change.

## Correctness

The new `V2_Integration_ROCmFloatingReductionHistoricalTree` registration is an
explicit production-preflight entry. Its test-only oracle retains literal
historical kernel bodies, compiled with the production FMA policy. It passed
against unchanged production before the optimization was applied. The candidate
then passed it and five neighboring registrations: request holes, packed-half
reductions, GDN verifier rows, short-half groups, and Release target selection.

The independent byte matrix crosses all three floating weight formats and both
plain/fused operations, M1/3/17, boundary K values through 10240, and narrow/wide
N. Every case replays twenty changing full/partial/empty/one-live histories,
poisoning inactive inputs and independent output banks. Mixed mantissas and
exponents exercise rounding; separate signed-zero, subnormal, extreme and
nonfinite cases avoid masking finite checks with NaNs. A deliberate different
association is a negative control. Existing request-hole and grouped tests
retain their broader physical-row coverage.

Complete Unit: **776/776**, zero skips. Frozen Release accuracy: **185/185 stored
logit rows byte-identical** to Automatic94, with reset and driver/retirement
checks passing. Required reference metrics are unchanged:

| Group | Phase | Rows | Mean KL, nats | Top-1 agreement | Mean top-5 overlap |
|---|---|---:|---:|---:|---:|
| Short | Prefill | 16 | 0.033535869 | 93.75% | 87.50% |
| Short | Decode | 96 | 0.032070057 | 97.92% | 92.29% |
| Held out | Prefill | 8 | 0.009691368 | 100% | 95.00% |
| Held out | Decode | 48 | 0.003837619 | 100% | 94.17% |

The byte comparison also includes retained diagnostic/reset rows. These required
groups do not replace complete model/topology or long-context qualification.

## Unprofiled component measurements

The Release fixture captures 256 launches of one exact operation. It compares
the current production entrypoint with the historical native oracle using twelve
alternating-order GPU-event pairs after warmup. Times below include retained
graph replay economics and are not model throughput or sums of removable model
costs. Inputs are synthetic; geometry relevance is labeled explicitly.

| Operation | M/N/K | Weight | Candidate, µs | Historical oracle, µs | Oracle/candidate |
|---|---|---|---:|---:|---:|
| HC inject geometry | 1/4/10240 | FP32 | 10.2731 | 10.5737 | 1.0293× |
| Indexer query geometry | 1/512/2560 | FP16 | 46.8655 | 46.8674 | 1.0000× |
| Indexer key geometry | 1/128/2560 | FP16 | 7.0525 | 7.2131 | 1.0228× |
| HC bulk geometry | 512/4/10240 | FP32 | 114.2605 | 113.4625 | 0.9930× |
| Additional native format | 1/512/2560 | BF16 | 46.8959 | 46.8883 | 0.9998× |
| Fused component | 17/33/257 | FP32 | 5.2187 | 5.5834 | 1.0699× |
| Fused component | 17/33/257 | FP16 | 5.6525 | 5.9681 | 1.0558× |
| Fused component | 17/33/257 | BF16 | 5.4403 | 5.7472 | 1.0564× |

The unchanged production HC control measured **10.4349 µs**, versus its literal
oracle at **10.5869 µs**. That existing 1.46% executable/oracle difference is not
an optimization gain. The candidate is approximately **1.55% faster than the
original production control** on this geometry. The larger indexer is neutral;
the slight bulk difference remains visible rather than being hidden by a mean.

## Isolated ROCm profiles

Separate profiler processes execute exactly one family, format and shape. Six
traces cover candidate/oracle for HC FP32, indexer FP16 and fused FP16. Each trace
authenticates twelve retained graph replays with 256 selected dispatches each;
setup copies are outside that captured family. Hardware-counter passes select
one matching dispatch and one metric each, keeping gfx906 memory groups separate.
Profiler times are diagnostic and never enter the unprofiled comparison.

| Family | VGPR / SGPR, compiler counts | LDS | Scratch / spills | Barriers before → after |
|---|---:|---:|---:|---:|
| FP32 small-N | 10 / 28 | 1024 B | 0 / 0 | 9 → 3 |
| FP32 × native half | 15 / 28 | 1024 B | 0 / 0 | 9 → 3 |
| Fused floating down | 23 / 32 | 1024 B | 0 / 0 | 9 → 3 |

The profiler reports rounded register allocation (12/16/24 VGPR and 32 SGPR),
which is distinct from compiler usage. The unchanged FMA sites remain visible;
six shared-memory reduction rounds become six wave exchanges. All three measured
families report zero LDS bank conflicts. Wave counts match actual live launch
geometry: HC16, indexer2048, fused2244. The HC case has only four blocks across
the GPU, so low resource pressure does not imply high device occupancy.

The SDK has no derived mean-occupancy counter for gfx906; register/LDS admission
and actual wave counts constrain occupancy without claiming an attained value.
`VALUUtilization` measures active lanes inside a wave, not device occupancy.
The additional 1025th-dispatch observations show HC `VALUBusy` about 0.172%,
indexer 6.51%, and fused 18.37%. Indexer memory-unit busy is 12.10%, memory-unit
stalled 0.027%, L2 hit 19.64%, and off-chip fetch 2570.4 KiB. These are separate
instrumented observations, not additive bottleneck percentages. Small write
counts can reflect deferred writeback rather than missing output stores.

ISA inspection identifies an additional indexer opportunity: the uniform
FP16/BF16 choice is still tested inside each K iteration, with a weight-load
wait followed by conversion and a later activation-load wait. A separate
source-format specialization candidate is prepared privately. It is not part
of Candidate96 or its accuracy/timing claims.

## Matched Release model qualification

The complete A/B/B/A comparison uses the accepted Automatic94 archive and the
new immutable reduction96 archive, with identical model shards, native
libraries, precision selectors and production placement. All four arms retire
normally with clean driver windows and unchanged source/runtime identities.
All twelve natural 256-token sequences match exactly. Each arm contains one
warmup and three measured requests; no profiler or PerfStats instrumentation
runs in these measurements.

| Phase | Control median | Candidate median | Change | Control range | Candidate range |
|---|---:|---:|---:|---:|---:|
| Prefill, tok/s | 791.395 | 796.370 | +0.629% | 791.143–791.596 | 790.731–797.119 |
| Decode, tok/s | 36.654 | 36.750 | +0.260% | 36.584–36.811 | 36.645–37.017 |

Both candidate process medians improve prefill. Decode remains effectively
unchanged within the observed dispersion; this is not a material decode win.
The qualification retains the original numerical policy and installs the
smaller synchronization tree with the focused native byte proof. It does not
claim a 2.93% model gain from the isolated HC oracle ratio.

The workload is the dashboard's original four-MI50 TP4 cell:
`TensorParallel + GateUpOwnedDownColumns`, Automatic-resolved Four workers,
`VocabularySharded`, static GPU experts and RAM ngrams. It uses 512 authenticated
prompt tokens, 256 naturally generated tokens (255 after-prefill decode steps),
context 4096, bucket 512, FP16 KV, MTP disabled with zero reserved draft capacity,
prefix caching disabled, greedy seed 42 and 28 physical NUMA1 workers. Existing
six precision selectors remain unchanged. This is a focused checkpoint; the
last complete production-preflight qualification remains G60.

## Reproduction and retained evidence

`v2_perf_rocm_floating_reduction` is an explicit Release timing target, excluded
from automatic Unit/preflight dependencies. The same oracle header is shared
with the functional fixture. CTest registration is
`V2_Perf_ROCmFloatingReductionExactShape`; the disabled GTest is opted into only
by this performance registration or an explicit profiler invocation.

Select `QFN_REDUCTION_M`, `_N`, `_K`, `_TYPE` (0 FP32, 1 FP16, 2 BF16), and
`_FUSED` (0/1). For an isolated profiler process also set
`QFN_REDUCTION_PROFILE_SIDE=candidate` or `oracle` and invoke the same executable
with `--gtest_also_run_disabled_tests` and
`--gtest_filter=ROCmFloatingReduction.DISABLED_ExactShapePairedEvents`.
Profile the actual selected HIP runtime and Release core, with one exact case
per launch. Ordinary timing leaves the profile-side variable unset.

Local evidence under `parity-results/qwen-flash-next/`:

- `floating-reduction96-original-test-r1`, `floating-reduction96-candidate-native-r1`;
- `floating-reduction96-unit-r1`, `reduction96-release-accuracy-r1`;
- `reduction96-model-{a1,b1,b2,a2}` and `reduction96-model-comparison.json`;
- `floating-reduction96-component-timing-summary-r1.json` and its raw paired runs;
- `floating-reduction96-original-isa-r2`, `floating-reduction96-candidate-isa-r1`;
- `floating-reduction96-profile-summary-r3.json` and the separate `reduction96-*`
  trace/counter directories.

The first Release target/registration attempts failed before native execution;
their logs remain retained. The corrected target is available in Release and
the existing executable-CMake regression covers its native-language metadata.
Complete production preflight remains pending.
