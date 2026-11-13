# QFN floating projection reduction tuning

Updated: 2026-10-22. **L97 source-format specialization is qualified at
814.368 prefill / 36.727 decode tok/s.** Its matched L96 control measures
796.421 / 36.812: **+2.254% prefill / −0.229% decode**, with overlapping decode
ranges. All 185 logit rows and twelve generated sequences remain byte-identical;
776 Unit tests and six focused registrations pass. Full preflight is pending.

Candidate96 preserves the original floating arithmetic
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

## Native source-format specialization: candidate97

The next candidate specializes the existing FP32-activation/native-16-bit-weight
projection on its validated FP16 or BF16 source encoding. The host selects the
encoding at the existing launch boundary, removing the dtype branch from every
K iteration. Conversion, FP32 FMA order, stride-256 lane ownership, the accepted
three-barrier reduction, live-row masks, explicit stream and launch geometry
remain unchanged. This introduces no shape threshold, requantization or new
allocation.

All six final focused registrations pass in `x16-97-final-native-r1`. The
historical native oracle now also covers raw FP16/BF16 signed subnormals,
minimum normals and signaling/quiet NaNs. K1 witnesses scale tiny weights into
nonzero normal FP32 products, preventing matching zeros from concealing loss.
Paired NaN payloads exercise the original 0/128, 64/192 and 0/64 edges in both
operand orders and all sign combinations, for all three floating formats and
plain/fused families. These remain part of the explicit historical-tree
production-preflight registration. Complete Unit passes **776/776**, zero skips,
in `x16-97-unit-r1`. The unchanged Release accuracy panel passes in
`x16-97-release-accuracy-r1`: all **185 final-logit rows / 26 files** are
byte-identical to L96, including reset checks. Required KL and top-1/top-5
metrics therefore remain those reported above.

### Unprofiled component measurements

These captured native-event medians compare production96 with production97,
using synthetic inputs and twelve alternating pairs with an independent native
oracle in each process. K is 2560 throughout. They measure components rather
than whole-model throughput.

| Source | M | N | Control, µs | Candidate, µs | Speedup |
|---|---:|---:|---:|---:|---:|
| FP16 | 1 | 128 | 7.050660 | 5.243765 | 1.345× |
| BF16 | 1 | 128 | 6.823150 | 5.053450 | 1.350× |
| FP16 | 1 | 512 | 46.893050 | 46.916700 | 0.999× |
| BF16 | 1 | 512 | 46.850200 | 46.901100 | 0.999× |
| FP16 | 9 | 512 | 72.401000 | 47.221700 | 1.533× |
| BF16 | 9 | 512 | 75.178150 | 42.872950 | 1.754× |
| FP16 | 512 | 512 | 3584.785000 | 2760.580000 | 1.299× |
| BF16 | 512 | 512 | 3523.500000 | 2650.825000 | 1.329× |

The wide single-row query is effectively unchanged. The narrower key and
larger-M components improve. M512 controls use the retained96 archive core;
their declared loader closure was checked before execution. Evidence is in
`x16-97-component-comparison-r1.json` and
`x16-97-bulk-component-comparison-r1.json` under the local QFN result root.

### Isolated profiling

Six separate format/shape traces each authenticate 3,072 selected dispatches:
twelve replays of 256 captured kernels, with exact geometry. Twenty-two
additional counter processes each select one metric on the 1,025th matching
dispatch. Profiled durations are excluded from the timing table.

Both specializations retain 1 KiB LDS, three barriers, zero spills and zero
private storage on gfx906. FP16 compiler usage falls from 15 to 14 VGPRs;
BF16 falls from 15 to 10. Both use 28 SGPRs. Rounded hardware allocations are
16/32 and 12/32 VGPR/SGPR respectively. The M9 traces report L2 hit rates of
88.988% for FP16 and 87.843% for BF16, with zero measured LDS bank conflicts.
These observations do not establish attained CU occupancy or a general
bandwidth explanation. Exact records and ISA are in
`x16-97-profile-summary-r1.json` and `x16-97-candidate-isa-r1`.

The wide M1 query remains open. Its observed fetch is close to its 2,560 KiB
native weight footprint, versus 640 KiB for N128; removing repeated activation
loads cannot be claimed to eliminate that measured weight traffic. The indexer
has four heads independent of the participant's ordinary attention heads, and
scoring consumes all four. Sharding therefore requires an explicit live-query
collective and original-head normalization/observation contracts. That larger
graph change is separate from this local specialization. Recorded-input GDN
scaling remains the next measurement slice.

### Matched Release qualification: L97

Four clean A/B/B/A arms compare the immutable accepted L96 archive with the
new specialization archive. The complete model, prompt, precision selectors,
placement, native dependency closure and generation settings are identical to
the workload documented above. Each process has one warmup and three measured
requests, without a profiler or PerfStats instrumentation. All twelve natural
256-token sequences match exactly. Source/runtime identities, observed native
libraries, driver windows and normal retirement pass in every arm.

| Phase | L96 control median | L97 median | Change | Control range | Candidate range |
|---|---:|---:|---:|---:|---:|
| Prefill, tok/s | 796.421 | 814.368 | +2.254% | 796.171–797.193 | 813.664–815.496 |
| Decode, tok/s | 36.812 | 36.727 | −0.229% | 36.729–36.904 | 36.685–36.765 |

Both candidate process medians improve prefill. Decode is slightly lower with
overlapping ranges; this checkpoint claims a prefill gain, not a decode gain.
Its complete evidence is `x16-97-model-{a1,b1,b2,a2}` and
`x16-97-model-comparison.json`. The original L96 archive retains its actual
precommit source receipt; it has not been relabeled as the current source.
The latest complete production-preflight checkpoint remains G60.

## Deferred one-barrier experiment

An independent candidate moved both original cross-wave edges into the first
wave after one LDS publication. Its initial gate failed all 144 paired-NaN
cases. The finite addition tree was unchanged, but LLVM reversed the native
operand order of three additions, changing which NaN payload survived. ISA
inspection identified the original high-before-low ordering and the candidate's
low-before-high ordering.

Three explicit native adds restored the original operand order without a NaN
scan, extra arithmetic or additional buffers. All six focused gates then
passed. Release ISA and three isolated traces confirmed one barrier, unchanged
register/LDS allocations and zero spills/private storage. This demonstrates why
native byte tests are necessary even when source parenthesization is preserved.

The nine-case comparison measured approximately 2.8–2.9% improvement for narrow
single-row keys, 4.1–5.2% for nine-row queries and 2.8–4.0% for fused projections;
wide single-row queries remained neutral. HC initially measured 10.205 µs for
the control and 10.549 µs for the candidate, but a fresh candidate process
measured 10.065 µs. Historical-oracle medians in those processes were 10.539,
10.454 and 10.384 µs respectively. The repeat prevents a conclusive HC
regression claim and exposes interprocess variability.

This optional change is deferred because its smaller gains and mixed timing
evidence do not yet justify the operand-order specialization. It is absent from
selected candidate97. Its paired-NaN regressions remain. Retained evidence:
`floating97b-candidate-native-r1` (initial failure), `floating97c-native-r1`
(corrected bytes), `floating97c-component-comparison-r1.json`,
`floating97c-hc-confirm-r1` and `floating97c-profile-summary-r1.json`.

## L106: overlap loads while preserving each original dot product

The Castagna comparison identified a different, measurable limitation in the
FP32 HC injection. The original gfx906 loop issues one A/B pair, waits for
memory, and updates its accumulator before requesting the next pair. L106
loads four or eight original stride-256 pairs ahead, then consumes them in the
same ascending order. It retains one accumulator, the original FP32 FMA operand
order, all three reduction barriers, the exact scalar K tail, and live-row
ownership. It does not copy Castagna's different float4 product grouping.

The complete eight-load experiment passes the original native byte gates and
all 185 canonical final-logit rows, but its isolated M512 latency is 3.86% higher.
The final implementation therefore has scalar, four-load and eight-load
compiled schedules. Dispatch uses the immutable number of actual CTAs in each
native chunk (`M*N*chunkProjections`), rather than an M lookup or mutable live
count: at most 64 selects eight, at most 256 selects four, otherwise scalar.
K below 2048 caps the choice at four; below 1024 selects scalar because no full
prefetch group exists. Division guards prevent overflow even at INT_MAX.
No stream, graph lifecycle, binding, allocation, transfer or precision policy
changes. Existing M/N/K/chunk identity determines the retained kernel choice.

The [crossover sweep](../../../../parity-results/qwen-flash-next/injection106-crossover-r1.json)
supports eight loads through M16, four at M32/M64, and scalar at large prefills
for N4/K10240. Extra width and short-K controls support the dispatch; this is
measured coverage, not a claim that every unmeasured N/K shape is optimal.

Final [unprofiled component ABBA](../../../../parity-results/qwen-flash-next/injection106-split-component-comparison-r1.json)
uses the same frozen Release fixture and immediate scalar production control.
Each process contributes 12 event pairs; two processes per role are retained.
The historical nine-barrier oracle verifies bytes but is not the timing control.

| HC shape N4/K10240 | Original scalar us | Selected schedule us | Speedup |
|---|---:|---:|---:|
| M1, eight loads | 10.39905 | 4.637805 | 2.242× |
| M32, four loads | 12.19340 | 8.475935 | 1.439× |
| M512, scalar | 113.55300 | 113.421000 | 1.001× |

The [isolated profiles and ISA](../../../../parity-results/qwen-flash-next/injection106-split-profile-summary-r1.json)
authenticate actual dispatched modes 8/4/1, exact core ownership and retained
graphs. Allocated VGPR/SGPR counts are 24/48, 16/32 and 12/32 respectively; all
retain 1024 B LDS, three barriers, zero register spills and zero private storage.
The shipped ROCm target is gfx906. Static used-register counts are recorded
separately from native allocation. All three measured schedules have zero LDS
bank conflicts. FetchSize is 201.94 KiB at M1, 1440.38 KiB at M32 and 20865.94 KiB
at M512. The original M1 control was 201.75 KiB: fewer bytes do not explain the
decode win. VALUUtilization measures active lanes, not GPU occupancy; the
four-CTA M1 launch still occupies at most four CUs. Counters and instrumented
durations are diagnostic and do not replace the event or model timings.

All [six focused registrations](../../../../parity-results/qwen-flash-next/injection106-split-focused-r1/result.json)
and [776 Unit tests](../../../../parity-results/qwen-flash-next/injection106-split-unit-r1/result.json)
pass. The new explicit preflight registration compares independent frozen
native arithmetic at load-group tails, cancellation and NaN payload boundaries.
Captured graph inspection also authenticates each real chunk's mode, including
partial final chunks, request holes and 20 dirty full/partial/empty replays.
Device-free policy tests cover K/CTA boundaries, factorization and overflow.
The final split's fresh [canonical model panel](../../../../parity-results/qwen-flash-next/injection106-split-release-accuracy-r1/report.json)
passes, with [all 26 files / 185 final-logit rows byte-identical](../../../../parity-results/qwen-flash-next/injection106-split-release-accuracy-r1/logit-byte-comparison.json)
to the immutable HC103 control. These are the final split's own observations;
the all-eight candidate's prior pass is retained separately.

The [matched Release ABBA](../../../../parity-results/qwen-flash-next/injection106-split-model-comparison.json)
uses the unchanged four-MI50 A1 topology, native dependencies, precision and
host placement: `TensorParallel` + `GateUpOwnedDownColumns`, static GPU experts,
sharded vocabulary, Automatic-resolved Four workers, RAM ngrams, FP32 model
activations and FP16 KV, context 4096, M512, no MTP or prefix cache, and 28
physical NUMA1 host cores. Each process has one warmup and three measured
requests of 512 prompt tokens / 256 emitted tokens / 255 after-prefill forwards.

| Phase | HC103 control median | L106 split median | Change | Control range | Candidate range |
|---|---:|---:|---:|---:|---:|
| Prefill, tok/s | 815.330 | 815.992 | +0.081% | 806.490–816.660 | 815.451–816.255 |
| Decode, tok/s | 37.520 | 39.274 | +4.676% | 37.480–37.635 | 39.186–39.415 |

All twelve natural generated sequences match exactly. Source/runtime and actual
loader identities, driver windows and normal retirement pass in all four arms.
Both candidate process medians improve decode, with nonoverlapping pooled
decode ranges; prefill is effectively unchanged. No sample is discarded,
including the slower control prefill. This is a measured decode improvement,
not completion of the historical Castagna 42.959 tok/s target. Complete
ProductionTestPreflight remains G60.

The selector's INT_MAX tests prove bounded dispatch arithmetic only. Source
review also records a pre-existing native signed-loop limit near INT_MAX K;
that extreme traversal is not certified by these selector tests or by the
measured K range through 10240.
