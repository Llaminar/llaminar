# QFN GDN recurrence tuning

## Production layout integration190, 2026-10-24

The current default-policy focused gate passes **873 registrations: all 778 Unit
tests and 95 affected integrations**, with no CTest failures or skips. The existing exact
legacy intra-GTest Unit coverage limitations remain separately recorded. Its
earlier explicit-policy run found one stale CUDA function-name lookup in an architecture test;
that lookup is repaired and the actual typed stream/snapshot forwarding checks
are stronger. No production numerical defect was found in that run. Both native
two-device cold layout/capture/restore tests pass. Evidence:
`gdn190-default-focused-gate-r2` and final builds
`gdn190-default-{release,integration}-build-r2`. The earlier explicit-policy
872-test pass remains in `gdn190-focused-gate-r2`.

The complete **production** convolution-plus-recurrence boundary now measures
**15.3450 µs KeyMajor versus 11.0900 µs ValueMajor**, or **27.73% less time**.
It retains two kernels, 64 complete operations per graph, five warmups, twelve
alternating samples, and full before/after byte checks. This is a component
result. The separate matched full-model A/B/A now passes:

| Full TP4 Release workload, tok/s | Frozen180 controls | Explicit ValueMajor190 |
|---|---:|---:|
| Prefill median | 818.917787 | 820.871265 |
| Decode median | 40.905758 | 41.536547 |
| Decode range | 40.754289–41.004639 | 41.518561–41.573551 |

Decode improves **1.542%**, with non-overlapping measured ranges; prefill improves
0.239%. All nine natural 256-token trajectories match. The canonical selected
QFN accuracy panel also passes: **185 final-logit rows in 26 files are byte
identical to both accepted180 and151**. Actual owner-derived submission counters
prove ValueMajor on all four ranks; they are not device replay counts. This is
the selected topology's required-group panel, not whole-corpus certification.
The candidate also contains the nested Fold12 attention merge. This model delta
belongs to the combined checkpoint; the isolated GDN boundary above establishes
the recurrence improvement independently.

Settings remain the same UD-Q4_K_XL GGUF, four MI50s in full TP4, sharded
vocabulary, RAM ngrams, 28 physical NUMA1 workers, prompt512/emitted256/context4096,
FP16 KV and explicit Prefill8/Decode33 attention, MTP/prefix off, greedy seed42,
one warmup and three measured requests per A/B/A arm. The candidate alone has
the explicit ROCm ValueMajor selector. All arms pass driver, source/runtime,
resource and normal-retirement checks. Evidence: `gdn190-model-fp16-r1`,
`gdn190-model-throughput-{a1,b1,a2}-r1`, and
`/tmp/qfn-gdn190-model-benchmark-r1/summary.json`. The **42.959 tok/s target remains
open**. Default-selection model qualification is recorded separately below.

The default-only Release and Integration rebuilds pass. Bounded linked-code inspection proves
all 140 CUDA GDN variants across SM80/86/89/90 and all 45 gfx906 variants retain
the explicit190 bodies and resources; all new variants remain spill-free.
Evidence: `gdn190-default-release-build-r1` and
`/tmp/qfn-gdn190-value-major-root-private/{cuda-core-isa-default-r1,rocm-core-isa-default-r1}`.
The first default gate passes 872 of 873 registrations. Its only failure is
the existing 60-second `ROCmGDNMergedCoordinates` timeout; the driver interval
is clean. This fixture still exercises the original KeyMajor kernels, and an
unchanged isolated rerun passes in 45.44 seconds. The prior complete gate took
51.62 seconds for this test, leaving little scheduling margin.

The fixture downloaded the same terminal candidate images repeatedly. Keeping
its three reused images after the terminal join removes 5,220 duplicate D2H
calls and 23.28 GiB of transfers across the complete sweep. Each independent
oracle is still read and compared; all 261 cases, four replay states, seven
pair comparisons, source/mapping checks and the 60-second timeout remain.
Both fixture-only r2 rebuilds pass. Three fresh native lifetimes pass in
35.42, 38.50 and 40.77 seconds, each retaining all 783 capture records for the
261 cases. The refreshed 873-registration gate passes in 270.38 seconds,
including this fixture in 39.78 seconds under the concurrent scheduler. Driver,
source/runtime and normal-retirement checks pass. The absent-override canonical
model panel also passes in 98.42 seconds: both backend layout selectors are
absent, actual bound-owner telemetry identifies ValueMajor on all four ROCm
participants, and all 185 final-logit rows remain byte-identical to accepted180
and151. The final candidate-only benchmark passes at **820.776737 prefill /
41.561713 decode tok/s**. Its three decode samples span 41.539201–41.629756;
all three generated sequences match the three explicit190 reference requests.
Both layout overrides are absent in the actual native process, and driver,
runtime/source, resource and normal-retirement checks pass. This one-candidate
repeat verifies default behavior; the earlier A/B/A owns the causal speedup.
Evidence:
`gdn190-default-focused-gate-r1`, `gdn190-merged-coordinates-baseline-r1` and
`gdn190-default-{release,integration}-build-r2`, and
`gdn190-merged-coordinates-readback-r{1,2,3}`; default model evidence is
`gdn190-default-model-fp16-r1`, `gdn190-default-model-throughput-b1-r1`, and
`/tmp/qfn-gdn190-default-model-benchmark-r1/summary.json`.

Twelve separate exact-dispatch profiles join successfully to the actual loaded
core. The recurrence changes from 96 to 384 wavefronts and from 796.5 to 502
VALU instructions per wave, so it issues more total vector instructions despite
finishing faster. Fetched bytes stay at 778.5 KiB; measured LDS conflicts are
zero on both sides. The candidate has no static/dynamic LDS or private storage;
the control uses 7,184 B dynamic LDS. Registers change from 119/120 to 54/56
used/allocated VGPRs. These facts do not measure achieved occupancy.

The first offline profile join incorrectly equated the descriptor's encoded
8-SGPR units with gfx906's physical 16-SGPR allocation units. Canonical SDK and
LLVM sources resolve that distinction; a separate corrected reader accepts all
twelve saved profiles with seven focused positive/negative controls. Original
collectors, failed join and native evidence remain immutable. Evidence:
`gdn190-production-complete-{functional-r3,events-r1}` and
`/tmp/qfn-gdn190-sgpr-join-repair-private/{README.md,all12.json}`.

The raw register/shuffle candidate now has a coherent production implementation
under an explicit cold diagnostic selection. Persistent recurrent banks use the
same chosen physical layout for prefill, scalar/grouped decode, snapshots,
request reset, prefix archives and whole-head linked-state exchange. No live
state transpose, host mirror, new allocation or second admission ledger is
introduced. CPU and CUDA defaults remain key-major. ROCm's default ValueMajor
selection now passes native and model qualification with the override absent;
the final matched default benchmark also passes.

`LLAMINAR_CUDA_GDN_STATE_LAYOUT` and `LLAMINAR_ROCM_GDN_STATE_LAYOUT` independently
select the canonical names `key_major_value_contiguous` or
`value_major_key_contiguous` at cold setup. Unknown values reject. The bound
format participates in cache and capture identity; same-sized foreign state
cannot be imported or bound as a snapshot/work bank. Recurrent disk records
without format metadata are invalidated on upgrade; there is no live conversion.
Pure-attention archive identity remains unchanged. Explicit diagnostic observations canonicalize state
coordinates only after download, never in the production data path.

Independent review confirmed concurrent participant preparation through the
actual TP worker pool, including serving graph materialization. Both native
two-device cold preparation/capture/replay regressions now pass in the complete
872-entry focused gate. The subsequent mixed CUDA KeyMajor/ROCm ValueMajor PP
regression also passes: real owners, RAM restore, durable flush, complete owner
retirement, reopened-disk restore and four foreign-format rejection checks.
Its exact preflight alias is `V2_Integration_PipelinePrefixMixedGDNLayouts`;
evidence is `gdn190-mixed-prefix-native-r1`. This small native fixture proves
archive ownership, not a heterogeneous real-model qualification.

Release build r3 passes with all shipped CUDA SM80/86/89/90 and ROCm gfx906
spill guards enabled. The two earlier red receipts retain the corrected ROCm
forward-declaration ordering and missing CUDA test-buffer constructor defects.
Inspection of the freshly linked core proves **all 100 original CUDA kernel
variants (25 per architecture) and all 23 original ROCm variants byte-identical
in bodies and resources** to frozen180. All **40 new CUDA variants** and **22 new
ROCm variants** have zero private storage or spills. The initial CUDA
object-versus-linked comparison is retained as an invalid comparison boundary;
matching linked artifacts resolves its relocation and shared-reservation
differences. These compiler facts do not measure achieved occupancy.

The fresh ROCm production tests pass **354 geometry/policy cases, 1,416 dirty
states and 2,832 captured replays**, compared byte for byte with the independent
frozen native oracle. Coverage includes scalar/grouped and long prefill through
M4096, DK64/128, value tails, head shards, request banks, signed offsets,
normalization, aligned/scalar endpoints, and snapshot bounds. Native graph
inspection authenticates actual production entrypoints, arguments, geometry and
resources. The owner binding/restore test additionally passes twelve geometry
cells with three dirty cycles each; rejected foreign bindings preserve the
original banks and restored state drives captured continuation. The complete
production convolution/recurrence pair passes six normalization/offset cells,
four dirty states each and 48 graph replays. These correctness-only follow-ups
ran against the immutable Release binary while the independent Integration tree
compiled; their elapsed times are not kernel performance measurements.
Both fresh CUDA production tests also pass, covering M1–16/33/129/512,
geometry/tails, bindings, snapshots, state restore and continuation. These are
focused component results, not current-source Unit or whole-model certificates.

Release build r4 removes the retired private experiment APIs and fixtures while
retaining the production entrypoints and canonical historical regressions. All
22 new compiled ROCm bodies/resources are identical before and after cleanup,
as are the 23 original programs. The resulting bounded code object is 422,768 B.

Evidence: `gdn190-value-major-release-build-r{1,2,3,4}`,
`gdn190-rocm-production-functional-r2`, `gdn190-cuda-production-functional-r1`,
`gdn190-rocm-owner-bindings-r1`, `gdn190-production-complete-functional-r1`,
`/tmp/qfn-gdn190-value-major-root-private/{cuda,rocm}-core-isa-r1/result.json`,
`/tmp/qfn-gdn-layout-integration-private/syntax-result.json`, and
`/tmp/qfn-gdn-layout-independent-review-private/REVIEW.md`. The current collective,
Unit/affected, canonical QFN accuracy and matched Release model evidence is
recorded above; these older intermediate receipts remain immutable.
The directed cached long-prefill scalar-endpoint snapshot supplement now passes.
The independent resource join accounts for all 1,264 captured nodes: 910 exact
core symbol/resource matches and 354 separately identified historical-oracle
nodes in the test executable. All **22 of 22 emitted production variants** are
exercised. This proves coverage and resource identity, not achieved occupancy;
the receipt is `/tmp/qfn-gdn190-independent-resource-r2-private/result.json`.
Prepared Q/K publication remains a separate later candidate. The qualified
explicit190 score is **820.871265 prefill / 41.536547 decode tok/s**; the prior
default checkpoint180 remains separately retained.

## Scope and current disposition

Candidate177 is installed in the ordinary nonidentity M1 CUDA and ROCm paths.
It reads post-convolution merged QKV directly in the recurrence kernel, removing
one deinterleave launch while retaining the original FP32 operation order,
cache-owned live/speculative banks and snapshot publication. CPU already has a
direct merged-input path. Identity M1 and optimized multi-row prefill retain
their existing execution programs. No allocation or admission ledger is added;
the existing maximum-prefill scratch remains admitted, so this is not a peak
memory reduction.

The final focused gate passes **804 registrations: all 778 Unit tests, 25 affected
regressions and their fixture**. There are no CTest failures or skips; the exact
pre-existing intra-GTest Unit coverage limitations remain recorded separately.
The canonical selected QFN panel passes, with all **185 final-logit rows in 26
files byte-identical to accepted151**. Submission counters prove direct merged
M1 recording on all four MI50s, without claiming device replay counts.

| Final Release complete boundary | Original two-node control, µs | Direct merged M1, µs | Reduction |
|---|---:|---:|---:|
| ROCm H12 / keyH4 | 14.47875 | 12.09750 | 16.45% |
| CUDA H12 / keyH4 | 5.10400 | 3.56800 | 30.09% |
| CUDA H24 / keyH8 | 5.77600 | 4.27150 | 26.05% |

These event cells use DK128/DV128, ordinary in-place state, normalization,
64 repeated complete boundaries, five warmups and twelve alternating samples.
Native graph inspection proves 128 control nodes versus 64 candidate nodes;
byte checks run before and after the timed interval. The broader regressions
cover DK64/128, value-width tails, signed offsets, retained snapshot envelopes,
state restore, grouped equivalence and dirty captured replay.

Fresh ROCm profiles retain 96 waves, 796.5 VALU instructions per wave,
778.5 KiB fetched, zero LDS conflicts, 7184 B dynamic LDS and zero spills.
The actual loaded 172064-byte code object matches the final compiled object;
its raw and merged bodies are byte-identical to qualified177. The merged body
uses 119/120 VGPRs and 95/96 SGPRs (used/allocated). These are allocation limits,
not achieved occupancy. CUDA compilation passes the spill guard for SM80/86/89/90;
all 16 DK/state specializations have zero local storage and spills. Its nominal
static shared section is 12 B, while the loaded recurrence reports 16 B required
storage. The private collector now distinguishes those observations explicitly;
it does not infer a universal alignment rule. The original rejected H12 parser
receipt remains preserved beside its strict offline readmission. CUDA uses native
events and compiler metadata here; no new CUPTI attachment was attempted after
the previously recorded driver assertion.

| Matched full TP4 Release model throughput, tok/s | Frozen175 control | Candidate177 |
|---|---:|---:|
| Prefill median | 820.235 | 819.876 |
| Prefill range | 818.797–820.908 | 819.083–820.479 |
| Decode median | 40.151 | 40.328 |
| Decode range | 40.026–40.293 | 40.291–40.427 |

The measured changes are −0.044% prefill and +0.439% decode. Decode ranges
narrowly overlap and the second control is faster than the first: this is a small
measured gain, not a claim that the 42.959 tok/s Castagna target is reached.
All nine natural 256-token sequences match. A/B/A uses one warmup and three
measured requests per arm, the same Qwen3.8-Flash-Next UD-Q4_K_XL GGUF, full TP4
on four MI50s, sharded vocabulary, RAM ngrams, 28 physical NUMA1 workers, FP16 KV,
explicit Prefill8/Decode33 attention, prompt512/decode256/context4096, MTP/prefix
off and greedy seed42. All arms pass runtime, source, driver, resource and normal
retirement checks. Profiler durations are excluded from model throughput.

Integration caught and fixed the new merged launcher's overly strict snapshot
capacity equality: an M1 call may retain a larger verifier envelope. Both backends
now test that configuration, restore and subsequent live-state updates. The new
telemetry initially used broad enablement; the existing source guard caught it,
and it now uses the exact GDN domain. Focused regressions are explicitly in
production preflight, including `V2_Integration_ROCmGDNMergedVerifierEnvelope`,
`V2_Integration_CUDAGDNMergedDecode`, `V2_Integration_GDNMergedInputMetadata` and
`V2_Integration_ModelKLDDiagnosticTelemetryBoundary`.

Evidence is indexed by `gdn177-production-checkpoint-r1.json` under the QFN
local result root. The matched samples/settings are in
`gdn177-model-throughput-comparison-r1.json`; final gate and accuracy receipts
are `gdn177-unit-affected-r2` and `gdn177-model-fp16-r1`. This is a focused
development qualification, not a rerun of complete ProductionTestPreflight or
whole-corpus/image certification. The next experiment is an exact-order NativeQ8
producer/reducer fusion; no performance claim is made for it yet.

## Earlier component experiments

### Producer normalization screen185, 2026-10-24

The [three-way source map](QFN_GDN_STRUCTURE_COMPARISON.md) led to a complete
convolution/recurrence experiment. Its producer preserves raw FP32 convolution
output and publishes compact normalized Q/K once per original key head. V
remains in raw merged storage. The prepared recurrence retains the original
state update, reductions, gates and snapshots. Both control and candidate
capture two kernels; the compact planes require 4096 bytes per physical row
at H12/keyH4/DK128/DV128.

The first screen passes **96 geometry/policy cases, 384 dirty states and 768
captured replays**, covering M1 through M16, normalization on/off and offsets
0/+5/-5. Whole raw output, recurrent output/state and both snapshot families
match exactly. Independent convolution-history and guard checks pass, along
with 23 invalid-span cases and two null-stream rejections. All five original
control bodies and resource records remain byte-identical to frozen180.

| M1 complete convolution + recurrence, µs | Control | Candidate185 |
|---|---:|---:|
| Median, 12 alternating pairs | 15.24125 | 15.39375 |

The candidate is **1.001% slower** and is not qualified for production. Timing
includes 64 complete operations per retained graph, five warmup pairs and
byte checks before and after the measured interval. Inputs and initial states
are immutable synthetic values; these are not full-model throughput results.

Eight separately isolated counter launches and exact native/runtime joins pass:

| Role | Control VALU instructions/wave | Candidate VALU instructions/wave | Control/candidate fetched KiB |
|---|---:|---:|---:|
| Convolution/preparation | 106.0 | 120.6 | 90.9375 / 90.9375 |
| Recurrence | 796.5 | 692.5 | 778.5 / 778.5 |

Both convolution programs launch 40 wave64 groups; each recurrence launches
96. Those geometries imply 80704 versus 71304 vector instruction issues over
the complete operation, an 11.65% reduction. The wave counts here derive from
authenticated launch geometry; they are not an achieved occupancy counter.
Despite fewer instructions and unchanged fetched bytes, elapsed time does not
improve. Recurrence allocation remains 120 VGPRs, with 117 used versus the
control's 119. The producer uses 35/36 used/allocated VGPRs. All inspected
specializations have zero spills/private storage.

The follow-up185b specializes immutable M1 geometry so gate/QK setup precedes
state loading. Its complete pipeline improves **15.24125 to 14.67625 µs**, a
**3.707% time reduction**, and the same 96 cases pass. Register allocation
falls to **56 VGPRs**, with zero spills. Three isolated recurrence profiles
report 623.0 VALU instructions/wave, 96 wavefronts and unchanged 778.5 KiB
fetched. The producer, grouped prepared kernels and five original control
bodies/resources remain byte-identical. This is a component gain; no matched
model score or production promotion is established.

The raw-input follow-up185c moves the original normalization/gates before state
loading without changing the convolution or allocating prepared Q/K planes.
Six M1 normalization/offset cells pass, including 24 dirty states and 48 retained
replays. Two independent timing runs reduce complete-chain time by 2.782% and
2.802%: the latter measures **15.2574 to 14.8299 µs**. Recurrence uses 56 VGPRs
but retains the 22-register/10-LDS state split. Exact isolated profiles report
742.5 VALU instructions/wave, 96 waves and 778.5 KiB fetched.

Candidate185d keeps all 32 scalar state rows in registers after setup. Its
initial six-cell byte gate passes and complete-chain time falls from
**15.235 to 11.430 µs, a 24.976% reduction**. It uses **74/76 used/allocated
VGPRs**, zero spills/private storage and **2064 B dynamic LDS**, versus 7184 B
for the original recurrence. The DK64 scalar specialization remains 41/44
VGPRs with 1552 B dynamic LDS. Exact profiles report 595.5 VALU instructions
per wave, 96 waves and unchanged 778.5 KiB fetched. This is not an achieved
occupancy measurement. Original control kernels and the preceding prepared
variants remain identical in compiled bodies/resources; only raw scalar DK128
changes from185c.

All these timings cover the complete unchanged convolution plus recurrence,
with identical immutable synthetic inputs and initial state, 64 operations per
retained graph, five warmup pairs and twelve alternating measured pairs.
The expanded185d gate now passes **272 cases, 1088 dirty states and2176 new
captured serial replays**, compared against the original grouped/planar and
independent frozen historical programs. It covers M1..16, DK64/128,
DV1/65/128/129, normalization policies, signed/INT32-limit head offsets,
independent requests, in-place state and padded retained snapshot envelopes.
Gate inputs include finite softplus threshold and sigmoid saturation cases.
No real-model gain or production promotion is established by these component
results. The promising185d implementation requires neither new storage nor a
producer change.

Evidence: `gdn185-fixture-release-build-r2`, `gdn185-functional-r1`,
`gdn185-m1-events-r1` and the eight `gdn185-m1-*-{valu,fetch}-r1` receipts/joins
under the QFN result root. Source, binary and exact settings are archived at
`/tmp/qfn-gdn185-first-screen-r1`; resource/control equality is in
`/tmp/qfn-gdn185-root-private/isa-r2/result.json`. Production remains
checkpoint180 at 819.594 prefill / 41.030 decode tok/s.

The185b receipts are `gdn185b-fixture-release-build-r1`,
`gdn185b-functional-r1`, `gdn185b-m1-events-r1` and
`gdn185b-m1-candidate-recurrence-{valu,fetch,waves}-r1`. Its source/runtime archive
is `/tmp/qfn-gdn185b-component-green-r1`, with compiled resources at
`/tmp/qfn-gdn185b-root-private/isa-r1/result.json`.

The raw candidates are archived at `/tmp/qfn-gdn185c-component-green-r1` and
`/tmp/qfn-gdn185d-component-green-r1`. Their manifests bind exact sources,
copied runtime assets, Release build, functional/event and isolated profile
receipts. The185c profiler's original source-text prerequisite rejected the
refactored marker before native launch; the corrected frozen collector retains
strict runtime markers/native joins and adds a missing-marker negative check.
Fresh185c `functional-r2`, `m1-events-r2` and three recurrence profile receipts
pass. The185d equivalents are `functional-r1`, `m1-events-r1` and
`m1-candidate-recurrence-{valu,fetch,waves}-r1`; all driver and retirement checks
pass. Bounded ISA reports live under each `/tmp/qfn-gdn185{c,d}-root-private/isa-r1`.

Expanded185d evidence is `gdn185d-extended-release-build-r2` and
`gdn185d-extended-functional-r2`, archived at
`/tmp/qfn-gdn185d-expanded-green-r1`. The first attempt caught a test-owner
mistake: `HipIntBuffer(int)` initializes one integer, not an array capacity.
The array owner now takes its explicit authored vector; the full matrix then
passes in26.26 seconds. The failed first receipt remains intact, with clean
driver/retirement evidence. This was a fixture publication error before the
multi-row kernel call, not a production kernel defect.

### Transposed register/shuffle screen186, 2026-10-24

The user-approved [layout experiment](QFN_GDN_STRUCTURE_COMPARISON.md#approved-transposed-state-experiment-2026-10-24)
now has an initial measured candidate. Persistent state is physically
`[head,value,key]`, with16 lanes per value column and8 state words per lane
at DK128. It uses vector state/Q/K loads and **zero static or dynamic shared
storage**, without CTA barriers. Each original32-key partial accumulator passes
through four ordered8-key owners, then uses the original ascending four-part
fold. The fork's different reduction and decay arithmetic are not installed.

Six complete-convolution cells pass normalization on/off and offsets0/+5/-5,
with24 dirty states and48 retained replays. Output and logical state words are
exactly equal. State permutation occurs only in test setup and terminal
observation; the two captured kernels contain no conversion. Production must
implement the persistent format coherently before this becomes dispatchable.

| Complete convolution + recurrence, µs | Original control | Transposed186 |
|---|---:|---:|
| Median,12 alternating pairs |15.3510|11.1004|
| Sample range |15.3350–15.3575|11.0632–11.1275|

This is **27.689% less complete-chain time**, using64 operations per graph,
five warmup pairs and byte observations before/after timing. It is a synthetic
component result, not a new model score. The actual final timed banks are
observed before replay or reset. The original compiled programs, including185d,
remain byte-identical in bodies and resources.

DK128 uses54/56 used/allocated VGPRs; DK64 uses34/36. Both have zero spills or
private storage. Six separate exact-dispatch profiles and native/compiler joins
pass: **384 wavefronts**, **502.0 VALU instructions/wave**, **778.5 KiB fetched**,
**42.193% VALU utilization**, **17.976% memory-unit busy**, and **35.343% L2 hit**.
These diagnostic counters do not establish achieved occupancy or unprofiled
model-cache behavior. In particular, four times as many waves execute compared
with the original96-wave kernel; fewer instructions per wave is not fewer
total instruction issues. Ordered handoffs and repeated per-wave normalization
remain costs. Producer-prepared Q/K is the next separate complete-boundary test.

The archive is `/tmp/qfn-gdn186-component-green-r1`, including exact sources,
copied runtime, build and timing/profile receipts. The result root contains
`gdn186-fixture-release-build-r1`, `gdn186-functional-r1`, `gdn186-m1-events-r1`
and `gdn186-m1-candidate-recurrence-{valu,fetch,waves,utilization,memory-busy,l2-hit}-r1`.
The expanded transposed gate passes **272 serial/grouped cases plus 16 actual
three-request cases, totaling 2,464 new captured replays**. It covers DK64/128,
DV1/65/128/129, normalization on/off, offsets and integer-boundary offsets,
M1–16 serial/grouped comparisons, full/partial/empty/reset lifecycles, in-place
and distinct destinations, padded retained snapshots, and null versus device
live-count inputs. The original checkpoints remain unchanged. The receipt is
`gdn186-extended-functional-r1`; the fixture-only archive is
`/tmp/qfn-gdn186-expanded-green-r1`, referring to the frozen component core.
Production remains checkpoint180; prefill integration, cache-format identity,
model accuracy and matched model throughput are still required.

### Prepared transposed186b and eight-lane186c screens, 2026-10-24

The same Release binary contains two distinct follow-ups. Candidate186b uses
original185's convolution producer to publish compact normalized/scaled Q and
normalized K, followed by the sixteen-lane transposed recurrence. Candidate186c
keeps the original convolution and raw Q/K, with eight lanes per state column.
Both preserve the original four sequential key partitions and ascending fold.
No production dispatch, state owner or cache layout has changed.

| Same-binary complete convolution + recurrence | Original control, µs | Candidate, µs | Less time |
|---|---:|---:|---:|
|186b, prepared Q/K, sixteen lanes|15.34885|10.5875|31.021%|
|186c, raw Q/K, eight lanes|15.3300|12.4875|18.542%|

Each median has 12 alternating pairs, five warmup pairs and 64 operations per
retained graph. Each initial functional gate passes six normalization/offset
cells, 24 dirty states and 48 captured replays. The actual timed state/output
banks are checked before reset or additional replay. These are synthetic
complete-boundary measurements, not model throughput. The prepared variant is
the candidate for broader qualification; the eight-lane variant is parked.

At DK128,186b uses 42/44 used/allocated VGPRs, versus186c's 89/92. At DK64,
the counts are 30/32 and 54/56 respectively. Both recurrences have zero static
or dynamic LDS, zero private storage and zero spills. All 15 pre-existing
compiled role bodies/resources remain identical after the combined build.

Exact singleton recurrence profiles report:

| Diagnostic counter |186b|186c|
|---|---:|---:|
|Wavefronts|384|192|
|VALU instructions per wave|343.0|544.0|
|Fetched KiB|778.5|778.5|

The eight-lane program issues fewer total vector instructions but runs slower;
register pressure, occupancy capacity and dependency length matter alongside
instruction count. These counters do not by themselves isolate causality.
Prepared186b reports35.458% VALU utilization,18.793% memory-unit busy and35.348%
L2 hit in separate diagnostic runs. The prepared convolution uses40waves,
120.6 VALU instructions/wave,92.5KiB fetched and16B static LDS. Its recurrence
has no shared memory; the complete two-kernel pipeline is not wholly LDS-free.
The actual prepared-plane allocation is8192B including guards, with4096B live.

All build, functional, timing and exact-dispatch joins pass in
`gdn186bc-fixture-release-build-r1`, `gdn186{b,c}-functional-r1`,
`gdn186{b,c}-m1-events-r1` and their `candidate-{role}-{counter}-r1` directories.
The immutable component archive is `/tmp/qfn-gdn186bc-component-green-r1`.
The broader186b gate now passes **272 serial/grouped cases and16 actual packed
R3 cases, with2,464 recurrence graph replays**, in38.71seconds. Every phase
publishes Q/K through the actual convolution producer and compares its raw output
and history/snapshots with independent original convolution. Recurrence compares
against original planar, merged, historical and serial references. The producer
API admits one request with M1–16, so each diagnostic request has its own producer;
the packed test then executes a real R3/M1 recurrence. These correctness graphs
are distinct from the two-node complete-boundary timing graphs. The receipt is
`gdn186b-extended-functional-r1`; fixture archive
`/tmp/qfn-gdn186b-expanded-green-r1` refers to the frozen186bc core. Persistent
prefill/cache-format integration, canonical model accuracy and matched model
performance remain required.

### Transposed M512 prefill endpoints187, 2026-10-24

The first prefill program over the same value-major state is byte-correct at
R1/M512/H12/DK128/DV128. It retains the original resident recurrence loop,
normalization/gates, sixteen-row Q/K/V tile and four-part reductions. Only
persistent state endpoints and snapshot indices differ: aligned float4 loads
and final stores replace scalar strided endpoints; padded snapshots use safe
scalar stores. No conversion, extra graph node or persistent state bank is added.

Twelve cases cover normalization on/off, in-place/distinct state, and terminal-only
or one/seventeen retained snapshots. Full512→short17→empty0→full512 dirty replay
passes48 states and96 graph launches. This is ordinary finite input plus authored
signed-zero coverage, not a dedicated subnormal gate. Original output, logical
terminal/snapshot words, immutable sources and all guards remain exact.

The captured planar recurrence's twelve alternating pairs have medians
**991.751µs control /975.9115µs candidate**, or **1.597% less time**, with ranges
990.391–995.511 and972.471–978.231µs. This is one M512 recurrence, not complete
convolution or model prefill throughput. Actual final timed buffers are checked
before reset/replay. Compiler resource use is identical: terminal123/124 used/
allocated VGPRs, snapshot120/120,8B static plus28,736B dynamic LDS; no private
storage or spills. All seven frozen180 control bodies/resources (including both
prefill programs) and all nineteen earlier private roles remain unchanged.

Receipts: `gdn187-prefill-release-build-r1`, `gdn187-prefill-functional-r1`,
`gdn187-prefill-m512-events-r1`; archive
`/tmp/qfn-gdn187-prefill-component-green-r1`. Exact isolated profiler evidence,
generic M/R/DK/DV routes, grouped state production, full persistent-layout identity,
model accuracy and matched Release throughput remain outstanding. This deliberately
scoped prototype does not advertise a complete production layout implementation.

### Historical176 and177 trials

The following records describe prototype qualification before production wiring.
The176 preparation variants were rejected for scalar decode and their machinery
has been removed from the production source.

Candidate176 tests a complete existing deinterleave-plus-recurrence boundary.
Its preparation kernel publishes canonically normalized Q/K and rounded gates
once per head/row; the recurrence consumes those values with the original state
arithmetic. It adds no launch, but needs two explicitly owned gate planes before
any production installation. That prototype had no model-stage wiring.
The guarded Release build passes, and the complete-pair byte trial plus both
historical scalar/grouped tests pass. Retained full→partial→empty→full replays
preserve outputs, terminal state, snapshots and raw V bytes. This is a scoped
H12/keyH4/DK128/DV128 test, not full geometry qualification.

|176 complete two-node pair | Raw control µs | Prepared candidate µs | Latency change |
|---|---:|---:|---:|
| M1, live1 |14.47625 |14.61625 |+0.967% |
| M16, live16 |118.925 |97.0898 |−18.360% |
| M16, live15 |112.725 |92.0399 |−18.350% |
| M1, live0 |8.01123 |7.50499 |−6.319% |
| M16, live0 |14.85 |14.36495 |−3.266% |

Each cell includes five warmups, twelve alternating complete-pair event
samples and byte checks before/after timing. The one-row result does not qualify
the candidate for the decode target. Twelve exact-role profiles pass. Recurrence
VALU instructions per wave fall805.5→601.0, with96 waves and essentially unchanged
FetchSize786.5→786.375KiB. Preparation changes72→48 waves,99→171.25 VALU
instructions per wave and11.625→10.4375KiB fetched. State traffic is retained.
Prepared recurrence uses111/112 used/allocated VGPRs,7184B dynamic LDS and zero
scratch/spills. Static and dynamic LDS are checked separately: rocprof's CSV
LDS field describes the static allocation, not the runtime dynamic allocation.

The raw175/176 ISA comparison finds a real control change: the factored raw
recurrence changes126/128→124/124 used/allocated VGPRs and6612→6532 code bytes.
Its deinterleave body is identical. Historical numerical tests pass, but the
timing table is a same-source control comparison, not frozen175 performance.
A separate old-build complete-boundary control is required before promotion.
Evidence: `gdn176-component-build-r1`, `gdn176-functional-r1`, and
`gdn176-complete-pair-summary-r1.json` under the QFN local result root.

The176b wave-local four-part fold removes two barriers while preserving each
ascending reduction from positive zero. It remains byte-exact but increases
M1 complete-pair time to14.93135µs against14.47135 control.176c transposes shared
Q/K addresses, removing measured LDS conflicts2.82925→0, but takes15.0038µs
against14.4688 control. Allocated VGPRs remain112 and state traffic is unchanged.
Both retain zero spills and the same raw/control/preparation bodies as176.
The ISA shows native DPP folds and their dependency waits; it does not show
loss of packed LDS loads. Neither variant qualifies for single-token decode.
Their M16 results improve to95.74525/95.8103µs against approximately118.9µs;
this grouped result is separate from the active decode target. Evidence:
`gdn176{b,c}-*` and `/tmp/qfn-gdn176-bc-rejection/REPORT.md`.

Candidate177 instead reads original merged post-convolution QKV directly. It
retains raw normalization, gates and recurrence arithmetic, and removes the
deinterleave launch and expanded-plane traffic. The current prototype needs no
new gate planes or workspace and initially had no production-stage wiring. Its guarded
Release build and three focused native tests pass, including historical
scalar/grouped checks and exact complete-boundary outputs/state/snapshots for
M1/M16, offsets0/+5/−5 and dirty full/partial/empty replay.

|177 complete boundary | Same-source two-node control µs | One-node merged candidate µs |
|---|---:|---:|
| M1, live1 |14.4825 |12.10875 |
| M16, live16 |118.905 |115.195 |
| M16, live15 |112.745 |109.06 |
| M1, live0 |8.022555 |6.47129 |
| M16, live0 |14.8251 |12.485 |

The M1 component gain repeats at14.46125→12.10000µs. A separate captured
complete-boundary A/B/A against the immutable175 binary measures14.53395 and
14.52420µs for the old path, versus12.10125µs for177: a16.71% reduction against
the14.529075µs control median. The same-source raw control is14.475µs and is
kept distinct from that frozen baseline. These are component measurements;
whole-model throughput and production integration were still open at this point.

The extended native gate passes261 cells spanning every M1..16, DK64/128,
DV64/65/128/129, both normalization policies, three independent requests,
request holes and signed modular head offsets including INT_MIN/INT_MAX.
Four retained dirty replays per cell compare complete output, state and
snapshots against independent historical grouped and serial-row kernels.
An additional12-cell M1 test passes12 successive live/empty transitions with
the same persistent state bank as both input and output. The original extended
reporting script incorrectly expected a flattened deinterleave grid; its
offline recheck records the correction to per-request X extent and request Y
extent, preserving the original failed report and passing native run.

Exact-role rocprof joins report96 waves,796.5 VALU instructions per wave,
778.5KiB fetched and zero LDS bank conflicts for merged recurrence. Bounded
ISA inspection reports119/120 used/allocated VGPRs,95/96 SGPRs,7184B dynamic
LDS and zero private memory or spills. This retains the same register-limited
occupancy capacity as the original kernel; achieved occupancy was not measured.
CFG analysis proves signed modulo and invariant input-address setup remain
outside the causal row loop. The raw177 control body is byte-identical to176r2.

Evidence: `gdn177-frozen175-component-comparison-r1.json`,
`gdn177-extended-r2-recheck-r1.json`, `gdn177-inplace-r1`, and
`/tmp/qfn-gdn177-isa-private/REPORT.md`. No model speedup or default promotion
is claimed from these component results alone; final qualification is above.

The active TP4 performance target remains Castagna's42.959 decode tok/s while
preserving roughly815 prefill tok/s. The earlier recurrence-only candidates
below isolate the FP32 recurrence in `ROCmGatedDeltaNetKernels.hip`; their event
intervals exclude QKV preparation, projections, convolution, output normalization,
communication and the model. Candidate177's complete-boundary and model timings
are reported separately above.

QFN's ordinary four-MI50 TP4 plan gives each participant12 value heads,
DK128 and DV128. The original recurrence-only fixtures use an immutable initial
bank distinct from published state. Their one-row lane writes terminal state;
the16-row verifier lane writes every row snapshot and leaves the terminal bank's
sentinel untouched. They use the real public recurrence launcher and retained HIP
graphs. Candidate177 additionally qualifies ordinary in-place state and retained
verifier destinations as described above.

## Independent baseline and regression

`ROCmGDNHistoricalRecurrentOracle.hip` preserves the literal original kernel in
a separate test translation unit. Before changing production, the native
historical tests passed against it. Coverage includes all M1..16, DK64/128,
DV64/128 and the65-column boundary, two unequal requests, full/partial/empty
live lengths, signed/zero inputs, dirty retained storage and20 M16 replays.
Every output, complete snapshot and terminal state is compared byte for byte
with both historical grouped execution and historical serial-row execution.
The explicit preflight registration is
`V2_Integration_ROCmGDNPairedQKHistorical`.

The separate Release performance fixture records64 complete recurrence nodes
inside one retained graph. After byte sanity and five warmups, twelve native
event pairs alternate production/historical order. It repeats the byte sanity
after timing. Native graph inspection proves function identity, every node's
geometry and a single ordered dependency chain. Setup, transfers and result
collection are outside the event interval. An isolated profiler process keeps
one selected family, five warmups and the sixth captured dispatch.

## Measured candidates

The unchanged control and literal historical kernel agree in timing as well as
bytes. Each candidate is paired with that same historical implementation in
its process; small changes in historical time expose the run's noise.

| Candidate | M1 production / historical µs | M16 production / historical µs | Native byte gates |
|---|---:|---:|---|
| Original256-thread recurrence | 12.639969 /12.639969 | 122.651085 /122.657336 | Pass |
|116: paired Q/K normalization | 12.305344 /12.654094 | 119.508445 /122.679703 | Pass |
|117: paired normalization,128 threads | 12.197657 /12.661414 | 115.625329 /122.685384 | Pass |

116 keeps each original square, wave64 XOR reduction, ascending four-wave
fold and final quotient/multiply expression. Independent Q/K work shares
publication edges, removing three CTA barriers per live row. K's temporary
four-wave totals borrow the existing recurrence reduction storage before that
storage's later use; the existing publication barrier retires the readers.
There is no new allocation, launch, global workspace or state authority.

117 keeps four K-row partitions and gives each CTA32 value columns instead of64.
The TP4 shape therefore launches48 rather than24 CTAs. The original norm
lanes0..127 retain their wave64 grouping; the absent old waves2/3 explicitly
publish their exact +0 totals before the same four-term fold. Per-column state,
reduction and output arithmetic are unchanged. Dynamic LDS falls13328→7184B
at DK128, and2576→1552B at DK64. External state/snapshot/workspace sizes stay
fixed. More CTAs duplicate more per-head norm/gate work, so broader shape
measurements are required before choosing this geometry globally.

The first117 build incorrectly changed the second HIP launch bound from2 to4
using a CUDA block-count interpretation. The mandatory spill guard rejected
DK64 with15 spill slots/88B and DK128 with35 slots/220B; that executable never
ran. HIP's second bound is minimum waves per execution unit, as the selected
SDK's `amd_hip_runtime.h` attributes and
[AMD's language-extension documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html#launch-bounds)
confirm. Restoring2 preserves the original register constraint and compiles
without spills. The guard was not changed or bypassed.

## Evidence and remaining qualification

The baseline profiler reports124 VGPRs,96 SGPRs and no private storage for the
DK128 recurrence;116 reports128/96/0. Both retain the original grid and dynamic
LDS. The corrected117 profile reports128 allocated VGPRs,96 SGPRs,7184B
dynamic LDS and zero private storage; the native function attribute reports126
used registers. Its48 two-wave CTAs still execute96 wavefronts, matching the
original24 four-wave CTAs. These are resource and work observations, not
achieved occupancy. Counter collection is separate from unprofiled timing.

The118 breadth sweep covers32 cells: DK64/128, DV64/128,6/12/24/48 local value
heads, and M1/16. Every cell passes full output/state/snapshot byte comparisons
before and after timing. Thirty cells improve; DK128/DV64/H48/M1 is0.312%
slower, and DK128/DV128/H48/M16 is3.825% slower. The latter is a real limitation
of choosing128 threads globally, not evidence for a universal win. The current
TP4 geometry improves3.878% at M1 and5.609% at M16 in this sweep. These are
recurrence-only measurements; they do not certify entire topologies.

The unchanged185-row real-model panel also passes its required short and
held-out prefill/decode groups. All26 retained logit files are byte-identical
to the accepted observation115 control, including threshold and reset rows.
Threshold rows remain diagnostic rather than acquiring a new absolute KL
acceptance claim. The matched N512/D256 Release A/B/A is complete. Six control
samples have median 815.591867 prefill / 39.254592 decode tok/s; the three
candidate samples have median 815.849609 / 39.237071. The changes of +0.0316%
and -0.0446% are within the observed variation. All nine 256-token sequences
are identical, and every runtime, source, loader, driver and retirement check
passes. This is neutral model throughput, not a model-level improvement.

The complete ROCm GDN fixture's30 tests and the two explicitly selected
historical tests pass for both116 and corrected117, including long-prefill/
decode equivalence and captured state publication. Each CTest slice also
includes its required startup fixture. Driver windows and normal retirement
pass. The subsequent observation120 gate refresh passes776 Unit,46 affected
Integration and one setup registration, including these production GDN edits.
Full production preflight remains G60.

Local evidence under `parity-results/qwen-flash-next/`:

- `gdn116-baseline-m{1,16}-events-r1` and baseline Release build/archive.
- `gdn116-paired-native-r1`, `gdn116-paired-m{1,16}-events-r1`.
- `gdn117-128-integration-build-r1`: preserved rejected spill result.
- `gdn117-128-integration-build-r2`, `gdn117-128-native-r1` and Release build.
- `gdn117-128-m{1,16}-events-r1`.
- `gdn{116-baseline,116-paired,117-128}-m1-wavefronts-r1`: exact profile receipts.
- `gdn118-128-breadth-r1`: all32 event cells and their independently checked
  byte, native geometry, source, loader and retirement evidence.
- `gdn118-release-accuracy-r1`: canonical required groups and the complete
  `observation115-logit-byte-comparison.json`.
- `gdn118-model-summary-r2`: complete matched comparison, settings, samples
  and exact generated-token agreement. The initial wrapper failed before
  native launch because its local prompt copy was missing; the corrected run
  uses a fresh package and output directories rather than overwriting it.

The121 test-only extension adds independent historical scale-only coverage
beside L2 normalization. Both policies pass the focused padded-real-length
and historical registrations in18.34 seconds, including repeated dirty
replays. The proposed additional Q/K-load barrier removal remains unapplied.
Further GDN tuning is held while attention takes priority. The larger grouped
shape regression remains open before selecting a global geometry; no final
production promotion, refreshed full preflight or feature checkpoint follows.
