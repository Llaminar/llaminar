# QFN captured compute scaling

Updated: 2026-10-23. Current A1 checkpoint **L102** measures **815.053 prefill /
37.424 decode tok/s** against matched **814.156 / 37.134**: **+0.110% /
+0.781%**, with nonoverlapping decode ranges and effectively unchanged prefill.
All 185 final-logit rows and twelve generated sequences remain byte-identical;
all 776 Unit and focused native gates pass. See the
[captured floating bindings qualification](#floating-projections-immutable-captured-bindings-l102).
Complete production preflight remains G60.

The preceding checkpoint **L101** measures **814.610 prefill /
37.088 decode tok/s**, effectively unchanged versus matched L100 **814.645 /
37.103**. Its exact-order branch norm improves component M1 latency 2.794%; see
[qualification below](#branch-normalization-original-reduction-order-with-fewer-barriers).
The preceding throughput win is **L100**, **814.527 prefill /
37.063 decode tok/s**, versus matched archived L97 **814.133 / 36.824**:
**+0.048% / +0.648%**, with nonoverlapping decode ranges. The original precision,
arithmetic and placement settings remain fixed; all 185 final-logit rows and
twelve natural sequences match exactly. See the [dashboard](QFN_TUNING_DASHBOARD.md)
and [matched Release comparison](../../../../parity-results/qwen-flash-next/reducer100-model-comparison.json).
The grouped-reducer [recorded GDN decode bundle](#grouped-ordinary-m1-reducers-l100)
improves TP1/TP2/TP4 latency 5.019%/5.684%/3.125%, but its matched concurrent
scaling remains **1.304× TP2 / 1.710× TP4**, missing both targets. The recorded
vocabulary-projection local-shape and concurrent TP1/TP2/TP4 measurements remain
green. Full production preflight remains pending; G60 is the last complete gate.

The user selected **preserve recorded inputs and expert routes**. The diagnostic
will measure the original production compute without executing communication
inside its timed captured component. It will not run a generation graph with
invalid collective outputs. The old `LLAMINAR_SKIP_ALLREDUCE` switch is incomplete
and changes downstream activations; it cannot answer this question.

## Measurement contract

The targets are independent cumulative speedups: **T1/T2 ≥1.85** and
**T1/T4 ≥3.6**. Report absolute latencies and every raw sample. For actual TP,
take the slowest participating device in each complete matched round, then the
median of those rounds. Report both the original operation and physical local
work, source format/prepared layout, arithmetic policy, rank/device membership,
captured entrypoints and stream/event boundaries. TP2→TP4 is descriptive only.

An isolated local shape on one device is labeled `local_shape`, even when its
dimensions correspond to TP4. It does not prove concurrent four-device scaling.
Actual TP requires all participant observations and identical round sets at
degrees one, two and four. This host has four MI50s and two RTX 3090s; a CUDA
four-device result cannot be fabricated from a one-device shape sweep.

Partitioned work has a useful ideal T1/P comparison. Explicitly replicated
norms/router/control operations and communication retain their costs and
ownership labels. Their local shape measurements are unassessed, never green
scaling passes. Actual TP target misses remain visible for every declared
operation. Empty/skewed expert owners must not make a different workload look
faster. Family times and operation constituents may overlap; their excess over
ideal scaling is a prioritization aid, not additive model savings.

## Preserve valid work

Start with a small prepared GEMM adapter using the existing `ComputeGraph`,
`DeviceGraphExecutor::recordSetupGraphMaterialization`, graph capture, prepared
weight store and admitted arena. A fresh diagnostic graph records the same
production stage. It has no communication nodes; the original communication's
valid outputs are already retained inputs. Ordinary inference graphs and
their publication semantics remain unchanged.

Bind actual recorded input storage, its producing snapshot/event and complete
physical identity. A text provenance label is insufficient. Check physical
extents and aliases against the stage's real bindings. Retain prepared weights,
workspace generations and source precision. Reject unsupported hidden reads:
the ordinary GEMM buffer contract alone does not describe previous C when beta
is nonzero, raw bias, count pointers or arbitrary fused inputs. Expand support
only with explicit state ownership and a native byte regression.

Mutable inputs and recurrent state require admitted same-device restore banks.
Restore on the exact stream outside the compute interval. Replay cannot advance
the model's sampler, KV history, route choices or request controller. No host
count readback is inserted into production inference. Missing input/reset
ownership, foreign storage, unsupported protocol stages and stale generations
fail before capture. No node-name filter substitutes for a complete contract.

The full model need not fit on one accelerator to measure its original TP1
operation. The vocabulary projection is an initial candidate: its original
248320×2560 source and the actual 124160/62080-column TP slices fit as isolated
prepared operations. Confirm its exact production stage and arithmetic policy
before using it. Synthetic primitive fixtures qualify the mechanism but cannot
be reported as recorded real-model work.

The original canonical ROCm TP4 observation used a **mirrored full-vocabulary
head**, physical N=248320 with `serial_equivalent_partition_width=62080`.
MTP is disabled. The isolation adapter must preserve that explicit arithmetic
partition while testing physical widths 248320, 124160 and 62080. These are
local shape experiments; that observation's replicated head work is not
reported as sharded TP speedup. The byte oracle is the real `LMHeadStage` plus
its original exported `LM_HEAD`, with `FINAL_NORM` as input.

## Coverage and prioritization

Use the [accepted-runtime profile](../../../../parity-results/qwen-flash-next/moe-model90-four-profile-analysis-r1/REPORT.md)
to choose cohorts. It authenticates recorded execution and overlap; profiler
duration is not canonical throughput. Extend the initial adapter through the
existing component fixtures, without adding another serving orchestrator:

| Cohort | Work and state that must remain fixed |
|---|---|
| Dense quantized projections and reductions | Original source slices, activation quantization and arithmetic partitions; genuine prepared layout |
| Floating/router/alpha-beta and TwoTerm projections | Frozen arithmetic policy and input; retain replicated ownership explicitly |
| Routed gate/up and down | Original expert IDs/weights, live counts, owner map and imported intermediate packets; empty/skewed owners |
| Attention and GDN | Head assignment, original position/context, KV/recurrent state and exact reset; no synthetic shorter history |
| Norms, activation preparation and layout | Actual live tensor extents and aliases; replicated versus sharded consumers |
| Ngrams, indexer and other service-dependent stages | Explicit input/service boundary; reject isolation until its complete state contract exists |

The next recorded boundary is the complete layer-zero GDN projection bundle.
The fresh `recorded-gdn96-native-input-r1` canonical TP4 observation passes all
eight CSV checks, prefix restoration and driver/retirement checks. It exports
the actual normalized input and QKV/Z/alpha/beta outputs for nine prefill rows
and two individual decode steps. Those nine rows are not an M512 performance
workload. The original QKV/Z sources are Q8_0; alpha/beta arrive as FP32 and
retain the selected prepared FP16 policy. Native submission/capture records
identify a QKV+Z subgroup (local N=2560+1536, K=2560) and an alpha+beta subgroup
(N=12+12). Their counts are not device replay counts.

The isolation extension must retain both fused subgroups. GDNHeadAssignment
owns the original packed Q/K/V and modulo-linked value-head spans, including
Z/alpha/beta; equal contiguous row slicing would test different ownership.
Every degree must match the saved original output bytes before timing. Actual
TP1/TP2/TP4 GDN decode measurements are reported below. The nine-row TP2
prefill comparison remains unqualified because its selected floating reduction
differs from the original TP4 observation; head results do not qualify GDN.

For each supported cohort, cover decode and production prefill shapes plus
partial buckets. Extend reusable kernel improvements across applicable physical
formats, not only the current GGUF's formats. CPU/CUDA/ROCm share the diagnostic
contracts; a backend-specific implementation or unavailable device count is
explicitly pending until exercised.

## Gates and current evidence

### Recorded GDN projection measurements

Slice98 adds the complete original `GDNProjectionStage` to the existing recorded
projection executable. It retains production source slicing, prepared engines,
both fused subgroups, exact streams, captured compute and separate captured
restoration. Shared head/GDN runtime code owns cohort admission, complete rounds
and retirement. It is a diagnostic, with no production arithmetic change.

The rebuilt Unit suite passes **776/776**, zero skips, and all **17** focused
registrations pass. Native CUDA and ROCm fixtures compare independent complete
captures at M1/M9 and every local TP1/2/4 participant, including twenty dirty
input/output/workspace histories. Initial failures exposed diagnostic setup
defects: immutable Q8 descriptors legitimately omit reusable-slot capacity,
and the cold stage must call its canonical launch preparation before capture.
Metadata regressions retain semantic boolean fields and empty local inventories.
The existing real-input TP4 vocabulary-head cohort also passes after the shared
runtime refactor. Evidence: `recorded-gdn98-focused-r1`, `recorded-gdn98-unit-r1`,
`recorded-gdn98-release-build-r1` and `recorded-head98-compat-r1`.

Every decode cohort below passes all four original output byte oracles, original
input restoration, twenty dirty histories and twenty composed-parent histories
on every participant. Each has eight warmups and 31 measured complete rounds;
the reported value is the median of each round's slowest participant. Drivers
and owned native retirement are clean. K=2560; original QKV/Z widths are
10240/6144, and alpha/beta widths are 48 each. Native Q8 sources and the selected
half-RNE control policy remain unchanged.

| Recorded input | TP1 µs | TP2 µs | TP4 µs | TP1/TP2 | TP1/TP4 | Targets |
|---|---:|---:|---:|---:|---:|---|
| Decode 0, M1 | 89.285 | 64.161 | 51.041 | 1.392× | 1.749× | Both missed |
| Decode 1, M1 | 89.280 | 64.320 | 51.520 | 1.388× | 1.733× | Both missed |

The [raw cohorts and scaling report](../../../../parity-results/qwen-flash-next/recorded-gdn98-scaling-r1/index.json)
retain all participants and samples; the existing reporter produces the
[scaling chart](../../../../parity-results/qwen-flash-next/recorded-gdn98-scaling-r1/scaling.svg).
These are complete projection-bundle compute intervals, excluding restoration
and communication. They are not whole-layer, recurrent-state, token-throughput
or pure per-kernel measurements. Native graph scheduling gaps may be included.

The actual decode graph contains seven kernels: one Q8 quantizer, separate
QKV/Z producer and ordered-reducer pairs, one half conversion and the combined
short-half alpha/beta projection. Q8 producers use TN128/CPT1 and sixteen K
partitions at all measured degrees. On TP4 their grids are 20×16 and 12×16;
the half group uses 24 blocks of 128 threads. All inspected leaves report zero
local-memory bytes.

The subsequent [isolated profile report](../../../../parity-results/qwen-flash-next/recorded-gdn98-profile-summary-r1/report.json)
retains fifteen separate exact-symbol/shape trace launches and 22 singleton
counter observations. Each native run preserves the original decode0 byte
oracles and passes driver/retirement checks. These are **instrumented local
shape** measurements on one MI50, not the concurrent TP cohort timings above.
Each timing keeps all 31 samples after twenty proof replays and eight warmups.
Same-symbol QKV/Z kernels are separated by their actual launch geometry; each
counter is joined to one warmed native dispatch. The initial rejected trace
receipt remains excluded: HIP's function name and HSA's `.kd` descriptor now
have an explicit mapping authenticated by the loaded code-object metadata.

| Selected kernel | TP1 shape µs | TP2 shape µs | TP4 shape µs | T1/T2 | T1/T4 |
|---|---:|---:|---:|---:|---:|
| QKV producer | 36.16 | 19.20 | 10.08 | 1.883× | 3.587× |
| QKV ordered reducer | 6.56 | 5.28 | 4.16 | 1.242× | 1.577× |
| Z producer | 22.88 | 12.32 | 7.36 | 1.857× | 3.109× |
| Z ordered reducer | 6.08 | 4.32 | 4.16 | 1.407× | 1.462× |
| Combined half alpha/beta | 7.52 | 7.52 | 7.52 | 1.000× | 1.000× |

The producer inner loop scales much better than the complete bundle. The
reducers and tiny floating controls therefore merit attention before changing
Q8 arithmetic. The half group launches 48 wavefronts at TP4, with 16 allocated
VGPRs, 48 SGPRs, 1 KiB LDS and zero scratch. QKV producer/reducer launch 640/40
wavefronts, use 20/24 allocated VGPRs and no LDS or scratch. Active-vector-lane
utilization is 99.79%/99.12%/100% for half/producer/reducer; all three report
zero LDS-bank-conflict stalls. Those counters do not measure achieved occupancy.
Fetched memory falls with partition size: QKV 27321.875→6803.188 KiB and its
reducer 640.063→160.063 KiB; half controls 485.500→125.750 KiB. Cached output
stores can report zero video-memory write traffic within a dispatch, so that
counter is not an output-store count. Do not sum leaf medians into a model
critical path or use profiled times as canonical throughput.

The first one-wave half candidates are rejected on measured economics. Four
original stride256 accumulators per lane preserve the original tree without
LDS or barriers, but scalar loads introduce additional memory waits. Native
ISA inspection also exposed a numerical defect in the initial candidate:
the actual HALF oracle requires activation-before-weight FMACs and
low-before-high additions, including NaN priority. Its contract differs from
the earlier FP32-activation helper. Explicit native operand order restores
every exact-byte test without relaxing comparison.

| Recorded decode0 bundle candidate | Matched TP4 control µs | Candidate µs | Latency change |
|---|---:|---:|---:|
| Corrected one-wave scalar loads | 51.200 | 65.120 | +27.187% |
| Unconditional full chunks, checked final tail | 51.600 | 70.560 | +36.744% |

Both corrected variants pass all six focused native registrations. The first
also loses at TP1/TP2 by 12.522%/10.545%; TP2 samples retain their wider spread.
These are unprofiled A/B/B/A concurrent bundle measurements, 31 rounds per
arm, using the immutable original Release control. Every original output byte,
dirty replay and native retirement check passes. Evidence is
`half-wave99-component-abba-r1` and `half-wave99-component-abba-r2`; the
second disposition and source patch remain in `half-wave99-retired-scalar-r1`.
No full-model benchmark was run for these losing candidates.

The corrected scalar kernel alone measures 18.88 µs under the isolated TP4
trace, versus the original 7.52 µs, despite nearly identical fetched memory
(125.250 versus 125.750 KiB). It has zero LDS/scratch but eight scalar loads
and eight waits per generic K-loop iteration. Removing guards reduces loop
instruction count from 105 to 85 yet serializes those loads further. The
original uses two packed loads per wave and one wait per wave. High active-lane
utilization does not establish GPU-wide saturation.

The original half kernel remains in production. Expanded actual-half tests
retain K2559/2560/2561, signed-zero and paired-NaN reduction-edge witnesses,
raw half encodings and exact captured replay.

### Grouped ordinary M1 reducers (L100)

L100 retains the original activation quantizer, QKV/Z producers, source
formats, conversion policies, K partitions and per-column ascending-partition
FP32 additions. Typed deferred tickets retain the existing four admitted
partial slots; ordered waves retire reducers and bias before slot reuse.
Only compatible original tile geometries share a reducer. Explicit concurrent
execution keeps precedence, and mapped destinations keep their original
serialized publication. CUDA's existing fused-KPAR path is unchanged; this
measured opportunity concerns ordinary direct-output ROCm M1.

The [unprofiled recorded decode0 ABBA](../../../../parity-results/qwen-flash-next/grouped-reducer100-component-abba-r1/result.json)
uses the immutable frozen98 component archive (original L97 arithmetic),
31 complete matched rounds per arm after eight warmups, original output-byte
oracles and twenty dirty/composed histories. Latencies are medians of each
round's slowest actual participant; every raw sample is retained.

| Degree | Matched original µs | L100 µs | Latency change |
|---|---:|---:|---:|
| TP1 | 89.280 | 84.799 | −5.019% |
| TP2 | 68.960 | 65.040 | −5.684% |
| TP4 | 51.200 | 49.600 | −3.125% |

L100's T1/T2=1.304× and T1/T4=1.710× still miss 1.85×/3.6×. These are a
new matched cohort, not replacements for the earlier recorded98 samples;
TP2 retains wider variability. The graph has six kernels instead of seven:
the separate QKV/Z reductions become one grid32/block128 launch at TP4.
Intervals exclude restoration and communication but include native graph
scheduling gaps. No leaf-median sum is a model critical path.

The [isolated grouped-reducer profile](../../../../parity-results/qwen-flash-next/grouped-reducer100-profile-summary-r1/report.json)
selects exact `nativeM1GroupedReduce<128,1>` with grid[32,1,1], block[128,1,1].
Its instrumented median is 7.040 µs (6.880–8.320), allocated VGPR24/SGPR48,
LDS 0/scratch 0, 64 waves and 256.125 KiB fetched. The native occupancy query
reports a potential maximum of 16 active 128-thread blocks per multiprocessor;
the actual grid has only 32 blocks. Active VALU utilization 98.785% is not
achieved occupancy. Source-bound gfx906 static inspection of all five
compiled TN/CPT leaves confirms zero SGPR/VGPR spills and zero private/LDS bytes,
with original native add operand order. This does not certify another GPU
architecture or turn instrumented leaf time into model savings.

The [final focused gate](../../../../parity-results/qwen-flash-next/grouped-reducer100-native-r2/result.json)
passes 12 registrations, including the all-format captured original-native
partial oracle, dirty histories, mapped outputs, bias, aliases and 2/4/5/8-slot
waves; [complete Unit](../../../../parity-results/qwen-flash-next/grouped-reducer100-unit-r1/result.json)
passes 776. The subsequent [all-five scoped-Q8 grouped capture extension](../../../../parity-results/qwen-flash-next/grouped-reducer100-scoped-native-r1/result.json)
and slot/bias neighbor pass in 6.57 seconds. Every compiled geometry is exercised
at K32 direct publication, K640 with six empty KB16 planes and K2560 with a
populated tree, including odd output tails, output/partial guards, actual
native launch inventory and retained replay after the diagnostic scope retires.
NaN, infinity and signed-zero witnesses preserve the frozen original RN fold.
This is a test-only extension: [all four production source pins and all 57
immutable runtime metadata entries](../../../../parity-results/qwen-flash-next/grouped-reducer100-test-extension-production-identity.json)
still match the accepted model observation; the production binary was not rebuilt.
The [Release build](../../../../parity-results/qwen-flash-next/grouped-reducer100-release-build-r1/result.json)
retains the spill guard. The current CMake ROCm target set is gfx906, which was
compiled and inspected; CUDA kernel source is unchanged.

The separate [full-model Release ABBA](../../../../parity-results/qwen-flash-next/reducer100-model-comparison.json)
uses the actual archived L97 serving runtime and its original accuracy/source
receipt, not the component-only frozen98 archive. Four clean arms retain twelve
identical natural 256-token sequences; [26 logit files / 185 rows](../../../../parity-results/qwen-flash-next/reducer100-release-accuracy-r1/logit-byte-comparison.json)
are byte-identical. Candidate 814.527/37.063 versus control 814.133/36.824 tok/s
is +0.048% prefill / +0.648% decode; decode ranges 36.931–37.230 and 36.810–36.907
do not overlap. Model, source precision, activation arithmetic, FP16 K/V,
Automatic-resolved Four policy, vocabulary-sharded terminal ownership, RAM
ngrams, no MTP/prefix cache, context 4096, 512/256 workload and 28 physical host cores match.
Complete preflight remains pending; G60 is the last complete gate. The [full
captured-decode trace](../../../../parity-results/qwen-flash-next/reducer100-fulltrace-r1/result.json)
completed before the test-only extension, preserving all 256 generated IDs,
four retained M512 participants and clean retirement. The
[completed attribution](../../../../parity-results/qwen-flash-next/reducer100-fulltrace-analysis-r1/REPORT.md)
joins every graph/correlation count and all 255 measured forwards per rank;
131 measured kernel names are classified, with none unclassified. The current
instrumented envelope is 108.882–108.883 ms/forward versus 26.981 ms/token
unprofiled. Collective waits and idle intervals cannot be subtracted as savings.
Each rank executes 100 branch norms and 120 floating pointer-setup calls per
forward. L101 reduces the norm's synchronization while preserving its original
arithmetic tree; its component gain leaves model throughput effectively unchanged.
Pointer setup remains a separate candidate without measured results.

The original **nine-row prefill** passes at TP1 and TP4, measuring 470.241 and
426.560 µs. TP2 fails the original-output byte gate at alpha row0/column0:
actual word3194733219 versus original3194733206. Its native library selection
uses MT64×64×8/GLVW2 with GSU7, while TP1 and the original TP4 select
MT64×64×4/GLVW1 with GSU7. The source slices and QKV/Z outputs pass; this
comparison does not establish a model-level accuracy failure. Keep TP2 red and
exclude its timing until original floating-arithmetic identity is addressed.
Applying the existing fixed-column scope would substitute a different half
reduction and is not an identity repair. No tolerance has been relaxed, and
these nine rows must not be relabeled as the production M512 prefill workload.

The existing scaling reporter now accepts the two independent targets and
optional complete actual-TP participant rounds. Its 18 device-free tests pass,
including inclusive boundaries, independent misses, duplicate/missing ranks,
paired-round validation, max-rank-before-median and raw-evidence retention.
Both the existing Unit registration and the explicit
`V2_Integration_KernelShardScalingTargets` preflight registration pass.

The first adapter mechanism gate passed five registrations: the pure ownership
contract, CUDA and ROCm captured replays, explicit-stream asynchronous reset
policy, and the existing QFN final-logit policy. Each native backend completed
20 poisoned replay histories. Evidence: `compute-isolation92-focused-r2`.
That gate used synthetic operands and a zero partition-width override; it is
not evidence for the real mirrored head or the requested TP scaling targets.

The first real-input export (`recorded-head92-native-input-r3`) completed all
three ordinary numerical checkpoints: prefill KL 0.0137324, decode0 0.0925194,
decode1 0.0936541. It remains a **failed observation** because the partial-prefix
check applied the old cosine threshold rather than QFN's declared final-logit
KL policy. The restored output was byte-identical to the original decode0
output. A focused correction preserves exact prefix/state checks and raw cosine
diagnostics while applying the declared decode KL bound separately to every
published full-vocabulary row. Unknown `PREFILL_CHUNK_0_FINAL_NORM` and
`PREFILL_CHUNK_0_LM_HEAD` metadata also exposed a missing chunk-scope parser;
the correction preserves the original prefill layout across later decode.
Neither correction changes inference arithmetic or cache behavior.

The corrected adapter and partial-prefix policy passed 17 focused registrations
(`compute-isolation93-focused-r1`), including both backends with an explicit
positive output-partition width. The complete rebuilt Unit suite also passed
(`compute-isolation93-unit-r1`). A subsequent source review found a second
retained-head issue: a historical chunk could consult the current logits
gatherer. Its focused correction marks completed vocabulary gathers as complete
publications and forbids that temporal substitution. The fresh focused gate
passes 19/19 registrations (`compute-isolation93-focused-r2`). The canonical
artifact validator now accepts the explicitly declared partial-prefix KL
columns and rejects missing, duplicate or reordered columns; its focused
preflight registration is `V2_Integration_PrefixPartialLogitArtifactSchema`.

The new real-model export `recorded-head93-native-input-r5` is fully green,
including canonical artifact validation, original native snapshots, prefix
state and clean retirement. Complete Unit passes 771/771. Those observations
qualify the input source. The Release local-shape consumer now passes all nine
recorded-row/physical-width cases. The later concurrent cohort below also passes
the TP1-relative targets for this partitioned projection compute boundary.
The parallel KV work follows the
[consumer coverage plan](QFN_TP4_KV_COVERAGE_PLAN.md): eliminate each rank's
unused KV head while retaining the actual required subset replicas and every
original-head arithmetic, cache and snapshot contract.

The local consumer's first launch exposed a missing explicit CPU-backend
initialization before its host-memory admission query. GPU initialization was
already lazy. The standalone consumer now initializes its aggregate CPU memory
owner explicitly; the failed launch and successful rebuild remain retained.
This changes diagnostic startup, not model execution or arithmetic.

All nine corrected Release cases pass the captured `LMHeadStage` and original
saved `LM_HEAD` byte oracles, twenty poisoned restore/replays each and 31 timing
samples each. The physical projection widths are 248320, 124160 and 62080,
K=2560, M=1, original Q8_0 source and fixed serial arithmetic width 62080.
The prefill entry is the **terminal M1 head**, not bulk-prefill GEMM throughput.

| Original recorded row | Full width µs | Half width µs | Quarter width µs | Full/half | Full/quarter |
|---|---:|---:|---:|---:|---:|
| Prefill terminal | 993.921 | 507.201 | 260.640 | 1.960× | 3.813× |
| Decode 0 | 993.441 | 506.560 | 259.681 | 1.961× | 3.826× |
| Decode 1 | 992.001 | 507.361 | 260.160 | 1.955× | 3.813× |

These are **local_shape** observations on ROCm device zero, with native event
intervals that may include host submission gaps. All owned processes retired
normally and driver windows are clean. Raw samples, source/kernel identities and the existing
reporter's chart are retained under `recorded-head94-local-shapes-r1`.

The matched **actual TP cohort** now passes all nine cases on physical ROCm
devices `[0]`, `[0,1]` and `[0,1,2,3]`. Every participant retains the same
recorded source/input/arithmetic identity and passes twenty standalone plus
twenty timing-parent poisoned replay checks. Native graph events bracket the
production compute fragment after restore. The report uses the slowest rank in
each complete round, then the median of 31 rounds; host submission and restore
are excluded, while native graph scheduling gaps can remain.

| Original recorded row | TP1 µs | TP2 µs | TP4 µs | TP1/TP2 | TP1/TP4 |
|---|---:|---:|---:|---:|---:|
| Prefill terminal | 989.759 | 501.119 | 257.280 | 1.975× | 3.847× |
| Decode 0 | 990.719 | 500.800 | 256.319 | 1.978× | 3.865× |
| Decode 1 | 989.919 | 501.759 | 257.120 | 1.973× | 3.850× |

All three curves meet **1.85× TP2 / 3.6× TP4**. This measures concurrent
participant projection compute, excluding vocabulary publication/collectives;
it is not whole-head or inference throughput, nor a hardware-overlap trace.
Every full original logit column is checked through the real captured
`LMHeadStage` and saved logits. Driver/resource observation and process
retirement pass for every case. The first individually green cohort used
different prepared model-context identities across degrees; aggregation
rejected that mismatch. The complete matched cohort was rerun with identical
identity, without weakening the checker. Final raw metadata, rank samples,
reductions and chart are in `recorded-head94-concurrent-r2`.

Before runtime measurements, require device-free contract negatives and native
captured replay with poisoned/changing inputs, byte comparison to the original
production operation, stable storage and zero communication in the isolated
graph. Recorded real-model inputs and production specialization must be proven
separately from synthetic mechanism tests. Keep full Unit green as adapters are
added. Focused gates support tuning; they do not replace final complete
production preflight, model accuracy or matched uninstrumented Release timing.

## Default-policy qualification

The user explicitly requested that measured faster accurate policies become
defaults, including sharded vocabulary for TP4. Accepted L90 timings already
used explicit vocabulary sharding; the canonical automatic policy still chose
mirroring. Retain this distinction in every result. Before changing automatic
selection, compare both terminal policies on the same Four-worker Release
archive, canonical numerical panel and uninstrumented A/B/B/A workload. The
older L84 head-policy comparison found a 1.2368% decode gain, but L90's worker
comparison used sharding in both arms and is not a new head-policy comparison.

The fresh same-Four comparison is complete:

| Explicit terminal policy | Prefill tok/s | Decode tok/s | Decode sample range |
|---|---:|---:|---:|
| Mirrored | 784.992 | 36.179 | 36.163–36.217 |
| Vocabulary sharded | 785.553 | 36.629 | 36.557–36.835 |

This is **+0.071% prefill / +1.243% decode**, with all twelve generated
256-token sequences equal. Both canonical panels pass and all 26 files / 185
logit rows are byte-identical to each other and the retained controls. The
workload remains 512 prompt tokens, 256 emitted tokens, context 4096, M512,
RAM ngrams, MTP off with capacity zero, four MI50s, and 28 physical NUMA1
workers. Both arms use the same immutable Four90 archive. Evidence is
`four90-vocabulary93-abba-comparison-r2.json`; individual arms retain every
sample, actual loader/driver checks and clean retirement. The first launch
was rejected before native execution because the outer invocation omitted the
declared CPU affinity; the correction is retained separately, with fresh r2
outputs and unchanged numerical inputs/runtime.

The planner now selects sharding automatically for that qualified ordinary
QFN Q8_0 MI50/RCCL TP4 family. Explicit overrides remain first-class. The new
`V2_Integration_QFNOrdinaryTP4HeadDefaults` regression reproduces the old
mirrored choice, then passes with the new resolver; the existing terminal-head
policy suite and all 771 Unit registrations also pass. Other model, hardware,
transport, unequal-share, phase-split and retained-MTP configurations retain
their existing policies. Current-source native qualification passes in
`recorded-head93-native-auto-r6`, including full/partial prefix restore. All ten
exported FINAL_NORM/LM_HEAD records are byte-identical to the earlier mirrored
run (`vocabulary93-auto-snapshot-comparison-r1.json`). This focused qualification
and the archived matched result are not a new full production-preflight
certificate.

### Branch normalization: original reduction order with fewer barriers

Candidate101a retains 256 threads per row/branch and the original FP32 FMA
loop and 128/64/32/.../1 addition tree. On HIP, the last six edges run in wave64
registers; a final CTA barrier publishes the sum to every output lane. This
reduces nine barriers to three without changing the inverse, gain arithmetic,
input precision, launch geometry or CUDA/CPU paths. The independent frozen
original GPU kernel first passed calibration against the archived L100 core.

The [final paired native-event run](../../../../parity-results/qwen-flash-next/branch-norm101-paired-final-r1/result.json)
retains 62 samples per implementation and shape, balanced across 31 ABBA rounds.
Each captured sample contains 128 normalization calls; timings below are per
call. These are component measurements, not model throughput estimates.

| Rows / hidden width | Original median µs | Candidate101a median µs | Latency change |
|---|---:|---:|---:|
| 1 / 2560 | 6.890 | 6.698 | −2.794% |
| 512 / 2560 | 121.513 | 121.401 | −0.092% |
| 1 / 2559 | 6.813 | 6.621 | −2.817% |
| 1 / 2561 | 7.250 | 7.057 | −2.655% |

The isolated production profile records grid four CTAs, block256, 16 wavefronts,
12 allocated VGPRs, 32 allocated SGPRs, 1 KiB LDS and zero scratch. Native ISA
inspection preserves every reduction operand order, epilogue and floating-mode
descriptor; there are no register spills. The 99.853% VALU utilization counter
measures active lanes, not achieved occupancy. Profiling overhead is excluded
from the event timings. Evidence: `branch-norm101-Wavefronts-final-r1`,
`branch-norm101-VALU-final-r1` and `branch-norm101-FetchSize-final-r1`.

The independent captured regression covers odd widths and reduction boundaries,
partial and empty replays, exact input/output aliasing, inactive/guard bytes,
signed zero, subnormals, infinities and paired NaN payloads. It is explicitly
registered in production preflight as
`V2_Integration_ROCmBranchNormHistoricalTree`. Both backend primitive suites
and the new regression pass; [complete Unit](../../../../parity-results/qwen-flash-next/branch-norm101-unit-final-r2/result.json)
passes 776/776 after extending the performance-target inventory assertion.
The first failed Unit receipt is retained. Release and Integration retain the
spill guard, gfx906 and CUDA SM80/86/89/90 compilation targets.

The [canonical Release panel](../../../../parity-results/qwen-flash-next/branch-norm101-release-accuracy-r1/report.json)
passes, and [all 26 files / 185 final-logit rows](../../../../parity-results/qwen-flash-next/branch-norm101-release-accuracy-r1/logit-byte-comparison.json)
are byte-identical to L100. The [complete matched Release ABBA](../../../../parity-results/qwen-flash-next/branch-norm101-model-comparison.json)
passes all four arms and preserves all twelve natural 256-token sequences.
Candidate **814.610 prefill / 37.088 decode tok/s** versus actual L100 control
**814.645 / 37.103** gives **−0.004% / −0.041%**, with overlapping ranges in
both phases. Decode ranges are 37.047–37.157 versus 37.025–37.203. This is a
qualified component optimization with unchanged model throughput, not an
end-to-end speedup. It adds no precision mode, workspace or runtime selector.
All model/runtime/topology settings remain the A1 settings above. Driver,
resource, source/runtime identity and native retirement checks pass. Complete
production preflight remains at G60.

The separate physical64-thread candidate101b is retired. After correcting
compiler-commuted NaN operands it passed the same byte gate, but decode-shape
latency increased from 6.885 to 18.139 µs and M512 increased 3.946%. Its result
is retained in `branch-norm101b-paired-r1`; none of that candidate remains in
production source.

### Floating projections: immutable captured bindings (L102)

The custom ROCm FP32 and native FP16/BF16 projections now receive a bounded,
typed value containing their original A/B/C addresses. Capture retains those
addresses directly. Their original FMA order, native weight conversion,
256-lane K ownership, reduction tree, row guards and launch geometry remain
unchanged. Real hipBLAS and physical-half table consumers still stage their
required tables. No model precision, prepared weights, workspace allocation,
host count readback or request-time transfer is introduced.

The independent immediate-original HIP oracle first calibrated against the
old implementation. The fresh control archive includes the same cold hipBLAS
workspace fix as the candidate, so the matched timing isolates the projection
binding change. The historical norm101 archive is not relabeled as this control.

The [paired unprofiled native-event measurement](../../../../parity-results/qwen-flash-next/pointer102-paired-release-r1/paired-summary.json)
uses twelve balanced pairs per shape. Original timing includes its real
pointer-publication kernel and consumer; candidate timing includes the new
consumer. These synthetic component pipelines do not establish model savings.

| Weight / M / N / K | Original µs | Candidate µs | Latency change |
|---|---:|---:|---:|
| FP32 / 1 / 4 / 10240 | 11.830 | 10.365 | −12.386% |
| FP32 / 512 / 4 / 10240 | 138.800 | 140.320 | +1.095% |
| BF16 / 1 / 128 / 2560 | 6.604 | 5.251 | −20.494% |
| BF16 / 512 / 128 / 2560 | 674.719 | 668.079 | −0.984% |
| BF16 / 1 / 512 / 2560 | 47.612 | 46.033 | −3.316% |
| BF16 / 512 / 512 / 2560 | 2588.320 | 2559.280 | −1.122% |

The HC prefill component loss remains visible; whole-model prefill is
effectively unchanged. The additional K2560 FP32 diagnostic shapes remain in
the raw paired summary and are not presented as actual HC model geometry.

[Isolated profiles](../../../../parity-results/qwen-flash-next/pointer102-profile-summary-r1/summary.json)
authenticate twelve shape/side traces and sixty consumer counter runs, each
selecting one exact warmed kernel occurrence. Five separate counter runs
measure the removed publication. The first BF16 counter attempt collected no
counter because its selector depended on formatted-name punctuation; the
corrected family selector passes the same exact dispatched-symbol, format,
ordinal, geometry and code-object checks. Both receipts remain recorded.
Profiler timings are excluded from the table and the model comparison.

[Native ISA inspection](../../../../parity-results/qwen-flash-next/pointer102-profile-summary-r1/isa-REVIEW.md)
finds the same 73-instruction reduction region and FMA operand order in all
three formats, including NaN priority and floating-mode descriptors. FP32/BF16
retain allocated VGPR12/SGPR32; FP16 retains VGPR16/SGPR32. All retain 1 KiB
LDS, zero private bytes and zero spills. FP16 sibling inspection is static,
with separate captured functional coverage; the isolated QFN timing/counter
shapes use FP32 and BF16. Consumer kernarg size grows **88→264 bytes**, while
the separate original 220-byte publication launch is removed. This is an
explicit argument-layout tradeoff, not a scratch allocation.

Block size remains 256, with N×M CTAs and four wave64 wavefronts per CTA.
Counters match those launched-wave counts, show zero LDS bank conflicts, and
show essentially unchanged payload fetches at M1. HC fetches 201.75 KiB on both
sides; key fetches 650.375 KiB; query fetches 2570.375 KiB. Candidate VALU active
lanes are 99.919% for FP32 and 99.767% for BF16. These are lane-utilization
measurements, not achieved occupancy. Register/LDS limits are unchanged; the
M1 HC launch has only four CTAs, so at most four of the 60 CUs can participate.
No dynamic occupancy or bandwidth-saturation claim follows from these counters.
The removed publication is one wave with 7.031% active lanes and 0.5 KiB fetched.

The captured regression covers all three source formats, groups 1–8 and an
8/8/3 split, distinct/shared inputs, prefix counts and request holes, twenty
dirty full/partial/empty histories, guard bytes, signed zero, subnormals,
infinities and competing NaN payloads. Full-live mapped output, actual BLAS
table consumers and physical-half table consumers are retained. The new
preflight registrations are `V2_Integration_ROCmFloatingPointerImmediateOriginal`,
`V2_Integration_FixedFloatingProjectionBindings`,
`V2_Integration_ROCmHipBLASColdCapturedWorkspace` and
`V2_Integration_ROCmFloatingProjectionPreparedInputs`.

Two setup defects were exposed and fixed. First-use captured hipBLAS attempted
to free its initial vendor workspace when binding the admitted arena. Context
initialization now retires that initial allocation before publication; every
real arithmetic submission still binds its admitted workspace. The fresh-process
regression enters its first real BLAS call inside capture without an eager
warmup. Separately, an older direct-kernel test uploaded on a different stream
without joining its event to the consumer. It failed on iteration 17 of 20 after
the incidental pointer-launch delay disappeared. The fixture now uses exact-stream
`TransferEngine::prepareDeviceInput` and output preparation; its 20-repeat
regression passes. This was a test ordering defect, not observed model drift.

[Complete Unit](../../../../parity-results/qwen-flash-next/pointer102-unit-r1/result.json)
passes 776/776. The [final focused run](../../../../parity-results/qwen-flash-next/pointer102-candidate-focused-r2/result.json)
passes 16 registrations plus its driver fixture. The heavier CUDA and ROCm GDN
capture tests passed in the earlier run on the same production implementation;
that run's overall failure is retained because of the subsequently fixed test
ordering defect. Release and Integration retain mandatory spill checks, gfx906
and the configured CUDA SM80/86/89/90 targets.

The [Release accuracy panel](../../../../parity-results/qwen-flash-next/pointer102-release-accuracy-r1/report.json)
passes its required groups. All [26 files / 185 final-logit rows](../../../../parity-results/qwen-flash-next/pointer102-release-accuracy-r1/logit-byte-comparison.json)
are byte-identical to the fresh control, including retained diagnostic rows and
reset replay. The [matched model ABBA](../../../../parity-results/qwen-flash-next/pointer102-model-comparison.json)
passes all four native lifetimes with twelve identical 256-token sequences:
**815.053 prefill / 37.424 decode tok/s**, versus **814.156 / 37.134**
(**+0.110% / +0.781%**). Decode ranges do not overlap: 37.360–37.504 versus
37.014–37.210. Prefill ranges overlap and its small change is not a separate win.

Settings remain QFN UD-Q4_K_XL, four MI50s, `TensorParallel` plus
`GateUpOwnedDownColumns`, static GPU experts, Automatic-resolved Four workers,
sharded vocabulary head, RAM ngrams, FP32 activations/FP16 KV, FP16 router and
MoE alpha/beta, TwoTerm HC-up with the other recorded projections SingleQ8,
context 4096, M512, 512 prompt tokens and 256 emitted tokens, greedy seed 42,
28 physical NUMA1 cores, and MTP/prefix cache disabled. All source/runtime,
driver, resource and native-retirement checks pass. This is a focused A1
qualification; complete production preflight remains G60.

Remaining work includes broader kernel scaling and the 42.959 tok/s Castagna
decode target. The next component candidate fuses HC activation with TwoTerm
publication inside the existing prepared GEMM stage; the prior trace records 97
of each per forward/rank. Its potential 97-launch removal has no measured speed
claim yet. A separate source audit found that counted mapped-output publication
can copy inactive redirect capacity in existing floating paths; it needs its
own negative regression and symmetric row-aware TransferEngine repair. The
full-live mapped test above does not qualify that case, and it is not an
observed failure in this A1 model run.

### HC fusion control calibration and counted-row defect

The first HC103 step adds independent frozen activation, TwoTerm publication
and residual-read programs to the existing CPU/GPU test and performance
targets. Both complete chains retain the actual prepared down/up engines;
the unchanged up projection is a shared boundary, not an independently frozen
projection oracle. The separate frozen GPU bank witness is outside component
timing. No fusion production change has been applied at this checkpoint.

The original calibration against production `71267dfa1` failed 3 of 10
registrations; the original receipt is retained in
`hc103-original-calibration-r1`. CPU Default/AVX2, the target-selection unit
and both GPU all-format M3 controls passed. The two GPU chain tests found an
inactive-output overwrite at physical M1 with zero live rows on both CUDA and
ROCm. Source diagnosis confirms that the stage carries the count, but native
adapter predicates exclude M1 from counted execution. The ordinary quantizer
and projection then process the row despite the zero count. This is an
existing counted execution defect, not a demonstrated ordinary A1 model
accuracy failure. Source review additionally found that counted M>1 consumers
mask outputs while several quantizers still process inactive input and scratch.

The scalar failure was a test admission mistake: NativeVNNI's original
participant geometry supports AVX2/AVX512 and deliberately rejects Scalar.
The corrected scalar registration proves actual scalar activation/read
arithmetic and the precise NativeVNNI rejection. It does not fabricate an
AVX2 workspace or claim a scalar NativeVNNI chain. All five corrected CPU
registrations pass in `hc103-original-cpu-calibration-r2` (12.43 s), retaining
all frozen arithmetic. The initial build's namespace ambiguity and its fixed
build receipt are retained separately.

Before fusion qualification, preserve ordinary full-active decode and correct
the explicit counted operation on both GPU backends. Forward one row authority
through quantization and projection, guard inactive input/bank accesses,
retain capture-safe device counts, and test plain, fused and SwiGLU paths over
all admitted source formats. Explicit count-aware bridge APIs preserve the
existing ordinary bridge ABI; compile-time kernel specialization must keep
ordinary execution free of count reads. Focused tests must check scratch as
well as final output across full/small/empty replay histories and be registered
in production preflight. The focused implementation and results follow below.

Component timing uses ordinary full-active down projection, matching QFN's
non-MTP production context. Counted verifier down remains a separate
correctness scenario. The initial fixture's forced-verifier timing policy was
corrected before collecting timings; no performance result from that mismatched
policy is accepted.

### Counted-row repair: focused qualification

The [final focused run](../../../../parity-results/qwen-flash-next/counted104-final-focused-r2/result.json)
passes **27/27** registrations in **78.71 seconds**, with clean driver and native
retirement checks. Integration compilation includes CUDA SM80/86/89/90 and HIP
gfx906 with mandatory spill checks. This is a focused repair qualification;
the rebuilt [complete Unit suite](../../../../parity-results/qwen-flash-next/counted104-unit-r3/result.json)
also passes **776/776** in **117.19 seconds**. Release model accuracy passes
as detailed below; matched performance checks remain in progress. L102 remains
the accepted model-throughput checkpoint.

The reduced original regressions independently exposed four production faults:

- Both GPU adapters excluded physical M1 from the device-counted path, allowing
  empty retained replays to overwrite output and activation scratch.
- Several activation quantizers processed inactive rows even when their grouped
  projection consumers masked those rows.
- Ordinary and verifier `GEMMStage` calls dropped typed bias on CPU, CUDA and
  ROCm. Independent ordinary calls and a small exact CPU calculation reproduce
  the lost bias without using the defective stage to create expectations.
- The ROCm grouped bias epilogue wrote inactive output rows. A separate M3
  regression exposed this after bias forwarding, independently of M1 admission.

The repair carries `DeviceRowRange` through explicit quantizer and epilogue
bridges, copied into native arguments. Inactive accesses are rejected before
input reads or scratch/output writes. Count-free specialization retains the
ordinary ABI and arithmetic. Raw native grouped entrypoints now admit physical
M1 only with an actual device count owner. CUDA uses the existing two-row DP4A
tile with the second lane masked, retaining the original serial family and K
partitions. Its canonical generator, strict refresh parser and installed policy
agree; a compiled test checks every registry-supported codebook and both
eager/captured selectors. CUDA M1 fused side slots are now explicitly declared
in workspace requirements. Bias forwarding preserves the original typed tensor
or view and rejects incompatible metadata before execution.

The new native suite covers all 21 source formats at M1/3/9/17, plain, biased,
SwiGLU and fused projections, plus mixed Q4_0/IQ4_NL M1. Twenty dirty replay
histories check full, partial and empty outputs, quantized values, scales, sums,
empty partial banks and guard capacity. Direct producer cases cover malformed
admission, CUDA sums/no-sums, ROCm variants 1–6, tail widths, aligned/unaligned
inputs and generic 64/128-element blocks. Captured inventories prove the actual
optimized producers execute and expose zero local memory. Independent HC
original-control chains now pass on both GPUs.

The retained negative receipts are `counted104-original-negative-r1/r2/r3`,
`counted104-original-cpu-negative-r1` and
`counted104-direct-and-epilogue-r1`. The intermediate
`counted104-native-candidate-focused-r2` passed 19/20: its remaining assertion
compared HIP's distinct ordinary single and single-projection fused bias APIs.
The corrected inventory compares the same operation boundary while retaining
the independent ordinary biased output witness and every byte assertion.
Earlier fused-fixture workspace/stream setup corrections are retained with
their failed receipts. None is presented as an observed ordinary A1 model error.

The first complete Unit run passed 774/776. Its two failures were maintenance
contracts: the unchanged codebook validator interpreted a new comment as a
format label, and an old source check still required rejection of counted M1.
The generator now emits the comment inside the branch, its refresh regression
runs the real validator, and the source check proves admission requires a real
device count owner. No dispatch statements or numerical expectations changed.

The [adjacent native run](../../../../parity-results/qwen-flash-next/counted104-neighbor-focused-r1/result.json)
passed every GEMM numerical check: canonical partial overwrite, captured bias,
CUDA/ROCm original output slices, grouped dense/local-TP projections, ROCm
prefill bucket invariance and both TwoTerm all-format sweeps. Its sole failing
registration was an old quantizer invalid-stream test expecting `false` instead
of the canonical pre-launch exception. That test now uses the canonical bridge
declaration and rejects null, legacy and per-thread default streams; the entire
registration passes in the final focused run. Original failed receipts remain
retained. These results do not replace complete ProductionTestPreflight.

A final documentation-only change adds the required generated-file Doxygen
header and keeps it in canonical emission/refresh. The Unit closure was rebuilt
and the full suite rerun afterward; the native policy statements are unchanged.
The [final Release build](../../../../parity-results/qwen-flash-next/counted104-release-build-r2/result.json)
passes for SM80/86/89/90 and gfx906 with spill enforcement enabled. The immutable
original component archive also [passes both backend controls](../../../../parity-results/qwen-flash-next/counted104-control-component-r1/result.json).
Its labels H2560/R320 mean down-projection **K10240** (four residual branches)
and up-projection K320; K2560 profiling describes a different projection shape.

The frozen [Release accuracy panel](../../../../parity-results/qwen-flash-next/counted104-release-accuracy-r1/report.json)
passes its unchanged required groups in **100.72 seconds**, with normal native
retirement and unchanged source/runtime metadata. All [26 final-logit files /
185 rows](../../../../parity-results/qwen-flash-next/counted104-release-accuracy-r1/logit-byte-comparison.json)
are byte-identical to accepted L102, including heldout histories and reset.
The original four-participant KV geometry and resolved four-worker policy are
also verified. This is the existing A1 diagnostic panel, not complete topology
or image certification.

The [matched four-arm model comparison](../../../../parity-results/qwen-flash-next/counted104-model-comparison.json)
passes all original lifecycle, loader, driver and resource checks. All twelve
measured 256-token output sequences match. Six samples per implementation give
control **815.180504 prefill / 37.420065 decode tok/s** and candidate
**815.121374 / 37.459859**: **−0.007254% / +0.106344%**. Both ranges overlap
(prefill 814.056–815.596 versus 813.086–815.528; decode 37.267–37.657 versus
37.331–37.632). This supports unchanged throughput for this correctness repair,
not a new performance win. The workload, arithmetic, four-worker policy and
sharded vocabulary remain the accepted A1 configuration.

The [four-process component follow-up](../../../../parity-results/qwen-flash-next/counted104-component-abba-r1/REVIEW.md)
retains all original observations. An initial approximately 3% CUDA M512 shift
reversed on the repeat; equal-weight process medians differ by **+0.102%** for
the production chain. CUDA M1/M9 differ by +0.084%/+0.215%; ROCm M1/M9/M512 by
+0.344%/−0.001%/−0.176%. These are descriptive component observations. The
within-process frozen primitive side shares the binary's quantizer and prepared
projection, so equal movement on both sides cannot rule out a production cost.
Native quantizer geometry and recorded register/LDS/local-memory resources are
unchanged.

The [completed ROCm profile matrix](../../../../parity-results/qwen-flash-next/counted104-profile-summary-r1/REVIEW.md)
contains **40 isolated runs**: M1/M512, K10240, sums/no-sums, ordinary/counted,
and five separate hardware counters. Each run validates 96 actual dispatches in
six retained 16-node graph replays and joins its counter to occurrence 81 after
warmup. Source, runtime, loader, driver and retirement checks pass. M1 uses
320 blocks of 32 threads; M512 uses 10,240 blocks of 128 threads. Allocated
VGPR/SGPR counts remain 8/48 or 16/48 at M1, and 16/32 or 20/32 at M512,
depending on sums. The counted M512 compiler demand adds two SGPRs but stays
within the same allocated granule. LDS, private scratch and register spills are
zero. Vector-lane utilization is 46.9–47.1% for the half-wave M1 program and
94.2–95.2% for M512. This measures active lanes, not achieved CU occupancy.

All eight dispatched specializations join the same bounded loaded code-object
extraction. Its [26 quantizer/epilogue specializations](../../../../parity-results/qwen-flash-next/counted104-quantizer-static-inventory-r1/report.json)
have zero native spills/scratch. The separately
[inspected fused SwiGLU producers](../../../../parity-results/qwen-flash-next/counted104-fused-producer-static-r2/report.json)
also have zero spills/scratch: 28 VGPRs, 46/48 compiler SGPRs for
ordinary/counted. That inspection is compiled-object evidence; it is not a
hardware counter measurement of those fused producers. CUDA resource inspection
and the Release SM80/86/89/90 spill guard pass; no CUDA profiler was attached.
Profile intervals and cache-sensitive traffic counters are diagnostic only;
the unprofiled component/model runs above own performance acceptance.

Two failed collector receipts remain retained. The first filter missed the
demangler's leading `void`; the second used a noncapturing regex group that the
pinned SDK's custom parser does not support. The corrected capturing group
preserves the exact family/template and occurrence-81 checks. No numerical,
graph, lifecycle or counter assertion was weakened. The first additional static
SwiGLU collection expected four specializations incorrectly; inspection confirms
two counted specializations with a runtime optional sums pointer, and the final
receipt checks that exact pair. These were diagnostic collector corrections,
not production failures. L104 is a focused correctness checkpoint with unchanged
model throughput; complete ProductionTestPreflight remains G60.

## HC activation and publication fusion: unqualified candidate

Counted104 is committed and pushed as `5cb59baeae971ab415038ac1a40a7a65099fa1b3`.
The subsequent local HC candidate combines the original /4 SiLU activation and
TwoTerm Q8 publication in the prepared up operation on CPU, CUDA and ROCm.
Its declarative TwoTerm graph has norm, down, up and read; up retains the
activated FP32 output and both original Q8 terms. Original projection and
reduction arithmetic are unchanged. Removing one activation dispatch at each
of the 97 measured HC sites is a structural opportunity, not a measured gain.

Cold preparation retains the original native allocation records for captured
operands, the original mapped-region owner where applicable, and the existing
workspace bank leases. Borrow acquisition and native free share the existing
backend lifecycle lock. A borrowed allocation cannot be freed or recycled;
there is no additional memory ledger or GPU allocation. Stage destruction
releases these leases only after the existing graph retirement boundary.
Invalid public tensor release may perform its existing coherence work before
native free rejects; this does not claim a new concurrent tensor-coherence
protocol.

The [first Integration rebuild](../../../../parity-results/qwen-flash-next/hc103-captured-ownership-build-r1/result.json)
passes in **552.63 seconds**, including SM80/86/89/90 and gfx906 spill enforcement.
The [ownership and original-control slice](../../../../parity-results/qwen-flash-next/hc103-captured-ownership-focused-r1/result.json)
passes **7/7 in 22.99 seconds**, with clean drivers and normal retirement.
CUDA/ROCm captured-storage cases compare twenty rejection histories with twenty
ordinary replays from identical finite input, checking exact output bytes,
unchanged allocation count/bytes and pointer retention. Mapped ownership and
device-free span/identity admission are covered. Original GPU control chains
and all-format calibration remain green.

The separate actual-builder calibration extension is applied but unqualified.
It compares retained activation, both Q8 banks/scales, up and final read against
the frozen original controls, including all 21 formats, M1/M512, aliases and
dirty full/partial/empty histories. Its
[first native calibration](../../../../parity-results/qwen-flash-next/hc103-calibration-focused-r1/result.json)
passes **5/11 in 38.45 seconds**, with clean drivers and normal retirement.
The fused CPU stage/typed-overload and admission cases pass on both default and
AVX2 ISAs, as does declarative composer coverage. Both CPU actual-builder cases
fail retained activation bytes at Q4_0, M1, exact alias, replay 2 with zero live
rows. All four CUDA/ROCm actual-builder registrations fail prepared-binding
admission before numerical execution. These failing receipts remain retained;
diagnosis must preserve production admission and exact arithmetic assertions.
Component performance, full Unit and model gates remain outstanding. L104
remains the accepted performance checkpoint.

The CPU failure identifies a production counted-row defect in the ordinary
grouped-verifier down projection: its CPU adapter processes physical capacity
without consuming the scoped live-row descriptor. Empty work can therefore
overwrite the retained aliased activation before the correctly guarded fused
up stage runs. A narrow live-row repair and direct regression are in progress.
The GPU fixture passed its original caller registration metadata, whose
prepared ref is absent; consuming the exact store-owned bindings preserves the
builder's production admission contract. After that fixture repair,
[both GPU all-format registrations pass](../../../../parity-results/qwen-flash-next/hc103-gpu-binding-focused-r1/result.json).
The other two registrations expose a second fixture error: a null-stream
setter already rejects before the test's later preparation assertion. The
corrected test asserts rejection at the public setter and unchanged retained
identity. The failed aggregate remains retained (**2/4**, 50.83 seconds,
clean drivers); it is not a complete green candidate slice.

The [corrected GPU builder slice](../../../../parity-results/qwen-flash-next/hc103-stream-assertion-focused-r1/result.json)
passes **2/2 in 13.45 seconds**, including exact captured activation, publication,
up and read bytes and the retained identity/admission negatives. Together with
the preceding two all-format passes, this closes the GPU fixture failures.

The first [CPU counted-row repair slice](../../../../parity-results/qwen-flash-next/hc103-cpu-counted105-focused-r1/result.json)
passes **10/12 in 63.45 seconds**, including both formerly failing actual HC
builders, all-format live-row histories, typed admission and row geometry on
default and AVX2. The two failures are in a new partial-bank fixture: N64/K32768
is below the canonical planner's 16M FLOP K-parallel threshold, so the fixture
incorrectly expects shared partials. N64/K131072 with three workers and two
members exercises the real shared-bank path; the corrected test asserts that
route before checking one-float-short rejection without publication or output
mutation. Production planning thresholds are unchanged.

A separate [negative control](../../../../parity-results/qwen-flash-next/hc103-cpu-two-term-negative-r1/result.json)
reproduces ordinary CPU TwoTerm's rejection of valid count-free expanded/sliced
rows on both ISAs. The implementation used expanded physical capacity as its
logical count. The one-line correction consumes `countCapacity()` through the
existing `activeRowsFor()` authority. The regression checks exact output and
both Q8 block banks across fixed and twenty changing counted histories. Its
failing receipt is retained; the corrected final focused and complete Unit
gates remain pending. Neither CPU fix changes the projection arithmetic.

The combined [focused slice](../../../../parity-results/qwen-flash-next/hc103-final-focused-r1/result.json)
then passes **50/50 in 204.43 seconds**, including both real shared-bank
admission cases. The first complete Unit run passes **775/776**; its sole
failure is the stream-presence source guard. The new HC binding queried
`gpuStream()` as a Boolean even though that accessor throws when a GPU stage
is unbound. This also prevents legitimate identity inspection after session
reset. Presence now uses `hasGPUStream()` and required execution bindings use
`requireGPUStream()`; unbound prepare/execute remain fatal.

The native ownership regression rotates all three reset APIs after actual
capture, verifies unchanged retained identity without a host stream, rejects
unbound prepare/execute, and rebinds the original exact stream. Both twenty-run
histories still replay the same graph and compare final activation/output
bytes. The [narrow follow-up](../../../../parity-results/qwen-flash-next/hc103-final-focused-r2/result.json)
passes **9/9 in 32.05 seconds**, including the architecture guard, CUDA/ROCm
reset/ownership and builder cases, CPU functional paths, composer wiring and
scalar boundary. The [complete Unit rerun](../../../../parity-results/qwen-flash-next/hc103-unit-r2/result.json)
passes **776/776 in 116.32 seconds**. All successful native slices have clean
driver windows and normal retirement. Release component/model qualification
remains pending; complete ProductionTestPreflight remains G60.

The first Release build was interrupted by session termination; its receipt
records SIGTERM and complete retirement without surviving processes. The
[incremental continuation](../../../../parity-results/qwen-flash-next/hc103-release-build-r2/result.json)
passes in **477.76 seconds**, including all four CUDA architectures and gfx906.
The [component results](../../../../parity-results/qwen-flash-next/hc103-component-summary-r2/report.json)
contain eighteen clean processes and 558 alternating candidate/control pairs.
Both GPU M1/M512 chains and every CPU byte check pass. Initial complete-chain
latency changes are CUDA −4.60%/−0.55% and ROCm −1.97%/−0.59% at M1/M512.
CPU AVX512 socket-wide M1 improves 5.61–5.74% across three fresh processes,
but runtime AVX2 M1 regresses 2.34–3.60% with one worker and 4.66–4.95% across
the socket. These AVX2 measurements use the AVX512 build's runtime ISA override;
they do not certify a separately compiled AVX2 image. The regression must be
resolved before accepting the CPU economics. No model gain is qualified yet.

The [isolated ROCm publisher matrix](../../../../parity-results/qwen-flash-next/hc103-publisher-profile-summary-r1/report.json)
passes ten runs: two shapes and five separate counters. Actual graph and dispatch
joins prove 37 candidate, 37 reference and one independent witness replay per
process. M1 has two 256-thread blocks; M512 has 640. Both use the same loaded
kernel: compiler demand is 17 VGPRs/36 SGPRs, allocated granules are 20/48,
and LDS, private scratch and register spills are zero. VALU lane utilization
is about 90.7%; this is not achieved CU occupancy. Cache-sensitive traffic
counters do not represent logical output-byte accounting. No CUDA profiler was
attached, and profiled times are excluded from performance acceptance.

CPU exact-chain counters initially fail with permission denied under the normal
user. Four fresh observations through the existing privileged path then pass
without changing `perf_event_paranoid=4`, using the authenticated 28 physical
cores on socket 1. All required counters report 100% running time. AVX2's fused
chain retires more instructions than its reference, while AVX512 retires fewer.
The interval includes normal OpenMP work and waiting, so these observations
alone do not identify an arithmetic loop as the cause. Source investigation and
the unprofiled repetitions own the next CPU tuning decision.

The next CPU candidate retains the original 256-channel activation grain inside
the fused workshare. K320 has two publication owners instead of one row owner,
and each owner publishes its original complete Q8 blocks. The existing explicit
preflight regression now also covers K512 on every source format and multiworker
K320 partial tiles, including aliases and changing live prefixes. Its Release
build passes in 109.98 seconds. Initial paired byte checks pass; the socket-wide
AVX2 timing difference shrinks to 1.63%, while one-worker AVX2 remains 2.67%.

These CPU timings subsequently expose a benchmark-boundary defect: the reference
calls frozen header-only activation/read oracles, which omit production public
validation and compile in the test executable; the candidate calls compiled
production primitives. Matching output bytes does not make those timings a
matched production comparison. Retain the earlier samples and counters as
diagnostics, but do not use their gains or regressions for CPU performance
acceptance. The next timing control must execute the production unfused chain;
both production chains must still match the independent frozen byte oracle
outside all timed/counter regions. GPU captured-event measurements have a
separate kernel-only boundary and are not changed by this CPU finding.

The final CPU timing control now calls the production unfused primitives; both
timed chains independently match the frozen original byte oracle outside the
measurement. Publication uses complete 256-channel tiles, and both paths call
one compiled projection workshare after their existing publication barrier.
This retains the source-format dispatch, Q8 bank ownership, reduction order and
one OpenMP team. The actual test executable's ISA confirms the shared call
target; a core-library-only disassembly would not establish that executable's
template binding.

The [final component summary](../../../../parity-results/qwen-flash-next/hc103-cpu-final-component-summary-r1/report.json)
retains all 248 alternating pairs from eight processes. On the AVX512 build,
default M1 latency is unchanged with one worker and improves 6.29% with the
physical socket. Its runtime AVX2 override improves 2.37% with the socket but
still regresses 3.80% with one worker. A separate CPU-only AVX2 Release build
passes all measured byte checks: latency changes are −0.19%/−1.91% at M1 and
−0.06%/−0.68% at M512 for one/socket workers. These are component observations,
not a universal CPU performance win or full-stage timing proof. In particular,
the mixed-build AVX2 one-worker regression remains visible. CPU publication is
a no-op in the fused GPU-publication helper; missing CPU publication is not an
explanation for the control difference.

The final [focused regression slice](../../../../parity-results/qwen-flash-next/hc103-final-focused-r3/result.json)
passes **23/23 in 93.79 seconds**, including expanded all-format CPU publication
tiles, counted-row admission and GPU capture/reset/builder cases. The final
[complete Unit suite](../../../../parity-results/qwen-flash-next/hc103-unit-r3/result.json)
passes **776/776 in 117.18 seconds**. The archived Release runtime passes its
[canonical accuracy group](../../../../parity-results/qwen-flash-next/hc103-release-accuracy-r1/report.json)
and preserves **all 185 final-logit rows across 26 files byte for byte** against
accepted L104, including the sparse-indexer boundary and reset cases. These
receipts qualify the stated focused scope; complete ProductionTestPreflight
still refers to G60. Matched model ABBA timing and paired traces follow with
the source and runtime archives frozen.

The [completed model ABBA](../../../../parity-results/qwen-flash-next/hc103-model-comparison.json)
retains all twelve measured sequences and their original source/runtime proofs.
All generated IDs match. Median control throughput is **814.723 prefill /
37.439 decode tok/s**; HC103 measures **815.461 / 37.576**, changes of
**+0.091% / +0.367%**. Both phase ranges overlap: this is a small descriptive
improvement, not a decisive model-throughput win. Select the qualified HC103
runtime for the subsequent stage trace, while retaining L104 and the original
Castagna target. No precision change contributes to the difference.

## Next: paired Castagna and Llaminar stage attribution

After the HC fusion's numerical, component and matched model decision, compare
the accepted Llaminar runtime with the pinned Castagna F62 A1 configuration.
Use the comparison skill's existing matched Release workflow and retained
512-token prompt, 256-token output, four MI50s, 28 physical workers on NUMA1,
context, cache and arithmetic declarations. Refresh unprofiled measurements
before diagnostic tracing; profiled throughput is not the acceptance result.

Authenticate prefill and decode windows from actual graph/token evidence and
retain every physical prefill batch. Fresh source/native evidence proves
Castagna's checkpointed320+188+4 split, correcting the earlier320+192 assumption.
Compare projections and reductions, HC, attention/indexing, GDN/convolution,
expert routing/GEMM, normalization, ngrams, vocabulary head, sampling/control,
state/layout and communication. Keep indivisible fusions intact. Report summed,
union and exposed intervals with overlap and no-kernel time accounted for;
do not add per-card or overlapping times into a supposed request latency.

Rank the largest paired exposed-time deficits, then inspect the corresponding
exact kernels' source, arithmetic/precision, ownership/sharding, launch geometry,
memory traffic, registers, spills and occupancy. Follow with isolated ROCm
counter/ISA measurements and a matched unprofiled model decision. The goal is
to identify which Castagna kernels are faster and why, while distinguishing
kernel architecture from the engines' different expert partitioning and host
control. Existing CUDA profiler restrictions remain in force. This is the next
investigation, not an assertion that paired stage evidence is already complete.

The [fresh Castagna F62 repeat](../../../../parity-results/qwen-flash-next/castagna-f62-repeat-r1/result.json)
passes normal retirement, driver and resource checks. Its warmup and all three
measured requests reproduce the retained 256 generated IDs with no prefix-cache
tokens. Measured medians are **465.370 tok/s prefill and 38.192 tok/s decode**;
decode samples are 39.600, 38.192 and 38.147 tok/s, below the retained 42.959
tok/s reference. No new Llaminar win may be inferred by pairing that repeat with
historical Llaminar throughput. Fresh matched arms and diagnostic perturbation
checks remain required.

The [second independent Castagna repeat](../../../../parity-results/qwen-flash-next/castagna-f62-repeat-r3/result.json)
also passes with identical original generated IDs, clean driver/resource checks
and normal retirement. Its medians are **453.233 tok/s prefill / 42.468 tok/s
decode**; decode samples are 42.468, 42.396 and 42.481. This is close to the
historical decode target. Both fresh observations remain visible; their
variation is unexplained and the lower first repeat does not replace the target.
The selected HC103 trace likewise passes its native workload and retirement
checks. Its instrumented 10.355 decode tok/s is diagnostic, never a replacement
for the unprofiled 37.576 tok/s ABBA result.

Source review identifies two comparison details to authenticate in fresh traces.
Castagna rotates two active full-attention participants per layer because its
GQA sharding grain is 3072 channels; each card participates in six of twelve
full-attention layers. Its persistent HC kernel can also absorb router, shared
expert and input-projection work through runtime extension parameters that are
not encoded in the kernel name. Keep those fused boundaries whole until their
actual extension membership is proven.

Llaminar's short-context sparse indexer already skips score dot products and
ranking, but still projects and prepares indexer queries that those rows never
consume. Ranking first becomes necessary at **2052 visible tokens**: 512 complete
four-token pools and up to three tail tokens still fit before that frontier.
The 512-prompt/255-after-prefill workload stays below it. Compare the actual
indexer-query cohort in the traces before implementing an optimization. Raw-key
projection, incremental prepared pools, accepted tails and ascending selected
indices remain necessary for correct later crossings. A future predicate must
be typed and device-owned, handle mixed request/grouped suffixes, and update
diagnostic liveness rather than mutating global live-row counts or fabricating
unused query values. Existing main-attention arithmetic must remain unchanged.

The [fresh paired stage comparison](QFN_CASTAGNA_STAGE_COMPARISON.md) now records
all four physical GPU pairings, complete native trace joins, source-semantic
cohorts and explicit unresolved callers. Its strongest next precision-preserving
target is the N4/K10240 FP32 HC injection projection: median per-GPU diagnostic
kernel work is1.927ms per decode step versus Castagna's0.763ms. Source review
identifies a candidate that reduces three barriers to one while retaining the
original arithmetic tree. Attention and GDN also have larger differences, but
their precision and fused preconditioning boundaries require separate treatment.

Final Default-ISA M512 CPU component checks pass byte equality in both physical
worker configurations. The one-worker medians are60712.6/60942.9us
(control/candidate,+0.379% latency); socket medians are4764.04/4736.13us
(−0.586%). These single-process31-pair observations do not remove the separately
reported mixed-build AVX2 M1 one-worker regression.

The additional [pure AVX2 Release functional slice](../../../../parity-results/qwen-flash-next/hc103-pure-avx2-functional-r1/result.json)
passes **8/8 in 29.95 seconds**, including all-format HC controls, fused/stage
builders, typed admission, live-row geometry and shared-bank rejection. The
Release-only tree has no Integration executable target; an attempted build of
that target fails before compilation and its receipt is retained. The follow-up
executes the eight original canonical selectors through their already compiled
common fixture in the Release performance binary, with unchanged test filters,
environment and timeouts. JUnit and actual native command/test counts both prove
eight executed tests. This is additional ISA coverage, not a replacement full
Integration or ProductionTestPreflight run.
