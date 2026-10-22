# QFN captured compute scaling

Updated: 2026-10-22. The recorded vocabulary-projection local-shape and concurrent
TP1/TP2/TP4 measurements are green. Other kernel families remain in progress.
The accepted A1 checkpoint is L97, 814.368 prefill / 36.727 decode tok/s
on four MI50s, including [profiled native-format specialization](QFN_FLOATING_REDUCTION_TUNING.md#native-source-format-specialization-candidate97).
See the [dashboard](QFN_TUNING_DASHBOARD.md) for its exact settings and matched
comparison with L96. Prefill improves 2.254%; decode is 0.229% lower with
overlapping ranges. [Recorded-input GDN decode scaling](#recorded-gdn-projection-measurements)
now measures only 1.388–1.392× at TP2 and 1.733–1.749× at TP4. Both targets
remain missed; the original output bytes pass at every decode participant.

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

The next bounded candidates preserve the original producer/reduction order:
combine existing per-projection reducer launches using distinct admitted
partial slots, and evaluate a wave-local half reduction without LDS barriers.
Neither candidate is installed or has a performance claim at this checkpoint.

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
