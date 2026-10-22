# QFN captured compute scaling

Updated: 2026-10-22. The recorded vocabulary-projection local-shape and concurrent
TP1/TP2/TP4 measurements are green. Other kernel families remain in progress.
The accepted A1 checkpoint is L97, 814.368 prefill / 36.727 decode tok/s
on four MI50s, including [profiled native-format specialization](QFN_FLOATING_REDUCTION_TUNING.md#native-source-format-specialization-candidate97).
See the [dashboard](QFN_TUNING_DASHBOARD.md) for its exact settings and matched
comparison with L96. Prefill improves 2.254%; decode is 0.229% lower with
overlapping ranges. Recorded-input GDN scaling is the next measurement slice.

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
TP1/TP2/TP4 GDN measurements remain pending; the completed head measurements
do not establish scaling for this bundle.

For each supported cohort, cover decode and production prefill shapes plus
partial buckets. Extend reusable kernel improvements across applicable physical
formats, not only the current GGUF's formats. CPU/CUDA/ROCm share the diagnostic
contracts; a backend-specific implementation or unavailable device count is
explicitly pending until exercised.

## Gates and current evidence

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
