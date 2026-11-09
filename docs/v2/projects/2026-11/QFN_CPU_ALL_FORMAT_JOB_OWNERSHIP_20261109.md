# CPU native quantised output-job ownership

Status: automatic panel-block ownership is qualified as the AVX512 GPU-aligned
ordinary-prefill default by focused native and model gates. Full preflight
remains deferred for this iteration.
This is a detailed qualification note, not a tuning dashboard.

## Scope and arithmetic

The common output-job scheduler must benefit models using different tensor
formats. `FusedOutputJobOwnership::Contiguous` gives each actual OpenMP worker
a contiguous interval of the existing independent jobs. It preserves native
weight bytes, activation quantisation, row kernels, fixed K partitions,
reduction order, output ownership and the existing partial-storage plan.
It changes no graph, transport or admission policy.

The new control is independent of `VerifierRowsPolicy`. The three-row arithmetic
candidate has a narrower five-format nibble domain; it is not an implementation
for all codebooks. The common ownership control has no source-format whitelist.
The existing canonical registry supplies all 21 formats to its lifecycle tests
and performance panel, including mixed adjacent source formats. Q8_1 coverage
includes prepared-kernel storage and does not advertise a new GGUF input format.

Correctness covers serial-row byte equality, dirty buffers, repeated execution,
large-to-small live prefixes, N tails, explicit/existing OpenMP teams and invalid
invocations rejected before writes. The explicit persistent FFN control also
checks both requested phases through the original planner before its first
write. That validation adds diagnostic work to complete-FFN clocks; individual
phase clocks give the cleaner work-assignment comparison.

## Fixture fidelity

The earlier FFN diagnostics used `std::vector` phase roots without recording
their alignment. Their correctness results stand, but their economy is not a
production-layout certificate. An unaligned root can introduce false sharing;
the old records do not prove that any particular root was misaligned.

Both new controls use the existing production `AlignedVector` for FP32 and Q8
phase roots and independent serial-oracle storage. Live element counts and row
strides remain unchanged. Q8 rows may legitimately begin inside a cache line;
there is no padding to make their start residues zero. Layout records publish
base alignment, exact live bytes, row stride and each expert's start residue,
after timing or profiler disarm. No raw addresses or payload hashes are needed.
The explicit `aligned_phase_roots_v1` revision distinguishes new clocks.

The original FFN owner has a focused `OutputLayoutContract` in its existing
production-preflight registration. The coherent Release build and 17 selected
native/fixture registrations pass with no skips. AVX2 checks here use runtime
selection in an AVX512 build, not a separate pure-AVX2 compilation certificate.

## Matched all-format panel

The original full-FFN owner uses authored H2560/F512, 16 experts and 608 routed
rows (M38 each), with native format matching on gate/up and down. These are
legal common geometries across the registry, not a recorded joint model route.
Both controls explicitly select Pairwise, preserving arithmetic independently
of any future Auto table. The candidate changes ownership for both phases.

Each process uses one real socket's 28 disjoint physical workers, a retained
team, 32 warmup batches and 30 samples of one invocation. Four fresh lifetimes
run Current, Contiguous, Contiguous, Current. Every lifecycle checks all phase
bytes, actual selected policy, layout metadata, unchanged binary/core identity
and complete retirement of the original MPI session helper. Timing suppresses
PerfStats; fresh proof calls occur outside the clocks. No profiler runs overlap.

All 21 formats pass byte equality, route, aligned-layout and clean-retirement
checks in all four lifetimes. Gate/up and down latency improve in both pairs
for every format. Nonpersistent complete FFN also improves in both pairs.
The table reports each pair separately; improvements are latency reductions,
not model-throughput gains.

| Source format | Gate/up reduction, % | Down reduction, % | Complete FFN reduction, % |
|---|---:|---:|---:|
| Q4_0 | 14.39 / 11.77 | 14.54 / 14.30 | 19.29 / 19.48 |
| IQ4_NL | 12.45 / 12.09 | 16.90 / 12.67 | 17.20 / 17.04 |
| IQ4_XS | 9.98 / 9.44 | 8.35 / 9.68 | 19.24 / 21.11 |
| Q4_1 | 16.28 / 15.17 | 15.19 / 14.30 | 21.02 / 21.87 |
| Q4_K | 14.38 / 14.45 | 14.04 / 13.51 | 20.71 / 21.47 |
| Q5_0 | 25.83 / 26.26 | 24.60 / 21.44 | 35.32 / 35.48 |
| Q5_1 | 26.22 / 27.06 | 25.04 / 25.15 | 36.34 / 35.22 |
| Q5_K | 26.24 / 27.31 | 25.67 / 24.48 | 34.85 / 36.95 |
| Q6_K | 13.25 / 13.35 | 8.02 / 8.88 | 16.56 / 13.56 |
| Q3_K | 6.49 / 7.21 | 5.32 / 5.76 | 8.68 / 9.32 |
| Q2_K | 4.95 / 5.50 | 4.66 / 4.75 | 5.70 / 6.50 |
| IQ3_S | 27.76 / 26.85 | 23.57 / 23.60 | 38.73 / 42.02 |
| IQ3_XXS | 28.01 / 27.45 | 22.19 / 22.73 | 34.99 / 34.56 |
| IQ2_S | 5.95 / 6.49 | 6.44 / 5.73 | 8.08 / 8.92 |
| IQ2_XS | 4.25 / 4.67 | 4.73 / 3.77 | 6.47 / 6.74 |
| IQ2_XXS | 26.54 / 26.39 | 21.87 / 24.08 | 35.24 / 36.11 |
| IQ1_S | 23.06 / 26.89 | 24.59 / 20.78 | 34.11 / 34.27 |
| IQ1_M | 3.52 / 3.95 | 3.24 / 3.96 | 4.85 / 5.06 |
| Q8_0 | 26.71 / 27.98 | 24.32 / 23.86 | 35.88 / 34.61 |
| Q8_1 | 25.55 / 23.46 | 24.62 / 22.72 | 34.54 / 34.82 |
| Q8_K | 25.01 / 24.58 | 23.59 / 23.40 | 34.68 / 35.16 |

Persistent FFN improved for 19 formats in both original pairs. Q4_0 and Q4_1
had multi-sample slow bursts in the first Contiguous lifetime, producing median
regressions of 31.74% and 21.90%; both returned to normal within their sample
series. The second Contiguous lifetime improved them by 19.39% and 21.76%.
A separate four-lifetime two-format repeat improved Q4_0 persistent latency by
17.47% / 17.77% and Q4_1 by 17.72% / 19.38%. All observations are retained;
the cause of the transient bursts is not established by these timings. They
must not be discarded or described as a proven format-specific regression.

The alignment-unspecified diagnostics remain under
`/tmp/qfn-cpu-prefill-choice-and-ownership-r1131/all-format-ownership/` and are
not relabeled. Fresh coherent build, 17 focused registrations, full aligned
panel, raw six-phase samples, and bounded persistent repeat belong to
`/tmp/qfn-cpu-all-format-aligned-r1133/`.

## Isolated execution evidence

Twelve original LinuxPerfControl lifetimes pass: three repetitions of each
ownership/phase combination at Q4_K gate/up, Q5_1 down, H2560/F320, E84/R3208
(M38/39), 28 physical workers. Each profiles one warmed production call after
separate byte checks and actual route proof. Profiling is separate from timing;
the wall-clock values below do not replace unprofiled samples.

Median aggregate per-worker counters:

| Phase / ownership | Instructions | Cycles | IPC | L1 load misses | LLC load misses |
|---|---:|---:|---:|---:|---:|
| Gate/up Current | 572.72M | 292.68M | 1.96 | 32.34M | 426,227 |
| Gate/up Contiguous | 574.00M | 244.99M | 2.34 | 32.27M | 394,854 |
| Down Current | 169.89M | 183.50M | 0.93 | 8.67M | 745,819 |
| Down Contiguous | 170.57M | 108.81M | 1.57 | 3.95M | 632,534 |

One Contiguous gate/up profile is slower (347.69M cycles, IPC 1.69); it is
retained with the other two, not discarded. The down profiles consistently
reduce cache misses and cycles. These aggregate counters support better
execution/cache locality; they do not isolate every stall or prove a cause
for the occasional slow burst.

Read-only ISA review in the preceding coherent build confirms integer chunk
assignment and no dynamic/atomic scheduler. Representative Q5, compact Q3/IQ2
hot bodies have no stack-relative SIMD instructions. Source-factor/Q6 paths
retain existing stack decoder constants/scalars; no blanket zero-stack claim
is made. The ownership change does not replace any arithmetic microkernel.
Evidence: `profiles/` under r1133 and
`/tmp/qfn-cpu-prefill-choice-and-ownership-r1131/isa/NOTE.md`.

## Exact-choice qualification and remaining work

Fresh original exact-batch collection at Q4_K N320/K2560, M64, 28 workers passes
all three choices and mixed-singleton holdouts. The small training bundle keeps
Pairwise: 153.986 microseconds versus 159.267 for OrderedThreeRows and 208.547
for WideRows. The candidate's large-bundle wins do not establish a universal
row-policy winner. The selected baseline has separate native Linux perf
evidence. No generated table was installed from this diagnostic.

Evidence: `/tmp/qfn-cpu-ordered-three-exact-r1132/`. Completed compatible sweep
observations stay retained; no broad sweep was restarted.

Before any default promotion, qualify mixed row counts and policy combinations,
repeat aligned candidate/baseline profiling and measure the real hybrid model.
The last model result remains 285.983 prefill / 40.013 benchmark decode tok/s
(39.857 after-prefill). These component tests do not meet the 500/40 target.

## Mixed-row qualification and promotion hold

The explicit local-output control now admits already validated singleton,
Pairwise, WideRows and OrderedThreeRows members without changing their
arithmetic. Mixed ownership records defer row-policy identity to the original
per-descriptor observations. The fixture accepts one positive authored
`ROW_COUNTS` vector; optional expert-count/total-row scalars must agree.
Nineteen focused registrations pass without skips after the coherent build,
including all-format mixed byte lifecycle, original phase exclusions and the
new row-vector parser contract. Production defaults remain unchanged.

The all-format Auto panel uses H2560/F512, E16/R608 and this exact authored
vector in both controls:

`1,1,2,2,3,3,4,4,9,9,16,16,72,128,169,169`

All four native lifetimes pass byte, policy, layout and retirement checks.
Unlike the uniform Pairwise panel, the mixed Auto panel does not justify a
blanket contiguous policy. Persistent Q3_K regresses 17.12% / 16.73%; IQ2_S
regresses 16.17% / 16.18%; IQ2_XS regresses 18.24% / 16.29%. Q6_K, Q2_K,
IQ1_M and IQ4_XS also regress. Other families are near flat or improve; all
phase clocks remain in the comparison artifact. These are repeated economy
failures, not numerical failures. Equal job count is not equal work when
singleton/pair/wide jobs coexist. Work-distribution analysis is in progress;
no default selection is promoted from the uniform result.

The actual QFN-shaped source pair has a different result. H2560/F320 uses
Q4_K gate/up and Q5_1 down, E84/R3208 with this authored multiset:

`[1]*16 + [2]*16 + [3]*10 + [4] + [50]*5 + [72]*34 + [77,351]`

Ascending and descending orders each run four fresh lifetimes with Auto row
policies, the original 28 physical workers and the same 32/30/1 timing settings.
Both orders pass exact bytes and improve the complete FFN:

| Row order | Gate/up reduction, % | Down reduction, % | Complete reduction, % | Persistent reduction, % |
|---|---:|---:|---:|---:|
| Ascending | 19.73 / 19.86 | 16.42 / 18.68 | 29.71 / 28.03 | 29.88 / 32.54 |
| Descending | 20.30 / 20.85 | 19.24 / 18.94 | 29.61 / 28.78 | 29.92 / 30.00 |

The separate balanced QFN-shaped aligned control at M38/39 also passes:
gate/up saves 21.93% / 21.83%, down 47.40% / 48.99%, complete FFN
39.31% / 41.41%. Persistent controls vary (9.07/10.65 ms versus 6.24 ms).
This remains component evidence, not a new model-throughput result.

Evidence: `/tmp/qfn-cpu-mixed-prefill-ownership-r1134/` contains the complete
mixed matrix; the balanced QFN source pair is `qfn-shaped/` under r1133.
The fresh unchanged-default model baseline completed successfully at 283.493
prefill / 39.129 benchmark decode tok/s (38.976 after-prefill). All 768 emitted
token IDs and MTP work match the retained baseline, and both native ranks
retired. Its driver observation fails on two new AMD SMC/fan-RPM query messages
for `0000:91:00.0`; the process did not hang. This is diagnostic evidence, not a
clean performance receipt. All GPUs remain PCIe-readable. The run-owned disk
cache was removed after retirement (727,912,496 bytes), retaining result/log
artifacts under `/tmp/qfn-cpu-ownership-baseline-model-r1135/`.

The broader format regressions remain a promotion blocker even if this model
improves. Row-only cost balancing is insufficient: compact Auto cases have
only about 4–5% maximum/mean live-row imbalance, while weighting only rows would
badly imbalance decoder/tile invocations. Source analysis points to a locality
interaction: row-major Wide jobs visit more N panels under contiguous ownership
than under the cyclic assignment. The next isolated candidate changes panel
ordering without changing jobs, physical tile kernels or arithmetic. A general
selection rule may use codebook/prepared footprint, N/K geometry, actual row
policies, job size and worker count; the existing exact kernel overlays remain
separate. These are hypotheses until matched profiling and heldout shapes
establish economy.

## General scheduling-rule experiment

The requested target is one deterministic scheduling decision driven by the
actual prepared work, rather than model names or a list of lucky M values.
Candidate features are:

- Source/prepared codebook and physical row-kernel family, including source-scale
  planes. Formats sharing a logical ID need not have the same decoder cost.
- N/K geometry, panel count and exact prepared panel footprint. Aspect ratio
  alone cannot distinguish two matrices with different working-set sizes.
- Live row/tile distribution, same-panel reuse and available jobs per actual
  worker. Allocation capacity is not live work.
- Existing CPU ISA/cache metadata. Scheduling must remain total for positive
  worker counts and supported shapes outside the measured panel.

First compare Current, equal-job Contiguous and an explicit PanelContiguous
candidate in one coherent binary. The new candidate changes the two-dimensional
job enumeration only; it keeps tile grouping, arithmetic, partition reductions,
scratch demand and task count unchanged. This isolates ordering before any
additional grouping or work-weighted partition. No automatic default changes
are justified by the experiment alone.

Qualification uses the already defined all-format uniform and mixed workloads,
the QFN-shaped pair, reversed row order and additional N/K geometries as heldout
cases. Reuse existing source preparation, timing owners, isolated profiling and
byte oracles. A useful rule must retain its gain without material regressions
on the heldouts, including complete FFNs; isolated phase medians are insufficient.
The existing exact microkernel overlays remain separate. This is not a revival
of the retired generic-fit kernel workflow.

The original mixed Q3_K profiles have now completed. Gate/up Contiguous has
146.39M median cycles versus 160.40M Current; down has 90.03M versus 85.83M.
L1/LLC counters do not establish the hypothesized larger-panel-span cost, and
the isolated warmed phase measurements do not explain the complete-FFN
regression. Retain this conflicting evidence. The ordering-only experiment
is needed before assigning a causal explanation. Profiles are under
`/tmp/qfn-cpu-mixed-prefill-ownership-r1134/mixed-compact-profiles/`.

A role-isolated full-FFN test narrows the Q3_K regression. Changing only gate/up
ownership raises persistent latency 17.71% / 18.28%; changing only down raises
it 1.27% / 1.32%. Both fresh paired controls retain byte equality. IQ4_NL is
near flat in the gate-only complete-FFN control, while down-only persistent
latency rises 5.88% / 6.97%. These measurements distinguish the phases without
claiming a cache-level cause. Evidence: `mixed-one-phase-gate_up/` and
`mixed-one-phase-down/` under r1134.

## Panel-first ordering: qualified control, insufficient automatic rule

The ordering-only `PanelContiguous` control preserves original job cardinality,
4/2/1 grouping and microkernel calls. Single-tile jobs become panel-major; grouped
jobs already were. Its counter distinguishes the actual enumeration, and fresh
fixture proofs authenticate the requested mode, K and ordering. The coherent
Release build passes nineteen focused registrations, including all21 source
formats in mixed-policy byte/lifecycle tests on AVX2 and AVX512 runtime dispatch.

Three matched two-format panels run Current, Contiguous, PanelContiguous, then
the reverse order in six fresh lifetimes each. All byte, aligned-storage,
policy, closure and native-retirement checks pass. These are authored routes.
The original row vector, its reverse and heldout H2048/F768 use Auto row policies.

| Q3_K workload | Contiguous persistent latency change | PanelContiguous persistent latency change |
|---|---:|---:|
| H2560/F512, ascending mixed rows | +20.65 / +21.97% | +9.72 / +9.46% |
| Same rows reversed | +22.01 / +19.41% | +13.18 / +13.35% |
| Heldout H2048/F768 | +28.64 / +29.91% | +17.13 / +17.46% |

Panel-first improves over Contiguous but does not beat Current on this compact
format. IQ4_NL's heldout down phase also remains 30–35% slower, making its
persistent FFN 8–9% slower. Reversed IQ4_NL has retained slow bursts (including
one +54.58% paired persistent result); these are not discarded or certified
as steady performance. Full-FFN changes differ from isolated phase changes.

The next small diagnostic balances unchanged jobs by their live row visits
times K blocks. This is an explicit work proxy, not fitted cycles. The analysis
distinguishes true four-row scratch/reduce bodies from cheaper singleton/pair
register trees; equal total job or tile-call counts hide this mix. This proxy
will be measured before adding a more detailed codebook/decoder cost model.
No automatic scheduling, exact overlay or numerical policy changes are installed.
Evidence: `/tmp/qfn-cpu-panel-ownership-r1136/`.

Twelve isolated Q3_K phase profiles also pass. PanelContiguous versus Current
gate/up median aggregate cycles are 130.85M versus 123.92M (IPC 1.98 versus
2.08); down is 84.98M versus 86.96M (IPC 2.11 versus 2.07). Gate/up LLC misses
decrease, while down L1 misses decrease and LLC misses increase. These warmed
phase observations neither explain nor override the complete-FFN timing
regressions. Profiling stays separate from canonical clocks; all seven native
counters and the original raw profiles remain in `profiles/` under r1136.

## Live-row partition diagnostic

An explicit checked whole-job partition balances live row visits times K blocks.
It reuses the existing descriptor-prefix storage and preserves original kernels
and synchronization. Nineteen focused registrations pass, including all-format
mixed bytes and independent partition coverage at1–65 workers. A test-only
namespace ambiguity was fixed before native qualification; the failed compiler
attempt remains in the artifacts. The source proof now distinguishes square
FFN roles using actual descriptor count as well as K.

The first matched two-format test rejects this proxy as an automatic policy:
Q3_K persistent FFN rises from3.439/3.427ms to5.705/5.765ms; IQ4_NL rises
from2.158/2.178ms to2.932/2.900ms. All six lifetimes pass bytes and retire
cleanly. No broader performance sweep or model run is launched for this loser.

ISA review also catches an introduced code-generation hazard: extracting a shared
job lambda outlines it, placing four normal stack-page probe writes and a0x4480
frame setup on every job. This is deliberate scratch, not a16KiB clear or proof
of register spills. Fresh Current prefill controls are roughly stable, so this
does not explain all of the weighted slowdown; it remains an avoidable overhead
for small jobs. The next revision restores the body directly inside one worker
loop, proving the original static ownership mapping. A small block-cyclic panel
control will test locality without concentrating many tiny experts on one worker.
Source/default policies remain unpromoted. Evidence: r1137 `mixed-first/` and
`isa/`. Optional requested `stats.json` was not emitted by this test owner; native
proof lines and focused range-counter tests are the actual available evidence.

The closed-cohort address-span analysis explains why rows alone are insufficient
as a scheduling proxy. For Q3_K gate/up, each Current worker visits3.28–3.40MiB
of distinct prepared panels; PanelContiguous spans0.234–10.664MiB and the
live-row partition0.352–15.352MiB. Live-row worker0 visits131 of256 gate/up
panels and328 of640 down panels. These are unique prepared address spans, not
measured memory traffic. Gate/up's prepared set is30MiB and down's15MiB; their
combined footprint exceeds this socket's38.5MiB LLC although either alone fits.
This is a cache-history hypothesis, not an established causal attribution.
The source analysis for bounded cyclic panel blocks distributes the spans much
more evenly; native timing remains the decision. Detailed enumeration is in
r1137 `analysis/unique-prepared-panels.json`.

## Bounded cyclic panel blocks

r1138 restores the physical job body inside one worker invocation. ISA inspection
places its scratch-frame setup and four stack-page probes before the job loop;
five representative pair helpers retain the saved baseline mnemonic/resource
counts. This is not a new all-format spill certificate. The live-row partition
still loses 68–70% on Q3_K and about 31% on IQ4_NL after this repair, so it is
rejected rather than expanded into a cost-fitting system.

The small-block control enumerates unchanged jobs panel-first and assigns cyclic
blocks of `min(4, max(1, floor(J / T) / 4))` jobs to the actual team. It preserves
the original row grouping, codebook operations and K reduction tree. Nineteen
focused registrations pass, including all-format byte/lifecycle controls and
an independent OpenMP static-schedule oracle with partial/empty workers.

Four fresh lifetimes per panel run Current, PanelBlocks, PanelBlocks, Current.
The mixed all-format panel improves persistent complete-FFN latency in16 of 21
formats, about 3–14%; Q3_K, Q2_K, IQ2_S, IQ2_XS and IQ1_M are 0.6–2.8% slower.
That small compact-format regression is retained explicitly. Reversed row order,
heldout H2048/F768 and a 27-physical-worker control do not expose the large
regressions of earlier partitioning candidates. These are authored inputs, not
claimed model route captures.

For the QFN-shaped H2560/F320, 84-expert, 3208-row workload, nonpersistent complete
FFN falls from 9.24–9.57 ms to 6.86–6.90 ms in ascending order and 9.44–9.47 ms to
6.88 ms in descending order: a consistent 26–28% reduction. Persistent clocks
improve 25–38%, with greater baseline variability. Gate/up improves 14–16%; down
improves about 28%. These component gains do not establish model throughput.

Twenty-four isolated phase profiles remain separate from the timing panel.
QFN-shaped down median aggregate cycles decrease 155.44M→115.95M and IPC rises
1.098→1.469, with L1 misses 8.499M→4.947M. Gate/up cycles decrease 295.36M→285.46M.
Compact gate/up cycles slightly increase 129.00M→135.44M while down decreases
104.46M→78.58M. These observations support the measured down-stage gain but
do not prove a universal cache explanation. Raw outliers are retained.
Evidence: `/tmp/qfn-cpu-panel-blocks-r1138/`, including each panel's
`comparison.json`, `compact-profiles/`, `qfn-profiles/` and `isa/NOTE.md`.

The automatic candidate uses canonical prepared-codebook byte geometry, original
row/run widths and K partitions to estimate the dominant job working set. Its
block size is bounded by measured four-job grouping, four waves of available
work and the existing immutable private-L2 cache budget. This estimate is not
physical allocation accounting or an exact enumeration of every temporary.
If fewer than two jobs fit the bounds, the original schedule is selected.
Decode, MTP, other numerical/ISA/partial-storage domains and explicit three-row
policy retain their original schedules. No model-name or codebook whitelist,
new hardware probe, learned generic-fit table or enlarged scratch is involved.
Failed PanelContiguous and live-row diagnostics are being removed.

## Fresh full-model control

r1139 is a clean unchanged-default Release control after the worker-frame repair:
**282.862 prefill /38.961 benchmark decode /38.808 after-prefill decode tok/s**.
The 512-input/256-generated workload retains one warmup and three measurements,
2×MI50 continuation plus two real CPU MPI sockets, 28 workers/socket, dynamic MTP
and expert balancing, 262144 context, RAM ngrams and 16/32 GiB RAM/disk prefix tiers.
All 768 generated token IDs, MTP work and initial placement match r1126. This is
1.09% lower prefill and 2.63% lower decode than that retained control; a single
comparison does not establish causation or a sustained 40 tok/s certificate.

Native and driver owners pass with zero new driver records. All processes retire;
the run-owned disk cache is cleaned afterward using metadata only (18 files,
727,912,508 bytes). Passive resource observation reports no collection errors,
but is not a long-duration leak or complete physical-memory certificate.
Evidence: `/tmp/qfn-cpu-panel-blocks-model-baseline-r1139/passive-comparison.json`.
The next model candidate must authenticate actual Auto choices and preserve the
same output/MTP/placement and retirement controls before default promotion.

## Automatic rule and boundary qualification

r1140 installs the candidate in the coherent local binary. The common rule is:

```text
work_cap  = max(1, floor(floor(original_jobs / actual_workers) / 4))
cache_cap = max(1, floor(existing_safe_private_L2 / largest_job_estimate))
block     = min(4, work_cap, cache_cap)
```

For an eligible AVX512, GPU-aligned, ordinary-prefill local-output bundle,
block sizes 2–4 select panel-major cyclic blocks. Size 1 keeps the complete
original schedule. The four-wave bound preserves parallel work; the measured
four-job cap limits grouping. N contributes the original N64 panel count,
M contributes live row tiles and runs, and K/codebook contribute the canonical
prepared-panel footprint. There are no model names, selected M buckets or
format whitelists. The cache estimate is computed from existing immutable CPU
metadata and prepared-layout facts, without allocation or hardware probing.
Oversized jobs remain supported streaming work. No admission limit changes.

The public fused defaults select Auto; retained fixture baselines explicitly
select Current. OrderedThreeRows, Decode/MTP, other ISAs, numerical domains and
nonlocal partial routes retain their original implementations. Worker decisions
are local values derived from common immutable inputs, so unequal scheduling
decisions cannot arise from shared mutable state. Failed diagnostic modes and
their prefix storage, per-worker records and parser entries are removed.

All 21 selected CTest registrations pass without skips in 23.90 seconds,
including complete AVX2/AVX512-runtime fused owners. These cover every registry
format, mixed/tail rows, reset/repetition, worker ownership, Auto domain/cache
bounds, actual Decode/MTP and BackendNative FullK controls, plus null-output
planning capacity. The three original counter-family/epoch tests separately
pass through their actual native owner. Their full-preflight registration is
updated; this performance-only build does not register that CTest alias.
AVX2 runtime coverage remains distinct from a pure-AVX2 build certificate.

The Release build passes for CUDA 80/86/89/90 and HIP gfx906/gfx1100 with existing
spill guards. Selected CPU ISA inspection finds no renewed per-job wrapper or
stack probes. Worker scratch reservation decreases 128 bytes; four representative
helper sequences are unchanged. Signed-expanded code generation changes without
frame growth. This is targeted compiler evidence, not a blanket no-spill claim
for every CPU helper. No profiler or build overlaps the canonical timing runs.

The four-lifetime all-format Auto panel reproduces the explicit candidate:
16 formats improve persistent FFN latency by 2.5–13.6%; the five compact formats
range from 0.37–2.48% slower. All bytes, actual selected choices, original layout
and closure checks pass. The QFN-shaped ascending complete FFN falls
9.67–9.83→6.78–6.80 ms (30–31%); descending falls 9.18→6.81–6.82 ms (26%).
Persistent reductions are 31–32% and 27%, respectively. The different baseline
latencies and both orders are retained rather than pooled into a best result.

Natural high-K heldouts use original mixed rows, H as shown, F512, 16 experts,
608 routed rows and 28 physical workers. Both phases select Auto independently;
down selects block 4 in these cases. The table is complete persistent FFN
latency relative to Current, not isolated gate/up speed or model throughput.

| Source / H | Actual gate/up block | Paired FFN latency change |
|---|---:|---:|
| Q8_0 / 3072 | 3 | −13.66 / −14.18% |
| Q8_0 / 4096 | 2 | −19.30 / −19.16% |
| Q4_K / 6144 | 3 | −11.69 / −12.31% |
| Q3_K / 6144 | 2 | −1.73 / −2.22% |
| Q8_0 / 6144 | 1, original schedule | −8.68 / −7.98% |
| Q4_K / 8192 | 2 | −13.17 / −13.97% |

The Q8_0/H6144 counter reports a 499,712-byte job estimate and cache cap 1;
gate/up keeps Current while down can still benefit. The large-K compact result
also shows why a blanket compact-format exclusion would discard useful cases.
Every decision is authenticated by a fresh native counter outside timing.
Evidence: `/tmp/qfn-cpu-auto-panel-blocks-r1140/`, its `panel-plan.json`,
per-panel comparison/ownership files, `family-gate/` and `isa/NOTE.md`.
The unchanged-filter full-model candidate and actual policy proof follow below.

## Matched model and actual production choices

r1141 passes the original native, driver, resource-observation and retirement
owners. The same Release 512/256 workload measures **292.594 prefill /39.764
benchmark decode /39.609 after-prefill decode tok/s** against the fresh r1139
control: **+3.44% prefill and +2.06% observed decode**. Auto intentionally retains
Decode/MTP scheduling, so the observed decode difference is not attributed to
a new decode algorithm. The 500/40 target is still open.

All 768 generated token IDs, measured MTP work, public/runtime configuration
and initial expert placement match. Each request retains 192 accepted drafts,
64 verifier runs and 256 verified rows. The three prefill samples are
302.833, 292.052 and 283.533 tok/s; all remain in the result. Source-component
improvement is substantially larger than the model effect, so it must not be
reported as a 26–31% inference gain. There are zero driver findings; both GPU
allocation ledgers reach zero during retirement. Run-owned prefix files are
removed afterward using metadata only (18 files, 727,912,505 bytes).

r1142 separately reuses the admitted configuration for one profiled 512/16
measurement. It passes actual process/driver/resource/DSO retirement, identical
first 16 generated tokens, declared configuration differences only and the
same initial placement. Root terminal-epoch counters prove 192 prefill Auto
calls: 190 use block 4 and two use block 3, all with 28 actual workers. The
408 other-purpose calls retain Current. Follower process-lifetime evidence
separately records 576 PanelBlocks calls and 24 insufficient-work prefill calls
retaining Current; these include preparation/warmup and are not divided by the
root measurement count. Missing purpose tags remain explicitly unknown.

The extra ownership leaf enables sibling kernel diagnostics and the selected
SDK intercepts GPU queues. Its clocks therefore are observations only, not a
second throughput comparison. The closed trace again exposes CPU-result
readiness as a major remaining frontier. Follower RCCL overlaps the root wait;
it cannot be added as a separate communication cost. Further attribution must
retain those overlap and cohort boundaries before choosing the next optimization.

Evidence: r1141 `passive-comparison.json`, `passive/benchmark.json` and original
native receipts; `/tmp/qfn-post-auto-attribution-r1142/model-trace/post-auto-evidence/`
and its closed original profiler/retirement records. No full preflight,
OpenCode application or long-duration leak certificate is claimed by this slice.

## Matched host attribution and mixed-row follow-up

r1143 compares fresh Current and restored Auto Release builds with identical
host-only diagnostic filters. Each performs one warmup and three measured
512-input/16-generated requests. Both native lifetimes, driver observers,
resource observation and process retirement pass; both owned prefix caches
are removed. All 48 output IDs, MTP work, public configuration and initial
placement match. The default remains Auto in the restored production binary.

| Root measured prefill scope, 288 calls | Current, ms | Auto, ms |
|---|---:|---:|
| Native complete FFN | 1,866.721 | 1,712.120 |
| Endpoint compute submission | 2,150.187 | 1,994.837 |
| Canonical route-slot publication | 94.900 | 92.954 |
| Transported router-Q8 publication | 106.436 | 108.100 |
| Route scheduling/bundle validation | 38.313 | 38.154 |

These scopes are not all additive. The actual FFN reduction is **8.28%**;
the FFN occupies 33.84% of Current's total measured prefill time. Total
prefill latency falls 117.694 ms across three requests, while FFN time falls
154.601 ms. The observed throughput change is 278.451 to 284.521 tok/s,
**+2.18%**. This explains the dilution without projecting the authored
component fixture's 26–31% gain onto the whole model. Canonical unprofiled
512/256 throughput remains the earlier 292.594/39.764 result; these added host
clocks and short decode workload are a separate diagnostic.

Routed work is close but not identical: root live route rows rise 506,687 to
506,964 (+0.055%); active-expert observations rise 25,959 to 26,077 (+0.455%).
Only 70 of 288 full layer/row/expert marginals match exactly. Later expert
movement also differs. Therefore neither the sampled workload nor this
attribution supplies a reconstructed identical joint expert vector. Follower
whole-lifetime FFN falls 10.47%; its 492 calls include 12 setup calls outside
the 480 layer-execute cohort and must not be divided into invented requests.

The host `captured_ticket_publication_wait` is **GPU-to-CPU input readiness**:
`MoEExpertDispatchStage` waits for the mapped dispatch input publication before
local computation. It is not a nested FFN/compute-submission timer. The
separate GPU `awaitCanonicalRouteTicketKernel` waits for CPU-to-GPU results.
Confusing these opposite directions would incorrectly treat the almost-flat
host wait as proof that the FFN saving was ineffective.

The original FFN fixture now supports an explicit per-descriptor
`EligibleOrderedThreeRows` experiment. Legal gate/up members use the existing
three-row implementation; M1/M2 retain Auto and down independently retains
Auto. This changes no production dispatch. Any three-row member currently
makes gate ownership Auto choose the complete original Current schedule.
Fresh evidence checks every member and that whole-bundle ownership consequence.

Five focused registrations pass, including mixed-row forward/reverse native
byte/lifecycle cases. Eight original-owner timings use the already-paid
E84/R3208, H2560/F320, Q4_K gate/up and Q5_1 down vectors, 28 physical workers,
32 warmups and 30 samples, in Auto/candidate/candidate/Auto order. Persistent
FFN drops 6.738–6.739 to 6.474–6.476 ms in ascending order (**3.89–3.94%**),
and 6.745 to 6.511–6.514 ms reversed (**3.43–3.47%**). All byte and fresh
selection proofs pass; processes/session state retire. These component gains
are not yet model or automatic-dispatch qualification.

Evidence: `/tmp/qfn-cpu-auto-host-ab-r1143/analysis/NOTE.md`, its original
`comparison.json` and closed native records; mixed fixture/timing records in
`/tmp/qfn-cpu-mixed-three-gate-auto-down-next/`. The scoped fixture build first
exposed an incorrect test-only encoding enum spelling; it was corrected to
the existing `ExpandedInt8` before the successful build and tests.

## Three-row eligibility and contrary measurements

The explicit mixed fixture also permits gate/up PanelBlocks while leaving
down Auto. Its forward/reverse byte and selection contracts pass, but the
QFN-shaped timings regress **0.52–2.38%** against the existing three-row
Auto/Current ownership. Keep the current ownership guard. A row-reuse win
does not imply that both scheduling changes compose into another win.

The remaining eligible canonical nibble sources were measured on the same
E84/R3208, H2560/F320, 28-worker workload. Q4_0 improves persistent complete
FFN **3.98–6.26%**, IQ4_NL **8.84–9.10%**, IQ4_XS **8.96–9.98%**, and
Q4_1 **1.77–3.50%**. The Q4_K measurements above are retained rather than
reswept. Every run passes the original serial-row byte oracle, fresh member
selection proof and session retirement. IQ4_XS uses the existing canonical
nibble source-scale representation; no tensor is requantized.

Broader aspect and work holdouts rule out promoting three rows for every
projection with N no larger than K. At K512/N512, Q4_0 and Q4_K gate/up lose
roughly 22–31%; K2048/N768 also regresses. K2560/N320 with two M64 experts
loses 7.46–8.13% complete FFN, whereas eight such experts improve
4.86–5.16%. Work and aspect therefore both matter.

A K/N bound of eight is still only a candidate, not a qualified general rule.
At K2560/N640 with 28 workers, Q4_K gate/up loses 1.82–2.98%; at K4096/N512
it is approximately flat. On the original K2560/N320 shape with 16 mixed-row
experts and seven workers, Q4_1 and Q4_K gate/up lose **6.44–7.80%** and the
separate complete operation loses **2.69–3.00%**. The persistent-team samples
in that last case visibly change regime within the timed series, falling
from about 4 ms toward 2.8–3.1 ms. Their apparent gains do not qualify the
dispatch decision. The raw chronological samples remain available; no rows
or inconvenient runs are discarded. Cache and worker effects need attribution
before installing a rule across positive worker counts.

Evidence: `/tmp/qfn-cpu-mixed-three-panel-cross-next/economy/`,
`/tmp/qfn-cpu-mixed-three-eligible-formats-r1144/`,
`/tmp/qfn-cpu-three-aspect-work-r1145/` and
`/tmp/qfn-cpu-three-ratio-boundaries-r1146/`. These are authored component
holdouts, not reconstructed model traces or full-model qualification.

The existing isolated profiler subsequently measured the same Q4_K gate/up
operation at 7/28 workers and E16/E84, two fresh observations each. Three-row
execution lowers instructions about 6–7% and L1 load misses about 29–30% in
all four cases. Small E16/seven-worker cycles rise 5.05% with essentially
unchanged effective frequency; IPC falls 11.38%. Large E84 cases reduce
cycles 0.79–3.11%. Fewer loads alone do not establish faster execution.
One-call profiler durations are not canonical timing results.

A declared 128-warmup/40-sample follow-up retains all earlier observations.
E16/28 workers is approximately flat: gate/up −0.79/+1.04%, complete
persistent FFN −1.09/−0.19%. E84/seven workers improves gate/up
3.83–6.68% and complete persistent FFN 2.27–2.99%. These clocks use the
original FFN owner and pass bytes, source/binary closure and retirement.
Evidence: `/tmp/qfn-cpu-three-worker-profiles-r1148/` and
`/tmp/qfn-cpu-three-settled-holdouts-r1149/`.

**Automatic three-row promotion is held.** A cache-footprint condition could
exclude the small resident workloads, but summing descriptor bytes would
double-count aliased weights. Sorting ranges in the current planner would
repeat on every worker; an allocation-order-sensitive estimate is also not
an attractive dispatch contract. The private prototype is not installed.
Keep the proven Auto panel scheduler and the focused comparison harness while
investigating a simpler explanation or a more valuable throughput target.
