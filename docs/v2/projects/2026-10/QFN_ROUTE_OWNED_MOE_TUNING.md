# QFN route-owned MoE tuning

Updated: 2026-10-22. The explicit Four-worker A1 Release checkpoint is now
**785.806 prefill / 36.780 decode tok/s**, versus same-source Serial
**785.821 / 34.982**: **−0.002% prefill / +5.142% decode**. All twelve measured
output sequences match and all 185 retained accuracy rows are byte-identical.
Castagna Veloce's matched decode reference remains **42.959 tok/s**. The compiled
option remains explicit; the ordinary build default is still `serial` while
economy outside the qualified QFN geometry remains unmeasured.

## Actual production path

The [accepted Release84 profile](../../../../parity-results/qwen-flash-next/sharded84-rocm-profile86/result.json)
completed with matching generated tokens, normal native retirement and a clean
driver window. It used the retained Release84 executable, not the later source
tree. The workload was TP4 `TensorParallel + GateUpOwnedDownColumns`, RAM
ngrams, 512 prompt tokens, 256 emitted tokens, context 4096, M512 prefill,
28 physical NUMA1 workers, MTP off and explicit vocabulary sharding.

The existing trace tools joined all 4,108 graph launches and 7,087,096 captured
dispatches among 7,200,364 total dispatches. All 127 measured kernel names were
classified. The dominant M1 gate/up leaf is
`moe_grouped_gate_up_decode_equivalent_prefill_kernel<5,64,false,1>`:
11,985 calls per GPU, or 47 layers times 255 decode forwards. Its workgroup is
64 threads; its grid retains ten column groups, ten route slots and two
projection roles. Live owner routes remain device-owned. The dominant Q5_1 down
leaf is `moe_grouped_down_publish_canonical_routes_prefill_kernel<7>`, with
10,965 calls per GPU, physical N640/K640 and original H2560 arithmetic policy.
Both report zero scratch in the trace.

Existing expert-tiled pair settings do not select these M1 gate/up variants.
Consequently the next contest measures actual TN64/128/256, each with one
serial-K lane, while down64 stays a separately named fixed control. Increasing
TN can reduce the number of active workgroups for a sparse owner; a larger tile
is not assumed to be faster.

Profile durations are diagnostic only. The instrumented decode envelope was
about 30.46 seconds, far above the unprofiled workload. RCCL durations include
peer waits and cannot be treated as removable latency. An additional 255
`__amd_rocclr_copyBuffer` records per GPU have unknown endpoints/direction;
their symbol does not prove a kernel-argument publication or model-state copy.
The absence of separate DMA records does not exclude compute-based copies.

## Operation qualification repair

Installed exact MoE measurements previously omitted whether the operation was
complete or split, which physical columns it published, and whether grouping
covered all routes. Both GPU selectors now require the typed complete,
full-width, token-row, all-route identity before using those measurements.
Grouping provenance belongs to one kernel instance and is published only by a
successful setup transaction. Replacement, failure, reset and raw workspace
rebinding revoke old provenance. No device counts are read back or mirrored.

This preserves generic geometry and original reduction policy. It does not
claim a QFN speed improvement: no installed exact row matches QFN's
H2560/F640/E512/top10 tuple. Device-free negative controls retained the old
selector behavior behind the new interface and failed five qualification
cases; all twelve cases pass with the repair. This is an API-level negative
control, not an execution of an untouched historical GPU binary.

Validation:

- [Complete Unit](../../../../parity-results/qwen-flash-next/moe-operation86-unit/result.json):
  **764/764**, no skips, 116.13 seconds.
- [Focused device-free checks](../../../../parity-results/qwen-flash-next/moe-operation86-green-devicefree/result.json):
  both registrations pass.
- [Affected native closure](../../../../parity-results/qwen-flash-next/moe-operation86-native/result.json):
  projection phases, compact packets, grouping, canonical fold, ROCm boundary /
  exchange / quantization and ROCm grouped verifier checks pass. Two CUDA
  verifier cases initially stopped at stale expected descriptor IDs: IQ4_XS
  now preserves source-scale identity 24 rather than IQ4_NL identity 4.
- The [focused CUDA correction](../../../../parity-results/qwen-flash-next/moe-operation86-cuda-source-identity/result.json)
  passes both cases with their numerical and captured serial-byte comparisons
  intact. It is explicitly registered in production preflight.

The Integration build covered CUDA SM80/86/89/90 and ROCm gfx906 with the spill
guard enabled. These results are focused qualification, not a refreshed full
`ProductionTestPreflight` or a new whole-model accuracy claim.

## Route-owned component contest

The existing canonical MoE trainer now has an explicit descriptor-derived M1
operation branch. It retains original source aliases and original H2560 while
preparing actual participant-owned gate/up weights and one of four N640 down
intervals. Weights are seeded diagnostic inputs, not a published real-weight
corpus. One prepared owner is reused across candidates and timing rounds.

Each candidate owns a fresh captured graph. Cold inspection authenticates its
actual function, registers, LDS, private memory and theoretical residency;
recorded function/grid/block must match the admitted candidate after the
override retires. Scope overlap, late resource queries, default streams,
foreign decoders and incompatible phases fail explicitly. Ordinary production
dispatch and arithmetic are unchanged.

The [focused native gate](../../../../parity-results/qwen-flash-next/moe-route87-kernel-focused/result.json)
passes both registrations, including all physical-format resource checks and
twenty changed-input captured replays against public serial M1 bytes. A partial
F320 tile, original source aliases and Q4_K/Q5_1 mixed projections are covered.
The first build exposed a missing test interface include; adding the explicit
`IMoEKernel` header fixed it. Production HIP compilation passed the spill guard.

The full-shape trainer additionally checks owner-selected compact records,
canonical contributions and folded outputs against an independent full-H
serial oracle. It poisons all six candidate activation/intermediate banks
before every proof replay, along with packet counts and public outputs, so
eager reference setup cannot conceal a missing captured write. Full, empty and
partial route histories change hidden inputs and route order. Down64 is a
separate prepopulated down-plus-fold control, not a peer-exchange measurement.

Timing uses the existing event timer with balanced candidate order and retains
every raw sample. Results have a distinct diagnostic operation identity and
are explicitly non-installable. Release timing, all source tuples / owner
intervals, isolated profiler evidence, actual four-GPU exchange, frozen model
accuracy and matched end-to-end timing remain required before promotion.

### Retained Release r7 screening

The full-shape trainer now passes the Q4_K/Q5_1 concentrated-owner control and
all four owner intervals for both Q4_K/Q5_1 and Q4_K/Q8_0 with the existing
power-law route fixture. Each cell proves twenty full/empty/partial/reordered
input histories against the independent public serial M1 oracle before timing.
The power-law row has **4, 4, 1, 1** live routes on the four owners. The M1
`uniform` fixture assigns all ten routes to owner0 under contiguous ordinal
ownership; it is a concentrated control, not balanced TP4 traffic.

The first run exposed incorrect untimed peer setup: exporting an all-route
inverse map as each peer's producer violates the packer's unowned-route `-1`
contract. Peer reference construction now filters the authored original route
slots for each owner before grouping/projecting/exporting; production guards
and arithmetic are unchanged. The independent serial API also requires its
nonnull output binding even when canonical addends are requested. The trainer
now uses the existing MPI entrypoint and C++ linker authority. Its real MPI
bootstrap and GNU OpenMP loader closure have explicit preflight regressions;
the [six focused registrations](../../../../parity-results/qwen-flash-next/moe-route87-integration-focused-r1/result.json)
pass with clean native retirement and driver observations.

Balanced captured batches of 32 complete owner transactions reduce host
submission overhead per operation. They retain 31 warmups, 101 samples per
candidate, all three candidate graphs, and a separate fixed down-plus-fold
control. These are component timings in microseconds per operation:

| Gate/up source | Down source | Owner | Live routes | TN64 | TN128 | TN256 | Down64 + fold |
|---|---|---:|---:|---:|---:|---:|---:|
| Q4_K | Q5_1 | 0 | 4 | 67.850 | 70.250 | 68.465 | 20.910 |
| Q4_K | Q5_1 | 1 | 4 | 68.225 | 70.115 | 68.845 | 21.035 |
| Q4_K | Q5_1 | 2 | 1 | 51.770 | 52.330 | 52.520 | 21.040 |
| Q4_K | Q5_1 | 3 | 1 | 52.035 | 52.645 | 53.065 | 20.885 |
| Q4_K | Q8_0 | 0 | 4 | 67.840 | 70.295 | 68.455 | 17.670 |
| Q4_K | Q8_0 | 1 | 4 | 68.600 | 70.565 | 69.230 | 17.860 |
| Q4_K | Q8_0 | 2 | 1 | 51.525 | 52.010 | 51.930 | 17.630 |
| Q4_K | Q8_0 | 3 | 1 | 51.375 | 52.010 | 52.505 | 17.555 |

Evidence is retained under `parity-results/qwen-flash-next/` as
`moe-route87-release-layer{0,4}-owner{0,1,2,3}-powerlaw-batch32-r1`, alongside
the corresponding one-operation runs. Every cell has four summary rows and
404 raw samples. TN64 has the lowest median in all eight batch32 cells, so
these results support retaining the current tile. They do not include real
peer transport or establish an end-to-end speed gain.

The Q5_K/Q8_0 cell exposed a coverage gap in the diagnostic capability:
runtime-mutable descriptors correctly retain both compact and CPU-migration
decoder families, whereas the first diagnostic admitted only one family.
The corrected capability retains the exact physical/source-policy envelope,
inspects every normalized launch family before capture, and publishes bounded
host construction records only after the whole producer succeeds. The original
kernel bodies, launch order and ordinary defaults remain unchanged. IQ4 family
normalization uses one shared rule rather than duplicated mask arithmetic.

The [Release rebuild](../../../../parity-results/qwen-flash-next/moe-route87-envelope-release-build-r1/result.json)
passes with the spill guard enabled. All four Q5_K/Q8_0 owner intervals now pass
the full-shape twenty-history byte/extent proofs and the one-operation timing
contest (`moe-route87-envelope-release-layer2-owner{0,1,2,3}-powerlaw-r1`).
These v2 records distinguish prepared decoder7 from the reachable7/23 pair and
authenticate both captured symbols per operation. Dormant migration-kernel
capture is covered; these synthetic cells do not claim actual CPU promotion.
The first owner medians are **127.521 / 134.080 / 128.801 µs** for TN64/128/256,
again supporting retention of TN64 rather than a tile promotion.

The same batch32 timing procedure now covers all four Q5_K/Q8_0 owner intervals:

| Owner | Live routes | TN64 | TN128 | TN256 | Down64 + fold |
|---|---:|---:|---:|---:|---:|
| 0 | 4 | 116.850 | 124.650 | 119.985 | 17.795 |
| 1 | 4 | 116.965 | 124.360 | 120.685 | 17.800 |
| 2 | 1 | 88.830 | 89.860 | 94.635 | 17.610 |
| 3 | 1 | 87.890 | 89.700 | 93.740 | 17.605 |

These v2 results are retained as
`moe-route87-envelope-release-layer2-owner{0,1,2,3}-powerlaw-batch32-r1`.
All four pass the same twenty-history correctness proofs, complete captured
decoder-set checks and 404-sample contest. TN64 has the lowest median in all
twelve batch32 source/owner cells. No larger tile is promoted.

## Final focused checkpoint

The first [complete Unit run](../../../../parity-results/qwen-flash-next/moe-route87-unit-r1/result.json)
passed 765/766 registrations. Its single failure caught dummy catalog bindings
in the new device-free parser target. That target now uses the common CMake
catalog variables, and the strengthened [focused regression](../../../../parity-results/qwen-flash-next/moe-route87-shared-catalog-focused-r1/result.json)
passes. The final rebuilt [complete Unit suite](../../../../parity-results/qwen-flash-next/moe-route87-envelope-unit-r1/result.json)
passes **766/766**, no skips, in **116.83 seconds**. The final
[focused Integration/Unit gate](../../../../parity-results/qwen-flash-next/moe-route87-envelope-focused-r1/result.json)
passes **21/21**, no skips, in **232.59 seconds**, covering all-format captured
static/runtime decoder envelopes, CUDA/ROCm projection, packet, grouping and
fold contracts, trainer bootstrap/loader closure and shared catalog bindings.
Native retirement and driver windows are clean.

The final-source [archived Release model run](../../../../parity-results/qwen-flash-next/operation87-envelope-release-accuracy-sharded/runtime-observation.json)
passes the unchanged canonical required groups in **101.21 seconds**. The
[full-vocabulary comparison](../../../../parity-results/qwen-flash-next/operation87-envelope-release-accuracy-sharded/logit-byte-comparison.json)
retains all **26 files / 185 final-logit rows byte-identically** against both L74
and accepted84 sharded, including diagnostic and reset rows. Reset/replay,
runtime/source identity and normal native retirement pass. The required
numerical groups remain:

| Context group | Phase | Mean KL (nats) | Top-1 agreement | Mean top-5 overlap |
|---|---|---:|---:|---:|
| Short | Prefill | 0.033536 | 93.75% | 87.50% |
| Short | Decode | 0.032070 | 97.92% | 92.29% |
| Held-out short | Prefill | 0.009691 | 100.00% | 95.00% |
| Held-out short | Decode | 0.003838 | 100.00% | 94.17% |

This qualifies the unchanged-default TP4 M1 vocabulary-sharded slice, with MTP
off and six decode steps per ordinary prompt. It does not promote a route-owned
diagnostic override, qualify all-position output, refresh complete production
preflight, or establish a new whole-model speed. Accepted L84 throughput remains
**786.235 / 35.056 tok/s**. The next kernel investigation targets redundant
activation-correction work inside the dominant Q4_K gate/up leaf; changing TN
alone did not improve the measured component.

## Staged activation sums: rejected component candidate

The exact Release gfx906 `<5,64,false,1>` code object confirms eight
`v_dot4_i32_i8` activation-sum instructions against packed ones alongside the
eight weight-dot instructions in each K32 iteration. Activation values use
scalar loads, but the repeated correction dots remain in the compiled loop.
The exact leaf has **23 VGPRs, 80 SGPRs, zero static LDS, zero private memory and
zero spills**. Extraction commands, selected disassembly and original code
object metadata are retained in
`parity-results/qwen-flash-next/moe-correction88-original-isa/`.

The explicit diagnostic candidate prepared exact INT32 activation sums
cooperatively once per workgroup, using **320 bytes of LDS at K2560**, while
retaining the original corrected FP32 commit, source-dependent partition boundaries
and ordered fold. Both variants used TN64/K1. The [five focused registrations](../../../../parity-results/qwen-flash-next/moe-correction88-native-focused-r1/result.json)
passed, including twenty changed-input captured replays, partial N, small K,
static/runtime tables and real CPU-prepared Q5_K promotion to physical decoder 23.
Both Integration and Release builds passed with the spill guard enabled.

The [exact Release ISA](../../../../parity-results/qwen-flash-next/moe-correction88-release-isa/observation.json)
shows no spills or private memory in any of the four staged leaves. For Q4_K,
VGPR usage falls from 23 to 21, with 80 SGPRs unchanged. The compiler eliminates the
single-wave barrier instruction, but staged sums add LDS reads. In the original
Q4_K loop, weight payload and scale/min loads are issued before the correction
dots. The staged loop defers scale/min loads until after the weight dots and
then waits again. Reduced overlap is a plausible contributor to the slowdown;
static ISA does not establish a complete latency attribution.

Matched Release batch32 results, in microseconds per complete owner component:

| Gate/up source | Owner | Live routes | Inline | Staged | Latency change |
|---|---:|---:|---:|---:|---:|
| Q4_K | 0 | 4 | 71.730 | 95.705 | +33.42% |
| Q4_K | 2 | 1 | 52.155 | 65.940 | +26.43% |
| Q5_K | 0 | 4 | 118.795 | 117.810 | −0.83% |
| Q5_K | 2 | 1 | 89.135 | 89.150 | +0.02% |

All four cells passed the original serial-byte and compact-extent checks, with
three summary rows and 303 raw samples each. Native retirement and driver windows
were clean. The [disposition record](../../../../parity-results/qwen-flash-next/moe-correction88-disposition/report.json)
retains every measured cell, source identities and the exact candidate patch.
The candidate and its comparison-only additions were removed from the source
tree. No production default or throughput claim changes, and there is no reason
to run a complete model benchmark for this losing component.

## Four-worker execution of original source partitions

Candidate89 evaluates **existing source-owned K partitions in parallel**. Four
64-column planes compute disjoint original partitions, then worker0 folds the
partials in their original ascending order. Partition count, range arithmetic,
activation quantization and source-format policy are unchanged. The initial
diagnostic domain covers corrected single-scale decoders 5/7/16/23; this is a
bounded first screen, not the intended limit of applicable format coverage.

The native checks cover exact N/K/envelope admission, partial-N barrier safety,
empty partitions, the single-partition path without an extra addition, static
and runtime tables, and actual CPU-prepared Q5_K promotion to decoder23.
Twenty changed-input captured histories preserve original bytes and live
communication extents. The [four native/Unit registrations](../../../../parity-results/qwen-flash-next/moe-partition89-native-focused-r1/result.json)
and [three parser registrations](../../../../parity-results/qwen-flash-next/moe-partition89-parser-focused-r1/result.json)
pass, with clean retirement and driver windows. Integration and Release builds
pass the spill guard. Focused regressions are explicitly registered in production
preflight; these runs do not refresh the complete preflight gate.

The paired trainer retains the original compact owner transaction, preparation,
oracle and fixed down/fold control. Each cell has 31 warmups, 101 samples per
candidate and batches of 32 operations: three summary rows and 303 raw samples.
Matched Release component medians are microseconds per operation:

| Gate/up source | Down source | Owner | Live routes | Serial partitions | Four workers | Latency reduction |
|---|---|---:|---:|---:|---:|---:|
| Q4_K | Q5_1 | 0 | 4 | 70.730 | 54.335 | 23.18% |
| Q4_K | Q5_1 | 1 | 4 | 71.270 | 53.760 | 24.57% |
| Q4_K | Q5_1 | 2 | 1 | 52.185 | 40.175 | 23.01% |
| Q4_K | Q5_1 | 3 | 1 | 52.525 | 40.545 | 22.81% |
| Q5_K | Q8_0 | 0 | 4 | 117.035 | 73.305 | 37.36% |
| Q5_K | Q8_0 | 1 | 4 | 117.040 | 73.445 | 37.25% |
| Q5_K | Q8_0 | 2 | 1 | 88.735 | 54.320 | 38.78% |
| Q5_K | Q8_0 | 3 | 1 | 87.925 | 53.770 | 38.85% |
| Q4_K | Q8_0 | 0 | 4 | 70.820 | 54.495 | 23.05% |
| Q4_K | Q8_0 | 1 | 4 | 71.070 | 53.135 | 25.24% |
| Q4_K | Q8_0 | 2 | 1 | 52.185 | 39.895 | 23.55% |
| Q4_K | Q8_0 | 3 | 1 | 51.885 | 39.970 | 22.96% |

All twelve cells pass. The [disposition record](../../../../parity-results/qwen-flash-next/moe-partition89-component-disposition/report.json)
links every original result. At K2560 the source resolver retains 16 partitions,
so the four-worker layout needs exactly 4 KiB of LDS and 256 threads. Selected
gfx906 ISA has zero spills/private memory for every implemented decoder; VGPRs
are 34/38/37/34 and SGPRs 60/60/61/62 for decoders 5/7/16/23. Native resource
receipts identify exact symbols, grids, blocks and dynamic storage. For decoder5,
the occupancy API admits seven blocks per CU versus 32 one-wave serial blocks.

The saved ISA still computes the worker id in a vector register, then uses vector
activation loads even though each complete wave shares its worker. A separate
candidate will expose that wave-uniform index to the compiler. Exact bytes,
resources and timing must be checked again; scalarization alone is not a speed
claim. Format expansion must compare against each format's actual optimized
default, including existing dual-scale subgroup and IQ codebook staging paths.

The component diagnostic scope is not model capture authority. The model screen
will retain an immutable graph-local/table capability, pass it explicitly to the
same leaf and include it in capture identity. Separate source-bound Serial and
FourWorkerPlanes Release archives must pass the unchanged 185-row accuracy panel
before matched A/B/B/A timing. No diagnostic row is installable as a complete-MoE
overlay entry. Accepted whole-model throughput remains L84 **786.235 prefill /
35.056 decode tok/s**.

### Wave-uniform worker addressing

Candidate90 makes `threadIdx.x / 64` explicitly wave-uniform before the partial-N
predicate. It does not change any thread's worker assignment: one wave64 or two
wave32s share each 64-column plane. The original partition helper, LDS ownership,
barrier and ordered reduction remain unchanged. The [focused native/Unit gate](../../../../parity-results/qwen-flash-next/moe-worker90-native-focused-r1/result.json)
passes all four registrations, including the small/empty partition and partial-N
cases and real decoder23 promotion. Release and Integration spill guards pass.

All twelve matched component cells pass their twenty-history byte/extent proofs.
Compared with the simultaneous original Serial control, Q4_K latency falls
**34.3–38.5%** and Q5_K falls **47.7–48.6%**. Representative medians are
**70.760 → 44.020 µs** for Q4_K with four live owner routes and
**117.095 → 60.300 µs** for Q5_K. The [complete disposition](../../../../parity-results/qwen-flash-next/moe-worker90-component-disposition/report.json)
retains every owner/source cell; no losing cell is omitted. This succeeds the
initial vector-index four-worker layout above, whose original results remain
separately retained.

The [selected gfx906 ISA and resources](../../../../parity-results/qwen-flash-next/moe-worker90-release-isa/observation.json)
confirm scalar activation loads in the inner loop, with no global vector
activation-load pair. All four implemented leaves retain zero spills/private
memory. Decoder5 uses 29 VGPRs/69 SGPRs, with eight resident 256-thread blocks
admitted by the native occupancy query, compared with seven before scalarization.
Decoder7 uses 38/67, decoder16 26/68 and decoder23 30/67. Dynamic storage remains
the exact 4 KiB original-partition bank at P16. Static instruction evidence does
not substitute for model timing.

The extension now compiles all eighteen physical NativeVNNI decoders and their
original source envelopes with the spill guard enabled. The same range/commit
arithmetic serves symmetric, corrected, dual-scale and IQ formats. IQ2_S uses a
shared exact layout with an 8 KiB immutable codebook prefix followed by the
original-partition bank; no writer can alias the grid. The native resource query,
launch and evidence validator share that layout.

The [expanded gate](../../../../parity-results/qwen-flash-next/moe-allformat90-native-focused-r1/result.json)
passes **7/7 registrations**, no skips, in **40.46 seconds**, with clean driver
observations and normal retirement. It covers canonical source aliases,
static/runtime envelopes, exact real automatic controls, partial N, legal source
superblocks, small/empty partitions, IQ4 mixed families and actual decoder23
promotion. Twenty changing captured histories per case preserve bytes. IQ2_S's
larger-capacity automatic branch is checked at two rows/topK9; the M1 diagnostic
contract stays unchanged. All-format and IQ2 capacity checks have explicit
production-preflight registrations.

The true automatic comparison now passes all **22 timing cases**: 21 canonical
source aliases plus the IQ2_S topK16 boundary. It captures each format's actual
unscoped default, including Q6_K/IQ2_S adjacent K_LANES4, and authenticates native
symbols, resources and geometry independently of the scoped worker candidate.
The [four focused parser registrations](../../../../parity-results/qwen-flash-next/moe-automatic90-parser-focused-r1/result.json)
also pass. All timing cells retain twenty changing captured histories, identical
bytes and compact transfer extents, 101 samples per contender at 32 complete
operations per sample, clean drivers and normal retirement.

These synthetic source fixtures share N640/K2560, E32 and ordinal TP4 owner0,
with eight live local routes. They are an explicit skewed component comparison,
not the earlier four/one-live-route model-derived cells or whole-model timing.
Every source wins against its current default at this geometry, with latency
reductions of **12.6–56.3%**. The
[complete disposition](../../../../parity-results/qwen-flash-next/moe-automatic90-component-disposition/report.json)
retains all results, raw samples and actual default/candidate identities. Median
latency is microseconds per operation:

| Source | Default µs | Four workers µs | Reduction |
|---|---:|---:|---:|
| IQ1_M | 130.885 | 76.365 | 41.65% |
| IQ1_S | 109.135 | 73.370 | 32.77% |
| IQ2_S-topk16 | 102.540 | 64.770 | 36.83% |
| IQ2_S | 103.035 | 64.925 | 36.99% |
| IQ2_XS | 140.735 | 61.530 | 56.28% |
| IQ2_XXS | 110.510 | 76.430 | 30.84% |
| IQ3_S | 161.630 | 93.870 | 41.92% |
| IQ3_XXS | 167.685 | 90.455 | 46.06% |
| IQ4_NL | 101.245 | 56.795 | 43.90% |
| IQ4_XS | 86.790 | 55.300 | 36.28% |
| Q2_K | 97.270 | 54.155 | 44.33% |
| Q3_K | 103.995 | 56.800 | 45.38% |
| Q4_0 | 70.370 | 51.605 | 26.67% |
| Q4_1 | 68.730 | 53.615 | 21.99% |
| Q4_K | 68.770 | 53.630 | 22.02% |
| Q5_0 | 116.935 | 68.760 | 41.20% |
| Q5_1 | 117.195 | 69.975 | 40.29% |
| Q5_K | 116.765 | 68.965 | 40.94% |
| Q6_K | 104.540 | 69.135 | 33.87% |
| Q8_0 | 79.440 | 69.380 | 12.66% |
| Q8_1 | 79.490 | 69.430 | 12.66% |
| Q8_K | 79.410 | 69.375 | 12.64% |

The active source-format sweep covers seventeen physical decoders. The separate
genuine CPU-promoted decoder23 fixture below completes active timing coverage;
an inactive reachable leaf is not performance coverage.
The [updated gfx906 ISA](../../../../parity-results/qwen-flash-next/moe-allformat90-release-isa/observation.json)
covers all eighteen worker leaves, with 19–38 VGPRs, 63–71 SGPRs, zero register
spills and zero private memory. Native per-format occupancy and exact LDS counts
remain attached to each timing cell. Static ISA is not dynamic profiler
throughput. These component results do not establish broader geometry economy
or default promotion. The model accuracy and throughput qualification follows
below.

### Immutable model binding qualification

The model experiment now binds its prepared expert-table capability before the
first graph identity is computed. Passive descriptor inspection does no upload
or conversion; CUDA and ROCm expose equivalent inspection contracts. A declared
worker mode rejects missing or mismatched bindings before execution. The
source-local CMake setting selects `serial` or `four_worker_planes` in the owning
ROCm MoE translation unit; the default remains `serial`. It adds no serving
environment override or mutable model-global policy.

Enabling that compiled option selects every eligible ROCm native runtime-table
owner GateUp M1 stage, including singleton prefill/MTP stages and inherited
shared-FFN paths; it is not scoped to the measured QFN geometry or gfx906.
Small source-partition counts, smaller N/K, other local TP degrees and other
architectures therefore still need economy evidence before global promotion.
Cold resource admission proves launchability, not a speed advantage.

The [Serial focused gate](../../../../parity-results/qwen-flash-next/moe-model90-serial-focused-r1/result.json)
passes **18/18** registrations, and the [complete Unit suite](../../../../parity-results/qwen-flash-next/moe-model90-serial-unit-r1/result.json)
passes **769/769** in **116.68 seconds**. The
[Four-worker focused gate](../../../../parity-results/qwen-flash-next/moe-model90-four-focused-r1/result.json)
passes **20/20**, including the real two- and four-ROCm binding checks, both
backends' passive inspection tests, and surrounding projection suites. All
three gates have no skips, clean driver observations and normal retirement.
Switching the Integration mode rebuilt one C++ translation unit and relinked
the requested targets in **56.83 seconds**.

The first Integration build exposed a test translation unit that included both
CUDA and HIP native type definitions. Separate backend test translation units
fix that compiler collision while retaining the shared fixture and actual
native-node checks. The original failed build remains recorded. The same-source
Serial/Four A/B/B/A comparison below supplies the separate model throughput
evidence. These focused checks do not refresh complete production preflight.

### Active CPU-promoted decoder23

The [promoted-source comparison](../../../../parity-results/qwen-flash-next/moe-promoted23-component-disposition/report.json)
passes the Q5_K one-route smoke and all three eligible original source formats:
Q5_1, Q5_K and IQ1_S. It performs real CPU preparation, transfer completion and
borrowed-storage engine publication before capture. The timed gate/up engines
are resident decoder23; original compact GPU engines supply the independent
byte oracle. Every projection records one completed transfer, zero blocking
synchronizations and zero inference-stream waits. All four cells retain the
twenty changing histories, exact bytes/extents, 101 samples per contender at
32 operations per sample, clean drivers and normal retirement.

| Original source | Live owner routes | Default µs | Four workers µs | Reduction |
|---|---:|---:|---:|---:|
| Q5_K smoke | 1 | 69.680 | 42.880 | 38.46% |
| Q5_1 | 8 | 82.970 | 76.600 | 7.68% |
| Q5_K | 8 | 82.895 | 76.860 | 7.28% |
| IQ1_S | 8 | 83.935 | 77.215 | 8.01% |

This closes the eighteenth active physical decoder at the tested geometry.
The smaller gain with eight live promoted routes remains explicit; the
one-route smoke result does not stand in for that workload. These are
component results, with no model throughput or default-promotion claim.

### Same-source Release model accuracy

Both actual Release archives now pass the unchanged canonical sharded-M1 panel:
[Serial](../../../../parity-results/qwen-flash-next/moe-model90-serial-release-accuracy-r1/runtime-observation.json)
in **101.00 seconds**, and
[Four workers](../../../../parity-results/qwen-flash-next/moe-model90-four-release-accuracy-r1/runtime-observation.json)
in **103.33 seconds**. These are qualification elapsed times, not performance
measurements. All **26 files / 185 final-logit rows** are
[byte-identical](../../../../parity-results/qwen-flash-next/moe-model90-four-release-accuracy-r1/logit-byte-comparison.json)
to L74, accepted84 sharded and the fresh same-source Serial control, including
reset and retained diagnostic rows. All required numerical groups pass, with
clean driver windows and normal retirement. The four-device host submission
records show the bound worker mode; they are not token replay counts.

The original full-vocabulary PyTorch-reference metrics are unchanged. Means
retain equal prompt weighting within each context/phase group:

| Required group | Mean KL, nats | Top-1 agreement | Mean top-5 overlap |
|---|---:|---:|---:|
| Short prefill | 0.033536 | 93.75% | 87.50% |
| Short decode | 0.032070 | 97.92% | 92.29% |
| Held-out short prefill | 0.009691 | 100.00% | 95.00% |
| Held-out short decode | 0.003838 | 100.00% | 94.17% |

The existing QSA-threshold diagnostic remains outside the required acceptance
groups; this pass does not expand its qualification or refresh full production
preflight. The separate
[native graph observation](../../../../parity-results/qwen-flash-next/moe-model90-four-graph-inventory-r1/analysis.json)
passes on all four MI50s, with the same 26 files / 185 rows and exact reset.
Each device records 159 full decode replays and the expected Four-worker
producer templates in `main_decode/full/decode_position_snapshot`: physical
decoders 5, 7 and reachable 23, grid `10x10x2`, block `256x1x1`, 4096 dynamic LDS
bytes and zero private bytes. Register counts are 29, 38 and 30 respectively.
The two captures aggregate under each device's family tags; the result therefore
proves family membership and explicitly leaves per-instance identity unavailable.
Reachable decoder23 is not labeled useful active work in this model observation.

The first throughput launch stopped before native execution because the private
benchmark package omitted its original adjacent prompt fixtures. The
[failed prelaunch attempt](../../../../parity-results/qwen-flash-next/moe-model90-abba-prelaunch-failure-r1/result.json)
is retained. A fresh package restores the authenticated original fixtures and
preserves all eleven runner modules, native arguments and qualification inputs.
The corrected package completes the uninstrumented comparison below without
changing the source, runtime archives or benchmark policies.

### Matched Release model throughput: L90

The [complete A/B/B/A comparison](../../../../parity-results/qwen-flash-next/moe-model90-release-abba-comparison-r1)
passes all four native arms and all **twelve generated 256-token sequences**
match. Each arm has one warmup and three measured requests; each role therefore
contributes six samples. Source/runtime observations, driver windows and normal
30-second retirement checks are clean. No profiler or PerfStats instrumentation
runs inside the timing comparison.

| Compiled partition execution | Median prefill tok/s | Prefill range | Median decode tok/s | Decode range |
|---|---:|---:|---:|---:|
| Serial | 785.821 | 785.549–786.225 | 34.982 | 34.850–35.083 |
| Four worker planes | 785.806 | 784.398–786.710 | 36.780 | 36.690–36.842 |

Decode improves **5.142%**, with separated measured ranges. Prefill changes
**−0.002%**, with overlapping ranges. The candidate is still **14.38% below**
Castagna's 42.959 tok/s decode reference; matching it requires another **16.80%**
throughput increase from this checkpoint.

Both archives share the exact source and compiler/dependency closure. Their sole
compiled difference is `LLAMINAR_ROCM_MOE_M1_OWNER_PARTITIONS=serial` versus
`four_worker_planes` in the owning ROCm MoE translation unit. Both retain TP4
`TensorParallel + GateUpOwnedDownColumns`, RAM ngrams, explicit vocabulary-sharded
terminal output, MTP off with zero reserved draft capacity, the original QFN
four-shard UD-Q4_K_XL weights, all six frozen arithmetic selectors, 512 copies
of token 1228, 256 natural output tokens, C4096/M512, prefix caching off, and
28 physical NUMA1 workers. The [dashboard settings](QFN_TUNING_DASHBOARD.md#measured-performance-checkpoints)
record the shared workload and previous accepted checkpoints.

This qualifies the explicit build for this measured model/configuration. It
does not promote Four workers for every eligible geometry, certify per-instance
graph identity, or refresh the complete production-preflight gate. The next
component contest varies worker count while preserving the original source
partitions, all-format decoders and ordered arithmetic.
