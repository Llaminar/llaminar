# QFN route-owned MoE tuning

Updated: 2026-10-22. The accepted A1 Release score remains **786.235 prefill /
35.056 decode tok/s** with explicit vocabulary sharding. Castagna Veloce's
matched decode reference remains **42.959 tok/s**. This work has not established
a new throughput winner or changed the production tile defaults.

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
