# Native Q8 prefill: architecture-specific row tiles

The focused ROCm 10.1 collection finds an existing cooperative tile that improves
the four N320 projection shapes used by the current hybrid QFN workload: HC
read-down at K10240 and GPU shared gate/up at K2560. MI50 prefers MT16 throughout;
the 7900 XTX prefers MT16 at M64 and MT32 at M448. Canonical collection, isolated
profiling and installed-Auto correctness pass. **The eight candidate rows remain
unpromoted** because the matched model observations do not clear the decode
performance concern. Production retains its preceding 576 measured rows. The model target remains 500 prefill /40 decode tok/s on
2×MI50 +2 physical CPU sockets.

## Physical scope and clocks

Release, selected ROCm 10.1 SDK, native gfx906 and gfx1100 without ISA overrides.
The existing trainer independently prepares Q8_0, Q8_1 and Q8_K sources using
the native Q8 ABI. Its separate expanded INT8 ABI keeps its previous records.
Each architecture has 12 source cells: three formats ×two K values ×two M
values. Every candidate has 128 warm graph replays and 40 event samples; each
sample contains 16 complete activation-quantization/projection operations.
The roster retains actual installed Auto, N64/MT16, N64/MT32 and N128/MT32, all
with MB1/U2 and checked tails. Profiling is separate from these clocks.

The existing winner selector pools only the three independently qualified source
aliases. Every selected tile wins all three sources; maximum alias regret is
zero. Ratios below are the selector's geometric-mean speedup over installed
Auto, not full-model throughput.

| Native target | M | N | K | Selected NT/MT | Component speedup |
|---|---:|---:|---:|---|---:|
| gfx906 | 64 | 320 | 2560 | 64/16 | 1.360× |
| gfx906 | 64 | 320 | 10240 | 64/16 | 1.366× |
| gfx906 | 448 | 320 | 2560 | 64/16 | 1.106× |
| gfx906 | 448 | 320 | 10240 | 64/16 | 1.112× |
| gfx1100 | 64 | 320 | 2560 | 64/16 | 1.101× |
| gfx1100 | 64 | 320 | 10240 | 64/16 | 1.088× |
| gfx1100 | 448 | 320 | 2560 | 64/32 | 1.103× |
| gfx1100 | 448 | 320 | 10240 | 64/32 | 1.091× |

Short original-fixture event pairs are retained as exploratory controls. Their
XTX clocks varied enough that they cannot replace this longer collection.

## Implementation and correctness

The existing exact table gains architecture identity only for its previously
empty native-Q8 domain. Historical entries remain unchanged and unobserved;
their architecture is not invented. Lookup consumes the canonical immutable
ROCm inventory, chooses actual architecture first and then the nearest measured
compatible architecture, and checks actual launch resources. Unmeasured shapes
retain the existing complete general dispatcher. Ordinal is not an architecture.

Stable entry/config declarations now live outside the generated row include.
Measured table updates rebuild the host lookup and explicit table readers
instead of every native HIP shard: the first installed rebuild took 28.48 seconds;
a subsequent application-only table control took 3.90 seconds, with no HIP shard
compilation. There is no kernel-arithmetic change.

The original retained-graph fixture keeps six graphs, dirty large/small histories,
guard rows and actual function/grid/resource checks. Explicit MT32 is the byte
oracle, so an installed MT16 Auto cannot accidentally compare against itself.
The original two row-reset aliases independently cover all three Q8 sources.

Current focused results:

- Both AMD targets compile with the existing spill guard; core/native build
  took 165.81 seconds. The collector-only rebuild took 26.25 seconds.
- Fifteen device-free CTest registrations pass, including architecture lookup,
  collection/retention/profiling contracts and actual GNU OpenMP binding.
- Five original native capture/all-format registrations pass on each card.
  Each architecture retains six source-tagged row-history byte certificates.
- The collector's new receipt path exposed a cold-start ordering error: it
  read hardware metadata before preparing the first tensor had initialized it.
  The fixture now explicitly initializes the canonical authority and uses its
  cached device name. A new original-owner metadata-startup regression passes
  on both cards and is explicitly registered in ProductionTestPreflight.

The first collection failed before timing and remains a separate red artifact.
A local Release invocation also selected no startup tests until the original
trainer owner was explicitly admitted in the build configuration; that result
is retained separately. Neither supplies measurements or a passing certificate.
Full production preflight remains deferred; the focused results above do not
stand in for it.

## Isolated profiles and publication regression

Each architecture has 24 complete winner/actual-Auto records: every Q8 source,
geometry and physical control is profiled independently. MI50 required 216 native
profiler passes and XTX 218. All recorded projection scratch sizes are zero.
MI50 MT16 uses 72 allocated VGPRs and 10,752 LDS bytes, versus MT32's 84 VGPRs and
12,800 LDS bytes. The native function API reports unrounded register counts.
Achieved occupancy is unavailable from this tool; XTX zero-valued utilization
counters are not useful throughput evidence. Canonical event clocks remain
separate from these perturbed profile durations.

Publication exposed a second tooling defect: the native-Q8 boundary read
`metric_set_version` from a typed manifest that deliberately stores the version
only in serialized root metadata and typed records. The reader and coverage
validator already authenticate both. Removing that invalid redundant access
preserves the checks. The new real-reader regression reproduces the original
AttributeError and passes with the fix; stale root/record versions, missing Auto
coverage and nonzero scratch remain rejected. It belongs to the existing
`V2_Integration_NativeQuantisedDensePrefillEvidenceIdentity` preflight owner.

The canonical emitter retained all 576 old rows and independently added four
keys per architecture. Temporary installation passed nine native registrations
on each architecture, including original source-group Auto and dirty/reset
fixtures. Saved actual graph tuples prove all twelve source/geometry keys per
card, including XTX's distinct M448 tile. Six refreshed host registrations pass.
The production include is restored to the original rows; refreshed host gates
and both original row-reset aliases pass again on each AMD architecture. The
emitted candidates and retention receipts remain in the timing corpus for
subsequent qualification.

## Matched model result and promotion decision

Original Release owner, Qwen3.8-Flash-Next UD-Q4_K_XL plus its Q8_0 MTP sidecar;
2×MI50 continuation/shared dense TP, apportioned GPU experts and two real CPU
socket ranks with tensor-sharded experts. Preserve 28 physical workers per rank,
context262144, dynamic MTP and balancing, RAM ngrams/local GPU caches,16/32GiB
RAM/disk prefix budgets,512 prompt/256 generated tokens,seed42,one warmup and
three measured requests. All four fresh lifetimes use the same passive counters;
no profiler runs overlap timing. Decode below excludes the first prefill-produced
token, using the existing after-prefill denominator.

| Fresh lifetime, in execution order | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| Original table A | 298.747 | 38.152 |
| Candidate B | 302.444 | 36.960 |
| Original table A repeat | 299.003 | 38.292 |
| Candidate B repeat | 301.479 | 37.736 |

Every candidate/control pair preserves all768 generated token IDs, MTP work and
semantic runtime configuration including the initial placement. Actual Dynamic
movement remains enabled and differs between lifetimes. CPU completed-route
counts in the first pair differ by less than0.03%; this does not authenticate an
identical expert mix or timing. Passive thread-mask/count observations agree.
All original native/driver/resource/retirement checks pass, and owned prefix
payloads are removed after retirement using metadata only (zero payload reads).

**Do not claim an end-to-end speed win or install these rows by default.** Prefill
improves only modestly; both candidate lifetime means decode more slowly. Source
shows no direct M64/M448 match during this workload's Decode/MTP: HC widens K,
shared FFN retains query rows, the verifier permits at most16 rows, and routed
expert flattening executes a separate grouped kernel family. That excludes a
simple capacity-row explanation but does not establish timing causality. Current
passive artifacts also lack per-projection exact-hit counters. Keep the measured
component candidates and this unresolved model trend explicit.

A separate fresh automatic GPU residency experiment is now complete. The
4033 experts/card in the four table comparisons were saved resolved quotas,
revalidated rather than maximized during those runs. The closed peak resident VRAM is about30GiB
per selected GPU; post-teardown free bytes do not establish live admission
headroom. The independent arm clears only both tiers' resolved quota vectors
and derived placements, retaining every context/MTP/cache charge. Fresh admission
accepts262144 in7.42s and resolves the same8066 GPU/17022 CPU-cohort logical
experts across49 main/MTP layers. Full inference then passes at297.572 prefill
and37.389 decode tok/s. Its entire resolved routed-expert plan, all768 output IDs
and actual MTP work equal the fixed-plan control. No extra GPU capacity or speed
gain is established, and no accounting reserve was removed.

This additional **old-table** observation overlaps candidate decode rates. It
weakens attribution of the earlier decode trend to the table; it does not create
a controlled proof that the table is performance-neutral. Keep all five
lifetimes, retain the conservative default, and investigate dominant CPU expert
work/maintenance cadence rather than repeat short model clocks indefinitely.

## Evidence and remaining qualification

`/tmp/qfn-rocm-native-q8-candidate/` contains the native and startup results.
`/tmp/qfn-cpu-eight-row-rocm101-root/native-q8-policy/` and
`native-q8-collector-contracts/` contain the focused host results.
`/tmp/qfn-rocm-native-q8-exact-sdk101-r2/{gfx906,gfx1100}/` contains all raw
timings, independently observed device identity, original trainer provenance,
measured winners and isolated profile requests/results. The first failed
collection remains under `/tmp/qfn-rocm-native-q8-exact-sdk101/`.

`/tmp/qfn-rocm-native-q8-installed-sdk101/` retains temporary installed-Auto
proofs and actual tuple assertions; `/tmp/qfn-native-q8-model-ab-sdk101/` retains
all four model lifetimes, semantic comparisons, passive journals and cleanup
receipts. `/tmp/qfn-current-gpu-capacity-next/{PLAN.md,ADMISSION_RESULT.md,automatic-arm/}`
retains the separate automatic sizing configuration and completed evidence. The target500/40 and full production preflight remain
open. Keep local timing/profile payloads outside the source commit.
