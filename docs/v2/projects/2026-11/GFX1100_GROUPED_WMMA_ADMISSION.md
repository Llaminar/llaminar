# Grouped WMMA whole-expert admission

This investigation qualifies explicit gfx1100 candidates. Automatic dispatch
still selects the installed Dot4 implementation. A row threshold is an experiment,
not an installed performance claim.

## Ownership and arithmetic

`MoEGroupedDotAdmission` binds projection role, original output width, physical
output width, contraction width, source/physical codebook masks and a whole-expert
row floor. Gate/up requires the original and physical widths to agree. A down
slice retains its original width and the existing source-owned partition policy.
Both Dot4 and WMMA consume the same `group_counts[expert]`, after their existing
descriptor validation. They cannot select different engines for the final
partial tile. Malformed live admission metadata traps.

The experimental floors are 4, 8, 12 and 16 rows for both M16N32 and M16N64.
Global RouteOwned/ExpertTiled admission is unchanged. No host reads live counts,
and all candidate graphs retain their original pointers and policies. The
original two-family experiment sequenced ordered leaves before directory16
overwrote their shared directory. The fused candidate below replaces that
experiment with one directory and one consumer per projection/codebook.

The native operand producer now visits only live K32 stages in the current
canonical partition. This removes decoding and LDS stores for a nonexistent
second stage while preserving adjacent activation-half owners and every FP32
contribution/fold. Before adding hybrid admission, clean Release comparisons
with 31 interleaved samples of eight complete operations gave:

| Complete operation | Dot4 µs | N32 µs | Dot4 / N32 |
|---|---:|---:|---:|
| Qwen whole-expert shard2, capacity512/live448 | 904.686 | 735.090 | 1.230715 |
| Compact Q4_0, M512 | 298.487 | 283.227 | 1.053880 |
| Compact IQ2_S, M512 | 346.847 | 340.542 | 1.018515 |

These are the **pre-hybrid producer** measurements in
`grouped-live-stages-receipt.json`, not timing qualification for a rebuilt binary.

## Replay proof

All 91 focused GPU cases passed: 45 serial-byte/promoted-storage cases, 22
split/slice cases and 24 hybrid cases. Final hybrid routing replay took 70.360 s,
all-format hybrid replay 24.038 s and split replay 29.056 s. The original serial
suite took 82.997 s. Their logs are `grouped-hybrid-original-replay.log` and
`grouped-hybrid-final-feature-suite.log`. Eight added device-free cases cover row-floor boundaries,
unequal tile ownership, source aliases, promoted physical23, physical4/24,
sliced-down bindings and invalid metadata.

The expanded component owner retains eleven complete graphs: ordered Dot4,
two pure native tiles and eight hybrid policies. Device route data crosses
empty, partial, threshold and restored extents without recapture. Every policy
also executes separately captured public phases and a source-prepared down
slice; all policies share one stable prepared slice.

Uniform E8/top-k8 routing cannot prove mixed writer ownership. The all-format
fixture therefore additionally publishes expert counts
`{3,4,7,8,12,16,17,31}`. Its 98 routes keep ExpertTiled active and straddle every
floor. An explicit Dot4 graph establishes the new reference. Disabling every
ordered down node, then every native down node, must independently break that
reference. Poisoned scratch, public output and canonical routes prevent stale
storage from supplying missing work. The Qwen power-law cases carry the same
independent omitted-writer controls.

Focused registrations are `V2_Integration_ROCmGroupedWmmaReplay`,
`V2_Integration_ROCmGroupedWmmaSplitReplay`,
`V2_Integration_ROCmGroupedWmmaHybridReplay` and
`V2_Integration_ROCmGroupedWmmaHybridAllCodebooksReplay`. All are explicitly in
production preflight; this development slice runs them directly, without the
full preflight suite, as requested.

## Evidence ownership

Local evidence lives under `/tmp/llaminar-gfx1100-wmma-prefill-evidence/`.
`grouped-hybrid-producer-receipt.json` authenticates the final Release component
binary, core and relevant source inputs for isolated profiles. It makes no
timing claim. A later clean timing receipt must match those hashes before the
profile and timing evidence can be joined. Profiles restore the original AMD
power policy and are never used as canonical timing.

Both gfx906 and gfx1100 compile under the mandatory spill guard in Release and
Integration. Native runtime qualification here uses gfx1100, PCI 0000:b5:00.0;
compilation alone does not certify another GPU's runtime behavior.

## Rebuilt native profiles

The final Release Q4_0 gate/up profiles isolate one actual dispatch per counter
pass, with seven counter passes per tile; activity and wave counts are positive. Both are wave64 with two INT8
WMMA instruction sites and zero scratch instructions, private storage or spills.

| Tile | Logical / allocated VGPRs | Logical SGPRs | Static / allocated LDS bytes | Threads |
|---|---:|---:|---:|---:|
| N32 | 53 / 56 | 54 | 3392 / 3584 | 128 |
| N64 | 54 / 56 | 54 | 5568 / 5632 | 256 |

The profiler reports 128 allocated SGPRs for both entries; the logical counts
above come from the exact loaded code-object metadata. Both runs report 8448
waves and zero LDSBankConflict. N32/N64 L2 hit rates are 80.72%/75.73%, with memory
unit activity 69.33%/68.25%. These counters describe the isolated dispatches;
they are not complete-operation performance results. Receipts and extracted ISA
are in `grouped-hybrid-n32-profile/` and `grouped-hybrid-n64-profile/`. Both power
receipts prove restoration of the original AUTO state.

The active ordered side of the N32/Floor8 Qwen power-law transaction was also
profiled in isolation: IQ2_S gate/up M12N256 uses wave64, 56 logical VGPRs,
78 logical SGPRs and 480 bytes of static LDS, with zero private storage or spills.
Seven exact-dispatch passes report 9568 waves and positive GPU/memory activity.
Its ISA contains no WMMA instructions, as expected for the ordered producer.
See `grouped-hybrid-ordered-profile/summary.json`; AUTO power was restored.

## Archived external-event tournament

These older samples include the host submission boundary between the external
start event and graph launch. They remain historical evidence; the internal-event
tournament below owns current candidate comparisons.

`grouped-hybrid-receipt.json` binds its Release producer with no
compiler overlap, unchanged source/binary hashes, and exact equality to the
profile producer receipt. The run took 202.228 s. It contains five cases,
17 live-row points, eleven policies and 31 interleaved samples of eight complete
operations: 28,985 sample rows, with output bytes checked after every sample.
The retained live points are 0,1,4,8,12,16,17,31,32,48,64,68,124,128,256,384,512.

Unlike the earlier Qwen anchor, this tournament restores all 512 live rows at
its endpoint. Ratios below are Dot4 median latency divided by candidate median
latency; larger is better.

| Full-live workload | Dot4 µs | N32 µs | Pure N32 | N32/floor8 | N32/floor16 |
|---|---:|---:|---:|---:|---:|
| Qwen uniform shard2 | 924.344 | 713.692 | 1.2952 | 1.2505 | 1.2520 |
| Qwen hotset shard2 | 763.343 | 657.117 | 1.1617 | 1.1274 | 1.1265 |
| Qwen power-law shard2 | 957.540 | 918.495 | 1.0425 | 0.9114 | 0.9096 |
| Compact Q4_0 | 294.198 | 275.448 | 1.0681 | 1.0360 | 1.0365 |
| Compact IQ2_S | 338.793 | 314.858 | 1.0760 | 1.0452 | 1.0444 |

Paired-sample median bootstrap intervals (4096 resamples, fixed seed) exclude
1.0 for all five pure-N32 full-live wins. The power-law N32/floor8 interval is
0.9072–0.9138, confirming that the mixed-writer policy loses despite the pure
native win. Sparse uniform Qwen also remains unfavorable: pure N32 ratios at
live64/128/256 are 0.6846/0.7405/0.8841. Floor16 improves those to
0.9156/0.9276/0.9391 but still pays excess whole-operation cost. Empty hybrid
replay is about 0.86× the Dot4 rate. These observations reject blanket automatic
installation of either tested strategy.

`grouped-hybrid-summary.json` validates every sample/median cell and preserves
the full-live intervals and selected partial points. The next optimization must
reduce cold-work and inactive-launch costs while retaining the demonstrated
exclusive writers and exact arithmetic. Automatic stays Dot4 until that work
has its own clean complete-operation evidence.

## Internal-event timing and replay

The timing owner now wraps each retained eight-operation source graph in one
`GPUOrderedTimelineStep` with `PerStepEvents`. It verifies an identical complete
kernel inventory and exactly two added event nodes. Both numerical replay and
omitted-writer controls execute the imported nodes in that measured parent.
Each sample poisons scratch, routing, directory and output storage, launches one
parent, consumes one named internal interval, and observes bytes before another
policy can repair the result. The terminal join and host observation remain
outside the interval; a second consume must report no pending observation.
Graph parents retire before their source fragments and failed observation drains
terminate before any asynchronous host destination can retire.

`V2_Integration_ROCmGroupedWmmaTimingReplay` explicitly joins production preflight.
Its Q4_0 and Qwen power-law fixtures pass twenty empty/partial/threshold/reset
transitions, stale-event rejection and independent omitted-writer controls.
The focused registered run passed in25.15s. Both guarded HIP targets compile.

The current clean corpus contains five individually qualified cases,17 live
counts,11 policies and31 samples of eight complete operations:28,985 sample rows.
Every case authenticates the same binary/core/source hashes and no overlapping
build, profiler or inference process. The first whole-corpus attempt passed all
native byte checks but overlapped external integration jobs; its measurements
remain rejected in `grouped-internal-overlap-rejected/`.

| Full-live workload | Dot4 µs | N32 µs | Pure N32 | N32/floor8 | N32/floor16 |
|---|---:|---:|---:|---:|---:|
| Qwen uniform shard2 | 926.970 | 712.057 | 1.3018 | 1.2626 | 1.2617 |
| Qwen hotset shard2 | 752.857 | 648.761 | 1.1605 | 1.1275 | 1.1276 |
| Qwen power-law shard2 | 945.109 | 904.754 | 1.0446 | 0.9074 | 0.9135 |
| Compact Q4_0 | 295.553 | 281.808 | 1.0488 | 1.0171 | 1.0174 |
| Compact IQ2_S | 329.498 | 315.773 | 1.0435 | 1.0053 | 1.0058 |

The paired bootstrap intervals exclude parity for all five pure-N32 full-live
wins. The same timing boundary confirms that the existing hybrid is unsuitable
for installation: power-law floor8 is0.9074× and uniform live64/128/256 floor16 is
0.9234/0.9239/0.9376×. Empty uniform native/hybrid replay is0.9125/0.8326×.

`grouped-internal-receipt.json`, `grouped-internal-tournament.log` and
`grouped-internal-summary.json` own these comparisons. Exact native/ordered
profiles now match this unchanged producer. Each has seven isolated dispatch
counter passes, positive activity, the authenticated gfx1100 PCI/UUID, zero
scratch/spills and restored AUTO power. N32/N64 retain53/54 logical VGPRs,
54 SGPRs and3392/5568 static LDS bytes. Their memory activity is68.25%/68.40%,
L2 hit80.44%/75.89%, and both report8448 waves with no LDS conflicts. The ordered
IQ2_S M12N256 leaf retains56 VGPRs,78 SGPRs and480 LDS bytes; it has no WMMA
instructions. See `grouped-internal-{n32,n64,ordered}-profile/summary.json`.
Overlapped attempts remain preserved in `grouped-internal-profile-rejected/`
and are never promoted.

## Fused candidate

The fused candidate shares one 16-row directory and one projection consumer.
It reads the authoritative whole-expert count once, then selects signed Dot4 or
WMMA uniformly across the block. Both branches feed the existing source-specific
FP32 scale/correction helpers and retain canonical K partitions, ascending block
order, empty-partition positive-zero folds and the partition-one assignment.
The ordered branch maps its output row to `4*element + lane/16`; it does not
reuse the WMMA operand-row mapping. Both branches consume the same swizzled LDS
producers. Pure native candidates compile a separate specialization without
the ordered arithmetic. Native recording no longer emits obsolete tiled Dot4
leaves or their extra directory changes; RouteOwned remains a separate complete
implementation selected by the existing device route policy.

`ROCmGroupedBranchMutation` proves branch ownership in actual retained native
down nodes, including every repetition in the measured timing parent. Normal
production submission rejects branch suppression. The test temporarily changes
only that captured argument, retains both packets, and restores them after an
exact-stream drain on every exit. It checks canonical routes against an explicit
Dot4 oracle: suppressing one branch must remove a contribution for each affected
expert while leaving every surviving expert's bytes intact. Restoring the packet
must independently recover all routes after poisoning.

The fixture retains the all-codebook mixed histogram and extends physical4/24,
physical23 promotion and odd K/N tails through nonzero floors. Cold profiling
uses capacity512/live64 for both Dot4 and the adaptive kernel. Before opening a
profiler region, it copies each selected graph's immutable down admission object
and applies `strategyForSlots` to the fixture's known route count. This applies
the complete captured geometry/codebook policy, without reading device counts,
and rejects a tiled profile whose actual route strategy would return early.

The expanded 45-case serial-byte suite passed in 108.81 s, including physical4/24,
promoted physical23 and tail partitions. The timing replay additionally proves
that every actual parent admits the full input count, rejects empty tiled work,
and rejects a count beyond the captured integer ABI. It passed in 29.39 s before
the setup improvement below. Both gfx906 and gfx1100 pass the Release/Integration
spill guards; these native runtime checks execute on gfx1100.

The component fixture now synthesizes each original seeded tensor independently
on the canonical physical-core/OpenMP budget, then uses the existing production
batch loader for quantized weights. Source ordering, seeds and per-weight packing
rules are retained. Shared source ownership outlives the device worker's prepared
handles and down-column views. Floating preparation and source-before-pack down
slicing retain their existing paths. A selected Qwen profiler check takes 1.391 s
versus 18.467 s with the previous setup: 768 source tensors take 0.321 s on 56 physical
workers and batch preparation takes 0.327 s. These are fixture setup measurements.
After this ownership change, all four affected registered component suites pass
in 62.43 s total, including all codebooks, split phases, routing and timing replay.
Seven individually selected profiler-owner cases also pass. Evidence is in
`grouped-fused-original-replay.log`, `grouped-fused-batched-feature-suite.log` and
the `grouped-fused-profile-owner-*` logs.

Five fresh isolated runs now contain 17 live counts × 11 policies × 31 paired
samples of eight complete operations, with output bytes checked after each
sample. All 28,985 sample rows pass. The timing and eight separate exact-dispatch
profiles bind the same executable, loaded core and source closure through
`grouped-fused-profile-timing-join.json`. Interrupted attempts are retained and
excluded; profiler timings do not contribute to these latency results.

| Full-live case | Dot4 µs | Pure N32 µs | Pure N32 gain | Fused N32 floor8 gain |
|---|---:|---:|---:|---:|
| Qwen uniform shard2 | 915.776 | 704.933 | 1.2991× | 1.2767× |
| Qwen hotset shard2 | 752.719 | 644.517 | 1.1679× | 1.1605× |
| Qwen power-law shard2 | 948.046 | 901.715 | 1.0514× | 1.0435× |
| Compact Q4_0 | 293.518 | 278.423 | 1.0542× | 1.0386× |
| Compact IQ2_S | 334.839 | 312.178 | 1.0726× | 1.0669× |

The paired bootstrap intervals exclude parity for these full-live wins. Fusion
removes the earlier power-law floor8 regression, but its ordered branch still
uses the native tile geometry. Uniform live64/128/256 floor8 gains are only
0.7757/0.8165/0.8815×. The same live64 bucket with hotset routing gains 1.3585×.
Neither capacity nor total live count alone identifies every winning path.
Automatic therefore remains the established Dot4 implementation. The next
candidate must preserve economical ordered tiles for sparse experts and retain
WMMA reuse for larger groups; a row-floor rule alone is insufficient.

All eight profiles authenticate the native gfx1100 PCI/UUID, loaded code object,
seven finite counters with positive activity, and restored AUTO power. Pure
Q4_0 N32/N64 retain 53 VGPRs, 54 SGPRs and 3392/5568 static LDS bytes. The fused
IQ2_S N32 entry uses 57 VGPRs, 73 SGPRs and 3520 LDS bytes; its static ISA contains
two WMMA and 32 signed Dot4 sites. The ordered IQ2_S M12N256 entry uses 56 VGPRs,
78 SGPRs and 480 LDS bytes. All have zero scratch, private storage and spills;
resource counts describe allocation, not achieved occupancy or branch counts.
Cold fused/ordered profiles fetch approximately 106875/106765 KiB while issuing
32768/9568 waves. Full uniform fused/ordered fetches approximately 109064/204222
KiB. These observations motivate preserving sparse Dot4 geometry while retaining
WMMA's greater reuse on dense groups. See `grouped-fused-summary.json` and
`grouped-fused-{n32,n64,cold,hot,mixed,ordered_cold,ordered_hot,ordered_mixed}-profile/summary.json`.
