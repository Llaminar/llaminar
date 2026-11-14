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

## Capped directory experiment

The next candidate caps the directory dimension of the captured grid at 16,
32 or 64 blocks. Each block visits directory entries at a fixed `gridDim.y`
stride; the original All policy retains one block per possible tile. A cap of
one is reserved for adversarial replay. Global route admission happens before
any directory read because an inactive planner leaves that storage unspecified.
A sentinel ends the valid prefix, while a different codebook or a diagnostic
branch suppression skips only the current entry. Each iteration resets its
partition accumulators and retains the original source-specific FP32 order.
The cap is part of the recorded grid, with no host live-count observation or
new replay-time policy lookup.

An initial extracted device helper caused 232–336 bytes of private storage
despite zero register-memory spills. The existing captured replay resource
assertions rejected it. The final arithmetic body is directly inside the
directory loop; the helper is removed. Both Release and Integration compile
gfx906 and gfx1100 under the unchanged spill guard. All 144 gfx1100 native
specializations now have zero private storage and zero scratch instructions.
SGPR spill metadata records proven transfers into VGPR lanes, which are distinct
from memory spills. The N32 gate/up Q4_0 pure/adaptive entries use 66/70 VGPRs;
IQ2_S uses 72/71. Their static LDS remains 3392/3520 bytes respectively. The
additional register pressure still requires timing qualification.

The 45 serial-byte, mixed-codebook, partial-tile and promoted-storage replays
force cap one and check the actual captured grid. They pass in 98.24 s. Seven
focused Integration registrations pass in 196.55 s, including the new explicit
preflight entries `V2_Integration_ROCmGroupedWmmaDirectoryReplay` and
`V2_Integration_ROCmGroupedWmmaDirectoryTimingReplay`. A direct Release run
executes and passes all 26 selected directory cases; a prior zero-test Release
CTest invocation is explicitly rejected as evidence. Device-free cap totality
also passes. The full preflight suite is not run.

The additional profiler owners pass 14 separate one-case smoke checks. Full
uncapped Qwen N32 has an explicit 512-live-row owner; the older 448-row probe
keeps its original meaning. Compiler/source proof is retained in
`grouped-directory-full-profile-resource-review.json`, with the earlier proof
preserved. `grouped-directory-additive-smokes-receipt.json` distinguishes these
new harness checks from the original 26/7 gate evidence and binds the unchanged
core libraries.

All five canonical cases now qualify 26,350 samples: 17 live counts, ten
policies and 31 samples of eight complete operations. Each case binds the same
20-file producer and proves no observed compiler/profiler/inference overlap.
An overlapped compact-Q4 attempt is retained and excluded; its replacement
passes. `grouped-directory-summary.json` and `grouped-directory-receipt.json`
own the results below. Values are Dot4 median latency divided by candidate
median latency at full live work.

| Case | Dot4 µs | Pure N32, All | Cap16 | Cap32 | Cap64 |
|---|---:|---:|---:|---:|---:|
| Qwen uniform | 925.446 | 1.1894× | 0.9723× | 1.0753× | 1.1885× |
| Qwen hotset | 758.493 | 1.0944× | 0.8926× | 0.9999× | 1.0948× |
| Qwen power-law | 946.926 | 0.9689× | 0.7608× | 0.8817× | 0.9599× |
| Compact Q4_0 | 293.582 | 1.0081× | 0.6602× | 0.8559× | 0.8815× |
| Compact IQ2_S | 324.018 | 0.9747× | 0.6488× | 0.8204× | 0.8536× |

Cap16 helps small concentrated work: hotset live64 reaches 1.4071×. However,
uniform live64 floor8 reaches only 0.7013× with All and 0.6954× with cap64.
The capped loop also reduces the earlier pure-N32 full-work gains. These
results reject installing this family as an Automatic policy; a fixed cap and
row floor cannot retain both geometries' advantages.

Four exact diagnostic profiles now qualify all 28 separate counter invocations
for cap64 floor8 cold/hot, full uncapped N32 and the cold ordered control. They
bind the same timing producer and restore AUTO after every invocation. All
selected entries have zero private storage, VGPR memory spills and scratch
instructions. The native entries retain 23 or 34 proven SGPR-to-VGPR lane moves.

| IQ2_S gate/up profile | VGPRs | Static LDS bytes | Waves | Fetch KiB |
|---|---:|---:|---:|---:|
| Cap64 floor8, live64 | 71 | 3520 | 4096 | 107056.875 |
| Cap64 floor8, live512 | 71 | 3520 | 4096 | 109338.875 |
| Uncapped pure N32, live512 | 72 | 3520 | 32768 | 109020.625 |
| Ordered TM12/TN256, live64 | 56 | 480 | 9568 | 106767.938 |

Fewer launched waves do not remove the sparse branch's work or weight traffic.
The exact summaries are `grouped-directory-<mode>-profile/summary.json`. They
provide allocation and activity evidence, without an achieved-occupancy
measurement. These profiler observations are not canonical timing labels.
`grouped-directory-profile-timing-join.json` binds these observations to their
matching canonical cells. The failed directory-loop implementation, cap API
and cap-only registrations have been retired; the source patch, binaries,
native objects and provenance are retained in
`grouped-directory-producer-snapshot/snapshot.json`. The full-capacity N32
profiler owner remains useful independently of that experiment. The next candidate
keeps the legacy Dot4 column width and uses four sequential WMMA column chunks
inside the same block, with one device-owned whole-expert predicate controlling
both directory row steps and the arithmetic branch. This is a design to test,
not an installed or qualified policy. Automatic remains OrderedDot4.

The restored Release core is byte-identical to the qualified fused core
(`9b378b06625749bf808be5602f7e233665a59aab167e66959b5d8bbb1de333dd`),
and its nine native source/header inputs match the retained snapshot. Both
shipping HIP targets pass the restored Release/Integration spill-guard builds.
All 17 dispatch Unit cases and the two new full-capacity N32 controls (uniform
and power-law) pass. `grouped-retired-checkpoint-receipt.json` binds those checks
to the rebuilt artifacts and original native evidence.

## Wide Dot4 and four-chunk WMMA candidate

The follow-up candidate is implemented as an explicit `WideDot4Wmma` recording
choice. Automatic remains OrderedDot4 while its full qualification runs. Gate/up
retains TM12/TN256 for cold experts and down retains TM16/TN128. Hot experts use
sixteen rows and four consecutive WMMA column chunks in the same workgroup.
Each chunk retains only four FP32 outputs and the original serial partition order.
One shared-memory union serves the mutually exclusive branches. Cooperative
integer activation sums publish with operands at the first barrier; they do not
add a third barrier per cold K32 block.

`MoEGroupedRowPlan` seals role, floor and row-step kind into three bytes without
expanding the 24-byte dot-admission packet. Both planner and consumer borrow it.
The shared gate/up admission has no independently filtered codebook masks, so its
two descriptors cannot disagree about directory offsets. The directory cache
compares the complete plan. At floor16, counts13/15 produce two cold TM12 tiles;
counts16/17/31 use native TM16 tiles. Down retains its own role-bound plan and
source-width partition authority, including 96-column prepared slices.

Release and Integration builds pass for gfx906 and gfx1100. The first Integration
attempt found a test-only default-construction error for a required-admission
argument packet; captured bytes now enter through `bit_cast`, and the final build
passes. Release final allocation contains all18 physical formats ×2 roles ×2
adaptive/pure variants:72 kernels,54–86 VGPRs,52–91 SGPRs,3536–6608 LDS bytes,
wave64 and zero private memory, register spills or scratch instructions. gfx906
emits no native WMMA body. These are compiler/ISA facts, not achieved occupancy.

The device-free registration passes19 cases in0.46s, including row ownership,
source aliases, malformed plans and capture-cache distinctions. Four Release
smokes (Q4_0, Q2_K, IQ1_M and IQ4_XS) pass complete dirty replay, mixed counts,
branch suppression and a captured96-column down/fold transaction. The latter
poisons consumed down scratch and both publication banks; independently omitting
down or fold must fail, followed by two successful dirty restored replays.
Every native owner retired and the profiling hold was released. The source-bound
receipt is `grouped-wide-smoke-receipt.json`; allocation and exact disassembly
are bound by `grouped-wide-resource-review.json`.

The focused feature gate contains nine registrations, preserving the old
microtile inventories and adding MacroReplay, MacroPureReplay, MacroSplitReplay
and MacroTimingReplay. New coverage includes all21 source formats, every floor
through mixed IQ4 tables, physical23 promotion with13/31 and15/31 histograms,
odd K32 tails, actual imported timing parents and exact role/row-plan assertions.
Those registrations are explicitly listed in ProductionTestPreflight. The nine
focused registrations pass in256.91s, containing162 actual GTests with no skips.
The wrapper's exact stdout-summary assertion rejected CTest's optional spelling
without the zero-failure clause. `grouped-wide-feature-authenticated.json` binds
the original returncode0, nine successful registrations, all162 native outcomes,
unchanged26-file producer and complete owner retirement; the original failed
wrapper receipt is preserved and no tests were repeated.

Canonical timing qualifies all five complete-operation cases: 17 live counts,
four policies and 31 paired samples of eight operations, totaling 10,540 rows.
Each accepted case has unchanged producer hashes and no observed competing work;
five overlapped hotset attempts remain excluded. `grouped-wide-summary.json`
and `grouped-wide-receipt.json` bind the results. Ratios below are Dot4 median
latency divided by candidate median latency at full live work.

| Case | Dot4 µs | N32 | N64 | Wide floor16 |
|---|---:|---:|---:|---:|
| Qwen uniform | 924.560 | 1.3070× | 1.2577× | 0.9161× |
| Qwen hotset | 753.457 | 1.1690× | 1.1097× | 1.0206× |
| Qwen power-law | 942.205 | 1.0511× | 1.0046× | 1.0593× |
| Compact Q4_0 | 293.021 | 1.0489× | 0.9973× | 0.9018× |
| Compact IQ2_S | 334.581 | 1.0687× | 0.9825× | 0.9088× |

Wide floor16 retains near-Dot4 sparse uniform behavior (live64/128/256 ratios
0.9856/0.9944/0.9889), but loses full uniform and both compact cases. It improves
power-law live128/256/512 by 1.0416/1.0943/1.0593×. These results do not qualify
an Automatic policy. All five exact profiles now qualify 35 independent counter
passes. The timing/profile join is `grouped-wide-profile-timing-join.json`
(SHA256 `720e84f19001e7712a5ccf580591c6cd2abadf8529264944fa92e2ebb88dee75`).
IQ2_S gate/up uses 78 logical VGPRs, 5968 logical LDS bytes and zero scratch;
hardware allocation is 80 VGPRs and 6144 LDS bytes. Its HIP resource query permits
nine resident blocks per SM; this is not achieved occupancy. Hot/cold/mixed
profiles report zero LDS bank conflicts. The down profile uses 86 logical VGPRs,
3664 logical LDS bytes and 4096 allocated LDS bytes.

The first reader rejected a valid native pass because it rounded LDS in bytes
instead of DWORDs. GFX7–GFX11 allocation granularity is 128 DWORDs (512 bytes).
A separately versioned reader and eleven regression controls authenticate the
correction without modifying the frozen 38-file producer or recollecting valid
raw observations. `grouped-wide-reader-v2-correction.json` retains the original
failure and corrected proof.

The next reviewed experiment uses a fine native-column grid: each hot block
owns one N64 gate/up or N32 down chunk. Cold experts retain TM12/TN256 or
TM16/TN128 using every fourth block; the other blocks return uniformly before
any shared publication. This exposes hot column parallelism while preserving
cold arithmetic and the sealed row plan, at the cost of extra inactive cold
blocks. The fine grid is now applied after archiving the completed serial-chunk
evidence, with pure-native 96/288 column-tail coverage. Normal guarded Release
and Integration builds pass for gfx906/gfx1100. The actual Release object has
72 variants: 51–87 VGPRs, 50–106 SGPRs, zero private memory, memory spills and
scratch instructions. IQ1_M down/adaptive has two guard-authenticated scalar
register moves; these do not spill to memory. All 19 device-free dispatch cases
and four Release split-replay smoke tests pass.

The promoted-storage capture test had one additional old-grid assertion outside
the initial patch. It now checks N64/N32 native columns; the test owner was rebuilt,
and all nine registrations/162 native tests pass in 262.65 seconds with no skips.
The interrupted attempt and original binary remain in `grouped-fine-feature-repair/`. Release kernels and
profile/timing producer identity did not change.

The separate timing validator passes 13 negative controls; the profiler reader
passes nine raw-record controls plus finite-scope CLI rejection checks. The
serial-chunk producer is preserved in
`grouped-wide-producer-snapshot/snapshot.json` (44 files plus the tracked patch);
the completed join and corrected reader are also saved in
`grouped-wide-final-qualification/`. The finer-grid candidate has its own complete
producer, replay, timing and profile evidence:
`grouped-fine-summary.json` authenticates all 10,540 samples, and all five profiles
have seven independent accepted counters. `grouped-fine-profile-timing-join.json`
binds them (SHA256 `092b468b226d748de40d711d317e07ceb297c0c7e83f91176e25ae27abd51983`).
The 72-file archive is
`grouped-fine-producer-snapshot/snapshot.json`, SHA256
`c649d9f3064f4c26d8c2981970c0fa51cd4805eb57ee20c0ab8dad5bd499812e`.

| Fine-grid case, full512 | Dot4 µs | N32 gain | Fine adaptive gain |
|---|---:|---:|---:|
| Qwen uniform | 911.225 | 1.3020× | 1.1890× |
| Qwen hotset | 755.637 | 1.1747× | 1.1201× |
| Qwen power-law | 942.160 | 1.0536× | 1.0606× |
| Compact Q4_0 | 293.070 | 1.0486× | 0.9884× |
| Compact IQ2_S | 334.621 | 1.0703× | 0.9723× |

Sparse uniform live64/128/256 gives 0.9696/0.9651/0.9808×; power-law at those
points gives 1.0887/1.0639/1.0704×. Live1 still regresses approximately 10% in
Qwen shapes. These results improve the serial-chunk experiment but do not yet
justify Automatic installation. The native-only N32 policy remains preferable
at many dense work sizes.

## Completed cold-decode fence qualification

The cold-decoder scheduling fence is now qualified as an explicit candidate.
Its eight-register empty assembly constraint completes decoding before
activation publication. IQ2_S gate/up drops79→72 VGPRs and down87→81;
26 adaptive variants improve and ten are unchanged. All36 pure variants retain
identical resources. ISA review confirms the sign operations precede the barrier.
Normal guarded Release and Integration builds pass for gfx906/gfx1100 with zero
private memory, register-memory spills and scratch instructions. The remaining
compiler-proven scalar-to-vector register moves are retained as register evidence.

All19 device-free dispatch tests and four Release split-replay smokes pass.
The nine focused registrations contain162 native tests and pass in274.80s.
Five complete-operation cases retain17 live counts × four policies ×31 paired
samples, for10,540 timing observations. All five exact profiles complete seven
independent counter passes each, for35 passes.

| Fence case, full512 | Dot4 µs | N32 gain | Fenced adaptive gain |
|---|---:|---:|---:|
| Qwen uniform | 913.618 | 1.3035× | 1.2174× |
| Qwen hotset | 780.207 | 1.1645× | 1.1330× |
| Qwen power-law | 944.209 | 1.0557× | 1.0685× |
| Compact Q4_0 | 293.933 | 1.0483× | 0.9890× |
| Compact IQ2_S | 324.078 | 1.0312× | 0.9448× |

Sparse Qwen uniform live64/128/256 remains0.9561/0.9302/0.9630× Dot4.
The compact regressions and sparse/empty costs still reject blanket Automatic
installation. Pure N32 remains preferable at many dense work sizes. Production
Automatic remains OrderedDot4; the fence has not changed its admission.

`grouped-fence-summary.json` owns the timing rows.
`grouped-fence-profile-timing-join.json` binds all10,540 samples and35 passes
(SHA256 `43ca27b032e651af5e4e301f6a007800b51a6cebdafaab4aa35cd892507fea52`).
The63-file producer archive is `grouped-fence-producer-snapshot/snapshot.json`,
SHA256 `838520de0f33e8dd92be6bdb3e446baab23e2a23c1f5a6e92bddab01465da68d`.
Build, source, loaded-code, profile and numerical evidence remain bound to this
explicit fenced producer; profiler durations are not canonical latency labels.

Bounded-workloop diagnostics under `/tmp` compile but increase register use.
The separate local-thread-id variant is also diagnostic. Neither experiment
changes production dispatch or supplies a measured performance claim.
