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
and all candidate graphs retain their original pointers and policies. Ordered
leaves finish before directory16 overwrites the shared directory for native
leaves; an adaptive directory never survives that overwrite.

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

## Clean complete-operation tournament

`grouped-hybrid-receipt.json` qualifies the final Release producer with no
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
