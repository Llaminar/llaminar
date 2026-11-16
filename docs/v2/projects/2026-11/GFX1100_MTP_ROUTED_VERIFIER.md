# gfx1100 routed MTP transaction

The current grouping-owned offset-slot candidate passes serial-byte replay in
Release and Integration for15 capacity/distribution cases and180 live-prefix
keys. Ordinary AUTO-power timing shows concentrated and mixed-work gains, but
uniform full16 remains about7% slower than Dot4. Capture-time route-policy
lowering passes the nine affected Integration registrations/75 GTests. A bounded
split cold-Dot4/hot-WMMA experiment also passes byte replay, but still loses
full16 Uniform and sparse controls; it is archived and removed. Routed Automatic
is unchanged. This is a component transaction, not model-level
MTP throughput. Earlier experiments below retain their original scope.

The early routed timing drivers used `STABLE_STD`, the fixed profiling power mode.
Those paired timings below are diagnostic candidate comparisons, not canonical
ordinary-power qualification. Actual Automatic admission requires fresh
unprofiled `AUTO` timing over every retained live prefix. The corrected
`run_routed_verifier_smallrows_auto.py` verifies `AUTO` before and after collection
without changing power policy. The frozen records retain their original scope.

## Production operation and independent oracle

`Perf__GPUExpertPipeline.cpp::runExpertPipeline` retains its existing physical-CPU
source synthesis and batch GPU preparation. The new private
`ROCmRoutedVerifierTransaction.h` branch borrows those independent original
expert triplets. It uses H2048/I512/E256/top8, IQ2_S gate/up, IQ4_XS down and
physical capacities 8, 9 and 16. Each Dot4/N32/N64 candidate has separate stage,
workspace, router publication, output, retained graph and timeline owners.

The first Q8 block of each hidden row contains a one-hot row code; remaining
dimensions contain dense nontrivial values. Sparse immutable router weights
produce eight well-separated positive logits. Uniform selects eight distinct
experts per row, with no overlap between rows. Concentrated8 selects the same
eight experts for every row. This explicitly named eight-expert distribution is
not the canonical32-expert Hotset profile. The real
`routeVerifierRowsDecodeEquivalent(..., count)` produces routing IDs, weights,
Q8 hidden rows and inactive(-1,0) sentinels inside capture. Input/gate upload
happens once, and only the fixture's device-count input changes between replays.
No host route rewriting or test GPU publisher is part of the transaction.

Actual `MoEExpertComputeStage::execute` runs with the decode-equivalent verifier
purpose and pre-resolved engines. Its graph-local `MoERoutedPipelineKernelOwner`
owns the paired router/expert kernel and Q8 publication. The serial oracle has a
separate kernel, workspace and managed M1 runtime table. It executes the real
M1 `decodeRouteSelect` followed by
`groupedExpertDecodeFromRuntime(...RuntimePlacementTable)`. Original source
payloads remain alive throughout both oracle and captured replay.

Expected router IDs come from the independently declared distribution. Captured
weights and output must match M1 bytes, including positive-zero output suffixes.
Replay covers every live prefix, full→8→9→empty→full, repeated full work and
session reset without consuming an unpublished count. Consumed scratch and
outputs are poisoned. The private input-Q8 bank must retain its poison because
both RouteOwned and tiled GU consume the router's original publication.

## Capture and timing proof

The real parent graphs authenticate both GU/down roles, codebooks, N32/N64 tile
symbols, block/grid geometry, N/K, row/top-k/expert geometry, directory bindings
and each role's input/output pointers. GU reads OriginalTokens with the actual
group-token mapping; down reads GroupedRows with a null mapping. The latter
correctly rejected an initially over-broad test assertion, which was repaired
without a production change. Both the explicit RouteOwned GU packet and native
GU packet bind the same router Q8 publication.

Native-node omission changes output only when the captured route-admission
policy selects ExpertTiled. Physical8 has no native nodes; physical9/16 retain
those nodes while live≤8 executes RouteOwned. Router-Q8 omission must break
nonempty output, and restoring either family must immediately restore bytes.
Per-engine submission and Q8-reuse counters advance during warmup/capture and
remain unchanged throughout replay and timing. Timeline checks authenticate the
CapturedFragment and reject a second stale timing consumption.

Each timing cell uses31 alternating candidate-order triples with eight complete
operations per retained internal-event interval. Every sample independently
overcomes poison and passes the serial oracle; setup, poison and downloads stay
outside the event interval. The complete roster has2,232 triples across72 keys.

| Full-capacity case | Dot4 µs | N32 µs | N64 µs | Dot4/N32 | Dot4/N64 |
|---|---:|---:|---:|---:|---:|
| Uniform8 |323.499|318.114|323.209|1.01693×|1.00090×|
| Uniform9 |375.634|566.262|596.642|0.66336×|0.62958×|
| Uniform16 |580.977|898.585|947.081|0.64655×|0.61344×|
| Concentrated8, capacity8 |234.203|234.438|233.933|0.99900×|1.00115×|
| Concentrated8, capacity9 |247.728|213.333|209.112|1.16123×|1.18466×|
| Concentrated8, capacity16 |293.848|245.728|250.368|1.19583×|1.17367×|

All nine concentrated native-active cells win all31 pairs with both native tiles.
N32's range is1.13174–1.19583×, geometric mean1.15912×. N64 wins the physical9
boundary; N32 wins all physical16 medians. All nine uniform native-active cells
lose with N32 (Dot4/N32 range0.64655–0.73884×). The54 sparse/control keys retain
no native arithmetic, and dormant native topology/separate graph placement can
increase latency by up to5.18%. Physical8 results are Dot4 topology controls,
not native speedups.

The next policy must account for per-expert work distribution and physical
capacity. Unconditional WMMA would turn the concentrated win into a substantial
uniform regression. Existing adaptive row-floor candidates provide the next
bounded comparison before installing a production decision.

## Seven-path adaptive panel (frozen)

The expanded fixture retains the three original controls and adds N32Floor8,
N64Floor8, WideFloor8 and WideFloor16. It covers four independently declared
distributions at capacities8/9/16: Uniform, canonical Hotset32, Concentrated8
and MixedCohorts. MixedCohorts assigns the first eight rows to the same eight
experts and every later row to a distinct set. Its hot and cold token cohorts
allow both branch contribution and surviving-row byte checks in one graph.
Hotset32 is an ordered-work control at these capacities, with every expert
below floor8. WideFloor16 cannot execute WMMA at capacity9.

Capture checks bind each candidate's actual role, row floor, row plan, symbol
and geometry, including the wide GU256/down128-thread layout. The test-only
branch mutation helper now accepts an explicit GU or down role, retaining down
as its default for existing tests. Each branch is suppressed independently in
each role. An immutable fixture histogram predicts affected rows; unaffected
rows must retain exact serial bytes and every affected row must differ. The
actual production path still obtains all counts from device grouping. Restoring
the captured argument must immediately restore complete serial-byte output.

The expanded roster passes Release and Integration replay:12 cases,144
live-bearing keys and432 seven-path samples in each build. The diagnostic
Release timing panel passes4,464 seven-path samples,31 per key, with eight
complete transactions per interval. All captured resources report zero private
scratch. The shared profiler was drained before these GPU runs and resumed
afterward; its timed source and binaries remain unchanged. Neither custom
kernel profiles nor Automatic admission are established by this panel.

| ExpertTiled distribution (9 keys each) | WideFloor8 median wins | Wins in all31 pairs | Geometric speedup | Speedup range |
|---|---:|---:|---:|---:|
| Concentrated8 |9|9|1.20952×|1.09766–1.25798×|
| MixedCohorts |9|9|1.09562×|1.04575–1.17467×|
| Hotset32 |9|1|1.00990×|1.00080–1.02347×|
| Uniform |3|3|0.96576×|0.92127–1.04686×|

The microtile floor8 candidates still lose every Uniform tiled cell. WideFloor8
retains the native concentrated gains and economical mixed expert execution,
but its ordered branch does not replace the genuine original Dot4 producer
economically at every sparse size. WideFloor16 wins only the full16 concentrated
cell and loses the other eight concentrated tiled cells. It is not a general
replacement. Hotset32 has no WMMA arithmetic under either floor and its small
median gains are not consistently separated in the paired samples.

The next bounded diagnostic retains genuine ordered ExpertTiled for a maximum
expert count below8 and WideFloor8 otherwise, preserving existing RouteOwned
precedence. General grouping already publishes the device-owned maximum of the
same local counts; no host readback or extra histogram scan is required. Bind
that publication explicitly to both complementary admissions and reject
grouping paths without a matching producer. Common SwiGLU/final reduction must
keep unfiltered admission. Directory reuse must include producer admission or
use a common unconditional builder: an inactive ordered builder must not mark a
shared fixed16 directory as published. A cold-first/hot-last cohort will cross
the max-count boundary inside one retained physical16 graph. Extra dormant
launches require fresh complete-operation timing before any Automatic change.

Expanded evidence:

- Release replay: `mtp-routed-verifier-replay-1794777292352196700/receipt.json`.
- Integration replay: `mtp-routed-verifier-replay-1794777341433046041/receipt.json`.
- Release timing: `mtp-routed-verifier-timing-1794777402249773307/receipt.json`.
- Analysis: `routed-verifier-adaptive-summary.json`.
- Builds: `routed-verifier-adaptive-{release,integration}-build.log`.

The adaptive panel is frozen in `mtp-routed-verifier-adaptive-snapshot/snapshot.json`:
three evidence records and27 files, manifest SHA256
`c79d830ea0b9a0f079d59f77ada4096991ac790a9be8666d85ceccb0b02984ff`.

## Maximum-count experiment: correct, uneconomical

`MaxCountWide8` is an explicit recording candidate; Automatic remains unchanged.
The same base route admission still selects RouteOwned through64 live slots.
For ExpertTiled work, complementary immutable packets select genuine ordered
GU/down when the device maximum is below8 and WideFloor8 when it is at least8.
All directory builders, SwiGLU and final publication retain the unfiltered base
admission. Directory reuse therefore cannot depend on a dormant writer having
published a directory. Ordered and adaptive consumers borrow the same counts,
offsets and maximum and execute on the same exact stream.

The existing grouping provenance now retains a move-only `CountsOffsetsMaximum`
publication. Only successful general and mask-selected grouping publish it,
after their existing maximum reduction is enqueued. Small, shared, failed,
abandoned, reset and runtime-plan replacements revoke it even if the allocation
address survives. Consumers validate pointers, capacity, expert count, top-k,
device and stream. A const borrowed publication cannot be copied or moved out
to preserve stale authority. This records producer identity only; no live count
is copied to the host and no new device allocation or classifier is introduced.

The fixture adds an eighth candidate and a ColdThenHot distribution. Eight
distinct cold rows precede eight rows sharing eight experts, so live9–15 stays
ordered while live16 crosses max7→8 in the same retained graph. Capture checks
authenticate both tiled packet guards and the real maximum reduction. Whole
family omissions prove complementary ownership. Separate maximum-producer
omissions inject legal values1/8 before launch and omit the opposite writer,
proving the captured word controls execution. An invalid maximum is ignored
under RouteOwned as required. These overrides are untimed test controls; normal
replay and timing always use the real device producer.

The roster contains15 cases and180 live-bearing keys. New explicit
preflight registrations are `V2_Integration_ROCmRoutedVerifierTransactionReplay`
and device-free `V2_Integration_MoEGroupingMaximumPublicationReplay`. Both builds
compile gfx906/gfx1100 and sm86 with spill guards intact. All15 native replay
cases pass in both builds (180 keys/540 eight-path samples each), including
maximum-producer omissions and max7→8 replay. All captured kernels report zero
private scratch. Three final host registrations pass after the move-only fix.

The complete Release panel has5,580 eight-path samples,31 per key. This
candidate is rejected for Automatic economy:

| Selected family | Keys | Median wins | Wins in all31 pairs | Geometric speedup | Range |
|---|---:|---:|---:|---:|---:|
| RouteOwned controls |135|19|0|0.91793×|0.74573–1.00328×|
| Ordered ExpertTiled |26|0|0|0.91698×|0.89279–0.94209×|
| Adaptive ExpertTiled |19|15|11|1.05876×|0.93689–1.18271×|

Physical8 controls retain identical Dot4 topology. The material RouteOwned
losses occur in wider captures retaining extra dormant families. Every
ordered-tiled key loses, and the concentrated capacity9 boundary also loses.
Full concentrated16 improves294.363→248.888µs, but pure WideFloor8 reaches
232.273µs. Full mixed16 improves only455.575→452.390µs, versus428.100µs with
WideFloor8. Correct device selection alone does not make the combined graph
economical. No counter sweep is needed to reject a measured slower candidate.

Next, test cold GU4×256 and down12×128 inside a dedicated wide per-expert
producer, retaining native16 rows above floor8. Thread widths already match
the wide producer, so this can avoid another retained family and global
classifier. It changes mixed-cold geometry as well as all-cold work and still
reserves the largest branch's registers/LDS; only fresh complete timing can
qualify it. A smaller alternative is guarded directory publication with reuse
identity including row plan, maximum pointer and branch, but it retains dormant
consumer launches. Retire the rejected maximum experiment when replacing it;
do not install its policy or retain unused maximum-publication machinery.

Evidence under the same root:

- Release replay: `mtp-routed-verifier-maximum-replay-1794779174006237865/receipt.json`.
- Integration replay: `mtp-routed-verifier-maximum-replay-1794779293876591732/receipt.json`.
- Timing: `mtp-routed-verifier-maximum-timing-1794779375901465721/receipt.json`.
- Analysis: `routed-verifier-maximum-summary.json`.
- Final host checks: `routed-maximum-host-final-receipt.json`.
- Builds: `routed-maximum-{release,integration}-build.log`.

This rejected experiment is frozen in `mtp-routed-verifier-maximum-snapshot/snapshot.json`
(four records,35 files), manifest SHA256
`439e02014420426610600452f7b2222f29cb219863d5f3ebd106ab35e151dac7`.
Its extra family and maximum-publication plumbing are retired by the next experiment.

The frozen shared profiling sweep continues to qualify its original committed
producer, not these changed kernels.

## Rejected small-row experiment (retired)

The now-removed `WideSmallRowsWmma` retained one directory and adaptive projection per role,
with cold GU4×256/down12×128 and hot native16 at floor8. The original whole-expert
count selects the branch; no global maximum or separate tiled family is retained.
The typed cold geometry is encoded in the existing sealed row-plan byte and
belongs to directory reuse and captured producer identity. Compile-time cold
rows precede the two existing role/adaptive template booleans. The old wide
candidate keeps GU12/down16 as an independent measured control.

Read-only review identified that a TM4 conservative directory can exceed the
old dual-TM12/TM16 reservation: M512/top8/E8 needs1032 words, versus616. The
canonical workspace BOM now reserves the maximum of those mutually exclusive
layouts. A focused bank-sizing replay is explicitly registered in preflight,
and the host coverage includes small expert counts and large route capacities.
The routed native registration is now appended after CMake's canonical list
initialization, so that initialization cannot overwrite its membership.

The routed fixture retains five route distributions and all live prefixes,
per-role arithmetic omissions, dirty replay/reset and independent serial bytes.
Capture inspection rejects separate ordered tiled families and requires one
correctly bound directory plus one adaptive projection per role. Release and
Integration builds pass for the complete shipping target set. Both native replay
panels pass all15 cases/180 live-prefix keys; all four focused host registrations
pass, including the workspace reservation regression.

The diagnostic timing panel completes5,580 eight-path samples. WideFloor8 wins
all31 pairs in each of the19 hot-expert cells, geometric gain1.16514× against
Dot4. SmallRows wins only16/19 medians and13/19 complete paired sets, geometric
gain1.06147×; its concentrated16 result is276.583µs versus218.633µs for WideFloor8.
It improves some mixed/cold cases but is not a general replacement. Uniform16
still favors Dot4 (579.682µs versus634.668µs for SmallRows). Retained capacities9/16
pay dormant directory/native launches when live rows select RouteOwned; physical8
controls with identical topology cannot establish a native advantage. Automatic
remains unchanged. The next design must preserve sparse economy as well as hot
expert throughput, and its actual selection needs ordinary-power qualification.

Small-row evidence:

- Release replay: `mtp-routed-verifier-smallrows-replay-1794781806606132906/receipt.json`.
- Integration replay: `mtp-routed-verifier-smallrows-replay-1794781875703293656/receipt.json`.
- Diagnostic timing: `mtp-routed-verifier-smallrows-timing-1794781955231543362/receipt.json`.
- Analysis: `routed-verifier-smallrows-summary.json`.
- Host replay: `routed-smallrows-host-receipt.json`.
- Frozen41-file/four-record snapshot: `mtp-routed-verifier-smallrows-snapshot/snapshot.json`,
  SHA256 `46bbb9a34d6d9771f90f14f23370bc1d568511a3f6a6ac383a593871c255d10a`.

The slower SmallRows implementation has been retired. Its workspace-capacity
correction remains because the existing fixed-TM4 planner needs the larger bound.
These old STABLE_STD panels remain diagnostic evidence, not ordinary-power
qualification or an installed policy.

## Grouping-owned offset-slot candidate (in development)

The explicit WideFloor8 candidate can publish its packed directory in the existing
active-expert-list launch, after count/scan and before scatter. It retains the
separate ascending raw active-ID bank. Cold counts1–7 publish one descriptor at
the original offset; counts8+ publish native16 descriptors at offset+0/16/32/etc.
This preserves repeated routes, including counts17/33/128. Interior sentinel holes
are legal because each native CTA checks its own descriptor independently.

The existing grouping owner publishes the binding only after the whole grouping
preparation succeeds. Lexical borrows validate original banks, physical geometry,
workspace, device, stream, generation and owner lifetime. Reset/rebinding and an
incompatible role planner revoke that authority. Both native roles borrow the same
packed bank; neither role planner launches. The arithmetic and native packet stay
unchanged. Initial explicit admission is physical M9/16, top8/E256, H2048/I512,
static complete token publication. Automatic remains unchanged.

Device-free coverage includes duplicate intervals, invalid bindings, projection
packet mismatches, failed/reset preparations and owner destruction/address reuse.
Captured tests add supplied duplicate-route grouping, non-sentinel dirty reset,
combined-publisher omission, actual-router live prefixes, and old-planner absence.
The bounded comparison now retains Dot4, existing WideFloor8 and the offset-slot
candidate; physical M8 retains the first two identical RouteOwned controls.
The initial AUTO-power panel is complete: both builds pass15 cases/180 live-prefix
keys; Release timing retains5,580 paired sample rows. All five focused host
registrations and the native duplicate-route registration pass. Builds cover
gfx906/gfx1100 and sm86 with the spill guard intact. The combined publisher uses
12 reported registers and72 LDS bytes, compared with10/72 for raw IDs alone;
captured inventory reports no private scratch. Counter profiling remains pending.

| Offset-slot candidate versus Dot4 | Cells | Median wins | All31 wins | Geometric speedup |
|---|---:|---:|---:|---:|
| Retained M9/16 RouteOwned live0–8 |90|72|10|1.00779×|
| Concentrated8 tiled |9|9|9|1.20480×|
| MixedCohorts tiled |9|9|8|1.13825×|
| Hotset32 tiled |9|9|7|1.04064×|
| ColdThenHot tiled |9|8|8|1.08115×|
| Uniform tiled |9|4|3|0.98813×|

Physical M8 has no offset-slot candidate and is excluded from this comparison.
Uniform full16 remains slower:334.609µs Dot4 versus356.189µs offset slots
(0.93941×). Full concentrated16 improves190.307→147.632µs; full mixed16 improves
275.863→240.573µs. This does not qualify unconditional Automatic admission.

Baseline evidence:

- Integration: `mtp-routed-verifier-offset-slots-replay-1794804992177648932/receipt.json`.
- Release replay: `mtp-routed-verifier-offset-slots-replay-1794805067512234683/receipt.json`.
- Release timing: `mtp-routed-verifier-offset-slots-timing-1794805101943043283/receipt.json`.
- Analysis: `routed-offset-timing-summary.json`; host: `routed-offset-host-receipt.json`.
- Registered native replay: `routed-offset-native-registration.xml`.
- Frozen baseline: `mtp-routed-verifier-offset-slots-baseline-snapshot-1794805635615682182/snapshot.json`,
  SHA256 `5b748dd5dc43979a7b042d80c46c4a70a9e7adc3c641a76b0be40d10214a60cf`.

Two earlier collection attempts remain incomplete diagnostics: one rejected an
incorrect expected encoding of the device UUID, and one rejected a timing row
split by merged MPI stdout/stderr. The final collector parses the separately
retained rank stdout and validates exact XML, source, geometry, device, candidate
and sample inventories. It does not repair or qualify either failed attempt.

A bounded early cold-column exit was measured and rejected. It retained
block-zero validation, arithmetic, row plans, ABI and topology, but Uniform
full16 worsened to0.93347× versus Dot4; uniform geometric gain was0.98189×.
All15 cases/180 cells/5,580 samples passed correctness; GU72/5968 and Down71/3664
register/LDS resources remained unchanged, with zero private storage or spills.
The source change is reverted. Rejected evidence:
`mtp-routed-offset-cold-exit-rejected-snapshot-1794806086603312458/snapshot.json`,
SHA256 `d7773f0a132ba7fc359c2385f1d8334a8a885579e2a5a6c48d587c8e360a4e76`.

ISA inspection also identified substantial repeated route-policy work before
directory access: each CTA evaluates immutable geometry/exact-key comparisons
and divides live slots by top-k. The new lowering resolves the same canonical
policy into ordered slot intervals during capture, leaving counts device-owned.
It preserves full exact keys, table priority and empty-work semantics. Focused
device-free replay compares every original codebook pair, nonmultiple slot
counts, exact-row boundaries and extreme integer capacities with the unchanged
scalar selector. Both the full dispatch registration and explicit
`V2_Integration_ROCmGroupedLiveSlotPolicyReplay` pass
(`routed-slot-policy-host.xml`). Release replay and ordinary AUTO-power timing
pass15 cases/180 cells with540 and5,580 paired sample rows respectively. The
capture packet grows40 bytes; the exact selected Dot4 GU/down and wide GU/down
keep their VGPR/LDS allocation, reduce SGPR allocation by2, and have zero private
storage or memory spills. The old first160-instruction entry prefixes contain
one reciprocal sequence each; the lowered prefixes contain none.

This simplifies policy execution but does not close sparse economy. Full16
Uniform is334.141µs Dot4 versus356.971µs offset slots (0.93605×). Concentrated8
is186.071→144.971µs (1.28351×); MixedCohorts is272.306→237.731µs (1.14544×).
Independent panels are not an interleaved old/new comparison, so their absolute
latency differences do not establish a causal speedup from lowering. The within-
panel Dot4 comparison remains the candidate-selection evidence. Automatic is
unchanged. The focused native Integration feature suite passes9 registrations/75
GTests, including routed/shared transactions, grouping offsets, all-codebook
RouterQ8, population and live/source-row/phase controls (318.04s). Selected-kernel
counters for this exact candidate remain pending.

- Release replay: `mtp-routed-verifier-offset-slots-replay-1794807107887825269/receipt.json`.
- Release timing: `mtp-routed-verifier-offset-slots-timing-1794807173092290269/receipt.json`.
- Summary: `routed-slot-policy-timing-summary.json`.
- Exact resources/ISA: `routed-slot-policy-resource-comparison.json`.
- Frozen component: `routed-slot-policy-qualified-component-1794807530415389501/snapshot.json`,
  SHA256 `26b5cec2bde4053e380be6e62724e7a332ece5330e4f12f25ace67718e2c0791`.

The focused Integration run is preserved separately:
`routed-slot-policy-feature-suite-1794807970488919892/receipt.json` and
`routed-slot-policy-integration-feature-snapshot-1794808346598358370/snapshot.json`,
SHA256 `da6cbfd8f9f868e40562b7a6842679d975ca6d919eb182690d047be2ff41f20b`.

## Evidence and outstanding work

Evidence root: `/tmp/llaminar-gfx1100-wmma-prefill-evidence`.

- Release replay: `mtp-routed-verifier-replay-1794776201785844090/receipt.json`.
- Integration replay: `mtp-routed-verifier-replay-1794776290392205382/receipt.json`.
- Release timing: `mtp-routed-verifier-timing-1794776214266873009/receipt.json`.
- Analysis: `routed-verifier-forced-summary.json`.
- Builds: `routed-verifier-transaction-release-build-r3.log` and `routed-verifier-transaction-integration-build.log`.

Selected-kernel profiles, partial-ownership/interior-hole controls, broader source
coverage, and actual Automatic admission/replay/timing remain open. Current
coverage proves original local complete-expert work and device-counted suffixes;
it does not qualify arbitrary holes or collective ownership. The shared-expert
Automatic source/role profiler sweep runs independently against its unchanged
producer. No full production-preflight or model E2E claim is made.

The initial forced panel is frozen in
`mtp-routed-verifier-forced-seed-snapshot/snapshot.json`: three evidence records,
25 files, manifest SHA256
`898ec46354c7be66625841009f3c296f8a5af27e582555cbc59e4ff331b4eb4a`.
It preserves the exact Release/Integration binaries and source before subsequent
adaptive-policy experiments.

## Bounded independent cold/hot leaves: correct, rejected

A diagnostic M16/H2048/I512/E256/top8 candidate separated the original cold
GU4/TN256 and down12/TN128 Dot4 bodies from pure native16 GU/N64 and down/N32
leaves. One grouping-owned offset-slot directory retained original expert
intervals. Cold counts1–4 published row0, counts5–7 published rows0/4, and hot
counts≥8 published sixteen-row steps. Cold down consumed only row0. This avoided
sharing each cold leaf's register allocation with the native branch, at the cost
of two additional retained launches and dormant native CTAs.

The bounded key required original source13/superblock IQ2_S gate/up and
source4/superblock IQ4_XS down, including separate physical decoder24. Capture
validated every original expert descriptor. Direct native submission required a
live grouping borrow with the split layout; cold and hot leaves independently
checked physical and original source identity. Device-free tests rejected
geometry, source, publication, pointer and device mutations. No live count was
read by the host and no Automatic key was installed.

Release and Integration compiled gfx906/gfx1100 and CUDA86 with spill guards.
Six host registrations passed. Integration replay passed Uniform, Concentrated8
and MixedCohorts at every live prefix0–16:51 cells and153 four-arm sample rows.
The real router/stage transaction retained independent M1 bytes across shrink,
empty/reset and repeated replay. Physical cold/hot omissions in each role proved
mixed-cohort writer ownership. Supplied-route replay also covered duplicate
counts128/17/33/5/7, mixed counts, empty and repeated work. All five new producer
symbols were bound to their Release code objects/core and inspected for scratch
instructions; none occurred. Native GU/down used56/52 VGPRs, cold GU/down40/45,
and the grouping publisher13. Logical LDS was5824/3520,160/432 and72 bytes
respectively; all reported zero private storage and register spills. These are
static resource/ISA observations, not achieved-occupancy or counter qualification.

Ordinary AUTO-power Release timing retained four separate operation owners:
Dot4, WideFloor8, WideGroupingOffsetSlots and SplitGroupingOffsetSlots. Two cases
covered34 live-prefix cells with31 alternating-order four-arm observations each
(1054 sample rows, eight complete operations per interval). Every sample passed
serial bytes. MixedCohorts was replay-only for this bounded decision.

| Full M16 case | Dot4 µs | Wide offset µs | Split µs | Dot4/Split | Split wins vs Dot4 |
|---|---:|---:|---:|---:|---:|
| Uniform |336.027|359.367|340.707|0.986264×|0/31|
| Concentrated8 |244.116|200.436|208.126|1.172925×|31/31|

Across18 RouteOwned control cells, Split loses every median and has geometric
Dot4/Split0.975746×. Across eight tiled Uniform cells, only three medians/two
all-pair cells win (geometric0.995632×). All eight tiled Concentrated8 cells win
against Dot4 (geometric1.116304×), but the full-M16 fused offset candidate beats
Split in all31 pairs. Lower registers therefore did not deliver the required
complete-operation economy. The split enum, directory layout, kernels, submit
extension and dedicated tests were removed together. The qualified earlier
Wide offset-slot publication and live-slot lowering were restored byte-for-byte;
normal routed Automatic remains Dot4.

Evidence under the local campaign root:

- `routed-split-timing-summary.json`: both source/binary-bound timing receipts.
- `routed-split-{release,integration}-build-r2.log`, `routed-split-host-receipt.json`.
- `routed-split-integration-{Uniform,Concentrated8,MixedCohorts}_M16.log`.
- `routed-split-release-isa/inspection.json`: five exact Release producer joins.
- `routed-split-bounded-snapshot-1794815197849811983/snapshot.json`, SHA256
  `b8886aa237c004570792b47e9c5570196babbc0113c1d080e0a83c737231535e`.
- `routed-split-retirement.json`:14 explicit restorations from the frozen
  pre-experiment source; unrelated edits and prior qualified routed work retained.

No hardware-counter qualification, actual-Automatic speedup, all-format routed
coverage or model-level throughput is claimed by this rejected experiment.

After retirement, fresh Release/Integration shipping-target builds and all six
focused host registrations pass. The affected feature suite passes all nine
registrations/75 GTests in
`routed-split-restored-feature-suite-1794815350727294757/receipt.json`.
This rechecks the retained ordinary/shared/routed MoE, router-Q8 all-codebook,
original-source, live-row and projection-phase paths on the restored core.
