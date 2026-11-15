# Grouped prefill operation qualification

## Serving boundary

The canonical-route component tournament measures a useful public operation:
grouping, expert gate/up, SwiGLU, down publication into original route rows, and
a separate ordered fold. Its result does not establish ordinary single-device
prefill performance. Ordinary Qwen prefill normally consumes the router's
published Q8 hidden rows and folds directly into token outputs.

`QwenGraph` selects direct ordered folding for ordinary prefill and binds the
shared router publication. `ROCmMoEKernel::executeGroupedPrefillProjection`
consumes that publication when its source, device, geometry and capture ownership
match. Selected-route and split-projection operations have distinct contracts.
The broad `Unmeasured` operation value must never act as an Automatic admission
key: it also describes other publication and projection combinations.

## Component proof

The ordinary-operation fixture must retain a real router and its Q8 producer,
separate router output tensors, the expert kernel, prepared weights, workspace,
device live-count scalar, and complete captured operation. Controlled expert
route inputs remain explicit injected data; they do not claim to be the
router-selected distribution. Every timed transaction includes the real routing
operation, controlled expert grouping, and direct token publication. Qwen
ordinary-operation qualification uses the all-local shard1 cases; shard2/4
controls intentionally discard routes and cannot represent ordinary single-device
work economics.

The actual imported timing parent must prove:

- Every repeated router producer reads the intended FP32 input and writes the
  published Q8/scales addresses using the retained device count.
- Native gate/up nodes consume those exact addresses with original-token
  indexing and the grouping authority's index pointer.
- No separate expert hidden-row quantizer was recorded.
- The router-reuse submission counter increases during capture and stays fixed
  during replay. It describes host submissions, not device branch executions.
- Each candidate repairs poisoned active Q8/scales and output bytes. Inactive Q8
  tails remain poisoned because the producer does not own those rows.
- Omitting all repeated Q8 producers breaks the output; restoring them recovers
  exact bytes without another candidate repairing the banks.
- For mixed expert counts, independently omitting the ordered and native down
  branches breaks token output. Existing canonical-route tests additionally
  prove the surviving experts' individual contribution bytes.

Cross full, empty, partial, row-floor and restored live work in retained graphs.
Inspect each timing sample before launching another candidate. Keep source-row
serial oracles and all-codebook canonical replay alongside this operation proof.

## Automatic selection

Qualify ordinary-operation timings before installing an Automatic entry. Use
one immutable key containing the full-width Complete/TokenRows/AllRoutes
operation, physical bucket, H/I/E/top-k geometry, all four physical/source masks,
original-token versus route-slot addressing, and published-Q8 versus internal
quantization. Keep physical IQ4 variants and source identities distinct. Resolve
the key once against cached native architecture/resource capabilities.

Live expert counts remain device-owned. Existing whole-expert row admission
selects ordered or native arithmetic from the same count in both projections;
no host histogram, count download or recapture is needed.

PerfStats must consume a caller-owned successful-submission result from the
low-level pipeline. Return the actual selected engine, row floor, role geometry
and admission after all submissions succeed. Do not independently reselect a
legacy tile in the reporting layer or infer the executed branch from capacity.

## Current status

The ordinary-operation fixture and its focused all-codebook/Qwen replay
registrations are implemented. Normal Release/Integration builds and both focused
replay registrations pass (24 native cases,51.87s). The ordinary Release timing
campaign is complete: five cases ×17 device-live points ×six policies ×31 samples
=15,810 qualified timing rows. The roster is Dot4, pure N32/N64, adaptive
N32Floor16/N64Floor16, and WideFloor16. Every sample times eight complete captured
transactions with internal event boundaries, then observes output and published
Q8/scales before another candidate can repair them.

The five cases are all-local Qwen shard1 uniform, hotset and power-law at
H2048/I512/E256/top-k8/capacity512, plus uniform compact Q4_0 and IQ2_S at
H512/I256/E8/top-k8/capacity512. Route IDs and weights are explicit injected
fixture inputs; the actual production router writes separate route outputs and
publishes the real Q8 bank consumed by the expert pipeline. These timings include
router work and direct TokenRows publication, not canonical-route folding or a
real model's chosen routing distribution.

At full512 live rows, speedup over the same operation with explicit Dot4 is:

| Case | Fastest measured policy | Speedup |
|---|---|---:|
| Qwen uniform | N32 | 1.20750× |
| Qwen hotset | N32 | 1.15204× |
| Qwen power-law | WideFloor16 | 1.05358× |
| Compact Q4_0 | N32 | 1.05295× |
| Compact IQ2_S | N32 | 1.04256× |

Sparse work still favors Dot4; pure N32 loses to Dot4 at full Qwen power-law
(0.97897×). There is no universal winning candidate. The accepted operation corpus motivates population-based selection. The later
Automatic experiments below have their own correctness and timing evidence;
compact mixed-count work and unsampled populations retain no performance claim.

Evidence is
`/tmp/llaminar-gfx1100-wmma-prefill-evidence/grouped-serving-timings/campaign.json`,
producer SHA256
`90dfa911df343259ac4ab22783bb91eeeda7368c1958d84b8e33ac46549e9610`.
The campaign records complete canonical timing and restored AUTO power. This
ordinary-operation corpus is separate from the canonical-route shard2 controls.
The rejected bounded511-worker experiment is archived and the previously
profiled fine-grid fence kernel is restored. Grouped Automatic remains Dot4.
The narrower128-column gate/up experiment is complete but unpromoted: both
full-target builds,72-variant ISA/resource inspection,186 native cases and15,810
timing rows pass. Normalized wide gains change only0.9988–1.0138× across the five
cases and leave the three-family requirement intact. No runtime hardware-counter
qualification is claimed; tooling and the88-file archive are retained in the
[detailed investigation](GFX1100_GROUPED_WMMA_ADMISSION.md).
## Device-population Automatic experiments

The published feature checkpoint remains `2f3fe48cf`; its grouped Automatic
selection is ordered Dot4. The working candidate adds an exact immutable key
and one device-owned population decision, retaining RouteOwned precedence.
Qwen admission requires M512/H2048/I512/E256/top8, original token indices,
router-published Q8, physical GU13/down24 and source GU13/down4. Compact
admission remains unfinished. Source partitions and publication arithmetic
are unchanged.

The first retained topology captured guarded Dot4, N32 and WideFloor16 families.
It passed 231 GPU cases and three host registrations across Release/Integration,
with zero memory spills on gfx906/gfx1100. Its complete Qwen uniform tournament
contained 35 live extents × seven policies ×31 samples, eight operations per
sample. Automatic won only four extents, geometric-mean Dot4/Automatic0.840273;
full512 reached1.13450×. The extra91,488 dormant native CTAs per operation made
that topology uneconomical. It is rejected, with57 files archived under
`grouped-population-three-family-snapshot/snapshot.json`, SHA256
`16918ccb51f9ca6cc700d542d8f34aa05f00ed6b7482e7bbc984f6414b441b6f`.

The replacement fuses the arithmetic families into one128-thread consumer per
projection role. Global N32 uses16-row directory entries even at concentrated
counts7–15; other cold Qwen GU tiles retain12 rows, avoiding overlapping native
writers. The actual Automatic replay passed family omissions, disjoint mixed
writer omissions, changed inputs and retained reset/replay. Release gfx906 and
gfx1100 compilation passed; native GU/down used72/71 VGPRs,3664 LDS bytes and
no private memory or memory spills. Four focused host policy tests passed.

That static-grid fused version completed the same7,595-sample tournament:
8/35 extents beat Dot4, geometric mean0.986162×, full512 gain1.152×. Small work
still lost (empty0.900×, live1 0.916×), so fusion alone is not qualified for
Automatic promotion. Its36-file archive is
`grouped-population-fused-static-snapshot/snapshot.json`, SHA256
`3c4fe9782fa3021dee6c2441e7ac74ec8a9f3019b9df1890f2968fdac4599177`.
The detailed decision is `grouped-population-fused-tournament-decision.json`.
No hardware-counter qualification is claimed for either rejected topology.

The next experiment used2,047 persistent workers, gate/up z=2,
and a device-published live directory length. The trailing count word fits the
existing canonical work bank; each role rebuilds it on the same stream. Workers
visit only live tiles, and the odd stride rotates cold column ownership.
A128-expert mixed15/16-row trace crosses the worker bound in both roles.
This differs from the previously rejected511-worker candidate. Five host policy
tests and the full Qwen Automatic replay pass. Both architectures compile with
zero private memory or vector spills; GU/down use57/68 VGPRs,3664 LDS bytes and
22 active blocks per WGP. ISA has39/51 scalar-to-vector register moves and
151/187 readlane instructions, without scratch loads/stores.

The7,595-sample tournament wins13/35 extents with geometric mean0.987460×.
Empty/live1 improve to1.024×/1.012×, but full512 falls to1.005× against the
explicit N32 candidate's1.212×. This variant is also rejected. The52-file archive
is `grouped-population-persistent2047-snapshot/snapshot.json`, SHA256
`af782a9f570a0769fab28605f3a50c46509996cb43f9e853dc798545909ee3a7`.
`grouped-population-persistent-tournament-decision.json` owns all35 cell medians.
Hardware counters remain uncollected. Next work targets loop-carried register
lifetimes and compact scheduling when the global decision is wholly Dot4; neither
is a qualified implementation yet.

The new classifier, planners and consumers compile in their own HIP shard.
Its IQ2_S lookup table is initialized within the existing runtime-generation
transaction before capture. Shared arithmetic helpers retain one source; the
canonical workspace geometry is in a lightweight header usable by HIP and the
workspace BOM. This limits subsequent tuning rebuilds to the new kernel family.

Further complete-operation experiments retain the same35 live extents,
seven policies and31 samples per cell:

| Variant | Geometric Dot4/Auto | Winning extents | Full512 |
|---|---:|---:|---:|
| Fixed Qwen geometry |1.00009×|15/35|1.0182×|
| Coarse Dot4 work enumeration |1.01963×|23/35|1.0139×|
| Separate ordered/native consumers |0.99647×|9/35|1.055×|
| Separate, canonical partitions, native lane ranges |1.00672×|9/35|1.1717×|

All passed retained Qwen replay and compiled for gfx906/gfx1100 without private
memory or vector spills. The coarse-work variant is called `compact` in local
experiment filenames; it does not admit the Compact E8 model geometry. An
opaque argument-reload probe introduced232 private bytes and was rejected
before GPU execution. The separate consumers' stronger omission proof disables
actual physical down-producer nodes, covering Dot4, N32 and mixed choices,
inactive producers, changed inputs and two restored dirty replays.

Canonical partition specialization derives16 folds from
`MoEProjectionNumericalContract`, with GU/down spans4/1 and staging depths2/1.
The host submission and device arguments must match that same source contract;
descriptor provenance remains checked. This removes unused down staging without
changing arithmetic order. With the native lane fence removed, native GU/down
use67/59 VGPRs,3664/1872 LDS bytes and no scalar register moves. Ordered consumers
use56/62 VGPRs and audited register-lane moves, with no scratch memory.
Small-work launch overhead still loses3–5%, so this candidate is not promoted.

Archives and manifest SHA256:

- `grouped-population-specialized-snapshot`: `67f4419eba1e24742588e29b9b5fe896d27a25931c25204da3334b75fbd00eea`.
- `grouped-population-compact-snapshot`: `af52d88429a689547c5ea29109be6fcb68df717b9b94329d22cb7a8ced54d788`.
- `grouped-population-separate-snapshot`: `0bb0b23a291873cf7b3b4a10c11949dec860748df561aee4a2efbcf446dddfb5`.
- `grouped-population-contract-snapshot`: `038a515c5b47e0554ad93d386848a0e6ccf7a079d73395f8a32cc9dfc2fd5983` (replay/resource proof only; not timed).
- `grouped-population-native-lanes-snapshot`: `b4ca29f01f77629b7b5e4857082aa64aebe0e4830182df94f706fd096ab98e48`.

Recombining the canonical-partition consumers while confining the opaque lane
fence to the cold branch passes replay, but raises GU/down allocation to82/83
VGPRs. Its uniform tournament wins19/35 extents, geometric mean1.018508× and
full512 1.1062×. This43-file archive is `grouped-population-refused-snapshot`,
manifest SHA256 `5844372c7c069ffdfc2c5a2091951b44db28f9d5992ff68ed6ebb7dcd4d47a75`.
An early replay during the fixture rebuild used mismatched binary topology and
failed; it is retained as invalid build-order evidence. The later replay and
tournament authenticate the completed executable and core before/after execution.

The following experiment removes the standalone classifier. Gate/up directory
planning reduces its already loaded counts across eight wave64s, publishes one
typed device decision, and broadcasts it before computing tile rows. Sparse
and empty replays still publish Dot4; down retains a read-only binding. It uses
no extra arena storage. Both shipped targets compile without private memory or
vector spills. The five host policy tests and actual Qwen replay pass, including
new exact-size/one-word-short workspace probes. These probes capture the real
submission boundary without launching the auxiliary graphs: exact storage
records two kernels per role; short storage rejects with an empty receipt/graph.

The folded uniform tournament wins23/35 extents, geometric mean1.027127×,
full512 1.1130×. Empty and tiny work improve, but live15 still loses5.15% and
several middle extents lose1–2%. Concentrated routing wins35/35 extents with
geometric mean1.149130×. Decisions are in
`grouped-population-folded-tournament-decision.json` and
`grouped-population-folded-hotset-tournament-decision.json`.
The power-law tournament passes byte correctness, but several cells have large
within-cell timing variability. For live41, Automatic ranges813–3353µs; at
live48, Dot4 ranges547–4791µs and Automatic852–4760µs. The raw medians are retained
in `grouped-population-folded-power-law-tournament-decision.json`; they do not
qualify a dispatch threshold. The source of this variability is not inferred.

Integration compilation completes for gfx906/gfx1100. All1,084 final projection
metadata records have zero private storage and vector spills, in
`grouped-population-folded-integration-allocation/allocation.json`.
All231 focused native cases pass:45 population,24 router-Q8 and162 grouped
cases across15 explicit registrations. Review found a separate legacy counter assertion in
`ROCm_M832256_RoutedExpertBatchInvariantEconomy`: it expected removed route
labels and invented1×64 geometry for route-owned work. Its assertions now check
actual route/tile rosters, hidden preparation and host-submission semantics;
the updated benchmark builds and passes all three row sizes8/32/256.
Hardware-counter qualification has not been collected for these variants.

The complete folded baseline is frozen in the98-file
`grouped-population-folded-snapshot/snapshot.json`, SHA256
`acb5325cc3d84cb9b0b557dc40a07e66f7485afe462b3c64e4ffedc2f8f55f1f`.
This snapshot includes both Release and Integration producers and all focused
test receipts. Later experiments below have their own narrower evidence.

Transposing native activation scales into aligned four-float lane groups removes
four scalar LDS loads and adds one128-bit load per shader. Distinct scalar
writers avoid vector-component read/modify/write races; row mapping and commits
remain unchanged. ISA joins and instruction censuses are in
`grouped-population-scale-vector-isa-proof.json`. GU/down resources stay82/83
VGPRs and3664/1872 LDS bytes, with no private memory. Replay passes;7,595 uniform
timing samples yield23/35 wins, geometric mean1.029051× and full512 1.1559×.
The41-file archive is `grouped-population-scale-vector-snapshot/snapshot.json`,
SHA256 `cfe2bfafdb66c634f3837f3a73e077576268c5027ed48e45a7fd170a557fea68`.

Down now derives its contiguous grouped source row directly from the validated
tile interval. Gate/up retains its indirect token map. Removing the redundant
down row-map publication removes one barrier, four scalar LDS load sites and
one scalar store site; all24 scalar-to-vector register moves disappear. Down
retains83 VGPRs and uses1808 LDS bytes, with private/vector spills zero.
Its final K barrier already retires readers before the next persistent tile.
Replay and7,595 uniform samples pass:24/35 wins, geometric mean1.032144×,
full512 1.1733×. Live15 still loses4.3%. The39-file archive is
`grouped-population-direct-down-snapshot/snapshot.json`, SHA256
`9a1e48a5fc0d981609696355262fe6587963adb002d6e2f3be09ab10f44da78e`.

A subsequent unqualified experiment retains fused gate/up and splits only down
into independently allocated ordered/native consumers. It preserves the same
directory, decision and direct grouped addressing. The compiler reports GU82,
ordered down63 and native down65 VGPRs, all without private memory; the down
staging footprints are640/1808 bytes. Capture proof authenticates all three
physical specializations, actual down-node omissions and exact submission
rosters. Replay and7,595 samples pass, but uniform geometric mean falls to1.017025×
with16/35 wins; live15 loses5.5%. Full512 remains1.1753×. The extra
launch is uneconomical, so this topology is rejected and direct-down restored.
The40-file archive is `grouped-population-hybrid-down-snapshot/snapshot.json`,
SHA256 `b0ff4c2f3f34a6790de4c3fa52a3f0e449585304315f38649edbf4571fb59d7b`.


### Single-row gate/up investigation

Exact selected-region traces retain512-row capacity with15 live tokens:120
experts each own one row. Auto GU took288,321ns versus ordered256,361ns;
down took184,480ns versus183,201ns. These are diagnostic profiler intervals,
not canonical timing. Separate GU passes report positive GPUBusy and wavefronts,
loaded native code bytes and zero scratch. The87-file baseline profile archive
is `grouped-population-live-profile-snapshot/snapshot.json`, SHA256
`50ac99877f56e8b48201710f2225a9b908bfc104570d448f5f58bb7ffa8217d1`.

A GU-only ordered single-row branch now reads its immutable Q8 activation
row directly. It retains source13's two independent dot chains and sixteen
ordered FP32 partition folds. It includes cold one-row tails and excludes
native tails. ISA confirms two128-bit activation loads, eight signed dot4 sites,
no LDS access and no barrier in that branch. GU remains82VGPR/106SGPR with21
register-lane moves and3664 LDS bytes; Down remains unchanged; both have zero
private/vector spills. Added heldouts cover256×13 cold tails, reversed single/pair
counts, mixed single/native experts and native17-row tails through the retained
production graph. Replay and all7,595 samples pass. Uniform wins32/35 sizes,
geometric mean1.043996×, full512 1.1808×; live15 still loses2.3%.
The42-file archive is `grouped-population-single-row-snapshot/snapshot.json`,
SHA256 `f1f2f28ec707b2f4f4d215c564b8ea46ce7cef650887a6f3e14b4dd7ded0ea9a`.

The next compiler-only probe requests a minimum10 waves per execution unit for
GU while keeping Down's bound unchanged. Release compiles both targets with
zero private/vector spills; GU decreases82→72VGPR, with the same21 register-lane
moves. Replay and7,595 samples pass, but only19/35 sizes win; geometric
mean falls to1.025794× and several cells lose3.9%. This probe is rejected.
Its40-file archive is `grouped-population-gu-waves10-snapshot/snapshot.json`,
SHA256 `8306cdc033bfee9e7933a1d6d63d5b900bc568fb727a1ffa26bcf1910c92e78b`.

A subsequent experiment gives GU separate ordered/native consumers and keeps
Down fused. It retains the direct single-row branch and one shared decision/
directory. Compiler allocations are57/67VGPR and544/3664 LDS bytes for the two
GU producers; their register-lane moves are53/0. Down remains83VGPR/1808LDS
with zero lane moves. Both targets compile without private/vector spills;
capture/replay and all22,785 samples across three routing profiles pass.
Uniform wins32/35 sizes (geometric1.048364×, full1.174997×), hotset34/35
(1.151460×, full1.155274×) and power-law28/35 (1.028069×, full1.033593×).
Remaining losses reach1.2% uniform,0.3% hotset and1.9% power-law. These are
retained comparison results, not an all-sizes no-regression claim.
The initial uniform39-file archive is
`grouped-population-hybrid-gu-snapshot/snapshot.json`, SHA256
`496092d5db8d687dfd5ee4fb4073ecb0bb4b9b55ac2bf00bce88f8c4b0ac02a8`.
The matching Integration build passes both architecture spill checks. Its full
231-case focused suite passes15 registrations without skips; the receipt is
`grouped-population-hybrid-gu-focused-all-1794719380208072973/receipt.json`.
Read-only review of the complete production diff found no blocker. Both host
registrations and the legacyM8/M32/M256 counter-owner benchmark also pass
against the current core. Release and Integration resource censuses each cover
1,085 kernels acrossgfx906/gfx1100, with zero private/vector spills.

All42 exact counter passes succeed: seven metrics each for orderedGU/live15,
nativeGU/live512, Down/live15 and512, and each directory/live512. Every pass
authenticates loaded code bytes, native PCI/UUID, captured dispatch identity,
positive GPU activity/wave counts, zero scratch and original power restoration.
The campaign is `grouped-population-hybrid-gu-counter-campaign-1794719981575775408/campaign.json`;
`grouped-population-hybrid-gu-profile-summary.json` records resources and metrics.
Both GU leaves and Down report zero LDS bank conflicts. Logical consumer
VGPR/LDS allocations are57/544,67/3664 and83/1808; GU directory uses40/4356,
Down directory6/4096. OrderedGU has53 audited scalar-to-vector register moves,
with no memory spills. The captured HIP residency ceilings are22/22/18 blocks
for the consumers; these are static resource limits, not achieved occupancy.
Profiler intervals remain diagnostic; unprofiled whole-operation samples own
performance conclusions. The full evidence archive is
`grouped-population-hybrid-gu-green-snapshot/snapshot.json`.

The committed feature installs this measured Qwen population rule. Compact Auto
and remaining small-work regressions remain open.

### Directory experiments and residual-loss attribution

Two directory rewrites passed byte replay and the complete 7,595-sample uniform
tournament but did not improve whole-operation economics. The wave-local scan
with one LDS publication barrier achieved a1.044125 geometric ratio and32/35
wins. A single wave64, with four consecutive experts per lane and no LDS or
barriers, achieved1.033811 and31/35, versus the committed1.048364 and32/35.
Both were rejected; production retains its committed planner. Exact snapshots:

- `grouped-population-wave-scan-snapshot/snapshot.json`, SHA256
  `85fb5cd7f4684977c71ad5a5a4e0605444139f0c2cccf4e4ec76e8092a1b27b7`.
- `grouped-population-single-wave-snapshot/snapshot.json`,54 files, SHA256
  `6ec2b51c33b3c340f4404bd18e1c1d595cebea89466a7f06f90cb8c1ee195f4f`.

The single-wave producer compiled with GU29/52 andDown25/26 VGPR/SGPR, zero LDS,
private bytes and vector spills. Projection allocations remained unchanged.
Exact GPUBusy profiler passes authenticated one64-thread directory dispatch each,
zero scratch and restored power. Diagnostic GU/Down intervals were15.601/7.600µs,
versus9.800/5.000µs in the corresponding committed GPUBusy passes. These isolated
profile intervals support attribution; the unprofiled tournaments own the
rejection. Release replay and both host registrations passed. The archive also
contains a nonqualifying initial replay attempted before the test relink; its
old512-thread expectation correctly rejected the64-thread producer. Only the
subsequent completed-build replay and tournament count as candidate evidence.

The retained focused replay now adds an interior-zero histogram with108 routes
across11 experts. It crosses four-expert lane boundaries, the midpoint and the
last expert, with mixed12/16-row tiles. Real grouping, poisoned banks, producer
omissions, restored repeated replay and empty/full reset exercise the production
parent. This case remains useful after reverting the planner experiment.

A separate deterministic100,000-resample analysis authenticates all three
committed tournament logs. Uniform39/40/41 and power-law40/41/63/64/65/68 each
show Auto slower in31/31 paired rounds and strictly positive block-bootstrap
loss intervals. The largest median loss is1.897%; these are within-run findings,
not independent-run reproducibility claims. Hotset7 is noise-compatible, while
power-law48 is only weak evidence of a0.218% loss. See
`grouped-population-hybrid-gu-statistical-review-1794722494317292469.json`, SHA256
`428f1ff15ff1f67e371c3ebc91b5e94717e589294311e038b4e77ebcddbba9b1`.
Two direct-two-row GU candidates also preserve exact bytes but lose aggregate
performance. Shared Rows1/Rows2 control flow gives1.029597 geometric speedup
and22/35 wins; separate one-row and two-row branches give1.028590 and24/35.
Both improve the original39–41-row losses while introducing larger16/17/31/32
losses. They are rejected; the committed one-row branch remains authoritative.
The snapshots are `grouped-population-direct-pair-snapshot/snapshot.json`
(SHA256 `d37de111d1286b9c5ea826399a2ab68ed9ed558c37e8b4071c2812adb838189b`)
and `grouped-population-direct-pair-split-snapshot/snapshot.json`
(SHA256 `97f38d6e6e08329b7fd8fffe289772fa7f9e6db038f203d3b6e07947309ca804`).
Their GU allocations are63/106 and62/106 VGPR/SGPR, with zero private/vector
spills; exact GPUBusy profiles authenticate each producer. Static ISA comparison
shows the one-row K32 arithmetic/load sequence unchanged and no overlap between
both rows' activation vectors. It does not establish a cause for the regressions.
Cold14/native18 tail traces and the exact40-live-row profile hook remain useful
regressions/attribution fixtures after reverting these candidates.

Reducing only native GU to1,023 workers gives1.040186 geometric speedup and32/35
wins, but full-live speedup drops to1.139034. It is also rejected. This probe
changes launch geometry only; ordered GU/down and all arithmetic stay unchanged.
Its archive is `grouped-population-native-1023-snapshot/snapshot.json`, SHA256
`4fefd2e27be4a0ea0afd3cef40e34b35f653e0f66be89377455fdd512d95702b`.
The final grid-only candidate uses orderedGU1,024/nativeGU2,048/down2,047.
Replay and7,595 paired samples pass, but geometric speedup is1.036885 with32/35
wins. Full-live improves to1.184238 while the original39–41 losses remain, so it
also loses the aggregate comparison and is rejected. Exact GU profile passes
authenticate the new grids and unchanged57/67-VGPR, scratch-free kernels. Its
44-file archive is `grouped-population-balanced-gu-snapshot/snapshot.json`, SHA256
`3ecf6153c34a17e05f2f3d5257058acdeef5c802f378e8aa1485675c13768e70`.
All production code is restored to the committed hybrid-GU implementation before
Compact work; only stronger replay/attribution fixtures and these notes survive.
The restored producer plus expanded fixtures passes all231 native cases across15
focused registrations: `grouped-population-expanded-replay-focused-all-1794725750567670704/receipt.json`.
An initial all-codebook run reached its unchanged180-second timeout without a
numerical failure; the isolated42-case rerun passed in93.31seconds, followed by
the complete green rerun. Neither timeout nor test scope was relaxed. Both
production libraries are byte-identical to the prior green checkpoint. The
24-file input/artifact snapshot is `grouped-population-expanded-replay-snapshot/snapshot.json`,
SHA256 `c2d04cc4d3003fe1d84766afb3049fc9e0b8d41d6d31acd4fc84d63ce7d97f31`.


### Compact Auto implementation boundary

The audited compact domains retainM512/H512/I256/E8/top8 with source/physical
Q4_0=0 orIQ2_S=13. Both roles use ordered16×256/block256 and native16×32/block128;
source-owned partition counts remain16 forGU(K512) and8 forDown(K256). N32 wins
overN64 at every previously measured active tiled compact point, including full
Q4_0/IQ2_S gains1.053×/1.043× overDot4. These are explicit-candidate results;
compact Automatic still selectsDot4.

Both compact roles may share one TM16 directory: counts and offsets remain
read-only, and SwiGLU has no directory argument. Bind the same264-entry capacity
and its live-count word at264 to both roles; the current Qwen-style GU12 bound
would instead use350 and cannot be reused unchanged. Canonical storage already
reserves616 words. Reuse needs a typed capture-time publication binding from GU
submission to Down on the same exact stream, including domain, pointer/layout,
count/offset and selection bindings; pointer presence is not readiness. Empty
and sparse consumers must reject ExpertTiled before reading stale directory bytes.
This scope remains Complete/full-width/TokenRows/AllRoutes only.

Generalize compile-time geometry rather than copying arithmetic. Derive ordered/
native column ratio fromTN/BN: compact uses8, unlikeQwen's4. Compact GU/down
native columns are8/16 and ordered columns1/2. Parameterize source identity,
partition tree, row steps, planner domain and ordered/native thread counts.
Replay must prove shared pointer/capacity and one planner per complete operation,
cold/hot/empty resets, poisoned unused capacity, source0/13 ownership and both
physical producers. Added inactive launches and removed Down planner must be
measured together before installing compact Automatic. Bound each compact
ordered grid by its immutable directory capacity times actual ordered columns:
GU264 andDown528 workers for capacity264. Launching2,047 workers for these coarse
extents would add avoidable dormant work. Native workers need their own measured
bound. A move-only produced-directory token carries the successful GU receipt,
complete immutable domain, exact stream/device, counts/offsets, selection and
fixed-TM16 layout; Down consumes it once within the same lexical operation owner.
This is capture construction metadata, never a host copy of live device state.

All relative evidence paths above are under
`/tmp/llaminar-gfx1100-wmma-prefill-evidence/`. These are local investigation
artifacts, not published corpus payloads or a completed performance claim.
