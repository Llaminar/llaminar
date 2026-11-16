# gfx1100 WMMA for grouped MTP decode

Status: native M16/N64 verifier kernels and exact-hosted Automatic selection
now cover all21 source formats at the three measured projection/SwiGLU shapes.
The actual Automatic sweep passes1,890 cells: all1,264 native-selected cells win
1.00198–2.00465×, geometric mean1.28647×;626 keys retain Dot4. Current Release
replay passes171 native cases and five focused ROCm registrations pass. Prior
seven epilogue and six host checks remain green. All168 per-source profiles pass
with zero memory spills. Router Automatic qualifies85/120 exact-hosted keys
(1.00291–1.28437×, geometric mean1.10438×); seven focused registrations pass.
All80 router profiles pass without spills. A forced FP16/HD256 grouped-attention
candidate passes captured serial-byte replay, real append/RoPE replay and
special values under all four original wave counts. The single-row/four-head
layout improves native latency in83/90 cells over the eight-query candidate
(geometric mean1.09488×), but still loses all90 comparisons with Dot2. Automatic
therefore remains Dot2. Fixed-query32K→64K scaling passes. Both shipped targets
compile without spills; the native producer uses49 VGPRs and27,152 LDS bytes.
All29 affected feature GTests and eight exact producer profiles pass, including
explicit finite/special TPB192 multi-tile replay. The bounded grouped-attention
experiment is closed with Dot2: native loses all2,790 paired samples. Full-curve
fixed-query fits through64K have R²≥0.999892 for Dot2 and≥0.999771 for WMMA.
QKV and GDN Automatic qualify measured complete bundles; routed MTP remains open. See the
[attention proof](GFX1100_MTP_ATTENTION_WMMA.md). Broader bundles and
routed operations remain open. Direct floating WMMA substitutions fail their byte contracts. These component results do
not claim model-level MTP speedups or full production certification.
Shared-verifier Automatic now admits all21 same-source M16 tuples and mixed
IQ2_S/IQ4_XS M9/M16. All28 replay cases pass453 source/capacity/live keys.
All177 native-active Automatic cells win all31 pairs by1.23091–1.88403×;
sparse admitted cells have geometric gain1.03576× and worst latency loss1.16%.
Four focused host and six CUDA/ROCm feature registrations pass. All368
source/role/counter profile passes are complete and independently audited;
the frozen producer retains zero private storage and memory spills.
See the [shared transaction proof](GFX1100_MTP_SHARED_VERIFIER.md).
This implementation plan extends the [prefill campaign](GFX1100_PREFILL_WMMA.md)
and the [MTP project plan](../2026-06/MTP_VLLM_STYLE_PROJECT_PLAN.md).
It records the implementation frontier and remaining qualification, not a
performance certificate for MTP.

## Existing execution and missing coverage

Grouped execution already exists. The missing work is native matrix instruction
producers and measured production selection for grouped decode, with its stricter
serial-row byte contract. Ordinary prefill evidence does not qualify MTP.

| Operation | Current ROCm execution | WMMA work required |
|---|---|---|
| Dense quantized projections | Dedicated M16/N64 native leaves preserve serial arithmetic through standalone, homogeneous and mixed APIs. Automatic uses source/operation/shape-specific masks for all21 source formats at M2–16; Dot4 owns other keys. | Per-source profiling passes; qualify other geometries, multi-output bundles and real grouped transactions. |
| Routed expert gate/up and down | Multi-row MTP enters the fixed-topology grouped transaction in `MoEExpertComputeStage::executeDecodeEquivalentVerifierPrefill`. Native projection candidates and M512 population policies exist; those policies do not admit ordinary verifier capacities. Sparse RouteOwned execution remains a distinct Dot4 producer. | Reuse and qualify native leaves for actual verifier routes, including sparse groups; install measured selection. The prefill-named interface alone does not prove MTP coverage. |
| Floating grouped projections | `ROCmFloatingPointGemmKernel.cpp` selects `rocmFp16_batched_projection`, `rocmFp32_small_n_batched_projection`, or `rocmFp32x16_batched_projection`, preserving the serial fixed reduction tree. | Direct half/BF16 substitution changes native FP32 operands; a real half-input WMMA witness also changes the explicit FP16RNE reduction. Retain the existing exact producers; no ordinary rocBLAS replacement. |
| Q8 router | The real verifier API already reaches the forced-capable byte-exact WMMA leaf. Automatic admits the original M512 key and85 exact-hosted M2–16 keys across four measured K/E shapes and both bases. | Actual Automatic component timing, seven focused registrations and80 source-bound profiles pass. Full routing/top-k remains a separate transaction boundary. |
| Verifier attention | Explicit FP16/HD256 `WmmaLaneExact` preserves lane-owned dot2/XOR, softmax/PV and original split reduction. Captured ring/sentinel/special-value and append/RoPE replay pass. Fixed-query scaling through64K passes; current single-row timing loses all90 cells. Automatic remains ordered. | Qualify a profitable measured domain, retaining exact feature and resource gates; current native layouts remain slower than Dot2. See the [proof](GFX1100_MTP_ATTENTION_WMMA.md). |

The dense native decoder switch currently contains 18 physical cases. Qualification
must enumerate the repository's full source-format inventory, including normalized
aliases and promoted descriptors; the physical switch count is not source coverage.

CUDA provides a useful existing implementation pattern:
`CUDANativeQuantisedGemvShardImpl.cu.inc` selects
`GeneratedGroupedKernel::TensorCoreMma16` through installed grouped tuning, then
launches `launchTensorCoreSmallMGenerated`. Its `verifierMmaM16N8K32` helper uses
signed INT8 `mma.sync.aligned.m16n8k32`. Preserve the ROCm serial arithmetic
contract when adapting operand staging, integer accumulation and epilogues.

## First implementation slice

Start with device-counted dense NativeQuantised standalone and fused projection
APIs in the actual decode-equivalent verifier scope. The existing
`ROCmNativeQuantisedWmma.hip.inc` already accepts serial-M1 partition counts and
preserves each K32 commit and ordered FP32 fold. The missing authority is row
ownership: `ROCmNativeQuantisedWmmaLaunch.h` has static M/N/K and serial partitions,
but no `DeviceRowRange`/`ProjectionWork`. Reuse that arithmetic while adding the
existing device-owned row contract; launching over allocated capacity would
overwrite inactive rows and does not qualify a verifier producer.

Extend `tests/v2/integration/kernels/rocm/Test__ROCmDeviceCountedVerifierRows.cpp`.
It already captures public verifier APIs, varies prefix counts, compares raw
bytes against independently executed production M1 rows, and checks poisoned
inactive outputs. Cover standalone projection, homogeneous and mixed fused
bundles, and fused SwiGLU/down through one retained capacity31 graph. Exercise
full, small, empty and regrown work, every live row count 2–16, and 17/31. Its
current `deviceCounted(M,count)` fixture does not establish nonzero origins or
request-batched ownership; add those separately. Enumerate the source-format
registry and promoted aliases. Replace Dot4-specific grid certificates with
candidate-specific WMMA identity and contribution checks.

The first economy panel can reuse that fixture's Qwen3.8 IQ4_XS geometries:
N17408/K5120, N5120/K17408 and N10240/K5120, with capacities3/16 and live2/3/16.
Compare complete captured operations, including preparation and epilogues,
against the current grouped producer with31 paired samples. Existing seven-sample
timing is exploratory evidence. The model-loading `V2_Perf_DenseVerifierRows_ROCm`
M2/3/4 gate follows component qualification. Register the focused native replay
explicitly in production preflight before treating this slice as complete.

The standalone grouped epilogue prerequisite is implemented. Nonzero beta and
mapped output now use one grouped producer and a count-aware persistent-scratch
combine; the production serial-row loop is removed. Four captured replay cases
pass in Release and Integration, and the affected all-format/device-counted
feature suite passes. Request holes, a nonzero subview origin and bit-preserving
mapped copies have explicit regressions. Full-target compilation reports zero
spills; all 28 exact counter passes completed. See the [repair proof and evidence](GFX1100_MTP_VERIFIER_EPILOGUE.md).
This prerequisite does not establish a WMMA verifier candidate or speedup.

Implementation seams are the standalone, homogeneous-batched and mixed-batched
`rocmLaunch_native_quantised_small_m_*` functions in
`ROCmGemvKernel_native_quantised.hip`, declared through
`ROCmGroupedVerifierLaunch.h`. Use `projectionWorkForLaunch(M, row_range)` and
the existing compile-time work dispatch. Cursor bounds name original physical
rows, including request holes and nonzero origins; they are not compacted row
counts. Preserve the source-policy serial partition selection already resolved
by those adapters, including original unsliced output widths for fused bundles.

Add a typed verifier candidate scope separate from the prefill override, and
reject incompatible forcing before recording graph nodes. The current WMMA
body uses four waves and `RM=BM/32`; an M16 enum alone would create zero row
fragments. An economical M16 candidate needs an explicit two-wave mapping while
sharing decoder and commit helpers. The repaired combine epilogue already
forwards both the verifier prefix and independent request-work authority.

The first producer should use a dedicated wave64 translation unit and M16/N64
tile: two waves own columns 0–31 and 32–63, with two column fragments per wave.
Retain one CTA per canonical serial-M1 K partition and the existing ordered
reducer. Each partition writes the established `[P, physical_tile_M, N]` scratch
layout; P1 publishes directly. This retains K parallelism at shallow M and keeps
the existing TN64 fused-bundle column offsets. Original request intervals drive
bounded row cursors before weight decoding; tails must not read or write inactive
rows. A later N32 candidate needs its own producer geometry, rather than reusing
TN64 bundle offsets. Existing prefill entrypoints and candidate selection remain
separate from this verifier implementation.

## Historical candidate experiments

This section records superseded candidates and their original receipts. The
combined and split adaptive producers were rejected and have been removed from
production. Current implementation and final evidence follow below.

The first candidate added a dedicated wave64 M16/N64 producer, selected
through a device-bound `ScopedNativeQuantisedVerifierCandidate`. Automatic
still used the ordered implementation at this stage. All three adapters
(standalone, homogeneous fused and mixed-dispatch fused) preserve their original
serial-M1 partitions, physical scratch chunks and ordered reducers. A single
mixed-decoder kernel shares the existing decoder and contribution helpers,
source LUT initialization transaction and one 7,040-byte LDS bank.

Original device row cursors guard every activation/scale/sum read and output
publication. Scalar LDS sum slots avoid cross-lane vector-component writes;
the existing supplied whole-block sum plane remains authoritative. Direct
standalone publication and direct fused bias/+0 remain distinct. Empty trailing
partitions publish +0 for active rows. The submission boundary authenticates
actual gfx1100 stream/device identity and publishes a successful WMMA submission
counter containing function identity and launch geometry.

Initial evidence on gfx1100: **63 all-format cases plus two epilogue cases pass
in Release**, and all **five focused Integration registrations pass**, including
the existing device-counted/all-format/epilogue controls. The 63 cases cover 21
source formats through the three launch APIs at physical M31, including retained
small/full/empty/regrown counts and the scratch boundary at 16 rows. Epilogue
cases cover fixed/counted rows and request holes with a nonzero subview origin.
Mixed-dispatch cases currently repeat one prepared matrix; they do not establish
heterogeneous-weight coverage.

Both builds compile the shipping gfx906/gfx1100 set with zero private-memory or
VGPR spills. gfx906 emits no WMMA entry. Six gfx1100 publication/work variants
use 80–87 VGPRs and 7,040 LDS bytes; generic request variants also use 82–84
scalar-to-vector register-lane moves, which are not memory spills. These compiler
results do not establish occupancy, useful throughput or a dispatch winner.
Evidence root remains `/tmp/llaminar-gfx1100-wmma-prefill-evidence`:

- `mtp-verifier-wmma-release-rows-1794737423382100155/receipt.json`
- `mtp-verifier-wmma-release-epilogue-1794737432949997951/receipt.json`
- `mtp-verifier-wmma-focused-1794737511884556696/receipt.json`
- `mtp-verifier-wmma-initial-{release,integration}-allocation/allocation.json`

The additional 42 replay cases now pass in both builds: every physical M1–16,
N65 column tails, fused bias/no-bias and actual empty K-partition planes. The
empty-plane witness reads poisoned scratch back and requires exact +0 in every
active empty plane. Together with the initial runs, 107 native cases pass.
Actual heterogeneous weights and a supplied-sum mutation witness remain open. Paired timing must capture both candidates with
identical operands/workspace/counts and alternate AB/BA across 31 rounds. The
existing raw-launch economy cases exclude quantization; only public projection
and SwiGLU adapters currently measure the complete operation. Add counted
quantization inside raw fused timing graphs before treating those as complete
operation results.

The first paired Release panel completes all 1,860 samples across 30 IQ4_XS
cells (31 AB/BA pairs each, eight complete public operations per retained graph).
WMMA wins all six live16 cells by 1.370–1.606×; Dot4 wins all live2/3 cells.
The all-cell geometric mean is 0.904×, so selecting WMMA unconditionally would
regress this panel. The subsequent 114-cell panel (7,068 samples) resolves every
positive live prefix: all six operation/shape combinations favor Dot4 at live1–4
and WMMA at live5–16. First native wins at five rows are 1.031–1.091×; live16
wins are 1.373–1.607×. This is forced-candidate evidence, not installed Automatic
performance.

A first device-side adaptive candidate preserves byte equality in all 63 new
format/API cases plus two epilogue cases. Its actual 114-cell panel retains all
72 native wins but loses all 42 shallow cells: speedup 0.733–0.970× at live1–4.
That two-wave ordered branch duplicates decoding and retains the native resource
budget. It is not economical enough to install. A one-wave candidate also passes all 168 native cases and five epilogue
cases, including explicit 4→5→4 poisoning and request intervals on opposite
sides of the threshold. Both shipping architecture builds are clean. Its actual
114-cell timing regresses to 51 wins and geometric mean 1.052×; it is rejected
for Automatic economy. At N5120/K17408/live4, separate exact counter passes
measure 17.49% occupancy for that adaptive producer versus 32.12% for original
Dot4. The loaded descriptor allocates 124 VGPRs despite a logical count of 100.
This explains why metadata-only assumptions are insufficient.

The subsequent split prototype recorded complementary ordered and matrix packets before
the existing reducer. Both traverse the same original M16 intervals; <=4 rows
belong only to the ordered packet and >4 only to the matrix packet. Original
device publications remain the sole count authority. Ordered prefix variants
use 44–46 VGPRs and no LDS; native variants use 80–82 VGPRs and 7,040 LDS bytes.
Full-target allocation has zero private storage and no memory spills in both
builds. All 63 adaptive replay cases and three adaptive epilogue cases pass in
Release. An additional 21 omission cases cover all source formats through the
three launch APIs: disabling each actual producer leaves only its assigned
partials poisoned, inactive rows remain unchanged, and two restored dirty
sequences recover serial bytes. The same M16 cursor drives both packets. All five focused Integration
registrations pass (93.80s), including the existing all-format, device-counted
and epilogue controls. The result reader initially expected an obsolete CTest
success sentence; independent validation of all five exact pass records, exit0,
log hash and unchanged producers accepts the retained result without rerunning.

The complete 114-cell paired panel yields 72 wins and geometric mean 1.1224×.
All live5–16 cells win (1.0169–1.5894×); all shallow cells still lose
(0.8619–0.9789×). Automatic remains Dot4. The ordered leaf has similar fetched
bytes and VALU instruction counts to original Dot4 and comparable measured
occupancy (30.85% versus32.12% at N5120/K17408/live4). These counters do not
by themselves explain the remaining latency; native packet overhead and ordered
address/decoder scheduling remain tuning candidates. Exact full-operation
timings remain separate from counter-run durations.

The two-wave adaptive build passes full-target allocation checks, with no
private storage or VGPR memory spills. A generic row-cursor helper initially
spilled four bytes; local call-site inlining removes that spill without changing
the guard. Profiling and final allocation evidence must follow the final candidate.
The panel covers one-output projection and SwiGLU/down, not three-output QKV
or heterogeneous bundles. Histograms authenticate eight copies of each single
operation's kernel/geometry inventory; shared enqueue code preserves bindings.
No profiler or occupancy qualification has yet been attached to this candidate.

Additional evidence:

- Expanded Release replay: `mtp-verifier-wmma-release-rows-1794737965147827976/receipt.json`
- Expanded Integration replay: `mtp-verifier-wmma-integration-rows-1794738055957670883/receipt.json`
- Initial paired timing: `mtp-verifier-wmma-paired-release-1794738300242349419/receipt.json`
- Full-prefix forced timing: `mtp-verifier-wmma-all-counts-release-1794738654421536888/receipt.json`
- Two-wave adaptive replay: `mtp-verifier-wmma-release-rows-1794739740054984811/receipt.json`
- Two-wave adaptive epilogues: `mtp-verifier-wmma-release-epilogue-1794739795033655417/receipt.json`
- Two-wave adaptive timing: `mtp-verifier-adaptive-all-counts-release-1794739808169798713/receipt.json`
- One-wave replay: `mtp-verifier-wmma-release-rows-1794740169834418342/receipt.json`
- One-wave epilogues: `mtp-verifier-wmma-release-epilogue-1794740257794128914/receipt.json`
- One-wave timing: `mtp-verifier-adaptive-all-counts-release-1794740287062271774/receipt.json`
- Split replay: `mtp-verifier-wmma-release-rows-1794740787362979646/receipt.json`
- Split epilogues: `mtp-verifier-wmma-release-epilogue-1794740856158896430/receipt.json`
- Split timing: `mtp-verifier-adaptive-all-counts-release-1794740891943867539/receipt.json`
- Focused Integration: `mtp-verifier-split-focused-1794741459832595525/validated-result.json`
- Split omission proof: `mtp-verifier-wmma-release-rows-1794741292173644368/receipt.json`
- Frozen split candidate: `mtp-verifier-split-snapshot/snapshot.json` (original timing binary SHA reproduced exactly)
- Frozen one-wave prototype: `mtp-verifier-adaptive-one-wave-snapshot/snapshot.json`
- Frozen adaptive prototype: `mtp-verifier-adaptive-two-row-snapshot/snapshot.json`
- Frozen initial candidate: `mtp-verifier-wmma-initial-candidate-snapshot/snapshot.json`

## Exact hosted transaction selection

The combined/two-packet candidates do not meet shallow-row economy. Unsigned
address calculations regress the114-cell geometric mean to1.1118× and are
reverted. Binding homogeneous decoders before capture produces zero-spill
specializations but does not resolve latency (72/114wins, geometric mean1.1212×).
These remain experiments, not Automatic policy. The new independent-data replay
passes42 cases:21 primary source formats paired with separately prepared IQ2_S
and Q4_1 weights, distinct seeds/biases, common N65 and ragged N17/65/129, prefix
and request-subview replay. Common-width adaptive captures authenticate the
RuntimeDecoder symbol; ordinary homogeneous captures authenticate FixedDecoder.

The production ownership audit identifies an existing route to best-candidate
selection without adding a packet, ticket field or captured transaction variant:

- Hosted HIP MTP already selects `MTPVerifierGraphFamily::ExactRows`, retaining
  each physical width2..maximum. Its existing authenticated ticket names depthd,
  and the selected verifier signature has widthd+1.
- Controlled verifier preparation requires the device-owned active verifier
  count to equal current depth+1. Response/maintenance clipping changes accepted
  publication budget, not healthy executed verifier width (`MTPVerifierPolicy.h`).
- `executeMTPMainForward` holds the typed verifier policy, resolved family and
  grouped construction scope. Carry an immutable exact-hosted-single-request
  extent fact from that admitted transaction through existing policy into GEMM
  construction, and authenticate it in graph signature equality/hash and replay.
- Auto can then combine that fact with measured device/format/operation/N/K/width
  keys: original Dot4 for live1–4 and native WMMA for live5–16 in the measured
  IQ4_XS cells. Retain original device guards and canonical serial partitions.
  Mutable counts never enter host state or cache keys.
- ExactRows or physical capacity alone is insufficient authority. Bounded
  envelopes, arbitrary request batches/holes, selected LM-head rows and
  subprojections need their own extent proof and qualification. Followers receive
  the fact only through the authenticated complete-transaction/transport binding.

The typed propagation/cache proof and actual Automatic component replay are now
implemented and pass. The existing authenticated ticket still selects retained
exact-width graphs; no new ticket field or live host count was introduced.
Rejected combined/split producers, overrides and implementation-specific tests
are retired. Their unique row, format and heterogeneous-weight coverage remains
in the pure WMMA replay suite. Full transaction/model qualification remains open.

Additional current evidence under the same ignored evidence root:

- Unsigned experiment: `mtp-verifier-adaptive-all-counts-release-1794741795101731258/receipt.json`, archived in `mtp-verifier-unsigned-snapshot/`.
- Bound decoder replay: `mtp-verifier-wmma-release-rows-1794742021755239972/receipt.json` and `mtp-verifier-wmma-release-epilogue-1794742119284060819/receipt.json`.
- Bound decoder timing: `mtp-verifier-adaptive-all-counts-release-1794742143971408167/receipt.json`, archived in `mtp-verifier-fixed-decoder-snapshot/`.
- Independent heterogeneous replay: `mtp-verifier-wmma-release-rows-1794742455901219920/receipt.json`.

## Implementation work list

- [x] Remove standalone beta/mapped verifier row replay; prove grouped combine
  correctness, original row ownership and inactive-byte preservation.
- [ ] Freeze a source-owned operation/format/geometry inventory and baseline the
  actual captured ROCm verifier path. Include dense QKV, attention output,
  gate/up, SwiGLU/down, shared experts, GDN projections and selected LM-head rows;
  preserve the existing target/bonus-row ownership of terminal logits.
- [x] Implement bounded WMMA row tiles for dense NativeQuantised grouped decode,
  including fused APIs. Reuse immutable weight decoding, scale/minimum helpers
  and fixed partition trees where their arithmetic contracts agree. Keep each
  row's FP32 commits in the same order as production serial M=1.
- [ ] Qualify routed MoE WMMA through the actual grouped verifier transaction,
  including static-owner assignment, expert-local live counts, empty and skewed
  participants, compact/promoted descriptors, and sliced down projections.
  Reuse ongoing grouped-prefill kernel work without treating prefill qualification
  as a verifier certificate or introducing prefill LLEP assignment into decode.
- [x] Evaluate floating grouped projections, router and verifier attention as
  separate candidates. Router has a measured native domain; direct floating
  substitutions retain their exact existing producer because WMMA changes bytes.
  The bounded verifier-attention candidate is exact and linear through64K but
  loses all2,790 pairs, so Automatic remains Dot2. Ordinary rocBLAS stays in scope
  as the qualified floating producer; no broader vendor replacement is pursued.
- [ ] Measure complete captured operations against the current operation-specific
  grouped implementation and serial throughput. Install work-size-aware Dot/WMMA
  selection only from qualified measurements, then verify actual Automatic hits.
  Dot4 remains a first-class selected implementation wherever it wins.
- [x] Add focused native replay and dispatch regressions, explicitly registered
  in `ProductionTestPreflight` and the applicable grouped-verifier feature suite.
  Run the focused and feature suites requested by the user; this work item does
  not claim full-preflight or model-level certification from component results.

## Initial exact-hosted IQ4_XS qualification

`CapturedVerifierExtent` now binds the original participant, count pointer and
unsliced single-request width without reading count contents. Only the admitted
hosted controller's ExactRows family issues this fact; bounded graphs, generic
verifier calls and pipeline followers retain guarded capacity. Forward input
validation rejects contradictory roles, batches, pointers and bucket metadata.
The fact belongs to signature equality/hash, the capture policy and native graph
cache, including moves, reset, retained export and direct replay checks.

Cold capture installs an RAII scope after the original GPU worker handoff.
Diagnostic paths that execute stages enter that same worker and scope before
recording. Ordinary retained native replay does not consult host scheduling
state. Device row guards, source bindings and serial partitions are unchanged.

Five new focused cases cover original geometry, nested/throwing/thread scopes,
signature separation, actual executor/controller capture and diagnostic replay,
and contradictory input rejection. Three explicit preflight registrations own
these device-free regressions. All three and the three affected Unit registrations
pass. Sol6.1 reviews found no remaining blockers in this implementation slice.

The initial Release paired panel measures physical widths2–16 independently for both
complete public projection/SwiGLU operations and all three IQ4_XS geometries:
90 cells,31 AB/BA pairs and8 complete operations per sample. The forced comparison
selects Dot4 at every M2–4 key and WMMA at every M5–16 key. The subsequent actual
Automatic panel contains5,580 samples: all72 M5–16 cells win1.031706–1.617845×,
geometric mean1.260234×. The18 shallow cells execute identical Dot4 kernels and
range0.995374–1.004543×, geometric mean0.999893×. Across all90 cells the geometric
mean is1.203238×. These are complete public projection/SwiGLU operations, including
preparation and epilogues, at N/K17408/5120,5120/17408 and10240/5120.

Automatic requires actual gfx1100, full IQ4_XS provenance and physical decoder,
the original prepared payload and complete single public operation, exact hosted
row authority, and canonical serial partition/launch policy. It excludes biased,
mapped, sliced, multi-output, request-batched and diagnostic-policy work. Both
APIs bind the original operation before any internal policy grouping, preventing
a larger bundle from gaining eligibility through singleton subgroups. Other
work remains on its established Dot4 implementation.

The native worker regression uses the production executor/controller, real GPU
worker, arena contracts and full coherence. Guarded and exact M4/5/16 captures
exercise both public APIs. Captured symbols prove selection; independent serial
M1 output, poisoned partial/empty/regrown replay and unchanged construction counts
prove bytes and graph reuse. That native replay covers21 source formats,
heterogeneous weights, request subviews, physical tails and empty K partitions.
Both optimized builds compile gfx906/gfx1100 with zero private storage or VGPR
memory spills. gfx906 emits no matrix entry; six gfx1100 entries use80–87 logical
VGPRs and7,040 LDS bytes. The reported72 scalar spills in generic request variants
are certified scalar-to-vector register-lane moves, not memory traffic.

Initial IQ4_XS evidence under `/tmp/llaminar-gfx1100-wmma-prefill-evidence`:

- Forced exact widths: `mtp-verifier-wmma-exact-widths-release-1794744248860220133/receipt.json`.
- Actual Automatic timing: `mtp-verifier-automatic-exact-widths-release-1794746462444185812/receipt.json`.
- Release149 native cases: `mtp-verifier-automatic-release-rows-1794746238316604572/receipt.json`.
- Release7 epilogue cases: `mtp-verifier-automatic-release-epilogue-1794746320314555519/receipt.json`.
- Five focused ROCm registrations: `mtp-verifier-final-focused-1794746330414495059/receipt.json`.
- Six host registrations: `mtp-verifier-final-host-tests.log`.
- Final allocation: `mtp-verifier-wmma-final-{release,integration}-allocation/allocation.json`.

All24 selected-kernel profiles pass, one metric per kernel/launch at exact M4/5/16
and N5120/K17408. Receipts authenticate the native GPU, loaded code object,
captured graph/stream and unchanged timing producer. Dot4 M4 uses42 logical/44
allocated VGPRs,64 threads and no LDS; WMMA M5/16 uses80 logical/allocated VGPRs,
128 threads and7,040 LDS bytes. All have zero private storage or VGPR memory
spills. Measured occupancy is30.36%,35.33%,34.25% respectively; memory-unit busy
is73.02%,62.74%,56.95%. WMMA LDS conflicts measure2.31%; Dot4 has no LDS work.
Those counters characterize the selected kernels; unprofiled complete-operation
events above own the latency claim. Profile summary:
`mtp-verifier-automatic-profile-summary.json`, SHA256
`01a772c39b9f17c684a407856f729d58fa9be4ecec9f4f5f589f7ec03c69a91f`.
The initial280-file source/binary/ISA/evidence archive is
`mtp-verifier-automatic-qualified-snapshot/snapshot.json`, SHA256
`09be5854cf2bfa9bb17297d26abb36df35ec75cbebad1778394bd5ea1da392e6`.

## All-codebook Automatic economy

The performance-only `AllFormats/ROCmVerifierWmmaFormatEconomy` panels enumerate
all21 source formats from the canonical verifier registry, retaining normalized
aliases separately. Both complete public projection and SwiGLU operations cover
M2–16 at N/K17408/5120,5120/17408 and10240/5120:1,890 cells,31 alternating AB/BA
pairs and eight complete captured operations per sample,117,180 raw samples.
Timing includes preparation and epilogues. These amortized native-event operation
measurements are neither isolated producer cycles nor model/E2E throughput.

Each key installs WMMA only when its forced ratio of medians exceeds one and
at least22 of31 paired ratios exceed one. This rejects the two median-only IQ1_M
near ties. The rank10/rank22 interval is a per-cell97.055% order-statistic interval
under independent sampling; alternation does not establish independence, and
no simultaneous-confidence claim is made across the sweep. Every actual
Automatic native cell subsequently passed both criteria. The smallest measured
win is about0.20%; the rule does not imply a large gain at every admitted key.

| Source | Native cells/90 | Native speedup range |
|---|---:|---:|
|Q4_0|72|1.0178–1.5540×|
|IQ4_NL|72|1.0169–1.5737×|
|IQ4_XS|72|1.0343–1.6162×|
|Q4_1|72|1.0107–1.6451×|
|Q4_K|72|1.0099–1.6439×|
|Q5_0|72|1.0795–1.7779×|
|Q5_1|72|1.0731–1.7887×|
|Q5_K|72|1.0744–1.7886×|
|Q6_K|72|1.1263–1.8951×|
|Q3_K|48|1.1161–1.4335×|
|Q2_K|48|1.0474–1.3192×|
|IQ3_S|52|1.0034–1.5625×|
|IQ3_XXS|50|1.0129–1.5245×|
|IQ2_S|48|1.0299–1.4695×|
|IQ2_XS|48|1.0429–1.4377×|
|IQ2_XXS|46|1.0096–1.4352×|
|IQ1_S|32|1.0532–1.3593×|
|IQ1_M|28|1.0020–1.2889×|
|Q8_0|72|1.1660–2.0046×|
|Q8_1|72|1.1813–1.9936×|
|Q8_K|72|1.1659–1.9937×|

The native-selected aggregate is1,264 wins/1,264 cells, range1.001978–2.004648×,
geometric mean1.286471×. The626 deliberately ordered controls range
0.993164–1.007960×. Their captured inventory confirms the same Dot4 producer.

Production stores independent projection/SwiGLU width masks per complete source
identity and exact N/K shape. Physical aliases do not substitute for provenance.
Full payload/single-operation/exact-hosted/serial-policy/gfx1100 guards remain
mandatory. Unmeasured geometry/widths, generic guarded capacities, request
batches, expanded descriptors, bias, mapped or sliced output and multi-output
bundles remain outside this admission. The independent test table coversM2–33,
neighboring shapes, all source/superblock/presence/physical identities atM5 andM16,
and the existing ownership and policy exclusions.

`AllFormats/ROCmDeviceCountedVerifierWmma.ExactHostedAutomaticReplay` exercises
actual native capture for every source at long K, both operations, endpoints and
both sides of each mask transition. Kernel inventory proves the selected engine;
serial-M1 bytes and poisoned partial/empty/regrown replay prove correctness.
The existing explicitly registered
`V2_Integration_GroupedVerifierRows_ROCm_WmmaReplay` includes these regressions.
All171 Release native cases pass; all five affected Integration registrations
pass. Release and Integration compile the declared gfx906/gfx1100 set; device
arithmetic and the previously qualified zero-memory-spill objects are unchanged.
All168 all-source M16/N5120/K17408 profiles pass, one metric and exact kernel
per launch. Each actual selected decoder uses128 threads, wave64,80 logical and
allocated VGPRs,67 logical SGPRs,7,040 bytes of LDS (7,168 allocated), and zero
private storage/VGPR memory spills. Measured occupancy ranges32.95–39.15%,
memory-unit busy50.76–69.00% and L2 hit rate12.88–24.05%.
These counters describe the selected producer; unprofiled complete-operation
timings own the speedup claims. Joins authenticate each source/test/metric,
raw counter/trace/code-object/log hashes and the unchanged timing producer.

The sweep reuses one immutable source tensor and prepared handle per source/N/K.
Every width/operation owns fresh workspace, streams, activations, count, serial
oracle and captured graphs. Source/device/geometry mismatches fail. RAII drains
work and retires graphs before clearing bindings and retiring workspace, including
fatal assertion exits. The focused `PreparedSourceReuse` regression tests
projection/SwiGLU grow/shrink reuse and rejects alias/device/geometry mismatches.

Current evidence under `/tmp/llaminar-gfx1100-wmma-prefill-evidence`:

- Forced all21 sweep: `mtp-verifier-codebook-full-width-batch-1794750171902086513/progress.json`.
- Exact qualified selection: `mtp-verifier-measured-policy-work/selection.json`.
- Actual Automatic sweep: `mtp-verifier-codebook-automatic-batch-1794754086313447250/progress.json`.
- Release171 cases: `mtp-verifier-automatic-release-rows-1794753873809409359/receipt.json`.
- Five focused Integration registrations: `mtp-verifier-final-focused-1794753954817390682/receipt.json`.
- Both optimized builds: `mtp-verifier-all-codebook-policy-{release,integration}-build2.log`.
- All168 exact profiles: `mtp-verifier-codebook-profile-batch-1794755510052370606/progress.json`.
- Frozen forced producer/results: `mtp-verifier-all-codebook-forced-snapshot/snapshot.json`;
  `mtp-verifier-codebook-before-reuse-snapshot/snapshot.json` preserves the first
  seven sources' earlier producer. Setup reuse changed only between completed
  source panels, never during a timed cell.

The profile summary is `mtp-verifier-all-codebook-automatic-profile-summary.json`,
SHA256 `37a52709f448189d73ee6b611f7763d365ac1bbe7538f689f6e7d5fc9a9411ec`. The complete qualified archive is
`mtp-verifier-all-codebook-automatic-qualified-snapshot/snapshot.json`,
SHA256 `93c3a3749bde69953f49f581073c4065a8bf8693b52801a5a1d318ce54294a41`, with1619 file records. It preserves the
source patch, binaries, paired samples, raw profiles and focused gate evidence.
Its prior forced and initial IQ4_XS archives remain explicit hash-pinned external
references; retain those alongside this archive when moving the evidence.

## Routed/shared verifier qualification boundary

The next finite implementation slice extends existing owners rather than adding
a new harness. `Perf__GPUExpertPipeline.cpp::runExpertPipeline` already owns
prepared descriptors, routes, internal events and dirty observations; add a
separate verifier roster because its ordinary WMMA roster excludes M<64.
Capture the actual `MoEExpertComputeStage::execute` with
`force_decode_equivalent_verifier_prefill=true`, normally prepared engines and
`active_row_count_device`. Prepare the actual verifier router under
`DecodeEquivalentVerifier` and retain its Q8 producer in the transaction.

Reuse `Perf__GPUExpertPipeline.cpp::runExpertPipeline` for a finite Qwen
H2048/I512/E256/top8 panel at physical M8/9/16, IQ2_S gate/up and IQ4_XS down.
Start with Dot4/N32/N64 EveryExpert and Uniform/Hotset routing; sparse and empty
work comes from device live prefixes or explicit route holes, not an invented
routing-profile label. Retain the existing paired internal events, poison,
publication observations and owned retirement. Include retained
16→8→9→0→16 transitions without recapture.

The capture and active-work proofs must distinguish topology from execution.
M8 has no ExpertTiled/native leaves; M9/16 retain them but sparse live work can
select RouteOwned. Bind actual native/ordered symbols and router-Q8 consumers,
including the RouteOwned gate/up ABI. Native writer omission must change output
only while that family owns active work; it must leave RouteOwned output intact.
Do not require both threshold writers when the independent route histogram
cannot exercise both. Profile the actual selected family and retain live-bearing
sample keys so different retained workloads cannot collide in the timing corpus.

The existing performance fixture's numerical snapshot is eager grouped Dot4,
not a serial-runtime oracle. Before timing, reuse
`groupedExpertDecodeFromRuntime(...RuntimePlacementTable)` from
`Test__ROCmMoEKernel`'s Qwen runtime comparison and require its original per-row
bytes to match the grouped snapshot and candidates. Production optimizations
remain enabled. For the verifier router boundary, use the existing
`routeVerifierRowsDecodeEquivalent` entry with matching preparation; controlled
expert routes remain explicitly injected into separate buffers. This does not
claim that the real router selected the synthetic distribution.

The old performance fixture rewrites and uploads route IDs for each live prefix;
that does not qualify device-counted verifier replay. Controlled Uniform/Hotset
templates need a captured device-counted publisher that preserves live slot
IDs/weights and writes(-1,0) beyond the prefix. No existing single-device API
provides that copy: `publishOwnedRouteRows` requires real multi-participant packet
ownership and cannot honestly be borrowed here. Use a narrow test-owned publisher
or actual router selections, and keep template inputs separate from real router
outputs. The serial runtime oracle must preserve holes and original slot order
without renormalizing the surviving weights.

Shared experts need their actual `SharedExpertFFNStage` grouped path separately.
Extend `Perf__ROCmMoEVerifierPrefill.cpp::runROCmSharedExpertStageCase` with one
persistent INT32 count and retained capacity16. Bind
`active_row_count_device`, `force_grouped_verifier_prefill_for_decode=true` and
`force_decode_equivalent_verifier_prefill=false`. Preserve the two kernel
objects and shared Q8 publication: router ProducerAndConsumer, shared
RequiredConsumer. Prepare the router as DecodeEquivalentVerifier, prepare the
stage before capture, and record `routeVerifierRowsDecodeEquivalent(..., count)`
immediately before `grouped_stage.execute()`. Retain16→8→9→0→16→16, including
reset without rewriting an unpublished count.

Append the real `SharedExpertGateStage` with the same count, a separate gate
vector and an explicitly injected routed residual. This qualifies the complete
router→shared FFN→gate/add operation; it does not claim routed-expert execution.
Compare raw FFN, gated shared output and combined output in native bytes against
independent M1 table-based gate/up/down plus the public serial gate/add. Preserve
zero suffixes. Existing effective-length gate/add tests use active tolerances,
so they do not replace this serial-byte check.

Record Dot4/N32/N64 EveryExpert graphs under the worker-local selector. Retained
native topology at capacity16 is distinct from active RouteOwned work at live8
or0. Authenticate the active family before omission controls. Poison consumed
banks and outputs; prove the private input-Q8 bank stays untouched while the
router publication is borrowed. Use internal retained event nodes for paired
complete transactions and keep every staging allocation alive behind a fatal
drain that outlives the graph owners. Add a dedicated shared replay preflight
registration. A synthetic one-expert route alone does not qualify these handoff
and final-gate semantics. No MoE selection change is installed by this audit.

## Remaining bundle inventory

The actual QKV panel now lives in `Test__ROCmQKVVerifierTransaction.cpp`.
`makeGpuPreparedQKVFixture` retains three independently seeded weights in one
prepared store per candidate. A third independent store/kernel/workspace owner
runs actual M1 `FusedQKVGEMMStage` calls as the oracle. Grouped owners bind
`force_decode_equivalent_verifier_prefill` and the original `verifier_row_range`.
Preparation and input coherence complete before capture; the retained fragment
records the actual stage under its graph dependency ledger. The Release native
Q4_0 local-MoE smoke passes both QKV/KV-only, every physicalM2–16 and live prefix,
dirty/reset replay, quantizer/native omissions and exact captured arguments.
All85 Release cases and all five registered Integration replays now pass,
covering the21-format/four-geometry inventory and mixed-source/bias control.
Initial Q4_0/IQ4_XS timing passes1,200 physical/live cells. Full dense QKV wins
all31 pairs fromM9 and KV-only fromM5; M16 gains are1.389–1.448× and2.083–2.251×,
respectively. The local-MoE crossover differs by source. All-source/four-geometry
full-extent timing now passes84 cases/2,520 cells/78,120 pairs. The proposed
Automatic policy retains1,217 of2,205 distinct source/operation/width keys
with a5% qualification margin across the candidate and two actual-Automatic panels.
Final actual-Automatic timing passes2,520 cells: all1,415 native-selected cells
win at the median (geometric mean1.32030×),1,105 retain Dot4. All five Automatic
Integration registrations/86 cases, ten affected registrations and all64 exact
geometry profiles pass.
See the [QKV component proof](GFX1100_MTP_QKV_VERIFIER.md).

The source audit distinguishes local projection widths from original-output
slicing. `Qwen35Graph.cpp` attaches original K/V slices for QueryInterval
coverage, and that mode explicitly rejects retained MTP graph capacity. Ordinary
local QKV/KV arithmetic uses its local width/serial partition authority; the new
component shapes preserve that behavior. Original-output-slice bytes remain
covered by their separate component fixture. They do not establish MTP economy
or admit QueryInterval for MTP. The stage has no request-hole declaration;
lower-level hole coverage is not actual-stage evidence. Forced candidate evidence remains separate from the final actual-Automatic
panel. Unmeasured complete bundles retain the ordered implementation.

The remaining dense work is complete bundle qualification, not another
all-format leaf implementation. Derive geometry/provenance from
`qwen_release_geometry.py::model_projection_geometries` and `QwenProjectionUse`,
and operation identity from the actual typed stage descriptors. Each record
needs ordered prepared sources/codebooks, original and physical N/K, projection
count, TP slices, row authority, epilogue and complete timed boundary. Shape
deduplication alone cannot distinguish QKV, GDN and selected LM-head work.

| Bundle | Existing owner to extend | Missing evidence |
|---|---|---|
| Attention QKV or K/V-only | Actual unequal-width `FusedQKVGEMMStage` with independent prepared owners | Actual-Automatic confirmation, focused gates and64 geometry profiles pass; QueryInterval slicing excludes retained MTP |
| Attention output | `GEMMStage` verifier path and Wo serial-byte tests | Source-owned output geometry and TP partition |
| Gate/up through SwiGLU/down | `FusedGateUpGEMMStage` and public SwiGLU/down adapters | Connected captured operation with shared quantization and epilogues |
| GDN projections | Actual four-source `GDNProjectionStage` with independent prepared owners | Automatic161 keys win in both630-cell panels;16 profiles, six host gates/32 cases, eight GPU gates/128 cases and CUDA/ROCm capture/12 cases pass |
| Shared expert FFN | Actual shared-stage owner above | Qualified through gate/add and verifier-router Q8 handoff;368 profiles complete; residual sparse overhead remains visible |
| Selected LM-head | `HiddenStateRowsSelectStage` through `LMHeadStage` | Selection-plus-head boundary and original vocabulary/row ownership |

Use the existing registered Qwen dense/GDN and MoE/shared profiles, including
their full and TP-sharded declarations. Retain all21 source formats from the
owning format inventory and separate mixed-format controls. Prove every M2–16;
start paired economics at M4/5/8/16 and resolve any observed crossover before
admission. Reuse the complete-operation event/poison/serial mechanics in
`Test__ROCmDeviceCountedVerifierRows.cpp`; its current timing admission is
singleton projection/SwiGLU only. Merely timing low-level bundle enums that
reuse one matrix would not qualify distinct production sources. Automatic's
multi-output and `ProjectionRowWork` exclusions remain until this proof exists.

See the [complete GDN proof](GFX1100_MTP_GDN_VERIFIER.md) for the four-source
SingleQ8/TwoTermQ8 fixture, independent M1 contract and work-size crossovers.

## Connected dense FFN boundary

The actual ordinary dense builder in `QwenGraphBase.cpp` records
`FusedGateUpGEMMStage` followed by `GEMMStage` with `gate_input`, fused SwiGLU,
alpha1/beta0 and the same original verifier rows. The installed
Qwen3.8-27B GGUF directory supplies hidden5120/FFN17408 and original IQ4_XS
weights: gate/up N17408/K5120 twice, down N5120/K17408. Source4/superblock and
physical24 remain distinct. Ordinary dense FFN uses SingleQ8; QFN HC-up
TwoTerm belongs to a different residual-mixer path.

The next complete component should retain normalized input → gate/up → fused
SwiGLU/down → branch output, using three independent `makeGpuPreparedFFNFixture`
owners and one merged persistent workspace per owner. Compare gate/up and final
bytes to complete independent M1 rows, with dirty prefixes, reset and separate
producer omissions. The qualified `SharedExpertFFNStage` is a different owner
and does not certify this chain. Gate/up needs its own complete original-source
identity; it must not borrow `AttentionKV`. Existing scalar SwiGLU Automatic may
already admit down, so measure the current Automatic chain as well as forced
Dot4/WMMA before claiming an incremental production gain.

## Verifier router economy and attention arithmetic

The complete Q8 router tournament covers every M2–16 at K/E2048/256,
2560/512,3072/256 and8192/512 in both bases. Sixteen waves beats four and eight
at all120 measured keys. Initial forced timing qualifies87 keys using median
speedup plus at least22/31 paired wins. Actual Automatic narrows that to85:
Original M5/K2048 and Original M5/K3072 win their medians by only0.22/0.24%,
but win17/31 and21/31 pairs, respectively. They retain Dot4.

| K/E | Original basis WMMA widths | Hadamard basis WMMA widths |
|---|---|---|
|2048/256|6–16|5–16|
|2560/512|5–16|5–16|
|3072/256|6–16|6–16|
|8192/512|9–16|9–16|

The final actual-Automatic panel covers all120 keys and3,720 paired records,
31 AB/BA pairs of16 complete captured component operations per key. All85
native selections qualify:1.002909–1.284372×, geometric mean1.104376×, minimum
22/31 paired wins. The35 retained-Dot4 controls vary0.997557–1.006385×. The
smallest native margin is0.29%; this is not a claim of large gains at every key.
Each operation includes hidden quantization, logits and original expert-Q8
restoration. Full routing/top-k and model-level MTP are not certified here.

Selection binds the original exact-hosted device/count/width authority and
rejects rebinding at native submission. Compact-owned rows, transformed rows,
guarded capacities and unmeasured geometries retain their measured first-class
policies. The original M512 rule is unchanged. Both optimized builds compile
changed HIP producers for gfx906/gfx1100 with spill guards enabled. Seven focused
policy/native/timing registrations pass on the final85-key selection. Exact
native-boundary profiles and Dot4 controls pass80 separate counter launches,
bound to the same unprofiled timing producer. Native kernels use74/79 logical
VGPRs (76/80 allocated),19/20 SGPRs,1,024 threads and16KiB LDS; Dot4 controls
use55/56 VGPRs,29 SGPRs,128 threads and512B LDS. Both have zero private
storage/VGPR memory spills and zero measured LDS bank conflicts. Native measured
occupancy is1.886–7.922% across these short boundary keys; these are small grids,
not evidence of whole-device saturation. Memory busy is14.875–45.381%,
L2 hit10.299–72.594%, and GPUBusy100%. Complete-operation timing owns the wins.

Evidence root: `/tmp/llaminar-gfx1100-wmma-prefill-evidence`.
- Accepted timing: `mtp-router-automatic-exact-widths-1794759414664455953/receipt.json`.
- Final focused checks: `mtp-router-exact-focused-1794760956133415385/receipt.json`; verbose output
  proves25 individual executions and no skips across the seven registrations.
- Rejected first admission: `mtp-router-initial-automatic-snapshot/snapshot.json`,
  preserving source, binaries and the first actual-Automatic record.
- Measured selection: `mtp-router-verifier-selection-qualified-preview.json`.
- Complete profiles: `mtp-router-exact-profile-batch-1794759835343054515/progress.json`.
- Joined summary: `mtp-router-exact-automatic-profile-summary.json`, SHA256
  `cde1edacd2f2c5c09f21f87c8b7ed9bd57995c19cbb08bddafc452c2b525df82`.
- Qualified code/evidence snapshot: `mtp-router-exact-qualified-snapshot/snapshot.json`,
  SHA256 `3db4d0e4c17afcb7da9a5302157ab81e031c7177d44f866098cf8a8745417907`.
  Its879 file records deduplicate into719 objects; the rejected first snapshot
  remains an explicit hash-pinned external reference. Project docs belong to
  the feature commit rather than this code/evidence archive.

A standalone FP16/HD256 attention arithmetic probe compares each original
lane's two ordered dot2 updates against one sparse native matrix plane. Both
operands zero-fill the six unused pairs before the two real pairs. All2,097,152
sampled partials match bitwise, including broad finite values, subnormals,
extrema, signed zeros, infinities and quiet NaNs. The loaded matrix descriptor
reports46 registers and zero private/shared storage. Evidence:
`mtp-attention-dot2-probe-run.json` and its log/source/binary siblings.
This proves sampled intermediate arithmetic only. A second probe adds signaling
NaNs and authored cancellation/subnormal/zero witnesses, then reconstructs one
scaled score with a bit-reversed carry tree. Its2,097,152 partials still match;
all finite complete-score comparisons match, but16,078 special-value comparisons
out of524,288 total scores differ. The score implementation also reports192
registers and1,084 bytes of private storage. It is unqualified: a shared root
must not replace the original per-lane arithmetic without proving the complete
byte contract, and these resources do not qualify a production candidate.
Evidence: `mtp-attention-score-probe-run.json` and its siblings. These probes did
not establish per-row split traversal, online softmax/V state or complete economy.
The later complete producer and bounded Dot2 disposition are documented in the
[attention proof](GFX1100_MTP_ATTENTION_WMMA.md).
Two separate diagnostics preserve the original source and control kernels.
`mtp-attention-carry-only-probe.hip` feeds the already-compared serial partials
through indexed and named-scalar carry trees, reporting lane0, other lanes and
fused-score differences separately. Both compile and load at13 VGPRs/private0.
`mtp-attention-rolled-score-probe.hip` additionally retains the sparse WMMA tree
while preventing outer-loop unrolling; both carry representations compile and
load at34 VGPRs, zero private storage and zero VGPR spills. ISA has one static
WMMA site and no scratch sites, versus64 WMMA/225 scratch sites in the unchanged
old192-VGPR/private1084 control. This resolves that diagnostic's resource issue
without proving complete attention economy.

Both new runs retain2,097,152 exact matrix/serial partials. The named carry-only
and named rolled-WMMA roots match all8,192 serial lane0 scores, including special
values; they match each other. The indexed carry reproduces the old fused score,
including269 lane0 NaN-bit differences. Every finite comparison across all64
lanes passes. Named root versus the other63 serial lanes still differs at15,964
special-value observations, so lane0 agreement alone does not certify the full
attention contract. The `*-run.json` records own per-variant results: exit1 also
includes the retained old failing control and is not an independent verdict on
each candidate. Source/binary/allocation/ISA artifacts share each probe's prefix.

The downstream question now has a concrete negative result.
`mtp-attention-output-tiled-probe` invokes the same compiled softmax/PV consumer
on the original64 lane scores and the replicated named WMMA root. Its copied
HD256 FP16 arithmetic is bound to the production source SHA: four wave-strided
key sequences, ascending wave merge, ascending split reduction and two64-thread
output tiles of128 dimensions. It covers512 original16-key streams plus three
NaN-only differing-score witnesses at first/middle/last positions, finite half V
including signed zeros, and both one/two splits. All finite outputs match, but
final FP32 words differ at4,644/131,840 and3,084/131,840 respectively. Both m/l
partial banks match; O partials and final O retain differing NaN bits. The earlier
single-output-tile diagnostic produces identical mismatch counts.

This rejects the shared-root score candidate under the complete byte contract;
softmax does not erase its payload differences. It does not prove WMMA hardware
cannot implement exact attention. A subsequent candidate must preserve the
original lane-specific arithmetic through final output; merely fixing lane0 or
relaxing special-value equality cannot qualify it. These are diagnostic copies,
not production capture or economy evidence. No attention dispatch change is
installed.

The subsequent all-lane candidate stores all64 original partials in LDS, then
uses each consuming wave's original XOR32..1 reduction. The bounded common-KV
probe passes524,288 score-word comparisons, including finite edges and NaN
payloads. Feeding these matrix-generated scores to the unchanged downstream
consumer also passes all m/l/partial-O and131,840 final words for each ofS1/S2,
including special scores placed first/middle/last. The same consumer previously
rejected shared-root scores. Score resources are32 VGPRs,16KiB LDS and zero
private storage at1,024 threads. This is arithmetic evidence only: independent
row split/ring metadata, live tails, production capture and complete-operation
economy remain unqualified. Do not infer16KiB for a fused attention kernel;
wave-output merge storage needs another16KiB unless safely aliased after the
last score-consumer barrier.

Evidence: `mtp-attention-lds-score-probe-{build,run}.json` and
`mtp-attention-lds-output-probe-{build,run}.json`, with source/binary/log hashes.
A read-only review confirms independent Q/K production, unchanged downstream
machine source, every lane's matrix-generated witness payload and failure on
any score/m/l/partial-O/final-O difference. A next candidate must preserve each
row's split starts, key sequence, ring address and ascending wave/split merges.

A separate union-tile score probe gives each query/wave its own matrix column
and physical key address. It passes150,976 active score words and preserves
897,600 poisoned inactive words across independent per-row lengths, S1/S2,
ring origins/capacities and clamped diagnostic live-query counts. It uses45
VGPRs,16KiB LDS and zero private storage at1,024 threads. Evidence:
`mtp-attention-union-score-probe-{build,run}.json`. This test does not qualify
unequal-row softmax/PV or performance. Its explicit count is a diagnostic input:
production AttentionDeviceParams has no active-row field and instead publishes
kv_len=1 for padded rows. Production integration must preserve that owner's
sentinel-row semantics rather than importing the diagnostic count contract.

The production integration seam is
`compute_verifier_rows_decode_equivalent` →
`hipFlashAttn_decode_fp16_grouped_verifier_rows` →
`launch_grouped_decode<FP16,SharedKVVerifier>`. Reuse the existing persistent
PARTIAL_OUTPUT/M/L allocation and reducer. m/l indexing is
`(row*heads+head)*max_splits+split`; O indexing is
`((row*heads+head)*256+dimension)*max_splits+split`. Preserve each row's
`selectDecodeRowConfig`, including its original1/2/4-wave stride and ascending
merges. A separate grouped candidate is needed; ordinary-prefill
`FlashAttentionQKEngine` has a different arithmetic tree.

A two-producer candidate can use one pure device predicate over
`(head,split,row/4)` to give complementary ownership to the current partial
producer and a4-query WMMA producer, followed by the unchanged reducer. No
extra live selection state or host readback is needed. This may lose to a single
producer because of dormant grids and another launch; measure complete graphs
before Automatic admission. Any production-selectable engine belongs in immutable
owner/cache identity. Check the existing TPB-only override behavior before using
it as a control: the grouped bridge currently also requires the NUM_SPLITS
presence flag, although DebugEnv parses TPB independently. Any correction needs
its own focused replay/preflight regression and identical effective controls in
both producers and the reducer.

## Floating grouped arithmetic disposition

Direct gfx1100 WMMA replacement is incompatible with the existing native
FP16/BF16/FP32 grouped contracts. Activations remain full FP32 even when weights
are stored in half/BF16; the instruction's FP32 accumulator does not preserve
an activation rounded to its supported half/BF16 input. The finite witness
A=1+2^-11,B=1 produces serial `0x3f801000`, versus half-input WMMA `0x3f800000`.

Explicit FP16RNE is a separate contract with actual half operands. It still
preserves256 independent FP32 chains and their fixed reduction tree. A real
K512 gfx1100 witness has nonzero products at0,1,256: +2^24,+1,-2^24, realized
by exact-half operands4096*4096,1*1,-4096*4096. The original chain/tree yields
`0x3f800000` (+1), while this direct contiguous-K WMMA candidate yields
`0xbf800000` (-1), uniformly across256 replicated matrix outputs. An earlier
assumed matrix result of zero was wrong; the recorded hardware result owns the
disposition. This rejects this direct substitution, not every possible emulation.
No expensive decomposition or ordinary rocBLAS replacement is proposed.

`ROCmSmallNFP32ProjectionKernels.hip` and the existing fixed-order dispatch are
unchanged. `V2_Integration_ROCmHalfProjectionPackedReduction` passes all three
cases: packed-half capture plus all-format native floating projection and
SwiGLU/down serial-row gates. Evidence:
`mtp-floating-direct-wmma-witness-v3-{build,run}.json` and
`mtp-floating-contract-focused-1794759706330185082/receipt.json`.
The witness kernel uses22 VGPRs with zero private/shared storage; it is a
standalone diagnostic, not production-path or performance qualification.

## Required acceptance evidence

1. **Rows and formats.** Use production M=1 as the independent serial oracle;
   sweep every integer M=2..16 and M31 for all supported source formats. Extend
   the contiguous sweep through any larger admitted graph capacity. The current
   single-request controller supports 15 drafts plus the target/anchor input row,
   yielding at most16 logical verifier rows; that is not a universal limit for
   runtime or request-batched row APIs. Exercise mixed
   request lengths, nonzero row origins, zero/partial live work and tile tails.
2. **Arithmetic and lifecycle.** Require native-byte equality against serial
   decode, fixed partition/reduction order, correct fused epilogues and untouched
   inactive rows. Retain captured graphs across dirty replay, full/small/empty
   transitions, reset and regrowth. For integration, verify accepted-state
   publication and continuation under greedy/stochastic and dynamic-depth policy.
3. **Actual production selection.** Enter the common verifier scope and existing
   device-owned row APIs. Authenticate captured symbols, engine counters and
   branch contributions, including deliberate omission controls. Exercise dense
   and MoE participants, TP column slices, replicated outputs and declared PP
   boundaries. No host live-count readback, recapture, allocation or row replay.
4. **Economy.** Time complete retained operations in Release, with canonical
   unprofiled paired samples and unchanged inputs/routes/accepted work. Measure
   live rows as well as capacity, N/K, codebook and route occupancy. Never infer
   a shallow-verifier win from M512 timing or count padded rows as useful TOPS.
   Where a matched RTX3090 comparison applies, report useful TOPS divided by
   each device's relevant theoretical peak; latency remains the dispatch criterion.
5. **Profiles and resources.** Bind each isolated profiler launch to its exact
   timing producer, shape and path. Report ISA, launch geometry, registers,
   occupancy evidence, LDS, memory activity and throughput; require zero memory
   spills/private scratch. Build the complete shipping architecture set, currently
   gfx906 and gfx1100, with the existing spill guards intact.
6. **Total dispatch.** Bind selection to hardware, arithmetic/source identity,
   geometry, ownership, execution phase and capture policy. Handle positive runtime
   M beyond measured points with the established complete policy. Device-owned
   live data may select branches inside a retained kernel; it must not become
   host scheduling state. Authenticate installed Automatic with fresh timing
   and replay evidence instead of relabeling forced-candidate observations.

Current result: native verifier replay, narrow exact-hosted Automatic selection
and profiling pass the focused component gates and are frozen for the feature
checkpoint. Broader operation/geometry and model qualification follow. This
does not close the full prefill/MTP goal or claim all-codebook MTP economy.
