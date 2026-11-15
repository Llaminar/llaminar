# gfx1100 WMMA for grouped MTP decode

Status: the grouped epilogue prerequisite is pushed at `73f7e2adc`. Native
M16/N64 verifier kernels and exact-hosted Automatic selection are implemented.
Final Release replay passes149 native cases and7 epilogue cases; five focused
ROCm and six host registrations pass. Actual Automatic wins all72 measured
IQ4_XS M5–16 cells by1.032–1.618×; M2–4 retains the original Dot4 producer.
All24 final selected-kernel profiles pass. Broader operation/codebook
economy and model-level MTP qualification remain open.
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
| Dense quantized projections | Dedicated M16/N64 native leaves preserve serial arithmetic through standalone, homogeneous and mixed APIs. Automatic selects measured exact-hosted IQ4_XS single projection/SwiGLU keys at M5–16; Dot4 owns other keys. | Extend measured selection to other codebooks, geometries, multi-output bundles and real grouped transactions. |
| Routed expert gate/up and down | Multi-row MTP enters the fixed-topology grouped transaction in `MoEExpertComputeStage::executeDecodeEquivalentVerifierPrefill`. Native projection candidates and M512 population policies exist; those policies do not admit ordinary verifier capacities. Sparse RouteOwned execution remains a distinct Dot4 producer. | Reuse and qualify native leaves for actual verifier routes, including sparse groups; install measured selection. The prefill-named interface alone does not prove MTP coverage. |
| Floating grouped projections | `ROCmFloatingPointGemmKernel.cpp` selects `rocmFp16_batched_projection`, `rocmFp32_small_n_batched_projection`, or `rocmFp32x16_batched_projection`, preserving the serial fixed reduction tree. | Evaluate matrix candidates under each declared arithmetic contract. Half storage does not authorize changing FP32 operands or reduction order. |
| Q8 router | The real verifier API already reaches the forced-capable byte-exact WMMA leaf. Automatic admits only whole-row capacity512/K2560/E512; forced replay covers M1–65 and both bases. | Measure verifier geometry economy, then qualify full-router Automatic selection and reuse of the original Q8 publication. No new arithmetic leaf is initially required. |
| Verifier attention | Typed grouped decode uses `flash_decoding_mi50_kernel` plus its reducer, with no WMMA choice. The M512 prefill helper has a different arithmetic tree. | Implement a decode-specific candidate preserving lane-owned dot2 chains, wave64 reduction, online softmax and split merge; qualify captured append-plus-attention and fixed-query scaling through64K KV. |

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
- [ ] Evaluate floating grouped projections, router and verifier attention as
  separate candidates. Record a precise unsupported result where an instruction
  cannot satisfy the arithmetic contract; never relax byte equality to admit it.
- [ ] Measure complete captured operations against the current operation-specific
  grouped implementation and serial throughput. Install work-size-aware Dot/WMMA
  selection only from qualified measurements, then verify actual Automatic hits.
  Dot4 remains a first-class selected implementation wherever it wins.
- [x] Add focused native replay and dispatch regressions, explicitly registered
  in `ProductionTestPreflight` and the applicable grouped-verifier feature suite.
  Run the focused and feature suites requested by the user; this work item does
  not claim full-preflight or model-level certification from component results.

## Exact hosted width implementation

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

The Release paired panel now measures physical widths2–16 independently for both
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
prove bytes and graph reuse. Final native replay covers21 source formats,
heterogeneous weights, request subviews, physical tails and empty K partitions.
Both optimized builds compile gfx906/gfx1100 with zero private storage or VGPR
memory spills. gfx906 emits no matrix entry; six gfx1100 entries use80–87 logical
VGPRs and7,040 LDS bytes. The reported72 scalar spills in generic request variants
are certified scalar-to-vector register-lane moves, not memory traffic.

Final evidence under `/tmp/llaminar-gfx1100-wmma-prefill-evidence`:

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
The final280-file source/binary/ISA/evidence archive is
`mtp-verifier-automatic-qualified-snapshot/snapshot.json`, SHA256
`09be5854cf2bfa9bb17297d26abb36df35ec75cbebad1778394bd5ea1da392e6`.

## Broader codebook economy

The performance-only `AllFormats/ROCmVerifierWmmaFormatEconomy` panels derive all21
source formats from the canonical verifier registry. They retain source aliases
separately and reuse the existing complete public projection/SwiGLU byte oracle,
poisoned replay, native inventory and31 paired samples of eight operations.
`DISABLED_ExactWidthEndpoints` brackets M4/5/8/16; `DISABLED_ExactWidthSweep`
measures every M2–16 at all three established N/K shapes. Their corpora remain
separate because endpoint keys overlap the full sweep. These tests do not expand
Automatic admission or routine preflight budgets.

The first Q4_0 endpoint panel passes all24 cells/1,488 samples. Dot4 wins all six
M4 cells; WMMA wins all18 M5/8/16 cells, with complete-operation gains from1.019×
to1.548×. This brackets the crossover but does not yet qualify all widths or an
Automatic policy. Receipt: `mtp-verifier-codebook-endpoints-Q4_0-1794749957198871476/receipt.json`
under the existing local evidence root. Full-width collection follows before
extending production selection.

The full sweep retains one immutable source tensor and production-prepared
handle per codebook/N/K. Every width and public operation still gets fresh
workspace, streams, activations, row publication, serial oracle and captured
graphs. Typed source/device/geometry checks reject borrowing another matrix.
Case-scope cleanup drains work, retires graphs and clears kernel bindings before
workspace retirement, including fatal assertion exits. The focused
`ROCmVerifierWmmaDispatch.PreparedSourceReuse` regression passes in Release and
Integration across grow/shrink and projection/SwiGLU reuse, and rejects source4
alias, device and N/K mismatches. It runs in the existing explicitly registered
`V2_Integration_GroupedVerifierRows_ROCm_WmmaReplay` feature gate.

Before this setup change, completed Q4_0 through Q5_1 records and their exact
producer are preserved in `mtp-verifier-codebook-before-reuse-snapshot/`.
The collection resumes only between completed format records; it does not
replace or interrupt a timed cell. Reuse regression receipts are
`mtp-verifier-automatic-release-rows-1794751923977198886/receipt.json` and
`mtp-verifier-automatic-integration-rows-1794751927971381609/receipt.json`.
All150 Release native cases pass in
`mtp-verifier-automatic-release-rows-1794752022508852148/receipt.json`; all five
affected focused registrations pass in69.63s in
`mtp-verifier-final-focused-1794752070785011663/receipt.json`.

## Verifier router economy and attention arithmetic

The router performance inventory now includes every physical M2–16, with
separate complete-operation four/eight/sixteen-wave tournaments. The first
endpoint panels cover M2/5/8/16, K/E2048/256,2560/512,3072/256,8192/512 and
both bases. Each of96 candidate cells retains31 paired internal-event samples,
poisoned output/Q8-bank observations and exact native geometry checks.

| Native tile | Wins/32 endpoint cells | All-cell geometric mean | M16 range |
|---|---:|---:|---:|
| Four waves |4|0.838×|0.818–1.032×|
| Eight waves |12|0.962×|1.014–1.188×|
| Sixteen waves |20|1.035×|1.130–1.283×|

All M2 cells favor Dot4. Sixteen waves wins six of eight M5 and M8 cells,
and all M16 cells. This component includes hidden quantization, logits and
original expert-Q8 restoration; it is not full routing/top-k certification.
Resolve every intermediate width and original row ownership before installing
an exact-hosted router rule. Automatic remains the existing M512 policy.
Receipts: `mtp-router-4wave-endpoints-1794751808042792414/receipt.json`,
`mtp-router-8wave-endpoints-1794751845878501716/receipt.json` and
`mtp-router-16wave-endpoints-1794751866239827258/receipt.json`; the collector records each
exact source, binary, core, sample and restored power mode.

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
Evidence: `mtp-attention-score-probe-run.json` and its siblings. Per-row split
traversal, online softmax/V state and complete attention economy remain open.
No attention dispatch change is installed.

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
checkpoint. Broader operation/codebook and model qualification follow. This
does not close the full prefill/MTP goal or claim all-codebook MTP economy.
