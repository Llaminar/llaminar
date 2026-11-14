# ROCm prefill matrix-instruction coverage

Dashboard byte limit: **15,000 UTF-8 bytes**. This file records current coverage
and qualification; backend procedures remain in the ROCm tuning skill.

The active objective covers four areas: broader dense dispatch, grouped routed
experts, attention, and Q8 router logits. Each requires production-path component
correctness, captured replay, unprofiled Release timing and isolated profiling.
Attention additionally requires linear complete-operation scaling through 64K
KV tokens with the query bucket held fixed, as explicitly confirmed by the user.
Floating half-operand GEMM is an additional confirmed gap in the sampled vendor
path: its actual loaded gfx1100 code object uses vector mixed-FP16 FMA, not WMMA.
The earlier two-anchor NativeQuantised objective is documented separately in
[GFX1100_NATIVE_QUANTISED_WMMA.md](GFX1100_NATIVE_QUANTISED_WMMA.md).

## Coverage found in the source audit

| Operation | CUDA implementation | ROCm implementation and gap |
|---|---|---|
| Dense quantized prefill | Native prepared-weight adapters select INT8 matrix tiles and preserve source scale arithmetic. | `ROCmNativeQuantisedWmma.hip` already implements all physical decoders on actual gfx1100. The generated production WMMA entries cover only N8192/K2048 at M512 and M1024. Broaden measured exact dispatch for QKV, attention outputs, FFN, shared experts and declared TP participants. |
| Routed quantized MoE | `CUDAMoEGroupedPrefillKernels.cu` uses `mma.sync.m16n8k32` for grouped projections. | `ROCmMoEGroupedWmma.hip` implements source-exact M16N32/N64 leaves under existing ExpertTiled device admission. Automatic remains ordered Dot4 while measured selection is qualified; RouteOwned retains its admitted sparse/empty domain. |
| Attention prefill | `CUDAFlashAttentionKernels.cu:1109` uses WMMA half QK with FP32 score accumulation; its general PV is scalar FP32. The active bounded HD256 specialization additionally uses tensor-core PV at `CUDATensorPrefillKernels.cu:328`. | `ROCmFlashAttentionKernels.hip` now has an explicit native four-plane QK candidate preserving the existing half-dot chains and FP32 PV. Automatic remains Dot2 pending admission that avoids the measured short-context loss. |
| Q8 router logits | `CUDARouterQ8PrefillDevice.cuh` uses exact INT8 MMA with the router's fixed FP32 tree. | Four/eight/sixteen-wave candidates preserve the ROCm adjacent-leaf tree. Automatic installs Sixteen for whole-row M512/K2560/E512 in both bases; compact-owned rows remain Dot4. |
| Explicit half-operand floating projection | `CuBLASGemmKernel.cu:380` submits FP16 A/B and FP32 C through the tensor-operation library interface. | `HipBLASGemmKernel.cpp:291` submits half A/B and FP32 C through `hipblasGemmEx`, but the sampled M512/N8192/K2048 gfx1100 dispatch selects a vector FMA implementation. Library API admission does not establish matrix instruction use. |

Native FP16/BF16 storage alone does not imply the half-operand arithmetic policy.
Other floating projections keep their declared FP32 activation/reduction contract
on both backends. Obsolete blockwise adapter families are not evidence of current
production dispatch: GPU prepared quantized weights use NativeQuantised adapters.

## Qualification status

| Area | Implementation | Functional/component evidence | Performance evidence |
|---|---|---|---|
| Broader dense dispatch | Bound installed-Auto/WMMA collector and selective publisher | Core-table ownership replay, eight binding and 16 decision regressions pass | 84 cells complete; 68 candidate keys gain1.25–2.04×; profile/install pending |
| Grouped MoE | Wave64 M16N32/N64 plus explicit whole-expert row floors4/8/12/16; all physical decoders | Serial, promoted, all-format mixed-writer, threshold/reset and split/sliced-down replay pass | Clean full-live N32 gains 1.04–1.30× across five anchors. Hybrid power-law and sparse replay regress; Automatic remains Dot4 |
| Attention | Four-plane QK plus one-kernel live-prefix adaptive candidate | All-format, threshold/ring/partial/reset and internal timing-owner replays pass | Adaptive fixed-M51264K gains1.158×/1.138×; doubling1.989×/1.987×. Short dense loses0.2–0.4%; Automatic remains Dot2 |
| Q8 router | Automatic Sixteen-wave tile for exact whole-row M512/K2560/E512; explicit 4/8/16-wave candidates retained | 14 GPU feature cases, five pure policy cases and both timing owners pass; actual Auto template, ownership and submission counters authenticated | Actual Auto wins all 12 measured prefix/basis cells: full512 gains 2.744×/2.551×; fresh profile has zero spills |
| Floating half GEMM | Existing production hipBLAS path; sampled native matrix gap confirmed | Three retained graph replays pass, each containing cast and vendor GEMM | Loaded bytes/ISA authenticated; complete Release cast+GEMM timing pending |

[Router qualification](GFX1100_ROUTER_LIVE_PREFILL.md) records admission, retained-prefix timing and profiles.

Router timing covers quantization, logits and original Q8 restoration, with
31 paired samples of16 complete captured operations. Profiling is separate.

Grouped functional owners are `V2_Integration_ROCmGroupedWmmaReplay` and
`V2_Integration_ROCmGroupedWmmaSplitReplay`, both explicitly registered in
production preflight. The first exercises serial-row bytes, native gate/up and
down roles, sparse/empty handoff, source4/physical24 mixed tables and K/N tails.
The physical23 fixture uses real CPU packing/promotion, an independent compact
Q5_K Dot4 reference, both public output banks and changed-input retained replay.
Disabling each native projection independently makes that reference fail, proving
the recorded kernels contribute. Split replay covers both tiles, all 21 source
formats, whole-expert phases and compact column-owned down slices.

The live-K producer avoids unused second-stage decoding/stores. All 91 focused
GPU cases and the device-free admission owner pass. The final clean Release
run spans five cases ×17 live counts ×11 policies ×31 samples of eight complete
operations, with exact profile/timing producer hashes. At 512 live rows, pure
N32 gains 1.295×/1.162×/1.043× for Qwen uniform/hotset/power-law, and
1.068×/1.076× for compact Q4_0/IQ2_S. Hybrid floors add launch/work overhead:
power-law N32/floor8 loses at 0.911×, and sparse uniform replay also regresses.
Automatic stays Dot4. See [admission investigation](GFX1100_GROUPED_WMMA_ADMISSION.md)
for ownership, omitted-writer controls, complete timings and remaining work.

Rebuilt native Q4_0 gate/up profiles each isolate one dispatch across seven
counter passes. N32/N64 use 53/54 logical VGPRs (56 allocated), 54 logical SGPRs,
3392/5568-byte static LDS and wave64; both have zero scratch/spills and two INT8
WMMA instruction sites. Positive activity is authenticated against loaded code
objects, and original AUTO power is restored. Current evidence is in
`grouped-hybrid-{n32,n64}-profile/`, bound to
`grouped-hybrid-producer-receipt.json`. Clean timing is in `grouped-hybrid-receipt.json`; profiles do not supply timing labels.

Attention's adaptive candidate uses one retained query kernel and a device-uniform
choice: WMMA for complete linear M512 prefixes with KV1024–65536, Dot2 otherwise.
Its explicit immutable domain is FP16/B1/HD256/capacity65536 with Q/KV12/2 or8/1.
Automatic stays Dot2. [Qualification](GFX1100_ATTENTION_ADAPTIVE_QK.md) records
admission, arithmetic, timing ownership, resources and remaining work.

Final Q/K timings use internal singleton timeline events around complete retained
producer/query batches:16 operations below2048 keys, one thereafter. Every
sample poisons and independently observes parameter/output bytes outside timing.
Twenty threshold/reset transitions plus64K/reset certify the event owner; stale
samples are rejected. No compiler or competing GPU work overlaps collection.

| Heads | Dot2 at64K | Adaptive | Ratio |32K→64K growth| Affine R² |
|---|---:|---:|---:|---:|---:|
| Dense Q12/KV2 |46.216ms|39.908ms|1.158×|1.989×|0.999931|
| MoE Q8/KV1 |34.855ms|30.641ms|1.138×|1.987×|0.999967|

The registered fixed-query gate requires R²≥0.995 and last doubling≤2.2 across
all seven ascending powers from1K to64K. Threshold1023/1025,8193,reset512 and
regrowth1024 remain separate. Short dense ratios0.9960–0.9978 still show a small
regression; MoE short cells are near parity. No universal or installed Auto win
is claimed. Original launch-inclusive evidence is archived separately.

Both guarded HIP targets compile. Exact512/65536 profiles use96VGPR/50SGPR,
16KiB dynamic LDS,256threads,wave64 and zero scratch/spills. Seven isolated
counter passes have positive activity; both restore original AUTO power and
match final timing files. Sixteen static WMMA sites do not claim that short
replays take the matrix branch. `attention-adaptive-scaling-receipt.json` binds
26 lifecycle cells/806 paired samples; `attention-adaptive-{512,65536}-profile/`
holds loaded ISA/counters. Functional and timing regressions are explicit
production-preflight entries; only focused/feature gates run for this task.

Router Automatic now resolves an immutable engine/tile from original capacity,
K/E, basis, ownership and cached hardware. Only whole-row M512/K2560/E512 selects
Sixteen on gfx1100 with 1024 threads/block and 16KiB LDS. Native submission rejects
wrong device, geometry, basis, ownership or ordered-engine bindings. Diagnostic
TLS cannot override Auto. Compact local 512 can come from a larger matrix and
remains Dot4. The default bucket ladder cannot prove a minimum live prefix.

Five policy cases and 14 GPU router feature cases pass. Registered Auto capture
and timing replays pass 23.70/3.82s; the existing timing owner also passes.
Actual captured symbols, 1024-thread/16KiB geometry, independent Dot4 bytes and
original Q8 publication survive full/empty/one/half/full replay. Neighbor shapes
and owned rows remain ordered. Exact engine-specific submission counters record
capture, stay unchanged during GPU replay, and omit empty static owned projections.
Both shipping HIP targets pass Release/Integration spill guards; Sol reviews clear.

Fresh actual-Auto Release timing has 12 cells of 31 paired samples,16 complete
operations each, with clean compiler-overlap and unchanged-producer receipts.
Full 512 improves 309.105→112.646µs (Original,2.744×) and 318.130→124.723µs
(Hadamard32,2.551×). All live 0/1/16/256/511/512 cells win; live1 paired 95%
intervals are 1.0061–1.0139 and 1.0078–1.0125. Other candidate widths retain live1
losses and are not installed. The actual-Auto profile matches the timing producer:
wave64, 74 logical/76 allocated VGPRs, 19 SGPRs, 16KiB LDS, 2 INT8 WMMA sites,
zero scratch/spills/LDS conflicts. GPU busy 100%, memory busy 71.47%, L2 hit 95.83%;
AUTO power restored. [Qualification](GFX1100_ROUTER_LIVE_PREFILL.md) separates
current Auto evidence from the prior 144-cell candidate tournament.

Dense v6 plans now bind original include bytes to the compiled core table and
Release producer. A loader replay reproduces and guards the repaired table
interposition defect. Complete typed install/retain decisions preserve every
unchanged row and generic absence; only per-alias profitable WMMA changes need
new profiles. All-retain produces byte-identical policy and a no-change receipt.
The broader slice completed84cells (two shapes, M512/M1024, all21formats).
All68 runtime keys have profitable WMMA candidates, gaining1.25–2.04×. Six
focused registered gates pass;168 winner/baseline profiles and installed-Auto
qualification remain. [Evidence and protocol](GFX1100_DENSE_PREFILL_CANDIDATE_SCOPE.md).

The floating audit authenticates the runtime-dumped vendor code object against
the installed rocBLAS image, including all 17 traced symbols/entry offsets:
SHA256 `5dd5b5d0e473562279a87437587c5eb5adf32ab665796b242aaa8538ce11461b`.
Its selected kernel is wave32 with 47 logical VGPRs, 38 SGPRs, 1536 bytes LDS,
zero spills/private storage and no WMMA/MFMA instructions. The actual kernel
dispatches identify gfx1100; the profiler's graph-level agent field incorrectly
labels gfx906. Join graph correlations to kernel dispatches for device identity.
See local `floating-half-authenticated-profile/derived/` for the dumped image,
graph joins, ISA and resource evidence. This Integration trace is a correctness
and ISA observation, not a Release timing qualification.

The local qualification
scope remains the user-authorized focused tests and feature suite.

## Remaining implementation constraints

Dense collection uses the common canonical inventory: 95 modern physical N/K
shapes, 22 runtime M buckets and 21 source formats (43,890 cells). Preserve source
aliases during measurement and pool them only in the exact emitter. Corpus plans
bind the producer and candidate inventory; a changed producer needs a new plan.
Additive installation retains unrelated base rows through an authenticated receipt.

Grouped MoE preserves directory16, source-exact contribution helpers and two
staged K blocks. Its original serial partition policy applies to down slices;
route-slot/token indexing and fixed ordered publication remain unchanged.
Physical IQ4_NL (4) and source IQ4_XS (24) have separate native decoder launches.
The dedicated translation unit initializes its IQ constants before capture.

Attention retains canonical 256-key partitions and the existing FP32
summary/reducer layout. The same native numerical entry applies to both
query/context schedules and all request batch sizes in its admitted domain.
CUDA's batch-one-only restriction would break existing ROCm byte invariants.
CPU numerical thresholds and same-backend byte comparisons must remain intact.
The QK probe gathers four planes with dimension `8*(k/2)+2*plane+(k%2)`;
four consecutive K16 WMMA updates per plane reproduce the existing fdot2
chains and their final ordered FP32 fold. Keep FP32 probabilities and ascending
conditional PV updates initially: FP16 PV would introduce a separate numerical
change. The fixed-query results above include complete captured attention work,
partial/reset checks and the canonical context-reduction oracle.

Local raw evidence for this extension is under
`/tmp/llaminar-gfx1100-wmma-prefill-evidence`; the earlier goal's evidence and
frozen measurement binaries remain separate. No full production-preflight or
real-model E2E qualification is claimed by these component measurements.
