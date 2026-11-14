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
| Attention prefill | `CUDAFlashAttentionKernels.cu:1109` uses WMMA half QK with FP32 score accumulation; its general PV is scalar FP32. The active bounded HD256 specialization additionally uses tensor-core PV at `CUDATensorPrefillKernels.cu:328`. | `ROCmFlashAttentionKernels.hip` now has an explicit native four-plane QK candidate preserving the existing half-dot chains and FP32 PV. Automatic remains Dot2 pending full-path timing and isolated profiling. |
| Q8 router logits | `CUDARouterQ8PrefillDevice.cuh` uses exact INT8 MMA with the router's fixed FP32 tree. | Initial `ROCmRouterQ8Prefill.hip` candidate is implemented and under qualification. The ROCm tree combines adjacent leaves first; copying CUDA's leaf permutation would change bytes. Automatic dispatch remains ordered dot products until measured admission is installed. |
| Explicit half-operand floating projection | `CuBLASGemmKernel.cu:380` submits FP16 A/B and FP32 C through the tensor-operation library interface. | `HipBLASGemmKernel.cpp:291` submits half A/B and FP32 C through `hipblasGemmEx`, but the sampled M512/N8192/K2048 gfx1100 dispatch selects a vector FMA implementation. Library API admission does not establish matrix instruction use. |

Native FP16/BF16 storage alone does not imply the half-operand arithmetic policy.
Other floating projections keep their declared FP32 activation/reduction contract
on both backends. Obsolete blockwise adapter families are not evidence of current
production dispatch: GPU prepared quantized weights use NativeQuantised adapters.

## Qualification status

| Area | Implementation | Functional/component evidence | Performance evidence |
|---|---|---|---|
| Broader dense dispatch | Existing native kernel; inventory audited | Prior two-anchor all-codebook proof remains valid for its original producer | Broader collection and generated installation pending |
| Grouped MoE | Wave64 M16N32/N64 plus explicit whole-expert row floors4/8/12/16; all physical decoders | Serial, promoted, all-format mixed-writer, threshold/reset and split/sliced-down replay pass | Clean full-live N32 gains 1.04–1.30× across five anchors. Hybrid power-law and sparse replay regress; Automatic remains Dot4 |
| Attention | Native four-plane QK candidate for HD256/tile16×16, all native KV formats and query/context schedules | Four new retained tests pass: all formats, B2/GQA/ring, 64K/partial/reset, half extrema and forced rejection. Existing lifecycle/reference gates pass | Fixed-M512 1K–64K scaling gate passes; 64K gains 1.15× dense / 1.13× MoE. Native direct/context profiles have zero spills; Automatic remains Dot2 |
| Q8 router | Typed forced recording; normal and compact owned-row bridges; cooperative K subtrees, fixed-width variants and local scale shuffles | Ten focused ROCm tests pass, including normal/owned capture, finite-scale extremes, raw compact logits and poisoned tails | Clean retained-prefix48-cell run confirms full-live gains1.37–2.52× at three widths, but live1 regresses on most bases. Custom512-only buckets admit those tails; Automatic remains Dot4 |
| Floating half GEMM | Existing production hipBLAS path; sampled native matrix gap confirmed | Three retained graph replays pass, each containing cast and vendor GEMM | Loaded bytes/ISA authenticated; complete Release cast+GEMM timing pending |

[Router live-prefix qualification](GFX1100_ROUTER_LIVE_PREFILL.md) records the small-tail admission gap.

Router timing includes input quantization, logits and, for Hadamard input,
restoration of the original expert Q8 bank. The component harness alternates
31 paired native-event samples over 16 serial captured operations, after ten
warmups. Profiling uses a separate explicitly selected test and one ROCTx region;
it cannot run accidentally as part of the ordinary MoE speedometer.

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

Attention's final fixed-query Release run holds M512, HD256, FP16 and capacity
65536 constant, with dense Q12/KV2 and MoE Q8/KV1 heads. Each point uses 31 paired,
alternating native-event samples over complete count-producer/query graphs.
Canonical context graphs independently check summaries/reduction bytes before
and after timing. All boundary/reset/regrowth checks pass, with no compiler
work overlapping collection. Current complete-query medians at 64K are:

| Heads | Dot2 | Native WMMA | Ratio | WMMA 32K→64K growth | Affine R² |
|---|---:|---:|---:|---:|---:|
| Dense Q12/KV2 | 44.855 ms | 38.969 ms | 1.151× | 1.947× | 0.999754 |
| MoE Q8/KV1 | 33.801 ms | 29.914 ms | 1.130× | 1.960× | 0.999854 |

`V2_Perf_ROCmFlashAttentionWmmaPrefixScaling` gates affine R²≥0.995 and long-end
doubling≤2.2, allowing 10% timing variation above ideal doubling. It preserves
all seven ascending powers of two; 8193, reset512 and regrowth1024 stay separate.
The short MoE512 reset remains slower (0.966×), so blanket automatic selection
is not qualified. Both isolated native profiles authenticate 16 WMMA instruction
sites, wave64, 92 logical/allocated VGPRs and zero scratch/spills. Query/context
use 50/102 logical SGPRs and 16384/17984 dynamic LDS bytes respectively. Each of
seven counter passes selects exactly one native dispatch, with positive activity;
original AUTO power is restored. Evidence is bound by
`attention-wmma-scaling-receipt.json`, with curve/gate summary in
`attention-wmma-scaling-summary.json` and loaded ISA/counters in
`attention-qk-{default,context}-profile/`. Occupancy queries report limits, not
achieved occupancy; profile durations are not timing labels.

Router functional owners are `RouterHadamardQ8.ROCmWmmaCapturedReplay`,
`MoERouterOwnedRows.ROCmWmmaCapturedReplay` and
`MoERouterHiddenQ8ScaleRange.ROCmWmmaCapturedReplay`, explicitly registered together as
`V2_Integration_ROCmRouterQ8WmmaReplay` in production preflight. They authenticate
the selected native symbol, zero private storage, normal/rotated bases, partial
tiles, K-owner boundaries, live empty/shrink/grow transitions, compact logits,
top-k and publication of all original expert-input rows. The current candidate's
isolated native profile reports 67 logical VGPRs (68 allocated), 19 logical
SGPRs, 4096 bytes LDS, zero scratch/spills, 2048 waves, zero LDS bank conflicts
and 96.20% L2 hit rate for M512/K2048/E256. The exact wave64 kernel uses 256
threads and a 16×32 block grid. Profiler launches run inside `mpirun -np 1` so the
MPI singleton helper is not instrumented; wrapping the singleton directly hung
in profiler child finalization. The original AUTO power mode was restored.
The exact linked-core identity matches the unprofiled timing receipt. The source
16-lane scale shuffle still lowers to cross-half-wave instructions in current
ISA; no shuffle-elimination claim is made. Measured automatic admission and
production engine-specific PerfStats remain open.

At M512, current complete-component median ratios for Original / Hadamard bases
are 1.46× / 1.38× (K2048/E256), 2.52× / 2.31× (K2560/E512),
1.85× / 1.68× (K3072/E256), and 0.90× / 0.92× (K8192/E512). The width8192
regression prevents blanket WMMA selection. These are component improvements,
not end-to-end model throughput or the earlier CUDA percentage-of-peak objective.
All eight benchmark cases and ten focused router tests pass. Runtime parameter
registration now borrows build-embedded canonical geometry; the device-free
bootstrap passes with zero source-manifest opens under syscall tracing. Both
normal and compact reference submissions explicitly pin ordered Dot4 so later
automatic WMMA installation cannot turn the numerical oracle into a self-check.

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
change. The independent million-output probe is supplemented by four production captured
replay tests registered as `V2_Integration_ROCmFlashAttentionWmmaQKReplay`
(7.49 seconds), plus the existing register-score lifecycle and independent ring
reference gates. Both shipping HIP architectures compile with zero spills.
These correctness results do not establish performance.
Scaling must hold fixed query rows, geometry, precision and retained capacity;
measure 1K/2K/4K/8K/16K/32K/64K live KV including each selected graph’s work.
The M512 default graph contains its count producer and direct attention; the
separate canonical context oracle includes phase summaries and the reducer.
Fit latency with an intercept: causal useful pairs are `512*(KV-255.5)`.
Retain byte checks, partial queries and ring-wrap/reset coverage at the long end.

Local raw evidence for this extension is under
`/tmp/llaminar-gfx1100-wmma-prefill-evidence`; the earlier goal's evidence and
frozen measurement binaries remain separate. No full production-preflight or
real-model E2E qualification is claimed by these component measurements.
