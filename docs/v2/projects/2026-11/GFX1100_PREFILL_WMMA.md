# ROCm prefill matrix-instruction coverage

Dashboard byte limit: **15,000 UTF-8 bytes**. This file records current coverage
and qualification; backend procedures remain in the ROCm tuning skill.

The active objective covers four areas: broader dense dispatch, grouped routed
experts, attention, and Q8 router logits. Each requires production-path component
correctness, captured replay, unprofiled Release timing and isolated profiling.
Floating half-operand GEMM is an additional confirmed gap in the sampled vendor
path: its actual loaded gfx1100 code object uses vector mixed-FP16 FMA, not WMMA.
The earlier two-anchor NativeQuantised objective is documented separately in
[GFX1100_NATIVE_QUANTISED_WMMA.md](GFX1100_NATIVE_QUANTISED_WMMA.md).

## Coverage found in the source audit

| Operation | CUDA implementation | ROCm implementation and gap |
|---|---|---|
| Dense quantized prefill | Native prepared-weight adapters select INT8 matrix tiles and preserve source scale arithmetic. | `ROCmNativeQuantisedWmma.hip` already implements all physical decoders on actual gfx1100. The generated production WMMA entries cover only N8192/K2048 at M512 and M1024. Broaden measured exact dispatch for QKV, attention outputs, FFN, shared experts and declared TP participants. |
| Routed quantized MoE | `CUDAMoEGroupedPrefillKernels.cu` uses `mma.sync.m16n8k32` for grouped projections. | `ROCmMoEGroupedPrefillKernels.hip` still uses ordered integer dot products. Add native WMMA leaves under the existing ExpertTiled device-owned admission; retain RouteOwned for its admitted live-work domain. |
| Attention prefill | `CUDAFlashAttentionKernels.cu:1109` uses WMMA half QK with FP32 score accumulation; its general PV is scalar FP32. The active bounded HD256 specialization additionally uses tensor-core PV at `CUDATensorPrefillKernels.cu:328`. | `ROCmFlashAttentionKernels.hip:1440` uses vector half dot products and ordered PV arithmetic. A native entry must preserve same-ROCm bytes across query/context scheduling, batch sizes, ring layouts and repeated replay. |
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
| Grouped MoE | Design reviewed; implementation pending | Pending | Pending |
| Attention | Numerical and lifecycle constraints reviewed; implementation pending | Pending | Pending |
| Q8 router | Typed forced recording; normal and compact owned-row bridges; cooperative K subtrees, fixed-width variants and local scale shuffles | Ten focused ROCm tests pass, including normal/owned capture, finite-scale extremes, raw compact logits and poisoned tails | Expanded 384-cell and fixed-width 192-cell exploratory sweeps pass. Latest M512 anchors improve three model widths; K8192/E512 regresses and remains excluded. Automatic installation pending |
| Floating half GEMM | Existing production hipBLAS path; sampled native matrix gap confirmed | Three retained graph replays pass, each containing cast and vendor GEMM | Loaded bytes/ISA authenticated; complete Release cast+GEMM timing pending |

Router timing includes input quantization, logits and, for Hadamard input,
restoration of the original expert Q8 bank. The component harness alternates
31 paired native-event samples over 16 serial captured operations, after ten
warmups. Profiling uses a separate explicitly selected test and one ROCTx region;
it cannot run accidentally as part of the ordinary MoE speedometer.

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

Grouped MoE should reuse the directory16 ABI and source-exact contribution
helpers. Initial tile candidates are M16N32 and M16N64 with two staged K blocks.
Preserve original serial partition policy for down slices, route-slot versus
token-index addressing, split phases, fixed column ownership and ordered final
publication. Physical IQ4_NL (4) and source IQ4_XS (24) share a grouped launch
family but have different auxiliary metadata; do not erase that distinction.
Any new translation unit owns and initializes its IQ constant tables before capture.

Attention should retain canonical 256-key partitions and the existing FP32
summary/reducer layout. The same native numerical entry must apply to both
query/context schedules and all request batch sizes in its admitted domain.
CUDA's batch-one-only restriction would break existing ROCm byte invariants.
CPU numerical thresholds and same-backend byte comparisons must remain intact.

Local raw evidence for this extension is under
`/tmp/llaminar-gfx1100-wmma-prefill-evidence`; the earlier goal's evidence and
frozen measurement binaries remain separate. No full production-preflight or
real-model E2E qualification is claimed by these component measurements.
