# ROCm prefill matrix-instruction coverage

Dashboard limit: **15,000 UTF-8 bytes**. Current coverage and qualification; backend procedures remain in the ROCm tuning skill.

The active objective covers four areas: broader dense dispatch, grouped routed
experts, attention, and Q8 router logits. Each requires production-path component
correctness, captured replay, unprofiled Release timing and kernel profiling.
Attention additionally requires linear complete-operation scaling through 64K
KV tokens with the query bucket held fixed, as explicitly confirmed by the user.
The added [MTP work list](GFX1100_MTP_GROUPED_WMMA_PLAN.md) has all21-source exact-hosted
Auto timing:1,264 native wins/1,890 cells (1.002–2.005×);626 keys keep Dot4.
Release171 native cases, five focused ROCm registrations and all168 per-source
profiles pass with zero memory spills. MTP router Auto now wins85/120 keys
(geo1.104×), with seven focused registrations and80 spill-free profiles green.
Shared-expert MTP Auto wins all177 native-active cells (geo1.437×), with all368
profiles complete. QKV/KV Auto wins all1,415 native cells (geo1.320×); focused
gates and64 profiles pass. GDN Auto qualifies161 keys with16 profiles; connected
FFN Auto qualifies136 GU keys (confirmation geo1.112×), with six host/26
GPU gates and16 profiles. Routed MTP, Wo and selected LM-head remain.
rocBLAS remains the half-operand implementation: three custom WMMA variants lost
every measured cell. [Investigation and deferral](GFX1100_HALF_OPERAND_PREFILL_INVESTIGATION.md).
The earlier two-anchor NativeQuantised objective is documented separately in
[GFX1100_NATIVE_QUANTISED_WMMA.md](GFX1100_NATIVE_QUANTISED_WMMA.md).

## Coverage found in the source audit

| Operation | CUDA implementation | ROCm implementation and gap |
|---|---|---|
| Dense quantized prefill | Native prepared-weight adapters select INT8 matrix tiles and preserve source scale arithmetic. | `ROCmNativeQuantisedWmma.hip` already implements all physical decoders on actual gfx1100. The original installed anchors cover N8192/K2048 at M512/1024. The installed 68-key QKV/shared-expert expansion passes all84 rebuilt actual-Auto cells. |
| Routed quantized MoE | `CUDAMoEGroupedPrefillKernels.cu` uses `mma.sync.m16n8k32` for grouped projections. | `ROCmMoEGroupedWmma.hip` implements source-exact M16N32/N64 leaves under existing ExpertTiled device admission. Automatic admits measured Qwen and Compact population domains. RouteOwned owns sparse/empty work. |
| Attention prefill | `CUDAFlashAttentionKernels.cu:1109` uses WMMA half QK with FP32 score accumulation; its general PV is scalar FP32. The active bounded HD256 specialization additionally uses tensor-core PV at `CUDATensorPrefillKernels.cu:328`. | `ROCmFlashAttentionKernels.hip` now has an explicit native four-plane QK candidate preserving the existing half-dot chains and FP32 PV. Automatic installs one-kernel Dot2/WMMA live-prefix selection in the measured FP16 M512 domain. |
| Q8 router logits | `CUDARouterQ8PrefillDevice.cuh` uses exact INT8 MMA with the router's fixed FP32 tree. | Four/eight/sixteen-wave candidates preserve the ROCm adjacent-leaf tree. Automatic installs Sixteen for whole-row M512/K2560/E512 in both bases; compact-owned rows remain Dot4. |
| Explicit half-operand floating projection | `CuBLASGemmKernel.cu:380` submits FP16 A/B and FP32 C through the tensor-operation library interface. | `HipBLASGemmKernel.cpp:291` submits half A/B and FP32 C through `hipblasGemmEx`, but the sampled M512/N8192/K2048 gfx1100 dispatch selects a vector FMA implementation. Custom WMMA variants lost; retain rocBLAS. |

Native FP16/BF16 storage alone does not imply the half-operand arithmetic policy.
Other floating projections keep their declared FP32 activation/reduction contract
on both backends. Obsolete blockwise adapter families are not evidence of current
production dispatch: GPU prepared quantized weights use NativeQuantised adapters.

## Qualification status

| Area | Implementation | Functional/component evidence | Performance evidence |
|---|---|---|---|
| Broader dense dispatch | Bound installed-Auto/WMMA collector and selective publisher | Core-table ownership replay, eight binding and 16 decision regressions pass | 68 keys installed (684 total);168 profiles and84 rebuilt Auto cells pass; Auto gains1.22653–2.05860×, geometric mean1.39951× |
| Grouped MoE | Wave64 M16N32/N64; fused Dot4/WMMA whole-expert floors4/8/12/16; all physical decoders | Serial, promoted, mixed-writer omission, threshold/reset and split/sliced-down replay pass | Ordinary router-Q8/token-output tournament:15,810 samples, six policies/five cases; full Qwen N32 wins1.20750×/1.15204×, mixed wide1.05358×; Qwen Auto results below |
| Attention | Automatic one-kernel Dot2/WMMA live-prefix selection in the measured FP16 M512 domain | Actual Auto, immutable exclusions, counters, all-format, threshold/ring/partial/reset and internal timing-owner replays pass | Accepted64K gains1.155×/1.138× and doubling1.998×/1.993×;26 cells/14 counter passes |
| Q8 router | Automatic Sixteen-wave tile for exact whole-row M512/K2560/E512; explicit 4/8/16-wave candidates retained | 14 GPU feature cases, five pure policy cases and both timing owners pass; actual Auto template, ownership and submission counters authenticated | Accepted full512 gains2.744×/2.551×;12 cells/seven counter passes |
| Floating half GEMM | Retain production hipBLAS after bounded custom-kernel investigation | Three candidate variants pass retained byte/replay checks | All six complete cast+GEMM cells lose for every variant; custom replacement deferred |

[Router qualification](GFX1100_ROUTER_LIVE_PREFILL.md) records admission, retained-prefix timing and profiles.

Focused replay covers every source format, serial-row bytes, mixed/promoted
descriptors, K/N tails, column slices, branch omissions and dirty/reset replay.

[Operation proof](GFX1100_GROUPED_PREFILL_OPERATION_PROOF.md) records the accepted
ordinary router-Q8/token-output tournament:15,810 samples across five cases and
six policies. Full Qwen uniform/hotset favors N32; power-law favors WideFloor16;
sparse work favors Dot4. GU128 was correct but unpromoted. Bounded511 was rejected.

The feature branch admits population Automatic for the exact Qwen domain,
using separate ordered/native GU consumers and fused Down, with direct one-row
GU arithmetic and selection folded into the GU directory. All22,785 Release
refreshed samples pass: uniform29/35 wins (mean1.040×), hotset33/35 (1.146×),
power-law26/35 (1.031×); full gains1.183×/1.160×/1.041×. Remaining losses reach
1.2%/0.9%/1.5%.
Both Release and Integration compile the complete target set without memory
spills. All233 focused native executions and both host checks are green.
Compact Auto shares one typed directory and one adaptive consumer per role.
All 233 focused native executions pass, plus both cases after the IQ2_S grid
change. Q4_0 geometric/full gains are 1.107×/1.101×; IQ2_S gains 1.080×/1.051×.
Sparse latency increases reach 2.3%/3.3%. All 70 exact counter passes are green, with zero
scratch and zero reported LDS bank conflicts. All42 refreshed Qwen counter passes
are also green. This slice is qualified; residual small-work losses remain open.
The [operation proof](GFX1100_GROUPED_PREFILL_OPERATION_PROOF.md) owns measurements,
ownership regressions and archive identities.

The user removed isolation receipts and recollection from the work requirements.
Existing dense/CUDA-reference, router and attention evidence remains accepted;
the [isolation audit](GFX1100_PREFILL_ISOLATION_REQUALIFICATION.md) is superseded.

Attention Automatic resolves the measured immutable domain once at capture:
FP16/B1/M512/HD256/capacity65536,Q/KV12/2 or8/1, full causal query scheduling and
register-owned scores. One retained kernel chooses WMMA for complete linear
prefixes at KV1024–65536 and Dot2 for short, partial or ring work. Unmeasured
immutable domains remain Dot2. Submission counters identify the actual captured
engine/branch policy and stay unchanged during replay. Existing graph ownership
and identity cover this deterministic selection; no host live-count cache is added.
[Qualification](GFX1100_ATTENTION_ADAPTIVE_QK.md) separates actual Auto evidence
from the earlier explicit candidate.

Fresh actual-Auto timing uses internal events around complete retained
producer/query batches:16 operations below2048 keys, one thereafter. Every
sample poisons and independently observes parameter/output bytes.26 lifecycle
cells/806 pairs include threshold1023/1024/1025,8193,64K,reset512 and regrowth.
Producer hashes match the recorded runs.

| Heads | Dot2 at64K | Automatic | Ratio |32K→64K growth| Affine R² |
|---|---:|---:|---:|---:|---:|
| Dense Q12/KV2 |47.073ms|40.747ms|1.155×|1.998×|0.999941|
| MoE Q8/KV1 |35.571ms|31.269ms|1.138×|1.993×|0.999981|

The actual-Auto fixed-query gate passes R²≥0.995 and last doubling≤2.2 across
1K–64K. Short prefixes remain near Dot2; dense1023 retains0.27% median overhead,
so no universal per-size win is claimed. Three host policy registrations pass
2.53s and six native feature registrations pass168.66s, including all immutable
exclusions, independent engine expectations, reset/ring/partial bytes and counters.

Both guarded HIP targets compile in Release/Integration. Exact512/65536 Auto
profiles use96VGPR/50SGPR,16KiB dynamic LDS,256threads,wave64 and zero scratch/spills.
Each profile's seven passes authenticate loaded code, PCI/UUID, positive activity and
restored AUTO power. Sixteen static WMMA sites do not imply that512 executes WMMA.
Current evidence is `attention-automatic-scaling-{receipt,summary}.json` and
`attention-automatic-{512,65536}-profile/`. All focused replay registrations are
explicit in production preflight; only focused/feature gates ran for this task.

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

Router's final 12-cell/31-pair panel wins every retained live-prefix cell in both
bases. Profiles authenticate loaded code and restored AUTO power, with zero
scratch/spills. The linked router proof retains raw timings, counter values,
source identities and the earlier candidate tournament.

Dense plans bind policy bytes, compiled core and Release producer. Loader replay
guards table interposition; additive installation preserves unchanged rows.
All168 broader profiles pass;68 additions compose the684-row table. Six focused
gates,15 host cases and12 installed-source replay groups pass.
All84 rebuilt actual-Auto source cells now pass: every median beats original
Auto by1.22653–2.05860×, geometric mean1.39951×, with authenticated684-row table
and restored AUTO power (`dense-installed-auto-accepted/campaign.json`). Earlier
anchors/CUDA comparisons remain usable. [Evidence](GFX1100_DENSE_PREFILL_CANDIDATE_SCOPE.md).

The floating audit authenticates the actual gfx1100 rocBLAS vector-FMA dispatch
with zero spills; its Integration trace is not Release timing qualification.
See `floating-half-authenticated-profile/derived/` for the loaded object/ISA.
Grouped floating direct-WMMA witnesses change native input precision and the
explicit-half reduction result. The existing three byte gates pass; retain their
fixed-order producers. Grouped attention now passes production row/capture,
special-value and append/RoPE replay, plus fixed-query64K scaling. Its current
byte-exact WMMA candidate loses all90 measured comparisons; Automatic retains
Dot2. The bounded grouped-attention economy experiment is closed; see the
[MTP work list](GFX1100_MTP_GROUPED_WMMA_PLAN.md).

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
