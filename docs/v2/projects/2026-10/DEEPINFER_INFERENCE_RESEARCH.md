# DeepInfer research for Llaminar inference

Research date: 2026-10-20. Status: source assessment and proposed experiments.
Llaminar references describe the working tree based on
`9a0e26bcc5e434464f98a78f2ad242dd7ecb69b3`, including in-progress Qwen Flash Next
implementation changes. Performance benefits proposed here require measurement.

DeepInfer offers useful kernel algorithms and scheduling policies for Llaminar.
The strongest immediate candidate is DeepSelect's bounded-candidate top-k
algorithm, followed by broader attention fusion and expert computation with
overlapped communication. Its cache and cluster architecture provides a useful
direction for shared prefix storage and scheduling by estimated work.

Adoption must preserve Llaminar's device-owned execution, exact grouped
verification, explicit collectives, physical-memory authority, and communication
of only live data. Several DeepInfer runtime choices require a different
implementation under those contracts.

## Release scope and source identity

The reviewed release is
[DeepInfer 8b16149741a4935e1dec8cea27853f273b374c5b](https://github.com/deepseek-ai/DeepInfer/tree/8b16149741a4935e1dec8cea27853f273b374c5b).
It supports DeepSeek-V4.1-Flash on NVIDIA SM100 GPUs and Huawei Ascend 950 NPUs.
The published engine provides offline inference. A complete KV-cache service,
vision/prefill/decode disaggregation, and balancing across prefill/decode
replicas are explicitly outside the released service implementation.
[Release scope](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/README.md)

The kernel assessment uses the revisions pinned by that engine:

| Component | Reviewed revision | Relevant capability |
|---|---|---|
| [DeepSelect](https://github.com/deepseek-ai/DeepSelect/tree/bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18) | `bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18` | Sparse-indexer and sampler top-k |
| [FlashMLA](https://github.com/deepseek-ai/FlashMLA/tree/2e5429fc5653bab6e081f09477126f731882a6a9) | `2e5429fc5653bab6e081f09477126f731882a6a9` | Sparse attention and fused transforms |
| [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM/tree/1fc9c66975ee322d628b3288534ca0e9ce12a64d) | `1fc9c66975ee322d628b3288534ca0e9ce12a64d` | Grouped GEMMs, indexer scoring, and Mega MoE |
| [DeepEP](https://github.com/deepseek-ai/DeepEP/tree/93eb6eb238127e96c6d7a4a625a6dad158348509) | `93eb6eb238127e96c6d7a4a625a6dad158348509` | Expert dispatch/combine and communication buffers |

The engine's support matrix and each library's support matrix differ. In
particular, the pinned FlashMLA release targets SM100/SM103 and Ascend 950 and
removed Hopper support from this generation. DeepGEMM also describes SM90
support. Neither fact establishes a complete DeepInfer deployment on
Llaminar's CPU or ROCm backends.
[FlashMLA requirements](https://github.com/deepseek-ai/FlashMLA/blob/2e5429fc5653bab6e081f09477126f731882a6a9/README.md),
[DeepGEMM requirements](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/README.md)

## Adoption priorities

| Priority | Candidate | Llaminar application | Main qualification |
|---|---|---|---|
| First | DeepSelect algorithm | Qwen sparse selection, then sampler top-k | Identical selected indices and tie ordering with lower measured cost |
| Next | Attention transform fusion | Remove intermediate writes around existing attention/cache stages | Preserve rounding, cache encoding, and exact producer events |
| Next | Grouped GEMM and communication overlap | Improve existing MoE execution | Fixed arithmetic order, economical live payloads, and explicit graph collectives |
| Next | Expert placement using load histories | Extend existing residency policy if measured results justify it | Include device service times and migration cost |
| Later | Confidence-scheduled verification | Extend dynamic MTP allocation | Model-provided confidence, unbiased verification, and device-owned decisions |
| Separate design | Shared prefix archives and phase scheduling | Multi-replica serving and cluster deployment | Compatible state transfer, retrieval economics, and failure ownership |
| Excluded from exact optimization | Approximate SWA bounded replay | No direct adoption into exact restoration | Changes reconstructed attention state |

## Kernel opportunities

### Exact top-k selection

DeepSelect scans scores in blocks, filters against an increasing threshold, and
periodically reduces a small candidate buffer back to the requested top-k.
Its algorithm reads each input element once from global memory and keeps most
selection work in shared memory. Randomized block traversal limits unfavorable
input-order effects; the small-batch CUDA implementation also uses thread-block
clusters to divide a row across SMs.
[Algorithm and implementation](https://github.com/deepseek-ai/DeepSelect/blob/bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18/docs/DeepSelect-deep-dive.md)

Llaminar's current
[`selectPoolsWide`](../../../../src/v2/kernels/qwen4exp/QwenSparseGPU.cuh)
scans the visible score row for each of 31 threshold bits, then resolves ties
and writes indices. The Qwen sparse contract selects 512 complete pools and
appends the incomplete token tail. This makes long-context, small-M selection
the first useful experiment. Repeated source-level loads do not imply 31 times
the physical HBM traffic, so the expected gain must be measured.
[Sparse geometry and ownership](../../../../src/v2/kernels/qwen4exp/QwenSparseAttention.h)

A native adaptation should retain FP32 score values, canonical index
tie-breaking, ascending selected-token order, and device-owned live extents.
Traversal order must not change the selected set when scores tie. Upstream
tests check top-k membership, uniqueness, and requested ordering; they do not
prove Llaminar's particular tie-selection rule.
[DeepSelect tests](https://github.com/deepseek-ai/DeepSelect/blob/bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18/tests/test.py)

Implement the algorithm through the existing CPU/CUDA/ROCm contracts with
backend-specific scheduling. CUDA PTX and cluster communication need distinct
ROCm and CPU implementations. Extend the comparison to vocabulary top-k only
after sparse selection establishes correctness and economy; sampler ordering
also affects subsequent probability accumulation.

### Attention and cache transform fusion

FlashMLA fuses operations around core attention, including query rotary
transforms, output rotary transforms, and FP8 conversion. Its fused path uses
prepared projection-weight permutations. DeepInfer separately provides a
configurable normalization, RoPE, Hadamard, quantization, and cache-store
pipeline.
[FlashMLA fusion](https://github.com/deepseek-ai/FlashMLA/blob/2e5429fc5653bab6e081f09477126f731882a6a9/README.md),
[transform implementation](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/layers/tilelang_ops/cuda/fused_transform.py)

Llaminar already combines Q/K RoPE and has compressed cache append and optional
RoPE/quantization fusion. The useful question is which remaining intermediate
writes and launches cost enough to justify a larger fused operation.
[CUDA RoPE](../../../../src/v2/kernels/cuda/ops/CUDARoPEKernels.cu),
[cache append](../../../../src/v2/execution/compute_stages/stages/KVCacheAppendStage.cpp),
[TurboQuant kernels](../../../../src/v2/kernels/cuda/kvcache/CUDATurboQuantKernels.cu)

Keep required intermediate rounding explicit inside the fused kernel. Weight
permutations belong to prepared-weight ownership and capture identity. Native
KV readers must preserve the selected cache format; changing to a DeepSeek
FP8/FP4 layout is a separate precision policy requiring its own correctness and
economy evidence.

### Grouped expert computation and communication overlap

DeepGEMM supplies grouped GEMMs with device-visible valid work and specialized
layouts for expert computation. Mega MoE combines dispatch, both expert
projections, SwiGLU, and combine in a kernel that overlaps NVLink communication
with tensor-core work. Its symmetric storage and FP8/FP4/BF16 contracts shape
the integration.
[Grouped GEMM and Mega MoE interfaces](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/README.md)

Llaminar's grouped IMMA implementation already fuses gate/up projections and
SwiGLU requantization while retaining the serial arithmetic boundaries.
Broader overlap and removal of remaining expert intermediates are the
interesting comparisons.
[Grouped MoE kernels](../../../../src/v2/kernels/cuda/gemm/CUDAMoEGroupedPrefillKernels.cu)

An integration must express inter-device behavior as an explicit graph
collective with participant-local work. It cannot conceal remote progress
inside an ordinary compute stage. Symmetric buffers need an admitted
`PhysicalMemoryAuthority` BOM, and grouped computation must consume device
counts without host readbacks. New precision families cannot replace existing
NativeVNNI/codebook behavior implicitly.

## KV cache and attention ownership

DeepInfer's cache engine distinguishes paged compressed KV, bounded
per-sequence SWA storage, compression accumulators, and indexer state. Its
cross-layer topology identifies the source and consumers of KV, index keys,
and selected candidates. These are useful references for model-declared
sharing and state-specific persistence.
[Cache engine](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/cache_engine.py),
[cross-layer topology](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/models/v4_modules/cross_layer.py)

For Llaminar, sharing should be declared in model schemas, resolved into typed
producer/consumer bindings, and priced once per physical owner. Compression
and sharing that are part of DeepSeek's trained architecture require matching
model semantics. Qwen sparse attention and GDN retain their own state and
acceptance contracts.

The deployment Q&A describes shared SSD-backed block-KV storage through 3FS
and a separate RAM directory for SWA state. Shared prefix availability reduces
the conflict between cache affinity and load balancing.
[Deployment Q&A, pages 5–6](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/doc/DeepSeek%20V4.1%20Flash%20推理部署%20Q%26A.pdf)

A shared immutable archive backend is therefore worth designing through
[`IPrefixStorageBackend`](../../../../src/v2/execution/prefix_cache/PrefixStorageBackend.h).
It should preserve
[model, topology, and arithmetic fingerprints](../../../../src/v2/execution/prefix_cache/PrefixCacheFingerprint.h),
publication readiness, allocation ownership, and incarnation-safe retirement.
Equivalent prompt text alone cannot establish compatibility between
differently sharded or numerically different state images. Any redistribution
needs an explicit transfer contract.

The same Q&A explicitly calls SWA bounded replay an approximate reconstruction
that truncates attention visibility during replay.
[Deployment Q&A, page 7](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/doc/DeepSeek%20V4.1%20Flash%20推理部署%20Q%26A.pdf)
Exclude this mechanism from exact prefix restoration and grouped-verifier
optimization. It cannot justify recomputing an approximation when an exact
checkpoint is required.

## Cluster orchestration and scheduling

### Schedule by estimated work

DeepSeek describes smaller expert-parallel groups for prefill and wider groups
for decode. Its prefill scheduler prices incoming and cached context lengths;
decode balancing considers both request count and total KV length.
[Deployment Q&A, pages 2–5](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/doc/DeepSeek%20V4.1%20Flash%20推理部署%20Q%26A.pdf)

For Llaminar, evaluate replica placement with queue delay, missing-prefix
computation, archive retrieval, state transfer, and expected decode work.
Extend the existing
[measured request cost model](../../../../src/v2/planning/PlanningRequestCostModel.h)
instead of introducing an independent hardware or memory ledger. Cached and
uncached prompts with equal total token counts can have very different
execution costs.

Phase-specific placement belongs in an explicit service design: authenticate
the state produced by prefill, transfer only the receiver's required state,
and define when the decode participant becomes its live owner. Size these
deployments from measured workload and fabric behavior; DeepSeek's deployment
ratios are not portable defaults for heterogeneous CPU/CUDA/ROCm clusters.

### Place experts using simultaneous load

DeepInfer's expert balancer allocates redundant copies to hot experts and
places physical experts using historical per-step load vectors. Its placement
objective sums the maximum rank load over those steps, accounting for experts
that peak together.
[Placement policy](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/eplb/policy.py)

Use that objective as a candidate against Llaminar's
[existing rebalance policy](../../../../src/v2/execution/moe/DeviceMoERebalancePolicyShared.h).
Convert routed-token demand into measured service cost per backend, and include
weight movement, replica capacity, and amortization over subsequent work.
Better balance alone does not prove lower latency. Device-owned retained
statistics and bounded admitted scratch must replace any host-driven dynamic
allocation in the reference policy.

### Allocate verification work by expected benefit

DSpark uses draft confidence estimates and a measured execution curve to choose
verification work. DeepInfer's policy maximizes expected accepted tokens per
estimated step time and distributes the selected budget across requests.
[Verification policy](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/csrc/engine/mtp_varlen.hpp),
[device confidence allocation](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/layers/tilelang_ops/cuda/schedule_mtp_varlen.py)

Llaminar already has
[learned depth decisions](../../../../src/v2/execution/mtp/MTPDepthLearnedPolicy.h)
and a
[device generation policy](../../../../src/v2/execution/mtp/MTPDeviceGenerationPolicy.h).
The useful extension is a measured marginal-benefit policy where the model
provides qualified confidence estimates. A draft probability cannot be assumed
to be a calibrated acceptance estimate.

Budgeting, request remapping, and per-request verification lengths must remain
device-owned. Prove prefix-closed allocation, correct stochastic verification,
and serial-row byte equivalence before measuring throughput. Future draft
information must not bias an earlier token's acceptance.

## Integration constraints

| Area | Observed constraint | Required Llaminar treatment |
|---|---|---|
| Runtime scheduling | DeepInfer's worker copies corrected lengths to CPU and synchronizes completion events. | Retain complete device-controlled generation and the permitted terminal/ticket boundaries. |
| Context parallelism | A YoCo compaction path gathers padded contributions and selects useful rows afterward. | Compact payloads before communication; capacity must not determine transfer extent. |
| Attention determinism | FlashMLA disables split-KV when batch invariance is enabled. | Measure the required invariant arithmetic path and preserve economical fixed reductions. |
| Dependency ABI | Pinned DeepEP requires NCCL 2.32.3 or newer; Llaminar pins patched 2.28.9. | Port and requalify capture-reentry and live-row repairs before dependency integration. |
| Communication resources | DeepEP dispatch/combine consume GPU SMs; PP/CP/DP bucket primitives are experimental. | Price communication SM use and qualify each selected transport and collective. |
| Kernel preparation | DeepGEMM and DeepEP use runtime kernel compilation. | Prepare admitted specializations and persistent storage before capture; deployment artifacts must authenticate compiler and architecture identity. |

The scheduler observations come from the
[worker implementation](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/worker.py);
the padded collective is in
[YoCo boundary compaction](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/models/v4_modules/yoco.py).
FlashMLA's
[split-KV condition](https://github.com/deepseek-ai/FlashMLA/blob/2e5429fc5653bab6e081f09477126f731882a6a9/csrc/api/sparse_decode.cpp#L219)
makes batch invariance affect the available parallel schedule.
The dependency and resource requirements are documented in
[pinned DeepEP](https://github.com/deepseek-ai/DeepEP/blob/93eb6eb238127e96c6d7a4a625a6dad158348509/README.md)
and
[pinned DeepGEMM](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/README.md);
Llaminar's current native dependency authority is
[install-nccl.sh](../../../../scripts/docker/install-nccl.sh).

These observations do not establish a defect in DeepInfer's own supported
contract. They identify where direct reuse would violate Llaminar's
[architecture and ownership requirements](../../../../.github/instructions/llaminar-v2-architecture.instructions.md).

## Proposed evaluation sequence

### First experiment

Implement a DeepSelect-style candidate for the existing Qwen sparse selector.
Compare against the current implementation with identical stored scores and
output geometry.

- Sweep M from one through the admitted grouped range, representative context
  lengths, the selection threshold, all tail lengths, and partial buckets.
- Exercise equal scores, boundary ties, empty/inactive rows, and large-to-small
  captured replays. Preserve the existing non-finite-input contract.
- Require identical selected indices and downstream attention results. A
  merely valid top-k set is insufficient.
- Measure kernel latency, actual memory traffic, scratch capacity, registers,
  spills, and occupancy, then measure complete Release decode and MTP
  transactions with the production arithmetic policy.

### Subsequent experiments

| Work | Comparison | Evidence required before adoption |
|---|---|---|
| Attention fusion | Current fused boundaries versus a larger native fusion | Exact intermediate semantics, lower traffic or launch cost, and improved complete-request latency |
| Expert overlap | Current grouped execution versus a declared compute/collective implementation with overlap | Actual overlap, lower critical-path time, and exact live communication extents |
| Expert placement | Current policy versus placement using per-step load histories | Lower measured service time after migration cost across stationary and shifting hotspots |
| Verification allocation | Current dynamic policy versus confidence/cost allocation | Unchanged verifier semantics and higher accepted-token throughput |
| Shared archives | Current local tiers versus an explicit shared backend | Compatible cross-replica restore, retrieval/recompute economics, and sound cancellation/retirement behavior |

Keep allocation admission in `PhysicalMemoryAuthority`, movement in
`TransferEngine`, and backend selection in typed kernel interfaces. GPU
integration must retain complete capture, exact streams/events, reset/replay
equivalence, and device-owned live counts. Communication tests must measure
payload extents for empty participants, skewed ownership, partial buckets, and
large-to-small replay.

Follow the canonical
[testing workflow](../../../../.agents/llaminar-testing/SKILL.md) and applicable
[CPU](../../../../.agents/cpu-tuning/SKILL.md),
[CUDA](../../../../.agents/cuda-tuning/SKILL.md), and
[ROCm](../../../../.agents/rocm-tuning/SKILL.md) profiling procedures when
implementation starts. New production defects need focused regressions in
`ProductionTestPreflight`; supported formats, backends, and geometries need
their full correctness and economy coverage. Compile every shipped GPU
architecture when qualifying new kernels.

Complete-request comparisons must hold model, prompt tokens, sampling policy,
cache state, precision, and placement constant. Use PerfStats to authenticate
the exercised production path, and keep profiler runs separate from timing.
Kernel wins that disappear after selection, conversion, storage, or
communication costs do not justify adoption.
