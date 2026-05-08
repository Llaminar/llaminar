# MoE Expert Parallel GPU-Hot / CPU-Cold Project Plan

**Date:** 2026-05-08
**Status:** Proposed phased project plan
**Branch context:** `feat/qwen35-moe`
**Scope:** Same-layer Expert Parallel execution for Qwen3.5 MoE when VRAM is constrained.

## Summary

The current `Qwen35MoEHybridPPTPParityTest.PrefillParityWithGpuExpertCache` exposed an important architecture mismatch. The test was intended to model consumer inference where GPUs hold the dense path and a bounded cache of hot routed experts, while CPU sockets compute cold routed experts with cross-socket tensor parallelism. The implementation modeled this as sequential pipeline parallelism:

```text
PipelineParallel(
    LocalTP(ROCm GPUs)              # layers 0..19
    NodeLocalTP(CPU sockets)        # layers 20..39
)
```

That is the wrong abstraction. Sequential PP assigns different layers to different domains. The intended behavior assigns different expert work for the same MoE layer to different domains and then sums the partial outputs:

```text
Layer N MoE block:
    shared expert        -> GPU LocalTP if it fits
    hot routed experts   -> GPU LocalTP cache
    cold routed experts  -> CPU NodeLocalTP
    final MoE output     -> sum(shared + hot + cold)
```

This project introduces a first-class same-layer Expert Parallel overlay for MoE blocks. It should not be expressed as `--pp-stage` or `GlobalPPTopology` stage ownership.

## Related Documents

- `docs/v2/HANDOVER_QWEN35_MOE_HYBRIDPPTP_CORRECTNESS_2026-05-08.md`
- `docs/v2/MOE_EXPERT_PLACEMENT_DESIGN.md`
- `docs/v2/QWEN35_MOE_EXPERT_REBALANCING_PLAN.md`
- `docs/v2/MOE_STAGE_DECOMPOSITION_PLAN.md`
- `docs/v2/cleanup/MULTI_DOMAIN_PIPELINE_EXECUTION_PLAN.md`

## Goals

1. Fit all shared expert weights on GPU when the VRAM budget allows.
2. Fit a cache of hot routed experts on GPU, bounded by a configured expert count or remaining VRAM budget.
3. Execute cold routed experts on CPU using NodeLocalTP across CPU socket ranks.
4. Preserve exact MoE numerics: every selected routed expert contribution is computed exactly once unless an explicit replicated-expert dispatch policy selects one owner per token.
5. Keep the existing tensor-parallel and prepared-weight ownership rules: model paths resolve through `PreparedWeightStore` and binding ids, not global `KernelFactory` state or raw tensor-pointer guessing.
6. Produce parity coverage that validates the consumer-inference topology directly, instead of validating a sequential PP approximation.

## Non-Goals

- Do not make sequential PP compute cold experts for earlier GPU-owned layers.
- Do not fix this only in the parity harness by allreducing snapshots.
- Do not add an ad hoc special case to `Qwen35MoEHybridPPTPParityTest` that hides missing cold expert work.
- Do not reintroduce global prepared-weight caches for model GEMM or expert GEMM ownership.
- Do not require GPU-to-CPU cold expert fallback to be fast in the first MVP. Correctness and explicit scheduling come first.

## Current Failure That Motivates This Plan

The latest focused HybridPPTP parity run reaches comparison but fails at routed MoE expert output:

```text
LM_HEAD KL divergence: 0.890850782 (threshold: 0.0600)
LM_HEAD Top-5: 40.0%
Early layers passed: 1/6 (threshold: 4/6)
```

The first large drop is layer 0 `MOE_EXPERT_OUTPUT`. The Llaminar output is hot-only and sparse, often with row zero fractions such as `0.666667`, `0.888889`, or `1.0`, while PyTorch is dense. This is expected if the GPU PP stage computes only hot experts and no same-layer CPU cold domain contributes the missing experts.

Relevant current behavior:

- `src/v2/models/qwen35moe/Qwen35MoEGraph.cpp` marks routed expert output partial whenever `expert_mask` is non-empty, then inserts a TP allreduce only inside that stage's TP domain.
- `tests/v2/integration/parity/qwen35moe/Test__Qwen35MoE_HybridPPTP_Parity.cpp` applies hot masks to the ROCm domain and cold masks to the CPU domain, but those domains own different PP layer ranges.
- The CPU domain never executes layers 0..19 in the current topology, so it cannot contribute cold experts for layer 0.

## Target Architecture

### Conceptual Execution

For each MoE layer:

```text
FusedResidualNorm
MoERouting
MoEExpertDispatch
    GPU hot/shared domain work
    CPU cold NodeLocalTP domain work
MoEExpertParallelReduce
MoECombine
FFN residual / next layer
```

The MoE block is a same-layer composite. The continuation domain is usually the GPU/root domain, so the final reduced MoE output should land back on the GPU/root activation buffer.

### Domain Roles

```text
gpu_hot_domain:
    scope: local
    backend: rccl or nccl
    devices: GPU devices owned by one MPI rank
    shared experts: all if VRAM allows
    routed experts: hot cache up to budget

cpu_cold_domain:
    scope: node_local
    backend: upi or mpi
    devices: one CPU participant per socket rank
    routed experts: complement of GPU hot cache
```

### Expert Weight Parallelism Modes

The design must distinguish expert placement from expert weight sharding:

```cpp
enum class ExpertWeightParallelism {
    ReplicatedExperts,       // each participant has full selected experts
    TensorParallelExperts,   // each selected expert's GEMMs are sharded across participants
};
```

Initial correctness can use `ReplicatedExperts` for CPU cold work if necessary. The target for CPU cold work is `TensorParallelExperts`, so both CPU sockets help compute every cold expert's gate/up/down path rather than merely splitting experts by id.

## Proposed Core Types

Place these near the MoE execution/config layer, not in global PP topology. A likely home is under `src/v2/execution/moe/` with a compact value embedded in `GraphConfig::MoEConfig`.

```cpp
enum class ExpertDomainKind {
    SingleDevice,
    LocalTP,
    NodeLocalTP,
};

enum class ExpertPlacementRole {
    SharedExpert,
    RoutedHotCache,
    RoutedColdFallback,
};

enum class ExpertResidencyPolicy {
    Disabled,
    StaticById,
    HistogramHotCache,
    ExplicitMasks,
};

struct ExpertComputeDomain {
    std::string name;
    ExpertDomainKind kind = ExpertDomainKind::SingleDevice;
    CollectiveBackendType backend = CollectiveBackendType::AUTO;
    std::vector<GlobalDeviceAddress> participants;
    ExpertWeightParallelism weight_parallelism = ExpertWeightParallelism::ReplicatedExperts;
};

struct ExpertLayerPlacement {
    int layer = -1;
    std::vector<bool> gpu_hot_experts;
    std::vector<bool> cpu_cold_experts;
};

struct MoEExpertParallelPlan {
    bool enabled = false;
    std::string continuation_domain;
    std::string shared_expert_domain;
    std::string hot_expert_domain;
    std::string cold_expert_domain;
    ExpertResidencyPolicy residency_policy = ExpertResidencyPolicy::Disabled;
    std::vector<ExpertComputeDomain> domains;
    std::vector<ExpertLayerPlacement> placements;
};
```

### Required Invariants

Validation must enforce:

1. For every MoE layer, every routed expert is assigned to exactly one primary domain.
2. `gpu_hot_experts[layer][e]` and `cpu_cold_experts[layer][e]` cannot both be true unless explicit replicated dispatch is enabled.
3. If shared experts are assigned to GPU, the shared expert weights must be resident and prepared on that domain before execution.
4. CPU cold domain participants must share a domain-scoped TP context when `TensorParallelExperts` is enabled.
5. The continuation domain receives exactly one reduced `[tokens, d_model]` output for the layer.

## Config Surface

Start with YAML support. CLI flags can come later.

```yaml
moe_expert_parallel:
  enabled: true
  continuation_domain: gpu_hot
  shared_expert_domain: gpu_hot
  hot_expert_domain: gpu_hot
  cold_expert_domain: cpu_cold

  hot_cache:
    mode: histogram
    max_experts_per_layer: 8
    vram_budget_mb: 0        # 0 means infer from remaining VRAM

  domains:
    gpu_hot:
      scope: local
      backend: rccl
      devices: [0:rocm:0, 0:rocm:1]
      weight_parallelism: replicated_experts

    cpu_cold:
      scope: node_local
      backend: upi
      devices: [0:cpu:0, 1:cpu:0]
      weight_parallelism: tensor_parallel_experts
```

This should be parsed separately from `--define-domain`/`--pp-stage`. Named domains can share parsing helpers, but this feature is not pipeline stage assignment.

## Runtime Design

### Dispatch Contract

`MoEExpertDispatchStage` should consume routing outputs and produce compact per-domain work descriptors.

For prefill, the cold CPU path should eventually receive only token rows that route to cold experts:

```text
hidden_rows_for_cold_tokens
route_indices_for_cold_entries
route_weights_for_cold_entries
token_row_indices
```

For decode, the descriptor is small:

```text
single hidden row
top-k cold expert ids
top-k cold route weights
```

The first MVP may send the full normalized hidden tensor to the CPU domain for simplicity. A later phase must optimize this to sparse token-row transfer.

### GPU Hot Domain

The GPU domain computes:

1. Shared expert output if shared experts are resident on GPU.
2. Routed expert output for `gpu_hot_experts[layer]`.
3. Optional local TP reductions inside the GPU domain.

The current `MoEExpertComputeStage` mask behavior is useful here, but the mask must represent a complete same-layer partition together with the CPU cold domain, not a PP-stage-only partial.

### CPU Cold Domain

The CPU domain computes routed cold expert partials.

MVP mode:

```text
ReplicatedExperts:
    each CPU participant can compute its assigned cold experts or a token partition
    domain allreduce sums partial routed output
```

Target mode:

```text
TensorParallelExperts:
    each cold expert GEMM is sharded across CPU participants
    gate/up column-parallel + down input-parallel
    domain allreduce produces the cold routed output
```

### Cross-Domain Reduce

Add a reducer that combines domain partials onto the continuation domain:

```text
gpu_hot_partial + cpu_cold_partial + shared_partial -> moe_combined_output
```

This is not a normal allreduce. Usually the final result only needs to exist on the continuation domain. The reducer can be implemented initially as host-staged transfer plus add, using existing `TransferEngine` and tensor coherence APIs.

## File Landmarks

Likely files to modify or add:

```text
src/v2/config/OrchestrationConfig.h
src/v2/config/OrchestrationConfigParser.cpp
src/v2/models/GraphTypes.h
src/v2/models/qwen35moe/Qwen35MoEGraph.cpp
src/v2/models/qwen35moe/Qwen35MoEGraphConfigBuilder.cpp
src/v2/execution/moe/MoEExpertParallelPlan.h
src/v2/execution/moe/MoEExpertParallelPlanner.h
src/v2/execution/moe/MoEExpertParallelPlanner.cpp
src/v2/execution/compute_stages/stages/MoEExpertDispatchStage.h
src/v2/execution/compute_stages/stages/MoEExpertDispatchStage.cpp
src/v2/execution/compute_stages/stages/MoEExpertParallelReduceStage.h
src/v2/execution/compute_stages/stages/MoEExpertParallelReduceStage.cpp
src/v2/execution/compute_stages/ComputeStageFactory.h
src/v2/execution/compute_stages/ComputeStageFactory.cpp
```

Likely tests:

```text
tests/v2/unit/execution/moe/Test__MoEExpertParallelPlan.cpp
tests/v2/unit/execution/moe/Test__MoEExpertParallelPlanner.cpp
tests/v2/unit/execution/compute_stages/Test__MoEExpertDispatchStage.cpp
tests/v2/unit/execution/compute_stages/Test__MoEExpertParallelReduceStage.cpp
tests/v2/integration/parity/qwen35moe/Test__Qwen35MoE_ExpertParallel_Parity.cpp
```

## Phased Delivery Plan

### Phase 0: Freeze the Diagnosis and Retire the Wrong Test Semantics

Tasks:

- Add a short note to the existing HybridPPTP handover or test comments explaining that sequential PP cannot represent same-layer GPU-hot/CPU-cold expert split.
- Rename or disable the current `PrefillParityWithGpuExpertCache` expectation as a known-invalid topology until Expert Parallel exists.
- Preserve the multi-domain PP runner tests, since they validate useful sequential PP infrastructure.

Acceptance criteria:

- Future agents do not try to fix the current parity failure by snapshot allreduce or by broadening PP masks.
- The test suite no longer treats sequential PP plus hot/cold masks as the consumer-inference correctness target.

### Phase 1: Add Value Types and Validation

Tasks:

- Add `MoEExpertParallelPlan`, `ExpertComputeDomain`, and related enums.
- Embed an optional plan in `GraphConfig::MoEConfig`.
- Add validation helpers that enforce per-layer expert coverage and domain references.
- Add unit tests for valid and invalid plans.

Acceptance criteria:

- Unit tests cover full coverage, missing expert, duplicate expert, missing domain, and unsupported weight-parallelism combinations.
- No graph execution changes yet.

### Phase 2: Parse YAML Configuration

Tasks:

- Extend YAML parsing for `moe_expert_parallel`.
- Convert YAML domains into `ExpertComputeDomain` values.
- Parse hot-cache configuration and explicit masks if provided.
- Keep CLI changes out of this phase unless trivial.

Acceptance criteria:

- Parser tests construct the sample GPU-hot/CPU-cold plan.
- Invalid configs produce validation errors before model execution.

### Phase 3: Static Residency Planner

Tasks:

- Add `MoEExpertParallelPlanner`.
- Implement `StaticById` or explicit-mask placement first.
- Implement `HistogramHotCache` as a deterministic policy using existing `DecodeExpertHistogram` data when available, with a fallback to by-id hot cache.
- Estimate shared expert and routed expert memory footprint from model metadata and tensor types.

Acceptance criteria:

- Unit tests prove the planner assigns shared experts to GPU first, hot routed experts to GPU up to the configured limit, and the complement to CPU.
- Planner output satisfies the Phase 1 coverage validator.

### Phase 4: Model-Light Dispatch and Reduce Stages

Tasks:

- Add `MoEExpertDispatchStage` for producing hot/cold work descriptors from routing results.
- Add `MoEExpertParallelReduceStage` that sums two dense partial tensors into the continuation output.
- First implementation may use host memory and CPU tensors only.

Acceptance criteria:

- Unit tests use synthetic routing and partial tensors to prove hot + cold + shared contributions sum exactly.
- Dispatch tests cover prefill and decode shapes.

### Phase 5: Single-Process Correctness MVP

Tasks:

- Teach `Qwen35MoEGraph::buildFFNGraph()` to emit the Expert Parallel composite path when `config_.moe.expert_parallel.enabled` is true.
- In the MVP, run both hot and cold domains in one process with CPU tensors or mock contexts.
- Reuse `MoEExpertComputeStage` masks for hot and cold partitions.

Acceptance criteria:

- A model-light graph test proves all routed experts are covered and the final MoE output matches a reference full-expert compute.
- Existing non-EP Qwen35 MoE tests continue to use the old path.

### Phase 6: Real CPU Cold NodeLocalTP Domain

Tasks:

- Create a domain-scoped NodeLocalTP context for the cold expert domain.
- Move cold expert computation to CPU socket ranks.
- Add activation transfer from continuation/GPU domain to CPU cold domain.
- Add return transfer of the cold partial output to the continuation domain.
- Start with `ReplicatedExperts` if needed for correctness.

Acceptance criteria:

- MPI integration test with two CPU ranks computes cold routed partials and returns a correct reduced result.
- No deadlocks: all MPI communicator splits and transfers are deterministic and domain-scoped.

### Phase 7: CPU TensorParallelExperts for Cold Experts

Tasks:

- Extend weight planning/materialization for cold expert gate/up/down tensors sharded across CPU NodeLocalTP participants.
- Support per-expert tensor-parallel GEMM execution:
  - gate/up column-parallel
  - down input-parallel
  - domain allreduce for each cold partial output
- Keep prepared-weight ownership stage/domain scoped.

Acceptance criteria:

- CPU cold domain uses both socket ranks to compute the same selected cold experts.
- Output matches replicated-expert CPU cold mode within established tolerances.
- Prepared handles from GPU hot and CPU cold domains do not overwrite each other.

### Phase 8: Sparse Token-Row Transfer Optimization

Tasks:

- Optimize prefill dispatch to send only token rows that route to cold experts.
- Preserve original token row indices for scatter-add on return.
- Add decode fast path for one-token cold routes.

Acceptance criteria:

- Prefill data transferred to CPU scales with cold-routed token rows, not full sequence length.
- Decode cold path transfers one hidden row plus top-k metadata.

### Phase 9: Qwen35 Expert Parallel Parity Test

Tasks:

- Add `Test__Qwen35MoE_ExpertParallel_Parity.cpp`.
- Configure:
  - GPU LocalTP domain for shared experts and hot routed expert cache.
  - CPU NodeLocalTP domain for cold routed experts.
  - Same-layer Expert Parallel plan for all MoE layers.
- Validate topology before running parity.
- Compare `MOE_EXPERT_OUTPUT`, `MOE_COMBINED_OUTPUT`, and `LM_HEAD` against PyTorch snapshots.

Acceptance criteria:

- `MOE_EXPERT_OUTPUT` no longer shows hot-only sparsity.
- Early-layer MoE parity matches the NodeLocalTP baseline envelope.
- LM head KL/top-k thresholds match or improve on the current NodeLocalTP MoE parity thresholds.

## Test Strategy

Use Integration builds for parity and MPI tests:

```bash
cmake --build build_v2_integration --parallel
ctest --test-dir build_v2_integration -R "V2_Unit_.*MoEExpertParallel" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "V2_Integration_.*MoEExpertParallel" --output-on-failure --parallel
```

For the eventual Qwen35 parity test:

```bash
ctest --test-dir build_v2_integration \
  -R "Qwen35MoEExpertParallel.*PrefillParity" \
  --output-on-failure --parallel
```

Do not artificially limit build or test parallelism.

## Diagnostics and Observability

Add logs behind existing or new debug environment settings:

```text
LLAMINAR_MOE_EP_TRACE=1
LLAMINAR_MOE_EP_DUMP_PLACEMENT=1
LLAMINAR_MOE_EP_TRANSFER_TRACE=1
```

Recommended trace output:

```text
layer, domain, role, enabled_experts, selected_token_rows, transferred_bytes
layer, hot_experts, cold_experts, shared_on_gpu, gpu_budget_mb
layer, hot_partial_norm, cold_partial_norm, final_moe_norm
```

CSV output should be optional and should not run in hot paths by default.

## Open Decisions

1. Should the first correctness MVP implement CPU cold `ReplicatedExperts` before `TensorParallelExperts`, or go straight to sharded CPU expert GEMMs?
2. Should router execution always stay on GPU/root, or should CPU cold participants recompute routing from the transferred hidden state for simpler data movement?
3. Should shared expert GPU residency be mandatory for this mode, or should the planner fall back to CPU shared expert when VRAM is too small?
4. How should hot expert cache eviction work during decode: per-layer fixed count, global VRAM budget, or hybrid?
5. Should the cross-domain reducer be a compute stage in the normal graph or a higher-level sub-orchestrator call owned by a composite MoE stage?

## Definition of Done

The feature is complete when Llaminar can express and execute this same-layer MoE Expert Parallel topology:

```text
MoEExpertParallel(
    continuation = LocalTP(ROCm GPUs),
    shared_expert = LocalTP(ROCm GPUs),
    routed_hot = LocalTP(ROCm GPUs, cache limited by VRAM),
    routed_cold = NodeLocalTP(CPU socket ranks, tensor-parallel expert GEMMs)
)
```

and Qwen3.5 MoE parity demonstrates:

- every routed expert contribution is covered exactly once,
- hot-only sparsity in `MOE_EXPERT_OUTPUT` disappears,
- shared expert output remains GPU-resident when configured,
- cold expert output is computed by CPU NodeLocalTP participants,
- final logits match PyTorch within the calibrated Qwen35 MoE parity thresholds.
