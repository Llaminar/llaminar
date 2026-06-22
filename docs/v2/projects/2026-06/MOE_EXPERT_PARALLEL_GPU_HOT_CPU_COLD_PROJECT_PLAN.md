# MoE Expert Parallel Tiered GPU / CPU Project Plan

**Date:** 2026-05-08
**Status:** Proposed phased project plan
**Branch context:** `feat/qwen35-moe`
**Scope:** Same-layer Expert Parallel execution for Qwen3.5 MoE across ordered accelerator and CPU expert tiers when VRAM is constrained.

## Summary

The current `Qwen35MoEHybridPPTPParityTest.PrefillParityWithGpuExpertCache` exposed an important architecture mismatch. The test was intended to model consumer inference where GPUs hold the dense path and a bounded cache of hot routed experts, while CPU sockets compute cold routed experts with cross-socket tensor parallelism. The implementation modeled this as sequential pipeline parallelism:

```text
PipelineParallel(
    LocalTP(ROCm GPUs)              # layers 0..19
    NodeLocalTP(CPU sockets)        # layers 20..39
)
```

That is the wrong abstraction. Sequential PP assigns different layers to different domains. The intended behavior assigns different expert work for the same MoE layer to different domains and then sums the partial outputs. The simplest instance is GPU-hot / CPU-cold:

```text
Layer N MoE block:
    shared expert        -> GPU LocalTP if it fits
    hot routed experts   -> GPU LocalTP cache
    cold routed experts  -> CPU NodeLocalTP
    final MoE output     -> sum(shared + hot + cold)
```

This project introduces a first-class same-layer Expert Parallel overlay for MoE blocks. It should not be expressed as `--pp-stage` or `GlobalPPTopology` stage ownership.

The design must also support more than two routed tiers, for example:

```text
Layer N MoE block:
    shared expert             -> fast NVIDIA LocalTP if it fits
    hottest routed experts    -> fast NVIDIA LocalTP
    warm routed experts       -> slower AMD MI50 LocalTP
    cold routed experts       -> CPU NodeLocalTP
    final MoE output          -> sum(shared + nvidia_hot + amd_warm + cpu_cold)
```

This plan intentionally uses "expert parallel" as a family term. The concrete implementation must distinguish the low-level stage sharding mode from the higher-level MoE execution policy.

## Related Documents

- `docs/v2/projects/2026-06/HANDOVER_QWEN35_MOE_HYBRIDPPTP_CORRECTNESS_2026-05-08.md`
- `docs/v2/projects/2026-04/MOE_EXPERT_PLACEMENT_DESIGN.md`
- `docs/v2/projects/2026-06/QWEN35_MOE_EXPERT_REBALANCING_PLAN.md`
- `docs/v2/projects/2026-06/MOE_STAGE_DECOMPOSITION_PLAN.md`
- `docs/v2/projects/2026-06/MULTI_DOMAIN_PIPELINE_EXECUTION_PLAN.md`

## Expert Parallel Taxonomy

The existing code and docs use `ExpertParallel` in several nearby places. This feature needs stricter terminology so future agents do not collapse distinct designs into one flag.

### Stage-Local Tensor Parallel Mode

`TPMode::ExpertParallel` should mean one narrow thing:

```text
Inside one TP context, shard the MoE routed-expert dimension across participants.
Each participant computes the selected routed experts that it owns for the same layer.
The stage output is a partial [tokens, d_model] tensor.
A domain-local sum/allreduce produces the full routed-expert output for that layer.
```

This is analogous to `TPMode::RowParallel` and `TPMode::ColumnParallel`: it tells graph lowering which collective is needed after a stage. It must not encode GPU-vs-CPU placement, hot-cache policy, shared-expert residency, or cross-domain dispatch.

Use `TPMode::ExpertParallel` when the question is:

```text
How is this one MoE expert compute stage sharded inside this one collective domain?
```

Do not use `TPMode::ExpertParallel` when the question is:

```text
Which devices should own hot experts, cold experts, or shared experts for a layer?
```

### Static Expert Weight Sharding

`WeightShardingMode::ExpertParallel` should remain a load/materialization concept:

```text
Split a 3D expert weight tensor by expert id at materialization time.
```

That is useful for static expert-id sharding. It is not sufficient for dynamic rebalancing, hot mirroring, or GPU-cache residency because those modes need a placement service that can prepare, migrate, mirror, or evict experts without assuming one immutable owner per expert for the whole run.

### MoE Execution Policy

The consumer-inference feature in this plan needs a higher-level policy above `TPMode`:

```cpp
enum class MoEExpertExecutionKind {
    SingleDomainExpertSharded,   // one domain owns the layer's routed experts by expert id
    TieredExpertOverlay,         // ordered role domains contribute to the same MoE layer
};
```

`SingleDomainExpertSharded` describes the existing CPU-socket scenario:

```text
NodeLocalTP(cpu socket ranks)
    both sockets participate in every MoE layer
    routed experts are assigned or rebalanced across sockets
    selected hot experts may be mirrored on both sockets
    domain-local collectives produce the full routed output
```

`TieredExpertOverlay` describes the GPU-hot / CPU-cold scenario and generalizes to any number of ordered routed tiers:

```text
same MoE layer
    shared expert domain        -> usually fastest GPU tier
    routed tier 0               -> hottest experts, fastest domain
    routed tier 1..N-2          -> progressively cooler experts, lower tiers
    routed fallback tier N-1    -> cold experts, usually CPU NodeLocalTP
    overlay reduce             -> sums all role-domain outputs onto the continuation domain
```

The overlay may contain domains that internally use `TPMode::ExpertParallel`, but the overlay itself is not `TPMode::ExpertParallel`. It is a cross-domain MoE execution plan.

### Domain-Internal Compute Kind

Each expert domain needs to declare how work is split internally:

```cpp
enum class ExpertDomainComputeKind {
    ReplicatedExperts,      // every participant has full selected experts; scheduler picks owner work
    ExpertIdSharded,        // participants own different expert ids; maps to TPMode::ExpertParallel
    TensorParallelExperts,  // each selected expert's GEMMs are tensor-parallel within the domain
};
```

The distinction matters for the CPU cold path. If both CPU sockets help compute every cold expert's gate/up/down projections, the cold domain is `TensorParallelExperts`, not `ExpertIdSharded`. If socket 0 owns cold experts `{0,2,4}` and socket 1 owns `{1,3,5}`, the cold domain is `ExpertIdSharded` and may lower through `TPMode::ExpertParallel`.

### Naming Rule

Use these names consistently:

| Concept | Preferred Name | Do Not Use As |
| --- | --- | --- |
| Stage-local expert-id sharding | `TPMode::ExpertParallel` | Cross-domain hot/cold placement policy |
| Static 3D expert tensor slicing | `WeightShardingMode::ExpertParallel` | Dynamic residency or rebalancing policy |
| Existing CPU socket EP | `SingleDomainExpertSharded` | Sequential PP or GPU-cache overlay |
| New consumer-inference EP | `TieredExpertOverlay` | One `TPMode::ExpertParallel` domain |
| CPU sockets tensor-parallelizing each cold expert | `TensorParallelExperts` | Expert-id sharding |

## Goals

1. Fit all shared expert weights on GPU when the VRAM budget allows.
2. Fit a cache of hot routed experts on GPU, bounded by a configured expert count or remaining VRAM budget.
3. Support multiple ordered routed-expert tiers, such as fast NVIDIA GPUs for hottest experts, slower AMD GPUs for warm experts, and CPU sockets for cold experts.
4. Execute cold routed experts on CPU using NodeLocalTP across CPU socket ranks.
5. Preserve exact MoE numerics: every selected routed expert contribution is computed exactly once unless an explicit replicated-expert dispatch policy selects one owner per token.
6. Keep the existing tensor-parallel and prepared-weight ownership rules: model paths resolve through `PreparedWeightStore` and binding ids, not global `KernelFactory` state or raw tensor-pointer guessing.
7. Produce parity coverage that validates the consumer-inference topology directly, instead of validating a sequential PP approximation.

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
    shared expert domain work
    routed tier 0 domain work
    routed tier 1..N domain work
    fallback CPU NodeLocalTP domain work
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
    routed experts: highest-priority cache up to budget

gpu_warm_domain:
    scope: local
    backend: rccl, nccl, or host-staged heterogeneous as needed
    devices: slower GPU devices owned by one MPI rank
    routed experts: lower-priority warm cache up to budget

cpu_cold_domain:
    scope: node_local
    backend: upi or mpi
    devices: one CPU participant per socket rank
    routed experts: fallback complement of faster routed tiers
```

### Expert Domain Compute Modes

The design must distinguish expert placement from expert weight sharding and domain-internal compute:

```cpp
enum class ExpertDomainComputeKind {
    ReplicatedExperts,      // each participant has full selected experts
    ExpertIdSharded,        // split selected experts by expert id; maps to TPMode::ExpertParallel
    TensorParallelExperts,  // each selected expert's GEMMs are sharded across participants
};
```

Initial correctness can use `ReplicatedExperts` for CPU cold work if necessary. The target for CPU cold work is `TensorParallelExperts`, so both CPU sockets help compute every cold expert's gate/up/down path rather than merely splitting experts by id.

The existing CPU-socket EP scenario should be documented and tested separately as `SingleDomainExpertSharded`: both sockets participate in all layers, experts are rebalanced across sockets, and hot experts may be mirrored inside that same NodeLocalTP domain.

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
    RoutedExpertTier,
};

enum class ExpertResidencyPolicy {
    Disabled,
    StaticById,
    HistogramTieredCache,
    ExplicitMasks,
};

enum class MoEExpertExecutionKind {
    SingleDomainExpertSharded,
    TieredExpertOverlay,
};

struct ExpertRoutedTier {
    std::string name;
    std::string domain;
    int priority = 0;                 // Lower value means hotter / preferred.
    int max_experts_per_layer = 0;    // 0 means budget-driven.
    size_t memory_budget_bytes = 0;   // 0 means infer from remaining capacity.
    bool fallback = false;            // Exactly one routed tier should be fallback.
};

struct ExpertComputeDomain {
    std::string name;
    ExpertDomainKind kind = ExpertDomainKind::SingleDevice;
    CollectiveBackendType backend = CollectiveBackendType::AUTO;
    std::vector<GlobalDeviceAddress> participants;
    ExpertDomainComputeKind compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
};

struct ExpertLayerPlacement {
    int layer = -1;
    std::vector<int> routed_expert_tier; // size = num routed experts, value indexes routed_tiers
};

struct MoEExpertParallelPlan {
    bool enabled = false;
    MoEExpertExecutionKind execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
    std::string continuation_domain;
    std::string shared_expert_domain;
    ExpertResidencyPolicy residency_policy = ExpertResidencyPolicy::Disabled;
    std::vector<ExpertComputeDomain> domains;
    std::vector<ExpertRoutedTier> routed_tiers;
    std::vector<ExpertLayerPlacement> placements;
};
```

### Required Invariants

Validation must enforce:

1. For every MoE layer, every routed expert is assigned to exactly one primary domain.
2. `routed_expert_tier[layer][e]` must reference exactly one valid routed tier unless explicit replicated dispatch is enabled.
3. If shared experts are assigned to GPU, the shared expert weights must be resident and prepared on that domain before execution.
4. CPU cold domain participants must share a domain-scoped TP context when `TensorParallelExperts` is enabled.
5. The continuation domain receives exactly one reduced `[tokens, d_model]` output for the layer.
6. `TPMode::ExpertParallel` may only be selected for domains whose `compute_kind` is `ExpertIdSharded`.
7. `TieredExpertOverlay` must not be lowered as sequential `GlobalPPTopology`; all role domains contribute to the same logical layer.
8. Routed tiers are ordered by priority and must have exactly one fallback tier for experts that do not fit any faster tier.

## Config Surface

Support both YAML and CLI. Both surfaces must parse into the same `MoEExpertParallelPlan` value, and validation must run after merging all config sources.

YAML is the most readable surface for persistent topology files:

```yaml
moe_expert_parallel:
    enabled: true
    execution_kind: tiered_expert_overlay
    continuation_domain: nvidia_fast
    shared_expert_domain: nvidia_fast

    residency:
        mode: histogram

    routed_tiers:
        - name: hottest
          domain: nvidia_fast
          priority: 0
          max_experts_per_layer: 8
          memory_budget_mb: 0    # 0 means infer from remaining capacity

        - name: warm
          domain: amd_warm
          priority: 1
          max_experts_per_layer: 12
          memory_budget_mb: 0

        - name: cold
          domain: cpu_cold
          priority: 2
          fallback: true

    domains:
        nvidia_fast:
            scope: local
            backend: nccl
            devices: [0:cuda:0, 0:cuda:1]
            compute_kind: tensor_parallel_experts

        amd_warm:
            scope: local
            backend: rccl
            devices: [0:rocm:0, 0:rocm:1]
            compute_kind: tensor_parallel_experts

        cpu_cold:
            scope: node_local
            backend: upi
            devices: [0:cpu:0, 1:cpu:0]
            compute_kind: tensor_parallel_experts
```

This should be parsed separately from `--define-domain`/`--pp-stage`. Named domains can share parsing helpers, but this feature is not pipeline stage assignment.

### CLI Surface

Use `--moe-expert-overlay` as the canonical CLI entry point. The flag enables the same-layer MoE overlay and selects the execution kind:

```text
--moe-expert-overlay off|single-domain|tiered
```

Recommended values:

| CLI value | Plan value | Meaning |
| --- | --- | --- |
| `off` | `enabled=false` | Disable MoE expert overlay. |
| `single-domain` | `SingleDomainExpertSharded` | Existing CPU socket EP: one domain covers every routed expert by expert-id ownership/rebalancing. |
| `tiered` | `TieredExpertOverlay` | Ordered routed tiers contribute to the same MoE layer. |

#### Tiered Overlay Example

The NVIDIA + AMD + CPU layout above should be expressible as:

```bash
./build_v2_release/llaminar2 oneshot \
  -m models/qwen35-moe.gguf \
  -p "Hello" \
  --moe-expert-overlay tiered \
  --moe-expert-overlay-continuation nvidia_fast \
  --moe-expert-overlay-shared-domain nvidia_fast \
  --moe-expert-overlay-residency histogram \
  --moe-expert-overlay-domain "nvidia_fast=0:cuda:0,0:cuda:1;scope=local;backend=nccl;compute=tensor_parallel_experts" \
  --moe-expert-overlay-domain "amd_warm=0:rocm:0,0:rocm:1;scope=local;backend=rccl;compute=tensor_parallel_experts" \
  --moe-expert-overlay-domain "cpu_cold=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;compute=tensor_parallel_experts" \
  --moe-expert-overlay-tier "hottest@nvidia_fast;priority=0;max-experts-per-layer=8;memory-mb=auto" \
  --moe-expert-overlay-tier "warm@amd_warm;priority=1;max-experts-per-layer=12;memory-mb=auto" \
  --moe-expert-overlay-tier "cold@cpu_cold;priority=2;fallback=true"
```

This is intentionally not written as `--tp-devices` or `--pp-stage`. Each overlay domain may have its own inner TP mode, but the overlay itself is a same-layer MoE dispatch/reduce policy.

#### CLI Grammar

```text
--moe-expert-overlay <kind>
    kind := off | single-domain | tiered

--moe-expert-overlay-continuation <domain-name>
    domain-name := name of the domain that receives the final reduced MoE output

--moe-expert-overlay-shared-domain <domain-name>
    domain-name := domain where shared experts are resident and executed

--moe-expert-overlay-residency <policy>
    policy := static-by-id | histogram | explicit-masks

--moe-expert-overlay-domain "<name>=<devices>;scope=<scope>;backend=<backend>;compute=<compute>[;owner=<rank>][;ranks=<rank-list>]"
    devices := comma-separated GlobalDeviceAddress values, e.g. 0:cuda:0,0:cuda:1 or 0:cpu:0,1:cpu:0
    scope := single | local | node_local
    backend := auto | nccl | rccl | host | mpi | upi
    compute := replicated_experts | expert_id_sharded | tensor_parallel_experts

--moe-expert-overlay-tier "<name>@<domain>;priority=<n>[;max-experts-per-layer=<n>][;memory-mb=<n|auto>][;fallback=true]"
    Lower priority means hotter / more preferred.
    Exactly one tier must set fallback=true.
```

#### YAML Mapping

The CLI example maps to YAML as follows:

| CLI flag | YAML field |
| --- | --- |
| `--moe-expert-overlay tiered` | `moe_expert_parallel.enabled=true`, `execution_kind=tiered_expert_overlay` |
| `--moe-expert-overlay-continuation nvidia_fast` | `continuation_domain: nvidia_fast` |
| `--moe-expert-overlay-shared-domain nvidia_fast` | `shared_expert_domain: nvidia_fast` |
| `--moe-expert-overlay-residency histogram` | `residency.mode: histogram` |
| `--moe-expert-overlay-domain "name=..."` | `domains.<name>` |
| `--moe-expert-overlay-tier "name@domain;..."` | one item in `routed_tiers` |

#### Single-Domain CPU Socket Example

The existing CPU-socket EP scenario should use the same flag family but select `single-domain`:

```bash
./build_v2_release/llaminar2 oneshot \
  -m models/qwen35-moe.gguf \
  -p "Hello" \
  --moe-expert-overlay single-domain \
  --moe-expert-overlay-continuation cpu_sockets \
  --moe-expert-overlay-shared-domain cpu_sockets \
  --moe-expert-overlay-domain "cpu_sockets=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;compute=expert_id_sharded" \
  --moe-expert-overlay-tier "routed@cpu_sockets;priority=0;fallback=true"
```

This keeps the term "expert overlay" at the user-facing orchestration layer while preserving `TPMode::ExpertParallel` as the lower-level expert-id sharding mode inside the `cpu_sockets` domain.

#### CLI Validation Rules

Validation must reject:

1. `--moe-expert-overlay tiered` with fewer than one `--moe-expert-overlay-tier`.
2. Any tier whose domain was not declared by `--moe-expert-overlay-domain` or YAML `domains`.
3. More or fewer than one fallback tier.
4. Duplicate domain names or duplicate tier names.
5. `compute=expert_id_sharded` without a compatible stage-local `TPMode::ExpertParallel` lowering path.
6. `compute=tensor_parallel_experts` without a domain-scoped TP context.
7. Combining `--moe-expert-overlay-*` tier/domain flags with `--pp-stage` as if they assigned layer ranges. Overlay domains may coexist with PP in a future design, but these flags must not be interpreted as PP stage ownership.
8. `--moe-expert-overlay off` with any other `--moe-expert-overlay-*` flag except possibly diagnostics.

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

### Accelerator Routed Domains

Each accelerator routed tier computes the routed experts assigned to that tier. In a two-tier setup this may be only `gpu_hot_domain`. In a heterogeneous setup it can include a fast NVIDIA tier and a slower AMD tier.

An accelerator domain computes:

1. Shared expert output if shared experts are resident on GPU.
2. Routed expert output for the tier's expert assignment for the layer.
3. Optional local TP reductions inside the domain.

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
shared_partial + routed_tier_0_partial + routed_tier_1_partial + ... + routed_fallback_partial -> moe_combined_output
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

Each phase must leave the tree in a testable state. A phase is not done until its named unit or integration target exists, is wired into `tests/v2/CMakeLists.txt`, and passes in `build_v2_integration` with full parallelism.

### Phase 0: Freeze the Diagnosis and Retire the Wrong Test Semantics

Deliverables:

- Add a short note to the existing HybridPPTP handover or test comments explaining that sequential PP cannot represent same-layer GPU-hot/CPU-cold expert split.
- Rename, skip, or re-scope the current `PrefillParityWithGpuExpertCache` expectation as a known-invalid topology until Expert Overlay exists.
- Preserve the multi-domain PP runner tests, since they validate useful sequential PP infrastructure.

Required tests:

```bash
ctest --test-dir build_v2_integration -R "Qwen35MoEHybridPPTP" --output-on-failure --parallel
```

Acceptance criteria:

- Future agents do not try to fix the current parity failure by snapshot allreduce or by broadening PP masks.
- The test suite no longer treats sequential PP plus hot/cold masks as the consumer-inference correctness target.
- Existing HybridPPTP tests still pass or explicitly skip the invalid GPU-cache parity case with a reason that points to this plan.

### Phase 1: Add Value Types and Validation

Deliverables:

- Add `src/v2/execution/moe/MoEExpertParallelPlan.h` with `MoEExpertExecutionKind`, `ExpertDomainComputeKind`, `ExpertComputeDomain`, `ExpertRoutedTier`, `ExpertLayerPlacement`, and `MoEExpertParallelPlan`.
- Embed an optional plan in `GraphConfig::MoEConfig`.
- Add validation helpers that enforce domain references, tier references, coverage, one fallback tier, and compatible `compute_kind` choices.
- Add `tests/v2/unit/execution/moe/Test__MoEExpertParallelPlan.cpp`.

Required tests:

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*MoEExpertParallelPlan" --output-on-failure --parallel
```

Acceptance criteria:

- Unit tests cover valid two-tier and three-tier plans.
- Unit tests reject missing expert coverage, duplicate expert ownership, missing domain, duplicate domain names, duplicate tier names, missing fallback tier, multiple fallback tiers, and unsupported `compute_kind` combinations.
- No graph execution changes yet.

### Phase 2: Parse YAML and `--moe-expert-overlay` CLI

Deliverables:

- Extend YAML parsing for `moe_expert_parallel`.
- Add `--moe-expert-overlay` and related CLI flags:
    - `--moe-expert-overlay-continuation`
    - `--moe-expert-overlay-shared-domain`
    - `--moe-expert-overlay-residency`
    - `--moe-expert-overlay-domain`
    - `--moe-expert-overlay-tier`
- Convert YAML domains into `ExpertComputeDomain` values.
- Convert CLI domain and tier specs into the same `MoEExpertParallelPlan` shape as YAML.
- Parse tiered-cache configuration and explicit masks if provided.
- Run one shared validator after YAML and CLI values are merged.
- Add parser tests under `tests/v2/unit/config/`, either in the existing orchestration parser test file or in a new `Test__MoEExpertOverlayConfig.cpp`.

Required tests:

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*(OrchestrationConfigParser|MoEExpertOverlayConfig)" --output-on-failure --parallel
```

Acceptance criteria:

- Parser tests construct the 2x ROCm TP + CPU NodeLocalTP plan from YAML and CLI.
- Parser tests construct the 1x CUDA + 2x ROCm + CPU NodeLocalTP plan from YAML and CLI.
- Parser tests cover the `single-domain` CPU socket EP CLI form.
- Invalid configs produce validation errors before model execution.

### Phase 3: Static Residency Planner

Deliverables:

- Add `src/v2/execution/moe/MoEExpertParallelPlanner.{h,cpp}`.
- Implement `StaticById` and explicit-mask placement first.
- Implement `HistogramTieredCache` as a deterministic policy using existing `DecodeExpertHistogram` data when available, with a fallback to by-id tier assignment.
- Estimate shared expert and routed expert memory footprint from model metadata and tensor types.
- Add `tests/v2/unit/execution/moe/Test__MoEExpertParallelPlanner.cpp`.

Required tests:

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*MoEExpertParallelPlanner" --output-on-failure --parallel
```

Acceptance criteria:

- Unit tests prove the planner assigns shared experts to the configured shared domain first.
- Unit tests prove routed experts fill tiers in priority order before assigning the fallback tier.
- Unit tests prove the planner is deterministic for equal histogram counts.
- Planner output satisfies the Phase 1 coverage validator for both required final topologies.

### Phase 4: Model-Light Dispatch and Reduce Stages

Deliverables:

- Add `MoEExpertDispatchStage` for producing per-tier work descriptors from routing results.
- Add `MoEExpertParallelReduceStage` that sums N dense partial tensors into the continuation output.
- First implementation may use host memory and CPU tensors only.
- Add `tests/v2/unit/execution/compute_stages/Test__MoEExpertDispatchStage.cpp`.
- Add `tests/v2/unit/execution/compute_stages/Test__MoEExpertParallelReduceStage.cpp`.

Required tests:

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*MoEExpert(Dispatch|ParallelReduce)Stage" --output-on-failure --parallel
```

Acceptance criteria:

- Dispatch tests cover prefill and decode shapes.
- Dispatch tests cover two routed tiers and three routed tiers.
- Reduce tests use synthetic partial tensors to prove shared + every routed tier contribution sums exactly.
- Reduce tests preserve the continuation-domain output shape and do not mutate input partials unexpectedly.

### Phase 5: Single-Process Correctness MVP

Deliverables:

- Teach `Qwen35MoEGraph::buildFFNGraph()` to emit the Expert Overlay composite path when `config_.moe.expert_parallel.enabled` is true.
- In the MVP, run all routed tiers in one process with CPU tensors or mock contexts.
- Reuse `MoEExpertComputeStage` masks for per-tier partitions.
- Add a model-light graph test, for example `tests/v2/unit/models/qwen35moe/Test__Qwen35MoEExpertOverlayGraph.cpp`.

Required tests:

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*Qwen35MoEExpertOverlayGraph" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "V2_Integration_Parity_Qwen35MoE_SingleDevice" --output-on-failure --parallel
```

Acceptance criteria:

- The model-light graph test proves all routed experts are covered exactly once.
- The final MoE output matches a reference full-expert compute for two-tier and three-tier placements.
- Existing non-overlay Qwen35 MoE tests continue to use the old path.

### Phase 6: Real CPU Fallback NodeLocalTP Domain

Deliverables:

- Create a domain-scoped NodeLocalTP context for the CPU fallback expert domain.
- Move fallback expert computation to CPU socket ranks.
- Add activation transfer from the continuation domain to the CPU fallback domain.
- Add return transfer of the CPU fallback partial output to the continuation domain.
- Start with `ReplicatedExperts` if needed for correctness.
- Add an MPI integration test, for example `tests/v2/integration/moe/Test__MoEExpertOverlay_CPUFallback_MPI.cpp`.

Required tests:

```bash
ctest --test-dir build_v2_integration -R "V2_Integration_.*MoEExpertOverlay_CPUFallback" --output-on-failure --parallel
```

Acceptance criteria:

- Two CPU ranks compute fallback routed partials and return a correct reduced result.
- Communicator creation is deterministic and domain-scoped.
- The test fails fast on missing MPI ranks instead of hanging.

### Phase 7: CPU `TensorParallelExperts` for Fallback Experts

Deliverables:

- Extend weight planning/materialization for fallback expert gate/up/down tensors sharded across CPU NodeLocalTP participants.
- Support per-expert tensor-parallel GEMM execution:
    - gate/up column-parallel
    - down input-parallel
    - domain allreduce for each fallback partial output
- Keep prepared-weight ownership stage/domain scoped.
- Add `tests/v2/integration/moe/Test__MoEExpertOverlay_CPUTensorParallelExperts_MPI.cpp`.

Required tests:

```bash
ctest --test-dir build_v2_integration -R "V2_Integration_.*MoEExpertOverlay_CPUTensorParallelExperts" --output-on-failure --parallel
```

Acceptance criteria:

- CPU fallback domain uses both socket ranks to compute the same selected fallback experts.
- Output matches replicated-expert CPU fallback mode within established tolerances.
- Prepared handles from shared, accelerator-tier, and CPU fallback domains do not overwrite each other.

### Phase 8: Multi-Accelerator Tier Integration

Deliverables:

- Support more than one accelerator routed tier in the same overlay plan.
- Support a single-device CUDA tier with `compute=replicated_experts`.
- Support a 2x ROCm tier with `compute=tensor_parallel_experts` and RCCL local TP.
- Add cross-domain reduce from ROCm partials back to a CUDA continuation domain.
- Add an integration test, for example `tests/v2/integration/moe/Test__MoEExpertOverlay_MultiAcceleratorTiers.cpp`.

Required tests:

```bash
ctest --test-dir build_v2_integration -R "V2_Integration_.*MoEExpertOverlay_MultiAcceleratorTiers" --output-on-failure --parallel
```

Acceptance criteria:

- Synthetic three-tier execution succeeds with `cuda_fast`, `rocm_hot`, and `cpu_cold` domains.
- Cross-domain reduce produces the same result as a single-domain reference.
- Tests skip with a clear message when CUDA or ROCm hardware is unavailable.

### Phase 9: Sparse Token-Row Transfer Optimization

Deliverables:

- Optimize prefill dispatch to send only token rows that route to non-continuation or fallback tiers.
- Preserve original token row indices for scatter-add on return.
- Add decode fast path for one-token fallback routes.
- Add transfer-volume assertions to the dispatch/reduce integration tests.

Required tests:

```bash
ctest --test-dir build_v2_integration -R "V2_Integration_.*MoEExpertOverlay.*(Sparse|Transfer)" --output-on-failure --parallel
```

Acceptance criteria:

- Prefill data transferred to CPU scales with fallback-routed token rows, not full sequence length.
- Decode fallback path transfers one hidden row plus top-k metadata.
- Dense-transfer fallback remains available behind a debug or compatibility option.

### Phase 10: Qwen35 MoE Expert Overlay Parity

Deliverables:

- Add `tests/v2/integration/parity/qwen35moe/Test__Qwen35MoE_ExpertOverlay_Parity.cpp`.
- Wire it into `tests/v2/CMakeLists.txt` with labels such as `Qwen35MoE;ExpertOverlay;MixtureOfExperts;Parity;MPI`.
- Add final parity cases for both required layouts:
    - `PrefillParity_ROCm2TP_SharedHot_CPU2NodeLocalTP_Cold`
    - `DecodeParity_ROCm2TP_SharedHot_CPU2NodeLocalTP_Cold`
    - `PrefillParity_CUDA1_SharedHot_ROCm2TP_Hot_CPU2NodeLocalTP_Cold`
    - `DecodeParity_CUDA1_SharedHot_ROCm2TP_Hot_CPU2NodeLocalTP_Cold`
- Validate overlay topology before running parity.
- Compare `MOE_EXPERT_OUTPUT`, `MOE_COMBINED_OUTPUT`, and `LM_HEAD` against PyTorch snapshots.

Required final test command:

```bash
ctest --test-dir build_v2_integration -R "Qwen35MoEExpertOverlay" --output-on-failure --parallel
```

Required final parity layouts:

```text
Layout A: ROCm shared/hot + CPU cold
  continuation_domain = rocm_hot
  shared_expert_domain = rocm_hot
  routed_tier_0 = rocm_hot, devices 0:rocm:0,0:rocm:1, scope local, backend rccl, compute tensor_parallel_experts
  routed_fallback = cpu_cold, devices 0:cpu:0,1:cpu:0, scope node_local, backend upi, compute tensor_parallel_experts

Layout B: CUDA shared/hottest + ROCm hot + CPU cold
  continuation_domain = cuda_fast
  shared_expert_domain = cuda_fast
  routed_tier_0 = cuda_fast, devices 0:cuda:0, scope single, backend auto, compute replicated_experts
  routed_tier_1 = rocm_hot, devices 0:rocm:0,0:rocm:1, scope local, backend rccl, compute tensor_parallel_experts
  routed_fallback = cpu_cold, devices 0:cpu:0,1:cpu:0, scope node_local, backend upi, compute tensor_parallel_experts
```

Acceptance criteria:

- Both final layouts pass prefill parity.
- Decode parity passes for at least one generated token per layout, or is explicitly tracked as a separate blocker with a failing test if decode support is incomplete.
- `MOE_EXPERT_OUTPUT` no longer shows hot-only sparsity.
- Early-layer MoE parity matches the NodeLocalTP baseline envelope.
- LM head KL/top-k thresholds match or improve on the current NodeLocalTP MoE parity thresholds.
- Tests skip with explicit hardware-precondition messages when the required CUDA, ROCm, or two-rank CPU NodeLocalTP resources are unavailable.

## Test Strategy

Use Integration builds for parity and MPI tests:

```bash
cmake --build build_v2_integration --parallel
ctest --test-dir build_v2_integration -R "V2_Unit_.*MoEExpert(Parallel|Overlay)" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "V2_Integration_.*MoEExpert(Parallel|Overlay)" --output-on-failure --parallel
```

For the final Qwen35 parity matrix:

```bash
ctest --test-dir build_v2_integration \
    -R "Qwen35MoEExpertOverlay" \
  --output-on-failure --parallel
```

The final parity target must include these test cases:

```text
Qwen35MoEExpertOverlayParity.PrefillParity_ROCm2TP_SharedHot_CPU2NodeLocalTP_Cold
Qwen35MoEExpertOverlayParity.DecodeParity_ROCm2TP_SharedHot_CPU2NodeLocalTP_Cold
Qwen35MoEExpertOverlayParity.PrefillParity_CUDA1_SharedHot_ROCm2TP_Hot_CPU2NodeLocalTP_Cold
Qwen35MoEExpertOverlayParity.DecodeParity_CUDA1_SharedHot_ROCm2TP_Hot_CPU2NodeLocalTP_Cold
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
layer, tier, domain, role, enabled_experts, selected_token_rows, transferred_bytes
layer, tier_order, fallback_tier, shared_domain, continuation_domain
layer, shared_partial_norm, tier_partial_norms, final_moe_norm
```

CSV output should be optional and should not run in hot paths by default.

## Open Decisions

1. Should the first correctness MVP implement CPU fallback `ReplicatedExperts` before `TensorParallelExperts`, or go straight to sharded CPU expert GEMMs?
2. Should router execution always stay on the continuation domain, or should fallback participants recompute routing from the transferred hidden state for simpler data movement?
3. Should shared expert GPU residency be mandatory for this mode, or should the planner fall back to CPU shared expert when VRAM is too small?
4. How should tier cache eviction work during decode: per-layer fixed count, per-tier VRAM budget, global VRAM budget, or hybrid?
5. Should the cross-domain reducer be a compute stage in the normal graph or a higher-level sub-orchestrator call owned by a composite MoE stage?

## Definition of Done

The feature is complete when Llaminar can express and execute same-layer MoE Expert Overlay topologies where ordered expert tiers contribute to the same MoE layer and reduce back to the continuation domain.

The final implementation must support at least these two layouts:

```text
Layout A:
MoEExpertOverlay(
    continuation = LocalTP(2x ROCm GPUs),
    shared_expert = LocalTP(2x ROCm GPUs),
    routed_tier_0 = LocalTP(2x ROCm GPUs, shared/hot experts),
    routed_fallback = NodeLocalTP(2 CPU socket ranks, cold experts, tensor-parallel expert GEMMs)
)

Layout B:
MoEExpertOverlay(
    continuation = SingleDevice(1x CUDA GPU),
    shared_expert = SingleDevice(1x CUDA GPU),
    routed_tier_0 = SingleDevice(1x CUDA GPU, shared/hottest experts),
    routed_tier_1 = LocalTP(2x ROCm GPUs, hot routed experts),
    routed_fallback = NodeLocalTP(2 CPU socket ranks, cold experts, tensor-parallel expert GEMMs)
)
```

and Qwen3.5 MoE parity demonstrates:

- every routed expert contribution is covered exactly once,
- hot-only sparsity in `MOE_EXPERT_OUTPUT` disappears,
- shared expert output remains GPU-resident when configured,
- fallback expert output is computed by CPU NodeLocalTP participants,
- intermediate accelerator tier output is included when configured,
- final logits match PyTorch within the calibrated Qwen35 MoE parity thresholds.
