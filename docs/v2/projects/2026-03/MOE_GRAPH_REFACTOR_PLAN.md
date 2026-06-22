# MoE Graph Refactor Plan

**Date**: 2026-03-12  
**Status**: Proposed  
**Scope**: Graph building, graph execution, model metadata, MoE routing/expert execution, stage-domain binding for heterogeneous expert placement, and GPU graph capture strategy for Qwen3-MoE and Qwen3.5 GatedDeltaNet MoE

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Goals](#goals)
- [Non-Goals](#non-goals)
- [Current State](#current-state)
- [Target Architecture](#target-architecture)
- [Design Details](#design-details)
- [GPU Graph Capture Strategy](#gpu-graph-capture-strategy)
- [Distributed and Expert-Parallel Strategy](#distributed-and-expert-parallel-strategy)
- [CPU Offloaded Expert TP](#cpu-offloaded-expert-tp)
- [CLI And Orchestrator Implications](#cli-and-orchestrator-implications)
- [Concrete CLI Proposal](#concrete-cli-proposal)
- [Qwen Family Mapping](#qwen-family-mapping)
- [Implementation Phases](#implementation-phases)
- [File-by-File Refactor Plan](#file-by-file-refactor-plan)
- [Validation Plan](#validation-plan)
- [Risks and Open Questions](#risks-and-open-questions)

---

## Problem Statement

Llaminar's current graph runtime is structurally generic, but its actual graph-building contract is still dense-Qwen specific.

The main symptoms are:

1. `DeviceGraphOrchestrator` is still concretely coupled to `Qwen2Graph`, even though its header describes a generic graph execution layer.
2. `GraphConfig`, `LayerWeights`, and `ActivationBuffers` model a dense transformer block with attention plus a single SwiGLU FFN.
3. Qwen3 has a separate schema path, but graph config building and graph execution still converge on dense Qwen2-shaped runtime contracts rather than a fully first-class architecture family end-to-end.
4. MoE support exists only as partial substrate: placeholder stages, preliminary expert placement APIs, and `ExpertParallel` enums that are not resolved into executable graphs.
5. GPU graph capture is designed for stable dense decode graphs. A naive MoE implementation with variable expert subgraphs will fight the current capture model.

This is acceptable for Qwen2/Qwen3 dense models, but it will not scale cleanly to:

- **Qwen3-MoE**: standard attention blocks with routed MoE FFNs.
- **Qwen3.5 GatedDeltaNet MoE**: hybrid per-layer sequence mixer selection plus routed MoE FFNs.

The refactor goal is therefore not "add one more model type." The goal is to make the graph system represent:

- per-layer **sequence mixer kind**,
- per-layer **feed-forward kind**,
- per-layer **state kind**,
- per-stage or per-sub-block **parallelism strategy**,
- per-stage **execution domain binding**,

without forcing every new architecture through `Qwen2Graph` and dense-only weight/buffer structs.

---

## Goals

1. Make MoE a first-class graph concept rather than a special case inside the dense FFN builder.
2. Separate sequence mixing from feed-forward structure so hybrid models do not require a monolithic custom graph builder.
3. Preserve the existing strengths of the runtime: declarative graph building, imperative execution, segmented GPU graph replay, TP/PP support, and buffer contract validation.
4. Provide a correctness-first path for MoE graphs before chasing full capture or fused sparse execution.
5. Reuse existing MoE substrate where it is directionally correct: expert placement, routing/distribution helpers, and schema-level `ExpertParallel` intent.
6. Keep Qwen2 and Qwen3 dense support working during the migration.
7. Make CPU-offloaded experts a first-class placement mode inside the same refactor, rather than a later orchestration-only special case.

---

## Non-Goals

1. This plan does not attempt to solve all optimized MoE kernels in one step.
2. This plan does not require monolithic GPU graph capture for routed MoE in the first implementation.
3. This plan does not redesign the full pipeline-parallel subsystem.
4. This plan does not replace existing dense Qwen execution paths immediately; it introduces a migration path.
5. This plan does not define the final low-level math kernels for Gated DeltaNet. That is covered by the existing GDN project planning and is only integrated here at the graph-contract level.

---

## Current State

### What Works Today

- `DeviceGraphOrchestrator` owns execution state, graph cache, and decode replay behavior.
- `DeviceGraphExecutor` executes declarative `ComputeGraph` DAGs.
- `Qwen2Graph` builds dense transformer graphs for embedding, attention, FFN, and LM head.
- Qwen3 has its own schema path, but graph config building and execution still partially fold through shared dense-Qwen machinery such as `Qwen2GraphConfigBuilder` and optional Q/K norm handling.
- Decode graph capture supports monolithic or segmented replay depending on stage capturability.
- Weight placement and work distribution already contain some MoE-oriented concepts.

### What Is Structurally Wrong for MoE

#### 1. Orchestrator-to-builder coupling

`DeviceGraphOrchestrator` stores a `std::shared_ptr<Qwen2Graph>` and exposes `graphBuilder()` as `Qwen2Graph*`. That means the execution layer is not actually builder-agnostic.

#### 2. Dense-only model contract

`GraphTypes.h` defines:

- `GraphConfig` with dense-transformer dimensions only.
- `LayerWeights` with attention weights plus `gate_proj`, `up_proj`, `down_proj`, `ffn_norm`.
- `ActivationBuffers` with `gate`, `up`, `ffn_output`, but no routing or expert-pack buffers.

This contract cannot represent:

- routed experts,
- shared experts,
- top-k routing metadata,
- expert pack/scatter buffers,
- DeltaNet recurrent state.

#### 3. Qwen3 is not fully first-class end-to-end

Qwen3 already has a separate schema/factory path, but `Qwen2GraphConfigBuilder` still accepts both `qwen2` and `qwen3`, and the downstream contract remains the dense-Qwen contract. Qwen3-specific behavior is therefore only partially first-class today.

#### 4. Existing MoE stages are placeholders

`MoEStages` currently express the rough shape of router/expert/combine work, but:

- router execution is CPU-only,
- expert execution is placeholder-only,
- combine execution is not the final packed/scatter model,
- stage params are host-vector centric instead of stable tensor-contract centric.

#### 5. Resolver support is incomplete

`GraphSchema` already defines `StageType::MoERouter`, `StageType::MoEFFN`, and `TPMode::ExpertParallel`, but `GraphResolver` explicitly logs that `ExpertParallel` is not implemented.

#### 6. Model metadata is too weak

`IExecutionPlanBuilder::ModelConfig` and the loader-exposed model metadata do not yet carry MoE and hybrid-mixer information such as:

- number of experts,
- activated experts per token,
- shared experts,
- expert intermediate sizes,
- per-layer block kinds,
- non-attention recurrent state dimensions.

#### 7. Domain routing is too coarse

The current multi-domain routing model effectively answers only:

- which domain attention uses,
- which domain non-attention uses.

That is not enough for MoE layers that need separate routing for:

- router,
- local experts,
- CPU-offloaded experts,
- combine/scatter.

---

## Target Architecture

The core architectural change is to make a layer definition compositional instead of hard-coding "attention + dense FFN".

### New Layer Model

Each transformer layer should be represented by a `LayerSpec` with three structural axes plus two execution policies:

1. **SequenceMixerKind**
   - `Attention`
   - `GatedDeltaNet`

2. **FeedForwardKind**
   - `DenseSwiGLU`
   - `RoutedMoE`

3. **SequenceStateKind**
   - `KVCache`
   - `DeltaNetState`
   - `None`

4. **BlockParallelismPolicy**
  - records parallelism per sub-block or stage role, not as one enum for the whole layer
  - examples:
    - attention path -> `TensorParallel`
    - routed experts -> `ExpertParallel`
    - layer ownership -> `PipelinePartitioned`

5. **ExecutionDomainPolicy**
  - maps stage roles inside the layer to concrete TP domains or placement classes
  - distinguishes router, expert execution, and combine from the coarse FFN bucket

That gives clean composition:

| Family | Mixer | FFN | State |
|--------|-------|-----|-------|
| Qwen2 | Attention | DenseSwiGLU | KVCache |
| Qwen3 | Attention + QK norm | DenseSwiGLU | KVCache |
| Qwen3-MoE | Attention | RoutedMoE | KVCache |
| Qwen3.5 GDN-MoE | GatedDeltaNet or Attention, per layer | RoutedMoE | DeltaNetState or KVCache, per layer |

### High-Level Refactor Principle

The graph system should stop asking "is this Qwen2Graph or not?" and start asking:

- what block kinds exist in this layer,
- which weights are present for those blocks,
- which buffers and state objects are required,
- which parts are capturable,
- which parallelism policy applies to each stage role.
- which stage roles bind to which execution domains.

### Stage-domain model

MoE support is not complete unless the graph contract can express that different stages inside one layer may execute in different domains.

The target model should make stage roles explicit, for example:

- `AttentionTP`
- `DenseFFNTP`
- `MoERouter`
- `MoEExpertLocal`
- `MoEExpertCPU`
- `MoECombine`

These roles are then bound during graph lowering to concrete domains such as:

- local GPU TP domains,
- local mixed-vendor GPU domains,
- within-node `CPU_CROSS_RANK` domains.

---

## Design Details

## 1. Promote architecture metadata to first-class config

### New model-level concepts

Add the following model metadata concepts to the runtime contract:

- `ArchitectureFamily`
  - `Qwen2Dense`
  - `Qwen3Dense`
  - `Qwen3MoE`
  - `Qwen3_5HybridMoE`

- `LayerSpec`
  - `mixer_kind`
  - `ffn_kind`
  - `state_kind`
  - `uses_qk_norm`
  - `uses_shared_expert`
  - `router_top_k`
  - `num_local_experts`
  - `expert_intermediate_dim`

- `MoEConfig`
  - `num_experts`
  - `num_shared_experts`
  - `top_k`
  - `router_score_dtype`
  - `router_logits_dtype`
  - `expert_parallel_mode`

- `HybridSequenceConfig`
  - per-layer mixer layout
  - DeltaNet head dimensions/state dimensions
  - attention-layer rope metadata if mixed layouts differ

### Why this change is necessary

Without these types, every downstream subsystem has to infer MoE from weight-name accidents or model-string switches. That is brittle and guarantees more `if (architecture == ...)` branches.

## 2. Replace dense-only `LayerWeights` with composable block weights

### Proposed new weight structs

Replace the monolithic dense layer weight contract with sub-weights:

- `AttentionWeights`
  - `wq`, `wk`, `wv`, `wo`, `attn_norm`, optional biases, optional q/k norm

- `DenseFFNWeights`
  - `ffn_norm`, `gate_proj`, `up_proj`, `down_proj`

- `MoERouterWeights`
  - router gate projection, optional router norm, optional router bias

- `ExpertFFNWeights`
  - `gate_proj`, `up_proj`, `down_proj`

- `MoEWeights`
  - `ffn_norm`
  - router weights
  - list or accessor for local expert weights
  - optional shared expert weights

- `DeltaNetWeights`
  - all sequence-mixer-specific tensors required by the GDN implementation

- `LayerWeights`
  - `LayerSpec spec`
  - optional `AttentionWeights attention`
  - optional `DenseFFNWeights dense_ffn`
  - optional `MoEWeights moe_ffn`
  - optional `DeltaNetWeights deltanet`

### Design requirement

The final `LayerWeights` type must allow a layer to contain:

- attention and dense FFN,
- attention and MoE FFN,
- DeltaNet and MoE FFN,

without introducing illegal null pointer combinations that callers must guess about.

## 3. Replace dense-only activation buffers with block-specific working sets

### Proposed new buffer structs

- `AttentionBuffers`
- `DenseFFNBuffers`
- `MoEBuffers`
  - router logits
  - router top-k indices
  - router top-k weights
  - token pack offsets
  - packed token buffer
  - packed expert output buffer
  - shared expert output buffer
  - combine/scatter output buffer

- `DeltaNetBuffers`
  - projection outputs
  - recurrent state workspaces
  - mixer output buffers

- `LayerBuffers`
  - `residual`, `normalized`, `current_hidden`
  - optional `AttentionBuffers`
  - optional `DenseFFNBuffers`
  - optional `MoEBuffers`
  - optional `DeltaNetBuffers`

This removes the current assumption that every layer always needs `gate`, `up`, and `ffn_output`.

## 4. Make the graph builder interface truly architecture-neutral

### Current issue

Although the comments say `DeviceGraphOrchestrator` is graph-builder based, the concrete type is still `Qwen2Graph`.

### Target structure

Introduce or strengthen a truly generic builder interface with capabilities like:

- build embedding graph
- build layer graph
- build mixer subgraph
- build FFN subgraph
- build LM head graph
- expose schema
- expose graph-specific state requirements

Then make the orchestrator hold `std::shared_ptr<IGraphBuilder>` instead of `std::shared_ptr<Qwen2Graph>`.

### Block builders

Within model families, compose the final graph from block builders:

- `AttentionBlockBuilder`
- `DenseFFNBlockBuilder`
- `MoEBlockBuilder`
- `DeltaNetBlockBuilder`

`Qwen2Graph` then becomes a composition root for dense Qwen, not the universal representation of all Qwen-like models.

## 5. Add a stage-role domain model

### Why this is part of the core refactor

CPU-offloaded expert TP is not an optional deployment detail. It changes what the graph contract must be able to express.

If the graph only knows "attention" and "FFN", then the runtime has nowhere to represent:

- router on GPU,
- expert dispatch to CPU,
- TP-sharded expert execution in a CPU domain,
- return/combine on the original continuation device.

So stage-domain binding needs to be part of the same refactor as `LayerSpec`, `GraphTypes`, and `GraphResolver`.

### Recommended abstraction

Add an explicit stage-domain binding concept alongside per-role parallelism metadata, rather than trying to summarize the entire layer with one parallelism enum.

Suggested shape:

- `StageDomainRole` enum for graph/schema intent,
- `ExecutionDomainPolicy` or equivalent on `LayerSpec` / graph config,
- resolved graph nodes carry either `domain_role` or a concrete bound `TPDomain` reference.

### Expected behavior

This model should support all of the following without special-case orchestration logic:

- dense attention on a GPU TP domain,
- dense FFN on a different domain,
- routed MoE experts split across local and CPU expert domains,
- router and combine remaining on the primary layer device.

## 6. Build MoE as a fixed-shape graph shell

### Key rule

Do **not** model the first MoE implementation as "build a different expert subgraph for each routing outcome." That breaks graph caching and fights graph capture.

### Instead, represent MoE as a stable graph shell

Recommended fixed-shape stage sequence:

1. `router_logits`
2. `router_topk_select`
3. `token_pack`
4. `expert_dispatch`
5. `expert_ffn_execute`
6. `shared_expert_execute`
7. `expert_combine`
8. `residual`

The topology remains constant. Routing decisions become tensor data, not control flow.

### Why this matters

This gives:

- deterministic graph structure,
- stable buffer contracts,
- segmented replay compatibility,
- a path to later GPU-kernel optimization.

## 7. Introduce a generic sequence-state abstraction

Qwen3.5 hybrid MoE needs more than MoE support. It needs a graph/runtime contract for non-KV recurrent state.

### Proposed state abstraction

- `ISequenceState`
  - clear/reset
  - per-layer state lookup
  - shape and device introspection

- `KVSequenceState` wrapping existing KV-cache behavior
- `DeltaNetSequenceState` for Gated DeltaNet recurrent state

This prevents the system from assuming that every sequence mixer owns a KV cache.

## 8. Add a first-class MoE schema and resolver contract

### Current issue

The schema already hints at MoE support, but the resolver does not lower it into concrete runtime graph pieces.

### Target

Extend `GraphSchema` and `GraphResolver` so MoE stages have:

- explicit input/output buffer contracts,
- explicit tensor parallel vs expert parallel lowering behavior,
- explicit collectives for packed-token exchange if expert ownership is distributed,
- explicit capturability metadata.

This is the point where `TPMode::ExpertParallel` stops being a placeholder and becomes a real lowering strategy.

---

## GPU Graph Capture Strategy

## Principle

The initial MoE implementation should preserve correctness and stable replay before trying to fully capture sparse routing.

### Phase 1 capture policy

- Keep embedding, dense pre/post blocks, and LM head behavior unchanged.
- Treat router, token pack, expert dispatch, and combine as **manual or non-capturable segments** unless their implementation is already shape-stable and device-local.
- Allow the surrounding dense shell to continue participating in segmented decode replay.

### Phase 2 capture policy

Once router/top-k/pack/combine are implemented as stable tensor-contract stages with fixed workspaces:

- mark them capturable where backend support is solid,
- preserve segmented capture as the default,
- consider monolithic capture only after the shell is proven stable.

### Explicit design decision

The system should optimize for **capture-compatible stable topology**, not for host-side routing logic that happens to work in eager execution.

---

## Distributed and Expert-Parallel Strategy

## Existing reusable substrate

The repository already has useful starting points:

- `IWorkDistributor` already models `ExpertAssignment` and `TokenRouting`.
- `WeightPlacementMap` already has shared-expert and local-expert placement helpers.
- `GraphSchema` already has `ExpertParallel` intent.

## Required refactor

### 1. Plan/build-time metadata

Add expert-parallel information to the model and execution-plan contracts so placement and graph lowering are driven by typed metadata, not by name conventions.

### 2. Runtime graph lowering

Implement `ExpertParallel` lowering in `GraphResolver` with explicit stage insertion for:

- token routing metadata movement,
- expert-local execution,
- optional packed-token all-to-all or gather/scatter,
- output combine synchronization.

### 3. Shared expert handling

Shared experts must be represented distinctly from routed experts because they have different ownership and execution patterns.

### 4. Local-first correctness path

The first production MoE graph should support:

- single-device MoE,
- local multi-device expert placement,

before full cross-rank expert-parallel routing is treated as required for correctness.

---

## CPU Offloaded Expert TP

This plan also needs to cover a more specific deployment mode: routed MoE experts that are offloaded to CPU and executed as a tensor-parallel subgraph across the participating CPU MPI ranks.

### What the current runtime already gets right

The codebase already has the correct physical abstraction for this execution model:

- one MPI rank per socket,
- socket-local NUMA allocation per rank,
- explicit collective communication across sockets,
- `TPDomainType::CPU_CROSS_RANK` for CPU TP,
- domain-aware collective execution in `CollectiveContext`.

That means the physical execution substrate does **not** need to be reinvented.

### What is still missing

The current domain-routing API is still effectively layer-scoped:

- attention path asks for one domain,
- non-attention path asks for one domain.

That works for dense "attention on one domain, FFN on another" execution, but it is too coarse for MoE because a single layer may need all of the following at once:

- router on the main accelerator path,
- local experts on the current device,
- CPU-offloaded experts on a `CPU_CROSS_RANK` domain,
- combine on the original continuation device.

So the real gap is not missing CPU TP. The gap is missing **sub-layer domain binding**.

### Recommended design direction

Keep the existing TP-domain machinery, but move from:

- `domainForLayer(layer_idx, is_attention)`

to a stage-role-aware model such as:

- `domainForStageRole(layer_idx, StageDomainRole role)`

or explicit per-stage domain binding in the resolved graph.

This is why CPU expert TP belongs inside the main refactor plan rather than after it: the same stage-domain model is required for generic MoE lowering even before CPU offload is enabled.

### New stage-domain roles

The graph contract should distinguish at least these execution roles:

- `AttentionTP`
- `DenseFFNTP`
- `MoERouter`
- `MoEExpertLocal`
- `MoEExpertCPU`
- `MoECombine`

For CPU expert offload, `MoEExpertCPU` should bind to a dedicated `CPU_CROSS_RANK` TP domain.

### Execution model for CPU-offloaded experts

The recommended execution flow is:

1. Run router logits and top-k on the main layer device.
2. Partition selected tokens by expert and by target expert domain.
3. Pack activations into per-domain expert dispatch buffers.
4. Dispatch packed activations to the CPU expert domain.
5. Execute the expert FFN there as a TP-sharded expert subgraph across all participating CPU ranks.
6. Use domain-scoped collectives inside that CPU domain where the expert's TP layout requires allreduce or allgather.
7. Return packed outputs to the combine stage.
8. Scatter/combine back into token order on the layer continuation device.

This preserves the desired NUMA behavior: each socket stores only its local shard of the offloaded expert weights, and cross-socket traffic remains explicit and bounded.

### Important modeling rule

CPU-offloaded experts should be modeled as **expert-domain execution**, not as a generic "CPU FFN layer" fallback.

That distinction matters because the router and combine stages usually belong on a different device/domain than the expert compute itself.

### Initial scope recommendation

The first implementation should keep the scope intentionally narrow:

- one CPU expert TP domain per physical node,
- within-node cross-rank CPU TP only,
- no cross-node CPU expert routing in the first version,
- no attempt to make the whole MoE path graph-capturable immediately.

That gives a clean path to correctness without entangling the MoE refactor with wide-area expert routing.

### How this fits the implementation phases

- phase 2 introduces the config types that can express domain policy,
- phase 4 introduces the stable MoE shell stages and domain-role binding,
- phase 5 refactors planner/runner composition so mixed domain participation can be carried end-to-end,
- phase 6 lowers local expert-parallel MoE using those roles,
- phase 7 adds the `CPU_CROSS_RANK` expert domain as a concrete execution target.

### Design conclusion

The existing TP-domain system is the correct foundation for CPU-offloaded expert TP, but the graph/runtime contract must become expert-aware.

The clean path is:

- keep `CPU_CROSS_RANK` as the physical execution substrate,
- add stage-role-aware domain binding,
- bind CPU-offloaded expert stages to a dedicated expert domain,
- and lower expert execution as a TP-sharded subgraph inside that domain.

---

## CLI And Orchestrator Implications

This refactor should be explicit about what changes in the user-facing orchestration model and what does not.

### What can stay as-is

The current top-level execution concepts are still the right ones:

- single-device,
- tensor parallel,
- pipeline parallel,
- named domains,
- topology tree.

MoE does **not** require a new top-level parallelism family alongside TP and PP.

What should **not** be inferred from this: the current implementation can already compose these concepts arbitrarily. Today the runner still chooses one primary graph-build path at a time (single-device, LOCAL TP, or LOCAL PP), and the topology-tree path still falls back to the standard runner path. So the top-level concepts can stay, but the planner/runner composition logic still needs substantial refactoring before mixed deployments like "local GPU TP plus global CPU expert domain" are executable.

### What is insufficient today

The current CLI and orchestration model is device- and layer-centric. It can express:

- which device a rank runs on,
- which devices belong to a TP domain,
- which layers belong to a PP stage.

It cannot express, within a single layer, that:

- router uses one domain,
- local experts use another domain,
- CPU-offloaded experts use a `CPU_CROSS_RANK` domain,
- combine returns to the continuation domain.

So the missing concept is not "MoE mode". The missing concept is **stage-role-to-domain binding**.

### Recommended user-facing model

Keep the current hierarchy of concepts:

- TP and PP define which devices and ranks can execute together,
- named domains define explicit reusable device groups,
- MoE adds a binding layer that maps stage roles to those domains.

That means MoE should extend the orchestration language, not replace it.

### Single-device implications

Single-device execution needs no conceptual CLI change.

In ordinary single-device MoE:

- all stage roles bind implicitly to the same device,
- `-d` keeps its current meaning,
- no new flags are required.

### Tensor-parallel implications

Simple TP flags such as:

- `-tp N`
- `--tp-devices`
- `--tp-weights`

should remain focused on the primary TP group for the layer's main execution path.

They should **not** be overloaded to describe router-domain, expert-local-domain, and CPU-expert-domain placement simultaneously.

### Pipeline-parallel implications

PP stages remain the right abstraction for layer ownership. MoE does not change that.

What changes is that a PP stage may reference auxiliary domains for expert execution inside the layers it owns. So PP stage syntax should keep defining layer ownership and primary stage domains, while expert-domain binding stays a separate concern.

### Named-domain implications

Named domains are the best existing CLI surface for advanced MoE placement.

Today they define:

- domain name,
- devices,
- optional weights,
- backend.

That is enough to define the domain itself, but not enough to say which MoE stage role should use it.

### Recommended new config concept

Add a binding layer rather than a new family of domain types.

Suggested concept:

- `DomainBindingDefinition`

It would map stage roles to domain names, for example:

- primary layer path -> `gpu_main`
- CPU MoE experts -> `cpu_expert`
- combine -> `gpu_main`

This could be surfaced through CLI, YAML, and eventually topology-tree annotations.

### Why this is better than adding a new top-level mode

Adding a separate "MoE orchestration mode" would force users to choose between overlapping abstractions.

The cleaner split is:

- TP/PP/topology decide **who can work together**,
- bindings decide **which stage role uses which group**.

### MultiDomainTPConfig implications

`MultiDomainTPConfig` is the runtime abstraction that most clearly needs to evolve.

Today it assumes:

- one GPU domain,
- one CPU domain,
- one `domainForLayer(layer_idx, is_attention)` selector.

For MoE-first-class orchestration, it should support:

- multiple named domains of the same broad type,
- lookup by stage role rather than only attention vs FFN,
- expert-specific domain selection without abusing the FFN bucket.

### Topology-tree implications

The topology tree does not need a new node type for MoE.

Existing node kinds remain sufficient:

- `PIPELINE_PARALLEL`
- `TENSOR_PARALLEL`
- `DEVICE`

If the topology tree becomes the preferred advanced orchestration surface, it should later gain annotations for stage-role domain bindings, not a new expert-parallel node type.

### Validation implications

Once bindings exist, validation needs to grow accordingly.

Examples:

- bound domains must exist,
- CPU expert bindings must target CPU-capable domains,
- no invalid mix of legacy coarse FFN routing and new stage-role bindings,
- router/combine bindings must be compatible with the primary layer device policy.

### Net recommendation

For MoE to be first-class in the orchestrator:

- keep the existing top-level orchestration concepts,
- do not add a new top-level PP/TP sibling for MoE,
- extend the domain system with stage-role bindings,
- generalize `MultiDomainTPConfig` beyond one GPU + one CPU bucket,
- and add CLI/YAML binding syntax for expert-related stage roles.

---

## Concrete CLI Proposal

This section turns the orchestration recommendation into a concrete proposed CLI and YAML surface.

Important: this is a **target-state** CLI, not a claim that the current runner can execute these commands today. The current implementation still selects one primary compute-graph path at a time and does not yet compose local GPU TP, PP ownership, and auxiliary expert domains in the way proposed below. So this section describes the intended UX that the planner/runner refactor should make real.

### 1. Extend domain definitions with scope

Keep `--define-domain` as the way users define reusable execution groups, but extend its format with a scope field.

#### Proposed format

`name=device1,device2,...[;scope=local|global][;weights=w1,w2,...][;backend=type]`

#### Meaning

- `scope=local`
  - the domain is instantiated independently inside each MPI rank
  - good for per-rank GPU TP groups such as two local MI50s attached to one socket

- `scope=global`
  - the domain spans participating MPI ranks
  - good for `CPU_CROSS_RANK` domains or future cross-rank expert domains

#### Examples

- `attn_gpu=rocm:0,rocm:1;scope=local;backend=rccl`
- `cpu_experts=cpu;scope=global;backend=mpi`

The key distinction is that `attn_gpu` means "two local GPUs inside each rank," while `cpu_experts` means "the rank-local CPU participant from every rank in the global CPU expert group."

### 2. Add explicit stage-role bindings

Add one new repeatable flag:

- `--bind-domain <spec>`

#### Proposed format

`selector.role=domain_name`

#### Selectors

- `default`
  - applies to every matching layer unless a more specific binding overrides it
- `layers=A-B`
  - applies only to a layer range
- `layer=N`
  - applies only to one layer

#### Proposed roles

- `attention`
- `dense_ffn`
- `moe.router`
- `moe.experts`
- `moe.shared_experts`
- `moe.combine`
- `gdn.mixer`

#### Examples

- `default.attention=attn_gpu`
- `default.moe.router=attn_gpu`
- `default.moe.experts=cpu_experts`
- `default.moe.combine=attn_gpu`
- `layers=40-79.moe.experts=cpu_experts`

This keeps the model simple:

- domains describe execution groups,
- bindings assign graph stage roles to those groups.

### 3. Proposed config API sketch

The CLI proposal should map directly onto a small set of new config types instead of being implemented as ad hoc parser-side strings.

#### `DomainScope`

```cpp
enum class DomainScope
{
  LOCAL,
  GLOBAL,
};
```

Add this to `DomainDefinition` so a domain can explicitly describe whether it is instantiated per rank or across participating ranks.

#### `StageDomainRole`

This should stay in the shared graph/runtime layer, but the orchestration config should reference the same enum values.

```cpp
enum class StageDomainRole
{
  Attention,
  DenseFFN,
  MoERouter,
  MoEExperts,
  MoESharedExperts,
  MoECombine,
  GatedDeltaNetMixer,
};
```

#### `DomainBindingSelector`

```cpp
enum class DomainBindingSelectorKind
{
  DEFAULT,
  SINGLE_LAYER,
  LAYER_RANGE,
};

struct DomainBindingSelector
{
  DomainBindingSelectorKind kind = DomainBindingSelectorKind::DEFAULT;
  int first_layer = -1;
  int last_layer = -1;
};
```

This replaces ambiguous stringly typed selector handling in the runtime core.

#### `DomainBindingDefinition`

```cpp
struct DomainBindingDefinition
{
  DomainBindingSelector selector;
  StageDomainRole role;
  std::string domain_name;

  static DomainBindingDefinition parse(const std::string &spec);
  static std::optional<DomainBindingDefinition> tryParse(const std::string &spec);
  std::vector<std::string> validate() const;
  std::string toString() const;
};
```

#### `OrchestrationConfig` additions

```cpp
struct OrchestrationConfig
{
  std::vector<DomainDefinition> domain_definitions;
  std::vector<PPStageDefinition> pp_stage_definitions;
  std::vector<DomainBindingDefinition> domain_bindings;

  bool usesNamedDomains() const;
  bool hasDomainBindings() const;
};
```

#### Parser and validator consequences

- `DomainDefinition::parse()` must accept `scope=local|global`.
- `OrchestrationConfigParser` adds repeatable `--bind-domain` parsing.
- YAML parsing adds a `domain_bindings:` list.
- `ConfigValidator` checks domain existence, selector overlap and precedence, and role/domain compatibility.

#### Binding precedence

The runtime should resolve bindings with a simple deterministic rule:

1. `layer=N` overrides `layers=A-B`
2. `layers=A-B` overrides `default`
3. later bindings at the same specificity override earlier ones

That keeps CLI behavior predictable and matches how users normally expect layered overrides to work.

### 4. Optional YAML shape

The same information should be expressible in YAML without inventing a separate MoE config family.

```yaml
domains:
  - name: attn_gpu
    devices: [rocm:0, rocm:1]
    scope: local
    backend: rccl

  - name: cpu_experts
    devices: [cpu]
    scope: global
    backend: mpi

pp_stages:
  - stage_id: 0
    domain: attn_gpu
    first_layer: 0
    last_layer: 59

domain_bindings:
  - selector: default
    role: attention
    domain: attn_gpu

  - selector: default
    role: moe.router
    domain: attn_gpu

  - selector: default
    role: moe.experts
    domain: cpu_experts

  - selector: default
    role: moe.combine
    domain: attn_gpu
```

### 5. Full YAML equivalent for the same machine

For the exact two-socket, two-rank, four-MI50 deployment, the YAML should be able to express the same placement and bindings without shell quoting.

```yaml
model_path: models/Qwen3-MoE.gguf
prompt: "Summarize the attached report"
n_predict: 128

domains:
  - name: attn_gpu
    devices: [rocm:0, rocm:1]
    scope: local
    backend: rccl

  - name: cpu_experts
    devices: [cpu]
    scope: global
    backend: mpi

pp_stages:
  - stage_id: 0
    domain: attn_gpu
    first_layer: 0
    last_layer: 59

domain_bindings:
  - selector: default
    role: attention
    domain: attn_gpu

  - selector: default
    role: moe.router
    domain: attn_gpu

  - selector: default
    role: moe.experts
    domain: cpu_experts

  - selector: default
    role: moe.combine
    domain: attn_gpu
```

Representative launch shape:

```bash
mpirun -np 2 \
  --map-by ppr:1:socket \
  --bind-to socket \
  ./build_v2_release/llaminar2 --config qwen3_moe_mi50_cpu_tp.yaml
```

### 6. Concrete user example: 4x MI50 attention TP plus 2-rank CPU expert TP

Assume the physical machine looks like this:

- 2 CPU sockets,
- 2 MPI ranks, one rank pinned to each socket,
- 4 AMD MI50 GPUs total,
- each rank owns the 2 GPUs local to its socket,
- MoE router and combine stay on the GPU attention path,
- MoE experts execute in a global CPU TP domain spanning the 2 ranks.

#### Target-state launch command

```bash
mpirun -np 2 \
  --map-by ppr:1:socket \
  --bind-to socket \
  ./build_v2_release/llaminar2 \
    -m models/Qwen3-MoE.gguf \
    -p "Summarize the attached report" \
    -n 128 \
    --define-domain "attn_gpu=rocm:0,rocm:1;scope=local;backend=rccl" \
    --define-domain "cpu_experts=cpu;scope=global;backend=mpi" \
    --pp-stage "0=attn_gpu:0-59" \
    --bind-domain "default.attention=attn_gpu" \
    --bind-domain "default.moe.router=attn_gpu" \
    --bind-domain "default.moe.experts=cpu_experts" \
    --bind-domain "default.moe.combine=attn_gpu"
```

#### Interpretation

- `attn_gpu=rocm:0,rocm:1;scope=local;backend=rccl`
  - each rank forms a local 2-way GPU TP group for the main attention path

- `cpu_experts=cpu;scope=global;backend=mpi`
  - the CPU expert domain spans both ranks, one socket-local CPU participant per rank

- `--pp-stage "0=attn_gpu:0-59"`
  - the model is not pipeline-partitioned; all layers belong to a single stage whose primary path is the GPU domain

- `default.moe.router=attn_gpu`
  - routing happens on the GPU-side continuation path

- `default.moe.experts=cpu_experts`
  - expert FFN execution is offloaded into the cross-rank CPU TP domain

- `default.moe.combine=attn_gpu`
  - expert outputs are reduced/scattered back to the GPU continuation path before residual write-back

### 7. Why this launch shape is the right user model

This command exposes the physical layout without forcing the user to learn new orchestration families:

- GPU attention TP is still just a domain,
- CPU expert TP is still just a domain,
- MoE routing is expressed by stage-role bindings,
- PP stays responsible only for layer ownership.

That is exactly the separation the runtime should implement internally as well.

---

## Qwen Family Mapping

The system should explicitly model the following family relationships:

### Qwen2

- Standard GQA attention
- Dense SwiGLU FFN
- KV cache state

### Qwen3 dense

- Same block structure as Qwen2
- Optional per-head Q/K normalization before RoPE

### Qwen3-MoE

- Standard attention path
- Routed MoE FFN in place of dense FFN
- Likely top-k routed experts with no new sequence-state abstraction required

### Qwen3.5 GatedDeltaNet MoE

- Hybrid per-layer sequence mixer layout
- Some layers use Gated DeltaNet, some use attention
- All or most layers use routed MoE FFN
- Shared experts are part of the architecture and must be represented explicitly
- Requires sequence-state abstraction because GDN layers are not KV-cache layers

This is why the refactor must generalize both the **mixer axis** and the **FFN axis**.

---

## Implementation Phases

## Phase 1: Introduce typed architecture metadata

### Deliverables

- Add `ArchitectureFamily`, `LayerSpec`, `MoEConfig`, and hybrid-mixer metadata to model config surfaces.
- Extend loader/model-context APIs to expose these values.
- Keep old dense fields during transition.

### Outcome

The runtime can tell, structurally, whether a layer is dense, MoE, attention-based, or DeltaNet-based.

## Phase 2: Generalize graph types, buffers, and domain policy

### Deliverables

- Refactor `GraphTypes.h` into composable block-specific weight and buffer structs.
- Add execution-domain policy data so MoE layers can express router/expert/combine placement intent.
- Preserve dense Qwen execution through adapters or temporary compatibility constructors.

### Outcome

Graph building no longer assumes every layer is dense FFN, and the config contract can express sub-layer domain intent.

## Phase 3: Decouple orchestrator from `Qwen2Graph`

### Deliverables

- Convert `DeviceGraphOrchestrator` to hold `IGraphBuilder`.
- Move any remaining `Qwen2Graph`-specific helpers behind builder or schema APIs.
- Update factory creation flow to instantiate graph builders by architecture family.

### Outcome

Execution becomes graph-builder generic in reality, not only in comments.

## Phase 4: Introduce MoE block builder, stable stage contracts, and stage-domain binding

### Deliverables

- Replace placeholder MoE stage params with fixed-shape tensor-contract stages.
- Introduce `MoEBlockBuilder`.
- Add MoE buffer allocation/resolution support.
- Add stage-domain-role binding so MoE shell stages can target different execution domains within one layer.

### Outcome

Single-device routed MoE graphs can execute without builder-level control-flow explosion, and the graph contract is ready for heterogeneous expert placement.

## Phase 5: Refactor planner/runner composition for mixed execution domains

### Why this phase needs to exist explicitly

The proposed MoE CLI/domain model is not implementable by parser changes alone.

Today the runtime still chooses one primary execution path at graph-build time:

- single-device,
- LOCAL TP,
- or LOCAL PP.

That is sufficient for dense deployments, but it is not sufficient for MoE deployments where one rank may need to participate in:

- a primary GPU attention domain,
- a separate expert domain,
- pipeline ownership for some layer range,
- and possibly cross-rank CPU expert execution.

So before expert-parallel lowering becomes real, the planner and runner need to carry mixed domain participation as a first-class concept.

### Deliverables

- Extend `RankExecutionPlan` so it can describe more than one active execution domain per rank, with a distinction between:
  - primary layer-execution path,
  - auxiliary expert domains,
  - and stage-role bindings.
- Extend `ExecutionPlanBuilder` so named domains plus domain bindings are resolved into per-rank participation metadata rather than being left as loose config strings.
- Refactor `OrchestrationRunner` so it no longer treats LOCAL TP, LOCAL PP, and the standard path as mutually exclusive graph-build categories when the plan requires mixed participation.
- Define a composition model where the runner can instantiate:
  - one primary inference path,
  - zero or more auxiliary domain contexts,
  - and the stage-role/domain binding data needed by the graph/orchestrator layer.
- Keep the external runner categories stable. This phase should not add a new user-facing top-level "MoE runner mode".
- Keep topology-tree support aligned with the same composition model instead of creating a separate orchestration stack for MoE.

### Required sequencing

- This phase comes **after** phase 4, because the runner needs stable stage roles and domain-binding semantics to carry forward.
- This phase comes **before** expert-parallel lowering, because `GraphResolver` and MoE execution stages need a runtime plan that can already describe mixed domain participation.
- This phase should land before CPU-offloaded expert TP, because cross-rank CPU expert execution depends on the same primary-vs-auxiliary domain split.
- This phase should preserve dense behavior at every step. A good migration strategy is:
  - first extend plan/config data structures,
  - then make the runner able to retain auxiliary domain contexts without using them,
  - then switch MoE lowering to consume those new plan surfaces.

### Validation focus

- Unit-test plan building for configurations that mix:
  - a primary GPU domain,
  - one auxiliary expert domain,
  - and PP layer ownership.
- Add runner-level tests that verify mixed-domain plans do not collapse back to the legacy mutually exclusive path selection.
- Preserve existing dense single-device, LOCAL TP, and LOCAL PP tests unchanged while the composition model lands.

### Outcome

The runtime can carry mixed participation state end-to-end, which makes the target-state CLI feasible and gives later MoE lowering phases a real execution substrate instead of only a schema-level design.

## Phase 6: Implement local expert-parallel lowering

### Deliverables

- Implement `TPMode::ExpertParallel` in `GraphResolver`.
- Connect `IWorkDistributor` expert routing to graph execution.
- Add local multi-device expert placement first.
- Bind local expert stages through the new stage-domain model rather than ad hoc FFN routing.

### Outcome

The system can represent local MoE graph execution as an explicit parallelism mode with explicit domain bindings.

## Phase 7: Add CPU-offloaded expert TP

### Deliverables

- Introduce one within-node `CPU_CROSS_RANK` expert domain per physical node.
- Add dispatch/return stages for packed expert buffers entering and leaving that domain.
- Implement TP-sharded expert execution within the CPU expert domain.
- Carry any additional expert-domain participation metadata through planning and runtime config.

### Outcome

CPU-offloaded experts become a native part of the MoE graph model instead of an orchestration-only escape hatch.

## Phase 8: Add sequence-state abstraction and DeltaNet block integration

### Deliverables

- Add `ISequenceState`, `KVSequenceState`, `DeltaNetSequenceState`.
- Integrate `DeltaNetBlockBuilder`.
- Allow per-layer mixer dispatch from `LayerSpec`.

### Outcome

Qwen3.5 hybrid MoE becomes architecturally representable.

## Phase 9: Capture-aware optimization pass

### Deliverables

- Make router/top-k/pack/combine stages capture-friendly where practical.
- Update capturability declarations and segmented replay policy.
- Add dedicated capture tests for dense+MoE mixed decode.

### Outcome

MoE decode participates in the existing graph replay system without compromising correctness.

---

## File-by-File Refactor Plan

This section lists the concrete files that should be created or changed, with the reason for each change.

## Files To Create

### `src/v2/models/LayerSpec.h`

Create a new architecture-neutral header for:

- `ArchitectureFamily`
- `SequenceMixerKind`
- `FeedForwardKind`
- `SequenceStateKind`
- `LayerSpec`
- `MoEConfig`
- hybrid per-layer layout helpers

This file becomes the typed architecture contract shared by model loading, graph config building, and graph building.

### `src/v2/models/sequence_state/ISequenceState.h`

Create a generic interface for sequence-state ownership and layer access.

### `src/v2/models/sequence_state/KVSequenceState.h`

Adapter around existing KV-cache behavior for attention layers.

### `src/v2/models/sequence_state/DeltaNetSequenceState.h`

State holder for Gated DeltaNet recurrent state.

### `src/v2/execution/local_execution/graph/StageDomainRole.h`

Shared definitions for stage-domain roles and related binding helpers used by schema, config, and resolver layers.

### `src/v2/models/blocks/MoEBlockBuilder.h`

Block builder for routed MoE FFN graphs.

### `src/v2/models/blocks/MoEBlockBuilder.cpp`

Implementation of the MoE FFN graph builder.

### `src/v2/models/blocks/DenseFFNBlockBuilder.h`

Extract dense FFN graph building out of `Qwen2Graph` so dense and MoE FFN paths are sibling concepts.

### `src/v2/models/blocks/DenseFFNBlockBuilder.cpp`

Implementation of the dense FFN block builder.

### `src/v2/models/blocks/AttentionBlockBuilder.h`

Optional extraction target for the current attention block logic.

### `src/v2/models/blocks/AttentionBlockBuilder.cpp`

Implementation of the dense/Qwen attention block builder.

### `src/v2/models/blocks/DeltaNetBlockBuilder.h`

Builder for Gated DeltaNet sequence-mixer graphs.

### `src/v2/models/blocks/DeltaNetBlockBuilder.cpp`

Implementation for hybrid Qwen3.5 sequence mixer support.

### `src/v2/models/qwen/QwenMoEGraph.h`

Graph builder composition root for Qwen3-MoE if the team prefers a family-specific builder wrapper rather than one very large generalized Qwen builder.

### `src/v2/models/qwen/QwenMoEGraph.cpp`

Implementation of the family-specific builder wrapper around shared block builders.

### `src/v2/models/qwen3_5/Qwen35HybridSchema.h`

Declarative schema for hybrid GDN/attention plus MoE layers.

### `src/v2/models/qwen3_5/Qwen35HybridGraph.h`

Optional graph builder wrapper for the Qwen3.5 family.

### `src/v2/models/qwen3_5/Qwen35HybridGraph.cpp`

Implementation of the Qwen3.5 hybrid graph builder.

### `src/v2/execution/compute_stages/stages/MoEPackStage.h`

New stable tensor-contract stage for packing routed tokens.

### `src/v2/execution/compute_stages/stages/MoEPackStage.cpp`

Implementation of token packing.

### `src/v2/execution/compute_stages/stages/MoETopKStage.h`

New stage for router top-k selection and weight extraction.

### `src/v2/execution/compute_stages/stages/MoETopKStage.cpp`

Implementation of top-k routing selection.

### `src/v2/execution/compute_stages/stages/MoEScatterCombineStage.h`

New stage for combining expert outputs back into token order.

### `src/v2/execution/compute_stages/stages/MoEScatterCombineStage.cpp`

Implementation of final MoE output combine.

### `src/v2/execution/compute_stages/stages/MoEDomainDispatchStage.h`

Explicit stage for dispatching packed expert tokens into a target expert domain.

### `src/v2/execution/compute_stages/stages/MoEDomainDispatchStage.cpp`

Implementation of domain-bound dispatch for CPU expert TP.

### `src/v2/execution/compute_stages/stages/MoEExpertTPStage.h`

Expert execution stage whose internal FFN projections are TP-sharded within a bound domain.

### `src/v2/execution/compute_stages/stages/MoEExpertTPStage.cpp`

Implementation of TP-sharded expert execution for CPU-offloaded experts.

### `src/v2/execution/compute_stages/stages/MoEDomainReturnStage.h`

Explicit stage for returning packed expert outputs from an expert domain back to the combine path.

### `src/v2/execution/compute_stages/stages/MoEDomainReturnStage.cpp`

Implementation of domain-return behavior for offloaded expert outputs.

### `tests/v2/unit/models/Test__LayerSpec.cpp`

Unit tests for architecture and layer metadata types.

### `tests/v2/unit/execution/Test__MoEGraphResolver.cpp`

Unit tests for MoE resolver lowering and expert-parallel insertion.

### `tests/v2/integration/Test__Qwen3MoEGraph.cpp`

Integration coverage for single-device Qwen3-MoE graph construction/execution.

### `tests/v2/integration/Test__Qwen35HybridMoEGraph.cpp`

Integration coverage for hybrid attention/GDN plus MoE graph composition.

## Files To Change

### `src/v2/models/GraphTypes.h`

Refactor this file from dense-transformer types into composable graph contract types.

#### Required changes

- Extend `GraphConfig` with typed architecture metadata.
- Replace or augment dense-only `LayerWeights`.
- Replace or augment dense-only `ActivationBuffers`.
- Add block-specific sub-weights and sub-buffers.
- Add helper methods for querying layer capabilities.
- Add execution-domain policy fields or references needed for stage-role-aware lowering.

#### Migration note

Keep dense aliases or compatibility constructors temporarily so Qwen2/Qwen3 dense paths do not break all at once.

### `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h`

Break the hard dependency on `Qwen2Graph`.

#### Required changes

- Replace `#include` and member types from `Qwen2Graph` to `IGraphBuilder`.
- Change constructor signatures to accept generic builders.
- Remove `graphBuilder()` accessors returning `Qwen2Graph*`.
- Update cache comments and state comments so they are not Qwen2-specific.
- Replace any hidden assumptions that a layer cache always consists of attention + FFN graphs.
- Stop treating domain lookup as a two-bucket attention-vs-FFN decision.

### `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp`

Make execution logic builder-neutral and state-kind aware.

#### Required changes

- Update graph build calls to use generic block/layer builder capabilities.
- Route sequence state through `ISequenceState` instead of assuming KV cache everywhere.
- Adjust decode graph cache logic to support MoE and hybrid mixer layers.
- Update capture policy construction to recognize MoE non-capturable/capturable segments.
- Replace the current layer-level attention/FFN domain lookup path with stage-role-aware domain lookup for MoE expert execution.

### `src/v2/config/OrchestrationConfig.h`

Extend the user-facing orchestration contract so MoE can bind stage roles to named domains without introducing a new top-level orchestration family.

#### Required changes

- Add a binding-level config concept for stage-role-to-domain mapping.
- Add `DomainScope` to `DomainDefinition`.
- Add `DomainBindingSelector` and `DomainBindingDefinition`.
- Add `domain_bindings` storage on `OrchestrationConfig`.
- Add helper APIs for binding presence and effective-binding lookup.
- Keep existing `DomainDefinition` semantics focused on describing device groups.
- Avoid adding a separate top-level "MoE mode" enum.

### `src/v2/config/OrchestrationConfigParser.cpp`

Parse any new CLI or YAML bindings for MoE stage-role domain assignment.

#### Required changes

- Add parsing for domain-binding specifications.
- Extend `--define-domain` parsing to accept `scope=local|global`.
- Add YAML parsing for `domain_bindings:`.
- Add CLI help text and examples for `--bind-domain`.
- Preserve current single-device, TP, PP, named-domain, and topology-tree behavior.

### `src/v2/config/ConfigValidator.cpp`

Validate MoE domain bindings alongside existing device-selection rules.

#### Required changes

- Ensure referenced domains exist.
- Ensure binding selectors parse cleanly and apply deterministically.
- Ensure overlapping bindings resolve by explicit precedence rules.
- Ensure CPU expert bindings resolve to CPU-capable domains.
- Ensure global expert bindings are only used with globally meaningful domains.
- Add conflict rules between legacy coarse FFN routing and new stage-role bindings where needed.

### `src/v2/config/TPDomain.h`

Extend TP-domain routing beyond the current attention-vs-FFN split.

#### Required changes

- Add a stage-role-oriented query path, not only `domainForLayer(layer_idx, is_attention)`.
- Allow explicit lookup for CPU expert TP domains.
- Preserve the current `CPU_CROSS_RANK` model as the physical execution substrate.

### `src/v2/config/TPDomainConfig.h`

Extend configuration-level domains to support stage-role-aware expert execution.

#### Required changes

- Add metadata for domain purpose or supported stage roles.
- Keep the existing CPU-domain validation semantics for within-node cross-rank TP.

### `src/v2/execution/factory/InferenceRunnerFactory.cpp`

Factory should instantiate builders by architecture family, not by `qwen2`/`qwen3` string special cases inside the same dense builder.

#### Required changes

- Select graph builder implementation using typed architecture metadata.
- Select graph-config builder implementation using typed architecture metadata.
- Stop directly depending on `Qwen2Graph` as the sole builder implementation.

### `src/v2/execution/runner/OrchestrationRunner.cpp`

Keep current top-level runner selection logic, but make sure MoE-specific domain bindings are carried into the graph/orchestrator layer.

#### Required changes

- Do not add a new top-level runner category just for MoE.
- Replace the current mutually exclusive build-path assumption with a composition model that can keep one primary execution path while also provisioning auxiliary expert domains.
- Build and retain auxiliary domain contexts needed for MoE expert execution instead of discarding everything outside the selected primary path.
- Pass stage-role-aware domain configuration through to the created inference runner.
- Keep the single-device, LOCAL TP, and LOCAL PP user model stable while changing their internal composition semantics.

### `src/v2/execution/runner/OrchestrationRunnerFactory.cpp`

Keep the current factory surface, but ensure advanced MoE orchestration remains compatible with named domains and topology trees.

#### Required changes

- Preserve current CLI entry points.
- Ensure explicit-domain and topology-tree creation paths can carry MoE domain bindings once implemented.
- Stop treating topology-tree handling as a logging/fallback side path once the mixed-domain composition model lands.
- Create runners from a plan shape that can express both primary and auxiliary domain participation.

### `src/v2/models/IGraphConfigBuilder.h`

Broaden the config builder contract to support layer specs, MoE metadata, and hybrid sequence layouts.

#### Required changes

- Add or expose execution-domain policy information so graph builders can request stage-role-aware lowering.

### `src/v2/models/qwen/Qwen2GraphConfigBuilder.h`

Either narrow this file back down to dense Qwen responsibility or rename/generalize it.

#### Preferred direction

- Keep this builder focused on dense Qwen2/Qwen3.
- Introduce separate builders for Qwen3-MoE and Qwen3.5 hybrid MoE.

### `src/v2/models/qwen/Qwen2GraphConfigBuilder.cpp`

Remove the hidden architectural overloading where `qwen3` is just `qwen2` plus optional fields if that structure starts leaking MoE concerns.

#### Required changes

- Keep dense-family configuration here.
- Move MoE and hybrid-specific metadata loading into dedicated builders or helper modules.

### `src/v2/models/qwen/Qwen2Graph.h`

Shrink this file from universal Qwen runtime graph builder to dense-Qwen graph composition root.

#### Required changes

- Remove responsibility for future MoE branching.
- Delegate dense attention and dense FFN assembly to block builders if extraction is adopted.
- Keep only dense-family graph behavior.

### `src/v2/models/qwen/Qwen2Graph.cpp`

Extract the current dense `buildFFNGraph()` implementation into `DenseFFNBlockBuilder` or equivalent, leaving `Qwen2Graph` to compose blocks.

#### Specific code changes

- Keep existing Qwen3 Q/K norm path in the attention block.
- Stop treating FFN as permanently dense.
- Convert layer assembly to branch on `LayerSpec` when the generalized path lands.

### `src/v2/execution/local_execution/graph/GraphSchema.h`

Make MoE and hybrid sequence-mixer constructs first-class in the declarative schema.

#### Required changes

- Add buffer kinds for routing, packing, expert outputs, and combine outputs.
- Add stage kinds for top-k selection, token pack, dispatch, shared expert execution, and combine.
- Add state kinds for DeltaNet sequence state.
- Add domain-role annotations so stages inside one MoE layer can bind to different TP domains.

### `src/v2/execution/local_execution/graph/GraphResolver.cpp`

Implement actual lowering for `ExpertParallel` and MoE stages.

#### Required changes

- Stop returning TODO for `ExpertParallel`.
- Lower MoE schema nodes into concrete stage wiring.
- Insert routing-related collectives when expert ownership is distributed.
- Distinguish local-only expert execution from cross-device or cross-rank execution.
- Bind CPU-offloaded expert execution to a dedicated CPU expert domain rather than treating it as the generic FFN domain.

### `src/v2/execution/compute_stages/stages/MoEStages.h`

Refactor the current placeholder stages to a real stable contract.

#### Required changes

- Remove host-vector-centric execution params where possible.
- Add tensor-based routing metadata interfaces.
- Separate router, top-k selection, pack, expert execute, shared expert execute, and combine into explicit stage classes or stage families.

### `src/v2/execution/compute_stages/stages/MoEStages.cpp`

Replace placeholder logic with real tensor-contract execution.

#### Required changes

- Remove CPU-only assumptions from the stage contract.
- Keep CPU reference implementations first if needed.
- Make buffer requirements and dump info reflect the stable MoE shell.

### `src/v2/execution/mpi_orchestration/IExecutionPlanBuilder.h`

Extend `ModelConfig` so plan building knows about MoE and hybrid layout.

#### Required changes

- Add MoE fields.
- Add per-layer layout or a reference to layer specs.
- Add state-related dimensions required for DeltaNet layers.
- Add any domain-policy metadata needed to describe CPU expert-domain participation.
- Add explicit plan-time types for mixed domain participation rather than assuming one active domain family per rank.

### `src/v2/execution/mpi_orchestration/ExecutionPlanBuilder.cpp`

Carry named-domain and binding information into per-rank execution plans without introducing a new top-level orchestration family.

#### Required changes

- Resolve stage-role domain bindings into per-rank domain participation metadata.
- Preserve current single-device, simple TP, and named-domain planning paths.
- Introduce explicit plan-time separation between primary layer execution domains and auxiliary expert domains.
- Emit enough plan data for the runner to instantiate mixed domain contexts without re-deriving orchestration intent ad hoc.

### `src/v2/execution/mpi_orchestration/RankExecutionPlan.h`

Add expert-parallel ownership data if plan-time routing/domain ownership needs to be carried per rank.

#### Required changes

- Carry expert-domain participation separately from dense FFN domain participation if the same rank participates in both.
- Add ownership metadata for CPU-offloaded expert shards where needed.
- Add a stable representation of stage-role bindings or their resolved per-rank projection so the runner/orchestrator layer does not need to reinterpret raw config.

### `src/v2/execution/mpi_orchestration/IWorkDistributor.h`

Broaden the expert-routing contract if needed for shared experts and stable packed-token exchange.

#### Required changes

- Distinguish routed experts from shared experts.
- Add APIs needed for pack/scatter planning.
- Add APIs needed to partition routed tokens by expert domain, not only by expert id.

### `src/v2/execution/mpi_orchestration/WorkDistributor.cpp`

Implement whatever additional routing helpers are needed for shared expert placement and packed-token execution.

#### Required changes

- Add expert-domain-aware routing helpers for CPU-offloaded expert dispatch.

### `src/v2/execution/parallelism_tree/ParallelismTree.h`

Keep the current PP/TP/DEVICE node taxonomy, but allow future MoE-specific annotations if the topology tree becomes the advanced orchestration surface.

#### Required changes

- No new node type required.
- Optionally add metadata hooks or annotations for stage-role domain binding.
- Align any new annotations with the same primary-domain plus auxiliary-domain composition model used by `ExecutionPlanBuilder` and `OrchestrationRunner`.

### `src/v2/loaders/IModelLoader.h`

Extend loader-facing architecture metadata APIs so MoE and hybrid layout are exposed cleanly.

### `src/v2/interfaces/IModelContext.h`

Add getters for:

- architecture family,
- per-layer block kinds,
- MoE counts/top-k/shared experts,
- DeltaNet-specific dimensions and state metadata.

### `src/v2/loaders/ModelLoader.cpp`

Parse and expose the new model metadata from GGUF or model config.

### `src/v2/loaders/WeightPlacementMap.h`

Keep the existing MoE helper direction, but align its naming and structure with the new typed architecture metadata.

#### Required changes

- Ensure shared/local expert placement maps align with the final `MoEWeights` structure.
- Add helper coverage for hybrid MoE families if weight naming differs.

### `src/v2/loaders/WeightPlacementMap.cpp`

Update placement logic to use typed layer/expert metadata rather than name patterns alone where possible.

### `src/v2/execution/local_execution/graph/DeviceGraphCaptureController.cpp`

Teach segmented replay policy about MoE shell stages and new sequence-mixer stage kinds.

#### Required changes

- Allow router/pack/combine stages to remain manual at first.
- Recognize stable capturable MoE stages later.

### `src/v2/backends/IGPUGraphCapture.h`

Likely no semantic API rewrite is needed, but update comments/contracts if capturable stage expectations become more explicit.

### `src/v2/execution/compute_stages/ComputeStages.h`

Register new MoE and DeltaNet stage factories/types.

### `src/v2/execution/compute_stages/ComputeStageFactory.*`

Add factories for new routing/pack/combine/shared-expert stages.

### `CMakeLists.txt` and `src/v2/CMakeLists.txt`

Wire in all new source files, tests, and any new block-builder or sequence-state modules.

### `docs/v2/projects/2026-03/QWEN3NEXT_GDN_PROJECT_PLAN.md`

Update this doc after landing the architecture-level refactor so the GDN plan references the new builder/state abstractions rather than the old dense-Qwen assumptions.

### `docs/v2/projects/2026-03/QWEN3_5_RESEARCH.md`

Update terminology and implementation cross-references once the concrete hybrid graph path is agreed.

---

## Validation Plan

Validation must happen in layers, not only at end-to-end inference.

### 1. Contract-level unit tests

- `LayerSpec` and architecture metadata parsing
- graph type compatibility/adapters
- resolver lowering for `ExpertParallel`
- MoE buffer contract validation
- stage-domain policy parsing and binding

### 2. Stage-level tests

- router logits correctness
- top-k selection correctness
- token pack/scatter correctness
- shared expert execution correctness
- combine correctness

### 3. Builder-level tests

- Qwen2 dense graph still builds identically
- Qwen3 dense graph still inserts Q/K norm correctly
- Qwen3-MoE graph builds a fixed-shape MoE shell
- Qwen3.5 hybrid graph selects mixer type per layer

### 4. Execution tests

- single-device eager execution
- local multi-device expert placement
- within-node CPU-offloaded expert TP
- segmented decode replay with MoE manual segments

### 5. Future parity tests

Once reference implementations are available:

- Qwen3-MoE parity against PyTorch or trusted framework outputs
- Qwen3.5 hybrid MoE parity for both attention layers and GDN layers

---

## Risks and Open Questions

### Risk 1: Over-generalizing too early

If the first refactor tries to land dense, MoE, hybrid attention, and optimized expert-parallel routing at once, it will likely stall.

**Mitigation**: land the typed architecture contract first, then stage-domain-aware MoE shell, then planner/runner composition refactor, then local expert-parallel lowering, then CPU expert TP, then hybrid sequence-state integration.

### Risk 2: Capture regressions

If MoE is implemented with host-side dynamic control flow, decode capture behavior will regress.

**Mitigation**: enforce stable graph-shell design from the start.

### Risk 3: Dense-path regression

Because `GraphTypes.h` is widely used, refactoring it can destabilize existing Qwen2/Qwen3 execution.

**Mitigation**: preserve compatibility adapters during migration and keep dense-family builders unchanged until the new contract is proven.

### Risk 4: Ambiguous family naming

The repository currently has both Qwen3.5 naming and Qwen3-Next/GDN planning terminology.

**Mitigation**: standardize one internal naming scheme in code. Recommended: use explicit architecture-family enum values rather than relying on marketing names.

### Open Question 1

Should Qwen3-MoE and Qwen3.5 hybrid MoE be expressed as separate graph-builder classes or as one generalized Qwen graph family with pluggable block builders?

**Recommendation**: use shared block builders plus thin family-specific composition roots. Avoid one giant "everything Qwen" builder.

### Open Question 2

Should shared experts be represented as a special expert with a flag, or as a separate explicit block in `MoEWeights` and `MoEBuffers`?

**Recommendation**: model shared experts explicitly. Their execution and placement semantics differ from routed experts.

### Open Question 3

Should stage-domain binding live in the declarative schema, in the resolved graph only, or partially in both?

**Recommendation**: keep role intent in the schema/config layer and bind to concrete domains in the resolved graph. That preserves portability while still letting lowering make topology-aware decisions.

### Open Question 3

Should expert-parallel lowering be introduced before or after local single-device MoE correctness?

**Recommendation**: after. Single-device and local multi-device MoE correctness should come first.

---

## Recommended Order of Attack

If this plan is executed incrementally, the recommended order is:

1. Add typed architecture metadata and layer specs.
2. Refactor `GraphTypes.h` to support block-specific weights and buffers.
3. Decouple `DeviceGraphOrchestrator` from `Qwen2Graph`.
4. Extract dense FFN and attention block builders.
5. Refactor execution planning and runner composition so mixed primary and auxiliary domains can be carried end-to-end.
6. Introduce a fixed-shape `MoEBlockBuilder` and productionize MoE stages for single-device execution.
7. Implement `ExpertParallel` lowering and local multi-device expert placement.
8. Add CPU-offloaded expert TP on top of the mixed-domain execution substrate.
9. Add sequence-state abstraction and integrate `DeltaNetBlockBuilder`.
10. Revisit capturability and segmented replay once the MoE shell is stable.

That sequence gets Qwen3-MoE support onto a clean path quickly, while also building the right architectural base for Qwen3.5 GatedDeltaNet MoE rather than forcing another dense-Qwen special case.