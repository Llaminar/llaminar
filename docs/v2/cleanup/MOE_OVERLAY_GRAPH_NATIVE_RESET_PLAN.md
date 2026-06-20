# MoE Overlay Graph-Native Reset Plan

**Date:** 2026-05-15  
**Status:** Proposed big-bang cleanup plan  
**Scope:** Qwen3.5 MoE expert overlay execution, overlay rank lifecycle, sparse dispatch/return transport, tiered expert-parallel participants, CPU expert participants, dense TP boundary rules, and removal of the current shadow-domain overlay runtime.

## Executive Summary

The current MoE expert overlay system is not a bridge worth stabilizing. It violates Llaminar's core execution contract and is failing in practice. The migration should therefore be a big-bang production reset: remove the domain-runtime path from active overlay execution and replace it with graph-native participant-local stages plus explicit sparse collectives.

The target architecture is simple:

```text
MoERoutingStage
MoESparseDispatchStage
MoELocalExpertStage
MoESparseReturnReduceStage
```

Every participating device/rank runs a graph with the same logical MoE rendezvous points. A participant with no routed rows publishes a no-op payload and waits at the collective. No stage owns peer devices, peer scratch, peer streams, or a hidden domain scheduler.

The routed-expert overlay is expert parallel, not tensor parallel. A tier with two devices means two expert-owning participants with disjoint full-expert residency, not a LocalTP group that shards each expert GEMM. True dense TP remains available for dense model weights or shared experts when the continuation domain actually has a sharded dense layout, but it is a separate boundary from routed expert dispatch.

This plan intentionally avoids the overengineered parts of the earlier sparse-collectives proposal. In particular, sparse partial buffers do **not** need to be added to `BufferArena` for the first reset. The first implementation should use preallocated MoE collective workspaces owned by the graph participant or sparse collective context. `BufferArena` remains responsible for dense activation buffers that already exist in the normal graph path.

## Why The Current System Must Be Replaced

The broken shape is visible in the current source:

- `Qwen35MoEGraph` lowers routed tiers to `MoEOverlayDomainRuntimeStage` and passes a `MoEOverlayDomainWorkRequest` instead of graph-native local stage params.
- `MoEOverlayDomainRuntimeStage` calls `IOverlayDomainRuntime::submit()` from inside graph execution.
- `MoEOverlayDomainRuntime` can instantiate and execute other stages or dispatch to a side endpoint from inside `submit()`.
- `MoEExpertOverlayLocalTPStage` owns a vector of prepared participants, prepared partial views, domain descriptors, and a domain-scoped `ILocalTPContext`.
- `MoEExpertOverlayLocalTPExecutor` loops over every participant in a LocalTP expert domain, binds peer streams, launches peer-device expert kernels, runs hidden allreduces, and scatters the result from one participant's stage.
- `MoEOverlayCPUFallbackParticipantRunner` is a sidecar receive pump, not a normal graph participant.

That creates a second execution model inside Llaminar. The continuation device can try to run a shadow LocalTP domain. CPU fallback can progress through an endpoint protocol adjacent to the graph. This breaks the two important assumptions:

1. A graph and its stages execute on one device.
2. Multiple devices coordinate only at explicit collective stages that every participant enters symmetrically.

## Design Principles

1. **Participant-local compute only.** A compute stage may launch kernels only on its graph participant's local device.
2. **Collectives are graph stages.** Dispatch and return/reduce are explicit stages with ordered keys.
3. **No side schedulers in stage params.** Graph-native stage params must not contain `IOverlayDomainRuntime*`, `MoEGraphRoleRunner*`, or domain work requests.
4. **No peer participant vectors in local expert stages.** A local expert stage gets only the current participant's weights, scratch, and compact input payload.
5. **No-op is first class.** Participants with no work still enter the same collective key with zero rows and zero entries.
6. **Host-first is acceptable for the reset.** Correct graph shape matters more than immediate GPU-native sparse packing. Optimize after the execution model is sane.
7. **Arena changes are earned, not assumed.** Use `BufferArena` for dense graph activations. Use MoE-owned collective workspaces for dynamic sparse payloads until a general sparse buffer abstraction is proven necessary.
8. **Routed experts are whole-expert owned by default.** A warm tier with two GPUs is two expert-parallel participants with disjoint expert-id sets, not a dense LocalTP group.
9. **Dense TP is a separate boundary.** LocalTP/GlobalTP may shard attention, dense FFNs, shared experts, or LM head inside an eligible continuation domain, but routed expert ownership must not inherit TP assumptions by accident.

## Target Runtime Shape

### Composite Role Runner

Add a new overlay-aware `IInferenceRunner`, provisionally `MoEGraphRoleRunner`.

Its job is lifecycle and scalar coordination only:

- build and own local participant graph runners;
- broadcast or receive scalar step headers;
- run participant graphs for prefill/decode;
- collect status and propagate abort/shutdown;
- expose logits only on the continuation root.

It must not own routed tensor rows, expert partial tensors, or peer-device execution. It should have no API such as `submitDomainWork()`, `dispatchRows()`, or `runDomain()`.

```cpp
struct MoEGraphStepHeader {
    uint64_t generation_id = 0;
    uint64_t step_id = 0;
    int seq_len = 0;
    int position_offset = 0;
    bool decode = false;
};

class MoEGraphRoleRunner final : public IInferenceRunner {
public:
    bool forward(const int *tokens, int seq_len) override;
    const float *logits() const override;
    void clear_cache() override;

private:
    bool publishStepHeader(const MoEGraphStepHeader &header);
    bool runLocalParticipantGraphs(const MoEGraphStepHeader &header,
                                   const int *root_tokens);
    bool collectStatuses(const MoEGraphStepHeader &header);
};
```

### Participant Graphs

Every overlay participant gets a graph assignment:

- continuation participant: dense model path plus shared expert, MoE routing, sparse dispatch, and sparse return/reduce;
- accelerator expert participant: no-op dense stages as needed, MoE dispatch receive, local full-expert compute, return;
- CPU expert participant: no-op dense stages as needed, MoE dispatch receive, CPU local full-expert compute, return;
- relay/no-work participant: no-op at MoE collectives, status only.

The graph builder may initially generate specialized participant graphs rather than forcing every stage from the dense path into every participant. The important invariant is that all participants enter the same MoE collective sequence for the layers they participate in.

## Required Data Model

The reset needs compact sparse payloads, but it does not need those payloads to be arena buffers.

### Collective Key

Every sparse collective call must be keyed. A compact key is enough:

```cpp
enum class MoEOverlayCollectiveDirection : uint8_t {
    Dispatch,
    ReturnReduce,
};

struct MoEOverlayCollectiveKey {
    uint64_t generation_id = 0;
    uint64_t step_id = 0;
    int layer_idx = -1;
    int tier_idx = -1;
    int domain_id = -1;
    MoEOverlayCollectiveDirection direction = MoEOverlayCollectiveDirection::Dispatch;
    uint64_t sequence = 0;
};
```

Rules:

- stale keys are rejected;
- slot reuse is illegal until all participants complete or abort;
- no-op participants publish the same key with zero counts;
- trace logs always include the key and participant identity.

### Expert Ownership Table

The sparse dispatcher needs one canonical execution owner for every routed expert in every MoE layer. Caches and residency tiers are placement details over this owner table; they must not imply duplicate execution unless the return/reduce stage is explicitly told to combine replicas.

```cpp
struct MoEExpertOwner {
    int layer_idx = -1;
    int expert_id = -1;
    int tier_idx = -1;
    int owner_participant = -1;
    DeviceId device = DeviceId::cpu();
    bool resident = false;
};
```

For the motivating three-tier layout:

- `cuda_hot` owns the continuation device, all shared expert work, and a bounded cache of hottest routed experts;
- `rocm_warm` owns the next hottest routed experts across two Mi50 participants, with each participant owning full experts by id;
- `cpu_cold` owns every routed expert not assigned to hot or warm tiers.

If a routed expert appears in both a cache and a fallback tier, the planner must choose the cache owner or fallback owner for the step before dispatch. The local expert stage should never compute the same routed expert contribution on multiple participants as an accidental side effect of residency.

### Sparse Payload

A dispatch payload needs compact hidden rows plus routing entries. CSR over compact rows is enough.

```cpp
struct MoEOverlaySparseRows {
    MoEOverlayCollectiveKey key;
    int source_participant = -1;
    int target_participant = -1;
    int d_model = 0;
    int top_k = 0;

    int row_count = 0;
    int row_capacity = 0;
    int routed_entry_count = 0;
    int routed_entry_capacity = 0;

    TensorBase *row_ids = nullptr;          // int32 [row_capacity]
    TensorBase *hidden_rows = nullptr;      // FP32 [row_capacity, d_model]
    TensorBase *entry_offsets = nullptr;    // int32 [row_capacity + 1]
    TensorBase *expert_ids = nullptr;       // int32 [routed_entry_capacity]
    TensorBase *route_weights = nullptr;    // FP32 [routed_entry_capacity]
};
```

A return payload needs row ids plus compact output rows:

```cpp
struct MoEOverlayReturnRows {
    MoEOverlayCollectiveKey key;
    int source_participant = -1;
    int target_participant = -1;
    int d_model = 0;
    int row_count = 0;
    int row_capacity = 0;

    TensorBase *row_ids = nullptr;          // int32 [row_capacity]
    TensorBase *output_rows = nullptr;      // FP32 [row_capacity, d_model]
};
```

The exact tensor class for int32 metadata can be chosen during implementation. If existing tensor types make int32 awkward, use host vectors for metadata in the first host-staged version and keep FP32 tensors for values. Do not block the architecture reset on a perfect generic sparse tensor type.

## Are Sparse Partial Arena Buffers Required?

No, not for the first reset.

### What Is Actually Required

We need preallocated compact storage with:

- row capacity;
- live row count;
- routed entry capacity;
- live routed entry count;
- key identity;
- source and target participant identity;
- device/host residency for the compact values;
- lifetime through dispatch, local compute, and return/reduce.

That storage can be owned by a `MoEOverlayCollectiveWorkspace` or by the sparse collective context. It does not need `BufferId`, alias tracking, arena borrow tracking, or global stage-bound accessors to be correct.

### Why `BufferArena` Is The Wrong First Step

`BufferArena` is excellent for stable dense activation buffers named by `BufferId`: hidden state, logits, attention buffers, MoE router indices, and final MoE combined output. Sparse overlay payloads are different:

- they are keyed by generation/step/layer/tier/direction, not by a small stable model buffer id;
- they are dynamic live-prefix views over preallocated compact capacity;
- they often cross graph participants and ranks;
- they are transport slots as much as activation tensors;
- their first implementation can be host-staged and explicitly synchronized inside collective stages.

Adding sparse payloads to `BufferArena` now would force changes to `StageBufferContract`, `StageBoundBuffers`, `StageVerifier`, `StageDumper`, and arena coherence before the graph architecture is even corrected. That is needless surface area while the current production path is broken.

### What To Do Instead

Add a MoE-specific workspace owner:

```cpp
class MoEOverlayCollectiveWorkspace {
public:
    MoEOverlaySparseRows &dispatchReceive(int layer_idx, int tier_idx);
    MoEOverlaySparseRows &localExpertInput(int layer_idx, int tier_idx);
    MoEOverlayReturnRows &localExpertOutput(int layer_idx, int tier_idx);
    MoEOverlayReturnRows &returnReceive(int layer_idx, int tier_idx);

    bool ensureCapacity(int max_rows, int max_entries, int d_model, int top_k, DeviceId device);
    void resetForStep(uint64_t generation_id, uint64_t step_id);
};
```

Ownership options:

- one workspace per participant graph runner;
- one workspace per sparse collective context;
- one shared workspace object injected into MoE stages by graph construction.

All three are acceptable if lifetime is stable and no hot-path allocation occurs.

### When Arena Integration Becomes Worth It

Revisit arena sparse bindings only if at least one of these becomes true:

- graph capture requires sparse payloads to participate in the same binding machinery as dense buffers;
- validation/dump tooling must generically inspect sparse payloads across many non-MoE features;
- sparse payloads are reused by attention/sequence compaction beyond MoE;
- the manual collective workspace becomes a second general-purpose arena by accident.

Until then, keep sparse payloads MoE-local and explicit.

## Required Stages

### 1. `MoESparseDispatchStage`

Purpose: move compact selected rows and routing entries from routing/continuation participants to expert participants.

Inputs on continuation/routing participant:

- dense hidden state;
- routing expert ids;
- routing weights;
- placement table for expert-to-tier/domain mapping.

Inputs on expert participants:

- none, or an empty local call.

Outputs:

- compact dispatch payload in the participant workspace;
- no-op payload when row count is zero.

First implementation:

- host-staged packing via a cleaned-up form of `MoEExpertTokenRowTransfer`;
- local same-rank rendezvous by fixed slots;
- cross-rank CPU path via MPI count exchange plus payload send/receive;
- no full dense hidden-state copies to expert participants.

### 2. `MoELocalExpertStage`

Purpose: execute only this participant's expert work.

Inputs:

- compact dispatch payload for this participant;
- this participant's prepared expert weights;
- this participant's scratch tensors.

Outputs:

- compact return rows for this participant;
- zero-live output if no routed rows match local experts.

Rules:

- no peer participant vectors;
- no peer stream binding;
- no domain-wide allreduce;
- no scatter into the continuation dense output;
- no allocation in `execute()`.

### 3. `MoESparseReturnReduceStage`

Purpose: move compact expert outputs back to continuation participants and accumulate by original token row into the dense MoE output.

Inputs:

- compact return rows from local expert stages;
- continuation dense output buffer on the continuation participant.

Outputs:

- dense `MOE_COMBINED_OUTPUT` on continuation participant;
- no-op on non-continuation participants.

Implementation notes:

- extract the selected-row accumulation semantics from `MoEExpertParallelReduceStage`;
- host-staged scatter-add is acceptable first;
- GPU-native scatter-add can follow once the graph shape is correct.

### Dense TP Boundary

Dense LocalTP/GlobalTP is still valid, but it is not part of the routed-expert sparse collective.

Apply dense TP only where the model graph already has dense sharding semantics:

- attention projections and output projection;
- dense FFNs in dense models;
- shared expert gate/up/down weights when the continuation domain has multiple dense participants;
- LM head or logits gather.

The routed MoE overlay should see a clear continuation-domain activation layout before dispatch and produce the same layout after return/reduce. In the first reset that can be a single continuation device with full hidden rows. If a future continuation domain is dense-sharded, add explicit layout conversion at the MoE boundary rather than making routed expert participants pretend to be a LocalTP domain.

True `TensorParallelExperts`, where each selected expert's GEMMs are sharded across participants, is a separate future feature. It would need a different owner table and an explicit expert-GEMM reduction stage. It is not required for the tiered overlay reset.

### Continuation TP Plus Expert-Parallel Overlay Shape

The next architecture step is to let a LocalTP, NodeLocalTP, or GlobalTP continuation domain own the dense model path while routed experts still use sparse expert-parallel dispatch. The continuation domain is then the dense owner of attention, dense residual flow, shared expert, and LM head. Routed tiers remain a separate ordered list of expert-owning participants.

The important separation is:

- continuation dense TP uses `ITPContext` collectives for dense row-parallel/column-parallel work;
- routed MoE tiers use sparse dispatch/return collectives keyed by layer and step;
- a participant can belong to the continuation TP domain, a routed expert tier, both by explicit configuration, or neither;
- membership in a multi-device routed tier means whole-expert ownership by participant unless a future `TensorParallelExperts` feature is explicitly enabled.

The tier names must be user-defined, not baked into the execution model. These should all be valid shapes if memory and backend validation pass:

- `hot_cuda`, `warm_cuda`, `warm_rocm`, `cold_cpu`;
- `cuda_continuation`, `rocm_warm`, with no CPU fallback;
- GlobalTP CPU or GPU continuation plus same-node accelerator expert caches;
- a single continuation GPU with all routed experts distributed across several GPU-only tiers.

The planner should model this as roles over generic execution domains:

```cpp
enum class MoEContinuationActivationLayout {
    ReplicatedHidden,
    RootOnlyHidden,
    ShardedHiddenRequiresGather,
};

struct MoEContinuationDomainSpec {
    std::string domain;
    int logical_root_participant = 0;
    bool dense_tp_enabled = false;
    MoEContinuationActivationLayout hidden_layout = MoEContinuationActivationLayout::ReplicatedHidden;
    bool shared_expert_uses_dense_tp = true;
};

struct MoERoutedTierSpec {
    std::string name;
    std::string domain;
    int priority = 0;
    size_t memory_budget_bytes = 0;
    int max_experts_per_layer = 0;
    bool fallback = false;
};
```

The first continuation-TP implementation should use the simplest safe boundary:

1. Dense TP participants run the dense graph through the router input and reach the MoE dispatch key.
2. If hidden state is replicated after the preceding dense TP allreduce, only the logical continuation root packs sparse dispatch rows; other continuation TP participants publish no-op dispatch payloads for that key.
3. Routed expert tiers compute whole experts and return compact rows to the continuation root.
4. The continuation root scatter-adds routed outputs into `MOE_COMBINED_OUTPUT`.
5. A continuation import step broadcasts or otherwise materializes the combined MoE output into the activation layout required by the next dense TP stage.

If a later dense TP layout keeps hidden state sharded at the MoE boundary, add an explicit gather/export stage before sparse dispatch and an import/scatter stage after return/reduce. Do not let sparse routed tiers infer dense TP layout implicitly.

Existing code already points in this direction but needs cleanup before it is safe:

- `GraphConfig::tp_ctx` is already an `ITPContext*`, and `TPAllreduceStage` is polymorphic for LocalTP and GlobalTP.
- `TPContextFactory` can create LocalTP or GlobalTP contexts from execution plans/domains.
- `GraphConfig::domain_tp_contexts` and related named-domain plumbing are still `ILocalTPContext*`-shaped and need to become generic `ITPContext*` plumbing for continuation/base/shared domains.
- `ExpertComputeDomain::fromExecutionDomainDefinition()` currently rejects `scope=global`; that may remain true for routed expert tiers in the first reset, but continuation/base/shared dense domains must be allowed to be global.
- `Qwen35MoESchema` already treats routed expert weights as `ExpertParallel` while shared expert weights are dense column/input parallel, which is the desired split.

## Transport Backend Requirements

Introduce a small MoE sparse collective context rather than a domain runtime:

```cpp
class IMoEOverlaySparseCollectiveContext {
public:
    virtual bool dispatch(const MoEOverlayCollectiveKey &key,
                          const MoEOverlaySparseRows *outbound,
                          MoEOverlaySparseRows *inbound,
                          IDeviceContext *ctx) = 0;

    virtual bool returnReduce(const MoEOverlayCollectiveKey &key,
                              const MoEOverlayReturnRows *outbound,
                              MoEOverlayReturnRows *inbound,
                              IDeviceContext *ctx) = 0;

    virtual void abort(const MoEOverlayCollectiveKey &key, int reason_code) = 0;
};
```

This context is allowed to move tensor payloads because it is the collective backend. It is not allowed to decide model layer progress or run expert compute.

Backend order:

1. in-process/local rendezvous for unit tests and same-rank participant graphs;
2. MPI host-staged transport for CPU participants;
3. same-rank GPU host-staged transport;
4. GPU-native pack/unpack and peer transport only after correctness is stable.

## What To Delete Or Quarantine

Production overlay execution should stop using:

- `MoEOverlayDomainRuntimeStage`;
- `IOverlayDomainRuntime` and `MoEOverlayDomainRuntime`;
- `MoEExpertOverlayLocalTPStage`;
- domain-wide execution in `MoEExpertOverlayLocalTPExecutor`;
- routed-tier lowering that requires `TensorParallelExperts` or domain-scoped `ILocalTPContext` for whole-expert ownership;
- `MoEOverlayCPUFallbackParticipantRunner` as the production CPU participant model;
- compatibility paths where runtime code instantiates stages inside `submit()`.

During the big-bang branch, these can remain temporarily for tests or reference, but the Qwen3.5 MoE production graph must not lower to them.

## Test Gate Policy And Earliest MVP

Every phase below has to land with a real CTest gate. Do not advance the architecture by several phases on paper and discover at Phase 8 that the graph contract is unworkable. The intended workflow is:

1. Add or update the unit/integration test first, with the legacy runtime path disabled or clearly isolated.
2. Implement the smallest code slice required for that phase.
3. Run the phase gate in `build_v2_integration` with full CTest parallelism.
4. Keep the previous phase gates passing, especially the earliest graph-native MoE MVP.

The earliest MVP must be model-light and mathematically strict. It should not require loading the 35B GGUF or running server mode. Use deterministic FP32 toy weights, fixed routing indices/weights, small dimensions, and compare the graph-native tiered result against a single-domain `MoEExpertComputeStage` reference with `EXPECT_NEAR` tolerance on every output element.

### MVP Integration Test: `rocm_hot + cpu_cold`

Add a new registered CTest target as soon as the new sparse workspace, dispatch, local expert, and return/reduce stages can execute one layer end to end:

```text
tests/v2/integration/moe/Test__MoEGraphNative_RocmHotCpuCold_MVP.cpp
CTest: V2_Integration_MoEGraphNative_RocmHotCpuCold_MVP
Labels: V2;Integration;MoE;GraphNative;ExpertParallel;ROCm;CPU;MPI;MVP
MPI_PROCS: 2
```

Topology:

- rank 0: continuation root plus `rocm_hot` participant on `rocm:0`;
- rank 1: `cpu_cold` participant;
- routed expert owner map: some experts owned by `rocm_hot`, the remaining experts owned by `cpu_cold`;
- no `TensorParallelExperts`, no routed expert allreduce, no `MoEOverlayDomainRuntimeStage`, no `MoEExpertOverlayLocalTPStage`, no `MoEOverlayCPUFallbackParticipantRunner`.

Math contract:

- build hidden rows, top-k routing, route weights, and gate/up/down expert weights deterministically;
- compute full reference output with all experts on CPU via `MoEExpertComputeStage`;
- dispatch only rows required by each owner tier;
- run `MoELocalExpertStage` locally on `rocm_hot` and `cpu_cold`;
- return compact rows and scatter-add on the continuation root;
- compare final `MOE_COMBINED_OUTPUT` against the full CPU reference element-by-element.

Hardware handling:

- the test may `GTEST_SKIP()` when the build lacks `HAVE_ROCM` or no ROCm device is visible;
- the CI/hardware lane for this project must include a non-skipped run before the refactor is considered feasible;
- a CPU-only graph-native MVP should exist earlier for fast developer feedback, but it is not a substitute for the `rocm_hot + cpu_cold` MVP.

Existing tests are useful reference material but not sufficient gates by themselves. `tests/v2/integration/moe/Test__MoEExpertOverlay_CPUFallback_MPI.cpp` already has sparse CPU fallback math and MPI transfer assertions, and `tests/v2/integration/moe/Test__MoEExpertOverlay_MultiAcceleratorTiers.cpp` has deterministic tiered FP32 reference math. Reuse those fixtures and expectations, but move the MVP onto graph-native stages and register it directly in `tests/v2/CMakeLists.txt` instead of relying on legacy filtered bridge tests.

## Implementation Plan

### Phase 0: Stop The Bleeding

Goal: make it impossible to accidentally benchmark or ship the broken overlay runtime.

Tasks:

- Add a feature guard or config error for old overlay-domain-runtime execution.
- Keep single-device non-overlay MoE paths working if they do not use shadow domains.
- Rename diagnostics for old bridge code to `legacy_overlay_domain_runtime` so logs make the old path obvious.
- Mark old runtime-stage tests as legacy or rewrite them to target the new stages.

Acceptance:

- Qwen3.5 overlay configs fail fast unless the new graph-native path is selected.
- No production graph contains `MoEOverlayDomainRuntimeStage`.

Test gate:

- Update `tests/v2/unit/config/Test__MoEExpertOverlayConfig.cpp` and `tests/v2/unit/models/qwen35moe/Test__Qwen35MoEExpertOverlayGraph.cpp` so overlay-domain-runtime lowering is rejected unless an explicit legacy test flag is set.
- Run `ctest --test-dir build_v2_integration -R "V2_Unit_(MoEExpertOverlayConfig|Qwen35MoEExpertOverlayGraph)" --output-on-failure --parallel`.

### Phase 1: Participant Planning And Role Runner

Goal: all overlay ranks/devices become graph participants.

Tasks:

- Define `MoEGraphParticipantSpec` with rank-qualified participant identity.
- Keep `DeviceId::cpu()` rank-local; CPU participant identity must include world rank and participant id.
- Build `MoEGraphRoleRunner` with scalar step headers and status collection.
- Teach `OrchestrationRunner` to create this runner for overlay plans.
- Build participant graph assignments for continuation, accelerator expert, CPU expert, and relay roles.

Acceptance:

- Non-root CPU ranks can run an empty/no-op participant graph for a step and report completion.
- The role runner cannot carry tensor payloads by type or API.

Test gate:

- Add `V2_Unit_MoEGraphRoleRunner_NoTensorPayloadAPI` to assert the role runner only moves scalar step headers/status and owns participant graph runners.
- Add `V2_Integration_MoEGraphRoleRunner_NoOpMPI` with at least 2 MPI ranks where continuation and non-continuation participants enter matching no-op MoE keys and complete a step.
- Run `ctest --test-dir build_v2_integration -R "V2_(Unit_MoEGraphRoleRunner|Integration_MoEGraphRoleRunner_NoOpMPI)" --output-on-failure --parallel`.

### Phase 2: MoE Collective Workspace And Host-Staged Transport

Goal: create the minimal compact payload system without touching `BufferArena`.

Tasks:

- Add `MoEOverlayCollectiveKey`.
- Add `MoEOverlaySparseRows` and `MoEOverlayReturnRows` payload views.
- Add `MoEOverlayCollectiveWorkspace` with preallocated capacities.
- Add local rendezvous backend and MPI host-staged backend.
- Reuse/refactor `MoEExpertTokenRowTransfer` for CPU host pack/scatter correctness.

Acceptance:

- Unit tests move compact rows between fake participants by key.
- No-op participants complete the same key.
- Stale key reuse is rejected.
- No per-token heap allocation occurs after workspace initialization.

Test gate:

- Add `V2_Unit_MoEOverlayCollectiveWorkspace` for capacity reuse, live-prefix reset, stale key rejection, and no-op payloads.
- Add or replace `V2_Unit_MoEOverlayDispatchCollective` coverage so it moves `MoEOverlaySparseRows` / `MoEOverlayReturnRows`, not only control headers.
- Add `V2_Integration_MoEOverlaySparseTransport_MPI` with 2 MPI ranks exchanging compact rows and return rows by key.
- Run `ctest --test-dir build_v2_integration -R "V2_(Unit_MoEOverlayCollectiveWorkspace|Unit_MoEOverlayDispatchCollective|Integration_MoEOverlaySparseTransport_MPI)" --output-on-failure --parallel`.

### Phase 3: New Graph Stages

Goal: replace domain runtime stages with explicit graph stages.

Tasks:

- Implement `MoESparseDispatchStage`.
- Implement participant-local `MoELocalExpertStage` for CPU first, then accelerator prepared weights.
- Implement `MoESparseReturnReduceStage`.
- Update `Qwen35MoEGraph` lowering to emit these stages for overlay mode.

Acceptance:

- A CPU-only overlay fixture routes rows, computes local expert output, returns compact partials, and accumulates into dense MoE output.
- A participant with no routed rows enters all collective stages and produces no contribution.
- Stage params for local expert compute contain no peer participant vectors.

Test gate:

- Add `V2_Integration_MoEGraphNative_CPUOnly_MVP` using deterministic FP32 toy weights, fixed routing, graph-native sparse dispatch/local compute/return-reduce, and a full `MoEExpertComputeStage` CPU reference.
- Add a unit guard that `MoELocalExpertStage::Params` has no peer participant vector, runtime pointer, or runner pointer dependency.
- Run `ctest --test-dir build_v2_integration -R "V2_(Integration_MoEGraphNative_CPUOnly_MVP|Unit_MoELocalExpertStage_Params)" --output-on-failure --parallel`.

### Phase 4: Split Accelerator Expert-Parallel Tiers

Goal: remove shadow LocalTP execution completely and lower multi-device warm tiers as expert-parallel participants.

Tasks:

- Extract per-participant expert GEMM execution from `MoEExpertOverlayLocalTPExecutor` into a local helper.
- Build the layer/expert owner table for hot, warm, and cold tiers.
- Assign disjoint full routed experts to each warm-tier participant according to residency budget and histogram priority.
- Reject or explicitly mark routed-tier `TensorParallelExperts` unsupported in the first graph-native reset.
- Bind only the current participant's stream and workspace.
- Use prepared expert weights from `PreparedWeightStore` / expert GEMM registry per participant.
- Keep dense LocalTP, if present, confined to the continuation/shared-expert dense boundary.

Acceptance:

- ROCm:0 graph launches only ROCm:0 full-expert kernels for its owned expert ids.
- ROCm:1 graph launches only ROCm:1 full-expert kernels for its owned expert ids.
- Routed expert compute requires no RCCL/NCCL allreduce and no domain-scoped `ILocalTPContext`.
- No stage allocates peer workspaces or resolves peer streams.

Test gate:

- Add `V2_Integration_MoEGraphNative_RocmHotCpuCold_MVP` described in the MVP section. This is the first required heterogeneous feasibility gate.
- Add `V2_Unit_MoEExpertOwnerMap_DisjointAcceleratorParticipants` proving multi-device accelerator tiers lower to disjoint whole-expert owners, not to `TensorParallelExperts`.
- Run `ctest --test-dir build_v2_integration -R "V2_(Integration_MoEGraphNative_RocmHotCpuCold_MVP|Unit_MoEExpertOwnerMap_DisjointAcceleratorParticipants)" --output-on-failure --parallel`.
- A skipped ROCm MVP is acceptable for a developer laptop but not for declaring Phase 4 complete on the project hardware lane.

### Phase 5: Retire CPU Sidecar Fallback

Goal: CPU experts run through normal graph stages.

Tasks:

- Port useful CPU fallback math into participant-local `MoELocalExpertStage` helpers.
- Delete or quarantine `MoEOverlayCPUFallbackParticipantRunner`.
- Remove endpoint receive-pump dependency from production overlay inference.
- Keep CPU prepared packed VNNI weights eager and resident according to existing memory work.

Acceptance:

- CPU expert ranks are normal participant graphs.
- CPU participants consume dispatch payloads and produce return payloads.
- No CPU participant infers layer progress from endpoint messages.

Test gate:

- Replace the production-path dependency in `V2_Integration_MoEExpertOverlay_CPUFallback_MPI` with `V2_Integration_MoEGraphNative_CPUCold_MPI`, preserving its sparse-row masked-reference math.
- Add a guard test that production graph construction contains no `MoEOverlayCPUFallbackParticipantRunner` or endpoint receive-pump dependency.
- Re-run `V2_Integration_MoEGraphNative_RocmHotCpuCold_MVP` after CPU sidecar removal.
- Run `ctest --test-dir build_v2_integration -R "V2_(Integration_MoEGraphNative_CPUCold_MPI|Integration_MoEGraphNative_RocmHotCpuCold_MVP|Unit_MoEOverlayNoCPUFallbackSidecar)" --output-on-failure --parallel`.

### Phase 6: Cleanup And Hardening

Goal: remove legacy paths and add diagnostics that protect the architecture.

Tasks:

- Remove production registration for old overlay runtime stages.
- Add unit tests that fail if graph-native stage params contain `IOverlayDomainRuntime`, peer prepared participant vectors, or runner pointers.
- Add collective traces with key, participant id, device, row counts, entry counts, and no-op state.
- Add transfer counters for dense bytes avoided versus compact bytes moved.
- Revisit sparse arena integration only if the workspace abstraction is being reused outside MoE.

Acceptance:

- Search for `MoEOverlayDomainRuntimeStage` finds no production graph lowering.
- Search for `prepared_participants` finds no graph-native local expert stage params.
- One-token decode does not perform full hidden-state D2H except under explicit diagnostics.

Test gate:

- Add `V2_Unit_MoEGraphNative_ForbiddenDependencyScan` that inspects graph-native stage params or graph lowering helpers for forbidden runtime/peer dependencies.
- Add `V2_Integration_MoEGraphNative_TransferCounters` proving compact bytes moved are less than dense bytes for a routed sparse fixture.
- Re-run the CPU-only MVP, the `rocm_hot + cpu_cold` MVP, and the CPU-cold MPI graph-native test.
- Run `ctest --test-dir build_v2_integration -R "V2_(Unit_MoEGraphNative_ForbiddenDependencyScan|Integration_MoEGraphNative_TransferCounters|Integration_MoEGraphNative_(CPUOnly_MVP|RocmHotCpuCold_MVP|CPUCold_MPI))" --output-on-failure --parallel`.

### Phase 7: Generic Continuation Domain TP Contract

Goal: make continuation/base/shared dense domains first-class generic execution domains rather than expert-domain special cases.

Tasks:

- Add `MoEContinuationDomainSpec` or equivalent plan fields that identify the continuation domain, logical continuation root, dense TP scope, and MoE boundary activation layout.
- Split validation rules for continuation/base/shared domains from routed expert tier domains. Continuation dense domains may be `SINGLE`, `LOCAL`, `NODE_LOCAL`, or `GLOBAL`; routed tiers remain whole-expert ownership domains for the reset.
- Generalize named TP context plumbing from `ILocalTPContext*` maps to `ITPContext*` maps where graph builders need polymorphic LocalTP/GlobalTP dense collectives.
- Use `TPContextFactory::createFromDomain()` or equivalent plan-derived creation for continuation domains instead of routed-tier LocalTP bridge contexts.
- Keep `TensorParallelExperts` rejected for routed tiers unless a separate feature flag explicitly enables true tensor-sharded expert GEMMs.
- Define one logical continuation root per continuation TP domain for logits ownership, sparse dispatch packing, and return/reduce accumulation.

Acceptance:

- A MoE overlay plan can name a LocalTP or GlobalTP continuation domain without converting that domain into a routed expert tier.
- Dense graph stages in the continuation domain receive an `ITPContext*` and create normal `TPAllreduceStage` / logits gather stages.
- Routed tier validation still guarantees one canonical whole-expert execution owner per routed expert.
- No graph-native routed expert stage requires `domain_tp_contexts` or `ILocalTPContext`.

Test gate:

- Add `V2_Unit_MoEContinuationDomainSpec` for `SINGLE`, `LOCAL`, `NODE_LOCAL`, and `GLOBAL` continuation validation.
- Add `V2_Unit_MoEContinuationTPContextPlumbing` proving named continuation/base/shared domains expose polymorphic `ITPContext*` to graph builders.
- Add config-only integration tests for LocalTP and GlobalTP continuation plans that still reject routed-tier `TensorParallelExperts` by default.
- Run `ctest --test-dir build_v2_integration -R "V2_(Unit_MoEContinuationDomainSpec|Unit_MoEContinuationTPContextPlumbing|Integration_MoEContinuation.*Config)" --output-on-failure --parallel`.

### Phase 8: Continuation TP MoE Boundary Stages

Goal: connect dense TP continuation execution to sparse ExpertParallel dispatch without duplicating routed work.

Tasks:

- Add explicit continuation export/import behavior around the MoE sparse stages. This can be standalone stages or a documented contract inside `MoESparseDispatchStage` and `MoESparseReturnReduceStage`.
- Start with `ReplicatedHidden`: only `logical_root_participant` packs dispatch rows; other continuation TP participants enter sparse collectives as no-op participants.
- Return compact routed rows to the logical continuation root and scatter-add into root `MOE_COMBINED_OUTPUT`.
- Materialize `MOE_COMBINED_OUTPUT` back into the continuation domain layout needed by the next dense stage, initially by TP broadcast from the logical root for replicated-hidden continuation layouts.
- Keep shared expert execution in the continuation dense graph. If the continuation domain has dense TP, shared expert gate/up/down weights use the existing column/input-parallel sharding and dense TP allreduces.
- Add explicit assertions that replicated hidden rows are not dispatched once per continuation TP participant.

Acceptance:

- A two-device LocalTP continuation domain dispatches each routed row once, not once per dense TP participant.
- Non-root continuation TP participants enter sparse MoE keys and make progress with zero outbound routed rows.
- After return/reduce, every continuation TP participant has the correct activation layout for the next dense graph stage.
- Shared expert dense TP and routed expert sparse dispatch can coexist in the same MoE layer without sharing collective contexts.

Test gate:

- Add `V2_Integration_MoEGraphNative_LocalTPContinuation_RocmHotCpuCold_MVP` for replicated-hidden export/import with a LocalTP continuation domain and sparse routed tiers.
- Add `V2_Integration_MoEGraphNative_GlobalTPContinuation_CPUCold_MVP` with 2 MPI ranks proving GlobalTP dense continuation collectives can coexist with sparse expert dispatch.
- Assert dispatch row count equals the reference routed row count, not routed row count multiplied by continuation TP degree.
- Run `ctest --test-dir build_v2_integration -R "V2_Integration_MoEGraphNative_(LocalTPContinuation_RocmHotCpuCold_MVP|GlobalTPContinuation_CPUCold_MVP)" --output-on-failure --parallel`.

### Phase 9: Flexible Routed Tier Planner

Goal: support configurable routed expert tiers around a dense TP continuation domain without hardcoding hot/warm/cold assumptions.

Tasks:

- Treat tier names as arbitrary user labels and tier order as explicit priority, not as fixed `hot_cuda`, `warm_rocm`, or `cold_cpu` roles.
- Build the layer/expert owner table from tier budgets, residency policy, explicit masks, and optional fallback tiers.
- Allow GPU-only plans with no CPU fallback if configured tiers cover every routed expert required for the model/layer.
- Allow mixed tiers such as fast CUDA hot cache, slower CUDA warm cards, ROCm warm cards, and CPU cold fallback when those domains are present.
- Validate that each routed expert has exactly one canonical execution owner for each layer/step, even if several domains keep resident cached copies.
- Teach sparse transport planning to select local rendezvous, host-staged GPU transfer, MPI host-staged transfer, or later GPU-native peer transfer per source/target pair without changing local expert compute semantics.

Acceptance:

- Plans with `hot_cuda`, `warm_cuda`, `warm_rocm`, and `cold_cpu` validate through the same owner-map code as GPU-only plans.
- Removing `cold_cpu` is valid when all routed experts are covered by configured accelerator tiers.
- No routed-tier combination introduces dense TP allreduce for routed expert compute.
- Diagnostics print continuation domain, continuation root, tier name, owner participant, device, budget, resident expert count, and fallback coverage.

Test gate:

- Add `V2_Unit_MoERoutedTierPlanner_FlexibleNamesAndBudgets` for arbitrary tier names, GPU-only coverage, optional fallback, cached residency, and exactly-one-owner validation.
- Add `V2_Integration_MoEGraphNative_GPUOnlyTiers_MVP` showing no `cold_cpu` is required when accelerator tiers cover all routed experts.
- Add `V2_Integration_MoEGraphNative_MixedCudaRocmCpuTiers_MVP` if hardware is available; otherwise make it a hardware-gated test that is required on the project lane.
- Run `ctest --test-dir build_v2_integration -R "V2_(Unit_MoERoutedTierPlanner_FlexibleNamesAndBudgets|Integration_MoEGraphNative_(GPUOnlyTiers_MVP|MixedCudaRocmCpuTiers_MVP))" --output-on-failure --parallel`.

## Testing Strategy

### Unit Tests

- `MoEOverlayCollectiveKey` equality, stale rejection, and slot reuse.
- `MoEOverlayCollectiveWorkspace` capacity reuse and no hot-path allocation.
- Local rendezvous dispatch with routed, no-op, cancel, and stale-key cases.
- MPI host-staged dispatch payload count exchange and row transfer.
- Expert owner table construction for `cuda_hot`, two-participant `rocm_warm`, and `cpu_cold`.
- Expert owner table construction for arbitrary tier names, GPU-only coverage, and no-fallback plans.
- Warm-tier disjoint expert masks by participant, including no-work participants.
- Continuation domain validation for `SINGLE`, `LOCAL`, `NODE_LOCAL`, and `GLOBAL` scopes without applying routed-tier expert compute rules.
- Named continuation TP context plumbing exposes `ITPContext*`, not only `ILocalTPContext*`.
- Replicated-hidden continuation export packs routed rows only from the logical root.
- Continuation import broadcasts or materializes returned `MOE_COMBINED_OUTPUT` for every continuation TP participant.
- `MoESparseDispatchStage` packs rows and CSR routing metadata correctly.
- `MoELocalExpertStage` no-ops on zero rows and computes expected output on CPU fixture.
- Routed expert dispatch sends rows only to the canonical owner participant for each expert id.
- No routed ExpertParallel test invokes an allreduce backend.
- `MoESparseReturnReduceStage` accumulates duplicate row contributions correctly.
- Graph-stage param guard tests for forbidden dependencies.

### Integration Tests

- Single-rank CPU continuation plus CPU expert participant graph.
- Multi-rank CPU cold expert participant with host-staged MPI transport.
- Same-rank two-accelerator warm tier with disjoint expert owners, no routed-expert allreduce.
- Reduced Layout A ROCm repro as expert-parallel participants, not a LocalTP expert domain.
- Two-device LocalTP continuation with sparse routed dispatch to a separate expert tier.
- Two-rank GlobalTP continuation with sparse routed dispatch to a CPU or accelerator tier.
- GPU-only tier plan with no `cold_cpu`, proving fallback is optional when configured tiers cover all routed experts.
- Mixed `hot_cuda`, `warm_cuda`, `warm_rocm`, `cold_cpu` tier plan using the same owner-map and sparse transport path.
- Qwen3.5 MoE parity on the graph-native path.

### Diagnostics

Every MoE sparse stage should be able to log, when tracing is enabled:

- collective key;
- participant id and device;
- continuation domain and logical continuation root;
- domain and tier;
- direction;
- row count and row capacity;
- routed entry count;
- no-op versus routed work;
- bytes moved;
- wait time;
- abort reason.

## Definition Of Done

The reset is complete when:

1. Qwen3.5 MoE overlay production graphs do not use `MoEOverlayDomainRuntimeStage` or `IOverlayDomainRuntime`.
2. Every overlay participant executes a graph participant role rather than a sidecar domain worker.
3. Local expert compute stages launch only local-device work.
4. Routed expert overlay does not require `ILocalTPContext`, `TensorParallelExperts`, or hidden dense-TP collectives.
5. CPU expert participants consume and produce compact payloads through graph collective stages.
6. Sparse dispatch and return payloads are compact, keyed, preallocated, and no-op capable.
7. Sparse payload storage is owned by MoE collective workspace/context, not `BufferArena`, unless a later measured need justifies generalization.
8. Old bridge/runtime tests are deleted, rewritten, or clearly marked legacy.
9. Reduced ROCm and CPU cold overlay probes run through the same graph-native path used by parity, benchmark, oneshot, and serve.
10. Dense LocalTP, when enabled, is bounded to dense graph stages or an explicit MoE boundary conversion.
11. LocalTP and GlobalTP continuation domains can run dense TP collectives while routed tiers still use sparse ExpertParallel dispatch.
12. Flexible routed tiers are configured by domain, priority, budget, residency, and fallback coverage, not by hardcoded hot/warm/cold names.
13. One-token decode avoids repeated full dense activation transfers outside explicit debug dumps.

## Open Questions

1. Do we need a small int32 tensor type for row ids and CSR metadata immediately, or are host vectors acceptable for the host-staged reset?
2. Should shared experts remain continuation-domain dense work for the first reset, or should they get a separate explicit boundary when dense TP is active?
3. For LocalTP/GlobalTP continuation, should the first implementation require `ReplicatedHidden`, or should it immediately support sharded hidden export/import?
4. Should sparse dispatch be root-packed for all decode/prefill first, or should prefill support row-owned packing across continuation TP participants to avoid root bandwidth pressure?
5. Should true `TensorParallelExperts` be supported as a later, separate feature, or should the overlay only promise whole-expert ownership?
6. Should the first MPI backend be point-to-point `Isend/Irecv` or a simpler root-mediated scatter/gather?
7. How much of the existing `MoEExpertOverlayCPUFallback` code can be kept once sidecar domain context ownership is removed?
