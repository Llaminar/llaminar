# Multi-Domain Pipeline Execution Plan

**Date**: 2026-05-07
**Status**: Proposed
**Scope**: Enable pipeline-parallel topologies where one MPI rank can participate in multiple execution domains, including local GPU TP stages and cross-rank node-local CPU TP stages.

---

## Summary

The Qwen 3.5 35B MoE GPU-cache parity topology exposed a real architectural gap. The intended topology is:

```text
PipelineParallel(
    LocalTP(0:rocm:0, 0:rocm:1),
    NodeLocalTP(0:cpu:0, 1:cpu:0)
)
```

That means:

```text
rank 0:
    stage 0 runner: LocalTP over ROCm GPUs on socket 0
    stage 1 runner: NodeLocalTP CPU shard for socket 0

rank 1:
    stage 1 runner: NodeLocalTP CPU shard for socket 1
```

The larger extensible target also includes multiple local domains before a node-local CPU tail:

```text
PipelineParallel(
    LocalTP(0:rocm:0, 0:rocm:1),
    LocalTP(0:cuda:0, 0:cuda:1),
    NodeLocalTP(0:cpu:0, 1:cpu:0)
)
```

In that case:

```text
rank 0:
    stage 0 runner: LocalTP(rocm:0, rocm:1)
    stage 1 runner: LocalTP(cuda:0, cuda:1)
    stage 2 runner: NodeLocalTP CPU shard for socket 0

rank 1:
    stage 2 runner: NodeLocalTP CPU shard for socket 1
```

Those rank numbers are examples, not a special rule. Any rank must be allowed to execute multiple stage runners in sequence. If the local GPU domains are attached to socket 1 instead of socket 0, the shape becomes:

```text
PipelineParallel(
    NodeLocalTP(0:cpu:0, 1:cpu:0),
    LocalTP(1:rocm:0, 1:rocm:1),
    LocalTP(1:cuda:0, 1:cuda:1)
)
```

and rank ownership flips:

```text
rank 0:
    stage 0 runner: NodeLocalTP CPU shard for socket 0

rank 1:
    stage 0 runner: NodeLocalTP CPU shard for socket 1
    stage 1 runner: LocalTP(rocm:0, rocm:1)
    stage 2 runner: LocalTP(cuda:0, cuda:1)
```

The current runtime can describe parts of this, but it cannot execute the full shape cleanly because `GlobalOrchestrator` owns a single `rank_runner_`. The fix is to make stage/domain execution first class: a rank owns a registry of stage runners, and each `GlobalPPRankPlan` step dispatches to the runner for its `stage_id`.

---

## Current Findings

### What Already Exists

`GlobalPPTopology` can represent two important concepts:

- single-rank stages with `inner_mode = LOCAL_TP`, and
- multi-rank stages with `is_global_tp = true` and `participating_ranks`.

`GlobalPPRankPlan` already supports multiple `EXECUTE_STAGE` steps per rank. The plan format does not inherently require one stage per rank.

`GlobalTPContext::createWithSplit()` already provides domain-specific MPI communicator creation via `MPI_Comm_split`.

`PPStage` and `HierarchicalPPContext` already model stage types such as `TP_DOMAIN` and `GLOBAL_TP_DOMAIN`, and already contain transfer semantics for local TP to global TP, global TP to local TP, and related local handoff cases.

`RankOrchestrator` can already create nested local TP runners inside local PP stages by using `nested_pp_stage_config`.

### What Breaks

`GlobalOrchestrator::Config` has only one owned runner:

```cpp
std::unique_ptr<IInferenceRunner> rank_runner;
```

The implementation delegates all stage execution, hidden-state IO, logits, snapshots, timeline control, and cache control to that single runner. This is the root mismatch.

`TreeToRunnerCompiler::PipelineRunner` is not enough for this case either. It assigns a stage to a canonical owning rank, which falls over for a stage that is a cross-rank TP domain. It also does not solve the general case where any rank must execute both one or more local GPU TP stages and its CPU shard of a node-local TP stage.

`GlobalPPTopology::build()` currently treats same-rank adjacent stages as no-op transfers. That is correct only when the same runner continues execution. With multiple stage runners on the same rank, a same-rank transition still needs a local handoff:

```text
stage 0 LocalTP GPU runner -> stage 1 CPU TP shard runner on the same rank
```

For single-rank to global-TP transitions, the current topology creates MPI fan-out transfers only to non-source participants. The source rank can also be a participant in the destination global TP domain, so it needs a local handoff action even though it does not need MPI.

The parity test `Test__Qwen35MoE_HybridPPTP_Parity.cpp` currently uses `devices = {ROCm, CPU, ROCm, CPU}` with `pp_stage_sizes = {2, 2}`. The test harness slices the device vector by adjacency, yielding mixed TP stages `{ROCm:0, CPU}` and `{ROCm:1, CPU}` instead of the intended `LocalTP(ROCm, ROCm)` plus `NodeLocalTP(CPU, CPU)`.

---

## Design Principles

1. **Stage participation is the execution unit.** A rank can participate in zero, one, or many pipeline stages.
2. **Pipeline stages own domains, not just devices.** A stage can be a single device, local TP domain, local PP domain later, or cross-rank TP domain.
3. **Runner dispatch is by `stage_id`.** `GlobalOrchestrator` should not assume one rank runner.
4. **Local handoff is a real transfer action.** Same-rank adjacent stages still need hidden-state movement when they are different runners or domains.
5. **MPI communicators are domain scoped.** Node-local/global TP stages should use a communicator for that stage/domain, not an implicit `MPI_COMM_WORLD` context.
6. **Stage-local build configuration must be immutable after runner creation.** Multiple runners on the same rank may need different TP configs, devices, layer ranges, and weight materialization decisions.
7. **Existing simple paths remain simple.** Single-device, pure local TP, pure local PP, and pure NodeLocalTP should keep working through adapters or degenerate registries.
8. **MoE residency remains dynamic but domain aware.** Dynamic expert rebalancing should target explicit stage/domain runners and respect orchestrator-owned residency budgets.

---

## Target Runtime Model

### Stage Domain Spec

Evolve `GlobalPPStageSpec` or introduce a sibling model that can represent every pipeline stage as an execution domain.

Suggested shape:

```cpp
enum class StageDomainKind {
    SingleDevice,
    LocalTP,
    LocalPP,
    GlobalTP,
};

struct StageExecutionDomain {
    int stage_id = -1;
    std::string domain_name;
    StageDomainKind kind = StageDomainKind::SingleDevice;

    int first_layer = -1;
    int last_layer = -1; // inclusive at topology level
    bool has_embedding = false;
    bool has_lm_head = false;

    // Single-rank domains.
    int owning_rank = -1;
    std::vector<GlobalDeviceAddress> local_devices;
    std::vector<float> tp_weights;

    // Cross-rank TP domains.
    std::vector<int> participating_ranks;
    std::vector<GlobalDeviceAddress> per_rank_devices;

    CollectiveBackendType backend = CollectiveBackendType::AUTO;
};
```

`GlobalPPStageSpec` already has most of these fields. The likely path is to extend it rather than replace it:

- add `domain_name`,
- add explicit backend for both local and global TP domains,
- support per-rank devices for global TP, instead of a single `per_rank_device`,
- distinguish `LOCAL_TP` single-rank stages from `GLOBAL_TP` multi-rank stages without overloading `owning_rank` semantics.

### Stage Runner Entry

Each rank should build one entry for each stage it executes.

```cpp
struct StageRunnerEntry {
    int stage_id = -1;
    std::string domain_name;
    RankStageAction action;
    std::unique_ptr<IInferenceRunner> runner;

    std::shared_ptr<ILocalTPContext> local_tp_ctx;
    std::shared_ptr<IGlobalTPContext> global_tp_ctx;
    std::optional<FactoryPPStageConfig> pp_stage_config;
};
```

### Stage Runner Registry

Add a small ownership class rather than spreading maps through `GlobalOrchestrator`.

```cpp
class StageRunnerRegistry {
public:
    void add(StageRunnerEntry entry);
    IInferenceRunner *runnerForStage(int stage_id);
    const IInferenceRunner *runnerForStage(int stage_id) const;

    IInferenceRunner *pipelineHeadRunner();
    IInferenceRunner *pipelineTailRunner();

    void clearCacheAll();
    void enableSnapshotCaptureAll(const std::string &output_dir);
    void clearSnapshotsAll();
    std::vector<std::string> snapshotKeysAll() const;
};
```

`GlobalOrchestrator::Config` should move from a single `rank_runner` to a registry or vector of stage entries:

```cpp
struct Config {
    GlobalPPTopology topology;
    int rank = 0;
    int world_size = 1;
    IMPIContext *mpi_ctx = nullptr;

    std::vector<StageRunnerEntry> stage_runners;

    int vocab_size = 0;
    int d_model = 0;
    std::string architecture_name;
};
```

Keep a compatibility adapter during migration: if legacy code supplies `rank_runner`, convert it into a one-entry registry for the rank's only executable stage.

---

## Transfer and Handoff Model

### Add Local Handoff Actions

Extend `RankTransferAction::Direction`:

```cpp
enum class Direction {
    SEND,
    RECV,
    LOCAL_HANDOFF,
    NONE,
};
```

`LOCAL_HANDOFF` means: both source and destination stage participants are this rank, but the destination stage runner still needs its hidden-state input populated.

Examples that require `LOCAL_HANDOFF`:

- `LocalTP(rocm:0, rocm:1)` on a rank -> `LocalTP(cuda:0, cuda:1)` on that same rank.
- `LocalTP(rocm:0, rocm:1)` on a rank -> `NodeLocalTP(cpu socket 0, cpu socket 1)`, for that same rank's CPU shard.
- `GlobalTP(ranks 0,1)` -> `LocalTP(cuda:0,cuda:1)` on rank 0 or rank 1, when that rank is a participant and owns the next local stage.
- `GlobalTP(ranks 0,1)` -> `GlobalTP(ranks 0,1)` for a distinct later stage/domain.

### Activation Router

Introduce a focused helper for stage-to-stage activation movement:

```cpp
class StageActivationRouter {
public:
    bool executeTransfer(const RankTransferAction &action,
                         StageRunnerRegistry &registry,
                         IMPIContext &mpi_ctx,
                         int seq_len,
                         int d_model);
};
```

Responsibilities:

- For `SEND`, read hidden state from `from_stage` runner and send via MPI.
- For `RECV`, receive into a correctly sized activation tensor and call `setHiddenState()` on the `to_stage` runner.
- For `LOCAL_HANDOFF`, read hidden state from the `from_stage` runner, move it to the destination domain if needed, and call `setHiddenState()` on the `to_stage` runner.

Reuse `PPStage` and `HierarchicalPPContext` for local handoff semantics where possible. That code already knows how to transfer between local TP domains, single devices, and global TP domain representatives. If it is awkward to reuse directly, use it as the reference behavior and factor the reusable pieces into a non-`ILocalPPContext` helper.

### Transfer Derivation Rules

Update `GlobalPPTopology::build()` and `GlobalPPRankPlanBuilder` so adjacent stages produce all required per-rank transfer actions.

For stage A -> stage B:

1. Let `A_ranks` be ranks participating in A.
2. Let `B_ranks` be ranks participating in B.
3. For every rank in `A_ranks` intersect `B_ranks`, emit `LOCAL_HANDOFF` unless the same runner continues both stages.
4. For every rank in `B_ranks - A_ranks`, emit a `RECV` action on that rank and a matching `SEND` action on a designated source rank from A.
5. If A is single-rank and B is multi-rank, the designated source is A's owning rank.
6. If A is multi-rank and B is single-rank outside A, the designated source should be deterministic, usually the first participant in A unless topology policy specifies otherwise.
7. If A and B are both multi-rank with partial overlap, overlapping ranks use local handoff and non-overlapping B ranks receive fan-out from a deterministic A participant.

This turns same-rank transitions from a skipped no-op into an explicit local action whenever the destination runner is different.

---

## Stage Runner Construction

Add a `StageRunnerFactory` that builds the right runner for a `RankStageAction`.

```cpp
class StageRunnerFactory {
public:
    StageRunnerEntry create(const GlobalPPStageSpec &stage,
                            const RankStageAction &action,
                            const StageBuildContext &ctx);
};
```

### Single Device Stage

Use `createPPStageRunner(model_ctx, device, pp_stage_config, runner_config)`.

### Local TP Stage

Use `RankOrchestrator` in TP mode with `nested_pp_stage_config`:

```cpp
RankOrchestrator::Config config;
config.mode = RankOrchestrator::ParallelismMode::TP;
config.devices = stage.devices;
config.weights = stage.tp_weights;
config.backend = stage.backend;
config.nested_pp_stage_config = pp_stage_config;
```

This is the same pattern that currently powers TP domains inside local PP.

### NodeLocalTP / GlobalTP Stage

Create or retrieve a domain-specific `GlobalTPContext` and build a per-rank stage runner with:

- `InferenceRunnerConfig::pp_stage_config` set to the stage layer range,
- `InferenceRunnerConfig::tp_ctx` set to the `GlobalTPContext`,
- `tp_device_index` set from `global_tp_ctx->myIndex()`,
- `default_device` / `device` set to the rank's CPU NUMA device,
- weight sharding based on this domain's rank index and domain size.

Current `createTestableInferenceRunner()` has most of the PP-stage graph plumbing, but it treats non-local TP mostly as an implicit world-size path. Add a path that accepts a pre-created global `ITPContext` directly:

```cpp
if (tp_ctx && !tp_ctx->isLocal()) {
    graph_config.tp_ctx = tp_ctx;
    graph_config.tp_device_idx = tp_ctx->myIndex();
    applyGlobalTPAssignmentForContext(...);
}
```

This avoids accidentally using `MPI_COMM_WORLD` for every global TP stage.

### Domain Communicator Registry

Global TP communicator creation must be collective and deterministic across all MPI ranks. Add a registry that every rank initializes from the full topology:

```cpp
class DomainCommunicatorRegistry {
public:
    void initialize(const GlobalPPTopology &topology, MPI_Comm world);
    std::shared_ptr<IGlobalTPContext> globalTPContextForStage(int stage_id) const;
};
```

For each global TP stage in ascending `stage_id` order:

- participants call `MPI_Comm_split(world, domain_color, key, &comm)`,
- non-participants call the same split with `MPI_UNDEFINED`,
- participants wrap the resulting communicator in `GlobalTPContext`,
- non-participants store no context for that stage.

This prevents communicator mismatches once there are multiple global/node-local TP stages.

---

## Stage-Scoped Weight and Model Context Work

This is the riskiest implementation area.

Today, several paths configure a shared `WeightManager` with an active TP config before runner creation. That is manageable when a rank has one active TP domain. It becomes ambiguous when any rank builds:

- ROCm local TP stage,
- CUDA local TP stage,
- CPU global TP shard stage.

Each stage can have different devices, backend, TP degree, layer range, and resident weight set. The long-term contract should be stage-scoped materialization:

```cpp
struct StageWeightContext {
    int stage_id;
    FactoryPPStageConfig pp_stage_config;
    WeightManagerConfig weight_config;
    FrozenModelWeightSet frozen_weights;
    std::shared_ptr<PreparedWeightStore> prepared_store;
};
```

Implementation options:

1. Configure the shared `WeightManager`, immediately materialize immutable stage weights, then never rely on the mutable active config for that runner.
2. Give each `StageRunnerEntry` its own stage-local `PreparedWeightStore` keyed by binding ids and stage id.
3. Keep the shared loader/model metadata, but make prepared/model weights stage-owned after materialization.

The project should avoid reintroducing tensor-pointer lookup or global `KernelFactory` prepared-weight ownership. Prepared weight resolution should remain binding/ref based.

Minimum first milestone: prove local TP GPU stage plus CPU global TP stage can be built without the second stage's TP config invalidating prepared handles used by the first stage.

---

## Config Surface

The target topology should be expressible through named domains rather than parity-test-only structures.

Suggested CLI/YAML semantics:

```bash
--define-domain "rocm_socket0=0:rocm:0,0:rocm:1;scope=local;backend=rccl;owner=0"
--define-domain "cpu_sockets=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;ranks=0,1"
--pp-stage "0=rocm_socket0:0-19"
--pp-stage "1=cpu_sockets:20-39"
```

Extended example:

```bash
--define-domain "rocm_socket0=0:rocm:0,0:rocm:1;scope=local;backend=rccl;owner=0"
--define-domain "cuda_socket0=0:cuda:0,0:cuda:1;scope=local;backend=nccl;owner=0"
--define-domain "cpu_sockets=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;ranks=0,1"
--pp-stage "0=rocm_socket0:0-12"
--pp-stage "1=cuda_socket0:13-25"
--pp-stage "2=cpu_sockets:26-39"
```

Notes:

- The precise current syntax for CPU sockets should use NUMA-qualified `GlobalDeviceAddress` values such as `0:cpu:0` and `1:cpu:0`. Conceptually these are `cpu:0` and `cpu:1` sockets, but `DeviceId::cpu()` is still effectively singleton-like in many lower layers.
- `scope=local` means all devices are controlled by one MPI rank.
- `scope=node_local` means one domain spans multiple ranks on the same host, typically one rank per CPU socket.
- `owner=0` is required for local domains unless it can be inferred from inventory.
- `ranks=0,1` is required for node-local/global TP domains unless it can be inferred from device placement.

`ExecutionPlanBuilder` should eventually produce a `GlobalPPTopology` directly for any named-domain PP configuration that mixes local and cross-rank domains.

---

## MoE Expert Residency Implications

The immediate topology project should not remove dynamic histogram-based MoE expert rebalancing. It should make it domain aware.

Required changes:

- `GlobalOrchestrator` should expose a way to visit MoE-capable stage runners by domain or stage id.
- The Qwen 3.5 MoE GPU-cache parity test should apply hot expert cache masks to the ROCm local TP stage and cold/shared placement to the CPU node-local TP stage through the stage runner registry, not by casting the root runner to `RankOrchestrator`.
- The future `MoEExpertResidencyPlan` should attach budgets to execution domains, not only devices. A local TP domain may budget shared experts across its GPU devices, while a node-local CPU TP domain may own the cold expert source of truth.
- Dynamic migration remains valid at inference time, but it must respect the per-domain hot cache budget and should not mutate a domain that does not participate in the MoE layers for the current stage.

---

## Implementation Phases

### Phase 0: Lock the Topology Bug With Tests

Add model-free unit tests before changing runtime behavior.

Tasks:

- Add a `GlobalPPTopology` / `GlobalPPRankPlanBuilder` test for:
  - stage 0: rank 0 local TP over ROCm devices,
  - stage 1: ranks 0 and 1 node-local CPU TP.
- Assert rank 0 plan has:
  - execute stage 0,
  - local handoff stage 0 -> stage 1,
  - send stage 0 -> stage 1 to rank 1,
  - execute stage 1.
- Assert rank 1 plan has:
  - receive stage 0 -> stage 1 from rank 0,
  - execute stage 1.
- Add the three-stage ROCm LocalTP -> CUDA LocalTP -> NodeLocalTP plan test.
- Add the mirrored socket-locality test where rank 1, not rank 0, owns the local GPU TP stages plus its NodeLocalTP CPU shard.

Acceptance:

- Tests fail against the current no-op same-rank transfer behavior.
- Tests pass after `LOCAL_HANDOFF` is represented.

### Phase 1: Extend Topology and Rank Plan Semantics

Tasks:

- Add domain metadata to `GlobalPPStageSpec`.
- Add `LOCAL_HANDOFF` to `RankTransferAction`.
- Update transfer derivation to emit same-rank handoffs between distinct adjacent stages.
- Preserve existing pure global PP and pure global TP behavior.
- Update `toString()` and `toTable()` output to show local handoff actions and domain names.

Acceptance:

- Existing `V2_Unit_GlobalPPTopology` / rank-plan tests pass.
- New multi-domain plan tests pass.
- Topology tables make every rank's stage participations visible, including ranks with more than one local runner.

### Phase 2: Add Stage Runner Registry to GlobalOrchestrator

Tasks:

- Introduce `StageRunnerEntry` and `StageRunnerRegistry`.
- Change `GlobalOrchestrator::Config` to accept multiple stage runners.
- Keep compatibility for single-runner construction during migration.
- Dispatch `executeStage()` by `stage_id`.
- Dispatch `executeTransfer()` through a `StageActivationRouter` that knows `from_stage` and `to_stage`.
- Aggregate `clear_cache`, snapshot, timeline, and skip-logits controls across all local stage runners.
- Route `logits()` and sampling through the local tail stage runner only.

Acceptance:

- Model-free tests with mock runners prove any rank can execute two or more stage runners in order.
- Local handoff populates the destination runner's hidden state.
- Existing GlobalOrchestrator tests keep passing through compatibility mode.

### Phase 3: Build Stage Runners Per Domain

Tasks:

- Add `StageRunnerFactory`.
- Build single-device PP stage runners with `createPPStageRunner`.
- Build local TP stage runners with `RankOrchestrator` and `nested_pp_stage_config`.
- Build global/node-local TP stage runners with a pre-created `GlobalTPContext` and stage-local `pp_stage_config`.
- Add direct support for global `ITPContext` injection in `InferenceRunnerFactory` so it does not silently create a world-sized context.
- Add `DomainCommunicatorRegistry` and deterministic communicator splitting for all global TP stages.

Acceptance:

- Mock/model-light integration tests build both orientations: rank 0 with multiple stage runners and rank 1 with multiple stage runners.
- No stage creates a global TP communicator over ranks outside its domain.
- A rank can build both a local TP stage and a global TP shard stage in one process, regardless of rank id.

### Phase 4: Make Weight Materialization Stage Scoped

Tasks:

- Audit `WeightManager::configure()` and prepared weight creation under multiple stage domains on one rank.
- Introduce `StageWeightContext` or equivalent immutable materialization per stage.
- Ensure local TP ROCm, local TP CUDA, and CPU global TP shard runners do not share mutable prepared-weight state accidentally.
- Keep binding/ref based prepared weight resolution.

Acceptance:

- Building a second stage runner cannot invalidate the first stage runner's prepared handles.
- Stage runners can have different TP degrees/devices/backends in one process.
- Existing prepared-weight ownership guard tests continue passing.

### Phase 5: Config and Factory Integration

Tasks:

- Extend named domain parsing with `scope`, `owner`, `ranks`, and backend semantics as needed.
- Teach `ExecutionPlanBuilder` to emit `GlobalPPTopology` for mixed local/cross-rank named-domain PP.
- Activate `GlobalOrchestratorRunner` creation in `OrchestrationRunnerFactory` once model metadata is available.
- Add `--show-topology` output for multi-domain stage participation and per-rank runner counts.

Acceptance:

- The target topology can be expressed through CLI/YAML, not only parity code.
- The factory no longer logs that global orchestration is detected but falling back to standard path for this topology.

### Phase 6: Qwen 3.5 MoE Parity Migration

Tasks:

- Update `v2_integration_parity_qwen35moe_hybrid_pptp` to run with `MPI_PROCS 2`.
- Replace adjacency-based `pp_stage_sizes` with explicit topology/domain construction.
- Assert the built topology is:

```text
PipelineParallel(
    LocalTP(0:rocm:0, 0:rocm:1),
    NodeLocalTP(0:cpu:0, 1:cpu:0)
)
```

- Update GPU expert cache mask application to use stage/domain runner access.
- Add timing instrumentation around setup, warmup, mask application, and parity execution after topology correctness is fixed.

Acceptance:

- Rank 0 logs two stage runners.
- Rank 1 logs one stage runner.
- A mirrored topology test logs rank 1 with multiple stage runners and rank 0 with only its CPU TP shard runner.
- The ROCm stage uses RCCL/local TP over ROCm devices only.
- The CPU stage uses node-local/global TP over two MPI ranks only.
- Prefill and decode parity run against the intended topology.

---

## Test Plan

### Unit Tests

- `GlobalPPTopology` transfer derivation for local handoff.
- `GlobalPPRankPlanBuilder` for multi-stage-per-rank plans.
- `StageRunnerRegistry` lookup, tail/head selection, cache and snapshot fan-out.
- `StageActivationRouter` with mock runners for local handoff, send, and receive behavior.
- `DomainCommunicatorRegistry` deterministic split ordering with mocked or small MPI tests.

### Integration Tests

- Model-free/mock runner global orchestrator test:

```text
rank 0: execute stage 0, local handoff, send, execute stage 1
rank 1: receive, execute stage 1
```

- Mirrored mock runner global orchestrator test:

```text
rank 0: execute NodeLocalTP shard stage 0, send or local handoff as required
rank 1: execute NodeLocalTP shard stage 0, local handoff, execute local TP stage 1, execute local TP stage 2
```

- Existing `HierarchicalPP_GlobalTP` tests should continue passing.
- Existing local TP and NodeLocalTP parity tests should continue passing.
- New Qwen 3.5 MoE HybridPPTP topology smoke test should verify runner/domain shape before running expensive parity.

### Parity Tests

- Migrate Qwen 3.5 MoE HybridPPTP prefill/decode parity to the actual topology.
- Keep the old accidental mixed ROCm+CPU topology only if it has independent value, and rename it accordingly.

---

## Risks and Open Questions

### Mutable WeightManager State

The largest risk is stage runner construction mutating shared weight configuration. This must be addressed before relying on multiple TP domains in one rank.

### CPU Socket Identity

Current lower layers often collapse CPU to `DeviceId::cpu()`. The topology should use NUMA-qualified `GlobalDeviceAddress` for placement, but execution may still rely on MPI rank binding to distinguish CPU sockets.

### Tail Semantics for Global TP

If the tail stage is a global TP domain, all participating ranks execute the tail layers, but only one rank should be the canonical sampler/broadcaster. Existing behavior chooses the first participating rank; keep that unless a better policy is needed.

### Local Handoff Tensor Ownership

For local handoff, avoid aliasing a tensor whose coherence state belongs to the source runner if the destination runner may mutate or retain it. Start with explicit transfer/copy semantics, then optimize zero-copy later.

### Multiple Global TP Domains

Every MPI rank must call communicator splits in the same deterministic order. Centralizing this in `DomainCommunicatorRegistry` avoids subtle deadlocks.

### Resource Budgeting

Any rank may own several heavy runners. The expert residency budget work should eventually account for all local domains on a rank, not just one runner.

---

## Definition of Done

The project is complete when Llaminar can express, build, and execute this topology with two MPI ranks:

```text
PipelineParallel(
    LocalTP(0:rocm:0, 0:rocm:1),
    NodeLocalTP(0:cpu:0, 1:cpu:0)
)
```

and this topology without additional architecture changes:

```text
PipelineParallel(
    LocalTP(0:rocm:0, 0:rocm:1),
    LocalTP(0:cuda:0, 0:cuda:1),
    NodeLocalTP(0:cpu:0, 1:cpu:0)
)
```

with these observable properties:

- any rank can own multiple stage runners,
- any other participating rank can own only a CPU TP shard runner,
- same-rank stage transitions perform local handoff,
- cross-rank destination participants receive activations through MPI,
- global/node-local TP collectives are scoped to the intended domain,
- per-stage TP configs and prepared weights do not conflict,
- Qwen 3.5 MoE HybridPPTP parity runs against the intended topology.
