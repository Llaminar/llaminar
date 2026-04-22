# Cross-Machine MPI Cluster Inference: GlobalOrchestrator Project Plan

**Date**: 2026-04-22
**Status**: Draft
**Branch**: `feature/global-orchestrator`

---

## Overview

Enable full cross-machine inference over an MPI cluster with multiple hosts.
The design introduces a **GlobalOrchestrator** that coordinates cross-rank
pipeline parallelism (PP) and tensor parallelism (TP), delegating per-rank
execution to **RankOrchestrator** (renamed from `MultiDeviceOrchestrator`).

### Three-Tier Orchestration Stack

```
GlobalOrchestrator               ← NEW (cross-rank: MPI PP + Global TP)
  └─ RankOrchestrator            ← RENAME of MultiDeviceOrchestrator (per-rank: local devices)
       └─ DeviceGraphOrchestrator  ← UNCHANGED (per-device: graph execution)
```

| Tier | Scope | Coordination | Transport |
|------|-------|-------------|-----------|
| GlobalOrchestrator | Cluster (all MPI ranks) | PP send/recv, Global TP allreduce | MPI, cross-rank NCCL/RCCL |
| RankOrchestrator | Single MPI rank (N local devices) | LOCAL TP, LOCAL PP | NCCL, RCCL, HOST |
| DeviceGraphOrchestrator | Single device | Compute graph DAG execution | Device-local |

### Design Principles

1. **Implements `IInferenceRunner`** — transparent to callers (`ChatCompletionHandler`, `BenchmarkMode`, etc.)
2. **One GlobalOrchestrator per MPI rank** — each rank consults its `GlobalPPRankPlan`
3. **Graph-based PP** — send/recv stages inserted into compute graph via `IPipelineParallelGraphBuilder`, not imperative calls
4. **Naive sequential PP first** — micro-batch scheduling layered later
5. **Degenerate cases handled** — global-TP-only (no PP) and global-PP-only (no TP) both route through GlobalOrchestrator

---

## Current State: What Exists

### Ready to Use (no changes needed)

| Component | File | Status |
|-----------|------|--------|
| `GlobalPPTopology` + `GlobalPPStageSpec` | `execution/global_pp/GlobalPPTopology.h` | 100% — stage→layer→rank mapping |
| `GlobalPPRankPlanBuilder` | `execution/global_pp/GlobalPPRankPlanBuilder.h` | 100% — deadlock-free plan derivation |
| `GlobalPPRankPlan` + `RankStageAction` | `execution/global_pp/GlobalPPRankPlan.h` | 100% — per-rank step sequence |
| `GlobalPPTransferStage` | `execution/compute_stages/stages/GlobalPPTransferStage.h` | 100% — MPI Send/Recv + coherence (17 unit tests) |
| `SendActivationsStage` / `ReceiveActivationsStage` | `execution/compute_stages/stages/` | 100% — async-capable MPI P2P |
| `IPipelineParallelGraphBuilder` | `execution/local_execution/graph/IPipelineParallelGraphBuilder.h` | 100% — `insertSendStage()` / `insertReceiveStage()` |
| `TPContextFactory` | polymorphic LOCAL/GLOBAL factory | 100% — `createGlobal()` with `MPI_Comm_split` |
| `GlobalTPContext` | domain-specific MPI communicator | 100% — stages are backend-agnostic via `ITPContext` |
| `AllreduceStage` (dual-mode) | `execution/compute_stages/stages/AllreduceStage.h` | 100% — CollectiveContext + legacy MPI |
| `AllGatherStage` / `AllGatherVStage` | `execution/compute_stages/stages/` | 100% — logit reconstruction |
| `WeightShardingMode` | `Qwen2Schema.h`, `Qwen3Schema.h` | 100% — ColumnParallel, InputParallel, Replicate per-weight |
| `ModelLoader::loadTensorColumnSlice/RowSlice` | `loaders/ModelLoader.cpp` | 100% — sharded weight loading |
| `LogitsGatherer` | `execution/local_execution/orchestrators/LogitsGatherer.h` | 100% — per-device shards → full logits |
| NCCL cross-rank init | `collective/backends/NCCLBackend.cpp:273-320` | 100% — `ncclGetUniqueId` → `MPI_Bcast` → `ncclCommInitRank` |
| RCCL cross-rank init | `collective/backends/RCCLBackend` | 100% — same pattern as NCCL |
| Backend auto-selection | `collective/BackendRouter.cpp:747` | 100% — NCCL/RCCL for homogeneous even cross-rank |
| MPI bootstrap + hostfile | `app/MPIBootstrapPhase.cpp` | 100% — self-launch, topology detection, NUMA binding |
| `OrchestrationConfig` (CLI/YAML) | `config/OrchestrationConfig.h` | 100% — `tp_scope`, `pp_degree`, domains |
| `RankExecutionPlan` | `config/RankExecutionPlan.h` | 100% — global_tp fields, weight_shard, layer ranges |
| Async MPI primitives | `utils/MPIContext.h:477-520` | 100% — `isend()` / `irecv()` wrappers |

### Stubbed / Partial

| Component | File | Issue |
|-----------|------|-------|
| Heterogeneous multi-phase AllReduce | `collective/BackendRouter.cpp:500` | `LOG_WARN "not yet implemented"` → HOST fallback |
| Heterogeneous multi-phase AllGather | `collective/BackendRouter.cpp:527` | Same stub |
| Legacy `OrchestrationRunner` PP | `execution/runner/OrchestrationRunner.cpp:1243-1268` | `sendActivationsToNextStage()` / `receiveActivationsFromPrevStage()` are TODO stubs |
| RCCL RDMA config | `collective/backends/RCCLBackendHIP.cpp:348` | Hardcoded P2P_DISABLE for single-device only |

---

## Phase 0: Rename `RankOrchestrator` → `RankOrchestrator`

**Goal**: Align naming with the three-tier stack before adding new code.

### Tasks

- [ ] **0.1** Rename `RankOrchestrator` → `RankOrchestrator` (class, files, CMake)
- [ ] **0.2** Rename `IRankOrchestrator` → `IRankOrchestrator`
- [ ] **0.3** Rename `MockRankOrchestrator` → `MockRankOrchestrator`
- [ ] **0.4** Update all `#include` paths, forward declarations, factory references
- [ ] **0.5** Update docs referencing old name (`MULTI_DEVICE_ARCHITECTURE.md`, `MULTI_DEVICE_ORCHESTRATOR_PLAN.md`, etc.)
- [ ] **0.6** Full build + full test suite (unit, integration, parity)

### Files Affected

```
src/v2/execution/local_execution/orchestrators/RankOrchestrator.h  → RankOrchestrator.h
src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp → RankOrchestrator.cpp
src/v2/execution/local_execution/orchestrators/IRankOrchestrator.h → IRankOrchestrator.h
src/v2/interfaces/IRankOrchestrator.h                              → IRankOrchestrator.h
src/v2/execution/factory/InferenceRunnerFactory.h  (forward decl)
tests/v2/unit/**/MockRankOrchestrator.*     → MockRankOrchestrator.*
tests/v2/unit/**/Test__RankOrchestrator.*   → Test__RankOrchestrator.*
```

### Acceptance Criteria

- All 397+ unit tests pass
- All 40+ parity tests pass
- `grep -r RankOrchestrator src/ tests/` returns zero hits

---

## Phase 1: GlobalOrchestrator — Core Structure

**Goal**: Minimal GlobalOrchestrator that implements `IInferenceRunner`, handles the pure-global-TP case (no PP), and passes through to a single `RankOrchestrator`.

### Class Design

```
src/v2/execution/global/GlobalOrchestrator.h
src/v2/execution/global/GlobalOrchestrator.cpp
```

```cpp
namespace llaminar2 {

class GlobalOrchestrator : public IInferenceRunner {
public:
    struct Config {
        GlobalPPTopology topology;          // Cluster-wide stage layout
        GlobalPPRankPlan rank_plan;         // This rank's execution steps
        int rank = 0;                       // MPI rank
        int world_size = 1;                 // Total MPI ranks
        IMPIContext* mpi_ctx = nullptr;     // MPI communicator
        ITPContext* global_tp_ctx = nullptr; // Global TP context (if any)
        
        // Per-rank local config (forwarded to RankOrchestrator)
        RankOrchestrator::Config rank_config;
        
        // Model loading
        IModelLoader* model_loader = nullptr;
        WeightShardInfo weight_shard;
    };

    explicit GlobalOrchestrator(Config config);
    ~GlobalOrchestrator() override;

    // IInferenceRunner interface
    bool forward(const int* tokens, int seq_len) override;
    const float* logits() const override;
    int vocab_size() const override;
    void clear_cache() override;
    int sampleGreedyOnDevice() override;
    int sampleOnDevice(const SamplingParams& params) override;

    // Query
    bool isPipelineHead() const;   // Has embedding layer
    bool isPipelineTail() const;   // Has LM head
    int pipelineDepth() const;     // Number of PP stages
    const GlobalPPRankPlan& rankPlan() const;

private:
    // Per-rank local execution
    std::unique_ptr<IRankOrchestrator> rank_orchestrator_;
    
    // PP state
    std::shared_ptr<TensorBase> activation_send_buf_;
    std::shared_ptr<TensorBase> activation_recv_buf_;
    
    // Logit gathering (tail rank only)
    std::unique_ptr<LogitsGatherer> logits_gatherer_;
    
    // Config
    Config config_;
    
    // Internal
    bool executePipelineStep(const GlobalPPRankPlan::Step& step);
    bool executeStage(const RankStageAction& action, const int* tokens, int seq_len);
    bool executeTransfer(const RankTransferAction& action);
};

} // namespace llaminar2
```

### Tasks

- [ ] **1.1** Create `GlobalOrchestrator.h` with the struct/class above
- [ ] **1.2** Implement constructor: build `GlobalPPRankPlan` from topology, create `RankOrchestrator` (or `DeviceGraphOrchestrator` for single-device stages), create `GlobalTPContext` if needed
- [ ] **1.3** Implement `forward()` — iterate `rank_plan_.steps`, dispatch EXECUTE_STAGE vs TRANSFER
- [ ] **1.4** Implement `executeStage()` — delegate to `rank_orchestrator_->forward()` for this rank's layers
- [ ] **1.5** Implement `executeTransfer()` — use `GlobalPPTransferStage` for MPI Send/Recv
- [ ] **1.6** Implement `logits()` — return valid logits on tail rank, nullptr on others
- [ ] **1.7** Implement `clear_cache()` — delegate to rank_orchestrator + MPI_Barrier
- [ ] **1.8** Implement `sampleGreedyOnDevice()` / `sampleOnDevice()` — tail rank samples, MPI_Bcast token to all ranks
- [ ] **1.9** Register in CMakeLists.txt

### Acceptance Criteria

- Compiles and links
- Unit tests for GlobalOrchestrator with MockRankOrchestrator (mock MPI)
- Pure-global-TP case (no PP, all ranks run all layers) works end-to-end

---

## Phase 2: Global Pipeline Parallelism (PP-only)

**Goal**: Two MPI ranks on separate machines, each running half the layers, connected by MPI Send/Recv.

### Tasks

- [ ] **2.1** Wire `IPipelineParallelGraphBuilder::insertSendStage()` / `insertReceiveStage()` into the graph built by `GlobalOrchestrator::executeStage()` based on `RankTransferAction`
- [ ] **2.2** Implement activation buffer sizing — hidden state dimensions from model config
- [ ] **2.3** Handle embedding: only pipeline-head rank runs embedding, sends hidden state to next stage
- [ ] **2.4** Handle LM head: only pipeline-tail rank runs LM head + logit output
- [ ] **2.5** Handle decode loop: tail rank broadcasts sampled token to all ranks via `MPI_Bcast`, all ranks advance their KV cache position
- [ ] **2.6** Integration test: 2-rank PP with Qwen2.5-0.5B on simulated MPI (or `mpirun -np 2` on single machine)

### Acceptance Criteria

- `mpirun -np 2 llaminar2 oneshot --pp 2 -m model.gguf -p "Hello" -n 10 -t 0` produces correct greedy output
- Token predictions match single-device reference (greedy, temperature=0)
- `--dry-run --explain-placement` shows correct layer→rank assignment

---

## Phase 3: Global TP + PP Composition

**Goal**: Combine global TP within PP stages. Example: 4 ranks, 2 PP stages × 2-way TP.

### Tasks

- [ ] **3.1** For PP stages with `is_global_tp=true`: create per-stage `GlobalTPContext` via `MPI_Comm_split` (color=stage_id)
- [ ] **3.2** Configure `RankOrchestrator` with the stage's TP context and weight shard info
- [ ] **3.3** Weight loading: use `WeightShardInfo` from `RankStageAction` to load column/row slices
- [ ] **3.4** Insert `AllreduceStage` after row-parallel GEMMs (Wo, FFN down) using stage's TP communicator
- [ ] **3.5** Insert `AllGatherStage` after column-parallel LM head for logit reconstruction
- [ ] **3.6** Integration test: 4-rank (2 PP × 2 TP) with medium model

### Acceptance Criteria

- `mpirun -np 4 llaminar2 oneshot --pp 2 --tp 2 --tp-scope global -m model.gguf -p "Hello" -n 10 -t 0` produces correct output
- Weight memory per rank ≈ 1/(PP×TP) of full model

---

## Phase 4: Factory Integration + Rank Locality

**Goal**: Wire `GlobalOrchestrator` into `OrchestrationRunnerFactory` so the CLI/server paths use it automatically, and add rank locality metadata so the framework can make topology-aware strategy decisions.

### Rank Locality

MPI ranks may be co-located on a single physical machine (NodeLocal) or distributed
across machines (Global). The orchestrator code path is the same for both, but
**strategy selectors** need this information to make good decisions:

| Consumer | What it uses locality for |
|----------|--------------------------|
| `BackendRouter` | Prefer shared-memory/HOST for intra-node, NCCL-over-IB for inter-node |
| Topology builder (`4.3`) | `--tp-scope auto` → TP within a node, PP across nodes |
| `--explain-placement` | "ranks 0,1 on node alpha (NodeLocal TP), PP to node beta" |
| Future 1F1B scheduler | Prioritize compute-communication overlap for inter-node transfers |

**Data model** (added to `GlobalPPTopology`):

```cpp
struct RankLocality {
    int rank;
    std::string hostname;   // from MPI_Get_processor_name()
    int node_id;            // derived: ranks with same hostname get same node_id
};

// In GlobalPPTopology:
std::vector<RankLocality> rank_localities;
bool areColocated(int rank_a, int rank_b) const;
std::vector<int> ranksOnNode(int node_id) const;
int nodeCount() const;
```

**Transfer locality** (added to `GlobalPPTransfer`):

```cpp
enum class TransferLocality { INTRA_NODE, INTER_NODE, UNKNOWN };
TransferLocality locality = TransferLocality::UNKNOWN;
```

Populated during `GlobalPPTopology::build()` when rank localities are available.

### Tasks

- [ ] **4.1** Add `RankLocality` struct to `GlobalPPTopology.h` with `hostname`, `node_id`
- [ ] **4.2** Add `areColocated()`, `ranksOnNode()`, `nodeCount()` queries to `GlobalPPTopology`
- [ ] **4.3** Add `TransferLocality` enum to `GlobalPPTransfer`; populate in `build()` from rank localities
- [ ] **4.4** Build `GlobalPPTopology` from `OrchestrationConfig` in `OrchestrationRunnerFactory`
  - Parse `pp_degree`, `tp_scope`, domain definitions into `GlobalPPStageSpec` list
  - Query `MPI_Get_processor_name()` per rank to populate `RankLocality`
  - `--tp-scope auto` with localities: prefer TP within a node, PP across nodes
- [ ] **4.5** Instantiate `GlobalOrchestrator` when `world_size > 1` or `pp_degree > 1`
  - `GlobalOrchestrator` wraps a per-rank `DeviceGraphOrchestrator` (created via existing `InferenceRunnerFactory`)
  - Wrap in `IOrchestrationRunner` adapter for `AppContext` compatibility
- [ ] **4.6** Ensure single-rank paths are unchanged (`GlobalOrchestrator` not created)
- [ ] **4.7** Update `--dry-run` / `--explain-placement` to show `GlobalOrchestrator` topology
  - Display rank→node mapping, transfer localities, TP/PP structure
  - Example: `GlobalOrchestrator: 4 ranks on 2 nodes, 2 PP stages × 2-way NodeLocal TP`
- [ ] **4.8** Unit tests for rank locality (co-location queries, transfer locality derivation)
- [ ] **4.9** Unit tests for topology builder (OrchestrationConfig → GlobalPPTopology)

### Acceptance Criteria

- `mpirun -np 2 llaminar2 oneshot --pp 2 ...` works without explicit GlobalOrchestrator construction
- Single-rank inference is unchanged (no regression)
- `--dry-run` shows topology with rank→node locality: `ranks 0,1 on node0 (NodeLocal TP), PP → ranks 2,3 on node1`
- `GlobalPPTransfer::locality` correctly reflects INTRA_NODE vs INTER_NODE
- `areColocated(0, 1)` returns true for same-machine ranks, false for cross-machine

---

## Phase 5: Retire Legacy `OrchestrationRunner` PP Stubs

**Goal**: Remove dead code that GlobalOrchestrator replaces.

### Tasks

- [ ] **5.1** Remove `OrchestrationRunner::sendActivationsToNextStage()` stub
- [ ] **5.2** Remove `OrchestrationRunner::receiveActivationsFromPrevStage()` stub
- [ ] **5.3** Remove `OrchestrationRunner::isPipelineHead()` / `isPipelineTail()` (if unused)
- [ ] **5.4** Audit remaining `OrchestrationRunner` usage — migrate callers if any remain
- [ ] **5.5** Full test suite pass

### Acceptance Criteria

- `grep -r 'sendActivationsToNext\|receiveActivationsFromPrev' src/` returns zero hits
- No test regressions

---

## Phase 6: Hardening & Observability

**Goal**: Production-ready error handling, diagnostics, and profiling.

### Tasks

- [ ] **6.1** MPI error handling: wrap all MPI calls in GlobalOrchestrator with error checking + rank-aware error messages
- [ ] **6.2** Timeout detection: if MPI_Recv blocks > configurable threshold, log warning with peer rank and expected tensor size
- [ ] **6.3** `LLAMINAR_MPI_LOG_COLLECTIVES` support: log all send/recv/allreduce with timing and bandwidth
- [ ] **6.4** `--show-topology` integration: print GlobalPPTopology table with ranks, layers, devices, TP groups
- [ ] **6.5** Stage dump support: `LLAMINAR_STAGE_DUMP_RANK` works with GlobalOrchestrator
- [ ] **6.6** Parity tests: multi-rank parity vs single-device PyTorch reference
- [ ] **6.7** Benchmark mode: `mpirun -np N llaminar2 benchmark --pp N -m model.gguf` with per-rank timing breakdown

### Acceptance Criteria

- No silent failures — all MPI errors surface as exceptions with rank context
- `--show-topology` renders correct table for any PP×TP configuration
- Multi-rank parity test matches single-device reference within tolerance

---

## Phase 7: Parity Test Migration to GlobalOrchestrator

**Goal**: Migrate existing NodeLocalTP parity tests to use `GlobalOrchestrator` instead of ad-hoc factory wiring, and add a new NodeLocalPP parity test. This ensures parity tests exercise the production code path end-to-end.

### Background

The current NodeLocalTP parity tests (`Test__Qwen3_NodeLocalTP_Parity`, `Test__Qwen35_NodeLocalTP_Parity`) work around the absence of `GlobalOrchestrator` by:
1. Calling `InferenceRunnerFactory::createInferenceRunner()` directly with an `mpi_ctx`
2. The factory detects `world_size > 1` and applies global TP weight sharding internally
3. Each rank gets a `DeviceGraphOrchestrator` — there is no orchestrator coordinating across ranks
4. A separate `GlobalTPContext::createForTest()` is created just for infrastructure tests

This works for pure global TP (no PP, all ranks run all layers), but:
- It bypasses the production `GlobalOrchestrator` → `RankOrchestrator` → `DeviceGraphOrchestrator` stack
- It can't test PP at all (different ranks running different layers)
- If the production path diverges from the test wiring, parity tests won't catch it

### Tasks

- [ ] **7.1** Add `GlobalOrchestrator`-aware pipeline setup path to `ConfigDrivenParityTest`
  - New method `setupGlobalOrchestratorPipeline()` that builds `GlobalPPTopology` → `GlobalOrchestrator`
  - Route `Parallelism::NodeLocalTP` and `Parallelism::GlobalTP` configs through this path
  - `GlobalOrchestrator` wraps the per-rank `DeviceGraphOrchestrator` (created via existing factory)
  - The `GlobalTPContext` is owned by `GlobalOrchestrator`, not created separately
- [ ] **7.2** Migrate `Test__Qwen3_NodeLocalTP_Parity` to use GlobalOrchestrator path
  - `setupPipeline()` → `setupGlobalOrchestratorPipeline()` → `GlobalOrchestrator` as runner
  - Verify same parity results (cosine similarity, KL divergence) as before migration
  - Infrastructure tests (allreduce, broadcast, barrier) use `GlobalOrchestrator::globalTPContext()`
- [ ] **7.3** Migrate `Test__Qwen35_NodeLocalTP_Parity` (same changes, all 3 model configs)
- [ ] **7.4** Add `Parallelism::NodeLocalPP` enum value and test config support
  - 2-rank PP on a single machine (rank 0 = first half of layers on socket 0, rank 1 = second half on socket 1)
  - Embedding on rank 0, LM head on rank 1
  - MPI send/recv of activations between ranks (exercises `GlobalOrchestrator::executeTransfer()`)
- [ ] **7.5** Create `Test__Qwen3_NodeLocalPP_Parity` test file
  - Config: `{ Parallelism::NodeLocalPP, Collective::MPI, mpi_ranks=2, model="Qwen3-0.6B-Q8_0.gguf" }`
  - Tests: `PrefillParity`, `DecodeParity` — compare against single-device PyTorch reference
  - Verify greedy token predictions match single-device (PP should be numerically identical for same-precision weights)
- [ ] **7.6** Create `Test__Qwen35_NodeLocalPP_Parity` (0.8B model, 2-rank PP)
- [ ] **7.7** Remove old `setupGlobalTPPipeline()` and `GlobalTPContext::createForTest()` usage from migrated tests
- [ ] **7.8** Full parity test suite pass (all existing + new tests)

### NodeLocalPP Topology Construction

For a model with N layers on 2 ranks:
```
GlobalPPStageSpec stage_0: layers [0, N/2-1], has_embedding=true,  owning_rank=0
GlobalPPStageSpec stage_1: layers [N/2, N-1], has_lm_head=true,    owning_rank=1
Transfer: rank 0 → rank 1 (MPI send/recv of hidden state)
```

### Acceptance Criteria

- All migrated NodeLocalTP parity tests produce identical results to before migration
- New NodeLocalPP parity tests pass with greedy token prediction match
- No test creates `GlobalTPContext::createForTest()` for production-path TP — all TP contexts come from `GlobalOrchestrator`
- `grep -r 'createForTest' tests/v2/integration/parity/` returns zero hits (after migration)
- 2-rank PP parity shows cosine similarity > 0.999 vs single-device at every layer

---

## Future Work (Not In Scope)

These items are tracked but deferred beyond this project:

| Item | Reason to Defer |
|------|----------------|
| **Micro-batch pipeline scheduling (1F1B)** | Requires overlap of compute and communication; async primitives exist but scheduling framework does not. Layer on top of naive sequential PP. |
| **Heterogeneous cross-machine AllReduce** | Only needed for CUDA+ROCm across machines. Stubbed in `BackendRouter.cpp:500`. Rare scenario. |
| **InfiniBand/RoCE tuning knobs** | NCCL auto-detects transport. Add CLI flags (`--nccl-ib-hca`, `--nccl-socket-ifname`) as needed. |
| **GPU-aware MPI** | MPIBackend uses host staging by design. NCCL/RCCL handle GPU→GPU directly. |
| **KV cache migration** | PP stages have disjoint layers, so each stage owns its own KV cache. Migration only needed if layer assignment changes at runtime (dynamic load balancing). |
| **Speculative decoding across ranks** | Requires draft model on separate rank(s). Orthogonal to GlobalOrchestrator. |

---

## Test Plan

### Unit Tests (`tests/v2/unit/execution/global/`)

| Test | Description |
|------|-------------|
| `Test__GlobalOrchestrator_Construction` | Config validation, topology parsing |
| `Test__GlobalOrchestrator_PureTp` | Global TP without PP (all ranks, all layers) |
| `Test__GlobalOrchestrator_PurePp` | 2-way PP (rank 0 = layers 0-13, rank 1 = layers 14-27) |
| `Test__GlobalOrchestrator_TpPp` | 4-rank: 2 PP × 2 TP |
| `Test__GlobalOrchestrator_HeadTail` | Pipeline head/tail detection |
| `Test__GlobalOrchestrator_TokenBroadcast` | Tail rank broadcasts sampled token |
| `Test__GlobalOrchestrator_ClearCache` | All ranks clear KV cache with barrier |

### Integration Tests (`tests/v2/integration/global/`)

| Test | Description |
|------|-------------|
| `Test__GlobalPP_TwoRank_Qwen2` | `mpirun -np 2 --pp 2` with Qwen2.5-0.5B, greedy match |
| `Test__GlobalTP_TwoRank_Qwen2` | `mpirun -np 2 --tp 2 --tp-scope global` with Qwen2.5-0.5B |
| `Test__GlobalTpPp_FourRank_Qwen2` | `mpirun -np 4 --pp 2 --tp 2` with Qwen2.5-7B |

### Parity Tests (`tests/v2/integration/parity/`)

| Test | Description |
|------|-------------|
| `V2_Parity_GlobalPP_Qwen2_CPU` | 2-rank PP parity vs single-device PyTorch |
| `V2_Parity_GlobalTP_Qwen2_CUDA` | 2-rank TP parity vs single-device PyTorch |
| `V2_Parity_NodeLocalTP_Qwen3` | Migrated: 2-rank TP via GlobalOrchestrator (was direct factory) |
| `V2_Parity_NodeLocalTP_Qwen35` | Migrated: 2-rank TP via GlobalOrchestrator (0.8B, 4B, 27B) |
| `V2_Parity_NodeLocalPP_Qwen3` | **NEW**: 2-rank PP (half layers per socket) via GlobalOrchestrator |
| `V2_Parity_NodeLocalPP_Qwen35` | **NEW**: 2-rank PP via GlobalOrchestrator (0.8B) |

---

## Dependency Graph

```
Phase 0 (Rename)
    │
    ▼
Phase 1 (Core Structure)
    │
    ├──────────────┐
    ▼              ▼
Phase 2 (PP)   Phase 3 (TP+PP)
    │              │
    └──────┬───────┘
           ▼
    Phase 4 (Factory)
           │
           ▼
    Phase 5 (Retire Legacy)
           │
           ▼
    Phase 6 (Hardening)
           │
           ▼
    Phase 7 (Parity Migration)
```

Phase 2 and Phase 3 can proceed in parallel after Phase 1.
Phase 3 depends on Phase 2 only for the integration test (not implementation).
Phase 7 depends on Phase 4 (factory integration) so that parity tests can
construct GlobalOrchestrator through the production path. Phase 6 is a soft
dependency — hardening is nice-to-have before parity migration but not blocking.
