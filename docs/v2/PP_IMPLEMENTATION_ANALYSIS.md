# Pipeline Parallel Implementation Analysis

## Executive Summary

Llaminar V2 has **substantial PP infrastructure already in place**, but it's not fully wired together. The existing components are well-designed and tested in isolation. What's missing is the final integration work to connect these pieces for actual distributed PP execution.

## Current State Assessment

### ✅ What's Already Implemented

| Component | Location | Status |
|-----------|----------|--------|
| **PipelineParallelConfig** | `src/v2/config/PipelineParallelConfig.h` | Complete |
| **PPStageDefinition** | `src/v2/config/OrchestrationConfig.h` | Complete |
| **SendActivationsStage** | `src/v2/execution/compute_stages/stages/SendActivationsStage.h` | Complete |
| **ReceiveActivationsStage** | `src/v2/execution/compute_stages/stages/ReceiveActivationsStage.h` | Complete |
| **IPipelineParallelGraphBuilder** | `src/v2/execution/local_execution/graph/IPipelineParallelGraphBuilder.h` | Complete |
| **PipelineParallelGraphBuilder** | `src/v2/execution/local_execution/graph/PipelineParallelGraphBuilder.cpp` | Complete |
| **RankExecutionPlan PP fields** | `prev_rank`, `next_rank`, `pp_stage_id`, `first_layer`, `last_layer` | Complete |
| **CLI parsing for PP** | `--pp-stage`, `--define-domain` | Complete |
| **Unit tests for PP stages** | `Test__PPActivationStages.cpp`, `Test__PipelineParallelGraphBuilder.cpp` | Complete |

### ❌ What's Missing / Stub-Only

1. **OrchestrationRunner::sendActivationsToNextStage()** - Currently just logs, doesn't actually MPI_Send
2. **OrchestrationRunner::receiveActivationsFromPrevStage()** - Currently just logs, doesn't actually MPI_Recv
3. **buildComputeGraph() doesn't call insertPPStages()** - PP stages not wired into the compute graph
4. **Partial model loading** - No support for loading only assigned layers (`first_layer` to `last_layer`)
5. **Token broadcasting** - Last PP stage samples token but doesn't broadcast to other ranks
6. **Activation buffer management** - No shared buffer for PP communication

---

## Gap Analysis by Feature

### Gap 1: PP Communication Stubs

**Current state** (OrchestrationRunner.cpp lines 866-890):
```cpp
void OrchestrationRunner::sendActivationsToNextStage()
{
    if (!plan_.next_rank.has_value()) return;
    // TODO: Implement MPI_Send for activations
    LOG_DEBUG("PP: Would send activations to rank " << *plan_.next_rank);
}

void OrchestrationRunner::receiveActivationsFromPrevStage()
{
    if (!plan_.prev_rank.has_value()) return;
    // TODO: Implement MPI_Recv for activations
    LOG_DEBUG("PP: Would receive activations from prev_rank");
}
```

**What's needed:**
- Get the hidden states tensor from the runner
- Determine buffer format (FP32 hidden states: `[batch_size, seq_len, hidden_dim]`)
- Use MPI_Send/MPI_Recv with proper tags (use `makePPTag()` from PipelineParallelGraphBuilder.cpp)

### Gap 2: Graph Building Doesn't Use PP Infrastructure

**Current state** (OrchestrationRunner.cpp):
```cpp
bool OrchestrationRunner::buildComputeGraph()
{
    if (hasLocalTP()) {
        return buildMultiDeviceComputeGraph();  // No PP integration
    }
    return buildSingleDeviceComputeGraph();      // No PP integration
}
```

**The existing `PipelineParallelGraphBuilder` is complete but never called!**

**What's needed:**
```cpp
// After building the base graph:
if (plan_.next_rank.has_value() || plan_.prev_rank.has_value()) {
    auto pp_builder = createPipelineParallelGraphBuilder(mpi_ctx_);
    pp_builder->insertPPStages(graph, plan_, input_buffer, output_buffer);
}
```

### Gap 3: Partial Model Loading

For PP to work, each rank should only load its assigned layers.

**Current state:** OrchestrationRunner always loads the full model.

**What's needed:**
- ModelLoader needs to accept layer range parameters
- `loadWeights()` should pass `plan_.first_layer` and `plan_.last_layer`
- Need to handle embedding (only on head stage) and LM head (only on tail stage)

### Gap 4: Token Broadcasting

After the tail stage samples a token, all ranks need to know it for the next decode step.

**Current state:** Only tail stage has the token, other stages don't get it.

**What's needed:**
```cpp
// In decodeStep():
if (isPipelineTail()) {
    int token = sampler_.sample(logits_vec, params);
    // Broadcast to all ranks
    MPI_Bcast(&token, 1, MPI_INT, mpi_ctx_->world_size() - 1, MPI_COMM_WORLD);
} else {
    // Non-tail stages receive the token
    MPI_Bcast(&token, 1, MPI_INT, mpi_ctx_->world_size() - 1, MPI_COMM_WORLD);
}
```

---

## Implementation Plan

### Phase 1: Simple PP (Single Node, Multiple GPUs)

**Goal:** Support `--pp 2 --pp-devices "cuda:0,cuda:1"` to split layers across 2 GPUs.

**Key difference from multi-rank PP:** No MPI, use direct device-to-device activation transfer via existing collective backends.

#### Existing Backend Infrastructure for PP Activation Transfer

**Great news:** The `ICollectiveBackend` interface already has P2P operations we can leverage!

From `src/v2/collective/ICollectiveBackend.h`:
```cpp
// Point-to-point (sync)
virtual bool send(void *buffer, size_t count, CollectiveDataType dtype, int peer, int tag = 0);
virtual bool recv(void *buffer, size_t count, CollectiveDataType dtype, int peer, int tag = 0);
virtual bool sendrecv(void *sendbuf, void *recvbuf, size_t count, CollectiveDataType dtype, int peer);

// Point-to-point (async, stream-based)
virtual bool sendAsync(void *buffer, size_t count, CollectiveDataType dtype, int peer, void *stream, int tag = 0);
virtual bool recvAsync(void *buffer, size_t count, CollectiveDataType dtype, int peer, void *stream, int tag = 0);
virtual bool sendrecvAsync(void *sendbuf, void *recvbuf, size_t count, CollectiveDataType dtype, int peer, void *stream);

// Broadcast (one-to-all)
virtual bool broadcast(void *buffer, size_t count, CollectiveDataType dtype, int root_rank);
```

#### Backend Support Matrix for PP

| Scenario | Backend | Method | Implementation Status |
|----------|---------|--------|----------------------|
| **CUDA↔CUDA** | NCCL | `ncclSend/ncclRecv` | ✅ Supported via NCCL API |
| **ROCm↔ROCm** | RCCL | `rcclSend/rcclRecv` | ✅ Loaded via RCCLDynamicLoader |
| **CUDA↔ROCm** | PCIeBAR | BAR memory mapping | ⚠️ Partial: CUDA→ROCm works, ROCm→CUDA TODO |
| **GPU↔CPU** | HOST | Host staging | ✅ Use `cudaMemcpyAsync(Host)` |
| **CPU↔CPU** (cross-socket) | MPI | `MPI_Send/MPI_Recv` | ✅ Already works |

#### PCIeBAR P2P Implementation (Cross-Vendor)

From `src/v2/collective/backends/PCIeBARBackend.cpp`:
```cpp
// send() from CUDA to ROCm works via BAR write
if (peer == 1) {  // CUDA (rank 0) sending to ROCm (rank 1)
    transferCUDAtoROCm(buffer, 0, bytes);  // Write to BAR
}

// recv() on CUDA from ROCm works via BAR read
if (peer == 1) {  // CUDA (rank 0) receiving from ROCm (rank 1)
    transferROCmtoCUDA(0, buffer, bytes);  // Read from BAR
}
```

**Gap:** ROCm→CUDA direction not fully implemented yet (ROCm side send/recv need work).

#### Activation Transfer Strategy for PP

For PP activation transfer between stages, we can use **broadcast** semantics:
- Stage N produces activations on device D_N
- Stage N+1 consumes activations on device D_N+1
- Use `backend->broadcast(activations, count, dtype, root=N)` where N is the sending stage

**Why broadcast instead of send/recv?**
1. Simpler API - no need for matching send/recv pairs
2. Works with any backend (all support broadcast)
3. For 2-device PP, broadcast degenerates to a single transfer
4. Naturally extends to 1:N patterns if needed (one sender, multiple receivers)

#### Implementation Tasks

| Task | Effort | Dependencies |
|------|--------|--------------|
| Add `--pp-devices` CLI option (similar to `--tp-devices`) | Low | None |
| Create `IPPActivationTransfer` interface | Low | None |
| Implement `CollectiveBackendPPTransfer` using backend->broadcast() | Low | IPPActivationTransfer |
| Create `SingleNodePPOrchestrator` | Medium | IPPActivationTransfer |
| Integrate partial graph building per PP stage | High | None |
| Wire activation transfer stages into graph | Low | PipelineParallelGraphBuilder (exists) |
| Integration tests | Medium | All above |

#### Proposed IPPActivationTransfer Interface

```cpp
class IPPActivationTransfer {
public:
    virtual ~IPPActivationTransfer() = default;
    
    // Transfer activations from stage_from to stage_to
    // Uses the appropriate backend based on device types
    virtual bool transfer(
        TensorBase* activations,
        int stage_from,  // PP stage index (device index)
        int stage_to,
        void* stream = nullptr  // Optional async stream
    ) = 0;
    
    // Async version with explicit stream
    virtual bool transferAsync(
        TensorBase* activations,
        int stage_from,
        int stage_to,
        void* stream
    ) = 0;
    
    virtual void synchronize() = 0;
};
```

#### Backend Selection Logic for PP

```cpp
CollectiveBackendType selectPPBackend(DeviceType src, DeviceType dst) {
    if (src == DeviceType::CUDA && dst == DeviceType::CUDA)
        return CollectiveBackendType::NCCL;
    if (src == DeviceType::ROCm && dst == DeviceType::ROCm)
        return CollectiveBackendType::RCCL;
    if ((src == DeviceType::CUDA && dst == DeviceType::ROCm) ||
        (src == DeviceType::ROCm && dst == DeviceType::CUDA))
        return CollectiveBackendType::PCIE_BAR;
    if (src == DeviceType::CPU || dst == DeviceType::CPU)
        return CollectiveBackendType::HOST;
    return CollectiveBackendType::MPI;  // Fallback
}
```

### Phase 2: Multi-Rank PP (MPI Distributed)

**Goal:** Support scenario where each PP stage is a separate MPI rank.

| Task | Effort | Dependencies |
|------|--------|--------------|
| Implement `sendActivationsToNextStage()` | Low | None |
| Implement `receiveActivationsFromPrevStage()` | Low | None |
| Add partial model loading | Medium | ModelLoader changes |
| Add token broadcasting | Low | None |
| Wire `insertPPStages()` into buildComputeGraph | Low | PipelineParallelGraphBuilder (exists) |
| Integration tests with mpirun | Medium | All above |

### Phase 3: Hybrid PP+TP (Scenario 7)

**Goal:** Support complex configurations like:
```bash
--define-domain "gpu0=0:cuda:0,0:cuda:1"
--define-domain "gpu1=1:rocm:0,1:rocm:1"
--define-domain "cpu=cpu:0,cpu:1"
--pp-stage "0=gpu0:0-7"
--pp-stage "1=gpu1:8-15"
--pp-stage "2=cpu:16-23"
```

**This is mostly configuration/planning work - the execution primitives exist.**

| Task | Effort | Dependencies |
|------|--------|--------------|
| ExecutionPlanBuilder handles named domains | Medium | Partially done |
| Map domains to ranks/devices | Medium | DeviceInventory exists |
| Create per-stage LocalTPContext | Medium | LocalTPContext exists |
| Wire together TP reduction + PP send | High | Phase 2 complete |

---

## Cross-Socket PP (TP Domain Boundary)

When a PP stage boundary crosses socket boundaries (e.g., PP from a `TP(Socket0 × Socket1)` domain), MPI is the right choice:

```
┌────────────────────────────────────────────────────────────────┐
│ Node 0                                                          │
│ ┌─────────────────────────┐    ┌─────────────────────────────┐ │
│ │ Socket 0                │    │ Socket 1                     │ │
│ │ ┌───────┐  ┌───────┐    │    │ ┌───────┐  ┌───────┐        │ │
│ │ │CUDA:0 │  │CUDA:1 │    │    │ │ROCm:0 │  │ROCm:1 │        │ │
│ │ └───┬───┘  └───┬───┘    │    │ └───┬───┘  └───┬───┘        │ │
│ │     │          │        │    │     │          │            │ │
│ │     └────┬─────┘        │    │     └────┬─────┘            │ │
│ │          │ NCCL         │    │          │ RCCL             │ │
│ │          ▼              │    │          ▼                  │ │
│ │   TP Domain 0           │    │   TP Domain 1               │ │
│ │   (PP Stage 0)          │    │   (PP Stage 1)              │ │
│ │   Layers 0-11           │    │   Layers 12-23              │ │
│ └─────────────────────────┘    └─────────────────────────────┘ │
│               │                              ▲                  │
│               └──────── MPI_Send/Recv ───────┘                  │
│                    (Cross-socket PP)                            │
└────────────────────────────────────────────────────────────────┘
```

**Communication Pattern:**
1. **Intra-domain TP**: NCCL/RCCL/PCIeBAR for allreduce within each TP domain
2. **Inter-domain PP**: MPI for activation transfer between PP stages
3. **Cross-vendor within domain**: PCIeBAR if mixing CUDA+ROCm in same TP domain

**This matches the existing SendActivationsStage/ReceiveActivationsStage design which uses MPI.**

---

## Scenario 7 Deep Dive: PP(TP + TP + TP)

**Hardware** (from `Test__Scenario7_HeterogeneousPipelineCluster.cpp`):
- 2 physical nodes × 2 sockets each = 4 MPI ranks
- Each socket: 1× RTX 3090 (CUDA) + 1× MI50 (ROCm)
- InfiniBand interconnect between nodes

**Domain Structure per node (3 PP stages):**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Node 0 (PP Stage 0 of inter-node pipeline)                                  │
│                                                                             │
│  ┌─────────────────────────────┐    ┌─────────────────────────────┐        │
│  │ PP Stage 0 (layers 0-4)     │    │ PP Stage 1 (layers 5-9)     │        │
│  │ ┌─────────┐  ┌─────────┐    │    │ ┌─────────┐  ┌─────────┐    │        │
│  │ │ 3090_0  │  │ MI50_0  │    │    │ │ 3090_1  │  │ MI50_1  │    │        │
│  │ │ (CUDA)  │  │ (ROCm)  │    │    │ │ (CUDA)  │  │ (ROCm)  │    │        │
│  │ └────┬────┘  └────┬────┘    │    │ └────┬────┘  └────┬────┘    │        │
│  │      │   PCIeBAR  │         │    │      │   PCIeBAR  │         │        │
│  │      └──────┬─────┘         │    │      └──────┬─────┘         │        │
│  │      LocalTPContext         │    │      LocalTPContext         │        │
│  │      (GPU_TP_0, 2 devices)  │    │      (GPU_TP_1, 2 devices)  │        │
│  │             │                │    │             │                │        │
│  │      MPI Rank 0             │    │      MPI Rank 1             │        │
│  └──────────────┼──────────────┘    └──────────────┼──────────────┘        │
│                 │                                   │                       │
│                 │    PCIeBAR/HOST PP transfer       │                       │
│                 └───────────────┬───────────────────┘                       │
│                                 │                                           │
│                                 ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ PP Stage 2 (layers 10-13)                                             │  │
│  │ ┌──────────────────────────────────────────────────────────────────┐ │  │
│  │ │                    CPU TP Domain                                  │ │  │
│  │ │    ┌───────────────┐              ┌───────────────┐               │ │  │
│  │ │    │   Socket 0    │              │   Socket 1    │               │ │  │
│  │ │    │   (Rank 0)    │◄────UPI─────►│   (Rank 1)    │               │ │  │
│  │ │    └───────────────┘              └───────────────┘               │ │  │
│  │ │                    LocalTPContext (CPU_TP, 2 ranks via MPI)       │ │  │
│  │ └──────────────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                 │                                           │
│                      MPI PP to Node 1 (InfiniBand)                         │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  ▼
                            Node 1 (mirror layout)
```

### Communication Backends by Stage Transition

| Transition | Transfer Type | Backend | Notes |
|------------|--------------|---------|-------|
| **Within GPU_TP_0** | TP allreduce | PCIeBAR | 3090↔MI50 cross-vendor |
| **Within GPU_TP_1** | TP allreduce | PCIeBAR | 3090↔MI50 cross-vendor |
| **GPU_TP_0 → GPU_TP_1** | PP activation | HOST or PCIeBAR | Same node, cross-socket |
| **GPU_TP_1 → CPU_TP** | PP activation | HOST | GPU→CPU, host staging |
| **Within CPU_TP** | TP allreduce | MPI (UPI) | 2 ranks, same node |
| **Node 0 → Node 1** | PP activation | MPI (InfiniBand) | Cross-node |

### Execution Flow for One Forward Pass

```
1. GPU_TP_0 (Rank 0):
   - Embedding lookup
   - For each layer 0-4:
     - Column-parallel Q/K/V projection (both GPUs work on partial heads)
     - Attention (local to each GPU's heads)
     - Row-parallel Wo projection
     - PCIeBAR allreduce (3090↔MI50)  ← LocalTPContext
     - Column-parallel FFN up/gate
     - Row-parallel FFN down  
     - PCIeBAR allreduce (3090↔MI50)  ← LocalTPContext
   - PP send activations to GPU_TP_1  ← LocalPPContext (HOST backend)

2. GPU_TP_1 (Rank 1):
   - PP recv activations from GPU_TP_0
   - For each layer 5-9:
     - Same TP pattern with PCIeBAR allreduce
   - PP send activations to CPU_TP  ← LocalPPContext (HOST backend)

3. CPU_TP (Ranks 0+1):
   - PP recv activations (from GPU to host memory)
   - For each layer 10-13:
     - Column-parallel projections (each rank processes partial heads)
     - Attention (local to each rank's heads)
     - Row-parallel projections
     - MPI allreduce (over UPI)  ← LocalTPContext
   - LM head (if last PP stage of node)
   - PP send activations to Node 1  ← MPI SendActivationsStage
```

### Architecture Support Assessment

**✅ Fully Supported with Proposed Architecture:**

| Component | How It's Supported |
|-----------|-------------------|
| **Per-stage LocalTPContext** | Each PP stage creates its own LocalTPContext with appropriate backend |
| **TP allreduce within stage** | LocalTPContext.allreduce() - already implemented |
| **PP transfer between GPU domains** | LocalPPContext using HOST/PCIeBAR backends |
| **PP transfer GPU→CPU** | LocalPPContext using HOST backend |
| **PP transfer cross-node** | MPI SendActivationsStage/ReceiveActivationsStage - already exists |
| **Mixed backends per stage** | Each LocalTPContext/LocalPPContext selects backend based on device types |

### Key Insight: Separation of Concerns

The architecture naturally separates:

1. **LocalTPContext** - Handles TP allreduce **within** a PP stage
   - One per PP stage
   - Backend selected based on devices in that stage (NCCL/RCCL/PCIeBAR/MPI)

2. **LocalPPContext** (new) - Handles activation transfer **between** PP stages on same node
   - One per node (manages all local PP transfers)
   - Backend selected based on src/dst device types

3. **SendActivationsStage/ReceiveActivationsStage** - Handles PP **between** nodes
   - Uses MPI
   - Already implemented

### CLI for Scenario 7

```bash
# Per-node (run on each node with different rank ranges)
mpirun -np 4 \
  --map-by socket --bind-to socket \
  ./llaminar2 \
    --define-domain "gpu_tp_0=0:cuda:0,0:rocm:0" \
    --define-domain "gpu_tp_1=1:cuda:0,1:rocm:0" \
    --define-domain "cpu_tp=cpu:0,cpu:1" \
    --pp-stage "0=gpu_tp_0:0-4" \
    --pp-stage "1=gpu_tp_1:5-9" \
    --pp-stage "2=cpu_tp:10-13" \
    -m models/qwen2-72b-q4_0.gguf \
    -p "Hello, world!"
```

The orchestrator would:
1. Parse domain definitions → 3 domains with different backends
2. Parse PP stage definitions → 3 stages with layer assignments
3. For each rank, build execution plan with its assigned domain(s)
4. Create appropriate LocalTPContext per domain
5. Create LocalPPContext for intra-node PP transfers
6. Wire PP communication into compute graph

---

## Single-Node PP Communication Architecture

For single-node multi-GPU PP (Phase 1), we leverage the existing `ICollectiveBackend` infrastructure:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Single Node: 4-GPU PP Pipeline                                          │
│                                                                         │
│  PP Stage 0         PP Stage 1         PP Stage 2         PP Stage 3   │
│  ┌─────────┐        ┌─────────┐        ┌─────────┐        ┌─────────┐  │
│  │ CUDA:0  │        │ CUDA:1  │        │ ROCm:0  │        │ ROCm:1  │  │
│  │         │        │         │        │         │        │         │  │
│  │ Embed   │        │         │        │         │        │ LM Head │  │
│  │ Layer0-5│        │Layer6-11│        │Layer12-17│       │Layer18-23│  │
│  └────┬────┘        └────┬────┘        └────┬────┘        └─────────┘  │
│       │                  │                  │                   ▲       │
│       │    NCCL          │    PCIeBAR       │    RCCL          │       │
│       │    send()        │    broadcast()   │    send()        │       │
│       └─────────────────►└─────────────────►└──────────────────┘       │
│                                                                         │
│  Backend Selection:                                                     │
│  • CUDA→CUDA: NCCL (ncclSend/ncclRecv)                                 │
│  • CUDA→ROCm: PCIeBAR (BAR memory write)                               │
│  • ROCm→ROCm: RCCL (rcclSend/rcclRecv)                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### LocalPPContext Design

Similar to `ILocalTPContext` for TP, we can create a `LocalPPContext` for single-node PP:

```cpp
/**
 * @brief Context for single-node pipeline parallelism
 * 
 * Manages activation transfer between PP stages using appropriate backends.
 * Uses the same backend infrastructure as LocalTPContext but for P2P transfers.
 */
class LocalPPContext {
public:
    struct Config {
        std::vector<GlobalDeviceAddress> stage_devices;  // One device per PP stage
        std::vector<int> layer_boundaries;  // [0, 6, 12, 18, 24] for 4 stages
    };
    
    // Transfer activations from stage_from to stage_to
    bool transferActivations(
        TensorBase* activations,
        int stage_from,
        int stage_to
    );
    
    // Async transfer with stream
    bool transferActivationsAsync(
        TensorBase* activations,
        int stage_from,
        int stage_to,
        void* stream
    );
    
private:
    // Backend selection per stage pair
    ICollectiveBackend* getBackendForTransfer(int from, int to);
    
    // Backends
    std::unique_ptr<ICollectiveBackend> nccl_backend_;   // CUDA↔CUDA
    std::unique_ptr<ICollectiveBackend> rccl_backend_;   // ROCm↔ROCm  
    std::unique_ptr<ICollectiveBackend> pciebar_backend_; // CUDA↔ROCm
    std::unique_ptr<ICollectiveBackend> host_backend_;   // GPU↔CPU fallback
};
```

---

## Architecture Decision: Where to Put PP Integration

### Option A: Extend OrchestrationRunner (Recommended for Phase 2)

Keep the current structure, just fill in the stubs.

**Pros:**
- Minimal code changes
- Reuses existing send/receive stage infrastructure
- Works with both single-device and multi-device runners

**Cons:**
- OrchestrationRunner becomes more complex
- PP logic scattered between graph building and runtime

### Option B: Create PPOrchestrator (Recommended for Phase 1)

New orchestrator that manages PP-specific logic for single-node multi-GPU PP.

```cpp
class SingleNodePPOrchestrator : public IInferenceRunner {
    std::vector<std::unique_ptr<GraphExecutor>> stage_executors_;
    std::vector<TensorBase*> activation_buffers_;
    
    bool forward(const int* tokens, int seq_len) override {
        // Stage 0: embedding + layers 0-N
        stage_executors_[0]->execute();
        
        // Transfer activations
        transferActivations(0, 1);
        
        // Stage 1: layers N+1 to M + LM head
        stage_executors_[1]->execute();
    }
};
```

---

## Key Files to Modify

### For Simple PP (Single-Node)

1. `src/v2/execution/runner/OrchestrationRunner.cpp`
   - Wire `insertPPStages()` into graph building
   - Add activation buffer management

2. `src/v2/execution/local_execution/orchestrators/SingleNodePPOrchestrator.cpp` (new)
   - Multi-GPU PP without MPI

3. `src/v2/config/OrchestrationConfig.h`
   - Add `pp_devices` for single-node PP

### For Multi-Rank PP

1. `src/v2/execution/runner/OrchestrationRunner.cpp`
   - Implement `sendActivationsToNextStage()`
   - Implement `receiveActivationsFromPrevStage()`
   - Add token broadcasting

2. `src/v2/loaders/ModelLoader.cpp`
   - Support partial model loading (layer range)

### For Scenario 7 (Hybrid PP+TP)

1. `src/v2/execution/mpi_orchestration/ExecutionPlanBuilder.cpp`
   - Better domain-to-rank mapping

2. `src/v2/execution/runner/OrchestrationRunner.cpp`
   - Integrate TP reduction with PP send

---

## Recommended Next Steps

1. **Start with multi-rank PP** (Phase 2) - it's the simplest to implement because:
   - SendActivationsStage and ReceiveActivationsStage already exist and are tested
   - PipelineParallelGraphBuilder already exists and is tested
   - Just need to wire them together

2. **Add a simple integration test first:**
   ```cpp
   // Simulated 2-rank PP test with thread-based MPI simulation
   TEST(Test__PipelineParallel, TwoRankPP_Works) {
       // Rank 0: embedding + layers 0-11, send to rank 1
       // Rank 1: receive, layers 12-23 + LM head
   }
   ```

3. **Then tackle single-node PP** - requires more infrastructure for efficient GPU-GPU transfer

---

## Existing Test Coverage

| Test File | Coverage |
|-----------|----------|
| `Test__PPActivationStages.cpp` | Send/Receive stage unit tests |
| `Test__PipelineParallelGraphBuilder.cpp` | Graph builder unit tests |
| `Test__Scenario7_HeterogeneousPipelineCluster.cpp` | Config parsing + plan building |
| `Test__Scenario7_MultiDomainPP.cpp` | Full integration (mostly skipped) |

The integration tests are mostly skipped because they need real device discovery. Once the wiring is complete, we can enable them.

---

## Estimated Effort

| Phase | Effort | Complexity |
|-------|--------|------------|
| Phase 1 (Single-node PP via backends) | 3-4 days | Medium |
| Phase 2 (Multi-rank MPI PP) | 2-3 days | Medium |
| Phase 3 (Hybrid PP+TP) | 3-5 days | High |

**Total: 8-12 days** to full PP support

The infrastructure is ~70% complete. The remaining work is mostly integration and wiring.

---

## Recommended Implementation Order

**Recommendation: Start with Phase 1** (single-node PP via collective backends)

**Rationale:**
1. **Reuses proven infrastructure** - NCCL/RCCL/PCIeBAR backends already tested for TP
2. **No MPI complexity** - Single-node testing is simpler
3. **Higher performance** - Direct GPU-GPU transfer avoids host staging
4. **Natural stepping stone** - Once LocalPPContext works, extending to MPI is straightforward

**Alternative view (if prioritizing distributed first):**
- Phase 2 (MPI PP) could be done first since SendActivationsStage/ReceiveActivationsStage exist
- But the MPI stages are designed for multi-process, not single-process multi-GPU

**Phase 1 unlocks:**
- `./llaminar2 --pp 2 --pp-devices "cuda:0,cuda:1" -m model.gguf`
- `./llaminar2 --pp 4 --pp-devices "cuda:0,cuda:1,rocm:0,rocm:1" -m model.gguf`

**Phase 2 then adds:**
- `mpirun -np 2 ./llaminar2 --pp 2 -m model.gguf` (each rank is one PP stage)

**Phase 3 composes them:**
- Each MPI rank can internally use LocalPPContext for multi-GPU within the rank
- Or use LocalTPContext for TP within the rank
- PP between ranks uses MPI

---

## Summary: Complete Backend Matrix

| Parallelism | Scope | Backend | Implementation |
|-------------|-------|---------|----------------|
| **TP allreduce** | CUDA↔CUDA same socket | NCCL | LocalTPContext ✅ |
| **TP allreduce** | ROCm↔ROCm same socket | RCCL | LocalTPContext ✅ |
| **TP allreduce** | CUDA↔ROCm same socket | PCIeBAR | LocalTPContext ✅ |
| **TP allreduce** | CPU↔CPU same node | MPI (UPI) | LocalTPContext ✅ |
| **PP transfer** | GPU→GPU same socket | NCCL/RCCL | LocalPPContext (new) |
| **PP transfer** | GPU→GPU cross-socket | HOST or PCIeBAR | LocalPPContext (new) |
| **PP transfer** | GPU→CPU same node | HOST | LocalPPContext (new) |
| **PP transfer** | CPU→CPU cross-socket | MPI (UPI) | LocalPPContext (new) |
| **PP transfer** | Any→Any cross-node | MPI (InfiniBand) | SendActivationsStage ✅ |

**Legend:**
- ✅ = Already implemented and tested
- (new) = To be implemented in Phase 1

The key insight is that **LocalPPContext** is the missing piece - it provides the same backend abstraction for PP transfers that LocalTPContext provides for TP collectives.
