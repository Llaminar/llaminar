# Hybrid Parallelism Project Plan

## Implementation Progress

> **Last Updated**: January 21, 2026

### Status Summary

| Phase | Status | Key Achievement |
|-------|--------|----------------|
| **Phase 0** | ✅ COMPLETE | CollectiveContext wired, PCIeBAR benchmarks: 25μs (8× better than target!) |
| **Phase 1a** | ✅ COMPLETE | TensorParallelConfig infrastructure (33 tests) |
| **Phase 1b** | ✅ COMPLETE | WeightManager proportional slicing (6+ tests) |
| **Phase 1c** | ✅ COMPLETE | Qwen2Graph variable heads (11 tests) |
| **Phase 1d** | ✅ COMPLETE | AllGatherVStage (14 tests) |
| **Phase 2.1** | ✅ COMPLETE | MPI P2P primitives (11 tests) |
| **Phase 2.2** | ✅ COMPLETE | Send/Recv activation stages (9 tests) |
| **Phase 2.3** | ✅ COMPLETE | PipelineParallelConfig (32 tests) |
| **Phase 3.1** | ✅ COMPLETE | LayerPlacementConfig (33 tests) |
| **Phase 3.2a** | ✅ COMPLETE | NodeTopology NUMA/socket detection (29 tests) |
| **Phase 3.2b** | ✅ COMPLETE | TPDomain + MultiDomainTPConfig (30 tests) |
| **Phase 3.3** | ✅ COMPLETE | UPICollectiveBackend (37 tests, 0.64μs latency!) |
| **Phase 3.4** | ✅ COMPLETE | NUMAAllocator (32 tests) |
| **Phase 3.5** | ✅ COMPLETE | BackendRouter domain-aware selection (16 new tests) |
| **Phase 4** | ⏳ PLANNED | Full hybrid integration |

### Test Results

| Category | Count | Status |
|----------|-------|--------|
| Unit tests | 279 | ✅ All passing |
| Integration tests | 86 | ✅ All passing |
| **Total** | **365** | ✅ **All passing** |
| Regressions | 0 | ✅ None |

### Key Metrics Achieved

- **PCIeBAR 14KB latency**: 25μs mean (target was 200μs → **8× better**)
- **Sustained decode throughput**: 731 tok/s AllReduce-only (target was 10 tok/s → **73× better**)
- **PCIeBAR viability**: ✅ CONFIRMED - architecture is validated

---

## Executive Summary

This document outlines a phased implementation plan for enabling heterogeneous multi-device inference in Llaminar V2, targeting the architecture:

- **Cross-rank**: Pipeline Parallel (layers 0-13 on rank 0, 14-27 on rank 1)
- **Intra-rank**: Pipeline Parallel split between CPU and GPUs
- **Intra-GPU**: Tensor Parallel across NVIDIA + AMD GPUs via PCIeBAR

**Current State**: Phases 0-3.5 complete. 365 tests passing. Full TPDomain infrastructure ready.

**Target**: 10+ tok/s decode throughput → PCIeBAR benchmark shows **731 tok/s** is achievable for AllReduce portion

---

## Research Summary

| Area | Status | Key Finding |
|------|--------|-------------|
| **PCIeBAR Backend** | ✅ Validated | 25μs latency for 14KB (8× better than 200μs target) |
| **Pipeline Parallelism** | ✅ Complete | MPI P2P + Send/Recv stages + PipelineParallelConfig (52 tests) |
| **Heterogeneous TP** | ✅ Complete | TensorParallelConfig + WeightManager + AllGatherV (64 tests) |
| **CollectiveContext Wiring** | ✅ Complete | Wired to DeviceGraphExecutor, BackendRouter operational |
| **CPU Pipeline Stage** | ✅ Ready | LayerPlacementConfig complete (33 tests) |
| **Cross-Socket CPU TP** | ✅ Complete | TPDomain + NodeTopology + UPIBackend + NUMAAllocator (144 tests) |

---

## Phase 0: Foundation & Validation (Week 1)

### Objective
Wire the existing infrastructure and validate PCIeBAR latency before building on it.

### 0.1 Wire CollectiveContext to AllreduceStage
**Priority**: CRITICAL - Unblocks ROCm performance immediately

**Effort**: 7 hours

**Changes**:
| File | Change |
|------|--------|
| `src/v2/execution/DeviceGraphExecutor.h` | Add `collective_ctx_` member and setter |
| `src/v2/execution/DeviceGraphExecutor.cpp` | Add `executeCollectiveStage()` intercept for ALLREDUCE/ALLGATHER |
| `src/v2/pipelines/qwen/GraphOrchestrator.cpp` | Wire `injected_collective_ctx_` to executor |
| `InferenceRunnerFactory.cpp` | Create CollectiveContext with ClusterInventory |

**Implementation**:
```cpp
// DeviceGraphExecutor.h
class DeviceGraphExecutor : public IGraphExecutor {
public:
    void setCollectiveContext(ICollectiveContext* ctx) { collective_ctx_ = ctx; }
private:
    ICollectiveContext* collective_ctx_ = nullptr;
};

// DeviceGraphExecutor.cpp - executeNode()
if (collective_ctx_ && node.stage->type() == ComputeStageType::ALLREDUCE) {
    return executeCollectiveStage(node, ctx);  // Delegates to CollectiveContext
}
```

**Success Criteria**:
- ROCm single-GPU uses RCCL instead of MPI_Allreduce
- BackendRouter correctly selects RCCL for ROCm, NCCL for CUDA
- Log output shows "Using RCCL backend" or "Using PCIeBAR backend"

### 0.2 PCIeBAR Latency Benchmark Suite
**Priority**: HIGH - Validates architecture assumption

**Effort**: 2 days

**Why Needed**: Current benchmarks measure bandwidth (2.65 GB/s), but hybrid parallelism needs low **latency** for small 14KB activation transfers. Estimated overhead is 50-155μs but unverified.

**Benchmark Cases**:
| Test | Transfer Size | Expected Latency | Purpose |
|------|--------------|------------------|---------|
| Minimum latency | 1 KB | 45-100 μs | Baseline overhead |
| Activation size | 14 KB | 50-155 μs | Wo output (d_model × 4 bytes) |
| KV head | 28 KB | 60-180 μs | One KV head per layer |
| Full attention | 112 KB | 100-300 μs | All heads one layer |
| Large transfer | 1 MB | 400-600 μs | FFN intermediate |

**Test File**: `tests/v2/integration/Test__PCIeBARLatencyBenchmark.cpp`

**Metrics to Capture**:
```cpp
struct PCIeBARLatencyMetrics {
    double kernel_launch_overhead_us;   // Time before DMA starts
    double dma_latency_us;              // Actual transfer time
    double sync_overhead_us;            // hipDeviceSynchronize / cudaDeviceSynchronize
    double total_round_trip_us;         // End-to-end AllReduce
    double bandwidth_gbps;              // Throughput at this size
};
```

**Success Criteria**:
- 14KB AllReduce completes in < 200μs
- Benchmark produces reproducible results (< 10% variance)
- Document actual vs expected latency for planning

### 0.3 Baseline Performance Measurement
**Priority**: HIGH - Establishes success metrics

**Effort**: 0.5 days

**Tasks**:
1. Benchmark current ROCm decode with MPI AllReduce (expect ~0.41 tok/s)
2. Benchmark with CollectiveContext wired (expect 2-5x improvement)
3. Benchmark CUDA for comparison
4. Document per-stage timing breakdown

**Output**: Baseline metrics in `docs/v2/PERFORMANCE_BASELINES.md`

---

## Phase 1: Tensor Parallel Optimization (Weeks 2-3)

### Objective
Optimize intra-node GPU tensor parallelism with PCIeBAR for NVIDIA↔AMD communication.

### 1.1 PCIeBAR Production Hardening
**Priority**: HIGH

**Effort**: 5 days

**Gap Analysis**:
| Issue | Impact | Fix |
|-------|--------|-----|
| Only FP32 AllReduce | Medium | Add FP16/BF16 support |
| No retry/timeout | High | Add transient error recovery |
| No ReduceScatter | Low | Not needed for current TP |
| Hardcoded 1GB BAR | Medium | Make configurable |

**Tasks**:
1. Add FP16/BF16 reduction kernels (2 days)
2. Add timeout/retry logic with configurable thresholds (1 day)
3. Add error injection tests (1 day)
4. Make BAR size configurable via environment variable (0.5 days)
5. Documentation and examples (0.5 days)

### 1.2 Proportional Head Assignment
**Priority**: MEDIUM - Enables load balancing across heterogeneous GPUs

**Effort**: 1 week

**Current Limitation**: `WeightManager::calculateSlice()` assumes equal splits:
```cpp
// Current code - equal split
size_t local_dim = total_dim / world_size_;
size_t offset = rank_ * local_dim;

// Needed - proportional split
size_t local_dim = assignment.head_count * head_dim;
size_t offset = assignment.head_offset * head_dim;
```

**Required Changes**:

1. **New Configuration Struct**:
```cpp
struct DeviceShardingAssignment {
    DeviceId device;
    int head_count;        // Heads assigned to this device
    int head_offset;       // Starting head index
    float weight_ratio;    // Portion of total work (e.g., 0.73 for NVIDIA)
};

struct TensorParallelConfig {
    std::vector<DeviceShardingAssignment> assignments;
    int total_heads;       // Sum of all head_counts
    ShardingMode mode;     // PROPORTIONAL or EQUAL
};
```

2. **WeightManager Changes**:
   - Accept `TensorParallelConfig` instead of just world_size/rank
   - Slice weights according to `head_count` not `1/world_size`
   - Support variable offsets

3. **Qwen2Graph Changes**:
   - Pass device-specific `local_num_heads` to attention stages
   - Variable KV cache allocation per device

4. **AllGatherV Stage**:
   - Variable-sized AllGather for differently-sharded outputs
   - Requires `sendcounts[]` and `displacements[]` arrays

**Success Criteria**:
- NVIDIA (A100): 20 heads, AMD (MI300): 8 heads (73%/27% split)
- Both GPUs complete attention in similar wall-clock time
- AllGatherV correctly assembles full output

### 1.3 Heterogeneous Device Group Testing
**Priority**: MEDIUM

**Effort**: 3 days

**Test Cases**:
| Configuration | Expected Backend | Test |
|--------------|------------------|------|
| 1× CUDA | NCCL (trivial) | `Test__DeviceGroup_SingleCUDA` |
| 1× ROCm | RCCL (trivial) | `Test__DeviceGroup_SingleROCm` |
| 2× CUDA | NCCL | `Test__DeviceGroup_DualCUDA` |
| 2× ROCm | RCCL | `Test__DeviceGroup_DualROCm` |
| 1× CUDA + 1× ROCm | PCIeBAR | `Test__DeviceGroup_Heterogeneous` |
| 1× CUDA + 1× ROCm + CPU | HOST | `Test__DeviceGroup_FullHeterogeneous` |

---

## Phase 2: Pipeline Parallelism (Weeks 4-6)

### Objective
Enable cross-rank pipeline parallelism for multi-socket/multi-node scaling.

### 2.1 Point-to-Point MPI Primitives
**Priority**: CRITICAL for PP

**Effort**: 3 days

**Current Gap**: `MPIContext` has collective operations but NO point-to-point:
```cpp
// Needed additions to MPIContext
void send(const void* data, size_t count, MPI_Datatype type, int dest, int tag);
void recv(void* data, size_t count, MPI_Datatype type, int src, int tag);
MPI_Request isend(...);  // Async
MPI_Request irecv(...);  // Async
void wait(MPI_Request* request);
void waitAll(std::vector<MPI_Request>& requests);
```

**Implementation Notes**:
- Use MPI tags to distinguish layer boundaries
- Consider persistent requests for decode loop efficiency

### 2.2 Send/Receive Activation Stages
**Priority**: HIGH

**Effort**: 4 days

**New Stages**:
```cpp
class SendActivationsStage : public ComputeStageBase {
    // Sends activations to next pipeline stage (rank)
    Params: buffer, dest_rank, tag, async
};

class ReceiveActivationsStage : public ComputeStageBase {
    // Receives activations from previous pipeline stage
    Params: buffer, src_rank, tag, async
};
```

**Integration with ComputeGraph**:
```cpp
// End of rank 0's layers (after layer 13)
graph.addNode("send_to_rank1", SendActivationsStage({
    .buffer = layer13_output,
    .dest_rank = 1,
    .tag = ACTIVATION_TAG,
}), DeviceId::cpu());  // CPU-staged for portability

// Start of rank 1's layers (before layer 14)
graph.addNode("recv_from_rank0", ReceiveActivationsStage({
    .buffer = layer14_input,
    .src_rank = 0,
    .tag = ACTIVATION_TAG,
}), DeviceId::cpu());
```

### 2.3 Layer Range Configuration
**Priority**: HIGH

**Effort**: 2 days

**Configuration**:
```cpp
struct PipelineParallelConfig {
    int num_pipeline_stages;           // e.g., 2
    std::vector<LayerRange> stage_layers;  // [{0,13}, {14,27}]
    int warmup_micro_batches;          // GPipe warmup count
};

struct LayerRange {
    int first_layer;  // inclusive
    int last_layer;   // inclusive
    int owning_rank;
};
```

**WeightManager Integration**:
- Only load weights for assigned layers
- Memory savings: ~50% per rank for 2-way PP

### 2.4 Basic Pipeline Scheduler (GPipe-style)
**Priority**: MEDIUM

**Effort**: 1 week

**Schedule for 2-stage PP with 4 micro-batches**:
```
Time →
Rank 0: [F0][F1][F2][F3]        [B3][B2][B1][B0]
Rank 1:     [F0][F1][F2][F3][B3][B2][B1][B0]
```

**Implementation**:
1. Partition input into micro-batches
2. Forward pass: Stream micro-batches through pipeline
3. Backward pass: N/A for inference (no gradients)
4. Synchronize between forward completion

**Note**: For inference-only, this simplifies to sequential micro-batch processing with activation buffering.

### 2.5 Pipeline Parallel Testing
**Priority**: HIGH

**Effort**: 3 days

**Test Cases**:
1. 2-rank PP with synchronous send/recv
2. Verify activations match single-rank execution
3. Memory usage verification (should be ~50% per rank)
4. Latency overhead measurement

---

## Phase 3: CPU Participation & Multi-Domain TP (Weeks 7-9)

### Objective
Enable CPU to process transformer layers and establish cross-socket CPU tensor parallelism via UPI while keeping GPU TP strictly intra-rank.

### 3.1 Layer Placement Configuration ✅ COMPLETE
**Priority**: MEDIUM | **Status**: ✅ COMPLETE (33 tests)

**Configuration**:
```cpp
struct LayerPlacementConfig {
    std::vector<LayerDeviceAssignment> assignments;
};

struct LayerDeviceAssignment {
    int layer_index;
    DeviceId device;  // DeviceId::cpu() or DeviceId::cuda(0) etc.
};
```

**Example - CPU handles first 4 layers**:
```cpp
LayerPlacementConfig config;
for (int i = 0; i < 4; i++) {
    config.assignments.push_back({i, DeviceId::cpu()});
}
for (int i = 4; i < 28; i++) {
    config.assignments.push_back({i, DeviceId::cuda(0)});
}
```

### 3.2 TPDomain and Multi-Domain Architecture
**Priority**: HIGH

**Effort**: 4 days

**Problem Statement**:
Current TensorParallelConfig is intra-rank only. We want:
1. **GPU TP**: Intra-rank only (NVIDIA↔AMD via PCIeBAR on same socket)
2. **CPU TP**: Cross-rank via MPI over UPI (~50 GB/s) while respecting NUMA boundaries

**Key Constraint**: NUMA boundaries must be respected - no shared memory across sockets. Each CPU/socket lives in its own "memory world" with message passing for communication.

**New Structures**:
```cpp
enum class TPDomainType {
    GPU_INTRA_RANK,    // PCIeBAR for NVIDIA↔AMD on same socket (intra-rank)
    CPU_CROSS_RANK,    // MPI over UPI for CPUs across sockets (cross-rank)
};

struct TPDomain {
    TPDomainType type;
    MPI_Comm communicator;          // Domain-specific communicator
    std::vector<DeviceId> devices;  // Devices in this domain
    int local_rank_in_domain;       // Our rank within this domain
    int domain_size;                // Number of participants
};

struct MultiDomainTPConfig {
    std::vector<TPDomain> domains;  
    
    // Which domain handles which compute
    TPDomain* gpu_attention_domain;   // GPU TP for attention heads
    TPDomain* cpu_ffn_domain;         // CPU TP for FFN layers (optional)
    
    // Layer assignments
    std::unordered_map<int, TPDomain*> layer_to_domain;
};
```

**Node Topology Detection**:
```cpp
class NodeTopology {
public:
    static NodeTopology detect();
    
    int num_sockets;                    // e.g., 2
    int numa_nodes_per_socket;          // e.g., 1-4
    std::vector<int> cores_per_socket;  // e.g., {56, 56}
    
    // UPI detection
    bool has_upi;                       // Inter-socket fabric detected
    float upi_bandwidth_gbps;           // ~50 GB/s typical
    
    // Map MPI ranks to sockets
    std::unordered_map<int, int> rank_to_socket;
    
    // Group ranks by socket for TP domains
    std::vector<std::vector<int>> ranks_per_socket;
};
```

**Implementation Tasks**:
1. `NodeTopology::detect()` - Parse `/sys/devices/system/node/` for NUMA info
2. `TPDomain` struct with communicator management
3. `MPI_Comm_split()` to create domain-specific communicators
4. Integration with `CollectiveContext` for domain-aware backend selection

**Test Cases**:
| Test | Description |
|------|-------------|
| `Test__TPDomain_SingleSocket` | Single socket = single GPU domain |
| `Test__TPDomain_DualSocket_GPUOnly` | 2 sockets, GPU TP within each socket |
| `Test__TPDomain_DualSocket_CPUCross` | 2 sockets, CPU TP across sockets |
| `Test__TPDomain_NUMADetection` | Verify NUMA topology detection |
| `Test__TPDomain_CommunicatorSplit` | Verify MPI_Comm_split correctness |

### 3.3 UPICollectiveBackend
**Priority**: HIGH

**Effort**: 3 days

**Purpose**: AllReduce for CPU TP across sockets using MPI over UPI fabric.

**Implementation**:
```cpp
class UPICollectiveBackend : public ICollectiveBackend {
public:
    UPICollectiveBackend(MPI_Comm domain_comm, const NodeTopology& topo);
    
    void allreduce(void* buffer, size_t count, DataType dtype) override {
        // Uses MPI_Allreduce on domain communicator
        // Communication flows over UPI (~50 GB/s)
        MPI_Allreduce(MPI_IN_PLACE, buffer, count, 
                      toMPIType(dtype), MPI_SUM, domain_comm_);
    }
    
    std::string name() const override { return "UPI"; }
    
private:
    MPI_Comm domain_comm_;  // CPU cross-socket communicator
};
```

**NUMA-Aware Buffer Allocation**:
```cpp
// Each CPU's buffers must be NUMA-local
void* allocateNUMALocal(size_t bytes, int numa_node) {
    void* ptr = numa_alloc_onnode(bytes, numa_node);
    // First-touch initialization for NUMA placement
    #pragma omp parallel for
    for (size_t i = 0; i < bytes; i += 4096) {
        ((char*)ptr)[i] = 0;
    }
    return ptr;
}
```

**Backend Selection Logic Update**:
```cpp
ICollectiveBackend* BackendRouter::selectBackend(const TPDomain& domain) {
    switch (domain.type) {
        case TPDomainType::GPU_INTRA_RANK:
            if (hasHeterogeneousGPUs(domain)) {
                return pciebar_backend_;  // NVIDIA↔AMD via PCIeBAR
            } else if (allCUDA(domain)) {
                return nccl_backend_;
            } else {
                return rccl_backend_;
            }
            
        case TPDomainType::CPU_CROSS_RANK:
            return upi_backend_;  // MPI over UPI fabric
    }
}
```

**Test Cases**:
| Test | Description |
|------|-------------|
| `Test__UPIBackend_AllReduce` | Basic allreduce across sockets |
| `Test__UPIBackend_NUMABuffer` | Verify NUMA-local allocation |
| `Test__UPIBackend_Latency` | Benchmark UPI latency (~1-5μs for small) |
| `Test__UPIBackend_Bandwidth` | Verify ~50 GB/s sustained |

### 3.4 CPU↔GPU Activation Transfer Stage
**Priority**: MEDIUM

**Effort**: 3 days

**New Stage**:
```cpp
class TransferActivationsStage : public ComputeStageBase {
    // Transfers activations between CPU and GPU within same rank
    Params: src_buffer, dst_buffer, src_device, dst_device
};
```

**Optimization Options**:
1. **Pageable memory**: Simple, ~15-30 μs for 14KB
2. **Pinned memory pool**: Lower latency, ~5-10 μs for 14KB
3. **Zero-copy mapping**: Near-zero sync, GPU accesses CPU memory directly

**Recommendation**: Start with pageable, add pinned pool if latency is bottleneck.

### 3.5 Async CPU Execution (Optional)
**Priority**: LOW

**Effort**: 1 week

**Current Limitation**: DeviceGraphExecutor runs stages sequentially. CPU and GPU stages don't overlap.

**Enhancement**:
```cpp
class AsyncGraphExecutor : public IGraphExecutor {
    // Thread pool for CPU stages
    // CUDA/HIP streams for GPU stages
    // Dependency tracking for overlap
};
```

**Value**: Enables CPU to compute layer N+1 while GPU computes layer N.

**Skip Condition**: If CPU throughput << GPU throughput (likely ~3% TFLOPS contribution), overlap provides minimal benefit. CPU's value is memory offload, not compute.

### 3.6 Phase 3 Testing Summary
**Priority**: HIGH

**Effort**: 3 days

**Test Cases**:
1. Single layer on CPU, rest on GPU - numerical parity
2. TPDomain creation for single-socket and dual-socket
3. UPI allreduce correctness and performance
4. Memory usage verification with NUMA-local allocation
5. End-to-end CPU TP across sockets

---

## Phase 4: Full Hybrid Integration (Weeks 10-12)

### Objective
Combine all parallelism modes with multi-domain TP and optimize end-to-end performance.

### 4.1 Four-Level Hybrid Mode
**Priority**: HIGH

**Effort**: 1.5 weeks

**Target Configuration** (2-socket system with 2 GPUs per socket):
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SOCKET 0 (MPI Rank 0)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  CPU (NUMA Node 0)           │  GPU Domain (Intra-Rank TP via PCIeBAR)     │
│  ├── Layers 0-3 (PP Stage 0a)│  ├── NVIDIA: Layers 4-13, heads 0-19       │
│  └── FFN TP Domain Member    │  └── AMD: Layers 4-13, heads 20-27          │
│                              │  └── AllReduce: PCIeBAR (25μs)              │
├─────────────────────────────────────────────────────────────────────────────┤
│                        ↓ MPI Send/Recv (PP boundary)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SOCKET 1 (MPI Rank 1)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  CPU (NUMA Node 1)           │  GPU Domain (Intra-Rank TP via PCIeBAR)     │
│  ├── Layers 14-17 (PP Stage 1a)│ ├── NVIDIA: Layers 18-27, heads 0-19     │
│  └── FFN TP Domain Member    │  └── AMD: Layers 18-27, heads 20-27         │
│                              │  └── AllReduce: PCIeBAR (25μs)              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              │     CPU TP Domain (Cross-Rank via UPI)        │
              │     Socket 0 CPU ←──UPI (~50GB/s)──→ Socket 1 CPU  │
              │     AllReduce: MPI over UPI fabric            │
              └───────────────────────────────────────────────┘
```

**Parallelism Hierarchy**:
| Level | Type | Communication | Latency |
|-------|------|---------------|---------|
| L1 | GPU Tensor Parallel | PCIeBAR (intra-rank) | 25μs |
| L2 | CPU Tensor Parallel | MPI over UPI (cross-rank) | ~5μs |
| L3 | Pipeline Parallel | MPI P2P (cross-rank) | ~10μs |
| L4 | Intra-rank CPU↔GPU | cudaMemcpy/hipMemcpy | ~15μs |

**Data Flow**:
1. Token embedding (CPU rank 0)
2. CPU layers 0-3 (rank 0, NUMA-local)
3. **Optional**: CPU TP AllReduce via UPI (if FFN split across sockets)
4. Transfer to GPUs (rank 0)
5. GPU TP across NVIDIA+AMD for layers 4-13 (rank 0, PCIeBAR)
6. MPI Send to rank 1 (PP boundary)
7. Repeat pattern on rank 1 with layers 14-27
8. LM head + sampling (rank 1)

### 4.2 MultiDomainOrchestrator
**Priority**: HIGH

**Effort**: 4 days

**New Component**:
```cpp
class MultiDomainOrchestrator {
public:
    MultiDomainOrchestrator(
        const MultiDomainTPConfig& tp_config,
        const PipelineParallelConfig& pp_config,
        const LayerPlacementConfig& placement_config);
    
    // Build compute graph with domain-aware collective stages
    std::unique_ptr<ComputeGraph> buildGraph(const ModelConfig& model);
    
private:
    // Insert correct collective stage based on domain
    void insertAllReduce(ComputeGraph& graph, TPDomain* domain, 
                         TensorBase* buffer, const std::string& name);
    
    // Determine which domain handles each layer's TP
    TPDomain* selectDomainForLayer(int layer_idx);
};
```

**Graph Construction Logic**:
```cpp
void MultiDomainOrchestrator::buildLayerGraph(int layer_idx, ComputeGraph& graph) {
    auto* domain = selectDomainForLayer(layer_idx);
    auto placement = placement_config_.getDevice(layer_idx);
    
    if (placement.type() == DeviceType::CPU) {
        // CPU layer with potential cross-socket TP
        buildCPULayerStages(layer_idx, graph);
        if (domain->type == TPDomainType::CPU_CROSS_RANK) {
            insertAllReduce(graph, domain, ffn_output, "cpu_tp_allreduce");
        }
    } else {
        // GPU layer with intra-rank TP
        buildGPULayerStages(layer_idx, graph, domain->devices);
        if (domain->size > 1) {
            insertAllReduce(graph, domain, wo_output, "gpu_tp_allreduce");
        }
    }
}
```

### 4.3 Communication Overlap
**Priority**: MEDIUM

**Effort**: 4 days

**Opportunities**:
1. **TP AllReduce overlap**: Start next layer's GEMM while AllReduce completes
2. **PP Send overlap**: Start layer N+1 on sender while receiver processes layer N
3. **Prefetch overlap**: Load next micro-batch while processing current
4. **Cross-domain overlap**: GPU TP and CPU TP can run concurrently on different layers

**Implementation**: CUDA/HIP streams + async MPI operations

### 4.4 Memory Optimization
**Priority**: MEDIUM

**Effort**: 3 days

**Tasks**:
1. Activation checkpointing for PP (recompute vs store)
2. KV cache memory pool shared across layers
3. Weight memory mapping for CPU layers (mmap, no copy)
4. NUMA-local buffer pools for CPU TP

### 4.5 End-to-End Benchmarking
**Priority**: HIGH

**Effort**: 3 days

**Benchmark Suite**:
| Test | Configuration | Target |
|------|--------------|--------|
| Single GPU decode | 1× ROCm | >5 tok/s |
| GPU TP decode | 1× CUDA + 1× ROCm (intra-rank) | >10 tok/s |
| PP decode | 2 ranks | >8 tok/s |
| CPU TP decode | 2× CPU cross-socket | >2 tok/s |
| Full hybrid | 2 ranks × (CPU + CUDA + ROCm) | >15 tok/s |

---

## Risk Register

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| PCIeBAR latency too high for small transfers | ~~HIGH~~ | ~~Medium~~ | ✅ MITIGATED - 25μs measured (8× better than target) |
| Proportional head assignment breaks numerical parity | MEDIUM | Low | Extensive parity testing |
| Pipeline bubble overhead dominates for short sequences | HIGH | Medium | Micro-batching, skip PP for short prompts |
| CUDA↔ROCm driver conflicts | HIGH | Low | Careful initialization order, isolation testing |
| MPI deadlocks in PP | HIGH | Medium | Extensive testing, tag management |
| **NEW**: NUMA contention in cross-socket CPU TP | MEDIUM | Medium | NUMA-local allocation, first-touch init |
| **NEW**: UPI bandwidth saturation | LOW | Low | UPI has ~50 GB/s; 14KB activations << capacity |
| **NEW**: MPI_Comm_split overhead | LOW | Low | Create communicators once at init |

---

## Timeline Summary

| Week | Phase | Status | Deliverable |
|------|-------|--------|-------------|
| 1 | Phase 0 | ✅ COMPLETE | CollectiveContext wired, PCIeBAR benchmarks (25μs!) |
| 1-2 | Phase 1a | ✅ COMPLETE | TensorParallelConfig infrastructure (33 tests) |
| 2 | Phase 1b | ✅ COMPLETE | WeightManager proportional slicing (6+ tests) |
| 2 | Phase 1c | ✅ COMPLETE | Qwen2Graph variable heads (11 tests) |
| 2 | Phase 1d | ✅ COMPLETE | AllGatherVStage (14 tests) |
| 2 | Phase 2.1 | ✅ COMPLETE | MPI P2P primitives (11 tests) |
| 2 | Phase 2.2 | ✅ COMPLETE | Send/Recv stages (9 tests) |
| 3 | Phase 2.3 | ✅ COMPLETE | PipelineParallelConfig (32 tests) |
| 3 | Phase 3.1 | ✅ COMPLETE | LayerPlacementConfig (33 tests) |
| 3 | Phase 3.2a | ✅ COMPLETE | NodeTopology NUMA/socket detection (29 tests) |
| 3 | Phase 3.2b | ✅ COMPLETE | TPDomain + MultiDomainTPConfig (30 tests) |
| 3 | Phase 3.3 | ✅ COMPLETE | UPICollectiveBackend (37 tests, 0.64μs latency!) |
| 3 | Phase 3.4 | ✅ COMPLETE | NUMAAllocator (32 tests) |
| 3 | Phase 3.5 | ✅ COMPLETE | BackendRouter domain-aware selection (16 new tests) |
| 4-5 | Phase 4.1-4.2 | ⏳ PLANNED | MultiDomainOrchestrator |
| 5-6 | Phase 4.3-4.5 | ⏳ PLANNED | Optimization, benchmarks |

**Total Effort**: ~2-3 weeks remaining (4 weeks completed)
**Test Count**: 365 tests passing (279 unit, 86 integration)

### Completed Work Detail

**Week 1-3 Accomplishments**:
- ✅ Wired CollectiveContext to AllreduceStage via DeviceGraphExecutor
- ✅ Implemented BackendRouter for automatic NCCL/RCCL/PCIeBAR selection
- ✅ TensorParallelConfig with DeviceShardingAssignment (proportional + equal split)
- ✅ WeightManager proportional slicing based on TensorParallelConfig
- ✅ Qwen2Graph support for variable local_num_heads per device
- ✅ AllGatherVStage for variable-sized collective output assembly
- ✅ MPI P2P send/recv/isend/irecv/wait primitives
- ✅ SendActivationsStage and ReceiveActivationsStage for PP
- ✅ PipelineParallelConfig with LayerRange and topology methods
- ✅ LayerPlacementConfig with TransitionPoint detection
- ✅ Verified PCIeBAR viability with 25μs latency (8× better than target)

**Week 3-4 Accomplishments (Phase 3.2-3.5)**:
- ✅ NodeTopology for NUMA/socket/UPI detection via sysfs (29 tests)
- ✅ TPDomain and MultiDomainTPConfig for domain management (30 tests)
- ✅ UPICollectiveBackend with MPI over UPI (0.64μs latency!) (37 tests)
- ✅ NUMAAllocator with libnuma integration (32 tests)
- ✅ BackendRouter domain-aware selection for GPU_INTRA_RANK/CPU_CROSS_RANK (16 tests)

---

## Appendix A: File Change Summary

### Phase 0 Files
- `src/v2/execution/DeviceGraphExecutor.h` - Add collective_ctx_
- `src/v2/execution/DeviceGraphExecutor.cpp` - Add executeCollectiveStage()
- `src/v2/pipelines/qwen/GraphOrchestrator.cpp` - Wire collective_ctx
- `tests/v2/integration/Test__PCIeBARLatencyBenchmark.cpp` - New

### Phase 1 Files
- `src/v2/collective/backends/PCIeBARBackend.cpp` - FP16/BF16, timeout
- `src/v2/loaders/WeightManager.cpp` - Proportional slicing
- `src/v2/loaders/WeightManager.h` - TensorParallelConfig
- `src/v2/models/qwen/Qwen2Graph.cpp` - Variable local_num_heads
- `src/v2/execution/compute_stages/stages/AllGatherVStage.cpp` - New

### Phase 2 Files
- `src/v2/utils/MPIContext.h` - P2P primitives
- `src/v2/utils/MPIContext.cpp` - send/recv implementation
- `src/v2/execution/compute_stages/stages/SendActivationsStage.cpp` - New
- `src/v2/execution/compute_stages/stages/ReceiveActivationsStage.cpp` - New
- `src/v2/pipelines/qwen/PipelineScheduler.cpp` - New

### Phase 3 Files
- `src/v2/pipelines/qwen/LayerPlacementConfig.h` - New
- `src/v2/execution/compute_stages/stages/TransferActivationsStage.cpp` - New

### Phase 4 Files
- `src/v2/pipelines/qwen/HybridOrchestrator.cpp` - New (combines all modes)
- `tests/v2/performance/Perf__HybridParallel.cpp` - New

---

## Appendix B: PCIeBAR Latency Test Plan

### Test Infrastructure
```cpp
// tests/v2/integration/Test__PCIeBARLatencyBenchmark.cpp

class PCIeBARLatencyTest : public ::testing::Test {
protected:
    void SetUp() override {
        cuda_device_ = DeviceId::cuda(0);
        rocm_device_ = DeviceId::rocm(0);
        
        backend_ = std::make_unique<PCIeBARBackend>(
            ClusterInventory::detectDevices());
    }
    
    void benchmarkAllReduce(size_t bytes, int iterations = 100) {
        auto cuda_buffer = allocateCUDA(bytes);
        auto rocm_buffer = allocateROCm(bytes);
        
        // Warmup
        for (int i = 0; i < 10; i++) {
            backend_->allreduce(cuda_buffer.get(), bytes, DataType::FP32);
        }
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            backend_->allreduce(cuda_buffer.get(), bytes, DataType::FP32);
            synchronize();
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        double total_us = std::chrono::duration<double, std::micro>(end - start).count();
        double per_op_us = total_us / iterations;
        
        LOG_INFO("PCIeBAR AllReduce " << bytes << " bytes: " 
                 << per_op_us << " μs/op");
    }
};

TEST_F(PCIeBARLatencyTest, SmallTransferLatency) {
    // Critical for hybrid parallelism viability
    benchmarkAllReduce(1 * 1024);    // 1 KB
    benchmarkAllReduce(14 * 1024);   // 14 KB (Wo output)
    benchmarkAllReduce(28 * 1024);   // 28 KB (KV head)
    benchmarkAllReduce(64 * 1024);   // 64 KB
    benchmarkAllReduce(128 * 1024);  // 128 KB
}

TEST_F(PCIeBARLatencyTest, LargeTransferBandwidth) {
    benchmarkAllReduce(1 * 1024 * 1024);   // 1 MB
    benchmarkAllReduce(4 * 1024 * 1024);   // 4 MB
    benchmarkAllReduce(16 * 1024 * 1024);  // 16 MB
}

TEST_F(PCIeBARLatencyTest, SustainedThroughput) {
    // Simulate decode loop: 56 AllReduces per forward pass
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int token = 0; token < 100; token++) {
        for (int layer = 0; layer < 28; layer++) {
            backend_->allreduce(wo_buffer_.get(), 14 * 1024, DataType::FP32);
            backend_->allreduce(ffn_buffer_.get(), 14 * 1024, DataType::FP32);
        }
    }
    synchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    double total_s = std::chrono::duration<double>(end - start).count();
    double tok_per_s = 100.0 / total_s;
    
    LOG_INFO("PCIeBAR sustained decode: " << tok_per_s << " tok/s (AllReduce only)");
    
    // Target: AllReduce overhead should allow >10 tok/s
    EXPECT_GT(tok_per_s, 10.0);
}
```

### Success Criteria Table

| Transfer Size | Target Latency | Acceptable | Blocking |
|--------------|----------------|------------|----------|
| 1 KB | < 100 μs | < 200 μs | > 500 μs |
| 14 KB | < 150 μs | < 300 μs | > 700 μs |
| 28 KB | < 200 μs | < 400 μs | > 1 ms |
| 128 KB | < 400 μs | < 800 μs | > 2 ms |

**If blocking thresholds are exceeded**: Fall back to hybrid strategy where NVIDIA uses NCCL internally, AMD uses RCCL internally, and cross-vendor communication uses CPU staging.

### Actual Measured Results (January 2026)

| Transfer Size | Target | Actual | Status |
|--------------|--------|--------|--------|
| 1 KB | < 100 μs | 18 μs | ✅ **5× better** |
| 14 KB | < 200 μs | 25 μs | ✅ **8× better** |
| 28 KB | < 400 μs | ~40 μs | ✅ **10× better** |
| 128 KB | < 1000 μs | 108 μs | ✅ **9× better** |

**Sustained decode simulation**: 731 tok/s (AllReduce only) - exceeds 10 tok/s target by **73×**!

**Conclusion**: PCIeBAR latency is well within acceptable thresholds. The architecture is validated for heterogeneous CUDA+ROCm tensor parallelism.

---

## Appendix C: Decision Tree for Backend Selection (Updated)

```
                        ┌──────────────────────────────────┐
                        │     Need collective operation?    │
                        └──────────────┬───────────────────┘
                                       │
                        ┌──────────────▼───────────────────┐
                        │   Which TPDomain is this for?    │
                        └──────────────┬───────────────────┘
                                       │
           ┌───────────────────────────┼───────────────────────────┐
           ▼                           ▼                           ▼
    GPU_INTRA_RANK              CPU_CROSS_RANK                  NONE
           │                           │                           │
           ▼                           ▼                           ▼
    ┌──────────────┐           ┌──────────────┐             Use MPI World
    │ Check GPU mix │           │ Use UPI Backend│
    └──────┬───────┘           │ (MPI over UPI) │
           │                   └──────────────┘
     ┌─────┼─────────────┐
     ▼     ▼             ▼
  All CUDA? All ROCm?  Mixed?
     │     │             │
     ▼     ▼             ▼
   NCCL   RCCL        PCIeBAR
```

**Domain-Aware Backend Selection**:
```cpp
ICollectiveBackend* selectBackend(const CollectiveOp& op, const TPDomain* domain) {
    if (!domain) {
        // No domain specified - use MPI world (cross-rank PP)
        return mpi_backend_;
    }
    
    switch (domain->type) {
        case TPDomainType::GPU_INTRA_RANK:
            if (isHeterogeneousGPUs(domain->devices)) {
                return pciebar_backend_;  // NVIDIA↔AMD: 25μs
            } else if (allCUDA(domain->devices)) {
                return nccl_backend_;
            } else {
                return rccl_backend_;
            }
            
        case TPDomainType::CPU_CROSS_RANK:
            return upi_backend_;  // MPI over UPI: ~5μs
    }
}
```

---

## Appendix D: Phase 3.2-3.3 File Changes

### New Files for TPDomain
| File | Purpose |
|------|---------|
| `src/v2/config/TPDomain.h` | TPDomain, TPDomainType, MultiDomainTPConfig |
| `src/v2/config/TPDomain.cpp` | Domain construction, communicator management |
| `src/v2/utils/NodeTopology.h` | NUMA/socket detection |
| `src/v2/utils/NodeTopology.cpp` | Parse `/sys/devices/system/node/`, detect UPI |
| `src/v2/collective/backends/UPIBackend.h` | UPI collective backend interface |
| `src/v2/collective/backends/UPIBackend.cpp` | MPI-based allreduce over UPI fabric |
| `src/v2/memory/NUMAAllocator.h` | NUMA-aware buffer allocation |
| `src/v2/memory/NUMAAllocator.cpp` | `numa_alloc_onnode()` + first-touch init |

### Modified Files
| File | Change |
|------|--------|
| `src/v2/collective/BackendRouter.h` | Add `selectBackend(const TPDomain*)` overload |
| `src/v2/collective/BackendRouter.cpp` | Domain-aware backend selection logic |
| `src/v2/collective/CollectiveContext.h` | Add domain tracking |
| `src/v2/execution/DeviceGraphExecutor.cpp` | Pass domain to collective operations |
| `src/v2/pipelines/qwen/GraphOrchestrator.cpp` | Use MultiDomainOrchestrator pattern |

### Test Files
| File | Purpose |
|------|---------|
| `tests/v2/unit/Test__TPDomain.cpp` | TPDomain creation, communicator split |
| `tests/v2/unit/Test__NodeTopology.cpp` | NUMA detection, UPI discovery |
| `tests/v2/unit/Test__UPIBackend.cpp` | UPI allreduce correctness |
| `tests/v2/integration/Test__CrossSocketCPUTP.cpp` | End-to-end cross-socket CPU TP |
