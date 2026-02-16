# Distributed Architecture Proposal: Tensor-Parallel GraphOrchestrator

**Author**: David Sanftenberg  
**Date**: December 2025  
**Status**: Implementation In Progress

## Executive Summary

This document proposes a redesigned distributed inference architecture for Llaminar V2 that:
1. Eliminates redundant computation across MPI ranks
2. Properly separates physical topology (nodes/sockets) from logical topology (MPI ranks)
3. Uses explicit graph stages for tensor slicing and gathering
4. Removes the legacy `MpiAttentionOrchestrator` in favor of declarative graph operations
5. **ALL ranks (including rank 0) participate in compute by default**
6. **Future support for heterogeneous work distribution by device capability**

## Key Design Decisions

### Decision 1: All Ranks Compute (Including Rank 0)
Unlike traditional coordinator/worker patterns, **rank 0 participates in computation** by default:
- Eliminates idle coordinator waste (especially important for 2-rank systems)
- Uses equal work division initially
- `is_compute_participant()` defaults to `true` for all ranks
- Only explicit opt-out changes this behavior

### Decision 2: Use Existing TensorSlice Class
**Do NOT create a duplicate `TensorSlice` struct.** We already have:
- `src/v2/tensors/TensorSlice.h` - Full tensor slice wrapper class
- `SliceMetadata` struct with `ROW_PARALLEL`, `COLUMN_PARALLEL`, `FULL` modes
- Factory methods: `SliceMetadata::forRowParallel()`, `SliceMetadata::forColumnParallel()`

For simple work ranges (heads, vocab indices), use `WorkRange` struct which is lighter weight.
For tensor parallelism metadata, use `SliceMetadata` from the existing TensorSlice.h.

### Decision 3: Device Capability Exchange at Startup
At construction, `MPITopology`:
1. Detects local devices (CPU, CUDA, ROCm)
2. AllGathers device capabilities from all ranks
3. Stores in `all_placements_` vector
4. Enables future weighted work distribution

### Decision 4: Deprecated Components (Do NOT Use)

The following components are **DEPRECATED** and must NOT be used for new development:

| Component | Status | Replacement |
|-----------|--------|-------------|
| `PipelineBase` / `Qwen2Pipeline` | ⛔ DEPRECATED | `GraphOrchestrator` + `ComputeStage` |
| `MpiAttentionOrchestrator` | ⛔ DEPRECATED | `AttentionComputeStage` with TP params |
| `GQAAttention` | ⛔ DEPRECATED | `AttentionComputeStage` |
| `project_row_parallel()` | ⛔ DEPRECATED | `GEMMStage` with `WorkRange` |
| `project_column_parallel()` | ⛔ DEPRECATED | `GEMMStage` with `WorkRange` + `AllreduceStage` |

**Rationale**: The legacy Pipeline system (~3000 lines) duplicates graph execution logic and hides MPI communication. The graph-based approach makes tensor parallelism explicit and composable.

**Note**: The Pipeline implementations contain working TP code that serves as **reference implementations** for migrating functionality to the Stage API. The kernel-level TP support (row-sliced constructors, head ranges) was developed for the Pipeline and can be reused.

## Device Support Status

### Current Infrastructure

The graph execution system has a well-defined device abstraction layer:

| Component | Status | Description |
|-----------|--------|-------------|
| `IDeviceContext` | ✅ Complete | Abstract interface for CPU/GPU execution contexts |
| `CPUDeviceContext` | ✅ Complete | OpenMP-based parallel execution, NUMA-aware allocation |
| `CUDADeviceContext` | ✅ Complete | CUDA stream/memory management (when `HAVE_CUDA`) |
| `ROCmDeviceContext` | ✅ Complete | HIP stream/memory management (when `HAVE_ROCM`) |
| `DeviceManager` | ✅ Complete | Enumerate all devices, NUMA filtering for MPI ranks |
| `KernelFactory` | ⚠️ Partial | CPU kernels complete, GPU kernels partial |

### Kernel Device Support Matrix

| Kernel Type | CPU | CUDA | ROCm | Notes |
|-------------|-----|------|------|-------|
| **GEMM (Quantized)** | ✅ AVX512-VNNI | ✅ JIT + CUTLASS | ❌ Stub | `CudaGemmFactory.cu` |
| **GEMM (FP32/BF16)** | ✅ OneDNN | ⚠️ cuBLAS | ⚠️ rocBLAS | Fallback to CPU |
| **Attention** | ✅ Typed kernels | ❌ CPU fallback | ❌ CPU fallback | Flash Attention TODO |
| **RMSNorm** | ✅ SIMD | ❌ CPU fallback | ❌ CPU fallback | |
| **RoPE** | ✅ SIMD | ❌ CPU fallback | ❌ CPU fallback | |
| **SwiGLU** | ✅ SIMD | ❌ CPU fallback | ❌ CPU fallback | |
| **Softmax** | ✅ Typed | ❌ CPU fallback | ❌ CPU fallback | |
| **Embedding** | ✅ Complete | ❌ CPU fallback | ❌ CPU fallback | |

**Legend**: ✅ = Implemented, ⚠️ = Partial/External, ❌ = Not implemented (uses CPU fallback)

### Heterogeneous Execution Stumbling Blocks

#### Block 1: Attention Kernels GPU-Only Incomplete

**Problem**: `KernelFactory::createAttention()` falls back to CPU for all GPU requests:
```cpp
case DeviceType::CUDA:
    LOG_DEBUG("CUDA Attention not implemented, using CPU fallback");
    return std::make_unique<CPUAttentionKernelTyped<FP32>>();
```

**Impact**: Attention is compute-bound on large models. Without GPU attention, heterogeneous execution loses most benefit.

**Solution Path**:
1. Integrate FlashAttention-2 for CUDA
2. Add hipFlashAttention for ROCm
3. Alternative: Use cuDNN/MIOpen attention APIs

#### Block 2: Cross-Device Data Transfer

**Problem**: When tensor parallelism splits work across CPU and GPU, intermediate results must transfer:
- Q/K/V computed on GPU → Attention on CPU (if GPU attention missing)
- Attention output on CPU → Wo projection on GPU

**Current State**: `IDeviceContext::copyFromDevice()` exists but isn't integrated into stage execution flow.

**Solution Path**:
1. Add `DataTransferStage` to compute graph for explicit cross-device copies
2. Or: Automatic transfer detection in `DeviceGraphExecutor` when stage device differs from input device

#### Block 3: MPI + GPU Work Division

**Problem**: `MPITopology::WorkRange::for_rank_weighted()` supports heterogeneous work distribution, but:
- GPU compute power estimation is heuristic (`relative_compute = 10.0` for GPU vs `1.0` for CPU)
- No runtime calibration of actual throughput
- Work granularity may not match GPU tile sizes efficiently

**Current State**: `DeviceCapability` struct exists with `relative_compute` field, but not calibrated.

**Solution Path**:
1. Add benchmark-based calibration at startup (run small GEMM, measure throughput)
2. Align work ranges to GPU-friendly boundaries (multiple of 32/64 for warps)
3. Consider dynamic work stealing for load balancing

#### Block 4: Graph Stage Device Affinity

**Problem**: Current `ComputeStage` params have `device_idx` but:
- No mechanism to specify "prefer GPU" vs "require GPU"
- No automatic fallback to CPU if GPU unavailable
- No multi-device stage support (e.g., data-parallel across 2 GPUs)

**Solution Path**:
1. Add `DevicePreference` enum: `REQUIRE_GPU`, `PREFER_GPU`, `CPU_ONLY`, `ANY`
2. `DeviceGraphExecutor` selects device based on preference + availability
3. Support device lists for data-parallel stages

### Device-Specific Weight Packing

**Key Insight**: CPU and GPU backends require different weight packing formats for optimal GEMM performance:
- **CPU (AVX512-VNNI)**: `[N/64][K/4][64][4]` layout with interleaved scales/mins
- **CUDA (INT8)**: COL32 or COL32_2R_4R4 layout for Tensor Cores
- **ROCm (INT8)**: Custom layout for Matrix Cores

**Design Decision**: Because `WeightPlacementMap` assigns each weight tensor to a **single device** for its lifetime (layer-level or block-level granularity), we can pack weights in device-specific formats without conflict.

**How It Works**:

```
WeightPlacementMap                    KernelFactory::getOrCreateGemm()
┌─────────────────┐                   ┌───────────────────────────────┐
│ Layer 0: GPU 0  │───tensor W_q──────►│ dev_type = CUDA              │
│ Layer 1: GPU 0  │                   │ → packForCuda(W_q)           │
│ Layer 2: CPU    │                   │ → tensor->cache_ = gpu_packed│
│ ...             │                   └───────────────────────────────┘
│ Layer 23: CPU   │───tensor W_k──────►┌───────────────────────────────┐
└─────────────────┘                   │ dev_type = CPU               │
                                      │ → packForVNNI(W_k)           │
                                      │ → tensor->cache_ = cpu_packed│
                                      └───────────────────────────────┘
```

**Implementation Pattern**:

1. **At model load time**: `WeightPlacementMap` assigns tensors to devices (via `setLayerDevice()`, `setAttentionDevice()`, `setFFNDevice()`)

2. **At first kernel creation**: `KernelFactory::getOrCreateGemm()` checks `tensor->device_index()` and dispatches to device-specific packing:
   ```cpp
   auto dev_type = getDeviceType(tensor->device_index());
   
   if (dev_type == DeviceType::CPU) {
       // Pack into VNNI format, store in tensor->cache_
       packWeightsInto(tensor, new_cache->packed, VNNI_LAYOUT);
   }
   #ifdef HAVE_CUDA
   else if (dev_type == DeviceType::CUDA) {
       // Pack into CUDA-optimal format (e.g., COL32)
       packForCuda(tensor, new_cache->packed);
   }
   #endif
   ```

3. **At kernel execution**: The kernel uses the cached packed weights without re-packing.

**Benefits**:
- No cross-device packing conflicts (tensor is device-bound)
- Pack once, use many times (cached in `tensor->cache_`)
- Each backend uses its optimal layout
- GPU kernels can pack directly to device memory if desired

### Intra-Node Work Division: CPU vs GPU

**Problem**: On a node with both CPUs and GPUs, how do we decide which layers/operations run where?

**Current Architecture**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              NODE (1 MPI Rank)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────┐          ┌─────────────────────────────────┐   │
│  │  WeightPlacementMap │──────────►│        GraphOrchestrator       │   │
│  │  (Layer → Device)   │          │                                 │   │
│  └─────────────────────┘          │  ┌───────────┐ ┌───────────┐   │   │
│           │                       │  │ GEMMStage │ │ Attention │   │   │
│           │                       │  │ device=0  │ │ device=1  │   │   │
│           ▼                       │  └─────┬─────┘ └─────┬─────┘   │   │
│  ┌─────────────────────┐          └────────┼─────────────┼─────────┘   │
│  │    DeviceManager    │                   │             │             │
│  │  Enumerate devices  │                   ▼             ▼             │
│  │  NUMA filtering     │          ┌─────────────┐ ┌─────────────┐      │
│  └─────────────────────┘          │ CPU Context │ │ GPU Context │      │
│           │                       │ (OpenBLAS)  │ │ (CUDA/HIP)  │      │
│           ▼                       └─────────────┘ └─────────────┘      │
│  device_idx=0: CPU Socket 0                                            │
│  device_idx=1: CUDA GPU 0 (NUMA-filtered)                              │
│  device_idx=2: CUDA GPU 1 (if available)                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Responsibility Breakdown**:

| Component | Role | When |
|-----------|------|------|
| `DeviceManager` | Enumerate available devices (CPU sockets, GPUs) | Startup |
| `DeviceManager` | NUMA-filter GPUs for this MPI rank | Startup |
| `WeightPlacementMap` | Policy: Assign weights to devices by layer/block | Model load |
| `KernelFactory` | Create device-specific kernels, pack weights | First execution |
| `GraphOrchestrator` | Execute stages on assigned devices | Forward pass |
| `IDeviceContext` | Actual computation (OpenMP/CUDA/HIP) | Kernel execution |

**Work Division Strategies**:

1. **Layer-Level** (Simple): Entire layers on one device
   ```cpp
   placement_map->setLayerDevice(0, GPU_0);  // Layer 0 → GPU
   placement_map->setLayerDevice(1, GPU_0);  // Layer 1 → GPU
   placement_map->setLayerDevice(2, CPU_0);  // Layer 2 → CPU (offload)
   ```

2. **Block-Level** (Fine-grained): Attention vs FFN on different devices
   ```cpp
   placement_map->setAttentionDevice(layer, GPU_0);  // Attention → GPU (memory-bound)
   placement_map->setFFNDevice(layer, CPU_0);        // FFN → CPU (compute-bound, AVX512)
   ```

3. **MoE Expert-Level** (Future): Individual experts on different devices
   ```cpp
   placement_map->setLocalExpertDevice(layer, expert_0, GPU_0);
   placement_map->setLocalExpertDevice(layer, expert_1, CPU_0);
   ```

**Where This Fits in the MPI Framework**:

```
INTER-NODE (MPI)                    INTRA-NODE (Device)
┌──────────────┐                    ┌──────────────────────────────────┐
│   Rank 0     │◄───AllReduce──────►│   Rank 0 Node                    │
│   Node 0     │                    │   ┌────────┐  ┌────────────────┐ │
│              │                    │   │ GPU 0  │  │ CPU (2 sockets)│ │
└──────────────┘                    │   │Layer 0 │  │ Layer 2-23     │ │
       │                            │   │Layer 1 │  │                │ │
   Allreduce                        │   └────────┘  └────────────────┘ │
   AllGather                        └──────────────────────────────────┘
       │                            
┌──────────────┐                    ┌──────────────────────────────────┐
│   Rank 1     │◄───AllReduce──────►│   Rank 1 Node                    │
│   Node 1     │                    │   ┌────────┐  ┌────────────────┐ │
│              │                    │   │ GPU 0  │  │ CPU (2 sockets)│ │
└──────────────┘                    │   │Layer 0 │  │ Layer 2-23     │ │
                                    │   │Layer 1 │  │                │ │
                                    │   └────────┘  └────────────────┘ │
                                    └──────────────────────────────────┘
```

**Key Points**:
- **MPI ranks** handle **inter-node** tensor parallelism (weight sharding, AllReduce)
- **DeviceManager + WeightPlacementMap** handle **intra-node** heterogeneous execution
- These are orthogonal: a rank can shard weights across nodes AND offload layers to GPU
- Cross-device transfers (GPU↔CPU) use `DataTransferStage` or automatic transfers in `DeviceGraphExecutor`

### Work Distribution Decision Flow

**Problem**: How does the coordinator's (rank 0) work distribution decision propagate to all ranks and into `WeightPlacementMap`?

**Current State**: `MPITopology::exchangeCapabilities()` does an `MPI_Allgather` of compute weights, but there's no centralized decision-making or broadcast of placement decisions.

**Proposed Flow** (to be implemented):

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    WORK DISTRIBUTION DECISION FLOW                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  PHASE 1: Capability Discovery (All Ranks)                                      │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                                                                           │  │
│  │  Rank 0                  Rank 1                  Rank N                   │  │
│  │  ┌────────────────┐      ┌────────────────┐      ┌────────────────┐      │  │
│  │  │ Detect local   │      │ Detect local   │      │ Detect local   │      │  │
│  │  │ devices (CPU,  │      │ devices (CPU,  │      │ devices (CPU,  │      │  │
│  │  │ GPU, memory)   │      │ GPU, memory)   │      │ GPU, memory)   │      │  │
│  │  └───────┬────────┘      └───────┬────────┘      └───────┬────────┘      │  │
│  │          │                       │                       │               │  │
│  │          └───────────────────────┴───────────────────────┘               │  │
│  │                                  │                                        │  │
│  │                          MPI_Allgather                                    │  │
│  │                     (DeviceCapability structs)                            │  │
│  │                                  │                                        │  │
│  │                                  ▼                                        │  │
│  │          ┌───────────────────────────────────────────────┐               │  │
│  │          │  all_placements_[] populated on ALL ranks     │               │  │
│  │          │  (each rank knows everyone's capabilities)    │               │  │
│  │          └───────────────────────────────────────────────┘               │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  PHASE 2: Coordinator Decision (Rank 0 Only)                                    │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                                                                           │  │
│  │  Rank 0 (Coordinator)                                                     │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │  │
│  │  │  PlacementStrategy::compute(                                        │  │  │
│  │  │      model_config,          // n_layers, n_heads, d_model, etc.     │  │  │
│  │  │      all_placements_,       // Device capabilities from AllGather   │  │  │
│  │  │      memory_budget          // Optional: max memory per device      │  │  │
│  │  │  ) → PlacementPlan {                                                │  │  │
│  │  │      layer_to_rank: [0,0,0,1,1,1,...],   // TP across ranks         │  │  │
│  │  │      layer_to_device: [GPU,GPU,CPU,...], // Per-rank device         │  │  │
│  │  │      attention_devices: [...],           // Block-level overrides   │  │  │
│  │  │      ffn_devices: [...],                                            │  │  │
│  │  │  }                                                                  │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                           │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  PHASE 3: Plan Broadcast (Rank 0 → All)                                         │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                                                                           │  │
│  │          Rank 0 ────────── MPI_Bcast(PlacementPlan) ──────────► All      │  │
│  │                              (serialized plan)                            │  │
│  │                                                                           │  │
│  │  Alternative: Each rank computes same plan locally (deterministic algo)   │  │
│  │               This avoids broadcast but requires identical logic          │  │
│  │                                                                           │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  PHASE 4: Local Application (All Ranks)                                         │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                                                                           │  │
│  │  Each Rank                                                                │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │  │
│  │  │  // Extract this rank's portion of the plan                         │  │  │
│  │  │  auto my_layers = plan.getLayersForRank(my_rank);                   │  │  │
│  │  │  auto placement_map = std::make_shared<WeightPlacementMap>();       │  │  │
│  │  │                                                                     │  │  │
│  │  │  for (int layer : my_layers) {                                      │  │  │
│  │  │      int device = plan.getDeviceForLayer(layer);                    │  │  │
│  │  │      placement_map->setLayerDevice(layer, device);                  │  │  │
│  │  │                                                                     │  │  │
│  │  │      // Optional block-level overrides                              │  │  │
│  │  │      if (plan.hasAttentionOverride(layer)) {                        │  │  │
│  │  │          placement_map->setAttentionDevice(layer, ...);             │  │  │
│  │  │      }                                                              │  │  │
│  │  │  }                                                                  │  │  │
│  │  │                                                                     │  │  │
│  │  │  // Pass to WeightManager for loading                               │  │  │
│  │  │  WeightManager wm(loader, mpi_ctx, placement_map, strategy);        │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                           │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Components to Implement**:

| Component | Status | Description |
|-----------|--------|-------------|
| `MPITopology::exchangeCapabilities()` | ✅ Exists | AllGather of device capabilities |
| `PlacementStrategy` | ❌ TODO | Algorithm to compute optimal placement from capabilities |
| `PlacementPlan` | ❌ TODO | Serializable struct with full placement decisions |
| `PlacementPlan::broadcast()` | ❌ TODO | MPI_Bcast of serialized plan |
| `WeightPlacementMap::applyPlan()` | ❌ TODO | Populate map from plan for this rank |

**Strategy Variants** (future):

1. **Greedy Memory-First**: Fill GPU memory, overflow to CPU
2. **Layer-Balanced**: Equal layers per device, respecting memory
3. **Compute-Weighted**: Distribute by `relative_compute` from capabilities
4. **Latency-Optimized**: Minimize cross-device transfers

**Decision: Broadcast vs Local Computation**:

We recommend **local computation** (each rank computes the same plan) because:
- Simpler (no serialization/deserialization)
- Deterministic algorithms guarantee identical results
- Lower latency (no broadcast wait)
- Easier debugging (all ranks have same code path)

This requires `PlacementStrategy::compute()` to be pure and deterministic.

### Recommended Device Integration Order

**Phase A (Current)**: CPU-only tensor parallelism via MPI
- All ranks use `device_idx = 0` (CPU)
- Focus on getting TP correct with `WorkRange` in stages

**Phase B**: Single-GPU per rank
- Each MPI rank owns one GPU
- `DeviceManager` filters GPUs by NUMA affinity (already implemented)
- GEMM runs on GPU, other ops on CPU (hybrid)

**Phase C**: Heterogeneous CPU+GPU per rank
- Single rank uses both CPU and GPU
- Work split by `relative_compute` weights
- Requires cross-device transfers and GPU attention kernels

**Phase D**: Multi-GPU per rank
- Single rank drives multiple GPUs
- Data parallelism within rank, tensor parallelism across ranks

## Current State Problems

### Problem 1: Redundant Computation
Currently, most operations are **replicated** across all ranks:
- Embedding lookup: All ranks compute full embeddings
- QKV projections: All ranks compute full Q/K/V
- LM Head: All ranks compute full logits (151K vocab × batch × 4B = massive waste)

Only Wo and Down projections use row-parallel sharding with AllreduceStage.

### Problem 2: Hidden MPI Logic
The `MpiAttentionOrchestrator` (~1600 lines) hides tensor-parallel attention logic inside a monolithic class, making it:
- Hard to debug (batched attention bug we were investigating)
- Hard to compose with other stages
- Inconsistent with the declarative graph approach

### Problem 3: Missing Physical Topology
`NUMATopology` handles intra-node CPU/GPU affinity but there's no abstraction for:
- Multi-node clusters
- Rank-to-node mapping
- Optimized communication patterns (shared memory intra-node vs network inter-node)

## Proposed Architecture

### 1. MPITopology Class

The `MPITopology` class (implemented in `src/v2/utils/MPITopology.h`) provides:

```cpp
// src/v2/utils/MPITopology.h

namespace llaminar2 {

/**
 * @brief Device capability for heterogeneous work distribution
 */
struct DeviceCapability {
    enum class Type { CPU, CUDA, ROCm };
    Type type = Type::CPU;
    int device_id = 0;
    float relative_compute = 1.0f;  ///< Relative compute power (CPU=1.0, GPU~=10.0)
    size_t memory_bytes = 0;
    std::string name;
};

/**
 * @brief Information about a single MPI rank's placement
 */
struct RankPlacement {
    int rank;                    ///< MPI rank (0..world_size-1)
    int node_id;                 ///< Physical node/machine (0..node_count-1)
    int local_rank;              ///< Rank within node (0..ranks_per_node-1)
    int socket_id;               ///< CPU socket within node
    int numa_node;               ///< NUMA node for memory affinity
    std::string hostname;        ///< Machine hostname
    std::vector<DeviceCapability> devices;  ///< All devices available to this rank
};

/**
 * @brief Simple work range (lighter weight than TensorSlice)
 * 
 * For tensor parallelism metadata, use SliceMetadata from tensors/TensorSlice.h
 */
struct WorkRange {
    size_t start;  ///< Start index (inclusive)
    size_t end;    ///< End index (exclusive)
    size_t count;  ///< Number of elements
    
    bool empty() const { return count == 0; }
    
    /// Create equal work range for rank
    static WorkRange for_rank_equal(size_t total, int rank, int world_size);
    
    /// Create weighted work range (future: by device capability)
    static WorkRange for_rank_weighted(size_t total, int rank, int world_size,
                                        const std::vector<float>& weights);
};

/**
 * @brief MPI topology abstraction for distributed inference
 * 
 * Key design principles:
 * 1. ALL ranks (including rank 0) participate in compute by default
 * 2. Equal work division initially; future support for weighted distribution
 * 3. Integrates with existing SliceMetadata from tensors/TensorSlice.h
 */
class MPITopology {
public:
    explicit MPITopology(MPI_Comm comm = MPI_COMM_WORLD);
    
    // Basic queries
    int rank() const;
    int world_size() const;
    bool is_coordinator() const { return rank_ == 0; }
    bool is_compute_participant() const { return compute_participant_; }  // Default: true
    
    // Work distribution (returns WorkRange for simple indices)
    WorkRange get_head_range(int total_heads) const;
    WorkRange get_kv_head_range(int total_kv_heads) const;
    WorkRange get_column_range(size_t total_cols) const;
    WorkRange get_row_range(size_t total_rows) const;
    WorkRange get_vocab_range(size_t vocab_size) const;
    
    // SliceMetadata creation (integrates with TensorSlice.h)
    SliceMetadata createRowParallelMeta(size_t rows, size_t cols, bool presliced = false) const;
    SliceMetadata createColumnParallelMeta(size_t rows, size_t cols, bool presliced = false) const;
    
    // Device capabilities (gathered from all ranks at startup)
    const std::vector<RankPlacement>& all_placements() const;
    std::vector<float> get_compute_weights() const;
    
    // Communicators
    MPI_Comm intra_node_comm() const;
    MPI_Comm inter_node_comm() const;
    MPI_Comm world_comm() const;

private:
    bool compute_participant_ = true;  // ALL ranks compute by default
    std::vector<RankPlacement> all_placements_;  // From AllGather
};

} // namespace llaminar2
```

### 2. Distributed Graph Stages

New stages that make tensor parallelism explicit in the compute graph:

```cpp
// src/v2/execution/DistributedStages.h

namespace llaminar2 {
};

} // namespace llaminar2
```

### 2. Distributed Graph Stages

New stages that make tensor parallelism explicit in the compute graph:

```cpp
// src/v2/execution/DistributedStages.h

namespace llaminar2 {

// =============================================================================
// SliceStage: Extract local portion of a tensor
// =============================================================================

/**
 * @brief Extract a slice of input tensor for local computation
 * 
 * Used at the START of tensor-parallel regions to get this rank's portion.
 */
class SliceStage : public IComputeStage {
public:
    struct Params {
        TensorBase* input;           ///< Full tensor (may be nullptr on non-root ranks)
        TensorBase* output;          ///< Local slice output
        WorkRange range;             ///< Which portion to extract
        int dim;                     ///< Dimension to slice (0=rows, 1=cols)
        const MPITopology* topology; ///< For rank-aware slicing
    };
    
    bool execute(IDeviceContext* ctx) override;
    ComputeStageType type() const override { return ComputeStageType::SLICE; }
};

// =============================================================================
// GatherStage: Collect slices from all ranks to one rank
// =============================================================================

/**
 * @brief Gather tensor slices from all ranks to coordinator (rank 0)
 * 
 * Used at the END of tensor-parallel regions to collect results.
 * NOTE: Rank 0 also contributes its computed portion (all ranks compute).
 */
class GatherStage : public IComputeStage {
public:
    struct Params {
        TensorBase* local_input;     ///< This rank's local slice
        TensorBase* gathered_output; ///< Full tensor (only valid on root)
        int root;                    ///< Rank to gather to (usually 0)
        int dim;                     ///< Dimension that was sliced
        const MPITopology* topology;
    };
    
    bool execute(IDeviceContext* ctx) override;
    ComputeStageType type() const override { return ComputeStageType::GATHER; }
};

// =============================================================================
// AllGatherStage: Collect slices to ALL ranks
// =============================================================================

/**
 * @brief AllGather tensor slices so all ranks have full tensor
 * 
 * Used when subsequent computation needs full tensor (e.g., attention needs full K/V).
 */
class AllGatherStage : public IComputeStage {
public:
    struct Params {
        TensorBase* local_input;     ///< This rank's local slice  
        TensorBase* full_output;     ///< Full tensor on ALL ranks
        int dim;                     ///< Dimension that was sliced
        const MPITopology* topology;
    };
    
    bool execute(IDeviceContext* ctx) override;
    ComputeStageType type() const override { return ComputeStageType::ALLGATHER; }
};

// =============================================================================
// ScatterStage: Distribute tensor from root to all ranks
// =============================================================================

/**
 * @brief Scatter tensor from coordinator to all ranks
 * 
 * Used to distribute input tokens at start of forward pass.
 */
class ScatterStage : public IComputeStage {
public:
    struct Params {
        TensorBase* full_input;      ///< Full tensor (only valid on root)
        TensorBase* local_output;    ///< This rank's slice
        int root;                    ///< Rank that has full input (usually 0)
        int dim;                     ///< Dimension to scatter along
        const MPITopology* topology;
    };
    
    bool execute(IDeviceContext* ctx) override;
    ComputeStageType type() const override { return ComputeStageType::SCATTER; }
};

// =============================================================================
// ReduceScatterStage: Reduce + Scatter in one operation
// =============================================================================

/**
 * @brief ReduceScatter for efficient row-parallel GEMM
 * 
 * Combines partial results and distributes slices in one collective.
 * More efficient than Allreduce when subsequent op only needs a slice.
 */
class ReduceScatterStage : public IComputeStage {
public:
    struct Params {
        TensorBase* input;           ///< Full partial result on each rank
        TensorBase* output;          ///< Reduced slice for this rank
        MPI_Op op;                   ///< Reduction operation (usually MPI_SUM)
        const MPITopology* topology;
    };
    
    bool execute(IDeviceContext* ctx) override;
    ComputeStageType type() const override { return ComputeStageType::REDUCE_SCATTER; }
};

} // namespace llaminar2
```

### 3. Tensor-Parallel Graph Structure

Here's how a layer would look with explicit tensor parallelism:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         TENSOR-PARALLEL TRANSFORMER LAYER                       │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                    ATTENTION (Column-Parallel Q/K/V)                       │ │
│  │                                                                            │ │
│  │  hidden ──► RMSNorm ──► normalized                                         │ │
│  │                              │                                             │ │
│  │              ┌───────────────┼───────────────┐                             │ │
│  │              ▼               ▼               ▼                             │ │
│  │         ┌─────────┐    ┌─────────┐    ┌─────────┐                          │ │
│  │         │ Wq GEMM │    │ Wk GEMM │    │ Wv GEMM │  ◄── Column-parallel     │ │
│  │         │ (local) │    │ (local) │    │ (local) │      Each rank has       │ │
│  │         └────┬────┘    └────┬────┘    └────┬────┘      Wq[:, slice]        │ │
│  │              │              │              │                               │ │
│  │              ▼              ▼              ▼                               │ │
│  │         Q_local        K_local        V_local                              │ │
│  │         [m, heads/r]   [m, kv_h/r]    [m, kv_h/r]                          │ │
│  │              │              │              │                               │ │
│  │              ▼              ▼              ▼                               │ │
│  │         ┌─────────┐    ┌─────────┐    ┌─────────┐                          │ │
│  │         │  RoPE   │    │  RoPE   │    │   ---   │  ◄── Local RoPE          │ │
│  │         └────┬────┘    └────┬────┘    └────┬────┘                          │ │
│  │              │              │              │                               │ │
│  │              │         ┌────┴────┐         │                               │ │
│  │              │         │KV Cache │         │  ◄── Local KV cache           │ │
│  │              │         │ Append  │         │      (sharded by heads)       │ │
│  │              │         └────┬────┘         │                               │ │
│  │              │              │              │                               │ │
│  │              └──────────────┼──────────────┘                               │ │
│  │                             ▼                                              │ │
│  │                    ┌─────────────────┐                                     │ │
│  │                    │ Local Attention │  ◄── Each rank computes             │ │
│  │                    │ (heads/world_sz)│      attention for its heads        │ │
│  │                    └────────┬────────┘                                     │ │
│  │                             │                                              │ │
│  │                             ▼                                              │ │
│  │                      attn_local                                            │ │
│  │                      [m, heads/r × head_dim]                               │ │
│  │                             │                                              │ │
│  │                             ▼                                              │ │
│  │                    ┌─────────────────┐                                     │ │
│  │                    │    Wo GEMM      │  ◄── Row-parallel: Wo[slice, :]     │ │
│  │                    │    (local)      │      Each rank has partial Wo       │ │
│  │                    └────────┬────────┘                                     │ │
│  │                             │                                              │ │
│  │                             ▼                                              │ │
│  │                    ┌─────────────────┐                                     │ │
│  │                    │   AllReduce     │  ◄── Sum partial results            │ │
│  │                    │    (SUM)        │      All ranks get full output      │ │
│  │                    └────────┬────────┘                                     │ │
│  │                             │                                              │ │
│  │                             ▼                                              │ │
│  │                       attn_output                                          │ │
│  │                       [m, d_model]  ◄── Full on all ranks                  │ │
│  │                             │                                              │ │
│  └─────────────────────────────┼──────────────────────────────────────────────┘ │
│                                │                                                │
│                                ▼                                                │
│                         ResidualAdd                                             │
│                                │                                                │
│  ┌─────────────────────────────┼──────────────────────────────────────────────┐ │
│  │                         FFN (Column-Parallel Gate/Up)                      │ │
│  │                                                                            │ │
│  │  hidden ──► RMSNorm ──► normalized                                         │ │
│  │                              │                                             │ │
│  │              ┌───────────────┴───────────────┐                             │ │
│  │              ▼                               ▼                             │ │
│  │         ┌─────────┐                    ┌─────────┐                         │ │
│  │         │Gate GEMM│                    │ Up GEMM │  ◄── Column-parallel    │ │
│  │         │ (local) │                    │ (local) │      Gate[:, slice]     │ │
│  │         └────┬────┘                    └────┬────┘      Up[:, slice]       │ │
│  │              │                              │                              │ │
│  │              └──────────────┬───────────────┘                              │ │
│  │                             ▼                                              │ │
│  │                    ┌─────────────────┐                                     │ │
│  │                    │     SwiGLU      │  ◄── Local SwiGLU on slice          │ │
│  │                    │    (local)      │                                     │ │
│  │                    └────────┬────────┘                                     │ │
│  │                             │                                              │ │
│  │                             ▼                                              │ │
│  │                    ┌─────────────────┐                                     │ │
│  │                    │   Down GEMM     │  ◄── Row-parallel: Down[slice, :]   │ │
│  │                    │    (local)      │                                     │ │
│  │                    └────────┬────────┘                                     │ │
│  │                             │                                              │ │
│  │                             ▼                                              │ │
│  │                    ┌─────────────────┐                                     │ │
│  │                    │   AllReduce     │  ◄── Sum partial results            │ │
│  │                    │    (SUM)        │                                     │ │
│  │                    └────────┬────────┘                                     │ │
│  │                             │                                              │ │
│  │                             ▼                                              │ │
│  │                       ffn_output                                           │ │
│  │                       [m, d_model]                                         │ │
│  │                             │                                              │ │
│  └─────────────────────────────┼──────────────────────────────────────────────┘ │
│                                │                                                │
│                                ▼                                                │
│                         ResidualAdd                                             │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 4. Weight Sharding Strategy (Full)

| Weight | Parallelism | Rank i Has | Collective After |
|--------|-------------|------------|------------------|
| `embed_table` | **Replicated** | Full [V, D] | None (fast lookup) |
| `attn_norm.gamma` | **Replicated** | Full [D] | None |
| `wq` | **Column-parallel** | [D, heads_i × head_dim] | None (concat implicit) |
| `wk` | **Column-parallel** | [D, kv_heads_i × head_dim] | None |
| `wv` | **Column-parallel** | [D, kv_heads_i × head_dim] | None |
| `wo` | **Row-parallel** | [heads_i × head_dim, D] | **AllReduce(SUM)** |
| `ffn_norm.gamma` | **Replicated** | Full [D] | None |
| `gate_proj` | **Column-parallel** | [D, ffn_dim/r] | None |
| `up_proj` | **Column-parallel** | [D, ffn_dim/r] | None |
| `down_proj` | **Row-parallel** | [ffn_dim/r, D] | **AllReduce(SUM)** |
| `final_norm.gamma` | **Replicated** | Full [D] | None |
| `lm_head` | **Column-parallel** | [D, vocab/r] | **AllGather** (rank 0 for sampling) |

### 5. LM Head: Special Handling

The LM head is expensive (vocab_size × d_model ≈ 151K × 896 for Qwen2.5-0.5B). Options:

**Option A: Column-Parallel + AllGather (Full Logits)**
```
Rank 0: lm_head[:, 0:vocab/2]  → logits[:, 0:vocab/2]
Rank 1: lm_head[:, vocab/2:]   → logits[:, vocab/2:]
                                     │
                              AllGather to rank 0
                                     │
                              Full logits on rank 0 for sampling
```

**Option B: Column-Parallel + Reduce to get argmax (Efficient for Greedy)**
```
Each rank: local_argmax = argmax(local_logits)
           local_max = max(local_logits)
                                     │
                              AllReduce(MAX) + voting
                                     │
                              Global argmax without full logits transfer
```

Option B is more efficient for greedy sampling but requires custom reduction.

### 6. KV Cache Sharding

With head-parallel attention, each rank only needs to cache its assigned heads:

```cpp
// Before (replicated):
KVCache: [n_layers, batch_size, max_seq, n_kv_heads, head_dim]
         ≈ 24 × 1 × 2048 × 2 × 64 × 4B = 25MB per rank (duplicated!)

// After (sharded):
KVCache: [n_layers, batch_size, max_seq, n_kv_heads/world_size, head_dim]
         ≈ 24 × 1 × 2048 × 1 × 64 × 4B = 12.5MB per rank (halved)
```

**Challenge**: For GQA, n_kv_heads (2) may be less than world_size (2). In this case, one rank handles all KV heads for half the layers, the other for the other half (pipeline parallelism hybrid).

### 7. Coordinator vs Worker Model

Rather than separate graph classes, use a **role-aware execution model**:

```cpp
enum class RankRole {
    Coordinator,  // Rank 0: handles I/O, sampling, may have extra stages
    Worker        // Ranks 1..n: pure compute
};

class DistributedGraphOrchestrator {
public:
    // All ranks execute the same forward graph
    const float* forward(const int* tokens, int seq_len, int batch_size);
    
private:
    RankRole role_;
    std::shared_ptr<MPITopology> topology_;
    
    // Role-specific behavior encapsulated in stages, not separate graphs
    // - InputStage: Only coordinator reads tokens, then Bcast
    // - OutputStage: Only coordinator samples from gathered logits
};
```

**Graph Structure (same for all ranks):**
```
[InputStage]        ← Coordinator: read tokens; Workers: receive Bcast
    │
[EmbeddingStage]    ← All ranks: lookup (replicated embed table)
    │
[Layer 0..n-1]      ← All ranks: compute local portions
    │
[FinalNormStage]    ← All ranks: local norm (replicated gamma)
    │
[LMHeadStage]       ← All ranks: local GEMM (column-parallel lm_head)
    │
[GatherStage]       ← All ranks → Coordinator: gather logits
    │
[SampleStage]       ← Coordinator: sample next token; Workers: receive Bcast
```

## Migration Path

### Phase 1: MPITopology (Foundation) ✅ COMPLETE
1. ✅ Create `MPITopology` class with topology detection
2. ✅ Add `WorkRange` struct for simple index ranges
3. ✅ Add work distribution methods (`get_head_range`, `get_column_range`, etc.)
4. ✅ Device capability exchange via AllGather at startup

### Phase 2: Stage-Level TP Integration (Current Focus)

Extend existing `ComputeStage` classes to accept TP parameters from `MPITopology`.

#### 2.1 GEMMStage TP Support

**Current State**: `GEMMStage` has `mpi_ctx` parameter but does NOT use it for sharding.

**Kernel Support Available**:
- ✅ `QuantisedGemmKernel(weights, row_start, row_end)` - Row-sliced constructor exists
- ✅ `KernelFactory::getOrCreateGemm()` - Caches packed weights
- ❌ `KernelFactory::getOrCreateGemmSliced()` - Needs implementation

**Changes Required**:
```cpp
struct GEMMStage::Params {
    // ... existing fields ...
    
    // TP parameters (optional, defaults to full computation)
    std::optional<WorkRange> output_range;  // For row-parallel (N dimension)
    std::optional<WorkRange> input_range;   // For column-parallel (K dimension)
    bool needs_allreduce = false;           // After column-parallel GEMM
    bool needs_allgather = false;           // After row-parallel GEMM
};
```

**Implementation**:
1. Add `KernelFactory::getOrCreateGemmSliced(tensor, row_start, row_end)`
2. Modify `GEMMStage::execute()` to use sliced kernel when `output_range` set
3. Integrate allreduce/allgather post-GEMM (or chain with separate stage)

#### 2.2 AttentionComputeStage TP Support

**Current State**: Passes full `n_heads` to kernel. Comment explicitly says:
> "mpi_ctx for distributed logging, not tensor-parallel here"

**Kernel Support Available**:
- ✅ `CPUAttentionKernelTyped` loops `for h in 0..n_heads` - easy to modify
- ✅ Per-head computation is independent (embarrassingly parallel)
- ❌ No `head_start/head_end` parameters currently

**Changes Required**:
```cpp
struct AttentionComputeStage::Params {
    // ... existing fields ...
    
    // TP parameters (optional)
    int head_start = 0;          // First head for this rank
    int local_n_heads = -1;      // -1 = use n_heads (full computation)
    int local_n_kv_heads = -1;   // -1 = use n_kv_heads (full computation)
};
```

**Implementation**:
1. Add head range params to `ITensorAttention::compute_tensor()`
2. Modify `CPUAttentionKernelTyped` loop bounds: `for (h = head_start; h < head_end; ++h)`
3. GPU kernels: Adjust thread block grid to only cover local heads

#### 2.3 LMHeadStage TP Support (Future)

**Current State**: Computes full `[m, vocab_size]` logits on every rank.

**Changes Required**: Column-parallel with AllGather or efficient argmax reduction.

### Phase 3: Column-Parallel QKV
1. Update `WeightManager` to shard Wq/Wk/Wv by columns (output dim)
2. Modify `Qwen2Graph::buildAttentionGraph` to pass `WorkRange` to stages
3. Each rank computes Q/K/V for its assigned heads only

### Phase 4: Column-Parallel FFN
1. Update `WeightManager` to shard Gate/Up by columns
2. Modify `Qwen2Graph::buildFFNGraph` for local FFN dim
3. Add `AllreduceStage` after Down projection

### Phase 5: Column-Parallel LM Head
1. Shard `lm_head` by vocab dimension
2. Add `AllGatherStage` before sampling
3. Implement efficient argmax reduction (optional)

### Phase 6: Sharded KV Cache
1. Modify `UnifiedKVCache` to only store local heads
2. Update cache append stages for local head indices

### Phase 7: Cleanup
1. ⛔ Remove `MpiAttentionOrchestrator` (after TP migrated to stages)
2. ⛔ Remove `GQAAttention` class
3. ⛔ Deprecate `PipelineBase` TP methods (`project_row_parallel`, etc.)
4. Update tests to use graph-based execution only

---

## GPU Integration Phases

The following phases run in parallel with or after the CPU tensor parallelism phases above.

### Phase G1: Single-GPU Per Rank (NUMA-Filtered)

**Goal**: Each MPI rank uses one GPU for GEMM, CPU for other ops.

**Prerequisites**: Phase 2 (Stage-Level TP) complete

**Tasks**:
1. ✅ `DeviceManager` already filters GPUs by NUMA affinity
2. Configure `GraphOrchestrator` to use `device_idx` from DeviceManager (not hardcoded 0)
3. Update `GEMMStage` to select GPU kernel when `device_idx` points to GPU
4. Validate CUDA quantized GEMM (JIT + CUTLASS) works with TP sliced weights
5. Add integration test: 2-rank MPI with each rank on separate GPU

**Expected Outcome**: GEMM runs on GPU, attention/norms run on CPU with automatic fallback

### Phase G2: GPU Attention Kernels

**Goal**: Run attention computation on GPU (eliminates biggest CPU bottleneck)

**Prerequisites**: Phase G1 complete

**Tasks**:
1. Integrate FlashAttention-2 for CUDA:
   - Add `CudaFlashAttentionKernel` implementing `ITensorAttention`
   - Wire into `KernelFactory::createAttention()` for `DeviceType::CUDA`
2. Add hipFlashAttention for ROCm (or MIOpen attention)
3. Update `AttentionComputeStage` to pass device context to kernel
4. Add head-range support to GPU attention (for TP)

**Implementation Note**:
```cpp
// KernelFactory.cpp - Phase G2 target
case DeviceType::CUDA:
    return std::make_unique<CudaFlashAttentionKernel>(/* head_start, head_end */);
```

### Phase G3: Cross-Device Data Transfer

**Goal**: Enable stages to run on different devices within same rank

**Prerequisites**: Phase G2 complete

**Tasks**:
1. Add `DataTransferStage` to compute graph:
   ```cpp
   class DataTransferStage : public IComputeStage {
       Params: src_device, dst_device, tensor, async
   };
   ```
2. Or: Automatic transfer detection in `DeviceGraphExecutor`:
   - Track each tensor's current device
   - Insert implicit copies when stage device differs from input device
3. Profile and optimize: Prefer keeping data on one device when possible

### Phase G4: Heterogeneous Work Distribution

**Goal**: Split work between CPU and GPU based on compute power

**Prerequisites**: Phase G3 complete

**Tasks**:
1. Add startup calibration routine:
   - Run small GEMM on each device
   - Measure GFLOPS, store in `DeviceCapability::relative_compute`
2. Update `MPITopology::get_*_range()` to use calibrated weights
3. Align GPU work ranges to warp-friendly boundaries (multiple of 32)
4. Add `DevicePreference` to stage params:
   ```cpp
   enum class DevicePreference { REQUIRE_GPU, PREFER_GPU, CPU_ONLY, ANY };
   ```

### Phase G5: Multi-GPU Per Rank (Future)

**Goal**: Single MPI rank drives multiple GPUs with data parallelism

**Prerequisites**: Phase G4 complete, multi-GPU hardware available

**Tasks**:
1. Extend `RankPlacement::devices` to track multiple GPUs per rank
2. Add `DataParallelStage` that replicates computation across devices
3. Implement gradient-style reduction for data-parallel outputs
4. Consider NCCL for intra-rank GPU-GPU communication

---

## Memory Savings Analysis

For Qwen2.5-0.5B with 2 ranks:

| Component | Before (per rank) | After (per rank) | Savings |
|-----------|-------------------|------------------|---------|
| Wq (per layer) | 896 × 896 × 4B = 3.2MB | 896 × 448 × 4B = 1.6MB | 50% |
| Wk (per layer) | 896 × 128 × 4B = 458KB | 896 × 64 × 4B = 229KB | 50% |
| Wv (per layer) | 896 × 128 × 4B = 458KB | 896 × 64 × 4B = 229KB | 50% |
| Gate (per layer) | 896 × 4864 × 4B = 17.4MB | 896 × 2432 × 4B = 8.7MB | 50% |
| Up (per layer) | 896 × 4864 × 4B = 17.4MB | 896 × 2432 × 4B = 8.7MB | 50% |
| LM Head | 896 × 151936 × 4B = 544MB | 896 × 75968 × 4B = 272MB | 50% |
| KV Cache | 24 × 2048 × 128 × 4B = 25MB | 24 × 2048 × 64 × 4B = 12.5MB | 50% |
| **Total Model** | ~2.8GB | ~1.5GB | **46%** |

For larger models (7B+), savings are even more significant as model size dominates.

## Open Questions

1. **GQA with few KV heads**: When `n_kv_heads < world_size`, how to distribute?
   - Option: Hybrid tensor + pipeline parallelism
   - Option: Replicate KV heads, only shard Q heads

2. **Batch parallelism**: Should we also support data parallelism (different sequences on different ranks)?

3. **Sequence parallelism**: For long sequences, split sequence dimension across ranks?

4. **Communication optimization**: Use NCCL for GPU-GPU, MPI for CPU-CPU?

## Conclusion

This proposal outlines a clean tensor-parallel architecture that:
- Makes MPI communication explicit in the compute graph
- Properly separates topology concerns
- Eliminates redundant computation
- Scales memory usage with number of ranks

The migration can be done incrementally, with each phase delivering measurable benefits.
