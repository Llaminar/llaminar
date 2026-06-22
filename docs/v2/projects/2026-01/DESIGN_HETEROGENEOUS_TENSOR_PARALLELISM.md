# Heterogeneous Tensor Parallelism Design

## Overview

This document describes the architecture for **intra-node multi-device tensor parallelism** supporting heterogeneous mixes of CPU, CUDA, and ROCm devices. The design extends the existing MPI-based inter-node tensor parallelism to support:

1. **Intra-node GPU-GPU communication** via NCCL (NVIDIA) and RCCL (AMD)
2. **Heterogeneous device mixes** (CPU + CUDA + ROCm on same node)
3. **Backend-agnostic model graphs** - models declare WHAT collectives, not HOW
4. **Hierarchical tensor parallelism**: within-node (NCCL/RCCL/Host) → across-nodes (MPI)

---

## Design Principles

### Separation of Concerns

| Layer | Responsibility | Does NOT know about |
|-------|---------------|---------------------|
| **Model Graph** (Qwen2Graph) | Declares computational DAG with abstract collective nodes | NCCL, RCCL, MPI, device groups |
| **Collective Layer** (CollectiveStage) | Abstract collective operations | Backend implementations |
| **Backend Router** | Selects backend based on device topology | Model structure |
| **Backend Implementations** | Execute collectives on specific devices | Other backends |

### Model Graphs Stay Pure

**❌ WRONG - Backend details in model graph:**
```cpp
// Pollutes model graph with orchestration concerns
graph.addNode("wo_allreduce",
    ComputeStageFactory::createAllreduce(
        AllreduceStage::Params{
            .buffer = buffer,
            .backend = CollectiveBackendType::NCCL,  // ← BAD: Model knows about NCCL
            .device_group = cuda_group               // ← BAD: Model manages device groups
        }));
```

**✅ CORRECT - Abstract collective in model graph:**
```cpp
// Model only declares WHAT collective is needed
graph.addNode("wo_allreduce",
    ComputeStageFactory::createAllreduce(
        AllreduceStage::Params{
            .buffer = buffer,
            .count = count
        }));

// Runtime/orchestrator resolves HOW to execute it
```

The `DeviceGraphExecutor` or a `CollectiveResolver` layer binds abstract collective stages to concrete backends at execution time, based on:
- Device topology discovered at startup
- Runtime configuration (prefer NCCL? Force MPI?)
- Tensor locations (which device is the buffer on?)

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODEL LAYER                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  Qwen2Graph                                                              ││
│  │    ├── WoProjectionStage                                                ││
│  │    ├── AllreduceStage  ←─── ABSTRACT (just buffer + count)             ││
│  │    ├── ResidualAddStage                                                 ││
│  │    └── ...                                                              ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│        │ declares WHAT                                                       │
│        ▼                                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                         EXECUTION LAYER                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  DeviceGraphExecutor                                                           ││
│  │    │                                                                     ││
│  │    └── CollectiveContext  ←─── Decides HOW to execute collectives      ││
│  │          │                                                               ││
│  │          └── BackendRouter (selects optimal backend)                    ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│        │ routes to                                                           │
│        ▼                                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                         BACKEND LAYER (Internal)                             │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐│
│  │ MPIBackend │ │NCCLBackend │ │RCCLBackend │ │PCIeBARBack │ │HostBackend ││
│  │            │ │            │ │            │ │            │ │            ││
│  │MPI_Allreduce││ncclAllReduce││rcclAllReduce││DirectP2P   ││ memcpy +   ││
│  │MPI_Allgather││ncclAllGather││rcclAllGather││via BAR map ││ host sync  ││
│  │            │ │            │ │            │ │            │ │            ││
│  │ inter-node │ │ CUDA-CUDA  │ │ ROCm-ROCm  │ │ CUDA↔ROCm  │ │ any↔any   ││
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘ └────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Insight**: The model graph only knows it needs "an allreduce here". Everything else is resolved at runtime by the execution layer.

**PCIe BAR Backend**: When CUDA and ROCm GPUs need to communicate, the router prefers `PCIeBARBackend` (direct ~2.65 GB/s) over `HostBackend` (staged ~1.5 GB/s) when BAR P2P is available.

---

## 1. Collective Communication Abstraction (Internal)

The following interfaces are **internal implementation details** - model graphs never see them.

### 1.1 Backend Taxonomy

| Backend | Scope | Devices | Library | Use Case |
|---------|-------|---------|---------|----------|
| **MPI** | Inter-node | Any (via host) | OpenMPI/MPICH | Cross-machine communication |
| **NCCL** | Intra-node | CUDA GPUs | NVIDIA NCCL | NVLink/PCIe GPU-GPU |
| **RCCL** | Intra-node | ROCm GPUs | AMD RCCL | Infinity Fabric GPU-GPU |
| **PCIeBAR** | Intra-node | CUDA ↔ ROCm | DirectP2PEngine | Cross-vendor GPU direct P2P |
| **Host** | Intra-node | CPU ↔ GPU | memcpy/DMA | Heterogeneous fallback |

### 1.2 Interface: `ICollectiveBackend`

```cpp
// src/v2/collective/ICollectiveBackend.h

namespace llaminar2 {

enum class CollectiveBackendType {
    MPI,      // MPI_Allreduce, MPI_Allgather (inter-node or CPU-only)
    NCCL,     // ncclAllReduce, ncclAllGather (NVIDIA GPUs)
    RCCL,     // rcclAllReduce, rcclAllGather (AMD GPUs)  
    PCIE_BAR, // DirectP2PEngine (CUDA ↔ ROCm direct via PCIe BAR)
    HOST,     // CPU↔GPU via staged host buffer
    AUTO      // Runtime selection based on tensor locations
};

enum class CollectiveOp {
    ALLREDUCE_SUM,
    ALLREDUCE_MAX,
    ALLREDUCE_MIN,
    ALLGATHER,
    REDUCE_SCATTER,
    BROADCAST
};

/**
 * @brief Abstract interface for collective communication backends
 * 
 * Each backend implements device-specific collective operations.
 * Backends are stateful (hold communicator handles, streams, etc.)
 */
class ICollectiveBackend {
public:
    virtual ~ICollectiveBackend() = default;
    
    /// Backend identifier
    virtual CollectiveBackendType type() const = 0;
    virtual std::string name() const = 0;
    
    /// Capability queries
    virtual bool supportsDevice(DeviceType type) const = 0;
    virtual bool supportsDirectTransfer(DeviceId src, DeviceId dst) const = 0;
    
    /// Collective operations
    virtual bool allreduce(
        void* buffer,              // In-place buffer
        size_t count,              // Element count
        DataType dtype,            // FP32, BF16, etc.
        CollectiveOp op,           // SUM, MAX, etc.
        const DeviceGroup& group   // Participating devices
    ) = 0;
    
    virtual bool allgather(
        const void* send_buf,      // Local slice
        void* recv_buf,            // Full output buffer
        size_t send_count,         // Elements per device
        DataType dtype,
        const DeviceGroup& group
    ) = 0;
    
    virtual bool reduceScatter(
        const void* send_buf,      // Full input buffer
        void* recv_buf,            // Local slice output
        size_t recv_count,         // Elements per device
        DataType dtype,
        CollectiveOp op,
        const DeviceGroup& group
    ) = 0;
    
    /// Synchronization
    virtual bool synchronize() = 0;
    
    /// Resource management
    virtual bool initialize(const DeviceGroup& group) = 0;
    virtual void shutdown() = 0;
};

} // namespace llaminar2
```

### 1.3 Device Group Concept

A `DeviceGroup` represents a set of devices that participate in a collective operation:

```cpp
// src/v2/collective/DeviceGroup.h

namespace llaminar2 {

/**
 * @brief A group of devices participating in collective operations
 * 
 * DeviceGroups can be:
 * - Homogeneous: All CUDA GPUs, all ROCm GPUs, or all CPUs
 * - Heterogeneous: Mix of device types (requires HOST backend fallback)
 * 
 * Groups are hierarchical:
 * - LocalGroup: Devices within a single MPI rank (intra-node)
 * - GlobalGroup: Devices across all MPI ranks (inter-node via MPI)
 */
struct DeviceGroup {
    std::string name;                        // e.g., "cuda_gpus_rank0", "all_devices_node0"
    std::vector<DeviceId> devices;           // Ordered list of participating devices
    int local_rank;                          // This process's index in group (0 to size-1)
    
    // Topology hints
    bool is_homogeneous = true;              // All same DeviceType?
    DeviceType primary_type = DeviceType::CPU;
    
    // Backend preference (resolved at runtime)
    CollectiveBackendType preferred_backend = CollectiveBackendType::AUTO;
    
    // Derived
    size_t size() const { return devices.size(); }
    DeviceId localDevice() const { return devices[local_rank]; }
    
    // Predicates
    bool allCUDA() const;
    bool allROCm() const;
    bool allCPU() const;
    bool hasGPU() const;
    bool isHeterogeneous() const { return !is_homogeneous; }
};

/**
 * @brief Factory for creating device groups from ClusterInventory
 */
class DeviceGroupFactory {
public:
    /// Create group of all CUDA GPUs on this rank
    static DeviceGroup createLocalCUDAGroup(const RankInventory& inv, int local_device_idx);
    
    /// Create group of all ROCm GPUs on this rank
    static DeviceGroup createLocalROCmGroup(const RankInventory& inv, int local_device_idx);
    
    /// Create group of all devices on this rank (heterogeneous)
    static DeviceGroup createLocalAllDevicesGroup(const RankInventory& inv, int local_device_idx);
    
    /// Create group spanning all ranks (for MPI collectives)
    static DeviceGroup createGlobalGroup(const ClusterInventory& inv, int rank, DeviceId local_device);
};

} // namespace llaminar2
```

---

## 2. Backend Implementations

### 2.1 MPI Backend (Existing, Renamed)

```cpp
// src/v2/collective/backends/MPIBackend.h

class MPIBackend : public ICollectiveBackend {
public:
    explicit MPIBackend(const MPIContext* mpi_ctx);
    
    CollectiveBackendType type() const override { return CollectiveBackendType::MPI; }
    std::string name() const override { return "MPI"; }
    
    bool supportsDevice(DeviceType type) const override {
        // MPI works with any device via host staging
        return true;
    }
    
    bool supportsDirectTransfer(DeviceId src, DeviceId dst) const override {
        // MPI always goes through host memory (no GPU-direct in basic MPI)
        return src.isCPU() && dst.isCPU();
    }
    
    bool allreduce(...) override;  // MPI_Allreduce
    bool allgather(...) override;  // MPI_Allgather
    
private:
    const MPIContext* mpi_ctx_;
    
    // Host staging buffers for GPU data
    std::unique_ptr<float[]> host_staging_buffer_;
    size_t staging_buffer_size_ = 0;
    
    void ensureStagingBuffer(size_t bytes);
    void copyToHost(const void* device_ptr, DeviceId device, size_t bytes);
    void copyFromHost(void* device_ptr, DeviceId device, size_t bytes);
};
```

### 2.2 NCCL Backend (New)

```cpp
// src/v2/collective/backends/NCCLBackend.h

#ifdef HAVE_NCCL
#include <nccl.h>
#endif

class NCCLBackend : public ICollectiveBackend {
public:
    NCCLBackend();
    ~NCCLBackend() override;
    
    CollectiveBackendType type() const override { return CollectiveBackendType::NCCL; }
    std::string name() const override { return "NCCL"; }
    
    bool supportsDevice(DeviceType type) const override {
        return type == DeviceType::CUDA;
    }
    
    bool supportsDirectTransfer(DeviceId src, DeviceId dst) const override {
        return src.isCUDA() && dst.isCUDA();
    }
    
    bool initialize(const DeviceGroup& group) override;
    bool allreduce(...) override;  // ncclAllReduce
    bool allgather(...) override;  // ncclAllGather
    void shutdown() override;
    
private:
#ifdef HAVE_NCCL
    ncclComm_t comm_ = nullptr;
    cudaStream_t stream_ = nullptr;
    std::vector<int> device_ids_;  // CUDA device ordinals in group
#endif
    
    bool initialized_ = false;
};
```

### 2.3 RCCL Backend (New, ROCm)

```cpp
// src/v2/collective/backends/RCCLBackend.h

#ifdef HAVE_RCCL
#include <rccl/rccl.h>
#endif

class RCCLBackend : public ICollectiveBackend {
public:
    RCCLBackend();
    ~RCCLBackend() override;
    
    CollectiveBackendType type() const override { return CollectiveBackendType::RCCL; }
    std::string name() const override { return "RCCL"; }
    
    bool supportsDevice(DeviceType type) const override {
        return type == DeviceType::ROCm;
    }
    
    bool supportsDirectTransfer(DeviceId src, DeviceId dst) const override {
        return src.isROCm() && dst.isROCm();
    }
    
    bool initialize(const DeviceGroup& group) override;
    bool allreduce(...) override;  // rcclAllReduce
    bool allgather(...) override;  // rcclAllGather
    void shutdown() override;
    
private:
#ifdef HAVE_RCCL
    rcclComm_t comm_ = nullptr;
    hipStream_t stream_ = nullptr;
    std::vector<int> device_ids_;  // ROCm device ordinals
#endif
    
    bool initialized_ = false;
};
```

### 2.4 Host Backend (Heterogeneous Fallback)

```cpp
// src/v2/collective/backends/HostBackend.h

/**
 * @brief Host-memory-staged collective operations
 * 
 * Used when devices cannot communicate directly:
 * - CPU ↔ GPU transfers
 * - CUDA ↔ ROCm transfers (no direct path)
 * - Fallback when NCCL/RCCL unavailable
 * 
 * Algorithm for allreduce across heterogeneous devices:
 * 1. Each device copies its buffer to host
 * 2. CPU performs reduction on host buffers
 * 3. Result broadcast back to all devices
 */
class HostBackend : public ICollectiveBackend {
public:
    HostBackend();
    
    CollectiveBackendType type() const override { return CollectiveBackendType::HOST; }
    std::string name() const override { return "Host"; }
    
    bool supportsDevice(DeviceType type) const override {
        return true;  // Works with any device via memcpy
    }
    
    bool supportsDirectTransfer(DeviceId src, DeviceId dst) const override {
        return true;  // Always via host staging
    }
    
    bool allreduce(...) override;
    bool allgather(...) override;
    
private:
    // Per-device staging buffers (pinned memory for async transfers)
    std::unordered_map<DeviceId, std::unique_ptr<float[]>> staging_buffers_;
    
    // Device-specific copy functions
    void copyToHost(void* host_dst, const void* device_src, DeviceId device, size_t bytes);
    void copyFromHost(void* device_dst, const void* host_src, DeviceId device, size_t bytes);
    void synchronizeDevice(DeviceId device);
};
```

### 2.5 PCIe BAR Backend (Cross-Vendor Direct P2P)

```cpp
// src/v2/collective/backends/PCIeBARBackend.h

/**
 * @brief Direct GPU-to-GPU communication via PCIe BAR mapping
 * 
 * Enables CUDA ↔ ROCm direct transfers WITHOUT host staging by mapping
 * AMD GPU's BAR (Base Address Register) into CUDA's address space.
 * 
 * Performance characteristics (measured on RTX 3090 ↔ MI50):
 * - Bandwidth: ~2.65 GB/s (PCIe 3.0 x16 limited)
 * - Latency: Lower than host-staged (no CPU involvement)
 * - Symmetric: Read/write speeds nearly identical
 * 
 * Requirements:
 * - AMD GPU with large BAR support (e.g., MI50 with 32GB BAR)
 * - CAP_SYS_ADMIN or appropriate permissions for BAR access
 * - CUDA driver API for cuMemHostRegister
 * 
 * Algorithm for allreduce (CUDA + ROCm):
 * 1. CUDA computes partial result in its buffer
 * 2. ROCm computes partial result in its buffer  
 * 3. CUDA reads ROCm's result via PCIe BAR → local temp buffer
 * 4. CUDA performs reduction (local + remote)
 * 5. CUDA writes result back to ROCm via PCIe BAR
 * 6. Both GPUs now have identical reduced values
 */
class PCIeBARBackend : public ICollectiveBackend {
public:
    explicit PCIeBARBackend(DirectP2PEngine* engine);
    
    CollectiveBackendType type() const override { return CollectiveBackendType::PCIE_BAR; }
    std::string name() const override { return "PCIe_BAR"; }
    
    bool supportsDevice(DeviceType type) const override {
        return type == DeviceType::CUDA || type == DeviceType::ROCm;
    }
    
    bool supportsDirectTransfer(DeviceId src, DeviceId dst) const override {
        // True if one is CUDA and one is ROCm (and P2P is initialized)
        return engine_->isPCIeBarActive() &&
               ((src.isCUDA() && dst.isROCm()) || (src.isROCm() && dst.isCUDA()));
    }
    
    bool allreduce(
        void* buffer,
        size_t count,
        DataType dtype,
        CollectiveOp op,
        const DeviceGroup& group
    ) override;
    
    bool allgather(
        const void* send_buf,
        void* recv_buf,
        size_t send_count,
        DataType dtype,
        const DeviceGroup& group
    ) override;
    
    bool initialize(const DeviceGroup& group) override;
    void shutdown() override;
    bool synchronize() override;
    
private:
    DirectP2PEngine* engine_;  // Owns BAR mapping lifecycle
    
    // Internal buffers for reduction operations
    void* cuda_temp_buffer_ = nullptr;
    size_t cuda_temp_size_ = 0;
    
    // P2P transfer helpers (use engine internally)
    bool transferCUDAtoROCm(const void* cuda_src, size_t rocm_offset, size_t bytes);
    bool transferROCmtoCUDA(size_t rocm_offset, void* cuda_dst, size_t bytes);
};
```

**Key advantage over HostBackend**: No CPU involvement means:
- Lower latency (no PCIe round-trip through host memory)
- CPU is free to do other work during GPU communication
- Memory bandwidth not shared with host operations

---

## 3. Backend Router (Auto Selection)

```cpp
// src/v2/collective/BackendRouter.h

/**
 * @brief Routes collective operations to appropriate backend
 * 
 * Selection logic (priority order):
 * 1. If all devices are CUDA GPUs → NCCL
 * 2. If all devices are ROCm GPUs → RCCL  
 * 3. If crossing MPI rank boundaries → MPI
 * 4. If CUDA + ROCm and PCIe BAR P2P available → PCIE_BAR (direct)
 * 5. Otherwise → Host (heterogeneous fallback via CPU staging)
 * 
 * The PCIe BAR backend is preferred over Host when available because:
 * - ~2.65 GB/s direct transfer vs ~1.5 GB/s via host staging
 * - No CPU involvement (lower latency, CPU free for other work)
 * - Symmetric read/write performance
 */
class BackendRouter {
public:
    BackendRouter(
        std::shared_ptr<MPIContext> mpi_ctx,
        const ClusterInventory& cluster_inventory,
        std::unique_ptr<DirectP2PEngine> p2p_engine = nullptr  // Optional P2P support
    );
    
    /**
     * @brief Get optimal backend for a device group
     */
    ICollectiveBackend* getBackend(const DeviceGroup& group);
    
    /**
     * @brief Get backend by explicit type
     */
    ICollectiveBackend* getBackend(CollectiveBackendType type);
    
    /**
     * @brief Determine optimal backend for collective between devices
     */
    CollectiveBackendType selectBackend(
        const std::vector<DeviceId>& devices,
        bool crosses_rank_boundary
    ) const;
    
    /**
     * @brief Check if direct CUDA↔ROCm P2P is available
     */
    bool hasPCIeBarP2P() const { return p2p_engine_ && p2p_engine_->isPCIeBarActive(); }
    
private:
    std::shared_ptr<MPIContext> mpi_ctx_;
    ClusterInventory cluster_inventory_;
    
    // Lazily initialized backends
    std::unique_ptr<MPIBackend> mpi_backend_;
    std::unique_ptr<NCCLBackend> nccl_backend_;
    std::unique_ptr<RCCLBackend> rccl_backend_;
    std::unique_ptr<PCIeBARBackend> pcie_bar_backend_;  // Cross-vendor direct P2P
    std::unique_ptr<HostBackend> host_backend_;
    
    // P2P engine (owned, shared with PCIeBARBackend)
    std::unique_ptr<DirectP2PEngine> p2p_engine_;
    
    // Device group → initialized backend cache
    std::unordered_map<std::string, ICollectiveBackend*> group_backend_cache_;
    
    // Backend selection helper
    bool isCUDAROCmMix(const std::vector<DeviceId>& devices) const;
};
```

### 3.1 Backend Selection Flow

```
selectBackend(devices, crosses_rank):
    │
    ├─ crosses_rank_boundary? ────────────────────────► MPI
    │
    ├─ all CUDA? ─────────────────────────────────────► NCCL
    │
    ├─ all ROCm? ─────────────────────────────────────► RCCL
    │
    ├─ CUDA + ROCm mix?
    │   │
    │   ├─ hasPCIeBarP2P()? ──────────────────────────► PCIE_BAR (direct, ~2.65 GB/s)
    │   │
    │   └─ else ──────────────────────────────────────► HOST (staged, ~1.5 GB/s)
    │
    └─ default ───────────────────────────────────────► HOST
```

---

## 4. Compute Stage Architecture

### 4.1 Design: Model Graphs Stay Abstract

Model graphs (like `Qwen2Graph`) declare **abstract collective stages** without any backend knowledge:

```cpp
// In Qwen2Graph::buildAttentionGraph() - NO CHANGE from current code
graph.addNode(prefix + "wo_allreduce",
    ComputeStageFactory::createAllreduce(
        AllreduceStage::Params{
            .buffer = buffer,
            .count = count
        }),
    device);
```

The `AllreduceStage::Params` does NOT contain:
- ❌ `CollectiveBackendType backend` - no backend selection
- ❌ `DeviceGroup group` - no device group management
- ❌ `BackendRouter* router` - no router reference

**Why?** The model graph is a pure computational specification. Backend selection is an orthogonal runtime concern handled by the execution layer.

### 4.2 Backend Resolution: Two Approaches

#### Approach A: DeviceGraphExecutor Resolves at Execution Time (Recommended)

The `DeviceGraphExecutor` holds a `CollectiveContext` that knows how to execute collectives:

```cpp
// src/v2/execution/CollectiveContext.h

/**
 * @brief Runtime context for collective operations
 * 
 * Injected into DeviceGraphExecutor at construction time.
 * Encapsulates ALL collective backend knowledge.
 */
class CollectiveContext {
public:
    CollectiveContext(
        std::shared_ptr<MPIContext> mpi_ctx,
        const ClusterInventory& cluster_inventory
    );
    
    /// Execute an allreduce - selects backend automatically
    bool executeAllreduce(
        ITensor* buffer,
        size_t count,
        DeviceId tensor_device  // Where the tensor lives
    );
    
    /// Execute an allgather
    bool executeAllgather(
        ITensor* local_input,
        ITensor* full_output,
        size_t actual_seq_len,
        DeviceId tensor_device
    );
    
    /// Query: does this configuration require collectives?
    bool requiresCollectives() const { return world_size_ > 1 || local_devices_.size() > 1; }
    
private:
    std::shared_ptr<MPIContext> mpi_ctx_;
    int world_size_ = 1;
    std::vector<DeviceId> local_devices_;  // Devices on this rank
    
    // Backend instances (lazily initialized)
    std::unique_ptr<BackendRouter> router_;
};
```

The `DeviceGraphExecutor` uses this context when executing collective stages:

```cpp
// In DeviceGraphExecutor::executeStage()
void DeviceGraphExecutor::executeStage(const ComputeNode& node) {
    if (node.stage->type() == ComputeStageType::ALLREDUCE) {
        // Cast to get params
        auto* allreduce = static_cast<AllreduceStage*>(node.stage.get());
        collective_ctx_->executeAllreduce(
            allreduce->buffer(),
            allreduce->count(),
            node.device_id
        );
        return;
    }
    // ... normal stage execution
}
```

#### Approach B: Stage Receives Backend at Construction (Alternative)

A `CollectiveStageFactory` wraps `ComputeStageFactory` and injects backend:

```cpp
// Called by GraphOrchestrator, NOT by model graphs
auto stage = CollectiveStageFactory::createAllreduce(
    basic_params,
    collective_context  // Injected by orchestrator
);
```

**Recommendation**: Approach A is cleaner because stages remain stateless regarding backends.

### 4.3 Abstract Collective Stages (Model-Facing)

The stages that model graphs create are intentionally simple:

```cpp
// src/v2/execution/compute_stages/stages/AllreduceStage.h

/**
 * @brief Abstract AllReduce stage - declares WHAT, not HOW
 * 
 * Model graphs create these to declare "I need an allreduce here".
 * The DeviceGraphExecutor/CollectiveContext decides how to execute it.
 */
class AllreduceStage : public IComputeStage {
public:
    struct Params {
        ITensor* buffer = nullptr;  ///< Buffer to allreduce (in-place)
        size_t count = 0;           ///< Element count (0 = buffer->numel())
        // That's it! No backend, no device group, no router.
    };
    
    explicit AllreduceStage(Params params);
    
    // Accessors for DeviceGraphExecutor to use
    ITensor* buffer() const { return params_.buffer; }
    size_t count() const { return params_.count; }
    
    // IComputeStage interface
    bool execute(IDeviceContext* ctx) override;
    ComputeStageType type() const override { return ComputeStageType::ALLREDUCE; }
    CoherencePolicy coherencePolicy() const override { return CoherencePolicy::NONE; }
    
private:
    Params params_;
};
```

Note: The `execute()` method either:
- Delegates to `CollectiveContext` (Approach A)
- Uses an injected backend (Approach B)
- Falls back to legacy MPI path for backward compatibility

### 4.4 AllGatherStage (Same Pattern)

```cpp
class AllGatherStage : public IComputeStage {
public:
    struct Params {
        ITensor* local_input = nullptr;   // [seq_len, local_dim]
        ITensor* full_output = nullptr;   // [seq_len, full_dim]
        size_t actual_seq_len = 0;
        // No backend params!
    };
    
    explicit AllGatherStage(Params params);
    
    ITensor* localInput() const { return params_.local_input; }
    ITensor* fullOutput() const { return params_.full_output; }
    size_t actualSeqLen() const { return params_.actual_seq_len; }
    
    bool execute(IDeviceContext* ctx) override;
    ComputeStageType type() const override { return ComputeStageType::ALLGATHER; }
    CoherencePolicy coherencePolicy() const override { return CoherencePolicy::NONE; }
    
private:
    Params params_;
};
```

### 4.5 Backward Compatibility

The existing code passes `mpi_ctx` in stage params:

```cpp
// Current code in Qwen2Graph.cpp
AllreduceStage::Params{buffer, mpi_ctx_.get(), count}
```

For backward compatibility, keep `mpi_ctx` as an optional param:

```cpp
struct Params {
    ITensor* buffer = nullptr;
    size_t count = 0;
    
    // Legacy: if set, stage can execute standalone (used in tests)
    // If null, stage requires CollectiveContext from executor
    const MPIContext* mpi_ctx = nullptr;
};
```

When `mpi_ctx` is set, the stage works as before. When null, it requires the executor to provide a `CollectiveContext`.

---

## 5. Backend Implementations (Internal)

These are internal implementation details - NOT exposed to model graphs.

The `CollectiveContext` internally uses backend implementations:

```cpp
// Internal to CollectiveContext - model graphs never see these
class MPIBackend : public ICollectiveBackend { ... };
class NCCLBackend : public ICollectiveBackend { ... };
class RCCLBackend : public ICollectiveBackend { ... };
class HostBackend : public ICollectiveBackend { ... };
```

See Section 2 for backend interface definitions.

---

## 6. Intra-Node Tensor Parallelism

### 5.1 Multi-Device Weight Sharding

Extend `WeightManager` to support sharding within a rank:

```cpp
// Additions to WeightManager.h

/**
 * @brief Extended sharding mode for intra-node parallelism
 */
enum class ShardingScope {
    INTER_RANK,    // Shard across MPI ranks (existing behavior)
    INTRA_RANK,    // Shard across devices within a rank (new)
    HIERARCHICAL   // Both: shard across ranks, then across devices per rank
};

struct IntraNodeShardingConfig {
    ShardingScope scope = ShardingScope::INTER_RANK;
    int devices_per_rank = 1;        // Number of local devices to shard across
    DeviceGroup local_device_group;  // Which local devices participate
};

class WeightManager {
public:
    // Existing methods...
    
    /**
     * @brief Get weight tensor sharded for a specific local device
     * 
     * When intra-node sharding is enabled, returns the slice of the weight
     * assigned to the specified local device.
     * 
     * @param name Weight tensor name
     * @param local_device_idx Index within local device group (0 to devices_per_rank-1)
     * @return Sharded weight tensor for this local device
     */
    std::shared_ptr<TensorBase> getWeightForDevice(
        const std::string& name,
        int local_device_idx
    );
    
    /**
     * @brief Configure intra-node weight sharding
     */
    void setIntraNodeSharding(const IntraNodeShardingConfig& config);
    
private:
    IntraNodeShardingConfig intra_node_config_;
    
    // Cache: (weight_name, local_device_idx) → sharded tensor
    std::unordered_map<std::pair<std::string, int>, std::shared_ptr<TensorBase>> 
        intra_node_weight_cache_;
};
```

### 5.2 Multi-Device GraphOrchestrator

```cpp
// Additions to GraphOrchestrator

class GraphOrchestrator : public IInferenceRunner {
public:
    // Existing...
    
    /**
     * @brief Enable intra-node tensor parallelism
     * 
     * @param local_devices Devices to parallelize across within this rank
     * @param router Backend router for collective operations
     */
    void enableIntraNodeTensorParallel(
        const DeviceGroup& local_devices,
        std::shared_ptr<BackendRouter> router
    );
    
private:
    // Intra-node parallelism state
    bool intra_node_tp_enabled_ = false;
    DeviceGroup local_device_group_;
    std::shared_ptr<BackendRouter> backend_router_;
    
    // Per-device state (buffers, KV caches)
    struct PerDeviceState {
        DeviceId device;
        std::shared_ptr<TensorBase> hidden_local;    // Local slice of hidden state
        std::shared_ptr<TensorBase> attention_out;   // Local attention output
        std::unique_ptr<IUnifiedKVCache> kv_cache;   // Sharded KV cache
    };
    std::vector<PerDeviceState> device_states_;
    
    // Collective stages for intra-node sync
    std::unique_ptr<AllReduceStage> intra_node_allreduce_;
    std::unique_ptr<AllGatherStage> intra_node_allgather_;
};
```

---

## 6. Heterogeneous Execution Flow

### 6.1 Example: 2 CUDA GPUs + 1 CPU on Single Node

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RANK 0 (Single Node)                               │
│                                                                              │
│  DeviceGroup: [GPU:0 (CUDA), GPU:1 (CUDA), CPU:0]                           │
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │     GPU:0        │  │     GPU:1        │  │     CPU:0        │          │
│  │  (CUDA, 24GB)    │  │  (CUDA, 24GB)    │  │  (128GB RAM)     │          │
│  ├──────────────────┤  ├──────────────────┤  ├──────────────────┤          │
│  │ Q/K/V heads 0-4  │  │ Q/K/V heads 5-9  │  │ Q/K/V heads 10-13│          │
│  │ FFN d_ff[0:1620] │  │ FFN d_ff[1620:..│  │ FFN d_ff[3240:..]│          │
│  │ Vocab[0:50K]     │  │ Vocab[50K:100K] │  │ Vocab[100K:150K] │          │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘          │
│           │                     │                      │                    │
│           │     NCCL AllReduce  │                      │                    │
│           └──────────┬──────────┘                      │                    │
│                      │                                 │                    │
│                      │         Host AllReduce          │                    │
│                      └─────────────┬───────────────────┘                    │
│                                    │                                        │
│                            Combined Result                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Two-Phase AllReduce for Heterogeneous Groups

```cpp
/**
 * @brief Heterogeneous AllReduce algorithm
 * 
 * For groups with mixed device types (e.g., CUDA + CPU):
 * 1. Phase 1: AllReduce within same-type subgroups (NCCL for CUDAs, RCCL for ROCms)
 * 2. Phase 2: AllReduce across subgroup representatives via Host backend
 * 
 * Example: 2 CUDA GPUs + 1 CPU
 * - Phase 1: NCCL AllReduce on GPU:0 and GPU:1 → result on GPU:0
 * - Phase 2: Host AllReduce between GPU:0 and CPU:0 → final result
 * - Phase 3: Broadcast GPU:0 result to GPU:1 (NCCL)
 */
class HeterogeneousAllReduceStrategy {
public:
    bool execute(
        const DeviceGroup& full_group,
        void* buffer,
        size_t count,
        DataType dtype,
        CollectiveOp op,
        BackendRouter* router
    );
    
private:
    // Partition group by device type
    struct Subgroup {
        DeviceType type;
        std::vector<int> device_indices;  // Indices into full_group.devices
        ICollectiveBackend* backend;
    };
    
    std::vector<Subgroup> partitionByType(const DeviceGroup& group);
};
```

---

## 7. Integration with Existing Components

### 7.1 Qwen2Graph Modifications

```cpp
// In Qwen2Graph::buildFFNGraph()

void Qwen2Graph::buildFFNGraph(...) {
    // ... existing stages ...
    
    // After Down projection, add AllReduce
    if (mpi_ctx_ && mpi_ctx_->world_size() > 1) {
        // Inter-node AllReduce (MPI)
        graph.addNode("ffn_allreduce_mpi", 
            std::make_unique<AllReduceStage>(AllReduceStage::Params{
                .buffer = ffn_output,
                .backend = CollectiveBackendType::MPI,
                .mpi_ctx = mpi_ctx_
            }));
    }
    
    if (config_.intra_node_tp_enabled) {
        // Intra-node AllReduce (NCCL/RCCL/Host based on devices)
        graph.addNode("ffn_allreduce_local",
            std::make_unique<AllReduceStage>(AllReduceStage::Params{
                .buffer = ffn_output,
                .backend = CollectiveBackendType::AUTO,
                .router = backend_router_,
                .group = local_device_group_
            }));
    }
}
```

### 7.2 PlacementStrategy Integration

The `PlacementStrategy` now has additional responsibility:

```cpp
struct PlacementInput {
    // Existing fields...
    
    // Intra-node tensor parallelism config
    struct IntraNodeTPConfig {
        bool enabled = false;
        int target_devices_per_rank = 1;  // How many local devices to use
        bool prefer_homogeneous = true;   // Prefer same-type devices for perf
    } intra_node_tp;
};

struct LayerPlacement {
    // Existing fields...
    
    // For intra-node TP: which local devices share this layer's computation
    DeviceGroup local_devices;
    
    // Collective config for this layer
    CollectiveBackendType attention_allreduce_backend = CollectiveBackendType::AUTO;
    CollectiveBackendType ffn_allreduce_backend = CollectiveBackendType::AUTO;
};
```

---

## 8. File Structure

```
src/v2/
├── collective/
│   ├── ICollectiveBackend.h          # Abstract backend interface
│   ├── DeviceGroup.h                 # Device group concept
│   ├── BackendRouter.h/.cpp          # Backend selection logic
│   ├── HeterogeneousStrategy.h/.cpp  # Multi-phase heterogeneous collectives
│   └── backends/
│       ├── MPIBackend.h/.cpp         # MPI implementation
│       ├── NCCLBackend.h/.cpp        # NCCL implementation (CUDA)
│       ├── RCCLBackend.h/.cpp        # RCCL implementation (ROCm)
│       └── HostBackend.h/.cpp        # Host-staged fallback
│
├── execution/
│   ├── compute_stages/
│   │   └── stages/
│   │       ├── AllReduceStage.h/.cpp    # Unified (replaces AllreduceStage)
│   │       ├── AllGatherStage.h/.cpp    # Unified (replaces old)
│   │       └── CollectiveStageBase.h    # Base class
│   ├── DeviceInventory.h             # (existing) Device discovery
│   ├── PlacementStrategy.h/.cpp      # (extended) Intra-node TP config
│   └── GraphOrchestrator.h/.cpp      # (extended) Multi-device execution
│
└── loaders/
    └── WeightManager.h/.cpp          # (extended) Intra-node sharding
```

---

## 9. Build System Changes

### 9.1 CMake Configuration

```cmake
# cmake/FindNCCL.cmake
find_path(NCCL_INCLUDE_DIR nccl.h HINTS ${CUDA_TOOLKIT_ROOT_DIR}/include)
find_library(NCCL_LIBRARY nccl HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
    set(HAVE_NCCL TRUE)
    add_definitions(-DHAVE_NCCL)
endif()

# cmake/FindRCCL.cmake  
find_path(RCCL_INCLUDE_DIR rccl/rccl.h HINTS ${ROCM_PATH}/include)
find_library(RCCL_LIBRARY rccl HINTS ${ROCM_PATH}/lib)
if(RCCL_INCLUDE_DIR AND RCCL_LIBRARY)
    set(HAVE_RCCL TRUE)
    add_definitions(-DHAVE_RCCL)
endif()
```

### 9.2 Conditional Compilation

```cpp
// In NCCLBackend.cpp
#ifdef HAVE_NCCL

bool NCCLBackend::allreduce(...) {
    ncclResult_t result = ncclAllReduce(
        buffer, buffer, count,
        toNCCLDataType(dtype),
        toNCCLOp(op),
        comm_, stream_
    );
    return result == ncclSuccess;
}

#else

bool NCCLBackend::allreduce(...) {
    LOG_ERROR("NCCL not available - rebuild with HAVE_NCCL=ON");
    return false;
}

#endif
```

---

## 10. Migration Path

### Phase 1: Rename Existing Stages (Non-Breaking)
1. Rename `AllreduceStage` → `AllReduceStage` (fix casing)
2. Add `backend` parameter defaulting to `MPI`
3. Update all call sites to explicit `CollectiveBackendType::MPI`

### Phase 2: Add Backend Infrastructure
1. Implement `ICollectiveBackend` interface
2. Implement `MPIBackend` (extract from existing stage)
3. Implement `HostBackend` for heterogeneous fallback
4. Add `BackendRouter`

### Phase 3: Add GPU Collective Backends
1. Implement `NCCLBackend` (requires NCCL library)
2. Implement `RCCLBackend` (requires RCCL library)
3. Add heterogeneous multi-phase strategy

### Phase 4: Enable Intra-Node Tensor Parallelism
1. Extend `WeightManager` for intra-node sharding
2. Extend `GraphOrchestrator` for multi-device execution
3. Update `Qwen2Graph` to generate intra-node collective stages
4. Add tests for heterogeneous configurations

---

## 11. Testing Strategy

### Unit Tests
- Backend interface compliance for each backend
- DeviceGroup creation and partitioning
- BackendRouter selection logic

### Integration Tests  
- NCCL AllReduce correctness (2+ CUDA GPUs)
- RCCL AllReduce correctness (2+ ROCm GPUs)
- Host AllReduce correctness (CPU + GPU)
- Heterogeneous AllReduce (CUDA + ROCm + CPU)

### Parity Tests
- Compare NCCL vs MPI results for same input
- Compare heterogeneous vs homogeneous results

### Performance Tests
- NCCL vs Host backend latency
- Scaling efficiency with device count
- Heterogeneous overhead measurement

---

## 12. Example Usage

```cpp
// Configure intra-node tensor parallelism with 4 GPUs
InferenceRunnerConfig config;
config.intra_node_tp.enabled = true;
config.intra_node_tp.target_devices_per_rank = 4;

auto runner = createInferenceRunner(model_ctx, mpi_ctx, config);

// The runner will:
// 1. Detect 4 CUDA GPUs on this node
// 2. Create DeviceGroup with all 4
// 3. Shard weights across 4 GPUs (d_ff/4 per GPU)
// 4. Use NCCL for intra-node AllReduce
// 5. Use MPI for inter-node AllReduce (if multi-node)

runner->forward(tokens.data(), seq_len);
```

---

## 13. Open Questions

1. **Pinned Memory Management**: Should we use CUDA pinned memory for Host backend transfers?
2. **Stream Management**: Separate streams per collective operation or shared?
3. **Async Collectives**: Support for overlapping compute with communication?
4. **Gradient Checkpointing**: Relevant for training, but inference-only for now
5. **Topology-Aware Placement**: Should we detect NVLink/PCIe topology for optimal device pairing?
