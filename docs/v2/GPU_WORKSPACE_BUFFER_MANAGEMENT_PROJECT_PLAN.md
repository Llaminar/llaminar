# GPU Workspace Buffer Management Project Plan

**Author:** David Sanftenberg  
**Created:** January 14, 2026  
**Status:** Planning  
**Priority:** High  

## Executive Summary

This project introduces centralized GPU workspace buffer management into Llaminar's graph execution framework. Currently, GPU kernels (CUDA and ROCm) each manage their own ad-hoc workspace buffers via PIMPL structs, leading to:

- Fragmented VRAM allocation with no global memory budget
- Duplicated buffer management code across kernel types
- No support for streaming/chunked operations that trade throughput for memory
- Hot-path allocations in performance-critical code paths (e.g., FP16 GEMM)

The solution is to extend `GraphBufferManager` to handle GPU workspace buffers as a first-class concern, enabling memory-bounded inference and slab-based GPU operations.

---

## Table of Contents

1. [Motivation](#motivation)
2. [Phase 0: Generic Buffer Declaration System](#phase-0-generic-buffer-declaration-system)
3. [Phase 0.5: CPUBackend Infrastructure](#phase-05-cpubackend-infrastructure) ◄── **PREREQUISITE**
4. [Architecture Overview](#architecture-overview)
   - [Integration with Llaminar Architecture Layers](#integration-with-llaminar-architecture-layers)
   - [Two Distinct Memory Budget Decisions](#two-distinct-memory-budget-decisions)
   - [Timeline: Memory Decisions](#timeline-memory-decisions)
   - [Component Interaction Flow](#component-interaction-flow)
5. [Phase 1: GPU Workspace Manager](#phase-1-gpu-workspace-manager)
   - [Budget Computation via IBackend](#14-budget-computation-via-ibackend)
6. [Phase 2: Kernel Integration](#phase-2-kernel-integration)
7. [Phase 3: Slab-Based FP16 GEMM](#phase-3-slab-based-fp16-gemm)
8. [Phase 4: Memory Budget Enforcement](#phase-4-memory-budget-enforcement)
9. [Testing Strategy](#testing-strategy)
10. [Migration Guide](#migration-guide)
11. [Risk Assessment](#risk-assessment)
12. [Timeline](#timeline)

---

## Motivation

### Current State Problems

**1. Ad-hoc Workspace Management**

Each kernel manages its own buffers in PIMPL structs:

```cpp
// ROCmQuantisedGemmKernel::Impl (current)
struct Impl {
    int8_t *d_A_int8 = nullptr;    // Work buffer
    float *d_scales_A = nullptr;   // Work buffer  
    int32_t *d_C_int32 = nullptr;  // Work buffer
    float *d_A_fp32 = nullptr;     // Cached temp
    float *d_C_fp32 = nullptr;     // Cached temp
    void *d_A_fp16 = nullptr;      // FP16 workspace
    void *d_B_fp16 = nullptr;      // FP16 workspace
    void *d_C_fp16 = nullptr;      // FP16 workspace
    // ... capacity tracking for each
};
```

This pattern is duplicated in `CUDAQuantisedGemmKernel::Impl` with slight variations.

**2. No Memory Budget**

VRAM allocation is unbounded. For a 7B model FFN layer:
- INT8 weights: K×N = 4096×14336 = 56MB
- FP16 conversion: 2× = 112MB additional
- No way to limit or trade throughput for memory

**3. Hot-Path Allocations**

The current FP16 path allocates 3 buffers per GEMM call:
```cpp
// Current FP16 implementation (BAD)
hipMalloc(&d_A_fp16, M * K * 2);  // Per-call allocation!
hipMalloc(&d_B_fp16, K * N * 2);  // Per-call allocation!
hipMalloc(&d_C_fp16, M * N * 2);  // Per-call allocation!
```

**4. MI50 FP16 Performance Opportunity**

MI50/MI60 (gfx906) lack INT8 Tensor Cores but have efficient FP16 MFMA. Benchmarks show FP16 path is 20-35% faster for M>128, but the current implementation wastes VRAM on full-matrix FP16 conversion.

**5. Missing GPU Workspace in Stage Buffer System**

The existing `StageBufferRequirements` with `addScratch()` only manages **CPU-side** buffers via `LivenessAnalyzer`. There is no equivalent for GPU workspace:

```cpp
// Current: Only CPU-side scratch buffers
StageBufferRequirements GEMMStage::getBufferRequirements() const {
    StageBufferRequirements reqs;
    reqs.addInput("input", {M, K}, BufferTensorType::FP32);  // CPU
    reqs.addOutput("output", {M, N}, BufferTensorType::FP32); // CPU
    // Where do we declare d_A_int8, d_C_int32, d_slab_fp16? NOWHERE!
    return reqs;
}
```

### Target State

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     GraphBufferManager (Extended)                        │
├─────────────────────────────────────────────────────────────────────────┤
│  CPU Activation Buffers          │  GPU Workspace Buffers               │
│  (existing LivenessAnalyzer)     │  (NEW: DeviceWorkspaceManager)          │
│                                  │                                      │
│  ┌─────────────────────────┐     │  ┌─────────────────────────────┐    │
│  │ FP32 Q, K, V tensors    │     │  │ Device 0: CUDA workspace    │    │
│  │ FP32 FFN intermediates  │     │  │   - gemm_slab_a: 64MB       │    │
│  │ FP32 residual buffers   │     │  │   - gemm_slab_b: 64MB       │    │
│  └─────────────────────────┘     │  │   - gemm_slab_c: 32MB       │    │
│                                  │  │   - quant_buffer: 16MB       │    │
│                                  │  │   - attn_scores: var        │    │
│                                  │  └─────────────────────────────┘    │
│                                  │  ┌─────────────────────────────┐    │
│                                  │  │ Device 1: ROCm workspace    │    │
│                                  │  │   - fp16_slab_a: 64MB       │    │
│                                  │  │   - fp16_slab_b: 64MB       │    │
│                                  │  │   - fp16_slab_c: 32MB       │    │
│                                  │  │   - int8_buffer: 16MB       │    │
│                                  │  │   - attn_scores: var        │    │
│                                  │  └─────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 0: Generic Buffer Declaration System

**Goal:** Establish a semantics-free buffer declaration system where stages declare their needs and the graph just allocates.

### Design Philosophy

**Core Principles:**
1. **Stage-Centric Declaration**: Stages declare what buffers they need. The graph doesn't interpret buffer semantics.
2. **No Hardcoded Semantics**: No enums like `GEMM_QUANTIZE` or `ATTENTION_SCORES` in the graph/buffer system.
3. **Type + Size Only**: The graph only needs data type, size, and alignment - not purpose.
4. **Dynamic Sizing**: Buffer sizes computed by stages based on workload parameters.
5. **Declarative Graph**: Buffers are just another declaration stages make.

### What the Graph Knows vs. What Stages Know

| Concern | Graph/Buffer Manager | Stage |
|---------|---------------------|-------|
| Buffer name | Opaque string key | Meaningful internal name |
| Buffer purpose | Don't care | Knows exactly what it's for |
| Data type | `DataType::FP32`, etc. | Knows why that type is needed |
| Size | Bytes (computed) | Knows formula: `M * K * sizeof(int8_t)` |
| Aliasability | Hint from stage | Knows if buffer can be shared |
| Lifetime | Tracks via liveness | Knows when it's needed |

### 0.1 Buffer Type Analysis (Informational Only)

Llaminar has multiple distinct buffer categories:

| Buffer Type | Memory Location | Allocation Timing | Lifetime | Current System |
|-------------|-----------------|-------------------|----------|----------------|
| **Activation Buffers** | CPU (host) | Graph construction | Graph lifetime | `GraphBufferManager` + `LivenessAnalyzer` |
| **KV Cache** | CPU (host) | Model init | Session lifetime | `IKVCache` |
| **Weights** | CPU + GPU | Model load | Model lifetime | `WeightManager` + `ensureOnDevice()` |
| **CPU Scratch** | CPU (host) | Graph construction | Stage lifetime | `BufferRole::SCRATCH` + aliasing |
| **GPU Activation Staging** | GPU (device) | Per-execute | Stage lifetime | `ensureOnDevice()` - automatic |
| **GPU Workspace** | GPU (device) | **Graph construction (NEW)** | **Stage lifetime (NEW)** | **`DeviceWorkspaceManager` (NEW)** |

### 0.2 Stage GPU Workspace Requirements

Stages that need GPU workspace will override `getWorkspaceRequirements()`:

| Stage | Example Buffers | Sizing Formula (stage-computed) |
|-------|-----------------|--------------------------------|
| `GEMMStage` | quant_a, scales_a, acc | `M*K`, `M`, `M*N` |
| `FusedQKVGEMMStage` | quant_qkv, scales_qkv, acc | `3*M*K`, `3*M`, `M*(3*N)` |
| `AttentionComputeStage` | scores, context | `n_heads*S*S`, `S*head_dim*n_heads` |
| `AllreduceStage` | staging | `tensor_elements` |
| `RMSNormStage` | (none - elementwise) | N/A |
| `SwiGLUStage` | (none - elementwise) | N/A |

### 0.3 Extended Stage Buffer Declaration Model

```cpp
// BEFORE: Only CPU-side buffers
class IComputeStage {
    virtual StageBufferRequirements getBufferRequirements() const;
};

// AFTER: Add GPU workspace declarations (generic, semantics-free)
class IComputeStage {
    // Existing - declares CPU-side buffers
    virtual StageBufferRequirements getBufferRequirements() const;
    
    // NEW - declares GPU workspace (default: none)
    // Returns generic buffer descriptors with name/size/type only
    virtual WorkspaceRequirements getWorkspaceRequirements() const {
        return {};
    }
    
    // NEW - receives allocated workspace buffers
    virtual void setGpuWorkspaceBuffers(
        const std::unordered_map<std::string, void*>& buffers) {}
};
```

### 0.4 Graph Builder Integration (Buffer-Agnostic)

```cpp
// Graph builder collects requirements from stages, doesn't interpret them
void GraphBuilder::build() {
    for (auto& stage : stages_) {
        // GPU workspace - graph is agnostic to buffer purposes
        auto gpu_reqs = stage->getWorkspaceRequirements();
        if (!gpu_reqs.empty()) {
            workspace_manager_.registerBuffers(stage->name(), gpu_reqs);
        }
    }
    
    // Allocate all buffers
    workspace_manager_.allocate();
    
    // Distribute buffers back to stages
    for (auto& stage : stages_) {
        auto buffers = workspace_manager_.getBuffersFor(stage->name());
        if (!buffers.empty()) {
            stage->setGpuWorkspaceBuffers(buffers);
        }
    }
}
```

### 0.5 Phase 0 Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/v2/execution/WorkspaceDescriptor.h` | CREATE | Generic descriptor (name, size, dtype, alignment - NO semantic enums) |
| `src/v2/execution/compute_stages/IComputeStage.h` | MODIFY | Add `getWorkspaceRequirements()` + `setGpuWorkspaceBuffers()` |

### 0.6 Phase 0 Acceptance Criteria

- [ ] `WorkspaceDescriptor` has NO semantic/category fields (no `GEMM_QUANTIZE` enums)
- [ ] Stages can declare arbitrary buffers with name/size/type
- [ ] Graph builder is completely buffer-agnostic
- [ ] Buffer sizing formulas live in stages, not graph
- [ ] Existing tests pass (no behavior change)

---

## Phase 0.5: CPUBackend Infrastructure

**Goal:** Create `CPUBackend` implementing `IBackend` to unify memory query and allocation across all device types.

**Rationale:** Without `CPUBackend`, the `GraphBufferManager` would need device-type switches to query memory:

```cpp
// WITHOUT CPUBackend (asymmetric, error-prone)
size_t GraphBufferManager::computeWorkspaceBudget(DeviceId device, float fraction) {
    if (device.is_gpu()) {
        IBackend* backend = BackendManager::getBackendFor(device);
        return backend->deviceMemoryFree(device.index()) * fraction;
    } else {
        // Special case for CPU - different API!
        return sysconf(_SC_AVPHYS_PAGES) * sysconf(_SC_PAGE_SIZE) * fraction;
    }
}

// WITH CPUBackend (symmetric, clean)
size_t GraphBufferManager::computeWorkspaceBudget(DeviceId device, float fraction) {
    IBackend* backend = BackendManager::getBackendFor(device);  // Works for ALL devices
    return backend->deviceMemoryFree(device.index()) * fraction;
}
```

### 0.5.1 CPUBackend Implementation (Rank-Local Design)

```cpp
// src/v2/backends/cpu/CPUBackend.h

namespace llaminar2 {

/**
 * @brief CPU backend implementing IBackend interface (RANK-LOCAL)
 *
 * Provides unified memory query and allocation for THIS rank's CPU/NUMA node.
 * Each MPI rank creates its own CPUBackend with its local NUMA node.
 *
 * **Design: Rank-Local View**
 * - deviceCount() always returns 1 (this rank's CPU only)
 * - Memory queries return this NUMA node's memory, not system total
 * - Allocations happen on this rank's NUMA node for locality
 * - Consistent with GPU backends which filter by NUMA affinity
 *
 * **Initialization:**
 * ```cpp
 * // In BackendManager or GraphOrchestrator:
 * int numa_node = mpi_topology.placement().numa_node;
 * auto cpu_backend = std::make_unique<CPUBackend>(numa_node);
 * ```
 */
class CPUBackend : public IBackend {
public:
    /**
     * @brief Construct CPUBackend for a specific NUMA node
     * @param local_numa_node The NUMA node this rank owns (from MPITopology)
     */
    explicit CPUBackend(int local_numa_node);
    ~CPUBackend() override = default;

    // =========================================================================
    // Memory Transfer (trivial for CPU - just memcpy)
    // =========================================================================
    
    bool deviceToHost(void* dst, const void* src, size_t bytes, int device_id) override {
        std::memcpy(dst, src, bytes);
        return true;
    }
    
    bool hostToDevice(void* dst, const void* src, size_t bytes, int device_id) override {
        std::memcpy(dst, src, bytes);
        return true;
    }
    
    // =========================================================================
    // Synchronization (no-op for CPU - always synchronous)
    // =========================================================================
    
    bool synchronize(int device_id) override { return true; }
    bool setDevice(int device_id) override { return true; }
    
    // =========================================================================
    // Memory Allocation (NUMA-aware when available)
    // =========================================================================
    
    void* allocate(size_t bytes, int device_id) override;
    void free(void* ptr, int device_id) override;
    bool memset(void* ptr, int value, size_t bytes, int device_id) override;
    
    // =========================================================================
    // Device Query (RANK-LOCAL: always 1 device)
    // =========================================================================
    
    int deviceCount() const override { return 1; }  // This rank's CPU only
    std::string backendName() const override { return "CPU"; }
    std::string deviceName(int device_id) const override;
    
    // =========================================================================
    // Memory Query (THE KEY BENEFIT - queries THIS rank's NUMA node)
    // =========================================================================
    
    /**
     * @brief Get total memory for THIS rank's NUMA node
     * @param device_id Ignored (always uses local_numa_node_)
     * @return Total memory in bytes for this NUMA node
     */
    size_t deviceMemoryTotal(int device_id) const override;
    
    /**
     * @brief Get available memory for THIS rank's NUMA node
     * @param device_id Ignored (always uses local_numa_node_)
     * @return Available memory in bytes for this NUMA node
     *
     * **Note:** CPU memory is shared with other processes. Use conservative
     * fractions (0.3-0.5) for workspace budgets, not 0.8 like GPUs.
     */
    size_t deviceMemoryFree(int device_id) const override;
    
    // =========================================================================
    // Capability Queries (CPU always has these via software)
    // =========================================================================
    
    bool supportsBF16(int device_id) const override { return true; }  // Software emulation
    bool supportsFP16(int device_id) const override { return true; }  // Software emulation  
    bool supportsINT8(int device_id) const override { return true; }  // Native
    
    // =========================================================================
    // GEMM (delegate to CPU kernels - not used via IBackend typically)
    // =========================================================================
    
    bool gemmIQ4NL(const void* A, const void* B, void* C,
                   int m, int n, int k, int device_id) override;

private:
    int local_numa_node_;    ///< This rank's NUMA node (from MPITopology)
    bool numa_available_;    ///< Whether libnuma is available
};

} // namespace llaminar2
```

### 0.5.2 CPUBackend.cpp Implementation (Rank-Local)

```cpp
// src/v2/backends/cpu/CPUBackend.cpp

#include "CPUBackend.h"
#include "../../utils/Logger.h"
#include <cstring>
#include <unistd.h>  // sysconf

#ifdef HAVE_NUMA
#include <numa.h>
#endif

namespace llaminar2 {

CPUBackend::CPUBackend(int local_numa_node)
    : local_numa_node_(local_numa_node)
{
#ifdef HAVE_NUMA
    numa_available_ = (numa_available() >= 0);
#else
    numa_available_ = false;
#endif
    LOG_DEBUG("[CPUBackend] Initialized for NUMA node " << local_numa_node_
              << " (libnuma=" << (numa_available_ ? "yes" : "no") << ")");
}

std::string CPUBackend::deviceName(int device_id) const {
    if (numa_available_) {
        return "CPU:NUMA" + std::to_string(local_numa_node_);
    }
    return "CPU";
}

void* CPUBackend::allocate(size_t bytes, int device_id) {
#ifdef HAVE_NUMA
    if (numa_available_) {
        // Always allocate on THIS rank's NUMA node
        void* ptr = numa_alloc_onnode(bytes, local_numa_node_);
        if (ptr) {
            LOG_DEBUG("[CPUBackend] Allocated " << bytes << " bytes on NUMA node " << local_numa_node_);
            return ptr;
        }
    }
#endif
    // Fallback: aligned allocation
    void* ptr = std::aligned_alloc(64, (bytes + 63) & ~63ULL);
    LOG_DEBUG("[CPUBackend] Allocated " << bytes << " bytes (aligned)");
    return ptr;
}

void CPUBackend::free(void* ptr, int device_id) {
    if (!ptr) return;
#ifdef HAVE_NUMA
    if (numa_available_) {
        // Note: numa_free needs size, but we don't track it
        // In practice, use std::free which works for both
        std::free(ptr);
        return;
    }
#endif
    std::free(ptr);
}

bool CPUBackend::memset(void* ptr, int value, size_t bytes, int device_id) {
    std::memset(ptr, value, bytes);
    return true;
}

size_t CPUBackend::deviceMemoryTotal(int device_id) const {
#ifdef HAVE_NUMA
    if (numa_available_) {
        // Return THIS rank's NUMA node memory, not system total
        long long size = numa_node_size64(local_numa_node_, nullptr);
        return size > 0 ? static_cast<size_t>(size) : 0;
    }
#endif
    // Fallback: system total (single-socket assumption)
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return static_cast<size_t>(pages) * page_size;
}

size_t CPUBackend::deviceMemoryFree(int device_id) const {
#ifdef HAVE_NUMA
    if (numa_available_) {
        // Return THIS rank's NUMA node available memory
        long long free_bytes;
        numa_node_size64(local_numa_node_, &free_bytes);
        return free_bytes > 0 ? static_cast<size_t>(free_bytes) : 0;
    }
#endif
    // Fallback: system available
    long available = sysconf(_SC_AVPHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return static_cast<size_t>(available) * page_size;
}

bool CPUBackend::gemmIQ4NL(const void* A, const void* B, void* C,
                           int m, int n, int k, int device_id) {
    // IQ4_NL GEMM not typically called via IBackend for CPU
    // Use KernelFactory::createGemm() instead
    LOG_WARN("[CPUBackend] gemmIQ4NL called via IBackend - use KernelFactory instead");
    return false;
}

} // namespace llaminar2
```

### 0.5.3 BackendManager Extension

```cpp
// Additions to BackendManager.h

class BackendManager {
public:
    /**
     * @brief Initialize CPU backend for this rank
     * @param local_numa_node NUMA node from MPITopology::placement().numa_node
     *
     * Must be called early in rank initialization:
     * ```cpp
     * MPITopology topo(MPI_COMM_WORLD);
     * BackendManager::initCPUBackend(topo.placement().numa_node);
     * ```
     */
    static void initCPUBackend(int local_numa_node);
    
    /**
     * @brief Get CPU backend for this rank
     * @return CPUBackend* (always available after initCPUBackend)
     */
    static IBackend* getCPUBackend();

    /**
     * @brief Get backend for any device (CPU, CUDA, or ROCm)
     * @param device DeviceId to get backend for
     * @return IBackend* for the device type
     *
     * **Unified API:**
     * ```cpp
     * IBackend* backend = BackendManager::getBackendFor(device);
     * size_t free = backend->deviceMemoryFree(device.index());
     * // Works for CPU:0, CUDA:0, ROCm:0 identically!
     * ```
     */
    static IBackend* getBackendFor(DeviceId device);
    
private:
    static std::unique_ptr<CPUBackend> cpu_backend_;
};

// In BackendManager.cpp
void BackendManager::initCPUBackend(int local_numa_node) {
    cpu_backend_ = std::make_unique<CPUBackend>(local_numa_node);
}

IBackend* BackendManager::getBackendFor(DeviceId device) {
    switch (device.type()) {
        case DeviceType::CPU:  return getCPUBackend();
        case DeviceType::CUDA: return getCUDABackend();
        case DeviceType::ROCm: return getROCmBackend();
        default: return nullptr;
    }
}
```

### 0.5.4 MPI Topology, NUMA Nodes, and CPUBackend Device Model

**Critical Design Question**: How does `CPUBackend::deviceCount()` relate to MPI ranks?

Llaminar's MPI model creates **one rank per socket** by default:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LLAMINAR MPI TOPOLOGY MODEL                               │
│                                                                              │
│   SINGLE MACHINE (2 sockets):                                                │
│   ┌───────────────────────────────┬───────────────────────────────┐          │
│   │       SOCKET 0 (NUMA 0)       │       SOCKET 1 (NUMA 1)       │          │
│   │         MPI Rank 0            │         MPI Rank 1            │          │
│   │  ┌─────────────────────────┐  │  ┌─────────────────────────┐  │          │
│   │  │   28 CPU cores         │  │  │   28 CPU cores         │  │          │
│   │  │   64 GB DDR4           │  │  │   64 GB DDR4           │  │          │
│   │  │   OMP_NUM_THREADS=28   │  │  │   OMP_NUM_THREADS=28   │  │          │
│   │  └─────────────────────────┘  │  └─────────────────────────┘  │          │
│   │  ┌─────────────────────────┐  │  ┌───────────┐ ┌───────────┐  │          │
│   │  │   CUDA:0 (RTX 3090)    │  │  │ ROCm:0    │ │ ROCm:1    │  │          │
│   │  └─────────────────────────┘  │  └───────────┘ └───────────┘  │          │
│   └───────────────────────────────┴───────────────────────────────┘          │
│                                                                              │
│   Key relationships:                                                         │
│   - MPITopology detects: ranks_per_node=2, node_count=1                     │
│   - Each rank has placement.numa_node = local_rank (0 or 1)                 │
│   - Each rank sees ONLY devices affine to its socket                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**The Design Decision**: Should `CPUBackend` enumerate:
1. **All NUMA nodes system-wide** (`deviceCount() = numa_num_configured_nodes()`)  
2. **Only THIS rank's NUMA node** (`deviceCount() = 1, device_id = my_numa_node`)

**Answer: Option 2 - Per-Rank Local View**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CPUBackend: RANK-LOCAL VIEW                               │
│                                                                              │
│   Each MPI rank already "owns" one NUMA node (via mpirun --bind-to socket). │
│   CPUBackend should reflect this LOCAL view, not the global system view.    │
│                                                                              │
│   RANK 0 sees:                         RANK 1 sees:                         │
│   ┌────────────────────────┐           ┌────────────────────────┐           │
│   │ CPUBackend             │           │ CPUBackend             │           │
│   │   deviceCount() = 1    │           │   deviceCount() = 1    │           │
│   │   deviceName(0)="CPU"  │           │   deviceName(0)="CPU"  │           │
│   │   deviceMemoryTotal(0) │           │   deviceMemoryTotal(0) │           │
│   │     → NUMA 0's 64GB    │           │     → NUMA 1's 64GB    │           │
│   │   allocate() →         │           │   allocate() →         │           │
│   │     numa_alloc_onnode( │           │     numa_alloc_onnode( │           │
│   │       bytes, 0)        │           │       bytes, 1)        │           │
│   └────────────────────────┘           └────────────────────────┘           │
│                                                                              │
│   CUDABackend (rank 0):                ROCmBackend (rank 1):                │
│   ┌────────────────────────┐           ┌────────────────────────┐           │
│   │   deviceCount() = 1    │           │   deviceCount() = 2    │           │
│   │   (only CUDA:0)        │           │   (ROCm:0, ROCm:1)     │           │
│   └────────────────────────┘           └────────────────────────┘           │
│                                                                              │
│   WHY: Backends already filter by NUMA affinity (via CUDA/HIP_VISIBLE_DEVS) │
│   CPUBackend should follow the same pattern for consistency.                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Implementation Consequence**: CPUBackend needs to know its NUMA node:

```cpp
class CPUBackend : public IBackend {
public:
    // Constructor receives the local NUMA node from MPITopology
    explicit CPUBackend(int local_numa_node);
    
    int deviceCount() const override { return 1; }  // This rank's CPU only
    
    size_t deviceMemoryTotal(int device_id) const override {
        // Return this NUMA node's memory, not system total
        return numa_node_size64(local_numa_node_, nullptr);
    }
    
    void* allocate(size_t bytes, int device_id) override {
        // Allocate on THIS rank's NUMA node
        return numa_alloc_onnode(bytes, local_numa_node_);
    }
    
private:
    int local_numa_node_;  // From MPITopology::placement().numa_node
};
```

**Multi-Machine Scenario**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-NODE CLUSTER (6 machines)                           │
│                                                                              │
│   MPITopology detects:                                                       │
│   - world_size = 12 ranks                                                   │
│   - node_count = 6 machines                                                 │
│   - ranks_per_node = 2                                                      │
│                                                                              │
│   Machine 0:          Machine 1:          ...   Machine 5:                  │
│   ┌───────┬───────┐   ┌───────┬───────┐         ┌───────┬───────┐          │
│   │Rank 0 │Rank 1 │   │Rank 2 │Rank 3 │   ...   │Rank 10│Rank 11│          │
│   │NUMA 0 │NUMA 1 │   │NUMA 0 │NUMA 1 │         │NUMA 0 │NUMA 1 │          │
│   └───────┴───────┘   └───────┴───────┘         └───────┴───────┘          │
│                                                                              │
│   Each rank's CPUBackend:                                                   │
│   - deviceCount() = 1                                                       │
│   - Reports memory for its LOCAL NUMA node only                             │
│   - Allocates on its LOCAL NUMA node only                                   │
│                                                                              │
│   PlacementStrategy sees each rank as having:                               │
│   - 1 CPU (its local NUMA node)                                             │
│   - N GPUs (affine to its socket)                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 0.5.5 DeviceId Naming Convention

With the rank-local view, DeviceIds are simpler:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     UNIFIED DEVICE MODEL (Per-Rank)                          │
│                                                                              │
│   DeviceId examples (from rank 0's perspective):                            │
│     CPU:0     → This rank's NUMA node     CPUBackend::deviceMemoryFree(0)   │
│     CUDA:0    → First CUDA GPU visible    CUDABackend::deviceMemoryFree(0)  │
│     ROCm:0    → First ROCm GPU visible    ROCmBackend::deviceMemoryFree(0)  │
│                                                                              │
│   GraphBufferManager (running on each rank):                                │
│     for (DeviceId device : stage->getUsedDevices()) {                       │
│         IBackend* backend = BackendManager::getBackendFor(device);          │
│         size_t budget = backend->deviceMemoryFree(device.index()) * 0.5f;   │
│     }                                                                        │
│                                                                              │
│   PlacementStrategy reasons about:                                          │
│     "On THIS rank, put layers 0-11 on CUDA:0, layers 12-23 on CPU:0"       │
│     (Different ranks may have different placements)                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 0.5.6 Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/v2/backends/cpu/CPUBackend.h` | CREATE | IBackend implementation for CPU |
| `src/v2/backends/cpu/CPUBackend.cpp` | CREATE | Implementation with optional NUMA |
| `src/v2/backends/BackendManager.h` | MODIFY | Add `getCPUBackend()`, `getBackendFor(DeviceId)` |
| `src/v2/backends/BackendManager.cpp` | MODIFY | Register CPUBackend |
| `src/v2/CMakeLists.txt` | MODIFY | Add cpu/ subdirectory, optional libnuma linking |
| `tests/v2/unit/Test__CPUBackend.cpp` | CREATE | Unit tests |

### 0.5.7 CMake Integration

```cmake
# In src/v2/CMakeLists.txt

# Optional NUMA support
find_package(NUMA QUIET)
if(NUMA_FOUND)
    add_compile_definitions(HAVE_NUMA)
    target_link_libraries(llaminar2_core PRIVATE numa)
    message(STATUS "NUMA support: ENABLED")
else()
    message(STATUS "NUMA support: DISABLED (libnuma not found)")
endif()

# Add CPU backend sources
add_subdirectory(backends/cpu)
```

### 0.5.8 Phase 0.5 Acceptance Criteria

- [ ] `CPUBackend` implements all `IBackend` methods
- [ ] `BackendManager::getCPUBackend()` returns valid backend
- [ ] `BackendManager::getBackendFor(DeviceId)` works for CPU, CUDA, and ROCm
- [ ] `deviceMemoryTotal()` returns system memory (or per-NUMA node)
- [ ] `deviceMemoryFree()` returns available memory
- [ ] `allocate()` / `free()` work correctly
- [ ] NUMA allocation works when libnuma is available
- [ ] Graceful fallback when libnuma is not available
- [ ] Unit tests pass
- [ ] No memory leaks (ASAN clean)

---

## Architecture Overview

### Integration with Llaminar Architecture Layers

GPU workspace management integrates at the **GRAPH LAYER**, not the ORCHESTRATION LAYER. This is a critical distinction:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            ORCHESTRATION LAYER                               │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  PlacementStrategy.compute()                                          │   │
│  │    • Uses: DeviceInventory.gpu_memory (TOTAL static capacity)        │   │
│  │    • Decides: "Which layers fit on which GPU?" (WEIGHT placement)    │   │
│  │    • Output: PlacementPlan                                           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ⚠️  Weight placement uses TOTAL memory. Workspace uses FREE memory.        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GRAPH LAYER ◄── GPU WORKSPACE LIVES HERE       │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  GraphOrchestrator.build()                                            │   │
│  │    • Creates stages with device assignments from PlacementPlan       │   │
│  │    • Each stage declares: gpuWorkspaceRequirements()                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                          │                                                   │
│                          ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  GraphBufferManager (EXTENDED)                                        │   │
│  │                                                                       │   │
│  │    EXISTING:                          NEW (GPU Workspace):            │   │
│  │    ├── allocateTensors()              ├── allocateDeviceWorkspace()      │   │
│  │    │   (KV cache, activations)        │   (scratch buffers)           │   │
│  │    │                                  │                               │   │
│  │    │                                  └── Uses:                       │   │
│  │    │                                       • IBackend::deviceMemoryFree() │
│  │    │                                       • DeviceWorkspaceManager (per device) │
│  │    │                                       • Stage requirements         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                          │                                                   │
│                          ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  DeviceWorkspaceManager (per device) ◄── NEW                             │   │
│  │    • device_id, budget_bytes                                         │   │
│  │    • allocate(WorkspaceRequirements)                              │   │
│  │    • getBuffer(name) → void*                                         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            EXECUTION LAYER                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  GraphExecutor                                                        │   │
│  │    • Runs stages in topological order                                │   │
│  │    • Stages already have workspace pointers via setGpuWorkspaceBuffers() │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             KERNEL LAYER                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  ROCmQuantisedGemmKernel / CUDAQuantisedGemmKernel                    │   │
│  │    • Receives workspace pointers - NO internal allocation            │   │
│  │    • Uses: workspace->getBuffer("quant_a") etc.                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Two Distinct Memory Budget Decisions

| Concern | When Decided | Who Decides | Data Source | What It's For |
|---------|--------------|-------------|-------------|---------------|
| **Weight Placement** | At startup | PlacementStrategy | `DeviceInventory.gpu_memory` (TOTAL) | "Which layers fit on which GPU?" |
| **Workspace Budget** | After weights loaded | GraphBufferManager | `IBackend::deviceMemoryFree()` (FREE) | "How much scratch space for kernels?" |

### Timeline: Memory Decisions

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  1. MPITopology + DeviceManager    →  Discover hardware, TOTAL VRAM per GPU │
│         ↓                                                                    │
│  2. PlacementStrategy.compute()    →  Decide weight placement               │
│         ↓                               "24 layers × 200MB = 4.8GB weights" │
│         ↓                               "Fits on 16GB GPU with room to spare"│
│         ↓                                                                    │
│  3. ModelLoader.load()             →  Weights uploaded to GPU               │
│         ↓                                                                    │
│  4. IBackend::deviceMemoryFree()   →  Query ACTUAL free VRAM: ~10.5 GB     │
│         ↓                               (After weights, driver overhead)     │
│         ↓                                                                    │
│  5. GraphBufferManager             →  Compute workspace budget              │
│     .allocateDeviceWorkspace()             "Budget = 10.5GB × 0.8 = 8.4GB"     │
│         ↓                               (Leave headroom for KV cache growth) │
│         ↓                                                                    │
│  6. GraphExecutor.run()            →  Stages use pre-allocated workspace    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Stage-Centric**: Stages declare what they need; the graph doesn't interpret semantics
2. **Budget-First**: GPU workspace has a configurable memory limit per device
3. **Centralized Allocation**: All GPU workspace comes from `DeviceWorkspaceManager`
4. **Fixed Buffers**: Kernels receive buffer pointers, not sizes to allocate
5. **Slab Operations**: Large operations chunk into fixed-size slabs
6. **Backend Agnostic**: Works with CUDA, ROCm, and future backends via `IBackend`
7. **Declarative**: Buffers are just another declaration stages make
8. **Late Binding**: Workspace budget computed AFTER weights loaded (uses actual free memory)

### New Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `WorkspaceDescriptor` | `src/v2/execution/` | Generic buffer descriptor (name, size, dtype) |
| `WorkspaceRequirements` | `src/v2/execution/` | Collection of buffer descriptors |
| `DeviceWorkspaceManager` | `src/v2/execution/` | Per-device workspace allocation |
| `SlabGemmConfig` | `src/v2/kernels/` | Chunked GEMM configuration |

**Note:** No `GpuWorkspaceCategory` enum - the system is semantics-free.

### Integration with Existing Infrastructure

| Existing Component | How It's Used | Notes |
|--------------------|---------------|-------|
| `IBackend` | `deviceMemoryFree()`, `allocate()`, `free()` | Already implemented for ROCm |
| `DeviceId` | Identifies target device for workspace | Existing type |
| `GraphBufferManager` | Extended with `allocateDeviceWorkspace()` | Existing class |
| `IComputeStage` | Extended with `getWorkspaceRequirements()` | Existing interface |
| `BackendManager` | Gets `IBackend*` for device type | Existing singleton |

### Component Interaction Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   GraphBufferManager                                                        │
│          │                                                                   │
│          │ 1. Enumerate devices from graph                                  │
│          ▼                                                                   │
│   ┌──────────────────┐                                                      │
│   │ For each device: │                                                      │
│   │   query budget   │◄─── IBackend::deviceMemoryFree()                     │
│   └────────┬─────────┘                                                      │
│            │                                                                 │
│            │ 2. Collect requirements from stages on this device             │
│            ▼                                                                 │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │ stage->getWorkspaceRequirements()                             │      │
│   │   returns: WorkspaceRequirements {                            │      │
│   │     { "quant_a", M*K, DataType::INT8, ... },                     │      │
│   │     { "scales_a", M, DataType::FP32, ... },                      │      │
│   │     { "acc", M*N, DataType::INT32, ... }                         │      │
│   │   }                                                              │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│            │                                                                 │
│            │ 3. Create DeviceWorkspaceManager with budget                      │
│            ▼                                                                 │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │ DeviceWorkspaceManager(device, budget)                              │      │
│   │   .allocate(combined_requirements)                               │      │
│   │                                                                   │      │
│   │ Internally: backend_->allocate(size, device_idx)                 │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│            │                                                                 │
│            │ 4. Distribute buffers back to stages                           │
│            ▼                                                                 │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │ stage->setGpuWorkspaceBuffers({                                  │      │
│   │   {"quant_a", ptr1},                                             │      │
│   │   {"scales_a", ptr2},                                            │      │
│   │   {"acc", ptr3}                                                  │      │
│   │ })                                                               │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: GPU Workspace Manager

**Goal:** Create centralized GPU workspace allocation with memory budgets.

### 1.1 WorkspaceDescriptor

```cpp
// src/v2/execution/WorkspaceDescriptor.h

namespace llaminar2 {

/**
 * @brief Describes a GPU workspace buffer requirement
 * 
 * NOTE: This is intentionally generic. No semantic fields like "category"
 * or "purpose". The graph doesn't know what buffers are for - just their
 * type, size, and constraints.
 */
struct WorkspaceDescriptor {
    std::string name;           ///< Stage-internal name (opaque to graph)
    size_t size_bytes;          ///< Required size in bytes
    DataType dtype;             ///< Data type (FP32, INT8, INT32, FP16, etc.)
    size_t alignment = 256;     ///< Alignment requirement (default 256 for GPU)
    bool aliasable = false;     ///< Can buffer be shared with non-overlapping stages?
    bool required = true;       ///< If false, allocation failure is not fatal
};

/**
 * @brief Collection of workspace buffer requirements for a stage
 * 
 * Stages populate this with their buffer needs. The graph collects these
 * and passes to the workspace manager - it doesn't interpret the contents.
 */
struct WorkspaceRequirements {
    std::vector<WorkspaceDescriptor> buffers;
    
    size_t totalBytes() const;
    bool empty() const { return buffers.empty(); }
    
    // Generic add methods - NO semantic factory methods here
    void add(const std::string& name, size_t size_bytes, DataType dtype,
             bool aliasable = false);
    void addElements(const std::string& name, size_t count, DataType dtype,
                     bool aliasable = false);
    void addOptional(const std::string& name, size_t size_bytes, DataType dtype);
};

} // namespace llaminar2
```

### 1.2 DeviceWorkspaceManager

```cpp
// src/v2/execution/DeviceWorkspaceManager.h

namespace llaminar2 {

/**
 * @brief Manages GPU workspace buffers for a single device
 *
 * Allocates workspace buffers upfront within a memory budget.
 * Buffers are reused across kernel calls - no hot-path allocations.
 * 
 * DESIGN: This class is buffer-agnostic. It allocates bytes, not semantics.
 */
class DeviceWorkspaceManager {
public:
    /**
     * @brief Construct workspace manager for a device
     * @param device Target GPU device
     * @param budget_bytes Maximum GPU memory to use for workspace (0 = auto)
     */
    DeviceWorkspaceManager(DeviceId device, size_t budget_bytes = 0);
    ~DeviceWorkspaceManager();
    
    // Non-copyable, movable
    DeviceWorkspaceManager(const DeviceWorkspaceManager&) = delete;
    DeviceWorkspaceManager(DeviceWorkspaceManager&&) noexcept;
    
    // =========================================================================
    // Allocation
    // =========================================================================
    
    /**
     * @brief Allocate workspace from requirements
     * @param requirements Workspace requirements to satisfy (from stage)
     * @return true if all required buffers were allocated
     */
    bool allocate(const WorkspaceRequirements& requirements);
    
    /**
     * @brief Release all workspace buffers
     */
    void release();
    
    // =========================================================================
    // Buffer Access
    // =========================================================================
    
    /**
     * @brief Get a workspace buffer by name
     * @param name Buffer name (stage-defined, opaque to manager)
     * @return Device pointer (nullptr if not found)
     */
    void* getBuffer(const std::string& name);
    
    /**
     * @brief Get buffer size
     */
    size_t getBufferSize(const std::string& name) const;
    
    // =========================================================================
    // Query
    // =========================================================================
    
    DeviceId device() const { return device_; }
    size_t budgetBytes() const { return budget_bytes_; }
    size_t allocatedBytes() const { return allocated_bytes_; }
    size_t availableBytes() const { return budget_bytes_ - allocated_bytes_; }
    
private:
    DeviceId device_;
    size_t budget_bytes_;
    size_t allocated_bytes_ = 0;
    
    struct BufferInfo {
        void* ptr = nullptr;
        size_t size = 0;
    };
    std::unordered_map<std::string, BufferInfo> buffers_;
};

} // namespace llaminar2
```

### 1.3 GraphBufferManager Extension

```cpp
// Additions to GraphBufferManager

class GraphBufferManager : public IGraphBufferManager {
public:
    // ... existing interface ...
    
    // =========================================================================
    // GPU Workspace Management (NEW)
    // =========================================================================
    
    /**
     * @brief Set GPU workspace budget for a device
     * @param device Target device
     * @param budget_bytes Maximum workspace memory (0 = auto-detect)
     */
    void setGpuWorkspaceBudget(DeviceId device, size_t budget_bytes);
    
    /**
     * @brief Get workspace manager for a device
     * @return Workspace manager (creates if needed)
     */
    DeviceWorkspaceManager* getDeviceWorkspace(DeviceId device);
    
    /**
     * @brief Allocate GPU workspace for all devices in graph
     * @param graph Compute graph (to determine which devices are used)
     * @return true if allocation succeeded
     */
    bool allocateDeviceWorkspace(const ComputeGraph& graph);
    
    /**
     * @brief Release all GPU workspace
     */
    void releaseDeviceWorkspace();
    
private:
    std::unordered_map<DeviceId, std::unique_ptr<DeviceWorkspaceManager>> device_workspaces_;
    std::unordered_map<DeviceId, size_t> device_workspace_budgets_;
};
```

### 1.4 Budget Computation via IBackend

The workspace budget is computed **after weights are loaded**, using actual free memory:

```cpp
// New methods in GraphBufferManager

class GraphBufferManager {
public:
    // =========================================================================
    // Memory Query (uses IBackend via BackendManager)
    // =========================================================================
    
    /**
     * @brief Query available GPU memory for workspace
     * @param device Target device
     * @return Available bytes (after weights loaded)
     */
    size_t queryAvailableGpuMemory(DeviceId device) {
        // Uses unified BackendManager::getBackendFor() from Phase 0.5.3
        IBackend* backend = BackendManager::getBackendFor(device);
        if (!backend) return 0;
        
        // device.index() works for any device type
        return backend->deviceMemoryFree(device.index());
    }
    
    /**
     * @brief Auto-compute workspace budget for a device
     * 
     * Policy: Use a fraction of free VRAM, leaving headroom for:
     * - KV cache growth during generation
     * - CUDA/HIP runtime overhead
     * - Safety margin
     * 
     * @param device Target device
     * @param fraction Fraction of free memory to use (default: 0.8)
     * @return Computed budget in bytes
     */
    size_t computeWorkspaceBudget(DeviceId device, float fraction = 0.8f) {
        size_t available = queryAvailableGpuMemory(device);
        
        // Reserve headroom for KV cache growth and driver overhead
        constexpr size_t HEADROOM_BYTES = 256 * 1024 * 1024;  // 256 MB
        
        if (available <= HEADROOM_BYTES) {
            LOG_WARN("[GraphBufferManager] Very low GPU memory on " 
                     << device.toString() << ": " << available / (1024*1024) << " MB");
            return 0;
        }
        
        size_t usable = available - HEADROOM_BYTES;
        size_t budget = static_cast<size_t>(usable * fraction);
        
        LOG_INFO("[GraphBufferManager] Device " << device.toString() 
                 << ": " << available / (1024*1024) << " MB free, "
                 << budget / (1024*1024) << " MB workspace budget");
        
        return budget;
    }
    
    // =========================================================================
    // Main Allocation Entry Point
    // =========================================================================
    
    /**
     * @brief Allocate GPU workspace for all devices used by graph
     */
    bool allocateDeviceWorkspace(const ComputeGraph& graph) {
        // 1. Discover devices used by graph
        std::set<DeviceId> devices = graph.getUsedDevices();
        
        for (DeviceId device : devices) {
            if (!device.is_gpu()) continue;  // Skip CPU
            
            // 2. Compute budget if not explicitly set
            size_t budget = device_workspace_budgets_.count(device) 
                ? device_workspace_budgets_[device]
                : computeWorkspaceBudget(device);
            
            if (budget == 0) {
                LOG_WARN("[GraphBufferManager] Zero budget for " << device.toString());
                continue;
            }
            
            // 3. Collect requirements from all stages on this device
            WorkspaceRequirements combined;
            for (const auto& node : graph.nodes()) {
                if (node.device != device) continue;
                
                auto stage_reqs = node.stage->getWorkspaceRequirements();
                combined.merge(stage_reqs);
            }
            
            if (combined.empty()) continue;
            
            // 4. Create manager and allocate
            auto manager = std::make_unique<DeviceWorkspaceManager>(device, budget);
            if (!manager->allocate(combined)) {
                LOG_ERROR("[GraphBufferManager] Failed to allocate workspace on " 
                          << device.toString());
                return false;
            }
            
            // 5. Distribute buffers back to stages
            for (const auto& node : graph.nodes()) {
                if (node.device != device) continue;
                
                auto buffers = manager->getBuffersForStage(node.stage->name());
                if (!buffers.empty()) {
                    node.stage->setGpuWorkspaceBuffers(buffers);
                }
            }
            
            device_workspaces_[device] = std::move(manager);
        }
        
        return true;
    }
};
```

### 1.5 System DRAM Query via CPUBackend

For CPU-side workspace budgets (host pinned buffers, weight streaming), use `CPUBackend` from Phase 0.5:

```cpp
// Query CPU memory via unified IBackend API (same as GPU!)
size_t GraphBufferManager::queryAvailableCpuMemory() {
    IBackend* backend = BackendManager::getCPUBackend();
    // deviceMemoryFree(0) returns this rank's NUMA node available memory
    return backend->deviceMemoryFree(0);
}

size_t GraphBufferManager::computeCpuWorkspaceBudget(float fraction) {
    size_t available = queryAvailableCpuMemory();
    // CPU memory is shared - use conservative fraction (0.3-0.5)
    return static_cast<size_t>(available * fraction);
}
```

**Note:** Unlike GPU budgets (0.8 fraction typical), CPU workspace should use conservative fractions (0.3-0.5) because CPU memory is shared with:
- Operating system
- Other MPI ranks on same machine
- Model weights (replicated portions)
- KV cache (CPU portions)

### 1.6 Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/v2/execution/WorkspaceDescriptor.h` | CREATE | Workspace requirement types |
| `src/v2/execution/DeviceWorkspaceManager.h` | CREATE | Per-device workspace manager |
| `src/v2/execution/DeviceWorkspaceManager.cpp` | CREATE | Implementation |
| `src/v2/execution/GraphBufferManager.h` | MODIFY | Add GPU workspace methods + budget queries |
| `src/v2/execution/GraphBufferManager.cpp` | MODIFY | Add GPU workspace implementation |
| `tests/v2/unit/Test__DeviceWorkspaceManager.cpp` | CREATE | Unit tests |

### 1.7 Acceptance Criteria

**Prerequisites:** Phase 0.5 (CPUBackend) must be complete for unified `BackendManager::getBackendFor()` API.

- [ ] `DeviceWorkspaceManager` can allocate/release buffers within budget
- [ ] `GraphBufferManager::allocateDeviceWorkspace()` works for multi-device
- [ ] Memory budget is respected (allocation fails gracefully if exceeded)
- [ ] `queryAvailableGpuMemory()` returns accurate free memory via `IBackend`
- [ ] `queryAvailableCpuMemory()` returns accurate free memory via `CPUBackend`
- [ ] `computeWorkspaceBudget()` correctly applies fraction and headroom
- [ ] Budget computation happens AFTER weights loaded (uses actual free memory)
- [ ] Unit tests pass for CPU, CUDA, and ROCm backends
- [ ] No memory leaks (valgrind/ASAN clean)

---

## Phase 2: Kernel Integration

**Goal:** Modify GPU kernels to receive workspace buffers instead of allocating internally.

### 2.1 IWorkspaceConsumer Interface

```cpp
// src/v2/interfaces/IWorkspaceConsumer.h

namespace llaminar2 {

/**
 * @brief Interface for kernels that consume GPU workspace
 *
 * Kernels implementing this interface declare their workspace
 * requirements and receive allocated buffers before execution.
 */
class IWorkspaceConsumer {
public:
    virtual ~IWorkspaceConsumer() = default;
    
    /**
     * @brief Get workspace requirements for this kernel
     * @param m, n, k GEMM dimensions (for size calculation)
     * @return Workspace requirements
     */
    virtual WorkspaceRequirements getWorkspaceRequirements(int m, int n, int k) const = 0;
    
    /**
     * @brief Bind workspace manager to this kernel
     * @param workspace Workspace manager with allocated buffers
     */
    virtual void bindWorkspace(DeviceWorkspaceManager* workspace) = 0;
    
    /**
     * @brief Check if workspace is bound
     */
    virtual bool hasWorkspace() const = 0;
};

} // namespace llaminar2
```

### 2.2 ROCmQuantisedGemmKernel Modifications

```cpp
// Changes to ROCmQuantisedGemmKernel

class ROCmQuantisedGemmKernel : public ITensorGemm, public IWorkspaceConsumer {
public:
    // ... existing interface ...
    
    // =========================================================================
    // IWorkspaceConsumer Implementation
    // =========================================================================
    
    WorkspaceRequirements getWorkspaceRequirements(int m, int n, int k) const override;
    void bindWorkspace(DeviceWorkspaceManager* workspace) override;
    bool hasWorkspace() const override;
    
private:
    // Remove ad-hoc buffers from Impl:
    // - d_A_int8, d_scales_A, d_C_int32 → from workspace
    // - d_A_fp16, d_B_fp16, d_C_fp16 → from workspace (slab buffers)
    
    DeviceWorkspaceManager* workspace_ = nullptr;  // NOT owned
};
```

### 2.3 CUDAQuantisedGemmKernel Modifications

Similar changes to `CUDAQuantisedGemmKernel`:
- Implement `IWorkspaceConsumer`
- Remove internal buffer allocations
- Use workspace buffers for quantization and GEMM intermediates

### 2.4 Backward Compatibility

During migration, kernels should work in two modes:

1. **Legacy Mode** (no workspace bound): Allocate internally (current behavior)
2. **Managed Mode** (workspace bound): Use workspace buffers

```cpp
bool ROCmQuantisedGemmKernel::multiply_tensor(...) {
    if (workspace_) {
        // Use managed workspace buffers
        void* d_A_int8 = workspace_->getBuffer("quant_a");
        void* d_scales_A = workspace_->getBuffer("scales_a");
        // ...
    } else {
        // Legacy: use internal Impl buffers (deprecated path)
        ensureWorkBuffers(m);
        void* d_A_int8 = impl_->d_A_int8;
        // ...
    }
}
```

### 2.5 Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/v2/interfaces/IWorkspaceConsumer.h` | CREATE | Workspace consumer interface |
| `src/v2/kernels/rocm/ROCmQuantisedGemmKernel.h` | MODIFY | Add IWorkspaceConsumer |
| `src/v2/kernels/rocm/ROCmQuantisedGemmKernel.cpp` | MODIFY | Use workspace buffers |
| `src/v2/kernels/cuda/CUDAQuantisedGemmKernel.h` | MODIFY | Add IWorkspaceConsumer |
| `src/v2/kernels/cuda/CUDAQuantisedGemmKernel.cpp` | MODIFY | Use workspace buffers |

### 2.6 Acceptance Criteria

- [ ] Kernels work in both legacy and managed modes
- [ ] All existing tests pass (backward compatibility)
- [ ] New tests verify managed mode operation
- [ ] Memory usage is bounded by workspace budget in managed mode

---

## Phase 3: Slab-Based FP16 GEMM

**Goal:** Implement chunked FP16 GEMM that converts INT8→FP16 in fixed-size slabs.

### 3.1 Slab GEMM Algorithm

For GEMM: C[M×N] = A[M×K] × B[K×N]

With slab buffers: `slab_a[slab_m × slab_k]`, `slab_b[slab_k × slab_n]`, `slab_c[slab_m × slab_n]`

```
Algorithm: Slab-Based INT8→FP16 GEMM

Input:
  A_int8[M × K], B_int8[K × N] - INT8 matrices on GPU
  scaleA[M], scaleB[N] - per-row/col scales
  
Output:
  C_fp32[M × N] - FP32 result

Parameters:
  slab_m, slab_n, slab_k - slab dimensions (from workspace budget)

Procedure:
  1. Zero-initialize C_fp32 (or use FP32 accumulator)
  
  2. For k_start = 0 to K step slab_k:
       k_end = min(k_start + slab_k, K)
       actual_k = k_end - k_start
       
       // Convert B slab: INT8 → FP16 with scale
       For n_start = 0 to N step slab_n:
         n_end = min(n_start + slab_n, N)
         convert_int8_to_fp16_with_scale(
           B_int8[k_start:k_end, n_start:n_end],
           scaleB[n_start:n_end],
           slab_b_fp16)
         
         // Process A slabs against this B slab
         For m_start = 0 to M step slab_m:
           m_end = min(m_start + slab_m, M)
           
           // Convert A slab: INT8 → FP16 with scale
           convert_int8_to_fp16_with_scale(
             A_int8[m_start:m_end, k_start:k_end],
             scaleA[m_start:m_end],
             slab_a_fp16)
           
           // FP16 GEMM (hipBLAS hgemm)
           slab_c_fp16 = slab_a_fp16 @ slab_b_fp16
           
           // Accumulate to FP32 output
           C_fp32[m_start:m_end, n_start:n_end] += fp16_to_fp32(slab_c_fp16)
```

### 3.2 SlabGemmConfig

```cpp
// src/v2/kernels/SlabGemmConfig.h

namespace llaminar2 {

/**
 * @brief Configuration for slab-based GEMM execution
 */
struct SlabGemmConfig {
    int slab_m = 256;   ///< Rows per slab
    int slab_n = 256;   ///< Columns per slab  
    int slab_k = 512;   ///< Inner dimension per slab
    
    /**
     * @brief Calculate workspace requirements
     * @param dtype Element data type (FP16, FP32, etc.)
     */
    WorkspaceRequirements workspaceRequirements(DataType dtype) const;
    
    /**
     * @brief Create config from memory budget
     * @param budget_bytes Available workspace memory
     * @param m, n, k Full GEMM dimensions
     * @param dtype Element data type
     */
    static SlabGemmConfig fromBudget(size_t budget_bytes, int m, int n, int k, DataType dtype);
    
    /**
     * @brief Estimate number of slab iterations
     */
    int estimateIterations(int m, int n, int k) const;
};

} // namespace llaminar2
```

### 3.3 ROCm FP16 Slab GEMM

```cpp
// New function in ROCmQuantisedGemmKernel_FP16.hip

/**
 * @brief Execute slab-based INT8→FP16 GEMM with fixed workspace
 *
 * Converts INT8 data to FP16 in slabs and accumulates FP32 result.
 * Uses fixed workspace buffers - no allocations during execution.
 *
 * @param d_A_int8     INT8 activations [M × K]
 * @param d_B_int8     INT8 weights [K × N] (transposed)
 * @param d_C_fp32     FP32 output [M × N]
 * @param d_scaleA     Per-row scales [M]
 * @param d_scaleB     Per-column scales [N]
 * @param M, N, K      Full GEMM dimensions
 * @param slab_a_fp16  Workspace for A slab [slab_m × slab_k]
 * @param slab_b_fp16  Workspace for B slab [slab_k × slab_n]
 * @param slab_c_fp16  Workspace for C slab [slab_m × slab_n]
 * @param config       Slab configuration
 * @param device_id    ROCm device ID
 * @param stream       HIP stream
 */
extern "C" bool rocmQuantGemm_executeSlabFP16(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8,
    float* d_C_fp32,
    const float* d_scaleA,
    const float* d_scaleB,
    int M, int N, int K,
    void* slab_a_fp16,
    void* slab_b_fp16,
    void* slab_c_fp16,
    const SlabGemmConfig* config,
    int device_id,
    void* stream);
```

### 3.4 Memory Analysis

For a 7B model (Qwen2.5-7B):
- FFN dimensions: K=3584, N=18944 (ffn_down)
- Prefill M=512 typical

**Current FP16 path memory:**
```
A_fp16: 512 × 3584 × 2 = 3.5 MB
B_fp16: 3584 × 18944 × 2 = 129 MB  ← Problem!
C_fp16: 512 × 18944 × 2 = 18.4 MB
Total: ~151 MB per GEMM call
```

**Slab-based with 64MB budget:**
```
slab_m=256, slab_k=512, slab_n=4096

slab_a_fp16: 256 × 512 × 2 = 256 KB
slab_b_fp16: 512 × 4096 × 2 = 4 MB
slab_c_fp16: 256 × 4096 × 2 = 2 MB
Total workspace: ~6.3 MB

Iterations: ceil(512/256) × ceil(3584/512) × ceil(18944/4096)
          = 2 × 7 × 5 = 70 slab GEMMs
```

Trade-off: 70 small GEMMs vs 1 large GEMM with 24× memory reduction.

### 3.5 Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/v2/kernels/SlabGemmConfig.h` | CREATE | Slab configuration |
| `src/v2/kernels/SlabGemmConfig.cpp` | CREATE | Budget-based config |
| `src/v2/kernels/rocm/ROCmQuantisedGemmKernel_FP16.hip` | MODIFY | Add slab GEMM |
| `src/v2/kernels/rocm/ROCmQuantisedGemmKernel.cpp` | MODIFY | Use slab GEMM |
| `tests/v2/integration/Test__SlabFP16Gemm.cpp` | CREATE | Integration tests |
| `tests/v2/performance/Perf__SlabVsFullFP16Gemm.cpp` | CREATE | Performance comparison |

### 3.6 Acceptance Criteria

- [ ] Slab GEMM produces numerically identical results to full GEMM
- [ ] Memory usage stays within configured budget
- [ ] Performance regression <30% vs full FP16 for prefill (M≥128)
- [ ] Decode (M=1) performance unchanged (uses INT8 path)
- [ ] Works with all supported weight quantization formats

---

## Phase 4: Memory Budget Enforcement

**Goal:** Add memory budget configuration and enforcement across the system.

### 4.1 Configuration

```cpp
// src/v2/config/MemoryConfig.h

namespace llaminar2 {

struct GpuMemoryConfig {
    size_t workspace_budget_bytes = 0;  ///< 0 = auto (10% of VRAM)
    size_t weight_budget_bytes = 0;     ///< 0 = unlimited
    size_t kv_cache_budget_bytes = 0;   ///< 0 = auto
    
    // Slab GEMM tuning
    int slab_m = 256;
    int slab_n = 256;
    int slab_k = 512;
    
    static GpuMemoryConfig autoDetect(DeviceId device);
    static GpuMemoryConfig fromEnvironment();
};

} // namespace llaminar2
```

### 4.2 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLAMINAR_GPU_WORKSPACE_MB` | Workspace budget per GPU (MB) | Auto (10% VRAM) |
| `LLAMINAR_GPU_SLAB_M` | Slab M dimension | 256 |
| `LLAMINAR_GPU_SLAB_N` | Slab N dimension | 256 |
| `LLAMINAR_GPU_SLAB_K` | Slab K dimension | 512 |
| `LLAMINAR_GPU_FORCE_SLAB` | Force slab GEMM even with enough memory | 0 |

### 4.3 Acceptance Criteria

- [ ] Memory config can be set via environment variables
- [ ] Auto-detection works for CUDA and ROCm
- [ ] Exceeding budget produces clear error message
- [ ] Slab dimensions can be tuned per deployment

---

## Testing Strategy

### Unit Tests

| Test | Scope | Location |
|------|-------|----------|
| `Test__DeviceWorkspaceManager` | Allocation, release, budget | `tests/v2/unit/` |
| `Test__SlabGemmConfig` | Config calculation, budget fitting | `tests/v2/unit/` |
| `Test__GpuWorkspaceConsumer` | Interface compliance | `tests/v2/unit/` |

### Integration Tests

| Test | Scope | Location |
|------|-------|----------|
| `Test__SlabFP16Gemm` | Correctness vs full GEMM | `tests/v2/integration/` |
| `Test__ManagedKernelExecution` | Kernel with workspace | `tests/v2/integration/` |
| `Test__MemoryBudgetEnforcement` | Budget limits work | `tests/v2/integration/` |

### Performance Tests

| Test | Scope | Location |
|------|-------|----------|
| `Perf__SlabVsFullFP16Gemm` | Throughput comparison | `tests/v2/performance/` |
| `Perf__WorkspaceOverhead` | Managed vs legacy | `tests/v2/performance/` |
| `Perf__MemoryBudgetScaling` | Memory vs throughput curve | `tests/v2/performance/` |

### Parity Tests

| Test | Scope | Location |
|------|-------|----------|
| `Parity__SlabGemmVsPyTorch` | Numerical parity | `tests/v2/integration/parity/` |

---

## Migration Guide

### For Kernel Developers

**Before (legacy):**
```cpp
// Kernel allocates its own buffers
class MyGpuKernel {
    struct Impl {
        void* d_temp_buffer = nullptr;
    };
    
    void compute() {
        if (!impl_->d_temp_buffer) {
            hipMalloc(&impl_->d_temp_buffer, size);  // Ad-hoc allocation
        }
    }
};
```

**After (managed):**
```cpp
// Kernel declares requirements, receives workspace
class MyGpuKernel : public IWorkspaceConsumer {
    WorkspaceRequirements getWorkspaceRequirements(int m, int n, int k) const override {
        return {
            {{"temp_buffer", size, 256, true}}
        };
    }
    
    void compute() {
        void* d_temp_buffer = workspace_->getBuffer("temp_buffer");
        // Use buffer...
    }
};
```

### For Graph Developers

```cpp
// Setup workspace before graph execution
GraphBufferManager manager(&factory, &mpi_ctx);

// Set GPU memory budget (optional - auto-detected if not set)
manager.setGpuWorkspaceBudget(DeviceId::cuda(0), 256 * 1024 * 1024);  // 256 MB

// Allocate all buffers including GPU workspace
manager.allocateForGraph(graph);
manager.allocateDeviceWorkspace(graph);

// Execute graph
executor.execute(graph, ctx);

// Cleanup
manager.releaseDeviceWorkspace();
manager.releaseAll();
```

---

## Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Slab GEMM performance regression | High | Medium | Tune slab sizes, benchmark extensively |
| Memory budget too restrictive | Medium | Medium | Auto-detect sensible defaults |
| CUDA/ROCm API differences | Medium | Low | Abstract via IBackend |
| Numerical precision loss | High | Low | Comprehensive parity tests |
| Breaking existing kernels | High | Medium | Backward compatibility mode |

---

## Timeline

| Phase | Duration | Dependencies | Deliverables |
|-------|----------|--------------|--------------|
| Phase 0: Generic Buffer Declaration | 0.5 weeks | None | `WorkspaceDescriptor`, stage interface extensions |
| Phase 0.5: CPUBackend Infrastructure | 0.5 weeks | None | `CPUBackend`, `BackendManager::getBackendFor()` |
| Phase 1: GPU Workspace Manager | 1 week | Phase 0, 0.5 | `DeviceWorkspaceManager`, tests |
| Phase 2: Kernel Integration | 1 week | Phase 1 | Modified kernels, backward compat |
| Phase 3: Slab FP16 GEMM | 1.5 weeks | Phase 1, 2 | `rocmQuantGemm_executeSlabFP16`, tests |
| Phase 4: Memory Budget | 0.5 weeks | Phase 1-3 | Config, env vars, docs |
| **Total** | **5 weeks** | | |

**Critical Path:** Phase 0 → Phase 0.5 → Phase 1 → Phase 2/3 (parallel) → Phase 4

---

## Appendix A: Workspace Buffer Naming Convention

| Buffer Name | Type | Description |
|-------------|------|-------------|
| `gemm_slab_a` | FP16 | Activation slab for FP16 GEMM |
| `gemm_slab_b` | FP16 | Weight slab for FP16 GEMM |
| `gemm_slab_c` | FP16 | Output slab for FP16 GEMM |
| `quant_a` | INT8 | Quantized activations |
| `quant_scales` | FP32 | Quantization scales |
| `accum_c` | INT32 | INT8 GEMM accumulator |
| `fp32_temp` | FP32 | General FP32 workspace |

## Appendix B: Related Documentation

- [CUDA_KERNELS_PROJECT_PLAN.md](CUDA_KERNELS_PROJECT_PLAN.md) - CUDA kernel architecture
- [ROCm_QUANTISED_GEMM_PROJECT_PLAN.md](ROCm_QUANTISED_GEMM_PROJECT_PLAN.md) - ROCm GEMM implementation
- [ARCHITECTURE_REFACTORING_SUMMARY.md](../ARCHITECTURE_REFACTORING_SUMMARY.md) - Overall architecture
- [copilot-instructions.md](../../.github/copilot-instructions.md) - Development guidelines
