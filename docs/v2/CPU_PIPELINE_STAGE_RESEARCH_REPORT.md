# CPU as Pipeline Stage - Research Report

**Date**: January 20, 2026  
**Author**: Research Investigation for Llaminar V2  
**Purpose**: Comprehensive analysis of what's needed for CPU to act as a pipeline stage in heterogeneous CPU+GPU execution

---

## Executive Summary

Llaminar V2 has **strong foundational support** for CPU as a first-class pipeline stage. The architecture was designed from the ground up for heterogeneous execution with per-tensor device affinity and unified kernel dispatch. However, some gaps exist around data transfer optimization and execution scheduling that would need attention for production use.

**Readiness Assessment**:
| Component | Status | Notes |
|-----------|--------|-------|
| CPU Kernel Infrastructure | ✅ **Production-Ready** | Full GEMM, attention, ops coverage |
| Device Abstraction | ✅ **First-Class** | `DeviceId::cpu()` is fully supported |
| Data Transfer (CPU↔GPU) | ⚠️ **Functional** | Coherence exists, pinned memory optional |
| Memory Management | ✅ **NUMA-Aware** | CPUBackend with numa_alloc support |
| Execution Model | ⚠️ **Sequential** | No async CPU↔GPU overlap yet |

---

## 1. CPU Kernel Infrastructure

### 1.1 Kernel Inventory

Location: `src/v2/kernels/cpu/`

```
kernels/cpu/
├── CPURingKVCache.cpp/h      # Ring KV cache for CPU layers
├── CPUKernelBase.h           # Base class for CPU kernels
├── attention/                 # Attention kernels
│   ├── CPUAttentionKernelT.h  # Template attention (FP32/BF16/FP16/Q8_1)
│   ├── AttentionUtils.h
│   ├── q8_1/                  # Q8_1 fused attention
│   │   ├── FusedAttentionWoKernel.h
│   │   ├── FusedAttentionWoTiled.h
│   │   └── jit/              # AVX-512 JIT microkernels
│   └── q16_1/                # Q16_1 attention variants
├── gemm_v4/                   # High-performance GEMM
│   ├── QuantisedGemmKernel.h  # INT8 VNNI JIT kernel
│   ├── FloatingPointGemmKernel.h
│   ├── FusedGEMM.h           # Fused QKV/GateUp
│   ├── QuantisedGemmJit_M1.h  # M=1 decode kernel
│   ├── QuantisedGemmJit_M2.h  # M≥2 batch kernel
│   └── QuantisedAttentionJit_Q8_1_Fused.h
├── ops/                       # Element-wise operations
│   ├── CPUEmbeddingKernelT.h
│   ├── CPURMSNormKernelT.h
│   ├── CPURoPEKernelT.h
│   ├── CPUSoftmaxKernelT.h
│   ├── CPUSwiGLUKernelT.h
│   └── CPUResidualAddKernelT.h
├── primitives/                # Low-level SIMD primitives
│   ├── RMSNormPrimitives.h
│   ├── RoPEPrimitives.h
│   ├── SoftmaxPrimitives.h
│   └── SwiGLUPrimitives.h
└── jit/                       # JIT infrastructure
    ├── RegisterAllocation.h
    ├── RegisterGuard.h
    └── JitMicrokernelBase.h
```

### 1.2 Production Readiness Assessment

| Kernel | Status | Features |
|--------|--------|----------|
| **GEMM** | ✅ Production | AVX-512 VNNI JIT, M=1/M≥2 dispatch, cache-aware blocking |
| **Attention** | ✅ Production | Fused QKV, online softmax, GQA/MQA support |
| **RMSNorm** | ✅ Production | FP32/BF16/FP16/Q8_1 templates |
| **RoPE** | ✅ Production | Position embedding with cos/sin caching |
| **SwiGLU** | ✅ Production | Fused gate+up activation |
| **Embedding** | ✅ Production | Vocabulary lookup |
| **KV Cache** | ✅ Production | HEAD_MAJOR/POSITION_MAJOR layouts, sharding |

### 1.3 Quantized Format Support

From `KernelFactory.h` and `QuantisedGemmKernel.h`:

| Format | CPU Support | GPU Support | Notes |
|--------|-------------|-------------|-------|
| **Q4_0** | ✅ | ✅ | 4-bit quantization, standard |
| **Q4_1** | ✅ | ✅ | 4-bit with bias |
| **Q5_0** | ✅ | ✅ | 5-bit quantization |
| **Q5_1** | ✅ | ✅ | 5-bit with bias |
| **Q6_K** | ✅ | ✅ | 6-bit k-quant |
| **Q8_0** | ✅ | ✅ | 8-bit quantization |
| **Q8_1** | ✅ | ✅ | 8-bit with scale+sum (activations) |
| **Q16_1** | ✅ | ⚠️ Limited | High-precision KV cache |
| **IQ4_NL** | ✅ | ✅ | Improved 4-bit non-linear |
| **Q2_K/Q3_K/Q4_K/Q5_K/Q8_K** | ✅ | ✅ | K-quant family |
| **IQ1_M/IQ1_S/IQ2_*/IQ3_*/IQ4_XS** | ✅ | ✅ | Ultra-low bit variants |

**All quantized formats have CPU kernel implementations via the strategy pattern** (`ITensorGemmTileDataProvider`).

---

## 2. Device Abstraction

### 2.1 DeviceId System

From `src/v2/backends/DeviceId.h`:

```cpp
struct DeviceId {
    DeviceType type;  // CPU, CUDA, ROCm
    int ordinal;      // GPU index (0 for CPU)
    
    // Factory methods
    static DeviceId cpu();       // ✅ First-class CPU support
    static DeviceId cuda(int n);
    static DeviceId rocm(int n);
    
    // Predicates
    bool is_cpu() const;
    bool is_cuda() const;
    bool is_rocm() const;
    bool is_gpu() const;
    bool is_valid() const;
};
```

**CPU is explicitly a first-class device**, not a fallback. The `DeviceId::cpu()` factory creates a valid CPU device that can be assigned to stages.

### 2.2 Stage Device Assignment

From `IComputeStage.h` and `StageParamsBase.h`:

```cpp
// All stage Params MUST include:
struct Params {
    DeviceId device_id = DeviceId::cpu();  // REQUIRED
    const MPIContext* mpi_ctx = nullptr;   // REQUIRED
    // ... stage-specific fields
};
static_assert(StageParamsRequired<Params>);

// Stages store their device and report it:
class IComputeStage {
    DeviceId device_id_;
public:
    explicit IComputeStage(DeviceId device) : device_id_(device) {}
    DeviceId device() const { return device_id_; }
};
```

**Stages CAN be explicitly assigned to `DeviceId::cpu()`**. The `StageParamsRequired` concept enforces that all stages have `device_id` at compile time.

### 2.3 KernelFactory Device Dispatch

From `KernelFactory.h`:

```cpp
class KernelFactory {
    // Device type resolution
    static DeviceType getDeviceType(DeviceId device_id);
    
    // Cached kernel creation with device routing
    static ITensorGemm* getOrCreateGemm(
        const TensorBase* tensor,
        DeviceType target_device);  // Can be DeviceType::CPU
    
    static ITensorGemm* getOrCreateGemm(
        const TensorBase* tensor,
        DeviceId target_device);    // Can be DeviceId::cpu()
    
    // All kernel factories support CPU:
    static unique_ptr<ITensorAttention> createAttention(tensor, DeviceType::CPU);
    static unique_ptr<ITensorRMSNorm> createRMSNorm(tensor, DeviceType::CPU);
    static unique_ptr<ITensorRoPE> createRoPE(tensor, DeviceType::CPU);
    static unique_ptr<ITensorSwiGLU> createSwiGLU(tensor, DeviceType::CPU);
    // ... etc
};
```

**KernelFactory explicitly handles `DeviceType::CPU`** in its dispatch logic. Passing CPU creates CPU kernel implementations.

---

## 3. CPU↔GPU Data Transfer

### 3.1 Coherence Protocol

From `src/v2/tensors/cpu/CPUTensors.h` and `StageCoherence.h`:

```cpp
class CPUTensorBase {
    // State tracking
    bool host_valid_;     // Host data is authoritative
    bool device_valid_;   // GPU data is authoritative
    void* gpu_data_ptr_;  // GPU buffer (nullptr if not uploaded)
    
    // Coherence API
    bool ensureOnDevice(DeviceId target_device);  // CPU→GPU upload
    bool ensureOnHost();                          // GPU→CPU download
    void mark_device_dirty();                     // Mark GPU as authoritative
    void invalidateGpuData();                     // Force re-upload
    
    // Data access (auto-syncs from GPU if needed)
    const float* data();      // Syncs from GPU if device_valid_
    float* mutable_data();    // Marks host as authoritative
};
```

The coherence protocol is **automatic at stage boundaries**:

```cpp
// DeviceGraphExecutor handles coherence per stage's policy:
bool DeviceGraphExecutor::executeNode(ComputeNode& node, IDeviceContext* ctx) {
    DeviceId target_device = node.device;
    auto policy = node.stage->coherencePolicy();
    
    if (policy == FULL || policy == INPUT) {
        auto inputs = extractInputBuffers(dump_info);
        cohereInputs(inputs, target_device);  // CPU→GPU if target is GPU
    }
    
    node.stage->execute(ctx);
    
    if (policy == FULL || policy == OUTPUT) {
        markOutputsDirty(outputs);  // Mark outputs as device-authoritative
    }
}
```

### 3.2 Pinned Memory Support

From `CPUTensors.h`:

```cpp
class CPUTensorBase {
    bool ensureHostPinned();  // cudaHostRegister / hipHostRegister
    bool is_mapped_;          // True if using zero-copy mapped memory
    
    // For mapped tensors:
    // - ensureOnDevice() becomes a no-op
    // - GPU reads directly from host memory via PCIe
};
```

Pinned memory is **optional but supported**:
- `ensureHostPinned()` pins host memory for faster DMA transfers
- Mapped memory (`hipHostMallocMapped`) enables zero-copy GPU access
- Used automatically by the coherence system when available

### 3.3 Expected Transfer Latency

For a **14KB activation transfer** (e.g., 3584 elements × 4 bytes):

| Transfer Type | Expected Latency | Notes |
|---------------|------------------|-------|
| Pageable H2D | ~15-30 μs | Standard memcpy + PCIe |
| Pinned H2D | ~5-10 μs | DMA transfer |
| Zero-copy | ~0 μs sync | GPU reads via PCIe on demand |
| PCIe Gen4 x16 | ~0.5 μs theoretical | 32 GB/s bandwidth |

**Key insight**: For 14KB transfers, **latency dominates bandwidth**. The PCIe setup overhead (~5μs) is significant relative to the actual transfer time (<1μs for 14KB at 32GB/s).

### 3.4 Data Transfer Mechanisms

From `IBackend.h` and `BackendManager.h`:

```cpp
class IBackend {
    // Synchronous transfers
    virtual bool deviceToHost(void* dst, const void* src, size_t bytes, int device_id) = 0;
    virtual bool hostToDevice(void* dst, const void* src, size_t bytes, int device_id) = 0;
    
    // Event-based synchronization
    virtual void* createEvent(int device_id) = 0;
    virtual void recordEvent(void* event, int device_id) = 0;
    virtual void waitEvent(void* event, int device_id) = 0;
};

// Unified backend access
IBackend* getBackendFor(DeviceId device);  // Works for CPU, CUDA, ROCm
```

---

## 4. Memory Management

### 4.1 CPUBackend - NUMA-Aware Allocation

From `src/v2/backends/CPUBackend.h`:

```cpp
class CPUBackend : public IBackend {
    int local_numa_node_;  // NUMA node for this MPI rank
    
public:
    explicit CPUBackend(int local_numa_node);
    
    // Memory allocation (NUMA-aware)
    void* allocate(size_t bytes, int device_id) override;
    // Uses numa_alloc_onnode() if libnuma available,
    // otherwise aligned_alloc() with first-touch
    
    // Memory query (per-NUMA node)
    size_t deviceMemoryTotal(int device_id) const override;
    size_t deviceMemoryFree(int device_id) const override;
    // Reads from /sys/devices/system/node/nodeN/meminfo
};
```

**NUMA-aware by design**: Each MPI rank gets its own `CPUBackend` bound to its local NUMA node.

### 4.2 CPU KV Cache

From `src/v2/kernels/cpu/CPUKVCache.h`:

```cpp
class ICPUKVCache : public IKVCache {
    // Supports FP32, BF16, FP16, Q8_1, Q16_1 precision
    virtual ActivationPrecision precision() const = 0;
    
    // Layout modes for different access patterns
    // POSITION_MAJOR: [position][n_kv_heads][head_dim] - cache-append friendly
    // HEAD_MAJOR: [n_kv_heads][position][head_dim] - attention-compute friendly
    virtual KVCacheLayoutMode layout() const = 0;
    
    // Sharding for tensor parallelism
    virtual bool is_sharded() const = 0;
    virtual int local_n_kv_heads() const = 0;
};
```

**KV cache CAN live on CPU for CPU-owned layers**. The `KernelFactory::createCPUKVCache()` creates appropriately typed caches.

### 4.3 Weight Tensor Placement

Weight tensors are **CPU-resident by default**:

```cpp
// Weights loaded from GGUF stay on CPU
auto weight = loader.loadTensor("layer.0.attn.q_proj.weight");
// weight->home_device() == DeviceId::cpu()

// GPU upload happens on demand via coherence:
weight->ensureOnDevice(DeviceId::rocm(0));  // Uploads if needed
```

For CPU-only layers, weights **never need to be uploaded to GPU**.

---

## 5. Execution Model

### 5.1 Current DeviceGraphExecutor Behavior

From `src/v2/execution/DeviceGraphExecutor.cpp`:

```cpp
bool DeviceGraphExecutor::executeNode(ComputeNode& node, IDeviceContext* ctx) {
    // 1. Determine target device from node or stage
    DeviceId target_device = node.device.is_valid() ? node.device : node.stage->device();
    
    // 2. Cohere inputs to target device
    if (policy == INPUT || policy == FULL) {
        cohereInputs(inputs, target_device);
        cohereInputs(weights, target_device);  // Weights too
    }
    
    // 3. Allocate output buffers on target device
    if (policy == OUTPUT || policy == FULL) {
        cohereOutputs(outputs, target_device);
    }
    
    // 4. Execute stage
    node.stage->execute(ctx);
    
    // 5. Mark outputs dirty (device-authoritative)
    markOutputsDirty(outputs);
}
```

**Mixed CPU/GPU execution is ALREADY supported** - each node has its own `device` field, and coherence handles data movement automatically.

### 5.2 ComputeGraph with Mixed Devices

```cpp
ComputeGraph graph;

// CPU stage for layer 0
graph.addNode("layer0_attn", createAttentionStage(params_cpu), DeviceId::cpu());

// GPU stages for layer 1+
graph.addNode("layer1_attn", createAttentionStage(params_gpu), DeviceId::rocm(0));
graph.addNode("layer2_attn", createAttentionStage(params_gpu), DeviceId::rocm(0));

// Dependencies determine execution order
graph.addDependency("layer1_attn", "layer0_attn");
graph.addDependency("layer2_attn", "layer1_attn");

// Execute - coherence handles CPU→GPU transfer between layer0 and layer1
executor.execute(graph, ctx);
```

### 5.3 Async Execution - Current Gaps

**Current limitation**: DeviceGraphExecutor executes stages **sequentially**:

```cpp
// Current: sequential execution
for (const auto& name : execution_order) {
    executeNode(*graph.getNode(name), ctx);  // Blocks until complete
}
```

**What's missing for true CPU/GPU overlap**:
1. **Async CPU stage execution**: CPU stages could run on separate threads
2. **Stream-based GPU scheduling**: GPU stages could use CUDA/HIP streams
3. **Dependency-aware scheduling**: Ready nodes could execute in parallel

The architecture supports this (nodes have devices, dependencies are tracked), but implementation is sequential.

---

## 6. Gaps and Recommendations

### 6.1 Device Abstraction - No Gaps ✅

- CPU is a first-class `DeviceId`
- `KernelFactory` dispatches correctly to CPU kernels
- Stages can be assigned to `DeviceId::cpu()`

### 6.2 Data Transfer - Minor Gaps ⚠️

**Gap**: No explicit pinned memory pool for small activation transfers.

**Impact**: Each CPU→GPU transfer may incur pinning overhead.

**Recommendation**: 
```cpp
// Add to DeviceGraphBufferManager or DeviceWorkspaceManager:
class PinnedMemoryPool {
    void* allocatePinned(size_t bytes);  // From pre-pinned pool
    void deallocate(void* ptr);
};
```

### 6.3 Execution Model - Significant Gap ⚠️

**Gap**: No async/parallel stage execution.

**Impact**: CPU stages block GPU, GPU blocks CPU - no overlap.

**Recommendation** (phased approach):

**Phase 1**: Async GPU stages (low effort)
```cpp
// GPU stages enqueue work, don't block
stage->execute(ctx);  // Returns immediately
// Only sync at dependency boundaries
```

**Phase 2**: Thread pool for CPU stages (medium effort)
```cpp
// CPU stages run on worker threads
if (node.device.is_cpu()) {
    thread_pool.submit([&]() { stage->execute(ctx); });
} else {
    stage->execute(ctx);  // GPU async on stream
}
```

**Phase 3**: Full dependency-aware scheduler (higher effort)
```cpp
// WorkStealingScheduler examines ready nodes, dispatches to appropriate device
class HeterogeneousScheduler {
    void scheduleGraph(ComputeGraph& graph) {
        while (!graph.allCompleted()) {
            auto ready = graph.getReadyNodes();
            for (auto& name : ready) {
                auto* node = graph.getNode(name);
                if (node->device.is_cpu()) {
                    cpu_pool_.submit(node);
                } else {
                    gpu_stream_.enqueue(node);
                }
            }
            // Wait for any completion
            waitAny();
        }
    }
};
```

### 6.4 Memory Management - Minor Gap ⚠️

**Gap**: No explicit guidance for which layers to keep on CPU.

**Recommendation**: Add layer placement strategy:
```cpp
struct LayerPlacementConfig {
    int cpu_layers_start = 0;   // First N layers on CPU
    int cpu_layers_end = 0;     // Or specific range
    bool cpu_kv_cache = false;  // KV cache on CPU for CPU layers
};
```

---

## 7. Recommended Implementation Approach

### Minimal Viable CPU Pipeline Stage

To enable CPU as a pipeline stage with current architecture:

1. **Stage Creation**: Use `DeviceId::cpu()` in stage Params
   ```cpp
   GEMMStage::Params params;
   params.device_id = DeviceId::cpu();
   params.weights = cpu_weights;
   params.input = cpu_activations;
   auto stage = std::make_unique<GEMMStage>(params);
   ```

2. **Graph Construction**: Assign device to node
   ```cpp
   graph.addNode("cpu_layer0_gemm", std::move(stage), DeviceId::cpu());
   ```

3. **Coherence**: Automatic - `DeviceGraphExecutor` handles CPU↔GPU at boundaries

4. **KV Cache**: Use `KernelFactory::createCPUKVCache()` for CPU layers

### Performance Optimization Path

1. **Immediate** (no code changes):
   - Use `LLAMINAR_STAGE_OUTPUT_PRINT` to verify data flow
   - Profile with `LLAMINAR_PROFILE_KERNELS=1`

2. **Short-term** (minor changes):
   - Add pinned memory pool for activation buffers
   - Enable mapped memory for output tensors going GPU→CPU

3. **Medium-term** (architecture changes):
   - Implement async GPU stage execution via streams
   - Add CPU thread pool for parallel CPU stage execution

4. **Long-term** (major feature):
   - Full heterogeneous scheduler with overlap optimization

---

## 8. Conclusion

Llaminar V2 is **architecturally ready** for CPU as a pipeline stage:

| Requirement | Status | Action Needed |
|-------------|--------|---------------|
| CPU kernels for all ops | ✅ Done | None |
| DeviceId::cpu() support | ✅ Done | None |
| Stage device assignment | ✅ Done | None |
| KernelFactory CPU dispatch | ✅ Done | None |
| CPU↔GPU coherence | ✅ Done | None |
| CPU KV cache | ✅ Done | None |
| NUMA-aware allocation | ✅ Done | None |
| Pinned memory pool | ⚠️ Optional | Minor addition |
| Async execution | ⚠️ Missing | Medium effort |
| Layer placement config | ⚠️ Missing | Minor addition |

**Bottom line**: CPU stages work TODAY with the existing infrastructure. The main limitation is sequential execution (no CPU/GPU overlap), which can be addressed incrementally without architectural changes.
