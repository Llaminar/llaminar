# Project Plan: Device Abstraction and Stage Simplification

**Date**: January 10, 2026  
**Status**: ✅ IMPLEMENTED  
**Branch**: `feature/cuda-kernels`

---

## Implementation Summary

All five phases of this project plan have been successfully implemented:

| Phase | Status | Key Deliverables |
|-------|--------|------------------|
| **Phase 1: KVCache in KernelFactory** | ✅ Complete | `KVCacheConfig`, `createKVCache()`, `createCPUKVCache()` |
| **Phase 2: Stage Boundary Coherence** | ✅ Complete | `StageCoherence.h/cpp`, `CoherencePolicy`, `coherencePolicy()` |
| **Phase 3: Stage Cleanup** | ✅ Complete | Guarded manual coherence calls with `isAutoCoherenceEnabled()` |
| **Phase 4: Kernel Interface Refactoring** | ✅ Complete | `ITensorFusedQKVGemm`, `ITensorFusedGateUpGemm`, KernelFactory integration |
| **Phase 5: Integration Testing** | ✅ Complete | 224+ unit tests passing, integration tests verified |

### New Files Created
- `src/v2/execution/StageCoherence.h` - Coherence policy and helper declarations
- `src/v2/execution/StageCoherence.cpp` - Coherence implementation
- `tests/v2/unit/Test__KernelFactory_KVCache.cpp` - KVCache factory unit tests (16 tests)
- `tests/v2/unit/Test__StageCoherence.cpp` - Stage coherence unit tests (23 tests)

### Modified Files
- `src/v2/kernels/KernelFactory.h/cpp` - Added KVCache, FusedQKVGemm, FusedGateUpGemm factories
- `src/v2/execution/GraphOrchestrator.cpp` - Uses KernelFactory for KVCache creation
- `src/v2/execution/DeviceGraphExecutor.cpp` - Automatic coherence at stage boundaries
- `src/v2/execution/compute_stages/IComputeStage.h` - Added `coherencePolicy()`, `preferredDevice()`, tensor pointers
- `src/v2/execution/compute_stages/stages/*Stage.cpp` - Tensor pointers and coherence policies
- `src/v2/tensors/TensorKernels.h` - Added `ITensorFusedQKVGemm`, `ITensorFusedGateUpGemm`

### Usage

Automatic stage coherence is always enabled. The DeviceGraphExecutor handles device coherence automatically at stage boundaries based on each stage's `coherencePolicy()`:
```bash
# No environment variable needed - coherence is automatic
./build_v2_release/llaminar2 -m model.gguf -p "Hello"
```

---

## Executive Summary

This document outlines three interconnected architectural improvements to the Llaminar V2 inference engine:

1. **KVCache in KernelFactory** - Treat KVCache as a kernel, enable device-aware creation
2. **Automatic Stage Boundary Coherence** - ✅ INFRASTRUCTURE COMPLETE - Eliminate manual `ensureOnDevice()` boilerplate
3. **Stages Using Kernel Interfaces** - Unified device-agnostic kernel access pattern

These changes will simplify GPU support, reduce boilerplate, and create a single code path for CPU/CUDA/ROCm execution.

---

## Table of Contents

- [1. KVCache Creation via KernelFactory](#1-kvcache-creation-via-kernelfactory)
- [2. Automatic Stage Boundary Coherence](#2-automatic-stage-boundary-coherence)
- [3. Stages Using Kernel Interfaces](#3-stages-using-kernel-interfaces)
- [Implementation Order](#implementation-order)
- [Risk Assessment](#risk-assessment)
- [Testing Strategy](#testing-strategy)

---

## 1. KVCache Creation via KernelFactory

### Goal

Enable device-aware KVCache creation through `KernelFactory::createKVCache()`. Consumers request a cache by DeviceId and receive the appropriate implementation (CPU or CUDA).

### Current State

| Implementation | File | Precision Support |
|---------------|------|-------------------|
| `CPURingKVCache<T>` | `src/v2/kernels/cpu/CPURingKVCache.h` | FP32, BF16, FP16, Q8_1, Q16_1 |
| `CUDARingKVCache<T>` | `src/v2/kernels/cuda/CUDARingKVCache.h` | FP32, BF16, FP16 |
| `ShardedCPURingKVCache<T>` | (via CPURingKVCache constructors) | Same as CPU |

**Current Creation (GraphOrchestrator.cpp lines 1000-1060)**:
```cpp
// Always creates CPU cache - no CUDA path!
if (use_sharded_cache && mpi_ctx_->world_size() > 1) {
    state_.kv_cache = createShardedCPURingKVCache(...);
} else {
    state_.kv_cache = createCPURingKVCache(...);
}
```

### Proposed API

#### KVCacheConfig Struct

```cpp
// src/v2/kernels/KernelFactory.h

struct KVCacheConfig {
    // Required
    int n_layers;
    int n_kv_heads;
    int head_dim;
    int max_seq_len;
    
    // Optional
    int batch_size = 1;
    ActivationPrecision precision = ActivationPrecision::FP32;
    KVCacheLayoutMode layout_mode = KVCacheLayoutMode::POSITION_MAJOR;
    
    // Sharding (tensor parallelism)
    int local_n_kv_heads = 0;   // 0 = not sharded
    int kv_head_start = 0;
    
    // Device
    DeviceId device = DeviceId::cpu();
    const MPIContext* mpi_ctx = nullptr;  // Required for CPU
    
    // Helpers
    bool is_sharded() const { return local_n_kv_heads > 0 && local_n_kv_heads < n_kv_heads; }
    bool is_cuda() const { return device.is_cuda(); }
};
```

#### Factory Methods

```cpp
class KernelFactory {
public:
    // Device-aware dispatch
    static std::unique_ptr<IKVCache> createKVCache(const KVCacheConfig& config);
    
    // Type-specific (when caller knows device)
    static std::unique_ptr<ICPUKVCache> createCPUKVCache(const KVCacheConfig& config);
#ifdef HAVE_CUDA
    static std::unique_ptr<ICUDARingKVCache> createCUDAKVCache(const KVCacheConfig& config);
#endif
};
```

### Files to Modify

| File | Changes |
|------|---------|
| `src/v2/kernels/KernelFactory.h` | Add `KVCacheConfig`, factory method declarations |
| `src/v2/kernels/KernelFactory.cpp` | Add implementations |
| `src/v2/execution/GraphOrchestrator.cpp` | Use `KernelFactory::createKVCache()` |
| **New**: `tests/v2/unit/Test__KernelFactory_KVCache.cpp` | Unit tests |

### Implementation Details

```cpp
// KernelFactory.cpp

std::unique_ptr<IKVCache> KernelFactory::createKVCache(const KVCacheConfig& config) {
    DeviceType dev_type = getDeviceType(config.device);
    
    switch (dev_type) {
    case DeviceType::CPU:
        return createCPUKVCache(config);
        
    case DeviceType::CUDA:
#ifdef HAVE_CUDA
        return createCUDAKVCache(config);
#else
        throw std::runtime_error("CUDA KVCache requested but HAVE_CUDA not enabled");
#endif
        
    default:
        throw std::runtime_error("Unsupported device type for KVCache");
    }
}

std::unique_ptr<ICPUKVCache> KernelFactory::createCPUKVCache(const KVCacheConfig& config) {
    if (!config.mpi_ctx) {
        throw std::runtime_error("KernelFactory::createCPUKVCache: mpi_ctx required");
    }
    
    if (config.is_sharded()) {
        return llaminar2::createShardedCPURingKVCache(
            config.precision, *config.mpi_ctx,
            config.n_layers, config.batch_size, config.max_seq_len,
            config.n_kv_heads, config.local_n_kv_heads, config.kv_head_start,
            config.head_dim, config.device, config.layout_mode);
    } else {
        return llaminar2::createCPURingKVCache(
            config.precision, *config.mpi_ctx,
            config.n_layers, config.batch_size, config.max_seq_len,
            config.n_kv_heads, config.head_dim, config.device, config.layout_mode);
    }
}

#ifdef HAVE_CUDA
std::unique_ptr<ICUDARingKVCache> KernelFactory::createCUDAKVCache(const KVCacheConfig& config) {
    if (config.is_sharded()) {
        LOG_WARN("[KernelFactory] Sharded CUDA KVCache not yet supported");
    }
    
    return llaminar2::createCUDARingKVCache(
        config.precision, config.n_layers, config.batch_size, config.max_seq_len,
        config.n_kv_heads, config.head_dim, config.device.cuda_ordinal());
}
#endif
```

### Edge Cases

| Case | Handling |
|------|----------|
| Multi-GPU | Each call creates cache on specified `config.device` |
| Sharded CUDA | Not supported - warn and use full KV heads |
| Q8_1/Q16_1 on CUDA | Not supported - throw error |
| Missing MPI context | Throw for CPU, ignore for CUDA |

### Estimated Effort

- **Implementation**: 4 hours
- **Testing**: 2 hours
- **GraphOrchestrator migration**: 1 hour

---

## 2. Automatic Stage Boundary Coherence ✅ INFRASTRUCTURE COMPLETE

**Implementation Date**: January 10, 2026

### Infrastructure Files Created/Modified

| File | Status | Description |
|------|--------|-------------|
| `src/v2/execution/StageCoherence.h` | ✅ Created | Coherence policy enum, helper declarations |
| `src/v2/execution/StageCoherence.cpp` | ✅ Created | Implementation of coherence helpers |
| `src/v2/execution/compute_stages/IComputeStage.h` | ✅ Modified | Added `coherencePolicy()`, `preferredDevice()`, tensor pointers in buffers |
| `src/v2/execution/DeviceGraphExecutor.cpp` | ✅ Modified | Added automatic coherence at stage boundaries |
| `src/v2/CMakeLists.txt` | ✅ Modified | Added StageCoherence.cpp to build |

### How It Works

Automatic coherence is always enabled. The DeviceGraphExecutor handles device coherence at stage boundaries based on each stage's `coherencePolicy()` (FULL by default):

```bash
# No configuration needed - coherence is automatic
./build_v2_release/llaminar2 -m model.gguf -p "test"
```

### Next Steps

- **Phase 2b**: Update stages to populate `tensor` pointers in `getDumpInfo()` return values
- **Phase 2c**: Enable by default and remove manual coherence calls from stages

### Goal

Eliminate manual `ensureOnDevice()`/`mark_device_dirty()` calls in stages by adding automatic coherence at stage boundaries in `DeviceGraphExecutor`.

### Current State

**Manual Coherence Pattern (scattered throughout stages)**:
```cpp
// In AttentionComputeStage.cpp
if (!output_base->ensureOnDevice(params_.device_id)) {
    LOG_ERROR("Failed to ensure output tensor on device");
    return false;
}
// ... GPU kernel execution ...
output_base->mark_device_dirty();
```

**Statistics**:
- ~6-10 manual coherence calls per GPU-capable stage
- ~200 lines of boilerplate across all stages
- Risk of bugs from forgotten coherence calls

### Proposed Design

#### Coherence Policy Enum

```cpp
// src/v2/execution/StageCoherence.h

enum class CoherencePolicy {
    NONE,           // No automatic coherence (legacy/special)
    INPUTS_ONLY,    // Sync inputs before execute
    OUTPUTS_ONLY,   // Mark outputs after execute
    FULL,           // Both inputs and outputs (default)
};
```

#### IComputeStage Extension

```cpp
class IComputeStage {
public:
    // Existing methods...
    
    // NEW: Coherence hints (with defaults for backward compatibility)
    virtual CoherencePolicy coherencePolicy() const { 
        return CoherencePolicy::FULL; 
    }
    
    virtual DeviceId preferredDevice() const {
        return DeviceId::cpu();
    }
};
```

#### StageDumpInfo Extension

```cpp
struct StageDumpInfo {
    struct InputBuffer {
        const char* name = nullptr;
        const void* data = nullptr;
        const ITensor* tensor = nullptr;  // NEW: For coherence
        size_t rows = 0;
        size_t cols = 0;
        const char* dtype = "FP32";
    };
    
    struct OutputBuffer {
        const char* name = nullptr;
        const void* data = nullptr;
        ITensor* tensor = nullptr;  // NEW: Non-const for mark_dirty
        size_t rows = 0;
        size_t cols = 0;
        const char* dtype = "FP32";
    };
};
```

#### DeviceGraphExecutor Integration

```cpp
// DeviceGraphExecutor.cpp

bool DeviceGraphExecutor::executeNode(ComputeNode& node, IDeviceContext* ctx) {
    if (!node.stage) return false;
    
    // Determine execution device
    DeviceId exec_device = determineExecutionDevice(node, ctx);
    
    // PRE-EXECUTION: Cohere inputs
    CoherencePolicy policy = node.stage->coherencePolicy();
    if (policy == CoherencePolicy::FULL || policy == CoherencePolicy::INPUTS_ONLY) {
        if (!cohereStageInputs(node.stage.get(), exec_device)) {
            LOG_ERROR("Failed to cohere stage inputs for " << node.name);
            return false;
        }
    }
    
    // Entry verification...
    
    // EXECUTION
    bool success = node.stage->execute(ctx);
    
    // POST-EXECUTION: Mark outputs dirty
    if (success && (policy == CoherencePolicy::FULL || policy == CoherencePolicy::OUTPUTS_ONLY)) {
        markStageOutputsDirty(node.stage.get(), exec_device);
    }
    
    // Exit verification...
    
    return success;
}
```

### Files to Modify

| File | Changes |
|------|---------|
| **New**: `src/v2/execution/StageCoherence.h` | Coherence helpers |
| **New**: `src/v2/execution/StageCoherence.cpp` | Implementation |
| `src/v2/execution/compute_stages/IComputeStage.h` | Add `coherencePolicy()`, `preferredDevice()` |
| `src/v2/execution/compute_stages/StageDumpInfo.h` | Add `ITensor*` to buffers |
| `src/v2/execution/DeviceGraphExecutor.cpp` | Add coherence calls |
| All GPU-capable stages | Remove manual coherence |

### Stages to Clean Up

| Stage | Manual Coherence Calls to Remove |
|-------|----------------------------------|
| `AttentionComputeStage` | ~10 lines |
| `GEMMStage` | ~6 lines (in kernel) |
| `FusedQKVGEMMStage` | ~8 lines |
| `FusedGateUpGEMMStage` | ~4 lines |
| `RMSNormStage` | ~4 lines |
| `LMHeadStage` | ~6 lines |

### Special Cases

| Stage | Policy | Reason |
|-------|--------|--------|
| `ResidualAddStage` | FULL | Has INOUT buffer - cohere as INPUT, mark as OUTPUT |
| `KVCacheAppendStage` | NONE | KV cache manages its own coherence |
| `AllreduceStage` | NONE | MPI operations, no device transfer |
| `AllGatherStage` | NONE | MPI operations, no device transfer |

### Estimated Effort

- **Infrastructure (StageCoherence.h/cpp)**: 3 hours
- **StageDumpInfo extension**: 1 hour
- **DeviceGraphExecutor integration**: 2 hours
- **Stage cleanup**: 4 hours (20 stages × 12 min each)
- **Testing**: 3 hours

---

## 3. Stages Using Kernel Interfaces

### Goal

All ComputeStages should use kernel interfaces (ITensorGemm, ITensorAttention, etc.) via KernelFactory, with no device-specific branching in stage code.

### Current State Summary

| Category | Stages | Status |
|----------|--------|--------|
| ✅ Using KernelFactory correctly | RMSNormStage, RoPEStage, SwiGLUStage, GEMMStage, AttentionComputeStage | Good |
| ⚠️ Direct kernel instantiation | FusedQKVGEMMStage, FusedAttentionWoStage, FusedGateUpGEMMStage | Need refactor |
| ⚠️ Legacy patterns | EmbeddingStage | Needs migration |

### Recommended Pattern

```cpp
bool MyStage::execute(IDeviceContext* ctx) {
    // 1. Get TensorBase* for kernel API
    auto* input_base = requireTensorBase(params_.input, "input");
    auto* output_base = requireTensorBase(params_.output, "output");

    // 2. Resolve device type from stage's DeviceId
    auto dev_type = KernelFactory::getDeviceType(params_.device_id);

    // 3. Create/get kernel via factory
    auto kernel = KernelFactory::createMyKernel(input_base, dev_type);
    if (!kernel) {
        LOG_ERROR("[MyStage] Failed to create kernel");
        return false;
    }

    // 4. Execute via interface - kernel handles device internally
    return kernel->apply_tensor(input_base, output_base, ...);
}
```

### Stages Requiring Refactoring

#### Priority 1: Direct Kernel Instantiation

**FusedQKVGEMMStage** (Before):
```cpp
FusedGEMM fused_gemm(wq_base, wk_base, wv_base);  // Direct instantiation!
bool success = fused_gemm.execute_q8_1_mixed_qkv(...);
```

**FusedQKVGEMMStage** (After):
```cpp
auto dev_type = KernelFactory::getDeviceType(params_.device_id);
auto kernel = KernelFactory::getOrCreateFusedQKVGemm(wq_base, wk_base, wv_base, dev_type);
return kernel->execute(params_.input, params_.output_q, params_.output_k, params_.output_v, ...);
```

**Required Factory Additions**:
```cpp
// KernelFactory.h
static std::unique_ptr<ITensorFusedQKVGemm> createFusedQKVGemm(
    const TensorBase* wq, const TensorBase* wk, const TensorBase* wv, DeviceType dev_type);
static ITensorFusedQKVGemm* getOrCreateFusedQKVGemm(
    const TensorBase* wq, const TensorBase* wk, const TensorBase* wv, DeviceType dev_type);

static std::unique_ptr<ITensorFusedAttentionWo> createFusedAttentionWo(
    const TensorBase* wo, DeviceType dev_type, FusedAttentionBackend backend);
```

#### Priority 2: Legacy Pattern

**EmbeddingStage** (Before):
```cpp
auto* activation = dynamic_cast<IActivationTensor*>(output_base);
auto kernel = activation->createEmbedding();  // Tensor creates its own kernel!
```

**EmbeddingStage** (After):
```cpp
auto dev_type = KernelFactory::getDeviceType(params_.device_id);
auto kernel = KernelFactory::createEmbedding(vocab_base, dev_type);
return kernel->apply(token_ids, output_base, ...);
```

### Files to Modify

| File | Changes |
|------|---------|
| `src/v2/kernels/KernelFactory.h` | Add fused kernel factory methods |
| `src/v2/kernels/KernelFactory.cpp` | Implement fused kernel creation |
| **New**: `src/v2/kernels/interfaces/ITensorFusedQKVGemm.h` | Interface definition |
| **New**: `src/v2/kernels/interfaces/ITensorFusedAttentionWo.h` | Interface definition |
| `src/v2/execution/compute_stages/stages/FusedQKVGEMMStage.cpp` | Refactor |
| `src/v2/execution/compute_stages/stages/FusedAttentionWoStage.cpp` | Refactor |
| `src/v2/execution/compute_stages/stages/FusedGateUpGEMMStage.cpp` | Refactor |
| `src/v2/execution/compute_stages/stages/EmbeddingStage.cpp` | Refactor |

### Estimated Effort

- **ITensorFusedQKVGemm interface**: 2 hours
- **ITensorFusedAttentionWo interface**: 2 hours
- **KernelFactory additions**: 3 hours
- **FusedQKVGEMMStage refactor**: 2 hours
- **FusedAttentionWoStage refactor**: 2 hours
- **FusedGateUpGEMMStage refactor**: 1 hour
- **EmbeddingStage refactor**: 1 hour
- **Testing**: 3 hours

---

## Implementation Order

### Phase 1: Foundation (Week 1)

| Task | Priority | Dependencies | Effort |
|------|----------|--------------|--------|
| 1.1 Add KVCacheConfig to KernelFactory.h | P0 | None | 1h |
| 1.2 Implement createKVCache() in KernelFactory.cpp | P0 | 1.1 | 2h |
| 1.3 Write KVCache factory unit tests | P0 | 1.2 | 2h |
| 1.4 Migrate GraphOrchestrator to use factory | P0 | 1.2 | 1h |

**Deliverable**: `KernelFactory::createKVCache()` working, GraphOrchestrator using it.

### Phase 2: Coherence Infrastructure (Week 1-2)

| Task | Priority | Dependencies | Effort |
|------|----------|--------------|--------|
| 2.1 Create StageCoherence.h/cpp | P0 | None | 2h |
| 2.2 Extend StageDumpInfo with ITensor* | P1 | None | 1h |
| 2.3 Add coherencePolicy() to IComputeStage | P1 | None | 0.5h |
| 2.4 Integrate coherence in DeviceGraphExecutor | P0 | 2.1, 2.2, 2.3 | 2h |
| 2.5 Write coherence unit tests | P0 | 2.4 | 2h |

**Deliverable**: Automatic coherence working, can be enabled via env var.

### Phase 3: Stage Cleanup (Week 2)

| Task | Priority | Dependencies | Effort |
|------|----------|--------------|--------|
| 3.1 Clean up AttentionComputeStage | P1 | 2.4 | 1h |
| 3.2 Clean up GEMMStage | P1 | 2.4 | 0.5h |
| 3.3 Clean up RMSNormStage | P1 | 2.4 | 0.5h |
| 3.4 Clean up remaining GPU stages | P1 | 2.4 | 2h |
| 3.5 Set CoherencePolicy::NONE for MPI stages | P1 | 2.3 | 0.5h |

**Deliverable**: All stages using automatic coherence, manual calls removed.

### Phase 4: Kernel Interface Refactoring (Week 2-3)

| Task | Priority | Dependencies | Effort |
|------|----------|--------------|--------|
| 4.1 Create ITensorFusedQKVGemm interface | P1 | None | 2h |
| 4.2 Create ITensorFusedAttentionWo interface | P1 | None | 2h |
| 4.3 Add factory methods to KernelFactory | P1 | 4.1, 4.2 | 3h |
| 4.4 Refactor FusedQKVGEMMStage | P1 | 4.3 | 2h |
| 4.5 Refactor FusedAttentionWoStage | P1 | 4.3 | 2h |
| 4.6 Refactor FusedGateUpGEMMStage | P2 | 4.3 | 1h |
| 4.7 Refactor EmbeddingStage | P2 | 4.3 | 1h |

**Deliverable**: All stages using KernelFactory, no direct kernel instantiation.

### Phase 5: Integration Testing (Week 3)

| Task | Priority | Dependencies | Effort |
|------|----------|--------------|--------|
| 5.1 Run full unit test suite | P0 | All | 1h |
| 5.2 Run integration tests | P0 | All | 2h |
| 5.3 Run E2E parity tests | P0 | All | 2h |
| 5.4 Performance benchmarks | P1 | All | 2h |
| 5.5 Documentation updates | P1 | All | 2h |

---

## Risk Assessment

### Low Risk

| Risk | Mitigation |
|------|------------|
| KVCache factory breaks tests | Existing tests verify behavior, factory is additive |
| Coherence overhead | O(1) checks, negligible vs. kernel execution |

### Medium Risk

| Risk | Mitigation |
|------|------------|
| Coherence changes execution order | Enable via env var first, test thoroughly |
| Fused kernel interfaces miss edge cases | Study existing implementations carefully |

### High Risk

| Risk | Mitigation |
|------|------------|
| Breaking multi-GPU inference | Phase 1 testing with 2-GPU config before merge |
| Performance regression | Benchmark before/after each phase |

---

## Testing Strategy

### Unit Tests

```cpp
// Test__KernelFactory_KVCache.cpp
TEST(Test__KernelFactory_KVCache, CreateCPUKVCache_FP32);
TEST(Test__KernelFactory_KVCache, CreateCPUKVCache_BF16);
TEST(Test__KernelFactory_KVCache, CreateShardedCPUKVCache);
TEST(Test__KernelFactory_KVCache, CreateCUDAKVCache_FP16);
TEST(Test__KernelFactory_KVCache, InvalidConfig_NoMPIContext);

// Test__StageCoherence.cpp
TEST(Test__StageCoherence, CohereInputs_CPUToCUDA);
TEST(Test__StageCoherence, CohereInputs_CUDAToCPU);
TEST(Test__StageCoherence, MarkOutputsDirty_CUDA);
TEST(Test__StageCoherence, Policy_NONE_SkipsCoherence);
```

### Integration Tests

```cpp
// Test__CUDAFullModelInference.cpp - existing tests should pass
TEST(Test__CUDAFullModelInference, SingleTokenPrediction_MatchesCPU);
TEST(Test__CUDAFullModelInference, MultiTokenGeneration);

// Test__MixedCPUGPUPipeline.cpp - new
TEST(Test__MixedCPUGPUPipeline, GPUAttention_CPUFFNorm);
TEST(Test__MixedCPUGPUPipeline, AutomaticCoherence);
```

### Performance Benchmarks

```bash
# Before/after comparison
LLAMINAR_PROFILE_KERNELS=1 ./build_v2_release/llaminar2 --benchmark \
    -m models/qwen2.5-0.5b-instruct-q4_0.gguf -n 128

# Multi-GPU with new KVCache
./build_v2_release/llaminar2 --mpi-procs 2 --benchmark \
    -m models/qwen2.5-7b-instruct-q4_0.gguf -n 128
```

---

## Summary

| Change | Benefit | Effort |
|--------|---------|--------|
| KVCache in KernelFactory | Device-aware cache creation, GPU decode support | 7 hours |
| Automatic Coherence | ~200 lines boilerplate removed, fewer bugs | 13 hours |
| Stages Using Interfaces | Single code path for all devices | 16 hours |

**Total Estimated Effort**: ~36 hours (1 developer-week)

**Key Outcomes**:
1. GraphOrchestrator creates CUDA KVCache automatically for GPU inference
2. Stages no longer manage device coherence manually
3. All stages use KernelFactory for kernel creation, enabling true device abstraction
