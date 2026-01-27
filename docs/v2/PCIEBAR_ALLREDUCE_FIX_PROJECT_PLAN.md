# PCIeBAR Allreduce Fix - Phased Project Plan

## Executive Summary

**Problem**: The LOCAL TP PCIeBAR parity test (`V2_Integration_Parity_Qwen2_LocalTP_PCIeBAR_vs_PyTorch`) fails during decode with cosine similarity 0.05-0.35 (threshold: 0.80). Root cause: `executePCIeBarAllreduce()` only uses ONE tensor (CUDA's), completely ignoring ROCm device's partial result. This computes `A + A = 2A` instead of `A + B`.

**Solution**: Implement BARBackedTensor for allreduce output buffers, enabling true zero-copy cross-vendor allreduce where ROCm kernels write directly to BAR region at full VRAM speed (~900 GB/s), and CUDA reads through PCIe (~2.65 GB/s) for reduction.

**Hardware Target**: CUDA:0 (RTX 3090) + ROCm:0 (MI50/MI60) heterogeneous LOCAL TP.

---

## Architecture Overview

### Current Broken Data Flow
```
┌─────────────────────┐          ┌─────────────────────┐
│ CUDA:0 (RTX 3090)   │          │ ROCm:0 (MI50)       │
│                     │          │                     │
│  FFN_DOWN kernel    │          │  FFN_DOWN kernel    │
│       ↓             │          │       ↓             │
│  output_A (VRAM)    │          │  output_B (HIP VRAM)│ ← IGNORED!
│       ↓             │          │       ↓             │
│  [tensor->data()]   │          │  [arrives at barrier]│
│       ↓             │          │       ↓             │
│  host staging ──────┼──────────┼── memcpy to BAR ────│← Only CUDA tensor copied!
│       ↓             │          │                     │
│  allreduce(A, A)    │          │  [waiting...]       │
│       ↓             │          │                     │
│  Result = 2A        │ ← WRONG! │                     │
└─────────────────────┘          └─────────────────────┘
```

### Target Zero-Copy Data Flow
```
┌─────────────────────┐          ┌─────────────────────┐
│ CUDA:0 (RTX 3090)   │          │ ROCm:0 (MI50)       │
│                     │          │                     │
│  FFN_DOWN kernel    │          │  FFN_DOWN kernel    │
│       ↓             │          │       ↓             │
│  output_A (BAR-     │          │  output_B (BAR-     │← ROCm writes to BAR
│  backed CUDA view)  │          │  backed local view) │  at full ~900 GB/s!
│       │             │          │       │             │
│       ↓             │          │       ↓             │
│  [at barrier]       │  PCIe    │  [at barrier]       │
│       ↓             │  ~2.65   │       ↓             │
│  allreduce:         │  GB/s    │  [passive wait]     │
│   read A from CUDA  │←─────────┤                     │
│   read B from BAR   │──────────→                     │
│   reduce: A + B     │          │                     │
│   write to CUDA out │          │                     │
│       ↓             │          │       ↓             │
│  Result = A + B ✓   │          │  [notified done]    │
└─────────────────────┘          └─────────────────────┘
```

---

## Phase 1: TensorFactory BAR-Backed Tensor Support

**Goal**: Extend existing `TensorFactory` with BAR-backed tensor creation for row-parallel output buffers.

### 1.1 Deliverables

| File | Change |
|------|--------|
| `src/v2/tensors/TensorFactory.h` | Add `createFP32BARBacked()` method and P2P context |
| `src/v2/tensors/TensorFactory.cpp` | Implement BAR-backed allocation using DirectP2P |

### 1.2 TensorFactory Extension

```cpp
class TensorFactory {
public:
    // ... existing methods ...
    
    /**
     * @brief Set DirectP2P context for BAR-backed tensor allocation
     * 
     * Must be called before createFP32BARBacked() can be used.
     * Typically called during LOCAL TP initialization when PCIeBAR backend is selected.
     * 
     * @param p2p DirectP2P context with mapped BAR regions
     */
    void setDirectP2P(std::shared_ptr<DirectP2P> p2p);
    
    /**
     * @brief Check if BAR-backed tensor allocation is available
     * @return true if DirectP2P context is set and BAR regions are mapped
     */
    bool canCreateBARBacked() const;
    
    /**
     * @brief Create FP32 tensor backed by PCIe BAR memory
     * 
     * Creates a tensor in ROCm device's BAR-exposed VRAM region.
     * - ROCm device writes at full VRAM bandwidth (~900 GB/s)
     * - CUDA device reads through PCIe BAR (~2.65 GB/s)
     * 
     * Use for row-parallel output buffers (FFN_DOWN, attention Wo) in
     * heterogeneous LOCAL TP with PCIeBAR backend.
     * 
     * @param shape Tensor dimensions
     * @param rocm_device ROCm device that owns the BAR memory (writes locally)
     * @param cuda_device CUDA device that will read through PCIe
     * @return FP32 tensor with dual pointers (rocm_data_ptr, cuda_data_ptr)
     * @throws std::runtime_error if DirectP2P not set or BAR not available
     */
    std::unique_ptr<FP32Tensor> createFP32BARBacked(
        const std::vector<size_t>& shape,
        DeviceId rocm_device,
        DeviceId cuda_device);

private:
    // ... existing members ...
    std::shared_ptr<DirectP2P> direct_p2p_;  // For BAR-backed allocation
};
```

### 1.3 Unit Tests

| Test | Description |
|------|-------------|
| `Test__TensorFactory_SetDirectP2P` | Factory accepts P2P context |
| `Test__TensorFactory_CanCreateBARBacked_True` | Returns true when P2P set |
| `Test__TensorFactory_CanCreateBARBacked_False` | Returns false without P2P |
| `Test__TensorFactory_CreateFP32BARBacked` | Creates valid BAR-backed FP32 tensor |
| `Test__TensorFactory_CreateFP32BARBacked_Shape` | Shape matches requested dimensions |
| `Test__TensorFactory_CreateFP32BARBacked_DeviceAffinity` | Tensor reports correct ROCm device affinity |
| `Test__TensorFactory_CreateFP32BARBacked_DualPointers` | Both `rocm_data_ptr()` and `cuda_data_ptr()` are valid |
| `Test__TensorFactory_CreateFP32BARBacked_ThrowsWithoutP2P` | Throws if P2P not set |

### 1.4 Success Criteria

- [ ] `setDirectP2P()` stores P2P context
- [ ] `canCreateBARBacked()` correctly reports availability
- [ ] `createFP32BARBacked()` creates tensors with correct shape
- [ ] `is_bar_backed()` returns true for created tensors
- [ ] `rocm_data_ptr()` returns valid HIP pointer
- [ ] `cuda_data_ptr()` returns valid CUDA pointer to same physical memory
- [ ] Throws appropriate error if P2P not configured
- [ ] All unit tests pass

---

## Phase 2: LocalTPContext BAR Tensor Tracking

**Goal**: Extend `LocalTPContext` to track BAR-backed output tensors for zero-copy allreduce.

**Rationale**: `LocalTPContext` is the right place for this because:
- It already manages LOCAL TP collectives with PCIeBAR backend
- It has the barrier synchronization where BAR tensors are used
- It knows the device list and weights for proportional allocation
- Separate from `CollectiveContext` which handles GLOBAL/MPI-level operations

### 2.1 Deliverables

| File | Change |
|------|--------|
| `src/v2/collective/LocalTPContext.h` | Add `bar_output_tensors_` map, registration API |
| `src/v2/collective/LocalTPContext.cpp` | Implement registration and lookup |

### 2.2 Interface Changes

```cpp
class LocalTPContext {
public:
    // Register a BAR-backed tensor for a specific stage's output
    // Called during graph construction for row-parallel stages
    void registerBARBackedOutput(
        const std::string& stage_name,
        DeviceId device,
        FP32Tensor* tensor);
    
    // Get all BAR-backed tensors for a stage (one per device)
    std::vector<FP32Tensor*> getBARBackedOutputs(
        const std::string& stage_name) const;
    
    // Check if stage has BAR-backed outputs registered
    bool hasBARBackedOutputs(const std::string& stage_name) const;
    
private:
    // Map: stage_name -> (device_id -> BAR-backed tensor)
    std::unordered_map<std::string, 
        std::unordered_map<DeviceId, FP32Tensor*>> bar_output_tensors_;
};
```

### 2.3 Unit Tests

| Test | Description |
|------|-------------|
| `Test__LocalTPContext_RegisterBAROutput` | Can register BAR tensor for stage |
| `Test__LocalTPContext_GetBAROutputs_MultiDevice` | Returns tensors for all devices |
| `Test__LocalTPContext_HasBAROutputs_True` | Returns true when registered |
| `Test__LocalTPContext_HasBAROutputs_False` | Returns false when not registered |
| `Test__LocalTPContext_RegisterMultipleStages` | Tracks tensors per-stage correctly |

### 2.4 Success Criteria

- [ ] Can register BAR-backed tensors per stage per device
- [ ] Can retrieve all tensors for a stage
- [ ] Correctly reports whether stage has BAR outputs
- [ ] All unit tests pass

---

## Phase 3: Graph Buffer Allocation with BAR Support

**Goal**: Modify graph construction to allocate BAR-backed tensors for row-parallel output buffers when LOCAL TP uses PCIeBAR backend.

### 3.1 Deliverables

| File | Change |
|------|--------|
| `src/v2/pipelines/qwen/Qwen2BufferSpec.h` | Add `AllocationStrategy` enum |
| `src/v2/pipelines/qwen/Qwen2BufferSpec.cpp` | Add `requiresBARBacked()` method |
| `src/v2/pipelines/qwen/Qwen2Graph.cpp` | Use BAR-backed allocation for row-parallel outputs |
| `src/v2/execution/BufferPool.h` | Support BAR-backed tensor creation |
| `src/v2/execution/BufferPool.cpp` | Implement BAR-backed allocation path |

### 3.2 Buffer Spec Extension

```cpp
enum class AllocationStrategy {
    STANDARD,       // Regular device memory
    BAR_BACKED,     // PCIe BAR-backed for cross-vendor allreduce
    PINNED_HOST     // Pinned host memory (future)
};

class Qwen2BufferSpec {
public:
    // Check if buffer needs BAR-backed allocation
    // Returns true for: ffn_down_output, attention_wo_output when LOCAL TP + PCIeBAR
    static bool requiresBARBacked(
        const std::string& buffer_name,
        const OrchestrationConfig& config);
};
```

### 3.3 Integration Points

```cpp
// In Qwen2Graph::allocateLayerBuffers()
if (Qwen2BufferSpec::requiresBARBacked("ffn_down_output", config_) &&
    tensor_factory_->canCreateBARBacked()) {
    
    auto ffn_down_output = tensor_factory_->createFP32BARBacked(
        {seq_len, hidden_dim},
        rocm_device_,
        cuda_device_);
    
    // Register with LocalTPContext for allreduce
    local_tp_context_->registerBARBackedOutput(
        "layer" + std::to_string(layer) + "_ffn_down_allreduce",
        rocm_device_,
        ffn_down_output.get());
}
```

### 3.4 Unit Tests

| Test | Description |
|------|-------------|
| `Test__Qwen2BufferSpec_RequiresBARBacked_FFNDown` | FFN down output requires BAR when LOCAL TP + PCIeBAR |
| `Test__Qwen2BufferSpec_RequiresBARBacked_AttnWO` | Attention WO output requires BAR when LOCAL TP + PCIeBAR |
| `Test__Qwen2BufferSpec_RequiresBARBacked_False_NoTP` | Returns false without TP |
| `Test__Qwen2BufferSpec_RequiresBARBacked_False_NCCL` | Returns false with NCCL backend |
| `Test__BufferPool_CreateBARBacked` | BufferPool creates BAR-backed when requested |

### 3.5 Success Criteria

- [ ] `requiresBARBacked()` correctly identifies row-parallel outputs
- [ ] Graph allocates BAR-backed tensors for identified buffers
- [ ] BAR-backed tensors registered with LocalTPContext
- [ ] All unit tests pass

---

## Phase 4: Zero-Copy Allreduce Implementation

**Goal**: Rewrite `executePCIeBarAllreduce()` to use BAR-backed tensors for true zero-copy reduction.

### 4.1 Deliverables

| File | Change |
|------|--------|
| `src/v2/collective/LocalTPContext.cpp` | Rewrite `executePCIeBarAllreduce()` |
| `src/v2/collective/backends/PCIeBARBackend.cpp` | Add `allreduceZeroCopy()` method |

### 4.2 Zero-Copy Allreduce Flow

```cpp
void LocalTPContext::executePCIeBarAllreduce(
    const std::string& stage_name,
    TensorBase* output) {
    
    // Check if we have BAR-backed tensors for this stage
    if (hasBARBackedOutputs(stage_name)) {
        // Zero-copy path: tensors already in correct memory locations
        auto bar_tensors = getBARBackedOutputs(stage_name);
        
        // CUDA device reads from all BAR tensors + its own local tensor
        // Reduces in CUDA VRAM, writes to output
        pcie_bar_backend_->allreduceZeroCopy(
            bar_tensors,      // BAR-backed inputs from ROCm devices
            output,           // CUDA output tensor
            cuda_stream_);
    } else {
        // Fallback: host-staged path (existing code, for non-BAR tensors)
        executePCIeBarAllreduceLegacy(stage_name, output);
    }
}
```

### 4.3 Backend Zero-Copy Implementation

```cpp
void PCIeBARBackend::allreduceZeroCopy(
    const std::vector<FP32Tensor*>& bar_inputs,  // From ROCm devices
    TensorBase* cuda_output,                      // CUDA output
    cudaStream_t stream) {
    
    const size_t n = cuda_output->numel();
    float* out_ptr = cuda_output->mutable_data_ptr<float>();
    
    // Step 1: Copy CUDA's own partial result to output (device-local, fast)
    // The CUDA device's result is already in cuda_output from kernel execution
    
    // Step 2: For each ROCm device's BAR-backed tensor, add to output
    for (auto* bar_tensor : bar_inputs) {
        // Get CUDA-accessible pointer to ROCm's BAR region
        float* bar_ptr = bar_tensor->cuda_data_ptr();
        
        // Launch CUDA kernel to add BAR data to output
        // This reads through PCIe at ~2.65 GB/s
        cuda_vector_add_inplace(out_ptr, bar_ptr, n, stream);
    }
    
    cudaStreamSynchronize(stream);
}
```

### 4.4 Unit Tests

| Test | Description |
|------|-------------|
| `Test__LocalTPContext_ExecutePCIeBarAllreduce_ZeroCopy` | Uses zero-copy path when BAR tensors registered |
| `Test__LocalTPContext_ExecutePCIeBarAllreduce_Legacy` | Falls back to legacy when no BAR tensors |
| `Test__PCIeBARBackend_AllreduceZeroCopy_Correctness` | Sum is correct (A + B) |
| `Test__PCIeBARBackend_AllreduceZeroCopy_MultiDevice` | Works with 1 CUDA + N ROCm |

### 4.5 Success Criteria

- [ ] Zero-copy path selected when BAR tensors registered
- [ ] Allreduce computes correct sum (A + B, not 2A)
- [ ] No host memory staging in zero-copy path
- [ ] All unit tests pass

---

## Phase 5: Integration Testing

**Goal**: Verify end-to-end correctness with parity test.

### 5.1 Deliverables

| File | Change |
|------|--------|
| `tests/v2/integration/parity/Test__Qwen2_LocalTP_PCIeBAR_vs_PyTorch.cpp` | Add diagnostic logging |

### 5.2 Integration Tests

| Test | Description | Success Criteria |
|------|-------------|------------------|
| `V2_Integration_Parity_Qwen2_LocalTP_PCIeBAR_vs_PyTorch` | Full parity test | Decode cosine ≥ 0.80 |
| `V2_Integration_LocalTP_PCIeBAR_AllreduceCorrectness` | Isolated allreduce check | CUDA + ROCm results sum correctly |
| `V2_Integration_LocalTP_PCIeBAR_MultiLayer` | All 24 layers pass | Per-layer cosine ≥ 0.95 |

### 5.3 Diagnostic Additions

```cpp
// Add to parity test for debugging
LOG_INFO("[PCIeBAR Parity] Layer " << layer << " allreduce:");
LOG_INFO("  CUDA partial sum: " << compute_sum(cuda_tensor));
LOG_INFO("  ROCm partial sum: " << compute_sum(rocm_bar_tensor));
LOG_INFO("  Combined result:  " << compute_sum(output));
LOG_INFO("  Expected:         " << compute_sum(cuda_tensor) + compute_sum(rocm_bar_tensor));
```

### 5.4 Success Criteria

- [ ] Parity test passes with decode cosine ≥ 0.80
- [ ] No "all zeros" tensor warnings
- [ ] Correct summation verified in logs
- [ ] Performance within 10% of NCCL baseline

---

## Phase 6: Performance Validation

**Goal**: Ensure BAR-backed allocation doesn't regress performance.

### 6.1 Benchmarks

| Benchmark | Metric | Target |
|-----------|--------|--------|
| ROCm kernel write throughput | GB/s | ≥ 800 GB/s (should match non-BAR) |
| Allreduce latency (896 elements) | μs | < 50 μs |
| Allreduce latency (4864 elements) | μs | < 100 μs |
| End-to-end decode tok/s | tok/s | ≥ 90% of NCCL baseline |

### 6.2 Success Criteria

- [ ] ROCm write throughput ≥ 800 GB/s
- [ ] Allreduce latency within 2x of device-local
- [ ] End-to-end decode within 10% of NCCL baseline

---

## Implementation Order

```
Phase 1: BARBackedTensor Factory (Foundation)
    ↓
Phase 2: LocalTPContext Management (Tracking)
    ↓
Phase 3: Graph Buffer Allocation (Integration)
    ↓
Phase 4: Zero-Copy Allreduce (Core Fix)
    ↓
Phase 5: Integration Testing (Validation)
    ↓
Phase 6: Performance Validation (Optimization)
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| ROCm BAR allocation fails | Fall back to HIP D2H copy path |
| PCIe bandwidth insufficient | Use pipelining (existing code) |
| Memory pressure from BAR allocation | Limit BAR usage to row-parallel outputs only |
| Multi-ROCm device support | Design for N ROCm devices from start |

---

## Files to Create/Modify Summary

### New Files
- `tests/v2/unit/Test__TensorFactory_BARBacked.cpp` - BAR-backed creation unit tests
- `tests/v2/unit/Test__LocalTPContext_BARManagement.cpp` - Management unit tests
- `tests/v2/integration/Test__PCIeBAR_AllreduceZeroCopy.cpp` - Zero-copy integration test

### Modified Files
- `src/v2/tensors/TensorFactory.h` - Add `setDirectP2P()`, `canCreateBARBacked()`, `createFP32BARBacked()`
- `src/v2/tensors/TensorFactory.cpp` - Implement BAR-backed allocation
- `src/v2/collective/LocalTPContext.h` - Add BAR tensor tracking
- `src/v2/collective/LocalTPContext.cpp` - Implement tracking + zero-copy allreduce
- `src/v2/collective/backends/PCIeBARBackend.cpp` - Add `allreduceZeroCopy()`
- `src/v2/pipelines/qwen/Qwen2BufferSpec.h` - Add allocation strategy
- `src/v2/pipelines/qwen/Qwen2BufferSpec.cpp` - Add `requiresBARBacked()`
- `src/v2/pipelines/qwen/Qwen2Graph.cpp` - Use BAR allocation for row-parallel
- `src/v2/execution/BufferPool.h` - Support BAR allocation
- `src/v2/execution/BufferPool.cpp` - Implement BAR allocation path

---

## Subagent Dispatch Plan

| Phase | Subagent Prompt Summary |
|-------|------------------------|
| Phase 1 | "Extend TensorFactory with BAR-backed tensor support. Add setDirectP2P(), canCreateBARBacked(), and createFP32BARBacked() methods. The new method creates FP32 tensors in ROCm's BAR region using DirectP2P. Include 8 unit tests covering P2P setup, availability checks, creation, shape, device affinity, dual pointers, and error handling." |
| Phase 2 | "Add BAR tensor tracking to LocalTPContext. Implement registerBARBackedOutput(), getBARBackedOutputs(), hasBARBackedOutputs(). Include 5 unit tests covering registration, multi-device retrieval, existence checks, and multi-stage tracking." |
| Phase 3 | "Modify graph buffer allocation to use BAR-backed tensors for row-parallel outputs (ffn_down, attention_wo) when LOCAL TP uses PCIeBAR. Add AllocationStrategy enum to BufferSpec, implement requiresBARBacked(). Include 5 unit tests." |
| Phase 4 | "Rewrite executePCIeBarAllreduce() for zero-copy using BAR tensors. Add allreduceZeroCopy() to PCIeBARBackend. CUDA reads from BAR, reduces, writes to output. Include 4 unit tests verifying correctness and multi-device support." |
| Phase 5 | "Run integration tests and add diagnostic logging. Verify V2_Integration_Parity_Qwen2_LocalTP_PCIeBAR_vs_PyTorch passes with decode cosine ≥ 0.80. Add per-layer sum verification logging." |
| Phase 6 | "Benchmark BAR-backed tensor performance. Verify ROCm write throughput ≥ 800 GB/s, allreduce latency < 100μs for FFN sizes, end-to-end decode within 10% of NCCL baseline." |

---

## Acceptance Criteria (Final)

1. ✅ `V2_Integration_Parity_Qwen2_LocalTP_PCIeBAR_vs_PyTorch` passes
2. ✅ Decode cosine similarity ≥ 0.80 (was: 0.05-0.35)
3. ✅ No "all zeros" tensor warnings during allreduce
4. ✅ Correct `A + B` summation (not `2A`)
5. ✅ ROCm write throughput unaffected (≥ 800 GB/s)
6. ✅ All new unit tests pass
7. ✅ All existing tests continue to pass
