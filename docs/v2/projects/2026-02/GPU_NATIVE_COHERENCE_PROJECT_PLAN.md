# GPU-Native Tensor Coherence: Project Plan

**Status**: Implementation  
**Author**: Claude + David  
**Created**: 2026-02-01  
**Design Doc**: [GPU_NATIVE_TENSOR_COHERENCE_DESIGN.md](GPU_NATIVE_TENSOR_COHERENCE_DESIGN.md)

## Executive Summary

Extend the tensor coherence system to support **direct GPU-to-GPU transfers** without host staging, and **GPU-native tensors** that have no host backing. This enables:

- **2× faster PP activation transfers** (1 P2P/BAR vs 2 PCIe transfers)
- **GPU-native activation buffers** (no wasted host memory)
- **Future weight streaming** (move weights between GPUs at runtime)

## Project Phases

| Phase | Description | Estimated Effort | Dependencies |
|-------|-------------|------------------|--------------|
| **Phase 1** | Add `authoritative_device_` tracking | 1 day | None |
| **Phase 2** | Implement `transferTo()` method | 1 day | Phase 1, copy() API |
| **Phase 3** | Implement `copyTo()` for dual residency | 0.5 day | Phase 2 |
| **Phase 4** | GPU-native tensor creation (`createOnDevice`) | 1 day | Phase 1 |
| **Phase 5** | Update LocalPPContext to use new API | 0.5 day | Phase 2 |
| **Phase 6** | Update existing integration tests | 0.5 day | All phases |
| **Phase 7** | Performance benchmarks | 0.5 day | Phase 5 |

**Total**: ~5 days

---

## Phase 1: Add `authoritative_device_` Tracking

### Goal
Add multi-device awareness to TensorBase without breaking existing behavior.

### Changes

**File: `src/v2/tensors/TensorClasses.h`**

Add new protected member:
```cpp
// In TensorBase protected section:
std::optional<DeviceId> authoritative_device_;  // Which device has current data (nullopt = host)
```

Add new public methods:
```cpp
/**
 * @brief Get the device that currently has authoritative data
 * @return DeviceId if a GPU is authoritative, nullopt if host is authoritative
 */
std::optional<DeviceId> getAuthoritativeDevice() const { return authoritative_device_; }

/**
 * @brief Check if host memory is authoritative (has current data)
 */
bool isHostAuthoritative() const { return !authoritative_device_.has_value(); }

/**
 * @brief Check if a specific device is authoritative
 */
bool isDeviceAuthoritative(DeviceId device) const {
    return authoritative_device_.has_value() && *authoritative_device_ == device;
}
```

Update existing methods to maintain `authoritative_device_`:
- `mark_device_dirty()`: Set `authoritative_device_ = gpu_device_`
- `ensureOnHost()`: Clear `authoritative_device_` after sync
- `ensureOnDevice()`: If host is authoritative, just upload (no change to authoritative)

### Unit Tests

**File: `tests/v2/unit/tensors/Test__TensorBase_AuthoritativeDevice.cpp`**

| Test | Description |
|------|-------------|
| `InitialState_HostAuthoritative` | New tensor has `authoritative_device_ = nullopt` |
| `AfterMarkDeviceDirty_DeviceAuthoritative` | `mark_device_dirty()` sets authoritative |
| `AfterEnsureOnHost_HostAuthoritative` | `ensureOnHost()` clears authoritative |
| `IsDeviceAuthoritative_Correct` | Query methods return correct values |
| `EnsureOnDevice_DoesNotChangeAuthoritative` | Upload doesn't change who's authoritative |

### Acceptance Criteria
- [ ] All existing tests pass (backward compatibility)
- [ ] New unit tests pass
- [ ] `authoritative_device_` correctly tracks state

---

## Phase 2: Implement `transferTo()` Method

### Goal
Enable direct GPU-to-GPU transfer using `ICollectiveBackend::copy()`.

### Changes

**File: `src/v2/tensors/TensorClasses.h`**

Add public method declaration:
```cpp
/**
 * @brief Transfer tensor data directly to another GPU device
 * 
 * Uses ICollectiveBackend::copy() for direct P2P/BAR transfer.
 * Does NOT go through host staging.
 * 
 * @param dst_device Target GPU device
 * @return true on success, false if no backend supports this transfer
 * 
 * @pre Tensor must be on a GPU (getAuthoritativeDevice().has_value())
 * @post getAuthoritativeDevice() == dst_device
 * @post Source device buffer is now stale
 * 
 * @note Fails fast if no backend supports the transfer path.
 *       Does NOT silently fall back to host staging.
 */
bool transferTo(DeviceId dst_device);
```

**File: `src/v2/tensors/TensorClasses.cpp`** (or inline in header)

```cpp
bool TensorBase::transferTo(DeviceId dst_device) {
    // 1. Validate preconditions
    if (!authoritative_device_.has_value()) {
        LOG_ERROR("transferTo: Tensor not on GPU. Use ensureOnDevice() first.");
        return false;
    }
    
    DeviceId src_device = *authoritative_device_;
    if (src_device == dst_device) {
        return true;  // Already there
    }
    
    // 2. Get backend that supports this transfer
    auto* router = GlobalBackendRouter::get();
    if (!router) {
        LOG_ERROR("transferTo: GlobalBackendRouter not initialized");
        return false;
    }
    
    ICollectiveBackend* backend = router->getBackendForCopy(src_device, dst_device);
    if (!backend || !backend->supportsCopy(src_device, dst_device)) {
        LOG_ERROR("transferTo: No backend supports " << src_device << " → " << dst_device);
        return false;  // Fail fast
    }
    
    // 3. Ensure destination buffer exists
    void* dst_ptr = getOrAllocateDeviceBuffer(dst_device);
    if (!dst_ptr) {
        LOG_ERROR("transferTo: Failed to allocate buffer on " << dst_device);
        return false;
    }
    
    // 4. Perform direct copy
    if (!backend->copy(dst_ptr, dst_device, gpu_data_ptr_, src_device, size_bytes())) {
        LOG_ERROR("transferTo: Backend copy failed");
        return false;
    }
    
    // 5. Update coherence state
    authoritative_device_ = dst_device;
    gpu_device_ = dst_device;
    gpu_data_ptr_ = dst_ptr;
    host_valid_ = false;  // Host is now stale
    
    // 6. Clear completion event (cross-vendor safety)
    if (src_device.type() != dst_device.type()) {
        clearCompletionEvent();
    }
    
    return true;
}
```

### Helper Method

Add `getOrAllocateDeviceBuffer()`:
```cpp
protected:
    /**
     * @brief Get existing buffer or allocate new one for device
     * 
     * Used by transferTo() and copyTo() to ensure destination exists.
     * Stores secondary buffers in secondary_device_buffers_ map.
     */
    void* getOrAllocateDeviceBuffer(DeviceId device);
    
    // Add to protected members:
    std::unordered_map<DeviceId, void*, DeviceIdHash> secondary_device_buffers_;
```

### Unit Tests

**File: `tests/v2/unit/tensors/Test__TensorBase_TransferTo.cpp`**

| Test | Description |
|------|-------------|
| `TransferTo_SameDevice_NoOp` | Transfer to same device returns true immediately |
| `TransferTo_NotOnGPU_Fails` | Fails if tensor not on any GPU |
| `TransferTo_NoBackend_FailsFast` | Fails if no backend supports path |
| `TransferTo_UpdatesAuthoritative` | After transfer, dst is authoritative |
| `TransferTo_InvalidatesHost` | After transfer, host_valid_ = false |

### Integration Tests

**File: `tests/v2/integration/tensors/Test__TensorBase_TransferTo_Integration.cpp`**

| Test | Description |
|------|-------------|
| `TransferTo_CUDA_to_CUDA` | Same-vendor P2P transfer |
| `TransferTo_ROCm_to_ROCm` | Same-vendor P2P transfer |
| `TransferTo_CUDA_to_ROCm` | Cross-vendor via PCIeBAR |
| `TransferTo_ROCm_to_CUDA` | Cross-vendor reverse |
| `TransferTo_DataIntegrity` | Verify data survives transfer |
| `TransferTo_MultiHop` | A→B→C transfers work |

### Acceptance Criteria
- [ ] `transferTo()` uses `ICollectiveBackend::copy()`
- [ ] Fails fast when no backend available (no host fallback)
- [ ] Correctly updates `authoritative_device_`
- [ ] All unit and integration tests pass

---

## Phase 3: Implement `copyTo()` for Dual Residency

### Goal
Enable copying to another device while keeping source valid.

### Changes

**File: `src/v2/tensors/TensorClasses.h`**

```cpp
/**
 * @brief Copy tensor data to another GPU, keeping both devices valid
 * 
 * Unlike transferTo(), this keeps the source device buffer valid.
 * Useful for read-only sharing or when source needs to continue computing.
 * 
 * @param dst_device Target GPU device
 * @return true on success
 * 
 * @post Both src and dst devices have valid data
 * @post authoritative_device_ unchanged (source still authoritative)
 */
bool copyTo(DeviceId dst_device);
```

Implementation similar to `transferTo()` but:
- Does NOT update `authoritative_device_`
- Does NOT invalidate source
- Marks destination buffer as valid (in `secondary_device_buffers_` tracking)

### Unit Tests

| Test | Description |
|------|-------------|
| `CopyTo_KeepsSourceValid` | Source device remains valid after copy |
| `CopyTo_AuthoritativeUnchanged` | `authoritative_device_` not modified |
| `CopyTo_BothAccessible` | Both buffers can be read |

### Acceptance Criteria
- [ ] `copyTo()` preserves source validity
- [ ] Dual residency works correctly
- [ ] Tests pass

---

## Phase 4: GPU-Native Tensor Creation

### Goal
Allow creating tensors directly on GPU without host backing.

### Changes

**File: `src/v2/tensors/TensorClasses.h`**

Add to TensorBase:
```cpp
/**
 * @brief Check if this tensor has host backing memory
 * 
 * GPU-native tensors created via createOnDevice() have no host backing.
 * Calling ensureOnHost() on such tensors will allocate host memory.
 */
bool hasHostBacking() const;
```

Add to FP32Tensor (and other activation tensor types):
```cpp
/**
 * @brief Create tensor directly on GPU device (no host backing)
 * 
 * This is optimal for activation buffers that never need to leave GPU.
 * 
 * @param rows Number of rows
 * @param cols Number of columns
 * @param device GPU device to create on
 * @return Tensor with GPU buffer only, or nullptr on failure
 * 
 * @note ensureOnHost() will allocate host buffer on-demand
 */
static std::unique_ptr<FP32Tensor> createOnDevice(size_t rows, size_t cols, DeviceId device);
```

### Modified `ensureOnHost()` Behavior

For GPU-native tensors, `ensureOnHost()` should:
1. Allocate host buffer if needed (lazy allocation)
2. Sync GPU → Host
3. Set `host_valid_ = true`

This allows debugging/inspection of GPU-native tensors without requiring upfront host allocation.

### Unit Tests

**File: `tests/v2/unit/tensors/Test__TensorBase_CreateOnDevice.cpp`**

| Test | Description |
|------|-------------|
| `CreateOnDevice_NoHostBacking` | `hasHostBacking()` returns false initially |
| `CreateOnDevice_OnGPU` | Tensor is on specified device |
| `CreateOnDevice_AuthoritativeDevice` | GPU is authoritative |
| `CreateOnDevice_EnsureOnHostAllocates` | `ensureOnHost()` creates host buffer |
| `CreateOnDevice_DataIntegrity` | Write to GPU, read from host works |

### Activation Tensor Types to Update

- [x] FP32Tensor
- [x] BF16Tensor  
- [x] FP16Tensor
- [x] Q8_1Tensor (activation quantization format)

### Acceptance Criteria
- [ ] `createOnDevice()` factory works for all activation types
- [ ] No host memory allocated until needed
- [ ] `ensureOnHost()` correctly lazy-allocates
- [ ] Tests pass

---

## Phase 5: Update LocalPPContext

### Goal
Replace `directMemcpy()` with `transferTo()`.

### Changes

**File: `src/v2/collective/LocalPPContext.cpp`**

Replace `transfer()` implementation:
```cpp
bool LocalPPContext::transfer(TensorBase* activations, int stage_from, int stage_to) {
    if (!activations) {
        LOG_ERROR("LocalPPContext::transfer: null activations tensor");
        return false;
    }
    
    DeviceId src_device = getDeviceForStage(stage_from);
    DeviceId dst_device = getDeviceForStage(stage_to);
    
    if (src_device == dst_device) {
        LOG_DEBUG("LocalPPContext::transfer: same device, no-op");
        return true;
    }
    
    // Ensure tensor is on source device first
    if (!activations->isDeviceAuthoritative(src_device)) {
        if (!activations->ensureOnDevice(src_device)) {
            LOG_ERROR("LocalPPContext::transfer: Failed to ensure on source device");
            return false;
        }
        activations->mark_device_dirty();
    }
    
    // Use direct GPU-to-GPU transfer
    if (!activations->transferTo(dst_device)) {
        LOG_ERROR("LocalPPContext::transfer: Direct transfer failed "
                  << src_device << " → " << dst_device);
        return false;
    }
    
    LOG_DEBUG("LocalPPContext::transfer: Direct P2P/BAR transfer completed");
    return true;
}
```

### Remove `directMemcpy()`

Once `transferTo()` is working:
1. Remove `directMemcpy()` private method
2. Remove `staging_buffer_` member (no longer needed)
3. Remove `reserveStagingBuffer()` method

### Integration Tests

| Test | Description |
|------|-------------|
| `LocalPP_TransferTo_CrossVendor` | PP transfer uses new path |
| `LocalPP_NoHostStaging` | Verify no D2H + H2D traffic |

### Acceptance Criteria
- [ ] `LocalPPContext::transfer()` uses `transferTo()`
- [ ] No host staging buffer used
- [ ] All existing LocalPP tests pass
- [ ] Performance improvement measurable

---

## Phase 6: Update Existing Integration Tests

### Goal
Update tests in `Test__MultiGPU_RealModel.cpp` to use new API.

### Tests to Update

**`Q4_0_Weight_CrossVendor`**:
```cpp
// Before (host staging):
q4_tensor->ensureOnHost();
q4_tensor->ensureOnDevice(rocm);

// After (direct transfer):
q4_tensor->transferTo(rocm);
```

**`DirectCopy_API`**:
- Update to actually use `transferTo()` instead of just checking `supportsCopy()`
- Verify data integrity after direct transfer

### New Tests to Add

| Test | Description |
|------|-------------|
| `GPU_Native_Activation_PP` | PP with GPU-native activation buffer |
| `TransferTo_Q4_0_CrossVendor` | Direct quantized weight transfer |
| `TransferTo_FP32_MultiHop` | A→B→C→A round-trip |

### Acceptance Criteria
- [ ] All 8 GPU-to-GPU tests pass
- [ ] Tests use `transferTo()` where appropriate
- [ ] Test comments updated to reflect new API

---

## Phase 7: Performance Benchmarks

### Goal
Measure performance improvement from direct transfers.

### Benchmarks

**File: `tests/v2/performance/Test__DirectTransfer_Benchmark.cpp`**

| Benchmark | Description |
|-----------|-------------|
| `Benchmark_HostStaging_1MB` | Current path (baseline) |
| `Benchmark_DirectTransfer_1MB` | New `transferTo()` path |
| `Benchmark_DirectTransfer_CrossVendor_1MB` | CUDA↔ROCm via BAR |
| `Benchmark_DirectTransfer_Scaling` | 1KB, 10KB, 100KB, 1MB, 10MB |

### Expected Results

| Transfer Type | Current | Expected | Improvement |
|---------------|---------|----------|-------------|
| CUDA→CUDA P2P | ~400μs | ~200μs | 2× |
| ROCm→ROCm P2P | ~400μs | ~200μs | 2× |
| CUDA↔ROCm BAR | ~600μs | ~350μs | 1.7× |

### Acceptance Criteria
- [ ] Benchmarks show measurable improvement
- [ ] Results documented in changelog

---

## Implementation Order

```
Week 1:
├── Phase 1: authoritative_device_ tracking
│   ├── TensorBase changes
│   └── Unit tests
├── Phase 2: transferTo() implementation
│   ├── TensorBase::transferTo()
│   ├── getOrAllocateDeviceBuffer()
│   ├── Unit tests
│   └── Integration tests

Week 2:
├── Phase 3: copyTo() for dual residency
│   └── Unit tests
├── Phase 4: GPU-native creation
│   ├── FP32Tensor::createOnDevice()
│   ├── Other activation types
│   └── Unit tests
├── Phase 5: LocalPPContext update
│   ├── Replace directMemcpy()
│   └── Integration tests
├── Phase 6: Update existing tests
└── Phase 7: Benchmarks
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking existing code | Phase 1 is additive; run all tests at each phase |
| Memory leaks | RAII cleanup in destructor for `secondary_device_buffers_` |
| Complex state machine | Add state assertions, comprehensive logging |
| Backend not initialized | Fail-fast with clear error message |

---

## Success Criteria

- [ ] All existing tests pass (no regressions)
- [ ] New unit tests for each phase pass
- [ ] Integration tests demonstrate direct GPU→GPU transfer
- [ ] LocalPPContext uses new API
- [ ] Benchmarks show ≥1.5× improvement
- [ ] No silent host fallbacks (fail-fast philosophy maintained)

---

## Files to Create/Modify

### New Files
- `tests/v2/unit/tensors/Test__TensorBase_AuthoritativeDevice.cpp`
- `tests/v2/unit/tensors/Test__TensorBase_TransferTo.cpp`
- `tests/v2/unit/tensors/Test__TensorBase_CopyTo.cpp`
- `tests/v2/unit/tensors/Test__TensorBase_CreateOnDevice.cpp`
- `tests/v2/integration/tensors/Test__TensorBase_TransferTo_Integration.cpp`
- `tests/v2/performance/Test__DirectTransfer_Benchmark.cpp`

### Modified Files
- `src/v2/tensors/TensorClasses.h` - Add new methods and members
- `src/v2/tensors/TensorClasses.cpp` - Implement new methods
- `src/v2/collective/LocalPPContext.cpp` - Use transferTo()
- `tests/v2/integration/pipelines/Test__MultiGPU_RealModel.cpp` - Update tests
- `tests/v2/CMakeLists.txt` - Add new test targets

---

## Appendix: DeviceIdHash

Needed for `std::unordered_map<DeviceId, void*>`:

```cpp
// In DeviceId.h or separate header:
struct DeviceIdHash {
    std::size_t operator()(const DeviceId& id) const {
        return std::hash<int>()(static_cast<int>(id.type())) ^ 
               (std::hash<int>()(id.ordinal()) << 1);
    }
};
```
