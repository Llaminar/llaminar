# Project Plan: Collective Backend Copy API

**Author**: GitHub Copilot  
**Date**: February 1, 2026  
**Status**: Draft  

## 1. Executive Summary

This project adds `copy()` and `copyAsync()` methods to the `ICollectiveBackend` interface, enabling unified GPU-to-GPU tensor transfers without host staging. This provides:

1. **Immediate value**: Efficient PP (Pipeline Parallel) activation transfers
2. **Future value**: Foundation for dynamic weight streaming, memory pressure management, and heterogeneous compute

### Key Benefits

| Benefit | Description |
|---------|-------------|
| **Performance** | Eliminates unnecessary PCIe round-trips via host staging |
| **Unified API** | Single interface for all GPU-to-GPU transfers |
| **Backend flexibility** | Each backend uses optimal transfer mechanism |
| **Future-proof** | Supports weight offloading, memory balancing, etc. |

---

## 2. Design Goals

### 2.1 Primary Goals (This Project)

1. **Add `copy()` and `copyAsync()` to ICollectiveBackend** - clean interface for point-to-point transfers
2. **Implement in all relevant backends** - NCCL, RCCL, PCIeBAR, Host
3. **Integrate with LocalPPContext** - replace `directMemcpy()` with backend-based copy
4. **Add comprehensive tests** - same-vendor, cross-vendor, async, edge cases

### 2.2 Future Goals (Out of Scope but Designed For)

1. **Weight streaming integration** - IWeightStreamer uses copy() for H2D/D2D
2. **Dynamic memory balancing** - move weights between GPUs at runtime
3. **KV cache migration** - move cache entries between devices
4. **Heterogeneous scheduling** - CPU↔GPU weight movement

---

## 3. Interface Design

### 3.1 New Methods in ICollectiveBackend

```cpp
// In src/v2/collective/ICollectiveBackend.h

/**
 * @brief Copy data between devices (same or different)
 * 
 * Unlike send()/recv() which require matched rank pairs, copy() is a 
 * single-sided operation that works within a single process. Uses the
 * optimal transfer mechanism for the device pair:
 * - Same CUDA device: cudaMemcpy (no-op if same pointer)
 * - Different CUDA devices: cudaMemcpyPeerAsync with P2P if available
 * - Different ROCm devices: hipMemcpyPeerAsync with P2P if available  
 * - Cross-vendor (CUDA↔ROCm): PCIe BAR1 mapping or host staging
 * - Host↔Device: cudaMemcpy/hipMemcpy
 * 
 * @param dst_ptr Destination pointer (must be valid on dst_device)
 * @param dst_device Destination device
 * @param src_ptr Source pointer (must be valid on src_device)
 * @param src_device Source device
 * @param bytes Number of bytes to copy
 * @return true on success
 * 
 * @note Synchronous - blocks until copy completes
 * @note Thread-safe with respect to other copy() calls
 */
virtual bool copy(
    void* dst_ptr, DeviceId dst_device,
    const void* src_ptr, DeviceId src_device,
    size_t bytes) { return false; }

/**
 * @brief Async copy data between devices
 * 
 * Same semantics as copy() but returns immediately. Completion can be
 * tracked via the stream or by calling synchronize().
 * 
 * @param dst_ptr Destination pointer
 * @param dst_device Destination device
 * @param src_ptr Source pointer  
 * @param src_device Source device
 * @param bytes Number of bytes to copy
 * @param stream Device stream for ordering (nullptr for default stream)
 * @return true if copy was successfully enqueued
 * 
 * @note Caller must synchronize before reading dst_ptr
 */
virtual bool copyAsync(
    void* dst_ptr, DeviceId dst_device,
    const void* src_ptr, DeviceId src_device,
    size_t bytes, void* stream = nullptr) { return false; }

/**
 * @brief Check if this backend supports copy between given device pair
 * 
 * @param src_device Source device
 * @param dst_device Destination device
 * @return true if copy() will work for this device pair
 */
virtual bool supportsCopy(DeviceId src_device, DeviceId dst_device) const {
    (void)src_device; (void)dst_device;
    return false;
}
```

### 3.2 Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Raw pointers** | Tensors already manage their buffers; copy is low-level |
| **DeviceId parameters** | Explicit about where data lives; no guessing |
| **Sync + Async variants** | PP needs sync; prefetch needs async |
| **No TensorBase dependency** | Keeps collective layer decoupled from tensor layer |
| **Default returns false** | Backends opt-in; clear which support it |

### 3.3 Relationship to Existing APIs

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ICollectiveBackend                               │
├─────────────────────────────────────────────────────────────────────┤
│ Multi-Rank Collectives:              Point-to-Point (NEW):          │
│   allreduce()  - reduce across ranks   copy()      - single-process │
│   allgather()  - gather to all ranks   copyAsync() - async version  │
│   broadcast()  - one-to-all                                         │
│   send()/recv()- paired rank exchange  supportsCopy() - capability  │
├─────────────────────────────────────────────────────────────────────┤
│ When to use what:                                                   │
│   • TP AllReduce after row-parallel → allreduce()                   │
│   • TP AllGather for LM head        → allgather()                   │
│   • PP activation transfer          → copy() (NEW)                  │
│   • Weight streaming H2D            → copy() (FUTURE)               │
│   • Multi-node send/recv            → send()/recv()                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Backend Implementations

### 4.1 Implementation Matrix

| Backend | Same-Vendor GPU→GPU | Cross-Vendor GPU→GPU | Host↔GPU | Notes |
|---------|---------------------|----------------------|----------|-------|
| **NCCLBackend** | `cudaMemcpyPeerAsync` | ❌ Returns false | `cudaMemcpy` | CUDA-only |
| **RCCLBackend** | `hipMemcpyPeerAsync` | ❌ Returns false | `hipMemcpy` | ROCm-only |
| **PCIeBARBackend** | Delegates to NCCL/RCCL | BAR1 mapping | ❌ Returns false | Cross-vendor specialist |
| **HostBackend** | ❌ Returns false | ❌ Returns false | `memcpy` | CPU↔CPU only |

### 4.2 Fail-Fast Philosophy

**No silent fallbacks.** If a backend can't perform the requested transfer optimally, it returns `false` rather than falling back to host staging. This ensures:

1. **Performance bugs are loud** - You'll know immediately if P2P isn't working
2. **Explicit backend selection** - Caller must choose the right backend for the job
3. **No surprise PCIe round-trips** - Host staging only happens when explicitly requested

```cpp
// Example: Trying NCCL for cross-vendor will fail fast
ICollectiveBackend* backend = getNcclBackend();
if (!backend->supportsCopy(cuda_device, rocm_device)) {
    // This is expected - NCCL doesn't do cross-vendor
    // Caller should use PCIeBAR backend instead
    backend = getPCIeBARBackend();
}
```

The **caller** (LocalPPContext, WeightStreamer, etc.) is responsible for:
1. Checking `supportsCopy()` to find a capable backend
2. Falling back to a different backend if needed
3. Logging warnings if using a slower path

### 4.3 NCCL Backend Implementation

```cpp
// NCCLBackend::copy()
bool copy(void* dst_ptr, DeviceId dst_device,
          const void* src_ptr, DeviceId src_device,
          size_t bytes) override {
    
    // Both must be CUDA - fail fast if not
    if (!src_device.is_cuda() || !dst_device.is_cuda()) {
        LOG_ERROR("NCCLBackend::copy: requires CUDA devices, got " 
                  << src_device << " -> " << dst_device);
        return false;
    }
    
    int src_idx = src_device.toKernelDeviceIndex();
    int dst_idx = dst_device.toKernelDeviceIndex();
    
    if (src_idx == dst_idx) {
        // Same device - use regular memcpy
        cudaSetDevice(dst_idx);
        CUDA_CHECK(cudaMemcpy(dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToDevice));
    } else {
        // Different CUDA devices - require P2P
        int can_access = 0;
        cudaDeviceCanAccessPeer(&can_access, dst_idx, src_idx);
        
        if (!can_access) {
            // FAIL FAST - no silent fallback to host staging
            LOG_ERROR("NCCLBackend::copy: P2P not available between CUDA:" 
                      << src_idx << " and CUDA:" << dst_idx 
                      << ". Enable P2P or use a different backend.");
            return false;
        }
        
        cudaSetDevice(dst_idx);
        CUDA_CHECK(cudaMemcpyPeer(dst_ptr, dst_idx, src_ptr, src_idx, bytes));
    }
    
    cudaDeviceSynchronize();
    return true;
}

bool supportsCopy(DeviceId src, DeviceId dst) const override {
    if (!src.is_cuda() || !dst.is_cuda()) return false;
    
    int src_idx = src.toKernelDeviceIndex();
    int dst_idx = dst.toKernelDeviceIndex();
    
    if (src_idx == dst_idx) return true;  // Same device always works
    
    // Check P2P capability
    int can_access = 0;
    cudaDeviceCanAccessPeer(&can_access, dst_idx, src_idx);
    return can_access != 0;
}
```

### 4.4 PCIeBAR Backend Implementation

```cpp
// PCIeBARBackend::copy()
bool copy(void* dst_ptr, DeviceId dst_device,
          const void* src_ptr, DeviceId src_device,
          size_t bytes) override {
    
    // Same vendor - delegate to appropriate backend
    if (src_device.is_cuda() && dst_device.is_cuda()) {
        if (!nccl_backend_ || !nccl_backend_->supportsCopy(src_device, dst_device)) {
            LOG_ERROR("PCIeBARBackend::copy: NCCL backend unavailable or P2P not supported");
            return false;
        }
        return nccl_backend_->copy(dst_ptr, dst_device, src_ptr, src_device, bytes);
    }
    if (src_device.is_rocm() && dst_device.is_rocm()) {
        if (!rccl_backend_ || !rccl_backend_->supportsCopy(src_device, dst_device)) {
            LOG_ERROR("PCIeBARBackend::copy: RCCL backend unavailable or P2P not supported");
            return false;
        }
        return rccl_backend_->copy(dst_ptr, dst_device, src_ptr, src_device, bytes);
    }
    
    // Cross-vendor - use BAR1 mapping (our specialty)
    if (src_device.is_cuda() && dst_device.is_rocm()) {
        return transferCUDAtoROCm(dst_ptr, dst_device, src_ptr, src_device, bytes);
    }
    if (src_device.is_rocm() && dst_device.is_cuda()) {
        return transferROCmtoCUDA(dst_ptr, dst_device, src_ptr, src_device, bytes);
    }
    
    // Host involved - NOT SUPPORTED (fail fast, no silent staging)
    LOG_ERROR("PCIeBARBackend::copy: Host transfers not supported. "
              << "Use HostBackend for CPU↔CPU or direct CUDA/HIP APIs for Host↔GPU. "
              << "Got: " << src_device << " -> " << dst_device);
    return false;
}

bool supportsCopy(DeviceId src, DeviceId dst) const override {
    // Same CUDA - delegate to NCCL
    if (src.is_cuda() && dst.is_cuda()) {
        return nccl_backend_ && nccl_backend_->supportsCopy(src, dst);
    }
    // Same ROCm - delegate to RCCL  
    if (src.is_rocm() && dst.is_rocm()) {
        return rccl_backend_ && rccl_backend_->supportsCopy(src, dst);
    }
    // Cross-vendor GPU - our specialty
    if ((src.is_cuda() && dst.is_rocm()) || (src.is_rocm() && dst.is_cuda())) {
        return true;  // PCIeBAR supports this
    }
    // Anything with host - not supported
    return false;
}
```

---

## 5. Integration with LocalPPContext

### 5.1 Current Flow (to be replaced)

```
LocalPPContext::transfer()
    └── directMemcpy()
            ├── ensureOnDevice(src)     // May trigger host sync
            ├── cudaMemcpy D2H          // GPU → Host staging
            ├── cudaMemcpy H2D          // Host staging → GPU
            └── mark_device_dirty()     // Update coherence
```

### 5.2 New Flow

```
LocalPPContext::transfer()
    └── getBackendForPair(src_device, dst_device)
            └── backend->copy(dst_ptr, dst_device, src_ptr, src_device, bytes)
                    ├── [NCCL] cudaMemcpyPeer     // Same-vendor P2P
                    ├── [PCIeBAR] BAR1 transfer   // Cross-vendor direct
                    └── [Host] staged memcpy      // Fallback
```

### 5.3 LocalPPContext Changes

```cpp
bool LocalPPContext::transfer(TensorBase* activations, int stage_from, int stage_to) {
    DeviceId src_device = config_.stage_devices[stage_from].toLocalDeviceId();
    DeviceId dst_device = config_.stage_devices[stage_to].toLocalDeviceId();
    
    // Ensure source data is on source device
    if (!activations->ensureOnDevice(src_device)) {
        return false;
    }
    
    // Ensure destination buffer exists
    if (!activations->ensureOnDevice(dst_device)) {
        return false;
    }
    
    // Get appropriate backend for this device pair
    ICollectiveBackend* backend = getBackendForCopy(src_device, dst_device);
    if (!backend) {
        LOG_ERROR("LocalPPContext::transfer: No backend registered for " 
                  << src_device << " -> " << dst_device);
        return false;
    }
    
    // Pre-check capability (fail fast with clear message)
    if (!backend->supportsCopy(src_device, dst_device)) {
        LOG_ERROR("LocalPPContext::transfer: Backend " << backend->name() 
                  << " does not support copy from " << src_device 
                  << " to " << dst_device 
                  << ". Check P2P capability or use a different backend.");
        return false;
    }
    
    // Perform the copy
    size_t bytes = activations->sizeInBytes();
    void* src_ptr = activations->gpu_data_ptr_for_device(src_device);
    void* dst_ptr = activations->gpu_data_ptr_for_device(dst_device);
    
    if (!backend->copy(dst_ptr, dst_device, src_ptr, src_device, bytes)) {
        LOG_ERROR("LocalPPContext::transfer: copy() failed");
        return false;
    }
    
    // Update coherence: destination now has authoritative copy
    activations->setAuthoritativeDevice(dst_device);
    
    return true;
}
```

---

## 6. TensorBase Coherence Updates

### 6.1 New Methods Needed

```cpp
class TensorBase {
public:
    // ... existing methods ...
    
    /**
     * @brief Get GPU data pointer for a specific device
     * 
     * Unlike gpu_data_ptr() which returns "the" GPU pointer, this returns
     * the pointer for a specific device. Used for multi-device scenarios.
     */
    void* gpu_data_ptr_for_device(DeviceId device);
    
    /**
     * @brief Set which device has the authoritative copy
     * 
     * Called after a direct GPU-to-GPU copy to indicate the destination
     * now has valid data without going through host.
     */
    void setAuthoritativeDevice(DeviceId device);
    
    /**
     * @brief Check if tensor has valid data on given device
     */
    bool hasValidDataOnDevice(DeviceId device) const;
};
```

### 6.2 Coherence State Machine Update

```
Current states:
    HOST_VALID          - Host has authoritative copy
    DEVICE_VALID        - Single device has authoritative copy
    SYNCHRONIZED        - Host and one device are in sync

New states needed:
    MULTI_DEVICE_VALID  - Multiple devices have valid copies (for replicated tensors)
    
Transitions for copy():
    
    [DEVICE_A_VALID] --copy(A→B)--> [DEVICE_B_VALID]
        • Destination becomes authoritative
        • Source copy is now stale (unless we track multi-device)
        
    For PP: We only need single-device-valid since activations move through pipeline
    For weight replication: Future work to track multi-device validity
```

---

## 7. Future Use Cases (Designed For)

### 7.1 Weight Streaming Integration

```cpp
// IWeightStreamer implementation can use copy() for transfers
bool LayerWeightStreamer::ensureLayerOnDevice(int layer_idx, DeviceId device) {
    if (isLayerCached(layer_idx, device)) {
        return true;  // Cache hit
    }
    
    // Get source pointer (could be host or another GPU)
    auto [src_ptr, src_device] = getWeightSource(layer_idx);
    
    // Allocate destination in GPU cache
    void* dst_ptr = allocateInCache(layer_idx, device);
    
    // Use collective backend for optimal transfer
    ICollectiveBackend* backend = backend_router_->getBackendForCopy(src_device, device);
    return backend->copy(dst_ptr, device, src_ptr, src_device, layer_size);
}
```

### 7.2 Dynamic Memory Balancing

```cpp
// Move weights between GPUs based on memory pressure
void MemoryBalancer::rebalance() {
    DeviceId high_pressure = findHighestMemoryPressure();
    DeviceId low_pressure = findLowestMemoryPressure();
    
    // Find evictable layer on high-pressure device
    int layer_to_move = selectLayerToEvict(high_pressure);
    
    // Move to low-pressure device
    TensorBase* weights = getLayerWeights(layer_to_move);
    backend_->copy(
        weights->gpu_data_ptr_for_device(low_pressure), low_pressure,
        weights->gpu_data_ptr_for_device(high_pressure), high_pressure,
        weights->sizeInBytes());
    
    // Update placement records
    layer_placement_[layer_to_move] = low_pressure;
}
```

### 7.3 Heterogeneous Compute

```cpp
// During low-memory situations, offload some layers to CPU
void HeterogeneousScheduler::offloadToCPU(int layer_idx) {
    TensorBase* weights = getLayerWeights(layer_idx);
    DeviceId gpu = getCurrentDevice(layer_idx);
    DeviceId cpu = DeviceId::cpu();
    
    // D2H copy
    backend_->copy(
        weights->mutable_data(), cpu,      // Host destination
        weights->gpu_data_ptr(), gpu,      // GPU source
        weights->sizeInBytes());
    
    // Free GPU memory
    weights->freeDeviceBuffer(gpu);
}
```

---

## 8. Implementation Plan

### Phase 1: Interface & Host Backend (2-3 hours)

| Task | Description | Files |
|------|-------------|-------|
| 1.1 | Add `copy()`, `copyAsync()`, `supportsCopy()` to interface | `ICollectiveBackend.h` |
| 1.2 | Implement in `HostBackend` (simple memcpy) | `HostBackend.cpp` |
| 1.3 | Add unit tests for host backend | `Test__HostBackend.cpp` |

### Phase 2: NCCL Backend (2-3 hours)

| Task | Description | Files |
|------|-------------|-------|
| 2.1 | Implement `copy()` with cudaMemcpyPeer | `NCCLBackend.cpp` |
| 2.2 | Implement `copyAsync()` with cudaMemcpyPeerAsync | `NCCLBackend.cpp` |
| 2.3 | Add P2P capability detection | `NCCLBackend.cpp` |
| 2.4 | Add unit tests | `Test__NCCLBackend.cpp` |

### Phase 3: RCCL Backend (2-3 hours)

| Task | Description | Files |
|------|-------------|-------|
| 3.1 | Implement `copy()` with hipMemcpyPeer | `RCCLBackendHIP.cpp` |
| 3.2 | Implement `copyAsync()` with hipMemcpyPeerAsync | `RCCLBackendHIP.cpp` |
| 3.3 | Add unit tests | `Test__RCCLBackend.cpp` |

### Phase 4: PCIeBAR Backend (3-4 hours)

| Task | Description | Files |
|------|-------------|-------|
| 4.1 | Implement `copy()` with cross-vendor support | `PCIeBARBackend.cpp` |
| 4.2 | Route same-vendor to NCCL/RCCL | `PCIeBARBackend.cpp` |
| 4.3 | Add ROCm→CUDA direction (currently missing) | `PCIeBARBackend.cpp` |
| 4.4 | Add cross-vendor tests | `Test__PCIeBARBackend.cpp` |

### Phase 5: LocalPPContext Integration (2-3 hours)

| Task | Description | Files |
|------|-------------|-------|
| 5.1 | Add `getBackendForCopy()` method | `LocalPPContext.cpp` |
| 5.2 | Replace `directMemcpy()` with backend copy | `LocalPPContext.cpp` |
| 5.3 | Update coherence handling | `LocalPPContext.cpp` |
| 5.4 | Integration tests | `Test__LocalPP_Copy.cpp` |

### Phase 6: TensorBase Updates (2-3 hours)

| Task | Description | Files |
|------|-------------|-------|
| 6.1 | Add `gpu_data_ptr_for_device()` | `TensorClasses.h` |
| 6.2 | Add `setAuthoritativeDevice()` | `TensorClasses.h` |
| 6.3 | Update coherence state machine | `TensorClasses.cpp` |
| 6.4 | Unit tests for new methods | `Test__TensorBase.cpp` |

### Phase 7: Testing & Validation (2-3 hours)

| Task | Description | Files |
|------|-------------|-------|
| 7.1 | Integration tests: PP parity with new copy | `Test__LocalPP_Parity.cpp` |
| 7.2 | Performance comparison: old vs new | `Test__PP_CopyPerformance.cpp` |
| 7.3 | Stress tests: many copies, large tensors | `Test__Copy_Stress.cpp` |

---

## 9. Testing Strategy

### 9.1 Unit Tests

```cpp
// Test matrix for copy()
TEST(Test__CollectiveBackend_Copy, CUDA_to_CUDA_SameDevice) { ... }
TEST(Test__CollectiveBackend_Copy, CUDA_to_CUDA_DifferentDevice) { ... }
TEST(Test__CollectiveBackend_Copy, CUDA_to_CUDA_P2PDisabled) { ... }
TEST(Test__CollectiveBackend_Copy, ROCm_to_ROCm_SameDevice) { ... }
TEST(Test__CollectiveBackend_Copy, ROCm_to_ROCm_DifferentDevice) { ... }
TEST(Test__CollectiveBackend_Copy, CUDA_to_ROCm_CrossVendor) { ... }
TEST(Test__CollectiveBackend_Copy, ROCm_to_CUDA_CrossVendor) { ... }
TEST(Test__CollectiveBackend_Copy, Host_to_CUDA) { ... }
TEST(Test__CollectiveBackend_Copy, CUDA_to_Host) { ... }
TEST(Test__CollectiveBackend_Copy, AsyncCompletion) { ... }
TEST(Test__CollectiveBackend_Copy, LargeTensor) { ... }
TEST(Test__CollectiveBackend_Copy, ZeroBytes) { ... }
```

### 9.2 Integration Tests

```cpp
// PP pipeline with new copy system
TEST(Test__LocalPP_Copy_Integration, TwoStage_CUDA_CUDA) { ... }
TEST(Test__LocalPP_Copy_Integration, TwoStage_ROCm_ROCm) { ... }
TEST(Test__LocalPP_Copy_Integration, TwoStage_CUDA_ROCm) { ... }
TEST(Test__LocalPP_Copy_Integration, ThreeStage_Mixed) { ... }
TEST(Test__LocalPP_Copy_Integration, Parity_WithOldDirectMemcpy) { ... }
```

### 9.3 Performance Tests

```cpp
// Compare old vs new transfer paths
TEST(Test__Copy_Performance, Bandwidth_CUDA_to_CUDA_Peer) { ... }
TEST(Test__Copy_Performance, Bandwidth_CUDA_to_CUDA_Staged) { ... }
TEST(Test__Copy_Performance, Bandwidth_CUDA_to_ROCm_BAR1) { ... }
TEST(Test__Copy_Performance, Bandwidth_CUDA_to_ROCm_Staged) { ... }
```

---

## 10. Success Criteria

| Criterion | Metric | Target |
|-----------|--------|--------|
| **Correctness** | All existing PP parity tests pass | 100% |
| **Performance** | Same-vendor P2P bandwidth | ≥90% of cudaMemcpyPeer |
| **Performance** | Cross-vendor bandwidth | ≥ current PCIeBAR implementation |
| **Code quality** | No new memory leaks | ASAN clean |
| **Test coverage** | New copy code paths | ≥80% line coverage |

---

## 11. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| P2P not available on some systems | `copy()` returns false | Clear error message with actionable advice; user must enable P2P or use different topology |
| ROCm P2P across GPUs is flaky | Data corruption | Validate transfers in debug mode; fail fast on errors |
| Coherence complexity increases | Bugs in multi-device tracking | Start simple (single-authoritative), expand later |
| NCCL/RCCL version incompatibility | Runtime failures | Dynamic capability detection via `supportsCopy()` |
| Users expect silent fallback | Confusion when `copy()` fails | Clear documentation; `supportsCopy()` for pre-checking |

---

## 12. Open Questions

1. **Should copy() update TensorBase coherence automatically?**
   - Option A: Yes, copy() calls setAuthoritativeDevice()
   - Option B: No, caller is responsible
   - **Recommendation**: Option B - keeps copy() low-level, coherence is tensor's job

2. **Should we support multi-device validity tracking?**
   - Needed for: weight replication, read-only tensors on multiple GPUs
   - **Recommendation**: Defer to future work; PP only needs single-authoritative

3. **Stream management for async copy?**
   - Should copy use backend's internal stream or caller-provided?
   - **Recommendation**: Accept optional stream parameter, default to backend's stream

---

## 13. Appendix: Related Files

### Modified Files

- `src/v2/collective/ICollectiveBackend.h`
- `src/v2/collective/backends/HostBackend.cpp`
- `src/v2/collective/backends/NCCLBackend.cpp`
- `src/v2/collective/backends/RCCLBackendHIP.cpp`
- `src/v2/collective/backends/PCIeBARBackend.cpp`
- `src/v2/collective/LocalPPContext.cpp`
- `src/v2/tensors/TensorClasses.h`
- `src/v2/tensors/TensorClasses.cpp`

### New Test Files

- `tests/v2/unit/collective/Test__CollectiveBackend_Copy.cpp`
- `tests/v2/integration/pipeline/Test__LocalPP_Copy_Integration.cpp`
- `tests/v2/performance/Test__Copy_Performance.cpp`

### Reference Files

- `src/v2/loaders/IWeightStreamer.h` (future integration)
- `docs/v2/OPTION_B_WEIGHT_STREAMING_DESIGN.md` (context)
