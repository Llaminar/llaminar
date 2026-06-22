# GPU-Native Tensor Coherence Design

**Status**: Design Proposal  
**Author**: Claude + David  
**Date**: 2026-02-01  

## Problem Statement

The current tensor coherence model assumes **host memory is always the source of truth**. This creates unnecessary overhead for GPU-native workloads:

```
Current: GPU_A → Host → GPU_B  (2 PCIe transfers)
Desired: GPU_A → GPU_B         (1 P2P/BAR transfer)
```

### Affected Use Cases

| Use Case | Current Overhead | With GPU-Native |
|----------|------------------|-----------------|
| **PP Activation Transfer** | 2× PCIe (H2D + D2H) | 1× P2P/BAR |
| **Weight Streaming** | Can't move weights between GPUs | Direct GPU→GPU |
| **KV Cache Migration** | Requires host staging | Direct GPU→GPU |
| **TP Intermediate Buffers** | Each device has copy | Share via P2P |

### Root Cause

Current state machine in `TensorBase`:
```
        HOST_AUTHORITATIVE:   host_valid_=true,  device_valid_=false
        DEVICE_AUTHORITATIVE: host_valid_=false, device_valid_=true  ← Only ONE device
        SYNCED:               host_valid_=true,  device_valid_=true
        INVALID:              host_valid_=false, device_valid_=false (ERROR)
```

**Limitation**: `device_valid_` tracks a single boolean, not per-device validity. When we call `ensureOnDevice(GPU_B)` while `device_valid_=true` on GPU_A, we must go through host because the model doesn't support tracking validity across multiple devices.

## Proposed Solution: Multi-Device Coherence

### New State Model

Replace single `device_valid_` with **authoritative device tracking**:

```cpp
// Current (single device):
bool device_valid_ = false;
std::optional<DeviceId> gpu_device_;

// Proposed (multi-device aware):
std::optional<DeviceId> authoritative_device_;  // Which device has current data (nullopt = host)
std::unordered_map<DeviceId, void*> device_buffers_;  // Per-device GPU pointers
std::unordered_map<DeviceId, bool> device_validity_;  // Per-device validity tracking
```

### New State Combinations

| State | `authoritative_device_` | `host_valid_` | Meaning |
|-------|------------------------|---------------|---------|
| HOST_AUTHORITATIVE | `nullopt` | `true` | Host is current, GPUs may be stale |
| GPU_AUTHORITATIVE | `DeviceId` | `false` | Specific GPU is current, host/others stale |
| HOST_SYNCED_WITH_GPU | `nullopt` | `true` | Host current, specific GPU also valid |
| GPU_NATIVE | `DeviceId` | N/A | **No host backing** (new!) |

### New API Methods

```cpp
class TensorBase {
public:
    // ===== Existing Methods (unchanged semantics) =====
    bool ensureOnDevice(DeviceId device);   // Upload from host to device
    bool ensureOnHost();                     // Sync device → host
    void mark_device_dirty();                // Current device has authoritative data
    
    // ===== New Methods for Multi-Device =====
    
    /**
     * @brief Transfer data directly from one GPU to another
     * 
     * Uses ICollectiveBackend::copy() for P2P/BAR transfer.
     * Does NOT go through host staging.
     * 
     * @param dst_device Target GPU device
     * @return true on success, false if transfer not supported
     * 
     * @pre Tensor must be on a GPU (authoritative_device_.has_value())
     * @post authoritative_device_ = dst_device (source becomes stale)
     */
    bool transferTo(DeviceId dst_device);
    
    /**
     * @brief Transfer data directly, keeping source valid (dual residency)
     * 
     * Like transferTo() but marks BOTH devices as valid.
     * Useful for read-only sharing or when source needs to continue.
     * 
     * @param dst_device Target GPU device
     * @return true on success
     */
    bool copyTo(DeviceId dst_device);
    
    /**
     * @brief Set which device is authoritative (for external transfers)
     * 
     * Called after ICollectiveBackend::copy() completes to update coherence state.
     * Invalidates host and all other GPU copies.
     * 
     * @param device Device that now has authoritative data
     */
    void setAuthoritativeDevice(DeviceId device);
    
    /**
     * @brief Get raw GPU pointer for a specific device
     * 
     * Returns pointer for use with ICollectiveBackend::copy().
     * May allocate buffer if not yet present on target device.
     * 
     * @param device Target device
     * @return GPU pointer or nullptr if allocation failed
     */
    void* getDevicePtr(DeviceId device);
    
    /**
     * @brief Check if host backing memory exists
     * 
     * GPU-native tensors created via createOnDevice() have no host backing.
     */
    bool hasHostBacking() const { return host_buffer_ != nullptr; }
    
    /**
     * @brief Create tensor directly on GPU (no host backing)
     * 
     * Static factory for GPU-native activations.
     * 
     * @note Only activation tensor types support this (FP32, BF16, etc.)
     */
    // Implemented in derived classes:
    // static std::unique_ptr<FP32Tensor> createOnDevice(Shape shape, DeviceId device);
};
```

## Implementation Plan

### Phase 1: Add Multi-Device Tracking (Non-Breaking)

**Changes to `TensorBase`:**

```cpp
// Add alongside existing fields (backward compatible):
std::optional<DeviceId> authoritative_device_;  // nullopt means host is authoritative
std::unordered_map<DeviceId, void*, DeviceIdHash> secondary_device_buffers_;

// Modify ensureOnDevice():
bool TensorBase::ensureOnDevice(DeviceId target_device) {
    // If we have authoritative device and it's different from target...
    if (authoritative_device_.has_value() && *authoritative_device_ != target_device) {
        // Try direct GPU→GPU transfer first
        if (tryDirectTransfer(*authoritative_device_, target_device)) {
            authoritative_device_ = target_device;
            return true;
        }
        // Fall back to host staging (current behavior)
    }
    // ... existing logic ...
}
```

### Phase 2: Add `transferTo()` Method

```cpp
bool TensorBase::transferTo(DeviceId dst_device) {
    if (!authoritative_device_.has_value()) {
        LOG_ERROR("transferTo: Tensor not on GPU (use ensureOnDevice first)");
        return false;
    }
    
    DeviceId src_device = *authoritative_device_;
    if (src_device == dst_device) {
        return true;  // Already there
    }
    
    // Get appropriate backend
    auto* backend = GlobalBackendRouter::getBackendForCopy(src_device, dst_device);
    if (!backend || !backend->supportsCopy(src_device, dst_device)) {
        LOG_ERROR("transferTo: No backend supports " << src_device << " → " << dst_device);
        return false;  // Fail-fast, no silent host fallback
    }
    
    // Ensure we have a buffer on destination
    void* dst_ptr = getDevicePtr(dst_device);
    if (!dst_ptr) {
        LOG_ERROR("transferTo: Failed to allocate buffer on " << dst_device);
        return false;
    }
    
    // Perform direct copy
    if (!backend->copy(dst_ptr, dst_device, gpu_data_ptr_, src_device, size_bytes())) {
        LOG_ERROR("transferTo: Backend copy failed");
        return false;
    }
    
    // Update coherence state
    authoritative_device_ = dst_device;
    gpu_device_ = dst_device;
    gpu_data_ptr_ = dst_ptr;
    
    // Source device buffer is now stale (but kept for potential reuse)
    device_validity_[src_device] = false;
    device_validity_[dst_device] = true;
    
    return true;
}
```

### Phase 3: GPU-Native Tensor Creation

Add static factory to activation tensor classes:

```cpp
// In FP32Tensor:
std::unique_ptr<FP32Tensor> FP32Tensor::createOnDevice(size_t rows, size_t cols, DeviceId device) {
    auto tensor = std::make_unique<FP32Tensor>();
    tensor->rows_ = rows;
    tensor->cols_ = cols;
    
    // Allocate GPU buffer directly (no host backing)
    size_t bytes = rows * cols * sizeof(float);
    tensor->gpu_data_ptr_ = allocateDeviceBuffer(device, bytes);
    if (!tensor->gpu_data_ptr_) {
        LOG_ERROR("createOnDevice: GPU allocation failed for " << bytes << " bytes");
        return nullptr;
    }
    
    tensor->gpu_device_ = device;
    tensor->authoritative_device_ = device;
    tensor->host_buffer_ = nullptr;  // No host backing!
    tensor->host_valid_ = false;
    
    return tensor;
}
```

### Phase 4: Update LocalPPContext

Replace `directMemcpy()` with `transferTo()`:

```cpp
bool LocalPPContext::transfer(TensorBase* activations, int stage_from, int stage_to) {
    DeviceId src_device = getDeviceForStage(stage_from);
    DeviceId dst_device = getDeviceForStage(stage_to);
    
    if (src_device == dst_device) {
        return true;  // Same device, no transfer needed
    }
    
    // Use new direct GPU→GPU transfer
    if (activations->transferTo(dst_device)) {
        LOG_DEBUG("LocalPPContext::transfer: Direct P2P/BAR transfer completed");
        return true;
    }
    
    // transferTo() returned false = no supported backend
    // This is a configuration error, not a silent fallback situation
    LOG_ERROR("LocalPPContext::transfer: No backend supports " 
              << src_device << " → " << dst_device);
    return false;
}
```

## Coherence State Transitions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        COHERENCE STATE MACHINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐                                                       │
│  │ HOST_ONLY       │ ← Initial state for host-created tensors              │
│  │ authoritative=∅ │                                                       │
│  │ host_valid=true │                                                       │
│  └────────┬────────┘                                                       │
│           │ ensureOnDevice(GPU_A)                                          │
│           ▼                                                                 │
│  ┌─────────────────┐                                                       │
│  │ HOST_AND_GPU_A  │ ← Host + GPU_A both valid                             │
│  │ authoritative=∅ │                                                       │
│  │ host_valid=true │                                                       │
│  │ GPU_A valid     │                                                       │
│  └────────┬────────┘                                                       │
│           │ mark_device_dirty()                                            │
│           ▼                                                                 │
│  ┌─────────────────┐         transferTo(GPU_B)    ┌─────────────────┐      │
│  │ GPU_A_AUTHORI-  │ ─────────────────────────────▶│ GPU_B_AUTHORI-  │      │
│  │ TATIVE          │  (direct P2P/BAR transfer)   │ TATIVE          │      │
│  │ authoritative=A │                               │ authoritative=B │      │
│  │ host_valid=false│                               │ host_valid=false│      │
│  └────────┬────────┘                               └────────┬────────┘      │
│           │ ensureOnHost()                                  │               │
│           ▼                                                 │ ensureOnHost()│
│  ┌─────────────────┐                                        ▼               │
│  │ HOST_SYNCED_A   │                               ┌─────────────────┐      │
│  │ authoritative=∅ │                               │ HOST_SYNCED_B   │      │
│  │ host_valid=true │                               │ authoritative=∅ │      │
│  │ GPU_A valid     │                               │ host_valid=true │      │
│  └─────────────────┘                               └─────────────────┘      │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                             │
│  ┌─────────────────┐                                                       │
│  │ GPU_NATIVE      │ ← Created via createOnDevice() (no host backing)      │
│  │ authoritative=A │                                                       │
│  │ host_buffer=∅   │   transferTo(GPU_B)   ┌─────────────────┐             │
│  │ GPU_A ptr set   │ ──────────────────────▶│ GPU_NATIVE      │             │
│  └─────────────────┘                        │ authoritative=B │             │
│                                              │ host_buffer=∅   │             │
│    ⚠️ ensureOnHost() will FAIL!              │ GPU_B ptr set   │             │
│       (no host backing exists)              └─────────────────┘             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Backend Selection for Direct Transfers

| Source → Dest | Backend | Method |
|---------------|---------|--------|
| CUDA → CUDA | NCCLBackend | `cudaMemcpyPeer` |
| ROCm → ROCm | RCCLBackend | `hipMemcpyPeer` |
| CUDA → ROCm | PCIeBARBackend | BAR1 mapping |
| ROCm → CUDA | PCIeBARBackend | BAR1 mapping |
| CPU → CPU | HostBackend | `memcpy` |
| CPU → GPU | (use ensureOnDevice) | existing H2D |
| GPU → CPU | (use ensureOnHost) | existing D2H |

## Migration Path

### Backward Compatibility

All existing code continues to work:
- `ensureOnDevice()` / `ensureOnHost()` unchanged
- `data()` / `mutable_data()` unchanged
- `mark_device_dirty()` unchanged

### New Code Path

For GPU-native workloads:
```cpp
// PP activation (created on GPU_A, computed, transferred to GPU_B)
auto activation = FP32Tensor::createOnDevice({batch, hidden}, cuda_devices[0]);
stage_A->compute(activation.get());
activation->transferTo(rocm_devices[0]);  // Direct P2P/BAR
stage_B->compute(activation.get());
```

## Testing Strategy

### Unit Tests
- `Test__TensorBase_TransferTo` - Direct GPU→GPU transfer
- `Test__TensorBase_CopyTo` - Dual-residency copy
- `Test__TensorBase_SetAuthoritativeDevice` - Manual coherence update
- `Test__TensorBase_CreateOnDevice` - GPU-native tensor creation
- `Test__TensorBase_EnsureOnHostNoBackingFails` - Error handling

### Integration Tests
- `Test__LocalPP_DirectTransfer` - PP with new transfer path
- `Test__CrossVendor_DirectTransfer` - CUDA↔ROCm via BAR

### Performance Tests
- `Benchmark__PP_Transfer_HostStaging` - Current path (baseline)
- `Benchmark__PP_Transfer_Direct` - New path (expected 2× faster)

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing code | Phase 1 is additive, old code paths unchanged |
| GPU-native tensors accessed via `data()` | Clear error message + documentation |
| Memory leaks from multi-device buffers | RAII cleanup in destructor |
| Complex state machine | Comprehensive state diagram + asserts |

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| PP transfer latency (1MB) | ~400μs (2× H2D/D2H) | ~200μs (1× P2P) |
| PP transfer latency (cross-vendor) | ~600μs (2× PCIe) | ~300μs (1× BAR) |
| Code complexity | Low | Medium (acceptable for 2× perf) |
| Test coverage | N/A | >90% of new paths |

## Appendix: Alternative Designs Considered

### Alternative A: Explicit GPU-Only Tensor Class

Create separate `GpuTensor` class hierarchy:
- Pro: Clean separation, no state machine changes
- Con: Code duplication, can't convert between types

### Alternative B: Multi-Buffer Map Only

Just track `device_buffers_` without `authoritative_device_`:
- Pro: Simpler state model
- Con: Can't distinguish "all valid" from "only one valid"

### Alternative C: Lazy Host Allocation

Allocate host buffer on first `ensureOnHost()` call:
- Pro: Backward compatible for existing tensors
- Con: Surprise allocations, unpredictable memory usage

**Selected**: Hybrid of A and C - `authoritative_device_` tracking with optional host backing.
