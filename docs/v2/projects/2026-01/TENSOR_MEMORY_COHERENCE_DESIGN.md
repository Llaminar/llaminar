# Tensor Host/Device Memory Coherence Design

**Status**: ✅ IMPLEMENTED (January 2026)

**Implementation Location**: `src/v2/tensors/cpu/CPUTensors.h` and `src/v2/tensors/cpu/CPUTensorBase.cpp`

## Overview

This document describes the two-flag coherence model for managing tensor data across host (CPU) and device (GPU) memory. The design enables efficient dual-residency where tensors can exist on both host and device simultaneously, with explicit tracking of which copy contains authoritative data.

## Historical Problem Statement

The original tensor host/device memory management was inconsistent and bug-prone:

1. **Single flag for bidirectional sync**: `host_invalid_` only tracked when GPU has newer data than host. There was no flag to track when host has newer data than GPU.

2. **Destructive invalidation**: `invalidateGpuData()` freed GPU memory entirely instead of just marking it stale. This caused expensive reallocation and data loss.

3. **Semantic confusion**: `mutable_data()` called `invalidateGpuData()` which freed GPU memory even if GPU had the only valid copy.

4. **Implicit side effects**: Getting a pointer triggered transfers and state changes invisibly to the caller.

5. **Inconsistent API**: Some operations synced, some didn't, some allocated, some freed.

## Original State Machine (Broken)

```
States (implicitly encoded):
- gpu_data_ptr_ == nullptr:           HOST_ONLY (data only on host)
- gpu_data_ptr_ != nullptr, host_invalid_ == false: SYNCED (both valid)
- gpu_data_ptr_ != nullptr, host_invalid_ == true:  GPU_ONLY (GPU is authoritative)

Missing state: HOST_AUTHORITATIVE (host has newer data, GPU has stale copy)
```

Original operations (broken):
- `mutable_data()` → ensureOnHost() + invalidateGpuData() → **FREED GPU MEMORY**
- `ensureOnDevice()` → allocated GPU + uploaded from host
- `mark_device_dirty()` → set host_invalid_ = true
- `invalidateGpuData()` → **FREED GPU memory** and set host_invalid_ = false

The problem: When `mutable_data()` was called while GPU had newer data:
1. `ensureOnHost()` downloaded GPU → host (correct)
2. `invalidateGpuData()` **freed GPU memory** (expensive, wasteful)
3. Next `ensureOnDevice()` had to reallocate and re-upload

## Implemented Design: Two-Flag Coherence Model

### State Definition

Two independent validity flags with type-safe device tracking:
```cpp
bool host_valid_ = true;                    // Host data matches authoritative copy
bool device_valid_ = false;                 // Device data matches authoritative copy
void* gpu_data_ptr_ = nullptr;              // GPU buffer (nullptr = not allocated)
std::optional<DeviceId> gpu_device_;        // Which GPU device (nullopt = not on GPU)
```

**Note**: The implementation uses `std::optional<DeviceId>` instead of a raw `int gpu_device_idx_` for type-safe device identification. `DeviceId` supports `DeviceId::cuda(n)` and `DeviceId::rocm(n)` for heterogeneous GPU backends.

### State Machine

```
┌─────────────────────────────────────────────────────────────────┐
│                    Valid State Combinations                      │
├───────────────┬───────────────┬─────────────────────────────────┤
│ host_valid_   │ device_valid_ │ State Name                      │
├───────────────┼───────────────┼─────────────────────────────────┤
│ true          │ false         │ HOST_AUTHORITATIVE              │
│ false         │ true          │ DEVICE_AUTHORITATIVE            │
│ true          │ true          │ SYNCED                          │
│ false         │ false         │ INVALID (error - should never   │
│               │               │ happen, at least one must be    │
│               │               │ valid)                          │
└───────────────┴───────────────┴─────────────────────────────────┘
```

Note: `gpu_data_ptr_ == nullptr` with `device_valid_ == false` is valid (HOST_AUTHORITATIVE before first GPU use).

### Implemented API

#### Read-Only Host Access: `data()`
```cpp
const float* data() const {
    if (!host_valid_) {
        // GPU has authoritative copy - sync to host
        syncDeviceToHost();  // Downloads, sets host_valid_ = true
    }
    return host_ptr_;
}
```
**Post-condition**: `host_valid_ == true`, state unchanged otherwise.

#### Write Host Access: `mutable_data()`
```cpp
float* mutable_data() {
    if (!host_valid_) {
        // GPU has authoritative copy - sync to host first
        syncDeviceToHost();  // Downloads, sets host_valid_ = true
    }
    // Host is about to be modified - mark device as stale (but DON'T free!)
    device_valid_ = false;
    return host_ptr_;
}
```
**Post-condition**: `host_valid_ == true`, `device_valid_ == false`.
**Key difference**: Device memory is NOT freed, just marked stale.

#### Device Validity Check: `is_on_device(DeviceId)`
```cpp
bool is_on_device(DeviceId device) const {
    if (device.is_cpu()) {
        // Asking about CPU/host - return true if host data is valid
        return host_valid_;
    }
    // Asking about a specific GPU - must be on that device AND have valid data
    return gpu_device_.has_value() && *gpu_device_ == device && device_valid_;
}
```
**Note**: This method checks CURRENT validity, not creation device. A tensor can be on multiple devices simultaneously (dual-residency).

#### Read-Only Device Access: `gpu_data_ptr()` + `ensureOnDevice()`
```cpp
void* gpu_data_ptr() const {
    // Returns raw pointer - caller must ensure device_valid_ via ensureOnDevice()
    return gpu_data_ptr_;
}

bool ensureOnDevice(DeviceId target_device) {
    // Validate target device - must be a GPU
    if (!target_device.is_gpu()) {
        return false;
    }
    
    // Case 1: Already on correct device and valid
    if (gpu_data_ptr_ && gpu_device_.has_value() && 
        *gpu_device_ == target_device && device_valid_) {
        return true;  // Nothing to do
    }
    
    // Case 2: Allocated on wrong device - free old and migrate
    if (gpu_data_ptr_ && gpu_device_.has_value() && *gpu_device_ != target_device) {
        IBackend* old_backend = getBackendForDevice(*gpu_device_);
        if (old_backend) {
            old_backend->free(gpu_data_ptr_, gpu_device_->gpu_ordinal());
        }
        gpu_data_ptr_ = nullptr;
        gpu_device_.reset();
        device_valid_ = false;
    }
    
    // Case 3: Need to allocate
    if (!gpu_data_ptr_) {
        IBackend* target_backend = getBackendForDevice(target_device);
        gpu_data_ptr_ = target_backend->allocate(byte_size(), target_device.gpu_ordinal());
        gpu_device_ = target_device;
        device_valid_ = false;  // Allocated but not yet populated
    }
    
    // Case 4: Device data stale - upload from host
    if (!device_valid_) {
        if (!host_valid_) {
            // COHERENCE ERROR: Both host and device are invalid!
            return false;
        }
        IBackend* backend = getBackendForDevice(target_device);
        backend->hostToDevice(gpu_data_ptr_, raw_host_data_ptr(), byte_size(), 
                              target_device.gpu_ordinal());
        device_valid_ = true;
        // host_valid_ stays true - we uploaded FROM host, so both are now in sync
    }
    
    return true;
}
```
**Post-condition**: `device_valid_ == true`, `gpu_data_ptr_ != nullptr`.

#### Write Device Access: `mark_device_dirty()`
```cpp
void mark_device_dirty() {
    device_valid_ = true;   // Device just got written
    host_valid_ = false;    // Host is now stale
}
```
**Post-condition**: `device_valid_ == true`, `host_valid_ == false`.

#### Explicit Sync: `ensureOnHost()`
```cpp
bool ensureOnHost() {
    if (host_valid_) {
        return true;  // Host already valid
    }
    
    if (!device_valid_) {
        // COHERENCE ERROR: Both host and device are invalid!
        return false;
    }
    
    if (gpu_data_ptr_ && gpu_device_.has_value()) {
        IBackend* backend = getBackendForDevice(*gpu_device_);
        backend->deviceToHost(raw_host_data_ptr(), gpu_data_ptr_, byte_size(),
                              gpu_device_->gpu_ordinal());
        host_valid_ = true;
        // device_valid_ stays true - we downloaded FROM device, so both are now in sync
    }
    
    return true;
}
```
**Post-condition**: `host_valid_ == true`.

#### Invalidate GPU Data: `invalidateGpuData()`
```cpp
void invalidateGpuData() {
    // Mark GPU data as stale - next ensureOnDevice() will re-upload from host
    // The GPU memory is kept allocated; next ensureOnDevice() will just re-upload.
    if (gpu_data_ptr_) {
        device_valid_ = false;  // Mark stale, DON'T free memory
    }
}
```
**Post-condition**: `device_valid_ == false`, GPU memory retained.
**Key difference from original**: Memory is NOT freed!

#### Force Release GPU Memory: `releaseDeviceMemory()`
```cpp
bool releaseDeviceMemory() {
    // Ensure host has valid copy before freeing
    if (!ensureOnHost()) {
        return false;
    }
    
    // Free device memory
    if (gpu_data_ptr_ && gpu_device_.has_value()) {
        IBackend* backend = getBackendForDevice(*gpu_device_);
        backend->free(gpu_data_ptr_, gpu_device_->gpu_ordinal());
        gpu_data_ptr_ = nullptr;
        device_valid_ = false;
        gpu_device_.reset();
    }
    
    return true;
}
```
**Post-condition**: `host_valid_ == true`, `device_valid_ == false`, `gpu_data_ptr_ == nullptr`.

### Helper Methods

#### `isOnGPU()` - Check if tensor has GPU allocation
```cpp
bool isOnGPU() const { return gpu_data_ptr() != nullptr; }
```

#### `isOnCPU()` - Check if host data is valid
```cpp
bool isOnCPU() const { return host_valid_; }
```

#### `isDeviceValid()` - Check if GPU data is current
```cpp
bool isDeviceValid() const { return device_valid_ && gpu_data_ptr_ != nullptr; }
```

## State Transition Diagram

```
                    ┌──────────────────────┐
                    │  HOST_AUTHORITATIVE  │
                    │  host_valid_=true    │
                    │  device_valid_=false │
                    └──────────┬───────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          │ mutable_data()     │ ensureOnDevice()   │ mark_device_dirty()
          │ (stay here)        │ (upload)           │ (after GPU write)
          │                    │                    │
          ▼                    ▼                    ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
│HOST_AUTHORITATIVE│  │      SYNCED      │  │ DEVICE_AUTHORITATIVE │
│ (no change)      │  │  host_valid_=true│  │  host_valid_=false   │
│                  │  │ device_valid_=true│  │  device_valid_=true  │
└──────────────────┘  └────────┬─────────┘  └──────────┬───────────┘
                               │                       │
          ┌────────────────────┼───────────────────────┤
          │                    │                       │
          │ mutable_data()     │ data()                │ data() or
          │ (mark device stale)│ (no change)           │ mutable_data()
          │                    │                       │ (download first)
          │                    │                       │
          ▼                    ▼                       ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
│HOST_AUTHORITATIVE│  │      SYNCED      │  │       SYNCED or      │
│                  │  │  (no change)     │  │   HOST_AUTHORITATIVE │
└──────────────────┘  └──────────────────┘  └──────────────────────┘
```

## Usage Patterns

### Pattern 1: CPU writes, GPU reads (e.g., embedding lookup)
```cpp
// CPU writes to tensor
float* data = tensor->mutable_data();  // Returns host ptr, marks device stale
fill_data(data);

// Later, GPU kernel needs it
tensor->ensureOnDevice(device);         // Uploads since device_valid_=false
float* gpu_ptr = tensor->gpu_data_ptr();
launch_kernel<<<...>>>(gpu_ptr, ...);
// (If kernel only reads, don't call mark_device_dirty)
```

### Pattern 2: GPU writes, CPU reads (e.g., attention output)
```cpp
// GPU kernel writes
tensor->ensureOnDevice(device);
float* gpu_ptr = tensor->gpu_data_ptr();
launch_kernel<<<...>>>(gpu_ptr, ...);
tensor->mark_device_dirty();            // GPU wrote, host is stale

// Later, CPU needs the result
const float* data = tensor->data();     // Downloads from GPU first
```

### Pattern 3: GPU writes, GPU reads (e.g., layer-to-layer)
```cpp
// Kernel 1 writes
tensor->ensureOnDevice(device);
float* gpu_ptr = tensor->gpu_data_ptr();
kernel1<<<...>>>(gpu_ptr, ...);
tensor->mark_device_dirty();            // GPU wrote

// Kernel 2 reads (same tensor)
// ensureOnDevice() is a no-op since device_valid_=true
tensor->ensureOnDevice(device);         
const float* gpu_ptr = tensor->gpu_data_ptr();
kernel2<<<...>>>(gpu_ptr, ...);
// NO host transfer needed!
```

### Pattern 4: Debug logging (problematic case we fixed)
```cpp
// GPU kernel wrote
tensor->mark_device_dirty();

// Debug log wants to see values
LOG_DEBUG("First values: " << tensor->data()[0] << ...);
// This downloads GPU→host (correct!) and sets host_valid_=true
// tensor is now SYNCED (both host_valid_ and device_valid_ are true)

// Next GPU kernel can still read gpu_data_ptr() - NO re-upload needed!
// (In old design, this would have freed GPU memory!)
```

## Backend Integration

The implementation uses `IBackend` interface for heterogeneous GPU support:

```cpp
IBackend* getBackendForDevice(DeviceId device) {
    if (device.is_cuda()) {
        return BackendManager::instance().getBackend(DeviceType::CUDA);
    } else if (device.is_rocm()) {
        return BackendManager::instance().getBackend(DeviceType::ROCm);
    }
    return nullptr;
}
```

The `IBackend` interface provides:
- `allocate(size_t bytes, int device_ordinal)` - Allocate device memory
- `free(void* ptr, int device_ordinal)` - Free device memory
- `hostToDevice(void* dst, const void* src, size_t bytes, int device_ordinal)` - Upload
- `deviceToHost(void* dst, const void* src, size_t bytes, int device_ordinal)` - Download

## Migration Plan (Completed)

### Step 1: ✅ Added two-flag model with DeviceId
- Replaced `int gpu_device_idx_` with `std::optional<DeviceId> gpu_device_`
- Added `host_valid_` (starts true) and `device_valid_` (starts false) flags
- Removed legacy `host_invalid_` flag

### Step 2: ✅ Updated ensureOnDevice/ensureOnHost
- `ensureOnDevice`: Checks `device_valid_`, handles multi-GPU migration
- `ensureOnHost`: Checks `host_valid_`, downloads if needed
- Both methods verify coherence invariant (at least one must be valid)

### Step 3: ✅ Updated invalidateGpuData() to NOT free GPU memory
```cpp
void invalidateGpuData() {
    if (gpu_data_ptr_) {
        device_valid_ = false;  // Mark stale, DON'T FREE!
    }
}
```

### Step 4: ✅ Added releaseDeviceMemory() for explicit memory release
- Separate method when you actually want to free GPU memory
- Downloads to host first, then frees

### Step 5: ✅ Updated all CUDA kernels to use consistent pattern
```cpp
// Before GPU kernel reads:
tensor->ensureOnDevice(device);

// After GPU kernel writes:
tensor->mark_device_dirty();
```

### Step 6: ✅ Added is_on_device(DeviceId) for dual-residency queries
- Returns true if tensor has valid data on specified device
- Supports both CPU (checks `host_valid_`) and GPU (checks `device_valid_`)

## Benefits

1. **Clear semantics**: Every state is well-defined with two orthogonal flags
2. **No wasteful allocation**: `invalidateGpuData()` doesn't free GPU memory
3. **Predictable behavior**: State transitions are explicit and documented
4. **Efficient GPU-to-GPU**: No unnecessary host transfers between GPU operations
5. **Debug-safe**: Logging via `data()` doesn't break GPU pipeline
6. **Type-safe device ID**: Uses `DeviceId` instead of raw integers for heterogeneous GPU support
7. **Dual-residency support**: `is_on_device(DeviceId)` enables checking validity on specific devices

---

## Implementation Summary (January 2026)

The two-flag coherence model has been fully implemented in `CPUTensorBase`:

### Key Code Locations

| Component | File | Description |
|-----------|------|-------------|
| State flags | `CPUTensors.h:1288-1302` | `host_valid_`, `device_valid_`, `gpu_data_ptr_`, `gpu_device_` |
| `mark_device_dirty()` | `CPUTensors.h:786-798` | Base class implementation that sets flags |
| `is_on_device()` | `CPUTensors.h:710-720` | Two-flag aware device query |
| `isOnGPU()` | `CPUTensors.h:773` | Check GPU allocation exists |
| `isOnCPU()` | `CPUTensors.h:800` | Check host data valid |
| `isDeviceValid()` | `CPUTensors.h:805` | Check GPU data current |
| `invalidateGpuData()` | `CPUTensorBase.cpp:552-563` | Marks device stale, retains memory |
| `ensureOnDevice()` | `CPUTensorBase.cpp:565-660` | Two-flag aware upload logic |
| `ensureOnHost()` | `CPUTensorBase.cpp:663-707` | Two-flag aware download logic |
| `releaseDeviceMemory()` | `CPUTensorBase.cpp:709-732` | Explicit GPU memory release |

### State Flag Documentation (in CPUTensors.h)

```cpp
// ===== Host/Device Memory Coherence State =====
// Two-flag model for tracking data validity:
//   host_valid_   = true:  Host data is current (can be read without sync)
//   device_valid_ = true:  Device data is current (can be read without sync)
//
// Valid state combinations:
//   HOST_AUTHORITATIVE:   host_valid_=true,  device_valid_=false
//   DEVICE_AUTHORITATIVE: host_valid_=false, device_valid_=true
//   SYNCED:               host_valid_=true,  device_valid_=true
//   INVALID:              host_valid_=false, device_valid_=false (ERROR STATE)
//
// See docs/v2/projects/2026-01/TENSOR_MEMORY_COHERENCE_DESIGN.md for full design.

void *gpu_data_ptr_ = nullptr;       // GPU buffer pointer (nullptr = not on GPU)
bool host_valid_ = true;             // Host data is current (starts true - data created on host)
bool device_valid_ = false;          // Device data is current (starts false - no GPU alloc)
std::optional<DeviceId> gpu_device_; // Which GPU device (nullopt = not on GPU)
```

### Verification

- All 222+ unit tests pass
- Tensor transfer tests verify correct coherence behavior
- CUDA kernel stages correctly call `mark_device_dirty()` after GPU writes
