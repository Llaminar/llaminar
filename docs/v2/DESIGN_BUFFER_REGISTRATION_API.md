# Buffer Registration API Design

## Executive Summary

The **Buffer Registration API** enables collective backends to track the location of device buffers for proper cross-device communication. This is essential for the `PCIeBARBackend` to perform allreduce operations where the ROCm buffer location must be known as an offset within the PCIe BAR mapping.

**Core Problem**: `PCIeBARBackend::allreduce(buffer, ...)` currently assumes ROCm data is at BAR offset 0, but actual ROCm allocations return arbitrary HIP device pointers that are not at that location.

**Solution**: A buffer registration system that maps device pointers to their backend-specific locations (BAR offsets for PCIe BAR, device IDs for NCCL, etc.).

---

## Table of Contents

1. [Problem Analysis](#problem-analysis)
2. [Design Goals](#design-goals)
3. [Architecture Options](#architecture-options)
4. [Recommended Design: Buffer Registration Interface](#recommended-design)
5. [Implementation Details](#implementation-details)
6. [Migration Path](#migration-path)
7. [Test Strategy](#test-strategy)
8. [Appendix: Alternative Approaches](#appendix)

---

## Problem Analysis

### Current Situation

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Current Flow                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. GraphOrchestrator allocates buffers via TensorFactory            │
│     └── ROCm buffer: hipMalloc() → returns void* at 0x7F4A00000000  │
│                                                                      │
│  2. PCIeBARBackend::allreduce(cuda_buffer, count, dtype, op)        │
│     └── Assumes ROCm data at BAR offset 0                           │
│     └── transferROCmtoCUDA(0, cuda_temp, bytes) ← WRONG OFFSET!     │
│                                                                      │
│  3. Result: Reads garbage from BAR offset 0 instead of actual data  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Root Cause

The `PCIeBARBackend` has no visibility into where device buffers are allocated. When `allreduce()` is called, it receives:
- `buffer`: A CUDA device pointer
- No information about where the corresponding ROCm buffer is located

The PCIe BAR mapping provides a 1GB window into ROCm device memory starting at offset 0. To read from the correct location, the backend needs to know the BAR offset corresponding to the ROCm buffer.

### Key Insight

The fundamental issue is that collective operations in llaminar currently use a **single-buffer model**:
```cpp
bool allreduce(void *buffer, size_t count, ...);
```

But heterogeneous (CUDA↔ROCm) collectives require a **multi-buffer model** where each device has its own buffer at a known location:
```cpp
// Each device has its own buffer:
// - cuda_buffer: CUDA device pointer
// - rocm_buffer: ROCm device pointer (or BAR offset)
```

---

## Design Goals

1. **Minimal API Change**: Extend existing interfaces without breaking compatibility
2. **Backend Agnostic**: Work for all backends (PCIe BAR, NCCL, RCCL, Host)
3. **Lazy Registration**: Support both pre-registration and just-in-time registration
4. **Orchestrator Integration**: Fit naturally with GraphOrchestrator's buffer management
5. **Zero Overhead**: No performance impact when registration is not needed (homogeneous groups)

---

## Architecture Options

### Option A: Extended Collective Signature (Per-Call Buffers)

```cpp
// ICollectiveBackend.h - New overload
virtual bool allreduce(
    const std::vector<BufferInfo>& buffers,  // Buffer per device
    size_t count,
    CollectiveDataType dtype,
    CollectiveOp op) = 0;

struct BufferInfo {
    DeviceId device;
    void* ptr;
    size_t offset;  // Backend-specific (BAR offset for PCIe)
};
```

**Pros**: Explicit, no state management
**Cons**: Changes every call site, verbose API

### Option B: Buffer Registration on Backend (Stateful)

```cpp
// ICollectiveBackend.h - New methods
virtual bool registerBuffer(DeviceId device, void* buffer, size_t size) = 0;
virtual bool unregisterBuffer(DeviceId device, void* buffer) = 0;
virtual void* getRegisteredBuffer(DeviceId device) const = 0;
```

**Pros**: Clean separation, registration can happen once at graph setup
**Cons**: State management complexity, backend must track buffers

### Option C: Buffer Registry (Separate Component)

```cpp
// BufferRegistry.h - Centralized registry
class BufferRegistry {
public:
    void registerBuffer(const std::string& name, DeviceId device, void* ptr, size_t size);
    BufferInfo lookup(const std::string& name, DeviceId device) const;
    std::vector<BufferInfo> lookupAll(const std::string& name) const;
};
```

**Pros**: Decoupled from backends, reusable across components
**Cons**: Requires name-based coordination, extra indirection

### Option D: Collective Context with Buffer Map (Recommended)

```cpp
// CollectiveContext.h - Enhanced context
class CollectiveContext {
public:
    // Existing
    bool allreduce(void* local_buffer, size_t count, ...);
    
    // New: Register buffer for this device
    void registerBuffer(const std::string& collective_id, void* buffer, size_t size);
    
    // New: Multi-device allreduce with explicit buffers
    bool allreduceMultiDevice(
        const std::string& collective_id,  // Links registered buffers
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op);
};
```

**Pros**: Context already owns backend lifecycle, natural extension
**Cons**: Slight API complexity for heterogeneous case

---

## Recommended Design

### Hybrid Approach: Option B + Option D

We extend both `ICollectiveBackend` and `CollectiveContext` to support buffer registration while keeping the existing single-buffer API for homogeneous cases.

### New Interface: `IBufferRegistration`

```cpp
// src/v2/collective/IBufferRegistration.h

/**
 * @brief Interface for backends that require buffer registration
 * 
 * Some backends (e.g., PCIeBARBackend) need to know buffer locations
 * for each device before performing collective operations. Backends
 * that support direct buffer addressing don't need this (e.g., NCCL
 * where all buffers are on the same device type).
 */
class IBufferRegistration {
public:
    virtual ~IBufferRegistration() = default;
    
    /**
     * @brief Register a device buffer with the backend
     * 
     * For PCIeBARBackend:
     * - CUDA buffers: stored directly (no translation needed)
     * - ROCm buffers: translated to BAR offset via hipGetDevicePointer
     * 
     * @param collective_id Identifier linking buffers for same collective
     * @param device Device where buffer resides
     * @param buffer Device pointer
     * @param size Buffer size in bytes
     * @return true if registration successful
     */
    virtual bool registerBuffer(
        const std::string& collective_id,
        DeviceId device,
        void* buffer,
        size_t size) = 0;
    
    /**
     * @brief Unregister a previously registered buffer
     */
    virtual void unregisterBuffer(
        const std::string& collective_id,
        DeviceId device) = 0;
    
    /**
     * @brief Get registered buffer info for a device
     * @return Buffer info, or nullopt if not registered
     */
    virtual std::optional<RegisteredBuffer> getBuffer(
        const std::string& collective_id,
        DeviceId device) const = 0;
    
    /**
     * @brief Check if buffer registration is required
     * 
     * @return true for backends like PCIeBAR that need it,
     *         false for backends like NCCL that don't
     */
    virtual bool requiresBufferRegistration() const = 0;
};

/**
 * @brief Information about a registered buffer
 */
struct RegisteredBuffer {
    DeviceId device;
    void* ptr;           ///< Original device pointer
    size_t size;
    size_t bar_offset;   ///< BAR offset (PCIeBAR backend only)
    bool is_primary;     ///< true for primary device (e.g., CUDA in PCIeBAR)
};
```

### Extended ICollectiveBackend

```cpp
// src/v2/collective/ICollectiveBackend.h

class ICollectiveBackend : public IBufferRegistration {
public:
    // ... existing interface ...
    
    // Default implementation for backends that don't need registration
    bool registerBuffer(...) override { return true; }
    void unregisterBuffer(...) override {}
    std::optional<RegisteredBuffer> getBuffer(...) const override { return std::nullopt; }
    bool requiresBufferRegistration() const override { return false; }
    
    /**
     * @brief AllReduce with registered buffers
     * 
     * Uses pre-registered buffers for each device in the collective.
     * Call registerBuffer() for each device before calling this.
     * 
     * @param collective_id Links to registered buffers
     * @param count Number of elements per device
     * @param dtype Data type
     * @param op Reduction operation
     * @return true on success
     */
    virtual bool allreduceRegistered(
        const std::string& collective_id,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op) = 0;
};
```

### PCIeBARBackend Implementation

```cpp
// src/v2/collective/backends/PCIeBARBackend.h

class PCIeBARBackend : public ICollectiveBackend {
public:
    // ... existing interface ...
    
    // IBufferRegistration implementation
    bool registerBuffer(
        const std::string& collective_id,
        DeviceId device,
        void* buffer,
        size_t size) override;
    
    void unregisterBuffer(
        const std::string& collective_id,
        DeviceId device) override;
    
    std::optional<RegisteredBuffer> getBuffer(
        const std::string& collective_id,
        DeviceId device) const override;
    
    bool requiresBufferRegistration() const override { return true; }
    
    // Registered allreduce
    bool allreduceRegistered(
        const std::string& collective_id,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op) override;

private:
    // Buffer registration storage
    struct CollectiveBuffers {
        RegisteredBuffer cuda_buffer;
        RegisteredBuffer rocm_buffer;
        bool cuda_registered = false;
        bool rocm_registered = false;
    };
    std::unordered_map<std::string, CollectiveBuffers> registered_collectives_;
    
    // Translate ROCm pointer to BAR offset
    std::optional<size_t> translateToBarOffset(void* rocm_ptr, size_t size) const;
};
```

### GraphOrchestrator Integration

```cpp
// src/v2/pipelines/GraphOrchestrator.h

class GraphOrchestrator {
public:
    /**
     * @brief Register buffers for collective operations
     * 
     * Called after buffer allocation, before graph execution.
     * Automatically detects which buffers need registration based on
     * stage requirements and device assignments.
     */
    void registerCollectiveBuffers();
    
private:
    /**
     * @brief Register buffers for a specific stage
     */
    void registerBuffersForStage(
        const std::string& stage_name,
        ComputeStage* stage);
};
```

---

## Implementation Details

### PCIeBARBackend::registerBuffer

```cpp
bool PCIeBARBackend::registerBuffer(
    const std::string& collective_id,
    DeviceId device,
    void* buffer,
    size_t size) 
{
    auto& coll = registered_collectives_[collective_id];
    
    RegisteredBuffer reg;
    reg.device = device;
    reg.ptr = buffer;
    reg.size = size;
    
    if (device.is_cuda()) {
        // CUDA buffers used directly - no translation needed
        reg.bar_offset = 0;  // Not applicable
        reg.is_primary = true;
        coll.cuda_buffer = reg;
        coll.cuda_registered = true;
        LOG_DEBUG("Registered CUDA buffer for collective " << collective_id 
                  << ": ptr=" << buffer << " size=" << size);
    } 
    else if (device.is_rocm()) {
        // ROCm buffers need BAR offset translation
        auto offset = translateToBarOffset(buffer, size);
        if (!offset.has_value()) {
            LOG_ERROR("Failed to translate ROCm pointer to BAR offset");
            return false;
        }
        reg.bar_offset = offset.value();
        reg.is_primary = false;
        coll.rocm_buffer = reg;
        coll.rocm_registered = true;
        LOG_DEBUG("Registered ROCm buffer for collective " << collective_id 
                  << ": ptr=" << buffer << " BAR offset=" << reg.bar_offset);
    }
    else {
        LOG_ERROR("PCIeBARBackend only supports CUDA and ROCm devices");
        return false;
    }
    
    return true;
}
```

### Critical Insight: BAR Memory vs HIP Allocations

The PCIe BAR mapping exposes a region of AMD GPU memory (e.g., 1GB starting at physical offset 0) to CUDA via `mmap()` + `cuMemHostRegister(IOMEMORY)`. However:

1. `hipMalloc()` allocates from HIP's virtual address space, which is **managed separately** from the BAR physical region
2. There's no API to query whether a HIP pointer corresponds to a BAR-accessible physical address
3. HIP virtual addresses ≠ BAR physical offsets

**Consequence**: We cannot simply translate arbitrary `hipMalloc()` pointers to BAR offsets.

### Solution: BAR-Managed Allocation

Instead of trying to translate existing allocations, we provide a **BAR-aware allocator** that carves buffers directly from the BAR-mapped region:

```cpp
// PCIeBARBackend provides BAR-region allocation
class PCIeBARBackend : public ICollectiveBackend {
public:
    /**
     * @brief Allocate a buffer within the BAR-mapped region
     * 
     * This returns a ROCm-accessible pointer that is guaranteed to be
     * at a known offset within the PCIe BAR mapping.
     * 
     * @param size Size in bytes
     * @return Pair of (ROCm pointer, BAR offset), or nullopt if insufficient space
     */
    std::optional<std::pair<void*, size_t>> allocateInBarRegion(size_t size);
    
    /**
     * @brief Free a BAR-region buffer
     */
    void freeBarBuffer(void* ptr);
    
private:
    // Simple bump allocator for BAR region
    size_t bar_alloc_offset_ = 0;
    size_t bar_total_size_ = 0;
    void* bar_base_ptr_ = nullptr;  // Host-mapped ptr to BAR region
    
    // Track allocations for cleanup
    std::vector<std::pair<void*, size_t>> bar_allocations_;
};
```

### Integration with GraphOrchestrator

The GraphOrchestrator needs to allocate ROCm tensors that participate in collectives from the BAR region:

```cpp
// DeviceGraphBufferManager.cpp - BAR-aware allocation
std::unique_ptr<ITensor> DeviceGraphBufferManager::allocateBuffer(
    const BufferDescriptor& desc,
    DeviceId device)
{
    if (device.is_rocm() && desc.participates_in_collective) {
        // Use BAR region for collective buffers
        auto bar_backend = getBarBackend();
        if (bar_backend && bar_backend->isPCIeBarActive()) {
            auto [rocm_ptr, bar_offset] = bar_backend->allocateInBarRegion(desc.size_bytes);
            
            // Create tensor wrapping BAR-region memory
            auto tensor = tensor_factory_.wrapExternalMemory(
                rocm_ptr, desc.shape, desc.dtype, device);
            
            // Register with backend
            bar_backend->registerBuffer(desc.collective_id, device, rocm_ptr, 
                                         desc.size_bytes, bar_offset);
            
            return tensor;
        }
    }
    
    // Standard allocation for non-collective buffers
    return tensor_factory_.create(desc.dtype, desc.shape, device);
}
```

### Alternative: Pre-Allocated Collective Buffers

For simpler integration, the backend can pre-allocate a pool of collective buffers:

```cpp
// PCIeBARBackend.cpp
bool PCIeBARBackend::initialize(const DeviceGroup& group) {
    // ... existing init ...
    
    // Pre-allocate standard collective buffer sizes
    // These are the buffers that will be used for allreduce operations
    constexpr size_t SMALL_BUFFER_SIZE = 4 * 1024 * 1024;   // 4MB
    constexpr size_t MEDIUM_BUFFER_SIZE = 64 * 1024 * 1024;  // 64MB
    constexpr size_t LARGE_BUFFER_SIZE = 256 * 1024 * 1024;  // 256MB
    
    preallocated_rocm_buffers_.small = allocateInBarRegion(SMALL_BUFFER_SIZE);
    preallocated_rocm_buffers_.medium = allocateInBarRegion(MEDIUM_BUFFER_SIZE);
    preallocated_rocm_buffers_.large = allocateInBarRegion(LARGE_BUFFER_SIZE);
    
    return true;
}

// Collective uses pre-allocated buffer
bool PCIeBARBackend::allreduce(void* cuda_buffer, size_t count, ...) {
    size_t bytes = count * datatypeSize(dtype);
    
    // Select appropriate pre-allocated ROCm buffer
    auto& rocm_buf = selectBuffer(bytes);
    
    // Copy user's ROCm data to our managed buffer (if needed)
    // ... or require user to use our managed buffers directly
}
```

### allreduceRegistered Implementation

```cpp
bool PCIeBARBackend::allreduceRegistered(
    const std::string& collective_id,
    size_t count,
    CollectiveDataType dtype,
    CollectiveOp op)
{
    auto it = registered_collectives_.find(collective_id);
    if (it == registered_collectives_.end()) {
        LOG_ERROR("No buffers registered for collective: " << collective_id);
        return false;
    }
    
    const auto& coll = it->second;
    if (!coll.cuda_registered || !coll.rocm_registered) {
        LOG_ERROR("Incomplete buffer registration for collective: " << collective_id);
        return false;
    }
    
    size_t bytes = count * datatypeSize(dtype);
    
    // Use registered BAR offset instead of hardcoded 0
    size_t rocm_offset = coll.rocm_buffer.bar_offset;
    void* cuda_buffer = coll.cuda_buffer.ptr;
    
    // Step 1: Read ROCm data via BAR at correct offset
    if (!transferROCmtoCUDA(rocm_offset, cuda_temp_buffer_, bytes)) {
        LOG_ERROR("Failed to read ROCm data from BAR offset " << rocm_offset);
        return false;
    }
    
    // Step 2: Reduce on CUDA
    if (!reduceOnCUDA(cuda_buffer, cuda_buffer, cuda_temp_buffer_, count, dtype, op)) {
        return false;
    }
    
    // Step 3: Write result back to ROCm at correct offset
    if (!transferCUDAtoROCm(cuda_buffer, rocm_offset, bytes)) {
        return false;
    }
    
    return synchronize();
}
```

---

## Migration Path

### Phase 1: Add Interface (Backward Compatible)

1. Add `IBufferRegistration` interface to `ICollectiveBackend.h`
2. Implement in `PCIeBARBackend` with registration storage
3. Add `allreduceRegistered()` alongside existing `allreduce()`
4. Existing tests continue to work (they'll remain skipped)

### Phase 2: Orchestrator Integration

1. Extend `AllreduceStage` to optionally carry a `collective_id`
2. Add `registerCollectiveBuffers()` to `GraphOrchestrator`
3. DeviceGraphExecutor calls registration before execution

### Phase 3: Update Tests

1. Update integration tests to use buffer registration
2. Un-skip the 4 currently skipped tests
3. Add new tests for registration API

### Phase 4: Deprecate Old API

1. Mark `allreduce(buffer, ...)` as deprecated for PCIeBARBackend
2. Eventually remove single-buffer overload for backends requiring registration

---

## Test Strategy

### Unit Tests

```cpp
// Test__PCIeBARBackend.cpp additions

TEST(Test__PCIeBARBackend, RegisterBuffer_CUDA)
{
    PCIeBARBackend backend;
    DeviceGroup group = createMixedGroup();
    ASSERT_TRUE(backend.initialize(group));
    
    void* cuda_ptr = /* allocate CUDA memory */;
    ASSERT_TRUE(backend.registerBuffer("test_coll", DeviceId::cuda(0), cuda_ptr, 1024));
    
    auto info = backend.getBuffer("test_coll", DeviceId::cuda(0));
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->ptr, cuda_ptr);
    EXPECT_TRUE(info->is_primary);
}

TEST(Test__PCIeBARBackend, RegisterBuffer_ROCm)
{
    PCIeBARBackend backend;
    // ... registration with BAR offset verification
}

TEST(Test__PCIeBARBackend, AllReduceRegistered_TwoDevices)
{
    // Full end-to-end test with registered buffers
}
```

### Integration Tests

```cpp
// Test__PCIeBARBackendIntegration.cpp - Update skipped tests

TEST(Test__PCIeBARBackendIntegration, AllReduceSmallBuffer_WithRegistration)
{
    // 1. Allocate CUDA and ROCm buffers
    // 2. Register both with backend
    // 3. Initialize with test data
    // 4. Call allreduceRegistered()
    // 5. Verify results
}
```

---

## Appendix: Alternative Approaches

### A. Unified Buffer Allocator

Instead of registering arbitrary allocations, provide a unified allocator:

```cpp
class CollectiveBufferAllocator {
public:
    // Allocate buffer that's automatically registered
    void* allocate(DeviceId device, size_t size, const std::string& collective_id);
    void free(void* ptr);
};
```

**Why not chosen**: Requires changing allocation flow for all tensors, invasive.

### B. IPC Memory Handles

Use CUDA IPC and HIP IPC to share memory handles:

```cpp
// CUDA side
cudaIpcMemHandle_t handle;
cudaIpcGetMemHandle(&handle, buffer);

// ROCm side - import handle
void* mapped_ptr;
hipIpcOpenMemHandle(&mapped_ptr, handle, ...);
```

**Why not chosen**: CUDA IPC to ROCm is not supported; BAR mapping is the only path.

### C. Page-Locked Host Memory Intermediary

Use pinned host memory as intermediary, avoiding BAR offset issues:

```cpp
// ROCm → Pinned Host → CUDA
hipMemcpy(pinned_host, rocm_ptr, size, hipMemcpyDeviceToHost);
cudaMemcpy(cuda_ptr, pinned_host, size, cudaMemcpyHostToDevice);
```

**Why not chosen**: Much slower than direct PCIe BAR (~1.5 GB/s vs ~2.65 GB/s).

---

## Document History

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2026-01-XX | 1.0 | David Sanftenberg | Initial design |
