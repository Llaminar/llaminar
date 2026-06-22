# GPU-Resident Execution Optimization Project Plan

**Author:** GitHub Copilot  
**Date:** January 19, 2026  
**Status:** Draft  
**Branch:** feature/cuda-kernels

## Executive Summary

This document outlines a phased approach to optimize Llaminar V2's tensor coherence system for GPU-resident execution. Analysis reveals the current architecture **already supports GPU-resident execution** - data stays on GPU between stages when the coherence system operates correctly. The focus shifts to:

1. **Verifying** the system works as designed
2. **Eliminating wasteful H2D copies** when allocating output buffers  
3. **Adding observability** for coherence debugging
4. **Enabling zero-copy snapshots** via mapped memory when debugging

---

## Problem Statement

### Initial Concern
Unnecessary GPU↔CPU data transfers between inference stages, causing performance degradation.

### Investigation Findings

The current coherence flow is:

```
DeviceGraphExecutor::executeNode()
├─ cohereInputs()      → ensureOnDevice() for inputs
├─ cohereOutputs()     → ensureOnDevice() for outputs (allocates GPU buffer)
├─ stage->execute()    → Kernel writes to gpu_data_ptr()
├─ markOutputsDirty()  → Sets device_valid_=true, host_valid_=false (NO D2H!)
└─ [Next stage]        → cohereInputs() sees data already on GPU, skips H2D
```

**Key finding:** D2H copies only occur when:
1. `fp32_data()` is called (triggers `ensureOnHost()`) - happens for final logits only
2. `snapshot_callback` is set (tests/debugging only)
3. Kernel falls back to CPU path (indicates coherence failure)

---

## Current Architecture

### Coherence State Machine

```
┌─────────────────────────────────────────────────────────────────┐
│                    Tensor Coherence States                      │
├─────────────────────────────────────────────────────────────────┤
│  HOST_AUTHORITATIVE:   host_valid_=true,  device_valid_=false   │
│  DEVICE_AUTHORITATIVE: host_valid_=false, device_valid_=true    │
│  SYNCED:               host_valid_=true,  device_valid_=true    │
│  INVALID:              host_valid_=false, device_valid_=false   │
└─────────────────────────────────────────────────────────────────┘
```

### Key Files

| File | Purpose |
|------|---------|
| `src/v2/tensors/cpu/CPUTensorBase.cpp` | `ensureOnDevice()`, `ensureOnHost()`, coherence logic |
| `src/v2/execution/StageCoherence.cpp` | `cohereInputs()`, `cohereOutputs()`, `markOutputsDirty()` |
| `src/v2/execution/DeviceGraphExecutor.cpp` | Orchestrates coherence at stage boundaries |
| `src/v2/kernels/rocm/ROCmQuantisedGemmKernel.cpp` | GPU/CPU path selection via `use_gpu_path` |

### GPU Path Detection (ROCmQuantisedGemmKernel)

```cpp
// Lines 658-660 in ROCmQuantisedGemmKernel.cpp
const float *d_input = static_cast<const float *>(A_fp32->gpu_data_ptr());
float *d_output = static_cast<float *>(C_fp32->gpu_data_ptr());
const bool use_gpu_path = (d_input != nullptr) && (d_output != nullptr);
```

If `use_gpu_path == false`, kernel falls back to CPU path which does explicit D2H at line ~1009.

### `mark_device_dirty()` Behavior

```cpp
virtual void mark_device_dirty()
{
    device_valid_ = true;      // Device just got written to
    host_valid_ = is_mapped_;  // Non-mapped: host stale; Mapped: host also valid
}
```

**Note:** For mapped tensors, both host and device remain valid since they share the same physical memory.

---

## Proposed Changes

### Phase 1: Verification & Observability (Low Risk)

**Goal:** Prove the system works correctly and add debugging tools.

#### 1.1 Add Coherence Tracing to Kernels

Add logging to `ROCmQuantisedGemmKernel::multiply_tensor()` to confirm GPU path usage:

```cpp
// After use_gpu_path calculation
LOG_INFO("[ROCmGEMM] use_gpu_path=" << use_gpu_path 
         << " d_input=" << (void*)d_input 
         << " d_output=" << (void*)d_output);
```

#### 1.2 Integration Test: GPU-Resident Data Flow

Create test that verifies:
- Output tensor's `gpu_data_ptr()` is non-null after `cohereOutputs()`
- No D2H copy occurs between stages (mock backend tracks transfers)
- D2H only on final `fp32_data()` call

**Test file:** `tests/v2/integration/Test__GpuResidentDataFlow.cpp`

#### 1.3 Environment Variable for Transfer Counting

Add `LLAMINAR_COUNT_TRANSFERS=1` to track H2D/D2H operations per forward pass.

---

### Phase 2: Output Buffer Optimization (Medium Risk)

**Goal:** Eliminate wasteful H2D uploads when allocating output buffers.

#### 2.1 Add `allocateOnDevice()` Method

New method in `CPUTensorBase` that only allocates GPU memory without uploading host data:

```cpp
// CPUTensorBase.h - new method declaration
/**
 * @brief Allocate GPU buffer without uploading host data
 * @param target_device The GPU device to allocate on
 * @return true if allocation succeeded
 * @note Use for OUTPUT tensors where kernel will overwrite contents
 */
bool allocateOnDevice(DeviceId target_device);
```

```cpp
// CPUTensorBase.cpp - implementation
bool CPUTensorBase::allocateOnDevice(DeviceId target_device) {
    if (!target_device.is_gpu()) {
        LOG_ERROR("[CPUTensorBase::allocateOnDevice] Must be GPU device");
        return false;
    }
    
    // Already allocated on this device?
    if (gpu_data_ptr_ && gpu_device_.has_value() && *gpu_device_ == target_device) {
        return true;
    }
    
    IBackend* backend = getBackendForDevice(target_device);
    if (!backend) {
        return false;
    }
    
    // Free old allocation if on different device
    if (gpu_data_ptr_ && gpu_device_.has_value()) {
        IBackend* old_backend = getBackendForDevice(*gpu_device_);
        old_backend->free(gpu_data_ptr_, gpu_device_->gpu_ordinal());
        gpu_data_ptr_ = nullptr;
    }
    
    // Allocate new buffer (no upload)
    size_t bytes = byte_size();
    gpu_data_ptr_ = backend->allocate(bytes, target_device.gpu_ordinal());
    if (!gpu_data_ptr_) {
        return false;
    }
    
    gpu_device_ = target_device;
    device_valid_ = false;  // Not populated yet - kernel will write
    // host_valid_ unchanged - host data is still valid
    
    return true;
}
```

#### 2.2 Update `cohereOutputs()` to Use New Method

```cpp
// StageCoherence.cpp - modify cohereOutputs()
bool cohereOutputs(const std::vector<CoherenceBuffer> &outputs, DeviceId target_device) {
    // ...existing validation...
    
    for (const auto &buf : outputs) {
        auto *tensor_base = dynamic_cast<CPUTensorBase *>(buf.tensor);
        if (!tensor_base) continue;
        
        // Use allocateOnDevice() for outputs - no wasteful H2D upload
        if (!tensor_base->allocateOnDevice(target_device)) {
            LOG_ERROR("[StageCoherence] Failed to allocate output buffer");
            return false;
        }
    }
    return true;
}
```

#### 2.3 Integration Test: No H2D for Output Buffers

Test that `cohereOutputs()` does NOT trigger `hostToDevice()` call on backend.

---

### Phase 3: Zero-Copy Snapshot Mode (Medium Risk)

**Goal:** When snapshot system is enabled, use mapped memory for zero-copy observability.

#### 3.1 Add Mapped Memory Tensor Factory Option

```cpp
// TensorFactory.h - new option
struct TensorFactoryOptions {
    bool use_mapped_memory = false;  // Use hipHostMallocMapped for zero-copy
    DeviceId target_device;
};

// TensorFactory.cpp
std::unique_ptr<FP32Tensor> TensorFactory::createFP32(
    const std::vector<size_t>& shape,
    const TensorFactoryOptions& opts) 
{
    if (opts.use_mapped_memory && opts.target_device.is_gpu()) {
        return FP32Tensor::createMapped(shape, opts.target_device);
    }
    return std::make_unique<FP32Tensor>(shape, opts.target_device);
}
```

#### 3.2 DeviceGraphBufferManager Mapped Mode

When `LLAMINAR_SNAPSHOT_USE_MAPPED=1`:
- Allocate activation buffers with mapped memory
- Both host and device can access without memcpy
- `ensureOnHost()` is a no-op for mapped tensors

#### 3.3 Integration Test: Mapped Memory Coherence

Test that mapped tensors:
- Return valid `gpu_data_ptr()` immediately after creation
- Return valid `data()` without triggering memcpy
- Kernel writes are visible to host without explicit sync

---

### Phase 4: Performance Validation (Low Risk)

**Goal:** Measure and document performance improvements.

#### 4.1 Benchmark: Before/After Transfer Counts

Run inference with transfer counting enabled, compare:
- H2D transfers per token
- D2H transfers per token
- Total transfer bytes per forward pass

#### 4.2 Benchmark: Decode Latency

Measure tok/s improvement for decode (single token) iterations where transfer overhead is most impactful.

---

## Test Plan

### New Test Files

| Test File | Phase | Purpose |
|-----------|-------|---------|
| `tests/v2/integration/Test__GpuResidentDataFlow.cpp` | 1 | Verify GPU path is taken, no spurious D2H |
| `tests/v2/unit/Test__TensorAllocateOnDevice.cpp` | 2 | Test new `allocateOnDevice()` method |
| `tests/v2/integration/Test__CoherenceOutputAllocation.cpp` | 2 | Verify `cohereOutputs()` skips H2D |
| `tests/v2/integration/Test__MappedMemoryCoherence.cpp` | 3 | Test mapped tensor behavior |

### Mock Requirements

**Existing Mocks (do NOT have transfer tracking):**
- `MockBackendWithDefaults : ICollectiveBackend` - for buffer registration tests
- `MockBackendWithRegistration : ICollectiveBackend` - for collective backend tests  
- Location: `tests/v2/mocks/MockCollectiveBackend.h`

**Existing Coherence Tests:**
- `tests/v2/integration/Test__TensorCoherence.cpp` - 697 lines, comprehensive state machine testing
- `tests/v2/unit/Test__StageCoherence.cpp` - 462 lines, CoherencePolicy and buffer extraction tests

#### New MockBackend Required

```cpp
// tests/v2/mocks/MockBackend.h (NEW FILE)
class MockBackend : public IBackend {
public:
    // Track transfer operations for verification
    struct TransferStats {
        size_t h2d_count = 0;
        size_t d2h_count = 0;
        size_t h2d_bytes = 0;
        size_t d2h_bytes = 0;
    };
    
    TransferStats getTransferStats() const { return stats_; }
    void resetTransferStats() { stats_ = {}; }
    
    bool hostToDevice(void* dst, const void* src, size_t bytes, int device) override {
        stats_.h2d_count++;
        stats_.h2d_bytes += bytes;
        // ... existing mock implementation
    }
    
    bool deviceToHost(void* dst, const void* src, size_t bytes, int device) override {
        stats_.d2h_count++;
        stats_.d2h_bytes += bytes;
        // ... existing mock implementation
    }
    
private:
    TransferStats stats_;
};
```

### Test Case: GPU-Resident Data Flow (Phase 1)

```cpp
TEST(Test__GpuResidentDataFlow, NoD2HBetweenStages) {
    // Setup: Create mock backend with transfer tracking
    auto mock_backend = std::make_shared<MockBackend>();
    BackendRegistry::instance().registerBackend(DeviceType::ROCM, mock_backend);
    
    // Create two sequential stages that share an intermediate buffer
    auto intermediate = std::make_unique<FP32Tensor>({128, 896});
    
    // Stage 1: Writes to intermediate
    MockGemmStage stage1;
    stage1.setOutput(intermediate.get());
    
    // Stage 2: Reads from intermediate  
    MockGemmStage stage2;
    stage2.setInput(intermediate.get());
    
    DeviceId gpu{DeviceType::ROCM, 0};
    
    // Execute stage 1 with coherence
    auto inputs1 = extractInputBuffers(stage1.getDumpInfo());
    auto outputs1 = extractOutputBuffers(stage1.getDumpInfo());
    cohereInputs(inputs1, gpu);
    cohereOutputs(outputs1, gpu);
    stage1.execute();
    markOutputsDirty(outputs1);
    
    mock_backend->resetTransferStats();  // Reset counters
    
    // Execute stage 2 with coherence
    auto inputs2 = extractInputBuffers(stage2.getDumpInfo());
    auto outputs2 = extractOutputBuffers(stage2.getDumpInfo());
    cohereInputs(inputs2, gpu);  // Should NOT trigger H2D - data already on GPU
    cohereOutputs(outputs2, gpu);
    stage2.execute();
    markOutputsDirty(outputs2);
    
    // Verify: NO transfers between stages
    auto stats = mock_backend->getTransferStats();
    EXPECT_EQ(stats.h2d_count, 0) << "Unexpected H2D transfer between stages";
    EXPECT_EQ(stats.d2h_count, 0) << "Unexpected D2H transfer between stages";
    
    // Now access data on host - should trigger exactly one D2H
    mock_backend->resetTransferStats();
    const float* host_data = intermediate->fp32_data();
    ASSERT_NE(host_data, nullptr);
    
    stats = mock_backend->getTransferStats();
    EXPECT_EQ(stats.d2h_count, 1) << "Expected exactly one D2H on fp32_data() access";
}
```

---

## Implementation Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Verification | 1-2 days | None |
| Phase 2: Output Optimization | 2-3 days | Phase 1 tests passing |
| Phase 3: Mapped Memory | 3-4 days | Phase 2 complete |
| Phase 4: Benchmarking | 1 day | All phases complete |

**Total Estimated Effort:** 7-10 days

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Coherence bugs cause data corruption | Low | High | Comprehensive test coverage with MockBackend |
| Performance regression | Low | Medium | Benchmark before/after each phase |
| Mapped memory not available on all GPUs | Medium | Low | Fall back to regular allocation |
| Breaking existing tests | Low | Medium | Run full test suite after each change |

---

## Success Criteria

1. **Phase 1:** All integration tests pass showing GPU path is taken
2. **Phase 2:** Zero H2D transfers for output buffer allocation
3. **Phase 3:** Snapshot system works with mapped memory, no explicit syncs
4. **Phase 4:** Measurable improvement in decode tok/s (target: >5%)

---

## Appendix: Key Code Locations

### CPUTensorBase Coherence Methods
- `ensureOnDevice()`: Lines 770-935 in `src/v2/tensors/cpu/CPUTensorBase.cpp`
- `ensureOnHost()`: Lines 937-1010 in `src/v2/tensors/cpu/CPUTensorBase.cpp`
- `mark_device_dirty()`: Lines 788-794 in `src/v2/tensors/cpu/CPUTensors.h`

### StageCoherence Functions
- `cohereInputs()`: Lines 22-99 in `src/v2/execution/StageCoherence.cpp`
- `cohereOutputs()`: Lines 104-175 in `src/v2/execution/StageCoherence.cpp`
- `markOutputsDirty()`: Lines 177-210 in `src/v2/execution/StageCoherence.cpp`

### DeviceGraphExecutor Coherence Integration
- Entry coherence: Lines 645-680 in `src/v2/execution/DeviceGraphExecutor.cpp`
- Exit marking: Lines 756-768 in `src/v2/execution/DeviceGraphExecutor.cpp`
- Snapshot callback: Lines 818-826 in `src/v2/execution/DeviceGraphExecutor.cpp`

### Kernel GPU Path Selection
- `use_gpu_path`: Lines 658-660 in `src/v2/kernels/rocm/ROCmQuantisedGemmKernel.cpp`
- CPU path D2H: Lines 1005-1015 in `src/v2/kernels/rocm/ROCmQuantisedGemmKernel.cpp`
