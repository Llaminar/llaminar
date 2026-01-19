# ROCm Performance Investigation and Optimization Plan

**Date:** January 19, 2026  
**Status:** Investigation Complete, Ready for Implementation  
**Test Case:** `v2_integration_parity_qwen2_rocm_vs_pytorch`

---

## Executive Summary

The ROCm parity test takes ~60+ seconds to execute what should be a sub-second inference. Investigation revealed three root causes:

1. **Output buffers are uploaded unnecessarily** (~248 MB of garbage data per execution)
2. **Synchronous memcpy blocks on all pending GPU work** (first D2H waits for all H2D + kernels)
3. **Non-GEMM weights (norms, biases) are not preloaded** (uploaded on-demand during execution)

The primary fix should focus on Issue #1, which will eliminate the majority of the performance problem.

---

## Table of Contents

1. [Investigation Timeline](#investigation-timeline)
2. [System Architecture](#system-architecture)
3. [Key Files Involved](#key-files-involved)
4. [Performance Metrics](#performance-metrics)
5. [Root Cause Analysis](#root-cause-analysis)
6. [Recommended Fixes](#recommended-fixes)
7. [Implementation Priority](#implementation-priority)
8. [Testing Approach](#testing-approach)

---

## Investigation Timeline

### Phase 1: Initial Benchmarking

Created microbenchmarks to isolate HIP memcpy performance:

```bash
# Location of benchmarks
tests/v2/microbench/hip_memcpy_benchmark.cpp   # Bandwidth test
tests/v2/microbench/hip_transfer_overhead.cpp  # Lifecycle overhead test
```

**Key Finding:** HIP memcpy bandwidth is NORMAL (~3.5 GB/s for MI60 PCIe 3.0)

### Phase 2: Debug Log Analysis

Enabled detailed logging and captured execution trace:

```bash
# Command used
LLAMINAR_LOG_LEVEL=DEBUG ./build_v2_integration/tests/v2/v2_integration_parity_qwen2_rocm_vs_pytorch 2>&1 | tee /tmp/parity_debug.log
```

**Log file:** `/tmp/parity_debug.log` (~60 second capture)

### Phase 3: Timeline Reconstruction

Key timestamps from execution:

| Time | Event |
|------|-------|
| 12:17:45.711 | WeightPreloader starts (169 GEMM weights) |
| 12:17:50.834 | WeightPreloader completes |
| 12:17:51.400 | First stage execution begins |
| 12:17:51.406 | First output buffer uploaded (14 MB of garbage!) |
| 12:17:51.422 | FusedQKVGEMMStage kernel launches |
| 12:17:57.240 | FusedQKVGEMMStage "completes" (5.8 second gap!) |

The 5.8 second gap is the kernel waiting for ALL prior H2D uploads to complete.

---

## System Architecture

### GPU Tensor Coherence Model

The coherence system tracks data validity across CPU and GPU:

```
┌─────────────────────────────────────────────────────────────────┐
│                     CPUTensorBase                               │
├─────────────────────────────────────────────────────────────────┤
│  host_valid_: bool      - Is CPU data current?                  │
│  device_valid_: bool    - Is GPU data current?                  │
│  gpu_data_ptr_: void*   - GPU memory pointer                    │
│  gpu_device_: DeviceId  - Which GPU device                      │
├─────────────────────────────────────────────────────────────────┤
│  ensureOnDevice(dev)    - Upload to GPU if needed               │
│  ensureOnHost()         - Download from GPU if needed           │
│  mark_device_dirty()    - Mark GPU as authoritative             │
└─────────────────────────────────────────────────────────────────┘
```

### Stage Coherence Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    GraphExecutor::executeStage()                 │
├──────────────────────────────────────────────────────────────────┤
│  1. StageCoherence::ensureInputsOnDevice()                       │
│     - For each input tensor:                                     │
│       - Call tensor->ensureOnDevice(target_device)               │
│       - This uploads host→device if device_valid_==false         │
│                                                                  │
│  2. StageCoherence::ensureOutputsAllocated()  ← PROBLEM HERE     │
│     - For each output tensor:                                    │
│       - Call tensor->ensureOnDevice(target_device)               │
│       - This UPLOADS garbage data even though kernel will write! │
│                                                                  │
│  3. Stage::execute() - Run the GPU kernel                        │
│                                                                  │
│  4. StageCoherence::markOutputsDirty()                           │
│     - For each output tensor:                                    │
│       - Call tensor->mark_device_dirty()                         │
└──────────────────────────────────────────────────────────────────┘
```

### Weight Preloading Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    WeightPreloader                               │
├──────────────────────────────────────────────────────────────────┤
│  preloadForDevice(DeviceType::ROCm)                              │
│    │                                                             │
│    ├─► For each weight in cache:                                 │
│    │     if (isGemmWeight(name))  ← Only GEMM weights!           │
│    │       packWeight(tensor, device)                            │
│    │         │                                                   │
│    │         ├─► KernelFactory::getOrCreateGemm()                │
│    │         │     - Creates ROCmQuantisedGemmKernel             │
│    │         │     - Packs INT8 weights on CPU                   │
│    │         │                                                   │
│    │         └─► kernel->ensureWeightsConverted()                │
│    │               - Uploads packed weights to GPU               │
│    │               - Uses synchronous hipMemcpy                  │
│    │                                                             │
│    └─► Skipped weights (NOT preloaded):                          │
│          - *_norm.weight (norms)                                 │
│          - *.bias (biases)                                       │
│          - token_embd* (embeddings)                              │
└──────────────────────────────────────────────────────────────────┘
```

### ROCm Backend Transfer Methods

```cpp
// File: src/v2/backends/rocm/ROCmBackend.cpp

// Host → Device (line 74)
bool ROCmBackend::hostToDevice(void* dst, const void* src, size_t bytes, int device_id) {
    hipSetDevice(device_id);
    hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice);  // SYNCHRONOUS!
    return true;
}

// Device → Host (line 57)
bool ROCmBackend::deviceToHost(void* dst, const void* src, size_t bytes, int device_id) {
    hipSetDevice(device_id);
    hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost);  // SYNCHRONOUS!
    return true;
}
```

**Critical:** Both methods use synchronous `hipMemcpy`. The D2H transfer will block until ALL previous operations (H2D uploads + kernels) complete.

---

## Key Files Involved

### Coherence System

| File | Purpose |
|------|---------|
| `src/v2/tensors/cpu/CPUTensorBase.h` | Coherence state tracking |
| `src/v2/tensors/cpu/CPUTensorBase.cpp` | `ensureOnDevice()`, `ensureOnHost()` |
| `src/v2/execution/StageCoherence.h` | Stage-level coherence helpers |
| `src/v2/execution/StageCoherence.cpp` | `ensureInputsOnDevice()`, `ensureOutputsAllocated()` |

### Backend System

| File | Purpose |
|------|---------|
| `src/v2/backends/rocm/ROCmBackend.h` | ROCm backend interface |
| `src/v2/backends/rocm/ROCmBackend.cpp` | `hostToDevice()`, `deviceToHost()` |
| `src/v2/backends/IBackend.h` | Backend interface definition |

### Weight Loading

| File | Purpose |
|------|---------|
| `src/v2/loaders/WeightPreloader.h` | Weight preloading interface |
| `src/v2/loaders/WeightPreloader.cpp` | `preloadForDevice()`, GEMM-only filtering |
| `src/v2/loaders/WeightManager.cpp` | `isGemmWeight()` classification |
| `src/v2/execution/GraphSchema.h` | `isNonGemmWeight()` definition |

### Kernel System

| File | Purpose |
|------|---------|
| `src/v2/kernels/rocm/ROCmQuantisedGemmKernel.cpp` | INT8 GEMM kernel, weight packing |
| `src/v2/kernels/KernelFactory.cpp` | Kernel creation and caching |

### Execution Pipeline

| File | Purpose |
|------|---------|
| `src/v2/execution/GraphExecutor.cpp` | Stage execution loop |
| `src/v2/execution/compute_stages/stages/FusedQKVGEMMStage.cpp` | QKV projection stage |
| `src/v2/execution/compute_stages/stages/FusedGateUpGEMMStage.cpp` | FFN gate/up projection |

### Test Files

| File | Purpose |
|------|---------|
| `tests/v2/integration/parity/Test__ParityQwen2_ROCm_vs_PyTorch.cpp` | Failing test |
| `tests/v2/microbench/hip_memcpy_benchmark.cpp` | Bandwidth microbenchmark |
| `tests/v2/microbench/hip_transfer_overhead.cpp` | Overhead microbenchmark |

---

## Performance Metrics

### Hardware Configuration

- **GPU:** AMD Instinct MI60 (2× available, using device 1)
- **Interface:** PCIe 3.0
- **Measured bandwidth:** ~3.5 GB/s (normal for PCIe 3.0)

### Benchmark Results

#### HIP Transfer Bandwidth (hip_memcpy_benchmark.cpp)

| Size | H2D Sync | D2H Sync | Bandwidth |
|------|----------|----------|-----------|
| 4 KB | 0.006 ms | 0.005 ms | 0.7 GB/s |
| 1 MB | 0.28 ms | 0.29 ms | 3.5 GB/s |
| 14 MB | 4.0 ms | 4.0 ms | 3.5 GB/s |
| 76 MB | 22 ms | 22 ms | 3.4 GB/s |
| 256 MB | 74 ms | 74 ms | 3.4 GB/s |

#### HIP Overhead Analysis (hip_transfer_overhead.cpp)

| Operation | 14 MB | 76 MB |
|-----------|-------|-------|
| `hipHostMalloc` (pinning) | 4,131 µs | 28,102 µs |
| `hipMalloc` (device alloc) | 1,050 µs | 1,102 µs |
| D2H after idle | 4,105 µs | 21,680 µs |
| D2H after heavy kernel | 11,538 µs | varies |
| Pin + memcpy combo | 8,392 µs | 49,555 µs |

**Key insight:** `hipDeviceSynchronize()` overhead is trivial (~0.15 µs). The sync cost comes from waiting for actual work to complete.

### Execution-Time Transfer Analysis

From `/tmp/parity_debug.log`:

| Phase | Uploads | Bytes | Time |
|-------|---------|-------|------|
| Weight preload | 169 | ~316 MB | 12:17:45 - 12:17:50 (~5s) |
| Execution | 22 | ~248 MB | 12:17:51+ |

#### Execution-Time Upload Breakdown

| Size | Count | Total | Description |
|------|-------|-------|-------------|
| 512 B | 2 | 1 KB | Small buffers |
| 3,584 B | 3 | 10 KB | Norm weights (896 dims × 4 bytes) |
| 64,512 B | 2 | 126 KB | Intermediate buffers |
| 451,584 B | 2 | 880 KB | Bias tensors |
| 2,097,152 B | 3 | 6 MB | Activation buffers |
| 2,451,456 B | 2 | 4.7 MB | Activation buffers |
| 14,680,064 B | 6 | 84 MB | Output buffers |
| 79,691,776 B | 2 | 152 MB | FFN output buffers |
| **TOTAL** | 22 | **~248 MB** | |

**The 14 MB and 76 MB uploads are OUTPUT BUFFERS containing garbage data that will be overwritten by kernels!**

---

## Root Cause Analysis

### Issue #1: Unnecessary Output Buffer Uploads (CRITICAL)

**Location:** `src/v2/execution/StageCoherence.cpp:131`

**Problem:** `ensureOutputsAllocated()` calls `ensureOnDevice()` for output tensors. This method uploads host data to GPU even though:
1. The output buffer contains garbage (uninitialized or stale data)
2. The GPU kernel will immediately overwrite it

**Evidence:**
```
[12:17:59.197] Allocating output buffer 'output_gate' on ROCm:1
[12:17:59.220] Uploaded 79691776 bytes to device ROCm:1  ← 76 MB of garbage!
```

**Impact:** ~248 MB of unnecessary uploads per inference pass.

### Issue #2: Synchronous Transfer Blocking (MODERATE)

**Location:** `src/v2/backends/rocm/ROCmBackend.cpp:57,74`

**Problem:** Both `hostToDevice()` and `deviceToHost()` use synchronous `hipMemcpy()`. When `ensureOnHost()` is called to download results, it blocks until ALL previous operations complete.

**Evidence:**
```
[12:17:51.422] FusedQKVGEMMStage Execute (kernel launched)
[12:17:57.240] FusedQKVGEMMStage GPU execution complete   ← 5.8 SECOND GAP!
```

The 5.8 seconds is waiting for:
- All 169 weight H2D uploads (queued but not yet complete)
- The GEMM kernel to execute

**Impact:** Artificial serialization, prevents overlapping computation with data transfer.

### Issue #3: Non-GEMM Weights Not Preloaded (MINOR)

**Location:** `src/v2/loaders/WeightPreloader.cpp:167`

**Problem:** `preloadForDevice()` filters with `isGemmWeight()`, which excludes:
- `*_norm.weight` (norms) - 3.5 KB each
- `*.bias` (biases)
- `token_embd*` (embeddings)

These are uploaded on-demand during execution.

**Evidence:**
```
[12:17:51.400] Uploading input 'gamma' to ROCm:1
[12:17:51.400] Uploaded 3584 bytes to device ROCm:1
```

**Impact:** Minor - these are small tensors (~3.5 KB norms). But uploading during execution adds latency and complexity.

---

## Recommended Fixes

### Fix #1: Add `allocateOnDevice()` Method (HIGH PRIORITY)

**Objective:** Allocate GPU memory without uploading host data.

**Implementation:**

1. Add new method to `CPUTensorBase`:

```cpp
// In src/v2/tensors/cpu/CPUTensorBase.h
bool allocateOnDevice(DeviceId target_device);  // Allocate only, no upload

// In src/v2/tensors/cpu/CPUTensorBase.cpp
bool CPUTensorBase::allocateOnDevice(DeviceId target_device)
{
    // Same as ensureOnDevice() but skip the upload step
    // Just allocate GPU memory and set device_valid_ = false
    // The kernel will write to the buffer and mark_device_dirty() will be called
    
    IBackend* target_backend = getBackendForDevice(target_device);
    if (!target_backend) return false;
    
    size_t bytes = byte_size();
    if (!gpu_data_ptr_) {
        gpu_data_ptr_ = target_backend->allocate(bytes, device_id);
        gpu_device_ = target_device;
        device_valid_ = false;  // Not uploaded - kernel will write
        ensureHostPinned();     // Pin for future D2H transfer
    }
    return gpu_data_ptr_ != nullptr;
}
```

2. Update `StageCoherence::ensureOutputsAllocated()`:

```cpp
// In src/v2/execution/StageCoherence.cpp
bool ensureOutputsAllocated(const std::vector<CoherenceBuffer>& outputs, DeviceId device)
{
    for (const auto& buf : outputs) {
        auto* tensor_base = dynamic_cast<CPUTensorBase*>(buf.tensor);
        if (!tensor_base) continue;
        
        // Use allocateOnDevice instead of ensureOnDevice
        if (!tensor_base->allocateOnDevice(device)) {
            LOG_ERROR("Failed to allocate output buffer");
            return false;
        }
    }
    return true;
}
```

**Expected Impact:** Eliminate ~248 MB of unnecessary uploads.

### Fix #2: Preload All GPU Weights (MEDIUM PRIORITY)

**Objective:** Include norms, biases, and embeddings in preloading.

**Implementation:**

Option A: Remove the GEMM-only filter:
```cpp
// In src/v2/loaders/WeightPreloader.cpp:167
// Change:
if (weight_manager_->isGemmWeight(name))
// To:
if (true)  // Preload ALL weights
```

Option B: Add separate `preloadNonGemmWeights()` method:
```cpp
bool WeightPreloader::preloadNonGemmWeights(DeviceId device)
{
    for (const auto& [name, tensor] : weight_manager_->cache_) {
        if (!weight_manager_->isGemmWeight(name)) {
            tensor->ensureOnDevice(device);
        }
    }
    return true;
}
```

**Expected Impact:** Eliminate ~10 KB of on-demand uploads during execution.

### Fix #3: Async Transfers with Events (LOW PRIORITY - COMPLEX)

**Objective:** Use async memcpy with HIP events for fine-grained synchronization.

**Implementation outline:**

1. Add async transfer methods to `ROCmBackend`:
```cpp
hipEvent_t hostToDeviceAsync(void* dst, const void* src, size_t bytes, hipStream_t stream);
hipEvent_t deviceToHostAsync(void* dst, const void* src, size_t bytes, hipStream_t stream);
```

2. Track completion events per tensor
3. Synchronize only when data is actually needed

**Expected Impact:** Enable overlap of computation and data transfer. Complex implementation.

---

## Implementation Priority

| Priority | Fix | Complexity | Impact | Files Changed |
|----------|-----|------------|--------|---------------|
| 1 | `allocateOnDevice()` | Low | HIGH (~248 MB saved) | CPUTensorBase.cpp/h, StageCoherence.cpp |
| 2 | Preload all weights | Low | Low (~10 KB saved) | WeightPreloader.cpp |
| 3 | Async transfers | High | Medium | ROCmBackend.cpp/h, CPUTensorBase.cpp |

**Recommendation:** Implement Fix #1 first. It's low complexity and high impact.

---

## Testing Approach

### Verification Steps

1. **Rebuild integration tests:**
```bash
cmake --build build_v2_integration --parallel
```

2. **Run parity test with timing:**
```bash
time ./build_v2_integration/tests/v2/v2_integration_parity_qwen2_rocm_vs_pytorch
```

3. **Expected improvement:** Test should complete in <10 seconds (vs 60+ seconds currently).

4. **Verify correctness:** Test should still PASS (numerical parity with PyTorch).

### Debug Verification

Run with debug logging to verify no unnecessary uploads:
```bash
LLAMINAR_LOG_LEVEL=DEBUG ./build_v2_integration/tests/v2/v2_integration_parity_qwen2_rocm_vs_pytorch 2>&1 | grep -E "Uploaded|Allocating output"
```

**Before fix:** Should see "Uploaded 79691776 bytes" for output buffers
**After fix:** Should see "Allocated output buffer" without upload

---

## Appendix: Debug Commands

### Analyze upload patterns
```bash
grep -E "Uploaded [0-9]+ bytes" /tmp/parity_debug.log | sed 's/.*Uploaded \([0-9]*\) bytes.*/\1/' | sort -n | uniq -c
```

### Count uploads by phase
```bash
# Preload phase (12:17:45-50)
grep -E "12:17:4[5-9].*Uploaded" /tmp/parity_debug.log | wc -l

# Execution phase (12:17:51+)
grep -E "12:17:5[1-9].*Uploaded" /tmp/parity_debug.log | wc -l
```

### Calculate total upload size
```bash
grep -oE "Uploaded [0-9]+ bytes" /tmp/parity_debug.log | awk '{sum += $2} END {print sum/1024/1024 " MB"}'
```

### Find output buffer uploads
```bash
grep -B5 "Uploaded 79691776" /tmp/parity_debug.log
```

---

## References

- [ROCm HIP API Documentation](https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-GUIDE.html)
- Llaminar coherence system: `docs/v2/GPU_COHERENCE.md` (if exists)
- Project instructions: `.github/copilot-instructions.md`
