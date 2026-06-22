# Debug Handover: RCCL LocalTP Multi-GPU Memory Access Fault

**Date**: January 29, 2026  
**Status**: In Progress  
**Branch**: `tensor-parallel`

---

## Test Under Investigation

**Test File**: `tests/v2/integration/parity/qwen2/Test__Qwen2_LocalTP_RCCL_vs_PyTorch.cpp`  
**Test Name**: `Test__Qwen2_LocalTP_RCCL_vs_PyTorch.PrefillParity_LocalTP`  
**Binary**: `build_v2_integration/tests/v2/v2_integration_parity_qwen2_localtp_rccl_vs_pytorch`

```bash
# Run the test
timeout 300 ./build_v2_integration/tests/v2/v2_integration_parity_qwen2_localtp_rccl_vs_pytorch \
  --gtest_filter="*PrefillParity*"
```

---

## Error Condition

### The Crash
```
Memory access fault by GPU node-3 (Agent handle: 0x8616ec0) on address 0x70fa059a6000. 
Reason: Page not present or supervisor privilege.
```

### HSA Node Mapping (CRITICAL)
The HSA node IDs do NOT match ROCm device ordinals:
- **Node 0**: CPU
- **Node 2**: GPU 0 (ROCm:0) - AMD Instinct MI60/MI50
- **Node 3**: GPU 1 (ROCm:1) - AMD Instinct MI60/MI50
- **Node 4**: GPU 2 (ROCm:2) - AMD Instinct MI60/MI50

**Therefore**: "GPU node-3" in error messages = **ROCm:1** (not ROCm:2!)

### Crash Location
The crash occurs immediately after embedding table upload on both devices:
```
[ROCm:0] [ROCmEmbeddingKernelT.cpp:307] Uploaded dequantized embedding table: 151936x896 (519 MB)
[ROCm:1] [ROCmEmbeddingKernelT.cpp:307] Uploaded dequantized embedding table: 151936x896 (519 MB)
Memory access fault by GPU node-3...
```

This suggests ROCm:1 is trying to access memory that belongs to ROCm:0 (or freed memory).

---

## Latest Debug Session (January 29, 2026)

### Enhanced Logging Added

Added debug logging in:
1. **WeightManager.cpp**: Tracks NON-GEMM weight (bias/norm) cloning with pointer addresses
2. **FusedQKVGEMMStage.cpp**: Logs bias tensor pointers at stage entry

### New Crash Analysis (with enhanced logging)

**Log file**: `/tmp/rccl_debug_bias.log`

**Crash Timeline**:
```
[13:54:36.746] ROCm:0 starts layer0_wo_allreduce, waits for ROCm:1
[13:54:36.746] ROCm:0 arrival #1 of 2, tensor ptr=0x3a1ecfa8
[13:54:36.869] ROCm:1 uploads embedding table (519 MB) workspace=0x7e7d54aae400
[13:54:36.869] ROCm:1 launches embedding kernel (d_output=0x7e7ac6800000)
[13:54:36.876] ROCm:1 ensureOnHost downloads 14680064 bytes (embedding output snapshot)
[13:54:36.876] ROCm:1 marks 'embeddings' as device-dirty
[13:54:36.880] ROCm:1 SECOND ensureOnHost downloads 14680064 bytes
Memory access fault by GPU node-3 on address 0x7e8740ba6000
```

**Key Observation**: 
- The crash happens during **ensureOnHost** (D2H transfer) NOT during FusedQKVGEMMStage execution
- ROCm:1 never reaches its FusedQKVGEMMStage - it crashes during snapshot callback after embedding
- The crash address `0x7e8740ba6000` is different from both embedding workspace addresses

**Hypothesis Shift**: 
The crash is NOT in bias tensor handling (original hypothesis). It's in the snapshot/coherence system:
- The second `ensureOnHost` call appears to be triggered by the snapshot callback
- This callback may be trying to read a tensor that has an invalid device pointer

### Bias Pointer Logging Results

ROCm:0 successfully logged bias pointers:
```
[FusedQKVGEMMStage] BIAS POINTERS: bias_q=0x7e824c000900 bias_k=0x7e824ce52280 bias_v=0x7e824ce968c0 stage_device=ROCm:0
```

ROCm:1 **never reached** its FusedQKVGEMMStage (crash occurred earlier during embedding snapshot).

---

## Best Practice: Capture Test Output to File

**IMPORTANT**: Do NOT re-run the test repeatedly just to change grep patterns. Instead:

```bash
# Step 1: Run test ONCE, capture ALL output to file
timeout 300 ./build_v2_integration/tests/v2/v2_integration_parity_qwen2_localtp_rccl_vs_pytorch \
  --gtest_filter="*PrefillParity*" 2>&1 | tee /tmp/rccl_localtp_test.log

# Step 2: Grep through the file as many times as needed
grep "\[ROCm:0\]" /tmp/rccl_localtp_test.log | head -50
grep "\[ROCm:1\]" /tmp/rccl_localtp_test.log | head -50
grep -E "allocated|ensureOnDevice|ensureOnHost" /tmp/rccl_localtp_test.log
grep "embedding" /tmp/rccl_localtp_test.log
grep -A5 "Memory access fault" /tmp/rccl_localtp_test.log

# Step 3: Only re-run when you need NEW output (e.g., after code changes)
```

This saves significant time since the test takes ~60-90 seconds to reach the crash point.

---

## Hardware Configuration

```
╔══════════════════════════════════════════════════════════════════╗
║        TRUE LOCAL TENSOR PARALLELISM (RCCL) PARITY TEST          ║
╠══════════════════════════════════════════════════════════════════╣
║  Device 0: ROCm:0 (AMD Instinct MI60/MI50, 31 GB, gfx906)        ║
║  Device 1: ROCm:1 (AMD Instinct MI60/MI50, 31 GB, gfx906)        ║
║  Backend: RCCL (GPU-native collectives via Infinity Fabric/PCIe) ║
║  Scope: LOCAL (single process, 2 devices)                        ║
║  Weight Sharding: ENABLED (Megatron-style TP)                    ║
╚══════════════════════════════════════════════════════════════════╝
```

Also has 2x NVIDIA RTX 3090 (CUDA) and 3x AMD MI60/MI50 (ROCm).

---

## Recent Refactoring (Completed)

### Parity Test Method Refactoring
The parity test methods were refactored to be explicit about single-device vs TP mode:

| Old Method | New Method | Purpose |
|------------|------------|---------|
| `runPrefillParity()` | `runSingleDevicePrefillParity()` | Single-device - always calls `setupPipeline()` |
| `runDecodeParity()` | `runSingleDeviceDecodeParity()` | Single-device - always calls `setupPipeline()` |
| `runTPPrefillParity()` | *(unchanged)* | Multi-device TP - requires pre-configured runner |
| *(new)* | `runTPDecodeParity()` | Multi-device TP - requires pre-configured runner |

**Files Modified**:
- `tests/v2/integration/parity/ParityTestBase.h`
- `tests/v2/integration/parity/qwen2/Qwen2ParityTestBase.h`
- `tests/v2/integration/parity/qwen2/Test__Qwen2_LocalTP_RCCL_vs_PyTorch.cpp`
- `tests/v2/integration/parity/qwen2/Test__Qwen2_LocalTP_NCCL_vs_PyTorch.cpp`
- `tests/v2/integration/parity/qwen2/Test__Qwen2_LocalTP_PCIeBAR_vs_PyTorch.cpp`
- `tests/v2/integration/parity/qwen2/Test__Qwen2_LocalTP_RCCL_Allreduce_vs_PyTorch.cpp`
- `tests/v2/integration/parity/README.md`

The test now correctly runs with 2-way TP (both devices loading sharded weights).

---

## Key Files Involved

### Core Execution
- `src/v2/execution/RankOrchestrator.cpp` - Parallel device execution via `std::async`
- `src/v2/execution/DeviceGraphOrchestrator.cpp` - Single-device graph execution
- `src/v2/execution/InferenceRunnerFactory.cpp` - Creates device runners

### GPU Memory/Kernels
- `src/v2/kernels/rocm/ROCmEmbeddingKernelT.cpp` - Embedding table upload (crash happens here)
- `src/v2/tensors/TensorBase.cpp` - Tensor coherence (`ensureOnDevice()`, `mark_device_dirty()`)

### Weight Management
- `src/v2/loaders/WeightManager.cpp` - Per-device weight sharding and caching
- `src/v2/tensors/TensorClasses.h` - GPU tensor allocation

### TP Context
- `src/v2/parallelism/LocalTPContext.cpp` - Local tensor parallelism context
- `src/v2/parallelism/RCCLBackend.cpp` - RCCL collective backend

### Logging
- `src/v2/utils/Logger.h` - Thread-local device prefix logging (`ScopedDeviceLog`)

---

## Logging Improvements Made

### ScopedDeviceLog (Thread-Local Device Prefix)
Added RAII-based device logging so each device's logs are clearly tagged:

```cpp
// In Logger.h
class ScopedDeviceLog {
    ScopedDeviceLog(const DeviceId& device);  // Sets [ROCm:0] prefix
    ~ScopedDeviceLog();                        // Clears prefix
};

// Usage in DeviceGraphOrchestrator.cpp
bool DeviceGraphOrchestrator::forward(...) {
    ScopedDeviceLog device_log(state_.device_id);  // All logs now prefixed
    // ... execution ...
}
```

**Log Output Example**:
```
[13:37:59.473] [INFO ] [ROCm:1] [WeightManager.cpp:2063] Device ROCm:1 (rank 1/2) column-parallel...
[13:37:59.481] [INFO ] [ROCm:0] [WeightManager.cpp:2063] Device ROCm:0 (rank 0/2) column-parallel...
```

---

## Investigation Avenues Explored

### 1. HSA Node Mapping Confusion ✓
- Confirmed HSA node IDs don't match ROCm ordinals
- GPU node-3 = ROCm:1, not ROCm:2

### 2. Memory Address Analysis
- Crash address `0x70fa059a6000` was not found in logged allocations
- May indicate:
  - Memory freed by another device
  - Cross-device pointer access
  - Stale pointer in kernel cache

### 3. KernelFactory Caching (Suspected)
- `KernelFactory` has static global caches keyed by `(tensor_ptr, device, ordinal)`
- If tensor pointers are reused across tests, stale kernels could be retrieved
- **Not yet fully investigated**

### 4. WeightManager Per-Device Caching
- Each device should get its own weight slice
- Logs show correct sharding: `rank 0/2 -> rows [0, 448)`, `rank 1/2 -> rows [448, 896)`
- **Appears correct but worth verifying actual GPU memory addresses**

### 5. Embedding Table Upload
- Both devices upload the FULL 519 MB embedding table (not sharded)
- Crash happens immediately after both uploads complete
- **Possible race condition or pointer confusion**

---

## Suggested Next Steps

### 1. Add GPU Memory Allocation Logging
Add logging to track actual GPU memory addresses for each device:
```cpp
// In ROCmEmbeddingKernelT.cpp after hipMalloc
LOG_DEBUG("[ROCm:" << device_id << "] Allocated embedding workspace at " 
          << std::hex << workspace_ptr << " (" << size_mb << " MB)");
```

### 2. Verify Tensor Device Affinity
Check that tensors created for ROCm:0 don't get passed to ROCm:1's kernels:
```cpp
// Add assertions in kernel entry points
LLAMINAR_ASSERT(tensor->device_id() == expected_device, 
                "Tensor device mismatch: expected " << expected_device 
                << " got " << tensor->device_id());
```

### 3. Check for Cross-Device Pointer Usage
In `RankOrchestrator::forward()`, verify each device's runner only accesses its own data:
- Look for shared state that might have GPU pointers
- Check if `TPSnapshot` or other shared structures contain device pointers

### 4. Investigate KernelFactory Cache
Examine if kernel caching could return a kernel configured for wrong device:
```bash
grep -r "KernelFactory" src/v2/kernels/ --include="*.cpp" | head -20
```

### 5. Run with ASAN (Address Sanitizer)
```bash
cmake -B build_v2_asan -S src/v2 -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-g3 -O0 -fno-omit-frame-pointer -fsanitize=address" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address"
cmake --build build_v2_asan --parallel

ASAN_OPTIONS=halt_on_error=0:detect_leaks=0 \
  timeout 300 ./build_v2_asan/tests/v2/v2_integration_parity_qwen2_localtp_rccl_vs_pytorch \
  --gtest_filter="*PrefillParity*" 2>&1 | tee /tmp/asan_output.log
```

### 6. Run with ROCm Debug Flags
```bash
AMD_LOG_LEVEL=4 HSA_TOOLS_LIB=/opt/rocm/lib/librocm-debug-agent.so \
  timeout 300 ./build_v2_integration/tests/v2/v2_integration_parity_qwen2_localtp_rccl_vs_pytorch \
  --gtest_filter="*PrefillParity*" 2>&1 | tee /tmp/rocm_debug.log
```

---

## Build Commands

```bash
# Integration build (has snapshots + debug symbols)
cmake -B build_v2_integration -S src/v2 -DCMAKE_BUILD_TYPE=Integration \
  -DHAVE_CUDA=ON -DHAVE_ROCM=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build_v2_integration --parallel

# Debug build (for unit tests)
cmake -B build_v2 -S src/v2 -DCMAKE_BUILD_TYPE=Debug \
  -DHAVE_CUDA=ON -DHAVE_ROCM=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build_v2 --parallel
```

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `LLAMINAR_LOG_LEVEL=DEBUG` | Enable debug logging |
| `LLAMINAR_TRACE_TRANSFERS=1` | Trace H2D/D2H memory transfers |
| `LLAMINAR_STAGE_DUMP_ENABLED=1` | Dump stage inputs/outputs |
| `LLAMINAR_VALIDATE_BUFFERS=1` | Validate buffers after stages |
| `AMD_LOG_LEVEL=4` | ROCm verbose logging |

---

## Key Hypothesis

The crash is likely caused by one of:

1. **Cross-device pointer**: ROCm:1 receives a pointer that was allocated on ROCm:0
2. **Stale cached kernel**: KernelFactory returns a kernel configured for wrong device
3. **Race condition**: Both devices try to access shared state concurrently
4. **Embedding table confusion**: Both devices upload to same workspace pointer

The crash happens at the transition from weight loading to actual forward pass execution, specifically after embedding tables are uploaded. Focus investigation on what happens immediately after embedding upload.

---

## Files to Watch

```
src/v2/kernels/rocm/ROCmEmbeddingKernelT.cpp:307  # Crash location
src/v2/execution/RankOrchestrator.cpp      # Parallel execution
src/v2/execution/DeviceGraphOrchestrator.cpp      # Per-device execution
src/v2/loaders/WeightManager.cpp                  # Weight distribution
src/v2/kernels/KernelFactory.h                    # Kernel caching
```
