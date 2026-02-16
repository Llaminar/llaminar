# CUDA Kernel Optimization Handover Document

**Date**: January 23, 2026  
**Branch**: `feature/cuda-kernels`  
**Author**: AI Agent Session  

---

## Executive Summary

This document captures the progress, methodology, and remaining work for CUDA kernel optimization in Llaminar V2. The primary achievement was identifying and fixing a **14.5% "Input Coherence" overhead** caused by CPU-based MPI collectives invalidating GPU tensor data, and wiring up NCCL/RCCL backends to enable GPU-native collective operations.

---

## Problem Identification

### Root Cause Analysis

Profiling revealed a significant performance bottleneck labeled "Input Coherence" consuming ~14.5% of inference time. Investigation traced this to:

1. **AllreduceStage** was using CPU-based MPI (`MPI_Allreduce`)
2. CPU MPI requires calling `mutable_data()` on tensors to access host memory
3. `mutable_data()` calls `invalidateGpuData()` which marks GPU buffers as stale
4. Subsequent GPU stages must re-upload data, causing unnecessary H2D transfers

### Profiling Command Used

```bash
LLAMINAR_EXECUTOR_PROFILING=1 \
LLAMINAR_LOG_LEVEL=INFO \
./build_v2_release/llaminar2 --benchmark -m models/qwen2.5-7b-instruct-q4_0.gguf -n 50
```

### Key Insight

The coherence protocol was working correctly—the problem was that CPU MPI collectives were forcing data through host memory even when all participating ranks had GPU-resident tensors. The solution: use GPU-native collectives (NCCL for CUDA, RCCL for ROCm) that operate directly on device memory.

---

## Solution Implemented

### Architecture Overview

```
InferenceRunnerFactory
    │
    ├── buildLocalClusterInventory()  ← NEW: Detects CUDA/ROCm GPUs
    │
    └── CollectiveContextFactory::createIntraNode(inventory, mpi_ctx)
            │
            └── BackendRouter::createBackend(device_group)
                    │
                    ├── NCCLBackend (all-CUDA groups)
                    └── RCCLBackend (all-ROCm groups)
```

### Files Modified

| File | Changes |
|------|---------|
| `src/v2/collective/BackendRouter.cpp` | Added NCCL/RCCL includes; `createBackend()` now instantiates real backends |
| `src/v2/execution/GraphOrchestrator.h` | Added `setCollectiveContext()`, `collectiveContext()`, `isGpuCollectivesEnabled()` |
| `src/v2/execution/GraphOrchestrator.cpp` | Implemented CollectiveContext setter that wires to internal DeviceGraphExecutor |
| `src/v2/execution/InferenceRunnerFactory.cpp` | Added `buildLocalClusterInventory()` helper; creates and wires CollectiveContext |
| `tests/v2/unit/Test__GraphOrchestrator.cpp` | Added 4 new unit tests for `setCollectiveContext()` |

### Key Code Changes

#### BackendRouter.cpp - Backend Instantiation

```cpp
// Before: returned nullptr for GPU backends
// After: actually creates NCCL/RCCL backends

#ifdef HAVE_NCCL
#include "collective/NCCLBackend.h"
#endif
#ifdef HAVE_ROCM
#include "collective/RCCLBackend.h"
#endif

std::unique_ptr<ICollectiveBackend> BackendRouter::createBackend(
    const DeviceGroup& group,
    std::shared_ptr<MPIContext> mpi_ctx) {
    
    if (group.isAllCuda()) {
#ifdef HAVE_NCCL
        return std::make_unique<NCCLBackend>(mpi_ctx);
#endif
    }
    if (group.isAllRocm()) {
#ifdef HAVE_ROCM  
        return std::make_unique<RCCLBackend>(mpi_ctx);
#endif
    }
    return std::make_unique<MPIBackend>(mpi_ctx);  // Fallback
}
```

#### InferenceRunnerFactory.cpp - Inventory Building

```cpp
static ClusterInventory buildLocalClusterInventory(std::shared_ptr<MPIContext> mpi_ctx) {
    ClusterInventory inventory;
    
#ifdef HAVE_CUDA
    int cuda_count = 0;
    if (cudaGetDeviceCount(&cuda_count) == cudaSuccess) {
        for (int i = 0; i < cuda_count; ++i) {
            inventory.addDevice(DeviceId{DeviceType::CUDA, i}, mpi_ctx->rank());
        }
    }
#endif

#ifdef HAVE_ROCM
    int rocm_count = 0;
    if (hipGetDeviceCount(&rocm_count) == hipSuccess) {
        for (int i = 0; i < rocm_count; ++i) {
            inventory.addDevice(DeviceId{DeviceType::ROCm, i}, mpi_ctx->rank());
        }
    }
#endif
    
    return inventory;
}
```

#### GraphOrchestrator - Context Injection

```cpp
void GraphOrchestrator::setCollectiveContext(std::shared_ptr<ICollectiveContext> ctx) {
    collective_ctx_ = std::move(ctx);
    if (executor_) {
        executor_->setCollectiveContext(collective_ctx_);
    }
}
```

---

## Test Coverage

### Existing Tests (Pre-Wiring)

| Test File | Coverage |
|-----------|----------|
| `Test__NCCLBackend.cpp` | NCCL allreduce, allgather, broadcast operations |
| `Test__RCCLBackend.cpp` | RCCL allreduce, allgather, broadcast operations |
| `Test__BackendRouter.cpp` | Router logic with mock factory |
| `Test__CollectiveContext.cpp` | Unit tests for context creation |
| `Test__CollectiveContext_GPU.cpp` | Integration tests with real GPU collectives |

### New Tests Added

```cpp
// tests/v2/unit/Test__GraphOrchestrator.cpp

TEST(Test__GraphOrchestrator, SetCollectiveContext_NullByDefault)
TEST(Test__GraphOrchestrator, SetCollectiveContext_SetAndGet)
TEST(Test__GraphOrchestrator, SetCollectiveContext_ClearWithNull)
TEST(Test__GraphOrchestrator, SetCollectiveContext_ReplacesExisting)
```

### Running Tests

```bash
# Unit tests (Debug build)
cmake -B build_v2 -S src/v2 -DCMAKE_BUILD_TYPE=Debug -DHAVE_CUDA=ON
cmake --build build_v2 --parallel
ctest --test-dir build_v2 -R "V2_Unit_GraphOrchestrator" --output-on-failure

# Collective-related tests
ctest --test-dir build_v2 -R "Collective|NCCL|RCCL|BackendRouter" --output-on-failure
```

---

## Debugging CUDA Kernel Performance

### 1. Profiling Infrastructure

```bash
# Enable executor-level profiling (per-stage timing)
LLAMINAR_EXECUTOR_PROFILING=1 ./build_v2_release/llaminar2 --benchmark -m model.gguf -n 50

# Enable kernel-level profiling (per-kernel timing)
LLAMINAR_PROFILE_KERNELS=1 ./build_v2_release/llaminar2 --benchmark -m model.gguf -n 50

# Both combined
LLAMINAR_EXECUTOR_PROFILING=1 LLAMINAR_PROFILE_KERNELS=1 \
./build_v2_release/llaminar2 --benchmark -m model.gguf -n 50
```

### 2. Transfer Tracing

To identify unnecessary H2D/D2H transfers:

```bash
# Trace all large D2H transfers (indicates potential coherence bugs)
LLAMINAR_TRACE_TRANSFERS=1 \
LLAMINAR_TRACE_TRANSFERS_ONLY_D2H=1 \
LLAMINAR_TRACE_TRANSFERS_MIN_BYTES=1000000 \
./build_v2_release/llaminar2 -m model.gguf -p "test" -n 10

# Throw exception on first transfer (for stack trace)
LLAMINAR_TRACE_TRANSFERS=1 \
LLAMINAR_TRACE_TRANSFERS_THROW=1 \
./build_v2_release/llaminar2 -m model.gguf -p "test"
```

### 3. Stage Dump Framework

For debugging specific stage inputs/outputs:

```bash
# Dump attention stages for layer 0
LLAMINAR_STAGE_DUMP_ENABLED=1 \
LLAMINAR_STAGE_DUMP_NAMES=fused_attn_wo \
LLAMINAR_STAGE_DUMP_LAYERS=0 \
./build_v2_release/llaminar2 -m model.gguf -p "test" -n 5

# Dumps written to /tmp/llaminar_stage_dumps/
```

### 4. NVIDIA Nsight Systems

```bash
# Full system trace
nsys profile --trace=cuda,nvtx,osrt \
  ./build_v2_release/llaminar2 -m model.gguf -p "test" -n 20

# Analyze in Nsight Systems GUI
nsys-ui report.nsys-rep
```

### 5. NVIDIA Nsight Compute (Kernel Analysis)

```bash
# Profile specific kernel
ncu --target-processes all \
    --set full \
    --kernel-name "gemm_kernel" \
    ./build_v2_release/llaminar2 -m model.gguf -p "test" -n 5
```

---

## Common Issues & Solutions

### Issue 1: Unexpected D2H Transfers

**Symptom**: Profiling shows high "Input Coherence" time  
**Cause**: Code calling `tensor->data()` or `mutable_data()` on GPU-resident tensors  
**Solution**: Use transfer tracing to find call sites, ensure GPU kernels use device pointers directly

### Issue 2: MPI Collectives Invalidating GPU Data

**Symptom**: AllreduceStage triggers GPU data invalidation  
**Cause**: CPU MPI requires host memory access  
**Solution**: Wire up NCCL/RCCL backends (implemented in this session)

### Issue 3: LOG_DEBUG/TRACE Causing Transfers

**Symptom**: Debug builds much slower than release  
**Cause**: Logging statements calling `tensor->data()` unconditionally  
**Solution**: Guard data access with log level checks:
```cpp
if (LOG_LEVEL_ENABLED(DEBUG)) {
    LOG_DEBUG("value: " << tensor->data()[0]);
}
```

---

## Remaining Work

### High Priority

1. **Verify NCCL/RCCL Performance Improvement**
   - Run benchmark before/after on multi-GPU setup
   - Confirm "Input Coherence" overhead is eliminated
   - Command: `LLAMINAR_EXECUTOR_PROFILING=1 ./build_v2_release/llaminar2 --mpi-procs 2 --benchmark -m model.gguf -n 50`

2. **End-to-End MPI+NCCL Test**
   - Add integration test that exercises full path with 2+ MPI ranks
   - Verify allreduce results match CPU MPI baseline

### Medium Priority

3. **Kernel Fusion Opportunities**
   - Profile decode phase to identify kernel launch overhead
   - Consider fusing RMSNorm + GEMM or attention + projection

4. **Memory Pool Optimization**
   - Analyze GPU memory fragmentation
   - Consider cudaMallocAsync for better memory reuse

### Low Priority

5. **Multi-Node Support**
   - Current implementation is intra-node only
   - Need `CollectiveContextFactory::createInterNode()` for multi-machine

---

## Build Commands Reference

```bash
# Debug build (for development)
cmake -B build_v2 -S src/v2 -DCMAKE_BUILD_TYPE=Debug -DHAVE_CUDA=ON -DHAVE_ROCM=ON
cmake --build build_v2 --parallel

# Release build (for benchmarks)
cmake -B build_v2_release -S src/v2 -DCMAKE_BUILD_TYPE=Release -DHAVE_CUDA=ON -DHAVE_ROCM=ON
cmake --build build_v2_release --parallel

# Integration build (for parity tests)
cmake -B build_v2_integration -S src/v2 -DCMAKE_BUILD_TYPE=Integration -DHAVE_CUDA=ON -DHAVE_ROCM=ON
cmake --build build_v2_integration --parallel
```

---

## Key Files for Reference

| Purpose | File Path |
|---------|-----------|
| GPU Coherence Protocol | `src/v2/tensors/TensorClasses.h` (`ensureOnDevice`, `mark_device_dirty`) |
| Stage Coherence Helper | `src/v2/execution/StageCoherence.h` |
| Collective Backends | `src/v2/collective/NCCLBackend.cpp`, `RCCLBackend.cpp` |
| Backend Routing | `src/v2/collective/BackendRouter.cpp` |
| Context Factory | `src/v2/collective/CollectiveContextFactory.cpp` |
| DeviceGraphExecutor Intercept | `src/v2/execution/DeviceDeviceGraphExecutor.cpp` (lines 719-731) |
| Debug Environment | `src/v2/utils/DebugEnv.h` |

---

## Contact Points

- **Architecture docs**: `.github/instructions/llaminar-architecture-v2.instructions.md`
- **Dev guidelines**: `.github/copilot-instructions.md`
- **Parity testing**: `tests/v2/integration/parity/README.md`

---

*End of Handover Document*
