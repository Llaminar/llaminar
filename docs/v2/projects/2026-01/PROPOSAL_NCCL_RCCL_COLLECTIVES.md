# Project Proposal: NCCL/RCCL GPU Collective Backends for Llaminar

**Author:** David Sanftenberg  
**Date:** January 22, 2026  
**Status:** Proposal  

## Executive Summary

This proposal outlines the implementation of fully-functional NCCL (NVIDIA) and RCCL (AMD) backends for GPU-based collective operations in Llaminar. The current architecture already has NCCL/RCCL backend stubs, but collectives currently fall back to CPU-based MPI AllReduce, causing significant performance overhead from GPU→CPU→GPU memory transfers.

**Expected Performance Gains:**
- Eliminate 2x PCIe transfers per AllReduce (currently: GPU→Host→MPI→Host→GPU)
- NVLink: 600+ GB/s (vs ~32 GB/s PCIe)
- Latency: ~2-5μs intra-node (vs ~50-100μs CPU-mediated)
- 10-30% end-to-end throughput improvement for multi-GPU inference

---

## Current State Analysis

### What Already Exists ✅

The collective infrastructure is well-designed and modular:

1. **ICollectiveBackend Interface** ([ICollectiveBackend.h](../../../../src/v2/collective/ICollectiveBackend.h))
   - Abstract interface with `allreduce()`, `allgather()`, `allgatherv()`, `broadcast()`, `reduceScatter()`
   - `CollectiveBackendType` enum includes `NCCL` and `RCCL`
   - `CollectiveDataType` supports FP32, FP16, BF16, INT32, INT8

2. **NCCLBackend Stub** ([NCCLBackend.cpp](../../../../src/v2/collective/backends/NCCLBackend.cpp))
   - Compiles with `#ifdef HAVE_NCCL`
   - Has all collective operations implemented (allreduce, allgather, etc.)
   - Has type conversion helpers (`toNcclDataType`, `toNcclRedOp`)
   - **Missing:** Multi-process communicator initialization via MPI unique ID exchange

3. **RCCLBackend Stub** ([RCCLBackend.cpp](../../../../src/v2/collective/backends/RCCLBackend.cpp))
   - Nearly identical structure to NCCLBackend (RCCL API mirrors NCCL)
   - Same completeness/gaps as NCCLBackend

4. **BackendRouter** ([BackendRouter.cpp](../../../../src/v2/collective/BackendRouter.cpp))
   - `selectBackendForDomain()` already has logic to route:
     - All-CUDA groups → NCCL
     - All-ROCm groups → RCCL
     - Heterogeneous → PCIe BAR or Host fallback
   - `DefaultBackendFactory::isAvailable()` checks `HAVE_NCCL`/`HAVE_RCCL`

5. **CollectiveContext** ([CollectiveContext.cpp](../../src/v2/execution/CollectiveContext.cpp))
   - High-level API: `executeAllreduce()`, `executeAllgather()`, etc.
   - Automatically delegates to router to select backend
   - Domain-aware variants for TP domains

6. **CMake Infrastructure** ([CMakeLists.txt](../../../../src/v2/CMakeLists.txt))
   - `HAVE_CUDA` and `HAVE_ROCM` options exist
   - Backend source files already listed (NCCLBackend.cpp, RCCLBackend.cpp)
   - **Missing:** `find_package(NCCL)` / RCCL detection and linkage

### What's Missing ❌

| Component | Gap | Effort |
|-----------|-----|--------|
| **CMake** | No NCCL/RCCL library detection or linking | Small |
| **NCCL Init** | Single-process init works; multi-process needs MPI unique ID broadcast | Medium |
| **RCCL Init** | Same as NCCL | Medium |
| **Stream Integration** | Backends create streams but don't integrate with CUDA/ROCm device context | Medium |
| **Tensor Device Query** | `executeAllreduce()` assumes CPU; needs to query tensor's actual device | Small |
| **GPU Pointer Access** | Need to use tensor's GPU buffer, not `mutable_data()` which syncs to CPU | Medium |
| **Testing** | No integration tests for NCCL/RCCL backends | Medium |

---

## Architecture

### Data Flow (Current vs Proposed)

**Current (CPU-Mediated MPI):**
```
GPU0 Buffer → cudaMemcpy D→H → Host Buffer → MPI_Allreduce → Host Buffer → cudaMemcpy H→D → GPU0 Buffer
                  ~6μs             ~50μs           ~50μs              ~6μs
Total: ~112μs + PCIe bandwidth bottleneck
```

**Proposed (NCCL/RCCL):**
```
GPU0 Buffer ────────────── ncclAllReduce ──────────────> GPU0 Buffer
                              ~3-5μs (NVLink)
                             ~10-20μs (PCIe P2P)
```

### Backend Selection Logic

```
                    ┌─────────────────────────┐
                    │   CollectiveContext     │
                    │  executeAllreduce()     │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │    BackendRouter        │
                    │  selectBackendForDomain │
                    └───────────┬─────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│  All CUDA?    │       │  All ROCm?    │       │ Cross-rank?   │
│  Use NCCL     │       │  Use RCCL     │       │ Use MPI       │
└───────────────┘       └───────────────┘       └───────────────┘
```

---

## Implementation Plan

### Phase 1: CMake and Library Detection (1-2 days)

**Files to modify:**
- `src/v2/CMakeLists.txt`

**Tasks:**
1. Add NCCL library detection:
   ```cmake
   if(HAVE_CUDA)
       # NCCL is typically installed with CUDA Toolkit or separately
       find_path(NCCL_INCLUDE_DIR nccl.h
           HINTS ${CUDAToolkit_INCLUDE_DIRS}
           PATHS /usr/include /usr/local/include /opt/nccl/include
       )
       find_library(NCCL_LIBRARY nccl
           HINTS ${CUDAToolkit_LIBRARY_DIR}
           PATHS /usr/lib/x86_64-linux-gnu /usr/local/lib /opt/nccl/lib
       )
       if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
           set(HAVE_NCCL ON)
           add_compile_definitions(HAVE_NCCL)
           message(STATUS "V2: NCCL found: ${NCCL_LIBRARY}")
       else()
           message(STATUS "V2: NCCL not found - GPU collectives will use MPI fallback")
       endif()
   endif()
   ```

2. Add RCCL library detection:
   ```cmake
   if(HAVE_ROCM)
       # RCCL is part of ROCm installation
       find_path(RCCL_INCLUDE_DIR rccl/rccl.h
           HINTS ${ROCM_PATH}/include
           PATHS /opt/rocm/include
       )
       find_library(RCCL_LIBRARY rccl
           HINTS ${ROCM_PATH}/lib
           PATHS /opt/rocm/lib
       )
       if(RCCL_INCLUDE_DIR AND RCCL_LIBRARY)
           set(HAVE_RCCL ON)
           add_compile_definitions(HAVE_RCCL)
           message(STATUS "V2: RCCL found: ${RCCL_LIBRARY}")
       endif()
   endif()
   ```

3. Link libraries to `llaminar2_core`:
   ```cmake
   if(HAVE_NCCL)
       target_include_directories(llaminar2_core PRIVATE ${NCCL_INCLUDE_DIR})
       target_link_libraries(llaminar2_core PRIVATE ${NCCL_LIBRARY})
   endif()
   if(HAVE_RCCL)
       target_include_directories(llaminar2_core PRIVATE ${RCCL_INCLUDE_DIR})
       target_link_libraries(llaminar2_core PRIVATE ${RCCL_LIBRARY})
   endif()
   ```

### Phase 2: Multi-Process Communicator Initialization (2-3 days)

**Files to modify:**
- `src/v2/collective/backends/NCCLBackend.cpp`
- `src/v2/collective/backends/NCCLBackend.h`
- `src/v2/collective/backends/RCCLBackend.cpp`
- `src/v2/collective/backends/RCCLBackend.h`

**Problem:**
The current `NCCLBackend::initialize()` works for single-process multi-GPU but not for multi-process (MPI) scenarios. NCCL requires a unique ID to be generated on rank 0 and broadcast to all ranks.

**Solution:**
Add MPIContext to backend constructors and use MPI to broadcast the unique ID:

```cpp
bool NCCLBackend::initialize(const DeviceGroup &group, std::shared_ptr<MPIContext> mpi_ctx)
{
    // ... existing validation ...
    
    ncclUniqueId id;
    
    if (mpi_ctx && mpi_ctx->world_size() > 1) {
        // Multi-process: rank 0 generates ID, broadcasts to all
        if (mpi_ctx->rank() == 0) {
            NCCL_CHECK(ncclGetUniqueId(&id));
        }
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_ctx->comm());
        
        // Each rank initializes with its local GPU
        NCCL_CHECK(ncclCommInitRank(&comm_, mpi_ctx->world_size(), id, mpi_ctx->rank()));
    } else {
        // Single process: use ncclCommInitAll for all local GPUs
        std::vector<int> dev_ids;
        for (const auto& dev : group.devices) {
            if (dev.type == DeviceType::CUDA) {
                dev_ids.push_back(dev.ordinal);
            }
        }
        NCCL_CHECK(ncclCommInitAll(&comms_, dev_ids.size(), dev_ids.data()));
    }
    
    // Create stream on the local device
    cudaSetDevice(group.localDevice().ordinal);
    cudaStreamCreate(&stream_);
    
    initialized_ = true;
    return true;
}
```

**Key Changes:**
1. Accept `MPIContext` in constructor (already passed by factory)
2. Use `ncclGetUniqueId()` on rank 0, `MPI_Bcast()` to all ranks
3. Call `ncclCommInitRank()` instead of `ncclCommInitAll()` for multi-process
4. Store `mpi_ctx_` for later use in collective operations

### Phase 3: Tensor Device-Aware Pointer Access (1-2 days)

**Files to modify:**
- `src/v2/execution/CollectiveContext.cpp`
- `src/v2/execution/compute_stages/stages/AllreduceStage.cpp`

**Problem:**
Current code calls `buffer->mutable_data()` which triggers GPU→Host sync. We need to use the GPU buffer directly.

**Solution:**
Use `active_mutable_data_ptr()` which returns the GPU pointer if the tensor is device-resident:

```cpp
bool CollectiveContext::executeAllreduce(
    ITensor *buffer,
    size_t count,
    DeviceId tensor_device,
    CollectiveOp op)
{
    // Get backend for this device group
    ICollectiveBackend *backend = router_->getBackend(group);
    
    // CRITICAL: Use active_mutable_data_ptr() to get GPU pointer
    // without invalidating device state
    void *data_ptr = buffer->active_mutable_data_ptr();
    
    // For GPU backends, ensure tensor is on device
    if (backend->type() == CollectiveBackendType::NCCL ||
        backend->type() == CollectiveBackendType::RCCL) {
        // Tensor must be on GPU; coherence should be handled by caller
        // via StageCoherence::ensureInputsOnDevice()
    }
    
    return backend->allreduce(data_ptr, actual_count, dtype, op);
}
```

**Note:** The `active_mutable_data_ptr()` method already exists and is used elsewhere in the codebase for this purpose.

### Phase 4: Stream and Synchronization Integration (1-2 days)

**Files to modify:**
- `src/v2/collective/backends/NCCLBackend.cpp`
- `src/v2/collective/backends/RCCLBackend.cpp`

**Problem:**
NCCL/RCCL operations are asynchronous and enqueued on a stream. We need proper synchronization.

**Options:**
1. **Synchronous mode** (simpler): Call `cudaStreamSynchronize()` after each collective
2. **Async mode** (better): Use CUDA events for dependency tracking

**Phase 4a: Synchronous Mode (initial implementation)**
```cpp
bool NCCLBackend::allreduce(void *buffer, size_t count, ...) {
    ncclAllReduce(buffer, buffer, count, dtype, op, comm_, stream_);
    
    // Synchronize to ensure completion before returning
    cudaStreamSynchronize(stream_);
    return true;
}
```

**Phase 4b: Async Mode (future optimization)**
```cpp
// Return event for dependency tracking
cudaEvent_t NCCLBackend::allreduceAsync(void *buffer, size_t count, ...) {
    ncclAllReduce(buffer, buffer, count, dtype, op, comm_, stream_);
    
    cudaEvent_t event;
    cudaEventCreate(&event);
    cudaEventRecord(event, stream_);
    return event;
}
```

### Phase 5: Testing (2-3 days)

**Files to create:**
- `tests/v2/integration/Test__NCCLBackend.cpp`
- `tests/v2/integration/Test__RCCLBackend.cpp`
- `tests/v2/integration/Test__CollectiveContext_GPU.cpp`

**Test Categories:**

1. **Unit Tests** (single-process, single-GPU)
   - Backend initialization/shutdown
   - AllReduce correctness (sum, max, min)
   - AllGather correctness
   - Data type handling (FP32, FP16, BF16)

2. **Multi-GPU Tests** (single-process, multi-GPU)
   - Requires 2+ GPUs of same type
   - Verify data consistency across GPUs
   - Performance comparison vs MPI

3. **Multi-Process Tests** (multi-process via MPI)
   - Requires mpirun with 2+ processes
   - Test unique ID broadcast
   - Verify AllReduce across processes

**Example Test:**
```cpp
TEST(Test__NCCLBackend, AllReduce_Sum_MultiGPU)
{
    // Skip if less than 2 CUDA GPUs
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count < 2) GTEST_SKIP();
    
    NCCLBackend backend;
    DeviceGroup group = DeviceGroupBuilder()
        .setName("test_group")
        .addDevice(DeviceId::cuda(0))
        .addDevice(DeviceId::cuda(1))
        .setLocalRank(0)
        .build();
    
    ASSERT_TRUE(backend.initialize(group));
    
    // Allocate GPU buffers with different values
    float *d_buf0, *d_buf1;
    cudaSetDevice(0);
    cudaMalloc(&d_buf0, 100 * sizeof(float));
    // ... fill with 1.0f
    
    cudaSetDevice(1);
    cudaMalloc(&d_buf1, 100 * sizeof(float));
    // ... fill with 2.0f
    
    // AllReduce should produce 3.0f on all GPUs
    backend.allreduce(d_buf0, 100, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM);
    
    // Verify results
    // ...
}
```

### Phase 6: Integration and Validation (1-2 days)

**Files to modify:**
- `src/v2/execution/DeviceGraphExecutor.cpp` (if needed)
- `src/v2/pipelines/qwen/Qwen2Graph.cpp` (if needed)

**Tasks:**
1. Run full inference pipeline with NCCL/RCCL enabled
2. Verify numerical parity with MPI backend
3. Benchmark performance improvement
4. Test with real models (Qwen2.5-7B multi-GPU)

**Validation Checklist:**
- [ ] `LLAMINAR_LOG_LEVEL=DEBUG` shows "Using NCCL backend" for all-CUDA groups
- [ ] AllReduce results match MPI reference
- [ ] No GPU→CPU transfers in collective operations
- [ ] Performance improvement measured

---

## Timeline Summary

| Phase | Description | Estimated Effort |
|-------|-------------|------------------|
| 1 | CMake and library detection | 1-2 days |
| 2 | Multi-process communicator init | 2-3 days |
| 3 | Tensor device-aware pointers | 1-2 days |
| 4 | Stream and synchronization | 1-2 days |
| 5 | Testing | 2-3 days |
| 6 | Integration and validation | 1-2 days |
| **Total** | | **8-14 days** |

---

## Dependencies

### External Libraries

| Library | Version | Required For | Installation |
|---------|---------|--------------|--------------|
| NCCL | 2.18+ | NVIDIA GPU collectives | `apt install libnccl-dev` or CUDA Toolkit |
| RCCL | 2.x | AMD GPU collectives | Part of ROCm installation |

### Internal Dependencies

- `MPIContext` (already exists)
- `DeviceGroup` and `DeviceGroupBuilder` (already exists)
- `ICollectiveBackend` interface (already exists)
- GPU tensor coherence system (already exists)

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| NCCL not available on dev machines | Medium | Low | Graceful fallback to MPI; skip tests |
| Multi-process init failures | Medium | Medium | Extensive error handling; timeout |
| Performance regression | Low | High | Benchmark before/after; easy revert |
| Numerical precision differences | Low | Medium | Tolerance-based comparison in tests |

---

## Success Criteria

1. **Functional:**
   - NCCL backend works for single-node multi-GPU CUDA
   - RCCL backend works for single-node multi-GPU ROCm
   - Both work with multi-process MPI
   - Automatic fallback to MPI when NCCL/RCCL unavailable

2. **Performance:**
   - >50% reduction in collective operation latency
   - No GPU→CPU→GPU transfers for intra-node collectives
   - Measurable end-to-end throughput improvement

3. **Quality:**
   - All existing tests pass
   - New tests for NCCL/RCCL backends
   - CI/CD integration (conditional on GPU availability)

---

## Future Enhancements (Out of Scope)

1. **Async collective operations** with CUDA events
2. **NCCL Graph capture** for reduced CPU overhead
3. **Multi-node NCCL** with RDMA/InfiniBand
4. **SHARP** integration for collective offload (Mellanox)
5. **AWS EFA / Azure InfiniBand** optimization

---

## Appendix: File Inventory

### Files to Modify

| File | Changes |
|------|---------|
| `src/v2/CMakeLists.txt` | Add NCCL/RCCL detection and linking |
| `src/v2/collective/backends/NCCLBackend.cpp` | Multi-process init, MPI unique ID broadcast |
| `src/v2/collective/backends/NCCLBackend.h` | Add MPIContext member |
| `src/v2/collective/backends/RCCLBackend.cpp` | Same as NCCLBackend |
| `src/v2/collective/backends/RCCLBackend.h` | Same as NCCLBackend |
| `src/v2/execution/CollectiveContext.cpp` | Use `active_mutable_data_ptr()` |

### Files to Create

| File | Purpose |
|------|---------|
| `tests/v2/integration/Test__NCCLBackend.cpp` | NCCL backend tests |
| `tests/v2/integration/Test__RCCLBackend.cpp` | RCCL backend tests |
| `tests/v2/integration/Test__CollectiveContext_GPU.cpp` | GPU collective context tests |
| `cmake/FindNCCL.cmake` | (Optional) CMake find module |
| `cmake/FindRCCL.cmake` | (Optional) CMake find module |

---

## Approval

- [ ] Architecture review
- [ ] Performance team review
- [ ] Implementation approved

**Approved by:** _________________  
**Date:** _________________
