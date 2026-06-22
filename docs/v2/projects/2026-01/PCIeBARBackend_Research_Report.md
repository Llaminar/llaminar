# PCIeBAR Backend Implementation Research Report

**Date**: January 20, 2026  
**Scope**: Analysis of `PCIeBARBackend` and `DirectP2PEngine` for CUDA↔ROCm direct P2P transfers

---

## Executive Summary

The PCIeBAR Backend is a **substantially implemented** but **not production-ready** collective backend for direct CUDA↔ROCm GPU-to-GPU communication via PCIe BAR mapping. It bypasses host memory staging by mapping AMD GPU's BAR (Base Address Register) into CUDA's address space.

| Aspect | Status | Notes |
|--------|--------|-------|
| Core Implementation | ✅ Complete | AllReduce, AllGather, Broadcast working |
| BackendRouter Integration | ✅ Complete | Auto-selected for CUDA+ROCm device groups |
| Unit Tests | ✅ Comprehensive | 684 lines in unit tests |
| Integration Tests | ✅ Comprehensive | 960 lines in integration tests |
| Missing Operations | ⚠️ Partial | ReduceScatter not implemented |
| Error Handling | ⚠️ Basic | Needs retry logic, timeout handling |
| Small Transfer Latency | ⚠️ Unknown | No benchmarks for 14KB tensors |
| Production Hardening | ❌ Needed | See recommendations below |

---

## 1. Current Implementation State

### 1.1 File Locations

| File | Lines | Purpose |
|------|-------|---------|
| [PCIeBARBackend.h](../../../src/v2/collective/backends/PCIeBARBackend.h) | 364 | Backend interface + BAR allocator |
| [PCIeBARBackend.cpp](../../../src/v2/collective/backends/PCIeBARBackend.cpp) | 745 | AllReduce, AllGather, Broadcast impl |
| [DirectP2P.h](../../../src/v2/backends/p2p/DirectP2P.h) | 427 | P2P engine interface |
| [DirectP2P.cpp](../../../src/v2/backends/p2p/DirectP2P.cpp) | 1253 | PCIe BAR mapping, transfers, benchmarks |
| [DirectP2P_ROCm.h](../../../src/v2/backends/p2p/DirectP2P_ROCm.h) | ~80 | DMA-BUF export/import (alternative path) |

### 1.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          PCIeBARBackend                                  │
│  (Implements ICollectiveBackend + IBufferRegistration)                   │
├─────────────────────────────────────────────────────────────────────────┤
│  • allreduce()       - SUM only, FP32                                   │
│  • allgather()       - Full buffer gather                               │
│  • broadcast()       - Root-based broadcast                             │
│  • reduceScatter()   - ❌ NOT IMPLEMENTED                               │
│  • BAR Allocator     - Bump allocator for ROCm buffer placement         │
│  • Buffer Registry   - Tracks CUDA/ROCm buffer pairs by collective_id   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    DirectP2PEngine      │
                    ├─────────────────────────┤
                    │  • initializePCIeBar()  │
                    │  • transferViaPCIeBar() │
                    │  • benchmarkPCIeBar()   │
                    │  • Multi-GPU support    │
                    │  • Overlapped transfers │
                    └────────────────────────┘
```

### 1.3 Implemented Collective Operations

| Operation | Implementation | Notes |
|-----------|---------------|-------|
| `allreduce()` | ✅ Implemented | SUM only, 3-step: Read→Reduce→Write |
| `allreduceRegistered()` | ✅ Implemented | Uses pre-registered buffer offsets |
| `allgather()` | ✅ Implemented | CUDA-centric, 2-device only |
| `broadcast()` | ✅ Implemented | Bidirectional root support |
| `reduceScatter()` | ❌ **NOT IMPLEMENTED** | Returns `false` |
| `synchronize()` | ✅ Implemented | CUDA device sync |

### 1.4 Initialization Flow

```cpp
// 1. BackendRouter creates PCIeBARBackend
auto backend = std::make_unique<PCIeBARBackend>();

// 2. Initialize with device group (internally calls DirectP2PEngine)
backend->initialize(group);  // group must have CUDA + ROCm

// Internally:
//   - Creates DirectP2PEngine
//   - Calls p2p_engine_->initializePCIeBar(cuda_device, rocm_device, 0, 1GB)
//   - Maps AMD GPU BAR via mmap() on /sys/bus/pci/devices/.../resource0
//   - Registers with CUDA using CU_MEMHOSTREGISTER_IOMEMORY flag
//   - Gets CUDA device pointer via cuMemHostGetDevicePointer()
//   - Runs benchmark to measure bandwidth

// 3. BAR Allocator initialized
bar_host_ptr_ = p2p_engine_->getBarHostPtr();  // Host-accessible AMD memory
bar_total_mapped_size_ = 1GB;
bar_alloc_offset_ = 0;  // Bump allocator start
```

### 1.5 Transfer Algorithm (AllReduce)

```cpp
// PCIeBARBackend::allreduce() for 2 devices (CUDA + ROCm)
// Buffer is on CUDA device

// Step 1: Read ROCm's partial result via BAR
transferROCmtoCUDA(0, cuda_temp_buffer_, bytes);
// → cuMemcpyDtoD(cuda_temp, bar_ptr, bytes)

// Step 2: Reduce on CUDA (SIMD vectorized kernel)
reduceOnCUDA(buffer, buffer, cuda_temp_buffer_, count, dtype, SUM);
// → cuda::launchVectorAddInplace_f32()

// Step 3: Write result back to ROCm via BAR
transferCUDAtoROCm(buffer, 0, bytes);
// → cuMemcpyDtoD(bar_ptr, buffer, bytes)
```

---

## 2. Integration Status

### 2.1 BackendRouter Integration

**Fully integrated** in [BackendRouter.cpp](../../../../src/v2/collective/BackendRouter.cpp):

```cpp
CollectiveBackendType BackendRouter::selectBackendType(const DeviceGroup &group) const
{
    // ...
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
    // CUDA + ROCm mix: prefer PCIe BAR if available (direct P2P)
    if (group.isHeterogeneous() && group.cuda_count > 0 && group.rocm_count > 0)
    {
        if (factory_->isAvailable(CollectiveBackendType::PCIE_BAR))
        {
            return CollectiveBackendType::PCIE_BAR;
        }
    }
#endif
    return CollectiveBackendType::HOST;  // Fallback
}
```

### 2.2 Availability Check

Runtime capability probe in `DefaultBackendFactory::isAvailable()`:

```cpp
case CollectiveBackendType::PCIE_BAR:
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
{
    auto caps = DirectP2PEngine::probeCapabilities();
    return caps.canDoPCIeBarP2P();
    // Checks:
    //   - pcie_bar_accessible (can open /sys/bus/pci/.../resource0)
    //   - pcie_bar_iomemory_supported (CUDA driver ≥11.0)
    //   - discovered_bars.size() > 0
}
#endif
```

### 2.3 Selection Logic Summary

| Device Group | Selected Backend | Reason |
|--------------|------------------|--------|
| Global scope | MPI | Inter-node communication |
| All CUDA | NCCL | Native NVIDIA P2P |
| All ROCm | RCCL | Native AMD P2P |
| **CUDA + ROCm** | **PCIE_BAR** | Direct cross-vendor P2P |
| Heterogeneous (no BAR) | HOST | CPU staging fallback |

---

## 3. Latency Characteristics

### 3.1 Measured Bandwidth (from benchmark code)

**PCIe 3.0 x16 (RTX 3090 ↔ MI50):**

| Direction | With rBAR | Without rBAR | Notes |
|-----------|-----------|--------------|-------|
| NVIDIA → AMD (write) | ~2.65 GB/s | ~2.6 GB/s | PCIe posted writes |
| AMD → NVIDIA (read) | ~2.67 GB/s | ~0.8 GB/s | **rBAR critical!** |
| Concurrent R+W | ~5.3 GB/s | N/A | Bidirectional overlap |

**Key insight**: Resizable BAR eliminates read/write asymmetry. Without rBAR, reads are 3x slower.

### 3.2 Synchronization Points

```
┌─────────────────────────────────────────────────────────────────┐
│                    AllReduce Latency Breakdown                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. transferROCmtoCUDA()                                        │
│     └── cuMemcpyDtoD()  ────────────────────────┬              │
│                                                   │              │
│  2. cuCtxSynchronize() ◄──────────────────────────┘  ← SYNC 1  │
│                                                                 │
│  3. reduceOnCUDA()                                              │
│     └── launchVectorAddInplace_f32()  ─────────┬               │
│                                                  │               │
│  4. (implicit kernel completion) ◄───────────────┘  ← SYNC 2   │
│                                                                 │
│  5. transferCUDAtoROCm()                                        │
│     └── cuMemcpyDtoD()  ────────────────────────┬              │
│                                                   │              │
│  6. synchronize() → cudaDeviceSynchronize() ◄─────┘  ← SYNC 3  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**3 synchronization points** per AllReduce:
1. After PCIe BAR read (wait for data arrival)
2. After reduction kernel (implicit)
3. After PCIe BAR write (explicit `synchronize()`)

### 3.3 Small Transfer Latency (14KB)

**⚠️ No existing benchmarks for small tensors.**

Estimated overhead components:
- CUDA kernel launch: ~5-10μs
- cuMemcpyDtoD call overhead: ~2-5μs per call
- cuCtxSynchronize: ~10-50μs
- PCIe transaction setup: ~1-5μs

**For 14KB transfer:**
```
Bandwidth-limited time = 14KB / 2.65GB/s ≈ 5μs
Overhead = 3 × (kernel + sync) ≈ 45-150μs
Total estimated latency ≈ 50-155μs
```

**This is significant** - for small tensors, fixed overhead dominates.

### 3.4 Existing Benchmark Code

Located in `DirectP2P.cpp`:

| Method | Purpose | Transfer Size |
|--------|---------|---------------|
| `benchmarkPCIeBar()` | Single-direction R/W | 64MB default |
| `benchmarkAllModes()` | All modes comprehensive | 32MB |
| No small-transfer benchmark | **MISSING** | **Need 1KB-64KB** |

---

## 4. Missing Functionality

### 4.1 Unimplemented Operations

| Operation | Status | Priority | Effort |
|-----------|--------|----------|--------|
| `reduceScatter()` | ❌ Not implemented | Medium | 2-3 days |
| `allreduceMax()` | ❌ Only SUM | Low | 1 day |
| `allreduceMin()` | ❌ Only SUM | Low | 1 day |
| FP16/BF16 reduction | ❌ FP32 only | High | 2-3 days |

### 4.2 Missing Error Handling

```cpp
// Current: No retry on transient failures
auto result = p2p_engine_->transferViaPCIeBar(...);
return result.success;  // Single attempt

// Needed:
for (int retry = 0; retry < MAX_RETRIES; ++retry) {
    auto result = p2p_engine_->transferViaPCIeBar(...);
    if (result.success) return true;
    if (!is_transient_error(result.error_message)) break;
    usleep(BACKOFF_US * (1 << retry));
}
```

Missing error handling:
- **No retry logic** for transient PCIe errors
- **No timeout handling** - transfers can hang indefinitely
- **No memory validation** before transfers
- **No BAR region exhaustion handling** (bump allocator never compacts)

### 4.3 Missing Configuration Options

| Option | Current | Needed |
|--------|---------|--------|
| `LLAMINAR_PCIE_BAR_SIZE` | Hardcoded 1GB | Environment variable |
| `LLAMINAR_PCIE_BAR_TIMEOUT_MS` | None | Configurable timeout |
| `LLAMINAR_PCIE_BAR_RETRY_COUNT` | None | Retry policy |
| `LLAMINAR_PCIE_BAR_FORCE_SYNC` | Always sync | Optional async |

### 4.4 Multi-Device Limitations

Current limitations:
- **Only 2-device AllReduce** - No tree reduction for >2 devices
- **CUDA must be ordinal 0** in device group
- **Single ROCm BAR** - No multi-AMD-GPU aggregation

---

## 5. Testing Infrastructure

### 5.1 Existing Tests

| File | Lines | Coverage |
|------|-------|----------|
| [Test__PCIeBARBackend.cpp](../../../tests/v2/unit/Test__PCIeBARBackend.cpp) | 684 | Unit tests (mock P2P) |
| [Test__PCIeBARBackendIntegration.cpp](../../../tests/v2/integration/Test__PCIeBARBackendIntegration.cpp) | 960 | Real hardware tests |
| [Test__DirectP2P.cpp](../../../tests/v2/integration/Test__DirectP2P.cpp) | 429 | P2P engine tests |
| [Test__BackendRouter.cpp](../../../tests/v2/unit/Test__BackendRouter.cpp) | ~200 | Selection logic |

### 5.2 Test Categories

**Unit Tests (Test__PCIeBARBackend.cpp):**
- Identity tests (type, name)
- Capability tests (device support, direct transfer)
- Lifecycle tests (initialize, shutdown)
- BAR allocator tests
- Buffer registration tests

**Integration Tests (Test__PCIeBARBackendIntegration.cpp):**
- Real CUDA↔ROCm allreduce
- GEMM-like tensor operations
- Performance benchmarks
- Error condition handling

**P2P Engine Tests (Test__DirectP2P.cpp):**
- Capability probing
- BAR initialization
- Single/multi-direction transfers
- 60-second stress test
- Multi-GPU support

### 5.3 Test Gaps

| Test Type | Status | Description |
|-----------|--------|-------------|
| Small transfer latency | ❌ Missing | 1KB-64KB transfers |
| Concurrent allreduce | ❌ Missing | Multiple simultaneous ops |
| BAR exhaustion | ❌ Missing | Allocator OOM handling |
| Error injection | ❌ Missing | Simulated failures |
| Timeout behavior | ❌ Missing | Hung transfer detection |
| Multi-node | ❌ Missing | MPI + PCIeBAR combo |

---

## 6. Recommendations

### 6.1 Recommended Test Plan for Isolation Testing

```bash
# 1. Capability probe (no hardware needed)
ctest -R "V2_Unit_PCIeBARBackend" --output-on-failure

# 2. Hardware capability check
./build_v2/tests/v2/v2_integration_directp2p --gtest_filter="*ProbeCapabilities*"

# 3. Single transfer correctness
./build_v2_integration/tests/v2/v2_integration_pciebar --gtest_filter="*AllReduce*"

# 4. Performance baseline (64MB)
./build_v2_integration/tests/v2/v2_integration_directp2p --gtest_filter="*Benchmark*"

# 5. NEW: Small transfer latency benchmark
# (needs to be implemented)
LLAMINAR_PCIE_BENCH_SIZES="1024,4096,14336,16384,65536" \
./build_v2_integration/tests/v2/v2_integration_directp2p --gtest_filter="*SmallTransfer*"
```

### 6.2 Estimated Effort to Production-Ready

| Task | Effort | Priority |
|------|--------|----------|
| Add small-transfer latency benchmark | 1 day | **High** |
| Implement reduceScatter() | 2 days | Medium |
| Add retry/timeout logic | 2 days | **High** |
| Add FP16/BF16 reduction kernels | 3 days | Medium |
| Add configuration environment vars | 1 day | Medium |
| Add error injection tests | 2 days | Medium |
| Multi-device (>2) tree reduction | 5 days | Low |
| Documentation | 1 day | Medium |

**Total: ~17 days for full production hardening**

### 6.3 Critical Issues for ROCm Coherence

For the current ROCm coherence investigation, the PCIeBAR backend is **not directly relevant** because:

1. It's designed for **cross-vendor** (CUDA↔ROCm) transfers
2. Single-device ROCm coherence issues occur **within** the ROCm backend
3. The HIP zerocopy coherence bugs are in the ROCm device's own memory management

However, if considering PCIeBAR for ROCm-only testing:
- Could use CUDA as a "coherence observer" to read/write ROCm memory via BAR
- Would bypass HIP runtime entirely, potentially isolating HIP-specific bugs

---

## 7. Code References

### Key Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `PCIeBARBackend::allreduce()` | [PCIeBARBackend.cpp#L337](../../../src/v2/collective/backends/PCIeBARBackend.cpp#L337) | Main allreduce |
| `DirectP2PEngine::initializePCIeBar()` | [DirectP2P.cpp#L365](../../../src/v2/backends/p2p/DirectP2P.cpp#L365) | BAR mapping |
| `DirectP2PEngine::transferViaPCIeBar()` | [DirectP2P.cpp#L542](../../../src/v2/backends/p2p/DirectP2P.cpp#L542) | Core transfer |
| `BackendRouter::selectBackendType()` | [BackendRouter.cpp#L303](../../../../src/v2/collective/BackendRouter.cpp#L303) | Selection logic |

### Build Requirements

```cmake
# Required defines for PCIeBAR support
-DHAVE_CUDA=ON
-DHAVE_ROCM=ON

# Runtime requirements
- CAP_SYS_ADMIN capability (for CUDA IOMEMORY registration)
- AMD GPU with large BAR support (32GB BAR for MI50)
- Resizable BAR enabled in BIOS (for symmetric read/write)
```

---

## Appendix: Performance Data from Code

From [DirectP2P.h](../../../src/v2/backends/p2p/DirectP2P.h) header:

```
## Benchmarked Performance (RTX 3090 + MI50, PCIe 3.0)

| Direction         | Method              | Speed     | Notes                        |
|-------------------|---------------------|-----------|------------------------------|
| NVIDIA → AMD      | CUDA writes to BAR  | 2.65 GB/s | PCIe posted writes           |
| AMD → NVIDIA      | CUDA reads from BAR | 2.67 GB/s | SYMMETRIC with rBAR enabled  |
| Mixed read+write  | Overlapped streams  | ~5.3 GB/s | Full bidirectional PCIe      |

### Without Resizable BAR (legacy 256MB BAR)

| Direction         | Method              | Speed     | Notes                        |
|-------------------|---------------------|-----------|------------------------------|
| NVIDIA → AMD      | CUDA writes to BAR  | ~2.6 GB/s | PCIe posted writes (fast)    |
| AMD → NVIDIA      | CUDA reads from BAR | ~0.8 GB/s | PCIe read completion (slow)  |
```
