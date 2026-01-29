# Heterogeneous Multi-Device AllReduce Design

## Status: DRAFT → V1 DESIGN COMPLETE
**Author:** David Sanftenberg  
**Date:** January 28, 2026  
**Target:** Llaminar V2

### V1 Scope
- [x] Fixed bridge selection: `cuda:0` + `rocm:0`
- [x] Simple 3-phase allreduce (reduce → bridge → broadcast)
- [x] Reduce-scatter + allgather pattern for large tensors
- [x] Chunk-based pipelining (Phase 2 → Phase 3 overlap)
- [ ] Implementation
- [ ] Testing

---

## 1. Problem Statement

The current `PCIeBARBackend` only supports **2-device allreduce** (1 CUDA + 1 ROCm). For larger heterogeneous configurations, the second device of each type is silently ignored:

| Configuration | Current Behavior | Desired Behavior |
|---------------|------------------|------------------|
| 1 CUDA + 1 AMD | ✅ Full allreduce | ✅ Full allreduce |
| 1 CUDA + 2 AMD | ⚠️ Only 2 GPUs participate | All 3 GPUs participate |
| 2 CUDA + 3 AMD | ⚠️ Only 2 GPUs participate | All 5 GPUs participate |
| 3 CUDA + 2 AMD | ⚠️ Only 2 GPUs participate | All 5 GPUs participate |

---

## 2. Design Goals

1. **Correctness**: All devices receive the globally reduced result
2. **Performance**: Maximize use of high-bandwidth same-vendor links (NVLink/xGMI)
3. **Minimize Cross-Vendor Traffic**: PCIe BAR is the bottleneck (~2.6 GB/s vs ~300+ GB/s NVLink)
4. **Simplicity**: Reuse existing NCCL, RCCL, and PCIeBAR backends
5. **Scalability**: Pattern should work for N CUDA + M ROCm devices

---

## 3. Proposed Architecture: Bridge-Mediated Hierarchical AllReduce

### 3.1 Core Concept

Nominate one CUDA device and one ROCm device as **bridge devices**. These form the cross-vendor communication link. All other devices communicate within their vendor domain using NCCL/RCCL.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HETEROGENEOUS ALLREDUCE TOPOLOGY                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   NVIDIA Domain (NCCL)              AMD Domain (RCCL)                   │
│   ════════════════════              ═══════════════════                 │
│                                                                         │
│   ┌──────────┐                          ┌──────────┐                    │
│   │ cuda:0   │◄─────── NVLink ─────────►│ cuda:1   │                    │
│   │ (BRIDGE) │                          │          │                    │
│   └────┬─────┘                          └──────────┘                    │
│        │                                                                │
│        │ PCIe BAR (~2.6 GB/s)                                           │
│        │                                                                │
│   ┌────▼─────┐                          ┌──────────┐                    │
│   │ rocm:0   │◄─────── xGMI ───────────►│ rocm:1   │                    │
│   │ (BRIDGE) │                          │          │                    │
│   └──────────┘                          └──────────┘                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Bridge Selection (V1: Simple)

**For the first iteration, we use a simple fixed selection:**

- **CUDA Bridge**: `cuda:0`
- **ROCm Bridge**: `rocm:0`

This avoids complexity around topology detection and benchmarking. Future iterations may add:
- PCIe proximity detection (same root complex / CPU socket)
- Dynamic benchmark-based selection
- Load balancing for asymmetric configs

---

## 4. Algorithm: Three-Phase Hierarchical AllReduce

### 4.1 Phase Overview

```
Phase 1: Intra-Domain Reduce-Scatter (parallel)
    - NCCL reduce-scatter within NVIDIA GPUs → bridge has partial sum
    - RCCL reduce-scatter within AMD GPUs → bridge has partial sum

Phase 2: Cross-Domain Bridge AllReduce (sequential)
    - PCIeBAR allreduce between cuda:bridge ↔ rocm:bridge
    - Now both bridges have the global sum

Phase 3: Intra-Domain Broadcast (parallel)
    - NCCL broadcast from cuda:bridge → all NVIDIA GPUs
    - RCCL broadcast from rocm:bridge → all AMD GPUs
```

### 4.2 Detailed Algorithm

```
INPUT:  buffer[i] on device[i], all with count elements
OUTPUT: buffer[i] = sum(all buffer[j]) on every device

// Configuration
cuda_devices = [cuda:0, cuda:1, ...]  // N CUDA devices
rocm_devices = [rocm:0, rocm:1, ...]  // M ROCm devices
cuda_bridge = cuda:0
rocm_bridge = rocm:0

// Phase 1: Intra-Domain Reduce (parallel execution)
parallel {
    if (N > 1) {
        NCCL_Reduce(cuda_devices, root=cuda_bridge, op=SUM)
        // cuda_bridge now has sum of all CUDA buffers
    }
    
    if (M > 1) {
        RCCL_Reduce(rocm_devices, root=rocm_bridge, op=SUM)
        // rocm_bridge now has sum of all ROCm buffers
    }
}

// Phase 2: Cross-Domain Bridge Exchange
PCIeBAR_AllReduce(cuda_bridge, rocm_bridge, op=SUM)
// Both bridges now have global sum

// Phase 3: Intra-Domain Broadcast (parallel execution)
parallel {
    if (N > 1) {
        NCCL_Broadcast(cuda_devices, root=cuda_bridge)
        // All CUDA devices have global sum
    }
    
    if (M > 1) {
        RCCL_Broadcast(rocm_devices, root=rocm_bridge)
        // All ROCm devices have global sum
    }
}
```

### 4.3 Complexity Analysis

| Phase | Time Complexity | Bandwidth Used |
|-------|-----------------|----------------|
| Phase 1 (Reduce) | O(log N) + O(log M) | NVLink + xGMI (parallel) |
| Phase 2 (Bridge) | O(1) | PCIe BAR |
| Phase 3 (Broadcast) | O(log N) + O(log M) | NVLink + xGMI (parallel) |

**Total**: O(log(max(N,M))) + O(PCIe_latency)

The PCIe BAR transfer is the **critical path bottleneck**.

---

## 5. Large Tensor Optimization: Reduce-Scatter + AllGather Pattern

**PLANNED FOR V1** - For large tensors (≥ threshold TBD), use reduce-scatter + allgather to minimize cross-vendor traffic:

### 5.1 Algorithm Overview

```
Phase 1: Intra-Domain Reduce-Scatter (parallel)
    - NCCL reduce-scatter within NVIDIA GPUs
    - RCCL reduce-scatter within AMD GPUs
    - Each bridge gets 1/N or 1/M of the partially reduced data

Phase 2: Bridge Exchange (pipelined chunks)
    - Exchange only the bridge's slice via PCIe BAR
    - Chunk-based pipelining overlaps transfer with next phase

Phase 3: Intra-Domain AllGather (parallel, pipelined)
    - NCCL allgather to reassemble full result on NVIDIA GPUs
    - RCCL allgather to reassemble full result on AMD GPUs
    - Can begin as soon as each chunk arrives from Phase 2
```

### 5.2 Bandwidth Savings

| Configuration | Reduce+Broadcast | Reduce-Scatter+AllGather | Savings |
|---------------|------------------|--------------------------|--------|
| 2 CUDA + 2 AMD | 100% of tensor | 50% of tensor | **2x** |
| 4 CUDA + 4 AMD | 100% of tensor | 25% of tensor | **4x** |
| 8 CUDA + 8 AMD | 100% of tensor | 12.5% of tensor | **8x** |

For N CUDA + M AMD, bridge traffic = `tensor_size / max(N, M)`

### 5.3 Pipelining Phase 2 → Phase 3

The bridge exchange (Phase 2) can be **pipelined** with the intra-domain allgather (Phase 3):

```
Time →
╔═══════════════════════════════════════════════════════════════════╗
║ Phase 2:  [chunk0]──[chunk1]──[chunk2]──[chunk3]                  ║
║ Phase 3:       [AG0]────[AG1]────[AG2]────[AG3]                   ║
╚═══════════════════════════════════════════════════════════════════╝

AG = AllGather for that chunk
```

As each chunk completes the PCIe BAR transfer, the allgather for that chunk can begin immediately. This hides allgather latency behind bridge transfer time.

### 5.4 Threshold Selection

Use reduce-scatter pattern when:
```cpp
constexpr size_t REDUCE_SCATTER_THRESHOLD = 4 * 1024 * 1024;  // 4 MB

if (tensor_bytes >= REDUCE_SCATTER_THRESHOLD && max(N, M) > 1) {
    return allreduceReduceScatterPattern(...);
} else {
    return allreduceSimplePattern(...);
}
```

**Rationale**: Below 4 MB, the overhead of extra collective calls outweighs bandwidth savings.

---

## 6. Implementation Plan

### 6.1 New Class: `HeterogeneousAllReduceBackend`

```cpp
class HeterogeneousAllReduceBackend : public ICollectiveBackend {
public:
    // Wraps NCCL, RCCL, and PCIeBAR backends
    bool initialize(const DeviceGroup& group) override;
    
    bool allreduce(void* buffer, size_t count, 
                   CollectiveDataType dtype, CollectiveOp op) override;

private:
    // Sub-backends
    std::unique_ptr<NCCLBackend> nccl_backend_;      // NVIDIA domain
    std::unique_ptr<RCCLBackend> rccl_backend_;      // AMD domain  
    std::unique_ptr<PCIeBARBackend> bridge_backend_; // Cross-domain
    
    // Bridge device selection
    DeviceId cuda_bridge_;
    DeviceId rocm_bridge_;
    
    // Device groups
    std::vector<DeviceId> cuda_devices_;
    std::vector<DeviceId> rocm_devices_;
    
    // Phase execution
    bool executePhase1_IntraDomainReduce(...);
    bool executePhase2_BridgeExchange(...);
    bool executePhase3_IntraDomainBroadcast(...);
};
```

### 6.2 Backend Selection Logic Update

```cpp
CollectiveBackendType LocalTPContext::selectBackend() {
    // Count device types
    int num_cuda = countDevices(DeviceType::CUDA);
    int num_rocm = countDevices(DeviceType::ROCm);
    
    // Homogeneous cases - use native backend
    if (num_cuda > 0 && num_rocm == 0) return NCCL;
    if (num_rocm > 0 && num_cuda == 0) return RCCL;
    
    // Heterogeneous cases
    if (num_cuda > 0 && num_rocm > 0) {
        if (num_cuda == 1 && num_rocm == 1) {
            return PCIE_BAR;  // Simple 2-device case
        }
        return HETEROGENEOUS;  // New hierarchical backend
    }
    
    return HOST;  // Fallback
}
```

### 6.3 File Structure

```
src/v2/collective/backends/
├── HeterogeneousBackend.h
├── HeterogeneousBackend.cpp
├── NCCLBackend.h          (existing)
├── RCCLBackend.h          (existing)
└── PCIeBARBackend.h       (existing)
```

---

## 7. Synchronization Considerations

### 7.1 The Barrier Problem

NCCL and RCCL are **blocking collectives** - all participants must call the collective before any can proceed. Our hierarchical approach requires careful synchronization:

```
Thread 0 (manages cuda:0, cuda:1):
    NCCL_Reduce(...)         // Blocks until all CUDA participate
    PCIeBAR_AllReduce(...)   // Must sync with Thread 1
    NCCL_Broadcast(...)

Thread 1 (manages rocm:0, rocm:1):
    RCCL_Reduce(...)         // Blocks until all ROCm participate
    PCIeBAR_AllReduce(...)   // Must sync with Thread 0
    RCCL_Broadcast(...)
```

### 7.2 Proposed Synchronization

Use **inter-thread barriers** between phases:

```cpp
std::barrier phase_barrier(2);  // 2 threads (CUDA manager + ROCm manager)

// Thread 0 (CUDA)
nccl_reduce();
phase_barrier.arrive_and_wait();  // Sync point 1
pcie_bar_allreduce();
phase_barrier.arrive_and_wait();  // Sync point 2
nccl_broadcast();

// Thread 1 (ROCm)
rccl_reduce();
phase_barrier.arrive_and_wait();  // Sync point 1
pcie_bar_allreduce();
phase_barrier.arrive_and_wait();  // Sync point 2
rccl_broadcast();
```

---

## 8. Performance Projections

### 8.1 Example: 2 CUDA + 2 AMD, 100MB tensor

| Phase | Data Volume | Bandwidth | Time |
|-------|-------------|-----------|------|
| Phase 1: NCCL Reduce | 100 MB | 300 GB/s (NVLink) | ~0.3 ms |
| Phase 1: RCCL Reduce | 100 MB | 200 GB/s (xGMI) | ~0.5 ms |
| Phase 2: PCIeBAR | 100 MB | 2.6 GB/s | **~38 ms** |
| Phase 3: NCCL Bcast | 100 MB | 300 GB/s | ~0.3 ms |
| Phase 3: RCCL Bcast | 100 MB | 200 GB/s | ~0.5 ms |
| **Total** | | | **~40 ms** |

**Bottleneck**: PCIe BAR is 100x slower than NVLink/xGMI.

### 8.2 Comparison with Host-Staged

Current alternative (no PCIe BAR):
```
GPU → Host: ~12 GB/s PCIe
Host reduce: ~50 GB/s memcpy + CPU
Host → GPU: ~12 GB/s PCIe
```

For 100 MB: ~17 ms (faster for small tensors due to lower latency)

**Crossover point**: PCIe BAR wins for latency-sensitive small tensors; host staging may win for large tensors on fast PCIe 4.0/5.0 systems.

---

## 9. Edge Cases

### 9.1 Single Device Per Domain

If N=1 or M=1, skip the intra-domain collective:

```cpp
if (cuda_devices_.size() == 1) {
    // Skip NCCL reduce/broadcast, bridge IS the only CUDA device
}
```

### 9.2 Asymmetric Configurations

For very asymmetric configs (e.g., 8 CUDA + 1 ROCm):
- The single ROCm device becomes a bottleneck
- Consider **weighted sharding** to give it less work
- Or use ROCm only for overflow layers (PP, not TP)

### 9.3 Bridge Device Failure

If the bridge device fails or becomes unavailable:
- Fallback to host-staged allreduce
- Or dynamically select alternate bridge

---

## 10. Testing Plan

### 10.1 Unit Tests

- [ ] `Test__HeterogeneousBackend_BridgeSelection`
- [ ] `Test__HeterogeneousBackend_Phase1_NCCLReduce`
- [ ] `Test__HeterogeneousBackend_Phase1_RCCLReduce`
- [ ] `Test__HeterogeneousBackend_Phase2_BridgeExchange`
- [ ] `Test__HeterogeneousBackend_Phase3_Broadcast`
- [ ] `Test__HeterogeneousBackend_FullAllReduce`

### 10.2 Integration Tests

- [ ] 1 CUDA + 2 ROCm end-to-end inference
- [ ] 2 CUDA + 2 ROCm parity test vs sequential execution
- [ ] 2 CUDA + 3 ROCm stress test

### 10.3 Performance Tests

- [ ] Bandwidth scaling with tensor size
- [ ] Comparison: Heterogeneous vs Host-staged
- [ ] Profile phase timing breakdown

---

## 11. Open Questions

1. ~~**Bridge Selection**: Should we benchmark all CUDA↔ROCm pairs to find optimal bridge?~~  
   **RESOLVED**: V1 uses fixed `cuda:0` + `rocm:0`. Revisit in V2 if needed.

2. ~~**Reduce-Scatter vs Reduce**: Is the complexity of reduce-scatter worth the bandwidth savings?~~  
   **RESOLVED**: Yes, include in V1 for large tensors (≥4 MB threshold).

3. ~~**Async Pipelining**: Can we overlap Phase 2 (bridge) with Phase 3 (broadcast) using non-blocking collectives?~~  
   **RESOLVED**: Yes, implement chunk-based pipelining in V1.

4. **Multi-Node Extension**: How does this pattern extend to multi-node (MPI) scenarios with heterogeneous nodes?

5. **Failure Handling**: What's the recovery strategy if NCCL/RCCL/PCIeBAR fails mid-collective?

---

## 12. References

- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [RCCL GitHub](https://github.com/ROCmSoftwarePlatform/rccl)
- [Megatron-LM Tensor Parallelism](https://arxiv.org/abs/1909.08053)
- [Hierarchical AllReduce in Horovod](https://github.com/horovod/horovod)
- Llaminar `PCIeBARBackend` implementation: [PCIeBARBackend.cpp](../../src/v2/collective/backends/PCIeBARBackend.cpp)

---

## Appendix A: Sequence Diagram

```
    cuda:0        cuda:1        rocm:0        rocm:1
    (bridge)                    (bridge)
       │            │             │             │
       │◄───NCCL────►            │◄───RCCL────►│
       │   Reduce   │             │   Reduce   │
       ├────────────┤             ├────────────┤
       │            │             │             │
       │◄═══════════PCIe BAR═════►│             │
       │   Bridge AllReduce       │             │
       ├──────────────────────────┤             │
       │            │             │             │
       │───NCCL────►│             │────RCCL───►│
       │  Broadcast │             │  Broadcast │
       ▼            ▼             ▼             ▼
    [DONE]       [DONE]        [DONE]        [DONE]
```

---

## Appendix B: Data Flow Example (4 GPUs, SUM)

```
Initial state:
  cuda:0 = [1, 2, 3]
  cuda:1 = [4, 5, 6]
  rocm:0 = [7, 8, 9]
  rocm:1 = [10, 11, 12]

After Phase 1 (Intra-Domain Reduce):
  cuda:0 = [5, 7, 9]      ← sum of cuda:0 + cuda:1
  cuda:1 = [4, 5, 6]      ← unchanged (not root)
  rocm:0 = [17, 19, 21]   ← sum of rocm:0 + rocm:1
  rocm:1 = [10, 11, 12]   ← unchanged (not root)

After Phase 2 (Bridge AllReduce):
  cuda:0 = [22, 26, 30]   ← sum of bridges
  rocm:0 = [22, 26, 30]   ← sum of bridges

After Phase 3 (Intra-Domain Broadcast):
  cuda:0 = [22, 26, 30]   ✓ global sum
  cuda:1 = [22, 26, 30]   ✓ global sum
  rocm:0 = [22, 26, 30]   ✓ global sum
  rocm:1 = [22, 26, 30]   ✓ global sum
```
