# Heterogeneous GPU Tensor Parallelism Research Report

**Date:** January 20, 2026  
**Subject:** Tensor Parallelism for NVIDIA + AMD GPUs in Llaminar V2  
**Author:** Research Analysis  

---

## Executive Summary

This report analyzes Llaminar V2's current tensor parallelism (TP) architecture and its readiness for **heterogeneous GPU configurations** (NVIDIA + AMD). The key findings:

| Aspect | Current State | Heterogeneous TP Readiness |
|--------|--------------|---------------------------|
| Weight Sharding | MPI rank-based, equal splits | ❌ Needs proportional assignment |
| Device-Aware Execution | Single device per stage | 🟡 Multi-device infrastructure exists |
| DeviceGroup Composition | Supports mixed types | ✅ CUDA + ROCm groups supported |
| Collective Backend | Auto-routing to NCCL/RCCL/Host | ✅ Heterogeneous fallback exists |
| Memory Coherence | Per-tensor, device-agnostic | ✅ Works with any device type |

**Bottom Line:** The infrastructure for heterogeneous TP largely exists, but **proportional head assignment** (unequal splits) requires significant changes to WeightManager and stage buffer allocation.

---

## 1. Current Weight Sharding Analysis

### 1.1 Source: [WeightManager.cpp](../../../../src/v2/loaders/WeightManager.cpp)

The current sharding is **MPI rank-based** with **equal splits**:

```cpp
// From WeightManager::getShardedWeight()
int rank = mpi_ctx_->rank();
int world_size = mpi_ctx_->world_size();

// Column-parallel: split rows equally
size_t rows_per_rank = total_rows / world_size;
size_t row_start = rows_per_rank * rank;
size_t row_end = (rank == world_size - 1) ? total_rows : row_start + rows_per_rank;

// Input-parallel: split columns equally  
size_t cols_per_rank = total_cols / world_size;
size_t col_start = cols_per_rank * rank;
size_t col_end = (rank == world_size - 1) ? total_cols : col_start + cols_per_rank;
```

### 1.2 Sharding Modes

| Mode | Application | Collective After |
|------|-------------|------------------|
| `COLUMN_PARALLEL` | Q, K, V projections (heads), Gate/Up | AllGather |
| `ROW_PARALLEL` | Wo (attention output) | AllReduce |
| `INPUT_PARALLEL` | FFN Down | AllReduce |
| `REPLICATE` | Norms, embeddings, biases | None |

**Configuration Source:** Model-specific schema (e.g., `Qwen2SchemaFactory::getWeightShardingConfig()`)

### 1.3 Current Limitations

1. **Equal Split Assumption:**
   ```cpp
   size_t rows_per_rank = total_rows / world_size;  // Always equal
   ```
   No support for `device_0_gets_20_heads, device_1_gets_8_heads`.

2. **MPI Rank = Device:**
   Sharding is tied to `mpi_ctx_->rank()`, not to a device ID within a rank.

3. **World Size Divisibility Check:**
   ```cpp
   if (n_heads % world_size == 0) { /* proceed */ }
   else { /* error: heads must divide evenly */ }
   ```

---

## 2. Proportional Head Assignment Analysis

### 2.1 What's Needed for NVIDIA: 20 heads, AMD: 8 heads?

For Qwen2.5-7B with 28 query heads (14 heads × 2 for GQA simplification):

| Device | Heads | Head Ratio | Memory Share |
|--------|-------|------------|--------------|
| NVIDIA GPU (RTX 4090, 24GB) | 20 | 71.4% | ~71% of QKV weights |
| AMD GPU (MI50, 16GB) | 8 | 28.6% | ~29% of QKV weights |

### 2.2 Changes Required

#### A. WeightManager: Proportional Slicing

```cpp
// NEW: Device-aware sharding config
struct DeviceShardingAssignment {
    DeviceId device;
    int head_start;      // First head index for this device
    int head_count;      // Number of heads assigned
    float weight_ratio;  // Fraction of weight tensor (for FFN)
};

// Modified slicing logic
std::shared_ptr<TensorBase> WeightManager::getWeightForDevice(
    const std::string& name,
    const DeviceShardingAssignment& assignment)
{
    // Q/K/V: slice by head count
    if (isColumnParallelAttention(name)) {
        size_t row_start = assignment.head_start * head_dim;
        size_t row_end = row_start + assignment.head_count * head_dim;
        return loader_.loadTensorRowSlice(name, row_start, row_end, ...);
    }
    
    // FFN: slice by weight_ratio
    if (isColumnParallelFFN(name)) {
        size_t row_start = /* cumulative from prior devices */;
        size_t row_count = static_cast<size_t>(total_rows * assignment.weight_ratio);
        return loader_.loadTensorRowSlice(name, row_start, row_start + row_count, ...);
    }
}
```

#### B. Stage Buffer Allocation: Variable Local Dimensions

```cpp
// Current: assumes equal local dimensions
size_t local_n_heads = n_heads / world_size;

// NEW: device-specific local dimensions
size_t local_n_heads = device_assignment.head_count;
size_t local_d_ff = static_cast<size_t>(d_ff * device_assignment.weight_ratio);
```

#### C. AllGather Output Reconstruction

```cpp
// Current AllGather: equal-sized chunks
// Gather into [seq_len, full_dim] from [seq_len, local_dim]

// NEW: Variable-sized AllGatherV
// Each device contributes different sizes
// Need MPI_Allgatherv or custom NCCL/RCCL implementation
std::vector<int> recv_counts = {20 * head_dim, 8 * head_dim};  // Per-device
std::vector<int> displacements = {0, 20 * head_dim};
MPI_Allgatherv(local_output, local_count, MPI_FLOAT,
               full_output, recv_counts.data(), displacements.data(), ...);
```

### 2.3 Affected Pipeline Components

| Component | Change Required |
|-----------|-----------------|
| `WeightManager` | Proportional row/column slicing |
| `Qwen2Graph` | Variable local_n_heads per device |
| `AllGatherStage` | Variable-sized gather (AllGatherV) |
| `AllReduceStage` | Works unchanged (element-wise, same output size) |
| `FusedQKVStage` | Local dimension from device assignment |
| `AttentionStage` | Local head count from device assignment |
| `RoPEStage` | Local head count |
| `KV Cache` | Proportional allocation per device |

---

## 3. Device-Aware Execution Analysis

### 3.1 Source: [DeviceGraphExecutor.cpp](../../src/v2/execution/DeviceGraphExecutor.cpp)

The `DeviceGraphExecutor` supports **multi-device execution** via `executeMultiDevice()`:

```cpp
bool DeviceGraphExecutor::executeMultiDevice(
    ComputeGraph& graph,
    const std::unordered_map<DeviceId, IDeviceContext*>& contexts)
{
    for (const auto& name : order) {
        auto* node = graph.getNode(name);
        
        // Find appropriate context for this node's device
        IDeviceContext* ctx = default_ctx;
        if (node->device.is_gpu()) {
            auto it = contexts.find(node->device);
            if (it != contexts.end()) {
                ctx = it->second;
            }
        }
        
        if (!executeNode(*node, ctx)) {
            return false;
        }
    }
}
```

### 3.2 Current Capabilities

1. **Per-Node Device Assignment:** Each `ComputeNode` has a `DeviceId device` field
2. **Multiple Contexts:** `executeMultiDevice()` accepts a map of device→context
3. **Automatic Coherence:** `StageCoherence` handles GPU↔CPU sync at boundaries

### 3.3 Limitations for Single-Rank Multi-Device

| Aspect | Current State | Needed for Heterogeneous TP |
|--------|--------------|----------------------------|
| Stage-to-device mapping | Manual per-node | Auto from `DeviceShardingAssignment` |
| Buffer ownership | Shared across stages | Per-device buffers for sharded outputs |
| KV cache | Single instance | Per-device sharded KV caches |

### 3.4 Multi-Device Graph Building

```cpp
// Current: All stages on same device
graph.addNode("layer0_qkv", qkv_stage, device);
graph.addNode("layer0_attn", attn_stage, device);

// NEW: Stages distributed across devices based on sharding
DeviceGroup devices = {cuda_gpu, rocm_gpu};
for (int d = 0; d < devices.size(); d++) {
    graph.addNode("layer0_qkv_dev" + std::to_string(d), 
                  createQKVStage(device_assignments[d]), 
                  devices.devices[d]);
}
// Then add collective stages to synchronize
graph.addNode("layer0_wo_allreduce", allreduce_stage, /* collective device */);
```

---

## 4. Device Groups and Collective Infrastructure

### 4.1 Source: [DeviceGroup.h](../../../../src/v2/collective/DeviceGroup.h)

DeviceGroups **already support heterogeneous configurations**:

```cpp
struct DeviceGroup {
    std::string name;
    std::vector<DeviceId> devices;
    int local_rank = 0;
    
    // Topology metadata
    bool is_homogeneous = true;
    DeviceType primary_type = DeviceType::CPU;
    int cuda_count = 0;
    int rocm_count = 0;
    int cpu_count = 0;
    
    // Predicates
    bool allCUDA() const { return is_homogeneous && primary_type == DeviceType::CUDA; }
    bool allROCm() const { return is_homogeneous && primary_type == DeviceType::ROCm; }
    bool isHeterogeneous() const { return !is_homogeneous; }
    bool hasGPU() const { return cuda_count > 0 || rocm_count > 0; }
};
```

### 4.2 Mixed CUDA + ROCm Group Example

```cpp
// From Test__DeviceGroup.cpp
TEST(Test__DeviceGroup, MixedCUDAAndROCm)
{
    auto group = DeviceGroupBuilder()
        .setName("mixed_gpus")
        .addDevice(DeviceId::cuda(0))
        .addDevice(DeviceId::rocm(0))
        .setLocalRank(0)
        .build();
    
    EXPECT_FALSE(group.allCUDA());
    EXPECT_FALSE(group.allROCm());
    EXPECT_TRUE(group.isHeterogeneous());
    EXPECT_EQ(group.cuda_count, 1);
    EXPECT_EQ(group.rocm_count, 1);
}
```

### 4.3 Backend Selection Logic

Source: [BackendRouter.h](../../../../src/v2/collective/BackendRouter.h)

```cpp
CollectiveBackendType BackendRouter::selectBackendType(const DeviceGroup& group) const
{
    // Priority order:
    // 1. Cross-rank → MPI
    // 2. All CUDA → NCCL
    // 3. All ROCm → RCCL
    // 4. CUDA + ROCm mix → PCIe_BAR (if available) or HOST
    // 5. Default → HOST
    
    if (group.isGlobal()) return CollectiveBackendType::MPI;
    if (group.allCUDA()) return CollectiveBackendType::NCCL;
    if (group.allROCm()) return CollectiveBackendType::RCCL;
    
    // Heterogeneous: prefer PCIe BAR direct P2P if available
    if (group.isHeterogeneous() && hasPCIeBarP2P()) {
        return CollectiveBackendType::PCIE_BAR;
    }
    
    return CollectiveBackendType::HOST;  // Fallback
}
```

### 4.4 Heterogeneous Collective Execution

```cpp
// From DESIGN_HETEROGENEOUS_TENSOR_PARALLELISM.md
bool BackendRouter::executeHeterogeneousAllReduce(
    const DeviceGroup& group,
    void* buffer,
    size_t count,
    CollectiveDataType dtype,
    CollectiveOp op)
{
    // Multi-phase algorithm:
    // Phase 1: AllReduce within same-type subgroups (NCCL for CUDAs, RCCL for ROCms)
    // Phase 2: AllReduce across subgroup representatives via Host
    // Phase 3: Broadcast results back to subgroup members
    
    auto subgroups = DeviceGroupFactory::partitionByType(group);
    // ... execute phases
}
```

---

## 5. Memory Coherence Analysis

### 5.1 Source: [CPUTensors.h](../../../../src/v2/tensors/cpu/CPUTensors.h)

The coherence system is **device-agnostic** - it works with any `DeviceId`:

```cpp
class TensorBase {
protected:
    // Coherence state - tracks which copy is authoritative
    std::optional<DeviceId> gpu_device_;  // Current GPU device (if any)
    void* gpu_data_ptr_ = nullptr;         // Device memory pointer
    bool device_valid_ = false;            // GPU data is authoritative
    bool host_valid_ = true;               // CPU data is authoritative
    
public:
    // Upload to ANY device type (CUDA or ROCm)
    virtual bool ensureOnDevice(DeviceId target_device);
    
    // Mark device as authoritative (works for any device type)
    virtual void mark_device_dirty();
    
    // Sync back to host (from any device type)
    virtual bool ensureOnHost();
};
```

### 5.2 Device-Type Dispatch in Coherence

```cpp
// From TensorBase.cpp (implementation)
bool TensorBase::ensureOnDevice(DeviceId target_device) {
    if (target_device.is_cuda()) {
        // Use CUDA backend for allocation and memcpy
        auto* backend = BackendManager::instance().getCUDABackend();
        gpu_data_ptr_ = backend->allocate(byte_size());
        backend->hostToDevice(gpu_data_ptr_, raw_host_data_ptr(), byte_size());
    } else if (target_device.is_rocm()) {
        // Use ROCm backend for allocation and memcpy
        auto* backend = BackendManager::instance().getROCmBackend();
        gpu_data_ptr_ = backend->allocate(byte_size());
        backend->hostToDevice(gpu_data_ptr_, raw_host_data_ptr(), byte_size());
    }
    gpu_device_ = target_device;
    device_valid_ = true;
    return true;
}
```

### 5.3 StageCoherence Helper

Source: [StageCoherence.h](../../src/v2/execution/StageCoherence.h)

```cpp
// Automatic coherence at stage boundaries - device-type agnostic
bool cohereInputs(const std::vector<CoherenceBuffer>& inputs, DeviceId target_device) {
    for (const auto& buf : inputs) {
        if (auto* tensor = dynamic_cast<TensorBase*>(buf.tensor)) {
            tensor->ensureOnDevice(target_device);  // Works for CUDA or ROCm
        }
    }
    return true;
}

void markOutputsDirty(const std::vector<CoherenceBuffer>& outputs) {
    for (const auto& buf : outputs) {
        if (auto* tensor = dynamic_cast<TensorBase*>(buf.tensor)) {
            tensor->mark_device_dirty();  // Device-agnostic
        }
    }
}
```

### 5.4 Coherence for Heterogeneous TP

| Scenario | Coherence Behavior |
|----------|-------------------|
| NVIDIA stage → AMD stage | NVIDIA output: `mark_device_dirty()` → AMD stage: needs explicit transfer |
| AMD stage → AllReduce | AMD output on AMD device → Host backend stages via CPU |
| Cross-device collective | Host backend does: GPU0→Host, GPU1→Host, reduce, Host→GPU0, Host→GPU1 |

**Key Insight:** Coherence is per-tensor, not per-device-pair. Each tensor tracks its own authoritative location. Cross-device data movement happens through collective operations, not automatic coherence.

---

## 6. Implementation Complexity Assessment

### 6.1 Required Changes Summary

| Component | Complexity | Description |
|-----------|------------|-------------|
| `DeviceShardingAssignment` struct | 🟢 Low | New config struct |
| `WeightManager::getWeightForDevice()` | 🟡 Medium | Proportional slicing logic |
| `PlacementStrategy` extension | 🟡 Medium | Device→head mapping |
| `Qwen2Graph` variable local dimensions | 🔴 High | Per-device buffer specs |
| `AllGatherV` implementation | 🔴 High | Variable-sized gather |
| Per-device KV cache allocation | 🟡 Medium | Partitioned KV caches |
| Testing infrastructure | 🔴 High | Heterogeneous parity tests |

### 6.2 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Numerical divergence (different GPUs) | Medium | High | Parity tests with tolerance |
| Collective synchronization deadlocks | Medium | High | Barrier placement, logging |
| Memory fragmentation | Low | Medium | Unified memory pools |
| Performance regression | High | Medium | Careful profiling per-device |
| PCIe bandwidth bottleneck | High | High | Minimize cross-device transfers |

### 6.3 Implementation Phases

**Phase 1: Infrastructure (1-2 weeks)**
- Add `DeviceShardingAssignment` struct
- Extend `PlacementStrategy` for device→head mapping
- Unit tests for proportional calculations

**Phase 2: Weight Sharding (2-3 weeks)**
- Modify `WeightManager` for proportional slicing
- Add `loadTensorRowSliceProportional()` API
- Test with dummy configurations

**Phase 3: Graph Building (3-4 weeks)**
- Modify `Qwen2Graph` for variable local dimensions
- Implement `AllGatherV` stage
- Per-device KV cache allocation

**Phase 4: Integration (2-3 weeks)**
- End-to-end heterogeneous inference
- Parity testing against single-device
- Performance profiling and optimization

---

## 7. Recommendations

### 7.1 Short-Term (Enable Basic Heterogeneous TP)

1. **Start with equal splits across device types:**
   - If NVIDIA has 4× memory of AMD, still assign equal heads initially
   - This validates collective infrastructure without proportional complexity
   
2. **Use Host backend for cross-device collectives:**
   - Simpler than PCIe BAR, already implemented
   - Performance acceptable for initial validation

3. **Single rank with multiple devices:**
   - Avoid MPI complexity initially
   - `DeviceGroup` already supports this

### 7.2 Medium-Term (Proportional Assignment)

1. **Implement `DeviceShardingAssignment` configuration:**
   ```yaml
   devices:
     - type: cuda
       index: 0
       head_count: 20
       ffn_ratio: 0.71
     - type: rocm
       index: 0
       head_count: 8
       ffn_ratio: 0.29
   ```

2. **Implement `AllGatherV` for variable-sized outputs**

3. **Profile and optimize cross-device transfer:**
   - Evaluate PCIe BAR backend
   - Consider NVLink for multi-NVIDIA setups

### 7.3 Long-Term (Production-Ready)

1. **Auto-tuning for device assignment:**
   - Profile memory bandwidth, compute, transfer speed
   - Automatically determine optimal head distribution

2. **Hierarchical TP:**
   - Intra-node: NCCL/RCCL for same-type devices
   - Cross-type: Host or PCIe BAR
   - Inter-node: MPI

3. **Dynamic rebalancing:**
   - Adjust load based on runtime performance metrics

---

## 8. Conclusion

Llaminar V2 has **solid infrastructure** for heterogeneous GPU tensor parallelism:

✅ **Ready:**
- DeviceGroup supports CUDA + ROCm
- BackendRouter auto-selects appropriate collective backend
- Memory coherence is device-agnostic
- Host backend provides heterogeneous fallback
- PCIe BAR backend design exists for direct GPU-GPU

❌ **Not Ready:**
- Equal-split assumption in WeightManager
- Fixed `local_n_heads = n_heads / world_size` everywhere
- AllGather assumes equal chunk sizes
- No configuration mechanism for proportional assignment

**Recommended Next Step:** Implement `DeviceShardingAssignment` configuration and modify `WeightManager::getWeightForDevice()` to support proportional slicing. This unblocks all downstream components.

---

## Appendix: Key Source Files

| File | Purpose |
|------|---------|
| [src/v2/loaders/WeightManager.cpp](../../../../src/v2/loaders/WeightManager.cpp) | Weight sharding implementation |
| [src/v2/execution/DeviceGraphExecutor.cpp](../../src/v2/execution/DeviceGraphExecutor.cpp) | Multi-device execution |
| [src/v2/collective/DeviceGroup.h](../../../../src/v2/collective/DeviceGroup.h) | Device group definition |
| [src/v2/collective/BackendRouter.h](../../../../src/v2/collective/BackendRouter.h) | Backend selection logic |
| [src/v2/execution/CollectiveContext.h](../../src/v2/execution/CollectiveContext.h) | Collective operation context |
| [src/v2/tensors/cpu/CPUTensors.h](../../../../src/v2/tensors/cpu/CPUTensors.h) | Memory coherence methods |
| [src/v2/execution/StageCoherence.h](../../src/v2/execution/StageCoherence.h) | Automatic stage coherence |
| [docs/v2/projects/2026-01/DESIGN_HETEROGENEOUS_TENSOR_PARALLELISM.md](DESIGN_HETEROGENEOUS_TENSOR_PARALLELISM.md) | Full design document |
