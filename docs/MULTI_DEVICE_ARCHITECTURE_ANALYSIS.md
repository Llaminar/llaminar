# Multi-Device Architecture Analysis

**Date:** October 24, 2025  
**Author:** David Sanftenberg  
**Status:** Critical Architecture Gap Identified

## Executive Summary

**Problem:** Current V2 architecture assumes each Pipeline has a **single device_idx**, but real-world scenarios require **multiple devices per rank** (e.g., MoE with shared experts on GPU, sparse experts on CPU).

**Impact:** Cannot support:
- MoE optimized placement (12 shexp on GPU, 88 experts on CPU)
- Multi-GPU per rank (future Phase 4+)
- Heterogeneous execution (different layers on different devices)

**Solution:** Refactor Pipeline to support **multi-device orchestration** with device-aware buffer pools and execution planning.

---

## 1. Scenario Analysis: MoE Dual-Socket + Dual-GPU

### Hardware Setup
```
┌─────────────────────────────────────────────────┐
│ Socket 0           │ Socket 1                   │
├────────────────────┼────────────────────────────┤
│ CPU 0 (28 cores)   │ CPU 1 (28 cores)          │
│ GPU 0 (CUDA)       │ GPU 1 (CUDA)              │
│ MPI Rank 0         │ MPI Rank 1                │
└─────────────────────────────────────────────────┘
```

### Model: MoE with 100 Experts
- **12 shared experts** → GPU (high reuse, worth transfer overhead)
- **88 sparse experts** → CPU (rarely used, avoid GPU memory waste)
- **Attention/FFN layers** → Decision per layer based on memory

### Expected Behavior

**Rank 0 should have:**
- CPU device (-1): 88 sparse expert weights
- GPU 0 device (0): 12 shared expert weights, some attention layers
- **Activation buffers on BOTH devices**

**Forward pass should:**
1. Start with hidden state on CPU (from embedding)
2. For CPU layers: Execute on CPU buffers
3. For GPU layers: 
   - Transfer hidden → GPU
   - Execute on GPU buffers  
   - Transfer result → CPU
4. For MoE: Route to appropriate expert device, aggregate results

---

## 2. Architecture Gap Analysis

### GAP 1: DeviceOrchestrator Doesn't Expose Placement Queries

**Current State:**
```cpp
class DeviceOrchestrator {
    std::shared_ptr<WeightPlacementMap> createPlacementMap(...);
    // ❌ No query API for "where is weight X?"
};
```

**Problem:**
- Pipeline receives WeightPlacementMap but can't query it
- Can't ask "which device is layer.wq on?"
- Can't determine execution device from weight placement

**What We Need:**
```cpp
// Option A: WeightPlacementMap query (already exists!)
int device = placement_map->getDeviceForWeight("blk.0.attn_q.weight", layer_idx);

// Option B: DeviceOrchestrator helper
std::vector<int> active_devices = orchestrator->getActiveDevices();
```

**Status:** ⚠️ **WeightPlacementMap already has getDeviceForWeight()!**  
We just need Pipeline to use it!

---

### GAP 2: Pipeline Has Single device_idx (Most Critical)

**Current State:**
```cpp
class Qwen2Pipeline {
    int device_idx_;  // ❌ Single device assumption
    ActivationBuffers activation_buffers_;  // ❌ One pool, tied to device_idx_
};
```

**Problem:**
- If device_idx_ = -1 (CPU), all buffers are CPU
- If device_idx_ = 0 (GPU), all buffers are GPU  
- **Cannot have buffers on multiple devices simultaneously**

**MoE Scenario Breaks:**
```cpp
// Pipeline constructed with device_idx = -1 (CPU)
Qwen2Pipeline pipeline(model_ctx, mpi_ctx, -1);

// Buffers allocated on CPU
activation_buffers_.Q = FP32Tensor([seq_len, d_model], -1);

// Layer 0 weight is on GPU
auto wq = layer_weights_[0].wq;  // device_idx = 0 (GPU)

// ❌ PROBLEM: Q buffer is CPU, wq is GPU, where does GEMM execute?
auto q_gemm = Q->createGemm();  // Returns CPU kernel
q_gemm->multiply(hidden, Q, ...);  // ❌ Can't access GPU weight!
```

**What We Need:**
```cpp
class Qwen2Pipeline {
    std::vector<int> active_devices_;  // List of devices this rank uses
    std::map<int, ActivationBuffers> buffers_per_device_;  // Buffer pool per device
    std::shared_ptr<WeightPlacementMap> placement_map_;  // Query weight locations
};
```

---

### GAP 3: No Device Transfer Orchestration

**Current State:**
- Kernels assume inputs are co-located
- No cross-device GEMM support
- No staging buffers for transfers

**Problem:**
```cpp
// Hidden state is on CPU
FP32Tensor* hidden = current_hidden_;  // device_idx = -1

// Weight is on GPU
auto wq = layer_weights_[0].wq;  // device_idx = 0

// ❌ No transfer mechanism!
// Need: hidden → transfer → GPU → GEMM → transfer → CPU
```

**What We Need:**
```cpp
class DeviceTransferManager {
    void transfer(TensorBase* src, TensorBase* dst);
    TensorBase* getStagingBuffer(size_t bytes, int device_idx);
};

// Or simpler: built into TensorBase
class TensorBase {
    void copyTo(TensorBase* dst);  // Handles cross-device transfer
    void syncToDevice(int target_device);
    void syncFromDevice();
};
```

---

### GAP 4: No Execution Planning

**Current State:**
```cpp
bool attention_block(int layer, TensorBase* hidden, TensorBase* output) {
    auto wq = layer_weights_[layer].wq;
    auto Q = activation_buffers_.Q;  // ❌ Which device's Q buffer?
    
    auto q_gemm = Q->createGemm();
    q_gemm->multiply(hidden, Q, ...);  // ❌ Where does this execute?
}
```

**Problem:**
- No decision logic for "where should this operation run?"
- No routing of activations to correct device
- No fallback if devices mismatch

**What We Need:**
```cpp
// Before each operation, plan execution
struct ExecutionPlan {
    int exec_device;           // Where operation executes
    bool need_input_transfer;  // Transfer input to exec device?
    bool need_output_transfer; // Transfer output back?
};

ExecutionPlan planOperation(TensorBase* input, TensorBase* weight) {
    int weight_device = weight->device_idx();
    int input_device = input->device_idx();
    
    // Strategy: Execute where weight lives (avoid moving large weights)
    ExecutionPlan plan;
    plan.exec_device = weight_device;
    plan.need_input_transfer = (input_device != weight_device);
    plan.need_output_transfer = false;  // Keep result where computed
    return plan;
}
```

---

### GAP 5: Kernel Interface Doesn't Handle Cross-Device

**Current State:**
```cpp
class ITensorGemm {
    virtual bool multiply(const float* A, float* C, ...) = 0;
    // ❌ Assumes A and weight are co-located
};
```

**Problem:**
- No device affinity checks
- No cross-device operation support
- Silent failures if devices mismatch

**What We Need:**
```cpp
class ITensorGemm {
    virtual bool multiply(const float* A, float* C, ...) = 0;
    virtual int device_idx() const = 0;  // Which device is this kernel for?
    virtual bool supports_device(int device_idx) const = 0;  // Already exists!
};

// In implementation:
bool FP32GemmKernel::multiply(const float* A, float* C, ...) {
    // Validate device compatibility
    if (this->device_idx() != input_device) {
        LOG_ERROR("Device mismatch: kernel on " << device_idx() 
                  << ", input on " << input_device);
        return false;
    }
    // ... execute
}
```

---

## 3. Proposed Solutions

### Solution 1: Multi-Device Pipeline Architecture (RECOMMENDED)

**Core Idea:** Pipeline manages multiple buffer pools, one per active device.

```cpp
class Qwen2Pipeline : public PipelineBase {
private:
    // Multi-device state
    std::vector<int> active_devices_;              // Devices this rank uses
    std::map<int, ActivationBuffers> buffers_;     // Buffer pool per device
    std::shared_ptr<WeightPlacementMap> placement_map_;  // Weight locations
    
    // Current activation tracking
    std::shared_ptr<FP32Tensor> current_hidden_;   // Tracks location
    
public:
    Qwen2Pipeline(std::shared_ptr<ModelContext> model_ctx,
                  std::shared_ptr<MPIContext> mpi_ctx,
                  std::shared_ptr<WeightPlacementMap> placement_map) {
        
        // Discover which devices we need
        active_devices_ = discoverActiveDevices(placement_map);
        
        // Allocate buffer pool for each active device
        for (int device_idx : active_devices_) {
            buffers_[device_idx] = allocateBuffers(max_seq_len, device_idx);
        }
    }
    
private:
    std::vector<int> discoverActiveDevices(
        std::shared_ptr<WeightPlacementMap> map) {
        
        std::set<int> devices;
        
        // Scan all layer weights
        for (int layer = 0; layer < n_layers_; ++layer) {
            devices.insert(map->getDeviceForWeight("blk." + std::to_string(layer) + ".attn_q.weight", layer));
            devices.insert(map->getDeviceForWeight("blk." + std::to_string(layer) + ".ffn_gate.weight", layer));
            // ... check all weight types
        }
        
        return std::vector<int>(devices.begin(), devices.end());
    }
    
    bool attention_block(int layer, TensorBase* hidden, TensorBase* output) {
        // Query weight device
        int wq_device = placement_map_->getDeviceForWeight(
            "blk." + std::to_string(layer) + ".attn_q.weight", layer);
        
        // Get buffer pool for that device
        ActivationBuffers& device_buffers = buffers_[wq_device];
        
        // Transfer if needed
        if (hidden->device_idx() != wq_device) {
            transferToDevice(hidden, wq_device);
        }
        
        // Execute on correct device
        auto wq = layer_weights_[layer].wq;
        auto Q = device_buffers.Q;
        auto q_gemm = Q->createGemm();
        q_gemm->multiply(hidden->data(), Q->data(), ...);
        
        // Update current location
        current_hidden_ = Q;  // Now on wq_device
    }
};
```

**Benefits:**
✅ Supports arbitrary multi-device scenarios  
✅ Explicit buffer management per device  
✅ Clear activation routing logic  
✅ Queries placement map for execution decisions  

**Costs:**
⚠️ More complex initialization  
⚠️ Transfer overhead management needed  
⚠️ Memory overhead (buffers on multiple devices)  

---

### Solution 2: Device Transfer Manager (Complementary)

**Add to TensorBase:**

```cpp
class TensorBase {
public:
    // Device synchronization
    virtual void syncToDevice(int target_device) {
        if (device_idx_ == target_device) return;
        
        // Allocate on target device if needed
        // Copy data: current device → target device
        // Update device_idx_
    }
    
    virtual void copyTo(TensorBase* dst) {
        // Handle cross-device copy automatically
        if (this->device_idx_ == dst->device_idx()) {
            // Same device: memcpy
        } else {
            // Cross-device: use CUDA/ROCm peer copy or CPU staging
        }
    }
};
```

**Or separate manager:**

```cpp
class DeviceTransferManager {
public:
    static void transfer(TensorBase* src, TensorBase* dst) {
        int src_device = src->device_idx();
        int dst_device = dst->device_idx();
        
        if (src_device == dst_device) {
            // Same device
            memcpy(dst->data(), src->data(), src->size_bytes());
        } else if (src_device == -1 && dst_device >= 0) {
            // CPU → GPU
            cudaMemcpy(dst->device_data(), src->data(), 
                      src->size_bytes(), cudaMemcpyHostToDevice);
        } else if (src_device >= 0 && dst_device == -1) {
            // GPU → CPU
            cudaMemcpy(dst->data(), src->device_data(), 
                      src->size_bytes(), cudaMemcpyDeviceToHost);
        } else {
            // GPU → GPU (peer copy or staging)
            // ... implementation
        }
    }
};
```

---

### Solution 3: Execution Planner (Complementary)

**Simple heuristic planner:**

```cpp
class ExecutionPlanner {
public:
    struct Plan {
        int exec_device;
        bool need_input_transfer;
        bool need_output_transfer;
    };
    
    static Plan planOperation(TensorBase* input, TensorBase* weight) {
        Plan plan;
        
        // Strategy: Execute where weight lives (weights are bigger)
        plan.exec_device = weight->device_idx();
        
        // Check if input needs transfer
        plan.need_input_transfer = (input->device_idx() != plan.exec_device);
        
        // Keep output where computed (avoid extra transfer)
        plan.need_output_transfer = false;
        
        return plan;
    }
};

// Usage in Pipeline:
bool attention_block(int layer, TensorBase* hidden, TensorBase* output) {
    auto wq = layer_weights_[layer].wq;
    auto plan = ExecutionPlanner::planOperation(hidden, wq);
    
    // Get buffers for execution device
    ActivationBuffers& buffers = buffers_[plan.exec_device];
    
    // Transfer if needed
    if (plan.need_input_transfer) {
        hidden->syncToDevice(plan.exec_device);
    }
    
    // Execute
    auto q_gemm = buffers.Q->createGemm();
    q_gemm->multiply(hidden->data(), buffers.Q->data(), ...);
    
    // Update tracking
    current_hidden_ = buffers.Q;  // Now on exec_device
}
```

---

## 4. Implementation Roadmap

### Phase 4.1: Multi-Device Pipeline Foundation
**Goal:** Support multiple buffer pools per Pipeline

**Tasks:**
1. ✅ WeightPlacementMap query API (already exists!)
2. ⬜ Add `std::vector<int> active_devices_` to Qwen2Pipeline
3. ⬜ Add `std::map<int, ActivationBuffers> buffers_per_device_`
4. ⬜ Implement `discoverActiveDevices(placement_map)`
5. ⬜ Implement `allocateBuffersPerDevice()`
6. ⬜ Update forward() to track current_hidden_ device location
7. ⬜ Update attention_block() to query weight device
8. ⬜ Add logging: "Executing attention layer X on device Y"

**Test Cases:**
- Single-device path still works (backward compat)
- Multi-device allocation succeeds
- Active device discovery finds all used devices

---

### Phase 4.2: Device Transfer Infrastructure
**Goal:** Enable cross-device data movement

**Tasks:**
1. ⬜ Add `syncToDevice(int target_device)` to TensorBase
2. ⬜ Implement CPU ↔ CPU (memcpy)
3. ⬜ Implement CPU → GPU (cudaMemcpy HtoD) [Phase 4 CUDA]
4. ⬜ Implement GPU → CPU (cudaMemcpy DtoH) [Phase 4 CUDA]
5. ⬜ Implement GPU ↔ GPU peer copy [Phase 4+ Multi-GPU]
6. ⬜ Add transfer profiling/logging

**Test Cases:**
- CPU → CPU transfer
- CPU → GPU transfer (when CUDA enabled)
- GPU → CPU transfer (when CUDA enabled)
- Transfer correctness (data integrity)

---

### Phase 4.3: Execution Planning
**Goal:** Automatic device routing for operations

**Tasks:**
1. ⬜ Create ExecutionPlanner class
2. ⬜ Implement planOperation(input, weight) heuristic
3. ⬜ Integrate into attention_block(), ffn_block()
4. ⬜ Add execution plan logging
5. ⬜ Profile transfer overhead

**Test Cases:**
- Same-device path has zero transfers
- Cross-device path triggers expected transfers
- Execution device matches plan

---

### Phase 4.4: MoE Validation
**Goal:** Prove multi-device architecture with real MoE model

**Tasks:**
1. ⬜ Implement MOE_OPTIMIZED strategy in DeviceOrchestrator
2. ⬜ Test with synthetic MoE (12 shexp GPU, 88 sparse CPU)
3. ⬜ Validate correctness vs single-device baseline
4. ⬜ Profile performance (transfer overhead vs GPU speedup)
5. ⬜ Document optimal MoE placement strategies

**Success Criteria:**
- MoE inference works with mixed CPU/GPU placement
- Shared experts benefit from GPU (>2× speedup vs CPU)
- Sparse experts acceptable on CPU (low overhead)
- Total throughput improvement ≥30% vs all-CPU

---

## 5. Design Decisions

### Decision 1: Pipeline Owns Buffer Pools ✅
**Rationale:** Pipeline controls execution flow, knows sequence length, responsible for memory lifecycle.

**Alternative Rejected:** External buffer manager  
**Why:** Would require complex synchronization between manager and pipeline.

---

### Decision 2: Execute Where Weight Lives ✅
**Rationale:** Weights are larger than activations, avoid moving weights.

**Alternative Considered:** Execute where activation lives  
**Why Rejected:** Activation changes device frequently, weights are static.

**Alternative Considered:** Always execute on GPU  
**Why Rejected:** Some weights may not fit on GPU, CPU-only experts.

---

### Decision 3: WeightPlacementMap Query at Runtime ✅
**Rationale:** Flexible, supports any placement strategy without pipeline changes.

**Alternative Considered:** Bake device_idx into LayerWeights struct  
**Why Rejected:** Duplication, inflexible, harder to change placement dynamically.

---

### Decision 4: Explicit Transfer Management ✅
**Rationale:** Predictable performance, clear cost model, easier to profile.

**Alternative Considered:** Automatic transparent transfer (Unified Memory)  
**Why Rejected:** Unpredictable performance, hidden costs, harder to optimize.

---

## 6. Open Questions

### Q1: Should activations stay on GPU between layers?
**Scenario:** Layers 0-11 on GPU, layers 12-27 on CPU.

**Option A:** Transfer back to CPU after GPU section  
- Pro: Predictable memory usage  
- Con: Extra transfer overhead  

**Option B:** Keep on GPU as long as possible  
- Pro: Fewer transfers  
- Con: Complex residual tracking  

**Recommendation:** Start with Option A (explicit), optimize to Option B later.

---

### Q2: How to handle residual connections across devices?
**Scenario:** Residual from CPU layer needs to add with GPU layer output.

**Option A:** Transfer residual to match layer output device  
**Option B:** Transfer layer output to match residual device  
**Option C:** Always keep residual on "primary device" (e.g., CPU)  

**Recommendation:** Option C (primary device for residuals) for simplicity.

---

### Q3: Should we support async transfers?
**Scenario:** Overlap transfer with computation on other device.

**Option A:** Synchronous transfers (blocking)  
- Pro: Simple, predictable  
- Con: Idle time during transfers  

**Option B:** Async transfers with CUDA streams  
- Pro: Better GPU utilization  
- Con: Complex synchronization, Phase 4+ feature  

**Recommendation:** Option A initially, Option B for Phase 5 optimization.

---

## 7. Performance Implications

### Memory Overhead
**Current:** 1 buffer pool × ~70 MB = 70 MB  
**Multi-device:** N devices × ~70 MB = 140 MB (2 devices), 210 MB (3 devices)

**Mitigation:** Only allocate buffers for active devices (not all possible devices).

---

### Transfer Overhead
**Worst Case:** Every layer on different device → 56 transfers per forward pass  
**Realistic:** Blocks of layers on same device → ~4-8 transfers per forward pass

**MoE Example (Qwen 2.5 0.5B):**
- Hidden size: 896 × 4 bytes = 3.5 KB per token  
- 8 tokens: 28 KB per transfer  
- PCIe 3.0 x16: ~12 GB/s → ~2.3 μs per transfer  
- 8 transfers: ~18 μs overhead (negligible vs ~15 ms forward pass)

**Conclusion:** Transfer overhead is **minimal for small models**.

---

### GPU Speedup Requirement
**Break-Even Analysis:**

For GPU layer to be worth transfer overhead:
```
GPU_time + 2 × transfer_time < CPU_time
GPU_time < CPU_time - 2 × transfer_time

If transfer_time = 2 μs, CPU_time = 500 μs:
GPU_time < 496 μs
Speedup > 500/496 = 1.008× (tiny speedup needed)
```

**Conclusion:** Even modest GPU speedup (1.1×) justifies placement.

---

## 8. Testing Strategy

### Unit Tests
1. **Multi-device buffer allocation**
   - Test: Pipeline with [CPU, GPU0] active devices
   - Verify: 2 buffer pools allocated
   - Verify: Correct device_idx on each buffer

2. **Active device discovery**
   - Test: Placement map with layers 0-11 GPU, 12-27 CPU
   - Verify: active_devices_ = [-1, 0]

3. **Cross-device transfer**
   - Test: Transfer tensor CPU → GPU → CPU
   - Verify: Data integrity (memcmp)

### Integration Tests
1. **Mixed-device attention**
   - Test: Attention with wq on GPU, wk on CPU
   - Verify: Transfers happen, result correct

2. **MoE routing**
   - Test: MoE with shexp GPU, sparse CPU
   - Verify: Correct experts activated, results match single-device baseline

### Performance Tests
1. **Transfer overhead**
   - Measure: Time for CPU ↔ GPU transfer (various sizes)
   - Target: <5 μs for 28 KB

2. **Multi-device throughput**
   - Test: All-CPU vs Mixed CPU/GPU vs All-GPU
   - Measure: Tokens/sec, transfer %, compute %

---

## 9. Risks and Mitigations

### Risk 1: Memory Exhaustion
**Scenario:** Multiple buffer pools exceed available memory.

**Mitigation:**
- Lazy allocation (only create buffers on first use)
- Fallback to single-device if multi-device allocation fails
- User warning: "Multi-device requires N MB, falling back to CPU"

---

### Risk 2: Transfer Bottleneck
**Scenario:** Transfer overhead dominates, no net speedup.

**Mitigation:**
- Batch transfers (transfer multiple tensors in one call)
- Async transfers (Phase 5)
- Smart grouping (co-locate dependent weights)

---

### Risk 3: Complexity Explosion
**Scenario:** Multi-device logic makes code unmaintainable.

**Mitigation:**
- Encapsulate in ExecutionPlanner, DeviceTransferManager
- Comprehensive logging for debugging
- Fallback to single-device mode for testing

---

## 10. Summary and Recommendation

**Current Architecture:** ❌ **Insufficient for multi-device scenarios**

**Critical Gaps:**
1. Single device_idx assumption in Pipeline
2. No cross-device buffer pools
3. No execution planning for mixed-device operations
4. No device transfer infrastructure

**Recommended Path Forward:**

**Phase 4.1** (Immediate - 2-3 days):
- Refactor Pipeline to support multiple buffer pools
- Implement active device discovery
- Add WeightPlacementMap query integration
- Validate with single-device (backward compat test)

**Phase 4.2** (1-2 days):
- Add device transfer methods to TensorBase
- Implement CPU ↔ GPU transfers (CUDA)
- Add transfer logging and profiling

**Phase 4.3** (1-2 days):
- Create ExecutionPlanner class
- Integrate into attention_block(), ffn_block()
- Add execution plan validation

**Phase 4.4** (2-3 days):
- Implement MOE_OPTIMIZED strategy
- Test with real MoE model
- Profile and optimize transfer patterns

**Total Estimate:** 1-2 weeks to production-ready multi-device support

**Success Metrics:**
- ✅ MoE inference works with mixed CPU/GPU placement
- ✅ No regression in single-device performance
- ✅ Transfer overhead <5% of total forward pass time
- ✅ GPU layers show ≥2× speedup vs CPU (when profitable)

---

## 11. Next Actions

1. ⬜ Review this analysis with team
2. ⬜ Prioritize: Phase 4.1 (multi-device buffers) vs other V2 tasks
3. ⬜ Create detailed task breakdown for Phase 4.1
4. ⬜ Set up test environment with multi-GPU system
5. ⬜ Begin implementation (or defer to Phase 5 if not critical for MVP)

**Recommendation:** Implement Phase 4.1 **now** (before MoE support), as it affects core Pipeline architecture and is harder to retrofit later.
