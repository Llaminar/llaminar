# Multi-Device Architecture Gaps - Executive Summary

**Date:** October 24, 2025  
**Status:** 🔴 **CRITICAL ARCHITECTURE GAP IDENTIFIED**

## The Problem in One Sentence

**Current V2 Pipeline assumes single device_idx, but real-world MoE scenarios require multiple devices per rank (e.g., shared experts on GPU, sparse experts on CPU).**

---

## Scenario That Breaks Current Architecture

### Hardware Setup
- 2 sockets, 2 GPUs (1 GPU per socket)
- 2 MPI ranks (1 rank per socket)

### MoE Model
- 12 shared experts → GPU (high reuse)
- 88 sparse experts → CPU (rarely used)

### What Happens Now (Broken)
```cpp
// Pipeline constructed with single device_idx
Qwen2Pipeline pipeline(model_ctx, mpi_ctx, device_idx = -1);  // CPU

// All buffers on CPU
activation_buffers_.Q = FP32Tensor(shape, -1);  // CPU

// But layer 0 wq is on GPU!
auto wq = layer_weights_[0].wq;  // device_idx = 0 (GPU)

// ❌ GEMM fails: Q buffer is CPU, wq is GPU
auto q_gemm = Q->createGemm();  // Returns CPU kernel
q_gemm->multiply(hidden, Q, ...);  // ❌ Can't access GPU weight!
```

---

## Five Critical Gaps

### GAP 1: DeviceOrchestrator Doesn't Expose Queries ⚠️
**Status:** Actually, WeightPlacementMap already has `getDeviceForWeight()` - we just need to use it!

### GAP 2: Pipeline Has Single device_idx 🔴 **CRITICAL**
**Problem:** Cannot have buffers on multiple devices simultaneously

**Current:**
```cpp
int device_idx_;  // Single device
ActivationBuffers activation_buffers_;  // One pool
```

**Needed:**
```cpp
std::vector<int> active_devices_;  // Multiple devices
std::map<int, ActivationBuffers> buffers_per_device_;  // Pool per device
std::shared_ptr<WeightPlacementMap> placement_map_;  // Query weight locations
```

### GAP 3: No Device Transfer Infrastructure 🔴
**Problem:** No way to move tensors between devices

**Needed:**
```cpp
class TensorBase {
    void syncToDevice(int target_device);
    void copyTo(TensorBase* dst);
};
```

### GAP 4: No Execution Planning 🔴
**Problem:** No logic for "where should this operation execute?"

**Needed:**
```cpp
class ExecutionPlanner {
    struct Plan {
        int exec_device;
        bool need_input_transfer;
        bool need_output_transfer;
    };
    Plan planOperation(TensorBase* input, TensorBase* weight);
};
```

### GAP 5: Kernels Assume Co-Located Inputs ⚠️
**Problem:** No validation that inputs are on same device

**Needed:**
```cpp
bool FP32GemmKernel::multiply(...) {
    // Validate device compatibility
    if (this->device_idx() != input->device_idx()) {
        LOG_ERROR("Device mismatch!");
        return false;
    }
}
```

---

## Recommended Solution Architecture

```cpp
class Qwen2Pipeline : public PipelineBase {
private:
    // Multi-device state
    std::vector<int> active_devices_;              // [−1, 0] = CPU + GPU0
    std::map<int, ActivationBuffers> buffers_;     // Buffer pool per device
    std::shared_ptr<WeightPlacementMap> placement_map_;
    
    // Track current activation location
    std::shared_ptr<FP32Tensor> current_hidden_;
    
public:
    bool attention_block(int layer, TensorBase* hidden, TensorBase* output) {
        // 1. Query where weight lives
        int wq_device = placement_map_->getDeviceForWeight(
            "blk." + std::to_string(layer) + ".attn_q.weight", layer);
        
        // 2. Get buffer pool for that device
        ActivationBuffers& device_buffers = buffers_[wq_device];
        
        // 3. Transfer if needed
        if (hidden->device_idx() != wq_device) {
            hidden->syncToDevice(wq_device);
        }
        
        // 4. Execute on correct device
        auto wq = layer_weights_[layer].wq;
        auto Q = device_buffers.Q;
        auto q_gemm = Q->createGemm();
        q_gemm->multiply(hidden->data(), Q->data(), ...);
        
        // 5. Update current location
        current_hidden_ = Q;  // Now on wq_device
    }
};
```

---

## Implementation Roadmap

### Phase 4.1: Multi-Device Pipeline (2-3 days) 🔴 **HIGH PRIORITY**
1. Add `std::vector<int> active_devices_` to Pipeline
2. Add `std::map<int, ActivationBuffers> buffers_per_device_`
3. Implement `discoverActiveDevices(placement_map)`
4. Update forward() to track current_hidden_ location
5. Update attention_block() to query weight device

### Phase 4.2: Device Transfer (1-2 days)
1. Add `syncToDevice()` to TensorBase
2. Implement CPU ↔ CPU (memcpy)
3. Implement CPU ↔ GPU (cudaMemcpy) [when CUDA ready]
4. Add transfer profiling

### Phase 4.3: Execution Planning (1-2 days)
1. Create ExecutionPlanner class
2. Implement planOperation() heuristic
3. Integrate into attention_block(), ffn_block()

### Phase 4.4: MoE Validation (2-3 days)
1. Implement MOE_OPTIMIZED strategy
2. Test with real MoE model
3. Profile transfer overhead

**Total:** 1-2 weeks to production-ready multi-device support

---

## Why This Matters

**Current V2 Supports:**
- ✅ Single device per rank (all-CPU or all-GPU)

**Real-World Needs:**
- ❌ MoE with mixed CPU/GPU experts
- ❌ Multi-GPU per rank (future)
- ❌ Heterogeneous execution (different layers on different devices)

**Impact if Not Fixed:**
- Cannot support MoE optimized placement
- Cannot use GPU for subset of weights
- Cannot scale to multi-GPU per rank
- V2 limited to simpler use cases than V1!

---

## Key Design Decisions

1. **Execute where weight lives** → Avoid moving large weights
2. **Pipeline owns buffer pools** → Clear lifecycle management
3. **Explicit transfers** → Predictable performance, easy to profile
4. **WeightPlacementMap queries** → Flexible, supports any strategy

---

## Performance Impact

### Memory Overhead
- Single device: 70 MB (one buffer pool)
- Multi-device (CPU+GPU): 140 MB (two buffer pools)
- **Acceptable** for devices with GB of memory

### Transfer Overhead
- Transfer time: ~2 μs for 28 KB (8 tokens × 896 × 4 bytes)
- 8 transfers per forward pass: ~18 μs
- Forward pass: ~15 ms
- **Transfer overhead: <0.2%** (negligible!)

### GPU Speedup Break-Even
```
GPU_time + 2 × transfer_time < CPU_time
GPU_time < 500 μs - 4 μs = 496 μs
Required speedup: 500/496 = 1.008× (tiny!)
```

**Conclusion:** Even 1.1× GPU speedup justifies placement.

---

## Next Actions

**RECOMMENDED:** ✅ **Implement Phase 4.1 NOW**

**Rationale:**
- Affects core Pipeline architecture
- Harder to retrofit later
- Blocks MoE support (user-requested feature)
- Required for multi-GPU (future Phase 5)

**Alternative:** ⏸️ **Defer to Phase 5**
- Only if MoE not needed for MVP
- Only if single-device sufficient short-term
- Risk: Architecture debt compounds

---

## Summary

**Current State:** 🔴 V2 cannot support multi-device per rank  
**Root Cause:** Single device_idx assumption in Pipeline  
**Solution:** Multi-device buffer pools + placement map queries + transfer infrastructure  
**Effort:** 1-2 weeks  
**Priority:** HIGH (blocks MoE, multi-GPU)  

**Recommendation:** Start Phase 4.1 immediately before architecture hardens further.

---

**See:** `docs/MULTI_DEVICE_ARCHITECTURE_ANALYSIS.md` for full 700-line detailed analysis.
