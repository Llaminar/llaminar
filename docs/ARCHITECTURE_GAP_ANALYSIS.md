# Architectural Analysis: Current State vs Goals
**Date**: October 20, 2025  
**Status**: GAP ANALYSIS

## Your Stated Goals

### Goal 1: Weights Stay in Native Quantized Format
**Status**: ✅ **MOSTLY WORKING**

**What exists**:
- `QuantizedTensor` class stores raw bytes (Q4_0, Q6_K, Q8_0, etc.)
- No persistent FP32 backing buffer (data() returns nullptr)
- Decode on demand via `decodeBlock()` method

**Gap**: Need to verify ModelLoader actually creates QuantizedTensor by default
- Older code comments mention "dequantized to FP32 once at load time"
- May have dual code paths (old full-dequant, new QuantizedTensor)
- **Action**: Audit ModelLoader.cpp to ensure quantized → QuantizedTensor path is active

### Goal 2: Persistent Reusable Buffers + JIT Dequantization
**Status**: ✅ **IMPLEMENTED** (QuantSlabCache)

**What exists**:
- `QuantSlabCache` class with 64 MB LRU cache
- Decodes quantized weight columns to BF16 on demand
- Cached slabs reused across forward passes
- Keyed by (weight_ptr, column_start, column_count)

**Implementation** (`src/operators/QuantSlabCache.{h,cpp}`):
```cpp
struct QuantSlab {
    size_t k, n;
    std::vector<bfloat16> data;  // Decoded to BF16 (not FP32!)
};

bool getOrDecode(const QuantizedTensor& tensor,
                 size_t col_start, size_t col_count,
                 QuantSlab& out_slab, bool reuse_allowed);
```

**Current usage**: `MPILinearOperator` lines 165-220
- Checks for QuantizedTensor
- Calls `QuantSlabCache::instance().getOrDecode()`
- Uses decoded BF16 slab in `adaptiveMatMulBF16()`

**Gap**: Need to verify this is THE active code path
- May have fallback to old full-dequant path
- **Action**: Add logging to confirm slab path hit rate >80%

### Goal 3: Pass BF16 Between Pipeline Stages
**Status**: ❌ **BROKEN** (Memory leak discovered)

**What exists**:
- `BF16Tensor` class with `std::vector<bfloat16> data_`
- `bf16_data()` method to access native BF16 pointer
- Designed to store activations in 2 bytes/element

**Critical bug**:
```cpp
// src/tensors/BF16Tensor.h lines 92-115
class BF16Tensor : public TensorBase {
private:
    std::vector<bfloat16> data_;           // 2 bytes/element
    mutable std::vector<float> fp32_cache_; // 4 bytes/element ← LEAK!
    
    float* data() override {
        update_cache();  // Allocates FULL FP32 copy on first call
        return fp32_cache_.data();
    }
};
```

**Consequence**:
- Operators call `tensor->data()` 
- BF16Tensor allocates full FP32 cache (never freed)
- **Memory**: BF16 (50%) + FP32 cache (100%) = 150% vs baseline!
- **Measured**: 4210 MB (FP32) → 5082-6498 MB (BF16+cache) = 21-54% INCREASE

**Root cause**: Interface incompatibility
- `TensorBase` defines `virtual float* data() = 0`
- BF16Tensor must provide FP32 pointer for compatibility
- No way to return BF16 pointer via TensorBase interface

**Gap**: Operators aren't BF16-aware
- Call `data()` blindly → triggers FP32 cache allocation
- Should check tensor type, call `bf16_data()` directly
- **Action**: Refactor ~50+ operator call sites

### Goal 4: Never Fully Dequantize Weights
**Status**: ✅ **WORKING** (when slab path active)

**Implementation**: QuantSlab decodes ONLY requested columns
```cpp
// MPILinearOperator lines 177-186
QuantSlab slab;
QuantSlabCache::instance().getOrDecode(
    *quant_tensor,
    col_start,  // Only decode columns needed by this rank
    col_count,
    slab, /*reuse=*/true
);

// Use decoded BF16 slab (NOT full weight)
adaptiveMatMulBF16(input_data, slab.data.data(), output, m, n, k);
```

**Verification needed**:
- Is slab path the default?
- Are weights actually staying quantized?
- **Action**: Memory profiling should show only 64 MB decode buffer, not full weights

---

## Architecture We Have vs Architecture We Want

### Current Reality

```
ModelLoader
    ↓
❓ QuantizedTensor OR fully dequantized SimpleTensor? (UNCLEAR)
    ↓
MPILinearOperator
    ↓ (if QuantizedTensor detected)
QuantSlabCache::getOrDecode() → BF16 slab (64 MB LRU) ✅
    ↓
adaptiveMatMulBF16() → accumulate to FP32 ✅
    ↓
Store in BF16Tensor
    ↓
❌ BUG: Operators call data() → allocates FP32 cache ❌
    ↓
Memory usage 150% instead of 50%!
```

### Target Architecture

```
ModelLoader
    ↓
✅ QuantizedTensor (Q4_0/Q6_K/Q8_0 raw bytes)
    ↓
MPILinearOperator
    ↓
QuantSlabCache::getOrDecode() → BF16 slab (cached) ✅
    ↓
adaptiveMatMulBF16(FP32_input, BF16_weight) → FP32 accum ✅
    ↓
Convert FP32 → BF16, store in BF16Tensor ✅
    ↓
✅ Operators call bf16_data() directly (NO data() call)
    ↓
Next stage uses BF16 input (no FP32 cache)
```

---

## What Needs To Be Done

### Fix 1: Remove BF16Tensor FP32 Cache (CRITICAL)
**File**: `src/tensors/BF16Tensor.h` lines 92-115

**Current (broken)**:
```cpp
mutable std::vector<float> fp32_cache_;
mutable bool cache_valid_;

float* data() override {
    update_cache();  // Allocates FP32 copy
    return fp32_cache_.data();
}
```

**Target (fixed)**:
```cpp
// Remove fp32_cache_ entirely

float* data() override {
    throw std::runtime_error(
        "BF16Tensor::data() not supported. "
        "Use bf16_data() for BF16-aware operations."
    );
}

bfloat16* bf16_data() { return data_.data(); }
```

**Impact**: Breaks all operators calling data() on BF16 tensors (GOOD - forces fixes)

### Fix 2: Make Operators BF16-Aware
**Files**: `src/operators/MPI{Linear,Attention,SwiGLU,RMSNorm}Operator.cpp`

**Current pattern** (~50+ call sites):
```cpp
float* input_data = input_tensor->data();  // ❌ Triggers cache
```

**Target pattern**:
```cpp
if (auto bf16 = std::dynamic_pointer_cast<BF16Tensor>(input_tensor)) {
    bfloat16* bf16_data = bf16->bf16_data();  // ✅ Direct access
    // Use BF16-aware kernel
} else {
    float* fp32_data = input_tensor->data();  // ✅ FP32 path
}
```

**Scope**: Every operator that touches activations
- MPILinearOperator (Q/K/V/O projections, FFN gate/up/down)
- MPIAttentionOperator (scores, context aggregation)
- MPISwiGLUOperator (gate × silu(up))
- MPIRMSNormOperator (normalization)

### Fix 3: Verify ModelLoader Path
**File**: `src/ModelLoader.cpp`

**Ensure this pattern**:
```cpp
if (is_quantized(tensor_type)) {
    // Store as QuantizedTensor (raw bytes)
    auto quant = std::make_shared<QuantizedTensor>(layout, raw_data);
    weights[name] = quant;
} else {
    // F32/F16: Store as SimpleTensor
    auto simple = std::make_shared<SimpleTensor>(shape, fp32_data);
    weights[name] = simple;
}
```

**Verify NOT doing**:
```cpp
// ❌ OLD PATH (full dequant at load time)
if (is_quantized(tensor_type)) {
    std::vector<float> dequantized = dequantize_all(raw_data);
    auto simple = std::make_shared<SimpleTensor>(shape, dequantized);
}
```

### Fix 4: Ensure QuantSlab Path is Default
**File**: `src/operators/MPILinearOperator.cpp` lines 165-220

**Current gating**:
```cpp
bool use_slab = debugEnv().quant.slab_enable && 
                debugEnv().quant_slab.enable;
```

**Verify these flags default to TRUE**:
- `LLAMINAR_QUANT_SLAB_ENABLE=1` (should be default)
- Check `src/utils/DebugEnv.cpp` for defaults

**Fallback path** (lines 221+):
- If !use_slab or weight not QuantizedTensor
- Falls back to old `weight->data()` call
- For QuantizedTensor, this returns nullptr → crash!

**Action**: Ensure slab path is default, remove unsafe fallback

---

## Implementation Timeline

### Week 1: Foundation Fixes

**Day 1: Fix BF16Tensor** (BREAKING CHANGE)
- Remove `fp32_cache_` and `cache_valid_` members
- Make `data()` throw exception
- Expect: All operators fail with clear error

**Day 2: Fix MPILinearOperator**
- Add type checks: `std::dynamic_pointer_cast<BF16Tensor>`
- Use `bf16_data()` instead of `data()`
- Test: Linear projection parity

**Day 3: Fix MPIAttentionOperator (partial)**
- Q/K/V projections (use linear pattern)
- Defer scores/softmax (needs more thought)

### Week 2: Complete Coverage

**Day 4: Finish Attention**
- Attention scores (expand BF16 → FP32 for stability)
- Softmax (always FP32)
- Context aggregation (can be BF16)

**Day 5: Other Operators**
- MPISwiGLUOperator
- MPIRMSNormOperator  
- MPIRoPEOperator

**Day 6: ModelLoader Verification**
- Audit weight loading code
- Ensure QuantizedTensor is default
- Add logging/assertions

**Day 7: Testing & Benchmarking**
- Run full parity tests
- Memory profiling (should be ~50%)
- Performance benchmarking

---

## Success Metrics

1. **Memory Usage**:
   - Before: 5082-6498 MB (BF16+cache leak)
   - After: 2000-2500 MB (50% of FP32 baseline)
   - **Target**: 50-60% reduction

2. **Weight Dequantization**:
   - Verify QuantizedTensor used for Q4_0/Q6_K/Q8_0
   - Check QuantSlab cache hit rate >80%
   - **Target**: No full-weight dequant in memory profile

3. **Numerical Correctness**:
   - All parity tests pass: 387/387 stages
   - Relative L2 error <1e-5
   - **Target**: No regression vs current (BF16 conversion works)

4. **Performance**:
   - Throughput within 90% of FP32 baseline
   - Decode overhead <10% (BF16 conversion cost)
   - **Target**: 2.2+ tok/s (vs 2.4 tok/s FP32)

---

## Conclusion

**Current State**:
- ✅ Infrastructure exists (QuantizedTensor, QuantSlabCache, BF16Tensor)
- ✅ Slab decode path implemented in MPILinearOperator
- ❌ BF16Tensor has fatal fp32_cache leak (150% memory)
- ❌ Operators not BF16-aware (call data() blindly)

**Path Forward** (1-2 weeks):
1. Remove BF16Tensor FP32 cache → Break everything (intentionally)
2. Fix operators incrementally (BF16-aware type checking)
3. Verify ModelLoader keeps weights quantized
4. Test & benchmark (memory, numerics, performance)

**Achievable Goals**:
- ✅ Weights stay quantized (Goal 1)
- ✅ JIT decode to BF16 slabs (Goal 2)
- ✅ BF16 activations between stages (Goal 3) ← After fixes
- ✅ No full weight dequant (Goal 4)

The architecture you want **already mostly exists**, it just needs the BF16Tensor interface fixed and operators refactored to use it properly. This is a 1-2 week refactor, not a ground-up redesign.
