# Llaminar Architecture: Quantized Weights + BF16 Activations
**Date**: October 20, 2025  
**Author**: David Sanftenberg  
**Status**: 🔴 **DESIGN DOCUMENT** - Refactoring Required

## Table of Contents
1. [Architectural Goals](#architectural-goals)
2. [Current State Analysis](#current-state-analysis)
3. [Key Problems Discovered](#key-problems-discovered)
4. [Proposed Architecture](#proposed-architecture)
5. [Implementation Plan](#implementation-plan)
6. [Success Criteria](#success-criteria)

---

## Architectural Goals

### Goal 1: Keep Weights in Native Quantized Format
**Principle**: Weights stay as Q4_0, Q6_K, Q8_0, etc. throughout their lifetime.

**Why**: 
- 4-16× smaller memory footprint (Q4_0 uses 4 bits/value vs 32 bits for FP32)
- Faster model loading (no dequantization at load time)
- Enables loading larger models that wouldn't fit if fully dequantized

**Example**: Qwen 2.5 0.5B Q8_0 model
- Quantized: 638 MB on disk
- If fully dequantized to FP32: ~2.0 GB in memory
- **Savings**: 3.1× memory reduction

### Goal 2: Just-In-Time Dequantization with Reusable Buffers
**Principle**: Dequantize only what's needed, when needed, into temporary BF16 buffers.

**Why**:
- Decode cost amortized via LRU cache
- BF16 decode target (2 bytes) gives bandwidth benefits vs FP32 (4 bytes)
- Reusable buffers avoid repeated allocation overhead

**Implementation**: `QuantSlabCache`
- 64 MB default capacity (configurable)
- LRU eviction policy
- Keyed by (weight_tensor, column_start, column_count)
- Decodes to BF16 for compute efficiency

### Goal 3: Pass BF16 Between Pipeline Stages
**Principle**: Activations (Q/K/V projections, FFN outputs, residuals) stored as BF16.

**Why**:
- 50% memory vs FP32 (2 bytes vs 4 bytes)
- Same exponent range as FP32 (8 bits) → no overflow/underflow risk
- Hardware acceleration on Ice Lake+, Zen 4+
- Numerically sufficient for neural network activations

**Critical Operations Still in FP32**:
- Softmax (numerical stability)
- RMSNorm denominators (precision-sensitive)
- Final logits (output quality)

### Goal 4: Never Fully Dequantize Weights
**Principle**: No persistent FP32/F16 copies of quantized weight tensors.

**Why**:
- Memory explosion (4-16× weight size)
- Defeats purpose of quantization
- Unnecessary bandwidth waste

**Verification**: Memory profiling should show only:
- Quantized weight storage (raw bytes)
- QuantSlab cache (64 MB BF16 working set)
- BF16 activation tensors
- Temporary FP32 buffers for softmax/norm

---

## Current State Analysis

### ✅ What Works

#### 1. QuantizedTensor Infrastructure
**File**: `src/tensors/TensorFactory.h`, `src/tensors/TensorFactory.cpp`

```cpp
class QuantizedTensor : public TensorBase {
    QuantStorageLayout layout_;  // Format metadata (Q4_0, Q6_K, etc.)
    std::vector<uint8_t> raw_;   // Raw quantized bytes
    
    void decodeBlock(size_t block_index, float* dst) const;
    
    // Critical: data() returns nullptr (no FP32 backing buffer)
    float* data() override { return nullptr; }
};
```

**Status**: ✅ Implemented, stores weights efficiently

#### 2. QuantSlabCache System
**Files**: `src/operators/QuantSlabCache.{h,cpp}`

```cpp
struct QuantSlab {
    size_t k, n;                // Dimensions
    std::vector<bfloat16> data; // Decoded weight slice in BF16
};

class QuantSlabCache {
    // LRU cache: 64 MB default
    std::shared_ptr<QuantSlab> getOrDecode(
        const QuantizedTensor& tensor,
        size_t col_start, size_t col_count
    );
};
```

**Features**:
- Decodes quantized weight columns on demand
- Caches in BF16 format (2 bytes/value)
- LRU eviction when capacity exceeded
- Thread-safe with mutex protection

**Status**: ✅ Implemented in `MPILinearOperator`

#### 3. BF16 GEMM Support
**Files**: `src/AdaptiveMatmul.{h,cpp}`, `src/backends/MKLBackend.{h,cpp}`

```cpp
bool adaptiveMatMulBF16(
    const float* A,           // Input (FP32)
    const bfloat16* B,        // Weight (BF16 from slab)
    float* C,                 // Output (FP32 accumulation)
    int m, int n, int k
);
```

**Backends**:
- **Intel MKL**: `cblas_sbgemm` (hardware accelerated on Ice Lake+)
- **OpenBLAS**: Software emulation (v0.3.26+, verified working)
- **Fallback**: Expand BF16→FP32, then standard SGEMM

**Status**: ✅ Fully functional with all backends

### ❌ What's Broken

#### 1. BF16Tensor Memory Leak (CRITICAL)
**File**: `src/tensors/BF16Tensor.h` lines 92-115

**Problem**: Persistent FP32 cache allocated on ANY call to `data()`

```cpp
class BF16Tensor : public TensorBase {
private:
    std::vector<bfloat16> data_;           // BF16 storage (2 bytes/element)
    mutable std::vector<float> fp32_cache_; // FP32 cache (4 bytes/element) ← LEAK!
    mutable bool cache_valid_ = false;
    
    void update_cache() const {
        if (cache_valid_) return;  // Cache persists forever!
        
        fp32_cache_.resize(data_.size()); // Allocate full FP32 copy
        // Convert BF16 → FP32 ...
        cache_valid_ = true;
    }
    
public:
    float* data() override {
        update_cache();  // Triggers on EVERY data() call
        return fp32_cache_.data();
    }
};
```

**Consequence**: 
- BF16 data: 2 bytes/element (50% of FP32)
- FP32 cache: 4 bytes/element (100% of FP32)
- **Total**: 150% of FP32 baseline memory! (26-54% INCREASE instead of 50% reduction)

**Root Cause**: Interface incompatibility
- `TensorBase` assumes `float* data()` interface
- `BF16Tensor` must provide FP32 pointer for compatibility
- Cache never deallocated (persists until tensor destroyed)

**Measurement**: 
- FP32 baseline: 4210-5171 MB
- BF16 (broken): 5082-6498 MB
- **Overhead**: +872 to +2288 MB

#### 2. Operator Call Pattern
**Files**: `src/operators/MPILinearOperator.cpp`, `src/operators/MPIAttentionOperator.cpp`, etc.

**Problem**: Operators blindly call `data()` which triggers FP32 cache

```cpp
// Current (broken):
float* input_data = input_tensor->data();  // If BF16Tensor, allocates cache!
float* weight_data = weight_tensor->data();

// Weight might be QuantizedTensor (data() returns nullptr) → crash
// Input might be BF16Tensor (data() allocates FP32 cache) → memory leak
```

**What Should Happen**:
```cpp
// Check tensor type first:
if (auto bf16_input = std::dynamic_pointer_cast<BF16Tensor>(input_tensor)) {
    // Use BF16 data directly
    bfloat16* bf16_data = bf16_input->bf16_data();
} else {
    // Legacy FP32 path
    float* fp32_data = input_tensor->data();
}

if (auto quant_weight = std::dynamic_pointer_cast<QuantizedTensor>(weight)) {
    // Decode to QuantSlab (cached BF16)
    QuantSlab slab;
    QuantSlabCache::instance().getOrDecode(*quant_weight, ...);
} else {
    // FP32 weight
    float* weight_data = weight->data();
}
```

#### 3. ModelLoader Dual Code Paths
**File**: `src/ModelLoader.cpp` (lines vary, need inspection)

**Problem**: Unclear which path is active
- Old path: Full dequantization at load time (line 51 comments mention this)
- New path: Create QuantizedTensor with raw bytes

**Need to verify**:
- Is QuantizedTensor the default for Q4_0, Q6_K, Q8_0?
- Or is old full-dequant path still active?
- Are we accidentally dequantizing everything?

---

## Key Problems Discovered

### Problem 1: TensorBase Interface Mismatch

**Fundamental Issue**: C++ interface inheritance forces FP32 compatibility

```cpp
// Base class contract:
class TensorBase {
    virtual float* data() = 0;  // ALL derived classes must return float*
};

// Derived class dilemma:
class BF16Tensor : public TensorBase {
    // Option A: Return nullptr → breaks operators expecting valid pointer
    float* data() override { return nullptr; }
    
    // Option B: Allocate FP32 cache → memory leak (current broken state)
    float* data() override { update_cache(); return fp32_cache_.data(); }
    
    // Option C: Throw error → runtime failures
    float* data() override { throw std::runtime_error("Use bf16_data()"); }
};
```

**Solutions**:
1. **Remove FP32 cache entirely** (Option C) → Forces operator fixes
2. **Add type-aware checks in ALL operators** → Verbose but safe
3. **Redesign TensorBase interface** → Major refactor

**Recommendation**: Option 1 (remove cache, throw error)
- Fail fast at development time (not production)
- Forces explicit BF16 handling
- Clear error messages guide fixes

### Problem 2: Hidden Memory Allocations

**Issue**: No visibility into when FP32 cache is allocated

```cpp
// Looks innocent:
auto output = operator_linear->forward(bf16_input, weight);

// But internally:
float* input_data = bf16_input->data();  // ← Silent 1 GB allocation!
```

**Need**:
- Logging/assertions when cache allocated
- Memory profiling hooks
- Compile-time prevention (make data() inaccessible for BF16)

### Problem 3: Parity Tests Didn't Catch This

**Why tests passed**:
- Tested numerical correctness only (BF16 ↔ FP32 conversions work)
- Never measured memory usage
- Small test tensors didn't show overhead

**Lesson**: Always test what you optimize
- Memory optimization → must measure memory
- Performance optimization → must benchmark throughput
- Numerical optimization → test accuracy

---

## Proposed Architecture

### Data Flow Diagram

```
WEIGHTS (Quantized Storage)
┌─────────────────────────────┐
│   QuantizedTensor           │
│   Format: Q4_0, Q6_K, Q8_0  │
│   Storage: raw uint8_t[]    │
│   Size: 4-16× compressed    │
└──────────┬──────────────────┘
           │
           │ On-demand decode (JIT)
           ↓
┌─────────────────────────────┐
│   QuantSlabCache            │
│   Format: BF16              │
│   Capacity: 64 MB (LRU)     │
│   Lifetime: Reused          │
└──────────┬──────────────────┘
           │
           │ adaptiveMatMulBF16()
           ↓
ACTIVATIONS (Persistent BF16)
┌─────────────────────────────┐
│   BF16Tensor                │
│   Format: bfloat16          │
│   Storage: 2 bytes/element  │
│   NO FP32 cache!            │
└──────────┬──────────────────┘
           │
           │ Expand for compute
           ↓
COMPUTE (Temporary FP32)
┌─────────────────────────────┐
│   FP32 Accumulators         │
│   - Matmul accumulation     │
│   - Softmax computation     │
│   - RMSNorm denominators    │
│   Lifetime: Function scope  │
└──────────┬──────────────────┘
           │
           │ Convert & store
           ↓
ACTIVATIONS (Next Stage)
┌─────────────────────────────┐
│   BF16Tensor                │
│   Ready for next operator   │
└─────────────────────────────┘
```

### Tensor Type System

```cpp
// Weight tensors (loaded once, never modified):
QuantizedTensor  → Q4_0, Q6_K, Q8_0 (4-16× compressed)
SimpleTensor     → F32, F16 (embedding table, norms)

// Activation tensors (intermediate computations):
BF16Tensor       → Q/K/V projections, FFN outputs, residuals
SimpleTensor     → Softmax scores, RMSNorm temp buffers (FP32 only)

// Temporary buffers (function-scoped):
std::vector<float>  → FP32 accumulation, attention scores
QuantSlab           → Decoded weight slice (BF16, cached)
```

### Operator Pattern (Correct Implementation)

```cpp
class MPILinearOperator {
    std::shared_ptr<TensorBase> forward(
        std::shared_ptr<TensorBase> input,
        std::shared_ptr<TensorBase> weight
    ) {
        // Step 1: Determine input type
        auto bf16_input = std::dynamic_pointer_cast<BF16Tensor>(input);
        auto fp32_input = bf16_input ? nullptr : input;
        
        // Step 2: Determine weight type
        auto quant_weight = std::dynamic_pointer_cast<QuantizedTensor>(weight);
        
        // Step 3: Execute appropriate path
        if (quant_weight) {
            // Decode weight to BF16 slab (cached)
            QuantSlab slab;
            bool reused = QuantSlabCache::instance().getOrDecode(
                *quant_weight, col_start, col_count, slab, /*reuse=*/true
            );
            
            // Create BF16 output tensor
            auto bf16_output = std::make_shared<BF16Tensor>(output_shape);
            
            if (bf16_input) {
                // BF16 × BF16 → BF16 (optimal path)
                adaptiveMatMulBF16(
                    bf16_input->bf16_data(),  // Input: BF16
                    slab.data.data(),         // Weight: BF16
                    bf16_output->bf16_data(), // Output: BF16
                    m, n, k
                );
            } else {
                // FP32 × BF16 → BF16 (mixed precision)
                adaptiveMatMulBF16(
                    fp32_input->data(),       // Input: FP32
                    slab.data.data(),         // Weight: BF16
                    bf16_output->bf16_data(), // Output: BF16
                    m, n, k
                );
            }
            
            return bf16_output;
        } else {
            // FP32/F16 weights (unquantized path)
            auto fp32_output = std::make_shared<SimpleTensor>(output_shape);
            
            adaptiveMatMul(
                fp32_input->data(),
                weight->data(),
                fp32_output->data(),
                m, n, k
            );
            
            return fp32_output;
        }
    }
};
```

### Memory Budget Breakdown (Target)

**Example**: Qwen 2.5 0.5B model, sequence length 512

| Component | Format | Size (MB) | Notes |
|-----------|--------|-----------|-------|
| **Weights (Quantized)** | Q8_0 | 638 | Stays compressed |
| **QuantSlab Cache** | BF16 | 64 | LRU working set |
| **Activations (per token)** | BF16 | ~4 | Q/K/V + FFN outputs |
| **KV Cache (512 tokens)** | BF16 | 48 | 2× layers × d_model × seq_len |
| **Temporary Buffers** | FP32 | ~20 | Softmax, norms, accumulation |
| **TOTAL** | — | **~770 MB** | vs 2.0 GB if all FP32 |

**Savings**: 2.6× reduction vs FP32 baseline

---

## Implementation Plan

### Phase 1: Fix BF16Tensor (Day 1) 🔴 CRITICAL

**Goal**: Remove FP32 cache, force explicit BF16 handling

**Changes**:
```cpp
// src/tensors/BF16Tensor.h
class BF16Tensor : public TensorBase {
    // REMOVE these members:
    // mutable std::vector<float> fp32_cache_;
    // mutable bool cache_valid_;
    // void update_cache() const;
    
    float* data() override {
        throw std::runtime_error(
            "BF16Tensor::data() not supported. Use bf16_data() for "
            "BF16-aware operations or cast to SimpleTensor for FP32."
        );
    }
    
    // Keep BF16 interface:
    bfloat16* bf16_data() { return data_.data(); }
    const bfloat16* bf16_data() const { return data_.data(); }
};
```

**Impact**:
- ✅ Removes 150% memory overhead
- ❌ Breaks all operators calling data() on BF16 tensors
- ✅ Clear error messages guide fixes

**Testing**:
```bash
# Expect failures:
ctest --test-dir build --output-on-failure -R "Parity|Integration"

# Fix operators incrementally, retest
```

### Phase 2: Refactor MPILinearOperator (Days 2-3)

**Goal**: Make linear projections BF16-aware

**Files**: `src/operators/MPILinearOperator.cpp`

**Pattern**:
```cpp
// Before: Blind data() calls
float* input_data = input->data();
float* weight_data = global_weight->data();

// After: Type-aware dispatch
if (auto quant = std::dynamic_pointer_cast<QuantizedTensor>(global_weight)) {
    // Quantized weight path (already partially implemented!)
    QuantSlab slab;
    QuantSlabCache::instance().getOrDecode(*quant, col_start, col_count, slab, true);
    
    if (auto bf16_in = std::dynamic_pointer_cast<BF16Tensor>(input)) {
        // BF16 input × BF16 weight → BF16 output
        auto bf16_out = std::make_shared<BF16Tensor>(output_shape);
        adaptiveMatMulBF16(
            bf16_in->bf16_data(), slab.data.data(), bf16_out->bf16_data(), m, n, k
        );
        return bf16_out;
    } else {
        // FP32 input × BF16 weight → FP32 output (or convert to BF16)
        // ... handle mixed precision
    }
} else {
    // FP32 weight path
    // ... standard SGEMM
}
```

**Testing**:
```bash
# Test each layer type:
ctest -R "MPILinear"
ctest -R "ParityFramework.*Linear"
```

### Phase 3: Refactor Attention Operator (Days 4-5)

**Goal**: Make attention BF16-aware

**Files**: `src/operators/MPIAttentionOperator.cpp`

**Challenges**:
- Q/K/V projections: Use linear pattern above
- Attention scores: Must be FP32 for softmax stability
- Context aggregation: Can be BF16
- Output projection: Use linear pattern

**Pattern**:
```cpp
// Q/K/V projections → BF16
auto q_bf16 = linear_q->forward(input_bf16, weight_q);  // Returns BF16Tensor
auto k_bf16 = linear_k->forward(input_bf16, weight_k);
auto v_bf16 = linear_v->forward(input_bf16, weight_v);

// Expand Q/K to FP32 for score computation
std::vector<float> q_fp32 = expand_to_fp32(q_bf16);
std::vector<float> k_fp32 = expand_to_fp32(k_bf16);

// Attention scores: Q @ K^T (FP32 for numerical stability)
std::vector<float> scores_fp32 = compute_attention_scores(q_fp32, k_fp32);

// Softmax: MUST be FP32
std::vector<float> weights_fp32 = softmax(scores_fp32);

// Context: weights @ V (can use BF16 V)
auto context_bf16 = aggregate_context(weights_fp32, v_bf16);

// Output projection → BF16
auto output_bf16 = linear_o->forward(context_bf16, weight_o);
```

**Key Decision**: When to expand BF16 → FP32?
- **Expand early** (at Q/K/V): Simpler, more FP32 compute
- **Expand late** (only for scores/softmax): More complex, less memory

**Recommendation**: Expand only for scores/softmax (minimize FP32 footprint)

### Phase 4: Refactor Remaining Operators (Days 6-7)

**Operators**:
- `MPISwiGLUOperator`: Gate/up projections → BF16, element-wise ops → FP32
- `MPIRMSNormOperator`: Input → BF16, denominator → FP32, output → BF16
- `MPIRoPEOperator`: Rotary embeddings (position-dependent, can be BF16)
- `MPIEmbeddingOperator`: Lookup table (usually F16/F32, keep as-is)

**Pattern** (RMSNorm example):
```cpp
// Input: BF16
auto bf16_input = std::dynamic_pointer_cast<BF16Tensor>(input);

// Expand to FP32 for RMS computation (precision-sensitive)
std::vector<float> fp32_data = expand_to_fp32(bf16_input);

// Compute RMS denominator in FP32
float rms = compute_rms(fp32_data);

// Apply normalization (can convert back to BF16)
auto bf16_output = std::make_shared<BF16Tensor>(input_shape);
apply_rmsnorm(bf16_input->bf16_data(), rms, gamma, bf16_output->bf16_data());
```

### Phase 5: ModelLoader Verification (Day 8)

**Goal**: Ensure quantized weights never fully dequantized

**Check**:
```cpp
// src/ModelLoader.cpp - Verify this pattern is active:
if (is_quantized_format(tensor_type)) {
    // Create QuantizedTensor with raw bytes
    auto quant_tensor = std::make_shared<QuantizedTensor>(layout, raw_bytes);
    weights[name] = quant_tensor;
} else {
    // F32/F16: Create SimpleTensor
    auto fp32_tensor = std::make_shared<SimpleTensor>(shape, data);
    weights[name] = fp32_tensor;
}
```

**Verify NO full dequant**:
- Search for "dequantize to FP32" patterns
- Check if old code path still active
- Add assertions that weight tensors are QuantizedTensor

**Testing**:
```bash
# Log all weight loads:
LLAMINAR_DEQUANT_STATS=1 ./build_release/llaminar ...

# Check logs: Should see "QuantizedTensor" for Q4_0/Q6_K/Q8_0
# Should NOT see "dequantizing X MB to FP32"
```

### Phase 6: Integration Testing (Days 9-10)

**Test Suite**:
```bash
# 1. Parity tests (numerics unchanged)
ctest -R "ParityFramework"

# 2. Memory usage tests (NEW!)
# Should show ~50% reduction vs FP32
./test_memory_usage.sh

# 3. Performance benchmarks
./run_batch_performance.sh

# 4. Cache efficiency
LLAMINAR_QUANT_SLAB_STATS=1 ./build_release/llaminar ...
# Check hit rate >80%
```

**Memory Profiling**:
```bash
# Before: BF16 broken (150% memory)
valgrind --tool=massif ./build_release/llaminar ...
# Expect: 5000-6000 MB

# After: BF16 fixed (50% memory)
# Expect: 2500-3000 MB
```

---

## Success Criteria

### 1. Memory Usage ✅ Target: 50% vs FP32
```bash
# Measure resident memory (VmRSS)
FP32_baseline=$(grep VmRSS /proc/$(pidof llaminar)/status)
BF16_optimized=$(grep VmRSS /proc/$(pidof llaminar)/status)

# Should show ~50% reduction
echo "Baseline: $FP32_baseline"
echo "Optimized: $BF16_optimized"
echo "Reduction: $(( (FP32 - BF16) * 100 / FP32 ))%"
```

**Expected**:
- FP32: 4000-5000 MB
- BF16: 2000-2500 MB
- **Reduction**: 50%

### 2. Quantized Weight Verification ✅ No Full Dequant
```bash
# Check weight tensor types:
LLAMINAR_DEQUANT_STATS=1 ./build_release/llaminar ... 2>&1 | grep "QuantizedTensor"

# Should see:
# [INFO] Loading blk.0.attn_q.weight as QuantizedTensor (Q6_K)
# [INFO] Loading blk.0.attn_k.weight as QuantizedTensor (Q6_K)
# ...

# Should NOT see:
# [WARN] Dequantizing blk.0.attn_q.weight: 3.2 MB → 12.8 MB FP32
```

### 3. QuantSlab Cache Efficiency ✅ Target: >80% Hit Rate
```bash
# Enable slab statistics:
LLAMINAR_QUANT_SLAB_ENABLE=1 \
LLAMINAR_QUANT_SLAB_STATS=1 \
./build_release/llaminar ... 2>&1 | grep "QuantSlab"

# Should see:
# [INFO] QuantSlab cache: 1024 requests, 876 hits (85.6%)
# [INFO] QuantSlab decode time: 12.3 ms total (0.012 ms/decode)
```

**Rationale**: 
- 24 layers, ~10 weight tensors/layer = 240 unique weight slabs
- With 64 MB cache, should fit ~8-12 layers
- Hit rate >80% indicates effective caching

### 4. Numerical Parity ✅ Target: <1e-5 Relative Error
```bash
# All parity tests must pass:
ctest -R "ParityFramework" --output-on-failure

# Expected results:
# ParityFramework.OpenBLASPrefillVsPyTorch ............ Passed (42.3s)
# ParityFramework.COSMAPrefillVsPyTorch ............... Passed (38.1s)
# ParityFramework.TrueIncrementalDecodeVsPyTorch ...... Passed (28.7s)
#
# All stages: 387/387 passing
# Max rel L2 error: 4.8e-6 (well below 1e-5 threshold)
```

### 5. Performance ✅ Target: >90% of FP32 Throughput
```bash
# Benchmark throughput:
./run_batch_performance.sh

# Expected results:
# FP32 Baseline: 2.5 tok/s (prefill), 2.4 tok/s (decode)
# BF16 Optimized: 2.3 tok/s (prefill), 2.2 tok/s (decode)
#
# Throughput retention: 92% prefill, 92% decode
# Decode overhead from BF16 conversion: ~8% (acceptable)
```

**Acceptable Performance Tradeoffs**:
- Prefill: 85-95% of FP32 (decode overhead amortized over large matmuls)
- Decode: 90-95% of FP32 (BF16 conversion cost)
- If <85%: Profile decode hotspots, may need larger slab cache

---

## Risk Mitigation

### Risk 1: Parity Tests Fail After Refactor
**Probability**: Medium  
**Impact**: High (blocks deployment)

**Mitigation**:
- Incremental changes: Fix one operator at a time
- Test after each operator refactor
- Use snapshot comparison to identify divergence points
- Keep FP32 fallback path for debugging

**Contingency**: Revert to FP32 path if >1e-4 divergence

### Risk 2: Performance Regression
**Probability**: Low-Medium  
**Impact**: Medium (slower but correct)

**Mitigation**:
- Profile decode cost per operator
- Measure QuantSlab cache hit rates
- May need to increase cache size (64 MB → 128 MB)
- Optimize hot decode paths (Q6_K, Q8_0)

**Contingency**: Accept 10-15% slowdown if memory savings achieved

### Risk 3: Hidden Memory Allocations Persist
**Probability**: Low  
**Impact**: High (defeats purpose)

**Mitigation**:
- Memory profiling with valgrind/massif
- Add assertions: `assert(fp32_cache_.empty())`
- Log tensor allocations >10 MB
- Track total BF16Tensor size vs FP32 baseline

**Contingency**: Add memory regression tests to CI

### Risk 4: MPI Deadlocks During Refactor
**Probability**: Low  
**Impact**: Critical (blocks testing)

**Mitigation**:
- Test single-rank first (no MPI collectives)
- Then test multi-rank with small models
- Use timeouts in tests (60s)
- Log before/after every MPI call during debug

**Contingency**: Use GDB with MPI to isolate hanging rank

### Risk 5: Incomplete Operator Coverage
**Probability**: Medium  
**Impact**: Medium (some paths broken)

**Mitigation**:
- Audit all operators: grep for `->data()`
- Create test matrix (FP32, BF16, Quantized × each operator)
- Run full integration tests (all 24 layers)

**Contingency**: Disable unsupported operators until fixed

---

## Appendix: Reference Implementations

### llama.cpp Pattern (JIT Decode)

**File**: `ggml/src/ggml.c` (hypothetical, not in our repo)

```c
// llama.cpp approach: Decode on the fly, no caching
void ggml_compute_forward_mul_mat_q4_0(
    struct ggml_tensor * dst,
    struct ggml_tensor * src0,  // Quantized weight
    struct ggml_tensor * src1   // FP32 input
) {
    // Each thread decodes its tile:
    for (int i = thread_id; i < num_blocks; i += num_threads) {
        float decoded[32];
        dequantize_row_q4_0(src0->data + i * block_size, decoded, 32);
        
        // Compute dot product with decoded tile
        for (int j = 0; j < 32; j++) {
            acc += decoded[j] * src1->data[j];
        }
    }
}
```

**Tradeoffs**:
- ✅ Simpler: No cache management
- ❌ Repeated decode cost: Every forward pass re-decodes
- ❌ Poor for MPI: Each rank re-decodes same weights

### Llaminar Approach (Cached Decode)

**File**: `src/operators/QuantSlabCache.cpp`

```cpp
// Our approach: Decode once, cache, reuse
bool QuantSlabCache::getOrDecode(
    const QuantizedTensor& tensor,
    size_t col_start, size_t col_count,
    QuantSlab& out_slab, bool reuse_allowed
) {
    QuantSlabKey key{tensor.raw(), col_start, col_count};
    
    // Check cache first
    if (reuse_allowed && map_.contains(key)) {
        out_slab = *map_[key].slab;
        touch(map_[key].lru_iter);
        return true;  // Cache hit!
    }
    
    // Cache miss: Decode to BF16
    out_slab.k = tensor.shape()[0];
    out_slab.n = col_count;
    out_slab.data.resize(out_slab.k * out_slab.n);
    
    // Decode quantized blocks → BF16
    for (size_t block = 0; block < num_blocks; block++) {
        tensor.decodeBlockBF16(block, out_slab.data.data() + offset);
    }
    
    // Cache for reuse
    map_[key] = {std::make_shared<QuantSlab>(out_slab), ...};
    return false;  // Decode happened
}
```

**Tradeoffs**:
- ✅ Amortized decode cost: 80%+ cache hit rate
- ✅ MPI-friendly: Ranks share decode work
- ✅ BF16 storage: 2× bandwidth vs FP32 decoded cache
- ❌ More complex: LRU eviction, thread safety
- ❌ Memory overhead: 64 MB cache capacity

**Why Better for Llaminar**:
- Distributed inference: Weights accessed repeatedly across ranks
- Autoregressive decode: Same weights used for every token
- Cache hit rate >>80% means decode cost amortized

---

## Conclusion

The current Phase 5 BF16 implementation is **fundamentally broken** due to the `fp32_cache_` memory leak in `BF16Tensor`. However, the infrastructure for the correct architecture already exists:

**Existing ✅**:
- QuantizedTensor (stores weights in native format)
- QuantSlabCache (JIT decode to BF16, 64 MB LRU cache)
- BF16 GEMM support (MKL + OpenBLAS backends)
- Parity test framework (validates numerical correctness)

**Needs Fixing ❌**:
- BF16Tensor interface (remove FP32 cache, force explicit BF16 handling)
- Operator call patterns (type-aware dispatch instead of blind data() calls)
- ModelLoader verification (ensure no hidden full dequant)

**Timeline**: 1-2 weeks of focused work
- Week 1: Fix BF16Tensor, refactor Linear/Attention operators
- Week 2: Complete operator coverage, integration testing, benchmarking

**Expected Outcome**:
- ✅ 50% memory reduction (2.0-2.5 GB vs 4.0-5.0 GB)
- ✅ Weights stay quantized (no full dequant)
- ✅ <10% performance overhead (vs FP32 baseline)
- ✅ All parity tests pass (<1e-5 error)

This refactor will achieve the original architectural goals and deliver a production-ready quantized inference system.
