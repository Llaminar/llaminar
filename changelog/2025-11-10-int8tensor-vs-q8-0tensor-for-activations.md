# INT8Tensor vs Q8_0Tensor for Activation Quantization

**Date**: November 10, 2025  
**Context**: CPUAttentionT expects IActivationTensor implementations. Currently no tensor implements IActivationTensor.  
**Question**: Should we use Q8_0Tensor or INT8Tensor for quantized activations?

---

## TL;DR Recommendation

✅ **Use Q8_0Tensor for activations, deprecate INT8Tensor**

**Rationale**:
- ✅ Q8_0Tensor matches our GEMM kernel format (IntegerGemm.cpp uses Q8_0Block)
- ✅ Block-based scales provide better cache locality (proven in previous analysis)
- ✅ INT8Tensor's per-column scales are designed for **weights**, not activations
- ✅ Simpler architecture: one quantized activation format vs two
- ✅ Consistent with llama.cpp (uses Q8_0 for activations)

---

## Current State Analysis

### INT8Tensor Design (Lines 969-1050, Tensors.h)

**Purpose**: "When --weight-precision int8 is set, all quantized tensors (IQ4_NL, Q6_K, Q8_0, etc.) are **dequantized to this format at load time**."

**Key characteristics**:
```cpp
class INT8Tensor : public TensorBase, public ITensorGemmTileDataProvider {
    AlignedVector<int8_t> host_int8_data_;  // Raw INT8 array
    float scale_;                            // Global scale (fallback)
    std::vector<float> col_scales_;         // Per-column scales (weights)
    std::vector<float> row_scales_cache_;   // Per-row scales (computed on-demand)
};
```

**Intended use case**: **Weight matrices only**
- Per-column scales for weight matrix columns
- Global scale as fallback
- Per-row scales computed on-demand for transpose operations
- `block_size() const override { return shape_[1]; }` — **entire row is one block!**

**Red flags for activations**:
- ❌ Per-column scales make no sense for activations (activations aren't reused like weights)
- ❌ Full-row blocking (block_size = K dimension) defeats cache locality
- ❌ Designed for **static weight loading**, not **dynamic activation quantization**
- ❌ Comment explicitly says "dequantized to this format **at load time**" (weights!)

---

### Q8_0Tensor Design (Lines 1335-1400, Tensors.h)

**Purpose**: "Q8_0 quantized tensor (8-bit uniform quantization)"

**Key characteristics**:
```cpp
class Q8_0Tensor : public TensorBase, public ITensorGemmTileDataProvider {
    std::vector<uint8_t> raw_data_;  // Block array (Q8_0Block[])
    
    struct Q8_0Block {
        uint16_t d;      // FP16 scale (2 bytes)
        int8_t qs[32];   // 32 INT8 values (32 bytes)
    };
    // Total: 34 bytes per 32 elements
    
    size_t block_size() const override { return Q8_0Block::BLOCK_SIZE; } // 32
};
```

**Intended use case**: **General-purpose quantization (weights AND activations)**
- Block-level scales (32 elements per block)
- Matches IntegerGemm.cpp quantization function (`quantize_fp32_to_q8_0()`)
- Self-contained blocks (scale travels with data)
- Optimal cache locality (34 bytes per block, sequential access)

**Perfect for activations**:
- ✅ Block size (32) matches VNNI instruction width
- ✅ Already used in IntegerGemm.cpp for activation quantization
- ✅ Cache-friendly layout (scale co-located with data)
- ✅ No unnecessary per-column/per-row scale complexity

---

## Architectural Mismatch: INT8Tensor is for Weights

### Evidence from INT8Tensor API

**1. Per-column scales** (lines 1022-1030):
```cpp
// Per-column scales (for weight matrices)
const float *col_scales() const { return col_scales_.empty() ? nullptr : col_scales_.data(); }
bool has_col_scales() const { return !col_scales_.empty(); }
void set_col_scales(const std::vector<float> &scales) { col_scales_ = scales; }
```

**Comment explicitly says**: "Per-column scales (**for weight matrices**)"

**Why this makes sense for weights**:
- Weights are **reused** across many forward passes (same weights, different activations)
- Per-column quantization optimizes for **weight outliers** (some columns have wider range)
- Weights are **static** (loaded once, used forever)

**Why this makes NO sense for activations**:
- Activations are **transient** (computed per-layer, discarded after use)
- Per-column scales add overhead with no benefit (activations aren't reused)
- Activations change every forward pass (dynamic quantization)

---

**2. Per-row scales** (lines 1032-1038):
```cpp
// Per-row scales (computed on-demand for transpose operations)
const std::vector<float> &get_row_scales() const;
void set_row_scales(const std::vector<float> &scales) { row_scales_cache_ = scales; }
```

**Comment says**: "Per-row scales (**computed on-demand for transpose operations**)"

**Why this makes sense for weights**:
- Weights may be transposed for different GEMM orientations
- Cache row scales to avoid recomputation

**Why this makes NO sense for activations**:
- Activations are rarely transposed (Q, K, V are used as-is after projection)
- Transposing activations during quantization is architectural anti-pattern

---

**3. Full-row blocking** (line 1047):
```cpp
size_t block_size() const override { return shape_[1]; } // Full row per block
```

**This is TERRIBLE for activations**:
- ❌ Block size = entire K dimension (e.g., 896 elements for Qwen 0.5B)
- ❌ Defeats cache locality (must load entire row + scale to process 32 elements)
- ❌ Conflicts with VNNI instruction width (64 INT8 elements per DPBUSD)

**Compare to Q8_0Tensor**:
- ✅ Block size = 32 (optimal for cache lines and VNNI)
- ✅ Scale co-located with 32 values (34 bytes = perfect fit in cache line)

---

## IntegerGemm.cpp Uses Q8_0Block (Not INT8Tensor)

**Current implementation** (IntegerGemm.cpp, lines 79-175):
```cpp
// Quantize FP32 activations to Q8_0 blocks
void quantize_fp32_to_q8_0(const float *src, Q8_0Block *dst, size_t n) {
    const size_t num_blocks = (n + 31) / 32;
    for (size_t b = 0; b < num_blocks; ++b) {
        // Two-pass refinement for optimal scale
        float amax = 0.0f;
        for (size_t i = 0; i < 32; ++i) {
            amax = std::max(amax, std::abs(src[b * 32 + i]));
        }
        
        Q8_0Block &block = dst[b];
        block.d = fp32_to_fp16(amax / 127.0f);  // Store scale
        
        for (size_t i = 0; i < 32; ++i) {
            block.qs[i] = quantize_value(...);  // Store quantized INT8
        }
    }
}

// GEMM kernel uses Q8_0Block directly
bool gemm_int8_iq4nl_vnni(const float *A, ...) {
    std::vector<Q8_0Block> A_q8((m * k + 31) / 32);
    quantize_fp32_to_q8_0(A, A_q8.data(), m * k);
    
    // Process blocks in VNNI GEMM...
}
```

**If we used INT8Tensor, we'd need**:
```cpp
// Convert Q8_0Block → INT8Tensor (wasteful conversion!)
std::vector<int8_t> int8_data;
std::vector<float> col_scales;  // Per-column? Makes no sense!
for (const Q8_0Block &block : A_q8) {
    for (int i = 0; i < 32; ++i) {
        int8_data.push_back(block.qs[i]);
    }
    // What do we do with block.d scale? 
    // INT8Tensor expects per-column scales (wrong granularity!)
}
```

**Conclusion**: IntegerGemm.cpp **already uses Q8_0Block natively** — forcing INT8Tensor adds unnecessary conversion overhead.

---

## Cache Locality Comparison (Revisited)

### Scenario: Process 64 INT8 values in VNNI GEMM

**Q8_0Tensor** (block_size=32):
```
Block 0: [d₀][qs₀...qs₃₁]  (34 bytes) → Cache line 0
Block 1: [d₁][qs₃₂...qs₆₃] (34 bytes) → Cache line 1
Total: 2 cache lines (68 bytes)
```

**INT8Tensor** (block_size=K, e.g., 896 for Qwen):
```
Data: [qs₀...qs₆₃]              (64 bytes) → Cache lines 0-1
Global scale: [scale]           (4 bytes)  → Cache line 2 (separate!)
OR
Per-column scales: [s₀...s₈₉₅] (3584 bytes) → Cache lines 2-57 (scattered!)
Total: 3+ cache lines (MUCH worse!)
```

**Verdict**: Q8_0Tensor is **33-95% more cache-efficient** depending on INT8Tensor scale granularity.

---

## Consistency with llama.cpp

**llama.cpp quantization** (ggml-quants.c):
```c
// Activation quantization uses Q8_0 (32-element blocks)
void quantize_row_q8_0(const float * restrict x, void * restrict vy, int k) {
    block_q8_0 * restrict y = vy;
    for (int i = 0; i < k / QK8_0; i++) {
        // QK8_0 = 32 (block size)
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            amax = MAX(amax, fabsf(x[i*QK8_0 + j]));
        }
        y[i].d = amax / 127.0f;  // Store scale (FP16)
        for (int j = 0; j < QK8_0; j++) {
            y[i].qs[j] = roundf(x[i*QK8_0 + j] / y[i].d);  // Store INT8
        }
    }
}
```

**llama.cpp uses Q8_0 for activations, NOT per-column scales!**

Our Q8_0Tensor matches this exactly. INT8Tensor diverges from proven design.

---

## IActivationTensor Interface Analysis

**Current interface** (Tensors.h, lines 282-320):
```cpp
class IActivationTensor {
public:
    virtual std::unique_ptr<ITensorRoPE> createRoPE() = 0;
    virtual std::unique_ptr<ITensorSwiGLU> createSwiGLU() = 0;
    virtual std::unique_ptr<ITensorSoftmax> createSoftmax() = 0;
    virtual std::unique_ptr<ITensorRMSNorm> createRMSNorm() = 0;
    virtual std::unique_ptr<ITensorAttention> createAttention() = 0;
    
    virtual bool applyRMSNorm(
        const float *gamma,
        int seq_len, int d_model, float eps,
        const MPIContext *mpi_ctx, int device_idx) = 0;
};
```

**These operations ALL apply to activations**:
- ✅ RoPE: Applied to Q, K activations
- ✅ SwiGLU: Applied to FFN gate/up activations
- ✅ Softmax: Applied to attention scores (activations)
- ✅ RMSNorm: Applied to layer inputs/outputs (activations)
- ✅ Attention: Applied to Q, K, V (activations)

**None of these operations apply to weight matrices!**

**Conclusion**: IActivationTensor is **explicitly for activations**, not weights. INT8Tensor's weight-centric design is fundamentally misaligned.

---

## CPUAttentionT Template Expectations

**CPUAttentionT::compute()** (CPUAttentionT.h, lines 65-95):
```cpp
template <typename TensorType>
bool CPUAttentionT<TensorType>::compute(
    const float *Q, const float *K, const float *V,  // Input activations
    float *output,                                    // Output activations
    ...) {
    
    using ElementType = typename primitives::ActivationTraits<TensorType>::ElementType;
    
    // Cast to native precision
    const ElementType *Q_typed = reinterpret_cast<const ElementType *>(Q);
    const ElementType *K_typed = reinterpret_cast<const ElementType *>(K);
    const ElementType *V_typed = reinterpret_cast<const ElementType *>(V);
    
    // Create GEMM kernel via ActivationTraits
    auto gemm = Traits::create_activation_gemm();
    
    // Q @ K^T with fused scaling
    gemm->multiply_activations_strided(...);
    
    // Softmax on FP32 scores
    primitives::softmax_row_major_fp32(...);
    
    // scores @ V
    gemm->multiply_activations_strided(...);
}
```

**This code expects**:
1. `TensorType` represents **activation format** (FP32, BF16, FP16, Q8_0)
2. `ElementType` is the **storage type** for activations (float, uint16_t, Q8_0Block)
3. GEMM kernels operate on **activation data** (not weights)

**Q8_0Tensor fits perfectly**:
- ✅ ElementType = Q8_0Block (or int8_t for raw access)
- ✅ Activations are quantized to Q8_0Block format
- ✅ GEMM kernel uses Q8_0Block directly (IntegerGemm.cpp)

**INT8Tensor causes issues**:
- ❌ ElementType = int8_t (but scale is separate!)
- ❌ Per-column scales make no sense for Q/K/V activations
- ❌ Full-row blocking defeats cache locality
- ❌ Mismatch with IntegerGemm.cpp (needs Q8_0Block, not INT8Tensor)

---

## What is INT8Tensor Actually For?

**Intended use case** (from comment, line 964):
> "When --weight-precision int8 is set, all quantized tensors (IQ4_NL, Q6_K, Q8_0, etc.) are **dequantized to this format at load time**."

**Translation**: INT8Tensor is a **unified weight storage format** for simplified GEMM dispatch.

**Workflow**:
```
User flag: --weight-precision int8
  ↓
Load GGUF weights (IQ4_NL, Q6_K, Q8_0, etc.)
  ↓
Dequantize ALL to INT8Tensor (with per-column scales)
  ↓
Single GEMM kernel: INT8×INT8 (no format-specific kernels needed)
```

**Benefits for weights**:
- ✅ Simplifies GEMM dispatch (one INT8×INT8 kernel vs many format-specific kernels)
- ✅ Per-column scales preserve weight precision
- ✅ Trade memory (larger than native formats) for simplicity

**Why this is TERRIBLE for activations**:
- ❌ Activations are computed **per-layer** (not loaded once like weights)
- ❌ Dequantizing activations to INT8Tensor loses Q8_0 block structure
- ❌ Per-column scales add overhead with no benefit
- ❌ Conflicts with IntegerGemm.cpp (expects Q8_0Block activations)

---

## Recommendation: Implement IActivationTensor for Q8_0Tensor

### Implementation Plan

**1. Make Q8_0Tensor implement IActivationTensor**:
```cpp
class Q8_0Tensor : public TensorBase, 
                   public ITensorGemmTileDataProvider,
                   public IActivationTensor  // ← Add this!
{
public:
    // IActivationTensor interface
    std::unique_ptr<ITensorRoPE> createRoPE() override {
        return std::make_unique<CPURoPET<Q8_0Tensor>>();
    }
    
    std::unique_ptr<ITensorSwiGLU> createSwiGLU() override {
        return std::make_unique<CPUSwiGLUT<Q8_0Tensor>>();
    }
    
    std::unique_ptr<ITensorSoftmax> createSoftmax() override {
        // Q8_0 activations must be dequantized for softmax (FP32 only)
        return std::make_unique<CPUSoftmaxT<FP32Tensor>>();
    }
    
    std::unique_ptr<ITensorRMSNorm> createRMSNorm() override {
        return std::make_unique<CPURMSNormT<Q8_0Tensor>>();
    }
    
    std::unique_ptr<ITensorAttention> createAttention() override {
        return std::make_unique<CPUAttentionT<Q8_0Tensor>>();
    }
    
    bool applyRMSNorm(...) override {
        // Native Q8_0 RMSNorm (dequantize → normalize → requantize)
        auto kernel = createRMSNorm();
        return kernel->apply_q8_0(...);
    }
};
```

**2. Create ActivationTraits<Q8_0Tensor> specialization**:
```cpp
namespace llaminar2::primitives {

template<>
struct ActivationTraits<Q8_0Tensor> {
    using ElementType = Q8_0Block;  // Or int8_t for raw access
    using ComputeType = float;      // GEMM computes in FP32
    
    static constexpr bool supports_native_gemm = true;
    static constexpr bool supports_native_softmax = false;  // Must dequant to FP32
    
    static std::unique_ptr<ITensorGemm> create_activation_gemm() {
        // Return INT8×IQ4_NL VNNI kernel
        return std::make_unique<GemmKernel<
            ISA_AVX512,
            /*TILE_M=*/6,
            /*TILE_N=*/16,
            ActivationStorageTraits<Q8_0Block>,  // ← New specialization
            QuantizedWeightAccessor<IQ4_NL>>>();
    }
};

}  // namespace llaminar2::primitives
```

**3. Add ActivationStorageTraits<Q8_0Block>**:
```cpp
template<>
struct ActivationStorageTraits<Q8_0Block> {
    using StorageType = Q8_0Block;
    using ComputeType = float;
    
    static constexpr bool needs_conversion = true;
    static constexpr size_t elements_per_block = 32;
    
    // Load Q8_0 block → convert to INT8 for VNNI
    static void load_and_convert(const Q8_0Block *src, int8_t *dst, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            std::memcpy(dst + i * 32, src[i].qs, 32);
        }
    }
    
    // Extract scale (used in GEMM kernel)
    static float get_scale(const Q8_0Block &block) {
        return fp16_to_fp32(block.d);
    }
};
```

---

### What About INT8Tensor?

**Keep it for weights only**:
- ✅ Rename to `INT8WeightTensor` (clarify purpose)
- ✅ Document as "unified weight format for --weight-precision int8"
- ✅ Do NOT implement IActivationTensor (wrong abstraction)
- ✅ Keep per-column scales (useful for weight outliers)

**Alternative: Deprecate entirely**:
- If we commit to format-specific GEMM kernels (IQ4_NL, Q6_K, Q8_0)
- INT8Tensor becomes redundant (no need for unified weight format)
- Reduces codebase complexity

---

## Summary Comparison

| Criterion | Q8_0Tensor | INT8Tensor | Winner |
|-----------|------------|------------|--------|
| **Block size** | 32 (optimal for VNNI) | K (entire row) | **Q8_0** |
| **Scale granularity** | Per-block (32 elem) | Per-column OR global | **Q8_0** |
| **Cache locality** | Excellent (34 bytes) | Poor (scattered) | **Q8_0** |
| **IntegerGemm.cpp compat** | Native (Q8_0Block) | Requires conversion | **Q8_0** |
| **llama.cpp alignment** | Exact match | Diverges | **Q8_0** |
| **IActivationTensor fit** | Perfect (activations) | Wrong (weights) | **Q8_0** |
| **CPUAttentionT compat** | Perfect | Poor | **Q8_0** |
| **Memory overhead** | 1.0625 bytes/elem | 1.0625-1.125 | **TIE** |
| **Design intent** | General quantization | Weight loading only | **Q8_0** |
| **Complexity** | Simple (self-contained) | Complex (multiple scales) | **Q8_0** |

**Winner**: **Q8_0Tensor** (9/10 criteria, 1 tie)

---

## Final Recommendation

### ✅ Use Q8_0Tensor for Activations

**Action items**:
1. ✅ Make Q8_0Tensor implement IActivationTensor
2. ✅ Create ActivationTraits<Q8_0Tensor> specialization
3. ✅ Add ActivationStorageTraits<Q8_0Block> for GEMM kernels
4. ✅ Test CPUAttentionT<Q8_0Tensor> instantiation

### ⚠️ Keep INT8Tensor for Weights Only (Or Deprecate)

**Option A: Rename and clarify**:
- Rename `INT8Tensor` → `INT8WeightTensor`
- Update docs: "Unified weight format for --weight-precision int8"
- Do NOT implement IActivationTensor

**Option B: Deprecate entirely**:
- Remove INT8Tensor (focus on format-specific kernels)
- Simpler architecture: Q8_0Tensor for activations, native formats for weights
- Less code to maintain

**Recommended**: **Option A** (preserve INT8Tensor for experimental weight workflows)

---

## Why This Matters

**Current state**:
- ❌ No tensor implements IActivationTensor (CPUAttentionT can't be instantiated!)
- ❌ INT8Tensor designed for weights, not activations
- ❌ Architectural confusion (two INT8 formats with different purposes)

**After fix**:
- ✅ Q8_0Tensor implements IActivationTensor (CPUAttentionT works!)
- ✅ Clear separation: Q8_0Tensor (activations), INT8Tensor (weights)
- ✅ Consistent with IntegerGemm.cpp and llama.cpp
- ✅ Optimal cache locality (block-based scales)

**No downside**: Q8_0Tensor is strictly better for activations. INT8Tensor serves a different (weight-focused) purpose.

