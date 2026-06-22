# Handover: HybridQ16 K_ROPE Precision Fix

**Date**: January 2, 2026  
**Branch**: `feature/typed-residuals`  
**Status**: Root cause identified, solution requires new kernel capability

---

## Problem Summary

The HybridQ16 pipeline has a **K_ROPE precision problem** causing ~0.878 cosine similarity vs FP32 reference (should be >0.99).

### Root Cause

**K_PROJECTION has large dynamic range (~130 max)** which causes Q8_1 quantization to lose small values:

```
Q8_1 quantization step = max_abs / 127 ≈ 130 / 127 ≈ 1.02
Values with |x| < 0.51 round to 0
```

**Evidence from debugging**:
```
K_PROJECTION FP32 values: -8.47655, -3.80529, -6.33802, 0.657984, -0.0529036, ...
K_PROJECTION Q8_1 qs:     -9, -3, -6, 1, 0, ...  ← Element 4 rounds to 0!
```

When both elements of a RoPE rotation pair are 0, the rotation produces 0, causing divergence.

### Why Q16_1 Would Fix This

Q16_1 has 256× finer precision:
```
Q16_1 quantization step = max_abs / 16383 ≈ 130 / 16383 ≈ 0.008
Values with |x| < 0.004 round to 0 (vs 0.51 for Q8_1)
```

Element 4 value `-0.0529` would be preserved in Q16_1 (qs ≈ -7), not lost.

---

## Current Pipeline Flow (HybridQ16)

```
Input (Q16_1) → RMSNorm → FP32 → QKV_GEMM → Q8_1 (Q,K,V)
                                              ↓
                                         RoPE (Q8_1→Q16_1)
                                              ↓
                                         KV Cache (Q16_1)
                                              ↓
                                         Attention (Q16 integer)
```

**The problem**: K goes through Q8_1 intermediate, losing precision before RoPE.

---

## Proposed Solution (Integer Domain Only)

**Output K as Q16_1 directly from GEMM**, bypassing Q8_1:

```
Input (Q16_1) → RMSNorm → FP32 → QKV_GEMM → Q8_1 (Q,V), Q16_1 (K)
                                              ↓
                                         RoPE (Q8_1→Q16_1 for Q, Q16_1→Q16_1 for K)
```

### Required Changes

1. **QuantisedGemmKernel**: Add `multiply_with_precomputed_q8_1_to_q16_1()` method
   - Similar to existing `multiply_with_precomputed_q8_1_to_q8_1()`
   - Output quantization: INT32 accumulator → Q16_1 blocks
   - No FP32 intermediate (must stay in integer domain)

2. **FusedGEMM**: Add `execute_to_q8_1_and_q16_1()` for mixed output
   - Q projection → Q8_1
   - K projection → Q16_1
   - V projection → Q8_1

3. **GraphOrchestrator**: Allocate K buffer as Q16_1 in HybridQ16 mode

4. **FusedQKVGEMMStage**: Use mixed output path when K is Q16_1

5. **RoPEStage**: Add Q16_1→Q16_1 RoPE path for K (simpler than Q8_1→Q16_1)

---

## CRITICAL CONSTRAINT

**The HybridQ16 path must remain 100% integer/quantized domain.**

Do NOT:
- Add FP32 intermediate buffers
- Dequantize to FP32 for processing
- Use FP32 RoPE followed by requantization

The previous agent attempted FP32 K paths multiple times - these are FORBIDDEN.

---

## Files to Modify

| File | Change |
|------|--------|
| `src/v2/kernels/cpu/gemm_v4/QuantisedGemmKernel.h` | Add `multiply_with_precomputed_q8_1_to_q16_1()` |
| `src/v2/kernels/cpu/gemm_v4/QuantisedGemmJit_M1.h` | Add `GemmOutputFormat::Q16_1` enum, JIT codegen for Q16_1 output |
| `src/v2/kernels/cpu/gemm_v4/QuantisedGemmJit_M2.h` | Same JIT changes for M2 kernel |
| `src/v2/kernels/cpu/gemm_v4/FusedGEMM.h` | Add mixed output method |
| `src/v2/kernels/cpu/gemm_v4/FusedGEMM.cpp` | Implement mixed output |
| `src/v2/execution/GraphOrchestrator.cpp` | K buffer as Q16_1 for HybridQ16 |
| `src/v2/execution/compute_stages/stages/FusedQKVGEMMStage.cpp` | Mixed precision dispatch |
| `src/v2/kernels/cpu/primitives/RoPEPrimitives.h` | Add Q16_1→Q16_1 RoPE |
| `src/v2/kernels/cpu/primitives/RoPEPrimitives.cpp` | Implement Q16_1→Q16_1 RoPE |
| `src/v2/execution/compute_stages/stages/RoPEStage.cpp` | Q16_1 K input path |

---

## Q16_1 Output Quantization (for GEMM)

The INT32 accumulator from GEMM needs to be converted to Q16_1:

```cpp
// After GEMM: acc_i32 contains INT32 results
// Need to convert to Q16_1 with dynamic scale

// Step 1: Find max absolute value across block
int32_t max_abs = 0;
for (int i = 0; i < block_size; ++i) {
    max_abs = std::max(max_abs, std::abs(acc_i32[i]));
}

// Step 2: Compute scale (d) to fit in int16 range
// Scale such that max_abs maps to ~16000 (leaving headroom)
float d = static_cast<float>(max_abs) / 16000.0f;
float inv_d = (d > 0) ? (1.0f / d) : 0.0f;

// Step 3: Quantize to int16
int32_t sum_qs = 0;
for (int i = 0; i < block_size; ++i) {
    int32_t q = static_cast<int32_t>(std::round(acc_i32[i] * inv_d));
    q = std::clamp(q, -16383, 16383);
    block.qs[i] = static_cast<int16_t>(q);
    sum_qs += block.qs[i];
}
block.d = d;
block.sum_qs = sum_qs;
```

**Note**: This can be done in integer domain with fixed-point arithmetic to avoid FP32.

---

## Test Commands

```bash
# Build integration tests
cmake --build build_v2_integration --parallel

# Run HybridQ16 vs FP32 comparison
./build_v2_integration/tests/v2/v2_integration_hybridq16_vs_fp32_pipeline

# Check specific stage cosines
LLAMINAR_LOG_LEVEL=INFO ./build_v2_integration/tests/v2/v2_integration_hybridq16_vs_fp32_pipeline 2>&1 | grep -E "K_ROPE|cosine"
```

---

## Expected Results After Fix

| Stage | Current Cosine | Expected Cosine |
|-------|---------------|-----------------|
| K_PROJECTION | ~0.999 | ~0.999 |
| K_ROPE | **~0.878** | **>0.99** |
| ATTENTION_CONTEXT | ~0.95 | >0.98 |
| Final logits | ~0.85 | >0.95 |

---

## Reference: Existing Q8_1 Output Implementation

Look at `multiply_with_precomputed_q8_1_to_q8_1()` in `QuantisedGemmKernel.h` (lines 1994-2198) for the pattern:

1. JIT kernel writes to `params.C_q8_1` instead of `params.C`
2. `params.output_format = GemmOutputFormat::Q8_1`
3. JIT code checks format and branches to quantization epilogue
4. Quantization happens in-register before memory write

The Q16_1 version would be similar but with:
- `GemmOutputFormat::Q16_1 = 4`
- Different quantization math (int16 range, different block structure)
- Q16_1Block output instead of Q8_1Block

---

## Questions for Continuation

1. Should Q16_1 GEMM output use fixed scale (like KV cache) or dynamic scale (like Q8_1)?
2. Should we add JIT support for Q16_1 output, or use a scalar/AVX epilogue?
3. Should Q16_1→Q16_1 RoPE be a separate kernel or templated with existing Q8_1→Q16_1?

---

## Files Already Reverted

The following files were modified but reverted back to baseline:
- `src/v2/execution/GraphOrchestrator.cpp`
- `src/v2/execution/compute_stages/stages/FusedQKVGEMMStage.cpp`
- `src/v2/execution/compute_stages/stages/RoPEStage.cpp`

The codebase is currently at a clean state on `feature/typed-residuals`.
