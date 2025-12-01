# SwiGLU Formula Bug Fix

**Date:** 2025-12-01  
**Type:** Critical Bug Fix  
**Impact:** Inference quality - fixes garbage output in all Qwen2 models

## Summary

Fixed a critical bug in the SwiGLU activation function where the gate and up projections were swapped in the formula. This bug caused all Llaminar inference to produce garbage output.

## Root Cause

The SwiGLU implementation was computing:
```
output = gate * silu(up)   ← WRONG
```

But the correct HuggingFace formula is:
```python
# From HuggingFace Qwen2MLP.forward():
# return self.down_proj(self.act_fn(gate_proj) * up_proj)
output = silu(gate) * up   ← CORRECT
```

The SiLU activation should be applied to the **gate** projection output, not the **up** projection.

## Files Modified

### Core Fix
- `src/v2/kernels/cpu/primitives/SwiGLUPrimitives.cpp` - Fixed all implementations:
  - AVX512 vectorized path (lines 163-193)
  - AVX2 vectorized path (lines 87-122)
  - Scalar fallback (lines 217-224)
  - BF16 implementation (lines 228-239)
  - FP16 implementation (lines 241-252)
  - Q8_1 AVX512 path (lines 452-465, 553-579)
  - Q8_1 AVX2 path (lines 311-327)
  - Q8_1 scalar fallback (lines 639-649)

### Documentation Updates
- `src/v2/pipelines/ops/SwiGLUOp.h` - Updated comments to reflect correct formula
- `src/v2/kernels/cpu/fused/FusedDequantSwiGLU.cpp` - Updated comment

### PyTorch Reference Fix
- `python/reference/generate_qwen2_pipeline_snapshots.py` - Fixed `_ffn_block_impl()` to use correct formula

### New Regression Test
- `tests/v2/unit/Test__SwiGLUFormula.cpp` - 6 test cases to lock in correct formula
- `tests/v2/CMakeLists.txt` - Added test target `V2_Unit_SwiGLUFormula`

## Test Cases Added

The new `Test__SwiGLUFormula` test suite includes:

1. **CorrectFormulaIsSiluGateTimesUp** - Verifies formula with values where the two formulations produce significantly different results
2. **AsymmetricValuesNotSwapped** - Ensures gate/up arguments aren't accidentally swapped
3. **DefinitiveFormulaTest** - Uses gate=1, up=10 where correct gives ~7.31 vs incorrect ~9.99
4. **LargeArraySIMD** - Tests 1024 elements to exercise AVX2/AVX512 paths
5. **EdgeCases** - Tests zeros, negatives, and large values
6. **DocumentReference** - Documents the HuggingFace formula for future reference

## Verification

### Before Fix
```
The capital of France is".



 levels]init".



地上时候".
```

### After Fix
```
The capital of France is Paris, and it is the capital of Europe.
```

### E2E Parity Test Results
```
LM head Top-1:     100.00%  ✅
LM head Top-5:     100.00%  ✅
LM head KL div:    0.001944 (PASS)
```

All layers show cosine similarity >= 0.997 against PyTorch reference.

## How This Bug Was Found

1. User reported garbage output even with temperature=0 (greedy sampling)
2. Traced divergence layer-by-layer using snapshot comparison
3. Found EMBEDDING layer matched (cosine 0.996)
4. Found Layer 0 FFN_SWIGLU diverged significantly
5. Inspected HuggingFace Qwen2MLP source code
6. Discovered formula mismatch: we had `gate * silu(up)` but HuggingFace uses `silu(gate) * up`

## Prevention

The new `Test__SwiGLUFormula` test suite will catch any regression to this bug. The test uses carefully chosen values where the two formulations produce significantly different outputs (e.g., gate=1, up=10 gives ~7.31 correct vs ~9.99 incorrect).

## Related Files

- HuggingFace source: `transformers/models/qwen2/modeling_qwen2.py` - `Qwen2MLP.forward()`
- llama.cpp reference: Uses same formula in `llama.cpp/src/llama-model.cpp`
