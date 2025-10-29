# IQ Format Test Additions

**Date**: October 29, 2025  
**Session**: Adding IQ quantization format equivalency tests

## Summary

Added 6 new IQ format equivalency tests to the test suite. Discovered that **2 IQ formats have bugs** in their implementations (IQ3_S, IQ2_S), while **4 formats are skipped** due to mixed quantization in available models (IQ4_XS, IQ3_XXS, IQ2_XXS, IQ2_XS).

## Test Results

### IQ Format Test Status

| Format | Test Status | Mismatches | Notes |
|--------|-------------|------------|-------|
| IQ4_XS | ⏸️ SKIPPED | N/A | Models use mixed quantization (IQ4_NL/Q5_K, no pure IQ4_XS weights) |
| IQ3_XXS | ⏸️ SKIPPED | N/A | IQ3_XS model uses IQ4_NL for tested weights |
| IQ3_S | ❌ **FAILED** | 4863/4864 (99.98%) | **Implementation bug discovered** |
| IQ2_XXS | ⏸️ SKIPPED | N/A | IQ2_M model doesn't contain IQ2_XXS tensors |
| IQ2_XS | ⏸️ SKIPPED | N/A | IQ2_M model doesn't contain IQ2_XS tensors |
| IQ2_S | ❌ **FAILED** | 3456/3584 (96.4%) | **Implementation bug discovered** |

### Full Test Suite Status (18 total tests)

**11 PASSING** (0 mismatches):
- ✅ Q8_0, IQ4_NL, Q4_0, Q4_1, Q5_0, Q5_1, Q6_K, Q2_K, Q3_K, Q4_K, Q5_K

**2 FAILING** (new bugs discovered):
- ❌ IQ3_S (99.98% mismatch)
- ❌ IQ2_S (96.4% mismatch)

**5 SKIPPED**:
- ⏸️ Q8_K (no model available)
- ⏸️ IQ4_XS, IQ3_XXS, IQ2_XXS, IQ2_XS (mixed quantization, no pure tensors)

**Test Coverage**: 61% (11/18 implemented formats passing)

## Files Modified

### tests/v2/integration/Test__DequantEquivalency.cpp

Added 6 new IQ format tests (~350 lines):

1. **IQ4_XS_Equivalency** (lines 687-745)
   - Model: `Qwen2-0.5B.IQ4_XS.gguf`
   - Weight: `blk.0.attn_q.weight`
   - Status: Skipped (mixed quantization)

2. **IQ3_XXS_Equivalency** (lines 747-805)
   - Model: `Qwen2-0.5B.IQ3_XS.gguf` (note: XS model for XXS format)
   - Weight: `blk.0.attn_q.weight`
   - Status: Skipped (mixed quantization)

3. **IQ3_S_Equivalency** (lines 807-865)
   - Model: `Qwen2-0.5B.IQ3_S.gguf`
   - Weight: `blk.0.ffn_down.weight` ← Fixed after discovering mixed quantization
   - **Status: FAILING (99.98% mismatch)**
   - Max abs diff: 0.0792149
   - Max rel diff: 2.48162

4. **IQ2_XXS_Equivalency** (lines 867-932)
   - Model: `Qwen2.5-7B-Instruct-IQ2_M.gguf` (7B model)
   - Weight: Tried `blk.0.attn_q.weight`, `blk.0.ffn_gate.weight`
   - Status: Skipped (no IQ2_XXS tensors in model)

5. **IQ2_XS_Equivalency** (lines 934-997)
   - Model: `Qwen2.5-7B-Instruct-IQ2_M.gguf`
   - Weight: Tried `blk.0.ffn_up.weight`, `blk.0.ffn_down.weight`
   - Status: Skipped (no IQ2_XS tensors in model)

6. **IQ2_S_Equivalency** (lines 999-1060)
   - Model: `Qwen2.5-7B-Instruct-IQ2_M.gguf`
   - Weight: `blk.0.attn_k.weight` (found IQ2_S!)
   - **Status: FAILING (96.4% mismatch)**
   - Max abs diff: 5.1827e+06
   - Max rel diff: 1.15704e+09

## Discovered Bugs

### IQ3_S Implementation Bug

**Symptoms**:
- 4863/4864 elements mismatch (99.98% failure)
- Max absolute difference: 0.079
- Max relative difference: 2.48×

**Example Mismatches**:
```
Llaminar=0.00121839, llama.cpp=0.0177991, abs_diff=0.0165807
Llaminar=0.000731036, llama.cpp=0.038141, abs_diff=0.03741
Llaminar=0.000243679, llama.cpp=0.0279701, abs_diff=0.0277264
```

**Root Cause**: Unknown - requires investigation of `IQ3_STensor::decode_block_at()`

### IQ2_S Implementation Bug

**Symptoms**:
- 3456/3584 elements mismatch (96.4% failure)
- Max absolute difference: 5.18 million (!)
- Max relative difference: 1.15 billion (!)

**Example Mismatches**:
```
Llaminar=-0.000186563, llama.cpp=-0.00503719, abs_diff=0.00485063
Llaminar=-0.000583008, llama.cpp=-0.0157412, abs_diff=0.0151582
Llaminar=-0.00100277, llama.cpp=0.00503719, abs_diff=0.00603996
```

**Root Cause**: Unknown - requires investigation of `IQ2_STensor::decode_block_at()`

**Severity**: Extremely high - max diff of 5.18M suggests catastrophic algorithmic error (similar to Q2_K/Q3_K bugs we just fixed)

## Mixed Quantization Discovery

Many quantized models use **mixed quantization** - different tensor types for different layers:

### Qwen2-0.5B.IQ4_XS.gguf (336MB)
```
blk.0.attn_q.weight (iq4_nl)    ← Not IQ4_XS!
blk.0.ffn_gate.weight (iq4_nl)
blk.0.ffn_up.weight (iq4_nl)
blk.0.attn_v.weight (q5_1)
blk.0.ffn_down.weight (q5_K)
```

### Qwen2-0.5B.IQ3_XS.gguf (323MB)
```
blk.0.attn_q.weight (iq4_nl)    ← Not IQ3_XXS!
blk.0.ffn_down.weight (iq3_s)
blk.0.attn_v.weight (q5_0)
```

### Qwen2-0.5B.IQ3_S.gguf (323MB)
```
blk.0.attn_q.weight (iq4_nl)
blk.0.ffn_down.weight (iq3_s)   ← Found IQ3_S here!
blk.0.attn_v.weight (q5_0)
```

### Qwen2.5-7B-Instruct-IQ2_M.gguf (1.7GB)
```
blk.0.attn_q.weight (iq2_s)     ← Found IQ2_S!
blk.0.attn_k.weight (iq2_s)
blk.0.attn_v.weight (q4_K)
blk.0.ffn_gate.weight (iq2_s)
blk.0.ffn_up.weight (iq2_s)
blk.0.ffn_down.weight (iq3_s)
blk.0.attn_output.weight (iq3_s)
```

**Key Insight**: IQ2_M model uses IQ2_S (not IQ2_XXS or IQ2_XS). Model file name doesn't always match tensor types!

## Lessons Learned

### 1. Mixed Quantization is Common

**Issue**: Models don't use a single quantization format throughout  
**Pattern**: Higher-precision formats (Q5_K, IQ4_NL) for critical layers (attention Q/K/V)  
**Impact**: Can't test all formats even with correct model names  
**Solution**: Diagnostic tensor listing to find actual format usage

### 2. Model Names Can Be Misleading

**Issue**: `IQ2_M.gguf` contains IQ2_S, not IQ2_M  
**Issue**: `IQ3_XS.gguf` contains IQ3_S, not IQ3_XXS  
**Learning**: Model name indicates target quantization level, not exact format usage  
**Action**: Always inspect actual tensor types before assuming

### 3. IQ Formats Are Buggy

**Issue**: 2 out of 2 testable IQ formats have bugs (IQ3_S, IQ2_S)  
**Pattern**: Similar to Q2_K/Q3_K bugs (before fixes) - near-total failure  
**Root Cause**: Likely implemented from spec without llama.cpp comparison  
**Next Steps**: Fix IQ3_S and IQ2_S using same methodology as Q2_K/Q3_K

### 4. Test Discovery Process

**Pattern**: Add test → Run → Discover mixed quantization → Update weight name → Rerun  
**Example**: IQ3_S test initially tried `attn_q.weight` (iq4_nl) → changed to `ffn_down.weight` (iq3_s)  
**Improvement**: Diagnostic output helped quickly identify correct weights

## Test Implementation Pattern

All IQ tests follow this pattern:

```cpp
TEST_F(DequantEquivalencyTest, IQX_Format_Equivalency)
{
    model_path_ = "/path/to/model.gguf";
    
    ModelLoader loader;
    if (!loader.loadModel(model_path_)) {
        GTEST_SKIP() << "Model not available";
        return;
    }
    
    // Try various weights (mixed quantization)
    auto weight = loadWeight("blk.0.weight_name");
    if (!weight) {
        GTEST_SKIP() << "Weight not available";
        return;
    }

    auto tensor = std::dynamic_pointer_cast<IQX_FormatTensor>(weight);
    if (!tensor) {
        // DEBUG: List tensor types in model
        std::cerr << "\nDEBUG: Listing all blk.0 tensors:\n";
        const auto& model = loader.getModel();
        for (const auto& tensor_info : model.tensors) {
            if (tensor_info.name.find("blk.0.") != std::string::npos) {
                std::cerr << "  " << tensor_info.name 
                         << " (" << ggml_type_name(static_cast<ggml_type>(tensor_info.type)) << ")\n";
            }
        }
        GTEST_SKIP() << "No tensors of target format found";
        return;
    }

    // Dequantize and compare
    // ... standard comparison logic
}
```

**Key Features**:
- Graceful skipping if model/weight unavailable
- Diagnostic output on tensor type mismatch
- Multiple weight attempts for mixed quantization models

## Next Steps

### Immediate: Fix IQ Format Bugs

1. **IQ3_S Bug Fix**:
   - Read `src/v2/tensors/IQ3_STensor.cpp`
   - Compare with llama.cpp `ggml-quants.c` (dequantize_row_iq3_s)
   - Identify algorithmic differences
   - Apply fix (similar to Q2_K/Q3_K methodology)
   - Validate: 0 mismatches expected

2. **IQ2_S Bug Fix**:
   - Read `src/v2/tensors/IQ2_STensor.cpp`
   - Compare with llama.cpp (dequantize_row_iq2_s)
   - Fix catastrophic error (5.18M max diff suggests major issue)
   - Validate: 0 mismatches expected

### Short-Term: Expand Coverage

3. **Find More IQ Format Models**:
   - Download models with pure IQ4_XS, IQ3_XXS, IQ2_XXS, IQ2_XS weights
   - Alternative: Accept mixed quantization as valid test strategy
   - Goal: Test at least one tensor per format

4. **Add IQ1_M and IQ1_S Tests**:
   - Models available: `Qwen2.5-VL-7B-Instruct-UD-IQ1_M.gguf` (2.1GB)
   - Models available: `Qwen2.5-VL-7B-Instruct-UD-IQ1_S.gguf` (2.0GB)
   - Priority: Lower (1-bit quantization extremely aggressive, rare usage)

### Long-Term: Test Framework Improvements

5. **Improve Mixed Quantization Handling**:
   - Automatically scan model for available tensor types
   - Select best weight for each format
   - Avoid manual weight name updates

6. **Add Tolerance Analysis**:
   - IQ formats may have higher quantization error than Q formats
   - Current tolerance: 1e-5 (may be too strict)
   - Research expected error bounds for IQ formats

## Impact Assessment

### Test Coverage Comparison

| Category | Before | After | Change |
|----------|--------|-------|--------|
| **Total Tests** | 12 | 18 | +6 |
| **Passing** | 11 | 11 | 0 |
| **Failing** | 0 | 2 | +2 ⚠️ |
| **Skipped** | 1 | 5 | +4 |
| **Coverage** | 92% | 61% | -31% 📉 |

**Note**: Coverage decreased because we added 6 tests but only 2 are runnable (both failing). This is actually **good** - we discovered hidden bugs!

### Bug Discovery Value

**Without IQ tests**: 11/12 formats tested → 92% coverage → False confidence  
**With IQ tests**: 11/18 formats passing → Discovered 2 critical bugs → **Real confidence**

**Impact**: IQ format bugs would cause silent failures in production inference with IQ-quantized models.

## Conclusion

Successfully added 6 IQ format equivalency tests, discovering:
- ✅ **2 new bugs** in IQ3_S and IQ2_S implementations (critical finds!)
- ✅ **Mixed quantization pattern** in modern quantized models
- ✅ **Test infrastructure works** (graceful skipping, diagnostics, easy bug discovery)

The bugs in IQ3_S and IQ2_S mirror the bugs we fixed in Q2_K and Q3_K earlier today - suggesting a pattern of implementing complex formats from spec without empirical validation. The same fix methodology should work:

1. Read llama.cpp reference implementation
2. Identify structural/algorithmic differences
3. Rewrite to match reference exactly
4. Validate with 0 mismatches

**Next Priority**: Fix IQ3_S and IQ2_S bugs before expanding to more IQ formats.
