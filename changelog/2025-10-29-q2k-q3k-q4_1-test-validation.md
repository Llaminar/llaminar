# Q2_K, Q3_K, Q4_1 Test Validation Results

**Date**: October 29, 2025  
**Session**: Model download and test validation for previously skipped tensor formats

## Objective

Enable and validate equivalency tests for Q2_K, Q3_K, and Q4_1 tensor formats by:
1. Downloading appropriate 7B quantized models
2. Updating test cases to use correct model paths
3. Identifying proper tensor weights for each format
4. Running equivalency tests against llama.cpp reference implementation

## Model Downloads

Successfully downloaded 5 out of 6 models (~18GB):

| Model | Size | Format Coverage | Status |
|-------|------|----------------|---------|
| Qwen2.5-7B-Instruct-Q2_K.gguf | 2.9GB | Q2_K (mixed quant) | ✅ Downloaded |
| Qwen2.5-7B-Instruct-Q3_K_M.gguf | 3.6GB | Q3_K (mixed quant) | ✅ Downloaded |
| Qwen2.5-7B-Instruct-Q4_0.gguf | 4.2GB | Q4_0 + Q4_1 (mixed) | ✅ Downloaded |
| Qwen2.5-7B-Instruct-IQ4_XS.gguf | 4.0GB | IQ4_XS | ✅ Downloaded |
| Qwen2.5-7B-Instruct-IQ3_XS.gguf | 3.2GB | IQ3_XXS + IQ3_S | ✅ Downloaded |
| Qwen2.5-7B-Instruct-IQ2_M.gguf | 2.6GB | IQ2_S | ❌ Incomplete (1.7GB) |

**Note**: CTest timeout (600s) interrupted download during 6th model.

## Mixed Quantization Discovery

All 7B models use **mixed quantization** strategies where different layers use different quantization formats. This is an optimization technique where more critical layers (e.g., FFN down projections, attention outputs) use higher precision formats.

### Q4_0 Model (labeled) - Actual Distribution

```
blk.0.attn_norm.weight     → f32 (full precision)
blk.0.ffn_down.weight      → q4_1 ✓ (target format)
blk.0.ffn_gate.weight      → q4_0
blk.0.ffn_up.weight        → q4_0
blk.0.attn_k.weight        → q4_0
blk.0.attn_q.weight        → q4_0
blk.0.attn_v.weight        → q4_0
blk.0.attn_output.weight   → q4_0
```

**Finding**: File labeled "Q4_0" actually contains Q4_1 tensors in FFN down projections. Test updated to use `blk.0.ffn_down.weight`.

### Q2_K Model - Actual Distribution

```
blk.0.attn_norm.weight     → f32
blk.0.ffn_down.weight      → q3_K (higher precision)
blk.0.ffn_gate.weight      → q2_K ✓
blk.0.ffn_up.weight        → q2_K ✓
blk.0.attn_k.weight        → q2_K ✓
blk.0.attn_q.weight        → q2_K ✓
blk.0.attn_output.weight   → q3_K (higher precision)
blk.0.attn_v.weight        → q4_K (even higher precision)
```

**Finding**: Critical layers (ffn_down, attn_output, attn_v) use higher precision (Q3_K/Q4_K). Test updated to use `blk.0.ffn_gate.weight`.

### Q3_K_M Model - Actual Distribution

```
blk.0.attn_norm.weight     → f32
blk.0.ffn_down.weight      → q5_K (higher precision)
blk.0.ffn_gate.weight      → q3_K ✓
blk.0.ffn_up.weight        → q3_K ✓
blk.0.attn_k.weight        → q3_K ✓
blk.0.attn_q.weight        → q3_K ✓
blk.0.attn_output.weight   → q4_K (higher precision)
blk.0.attn_v.weight        → q5_K (highest precision)
```

**Finding**: FFN down and attention V use higher precision (Q4_K/Q5_K). Test updated to use `blk.0.ffn_gate.weight`.

## Test Execution Results

### Q4_1 Equivalency Test
- **Status**: ❌ FAILED
- **Model**: Qwen2.5-7B-Instruct-Q4_0.gguf
- **Weight**: `blk.0.ffn_down.weight` (q4_1, 18944 elements)
- **Mismatches**: 12815 / 18944 (67.7% failure rate)
- **Max Absolute Diff**: 0.0831
- **Max Relative Diff**: 4125 (extremely high)

**Sample Mismatches**:
```
[9]:  Llaminar= 0.0218201, llama.cpp=-0.0148773, diff=0.0367
[10]: Llaminar=-0.0075378, llama.cpp= 0.0071411, diff=0.0147
[11]: Llaminar=-0.0001984, llama.cpp= 0.0071411, diff=0.0073
```

**Diagnosis**: Significant implementation bug in Q4_1 dequantization logic. Mismatches are not random noise but systematic errors (signs flipped, magnitudes wrong).

### Q2_K Equivalency Test
- **Status**: ❌ FAILED
- **Model**: Qwen2.5-7B-Instruct-Q2_K.gguf
- **Weight**: `blk.0.ffn_gate.weight` (q2_K, 3584 elements)
- **Mismatches**: 3542 / 3584 (98.8% failure rate)
- **Max Absolute Diff**: (not displayed)
- **Max Relative Diff**: (not displayed)

**Diagnosis**: Nearly total failure suggests fundamental bug in Q2_K dequantization algorithm.

### Q3_K Equivalency Test
- **Status**: ❌ FAILED
- **Model**: Qwen2.5-7B-Instruct-Q3_K_M.gguf
- **Weight**: `blk.0.ffn_gate.weight` (q3_K, 3584 elements)
- **Mismatches**: 3110 / 3584 (86.8% failure rate)
- **Max Absolute Diff**: (not displayed)
- **Max Relative Diff**: (not displayed)

**Diagnosis**: High failure rate indicates serious bug in Q3_K dequantization logic.

## Root Cause Analysis

### Why These Tests Were Skipped Before

1. **Q2_K and Q3_K**: Hard-coded `GTEST_SKIP()` with comment "mixed quantization (no pure Q2_K tensors available)"
   - Issue: 0.5B models use pure formats, but tests assumed 7B models
   - Reality: 7B models *do* have Q2_K/Q3_K tensors, just in mixed configuration
   
2. **Q4_1**: Used non-existent model `qwen2.5-0.5b-instruct-q4_1.gguf` (404 error)
   - Issue: Q4_1 is legacy format, rarely quantized at 0.5B scale
   - Solution: Found Q4_1 tensors in "Q4_0" labeled 7B model

### Why Tests Are Failing Now

These implementations were **never properly validated** against llama.cpp reference:

1. **No Ground Truth Testing**: Tests were skipped, so bugs never surfaced
2. **Incorrect Assumptions**: May have been implemented based on spec/docs without empirical validation
3. **Complex Formats**: K-quant formats are significantly more complex than simple formats (Q4_0, Q8_0)

**Evidence**: Other formats with working tests show 0 mismatches:
- Q8_0: ✅ 0 mismatches (perfect)
- IQ4_NL: ✅ 0 mismatches (perfect)
- Q4_0: ✅ 0 mismatches (perfect)
- Q5_0: ✅ 0 mismatches (perfect)
- Q5_1: ✅ 0 mismatches (perfect)
- Q6_K: ✅ 0 mismatches (perfect)
- Q4_K: ✅ 0 mismatches (perfect)
- Q5_K: ✅ 0 mismatches (perfect)

This proves our testing framework is sound. The failures indicate real bugs in Q2_K, Q3_K, and Q4_1 implementations.

## Test Infrastructure Improvements

### Debug Diagnostic Added

When a test fails type checking, it now lists all `blk.0.*` tensors with their actual types:

```cpp
if (!q2k_tensor) {
    std::cerr << "\nDEBUG: Listing all blk.0 tensors in Q2_K model:\n";
    const auto& model = loader.getModel();
    for (const auto& tensor_info : model.tensors) {
        if (tensor_info.name.find("blk.0.") != std::string::npos) {
            std::cerr << "  " << tensor_info.name 
                      << " (" << ggml_type_name(static_cast<ggml_type>(tensor_info.type)) 
                      << ")\n";
        }
    }
    GTEST_SKIP() << "blk.0.ffn_down.weight is not Q2_K type in this model";
    return;
}
```

This helped identify the correct weights to use for each format.

### Files Modified

1. **tests/v2/integration/Test__DequantEquivalency.cpp**:
   - Q4_1 test: Changed from `blk.0.attn_q.weight` (q4_0) to `blk.0.ffn_down.weight` (q4_1)
   - Q2_K test: Changed from `GTEST_SKIP()` to full implementation using `blk.0.ffn_gate.weight`
   - Q3_K test: Changed from `GTEST_SKIP()` to full implementation using `blk.0.ffn_gate.weight`
   - Added debug diagnostics to list available tensors on type mismatch

2. **scripts/fetch_test_models.sh**:
   - Added 6 new 7B model URLs to `LARGE_7B_IQ_MODELS` array
   - Gated by `LLAMINAR_FETCH_7B_IQ_MODELS=1` environment variable

## Next Steps

### Immediate (Blocking)

1. **Fix Q4_1 Dequantization** (`src/v2/tensors/Q4_1Tensor.cpp`):
   - Review decode algorithm against llama.cpp reference
   - Check scale/min handling (Q4_1 uses min offset, Q4_0 doesn't)
   - Validate bit packing/unpacking logic
   - **Expected**: 0 mismatches like other working formats

2. **Fix Q2_K Dequantization** (`src/v2/tensors/Q2_KTensor.cpp`):
   - Review K-quant superblock structure
   - Validate scale/delta decoding
   - Check bit unpacking for 2-bit quantization
   - Compare block layout with llama.cpp
   - **Expected**: 0 mismatches

3. **Fix Q3_K Dequantization** (`src/v2/tensors/Q3_KTensor.cpp`):
   - Review 3-bit packing logic
   - Validate scale/delta calculations
   - Check high-bit handling
   - **Expected**: 0 mismatches

### Follow-Up (Non-Blocking)

4. **Resume IQ2_M Download**:
   - Complete download of 6th model (~900MB remaining)
   - Validates IQ2_S format coverage

5. **Add IQ Format Tests** (4 new tests):
   - IQ4_XS_Equivalency (model ready)
   - IQ3_XXS_Equivalency (model ready)
   - IQ3_S_Equivalency (model ready)
   - IQ2_S_Equivalency (pending IQ2_M download)

6. **Document Mixed Quantization Strategy**:
   - Create guide explaining why 7B models use mixed formats
   - Document which layers typically get which precision
   - Add to `.github/instructions/`

## Lessons Learned

### Test-First Development

1. **Ground Truth Validation is Critical**: All 3 failing formats were never properly tested
2. **Skip Tests Are Technical Debt**: Hard-coded skips hide implementation bugs
3. **Model Availability Matters**: Need appropriate test models for all formats

### Mixed Quantization Reality

1. **Models Don't Match Labels**: "Q4_0" file contains Q4_1 tensors
2. **Strategic Precision**: Critical layers use higher precision (FFN down, attention output/V)
3. **Format Distribution**: Need to inspect actual tensor types, not assume from filename

### Debug Infrastructure Value

1. **Diagnostic Listings**: Printing available tensors saved hours of trial-and-error
2. **Type Validation**: Early type checking prevents cryptic runtime errors
3. **Incremental Testing**: Test one format at a time, fix before moving on

## Test Coverage Status

### Before This Session
- **8 PASSING**: Q8_0, IQ4_NL, Q4_0, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K
- **4 SKIPPING**: Q4_1, Q2_K, Q3_K, Q8_K (no models)
- **6 MISSING**: IQ4_XS, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ2_S, IQ3_S

### After This Session
- **8 PASSING**: Q8_0, IQ4_NL, Q4_0, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K
- **3 FAILING**: Q4_1, Q2_K, Q3_K (implementations buggy)
- **1 SKIPPING**: Q8_K (no model)
- **6 READY**: IQ4_XS, IQ3_XXS, IQ3_S (models downloaded), IQ2_XXS, IQ2_XS, IQ2_S (pending tests)

### Target (After Fixes)
- **11+ PASSING**: All above + Q4_1, Q2_K, Q3_K, IQ4_XS, IQ3_XXS, IQ3_S
- **Test Coverage**: 92%+ (11+ out of 12 implemented quantization formats)

## Performance Notes

### Download Timing
- **Total Size**: ~18GB (5 models)
- **Total Time**: ~600 seconds (10 minutes, CTest timeout)
- **Average Speed**: ~30 MB/s from HuggingFace

### Test Execution
- Q4_1 test: 292ms (loading 4.2GB model)
- Q2_K test: 199ms (loading 2.9GB model)
- Q3_K test: 204ms (loading 3.6GB model)

**Note**: Fast execution despite large models suggests efficient tensor-on-demand loading.

## References

- **Model Source**: [bartowski/Qwen2.5-7B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF)
- **llama.cpp Dequant Reference**: `third-party/ggml/src/ggml-quants.c`
- **Test Framework**: `tests/v2/integration/Test__DequantEquivalency.cpp`
- **Previous Session**: `changelog/2025-10-29-q5_0-q5_1-tensor-implementation.md`
