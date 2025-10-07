# COSMA vs PyTorch Parity Status

## Current Status: ❌ FAILING

**Test**: `ParityFramework.COSMAPrefillVsPyTorch`  
**Result**: 146/147 tests failed (99.3% failure rate)  
**First divergence**: `ATTENTION_NORM_layer0` with `max_abs=0.0851`, `rel_l2=0.021`

## What We Fixed

✅ **Scatter/Gather Bug** (RESOLVED):
- **Location**: `src/cosma_prefill_manager.cpp:1043-1073`
- **Issue**: Fast-path stored dangling pointer to temporary input
- **Symptom**: Small matrices (below threshold) returned zeros after reconstruction
- **Fix**: Dual-ownership pattern (`host_owned` + `original_row_major`)
- **Verification**: All scatter/gather roundtrip tests PASS ✅
- **Impact**: Fixed MLP TP parity test (rel_l2 improved from 0.94 to 2e-07) ✅

## What's Still Broken

❌ **PyTorch Parity** (UNSOLVED):
- **Symptom**: ALL layers diverge starting from layer 0
- **Error magnitude**: Moderate (max_abs ~0.08-250, rel_l2 ~0.02-1.9)
- **Pattern**: Consistent failures across ALL 147 comparison points
- **First failure**: ATTENTION_NORM_layer0 (the very first operation!)

## Analysis

### Different Root Causes

The scatter/gather bug and PyTorch parity are **separate issues**:

1. **Scatter/Gather Bug** (FIXED):
   - Affected: Small matrices only (< fast_path_threshold)
   - Symptom: Complete corruption (ALL ZEROS)
   - Test case: 8×8, 32×896 matrices → 100% mismatch
   - Fixed by: Data ownership fix

2. **PyTorch Parity** (UNFIXED):
   - Affected: ALL matrices, all layers
   - Symptom: Moderate numerical divergence
   - First failure: Layer 0, ATTENTION_NORM
   - Magnitude: Small errors that compound through layers

### Hypotheses for PyTorch Parity Failure

1. **Weight Layout Mismatch**:
   - PyTorch uses different memory layout than GGUF
   - COSMA may require specific data orientation
   - Need to verify transpose flags in COSMA calls

2. **Normalization Issues**:
   - First failure is in ATTENTION_NORM (RMSNorm)
   - Possible epsilon/gamma handling differences
   - May be related to distributed RMSNorm computation

3. **Activation Function Differences**:
   - SwiGLU, RoPE, or other nonlinearities
   - Precision differences (float32 vs PyTorch's mixed precision)

4. **Accumulation Order**:
   - COSMA may accumulate in different order than PyTorch
   - Could cause small FP errors that compound

5. **COSMA Parameter Tuning**:
   - Strategy selection may be suboptimal
   - Memory layout not matching PyTorch's expectations

## Critical Finding ⚠️

**OpenBLAS vs PyTorch ALSO FAILS** (145/147 tests, 98.6% failure rate)!

This proves the PyTorch parity problem is **NOT COSMA-specific**. It's a fundamental issue with llaminar's implementation compared to PyTorch.

### Test Results Comparison

| Test | Failed/Total | First Divergence | Root Cause |
|------|-------------|------------------|------------|
| COSMA vs PyTorch | 146/147 (99.3%) | ATTENTION_NORM_layer0 | ❓ llaminar vs PyTorch |
| OpenBLAS vs PyTorch | 145/147 (98.6%) | ATTENTION_OUTPUT_layer0 | ❓ llaminar vs PyTorch |
| MLP TP Parity (COSMA) | 0/1 (0%) | N/A - PASSED ✅ | scatter/gather (FIXED) |

### Conclusion

1. **Scatter/Gather Bug** (FIXED) ✅:
   - Specific to COSMA integration
   - Affected fast-path reconstruction
   - Fixed by dual-ownership pattern
   - MLP TP parity now PASSES

2. **PyTorch Parity** (UNFIXED) ❌:
   - Affects BOTH OpenBLAS and COSMA
   - Problem is in llaminar's core implementation
   - NOT related to backend selection
   - Likely architectural difference from PyTorch

## Next Steps

### Immediate Actions

1. **✅ COMPLETED: Compare OpenBLAS vs PyTorch**:
   - Result: BOTH fail similarly
   - Conclusion: Not a COSMA-specific bug
   - Root cause: llaminar vs PyTorch architectural difference

2. **Isolate First Failure**:
   - Debug ATTENTION_NORM_layer0 specifically
   - Compare input/output with PyTorch layer-by-layer
   - Check RMSNorm epsilon, gamma values

3. **Check Weight Loading**:
   - Verify GGUF→COSMA weight conversion
   - Compare weight values with PyTorch reference
   - Check transpose flags

4. **Enable Detailed Logging**:
   ```bash
   export LLAMINAR_COSMA_LOG_LEVEL=trace
   export LLAMINAR_COSMA_VALIDATE_TILE=64
   ```

### Diagnostic Commands

```bash
# Run OpenBLAS parity for comparison
cd /workspaces/llaminar
mpirun -np 2 ./build/test_parity_framework \
  --gtest_filter=ParityFramework.OpenBLASPrefillVsPyTorch

# Run with validation enabled
export LLAMINAR_COSMA_VALIDATE_TILE=64
export LLAMINAR_COSMA_LOG_LEVEL=trace
mpirun -np 2 ./build/test_parity_framework \
  --gtest_filter=ParityFramework.COSMAPrefillVsPyTorch

# Compare single layer in detail
export LLAMINAR_PARITY_LAYER_DETAIL=0
mpirun -np 2 ./build/test_parity_framework \
  --gtest_filter=ParityFramework.COSMAPrefillVsPyTorch
```

## Error Pattern Summary

From the log output, we see systematic divergence:

| Layer Stage | Error Pattern |
|------------|---------------|
| ATTENTION_NORM | max_abs: 0.08-59, rel_l2: 0.02-1.9 |
| ATTENTION_OUTPUT | max_abs: 2-29, rel_l2: 1.1-2.4 |
| ATTENTION_RESIDUAL | max_abs: 70-1572, rel_l2: 1.1-1.6 |
| FFN_NORM | max_abs: 22-249, rel_l2: 1.6-1.9 |
| FFN_DOWN | max_abs: 3-1491, rel_l2: 1.2-3.7 |
| FFN_RESIDUAL | max_abs: 70-1572, rel_l2: 1.3-1.9 |
| FINAL_NORM | max_abs: 220, rel_l2: 1.27 |
| LM_HEAD | max_abs: 38, rel_l2: 1.16 |

**Observation**: Residual connections have HUGE errors (~1500+) because they accumulate all previous layer errors.

## Conclusion

The scatter/gather bug fix was **correct and necessary**, but it does NOT solve the PyTorch parity problem. There is a **separate, fundamental issue** with how COSMA-based prefill computes results compared to PyTorch.

The next step is to determine if this is:
1. A llaminar implementation bug (fixable)
2. A COSMA library issue (needs upstream fix)
3. An expected numerical difference (needs tolerance tuning)

**Priority**: Run OpenBLAS vs PyTorch comparison FIRST to isolate whether this is COSMA-specific.
