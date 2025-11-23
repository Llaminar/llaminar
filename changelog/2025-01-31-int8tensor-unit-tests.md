# INT8Tensor Unit Tests & E2E Investigation Summary - 2025-01-31

## Unit Test Implementation ✅

### Overview
Created comprehensive unit test suite for INT8Tensor with **11 tests, all passing**.

Test file: `tests/v2/unit/Test__INT8Tensor.cpp`

### Tests Implemented

1. **BasicConstruction** - Verify INT8Tensor creation from FP32 data
2. **GlobalScaleDequantization** - Weight tensor path (single global scale)
3. **PerRowScaleDequantization** - FusedRMSNormQuantize output path (per-row scales)
4. **PartialBufferDequantization** ⭐ - CRITICAL: Tests the bug we fixed
   - Buffer capacity: 4096 rows
   - Valid data: 4 rows
   - Verifies only valid rows dequantized, rest zeroed
5. **INT8DataAccess** - Direct int8_t* access
6. **ScaleUpdate** - Scale modification handling
7. **EmptyRowScalesFallback** - Falls back to global scale when no row scales
8. **ZeroScalesHandling** - Handles zero-valued rows correctly
9. **RepeatedDequantization** - Caching behavior
10. **RealisticBatchScenario** ⭐ - Real-world batch: 512 buffer, 16 valid rows (d_model=896)
11. **ShapeValidation** - Shape and type metadata

### Test Results
```
[==========] 11 tests from 1 test suite ran. (123 ms total)
[  PASSED  ] 11 tests.
```

**Key validations:**
- Partial buffer dequantization works correctly ✅
- No "Insufficient row scales" errors ✅
- Zero-fill of unused buffer space ✅
- Per-row scale registration and retrieval ✅

## E2E Test Investigation - Current Status

### Before INT8Tensor Fix
```
E2E Results: 4/7 tests FAILING
Error: "[INT8Tensor] Insufficient row scales: have 4, need 4096"
Divergences: max_diff=22, rel_l2=1.39 (catastrophic)
```

### After INT8Tensor Fix
```
E2E Results: Still 4/7 tests FAILING
No "Insufficient scales" errors ✅
Divergences: SAME (max_diff=22, rel_l2=1.39)
```

### Critical Finding: Sequence-Specific Divergence

**MultiSequenceBatchEqualLength Test Results:**
```
Sequence 0:
  Max abs diff:   0        ✅ PERFECT!
  Mean abs diff:  0
  Rel L2 norm:    0
  Status:         PASSED

Sequence 1:
  Max abs diff:   22.0172  ❌ CATASTROPHIC!
  Mean abs diff:  3.26887
  Rel L2 norm:    1.39387
  Mismatches:     303825
  Status:         FAILED
```

**Interpretation:**
- First sequence in batch: **Perfect parity** (diff=0)
- Second sequence in batch: **Massive divergence** (diff=22)
- This is **NOT** an INT8Tensor dequantization issue
- This is a **batched execution bug** affecting only non-first sequences

### Hypothesis: Batched Padding/Offset Issue

Possible causes for sequence 1 divergence:
1. **Incorrect offset calculation** when extracting sequence 1 from batch
2. **Padding not handled correctly** between sequences
3. **K/V cache corruption** for second sequence
4. **Attention mask issues** in batched execution
5. **Residual connection** accumulating wrong data

### INT8Tensor Dequantization Status

**CONFIRMED WORKING:**
- ✅ Per-row scale registration
- ✅ Partial buffer dequantization (only valid rows)
- ✅ Zero-fill of unused buffer space
- ✅ No buffer overflows or scale access errors
- ✅ Unit tests validate all edge cases

**NOT THE PROBLEM:**
- INT8Tensor fix didn't change E2E divergences
- Divergence pattern (Seq0=perfect, Seq1=wrong) suggests higher-level issue

## Next Steps

### Immediate Investigation
1. **Check sequence extraction in test**
   - Verify offset calculation for sequence 1
   - Check if padding is accounted for

2. **Debug batched forward pass**
   - Add logging for sequence boundaries
   - Verify K/V cache indexing for each sequence
   - Check attention mask generation

3. **Compare sequential vs batched**
   - Run both paths with extensive logging
   - Find first point of divergence
   - Isolate layer causing sequence 1 corruption

### Code Review Targets
- `forward_batch()` in Qwen2Pipeline
- Batch padding utilities
- K/V cache management for batched execution
- Snapshot extraction logic (test-side issue?)

## Files Modified Today

1. **src/v2/tensors/INT8Tensor.cpp**
   - Fixed `data()` to dequantize only rows with scales
   - Added zero-fill for unused buffer space
   - Added debug logging for partial buffer case

2. **src/v2/pipelines/qwen/Qwen2Pipeline.cpp**
   - Fixed snapshot capture to use effective_seq_len
   - Added scale validation logging

3. **tests/v2/unit/Test__INT8Tensor.cpp** (NEW)
   - 11 comprehensive unit tests
   - All tests passing

4. **tests/v2/CMakeLists.txt**
   - Added v2_test_int8_tensor target

5. **changelog/2025-01-31-e2e-test-investigation.md**
   - Documented investigation process

## Conclusion

**INT8Tensor work: ✅ COMPLETE**
- Unit tests comprehensive and passing
- Dequantization bug fixed and validated
- No longer causing E2E test failures

**E2E divergence: ❌ STILL PRESENT**
- Root cause is NOT INT8Tensor
- Pattern suggests batched execution bug affecting non-first sequences
- Likely in forward_batch() pipeline logic, K/V cache, or test extraction
- Requires deeper investigation into batched forward pass

**Progress Assessment:**
- 🎉 Major win: INT8Tensor thoroughly tested and working
- 📍 Identified: E2E issue is sequence-specific (Seq0=perfect, Seq1=wrong)
- ⚠️ Blocker: Need to fix batched execution for E2E tests to pass
