# Batch Parity Test Status - January 17, 2025

## Summary

Investigated batch parity testing infrastructure. Found 1 working test suite (`test_batch_correctness`) with 3 test cases, and 1 broken/incomplete test (`test_batch_sequential_parity`).

## Test Suite: `test_batch_correctness.cpp`

**Status**: ✅ Executable builds and runs
**Location**: `tests/test_batch_correctness.cpp`
**Test Cases**: 3 tests

### Test 1: `BatchedAttentionStagesParity`
**Status**: ✅ **PASSING** (Validated)
**Runtime**: 72 seconds
**Description**: Stage-by-stage comparison of batch vs sequential execution through attention mechanism
**Coverage**: 8 stages validated:
- EMBEDDING
- ATTENTION_NORM
- Q_PROJECTION
- K_PROJECTION  
- V_PROJECTION
- ROPE_APPLICATION
- ATTENTION_CONTEXT
- ATTENTION_OUTPUT

**Results**:
```
726 total snapshots captured (387 sequential, 339 batch)
All 8 stages: max_diff=0 (exact match)
✓ ALL TESTED STAGES MATCH!
```

**Validation Command**:
```bash
timeout 120 mpirun -np 2 --oversubscribe ./build/test_batch_correctness \
  --gtest_filter="BatchCorrectnessTest.BatchedAttentionStagesParity"
```

### Test 2: `PrefillBatchVsSequential`
**Status**: ⏱️ **TIMEOUT** (180s → increased to 600s)
**Description**: End-to-end prefill comparison: batch execution vs sequential per-sequence
**Issue**: Loads full model multiple times (once for batch, once for each sequence in batch)
**Performance**: Very slow due to repeated model loading
**Next Steps**: 
- Validate with increased 600s timeout
- Consider optimizing to reuse model weights

### Test 3: `DecodeBatchVsSequential`  
**Status**: ⏳ **NOT YET TESTED**
**Description**: End-to-end decode comparison: batch execution vs sequential per-sequence
**Expected**: Similar timeout issues as PrefillBatchVsSequential

## Test Suite: `test_batch_sequential_parity.cpp` 

**Status**: ❌ **DISABLED** (Incomplete Implementation)
**Location**: `tests/test_batch_sequential_parity.cpp`
**Reason**: Uses non-existent API `PipelineSnapshotManager::getAllSnapshots()`
**CMakeLists**: Commented out lines 1310-1316
**Resolution**: Use `test_batch_correctness.cpp` instead (functional replacement)

## Configuration Changes

### CMakeLists.txt
```cmake
# Increased timeout from 180s to 600s (10 minutes)
set_tests_properties(BatchCorrectnessTest PROPERTIES TIMEOUT 600 
    LABELS "integration;batch;correctness")
```

**Rationale**: Tests load models multiple times which is extremely slow in Debug builds

### Build Directory Structure
- Debug builds: `build/` 
- Release builds: `build_release/`

## Next Actions

### Immediate (This Session)
1. ✅ Validate `BatchedAttentionStagesParity` passes (completed)
2. ⏳ Run `PrefillBatchVsSequential` with 600s timeout
3. ⏳ Run `DecodeBatchVsSequential` with 600s timeout

### Follow-up (Future Sessions)
1. **Optimization**: Refactor tests to reuse model weights instead of reloading
2. **Release Builds**: Run tests in Release mode for realistic performance
3. **Extended Coverage**: Expand `BatchedAttentionStagesParity` to cover FFN/LM head stages
4. **Documentation**: Update `.github/instructions/parity-test-framework.instructions.md`

## Performance Notes

### Debug vs Release Impact
- **Debug Build**: 
  - Model loading: ~14s per load
  - Prefill (4-5 tokens): ~30-60s
  - Total test: 180s+ timeout
  
- **Release Build** (Expected):
  - 5-10x faster overall
  - Model loading: ~2-3s
  - Prefill: ~5-10s
  - Should complete all tests in <120s total

### Test Design Observations

**`BatchedAttentionStagesParity` (Good Design)**:
- ✅ Loads model once, reuses for all comparisons
- ✅ Fast: 72s for 726 snapshots across 8 stages
- ✅ Granular: Stage-by-stage validation identifies exact divergence point

**`PrefillBatchVsSequential` / `DecodeBatchVsSequential` (Inefficient Design)**:
- ❌ Loads model multiple times (1 batch + N sequential = 3 total for batch_size=2)
- ❌ Slow: 180s+ timeout in Debug
- ❌ Coarse: Only validates final logits, not intermediate stages

**Recommendation**: Prioritize expanding `BatchedAttentionStagesParity` coverage over fixing slow end-to-end tests.

## Test Execution Commands

```bash
# All batch correctness tests (with increased timeout)
ctest --test-dir build --output-on-failure --timeout 600 -R "BatchCorrectness"

# Individual tests
mpirun -np 2 --oversubscribe ./build/test_batch_correctness \
  --gtest_filter="BatchCorrectnessTest.BatchedAttentionStagesParity"  # ✅ Fast, passing

mpirun -np 2 --oversubscribe ./build/test_batch_correctness \
  --gtest_filter="BatchCorrectnessTest.PrefillBatchVsSequential"     # ⏱️ Slow, timeout

mpirun -np 2 --oversubscribe ./build/test_batch_correctness \
  --gtest_filter="BatchCorrectnessTest.DecodeBatchVsSequential"      # ⏳ Not tested
```

## Files Modified

1. **`.vscode/tasks.json`**: Separated Debug (`build/`) and Release (`build_release/`) directories
2. **`run_performance_bench.sh`**: Updated to use `build_release/`
3. **`CMakeLists.txt`**: 
   - Increased `BatchCorrectnessTest` timeout: 180s → 600s
   - Disabled `test_batch_sequential_parity` (lines 1310-1316 commented out)
4. **`tests/test_batch_sequential_parity.cpp`**: Fixed include `MPIContext.h` → `MpiContext.h` (but test remains disabled)

## Conclusion

**Batch parity testing infrastructure is functional** with one robust test (`BatchedAttentionStagesParity`) validating attention mechanism correctness. Two additional end-to-end tests exist but are slow due to repeated model loading. The broken `test_batch_sequential_parity.cpp` has been disabled in favor of the working suite.

**Primary recommendation**: Focus on extending `BatchedAttentionStagesParity` to cover FFN and LM head stages rather than optimizing the slow end-to-end tests. This approach provides better granularity for debugging and faster execution.
