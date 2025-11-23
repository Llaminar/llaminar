# Integration Test Batch Handling Fixes

**Date**: 2025-01-24  
**Session Type**: Bug Investigation & Fix  
**Focus**: Integration test correctness for batch handling

## Summary

Successfully resolved all integration test failures in `V2_Integration_Qwen2Pipeline_BatchHandling` by identifying and fixing:
1. Attention mask dimension bug (wrong sequence length calculation)
2. Attention score buffer overflow (incorrect padding calculation)
3. Release build not having the fixes applied

## Problem Statement

After fixing 4 unit tests and moving the 5th test (`BatchHandling`) to the integration suite, the test was failing with NaN values appearing in `logits_1` (second sequence in batch) while `logits_0` was clean.

### Initial Error Pattern
```cpp
// Test: BufferAllocation_BatchSizing
// Error: NaN in logits_1[v] for vocab indices: 4, 5, 6, 7, 12, 13, 14, 15, ...
// Pattern: Only sequence 1 affected, sequence 0 was clean
```

## Root Cause Analysis

### Discovery Process

1. **Test Execution Path Investigation**
   - Found test running from `build_v2_release` directory despite being in debug workspace
   - Discovered CMakeLists.txt routing: Integration tests use Release build when CMAKE_BUILD_TYPE=Debug
   - Lines 131-133 in `tests/v2/CMakeLists.txt`:
     ```cmake
     if(CMAKE_BUILD_TYPE MATCHES "Debug" AND 
        ("${ARG_LABELS}" MATCHES "E2E" OR "${ARG_LABELS}" MATCHES "Integration"))
         set(USE_RELEASE_BUILD TRUE)
     endif()
     ```

2. **Build Status Check**
   - Release build (`build_v2_release`) did NOT have our recent fixes:
     - Attention mask dimension fix (lines 1099-1103)
     - Buffer overflow fix in attention scores allocation (lines 1105-1118)
   - Debug build (`build_v2`) had the fixes but wasn't being used for integration tests

3. **Root Cause**
   - Integration tests automatically route to Release build for performance
   - Release build was stale and had the old buffer overflow bug
   - This caused NaN values when accessing out-of-bounds memory during batch processing

## The Fixes

### 1. Attention Mask Dimensions (Previously Fixed in Unit Tests)
**File**: `src/v2/pipelines/qwen/Qwen2Pipeline.cpp`  
**Lines**: 1099-1103

```cpp
// BEFORE (bug - wrong dimension)
const int mask_col_size = effective_seq_len;  // Wrong: uses padded total
const int mask_size = mask_row_size * mask_col_size;

// AFTER (correct - per sequence)
const int per_sequence_mask_cols = padded_seq_len_;  // Correct per-sequence size
const int mask_size = mask_row_size * per_sequence_mask_cols;
```

**Impact**: Mask was using wrong dimensions for batched sequences, causing incorrect attention masking.

### 2. Attention Score Buffer Overflow (Previously Fixed in Unit Tests)
**File**: `src/v2/pipelines/qwen/Qwen2Pipeline.cpp`  
**Lines**: 1105-1118

```cpp
// BEFORE (bug - buffer overflow)
const int score_size_per_head = effective_seq_len * effective_seq_len;  // Wrong!

// AFTER (correct - proper padding)
// For batched sequences, each sequence needs padded_seq_len_ space
// and we have batch_size_ sequences, so total is batch_size_ * padded_seq_len_ * padded_seq_len_
const int score_size_per_head = batch_size_ * padded_seq_len_ * padded_seq_len_;
const int total_score_size = n_heads_ * score_size_per_head * max_threads_;
```

**Impact**: Buffer was too small for batched processing, causing memory corruption and NaN values when writing attention scores.

### 3. Release Build Update (This Session)
**Action**: Rebuilt Release build with all fixes
```bash
cmake --build build_v2_release --target v2_test_qwen2_pipeline_batch_handling --parallel
```

**Impact**: Integration tests now run against fixed code, all tests pass.

## Test Results

### Before Fix (Release Build Without Fixes)
```
V2_Integration_Qwen2Pipeline_BatchHandling.BufferAllocation_BatchSizing
ERROR: NaN in logits_1[4] (value: nan)
ERROR: NaN in logits_1[5] (value: nan)
... [hundreds of NaN values]
FAILED
```

### After Fix (Release Build With All Fixes)
```
[==========] 5 tests from Test__Qwen2Pipeline_BatchHandling
[  PASSED  ] BufferAllocation_BatchSizing
[  PASSED  ] SequentialVsBatched_SingleToken
[  PASSED  ] GetLogits_SequenceIndexing
[  PASSED  ] ResidualPaddingHygiene
[  PASSED  ] SequentialVsBatch_MultiToken
[==========] 5 tests (16.68 sec total)
100% tests passed
```

### Complete Test Suite Status
- **Unit Tests**: 102/102 passing ✅
- **Integration Tests**: 5/5 passing ✅
- **Total**: 107/107 tests passing ✅

## Technical Insights

### CMake Build Routing
Integration and E2E tests are intentionally routed to Release builds for performance:

```cmake
# tests/v2/CMakeLists.txt, lines 137-154
if(USE_RELEASE_BUILD)
    set(RELEASE_BUILD_DIR "${CMAKE_SOURCE_DIR}/../../build_v2_release")
    set(EXECUTABLE_PATH "${RELEASE_BUILD_DIR}/tests/v2/${TARGET_NAME}")
    # ... sets up MPI environment and execution ...
endif()
```

**Rationale**: Integration tests load full models and run complete inference, which is 5-10x slower in Debug mode.

**Learning**: When fixing bugs affecting integration tests, **always rebuild both Debug and Release builds**.

### Batch Processing Architecture

**Key Dimensions**:
- `batch_size_`: Number of sequences in batch (e.g., 2)
- `padded_seq_len_`: Max sequence length in batch (e.g., 2 tokens)
- `effective_seq_len`: Total tokens across all sequences (e.g., 4 = 2 × 2)

**Correct Buffer Sizing** (batched):
```cpp
// Attention scores: [n_heads, batch_size, padded_seq_len, padded_seq_len]
score_size = n_heads * batch_size * padded_seq_len * padded_seq_len * max_threads

// Attention mask: [batch_size, 1, padded_seq_len, padded_seq_len]
mask_size = batch_size * 1 * padded_seq_len * padded_seq_len
```

### Debug Logging Pattern

Added comprehensive batch debug logging at line 364-368:
```cpp
LOG_ERROR("[BATCH DEBUG] batch_size_=" << batch_size_ 
          << ", padded_seq_len_=" << padded_seq_len_ 
          << ", effective_seq_len=" << effective_seq_len);
for (int i = 0; i < batch_size_; ++i) {
    LOG_ERROR("[BATCH DEBUG]   Sequence " << i << ": " << seq_lengths[i] << " tokens");
}
```

**Usage**: Set `LLAMINAR_LOG_LEVEL=ERROR` to see batch structure without overwhelming output.

## Files Modified

### Tests
- `tests/v2/integration/Test__Qwen2Pipeline_BatchHandling.cpp` (moved from unit/)
  - 5 comprehensive batch handling tests
  - Tests buffer allocation, sequential vs batched parity, sequence indexing, residual padding

### Core Pipeline
- `src/v2/pipelines/qwen/Qwen2Pipeline.cpp`
  - Lines 1099-1103: Attention mask dimension fix
  - Lines 1105-1118: Attention score buffer overflow fix
  - Lines 364-368: Batch debug logging

### Build System
- Rebuilt `build_v2_release` with all fixes

## Performance Notes

Integration tests with Release build:
- **BufferAllocation_BatchSizing**: 4.0 sec
- **SequentialVsBatched_SingleToken**: 4.4 sec
- **GetLogits_SequenceIndexing**: 3.6 sec
- **ResidualPaddingHygiene**: 3.6 sec
- **Total Suite**: 16.7 sec

Debug build would be ~5-10x slower (80+ seconds for same tests).

## Lessons Learned

1. **Always Check Both Builds**: When fixing bugs, ensure both Debug and Release builds are updated
2. **Test Routing Matters**: Understanding CMake test routing is critical for debugging failures
3. **Buffer Sizing in Batched Code**: Batch dimensions multiply buffer requirements significantly
4. **Sequence Padding Hygiene**: Each sequence needs its own padded space, not shared
5. **Per-Sequence Calculations**: Masks and scores are per-sequence, not global

## Next Steps

1. ✅ All unit tests passing (102/102)
2. ✅ All integration tests passing (5/5)
3. **Consider**: Add E2E tests for ground truth validation
4. **Consider**: Add performance benchmarks for batch processing
5. **Consider**: Test with larger batch sizes (4, 8, 16, 32)

## Verification Commands

```bash
# Run all unit tests
cd /workspaces/llaminar/build_v2
ctest -R "^V2_Unit_" --output-on-failure --parallel

# Run integration tests (uses Release build)
cd /workspaces/llaminar/build_v2
ctest -R "^V2_Integration_" --output-on-failure

# Rebuild Release build with fixes
cmake --build build_v2_release --parallel

# Check specific batch handling test
cd /workspaces/llaminar/build_v2
ctest -R "V2_Integration_Qwen2Pipeline_BatchHandling" --verbose
```

## Related Issues

- Mask dimension calculation for batched sequences
- Buffer overflow in attention score allocation
- CMake test routing for Integration/E2E tests
- Release build staleness vs Debug build

## Contributors

- David Sanftenberg

---

**Status**: ✅ RESOLVED  
**Tests Passing**: 107/107 (100%)  
**Build Status**: Both Debug and Release builds synchronized
