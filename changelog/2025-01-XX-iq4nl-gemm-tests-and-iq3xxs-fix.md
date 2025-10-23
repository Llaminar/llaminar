# IQ4_NL GEMM Unit Tests and IQ3_XXS Test Fix

**Date**: January 2025  
**Status**: ✅ Complete  
**Test Results**: 60/60 tests passing (48 model loader + 12 IQ4_NL)

## Summary

Successfully created comprehensive IQ4_NL GEMM unit tests and fixed a critical bug in the LoadIQ3_XXSTensor test that was blocking pre-commit hooks.

## Changes Made

### 1. Created IQ4_NL GEMM Test Suite

**File**: `tests/v2/unit/Test__IQ4_NLTensor.cpp` (489 lines)

**Test Coverage** (12 tests, 5ms runtime):
- ✅ `ConstructAndDecode` - Tensor construction and block decode verification
- ✅ `GEMMSingleElement` - 1×1 matrix multiplication
- ✅ `GEMMSmallBatch` - Small batch (4×4 matrices)
- ✅ `GEMMMediumBatch` - Medium batch (16×8×8 matrices)
- ✅ `GEMMLargeBatch` - Large batch (32×32×16 matrices)
- ✅ `GEMMWithAlpha` - Scaling factor verification (α=2.0)
- ✅ `GEMMWithBeta` - Accumulation verification (β=0.5)
- ✅ `GEMMWithAlphaAndBeta` - Combined scaling and accumulation
- ✅ `GEMMNonAlignedK` - Non-block-aligned K dimension (K=50)
- ✅ `GEMMNonAlignedM` - Non-block-aligned M dimension (M=17)
- ✅ `GEMMDimensionMismatch` - Error handling validation
- ✅ `GEMMNumericalStability` - Large value range stability ([-100, 100])

**Mathematical Verification**:
- Reference FP32 GEMM implementation for ground truth comparison
- Configurable tolerance (default 1e-4 relative, 1e-5 absolute)
- Matrix equality checking with detailed mismatch reporting

**Key Features**:
- Tests both GEMM interface methods: `multiply()` and `operator()`
- Verifies IBlockDecoder strategy pattern integration
- Validates QuantizedGemmKernel generic implementation
- Edge case coverage (non-aligned dimensions, error handling)

### 2. Fixed IQ4_NLTensor Class Definition

**File**: `src/v2/tensors/IQ4_NLTensor.h`

**Issues Fixed**:
1. **Access Modifier** (Line 404-409):
   - **Before**: `createGemm()` and `createGemmRaw()` were private
   - **After**: Moved to public section (required by TensorBase interface)

2. **Const Qualifier** (Line 1684):
   - **Before**: `std::unique_ptr<ITensorGemm> createGemm() const`
   - **After**: `std::unique_ptr<ITensorGemm> createGemm()` (match TensorBase)

3. **Duplicate Members** (Lines 1325-1326):
   - **Before**: Duplicate `private:` section with `tensor_` member
   - **After**: Removed duplicate section

**File**: `src/v2/tensors/IQ4_NLTensor.cpp`

**Changes**:
- **Line 109**: Removed `const` from `createGemm()` implementation

### 3. Fixed LoadIQ3_XXSTensor Test Bug

**File**: `tests/v2/unit/Test__ModelLoader.cpp` (Lines 1983-1989)

**Root Cause**: 
The test was writing a corrupted GGUF file due to incorrect file seek logic:

**Before** (buggy code):
```cpp
uint64_t name_len = 15;
file.write(reinterpret_cast<const char *>(&name_len), 8);  // Write 15
file.write("iq3_xxs_tensor", 14);                          // Write 14 bytes
file.seekp(-14, std::ios::cur);                            // Rewind 14 bytes
name_len = 14;
file.write(reinterpret_cast<const char *>(&name_len), 8);  // Overwrites string!
file.write("iq3_xxs_tensor", 14);                          // Write again
```

**Problem**:
- Wrote `name_len=15` at offset 0x40
- Wrote "iq3_xxs_tensor" (14 bytes) at offset 0x48
- Seeked back to 0x48 (only -14 bytes)
- Wrote `name_len=14` at 0x48, **overwriting the first 8 bytes of the string**
- Wrote "iq3_xxs_tensor" again at 0x50
- Result: File had both length values and corrupted data

**After** (fixed code):
```cpp
uint64_t name_len = 14;
file.write(reinterpret_cast<const char *>(&name_len), 8);  // Write correct length
file.write("iq3_xxs_tensor", 14);                          // Write string
```

**Impact**:
- **Error Message**: "Failed to read dimension 20 for: iq3_xxs"
- **Cause**: ModelLoader read corrupted file and tried to read 20+ dimensions
- **Status**: Test now passes (7ms) after fix

### 4. Build System Updates

**File**: `tests/v2/CMakeLists.txt`

**Added**:
```cmake
# IQ4_NL GEMM tests
add_executable(v2_test_iq4nl_tensor unit/Test__IQ4_NLTensor.cpp)
target_link_libraries(v2_test_iq4nl_tensor llaminar2_core gtest gtest_main)
add_test(NAME v2_test_iq4nl_tensor COMMAND v2_test_iq4nl_tensor)
```

## Debugging Session Highlights

### Issue 1: Segmentation Fault

**Symptom**: Test crashed when accessing `tensor->shape()`

**Investigation**:
```bash
$ gdb ./build_v2_coverage/tests/v2/v2_test_iq4nl_tensor
(gdb) run
# Crashed at tensor->shape()[0]

(gdb) bt
#9  0x... in Test__IQ4_NLTensor::ConstructAndDecode() at Test__IQ4_NLTensor.cpp:142

(gdb) p tensor->shape()
$1 = std::vector of length 21, capacity 21 = {garbage...}
```

**Root Cause**: Including `IQ4_NLTensor.h` (standalone class) instead of `Tensors.h` (TensorBase-derived)

**Fix**: Changed include from `"IQ4_NLTensor.h"` to `"Tensors.h"` (line 8)

### Issue 2: Build Errors

**Errors**:
1. Missing `QuantTypes.h` - Commented out (not needed)
2. `createGemm()` private - Moved to public
3. Duplicate `tensor_` member - Removed duplicate private section
4. Const mismatch - Removed `const` to match base class

### Issue 3: Pre-Commit Hook Failure

**Symptom**: `git commit` blocked by failing LoadIQ3_XXSTensor test

**Investigation**:
```bash
$ ./build_v2_coverage/tests/v2/v2_test_model_loader
# 47/48 tests passed, 1 failed: LoadIQ3_XXSTensor

$ od -A x -t x1z /tmp/test_iq3_xxs.gguf
000040  74 65 73 74 0f 00 00 00 00 00 00 00 0e 00 00 00
        ^^^^-------- "test"
             ^^^^^^^^^^^^^^^^^^^^^ name_len = 0x0f = 15
                                   ^^^^^^^^^^^^^^^^^^^^^ name_len = 0x0e = 14
```

**Root Cause**: File writing logic wrote both `name_len=15` and `name_len=14`, corrupting the file

**Fix**: Simplified to write correct length directly (no rewind/correction needed)

## Test Results

### Before Fix
```
v2_test_model_loader: 47/48 tests passing (1 FAILED)
  Failed: LoadIQ3_XXSTensor (9715ms timeout-like)
  Error: "Failed to read dimension 20 for: iq3_xxs"
```

### After Fix
```
v2_test_model_loader: 48/48 tests passing ✅
  LoadIQ3_XXSTensor: PASSED (7ms)

v2_test_iq4nl_tensor: 12/12 tests passing ✅
  All GEMM tests: PASSED (5ms total)
```

## Files Modified

1. `tests/v2/unit/Test__IQ4_NLTensor.cpp` (new, 489 lines)
2. `tests/v2/CMakeLists.txt` (added IQ4_NL test target)
3. `tests/v2/unit/Test__ModelLoader.cpp` (lines 1983-1989 - fixed IQ3_XXS)
4. `src/v2/tensors/IQ4_NLTensor.h` (moved methods to public, removed const)
5. `src/v2/tensors/IQ4_NLTensor.cpp` (removed const from createGemm)

## Verification

**Commit Hash**: 0912a6e  
**Pre-Commit Hook**: ✅ Passing  
**Total Tests**: 60 tests (48 model loader + 12 IQ4_NL)  
**Pass Rate**: 100%  
**Runtime**: ~600ms total

## Next Steps

**Optional Enhancements**:
1. Add `v2_test_iq4nl_tensor` to pre-commit hook (currently only runs model loader)
2. Create similar test suites for other quantized formats (Q6_K, Q8_0, etc.)
3. Add performance benchmarks to compare against reference implementation
4. Extend test coverage to other ITensor methods (RoPE, attention, etc.)

**Recommended**:
- Keep pre-commit hook focused on critical tests (model loader) to avoid slowdown
- Run comprehensive test suite in CI/CD pipeline
- Consider adding GEMM tests to nightly regression suite

## Lessons Learned

1. **Include Conflicts**: V2 has dual class definitions (standalone vs TensorBase-derived)
   - Always use `Tensors.h` for proper inheritance chain
   - Standalone headers (`IQ4_NLTensor.h`) are for implementation only

2. **File I/O Testing**: Seek operations are error-prone
   - Write correct values initially rather than rewind/correct
   - Use hexdump (`od -A x -t x1z`) to verify file structure

3. **GDB for Segfaults**: Backtrace + variable inspection quickly identified class definition conflict
   - `p tensor->shape()` revealed corrupted vector (21 elements instead of 2)

4. **Pre-Commit Hooks**: Essential for preventing broken commits
   - Caught the IQ3_XXS bug that would have broken CI
   - Fast tests (<1s) keep developer workflow smooth

## Conclusion

Successfully delivered comprehensive IQ4_NL GEMM testing infrastructure and fixed a critical pre-commit blocker. All tests now passing, code ready for production use.
