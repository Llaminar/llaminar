# Q8_1 GEMM Buffer Overflow Fixes

**Date**: January XX, 2025  
**Status**: ✅ RESOLVED  
**Test Coverage**: 109/109 tests passing with ASAN validation

## Summary

Fixed **two distinct buffer overflow bugs** in the Q8_1 GEMM microkernel that were causing crashes on large matrices (512×512×512 and larger). Both bugs were caught using AddressSanitizer (ASAN) and have been thoroughly tested with comprehensive regression tests.

---

## Bug #1: `sum_qs` AVX Gather Overread

### Problem

The `sum_qs` buffer (INT16 array storing quantization sums) was vulnerable to buffer overreads when using AVX gather instructions to load 32-bit integers from a 16-bit array.

### Root Cause

**Instruction**: `_mm_i32gather_epi32` loads 4-byte (int32) values from memory  
**Buffer**: `sum_qs` stores 2-byte (int16) values  

When gathering at index `ir=MR-1` (last row), the gather instruction reads:
- Valid: `sum_qs(kb, MR-1)` at index `kb*MR + (MR-1)`
- **Overread**: `sum_qs(kb, MR)` at index `kb*MR + MR` ← OUT OF BOUNDS

**Example** (MR=8, K_blocks=28):
- Buffer size: `K_blocks * MR = 28 * 8 = 224` elements (indices 0-223)
- Last valid access: `sum_qs(27, 7) =  sum_qs_storage[27*8 + 7] = 223` ✅
- Gather overread: `sum_qs(27, 8) = sum_qs_storage[27*8 + 8] = 224` ❌

### Solution

Added `+MR` padding to the `sum_qs` buffer allocation:

```cpp
// OLD (buggy):
std::vector<int16_t> sum_qs_vec(K_blocks * MR, 0);

// NEW (fixed):
// CRITICAL FIX: Add padding for gather instruction overread
// The int32 gather reads 4 bytes (2 int16 values), so when ir=MR-1 it reads sum_qs(kb, MR-1) and sum_qs(kb, MR)
// We need space for K_blocks*MR elements PLUS MR extra elements for safe overread
std::vector<int16_t> sum_qs_vec(K_blocks * MR + MR, 0);
```

**Location**: `src/v2/kernels/cpu/gemm_v2/Q8_1GemmKernel.h:999`

### Impact

- Affects all Q8_1 GEMM operations using MR ∈ {8, 16, 32}
- Manifested as SEGV crashes after ~20 kernel configurations in parameter sweeps
- Would crash intermittently depending on memory layout (red zone violations)

---

## Bug #2: `b_scales_f32` AVX-512 Vector Load Overread

### Problem

The `b_scales_f32` buffer (float array storing B matrix scales) was vulnerable to buffer overreads when using AVX-512 `_mm512_loadu_ps` to load 16 floats starting at index `jr*K_blocks + kb`.

### Root Cause

**Instruction**: `_mm512_loadu_ps` loads 16 consecutive floats (64 bytes)  
**Buffer**: `b_scales_f32` sized as `NR * K_blocks` floats  
**Layout**: `[NR][K_blocks]` → `b_scales_f32(jr, kb) = jr*K_blocks + kb`

The load reads 16 floats starting at index `jr*K_blocks + kb`, which spans:
- Start: `jr*K_blocks + kb`
- **End**: `jr*K_blocks + kb + 15`

**Example 1** (NR=128, K_blocks=16, jr=127, kb=0):
- Start: `127*16 + 0 = 2032` ✅
- End: `127*16 + 15 = 2047` ✅ (within buffer size 2048)

**Example 2** (NR=128, K_blocks=16, jr=127, kb=1):
- Start: `127*16 + 1 = 2033` ✅
- **End**: `127*16 + 16 = 2048` ❌ (buffer size is 2048, indices are 0-2047!)

### Solution

Added `+VECTOR_WIDTH` padding to the `b_scales_f32` buffer allocation:

```cpp
// OLD (buggy):
std::vector<float> b_scales_f32_vec(NR * K_blocks);

// NEW (fixed):
// CRITICAL FIX: Add padding for AVX-512 vectorized load overread
// The _mm512_loadu_ps loads 16 floats, so when jr=NR-1 and kb>0,
// it reads b_scales_f32(jr, kb) through b_scales_f32(jr, kb+15),
// which crosses row boundaries: jr*K_blocks+kb+15 can exceed NR*K_blocks-1
// Example: NR=128, K_blocks=16, jr=127, kb=1 → index 127*16+1+15=2048 (out of bounds for size 2048)
// We need space for NR*K_blocks elements PLUS VECTOR_WIDTH extra elements for safe overread
std::vector<float> b_scales_f32_vec(NR * K_blocks + VECTOR_WIDTH);
```

**Locations**: 
- `src/v2/kernels/cpu/gemm_v2/Q8_1GemmKernel.h:1030` (microkernel_full_sumqs)
- `src/v2/kernels/cpu/gemm_v2/Q8_1GemmKernel.h:1817` (microkernel_full_sa)

### Impact

- Affects all large matrix operations where NR=128, K_blocks≥16
- Triggered ASAN heap-buffer-overflow at address `0x52500000c140` (first byte after 8256-byte buffer)
- Would crash on `LargeMatrix_512x512x512` test

---

## Bug #3: JR_BATCH Out-of-Bounds Column Access

### Problem

When `NR % JR_BATCH != 0`, the last batch would access columns beyond `NR`, causing buffer overruns in the `b_scales_f32` buffer and output matrix `C`.

### Root Cause

The batched reduction loop assumed `JR_BATCH` columns were always available:

```cpp
for (int jr_base = 0; jr_base < NR; jr_base += JR_BATCH)
{
    for (int jj = 0; jj < JR_BATCH; ++jj)  // ❌ BUG: Can exceed NR!
    {
        int jr = jr_base + jj;
        // Access b_scales_f32(jr, kb) where jr >= NR
    }
}
```

**Example** (NR=128, JR_BATCH=18):
- jr_base iterations: 0, 18, 36, 54, 72, 90, 108, 126
- Last batch (jr_base=126):
  - jj=0: jr=126 ✅
  - jj=1: jr=127 ✅
  - jj=2: jr=128 ❌ (out of bounds!)
  - ...
  - jj=17: jr=143 ❌ (way out of bounds!)

This caused the buffer overread in Bug #2 to manifest even after adding padding.

### Solution

Calculate the actual batch size to prevent accessing beyond NR:

```cpp
for (int jr_base = 0; jr_base < NR; jr_base += JR_BATCH)
{
    // CRITICAL: Calculate actual batch size to avoid accessing jr >= NR
    const int actual_batch_size = std::min(JR_BATCH, NR - jr_base);
    
    for (int jj = 0; jj < actual_batch_size; ++jj)  // ✅ FIXED
    {
        int jr = jr_base + jj;
        // Now jr < NR always holds
    }
}
```

**Removed**: Separate tail handling loop (no longer needed with `actual_batch_size`)

**Location**: `src/v2/kernels/cpu/gemm_v2/Q8_1GemmKernel.h:1416-1638`

### Impact

- Affects all kernel configurations where `NR % JR_BATCH != 0`
- Critical for kernel `<32, 128, 4, 0, 0, 2, 18>` where NR=128, JR_BATCH=18
- Without this fix, padding alone was insufficient (buffer still overflowed)

---

## Testing

### Verification with ASAN

All fixes verified using AddressSanitizer with build configuration:
```bash
cmake -B build_v2_asan -S src/v2 \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g"
```

### Test Results

**Before fixes**: ASAN heap-buffer-overflow crash at `LargeMatrix_512x512x512`  
**After fixes**: ✅ 109/109 tests PASSING

Test breakdown:
- ✅ 46 parity tests (Q8_1GemmKernel_ParityTest)
- ✅ 18 buffer overflow regression tests (Q8_1GemmKernel_BufferOverflowTest)
- ✅ 45 other tests (correctness, edge cases, helpers)

### Regression Test Coverage

Added 18 comprehensive buffer overflow tests in `Q8_1GemmKernel_BufferOverflowTest` fixture:

**Sum_qs gather overread tests**:
- `SumQsOverread_MR8_SingleKBlock` - Single K-block edge case
- `SumQsOverread_MR8_MultipleKBlocks` - Original bug scenario (MR=8, K_blocks=28)
- `SumQsOverread_MR16_LargeK` - MR=16 variant
- `SumQsOverread_MR32_LargeK` - MR=32 variant

**Gather width tests**:
- `GatherOverread_AVX2_4wide` - 4-wide gather (int32)
- `GatherOverread_AVX2_8wide` - 8-wide gather (int32)
- `GatherOverread_AVX512_16wide` - 16-wide gather (int32)

**Edge case tests**:
- `EdgeCase_KBlocks27_OriginalBug` - Exact original crash scenario
- `EdgeCase_KBlocks1_Minimal` - Single K-block minimal case
- `EdgeCase_KBlocks63_LargePowerOf2Minus1` - Near power-of-2 boundary
- `EdgeCase_KBlocks64_ExactPowerOf2` - Exact power-of-2 boundary

**Stress tests**:
- `StressTest_KBlocks128` - Large K dimension (4096 elements)
- `StressTest_KBlocks256` - Extra-large K dimension (8192 elements)

**Rectangular matrix tests**:
- `Rectangular_TallNarrow` - M >> N (tests row iteration)
- `Rectangular_ShortWide` - N >> M (tests column iteration)
- `Rectangular_TinyMHugeK` - Small M, large K (tests gather at boundaries)

**Template parameter tests**:
- `Template_MR8_NR8_JRBatch2` - Small tile with JR_BATCH=2
- `Template_MR8_NR16_AsymmetricTile` - Asymmetric MR ≠ NR

All tests execute with ASAN validation - would SEGV with old buggy code.

---

## Lessons Learned

### 1. AVX Gather Instructions Can Overread

**Insight**: `_mm_i32gather_epi32` loads 32-bit values (4 bytes) but can be used on 16-bit arrays, causing inherent overreads of 2 bytes per access.

**Pattern**: Always allocate padding when using gather on smaller element types:
- `int32 gather` on `int16` array → add `+gather_width` padding
- `int64 gather` on `int32` array → add `+gather_width * 2` padding

### 2. AVX-512 Vector Loads Cross Row Boundaries

**Insight**: `_mm512_loadu_ps` loads 16 consecutive floats regardless of logical 2D array structure.

**Pattern**: For 2D buffers accessed as `[row][col]`, vectorized column loads can read into next row:
- Last valid column: `col_max - VECTOR_WIDTH + 1`
- Solution: Add `+VECTOR_WIDTH` padding to buffer size

### 3. Batch Loop Bounds Must Respect Actual Size

**Insight**: Template parameters like `JR_BATCH` describe ideal batch size, not guaranteed divisors.

**Pattern**: Always calculate `actual_batch_size = min(BATCH_SIZE, REMAINING)` when iterating in batches.

### 4. ASAN is Essential for Catching Buffer Overreads

**Insight**: These bugs would manifest as:
- Intermittent crashes (depending on memory layout)
- Silent corruption (overwriting adjacent allocations)
- Crashes far from the actual bug site

**Practice**: Always run comprehensive test suite with ASAN for low-level SIMD code.

---

## Related Documentation

- **Buffer Overflow Tests**: `tests/v2/unit/Test__Q8_1GemmKernel.cpp` (lines 3507-3745)
- **Q8_1 GEMM Kernel**: `src/v2/kernels/cpu/gemm_v2/Q8_1GemmKernel.h`
- **Previous Fix**: `changelog/2025-01-XX-q8-1-gemm-buffer-overflow-fix.md` (sum_qs only)

---

## Verification

To reproduce and verify the fixes:

```bash
# Build with ASAN
cmake -B build_v2_asan -S src/v2 -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g"
cmake --build build_v2_asan --target v2_test_q8_1_gemm_kernel --parallel

# Run all tests (should PASS with fixes, SEGV without)
cd build_v2_asan
ASAN_OPTIONS=halt_on_error=1:detect_leaks=0 \
  mpirun -np 1 ./tests/v2/v2_test_q8_1_gemm_kernel

# Run specific failing test (large matrix)
ASAN_OPTIONS=halt_on_error=1:detect_leaks=0 \
  mpirun -np 1 ./tests/v2/v2_test_q8_1_gemm_kernel \
  --gtest_filter='Test__Q8_1GemmKernel.LargeMatrix_512x512x512'

# Run buffer overflow regression tests
ASAN_OPTIONS=halt_on_error=1:detect_leaks=0 \
  mpirun -np 1 ./tests/v2/v2_test_q8_1_gemm_kernel \
  --gtest_filter='Q8_1GemmKernel_BufferOverflowTest.*'
```

**Expected**: All 109 tests pass cleanly with ASAN (no buffer overflows detected).

---

## Status

- ✅ **Bug #1 (sum_qs)**: FIXED - Added +MR padding
- ✅ **Bug #2 (b_scales_f32)**: FIXED - Added +VECTOR_WIDTH padding in both microkernels
- ✅ **Bug #3 (JR_BATCH bounds)**: FIXED - Implemented actual_batch_size clamping
- ✅ **Regression Tests**: 18 comprehensive tests added and passing
- ✅ **Full Test Suite**: 109/109 tests passing with ASAN validation
- ✅ **Documentation**: Complete with root cause analysis and examples

**Resolution**: All three buffer overflow bugs are fixed, tested, and documented. Code is production-ready.
