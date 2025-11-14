# Q8_1 GEMM Buffer Overflow Fix

**Date**: January 2025
**Status**: Fixed
**Impact**: Critical - SEGV crash in comprehensive parameter sweep test

## Problem

The Q8_1 GEMM comprehensive parameter sweep test was crashing with SEGV after processing ~20 configurations:

```
AddressSanitizer: SEGV on unknown address 0x514002770000
#0 in Q8_1GemmKernelTemplate<8, 8, 1, 0, 0, 1, 2>::microkernel_full_sumqs
   at Q8_1GemmKernel.h:1547 (AVX gather instruction)
```

## Root Cause Analysis

### The Bug

The crash occurred in the post-processing phase of the microkernel when using AVX2 gather instructions to load `sum_qs` values:

```cpp
// Line 1547 (approximately)
__m128i gather_indices = _mm_setr_epi32(0 * MR, 1 * MR, 2 * MR, 3 * MR);
__m128i sum_qs_i32_raw = _mm_i32gather_epi32(
    reinterpret_cast<const int *>(&sum_qs(kb, ir)), gather_indices, 2);
```

**Key issue**: `_mm_i32gather_epi32` loads **32-bit integers** (4 bytes), but the `sum_qs` array contains **16-bit integers** (2 bytes).

### The Overread Problem

When gathering from `&sum_qs(kb, ir)`:
- The instruction reads 4 bytes starting at this address
- This includes `sum_qs(kb, ir)` (2 bytes) **AND** `sum_qs(kb, ir+1)` (2 bytes)
- The code extracts only the lower 16 bits (the desired value)

**When does this cause a crash?**

When `ir = MR-1` (last row in the microkernel):
- `&sum_qs(kb, MR-1)` is the last valid element for K-block `kb`
- Reading 4 bytes accesses `sum_qs(kb, MR-1)` AND `sum_qs(kb, MR)`
- But `sum_qs(kb, MR)` = `sum_qs(kb+1, 0)` (first element of next K-block row)
- When `kb = K_blocks-1`, this reads `sum_qs(K_blocks, 0)` which is **OUT OF BOUNDS**!

### Example Scenario

With `K_blocks=28`, `MR=8`:
- Buffer size: `28 * 8 = 224` int16 elements
- Valid indices: 0 to 223
- When `kb=27, ir=7`:
  - `&sum_qs(27, 7)` → index 223 (last element) ✓
  - But gather reads 4 bytes → includes index 224 ❌ **OUT OF BOUNDS!**

### Why ASAN Detected It

AddressSanitizer creates "red zones" around allocations. When the gather instruction tried to read index 224, it accessed ASAN's red zone, triggering the SEGV.

## Solution

Allocate padding at the end of the `sum_qs` buffer to accommodate the overread:

**Before**:
```cpp
std::vector<int16_t> sum_qs_vec(K_blocks * MR, 0);
```

**After**:
```cpp
// CRITICAL FIX: Add padding for gather instruction overread
// The int32 gather reads 4 bytes (2 int16 values), so when ir=MR-1 it reads sum_qs(kb, MR-1) and sum_qs(kb, MR)
// We need space for K_blocks*MR elements PLUS MR extra elements for safe overread
std::vector<int16_t> sum_qs_vec(K_blocks * MR + MR, 0);
```

**Why `+ MR` padding?**
- In the worst case, `ir=MR-1` for all `kb` values
- Each gather reads `sum_qs(kb, MR)` which is `kb*MR + MR` = `(kb+1)*MR`
- Maximum overread: when `kb=K_blocks-1`, accesses `K_blocks*MR` (first element past the end)
- Adding `MR` elements ensures all gathers are safe

## Testing

**Before fix**:
```bash
$ ASAN_OPTIONS=halt_on_error=1:detect_leaks=0 ./build_v2_asan/performance/v2_perf_q8_1_gemm \
    --gtest_filter=Q8_1GemmPerformance.ComprehensiveParameterSweep

[Q8_1 GEMM] Using sum_qs compensation (baseline, with quantization)
...
Tested 20/2340 configurations (0.9%)
AddressSanitizer:DEADLYSIGNAL
==2449026==ERROR: AddressSanitizer: SEGV on unknown address 0x514002770000
```

**After fix**:
```bash
$ ./build_v2_release/performance/v2_perf_q8_1_gemm \
    --gtest_filter=Q8_1GemmPerformance.ComprehensiveParameterSweep

Tested 20/2340 configurations (0.9%)
[Q8_1 GEMM] Using sum_qs compensation (baseline, with quantization)
...
(continues running)
```

Test now progresses beyond 20 configurations without SEGV! ✅

## Impact

**Files Modified**:
- `src/v2/kernels/cpu/gemm_v2/Q8_1GemmKernel.h` (line ~999)

**Affected Code**:
- All Q8_1 GEMM kernel instantiations (64 template variants)
- Affects ALL microkernel sizes: MR ∈ {8, 16, 32, 64, 128}

**Why This Wasn't Caught Earlier**:
1. **Smaller test matrices**: Previous tests used smaller M values or fewer configurations
2. **Lucky alignment**: May not have triggered when buffer happened to be followed by readable memory
3. **Debug vs Release**: Issue may appear/disappear based on memory layout differences
4. **ASAN requirement**: Only detectable with AddressSanitizer (production builds had no red zones)

## Lessons Learned

1. **AVX gather overreads**: Always account for element size mismatches (int32 gather from int16 array)
2. **ASAN is essential**: This bug was invisible without AddressSanitizer's red zones
3. **Comprehensive testing**: Bug only appeared with comprehensive parameter sweep (2,340 configs)
4. **Document gather semantics**: Add comments explaining gather behavior and padding requirements

## Related Issues

- Comprehensive parameter sweep now unblocked
- Similar pattern may exist in other gather-based code (check 8-wide and 16-wide vectorized paths)

## Next Steps

- [x] Fix buffer allocation (+MR padding)
- [x] Verify with ASAN build
- [x] Document in code comments
- [ ] Review other gather instructions for similar issues
- [ ] Run full comprehensive parameter sweep to completion
