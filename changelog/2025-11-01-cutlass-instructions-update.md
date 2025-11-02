# CUTLASS Instructions Update: Predication and Bug Fixes

**Date:** November 1, 2025  
**Type:** Documentation  
**Status:** ✅ Complete

## Summary

Updated `.github/instructions/cutlass.instructions.md` with comprehensive coverage of recent learnings from CuTe kernel development, including:

1. **CuTe Predication Pattern** - Complete guide to NVIDIA-recommended bounds checking
2. **Critical Kernel Bugs and Fixes** - Documented 3 major bugs encountered and solved
3. **Quick Reference Templates** - Copy-paste templates for predication patterns

## New Sections Added

### 1. CuTe Predication Pattern (Bounds Checking)

**Lines:** 530-750 (~220 lines)

**Coverage:**
- **Overview**: Identity tensors, `elem_less()`, and why predication is needed
- **The Problem**: Detailed explanation of out-of-bounds writes (m=1 edge case)
- **The Solution**: Step-by-step NVIDIA pattern with code examples
- **Predication for A-tile**: Applying same pattern to input loading
- **When NOT to Use**: Decoder-specific cases where manual bounds checking is appropriate
- **Performance Impact**: Before/after results (671B: crashes → 188-9,477 GFLOPS)
- **Reference Documentation**: Links to NVIDIA CuTe docs

**Key Code Patterns Documented:**

```cpp
// Identity tensor setup
Tensor cC = make_identity_tensor(shape(mC));
Tensor cta_cC = local_tile(cC, cta_tiler, cta_coord, Step<_1, _1, X>{});
Tensor tCcC = thr_mma.partition_C(cta_cC);

// Predicated output write
CUTE_UNROLL
for (int i = 0; i < size(tCgC); ++i) {
    if (elem_less(tCcC(i), shape(mC))) {
        tCgC(i) = tCrC(i);
    }
}

// Predicated input load
auto coord = gA_k_coord(row, col);
if (elem_less(coord, shape(mA))) {
    val = gA_k(row, col);
}
```

### 2. Critical Kernel Bugs and Fixes

**Lines:** 752-890 (~140 lines)

**Bug 1: TILE_K < BLOCK_SIZE Division by Zero**
- **Symptom**: 235B FFN crashes on all 24 tile configs
- **Root Cause**: When TILE_K=16 < BLOCK_SIZE=32, `K_BLOCKS_PER_TILE = 0`
- **Fix**: Block-intersection algorithm handles any TILE_K size
- **Result**: 0/24 → 24/24 configs working, ~5,000 GFLOPS

**Bug 2: Missing cp.async Predication**
- **Symptom**: Assumed async copy handles bounds checking automatically
- **Reality**: `cp.async` requires explicit predication
- **Fix**: Apply identity tensor pattern to async copy source

**Bug 3: Inconsistent Bounds Checking Patterns**
- **Problem**: Mixed manual and CuTe-style checks (hard to maintain)
- **Fix**: Standardize on CuTe where appropriate, manual for decoder-specific
- **Before/After**: Clear comparison showing consistency improvement

**Testing Coverage:**
- m=1 single token: Most extreme edge case
- Large matrices: Non-divisible dimensions
- TILE_K < BLOCK_SIZE: Small tiles with large blocks
- **Validation**: 22/22 tests passing (100% success)

### 3. Updated Quick Reference

**Enhanced predication section:**
```cpp
// Predication functions
make_identity_tensor(shape)             // Create coordinate tensor
elem_less(coord, bound)                 // Check if coord < bound (elementwise)
```

**New quick template:**
- Complete predication setup pattern
- Predicated input load example
- Predicated output write example
- Ready to copy-paste into new kernels

## Updated Metadata

**Header section updated:**
```markdown
**Recent Updates** (November 2025):
- ✅ CuTe Predication Pattern: Complete guide with identity tensors
- ✅ Critical Bug Fixes: TILE_K < BLOCK_SIZE, k-alignment, bounds checking
- ✅ Comprehensive Testing: 22/22 model sizes validated (0.5B-671B)
- ✅ Production Ready: All edge cases handled (m=1, large matrices, small tiles)
```

**Table of Contents updated:**
- Added "CuTe Predication Pattern (Bounds Checking)" ⭐ **NEW**
- Added "Critical Kernel Bugs and Fixes" ⭐ **NEW**

## Documentation Quality

### Before This Update

**Predication coverage:**
- ❌ No mention of identity tensors
- ❌ No `elem_less()` examples
- ❌ No guidance on bounds checking
- ❌ No edge case discussion

**Bug fix documentation:**
- ❌ TILE_K < BLOCK_SIZE issue not documented
- ❌ K-alignment requirement unclear
- ❌ No testing strategy for edge cases

### After This Update

**Predication coverage:**
- ✅ Complete NVIDIA-recommended pattern
- ✅ Step-by-step code examples
- ✅ Clear explanation of why it works
- ✅ When to use vs. when not to use
- ✅ Performance impact data
- ✅ Reference to official NVIDIA docs

**Bug fix documentation:**
- ✅ All 3 critical bugs documented
- ✅ Symptoms, root causes, and fixes explained
- ✅ Before/after performance data
- ✅ Complete testing validation strategy
- ✅ Copy-paste ready code fixes

## Benefits

### For Future Development

1. **Faster onboarding**: New developers can learn predication pattern quickly
2. **Avoid pitfalls**: All major bugs documented with solutions
3. **Copy-paste templates**: Quick reference for common patterns
4. **Testing guidance**: Clear edge cases to validate

### For Debugging

1. **Bug symptom lookup**: Match error messages to documented bugs
2. **Root cause analysis**: Understand why bugs happen (TILE_K < BLOCK_SIZE, etc.)
3. **Validation tests**: Know which test cases to run (m=1, non-divisible, etc.)

### For Code Review

1. **Pattern consistency**: Easy to verify CuTe patterns are used correctly
2. **Edge case coverage**: Check all documented edge cases are tested
3. **Performance expectations**: Compare against documented baselines

## Related Files

**Implementation:**
- `src/v2/kernels/cuda/CudaGemmKernelTensorCoreCuTe.cuh` - Uses all documented patterns

**Tests:**
- `tests/v2/performance/Perf__TensorCoreHeuristicValidation.cpp` - Validates all edge cases

**Previous Changelogs:**
- `changelog/2025-11-01-671b-cute-predication-fix.md` - Original predication fix
- `changelog/2025-11-01-cute-predication-cleanup.md` - Extended to A-tile loading

## Statistics

**Documentation Growth:**
- **Before**: ~1,003 lines
- **After**: ~1,370 lines (+367 lines, +37% content)

**New Content:**
- Predication section: ~220 lines
- Bug fixes section: ~140 lines
- Quick reference updates: ~7 lines

**Coverage Improvement:**
- Edge cases: 0 → 3 documented scenarios
- Bug fixes: 0 → 3 critical issues covered
 2 (added predication template)

## Future Maintenance

**Keep updated:**
1. Add new bug fixes as they're discovered
2. Update performance baselines as optimizations improve
3. Add examples for new CuTe patterns (pipelining, etc.)
4. Link to new NVIDIA documentation as it's released

**Review frequency:** After each major kernel change or bug fix

---

**Session Notes:**
- Complete documentation of predication learnings from 671B bug fix session
- All critical bugs from 235B and 671B fixes documented
- Ready for production use and onboarding new developers
- No code changes - pure documentation update

**Total Time:** ~30 minutes (writing + review)
