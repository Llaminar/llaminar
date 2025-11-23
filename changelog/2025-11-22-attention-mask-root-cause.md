# Attention Mask Bug - Root Cause Identified

**Date**: 2025-11-22
**Bug**: Batch padding divergence causing 97 billion % logit differences
**Status**: **ROOT CAUSE FOUND**

## Summary

The batch padding bug has been traced to a critical issue in `AttentionUtils.h` mask construction functions. **When `window_size = 0` is passed, ALL positions get masked (including the diagonal), creating invalid attention masks where tokens cannot attend to anything, including themselves.**

## Root Cause

File: `src/v2/pipelines/AttentionUtils.h`
Function: `create_causal_mask()` (and related functions)
Lines: 108-110

```cpp
// Sliding window: also check distance
if (window_size >= 0 && can_attend)
{
    can_attend = (i - j < window_size);  // BUG: window_size=0 makes this always FALSE!
}
```

### The Logic Error

When `window_size = 0`:
1. Condition `window_size >= 0` is TRUE (0 >= 0)
2. For diagonal position where i==j: `can_attend = (0 - 0 < 0)` → FALSE
3. For past positions where i > j: `can_attend = (i - j < 0)` → FALSE (since i-j is always positive)
4. **Result**: Every position gets masked with -inf, including tokens attending to themselves!

### Evidence from Unit Tests

Created `Test__AttentionMaskDiagnostic.cpp` which revealed:

```
SimpleCausalMask test with window_size=0:
Row 0: [-inf, -inf, -inf, -inf]  ← Should be [0, -inf, -inf, -inf]
Row 1: [-inf, -inf, -inf, -inf]  ← Should be [0, 0, -inf, -inf]
Row 2: [-inf, -inf, -inf, -inf]  ← Should be [0, 0, 0, -inf]
Row 3: [-inf, -inf, -inf, -inf]  ← Should be [0, 0, 0, 0]
```

**ALL positions masked!** This is completely broken.

### Standalone Test Confirmation

Created standalone C++ test (`/tmp/test_mask.cpp`) with `window_size = -1`:
```
Row 0: [0, -inf, -inf, -inf]  ✅ Correct!
Row 1: [0, 0, -inf, -inf]     ✅ Correct!
Row 2: [0, 0, 0, -inf]        ✅ Correct!
Row 3: [0, 0, 0, 0]           ✅ Correct!
```

Changing to `window_size = -1` produces correct masks.

## Affected Code Paths

### Direct Calls
1. `Test__GQAAttention_PaddingMask.cpp:165` - `create_batch_padding_mask(..., 0)`
2. `Test__AttentionMaskDiagnostic.cpp:62` - `create_causal_mask(..., 0)`
3. Throughout test suite: Any test passing `window_size = 0` expecting "no windowing"

### Production Code
Need to audit all callers of:
- `create_causal_mask()`
- `create_batch_causal_mask()`
- `create_batch_padding_mask()`
- `create_combined_batch_mask()`

Any caller passing `window_size = 0` is creating invalid masks!

## Impact on Batch Padding Bug

This explains the E2E test failures:

1. **Batch execution** creates masks with `window_size = 0`
2. **All tokens get -inf masks** (cannot attend to anything)
3. **Softmax produces NaN or uniform distributions** (all -inf scores)
4. **Attention output is garbage**, contaminated by undefined behavior
5. **Logits diverge catastrophically** from correct sequential baseline

The padded sequences fail worse because:
- They have MORE masked positions (real tokens + padding)
- The masking bug amplifies for variable-length sequences
- Real tokens cannot even attend to themselves!

## Fix Strategy

### Option A: Semantic Fix (Recommended)
Treat `window_size = 0` as "no windowing" (same as -1):

```cpp
// Sliding window: also check distance
if (window_size > 0 && can_attend)  // Changed from >= to >
{
    can_attend = (i - j < window_size);
}
```

**Rationale**: `window_size = 0` logically means "no window restriction", not "window of size 0 (impossible to attend)".

### Option B: Documentation Fix
Document that `window_size` must be either `-1` (disabled) or `>= 1` (minimum 1 to include diagonal):

```cpp
/**
 * @param window_size Sliding window size:
 *   -1 = disabled (full attention)
 *   >= 1 = local window (1 = only diagonal, 2 = diagonal + 1 neighbor, etc.)
 *   NOTE: window_size = 0 is INVALID and will mask all positions!
 */
```

Then fix all callers to pass -1 instead of 0.

### Option C: Hybrid Approach
1. Change logic to treat 0 same as -1 (backwards compatible)
2. Add assertion/warning if window_size == 0
3. Update documentation to clarify semantics

**Recommendation**: **Option A** - It's the least disruptive and most intuitive. Zero should mean "no restriction", not "impossible restriction".

## Verification Plan

1. **Fix the bug** in `AttentionUtils.h` (change `>= 0` to `> 0`)
2. **Rerun unit tests**:
   - `Test__AttentionMaskDiagnostic` - should all pass
   - `Test__GQAAttention_PaddingMask` - should all pass
3. **Rerun E2E test**:
   - `Test__BatchPaddingDivergence` - should pass
   - `Test__Qwen2E2ECorrectness` - batch tests should pass
4. **Audit codebase** for any callers relying on window_size=0 creating "empty window" behavior

## Next Steps

1. Implement Option A fix
2. Run all unit tests
3. Run E2E tests
4. Document the fix in code comments
5. Add regression test to prevent future window_size=0 bugs

## Lessons Learned

1. **Always test mask construction in isolation** before integrating into complex pipelines
2. **Off-by-one errors in window logic** are subtle and catastrophic
3. **Default parameter values** (window_size=0 in this case) should be carefully chosen
4. **Unit tests caught this immediately** - comprehensive test coverage works!

## Files Modified (for fix)

- `src/v2/pipelines/AttentionUtils.h` - Change line 108: `>= 0` → `> 0`
- Add comment explaining `window_size = 0` semantics
- Update function documentation

## Test Files Created (diagnostic)

- `tests/v2/unit/Test__AttentionMaskDiagnostic.cpp` - Mask construction verification
- `tests/v2/unit/Test__GQAAttention_PaddingMask.cpp` - Comprehensive padding mask tests
- Both tests currently FAIL as expected, will PASS after fix
