# Q2_K, Q3_K, Q4_1 Dequantization Bug Fixes

**Date**: October 29, 2025  
**Session**: Fixing dequantization bugs discovered during test validation

## Summary

Fixed critical dequantization bugs in Q4_1, Q2_K, and Q3_K tensor implementations. All three formats now achieve **0 mismatches** (bit-exact) equivalency with llama.cpp reference implementation.

## Test Results

### Before Fixes
- Q4_1: ❌ 67.7% failure (12815/18944 mismatches)
- Q2_K: ❌ 98.8% failure (3542/3584 mismatches)  
- Q3_K: ❌ 86.8% failure (3110/3584 mismatches)

### After Fixes
- Q4_1: ✅ 0 mismatches (perfect)
- Q2_K: ✅ 0 mismatches (perfect)
- Q3_K: ✅ 0 mismatches (perfect)

### Full Test Suite Status
**11 out of 12 formats passing:**
- ✅ Q8_0, IQ4_NL, Q4_0, Q4_1, Q5_0, Q5_1, Q6_K, Q2_K, Q3_K, Q4_K, Q5_K
- ⏸️ Q8_K (skipped - no model)

**Test Coverage**: 92% (11/12 implemented quantization formats validated)

## Bug Fixes

### 1. Q4_1 Dequantization (Q4_1Tensor.cpp)

**Root Cause**: Output layout mismatch  
**Symptom**: 67.7% mismatches with systematic sign errors

**Issue**:
- Our implementation: Sequential layout `[v0, v1, v2, v3, ...]`
- llama.cpp: Interleaved layout `[all low nibbles, all high nibbles]`

**Fix**:
```cpp
// BEFORE (wrong):
output[i * 2 + 0] = scale * v0 + min;
output[i * 2 + 1] = scale * v1 + min;

// AFTER (correct):
output[i] = scale * x0 + min;        // Low nibbles: [0-15]
output[i + 16] = scale * x1 + min;   // High nibbles: [16-31]
```

**Lines Changed**: Q4_1Tensor.cpp:131-151, 155-197

**Key Learning**: GGML formats use interleaved layouts for better cache performance.

### 2. Q2_K Dequantization (Q2_KTensor.cpp)

**Root Cause**: Incorrect layout and missing signed cast  
**Symptom**: 98.8% mismatches (nearly total failure)

**Issues**:
1. Processed as 16 sub-blocks instead of 2 chunks × 4 groups
2. Used unsigned `q2` instead of `(int8_t)` cast
3. Incorrect scale/min extraction from packed bytes

**Fix**:
```cpp
// Match llama.cpp structure:
// 2 chunks of 128 elements
// Each chunk: 4 groups of 32 elements (2×16)
// Each group shares a scale/min pair

for (int n = 0; n < 256; n += 128) {
    int shift = 0;
    for (int j = 0; j < 4; ++j) {
        uint8_t sc = block.scales[is++];
        float dl = d * (sc & 0xF);
        float ml = dmin * (sc >> 4);
        
        // First 16 elements
        for (int l = 0; l < 16; ++l) {
            *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;
        }
        
        // Second 16 elements
        sc = block.scales[is++];
        dl = d * (sc & 0xF);
        ml = dmin * (sc >> 4);
        for (int l = 0; l < 16; ++l) {
            *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3)) - ml;
        }
        
        shift += 2;
    }
    q += 32;
}
```

**Lines Changed**: Q2_KTensor.cpp:107-150

**Key Learning**: K-quant formats use hierarchical layouts (chunks → groups → elements) for better compression.

### 3. Q3_K Dequantization (Q3_KTensor.cpp)

**Root Cause**: Incorrect scale extraction and hmask handling  
**Symptom**: 86.8% mismatches

**Issues**:
1. Naive scale extraction (6 bits per scale from 12 bytes)
2. Missing scale bias subtraction (`-32`)
3. Wrong hmask interpretation (conditional subtraction)
4. Incorrect sub-block iteration

**Fix**:

#### Scale Extraction (Complex Bit Manipulation)
```cpp
// llama.cpp uses sophisticated bit packing:
const uint32_t kmask1 = 0x03030303;
const uint32_t kmask2 = 0x0f0f0f0f;

uint32_t aux[4];
memcpy(aux, block.scales, 12);
uint32_t tmp = aux[2];
aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

const int8_t *scales = (const int8_t *)aux;
```

#### Hmask Handling
```cpp
uint8_t m = 1;  // Mask shifts left each iteration

// Conditional bias based on hmask bit:
dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4))
//                                      ^^^^^^^^^^^^^^^^^
//                                      If bit set: no bias
//                                      If bit clear: -4 bias
```

**Lines Changed**: Q3_KTensor.cpp:107-155

**Key Learning**: Q3_K uses 3-bit encoding (2 bits + 1 high bit) with complex scale packing to achieve ~4.5x compression.

## Technical Insights

### Why These Bugs Existed

1. **Never Properly Tested**: Tests were hard-coded to skip
   - Q2_K/Q3_K: "mixed quantization (no pure tensors available)"
   - Q4_1: Used non-existent model (404 error)

2. **Implemented from Spec**: Based on documentation without empirical validation

3. **Complex Formats**: K-quant formats are significantly more complex than simple formats
   - Q4_0: Simple 4-bit with scale (✅ works)
   - Q2_K: Hierarchical chunks, groups, signed cast, packed scales (❌ was broken)

### Validation Methodology

Each fix followed this pattern:

1. **Read llama.cpp reference** (`ggml-quants.c`)
2. **Identify structural differences** (layout, loops, formulas)
3. **Rewrite to match reference** (line-by-line comparison)
4. **Test and verify** (must achieve 0 mismatches)

**Success Criteria**: 0 mismatches = bit-exact equivalency with llama.cpp

### Code Quality Improvements

All three implementations now:
- ✅ Match llama.cpp structure exactly
- ✅ Use correct layouts (interleaved, hierarchical chunks)
- ✅ Apply proper casts (`(int8_t)` for signed quantized values)
- ✅ Handle packed data correctly (scales, hmasks)
- ✅ Include detailed comments explaining layout

## Files Modified

### Q4_1Tensor.cpp
- **Lines 131-151**: `decodeBlock()` - Fixed sequential → interleaved layout
- **Lines 155-197**: `decodeBlockAVX2()` - Fixed SIMD version to match

### Q2_KTensor.cpp  
- **Lines 107-150**: `decodeBlock()` - Rewrote with 2-chunk × 4-group structure
- **Key Changes**: 
  - Added signed cast: `(int8_t)((q[l] >> shift) & 3)`
  - Fixed scale/min unpacking from packed bytes
  - Matched llama.cpp iteration pattern

### Q3_KTensor.cpp
- **Lines 107-155**: `decodeBlock()` - Complete rewrite
- **Key Changes**:
  - Complex scale extraction with bit masks
  - Conditional hmask handling: `((hm[l] & m) ? 0 : 4)`
  - Scale bias subtraction: `(scales[is++] - 32)`
  - Shifting mask pattern: `m <<= 1`

## Impact

### Test Coverage
- **Before**: 67% coverage (8/12 formats validated)
- **After**: 92% coverage (11/12 formats validated)

### Remaining Work
- Q8_K: Only needs model download (implementation likely correct)
- IQ formats: 4 new tests ready to add (models downloaded)

### Confidence Level
With 11/12 formats showing **0 mismatches**, we have high confidence in:
- Test framework correctness
- llama.cpp equivalency
- Production readiness for validated formats

## Lessons Learned

### 1. Test-First Development
**Issue**: Implementations existed without proper tests  
**Learning**: Ground truth validation is mandatory for quantization formats  
**Action**: Never merge tensor implementations without passing equivalency tests

### 2. Reference Implementation Trumps Documentation
**Issue**: Implemented based on spec/docs, not actual code  
**Learning**: Always compare against reference implementation (llama.cpp)  
**Action**: Line-by-line code review against ggml-quants.c

### 3. Complex Formats Need Extra Scrutiny
**Issue**: K-quant formats are 10× more complex than simple formats  
**Learning**: Hierarchical layouts (chunks/groups) and packed data require careful validation  
**Action**: Add detailed comments explaining layout and bit manipulation

### 4. Interleaved Layouts Are Common
**Issue**: Assumed sequential output layout  
**Learning**: GGML formats often use interleaved layouts for cache efficiency  
**Pattern**: Check llama.cpp for actual output ordering

### 5. Signed vs Unsigned Casts Matter
**Issue**: Used unsigned values where signed required  
**Learning**: Pay attention to `(int8_t)` casts in reference code  
**Impact**: Wrong sign can cause 50%+ mismatches

## Next Steps

### Immediate
1. ✅ Q4_1 fixed (0 mismatches)
2. ✅ Q2_K fixed (0 mismatches)
3. ✅ Q3_K fixed (0 mismatches)

### Short-Term
4. Add IQ format equivalency tests:
   - IQ4_XS (model ready)
   - IQ3_XXS (model ready)
   - IQ3_S (model ready)
   - IQ2_S (model download needed)

5. Download Q8_K model and enable test

### Long-Term
6. Review all tensor implementations for similar patterns
7. Add equivalency tests for remaining IQ formats
8. Document quantization format characteristics in user guide

## Conclusion

All three buggy implementations are now **bit-exact** with llama.cpp reference, achieving 0 mismatches across ~26K elements tested. This brings Llaminar's quantization format validation to 92% coverage, with high confidence in production readiness for all 11 validated formats.

**Key Metric**: 11 out of 12 tensor formats validated with 0 mismatches = 92% test coverage

The fixes demonstrate the critical importance of:
- Ground truth validation against reference implementations
- Never skipping tests (technical debt reveals bugs)
- Line-by-line code comparison for complex formats
- Empirical testing over documentation-based implementation
