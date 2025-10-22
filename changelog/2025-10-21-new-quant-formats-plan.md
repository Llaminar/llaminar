# New Quantization Formats Implementation Plan

**Date:** October 21, 2025  
**Status:** 🔄 In Progress - Q4_1 complete, Q2_K/Q3_K/Q5_K pending  
**Author:** David Sanftenberg

---

## Overview

Adding four new quantization formats to Llaminar's tensor system:
- ✅ **Q4_1**: 4-bit with scale + min (COMPLETE)
- 🔄 **Q2_K**: 2-bit K-quant with hierarchical scales
- 🔄 **Q3_K**: 3-bit K-quant  
- 🔄 **Q5_K**: 5-bit K-quant

All formats use the SIMDHelpers library for consistent, maintainable SIMD code.

---

## Q4_1 Tensor ✅ COMPLETE

**File:** `src/tensors/Q4_1Tensor.h` (455 lines)

**Block Structure:**
```cpp
struct Q4_1Block {
    uint16_t scale; // FP16 delta
    uint16_t min;   // FP16 minimum value
    uint8_t qs[16]; // 16 bytes (32 nibbles)
};
// Total: 20 bytes per block (32 elements)
```

**Decoding Formula:**
```
value[i] = scale * quant[i] + min
```

**Key Differences from Q4_0:**
- Q4_0: `value = scale * (quant - 8)`  → symmetric around 0
- Q4_1: `value = scale * quant + min` → asymmetric, better range

**Implementation Highlights:**
- AVX-512 path: Uses `simd::unpack_nibbles_convert_f32_first16_avx512()`
- AVX2 path: Uses `simd::extract_nibbles_scalar()` + `simd::convert_i8_to_f32_scaled_biased_avx2()`
- Formula adaptation: `(nibble * scale) + min` = `scale * (nibble - (-min/scale))`
- Scalar path: Direct `scale * quant + min` computation

**Tests Needed:**
```bash
tests/test_q4_1_tensor.cpp
- BasicConstruction
- DecodeRowZeros
- DecodeRowKnownPattern
- DecodeRowMultipleBlocks
- DecodeSpan
- DecodeRowToBF16
- OutOfBoundsAccess
- InvalidSizeMismatch
```

---

## Q2_K Tensor 🔄 PENDING

**File:** `src/tensors/Q2_KTensor.h` (to be created)

**Block Structure (from GGML):**
```cpp
struct Q2_KBlock {
    uint8_t scales[QK_K/16];  // 16 bytes - scales and mins (quantized with 4 bits)
    uint8_t qs[QK_K/4];       // 64 bytes - 2-bit quants (256 elements * 2 bits / 8)
    uint16_t d;               // FP16 - super-block scale for quantized scales
    uint16_t dmin;            // FP16 - super-block scale for quantized mins
};
// Total: 84 bytes per block (256 elements)
// QK_K = 256
```

**Decoding Formula:**
```
// 16 sub-blocks of 16 elements each
sub_block_idx = elem_idx / 16
scales_and_mins = scales[sub_block_idx]  // 4-bit scale + 4-bit min in same byte

scale_q = scales_and_mins & 0x0F
min_q = scales_and_mins >> 4

scale = d * scale_q
min = dmin * min_q

// Extract 2-bit quant
byte_idx = elem_idx / 4
bit_offset = (elem_idx % 4) * 2
quant_2bit = (qs[byte_idx] >> bit_offset) & 0x03

value = scale * quant_2bit + min
```

**Implementation Strategy:**

1. **Helper Function Needed in SIMDHelpers.h:**
```cpp
// Extract 2-bit values from packed bytes
void extract_2bit_values(const uint8_t* packed, size_t start_idx, int count, int8_t* output);
```

2. **AVX-512 Path:**
- Process 16 elements at a time (one scale/min pair)
- Extract scales and mins from packed byte
- Batch extract 16 2-bit values
- Convert to FP32 with scaling and bias

3. **AVX2 Path:**
- Process 8 elements at a time
- Similar to AVX-512 but half the width

4. **Scalar Path:**
- Per-element extraction
- Use helper for 2-bit extraction

**Challenges:**
- **Hierarchical scales**: Need to dequantize scales first
- **Packed mins**: 4-bit scale + 4-bit min in same byte
- **2-bit extraction**: Need new helper (similar to nibble extraction)

---

## Q3_K Tensor 🔄 PENDING

**File:** `src/tensors/Q3_KTensor.h` (to be created)

**Block Structure (from GGML):**
```cpp
struct Q3_KBlock {
    uint8_t hmask[QK_K/8];  // 32 bytes - high bit mask (1 bit per element)
    uint8_t qs[QK_K/4];     // 64 bytes - low 2 bits (2 bits per element)
    uint8_t scales[12];     // 12 bytes - scales quantized with 6 bits
    uint16_t d;             // FP16 - super-block scale
};
// Total: 110 bytes per block (256 elements)
```

**Decoding Formula:**
```
// 16 sub-blocks of 16 elements each
sub_block_idx = elem_idx / 16
scale_idx = sub_block_idx  // One scale per 16 elements (but packed 6-bit)

// Extract 6-bit scale (12 scales stored in 9 bytes)
// Scale extraction logic is complex - see GGML implementation
scale_6bit = extract_6bit_scale(scales, scale_idx)
scale = d * scale_6bit

// Extract 3-bit value: 2 low bits + 1 high bit
low_2bits = (qs[elem_idx / 4] >> ((elem_idx % 4) * 2)) & 0x03
high_bit = (hmask[elem_idx / 8] >> (elem_idx % 8)) & 0x01
quant_3bit = low_2bits | (high_bit << 2)

value = scale * (quant_3bit - 4)  // Offset by 4 for symmetric range
```

**Implementation Strategy:**

1. **Helper Functions Needed in SIMDHelpers.h:**
```cpp
// Extract 6-bit scales from packed bytes (12 scales in 9 bytes)
void extract_6bit_scales(const uint8_t* packed, int8_t* output_12_scales);

// Extract 3-bit values (2 low bits from qs + 1 high bit from hmask)
void extract_3bit_values_q3k(const uint8_t* qs, const uint8_t* hmask, 
                             size_t start_idx, int count, int8_t* output);
```

2. **AVX-512 Path:**
- Process 16 elements (one scale)
- Extract corresponding 6-bit scale
- Batch extract 16 3-bit values
- Convert with scale and bias (-4)

3. **Challenges:**
- **6-bit scale packing**: 12 scales in 9 bytes (complex bit manipulation)
- **3-bit construction**: Combine 2 bits from `qs` + 1 bit from `hmask`
- **Scale indexing**: 16 sub-blocks but only 12 scales (pattern repeats)

---

## Q5_K Tensor 🔄 PENDING

**File:** `src/tensors/Q5_KTensor.h` (to be created)

**Block Structure (from GGML):**
```cpp
struct Q5_KBlock {
    uint16_t d;                    // FP16 - super-block scale for quantized scales
    uint16_t dmin;                 // FP16 - super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE];  // 12 bytes - scales and mins (quantized with 6 bits)
    uint8_t qh[QK_K/8];            // 32 bytes - quants high bit
    uint8_t qs[QK_K/2];            // 128 bytes - quants low 4 bits
};
// Total: 176 bytes per block (256 elements)
// K_SCALE_SIZE = 12
```

**Decoding Formula:**
```
// 8 sub-blocks of 32 elements each
sub_block_idx = elem_idx / 32

// Extract scale and min from packed 6-bit values (complex packing)
scale_6bit = extract_scale_from_packed(scales, sub_block_idx)
min_6bit = extract_min_from_packed(scales, sub_block_idx)

scale = d * scale_6bit
min = dmin * min_6bit

// Extract 5-bit value: 4 low bits from qs + 1 high bit from qh
low_4bits = extract_nibble(qs[elem_idx / 2], elem_idx % 2)
high_bit = (qh[elem_idx / 8] >> (elem_idx % 8)) & 0x01
quant_5bit = low_4bits | (high_bit << 4)

value = scale * quant_5bit + min
```

**Implementation Strategy:**

1. **Helper Functions Needed in SIMDHelpers.h:**
```cpp
// Extract scales and mins from 6-bit packed format (Q4_K/Q5_K)
void extract_q5k_scales_mins(const uint8_t* scales_packed, int sub_block_idx, 
                              float d, float dmin, float* scale_out, float* min_out);

// Extract 5-bit values (4 bits from qs + 1 bit from qh)
void extract_5bit_values_q5k(const uint8_t* qs, const uint8_t* qh,
                             size_t start_idx, int count, int8_t* output);
```

2. **AVX-512 Path:**
- Process 16 or 32 elements
- Extract corresponding scale/min pair
- Batch extract 5-bit values
- Convert with scale and bias

3. **Similarity to Q6_K:**
- Both use similar bit extraction (ql + qh pattern)
- Q5_K: 4 low bits + 1 high bit
- Q6_K: 4 low bits + 2 high bits
- Can share extraction logic with minor modifications

**Challenges:**
- **6-bit scale/min packing**: Similar to Q3_K but with mins
- **5-bit construction**: Like Q6_K but simpler (1 high bit vs 2)
- **8 sub-blocks**: Different granularity than Q3_K (16) or Q6_K (16)

---

## SIMDHelpers.h Extensions Required

**New Functions to Add:**

```cpp
namespace llaminar::simd {

// ===== 2-bit Extraction (Q2_K) =====
void extract_2bit_scalar(uint8_t byte_val, int8_t* output_4_values);
void extract_2bit_values(const uint8_t* packed, size_t start_idx, int count, int8_t* output);

// ===== 3-bit Extraction (Q3_K) =====
void extract_6bit_scales(const uint8_t* packed, int8_t* output_12_scales);
void extract_3bit_values_q3k(const uint8_t* qs, const uint8_t* hmask, 
                             size_t start_idx, int count, int8_t* output);

// ===== 5-bit Extraction (Q5_K) =====
void extract_5bit_values_q5k(const uint8_t* qs, const uint8_t* qh,
                             size_t start_idx, int count, int8_t* output);

// ===== Hierarchical Scale/Min Extraction =====
void extract_q2k_scale_min(const uint8_t scales_byte, float d, float dmin,
                           float* scale_out, float* min_out);

void extract_q5k_scales_mins(const uint8_t* scales_packed, int sub_block_idx,
                              float d, float dmin, float* scale_out, float* min_out);

} // namespace llaminar::simd
```

---

## Testing Strategy

For each new tensor type, create comprehensive test suite:

**Test Files:**
- `tests/test_q4_1_tensor.cpp` ✅ (to be created)
- `tests/test_q2_k_tensor.cpp` 🔄
- `tests/test_q3_k_tensor.cpp` 🔄
- `tests/test_q5_k_tensor.cpp` 🔄

**Test Cases (per format):**
1. BasicConstruction - validate block size, shape
2. DecodeRowZeros - all zeros should decode correctly
3. DecodeRowKnownPattern - test with known quantized values
4. DecodeRowMultipleBlocks - test cross-block boundaries
5. DecodeSpan - test arbitrary element ranges
6. DecodeRowToBF16 - test BF16 decode path
7. MultipleRows - test multiple row decode
8. OutOfBoundsAccess - test error handling
9. InvalidSizeMismatch - test size validation
10. **ParityWithGGML** - compare against llama.cpp/GGML dequantization

**Parity Testing Approach:**
```cpp
// Load same GGUF model weights in both Llaminar and llama.cpp
// Compare decoded outputs element-wise
void test_parity_with_ggml(const std::string& gguf_file) {
    auto llaminar_tensor = load_q2k_tensor(gguf_file);
    auto ggml_reference = ggml_dequantize_q2k(gguf_file);
    
    for (int row = 0; row < rows; row++) {
        llaminar_tensor->decodeRow(row, llaminar_buf);
        compare_with_tolerance(llaminar_buf, ggml_reference + row*cols, cols, 1e-5);
    }
}
```

---

## Implementation Order

**Phase 1: Complete Q4_1** ✅
- [x] Q4_1Tensor.h implementation
- [x] QuantType enum updated
- [ ] test_q4_1_tensor.cpp
- [ ] Build and verify (8/8 tests passing)

**Phase 2: Add Q5_K** (Easiest K-quant - similar to Q6_K)
- [ ] Add 5-bit extraction helpers to SIMDHelpers.h
- [ ] Q5_KTensor.h implementation
- [ ] test_q5_k_tensor.cpp
- [ ] Build and verify (9/9 tests passing)

**Phase 3: Add Q3_K** (Medium complexity - 6-bit scales)
- [ ] Add 6-bit scale extraction to SIMDHelpers.h
- [ ] Add 3-bit value extraction to SIMDHelpers.h
- [ ] Q3_KTensor.h implementation
- [ ] test_q3_k_tensor.cpp
- [ ] Build and verify (9/9 tests passing)

**Phase 4: Add Q2_K** (Most complex - hierarchical scales + mins)
- [ ] Add 2-bit extraction helpers to SIMDHelpers.h
- [ ] Add hierarchical scale/min extraction
- [ ] Q2_KTensor.h implementation
- [ ] test_q2_k_tensor.cpp
- [ ] Build and verify (9/9 tests passing)

**Phase 5: Integration and Documentation**
- [ ] Update TensorFactory to recognize new formats
- [ ] Update ModelLoader to load new formats from GGUF
- [ ] Update documentation with new formats
- [ ] Binary verification (AVX-512/AVX2 instructions)
- [ ] Performance benchmarking
- [ ] Create changelog entry

---

## Estimated Effort

| Task | Lines of Code | Time Estimate |
|------|---------------|---------------|
| Q4_1Tensor.h | 455 | ✅ Complete |
| test_q4_1_tensor.cpp | ~250 | 1 hour |
| Q5_KTensor.h | ~550 | 2 hours |
| test_q5_k_tensor.cpp | ~300 | 1 hour |
| Q3_KTensor.h | ~580 | 2.5 hours |
| test_q3_k_tensor.cpp | ~300 | 1 hour |
| Q2_KTensor.h | ~600 | 3 hours |
| test_q2_k_tensor.cpp | ~300 | 1 hour |
| SIMDHelpers extensions | ~300 | 2 hours |
| Integration & testing | N/A | 2 hours |
| **TOTAL** | **~3635 lines** | **~15.5 hours** |

---

## Current Status

✅ **Q4_1Tensor.h**: Complete (455 lines)
- AVX-512/AVX2/scalar paths implemented
- Uses SIMDHelpers for consistency
- Proper bounds checking
- BF16 decode support
- Span decode support

🔄 **Next Steps**:
1. Create test_q4_1_tensor.cpp
2. Build and verify Q4_1 (expect 8/8 passing)
3. Move to Q5_K (easiest K-quant)

---

## Notes

**Design Principles Maintained:**
- All formats use SIMDHelpers library
- Consistent error handling across formats
- Proper bounds checking in all code paths
- BF16 decode support
- Span decode for arbitrary ranges
- Block-aligned SIMD processing with scalar fallback

**Performance Expectations:**
- Q2_K: ~2.63 bits/weight (highest compression)
- Q3_K: ~3.44 bits/weight
- Q4_1: ~5.0 bits/weight (better range than Q4_0)
- Q5_K: ~5.50 bits/weight

**References:**
- GGML structures: `/workspaces/llaminar/llama.cpp/ggml/src/ggml-common.h`
- Existing tensor patterns: Q4_0Tensor.h, Q6_KTensor.h, Q8_0Tensor.h
- SIMD helpers: src/utils/SIMDHelpers.h

---

**Document Version:** 1.0  
**Last Updated:** October 21, 2025
