# Tensor Decode Implementation Plan

**Date**: 2025-11-10  
**Status**: Accessor refactoring complete, tensor methods pending  
**Context**: Moved decode logic from accessor classes to tensor implementations for better encapsulation

## Completed Refactoring

All three accessor classes have been refactored to delegate decode operations to tensor methods:

### IQ4_NLCachedAccessor
```cpp
// BEFORE: Inline decode with lookup table
const Q8_0Block* get_q8_block(...) {
    return cache_.get_or_decode(key, [this, ...](Q8_0Block* output) {
        const IQ4_NLBlock* iq4 = tensor_->get_block(...);
        decodeIQ4NLBlock(*iq4, output);  // Static method with IQ4NL_LUT
    });
}

// AFTER: Delegate to tensor
const Q8_0Block* get_q8_block(...) {
    return cache_.get_or_decode(key, [this, ...](Q8_0Block* output) {
        tensor_->decode_to_q8_0(row_idx, k_block_offset, output);
    });
}
```

**Lines removed**: ~50 (IQ4NL_LUT table, decodeIQ4NLBlock method)

### Q6_KCachedAccessor
```cpp
// BEFORE: Inline 6-bit unpacking
const Q8_0Block* get_q8_block(...) {
    return cache_.get_or_decode(key, [this, ...](Q8_0Block* output) {
        const Q6_KBlock* q6k_block = tensor_->get_block(...);
        decodeQ6KSubBlock(*q6k_block, block_in_superblock, output);
    });
}

// AFTER: Delegate to tensor
const Q8_0Block* get_q8_block(...) {
    return cache_.get_or_decode(key, [this, ...](Q8_0Block* output) {
        tensor_->decode_to_q8_0(row_idx, k_block_offset, output);
    });
}
```

**Lines removed**: ~60 (decodeQ6KSubBlock method with hierarchical scale extraction)

### FP32CachedAccessor
```cpp
// BEFORE: Inline quantization
const Q8_0Block* get_q8_block(...) {
    return cache_.get_or_decode(key, [this, ...](Q8_0Block* output) {
        const float* row_data = tensor_->data() + row_idx * k_elements_;
        quantizeFP32ToQ8Block(row_data + k_start, k_count, output);
    });
}

// AFTER: Delegate to tensor
const Q8_0Block* get_q8_block(...) {
    return cache_.get_or_decode(key, [this, ...](Q8_0Block* output) {
        tensor_->decode_to_q8_0(row_idx, k_block_offset, output);
    });
}
```

**Lines removed**: ~40 (quantizeFP32ToQ8Block method with absmax + symmetric quantization)

**Total accessor simplification**: ~150 lines removed, cleaner separation of concerns

---

## Required Tensor Method Interface

All tensor classes must implement:

```cpp
/**
 * @brief Decode/quantize tensor block to Q8_0 format
 * 
 * @param row_idx Row index in the tensor
 * @param k_block_offset Block offset along K dimension (32 elements per block)
 * @param output Output Q8_0 block (caller-provided storage)
 * 
 * Implementation Details:
 * - For quantized formats: Decode from native format to Q8_0
 * - For FP32/FP16: Quantize using symmetric quantization
 * - Handle edge cases: partial blocks, zero values, extreme scales
 * - Must be thread-safe (called from OpenMP parallel regions)
 */
void decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block* output) const;
```

---

## Implementation Checklist

### Priority 1: Core Formats (Immediate)

#### 1. IQ4_NLTensor::decode_to_q8_0()
**File**: `src/v2/tensors/IQ4_NLTensor.cpp`

**Implementation**:
```cpp
void IQ4_NLTensor::decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block* output) const
{
    // IQ4_NL: 4-bit lookup-based quantization (32 elements per block)
    static constexpr int8_t IQ4NL_LUT[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
        1, 13, 25, 38, 53, 69, 89, 113
    };

    const IQ4_NLBlock* iq4_block = get_block(row_idx, k_block_offset);
    const float scale = fp16_to_fp32(iq4_block->d);

    // Decode 32 4-bit values using lookup table
    for (size_t i = 0; i < 32; ++i) {
        uint8_t nibble = (i % 2 == 0) 
            ? (iq4_block->qs[i / 2] & 0x0F)
            : (iq4_block->qs[i / 2] >> 4);
        
        int8_t dequant_val = IQ4NL_LUT[nibble];
        
        // Requantize to Q8_0 (scale adjustment)
        float fp_val = dequant_val * scale;
        output->qs[i] = static_cast<int8_t>(std::round(fp_val));
    }

    // Store scale (may need adjustment)
    output->d = iq4_block->d;
}
```

**Reference**: Remove IQ4NL_LUT and decodeIQ4NLBlock from Q8_0WeightAccessor.h (already removed)

**Testing**:
- Verify LUT matches llama.cpp ggml-quants.c
- Test scale preservation vs recomputation
- Edge case: all-zero blocks

---

#### 2. Q6_KTensor::decode_to_q8_0()
**File**: `src/v2/tensors/Q6_KTensor.cpp`

**Implementation**:
```cpp
void Q6_KTensor::decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block* output) const
{
    // Q6_K: 6-bit packed quantization (256 elements per super-block, 8 sub-blocks of 32)
    constexpr size_t Q6K_SUPERBLOCK_SIZE = 256;
    const size_t superblock_idx = (k_block_offset * 32) / Q6K_SUPERBLOCK_SIZE;
    const size_t sub_idx = k_block_offset % 8;

    const Q6_KBlock* q6k = get_block(row_idx, superblock_idx);

    // Extract super-block scale (FP16)
    float superblock_scale = fp16_to_fp32(q6k->d);

    // Extract sub-block scale (4-bit packed: 8 scales in 4 bytes)
    uint8_t scale_byte = q6k->scales[sub_idx / 2];
    uint8_t scale_nibble = (sub_idx % 2 == 0) ? (scale_byte & 0x0F) : (scale_byte >> 4);
    float sub_scale = static_cast<float>(scale_nibble) / 16.0f;
    float combined_scale = superblock_scale * sub_scale;

    // Decode 6-bit values (32 elements = 192 bits = 24 bytes)
    const size_t q_offset = sub_idx * 24; // 24 bytes per sub-block
    const uint8_t* ql = q6k->ql + q_offset;
    const uint8_t* qh = q6k->qh + (sub_idx * 4); // 4 bytes high bits per sub-block

    for (size_t i = 0; i < 32; ++i) {
        // 6-bit unpacking: 4 low bits from ql, 2 high bits from qh
        uint8_t low_4bits = (i % 2 == 0) ? (ql[i/2] & 0x0F) : (ql[i/2] >> 4);
        uint8_t high_2bits = (qh[i/8] >> ((i % 8) * 2)) & 0x03;
        int8_t q6_val = static_cast<int8_t>((high_2bits << 4) | low_4bits) - 32; // Symmetric

        // Requantize to Q8_0
        float fp_val = q6_val * combined_scale;
        output->qs[i] = static_cast<int8_t>(std::round(fp_val));
    }

    output->d = fp32_to_fp16(combined_scale);
}
```

**Reference**: llama.cpp `dequantize_row_q6_K()` in ggml-quants.c

**Testing**:
- Verify 6-bit unpacking correctness
- Test hierarchical scale computation
- Edge case: extreme scale ratios

---

#### 3. FP32Tensor::decode_to_q8_0()
**File**: `src/v2/tensors/FP32Tensor.cpp`

**Implementation**:
```cpp
void FP32Tensor::decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block* output) const
{
    const float* row_data = data() + row_idx * shape()[1];
    const size_t k_start = k_block_offset * 32;
    const size_t k_count = std::min<size_t>(32, shape()[1] - k_start);
    const float* src = row_data + k_start;

    // Find max absolute value (Q8_0 symmetric quantization)
    float amax = 0.0f;
    for (size_t i = 0; i < k_count; ++i) {
        amax = std::max(amax, std::fabs(src[i]));
    }

    if (amax == 0.0f) {
        // All zeros - special case
        std::memset(output->qs, 0, 32);
        output->d = fp32_to_fp16(0.0f);
        return;
    }

    // Quantize with scale = amax / 127
    float scale = amax / 127.0f;
    float inv_scale = 1.0f / scale;

    for (size_t i = 0; i < k_count; ++i) {
        int32_t q = static_cast<int32_t>(std::round(src[i] * inv_scale));
        q = std::max(-127, std::min(127, q)); // Clamp to symmetric range
        output->qs[i] = static_cast<int8_t>(q);
    }

    // Zero-pad remaining elements (partial blocks)
    for (size_t i = k_count; i < 32; ++i) {
        output->qs[i] = 0;
    }

    output->d = fp32_to_fp16(scale);
}
```

**Reference**: Moved from FP32CachedAccessor (already removed)

**Testing**:
- Verify symmetric quantization correctness
- Test partial blocks (k_count < 32)
- Edge case: near-zero values, large dynamic range

---

### Priority 2: Extended K-series (High Priority)

All K-series formats use super-blocks with hierarchical quantization:

#### 4. Q4_KTensor::decode_to_q8_0()
**Format**: 4-bit + super-block scales (256 elements per super-block)  
**Reference**: llama.cpp `dequantize_row_q4_K()`

#### 5. Q5_KTensor::decode_to_q8_0()
**Format**: 5-bit + super-block scales (256 elements per super-block)  
**Reference**: llama.cpp `dequantize_row_q5_K()`

#### 6. Q2_KTensor::decode_to_q8_0()
**Format**: 2-bit + super-block scales (256 elements per super-block)  
**Reference**: llama.cpp `dequantize_row_q2_K()`

#### 7. Q3_KTensor::decode_to_q8_0()
**Format**: 3-bit + super-block scales (256 elements per super-block)  
**Reference**: llama.cpp `dequantize_row_q3_K()`

#### 8. Q8_KTensor::decode_to_q8_0()
**Format**: 8-bit + super-block scales (closest to Q8_0, minimal conversion)  
**Reference**: llama.cpp `dequantize_row_q8_K()`

---

### Priority 3: Legacy/Simple Formats (Medium Priority)

#### 9. Q4_0Tensor::decode_to_q8_0()
**Format**: 4-bit symmetric, 32 elements per block (simple, no super-blocks)  
**Reference**: llama.cpp `dequantize_row_q4_0()`

#### 10. Q4_1Tensor::decode_to_q8_0()
**Format**: 4-bit asymmetric (scale + min), 32 elements per block  
**Reference**: llama.cpp `dequantize_row_q4_1()`

#### 11. Q5_0Tensor::decode_to_q8_0()
**Format**: 5-bit symmetric, 32 elements per block  
**Reference**: llama.cpp `dequantize_row_q5_0()`

#### 12. Q5_1Tensor::decode_to_q8_0()
**Format**: 5-bit asymmetric (scale + min), 32 elements per block  
**Reference**: llama.cpp `dequantize_row_q5_1()`

---

### Priority 4: Advanced IQ Formats (Lower Priority)

#### 13-19. IQ Series Tensors
- **IQ4_XSTensor**: 4-bit importance matrix quantization
- **IQ2_XXSTensor**: 2-bit ultra-low precision
- **IQ2_XSTensor**: 2-bit extended precision
- **IQ3_XXSTensor**: 3-bit ultra-low precision
- **IQ2_STensor**: 2-bit standard
- **IQ3_STensor**: 3-bit standard
- **IQ1_STensor**: 1-bit binary quantization

**Reference**: llama.cpp `ggml-quants.c` (each has specific lookup tables)

---

## Factory Function Updates

Once tensor methods are implemented, update `createQ8_0Accessor()` in Q8_0WeightAccessor.h:

```cpp
inline std::unique_ptr<Q8_0WeightAccessor> createQ8_0Accessor(
    const TensorBase* tensor, size_t cache_size = 1024)
{
    switch (tensor->native_type()) {
        case TensorType::Q8_0:
            return std::make_unique<Q8_0DirectAccessor>(
                static_cast<const Q8_0Tensor*>(tensor));

        // Priority 1 (already supported)
        case TensorType::IQ4_NL:
            return std::make_unique<IQ4_NLCachedAccessor>(...);
        case TensorType::Q6_K:
            return std::make_unique<Q6_KCachedAccessor>(...);
        case TensorType::FP32:
            return std::make_unique<FP32CachedAccessor>(...);

        // Priority 2: K-series (add as implemented)
        case TensorType::Q4_K:
            return std::make_unique<Q4_KCachedAccessor>(...);
        case TensorType::Q5_K:
            return std::make_unique<Q5_KCachedAccessor>(...);
        case TensorType::Q2_K:
            return std::make_unique<Q2_KCachedAccessor>(...);
        // ... etc

        default:
            return nullptr; // Unsupported format
    }
}
```

**Alternative**: Template-based generic accessor to avoid code duplication:

```cpp
template<typename TensorType>
class GenericCachedAccessor : public Q8_0WeightAccessor {
    // Exactly same implementation as IQ4_NL/Q6_K/FP32CachedAccessor
    // Just parameterized on tensor type
};
```

---

## Testing Strategy

### Unit Tests (extend Test__IntegerGemm.cpp)

#### 1. Decode Correctness Tests
```cpp
TEST(IntegerGemm, IQ4_NL_DecodeCorrectness) {
    // Create IQ4_NL tensor with known values
    // Call decode_to_q8_0()
    // Compare against reference implementation
}

TEST(IntegerGemm, Q6_K_DecodeCorrectness) {
    // Test 6-bit unpacking with hierarchical scales
}

TEST(IntegerGemm, FP32_QuantizeCorrectness) {
    // Test symmetric quantization with edge cases
}
```

#### 2. Cache Hit Rate Tests
```cpp
TEST(IntegerGemm, CacheHitRates) {
    // Simulate GEMM access pattern (M-tiles × N-tiles)
    // Measure cache hit rate (expect >95%)
    // Vary cache size: 256, 512, 1024, 2048 entries
}
```

#### 3. Zero-Copy Validation
```cpp
TEST(IntegerGemm, Q8_0_ZeroCopyPath) {
    // Q8_0 tensor should use direct accessor
    // Verify no cache allocation
    // Verify returned pointers are tensor pointers
}
```

#### 4. Thread Safety Tests
```cpp
TEST(IntegerGemm, MultiThreadedAccess) {
    // OpenMP parallel access from multiple threads
    // Verify mutex correctness
    // No deadlocks, no race conditions
}
```

#### 5. Edge Case Tests
```cpp
TEST(IntegerGemm, PartialBlocks) {
    // K dimension not multiple of 32
    // Verify zero-padding
}

TEST(IntegerGemm, ExtremeScales) {
    // Very large/small FP32 values
    // Verify no overflow in quantization
}
```

### Performance Benchmarks

```cpp
// Compare decode overhead: IQ4_NL cached vs Q8_0 zero-copy
void benchmark_decode_overhead();

// Measure cache size sensitivity
void benchmark_cache_size_sensitivity();

// Full GEMM comparison: Q8_0 vs IQ4_NL vs FP32
void benchmark_integer_gemm_formats();
```

---

## Implementation Order

### Week 1: Core Formats
1. ✅ **Refactor accessors** (COMPLETED)
2. **Implement IQ4_NLTensor::decode_to_q8_0()** (~30 lines)
3. **Implement Q6_KTensor::decode_to_q8_0()** (~50 lines, 6-bit unpacking)
4. **Implement FP32Tensor::decode_to_q8_0()** (~30 lines)
5. **Unit tests for 3 formats** (~200 lines)
6. **Performance validation** (cache hit rates, zero-copy)

### Week 2: Extended K-series
1. Q4_K, Q5_K, Q2_K, Q3_K, Q8_K tensors (~200 lines total)
2. Update factory function with 5 new cases
3. Unit tests for K-series (~100 lines)

### Week 3: Legacy + Advanced
1. Q4_0, Q4_1, Q5_0, Q5_1 tensors (~100 lines)
2. IQ series tensors (IQ4_XS, IQ2_XXS, etc.) (~300 lines)
3. Comprehensive testing (~200 lines)
4. Performance benchmarking report

---

## Success Criteria

- ✅ All accessor classes delegate to tensor methods (no inline decode logic)
- ✅ Zero-copy path confirmed for Q8_0 weights
- ⏳ IQ4_NL, Q6_K, FP32 decode methods implemented and tested
- ⏳ Cache hit rate >95% for typical GEMM patterns
- ⏳ Unit tests passing for all implemented formats
- ⏳ Performance benchmarks show:
  - Q8_0 zero-copy: 0% decode overhead
  - IQ4_NL cached: <5% overhead vs Q8_0
  - FP32 quantized: <10% overhead vs Q8_0
- ⏳ Thread safety validated under OpenMP parallelism

---

## References

### llama.cpp Decode Functions
- **File**: `ggml-quants.c` (lines 2000-5000)
- **Functions**:
  - `dequantize_row_q4_0()`, `dequantize_row_q4_1()`
  - `dequantize_row_q5_0()`, `dequantize_row_q5_1()`
  - `dequantize_row_q8_0()` (reference for Q8_0 format)
  - `dequantize_row_q2_K()`, `dequantize_row_q3_K()`, `dequantize_row_q4_K()`, `dequantize_row_q5_K()`, `dequantize_row_q6_K()`, `dequantize_row_q8_K()`
  - `dequantize_row_iq4_nl()`, `dequantize_row_iq4_xs()`, `dequantize_row_iq2_xxs()`, etc.

### Llaminar Tensor Definitions
- **File**: `src/v2/tensors/Tensors.h` (lines 1-500)
- **Classes**: Q8_0Tensor, IQ4_NLTensor, Q6_KTensor, FP32Tensor, Q4_KTensor, etc.
- **Block Structures**: Q8_0Block, IQ4_NLBlock, Q6_KBlock, etc.

### Llaminar Accessor (Refactored)
- **File**: `src/v2/kernels/cpu/gemm/Q8_0WeightAccessor.h`
- **Classes**: Q8_0DirectAccessor, IQ4_NLCachedAccessor, Q6_KCachedAccessor, FP32CachedAccessor
- **Cache**: WeightDecodeCache (LRU, 1024 entries default)

---

## Next Steps

**Immediate**:
1. Implement `IQ4_NLTensor::decode_to_q8_0()` in `src/v2/tensors/IQ4_NLTensor.cpp`
2. Implement `Q6_KTensor::decode_to_q8_0()` in `src/v2/tensors/Q6_KTensor.cpp`
3. Implement `FP32Tensor::decode_to_q8_0()` in `src/v2/tensors/FP32Tensor.cpp`
4. Add method declarations to respective header files
5. Build and verify compilation

**Follow-up**:
1. Write unit tests for 3 core formats
2. Performance validation (cache hit rates)
3. Extend to K-series formats
4. Documentation update (INTEGER_GEMM_QUICK_REF.md)

---

## File Structure

```
src/v2/
├── tensors/
│   ├── Tensors.h              (declarations: add decode_to_q8_0() signatures)
│   ├── IQ4_NLTensor.cpp       (NEW: implement decode_to_q8_0() + IQ4NL_LUT)
│   ├── Q6_KTensor.cpp         (NEW: implement decode_to_q8_0() + 6-bit unpack)
│   ├── FP32Tensor.cpp         (NEW: implement decode_to_q8_0() + quantize)
│   ├── Q4_KTensor.cpp         (TODO: implement decode_to_q8_0())
│   └── ...                    (TODO: other formats)
│
├── kernels/cpu/gemm/
│   ├── Q8_0WeightAccessor.h   (REFACTORED: delegates to tensor methods)
│   └── ...
│
└── tests/
    └── Test__IntegerGemm.cpp  (EXTEND: add decode correctness tests)
```

---

**Status**: Accessor refactoring complete ✅  
**Next Action**: Implement decode methods in tensor classes  
**Estimated Effort**: ~2-3 hours for core 3 formats + tests  
