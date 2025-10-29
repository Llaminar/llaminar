# IQ4_NL Tensor View Support - Phase 2 Complete

**Date**: 2025-10-29  
**Status**: ✅ Complete - All tests passing  
**Phase**: Tensor View Support Phase 2 (Quantized Tensors - Row-Slice Views)

## Summary

Successfully implemented row-slice view support for IQ4_NL quantized tensors, establishing the pattern for all 32-element block quantized formats (Q8_0, Q4_0, Q4_1). Views preserve block structure and enable zero-copy slicing of weight matrices.

## Changes Implemented

### IQ4_NL Tensor View Support

**Files Modified**:
- `src/v2/tensors/Tensors.h` (IQ4_NLTensor class)
- `src/v2/tensors/IQ4_NLTensor.cpp`
- `tests/v2/unit/tensors/Test__IQ4_NLTensor_Views.cpp` (NEW - 12 comprehensive tests)
- `tests/v2/CMakeLists.txt`

**Key Features**:
- **Row-slice views only**: Preserves K dimension (column count), slices along row dimension
- **Block structure preservation**: Views always start at row boundaries (block-aligned)
- **Zero-copy semantics**: Views borrow quantized data via raw pointer + byte offset
- **View chaining**: View of view chains to root parent automatically
- **IBlockDecoder compatibility**: All decode paths work correctly for views

### View Creation API

```cpp
// Create view: rows 5-9 from parent (10 rows × 64 cols)
auto view = parent->create_view({5, 64}, 5 * 64);
//                               ^shape  ^offset (element-aligned)
```

**Validation Rules**:
1. ✅ **2D Shape**: View must be 2D tensor
2. ✅ **K Dimension Match**: `view_shape[1] == parent_shape[1]` (row-slice only)
3. ✅ **Row Alignment**: Offset must be multiple of K (row-aligned)
4. ✅ **Bounds Checking**: `start_row + view_rows <= parent_rows`

**Error Handling**:
```cpp
// Mismatched K dimension
auto view = parent->create_view({5, 32}, 0);  // FAIL: K=32 != parent K=64

// Non-row-aligned offset
auto view = parent->create_view({5, 64}, 32);  // FAIL: offset % K != 0

// Out of bounds
auto view = parent->create_view({8, 64}, 5 * 64);  // FAIL: rows 5-12 exceeds parent[10]
```

### Implementation Details

**View Storage**:
```cpp
class IQ4_NLTensor {
private:
    std::vector<size_t> shape_;
    
    // Data ownership
    bool is_view_;                          // true if this is a view
    std::vector<uint8_t> raw_data_;         // Owned data (if !is_view_)
    const uint8_t *raw_data_ptr_;           // Borrowed parent data (if is_view_)
    size_t view_byte_offset_;               // Byte offset into parent's raw_data_
    std::shared_ptr<TensorBase> parent_;    // Keep parent alive
};
```

**Block Byte Offset Calculation**:
```cpp
// For row-slice view starting at row R:
size_t K = parent_shape[1];
size_t blocks_per_row = (K + 31) / 32;  // IQ4_NLBlock::BLOCK_SIZE = 32
size_t block_offset = R * blocks_per_row;
size_t byte_offset = block_offset * sizeof(IQ4_NLBlock);  // 20 bytes/block
```

**View Chaining**:
```cpp
std::shared_ptr<TensorBase> IQ4_NLTensor::create_view(...) {
    if (is_view_) {
        // Chain to existing parent
        root_parent = parent_;
        root_data_ptr = raw_data_ptr_;
        root_byte_offset = view_byte_offset_ + byte_offset;  // Accumulate offsets
    } else {
        // This is the root
        root_parent = shared_from_this();
        root_data_ptr = raw_data_.data();
        root_byte_offset = byte_offset;
    }
    return std::make_shared<IQ4_NLTensor>(new_shape, root_data_ptr, root_byte_offset, root_parent);
}
```

### Data Access Modifications

All methods updated to use view-aware data pointer:

```cpp
// Pattern used in all decode methods:
const uint8_t *data_ptr = is_view_ 
    ? (raw_data_ptr_ + view_byte_offset_) 
    : raw_data_.data();
const IQ4_NLBlock *blocks = reinterpret_cast<const IQ4_NLBlock*>(data_ptr);
```

**Updated Methods**:
- ✅ `decode_to_fp32()` - Full tensor decode to FP32
- ✅ `decode_to_bf16()` - Full tensor decode to BF16
- ✅ `decodeRow()` - Single row decode
- ✅ `decodeSpan()` - Arbitrary element range decode
- ✅ `get_block_at()` - Block-level access (GEMM helper)
- ✅ `decode_tile_blocks()` - Tile decode (GEMM helper)
- ✅ `decode_block_at()` - IBlockDecoder interface (inline, hot path)
- ✅ `get_raw_block_at()` - IBlockDecoder interface (inline, hot path)

**IBlockDecoder Inline Methods** (zero-overhead for GEMM):
```cpp
__attribute__((always_inline)) 
void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override {
    const uint8_t *data_ptr = is_view_ 
        ? (raw_data_ptr_ + view_byte_offset_) 
        : raw_data_.data();
    const IQ4_NLBlock *blocks = reinterpret_cast<const IQ4_NLBlock*>(data_ptr);
    const size_t block_idx = row_idx * blocks_per_row + k_block_offset;
    decodeBlock(blocks[block_idx], output);
}
```

### Test Suite

**12 Comprehensive Tests** (`Test__IQ4_NLTensor_Views`):

1. ✅ **BasicViewCreation** - Create view with zero offset
2. ✅ **ViewWithOffset** - Create view at non-zero row offset
3. ✅ **KDimensionMustMatch** - Reject views with mismatched K dimension
4. ✅ **OffsetMustBeRowAligned** - Reject non-row-aligned offsets
5. ✅ **ViewBoundsChecking** - Reject views exceeding parent bounds
6. ✅ **ViewDataAccess** - Verify decoded data matches parent rows
7. ✅ **ViewLifetime** - Parent kept alive by view's shared_ptr
8. ✅ **ViewChaining** - View of view chains to root parent
9. ✅ **IBlockDecoderInterface** - Verify IBlockDecoder methods work for views
10. ✅ **MultipleViews** - Multiple concurrent views of same parent
11. ✅ **NonStandardColumns** - Views with cols not multiple of 32 (tail blocks)
12. ✅ **DecodeRowFromView** - decodeRow() works correctly on views

**Test Results**:
```
[==========] 12 tests from 1 test suite ran. (1 ms total)
[  PASSED  ] 12 tests.
100% tests passed, 0 tests failed
```

**Integration Testing**:
- ✅ All existing FP16/BF16 view tests still pass (14 + 19 tests)
- ✅ IQ4_NL performance benchmarks still pass (GEMM, TileSweep)
- ✅ No regressions in existing unit tests

## Technical Details

### Row-Slice Restriction Rationale

**Why row-slice only?**
1. **Block Alignment**: IQ4_NL uses 32-element blocks arranged row-wise
2. **Decode Efficiency**: Partial blocks require complex offset handling
3. **GEMM Compatibility**: Row-major GEMM kernels naturally slice by rows
4. **Simplicity**: Single offset dimension is easier to reason about

**What's allowed:**
```cpp
// ✅ Full rows (complete blocks)
auto view = parent->create_view({5, 64}, 5 * 64);  // Rows 5-9

// ✅ Non-power-of-2 columns (tail blocks handled automatically)
auto view = parent->create_view({5, 50}, 0);  // 50 = 1 full block + 18 element tail
```

**What's not allowed:**
```cpp
// ❌ Column slicing (breaks block structure)
auto view = parent->create_view({10, 32}, 0);  // K must match parent

// ❌ Arbitrary element offsets (must be row-aligned)
auto view = parent->create_view({5, 64}, 32);  // offset % K must == 0
```

### Memory Layout Example

**Parent Tensor**: 10 rows × 64 cols
- Blocks per row: (64 + 31) / 32 = 2 blocks
- Total blocks: 10 × 2 = 20 blocks
- Raw data size: 20 × 20 bytes = 400 bytes

**View**: Rows 5-9 (5 rows × 64 cols, offset = 5 * 64 = 320 elements)
- Block offset: row 5 × 2 blocks/row = block 10
- Byte offset: 10 blocks × 20 bytes/block = 200 bytes
- View raw_data_ptr: `parent->raw_data_.data() + 200`
- View blocks per row: 2 (same as parent)
- View total blocks: 5 × 2 = 10 blocks

### Performance Characteristics

- **Zero-Copy**: Views add zero memory overhead (only metadata)
- **Inline Hot Path**: `decode_block_at()` uses `__attribute__((always_inline))` for GEMM performance
- **Cache Locality**: Views reference contiguous parent memory (row-major)
- **GEMM Integration**: QuantizedGemmKernel works identically on views and owned tensors

### Use Cases

**Weight Matrix Slicing**:
```cpp
// Shard 4096×4096 weight matrix across 4 devices
auto weights = loadIQ4NLWeights("model.gguf");  // 4096×4096

auto shard0 = weights->create_view({1024, 4096}, 0 * 1024 * 4096);  // Rows 0-1023
auto shard1 = weights->create_view({1024, 4096}, 1 * 1024 * 4096);  // Rows 1024-2047
auto shard2 = weights->create_view({1024, 4096}, 2 * 1024 * 4096);  // Rows 2048-3071
auto shard3 = weights->create_view({1024, 4096}, 3 * 1024 * 4096);  // Rows 3072-4095
```

**Batch Processing**:
```cpp
// Process 512 row tensor in 128-row batches
for (size_t i = 0; i < 512; i += 128) {
    auto batch = tensor->create_view({128, K}, i * K);
    processBatch(batch);  // Zero-copy slicing
}
```

## Next Steps (Phase 2 Continuation)

Per TENSOR_VIEW_SPECIFICATION.md, Phase 2 includes all 32-element block quantized formats:

### Remaining 32-Element Block Formats
- 🔲 **Q8_0**: 8-bit uniform quantization (32 elements/block, FP16 scale)
- 🔲 **Q4_0**: 4-bit uniform quantization (32 elements/block, FP16 scale)
- 🔲 **Q4_1**: 4-bit quantization with min offset (32 elements/block, FP16 scale + min)

**Implementation Effort**: ~2-3 hours total (identical pattern to IQ4_NL)
- Each format: ~30 minutes (header update + cpp update + tests)
- Pattern established, just apply to different block types

### Phase 3: K-Quant Tensors (Deferred)
- Q4_K, Q6_K, Q2_K, Q3_K, Q5_K, Q8_K
- Super-block alignment (256 elements)
- Estimated: 12-16 hours

## References

**Design Document**: `/workspaces/llaminar/TENSOR_VIEW_SPECIFICATION.md`
**Related Files**:
- `src/v2/tensors/Tensors.h` - IQ4_NLTensor class declaration
- `src/v2/tensors/IQ4_NLTensor.cpp` - Implementation (575 lines)
- `tests/v2/unit/tensors/Test__IQ4_NLTensor_Views.cpp` - Tests (389 lines)
- `tests/v2/CMakeLists.txt` - Test configuration

**Test Execution**:
```bash
# Run IQ4_NL view tests
cd /workspaces/llaminar/build_v2
ctest -R "V2_Unit_IQ4NL_Views" --output-on-failure

# Run all view tests
ctest -R "View" --output-on-failure

# Run specific test
./tests/v2/v2_test_iq4nl_views --gtest_filter="*ViewChaining"
```

## Lessons Learned

1. **Row-Slice Pattern Works**: Simple restriction (preserve K dimension) handles 90% of use cases
2. **Block Alignment Natural**: Row boundaries align with block boundaries for all row counts
3. **View Chaining Critical**: Must always chain to root parent (never view→view)
4. **Inline Matters**: `__attribute__((always_inline))` on IBlockDecoder methods essential for GEMM performance
5. **Test-Driven Development**: Comprehensive tests caught edge cases (tail blocks, chaining, bounds)

---

**Session**: Oct 29, 2025  
**Duration**: ~3 hours (implementation + testing + documentation)  
**Outcome**: Production-ready IQ4_NL view support with full test coverage, establishes pattern for all 32-element block formats
