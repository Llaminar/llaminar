/**
 * @file Test__Q16BlockStructures.cpp
 * @brief Unit tests for Q16_1 variable block size structures
 *
 * Tests the Q16_1Block_64 and Q16_1Block_128 structures
 * and the Q16BlockSize enum and type traits.
 *
 * Note: Q16_1Block_192 was removed. DeepSeek V3 MLA uses separate NOPE (128-dim)
 * and ROPE (64-dim) tensors with independent scales.
 */

#include <gtest/gtest.h>
#include "tensors/BlockStructures.h"
#include <cstring>
#include <cmath>
#include <type_traits>

using namespace llaminar2;

class Test__Q16BlockStructures : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// =============================================================================
// Structure Size and Layout Tests
// =============================================================================

TEST_F(Test__Q16BlockStructures, BlockSizesCorrect)
{
    // Verify BLOCK_SIZE constants match expected values
    EXPECT_EQ(Q16_1Block::BLOCK_SIZE, 32);
    EXPECT_EQ(Q16_1Block_64::BLOCK_SIZE, 64);
    EXPECT_EQ(Q16_1Block_128::BLOCK_SIZE, 128);
}

TEST_F(Test__Q16BlockStructures, StructureSizesCorrect)
{
    // Memory layout: float d (4) + int32_t sum_qs (4) + int16_t qs[N] (2*N)
    EXPECT_EQ(sizeof(Q16_1Block), 72);      // 4 + 4 + 64 = 72
    EXPECT_EQ(sizeof(Q16_1Block_64), 136);  // 4 + 4 + 128 = 136
    EXPECT_EQ(sizeof(Q16_1Block_128), 264); // 4 + 4 + 256 = 264
}

TEST_F(Test__Q16BlockStructures, MemoryOverheadDecreases)
{
    // Overhead = (scale_bytes) / (data_bytes) = 8 / (2 * BLOCK_SIZE)
    // Larger blocks should have lower overhead
    constexpr float overhead_32 = 8.0f / (2.0f * 32);   // 12.5%
    constexpr float overhead_64 = 8.0f / (2.0f * 64);   // 6.25%
    constexpr float overhead_128 = 8.0f / (2.0f * 128); // 3.125%

    EXPECT_GT(overhead_32, overhead_64);
    EXPECT_GT(overhead_64, overhead_128);
}

TEST_F(Test__Q16BlockStructures, FieldOffsetsConsistent)
{
    // All Q16_1 block variants should have same field layout:
    // d at offset 0, sum_qs at offset 4, qs at offset 8
    EXPECT_EQ(offsetof(Q16_1Block, d), 0);
    EXPECT_EQ(offsetof(Q16_1Block, sum_qs), 4);
    EXPECT_EQ(offsetof(Q16_1Block, qs), 8);

    EXPECT_EQ(offsetof(Q16_1Block_64, d), 0);
    EXPECT_EQ(offsetof(Q16_1Block_64, sum_qs), 4);
    EXPECT_EQ(offsetof(Q16_1Block_64, qs), 8);

    EXPECT_EQ(offsetof(Q16_1Block_128, d), 0);
    EXPECT_EQ(offsetof(Q16_1Block_128, sum_qs), 4);
    EXPECT_EQ(offsetof(Q16_1Block_128, qs), 8);
}

// =============================================================================
// Q16BlockSize Enum Tests
// =============================================================================

TEST_F(Test__Q16BlockStructures, EnumValuesMatchBlockSizes)
{
    EXPECT_EQ(static_cast<int>(Q16BlockSize::BLOCK_32), 32);
    EXPECT_EQ(static_cast<int>(Q16BlockSize::BLOCK_64), 64);
    EXPECT_EQ(static_cast<int>(Q16BlockSize::BLOCK_128), 128);
}

// =============================================================================
// Type Traits Tests
// =============================================================================

TEST_F(Test__Q16BlockStructures, TypeTraitsMappingCorrect)
{
    // Q16BlockType_t should map enum to struct
    static_assert(std::is_same_v<Q16BlockType_t<Q16BlockSize::BLOCK_32>, Q16_1Block>,
                  "BLOCK_32 should map to Q16_1Block");
    static_assert(std::is_same_v<Q16BlockType_t<Q16BlockSize::BLOCK_64>, Q16_1Block_64>,
                  "BLOCK_64 should map to Q16_1Block_64");
    static_assert(std::is_same_v<Q16BlockType_t<Q16BlockSize::BLOCK_128>, Q16_1Block_128>,
                  "BLOCK_128 should map to Q16_1Block_128");

    // Runtime check (compile-time already passed)
    EXPECT_EQ(sizeof(Q16BlockType_t<Q16BlockSize::BLOCK_32>), sizeof(Q16_1Block));
    EXPECT_EQ(sizeof(Q16BlockType_t<Q16BlockSize::BLOCK_64>), sizeof(Q16_1Block_64));
    EXPECT_EQ(sizeof(Q16BlockType_t<Q16BlockSize::BLOCK_128>), sizeof(Q16_1Block_128));
}

// =============================================================================
// optimal_q16_block_size Tests
// =============================================================================

TEST_F(Test__Q16BlockStructures, OptimalBlockSizeForCommonHeadDims)
{
    // Exact matches
    EXPECT_EQ(optimal_q16_block_size(64), Q16BlockSize::BLOCK_64);
    EXPECT_EQ(optimal_q16_block_size(128), Q16BlockSize::BLOCK_128);
}

TEST_F(Test__Q16BlockStructures, OptimalBlockSizeForMultiples)
{
    // Multiples should prefer largest fitting block
    EXPECT_EQ(optimal_q16_block_size(256), Q16BlockSize::BLOCK_128); // 256 / 128 = 2 blocks
    EXPECT_EQ(optimal_q16_block_size(512), Q16BlockSize::BLOCK_128); // 512 / 128 = 4 blocks
}

TEST_F(Test__Q16BlockStructures, OptimalBlockSizeFallback)
{
    // Non-standard dimensions should fall back to 64
    EXPECT_EQ(optimal_q16_block_size(96), Q16BlockSize::BLOCK_64); // 96 not divisible by 128/192
    EXPECT_EQ(optimal_q16_block_size(80), Q16BlockSize::BLOCK_64); // 80 not divisible by 128/192
    EXPECT_EQ(optimal_q16_block_size(32), Q16BlockSize::BLOCK_64); // Use 64 as minimum for attention
}

TEST_F(Test__Q16BlockStructures, OptimalBlockSizeForRealModels)
{
    // Test actual model head dimensions
    // Qwen2.5-0.5B: head_dim = 64
    EXPECT_EQ(optimal_q16_block_size(64), Q16BlockSize::BLOCK_64);

    // Llama-3, Qwen3, Mistral: head_dim = 128
    EXPECT_EQ(optimal_q16_block_size(128), Q16BlockSize::BLOCK_128);

    // Note: DeepSeek V3 MLA uses separate NOPE (128-dim) + ROPE (64-dim) tensors
    // with independent scales, not a combined 192-dim block.
}

// =============================================================================
// Quantization/Dequantization Round-Trip Tests
// =============================================================================

template <typename BlockType>
void test_quantize_dequant_roundtrip()
{
    constexpr size_t N = BlockType::BLOCK_SIZE;

    // Create test data with known pattern
    float original[N];
    for (size_t i = 0; i < N; ++i)
    {
        original[i] = static_cast<float>(i) - static_cast<float>(N / 2);
    }

    // Find max for scaling
    float max_abs = 0.0f;
    for (size_t i = 0; i < N; ++i)
    {
        max_abs = std::max(max_abs, std::abs(original[i]));
    }

    // Quantize
    BlockType block;
    block.d = max_abs / 32767.0f;
    block.sum_qs = 0;

    float inv_scale = (max_abs > 1e-10f) ? 32767.0f / max_abs : 0.0f;
    for (size_t i = 0; i < N; ++i)
    {
        int32_t q = static_cast<int32_t>(std::round(original[i] * inv_scale));
        q = std::max(-32767, std::min(32767, q));
        block.qs[i] = static_cast<int16_t>(q);
        block.sum_qs += block.qs[i];
    }

    // Dequantize
    float reconstructed[N];
    for (size_t i = 0; i < N; ++i)
    {
        reconstructed[i] = static_cast<float>(block.qs[i]) * block.d;
    }

    // Verify round-trip accuracy
    float max_error = 0.0f;
    for (size_t i = 0; i < N; ++i)
    {
        float err = std::abs(original[i] - reconstructed[i]);
        max_error = std::max(max_error, err);
    }

    // Error should be bounded by quantization step
    float quant_step = max_abs / 32767.0f;
    EXPECT_LT(max_error, quant_step * 1.1f) << "Max error exceeds quantization step for block size " << N;

    // Verify sum_qs is correct
    int32_t computed_sum = 0;
    for (size_t i = 0; i < N; ++i)
    {
        computed_sum += block.qs[i];
    }
    EXPECT_EQ(block.sum_qs, computed_sum) << "sum_qs mismatch for block size " << N;
}

TEST_F(Test__Q16BlockStructures, QuantDequantRoundTrip32)
{
    test_quantize_dequant_roundtrip<Q16_1Block>();
}

TEST_F(Test__Q16BlockStructures, QuantDequantRoundTrip64)
{
    test_quantize_dequant_roundtrip<Q16_1Block_64>();
}

TEST_F(Test__Q16BlockStructures, QuantDequantRoundTrip128)
{
    test_quantize_dequant_roundtrip<Q16_1Block_128>();
}

// =============================================================================
// SIMD Alignment Tests
// =============================================================================

TEST_F(Test__Q16BlockStructures, BlocksAreAligned)
{
    // Blocks should be aligned for SIMD access
    // qs[] starts at offset 8, which is 8-byte aligned
    // For AVX2/AVX512, we want 32/64-byte alignment for qs data

    Q16_1Block_64 block64;
    Q16_1Block_128 block128;

    // The qs array should be at least 4-byte aligned (for int16_t pairs)
    EXPECT_EQ(reinterpret_cast<uintptr_t>(&block64.qs) % 4, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(&block128.qs) % 4, 0);
}

TEST_F(Test__Q16BlockStructures, QsArraySizesSIMDFriendly)
{
    // qs array sizes should be multiples of common SIMD widths
    // AVX2 processes 16 int16_t at once (256 bits)
    // AVX512 processes 32 int16_t at once (512 bits)

    EXPECT_EQ(Q16_1Block_64::BLOCK_SIZE % 16, 0) << "64 should be divisible by 16 (AVX2)";
    EXPECT_EQ(Q16_1Block_128::BLOCK_SIZE % 32, 0) << "128 should be divisible by 32 (AVX512)";
}

// =============================================================================
// POD Type Requirements
// =============================================================================

TEST_F(Test__Q16BlockStructures, BlocksArePOD)
{
    // All block types should be POD (trivially copyable) for SIMD and MPI
    static_assert(std::is_trivially_copyable_v<Q16_1Block>,
                  "Q16_1Block must be trivially copyable");
    static_assert(std::is_trivially_copyable_v<Q16_1Block_64>,
                  "Q16_1Block_64 must be trivially copyable");
    static_assert(std::is_trivially_copyable_v<Q16_1Block_128>,
                  "Q16_1Block_128 must be trivially copyable");

    static_assert(std::is_standard_layout_v<Q16_1Block>,
                  "Q16_1Block must have standard layout");
    static_assert(std::is_standard_layout_v<Q16_1Block_64>,
                  "Q16_1Block_64 must have standard layout");
    static_assert(std::is_standard_layout_v<Q16_1Block_128>,
                  "Q16_1Block_128 must have standard layout");

    SUCCEED(); // Static asserts passed
}
// =============================================================================
// Helper Function Tests (q16_block_size_bytes, q16_block_size_elements)
// =============================================================================

TEST_F(Test__Q16BlockStructures, Q16BlockSizeBytesCorrect)
{
    // q16_block_size_bytes should return sizeof() for each block type
    EXPECT_EQ(q16_block_size_bytes(Q16BlockSize::BLOCK_32), sizeof(Q16_1Block));
    EXPECT_EQ(q16_block_size_bytes(Q16BlockSize::BLOCK_64), sizeof(Q16_1Block_64));
    EXPECT_EQ(q16_block_size_bytes(Q16BlockSize::BLOCK_128), sizeof(Q16_1Block_128));

    // Verify actual byte counts
    EXPECT_EQ(q16_block_size_bytes(Q16BlockSize::BLOCK_32), 72);   // 4+4+64
    EXPECT_EQ(q16_block_size_bytes(Q16BlockSize::BLOCK_64), 136);  // 4+4+128
    EXPECT_EQ(q16_block_size_bytes(Q16BlockSize::BLOCK_128), 264); // 4+4+256
}

TEST_F(Test__Q16BlockStructures, Q16BlockSizeElementsCorrect)
{
    // q16_block_size_elements should return the element count
    EXPECT_EQ(q16_block_size_elements(Q16BlockSize::BLOCK_32), 32);
    EXPECT_EQ(q16_block_size_elements(Q16BlockSize::BLOCK_64), 64);
    EXPECT_EQ(q16_block_size_elements(Q16BlockSize::BLOCK_128), 128);
}

TEST_F(Test__Q16BlockStructures, HelperFunctionsAreConstexpr)
{
    // These should be usable in compile-time contexts
    constexpr size_t bytes_32 = q16_block_size_bytes(Q16BlockSize::BLOCK_32);
    constexpr size_t bytes_64 = q16_block_size_bytes(Q16BlockSize::BLOCK_64);
    constexpr size_t elems_128 = q16_block_size_elements(Q16BlockSize::BLOCK_128);

    EXPECT_EQ(bytes_32, 72);
    EXPECT_EQ(bytes_64, 136);
    EXPECT_EQ(elems_128, 128);
}

TEST_F(Test__Q16BlockStructures, BlockSizeCalculationFormula)
{
    // Verify the formula: bytes = 8 + 2*elements (scale + sum + int16 array)
    auto verify_formula = [](Q16BlockSize size)
    {
        size_t elements = q16_block_size_elements(size);
        size_t expected_bytes = 8 + 2 * elements; // 4 (d) + 4 (sum_qs) + 2*N (qs)
        EXPECT_EQ(q16_block_size_bytes(size), expected_bytes)
            << "Formula mismatch for block size " << static_cast<int>(size);
    };

    verify_formula(Q16BlockSize::BLOCK_32);
    verify_formula(Q16BlockSize::BLOCK_64);
    verify_formula(Q16BlockSize::BLOCK_128);
}