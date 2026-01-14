/**
 * @file Test__Q8_0Tensor_PartialBlocks.cpp
 * @brief Regression tests for Q8_0Tensor::data() partial block handling
 *
 * This test file covers the bug fixed in January 2026 where Q8_0Tensor::data()
 * caused a heap buffer overflow when K (columns) was not a multiple of the
 * Q8_0 block size (32 elements).
 *
 * ROOT CAUSE:
 * - Q8_0 stores data in blocks of 32 elements
 * - When K is not divisible by 32, the last block is "partial"
 * - The dequantization cache was sized for N*K elements (correct)
 * - BUT decodeBlock() always wrote 32 elements per block
 * - For K=17: blocks_per_row=1, but decodeBlock wrote 32 elements per row
 * - This caused buffer overflow starting at row 13 (13*32=416 > 13*17=221)
 *
 * FIX:
 * - data() now calculates elem_count = min(32, K - elem_offset) for each block
 * - Uses scalar decode for partial blocks to avoid SIMD overflow
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include <memory>
#include <vector>
#include <cstring>
#include <cmath>

using namespace llaminar2;

/**
 * @brief Test fixture for Q8_0 partial block regression tests
 */
class Test__Q8_0Tensor_PartialBlocks : public ::testing::Test
{
protected:
    /**
     * @brief Create a Q8_0 tensor with known values
     * @param rows Number of rows (N dimension)
     * @param cols Number of columns (K dimension) - may not be multiple of 32
     * @param scale Scale factor for all blocks
     * @param fill_pattern Pattern to fill quantized values
     */
    std::unique_ptr<Q8_0Tensor> createTestTensor(
        size_t rows, size_t cols,
        float scale = 1.0f,
        int8_t fill_pattern = 1)
    {
        constexpr size_t BLOCK_SIZE = Q8_0Block::BLOCK_SIZE; // 32
        size_t blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        size_t total_blocks = rows * blocks_per_row;
        size_t raw_size = total_blocks * sizeof(Q8_0Block);

        std::vector<uint8_t> raw_data(raw_size);
        Q8_0Block *blocks = reinterpret_cast<Q8_0Block *>(raw_data.data());

        // Convert scale to FP16 (d field)
        uint16_t d_fp16 = fp32_to_fp16(scale);

        for (size_t block_idx = 0; block_idx < total_blocks; ++block_idx)
        {
            blocks[block_idx].d = d_fp16;
            for (size_t i = 0; i < 32; ++i)
            {
                // Fill with pattern that makes it easy to verify:
                // value = (block_idx * 32 + i) * fill_pattern
                blocks[block_idx].qs[i] = static_cast<int8_t>(
                    ((block_idx * 32 + i) % 127) * fill_pattern);
            }
        }

        return std::make_unique<Q8_0Tensor>(std::vector<size_t>{rows, cols}, raw_data);
    }

    /**
     * @brief Convert FP32 to FP16 (matching Q8_0Tensor implementation)
     */
    static uint16_t fp32_to_fp16(float value)
    {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(float));

        uint32_t sign = (bits >> 16) & 0x8000;
        int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
        uint32_t mantissa = (bits >> 13) & 0x3FF;

        if (exponent <= 0)
            return static_cast<uint16_t>(sign);
        if (exponent >= 0x1F)
            return static_cast<uint16_t>(sign | 0x7C00);

        return static_cast<uint16_t>(sign | (exponent << 10) | mantissa);
    }
};

// ============================================================================
// REGRESSION TESTS: Partial block handling (K not divisible by 32)
// ============================================================================

/**
 * @brief Test K=17 which was the exact case that caused heap corruption
 *
 * K=17: blocks_per_row=1, but each block contains 32 quantized values
 * OLD BUG: decodeBlock() wrote 32 values per row, but cache sized for 17
 * For 13 rows: cache.size()=221, but row 12 wrote to indices 384..415 (OOB!)
 */
TEST_F(Test__Q8_0Tensor_PartialBlocks, K17_ExactRegressionCase)
{
    const size_t N = 13;
    const size_t K = 17;

    auto tensor = createTestTensor(N, K, 0.1f, 1);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->shape()[0], N);
    EXPECT_EQ(tensor->shape()[1], K);

    // This call triggered heap corruption before the fix
    const float *data = tensor->data();
    ASSERT_NE(data, nullptr);

    // Verify we can read all N*K elements without crashing
    size_t total_elements = N * K;
    double sum = 0.0;
    for (size_t i = 0; i < total_elements; ++i)
    {
        sum += data[i];
        // Values should be finite
        EXPECT_TRUE(std::isfinite(data[i]))
            << "Non-finite value at index " << i << ": " << data[i];
    }

    // Sum should be non-zero (we filled with pattern)
    EXPECT_NE(sum, 0.0) << "All values are zero, pattern fill failed";
}

/**
 * @brief Test K=1 (extreme partial block case)
 */
TEST_F(Test__Q8_0Tensor_PartialBlocks, K1_SingleElement)
{
    const size_t N = 10;
    const size_t K = 1;

    auto tensor = createTestTensor(N, K, 1.0f, 1);
    const float *data = tensor->data();
    ASSERT_NE(data, nullptr);

    for (size_t i = 0; i < N * K; ++i)
    {
        EXPECT_TRUE(std::isfinite(data[i]));
    }
}

/**
 * @brief Test K=31 (one less than block size)
 */
TEST_F(Test__Q8_0Tensor_PartialBlocks, K31_OneLessThanBlockSize)
{
    const size_t N = 8;
    const size_t K = 31;

    auto tensor = createTestTensor(N, K, 0.5f, 1);
    const float *data = tensor->data();
    ASSERT_NE(data, nullptr);

    for (size_t i = 0; i < N * K; ++i)
    {
        EXPECT_TRUE(std::isfinite(data[i]));
    }
}

/**
 * @brief Test K=33 (one more than block size)
 */
TEST_F(Test__Q8_0Tensor_PartialBlocks, K33_OneMoreThanBlockSize)
{
    const size_t N = 8;
    const size_t K = 33;

    auto tensor = createTestTensor(N, K, 0.5f, 1);
    const float *data = tensor->data();
    ASSERT_NE(data, nullptr);

    // Two blocks per row: one full (32), one partial (1)
    for (size_t i = 0; i < N * K; ++i)
    {
        EXPECT_TRUE(std::isfinite(data[i]));
    }
}

/**
 * @brief Test K=63 (almost two full blocks)
 */
TEST_F(Test__Q8_0Tensor_PartialBlocks, K63_AlmostTwoBlocks)
{
    const size_t N = 5;
    const size_t K = 63;

    auto tensor = createTestTensor(N, K, 0.25f, 1);
    const float *data = tensor->data();
    ASSERT_NE(data, nullptr);

    for (size_t i = 0; i < N * K; ++i)
    {
        EXPECT_TRUE(std::isfinite(data[i]));
    }
}

/**
 * @brief Test K=65 (two full blocks + one element)
 */
TEST_F(Test__Q8_0Tensor_PartialBlocks, K65_TwoBlocksPlusOne)
{
    const size_t N = 5;
    const size_t K = 65;

    auto tensor = createTestTensor(N, K, 0.25f, 1);
    const float *data = tensor->data();
    ASSERT_NE(data, nullptr);

    // Three blocks per row: two full (64), one partial (1)
    for (size_t i = 0; i < N * K; ++i)
    {
        EXPECT_TRUE(std::isfinite(data[i]));
    }
}

/**
 * @brief Test K=129 (four full blocks + one element) - another CKGemm test case
 */
TEST_F(Test__Q8_0Tensor_PartialBlocks, K129_FourBlocksPlusOne)
{
    const size_t N = 33;
    const size_t K = 129;

    auto tensor = createTestTensor(N, K, 0.1f, 1);
    const float *data = tensor->data();
    ASSERT_NE(data, nullptr);

    for (size_t i = 0; i < N * K; ++i)
    {
        EXPECT_TRUE(std::isfinite(data[i]));
    }
}

// ============================================================================
// CONTROL TESTS: Full blocks (should work regardless of fix)
// ============================================================================

/**
 * @brief Test K=32 (exactly one full block) - baseline
 */
TEST_F(Test__Q8_0Tensor_PartialBlocks, K32_ExactlyOneBlock)
{
    const size_t N = 10;
    const size_t K = 32;

    auto tensor = createTestTensor(N, K, 1.0f, 1);
    const float *data = tensor->data();
    ASSERT_NE(data, nullptr);

    for (size_t i = 0; i < N * K; ++i)
    {
        EXPECT_TRUE(std::isfinite(data[i]));
    }
}

/**
 * @brief Test K=64 (exactly two full blocks) - baseline
 */
TEST_F(Test__Q8_0Tensor_PartialBlocks, K64_ExactlyTwoBlocks)
{
    const size_t N = 8;
    const size_t K = 64;

    auto tensor = createTestTensor(N, K, 0.5f, 1);
    const float *data = tensor->data();
    ASSERT_NE(data, nullptr);

    for (size_t i = 0; i < N * K; ++i)
    {
        EXPECT_TRUE(std::isfinite(data[i]));
    }
}

/**
 * @brief Test K=128 (exactly four full blocks) - baseline
 */
TEST_F(Test__Q8_0Tensor_PartialBlocks, K128_ExactlyFourBlocks)
{
    const size_t N = 5;
    const size_t K = 128;

    auto tensor = createTestTensor(N, K, 0.25f, 1);
    const float *data = tensor->data();
    ASSERT_NE(data, nullptr);

    for (size_t i = 0; i < N * K; ++i)
    {
        EXPECT_TRUE(std::isfinite(data[i]));
    }
}

// ============================================================================
// VALUE CORRECTNESS TESTS
// ============================================================================

/**
 * @brief Verify that partial block values are correctly decoded (not garbage)
 */
TEST_F(Test__Q8_0Tensor_PartialBlocks, ValueCorrectness_PartialBlock)
{
    const size_t N = 2;
    const size_t K = 17; // Partial block

    // Create tensor with scale=1.0 and simple fill pattern
    constexpr size_t BLOCK_SIZE = 32;
    size_t blocks_per_row = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t total_blocks = N * blocks_per_row;
    size_t raw_size = total_blocks * sizeof(Q8_0Block);

    std::vector<uint8_t> raw_data(raw_size);
    Q8_0Block *blocks = reinterpret_cast<Q8_0Block *>(raw_data.data());

    // Set scale = 1.0 (d in FP16)
    uint16_t d_fp16 = 0x3C00; // 1.0 in FP16

    // Fill with sequential values 0, 1, 2, ...
    for (size_t block_idx = 0; block_idx < total_blocks; ++block_idx)
    {
        blocks[block_idx].d = d_fp16;
        for (size_t i = 0; i < 32; ++i)
        {
            blocks[block_idx].qs[i] = static_cast<int8_t>(i);
        }
    }

    auto tensor = std::make_unique<Q8_0Tensor>(std::vector<size_t>{N, K}, raw_data);
    const float *data = tensor->data();
    ASSERT_NE(data, nullptr);

    // Check row 0: should have values 0, 1, 2, ..., 16
    for (size_t col = 0; col < K; ++col)
    {
        float expected = static_cast<float>(col); // scale=1.0, qs[col]=col
        EXPECT_NEAR(data[col], expected, 0.01f)
            << "Row 0, Col " << col << ": expected " << expected << ", got " << data[col];
    }

    // Check row 1: same pattern
    for (size_t col = 0; col < K; ++col)
    {
        float expected = static_cast<float>(col);
        EXPECT_NEAR(data[K + col], expected, 0.01f)
            << "Row 1, Col " << col << ": expected " << expected << ", got " << data[K + col];
    }
}

/**
 * @brief Verify cached data is consistent across multiple data() calls
 */
TEST_F(Test__Q8_0Tensor_PartialBlocks, CacheConsistency)
{
    const size_t N = 7;
    const size_t K = 23; // Partial block

    auto tensor = createTestTensor(N, K, 0.5f, 1);

    // First call
    const float *data1 = tensor->data();
    ASSERT_NE(data1, nullptr);

    // Second call should return same pointer (cached)
    const float *data2 = tensor->data();
    EXPECT_EQ(data1, data2) << "Cache should return same pointer";

    // Values should be identical
    for (size_t i = 0; i < N * K; ++i)
    {
        EXPECT_EQ(data1[i], data2[i]) << "Cache values differ at index " << i;
    }
}

/**
 * @brief Stress test with many different K values
 */
TEST_F(Test__Q8_0Tensor_PartialBlocks, StressTest_VariousKValues)
{
    const size_t N = 10;

    // Test many K values including edge cases
    std::vector<size_t> k_values = {
        1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 47, 48, 49,
        63, 64, 65, 95, 96, 97, 127, 128, 129, 255, 256, 257};

    for (size_t K : k_values)
    {
        auto tensor = createTestTensor(N, K, 0.1f, 1);
        const float *data = tensor->data();
        ASSERT_NE(data, nullptr) << "data() returned null for K=" << K;

        // Verify no crashes and all values finite
        for (size_t i = 0; i < N * K; ++i)
        {
            ASSERT_TRUE(std::isfinite(data[i]))
                << "Non-finite value at index " << i << " for K=" << K;
        }
    }
}
