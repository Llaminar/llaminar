/**
 * @file Test__Q8_0Tensor.cpp
 * @brief SIMD equivalency tests for Q8_0 tensor dequantization
 *
 * Tests that scalar, AVX2, and AVX512 implementations of Q8_0 block decoding
 * produce identical results.
 *
 * Q8_0 Format:
 * - Block size: 32 elements
 * - Quantization: 8-bit signed integers (-128 to 127)
 * - Storage: 2-byte FP16 scale (d) + 32 bytes of int8 values
 * - Dequant formula: output[i] = d * qs[i]
 *
 * @author David Sanftenberg
 * @date October 29, 2025
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <random>
#include "../../../../src/v2/tensors/Tensors.h"

using namespace llaminar2;

class Test__Q8_0Tensor : public ::testing::Test
{
protected:
    static constexpr float TOLERANCE = 1e-5f;
    static constexpr size_t BLOCK_SIZE = Q8_0Block::BLOCK_SIZE; // 32 elements

    /**
     * @brief Compare two float arrays for approximate equality
     */
    bool compareArrays(const float *arr1, const float *arr2, size_t count, float tolerance = TOLERANCE)
    {
        for (size_t i = 0; i < count; ++i)
        {
            float diff = std::abs(arr1[i] - arr2[i]);
            if (diff > tolerance)
            {
                std::cerr << "Mismatch at index " << i << ": "
                          << arr1[i] << " != " << arr2[i]
                          << " (diff = " << diff << ")" << std::endl;
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Create a Q8_0 block with specified scale and int8 values
     */
    Q8_0Block createBlock(float scale_fp32, const int8_t *values)
    {
        Q8_0Block block;
        block.d = fp32_to_fp16(scale_fp32);
        std::memcpy(block.qs, values, BLOCK_SIZE);
        return block;
    }
};

// ========================================================================
// SIMD Equivalency Tests
// ========================================================================

TEST_F(Test__Q8_0Tensor, ScalarVsAVX2Equivalency)
{
#if defined(__AVX2__)
    // Create test block with various int8 values
    int8_t test_values[BLOCK_SIZE];
    for (size_t i = 0; i < BLOCK_SIZE; ++i)
    {
        test_values[i] = static_cast<int8_t>(static_cast<int>(i) - 16); // Range: -16 to 15
    }

    Q8_0Block block = createBlock(0.5f, test_values);

    float output_scalar[BLOCK_SIZE];
    float output_avx2[BLOCK_SIZE];

    Q8_0Tensor::decodeBlockScalar(block, output_scalar);
    Q8_0Tensor::decodeBlockAVX2(block, output_avx2);

    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, BLOCK_SIZE))
        << "Scalar and AVX2 implementations should produce identical results";
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
#endif
}

TEST_F(Test__Q8_0Tensor, ScalarVsAVX512Equivalency)
{
#if defined(__AVX512F__)
    // Create test block with various int8 values
    int8_t test_values[BLOCK_SIZE];
    for (size_t i = 0; i < BLOCK_SIZE; ++i)
    {
        test_values[i] = static_cast<int8_t>(static_cast<int>(i) - 16);
    }

    Q8_0Block block = createBlock(0.5f, test_values);

    float output_scalar[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q8_0Tensor::decodeBlockScalar(block, output_scalar);
    Q8_0Tensor::decodeBlockAVX512(block, output_avx512);

    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, BLOCK_SIZE))
        << "Scalar and AVX512 implementations should produce identical results";
#else
    GTEST_SKIP() << "AVX512 not available on this platform";
#endif
}

TEST_F(Test__Q8_0Tensor, AVX2VsAVX512Equivalency)
{
#if defined(__AVX2__) && defined(__AVX512F__)
    // Create test block with various int8 values
    int8_t test_values[BLOCK_SIZE];
    for (size_t i = 0; i < BLOCK_SIZE; ++i)
    {
        test_values[i] = static_cast<int8_t>(static_cast<int>(i) - 16);
    }

    Q8_0Block block = createBlock(0.5f, test_values);

    float output_avx2[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q8_0Tensor::decodeBlockAVX2(block, output_avx2);
    Q8_0Tensor::decodeBlockAVX512(block, output_avx512);

    EXPECT_TRUE(compareArrays(output_avx2, output_avx512, BLOCK_SIZE))
        << "AVX2 and AVX512 implementations should produce identical results";
#else
    GTEST_SKIP() << "Both AVX2 and AVX512 required for this test";
#endif
}

// ========================================================================
// Edge Case Tests
// ========================================================================

TEST_F(Test__Q8_0Tensor, EdgeCase_ZeroScale)
{
    // All values should be zero when scale is zero
    int8_t test_values[BLOCK_SIZE];
    for (size_t i = 0; i < BLOCK_SIZE; ++i)
    {
        test_values[i] = static_cast<int8_t>(i); // Non-zero values
    }

    Q8_0Block block = createBlock(0.0f, test_values);

    float output_scalar[BLOCK_SIZE];
    float output_avx2[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q8_0Tensor::decodeBlockScalar(block, output_scalar);
#if defined(__AVX2__)
    Q8_0Tensor::decodeBlockAVX2(block, output_avx2);
#endif
#if defined(__AVX512F__)
    Q8_0Tensor::decodeBlockAVX512(block, output_avx512);
#endif

    // All outputs should be zero
    for (size_t i = 0; i < BLOCK_SIZE; ++i)
    {
        EXPECT_FLOAT_EQ(output_scalar[i], 0.0f);
#if defined(__AVX2__)
        EXPECT_FLOAT_EQ(output_avx2[i], 0.0f);
#endif
#if defined(__AVX512F__)
        EXPECT_FLOAT_EQ(output_avx512[i], 0.0f);
#endif
    }
}

TEST_F(Test__Q8_0Tensor, EdgeCase_MaxInt8Values)
{
    // Test with maximum int8 values (127 and -128)
    int8_t test_values[BLOCK_SIZE];
    for (size_t i = 0; i < BLOCK_SIZE; ++i)
    {
        test_values[i] = (i % 2 == 0) ? 127 : -128;
    }

    Q8_0Block block = createBlock(1.0f, test_values);

    float output_scalar[BLOCK_SIZE];
    float output_avx2[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q8_0Tensor::decodeBlockScalar(block, output_scalar);
#if defined(__AVX2__)
    Q8_0Tensor::decodeBlockAVX2(block, output_avx2);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, BLOCK_SIZE));
#endif
#if defined(__AVX512F__)
    Q8_0Tensor::decodeBlockAVX512(block, output_avx512);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, BLOCK_SIZE));
#endif
}

TEST_F(Test__Q8_0Tensor, EdgeCase_NegativeScale)
{
    // Q8_0 scale is FP16, which can be negative (though uncommon)
    int8_t test_values[BLOCK_SIZE];
    for (size_t i = 0; i < BLOCK_SIZE; ++i)
    {
        test_values[i] = static_cast<int8_t>(i);
    }

    Q8_0Block block = createBlock(-0.5f, test_values);

    float output_scalar[BLOCK_SIZE];
    float output_avx2[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q8_0Tensor::decodeBlockScalar(block, output_scalar);
#if defined(__AVX2__)
    Q8_0Tensor::decodeBlockAVX2(block, output_avx2);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, BLOCK_SIZE));
#endif
#if defined(__AVX512F__)
    Q8_0Tensor::decodeBlockAVX512(block, output_avx512);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, BLOCK_SIZE));
#endif
}

TEST_F(Test__Q8_0Tensor, EdgeCase_AlternatingPositiveNegative)
{
    // Test alternating positive and negative values
    int8_t test_values[BLOCK_SIZE];
    for (size_t i = 0; i < BLOCK_SIZE; ++i)
    {
        test_values[i] = (i % 2 == 0) ? static_cast<int8_t>(i) : static_cast<int8_t>(-static_cast<int>(i));
    }

    Q8_0Block block = createBlock(2.5f, test_values);

    float output_scalar[BLOCK_SIZE];
    float output_avx2[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q8_0Tensor::decodeBlockScalar(block, output_scalar);
#if defined(__AVX2__)
    Q8_0Tensor::decodeBlockAVX2(block, output_avx2);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, BLOCK_SIZE));
#endif
#if defined(__AVX512F__)
    Q8_0Tensor::decodeBlockAVX512(block, output_avx512);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, BLOCK_SIZE));
#endif
}

TEST_F(Test__Q8_0Tensor, EdgeCase_LargeScale)
{
    // Test with a large scale value
    int8_t test_values[BLOCK_SIZE];
    for (size_t i = 0; i < BLOCK_SIZE; ++i)
    {
        test_values[i] = static_cast<int8_t>(i % 64); // Small values
    }

    Q8_0Block block = createBlock(100.0f, test_values); // Large scale

    float output_scalar[BLOCK_SIZE];
    float output_avx2[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q8_0Tensor::decodeBlockScalar(block, output_scalar);
#if defined(__AVX2__)
    Q8_0Tensor::decodeBlockAVX2(block, output_avx2);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, BLOCK_SIZE));
#endif
#if defined(__AVX512F__)
    Q8_0Tensor::decodeBlockAVX512(block, output_avx512);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, BLOCK_SIZE));
#endif
}

TEST_F(Test__Q8_0Tensor, EdgeCase_RandomValues)
{
    // Test with random values for comprehensive coverage
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_int_distribution<int> dist(-128, 127);
    std::uniform_real_distribution<float> scale_dist(0.001f, 10.0f);

    int8_t test_values[BLOCK_SIZE];
    for (size_t i = 0; i < BLOCK_SIZE; ++i)
    {
        test_values[i] = static_cast<int8_t>(dist(rng));
    }

    Q8_0Block block = createBlock(scale_dist(rng), test_values);

    float output_scalar[BLOCK_SIZE];
    float output_avx2[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q8_0Tensor::decodeBlockScalar(block, output_scalar);
#if defined(__AVX2__)
    Q8_0Tensor::decodeBlockAVX2(block, output_avx2);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, BLOCK_SIZE));
#endif
#if defined(__AVX512F__)
    Q8_0Tensor::decodeBlockAVX512(block, output_avx512);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, BLOCK_SIZE));
#endif
}
