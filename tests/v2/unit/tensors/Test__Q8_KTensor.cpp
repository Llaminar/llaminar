/**
 * @file Test__Q8_KTensor.cpp
 * @brief Unit tests for Q8_K SIMD equivalency (scalar vs AVX2 vs AVX512)
 * @author David Sanftenberg
 * @date 2025-10-25
 *
 * Q8_K SIMD Equivalency Tests:
 * - ScalarVsAVX2Equivalency: Scalar vs AVX2 decode (8-wide processing)
 * - ScalarVsAVX512Equivalency: Scalar vs AVX512 decode (16-wide processing)
 * - AVX2VsAVX512Equivalency: AVX2 vs AVX512 decode (cross-validation)
 *
 * Edge Case Tests:
 * - EdgeCase_AllZeros: All int8 values = 0 → all float outputs = 0.0f
 * - EdgeCase_AllPositiveMax: All int8 values = 127 → all float outputs = 127.0f
 * - EdgeCase_AllNegativeMin: All int8 values = -128 → all float outputs = -128.0f
 * - EdgeCase_AlternatingSign: Positive/negative pattern (-128, 127, -128, 127, ...)
 * - EdgeCase_ChunkBoundaries: Test 8-element and 16-element boundaries (AVX2/AVX512 alignment)
 * - EdgeCase_RandomValues: Random int8 values across full range [-128, 127]
 *
 * Q8_K Block Structure:
 * - Block size: 256 elements (16 AVX512 chunks, 32 AVX2 chunks)
 * - Data: int8_t qs[256] (8-bit signed integers)
 * - Additional: int16_t bsums[16] (block sums for fast dot products, NOT used in decode)
 * - Total: 288 bytes (256 + 32)
 *
 * Q8_K Decode Formula:
 *   output[i] = (float)qs[i]
 *
 * This is the SIMPLEST quantization variant - just int8→float type conversion!
 * No scale factors, no bit unpacking, no lookup tables, no masks.
 *
 * SIMD Implementation:
 * - Scalar: Simple loop with static_cast<float>
 * - AVX2: Process 8 int8 at a time
 *   1. Load 8 int8 values (_mm_loadl_epi64)
 *   2. Sign-extend to int32 (_mm256_cvtepi8_epi32)
 *   3. Convert to float (_mm256_cvtepi32_ps)
 *   4. Store 8 floats (_mm256_storeu_ps)
 * - AVX512: Process 16 int8 at a time
 *   1. Load 16 int8 values (_mm_loadu_si128)
 *   2. Sign-extend to int32 (_mm512_cvtepi8_epi32)
 *   3. Convert to float (_mm512_cvtepi32_ps)
 *   4. Store 16 floats (_mm512_storeu_ps)
 *
 * Expected Performance:
 * - AVX512: 16 iterations (16 values per iteration)
 * - AVX2: 32 iterations (8 values per iteration)
 * - Scalar: 256 iterations (1 value per iteration)
 * - Speedup: AVX512 ~16×, AVX2 ~8× over scalar
 */

#include "tensors/Tensors.h"
#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <random>
#include <cmath>

using namespace llaminar2;

// Tolerance: Very strict since this is just int8→float conversion (exact)
constexpr float TOLERANCE = 1e-6f;

class Q8_KSIMDTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Nothing to set up - tests create blocks on the fly
    }

    void TearDown() override {}

    /**
     * @brief Create a Q8_KBlock with specified int8 values
     * @param values 256 int8_t values for qs[] array
     * @return Initialized Q8_KBlock
     */
    Q8_KBlock createBlock(const std::vector<int8_t> &values)
    {
        Q8_KBlock block;
        std::memset(&block, 0, sizeof(Q8_KBlock));

        // Copy values into qs[] (must be exactly 256 elements)
        assert(values.size() == 256 && "Q8_K block requires 256 elements");
        std::memcpy(block.qs, values.data(), 256);

        // bsums[] not used in decode - leave as zeros
        return block;
    }

    /**
     * @brief Compare two float arrays with tolerance
     * @param expected Expected float values
     * @param actual Actual float values
     * @param count Number of elements to compare
     * @param tolerance Maximum allowed absolute difference
     * @return true if all elements match within tolerance
     */
    bool compareArrays(const float *expected, const float *actual, size_t count, float tolerance)
    {
        for (size_t i = 0; i < count; ++i)
        {
            float diff = std::abs(expected[i] - actual[i]);
            if (diff > tolerance)
            {
                std::cerr << "Mismatch at index " << i << ": expected " << expected[i]
                          << ", got " << actual[i] << " (diff=" << diff << ")" << std::endl;
                return false;
            }
        }
        return true;
    }
};

// ===== SIMD Equivalency Tests =====

#ifdef __AVX2__
TEST_F(Q8_KSIMDTest, ScalarVsAVX2Equivalency)
{
    // Create test block with varying int8 values
    std::vector<int8_t> values(256);
    for (size_t i = 0; i < 256; ++i)
    {
        values[i] = static_cast<int8_t>((i % 256) - 128); // Range: -128 to 127
    }
    Q8_KBlock block = createBlock(values);

    // Decode with scalar and AVX2
    std::vector<float> scalar_output(256);
    std::vector<float> avx2_output(256);

    Q8_KTensor::decodeBlockScalar(block, scalar_output.data());
    Q8_KTensor::decodeBlockAVX2(block, avx2_output.data());

    // Compare results (should be identical - exact conversion)
    EXPECT_TRUE(compareArrays(scalar_output.data(), avx2_output.data(), 256, TOLERANCE))
        << "Scalar vs AVX2 mismatch";
}
#endif // __AVX2__

#ifdef __AVX512F__
TEST_F(Q8_KSIMDTest, ScalarVsAVX512Equivalency)
{
    // Create test block with varying int8 values
    std::vector<int8_t> values(256);
    for (size_t i = 0; i < 256; ++i)
    {
        values[i] = static_cast<int8_t>((i % 256) - 128); // Range: -128 to 127
    }
    Q8_KBlock block = createBlock(values);

    // Decode with scalar and AVX512
    std::vector<float> scalar_output(256);
    std::vector<float> avx512_output(256);

    Q8_KTensor::decodeBlockScalar(block, scalar_output.data());
    Q8_KTensor::decodeBlockAVX512(block, avx512_output.data());

    // Compare results (should be identical - exact conversion)
    EXPECT_TRUE(compareArrays(scalar_output.data(), avx512_output.data(), 256, TOLERANCE))
        << "Scalar vs AVX512 mismatch";
}
#endif // __AVX512F__

#if defined(__AVX2__) && defined(__AVX512F__)
TEST_F(Q8_KSIMDTest, AVX2VsAVX512Equivalency)
{
    // Create test block with varying int8 values
    std::vector<int8_t> values(256);
    for (size_t i = 0; i < 256; ++i)
    {
        values[i] = static_cast<int8_t>((i % 256) - 128); // Range: -128 to 127
    }
    Q8_KBlock block = createBlock(values);

    // Decode with AVX2 and AVX512
    std::vector<float> avx2_output(256);
    std::vector<float> avx512_output(256);

    Q8_KTensor::decodeBlockAVX2(block, avx2_output.data());
    Q8_KTensor::decodeBlockAVX512(block, avx512_output.data());

    // Compare results (should be identical - exact conversion)
    EXPECT_TRUE(compareArrays(avx2_output.data(), avx512_output.data(), 256, TOLERANCE))
        << "AVX2 vs AVX512 mismatch";
}
#endif // __AVX2__ && __AVX512F__

// ===== Edge Case Tests =====

TEST_F(Q8_KSIMDTest, EdgeCase_AllZeros)
{
    // All int8 values = 0 → all float outputs = 0.0f
    std::vector<int8_t> values(256, 0);
    Q8_KBlock block = createBlock(values);

    std::vector<float> output(256);
    Q8_KTensor::decodeBlock(block, output.data());

    // All outputs should be exactly 0.0f
    for (size_t i = 0; i < 256; ++i)
    {
        EXPECT_FLOAT_EQ(output[i], 0.0f) << "Index " << i << " should be 0.0f";
    }
}

TEST_F(Q8_KSIMDTest, EdgeCase_AllPositiveMax)
{
    // All int8 values = 127 (max positive) → all float outputs = 127.0f
    std::vector<int8_t> values(256, 127);
    Q8_KBlock block = createBlock(values);

    std::vector<float> output(256);
    Q8_KTensor::decodeBlock(block, output.data());

    // All outputs should be exactly 127.0f
    for (size_t i = 0; i < 256; ++i)
    {
        EXPECT_FLOAT_EQ(output[i], 127.0f) << "Index " << i << " should be 127.0f";
    }
}

TEST_F(Q8_KSIMDTest, EdgeCase_AllNegativeMin)
{
    // All int8 values = -128 (min negative) → all float outputs = -128.0f
    std::vector<int8_t> values(256, -128);
    Q8_KBlock block = createBlock(values);

    std::vector<float> output(256);
    Q8_KTensor::decodeBlock(block, output.data());

    // All outputs should be exactly -128.0f
    for (size_t i = 0; i < 256; ++i)
    {
        EXPECT_FLOAT_EQ(output[i], -128.0f) << "Index " << i << " should be -128.0f";
    }
}

TEST_F(Q8_KSIMDTest, EdgeCase_AlternatingSign)
{
    // Alternating positive/negative pattern: -128, 127, -128, 127, ...
    std::vector<int8_t> values(256);
    for (size_t i = 0; i < 256; ++i)
    {
        values[i] = (i % 2 == 0) ? -128 : 127;
    }
    Q8_KBlock block = createBlock(values);

    std::vector<float> output(256);
    Q8_KTensor::decodeBlock(block, output.data());

    // Check alternating pattern
    for (size_t i = 0; i < 256; ++i)
    {
        float expected = (i % 2 == 0) ? -128.0f : 127.0f;
        EXPECT_FLOAT_EQ(output[i], expected) << "Index " << i << " should be " << expected;
    }
}

TEST_F(Q8_KSIMDTest, EdgeCase_ChunkBoundaries)
{
    // Test AVX2 (8-element) and AVX512 (16-element) boundaries
    // Pattern: First 16 elements = -128, next 16 = 127, repeat
    std::vector<int8_t> values(256);
    for (size_t i = 0; i < 256; ++i)
    {
        values[i] = ((i / 16) % 2 == 0) ? -128 : 127;
    }
    Q8_KBlock block = createBlock(values);

    std::vector<float> output(256);
    Q8_KTensor::decodeBlock(block, output.data());

    // Verify pattern
    for (size_t i = 0; i < 256; ++i)
    {
        float expected = ((i / 16) % 2 == 0) ? -128.0f : 127.0f;
        EXPECT_FLOAT_EQ(output[i], expected) << "Index " << i << " should be " << expected;
    }
}

TEST_F(Q8_KSIMDTest, EdgeCase_RandomValues)
{
    // Random int8 values across full range [-128, 127]
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(-128, 127);

    std::vector<int8_t> values(256);
    for (size_t i = 0; i < 256; ++i)
    {
        values[i] = static_cast<int8_t>(dist(gen));
    }
    Q8_KBlock block = createBlock(values);

    std::vector<float> output(256);
    Q8_KTensor::decodeBlock(block, output.data());

    // Verify each output matches input (exact int→float conversion)
    for (size_t i = 0; i < 256; ++i)
    {
        float expected = static_cast<float>(values[i]);
        EXPECT_FLOAT_EQ(output[i], expected) << "Index " << i << " should be " << expected;
    }
}

// ===== Main =====
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
