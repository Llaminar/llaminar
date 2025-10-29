/**
 * @file Test__Q3_KTensor.cpp
 * @brief SIMD equivalency tests for Q3_K tensor dequantization
 *
 * Tests that scalar, AVX2, and AVX512 implementations of Q3_K block decoding
 * produce identical results.
 *
 * Q3_K Format:
 * - Block size: 256 elements (super-block)
 * - Quantization: 3-bit values (2 bits in qs[] + 1 high bit in hmask[])
 * - Storage: 32-byte hmask + 64-byte qs + 12-byte scales + 2-byte FP16 d
 * - Dequant formula: output[i] = d * scale * (low_bits - (high_bit ? 0 : 4))
 * - Scale unpacking: 16 scales (6 bits each) packed into 12 bytes
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

class Test__Q3_KTensor : public ::testing::Test
{
protected:
    static constexpr float TOLERANCE = 1e-4f;                   // Slightly relaxed for complex Q3_K math
    static constexpr size_t BLOCK_SIZE = Q3_KBlock::BLOCK_SIZE; // 256 elements

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
     * @brief Create a Q3_K block with specified parameters
     *
     * Q3_K uses 3-bit values: 2 low bits in qs[], 1 high bit in hmask[]
     * Formula: d * scale * (low_bits - (high_bit ? 0 : 4))
     */
    Q3_KBlock createBlock(float d_fp32, const uint8_t *hmask, const uint8_t *qs, const uint8_t *scales)
    {
        Q3_KBlock block;
        block.d = fp32_to_fp16(d_fp32);
        std::memcpy(block.hmask, hmask, 32);
        std::memcpy(block.qs, qs, 64);
        std::memcpy(block.scales, scales, 12);
        return block;
    }

    /**
     * @brief Create simple test data with predictable patterns
     */
    void createSimpleTestData(uint8_t *hmask, uint8_t *qs, uint8_t *scales)
    {
        // Simple pattern: alternating low bits and high bits
        for (size_t i = 0; i < 32; ++i)
        {
            hmask[i] = (i % 2 == 0) ? 0xFF : 0x00; // Alternating all on/off
        }

        for (size_t i = 0; i < 64; ++i)
        {
            // Each byte has 4 2-bit values at shifts 0,2,4,6
            // Pattern: 0b11100100 = values 0,1,2,3 at different shifts
            qs[i] = 0xE4;
        }

        // Scales: pack 16 6-bit values into 12 bytes
        // For simplicity, use pattern that unpacks to known values
        for (size_t i = 0; i < 12; ++i)
        {
            scales[i] = 0x55; // Pattern of 01010101
        }
    }
};

// ========================================================================
// SIMD Equivalency Tests
// ========================================================================

TEST_F(Test__Q3_KTensor, ScalarVsAVX2Equivalency)
{
#if defined(__AVX2__)
    uint8_t hmask[32], qs[64], scales[12];
    createSimpleTestData(hmask, qs, scales);

    Q3_KBlock block = createBlock(0.5f, hmask, qs, scales);

    float output_scalar[BLOCK_SIZE];
    float output_avx2[BLOCK_SIZE];

    Q3_KTensor::decodeBlockScalar(block, output_scalar);
    Q3_KTensor::decodeBlockAVX2(block, output_avx2);

    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, BLOCK_SIZE))
        << "Scalar and AVX2 implementations should produce identical results";
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
#endif
}

TEST_F(Test__Q3_KTensor, ScalarVsAVX512Equivalency)
{
#if defined(__AVX512F__)
    uint8_t hmask[32], qs[64], scales[12];
    createSimpleTestData(hmask, qs, scales);

    Q3_KBlock block = createBlock(0.5f, hmask, qs, scales);

    float output_scalar[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q3_KTensor::decodeBlockScalar(block, output_scalar);
    Q3_KTensor::decodeBlockAVX512(block, output_avx512);

    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, BLOCK_SIZE))
        << "Scalar and AVX512 implementations should produce identical results";
#else
    GTEST_SKIP() << "AVX512 not available on this platform";
#endif
}

TEST_F(Test__Q3_KTensor, AVX2VsAVX512Equivalency)
{
#if defined(__AVX2__) && defined(__AVX512F__)
    uint8_t hmask[32], qs[64], scales[12];
    createSimpleTestData(hmask, qs, scales);

    Q3_KBlock block = createBlock(0.5f, hmask, qs, scales);

    float output_avx2[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q3_KTensor::decodeBlockAVX2(block, output_avx2);
    Q3_KTensor::decodeBlockAVX512(block, output_avx512);

    EXPECT_TRUE(compareArrays(output_avx2, output_avx512, BLOCK_SIZE))
        << "AVX2 and AVX512 implementations should produce identical results";
#else
    GTEST_SKIP() << "Both AVX2 and AVX512 required for this test";
#endif
}

// ========================================================================
// Edge Case Tests
// ========================================================================

TEST_F(Test__Q3_KTensor, EdgeCase_ZeroScale)
{
    // All outputs should be zero when d is zero
    uint8_t hmask[32], qs[64], scales[12];
    createSimpleTestData(hmask, qs, scales);

    Q3_KBlock block = createBlock(0.0f, hmask, qs, scales);

    float output_scalar[BLOCK_SIZE];
    float output_avx2[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q3_KTensor::decodeBlockScalar(block, output_scalar);
#if defined(__AVX2__)
    Q3_KTensor::decodeBlockAVX2(block, output_avx2);
#endif
#if defined(__AVX512F__)
    Q3_KTensor::decodeBlockAVX512(block, output_avx512);
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

TEST_F(Test__Q3_KTensor, EdgeCase_AllHighBitsSet)
{
    // Test with all high bits in hmask set to 1
    uint8_t hmask[32], qs[64], scales[12];

    for (size_t i = 0; i < 32; ++i)
        hmask[i] = 0xFF; // All high bits set
    for (size_t i = 0; i < 64; ++i)
        qs[i] = 0xE4;
    for (size_t i = 0; i < 12; ++i)
        scales[i] = 0x55;

    Q3_KBlock block = createBlock(1.0f, hmask, qs, scales);

    float output_scalar[BLOCK_SIZE];
    float output_avx2[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q3_KTensor::decodeBlockScalar(block, output_scalar);
#if defined(__AVX2__)
    Q3_KTensor::decodeBlockAVX2(block, output_avx2);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, BLOCK_SIZE));
#endif
#if defined(__AVX512F__)
    Q3_KTensor::decodeBlockAVX512(block, output_avx512);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, BLOCK_SIZE));
#endif
}

TEST_F(Test__Q3_KTensor, EdgeCase_AllHighBitsClear)
{
    // Test with all high bits in hmask cleared to 0
    uint8_t hmask[32], qs[64], scales[12];

    for (size_t i = 0; i < 32; ++i)
        hmask[i] = 0x00; // All high bits clear
    for (size_t i = 0; i < 64; ++i)
        qs[i] = 0xE4;
    for (size_t i = 0; i < 12; ++i)
        scales[i] = 0x55;

    Q3_KBlock block = createBlock(1.0f, hmask, qs, scales);

    float output_scalar[BLOCK_SIZE];
    float output_avx2[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q3_KTensor::decodeBlockScalar(block, output_scalar);
#if defined(__AVX2__)
    Q3_KTensor::decodeBlockAVX2(block, output_avx2);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, BLOCK_SIZE));
#endif
#if defined(__AVX512F__)
    Q3_KTensor::decodeBlockAVX512(block, output_avx512);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, BLOCK_SIZE));
#endif
}

TEST_F(Test__Q3_KTensor, EdgeCase_MaxLowBits)
{
    // Test with maximum low bit values (0b11 = 3)
    uint8_t hmask[32], qs[64], scales[12];

    for (size_t i = 0; i < 32; ++i)
        hmask[i] = 0xAA; // Alternating pattern
    for (size_t i = 0; i < 64; ++i)
        qs[i] = 0xFF; // All low bits set to 11
    for (size_t i = 0; i < 12; ++i)
        scales[i] = 0x55;

    Q3_KBlock block = createBlock(2.5f, hmask, qs, scales);

    float output_scalar[BLOCK_SIZE];
    float output_avx2[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q3_KTensor::decodeBlockScalar(block, output_scalar);
#if defined(__AVX2__)
    Q3_KTensor::decodeBlockAVX2(block, output_avx2);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, BLOCK_SIZE));
#endif
#if defined(__AVX512F__)
    Q3_KTensor::decodeBlockAVX512(block, output_avx512);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, BLOCK_SIZE));
#endif
}

TEST_F(Test__Q3_KTensor, EdgeCase_ChunkBoundary)
{
    // Test transitions between 128-element chunks
    uint8_t hmask[32], qs[64], scales[12];

    // Different patterns for first and second chunk
    for (size_t i = 0; i < 16; ++i)
        hmask[i] = 0xFF;
    for (size_t i = 16; i < 32; ++i)
        hmask[i] = 0x00;

    for (size_t i = 0; i < 32; ++i)
        qs[i] = 0xE4;
    for (size_t i = 32; i < 64; ++i)
        qs[i] = 0x1B;

    for (size_t i = 0; i < 12; ++i)
        scales[i] = (i < 6) ? 0xAA : 0x55;

    Q3_KBlock block = createBlock(1.5f, hmask, qs, scales);

    float output_scalar[BLOCK_SIZE];
    float output_avx2[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q3_KTensor::decodeBlockScalar(block, output_scalar);
#if defined(__AVX2__)
    Q3_KTensor::decodeBlockAVX2(block, output_avx2);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, BLOCK_SIZE));
#endif
#if defined(__AVX512F__)
    Q3_KTensor::decodeBlockAVX512(block, output_avx512);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, BLOCK_SIZE));
#endif
}

TEST_F(Test__Q3_KTensor, EdgeCase_ScaleUnpacking)
{
    // Test various scale packing patterns to ensure correct unpacking
    uint8_t hmask[32], qs[64], scales[12];

    for (size_t i = 0; i < 32; ++i)
        hmask[i] = 0xAA;
    for (size_t i = 0; i < 64; ++i)
        qs[i] = 0xE4;

    // Different scale patterns to test unpacking logic
    scales[0] = 0x12;
    scales[1] = 0x34;
    scales[2] = 0x56;
    scales[3] = 0x78;
    scales[4] = 0x9A;
    scales[5] = 0xBC;
    scales[6] = 0xDE;
    scales[7] = 0xF0;
    scales[8] = 0x11;
    scales[9] = 0x22;
    scales[10] = 0x33;
    scales[11] = 0x44;

    Q3_KBlock block = createBlock(0.75f, hmask, qs, scales);

    float output_scalar[BLOCK_SIZE];
    float output_avx2[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q3_KTensor::decodeBlockScalar(block, output_scalar);
#if defined(__AVX2__)
    Q3_KTensor::decodeBlockAVX2(block, output_avx2);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, BLOCK_SIZE));
#endif
#if defined(__AVX512F__)
    Q3_KTensor::decodeBlockAVX512(block, output_avx512);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, BLOCK_SIZE));
#endif
}

TEST_F(Test__Q3_KTensor, EdgeCase_RandomValues)
{
    // Test with random values for comprehensive coverage
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_real_distribution<float> scale_dist(0.001f, 10.0f);

    uint8_t hmask[32], qs[64], scales[12];

    for (size_t i = 0; i < 32; ++i)
        hmask[i] = static_cast<uint8_t>(byte_dist(rng));
    for (size_t i = 0; i < 64; ++i)
        qs[i] = static_cast<uint8_t>(byte_dist(rng));
    for (size_t i = 0; i < 12; ++i)
        scales[i] = static_cast<uint8_t>(byte_dist(rng));

    Q3_KBlock block = createBlock(scale_dist(rng), hmask, qs, scales);

    float output_scalar[BLOCK_SIZE];
    float output_avx2[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q3_KTensor::decodeBlockScalar(block, output_scalar);
#if defined(__AVX2__)
    Q3_KTensor::decodeBlockAVX2(block, output_avx2);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, BLOCK_SIZE));
#endif
#if defined(__AVX512F__)
    Q3_KTensor::decodeBlockAVX512(block, output_avx512);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, BLOCK_SIZE));
#endif
}
