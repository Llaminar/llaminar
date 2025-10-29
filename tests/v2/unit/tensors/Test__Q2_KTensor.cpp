/**
 * @file Test__Q2_KTensor.cpp
 * @brief Unit tests for Q2_K SIMD equivalency
 * @author David Sanftenberg
 *
 * Tests that scalar, AVX2, and AVX512 implementations of Q2_K dequantization
 * produce identical numerical results.
 */

#include <gtest/gtest.h>
#include "../../../../src/v2/tensors/Tensors.h"
#include <vector>
#include <cmath>
#include <cstring>

using namespace llaminar2;

class Test__Q2_KTensor : public ::testing::Test
{
protected:
    // Helper to compare two float arrays with tolerance
    bool compareOutputs(const float *a, const float *b, size_t count, float tolerance = 1e-5f)
    {
        for (size_t i = 0; i < count; ++i)
        {
            if (std::fabs(a[i] - b[i]) > tolerance)
            {
                std::cerr << "Mismatch at index " << i << ": " << a[i] << " != " << b[i]
                          << " (diff = " << std::fabs(a[i] - b[i]) << ")" << std::endl;
                return false;
            }
        }
        return true;
    }
};

// ============================================================================
// SIMD Equivalency Tests
// ============================================================================

#if defined(__AVX2__)
TEST_F(Test__Q2_KTensor, ScalarVsAVX2Equivalency)
{
    Q2_KBlock test_block;
    test_block.d = 0x3C00;    // FP16 1.0
    test_block.dmin = 0x3800; // FP16 0.5

    // Set scales (4 bits each for scale and min)
    for (int i = 0; i < 16; ++i)
    {
        test_block.scales[i] = 0x85; // scale=5, min=8
    }

    // Set 2-bit quantized values (pattern: 0, 1, 2, 3 repeating)
    for (int i = 0; i < 64; ++i)
    {
        test_block.qs[i] = 0xE4; // 11 10 01 00 in binary = 3,2,1,0
    }

    std::vector<float> scalar_output(Q2_KBlock::BLOCK_SIZE);
    std::vector<float> avx2_output(Q2_KBlock::BLOCK_SIZE);

    Q2_KTensor::decodeBlockScalar(test_block, scalar_output.data());
    Q2_KTensor::decodeBlockAVX2(test_block, avx2_output.data());

    EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                               Q2_KBlock::BLOCK_SIZE, 1e-5f));
}
#endif

#if defined(__AVX512F__)
TEST_F(Test__Q2_KTensor, ScalarVsAVX512Equivalency)
{
    Q2_KBlock test_block;
    test_block.d = 0x3C00;    // FP16 1.0
    test_block.dmin = 0x3800; // FP16 0.5

    for (int i = 0; i < 16; ++i)
    {
        test_block.scales[i] = 0x85; // scale=5, min=8
    }

    for (int i = 0; i < 64; ++i)
    {
        test_block.qs[i] = 0xE4; // Pattern: 3,2,1,0
    }

    std::vector<float> scalar_output(Q2_KBlock::BLOCK_SIZE);
    std::vector<float> avx512_output(Q2_KBlock::BLOCK_SIZE);

    Q2_KTensor::decodeBlockScalar(test_block, scalar_output.data());
    Q2_KTensor::decodeBlockAVX512(test_block, avx512_output.data());

    EXPECT_TRUE(compareOutputs(scalar_output.data(), avx512_output.data(),
                               Q2_KBlock::BLOCK_SIZE, 1e-5f));
}
#endif

#if defined(__AVX2__) && defined(__AVX512F__)
TEST_F(Test__Q2_KTensor, AVX2VsAVX512Equivalency)
{
    Q2_KBlock test_block;
    test_block.d = 0x3C00;    // FP16 1.0
    test_block.dmin = 0x3800; // FP16 0.5

    for (int i = 0; i < 16; ++i)
    {
        test_block.scales[i] = 0x85;
    }

    for (int i = 0; i < 64; ++i)
    {
        test_block.qs[i] = 0xE4;
    }

    std::vector<float> avx2_output(Q2_KBlock::BLOCK_SIZE);
    std::vector<float> avx512_output(Q2_KBlock::BLOCK_SIZE);

    Q2_KTensor::decodeBlockAVX2(test_block, avx2_output.data());
    Q2_KTensor::decodeBlockAVX512(test_block, avx512_output.data());

    EXPECT_TRUE(compareOutputs(avx2_output.data(), avx512_output.data(),
                               Q2_KBlock::BLOCK_SIZE, 1e-5f));
}
#endif

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(Test__Q2_KTensor, EdgeCase_ZeroScales)
{
    Q2_KBlock test_block;
    test_block.d = 0x3C00;    // FP16 1.0
    test_block.dmin = 0x3800; // FP16 0.5

    // All scales zero
    for (int i = 0; i < 16; ++i)
    {
        test_block.scales[i] = 0x00;
    }

    for (int i = 0; i < 64; ++i)
    {
        test_block.qs[i] = 0xFF;
    }

    std::vector<float> scalar_output(Q2_KBlock::BLOCK_SIZE);

#if defined(__AVX2__)
    std::vector<float> avx2_output(Q2_KBlock::BLOCK_SIZE);
    Q2_KTensor::decodeBlockScalar(test_block, scalar_output.data());
    Q2_KTensor::decodeBlockAVX2(test_block, avx2_output.data());

    // With zero scales, all outputs should be zero (or -min)
    EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                               Q2_KBlock::BLOCK_SIZE, 1e-5f));
#endif
}

TEST_F(Test__Q2_KTensor, EdgeCase_Max2BitValues)
{
    Q2_KBlock test_block;
    test_block.d = 0x3C00;    // FP16 1.0
    test_block.dmin = 0x3800; // FP16 0.5

    for (int i = 0; i < 16; ++i)
    {
        test_block.scales[i] = 0xFF; // max scale and min (15 each)
    }

    // All 2-bit values = 3 (maximum)
    for (int i = 0; i < 64; ++i)
    {
        test_block.qs[i] = 0xFF; // 11 11 11 11 = all 3's
    }

    std::vector<float> scalar_output(Q2_KBlock::BLOCK_SIZE);

#if defined(__AVX2__)
    std::vector<float> avx2_output(Q2_KBlock::BLOCK_SIZE);
    Q2_KTensor::decodeBlockScalar(test_block, scalar_output.data());
    Q2_KTensor::decodeBlockAVX2(test_block, avx2_output.data());

    EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                               Q2_KBlock::BLOCK_SIZE, 1e-5f));
#endif
}

TEST_F(Test__Q2_KTensor, EdgeCase_SuperBlockScales)
{
    Q2_KBlock test_block;
    test_block.d = 0x4000;    // FP16 2.0 (higher super-block scale)
    test_block.dmin = 0x3000; // FP16 0.25 (lower min scale)

    // Varying scales across groups
    for (int i = 0; i < 16; ++i)
    {
        test_block.scales[i] = (i << 4) | (15 - i); // Varying pattern
    }

    for (int i = 0; i < 64; ++i)
    {
        test_block.qs[i] = 0x1B; // 00 01 10 11 pattern
    }

    std::vector<float> scalar_output(Q2_KBlock::BLOCK_SIZE);

#if defined(__AVX2__)
    std::vector<float> avx2_output(Q2_KBlock::BLOCK_SIZE);
    Q2_KTensor::decodeBlockScalar(test_block, scalar_output.data());
    Q2_KTensor::decodeBlockAVX2(test_block, avx2_output.data());

    EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                               Q2_KBlock::BLOCK_SIZE, 1e-5f));
#endif
}

TEST_F(Test__Q2_KTensor, EdgeCase_ChunkBoundary)
{
    // Test the boundary between the 2 chunks of 128 elements
    Q2_KBlock test_block;
    test_block.d = 0x3C00;    // FP16 1.0
    test_block.dmin = 0x3800; // FP16 0.5

    // Different scales for first and second chunk
    for (int i = 0; i < 8; ++i)
    {
        test_block.scales[i] = 0x3C; // First chunk: scale=12, min=3
    }
    for (int i = 8; i < 16; ++i)
    {
        test_block.scales[i] = 0xA5; // Second chunk: scale=5, min=10
    }

    for (int i = 0; i < 64; ++i)
    {
        test_block.qs[i] = (i & 1) ? 0xAA : 0x55; // Alternating pattern
    }

    std::vector<float> scalar_output(Q2_KBlock::BLOCK_SIZE);

#if defined(__AVX2__)
    std::vector<float> avx2_output(Q2_KBlock::BLOCK_SIZE);
    Q2_KTensor::decodeBlockScalar(test_block, scalar_output.data());
    Q2_KTensor::decodeBlockAVX2(test_block, avx2_output.data());

    EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                               Q2_KBlock::BLOCK_SIZE, 1e-5f));
#endif
}

TEST_F(Test__Q2_KTensor, EdgeCase_ShiftPattern)
{
    // Test all 4 shift positions (0, 2, 4, 6 bits)
    Q2_KBlock test_block;
    test_block.d = 0x3C00;    // FP16 1.0
    test_block.dmin = 0x3800; // FP16 0.5

    for (int i = 0; i < 16; ++i)
    {
        test_block.scales[i] = 0x77; // scale=7, min=7
    }

    // Each byte has different 2-bit values at each shift position
    for (int i = 0; i < 64; ++i)
    {
        test_block.qs[i] = 0x1B; // 00 01 10 11 (tests all shift patterns)
    }

    std::vector<float> scalar_output(Q2_KBlock::BLOCK_SIZE);

#if defined(__AVX2__)
    std::vector<float> avx2_output(Q2_KBlock::BLOCK_SIZE);
    Q2_KTensor::decodeBlockScalar(test_block, scalar_output.data());
    Q2_KTensor::decodeBlockAVX2(test_block, avx2_output.data());

    // Verify the shift pattern is correctly extracted
    EXPECT_TRUE(compareOutputs(scalar_output.data(), avx2_output.data(),
                               Q2_KBlock::BLOCK_SIZE, 1e-5f));
#endif
}
