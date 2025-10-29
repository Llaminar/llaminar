/**
 * @file Test__Q4_KTensor.cpp
 * @brief Unit tests for Q4_K SIMD equivalency (AVX2, AVX512)
 * @author David Sanftenberg
 *
 * Tests verify that scalar, AVX2, and AVX512 implementations produce identical results.
 */

#include <gtest/gtest.h>
#include "../../../../src/v2/tensors/Tensors.h"
#include <random>
#include <cmath>
#include <cstring>

using namespace llaminar2;

class Test__Q4_KTensor : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Seed for reproducible tests
        rng_.seed(42);
    }

    // Helper: Create Q4_KBlock with specific values
    Q4_KBlock createBlock(uint16_t d_raw, uint16_t dmin_raw,
                          const uint8_t scales[12],
                          const uint8_t qs[128])
    {
        Q4_KBlock block;
        block.d = d_raw;
        block.dmin = dmin_raw;
        std::memcpy(block.scales, scales, 12);
        std::memcpy(block.qs, qs, 128);
        return block;
    }

    // Helper: Create random Q4_KBlock
    Q4_KBlock createRandomBlock()
    {
        Q4_KBlock block;

        // Random d and dmin (FP16 format, avoid extremes)
        std::uniform_int_distribution<uint16_t> fp16_dist(0x3000, 0x4000); // ~0.25 to 2.0
        block.d = fp16_dist(rng_);
        block.dmin = fp16_dist(rng_);

        // Random scales (6-bit values packed)
        std::uniform_int_distribution<uint8_t> byte_dist(0, 255);
        for (int i = 0; i < 12; ++i)
        {
            block.scales[i] = byte_dist(rng_);
        }

        // Random qs (4-bit values in each nibble)
        for (int i = 0; i < 128; ++i)
        {
            block.qs[i] = byte_dist(rng_);
        }

        return block;
    }

    // Helper: Compare two float arrays
    void compareOutputs(const float *expected, const float *actual, size_t count,
                        float tolerance = 1e-4f, const std::string &label = "")
    {
        size_t mismatches = 0;
        float max_diff = 0.0f;
        size_t first_mismatch_idx = 0;

        for (size_t i = 0; i < count; ++i)
        {
            float diff = std::abs(expected[i] - actual[i]);
            if (diff > max_diff)
            {
                max_diff = diff;
            }
            if (diff > tolerance)
            {
                if (mismatches == 0)
                {
                    first_mismatch_idx = i;
                }
                mismatches++;
            }
        }

        if (mismatches > 0)
        {
            std::cerr << label << " MISMATCH:\n";
            std::cerr << "  First mismatch at index " << first_mismatch_idx << ":\n";
            std::cerr << "    Expected: " << expected[first_mismatch_idx] << "\n";
            std::cerr << "    Actual:   " << actual[first_mismatch_idx] << "\n";
            std::cerr << "  Total mismatches: " << mismatches << "/" << count << "\n";
            std::cerr << "  Max difference: " << max_diff << "\n";
        }

        EXPECT_EQ(mismatches, 0) << label;
    }

    std::mt19937 rng_;
};

// ============================================================================
// SIMD Equivalency Tests
// ============================================================================

#if defined(__AVX2__)
TEST_F(Test__Q4_KTensor, ScalarVsAVX2Equivalency)
{
    constexpr size_t BLOCK_SIZE = 256;

    // Test with multiple random blocks
    for (int iter = 0; iter < 10; ++iter)
    {
        Q4_KBlock block = createRandomBlock();

        float scalar_output[BLOCK_SIZE];
        float avx2_output[BLOCK_SIZE];

        Q4_KTensor::decodeBlockScalar(block, scalar_output);
        Q4_KTensor::decodeBlockAVX2(block, avx2_output);

        compareOutputs(scalar_output, avx2_output, BLOCK_SIZE, 1e-4f,
                       "ScalarVsAVX2 iteration " + std::to_string(iter));
    }
}
#endif

#if defined(__AVX512F__)
TEST_F(Test__Q4_KTensor, ScalarVsAVX512Equivalency)
{
    constexpr size_t BLOCK_SIZE = 256;

    // Test with multiple random blocks
    for (int iter = 0; iter < 10; ++iter)
    {
        Q4_KBlock block = createRandomBlock();

        float scalar_output[BLOCK_SIZE];
        float avx512_output[BLOCK_SIZE];

        Q4_KTensor::decodeBlockScalar(block, scalar_output);
        Q4_KTensor::decodeBlockAVX512(block, avx512_output);

        compareOutputs(scalar_output, avx512_output, BLOCK_SIZE, 1e-4f,
                       "ScalarVsAVX512 iteration " + std::to_string(iter));
    }
}
#endif

#if defined(__AVX512F__) && defined(__AVX2__)
TEST_F(Test__Q4_KTensor, AVX2VsAVX512Equivalency)
{
    constexpr size_t BLOCK_SIZE = 256;

    // Test with multiple random blocks
    for (int iter = 0; iter < 10; ++iter)
    {
        Q4_KBlock block = createRandomBlock();

        float avx2_output[BLOCK_SIZE];
        float avx512_output[BLOCK_SIZE];

        Q4_KTensor::decodeBlockAVX2(block, avx2_output);
        Q4_KTensor::decodeBlockAVX512(block, avx512_output);

        compareOutputs(avx2_output, avx512_output, BLOCK_SIZE, 1e-4f,
                       "AVX2VsAVX512 iteration " + std::to_string(iter));
    }
}
#endif

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(Test__Q4_KTensor, EdgeCase_ZeroScale)
{
    // Test with d=0 (all outputs should be near zero after subtracting min)
    constexpr size_t BLOCK_SIZE = 256;

    uint8_t scales[12] = {0};
    uint8_t qs[128];
    std::fill_n(qs, 128, 0x77); // Arbitrary nibbles

    Q4_KBlock block = createBlock(0x0000, 0x3C00, scales, qs); // d=0, dmin=1.0

    float scalar_output[BLOCK_SIZE];
    Q4_KTensor::decodeBlockScalar(block, scalar_output);

    // With d=0, all values should be -dmin*m (negative min value)
    // Values depend on scale unpacking, but should be consistent
    float first_val = scalar_output[0];
    for (size_t i = 0; i < BLOCK_SIZE; ++i)
    {
        // All values in same subgroup should have same sign/pattern
        // Just verify scalar produces consistent results
        EXPECT_TRUE(std::isfinite(scalar_output[i])) << "Index " << i;
    }

#if defined(__AVX2__)
    float avx2_output[BLOCK_SIZE];
    Q4_KTensor::decodeBlockAVX2(block, avx2_output);
    compareOutputs(scalar_output, avx2_output, BLOCK_SIZE, 1e-4f, "ZeroScale AVX2");
#endif

#if defined(__AVX512F__)
    float avx512_output[BLOCK_SIZE];
    Q4_KTensor::decodeBlockAVX512(block, avx512_output);
    compareOutputs(scalar_output, avx512_output, BLOCK_SIZE, 1e-4f, "ZeroScale AVX512");
#endif
}

TEST_F(Test__Q4_KTensor, EdgeCase_MaxNibbles)
{
    // Test with all nibbles set to 0xF (maximum 4-bit value)
    constexpr size_t BLOCK_SIZE = 256;

    uint8_t scales[12];
    std::fill_n(scales, 12, 0x3F); // All scales = 63 (max 6-bit)

    uint8_t qs[128];
    std::fill_n(qs, 128, 0xFF); // All nibbles = 15

    Q4_KBlock block = createBlock(0x3C00, 0x3800, scales, qs); // d=1.0, dmin=0.5

    float scalar_output[BLOCK_SIZE];
    Q4_KTensor::decodeBlockScalar(block, scalar_output);

#if defined(__AVX2__)
    float avx2_output[BLOCK_SIZE];
    Q4_KTensor::decodeBlockAVX2(block, avx2_output);
    compareOutputs(scalar_output, avx2_output, BLOCK_SIZE, 1e-4f, "MaxNibbles AVX2");
#endif

#if defined(__AVX512F__)
    float avx512_output[BLOCK_SIZE];
    Q4_KTensor::decodeBlockAVX512(block, avx512_output);
    compareOutputs(scalar_output, avx512_output, BLOCK_SIZE, 1e-4f, "MaxNibbles AVX512");
#endif
}

TEST_F(Test__Q4_KTensor, EdgeCase_AlternatingNibbles)
{
    // Test with alternating nibble patterns (0x0F, 0xF0, etc.)
    constexpr size_t BLOCK_SIZE = 256;

    uint8_t scales[12];
    for (int i = 0; i < 12; ++i)
    {
        scales[i] = (i % 2 == 0) ? 0x3F : 0x00; // Alternate max/min scales
    }

    uint8_t qs[128];
    for (int i = 0; i < 128; ++i)
    {
        qs[i] = (i % 2 == 0) ? 0x0F : 0xF0; // Alternate low/high nibbles
    }

    Q4_KBlock block = createBlock(0x4000, 0x3C00, scales, qs); // d=2.0, dmin=1.0

    float scalar_output[BLOCK_SIZE];
    Q4_KTensor::decodeBlockScalar(block, scalar_output);

#if defined(__AVX2__)
    float avx2_output[BLOCK_SIZE];
    Q4_KTensor::decodeBlockAVX2(block, avx2_output);
    compareOutputs(scalar_output, avx2_output, BLOCK_SIZE, 1e-4f, "AlternatingNibbles AVX2");
#endif

#if defined(__AVX512F__)
    float avx512_output[BLOCK_SIZE];
    Q4_KTensor::decodeBlockAVX512(block, avx512_output);
    compareOutputs(scalar_output, avx512_output, BLOCK_SIZE, 1e-4f, "AlternatingNibbles AVX512");
#endif
}

TEST_F(Test__Q4_KTensor, EdgeCase_GroupBoundaries)
{
    // Test with different patterns across 64-element group boundaries
    constexpr size_t BLOCK_SIZE = 256;

    uint8_t scales[12];
    for (int i = 0; i < 12; ++i)
    {
        scales[i] = static_cast<uint8_t>(i * 5); // Varying scales
    }

    uint8_t qs[128];
    for (int i = 0; i < 128; ++i)
    {
        // Different patterns for each 32-byte group (64 elements)
        int group = i / 32;
        qs[i] = static_cast<uint8_t>((group * 0x11) + (i % 16));
    }

    Q4_KBlock block = createBlock(0x3E00, 0x3A00, scales, qs); // d=1.5, dmin=0.75

    float scalar_output[BLOCK_SIZE];
    Q4_KTensor::decodeBlockScalar(block, scalar_output);

#if defined(__AVX2__)
    float avx2_output[BLOCK_SIZE];
    Q4_KTensor::decodeBlockAVX2(block, avx2_output);
    compareOutputs(scalar_output, avx2_output, BLOCK_SIZE, 1e-4f, "GroupBoundaries AVX2");
#endif

#if defined(__AVX512F__)
    float avx512_output[BLOCK_SIZE];
    Q4_KTensor::decodeBlockAVX512(block, avx512_output);
    compareOutputs(scalar_output, avx512_output, BLOCK_SIZE, 1e-4f, "GroupBoundaries AVX512");
#endif
}

TEST_F(Test__Q4_KTensor, EdgeCase_ScaleUnpacking)
{
    // Test scale unpacking edge cases (j < 4 vs j >= 4 in get_scale_min_k4)
    constexpr size_t BLOCK_SIZE = 256;

    // Craft scales to test both paths in get_scale_min_k4
    uint8_t scales[12] = {
        0x3F, 0x00, 0x15, 0x2A, // First 4 (j < 4 path)
        0xC0, 0xC0, 0xC0, 0xC0, // Next 4 (j >= 4 path, high bits set)
        0x0F, 0xF0, 0x55, 0xAA  // Last 4 (mixed patterns)
    };

    uint8_t qs[128];
    for (int i = 0; i < 128; ++i)
    {
        qs[i] = static_cast<uint8_t>(i ^ 0x5A); // XOR pattern
    }

    Q4_KBlock block = createBlock(0x3C00, 0x3800, scales, qs); // d=1.0, dmin=0.5

    float scalar_output[BLOCK_SIZE];
    Q4_KTensor::decodeBlockScalar(block, scalar_output);

#if defined(__AVX2__)
    float avx2_output[BLOCK_SIZE];
    Q4_KTensor::decodeBlockAVX2(block, avx2_output);
    compareOutputs(scalar_output, avx2_output, BLOCK_SIZE, 1e-4f, "ScaleUnpacking AVX2");
#endif

#if defined(__AVX512F__)
    float avx512_output[BLOCK_SIZE];
    Q4_KTensor::decodeBlockAVX512(block, avx512_output);
    compareOutputs(scalar_output, avx512_output, BLOCK_SIZE, 1e-4f, "ScaleUnpacking AVX512");
#endif
}

TEST_F(Test__Q4_KTensor, EdgeCase_RandomValues)
{
    // Comprehensive random test with many blocks
    constexpr size_t BLOCK_SIZE = 256;
    constexpr int NUM_BLOCKS = 100;

    for (int block_idx = 0; block_idx < NUM_BLOCKS; ++block_idx)
    {
        Q4_KBlock block = createRandomBlock();

        float scalar_output[BLOCK_SIZE];
        Q4_KTensor::decodeBlockScalar(block, scalar_output);

#if defined(__AVX2__)
        float avx2_output[BLOCK_SIZE];
        Q4_KTensor::decodeBlockAVX2(block, avx2_output);
        compareOutputs(scalar_output, avx2_output, BLOCK_SIZE, 1e-4f,
                       "RandomValues AVX2 block " + std::to_string(block_idx));
#endif

#if defined(__AVX512F__)
        float avx512_output[BLOCK_SIZE];
        Q4_KTensor::decodeBlockAVX512(block, avx512_output);
        compareOutputs(scalar_output, avx512_output, BLOCK_SIZE, 1e-4f,
                       "RandomValues AVX512 block " + std::to_string(block_idx));
#endif
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
