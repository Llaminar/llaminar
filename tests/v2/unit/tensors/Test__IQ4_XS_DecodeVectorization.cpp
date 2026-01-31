/**
 * @file Test__IQ4_XS_DecodeVectorization.cpp
 * @brief Unit tests for IQ4_XS decode_to_q8_0 SIMD implementations
 * @author David Sanftenberg
 */

#include "tensors/Tensors.h"
#include "tensors/SIMDHelpers.h"
#include "tensors/IQQuantTables.h"
#include <gtest/gtest.h>
#include <random>
#include <cmath>
#include <vector>
#include <iomanip>

using namespace llaminar2;
using namespace llaminar2::simd;

// Helper to create a deterministic IQ4_XS block for testing
IQ4_XSBlock make_test_iq4xs_block()
{
    IQ4_XSBlock block;
    block.d = fp32_to_fp16(0.5f); // Base scale

    // 8 sub-blocks with different 6-bit scales (0-63)
    block.scales_l[0] = (5 << 0) | (10 << 4);  // sub-blocks 0,1: ls=5,10
    block.scales_l[1] = (15 << 0) | (20 << 4); // sub-blocks 2,3: ls=15,20
    block.scales_l[2] = (25 << 0) | (30 << 4); // sub-blocks 4,5: ls=25,30
    block.scales_l[3] = (35 << 0) | (40 << 4); // sub-blocks 6,7: ls=35,40

    block.scales_h = 0; // High 2 bits all zero for simplicity

    // Fill qs with pattern: 16 bytes per sub-block (32 nibbles)
    for (size_t sb = 0; sb < 8; ++sb)
    {
        for (size_t j = 0; j < 16; ++j)
        {
            // Alternate between low values (0-7) and high values (8-15)
            uint8_t lo = (j % 2 == 0) ? (j % 8) : ((j % 8) + 8);
            uint8_t hi = (j % 2 == 0) ? ((j % 8) + 8) : (j % 8);
            block.qs[sb * 16 + j] = (hi << 4) | lo;
        }
    }

    return block;
}

// =====================================================================
// SCALAR IMPLEMENTATION TESTS
// =====================================================================

TEST(IQ4_XS_DecodeVectorization, ScalarCorrectness)
{
    IQ4_XSBlock test_block = make_test_iq4xs_block();

    // Decode sub-block 0
    int8_t qs[32];
    uint16_t scale_fp16;
    decode_iq4xs_to_q8_0_scalar(test_block, 0, qs, &scale_fp16);

    // Verify we got output (non-zero)
    bool has_nonzero = false;
    for (int i = 0; i < 32; ++i)
    {
        if (qs[i] != 0)
        {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero) << "Scalar decode produced all zeros";

    // Verify scale is reasonable (positive, finite)
    float scale = fp16_to_fp32(scale_fp16);
    EXPECT_GT(scale, 0.0f) << "Scale should be positive";
    EXPECT_TRUE(std::isfinite(scale)) << "Scale should be finite";
}

TEST(IQ4_XS_DecodeVectorization, ScalarMultipleSubBlocks)
{
    IQ4_XSBlock test_block = make_test_iq4xs_block();

    // Decode all 8 sub-blocks
    int8_t qs_all[8][32];
    uint16_t scales_all[8];

    for (size_t sb = 0; sb < 8; ++sb)
    {
        decode_iq4xs_to_q8_0_scalar(test_block, sb, qs_all[sb], &scales_all[sb]);

        // Verify non-zero output
        bool has_nonzero = false;
        for (int i = 0; i < 32; ++i)
        {
            if (qs_all[sb][i] != 0)
            {
                has_nonzero = true;
                break;
            }
        }
        EXPECT_TRUE(has_nonzero) << "Sub-block " << sb << " produced all zeros";

        // Verify scale
        float scale = fp16_to_fp32(scales_all[sb]);
        EXPECT_GT(scale, 0.0f) << "Sub-block " << sb << " scale should be positive";
    }
}

TEST(IQ4_XS_DecodeVectorization, LookupTableValues)
{
    // Verify kvalues_iq4nl[16] is accessible and has expected structure
    EXPECT_EQ(sizeof(kvalues_iq4nl) / sizeof(kvalues_iq4nl[0]), 16);

    // Check that all values are finite (not NaN or Inf)
    for (int i = 0; i < 16; ++i)
    {
        EXPECT_TRUE(std::isfinite(kvalues_iq4nl[i])) << "kvalues_iq4nl[" << i << "] is not finite";
    }

    // Verify at least some non-zero values exist
    bool has_positive = false, has_negative = false;
    for (int i = 0; i < 16; ++i)
    {
        if (kvalues_iq4nl[i] > 0.0f)
            has_positive = true;
        if (kvalues_iq4nl[i] < 0.0f)
            has_negative = true;
    }
    EXPECT_TRUE(has_positive) << "Lookup table should have positive values";
    EXPECT_TRUE(has_negative) << "Lookup table should have negative values";
}

// =====================================================================
// SIMD PARITY TESTS
// =====================================================================

#ifdef __AVX2__
TEST(IQ4_XS_DecodeVectorization, AVX2ParityWithScalar)
{
    IQ4_XSBlock test_block = make_test_iq4xs_block();

    for (size_t sb = 0; sb < 8; ++sb)
    {
        int8_t qs_scalar[32], qs_avx2[32];
        uint16_t scale_scalar, scale_avx2;

        decode_iq4xs_to_q8_0_scalar(test_block, sb, qs_scalar, &scale_scalar);
        decode_iq4xs_to_q8_0_avx2(test_block, sb, qs_avx2, &scale_avx2);

        // Compare scales
        float s_scalar = fp16_to_fp32(scale_scalar);
        float s_avx2 = fp16_to_fp32(scale_avx2);
        float scale_diff = std::abs(s_scalar - s_avx2) / std::max(std::abs(s_scalar), 1e-6f);
        EXPECT_LT(scale_diff, 0.01f) << "Sub-block " << sb << ": AVX2 scale mismatch";

        // Compare quantized values
        int mismatches = 0;
        for (int i = 0; i < 32; ++i)
        {
            if (std::abs(qs_scalar[i] - qs_avx2[i]) > 1)
            { // Allow ±1 rounding difference
                mismatches++;
            }
        }
        EXPECT_LE(mismatches, 2) << "Sub-block " << sb << ": AVX2 has " << mismatches << " mismatches";
    }
}
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
TEST(IQ4_XS_DecodeVectorization, AVX512ParityWithScalar)
{
    IQ4_XSBlock test_block = make_test_iq4xs_block();

    for (size_t sb = 0; sb < 8; ++sb)
    {
        int8_t qs_scalar[32], qs_avx512[32];
        uint16_t scale_scalar, scale_avx512;

        decode_iq4xs_to_q8_0_scalar(test_block, sb, qs_scalar, &scale_scalar);
        decode_iq4xs_to_q8_0_avx512(test_block, sb, qs_avx512, &scale_avx512);

        // Compare scales
        float s_scalar = fp16_to_fp32(scale_scalar);
        float s_avx512 = fp16_to_fp32(scale_avx512);
        float scale_diff = std::abs(s_scalar - s_avx512) / std::max(std::abs(s_scalar), 1e-6f);
        EXPECT_LT(scale_diff, 0.01f) << "Sub-block " << sb << ": AVX512 scale mismatch";

        // Compare quantized values
        int mismatches = 0;
        for (int i = 0; i < 32; ++i)
        {
            if (std::abs(qs_scalar[i] - qs_avx512[i]) > 1)
            {
                mismatches++;
            }
        }
        EXPECT_LE(mismatches, 2) << "Sub-block " << sb << ": AVX512 has " << mismatches << " mismatches";
    }
}
#endif

// =====================================================================
// INTEGRATION TEST (Full Tensor)
// =====================================================================

TEST(IQ4_XS_DecodeVectorization, TensorDecodeToQ8)
{
    // Create small 2D IQ4_XS tensor: 2 rows × 512 cols = 1024 elements
    // 512 cols / 256 elements per block = 2 blocks per row
    std::vector<size_t> shape = {2, 512};
    size_t blocks_per_row = 2;
    size_t total_blocks = 2 * blocks_per_row; // 4 blocks

    std::vector<uint8_t> raw_data(total_blocks * sizeof(IQ4_XSBlock));

    // Initialize blocks with pattern
    IQ4_XSBlock *blocks = reinterpret_cast<IQ4_XSBlock *>(raw_data.data());
    for (size_t i = 0; i < total_blocks; ++i)
    {
        blocks[i] = make_test_iq4xs_block();
        blocks[i].d = fp32_to_fp16(0.1f * (i + 1)); // Varying scales
    }

    IQ4_XSTensor tensor(shape, raw_data);

    // Decode row 0, block 0 → 8 Q8_0 blocks (256 elements total)
    Q8_0Block output[8];
    tensor.decode_to_q8_0(0, 0, output);

    // Verify all 8 sub-blocks decoded
    for (size_t sb = 0; sb < 8; ++sb)
    {
        float scale = fp16_to_fp32(output[sb].d);
        EXPECT_GT(scale, 0.0f) << "Sub-block " << sb << " scale should be positive";

        bool has_nonzero = false;
        for (int i = 0; i < 32; ++i)
        {
            if (output[sb].qs[i] != 0)
            {
                has_nonzero = true;
                break;
            }
        }
        EXPECT_TRUE(has_nonzero) << "Sub-block " << sb << " should have non-zero values";
    }
}

// =====================================================================
// FUZZ TEST (Random Input Stability)
// =====================================================================

TEST(IQ4_XS_DecodeVectorization, FuzzTestRandomInput)
{
    std::mt19937 rng(12345);
    std::uniform_int_distribution<uint8_t> byte_dist(0, 255);

    for (int trial = 0; trial < 100; ++trial)
    {
        IQ4_XSBlock block;
        block.d = fp32_to_fp16(0.01f + 0.01f * (trial % 50)); // Varying scales

        // Random scales
        for (int i = 0; i < 4; ++i)
        {
            block.scales_l[i] = byte_dist(rng);
        }
        block.scales_h = byte_dist(rng);

        // Random qs
        for (int i = 0; i < 128; ++i)
        {
            block.qs[i] = byte_dist(rng);
        }

        // Decode all sub-blocks with scalar (should never crash)
        for (size_t sb = 0; sb < 8; ++sb)
        {
            int8_t qs[32];
            uint16_t scale_fp16;
            EXPECT_NO_THROW(decode_iq4xs_to_q8_0_scalar(block, sb, qs, &scale_fp16))
                << "Trial " << trial << ", sub-block " << sb << " crashed";

            float scale = fp16_to_fp32(scale_fp16);
            EXPECT_TRUE(std::isfinite(scale)) << "Trial " << trial << ", sub-block " << sb << " produced non-finite scale";
        }

#ifdef __AVX2__
        // AVX2 should also not crash
        for (size_t sb = 0; sb < 8; ++sb)
        {
            int8_t qs[32];
            uint16_t scale_fp16;
            EXPECT_NO_THROW(decode_iq4xs_to_q8_0_avx2(block, sb, qs, &scale_fp16))
                << "Trial " << trial << ", sub-block " << sb << " AVX2 crashed";
        }
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
        // AVX512 should also not crash
        for (size_t sb = 0; sb < 8; ++sb)
        {
            int8_t qs[32];
            uint16_t scale_fp16;
            EXPECT_NO_THROW(decode_iq4xs_to_q8_0_avx512(block, sb, qs, &scale_fp16))
                << "Trial " << trial << ", sub-block " << sb << " AVX512 crashed";
        }
#endif
    }
}

// =====================================================================
// PERFORMANCE BENCHMARK (Optional, disabled by default)
// =====================================================================

TEST(IQ4_XS_DecodeVectorization, DISABLED_PerformanceBenchmark)
{
    constexpr size_t NUM_BLOCKS = 10000;
    std::vector<IQ4_XSBlock> blocks(NUM_BLOCKS);

    // Initialize with test pattern
    for (size_t i = 0; i < NUM_BLOCKS; ++i)
    {
        blocks[i] = make_test_iq4xs_block();
    }

    // Scalar baseline
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < NUM_BLOCKS; ++i)
        {
            for (size_t sb = 0; sb < 8; ++sb)
            {
                int8_t qs[32];
                uint16_t scale;
                decode_iq4xs_to_q8_0_scalar(blocks[i], sb, qs, &scale);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "Scalar: " << ms << " ms\n";
    }

#ifdef __AVX2__
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < NUM_BLOCKS; ++i)
        {
            for (size_t sb = 0; sb < 8; ++sb)
            {
                int8_t qs[32];
                uint16_t scale;
                decode_iq4xs_to_q8_0_avx2(blocks[i], sb, qs, &scale);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "AVX2: " << ms << " ms\n";
    }
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < NUM_BLOCKS; ++i)
        {
            for (size_t sb = 0; sb < 8; ++sb)
            {
                int8_t qs[32];
                uint16_t scale;
                decode_iq4xs_to_q8_0_avx512(blocks[i], sb, qs, &scale);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "AVX512: " << ms << " ms\n";
    }
#endif
}

// =====================================================================
// ERROR HANDLING TESTS
// =====================================================================

TEST(IQ4_XS_DecodeVectorization, OutOfRangeSubBlock)
{
    IQ4_XSBlock test_block = make_test_iq4xs_block();
    int8_t qs[32];
    uint16_t scale;

    // Sub-block 8 is out of range (only 0-7 valid)
    EXPECT_THROW(decode_iq4xs_to_q8_0_scalar(test_block, 8, qs, &scale), std::out_of_range);

#ifdef __AVX2__
    EXPECT_THROW(decode_iq4xs_to_q8_0_avx2(test_block, 8, qs, &scale), std::out_of_range);
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
    EXPECT_THROW(decode_iq4xs_to_q8_0_avx512(test_block, 8, qs, &scale), std::out_of_range);
#endif
}

TEST(IQ4_XS_DecodeVectorization, AutoDispatchWorks)
{
    IQ4_XSBlock test_block = make_test_iq4xs_block();

    // Auto-dispatch should select best available implementation
    int8_t qs_auto[32], qs_scalar[32];
    uint16_t scale_auto, scale_scalar;

    decode_iq4xs_to_q8_0(test_block, 0, qs_auto, &scale_auto);
    decode_iq4xs_to_q8_0_scalar(test_block, 0, qs_scalar, &scale_scalar);

    // Auto-dispatch should match scalar (or be very close)
    float s_auto = fp16_to_fp32(scale_auto);
    float s_scalar = fp16_to_fp32(scale_scalar);
    float scale_diff = std::abs(s_auto - s_scalar) / std::max(std::abs(s_scalar), 1e-6f);
    EXPECT_LT(scale_diff, 0.01f) << "Auto-dispatch scale mismatch";

    int mismatches = 0;
    for (int i = 0; i < 32; ++i)
    {
        if (std::abs(qs_auto[i] - qs_scalar[i]) > 1)
        {
            mismatches++;
        }
    }
    EXPECT_LE(mismatches, 2) << "Auto-dispatch has " << mismatches << " mismatches";
}
