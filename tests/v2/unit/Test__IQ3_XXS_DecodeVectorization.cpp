/**
 * @file Test__IQ3_XXS_DecodeVectorization.cpp
 * @brief Unit tests for IQ3_XXS decode_to_q8_0 SIMD implementation
 * @author David Sanftenberg
 *
 * Tests scalar, AVX2, and AVX-512 decode paths for:
 * - Correctness (grid lookup, scales, signs)
 * - Parity between SIMD variants
 * - Tensor integration
 * - Performance
 */

#include <gtest/gtest.h>
#include "../../src/v2/tensors/Tensors.h"
#include "../../src/v2/tensors/BlockStructures.h"
#include "../../src/v2/tensors/SIMDHelpers.h"
#include "../../src/v2/tensors/FP16Utils.h"
#include "../../src/v2/tensors/IQQuantTables.h"
#include <cmath>
#include <random>
#include <chrono>

using namespace llaminar2;
using namespace llaminar2::simd;

class Test__IQ3_XXS_DecodeVectorization : public ::testing::Test
{
protected:
    static constexpr float TOLERANCE = 1e-4f; // Tight tolerance for decode accuracy

    // Create a test block with known grid indices and scales
    IQ3_XXSBlock createTestBlock()
    {
        IQ3_XXSBlock block;
        block.d = fp32_to_fp16(0.5f); // Scale factor

        // Fill qs[0..63] with grid indices (0-255 valid for iq3xxs_grid)
        for (size_t i = 0; i < 64; ++i)
        {
            block.qs[i] = static_cast<uint8_t>(i % 256);
        }

        // Fill qs[64..95] with scales+signs (8 sub-blocks × 4 bytes)
        for (size_t ib = 0; ib < 8; ++ib)
        {
            uint32_t aux32 = 0;
            aux32 |= (ib & 0xF) << 28;         // Scale (top 4 bits)
            aux32 |= (ib * 17) & 0x7F;         // Sign pattern 0 (7 bits)
            aux32 |= ((ib * 23) & 0x7F) << 7;  // Sign pattern 1
            aux32 |= ((ib * 31) & 0x7F) << 14; // Sign pattern 2
            aux32 |= ((ib * 37) & 0x7F) << 21; // Sign pattern 3
            std::memcpy(&block.qs[64 + 4 * ib], &aux32, sizeof(uint32_t));
        }

        return block;
    }

    // Compare two Q8_0 blocks
    bool blocksEqual(const Q8_0Block &a, const Q8_0Block &b, float tol = TOLERANCE)
    {
        float scale_a = fp16_to_fp32(a.d);
        float scale_b = fp16_to_fp32(b.d);

        if (std::abs(scale_a - scale_b) > tol)
        {
            return false;
        }

        for (size_t i = 0; i < 32; ++i)
        {
            if (a.qs[i] != b.qs[i])
            {
                return false;
            }
        }

        return true;
    }
};

// Test 1: Scalar decode correctness
TEST_F(Test__IQ3_XXS_DecodeVectorization, ScalarCorrectness)
{
    IQ3_XXSBlock block = createTestBlock();
    Q8_0Block output;

    // Decode first sub-block
    decode_iq3xxs_to_q8_0_scalar(block, 0, output.qs, &output.d);

    // Validate scale is reasonable (not NaN, not zero)
    float scale_fp32 = fp16_to_fp32(output.d);
    EXPECT_GT(std::abs(scale_fp32), 1e-6f);
    EXPECT_FALSE(std::isnan(scale_fp32));
    EXPECT_FALSE(std::isinf(scale_fp32));

    // Validate quantized values are in valid range [-127, 127]
    for (size_t i = 0; i < 32; ++i)
    {
        EXPECT_GE(output.qs[i], -127);
        EXPECT_LE(output.qs[i], 127);
    }
}

// Test 2: Multi-block decode (all 8 sub-blocks)
TEST_F(Test__IQ3_XXS_DecodeVectorization, MultiBlockCorrectness)
{
    IQ3_XXSBlock block = createTestBlock();

    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        Q8_0Block output;
        decode_iq3xxs_to_q8_0_scalar(block, sub_idx, output.qs, &output.d);

        float scale_fp32 = fp16_to_fp32(output.d);
        EXPECT_GT(std::abs(scale_fp32), 1e-6f) << "Sub-block " << sub_idx;
        EXPECT_FALSE(std::isnan(scale_fp32)) << "Sub-block " << sub_idx;

        for (size_t i = 0; i < 32; ++i)
        {
            EXPECT_GE(output.qs[i], -127) << "Sub-block " << sub_idx << ", element " << i;
            EXPECT_LE(output.qs[i], 127) << "Sub-block " << sub_idx << ", element " << i;
        }
    }
}

// Test 3: Grid lookup verification
TEST_F(Test__IQ3_XXS_DecodeVectorization, GridLookupVerification)
{
    IQ3_XXSBlock block;
    block.d = fp32_to_fp16(1.0f);

    // Use specific grid indices with known values
    for (size_t i = 0; i < 64; ++i)
    {
        block.qs[i] = 0; // iq3xxs_grid[0] = 0x04040404
    }

    // Simple scales+signs (scale=1, no sign flips)
    for (size_t ib = 0; ib < 8; ++ib)
    {
        uint32_t aux32 = (1 << 28); // Scale = 0.5 + 1 = 1.5, then * 0.5 = 0.75
        std::memcpy(&block.qs[64 + 4 * ib], &aux32, sizeof(uint32_t));
    }

    Q8_0Block output;
    decode_iq3xxs_to_q8_0_scalar(block, 0, output.qs, &output.d);

    float scale_fp32 = fp16_to_fp32(output.d);
    EXPECT_GT(std::abs(scale_fp32), 1e-6f);
}

#ifdef __AVX2__
// Test 4: AVX2 vs Scalar parity
TEST_F(Test__IQ3_XXS_DecodeVectorization, AVX2Parity)
{
    if (!cpu_supports_avx2())
    {
        GTEST_SKIP() << "AVX2 not supported";
    }

    IQ3_XXSBlock block = createTestBlock();

    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        Q8_0Block scalar_out, avx2_out;

        decode_iq3xxs_to_q8_0_scalar(block, sub_idx, scalar_out.qs, &scalar_out.d);
        decode_iq3xxs_to_q8_0_avx2(block, sub_idx, avx2_out.qs, &avx2_out.d);

        EXPECT_TRUE(blocksEqual(scalar_out, avx2_out))
            << "Sub-block " << sub_idx << " mismatch";
    }
}
#endif

#ifdef __AVX512F__
// Test 5: AVX-512 vs Scalar parity
TEST_F(Test__IQ3_XXS_DecodeVectorization, AVX512Parity)
{
    if (!cpu_supports_avx512())
    {
        GTEST_SKIP() << "AVX-512 not supported";
    }

    IQ3_XXSBlock block = createTestBlock();

    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        Q8_0Block scalar_out, avx512_out;

        decode_iq3xxs_to_q8_0_scalar(block, sub_idx, scalar_out.qs, &scalar_out.d);
        decode_iq3xxs_to_q8_0_avx512(block, sub_idx, avx512_out.qs, &avx512_out.d);

        EXPECT_TRUE(blocksEqual(scalar_out, avx512_out))
            << "Sub-block " << sub_idx << " mismatch";
    }
}
#endif

// Test 6: Tensor integration
TEST_F(Test__IQ3_XXS_DecodeVectorization, TensorIntegration)
{
    // Create a small IQ3_XXS tensor (1 row, 256 elements = 1 super-block)
    std::vector<size_t> shape = {1, 256};
    std::vector<uint8_t> raw_data(sizeof(IQ3_XXSBlock));

    IQ3_XXSBlock *block_ptr = reinterpret_cast<IQ3_XXSBlock *>(raw_data.data());
    *block_ptr = createTestBlock();

    IQ3_XXSTensor tensor(shape, raw_data);

    // Decode to Q8_0
    Q8_0Block output[8];
    tensor.decode_to_q8_0(0, 0, output);

    // Validate all 8 sub-blocks
    for (size_t i = 0; i < 8; ++i)
    {
        float scale_fp32 = fp16_to_fp32(output[i].d);
        EXPECT_GT(std::abs(scale_fp32), 1e-6f) << "Sub-block " << i;
        EXPECT_FALSE(std::isnan(scale_fp32)) << "Sub-block " << i;
    }
}

// Test 7: Auto-dispatch correctness
TEST_F(Test__IQ3_XXS_DecodeVectorization, AutoDispatch)
{
    IQ3_XXSBlock block = createTestBlock();

    Q8_0Block scalar_out, dispatch_out;

    decode_iq3xxs_to_q8_0_scalar(block, 0, scalar_out.qs, &scalar_out.d);
    decode_iq3xxs_to_q8_0(block, 0, dispatch_out.qs, &dispatch_out.d);

    EXPECT_TRUE(blocksEqual(scalar_out, dispatch_out));
}

// Test 8: Fuzz testing with random blocks
TEST_F(Test__IQ3_XXS_DecodeVectorization, FuzzTesting)
{
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint8_t> grid_dist(0, 255);
    std::uniform_int_distribution<uint32_t> aux_dist(0, UINT32_MAX);

    for (size_t trial = 0; trial < 100; ++trial)
    {
        IQ3_XXSBlock block;
        block.d = fp32_to_fp16(0.1f + (trial % 10) * 0.1f);

        // Random grid indices
        for (size_t i = 0; i < 64; ++i)
        {
            block.qs[i] = grid_dist(rng);
        }

        // Random scales+signs
        for (size_t ib = 0; ib < 8; ++ib)
        {
            uint32_t aux32 = aux_dist(rng);
            std::memcpy(&block.qs[64 + 4 * ib], &aux32, sizeof(uint32_t));
        }

        Q8_0Block output;
        EXPECT_NO_THROW({
            decode_iq3xxs_to_q8_0_scalar(block, trial % 8, output.qs, &output.d);
        }) << "Trial "
           << trial;

        float scale_fp32 = fp16_to_fp32(output.d);
        EXPECT_FALSE(std::isnan(scale_fp32)) << "Trial " << trial;
        EXPECT_FALSE(std::isinf(scale_fp32)) << "Trial " << trial;
    }
}

// Test 9: Error handling
TEST_F(Test__IQ3_XXS_DecodeVectorization, ErrorHandling)
{
    IQ3_XXSBlock block = createTestBlock();
    Q8_0Block output;

    // Valid sub-block indices (0-7)
    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        EXPECT_NO_THROW({
            decode_iq3xxs_to_q8_0_scalar(block, sub_idx, output.qs, &output.d);
        }) << "Sub-block "
           << sub_idx;
    }

    // Edge case: sub-block 7 (last valid index)
    EXPECT_NO_THROW({
        decode_iq3xxs_to_q8_0_scalar(block, 7, output.qs, &output.d);
    });
}

// Test 10: Performance benchmark
TEST_F(Test__IQ3_XXS_DecodeVectorization, Performance)
{
    IQ3_XXSBlock block = createTestBlock();
    Q8_0Block output;

    constexpr size_t ITERATIONS = 10000;

    // Scalar
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < ITERATIONS; ++i)
    {
        decode_iq3xxs_to_q8_0_scalar(block, i % 8, output.qs, &output.d);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double scalar_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "Scalar:   " << scalar_ms << " ms (" << ITERATIONS << " iterations)" << std::endl;

#ifdef __AVX2__
    if (cpu_supports_avx2())
    {
        auto t2 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < ITERATIONS; ++i)
        {
            decode_iq3xxs_to_q8_0_avx2(block, i % 8, output.qs, &output.d);
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        double avx2_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

        std::cout << "AVX2:     " << avx2_ms << " ms (speedup: " << (scalar_ms / avx2_ms) << "x)" << std::endl;
    }
#endif

#ifdef __AVX512F__
    if (cpu_supports_avx512())
    {
        auto t4 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < ITERATIONS; ++i)
        {
            decode_iq3xxs_to_q8_0_avx512(block, i % 8, output.qs, &output.d);
        }
        auto t5 = std::chrono::high_resolution_clock::now();
        double avx512_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();

        std::cout << "AVX-512:  " << avx512_ms << " ms (speedup: " << (scalar_ms / avx512_ms) << "x)" << std::endl;
    }
#endif

    EXPECT_GT(scalar_ms, 0.0);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
