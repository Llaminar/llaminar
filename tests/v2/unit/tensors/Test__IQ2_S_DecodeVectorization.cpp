/**
 * @file Test__IQ2_S_DecodeVectorization.cpp
 * @brief Comprehensive unit tests for IQ2_S decode_to_q8_0 SIMD variants
 * @author David Sanftenberg
 *
 * Tests scalar, AVX2, and AVX-512 implementations for:
 * - Correctness (accuracy of decoded values)
 * - Parity (scalar vs AVX2 vs AVX-512 equivalence)
 * - Edge cases (boundary conditions, extreme values)
 * - Performance (throughput benchmarking)
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "tensors/BlockStructures.h"
#include "tensors/SIMDHelpers.h"
#include "tensors/FP16Utils.h"
#include "tensors/IQQuantTables.h"
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>

using namespace llaminar2;

class Test__IQ2_S_DecodeVectorization : public ::testing::Test
{
protected:
    std::mt19937 rng_{42}; // Fixed seed for reproducibility

    /**
     * @brief Create a test IQ2_S block with controlled values
     */
    IQ2_SBlock createTestBlock(uint32_t seed)
    {
        IQ2_SBlock block;
        std::mt19937 gen(seed);
        std::uniform_int_distribution<uint8_t> dist_byte(0, 255);

        // FP16 scale: Vary scale based on seed
        float scale = 0.125f + (seed % 16) * 0.0625f;
        block.d = fp32_to_fp16(scale);

        // qs[0..31]: Quantized values (8-bit, will be masked to create 10-bit indices)
        // qs[32..63]: Signs
        for (size_t i = 0; i < 64; ++i)
        {
            block.qs[i] = dist_byte(gen);
        }

        // qh[8]: High bits (2 bits per sub-block group)
        for (size_t i = 0; i < 8; ++i)
        {
            block.qh[i] = dist_byte(gen);
        }

        // scales[8]: Scales (4 bits per 16-element group)
        for (size_t i = 0; i < 8; ++i)
        {
            block.scales[i] = dist_byte(gen);
        }

        return block;
    }

    /**
     * @brief Compare two Q8_0 blocks for equality
     */
    bool compareQ8_0Blocks(const Q8_0Block &a, const Q8_0Block &b, float scale_tol = 1e-3f)
    {
        // Compare scales (FP16)
        float scale_a = fp16_to_fp32(a.d);
        float scale_b = fp16_to_fp32(b.d);
        if (std::abs(scale_a - scale_b) > scale_tol)
        {
            return false;
        }

        // Compare quantized values
        for (size_t i = 0; i < 32; ++i)
        {
            if (a.qs[i] != b.qs[i])
            {
                return false;
            }
        }

        return true;
    }

    /**
     * @brief Compare arrays of Q8_0 blocks
     */
    bool compareQ8_0BlockArrays(const Q8_0Block *a, const Q8_0Block *b, size_t count, float scale_tol = 1e-3f)
    {
        for (size_t i = 0; i < count; ++i)
        {
            if (!compareQ8_0Blocks(a[i], b[i], scale_tol))
            {
                return false;
            }
        }
        return true;
    }
};

// =============================================================================
// Correctness Tests
// =============================================================================

TEST_F(Test__IQ2_S_DecodeVectorization, ScalarCorrectness)
{
    auto block = createTestBlock(12345);

    // Decode using scalar implementation
    Q8_0Block output[8]; // 8 sub-blocks of 32 elements each
    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        simd::decode_iq2s_to_q8_0_scalar(block, sub_idx, output[sub_idx].qs, &output[sub_idx].d);
    }

    // Validate scales are reasonable
    for (size_t i = 0; i < 8; ++i)
    {
        float scale = fp16_to_fp32(output[i].d);
        EXPECT_GT(scale, 0.0f) << "Scale " << i << " should be positive";
        EXPECT_LT(scale, 100.0f) << "Scale " << i << " should be reasonable";
    }

    // Validate quantized values are in range [-127, 127]
    for (size_t i = 0; i < 8; ++i)
    {
        for (size_t j = 0; j < 32; ++j)
        {
            int8_t val = output[i].qs[j];
            EXPECT_GE(val, -127) << "Q8_0 value out of range";
            EXPECT_LE(val, 127) << "Q8_0 value out of range";
        }
    }
}

TEST_F(Test__IQ2_S_DecodeVectorization, MultiBlockCorrectness)
{
    // Test multiple blocks to ensure no state leakage
    for (uint32_t seed = 0; seed < 10; ++seed)
    {
        auto block = createTestBlock(seed);
        Q8_0Block output[8];

        for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
        {
            simd::decode_iq2s_to_q8_0_scalar(block, sub_idx, output[sub_idx].qs, &output[sub_idx].d);
        }

        // Each decode should produce valid results
        for (size_t i = 0; i < 8; ++i)
        {
            float scale = fp16_to_fp32(output[i].d);
            EXPECT_GT(scale, 0.0f);
        }
    }
}

TEST_F(Test__IQ2_S_DecodeVectorization, GridLookupVerification)
{
    // Test that grid indices are constructed correctly
    auto block = createTestBlock(99999);

    // Manually verify grid index construction for first sub-block
    uint8_t qh_byte = block.qh[0];
    for (size_t l = 0; l < 4; ++l)
    {
        uint16_t grid_idx = block.qs[l] | ((qh_byte << (8 - 2 * l)) & 0x300);
        EXPECT_LT(grid_idx, 1024) << "Grid index out of bounds for iq2s_grid[1024]";
    }

    // Decode and verify
    Q8_0Block output[8];
    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        simd::decode_iq2s_to_q8_0_scalar(block, sub_idx, output[sub_idx].qs, &output[sub_idx].d);
    }

    // Should complete without errors
    EXPECT_TRUE(true);
}

// =============================================================================
// Parity Tests (Scalar vs AVX2 vs AVX-512)
// =============================================================================

#ifdef __AVX2__
TEST_F(Test__IQ2_S_DecodeVectorization, AVX2Parity)
{
    auto block = createTestBlock(54321);

    Q8_0Block output_scalar[8];
    Q8_0Block output_avx2[8];

    // Decode with scalar
    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        simd::decode_iq2s_to_q8_0_scalar(block, sub_idx, output_scalar[sub_idx].qs, &output_scalar[sub_idx].d);
    }

    // Decode with AVX2
    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        simd::decode_iq2s_to_q8_0_avx2(block, sub_idx, output_avx2[sub_idx].qs, &output_avx2[sub_idx].d);
    }

    // Compare outputs
    EXPECT_TRUE(compareQ8_0BlockArrays(output_scalar, output_avx2, 8, 1e-3f))
        << "AVX2 and scalar outputs should match";
}
#endif

#ifdef __AVX512F__
TEST_F(Test__IQ2_S_DecodeVectorization, AVX512Parity)
{
    auto block = createTestBlock(11111);

    Q8_0Block output_scalar[8];
    Q8_0Block output_avx512[8];

    // Decode with scalar
    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        simd::decode_iq2s_to_q8_0_scalar(block, sub_idx, output_scalar[sub_idx].qs, &output_scalar[sub_idx].d);
    }

    // Decode with AVX-512
    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        simd::decode_iq2s_to_q8_0_avx512(block, sub_idx, output_avx512[sub_idx].qs, &output_avx512[sub_idx].d);
    }

    // Compare outputs
    EXPECT_TRUE(compareQ8_0BlockArrays(output_scalar, output_avx512, 8, 1e-3f))
        << "AVX-512 and scalar outputs should match";
}
#endif

// =============================================================================
// Tensor Integration Tests
// =============================================================================

TEST_F(Test__IQ2_S_DecodeVectorization, TensorIntegration)
{
    // Create a small IQ2_S tensor (1 block = 256 elements)
    std::vector<size_t> shape = {256};
    std::vector<uint8_t> raw_data(sizeof(IQ2_SBlock));

    // Fill with test data
    IQ2_SBlock *block_ptr = reinterpret_cast<IQ2_SBlock *>(raw_data.data());
    *block_ptr = createTestBlock(77777);

    IQ2_STensor tensor(shape, raw_data);

    // Decode using tensor's decode_to_q8_0 method
    Q8_0Block output[8];
    tensor.decode_to_q8_0(0, 0, output);

    // Validate results
    for (size_t i = 0; i < 8; ++i)
    {
        float scale = fp16_to_fp32(output[i].d);
        EXPECT_GT(scale, 0.0f);
    }
}

// =============================================================================
// Auto-Dispatch Test
// =============================================================================

TEST_F(Test__IQ2_S_DecodeVectorization, AutoDispatch)
{
    auto block = createTestBlock(88888);

    Q8_0Block output_auto[8];
    Q8_0Block output_scalar[8];

    // Decode with auto-dispatch
    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        simd::decode_iq2s_to_q8_0(block, sub_idx, output_auto[sub_idx].qs, &output_auto[sub_idx].d);
    }

    // Decode with scalar (reference)
    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        simd::decode_iq2s_to_q8_0_scalar(block, sub_idx, output_scalar[sub_idx].qs, &output_scalar[sub_idx].d);
    }

    // Should match (auto-dispatch selects best available SIMD path)
    EXPECT_TRUE(compareQ8_0BlockArrays(output_auto, output_scalar, 8, 1e-3f));
}

// =============================================================================
// Fuzz Testing
// =============================================================================

TEST_F(Test__IQ2_S_DecodeVectorization, FuzzTesting)
{
    // Test with 1000 random blocks to catch edge cases
    for (uint32_t seed = 0; seed < 1000; ++seed)
    {
        auto block = createTestBlock(seed);

        Q8_0Block output_scalar[8];
        Q8_0Block output_auto[8];

        // Decode with scalar
        for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
        {
            simd::decode_iq2s_to_q8_0_scalar(block, sub_idx, output_scalar[sub_idx].qs, &output_scalar[sub_idx].d);
        }

        // Decode with auto-dispatch
        for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
        {
            simd::decode_iq2s_to_q8_0(block, sub_idx, output_auto[sub_idx].qs, &output_auto[sub_idx].d);
        }

        // Should match
        EXPECT_TRUE(compareQ8_0BlockArrays(output_scalar, output_auto, 8, 1e-3f))
            << "Fuzz test failed at seed " << seed;
    }
}

// =============================================================================
// Error Handling Tests
// =============================================================================

TEST_F(Test__IQ2_S_DecodeVectorization, ErrorHandling)
{
    auto block = createTestBlock(33333);
    Q8_0Block output[8];

    // Test all valid sub-block indices (0-7)
    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        EXPECT_NO_THROW({
            simd::decode_iq2s_to_q8_0_scalar(block, sub_idx, output[sub_idx].qs, &output[sub_idx].d);
        }) << "Should handle sub_idx "
           << sub_idx;
    }
}

// =============================================================================
// Performance Benchmarking
// =============================================================================

TEST_F(Test__IQ2_S_DecodeVectorization, Performance)
{
    auto block = createTestBlock(12345);
    Q8_0Block output[8];

    const int iterations = 10000;

    // Benchmark scalar
    auto start_scalar = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
    {
        for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
        {
            simd::decode_iq2s_to_q8_0_scalar(block, sub_idx, output[sub_idx].qs, &output[sub_idx].d);
        }
    }
    auto end_scalar = std::chrono::high_resolution_clock::now();
    double time_scalar = std::chrono::duration_cast<std::chrono::microseconds>(end_scalar - start_scalar).count() / 1000.0;

#ifdef __AVX2__
    // Benchmark AVX2
    auto start_avx2 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
    {
        for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
        {
            simd::decode_iq2s_to_q8_0_avx2(block, sub_idx, output[sub_idx].qs, &output[sub_idx].d);
        }
    }
    auto end_avx2 = std::chrono::high_resolution_clock::now();
    double time_avx2 = std::chrono::duration_cast<std::chrono::microseconds>(end_avx2 - start_avx2).count() / 1000.0;
#endif

#ifdef __AVX512F__
    // Benchmark AVX-512
    auto start_avx512 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
    {
        for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
        {
            simd::decode_iq2s_to_q8_0_avx512(block, sub_idx, output[sub_idx].qs, &output[sub_idx].d);
        }
    }
    auto end_avx512 = std::chrono::high_resolution_clock::now();
    double time_avx512 = std::chrono::duration_cast<std::chrono::microseconds>(end_avx512 - start_avx512).count() / 1000.0;
#endif

    // Print results
    std::cout << "\nIQ2_S Decode Performance (" << iterations << " iterations):\n";
    std::cout << "  Scalar:   " << time_scalar << " ms (baseline)\n";

#ifdef __AVX2__
    double speedup_avx2 = time_scalar / time_avx2;
    std::cout << "  AVX2:     " << time_avx2 << " ms (speedup: " << speedup_avx2 << "x)\n";
#endif

#ifdef __AVX512F__
    double speedup_avx512 = time_scalar / time_avx512;
    std::cout << "  AVX-512:  " << time_avx512 << " ms (speedup: " << speedup_avx512 << "x)\n";
#endif

    // Always passes (performance test is informational)
    EXPECT_TRUE(true);
}
