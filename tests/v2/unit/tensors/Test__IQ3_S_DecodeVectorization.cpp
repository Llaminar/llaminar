/**
 * @file Test__IQ3_S_DecodeVectorization.cpp
 * @brief Comprehensive tests for IQ3_S decode_to_q8_0 SIMD implementation
 *
 * Tests scalar, AVX2, and AVX-512 decode paths for:
 * - Correctness (decode produces valid Q8_0 blocks)
 * - Parity (all SIMD variants match scalar exactly)
 * - Integration (IQ3_STensor::decode_to_q8_0 works end-to-end)
 * - Performance (measure SIMD speedups)
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

class Test__IQ3_S_DecodeVectorization : public ::testing::Test
{
protected:
    std::mt19937 rng_{42}; // Fixed seed for reproducibility

    /**
     * @brief Create a test IQ3_SBlock with random but valid data
     */
    IQ3_SBlock createTestBlock(uint32_t seed = 0)
    {
        std::mt19937 rng(seed == 0 ? rng_() : seed);
        std::uniform_real_distribution<float> scale_dist(0.01f, 10.0f);
        std::uniform_int_distribution<uint8_t> byte_dist(0, 255);

        IQ3_SBlock block;

        // Random FP16 scale
        block.d = fp32_to_fp16(scale_dist(rng));

        // Random qs[] (64 bytes) - grid indices (0-255, will add high bit later)
        for (size_t i = 0; i < 64; ++i)
        {
            block.qs[i] = byte_dist(rng);
        }

        // Random qh[] (8 bytes) - high bits for grid indices
        for (size_t i = 0; i < 8; ++i)
        {
            block.qh[i] = byte_dist(rng);
        }

        // Random signs[] (32 bytes)
        for (size_t i = 0; i < 32; ++i)
        {
            block.signs[i] = byte_dist(rng);
        }

        // Random scales[] (4 bytes) - 4-bit nibbles
        for (size_t i = 0; i < 4; ++i)
        {
            block.scales[i] = byte_dist(rng);
        }

        return block;
    }

    /**
     * @brief Compare two Q8_0 block arrays for equality (with tolerance)
     */
    bool compareQ8_0BlockArrays(const Q8_0Block *a, const Q8_0Block *b, size_t count, float tolerance = 1e-5f)
    {
        for (size_t i = 0; i < count; ++i)
        {
            // Compare scales (FP16 values)
            float scale_a = fp16_to_fp32(a[i].d);
            float scale_b = fp16_to_fp32(b[i].d);
            if (std::abs(scale_a - scale_b) > tolerance)
            {
                return false;
            }

            // Compare quantized values (int8_t)
            for (size_t j = 0; j < 32; ++j)
            {
                if (a[i].qs[j] != b[i].qs[j])
                {
                    return false;
                }
            }
        }
        return true;
    }
};

// ============================================================================
// Test 1: Scalar Correctness
// ============================================================================

TEST_F(Test__IQ3_S_DecodeVectorization, ScalarCorrectness)
{
    auto block = createTestBlock(100);

    Q8_0Block output[8]; // IQ3_S → 8 Q8_0 blocks

    // Decode all 8 sub-blocks
    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        simd::decode_iq3s_to_q8_0_scalar(block, sub_idx, output[sub_idx].qs, &output[sub_idx].d);
    }

    // Basic validation: scales should be positive, quantized values in range [-127, 127]
    for (size_t i = 0; i < 8; ++i)
    {
        float scale = fp16_to_fp32(output[i].d);
        EXPECT_GT(scale, 0.0f) << "Block " << i << " has non-positive scale";

        for (size_t j = 0; j < 32; ++j)
        {
            EXPECT_GE(output[i].qs[j], -127);
            EXPECT_LE(output[i].qs[j], 127);
        }
    }
}

// ============================================================================
// Test 2: Multi-Block Correctness
// ============================================================================

TEST_F(Test__IQ3_S_DecodeVectorization, MultiBlockCorrectness)
{
    for (uint32_t seed = 0; seed < 10; ++seed)
    {
        auto block = createTestBlock(seed * 42);
        Q8_0Block output[8];

        for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
        {
            simd::decode_iq3s_to_q8_0_scalar(block, sub_idx, output[sub_idx].qs, &output[sub_idx].d);
        }

        // All blocks should have valid scales
        for (size_t i = 0; i < 8; ++i)
        {
            float scale = fp16_to_fp32(output[i].d);
            EXPECT_GT(scale, 0.0f) << "Seed " << seed << ", block " << i;
        }
    }
}

// ============================================================================
// Test 3: Grid Lookup Verification
// ============================================================================

TEST_F(Test__IQ3_S_DecodeVectorization, GridLookupVerification)
{
    auto block = createTestBlock(999);

    // Verify grid indices are in bounds (0-511 for iq3s_grid[512])
    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        const size_t qs_offset = sub_idx * 8;
        const uint8_t qh_byte = block.qh[sub_idx];

        for (size_t l = 0; l < 4; ++l)
        {
            uint16_t grid_idx1 = block.qs[qs_offset + 2 * l + 0] | ((qh_byte << (8 - 2 * l)) & 256);
            uint16_t grid_idx2 = block.qs[qs_offset + 2 * l + 1] | ((qh_byte << (7 - 2 * l)) & 256);

            EXPECT_LT(grid_idx1, 512) << "Sub-block " << sub_idx << ", group " << l << ", grid1";
            EXPECT_LT(grid_idx2, 512) << "Sub-block " << sub_idx << ", group " << l << ", grid2";
        }
    }
}

// ============================================================================
// Test 4: AVX2 Parity
// ============================================================================

#if defined(__AVX2__)
TEST_F(Test__IQ3_S_DecodeVectorization, AVX2Parity)
{
    auto block = createTestBlock(777);

    Q8_0Block scalar_output[8];
    Q8_0Block avx2_output[8];

    // Decode with scalar
    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        simd::decode_iq3s_to_q8_0_scalar(block, sub_idx, scalar_output[sub_idx].qs, &scalar_output[sub_idx].d);
    }

    // Decode with AVX2
    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        simd::decode_iq3s_to_q8_0_avx2(block, sub_idx, avx2_output[sub_idx].qs, &avx2_output[sub_idx].d);
    }

    // Compare results
    EXPECT_TRUE(compareQ8_0BlockArrays(scalar_output, avx2_output, 8))
        << "AVX2 output does not match scalar";
}
#endif

// ============================================================================
// Test 5: AVX-512 Parity
// ============================================================================

#if defined(__AVX512F__)
TEST_F(Test__IQ3_S_DecodeVectorization, AVX512Parity)
{
    auto block = createTestBlock(888);

    Q8_0Block scalar_output[8];
    Q8_0Block avx512_output[8];

    // Decode with scalar
    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        simd::decode_iq3s_to_q8_0_scalar(block, sub_idx, scalar_output[sub_idx].qs, &scalar_output[sub_idx].d);
    }

    // Decode with AVX-512
    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        simd::decode_iq3s_to_q8_0_avx512(block, sub_idx, avx512_output[sub_idx].qs, &avx512_output[sub_idx].d);
    }

    // Compare results
    EXPECT_TRUE(compareQ8_0BlockArrays(scalar_output, avx512_output, 8))
        << "AVX-512 output does not match scalar";
}
#endif

// ============================================================================
// Test 6: Tensor Integration
// ============================================================================

TEST_F(Test__IQ3_S_DecodeVectorization, TensorIntegration)
{
    // Create a small IQ3_S tensor (1 row, 256 elements = 1 super-block)
    std::vector<size_t> shape = {1, 256};

    // Create raw data (110 bytes per super-block)
    std::vector<uint8_t> raw_data(110);
    IQ3_SBlock *block = reinterpret_cast<IQ3_SBlock *>(raw_data.data());
    *block = createTestBlock(555);

    // Create tensor
    auto tensor = std::make_shared<IQ3_STensor>(shape, raw_data);

    // Decode via tensor interface
    Q8_0Block tensor_output[8];
    tensor->decode_to_q8_0(0, 0, tensor_output);

    // Decode via SIMD helper (for comparison)
    Q8_0Block simd_output[8];
    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        simd::decode_iq3s_to_q8_0(*block, sub_idx, simd_output[sub_idx].qs, &simd_output[sub_idx].d);
    }

    // Compare results
    EXPECT_TRUE(compareQ8_0BlockArrays(tensor_output, simd_output, 8))
        << "Tensor decode does not match SIMD helper";
}

// ============================================================================
// Test 7: Auto-Dispatch
// ============================================================================

TEST_F(Test__IQ3_S_DecodeVectorization, AutoDispatch)
{
    auto block = createTestBlock(333);

    Q8_0Block scalar_output[8];
    Q8_0Block dispatch_output[8];

    // Decode with scalar
    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        simd::decode_iq3s_to_q8_0_scalar(block, sub_idx, scalar_output[sub_idx].qs, &scalar_output[sub_idx].d);
    }

    // Decode with auto-dispatch
    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        simd::decode_iq3s_to_q8_0(block, sub_idx, dispatch_output[sub_idx].qs, &dispatch_output[sub_idx].d);
    }

    // Compare results
    EXPECT_TRUE(compareQ8_0BlockArrays(scalar_output, dispatch_output, 8))
        << "Auto-dispatch output does not match scalar";
}

// ============================================================================
// Test 8: Fuzz Testing
// ============================================================================

TEST_F(Test__IQ3_S_DecodeVectorization, FuzzTesting)
{
    const size_t num_tests = 1000;

    for (size_t test_idx = 0; test_idx < num_tests; ++test_idx)
    {
        auto block = createTestBlock(test_idx * 13 + 7);

        Q8_0Block output[8];

        // Decode all sub-blocks
        for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
        {
            simd::decode_iq3s_to_q8_0_scalar(block, sub_idx, output[sub_idx].qs, &output[sub_idx].d);
        }

        // Validate all blocks
        for (size_t i = 0; i < 8; ++i)
        {
            float scale = fp16_to_fp32(output[i].d);
            ASSERT_GT(scale, 0.0f) << "Test " << test_idx << ", block " << i;

            for (size_t j = 0; j < 32; ++j)
            {
                ASSERT_GE(output[i].qs[j], -127) << "Test " << test_idx << ", block " << i << ", element " << j;
                ASSERT_LE(output[i].qs[j], 127) << "Test " << test_idx << ", block " << i << ", element " << j;
            }
        }
    }
}

// ============================================================================
// Test 9: Error Handling
// ============================================================================

TEST_F(Test__IQ3_S_DecodeVectorization, ErrorHandling)
{
    auto block = createTestBlock(111);
    Q8_0Block output[8];

    // Test boundary sub-block indices
    EXPECT_NO_THROW({
        simd::decode_iq3s_to_q8_0_scalar(block, 0, output[0].qs, &output[0].d);
        simd::decode_iq3s_to_q8_0_scalar(block, 7, output[7].qs, &output[7].d);
    });
}

// ============================================================================
// Test 10: Performance
// ============================================================================

TEST_F(Test__IQ3_S_DecodeVectorization, Performance)
{
    const size_t num_iterations = 10000;
    auto block = createTestBlock(12345);
    Q8_0Block output[8];

    // Benchmark scalar
    auto start_scalar = std::chrono::high_resolution_clock::now();
    for (size_t iter = 0; iter < num_iterations; ++iter)
    {
        for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
        {
            simd::decode_iq3s_to_q8_0_scalar(block, sub_idx, output[sub_idx].qs, &output[sub_idx].d);
        }
    }
    auto end_scalar = std::chrono::high_resolution_clock::now();
    double scalar_ms = std::chrono::duration<double, std::milli>(end_scalar - start_scalar).count();

#if defined(__AVX2__)
    // Benchmark AVX2
    auto start_avx2 = std::chrono::high_resolution_clock::now();
    for (size_t iter = 0; iter < num_iterations; ++iter)
    {
        for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
        {
            simd::decode_iq3s_to_q8_0_avx2(block, sub_idx, output[sub_idx].qs, &output[sub_idx].d);
        }
    }
    auto end_avx2 = std::chrono::high_resolution_clock::now();
    double avx2_ms = std::chrono::duration<double, std::milli>(end_avx2 - start_avx2).count();
#endif

#if defined(__AVX512F__)
    // Benchmark AVX-512
    auto start_avx512 = std::chrono::high_resolution_clock::now();
    for (size_t iter = 0; iter < num_iterations; ++iter)
    {
        for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
        {
            simd::decode_iq3s_to_q8_0_avx512(block, sub_idx, output[sub_idx].qs, &output[sub_idx].d);
        }
    }
    auto end_avx512 = std::chrono::high_resolution_clock::now();
    double avx512_ms = std::chrono::duration<double, std::milli>(end_avx512 - start_avx512).count();
#endif

    std::cout << "\nIQ3_S Decode Performance (" << num_iterations << " iterations):\n";
    std::cout << "  Scalar:   " << scalar_ms << " ms (baseline)\n";

#if defined(__AVX2__)
    double avx2_speedup = scalar_ms / avx2_ms;
    std::cout << "  AVX2:     " << avx2_ms << " ms (speedup: " << avx2_speedup << "x)\n";
#endif

#if defined(__AVX512F__)
    double avx512_speedup = scalar_ms / avx512_ms;
    std::cout << "  AVX-512:  " << avx512_ms << " ms (speedup: " << avx512_speedup << "x)\n";
#endif
}
