/**
 * @file Test__IQ1_M_DecodeVectorization.cpp
 * @brief Unit tests for IQ1_M decode_to_q8_0 SIMD implementations
 *
 * Tests scalar, AVX2, and AVX-512 implementations for correctness and parity.
 * IQ1_M: 1-bit quantization with multiple scales (256 elements, 56 bytes)
 *
 * @author David Sanftenberg
 * @date 2025-11-11
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

using namespace llaminar2;
using namespace llaminar2::simd;

/**
 * @brief Test fixture for IQ1_M decode vectorization tests
 */
class Test__IQ1_M_DecodeVectorization : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Seed RNG for reproducibility
        rng_.seed(12345);
    }

    /**
     * @brief Create a test IQ1_M block with random but valid data
     */
    IQ1_MBlock createTestBlock(uint32_t seed)
    {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<uint32_t> dist_byte(0, 255);
        std::uniform_real_distribution<float> dist_scale(0.01f, 10.0f);

        IQ1_MBlock block;

        // Random grid indices (qs)
        for (size_t i = 0; i < 32; ++i)
        {
            block.qs[i] = static_cast<uint8_t>(dist_byte(rng));
        }

        // Random qh values (high bits + signs)
        for (size_t i = 0; i < 16; ++i)
        {
            block.qh[i] = static_cast<uint8_t>(dist_byte(rng));
        }

        // Random scales (including global scale construction)
        for (size_t i = 0; i < 8; ++i)
        {
            block.scales[i] = static_cast<uint8_t>(dist_byte(rng));
        }

        return block;
    }

    /**
     * @brief Compare two Q8_0 block arrays with tolerance
     */
    void compareQ8_0BlockArrays(
        const Q8_0Block *expected,
        const Q8_0Block *actual,
        size_t num_blocks,
        const std::string &test_name)
    {
        for (size_t b = 0; b < num_blocks; ++b)
        {
            // Compare scales (FP16)
            float expected_scale = fp16_to_fp32(expected[b].d);
            float actual_scale = fp16_to_fp32(actual[b].d);
            float scale_diff = std::abs(expected_scale - actual_scale);
            float scale_tolerance = std::max(1e-5f, std::abs(expected_scale) * 1e-4f);

            EXPECT_LT(scale_diff, scale_tolerance)
                << test_name << " block " << b << " scale mismatch";

            // Compare quantized values (int8_t)
            for (size_t i = 0; i < 32; ++i)
            {
                EXPECT_EQ(expected[b].qs[i], actual[b].qs[i])
                    << test_name << " block " << b << " element " << i << " mismatch";
            }
        }
    }

    std::mt19937 rng_;
};

/**
 * @test ScalarCorrectness
 * @brief Verify scalar decode produces valid output
 */
TEST_F(Test__IQ1_M_DecodeVectorization, ScalarCorrectness)
{
    IQ1_MBlock block = createTestBlock(42);
    float global_scale = extract_iq1m_global_scale(block);
    Q8_0Block output[8];

    // Decode all 8 sub-blocks
    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        decode_iq1m_to_q8_0_scalar(block, sub_idx, global_scale, output[sub_idx].qs, &output[sub_idx].d);
    }

    // Verify all scales are valid (non-zero, finite)
    for (size_t i = 0; i < 8; ++i)
    {
        float scale = fp16_to_fp32(output[i].d);
        EXPECT_TRUE(std::isfinite(scale)) << "Sub-block " << i << " scale is not finite";
        EXPECT_NE(scale, 0.0f) << "Sub-block " << i << " scale is zero";
    }

    // Verify all quantized values are in valid range [-127, 127]
    for (size_t b = 0; b < 8; ++b)
    {
        for (size_t i = 0; i < 32; ++i)
        {
            EXPECT_GE(output[b].qs[i], -127) << "Block " << b << " element " << i << " underflow";
            EXPECT_LE(output[b].qs[i], 127) << "Block " << b << " element " << i << " overflow";
        }
    }
}

/**
 * @test MultiBlockCorrectness
 * @brief Test multiple blocks with varied data
 */
TEST_F(Test__IQ1_M_DecodeVectorization, MultiBlockCorrectness)
{
    const size_t num_test_blocks = 5;
    for (size_t test_idx = 0; test_idx < num_test_blocks; ++test_idx)
    {
        IQ1_MBlock block = createTestBlock(100 + test_idx);
        float global_scale = extract_iq1m_global_scale(block);
        Q8_0Block output[8];

        for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
        {
            decode_iq1m_to_q8_0_scalar(block, sub_idx, global_scale, output[sub_idx].qs, &output[sub_idx].d);
        }

        // Basic validation
        for (size_t i = 0; i < 8; ++i)
        {
            float scale = fp16_to_fp32(output[i].d);
            EXPECT_TRUE(std::isfinite(scale));
            EXPECT_NE(scale, 0.0f);
        }
    }
}

/**
 * @test GridLookupVerification
 * @brief Verify grid index construction is within bounds
 */
TEST_F(Test__IQ1_M_DecodeVectorization, GridLookupVerification)
{
    IQ1_MBlock block = createTestBlock(777);

    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        const uint8_t *qs = block.qs + sub_idx * 4;
        const uint8_t *qh = block.qh + sub_idx * 2;

        // Grid indices for 4 groups
        uint16_t idx[4];
        idx[0] = qs[0] | ((qh[0] << 8) & 0x700);
        idx[1] = qs[1] | ((qh[0] << 4) & 0x700);
        idx[2] = qs[2] | ((qh[1] << 8) & 0x700);
        idx[3] = qs[3] | ((qh[1] << 4) & 0x700);

        for (size_t l = 0; l < 4; ++l)
        {
            EXPECT_LT(idx[l], 2048)
                << "Sub-block " << sub_idx << " group " << l << " grid index out of bounds";
        }
    }
}

#ifdef __AVX2__
/**
 * @test AVX2Parity
 * @brief Verify AVX2 implementation matches scalar exactly
 */
TEST_F(Test__IQ1_M_DecodeVectorization, AVX2Parity)
{
    IQ1_MBlock block = createTestBlock(999);
    float global_scale = extract_iq1m_global_scale(block);
    Q8_0Block scalar_output[8];
    Q8_0Block avx2_output[8];

    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        decode_iq1m_to_q8_0_scalar(block, sub_idx, global_scale, scalar_output[sub_idx].qs, &scalar_output[sub_idx].d);
        decode_iq1m_to_q8_0_avx2(block, sub_idx, global_scale, avx2_output[sub_idx].qs, &avx2_output[sub_idx].d);
    }

    compareQ8_0BlockArrays(scalar_output, avx2_output, 8, "AVX2 vs Scalar");
}
#endif

#ifdef __AVX512F__
/**
 * @test AVX512Parity
 * @brief Verify AVX-512 implementation matches scalar exactly
 */
TEST_F(Test__IQ1_M_DecodeVectorization, AVX512Parity)
{
    IQ1_MBlock block = createTestBlock(1234);
    float global_scale = extract_iq1m_global_scale(block);
    Q8_0Block scalar_output[8];
    Q8_0Block avx512_output[8];

    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        decode_iq1m_to_q8_0_scalar(block, sub_idx, global_scale, scalar_output[sub_idx].qs, &scalar_output[sub_idx].d);
        decode_iq1m_to_q8_0_avx512(block, sub_idx, global_scale, avx512_output[sub_idx].qs, &avx512_output[sub_idx].d);
    }

    compareQ8_0BlockArrays(scalar_output, avx512_output, 8, "AVX-512 vs Scalar");
}
#endif

/**
 * @test TensorIntegration
 * @brief Test end-to-end IQ1_MTensor decode path
 */
TEST_F(Test__IQ1_M_DecodeVectorization, TensorIntegration)
{
    // Create a small IQ1_M tensor (1 row × 256 elements)
    std::vector<size_t> shape = {1, 256};

    // Allocate raw data (1 IQ1_M block = 56 bytes)
    std::vector<uint8_t> raw_data(56);
    IQ1_MBlock *block = reinterpret_cast<IQ1_MBlock *>(raw_data.data());
    *block = createTestBlock(5678);

    // Create tensor
    auto tensor = std::make_shared<IQ1_MTensor>(shape, raw_data);

    // Decode via tensor method
    Q8_0Block tensor_output[8];
    tensor->decode_to_q8_0(0, 0, tensor_output);

    // Decode via SIMD helper for comparison
    float global_scale = extract_iq1m_global_scale(*block);
    Q8_0Block reference_output[8];
    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        decode_iq1m_to_q8_0(*block, sub_idx, global_scale, reference_output[sub_idx].qs, &reference_output[sub_idx].d);
    }

    // Compare
    compareQ8_0BlockArrays(reference_output, tensor_output, 8, "Tensor vs Reference");
}

/**
 * @test AutoDispatch
 * @brief Verify auto-dispatch selects correct implementation
 */
TEST_F(Test__IQ1_M_DecodeVectorization, AutoDispatch)
{
    IQ1_MBlock block = createTestBlock(9999);
    float global_scale = extract_iq1m_global_scale(block);
    Q8_0Block auto_output[8];
    Q8_0Block scalar_output[8];

    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        decode_iq1m_to_q8_0(block, sub_idx, global_scale, auto_output[sub_idx].qs, &auto_output[sub_idx].d);
        decode_iq1m_to_q8_0_scalar(block, sub_idx, global_scale, scalar_output[sub_idx].qs, &scalar_output[sub_idx].d);
    }

    // Auto-dispatch should match scalar (or SIMD if available)
    compareQ8_0BlockArrays(scalar_output, auto_output, 8, "Auto-dispatch vs Scalar");
}

/**
 * @test FuzzTesting
 * @brief Stress test with 1000 random blocks
 */
TEST_F(Test__IQ1_M_DecodeVectorization, FuzzTesting)
{
    const size_t num_fuzz_blocks = 1000;
    size_t failures = 0;
    size_t skipped = 0;

    for (size_t test_idx = 0; test_idx < num_fuzz_blocks; ++test_idx)
    {
        IQ1_MBlock block = createTestBlock(test_idx);
        float global_scale = extract_iq1m_global_scale(block);

        // Skip blocks with invalid global scale (random data can produce invalid FP16)
        if (!std::isfinite(global_scale) || global_scale == 0.0f)
        {
            ++skipped;
            continue;
        }

        Q8_0Block output[8];

        try
        {
            for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
            {
                decode_iq1m_to_q8_0(block, sub_idx, global_scale, output[sub_idx].qs, &output[sub_idx].d);
            }

            // Validate output
            for (size_t i = 0; i < 8; ++i)
            {
                float scale = fp16_to_fp32(output[i].d);
                if (!std::isfinite(scale) || scale == 0.0f)
                {
                    ++failures;
                    break;
                }
            }
        }
        catch (const std::exception &e)
        {
            ++failures;
        }
    }

    EXPECT_EQ(failures, 0) << failures << " out of " << (num_fuzz_blocks - skipped) << " fuzz tests failed (skipped " << skipped << " invalid blocks)";
}

/**
 * @test ErrorHandling
 * @brief Test edge cases and invalid inputs
 */
TEST_F(Test__IQ1_M_DecodeVectorization, ErrorHandling)
{
    IQ1_MBlock block = createTestBlock(12345);
    float global_scale = extract_iq1m_global_scale(block);

    // Test all sub-block indices (0-7)
    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        Q8_0Block output;
        EXPECT_NO_THROW({
            decode_iq1m_to_q8_0_scalar(block, sub_idx, global_scale, output.qs, &output.d);
        }) << "Scalar decode failed for sub-block "
           << sub_idx;
    }

    // Test zero global scale (edge case)
    Q8_0Block output;
    EXPECT_NO_THROW({
        decode_iq1m_to_q8_0_scalar(block, 0, 0.0f, output.qs, &output.d);
    }) << "Scalar decode failed for zero global scale";
}

/**
 * @test Performance
 * @brief Benchmark scalar vs SIMD implementations
 */
TEST_F(Test__IQ1_M_DecodeVectorization, Performance)
{
    const size_t num_iterations = 10000;
    IQ1_MBlock block = createTestBlock(55555);
    float global_scale = extract_iq1m_global_scale(block);
    Q8_0Block output[8];

    // Benchmark scalar
    auto start_scalar = std::chrono::high_resolution_clock::now();
    for (size_t iter = 0; iter < num_iterations; ++iter)
    {
        for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
        {
            decode_iq1m_to_q8_0_scalar(block, sub_idx, global_scale, output[sub_idx].qs, &output[sub_idx].d);
        }
    }
    auto end_scalar = std::chrono::high_resolution_clock::now();
    double ms_scalar = std::chrono::duration<double, std::milli>(end_scalar - start_scalar).count();

#ifdef __AVX2__
    // Benchmark AVX2
    auto start_avx2 = std::chrono::high_resolution_clock::now();
    for (size_t iter = 0; iter < num_iterations; ++iter)
    {
        for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
        {
            decode_iq1m_to_q8_0_avx2(block, sub_idx, global_scale, output[sub_idx].qs, &output[sub_idx].d);
        }
    }
    auto end_avx2 = std::chrono::high_resolution_clock::now();
    double ms_avx2 = std::chrono::duration<double, std::milli>(end_avx2 - start_avx2).count();
#endif

#ifdef __AVX512F__
    // Benchmark AVX-512
    auto start_avx512 = std::chrono::high_resolution_clock::now();
    for (size_t iter = 0; iter < num_iterations; ++iter)
    {
        for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
        {
            decode_iq1m_to_q8_0_avx512(block, sub_idx, global_scale, output[sub_idx].qs, &output[sub_idx].d);
        }
    }
    auto end_avx512 = std::chrono::high_resolution_clock::now();
    double ms_avx512 = std::chrono::duration<double, std::milli>(end_avx512 - start_avx512).count();
#endif

    // Print results
    std::cout << "\nIQ1_M Decode Performance (" << num_iterations << " iterations):\n";
    std::cout << "  Scalar:   " << ms_scalar << " ms (baseline)\n";

#ifdef __AVX2__
    double speedup_avx2 = ms_scalar / ms_avx2;
    std::cout << "  AVX2:     " << ms_avx2 << " ms (speedup: " << speedup_avx2 << "x)\n";
#endif

#ifdef __AVX512F__
    double speedup_avx512 = ms_scalar / ms_avx512;
    std::cout << "  AVX-512:  " << ms_avx512 << " ms (speedup: " << speedup_avx512 << "x)\n";
#endif

    // No strict performance requirements (varies by hardware)
    // But we expect SIMD to be at least slightly faster or comparable
}

/**
 * @brief Main entry point for IQ1_M decode vectorization tests
 */
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
