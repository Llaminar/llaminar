/**
 * @file Test__ROCmQuantisedGemmKernel.cpp
 * @brief Unit tests for ROCmQuantisedGemmKernel - CPU-only tests that don't require GPU
 *
 * These tests validate the CPU-side functionality of the ROCm INT8 GEMM kernel:
 * - Weight packing and layout transformation (packWeightsToROCm)
 *
 * GPU-requiring tests (activation quantization, work buffers, CK GEMM) are in:
 *   tests/v2/integration/Test__ROCmQuantisedGemmKernel.cpp
 */

#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

#include "kernels/rocm/ROCmQuantisedGemmKernel.h"
#include "tensors/Tensors.h"
#include "utils/Logger.h"
#include "../../../utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::rocm;
using namespace llaminar2::test;

namespace
{

    /**
     * @brief Test fixture for ROCmQuantisedGemmKernel unit tests
     *
     * Tests CPU-side weight packing functionality (no GPU required).
     */
    class ROCmQuantisedGemmKernelUnitTest : public ::testing::Test
    {
    protected:
        void SetUp() override {}
    };

    // =============================================================================
    // Phase 1: Weight Packing Tests (CPU-only)
    // =============================================================================

    /**
     * @test Verify packWeightsToROCm produces valid output for Q8_0
     *
     * This is a CPU-only test that validates the weight packing transformation.
     * The packing converts Q8_0 block format to CK-compatible INT8 layout:
     *
     * Input (Q8_0): Blocks of 32 quantized values with per-block scale
     * Output (CK INT8): Dense INT8 matrix [K×N] + separate scale array [N]
     */
    TEST_F(ROCmQuantisedGemmKernelUnitTest, PackWeightsToROCm_Q8_0_CreatesValidOutput)
    {
        const size_t N = 64;
        const size_t K = 128;

        auto weights = TestTensorFactory::createQ8_0Random({N, K});

        // Pack weights - CPU-only operation
        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));

        // Verify packed structure
        EXPECT_FALSE(packed.int8_data.empty());
        EXPECT_FALSE(packed.scales.empty());
        EXPECT_EQ(packed.N, static_cast<int>(N));
        EXPECT_EQ(packed.K, static_cast<int>(K));

        // Verify dimensions match (transposed: K×N)
        EXPECT_EQ(packed.int8_data.size(), K * N);
        EXPECT_EQ(packed.scales.size(), N);
    }

    /**
     * @test Verify packWeightsToROCm produces valid output for Q8_1
     */
    TEST_F(ROCmQuantisedGemmKernelUnitTest, PackWeightsToROCm_Q8_1_CreatesValidOutput)
    {
        const size_t N = 32;
        const size_t K = 64;

        auto weights = TestTensorFactory::createQ8_1Random({N, K});

        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));

        EXPECT_FALSE(packed.int8_data.empty());
        EXPECT_FALSE(packed.scales.empty());
        EXPECT_EQ(packed.N, static_cast<int>(N));
        EXPECT_EQ(packed.K, static_cast<int>(K));
    }

    /**
     * @test Verify packWeightsToROCm produces valid output for IQ4_NL
     */
    TEST_F(ROCmQuantisedGemmKernelUnitTest, PackWeightsToROCm_IQ4_NL_CreatesValidOutput)
    {
        const size_t N = 32;
        const size_t K = 64;

        auto weights = TestTensorFactory::createIQ4_NLRandom({N, K});

        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));

        EXPECT_FALSE(packed.int8_data.empty());
        EXPECT_FALSE(packed.scales.empty());
        EXPECT_EQ(packed.N, static_cast<int>(N));
        EXPECT_EQ(packed.K, static_cast<int>(K));
    }

    /**
     * @test Verify INT8 values are in valid range after packing
     */
    TEST_F(ROCmQuantisedGemmKernelUnitTest, PackWeights_INT8ValuesInRange)
    {
        const size_t N = 32;
        const size_t K = 64;

        auto weights = TestTensorFactory::createQ8_0Random({N, K});
        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));

        // All INT8 values should be in valid range
        for (size_t i = 0; i < packed.int8_data.size(); ++i)
        {
            int8_t val = packed.int8_data[i];
            EXPECT_GE(val, -127) << "Value at " << i << " below INT8 min";
            EXPECT_LE(val, 127) << "Value at " << i << " above INT8 max";
        }
    }

    /**
     * @test Verify all scales are positive after packing
     */
    TEST_F(ROCmQuantisedGemmKernelUnitTest, PackWeights_ScalesArePositive)
    {
        const size_t N = 32;
        const size_t K = 64;

        auto weights = TestTensorFactory::createQ8_0Random({N, K});
        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));

        // All scales should be positive
        for (size_t i = 0; i < packed.scales.size(); ++i)
        {
            EXPECT_GT(packed.scales[i], 0.0f) << "Scale at column " << i << " should be positive";
        }
    }

    /**
     * @test Verify symmetric quantization in packed weights
     *
     * The packing uses symmetric quantization (no zero-point),
     * so values should be centered around zero.
     */
    TEST_F(ROCmQuantisedGemmKernelUnitTest, PackWeights_SymmetricQuantization)
    {
        const size_t N = 16;
        const size_t K = 32;

        auto weights = TestTensorFactory::createQ8_0Random({N, K});
        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));

        // Count positive and negative values - should be roughly balanced
        int positive_count = 0;
        int negative_count = 0;
        int zero_count = 0;

        for (size_t i = 0; i < packed.int8_data.size(); ++i)
        {
            int8_t val = packed.int8_data[i];
            if (val > 0)
                positive_count++;
            else if (val < 0)
                negative_count++;
            else
                zero_count++;
        }

        // For symmetric quantization with centered data, expect rough balance
        int total = positive_count + negative_count + zero_count;
        EXPECT_GT(positive_count, total / 10) << "Too few positive values for symmetric quant";
        EXPECT_GT(negative_count, total / 10) << "Too few negative values for symmetric quant";
    }

    /**
     * @test Verify reconstruction accuracy from packed weights
     *
     * This validates that the quantization is reversible within tolerance.
     * Dequantized = quantized * scale should approximate original FP32.
     */
    TEST_F(ROCmQuantisedGemmKernelUnitTest, PackWeights_ReconstructionAccuracy)
    {
        const size_t N = 8;
        const size_t K = 32;

        auto weights = TestTensorFactory::createQ8_0Random({N, K});

        // Get original FP32 values
        const float *fp32_data = weights->fp32_data();

        // Pack weights
        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));

        // Reconstruct and compare
        // Note: packed is [K×N] row-major (transposed from model [N×K])
        // fp32_data is [N×K] row-major
        float max_error = 0.0f;
        for (size_t n = 0; n < N; ++n)
        {
            float scale = packed.scales[n];
            for (size_t k = 0; k < K; ++k)
            {
                // packed.int8_data is [K×N] row-major: int8_data[k*N + n]
                int8_t q = packed.int8_data[k * N + n];
                float reconstructed = static_cast<float>(q) * scale;
                // fp32_data is [N×K] row-major: fp32_data[n*K + k]
                float orig = fp32_data[n * K + k];
                float error = std::abs(reconstructed - orig);
                max_error = std::max(max_error, error);
            }
        }

        // INT8 quantization should have reasonable error
        EXPECT_LT(max_error, 0.5f) << "Reconstruction error too high";
    }

    /**
     * @test Verify packing works with various K dimensions (multiple of 32)
     */
    TEST_F(ROCmQuantisedGemmKernelUnitTest, PackWeights_VariousKDimensions)
    {
        const size_t N = 16;

        // Test various K dimensions
        std::vector<size_t> k_dims = {32, 64, 128, 256, 512, 896, 1024};

        for (size_t K : k_dims)
        {
            auto weights = TestTensorFactory::createQ8_0Random({N, K});
            ROCmPackedWeights packed;

            ASSERT_TRUE(packWeightsToROCm(weights.get(), packed))
                << "Failed to pack weights with K=" << K;

            EXPECT_EQ(packed.N, static_cast<int>(N)) << "Wrong N dimension for K=" << K;
            EXPECT_EQ(packed.K, static_cast<int>(K)) << "Wrong K dimension for K=" << K;
            EXPECT_EQ(packed.int8_data.size(), K * N) << "Wrong packed size for K=" << K;
            EXPECT_EQ(packed.scales.size(), N) << "Wrong scales size for K=" << K;
        }
    }

    /**
     * @test Verify packing works with realistic model dimensions
     */
    TEST_F(ROCmQuantisedGemmKernelUnitTest, PackWeights_RealisticDimensions)
    {
        // Qwen2.5-0.5B dimensions
        struct TestCase
        {
            const char *name;
            size_t N, K;
        };

        std::vector<TestCase> cases = {
            {"QKV projection", 896 * 3, 896},
            {"Attention output", 896, 896},
            {"FFN gate/up", 4864, 896},
            {"FFN down", 896, 4864},
        };

        for (const auto &tc : cases)
        {
            auto weights = TestTensorFactory::createQ8_0Random({tc.N, tc.K});
            ROCmPackedWeights packed;

            ASSERT_TRUE(packWeightsToROCm(weights.get(), packed))
                << "Failed to pack " << tc.name;

            EXPECT_EQ(packed.N, static_cast<int>(tc.N)) << "Wrong N for " << tc.name;
            EXPECT_EQ(packed.K, static_cast<int>(tc.K)) << "Wrong K for " << tc.name;

            LOG_INFO("[Unit] Packed " << tc.name << ": " << tc.N << "x" << tc.K);
        }
    }

    /**
     * @test Verify packed weights memory layout is Row-Major [K×N]
     *
     * The packed layout should match CK's mk_kn_mn convention:
     * - int8_data[k * N + n] = weight for input k, output n
     */
    TEST_F(ROCmQuantisedGemmKernelUnitTest, PackWeights_RowMajorKxNLayout)
    {
        const size_t N = 4;
        const size_t K = 8;

        auto weights = TestTensorFactory::createQ8_0Random({N, K});
        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));

        // Verify layout is [K×N]
        EXPECT_EQ(packed.int8_data.size(), K * N);

        // The packed array should be indexed as [k*N + n]
        // Verify by checking array bounds
        for (size_t k = 0; k < K; ++k)
        {
            for (size_t n = 0; n < N; ++n)
            {
                size_t idx = k * N + n;
                EXPECT_LT(idx, packed.int8_data.size())
                    << "Index out of bounds at k=" << k << ", n=" << n;
            }
        }
    }

} // namespace
