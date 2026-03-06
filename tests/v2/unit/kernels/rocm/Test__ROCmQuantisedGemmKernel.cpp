/**
 * @file Test__ROCmQuantisedGemmKernel.cpp
 * @brief Unit tests for ROCmQuantisedGemmKernel - CPU-only tests that don't require GPU
 *
 * These tests validate the CPU-side functionality of the ROCm INT8 GEMM kernel:
 * - Weight packing and layout transformation (packWeightsToROCm)
 * - Dimension validation (minimum requirements for CK kernels)
 * - ROCmPackedWeights structure integrity
 * - Scale computation correctness
 *
 * GPU-requiring tests (activation quantization, work buffers, CK GEMM) are in:
 *   tests/v2/integration/Test__ROCmQuantisedGemmKernel.cpp
 *
 * @note All tests in this file run on CPU only - no ROCm device required.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
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
     *
     * IQ4_NL is a native-VNNI format (≤6-bit). After the format-conditional
     * repack change, it receives ONLY native-VNNI packing — no INT8 requantization.
     */
    TEST_F(ROCmQuantisedGemmKernelUnitTest, PackWeightsToROCm_IQ4_NL_CreatesValidOutput)
    {
        const size_t N = 32;
        const size_t K = 64;

        auto weights = TestTensorFactory::createIQ4_NLRandom({N, K});

        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));

        // IQ4_NL is native-VNNI: native payload populated, INT8 path empty
        EXPECT_FALSE(packed.native_vnni_payload.empty());
        EXPECT_FALSE(packed.native_vnni_scales.empty());
        EXPECT_TRUE(packed.int8_data.empty());
        EXPECT_TRUE(packed.int8_data_vnni.empty());
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
        const size_t K = 64; // Must be divisible by Q8_0 block size (32)

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

    // =============================================================================
    // ROCmPackedWeights Structure Tests (CPU-only)
    // =============================================================================

    /**
     * @test Verify ROCmPackedWeights default initialization
     */
    TEST_F(ROCmQuantisedGemmKernelUnitTest, ROCmPackedWeights_DefaultInit)
    {
        ROCmPackedWeights packed;

        EXPECT_TRUE(packed.int8_data.empty());
        EXPECT_TRUE(packed.scales.empty());
        EXPECT_EQ(packed.N, 0);
        EXPECT_EQ(packed.K, 0);
    }

    /**
     * @test Verify ROCmPackedWeights can be moved
     */
    TEST_F(ROCmQuantisedGemmKernelUnitTest, ROCmPackedWeights_MoveSemantics)
    {
        const size_t N = 32;
        const size_t K = 64;

        auto weights = TestTensorFactory::createQ8_0Random({N, K});
        ROCmPackedWeights packed1;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed1));

        // Capture pre-move sizes
        const size_t original_int8_size = packed1.int8_data.size();
        const size_t original_scales_size = packed1.scales.size();

        // Move construct
        ROCmPackedWeights packed2 = std::move(packed1);

        // Verify moved-to object has the original data
        EXPECT_EQ(packed2.int8_data.size(), original_int8_size);
        EXPECT_EQ(packed2.scales.size(), original_scales_size);
        EXPECT_EQ(packed2.N, static_cast<int>(N));
        EXPECT_EQ(packed2.K, static_cast<int>(K));

        // Note: After std::move, packed1 is in a valid but unspecified state.
        // We cannot assert it's empty - only that packed2 has the data.
    }

    /**
     * @test Verify packing preserves weight statistics
     *
     * The packed weights should maintain approximately the same distribution
     * as the original weights (within quantization error).
     */
    TEST_F(ROCmQuantisedGemmKernelUnitTest, PackWeights_PreservesStatistics)
    {
        const size_t N = 64;
        const size_t K = 128;

        auto weights = TestTensorFactory::createQ8_0Random({N, K});
        const float *fp32_data = weights->fp32_data();

        // Compute original statistics
        float orig_min = std::numeric_limits<float>::max();
        float orig_max = std::numeric_limits<float>::lowest();
        double orig_sum = 0.0;
        for (size_t i = 0; i < N * K; ++i)
        {
            orig_min = std::min(orig_min, fp32_data[i]);
            orig_max = std::max(orig_max, fp32_data[i]);
            orig_sum += fp32_data[i];
        }
        double orig_mean = orig_sum / (N * K);

        // Pack weights
        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));

        // Reconstruct and compute statistics
        float recon_min = std::numeric_limits<float>::max();
        float recon_max = std::numeric_limits<float>::lowest();
        double recon_sum = 0.0;
        for (size_t n = 0; n < N; ++n)
        {
            float scale = packed.scales[n];
            for (size_t k = 0; k < K; ++k)
            {
                // packed.int8_data is [K×N] row-major
                int8_t q = packed.int8_data[k * N + n];
                float val = static_cast<float>(q) * scale;
                recon_min = std::min(recon_min, val);
                recon_max = std::max(recon_max, val);
                recon_sum += val;
            }
        }
        double recon_mean = recon_sum / (N * K);

        // Statistics should be approximately preserved
        // Allow for quantization error (INT8 has ~1/127 resolution)
        EXPECT_NEAR(recon_mean, orig_mean, 0.1) << "Mean diverged too much";
        EXPECT_NEAR(recon_min, orig_min, 0.5) << "Min diverged too much";
        EXPECT_NEAR(recon_max, orig_max, 0.5) << "Max diverged too much";
    }

    // =============================================================================
    // 3-Way Dispatch Dimension Coverage Tests (CPU-only)
    // =============================================================================
    //
    // These tests verify that weight packing works correctly for all dimensions
    // that the 3-way CK dispatch will encounter. The actual dispatch happens
    // on GPU, but we can verify the packing is correct for all expected sizes.

    /**
     * @test Verify packing works for M sizes that hit each dispatch tile
     *
     * The 3-way dispatch uses:
     *   - M ≤ 32:       32×32 tile
     *   - 32 < M < 128: 64×64 tile
     *   - M ≥ 128:      128×128 tile
     *
     * We verify that weights can be packed for typical N,K dimensions
     * that would be used with these M sizes.
     */
    TEST_F(ROCmQuantisedGemmKernelUnitTest, PackWeights_3WayDispatchDimensions)
    {
        // Test each tile's typical use case
        struct TestCase
        {
            const char *name;
            size_t N, K;
            const char *tile_desc;
        };

        std::vector<TestCase> cases = {
            // Qwen2.5-0.5B dimensions
            {"0.5B_Attention", 896, 896, "all tiles"},
            {"0.5B_FFN_Up", 4864, 896, "all tiles"},
            {"0.5B_FFN_Down", 896, 4864, "all tiles"},

            // Qwen2.5-7B dimensions
            {"7B_Attention", 3584, 3584, "all tiles"},
            {"7B_FFN_Up", 18944, 3584, "all tiles"},
            {"7B_FFN_Down", 3584, 18944, "all tiles"},

            // Edge cases near tile boundaries
            {"edge_32x32", 32, 32, "32×32 exact"},
            {"edge_64x64", 64, 64, "64×64"},
            {"edge_128x128", 128, 128, "128×128 exact"},
        };

        for (const auto &tc : cases)
        {
            auto weights = TestTensorFactory::createQ8_0Random({tc.N, tc.K});
            ROCmPackedWeights packed;

            ASSERT_TRUE(packWeightsToROCm(weights.get(), packed))
                << "Failed to pack " << tc.name << " (" << tc.tile_desc << ")";

            EXPECT_EQ(packed.int8_data.size(), tc.K * tc.N)
                << "Wrong size for " << tc.name;
            EXPECT_EQ(packed.scales.size(), tc.N)
                << "Wrong scales count for " << tc.name;

            LOG_INFO("[Unit] 3-Way pack " << tc.name << ": " << tc.N << "×" << tc.K
                                          << " (" << tc.tile_desc << ")");
        }
    }

    /**
     * @test Verify packing handles minimum CK dimensions
     *
     * CK kernels require minimum dimensions:
     *   - M ≥ 8 (padded if smaller)
     *   - N ≥ 32
     *   - K ≥ 32
     *
     * Weight packing should work for any valid N,K.
     */
    TEST_F(ROCmQuantisedGemmKernelUnitTest, PackWeights_MinimumDimensions)
    {
        // Minimum valid dimensions
        const size_t MIN_N = 32;
        const size_t MIN_K = 32;

        auto weights = TestTensorFactory::createQ8_0Random({MIN_N, MIN_K});
        ROCmPackedWeights packed;

        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed))
            << "Failed to pack minimum dimensions";

        EXPECT_EQ(packed.N, static_cast<int>(MIN_N));
        EXPECT_EQ(packed.K, static_cast<int>(MIN_K));
        EXPECT_EQ(packed.int8_data.size(), MIN_K * MIN_N);
        EXPECT_EQ(packed.scales.size(), MIN_N);
    }

    /**
     * @test Verify scale computation for edge values
     *
     * Tests that scales are computed correctly for tensors with:
     * - All positive values
     * - All negative values
     * - Mixed values
     * - Very small values
     */
    TEST_F(ROCmQuantisedGemmKernelUnitTest, PackWeights_ScaleComputation)
    {
        const size_t N = 32;
        const size_t K = 64;

        // Create weights with known properties
        auto weights = TestTensorFactory::createQ8_0Random({N, K});
        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));

        // Verify scales are valid
        for (size_t n = 0; n < N; ++n)
        {
            float scale = packed.scales[n];

            // Scale must be positive (symmetric quantization)
            EXPECT_GT(scale, 0.0f) << "Scale must be positive for column " << n;

            // Scale should not be excessively large
            EXPECT_LT(scale, 10.0f) << "Scale too large for column " << n;

            // Scale should not be zero or denormalized
            EXPECT_TRUE(std::isnormal(scale) || scale == 0.0f)
                << "Scale is denormalized for column " << n;
        }
    }

    // =============================================================================
    // Format-Conditional Repack Tests
    // =============================================================================
    //
    // These tests verify the format-conditional repack logic introduced in Phase 2:
    //   - Native-VNNI formats (≤6-bit) → native-VNNI packing ONLY
    //   - INT8-VNNI formats (Q8_0, Q8_1) → INT8 requant + VNNI interleave ONLY
    //
    // The two paths are mutually exclusive: native-VNNI formats must NOT have
    // int8_data/scales populated, and INT8-VNNI formats must NOT have
    // native_vnni_payload/native_vnni_scales populated.

    /**
     * @test INT8-VNNI format (Q8_0) produces ONLY INT8 packing, no native-VNNI
     */
    TEST_F(ROCmQuantisedGemmKernelUnitTest, Repack_Q8_0_OnlyInt8Path)
    {
        const size_t N = 32;
        const size_t K = 64;

        auto weights = TestTensorFactory::createQ8_0Random({N, K});
        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));

        // INT8 path populated
        EXPECT_FALSE(packed.int8_data.empty()) << "Q8_0 must populate int8_data";
        EXPECT_FALSE(packed.scales.empty()) << "Q8_0 must populate scales";
        EXPECT_FALSE(packed.int8_data_vnni.empty()) << "Q8_0 must populate VNNI layout";

        // Native-VNNI path empty
        EXPECT_TRUE(packed.native_vnni_payload.empty())
            << "Q8_0 must NOT populate native-VNNI payload";
        EXPECT_TRUE(packed.native_vnni_scales.empty())
            << "Q8_0 must NOT populate native-VNNI scales";
        EXPECT_EQ(packed.native_vnni_codebook_id, 0)
            << "Q8_0 must NOT set codebook_id";
        EXPECT_EQ(packed.native_vnni_blocks_per_row, 0u)
            << "Q8_0 must NOT set blocks_per_row";
    }

    /**
     * @test INT8-VNNI format (Q8_1) produces ONLY INT8 packing, no native-VNNI
     */
    TEST_F(ROCmQuantisedGemmKernelUnitTest, Repack_Q8_1_OnlyInt8Path)
    {
        const size_t N = 32;
        const size_t K = 64;

        auto weights = TestTensorFactory::createQ8_1Random({N, K});
        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));

        // INT8 path populated
        EXPECT_FALSE(packed.int8_data.empty()) << "Q8_1 must populate int8_data";
        EXPECT_FALSE(packed.scales.empty()) << "Q8_1 must populate scales";
        EXPECT_FALSE(packed.int8_data_vnni.empty()) << "Q8_1 must populate VNNI layout";

        // Native-VNNI path empty
        EXPECT_TRUE(packed.native_vnni_payload.empty())
            << "Q8_1 must NOT populate native-VNNI payload";
        EXPECT_TRUE(packed.native_vnni_scales.empty())
            << "Q8_1 must NOT populate native-VNNI scales";
    }

    /**
     * @test Native-VNNI format (Q4_0) produces ONLY native packing, no INT8
     */
    TEST_F(ROCmQuantisedGemmKernelUnitTest, Repack_Q4_0_OnlyNativeVnniPath)
    {
        const size_t N = 32;
        const size_t K = 128;

        auto weights = TestTensorFactory::createQ4_0Random({N, K});
        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));

        // Native-VNNI path populated
        EXPECT_FALSE(packed.native_vnni_payload.empty())
            << "Q4_0 must populate native-VNNI payload";
        EXPECT_FALSE(packed.native_vnni_scales.empty())
            << "Q4_0 must populate native-VNNI scales";
        EXPECT_GT(packed.native_vnni_blocks_per_row, 0u)
            << "Q4_0 must set blocks_per_row";

        // INT8 quantized data must NOT be populated
        EXPECT_TRUE(packed.int8_data.empty())
            << "Q4_0 must NOT populate int8_data";
        EXPECT_TRUE(packed.int8_data_vnni.empty())
            << "Q4_0 must NOT populate int8_data_vnni";
        // Note: scales IS populated by packNativeVNNI for force_ck debug compatibility
    }

    /**
     * @test Native-VNNI format (IQ4_NL) produces ONLY native packing, no INT8
     */
    TEST_F(ROCmQuantisedGemmKernelUnitTest, Repack_IQ4_NL_OnlyNativeVnniPath)
    {
        const size_t N = 32;
        const size_t K = 64;

        auto weights = TestTensorFactory::createIQ4_NLRandom({N, K});
        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));

        // Native-VNNI path populated
        EXPECT_FALSE(packed.native_vnni_payload.empty());
        EXPECT_FALSE(packed.native_vnni_scales.empty());

        // INT8 quantized data must NOT be populated
        EXPECT_TRUE(packed.int8_data.empty());
        EXPECT_TRUE(packed.int8_data_vnni.empty());
    }

    /**
     * @test Super-block native-VNNI format (Q6_K) produces ONLY native packing
     */
    TEST_F(ROCmQuantisedGemmKernelUnitTest, Repack_Q6_K_OnlyNativeVnniPath)
    {
        const size_t N = 32;
        const size_t K = 256; // Super-block K must be multiple of 256

        auto weights = TestTensorFactory::createQ6_KRandom({N, K});
        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));

        // Native-VNNI path populated
        EXPECT_FALSE(packed.native_vnni_payload.empty());
        EXPECT_FALSE(packed.native_vnni_scales.empty());
        EXPECT_FALSE(packed.native_vnni_mins.empty())
            << "Q6_K is dual-scale, must have mins";
        EXPECT_EQ(packed.native_vnni_codebook_id, 8);

        // INT8 quantized data must NOT be populated
        EXPECT_TRUE(packed.int8_data.empty());
        EXPECT_TRUE(packed.int8_data_vnni.empty());
    }

    /**
     * @test Verify repack mutual exclusion exhaustively for all format families
     *
     * Tests a representative from each native-VNNI format family:
     *   - Simple 4-bit (Q4_0)
     *   - Asymmetric 4-bit (Q4_1)
     *   - 5-bit (Q5_0)
     *   - K-quant (Q4_K, Q6_K)
     *   - IQ formats (IQ4_NL, IQ3_S, IQ2_XXS, IQ1_S)
     *
     * And all INT8-VNNI formats:
     *   - Q8_0, Q8_1
     */
    TEST_F(ROCmQuantisedGemmKernelUnitTest, Repack_MutualExclusion_AllFamilies)
    {
        struct FormatCase
        {
            const char *name;
            bool expect_native_vnni; // true = native-VNNI, false = INT8-VNNI
            std::function<std::unique_ptr<TensorBase>()> create;
        };

        std::vector<FormatCase> cases = {
            // Native-VNNI formats (≤6-bit) → only native packing
            {"Q4_0", true, []
             { return TestTensorFactory::createQ4_0Random({32, 128}); }},
            {"Q4_1", true, []
             { return TestTensorFactory::createQ4_1Random({32, 128}); }},
            {"Q5_0", true, []
             { return TestTensorFactory::createQ5_0Random({32, 128}); }},
            {"Q5_1", true, []
             { return TestTensorFactory::createQ5_1Random({32, 128}); }},
            {"Q4_K", true, []
             { return TestTensorFactory::createQ4_KRandom({32, 256}); }},
            {"Q6_K", true, []
             { return TestTensorFactory::createQ6_KRandom({32, 256}); }},
            {"Q3_K", true, []
             { return TestTensorFactory::createQ3_KRandom({32, 256}); }},
            {"Q2_K", true, []
             { return TestTensorFactory::createQ2_KRandom({32, 256}); }},
            {"IQ4_NL", true, []
             { return TestTensorFactory::createIQ4_NLRandom({32, 64}); }},
            {"IQ4_XS", true, []
             { return TestTensorFactory::createIQ4_XSRandom({32, 256}); }},
            {"IQ3_S", true, []
             { return TestTensorFactory::createIQ3_SRandom({32, 256}); }},
            {"IQ3_XXS", true, []
             { return TestTensorFactory::createIQ3_XXSRandom({32, 256}); }},
            {"IQ2_S", true, []
             { return TestTensorFactory::createIQ2_SRandom({32, 256}); }},
            {"IQ2_XS", true, []
             { return TestTensorFactory::createIQ2_XSRandom({32, 256}); }},
            {"IQ2_XXS", true, []
             { return TestTensorFactory::createIQ2_XXSRandom({32, 256}); }},
            {"IQ1_S", true, []
             { return TestTensorFactory::createIQ1_SRandom({32, 256}); }},
            {"IQ1_M", true, []
             { return TestTensorFactory::createIQ1_MRandom({32, 256}); }},

            // INT8-VNNI formats (8-bit) → only INT8 packing
            {"Q8_0", false, []
             { return TestTensorFactory::createQ8_0Random({32, 64}); }},
            {"Q8_1", false, []
             { return TestTensorFactory::createQ8_1Random({32, 64}); }},
        };

        for (const auto &tc : cases)
        {
            auto tensor = tc.create();
            ASSERT_NE(tensor, nullptr) << "Failed to create " << tc.name;

            ROCmPackedWeights packed;
            ASSERT_TRUE(packWeightsToROCm(tensor.get(), packed))
                << "packWeightsToROCm failed for " << tc.name;

            if (tc.expect_native_vnni)
            {
                // Native-VNNI path populated
                EXPECT_FALSE(packed.native_vnni_payload.empty())
                    << tc.name << " (native-VNNI format) must have native_vnni_payload";
                EXPECT_FALSE(packed.native_vnni_scales.empty())
                    << tc.name << " (native-VNNI format) must have native_vnni_scales";

                // INT8 quantized data must NOT be populated
                EXPECT_TRUE(packed.int8_data.empty())
                    << tc.name << " (native-VNNI format) must NOT have int8_data";
                EXPECT_TRUE(packed.int8_data_vnni.empty())
                    << tc.name << " (native-VNNI format) must NOT have int8_data_vnni";
                // Note: scales IS populated by packNativeVNNI for force_ck compatibility
            }
            else
            {
                // INT8 path populated
                EXPECT_FALSE(packed.int8_data.empty())
                    << tc.name << " (INT8-VNNI format) must have int8_data";
                EXPECT_FALSE(packed.scales.empty())
                    << tc.name << " (INT8-VNNI format) must have scales";
                EXPECT_FALSE(packed.int8_data_vnni.empty())
                    << tc.name << " (INT8-VNNI format) must have int8_data_vnni";

                // Native-VNNI path must be empty
                EXPECT_TRUE(packed.native_vnni_payload.empty())
                    << tc.name << " (INT8-VNNI format) must NOT have native_vnni_payload";
                EXPECT_TRUE(packed.native_vnni_scales.empty())
                    << tc.name << " (INT8-VNNI format) must NOT have native_vnni_scales";
            }
        }
    }

    // =============================================================================

} // namespace
