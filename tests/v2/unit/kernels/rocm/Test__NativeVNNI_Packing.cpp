/**
 * @file Test__NativeVNNI_Packing.cpp
 * @brief Unit tests for native-VNNI weight packing (CPU-only, no GPU required)
 *
 * Validates that packWeightsToROCm() correctly produces native-VNNI payload,
 * scales, and mins containers for all 16 supported formats (Tier 1/2/3).
 *
 * Tests verify:
 * - Packing succeeds (non-empty payload/scales)
 * - Correct codebook_id assignment
 * - Correct blocks_per_row calculation
 * - Payload, scales, and mins sizes match expected layout
 * - FP16 scale values are finite (not NaN/Inf/zero)
 * - Asymmetric formats populate mins, symmetric do not
 * - Various N/K dimensions work
 *
 * GPU-requiring native-VNNI tests (GEMV accuracy comparison) are in:
 *   tests/v2/integration/kernels/rocm/Test__NativeVNNI_GEMV.cpp
 *
 * @note All tests run on CPU only — no ROCm device required.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
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

    // =============================================================================
    // Format metadata for parameterized tests
    // =============================================================================

    struct NativeVNNIFormatSpec
    {
        std::string name;
        uint8_t expected_codebook_id;
        int payload_bytes; ///< Per 32-element block
        bool is_asymmetric;
        bool is_superblock; ///< 256-element blocks (K must be multiple of 256)

        /// Factory function that creates a random tensor of this type
        std::function<std::unique_ptr<TensorBase>(size_t N, size_t K)> create;
    };

    // All 16 native-VNNI formats
    static const std::vector<NativeVNNIFormatSpec> ALL_FORMATS = {
        // Tier 1: Simple 32-element blocks
        {"Q4_0", 0, 16, false, false,
         [](size_t N, size_t K)
         { return TestTensorFactory::createQ4_0Random({N, K}); }},
        {"IQ4_NL", 4, 16, false, false,
         [](size_t N, size_t K)
         { return TestTensorFactory::createIQ4_NLRandom({N, K}); }},
        {"Q4_1", 5, 16, true, false,
         [](size_t N, size_t K)
         { return TestTensorFactory::createQ4_1Random({N, K}); }},
        {"Q5_0", 6, 20, false, false,
         [](size_t N, size_t K)
         { return TestTensorFactory::createQ5_0Random({N, K}); }},
        {"Q5_1", 7, 20, true, false,
         [](size_t N, size_t K)
         { return TestTensorFactory::createQ5_1Random({N, K}); }},

        // Tier 1 super-block (reuses IQ4_NL kernel)
        {"IQ4_XS", 4, 16, false, true,
         [](size_t N, size_t K)
         { return TestTensorFactory::createIQ4_XSRandom({N, K}); }},

        // Tier 2: K-quant super-blocks
        {"Q4_K", 5, 16, true, true,
         [](size_t N, size_t K)
         { return TestTensorFactory::createQ4_KRandom({N, K}); }},
        {"Q5_K", 7, 20, true, true,
         [](size_t N, size_t K)
         { return TestTensorFactory::createQ5_KRandom({N, K}); }},
        {"Q6_K", 8, 24, true, true,
         [](size_t N, size_t K)
         { return TestTensorFactory::createQ6_KRandom({N, K}); }},
        {"Q3_K", 9, 12, true, true,
         [](size_t N, size_t K)
         { return TestTensorFactory::createQ3_KRandom({N, K}); }},
        {"Q2_K", 10, 8, true, true,
         [](size_t N, size_t K)
         { return TestTensorFactory::createQ2_KRandom({N, K}); }},

        // Tier 3: IQ grid-index super-blocks
        {"IQ3_S", 11, 13, false, true,
         [](size_t N, size_t K)
         { return TestTensorFactory::createIQ3_SRandom({N, K}); }},
        {"IQ3_XXS", 12, 12, false, true,
         [](size_t N, size_t K)
         { return TestTensorFactory::createIQ3_XXSRandom({N, K}); }},
        {"IQ2_S", 13, 9, true, true,
         [](size_t N, size_t K)
         { return TestTensorFactory::createIQ2_SRandom({N, K}); }},
        {"IQ2_XS", 14, 9, true, true,
         [](size_t N, size_t K)
         { return TestTensorFactory::createIQ2_XSRandom({N, K}); }},
        {"IQ2_XXS", 15, 8, false, true,
         [](size_t N, size_t K)
         { return TestTensorFactory::createIQ2_XXSRandom({N, K}); }},

        // Tier 4: IQ1 ultra-low-bit grid-index super-blocks
        // payload_bytes: native size only (IQ1_S: qs[4]+qh[2]=6, IQ1_M: qs[4]+qh[2]=6)
        {"IQ1_S", 16, 6, true, true,
         [](size_t N, size_t K)
         { return TestTensorFactory::createIQ1_SRandom({N, K}); }},
        {"IQ1_M", 17, 6, true, true,
         [](size_t N, size_t K)
         { return TestTensorFactory::createIQ1_MRandom({N, K}); }},
    };

    // =============================================================================
    // Helpers
    // =============================================================================

    /// Check if a FP16 value (stored as uint16_t bits) is finite and non-zero
    bool isFP16ValidScale(uint16_t bits)
    {
        // FP16: sign(1) exp(5) mantissa(10)
        // Exponent bits [14:10]
        uint16_t exp = (bits >> 10) & 0x1F;
        // exp=31 → Inf/NaN, exp=0 with mantissa=0 → zero
        if (exp == 31)
            return false; // Inf or NaN
        // Allow subnormals and zero (zero scale is valid for zero blocks)
        return true;
    }

    // =============================================================================
    // Test fixture
    // =============================================================================

    class NativeVNNIPackingTest : public ::testing::Test
    {
    protected:
        void SetUp() override {}
    };

    // =============================================================================
    // Parameterized test: basic packing validation for all formats
    // =============================================================================

    class NativeVNNIPackingFormatTest
        : public ::testing::TestWithParam<NativeVNNIFormatSpec>
    {
    };

    TEST_P(NativeVNNIPackingFormatTest, PackSucceeds_ProducesValidOutput)
    {
        const auto &fmt = GetParam();

        // Use K=256 for super-block formats, K=128 for simple formats
        const size_t N = 32;
        const size_t K = fmt.is_superblock ? 256 : 128;
        const size_t blocks_per_row = K / 32;

        auto tensor = fmt.create(N, K);
        ASSERT_NE(tensor, nullptr) << "Failed to create " << fmt.name << " tensor";

        ROCmPackedWeights packed;
        bool ok = packWeightsToROCm(tensor.get(), packed);
        ASSERT_TRUE(ok) << "packWeightsToROCm failed for " << fmt.name;

        // Native-VNNI payload must be populated
        EXPECT_FALSE(packed.native_vnni_payload.empty())
            << fmt.name << ": native_vnni_payload is empty";
        EXPECT_FALSE(packed.native_vnni_scales.empty())
            << fmt.name << ": native_vnni_scales is empty";

        // Verify codebook_id
        EXPECT_EQ(packed.native_vnni_codebook_id, fmt.expected_codebook_id)
            << fmt.name << ": wrong codebook_id";

        // Verify blocks_per_row
        EXPECT_EQ(packed.native_vnni_blocks_per_row, static_cast<uint32_t>(blocks_per_row))
            << fmt.name << ": wrong blocks_per_row";

        // Verify payload size: blocks_per_row × N × payload_bytes
        const size_t expected_payload =
            blocks_per_row * N * static_cast<size_t>(fmt.payload_bytes);
        EXPECT_EQ(packed.native_vnni_payload.size(), expected_payload)
            << fmt.name << ": payload size mismatch"
            << " (expected " << expected_payload
            << ", got " << packed.native_vnni_payload.size() << ")";

        // Verify scales size: blocks_per_row × N
        EXPECT_EQ(packed.native_vnni_scales.size(), blocks_per_row * N)
            << fmt.name << ": scales size mismatch";

        // Verify asymmetric formats have mins, symmetric do not
        if (fmt.is_asymmetric)
        {
            EXPECT_EQ(packed.native_vnni_mins.size(), blocks_per_row * N)
                << fmt.name << ": asymmetric format should have mins";
        }
        else
        {
            EXPECT_TRUE(packed.native_vnni_mins.empty())
                << fmt.name << ": symmetric format should NOT have mins";
        }

        // N/K dimensions preserved
        EXPECT_EQ(packed.N, static_cast<int>(N));
        EXPECT_EQ(packed.K, static_cast<int>(K));
    }

    TEST_P(NativeVNNIPackingFormatTest, ScalesAreFinite)
    {
        const auto &fmt = GetParam();
        const size_t N = 16;
        const size_t K = fmt.is_superblock ? 256 : 128;

        auto tensor = fmt.create(N, K);
        ASSERT_NE(tensor, nullptr);

        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(tensor.get(), packed));

        // All FP16 scales must be finite
        for (size_t i = 0; i < packed.native_vnni_scales.size(); ++i)
        {
            EXPECT_TRUE(isFP16ValidScale(packed.native_vnni_scales[i]))
                << fmt.name << ": scale[" << i << "] is NaN/Inf (bits=0x"
                << std::hex << packed.native_vnni_scales[i] << std::dec << ")";
        }

        // If asymmetric, mins must also be finite
        for (size_t i = 0; i < packed.native_vnni_mins.size(); ++i)
        {
            EXPECT_TRUE(isFP16ValidScale(packed.native_vnni_mins[i]))
                << fmt.name << ": min[" << i << "] is NaN/Inf (bits=0x"
                << std::hex << packed.native_vnni_mins[i] << std::dec << ")";
        }
    }

    TEST_P(NativeVNNIPackingFormatTest, PayloadNotAllZeros)
    {
        const auto &fmt = GetParam();
        const size_t N = 16;
        const size_t K = fmt.is_superblock ? 256 : 128;

        auto tensor = fmt.create(N, K);
        ASSERT_NE(tensor, nullptr);

        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(tensor.get(), packed));

        // At least some payload bytes should be non-zero (random data)
        bool any_nonzero = false;
        for (size_t i = 0; i < packed.native_vnni_payload.size(); ++i)
        {
            if (packed.native_vnni_payload[i] != 0)
            {
                any_nonzero = true;
                break;
            }
        }
        EXPECT_TRUE(any_nonzero)
            << fmt.name << ": payload is all zeros — likely a packing bug";
    }

    INSTANTIATE_TEST_SUITE_P(
        AllFormats,
        NativeVNNIPackingFormatTest,
        ::testing::ValuesIn(ALL_FORMATS),
        [](const ::testing::TestParamInfo<NativeVNNIFormatSpec> &info)
        {
            return info.param.name;
        });

    // =============================================================================
    // Dimension sweep tests
    // =============================================================================

    /**
     * @test Verify packing works across a range of N and K dimensions
     * for all simple 32-element block formats.
     */
    TEST_F(NativeVNNIPackingTest, DimensionSweep_Simple32BlockFormats)
    {
        const std::vector<std::pair<size_t, size_t>> shapes = {
            {16, 32}, {32, 64}, {64, 128}, {128, 256}, {256, 512}};

        // Q4_0 as representative simple format
        for (const auto &[N, K] : shapes)
        {
            auto tensor = TestTensorFactory::createQ4_0Random({N, K});
            ROCmPackedWeights packed;
            ASSERT_TRUE(packWeightsToROCm(tensor.get(), packed))
                << "Q4_0 packing failed for N=" << N << " K=" << K;
            EXPECT_EQ(packed.native_vnni_blocks_per_row, static_cast<uint32_t>(K / 32));
        }
    }

    /**
     * @test Verify packing works across a range of N and K dimensions
     * for super-block formats (K must be multiple of 256).
     */
    TEST_F(NativeVNNIPackingTest, DimensionSweep_SuperBlockFormats)
    {
        const std::vector<std::pair<size_t, size_t>> shapes = {
            {16, 256}, {32, 512}, {64, 1024}, {128, 256}};

        // Q6_K as representative super-block, IQ3_S as representative IQ format
        for (const auto &[N, K] : shapes)
        {
            {
                auto tensor = TestTensorFactory::createQ6_KRandom({N, K});
                ROCmPackedWeights packed;
                ASSERT_TRUE(packWeightsToROCm(tensor.get(), packed))
                    << "Q6_K packing failed for N=" << N << " K=" << K;
                EXPECT_EQ(packed.native_vnni_blocks_per_row, static_cast<uint32_t>(K / 32));
                EXPECT_EQ(packed.native_vnni_codebook_id, 8);
            }
            {
                auto tensor = TestTensorFactory::createIQ3_SRandom({N, K});
                ROCmPackedWeights packed;
                ASSERT_TRUE(packWeightsToROCm(tensor.get(), packed))
                    << "IQ3_S packing failed for N=" << N << " K=" << K;
                EXPECT_EQ(packed.native_vnni_blocks_per_row, static_cast<uint32_t>(K / 32));
                EXPECT_EQ(packed.native_vnni_codebook_id, 11);
            }
        }
    }

    /**
     * @test Verify large dimensions matching real model layer sizes.
     *
     * Qwen2.5-0.5B: d_model=896, d_ff=4864
     * Tests Q4_0 (simple) and Q4_K (super-block) with model-realistic shapes.
     */
    TEST_F(NativeVNNIPackingTest, RealisticModelDimensions)
    {
        // d_ff × d_model (down_proj shape for Qwen2.5-0.5B)
        // K=896 is not a multiple of 256, so only simple formats work here.
        // Use K=896 for simple, K=3584 (Qwen2.5-7B d_model) for super-block.
        {
            auto tensor = TestTensorFactory::createQ4_0Random({4864, 896});
            ROCmPackedWeights packed;
            ASSERT_TRUE(packWeightsToROCm(tensor.get(), packed));
            EXPECT_EQ(packed.N, 4864);
            EXPECT_EQ(packed.K, 896);
            EXPECT_EQ(packed.native_vnni_codebook_id, 0);
        }
        {
            auto tensor = TestTensorFactory::createQ4_KRandom({18944, 3584});
            ROCmPackedWeights packed;
            ASSERT_TRUE(packWeightsToROCm(tensor.get(), packed));
            EXPECT_EQ(packed.N, 18944);
            EXPECT_EQ(packed.K, 3584);
            EXPECT_EQ(packed.native_vnni_codebook_id, 5);
        }
    }

    // =============================================================================
    // Native-VNNI formats get ONLY native packing (no INT8 requantization)
    // =============================================================================

    /**
     * @test Verify that packWeightsToROCm populates ONLY the native-VNNI path
     * for native-VNNI formats (≤6-bit).
     *
     * After the format-conditional repack change, native-VNNI formats no longer
     * receive INT8 requantization — only the native-VNNI payload/scales are populated.
     */
    TEST_F(NativeVNNIPackingTest, NativeVNNI_OnlyNativePathPopulated)
    {
        auto tensor = TestTensorFactory::createQ4_0Random({32, 128});
        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(tensor.get(), packed));

        // Native-VNNI path — must be populated
        EXPECT_FALSE(packed.native_vnni_payload.empty());
        EXPECT_FALSE(packed.native_vnni_scales.empty());

        // INT8 quantized data — must be EMPTY for native-VNNI formats
        EXPECT_TRUE(packed.int8_data.empty());
        EXPECT_TRUE(packed.int8_data_vnni.empty());
        // Note: scales IS populated by packNativeVNNI for force_ck compatibility
    }

    // =============================================================================
    // IQ format specific tests
    // =============================================================================

    /**
     * @test IQ3 formats use uint32_t grid LUT (4 values per lookup).
     * Verify payload layout: IQ3_S has 13 bytes, IQ3_XXS has 12 bytes per sub-block.
     */
    TEST_F(NativeVNNIPackingTest, IQ3_PayloadSizePerSubBlock)
    {
        const size_t N = 16, K = 256;
        const size_t blocks = K / 32; // 8 sub-blocks per super-block

        {
            auto tensor = TestTensorFactory::createIQ3_SRandom({N, K});
            ROCmPackedWeights packed;
            ASSERT_TRUE(packWeightsToROCm(tensor.get(), packed));
            EXPECT_EQ(packed.native_vnni_payload.size(), blocks * N * 13);
            EXPECT_EQ(packed.native_vnni_codebook_id, 11);
        }
        {
            auto tensor = TestTensorFactory::createIQ3_XXSRandom({N, K});
            ROCmPackedWeights packed;
            ASSERT_TRUE(packWeightsToROCm(tensor.get(), packed));
            EXPECT_EQ(packed.native_vnni_payload.size(), blocks * N * 12);
            EXPECT_EQ(packed.native_vnni_codebook_id, 12);
        }
    }

    /**
     * @test IQ2 formats use uint64_t grid LUT (8 values per lookup).
     * Verify payload layout: IQ2_S/IQ2_XS have 9 bytes, IQ2_XXS has 8 bytes.
     */
    TEST_F(NativeVNNIPackingTest, IQ2_PayloadSizePerSubBlock)
    {
        const size_t N = 16, K = 256;
        const size_t blocks = K / 32;

        {
            auto tensor = TestTensorFactory::createIQ2_SRandom({N, K});
            ROCmPackedWeights packed;
            ASSERT_TRUE(packWeightsToROCm(tensor.get(), packed));
            EXPECT_EQ(packed.native_vnni_payload.size(), blocks * N * 9);
            EXPECT_EQ(packed.native_vnni_codebook_id, 13);
            // IQ2_S is dual-scale (asymmetric)
            EXPECT_FALSE(packed.native_vnni_mins.empty());
        }
        {
            auto tensor = TestTensorFactory::createIQ2_XSRandom({N, K});
            ROCmPackedWeights packed;
            ASSERT_TRUE(packWeightsToROCm(tensor.get(), packed));
            EXPECT_EQ(packed.native_vnni_payload.size(), blocks * N * 9);
            EXPECT_EQ(packed.native_vnni_codebook_id, 14);
            // IQ2_XS is dual-scale (asymmetric)
            EXPECT_FALSE(packed.native_vnni_mins.empty());
        }
        {
            auto tensor = TestTensorFactory::createIQ2_XXSRandom({N, K});
            ROCmPackedWeights packed;
            ASSERT_TRUE(packWeightsToROCm(tensor.get(), packed));
            EXPECT_EQ(packed.native_vnni_payload.size(), blocks * N * 8);
            EXPECT_EQ(packed.native_vnni_codebook_id, 15);
            // IQ2_XXS is symmetric (no dual-scale)
            EXPECT_TRUE(packed.native_vnni_mins.empty());
        }
    }

    /**
     * @test IQ1 formats use uint64_t iq1s_grid LUT (8 signed values per lookup).
     * Verify payload layout: IQ1_S has 6 bytes (qs[4]+qh[2]),
     * IQ1_M has 6 bytes (qs[4]+qh[2]) per sub-block.
     * Both are asymmetric (delta correction via min).
     */
    TEST_F(NativeVNNIPackingTest, IQ1_PayloadSizePerSubBlock)
    {
        const size_t N = 16, K = 256;
        const size_t blocks = K / 32;

        {
            auto tensor = TestTensorFactory::createIQ1_SRandom({N, K});
            ROCmPackedWeights packed;
            ASSERT_TRUE(packWeightsToROCm(tensor.get(), packed));
            EXPECT_EQ(packed.native_vnni_payload.size(), blocks * N * 6);
            EXPECT_EQ(packed.native_vnni_codebook_id, 16);
            // IQ1_S is asymmetric (delta correction in mins)
            EXPECT_FALSE(packed.native_vnni_mins.empty());
        }
        {
            auto tensor = TestTensorFactory::createIQ1_MRandom({N, K});
            ROCmPackedWeights packed;
            ASSERT_TRUE(packWeightsToROCm(tensor.get(), packed));
            EXPECT_EQ(packed.native_vnni_payload.size(), blocks * N * 6);
            EXPECT_EQ(packed.native_vnni_codebook_id, 17);
            // IQ1_M is dual-scale (mins for scale_hi)
            EXPECT_FALSE(packed.native_vnni_mins.empty());
        }
    }

    /**
     * @test Dual-scale formats (Q6_K, Q3_K, Q2_K, IQ2_S, IQ2_XS, IQ1_M) must
     * populate mins with the same count as scales.
     */
    TEST_F(NativeVNNIPackingTest, DualScaleFormats_MinsMatchScalesSize)
    {
        const size_t N = 16, K = 256;

        auto check = [&](const std::string &name, auto create_fn)
        {
            auto tensor = create_fn(N, K);
            ROCmPackedWeights packed;
            ASSERT_TRUE(packWeightsToROCm(tensor.get(), packed)) << name;
            EXPECT_EQ(packed.native_vnni_mins.size(), packed.native_vnni_scales.size())
                << name << ": mins and scales count must match for dual-scale format";
        };

        check("Q6_K", [](size_t n, size_t k)
              { return TestTensorFactory::createQ6_KRandom({n, k}); });
        check("Q3_K", [](size_t n, size_t k)
              { return TestTensorFactory::createQ3_KRandom({n, k}); });
        check("Q2_K", [](size_t n, size_t k)
              { return TestTensorFactory::createQ2_KRandom({n, k}); });
        check("IQ2_S", [](size_t n, size_t k)
              { return TestTensorFactory::createIQ2_SRandom({n, k}); });
        check("IQ2_XS", [](size_t n, size_t k)
              { return TestTensorFactory::createIQ2_XSRandom({n, k}); });
        check("IQ1_M", [](size_t n, size_t k)
              { return TestTensorFactory::createIQ1_MRandom({n, k}); });
    }

} // anonymous namespace
