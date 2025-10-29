/**
 * @file Test__IQ3_XXSTensor.cpp
 * @brief SIMD equivalency tests for IQ3_XXS quantized tensor
 *
 * Tests that scalar, AVX2, and AVX512 decode implementations produce identical results.
 *
 * IQ3_XXS Structure:
 * - Block size: 256 elements (8 sub-blocks of 32 elements each)
 * - Quantization: 3-bit using iq3xxs_grid[512] lookup table (8-bit indices)
 * - Scaling: Single FP16 scale + per-sub-block scale from aux32 (4-bit)
 * - Formula: output = db * grid[j] * sign
 *   where db = d * (0.5 + (aux32 >> 28)) * 0.5
 * - Block structure: 98 bytes = d(2) + qs[96]
 * - qs layout: First 64 bytes are grid indices, next 32 bytes are scales/signs
 * - Grid lookup: Pairs of 8-bit grid indices (qs[2*l+0], qs[2*l+1])
 * - Sign handling: 7-bit sign masks from ksigns_iq2xs table
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "../../../../src/v2/tensors/Tensors.h"
#include "../../../../src/v2/tensors/FP16Utils.h"
#include "../../../../src/v2/tensors/IQQuantTables.h"
#include <cmath>
#include <cstring>
#include <random>
#include <random>

using namespace llaminar2;

class IQ3_XXSSIMDTest : public ::testing::Test
{
protected:
    std::mt19937 rng_;
    static constexpr float TOLERANCE = 1e-6f;

    // Helper: Create IQ3_XXS block with specific pattern
    IQ3_XXSBlock createBlock(float scale, const std::vector<uint8_t> &qs_values)
    {
        IQ3_XXSBlock block;
        block.d = fp32_to_fp16(scale);

        // Copy qs values (96 bytes total)
        // First 64 bytes: grid indices
        // Last 32 bytes: scales and signs
        if (qs_values.size() != 96)
        {
            throw std::invalid_argument("IQ3_XXS requires exactly 96 uint8_t qs values");
        }
        std::memcpy(block.qs, qs_values.data(), 96);

        return block;
    }

    // Helper: Compare two float arrays
    bool compareArrays(const float *a, const float *b, size_t count, float tolerance = TOLERANCE)
    {
        for (size_t i = 0; i < count; ++i)
        {
            if (std::fabs(a[i] - b[i]) > tolerance)
            {
                std::cerr << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i]
                          << " (diff: " << std::fabs(a[i] - b[i]) << ")" << std::endl;
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Create a random IQ3_XXS tensor for GEMM testing
     */
    std::unique_ptr<IQ3_XXSTensor> createRandomTensor(size_t rows, size_t cols)
    {
        std::vector<size_t> shape = {rows, cols};
        size_t blocks_per_row = (cols + 255) / 256; // 256 elements per block
        size_t total_blocks = rows * blocks_per_row;
        std::vector<uint8_t> raw_data(total_blocks * sizeof(IQ3_XXSBlock));

        std::uniform_real_distribution<float> scale_dist(0.1f, 2.0f);
        std::uniform_int_distribution<uint8_t> byte_dist(0, 255);

        for (size_t b = 0; b < total_blocks; ++b)
        {
            IQ3_XXSBlock *block = reinterpret_cast<IQ3_XXSBlock *>(raw_data.data() + b * sizeof(IQ3_XXSBlock));
            block->d = fp32_to_fp16(scale_dist(rng_));
            // Randomize all other bytes in the block
            uint8_t *block_bytes = reinterpret_cast<uint8_t *>(block);
            for (size_t i = 2; i < sizeof(IQ3_XXSBlock); ++i)
            {
                block_bytes[i] = byte_dist(rng_);
            }
        }

        return std::make_unique<IQ3_XXSTensor>(shape, raw_data);
    }

    /**
     * @brief Compute reference GEMM: C = A @ B^T
     */
    void referenceGEMM(const float *A, const float *B, float *C,
                       int m, int n, int k)
    {
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                float sum = 0.0f;
                for (int l = 0; l < k; ++l)
                {
                    sum += A[i * k + l] * B[j * k + l]; // B is [n, k]
                }
                C[i * n + j] = sum;
            }
        }
    }

    /**
     * @brief Check if two matrices are approximately equal
     */
    bool matricesEqual(const float *A, const float *B, int size, float tolerance = 1e-4f)
    {
        for (int i = 0; i < size; ++i)
        {
            float diff = std::abs(A[i] - B[i]);
            float rel_error = diff / (std::abs(A[i]) + 1e-8f);
            if (diff > tolerance && rel_error > tolerance)
            {
                return false;
            }
        }
        return true;
    }
};

// ============================================================================
// SIMD Equivalency Tests
// ============================================================================

TEST_F(IQ3_XXSSIMDTest, ScalarVsAVX2Equivalency)
{
    std::vector<uint8_t> qs_values(96);

    // First 64 bytes: grid indices (varying pattern)
    for (size_t i = 0; i < 64; ++i)
    {
        qs_values[i] = (i * 7) % 256; // 8-bit grid indices
    }

    // Last 32 bytes: scales and signs (4 bytes per sub-block, 8 sub-blocks)
    for (size_t i = 64; i < 96; ++i)
    {
        qs_values[i] = (i * 13) % 256;
    }

    auto block = createBlock(1.5f, qs_values);

    float output_scalar[256];
    float output_avx2[256];

    IQ3_XXSTensor::decodeBlockScalar(block, output_scalar);

#ifdef __AVX2__
    IQ3_XXSTensor::decodeBlockAVX2(block, output_avx2);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, 256))
        << "Scalar and AVX2 implementations should produce identical results";
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
#endif
}

TEST_F(IQ3_XXSSIMDTest, ScalarVsAVX512Equivalency)
{
    std::vector<uint8_t> qs_values(96);

    // Different pattern from AVX2 test
    for (size_t i = 0; i < 64; ++i)
    {
        qs_values[i] = (i * 11) % 256;
    }

    for (size_t i = 64; i < 96; ++i)
    {
        qs_values[i] = (i * 17) % 256;
    }

    auto block = createBlock(2.0f, qs_values);

    float output_scalar[256];
    float output_avx512[256];

    IQ3_XXSTensor::decodeBlockScalar(block, output_scalar);

#ifdef __AVX512F__
    IQ3_XXSTensor::decodeBlockAVX512(block, output_avx512);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, 256))
        << "Scalar and AVX512 implementations should produce identical results";
#else
    GTEST_SKIP() << "AVX512 not available on this platform";
#endif
}

TEST_F(IQ3_XXSSIMDTest, AVX2VsAVX512Equivalency)
{
#if defined(__AVX2__) && defined(__AVX512F__)
    std::vector<uint8_t> qs_values(96);

    for (size_t i = 0; i < 64; ++i)
    {
        qs_values[i] = (i * 23) % 256;
    }

    for (size_t i = 64; i < 96; ++i)
    {
        qs_values[i] = (i * 29) % 256;
    }

    auto block = createBlock(0.75f, qs_values);

    float output_avx2[256];
    float output_avx512[256];

    IQ3_XXSTensor::decodeBlockAVX2(block, output_avx2);
    IQ3_XXSTensor::decodeBlockAVX512(block, output_avx512);

    EXPECT_TRUE(compareArrays(output_avx2, output_avx512, 256))
        << "AVX2 and AVX512 implementations should produce identical results";
#else
    GTEST_SKIP() << "Both AVX2 and AVX512 required for this test";
#endif
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(IQ3_XXSSIMDTest, EdgeCase_ZeroScale)
{
    std::vector<uint8_t> qs_values(96, 0);
    auto block = createBlock(0.0f, qs_values);

    float output[256];
    IQ3_XXSTensor::decodeBlockScalar(block, output);

    // With zero scale, all outputs should be zero
    for (size_t i = 0; i < 256; ++i)
    {
        EXPECT_FLOAT_EQ(output[i], 0.0f) << "Zero scale should produce zero output at index " << i;
    }
}

TEST_F(IQ3_XXSSIMDTest, EdgeCase_AllZeroIndices)
{
    std::vector<uint8_t> qs_values(96, 0);

    // Set non-zero scales/signs in last 32 bytes
    for (size_t i = 64; i < 96; ++i)
    {
        qs_values[i] = 0x77;
    }

    auto block = createBlock(1.0f, qs_values);

    float output_scalar[256];
    IQ3_XXSTensor::decodeBlockScalar(block, output_scalar);

#ifdef __AVX2__
    float output_avx2[256];
    IQ3_XXSTensor::decodeBlockAVX2(block, output_avx2);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, 256));
#endif

#ifdef __AVX512F__
    float output_avx512[256];
    IQ3_XXSTensor::decodeBlockAVX512(block, output_avx512);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, 256));
#endif
}

TEST_F(IQ3_XXSSIMDTest, EdgeCase_AllMaxIndices)
{
    std::vector<uint8_t> qs_values(96, 0xFF);

    auto block = createBlock(1.0f, qs_values);

    float output_scalar[256];
    IQ3_XXSTensor::decodeBlockScalar(block, output_scalar);

#ifdef __AVX2__
    float output_avx2[256];
    IQ3_XXSTensor::decodeBlockAVX2(block, output_avx2);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, 256));
#endif

#ifdef __AVX512F__
    float output_avx512[256];
    IQ3_XXSTensor::decodeBlockAVX512(block, output_avx512);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, 256));
#endif
}

TEST_F(IQ3_XXSSIMDTest, EdgeCase_MaxScale)
{
    std::vector<uint8_t> qs_values(96);
    for (size_t i = 0; i < 96; ++i)
    {
        qs_values[i] = (i * 19) % 256;
    }

    // Use a large but valid FP16 value
    auto block = createBlock(65504.0f, qs_values);

    float output_scalar[256];
    IQ3_XXSTensor::decodeBlockScalar(block, output_scalar);

#ifdef __AVX2__
    float output_avx2[256];
    IQ3_XXSTensor::decodeBlockAVX2(block, output_avx2);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, 256));
#endif

#ifdef __AVX512F__
    float output_avx512[256];
    IQ3_XXSTensor::decodeBlockAVX512(block, output_avx512);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, 256));
#endif
}

// =============================================================================
// GEMM Tests
// =============================================================================

TEST_F(IQ3_XXSSIMDTest, GEMM_SmallBatch)
{
    auto tensor = createRandomTensor(8, 256); // 8 output features, 256 input features (1 block per row)

    auto gemm = tensor->createGemm();
    ASSERT_NE(gemm, nullptr);

    // A: [4, 256] - 4 input sequences
    std::vector<float> A(4 * 256);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &v : A)
        v = dist(rng_);

    // C: [4, 8] - result
    std::vector<float> C(4 * 8, 0.0f);

    bool success = gemm->multiply(A.data(), C.data(), 4, 8, 256, true);
    ASSERT_TRUE(success);

    // Reference computation
    std::vector<float> B_decoded(8 * 256);
    tensor->to_fp32(B_decoded.data());

    std::vector<float> C_expected(4 * 8, 0.0f);
    referenceGEMM(A.data(), B_decoded.data(), C_expected.data(), 4, 8, 256);

    EXPECT_TRUE(matricesEqual(C_expected.data(), C.data(), 4 * 8, 1e-3f));
}

TEST_F(IQ3_XXSSIMDTest, GEMM_MediumBatch)
{
    auto tensor = createRandomTensor(16, 512); // 16 output features, 512 input features (2 blocks per row)

    auto gemm = tensor->createGemm();

    // A: [16, 512]
    std::vector<float> A(16 * 512);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto &v : A)
        v = dist(rng_);

    // C: [16, 16]
    std::vector<float> C(16 * 16, 0.0f);

    bool success = gemm->multiply(A.data(), C.data(), 16, 16, 512, true);
    ASSERT_TRUE(success);

    // Reference
    std::vector<float> B_decoded(16 * 512);
    tensor->to_fp32(B_decoded.data());

    std::vector<float> C_expected(16 * 16, 0.0f);
    referenceGEMM(A.data(), B_decoded.data(), C_expected.data(), 16, 16, 512);

    EXPECT_TRUE(matricesEqual(C_expected.data(), C.data(), 16 * 16, 1e-3f));
}

TEST_F(IQ3_XXSSIMDTest, GEMM_LargeBatch)
{
    auto tensor = createRandomTensor(32, 768); // 32 output features, 768 input features (3 blocks per row)

    auto gemm = tensor->createGemm();

    // A: [32, 768]
    std::vector<float> A(32 * 768);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    for (auto &v : A)
        v = dist(rng_);

    // C: [32, 32]
    std::vector<float> C(32 * 32, 0.0f);

    bool success = gemm->multiply(A.data(), C.data(), 32, 32, 768, true);
    ASSERT_TRUE(success);

    // Reference
    std::vector<float> B_decoded(32 * 768);
    tensor->to_fp32(B_decoded.data());

    std::vector<float> C_expected(32 * 32, 0.0f);
    referenceGEMM(A.data(), B_decoded.data(), C_expected.data(), 32, 32, 768);

    EXPECT_TRUE(matricesEqual(C_expected.data(), C.data(), 32 * 32, 1e-3f));
}

TEST_F(IQ3_XXSSIMDTest, EdgeCase_RandomValues)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> scale_dist(0.1f, 10.0f);
    std::uniform_int_distribution<uint8_t> qs_dist(0, 0xFF);

    std::vector<uint8_t> qs_values(96);
    for (size_t i = 0; i < 96; ++i)
    {
        qs_values[i] = qs_dist(rng);
    }

    auto block = createBlock(scale_dist(rng), qs_values);

    float output_scalar[256];
    IQ3_XXSTensor::decodeBlockScalar(block, output_scalar);

#ifdef __AVX2__
    float output_avx2[256];
    IQ3_XXSTensor::decodeBlockAVX2(block, output_avx2);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, 256));
#endif

#ifdef __AVX512F__
    float output_avx512[256];
    IQ3_XXSTensor::decodeBlockAVX512(block, output_avx512);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, 256));
#endif
}
