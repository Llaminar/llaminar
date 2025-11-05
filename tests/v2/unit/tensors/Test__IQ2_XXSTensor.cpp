/**
 * @file Test__IQ2_XXSTensor.cpp
 * @brief SIMD equivalency tests for IQ2_XXS quantized tensor
 *
 * Tests that scalar, AVX2, and AVX512 decode implementations produce identical results.
 *
 * IQ2_XXS Structure:
 * - Block size: 256 elements (8 sub-blocks of 32 elements each)
 * - Quantization: 2-bit using iq2xxs_grid[256] lookup table
 * - Scaling: Single FP16 scale (d) + per-sub-block scale from aux32[1]
 * - Formula: output = db * grid[j] * sign
 *   where db = d * (0.5 + (aux32[1] >> 28)) * 0.25
 * - Block structure: 66 bytes = d(2) + qs[64]
 * - Grid lookup: 8-bit grid indices map to 64-bit grid values
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

class IQ2_XXSSIMDTest : public ::testing::Test
{
protected:
    std::mt19937 rng_;
    static constexpr float TOLERANCE = 1e-6f;

    // Helper: Create IQ2_XXS block with specific pattern
    IQ2_XXSBlock createBlock(float scale, const std::vector<uint16_t> &qs_values)
    {
        IQ2_XXSBlock block;
        block.d = fp32_to_fp16(scale);

        // Copy qs values (32 uint16_t for 256 elements)
        if (qs_values.size() != 32)
        {
            throw std::invalid_argument("IQ2_XXS requires exactly 32 uint16_t qs values");
        }
        std::memcpy(block.qs, qs_values.data(), 32 * sizeof(uint16_t));

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
     * @brief Create a random IQ2_XXS tensor for GEMM testing
     */
    std::unique_ptr<IQ2_XXSTensor> createRandomTensor(size_t rows, size_t cols)
    {
        std::vector<size_t> shape = {rows, cols};
        size_t blocks_per_row = (cols + 255) / 256; // 256 elements per block
        size_t total_blocks = rows * blocks_per_row;
        std::vector<uint8_t> raw_data(total_blocks * sizeof(IQ2_XXSBlock));

        std::uniform_real_distribution<float> scale_dist(0.1f, 2.0f);
        std::uniform_int_distribution<uint8_t> byte_dist(0, 255);

        for (size_t b = 0; b < total_blocks; ++b)
        {
            IQ2_XXSBlock *block = reinterpret_cast<IQ2_XXSBlock *>(raw_data.data() + b * sizeof(IQ2_XXSBlock));
            block->d = fp32_to_fp16(scale_dist(rng_));
            // Randomize all other bytes in the block
            uint8_t *block_bytes = reinterpret_cast<uint8_t *>(block);
            for (size_t i = 2; i < sizeof(IQ2_XXSBlock); ++i)
            {
                block_bytes[i] = byte_dist(rng_);
            }
        }

        return std::make_unique<IQ2_XXSTensor>(shape, raw_data);
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

TEST_F(IQ2_XXSSIMDTest, ScalarVsAVX2Equivalency)
{
    // Create block with varying qs values to exercise different grid lookups
    std::vector<uint16_t> qs_values(32);
    for (size_t i = 0; i < 32; ++i)
    {
        // Pack 4 grid indices (8-bit each) into each uint16_t
        // Pattern creates variety in grid lookups and signs
        qs_values[i] = ((i * 7) % 256) | (((i * 13) % 256) << 8);
    }

    auto block = createBlock(1.5f, qs_values);

    float output_scalar[256];
    float output_avx2[256];

    IQ2_XXSTensor::decodeBlockScalar(block, output_scalar);

#ifdef __AVX2__
    IQ2_XXSTensor::decodeBlockAVX2(block, output_avx2);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, 256))
        << "Scalar and AVX2 implementations should produce identical results";
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
#endif
}

TEST_F(IQ2_XXSSIMDTest, ScalarVsAVX512Equivalency)
{
    // Different pattern from AVX2 test
    std::vector<uint16_t> qs_values(32);
    for (size_t i = 0; i < 32; ++i)
    {
        // Different pattern to test various grid/sign combinations
        qs_values[i] = ((i * 11) % 256) | (((i * 17) % 256) << 8);
    }

    auto block = createBlock(2.0f, qs_values);

    float output_scalar[256];
    float output_avx512[256];

    IQ2_XXSTensor::decodeBlockScalar(block, output_scalar);

#ifdef __AVX512F__
    IQ2_XXSTensor::decodeBlockAVX512(block, output_avx512);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, 256))
        << "Scalar and AVX512 implementations should produce identical results";
#else
    GTEST_SKIP() << "AVX512 not available on this platform";
#endif
}

TEST_F(IQ2_XXSSIMDTest, AVX2VsAVX512Equivalency)
{
#if defined(__AVX2__) && defined(__AVX512F__)
    std::vector<uint16_t> qs_values(32);
    for (size_t i = 0; i < 32; ++i)
    {
        // Yet another pattern for cross-validation
        qs_values[i] = ((i * 23) % 256) | (((i * 29) % 256) << 8);
    }

    auto block = createBlock(0.75f, qs_values);

    float output_avx2[256];
    float output_avx512[256];

    IQ2_XXSTensor::decodeBlockAVX2(block, output_avx2);
    IQ2_XXSTensor::decodeBlockAVX512(block, output_avx512);

    EXPECT_TRUE(compareArrays(output_avx2, output_avx512, 256))
        << "AVX2 and AVX512 implementations should produce identical results";
#else
    GTEST_SKIP() << "Both AVX2 and AVX512 required for this test";
#endif
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(IQ2_XXSSIMDTest, EdgeCase_ZeroScale)
{
    // All grid indices = 0 (should give zero output regardless)
    std::vector<uint16_t> qs_values(32, 0);
    auto block = createBlock(0.0f, qs_values);

    float output[256];
    IQ2_XXSTensor::decodeBlockScalar(block, output);

    // With zero scale, all outputs should be zero
    for (size_t i = 0; i < 256; ++i)
    {
        EXPECT_FLOAT_EQ(output[i], 0.0f) << "Zero scale should produce zero output at index " << i;
    }
}

TEST_F(IQ2_XXSSIMDTest, EdgeCase_AllZeroIndices)
{
    // All grid indices = 0 (maps to iq2xxs_grid[0])
    std::vector<uint16_t> qs_values(32, 0);
    auto block = createBlock(1.0f, qs_values);

    float output_scalar[256];
    IQ2_XXSTensor::decodeBlockScalar(block, output_scalar);

#ifdef __AVX2__
    float output_avx2[256];
    IQ2_XXSTensor::decodeBlockAVX2(block, output_avx2);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, 256));
#endif

#ifdef __AVX512F__
    float output_avx512[256];
    IQ2_XXSTensor::decodeBlockAVX512(block, output_avx512);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, 256));
#endif
}

TEST_F(IQ2_XXSSIMDTest, EdgeCase_AllMaxIndices)
{
    // All grid indices = 255 (maps to iq2xxs_grid[255])
    std::vector<uint16_t> qs_values(32);
    for (size_t i = 0; i < 32; ++i)
    {
        qs_values[i] = 0xFFFF; // Both bytes = 255
    }

    auto block = createBlock(1.0f, qs_values);

    float output_scalar[256];
    IQ2_XXSTensor::decodeBlockScalar(block, output_scalar);

#ifdef __AVX2__
    float output_avx2[256];
    IQ2_XXSTensor::decodeBlockAVX2(block, output_avx2);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, 256));
#endif

#ifdef __AVX512F__
    float output_avx512[256];
    IQ2_XXSTensor::decodeBlockAVX512(block, output_avx512);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, 256));
#endif
}

TEST_F(IQ2_XXSSIMDTest, EdgeCase_MaxScale)
{
    // Maximum FP16 scale value
    std::vector<uint16_t> qs_values(32);
    for (size_t i = 0; i < 32; ++i)
    {
        qs_values[i] = ((i * 3) % 256) | (((i * 5) % 256) << 8);
    }

    // Use a large but valid FP16 value
    auto block = createBlock(65504.0f, qs_values); // Max normal FP16

    float output_scalar[256];
    IQ2_XXSTensor::decodeBlockScalar(block, output_scalar);

#ifdef __AVX2__
    float output_avx2[256];
    IQ2_XXSTensor::decodeBlockAVX2(block, output_avx2);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, 256));
#endif

#ifdef __AVX512F__
    float output_avx512[256];
    IQ2_XXSTensor::decodeBlockAVX512(block, output_avx512);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, 256));
#endif
}

// =============================================================================
// GEMM Tests
// =============================================================================

TEST_F(IQ2_XXSSIMDTest, GEMM_SmallBatch)
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

TEST_F(IQ2_XXSSIMDTest, GEMM_MediumBatch)
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

TEST_F(IQ2_XXSSIMDTest, GEMM_LargeBatch)
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

TEST_F(IQ2_XXSSIMDTest, EdgeCase_RandomValues)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> scale_dist(0.1f, 10.0f);
    std::uniform_int_distribution<uint16_t> qs_dist(0, 0xFFFF);

    std::vector<uint16_t> qs_values(32);
    for (size_t i = 0; i < 32; ++i)
    {
        qs_values[i] = qs_dist(rng);
    }

    auto block = createBlock(scale_dist(rng), qs_values);

    float output_scalar[256];
    IQ2_XXSTensor::decodeBlockScalar(block, output_scalar);

#ifdef __AVX2__
    float output_avx2[256];
    IQ2_XXSTensor::decodeBlockAVX2(block, output_avx2);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, 256));
#endif

#ifdef __AVX512F__
    float output_avx512[256];
    IQ2_XXSTensor::decodeBlockAVX512(block, output_avx512);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, 256));
#endif
}


// =============================================================================
// to<T>() Template Method Tests
// =============================================================================

TEST_F(IQ2_XXSSIMDTest, ToFloat_TemplateMethod)
{
    // Create a random tensor (1x256 = 1 block)
    auto tensor = createRandomTensor(1, 256);
    const size_t total = 256;

    // Test template method
    std::vector<float> template_output(total);
    tensor->to<float>(template_output.data());

    // Test legacy method
    std::vector<float> legacy_output(total);
    tensor->to_fp32(legacy_output.data());

    // Should produce identical results
    for (size_t i = 0; i < total; ++i) {
        EXPECT_FLOAT_EQ(template_output[i], legacy_output[i]) 
            << "Mismatch at index " << i;
    }
}

TEST_F(IQ2_XXSSIMDTest, ToBF16_TemplateMethod)
{
    // Create random tensor
    auto tensor = createRandomTensor(1, 256);
    const size_t total = 256;

    // Test template method
    std::vector<uint16_t> template_output(total);
    tensor->to<uint16_t>(template_output.data(), TensorType::BF16);

    // Test legacy method
    std::vector<uint16_t> legacy_output(total);
    tensor->to_bf16(legacy_output.data());

    // Should produce identical results
    for (size_t i = 0; i < total; ++i) {
        EXPECT_EQ(template_output[i], legacy_output[i]) 
            << "Mismatch at index " << i;
    }
}

TEST_F(IQ2_XXSSIMDTest, ToFP16_TemplateMethod)
{
    // Create random tensor
    auto tensor = createRandomTensor(1, 256);
    const size_t total = 256;

    // Test template method
    std::vector<uint16_t> template_output(total);
    tensor->to<uint16_t>(template_output.data(), TensorType::FP16);

    // Test legacy method
    std::vector<uint16_t> legacy_output(total);
    tensor->to_fp16(legacy_output.data());

    // Should produce identical results
    for (size_t i = 0; i < total; ++i) {
        EXPECT_EQ(template_output[i], legacy_output[i]) 
            << "Mismatch at index " << i;
    }
}

TEST_F(IQ2_XXSSIMDTest, ToINT8_TemplateMethod)
{
    // Create tensor
    auto tensor = createRandomTensor(1, 256);
    const size_t total = 256;

    // Convert to INT8
    std::vector<int8_t> int8_output(total);
    tensor->to<int8_t>(int8_output.data());

    // Verify INT8 range
    for (size_t i = 0; i < total; ++i) {
        EXPECT_GE(int8_output[i], -127);
        EXPECT_LE(int8_output[i], 127);
    }
}

TEST_F(IQ2_XXSSIMDTest, ToINT32_TemplateMethod)
{
    // Create tensor
    auto tensor = createRandomTensor(1, 256);
    const size_t total = 256;

    // Convert to INT32
    std::vector<int32_t> int32_output(total);
    tensor->to<int32_t>(int32_output.data());

    // Just verify no crash
    EXPECT_NE(int32_output.data(), nullptr);
}

TEST_F(IQ2_XXSSIMDTest, RoundTrip)
{
    // Create tensor
    auto tensor = createRandomTensor(1, 256);
    const size_t total = 256;

    // Round trip: IQ -> FP32 -> BF16 -> FP32
    std::vector<float> fp32_1(total);
    tensor->to<float>(fp32_1.data());

    // Create BF16 tensor from FP32 data
    auto fp32_temp = std::make_shared<FP32Tensor>(std::vector<size_t>{1, 256});
    std::memcpy(fp32_temp->mutable_data(), fp32_1.data(), total * sizeof(float));
    
    std::vector<uint16_t> bf16_data(total);
    fp32_temp->to<uint16_t>(bf16_data.data(), TensorType::BF16);
    
    auto bf16_tensor = std::make_shared<BF16Tensor>(std::vector<size_t>{1, 256}, bf16_data);

    // Convert back to FP32
    std::vector<float> fp32_2(total);
    bf16_tensor->to<float>(fp32_2.data());

    // Verify accuracy (BF16 precision ~3 decimal places)
    size_t mismatches = 0;
    for (size_t i = 0; i < total; ++i) {
        float diff = std::abs(fp32_1[i] - fp32_2[i]);
        float rel_error = (fp32_1[i] != 0.0f) ? diff / std::abs(fp32_1[i]) : diff;
        if (rel_error > 0.05f) { // 5% tolerance for BF16
            ++mismatches;
        }
    }
    EXPECT_LT(mismatches, 13) << "Too many mismatches in round-trip conversion";
}
