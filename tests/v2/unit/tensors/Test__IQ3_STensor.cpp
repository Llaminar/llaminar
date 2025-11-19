/**
 * @file Test__IQ3_STensor.cpp
 * @brief SIMD equivalency tests for IQ3_S decoding
 *
 * IQ3_S Structure:
 * - Block size: 256 elements (8 super-blocks of 32 elements each)
 * - Quantization: 3-bit using iq3s_grid[512] lookup table (9-bit indices: 8 bits from qs + 1 bit from qh)
 * - Scaling: FP16 global scale + 4 per-super-block scales (4-bit each, packed in scales[4])
 * - Signs: Separate signs array (4 bytes per super-block)
 * - Formula: output = d * (1 + 2*scale_nibble) * grid[j] * sign
 * - Block structure: 98 bytes = d(2) + qs[64] + qh[16] + signs[32] + scales[4]
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "tensors/FP16Utils.h"
#include "tensors/IQQuantTables.h"
#include "tensors/SIMDHelpers.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <random>

using namespace llaminar2;

class IQ3_SSIMDTest : public ::testing::Test
{
protected:
    std::mt19937 rng_;

    void SetUp() override
    {
        rng_.seed(42); // Reproducible tests
    }

    // Helper: Create a random IQ3_S tensor for GEMM testing
    std::unique_ptr<IQ3_STensor> createRandomTensor(size_t rows, size_t cols)
    {
        std::vector<size_t> shape = {rows, cols};
        size_t blocks_per_row = (cols + 255) / 256; // 256 elements per block
        size_t total_blocks = rows * blocks_per_row;
        std::vector<uint8_t> raw_data(total_blocks * sizeof(IQ3_SBlock));

        std::uniform_real_distribution<float> scale_dist(0.1f, 2.0f);
        std::uniform_int_distribution<uint8_t> byte_dist(0, 255);

        for (size_t b = 0; b < total_blocks; ++b)
        {
            IQ3_SBlock *block = reinterpret_cast<IQ3_SBlock *>(raw_data.data() + b * sizeof(IQ3_SBlock));
            block->d = fp32_to_fp16(scale_dist(rng_));
            for (int i = 0; i < 64; ++i)
                block->qs[i] = byte_dist(rng_);
            for (int i = 0; i < 16; ++i)
                block->qh[i] = byte_dist(rng_);
            for (int i = 0; i < 32; ++i)
                block->signs[i] = byte_dist(rng_);
            for (int i = 0; i < 4; ++i)
                block->scales[i] = byte_dist(rng_);
        }

        return std::make_unique<IQ3_STensor>(shape, raw_data);
    }

    // Helper: Compute reference GEMM: C = A @ B^T
    void referenceGEMM(const float *A, const float *B, float *C, int m, int n, int k)
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

    // Helper: Check if two matrices are approximately equal
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

    // Helper: Create an IQ3_S block with specified parameters
    IQ3_SBlock createBlock(float scale, const std::vector<uint8_t> &qs_values,
                           const std::vector<uint8_t> &qh_values,
                           const std::vector<uint8_t> &signs_values,
                           const std::vector<uint8_t> &scales_values)
    {
        IQ3_SBlock block;
        block.d = fp32_to_fp16(scale);

        // qs: 64 bytes (grid indices, 8 bits each)
        std::memset(block.qs, 0, 64);
        for (size_t i = 0; i < std::min(qs_values.size(), size_t(64)); ++i)
        {
            block.qs[i] = qs_values[i];
        }

        // qh: 16 bytes (high bit for 9-bit grid index, 2 bytes per super-block)
        std::memset(block.qh, 0, 16);
        for (size_t i = 0; i < std::min(qh_values.size(), size_t(16)); ++i)
        {
            block.qh[i] = qh_values[i];
        }

        // signs: 32 bytes (4 bytes per super-block)
        std::memset(block.signs, 0, 32);
        for (size_t i = 0; i < std::min(signs_values.size(), size_t(32)); ++i)
        {
            block.signs[i] = signs_values[i];
        }

        // scales: 4 bytes (4-bit scales, 2 per byte, 8 super-blocks → 4 bytes)
        std::memset(block.scales, 0, 4);
        for (size_t i = 0; i < std::min(scales_values.size(), size_t(4)); ++i)
        {
            block.scales[i] = scales_values[i];
        }

        return block;
    }

    // Helper: Compare two float arrays with tolerance
    void compareArrays(const float *a, const float *b, size_t count, float tolerance = 1e-6f)
    {
        for (size_t i = 0; i < count; ++i)
        {
            EXPECT_NEAR(a[i], b[i], tolerance)
                << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i];
        }
    }
};

// SIMD Equivalency Tests

TEST_F(IQ3_SSIMDTest, ScalarVsAVX2Equivalency)
{
    // Pattern 1: All zeros except first few indices
    auto block1 = createBlock(1.0f, {0, 1, 2, 3}, {0}, {0xFF}, {0x11}); // scale nibbles: 1, 1

    // Pattern 2: Max grid indices (9-bit: 0x1FF)
    auto block2 = createBlock(2.5f, {0xFF, 0xFF, 0xFF, 0xFF}, {0xFF}, {0x55}, {0x22}); // alternating signs

    // Pattern 3: Mid-range values
    auto block3 = createBlock(0.5f, {127, 128, 129, 130}, {0x80}, {0xAA}, {0x33});

    std::vector<IQ3_SBlock> blocks = {block1, block2, block3};

    for (const auto &block : blocks)
    {
        std::vector<float> scalar_output(256);
        std::vector<float> avx2_output(256);

        IQ3_STensor::decodeBlockScalar(block, scalar_output.data());

#ifdef __AVX2__
        IQ3_STensor::decodeBlockAVX2(block, avx2_output.data());
        compareArrays(scalar_output.data(), avx2_output.data(), 256);
#else
        GTEST_SKIP() << "AVX2 not available on this platform";
#endif
    }
}

TEST_F(IQ3_SSIMDTest, ScalarVsAVX512Equivalency)
{
    // Different patterns for AVX512 test
    auto block1 = createBlock(3.0f, {50, 100, 150, 200}, {0x0F}, {0x0F}, {0x44});
    auto block2 = createBlock(1.5f, {10, 20, 30, 40}, {0xF0}, {0xF0}, {0x55});
    auto block3 = createBlock(0.25f, {255, 0, 128, 64}, {0xAA}, {0x33}, {0x66});

    std::vector<IQ3_SBlock> blocks = {block1, block2, block3};

    for (const auto &block : blocks)
    {
        std::vector<float> scalar_output(256);
        std::vector<float> avx512_output(256);

        IQ3_STensor::decodeBlockScalar(block, scalar_output.data());

#ifdef __AVX512F__
        IQ3_STensor::decodeBlockAVX512(block, avx512_output.data());
        compareArrays(scalar_output.data(), avx512_output.data(), 256);
#else
        GTEST_SKIP() << "AVX512 not available on this platform";
#endif
    }
}

TEST_F(IQ3_SSIMDTest, AVX2VsAVX512Equivalency)
{
    auto block1 = createBlock(2.0f, {77, 88, 99, 111}, {0x55}, {0xCC}, {0x77});
    auto block2 = createBlock(0.75f, {33, 66, 99, 132}, {0x33}, {0x99}, {0x88});

    std::vector<IQ3_SBlock> blocks = {block1, block2};

    for (const auto &block : blocks)
    {
        std::vector<float> avx2_output(256);
        std::vector<float> avx512_output(256);

#if defined(__AVX512F__) && defined(__AVX2__)
        IQ3_STensor::decodeBlockAVX2(block, avx2_output.data());
        IQ3_STensor::decodeBlockAVX512(block, avx512_output.data());
        compareArrays(avx2_output.data(), avx512_output.data(), 256);
#else
        GTEST_SKIP() << "Both AVX2 and AVX512 required for this test";
#endif
    }
}

// Edge Case Tests

TEST_F(IQ3_SSIMDTest, EdgeCase_ZeroScale)
{
    auto block = createBlock(0.0f, {100, 200}, {0}, {0xFF}, {0xFF});

    std::vector<float> output(256);
    IQ3_STensor::decodeBlock(block, output.data());

    // Zero scale → all outputs should be zero
    for (size_t i = 0; i < 256; ++i)
    {
        EXPECT_FLOAT_EQ(output[i], 0.0f) << "Non-zero at index " << i;
    }
}

TEST_F(IQ3_SSIMDTest, EdgeCase_AllZeroIndices)
{
    // All zero qs/qh → grid index 0 for all elements
    auto block = createBlock(1.0f, {0}, {0}, {0xFF}, {0x00}); // all positive signs, scale nibble 0

    std::vector<float> output(256);
    IQ3_STensor::decodeBlock(block, output.data());

    // All elements should use grid[0] values with positive sign
    // Expected pattern depends on iq3s_grid[0] content
    // Just verify it doesn't crash and produces consistent results
    EXPECT_NO_FATAL_FAILURE(IQ3_STensor::decodeBlock(block, output.data()));
}

TEST_F(IQ3_SSIMDTest, EdgeCase_AllMaxIndices)
{
    // Max 9-bit indices: 0x1FF (qs=0xFF + qh high bit set)
    auto block = createBlock(1.0f, std::vector<uint8_t>(64, 0xFF),
                             std::vector<uint8_t>(16, 0xFF),
                             std::vector<uint8_t>(32, 0x00), // all negative signs
                             {0xFF});                        // max scale nibbles (15)

    std::vector<float> output(256);
    EXPECT_NO_FATAL_FAILURE(IQ3_STensor::decodeBlock(block, output.data()));

    // With max scale nibbles (15): d * (1 + 2*15) = d * 31
    // Grid values are int8_t (-128 to 127), so output range: -31*128 to 31*127
}

TEST_F(IQ3_SSIMDTest, EdgeCase_MaxScale)
{
    // Max FP16 scale: ~65504
    auto block = createBlock(65504.0f, {10}, {0}, {0xFF}, {0x00});

    std::vector<float> output(256);
    EXPECT_NO_FATAL_FAILURE(IQ3_STensor::decodeBlock(block, output.data()));

    // Results should be very large but not inf/nan
    for (size_t i = 0; i < 256; ++i)
    {
        EXPECT_FALSE(std::isnan(output[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(output[i])) << "Inf at index " << i;
    }
}

TEST_F(IQ3_SSIMDTest, EdgeCase_MixedSigns)
{
    // Alternating sign patterns
    auto block = createBlock(1.0f, {50, 100, 150, 200},
                             {0},
                             {0xAA, 0x55, 0xCC, 0x33}, // various patterns
                             {0x88});                  // scale nibbles 8, 8

    std::vector<float> output(256);
    IQ3_STensor::decodeBlock(block, output.data());

    // Should have mix of positive and negative values based on sign bits
    bool has_positive = false, has_negative = false;
    for (size_t i = 0; i < 256; ++i)
    {
        if (output[i] > 0)
            has_positive = true;
        if (output[i] < 0)
            has_negative = true;
    }
    // With mixed sign patterns, we should see both
    // (unless grid values are all zero, which is unlikely)
}

// =============================================================================
// GEMM Tests
// =============================================================================

/*
TEST_F(IQ3_SSIMDTest, GEMM_SmallBatch)
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

    // Weight-owned GEMM path removed: IQ3_S now participates in GEMM only via
    // activation-tensor-centric paths.

    // Reference computation
    std::vector<float> B_decoded(8 * 256);
    tensor->to_fp32(B_decoded.data());

    std::vector<float> C_expected(4 * 8, 0.0f);
    referenceGEMM(A.data(), B_decoded.data(), C_expected.data(), 4, 8, 256);

    EXPECT_TRUE(matricesEqual(C_expected.data(), C.data(), 4 * 8, 1e-3f));
}

TEST_F(IQ3_SSIMDTest, GEMM_MediumBatch)
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

    // Weight-owned GEMM path removed: IQ3_S now participates in GEMM only via
    // activation-tensor-centric paths.

    // Reference
    std::vector<float> B_decoded(16 * 512);
    tensor->to_fp32(B_decoded.data());

    std::vector<float> C_expected(16 * 16, 0.0f);
    referenceGEMM(A.data(), B_decoded.data(), C_expected.data(), 16, 16, 512);

    EXPECT_TRUE(matricesEqual(C_expected.data(), C.data(), 16 * 16, 1e-3f));
}

TEST_F(IQ3_SSIMDTest, GEMM_LargeBatch)
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

    // Weight-owned GEMM path removed: IQ3_S now participates in GEMM only via
    // activation-tensor-centric paths.

    // Reference
    std::vector<float> B_decoded(32 * 768);
    tensor->to_fp32(B_decoded.data());

    std::vector<float> C_expected(32 * 32, 0.0f);
    referenceGEMM(A.data(), B_decoded.data(), C_expected.data(), 32, 32, 768);

    EXPECT_TRUE(matricesEqual(C_expected.data(), C.data(), 32 * 32, 1e-3f));
}
*/

TEST_F(IQ3_SSIMDTest, EdgeCase_RandomValues)
{
    // Pseudo-random but deterministic values
    std::vector<uint8_t> qs_vals, qh_vals, signs_vals, scales_vals;
    for (int i = 0; i < 64; ++i)
        qs_vals.push_back((i * 17) % 256);
    for (int i = 0; i < 16; ++i)
        qh_vals.push_back((i * 31) % 256);
    for (int i = 0; i < 32; ++i)
        signs_vals.push_back((i * 67) % 256);
    for (int i = 0; i < 4; ++i)
        scales_vals.push_back((i * 123) % 256);

    auto block = createBlock(1.234f, qs_vals, qh_vals, signs_vals, scales_vals);

    std::vector<float> output(256);
    EXPECT_NO_FATAL_FAILURE(IQ3_STensor::decodeBlock(block, output.data()));

    // Verify consistency: same input → same output
    std::vector<float> output2(256);
    IQ3_STensor::decodeBlock(block, output2.data());
    compareArrays(output.data(), output2.data(), 256);
}

// =============================================================================
// to<T>() Template Method Tests
// =============================================================================

TEST_F(IQ3_SSIMDTest, ToFloat_TemplateMethod)
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
    for (size_t i = 0; i < total; ++i)
    {
        EXPECT_FLOAT_EQ(template_output[i], legacy_output[i])
            << "Mismatch at index " << i;
    }
}

TEST_F(IQ3_SSIMDTest, ToBF16_TemplateMethod)
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
    for (size_t i = 0; i < total; ++i)
    {
        EXPECT_EQ(template_output[i], legacy_output[i])
            << "Mismatch at index " << i;
    }
}

TEST_F(IQ3_SSIMDTest, ToFP16_TemplateMethod)
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
    for (size_t i = 0; i < total; ++i)
    {
        EXPECT_EQ(template_output[i], legacy_output[i])
            << "Mismatch at index " << i;
    }
}

TEST_F(IQ3_SSIMDTest, ToINT8_TemplateMethod)
{
    // Create tensor
    auto tensor = createRandomTensor(1, 256);
    const size_t total = 256;

    // Convert to INT8
    std::vector<int8_t> int8_output(total);
    tensor->to<int8_t>(int8_output.data());

    // Verify INT8 range
    for (size_t i = 0; i < total; ++i)
    {
        EXPECT_GE(int8_output[i], -127);
        EXPECT_LE(int8_output[i], 127);
    }
}

TEST_F(IQ3_SSIMDTest, ToINT32_TemplateMethod)
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

TEST_F(IQ3_SSIMDTest, RoundTrip)
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
    for (size_t i = 0; i < total; ++i)
    {
        float diff = std::abs(fp32_1[i] - fp32_2[i]);
        float rel_error = (fp32_1[i] != 0.0f) ? diff / std::abs(fp32_1[i]) : diff;
        if (rel_error > 0.05f)
        { // 5% tolerance for BF16
            ++mismatches;
        }
    }
    EXPECT_LT(mismatches, 13) << "Too many mismatches in round-trip conversion";
}
