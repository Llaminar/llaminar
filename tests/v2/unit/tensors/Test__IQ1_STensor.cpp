/**
 * @file Test__IQ1_STensor.cpp
 * @brief SIMD equivalency tests for IQ1_S decoding
 *
 * IQ1_S Structure:
 * - Block size: 256 elements (8 iterations of 32 elements each, 4 groups of 8 per iteration)
 * - Quantization: 1-bit using iq1s_grid[2048] lookup table (11-bit indices: 8 bits from qs + 3 bits from qh)
 * - Scaling: FP16 global scale + 8 sub-scales (3-bit each, from qh[ib] bits 12-14)
 * - Delta: Per-iteration delta (±IQ1S_DELTA, sign from qh[ib] bit 15)
 * - Formula: output = dl * (grid[j] + delta) where dl = d * (2 * sub_scale + 1)
 * - Block structure: 22 bytes = d(2) + qs[32] (but qh is uint16_t[8] = 16 bytes → total 50 bytes with padding)
 *
 * NOTE: IQ1_S has an unaligned pointer warning in the original implementation due to qh being uint16_t*
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

class IQ1_SSIMDTest : public ::testing::Test
{
protected:
    std::mt19937 rng_;

    void SetUp() override
    {
        rng_.seed(42);
    }

    // GEMM helper: Create random tensor
    std::unique_ptr<IQ1_STensor> createRandomTensor(size_t rows, size_t cols)
    {
        std::vector<size_t> shape = {rows, cols};
        size_t blocks_per_row = (cols + 255) / 256;
        size_t total_blocks = rows * blocks_per_row;
        std::vector<uint8_t> raw_data(total_blocks * sizeof(IQ1_SBlock));
        std::uniform_int_distribution<uint8_t> byte_dist(0, 255);
        for (size_t b = 0; b < total_blocks; ++b)
        {
            IQ1_SBlock *block = reinterpret_cast<IQ1_SBlock *>(raw_data.data() + b * sizeof(IQ1_SBlock));
            for (size_t i = 0; i < sizeof(IQ1_SBlock); ++i)
                reinterpret_cast<uint8_t *>(block)[i] = byte_dist(rng_);
        }
        return std::make_unique<IQ1_STensor>(shape, raw_data);
    }

    // GEMM helper: Reference implementation
    void referenceGEMM(const float *A, const float *B, float *C, int m, int n, int k)
    {
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                float sum = 0.0f;
                for (int l = 0; l < k; ++l)
                    sum += A[i * k + l] * B[j * k + l];
                C[i * n + j] = sum;
            }
        }
    }

    // GEMM helper: Matrix equality check
    bool matricesEqual(const float *A, const float *B, int size, float tolerance = 1e-4f)
    {
        for (int i = 0; i < size; ++i)
        {
            float diff = std::abs(A[i] - B[i]);
            float rel_error = diff / (std::abs(A[i]) + 1e-8f);
            if (diff > tolerance && rel_error > tolerance)
                return false;
        }
        return true;
    }

    // Helper: Create an IQ1_S block with specified parameters
    IQ1_SBlock createBlock(float scale, const std::vector<uint8_t> &qs_values,
                           const std::vector<uint16_t> &qh_values)
    {
        IQ1_SBlock block;
        block.d = fp32_to_fp16(scale);

        // qs: 32 bytes (grid index low 8 bits)
        std::memset(block.qs, 0, 32);
        for (size_t i = 0; i < std::min(qs_values.size(), size_t(32)); ++i)
        {
            block.qs[i] = qs_values[i];
        }

        // qh: 8 uint16_t values (grid index high 3 bits + 3-bit scale + delta sign)
        // Bits 0-2: grid high bits for group 0
        // Bits 3-5: grid high bits for group 1
        // Bits 6-8: grid high bits for group 2
        // Bits 9-11: grid high bits for group 3
        // Bits 12-14: 3-bit sub-scale
        // Bit 15: delta sign
        std::memset(block.qh, 0, sizeof(block.qh));
        for (size_t i = 0; i < std::min(qh_values.size(), size_t(8)); ++i)
        {
            block.qh[i] = qh_values[i];
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

TEST_F(IQ1_SSIMDTest, ScalarVsAVX2Equivalency)
{
    // Pattern 1: Simple values with positive delta
    auto block1 = createBlock(1.0f, {0, 1, 2, 3}, {0x1000}); // scale=1 (bits 12-14), delta=+ (bit 15=0)

    // Pattern 2: High grid indices with negative delta
    auto block2 = createBlock(2.5f, {0xFF, 0xFE, 0xFD, 0xFC}, {0xF000}); // scale=7, delta=- (bit 15=1)

    // Pattern 3: Mid-range with grid high bits
    auto block3 = createBlock(0.5f, {127, 128, 129, 130}, {0x5123}); // scale=5, grid bits 1,2,3 for groups

    std::vector<IQ1_SBlock> blocks = {block1, block2, block3};

    for (const auto &block : blocks)
    {
        std::vector<float> scalar_output(256);
        std::vector<float> avx2_output(256);

        IQ1_STensor::decodeBlockScalar(block, scalar_output.data());

#ifdef __AVX2__
        IQ1_STensor::decodeBlockAVX2(block, avx2_output.data());
        compareArrays(scalar_output.data(), avx2_output.data(), 256);
#else
        GTEST_SKIP() << "AVX2 not available on this platform";
#endif
    }
}

TEST_F(IQ1_SSIMDTest, ScalarVsAVX512Equivalency)
{
    auto block1 = createBlock(3.0f, {50, 100, 150, 200}, {0x2000});
    auto block2 = createBlock(1.5f, {10, 20, 30, 40}, {0x6FFF}); // various bits set
    auto block3 = createBlock(0.25f, {255, 0, 128, 64}, {0x4567});

    std::vector<IQ1_SBlock> blocks = {block1, block2, block3};

    for (const auto &block : blocks)
    {
        std::vector<float> scalar_output(256);
        std::vector<float> avx512_output(256);

        IQ1_STensor::decodeBlockScalar(block, scalar_output.data());

#ifdef __AVX512F__
        IQ1_STensor::decodeBlockAVX512(block, avx512_output.data());
        compareArrays(scalar_output.data(), avx512_output.data(), 256);
#else
        GTEST_SKIP() << "AVX512 not available on this platform";
#endif
    }
}

TEST_F(IQ1_SSIMDTest, AVX2VsAVX512Equivalency)
{
    auto block1 = createBlock(2.0f, {77, 88, 99, 111}, {0x3456});
    auto block2 = createBlock(0.75f, {33, 66, 99, 132}, {0x789A});

    std::vector<IQ1_SBlock> blocks = {block1, block2};

    for (const auto &block : blocks)
    {
        std::vector<float> avx2_output(256);
        std::vector<float> avx512_output(256);

#if defined(__AVX512F__) && defined(__AVX2__)
        IQ1_STensor::decodeBlockAVX2(block, avx2_output.data());
        IQ1_STensor::decodeBlockAVX512(block, avx512_output.data());
        compareArrays(avx2_output.data(), avx512_output.data(), 256);
#else
        GTEST_SKIP() << "Both AVX2 and AVX512 required for this test";
#endif
    }
}

// Edge Case Tests

TEST_F(IQ1_SSIMDTest, EdgeCase_ZeroScale)
{
    auto block = createBlock(0.0f, {100, 200}, {0x1000});

    std::vector<float> output(256);
    IQ1_STensor::decodeBlock(block, output.data());

    // Zero scale → all outputs should be zero
    for (size_t i = 0; i < 256; ++i)
    {
        EXPECT_FLOAT_EQ(output[i], 0.0f) << "Non-zero at index " << i;
    }
}

TEST_F(IQ1_SSIMDTest, EdgeCase_AllZeroIndices)
{
    // All zero qs/qh → grid index 0 for all elements
    auto block = createBlock(1.0f, {0}, {0x0000}); // scale=0 (d * 1), delta=+

    std::vector<float> output(256);
    EXPECT_NO_FATAL_FAILURE(IQ1_STensor::decodeBlock(block, output.data()));
}

TEST_F(IQ1_SSIMDTest, EdgeCase_AllMaxIndices)
{
    // Max 11-bit indices: 0x7FF (qs=0xFF + qh high bits = 7)
    auto block = createBlock(1.0f, std::vector<uint8_t>(32, 0xFF),
                             {0xFFF7, 0xFFF7, 0xFFF7, 0xFFF7, 0xFFF7, 0xFFF7, 0xFFF7, 0xFFF7});
    // bits 0-2, 3-5, 6-8, 9-11 all set to 7 (max 3-bit value)

    std::vector<float> output(256);
    EXPECT_NO_FATAL_FAILURE(IQ1_STensor::decodeBlock(block, output.data()));
}

TEST_F(IQ1_SSIMDTest, EdgeCase_MaxScale)
{
    // Max FP16 scale + max sub-scale (7)
    auto block = createBlock(65504.0f, {10}, {0x7000}); // sub-scale=7 (bits 12-14), delta=+

    std::vector<float> output(256);
    EXPECT_NO_FATAL_FAILURE(IQ1_STensor::decodeBlock(block, output.data()));

    for (size_t i = 0; i < 256; ++i)
    {
        EXPECT_FALSE(std::isnan(output[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(output[i])) << "Inf at index " << i;
    }
}

TEST_F(IQ1_SSIMDTest, EdgeCase_MixedDeltaSigns)
{
    // Alternating delta signs across iterations
    auto block = createBlock(1.0f, {50, 100, 150, 200},
                             {0x1000,   // delta=+ (bit 15=0)
                              0x9000,   // delta=- (bit 15=1)
                              0x2000,   // delta=+
                              0xA000,   // delta=-
                              0x3000,   // delta=+
                              0xB000,   // delta=-
                              0x4000,   // delta=+
                              0xC000}); // delta=-

    std::vector<float> output(256);
    EXPECT_NO_FATAL_FAILURE(IQ1_STensor::decodeBlock(block, output.data()));
}

// =============================================================================
// GEMM Tests
// =============================================================================

TEST_F(IQ1_SSIMDTest, GEMM_SmallBatch)
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

TEST_F(IQ1_SSIMDTest, GEMM_MediumBatch)
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

TEST_F(IQ1_SSIMDTest, GEMM_LargeBatch)
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

TEST_F(IQ1_SSIMDTest, EdgeCase_RandomValues)
{
    std::vector<uint8_t> qs_vals;
    std::vector<uint16_t> qh_vals;

    for (int i = 0; i < 32; ++i)
        qs_vals.push_back((i * 17) % 256);
    for (int i = 0; i < 8; ++i)
        qh_vals.push_back((i * 1237) % 65536);

    auto block = createBlock(1.234f, qs_vals, qh_vals);

    std::vector<float> output(256);
    EXPECT_NO_FATAL_FAILURE(IQ1_STensor::decodeBlock(block, output.data()));

    // Consistency check
    std::vector<float> output2(256);
    IQ1_STensor::decodeBlock(block, output2.data());
    compareArrays(output.data(), output2.data(), 256);
}


// =============================================================================
// to<T>() Template Method Tests
// =============================================================================

TEST_F(IQ1_SSIMDTest, ToFloat_TemplateMethod)
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

TEST_F(IQ1_SSIMDTest, ToBF16_TemplateMethod)
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

TEST_F(IQ1_SSIMDTest, ToFP16_TemplateMethod)
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

TEST_F(IQ1_SSIMDTest, ToINT8_TemplateMethod)
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

TEST_F(IQ1_SSIMDTest, ToINT32_TemplateMethod)
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

TEST_F(IQ1_SSIMDTest, RoundTrip)
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
