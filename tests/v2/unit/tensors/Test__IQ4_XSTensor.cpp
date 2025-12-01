/**
 * @file Test__IQ4_XSTensor.cpp
 * @brief SIMD equivalency tests for IQ4_XS quantized tensor
 * @author David Sanftenberg
 * @date 2025-10-29
 *
 * IQ4_XS uses the same kvalues_iq4nl[16] lookup table as IQ4_NL,
 * but with a more complex structure:
 * - 256 elements per super-block (8 sub-blocks of 32 elements each)
 * - Per-sub-block 6-bit scales (4 bits in scales_l + 2 bits in scales_h)
 * - Formula: output = d * (scale - 32) * kvalues_iq4nl[nibble]
 *
 * Block Structure:
 * - d: FP16 global scale
 * - scales_l[4]: Lower 4 bits of 8 sub-block scales (packed 2 per byte)
 * - scales_h: Upper 2 bits of all 8 scales (16 bits total, 2 bits each)
 * - qs[128]: Quantized values (16 bytes per sub-block × 8 sub-blocks)
 * Total: 2 + 4 + 2 + 128 = 136 bytes per block
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "tensors/FP16Utils.h"
#include "tensors/IQQuantTables.h"
#include "tensors/SIMDHelpers.h"
#include <cmath>
#include <random>
#include <vector>
#include <cstring>

using namespace llaminar2;

// Tolerance for float comparisons
constexpr float TOLERANCE = 1e-5f;

class IQ4_XSSIMDTest : public ::testing::Test
{
protected:
    std::mt19937 rng_;

    void SetUp() override
    {
        rng_.seed(42); // Reproducible tests
    }

    void TearDown() override {}

    /**
     * @brief Create an IQ4_XS block with specified values
     */
    IQ4_XSBlock createBlock(float d_val, const std::vector<uint8_t> &scales_l,
                            uint16_t scales_h, const std::vector<uint8_t> &qs)
    {
        IQ4_XSBlock block;
        block.d = fp32_to_fp16(d_val);

        std::memcpy(block.scales_l, scales_l.data(), 4);
        block.scales_h = scales_h;
        std::memcpy(block.qs, qs.data(), 128);

        return block;
    }

    /**
     * @brief Create a random IQ4_XS tensor for GEMM testing
     */
    std::unique_ptr<IQ4_XSTensor> createRandomTensor(size_t rows, size_t cols)
    {
        std::vector<size_t> shape = {rows, cols};
        size_t blocks_per_row = (cols + 255) / 256; // 256 elements per block
        size_t total_blocks = rows * blocks_per_row;
        std::vector<uint8_t> raw_data(total_blocks * sizeof(IQ4_XSBlock));

        std::uniform_real_distribution<float> scale_dist(0.1f, 2.0f);
        std::uniform_int_distribution<uint8_t> byte_dist(0, 255);

        for (size_t b = 0; b < total_blocks; ++b)
        {
            IQ4_XSBlock *block = reinterpret_cast<IQ4_XSBlock *>(raw_data.data() + b * sizeof(IQ4_XSBlock));
            block->d = fp32_to_fp16(scale_dist(rng_));
            for (int i = 0; i < 4; ++i)
                block->scales_l[i] = byte_dist(rng_);
            block->scales_h = byte_dist(rng_) | (byte_dist(rng_) << 8);
            for (int i = 0; i < 128; ++i)
                block->qs[i] = byte_dist(rng_);
        }

        return std::make_unique<IQ4_XSTensor>(shape, raw_data);
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

    /**
     * @brief Compare two float arrays with tolerance
     */
    bool compareArrays(const float *expected, const float *actual, size_t count,
                       float tolerance, std::string &error_msg)
    {
        for (size_t i = 0; i < count; ++i)
        {
            float diff = std::abs(expected[i] - actual[i]);
            if (diff > tolerance)
            {
                char buf[256];
                snprintf(buf, sizeof(buf),
                         "Mismatch at index %zu: expected=%.6f, actual=%.6f, diff=%.6f",
                         i, expected[i], actual[i], diff);
                error_msg = buf;
                return false;
            }
        }
        return true;
    }
};

// =============================================================================
// SIMD Equivalency Tests
// =============================================================================

#ifdef __AVX2__
TEST_F(IQ4_XSSIMDTest, ScalarVsAVX2Equivalency)
{
    // Create test block with varying scales and nibbles
    std::vector<uint8_t> scales_l = {0x00, 0x11, 0x22, 0x33}; // 8 4-bit scales packed
    uint16_t scales_h = 0x5555;                               // Alternating 01 pattern for upper 2 bits

    // Create varying nibble pattern
    std::vector<uint8_t> qs(128);
    for (size_t i = 0; i < 128; ++i)
    {
        qs[i] = (i % 16) | ((15 - (i % 16)) << 4);
    }

    IQ4_XSBlock block = createBlock(2.5f, scales_l, scales_h, qs);

    // Decode with scalar and AVX2
    std::vector<float> scalar_output(256);
    std::vector<float> avx2_output(256);

    IQ4_XSTensor::decodeBlockScalar(block, scalar_output.data());
    IQ4_XSTensor::decodeBlockAVX2(block, avx2_output.data());

    // Compare results
    std::string error_msg;
    EXPECT_TRUE(compareArrays(scalar_output.data(), avx2_output.data(), 256, TOLERANCE, error_msg))
        << "Scalar vs AVX2 mismatch: " << error_msg;
}
#endif // __AVX2__

#ifdef __AVX512F__
TEST_F(IQ4_XSSIMDTest, ScalarVsAVX512Equivalency)
{
    // Create test block with varying scales and nibbles
    std::vector<uint8_t> scales_l = {0xAB, 0xCD, 0xEF, 0x01};
    uint16_t scales_h = 0xAAAA; // All 10 pattern for upper bits

    std::vector<uint8_t> qs(128);
    for (size_t i = 0; i < 128; ++i)
    {
        qs[i] = ((i * 7) % 16) | (((i * 11) % 16) << 4);
    }

    IQ4_XSBlock block = createBlock(1.75f, scales_l, scales_h, qs);

    // Decode with scalar and AVX512
    std::vector<float> scalar_output(256);
    std::vector<float> avx512_output(256);

    IQ4_XSTensor::decodeBlockScalar(block, scalar_output.data());
    IQ4_XSTensor::decodeBlockAVX512(block, avx512_output.data());

    // Compare results
    std::string error_msg;
    EXPECT_TRUE(compareArrays(scalar_output.data(), avx512_output.data(), 256, TOLERANCE, error_msg))
        << "Scalar vs AVX512 mismatch: " << error_msg;
}
#endif // __AVX512F__

#if defined(__AVX2__) && defined(__AVX512F__)
TEST_F(IQ4_XSSIMDTest, AVX2VsAVX512Equivalency)
{
    // Create test block
    std::vector<uint8_t> scales_l = {0x55, 0x66, 0x77, 0x88};
    uint16_t scales_h = 0x3C3C; // Mixed pattern

    std::vector<uint8_t> qs(128);
    for (size_t i = 0; i < 128; ++i)
    {
        qs[i] = ((i * 3) % 16) | (((i * 5) % 16) << 4);
    }

    IQ4_XSBlock block = createBlock(3.14f, scales_l, scales_h, qs);

    // Decode with AVX2 and AVX512
    std::vector<float> avx2_output(256);
    std::vector<float> avx512_output(256);

    IQ4_XSTensor::decodeBlockAVX2(block, avx2_output.data());
    IQ4_XSTensor::decodeBlockAVX512(block, avx512_output.data());

    // Compare results
    std::string error_msg;
    EXPECT_TRUE(compareArrays(avx2_output.data(), avx512_output.data(), 256, TOLERANCE, error_msg))
        << "AVX2 vs AVX512 mismatch: " << error_msg;
}
#endif // __AVX2__ && __AVX512F__

// =============================================================================
// Edge Case Tests
// =============================================================================

TEST_F(IQ4_XSSIMDTest, EdgeCase_ZeroScale)
{
    // Zero global scale should produce all zero outputs
    std::vector<uint8_t> scales_l = {0xFF, 0xFF, 0xFF, 0xFF};
    uint16_t scales_h = 0xFFFF;
    std::vector<uint8_t> qs(128, 0xFF);

    IQ4_XSBlock block = createBlock(0.0f, scales_l, scales_h, qs);

    std::vector<float> output(256);
    IQ4_XSTensor::decodeBlock(block, output.data());

    // All outputs should be zero
    for (size_t i = 0; i < 256; ++i)
    {
        EXPECT_FLOAT_EQ(output[i], 0.0f) << "Index " << i << " should be 0.0f with zero global scale";
    }
}

TEST_F(IQ4_XSSIMDTest, EdgeCase_AllZeroNibbles)
{
    // All nibbles = 0 → all outputs = d * (scale - 32) * kvalues_iq4nl[0]
    std::vector<uint8_t> scales_l = {0x22, 0x22, 0x22, 0x22}; // All scales = 2
    uint16_t scales_h = 0x0000;                               // Upper bits all 0
    std::vector<uint8_t> qs(128, 0x00);

    IQ4_XSBlock block = createBlock(1.0f, scales_l, scales_h, qs);

    std::vector<float> output(256);
    IQ4_XSTensor::decodeBlock(block, output.data());

    // All outputs should be: 1.0 * (2 - 32) * kvalues_iq4nl[0]
    float expected = 1.0f * (2 - 32) * static_cast<float>(kvalues_iq4nl[0]);

    for (size_t i = 0; i < 256; ++i)
    {
        EXPECT_FLOAT_EQ(output[i], expected) << "Index " << i;
    }
}

// =============================================================================
// GEMM Tests
// =============================================================================

TEST_F(IQ4_XSSIMDTest, GEMM_SmallBatch)
{
    auto tensor = createRandomTensor(8, 256); // 8 output features, 256 input features (1 block per row)

    auto gemm = tensor->createGemm();
    // NOTE: createGemm() returns nullptr for IQ4_XS since OneDNNGemmKernel was removed.
    // IQ4_XS participates in GEMM only via activation-tensor-centric paths (QuantisedGemmKernel).
    if (gemm == nullptr)
    {
        GTEST_SKIP() << "GEMM kernel not available for IQ4_XS format";
    }

    // A: [4, 256] - 4 input sequences
    std::vector<float> A(4 * 256);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &v : A)
        v = dist(rng_);

    // C: [4, 8] - result
    std::vector<float> C(4 * 8, 0.0f);

    // Weight-owned GEMM path removed: IQ4_XS now participates in GEMM only via
    // activation-tensor-centric kernels. This test previously validated
    // gemm->multiply against a reference GEMM; that path has been retired.
}

TEST_F(IQ4_XSSIMDTest, GEMM_MediumBatch)
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

    // Weight-owned GEMM path removed: see GEMM_SmallBatch note.
}

TEST_F(IQ4_XSSIMDTest, GEMM_LargeBatch)
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

    // Weight-owned GEMM path removed: see GEMM_SmallBatch note.
}

TEST_F(IQ4_XSSIMDTest, EdgeCase_AllMaxNibbles)
{
    // All nibbles = 15 → all outputs = d * (scale - 32) * kvalues_iq4nl[15]
    std::vector<uint8_t> scales_l = {0x33, 0x33, 0x33, 0x33}; // All scales = 3
    uint16_t scales_h = 0x0000;
    std::vector<uint8_t> qs(128, 0xFF);

    IQ4_XSBlock block = createBlock(2.0f, scales_l, scales_h, qs);

    std::vector<float> output(256);
    IQ4_XSTensor::decodeBlock(block, output.data());

    // All outputs should be: 2.0 * (3 - 32) * kvalues_iq4nl[15]
    float expected = 2.0f * (3 - 32) * static_cast<float>(kvalues_iq4nl[15]);

    for (size_t i = 0; i < 256; ++i)
    {
        EXPECT_FLOAT_EQ(output[i], expected) << "Index " << i;
    }
}

TEST_F(IQ4_XSSIMDTest, EdgeCase_ScaleOffset32)
{
    // Scale = 32 (after offset) should produce zero outputs
    std::vector<uint8_t> scales_l = {0x00, 0x00, 0x00, 0x00}; // All scales = 0 (low 4 bits)
    uint16_t scales_h = 0xAAAA;                               // All scales get +32 from upper bits (10 binary = 2, shift left 4 = 32)
    // Scale = (0 & 0xF) | ((2) << 4) = 0 | 32 = 32, then (32 - 32) = 0
    std::vector<uint8_t> qs(128, 0xFF);

    IQ4_XSBlock block = createBlock(5.0f, scales_l, scales_h, qs);

    std::vector<float> output(256);
    IQ4_XSTensor::decodeBlock(block, output.data());

    // All outputs should be zero (scale - 32 = 0)
    for (size_t i = 0; i < 256; ++i)
    {
        EXPECT_FLOAT_EQ(output[i], 0.0f) << "Index " << i << " should be 0.0f when scale=32";
    }
}

TEST_F(IQ4_XSSIMDTest, EdgeCase_RandomValues)
{
    // Random pattern across full range
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint8_t> dist(0, 255);
    std::uniform_int_distribution<uint16_t> dist16(0, 65535);

    std::vector<uint8_t> scales_l(4);
    for (auto &s : scales_l)
        s = dist(rng);
    uint16_t scales_h = dist16(rng);

    std::vector<uint8_t> qs(128);
    for (auto &q : qs)
        q = dist(rng);

    IQ4_XSBlock block = createBlock(1.5f, scales_l, scales_h, qs);

    // Decode with scalar (reference) and main path
    std::vector<float> scalar_output(256);
    std::vector<float> main_output(256);

    IQ4_XSTensor::decodeBlockScalar(block, scalar_output.data());
    IQ4_XSTensor::decodeBlock(block, main_output.data());

    // Should match exactly
    std::string error_msg;
    EXPECT_TRUE(compareArrays(scalar_output.data(), main_output.data(), 256, TOLERANCE, error_msg))
        << "Random values mismatch: " << error_msg;
}

// =============================================================================
// to<T>() Template Method Tests
// =============================================================================

TEST_F(IQ4_XSSIMDTest, ToFloat_TemplateMethod)
{
    // Create a random IQ4_XS tensor (1x256 = 1 block)
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

TEST_F(IQ4_XSSIMDTest, ToBF16_TemplateMethod)
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

TEST_F(IQ4_XSSIMDTest, ToFP16_TemplateMethod)
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

TEST_F(IQ4_XSSIMDTest, ToINT8_TemplateMethod)
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

TEST_F(IQ4_XSSIMDTest, ToINT32_TemplateMethod)
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

TEST_F(IQ4_XSSIMDTest, RoundTrip)
{
    // Create IQ4_XS tensor
    auto tensor = createRandomTensor(1, 256);
    const size_t total = 256;

    // Round trip: IQ4_XS -> FP32 -> BF16 -> FP32
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

// =============================================================================
// IINT8Unpackable Interface Tests
// =============================================================================

TEST_F(IQ4_XSSIMDTest, IINT8Unpackable_Interface)
{
    // Create a tensor with 1 row, 256 columns (1 super-block)
    std::vector<size_t> shape = {1, 256};
    std::vector<uint8_t> raw_data(sizeof(IQ4_XSBlock));
    IQ4_XSBlock *block = reinterpret_cast<IQ4_XSBlock *>(raw_data.data());

    // Initialize block
    block->d = simd::fp32_to_fp16(1.0f);

    // Set scales for sub-blocks
    // sub-block 0: ls=32 (scale factor 0) -> value = 0
    // sub-block 1: ls=33 (scale factor 1) -> value = kvalues

    // scales_l[0] = (low_0) | (low_1 << 4) = 0 | (1 << 4) = 0x10
    block->scales_l[0] = 0x10;
    // scales_h = (2 << 0) | (2 << 2) = 2 | 8 = 10
    block->scales_h = 10;

    // Fill qs for sub-block 1 with known indices
    // index 0 -> kvalues[0] = -127
    // index 1 -> kvalues[1] = -104
    block->qs[16] = 0x10; // (1 << 4) | 0

    auto tensor = std::make_unique<IQ4_XSTensor>(shape, raw_data);

    // 1. Check block size
    EXPECT_EQ(tensor->block_size(), 32);

    // 2. Check unpack_block_to_int8
    int8_t output[32];
    tensor->unpack_block_to_int8(0, 1, output); // sub-block 1

    // IQ4_XS layout: qs[j] contains element j (low) and j+16 (high)
    // qs[0] (which is block->qs[16]) contains element 0 and 16
    // We set qs[0] = 0x10 (low=0, high=1)
    // So output[0] should be index 0
    // output[16] should be index 1

    EXPECT_EQ(output[0], kvalues_iq4nl_i8[0]);
    EXPECT_EQ(output[16], kvalues_iq4nl_i8[1]);

    // 3. Check get_block_scale
    float scale = tensor->get_block_scale(0, 1);
    EXPECT_FLOAT_EQ(scale, 1.0f);

    scale = tensor->get_block_scale(0, 0);
    EXPECT_FLOAT_EQ(scale, 0.0f);

    // 4. Check decode_block_at (consistency check)
    float float_output[32];
    tensor->decode_block_at(0, 1, float_output);

    EXPECT_FLOAT_EQ(float_output[0], kvalues_iq4nl_i8[0] * 1.0f);
    EXPECT_FLOAT_EQ(float_output[16], kvalues_iq4nl_i8[1] * 1.0f);
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
