/**
 * @file Test__IQ4_NLTensor.cpp
 * @brief Unit tests for IQ4_NLTensor class (GEMM operations)
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "../../src/v2/tensors/Tensors.h"
#include "../../src/v2/tensors/FP16Utils.h"
#include "../../src/v2/tensors/IQQuantTables.h"
#include <cmath>
#include <random>
#include <vector>
#include <iostream>

using namespace llaminar2;

/**
 * @brief Test fixture for IQ4_NL tensor tests
 */
class Test__IQ4_NLTensor : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Seed for reproducible tests
        rng_.seed(42);
    }

    /**
     * @brief Create a simple IQ4_NL tensor with known values
     *
     * Creates a 2×32 tensor (1 block per row) with predictable quantized values
     */
    std::unique_ptr<IQ4_NLTensor> createSimpleTensor()
    {
        std::vector<size_t> shape = {2, 32}; // 2 rows, 32 cols (1 block each)

        // Create 2 blocks (1 per row)
        std::vector<uint8_t> raw_data(2 * 18); // 2 blocks × 18 bytes

        // Block 0 (row 0): scale = 1.0 (as FP16), indices all 0 (maps to kvalues_iq4nl[0])
        IQ4_NLBlock *block0 = reinterpret_cast<IQ4_NLBlock *>(raw_data.data());
        block0->d = fp32_to_fp16(1.0f);
        for (int i = 0; i < 16; ++i)
        {
            block0->qs[i] = 0x00; // Low nibble=0, high nibble=0
        }

        // Block 1 (row 1): scale = 2.0 (as FP16), indices all 1 (maps to kvalues_iq4nl[1])
        IQ4_NLBlock *block1 = reinterpret_cast<IQ4_NLBlock *>(raw_data.data() + 18);
        block1->d = fp32_to_fp16(2.0f);
        for (int i = 0; i < 16; ++i)
        {
            block1->qs[i] = 0x11; // Low nibble=1, high nibble=1
        }

        return std::make_unique<IQ4_NLTensor>(shape, raw_data);
    }

    /**
     * @brief Create a random IQ4_NL tensor for stress testing
     */
    std::unique_ptr<IQ4_NLTensor> createRandomTensor(size_t rows, size_t cols)
    {
        std::vector<size_t> shape = {rows, cols};
        size_t blocks_per_row = (cols + 31) / 32;
        size_t total_blocks = rows * blocks_per_row;
        std::vector<uint8_t> raw_data(total_blocks * 18);

        std::uniform_real_distribution<float> scale_dist(0.1f, 2.0f);
        std::uniform_int_distribution<uint8_t> index_dist(0, 15);

        for (size_t b = 0; b < total_blocks; ++b)
        {
            IQ4_NLBlock *block = reinterpret_cast<IQ4_NLBlock *>(raw_data.data() + b * 18);
            block->d = fp32_to_fp16(scale_dist(rng_));
            for (int i = 0; i < 16; ++i)
            {
                uint8_t low = index_dist(rng_);
                uint8_t high = index_dist(rng_);
                block->qs[i] = (high << 4) | low;
            }
        }

        return std::make_unique<IQ4_NLTensor>(shape, raw_data);
    }

    /**
     * @brief Compute reference GEMM using decoded FP32 values
     *
     * C = alpha * A @ B^T + beta * C
     */
    void referenceGEMM(const float *A, const float *B, float *C,
                       int m, int n, int k,
                       float alpha = 1.0f, float beta = 0.0f)
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
                C[i * n + j] = alpha * sum + beta * C[i * n + j];
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
                std::cerr << "Mismatch at index " << i << ": expected " << A[i]
                          << ", got " << B[i] << " (diff=" << diff << ", rel=" << rel_error << ")" << std::endl;
                return false;
            }
        }
        return true;
    }

    std::mt19937 rng_;
};

// =============================================================================
// Basic Tensor Properties Tests
// =============================================================================

/**
 * Test basic tensor construction and shape
 */
TEST_F(Test__IQ4_NLTensor, Construction)
{
    auto tensor = createSimpleTensor();

    ASSERT_NE(tensor, nullptr);

    const auto &shape = tensor->shape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2);  // rows
    EXPECT_EQ(shape[1], 32); // cols

    EXPECT_EQ(tensor->size(), 64); // 2 × 32
    EXPECT_EQ(tensor->logical_k(), 32);
    EXPECT_EQ(tensor->padded_k(), 32);
}

/**
 * Test tensor with non-block-aligned dimensions
 */
TEST_F(Test__IQ4_NLTensor, NonAlignedDimensions)
{
    std::vector<size_t> shape = {3, 50}; // 50 cols → 2 blocks per row (32 + 18 padded to 64)
    size_t blocks_per_row = 2;           // (50 + 31) / 32 = 2
    size_t total_blocks = 3 * 2;
    std::vector<uint8_t> raw_data(total_blocks * 18, 0);

    auto tensor = std::make_unique<IQ4_NLTensor>(shape, raw_data);

    EXPECT_EQ(tensor->logical_k(), 50);
    EXPECT_EQ(tensor->padded_k(), 64); // 2 blocks × 32
}

/**
 * Test invalid construction (wrong dimensions)
 */
TEST_F(Test__IQ4_NLTensor, InvalidConstruction)
{
    // 1D tensor (should fail)
    std::vector<size_t> shape1d = {32};
    std::vector<uint8_t> raw_data(18, 0);

    EXPECT_THROW({ IQ4_NLTensor tensor(shape1d, raw_data); }, std::invalid_argument);

    // Mismatched data size
    std::vector<size_t> shape2d = {2, 32};
    std::vector<uint8_t> wrong_size(10, 0); // Should be 2 * 18 = 36 bytes

    EXPECT_THROW({ IQ4_NLTensor tensor(shape2d, wrong_size); }, std::invalid_argument);
}

// =============================================================================
// Decode Tests
// =============================================================================

/**
 * Test basic decode operation
 */
TEST_F(Test__IQ4_NLTensor, BasicDecode)
{
    auto tensor = createSimpleTensor();

    std::vector<float> decoded(64);
    tensor->to_fp32(decoded.data());

    // Row 0: scale=1.0, all indices=0 → all values = 1.0 * kvalues_iq4nl[0]
    // Row 1: scale=2.0, all indices=1 → all values = 2.0 * kvalues_iq4nl[1]

    // kvalues_iq4nl is defined in IQQuantTables.h as static constexpr float
    float expected_row0 = 1.0f * kvalues_iq4nl[0];
    float expected_row1 = 2.0f * kvalues_iq4nl[1];

    // Verify row 0
    for (int i = 0; i < 32; ++i)
    {
        EXPECT_NEAR(decoded[i], expected_row0, 1e-4f) << "Row 0, col " << i;
    }

    // Verify row 1
    for (int i = 32; i < 64; ++i)
    {
        EXPECT_NEAR(decoded[i], expected_row1, 1e-4f) << "Row 1, col " << i;
    }
}

// =============================================================================
// GEMM Tests
// =============================================================================

/**
 * Test simple 1×1 GEMM (single element result)
 */
TEST_F(Test__IQ4_NLTensor, GEMM_1x1)
{
    auto tensor = createSimpleTensor(); // 2×32 weight matrix

    // Create GEMM kernel
    auto gemm = tensor->createGemm();
    ASSERT_NE(gemm, nullptr);

    // A: [1, 32] - single row of all ones
    std::vector<float> A(32, 1.0f);

    // C: [1, 2] - result matrix (1 row, 2 output features)
    std::vector<float> C(2, 0.0f);

    // Multiply: C = A @ tensor^T
    // A is [1, 32], tensor is [2, 32], so C is [1, 2]
    bool success = gemm->multiply(A.data(), C.data(), 1, 2, 32, true);
    ASSERT_TRUE(success);

    // Compute expected result manually
    std::vector<float> B_decoded(64);
    tensor->to_fp32(B_decoded.data());

    std::vector<float> C_expected(2, 0.0f);
    referenceGEMM(A.data(), B_decoded.data(), C_expected.data(), 1, 2, 32);

    // Compare results
    EXPECT_TRUE(matricesEqual(C_expected.data(), C.data(), 2));
}

/**
 * Test small batch GEMM (m=4, cache-blocked algorithm)
 */
TEST_F(Test__IQ4_NLTensor, GEMM_SmallBatch)
{
    auto tensor = createRandomTensor(8, 64); // 8 output features, 64 input features

    auto gemm = tensor->createGemm();

    // A: [4, 64] - 4 input sequences
    std::vector<float> A(4 * 64);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &v : A)
        v = dist(rng_);

    // C: [4, 8] - result
    std::vector<float> C(4 * 8, 0.0f);

    bool success = gemm->multiply(A.data(), C.data(), 4, 8, 64, true);
    ASSERT_TRUE(success);

    // Reference computation
    std::vector<float> B_decoded(8 * 64);
    tensor->to_fp32(B_decoded.data());

    std::vector<float> C_expected(4 * 8, 0.0f);
    referenceGEMM(A.data(), B_decoded.data(), C_expected.data(), 4, 8, 64);

    // Allow slightly higher tolerance for accumulated floating point errors
    EXPECT_TRUE(matricesEqual(C_expected.data(), C.data(), 4 * 8, 1e-3f));
}

/**
 * Test medium batch GEMM (m=16, boundary case)
 */
TEST_F(Test__IQ4_NLTensor, GEMM_MediumBatch)
{
    auto tensor = createRandomTensor(16, 128); // 16 output features, 128 input features

    auto gemm = tensor->createGemm();

    // A: [16, 128]
    std::vector<float> A(16 * 128);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto &v : A)
        v = dist(rng_);

    // C: [16, 16]
    std::vector<float> C(16 * 16, 0.0f);

    bool success = gemm->multiply(A.data(), C.data(), 16, 16, 128, true);
    ASSERT_TRUE(success);

    // Reference
    std::vector<float> B_decoded(16 * 128);
    tensor->to_fp32(B_decoded.data());

    std::vector<float> C_expected(16 * 16, 0.0f);
    referenceGEMM(A.data(), B_decoded.data(), C_expected.data(), 16, 16, 128);

    EXPECT_TRUE(matricesEqual(C_expected.data(), C.data(), 16 * 16, 1e-3f));
}

/**
 * Test large batch GEMM (m=32, row-wise algorithm)
 */
TEST_F(Test__IQ4_NLTensor, GEMM_LargeBatch)
{
    auto tensor = createRandomTensor(32, 256); // 32 output features, 256 input features

    auto gemm = tensor->createGemm();

    // A: [32, 256]
    std::vector<float> A(32 * 256);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    for (auto &v : A)
        v = dist(rng_);

    // C: [32, 32]
    std::vector<float> C(32 * 32, 0.0f);

    bool success = gemm->multiply(A.data(), C.data(), 32, 32, 256, true);
    ASSERT_TRUE(success);

    // Reference
    std::vector<float> B_decoded(32 * 256);
    tensor->to_fp32(B_decoded.data());

    std::vector<float> C_expected(32 * 32, 0.0f);
    referenceGEMM(A.data(), B_decoded.data(), C_expected.data(), 32, 32, 256);

    EXPECT_TRUE(matricesEqual(C_expected.data(), C.data(), 32 * 32, 1e-3f));
}

/**
 * Test GEMM with alpha and beta parameters
 */
TEST_F(Test__IQ4_NLTensor, GEMM_AlphaBeta)
{
    auto tensor = createSimpleTensor(); // 2×32

    auto gemm = tensor->createGemm();

    // A: [2, 32]
    std::vector<float> A(64, 0.5f);

    // C: [2, 2] - pre-initialized with non-zero values
    std::vector<float> C = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> C_backup = C; // Save for reference

    float alpha = 2.0f;
    float beta = 0.5f;

    bool success = gemm->multiply(A.data(), C.data(), 2, 2, 32, true, alpha, beta);
    ASSERT_TRUE(success);

    // Reference
    std::vector<float> B_decoded(64);
    tensor->to_fp32(B_decoded.data());

    std::vector<float> C_expected = C_backup;
    referenceGEMM(A.data(), B_decoded.data(), C_expected.data(), 2, 2, 32, alpha, beta);

    EXPECT_TRUE(matricesEqual(C_expected.data(), C.data(), 4, 1e-3f));
}

/**
 * Test GEMM with non-block-aligned K dimension
 */
TEST_F(Test__IQ4_NLTensor, GEMM_NonAlignedK)
{
    auto tensor = createRandomTensor(4, 50); // K=50, not multiple of 32

    auto gemm = tensor->createGemm();

    // A: [3, 50]
    std::vector<float> A(3 * 50);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &v : A)
        v = dist(rng_);

    // C: [3, 4]
    std::vector<float> C(3 * 4, 0.0f);

    bool success = gemm->multiply(A.data(), C.data(), 3, 4, 50, true);
    ASSERT_TRUE(success);

    // Reference
    std::vector<float> B_decoded(4 * 50);
    tensor->to_fp32(B_decoded.data());

    std::vector<float> C_expected(3 * 4, 0.0f);
    referenceGEMM(A.data(), B_decoded.data(), C_expected.data(), 3, 4, 50);

    EXPECT_TRUE(matricesEqual(C_expected.data(), C.data(), 3 * 4, 1e-3f));
}

/**
 * Test GEMM dimension mismatch detection
 */
TEST_F(Test__IQ4_NLTensor, GEMM_DimensionMismatch)
{
    auto tensor = createRandomTensor(4, 64);
    auto gemm = tensor->createGemm();

    std::vector<float> A(32, 1.0f);
    std::vector<float> C(8, 0.0f);

    // Wrong K dimension (should be 64, not 32)
    bool success = gemm->multiply(A.data(), C.data(), 1, 4, 32, true);
    EXPECT_FALSE(success); // Should fail
}

/**
 * Test numerical stability with very small/large scales
 */
TEST_F(Test__IQ4_NLTensor, GEMM_NumericalStability)
{
    std::vector<size_t> shape = {4, 64};
    size_t blocks_per_row = 2;
    std::vector<uint8_t> raw_data(4 * 2 * 18);

    // Create blocks with extreme scales
    for (size_t i = 0; i < 8; ++i)
    {
        IQ4_NLBlock *block = reinterpret_cast<IQ4_NLBlock *>(raw_data.data() + i * 18);

        // Alternate between very small and very large scales
        float scale = (i % 2 == 0) ? 0.001f : 10.0f;
        block->d = fp32_to_fp16(scale);

        for (int j = 0; j < 16; ++j)
        {
            block->qs[j] = 0x55; // Mix of indices
        }
    }

    auto tensor = std::make_unique<IQ4_NLTensor>(shape, raw_data);
    auto gemm = tensor->createGemm();

    std::vector<float> A(2 * 64, 0.1f);
    std::vector<float> C(2 * 4, 0.0f);

    bool success = gemm->multiply(A.data(), C.data(), 2, 4, 64, true);
    ASSERT_TRUE(success);

    // Verify no NaN or Inf values
    for (size_t i = 0; i < C.size(); ++i)
    {
        EXPECT_FALSE(std::isnan(C[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(C[i])) << "Inf at index " << i;
    }
}

/**
 * @brief Test decode_to_bf16 method
 *
 * Verifies that decoding to BF16 produces reasonable results and can be
 * converted back to FP32 without major precision loss.
 */
TEST_F(Test__IQ4_NLTensor, DecodeToBF16)
{
    // Create a small IQ4_NL tensor (2×64 = 128 elements = 4 blocks)
    std::vector<size_t> shape = {2, 64};
    size_t blocks_per_row = 2;                        // 64 / 32 = 2 blocks per row
    size_t total_blocks = 2 * 2;                      // 2 rows × 2 blocks
    std::vector<uint8_t> raw_data(total_blocks * 18); // 18 bytes per IQ4_NL block

    // Initialize blocks with test data
    IQ4_NLBlock *blocks = reinterpret_cast<IQ4_NLBlock *>(raw_data.data());
    for (size_t i = 0; i < total_blocks; ++i)
    {
        blocks[i].d = fp32_to_fp16(1.0f); // Scale = 1.0
        for (int j = 0; j < 16; ++j)
        {
            blocks[i].qs[j] = (i * 16 + j) % 256; // Varying indices
        }
    }

    auto tensor = std::make_unique<IQ4_NLTensor>(shape, raw_data);

    // Decode to FP32 (ground truth)
    std::vector<float> decoded_fp32(2 * 64);
    tensor->to_fp32(decoded_fp32.data());

    // Decode to BF16
    std::vector<uint16_t> decoded_bf16(2 * 64);
    tensor->to_bf16(decoded_bf16.data());

    // Convert BF16 back to FP32 for comparison
    std::vector<float> bf16_as_fp32(2 * 64);
    for (size_t i = 0; i < 2 * 64; ++i)
    {
        // BF16->FP32: left shift by 16 bits
        uint32_t fp32_bits = static_cast<uint32_t>(decoded_bf16[i]) << 16;
        std::memcpy(&bf16_as_fp32[i], &fp32_bits, sizeof(float));
    }

    // Compare: BF16 should be close to FP32 (within BF16 precision)
    // BF16 has 7 mantissa bits vs FP32's 23, so expect ~3 decimal digits of precision
    float max_rel_error = 0.0f;
    for (size_t i = 0; i < 2 * 64; ++i)
    {
        float fp32_val = decoded_fp32[i];
        float bf16_val = bf16_as_fp32[i];

        if (std::abs(fp32_val) > 1e-6f)
        {
            float rel_error = std::abs(fp32_val - bf16_val) / std::abs(fp32_val);
            max_rel_error = std::max(max_rel_error, rel_error);
        }
    }

    // BF16 should preserve ~3 decimal digits (2^-7 ≈ 0.008 = 0.8% worst case)
    // Use 1% as tolerance to account for quantization + BF16 truncation
    EXPECT_LT(max_rel_error, 0.01f) << "BF16 precision loss too high: " << max_rel_error;
}

// =============================================================================
// SIMD EQUIVALENCY TESTS
// =============================================================================

/**
 * @brief Test scalar vs AVX2 equivalency for IQ4_NL block decode
 *
 * IQ4_NL uses lookup table quantization (kvalues_iq4nl[16]), so SIMD must
 * produce identical results to scalar when accessing the same lookup values.
 */
#ifdef __AVX2__
TEST_F(Test__IQ4_NLTensor, SIMD_ScalarVsAVX2Equivalency)
{
    // Create a test block with varying nibble indices (0-15)
    IQ4_NLBlock block;
    block.d = fp32_to_fp16(2.5f); // Non-trivial scale

    // Fill with pattern that exercises all 16 lookup table entries
    for (int i = 0; i < 16; ++i)
    {
        uint8_t low_nibble = i % 16;
        uint8_t high_nibble = (15 - i) % 16;
        block.qs[i] = (high_nibble << 4) | low_nibble;
    }

    // Decode with scalar and AVX2
    std::vector<float> scalar_output(32);
    std::vector<float> avx2_output(32);

    IQ4_NLTensor::decodeBlockScalar(block, scalar_output.data());
    IQ4_NLTensor::decodeBlockAVX2(block, avx2_output.data());

    // Compare outputs (should be identical - exact lookup table matches)
    constexpr float TOLERANCE = 1e-6f;
    for (size_t i = 0; i < 32; ++i)
    {
        float diff = std::abs(scalar_output[i] - avx2_output[i]);
        EXPECT_LT(diff, TOLERANCE)
            << "Mismatch at index " << i
            << ": scalar=" << scalar_output[i]
            << ", avx2=" << avx2_output[i]
            << ", diff=" << diff;
    }
}
#endif // __AVX2__

/**
 * @brief Test scalar vs AVX512 equivalency for IQ4_NL block decode
 */
#ifdef __AVX512F__
TEST_F(Test__IQ4_NLTensor, SIMD_ScalarVsAVX512Equivalency)
{
    // Create a test block with varying nibble indices
    IQ4_NLBlock block;
    block.d = fp32_to_fp16(1.75f); // Non-trivial scale

    // Fill with pattern that exercises all 16 lookup table entries
    for (int i = 0; i < 16; ++i)
    {
        uint8_t low_nibble = (i * 7) % 16; // Pseudorandom pattern
        uint8_t high_nibble = (i * 11) % 16;
        block.qs[i] = (high_nibble << 4) | low_nibble;
    }

    // Decode with scalar and AVX512
    std::vector<float> scalar_output(32);
    std::vector<float> avx512_output(32);

    IQ4_NLTensor::decodeBlockScalar(block, scalar_output.data());
    IQ4_NLTensor::decodeBlockAVX512(block, avx512_output.data());

    // Compare outputs
    constexpr float TOLERANCE = 1e-6f;
    for (size_t i = 0; i < 32; ++i)
    {
        float diff = std::abs(scalar_output[i] - avx512_output[i]);
        EXPECT_LT(diff, TOLERANCE)
            << "Mismatch at index " << i
            << ": scalar=" << scalar_output[i]
            << ", avx512=" << avx512_output[i]
            << ", diff=" << diff;
    }
}
#endif // __AVX512F__

/**
 * @brief Test AVX2 vs AVX512 cross-validation
 */
#if defined(__AVX2__) && defined(__AVX512F__)
TEST_F(Test__IQ4_NLTensor, SIMD_AVX2VsAVX512Equivalency)
{
    // Create a test block
    IQ4_NLBlock block;
    block.d = fp32_to_fp16(3.14f);

    // Fill with comprehensive pattern
    for (int i = 0; i < 16; ++i)
    {
        block.qs[i] = ((i * 3) % 16) | (((i * 5) % 16) << 4);
    }

    // Decode with both SIMD implementations
    std::vector<float> avx2_output(32);
    std::vector<float> avx512_output(32);

    IQ4_NLTensor::decodeBlockAVX2(block, avx2_output.data());
    IQ4_NLTensor::decodeBlockAVX512(block, avx512_output.data());

    // Compare outputs
    constexpr float TOLERANCE = 1e-6f;
    for (size_t i = 0; i < 32; ++i)
    {
        float diff = std::abs(avx2_output[i] - avx512_output[i]);
        EXPECT_LT(diff, TOLERANCE)
            << "Mismatch at index " << i
            << ": avx2=" << avx2_output[i]
            << ", avx512=" << avx512_output[i]
            << ", diff=" << diff;
    }
}
#endif // __AVX2__ && __AVX512F__

/**
 * @brief Edge case: All nibbles = 0 (first lookup table entry)
 */
TEST_F(Test__IQ4_NLTensor, SIMD_EdgeCase_AllZeroIndices)
{
    IQ4_NLBlock block;
    block.d = fp32_to_fp16(1.0f);

    // All nibbles = 0 -> all outputs = kvalues_iq4nl[0]
    for (int i = 0; i < 16; ++i)
    {
        block.qs[i] = 0x00;
    }

    std::vector<float> output(32);
    IQ4_NLTensor::decodeBlock(block, output.data());

    // All outputs should be 1.0f * kvalues_iq4nl[0]
    float expected = static_cast<float>(kvalues_iq4nl[0]);

    for (size_t i = 0; i < 32; ++i)
    {
        EXPECT_FLOAT_EQ(output[i], expected)
            << "Index " << i << " should be " << expected;
    }
}

/**
 * @brief Edge case: All nibbles = 15 (last lookup table entry)
 */
TEST_F(Test__IQ4_NLTensor, SIMD_EdgeCase_AllMaxIndices)
{
    IQ4_NLBlock block;
    block.d = fp32_to_fp16(2.0f);

    // All nibbles = 15 -> all outputs = 2.0f * kvalues_iq4nl[15]
    for (int i = 0; i < 16; ++i)
    {
        block.qs[i] = 0xFF;
    }

    std::vector<float> output(32);
    IQ4_NLTensor::decodeBlock(block, output.data());

    // All outputs should be 2.0f * kvalues_iq4nl[15]
    float expected = 2.0f * static_cast<float>(kvalues_iq4nl[15]);

    for (size_t i = 0; i < 32; ++i)
    {
        EXPECT_FLOAT_EQ(output[i], expected)
            << "Index " << i << " should be " << expected;
    }
}

/**
 * @brief Edge case: Zero scale (all outputs should be 0.0f)
 */
TEST_F(Test__IQ4_NLTensor, SIMD_EdgeCase_ZeroScale)
{
    IQ4_NLBlock block;
    block.d = fp32_to_fp16(0.0f); // Zero scale

    // Random nibble indices (shouldn't matter with zero scale)
    for (int i = 0; i < 16; ++i)
    {
        block.qs[i] = (i * 17) % 256;
    }

    std::vector<float> output(32);
    IQ4_NLTensor::decodeBlock(block, output.data());

    // All outputs should be 0.0f
    for (size_t i = 0; i < 32; ++i)
    {
        EXPECT_FLOAT_EQ(output[i], 0.0f)
            << "Index " << i << " should be 0.0f with zero scale";
    }
}

/**
 * @brief Edge case: Random nibbles across full range [0, 15]
 */
TEST_F(Test__IQ4_NLTensor, SIMD_EdgeCase_RandomNibbles)
{
    IQ4_NLBlock block;
    block.d = fp32_to_fp16(1.5f);

    // Random pattern covering all lookup table entries
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, 15);

    for (int i = 0; i < 16; ++i)
    {
        uint8_t low_nibble = dist(rng);
        uint8_t high_nibble = dist(rng);
        block.qs[i] = (high_nibble << 4) | low_nibble;
    }

    // Decode with scalar (reference) and main path
    std::vector<float> scalar_output(32);
    std::vector<float> main_output(32);

    IQ4_NLTensor::decodeBlockScalar(block, scalar_output.data());
    IQ4_NLTensor::decodeBlock(block, main_output.data());

    // Should match exactly
    constexpr float TOLERANCE = 1e-6f;
    for (size_t i = 0; i < 32; ++i)
    {
        float diff = std::abs(scalar_output[i] - main_output[i]);
        EXPECT_LT(diff, TOLERANCE)
            << "Mismatch at index " << i
            << ": scalar=" << scalar_output[i]
            << ", main=" << main_output[i];
    }
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
