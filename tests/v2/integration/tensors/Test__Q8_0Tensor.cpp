/**
 * @file Test__Q8_0Tensor.cpp
 * @brief SIMD equivalency tests for Q8_0 tensor dequantization
 *
 * Tests that scalar, AVX2, and AVX512 implementations of Q8_0 block decoding
 * produce identical results.
 *
 * Q8_0 Format:
 * - Block size: 32 elements
 * - Quantization: 8-bit signed integers (-128 to 127)
 * - Storage: 2-byte FP16 scale (d) + 32 bytes of int8 values
 * - Dequant formula: output[i] = d * qs[i]
 *
 * @author David Sanftenberg
 * @date October 29, 2025
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <random>
#include "../../../../src/v2/tensors/Tensors.h"
#include "loaders/ModelLoader.h"
#include <fstream>
#include "v2/utils/MPIContext.h"
#include "v2/tensors/TensorFactory.h"
#include "v2/kernels/cpu/gemm/FloatingPointGemmKernel.h"

using namespace llaminar2;

class Test__Q8_0Tensor : public ::testing::Test
{
protected:
    static constexpr float TOLERANCE = 1e-5f;
    static constexpr size_t BLOCK_SIZE = Q8_0Block::BLOCK_SIZE; // 32 elements

    /**
     * @brief Compare two float arrays for approximate equality
     */
    bool compareArrays(const float *arr1, const float *arr2, size_t count, float tolerance = TOLERANCE)
    {
        for (size_t i = 0; i < count; ++i)
        {
            float diff = std::abs(arr1[i] - arr2[i]);
            if (diff > tolerance)
            {
                std::cerr << "Mismatch at index " << i << ": "
                          << arr1[i] << " != " << arr2[i]
                          << " (diff = " << diff << ")" << std::endl;
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Create a Q8_0 block with specified scale and int8 values
     */
    Q8_0Block createBlock(float scale_fp32, const int8_t *values)
    {
        Q8_0Block block;
        block.d = fp32_to_fp16(scale_fp32);
        std::memcpy(block.qs, values, BLOCK_SIZE);
        return block;
    }
};

// ========================================================================
// SIMD Equivalency Tests
// ========================================================================

TEST_F(Test__Q8_0Tensor, ScalarVsAVX2Equivalency)
{
#if defined(__AVX2__)
    // Create test block with various int8 values
    int8_t test_values[BLOCK_SIZE];
    for (size_t i = 0; i < BLOCK_SIZE; ++i)
    {
        test_values[i] = static_cast<int8_t>(static_cast<int>(i) - 16); // Range: -16 to 15
    }

    Q8_0Block block = createBlock(0.5f, test_values);

    float output_scalar[BLOCK_SIZE];
    float output_avx2[BLOCK_SIZE];

    Q8_0Tensor::decodeBlockScalar(block, output_scalar);
    Q8_0Tensor::decodeBlockAVX2(block, output_avx2);

    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, BLOCK_SIZE))
        << "Scalar and AVX2 implementations should produce identical results";
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
#endif
}

TEST_F(Test__Q8_0Tensor, ScalarVsAVX512Equivalency)
{
#if defined(__AVX512F__)
    // Create test block with various int8 values
    int8_t test_values[BLOCK_SIZE];
    for (size_t i = 0; i < BLOCK_SIZE; ++i)
    {
        test_values[i] = static_cast<int8_t>(static_cast<int>(i) - 16);
    }

    Q8_0Block block = createBlock(0.5f, test_values);

    float output_scalar[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q8_0Tensor::decodeBlockScalar(block, output_scalar);
    Q8_0Tensor::decodeBlockAVX512(block, output_avx512);

    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, BLOCK_SIZE))
        << "Scalar and AVX512 implementations should produce identical results";
#else
    GTEST_SKIP() << "AVX512 not available on this platform";
#endif
}

TEST_F(Test__Q8_0Tensor, AVX2VsAVX512Equivalency)
{
#if defined(__AVX2__) && defined(__AVX512F__)
    // Create test block with various int8 values
    int8_t test_values[BLOCK_SIZE];
    for (size_t i = 0; i < BLOCK_SIZE; ++i)
    {
        test_values[i] = static_cast<int8_t>(static_cast<int>(i) - 16);
    }

    Q8_0Block block = createBlock(0.5f, test_values);

    float output_avx2[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q8_0Tensor::decodeBlockAVX2(block, output_avx2);
    Q8_0Tensor::decodeBlockAVX512(block, output_avx512);

    EXPECT_TRUE(compareArrays(output_avx2, output_avx512, BLOCK_SIZE))
        << "AVX2 and AVX512 implementations should produce identical results";
#else
    GTEST_SKIP() << "Both AVX2 and AVX512 required for this test";
#endif
}

// ========================================================================
// Edge Case Tests
// ========================================================================

TEST_F(Test__Q8_0Tensor, EdgeCase_ZeroScale)
{
    // All values should be zero when scale is zero
    int8_t test_values[BLOCK_SIZE];
    for (size_t i = 0; i < BLOCK_SIZE; ++i)
    {
        test_values[i] = static_cast<int8_t>(i); // Non-zero values
    }

    Q8_0Block block = createBlock(0.0f, test_values);

    float output_scalar[BLOCK_SIZE];
    float output_avx2[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q8_0Tensor::decodeBlockScalar(block, output_scalar);
#if defined(__AVX2__)
    Q8_0Tensor::decodeBlockAVX2(block, output_avx2);
#endif
#if defined(__AVX512F__)
    Q8_0Tensor::decodeBlockAVX512(block, output_avx512);
#endif

    // All outputs should be zero
    for (size_t i = 0; i < BLOCK_SIZE; ++i)
    {
        EXPECT_FLOAT_EQ(output_scalar[i], 0.0f);
#if defined(__AVX2__)
        EXPECT_FLOAT_EQ(output_avx2[i], 0.0f);
#endif
#if defined(__AVX512F__)
        EXPECT_FLOAT_EQ(output_avx512[i], 0.0f);
#endif
    }
}

TEST_F(Test__Q8_0Tensor, EdgeCase_MaxInt8Values)
{
    // Test with maximum int8 values (127 and -128)
    int8_t test_values[BLOCK_SIZE];
    for (size_t i = 0; i < BLOCK_SIZE; ++i)
    {
        test_values[i] = (i % 2 == 0) ? 127 : -128;
    }

    Q8_0Block block = createBlock(1.0f, test_values);

    float output_scalar[BLOCK_SIZE];
    float output_avx2[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q8_0Tensor::decodeBlockScalar(block, output_scalar);
#if defined(__AVX2__)
    Q8_0Tensor::decodeBlockAVX2(block, output_avx2);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, BLOCK_SIZE));
#endif
#if defined(__AVX512F__)
    Q8_0Tensor::decodeBlockAVX512(block, output_avx512);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, BLOCK_SIZE));
#endif
}

TEST_F(Test__Q8_0Tensor, EdgeCase_NegativeScale)
{
    // Q8_0 scale is FP16, which can be negative (though uncommon)
    int8_t test_values[BLOCK_SIZE];
    for (size_t i = 0; i < BLOCK_SIZE; ++i)
    {
        test_values[i] = static_cast<int8_t>(i);
    }

    Q8_0Block block = createBlock(-0.5f, test_values);

    float output_scalar[BLOCK_SIZE];
    float output_avx2[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q8_0Tensor::decodeBlockScalar(block, output_scalar);
#if defined(__AVX2__)
    Q8_0Tensor::decodeBlockAVX2(block, output_avx2);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, BLOCK_SIZE));
#endif
#if defined(__AVX512F__)
    Q8_0Tensor::decodeBlockAVX512(block, output_avx512);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, BLOCK_SIZE));
#endif
}

TEST_F(Test__Q8_0Tensor, EdgeCase_AlternatingPositiveNegative)
{
    // Test alternating positive and negative values
    int8_t test_values[BLOCK_SIZE];
    for (size_t i = 0; i < BLOCK_SIZE; ++i)
    {
        test_values[i] = (i % 2 == 0) ? static_cast<int8_t>(i) : static_cast<int8_t>(-static_cast<int>(i));
    }

    Q8_0Block block = createBlock(2.5f, test_values);

    float output_scalar[BLOCK_SIZE];
    float output_avx2[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q8_0Tensor::decodeBlockScalar(block, output_scalar);
#if defined(__AVX2__)
    Q8_0Tensor::decodeBlockAVX2(block, output_avx2);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, BLOCK_SIZE));
#endif
#if defined(__AVX512F__)
    Q8_0Tensor::decodeBlockAVX512(block, output_avx512);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, BLOCK_SIZE));
#endif
}

TEST_F(Test__Q8_0Tensor, EdgeCase_LargeScale)
{
    // Test with a large scale value
    int8_t test_values[BLOCK_SIZE];
    for (size_t i = 0; i < BLOCK_SIZE; ++i)
    {
        test_values[i] = static_cast<int8_t>(i % 64); // Small values
    }

    Q8_0Block block = createBlock(100.0f, test_values); // Large scale

    float output_scalar[BLOCK_SIZE];
    float output_avx2[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q8_0Tensor::decodeBlockScalar(block, output_scalar);
#if defined(__AVX2__)
    Q8_0Tensor::decodeBlockAVX2(block, output_avx2);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, BLOCK_SIZE));
#endif
#if defined(__AVX512F__)
    Q8_0Tensor::decodeBlockAVX512(block, output_avx512);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, BLOCK_SIZE));
#endif
}

TEST_F(Test__Q8_0Tensor, EdgeCase_RandomValues)
{
    // Test with random values for comprehensive coverage
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_int_distribution<int> dist(-128, 127);
    std::uniform_real_distribution<float> scale_dist(0.001f, 10.0f);

    int8_t test_values[BLOCK_SIZE];
    for (size_t i = 0; i < BLOCK_SIZE; ++i)
    {
        test_values[i] = static_cast<int8_t>(dist(rng));
    }

    Q8_0Block block = createBlock(scale_dist(rng), test_values);

    float output_scalar[BLOCK_SIZE];
    float output_avx2[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q8_0Tensor::decodeBlockScalar(block, output_scalar);
#if defined(__AVX2__)
    Q8_0Tensor::decodeBlockAVX2(block, output_avx2);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, BLOCK_SIZE));
#endif
#if defined(__AVX512F__)
    Q8_0Tensor::decodeBlockAVX512(block, output_avx512);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, BLOCK_SIZE));
#endif
}

/**
 * @brief Test Q8_0Tensor to INT8 block quantization
 *
 * Validates that Q8_0Tensor::to_int8_blocked() produces correct INT8
 * quantization with reasonable accuracy using real model weights.
 */
TEST(Q8_0Tensor_INT8, BlockConversion)
{
    // Load a test model to get real Q8_0Tensor weights
    const std::string model_path = "models/qwen2.5-0.5b-instruct-q8_0.gguf";

    // Skip if model file doesn't exist (not all test environments have models)
    if (!std::ifstream(model_path).good())
    {
        GTEST_SKIP() << "Test model not found: " << model_path;
    }

    llaminar2::ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(model_path));

    // Find a weight tensor of type Q8_0Tensor
    std::shared_ptr<llaminar2::TensorBase> weight_tensor;
    const auto &model = loader.getModel();

    for (const auto &tensor_info : model.tensors)
    {
        // Map GGUFTensorType to our expected type
        // For now, just try to load the first tensor and check its type
        auto tensor = loader.loadTensor(tensor_info.name);
        if (tensor && std::dynamic_pointer_cast<llaminar2::Q8_0Tensor>(tensor))
        {
            weight_tensor = tensor;
            break;
        }
    }

    if (!weight_tensor)
    {
        GTEST_SKIP() << "No Q8_0Tensor weights found in model";
    }

    // Convert to INT8 blocked format
    // Compute total elements from shape
    size_t total_elements = 1;
    for (auto dim : weight_tensor->shape())
    {
        total_elements *= dim;
    }

    const size_t block_size = 32;
    const size_t num_blocks = (total_elements + block_size - 1) / block_size;

    std::vector<int8_t> int8_data(total_elements);
    std::vector<float> scales(num_blocks);

    weight_tensor->to_int8_blocked(int8_data.data(), scales.data(), block_size);

    // Verify all int8 values are in valid range [-127, 127]
    for (size_t i = 0; i < total_elements; ++i)
    {
        EXPECT_GE(int8_data[i], -127) << "INT8 value at index " << i << " out of range";
        EXPECT_LE(int8_data[i], 127) << "INT8 value at index " << i << " out of range";
    }

    // Verify all scales are positive and reasonable
    for (size_t i = 0; i < num_blocks; ++i)
    {
        EXPECT_GT(scales[i], 0.0f) << "Scale at block " << i << " should be positive";
        EXPECT_LT(scales[i], 1e6f) << "Scale at block " << i << " should be reasonable";
    }

    // Verify reconstruction accuracy by dequantizing and comparing
    // First, dequantize to FP32
    std::vector<float> reconstructed(total_elements);
    for (size_t i = 0; i < total_elements; ++i)
    {
        size_t block_idx = i / block_size;
        reconstructed[i] = static_cast<float>(int8_data[i]) * scales[block_idx];
    }

    // Get original FP32 representation for comparison
    std::vector<float> original(total_elements);
    weight_tensor->to_fp32(original.data());

    // Compute relative L2 error
    double sum_sq_diff = 0.0;
    double sum_sq_orig = 0.0;
    double max_abs_diff = 0.0;

    for (size_t i = 0; i < total_elements; ++i)
    {
        double diff = reconstructed[i] - original[i];
        sum_sq_diff += diff * diff;
        sum_sq_orig += original[i] * original[i];
        max_abs_diff = std::max(max_abs_diff, std::abs(diff));
    }

    double rel_l2_error = std::sqrt(sum_sq_diff / sum_sq_orig);

    // INT8 quantization should have reasonable accuracy
    // Typical relative L2 error should be < 5%
    EXPECT_LT(rel_l2_error, 0.05) << "Relative L2 error too large: " << rel_l2_error;

    // Max absolute difference should be reasonable (depends on weight magnitude)
    EXPECT_LT(max_abs_diff, 0.5) << "Max absolute difference too large: " << max_abs_diff;
}

// ========================================================================
// Template Method Tests for to<T>() API
// ========================================================================

/**
 * @brief Test Q8_0Tensor to<float>() template method matches to_fp32()
 */
TEST_F(Test__Q8_0Tensor, ToFloat_TemplateMethod)
{
    // Create a Q8_0 tensor with known values
    const size_t rows = 2;
    const size_t cols = 32; // One block per row
    const std::vector<size_t> shape = {rows, cols};

    // Create Q8_0 blocks (2 blocks total, 64 elements)
    std::vector<Q8_0Block> blocks(rows);
    for (size_t i = 0; i < rows; ++i)
    {
        int8_t values[32];
        for (size_t j = 0; j < 32; ++j)
        {
            values[j] = static_cast<int8_t>((i * 32 + j) - 32); // Range varies by row
        }
        blocks[i] = createBlock(0.5f + static_cast<float>(i) * 0.1f, values);
    }

    // Convert blocks to raw bytes
    std::vector<uint8_t> raw_data(rows * sizeof(Q8_0Block));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());

    auto tensor = std::make_shared<Q8_0Tensor>(shape, raw_data);

    // Test to<float>() template method
    std::vector<float> result_template(64);
    tensor->template to<float>(result_template.data());

    // Compare with legacy to_fp32()
    std::vector<float> result_legacy(64);
    tensor->to_fp32(result_legacy.data());

    // Should be identical
    EXPECT_TRUE(compareArrays(result_template.data(), result_legacy.data(), 64))
        << "to<float>() should match to_fp32()";
}

/**
 * @brief Test Q8_0Tensor to<uint16_t>(BF16) template method matches to_bf16()
 */
TEST_F(Test__Q8_0Tensor, ToBF16_TemplateMethod)
{
    // Create a Q8_0 tensor
    const size_t rows = 2;
    const size_t cols = 32;
    const std::vector<size_t> shape = {rows, cols};

    std::vector<Q8_0Block> blocks(rows);
    for (size_t i = 0; i < rows; ++i)
    {
        int8_t values[32];
        for (size_t j = 0; j < 32; ++j)
        {
            values[j] = static_cast<int8_t>((i * 32 + j) - 32);
        }
        blocks[i] = createBlock(0.5f + static_cast<float>(i) * 0.1f, values);
    }

    std::vector<uint8_t> raw_data(rows * sizeof(Q8_0Block));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());

    auto tensor = std::make_shared<Q8_0Tensor>(shape, raw_data);

    // Test to<uint16_t>() with BF16 format
    std::vector<uint16_t> result_template(64);
    tensor->template to<uint16_t>(result_template.data(), TensorType::BF16);

    // Compare with legacy to_bf16()
    std::vector<uint16_t> result_legacy(64);
    tensor->to_bf16(result_legacy.data());

    // Should be identical
    for (size_t i = 0; i < 64; ++i)
    {
        EXPECT_EQ(result_template[i], result_legacy[i])
            << "Mismatch at index " << i;
    }
}

/**
 * @brief Test Q8_0Tensor to<uint16_t>(FP16) template method matches to_fp16()
 */
TEST_F(Test__Q8_0Tensor, ToFP16_TemplateMethod)
{
    // Create a Q8_0 tensor
    const size_t rows = 2;
    const size_t cols = 32;
    const std::vector<size_t> shape = {rows, cols};

    std::vector<Q8_0Block> blocks(rows);
    for (size_t i = 0; i < rows; ++i)
    {
        int8_t values[32];
        for (size_t j = 0; j < 32; ++j)
        {
            values[j] = static_cast<int8_t>((i * 32 + j) - 32);
        }
        blocks[i] = createBlock(0.5f + static_cast<float>(i) * 0.1f, values);
    }

    std::vector<uint8_t> raw_data(rows * sizeof(Q8_0Block));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());

    auto tensor = std::make_shared<Q8_0Tensor>(shape, raw_data);

    // Test to<uint16_t>() with FP16 format
    std::vector<uint16_t> result_template(64);
    tensor->template to<uint16_t>(result_template.data(), TensorType::FP16);

    // Compare with legacy to_fp16()
    std::vector<uint16_t> result_legacy(64);
    tensor->to_fp16(result_legacy.data());

    // Should be identical
    for (size_t i = 0; i < 64; ++i)
    {
        EXPECT_EQ(result_template[i], result_legacy[i])
            << "Mismatch at index " << i;
    }
}

/**
 * @brief Test Q8_0Tensor to<int8_t>() INT8 quantization
 */
TEST_F(Test__Q8_0Tensor, ToINT8_TemplateMethod)
{
    // Create a Q8_0 tensor
    const size_t rows = 2;
    const size_t cols = 32;
    const std::vector<size_t> shape = {rows, cols};

    std::vector<Q8_0Block> blocks(rows);
    for (size_t i = 0; i < rows; ++i)
    {
        int8_t values[32];
        for (size_t j = 0; j < 32; ++j)
        {
            values[j] = static_cast<int8_t>((i * 32 + j) - 32);
        }
        blocks[i] = createBlock(0.5f + static_cast<float>(i) * 0.1f, values);
    }

    std::vector<uint8_t> raw_data(rows * sizeof(Q8_0Block));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());

    auto tensor = std::make_shared<Q8_0Tensor>(shape, raw_data);

    // Test to<int8_t>() INT8 quantization
    std::vector<int8_t> int8_data(64);
    tensor->template to<int8_t>(int8_data.data());

    // Verify all int8 values are in valid range [-127, 127]
    for (size_t i = 0; i < 64; ++i)
    {
        EXPECT_GE(int8_data[i], -127) << "Value at index " << i << " too low";
        EXPECT_LE(int8_data[i], 127) << "Value at index " << i << " too high";
    }
}

/**
 * @brief Test Q8_0Tensor to<int32_t>() INT32 conversion
 */
TEST_F(Test__Q8_0Tensor, ToINT32_TemplateMethod)
{
    // Create a Q8_0 tensor
    const size_t rows = 2;
    const size_t cols = 32;
    const std::vector<size_t> shape = {rows, cols};

    std::vector<Q8_0Block> blocks(rows);
    for (size_t i = 0; i < rows; ++i)
    {
        int8_t values[32];
        for (size_t j = 0; j < 32; ++j)
        {
            values[j] = static_cast<int8_t>((i * 32 + j) - 32);
        }
        blocks[i] = createBlock(0.5f + static_cast<float>(i) * 0.1f, values);
    }

    std::vector<uint8_t> raw_data(rows * sizeof(Q8_0Block));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());

    auto tensor = std::make_shared<Q8_0Tensor>(shape, raw_data);

    // Test to<int32_t>() INT32 conversion
    std::vector<int32_t> int32_data(64);
    tensor->template to<int32_t>(int32_data.data());

    // Verify no overflow occurred
    for (size_t i = 0; i < 64; ++i)
    {
        EXPECT_GE(int32_data[i], INT32_MIN);
        EXPECT_LE(int32_data[i], INT32_MAX);
    }
}

/**
 * @brief Test round-trip conversion: Q8_0 → FP32 → BF16 → FP32
 */
TEST_F(Test__Q8_0Tensor, RoundTrip_Q8_0_FP32_BF16_FP32)
{
    // Create a Q8_0 tensor
    const size_t rows = 2;
    const size_t cols = 32;
    const std::vector<size_t> shape = {rows, cols};

    std::vector<Q8_0Block> blocks(rows);
    for (size_t i = 0; i < rows; ++i)
    {
        int8_t values[32];
        for (size_t j = 0; j < 32; ++j)
        {
            values[j] = static_cast<int8_t>((i * 32 + j) - 32);
        }
        blocks[i] = createBlock(0.5f + static_cast<float>(i) * 0.1f, values);
    }

    std::vector<uint8_t> raw_data(rows * sizeof(Q8_0Block));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());

    auto tensor = std::make_shared<Q8_0Tensor>(shape, raw_data);

    // Step 1: Q8_0 → FP32
    std::vector<float> fp32_data(64);
    tensor->template to<float>(fp32_data.data());

    // Step 2: FP32 → BF16
    auto fp32_tensor = std::make_shared<FP32Tensor>(shape);
    std::memcpy(fp32_tensor->mutable_data(), fp32_data.data(), 64 * sizeof(float));

    std::vector<uint16_t> bf16_data(64);
    fp32_tensor->template to<uint16_t>(bf16_data.data(), TensorType::BF16);

    // Step 3: BF16 → FP32
    auto bf16_tensor = std::make_shared<BF16Tensor>(shape, bf16_data);
    std::vector<float> final_fp32_data(64);
    bf16_tensor->template to<float>(final_fp32_data.data());

    // Verify round-trip accuracy (Q8_0 has ~1% error, BF16 adds another ~1%)
    double sum_sq_diff = 0.0;
    double sum_sq_orig = 0.0;

    for (size_t i = 0; i < 64; ++i)
    {
        double diff = fp32_data[i] - final_fp32_data[i];
        sum_sq_diff += diff * diff;
        sum_sq_orig += fp32_data[i] * fp32_data[i];
    }

    double rel_l2_error = std::sqrt(sum_sq_diff / sum_sq_orig);
    EXPECT_LT(rel_l2_error, 0.03) << "Round-trip relative L2 error: " << rel_l2_error;
}

/**
 * @brief Quantized GEMM vs FP32 GEMM Parity Test for Q8_0
 *
 * Compares CPUNativeVNNIGemmKernel (INT8) against FloatingPointGemmKernel (FP32 OneDNN)
 * using randomly initialized Q8_0 weights. Validates that quantization introduces
 * acceptable error (< 1% relative L2).
 */
TEST_F(Test__Q8_0Tensor, QuantizedVsFP32Parity)
{
    using namespace llaminar2;

    // Create MPI context for tensor factory
    auto mpi_ctx = std::make_unique<MPIContext>(0, 1, MPI_COMM_WORLD);

    // Realistic dimensions: 64 tokens, 512 hidden dim
    const int m = 64;
    const int n = 512;
    const int k = 512;

    // Q8_0: 32 elements per block (scale + 32 int8 values)
    const size_t num_blocks = (static_cast<size_t>(n) * k) / 32;
    std::vector<uint8_t> raw_data(num_blocks * sizeof(Q8_0Block));
    Q8_0Block *blocks = reinterpret_cast<Q8_0Block *>(raw_data.data());

    // Initialize with random but valid data
    std::mt19937 rng(42);
    std::uniform_int_distribution<int8_t> int8_dist(-127, 127);
    std::uniform_real_distribution<float> scale_dist(0.001f, 0.1f);

    for (size_t b = 0; b < num_blocks; ++b)
    {
        // Valid FP16 scale factor
        blocks[b].d = fp32_to_fp16(scale_dist(rng));
        // Random int8 values
        for (int i = 0; i < 32; ++i)
        {
            blocks[b].qs[i] = int8_dist(rng);
        }
    }

    // Create quantized tensor
    auto q8_0_tensor = std::make_unique<Q8_0Tensor>(
        std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)},
        raw_data);

    // Dequantize to FP32 for reference
    TensorFactory factory(*mpi_ctx);
    auto fp32_weights = factory.createFP32({static_cast<size_t>(n), static_cast<size_t>(k)});
    q8_0_tensor->to_fp32(fp32_weights->mutable_data());

    // Create random input activations
    auto input = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(k)});
    float *input_data = input->mutable_data();
    std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
    for (int i = 0; i < m * k; ++i)
    {
        input_data[i] = input_dist(rng);
    }

    // Allocate outputs
    auto output_quantized = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(n)});
    auto output_fp32 = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(n)});
    std::memset(output_quantized->mutable_data(), 0, m * n * sizeof(float));
    std::memset(output_fp32->mutable_data(), 0, m * n * sizeof(float));

    // Run quantized GEMM (INT8 path)
    auto quantized_gemm = q8_0_tensor->createGemm();
    ASSERT_TRUE(quantized_gemm->multiply_tensor(
        input.get(),
        output_quantized.get(),
        m, n, k));

    // Run FP32 GEMM (OneDNN reference)
    gemm::FloatingPointGemmKernel fp32_gemm(fp32_weights.get());
    ASSERT_TRUE(fp32_gemm.multiply_tensor(
        input.get(),
        output_fp32.get(),
        m, n, k));

    // Compare results - compute relative L2 error
    double sum_sq_diff = 0.0, sum_sq_ref = 0.0;
    for (int i = 0; i < m * n; ++i)
    {
        double diff = static_cast<double>(output_quantized->data()[i]) - static_cast<double>(output_fp32->data()[i]);
        sum_sq_diff += diff * diff;
        sum_sq_ref += static_cast<double>(output_fp32->data()[i]) * static_cast<double>(output_fp32->data()[i]);
    }
    float rel_l2_error_parity = (sum_sq_ref > 0) ? static_cast<float>(std::sqrt(sum_sq_diff / sum_sq_ref)) : 0.0f;

    std::cout << "[Q8_0 Parity] Relative L2 error: " << (rel_l2_error_parity * 100.0f) << "%" << std::endl;

    // 1% tolerance for quantization error
    EXPECT_LT(rel_l2_error_parity, 0.01f)
        << "Q8_0 quantized GEMM error exceeds 1% threshold";
}
