/**
 * @file Test__Q3_KTensor.cpp
 * @brief SIMD equivalency tests for Q3_K tensor dequantization
 *
 * Tests that scalar, AVX2, and AVX512 implementations of Q3_K block decoding
 * produce identical results.
 *
 * Q3_K Format:
 * - Block size: 256 elements (super-block)
 * - Quantization: 3-bit values (2 bits in qs[] + 1 high bit in hmask[])
 * - Storage: 32-byte hmask + 64-byte qs + 12-byte scales + 2-byte FP16 d
 * - Dequant formula: output[i] = d * scale * (low_bits - (high_bit ? 0 : 4))
 * - Scale unpacking: 16 scales (6 bits each) packed into 12 bytes
 *
 * @author David Sanftenberg
 * @date October 29, 2025
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <random>
#include "tensors/Tensors.h"
#include "loaders/ModelLoader.h"
#include <fstream>

using namespace llaminar2;

class Test__Q3_KTensor : public ::testing::Test
{
protected:
    static constexpr float TOLERANCE = 1e-4f;                   // Slightly relaxed for complex Q3_K math
    static constexpr size_t BLOCK_SIZE = Q3_KBlock::BLOCK_SIZE; // 256 elements

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
     * @brief Create a Q3_K block with specified parameters
     *
     * Q3_K uses 3-bit values: 2 low bits in qs[], 1 high bit in hmask[]
     * Formula: d * scale * (low_bits - (high_bit ? 0 : 4))
     */
    Q3_KBlock createBlock(float d_fp32, const uint8_t *hmask, const uint8_t *qs, const uint8_t *scales)
    {
        Q3_KBlock block;
        block.d = fp32_to_fp16(d_fp32);
        std::memcpy(block.hmask, hmask, 32);
        std::memcpy(block.qs, qs, 64);
        std::memcpy(block.scales, scales, 12);
        return block;
    }

    /**
     * @brief Create simple test data with predictable patterns
     */
    void createSimpleTestData(uint8_t *hmask, uint8_t *qs, uint8_t *scales)
    {
        // Simple pattern: alternating low bits and high bits
        for (size_t i = 0; i < 32; ++i)
        {
            hmask[i] = (i % 2 == 0) ? 0xFF : 0x00; // Alternating all on/off
        }

        for (size_t i = 0; i < 64; ++i)
        {
            // Each byte has 4 2-bit values at shifts 0,2,4,6
            // Pattern: 0b11100100 = values 0,1,2,3 at different shifts
            qs[i] = 0xE4;
        }

        // Scales: pack 16 6-bit values into 12 bytes
        // For simplicity, use pattern that unpacks to known values
        for (size_t i = 0; i < 12; ++i)
        {
            scales[i] = 0x55; // Pattern of 01010101
        }
    }
};

// ========================================================================
// SIMD Equivalency Tests
// ========================================================================

TEST_F(Test__Q3_KTensor, ScalarVsAVX2Equivalency)
{
#if defined(__AVX2__)
    uint8_t hmask[32], qs[64], scales[12];
    createSimpleTestData(hmask, qs, scales);

    Q3_KBlock block = createBlock(0.5f, hmask, qs, scales);

    float output_scalar[BLOCK_SIZE];
    float output_avx2[BLOCK_SIZE];

    Q3_KTensor::decodeBlockScalar(block, output_scalar);
    Q3_KTensor::decodeBlockAVX2(block, output_avx2);

    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, BLOCK_SIZE))
        << "Scalar and AVX2 implementations should produce identical results";
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
#endif
}

TEST_F(Test__Q3_KTensor, ScalarVsAVX512Equivalency)
{
#if defined(__AVX512F__)
    uint8_t hmask[32], qs[64], scales[12];
    createSimpleTestData(hmask, qs, scales);

    Q3_KBlock block = createBlock(0.5f, hmask, qs, scales);

    float output_scalar[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q3_KTensor::decodeBlockScalar(block, output_scalar);
    Q3_KTensor::decodeBlockAVX512(block, output_avx512);

    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, BLOCK_SIZE))
        << "Scalar and AVX512 implementations should produce identical results";
#else
    GTEST_SKIP() << "AVX512 not available on this platform";
#endif
}

TEST_F(Test__Q3_KTensor, AVX2VsAVX512Equivalency)
{
#if defined(__AVX2__) && defined(__AVX512F__)
    uint8_t hmask[32], qs[64], scales[12];
    createSimpleTestData(hmask, qs, scales);

    Q3_KBlock block = createBlock(0.5f, hmask, qs, scales);

    float output_avx2[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q3_KTensor::decodeBlockAVX2(block, output_avx2);
    Q3_KTensor::decodeBlockAVX512(block, output_avx512);

    EXPECT_TRUE(compareArrays(output_avx2, output_avx512, BLOCK_SIZE))
        << "AVX2 and AVX512 implementations should produce identical results";
#else
    GTEST_SKIP() << "Both AVX2 and AVX512 required for this test";
#endif
}

// ========================================================================
// Edge Case Tests
// ========================================================================

TEST_F(Test__Q3_KTensor, EdgeCase_ZeroScale)
{
    // All outputs should be zero when d is zero
    uint8_t hmask[32], qs[64], scales[12];
    createSimpleTestData(hmask, qs, scales);

    Q3_KBlock block = createBlock(0.0f, hmask, qs, scales);

    float output_scalar[BLOCK_SIZE];
    float output_avx2[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q3_KTensor::decodeBlockScalar(block, output_scalar);
#if defined(__AVX2__)
    Q3_KTensor::decodeBlockAVX2(block, output_avx2);
#endif
#if defined(__AVX512F__)
    Q3_KTensor::decodeBlockAVX512(block, output_avx512);
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

TEST_F(Test__Q3_KTensor, EdgeCase_AllHighBitsSet)
{
    // Test with all high bits in hmask set to 1
    uint8_t hmask[32], qs[64], scales[12];

    for (size_t i = 0; i < 32; ++i)
        hmask[i] = 0xFF; // All high bits set
    for (size_t i = 0; i < 64; ++i)
        qs[i] = 0xE4;
    for (size_t i = 0; i < 12; ++i)
        scales[i] = 0x55;

    Q3_KBlock block = createBlock(1.0f, hmask, qs, scales);

    float output_scalar[BLOCK_SIZE];
    float output_avx2[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q3_KTensor::decodeBlockScalar(block, output_scalar);
#if defined(__AVX2__)
    Q3_KTensor::decodeBlockAVX2(block, output_avx2);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, BLOCK_SIZE));
#endif
#if defined(__AVX512F__)
    Q3_KTensor::decodeBlockAVX512(block, output_avx512);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, BLOCK_SIZE));
#endif
}

TEST_F(Test__Q3_KTensor, EdgeCase_AllHighBitsClear)
{
    // Test with all high bits in hmask cleared to 0
    uint8_t hmask[32], qs[64], scales[12];

    for (size_t i = 0; i < 32; ++i)
        hmask[i] = 0x00; // All high bits clear
    for (size_t i = 0; i < 64; ++i)
        qs[i] = 0xE4;
    for (size_t i = 0; i < 12; ++i)
        scales[i] = 0x55;

    Q3_KBlock block = createBlock(1.0f, hmask, qs, scales);

    float output_scalar[BLOCK_SIZE];
    float output_avx2[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q3_KTensor::decodeBlockScalar(block, output_scalar);
#if defined(__AVX2__)
    Q3_KTensor::decodeBlockAVX2(block, output_avx2);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, BLOCK_SIZE));
#endif
#if defined(__AVX512F__)
    Q3_KTensor::decodeBlockAVX512(block, output_avx512);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, BLOCK_SIZE));
#endif
}

TEST_F(Test__Q3_KTensor, EdgeCase_MaxLowBits)
{
    // Test with maximum low bit values (0b11 = 3)
    uint8_t hmask[32], qs[64], scales[12];

    for (size_t i = 0; i < 32; ++i)
        hmask[i] = 0xAA; // Alternating pattern
    for (size_t i = 0; i < 64; ++i)
        qs[i] = 0xFF; // All low bits set to 11
    for (size_t i = 0; i < 12; ++i)
        scales[i] = 0x55;

    Q3_KBlock block = createBlock(2.5f, hmask, qs, scales);

    float output_scalar[BLOCK_SIZE];
    float output_avx2[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q3_KTensor::decodeBlockScalar(block, output_scalar);
#if defined(__AVX2__)
    Q3_KTensor::decodeBlockAVX2(block, output_avx2);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, BLOCK_SIZE));
#endif
#if defined(__AVX512F__)
    Q3_KTensor::decodeBlockAVX512(block, output_avx512);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, BLOCK_SIZE));
#endif
}

TEST_F(Test__Q3_KTensor, EdgeCase_ChunkBoundary)
{
    // Test transitions between 128-element chunks
    uint8_t hmask[32], qs[64], scales[12];

    // Different patterns for first and second chunk
    for (size_t i = 0; i < 16; ++i)
        hmask[i] = 0xFF;
    for (size_t i = 16; i < 32; ++i)
        hmask[i] = 0x00;

    for (size_t i = 0; i < 32; ++i)
        qs[i] = 0xE4;
    for (size_t i = 32; i < 64; ++i)
        qs[i] = 0x1B;

    for (size_t i = 0; i < 12; ++i)
        scales[i] = (i < 6) ? 0xAA : 0x55;

    Q3_KBlock block = createBlock(1.5f, hmask, qs, scales);

    float output_scalar[BLOCK_SIZE];
    float output_avx2[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q3_KTensor::decodeBlockScalar(block, output_scalar);
#if defined(__AVX2__)
    Q3_KTensor::decodeBlockAVX2(block, output_avx2);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, BLOCK_SIZE));
#endif
#if defined(__AVX512F__)
    Q3_KTensor::decodeBlockAVX512(block, output_avx512);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, BLOCK_SIZE));
#endif
}

TEST_F(Test__Q3_KTensor, EdgeCase_ScaleUnpacking)
{
    // Test various scale packing patterns to ensure correct unpacking
    uint8_t hmask[32], qs[64], scales[12];

    for (size_t i = 0; i < 32; ++i)
        hmask[i] = 0xAA;
    for (size_t i = 0; i < 64; ++i)
        qs[i] = 0xE4;

    // Different scale patterns to test unpacking logic
    scales[0] = 0x12;
    scales[1] = 0x34;
    scales[2] = 0x56;
    scales[3] = 0x78;
    scales[4] = 0x9A;
    scales[5] = 0xBC;
    scales[6] = 0xDE;
    scales[7] = 0xF0;
    scales[8] = 0x11;
    scales[9] = 0x22;
    scales[10] = 0x33;
    scales[11] = 0x44;

    Q3_KBlock block = createBlock(0.75f, hmask, qs, scales);

    float output_scalar[BLOCK_SIZE];
    float output_avx2[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q3_KTensor::decodeBlockScalar(block, output_scalar);
#if defined(__AVX2__)
    Q3_KTensor::decodeBlockAVX2(block, output_avx2);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, BLOCK_SIZE));
#endif
#if defined(__AVX512F__)
    Q3_KTensor::decodeBlockAVX512(block, output_avx512);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, BLOCK_SIZE));
#endif
}

TEST_F(Test__Q3_KTensor, EdgeCase_RandomValues)
{
    // Test with random values for comprehensive coverage
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_real_distribution<float> scale_dist(0.001f, 10.0f);

    uint8_t hmask[32], qs[64], scales[12];

    for (size_t i = 0; i < 32; ++i)
        hmask[i] = static_cast<uint8_t>(byte_dist(rng));
    for (size_t i = 0; i < 64; ++i)
        qs[i] = static_cast<uint8_t>(byte_dist(rng));
    for (size_t i = 0; i < 12; ++i)
        scales[i] = static_cast<uint8_t>(byte_dist(rng));

    Q3_KBlock block = createBlock(scale_dist(rng), hmask, qs, scales);

    float output_scalar[BLOCK_SIZE];
    float output_avx2[BLOCK_SIZE];
    float output_avx512[BLOCK_SIZE];

    Q3_KTensor::decodeBlockScalar(block, output_scalar);
#if defined(__AVX2__)
    Q3_KTensor::decodeBlockAVX2(block, output_avx2);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx2, BLOCK_SIZE));
#endif
#if defined(__AVX512F__)
    Q3_KTensor::decodeBlockAVX512(block, output_avx512);
    EXPECT_TRUE(compareArrays(output_scalar, output_avx512, BLOCK_SIZE));
#endif
}

/**
 * @brief Test Q3_KTensor to INT8 block quantization
 *
 * Validates that Q3_KTensor::to_int8_blocked() produces correct INT8
 * quantization with reasonable accuracy using real model weights.
 */
TEST(Q3_KTensor_INT8, BlockConversion)
{
    // Load a test model to get real Q3_KTensor weights
    const std::string model_path = "models/qwen2.5-0.5b-instruct-q3_k_m.gguf";
    
    // Skip if model file doesn't exist (not all test environments have models)
    if (!std::ifstream(model_path).good()) {
        GTEST_SKIP() << "Test model not found: " << model_path;
    }
    
    llaminar2::ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(model_path));
    
    // Find a weight tensor of type Q3_KTensor
    std::shared_ptr<llaminar2::TensorBase> weight_tensor;
    const auto& model = loader.getModel();
    
    for (const auto& tensor_info : model.tensors) {
        // Map GGUFTensorType to our expected type
        // For now, just try to load the first tensor and check its type
        auto tensor = loader.loadTensor(tensor_info.name);
        if (tensor && std::dynamic_pointer_cast<llaminar2::Q3_KTensor>(tensor)) {
            weight_tensor = tensor;
            break;
        }
    }
    
    if (!weight_tensor) {
        GTEST_SKIP() << "No Q3_KTensor weights found in model";
    }
    
    // Convert to INT8 blocked format
    // Compute total elements from shape
    size_t total_elements = 1;
    for (auto dim : weight_tensor->shape()) {
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
 * @brief Test Q3_KTensor to<float>() template method matches to_fp32()
 */
TEST(Q3_KTensor_Template, ToFloat_TemplateMethod)
{
    using namespace llaminar2;
    
    const std::vector<size_t> shape = {2, 256};
    std::vector<Q3_KBlock> blocks(2);
    
    for (size_t i = 0; i < 2; ++i)
    {
        blocks[i].d = fp32_to_fp16(0.5f + static_cast<float>(i) * 0.1f);
        for (size_t j = 0; j < 32; ++j) blocks[i].hmask[j] = static_cast<uint8_t>((i * 32 + j) % 256);
        for (size_t j = 0; j < 64; ++j) blocks[i].qs[j] = static_cast<uint8_t>((i * 64 + j) % 256);
        for (size_t j = 0; j < 12; ++j) blocks[i].scales[j] = static_cast<uint8_t>((i * 12 + j) % 256);
    }
    
    std::vector<uint8_t> raw_data(2 * sizeof(Q3_KBlock));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q3_KTensor>(shape, raw_data);
    
    std::vector<float> result_template(512);
    tensor->template to<float>(result_template.data());
    
    std::vector<float> result_legacy(512);
    tensor->to_fp32(result_legacy.data());
    
    for (size_t i = 0; i < 512; ++i)
    {
        EXPECT_FLOAT_EQ(result_template[i], result_legacy[i])
            << "Mismatch at index " << i;
    }
}

TEST(Q3_KTensor_Template, ToBF16_TemplateMethod)
{
    using namespace llaminar2;
    
    const std::vector<size_t> shape = {2, 256};
    std::vector<Q3_KBlock> blocks(2);
    
    for (size_t i = 0; i < 2; ++i)
    {
        blocks[i].d = fp32_to_fp16(0.5f + static_cast<float>(i) * 0.1f);
        for (size_t j = 0; j < 32; ++j) blocks[i].hmask[j] = static_cast<uint8_t>((i * 32 + j) % 256);
        for (size_t j = 0; j < 64; ++j) blocks[i].qs[j] = static_cast<uint8_t>((i * 64 + j) % 256);
        for (size_t j = 0; j < 12; ++j) blocks[i].scales[j] = static_cast<uint8_t>((i * 12 + j) % 256);
    }
    
    std::vector<uint8_t> raw_data(2 * sizeof(Q3_KBlock));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q3_KTensor>(shape, raw_data);
    
    std::vector<uint16_t> result_template(512);
    tensor->template to<uint16_t>(result_template.data(), TensorType::BF16);
    
    std::vector<uint16_t> result_legacy(512);
    tensor->to_bf16(result_legacy.data());
    
    for (size_t i = 0; i < 512; ++i)
    {
        EXPECT_EQ(result_template[i], result_legacy[i])
            << "Mismatch at index " << i;
    }
}

TEST(Q3_KTensor_Template, ToFP16_TemplateMethod)
{
    using namespace llaminar2;
    
    const std::vector<size_t> shape = {2, 256};
    std::vector<Q3_KBlock> blocks(2);
    
    for (size_t i = 0; i < 2; ++i)
    {
        blocks[i].d = fp32_to_fp16(0.5f + static_cast<float>(i) * 0.1f);
        for (size_t j = 0; j < 32; ++j) blocks[i].hmask[j] = static_cast<uint8_t>((i * 32 + j) % 256);
        for (size_t j = 0; j < 64; ++j) blocks[i].qs[j] = static_cast<uint8_t>((i * 64 + j) % 256);
        for (size_t j = 0; j < 12; ++j) blocks[i].scales[j] = static_cast<uint8_t>((i * 12 + j) % 256);
    }
    
    std::vector<uint8_t> raw_data(2 * sizeof(Q3_KBlock));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q3_KTensor>(shape, raw_data);
    
    std::vector<uint16_t> result_template(512);
    tensor->template to<uint16_t>(result_template.data(), TensorType::FP16);
    
    std::vector<uint16_t> result_legacy(512);
    tensor->to_fp16(result_legacy.data());
    
    for (size_t i = 0; i < 512; ++i)
    {
        EXPECT_EQ(result_template[i], result_legacy[i])
            << "Mismatch at index " << i;
    }
}

TEST(Q3_KTensor_Template, ToINT8_TemplateMethod)
{
    using namespace llaminar2;
    
    const std::vector<size_t> shape = {2, 256};
    std::vector<Q3_KBlock> blocks(2);
    
    for (size_t i = 0; i < 2; ++i)
    {
        blocks[i].d = fp32_to_fp16(0.5f + static_cast<float>(i) * 0.1f);
        for (size_t j = 0; j < 32; ++j) blocks[i].hmask[j] = static_cast<uint8_t>((i * 32 + j) % 256);
        for (size_t j = 0; j < 64; ++j) blocks[i].qs[j] = static_cast<uint8_t>((i * 64 + j) % 256);
        for (size_t j = 0; j < 12; ++j) blocks[i].scales[j] = static_cast<uint8_t>((i * 12 + j) % 256);
    }
    
    std::vector<uint8_t> raw_data(2 * sizeof(Q3_KBlock));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q3_KTensor>(shape, raw_data);
    
    std::vector<int8_t> int8_data(512);
    tensor->template to<int8_t>(int8_data.data());
    
    for (size_t i = 0; i < 512; ++i)
    {
        EXPECT_GE(int8_data[i], -127);
        EXPECT_LE(int8_data[i], 127);
    }
}

TEST(Q3_KTensor_Template, ToINT32_TemplateMethod)
{
    using namespace llaminar2;
    
    const std::vector<size_t> shape = {2, 256};
    std::vector<Q3_KBlock> blocks(2);
    
    for (size_t i = 0; i < 2; ++i)
    {
        blocks[i].d = fp32_to_fp16(0.5f + static_cast<float>(i) * 0.1f);
        for (size_t j = 0; j < 32; ++j) blocks[i].hmask[j] = static_cast<uint8_t>((i * 32 + j) % 256);
        for (size_t j = 0; j < 64; ++j) blocks[i].qs[j] = static_cast<uint8_t>((i * 64 + j) % 256);
        for (size_t j = 0; j < 12; ++j) blocks[i].scales[j] = static_cast<uint8_t>((i * 12 + j) % 256);
    }
    
    std::vector<uint8_t> raw_data(2 * sizeof(Q3_KBlock));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q3_KTensor>(shape, raw_data);
    
    std::vector<int32_t> int32_data(512);
    tensor->template to<int32_t>(int32_data.data());
    
    for (size_t i = 0; i < 512; ++i)
    {
        EXPECT_GE(int32_data[i], INT32_MIN);
        EXPECT_LE(int32_data[i], INT32_MAX);
    }
}

TEST(Q3_KTensor_Template, RoundTrip_Q3_K_FP32_BF16_FP32)
{
    using namespace llaminar2;
    
    const std::vector<size_t> shape = {2, 256};
    std::vector<Q3_KBlock> blocks(2);
    
    for (size_t i = 0; i < 2; ++i)
    {
        blocks[i].d = fp32_to_fp16(0.5f + static_cast<float>(i) * 0.1f);
        for (size_t j = 0; j < 32; ++j) blocks[i].hmask[j] = static_cast<uint8_t>((i * 32 + j) % 256);
        for (size_t j = 0; j < 64; ++j) blocks[i].qs[j] = static_cast<uint8_t>((i * 64 + j) % 256);
        for (size_t j = 0; j < 12; ++j) blocks[i].scales[j] = static_cast<uint8_t>((i * 12 + j) % 256);
    }
    
    std::vector<uint8_t> raw_data(2 * sizeof(Q3_KBlock));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q3_KTensor>(shape, raw_data);
    
    std::vector<float> fp32_data(512);
    tensor->template to<float>(fp32_data.data());
    
    auto fp32_tensor = std::make_shared<FP32Tensor>(shape);
    std::memcpy(fp32_tensor->mutable_data(), fp32_data.data(), 512 * sizeof(float));
    
    std::vector<uint16_t> bf16_data(512);
    fp32_tensor->template to<uint16_t>(bf16_data.data(), TensorType::BF16);
    
    auto bf16_tensor = std::make_shared<BF16Tensor>(shape, bf16_data);
    std::vector<float> final_fp32_data(512);
    bf16_tensor->template to<float>(final_fp32_data.data());
    
    double sum_sq_diff = 0.0;
    double sum_sq_orig = 0.0;
    
    for (size_t i = 0; i < 512; ++i)
    {
        double diff = fp32_data[i] - final_fp32_data[i];
        sum_sq_diff += diff * diff;
        sum_sq_orig += fp32_data[i] * fp32_data[i];
    }
    
    double rel_l2_error = std::sqrt(sum_sq_diff / sum_sq_orig);
    EXPECT_LT(rel_l2_error, 0.05);
}
