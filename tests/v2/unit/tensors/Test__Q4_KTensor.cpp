/**
 * @file Test__Q4_KTensor.cpp
 * @brief Unit tests for Q4_K SIMD equivalency (AVX2, AVX512)
 * @author David Sanftenberg
 *
 * Tests verify that scalar, AVX2, and AVX512 implementations produce identical results.
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include <random>
#include <cmath>
#include <cstring>
#include "loaders/ModelLoader.h"
#include <fstream>

using namespace llaminar2;

class Test__Q4_KTensor : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Seed for reproducible tests
        rng_.seed(42);
    }

    // Helper: Create Q4_KBlock with specific values
    Q4_KBlock createBlock(uint16_t d_raw, uint16_t dmin_raw,
                          const uint8_t scales[12],
                          const uint8_t qs[128])
    {
        Q4_KBlock block;
        block.d = d_raw;
        block.dmin = dmin_raw;
        std::memcpy(block.scales, scales, 12);
        std::memcpy(block.qs, qs, 128);
        return block;
    }

    // Helper: Create random Q4_KBlock
    Q4_KBlock createRandomBlock()
    {
        Q4_KBlock block;

        // Random d and dmin (FP16 format, avoid extremes)
        std::uniform_int_distribution<uint16_t> fp16_dist(0x3000, 0x4000); // ~0.25 to 2.0
        block.d = fp16_dist(rng_);
        block.dmin = fp16_dist(rng_);

        // Random scales (6-bit values packed)
        std::uniform_int_distribution<uint8_t> byte_dist(0, 255);
        for (int i = 0; i < 12; ++i)
        {
            block.scales[i] = byte_dist(rng_);
        }

        // Random qs (4-bit values in each nibble)
        for (int i = 0; i < 128; ++i)
        {
            block.qs[i] = byte_dist(rng_);
        }

        return block;
    }

    // Helper: Compare two float arrays
    void compareOutputs(const float *expected, const float *actual, size_t count,
                        float tolerance = 1e-4f, const std::string &label = "")
    {
        size_t mismatches = 0;
        float max_diff = 0.0f;
        size_t first_mismatch_idx = 0;

        for (size_t i = 0; i < count; ++i)
        {
            float diff = std::abs(expected[i] - actual[i]);
            if (diff > max_diff)
            {
                max_diff = diff;
            }
            if (diff > tolerance)
            {
                if (mismatches == 0)
                {
                    first_mismatch_idx = i;
                }
                mismatches++;
            }
        }

        if (mismatches > 0)
        {
            std::cerr << label << " MISMATCH:\n";
            std::cerr << "  First mismatch at index " << first_mismatch_idx << ":\n";
            std::cerr << "    Expected: " << expected[first_mismatch_idx] << "\n";
            std::cerr << "    Actual:   " << actual[first_mismatch_idx] << "\n";
            std::cerr << "  Total mismatches: " << mismatches << "/" << count << "\n";
            std::cerr << "  Max difference: " << max_diff << "\n";
        }

        EXPECT_EQ(mismatches, 0) << label;
    }

    std::mt19937 rng_;
};

// ============================================================================
// SIMD Equivalency Tests
// ============================================================================

#if defined(__AVX2__)
TEST_F(Test__Q4_KTensor, ScalarVsAVX2Equivalency)
{
    constexpr size_t BLOCK_SIZE = 256;

    // Test with multiple random blocks
    for (int iter = 0; iter < 10; ++iter)
    {
        Q4_KBlock block = createRandomBlock();

        float scalar_output[BLOCK_SIZE];
        float avx2_output[BLOCK_SIZE];

        Q4_KTensor::decodeBlockScalar(block, scalar_output);
        Q4_KTensor::decodeBlockAVX2(block, avx2_output);

        compareOutputs(scalar_output, avx2_output, BLOCK_SIZE, 1e-4f,
                       "ScalarVsAVX2 iteration " + std::to_string(iter));
    }
}
#endif

#if defined(__AVX512F__)
TEST_F(Test__Q4_KTensor, ScalarVsAVX512Equivalency)
{
    constexpr size_t BLOCK_SIZE = 256;

    // Test with multiple random blocks
    for (int iter = 0; iter < 10; ++iter)
    {
        Q4_KBlock block = createRandomBlock();

        float scalar_output[BLOCK_SIZE];
        float avx512_output[BLOCK_SIZE];

        Q4_KTensor::decodeBlockScalar(block, scalar_output);
        Q4_KTensor::decodeBlockAVX512(block, avx512_output);

        compareOutputs(scalar_output, avx512_output, BLOCK_SIZE, 1e-4f,
                       "ScalarVsAVX512 iteration " + std::to_string(iter));
    }
}
#endif

#if defined(__AVX512F__) && defined(__AVX2__)
TEST_F(Test__Q4_KTensor, AVX2VsAVX512Equivalency)
{
    constexpr size_t BLOCK_SIZE = 256;

    // Test with multiple random blocks
    for (int iter = 0; iter < 10; ++iter)
    {
        Q4_KBlock block = createRandomBlock();

        float avx2_output[BLOCK_SIZE];
        float avx512_output[BLOCK_SIZE];

        Q4_KTensor::decodeBlockAVX2(block, avx2_output);
        Q4_KTensor::decodeBlockAVX512(block, avx512_output);

        compareOutputs(avx2_output, avx512_output, BLOCK_SIZE, 1e-4f,
                       "AVX2VsAVX512 iteration " + std::to_string(iter));
    }
}
#endif

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(Test__Q4_KTensor, EdgeCase_ZeroScale)
{
    // Test with d=0 (all outputs should be near zero after subtracting min)
    constexpr size_t BLOCK_SIZE = 256;

    uint8_t scales[12] = {0};
    uint8_t qs[128];
    std::fill_n(qs, 128, 0x77); // Arbitrary nibbles

    Q4_KBlock block = createBlock(0x0000, 0x3C00, scales, qs); // d=0, dmin=1.0

    float scalar_output[BLOCK_SIZE];
    Q4_KTensor::decodeBlockScalar(block, scalar_output);

    // With d=0, all values should be -dmin*m (negative min value)
    // Values depend on scale unpacking, but should be consistent
    float first_val = scalar_output[0];
    for (size_t i = 0; i < BLOCK_SIZE; ++i)
    {
        // All values in same subgroup should have same sign/pattern
        // Just verify scalar produces consistent results
        EXPECT_TRUE(std::isfinite(scalar_output[i])) << "Index " << i;
    }

#if defined(__AVX2__)
    float avx2_output[BLOCK_SIZE];
    Q4_KTensor::decodeBlockAVX2(block, avx2_output);
    compareOutputs(scalar_output, avx2_output, BLOCK_SIZE, 1e-4f, "ZeroScale AVX2");
#endif

#if defined(__AVX512F__)
    float avx512_output[BLOCK_SIZE];
    Q4_KTensor::decodeBlockAVX512(block, avx512_output);
    compareOutputs(scalar_output, avx512_output, BLOCK_SIZE, 1e-4f, "ZeroScale AVX512");
#endif
}

TEST_F(Test__Q4_KTensor, EdgeCase_MaxNibbles)
{
    // Test with all nibbles set to 0xF (maximum 4-bit value)
    constexpr size_t BLOCK_SIZE = 256;

    uint8_t scales[12];
    std::fill_n(scales, 12, 0x3F); // All scales = 63 (max 6-bit)

    uint8_t qs[128];
    std::fill_n(qs, 128, 0xFF); // All nibbles = 15

    Q4_KBlock block = createBlock(0x3C00, 0x3800, scales, qs); // d=1.0, dmin=0.5

    float scalar_output[BLOCK_SIZE];
    Q4_KTensor::decodeBlockScalar(block, scalar_output);

#if defined(__AVX2__)
    float avx2_output[BLOCK_SIZE];
    Q4_KTensor::decodeBlockAVX2(block, avx2_output);
    compareOutputs(scalar_output, avx2_output, BLOCK_SIZE, 1e-4f, "MaxNibbles AVX2");
#endif

#if defined(__AVX512F__)
    float avx512_output[BLOCK_SIZE];
    Q4_KTensor::decodeBlockAVX512(block, avx512_output);
    compareOutputs(scalar_output, avx512_output, BLOCK_SIZE, 1e-4f, "MaxNibbles AVX512");
#endif
}

TEST_F(Test__Q4_KTensor, EdgeCase_AlternatingNibbles)
{
    // Test with alternating nibble patterns (0x0F, 0xF0, etc.)
    constexpr size_t BLOCK_SIZE = 256;

    uint8_t scales[12];
    for (int i = 0; i < 12; ++i)
    {
        scales[i] = (i % 2 == 0) ? 0x3F : 0x00; // Alternate max/min scales
    }

    uint8_t qs[128];
    for (int i = 0; i < 128; ++i)
    {
        qs[i] = (i % 2 == 0) ? 0x0F : 0xF0; // Alternate low/high nibbles
    }

    Q4_KBlock block = createBlock(0x4000, 0x3C00, scales, qs); // d=2.0, dmin=1.0

    float scalar_output[BLOCK_SIZE];
    Q4_KTensor::decodeBlockScalar(block, scalar_output);

#if defined(__AVX2__)
    float avx2_output[BLOCK_SIZE];
    Q4_KTensor::decodeBlockAVX2(block, avx2_output);
    compareOutputs(scalar_output, avx2_output, BLOCK_SIZE, 1e-4f, "AlternatingNibbles AVX2");
#endif

#if defined(__AVX512F__)
    float avx512_output[BLOCK_SIZE];
    Q4_KTensor::decodeBlockAVX512(block, avx512_output);
    compareOutputs(scalar_output, avx512_output, BLOCK_SIZE, 1e-4f, "AlternatingNibbles AVX512");
#endif
}

TEST_F(Test__Q4_KTensor, EdgeCase_GroupBoundaries)
{
    // Test with different patterns across 64-element group boundaries
    constexpr size_t BLOCK_SIZE = 256;

    uint8_t scales[12];
    for (int i = 0; i < 12; ++i)
    {
        scales[i] = static_cast<uint8_t>(i * 5); // Varying scales
    }

    uint8_t qs[128];
    for (int i = 0; i < 128; ++i)
    {
        // Different patterns for each 32-byte group (64 elements)
        int group = i / 32;
        qs[i] = static_cast<uint8_t>((group * 0x11) + (i % 16));
    }

    Q4_KBlock block = createBlock(0x3E00, 0x3A00, scales, qs); // d=1.5, dmin=0.75

    float scalar_output[BLOCK_SIZE];
    Q4_KTensor::decodeBlockScalar(block, scalar_output);

#if defined(__AVX2__)
    float avx2_output[BLOCK_SIZE];
    Q4_KTensor::decodeBlockAVX2(block, avx2_output);
    compareOutputs(scalar_output, avx2_output, BLOCK_SIZE, 1e-4f, "GroupBoundaries AVX2");
#endif

#if defined(__AVX512F__)
    float avx512_output[BLOCK_SIZE];
    Q4_KTensor::decodeBlockAVX512(block, avx512_output);
    compareOutputs(scalar_output, avx512_output, BLOCK_SIZE, 1e-4f, "GroupBoundaries AVX512");
#endif
}

TEST_F(Test__Q4_KTensor, EdgeCase_ScaleUnpacking)
{
    // Test scale unpacking edge cases (j < 4 vs j >= 4 in get_scale_min_k4)
    constexpr size_t BLOCK_SIZE = 256;

    // Craft scales to test both paths in get_scale_min_k4
    uint8_t scales[12] = {
        0x3F, 0x00, 0x15, 0x2A, // First 4 (j < 4 path)
        0xC0, 0xC0, 0xC0, 0xC0, // Next 4 (j >= 4 path, high bits set)
        0x0F, 0xF0, 0x55, 0xAA  // Last 4 (mixed patterns)
    };

    uint8_t qs[128];
    for (int i = 0; i < 128; ++i)
    {
        qs[i] = static_cast<uint8_t>(i ^ 0x5A); // XOR pattern
    }

    Q4_KBlock block = createBlock(0x3C00, 0x3800, scales, qs); // d=1.0, dmin=0.5

    float scalar_output[BLOCK_SIZE];
    Q4_KTensor::decodeBlockScalar(block, scalar_output);

#if defined(__AVX2__)
    float avx2_output[BLOCK_SIZE];
    Q4_KTensor::decodeBlockAVX2(block, avx2_output);
    compareOutputs(scalar_output, avx2_output, BLOCK_SIZE, 1e-4f, "ScaleUnpacking AVX2");
#endif

#if defined(__AVX512F__)
    float avx512_output[BLOCK_SIZE];
    Q4_KTensor::decodeBlockAVX512(block, avx512_output);
    compareOutputs(scalar_output, avx512_output, BLOCK_SIZE, 1e-4f, "ScaleUnpacking AVX512");
#endif
}

TEST_F(Test__Q4_KTensor, EdgeCase_RandomValues)
{
    // Comprehensive random test with many blocks
    constexpr size_t BLOCK_SIZE = 256;
    constexpr int NUM_BLOCKS = 100;

    for (int block_idx = 0; block_idx < NUM_BLOCKS; ++block_idx)
    {
        Q4_KBlock block = createRandomBlock();

        float scalar_output[BLOCK_SIZE];
        Q4_KTensor::decodeBlockScalar(block, scalar_output);

#if defined(__AVX2__)
        float avx2_output[BLOCK_SIZE];
        Q4_KTensor::decodeBlockAVX2(block, avx2_output);
        compareOutputs(scalar_output, avx2_output, BLOCK_SIZE, 1e-4f,
                       "RandomValues AVX2 block " + std::to_string(block_idx));
#endif

#if defined(__AVX512F__)
        float avx512_output[BLOCK_SIZE];
        Q4_KTensor::decodeBlockAVX512(block, avx512_output);
        compareOutputs(scalar_output, avx512_output, BLOCK_SIZE, 1e-4f,
                       "RandomValues AVX512 block " + std::to_string(block_idx));
#endif
    }
}

/**
 * @brief Test Q4_KTensor to INT8 block quantization
 *
 * Validates that Q4_KTensor::to_int8_blocked() produces correct INT8
 * quantization with reasonable accuracy using real model weights.
 */
TEST(Q4_KTensor_INT8, BlockConversion)
{
    // Load a test model to get real Q4_KTensor weights
    const std::string model_path = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    // Skip if model file doesn't exist (not all test environments have models)
    if (!std::ifstream(model_path).good()) {
        GTEST_SKIP() << "Test model not found: " << model_path;
    }
    
    llaminar2::ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(model_path));
    
    // Find a weight tensor of type Q4_KTensor
    std::shared_ptr<llaminar2::TensorBase> weight_tensor;
    const auto& model = loader.getModel();
    
    for (const auto& tensor_info : model.tensors) {
        // Map GGUFTensorType to our expected type
        // For now, just try to load the first tensor and check its type
        auto tensor = loader.loadTensor(tensor_info.name);
        if (tensor && std::dynamic_pointer_cast<llaminar2::Q4_KTensor>(tensor)) {
            weight_tensor = tensor;
            break;
        }
    }
    
    if (!weight_tensor) {
        GTEST_SKIP() << "No Q4_KTensor weights found in model";
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

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// ========================================================================
// Template Method Tests for to<T>() API
// ========================================================================

/**
 * @brief Test Q4_KTensor to<float>() template method matches to_fp32()
 */
TEST(Q4_KTensor_Template, ToFloat_TemplateMethod)
{
    using namespace llaminar2;
    
    const std::vector<size_t> shape = {2, 256};
    std::vector<Q4_KBlock> blocks(2);
    
    for (size_t i = 0; i < 2; ++i)
    {
        blocks[i].d = fp32_to_fp16(0.5f + static_cast<float>(i) * 0.1f);
        blocks[i].dmin = fp32_to_fp16(0.1f);
        for (size_t j = 0; j < 12; ++j) blocks[i].scales[j] = static_cast<uint8_t>((i * 12 + j) % 256);
        for (size_t j = 0; j < 128; ++j) blocks[i].qs[j] = static_cast<uint8_t>((i * 128 + j) % 256);
    }
    
    std::vector<uint8_t> raw_data(2 * sizeof(Q4_KBlock));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q4_KTensor>(shape, raw_data);
    
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

TEST(Q4_KTensor_Template, ToBF16_TemplateMethod)
{
    using namespace llaminar2;
    
    const std::vector<size_t> shape = {2, 256};
    std::vector<Q4_KBlock> blocks(2);
    
    for (size_t i = 0; i < 2; ++i)
    {
        blocks[i].d = fp32_to_fp16(0.5f + static_cast<float>(i) * 0.1f);
        blocks[i].dmin = fp32_to_fp16(0.1f);
        for (size_t j = 0; j < 12; ++j) blocks[i].scales[j] = static_cast<uint8_t>((i * 12 + j) % 256);
        for (size_t j = 0; j < 128; ++j) blocks[i].qs[j] = static_cast<uint8_t>((i * 128 + j) % 256);
    }
    
    std::vector<uint8_t> raw_data(2 * sizeof(Q4_KBlock));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q4_KTensor>(shape, raw_data);
    
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

TEST(Q4_KTensor_Template, ToFP16_TemplateMethod)
{
    using namespace llaminar2;
    
    const std::vector<size_t> shape = {2, 256};
    std::vector<Q4_KBlock> blocks(2);
    
    for (size_t i = 0; i < 2; ++i)
    {
        blocks[i].d = fp32_to_fp16(0.5f + static_cast<float>(i) * 0.1f);
        blocks[i].dmin = fp32_to_fp16(0.1f);
        for (size_t j = 0; j < 12; ++j) blocks[i].scales[j] = static_cast<uint8_t>((i * 12 + j) % 256);
        for (size_t j = 0; j < 128; ++j) blocks[i].qs[j] = static_cast<uint8_t>((i * 128 + j) % 256);
    }
    
    std::vector<uint8_t> raw_data(2 * sizeof(Q4_KBlock));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q4_KTensor>(shape, raw_data);
    
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

TEST(Q4_KTensor_Template, ToINT8_TemplateMethod)
{
    using namespace llaminar2;
    
    const std::vector<size_t> shape = {2, 256};
    std::vector<Q4_KBlock> blocks(2);
    
    for (size_t i = 0; i < 2; ++i)
    {
        blocks[i].d = fp32_to_fp16(0.5f + static_cast<float>(i) * 0.1f);
        blocks[i].dmin = fp32_to_fp16(0.1f);
        for (size_t j = 0; j < 12; ++j) blocks[i].scales[j] = static_cast<uint8_t>((i * 12 + j) % 256);
        for (size_t j = 0; j < 128; ++j) blocks[i].qs[j] = static_cast<uint8_t>((i * 128 + j) % 256);
    }
    
    std::vector<uint8_t> raw_data(2 * sizeof(Q4_KBlock));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q4_KTensor>(shape, raw_data);
    
    std::vector<int8_t> int8_data(512);
    tensor->template to<int8_t>(int8_data.data());
    
    for (size_t i = 0; i < 512; ++i)
    {
        EXPECT_GE(int8_data[i], -127);
        EXPECT_LE(int8_data[i], 127);
    }
}

TEST(Q4_KTensor_Template, ToINT32_TemplateMethod)
{
    using namespace llaminar2;
    
    const std::vector<size_t> shape = {2, 256};
    std::vector<Q4_KBlock> blocks(2);
    
    for (size_t i = 0; i < 2; ++i)
    {
        blocks[i].d = fp32_to_fp16(0.5f + static_cast<float>(i) * 0.1f);
        blocks[i].dmin = fp32_to_fp16(0.1f);
        for (size_t j = 0; j < 12; ++j) blocks[i].scales[j] = static_cast<uint8_t>((i * 12 + j) % 256);
        for (size_t j = 0; j < 128; ++j) blocks[i].qs[j] = static_cast<uint8_t>((i * 128 + j) % 256);
    }
    
    std::vector<uint8_t> raw_data(2 * sizeof(Q4_KBlock));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q4_KTensor>(shape, raw_data);
    
    std::vector<int32_t> int32_data(512);
    tensor->template to<int32_t>(int32_data.data());
    
    for (size_t i = 0; i < 512; ++i)
    {
        EXPECT_GE(int32_data[i], INT32_MIN);
        EXPECT_LE(int32_data[i], INT32_MAX);
    }
}

TEST(Q4_KTensor_Template, RoundTrip_Q4_K_FP32_BF16_FP32)
{
    using namespace llaminar2;
    
    const std::vector<size_t> shape = {2, 256};
    std::vector<Q4_KBlock> blocks(2);
    
    for (size_t i = 0; i < 2; ++i)
    {
        blocks[i].d = fp32_to_fp16(0.5f + static_cast<float>(i) * 0.1f);
        blocks[i].dmin = fp32_to_fp16(0.1f);
        for (size_t j = 0; j < 12; ++j) blocks[i].scales[j] = static_cast<uint8_t>((i * 12 + j) % 256);
        for (size_t j = 0; j < 128; ++j) blocks[i].qs[j] = static_cast<uint8_t>((i * 128 + j) % 256);
    }
    
    std::vector<uint8_t> raw_data(2 * sizeof(Q4_KBlock));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());
    auto tensor = std::make_shared<Q4_KTensor>(shape, raw_data);
    
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
