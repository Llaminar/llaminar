/**
 * @file test_iq4_nl_tensor.cpp
 * @brief Unit tests for IQ4_NLTensor
 * 
 * Tests for IQ4_NL (Non-Linear 4-bit) quantized tensor implementation.
 * 
 * @author David Sanftenberg
 * @date 2025-01-15
 */

#include <gtest/gtest.h>
#include "../src/tensors/IQ4_NLTensor.h"
#include "../src/utils/BFloat16.h"
#include "../src/utils/SIMDHelpers.h"
#include <vector>
#include <cmath>
#include <random>

using namespace llaminar;

/**
 * @brief Helper function to create dummy IQ4_NL data
 * 
 * Creates realistic test data with varied scales and indices.
 */
static std::vector<uint8_t> createDummyIQ4NL(size_t num_blocks) {
    std::vector<uint8_t> data(num_blocks * sizeof(IQ4_NLBlock));
    auto* blocks = reinterpret_cast<IQ4_NLBlock*>(data.data());
    
    std::mt19937 rng(42);  // Deterministic seed
    std::uniform_int_distribution<int> dist_idx(0, 15);  // 4-bit indices
    
    for (size_t i = 0; i < num_blocks; ++i) {
        // Varied scale using different patterns per block
        float scale = 0.01f + static_cast<float>(i) * 0.001f;
        blocks[i].d = simd::fp32_to_fp16(scale);
        
        // Fill qs[] with varied 4-bit indices (using prime multipliers for variation)
        for (size_t j = 0; j < 16; ++j) {
            uint8_t idx_low = (i * 17 + j * 3) % 16;
            uint8_t idx_high = (i * 13 + j * 5) % 16;
            blocks[i].qs[j] = (idx_high << 4) | idx_low;
        }
    }
    
    return data;
}

/**
 * @brief Create IQ4_NL tensor with given dimensions
 */
static IQ4_NLTensor createTensor(int rows, int cols) {
    size_t num_elements = rows * cols;
    size_t num_blocks = (num_elements + 31) / 32; // QK4_NL = 32
    auto data = createDummyIQ4NL(num_blocks);
    return IQ4_NLTensor({rows, cols}, data);
}

// ===== Basic Functionality Tests =====

TEST(IQ4_NLTensorTest, ConstructValid) {
    const size_t num_blocks = 4;
    const size_t num_elements = num_blocks * 32;  // QK4_NL = 32
    
    auto data = createDummyIQ4NL(num_blocks);
    
    ASSERT_NO_THROW({
        IQ4_NLTensor tensor({1, static_cast<int>(num_elements)}, data);
        EXPECT_EQ(tensor.size(), num_elements);
        EXPECT_EQ(tensor.element_count(), num_elements);
        EXPECT_EQ(tensor.raw_size(), data.size());
    });
}

TEST(IQ4_NLTensorTest, ConstructInvalidSize) {
    const size_t num_blocks = 4;
    const size_t num_elements = num_blocks * QK4_NL + 7;  // Not multiple of QK4_NL
    
    auto data = createDummyIQ4NL(num_blocks);
    
    EXPECT_THROW({
        IQ4_NLTensor tensor(num_elements, data.data(), data.size());
    }, std::invalid_argument);
}

TEST(IQ4_NLTensorTest, ConstructInvalidDataSize) {
    const size_t num_blocks = 4;
    const size_t num_elements = num_blocks * QK4_NL;
    
    auto data = createDummyIQ4NL(num_blocks);
    const size_t wrong_size = data.size() - 10;
    
    EXPECT_THROW({
        IQ4_NLTensor tensor(num_elements, data.data(), wrong_size);
    }, std::invalid_argument);
}

// ===== Decode Tests =====

TEST(IQ4_NLTensorTest, DecodeToFP32SingleBlock) {
    const size_t num_blocks = 1;
    const size_t num_elements = QK4_NL;
    
    auto data = createDummyIQ4NL(num_blocks);
    IQ4_NLTensor tensor(num_elements, data.data(), data.size());
    
    std::vector<float> output(num_elements);
    ASSERT_NO_THROW(tensor.decode(output.data()));
    
    // Verify all values decoded
    bool all_decoded = true;
    for (size_t i = 0; i < num_elements; ++i) {
        if (!std::isfinite(output[i])) {
            all_decoded = false;
            break;
        }
    }
    EXPECT_TRUE(all_decoded);
}

TEST(IQ4_NLTensorTest, DecodeToFP32MultiBlock) {
    const size_t num_blocks = 10;
    const size_t num_elements = num_blocks * QK4_NL;
    
    auto data = createDummyIQ4NL(num_blocks);
    IQ4_NLTensor tensor(num_elements, data.data(), data.size());
    
    std::vector<float> output(num_elements);
    ASSERT_NO_THROW(tensor.decode(output.data()));
    
    // Verify all values finite
    for (size_t i = 0; i < num_elements; ++i) {
        EXPECT_TRUE(std::isfinite(output[i]));
    }
}

TEST(IQ4_NLTensorTest, DecodeToBF16SingleBlock) {
    const size_t num_blocks = 1;
    const size_t num_elements = QK4_NL;
    
    auto data = createDummyIQ4NL(num_blocks);
    IQ4_NLTensor tensor(num_elements, data.data(), data.size());
    
    std::vector<bfloat16_t> output(num_elements);
    ASSERT_NO_THROW(tensor.decodeBF16(output.data()));
    
    // Convert back to FP32 and verify
    for (size_t i = 0; i < num_elements; ++i) {
        float val = static_cast<float>(output[i]);
        EXPECT_TRUE(std::isfinite(val));
    }
}

TEST(IQ4_NLTensorTest, DecodeToBF16MultiBlock) {
    const size_t num_blocks = 10;
    const size_t num_elements = num_blocks * QK4_NL;
    
    auto data = createDummyIQ4NL(num_blocks);
    IQ4_NLTensor tensor(num_elements, data.data(), data.size());
    
    std::vector<bfloat16_t> output(num_elements);
    ASSERT_NO_THROW(tensor.decodeBF16(output.data()));
    
    // Verify all values valid
    for (size_t i = 0; i < num_elements; ++i) {
        float val = static_cast<float>(output[i]);
        EXPECT_TRUE(std::isfinite(val));
    }
}

// ===== Streaming Decode Tests =====

TEST(IQ4_NLTensorTest, DecodeRangeValid) {
    const size_t num_blocks = 10;
    const size_t num_elements = num_blocks * QK4_NL;
    
    auto data = createDummyIQ4NL(num_blocks);
    IQ4_NLTensor tensor(num_elements, data.data(), data.size());
    
    // Decode middle blocks (2-6)
    const size_t start_block = 2;
    const size_t decode_blocks = 5;
    std::vector<float> output(decode_blocks * QK4_NL);
    
    ASSERT_NO_THROW(tensor.decodeRange(output.data(), start_block, decode_blocks));
    
    // Verify values
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_TRUE(std::isfinite(output[i]));
    }
}

TEST(IQ4_NLTensorTest, DecodeRangeOutOfBounds) {
    const size_t num_blocks = 10;
    const size_t num_elements = num_blocks * QK4_NL;
    
    auto data = createDummyIQ4NL(num_blocks);
    IQ4_NLTensor tensor(num_elements, data.data(), data.size());
    
    std::vector<float> output(5 * QK4_NL);
    
    // Try to decode past end
    EXPECT_THROW({
        tensor.decodeRange(output.data(), 8, 5);  // 8 + 5 > 10
    }, std::out_of_range);
}

TEST(IQ4_NLTensorTest, DecodeRangeBF16Valid) {
    const size_t num_blocks = 10;
    const size_t num_elements = num_blocks * QK4_NL;
    
    auto data = createDummyIQ4NL(num_blocks);
    IQ4_NLTensor tensor(num_elements, data.data(), data.size());
    
    const size_t start_block = 1;
    const size_t decode_blocks = 3;
    std::vector<bfloat16_t> output(decode_blocks * QK4_NL);
    
    ASSERT_NO_THROW(tensor.decodeRangeBF16(output.data(), start_block, decode_blocks));
    
    // Verify values
    for (size_t i = 0; i < output.size(); ++i) {
        float val = static_cast<float>(output[i]);
        EXPECT_TRUE(std::isfinite(val));
    }
}

// ===== Value Correctness Tests =====

TEST(IQ4_NLTensorTest, LookupTableBounds) {
    // Verify kvalues_iq4nl lookup table is properly bounded
    const size_t num_blocks = 1;
    const size_t num_elements = QK4_NL;
    
    // Create block with all possible indices (0-15)
    std::vector<uint8_t> data(sizeof(IQ4_NLBlock));
    auto* block = reinterpret_cast<IQ4_NLBlock*>(data.data());
    
    block->d = fp32_to_fp16(1.0f);
    for (size_t i = 0; i < 16; ++i) {
        block->qs[i] = (i << 4) | i;  // Same index in high and low nibbles
    }
    
    IQ4_NLTensor tensor(num_elements, data.data(), data.size());
    std::vector<float> output(num_elements);
    
    ASSERT_NO_THROW(tensor.decode(output.data()));
    
    // All indices should decode to valid values from kvalues_iq4nl
    // Range: -127 to 113
    for (size_t i = 0; i < num_elements; ++i) {
        EXPECT_GE(output[i], -127.0f);
        EXPECT_LE(output[i], 113.0f);
    }
}

TEST(IQ4_NLTensorTest, ScaleApplication) {
    // Verify scale is properly applied
    const size_t num_blocks = 1;
    const size_t num_elements = QK4_NL;
    
    std::vector<uint8_t> data(sizeof(IQ4_NLBlock));
    auto* block = reinterpret_cast<IQ4_NLBlock*>(data.data());
    
    const float scale = 2.5f;
    block->d = fp32_to_fp16(scale);
    
    // Set all indices to 0 (kvalues_iq4nl[0] = -127)
    for (size_t i = 0; i < 16; ++i) {
        block->qs[i] = 0x00;
    }
    
    IQ4_NLTensor tensor(num_elements, data.data(), data.size());
    std::vector<float> output(num_elements);
    tensor.decode(output.data());
    
    // All values should be scale * -127
    const float expected = scale * -127.0f;
    for (size_t i = 0; i < num_elements; ++i) {
        EXPECT_NEAR(output[i], expected, 0.1f);  // Allow FP16 conversion error
    }
}

TEST(IQ4_NLTensorTest, DifferentBlocksDecodeUniquely) {
    // Verify different blocks produce different outputs
    const size_t num_blocks = 3;
    const size_t num_elements = num_blocks * QK4_NL;
    
    auto data = createDummyIQ4NL(num_blocks);
    IQ4_NLTensor tensor(num_elements, data.data(), data.size());
    
    std::vector<float> output(num_elements);
    tensor.decode(output.data());
    
    // Compare first elements of each block (should be different due to different scales)
    EXPECT_NE(output[0], output[QK4_NL]);
    EXPECT_NE(output[QK4_NL], output[2 * QK4_NL]);
}

// ===== Edge Cases =====

TEST(IQ4_NLTensorTest, ZeroScale) {
    const size_t num_blocks = 1;
    const size_t num_elements = QK4_NL;
    
    std::vector<uint8_t> data(sizeof(IQ4_NLBlock));
    auto* block = reinterpret_cast<IQ4_NLBlock*>(data.data());
    
    block->d = 0;  // Zero scale
    for (size_t i = 0; i < 16; ++i) {
        block->qs[i] = 0xFF;  // Max indices
    }
    
    IQ4_NLTensor tensor(num_elements, data.data(), data.size());
    std::vector<float> output(num_elements);
    tensor.decode(output.data());
    
    // All values should be zero
    for (size_t i = 0; i < num_elements; ++i) {
        EXPECT_FLOAT_EQ(output[i], 0.0f);
    }
}

TEST(IQ4_NLTensorTest, LargeScale) {
    const size_t num_blocks = 1;
    const size_t num_elements = QK4_NL;
    
    std::vector<uint8_t> data(sizeof(IQ4_NLBlock));
    auto* block = reinterpret_cast<IQ4_NLBlock*>(data.data());
    
    const float large_scale = 10.0f;
    block->d = fp32_to_fp16(large_scale);
    
    // Set to index 15 (kvalues_iq4nl[15] = 113)
    for (size_t i = 0; i < 16; ++i) {
        block->qs[i] = 0xFF;
    }
    
    IQ4_NLTensor tensor(num_elements, data.data(), data.size());
    std::vector<float> output(num_elements);
    tensor.decode(output.data());
    
    const float expected = large_scale * 113.0f;
    for (size_t i = 0; i < num_elements; ++i) {
        EXPECT_NEAR(output[i], expected, 1.0f);
    }
}

// ===== Multi-threading Tests =====

#ifdef _OPENMP
TEST(IQ4_NLTensorTest, MultiThreadedDecode) {
    const size_t num_blocks = 100;
    const size_t num_elements = num_blocks * QK4_NL;
    
    auto data = createDummyIQ4NL(num_blocks);
    IQ4_NLTensor tensor(num_elements, data.data(), data.size());
    
    std::vector<float> output(num_elements);
    
    // Should use OpenMP internally
    ASSERT_NO_THROW(tensor.decode(output.data()));
    
    // Verify all values valid
    for (size_t i = 0; i < num_elements; ++i) {
        EXPECT_TRUE(std::isfinite(output[i]));
    }
}
#endif

// ===== Consistency Tests =====

TEST(IQ4_NLTensorTest, DecodeConsistency) {
    // Decode same data twice, should get same results
    const size_t num_blocks = 5;
    const size_t num_elements = num_blocks * QK4_NL;
    
    auto data = createDummyIQ4NL(num_blocks);
    IQ4_NLTensor tensor(num_elements, data.data(), data.size());
    
    std::vector<float> output1(num_elements);
    std::vector<float> output2(num_elements);
    
    tensor.decode(output1.data());
    tensor.decode(output2.data());
    
    for (size_t i = 0; i < num_elements; ++i) {
        EXPECT_FLOAT_EQ(output1[i], output2[i]);
    }
}

TEST(IQ4_NLTensorTest, FP32BF16Similarity) {
    // FP32 and BF16 decode should be similar
    const size_t num_blocks = 5;
    const size_t num_elements = num_blocks * QK4_NL;
    
    auto data = createDummyIQ4NL(num_blocks);
    IQ4_NLTensor tensor(num_elements, data.data(), data.size());
    
    std::vector<float> fp32_output(num_elements);
    std::vector<bfloat16_t> bf16_output(num_elements);
    
    tensor.decode(fp32_output.data());
    tensor.decodeBF16(bf16_output.data());
    
    // Convert BF16 to FP32 and compare
    for (size_t i = 0; i < num_elements; ++i) {
        float bf16_as_fp32 = static_cast<float>(bf16_output[i]);
        // BF16 has less precision, allow larger tolerance
        EXPECT_NEAR(fp32_output[i], bf16_as_fp32, std::abs(fp32_output[i]) * 0.01f + 0.001f);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
