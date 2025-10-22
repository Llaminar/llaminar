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
#include "../src/tensors/TensorFactory.h"
#include "../src/utils/SIMDHelpers.h"
#include <vector>
#include <cmath>

using namespace llaminar;

// Helper for FP16 conversion
static inline uint16_t fp32_to_fp16(float x) {
    // Simple FP16 conversion
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(float));
    uint16_t sign = (bits >> 16) & 0x8000;
    uint16_t exp = ((bits >> 23) & 0xFF) - 112;
    uint16_t frac = (bits >> 13) & 0x3FF;
    if (exp <= 0) return sign; // Underflow
    if (exp >= 31) return sign | 0x7C00; // Overflow/Inf
    return sign | (exp << 10) | frac;
}

/**
 * @brief Helper to create test IQ4_NL data
 */
static std::vector<uint8_t> createTestData(size_t num_blocks) {
    std::vector<uint8_t> data(num_blocks * sizeof(IQ4_NLBlock));
    auto* blocks = reinterpret_cast<IQ4_NLBlock*>(data.data());
    
    for (size_t i = 0; i < num_blocks; ++i) {
        float scale = 0.01f + static_cast<float>(i) * 0.001f;
        blocks[i].d = fp32_to_fp16(scale);
        
        for (size_t j = 0; j < 16; ++j) {
            uint8_t idx_low = (i * 17 + j * 3) % 16;
            uint8_t idx_high = (i * 13 + j * 5) % 16;
            blocks[i].qs[j] = (idx_high << 4) | idx_low;
        }
    }
    
    return data;
}

TEST(IQ4_NLTensorTest, ConstructValid) {
    auto data = createTestData(4);
    ASSERT_NO_THROW({
        IQ4_NLTensor tensor({1, 128}, data);
        EXPECT_EQ(tensor.size(), 128);
    });
}

TEST(IQ4_NLTensorTest, DecodeRow) {
    auto data = createTestData(8);  // 256 elements = 8 blocks
    IQ4_NLTensor tensor({8, 32}, data);
    
    std::vector<float> output(32);
    ASSERT_NO_THROW(tensor.decodeRow(0, output.data()));
    
    for (float val : output) {
        EXPECT_TRUE(std::isfinite(val));
    }
}

TEST(IQ4_NLTensorTest, DecodeToFP32) {
    auto data = createTestData(4);
    IQ4_NLTensor tensor({4, 32}, data);
    
    std::vector<float> output(128);
    ASSERT_NO_THROW(tensor.decode_to_fp32(output.data()));
    
    for (float val : output) {
        EXPECT_TRUE(std::isfinite(val));
    }
}

TEST(IQ4_NLTensorTest, DecodeSpan) {
    auto data = createTestData(8);
    IQ4_NLTensor tensor({1, 256}, data);
    
    std::vector<float> output(64);
    ASSERT_NO_THROW(tensor.decodeSpan(32, 64, output.data()));
    
    for (float val : output) {
        EXPECT_TRUE(std::isfinite(val));
    }
}

TEST(IQ4_NLTensorTest, BlockDescriptor) {
    auto data = createTestData(1);
    IQ4_NLTensor tensor({1, 32}, data);
    
    const auto& desc = tensor.block_descriptor();
    EXPECT_EQ(desc.elements_per_block, 32);
    EXPECT_EQ(desc.bytes_per_block, 18);
}

TEST(IQ4_NLTensorTest, QuantType) {
    auto data = createTestData(1);
    IQ4_NLTensor tensor({1, 32}, data);
    
    EXPECT_EQ(tensor.quant_type(), QuantType::IQ4_NL);
    EXPECT_FLOAT_EQ(tensor.compression_ratio(), 7.1f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
