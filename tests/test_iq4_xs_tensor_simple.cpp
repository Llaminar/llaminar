/**
 * @file test_iq4_xs_tensor.cpp
 * @brief Unit tests for IQ4_XSTensor
 * 
 * Tests for IQ4_XS (Extra Small 4-bit) quantized tensor implementation.
 * 
 * @author David Sanftenberg
 * @date 2025-01-15
 */

#include <gtest/gtest.h>
#include "../src/tensors/IQ4_XSTensor.h"
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
 * @brief Helper to create test IQ4_XS data
 */
static std::vector<uint8_t> createTestData(size_t num_blocks) {
    std::vector<uint8_t> data(num_blocks * sizeof(IQ4_XSBlock));
    auto* blocks = reinterpret_cast<IQ4_XSBlock*>(data.data());
    
    for (size_t i = 0; i < num_blocks; ++i) {
        float global_scale = 0.01f + static_cast<float>(i) * 0.0005f;
        blocks[i].d = fp32_to_fp16(global_scale);
        
        // Initialize scales
        blocks[i].scales_h = 0;
        for (size_t sb = 0; sb < 4; ++sb) {
            blocks[i].scales_l[sb] = 0;
        }
        
        // Pack 8 sub-block scales
        for (size_t sb = 0; sb < 8; ++sb) {
            int scale_6bit = ((i * 31 + sb * 17) % 64);
            int low_4 = scale_6bit & 0x0F;
            int high_2 = (scale_6bit >> 4) & 0x03;
            
            if (sb % 2 == 0) {
                blocks[i].scales_l[sb / 2] |= low_4;
            } else {
                blocks[i].scales_l[sb / 2] |= (low_4 << 4);
            }
            blocks[i].scales_h |= (high_2 << (2 * sb));
        }
        
        // Fill qs
        for (size_t j = 0; j < 128; ++j) {
            uint8_t idx_low = (i * 19 + j * 7) % 16;
            uint8_t idx_high = (i * 23 + j * 11) % 16;
            blocks[i].qs[j] = (idx_high << 4) | idx_low;
        }
    }
    
    return data;
}

TEST(IQ4_XSTensorTest, ConstructValid) {
    auto data = createTestData(4);
    ASSERT_NO_THROW({
        IQ4_XSTensor tensor({1, 1024}, data);
        EXPECT_EQ(tensor.size(), 1024);
    });
}

TEST(IQ4_XSTensorTest, DecodeRow) {
    auto data = createTestData(2);  // 512 elements = 2 blocks
    IQ4_XSTensor tensor({2, 256}, data);
    
    std::vector<float> output(256);
    ASSERT_NO_THROW(tensor.decodeRow(0, output.data()));
    
    for (float val : output) {
        EXPECT_TRUE(std::isfinite(val));
    }
}

TEST(IQ4_XSTensorTest, DecodeToFP32) {
    auto data = createTestData(2);
    IQ4_XSTensor tensor({2, 256}, data);
    
    std::vector<float> output(512);
    ASSERT_NO_THROW(tensor.decode_to_fp32(output.data()));
    
    for (float val : output) {
        EXPECT_TRUE(std::isfinite(val));
    }
}

TEST(IQ4_XSTensorTest, DecodeSpan) {
    auto data = createTestData(4);
    IQ4_XSTensor tensor({1, 1024}, data);
    
    std::vector<float> output(256);
    ASSERT_NO_THROW(tensor.decodeSpan(128, 256, output.data()));
    
    for (float val : output) {
        EXPECT_TRUE(std::isfinite(val));
    }
}

TEST(IQ4_XSTensorTest, BlockDescriptor) {
    auto data = createTestData(1);
    IQ4_XSTensor tensor({1, 256}, data);
    
    const auto& desc = tensor.block_descriptor();
    EXPECT_EQ(desc.elements_per_block, 256);
    EXPECT_EQ(desc.bytes_per_block, 136);
}

TEST(IQ4_XSTensorTest, QuantType) {
    auto data = createTestData(1);
    IQ4_XSTensor tensor({1, 256}, data);
    
    EXPECT_EQ(tensor.quant_type(), QuantType::IQ4_XS);
    EXPECT_FLOAT_EQ(tensor.compression_ratio(), 7.5f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
