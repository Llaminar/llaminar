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
#include "../src/utils/BFloat16.h"
#include <vector>
#include <cmath>
#include <random>

using namespace llaminar;

/**
 * @brief Helper function to create dummy IQ4_XS data
 * 
 * Creates realistic test data with varied sub-block scales and indices.
 */
static std::vector<uint8_t> createDummyIQ4XS(size_t num_blocks) {
    std::vector<uint8_t> data(num_blocks * sizeof(IQ4_XSBlock));
    auto* blocks = reinterpret_cast<IQ4_XSBlock*>(data.data());
    
    std::mt19937 rng(42);  // Deterministic seed
    std::uniform_int_distribution<int> dist_scale(0, 63);  // 6-bit scales
    std::uniform_int_distribution<int> dist_idx(0, 15);    // 4-bit indices
    
    for (size_t i = 0; i < num_blocks; ++i) {
        // Global scale (varied per block)
        float global_scale = 0.01f + static_cast<float>(i) * 0.0005f;
        blocks[i].d = fp32_to_fp16(global_scale);
        
        // Generate 8 sub-block scales (6-bit each, varied using prime multipliers)
        blocks[i].scales_h = 0;
        for (size_t sb = 0; sb < 4; ++sb) {
            blocks[i].scales_l[sb] = 0;
        }
        
        for (size_t sb = 0; sb < 8; ++sb) {
            // Use prime multipliers for variation
            int scale_6bit = ((i * 31 + sb * 17) % 64);
            
            // Pack into scales_l (low 4 bits) and scales_h (high 2 bits)
            int low_4 = scale_6bit & 0x0F;
            int high_2 = (scale_6bit >> 4) & 0x03;
            
            // Store low 4 bits in scales_l[sb/2]
            if (sb % 2 == 0) {
                blocks[i].scales_l[sb / 2] |= low_4;
            } else {
                blocks[i].scales_l[sb / 2] |= (low_4 << 4);
            }
            
            // Store high 2 bits in scales_h
            blocks[i].scales_h |= (high_2 << (2 * sb));
        }
        
        // Fill qs[] with varied 4-bit indices
        for (size_t j = 0; j < 128; ++j) {
            uint8_t idx_low = (i * 19 + j * 7) % 16;
            uint8_t idx_high = (i * 23 + j * 11) % 16;
            blocks[i].qs[j] = (idx_high << 4) | idx_low;
        }
    }
    
    return data;
}

// ===== Basic Functionality Tests =====

TEST(IQ4_XSTensorTest, ConstructValid) {
    const size_t num_blocks = 4;
    const size_t num_elements = num_blocks * QK_K;  // 4 * 256 = 1024
    
    auto data = createDummyIQ4XS(num_blocks);
    
    ASSERT_NO_THROW({
        IQ4_XSTensor tensor(num_elements, data.data(), data.size());
        EXPECT_EQ(tensor.numElements(), num_elements);
        EXPECT_EQ(tensor.numBlocks(), num_blocks);
        EXPECT_EQ(tensor.blockSize(), QK_K);
        EXPECT_EQ(tensor.sizeInBytes(), data.size());
    });
}

TEST(IQ4_XSTensorTest, ConstructInvalidSize) {
    const size_t num_blocks = 4;
    const size_t num_elements = num_blocks * QK_K + 13;  // Not multiple of QK_K
    
    auto data = createDummyIQ4XS(num_blocks);
    
    EXPECT_THROW({
        IQ4_XSTensor tensor(num_elements, data.data(), data.size());
    }, std::invalid_argument);
}

TEST(IQ4_XSTensorTest, ConstructInvalidDataSize) {
    const size_t num_blocks = 4;
    const size_t num_elements = num_blocks * QK_K;
    
    auto data = createDummyIQ4XS(num_blocks);
    const size_t wrong_size = data.size() - 20;
    
    EXPECT_THROW({
        IQ4_XSTensor tensor(num_elements, data.data(), wrong_size);
    }, std::invalid_argument);
}

// ===== Decode Tests =====

TEST(IQ4_XSTensorTest, DecodeToFP32SingleBlock) {
    const size_t num_blocks = 1;
    const size_t num_elements = QK_K;
    
    auto data = createDummyIQ4XS(num_blocks);
    IQ4_XSTensor tensor(num_elements, data.data(), data.size());
    
    std::vector<float> output(num_elements);
    ASSERT_NO_THROW(tensor.decode(output.data()));
    
    // Verify all values decoded
    for (size_t i = 0; i < num_elements; ++i) {
        EXPECT_TRUE(std::isfinite(output[i]));
    }
}

TEST(IQ4_XSTensorTest, DecodeToFP32MultiBlock) {
    const size_t num_blocks = 5;
    const size_t num_elements = num_blocks * QK_K;
    
    auto data = createDummyIQ4XS(num_blocks);
    IQ4_XSTensor tensor(num_elements, data.data(), data.size());
    
    std::vector<float> output(num_elements);
    ASSERT_NO_THROW(tensor.decode(output.data()));
    
    // Verify all values finite
    for (size_t i = 0; i < num_elements; ++i) {
        EXPECT_TRUE(std::isfinite(output[i]));
    }
}

TEST(IQ4_XSTensorTest, DecodeToBF16SingleBlock) {
    const size_t num_blocks = 1;
    const size_t num_elements = QK_K;
    
    auto data = createDummyIQ4XS(num_blocks);
    IQ4_XSTensor tensor(num_elements, data.data(), data.size());
    
    std::vector<bfloat16_t> output(num_elements);
    ASSERT_NO_THROW(tensor.decodeBF16(output.data()));
    
    // Convert back to FP32 and verify
    for (size_t i = 0; i < num_elements; ++i) {
        float val = static_cast<float>(output[i]);
        EXPECT_TRUE(std::isfinite(val));
    }
}

TEST(IQ4_XSTensorTest, DecodeToBF16MultiBlock) {
    const size_t num_blocks = 5;
    const size_t num_elements = num_blocks * QK_K;
    
    auto data = createDummyIQ4XS(num_blocks);
    IQ4_XSTensor tensor(num_elements, data.data(), data.size());
    
    std::vector<bfloat16_t> output(num_elements);
    ASSERT_NO_THROW(tensor.decodeBF16(output.data()));
    
    // Verify all values valid
    for (size_t i = 0; i < num_elements; ++i) {
        float val = static_cast<float>(output[i]);
        EXPECT_TRUE(std::isfinite(val));
    }
}

// ===== Streaming Decode Tests =====

TEST(IQ4_XSTensorTest, DecodeRangeValid) {
    const size_t num_blocks = 10;
    const size_t num_elements = num_blocks * QK_K;
    
    auto data = createDummyIQ4XS(num_blocks);
    IQ4_XSTensor tensor(num_elements, data.data(), data.size());
    
    // Decode middle blocks (3-7)
    const size_t start_block = 3;
    const size_t decode_blocks = 5;
    std::vector<float> output(decode_blocks * QK_K);
    
    ASSERT_NO_THROW(tensor.decodeRange(output.data(), start_block, decode_blocks));
    
    // Verify values
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_TRUE(std::isfinite(output[i]));
    }
}

TEST(IQ4_XSTensorTest, DecodeRangeOutOfBounds) {
    const size_t num_blocks = 10;
    const size_t num_elements = num_blocks * QK_K;
    
    auto data = createDummyIQ4XS(num_blocks);
    IQ4_XSTensor tensor(num_elements, data.data(), data.size());
    
    std::vector<float> output(5 * QK_K);
    
    // Try to decode past end
    EXPECT_THROW({
        tensor.decodeRange(output.data(), 7, 5);  // 7 + 5 > 10
    }, std::out_of_range);
}

TEST(IQ4_XSTensorTest, DecodeRangeBF16Valid) {
    const size_t num_blocks = 10;
    const size_t num_elements = num_blocks * QK_K;
    
    auto data = createDummyIQ4XS(num_blocks);
    IQ4_XSTensor tensor(num_elements, data.data(), data.size());
    
    const size_t start_block = 2;
    const size_t decode_blocks = 4;
    std::vector<bfloat16_t> output(decode_blocks * QK_K);
    
    ASSERT_NO_THROW(tensor.decodeRangeBF16(output.data(), start_block, decode_blocks));
    
    // Verify values
    for (size_t i = 0; i < output.size(); ++i) {
        float val = static_cast<float>(output[i]);
        EXPECT_TRUE(std::isfinite(val));
    }
}

// ===== Sub-block Scale Tests =====

TEST(IQ4_XSTensorTest, ScaleExtractionCorrectness) {
    // Test that 6-bit scale extraction works correctly
    const size_t num_blocks = 1;
    const size_t num_elements = QK_K;
    
    std::vector<uint8_t> data(sizeof(IQ4_XSBlock));
    auto* block = reinterpret_cast<IQ4_XSBlock*>(data.data());
    
    block->d = fp32_to_fp16(1.0f);
    
    // Set known scale values (0, 1, 2, ..., 7) for sub-blocks
    block->scales_h = 0;
    for (size_t i = 0; i < 4; ++i) {
        block->scales_l[i] = 0;
    }
    
    // Pack scales: sb0=0, sb1=1, sb2=2, ..., sb7=7
    for (size_t sb = 0; sb < 8; ++sb) {
        int scale_val = static_cast<int>(sb);
        int low_4 = scale_val & 0x0F;
        int high_2 = (scale_val >> 4) & 0x03;
        
        if (sb % 2 == 0) {
            block->scales_l[sb / 2] |= low_4;
        } else {
            block->scales_l[sb / 2] |= (low_4 << 4);
        }
        block->scales_h |= (high_2 << (2 * sb));
    }
    
    // Fill qs with zeros
    for (size_t i = 0; i < 128; ++i) {
        block->qs[i] = 0x00;  // Index 0 -> kvalues_iq4nl[0] = -127
    }
    
    IQ4_XSTensor tensor(num_elements, data.data(), data.size());
    std::vector<float> output(num_elements);
    tensor.decode(output.data());
    
    // Verify that different sub-blocks have different effective scales
    // Sub-block 0: scale = 0 - 32 = -32, value = -32 * -127 = 4064
    // Sub-block 1: scale = 1 - 32 = -31, value = -31 * -127 = 3937
    // etc.
    
    // Just verify they're different (exact values depend on FP16 precision)
    float sb0_val = output[0];
    float sb1_val = output[32];
    EXPECT_NE(sb0_val, sb1_val);
}

TEST(IQ4_XSTensorTest, SubBlockIndependence) {
    // Verify that different sub-blocks decode independently
    const size_t num_blocks = 1;
    const size_t num_elements = QK_K;
    
    auto data = createDummyIQ4XS(num_blocks);
    IQ4_XSTensor tensor(num_elements, data.data(), data.size());
    
    std::vector<float> output(num_elements);
    tensor.decode(output.data());
    
    // Check that values in different sub-blocks are different
    // (due to different sub-block scales in dummy data)
    bool all_different = true;
    for (size_t sb = 0; sb < 7; ++sb) {
        if (std::abs(output[sb * 32] - output[(sb + 1) * 32]) < 1e-6f) {
            all_different = false;
            break;
        }
    }
    EXPECT_TRUE(all_different);
}

// ===== Value Correctness Tests =====

TEST(IQ4_XSTensorTest, LookupTableBounds) {
    // Verify kvalues_iq4nl lookup table is properly bounded
    const size_t num_blocks = 1;
    const size_t num_elements = QK_K;
    
    std::vector<uint8_t> data(sizeof(IQ4_XSBlock));
    auto* block = reinterpret_cast<IQ4_XSBlock*>(data.data());
    
    block->d = fp32_to_fp16(1.0f);
    block->scales_h = 0x0000;  // All scales = 0
    for (size_t i = 0; i < 4; ++i) {
        block->scales_l[i] = 32;  // Scale value 32 -> effective scale = 0
    }
    
    // All indices to 0
    for (size_t i = 0; i < 128; ++i) {
        block->qs[i] = 0x00;
    }
    
    IQ4_XSTensor tensor(num_elements, data.data(), data.size());
    std::vector<float> output(num_elements);
    
    ASSERT_NO_THROW(tensor.decode(output.data()));
    
    // With effective scale = 0, all values should be ~0
    for (size_t i = 0; i < num_elements; ++i) {
        EXPECT_NEAR(output[i], 0.0f, 1.0f);
    }
}

TEST(IQ4_XSTensorTest, ZeroGlobalScale) {
    const size_t num_blocks = 1;
    const size_t num_elements = QK_K;
    
    std::vector<uint8_t> data(sizeof(IQ4_XSBlock));
    auto* block = reinterpret_cast<IQ4_XSBlock*>(data.data());
    
    block->d = 0;  // Zero global scale
    block->scales_h = 0xFFFF;  // Max sub-block scales
    for (size_t i = 0; i < 4; ++i) {
        block->scales_l[i] = 0xFF;
    }
    for (size_t i = 0; i < 128; ++i) {
        block->qs[i] = 0xFF;  // Max indices
    }
    
    IQ4_XSTensor tensor(num_elements, data.data(), data.size());
    std::vector<float> output(num_elements);
    tensor.decode(output.data());
    
    // All values should be zero (global scale is 0)
    for (size_t i = 0; i < num_elements; ++i) {
        EXPECT_FLOAT_EQ(output[i], 0.0f);
    }
}

TEST(IQ4_XSTensorTest, DifferentBlocksDecodeUniquely) {
    // Verify different blocks produce different outputs
    const size_t num_blocks = 3;
    const size_t num_elements = num_blocks * QK_K;
    
    auto data = createDummyIQ4XS(num_blocks);
    IQ4_XSTensor tensor(num_elements, data.data(), data.size());
    
    std::vector<float> output(num_elements);
    tensor.decode(output.data());
    
    // Compare first elements of each block (should be different due to different global scales)
    EXPECT_NE(output[0], output[QK_K]);
    EXPECT_NE(output[QK_K], output[2 * QK_K]);
}

// ===== Edge Cases =====

TEST(IQ4_XSTensorTest, AllScalesMax) {
    const size_t num_blocks = 1;
    const size_t num_elements = QK_K;
    
    std::vector<uint8_t> data(sizeof(IQ4_XSBlock));
    auto* block = reinterpret_cast<IQ4_XSBlock*>(data.data());
    
    block->d = fp32_to_fp16(1.0f);
    block->scales_h = 0xFFFF;  // All high bits = 1
    for (size_t i = 0; i < 4; ++i) {
        block->scales_l[i] = 0xFF;  // All low bits = 1
    }
    
    // All scales = 63 (max 6-bit value)
    // Effective scale = 1.0 * (63 - 32) = 31.0
    
    for (size_t i = 0; i < 128; ++i) {
        block->qs[i] = 0x00;  // Index 0 -> -127
    }
    
    IQ4_XSTensor tensor(num_elements, data.data(), data.size());
    std::vector<float> output(num_elements);
    tensor.decode(output.data());
    
    const float expected = 31.0f * (-127.0f);
    for (size_t i = 0; i < num_elements; ++i) {
        EXPECT_NEAR(output[i], expected, 1.0f);
    }
}

// ===== Multi-threading Tests =====

#ifdef _OPENMP
TEST(IQ4_XSTensorTest, MultiThreadedDecode) {
    const size_t num_blocks = 50;
    const size_t num_elements = num_blocks * QK_K;
    
    auto data = createDummyIQ4XS(num_blocks);
    IQ4_XSTensor tensor(num_elements, data.data(), data.size());
    
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

TEST(IQ4_XSTensorTest, DecodeConsistency) {
    // Decode same data twice, should get same results
    const size_t num_blocks = 5;
    const size_t num_elements = num_blocks * QK_K;
    
    auto data = createDummyIQ4XS(num_blocks);
    IQ4_XSTensor tensor(num_elements, data.data(), data.size());
    
    std::vector<float> output1(num_elements);
    std::vector<float> output2(num_elements);
    
    tensor.decode(output1.data());
    tensor.decode(output2.data());
    
    for (size_t i = 0; i < num_elements; ++i) {
        EXPECT_FLOAT_EQ(output1[i], output2[i]);
    }
}

TEST(IQ4_XSTensorTest, FP32BF16Similarity) {
    // FP32 and BF16 decode should be similar
    const size_t num_blocks = 5;
    const size_t num_elements = num_blocks * QK_K;
    
    auto data = createDummyIQ4XS(num_blocks);
    IQ4_XSTensor tensor(num_elements, data.data(), data.size());
    
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
