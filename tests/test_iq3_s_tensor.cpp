/**
 * @file test_iq3_s_tensor.cpp
 * @brief Unit tests for IQ3_STensor (3.4375 bpw quantization)
 * 
 * Tests cover:
 * - Basic instantiation and shape handling
 * - Block structure validation (110 bytes)
 * - Decode correctness (small and large tensors)
 * - Streaming decode (decodeRow, decodeSpan)
 * - BF16 decode
 * - Multi-threading (OpenMP)
 * 
 * @author David Sanftenberg
 * @date 2025-10-21
 */

#include "tensors/IQ3_STensor.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <numeric>

using namespace llaminar;

class IQ3_STensorTest : public ::testing::Test {
protected:
    /**
     * @brief Create dummy IQ3_S data for testing
     * 
     * Creates valid IQ3_S blocks with known patterns for verification.
     */
    std::vector<uint8_t> createDummyIQ3S(size_t num_blocks) {
        std::vector<uint8_t> data(num_blocks * sizeof(IQ3_SBlock));
        IQ3_SBlock* blocks = reinterpret_cast<IQ3_SBlock*>(data.data());
        
        for (size_t i = 0; i < num_blocks; ++i) {
            // Set FP16 scale (small positive value)
            blocks[i].d = 0x3C00; // FP16 for 1.0
            
            // Fill qs with pattern
            for (size_t j = 0; j < 64; ++j) {
                blocks[i].qs[j] = static_cast<uint8_t>((i * 64 + j) % 256);
            }
            
            // Fill qh (high bits)
            for (size_t j = 0; j < 8; ++j) {
                blocks[i].qh[j] = static_cast<uint8_t>((i * 8 + j) % 2);
            }
            
            // Fill signs
            for (size_t j = 0; j < 32; ++j) {
                blocks[i].signs[j] = static_cast<uint8_t>((i * 32 + j) % 256);
            }
            
            // Fill scales (4-bit values)
            for (size_t j = 0; j < IQ3_SBlock::N_SCALE; ++j) {
                blocks[i].scales[j] = static_cast<uint8_t>((i * 4 + j) % 16);
            }
        }
        
        return data;
    }
};

// ========== Basic Functionality ==========

TEST_F(IQ3_STensorTest, BasicInstantiation) {
    std::vector<int> shape = {2, 256};  // 2 rows, 256 cols = 512 elements = 2 blocks
    auto data = createDummyIQ3S(2);
    
    IQ3_STensor tensor(shape, data);
    
    EXPECT_EQ(tensor.shape()[0], 2);
    EXPECT_EQ(tensor.shape()[1], 256);
    EXPECT_EQ(tensor.size(), 512);
    EXPECT_EQ(tensor.ndim(), 2);
    EXPECT_EQ(tensor.element_count(), 512);
}

TEST_F(IQ3_STensorTest, QuantTypeAndCompression) {
    std::vector<int> shape = {1, 256};
    auto data = createDummyIQ3S(1);
    IQ3_STensor tensor(shape, data);
    
    EXPECT_EQ(tensor.quant_type(), QuantType::IQ3_S);
    EXPECT_FLOAT_EQ(tensor.compression_ratio(), 9.29f);
}

TEST_F(IQ3_STensorTest, BlockDescriptor) {
    std::vector<int> shape = {1, 256};
    auto data = createDummyIQ3S(1);
    IQ3_STensor tensor(shape, data);
    
    const auto& desc = tensor.block_descriptor();
    EXPECT_EQ(desc.elements_per_block, 256);
    EXPECT_EQ(desc.bytes_per_block, 110);
    EXPECT_EQ(desc.scale_count, 4);
    EXPECT_EQ(desc.bits_per_value, 3);
    EXPECT_FALSE(desc.is_k_quant);
}

TEST_F(IQ3_STensorTest, InvalidShapeThrows) {
    std::vector<int> shape_1d = {256};
    auto data = createDummyIQ3S(1);
    
    EXPECT_THROW(IQ3_STensor(shape_1d, data), std::invalid_argument);
}

TEST_F(IQ3_STensorTest, DataSizeMismatchThrows) {
    std::vector<int> shape = {2, 256};  // 2 blocks needed
    auto data = createDummyIQ3S(1);     // Only 1 block provided
    
    EXPECT_THROW(IQ3_STensor(shape, data), std::invalid_argument);
}

// ========== Decode Correctness ==========

TEST_F(IQ3_STensorTest, DecodeSmallTensor) {
    std::vector<int> shape = {1, 256};  // Single block
    auto data = createDummyIQ3S(1);
    IQ3_STensor tensor(shape, data);
    
    std::vector<float> decoded(256);
    tensor.decode_to_fp32(decoded.data());
    
    // Verify decode produced valid floats (not NaN/Inf)
    for (size_t i = 0; i < decoded.size(); ++i) {
        EXPECT_TRUE(std::isfinite(decoded[i])) << "Element " << i << " is not finite";
    }
    
    // IQ3_S values are quantized, so we just check range is reasonable
    float max_val = *std::max_element(decoded.begin(), decoded.end());
    float min_val = *std::min_element(decoded.begin(), decoded.end());
    EXPECT_LT(std::abs(max_val), 300.0f) << "Max value suspiciously large";
    EXPECT_LT(std::abs(min_val), 300.0f) << "Min value suspiciously large";
}

TEST_F(IQ3_STensorTest, DecodeLargeTensor) {
    std::vector<int> shape = {8, 512};  // 4096 elements = 16 blocks
    auto data = createDummyIQ3S(16);
    IQ3_STensor tensor(shape, data);
    
    std::vector<float> decoded(4096);
    tensor.decode_to_fp32(decoded.data());
    
    // Verify all values finite
    for (size_t i = 0; i < decoded.size(); ++i) {
        EXPECT_TRUE(std::isfinite(decoded[i])) << "Element " << i << " is not finite";
    }
}

// ========== Streaming Decode ==========

TEST_F(IQ3_STensorTest, DecodeRowSingleBlock) {
    std::vector<int> shape = {2, 256};  // 2 rows, 256 cols
    auto data = createDummyIQ3S(2);
    IQ3_STensor tensor(shape, data);
    
    std::vector<float> row0(256);
    std::vector<float> row1(256);
    tensor.decodeRow(0, row0.data());
    tensor.decodeRow(1, row1.data());
    
    // Verify both rows decoded successfully
    for (size_t i = 0; i < 256; ++i) {
        EXPECT_TRUE(std::isfinite(row0[i]));
        EXPECT_TRUE(std::isfinite(row1[i]));
    }
}

TEST_F(IQ3_STensorTest, DecodeSpanWithinBlock) {
    std::vector<int> shape = {1, 256};
    auto data = createDummyIQ3S(1);
    IQ3_STensor tensor(shape, data);
    
    std::vector<float> span(64);
    tensor.decodeSpan(64, 64, span.data());  // Elements 64-127
    
    for (size_t i = 0; i < span.size(); ++i) {
        EXPECT_TRUE(std::isfinite(span[i]));
    }
}

TEST_F(IQ3_STensorTest, DecodeSpanAcrossBlocks) {
    std::vector<int> shape = {2, 256};  // 512 elements, 2 blocks
    auto data = createDummyIQ3S(2);
    IQ3_STensor tensor(shape, data);
    
    std::vector<float> span(128);
    tensor.decodeSpan(192, 128, span.data());  // Cross block boundary at 256
    
    for (size_t i = 0; i < span.size(); ++i) {
        EXPECT_TRUE(std::isfinite(span[i]));
    }
}

TEST_F(IQ3_STensorTest, DecodeSpanOutOfRangeThrows) {
    std::vector<int> shape = {1, 256};
    auto data = createDummyIQ3S(1);
    IQ3_STensor tensor(shape, data);
    
    std::vector<float> span(128);
    EXPECT_THROW(tensor.decodeSpan(200, 128, span.data()), std::out_of_range);
}

// ========== BF16 Decode ==========

TEST_F(IQ3_STensorTest, DecodeToBF16) {
    std::vector<int> shape = {1, 256};
    auto data = createDummyIQ3S(1);
    IQ3_STensor tensor(shape, data);
    
    std::vector<bfloat16> decoded(256);
    tensor.decode_to_bf16(decoded.data());
    
    // Convert back to FP32 for validation
    for (size_t i = 0; i < decoded.size(); ++i) {
        float fp32_val = static_cast<float>(decoded[i]);
        EXPECT_TRUE(std::isfinite(fp32_val)) << "BF16 element " << i << " invalid";
    }
}

// ========== Multi-threading ==========

TEST_F(IQ3_STensorTest, MultiThreadDecode) {
    std::vector<int> shape = {16, 256};  // 16 rows triggers OMP (threshold = 4)
    auto data = createDummyIQ3S(16);
    IQ3_STensor tensor(shape, data);
    
    std::vector<float> decoded(4096);
    tensor.decode_to_fp32(decoded.data());
    
    // Verify all elements decoded correctly
    for (size_t i = 0; i < decoded.size(); ++i) {
        EXPECT_TRUE(std::isfinite(decoded[i]));
    }
}

// ========== Copy Operations ==========

TEST_F(IQ3_STensorTest, CopyTensor) {
    std::vector<int> shape = {1, 256};
    auto data = createDummyIQ3S(1);
    IQ3_STensor tensor(shape, data);
    
    auto copy = tensor.copy();
    ASSERT_NE(copy, nullptr);
    
    EXPECT_EQ(copy->shape(), tensor.shape());
    EXPECT_EQ(copy->size(), tensor.size());
}

TEST_F(IQ3_STensorTest, CopyFromThrows) {
    std::vector<int> shape = {1, 256};
    auto data = createDummyIQ3S(1);
    IQ3_STensor tensor1(shape, data);
    IQ3_STensor tensor2(shape, data);
    
    // copy_from not supported for quantized tensors
    EXPECT_THROW(tensor1.copy_from(tensor2), std::runtime_error);
}

// ========== Main ==========

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
