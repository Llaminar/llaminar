/**
 * @file test_iq1_s_tensor.cpp
 * @brief Unit tests for IQ1_STensor (1.5625 bpw quantization)
 * 
 * Tests cover:
 * - Basic instantiation and shape handling
 * - Block structure validation (50 bytes)
 * - Decode correctness (small and large tensors)
 * - Streaming decode (decodeRow, decodeSpan)
 * - BF16 decode
 * - Multi-threading (OpenMP)
 * 
 * @author David Sanftenberg
 * @date 2025-10-21
 */

#include "tensors/IQ1_STensor.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <numeric>

using namespace llaminar;

class IQ1_STensorTest : public ::testing::Test {
protected:
    /**
     * @brief Create dummy IQ1_S data for testing
     * 
     * Creates valid IQ1_S blocks with known patterns for verification.
     */
    std::vector<uint8_t> createDummyIQ1S(size_t num_blocks) {
        std::vector<uint8_t> data(num_blocks * sizeof(IQ1_SBlock));
        IQ1_SBlock* blocks = reinterpret_cast<IQ1_SBlock*>(data.data());
        
        for (size_t i = 0; i < num_blocks; ++i) {
            // Set FP16 scale (small positive value)
            blocks[i].d = 0x3C00; // FP16 for 1.0
            
            // Fill qs (grid indices low 8 bits)
            for (size_t j = 0; j < 32; ++j) {
                blocks[i].qs[j] = static_cast<uint8_t>((i * 32 + j) % 256);
            }
            
            // Fill qh (high 3 bits + delta sign + 3-bit scale)
            for (size_t j = 0; j < 8; ++j) {
                uint16_t qh_val = 0;
                // Set 3-bit scale (bits 12-14): value 0-7
                qh_val |= (static_cast<uint16_t>((i + j) % 8) << 12);
                // Set delta sign (bit 15): alternating
                if ((i + j) % 2 == 1) {
                    qh_val |= 0x8000;
                }
                // Set high 3 bits for 4 groups (bits 0-11): pattern
                qh_val |= static_cast<uint16_t>((i * 8 + j) % 4096);
                blocks[i].qh[j] = qh_val;
            }
        }
        
        return data;
    }
};

// ========== Basic Functionality ==========

TEST_F(IQ1_STensorTest, BasicInstantiation) {
    std::vector<int> shape = {2, 256};  // 2 rows, 256 cols = 512 elements = 2 blocks
    auto data = createDummyIQ1S(2);
    
    IQ1_STensor tensor(shape, data);
    
    EXPECT_EQ(tensor.shape()[0], 2);
    EXPECT_EQ(tensor.shape()[1], 256);
    EXPECT_EQ(tensor.size(), 512);
    EXPECT_EQ(tensor.ndim(), 2);
    EXPECT_EQ(tensor.element_count(), 512);
}

TEST_F(IQ1_STensorTest, QuantTypeAndCompression) {
    std::vector<int> shape = {1, 256};
    auto data = createDummyIQ1S(1);
    IQ1_STensor tensor(shape, data);
    
    EXPECT_EQ(tensor.quant_type(), QuantType::IQ1_S);
    EXPECT_FLOAT_EQ(tensor.compression_ratio(), 20.48f);
}

TEST_F(IQ1_STensorTest, BlockDescriptor) {
    std::vector<int> shape = {1, 256};
    auto data = createDummyIQ1S(1);
    IQ1_STensor tensor(shape, data);
    
    const auto& desc = tensor.block_descriptor();
    EXPECT_EQ(desc.elements_per_block, 256);
    EXPECT_EQ(desc.bytes_per_block, 50);
    EXPECT_EQ(desc.scale_count, 1);
    EXPECT_EQ(desc.bits_per_value, 2);  // 1.5625 bpw rounded
    EXPECT_FALSE(desc.is_k_quant);
}

TEST_F(IQ1_STensorTest, InvalidShapeThrows) {
    std::vector<int> shape_1d = {256};  // 1D not supported
    auto data = createDummyIQ1S(1);
    
    EXPECT_THROW({
        IQ1_STensor tensor(shape_1d, data);
    }, std::invalid_argument);
}

TEST_F(IQ1_STensorTest, DataSizeMismatch) {
    std::vector<int> shape = {1, 256};  // Expects 50 bytes
    std::vector<uint8_t> wrong_size_data(30);  // Wrong size
    
    EXPECT_THROW({
        IQ1_STensor tensor(shape, wrong_size_data);
    }, std::invalid_argument);
}

// ========== Decode Functionality ==========

TEST_F(IQ1_STensorTest, DecodeToFP32Small) {
    std::vector<int> shape = {1, 256};
    auto data = createDummyIQ1S(1);
    IQ1_STensor tensor(shape, data);
    
    std::vector<float> output(256);
    tensor.decode_to_fp32(output.data());
    
    // Check that decode produces finite values
    for (size_t i = 0; i < 256; ++i) {
        EXPECT_TRUE(std::isfinite(output[i])) << "Non-finite value at index " << i;
    }
}

TEST_F(IQ1_STensorTest, DecodeToFP32Large) {
    std::vector<int> shape = {16, 256};  // 16 blocks
    auto data = createDummyIQ1S(16);
    IQ1_STensor tensor(shape, data);
    
    std::vector<float> output(16 * 256);
    tensor.decode_to_fp32(output.data());
    
    // Check first and last values
    EXPECT_TRUE(std::isfinite(output[0]));
    EXPECT_TRUE(std::isfinite(output[16 * 256 - 1]));
}

TEST_F(IQ1_STensorTest, DecodeRowSingleBlock) {
    std::vector<int> shape = {2, 256};  // 2 rows
    auto data = createDummyIQ1S(2);
    IQ1_STensor tensor(shape, data);
    
    std::vector<float> row0(256);
    std::vector<float> row1(256);
    
    tensor.decodeRow(0, row0.data());
    tensor.decodeRow(1, row1.data());
    
    // Rows should have different values (different blocks)
    bool different = false;
    for (size_t i = 0; i < 256; ++i) {
        if (std::abs(row0[i] - row1[i]) > 1e-6f) {
            different = true;
            break;
        }
    }
    EXPECT_TRUE(different) << "Rows should decode differently";
}

TEST_F(IQ1_STensorTest, DecodeRowMultiBlock) {
    std::vector<int> shape = {1, 512};  // Spans 2 blocks
    auto data = createDummyIQ1S(2);
    IQ1_STensor tensor(shape, data);
    
    std::vector<float> row(512);
    tensor.decodeRow(0, row.data());
    
    // Check values are finite
    for (size_t i = 0; i < 512; ++i) {
        EXPECT_TRUE(std::isfinite(row[i]));
    }
}

TEST_F(IQ1_STensorTest, DecodeSpanPartialBlock) {
    std::vector<int> shape = {1, 256};
    auto data = createDummyIQ1S(1);
    IQ1_STensor tensor(shape, data);
    
    std::vector<float> output(128);
    tensor.decodeSpan(64, 128, output.data());  // Middle 128 elements
    
    for (size_t i = 0; i < 128; ++i) {
        EXPECT_TRUE(std::isfinite(output[i]));
    }
}

TEST_F(IQ1_STensorTest, DecodeSpanMultiBlock) {
    std::vector<int> shape = {1, 512};  // 2 blocks
    auto data = createDummyIQ1S(2);
    IQ1_STensor tensor(shape, data);
    
    std::vector<float> output(384);
    tensor.decodeSpan(64, 384, output.data());  // Spans across block boundary
    
    for (size_t i = 0; i < 384; ++i) {
        EXPECT_TRUE(std::isfinite(output[i]));
    }
}

TEST_F(IQ1_STensorTest, DecodeSpanOutOfBounds) {
    std::vector<int> shape = {1, 256};
    auto data = createDummyIQ1S(1);
    IQ1_STensor tensor(shape, data);
    
    std::vector<float> output(100);
    EXPECT_THROW({
        tensor.decodeSpan(200, 100, output.data());  // Exceeds tensor bounds
    }, std::out_of_range);
}

// ========== BF16 Decode ==========

TEST_F(IQ1_STensorTest, DecodeToBF16) {
    std::vector<int> shape = {1, 256};
    auto data = createDummyIQ1S(1);
    IQ1_STensor tensor(shape, data);
    
    std::vector<bfloat16> output(256);
    tensor.decode_to_bf16(output.data());
    
    // Check that values are finite
    for (size_t i = 0; i < 256; ++i) {
        float fp32_val = static_cast<float>(output[i]);
        EXPECT_TRUE(std::isfinite(fp32_val));
    }
}

// ========== Copy and Metadata ==========

TEST_F(IQ1_STensorTest, CopyTensor) {
    std::vector<int> shape = {1, 256};
    auto data = createDummyIQ1S(1);
    IQ1_STensor tensor(shape, data);
    
    auto copied = tensor.copy();
    ASSERT_NE(copied, nullptr);
    
    EXPECT_EQ(copied->shape()[0], 1);
    EXPECT_EQ(copied->shape()[1], 256);
    EXPECT_EQ(copied->size(), 256);
}

TEST_F(IQ1_STensorTest, RawDataAccess) {
    std::vector<int> shape = {1, 256};
    auto data = createDummyIQ1S(1);
    IQ1_STensor tensor(shape, data);
    
    const uint8_t* raw = tensor.raw_data();
    size_t raw_size = tensor.raw_size();
    
    EXPECT_NE(raw, nullptr);
    EXPECT_EQ(raw_size, sizeof(IQ1_SBlock));
}

// ========== Multi-threading ==========

TEST_F(IQ1_STensorTest, MultiThreadedDecode) {
    std::vector<int> shape = {8, 256};  // 8 rows (triggers OpenMP)
    auto data = createDummyIQ1S(8);
    IQ1_STensor tensor(shape, data);
    
    std::vector<float> output(8 * 256);
    tensor.decode_to_fp32(output.data());
    
    // Check all values are finite
    for (size_t i = 0; i < 8 * 256; ++i) {
        EXPECT_TRUE(std::isfinite(output[i]));
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
