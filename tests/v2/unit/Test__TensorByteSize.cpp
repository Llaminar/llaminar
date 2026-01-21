/**
 * @file Test__TensorByteSize.cpp
 * @brief Unit tests verifying size_bytes() implementation for all tensor types
 * @author David Sanftenberg
 * @date 2026-01-20
 *
 * This test file verifies that each tensor type in the Llaminar V2 codebase properly
 * implements the byte_size() method via the public size_bytes() interface.
 * This is critical because the base class TensorBase::byte_size() throws an
 * exception if not overridden, which causes GPU transfer failures.
 *
 * Tests cover:
 * - All 27 tensor types defined in CPUTensors.h
 * - Verification that size_bytes() doesn't throw
 * - Verification that size_bytes() matches expected calculations
 * - For block-quantized tensors: size_bytes matches num_blocks * sizeof(BlockType)
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "tensors/TensorClasses.h"
#include "tensors/TensorFactory.h"

using namespace llaminar2;

// ============================================================================
// Test Fixture
// ============================================================================

class Test__TensorByteSize : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Clean state for each test
    }

    void TearDown() override
    {
        // Cleanup if needed
    }
};

// ============================================================================
// Simple Format Tensors (element-based byte_size)
// ============================================================================

/**
 * @brief FP32Tensor: 4 bytes per element
 * byte_size() returns element_count() * sizeof(float)
 */
TEST_F(Test__TensorByteSize, FP32Tensor_HasCorrectByteSize)
{
    // Create tensor with known dimensions: 32 rows x 64 cols = 2048 elements
    FP32Tensor tensor({32, 64});

    // Verify size_bytes() is implemented (doesn't throw)
    size_t byte_size = 0;
    ASSERT_NO_THROW(byte_size = tensor.size_bytes());

    // Verify byte_size is accurate: 2048 * 4 = 8192 bytes
    const size_t expected = 32 * 64 * sizeof(float);
    EXPECT_EQ(byte_size, expected);
    EXPECT_EQ(byte_size, 8192);
    EXPECT_GT(byte_size, 0);
}

/**
 * @brief FP16Tensor: 2 bytes per element
 * byte_size() returns element_count() * sizeof(uint16_t)
 */
TEST_F(Test__TensorByteSize, FP16Tensor_HasCorrectByteSize)
{
    // 4x4 matrix = 16 elements
    std::vector<uint16_t> data(16, 0x3C00); // FP16 for 1.0
    FP16Tensor tensor({4, 4}, data);

    size_t byte_size = 0;
    ASSERT_NO_THROW(byte_size = tensor.size_bytes());

    // 16 * 2 = 32 bytes
    const size_t expected = 16 * sizeof(uint16_t);
    EXPECT_EQ(byte_size, expected);
    EXPECT_EQ(byte_size, 32);
    EXPECT_GT(byte_size, 0);
}

/**
 * @brief BF16Tensor: 2 bytes per element
 * byte_size() returns element_count() * sizeof(uint16_t)
 */
TEST_F(Test__TensorByteSize, BF16Tensor_HasCorrectByteSize)
{
    // 8x8 matrix = 64 elements
    std::vector<uint16_t> data(64, 0x3F80); // BF16 for 1.0
    BF16Tensor tensor({8, 8}, data);

    size_t byte_size = 0;
    ASSERT_NO_THROW(byte_size = tensor.size_bytes());

    // 64 * 2 = 128 bytes
    const size_t expected = 64 * sizeof(uint16_t);
    EXPECT_EQ(byte_size, expected);
    EXPECT_EQ(byte_size, 128);
    EXPECT_GT(byte_size, 0);
}

/**
 * @brief INT8Tensor: 1 byte per element
 * byte_size() returns element_count() * sizeof(int8_t)
 */
TEST_F(Test__TensorByteSize, INT8Tensor_HasCorrectByteSize)
{
    // 16x32 matrix = 512 elements
    INT8Tensor tensor({16, 32});

    size_t byte_size = 0;
    ASSERT_NO_THROW(byte_size = tensor.size_bytes());

    // 512 * 1 = 512 bytes
    const size_t expected = 512 * sizeof(int8_t);
    EXPECT_EQ(byte_size, expected);
    EXPECT_EQ(byte_size, 512);
    EXPECT_GT(byte_size, 0);
}

/**
 * @brief INT32Tensor: 4 bytes per element
 * byte_size() returns element_count() * sizeof(int32_t)
 */
TEST_F(Test__TensorByteSize, INT32Tensor_HasCorrectByteSize)
{
    // 8x16 matrix = 128 elements
    INT32Tensor tensor({8, 16});

    size_t byte_size = 0;
    ASSERT_NO_THROW(byte_size = tensor.size_bytes());

    // 128 * 4 = 512 bytes
    const size_t expected = 128 * sizeof(int32_t);
    EXPECT_EQ(byte_size, expected);
    EXPECT_EQ(byte_size, 512);
    EXPECT_GT(byte_size, 0);
}

// ============================================================================
// Simple Block-Quantized Tensors (32 elements per block)
// ============================================================================

/**
 * @brief IQ4_NLTensor: 32 elements per block, 18 bytes per block
 * byte_size() returns raw_data_.size()
 */
TEST_F(Test__TensorByteSize, IQ4_NLTensor_HasCorrectByteSize)
{
    // 4 rows x 64 cols = 256 elements = 8 blocks (64 cols / 32 elements per block = 2 blocks per row)
    const size_t rows = 4;
    const size_t cols = 64; // Must be multiple of 32
    const size_t elements = rows * cols;
    const size_t num_blocks = elements / IQ4_NLBlock::BLOCK_SIZE;
    const size_t expected_bytes = num_blocks * sizeof(IQ4_NLBlock);

    // Create raw data buffer
    std::vector<uint8_t> raw_data(expected_bytes, 0);
    IQ4_NLTensor tensor({rows, cols}, raw_data);

    size_t byte_size = 0;
    ASSERT_NO_THROW(byte_size = tensor.size_bytes());

    // 8 blocks * 18 bytes = 144 bytes
    EXPECT_EQ(byte_size, expected_bytes);
    EXPECT_EQ(byte_size, 8 * 18);
    EXPECT_GT(byte_size, 0);
}

/**
 * @brief Q8_0Tensor: 32 elements per block, 34 bytes per block
 * byte_size() returns raw_data_.size()
 */
TEST_F(Test__TensorByteSize, Q8_0Tensor_HasCorrectByteSize)
{
    const size_t rows = 4;
    const size_t cols = 64; // 2 blocks per row
    const size_t elements = rows * cols;
    const size_t num_blocks = elements / Q8_0Block::BLOCK_SIZE;
    const size_t expected_bytes = num_blocks * sizeof(Q8_0Block);

    std::vector<uint8_t> raw_data(expected_bytes, 0);
    Q8_0Tensor tensor({rows, cols}, raw_data);

    size_t byte_size = 0;
    ASSERT_NO_THROW(byte_size = tensor.size_bytes());

    // 8 blocks * 34 bytes = 272 bytes
    EXPECT_EQ(byte_size, expected_bytes);
    EXPECT_EQ(byte_size, 8 * 34);
    EXPECT_GT(byte_size, 0);
}

/**
 * @brief Q8_1Tensor: 32 elements per block, 36 bytes per block
 * byte_size() returns raw_data_.size()
 */
TEST_F(Test__TensorByteSize, Q8_1Tensor_HasCorrectByteSize)
{
    // For Q8_1Tensor, we use the shape-based constructor
    const size_t rows = 4;
    const size_t cols = 64; // 2 blocks per row
    Q8_1Tensor tensor({rows, cols});

    size_t byte_size = 0;
    ASSERT_NO_THROW(byte_size = tensor.size_bytes());

    // 8 blocks * 36 bytes = 288 bytes
    const size_t num_blocks = (rows * cols) / Q8_1Block::BLOCK_SIZE;
    const size_t expected_bytes = num_blocks * sizeof(Q8_1Block);
    EXPECT_EQ(byte_size, expected_bytes);
    EXPECT_EQ(byte_size, 8 * 36);
    EXPECT_GT(byte_size, 0);
}

/**
 * @brief Q16_1Tensor: Variable block size (32, 64, or 128), default is 64
 * byte_size() returns raw_data_.size()
 */
TEST_F(Test__TensorByteSize, Q16_1Tensor_HasCorrectByteSize_Block64)
{
    // Use block size 64 (default): 4 rows x 64 cols = 256 elements = 4 blocks
    const size_t rows = 4;
    const size_t cols = 64; // 1 block per row with BLOCK_64
    Q16_1Tensor tensor({rows, cols}, Q16BlockSize::BLOCK_64);

    size_t byte_size = 0;
    ASSERT_NO_THROW(byte_size = tensor.size_bytes());

    // 4 blocks * 136 bytes (Q16_1Block_64) = 544 bytes
    const size_t num_blocks = (rows * cols) / static_cast<size_t>(Q16BlockSize::BLOCK_64);
    const size_t expected_bytes = num_blocks * sizeof(Q16_1Block_64);
    EXPECT_EQ(byte_size, expected_bytes);
    EXPECT_EQ(byte_size, 4 * 136);
    EXPECT_GT(byte_size, 0);
}

TEST_F(Test__TensorByteSize, Q16_1Tensor_HasCorrectByteSize_Block32)
{
    // Use block size 32: 4 rows x 64 cols = 256 elements = 8 blocks
    const size_t rows = 4;
    const size_t cols = 64; // 2 blocks per row with BLOCK_32
    Q16_1Tensor tensor({rows, cols}, Q16BlockSize::BLOCK_32);

    size_t byte_size = 0;
    ASSERT_NO_THROW(byte_size = tensor.size_bytes());

    // 8 blocks * 72 bytes (Q16_1Block) = 576 bytes
    const size_t num_blocks = (rows * cols) / static_cast<size_t>(Q16BlockSize::BLOCK_32);
    const size_t expected_bytes = num_blocks * sizeof(Q16_1Block);
    EXPECT_EQ(byte_size, expected_bytes);
    EXPECT_EQ(byte_size, 8 * 72);
    EXPECT_GT(byte_size, 0);
}

TEST_F(Test__TensorByteSize, Q16_1Tensor_HasCorrectByteSize_Block128)
{
    // Use block size 128: 4 rows x 128 cols = 512 elements = 4 blocks
    const size_t rows = 4;
    const size_t cols = 128; // 1 block per row with BLOCK_128
    Q16_1Tensor tensor({rows, cols}, Q16BlockSize::BLOCK_128);

    size_t byte_size = 0;
    ASSERT_NO_THROW(byte_size = tensor.size_bytes());

    // 4 blocks * 264 bytes (Q16_1Block_128) = 1056 bytes
    const size_t num_blocks = (rows * cols) / static_cast<size_t>(Q16BlockSize::BLOCK_128);
    const size_t expected_bytes = num_blocks * sizeof(Q16_1Block_128);
    EXPECT_EQ(byte_size, expected_bytes);
    EXPECT_EQ(byte_size, 4 * 264);
    EXPECT_GT(byte_size, 0);
}

/**
 * @brief Q4_0Tensor: 32 elements per block, 18 bytes per block
 * byte_size() returns raw_data_.size()
 */
TEST_F(Test__TensorByteSize, Q4_0Tensor_HasCorrectByteSize)
{
    const size_t rows = 4;
    const size_t cols = 64; // 2 blocks per row
    const size_t elements = rows * cols;
    const size_t num_blocks = elements / Q4_0Block::BLOCK_SIZE;
    const size_t expected_bytes = num_blocks * sizeof(Q4_0Block);

    std::vector<uint8_t> raw_data(expected_bytes, 0);
    Q4_0Tensor tensor({rows, cols}, raw_data);

    size_t byte_size = 0;
    ASSERT_NO_THROW(byte_size = tensor.size_bytes());

    // 8 blocks * 18 bytes = 144 bytes
    EXPECT_EQ(byte_size, expected_bytes);
    EXPECT_EQ(byte_size, 8 * 18);
    EXPECT_GT(byte_size, 0);
}

/**
 * @brief Q4_1Tensor: 32 elements per block, 20 bytes per block
 * byte_size() returns raw_data_.size()
 */
TEST_F(Test__TensorByteSize, Q4_1Tensor_HasCorrectByteSize)
{
    const size_t rows = 4;
    const size_t cols = 64;
    const size_t elements = rows * cols;
    const size_t num_blocks = elements / Q4_1Block::BLOCK_SIZE;
    const size_t expected_bytes = num_blocks * sizeof(Q4_1Block);

    std::vector<uint8_t> raw_data(expected_bytes, 0);
    Q4_1Tensor tensor({rows, cols}, raw_data);

    size_t byte_size = 0;
    ASSERT_NO_THROW(byte_size = tensor.size_bytes());

    // 8 blocks * 20 bytes = 160 bytes
    EXPECT_EQ(byte_size, expected_bytes);
    EXPECT_EQ(byte_size, 8 * 20);
    EXPECT_GT(byte_size, 0);
}

/**
 * @brief Q5_0Tensor: 32 elements per block, 22 bytes per block
 * byte_size() returns raw_data_.size()
 */
TEST_F(Test__TensorByteSize, Q5_0Tensor_HasCorrectByteSize)
{
    const size_t rows = 4;
    const size_t cols = 64;
    const size_t elements = rows * cols;
    const size_t num_blocks = elements / Q5_0Block::BLOCK_SIZE;
    const size_t expected_bytes = num_blocks * sizeof(Q5_0Block);

    std::vector<uint8_t> raw_data(expected_bytes, 0);
    Q5_0Tensor tensor({rows, cols}, raw_data);

    size_t byte_size = 0;
    ASSERT_NO_THROW(byte_size = tensor.size_bytes());

    // 8 blocks * 22 bytes = 176 bytes
    EXPECT_EQ(byte_size, expected_bytes);
    EXPECT_EQ(byte_size, 8 * 22);
    EXPECT_GT(byte_size, 0);
}

/**
 * @brief Q5_1Tensor: 32 elements per block, 24 bytes per block
 * byte_size() returns raw_data_.size()
 */
TEST_F(Test__TensorByteSize, Q5_1Tensor_HasCorrectByteSize)
{
    const size_t rows = 4;
    const size_t cols = 64;
    const size_t elements = rows * cols;
    const size_t num_blocks = elements / Q5_1Block::BLOCK_SIZE;
    const size_t expected_bytes = num_blocks * sizeof(Q5_1Block);

    std::vector<uint8_t> raw_data(expected_bytes, 0);
    Q5_1Tensor tensor({rows, cols}, raw_data);

    size_t byte_size = 0;
    ASSERT_NO_THROW(byte_size = tensor.size_bytes());

    // 8 blocks * 24 bytes = 192 bytes
    EXPECT_EQ(byte_size, expected_bytes);
    EXPECT_EQ(byte_size, 8 * 24);
    EXPECT_GT(byte_size, 0);
}

// ============================================================================
// K-Quant Formats (256 elements per super-block)
// ============================================================================

/**
 * @brief Q6_KTensor: 256 elements per super-block, 210 bytes per block
 * byte_size() returns raw_data_.size()
 */
TEST_F(Test__TensorByteSize, Q6_KTensor_HasCorrectByteSize)
{
    const size_t rows = 4;
    const size_t cols = 256; // 1 super-block per row
    const size_t elements = rows * cols;
    const size_t num_blocks = elements / Q6_KBlock::BLOCK_SIZE;
    const size_t expected_bytes = num_blocks * sizeof(Q6_KBlock);

    std::vector<uint8_t> raw_data(expected_bytes, 0);
    Q6_KTensor tensor({rows, cols}, raw_data);

    size_t byte_size = 0;
    ASSERT_NO_THROW(byte_size = tensor.size_bytes());

    // 4 blocks * 210 bytes = 840 bytes
    EXPECT_EQ(byte_size, expected_bytes);
    EXPECT_EQ(byte_size, 4 * 210);
    EXPECT_GT(byte_size, 0);
}

/**
 * @brief Q2_KTensor: 256 elements per super-block, 84 bytes per block
 * byte_size() returns raw_data_.size()
 */
TEST_F(Test__TensorByteSize, Q2_KTensor_HasCorrectByteSize)
{
    const size_t rows = 4;
    const size_t cols = 256;
    const size_t elements = rows * cols;
    const size_t num_blocks = elements / Q2_KBlock::BLOCK_SIZE;
    const size_t expected_bytes = num_blocks * sizeof(Q2_KBlock);

    std::vector<uint8_t> raw_data(expected_bytes, 0);
    Q2_KTensor tensor({rows, cols}, raw_data);

    size_t byte_size = 0;
    ASSERT_NO_THROW(byte_size = tensor.size_bytes());

    // 4 blocks * 84 bytes = 336 bytes
    EXPECT_EQ(byte_size, expected_bytes);
    EXPECT_EQ(byte_size, 4 * 84);
    EXPECT_GT(byte_size, 0);
}

/**
 * @brief Q5_KTensor: 256 elements per super-block, 176 bytes per block
 * byte_size() returns raw_data_.size()
 */
TEST_F(Test__TensorByteSize, Q5_KTensor_HasCorrectByteSize)
{
    const size_t rows = 4;
    const size_t cols = 256;
    const size_t elements = rows * cols;
    const size_t num_blocks = elements / Q5_KBlock::BLOCK_SIZE;
    const size_t expected_bytes = num_blocks * sizeof(Q5_KBlock);

    std::vector<uint8_t> raw_data(expected_bytes, 0);
    Q5_KTensor tensor({rows, cols}, raw_data);

    size_t byte_size = 0;
    ASSERT_NO_THROW(byte_size = tensor.size_bytes());

    // 4 blocks * 176 bytes = 704 bytes
    EXPECT_EQ(byte_size, expected_bytes);
    EXPECT_EQ(byte_size, 4 * 176);
    EXPECT_GT(byte_size, 0);
}

/**
 * @brief Q3_KTensor: 256 elements per super-block, 110 bytes per block
 * byte_size() returns raw_data_.size()
 */
TEST_F(Test__TensorByteSize, Q3_KTensor_HasCorrectByteSize)
{
    const size_t rows = 4;
    const size_t cols = 256;
    const size_t elements = rows * cols;
    const size_t num_blocks = elements / Q3_KBlock::BLOCK_SIZE;
    const size_t expected_bytes = num_blocks * sizeof(Q3_KBlock);

    std::vector<uint8_t> raw_data(expected_bytes, 0);
    Q3_KTensor tensor({rows, cols}, raw_data);

    size_t byte_size = 0;
    ASSERT_NO_THROW(byte_size = tensor.size_bytes());

    // 4 blocks * 110 bytes = 440 bytes
    EXPECT_EQ(byte_size, expected_bytes);
    EXPECT_EQ(byte_size, 4 * 110);
    EXPECT_GT(byte_size, 0);
}

/**
 * @brief Q4_KTensor: 256 elements per super-block, 144 bytes per block
 * byte_size() returns raw_data_.size()
 */
TEST_F(Test__TensorByteSize, Q4_KTensor_HasCorrectByteSize)
{
    const size_t rows = 4;
    const size_t cols = 256;
    const size_t elements = rows * cols;
    const size_t num_blocks = elements / Q4_KBlock::BLOCK_SIZE;
    const size_t expected_bytes = num_blocks * sizeof(Q4_KBlock);

    std::vector<uint8_t> raw_data(expected_bytes, 0);
    Q4_KTensor tensor({rows, cols}, raw_data);

    size_t byte_size = 0;
    ASSERT_NO_THROW(byte_size = tensor.size_bytes());

    // 4 blocks * 144 bytes = 576 bytes
    EXPECT_EQ(byte_size, expected_bytes);
    EXPECT_EQ(byte_size, 4 * 144);
    EXPECT_GT(byte_size, 0);
}

/**
 * @brief Q8_KTensor: 256 elements per super-block, 288 bytes per block
 * byte_size() returns raw_data_.size()
 */
TEST_F(Test__TensorByteSize, Q8_KTensor_HasCorrectByteSize)
{
    const size_t rows = 4;
    const size_t cols = 256;
    const size_t elements = rows * cols;
    const size_t num_blocks = elements / Q8_KBlock::BLOCK_SIZE;
    const size_t expected_bytes = num_blocks * sizeof(Q8_KBlock);

    std::vector<uint8_t> raw_data(expected_bytes, 0);
    Q8_KTensor tensor({rows, cols}, raw_data);

    size_t byte_size = 0;
    ASSERT_NO_THROW(byte_size = tensor.size_bytes());

    // 4 blocks * 288 bytes = 1152 bytes
    EXPECT_EQ(byte_size, expected_bytes);
    EXPECT_EQ(byte_size, 4 * 288);
    EXPECT_GT(byte_size, 0);
}

// ============================================================================
// IQ Formats (Importance Quantization, 256 elements per super-block)
// ============================================================================

/**
 * @brief IQ4_XSTensor: 256 elements per block, 136 bytes per block
 * byte_size() returns raw_data_.size()
 */
TEST_F(Test__TensorByteSize, IQ4_XSTensor_HasCorrectByteSize)
{
    const size_t rows = 4;
    const size_t cols = 256;
    const size_t elements = rows * cols;
    const size_t num_blocks = elements / IQ4_XSBlock::BLOCK_SIZE;
    const size_t expected_bytes = num_blocks * sizeof(IQ4_XSBlock);

    std::vector<uint8_t> raw_data(expected_bytes, 0);
    IQ4_XSTensor tensor({rows, cols}, raw_data);

    size_t byte_size = 0;
    ASSERT_NO_THROW(byte_size = tensor.size_bytes());

    // 4 blocks * 136 bytes = 544 bytes
    EXPECT_EQ(byte_size, expected_bytes);
    EXPECT_EQ(byte_size, 4 * 136);
    EXPECT_GT(byte_size, 0);
}

/**
 * @brief IQ2_XXSTensor: 256 elements per block, 66 bytes per block
 * byte_size() returns raw_data_.size()
 */
TEST_F(Test__TensorByteSize, IQ2_XXSTensor_HasCorrectByteSize)
{
    const size_t rows = 4;
    const size_t cols = 256;
    const size_t elements = rows * cols;
    const size_t num_blocks = elements / IQ2_XXSBlock::BLOCK_SIZE;
    const size_t expected_bytes = num_blocks * sizeof(IQ2_XXSBlock);

    std::vector<uint8_t> raw_data(expected_bytes, 0);
    IQ2_XXSTensor tensor({rows, cols}, raw_data);

    size_t byte_size = 0;
    ASSERT_NO_THROW(byte_size = tensor.size_bytes());

    // 4 blocks * 66 bytes = 264 bytes
    EXPECT_EQ(byte_size, expected_bytes);
    EXPECT_EQ(byte_size, 4 * 66);
    EXPECT_GT(byte_size, 0);
}

/**
 * @brief IQ2_XSTensor: 256 elements per block, 74 bytes per block
 * byte_size() returns raw_data_.size()
 */
TEST_F(Test__TensorByteSize, IQ2_XSTensor_HasCorrectByteSize)
{
    const size_t rows = 4;
    const size_t cols = 256;
    const size_t elements = rows * cols;
    const size_t num_blocks = elements / IQ2_XSBlock::BLOCK_SIZE;
    const size_t expected_bytes = num_blocks * sizeof(IQ2_XSBlock);

    std::vector<uint8_t> raw_data(expected_bytes, 0);
    IQ2_XSTensor tensor({rows, cols}, raw_data);

    size_t byte_size = 0;
    ASSERT_NO_THROW(byte_size = tensor.size_bytes());

    // 4 blocks * 74 bytes = 296 bytes
    EXPECT_EQ(byte_size, expected_bytes);
    EXPECT_EQ(byte_size, 4 * 74);
    EXPECT_GT(byte_size, 0);
}

/**
 * @brief IQ3_XXSTensor: 256 elements per block, 98 bytes per block
 * byte_size() returns raw_data_.size()
 */
TEST_F(Test__TensorByteSize, IQ3_XXSTensor_HasCorrectByteSize)
{
    const size_t rows = 4;
    const size_t cols = 256;
    const size_t elements = rows * cols;
    const size_t num_blocks = elements / IQ3_XXSBlock::BLOCK_SIZE;
    const size_t expected_bytes = num_blocks * sizeof(IQ3_XXSBlock);

    std::vector<uint8_t> raw_data(expected_bytes, 0);
    IQ3_XXSTensor tensor({rows, cols}, raw_data);

    size_t byte_size = 0;
    ASSERT_NO_THROW(byte_size = tensor.size_bytes());

    // 4 blocks * 98 bytes = 392 bytes
    EXPECT_EQ(byte_size, expected_bytes);
    EXPECT_EQ(byte_size, 4 * 98);
    EXPECT_GT(byte_size, 0);
}

/**
 * @brief IQ2_STensor: 256 elements per block, 82 bytes per block
 * byte_size() returns raw_data_.size()
 */
TEST_F(Test__TensorByteSize, IQ2_STensor_HasCorrectByteSize)
{
    const size_t rows = 4;
    const size_t cols = 256;
    const size_t elements = rows * cols;
    const size_t num_blocks = elements / IQ2_SBlock::BLOCK_SIZE;
    const size_t expected_bytes = num_blocks * sizeof(IQ2_SBlock);

    std::vector<uint8_t> raw_data(expected_bytes, 0);
    IQ2_STensor tensor({rows, cols}, raw_data);

    size_t byte_size = 0;
    ASSERT_NO_THROW(byte_size = tensor.size_bytes());

    // 4 blocks * 82 bytes = 328 bytes
    EXPECT_EQ(byte_size, expected_bytes);
    EXPECT_EQ(byte_size, 4 * 82);
    EXPECT_GT(byte_size, 0);
}

/**
 * @brief IQ3_STensor: 256 elements per block, 110 bytes per block
 * byte_size() returns raw_data_.size()
 */
TEST_F(Test__TensorByteSize, IQ3_STensor_HasCorrectByteSize)
{
    const size_t rows = 4;
    const size_t cols = 256;
    const size_t elements = rows * cols;
    const size_t num_blocks = elements / IQ3_SBlock::BLOCK_SIZE;
    const size_t expected_bytes = num_blocks * sizeof(IQ3_SBlock);

    std::vector<uint8_t> raw_data(expected_bytes, 0);
    IQ3_STensor tensor({rows, cols}, raw_data);

    size_t byte_size = 0;
    ASSERT_NO_THROW(byte_size = tensor.size_bytes());

    // 4 blocks * 110 bytes = 440 bytes
    EXPECT_EQ(byte_size, expected_bytes);
    EXPECT_EQ(byte_size, 4 * 110);
    EXPECT_GT(byte_size, 0);
}

/**
 * @brief IQ1_STensor: 256 elements per block, 50 bytes per block
 * byte_size() returns raw_data_.size()
 */
TEST_F(Test__TensorByteSize, IQ1_STensor_HasCorrectByteSize)
{
    const size_t rows = 4;
    const size_t cols = 256;
    const size_t elements = rows * cols;
    const size_t num_blocks = elements / IQ1_SBlock::BLOCK_SIZE;
    const size_t expected_bytes = num_blocks * sizeof(IQ1_SBlock);

    std::vector<uint8_t> raw_data(expected_bytes, 0);
    IQ1_STensor tensor({rows, cols}, raw_data);

    size_t byte_size = 0;
    ASSERT_NO_THROW(byte_size = tensor.size_bytes());

    // 4 blocks * 50 bytes = 200 bytes
    EXPECT_EQ(byte_size, expected_bytes);
    EXPECT_EQ(byte_size, 4 * 50);
    EXPECT_GT(byte_size, 0);
}

/**
 * @brief IQ1_MTensor: 256 elements per block, 56 bytes per block
 * byte_size() returns raw_data_.size()
 */
TEST_F(Test__TensorByteSize, IQ1_MTensor_HasCorrectByteSize)
{
    const size_t rows = 4;
    const size_t cols = 256;
    const size_t elements = rows * cols;
    const size_t num_blocks = elements / IQ1_MBlock::BLOCK_SIZE;
    const size_t expected_bytes = num_blocks * sizeof(IQ1_MBlock);

    std::vector<uint8_t> raw_data(expected_bytes, 0);
    IQ1_MTensor tensor({rows, cols}, raw_data);

    size_t byte_size = 0;
    ASSERT_NO_THROW(byte_size = tensor.size_bytes());

    // 4 blocks * 56 bytes = 224 bytes
    EXPECT_EQ(byte_size, expected_bytes);
    EXPECT_EQ(byte_size, 4 * 56);
    EXPECT_GT(byte_size, 0);
}

// ============================================================================
// Edge Cases and Consistency Tests
// ============================================================================

/**
 * @brief Verify size_bytes() returns non-zero for all tensor types
 *
 * This is a sanity check that size_bytes() never returns zero for valid tensors.
 */
TEST_F(Test__TensorByteSize, AllTensorsHaveNonZeroByteSize)
{
    // Simple tensors
    EXPECT_GT(FP32Tensor({4, 4}).size_bytes(), 0);

    std::vector<uint16_t> fp16_data(16, 0);
    EXPECT_GT(FP16Tensor({4, 4}, fp16_data).size_bytes(), 0);

    std::vector<uint16_t> bf16_data(16, 0);
    EXPECT_GT(BF16Tensor({4, 4}, bf16_data).size_bytes(), 0);

    EXPECT_GT(INT8Tensor({4, 4}).size_bytes(), 0);

    EXPECT_GT(INT32Tensor({4, 4}).size_bytes(), 0);

    // Block-quantized tensors (32 elements per block)
    std::vector<uint8_t> iq4_nl_data(2 * sizeof(IQ4_NLBlock), 0);
    EXPECT_GT(IQ4_NLTensor({1, 64}, iq4_nl_data).size_bytes(), 0);

    std::vector<uint8_t> q8_0_data(2 * sizeof(Q8_0Block), 0);
    EXPECT_GT(Q8_0Tensor({1, 64}, q8_0_data).size_bytes(), 0);

    EXPECT_GT(Q8_1Tensor({1, 64}).size_bytes(), 0);
    EXPECT_GT(Q16_1Tensor({1, 64}).size_bytes(), 0);

    std::vector<uint8_t> q4_0_data(2 * sizeof(Q4_0Block), 0);
    EXPECT_GT(Q4_0Tensor({1, 64}, q4_0_data).size_bytes(), 0);

    std::vector<uint8_t> q4_1_data(2 * sizeof(Q4_1Block), 0);
    EXPECT_GT(Q4_1Tensor({1, 64}, q4_1_data).size_bytes(), 0);

    std::vector<uint8_t> q5_0_data(2 * sizeof(Q5_0Block), 0);
    EXPECT_GT(Q5_0Tensor({1, 64}, q5_0_data).size_bytes(), 0);

    std::vector<uint8_t> q5_1_data(2 * sizeof(Q5_1Block), 0);
    EXPECT_GT(Q5_1Tensor({1, 64}, q5_1_data).size_bytes(), 0);

    // K-quant tensors (256 elements per super-block)
    std::vector<uint8_t> q6_k_data(sizeof(Q6_KBlock), 0);
    EXPECT_GT(Q6_KTensor({1, 256}, q6_k_data).size_bytes(), 0);

    std::vector<uint8_t> q2_k_data(sizeof(Q2_KBlock), 0);
    EXPECT_GT(Q2_KTensor({1, 256}, q2_k_data).size_bytes(), 0);

    std::vector<uint8_t> q3_k_data(sizeof(Q3_KBlock), 0);
    EXPECT_GT(Q3_KTensor({1, 256}, q3_k_data).size_bytes(), 0);

    std::vector<uint8_t> q4_k_data(sizeof(Q4_KBlock), 0);
    EXPECT_GT(Q4_KTensor({1, 256}, q4_k_data).size_bytes(), 0);

    std::vector<uint8_t> q5_k_data(sizeof(Q5_KBlock), 0);
    EXPECT_GT(Q5_KTensor({1, 256}, q5_k_data).size_bytes(), 0);

    std::vector<uint8_t> q8_k_data(sizeof(Q8_KBlock), 0);
    EXPECT_GT(Q8_KTensor({1, 256}, q8_k_data).size_bytes(), 0);

    // IQ tensors (256 elements per super-block)
    std::vector<uint8_t> iq4_xs_data(sizeof(IQ4_XSBlock), 0);
    EXPECT_GT(IQ4_XSTensor({1, 256}, iq4_xs_data).size_bytes(), 0);

    std::vector<uint8_t> iq2_xxs_data(sizeof(IQ2_XXSBlock), 0);
    EXPECT_GT(IQ2_XXSTensor({1, 256}, iq2_xxs_data).size_bytes(), 0);

    std::vector<uint8_t> iq2_xs_data(sizeof(IQ2_XSBlock), 0);
    EXPECT_GT(IQ2_XSTensor({1, 256}, iq2_xs_data).size_bytes(), 0);

    std::vector<uint8_t> iq3_xxs_data(sizeof(IQ3_XXSBlock), 0);
    EXPECT_GT(IQ3_XXSTensor({1, 256}, iq3_xxs_data).size_bytes(), 0);

    std::vector<uint8_t> iq2_s_data(sizeof(IQ2_SBlock), 0);
    EXPECT_GT(IQ2_STensor({1, 256}, iq2_s_data).size_bytes(), 0);

    std::vector<uint8_t> iq3_s_data(sizeof(IQ3_SBlock), 0);
    EXPECT_GT(IQ3_STensor({1, 256}, iq3_s_data).size_bytes(), 0);

    std::vector<uint8_t> iq1_s_data(sizeof(IQ1_SBlock), 0);
    EXPECT_GT(IQ1_STensor({1, 256}, iq1_s_data).size_bytes(), 0);

    std::vector<uint8_t> iq1_m_data(sizeof(IQ1_MBlock), 0);
    EXPECT_GT(IQ1_MTensor({1, 256}, iq1_m_data).size_bytes(), 0);
}

/**
 * @brief Verify block size static assertions match expected values
 *
 * This test verifies that the block size constants in BlockStructures.h
 * match the expected values used in the byte_size calculations.
 */
TEST_F(Test__TensorByteSize, BlockSizeStaticAssertions)
{
    // Simple block formats (32 elements per block)
    EXPECT_EQ(IQ4_NLBlock::BLOCK_SIZE, 32);
    EXPECT_EQ(Q8_0Block::BLOCK_SIZE, 32);
    EXPECT_EQ(Q8_1Block::BLOCK_SIZE, 32);
    EXPECT_EQ(Q16_1Block::BLOCK_SIZE, 32);
    EXPECT_EQ(Q4_0Block::BLOCK_SIZE, 32);
    EXPECT_EQ(Q4_1Block::BLOCK_SIZE, 32);
    EXPECT_EQ(Q5_0Block::BLOCK_SIZE, 32);
    EXPECT_EQ(Q5_1Block::BLOCK_SIZE, 32);

    // K-quant formats (256 elements per super-block)
    EXPECT_EQ(Q6_KBlock::BLOCK_SIZE, 256);
    EXPECT_EQ(Q2_KBlock::BLOCK_SIZE, 256);
    EXPECT_EQ(Q3_KBlock::BLOCK_SIZE, 256);
    EXPECT_EQ(Q4_KBlock::BLOCK_SIZE, 256);
    EXPECT_EQ(Q5_KBlock::BLOCK_SIZE, 256);
    EXPECT_EQ(Q8_KBlock::BLOCK_SIZE, 256);

    // IQ formats (256 elements per super-block)
    EXPECT_EQ(IQ4_XSBlock::BLOCK_SIZE, 256);
    EXPECT_EQ(IQ2_XXSBlock::BLOCK_SIZE, 256);
    EXPECT_EQ(IQ2_XSBlock::BLOCK_SIZE, 256);
    EXPECT_EQ(IQ3_XXSBlock::BLOCK_SIZE, 256);
    EXPECT_EQ(IQ2_SBlock::BLOCK_SIZE, 256);
    EXPECT_EQ(IQ3_SBlock::BLOCK_SIZE, 256);
    EXPECT_EQ(IQ1_SBlock::BLOCK_SIZE, 256);
    EXPECT_EQ(IQ1_MBlock::BLOCK_SIZE, 256);

    // Q16_1 variable block sizes
    EXPECT_EQ(Q16_1Block_64::BLOCK_SIZE, 64);
    EXPECT_EQ(Q16_1Block_128::BLOCK_SIZE, 128);
}

/**
 * @brief Verify sizeof() for all block types matches expected values
 *
 * This ensures the packed structs have correct sizes for byte_size calculations.
 */
TEST_F(Test__TensorByteSize, BlockSizeofAssertions)
{
    // Simple block formats
    EXPECT_EQ(sizeof(IQ4_NLBlock), 18);
    EXPECT_EQ(sizeof(Q8_0Block), 34);
    EXPECT_EQ(sizeof(Q8_1Block), 36);
    EXPECT_EQ(sizeof(Q16_1Block), 72);
    EXPECT_EQ(sizeof(Q4_0Block), 18);
    EXPECT_EQ(sizeof(Q4_1Block), 20);
    EXPECT_EQ(sizeof(Q5_0Block), 22);
    EXPECT_EQ(sizeof(Q5_1Block), 24);

    // K-quant formats
    EXPECT_EQ(sizeof(Q6_KBlock), 210);
    EXPECT_EQ(sizeof(Q2_KBlock), 84);
    EXPECT_EQ(sizeof(Q3_KBlock), 110);
    EXPECT_EQ(sizeof(Q4_KBlock), 144);
    EXPECT_EQ(sizeof(Q5_KBlock), 176);
    EXPECT_EQ(sizeof(Q8_KBlock), 288);

    // IQ formats
    EXPECT_EQ(sizeof(IQ4_XSBlock), 136);
    EXPECT_EQ(sizeof(IQ2_XXSBlock), 66);
    EXPECT_EQ(sizeof(IQ2_XSBlock), 74);
    EXPECT_EQ(sizeof(IQ3_XXSBlock), 98);
    EXPECT_EQ(sizeof(IQ2_SBlock), 82);
    EXPECT_EQ(sizeof(IQ3_SBlock), 110);
    EXPECT_EQ(sizeof(IQ1_SBlock), 50);
    EXPECT_EQ(sizeof(IQ1_MBlock), 56);

    // Q16_1 variable block sizes
    EXPECT_EQ(sizeof(Q16_1Block_64), 136);
    EXPECT_EQ(sizeof(Q16_1Block_128), 264);
}
// ============================================================================
// TensorSlice (wrapper type) - delegates to inner tensor
// ============================================================================

#include "tensors/TensorSlice.h"

/**
 * @brief TensorSlice: delegates byte_size() to inner tensor
 * This test verifies TensorSlice correctly forwards size queries to the wrapped tensor.
 * Critical for GPU transfer of sharded weights.
 */
TEST_F(Test__TensorByteSize, TensorSlice_DelegatesToInner)
{
    // Create an inner FP32 tensor: 32x64 = 2048 elements = 8192 bytes
    auto inner = std::make_shared<FP32Tensor>(std::vector<size_t>{32, 64});
    const size_t expected_bytes = 32 * 64 * sizeof(float);

    // Create slice metadata (full tensor - no actual slicing for this test)
    SliceMetadata meta;
    meta.mode = SliceMode::FULL;
    meta.original_rows = 32;
    meta.original_cols = 64;

    // Create TensorSlice wrapper
    TensorSlice slice(inner, meta);

    // Verify size_bytes() works (public interface)
    size_t byte_size = 0;
    ASSERT_NO_THROW(byte_size = slice.size_bytes());
    EXPECT_EQ(byte_size, expected_bytes);
    EXPECT_EQ(byte_size, 8192);
    EXPECT_GT(byte_size, 0);
}

/**
 * @brief TensorSlice with row-parallel sharding - verifies inner delegation
 */
TEST_F(Test__TensorByteSize, TensorSlice_RowParallel_DelegatesToInner)
{
    // Create inner IQ4_NL tensor: 4x64 = 256 elements = 8 blocks * 18 bytes = 144 bytes
    const size_t rows = 4;
    const size_t cols = 64;
    const size_t num_blocks = (rows * cols) / IQ4_NLBlock::BLOCK_SIZE;
    const size_t expected_bytes = num_blocks * sizeof(IQ4_NLBlock);

    std::vector<uint8_t> raw_data(expected_bytes, 0);
    auto inner = std::make_shared<IQ4_NLTensor>(std::vector<size_t>{rows, cols}, raw_data);

    // Create row-parallel slice metadata
    SliceMetadata meta = SliceMetadata::forRowParallel(rows, cols, 0, 2, true);

    // Create TensorSlice wrapper
    TensorSlice slice(inner, meta);

    // Verify size_bytes() correctly delegates to inner
    size_t byte_size = 0;
    ASSERT_NO_THROW(byte_size = slice.size_bytes());
    EXPECT_EQ(byte_size, expected_bytes);
    EXPECT_EQ(byte_size, 144);
    EXPECT_GT(byte_size, 0);
}