/**
 * @file test_q8_0_tensor.cpp
 * @brief Unit tests for Q8_0Tensor implementation
 *
 * Validates Q8_0Tensor::decodeRow() produces identical results to
 * the current QuantizedTensor::decodeBlock() implementation.
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "tensors/Q8_0Tensor.h"
#include "tensors/TensorFactory.h"
#include <vector>
#include <cmath>
#include <random>

using namespace llaminar;

/**
 * @brief Helper: Create Q8_0 block data with known scale and values
 */
std::vector<uint8_t> create_q8_0_blocks(
    const std::vector<float> &scales,
    const std::vector<std::vector<int8_t>> &block_values)
{
    std::vector<uint8_t> raw_data;

    for (size_t i = 0; i < scales.size(); i++)
    {
        // Convert FP32 scale to FP16
        float scale = scales[i];
        uint32_t bits;
        std::memcpy(&bits, &scale, 4);

        uint32_t sign = (bits >> 16) & 0x8000;
        uint32_t exp = ((bits >> 23) & 0xFF);
        uint32_t mant = (bits & 0x7FFFFF);

        uint16_t fp16;
        if (exp == 0)
        {
            fp16 = sign; // Zero or denormal
        }
        else if (exp == 255)
        {
            fp16 = sign | 0x7C00 | (mant ? 0x0200 : 0); // Inf or NaN
        }
        else
        {
            int32_t new_exp = exp - 127 + 15;
            if (new_exp <= 0)
            {
                fp16 = sign; // Underflow
            }
            else if (new_exp >= 31)
            {
                fp16 = sign | 0x7C00; // Overflow
            }
            else
            {
                fp16 = sign | (new_exp << 10) | (mant >> 13);
            }
        }

        // Write FP16 scale (2 bytes)
        raw_data.push_back(fp16 & 0xFF);
        raw_data.push_back((fp16 >> 8) & 0xFF);

        // Write 32 int8 values
        for (int j = 0; j < 32; j++)
        {
            raw_data.push_back(static_cast<uint8_t>(block_values[i][j]));
        }
    }

    return raw_data;
}

/**
 * @brief Test: Basic construction and metadata
 */
TEST(Q8_0TensorTest, BasicConstruction)
{
    // Create 2x32 tensor (2 blocks)
    std::vector<float> scales = {1.0f, 2.0f};
    std::vector<std::vector<int8_t>> values(2, std::vector<int8_t>(32, 0));

    auto raw_data = create_q8_0_blocks(scales, values);
    Q8_0Tensor tensor({2, 32}, raw_data);

    // Validate metadata
    EXPECT_EQ(tensor.shape()[0], 2);
    EXPECT_EQ(tensor.shape()[1], 32);
    EXPECT_EQ(tensor.size(), 64);
    EXPECT_EQ(tensor.ndim(), 2);
    EXPECT_EQ(tensor.quant_type(), QuantType::Q8_0);
    EXPECT_FLOAT_EQ(tensor.compression_ratio(), 4.0f);

    // Validate block descriptor
    auto desc = tensor.block_descriptor();
    EXPECT_EQ(desc.elements_per_block, 32);
    EXPECT_EQ(desc.bytes_per_block, 34);
    EXPECT_EQ(desc.scale_count, 1);
    EXPECT_EQ(desc.bits_per_value, 8);
    EXPECT_FALSE(desc.is_k_quant);
}

/**
 * @brief Test: decodeRow with simple values
 */
TEST(Q8_0TensorTest, DecodeRowSimple)
{
    // Create 2x32 tensor with known values
    std::vector<float> scales = {0.5f, 1.0f};
    std::vector<std::vector<int8_t>> values(2);

    // Block 0: values 0-31 (scaled by 0.5)
    for (int i = 0; i < 32; i++)
    {
        values[0].push_back(static_cast<int8_t>(i - 16)); // -16 to 15
    }

    // Block 1: values 32-63 (scaled by 1.0)
    for (int i = 0; i < 32; i++)
    {
        values[1].push_back(static_cast<int8_t>(i - 16));
    }

    auto raw_data = create_q8_0_blocks(scales, values);
    Q8_0Tensor tensor({2, 32}, raw_data);

    // Decode row 0
    std::vector<float> row0(32);
    tensor.decodeRow(0, row0.data());

    // Validate row 0 (should be (-16 to 15) * 0.5)
    for (int i = 0; i < 32; i++)
    {
        float expected = static_cast<float>(i - 16) * 0.5f;
        EXPECT_NEAR(row0[i], expected, 1e-5f) << "Mismatch at index " << i;
    }

    // Decode row 1
    std::vector<float> row1(32);
    tensor.decodeRow(1, row1.data());

    // Validate row 1 (should be (-16 to 15) * 1.0)
    for (int i = 0; i < 32; i++)
    {
        float expected = static_cast<float>(i - 16) * 1.0f;
        EXPECT_NEAR(row1[i], expected, 1e-5f) << "Mismatch at index " << i;
    }
}

/**
 * @brief Test: decodeRow vs current QuantizedTensor::decodeBlock
 */
TEST(Q8_0TensorTest, ParityWithCurrentImplementation)
{
    // Create test data: 4x128 tensor (16 blocks)
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> scale_dist(0.01f, 2.0f);
    std::uniform_int_distribution<int> value_dist(-127, 127);

    const int rows = 4;
    const int cols = 128;
    const int num_blocks = (rows * cols + 31) / 32;

    std::vector<float> scales;
    std::vector<std::vector<int8_t>> values;

    for (int i = 0; i < num_blocks; i++)
    {
        scales.push_back(scale_dist(rng));
        std::vector<int8_t> block_vals;
        for (int j = 0; j < 32; j++)
        {
            block_vals.push_back(static_cast<int8_t>(value_dist(rng)));
        }
        values.push_back(block_vals);
    }

    auto raw_data = create_q8_0_blocks(scales, values);

    // Create new Q8_0Tensor
    Q8_0Tensor q8_tensor({rows, cols}, raw_data);

    // Create old QuantizedTensor for comparison
    QuantStorageLayout layout;
    layout.format = QuantFormat::Q8_0;
    layout.original_shape = {rows, cols};
    layout.total_blocks = num_blocks;
    layout.block_desc.elements_per_block = 32;
    layout.block_desc.bytes_per_block = 34;
    QuantizedTensor old_tensor(layout, raw_data);

    // Compare decoding row-by-row
    for (int row = 0; row < rows; row++)
    {
        // Decode with new Q8_0Tensor
        std::vector<float> new_row(cols);
        q8_tensor.decodeRow(row, new_row.data());

        // Decode with old QuantizedTensor (block-by-block)
        std::vector<float> old_row(cols);
        int first_block = (row * cols) / 32;
        int last_block = ((row + 1) * cols - 1) / 32;

        std::vector<float> block_buffer(32);
        for (int block_idx = first_block; block_idx <= last_block; block_idx++)
        {
            old_tensor.decodeBlock(block_idx, block_buffer.data());

            // Copy relevant elements to old_row
            int block_start_elem = block_idx * 32;
            int row_start_elem = row * cols;
            int copy_start = std::max(0, row_start_elem - block_start_elem);
            int copy_end = std::min(32, row_start_elem + cols - block_start_elem);

            for (int i = copy_start; i < copy_end; i++)
            {
                int row_idx = block_start_elem + i - row_start_elem;
                if (row_idx >= 0 && row_idx < cols)
                {
                    old_row[row_idx] = block_buffer[i];
                }
            }
        }

        // Compare results
        for (int col = 0; col < cols; col++)
        {
            EXPECT_NEAR(new_row[col], old_row[col], 1e-5f)
                << "Mismatch at row=" << row << " col=" << col;
        }
    }
}

/**
 * @brief Test: decodeSpan
 */
TEST(Q8_0TensorTest, DecodeSpan)
{
    // Create 2x64 tensor (4 blocks)
    std::vector<float> scales = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<std::vector<int8_t>> values(4);

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 32; j++)
        {
            values[i].push_back(static_cast<int8_t>(i * 32 + j));
        }
    }

    auto raw_data = create_q8_0_blocks(scales, values);
    Q8_0Tensor tensor({2, 64}, raw_data);

    // Decode span: elements 16-47 (crosses block boundary)
    std::vector<float> span(32);
    tensor.decodeSpan(16, 32, span.data());

    // Validate
    for (int i = 0; i < 32; i++)
    {
        float expected = static_cast<float>(16 + i);
        EXPECT_NEAR(span[i], expected, 1e-5f);
    }
}

/**
 * @brief Test: Error handling
 */
TEST(Q8_0TensorTest, ErrorHandling)
{
    std::vector<float> scales = {1.0f};
    std::vector<std::vector<int8_t>> values(1, std::vector<int8_t>(32, 0));
    auto raw_data = create_q8_0_blocks(scales, values);

    Q8_0Tensor tensor({1, 32}, raw_data);

    // Test out-of-bounds row access
    std::vector<float> buffer(32);
    EXPECT_THROW(tensor.decodeRow(10, buffer.data()), std::out_of_range);

    // Test data() throws
    EXPECT_THROW(tensor.data(), std::runtime_error);

    // Test unsupported operations
    EXPECT_THROW(tensor.zero(), std::runtime_error);
    EXPECT_THROW(tensor.fill(1.0f), std::runtime_error);
}

/**
 * @brief Benchmark: decodeRow performance
 */
TEST(Q8_0TensorTest, DecodeRowPerformance)
{
    // Create large tensor: 1000x4096 (~16MB compressed, ~16MB working set)
    const int rows = 1000;
    const int cols = 4096;
    const int num_blocks = (rows * cols + 31) / 32;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> scale_dist(0.01f, 2.0f);
    std::uniform_int_distribution<int> value_dist(-127, 127);

    std::vector<float> scales;
    std::vector<std::vector<int8_t>> values;

    for (int i = 0; i < num_blocks; i++)
    {
        scales.push_back(scale_dist(rng));
        std::vector<int8_t> block_vals;
        for (int j = 0; j < 32; j++)
        {
            block_vals.push_back(static_cast<int8_t>(value_dist(rng)));
        }
        values.push_back(block_vals);
    }

    auto raw_data = create_q8_0_blocks(scales, values);
    Q8_0Tensor tensor({rows, cols}, raw_data);

    // Warm-up
    std::vector<float> buffer(cols);
    tensor.decodeRow(0, buffer.data());

    // Benchmark: decode all rows
    auto start = std::chrono::high_resolution_clock::now();

    for (int row = 0; row < rows; row++)
    {
        tensor.decodeRow(row, buffer.data());
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    double throughput_gb_s = (rows * cols * sizeof(float) / 1e9) / (duration_ms / 1000.0);

    std::cout << "Q8_0Tensor decode throughput: " << throughput_gb_s << " GB/s" << std::endl;
    std::cout << "Average time per row: " << (duration_ms * 1000.0 / rows) << " μs" << std::endl;

// Performance expectations: Debug builds are slower
#ifdef NDEBUG
    EXPECT_GT(throughput_gb_s, 0.1) << "Release build should exceed 0.1 GB/s";
#else
    EXPECT_GT(throughput_gb_s, 0.05) << "Debug build should exceed 0.05 GB/s (got " << throughput_gb_s << ")";
#endif
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
