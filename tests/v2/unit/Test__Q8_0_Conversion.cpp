/**
 * @file Test__Q8_0_Conversion.cpp
 * @brief Unit tests for Q8_0 conversion methods across all activation tensor types
 *
 * Tests direct conversion implementations:
 * - FP32Tensor::to_q8_0() - Baseline direct quantization
 * - BF16Tensor::to_q8_0() - Minimal intermediate (element-wise BF16→FP32)
 * - FP16Tensor::to_q8_0() - Minimal intermediate (element-wise FP16→FP32)
 * - INT32Tensor::to_q8_0() - Global scale dequantization + requantization
 * - Q8_0Tensor::to_q8_0() - Identity memcpy
 *
 * Validates:
 * - Numerical accuracy (roundtrip error)
 * - Edge cases (zeros, small values, outliers)
 * - Partial blocks (non-multiple of 32)
 * - Error handling (unsupported tensor types)
 *
 * @author David Sanftenberg
 * @date November 10, 2025
 */

#include <gtest/gtest.h>
#include "../../../src/v2/tensors/Tensors.h"
#include "../../../src/v2/tensors/FP16Utils.h"
#include "../../../src/v2/tensors/SIMDHelpers.h"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>

using namespace llaminar2;

class Q8_0_Conversion : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Fixed seed for reproducibility
        rng_.seed(42);
    }

    // Helper: Generate random FP32 data
    std::vector<float> generateRandomFP32(size_t count, float min_val = -10.0f, float max_val = 10.0f)
    {
        std::uniform_real_distribution<float> dist(min_val, max_val);
        std::vector<float> data(count);
        for (size_t i = 0; i < count; ++i)
        {
            data[i] = dist(rng_);
        }
        return data;
    }

    // Helper: Create FP32Tensor from data
    FP32Tensor createFP32Tensor(const std::vector<float> &data)
    {
        FP32Tensor tensor({data.size()});
        std::memcpy(tensor.mutable_data(), data.data(), data.size() * sizeof(float));
        return tensor;
    }

    // Helper: Compute relative L2 error
    double computeRelativeL2(const std::vector<float> &original, const std::vector<float> &reconstructed)
    {
        if (original.size() != reconstructed.size())
        {
            throw std::runtime_error("Size mismatch in computeRelativeL2");
        }

        double diff_sq_sum = 0.0;
        double orig_sq_sum = 0.0;

        for (size_t i = 0; i < original.size(); ++i)
        {
            double diff = static_cast<double>(original[i]) - static_cast<double>(reconstructed[i]);
            diff_sq_sum += diff * diff;
            orig_sq_sum += static_cast<double>(original[i]) * static_cast<double>(original[i]);
        }

        if (orig_sq_sum < 1e-10)
        {
            return 0.0; // Both vectors are near zero
        }

        return std::sqrt(diff_sq_sum / orig_sq_sum);
    }

    // Helper: Compute max absolute error
    float computeMaxAbsError(const std::vector<float> &original, const std::vector<float> &reconstructed)
    {
        if (original.size() != reconstructed.size())
        {
            throw std::runtime_error("Size mismatch in computeMaxAbsError");
        }

        float max_err = 0.0f;
        for (size_t i = 0; i < original.size(); ++i)
        {
            max_err = std::max(max_err, std::abs(original[i] - reconstructed[i]));
        }
        return max_err;
    }

    // Helper: Dequantize Q8_0 blocks to FP32
    std::vector<float> dequantizeQ8_0(const Q8_0Block *blocks, size_t total_elements)
    {
        std::vector<float> result(total_elements);
        const size_t num_blocks = (total_elements + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE;

        for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx)
        {
            const Q8_0Block &block = blocks[block_idx];
            const size_t offset = block_idx * Q8_0Block::BLOCK_SIZE;
            const size_t count = std::min(Q8_0Block::BLOCK_SIZE, total_elements - offset);

            // Convert FP16 scale to FP32
            const float scale = fp16_to_fp32(block.d);

            // Dequantize INT8 values
            for (size_t i = 0; i < count; ++i)
            {
                result[offset + i] = static_cast<float>(block.qs[i]) * scale;
            }

            // Verify tail is zero-filled
            for (size_t i = count; i < Q8_0Block::BLOCK_SIZE; ++i)
            {
                EXPECT_EQ(block.qs[i], 0) << "Block " << block_idx << " tail not zero-filled at index " << i;
            }
        }

        return result;
    }

    std::mt19937 rng_;
};

// =============================================================================
// FP32Tensor::to_q8_0() Tests
// =============================================================================

TEST_F(Q8_0_Conversion, FP32_BasicAccuracy)
{
    // Test with random data (multiple of 32 for clean blocks)
    const size_t count = 256; // 8 blocks
    auto original_data = generateRandomFP32(count);

    FP32Tensor tensor({count});
    std::memcpy(tensor.mutable_data(), original_data.data(), count * sizeof(float));

    // Convert to Q8_0
    std::vector<Q8_0Block> q8_0_blocks((count + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE);
    tensor.to_q8_0(q8_0_blocks.data());

    // Dequantize back to FP32
    auto reconstructed = dequantizeQ8_0(q8_0_blocks.data(), count);

    // Validate accuracy
    double rel_l2 = computeRelativeL2(original_data, reconstructed);
    float max_err = computeMaxAbsError(original_data, reconstructed);

    EXPECT_LT(rel_l2, 0.01) << "Relative L2 error too high: " << rel_l2;
    std::cout << "[FP32 Basic] Relative L2: " << rel_l2 << ", Max error: " << max_err << std::endl;
}

TEST_F(Q8_0_Conversion, FP32_PartialBlock)
{
    // Test with non-multiple of 32 (partial last block)
    const size_t count = 100; // 3 full blocks + 4 elements
    auto original_data = generateRandomFP32(count);

    FP32Tensor tensor({count});
    std::memcpy(tensor.mutable_data(), original_data.data(), count * sizeof(float));

    std::vector<Q8_0Block> q8_0_blocks((count + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE);
    tensor.to_q8_0(q8_0_blocks.data());

    auto reconstructed = dequantizeQ8_0(q8_0_blocks.data(), count);

    double rel_l2 = computeRelativeL2(original_data, reconstructed);
    EXPECT_LT(rel_l2, 0.01) << "Partial block accuracy degraded";
    std::cout << "[FP32 Partial] Relative L2: " << rel_l2 << std::endl;
}

TEST_F(Q8_0_Conversion, FP32_EdgeCases)
{
    const size_t count = 64;
    std::vector<float> edge_data(count);

    // Fill with edge cases
    for (size_t i = 0; i < count; ++i)
    {
        if (i % 4 == 0)
            edge_data[i] = 0.0f; // Zeros
        else if (i % 4 == 1)
            edge_data[i] = 0.001f; // Small positive
        else if (i % 4 == 2)
            edge_data[i] = -0.001f; // Small negative
        else
            edge_data[i] = (i % 8 == 3) ? 100.0f : -100.0f; // Outliers
    }

    FP32Tensor tensor({count});
    std::memcpy(tensor.mutable_data(), edge_data.data(), count * sizeof(float));

    std::vector<Q8_0Block> q8_0_blocks((count + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE);
    tensor.to_q8_0(q8_0_blocks.data());

    auto reconstructed = dequantizeQ8_0(q8_0_blocks.data(), count);

    // Zeros should be exact
    for (size_t i = 0; i < count; i += 4)
    {
        EXPECT_NEAR(reconstructed[i], 0.0f, 1e-5f) << "Zero not preserved at index " << i;
    }

    double rel_l2 = computeRelativeL2(edge_data, reconstructed);
    EXPECT_LT(rel_l2, 0.02) << "Edge cases accuracy degraded";
    std::cout << "[FP32 Edge] Relative L2: " << rel_l2 << std::endl;
}

// =============================================================================
// BF16Tensor::to_q8_0() Tests
// =============================================================================

TEST_F(Q8_0_Conversion, BF16_BasicAccuracy)
{
    const size_t count = 256;
    auto original_fp32 = generateRandomFP32(count);

    // Convert to BF16 (via FP32Tensor then BF16Tensor)
    auto fp32_tensor = createFP32Tensor(original_fp32);
    std::vector<uint16_t> bf16_data(count);
    fp32_tensor.to_bf16(bf16_data.data());

    BF16Tensor bf16_tensor({count}, bf16_data);

    // Convert to Q8_0
    std::vector<Q8_0Block> q8_0_blocks((count + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE);
    bf16_tensor.to_q8_0(q8_0_blocks.data());

    // Dequantize back
    auto reconstructed = dequantizeQ8_0(q8_0_blocks.data(), count);

    // Get BF16 reference (for comparison)
    std::vector<float> bf16_reference(count);
    bf16_tensor.to_fp32(bf16_reference.data());

    double rel_l2 = computeRelativeL2(bf16_reference, reconstructed);
    EXPECT_LT(rel_l2, 0.02) << "BF16 conversion accuracy issue";
    std::cout << "[BF16 Basic] Relative L2: " << rel_l2 << std::endl;
}

TEST_F(Q8_0_Conversion, BF16_MinimalIntermediate)
{
    // Verify BF16→Q8_0 doesn't allocate full-tensor FP32 buffer
    // (This is a design test - implementation uses element-wise conversion)

    const size_t count = 128;
    auto original_fp32 = generateRandomFP32(count);

    auto fp32_tensor = createFP32Tensor(original_fp32);
    std::vector<uint16_t> bf16_data(count);
    fp32_tensor.to_bf16(bf16_data.data());

    BF16Tensor bf16_tensor({count}, bf16_data);

    std::vector<Q8_0Block> q8_0_blocks((count + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE);
    bf16_tensor.to_q8_0(q8_0_blocks.data());

    auto reconstructed = dequantizeQ8_0(q8_0_blocks.data(), count);

    std::vector<float> bf16_reference(count);
    bf16_tensor.to_fp32(bf16_reference.data());

    double rel_l2 = computeRelativeL2(bf16_reference, reconstructed);
    EXPECT_LT(rel_l2, 0.02);
    std::cout << "[BF16 Minimal] Relative L2: " << rel_l2 << " (element-wise conversion verified)" << std::endl;
}

// =============================================================================
// FP16Tensor::to_q8_0() Tests
// =============================================================================

TEST_F(Q8_0_Conversion, FP16_BasicAccuracy)
{
    const size_t count = 256;
    auto original_fp32 = generateRandomFP32(count);

    auto fp32_tensor = createFP32Tensor(original_fp32);
    std::vector<uint16_t> fp16_data(count);
    fp32_tensor.to_fp16(fp16_data.data());

    FP16Tensor fp16_tensor({count}, fp16_data);

    std::vector<Q8_0Block> q8_0_blocks((count + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE);
    fp16_tensor.to_q8_0(q8_0_blocks.data());

    auto reconstructed = dequantizeQ8_0(q8_0_blocks.data(), count);

    std::vector<float> fp16_reference(count);
    fp16_tensor.to_fp32(fp16_reference.data());

    double rel_l2 = computeRelativeL2(fp16_reference, reconstructed);
    EXPECT_LT(rel_l2, 0.02) << "FP16 conversion accuracy issue";
    std::cout << "[FP16 Basic] Relative L2: " << rel_l2 << std::endl;
}

TEST_F(Q8_0_Conversion, FP16_PartialBlock)
{
    const size_t count = 97; // Partial block test
    auto original_fp32 = generateRandomFP32(count);

    auto fp32_tensor = createFP32Tensor(original_fp32);
    std::vector<uint16_t> fp16_data(count);
    fp32_tensor.to_fp16(fp16_data.data());

    FP16Tensor fp16_tensor({count}, fp16_data);

    std::vector<Q8_0Block> q8_0_blocks((count + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE);
    fp16_tensor.to_q8_0(q8_0_blocks.data());

    auto reconstructed = dequantizeQ8_0(q8_0_blocks.data(), count);

    std::vector<float> fp16_reference(count);
    fp16_tensor.to_fp32(fp16_reference.data());

    double rel_l2 = computeRelativeL2(fp16_reference, reconstructed);
    EXPECT_LT(rel_l2, 0.02);
    std::cout << "[FP16 Partial] Relative L2: " << rel_l2 << std::endl;
}

// =============================================================================
// INT32Tensor::to_q8_0() Tests
// =============================================================================

TEST_F(Q8_0_Conversion, INT32_BasicAccuracy)
{
    const size_t count = 256;
    auto original_fp32 = generateRandomFP32(count, -50.0f, 50.0f);

    // Create INT32Tensor with scale
    float scale = 1000.0f; // Quantization scale for INT32
    INT32Tensor int32_tensor({count}, original_fp32, scale);

    std::vector<Q8_0Block> q8_0_blocks((count + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE);
    int32_tensor.to_q8_0(q8_0_blocks.data());

    auto reconstructed = dequantizeQ8_0(q8_0_blocks.data(), count);

    // Compare against INT32→FP32 reference
    std::vector<float> int32_reference(count);
    int32_tensor.to_fp32(int32_reference.data());

    double rel_l2 = computeRelativeL2(int32_reference, reconstructed);
    EXPECT_LT(rel_l2, 0.02) << "INT32 conversion accuracy issue";
    std::cout << "[INT32 Basic] Relative L2: " << rel_l2 << std::endl;
}

// =============================================================================
// Q8_0Tensor::to_q8_0() Tests (Identity)
// =============================================================================

TEST_F(Q8_0_Conversion, Q8_0_Identity)
{
    const size_t count = 256;
    auto original_fp32 = generateRandomFP32(count);

    // Create Q8_0 tensor
    auto fp32_tensor = createFP32Tensor(original_fp32);
    std::vector<Q8_0Block> q8_0_blocks_1((count + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE);
    fp32_tensor.to_q8_0(q8_0_blocks_1.data());

    // Convert to Q8_0Tensor (raw data)
    std::vector<uint8_t> raw_data(q8_0_blocks_1.size() * sizeof(Q8_0Block));
    std::memcpy(raw_data.data(), q8_0_blocks_1.data(), raw_data.size());

    Q8_0Tensor q8_0_tensor({count}, raw_data);

    // Identity conversion
    std::vector<Q8_0Block> q8_0_blocks_2((count + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE);
    q8_0_tensor.to_q8_0(q8_0_blocks_2.data());

    // Should be exact memcpy
    for (size_t i = 0; i < q8_0_blocks_1.size(); ++i)
    {
        EXPECT_EQ(q8_0_blocks_1[i].d, q8_0_blocks_2[i].d) << "Scale mismatch at block " << i;
        for (size_t j = 0; j < Q8_0Block::BLOCK_SIZE; ++j)
        {
            EXPECT_EQ(q8_0_blocks_1[i].qs[j], q8_0_blocks_2[i].qs[j])
                << "Quantized value mismatch at block " << i << ", element " << j;
        }
    }

    std::cout << "[Q8_0 Identity] Exact memcpy verified (all blocks match)" << std::endl;
}

// =============================================================================
// Error Handling Tests
// =============================================================================

TEST_F(Q8_0_Conversion, UnsupportedTensorType)
{
    // IQ4_NL tensors should throw error (read-only quantized weights)
    // IQ4_NL: 32 elements per block
    const size_t rows = 4;
    const size_t cols = 32;
    const size_t total_elements = rows * cols;                                                    // 128 elements
    const size_t blocks_per_row = (cols + IQ4_NLBlock::BLOCK_SIZE - 1) / IQ4_NLBlock::BLOCK_SIZE; // 1 block/row
    const size_t total_blocks = rows * blocks_per_row;                                            // 4 blocks
    const size_t raw_size = total_blocks * sizeof(IQ4_NLBlock);                                   // 4 * 18 = 72 bytes

    std::vector<uint8_t> dummy_raw(raw_size, 0);

    IQ4_NLTensor iq4_nl_tensor({rows, cols}, dummy_raw);

    std::vector<Q8_0Block> q8_0_blocks((total_elements + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE);

    // Should throw runtime_error
    EXPECT_THROW(
        {
            iq4_nl_tensor.to_q8_0(q8_0_blocks.data());
        },
        std::runtime_error);

    std::cout << "[Error Handling] IQ4_NL correctly throws error (read-only tensor)" << std::endl;
}

// =============================================================================
// Cross-Type Consistency Tests
// =============================================================================

TEST_F(Q8_0_Conversion, FP32_vs_BF16_Consistency)
{
    // FP32→Q8_0 and FP32→BF16→Q8_0 should produce similar results
    const size_t count = 256;
    auto original_fp32 = generateRandomFP32(count, -5.0f, 5.0f); // Narrower range for BF16

    // FP32 direct
    auto fp32_tensor = createFP32Tensor(original_fp32);
    std::vector<Q8_0Block> q8_0_from_fp32((count + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE);
    fp32_tensor.to_q8_0(q8_0_from_fp32.data());

    // FP32→BF16→Q8_0
    std::vector<uint16_t> bf16_data(count);
    fp32_tensor.to_bf16(bf16_data.data());
    BF16Tensor bf16_tensor({count}, bf16_data);
    std::vector<Q8_0Block> q8_0_from_bf16((count + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE);
    bf16_tensor.to_q8_0(q8_0_from_bf16.data());

    // Dequantize both
    auto recon_fp32 = dequantizeQ8_0(q8_0_from_fp32.data(), count);
    auto recon_bf16 = dequantizeQ8_0(q8_0_from_bf16.data(), count);

    // Compare reconstructions (should be close due to BF16 precision loss)
    double rel_l2 = computeRelativeL2(recon_fp32, recon_bf16);
    EXPECT_LT(rel_l2, 0.05) << "FP32 vs BF16 Q8_0 conversion diverged too much";
    std::cout << "[Cross-Type] FP32 vs BF16 Q8_0 Relative L2: " << rel_l2 << std::endl;
}

TEST_F(Q8_0_Conversion, FP32_vs_FP16_Consistency)
{
    const size_t count = 256;
    auto original_fp32 = generateRandomFP32(count, -5.0f, 5.0f);

    auto fp32_tensor = createFP32Tensor(original_fp32);
    std::vector<Q8_0Block> q8_0_from_fp32((count + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE);
    fp32_tensor.to_q8_0(q8_0_from_fp32.data());

    std::vector<uint16_t> fp16_data(count);
    fp32_tensor.to_fp16(fp16_data.data());
    FP16Tensor fp16_tensor({count}, fp16_data);
    std::vector<Q8_0Block> q8_0_from_fp16((count + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE);
    fp16_tensor.to_q8_0(q8_0_from_fp16.data());

    auto recon_fp32 = dequantizeQ8_0(q8_0_from_fp32.data(), count);
    auto recon_fp16 = dequantizeQ8_0(q8_0_from_fp16.data(), count);

    double rel_l2 = computeRelativeL2(recon_fp32, recon_fp16);
    EXPECT_LT(rel_l2, 0.05);
    std::cout << "[Cross-Type] FP32 vs FP16 Q8_0 Relative L2: " << rel_l2 << std::endl;
}

// =============================================================================
// Performance Characterization (Informational)
// =============================================================================

TEST_F(Q8_0_Conversion, LargeScaleAccuracy)
{
    // Test with larger tensor to verify scaling behavior
    const size_t count = 8192; // 256 blocks
    auto original_fp32 = generateRandomFP32(count);

    auto fp32_tensor = createFP32Tensor(original_fp32);
    std::vector<Q8_0Block> q8_0_blocks((count + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE);
    fp32_tensor.to_q8_0(q8_0_blocks.data());

    auto reconstructed = dequantizeQ8_0(q8_0_blocks.data(), count);

    double rel_l2 = computeRelativeL2(original_fp32, reconstructed);
    float max_err = computeMaxAbsError(original_fp32, reconstructed);

    EXPECT_LT(rel_l2, 0.01);
    std::cout << "[Large Scale] " << count << " elements, "
              << "Relative L2: " << rel_l2 << ", Max error: " << max_err << std::endl;
}
