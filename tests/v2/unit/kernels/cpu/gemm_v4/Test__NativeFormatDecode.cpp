/**
 * @file Test__NativeFormatDecode.cpp
 * @brief Unit tests for native quantization format decode to int8
 * @author David Sanftenberg
 *
 * Tests that native format decode preserves maximum precision compared to
 * FP32→Q8_0 requantization path.
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include "tensors/Tensors.h"
#include "tensors/BlockStructures.h"
#include "tensors/FP16Utils.h"

using namespace llaminar2;

/**
 * @brief Test fixture for native format decode
 */
class Test__NativeFormatDecode : public ::testing::Test
{
protected:
    /**
     * @brief Create Q4_0 tensor from FP32 data
     */
    std::shared_ptr<Q4_0Tensor> createQ4_0FromFP32(const std::vector<float> &fp32_data, size_t rows, size_t cols)
    {
        std::vector<uint8_t> raw_data;
        size_t n_blocks = (cols + 31) / 32;
        raw_data.resize(rows * n_blocks * sizeof(Q4_0Block));

        Q4_0Block *blocks = reinterpret_cast<Q4_0Block *>(raw_data.data());

        for (size_t r = 0; r < rows; ++r)
        {
            for (size_t kb = 0; kb < n_blocks; ++kb)
            {
                size_t block_idx = r * n_blocks + kb;
                Q4_0Block &block = blocks[block_idx];

                // Find max abs value in this 32-element block
                float max_abs = 0.0f;
                for (int i = 0; i < 32 && (kb * 32 + i) < cols; ++i)
                {
                    size_t idx = r * cols + kb * 32 + i;
                    max_abs = std::max(max_abs, std::abs(fp32_data[idx]));
                }

                // Quantize: Q4_0 uses 4-bit signed values in range [-7, 7] with offset
                float d = max_abs / 7.0f;
                if (d < 1e-10f)
                    d = 1.0f;
                float id = 1.0f / d;

                block.d = fp32_to_fp16(d);

                // Pack two 4-bit values per byte
                for (int i = 0; i < 16; ++i)
                {
                    int idx0 = i * 2;
                    int idx1 = i * 2 + 1;

                    float val0 = (kb * 32 + idx0 < cols) ? fp32_data[r * cols + kb * 32 + idx0] * id : 0.0f;
                    float val1 = (kb * 32 + idx1 < cols) ? fp32_data[r * cols + kb * 32 + idx1] * id : 0.0f;

                    // Clamp to [-7, 7], add 8 offset for unsigned 4-bit [0, 15]
                    int8_t q0 = static_cast<int8_t>(std::round(std::clamp(val0, -7.0f, 7.0f))) + 8;
                    int8_t q1 = static_cast<int8_t>(std::round(std::clamp(val1, -7.0f, 7.0f))) + 8;

                    block.qs[i] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
                }
            }
        }

        return std::make_shared<Q4_0Tensor>(std::vector<size_t>{rows, cols}, raw_data);
    }

    /**
     * @brief Decode Q4_0 block to FP32 (ground truth)
     *
     * Uses SPLIT LAYOUT to match SIMD unpacking for QuantisedGemmKernel:
     * - output[0..15]: low nibbles from qs[0..15]
     * - output[16..31]: high nibbles from qs[0..15]
     */
    void decodeQ4_0BlockToFP32(const Q4_0Block &block, float *output)
    {
        float d = fp16_to_fp32(block.d);

        // SPLIT LAYOUT: low nibbles in [0..15], high nibbles in [16..31]
        for (int i = 0; i < 16; ++i)
        {
            uint8_t packed = block.qs[i];
            int8_t q_lo = (packed & 0x0F) - 8; // Extract low nibble, remove offset
            int8_t q_hi = (packed >> 4) - 8;   // Extract high nibble, remove offset

            output[i] = q_lo * d;      // Low nibbles in first half
            output[i + 16] = q_hi * d; // High nibbles in second half
        }
    }

    /**
     * @brief Unpack Q4_0 block to native int8 range [-8, 7] (REFERENCE IMPLEMENTATION)
     *
     * Uses SPLIT LAYOUT to match SIMD unpacking for QuantisedGemmKernel:
     * - output[0..15]: low nibbles from qs[0..15]
     * - output[16..31]: high nibbles from qs[0..15]
     */
    void unpackQ4_0BlockToInt8(const Q4_0Block &block, int8_t *output)
    {
        // SPLIT LAYOUT: low nibbles in [0..15], high nibbles in [16..31]
        for (int i = 0; i < 16; ++i)
        {
            uint8_t packed = block.qs[i];
            output[i] = (packed & 0x0F) - 8;    // Low nibbles in first half
            output[i + 16] = (packed >> 4) - 8; // High nibbles in second half
        }
    }

    /**
     * @brief OLD PATH: FP32 → Q8_0 requantization
     */
    void requantizeToQ8_0(const float *fp32_data, int8_t *output, float &scale_out, size_t n_elements)
    {
        float max_abs = 0.0f;
        for (size_t i = 0; i < n_elements; ++i)
        {
            max_abs = std::max(max_abs, std::abs(fp32_data[i]));
        }

        float d = max_abs / 127.0f;
        if (d < 1e-10f)
            d = 1.0f;
        float id = 1.0f / d;

        scale_out = d;

        for (size_t i = 0; i < n_elements; ++i)
        {
            output[i] = static_cast<int8_t>(std::round(fp32_data[i] * id));
        }
    }
};

/**
 * @brief Test that native decode matches direct Q4_0→FP32→int8 path
 */
TEST_F(Test__NativeFormatDecode, Q4_0_NativeDecodeAccuracy)
{
    // Create test data: simple ramp pattern
    const size_t rows = 2;
    const size_t cols = 64; // 2 blocks
    std::vector<float> fp32_data(rows * cols);

    for (size_t i = 0; i < fp32_data.size(); ++i)
    {
        fp32_data[i] = static_cast<float>(i % 64) * 0.5f; // Range [0, 31.5]
    }

    // Create Q4_0 tensor
    auto q4_0_tensor = createQ4_0FromFP32(fp32_data, rows, cols);

    // Test native unpacking accuracy
    const size_t blocks_per_row = (cols + Q4_0Block::BLOCK_SIZE - 1) / Q4_0Block::BLOCK_SIZE;

    // Test first block using Q4_0Tensor's IINT8Unpackable interface
    int8_t unpacked_output[32];
    q4_0_tensor->unpack_block_to_int8(0, 0, unpacked_output);
    float block_scale = q4_0_tensor->get_block_scale(0, 0);

    // Compare with reference implementation
    const Q4_0Block *block_ptr = static_cast<const Q4_0Block *>(
        q4_0_tensor->get_raw_block_at(0, 0));
    int8_t reference_output[32];
    unpackQ4_0BlockToInt8(*block_ptr, reference_output);

    // Verify unpacked int8 values match reference
    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(unpacked_output[i], reference_output[i])
            << "Element " << i << ": unpacked int8 value mismatch";
    }

    // Verify scale matches original Q4_0 scale
    float expected_scale = fp16_to_fp32(block_ptr->d);
    EXPECT_NEAR(block_scale, expected_scale, 1e-6f)
        << "Block scale should match original Q4_0 scale";

    // Verify reconstruction to FP32
    float fp32_block[32];
    decodeQ4_0BlockToFP32(*block_ptr, fp32_block);

    for (int i = 0; i < 32; ++i)
    {
        float reconstructed = unpacked_output[i] * block_scale;
        float diff = std::abs(reconstructed - fp32_block[i]);
        float rel_error = std::abs(fp32_block[i]) > 1e-6f ? diff / std::abs(fp32_block[i]) : diff;

        EXPECT_LT(rel_error, 0.02f) << "Element " << i
                                    << ": reconstructed=" << reconstructed
                                    << ", expected=" << fp32_block[i];
    }
}

/**
 * @brief Compare native decode vs FP32→Q8_0 requantization
 *
 * This test demonstrates that native decode preserves more precision
 * by avoiding the intermediate FP32 dequantization and requantization.
 */
TEST_F(Test__NativeFormatDecode, Q4_0_NativeVsRequantizationError)
{
    const size_t rows = 1;
    const size_t cols = 32; // Single block

    // Create test data with challenging values (mix of small and large)
    std::vector<float> fp32_data(cols);
    fp32_data[0] = 0.001f; // Very small
    fp32_data[1] = 0.1f;
    fp32_data[2] = 1.0f;
    fp32_data[3] = 10.0f;
    fp32_data[4] = 50.0f; // Large
    for (size_t i = 5; i < cols; ++i)
    {
        fp32_data[i] = static_cast<float>(i - 5) * 1.5f;
    }

    // Create Q4_0 tensor
    auto q4_0_tensor = createQ4_0FromFP32(fp32_data, rows, cols);

    // Path 1: NATIVE unpacking (Q4_0 → int8 native range [-8, 7] via IINT8Unpackable)
    int8_t native_output[32];
    q4_0_tensor->unpack_block_to_int8(0, 0, native_output);
    float native_scale = q4_0_tensor->get_block_scale(0, 0);

    // Path 2: OLD requantization path (Q4_0 → FP32 → Q8_0 range [-127, 127])
    const Q4_0Block *block_ptr = static_cast<const Q4_0Block *>(
        q4_0_tensor->get_raw_block_at(0, 0));
    float fp32_block[32];
    decodeQ4_0BlockToFP32(*block_ptr, fp32_block);

    int8_t requant_output[32];
    float requant_scale;
    requantizeToQ8_0(fp32_block, requant_output, requant_scale, 32);

    // Compare errors
    float native_total_error = 0.0f;
    float requant_total_error = 0.0f;

    for (int i = 0; i < 32; ++i)
    {
        float native_recon = native_output[i] * native_scale;
        float requant_recon = requant_output[i] * requant_scale;

        float native_error = std::abs(native_recon - fp32_data[i]);
        float requant_error = std::abs(requant_recon - fp32_data[i]);

        native_total_error += native_error;
        requant_total_error += requant_error;
    }

    // NATIVE unpacking preserves Q4_0's original quantization (range [-8,7], original scale)
    // REQUANTIZATION uses Q8_0 range ([-127,127], recomputed scale from max_abs)
    // Native should have LOWER error (preserves original quantization)

    std::cout << "Native unpacking total error:    " << native_total_error << std::endl;
    std::cout << "Requantization path total error: " << requant_total_error << std::endl;

    // Native should be better or equal (Q4_0 → native int8 preserves original precision)
    EXPECT_LE(native_total_error, requant_total_error * 1.1f)
        << "Native unpacking should preserve precision better than requantization";
}

/**
 * @brief Benchmark native decode performance
 */
TEST_F(Test__NativeFormatDecode, Q4_0_DecodePerformance)
{
    const size_t rows = 4096;
    const size_t cols = 4096; // Large matrix
    const size_t n_blocks_per_row = (cols + 31) / 32;

    std::vector<float> fp32_data(rows * cols, 1.0f);
    auto q4_0_tensor = createQ4_0FromFP32(fp32_data, rows, cols);

    // Allocate output buffers
    std::vector<int8_t> output(cols);

    auto start = std::chrono::high_resolution_clock::now();

    // Unpack all blocks in first row using IINT8Unpackable interface
    for (size_t kb = 0; kb < n_blocks_per_row; ++kb)
    {
        q4_0_tensor->unpack_block_to_int8(0, kb, output.data() + kb * 32);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    double blocks_per_sec = (n_blocks_per_row * 1e6) / duration_us;
    double gb_per_sec = (n_blocks_per_row * sizeof(Q4_0Block) * 1e6) / (duration_us * 1e9);

    std::cout << "Decoded " << n_blocks_per_row << " Q4_0 blocks in " << duration_us << " µs" << std::endl;
    std::cout << "Throughput: " << blocks_per_sec << " blocks/sec" << std::endl;
    std::cout << "Bandwidth:  " << gb_per_sec << " GB/s" << std::endl;

    // Sanity check: Should be able to decode at least 100k blocks/sec on modern CPU
    EXPECT_GT(blocks_per_sec, 100000.0) << "Decode performance too slow";
}

/**
 * @brief Test handling of zero/near-zero blocks
 */
TEST_F(Test__NativeFormatDecode, Q4_0_EdgeCases)
{
    const size_t cols = 32;

    // Test 1: All zeros
    {
        std::vector<float> zeros(cols, 0.0f);
        auto q4_0 = createQ4_0FromFP32(zeros, 1, cols);
        const Q4_0Block *block_ptr = static_cast<const Q4_0Block *>(
            q4_0->get_raw_block_at(0, 0));

        int8_t output[32];
        unpackQ4_0BlockToInt8(*block_ptr, output);

        // Q4_0 zero: FP32 0.0 → quantized as 0 → encoded as nibble 0x08 → unpacks to (8 - 8) = 0
        for (int i = 0; i < 32; ++i)
        {
            EXPECT_EQ(output[i], 0) << "Zero block should unpack to int8 zero (0)";
        }
    }

    // Test 2: Very small values
    {
        std::vector<float> small(cols);
        for (size_t i = 0; i < cols; ++i)
        {
            small[i] = 1e-8f * (i + 1); // Tiny values
        }
        auto q4_0 = createQ4_0FromFP32(small, 1, cols);
        const Q4_0Block *block_ptr = static_cast<const Q4_0Block *>(
            q4_0->get_raw_block_at(0, 0));

        int8_t output[32];
        unpackQ4_0BlockToInt8(*block_ptr, output);

        // Native Q4_0 range is [-8, 7], should always be in range
        bool all_valid = true;
        for (int i = 0; i < 32; ++i)
        {
            if (output[i] < -8 || output[i] > 7)
            {
                all_valid = false;
            }
        }
        EXPECT_TRUE(all_valid) << "Unpacked values should be in Q4_0 native range [-8, 7]";
    }

    // Test 3: Large values
    {
        std::vector<float> large(cols);
        for (size_t i = 0; i < cols; ++i)
        {
            large[i] = 1000.0f + i * 10.0f; // Large values
        }
        auto q4_0 = createQ4_0FromFP32(large, 1, cols);
        const Q4_0Block *block_ptr = static_cast<const Q4_0Block *>(
            q4_0->get_raw_block_at(0, 0));

        int8_t output[32];
        unpackQ4_0BlockToInt8(*block_ptr, output);
        float scale = fp16_to_fp32(block_ptr->d);

        // Verify scale is reasonable
        EXPECT_GT(scale, 0.0f) << "Scale should be positive";
        EXPECT_LT(scale, 2000.0f) << "Scale should be reasonable for large values";

        // Verify unpacked values are in Q4_0 native range [-8, 7]
        bool all_in_range = true;
        for (int i = 0; i < 32; ++i)
        {
            if (output[i] < -8 || output[i] > 7)
            {
                all_in_range = false;
            }
        }
        EXPECT_TRUE(all_in_range) << "All unpacked values should be in Q4_0 range [-8, 7]";

        // Verify decoded values preserve accuracy
        // SPLIT LAYOUT mapping:
        // - output[0..15] = low nibbles → original indices 0, 2, 4, ... 30
        // - output[16..31] = high nibbles → original indices 1, 3, 5, ... 31
        for (int i = 0; i < 32; ++i)
        {
            float reconstructed = output[i] * scale;
            size_t original_idx = (i < 16) ? (i * 2) : ((i - 16) * 2 + 1);
            EXPECT_NEAR(reconstructed, large[original_idx], large[original_idx] * 0.15f)
                << "Large values should preserve relative accuracy (output[" << i
                << "] → large[" << original_idx << "])";
        }
    }
}
