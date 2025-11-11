/**
 * @file Test__TensorDecodeToQ8_0.cpp
 * @brief Unit tests for tensor decode_to_q8_0() methods
 *
 * Validates that all tensor types correctly implement decode_to_q8_0()
 * for use in integer GEMM pipelines. Tests correctness, thread safety,
 * and integration with Q8_0WeightAccessor.
 *
 * @author David Sanftenberg
 * @date November 2025
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "tensors/FP16Utils.h"
#include "tensors/SIMDHelpers.h"
#include "kernels/cpu/gemm/Q8_0WeightAccessor.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>
#include <thread>

using namespace llaminar2;
using namespace llaminar2::simd;
using namespace llaminar2::kernels::gemm;

// ============================================================================
// Test Fixture
// ============================================================================

class TensorDecodeToQ8_0Test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        rng_.seed(42);
    }

    std::mt19937 rng_;

    // Helper: Compare Q8_0 blocks with tolerance
    struct ComparisonResult
    {
        bool passed;
        int max_diff;
        int num_mismatches;
        float scale_diff;
        std::string error_msg;
    };

    ComparisonResult compare_q8_0_blocks(
        const Q8_0Block &a,
        const Q8_0Block &b,
        int tolerance = 1) // Tolerance in quantized int8 domain
    {
        ComparisonResult result{true, 0, 0, 0.0f, ""};

        // Compare scales
        float scale_a = fp16_to_fp32(a.d);
        float scale_b = fp16_to_fp32(b.d);
        result.scale_diff = std::fabs(scale_a - scale_b);

        // Compare quantized values
        for (int i = 0; i < 32; ++i)
        {
            int diff = std::abs(static_cast<int>(a.qs[i]) - static_cast<int>(b.qs[i]));
            result.max_diff = std::max(result.max_diff, diff);
            if (diff > tolerance)
            {
                result.num_mismatches++;
                result.passed = false;
                if (result.error_msg.empty())
                {
                    result.error_msg = "First mismatch at index " + std::to_string(i) +
                                       ": " + std::to_string(static_cast<int>(a.qs[i])) +
                                       " vs " + std::to_string(static_cast<int>(b.qs[i]));
                }
            }
        }

        return result;
    }

    // Helper: Decode Q8_0 block to FP32 for validation
    void decode_q8_0_to_fp32(const Q8_0Block &block, float *output)
    {
        float scale = fp16_to_fp32(block.d);
        for (int i = 0; i < 32; ++i)
        {
            output[i] = block.qs[i] * scale;
        }
    }

    // Helper: Generate random FP32 data
    void generate_random_fp32(float *data, size_t count, float min_val = -1.0f, float max_val = 1.0f)
    {
        std::uniform_real_distribution<float> dist(min_val, max_val);
        for (size_t i = 0; i < count; ++i)
        {
            data[i] = dist(rng_);
        }
    }

    // Helper: Create random IQ4_NL block
    void create_random_iq4nl_block(IQ4_NLBlock &block)
    {
        std::uniform_int_distribution<int> nibble_dist(0, 15);
        std::uniform_real_distribution<float> scale_dist(0.001f, 1.0f);

        for (int i = 0; i < 16; ++i)
        {
            uint8_t low = nibble_dist(rng_);
            uint8_t high = nibble_dist(rng_);
            block.qs[i] = (high << 4) | low;
        }
        block.d = fp32_to_fp16(scale_dist(rng_));
    }

    // Helper: Create random Q6_K block
    void create_random_q6k_block(Q6_KBlock &block)
    {
        std::uniform_int_distribution<int> uint8_dist(0, 255);
        std::uniform_int_distribution<int> int8_dist(-127, 127);
        std::uniform_real_distribution<float> scale_dist(0.001f, 1.0f);

        for (int i = 0; i < 128; ++i)
            block.ql[i] = uint8_dist(rng_);
        for (int i = 0; i < 64; ++i)
            block.qh[i] = uint8_dist(rng_);
        for (int i = 0; i < 16; ++i)
            block.scales[i] = int8_dist(rng_);
        block.d = fp32_to_fp16(scale_dist(rng_));
    }
};

// ============================================================================
// IQ4_NL Tests
// ============================================================================

TEST_F(TensorDecodeToQ8_0Test, IQ4_NL_BasicDecode)
{
    // Create a simple IQ4_NL tensor (1 row × 32 elements = 1 block)
    IQ4_NLBlock iq4_block;
    create_random_iq4nl_block(iq4_block);

    std::vector<uint8_t> raw_data(sizeof(IQ4_NLBlock));
    std::memcpy(raw_data.data(), &iq4_block, sizeof(IQ4_NLBlock));

    auto tensor = std::make_shared<IQ4_NLTensor>(
        std::vector<size_t>{1, 32},
        raw_data);

    // Decode to Q8_0
    Q8_0Block q8_output;
    tensor->decode_to_q8_0(0, 0, &q8_output);

    // Verify output is valid
    EXPECT_NE(q8_output.d, 0) << "Scale should not be zero";

    // Decode to FP32 and check range
    float fp32_output[32];
    decode_q8_0_to_fp32(q8_output, fp32_output);

    for (int i = 0; i < 32; ++i)
    {
        EXPECT_TRUE(std::isfinite(fp32_output[i])) << "Output should be finite at index " << i;
    }
}

TEST_F(TensorDecodeToQ8_0Test, IQ4_NL_MultipleBlocks)
{
    // Test multiple blocks in a row (1 row × 128 elements = 4 blocks)
    const size_t num_blocks = 4;
    std::vector<IQ4_NLBlock> iq4_blocks(num_blocks);
    for (auto &block : iq4_blocks)
    {
        create_random_iq4nl_block(block);
    }

    std::vector<uint8_t> raw_data(num_blocks * sizeof(IQ4_NLBlock));
    std::memcpy(raw_data.data(), iq4_blocks.data(), raw_data.size());

    auto tensor = std::make_shared<IQ4_NLTensor>(
        std::vector<size_t>{1, 128},
        raw_data);

    // Decode each block and verify
    for (size_t kb = 0; kb < num_blocks; ++kb)
    {
        Q8_0Block q8_output;
        tensor->decode_to_q8_0(0, kb, &q8_output);

        EXPECT_NE(q8_output.d, 0) << "Block " << kb << " scale should not be zero";

        // All quantized values should be in valid range
        for (int i = 0; i < 32; ++i)
        {
            EXPECT_GE(q8_output.qs[i], -127) << "Block " << kb << " index " << i;
            EXPECT_LE(q8_output.qs[i], 127) << "Block " << kb << " index " << i;
        }
    }
}

TEST_F(TensorDecodeToQ8_0Test, IQ4_NL_MultipleRows)
{
    // Test multiple rows (4 rows × 64 elements = 4 rows × 2 blocks)
    const size_t rows = 4;
    const size_t cols = 64;
    const size_t blocks_per_row = 2;

    std::vector<IQ4_NLBlock> iq4_blocks(rows * blocks_per_row);
    for (auto &block : iq4_blocks)
    {
        create_random_iq4nl_block(block);
    }

    std::vector<uint8_t> raw_data(iq4_blocks.size() * sizeof(IQ4_NLBlock));
    std::memcpy(raw_data.data(), iq4_blocks.data(), raw_data.size());

    auto tensor = std::make_shared<IQ4_NLTensor>(
        std::vector<size_t>{rows, cols},
        raw_data);

    // Decode all blocks and verify
    for (size_t r = 0; r < rows; ++r)
    {
        for (size_t kb = 0; kb < blocks_per_row; ++kb)
        {
            Q8_0Block q8_output;
            tensor->decode_to_q8_0(r, kb, &q8_output);

            EXPECT_NE(q8_output.d, 0) << "Row " << r << " block " << kb << " scale should not be zero";
        }
    }
}

// ============================================================================
// Q6_K Tests
// ============================================================================

TEST_F(TensorDecodeToQ8_0Test, Q6_K_BasicDecode)
{
    // Q6_K: 256 elements per super-block, 8 sub-blocks of 32 elements
    // Create 1 row × 256 elements = 1 super-block
    Q6_KBlock q6k_block;
    create_random_q6k_block(q6k_block);

    std::vector<uint8_t> raw_data(sizeof(Q6_KBlock));
    std::memcpy(raw_data.data(), &q6k_block, sizeof(Q6_KBlock));

    auto tensor = std::make_shared<Q6_KTensor>(
        std::vector<size_t>{1, 256},
        raw_data);

    // Decode each of the 8 sub-blocks
    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        Q8_0Block q8_output;
        tensor->decode_to_q8_0(0, sub_idx, &q8_output);

        EXPECT_NE(q8_output.d, 0) << "Sub-block " << sub_idx << " scale should not be zero";

        // Verify quantized values in range
        for (int i = 0; i < 32; ++i)
        {
            EXPECT_GE(q8_output.qs[i], -127) << "Sub-block " << sub_idx << " index " << i;
            EXPECT_LE(q8_output.qs[i], 127) << "Sub-block " << sub_idx << " index " << i;
        }
    }
}

TEST_F(TensorDecodeToQ8_0Test, Q6_K_MultipleRows)
{
    // 2 rows × 256 elements = 2 super-blocks
    const size_t rows = 2;
    const size_t cols = 256;
    const size_t superblocks = rows;
    const size_t sub_blocks_per_row = 8;

    std::vector<Q6_KBlock> q6k_blocks(superblocks);
    for (auto &block : q6k_blocks)
    {
        create_random_q6k_block(block);
    }

    std::vector<uint8_t> raw_data(q6k_blocks.size() * sizeof(Q6_KBlock));
    std::memcpy(raw_data.data(), q6k_blocks.data(), raw_data.size());

    auto tensor = std::make_shared<Q6_KTensor>(
        std::vector<size_t>{rows, cols},
        raw_data);

    // Decode all sub-blocks
    for (size_t r = 0; r < rows; ++r)
    {
        for (size_t sub_idx = 0; sub_idx < sub_blocks_per_row; ++sub_idx)
        {
            Q8_0Block q8_output;
            size_t k_block_offset = r * sub_blocks_per_row + sub_idx;
            tensor->decode_to_q8_0(r, sub_idx, &q8_output);

            EXPECT_NE(q8_output.d, 0) << "Row " << r << " sub-block " << sub_idx << " scale should not be zero";
        }
    }
}

// ============================================================================
// FP32 Tests
// ============================================================================

TEST_F(TensorDecodeToQ8_0Test, FP32_BasicQuantization)
{
    // Create FP32 tensor (1 row × 32 elements)
    std::vector<size_t> shape = {1, 32};
    auto tensor = std::make_shared<FP32Tensor>(shape, -1);

    // Fill with random data
    float *fp32_data = tensor->mutable_data();
    generate_random_fp32(fp32_data, 32, -10.0f, 10.0f);

    // Quantize to Q8_0
    Q8_0Block q8_output;
    tensor->decode_to_q8_0(0, 0, &q8_output);

    // Verify scale is reasonable
    float scale = fp16_to_fp32(q8_output.d);
    EXPECT_GT(scale, 0.0f) << "Scale should be positive";

    // Dequantize and check error
    float max_error = 0.0f;
    for (int i = 0; i < 32; ++i)
    {
        float dequant = q8_output.qs[i] * scale;
        float error = std::fabs(dequant - fp32_data[i]);
        max_error = std::max(max_error, error);
    }

    // Quantization error should be within ≈2× quantization step
    EXPECT_LT(max_error, scale * 2.0f) << "Quantization error too large";
}

TEST_F(TensorDecodeToQ8_0Test, FP32_AllZeros)
{
    // Edge case: All zeros
    std::vector<size_t> shape = {1, 32};
    auto tensor = std::make_shared<FP32Tensor>(shape, -1);
    // FP32Tensor constructor zero-initializes, no need to memset

    Q8_0Block q8_output;
    tensor->decode_to_q8_0(0, 0, &q8_output);

    // Should produce zero scale and zero values
    float scale = fp16_to_fp32(q8_output.d);
    EXPECT_EQ(scale, 0.0f) << "Scale should be zero for all-zero input";

    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(q8_output.qs[i], 0) << "All quantized values should be zero";
    }
}

TEST_F(TensorDecodeToQ8_0Test, FP32_PartialBlock)
{
    // Test partial block (tensor smaller than 32 elements)
    std::vector<size_t> shape = {1, 20};
    auto tensor = std::make_shared<FP32Tensor>(shape, -1);

    // Fill with random data
    float *fp32_data = tensor->mutable_data();
    generate_random_fp32(fp32_data, 20, -5.0f, 5.0f);

    Q8_0Block q8_output;
    tensor->decode_to_q8_0(0, 0, &q8_output);

    // Verify first 20 elements are quantized correctly
    float scale = fp16_to_fp32(q8_output.d);
    EXPECT_GT(scale, 0.0f);

    for (int i = 0; i < 20; ++i)
    {
        float dequant = q8_output.qs[i] * scale;
        float error = std::fabs(dequant - fp32_data[i]);
        EXPECT_LT(error, scale * 2.0f) << "Error at index " << i;
    }

    // Remaining elements should be zero-padded
    for (int i = 20; i < 32; ++i)
    {
        EXPECT_EQ(q8_output.qs[i], 0) << "Padding at index " << i << " should be zero";
    }
}

// ============================================================================
// Q2_K Tests (256-element super-blocks)
// ============================================================================

TEST_F(TensorDecodeToQ8_0Test, Q2_K_BasicDecode)
{
    // Q2_K super-block: 256 elements (8 × 32-element Q8_0 blocks)
    constexpr size_t SUPER_BLOCK_SIZE = 256;
    constexpr size_t Q8_BLOCKS_PER_SUPER = 8;

    std::vector<size_t> shape = {1, SUPER_BLOCK_SIZE};

    // Create Q2_K block with known structure
    Q2_KBlock q2k_block;
    std::memset(&q2k_block, 0, sizeof(q2k_block));

    // Set super-block scales
    q2k_block.d = fp32_to_fp16(0.5f);    // Super-block scale
    q2k_block.dmin = fp32_to_fp16(0.1f); // Min scale

    // Fill scales (16 × 4-bit scales packed in 16 bytes)
    for (int i = 0; i < 16; ++i)
    {
        q2k_block.scales[i] = i * 4; // Simple pattern
    }

    // Fill 2-bit quantized values (64 bytes, 4 values per byte)
    for (int i = 0; i < 64; ++i)
    {
        q2k_block.qs[i] = (i & 0x3) | ((i & 0x3) << 2) | ((i & 0x3) << 4) | ((i & 0x3) << 6);
    }

    std::vector<uint8_t> raw_data(sizeof(Q2_KBlock));
    std::memcpy(raw_data.data(), &q2k_block, sizeof(Q2_KBlock));

    auto tensor = std::make_shared<Q2_KTensor>(shape, raw_data);

    // Decode each Q8_0 block within the super-block
    for (size_t kb = 0; kb < Q8_BLOCKS_PER_SUPER; ++kb)
    {
        Q8_0Block q8_output;
        tensor->decode_to_q8_0(0, kb, &q8_output);

        // Verify scale is reasonable (derived from super-block scales)
        float scale = fp16_to_fp32(q8_output.d);
        EXPECT_GT(scale, 0.0f) << "Scale should be positive for block " << kb;
        EXPECT_LT(scale, 1.0f) << "Scale should be reasonable for block " << kb;

        // Verify quantized values are in valid int8 range
        for (int i = 0; i < 32; ++i)
        {
            EXPECT_GE(q8_output.qs[i], -127) << "Q8 value underflow at block " << kb << ", index " << i;
            EXPECT_LE(q8_output.qs[i], 127) << "Q8 value overflow at block " << kb << ", index " << i;
        }
    }
}

TEST_F(TensorDecodeToQ8_0Test, Q2_K_MultipleRows)
{
    // Test multiple rows of Q2_K super-blocks
    constexpr size_t rows = 4;
    constexpr size_t cols = 256;
    std::vector<size_t> shape = {rows, cols};

    // Create 4 Q2_K blocks (one per row)
    std::vector<Q2_KBlock> q2k_blocks(rows);
    for (size_t r = 0; r < rows; ++r)
    {
        std::memset(&q2k_blocks[r], 0, sizeof(Q2_KBlock));
        q2k_blocks[r].d = fp32_to_fp16(0.3f + r * 0.1f); // Varying scales
        q2k_blocks[r].dmin = fp32_to_fp16(0.05f);

        // Fill scales array
        for (int i = 0; i < 16; ++i)
        {
            q2k_blocks[r].scales[i] = (r + 1) * 8 + i;
        }

        // Fill with row-specific pattern
        for (int i = 0; i < 64; ++i)
        {
            q2k_blocks[r].qs[i] = (r + i) & 0xFF;
        }
    }

    std::vector<uint8_t> raw_data(sizeof(Q2_KBlock) * rows);
    std::memcpy(raw_data.data(), q2k_blocks.data(), raw_data.size());

    auto tensor = std::make_shared<Q2_KTensor>(shape, raw_data);

    // Decode first Q8_0 block from each row
    for (size_t r = 0; r < rows; ++r)
    {
        Q8_0Block q8_output;
        tensor->decode_to_q8_0(r, 0, &q8_output);

        float scale = fp16_to_fp32(q8_output.d);
        EXPECT_GT(scale, 0.0f) << "Row " << r << " should have positive scale";
    }
}

// ============================================================================
// Q8_K Tests (256-element super-blocks)
// ============================================================================

TEST_F(TensorDecodeToQ8_0Test, Q8_K_BasicDecode)
{
    // Q8_K super-block: 256 elements (8 × 32-element Q8_0 blocks)
    constexpr size_t SUPER_BLOCK_SIZE = 256;
    constexpr size_t Q8_BLOCKS_PER_SUPER = 8;

    std::vector<size_t> shape = {1, SUPER_BLOCK_SIZE};

    // Create Q8_K block
    Q8_KBlock q8k_block;
    std::memset(&q8k_block, 0, sizeof(q8k_block));

    // Fill with known int8 pattern
    for (int i = 0; i < 256; ++i)
    {
        q8k_block.qs[i] = (i % 2 == 0) ? 64 : -64; // Alternating ±64
    }

    // Fill block sums (16 values, one per 16-element sub-block)
    for (int i = 0; i < 16; ++i)
    {
        q8k_block.bsums[i] = i * 128; // Simple pattern
    }

    std::vector<uint8_t> raw_data(sizeof(Q8_KBlock));
    std::memcpy(raw_data.data(), &q8k_block, sizeof(Q8_KBlock));

    auto tensor = std::make_shared<Q8_KTensor>(shape, raw_data);

    // Decode each Q8_0 block within the super-block
    for (size_t kb = 0; kb < Q8_BLOCKS_PER_SUPER; ++kb)
    {
        Q8_0Block q8_output;
        tensor->decode_to_q8_0(0, kb, &q8_output);

        // Verify scale
        float scale = fp16_to_fp32(q8_output.d);
        EXPECT_GT(scale, 0.0f) << "Scale should be positive for block " << kb;

        // Verify quantized values match source pattern
        for (int i = 0; i < 32; ++i)
        {
            int source_idx = kb * 32 + i;
            int expected = (source_idx % 2 == 0) ? 64 : -64;

            // Allow small quantization error
            int diff = std::abs(q8_output.qs[i] - expected);
            EXPECT_LE(diff, 2) << "Block " << kb << ", index " << i
                               << ": expected ~" << expected << ", got " << static_cast<int>(q8_output.qs[i]);
        }
    }
}

TEST_F(TensorDecodeToQ8_0Test, Q8_K_MultipleRows)
{
    // Test multiple rows of Q8_K super-blocks
    constexpr size_t rows = 4;
    constexpr size_t cols = 256;
    std::vector<size_t> shape = {rows, cols};

    // Create 4 Q8_K blocks (one per row)
    std::vector<Q8_KBlock> q8k_blocks(rows);
    for (size_t r = 0; r < rows; ++r)
    {
        std::memset(&q8k_blocks[r], 0, sizeof(Q8_KBlock));

        // Fill with row-specific pattern
        for (int i = 0; i < 256; ++i)
        {
            q8k_blocks[r].qs[i] = (r * 32 + i) % 127 - 64; // Varying per row
        }

        for (int i = 0; i < 16; ++i)
        {
            q8k_blocks[r].bsums[i] = r * 100 + i;
        }
    }

    std::vector<uint8_t> raw_data(sizeof(Q8_KBlock) * rows);
    std::memcpy(raw_data.data(), q8k_blocks.data(), raw_data.size());

    auto tensor = std::make_shared<Q8_KTensor>(shape, raw_data);

    // Decode first Q8_0 block from each row
    for (size_t r = 0; r < rows; ++r)
    {
        Q8_0Block q8_output;
        tensor->decode_to_q8_0(r, 0, &q8_output);

        float scale = fp16_to_fp32(q8_output.d);
        EXPECT_GT(scale, 0.0f) << "Row " << r << " should have positive scale";

        // Verify values match row-specific pattern
        for (int i = 0; i < 32; ++i)
        {
            int expected = (r * 32 + i) % 127 - 64;
            int diff = std::abs(q8_output.qs[i] - expected);
            EXPECT_LE(diff, 2) << "Row " << r << ", index " << i;
        }
    }
}

// ============================================================================
// Q4_0 Tests (32-element blocks, simple quantization)
// ============================================================================

TEST_F(TensorDecodeToQ8_0Test, Q4_0_BasicDecode)
{
    // Q4_0: 32 elements per block
    constexpr size_t BLOCK_SIZE = 32;
    std::vector<size_t> shape = {1, BLOCK_SIZE};

    // Create Q4_0 block
    Q4_0Block q4_block;
    std::memset(&q4_block, 0, sizeof(q4_block));
    q4_block.d = fp32_to_fp16(0.5f); // Scale

    // Fill 4-bit values (2 values per byte, 16 bytes for 32 values)
    for (int i = 0; i < 16; ++i)
    {
        q4_block.qs[i] = (i & 0xF) | ((i & 0xF) << 4); // Both nibbles same value
    }

    std::vector<uint8_t> raw_data(sizeof(Q4_0Block));
    std::memcpy(raw_data.data(), &q4_block, sizeof(Q4_0Block));

    auto tensor = std::make_shared<Q4_0Tensor>(shape, raw_data);

    // Decode to Q8_0
    Q8_0Block q8_output;
    tensor->decode_to_q8_0(0, 0, &q8_output);

    // Verify scale is reasonable
    float scale = fp16_to_fp32(q8_output.d);
    EXPECT_GT(scale, 0.0f) << "Scale should be positive";

    // Verify quantized values are in valid int8 range
    for (int i = 0; i < 32; ++i)
    {
        EXPECT_GE(q8_output.qs[i], -127);
        EXPECT_LE(q8_output.qs[i], 127);
    }
}

// ============================================================================
// Q4_1 Tests (32-element blocks with min offset)
// ============================================================================

TEST_F(TensorDecodeToQ8_0Test, Q4_1_BasicDecode)
{
    // Q4_1: 32 elements per block with scale + min
    constexpr size_t BLOCK_SIZE = 32;
    std::vector<size_t> shape = {1, BLOCK_SIZE};

    // Create Q4_1 block
    Q4_1Block q4_block;
    std::memset(&q4_block, 0, sizeof(q4_block));
    q4_block.d = fp32_to_fp16(0.4f); // Scale
    q4_block.m = fp32_to_fp16(0.1f); // Min offset

    // Fill 4-bit values
    for (int i = 0; i < 16; ++i)
    {
        q4_block.qs[i] = (i & 0xF) | ((15 - i) << 4);
    }

    std::vector<uint8_t> raw_data(sizeof(Q4_1Block));
    std::memcpy(raw_data.data(), &q4_block, sizeof(Q4_1Block));

    auto tensor = std::make_shared<Q4_1Tensor>(shape, raw_data);

    // Decode to Q8_0
    Q8_0Block q8_output;
    tensor->decode_to_q8_0(0, 0, &q8_output);

    // Verify scale is reasonable
    float scale = fp16_to_fp32(q8_output.d);
    EXPECT_GT(scale, 0.0f) << "Scale should be positive";

    // Verify quantized values are in valid int8 range
    for (int i = 0; i < 32; ++i)
    {
        EXPECT_GE(q8_output.qs[i], -127);
        EXPECT_LE(q8_output.qs[i], 127);
    }
}

// ============================================================================
// Q5_0 Tests (32-element blocks, 5-bit quantization)
// ============================================================================

TEST_F(TensorDecodeToQ8_0Test, Q5_0_BasicDecode)
{
    // Q5_0: 32 elements per block with 5-bit values
    constexpr size_t BLOCK_SIZE = 32;
    std::vector<size_t> shape = {1, BLOCK_SIZE};

    // Create Q5_0 block
    Q5_0Block q5_block;
    std::memset(&q5_block, 0, sizeof(q5_block));
    q5_block.d = fp32_to_fp16(0.6f); // Scale

    // Fill lower 4 bits (16 bytes for 32 values)
    for (int i = 0; i < 16; ++i)
    {
        q5_block.qs[i] = (i & 0xF) | ((i & 0xF) << 4);
    }

    // Fill high bits (32 bits = 4 bytes)
    for (int i = 0; i < 4; ++i)
    {
        q5_block.qh[i] = i * 64; // Simple pattern
    }

    std::vector<uint8_t> raw_data(sizeof(Q5_0Block));
    std::memcpy(raw_data.data(), &q5_block, sizeof(Q5_0Block));

    auto tensor = std::make_shared<Q5_0Tensor>(shape, raw_data);

    // Decode to Q8_0
    Q8_0Block q8_output;
    tensor->decode_to_q8_0(0, 0, &q8_output);

    // Verify scale is reasonable
    float scale = fp16_to_fp32(q8_output.d);
    EXPECT_GT(scale, 0.0f) << "Scale should be positive";

    // Verify quantized values are in valid int8 range
    for (int i = 0; i < 32; ++i)
    {
        EXPECT_GE(q8_output.qs[i], -127);
        EXPECT_LE(q8_output.qs[i], 127);
    }
}

// ============================================================================
// Q5_1 Tests (32-element blocks, 5-bit with min offset)
// ============================================================================

TEST_F(TensorDecodeToQ8_0Test, Q5_1_BasicDecode)
{
    // Q5_1: 32 elements per block with 5-bit values + min
    constexpr size_t BLOCK_SIZE = 32;
    std::vector<size_t> shape = {1, BLOCK_SIZE};

    // Create Q5_1 block
    Q5_1Block q5_block;
    std::memset(&q5_block, 0, sizeof(q5_block));
    q5_block.d = fp32_to_fp16(0.45f); // Scale
    q5_block.m = fp32_to_fp16(0.05f); // Min offset

    // Fill lower 4 bits
    for (int i = 0; i < 16; ++i)
    {
        q5_block.qs[i] = (i & 0xF) | ((15 - i) << 4);
    }

    // Fill high bits
    for (int i = 0; i < 4; ++i)
    {
        q5_block.qh[i] = i * 32;
    }

    std::vector<uint8_t> raw_data(sizeof(Q5_1Block));
    std::memcpy(raw_data.data(), &q5_block, sizeof(Q5_1Block));

    auto tensor = std::make_shared<Q5_1Tensor>(shape, raw_data);

    // Decode to Q8_0
    Q8_0Block q8_output;
    tensor->decode_to_q8_0(0, 0, &q8_output);

    // Verify scale is reasonable
    float scale = fp16_to_fp32(q8_output.d);
    EXPECT_GT(scale, 0.0f) << "Scale should be positive";

    // Verify quantized values are in valid int8 range
    for (int i = 0; i < 32; ++i)
    {
        EXPECT_GE(q8_output.qs[i], -127);
        EXPECT_LE(q8_output.qs[i], 127);
    }
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST_F(TensorDecodeToQ8_0Test, ThreadSafety_ConcurrentAccess)
{
    // Create IQ4_NL tensor with multiple rows
    const size_t rows = 64;
    const size_t cols = 128;
    const size_t blocks_per_row = 4;

    std::vector<IQ4_NLBlock> iq4_blocks(rows * blocks_per_row);
    for (auto &block : iq4_blocks)
    {
        create_random_iq4nl_block(block);
    }

    std::vector<uint8_t> raw_data(iq4_blocks.size() * sizeof(IQ4_NLBlock));
    std::memcpy(raw_data.data(), iq4_blocks.data(), raw_data.size());

    auto tensor = std::make_shared<IQ4_NLTensor>(
        std::vector<size_t>{rows, cols},
        raw_data);

    // Access from multiple threads concurrently
    const int num_threads = 8;
    std::vector<std::thread> threads;
    std::vector<bool> success(num_threads, false);

    for (int t = 0; t < num_threads; ++t)
    {
        threads.emplace_back([&, t]()
                             {
            // Each thread decodes different rows
            size_t start_row = (t * rows) / num_threads;
            size_t end_row = ((t + 1) * rows) / num_threads;

            for (size_t r = start_row; r < end_row; ++r) {
                for (size_t kb = 0; kb < blocks_per_row; ++kb) {
                    Q8_0Block q8_output;
                    tensor->decode_to_q8_0(r, kb, &q8_output);

                    // Verify output is valid
                    if (q8_output.d == 0) {
                        return; // Failure
                    }
                }
            }
            success[t] = true; });
    }

    // Wait for all threads
    for (auto &thread : threads)
    {
        thread.join();
    }

    // Verify all threads succeeded
    for (int t = 0; t < num_threads; ++t)
    {
        EXPECT_TRUE(success[t]) << "Thread " << t << " failed";
    }
}

// ============================================================================
// Accessor Integration Tests
// ============================================================================

TEST_F(TensorDecodeToQ8_0Test, AccessorIntegration_IQ4_NL)
{
    // Verify accessor correctly uses tensor's decode_to_q8_0()
    const size_t rows = 4;
    const size_t cols = 64;
    const size_t blocks_per_row = 2;

    std::vector<IQ4_NLBlock> iq4_blocks(rows * blocks_per_row);
    for (auto &block : iq4_blocks)
    {
        create_random_iq4nl_block(block);
    }

    std::vector<uint8_t> raw_data(iq4_blocks.size() * sizeof(IQ4_NLBlock));
    std::memcpy(raw_data.data(), iq4_blocks.data(), raw_data.size());

    auto tensor = std::make_shared<IQ4_NLTensor>(
        std::vector<size_t>{rows, cols},
        raw_data);

    // Create accessor
    auto accessor = createQ8_0Accessor(tensor.get(), 16);
    ASSERT_NE(accessor, nullptr);
    EXPECT_FALSE(accessor->is_zero_copy()) << "IQ4_NL should use cached accessor";

    // Access blocks via accessor
    for (size_t r = 0; r < rows; ++r)
    {
        for (size_t kb = 0; kb < blocks_per_row; ++kb)
        {
            const Q8_0Block *block = accessor->get_q8_block(r, kb);
            ASSERT_NE(block, nullptr) << "Accessor returned null for (" << r << ", " << kb << ")";

            EXPECT_NE(block->d, 0) << "Scale should not be zero for (" << r << ", " << kb << ")";
        }
    }
}

TEST_F(TensorDecodeToQ8_0Test, AccessorIntegration_Q8_0_ZeroCopy)
{
    // Verify Q8_0 accessor uses zero-copy path
    const size_t rows = 4;
    const size_t cols = 64;
    const size_t blocks_per_row = 2;

    std::vector<Q8_0Block> q8_blocks(rows * blocks_per_row);

    // Fill with random data
    std::uniform_int_distribution<int> int8_dist(-127, 127);
    for (auto &block : q8_blocks)
    {
        for (int i = 0; i < 32; ++i)
        {
            block.qs[i] = static_cast<int8_t>(int8_dist(rng_));
        }
        block.d = fp32_to_fp16(0.5f);
    }

    std::vector<uint8_t> raw_data(q8_blocks.size() * sizeof(Q8_0Block));
    std::memcpy(raw_data.data(), q8_blocks.data(), raw_data.size());

    auto tensor = std::make_shared<Q8_0Tensor>(
        std::vector<size_t>{rows, cols},
        raw_data);

    // Create accessor
    auto accessor = createQ8_0Accessor(tensor.get());
    ASSERT_NE(accessor, nullptr);
    EXPECT_TRUE(accessor->is_zero_copy()) << "Q8_0 should use zero-copy accessor";

    // Verify direct pointer access
    for (size_t r = 0; r < rows; ++r)
    {
        for (size_t kb = 0; kb < blocks_per_row; ++kb)
        {
            const Q8_0Block *block1 = accessor->get_q8_block(r, kb);
            const Q8_0Block *block2 = accessor->get_q8_block(r, kb);

            // Zero-copy should return same pointer
            EXPECT_EQ(block1, block2) << "Zero-copy should return same pointer for (" << r << ", " << kb << ")";
        }
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
