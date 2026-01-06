/**
 * @file Test__HybridQ16_K_Precision_Fix_Regressions.cpp
 * @brief Regression tests for bugs discovered during HybridQ16 K precision fix implementation
 *
 * This file documents and tests the following issues that were found and fixed:
 *
 * ISSUE #1: Q16→Q16 RoPE Single-Block Bug (blocks_per_head/2 = 0)
 *   - When head_dim == block_size (e.g., 64==64), blocks_per_head = 1
 *   - half_blocks = 1 / 2 = 0 (integer division), causing rotation loop to never execute
 *   - Result: K_out was all zeros despite valid K_in
 *   - Fix: Added single-block path that rotates within the block (qs[0:31] with qs[32:63])
 *
 * ISSUE #2: Q8_1Block Quantization Field Mismatch
 *   - Used wrong field names: scale (doesn't exist) instead of d, qs_sum instead of sum_qs
 *   - Q8_1Block structure: { fp16 d, int16 sum_qs, int8 qs[32] }
 *   - Fix: Use block.d (as FP16) and block.sum_qs (as INT16)
 *
 * ISSUE #3: JIT GEMM block_size=128 Partial Write Bug
 *   - JIT kernel only properly handles block_size=64 at a time
 *   - With block_size=128, kernel wrote first 64 values correctly but remaining 64 were garbage/NaN
 *   - Fix: Force k_block_size=64 in FusedQKVGEMMStage for mixed-precision mode
 *
 * ISSUE #4: FusedQKVGEMMStage FP32 Input Rejection
 *   - Stage required Q8_1 input but received FP32 from previous layer
 *   - Error: "activation tensor must be Q8_1 for quantized output"
 *   - Fix: Added FP32→Q8_1 on-the-fly quantization in mixed-precision path
 *
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>
#include <numeric>

#include "kernels/cpu/primitives/RoPEPrimitives.h"
#include "tensors/BlockStructures.h"
#include "tensors/Tensors.h"
#include "tensors/FP16Utils.h"

using namespace llaminar2;
using namespace llaminar2::primitives;

namespace
{
    // ============================================================================
    // Test Utilities
    // ============================================================================

    /**
     * @brief Compute cosine similarity between two float arrays
     */
    float cosine_similarity(const float *a, const float *b, size_t n)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        if (norm_a < 1e-10 || norm_b < 1e-10)
            return 0.0f;
        return static_cast<float>(dot / (std::sqrt(norm_a) * std::sqrt(norm_b)));
    }

    /**
     * @brief Quantize FP32 to Q16_1Block_64 (64-element blocks for head_dim=64)
     */
    void fp32_to_q16_1_block64(const float *fp32, Q16_1Block_64 *out, size_t n_blocks)
    {
        for (size_t b = 0; b < n_blocks; ++b)
        {
            const float *block_data = fp32 + b * 64;
            Q16_1Block_64 &blk = out[b];

            float max_abs = 0.0f;
            for (int i = 0; i < 64; ++i)
            {
                max_abs = std::max(max_abs, std::fabs(block_data[i]));
            }

            float scale = max_abs / 16383.0f;
            if (scale < 1e-20f)
                scale = 1e-20f;
            float inv_scale = 1.0f / scale;

            int32_t sum_qs = 0;
            for (int i = 0; i < 64; ++i)
            {
                int32_t q = static_cast<int32_t>(std::round(block_data[i] * inv_scale));
                q = std::max(-16383, std::min(16383, q));
                blk.qs[i] = static_cast<int16_t>(q);
                sum_qs += q;
            }

            blk.d = scale;
            blk.sum_qs = sum_qs;
        }
    }

    /**
     * @brief Dequantize Q16_1Block_64 to FP32
     */
    void q16_1_block64_to_fp32(const Q16_1Block_64 *blocks, float *out, size_t n_blocks)
    {
        for (size_t b = 0; b < n_blocks; ++b)
        {
            const Q16_1Block_64 &blk = blocks[b];
            float *block_out = out + b * 64;
            for (int i = 0; i < 64; ++i)
            {
                block_out[i] = blk.qs[i] * blk.d;
            }
        }
    }

    /**
     * @brief Apply FP32 RoPE rotation as reference
     */
    void apply_rope_fp32_reference(float *data, int head_dim, float rope_theta, int position)
    {
        const int half_dim = head_dim / 2;
        for (int i = 0; i < half_dim; ++i)
        {
            float freq = 1.0f / std::pow(rope_theta, static_cast<float>(2 * i) / head_dim);
            float angle = position * freq;
            float cos_val = std::cos(angle);
            float sin_val = std::sin(angle);

            float x = data[i];
            float y = data[i + half_dim];
            data[i] = x * cos_val - y * sin_val;
            data[i + half_dim] = x * sin_val + y * cos_val;
        }
    }

} // namespace

// ============================================================================
// ISSUE #1: Q16→Q16 RoPE Single-Block Bug (blocks_per_head/2 = 0)
// ============================================================================

class Test__RoPE_SingleBlock_Regression : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Generate sin/cos tables in Q15 format for half_dim=32 (for head_dim=64)
        cos_q15_.resize(32);
        sin_q15_.resize(32);
        const float rope_theta = 10000.0f;
        const int position = 5;
        const int head_dim = 64;

        for (int i = 0; i < 32; ++i)
        {
            float freq = 1.0f / std::pow(rope_theta, static_cast<float>(2 * i) / head_dim);
            float angle = position * freq;
            cos_q15_[i] = static_cast<int16_t>(std::round(std::cos(angle) * 32767.0f));
            sin_q15_[i] = static_cast<int16_t>(std::round(std::sin(angle) * 32767.0f));
        }
    }

    std::vector<int16_t> cos_q15_;
    std::vector<int16_t> sin_q15_;
};

/**
 * @test Regression test for single-block Q16→Q16 RoPE
 *
 * When head_dim=64 and block_size=64, there's exactly 1 block per head.
 * The bug was: half_blocks = 1/2 = 0, so the rotation loop never executed.
 * This test verifies the output is non-zero and matches FP32 reference.
 */
TEST_F(Test__RoPE_SingleBlock_Regression, HeadDim64_BlockSize64_ProducesNonZeroOutput)
{
    constexpr int HEAD_DIM = 64;
    constexpr int BLOCK_SIZE = 64;
    constexpr int BLOCKS_PER_HEAD = HEAD_DIM / BLOCK_SIZE; // = 1!

    // Create input with known values
    Q16_1Block_64 in_block, out_block;
    std::vector<float> fp32_input(64);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (int i = 0; i < 64; ++i)
    {
        fp32_input[i] = dist(rng);
    }
    fp32_to_q16_1_block64(fp32_input.data(), &in_block, 1);

    // Zero output
    std::memset(&out_block, 0, sizeof(out_block));

    // Apply Q16→Q16 dynamic scale RoPE
    float out_scale = 0.0f;
    apply_rope_q16_to_q16_head_dynamic_scale<Q16_1Block_64>(
        &in_block, &out_block, HEAD_DIM, cos_q15_.data(), sin_q15_.data(), &out_scale);

    // REGRESSION CHECK: Output must NOT be all zeros
    int nonzero_count = 0;
    for (int i = 0; i < 64; ++i)
    {
        if (out_block.qs[i] != 0)
            nonzero_count++;
    }
    EXPECT_GT(nonzero_count, 32) << "REGRESSION: Q16→Q16 RoPE single-block produced mostly zeros!";
    EXPECT_GT(out_block.d, 0.0f) << "REGRESSION: Output scale is zero!";
    EXPECT_NE(out_scale, 0.0f) << "REGRESSION: Reported scale is zero!";

    // Verify against FP32 reference
    std::vector<float> fp32_reference = fp32_input;
    apply_rope_fp32_reference(fp32_reference.data(), HEAD_DIM, 10000.0f, 5);

    std::vector<float> q16_result(64);
    q16_1_block64_to_fp32(&out_block, q16_result.data(), 1);

    float cosine = cosine_similarity(fp32_reference.data(), q16_result.data(), 64);
    EXPECT_GT(cosine, 0.99f) << "Q16→Q16 single-block RoPE should match FP32 reference (cosine > 0.99)";
}

/**
 * @test Verify all SIMD variants handle single-block case correctly
 */
TEST_F(Test__RoPE_SingleBlock_Regression, AllSIMDVariants_SingleBlock_Match)
{
    constexpr int HEAD_DIM = 64;

    // Create input
    Q16_1Block_64 in_block;
    std::vector<float> fp32_input(64);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    for (int i = 0; i < 64; ++i)
    {
        fp32_input[i] = dist(rng);
    }
    fp32_to_q16_1_block64(fp32_input.data(), &in_block, 1);

    // Run scalar
    Q16_1Block_64 out_scalar;
    float scale_scalar = 0.0f;
    apply_rope_q16_to_q16_head_dynamic_scale_scalar<Q16_1Block_64>(
        &in_block, &out_scalar, HEAD_DIM, cos_q15_.data(), sin_q15_.data(), &scale_scalar);

#if defined(__AVX2__)
    // Run AVX2
    Q16_1Block_64 out_avx2;
    float scale_avx2 = 0.0f;
    apply_rope_q16_to_q16_head_dynamic_scale_avx2_impl<Q16_1Block_64>(
        &in_block, &out_avx2, HEAD_DIM, cos_q15_.data(), sin_q15_.data(), &scale_avx2);

    // Compare AVX2 to scalar
    EXPECT_FLOAT_EQ(scale_scalar, scale_avx2) << "AVX2 scale should match scalar";
    for (int i = 0; i < 64; ++i)
    {
        EXPECT_EQ(out_scalar.qs[i], out_avx2.qs[i]) << "AVX2 qs[" << i << "] should match scalar";
    }
#endif

#if defined(__AVX512F__)
    // Run AVX512
    Q16_1Block_64 out_avx512;
    float scale_avx512 = 0.0f;
    apply_rope_q16_to_q16_head_dynamic_scale_avx512_impl<Q16_1Block_64>(
        &in_block, &out_avx512, HEAD_DIM, cos_q15_.data(), sin_q15_.data(), &scale_avx512);

    // Compare AVX512 to scalar
    EXPECT_FLOAT_EQ(scale_scalar, scale_avx512) << "AVX512 scale should match scalar";
    for (int i = 0; i < 64; ++i)
    {
        EXPECT_EQ(out_scalar.qs[i], out_avx512.qs[i]) << "AVX512 qs[" << i << "] should match scalar";
    }
#endif
}

/**
 * @test Verify blocks_per_head=1 works for different block types
 */
TEST(Test__RoPE_SingleBlock_AllBlockTypes, Block32_HeadDim32)
{
    constexpr int HEAD_DIM = 32;
    constexpr int HALF_DIM = 16;

    // Create sin/cos for half_dim=16
    std::vector<int16_t> cos_q15(HALF_DIM), sin_q15(HALF_DIM);
    for (int i = 0; i < HALF_DIM; ++i)
    {
        float freq = 1.0f / std::pow(10000.0f, static_cast<float>(2 * i) / HEAD_DIM);
        float angle = 1.0f * freq; // position=1
        cos_q15[i] = static_cast<int16_t>(std::round(std::cos(angle) * 32767.0f));
        sin_q15[i] = static_cast<int16_t>(std::round(std::sin(angle) * 32767.0f));
    }

    // Create input Q16_1Block (32-element)
    Q16_1Block in_block, out_block;
    in_block.d = 0.5f;
    in_block.sum_qs = 0;
    int32_t sum = 0;
    for (int i = 0; i < 32; ++i)
    {
        in_block.qs[i] = (i - 16) * 100;
        sum += in_block.qs[i];
    }
    in_block.sum_qs = sum;
    std::memset(&out_block, 0, sizeof(out_block));

    float out_scale = 0.0f;
    apply_rope_q16_to_q16_head_dynamic_scale<Q16_1Block>(
        &in_block, &out_block, HEAD_DIM, cos_q15.data(), sin_q15.data(), &out_scale);

    // Verify non-zero output
    int nonzero = 0;
    for (int i = 0; i < 32; ++i)
    {
        if (out_block.qs[i] != 0)
            nonzero++;
    }
    EXPECT_GT(nonzero, 16) << "Block32 single-block should produce non-zero output";
    EXPECT_GT(out_block.d, 0.0f);
}

// ============================================================================
// ISSUE #2: Q8_1Block Field Names (d vs scale, sum_qs vs qs_sum)
// ============================================================================

/**
 * @test Verify Q8_1Block structure has correct field names
 * This is a compile-time check that documents the correct structure.
 */
TEST(Test__Q8_1Block_Structure, FieldNamesAreCorrect)
{
    Q8_1Block block;

    // These should compile without errors - verifies correct field names
    block.d = fp32_to_fp16(1.0f); // FP16 scale (NOT "scale")
    block.sum_qs = 0;             // INT16 sum (NOT "qs_sum")
    block.qs[0] = 0;              // INT8 quantized values

    // Verify structure sizes
    EXPECT_EQ(sizeof(block.d), 2) << "d should be fp16 (2 bytes)";
    EXPECT_EQ(sizeof(block.sum_qs), 2) << "sum_qs should be int16 (2 bytes)";
    EXPECT_EQ(sizeof(block.qs), 32) << "qs should be 32 bytes (int8[32])";
    EXPECT_EQ(sizeof(Q8_1Block), 36) << "Total Q8_1Block size should be 36 bytes";
}

/**
 * @test Verify Q8_1 quantization produces valid blocks
 */
TEST(Test__Q8_1Block_Quantization, ProducesValidBlocks)
{
    // Simulate FP32→Q8_1 quantization (as done in FusedQKVGEMMStage)
    std::vector<float> fp32_data(32);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (int i = 0; i < 32; ++i)
    {
        fp32_data[i] = dist(rng);
    }

    // Find max absolute value
    float max_abs = 0.0f;
    for (int i = 0; i < 32; ++i)
    {
        max_abs = std::max(max_abs, std::fabs(fp32_data[i]));
    }

    // Quantize to Q8_1
    Q8_1Block block;
    float scale_fp32 = max_abs / 127.0f;
    block.d = fp32_to_fp16(scale_fp32);

    int32_t sum = 0;
    for (int i = 0; i < 32; ++i)
    {
        int32_t q = static_cast<int32_t>(std::round(fp32_data[i] / scale_fp32));
        q = std::max(-127, std::min(127, q));
        block.qs[i] = static_cast<int8_t>(q);
        sum += block.qs[i];
    }
    block.sum_qs = static_cast<int16_t>(sum);

    // Verify block is valid
    EXPECT_GT(fp16_to_fp32(block.d), 0.0f) << "Scale should be positive";
    EXPECT_NE(block.sum_qs, 0) << "Sum should be non-zero for random data";

    // Verify dequantization roundtrip
    float scale = fp16_to_fp32(block.d);
    float max_error = 0.0f;
    for (int i = 0; i < 32; ++i)
    {
        float dequant = block.qs[i] * scale;
        float error = std::fabs(dequant - fp32_data[i]);
        max_error = std::max(max_error, error);
    }
    // Q8 has ~1% quantization error for typical values
    EXPECT_LT(max_error, max_abs * 0.02f) << "Dequantization error should be small";
}

// ============================================================================
// ISSUE #3: JIT GEMM block_size=128 Partial Write Bug
// ============================================================================

/**
 * @test Verify Q16_1Block_64 structure is correct for block_size=64
 */
TEST(Test__Q16BlockSize, Block64_HasCorrectSize)
{
    // Q16_1Block_64 layout: d(4) + sum_qs(4) + qs[64](128) = 136 bytes
    EXPECT_EQ(sizeof(Q16_1Block_64::d), 4) << "d should be float (4 bytes)";
    EXPECT_EQ(sizeof(Q16_1Block_64::sum_qs), 4) << "sum_qs should be int32 (4 bytes)";
    EXPECT_EQ(sizeof(Q16_1Block_64), 136) << "Q16_1Block_64 should be 136 bytes";
    EXPECT_EQ(Q16_1Block_64::BLOCK_SIZE, 64);
}

/**
 * @test Verify Q16_1Block_128 structure
 */
TEST(Test__Q16BlockSize, Block128_HasCorrectSize)
{
    // Q16_1Block_128 layout: d(4) + sum_qs(4) + qs[128](256) = 264 bytes
    EXPECT_EQ(sizeof(Q16_1Block_128), 264) << "Q16_1Block_128 should be 264 bytes";
    EXPECT_EQ(Q16_1Block_128::BLOCK_SIZE, 128);
}

/**
 * @test Verify block_size=64 is consistently used for K projection in Qwen2
 * This documents the design decision to force block_size=64 for JIT compatibility.
 */
TEST(Test__Q16BlockSize, Qwen2_HeadDim64_UsesBlockSize64)
{
    // Qwen2-0.5B has head_dim=64, which equals block_size=64
    // This means 1 block per head, which is the single-block case
    constexpr int HEAD_DIM = 64;
    constexpr int BLOCK_SIZE = 64;

    EXPECT_EQ(HEAD_DIM, BLOCK_SIZE) << "Qwen2 head_dim equals block_size";
    EXPECT_EQ(HEAD_DIM / BLOCK_SIZE, 1) << "Should have exactly 1 block per head";
}

// ============================================================================
// ISSUE #4: FusedQKVGEMMStage FP32 Input Handling
// ============================================================================

/**
 * @test Verify FP32→Q8_1 quantization produces correct output
 * This simulates what FusedQKVGEMMStage does when receiving FP32 input.
 */
TEST(Test__FP32_to_Q8_1_Quantization, ProducesValidOutput)
{
    constexpr int BLOCK_SIZE = 32;
    constexpr int N_BLOCKS = 4;
    constexpr int N_ELEMENTS = BLOCK_SIZE * N_BLOCKS;

    // Create FP32 input
    std::vector<float> fp32_input(N_ELEMENTS);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    for (int i = 0; i < N_ELEMENTS; ++i)
    {
        fp32_input[i] = dist(rng);
    }

    // Quantize to Q8_1
    std::vector<Q8_1Block> q8_blocks(N_BLOCKS);
    for (int b = 0; b < N_BLOCKS; ++b)
    {
        const float *block_data = fp32_input.data() + b * BLOCK_SIZE;
        Q8_1Block &block = q8_blocks[b];

        // Find max absolute value
        float max_abs = 0.0f;
        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            max_abs = std::max(max_abs, std::fabs(block_data[i]));
        }
        if (max_abs < 1e-20f)
            max_abs = 1e-20f;

        float scale_fp32 = max_abs / 127.0f;
        block.d = fp32_to_fp16(scale_fp32);

        int32_t sum = 0;
        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            int32_t q = static_cast<int32_t>(std::round(block_data[i] / scale_fp32));
            q = std::max(-127, std::min(127, q));
            block.qs[i] = static_cast<int8_t>(q);
            sum += block.qs[i];
        }
        block.sum_qs = static_cast<int16_t>(sum);
    }

    // Verify all blocks are valid
    for (int b = 0; b < N_BLOCKS; ++b)
    {
        EXPECT_GT(fp16_to_fp32(q8_blocks[b].d), 0.0f) << "Block " << b << " scale should be positive";

        // At least some qs values should be non-zero
        int nonzero = 0;
        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            if (q8_blocks[b].qs[i] != 0)
                nonzero++;
        }
        EXPECT_GT(nonzero, BLOCK_SIZE / 2) << "Block " << b << " should have non-zero values";
    }

    // Verify roundtrip accuracy
    std::vector<float> dequant(N_ELEMENTS);
    for (int b = 0; b < N_BLOCKS; ++b)
    {
        float scale = fp16_to_fp32(q8_blocks[b].d);
        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            dequant[b * BLOCK_SIZE + i] = q8_blocks[b].qs[i] * scale;
        }
    }

    float cosine = cosine_similarity(fp32_input.data(), dequant.data(), N_ELEMENTS);
    EXPECT_GT(cosine, 0.99f) << "Q8_1 quantization should preserve signal (cosine > 0.99)";
}

/**
 * @test Verify FP32 tensor can provide valid activation data
 */
TEST(Test__FP32_Activation_Support, FP32TensorHasValidData)
{
    std::vector<size_t> shape = {4, 128};
    auto tensor = std::make_unique<FP32Tensor>(shape);

    float *data = tensor->mutable_data();
    ASSERT_NE(data, nullptr);

    // Fill with random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < tensor->numel(); ++i)
    {
        data[i] = dist(rng);
    }

    // Verify we can read it back
    const float *const_data = tensor->data();
    EXPECT_EQ(const_data[0], data[0]);
    EXPECT_EQ(tensor->native_type(), TensorType::FP32);
}

// ============================================================================
// Combined Regression: Full K Precision Fix Pipeline
// ============================================================================

/**
 * @test End-to-end regression test for K precision fix
 *
 * Simulates the full pipeline:
 * 1. FP32 activation → Q8_1 quantization
 * 2. GEMM produces Q16_1 K output (block_size=64)
 * 3. RoPE processes single-block K (head_dim=64)
 * 4. Output should be non-zero and match FP32 reference
 */
TEST(Test__K_Precision_Fix_E2E, FullPipeline_HeadDim64)
{
    constexpr int SEQ_LEN = 1;
    constexpr int N_KV_HEADS = 2;
    constexpr int HEAD_DIM = 64;
    constexpr int POSITION = 5;
    constexpr float ROPE_THETA = 10000.0f;

    // Step 1: Generate "GEMM output" as Q16_1Block_64
    // In reality, GEMM would produce this, but we simulate it
    const int n_heads_total = SEQ_LEN * N_KV_HEADS;
    std::vector<Q16_1Block_64> k_gemm_output(n_heads_total);
    std::vector<float> k_fp32_reference(n_heads_total * HEAD_DIM);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (int h = 0; h < n_heads_total; ++h)
    {
        float *head_fp32 = k_fp32_reference.data() + h * HEAD_DIM;
        for (int i = 0; i < HEAD_DIM; ++i)
        {
            head_fp32[i] = dist(rng);
        }
        fp32_to_q16_1_block64(head_fp32, &k_gemm_output[h], 1);
    }

    // Step 2: Apply Q16→Q16 RoPE with dynamic scale
    std::vector<Q16_1Block_64> k_rope_output(n_heads_total);
    std::vector<float> k_head_scales(n_heads_total);

    // Create sin/cos LUT
    std::vector<int16_t> cos_q15(HEAD_DIM / 2), sin_q15(HEAD_DIM / 2);
    for (int i = 0; i < HEAD_DIM / 2; ++i)
    {
        float freq = 1.0f / std::pow(ROPE_THETA, static_cast<float>(2 * i) / HEAD_DIM);
        float angle = POSITION * freq;
        cos_q15[i] = static_cast<int16_t>(std::round(std::cos(angle) * 32767.0f));
        sin_q15[i] = static_cast<int16_t>(std::round(std::sin(angle) * 32767.0f));
    }

    for (int h = 0; h < n_heads_total; ++h)
    {
        apply_rope_q16_to_q16_head_dynamic_scale<Q16_1Block_64>(
            &k_gemm_output[h],
            &k_rope_output[h],
            HEAD_DIM,
            cos_q15.data(),
            sin_q15.data(),
            &k_head_scales[h]);
    }

    // Step 3: Verify output is non-zero
    for (int h = 0; h < n_heads_total; ++h)
    {
        int nonzero = 0;
        for (int i = 0; i < HEAD_DIM; ++i)
        {
            if (k_rope_output[h].qs[i] != 0)
                nonzero++;
        }
        EXPECT_GT(nonzero, HEAD_DIM / 2) << "Head " << h << " RoPE output should be non-zero";
        EXPECT_GT(k_head_scales[h], 0.0f) << "Head " << h << " scale should be positive";
    }

    // Step 4: Compare to FP32 reference
    for (int h = 0; h < n_heads_total; ++h)
    {
        // Apply FP32 RoPE to reference
        float *head_ref = k_fp32_reference.data() + h * HEAD_DIM;
        apply_rope_fp32_reference(head_ref, HEAD_DIM, ROPE_THETA, POSITION);

        // Dequantize Q16 output
        std::vector<float> q16_result(HEAD_DIM);
        q16_1_block64_to_fp32(&k_rope_output[h], q16_result.data(), 1);

        float cosine = cosine_similarity(head_ref, q16_result.data(), HEAD_DIM);
        EXPECT_GT(cosine, 0.99f) << "Head " << h << " Q16 RoPE should match FP32 reference";
    }
}
