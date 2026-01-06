/**
 * @file Test__Q16IntegerAttention.cpp
 * @brief Unit tests for the Q16 Integer Attention kernel (Phase 8)
 *
 * Tests the full Q16IntegerAttentionRef implementation:
 * - Flash Decode path (seq_len_q = 1)
 * - FA2 Prefill path (seq_len_q > 1)
 * - Parameter validation
 * - Snapshot capture
 *
 * Uses TestTensorFactory for convenient test data generation.
 *
 * @see docs/v2/PROJECT_Q16_INTEGER_ATTENTION_V2.md Phase 8
 */

#include <gtest/gtest.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "kernels/cpu/attention/q16_1/ref/Q16IntegerAttentionRef.h"
#include "tensors/BlockStructures.h"
#include "tensors/Tensors.h"

using namespace llaminar2::kernels::q16_1;
using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__Q16IntegerAttention : public ::testing::Test
{
protected:
    // Default test dimensions
    static constexpr int DEFAULT_HEAD_DIM = 64;
    static constexpr int DEFAULT_NUM_HEADS = 14;
    static constexpr int DEFAULT_NUM_KV_HEADS = 2;
    static constexpr int DEFAULT_KV_LEN = 32;

    std::mt19937 rng_{42};

    /**
     * @brief Create Q16_1 blocks with random INT16 values and uniform scale
     *
     * @tparam BlockType Q16_1Block_64 or Q16_1Block_128
     * @param num_blocks Total number of blocks to create
     * @param scale Scale factor for all blocks
     * @param value_range Max absolute value for INT16 (default 1000 for safe VNNI)
     */
    template <typename BlockType>
    std::vector<BlockType> createRandomBlocks(
        int num_blocks,
        float scale = 1.0f / 32767.0f,
        int value_range = 1000)
    {
        std::uniform_int_distribution<int16_t> dist(-value_range, value_range);
        std::vector<BlockType> blocks(num_blocks);

        for (int b = 0; b < num_blocks; ++b)
        {
            blocks[b].d = scale;
            int32_t sum = 0;
            for (int i = 0; i < BlockType::BLOCK_SIZE; ++i)
            {
                blocks[b].qs[i] = dist(rng_);
                sum += blocks[b].qs[i];
            }
            blocks[b].sum_qs = sum;
        }
        return blocks;
    }

    /**
     * @brief Create Q16_1 blocks from FP32 data with fixed scale
     */
    template <typename BlockType>
    std::vector<BlockType> quantizeFromFP32(
        const std::vector<float> &fp32_data,
        float scale)
    {
        constexpr int BLOCK_SIZE = BlockType::BLOCK_SIZE;
        int num_elements = static_cast<int>(fp32_data.size());
        int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

        std::vector<BlockType> blocks(num_blocks);
        float inv_scale = 1.0f / scale;

        for (int b = 0; b < num_blocks; ++b)
        {
            blocks[b].d = scale;
            int32_t sum = 0;
            int start = b * BLOCK_SIZE;
            for (int i = 0; i < BLOCK_SIZE; ++i)
            {
                int idx = start + i;
                float val = (idx < num_elements) ? fp32_data[idx] : 0.0f;
                int32_t q = static_cast<int32_t>(std::round(val * inv_scale));
                q = std::clamp(q, -32767, 32767);
                blocks[b].qs[i] = static_cast<int16_t>(q);
                sum += blocks[b].qs[i];
            }
            blocks[b].sum_qs = sum;
        }
        return blocks;
    }

    /**
     * @brief Create uniform head scales array
     */
    std::vector<float> createUniformScales(int num_heads, float scale = 1.0f)
    {
        return std::vector<float>(num_heads, scale);
    }

    /**
     * @brief Compute FP32 reference attention for comparison
     *
     * Standard attention: softmax(Q @ K^T / sqrt(d)) @ V
     */
    std::vector<float> referenceFP32Attention(
        const std::vector<float> &Q, // [num_heads, head_dim]
        const std::vector<float> &K, // [kv_len, num_kv_heads, head_dim]
        const std::vector<float> &V, // [kv_len, num_kv_heads, head_dim]
        int seq_len_q, int kv_len, int num_heads, int num_kv_heads, int head_dim,
        bool causal = false)
    {
        std::vector<float> output(seq_len_q * num_heads * head_dim, 0.0f);
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        for (int h = 0; h < num_heads; ++h)
        {
            int kv_h = (num_kv_heads == num_heads) ? h : (h / (num_heads / num_kv_heads));

            for (int q_pos = 0; q_pos < seq_len_q; ++q_pos)
            {
                // Compute Q @ K^T
                std::vector<float> scores(kv_len);
                float max_score = -std::numeric_limits<float>::infinity();

                for (int k_pos = 0; k_pos < kv_len; ++k_pos)
                {
                    // Causal mask: q_pos can only attend to k_pos <= q_pos
                    if (causal && k_pos > q_pos)
                    {
                        scores[k_pos] = -std::numeric_limits<float>::infinity();
                        continue;
                    }

                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d)
                    {
                        float q_val = Q[(h * seq_len_q + q_pos) * head_dim + d];
                        float k_val = K[(k_pos * num_kv_heads + kv_h) * head_dim + d];
                        dot += q_val * k_val;
                    }
                    scores[k_pos] = dot * scale;
                    max_score = std::max(max_score, scores[k_pos]);
                }

                // Softmax
                float sum = 0.0f;
                for (int k_pos = 0; k_pos < kv_len; ++k_pos)
                {
                    if (scores[k_pos] > -1e30f)
                    {
                        scores[k_pos] = std::exp(scores[k_pos] - max_score);
                        sum += scores[k_pos];
                    }
                    else
                    {
                        scores[k_pos] = 0.0f;
                    }
                }
                for (int k_pos = 0; k_pos < kv_len; ++k_pos)
                {
                    scores[k_pos] /= sum;
                }

                // Weighted sum of V
                for (int d = 0; d < head_dim; ++d)
                {
                    float val = 0.0f;
                    for (int k_pos = 0; k_pos < kv_len; ++k_pos)
                    {
                        float v_val = V[(k_pos * num_kv_heads + kv_h) * head_dim + d];
                        val += scores[k_pos] * v_val;
                    }
                    output[(h * seq_len_q + q_pos) * head_dim + d] = val;
                }
            }
        }
        return output;
    }

    /**
     * @brief Compute cosine similarity between two vectors
     */
    float cosineSimilarity(const float *a, const float *b, int n)
    {
        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        if (norm_a == 0.0f || norm_b == 0.0f)
            return 0.0f;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    /**
     * @brief Compute KL divergence between two probability distributions
     */
    float klDivergence(const float *p, const float *q, int n)
    {
        float kl = 0.0f;
        constexpr float eps = 1e-10f;
        for (int i = 0; i < n; ++i)
        {
            if (p[i] > eps)
            {
                kl += p[i] * std::log(p[i] / std::max(q[i], eps));
            }
        }
        return kl;
    }
};

// =============================================================================
// Parameter Validation Tests
// =============================================================================

TEST_F(Test__Q16IntegerAttention, Validate_NullPointers)
{
    Q16IntegerAttentionParams params;
    params.seq_len_q = 1;
    params.kv_len = 32;
    params.num_heads = 14;
    params.num_kv_heads = 2;
    params.head_dim = 64;
    params.block_size = Q16BlockSize::BLOCK_64;

    // Q, K, V all null
    EXPECT_FALSE(q16_validate_integer_params(params));

    // Only Q set
    int dummy_q = 0;
    params.Q = &dummy_q;
    EXPECT_FALSE(q16_validate_integer_params(params));

    // Q and K set
    int dummy_k = 0;
    params.K = &dummy_k;
    EXPECT_FALSE(q16_validate_integer_params(params));

    // All three set - should pass
    int dummy_v = 0;
    params.V = &dummy_v;
    EXPECT_TRUE(q16_validate_integer_params(params));
}

TEST_F(Test__Q16IntegerAttention, Validate_InvalidDimensions)
{
    Q16IntegerAttentionParams params;
    int dummy = 0;
    params.Q = &dummy;
    params.K = &dummy;
    params.V = &dummy;
    params.block_size = Q16BlockSize::BLOCK_64;

    // Zero seq_len_q
    params.seq_len_q = 0;
    params.kv_len = 32;
    params.num_heads = 14;
    params.num_kv_heads = 2;
    params.head_dim = 64;
    EXPECT_FALSE(q16_validate_integer_params(params));

    // Zero kv_len
    params.seq_len_q = 1;
    params.kv_len = 0;
    EXPECT_FALSE(q16_validate_integer_params(params));

    // Zero num_heads
    params.kv_len = 32;
    params.num_heads = 0;
    EXPECT_FALSE(q16_validate_integer_params(params));

    // Zero num_kv_heads
    params.num_heads = 14;
    params.num_kv_heads = 0;
    EXPECT_FALSE(q16_validate_integer_params(params));

    // Zero head_dim
    params.num_kv_heads = 2;
    params.head_dim = 0;
    EXPECT_FALSE(q16_validate_integer_params(params));
}

TEST_F(Test__Q16IntegerAttention, Validate_GQADivisibility)
{
    Q16IntegerAttentionParams params;
    int dummy = 0;
    params.Q = &dummy;
    params.K = &dummy;
    params.V = &dummy;
    params.seq_len_q = 1;
    params.kv_len = 32;
    params.head_dim = 64;
    params.block_size = Q16BlockSize::BLOCK_64;

    // num_heads not divisible by num_kv_heads
    params.num_heads = 14;
    params.num_kv_heads = 3; // 14 % 3 != 0
    EXPECT_FALSE(q16_validate_integer_params(params));

    // Valid GQA configuration
    params.num_kv_heads = 2; // 14 % 2 == 0
    EXPECT_TRUE(q16_validate_integer_params(params));

    // MHA (equal heads)
    params.num_heads = 12;
    params.num_kv_heads = 12;
    EXPECT_TRUE(q16_validate_integer_params(params));
}

TEST_F(Test__Q16IntegerAttention, Validate_InvalidBlockSize)
{
    Q16IntegerAttentionParams params;
    int dummy = 0;
    params.Q = &dummy;
    params.K = &dummy;
    params.V = &dummy;
    params.seq_len_q = 1;
    params.kv_len = 32;
    params.num_heads = 14;
    params.num_kv_heads = 2;
    params.head_dim = 64;

    // Invalid block size (use magic value)
    params.block_size = static_cast<Q16BlockSize>(99);
    EXPECT_FALSE(q16_validate_integer_params(params));

    // Valid block sizes
    params.block_size = Q16BlockSize::BLOCK_64;
    EXPECT_TRUE(q16_validate_integer_params(params));

    params.block_size = Q16BlockSize::BLOCK_128;
    EXPECT_TRUE(q16_validate_integer_params(params));
}

// =============================================================================
// Flash Decode Tests (seq_len_q = 1)
// =============================================================================

TEST_F(Test__Q16IntegerAttention, Decode_Block64_SingleHead)
{
    constexpr int HEAD_DIM = 64;
    constexpr int KV_LEN = 16;
    constexpr int NUM_HEADS = 1;
    constexpr int NUM_KV_HEADS = 1;
    constexpr int BLOCKS_PER_HEAD = 1; // HEAD_DIM / BLOCK_SIZE

    // Create Q, K, V blocks
    auto Q_blocks = createRandomBlocks<Q16_1Block_64>(NUM_HEADS * BLOCKS_PER_HEAD);
    auto K_blocks = createRandomBlocks<Q16_1Block_64>(KV_LEN * NUM_KV_HEADS * BLOCKS_PER_HEAD);
    auto V_blocks = createRandomBlocks<Q16_1Block_64>(KV_LEN * NUM_KV_HEADS * BLOCKS_PER_HEAD);

    // Create uniform head scales
    auto q_scales = createUniformScales(NUM_HEADS, 1.0f / 32767.0f);
    auto kv_scales = createUniformScales(NUM_KV_HEADS, 1.0f / 32767.0f);

    // Snapshot buffers
    std::vector<float> snapshot_weights(NUM_HEADS * KV_LEN);
    std::vector<float> snapshot_context(NUM_HEADS * HEAD_DIM);

    // Setup params
    Q16IntegerAttentionParams params;
    params.Q = Q_blocks.data();
    params.K = K_blocks.data();
    params.V = V_blocks.data();
    params.q_head_scales = q_scales.data();
    params.kv_head_scales = kv_scales.data();
    params.seq_len_q = 1;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = NUM_HEADS * HEAD_DIM;
    params.block_size = Q16BlockSize::BLOCK_64;
    params.snapshot_weights = snapshot_weights.data();
    params.snapshot_context = snapshot_context.data();

    // Execute
    bool success = q16_integer_attention_decode(params);
    EXPECT_TRUE(success);

    // Verify softmax weights sum to ~1.0
    float weight_sum = 0.0f;
    for (int k = 0; k < KV_LEN; ++k)
    {
        weight_sum += snapshot_weights[k];
    }
    EXPECT_NEAR(weight_sum, 1.0f, 0.01f) << "Softmax weights should sum to 1.0";

    // Verify no NaN/Inf in context
    for (int d = 0; d < HEAD_DIM; ++d)
    {
        EXPECT_FALSE(std::isnan(snapshot_context[d])) << "Context has NaN at d=" << d;
        EXPECT_FALSE(std::isinf(snapshot_context[d])) << "Context has Inf at d=" << d;
    }
}

TEST_F(Test__Q16IntegerAttention, Decode_Block64_MultiHead_GQA)
{
    constexpr int HEAD_DIM = 64;
    constexpr int KV_LEN = 32;
    constexpr int NUM_HEADS = 14;
    constexpr int NUM_KV_HEADS = 2;
    constexpr int BLOCKS_PER_HEAD = 1;

    auto Q_blocks = createRandomBlocks<Q16_1Block_64>(NUM_HEADS * BLOCKS_PER_HEAD);
    auto K_blocks = createRandomBlocks<Q16_1Block_64>(KV_LEN * NUM_KV_HEADS * BLOCKS_PER_HEAD);
    auto V_blocks = createRandomBlocks<Q16_1Block_64>(KV_LEN * NUM_KV_HEADS * BLOCKS_PER_HEAD);

    auto q_scales = createUniformScales(NUM_HEADS, 1.0f / 32767.0f);
    auto kv_scales = createUniformScales(NUM_KV_HEADS, 1.0f / 32767.0f);

    std::vector<float> snapshot_weights(NUM_HEADS * KV_LEN);
    std::vector<float> snapshot_context(NUM_HEADS * HEAD_DIM);

    Q16IntegerAttentionParams params;
    params.Q = Q_blocks.data();
    params.K = K_blocks.data();
    params.V = V_blocks.data();
    params.q_head_scales = q_scales.data();
    params.kv_head_scales = kv_scales.data();
    params.seq_len_q = 1;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = NUM_HEADS * HEAD_DIM;
    params.block_size = Q16BlockSize::BLOCK_64;
    params.snapshot_weights = snapshot_weights.data();
    params.snapshot_context = snapshot_context.data();

    bool success = q16_integer_attention_decode(params);
    EXPECT_TRUE(success);

    // Verify each head's weights sum to ~1.0
    for (int h = 0; h < NUM_HEADS; ++h)
    {
        float weight_sum = 0.0f;
        for (int k = 0; k < KV_LEN; ++k)
        {
            weight_sum += snapshot_weights[h * KV_LEN + k];
        }
        EXPECT_NEAR(weight_sum, 1.0f, 0.01f) << "Head " << h << " weights should sum to 1.0";
    }

    // Verify no NaN/Inf
    for (int h = 0; h < NUM_HEADS; ++h)
    {
        for (int d = 0; d < HEAD_DIM; ++d)
        {
            EXPECT_FALSE(std::isnan(snapshot_context[h * HEAD_DIM + d]))
                << "Context NaN at head=" << h << " d=" << d;
        }
    }
}

TEST_F(Test__Q16IntegerAttention, Decode_Block128_SingleHead)
{
    constexpr int HEAD_DIM = 128;
    constexpr int KV_LEN = 16;
    constexpr int NUM_HEADS = 1;
    constexpr int NUM_KV_HEADS = 1;
    constexpr int BLOCKS_PER_HEAD = 1;

    auto Q_blocks = createRandomBlocks<Q16_1Block_128>(NUM_HEADS * BLOCKS_PER_HEAD);
    auto K_blocks = createRandomBlocks<Q16_1Block_128>(KV_LEN * NUM_KV_HEADS * BLOCKS_PER_HEAD);
    auto V_blocks = createRandomBlocks<Q16_1Block_128>(KV_LEN * NUM_KV_HEADS * BLOCKS_PER_HEAD);

    auto q_scales = createUniformScales(NUM_HEADS, 1.0f / 32767.0f);
    auto kv_scales = createUniformScales(NUM_KV_HEADS, 1.0f / 32767.0f);

    std::vector<float> snapshot_weights(NUM_HEADS * KV_LEN);
    std::vector<float> snapshot_context(NUM_HEADS * HEAD_DIM);

    Q16IntegerAttentionParams params;
    params.Q = Q_blocks.data();
    params.K = K_blocks.data();
    params.V = V_blocks.data();
    params.q_head_scales = q_scales.data();
    params.kv_head_scales = kv_scales.data();
    params.seq_len_q = 1;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = NUM_HEADS * HEAD_DIM;
    params.block_size = Q16BlockSize::BLOCK_128;
    params.snapshot_weights = snapshot_weights.data();
    params.snapshot_context = snapshot_context.data();

    bool success = q16_integer_attention_decode(params);
    EXPECT_TRUE(success);

    float weight_sum = 0.0f;
    for (int k = 0; k < KV_LEN; ++k)
    {
        weight_sum += snapshot_weights[k];
    }
    EXPECT_NEAR(weight_sum, 1.0f, 0.01f);
}

TEST_F(Test__Q16IntegerAttention, Decode_LongSequence)
{
    constexpr int HEAD_DIM = 64;
    constexpr int KV_LEN = 512; // Longer sequence
    constexpr int NUM_HEADS = 4;
    constexpr int NUM_KV_HEADS = 4;
    constexpr int BLOCKS_PER_HEAD = 1;

    auto Q_blocks = createRandomBlocks<Q16_1Block_64>(NUM_HEADS * BLOCKS_PER_HEAD);
    auto K_blocks = createRandomBlocks<Q16_1Block_64>(KV_LEN * NUM_KV_HEADS * BLOCKS_PER_HEAD);
    auto V_blocks = createRandomBlocks<Q16_1Block_64>(KV_LEN * NUM_KV_HEADS * BLOCKS_PER_HEAD);

    auto q_scales = createUniformScales(NUM_HEADS, 1.0f / 32767.0f);
    auto kv_scales = createUniformScales(NUM_KV_HEADS, 1.0f / 32767.0f);

    std::vector<float> snapshot_weights(NUM_HEADS * KV_LEN);

    Q16IntegerAttentionParams params;
    params.Q = Q_blocks.data();
    params.K = K_blocks.data();
    params.V = V_blocks.data();
    params.q_head_scales = q_scales.data();
    params.kv_head_scales = kv_scales.data();
    params.seq_len_q = 1;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = NUM_HEADS * HEAD_DIM;
    params.block_size = Q16BlockSize::BLOCK_64;
    params.snapshot_weights = snapshot_weights.data();

    bool success = q16_integer_attention_decode(params);
    EXPECT_TRUE(success);

    // Verify numerical stability with long sequences
    for (int h = 0; h < NUM_HEADS; ++h)
    {
        float weight_sum = 0.0f;
        for (int k = 0; k < KV_LEN; ++k)
        {
            float w = snapshot_weights[h * KV_LEN + k];
            EXPECT_GE(w, 0.0f) << "Negative weight at head=" << h << " k=" << k;
            weight_sum += w;
        }
        EXPECT_NEAR(weight_sum, 1.0f, 0.02f) << "Long sequence weights sum diverged for head " << h;
    }
}

// =============================================================================
// FA2 Prefill Tests (seq_len_q > 1)
// =============================================================================

TEST_F(Test__Q16IntegerAttention, Prefill_Block64_SmallSequence)
{
    constexpr int HEAD_DIM = 64;
    constexpr int SEQ_LEN_Q = 8;
    constexpr int KV_LEN = 8;
    constexpr int NUM_HEADS = 2;
    constexpr int NUM_KV_HEADS = 2;
    constexpr int BLOCKS_PER_HEAD = 1;

    auto Q_blocks = createRandomBlocks<Q16_1Block_64>(SEQ_LEN_Q * NUM_HEADS * BLOCKS_PER_HEAD);
    auto K_blocks = createRandomBlocks<Q16_1Block_64>(KV_LEN * NUM_KV_HEADS * BLOCKS_PER_HEAD);
    auto V_blocks = createRandomBlocks<Q16_1Block_64>(KV_LEN * NUM_KV_HEADS * BLOCKS_PER_HEAD);

    auto q_scales = createUniformScales(NUM_HEADS, 1.0f / 32767.0f);
    auto kv_scales = createUniformScales(NUM_KV_HEADS, 1.0f / 32767.0f);

    std::vector<float> snapshot_weights(NUM_HEADS * SEQ_LEN_Q * KV_LEN);
    std::vector<float> snapshot_context(NUM_HEADS * SEQ_LEN_Q * HEAD_DIM);

    Q16IntegerAttentionParams params;
    params.Q = Q_blocks.data();
    params.K = K_blocks.data();
    params.V = V_blocks.data();
    params.q_head_scales = q_scales.data();
    params.kv_head_scales = kv_scales.data();
    params.seq_len_q = SEQ_LEN_Q;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = NUM_HEADS * HEAD_DIM;
    params.block_size = Q16BlockSize::BLOCK_64;
    params.snapshot_weights = snapshot_weights.data();
    params.snapshot_context = snapshot_context.data();

    bool success = q16_integer_attention_prefill(params);
    EXPECT_TRUE(success);

    // Verify each query's weights sum to ~1.0 (accounting for causal mask)
    for (int h = 0; h < NUM_HEADS; ++h)
    {
        for (int q = 0; q < SEQ_LEN_Q; ++q)
        {
            float weight_sum = 0.0f;
            int valid_positions = q + 1; // Causal mask: can attend to positions 0..q
            for (int k = 0; k < KV_LEN; ++k)
            {
                weight_sum += snapshot_weights[(h * SEQ_LEN_Q + q) * KV_LEN + k];
            }
            EXPECT_NEAR(weight_sum, 1.0f, 0.02f)
                << "Query " << q << " head " << h << " weights should sum to 1.0";
        }
    }
}

TEST_F(Test__Q16IntegerAttention, Prefill_Block64_MultiTile)
{
    // Test with sequence longer than tile size to exercise multi-tile logic
    constexpr int HEAD_DIM = 64;
    constexpr int SEQ_LEN_Q = 32; // > typical Br tile size
    constexpr int KV_LEN = 64;    // > typical Bc tile size
    constexpr int NUM_HEADS = 2;
    constexpr int NUM_KV_HEADS = 2;
    constexpr int BLOCKS_PER_HEAD = 1;

    auto Q_blocks = createRandomBlocks<Q16_1Block_64>(SEQ_LEN_Q * NUM_HEADS * BLOCKS_PER_HEAD);
    auto K_blocks = createRandomBlocks<Q16_1Block_64>(KV_LEN * NUM_KV_HEADS * BLOCKS_PER_HEAD);
    auto V_blocks = createRandomBlocks<Q16_1Block_64>(KV_LEN * NUM_KV_HEADS * BLOCKS_PER_HEAD);

    auto q_scales = createUniformScales(NUM_HEADS, 1.0f / 32767.0f);
    auto kv_scales = createUniformScales(NUM_KV_HEADS, 1.0f / 32767.0f);

    std::vector<float> snapshot_weights(NUM_HEADS * SEQ_LEN_Q * KV_LEN);

    Q16IntegerAttentionParams params;
    params.Q = Q_blocks.data();
    params.K = K_blocks.data();
    params.V = V_blocks.data();
    params.q_head_scales = q_scales.data();
    params.kv_head_scales = kv_scales.data();
    params.seq_len_q = SEQ_LEN_Q;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = NUM_HEADS * HEAD_DIM;
    params.block_size = Q16BlockSize::BLOCK_64;
    params.snapshot_weights = snapshot_weights.data();

    bool success = q16_integer_attention_prefill(params);
    EXPECT_TRUE(success);

    // Verify causal masking: weights after query position should be 0
    for (int h = 0; h < NUM_HEADS; ++h)
    {
        for (int q = 0; q < SEQ_LEN_Q; ++q)
        {
            for (int k = q + 1; k < KV_LEN; ++k)
            {
                float w = snapshot_weights[(h * SEQ_LEN_Q + q) * KV_LEN + k];
                EXPECT_NEAR(w, 0.0f, 1e-6f)
                    << "Causal mask violated: h=" << h << " q=" << q << " k=" << k;
            }
        }
    }
}

TEST_F(Test__Q16IntegerAttention, Prefill_Block128)
{
    constexpr int HEAD_DIM = 128;
    constexpr int SEQ_LEN_Q = 16;
    constexpr int KV_LEN = 16;
    constexpr int NUM_HEADS = 2;
    constexpr int NUM_KV_HEADS = 2;
    constexpr int BLOCKS_PER_HEAD = 1;

    auto Q_blocks = createRandomBlocks<Q16_1Block_128>(SEQ_LEN_Q * NUM_HEADS * BLOCKS_PER_HEAD);
    auto K_blocks = createRandomBlocks<Q16_1Block_128>(KV_LEN * NUM_KV_HEADS * BLOCKS_PER_HEAD);
    auto V_blocks = createRandomBlocks<Q16_1Block_128>(KV_LEN * NUM_KV_HEADS * BLOCKS_PER_HEAD);

    auto q_scales = createUniformScales(NUM_HEADS, 1.0f / 32767.0f);
    auto kv_scales = createUniformScales(NUM_KV_HEADS, 1.0f / 32767.0f);

    std::vector<float> snapshot_weights(NUM_HEADS * SEQ_LEN_Q * KV_LEN);

    Q16IntegerAttentionParams params;
    params.Q = Q_blocks.data();
    params.K = K_blocks.data();
    params.V = V_blocks.data();
    params.q_head_scales = q_scales.data();
    params.kv_head_scales = kv_scales.data();
    params.seq_len_q = SEQ_LEN_Q;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = NUM_HEADS * HEAD_DIM;
    params.block_size = Q16BlockSize::BLOCK_128;
    params.snapshot_weights = snapshot_weights.data();

    bool success = q16_integer_attention_prefill(params);
    EXPECT_TRUE(success);

    // Basic sanity check
    for (int h = 0; h < NUM_HEADS; ++h)
    {
        for (int q = 0; q < SEQ_LEN_Q; ++q)
        {
            float weight_sum = 0.0f;
            for (int k = 0; k < KV_LEN; ++k)
            {
                weight_sum += snapshot_weights[(h * SEQ_LEN_Q + q) * KV_LEN + k];
            }
            EXPECT_NEAR(weight_sum, 1.0f, 0.02f);
        }
    }
}

// =============================================================================
// Microkernel Integration Tests
// =============================================================================

TEST_F(Test__Q16IntegerAttention, QKDotProduct_Block64_OutputsValid)
{
    constexpr int HEAD_DIM = 64;
    constexpr int KV_LEN = 16;
    constexpr int BLOCKS_PER_HEAD = 1;

    auto Q_blocks = createRandomBlocks<Q16_1Block_64>(BLOCKS_PER_HEAD);
    auto K_blocks = createRandomBlocks<Q16_1Block_64>(KV_LEN * BLOCKS_PER_HEAD);

    // Compute using public dispatch API
    std::vector<int32_t> scores(KV_LEN);
    q16_integer_qk_dotproduct(
        Q_blocks.data(), K_blocks.data(),
        scores.data(), KV_LEN, HEAD_DIM,
        Q16BlockSize::BLOCK_64);

    // Verify outputs are valid (non-zero scores expected with random data)
    bool any_nonzero = false;
    for (int k = 0; k < KV_LEN; ++k)
    {
        if (scores[k] != 0)
            any_nonzero = true;
        // INT32 scores can be large with INT16 inputs
        EXPECT_GT(std::abs(scores[k]), 0) << "Score at k=" << k << " should be non-zero";
    }
    EXPECT_TRUE(any_nonzero) << "At least one score should be non-zero";
}

TEST_F(Test__Q16IntegerAttention, QKDotProduct_Block128_OutputsValid)
{
    constexpr int HEAD_DIM = 128;
    constexpr int KV_LEN = 8;
    constexpr int BLOCKS_PER_HEAD = 1;

    auto Q_blocks = createRandomBlocks<Q16_1Block_128>(BLOCKS_PER_HEAD);
    auto K_blocks = createRandomBlocks<Q16_1Block_128>(KV_LEN * BLOCKS_PER_HEAD);

    std::vector<int32_t> scores(KV_LEN);
    q16_integer_qk_dotproduct(
        Q_blocks.data(), K_blocks.data(),
        scores.data(), KV_LEN, HEAD_DIM,
        Q16BlockSize::BLOCK_128);

    bool any_nonzero = false;
    for (int k = 0; k < KV_LEN; ++k)
    {
        if (scores[k] != 0)
            any_nonzero = true;
    }
    EXPECT_TRUE(any_nonzero) << "At least one score should be non-zero";
}

TEST_F(Test__Q16IntegerAttention, PVAccumulate_Block64_OutputsValid)
{
    constexpr int HEAD_DIM = 64;
    constexpr int KV_LEN = 16;
    constexpr int BLOCKS_PER_HEAD = 1;

    auto V_blocks = createRandomBlocks<Q16_1Block_64>(KV_LEN * BLOCKS_PER_HEAD);

    // Create weights (uniform for simplicity - sum to 32767)
    std::vector<int16_t> weights(KV_LEN, 32767 / KV_LEN);

    std::vector<int32_t> context(HEAD_DIM, 0);
    q16_integer_pv_accumulate(
        weights.data(), V_blocks.data(),
        context.data(), KV_LEN, HEAD_DIM,
        Q16BlockSize::BLOCK_64);

    // Verify outputs are valid (non-zero context expected)
    bool any_nonzero = false;
    for (int d = 0; d < HEAD_DIM; ++d)
    {
        if (context[d] != 0)
            any_nonzero = true;
    }
    EXPECT_TRUE(any_nonzero) << "At least one context element should be non-zero";
}

TEST_F(Test__Q16IntegerAttention, PVAccumulate_Block128_OutputsValid)
{
    constexpr int HEAD_DIM = 128;
    constexpr int KV_LEN = 8;
    constexpr int BLOCKS_PER_HEAD = 1;

    auto V_blocks = createRandomBlocks<Q16_1Block_128>(KV_LEN * BLOCKS_PER_HEAD);

    std::vector<int16_t> weights(KV_LEN, 32767 / KV_LEN);

    std::vector<int32_t> context(HEAD_DIM, 0);
    q16_integer_pv_accumulate(
        weights.data(), V_blocks.data(),
        context.data(), KV_LEN, HEAD_DIM,
        Q16BlockSize::BLOCK_128);

    bool any_nonzero = false;
    for (int d = 0; d < HEAD_DIM; ++d)
    {
        if (context[d] != 0)
            any_nonzero = true;
    }
    EXPECT_TRUE(any_nonzero) << "At least one context element should be non-zero";
}

// =============================================================================
// Snapshot Capture Tests
// =============================================================================

TEST_F(Test__Q16IntegerAttention, Snapshot_WeightsCapture)
{
    constexpr int HEAD_DIM = 64;
    constexpr int KV_LEN = 16;
    constexpr int NUM_HEADS = 2;
    constexpr int NUM_KV_HEADS = 2;

    auto Q_blocks = createRandomBlocks<Q16_1Block_64>(NUM_HEADS);
    auto K_blocks = createRandomBlocks<Q16_1Block_64>(KV_LEN * NUM_KV_HEADS);
    auto V_blocks = createRandomBlocks<Q16_1Block_64>(KV_LEN * NUM_KV_HEADS);

    auto q_scales = createUniformScales(NUM_HEADS, 1.0f / 32767.0f);
    auto kv_scales = createUniformScales(NUM_KV_HEADS, 1.0f / 32767.0f);

    // With snapshot
    std::vector<float> snapshot_weights(NUM_HEADS * KV_LEN);
    Q16IntegerAttentionParams params;
    params.Q = Q_blocks.data();
    params.K = K_blocks.data();
    params.V = V_blocks.data();
    params.q_head_scales = q_scales.data();
    params.kv_head_scales = kv_scales.data();
    params.seq_len_q = 1;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = NUM_HEADS * HEAD_DIM;
    params.block_size = Q16BlockSize::BLOCK_64;
    params.snapshot_weights = snapshot_weights.data();

    bool success = q16_integer_attention_decode(params);
    EXPECT_TRUE(success);

    // Verify weights are captured (not all zeros)
    float total = 0.0f;
    for (size_t i = 0; i < snapshot_weights.size(); ++i)
    {
        total += std::abs(snapshot_weights[i]);
    }
    EXPECT_GT(total, 0.0f) << "Snapshot weights should be non-zero";
}

TEST_F(Test__Q16IntegerAttention, Snapshot_ContextCapture)
{
    constexpr int HEAD_DIM = 64;
    constexpr int KV_LEN = 16;
    constexpr int NUM_HEADS = 2;
    constexpr int NUM_KV_HEADS = 2;

    auto Q_blocks = createRandomBlocks<Q16_1Block_64>(NUM_HEADS);
    auto K_blocks = createRandomBlocks<Q16_1Block_64>(KV_LEN * NUM_KV_HEADS);
    auto V_blocks = createRandomBlocks<Q16_1Block_64>(KV_LEN * NUM_KV_HEADS);

    auto q_scales = createUniformScales(NUM_HEADS, 1.0f / 32767.0f);
    auto kv_scales = createUniformScales(NUM_KV_HEADS, 1.0f / 32767.0f);

    std::vector<float> snapshot_context(NUM_HEADS * HEAD_DIM);

    Q16IntegerAttentionParams params;
    params.Q = Q_blocks.data();
    params.K = K_blocks.data();
    params.V = V_blocks.data();
    params.q_head_scales = q_scales.data();
    params.kv_head_scales = kv_scales.data();
    params.seq_len_q = 1;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = NUM_HEADS * HEAD_DIM;
    params.block_size = Q16BlockSize::BLOCK_64;
    params.snapshot_context = snapshot_context.data();

    bool success = q16_integer_attention_decode(params);
    EXPECT_TRUE(success);

    // Verify context is captured (not all zeros)
    float total = 0.0f;
    for (size_t i = 0; i < snapshot_context.size(); ++i)
    {
        total += std::abs(snapshot_context[i]);
    }
    EXPECT_GT(total, 0.0f) << "Snapshot context should be non-zero";
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(Test__Q16IntegerAttention, EdgeCase_SingleKVPosition)
{
    constexpr int HEAD_DIM = 64;
    constexpr int KV_LEN = 1;
    constexpr int NUM_HEADS = 1;
    constexpr int NUM_KV_HEADS = 1;

    auto Q_blocks = createRandomBlocks<Q16_1Block_64>(NUM_HEADS);
    auto K_blocks = createRandomBlocks<Q16_1Block_64>(KV_LEN * NUM_KV_HEADS);
    auto V_blocks = createRandomBlocks<Q16_1Block_64>(KV_LEN * NUM_KV_HEADS);

    auto q_scales = createUniformScales(NUM_HEADS, 1.0f / 32767.0f);
    auto kv_scales = createUniformScales(NUM_KV_HEADS, 1.0f / 32767.0f);

    std::vector<float> snapshot_weights(NUM_HEADS * KV_LEN);

    Q16IntegerAttentionParams params;
    params.Q = Q_blocks.data();
    params.K = K_blocks.data();
    params.V = V_blocks.data();
    params.q_head_scales = q_scales.data();
    params.kv_head_scales = kv_scales.data();
    params.seq_len_q = 1;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = NUM_HEADS * HEAD_DIM;
    params.block_size = Q16BlockSize::BLOCK_64;
    params.snapshot_weights = snapshot_weights.data();

    bool success = q16_integer_attention_decode(params);
    EXPECT_TRUE(success);

    // Single position should get weight 1.0
    EXPECT_NEAR(snapshot_weights[0], 1.0f, 0.01f);
}

TEST_F(Test__Q16IntegerAttention, EdgeCase_ManyHeads)
{
    constexpr int HEAD_DIM = 64;
    constexpr int KV_LEN = 16;
    constexpr int NUM_HEADS = 32;
    constexpr int NUM_KV_HEADS = 8;

    auto Q_blocks = createRandomBlocks<Q16_1Block_64>(NUM_HEADS);
    auto K_blocks = createRandomBlocks<Q16_1Block_64>(KV_LEN * NUM_KV_HEADS);
    auto V_blocks = createRandomBlocks<Q16_1Block_64>(KV_LEN * NUM_KV_HEADS);

    auto q_scales = createUniformScales(NUM_HEADS, 1.0f / 32767.0f);
    auto kv_scales = createUniformScales(NUM_KV_HEADS, 1.0f / 32767.0f);

    Q16IntegerAttentionParams params;
    params.Q = Q_blocks.data();
    params.K = K_blocks.data();
    params.V = V_blocks.data();
    params.q_head_scales = q_scales.data();
    params.kv_head_scales = kv_scales.data();
    params.seq_len_q = 1;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = NUM_HEADS * HEAD_DIM;
    params.block_size = Q16BlockSize::BLOCK_64;

    bool success = q16_integer_attention_decode(params);
    EXPECT_TRUE(success);
}
// =============================================================================
// Per-Position K Scale Tests (Phase 8: Option C - Pass Scale to Softmax)
// =============================================================================

/**
 * @test Verify per-position K scales feature (has_per_position_k_scales())
 *
 * When k_position_scales is provided, the attention kernel should use
 * per-position K scales instead of uniform kv_head_scales for Q×K^T scaling.
 */
TEST_F(Test__Q16IntegerAttention, PerPositionKScales_FlagCheck)
{
    constexpr int HEAD_DIM = 64;
    constexpr int KV_LEN = 4;
    constexpr int NUM_HEADS = 2;
    constexpr int NUM_KV_HEADS = 1;

    auto Q_blocks = createRandomBlocks<Q16_1Block_64>(NUM_HEADS);
    auto K_blocks = createRandomBlocks<Q16_1Block_64>(KV_LEN * NUM_KV_HEADS);
    auto V_blocks = createRandomBlocks<Q16_1Block_64>(KV_LEN * NUM_KV_HEADS);

    auto q_scales = createUniformScales(NUM_HEADS, 1.0f / 32767.0f);
    auto kv_scales = createUniformScales(NUM_KV_HEADS, 1.0f / 32767.0f);

    // Per-position K scales: different scale for each position
    std::vector<float> k_pos_scales(KV_LEN * NUM_KV_HEADS);
    for (int pos = 0; pos < KV_LEN; ++pos)
    {
        for (int kv_h = 0; kv_h < NUM_KV_HEADS; ++kv_h)
        {
            k_pos_scales[pos * NUM_KV_HEADS + kv_h] = 1.0f / (32767.0f * (1.0f + 0.1f * pos));
        }
    }

    Q16IntegerAttentionParams params;
    params.Q = Q_blocks.data();
    params.K = K_blocks.data();
    params.V = V_blocks.data();
    params.q_head_scales = q_scales.data();
    params.kv_head_scales = kv_scales.data();
    params.k_position_scales = k_pos_scales.data();
    params.seq_len_q = 1;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = NUM_HEADS * HEAD_DIM;
    params.block_size = Q16BlockSize::BLOCK_64;

    // Verify the flag is set
    EXPECT_TRUE(params.has_per_position_k_scales());

    // Verify get_k_scale returns per-position scales
    EXPECT_FLOAT_EQ(params.get_k_scale(0, 0), k_pos_scales[0]);
    EXPECT_FLOAT_EQ(params.get_k_scale(1, 0), k_pos_scales[1]);
    EXPECT_FLOAT_EQ(params.get_k_scale(3, 0), k_pos_scales[3]);

    // Verify Q scale
    EXPECT_FLOAT_EQ(params.get_q_scale(0), q_scales[0]);
    EXPECT_FLOAT_EQ(params.get_q_scale(1), q_scales[1]);
}

/**
 * @test Verify per-position K scales produce different results from uniform scales
 *
 * Run attention twice: once with uniform kv_head_scales, once with per-position
 * k_position_scales. The outputs should differ when K scales vary by position.
 */
TEST_F(Test__Q16IntegerAttention, PerPositionKScales_OutputDiffers)
{
    constexpr int HEAD_DIM = 64;
    constexpr int KV_LEN = 8;
    constexpr int NUM_HEADS = 2;
    constexpr int NUM_KV_HEADS = 1;

    auto Q_blocks = createRandomBlocks<Q16_1Block_64>(NUM_HEADS);
    auto K_blocks = createRandomBlocks<Q16_1Block_64>(KV_LEN * NUM_KV_HEADS);
    auto V_blocks = createRandomBlocks<Q16_1Block_64>(KV_LEN * NUM_KV_HEADS);

    auto q_scales = createUniformScales(NUM_HEADS, 1.0f / 32767.0f);
    auto kv_scales = createUniformScales(NUM_KV_HEADS, 1.0f / 32767.0f);

    // Per-position K scales with varying values
    std::vector<float> k_pos_scales(KV_LEN * NUM_KV_HEADS);
    for (int pos = 0; pos < KV_LEN; ++pos)
    {
        for (int kv_h = 0; kv_h < NUM_KV_HEADS; ++kv_h)
        {
            // Significant variation: scale = 1/32767 * (0.5 + 0.1*pos)
            k_pos_scales[pos * NUM_KV_HEADS + kv_h] = (0.5f + 0.1f * pos) / 32767.0f;
        }
    }

    // Output buffers for both runs
    std::vector<float> context_uniform(NUM_HEADS * HEAD_DIM);
    std::vector<float> context_per_pos(NUM_HEADS * HEAD_DIM);

    // Run 1: Uniform K scales (standard)
    {
        Q16IntegerAttentionParams params;
        params.Q = Q_blocks.data();
        params.K = K_blocks.data();
        params.V = V_blocks.data();
        params.q_head_scales = q_scales.data();
        params.kv_head_scales = kv_scales.data();
        params.k_position_scales = nullptr; // Standard mode
        params.seq_len_q = 1;
        params.kv_len = KV_LEN;
        params.num_heads = NUM_HEADS;
        params.num_kv_heads = NUM_KV_HEADS;
        params.head_dim = HEAD_DIM;
        params.d_model = NUM_HEADS * HEAD_DIM;
        params.block_size = Q16BlockSize::BLOCK_64;
        params.snapshot_context = context_uniform.data();

        bool success = q16_integer_attention_decode(params);
        ASSERT_TRUE(success);
    }

    // Run 2: Per-position K scales (Option C)
    {
        Q16IntegerAttentionParams params;
        params.Q = Q_blocks.data();
        params.K = K_blocks.data();
        params.V = V_blocks.data();
        params.q_head_scales = q_scales.data();
        params.kv_head_scales = kv_scales.data();
        params.k_position_scales = k_pos_scales.data(); // Per-position mode
        params.seq_len_q = 1;
        params.kv_len = KV_LEN;
        params.num_heads = NUM_HEADS;
        params.num_kv_heads = NUM_KV_HEADS;
        params.head_dim = HEAD_DIM;
        params.d_model = NUM_HEADS * HEAD_DIM;
        params.block_size = Q16BlockSize::BLOCK_64;
        params.snapshot_context = context_per_pos.data();

        bool success = q16_integer_attention_decode(params);
        ASSERT_TRUE(success);
    }

    // Compare: outputs should differ since K scales are different
    float max_diff = 0.0f;
    for (size_t i = 0; i < context_uniform.size(); ++i)
    {
        float diff = std::abs(context_uniform[i] - context_per_pos[i]);
        max_diff = std::max(max_diff, diff);
    }

    // The difference should be non-trivial (K scales differ significantly)
    EXPECT_GT(max_diff, 0.0f) << "Per-position K scales should produce different output";

    std::cout << "[PerPositionKScales] Max diff between uniform and per-position: " << max_diff << std::endl;
}

/**
 * @test Verify per-position K scales with BLOCK_128
 */
TEST_F(Test__Q16IntegerAttention, PerPositionKScales_Block128)
{
    constexpr int HEAD_DIM = 128;
    constexpr int KV_LEN = 4;
    constexpr int NUM_HEADS = 2;
    constexpr int NUM_KV_HEADS = 1;

    auto Q_blocks = createRandomBlocks<Q16_1Block_128>(NUM_HEADS);
    auto K_blocks = createRandomBlocks<Q16_1Block_128>(KV_LEN * NUM_KV_HEADS);
    auto V_blocks = createRandomBlocks<Q16_1Block_128>(KV_LEN * NUM_KV_HEADS);

    auto q_scales = createUniformScales(NUM_HEADS, 1.0f / 32767.0f);
    auto kv_scales = createUniformScales(NUM_KV_HEADS, 1.0f / 32767.0f);

    // Per-position K scales
    std::vector<float> k_pos_scales(KV_LEN * NUM_KV_HEADS);
    for (int pos = 0; pos < KV_LEN; ++pos)
    {
        for (int kv_h = 0; kv_h < NUM_KV_HEADS; ++kv_h)
        {
            k_pos_scales[pos * NUM_KV_HEADS + kv_h] = (1.0f + 0.2f * pos) / 32767.0f;
        }
    }

    std::vector<float> context(NUM_HEADS * HEAD_DIM);

    Q16IntegerAttentionParams params;
    params.Q = Q_blocks.data();
    params.K = K_blocks.data();
    params.V = V_blocks.data();
    params.q_head_scales = q_scales.data();
    params.kv_head_scales = kv_scales.data();
    params.k_position_scales = k_pos_scales.data();
    params.seq_len_q = 1;
    params.kv_len = KV_LEN;
    params.num_heads = NUM_HEADS;
    params.num_kv_heads = NUM_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.d_model = NUM_HEADS * HEAD_DIM;
    params.block_size = Q16BlockSize::BLOCK_128;
    params.snapshot_context = context.data();

    EXPECT_TRUE(params.has_per_position_k_scales());

    bool success = q16_integer_attention_decode(params);
    EXPECT_TRUE(success);

    // Verify output is non-zero
    float sum = 0.0f;
    for (float c : context)
    {
        sum += std::abs(c);
    }
    EXPECT_GT(sum, 0.0f) << "Context should be non-zero";
}