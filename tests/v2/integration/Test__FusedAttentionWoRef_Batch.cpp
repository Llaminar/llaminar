/**
 * @file Test__FusedAttentionWoRef_Batch.cpp
 * @brief Tests for batched execution of fused attention + Wo kernel.
 *
 * Tests cover:
 * - Prefill (seq_len = prompt_length)
 * - Incremental decode (seq_len = 1, position_offset)
 * - Batched decode uniform KV lengths
 * - Batched decode variable KV lengths
 * - Long sequence support
 *
 * @author David Sanftenberg
 * @date December 2025
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>

#include "../../../src/v2/kernels/cpu/attention/q8_1/FusedAttentionWoRef.h"
#include "../../../src/v2/kernels/cpu/microkernels/q8_1/Q8DotProduct.h"
#include "../../../src/v2/tensors/FP16Utils.h"

using namespace llaminar::v2::kernels;
using namespace llaminar::v2::kernels::microkernels;

// =============================================================================
// Test Helpers
// =============================================================================

void quantize_to_q8_1(const float *input, Q8_1Block *output, int numel)
{
    const int num_blocks = numel / 32;

    for (int b = 0; b < num_blocks; ++b)
    {
        const float *block_data = input + b * 32;

        float absmax = 0.0f;
        for (int i = 0; i < 32; ++i)
        {
            absmax = std::max(absmax, std::abs(block_data[i]));
        }

        float scale = absmax > 0 ? absmax / 127.0f : 1.0f;
        float inv_scale = absmax > 0 ? 127.0f / absmax : 0.0f;

        output[b].d = llaminar2::fp32_to_fp16(scale);

        int32_t sum = 0;
        for (int i = 0; i < 32; ++i)
        {
            int8_t q = static_cast<int8_t>(std::round(block_data[i] * inv_scale));
            q = std::max(int8_t(-128), std::min(int8_t(127), q));
            output[b].qs[i] = q;
            sum += q;
        }
        output[b].sum_qs = static_cast<int16_t>(sum);
    }
}

void dequantize_from_q8_1(const Q8_1Block *input, float *output, int numel)
{
    const int num_blocks = numel / 32;

    for (int b = 0; b < num_blocks; ++b)
    {
        float scale = llaminar2::fp16_to_fp32(input[b].d);
        for (int i = 0; i < 32; ++i)
        {
            output[b * 32 + i] = static_cast<float>(input[b].qs[i]) * scale;
        }
    }
}

float cosine_similarity(const float *a, const float *b, int n)
{
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < n; ++i)
    {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a < 1e-10f || norm_b < 1e-10f)
        return 0.0f;
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

class FusedAttentionWoBatchTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        rng_.seed(42);
    }

    void generate_random_fp32(float *data, int n, float scale = 1.0f)
    {
        std::uniform_real_distribution<float> dist(-scale, scale);
        for (int i = 0; i < n; ++i)
        {
            data[i] = dist(rng_);
        }
    }

    std::mt19937 rng_;
};

// =============================================================================
// PREFILL TESTS - Single sequence, varying lengths
// =============================================================================

TEST_F(FusedAttentionWoBatchTest, Prefill_ShortSequence)
{
    // Prefill with seq_len = 8
    const int seq_len = 8;
    const int kv_seq_len = 8;
    const int num_heads = 2;
    const int num_kv_heads = 2;
    const int head_dim = 64;
    const int d_model = num_heads * head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int num_blocks = head_dim / 32;

    // Allocate tensors
    std::vector<float> Q_fp32(seq_len * num_heads * head_dim);
    std::vector<float> K_fp32(kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> V_fp32(kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> Wo_fp32(d_model * num_heads * head_dim);

    generate_random_fp32(Q_fp32.data(), Q_fp32.size());
    generate_random_fp32(K_fp32.data(), K_fp32.size());
    generate_random_fp32(V_fp32.data(), V_fp32.size());
    generate_random_fp32(Wo_fp32.data(), Wo_fp32.size());

    // Quantize
    std::vector<Q8_1Block> Q_q8(seq_len * num_heads * num_blocks);
    std::vector<Q8_1Block> K_q8(kv_seq_len * num_kv_heads * num_blocks);
    std::vector<Q8_1Block> V_q8(kv_seq_len * num_kv_heads * num_blocks);

    quantize_to_q8_1(Q_fp32.data(), Q_q8.data(), Q_fp32.size());
    quantize_to_q8_1(K_fp32.data(), K_q8.data(), K_fp32.size());
    quantize_to_q8_1(V_fp32.data(), V_q8.data(), V_fp32.size());

    // Output
    std::vector<float> output(seq_len * d_model, 0.0f);

    // Execute
    FusedAttentionWoParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.Wo = Wo_fp32.data();
    params.wo_type = WoWeightType::FP32;
    params.output = output.data();
    params.batch_size = 1;
    params.seq_len = seq_len;
    params.kv_seq_len = kv_seq_len;
    params.num_heads = num_heads;
    params.num_kv_heads = num_kv_heads;
    params.head_dim = head_dim;
    params.d_model = d_model;
    params.scale = scale;
    params.causal = true;
    params.position_offset = 0;

    ASSERT_TRUE(FusedAttentionWoRef::execute(params));

    // Verify output is non-zero (basic sanity check)
    float sum = 0.0f;
    for (int i = 0; i < seq_len * d_model; ++i)
    {
        sum += std::abs(output[i]);
    }
    EXPECT_GT(sum, 0.0f) << "Output should be non-zero";
}

TEST_F(FusedAttentionWoBatchTest, Prefill_LongSequence)
{
    // Prefill with seq_len = 512
    const int seq_len = 512;
    const int kv_seq_len = 512;
    const int num_heads = 2;
    const int num_kv_heads = 2;
    const int head_dim = 64;
    const int d_model = num_heads * head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int num_blocks = head_dim / 32;

    std::vector<float> Q_fp32(seq_len * num_heads * head_dim);
    std::vector<float> K_fp32(kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> V_fp32(kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> Wo_fp32(d_model * num_heads * head_dim);

    generate_random_fp32(Q_fp32.data(), Q_fp32.size());
    generate_random_fp32(K_fp32.data(), K_fp32.size());
    generate_random_fp32(V_fp32.data(), V_fp32.size());
    generate_random_fp32(Wo_fp32.data(), Wo_fp32.size());

    std::vector<Q8_1Block> Q_q8(seq_len * num_heads * num_blocks);
    std::vector<Q8_1Block> K_q8(kv_seq_len * num_kv_heads * num_blocks);
    std::vector<Q8_1Block> V_q8(kv_seq_len * num_kv_heads * num_blocks);

    quantize_to_q8_1(Q_fp32.data(), Q_q8.data(), Q_fp32.size());
    quantize_to_q8_1(K_fp32.data(), K_q8.data(), K_fp32.size());
    quantize_to_q8_1(V_fp32.data(), V_q8.data(), V_fp32.size());

    std::vector<float> output(seq_len * d_model, 0.0f);

    FusedAttentionWoParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.Wo = Wo_fp32.data();
    params.wo_type = WoWeightType::FP32;
    params.output = output.data();
    params.batch_size = 1;
    params.seq_len = seq_len;
    params.kv_seq_len = kv_seq_len;
    params.num_heads = num_heads;
    params.num_kv_heads = num_kv_heads;
    params.head_dim = head_dim;
    params.d_model = d_model;
    params.scale = scale;
    params.causal = true;
    params.position_offset = 0;

    ASSERT_TRUE(FusedAttentionWoRef::execute(params));

    // Verify non-zero output
    float sum = 0.0f;
    for (int i = 0; i < seq_len * d_model; ++i)
    {
        sum += std::abs(output[i]);
    }
    EXPECT_GT(sum, 0.0f);
}

// =============================================================================
// INCREMENTAL DECODE TESTS - seq_len = 1 with position_offset
// =============================================================================

TEST_F(FusedAttentionWoBatchTest, IncrementalDecode_SingleStep)
{
    // Decode single token at position 10 (kv_cache has 10 entries)
    const int seq_len = 1;
    const int kv_seq_len = 11;      // 0-10, query at position 10
    const int position_offset = 10; // Query's global position
    const int num_heads = 2;
    const int num_kv_heads = 2;
    const int head_dim = 64;
    const int d_model = num_heads * head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int num_blocks = head_dim / 32;

    std::vector<float> Q_fp32(seq_len * num_heads * head_dim);
    std::vector<float> K_fp32(kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> V_fp32(kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> Wo_fp32(d_model * num_heads * head_dim);

    generate_random_fp32(Q_fp32.data(), Q_fp32.size());
    generate_random_fp32(K_fp32.data(), K_fp32.size());
    generate_random_fp32(V_fp32.data(), V_fp32.size());
    generate_random_fp32(Wo_fp32.data(), Wo_fp32.size());

    std::vector<Q8_1Block> Q_q8(seq_len * num_heads * num_blocks);
    std::vector<Q8_1Block> K_q8(kv_seq_len * num_kv_heads * num_blocks);
    std::vector<Q8_1Block> V_q8(kv_seq_len * num_kv_heads * num_blocks);

    quantize_to_q8_1(Q_fp32.data(), Q_q8.data(), Q_fp32.size());
    quantize_to_q8_1(K_fp32.data(), K_q8.data(), K_fp32.size());
    quantize_to_q8_1(V_fp32.data(), V_q8.data(), V_fp32.size());

    std::vector<float> output(seq_len * d_model, 0.0f);

    FusedAttentionWoParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.Wo = Wo_fp32.data();
    params.wo_type = WoWeightType::FP32;
    params.output = output.data();
    params.batch_size = 1;
    params.seq_len = seq_len;
    params.kv_seq_len = kv_seq_len;
    params.num_heads = num_heads;
    params.num_kv_heads = num_kv_heads;
    params.head_dim = head_dim;
    params.d_model = d_model;
    params.scale = scale;
    params.causal = true;
    params.position_offset = position_offset;

    ASSERT_TRUE(FusedAttentionWoRef::execute(params));

    // Verify non-zero output
    float sum = 0.0f;
    for (int i = 0; i < d_model; ++i)
    {
        sum += std::abs(output[i]);
    }
    EXPECT_GT(sum, 0.0f);
}

TEST_F(FusedAttentionWoBatchTest, IncrementalDecode_GrowingKVCache)
{
    // Simulate multiple decode steps with growing KV cache
    const int num_heads = 2;
    const int num_kv_heads = 2;
    const int head_dim = 64;
    const int d_model = num_heads * head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int num_blocks = head_dim / 32;

    // Preallocate for max KV cache size
    const int max_kv_len = 20;
    std::vector<float> K_fp32(max_kv_len * num_kv_heads * head_dim);
    std::vector<float> V_fp32(max_kv_len * num_kv_heads * head_dim);
    std::vector<float> Wo_fp32(d_model * num_heads * head_dim);

    generate_random_fp32(K_fp32.data(), K_fp32.size());
    generate_random_fp32(V_fp32.data(), V_fp32.size());
    generate_random_fp32(Wo_fp32.data(), Wo_fp32.size());

    std::vector<Q8_1Block> K_q8(max_kv_len * num_kv_heads * num_blocks);
    std::vector<Q8_1Block> V_q8(max_kv_len * num_kv_heads * num_blocks);

    quantize_to_q8_1(K_fp32.data(), K_q8.data(), K_fp32.size());
    quantize_to_q8_1(V_fp32.data(), V_q8.data(), V_fp32.size());

    // Simulate 5 decode steps starting from position 10
    for (int step = 0; step < 5; ++step)
    {
        const int position = 10 + step;
        const int kv_len = position + 1; // KV cache grows each step

        SCOPED_TRACE("Decode step " + std::to_string(step) + ", position " + std::to_string(position));

        // Generate new Q for this decode step
        std::vector<float> Q_fp32(1 * num_heads * head_dim);
        generate_random_fp32(Q_fp32.data(), Q_fp32.size());

        std::vector<Q8_1Block> Q_q8(1 * num_heads * num_blocks);
        quantize_to_q8_1(Q_fp32.data(), Q_q8.data(), Q_fp32.size());

        std::vector<float> output(d_model, 0.0f);

        FusedAttentionWoParams params;
        params.Q = Q_q8.data();
        params.K = K_q8.data();
        params.V = V_q8.data();
        params.Wo = Wo_fp32.data();
        params.wo_type = WoWeightType::FP32;
        params.output = output.data();
        params.batch_size = 1;
        params.seq_len = 1;
        params.kv_seq_len = kv_len;
        params.num_heads = num_heads;
        params.num_kv_heads = num_kv_heads;
        params.head_dim = head_dim;
        params.d_model = d_model;
        params.scale = scale;
        params.causal = true;
        params.position_offset = position;

        ASSERT_TRUE(FusedAttentionWoRef::execute(params));

        float sum = 0.0f;
        for (int i = 0; i < d_model; ++i)
        {
            sum += std::abs(output[i]);
        }
        EXPECT_GT(sum, 0.0f);
    }
}

// =============================================================================
// BATCHED DECODE - Uniform KV lengths
// =============================================================================

TEST_F(FusedAttentionWoBatchTest, BatchedDecode_UniformKV)
{
    // Batch of 4 sequences, all with same KV length
    const int batch_size = 4;
    const int seq_len = 1; // Decode mode
    const int kv_seq_len = 16;
    const int position_offset = 15; // All at same position
    const int num_heads = 2;
    const int num_kv_heads = 2;
    const int head_dim = 64;
    const int d_model = num_heads * head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int num_blocks = head_dim / 32;

    // Batched tensors
    std::vector<float> Q_fp32(batch_size * seq_len * num_heads * head_dim);
    std::vector<float> K_fp32(batch_size * kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> V_fp32(batch_size * kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> Wo_fp32(d_model * num_heads * head_dim);

    generate_random_fp32(Q_fp32.data(), Q_fp32.size());
    generate_random_fp32(K_fp32.data(), K_fp32.size());
    generate_random_fp32(V_fp32.data(), V_fp32.size());
    generate_random_fp32(Wo_fp32.data(), Wo_fp32.size());

    std::vector<Q8_1Block> Q_q8(batch_size * seq_len * num_heads * num_blocks);
    std::vector<Q8_1Block> K_q8(batch_size * kv_seq_len * num_kv_heads * num_blocks);
    std::vector<Q8_1Block> V_q8(batch_size * kv_seq_len * num_kv_heads * num_blocks);

    quantize_to_q8_1(Q_fp32.data(), Q_q8.data(), Q_fp32.size());
    quantize_to_q8_1(K_fp32.data(), K_q8.data(), K_fp32.size());
    quantize_to_q8_1(V_fp32.data(), V_q8.data(), V_fp32.size());

    std::vector<float> output_batched(batch_size * seq_len * d_model, 0.0f);

    // Execute batched
    FusedAttentionWoParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.Wo = Wo_fp32.data();
    params.wo_type = WoWeightType::FP32;
    params.output = output_batched.data();
    params.batch_size = batch_size;
    params.seq_len = seq_len;
    params.kv_seq_len = kv_seq_len;
    params.num_heads = num_heads;
    params.num_kv_heads = num_kv_heads;
    params.head_dim = head_dim;
    params.d_model = d_model;
    params.scale = scale;
    params.causal = true;
    params.position_offset = position_offset;

    ASSERT_TRUE(FusedAttentionWoRef::execute(params));

    // Verify each batch has non-zero output
    for (int b = 0; b < batch_size; ++b)
    {
        float sum = 0.0f;
        for (int i = 0; i < d_model; ++i)
        {
            sum += std::abs(output_batched[b * d_model + i]);
        }
        EXPECT_GT(sum, 0.0f) << "Batch " << b << " should have non-zero output";
    }
}

TEST_F(FusedAttentionWoBatchTest, BatchedDecode_MatchesSingleSequence)
{
    // Verify batched execution matches sequential single-sequence execution
    const int batch_size = 3;
    const int seq_len = 1;
    const int kv_seq_len = 10;
    const int position_offset = 9;
    const int num_heads = 2;
    const int num_kv_heads = 2;
    const int head_dim = 64;
    const int d_model = num_heads * head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int num_blocks = head_dim / 32;

    // Generate batched data
    std::vector<float> Q_fp32(batch_size * seq_len * num_heads * head_dim);
    std::vector<float> K_fp32(batch_size * kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> V_fp32(batch_size * kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> Wo_fp32(d_model * num_heads * head_dim);

    generate_random_fp32(Q_fp32.data(), Q_fp32.size());
    generate_random_fp32(K_fp32.data(), K_fp32.size());
    generate_random_fp32(V_fp32.data(), V_fp32.size());
    generate_random_fp32(Wo_fp32.data(), Wo_fp32.size());

    std::vector<Q8_1Block> Q_q8(batch_size * seq_len * num_heads * num_blocks);
    std::vector<Q8_1Block> K_q8(batch_size * kv_seq_len * num_kv_heads * num_blocks);
    std::vector<Q8_1Block> V_q8(batch_size * kv_seq_len * num_kv_heads * num_blocks);

    quantize_to_q8_1(Q_fp32.data(), Q_q8.data(), Q_fp32.size());
    quantize_to_q8_1(K_fp32.data(), K_q8.data(), K_fp32.size());
    quantize_to_q8_1(V_fp32.data(), V_q8.data(), V_fp32.size());

    // Batched execution
    std::vector<float> output_batched(batch_size * d_model, 0.0f);

    FusedAttentionWoParams batch_params;
    batch_params.Q = Q_q8.data();
    batch_params.K = K_q8.data();
    batch_params.V = V_q8.data();
    batch_params.Wo = Wo_fp32.data();
    batch_params.wo_type = WoWeightType::FP32;
    batch_params.output = output_batched.data();
    batch_params.batch_size = batch_size;
    batch_params.seq_len = seq_len;
    batch_params.kv_seq_len = kv_seq_len;
    batch_params.num_heads = num_heads;
    batch_params.num_kv_heads = num_kv_heads;
    batch_params.head_dim = head_dim;
    batch_params.d_model = d_model;
    batch_params.scale = scale;
    batch_params.causal = true;
    batch_params.position_offset = position_offset;

    ASSERT_TRUE(FusedAttentionWoRef::execute(batch_params));

    // Sequential execution for each batch item
    const size_t q_stride = seq_len * num_heads * num_blocks;
    const size_t kv_stride = kv_seq_len * num_kv_heads * num_blocks;

    for (int b = 0; b < batch_size; ++b)
    {
        std::vector<float> output_single(d_model, 0.0f);

        FusedAttentionWoParams single_params;
        single_params.Q = Q_q8.data() + b * q_stride;
        single_params.K = K_q8.data() + b * kv_stride;
        single_params.V = V_q8.data() + b * kv_stride;
        single_params.Wo = Wo_fp32.data();
        single_params.wo_type = WoWeightType::FP32;
        single_params.output = output_single.data();
        single_params.batch_size = 1;
        single_params.seq_len = seq_len;
        single_params.kv_seq_len = kv_seq_len;
        single_params.num_heads = num_heads;
        single_params.num_kv_heads = num_kv_heads;
        single_params.head_dim = head_dim;
        single_params.d_model = d_model;
        single_params.scale = scale;
        single_params.causal = true;
        single_params.position_offset = position_offset;

        ASSERT_TRUE(FusedAttentionWoRef::execute(single_params));

        // Compare
        float cos_sim = cosine_similarity(
            output_batched.data() + b * d_model,
            output_single.data(),
            d_model);
        EXPECT_GT(cos_sim, 0.9999f)
            << "Batch " << b << " cosine similarity: " << cos_sim;
    }
}

// =============================================================================
// BATCHED DECODE - Variable KV lengths
// =============================================================================

TEST_F(FusedAttentionWoBatchTest, BatchedDecode_VariableKV)
{
    // Different sequences have different KV cache lengths
    const int batch_size = 4;
    const int seq_len = 1;
    const int max_kv_seq_len = 20;                       // Max for stride calculation
    const std::vector<int> kv_lens = {5, 10, 15, 20};    // Per-sequence KV lengths
    const std::vector<int> pos_offsets = {4, 9, 14, 19}; // Per-sequence positions
    const int num_heads = 2;
    const int num_kv_heads = 2;
    const int head_dim = 64;
    const int d_model = num_heads * head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int num_blocks = head_dim / 32;

    // Batched tensors (padded to max_kv_seq_len)
    std::vector<float> Q_fp32(batch_size * seq_len * num_heads * head_dim);
    std::vector<float> K_fp32(batch_size * max_kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> V_fp32(batch_size * max_kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> Wo_fp32(d_model * num_heads * head_dim);

    generate_random_fp32(Q_fp32.data(), Q_fp32.size());
    generate_random_fp32(K_fp32.data(), K_fp32.size());
    generate_random_fp32(V_fp32.data(), V_fp32.size());
    generate_random_fp32(Wo_fp32.data(), Wo_fp32.size());

    std::vector<Q8_1Block> Q_q8(batch_size * seq_len * num_heads * num_blocks);
    std::vector<Q8_1Block> K_q8(batch_size * max_kv_seq_len * num_kv_heads * num_blocks);
    std::vector<Q8_1Block> V_q8(batch_size * max_kv_seq_len * num_kv_heads * num_blocks);

    quantize_to_q8_1(Q_fp32.data(), Q_q8.data(), Q_fp32.size());
    quantize_to_q8_1(K_fp32.data(), K_q8.data(), K_fp32.size());
    quantize_to_q8_1(V_fp32.data(), V_q8.data(), V_fp32.size());

    std::vector<float> output_batched(batch_size * d_model, 0.0f);

    // Execute with variable KV lengths
    FusedAttentionWoParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.Wo = Wo_fp32.data();
    params.wo_type = WoWeightType::FP32;
    params.output = output_batched.data();
    params.batch_size = batch_size;
    params.seq_len = seq_len;
    params.kv_seq_len = max_kv_seq_len;           // Max for stride
    params.kv_seq_lens = kv_lens.data();          // Per-sequence lengths
    params.position_offsets = pos_offsets.data(); // Per-sequence offsets
    params.num_heads = num_heads;
    params.num_kv_heads = num_kv_heads;
    params.head_dim = head_dim;
    params.d_model = d_model;
    params.scale = scale;
    params.causal = true;

    ASSERT_TRUE(FusedAttentionWoRef::execute(params));

    // Verify each batch has non-zero output
    for (int b = 0; b < batch_size; ++b)
    {
        float sum = 0.0f;
        for (int i = 0; i < d_model; ++i)
        {
            sum += std::abs(output_batched[b * d_model + i]);
        }
        EXPECT_GT(sum, 0.0f) << "Batch " << b << " (kv_len=" << kv_lens[b] << ")";
    }
}

TEST_F(FusedAttentionWoBatchTest, BatchedDecode_VariableKV_MatchesSingle)
{
    // Verify variable-KV batched matches sequential single-sequence
    const int batch_size = 3;
    const int seq_len = 1;
    const int max_kv_seq_len = 16;
    const std::vector<int> kv_lens = {4, 8, 16};
    const std::vector<int> pos_offsets = {3, 7, 15};
    const int num_heads = 2;
    const int num_kv_heads = 2;
    const int head_dim = 64;
    const int d_model = num_heads * head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int num_blocks = head_dim / 32;

    std::vector<float> Q_fp32(batch_size * seq_len * num_heads * head_dim);
    std::vector<float> K_fp32(batch_size * max_kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> V_fp32(batch_size * max_kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> Wo_fp32(d_model * num_heads * head_dim);

    generate_random_fp32(Q_fp32.data(), Q_fp32.size());
    generate_random_fp32(K_fp32.data(), K_fp32.size());
    generate_random_fp32(V_fp32.data(), V_fp32.size());
    generate_random_fp32(Wo_fp32.data(), Wo_fp32.size());

    std::vector<Q8_1Block> Q_q8(batch_size * seq_len * num_heads * num_blocks);
    std::vector<Q8_1Block> K_q8(batch_size * max_kv_seq_len * num_kv_heads * num_blocks);
    std::vector<Q8_1Block> V_q8(batch_size * max_kv_seq_len * num_kv_heads * num_blocks);

    quantize_to_q8_1(Q_fp32.data(), Q_q8.data(), Q_fp32.size());
    quantize_to_q8_1(K_fp32.data(), K_q8.data(), K_fp32.size());
    quantize_to_q8_1(V_fp32.data(), V_q8.data(), V_fp32.size());

    // Batched with variable KV
    std::vector<float> output_batched(batch_size * d_model, 0.0f);

    FusedAttentionWoParams batch_params;
    batch_params.Q = Q_q8.data();
    batch_params.K = K_q8.data();
    batch_params.V = V_q8.data();
    batch_params.Wo = Wo_fp32.data();
    batch_params.wo_type = WoWeightType::FP32;
    batch_params.output = output_batched.data();
    batch_params.batch_size = batch_size;
    batch_params.seq_len = seq_len;
    batch_params.kv_seq_len = max_kv_seq_len;
    batch_params.kv_seq_lens = kv_lens.data();
    batch_params.position_offsets = pos_offsets.data();
    batch_params.num_heads = num_heads;
    batch_params.num_kv_heads = num_kv_heads;
    batch_params.head_dim = head_dim;
    batch_params.d_model = d_model;
    batch_params.scale = scale;
    batch_params.causal = true;

    ASSERT_TRUE(FusedAttentionWoRef::execute(batch_params));

    // Compare against single-sequence execution
    const size_t q_stride = seq_len * num_heads * num_blocks;
    const size_t kv_stride = max_kv_seq_len * num_kv_heads * num_blocks;

    for (int b = 0; b < batch_size; ++b)
    {
        std::vector<float> output_single(d_model, 0.0f);

        FusedAttentionWoParams single_params;
        single_params.Q = Q_q8.data() + b * q_stride;
        single_params.K = K_q8.data() + b * kv_stride;
        single_params.V = V_q8.data() + b * kv_stride;
        single_params.Wo = Wo_fp32.data();
        single_params.wo_type = WoWeightType::FP32;
        single_params.output = output_single.data();
        single_params.batch_size = 1;
        single_params.seq_len = seq_len;
        single_params.kv_seq_len = kv_lens[b]; // Actual KV length for this sequence
        single_params.num_heads = num_heads;
        single_params.num_kv_heads = num_kv_heads;
        single_params.head_dim = head_dim;
        single_params.d_model = d_model;
        single_params.scale = scale;
        single_params.causal = true;
        single_params.position_offset = pos_offsets[b];

        ASSERT_TRUE(FusedAttentionWoRef::execute(single_params));

        float cos_sim = cosine_similarity(
            output_batched.data() + b * d_model,
            output_single.data(),
            d_model);
        EXPECT_GT(cos_sim, 0.9999f)
            << "Batch " << b << " (kv_len=" << kv_lens[b] << ") cosine: " << cos_sim;
    }
}

// =============================================================================
// EDGE CASES
// =============================================================================

TEST_F(FusedAttentionWoBatchTest, EdgeCase_SingleTokenKV)
{
    // KV cache with only 1 entry
    const int seq_len = 1;
    const int kv_seq_len = 1;
    const int num_heads = 2;
    const int num_kv_heads = 2;
    const int head_dim = 64;
    const int d_model = num_heads * head_dim;
    const int num_blocks = head_dim / 32;

    std::vector<float> Q_fp32(seq_len * num_heads * head_dim);
    std::vector<float> K_fp32(kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> V_fp32(kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> Wo_fp32(d_model * num_heads * head_dim);

    generate_random_fp32(Q_fp32.data(), Q_fp32.size());
    generate_random_fp32(K_fp32.data(), K_fp32.size());
    generate_random_fp32(V_fp32.data(), V_fp32.size());
    generate_random_fp32(Wo_fp32.data(), Wo_fp32.size());

    std::vector<Q8_1Block> Q_q8(seq_len * num_heads * num_blocks);
    std::vector<Q8_1Block> K_q8(kv_seq_len * num_kv_heads * num_blocks);
    std::vector<Q8_1Block> V_q8(kv_seq_len * num_kv_heads * num_blocks);

    quantize_to_q8_1(Q_fp32.data(), Q_q8.data(), Q_fp32.size());
    quantize_to_q8_1(K_fp32.data(), K_q8.data(), K_fp32.size());
    quantize_to_q8_1(V_fp32.data(), V_q8.data(), V_fp32.size());

    std::vector<float> output(d_model, 0.0f);

    FusedAttentionWoParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.Wo = Wo_fp32.data();
    params.wo_type = WoWeightType::FP32;
    params.output = output.data();
    params.batch_size = 1;
    params.seq_len = seq_len;
    params.kv_seq_len = kv_seq_len;
    params.num_heads = num_heads;
    params.num_kv_heads = num_kv_heads;
    params.head_dim = head_dim;
    params.d_model = d_model;
    params.scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    params.causal = true;
    params.position_offset = 0;

    ASSERT_TRUE(FusedAttentionWoRef::execute(params));

    // With single KV position, softmax weight = 1.0
    // Output should be non-zero
    float sum = 0.0f;
    for (int i = 0; i < d_model; ++i)
    {
        sum += std::abs(output[i]);
    }
    EXPECT_GT(sum, 0.0f);
}

TEST_F(FusedAttentionWoBatchTest, EdgeCase_GQA_ManyToOne)
{
    // GQA: 8 query heads, 2 KV heads (4:1 ratio)
    const int seq_len = 4;
    const int kv_seq_len = 4;
    const int num_heads = 8;
    const int num_kv_heads = 2;
    const int head_dim = 64;
    const int d_model = num_heads * head_dim;
    const int num_blocks = head_dim / 32;

    std::vector<float> Q_fp32(seq_len * num_heads * head_dim);
    std::vector<float> K_fp32(kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> V_fp32(kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> Wo_fp32(d_model * num_heads * head_dim);

    generate_random_fp32(Q_fp32.data(), Q_fp32.size());
    generate_random_fp32(K_fp32.data(), K_fp32.size());
    generate_random_fp32(V_fp32.data(), V_fp32.size());
    generate_random_fp32(Wo_fp32.data(), Wo_fp32.size());

    std::vector<Q8_1Block> Q_q8(seq_len * num_heads * num_blocks);
    std::vector<Q8_1Block> K_q8(kv_seq_len * num_kv_heads * num_blocks);
    std::vector<Q8_1Block> V_q8(kv_seq_len * num_kv_heads * num_blocks);

    quantize_to_q8_1(Q_fp32.data(), Q_q8.data(), Q_fp32.size());
    quantize_to_q8_1(K_fp32.data(), K_q8.data(), K_fp32.size());
    quantize_to_q8_1(V_fp32.data(), V_q8.data(), V_fp32.size());

    std::vector<float> output(seq_len * d_model, 0.0f);

    FusedAttentionWoParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.Wo = Wo_fp32.data();
    params.wo_type = WoWeightType::FP32;
    params.output = output.data();
    params.batch_size = 1;
    params.seq_len = seq_len;
    params.kv_seq_len = kv_seq_len;
    params.num_heads = num_heads;
    params.num_kv_heads = num_kv_heads;
    params.head_dim = head_dim;
    params.d_model = d_model;
    params.scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    params.causal = true;
    params.position_offset = 0;

    ASSERT_TRUE(FusedAttentionWoRef::execute(params));

    float sum = 0.0f;
    for (int i = 0; i < seq_len * d_model; ++i)
    {
        sum += std::abs(output[i]);
    }
    EXPECT_GT(sum, 0.0f);
}

// =============================================================================
// VALIDATION TESTS
// =============================================================================

TEST_F(FusedAttentionWoBatchTest, Validation_InvalidBatchSize)
{
    FusedAttentionWoParams params;
    params.Q = reinterpret_cast<const Q8_1Block *>(0x1000);
    params.K = reinterpret_cast<const Q8_1Block *>(0x1000);
    params.V = reinterpret_cast<const Q8_1Block *>(0x1000);
    params.Wo = reinterpret_cast<const void *>(0x1000);
    params.output = reinterpret_cast<float *>(0x1000);
    params.batch_size = 0; // Invalid
    params.seq_len = 1;
    params.kv_seq_len = 1;
    params.num_heads = 2;
    params.num_kv_heads = 2;
    params.head_dim = 64;
    params.d_model = 128;

    EXPECT_FALSE(FusedAttentionWoRef::validate_params(params));
}

TEST_F(FusedAttentionWoBatchTest, Validation_KVLenExceedsMax)
{
    std::vector<Q8_1Block> Q(2), K(4), V(4);
    std::vector<float> output(128), Wo(128 * 64);
    std::vector<int> kv_lens = {5}; // Exceeds max of 2

    FusedAttentionWoParams params;
    params.Q = Q.data();
    params.K = K.data();
    params.V = V.data();
    params.Wo = Wo.data();
    params.wo_type = WoWeightType::FP32;
    params.output = output.data();
    params.batch_size = 1;
    params.seq_len = 1;
    params.kv_seq_len = 2;               // Max is 2
    params.kv_seq_lens = kv_lens.data(); // But this says 5
    params.num_heads = 2;
    params.num_kv_heads = 2;
    params.head_dim = 64;
    params.d_model = 128;

    EXPECT_FALSE(FusedAttentionWoRef::validate_params(params));
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
