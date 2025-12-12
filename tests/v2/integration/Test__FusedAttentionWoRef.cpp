/**
 * @file Test__FusedAttentionWoRef.cpp
 * @brief Integration tests for the fused attention + Wo projection reference kernel.
 *
 * Tests the composed kernel (using microkernels μK1-μK5) against:
 * 1. Separate attention + Wo GEMM computation
 * 2. Known numerical examples
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
#include "../../../src/v2/kernels/cpu/microkernels/q8_1/OnlineSoftmax.h"
#include "../../../src/v2/tensors/FP16Utils.h"

using namespace llaminar::v2::kernels;
using namespace llaminar::v2::kernels::microkernels;

/**
 * @brief Helper to quantize FP32 vector to Q8_1 blocks
 */
void quantize_to_q8_1(const float *input, Q8_1Block *output, int numel)
{
    const int num_blocks = numel / 32;

    for (int b = 0; b < num_blocks; ++b)
    {
        const float *block_data = input + b * 32;

        // Find absmax in block
        float absmax = 0.0f;
        for (int i = 0; i < 32; ++i)
        {
            absmax = std::max(absmax, std::abs(block_data[i]));
        }

        // Compute scale
        float scale = absmax > 0 ? absmax / 127.0f : 1.0f;
        float inv_scale = absmax > 0 ? 127.0f / absmax : 0.0f;

        // Store scale as FP16
        output[b].d = llaminar2::fp32_to_fp16(scale);

        // Quantize and compute sum_qs
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

/**
 * @brief Helper to dequantize Q8_1 blocks to FP32
 */
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

/**
 * @brief Reference attention computation (separate Q*K*V + Wo)
 *
 * This is the "unfused" baseline we compare against.
 */
void compute_attention_reference_fp32(
    const float *Q_fp32,  // [seq_len, num_heads, head_dim]
    const float *K_fp32,  // [kv_seq_len, num_kv_heads, head_dim]
    const float *V_fp32,  // [kv_seq_len, num_kv_heads, head_dim]
    const float *Wo_fp32, // [d_model, num_heads * head_dim]
    float *output,        // [seq_len, d_model]
    int seq_len,
    int kv_seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int d_model,
    float scale,
    bool causal,
    int position_offset)
{
    const int kv_head_ratio = num_heads / num_kv_heads;

    // Zero output
    std::fill(output, output + seq_len * d_model, 0.0f);

    // Process each query position
    for (int m = 0; m < seq_len; ++m)
    {
        // Process each head
        for (int h = 0; h < num_heads; ++h)
        {
            int kv_h = h / kv_head_ratio;

            // Get Q vector for this position and head
            const float *Q_row = Q_fp32 + (m * num_heads + h) * head_dim;

            // Compute attention scores
            int max_kv = causal ? std::min(m + position_offset + 1, kv_seq_len) : kv_seq_len;

            // Allocate temporary arrays
            std::vector<float> scores(max_kv);
            std::vector<float> weights(max_kv);
            std::vector<float> context(head_dim, 0.0f);

            // Compute Q*K^T scores
            float max_score = -std::numeric_limits<float>::infinity();
            for (int n = 0; n < max_kv; ++n)
            {
                const float *K_row = K_fp32 + (n * num_kv_heads + kv_h) * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d)
                {
                    dot += Q_row[d] * K_row[d];
                }
                scores[n] = dot * scale;
                max_score = std::max(max_score, scores[n]);
            }

            // Softmax
            float sum_exp = 0.0f;
            for (int n = 0; n < max_kv; ++n)
            {
                weights[n] = std::exp(scores[n] - max_score);
                sum_exp += weights[n];
            }
            for (int n = 0; n < max_kv; ++n)
            {
                weights[n] /= sum_exp;
            }

            // Compute context = softmax(Q*K^T) * V
            for (int n = 0; n < max_kv; ++n)
            {
                const float *V_row = V_fp32 + (n * num_kv_heads + kv_h) * head_dim;
                for (int d = 0; d < head_dim; ++d)
                {
                    context[d] += weights[n] * V_row[d];
                }
            }

            // Project through Wo
            // Wo layout: [d_model, num_heads * head_dim]
            // For head h: Wo[:, h*head_dim:(h+1)*head_dim]
            float *out_row = output + m * d_model;
            for (int o = 0; o < d_model; ++o)
            {
                float dot = 0.0f;
                const float *wo_row = Wo_fp32 + o * (num_heads * head_dim) + h * head_dim;
                for (int d = 0; d < head_dim; ++d)
                {
                    dot += context[d] * wo_row[d];
                }
                out_row[o] += dot; // Accumulate from all heads
            }
        }
    }
}

/**
 * @brief Compute cosine similarity between two vectors
 */
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

/**
 * @brief Test fixture for FusedAttentionWoRef tests
 */
class FusedAttentionWoRefTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Seed random generator for reproducibility
        rng_.seed(42);
    }

    /**
     * @brief Generate random FP32 values in range [-1, 1]
     */
    void generate_random_fp32(float *data, int n)
    {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int i = 0; i < n; ++i)
        {
            data[i] = dist(rng_);
        }
    }

    std::mt19937 rng_;
};

// =============================================================================
// Basic Functionality Tests
// =============================================================================

TEST_F(FusedAttentionWoRefTest, ValidateParams_NullQ_ReturnsFalse)
{
    FusedAttentionWoParams params;
    params.Q = nullptr;
    params.K = reinterpret_cast<const Q8_1Block *>(0x1000); // Non-null
    params.V = reinterpret_cast<const Q8_1Block *>(0x1000);
    params.Wo = reinterpret_cast<const void *>(0x1000);
    params.output = reinterpret_cast<float *>(0x1000);
    params.seq_len = 1;
    params.kv_seq_len = 1;
    params.num_heads = 1;
    params.num_kv_heads = 1;
    params.head_dim = 32;
    params.d_model = 32;

    EXPECT_FALSE(FusedAttentionWoRef::validate_params(params));
}

TEST_F(FusedAttentionWoRefTest, ValidateParams_ValidParams_ReturnsTrue)
{
    // Create minimal valid tensors
    std::vector<Q8_1Block> Q_blocks(1), K_blocks(1), V_blocks(1), Wo_blocks(1);
    std::vector<float> output(32);

    FusedAttentionWoParams params;
    params.Q = Q_blocks.data();
    params.K = K_blocks.data();
    params.V = V_blocks.data();
    params.Wo = Wo_blocks.data();
    params.wo_type = WoWeightType::Q8_1;
    params.output = output.data();
    params.seq_len = 1;
    params.kv_seq_len = 1;
    params.num_heads = 1;
    params.num_kv_heads = 1;
    params.head_dim = 32;
    params.d_model = 32;
    params.scale = 1.0f / std::sqrt(32.0f);

    EXPECT_TRUE(FusedAttentionWoRef::validate_params(params));
}

// =============================================================================
// Single Position, Single Head Tests
// =============================================================================

TEST_F(FusedAttentionWoRefTest, SinglePositionSingleHead_MatchesReference)
{
    // Test configuration: 1 position, 1 head, head_dim=64
    const int seq_len = 1;
    const int kv_seq_len = 1;
    const int num_heads = 1;
    const int num_kv_heads = 1;
    const int head_dim = 64;
    const int d_model = 64; // Same as head_dim for this test
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const int num_blocks = head_dim / 32;

    // Generate random FP32 data
    std::vector<float> Q_fp32(seq_len * num_heads * head_dim);
    std::vector<float> K_fp32(kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> V_fp32(kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> Wo_fp32(d_model * num_heads * head_dim);

    generate_random_fp32(Q_fp32.data(), Q_fp32.size());
    generate_random_fp32(K_fp32.data(), K_fp32.size());
    generate_random_fp32(V_fp32.data(), V_fp32.size());
    generate_random_fp32(Wo_fp32.data(), Wo_fp32.size());

    // Quantize to Q8_1
    std::vector<Q8_1Block> Q_q8(seq_len * num_heads * num_blocks);
    std::vector<Q8_1Block> K_q8(kv_seq_len * num_kv_heads * num_blocks);
    std::vector<Q8_1Block> V_q8(kv_seq_len * num_kv_heads * num_blocks);

    quantize_to_q8_1(Q_fp32.data(), Q_q8.data(), Q_fp32.size());
    quantize_to_q8_1(K_fp32.data(), K_q8.data(), K_fp32.size());
    quantize_to_q8_1(V_fp32.data(), V_q8.data(), V_fp32.size());

    // Output buffers
    std::vector<float> output_fused(seq_len * d_model, 0.0f);
    std::vector<float> output_ref(seq_len * d_model, 0.0f);

    // Dequantize for reference computation
    std::vector<float> Q_deq(Q_fp32.size()), K_deq(K_fp32.size()), V_deq(V_fp32.size());
    dequantize_from_q8_1(Q_q8.data(), Q_deq.data(), Q_fp32.size());
    dequantize_from_q8_1(K_q8.data(), K_deq.data(), K_fp32.size());
    dequantize_from_q8_1(V_q8.data(), V_deq.data(), V_fp32.size());

    // Compute reference (FP32 path with dequantized inputs)
    compute_attention_reference_fp32(
        Q_deq.data(), K_deq.data(), V_deq.data(), Wo_fp32.data(),
        output_ref.data(),
        seq_len, kv_seq_len, num_heads, num_kv_heads, head_dim, d_model,
        scale, false, 0);

    // Compute fused version
    FusedAttentionWoParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.Wo = Wo_fp32.data();
    params.wo_type = WoWeightType::FP32;
    params.output = output_fused.data();
    params.seq_len = seq_len;
    params.kv_seq_len = kv_seq_len;
    params.num_heads = num_heads;
    params.num_kv_heads = num_kv_heads;
    params.head_dim = head_dim;
    params.d_model = d_model;
    params.scale = scale;
    params.causal = false;
    params.position_offset = 0;

    ASSERT_TRUE(FusedAttentionWoRef::execute(params));

    // Compare outputs
    float cos_sim = cosine_similarity(output_fused.data(), output_ref.data(), d_model);
    EXPECT_GT(cos_sim, 0.99f) << "Cosine similarity: " << cos_sim;

    // Check element-wise error
    float max_diff = 0.0f;
    for (int i = 0; i < d_model; ++i)
    {
        max_diff = std::max(max_diff, std::abs(output_fused[i] - output_ref[i]));
    }
    EXPECT_LT(max_diff, 0.1f) << "Max element difference: " << max_diff;
}

// =============================================================================
// Multi-Position Causal Attention Tests
// =============================================================================

TEST_F(FusedAttentionWoRefTest, MultiPositionCausal_MatchesReference)
{
    // Test configuration: 4 positions, 2 heads, head_dim=64
    const int seq_len = 4;
    const int kv_seq_len = 4;
    const int num_heads = 2;
    const int num_kv_heads = 2;
    const int head_dim = 64;
    const int d_model = num_heads * head_dim; // 128
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const int num_blocks = head_dim / 32;

    // Generate random FP32 data
    std::vector<float> Q_fp32(seq_len * num_heads * head_dim);
    std::vector<float> K_fp32(kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> V_fp32(kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> Wo_fp32(d_model * num_heads * head_dim);

    generate_random_fp32(Q_fp32.data(), Q_fp32.size());
    generate_random_fp32(K_fp32.data(), K_fp32.size());
    generate_random_fp32(V_fp32.data(), V_fp32.size());
    generate_random_fp32(Wo_fp32.data(), Wo_fp32.size());

    // Quantize to Q8_1
    std::vector<Q8_1Block> Q_q8(seq_len * num_heads * num_blocks);
    std::vector<Q8_1Block> K_q8(kv_seq_len * num_kv_heads * num_blocks);
    std::vector<Q8_1Block> V_q8(kv_seq_len * num_kv_heads * num_blocks);

    quantize_to_q8_1(Q_fp32.data(), Q_q8.data(), Q_fp32.size());
    quantize_to_q8_1(K_fp32.data(), K_q8.data(), K_fp32.size());
    quantize_to_q8_1(V_fp32.data(), V_q8.data(), V_fp32.size());

    // Output buffers
    std::vector<float> output_fused(seq_len * d_model, 0.0f);
    std::vector<float> output_ref(seq_len * d_model, 0.0f);

    // Dequantize for reference
    std::vector<float> Q_deq(Q_fp32.size()), K_deq(K_fp32.size()), V_deq(V_fp32.size());
    dequantize_from_q8_1(Q_q8.data(), Q_deq.data(), Q_fp32.size());
    dequantize_from_q8_1(K_q8.data(), K_deq.data(), K_fp32.size());
    dequantize_from_q8_1(V_q8.data(), V_deq.data(), V_fp32.size());

    // Compute reference with causal mask
    compute_attention_reference_fp32(
        Q_deq.data(), K_deq.data(), V_deq.data(), Wo_fp32.data(),
        output_ref.data(),
        seq_len, kv_seq_len, num_heads, num_kv_heads, head_dim, d_model,
        scale, true, 0 // causal=true
    );

    // Compute fused version
    FusedAttentionWoParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.Wo = Wo_fp32.data();
    params.wo_type = WoWeightType::FP32;
    params.output = output_fused.data();
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

    // Check each position
    for (int m = 0; m < seq_len; ++m)
    {
        float cos_sim = cosine_similarity(
            output_fused.data() + m * d_model,
            output_ref.data() + m * d_model,
            d_model);
        EXPECT_GT(cos_sim, 0.98f) << "Position " << m << " cosine similarity: " << cos_sim;
    }
}

// =============================================================================
// GQA (Grouped Query Attention) Tests
// =============================================================================

TEST_F(FusedAttentionWoRefTest, GQA_MultipleQueryHeadsPerKV)
{
    // Test GQA: 4 query heads sharing 2 KV heads
    const int seq_len = 2;
    const int kv_seq_len = 2;
    const int num_heads = 4;
    const int num_kv_heads = 2; // GQA ratio = 2
    const int head_dim = 64;
    const int d_model = num_heads * head_dim; // 256
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const int num_blocks = head_dim / 32;

    // Generate random data
    std::vector<float> Q_fp32(seq_len * num_heads * head_dim);
    std::vector<float> K_fp32(kv_seq_len * num_kv_heads * head_dim); // Note: fewer KV heads
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

    // Outputs
    std::vector<float> output_fused(seq_len * d_model, 0.0f);
    std::vector<float> output_ref(seq_len * d_model, 0.0f);

    // Dequantize
    std::vector<float> Q_deq(Q_fp32.size()), K_deq(K_fp32.size()), V_deq(V_fp32.size());
    dequantize_from_q8_1(Q_q8.data(), Q_deq.data(), Q_fp32.size());
    dequantize_from_q8_1(K_q8.data(), K_deq.data(), K_fp32.size());
    dequantize_from_q8_1(V_q8.data(), V_deq.data(), V_fp32.size());

    // Reference
    compute_attention_reference_fp32(
        Q_deq.data(), K_deq.data(), V_deq.data(), Wo_fp32.data(),
        output_ref.data(),
        seq_len, kv_seq_len, num_heads, num_kv_heads, head_dim, d_model,
        scale, false, 0);

    // Fused
    FusedAttentionWoParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.Wo = Wo_fp32.data();
    params.wo_type = WoWeightType::FP32;
    params.output = output_fused.data();
    params.seq_len = seq_len;
    params.kv_seq_len = kv_seq_len;
    params.num_heads = num_heads;
    params.num_kv_heads = num_kv_heads;
    params.head_dim = head_dim;
    params.d_model = d_model;
    params.scale = scale;
    params.causal = false;
    params.position_offset = 0;

    ASSERT_TRUE(FusedAttentionWoRef::execute(params));

    // Compare
    float cos_sim = cosine_similarity(output_fused.data(), output_ref.data(), seq_len * d_model);
    EXPECT_GT(cos_sim, 0.98f) << "GQA cosine similarity: " << cos_sim;
}

// =============================================================================
// Q8_1 Wo Weight Tests
// =============================================================================

TEST_F(FusedAttentionWoRefTest, Q8_1_Wo_Weights)
{
    // Test with Q8_1 quantized Wo weights
    const int seq_len = 1;
    const int kv_seq_len = 2;
    const int num_heads = 2;
    const int num_kv_heads = 2;
    const int head_dim = 64;
    const int d_model = num_heads * head_dim; // 128
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const int num_blocks = head_dim / 32;
    const int wo_row_blocks = (num_heads * head_dim) / 32;

    // Generate random data
    std::vector<float> Q_fp32(seq_len * num_heads * head_dim);
    std::vector<float> K_fp32(kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> V_fp32(kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> Wo_fp32(d_model * num_heads * head_dim);

    generate_random_fp32(Q_fp32.data(), Q_fp32.size());
    generate_random_fp32(K_fp32.data(), K_fp32.size());
    generate_random_fp32(V_fp32.data(), V_fp32.size());
    generate_random_fp32(Wo_fp32.data(), Wo_fp32.size());

    // Quantize Q/K/V
    std::vector<Q8_1Block> Q_q8(seq_len * num_heads * num_blocks);
    std::vector<Q8_1Block> K_q8(kv_seq_len * num_kv_heads * num_blocks);
    std::vector<Q8_1Block> V_q8(kv_seq_len * num_kv_heads * num_blocks);

    quantize_to_q8_1(Q_fp32.data(), Q_q8.data(), Q_fp32.size());
    quantize_to_q8_1(K_fp32.data(), K_q8.data(), K_fp32.size());
    quantize_to_q8_1(V_fp32.data(), V_q8.data(), V_fp32.size());

    // Quantize Wo
    std::vector<Q8_1Block> Wo_q8(d_model * wo_row_blocks);
    for (int row = 0; row < d_model; ++row)
    {
        quantize_to_q8_1(
            Wo_fp32.data() + row * num_heads * head_dim,
            Wo_q8.data() + row * wo_row_blocks,
            num_heads * head_dim);
    }

    // Output buffers
    std::vector<float> output_q8_wo(seq_len * d_model, 0.0f);
    std::vector<float> output_fp32_wo(seq_len * d_model, 0.0f);

    // Run with FP32 Wo
    FusedAttentionWoParams params_fp32;
    params_fp32.Q = Q_q8.data();
    params_fp32.K = K_q8.data();
    params_fp32.V = V_q8.data();
    params_fp32.Wo = Wo_fp32.data();
    params_fp32.wo_type = WoWeightType::FP32;
    params_fp32.output = output_fp32_wo.data();
    params_fp32.seq_len = seq_len;
    params_fp32.kv_seq_len = kv_seq_len;
    params_fp32.num_heads = num_heads;
    params_fp32.num_kv_heads = num_kv_heads;
    params_fp32.head_dim = head_dim;
    params_fp32.d_model = d_model;
    params_fp32.scale = scale;
    params_fp32.causal = false;

    ASSERT_TRUE(FusedAttentionWoRef::execute(params_fp32));

    // Run with Q8_1 Wo
    FusedAttentionWoParams params_q8;
    params_q8.Q = Q_q8.data();
    params_q8.K = K_q8.data();
    params_q8.V = V_q8.data();
    params_q8.Wo = Wo_q8.data();
    params_q8.wo_type = WoWeightType::Q8_1;
    params_q8.output = output_q8_wo.data();
    params_q8.seq_len = seq_len;
    params_q8.kv_seq_len = kv_seq_len;
    params_q8.num_heads = num_heads;
    params_q8.num_kv_heads = num_kv_heads;
    params_q8.head_dim = head_dim;
    params_q8.d_model = d_model;
    params_q8.scale = scale;
    params_q8.causal = false;

    ASSERT_TRUE(FusedAttentionWoRef::execute(params_q8));

    // Q8_1 Wo should be close to FP32 Wo (allowing for quantization noise)
    float cos_sim = cosine_similarity(output_q8_wo.data(), output_fp32_wo.data(), d_model);
    EXPECT_GT(cos_sim, 0.95f) << "Q8_1 vs FP32 Wo cosine similarity: " << cos_sim;
}

// =============================================================================
// Decode Mode (KV Cache) Tests
// =============================================================================

TEST_F(FusedAttentionWoRefTest, DecodeMode_PositionOffset)
{
    // Simulate decode mode: single new token attending to KV cache
    const int seq_len = 1;    // Single query (new token)
    const int kv_seq_len = 5; // Previous tokens in cache
    const int num_heads = 2;
    const int num_kv_heads = 2;
    const int head_dim = 64;
    const int d_model = num_heads * head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const int num_blocks = head_dim / 32;

    // Generate data
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

    // Outputs
    std::vector<float> output_fused(seq_len * d_model, 0.0f);
    std::vector<float> output_ref(seq_len * d_model, 0.0f);

    // Dequantize
    std::vector<float> Q_deq(Q_fp32.size()), K_deq(K_fp32.size()), V_deq(V_fp32.size());
    dequantize_from_q8_1(Q_q8.data(), Q_deq.data(), Q_fp32.size());
    dequantize_from_q8_1(K_q8.data(), K_deq.data(), K_fp32.size());
    dequantize_from_q8_1(V_q8.data(), V_deq.data(), V_fp32.size());

    // Reference with position offset (decode at position 4, so can attend to 0-4)
    compute_attention_reference_fp32(
        Q_deq.data(), K_deq.data(), V_deq.data(), Wo_fp32.data(),
        output_ref.data(),
        seq_len, kv_seq_len, num_heads, num_kv_heads, head_dim, d_model,
        scale, true, 4 // position_offset = 4
    );

    // Fused
    FusedAttentionWoParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.Wo = Wo_fp32.data();
    params.wo_type = WoWeightType::FP32;
    params.output = output_fused.data();
    params.seq_len = seq_len;
    params.kv_seq_len = kv_seq_len;
    params.num_heads = num_heads;
    params.num_kv_heads = num_kv_heads;
    params.head_dim = head_dim;
    params.d_model = d_model;
    params.scale = scale;
    params.causal = true;
    params.position_offset = 4; // Decode position

    ASSERT_TRUE(FusedAttentionWoRef::execute(params));

    // Compare
    float cos_sim = cosine_similarity(output_fused.data(), output_ref.data(), d_model);
    EXPECT_GT(cos_sim, 0.98f) << "Decode mode cosine similarity: " << cos_sim;
}

// =============================================================================
// Single Head Execution Test
// =============================================================================

TEST_F(FusedAttentionWoRefTest, ExecuteSingleHead)
{
    // Test execute_single_head interface
    const int seq_len = 1;
    const int kv_seq_len = 2;
    const int num_heads = 4;
    const int num_kv_heads = 2;
    const int head_dim = 64;
    const int d_model = num_heads * head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const int num_blocks = head_dim / 32;

    // Generate data
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

    // Full execution output
    std::vector<float> output_full(seq_len * d_model, 0.0f);

    FusedAttentionWoParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.Wo = Wo_fp32.data();
    params.wo_type = WoWeightType::FP32;
    params.output = output_full.data();
    params.seq_len = seq_len;
    params.kv_seq_len = kv_seq_len;
    params.num_heads = num_heads;
    params.num_kv_heads = num_kv_heads;
    params.head_dim = head_dim;
    params.d_model = d_model;
    params.scale = scale;
    params.causal = false;

    ASSERT_TRUE(FusedAttentionWoRef::execute(params));

    // Single head execution output (accumulate all heads manually)
    std::vector<float> output_manual(d_model, 0.0f);
    std::vector<float> context_buffer(head_dim);

    for (int h = 0; h < num_heads; ++h)
    {
        FusedAttentionWoRef::execute_single_head(
            params, 0, h, context_buffer.data(), output_manual.data());
    }

    // Should match
    float cos_sim = cosine_similarity(output_full.data(), output_manual.data(), d_model);
    EXPECT_GT(cos_sim, 0.999f) << "Single head vs full execution similarity: " << cos_sim;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
