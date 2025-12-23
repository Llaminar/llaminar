/**
 * @file Test__FusedAttentionWoTiled.cpp
 * @brief Unit tests for the cache-blocked tiled attention + Wo projection.
 *
 * Tests validate:
 * 1. Tile size computation based on cache sizes
 * 2. Numerical parity with reference implementation (FusedAttentionWoRef)
 * 3. Correctness across various sequence lengths and configurations
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

#include "../../../../src/v2/kernels/cpu/attention/q8_1/FusedAttentionWoTiled.h"
#include "../../../../src/v2/kernels/cpu/attention/q8_1/FusedAttentionWoRef.h"
#include "../../../../src/v2/kernels/cpu/attention/q8_1/microkernels/Q8DotProduct.h"
#include "../../../../src/v2/tensors/FP16Utils.h"
#include "../../../../src/v2/utils/CPUFeatures.h"

using namespace llaminar::v2::kernels;
using namespace llaminar::v2::kernels::microkernels;
using namespace llaminar::v2;

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
 * @brief Test fixture for FusedAttentionWoTiled tests
 */
class FusedAttentionWoTiledTest : public ::testing::Test
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
// Tile Configuration Tests
// =============================================================================

TEST_F(FusedAttentionWoTiledTest, ComputeTileSizes_DetectsCaches)
{
    // Test that cache detection works and produces reasonable tile sizes
    // compute_tile_config(head_dim, d_model)
    const int head_dim = 64;
    const int d_model = 896; // Qwen2 0.5B d_model

    TileConfig config = llaminar::v2::kernels::compute_tile_config(head_dim, d_model);

    // Verify L2/L3 sizes are detected (should be > 0 on any real CPU)
    EXPECT_GT(config.l2_size, 0u) << "L2 cache not detected";
    EXPECT_GT(config.l3_size, 0u) << "L3 cache not detected";

    // Verify tile sizes are reasonable
    EXPECT_GT(config.kv_tile, 0) << "KV tile size must be positive";
    EXPECT_GT(config.q_tile, 0) << "Q tile size must be positive";

    // KV tile should be at least 32 and fit in cache
    EXPECT_GE(config.kv_tile, 32) << "KV tile should be at least 32";
    EXPECT_LE(config.kv_tile, 4096) << "KV tile suspiciously large";

    // Log detected values for debugging
    std::cout << "Detected cache sizes: L2=" << config.l2_size
              << " L3=" << config.l3_size << std::endl;
    std::cout << "Computed tile sizes: kv_tile=" << config.kv_tile
              << " q_tile=" << config.q_tile << std::endl;
}

TEST_F(FusedAttentionWoTiledTest, ComputeTileSizes_Qwen2_0_5B_Config)
{
    // Qwen 2.5 0.5B: 14 heads, 2 KV heads, head_dim=64, d_model=896
    const int head_dim = 64;
    const int d_model = 896;

    TileConfig config = llaminar::v2::kernels::compute_tile_config(head_dim, d_model);

    // KV tile should be reasonable
    EXPECT_GE(config.kv_tile, 32) << "KV tile should be at least 32 for Qwen2 0.5B";

    std::cout << "Qwen2 0.5B tile config: kv_tile=" << config.kv_tile
              << " q_tile=" << config.q_tile << std::endl;
}

TEST_F(FusedAttentionWoTiledTest, ComputeTileSizes_Qwen2_7B_Config)
{
    // Qwen 2.5 7B: 28 heads, 4 KV heads, head_dim=128, d_model=3584
    const int head_dim = 128;
    const int d_model = 3584;

    TileConfig config = llaminar::v2::kernels::compute_tile_config(head_dim, d_model);

    // Tile sizes should still be reasonable for larger model
    EXPECT_GE(config.kv_tile, 32) << "KV tile should be at least 32 for Qwen2 7B";

    std::cout << "Qwen2 7B tile config: kv_tile=" << config.kv_tile
              << " q_tile=" << config.q_tile << std::endl;
}

TEST_F(FusedAttentionWoTiledTest, ShouldTile_ShortSequence_ReturnsFalse)
{
    // Short sequences shouldn't benefit from tiling
    const int head_dim = 64;
    const int d_model = 896;

    TileConfig config = llaminar::v2::kernels::compute_tile_config(head_dim, d_model);

    // Sequences shorter than kv_tile shouldn't need tiling
    EXPECT_FALSE(config.should_tile(1));
    EXPECT_FALSE(config.should_tile(config.kv_tile / 2));
}

TEST_F(FusedAttentionWoTiledTest, ShouldTile_LongSequence_ReturnsTrue)
{
    // Long sequences should benefit from tiling
    const int head_dim = 64;
    const int d_model = 896;

    TileConfig config = llaminar::v2::kernels::compute_tile_config(head_dim, d_model);

    // Sequences longer than kv_tile should use tiling
    EXPECT_TRUE(config.should_tile(config.kv_tile + 1));
    EXPECT_TRUE(config.should_tile(config.kv_tile * 2));
    EXPECT_TRUE(config.should_tile(4096)); // Long context
}

// =============================================================================
// Tiled vs Reference Parity Tests
// =============================================================================

/**
 * @brief Helper to run parity test between tiled and reference implementations
 */
void run_parity_test(
    int seq_len,
    int kv_seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int d_model,
    bool causal,
    float min_cos_sim,
    std::mt19937 &rng)
{
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int blocks_per_head = head_dim / 32;

    // Generate random FP32 data
    std::vector<float> Q_fp32(seq_len * num_heads * head_dim);
    std::vector<float> K_fp32(kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> V_fp32(kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> Wo_fp32(d_model * num_heads * head_dim);

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &v : Q_fp32)
        v = dist(rng);
    for (auto &v : K_fp32)
        v = dist(rng);
    for (auto &v : V_fp32)
        v = dist(rng);
    for (auto &v : Wo_fp32)
        v = dist(rng);

    // Quantize to Q8_1
    std::vector<Q8_1Block> Q_q8(seq_len * num_heads * blocks_per_head);
    std::vector<Q8_1Block> K_q8(kv_seq_len * num_kv_heads * blocks_per_head);
    std::vector<Q8_1Block> V_q8(kv_seq_len * num_kv_heads * blocks_per_head);

    quantize_to_q8_1(Q_fp32.data(), Q_q8.data(), Q_fp32.size());
    quantize_to_q8_1(K_fp32.data(), K_q8.data(), K_fp32.size());
    quantize_to_q8_1(V_fp32.data(), V_q8.data(), V_fp32.size());

    // Output buffers
    std::vector<float> output_tiled(seq_len * d_model, 0.0f);
    std::vector<float> output_ref(seq_len * d_model, 0.0f);

    // Set up params
    FusedAttentionWoParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.Wo = Wo_fp32.data();
    params.wo_type = WoWeightType::FP32;
    params.seq_len = seq_len;
    params.kv_seq_len = kv_seq_len;
    params.num_heads = num_heads;
    params.num_kv_heads = num_kv_heads;
    params.head_dim = head_dim;
    params.d_model = d_model;
    params.scale = scale;
    params.causal = causal;
    params.position_offset = 0;

    // Run reference
    params.output = output_ref.data();
    FusedAttentionWoRef ref_kernel;
    ASSERT_TRUE(ref_kernel.execute(params)) << "Reference kernel failed";

    // Run tiled
    params.output = output_tiled.data();
    FusedAttentionWoTiled tiled_kernel;
    ASSERT_TRUE(tiled_kernel.execute(params)) << "Tiled kernel failed";

    // Compare outputs
    float cos_sim = cosine_similarity(output_ref.data(), output_tiled.data(), seq_len * d_model);

    std::cout << "Parity test: seq=" << seq_len << " kv=" << kv_seq_len
              << " heads=" << num_heads << "/" << num_kv_heads
              << " dim=" << head_dim << " d_model=" << d_model
              << " causal=" << causal
              << " cos_sim=" << cos_sim << std::endl;

    EXPECT_GE(cos_sim, min_cos_sim)
        << "Tiled output diverged from reference. Cosine similarity=" << cos_sim;

    // Also check max absolute difference
    float max_diff = 0.0f;
    for (int i = 0; i < seq_len * d_model; ++i)
    {
        max_diff = std::max(max_diff, std::abs(output_ref[i] - output_tiled[i]));
    }
    std::cout << "  Max absolute difference: " << max_diff << std::endl;
}

TEST_F(FusedAttentionWoTiledTest, Parity_ShortSequence_NoCausal)
{
    // Short sequence that fits in one tile
    run_parity_test(
        /*seq_len=*/4,
        /*kv_seq_len=*/4,
        /*num_heads=*/4,
        /*num_kv_heads=*/2,
        /*head_dim=*/64,
        /*d_model=*/256,
        /*causal=*/false,
        /*min_cos_sim=*/0.99f,
        rng_);
}

TEST_F(FusedAttentionWoTiledTest, Parity_ShortSequence_Causal)
{
    // Short sequence with causal masking
    run_parity_test(
        /*seq_len=*/4,
        /*kv_seq_len=*/4,
        /*num_heads=*/4,
        /*num_kv_heads=*/2,
        /*head_dim=*/64,
        /*d_model=*/256,
        /*causal=*/true,
        /*min_cos_sim=*/0.99f,
        rng_);
}

TEST_F(FusedAttentionWoTiledTest, Parity_MediumSequence_NoCausal)
{
    // Medium sequence that may span multiple tiles
    run_parity_test(
        /*seq_len=*/64,
        /*kv_seq_len=*/64,
        /*num_heads=*/14,
        /*num_kv_heads=*/2,
        /*head_dim=*/64,
        /*d_model=*/896,
        /*causal=*/false,
        /*min_cos_sim=*/0.99f,
        rng_);
}

TEST_F(FusedAttentionWoTiledTest, Parity_MediumSequence_Causal)
{
    // Medium sequence with causal masking
    run_parity_test(
        /*seq_len=*/64,
        /*kv_seq_len=*/64,
        /*num_heads=*/14,
        /*num_kv_heads=*/2,
        /*head_dim=*/64,
        /*d_model=*/896,
        /*causal=*/true,
        /*min_cos_sim=*/0.99f,
        rng_);
}

TEST_F(FusedAttentionWoTiledTest, Parity_LongSequence_ExceedsTileSize)
{
    // Long sequence that definitely exceeds tile size
    // This tests the online softmax correction logic
    run_parity_test(
        /*seq_len=*/256,
        /*kv_seq_len=*/256,
        /*num_heads=*/14,
        /*num_kv_heads=*/2,
        /*head_dim=*/64,
        /*d_model=*/896,
        /*causal=*/true,
        /*min_cos_sim=*/0.99f,
        rng_);
}

TEST_F(FusedAttentionWoTiledTest, Parity_Qwen2_0_5B_Dimensions)
{
    // Qwen 2.5 0.5B exact dimensions
    run_parity_test(
        /*seq_len=*/32,
        /*kv_seq_len=*/32,
        /*num_heads=*/14,
        /*num_kv_heads=*/2,
        /*head_dim=*/64,
        /*d_model=*/896,
        /*causal=*/true,
        /*min_cos_sim=*/0.99f,
        rng_);
}

TEST_F(FusedAttentionWoTiledTest, Parity_Qwen2_7B_Dimensions)
{
    // Qwen 2.5 7B dimensions (smaller seq for faster testing)
    run_parity_test(
        /*seq_len=*/16,
        /*kv_seq_len=*/16,
        /*num_heads=*/28,
        /*num_kv_heads=*/4,
        /*head_dim=*/128,
        /*d_model=*/3584,
        /*causal=*/true,
        /*min_cos_sim=*/0.99f,
        rng_);
}

TEST_F(FusedAttentionWoTiledTest, Parity_DecodeMode_SingleQuery)
{
    // Decode mode: single query against long KV cache
    run_parity_test(
        /*seq_len=*/1,
        /*kv_seq_len=*/512,
        /*num_heads=*/14,
        /*num_kv_heads=*/2,
        /*head_dim=*/64,
        /*d_model=*/896,
        /*causal=*/true,
        /*min_cos_sim=*/0.99f,
        rng_);
}

TEST_F(FusedAttentionWoTiledTest, Parity_LongKV_ExceedsTileSize)
{
    // Long KV cache that exceeds tile size (512), exercising the tiled code path
    // This is the key test that validates the tiled online softmax implementation
    run_parity_test(
        /*seq_len=*/16,
        /*kv_seq_len=*/1024, // > 512 kv_tile, forces tiled execution
        /*num_heads=*/14,
        /*num_kv_heads=*/2,
        /*head_dim=*/64,
        /*d_model=*/896,
        /*causal=*/true,
        /*min_cos_sim=*/0.99f, // Should still be nearly identical
        rng_);
}

TEST_F(FusedAttentionWoTiledTest, Parity_VeryLongKV_MultiTile)
{
    // Very long KV that requires multiple tiles (2048 / 512 = 4 tiles)
    // Tests accumulation across multiple tile passes
    run_parity_test(
        /*seq_len=*/4,
        /*kv_seq_len=*/2048, // 4 tiles worth
        /*num_heads=*/4,     // Fewer heads for faster test
        /*num_kv_heads=*/2,
        /*head_dim=*/64,
        /*d_model=*/256,
        /*causal=*/true,
        /*min_cos_sim=*/0.99f,
        rng_);
}

TEST_F(FusedAttentionWoTiledTest, Parity_CrossAttention_DifferentLengths)
{
    // Cross-attention: query length != KV length, non-causal
    run_parity_test(
        /*seq_len=*/32,
        /*kv_seq_len=*/128,
        /*num_heads=*/14,
        /*num_kv_heads=*/2,
        /*head_dim=*/64,
        /*d_model=*/896,
        /*causal=*/false, // Cross-attention is non-causal
        /*min_cos_sim=*/0.99f,
        rng_);
}

// =============================================================================
// Edge Case Tests
// =============================================================================

TEST_F(FusedAttentionWoTiledTest, EdgeCase_MinimalConfig)
{
    // Minimal valid configuration: 1 position, 1 head
    run_parity_test(
        /*seq_len=*/1,
        /*kv_seq_len=*/1,
        /*num_heads=*/1,
        /*num_kv_heads=*/1,
        /*head_dim=*/32, // Minimum: 1 Q8_1 block
        /*d_model=*/32,
        /*causal=*/false,
        /*min_cos_sim=*/0.99f,
        rng_);
}

TEST_F(FusedAttentionWoTiledTest, EdgeCase_HighGQARatio)
{
    // High GQA ratio: 32 query heads per KV head
    run_parity_test(
        /*seq_len=*/16,
        /*kv_seq_len=*/16,
        /*num_heads=*/32,
        /*num_kv_heads=*/1,
        /*head_dim=*/64,
        /*d_model=*/2048,
        /*causal=*/true,
        /*min_cos_sim=*/0.99f,
        rng_);
}

TEST_F(FusedAttentionWoTiledTest, EdgeCase_ManyKVHeads)
{
    // Many KV heads (MHA-like, but with GQA structure)
    run_parity_test(
        /*seq_len=*/16,
        /*kv_seq_len=*/16,
        /*num_heads=*/8,
        /*num_kv_heads=*/8, // MHA: ratio = 1
        /*head_dim=*/64,
        /*d_model=*/512,
        /*causal=*/true,
        /*min_cos_sim=*/0.99f,
        rng_);
}

// =============================================================================
// Batch Processing Tests
// =============================================================================

TEST_F(FusedAttentionWoTiledTest, BatchedExecution_MatchesReference)
{
    // Test batched execution (multiple sequences)
    const int batch_size = 4;
    const int seq_len = 32;
    const int kv_seq_len = 32;
    const int num_heads = 14;
    const int num_kv_heads = 2;
    const int head_dim = 64;
    const int d_model = 896;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int blocks_per_head = head_dim / 32;

    // Generate random data for batch
    std::vector<float> Q_fp32(batch_size * seq_len * num_heads * head_dim);
    std::vector<float> K_fp32(batch_size * kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> V_fp32(batch_size * kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> Wo_fp32(d_model * num_heads * head_dim);

    generate_random_fp32(Q_fp32.data(), Q_fp32.size());
    generate_random_fp32(K_fp32.data(), K_fp32.size());
    generate_random_fp32(V_fp32.data(), V_fp32.size());
    generate_random_fp32(Wo_fp32.data(), Wo_fp32.size());

    // Quantize
    std::vector<Q8_1Block> Q_q8(batch_size * seq_len * num_heads * blocks_per_head);
    std::vector<Q8_1Block> K_q8(batch_size * kv_seq_len * num_kv_heads * blocks_per_head);
    std::vector<Q8_1Block> V_q8(batch_size * kv_seq_len * num_kv_heads * blocks_per_head);

    quantize_to_q8_1(Q_fp32.data(), Q_q8.data(), Q_fp32.size());
    quantize_to_q8_1(K_fp32.data(), K_q8.data(), K_fp32.size());
    quantize_to_q8_1(V_fp32.data(), V_q8.data(), V_fp32.size());

    // Output buffers
    std::vector<float> output_tiled(batch_size * seq_len * d_model, 0.0f);
    std::vector<float> output_ref(batch_size * seq_len * d_model, 0.0f);

    // Set up batched params
    FusedAttentionWoParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.Wo = Wo_fp32.data();
    params.wo_type = WoWeightType::FP32;
    params.seq_len = seq_len;
    params.kv_seq_len = kv_seq_len;
    params.num_heads = num_heads;
    params.num_kv_heads = num_kv_heads;
    params.head_dim = head_dim;
    params.d_model = d_model;
    params.scale = scale;
    params.causal = true;
    params.position_offset = 0;
    params.batch_size = batch_size;

    // Run reference
    params.output = output_ref.data();
    FusedAttentionWoRef ref_kernel;
    ASSERT_TRUE(ref_kernel.execute(params)) << "Reference kernel failed";

    // Run tiled
    params.output = output_tiled.data();
    FusedAttentionWoTiled tiled_kernel;
    ASSERT_TRUE(tiled_kernel.execute(params)) << "Tiled kernel failed";

    // Compare each batch item
    for (int b = 0; b < batch_size; ++b)
    {
        const float *ref_batch = output_ref.data() + b * seq_len * d_model;
        const float *tiled_batch = output_tiled.data() + b * seq_len * d_model;

        float cos_sim = cosine_similarity(ref_batch, tiled_batch, seq_len * d_model);

        std::cout << "Batch " << b << " cosine similarity: " << cos_sim << std::endl;

        EXPECT_GE(cos_sim, 0.99f)
            << "Batch " << b << " tiled output diverged from reference";
    }
}

// =============================================================================
// WoWeightType Parity Tests
// =============================================================================

TEST_F(FusedAttentionWoTiledTest, Parity_Q8_1_Wo_Weights)
{
    // Test with Q8_1 Wo weights
    const int seq_len = 16;
    const int kv_seq_len = 32;
    const int num_heads = 4;
    const int num_kv_heads = 2;
    const int head_dim = 64;
    const int d_model = 256;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int blocks_per_head = head_dim / 32;

    // Generate random FP32 data
    std::vector<float> Q_fp32(seq_len * num_heads * head_dim);
    std::vector<float> K_fp32(kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> V_fp32(kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> Wo_fp32(d_model * num_heads * head_dim);

    generate_random_fp32(Q_fp32.data(), Q_fp32.size());
    generate_random_fp32(K_fp32.data(), K_fp32.size());
    generate_random_fp32(V_fp32.data(), V_fp32.size());
    generate_random_fp32(Wo_fp32.data(), Wo_fp32.size());

    // Quantize QKV and Wo
    std::vector<Q8_1Block> Q_q8(seq_len * num_heads * blocks_per_head);
    std::vector<Q8_1Block> K_q8(kv_seq_len * num_kv_heads * blocks_per_head);
    std::vector<Q8_1Block> V_q8(kv_seq_len * num_kv_heads * blocks_per_head);

    const int wo_blocks = (d_model * num_heads * head_dim) / 32;
    std::vector<Q8_1Block> Wo_q8(wo_blocks);

    quantize_to_q8_1(Q_fp32.data(), Q_q8.data(), Q_fp32.size());
    quantize_to_q8_1(K_fp32.data(), K_q8.data(), K_fp32.size());
    quantize_to_q8_1(V_fp32.data(), V_q8.data(), V_fp32.size());
    quantize_to_q8_1(Wo_fp32.data(), Wo_q8.data(), Wo_fp32.size());

    // Output buffers
    std::vector<float> output_tiled(seq_len * d_model, 0.0f);
    std::vector<float> output_ref(seq_len * d_model, 0.0f);

    // Set up params with Q8_1 Wo
    FusedAttentionWoParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.Wo = Wo_q8.data();
    params.wo_type = WoWeightType::Q8_1;
    params.seq_len = seq_len;
    params.kv_seq_len = kv_seq_len;
    params.num_heads = num_heads;
    params.num_kv_heads = num_kv_heads;
    params.head_dim = head_dim;
    params.d_model = d_model;
    params.scale = scale;
    params.causal = true;
    params.position_offset = 0;

    // Run reference
    params.output = output_ref.data();
    FusedAttentionWoRef ref_kernel;
    ASSERT_TRUE(ref_kernel.execute(params)) << "Reference kernel failed";

    // Run tiled
    params.output = output_tiled.data();
    FusedAttentionWoTiled tiled_kernel;
    ASSERT_TRUE(tiled_kernel.execute(params)) << "Tiled kernel failed";

    float cos_sim = cosine_similarity(output_ref.data(), output_tiled.data(), seq_len * d_model);

    std::cout << "Q8_1 Wo parity: cos_sim=" << cos_sim << std::endl;

    EXPECT_GE(cos_sim, 0.99f) << "Q8_1 Wo tiled output diverged from reference";
}
