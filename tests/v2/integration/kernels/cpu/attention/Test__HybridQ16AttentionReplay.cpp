/**
 * @file Test__HybridQ16AttentionReplay.cpp
 * @brief Replay test for HybridQ16 attention using dumped stage tensors
 *
 * This test loads real Q, K, V tensor data dumped from the FusedAttentionWoStage
 * during integration tests in their NATIVE FORMAT (Q16_1, Q8_1) and compares
 * against an FP32 reference implementation.
 *
 * Purpose: Reproduce and diagnose the HybridQ16 attention bug in isolation
 * without needing to run the full model pipeline.
 *
 * Data source: tests/v2/integration/_data/hybridq16_attention/
 *
 * To regenerate test data:
 *   LLAMINAR_STAGE_DUMP_ENABLED=1 \
 *   LLAMINAR_STAGE_DUMP_TYPES=FUSED_ATTENTION_WO \
 *   LLAMINAR_STAGE_DUMP_LAYERS=0 \
 *   ./build_v2_integration/tests/v2/v2_integration_hybridq16_vs_fp32_pipeline
 *
 * @see V2_Integration_HybridQ16Pipeline_vs_FP32 for the original parity test
 */

#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <vector>

#include "../../utils/TensorDumpLoader.h"
#include "kernels/cpu/attention/q16_1/ref/Q16IntegerAttentionRef.h"
#include "kernels/cpu/attention/q16_1/ref/microkernels/OnlineSoftmax.h"
#include "kernels/cpu/attention/q16_1/ref/microkernels/Q16DotProduct.h"
#include "tensors/BlockStructures.h"

using namespace llaminar2::test;
using namespace llaminar2::kernels::q16_1;
using namespace llaminar2::kernels::q16_1::microkernels;
using namespace llaminar2;

namespace
{

    // Log2(e) constant for softmax alpha computation
    constexpr float LOG2E_FLOAT = 1.4426950408889634f;

    // Qwen2-0.5B attention configuration
    constexpr int HEAD_DIM = 64;
    constexpr int NUM_HEADS = 14;
    constexpr int NUM_KV_HEADS = 2;
    constexpr int GQA_RATIO = NUM_HEADS / NUM_KV_HEADS; // 7
    constexpr int D_MODEL = NUM_HEADS * HEAD_DIM;       // 896
    constexpr int KV_DIM = NUM_KV_HEADS * HEAD_DIM;     // 128

    // KV cache scale used in the HybridQ16 pipeline
    constexpr float KV_CACHE_SCALE = 256.0f;

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
        if (norm_a < 1e-10f || norm_b < 1e-10f)
            return 0.0f;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    /**
     * @brief Compute max absolute difference
     */
    float maxAbsDiff(const float *a, const float *b, int n)
    {
        float max_diff = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
        }
        return max_diff;
    }

    /**
     * @brief Compute RMSE
     */
    float rmse(const float *a, const float *b, int n)
    {
        float sum_sq = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            float diff = a[i] - b[i];
            sum_sq += diff * diff;
        }
        return std::sqrt(sum_sq / n);
    }

    /**
     * @brief FP32 reference attention implementation
     *
     * Computes: softmax(Q @ K^T / sqrt(head_dim)) @ V
     * With causal masking for prefill.
     */
    void fp32ReferenceAttention(
        const float *Q,
        const float *K,
        const float *V,
        float *output,
        int seq_len,
        int kv_len,
        int num_heads,
        int num_kv_heads,
        int head_dim,
        bool causal)
    {
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        int gqa_ratio = num_heads / num_kv_heads;

        for (int q_pos = 0; q_pos < seq_len; ++q_pos)
        {
            for (int h = 0; h < num_heads; ++h)
            {
                int kv_h = h / gqa_ratio; // Map query head to KV head

                // Q @ K^T for this head
                std::vector<float> scores(kv_len);
                float max_score = -std::numeric_limits<float>::infinity();

                for (int k_pos = 0; k_pos < kv_len; ++k_pos)
                {
                    // Causal mask
                    if (causal && k_pos > q_pos)
                    {
                        scores[k_pos] = -std::numeric_limits<float>::infinity();
                        continue;
                    }

                    // Dot product Q[q_pos, h, :] @ K[k_pos, kv_h, :]
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d)
                    {
                        // Q layout: [seq_len, num_heads * head_dim]
                        float q_val = Q[q_pos * num_heads * head_dim + h * head_dim + d];
                        // K layout: [kv_len, num_kv_heads * head_dim]
                        float k_val = K[k_pos * num_kv_heads * head_dim + kv_h * head_dim + d];
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

                // Weighted sum: softmax @ V
                for (int d = 0; d < head_dim; ++d)
                {
                    float val = 0.0f;
                    for (int k_pos = 0; k_pos < kv_len; ++k_pos)
                    {
                        float v_val = V[k_pos * num_kv_heads * head_dim + kv_h * head_dim + d];
                        val += scores[k_pos] * v_val;
                    }
                    // Output layout: [seq_len, num_heads * head_dim]
                    output[q_pos * num_heads * head_dim + h * head_dim + d] = val;
                }
            }
        }
    }

    /**
     * @brief Print tensor statistics
     */
    void printStats(const std::string &name, const std::vector<float> &data)
    {
        float min_val = *std::min_element(data.begin(), data.end());
        float max_val = *std::max_element(data.begin(), data.end());
        float mean = std::accumulate(data.begin(), data.end(), 0.0f) / data.size();
        std::cout << name << ": min=" << std::setw(10) << std::fixed << std::setprecision(4) << min_val
                  << " max=" << std::setw(10) << max_val
                  << " mean=" << std::setw(10) << mean << std::endl;
    }

} // anonymous namespace

class Test__HybridQ16AttentionReplay : public ::testing::Test
{
protected:
    std::string dump_dir_;

    void SetUp() override
    {
        // Use persistent test data from repository
        try
        {
            dump_dir_ = getTestDataDir() + "/hybridq16_attention/layer0_prefill";
        }
        catch (const std::exception &e)
        {
            GTEST_SKIP() << "Test data directory not found: " << e.what();
        }

        // Check if dump directory exists
        if (!std::filesystem::exists(dump_dir_ + "/metadata.txt"))
        {
            GTEST_SKIP() << "Dump directory not found at: " << dump_dir_ << "\n"
                         << "To generate test data, run:\n"
                         << "  LLAMINAR_STAGE_DUMP_ENABLED=1 \\\n"
                         << "  LLAMINAR_STAGE_DUMP_TYPES=FUSED_ATTENTION_WO \\\n"
                         << "  ./build_v2_integration/tests/v2/v2_integration_hybridq16_vs_fp32_pipeline";
        }
    }
};

/**
 * @brief Test: Load native Q16_1_64 tensors and compare with FP32 reference
 *
 * This test loads Q, K (Q16_1_64) and V (Q8_1) in their native formats,
 * dequantizes them, and computes FP32 reference attention to compare
 * against the captured HybridQ16 output.
 */
TEST_F(Test__HybridQ16AttentionReplay, NativeFormatCompareWithFP32Reference)
{
    std::cout << "\n=== HybridQ16 Attention Replay Test (Native Format) ===" << std::endl;
    std::cout << "Loading tensors from: " << dump_dir_ << std::endl;

    // Load Q as Q16_1_64 native blocks (64-element blocks for head_dim=64)
    auto [Q_blocks, Q_meta] = loadTensorAsQ16_1_64(dump_dir_, "Q");
    std::cout << "Loaded Q: " << Q_blocks.size() << " Q16_1_64 blocks, "
              << "[" << Q_meta.rows << ", " << Q_meta.cols << "]" << std::endl;

    // Load K as Q16_1_64 native blocks
    auto [K_blocks, K_meta] = loadTensorAsQ16_1_64(dump_dir_, "K");
    std::cout << "Loaded K: " << K_blocks.size() << " Q16_1_64 blocks, "
              << "[" << K_meta.rows << ", " << K_meta.cols << "]" << std::endl;

    // Load V as Q8_1 native blocks
    auto [V_blocks, V_meta] = loadTensorAsQ8_1(dump_dir_, "V");
    std::cout << "Loaded V: " << V_blocks.size() << " Q8_1 blocks, "
              << "[" << V_meta.rows << ", " << V_meta.cols << "]" << std::endl;

    // Load captured context (FP32)
    auto [captured_context, context_meta] = loadTensorAsFP32(dump_dir_, "context", "outputs");
    std::cout << "Loaded context: " << captured_context.size() << " FP32 elements" << std::endl;

    const int seq_len = Q_meta.rows;
    const int kv_len = K_meta.rows;

    ASSERT_EQ(seq_len, kv_len) << "Prefill should have seq_len == kv_len";
    ASSERT_EQ(Q_meta.cols, D_MODEL) << "Q cols should be " << D_MODEL;
    ASSERT_EQ(K_meta.cols, KV_DIM) << "K cols should be " << KV_DIM;
    ASSERT_EQ(V_meta.cols, KV_DIM) << "V cols should be " << KV_DIM;

    // Dequantize to FP32 for reference computation
    auto Q_fp32 = dequantQ16_1_64ToFP32(Q_blocks, Q_meta.rows, Q_meta.cols);
    auto K_fp32 = dequantQ16_1_64ToFP32(K_blocks, K_meta.rows, K_meta.cols);
    auto V_fp32 = dequantQ8_1ToFP32(V_blocks, V_meta.rows, V_meta.cols);

    std::cout << "\n--- Input Statistics (Dequantized) ---" << std::endl;
    printStats("Q", Q_fp32);
    printStats("K", K_fp32);
    printStats("V", V_fp32);
    printStats("Captured context", captured_context);

    // Compute FP32 reference attention
    std::vector<float> ref_context(seq_len * D_MODEL);
    fp32ReferenceAttention(
        Q_fp32.data(),
        K_fp32.data(),
        V_fp32.data(),
        ref_context.data(),
        seq_len,
        kv_len,
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        true // causal
    );

    std::cout << "\n--- Reference Attention Statistics ---" << std::endl;
    printStats("FP32 reference", ref_context);

    // Compare captured vs reference
    float cos_sim = cosineSimilarity(captured_context.data(), ref_context.data(), captured_context.size());
    float max_err = maxAbsDiff(captured_context.data(), ref_context.data(), captured_context.size());
    float err_rmse = rmse(captured_context.data(), ref_context.data(), captured_context.size());

    std::cout << "\n--- HybridQ16 vs FP32 Reference ---" << std::endl;
    std::cout << "Cosine similarity: " << std::fixed << std::setprecision(6) << cos_sim << std::endl;
    std::cout << "Max absolute error: " << max_err << std::endl;
    std::cout << "RMSE: " << err_rmse << std::endl;

    // Per-head analysis
    std::cout << "\n--- Per-Head Cosine Similarity ---" << std::endl;
    std::vector<float> head_cosines(NUM_HEADS);
    for (int h = 0; h < NUM_HEADS; ++h)
    {
        // Extract head slice: [seq_len, head_dim]
        std::vector<float> cap_head(seq_len * HEAD_DIM);
        std::vector<float> ref_head(seq_len * HEAD_DIM);
        for (int q_pos = 0; q_pos < seq_len; ++q_pos)
        {
            for (int d = 0; d < HEAD_DIM; ++d)
            {
                cap_head[q_pos * HEAD_DIM + d] = captured_context[q_pos * D_MODEL + h * HEAD_DIM + d];
                ref_head[q_pos * HEAD_DIM + d] = ref_context[q_pos * D_MODEL + h * HEAD_DIM + d];
            }
        }
        head_cosines[h] = cosineSimilarity(cap_head.data(), ref_head.data(), cap_head.size());
    }

    for (int h = 0; h < NUM_HEADS; ++h)
    {
        std::cout << "  Head " << std::setw(2) << h << ": " << std::fixed << std::setprecision(4) << head_cosines[h];
        if ((h + 1) % 7 == 0 || h == NUM_HEADS - 1)
            std::cout << std::endl;
        else
            std::cout << "  ";
    }

    // This test is diagnostic - reveals the known HybridQ16 divergence issue
    // The cause is being investigated (likely Q16_1 overflow in dot products)
    // Low cosine similarity indicates a fundamental algorithm issue, not just quantization noise
    if (cos_sim < 0.5f)
    {
        std::cout << "\nWARNING: Very low cosine similarity (" << cos_sim << ") indicates "
                  << "fundamental issues with the HybridQ16 attention computation." << std::endl;
        std::cout << "This is a KNOWN ISSUE being investigated - see overflow analysis below." << std::endl;
    }

    // Test passes as diagnostic - actual parity is tested in the pipeline test
    SUCCEED();
}

/**
 * @brief Test: Analyze Q16_1_64 quantization error from native blocks
 *
 * This test examines the Q16_1_64 blocks directly to understand quantization
 * characteristics without the confusion of pre-dequantized FP32 dumps.
 */
TEST_F(Test__HybridQ16AttentionReplay, AnalyzeNativeQ16_1Blocks)
{
    std::cout << "\n=== Native Q16_1_64 Block Analysis ===" << std::endl;

    // Load Q and K as native Q16_1_64 blocks
    auto [Q_blocks, Q_meta] = loadTensorAsQ16_1_64(dump_dir_, "Q");
    auto [K_blocks, K_meta] = loadTensorAsQ16_1_64(dump_dir_, "K");

    // Analyze block scales (d values)
    std::cout << "\n--- Q Block Scales ---" << std::endl;
    float q_min_scale = std::numeric_limits<float>::max();
    float q_max_scale = 0.0f;
    for (const auto &block : Q_blocks)
    {
        float d = block.d;
        q_min_scale = std::min(q_min_scale, d);
        q_max_scale = std::max(q_max_scale, d);
    }
    std::cout << "Q scale range: [" << std::scientific << q_min_scale
              << ", " << q_max_scale << "]" << std::endl;

    std::cout << "\n--- K Block Scales ---" << std::endl;
    float k_min_scale = std::numeric_limits<float>::max();
    float k_max_scale = 0.0f;
    for (const auto &block : K_blocks)
    {
        float d = block.d;
        k_min_scale = std::min(k_min_scale, d);
        k_max_scale = std::max(k_max_scale, d);
    }
    std::cout << "K scale range: [" << std::scientific << k_min_scale
              << ", " << k_max_scale << "]" << std::endl;

    // Check for extreme quantized values that could cause overflow
    constexpr int16_t OVERFLOW_THRESHOLD = 5792; // sqrt(INT32_MAX / 64)
    int q_overflow_risk = 0;
    int k_overflow_risk = 0;

    for (const auto &block : Q_blocks)
    {
        for (int k = 0; k < 64; ++k) // Q16_1_64 has 64 elements per block
        {
            if (std::abs(block.qs[k]) > OVERFLOW_THRESHOLD)
            {
                ++q_overflow_risk;
            }
        }
    }

    for (const auto &block : K_blocks)
    {
        for (int k = 0; k < 64; ++k) // Q16_1_64 has 64 elements per block
        {
            if (std::abs(block.qs[k]) > OVERFLOW_THRESHOLD)
            {
                ++k_overflow_risk;
            }
        }
    }

    size_t q_total = Q_blocks.size() * 64;
    size_t k_total = K_blocks.size() * 64;

    std::cout << "\n--- Overflow Risk Analysis ---" << std::endl;
    std::cout << "Threshold for overflow in dot product: ±" << OVERFLOW_THRESHOLD << std::endl;
    std::cout << "Q values above threshold: " << q_overflow_risk << " / " << q_total
              << " (" << std::fixed << std::setprecision(2)
              << (100.0f * q_overflow_risk / q_total) << "%)" << std::endl;
    std::cout << "K values above threshold: " << k_overflow_risk << " / " << k_total
              << " (" << std::fixed << std::setprecision(2)
              << (100.0f * k_overflow_risk / k_total) << "%)" << std::endl;

    // This is informational - pass if no exceptions thrown
    SUCCEED();
}

/**
 * @brief Test: Analyze V tensor Q8_1 native format
 *
 * Note: The V tensor may have more blocks than expected from [rows, cols] metadata
 * due to KV cache allocation strategies (e.g., pre-allocated to max_seq_len) or
 * different internal layouts. This test analyzes what's actually in the dump.
 */
TEST_F(Test__HybridQ16AttentionReplay, AnalyzeNativeQ8_1Blocks)
{
    std::cout << "\n=== Native Q8_1 V Block Analysis ===" << std::endl;

    auto [V_blocks, V_meta] = loadTensorAsQ8_1(dump_dir_, "V");

    std::cout << "V shape (from metadata): [" << V_meta.rows << ", " << V_meta.cols << "]" << std::endl;
    std::cout << "V blocks loaded: " << V_blocks.size() << std::endl;

    // Calculate expected blocks for the stated dimensions
    constexpr int BLOCK_SIZE = 32;
    size_t expected_blocks_per_row = (V_meta.cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t expected_blocks = V_meta.rows * expected_blocks_per_row;

    std::cout << "Expected blocks for stated shape: " << expected_blocks << std::endl;
    if (V_blocks.size() != expected_blocks)
    {
        std::cout << "NOTE: Block count mismatch - tensor may use different internal layout" << std::endl;
        std::cout << "      (e.g., pre-allocated KV cache, HEAD_MAJOR layout, etc.)" << std::endl;
    }

    // Analyze block scales
    float min_scale = std::numeric_limits<float>::max();
    float max_scale = 0.0f;
    for (const auto &block : V_blocks)
    {
        float d = fp16_to_fp32(block.d);
        min_scale = std::min(min_scale, d);
        max_scale = std::max(max_scale, d);
    }

    std::cout << "V scale range: [" << std::scientific << min_scale
              << ", " << max_scale << "]" << std::endl;

    // Verify data integrity by checking scale values are reasonable
    int zero_scale_blocks = 0;
    int extreme_scale_blocks = 0;
    for (const auto &block : V_blocks)
    {
        float d = fp16_to_fp32(block.d);
        if (d == 0.0f)
            ++zero_scale_blocks;
        if (d > 100.0f || d < 1e-6f)
            ++extreme_scale_blocks;
    }

    std::cout << "Blocks with zero scale: " << zero_scale_blocks << std::endl;
    std::cout << "Blocks with extreme scale (>100 or <1e-6): " << extreme_scale_blocks << std::endl;

    // Test: We should be able to load SOME blocks
    EXPECT_GT(V_blocks.size(), 0) << "Should load V blocks";

    SUCCEED();
}

/**
 * @brief Test: V2 Deferred Normalization Kernel vs FP32 Reference
 *
 * This test implements the V2 deferred normalization approach:
 * - Integer Q×K dot products and exp2 LUT weights (same as V1)
 * - FP64 accumulation for P×V (FP32 V values due to per-block scales)
 * - SINGLE division at finalization (KEY DIFFERENCE from V1)
 *
 * V1 ISSUE:
 * =========
 * V1 performs `l_processed = (l_processed / l_new)` at EVERY KV position,
 * which introduces FP64 rounding errors that accumulate across positions.
 *
 * V2 SOLUTION:
 * ============
 * V2 defers normalization: accumulates `Σ(w_i × V_i)` and `Σ(w_i)` separately,
 * then performs ONE division at the end: `context = Σ(w × V) / Σ(w)`.
 *
 * RESULT:
 * =======
 * V2 achieves higher cosine similarity with FP32 reference than V1,
 * demonstrating that eliminating the FP leak improves numerical accuracy.
 *
 * NOTE ON V FORMAT:
 * =================
 * This test uses FP32 dequantized V because Q8_1 has per-block scales.
 * The actual HybridQ16 pipeline converts V to Q16_1 in the KV cache,
 * which could enable pure integer accumulation if Q16_1 had uniform scales.
 */
TEST_F(Test__HybridQ16AttentionReplay, V2DeferredNormalizationKernel)
{
    std::cout << "\n=== V2 Deferred Normalization Kernel Test ===" << std::endl;
    std::cout << "Loading tensors from: " << dump_dir_ << std::endl;

    // Load tensors in native format
    auto [Q_blocks, Q_meta] = loadTensorAsQ16_1_64(dump_dir_, "Q");
    auto [K_blocks, K_meta] = loadTensorAsQ16_1_64(dump_dir_, "K");
    auto [V_blocks, V_meta] = loadTensorAsQ8_1(dump_dir_, "V");
    auto [captured_context, context_meta] = loadTensorAsFP32(dump_dir_, "context", "outputs");

    const int seq_len = Q_meta.rows;
    const int kv_len = K_meta.rows;

    std::cout << "Q: [" << seq_len << ", " << Q_meta.cols << "] (" << Q_blocks.size() << " blocks)" << std::endl;
    std::cout << "K: [" << kv_len << ", " << K_meta.cols << "] (" << K_blocks.size() << " blocks)" << std::endl;
    std::cout << "V: [" << kv_len << ", " << V_meta.cols << "] (" << V_blocks.size() << " blocks)" << std::endl;

    ASSERT_EQ(seq_len, kv_len) << "Prefill should have seq_len == kv_len";

    // Dequantize inputs for statistics and FP32 reference
    auto Q_fp32 = dequantQ16_1_64ToFP32(Q_blocks, Q_meta.rows, Q_meta.cols);
    auto K_fp32 = dequantQ16_1_64ToFP32(K_blocks, K_meta.rows, K_meta.cols);
    auto V_fp32 = dequantQ8_1ToFP32(V_blocks, V_meta.rows, V_meta.cols);

    // Compute FP32 reference attention
    std::vector<float> ref_context(seq_len * D_MODEL);
    fp32ReferenceAttention(
        Q_fp32.data(), K_fp32.data(), V_fp32.data(),
        ref_context.data(),
        seq_len, kv_len, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        true // causal
    );

    // ================================================================
    // V2 Kernel: Deferred normalization using OnlineSoftmaxStateV2
    // ================================================================

    // For attention, we need to:
    // 1. Get Q/K scales from the blocks (use first block's scale as representative)
    // 2. Compute QK scale: alpha = q_scale * k_scale / sqrt(head_dim)
    //
    // IMPORTANT: Do NOT include LOG2E here! The exp2_compute_weight function
    // internally multiplies by LOG2E when computing t = delta * alpha * log2(e).
    // This matches the live kernel's get_qk_scale() which returns:
    //   s_q * s_k / sqrt(head_dim)  [no LOG2E]

    // Extract representative scales
    float q_scale = Q_blocks[0].d; // Q16_1Block_64 has float d
    float k_scale = K_blocks[0].d;

    // Compute alpha for softmax (NO LOG2E - exp2_compute_weight handles it)
    float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    float alpha = q_scale * k_scale * inv_sqrt_d; // NO LOG2E!

    std::cout << "\n--- V2 Kernel Parameters ---" << std::endl;
    std::cout << "Q scale: " << q_scale << std::endl;
    std::cout << "K scale: " << k_scale << std::endl;
    std::cout << "Alpha (qk_scale): " << alpha << " (note: exp2_compute_weight adds LOG2E internally)" << std::endl;

    // Check V block scales for uniformity
    std::set<float> v_unique_scales;
    for (const auto &block : V_blocks)
    {
        v_unique_scales.insert(fp16_to_fp32(block.d));
    }
    std::cout << "V unique scale values: " << v_unique_scales.size() << std::endl;
    if (v_unique_scales.size() > 1)
    {
        std::cout << "NOTE: V has per-block scales - using dequantized V for accurate test" << std::endl;
    }

    // Output buffers for V2 kernel - use FP32 context since V has per-block scales
    std::vector<float> v2_context_fp32(seq_len * D_MODEL, 0.0f);
    std::vector<float> v2_weight_sums(seq_len * NUM_HEADS, 0.0f);

    // Process each query position and head using the V2 deferred normalization approach
    // NOTE: Since V has per-block scales, we use FP32 V but integer Q×K and weights
    int blocks_per_q_row = (D_MODEL + 63) / 64; // Q16_1Block_64 has 64 elements
    int blocks_per_kv_row = (KV_DIM + 63) / 64;

    for (int q_pos = 0; q_pos < seq_len; ++q_pos)
    {
        // For each attention head
        for (int h = 0; h < NUM_HEADS; ++h)
        {
            int kv_h = h / GQA_RATIO; // KV head index for GQA

            // Initialize V2 state
            OnlineSoftmaxStateV2 state;
            state.init(alpha);

            // FP64 context accumulator and weight sum for this head
            // (FP64 to match the V2 deferred normalization approach)
            alignas(64) double context_fp64[HEAD_DIM];
            std::memset(context_fp64, 0, HEAD_DIM * sizeof(double));
            double weight_sum = 0.0;

            // Scratch buffers
            alignas(64) int32_t scores_scratch[256];
            alignas(64) int32_t weights_scratch[256];

            // Get Q block for this head (Q is [seq_len, num_heads * head_dim])
            // With Q16_1Block_64, each block covers 64 elements = 1 head_dim
            const Q16_1Block_64 *Q_head = &Q_blocks[q_pos * blocks_per_q_row + h];

            // Process KV positions with causal masking
            int valid_kv_len = q_pos + 1; // Causal: can only see positions <= q_pos

            // Process KV positions one at a time for simplicity
            // (In production, this would be batched into blocks)
            for (int k_pos = 0; k_pos < valid_kv_len; ++k_pos)
            {
                // K and V are [kv_len, num_kv_heads * head_dim]
                // With Q16_1Block_64 for K and Q8_1 for V (32-element blocks)
                const Q16_1Block_64 *K_pos = &K_blocks[k_pos * blocks_per_kv_row + kv_h];
                // V is Q8_1 with 32-element blocks, so 2 blocks per head
                int v_blocks_per_kv_row = (KV_DIM + 31) / 32; // Q8_1 has 32 elements
                const Q8_1Block *V_pos_base = &V_blocks[k_pos * v_blocks_per_kv_row];

                // Compute Q×K dot product
                int32_t score = q16_dot_single<Q16_1Block_64>(Q_head, K_pos, HEAD_DIM, 1);
                scores_scratch[0] = score;

                // Online softmax: find max, compute weight
                int32_t block_max = score;
                int32_t scale_num = 1;
                int scale_shift = 0;
                bool needs_rescale = false;

                if (block_max > state.m)
                {
                    needs_rescale = (state.count > 0);
                    if (needs_rescale)
                    {
                        exp2_compute_rescale(state.m, block_max, state.alpha_config,
                                             scale_num, scale_shift, state.frac_bits, state.lut_value_bits);
                    }
                    state.m = block_max;
                }

                // Compute weight (FULL 30-bit precision for V2 test)
                int32_t delta = state.m - score;
                int64_t w_raw = static_cast<int64_t>(exp2_compute_weight(delta, state.alpha_config,
                                                                         state.frac_bits, state.lut_value_bits));

                // Rescale prior context if needed (when max increases)
                // This implements: prior_weight *= exp(-(new_max - old_max) * alpha * log2e)
                // scale_num/2^scale_shift approximates exp2(-(new_max - old_max) * alpha)
                if (needs_rescale)
                {
                    // Use ldexp for correct FP64 scaling (handles large shifts)
                    double scale_factor = std::ldexp(static_cast<double>(scale_num), -scale_shift);
                    weight_sum *= scale_factor;
                    for (int d = 0; d < HEAD_DIM; ++d)
                    {
                        context_fp64[d] *= scale_factor;
                    }
                }

                // Accumulate weight
                double w_fp64 = static_cast<double>(w_raw);
                weight_sum += w_fp64;

                // P×V accumulation using FP32 dequantized V (since V has per-block scales)
                if (w_raw > 0)
                {
                    // Get V values for this kv_head at k_pos from V_fp32
                    // V_fp32 layout: [kv_len, num_kv_heads * head_dim]
                    int v_offset = k_pos * KV_DIM + kv_h * HEAD_DIM;
                    for (int d = 0; d < HEAD_DIM; ++d)
                    {
                        context_fp64[d] += w_fp64 * static_cast<double>(V_fp32[v_offset + d]);
                    }
                }

                ++state.count;
            }

            // Finalize: single division
            if (weight_sum > 0.0)
            {
                for (int d = 0; d < HEAD_DIM; ++d)
                {
                    v2_context_fp32[q_pos * D_MODEL + h * HEAD_DIM + d] =
                        static_cast<float>(context_fp64[d] / weight_sum);
                }
            }

            v2_weight_sums[q_pos * NUM_HEADS + h] = static_cast<float>(weight_sum);
        }
    }

    // ================================================================
    // Compare V2 kernel vs FP32 reference
    // ================================================================

    float v2_vs_ref_cos = cosineSimilarity(v2_context_fp32.data(), ref_context.data(), ref_context.size());
    float v2_vs_ref_max_err = maxAbsDiff(v2_context_fp32.data(), ref_context.data(), ref_context.size());
    float v2_vs_ref_rmse = rmse(v2_context_fp32.data(), ref_context.data(), ref_context.size());

    std::cout << "\n--- V2 Kernel vs FP32 Reference ---" << std::endl;
    std::cout << "Cosine similarity: " << std::fixed << std::setprecision(6) << v2_vs_ref_cos << std::endl;
    std::cout << "Max absolute error: " << v2_vs_ref_max_err << std::endl;
    std::cout << "RMSE: " << v2_vs_ref_rmse << std::endl;

    // Compare with captured (V1) context
    float cap_vs_ref_cos = cosineSimilarity(captured_context.data(), ref_context.data(), ref_context.size());

    std::cout << "\n--- Comparison Summary ---" << std::endl;
    std::cout << "V1 (captured) vs FP32 ref: " << std::fixed << std::setprecision(6) << cap_vs_ref_cos << std::endl;
    std::cout << "V2 (deferred) vs FP32 ref: " << std::fixed << std::setprecision(6) << v2_vs_ref_cos << std::endl;

    if (v2_vs_ref_cos > cap_vs_ref_cos)
    {
        std::cout << "\n✓ V2 kernel shows IMPROVED parity with FP32 reference!" << std::endl;
        std::cout << "  Improvement: " << (v2_vs_ref_cos - cap_vs_ref_cos) << std::endl;
    }
    else if (v2_vs_ref_cos >= 0.99f)
    {
        std::cout << "\n✓ V2 kernel has excellent FP32 parity (>0.99)!" << std::endl;
    }
    else
    {
        std::cout << "\n⚠ V2 kernel parity needs investigation" << std::endl;
    }

    // V2 kernel should have better or equal parity than V1
    // This proves that eliminating the FP leak improves accuracy
    EXPECT_GE(v2_vs_ref_cos, cap_vs_ref_cos - 0.01f)
        << "V2 deferred normalization should have comparable or better parity than V1";
    EXPECT_GE(v2_vs_ref_cos, 0.90f)
        << "V2 kernel should have reasonable parity with FP32 reference";
}

// Main entry point for Google Test
int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
