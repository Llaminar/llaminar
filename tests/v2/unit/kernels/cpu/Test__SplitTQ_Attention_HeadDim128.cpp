/**
 * @file Test__SplitTQ_Attention_HeadDim128.cpp
 * @brief Unit tests for split TQ8-K / TQ4-V KV cache + attention with head_dim=128 (Qwen3-like).
 *
 * The existing TQ KV cache tests use head_dim=64 (Qwen2-like). Qwen3 uses
 * head_dim=128, which triggers different code paths:
 *   - TQ8Block_128 (136 bytes) for K vs TQ4Block_128 (72 bytes) for V
 *   - AttentionComputeStage constructor skips K buffer pre-allocation for head_dim==128
 *   - Separate K (TQ8) and V (TQ4) dequant paths using the 128-element template specialization
 *
 * These tests validate:
 *   1. Split TQ round-trip quality: K(FP32→TQ8→FP32) and V(FP32→TQ4→FP32) with head_dim=128
 *   2. KV cache round-trip (append → gather → dequant) at head_dim=128 with split precision
 *   3. Separate K (TQ8) and V (TQ4) dequant correctness at head_dim=128
 *   4. Full attention pipeline: dequant(TQ8(K)), dequant(TQ4(V)) → attention kernel → output
 *      compared against attention with the original FP32 K/V (ground truth)
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "kernels/cpu/CPURingKVCache.h"
#include "kernels/cpu/turboquant/TurboQuantContext.h"
#include "kernels/cpu/turboquant/TurboQuantDequantizeTQ4.h"
#include "kernels/cpu/turboquant/TurboQuantDequantizeTQ8.h"
#include "tensors/Tensors.h"
#include "tensors/TensorKernels.h"
#include "utils/MPIContext.h"

using namespace llaminar2;

// ─────────────────────────────────────────────────────────────────────
// Scalar reference attention (for ground truth comparison)
// ─────────────────────────────────────────────────────────────────────
namespace ref
{
    /// Scalar softmax attention:  output = softmax(Q·K^T / sqrt(d)) · V
    /// Layout: Q  [seq_len,  n_heads * head_dim]
    ///         K  [kv_len,   n_kv_heads * head_dim]
    ///         V  [kv_len,   n_kv_heads * head_dim]
    ///         O  [seq_len,  n_heads * head_dim]
    static void attention(
        const float *Q, const float *K, const float *V, float *O,
        int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal)
    {
        const int heads_per_kv = n_heads / n_kv_heads;
        const int q_stride = n_heads * head_dim;
        const int kv_stride = n_kv_heads * head_dim;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        for (int h = 0; h < n_heads; ++h)
        {
            const int kv_h = h / heads_per_kv;
            for (int q_pos = 0; q_pos < seq_len; ++q_pos)
            {
                const float *q_ptr = Q + static_cast<size_t>(q_pos) * q_stride +
                                     static_cast<size_t>(h) * head_dim;
                float *out = O + static_cast<size_t>(q_pos) * q_stride +
                             static_cast<size_t>(h) * head_dim;

                // Compute scores
                std::vector<float> scores(kv_len);
                for (int k = 0; k < kv_len; ++k)
                {
                    const float *k_ptr = K + static_cast<size_t>(k) * kv_stride +
                                         static_cast<size_t>(kv_h) * head_dim;
                    bool masked = causal && k > q_pos;
                    if (masked)
                    {
                        scores[k] = -std::numeric_limits<float>::infinity();
                    }
                    else
                    {
                        float dot = 0.0f;
                        for (int d = 0; d < head_dim; ++d)
                            dot += q_ptr[d] * k_ptr[d];
                        scores[k] = dot * scale;
                    }
                }

                // Softmax
                float max_s = *std::max_element(scores.begin(), scores.end());
                float sum_exp = 0.0f;
                for (int k = 0; k < kv_len; ++k)
                {
                    if (std::isfinite(scores[k]))
                    {
                        scores[k] = std::exp(scores[k] - max_s);
                        sum_exp += scores[k];
                    }
                    else
                    {
                        scores[k] = 0.0f;
                    }
                }
                if (sum_exp > 0.0f)
                {
                    for (int k = 0; k < kv_len; ++k)
                        scores[k] /= sum_exp;
                }

                // Weighted sum of V
                for (int d = 0; d < head_dim; ++d)
                    out[d] = 0.0f;
                for (int k = 0; k < kv_len; ++k)
                {
                    if (scores[k] == 0.0f)
                        continue;
                    const float *v_ptr = V + static_cast<size_t>(k) * kv_stride +
                                         static_cast<size_t>(kv_h) * head_dim;
                    for (int d = 0; d < head_dim; ++d)
                        out[d] += scores[k] * v_ptr[d];
                }
            }
        }
    }
} // namespace ref

// ─────────────────────────────────────────────────────────────────────
// Helper: cosine similarity between two float vectors
// ─────────────────────────────────────────────────────────────────────
static double cosine_similarity(const float *a, const float *b, size_t n)
{
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    if (na < 1e-30 || nb < 1e-30)
        return 0.0;
    return dot / std::sqrt(na * nb);
}

// ─────────────────────────────────────────────────────────────────────
// Test fixture: head_dim=128 (Qwen3-like), with GQA
// ─────────────────────────────────────────────────────────────────────
class Test__SplitTQ_Attention_HeadDim128 : public ::testing::Test
{
protected:
    // Qwen3-0.6B-like dimensions: 16 query heads, 8 KV heads, head_dim=128
    static constexpr int HEAD_DIM = 128;
    static constexpr int N_HEADS = 16;
    static constexpr int N_KV_HEADS = 8;
    static constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM; // 1024

    MPIContext mpi_ctx_{0, 1, MPI_COMM_WORLD};
    std::unique_ptr<TurboQuantContext> turboquant_ctx_;

    void SetUp() override
    {
        turboquant_ctx_ = std::make_unique<TurboQuantContext>(
            HEAD_DIM, /*rotation_seed=*/42, /*projection_seed=*/42);
    }

    /// Create a random FP32 tensor with shape [num_tokens, dim]
    static std::shared_ptr<FP32Tensor> makeRandomFP32(int num_tokens, int dim, unsigned seed)
    {
        auto t = std::make_shared<FP32Tensor>(std::vector<size_t>{
            static_cast<size_t>(num_tokens), static_cast<size_t>(dim)});
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        float *d = t->mutable_data();
        for (size_t i = 0; i < t->numel(); ++i)
            d[i] = dist(rng);
        return t;
    }

    /// Quantize FP32 KV tensor to TQ4 (for V)
    std::shared_ptr<TQ4Tensor> quantizeTQ4(const FP32Tensor &src)
    {
        return TQ4Tensor::quantize_from_fp32(
            src.data(), src.shape(), HEAD_DIM, *turboquant_ctx_);
    }

    /// Quantize FP32 KV tensor to TQ8 (for K)
    std::shared_ptr<TQ8Tensor> quantizeTQ8(const FP32Tensor &src)
    {
        return TQ8Tensor::quantize_from_fp32(
            src.data(), src.shape(), HEAD_DIM, *turboquant_ctx_);
    }

    /// Compute per-row cosine similarity between original FP32 and round-tripped data
    struct CosineStats
    {
        double avg;
        double min;
        double max;
    };

    CosineStats computePerRowCosine(const float *orig, const float *reco, int n_rows, int dim)
    {
        double sum = 0.0, worst = 1.0, best = -1.0;
        for (int r = 0; r < n_rows; ++r)
        {
            double cos = cosine_similarity(orig + r * dim, reco + r * dim, dim);
            sum += cos;
            worst = std::min(worst, cos);
            best = std::max(best, cos);
        }
        return {sum / n_rows, worst, best};
    }
};

// ─────────────────────────────────────────────────────────────────────
// Test 1: Split TQ round-trip quality with head_dim=128
// K: FP32 → TQ8Block_128 → FP32  (tighter thresholds)
// V: FP32 → TQ4Block_128 → FP32
// ─────────────────────────────────────────────────────────────────────
TEST_F(Test__SplitTQ_Attention_HeadDim128, SplitTQ_RoundTrip_Quality_HeadDim128)
{
    constexpr int N_TOKENS = 16;

    auto fp32_k = makeRandomFP32(N_TOKENS, KV_DIM, 1000);
    auto fp32_v = makeRandomFP32(N_TOKENS, KV_DIM, 2000);

    // K: TQ8 round-trip (256 centroids → higher fidelity)
    auto tq8_k = quantizeTQ8(*fp32_k);
    ASSERT_EQ(tq8_k->block_bytes(), sizeof(TQ8Block_128))
        << "Expected TQ8Block_128 for head_dim=128";

    std::vector<float> reco_k(N_TOKENS * KV_DIM);
    tq8_k->dequantize_to_fp32(reco_k.data(), *turboquant_ctx_);

    // V: TQ4 round-trip (16 centroids → lower fidelity, acceptable for V)
    auto tq4_v = quantizeTQ4(*fp32_v);
    ASSERT_EQ(tq4_v->block_bytes(), sizeof(TQ4Block_128))
        << "Expected TQ4Block_128 for head_dim=128";

    std::vector<float> reco_v(N_TOKENS * KV_DIM);
    tq4_v->dequantize_to_fp32(reco_v.data(), *turboquant_ctx_);

    auto stats_k = computePerRowCosine(fp32_k->data(), reco_k.data(), N_TOKENS, KV_DIM);
    auto stats_v = computePerRowCosine(fp32_v->data(), reco_v.data(), N_TOKENS, KV_DIM);

    std::cout << "Split TQ round-trip (head_dim=128) K (TQ8): avg_cosine=" << stats_k.avg
              << " min_cosine=" << stats_k.min << std::endl;
    std::cout << "Split TQ round-trip (head_dim=128) V (TQ4): avg_cosine=" << stats_v.avg
              << " min_cosine=" << stats_v.min << std::endl;

    // TQ8 K: tighter thresholds (256 centroids)
    EXPECT_GT(stats_k.avg, 0.999) << "K (TQ8) round-trip avg cosine too low for head_dim=128";
    EXPECT_GT(stats_k.min, 0.99) << "K (TQ8) round-trip worst cosine too low for head_dim=128";
    // TQ4 V: relaxed thresholds (16 centroids)
    EXPECT_GT(stats_v.avg, 0.99) << "V (TQ4) round-trip avg cosine too low for head_dim=128";
    EXPECT_GT(stats_v.min, 0.95) << "V (TQ4) round-trip worst cosine too low for head_dim=128";
    // TQ8 should be better than TQ4
    EXPECT_GT(stats_k.avg, stats_v.avg) << "TQ8 K should have higher quality than TQ4 V";
}

// ─────────────────────────────────────────────────────────────────────
// Test 2: KV cache round-trip with split TQ8-K / TQ4-V at head_dim=128
// FP32 → TQ8/TQ4 → cache append → cache gather → dequant → FP32
// ─────────────────────────────────────────────────────────────────────
TEST_F(Test__SplitTQ_Attention_HeadDim128, SplitTQ_CacheRoundTrip_CosineSimilarity_HeadDim128)
{
    constexpr int MAX_SEQ = 32;
    constexpr int N_TOKENS = 16;

    CPURingKVCacheTQ cache(mpi_ctx_, /*n_layers=*/1, /*batch_size=*/1,
                           MAX_SEQ, N_KV_HEADS, HEAD_DIM, DeviceId::cpu());

    auto fp32_k = makeRandomFP32(N_TOKENS, KV_DIM, 3000);
    auto fp32_v = makeRandomFP32(N_TOKENS, KV_DIM, 4000);
    auto tq8_k = quantizeTQ8(*fp32_k);
    auto tq4_v = quantizeTQ4(*fp32_v);

    ASSERT_TRUE(cache.append_kv(0, 0, tq8_k.get(), tq4_v.get(), N_TOKENS));

    // Gather from cache — K as TQ8, V as TQ4
    auto out_k = std::make_shared<TQ8Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), KV_DIM}, HEAD_DIM);
    auto out_v = std::make_shared<TQ4Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), KV_DIM}, HEAD_DIM);
    std::vector<int> kv_lens;
    int max_kv = cache.gather_kv_batched(0, 1, out_k.get(), out_v.get(), kv_lens);
    ASSERT_EQ(max_kv, N_TOKENS);

    // Dequantize gathered tensors
    out_k->set_turboquant_context(turboquant_ctx_.get());
    out_v->set_turboquant_context(turboquant_ctx_.get());
    std::vector<float> gathered_k(N_TOKENS * KV_DIM);
    std::vector<float> gathered_v(N_TOKENS * KV_DIM);
    for (int r = 0; r < N_TOKENS; ++r)
    {
        out_k->to_fp32_row(r, gathered_k.data() + r * KV_DIM);
        out_v->to_fp32_row(r, gathered_v.data() + r * KV_DIM);
    }

    auto stats_k = computePerRowCosine(fp32_k->data(), gathered_k.data(), N_TOKENS, KV_DIM);
    auto stats_v = computePerRowCosine(fp32_v->data(), gathered_v.data(), N_TOKENS, KV_DIM);

    std::cout << "Cache round-trip (head_dim=128) K (TQ8): avg_cosine=" << stats_k.avg
              << " min_cosine=" << stats_k.min << std::endl;
    std::cout << "Cache round-trip (head_dim=128) V (TQ4): avg_cosine=" << stats_v.avg
              << " min_cosine=" << stats_v.min << std::endl;

    // TQ8 K: tighter thresholds
    EXPECT_GT(stats_k.avg, 0.999) << "Cache K (TQ8) avg cosine too low for head_dim=128";
    EXPECT_GT(stats_k.min, 0.99) << "Cache K (TQ8) worst cosine too low for head_dim=128";
    // TQ4 V: relaxed thresholds
    EXPECT_GT(stats_v.avg, 0.99) << "Cache V (TQ4) avg cosine too low for head_dim=128";
    EXPECT_GT(stats_v.min, 0.95) << "Cache V (TQ4) worst cosine too low for head_dim=128";
}

// ─────────────────────────────────────────────────────────────────────
// Test 3: Split dequant correctness at head_dim=128
// Compare per-tensor dequant vs per-row dequant helpers for K (TQ8) and V (TQ4).
// ─────────────────────────────────────────────────────────────────────
TEST_F(Test__SplitTQ_Attention_HeadDim128, SplitTQ_DequantizeRoutes_MatchesPerBlock_HeadDim128)
{
    constexpr int N_TOKENS = 10;

    auto fp32_k = makeRandomFP32(N_TOKENS, KV_DIM, 5000);
    auto fp32_v = makeRandomFP32(N_TOKENS, KV_DIM, 6000);
    auto tq8_k = quantizeTQ8(*fp32_k);
    auto tq4_v = quantizeTQ4(*fp32_v);

    // Method A: Per-tensor dequantize (TQ8Tensor/TQ4Tensor::dequantize_to_fp32)
    std::vector<float> dequant_a_k(N_TOKENS * KV_DIM);
    std::vector<float> dequant_a_v(N_TOKENS * KV_DIM);
    tq8_k->dequantize_to_fp32(dequant_a_k.data(), *turboquant_ctx_);
    tq4_v->dequantize_to_fp32(dequant_a_v.data(), *turboquant_ctx_);

    // Method B: Per-row dequant using low-level helpers
    // K (TQ8): turboquant_dequantize_row_tq8 per row
    std::vector<float> dequant_b_k(N_TOKENS * KV_DIM, 0.0f);
    alignas(64) float scratch_k[HEAD_DIM];
    const size_t k_row_bytes = tq8_k->blocks_per_row() * tq8_k->block_bytes();
    for (int r = 0; r < N_TOKENS; ++r)
    {
        const auto *row_blocks = reinterpret_cast<const TQ8Block_128 *>(
            tq8_k->typed_data() + r * k_row_bytes);
        turboquant_dequantize_row_tq8<128>(
            row_blocks, *turboquant_ctx_,
            dequant_b_k.data() + r * KV_DIM,
            N_KV_HEADS, scratch_k);
    }

    // V (TQ4): turboquant_dequantize_v_rows (bulk helper)
    std::vector<float> dequant_b_v(N_TOKENS * KV_DIM, 0.0f);
    turboquant_dequantize_v_rows(
        tq4_v->typed_data(),
        *turboquant_ctx_,
        dequant_b_v.data(),
        /*from_row=*/0, /*to_row=*/N_TOKENS,
        HEAD_DIM, N_KV_HEADS,
        tq4_v->blocks_per_row() * tq4_v->block_bytes(),
        tq4_v->block_bytes());

    // They should be IDENTICAL (both use the same underlying dequant algorithms)
    for (int i = 0; i < N_TOKENS * KV_DIM; ++i)
    {
        ASSERT_EQ(dequant_a_k[i], dequant_b_k[i])
            << "K (TQ8) dequant mismatch at element " << i
            << " (row=" << i / KV_DIM << ", col=" << i % KV_DIM << ")";
        ASSERT_EQ(dequant_a_v[i], dequant_b_v[i])
            << "V (TQ4) dequant mismatch at element " << i
            << " (row=" << i / KV_DIM << ", col=" << i % KV_DIM << ")";
    }
}

// ─────────────────────────────────────────────────────────────────────
// Test 4: Incremental dequant (the hot path optimization)
// K (TQ8): Dequant rows 0-5, then rows 5-10, vs all 0-10 at once.
// V (TQ4): Same incremental check using turboquant_dequantize_v_rows.
// ─────────────────────────────────────────────────────────────────────
TEST_F(Test__SplitTQ_Attention_HeadDim128, SplitTQ_IncrementalDequant_HeadDim128)
{
    constexpr int N_TOKENS = 10;

    auto fp32_k = makeRandomFP32(N_TOKENS, KV_DIM, 7000);
    auto fp32_v = makeRandomFP32(N_TOKENS, KV_DIM, 7100);
    auto tq8_k = quantizeTQ8(*fp32_k);
    auto tq4_v = quantizeTQ4(*fp32_v);

    // === K (TQ8): Full dequant all rows at once ===
    std::vector<float> full_k(N_TOKENS * KV_DIM, 0.0f);
    tq8_k->dequantize_to_fp32(full_k.data(), *turboquant_ctx_);

    // K: Incremental (rows 0-5, then 5-10) via per-row helper
    std::vector<float> incr_k(N_TOKENS * KV_DIM, 0.0f);
    alignas(64) float scratch_k[HEAD_DIM];
    const size_t k_row_bytes = tq8_k->blocks_per_row() * tq8_k->block_bytes();
    for (int r = 0; r < 5; ++r)
    {
        const auto *row_blocks = reinterpret_cast<const TQ8Block_128 *>(
            tq8_k->typed_data() + r * k_row_bytes);
        turboquant_dequantize_row_tq8<128>(
            row_blocks, *turboquant_ctx_,
            incr_k.data() + r * KV_DIM, N_KV_HEADS, scratch_k);
    }
    for (int r = 5; r < N_TOKENS; ++r)
    {
        const auto *row_blocks = reinterpret_cast<const TQ8Block_128 *>(
            tq8_k->typed_data() + r * k_row_bytes);
        turboquant_dequantize_row_tq8<128>(
            row_blocks, *turboquant_ctx_,
            incr_k.data() + r * KV_DIM, N_KV_HEADS, scratch_k);
    }

    for (int i = 0; i < N_TOKENS * KV_DIM; ++i)
    {
        ASSERT_EQ(full_k[i], incr_k[i])
            << "K (TQ8) incremental dequant mismatch at element " << i;
    }

    // === V (TQ4): Full dequant all rows at once ===
    const size_t v_row_bytes = tq4_v->blocks_per_row() * tq4_v->block_bytes();
    const size_t v_block_bytes = tq4_v->block_bytes();

    std::vector<float> full_v(N_TOKENS * KV_DIM, 0.0f);
    turboquant_dequantize_v_rows(
        tq4_v->typed_data(), *turboquant_ctx_,
        full_v.data(), 0, N_TOKENS,
        HEAD_DIM, N_KV_HEADS, v_row_bytes, v_block_bytes);

    // V: Incremental (rows 0-5, then 5-10)
    std::vector<float> incr_v(N_TOKENS * KV_DIM, 0.0f);
    turboquant_dequantize_v_rows(
        tq4_v->typed_data(), *turboquant_ctx_,
        incr_v.data(), 0, 5,
        HEAD_DIM, N_KV_HEADS, v_row_bytes, v_block_bytes);
    turboquant_dequantize_v_rows(
        tq4_v->typed_data(), *turboquant_ctx_,
        incr_v.data(), 5, N_TOKENS,
        HEAD_DIM, N_KV_HEADS, v_row_bytes, v_block_bytes);

    for (int i = 0; i < N_TOKENS * KV_DIM; ++i)
    {
        ASSERT_EQ(full_v[i], incr_v[i])
            << "V (TQ4) incremental dequant mismatch at element " << i;
    }
}

// ─────────────────────────────────────────────────────────────────────
// Test 5: Full attention pipeline with split TQ8-K / TQ4-V dequant (head_dim=128)
// This is the critical test: does attention(Q, dequant(TQ8(K)), dequant(TQ4(V)))
// produce results close to attention(Q, K, V)?
// ─────────────────────────────────────────────────────────────────────
TEST_F(Test__SplitTQ_Attention_HeadDim128, AttentionWithSplitTQKV_DecodePath_HeadDim128)
{
    constexpr int KV_LEN = 10;            // Prior context length
    constexpr int SEQ_LEN = 1;            // Decode: 1 query token
    const int Q_DIM = N_HEADS * HEAD_DIM; // 2048

    // Create random Q, K, V
    auto fp32_q = makeRandomFP32(SEQ_LEN, Q_DIM, 8000);
    auto fp32_k = makeRandomFP32(KV_LEN, KV_DIM, 8100);
    auto fp32_v = makeRandomFP32(KV_LEN, KV_DIM, 8200);

    // Ground truth: attention with original FP32 K/V
    std::vector<float> ref_output(SEQ_LEN * Q_DIM, 0.0f);
    ref::attention(fp32_q->data(), fp32_k->data(), fp32_v->data(),
                   ref_output.data(),
                   SEQ_LEN, KV_LEN, N_HEADS, N_KV_HEADS, HEAD_DIM,
                   /*causal=*/false);

    // Split TQ path: K → TQ8, V → TQ4, then dequant → attention
    auto tq8_k = quantizeTQ8(*fp32_k);
    auto tq4_v = quantizeTQ4(*fp32_v);

    std::vector<float> dequant_k(KV_LEN * KV_DIM);
    std::vector<float> dequant_v(KV_LEN * KV_DIM);
    tq8_k->dequantize_to_fp32(dequant_k.data(), *turboquant_ctx_);
    tq4_v->dequantize_to_fp32(dequant_v.data(), *turboquant_ctx_);

    // Check dequant quality
    auto k_stats = computePerRowCosine(fp32_k->data(), dequant_k.data(), KV_LEN, KV_DIM);
    auto v_stats = computePerRowCosine(fp32_v->data(), dequant_v.data(), KV_LEN, KV_DIM);
    std::cout << "Dequant K (TQ8) quality: avg_cos=" << k_stats.avg
              << " min_cos=" << k_stats.min << std::endl;
    std::cout << "Dequant V (TQ4) quality: avg_cos=" << v_stats.avg
              << " min_cos=" << v_stats.min << std::endl;

    // Attention with split-TQ dequanted K/V (scalar reference)
    std::vector<float> split_output(SEQ_LEN * Q_DIM, 0.0f);
    ref::attention(fp32_q->data(), dequant_k.data(), dequant_v.data(),
                   split_output.data(),
                   SEQ_LEN, KV_LEN, N_HEADS, N_KV_HEADS, HEAD_DIM,
                   /*causal=*/false);

    // Compare attention outputs
    double attn_cosine = cosine_similarity(ref_output.data(), split_output.data(),
                                           SEQ_LEN * Q_DIM);
    std::cout << "Attention output cosine (FP32 vs SplitTQ-dequant, head_dim=128): "
              << attn_cosine << std::endl;

    // Per-head analysis
    double min_head_cosine = 1.0;
    int worst_head = -1;
    for (int h = 0; h < N_HEADS; ++h)
    {
        double hcos = cosine_similarity(
            ref_output.data() + h * HEAD_DIM,
            split_output.data() + h * HEAD_DIM,
            HEAD_DIM);
        if (hcos < min_head_cosine)
        {
            min_head_cosine = hcos;
            worst_head = h;
        }
    }
    std::cout << "  Per-head worst cosine: head " << worst_head
              << " = " << min_head_cosine << std::endl;

    EXPECT_GT(attn_cosine, 0.95)
        << "Attention output cosine too low for head_dim=128 split TQ path";
    EXPECT_GT(min_head_cosine, 0.90)
        << "Per-head worst cosine too low for head_dim=128 split TQ path";
}

// ─────────────────────────────────────────────────────────────────────
// Test 6: Full attention pipeline using the real kernel (not scalar ref)
// ─────────────────────────────────────────────────────────────────────
TEST_F(Test__SplitTQ_Attention_HeadDim128, AttentionKernel_FP32vsSplitTQ_HeadDim128)
{
    constexpr int KV_LEN = 10;
    constexpr int SEQ_LEN = 1;
    const int Q_DIM = N_HEADS * HEAD_DIM;

    auto fp32_q = makeRandomFP32(SEQ_LEN, Q_DIM, 9000);
    auto fp32_k = makeRandomFP32(KV_LEN, KV_DIM, 9100);
    auto fp32_v = makeRandomFP32(KV_LEN, KV_DIM, 9200);

    // Create attention kernel via FP32Tensor::createAttention()
    auto kernel = fp32_q->createAttention();
    ASSERT_NE(kernel, nullptr) << "Failed to create attention kernel";

    // Workspace tensors
    auto ws_scores = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(N_HEADS * SEQ_LEN * KV_LEN)});
    auto ws_context = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(SEQ_LEN * Q_DIM)});

    // Ground truth: attention with FP32 K/V
    auto output_fp32 = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(SEQ_LEN), static_cast<size_t>(Q_DIM)});
    ASSERT_TRUE(kernel->compute_tensor(
        fp32_q.get(), fp32_k.get(), fp32_v.get(), output_fp32.get(),
        /*batch_size=*/1, SEQ_LEN, KV_LEN,
        N_HEADS, N_KV_HEADS, HEAD_DIM,
        /*causal=*/false, /*window_size=*/-1,
        ws_scores.get(), /*workspace_mask=*/nullptr,
        /*mpi_ctx=*/nullptr, /*device_idx=*/-1));

    // Split TQ path: K → TQ8, V → TQ4, dequant → attention
    auto tq8_k = quantizeTQ8(*fp32_k);
    auto tq4_v = quantizeTQ4(*fp32_v);

    std::vector<float> dequant_k(KV_LEN * KV_DIM);
    std::vector<float> dequant_v(KV_LEN * KV_DIM);
    tq8_k->dequantize_to_fp32(dequant_k.data(), *turboquant_ctx_);
    tq4_v->dequantize_to_fp32(dequant_v.data(), *turboquant_ctx_);

    auto dequant_k_tensor = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(KV_LEN), static_cast<size_t>(KV_DIM)});
    auto dequant_v_tensor = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(KV_LEN), static_cast<size_t>(KV_DIM)});
    std::memcpy(dequant_k_tensor->mutable_data(), dequant_k.data(),
                KV_LEN * KV_DIM * sizeof(float));
    std::memcpy(dequant_v_tensor->mutable_data(), dequant_v.data(),
                KV_LEN * KV_DIM * sizeof(float));

    auto output_split = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(SEQ_LEN), static_cast<size_t>(Q_DIM)});
    ASSERT_TRUE(kernel->compute_tensor(
        fp32_q.get(), dequant_k_tensor.get(), dequant_v_tensor.get(), output_split.get(),
        1, SEQ_LEN, KV_LEN,
        N_HEADS, N_KV_HEADS, HEAD_DIM,
        false, -1,
        ws_scores.get(), nullptr,
        nullptr, -1));

    // Compare outputs
    double attn_cosine = cosine_similarity(
        output_fp32->data(), output_split->data(), SEQ_LEN * Q_DIM);
    std::cout << "Real kernel attention cosine (FP32 vs SplitTQ, head_dim=128): "
              << attn_cosine << std::endl;

    // Per-head analysis
    double min_head_cosine = 1.0;
    int worst_head = -1;
    for (int h = 0; h < N_HEADS; ++h)
    {
        double hcos = cosine_similarity(
            output_fp32->data() + h * HEAD_DIM,
            output_split->data() + h * HEAD_DIM,
            HEAD_DIM);
        if (hcos < min_head_cosine)
        {
            min_head_cosine = hcos;
            worst_head = h;
        }
    }
    std::cout << "  Per-head worst cosine: head " << worst_head
              << " = " << min_head_cosine << std::endl;

    EXPECT_GT(attn_cosine, 0.95)
        << "Real kernel: attention output cosine too low for head_dim=128";
    EXPECT_GT(min_head_cosine, 0.90)
        << "Real kernel: per-head worst cosine too low for head_dim=128";
}

// ─────────────────────────────────────────────────────────────────────
// Test 7: Full pipeline — cache append → gather → split dequant → attention
// This mirrors exactly what happens in the Qwen3 inference pipeline
// with split TQ8-K / TQ4-V precision.
// ─────────────────────────────────────────────────────────────────────
TEST_F(Test__SplitTQ_Attention_HeadDim128, FullPipeline_CacheToAttention_HeadDim128)
{
    constexpr int MAX_SEQ = 64;
    constexpr int KV_LEN = 10;
    constexpr int SEQ_LEN = 1;
    const int Q_DIM = N_HEADS * HEAD_DIM;

    // Create KV cache (split TQ8-K / TQ4-V)
    CPURingKVCacheTQ cache(mpi_ctx_, /*n_layers=*/1, /*batch_size=*/1,
                           MAX_SEQ, N_KV_HEADS, HEAD_DIM, DeviceId::cpu());

    // Create random Q (decode token) and K/V (context)
    auto fp32_q = makeRandomFP32(SEQ_LEN, Q_DIM, 10000);
    auto fp32_k = makeRandomFP32(KV_LEN, KV_DIM, 10100);
    auto fp32_v = makeRandomFP32(KV_LEN, KV_DIM, 10200);

    // ═══ GROUND TRUTH: attention with original FP32 K/V ═══
    std::vector<float> ref_output(SEQ_LEN * Q_DIM, 0.0f);
    ref::attention(fp32_q->data(), fp32_k->data(), fp32_v->data(),
                   ref_output.data(),
                   SEQ_LEN, KV_LEN, N_HEADS, N_KV_HEADS, HEAD_DIM,
                   /*causal=*/false);

    // ═══ SPLIT TQ PIPELINE: TQ8(K)/TQ4(V) → cache → gather → dequant → attention ═══
    // Step 1: Quantize K to TQ8, V to TQ4, and append to cache
    auto tq8_k = quantizeTQ8(*fp32_k);
    auto tq4_v = quantizeTQ4(*fp32_v);
    ASSERT_TRUE(cache.append_kv(0, 0, tq8_k.get(), tq4_v.get(), KV_LEN));
    ASSERT_EQ(cache.ring_size(0, 0), KV_LEN);

    // Step 2: Read K/V from cache (same as AttentionComputeStage does)
    ITensor *cache_k = cache.get_k(0, 0);
    ITensor *cache_v = cache.get_v(0, 0);
    ASSERT_NE(cache_k, nullptr);
    ASSERT_NE(cache_v, nullptr);

    auto *cache_k_tq8 = dynamic_cast<TQ8Tensor *>(cache_k);
    auto *cache_v_tq4 = dynamic_cast<TQ4Tensor *>(cache_v);
    ASSERT_NE(cache_k_tq8, nullptr) << "Cache K should be TQ8Tensor";
    ASSERT_NE(cache_v_tq4, nullptr) << "Cache V should be TQ4Tensor";

    // Step 3: Dequantize K (TQ8) and V (TQ4) separately via per-row helpers
    // (cache tensor shape is [MAX_SEQ, KV_DIM]; only first KV_LEN rows valid)
    std::vector<float> dequant_k(KV_LEN * KV_DIM, 0.0f);
    std::vector<float> dequant_v(KV_LEN * KV_DIM, 0.0f);

    cache_k_tq8->set_turboquant_context(turboquant_ctx_.get());
    cache_v_tq4->set_turboquant_context(turboquant_ctx_.get());
    for (int r = 0; r < KV_LEN; ++r)
    {
        cache_k_tq8->to_fp32_row(r, dequant_k.data() + r * KV_DIM);
        cache_v_tq4->to_fp32_row(r, dequant_v.data() + r * KV_DIM);
    }

    // Check dequant quality
    auto k_stats = computePerRowCosine(fp32_k->data(), dequant_k.data(), KV_LEN, KV_DIM);
    auto v_stats = computePerRowCosine(fp32_v->data(), dequant_v.data(), KV_LEN, KV_DIM);
    std::cout << "Full pipeline dequant K (TQ8): avg_cos=" << k_stats.avg
              << " min_cos=" << k_stats.min << std::endl;
    std::cout << "Full pipeline dequant V (TQ4): avg_cos=" << v_stats.avg
              << " min_cos=" << v_stats.min << std::endl;

    EXPECT_GT(k_stats.avg, 0.999) << "Pipeline K (TQ8) dequant quality too low";
    EXPECT_GT(v_stats.avg, 0.99) << "Pipeline V (TQ4) dequant quality too low";

    // Step 4: Run attention with dequanted K/V
    std::vector<float> split_output(SEQ_LEN * Q_DIM, 0.0f);
    ref::attention(fp32_q->data(), dequant_k.data(), dequant_v.data(),
                   split_output.data(),
                   SEQ_LEN, KV_LEN, N_HEADS, N_KV_HEADS, HEAD_DIM,
                   /*causal=*/false);

    // ═══ COMPARE ═══
    double attn_cosine = cosine_similarity(
        ref_output.data(), split_output.data(), SEQ_LEN * Q_DIM);
    std::cout << "Full pipeline attention cosine (head_dim=128): "
              << attn_cosine << std::endl;

    double min_head_cosine = 1.0;
    int worst_head = -1;
    for (int h = 0; h < N_HEADS; ++h)
    {
        double hcos = cosine_similarity(
            ref_output.data() + h * HEAD_DIM,
            split_output.data() + h * HEAD_DIM,
            HEAD_DIM);
        std::cout << "  Head " << h << ": cosine=" << hcos << std::endl;
        if (hcos < min_head_cosine)
        {
            min_head_cosine = hcos;
            worst_head = h;
        }
    }
    std::cout << "  Worst head: " << worst_head
              << " cosine=" << min_head_cosine << std::endl;

    EXPECT_GT(attn_cosine, 0.95)
        << "Full pipeline attention cosine too low for head_dim=128";
    EXPECT_GT(min_head_cosine, 0.90)
        << "Full pipeline per-head worst cosine too low for head_dim=128";
}

// ─────────────────────────────────────────────────────────────────────
// Test 9: PIPELINE-EXACT decode path with for_layer() derivation
//
// This test exercises the EXACT same code path as the Qwen3 pipeline
// with split TQ8-K / TQ4-V precision:
//   KVCacheAppendStage: root.for_layer(layer_idx) → quantize_from_fp32
//                       K: TQ8Tensor::quantize_from_fp32
//                       V: TQ4Tensor::quantize_from_fp32
//   Cache memcpy:       TQ8/TQ4 bytes → ring buffer → get_k()/get_v()
//   AttentionCompute:   root.for_layer(layer_idx) → separate K (TQ8) / V (TQ4) dequant
//   Kernel dispatch:    compute_decode() (kv_len > seq_len)
//
// Previous tests used:
//   - Root context (no for_layer()): can't detect derivation bugs
//   - seq_len == kv_len: routes to compute() (prefill), not compute_decode()
// ─────────────────────────────────────────────────────────────────────
TEST_F(Test__SplitTQ_Attention_HeadDim128, PipelineExact_ForLayer_ComputeDecode_HeadDim128)
{
    constexpr int MAX_SEQ = 256;
    constexpr int PREFILL_LEN = 50;         // Tokens from prefill
    constexpr int DECODE_IDX = PREFILL_LEN; // First decode position
    constexpr int SEQ_LEN = 1;              // Decode: 1 query token
    constexpr int KV_LEN = PREFILL_LEN + 1; // After appending decode token
    const int Q_DIM = N_HEADS * HEAD_DIM;
    constexpr int LAYER_IDX = 5; // Non-zero layer to exercise for_layer()

    // Create random data
    auto fp32_q = makeRandomFP32(SEQ_LEN, Q_DIM, 20000);
    auto fp32_k_prefill = makeRandomFP32(PREFILL_LEN, KV_DIM, 20100);
    auto fp32_v_prefill = makeRandomFP32(PREFILL_LEN, KV_DIM, 20200);
    auto fp32_k_decode = makeRandomFP32(SEQ_LEN, KV_DIM, 20300);
    auto fp32_v_decode = makeRandomFP32(SEQ_LEN, KV_DIM, 20400);

    // ═══ GROUND TRUTH: attention with all FP32 K/V concatenated ═══
    // Concatenate prefill + decode K/V
    std::vector<float> full_k(KV_LEN * KV_DIM);
    std::vector<float> full_v(KV_LEN * KV_DIM);
    std::memcpy(full_k.data(), fp32_k_prefill->data(),
                PREFILL_LEN * KV_DIM * sizeof(float));
    std::memcpy(full_k.data() + PREFILL_LEN * KV_DIM, fp32_k_decode->data(),
                SEQ_LEN * KV_DIM * sizeof(float));
    std::memcpy(full_v.data(), fp32_v_prefill->data(),
                PREFILL_LEN * KV_DIM * sizeof(float));
    std::memcpy(full_v.data() + PREFILL_LEN * KV_DIM, fp32_v_decode->data(),
                SEQ_LEN * KV_DIM * sizeof(float));

    std::vector<float> ref_output(SEQ_LEN * Q_DIM, 0.0f);
    ref::attention(fp32_q->data(), full_k.data(), full_v.data(),
                   ref_output.data(),
                   SEQ_LEN, KV_LEN, N_HEADS, N_KV_HEADS, HEAD_DIM,
                   /*causal=*/false);

    // ═══ SPLIT TQ PIPELINE (exact pipeline path) ═══
    CPURingKVCacheTQ cache(mpi_ctx_, /*n_layers=*/8, /*batch_size=*/1,
                           MAX_SEQ, N_KV_HEADS, HEAD_DIM, DeviceId::cpu());

    // --- STEP 1: Quantize prefill K/V using DERIVED context (like KVCacheAppendStage) ---
    // Pipeline: root.for_layer(layer_idx) → passed to quantize_from_fp32
    const auto &layer_ctx = turboquant_ctx_->for_layer(LAYER_IDX);

    auto tq8_k_prefill = TQ8Tensor::quantize_from_fp32(
        fp32_k_prefill->data(), fp32_k_prefill->shape(), HEAD_DIM, layer_ctx);
    auto tq4_v_prefill = TQ4Tensor::quantize_from_fp32(
        fp32_v_prefill->data(), fp32_v_prefill->shape(), HEAD_DIM, layer_ctx);

    // Append PREFILL to cache
    ASSERT_TRUE(cache.append_kv(LAYER_IDX, 0, tq8_k_prefill.get(), tq4_v_prefill.get(), PREFILL_LEN));
    ASSERT_EQ(cache.ring_size(LAYER_IDX, 0), PREFILL_LEN);

    // --- STEP 2: Quantize decode token and append ---
    auto tq8_k_decode = TQ8Tensor::quantize_from_fp32(
        fp32_k_decode->data(), fp32_k_decode->shape(), HEAD_DIM, layer_ctx);
    auto tq4_v_decode = TQ4Tensor::quantize_from_fp32(
        fp32_v_decode->data(), fp32_v_decode->shape(), HEAD_DIM, layer_ctx);

    ASSERT_TRUE(cache.append_kv(LAYER_IDX, 0, tq8_k_decode.get(), tq4_v_decode.get(), SEQ_LEN));
    ASSERT_EQ(cache.ring_size(LAYER_IDX, 0), KV_LEN);

    // --- STEP 3: Read K/V from cache (same as AttentionComputeStage) ---
    ITensor *cache_k = cache.get_k(LAYER_IDX, 0);
    ITensor *cache_v = cache.get_v(LAYER_IDX, 0);
    ASSERT_NE(cache_k, nullptr);
    ASSERT_NE(cache_v, nullptr);

    auto *cache_k_tq8 = dynamic_cast<TQ8Tensor *>(cache_k);
    auto *cache_v_tq4 = dynamic_cast<TQ4Tensor *>(cache_v);
    ASSERT_NE(cache_k_tq8, nullptr);
    ASSERT_NE(cache_v_tq4, nullptr);

    // --- STEP 4: Dequantize K (TQ8) and V (TQ4) separately ---
    // (cache tensor shape is [MAX_SEQ, KV_DIM]; only first KV_LEN rows valid)
    std::vector<float> dequant_k(KV_LEN * KV_DIM, 0.0f);
    std::vector<float> dequant_v(KV_LEN * KV_DIM, 0.0f);

    cache_k_tq8->set_turboquant_context(&layer_ctx);
    cache_v_tq4->set_turboquant_context(&layer_ctx);
    for (int r = 0; r < KV_LEN; ++r)
    {
        cache_k_tq8->to_fp32_row(r, dequant_k.data() + r * KV_DIM);
        cache_v_tq4->to_fp32_row(r, dequant_v.data() + r * KV_DIM);
    }

    // Check dequant quality
    auto k_stats = computePerRowCosine(full_k.data(), dequant_k.data(), KV_LEN, KV_DIM);
    auto v_stats = computePerRowCosine(full_v.data(), dequant_v.data(), KV_LEN, KV_DIM);
    std::cout << "[PipelineExact] Dequant K: avg_cos=" << k_stats.avg
              << " min_cos=" << k_stats.min << std::endl;
    std::cout << "[PipelineExact] Dequant V: avg_cos=" << v_stats.avg
              << " min_cos=" << v_stats.min << std::endl;

    EXPECT_GT(k_stats.avg, 0.999) << "Pipeline dequant K (TQ8) quality too low";
    EXPECT_GT(v_stats.avg, 0.99) << "Pipeline dequant V (TQ4) quality too low";

    // --- STEP 5: Run attention via compute_decode() path ---
    // Create attention kernel
    auto kernel = fp32_q->createAttention();
    ASSERT_NE(kernel, nullptr);

    auto dequant_k_tensor = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(KV_LEN), static_cast<size_t>(KV_DIM)});
    auto dequant_v_tensor = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(KV_LEN), static_cast<size_t>(KV_DIM)});
    std::memcpy(dequant_k_tensor->mutable_data(), dequant_k.data(),
                KV_LEN * KV_DIM * sizeof(float));
    std::memcpy(dequant_v_tensor->mutable_data(), dequant_v.data(),
                KV_LEN * KV_DIM * sizeof(float));

    auto output_split = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(SEQ_LEN), static_cast<size_t>(Q_DIM)});

    // This call routes to compute_decode() because kv_len (51) > seq_len (1)
    ASSERT_TRUE(kernel->compute_tensor(
        fp32_q.get(), dequant_k_tensor.get(), dequant_v_tensor.get(), output_split.get(),
        /*batch_size=*/1, SEQ_LEN, KV_LEN,
        N_HEADS, N_KV_HEADS, HEAD_DIM,
        /*causal=*/false, /*window_size=*/-1,
        /*workspace_scores=*/nullptr, /*workspace_mask=*/nullptr,
        /*mpi_ctx=*/nullptr, /*device_idx=*/-1));

    // ═══ COMPARE ═══
    double attn_cosine = cosine_similarity(
        ref_output.data(), output_split->data(), SEQ_LEN * Q_DIM);
    std::cout << "[PipelineExact] Attention output cosine "
              << "(FP32 ref vs SplitTQ pipeline, compute_decode path): "
              << attn_cosine << std::endl;

    // Per-head analysis
    double min_head_cosine = 1.0;
    int worst_head = -1;
    for (int h = 0; h < N_HEADS; ++h)
    {
        double hcos = cosine_similarity(
            ref_output.data() + h * HEAD_DIM,
            output_split->data() + h * HEAD_DIM,
            HEAD_DIM);
        if (hcos < min_head_cosine)
        {
            min_head_cosine = hcos;
            worst_head = h;
        }
    }
    std::cout << "[PipelineExact] Per-head worst: head " << worst_head
              << " cosine=" << min_head_cosine << std::endl;

    // Thresholds: should be ~0.99 like the prefill tests
    // If this fails with cosine ~0.74 (like the integration test), the bug
    // is in one of: for_layer(), cache memcpy, or compute_decode()
    EXPECT_GT(attn_cosine, 0.95)
        << "Pipeline-exact decode attention cosine too low (head_dim=128)";
    EXPECT_GT(min_head_cosine, 0.90)
        << "Pipeline-exact decode per-head worst cosine too low (head_dim=128)";
}

// ─────────────────────────────────────────────────────────────────────
// Test 8: Compare head_dim=128 vs head_dim=64 quality for the split TQ regime
// ─────────────────────────────────────────────────────────────────────
TEST_F(Test__SplitTQ_Attention_HeadDim128, HeadDim128vs64_QualityComparison)
{
    constexpr int N_TOKENS = 16;

    // ═══ head_dim=128 roundtrip (K=TQ8, V=TQ4) ═══
    auto fp32_128 = makeRandomFP32(N_TOKENS, KV_DIM, 11000);

    auto tq8_128 = quantizeTQ8(*fp32_128);
    std::vector<float> reco_k128(N_TOKENS * KV_DIM);
    tq8_128->dequantize_to_fp32(reco_k128.data(), *turboquant_ctx_);
    auto k_stats_128 = computePerRowCosine(fp32_128->data(), reco_k128.data(), N_TOKENS, KV_DIM);

    auto tq4_128 = quantizeTQ4(*fp32_128);
    std::vector<float> reco_v128(N_TOKENS * KV_DIM);
    tq4_128->dequantize_to_fp32(reco_v128.data(), *turboquant_ctx_);
    auto v_stats_128 = computePerRowCosine(fp32_128->data(), reco_v128.data(), N_TOKENS, KV_DIM);

    // ═══ head_dim=64 roundtrip (K=TQ8, V=TQ4) ═══
    constexpr int HEAD_DIM_64 = 64;
    constexpr int N_KV_HEADS_64 = KV_DIM / HEAD_DIM_64; // 16 heads
    auto ctx_64 = std::make_unique<TurboQuantContext>(HEAD_DIM_64, 42, 42);

    auto tq8_64 = TQ8Tensor::quantize_from_fp32(
        fp32_128->data(), fp32_128->shape(), HEAD_DIM_64, *ctx_64);
    std::vector<float> reco_k64(N_TOKENS * KV_DIM);
    tq8_64->dequantize_to_fp32(reco_k64.data(), *ctx_64);
    auto k_stats_64 = computePerRowCosine(fp32_128->data(), reco_k64.data(), N_TOKENS, KV_DIM);

    auto tq4_64 = TQ4Tensor::quantize_from_fp32(
        fp32_128->data(), fp32_128->shape(), HEAD_DIM_64, *ctx_64);
    std::vector<float> reco_v64(N_TOKENS * KV_DIM);
    tq4_64->dequantize_to_fp32(reco_v64.data(), *ctx_64);
    auto v_stats_64 = computePerRowCosine(fp32_128->data(), reco_v64.data(), N_TOKENS, KV_DIM);

    std::cout << "head_dim=128 K(TQ8): avg_cosine=" << k_stats_128.avg
              << " min=" << k_stats_128.min << std::endl;
    std::cout << "head_dim=128 V(TQ4): avg_cosine=" << v_stats_128.avg
              << " min=" << v_stats_128.min << std::endl;
    std::cout << "head_dim=64  K(TQ8): avg_cosine=" << k_stats_64.avg
              << " min=" << k_stats_64.min << std::endl;
    std::cout << "head_dim=64  V(TQ4): avg_cosine=" << v_stats_64.avg
              << " min=" << v_stats_64.min << std::endl;

    // TQ8 K should be high quality at both dims
    EXPECT_GT(k_stats_128.avg, 0.999) << "TQ8 K head_dim=128 quality unexpectedly low";
    EXPECT_GT(k_stats_64.avg, 0.999) << "TQ8 K head_dim=64 quality unexpectedly low";

    // TQ4 V should be decent at both dims
    EXPECT_GT(v_stats_128.avg, 0.98) << "TQ4 V head_dim=128 quality unexpectedly low";
    EXPECT_GT(v_stats_64.avg, 0.98) << "TQ4 V head_dim=64 quality unexpectedly low";

    // Quality gap between dims should be small (<5% absolute) for both K and V
    EXPECT_LT(std::abs(k_stats_128.avg - k_stats_64.avg), 0.05)
        << "Large K quality gap between head_dim=128 and head_dim=64";
    EXPECT_LT(std::abs(v_stats_128.avg - v_stats_64.avg), 0.05)
        << "Large V quality gap between head_dim=128 and head_dim=64";
}

// ─────────────────────────────────────────────────────────────────────
// Test 10: Pipeline-exact with EXACT Qwen3-0.6B dimensions (split TQ8-K / TQ4-V)
//
// Qwen3-0.6B: n_heads=8, n_kv_heads=2, head_dim=128, d_model=1024
// This ensures n_kv_heads=2 (GQA ratio=4) doesn't trigger edge cases.
// ─────────────────────────────────────────────────────────────────────
TEST_F(Test__SplitTQ_Attention_HeadDim128, PipelineExact_Qwen3_0_6B_Dims)
{
    // EXACT Qwen3-0.6B dimensions
    constexpr int Q3_N_HEADS = 8;
    constexpr int Q3_N_KV_HEADS = 2;
    constexpr int Q3_KV_DIM = Q3_N_KV_HEADS * HEAD_DIM; // 256
    constexpr int Q3_Q_DIM = Q3_N_HEADS * HEAD_DIM;     // 1024

    constexpr int MAX_SEQ = 256;
    constexpr int PREFILL_LEN = 50;
    constexpr int SEQ_LEN = 1;
    constexpr int KV_LEN = PREFILL_LEN + 1;
    constexpr int LAYER_IDX = 5;

    // Create random data with exact Qwen3-0.6B dims
    auto fp32_q = makeRandomFP32(SEQ_LEN, Q3_Q_DIM, 30000);
    auto fp32_k_prefill = makeRandomFP32(PREFILL_LEN, Q3_KV_DIM, 30100);
    auto fp32_v_prefill = makeRandomFP32(PREFILL_LEN, Q3_KV_DIM, 30200);
    auto fp32_k_decode = makeRandomFP32(SEQ_LEN, Q3_KV_DIM, 30300);
    auto fp32_v_decode = makeRandomFP32(SEQ_LEN, Q3_KV_DIM, 30400);

    // Ground truth: concatenated FP32 K/V
    std::vector<float> full_k(KV_LEN * Q3_KV_DIM);
    std::vector<float> full_v(KV_LEN * Q3_KV_DIM);
    std::memcpy(full_k.data(), fp32_k_prefill->data(),
                PREFILL_LEN * Q3_KV_DIM * sizeof(float));
    std::memcpy(full_k.data() + PREFILL_LEN * Q3_KV_DIM, fp32_k_decode->data(),
                SEQ_LEN * Q3_KV_DIM * sizeof(float));
    std::memcpy(full_v.data(), fp32_v_prefill->data(),
                PREFILL_LEN * Q3_KV_DIM * sizeof(float));
    std::memcpy(full_v.data() + PREFILL_LEN * Q3_KV_DIM, fp32_v_decode->data(),
                SEQ_LEN * Q3_KV_DIM * sizeof(float));

    std::vector<float> ref_output(SEQ_LEN * Q3_Q_DIM, 0.0f);
    ref::attention(fp32_q->data(), full_k.data(), full_v.data(),
                   ref_output.data(),
                   SEQ_LEN, KV_LEN, Q3_N_HEADS, Q3_N_KV_HEADS, HEAD_DIM,
                   /*causal=*/false);

    // Split TQ pipeline: TQ8(K) / TQ4(V) → cache → dequant → attention
    CPURingKVCacheTQ cache(mpi_ctx_, /*n_layers=*/8, /*batch_size=*/1,
                           MAX_SEQ, Q3_N_KV_HEADS, HEAD_DIM, DeviceId::cpu());

    const auto &layer_ctx = turboquant_ctx_->for_layer(LAYER_IDX);

    // Quantize and append prefill (K→TQ8, V→TQ4)
    auto tq8_k_prefill = TQ8Tensor::quantize_from_fp32(
        fp32_k_prefill->data(), fp32_k_prefill->shape(), HEAD_DIM, layer_ctx);
    auto tq4_v_prefill = TQ4Tensor::quantize_from_fp32(
        fp32_v_prefill->data(), fp32_v_prefill->shape(), HEAD_DIM, layer_ctx);
    ASSERT_TRUE(cache.append_kv(LAYER_IDX, 0, tq8_k_prefill.get(), tq4_v_prefill.get(), PREFILL_LEN));

    // Quantize and append decode token (K→TQ8, V→TQ4)
    auto tq8_k_decode = TQ8Tensor::quantize_from_fp32(
        fp32_k_decode->data(), fp32_k_decode->shape(), HEAD_DIM, layer_ctx);
    auto tq4_v_decode = TQ4Tensor::quantize_from_fp32(
        fp32_v_decode->data(), fp32_v_decode->shape(), HEAD_DIM, layer_ctx);
    ASSERT_TRUE(cache.append_kv(LAYER_IDX, 0, tq8_k_decode.get(), tq4_v_decode.get(), SEQ_LEN));

    // Dequantize from cache (split: TQ8 for K, TQ4 for V)
    ITensor *cache_k = cache.get_k(LAYER_IDX, 0);
    ITensor *cache_v = cache.get_v(LAYER_IDX, 0);
    auto *cache_k_tq8 = dynamic_cast<TQ8Tensor *>(cache_k);
    auto *cache_v_tq4 = dynamic_cast<TQ4Tensor *>(cache_v);
    ASSERT_NE(cache_k_tq8, nullptr) << "Cache K should be TQ8Tensor";
    ASSERT_NE(cache_v_tq4, nullptr) << "Cache V should be TQ4Tensor";

    std::vector<float> dequant_k(KV_LEN * Q3_KV_DIM, 0.0f);
    std::vector<float> dequant_v(KV_LEN * Q3_KV_DIM, 0.0f);

    cache_k_tq8->set_turboquant_context(&layer_ctx);
    cache_v_tq4->set_turboquant_context(&layer_ctx);
    for (int r = 0; r < KV_LEN; ++r)
    {
        cache_k_tq8->to_fp32_row(r, dequant_k.data() + r * Q3_KV_DIM);
        cache_v_tq4->to_fp32_row(r, dequant_v.data() + r * Q3_KV_DIM);
    }

    // Check dequant quality
    auto k_stats = computePerRowCosine(full_k.data(), dequant_k.data(), KV_LEN, Q3_KV_DIM);
    auto v_stats = computePerRowCosine(full_v.data(), dequant_v.data(), KV_LEN, Q3_KV_DIM);
    std::cout << "[Qwen3-0.6B dims] Dequant K (TQ8): avg_cos=" << k_stats.avg
              << " min_cos=" << k_stats.min << std::endl;
    std::cout << "[Qwen3-0.6B dims] Dequant V (TQ4): avg_cos=" << v_stats.avg
              << " min_cos=" << v_stats.min << std::endl;

    EXPECT_GT(k_stats.avg, 0.999) << "Qwen3-dims: dequant K (TQ8) quality too low";
    EXPECT_GT(v_stats.avg, 0.99) << "Qwen3-dims: dequant V (TQ4) quality too low";

    // Run attention via compute_decode() with real kernel
    auto kernel = fp32_q->createAttention();
    ASSERT_NE(kernel, nullptr);

    auto dequant_k_tensor = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(KV_LEN), static_cast<size_t>(Q3_KV_DIM)});
    auto dequant_v_tensor = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(KV_LEN), static_cast<size_t>(Q3_KV_DIM)});
    std::memcpy(dequant_k_tensor->mutable_data(), dequant_k.data(),
                KV_LEN * Q3_KV_DIM * sizeof(float));
    std::memcpy(dequant_v_tensor->mutable_data(), dequant_v.data(),
                KV_LEN * Q3_KV_DIM * sizeof(float));

    auto output_split = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(SEQ_LEN), static_cast<size_t>(Q3_Q_DIM)});

    ASSERT_TRUE(kernel->compute_tensor(
        fp32_q.get(), dequant_k_tensor.get(), dequant_v_tensor.get(), output_split.get(),
        1, SEQ_LEN, KV_LEN,
        Q3_N_HEADS, Q3_N_KV_HEADS, HEAD_DIM,
        false, -1,
        nullptr, nullptr, nullptr, -1));

    // Compare
    double attn_cosine = cosine_similarity(
        ref_output.data(), output_split->data(), SEQ_LEN * Q3_Q_DIM);
    std::cout << "[Qwen3-0.6B dims] Attention cosine (compute_decode): "
              << attn_cosine << std::endl;

    double min_head_cosine = 1.0;
    int worst_head = -1;
    for (int h = 0; h < Q3_N_HEADS; ++h)
    {
        double hcos = cosine_similarity(
            ref_output.data() + h * HEAD_DIM,
            output_split->data() + h * HEAD_DIM,
            HEAD_DIM);
        if (hcos < min_head_cosine)
        {
            min_head_cosine = hcos;
            worst_head = h;
        }
    }
    std::cout << "[Qwen3-0.6B dims] Per-head worst: head " << worst_head
              << " cosine=" << min_head_cosine << std::endl;

    EXPECT_GT(attn_cosine, 0.95)
        << "Qwen3-0.6B dims: attention cosine too low";
    EXPECT_GT(min_head_cosine, 0.90)
        << "Qwen3-0.6B dims: per-head worst cosine too low";
}
