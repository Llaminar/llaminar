/**
 * @file Test__CPURingKVCache_TQ8.cpp
 * @brief Tests for TQ8 and TQ8/TQ4 split-precision KV cache modes.
 *
 * Validates:
 * - CPURingKVCacheTQ8 (symmetric TQ8 K + TQ8 V)
 * - CPURingKVCacheTQ  (asymmetric TQ8 K + TQ4 V)
 * - k_precision() / v_precision() correct reporting
 * - Append/gather round-trip (bitwise block identity)
 * - Ring wrap preserves both K and V
 * - Incremental decode
 * - Multi-layer independence
 * - Clear resets all layers
 * - K quality (TQ8) >> V quality (TQ4) for split cache
 * - Cosine similarity bounds
 */

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <random>

#include "kernels/cpu/CPURingKVCache.h"
#include "kernels/cpu/turboquant/TurboQuantContext.h"
#include "tensors/Tensors.h"
#include "utils/MPIContext.h"

using namespace llaminar2;

// ─────────────────────────────────────────────────────────────────────
// Test fixture
// ─────────────────────────────────────────────────────────────────────

class Test__CPURingKVCache_TQ8 : public ::testing::Test
{
protected:
    static constexpr int HEAD_DIM = 64;
    static constexpr int N_KV_HEADS = 2;
    static constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM; // 128

    MPIContext mpi_ctx_{0, 1, MPI_COMM_WORLD};

    std::unique_ptr<TurboQuantContext> turboquant_ctx_;

    void SetUp() override
    {
        turboquant_ctx_ = std::make_unique<TurboQuantContext>(HEAD_DIM, /*rotation_seed=*/42, /*projection_seed=*/42);
    }

    /// Create a random FP32 tensor with shape [num_tokens, KV_DIM].
    static std::shared_ptr<FP32Tensor> makeRandomFP32(int num_tokens, unsigned seed)
    {
        auto t = std::make_shared<FP32Tensor>(std::vector<size_t>{
            static_cast<size_t>(num_tokens), static_cast<size_t>(KV_DIM)});
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        float *d = t->mutable_data();
        for (size_t i = 0; i < t->numel(); ++i)
            d[i] = dist(rng);
        return t;
    }

    /// Quantize FP32 tensor to TQ4.
    std::shared_ptr<TQ4Tensor> quantizeTQ4(const FP32Tensor &src)
    {
        return TQ4Tensor::quantize_from_fp32(
            src.data(), src.shape(), HEAD_DIM, *turboquant_ctx_);
    }

    /// Quantize FP32 tensor to TQ8.
    std::shared_ptr<TQ8Tensor> quantizeTQ8(const FP32Tensor &src)
    {
        return TQ8Tensor::quantize_from_fp32(
            src.data(), src.shape(), HEAD_DIM, *turboquant_ctx_);
    }

    /// Compute MSE between two float buffers.
    static double computeMSE(const float *a, const float *b, size_t n)
    {
        double acc = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            acc += d * d;
        }
        return acc / static_cast<double>(n);
    }

    /// Compute cosine similarity between two float buffers.
    static double computeCosine(const float *a, const float *b, size_t n)
    {
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if (na < 1e-30 || nb < 1e-30)
            return 0.0;
        return dot / std::sqrt(na * nb);
    }

    /// Compare raw block bytes for the first `rows` rows of two tensors.
    template <typename TensorT>
    static bool compareRawRows(const TensorT &expected, const TensorT &actual, size_t rows)
    {
        const auto *expected_bytes = static_cast<const uint8_t *>(expected.raw_data());
        const auto *actual_bytes = static_cast<const uint8_t *>(actual.raw_data());
        if (!expected_bytes || !actual_bytes)
            return false;

        const size_t row_bytes = expected.blocks_per_row() * expected.block_bytes();
        for (size_t row = 0; row < rows; ++row)
        {
            if (std::memcmp(expected_bytes + row * row_bytes,
                            actual_bytes + row * row_bytes,
                            row_bytes) != 0)
            {
                return false;
            }
        }
        return true;
    }
};

// ═════════════════════════════════════════════════════════════════════
//  Part 1: Symmetric TQ8 cache (CPURingKVCacheTQ8)
// ═════════════════════════════════════════════════════════════════════

TEST_F(Test__CPURingKVCache_TQ8, TQ8_Precision_Report)
{
    CPURingKVCacheTQ8 cache(mpi_ctx_, 1, 1, 8, N_KV_HEADS, HEAD_DIM, DeviceId::cpu());
    EXPECT_EQ(cache.k_precision(), ActivationPrecision::TQ8);
    EXPECT_EQ(cache.v_precision(), ActivationPrecision::TQ8);
}

TEST_F(Test__CPURingKVCache_TQ8, TQ8_AppendAndGather_RoundTrip)
{
    constexpr int MAX_SEQ = 8;
    constexpr int N_TOKENS = 3;

    CPURingKVCacheTQ8 cache(mpi_ctx_, 1, 1, MAX_SEQ,
                            N_KV_HEADS, HEAD_DIM, DeviceId::cpu());

    auto fp32_k = makeRandomFP32(N_TOKENS, 100);
    auto fp32_v = makeRandomFP32(N_TOKENS, 200);
    auto tq8_k = quantizeTQ8(*fp32_k);
    auto tq8_v = quantizeTQ8(*fp32_v);

    ASSERT_TRUE(cache.append_kv(0, 0, tq8_k.get(), tq8_v.get(), N_TOKENS));
    EXPECT_EQ(cache.ring_size(0, 0), N_TOKENS);

    auto out_k = std::make_shared<TQ8Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), static_cast<size_t>(KV_DIM)}, HEAD_DIM);
    auto out_v = std::make_shared<TQ8Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), static_cast<size_t>(KV_DIM)}, HEAD_DIM);
    std::vector<int> kv_lens;

    int max_kv = cache.gather_kv_batched(0, 1, out_k.get(), out_v.get(), kv_lens);
    ASSERT_EQ(max_kv, N_TOKENS);
    ASSERT_EQ(kv_lens.size(), 1u);
    EXPECT_EQ(kv_lens[0], N_TOKENS);

    EXPECT_TRUE(compareRawRows(*tq8_k, *out_k, N_TOKENS))
        << "TQ8 K raw blocks changed after cache round-trip";
    EXPECT_TRUE(compareRawRows(*tq8_v, *out_v, N_TOKENS))
        << "TQ8 V raw blocks changed after cache round-trip";
}

TEST_F(Test__CPURingKVCache_TQ8, TQ8_RingWrap_PreservesNewestTokens)
{
    constexpr int MAX_SEQ = 4;

    CPURingKVCacheTQ8 cache(mpi_ctx_, 1, 1, MAX_SEQ,
                            N_KV_HEADS, HEAD_DIM, DeviceId::cpu());

    auto fp32_k = makeRandomFP32(6, 500);
    auto fp32_v = makeRandomFP32(6, 600);
    auto tq8_k = quantizeTQ8(*fp32_k);
    auto tq8_v = quantizeTQ8(*fp32_v);

    ASSERT_TRUE(cache.append_kv(0, 0, tq8_k.get(), tq8_v.get(), 6));
    EXPECT_EQ(cache.ring_size(0, 0), MAX_SEQ);

    auto out_k = std::make_shared<TQ8Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), static_cast<size_t>(KV_DIM)}, HEAD_DIM);
    auto out_v = std::make_shared<TQ8Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), static_cast<size_t>(KV_DIM)}, HEAD_DIM);
    std::vector<int> kv_lens;

    int max_kv = cache.gather_kv_batched(0, 1, out_k.get(), out_v.get(), kv_lens);
    ASSERT_EQ(max_kv, MAX_SEQ);

    // Expected: tokens 2,3,4,5 (last 4 of 6)
    auto ref_k_slice = std::make_shared<FP32Tensor>(std::vector<size_t>{4, static_cast<size_t>(KV_DIM)});
    auto ref_v_slice = std::make_shared<FP32Tensor>(std::vector<size_t>{4, static_cast<size_t>(KV_DIM)});
    std::memcpy(ref_k_slice->mutable_data(), fp32_k->data() + 2 * KV_DIM, 4 * KV_DIM * sizeof(float));
    std::memcpy(ref_v_slice->mutable_data(), fp32_v->data() + 2 * KV_DIM, 4 * KV_DIM * sizeof(float));
    auto ref_tq8_k = quantizeTQ8(*ref_k_slice);
    auto ref_tq8_v = quantizeTQ8(*ref_v_slice);

    EXPECT_TRUE(compareRawRows(*ref_tq8_k, *out_k, MAX_SEQ))
        << "TQ8 K ring wrap: gathered blocks don't match expected newest tokens";
    EXPECT_TRUE(compareRawRows(*ref_tq8_v, *out_v, MAX_SEQ))
        << "TQ8 V ring wrap: gathered blocks don't match expected newest tokens";
}

TEST_F(Test__CPURingKVCache_TQ8, TQ8_CosineSimilarity_Bounds)
{
    constexpr int MAX_SEQ = 32;
    constexpr int N_TOKENS = 16;

    CPURingKVCacheTQ8 cache(mpi_ctx_, 1, 1, MAX_SEQ,
                            N_KV_HEADS, HEAD_DIM, DeviceId::cpu());

    auto fp32_k = makeRandomFP32(N_TOKENS, 4100);
    auto fp32_v = makeRandomFP32(N_TOKENS, 4200);
    auto tq8_k = quantizeTQ8(*fp32_k);
    auto tq8_v = quantizeTQ8(*fp32_v);

    ASSERT_TRUE(cache.append_kv(0, 0, tq8_k.get(), tq8_v.get(), N_TOKENS));

    auto out_k = std::make_shared<TQ8Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), static_cast<size_t>(KV_DIM)}, HEAD_DIM);
    auto out_v = std::make_shared<TQ8Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), static_cast<size_t>(KV_DIM)}, HEAD_DIM);
    std::vector<int> kv_lens;
    int max_kv = cache.gather_kv_batched(0, 1, out_k.get(), out_v.get(), kv_lens);
    ASSERT_EQ(max_kv, N_TOKENS);

    // Dequantize gathered K/V
    out_k->set_turboquant_context(turboquant_ctx_.get());
    out_v->set_turboquant_context(turboquant_ctx_.get());
    std::vector<float> gathered_k(N_TOKENS * KV_DIM);
    std::vector<float> gathered_v(N_TOKENS * KV_DIM);
    for (int r = 0; r < N_TOKENS; ++r)
    {
        out_k->to_fp32_row(r, gathered_k.data() + r * KV_DIM);
        out_v->to_fp32_row(r, gathered_v.data() + r * KV_DIM);
    }

    double min_cos_k = 1.0, avg_cos_k = 0.0;
    double min_cos_v = 1.0, avg_cos_v = 0.0;

    for (int r = 0; r < N_TOKENS; ++r)
    {
        double cos_k = computeCosine(fp32_k->data() + r * KV_DIM,
                                     gathered_k.data() + r * KV_DIM, KV_DIM);
        double cos_v = computeCosine(fp32_v->data() + r * KV_DIM,
                                     gathered_v.data() + r * KV_DIM, KV_DIM);
        min_cos_k = std::min(min_cos_k, cos_k);
        min_cos_v = std::min(min_cos_v, cos_v);
        avg_cos_k += cos_k;
        avg_cos_v += cos_v;
    }
    avg_cos_k /= N_TOKENS;
    avg_cos_v /= N_TOKENS;

    std::cout << "TQ8 cache K: avg_cosine=" << avg_cos_k << " min_cosine=" << min_cos_k << std::endl;
    std::cout << "TQ8 cache V: avg_cosine=" << avg_cos_v << " min_cosine=" << min_cos_v << std::endl;

    // TQ8 quality should be much higher than TQ4 thresholds (avg > 0.90, min > 0.80)
    EXPECT_GT(avg_cos_k, 0.99) << "TQ8 K average cosine through cache too low";
    EXPECT_GT(avg_cos_v, 0.99) << "TQ8 V average cosine through cache too low";
    EXPECT_GT(min_cos_k, 0.98) << "TQ8 K worst-case cosine through cache too low";
    EXPECT_GT(min_cos_v, 0.98) << "TQ8 V worst-case cosine through cache too low";
}

// ═════════════════════════════════════════════════════════════════════
//  Part 2: Asymmetric TQ8/TQ4 split cache (CPURingKVCacheTQ)
// ═════════════════════════════════════════════════════════════════════

TEST_F(Test__CPURingKVCache_TQ8, SplitTQ_Precision_Report)
{
    CPURingKVCacheTQ cache(mpi_ctx_, 1, 1, 8, N_KV_HEADS, HEAD_DIM, DeviceId::cpu());
    EXPECT_EQ(cache.k_precision(), ActivationPrecision::TQ8);
    EXPECT_EQ(cache.v_precision(), ActivationPrecision::TQ4);
}

TEST_F(Test__CPURingKVCache_TQ8, SplitTQ_AppendAndGather_RoundTrip)
{
    constexpr int MAX_SEQ = 8;
    constexpr int N_TOKENS = 3;

    CPURingKVCacheTQ cache(mpi_ctx_, 1, 1, MAX_SEQ,
                           N_KV_HEADS, HEAD_DIM, DeviceId::cpu());

    auto fp32_k = makeRandomFP32(N_TOKENS, 100);
    auto fp32_v = makeRandomFP32(N_TOKENS, 200);
    auto tq8_k = quantizeTQ8(*fp32_k); // K → TQ8
    auto tq4_v = quantizeTQ4(*fp32_v); // V → TQ4

    ASSERT_TRUE(cache.append_kv(0, 0, tq8_k.get(), tq4_v.get(), N_TOKENS));
    EXPECT_EQ(cache.ring_size(0, 0), N_TOKENS);

    // Gather output: K as TQ8, V as TQ4
    auto out_k = std::make_shared<TQ8Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), static_cast<size_t>(KV_DIM)}, HEAD_DIM);
    auto out_v = std::make_shared<TQ4Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), static_cast<size_t>(KV_DIM)}, HEAD_DIM);
    std::vector<int> kv_lens;

    int max_kv = cache.gather_kv_batched(0, 1, out_k.get(), out_v.get(), kv_lens);
    ASSERT_EQ(max_kv, N_TOKENS);
    ASSERT_EQ(kv_lens.size(), 1u);
    EXPECT_EQ(kv_lens[0], N_TOKENS);

    EXPECT_TRUE(compareRawRows(*tq8_k, *out_k, N_TOKENS))
        << "Split TQ K (TQ8) raw blocks changed after cache round-trip";
    EXPECT_TRUE(compareRawRows(*tq4_v, *out_v, N_TOKENS))
        << "Split TQ V (TQ4) raw blocks changed after cache round-trip";
}

TEST_F(Test__CPURingKVCache_TQ8, SplitTQ_HeadMajor_AppendAndGather_RoundTrip)
{
    constexpr int MAX_SEQ = 8;
    constexpr int N_TOKENS = 4;

    CPURingKVCacheTQ cache(mpi_ctx_, 1, 1, MAX_SEQ,
                           N_KV_HEADS, HEAD_DIM, DeviceId::cpu(),
                           KVCacheLayoutMode::HEAD_MAJOR);

    auto fp32_k = makeRandomFP32(N_TOKENS, 300);
    auto fp32_v = makeRandomFP32(N_TOKENS, 400);
    auto tq8_k = quantizeTQ8(*fp32_k);
    auto tq4_v = quantizeTQ4(*fp32_v);

    ASSERT_TRUE(cache.append_kv(0, 0, tq8_k.get(), tq4_v.get(), N_TOKENS));
    EXPECT_EQ(cache.ring_size(0, 0), N_TOKENS);

    auto out_k = std::make_shared<TQ8Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), static_cast<size_t>(KV_DIM)}, HEAD_DIM);
    auto out_v = std::make_shared<TQ4Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), static_cast<size_t>(KV_DIM)}, HEAD_DIM);
    std::vector<int> kv_lens;

    int max_kv = cache.gather_kv_batched(0, 1, out_k.get(), out_v.get(), kv_lens);
    ASSERT_EQ(max_kv, N_TOKENS);

    EXPECT_TRUE(compareRawRows(*tq8_k, *out_k, N_TOKENS))
        << "HEAD_MAJOR split TQ K raw blocks changed after round-trip";
    EXPECT_TRUE(compareRawRows(*tq4_v, *out_v, N_TOKENS))
        << "HEAD_MAJOR split TQ V raw blocks changed after round-trip";
}

TEST_F(Test__CPURingKVCache_TQ8, SplitTQ_RingWrap_PreservesNewestTokens)
{
    constexpr int MAX_SEQ = 4;

    CPURingKVCacheTQ cache(mpi_ctx_, 1, 1, MAX_SEQ,
                           N_KV_HEADS, HEAD_DIM, DeviceId::cpu());

    auto fp32_k = makeRandomFP32(6, 500);
    auto fp32_v = makeRandomFP32(6, 600);
    auto tq8_k = quantizeTQ8(*fp32_k);
    auto tq4_v = quantizeTQ4(*fp32_v);

    ASSERT_TRUE(cache.append_kv(0, 0, tq8_k.get(), tq4_v.get(), 6));
    EXPECT_EQ(cache.ring_size(0, 0), MAX_SEQ);

    auto out_k = std::make_shared<TQ8Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), static_cast<size_t>(KV_DIM)}, HEAD_DIM);
    auto out_v = std::make_shared<TQ4Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), static_cast<size_t>(KV_DIM)}, HEAD_DIM);
    std::vector<int> kv_lens;

    int max_kv = cache.gather_kv_batched(0, 1, out_k.get(), out_v.get(), kv_lens);
    ASSERT_EQ(max_kv, MAX_SEQ);

    // Reference: tokens 2,3,4,5
    auto ref_k_slice = std::make_shared<FP32Tensor>(std::vector<size_t>{4, static_cast<size_t>(KV_DIM)});
    auto ref_v_slice = std::make_shared<FP32Tensor>(std::vector<size_t>{4, static_cast<size_t>(KV_DIM)});
    std::memcpy(ref_k_slice->mutable_data(), fp32_k->data() + 2 * KV_DIM, 4 * KV_DIM * sizeof(float));
    std::memcpy(ref_v_slice->mutable_data(), fp32_v->data() + 2 * KV_DIM, 4 * KV_DIM * sizeof(float));
    auto ref_tq8_k = quantizeTQ8(*ref_k_slice);
    auto ref_tq4_v = quantizeTQ4(*ref_v_slice);

    EXPECT_TRUE(compareRawRows(*ref_tq8_k, *out_k, MAX_SEQ))
        << "Split TQ K (TQ8) ring wrap: blocks don't match expected newest tokens";
    EXPECT_TRUE(compareRawRows(*ref_tq4_v, *out_v, MAX_SEQ))
        << "Split TQ V (TQ4) ring wrap: blocks don't match expected newest tokens";
}

TEST_F(Test__CPURingKVCache_TQ8, SplitTQ_IncrementalAppend_DecodeLike)
{
    constexpr int MAX_SEQ = 16;

    CPURingKVCacheTQ cache(mpi_ctx_, 1, 1, MAX_SEQ,
                           N_KV_HEADS, HEAD_DIM, DeviceId::cpu());

    // Prefill 5 tokens
    auto prefill_k = makeRandomFP32(5, 700);
    auto prefill_v = makeRandomFP32(5, 800);
    ASSERT_TRUE(cache.append_kv(0, 0, quantizeTQ8(*prefill_k).get(),
                                quantizeTQ4(*prefill_v).get(), 5));
    EXPECT_EQ(cache.ring_size(0, 0), 5);

    // 3 decode steps
    for (int step = 0; step < 3; ++step)
    {
        auto dec_k = makeRandomFP32(1, 900 + step);
        auto dec_v = makeRandomFP32(1, 1000 + step);
        ASSERT_TRUE(cache.append_kv(0, 0, quantizeTQ8(*dec_k).get(),
                                    quantizeTQ4(*dec_v).get(), 1));
    }
    EXPECT_EQ(cache.ring_size(0, 0), 8); // 5 + 3

    // Gather and verify no NaN/Inf
    auto out_k = std::make_shared<TQ8Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), static_cast<size_t>(KV_DIM)}, HEAD_DIM);
    auto out_v = std::make_shared<TQ4Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), static_cast<size_t>(KV_DIM)}, HEAD_DIM);
    std::vector<int> kv_lens;

    int max_kv = cache.gather_kv_batched(0, 1, out_k.get(), out_v.get(), kv_lens);
    ASSERT_EQ(max_kv, 8);

    out_k->set_turboquant_context(turboquant_ctx_.get());
    out_v->set_turboquant_context(turboquant_ctx_.get());
    std::vector<float> fp32_k(8 * KV_DIM), fp32_v(8 * KV_DIM);
    for (int r = 0; r < 8; ++r)
    {
        out_k->to_fp32_row(r, fp32_k.data() + r * KV_DIM);
        out_v->to_fp32_row(r, fp32_v.data() + r * KV_DIM);
    }

    for (size_t i = 0; i < fp32_k.size(); ++i)
    {
        ASSERT_FALSE(std::isnan(fp32_k[i])) << "NaN in gathered K at index " << i;
        ASSERT_FALSE(std::isinf(fp32_k[i])) << "Inf in gathered K at index " << i;
        ASSERT_FALSE(std::isnan(fp32_v[i])) << "NaN in gathered V at index " << i;
        ASSERT_FALSE(std::isinf(fp32_v[i])) << "Inf in gathered V at index " << i;
    }
}

TEST_F(Test__CPURingKVCache_TQ8, SplitTQ_MultiLayer_IndependentData)
{
    constexpr int MAX_SEQ = 8;
    constexpr int N_LAYERS = 3;

    CPURingKVCacheTQ cache(mpi_ctx_, N_LAYERS, 1, MAX_SEQ,
                           N_KV_HEADS, HEAD_DIM, DeviceId::cpu());

    for (int l = 0; l < N_LAYERS; ++l)
    {
        auto k = makeRandomFP32(2, 1100 + l * 100);
        auto v = makeRandomFP32(2, 1200 + l * 100);
        ASSERT_TRUE(cache.append_kv(l, 0, quantizeTQ8(*k).get(),
                                    quantizeTQ4(*v).get(), 2));
        EXPECT_EQ(cache.ring_size(l, 0), 2);
    }

    // Gather each layer and verify they're different
    std::vector<std::vector<float>> layer_k_data(N_LAYERS);
    for (int l = 0; l < N_LAYERS; ++l)
    {
        auto out_k = std::make_shared<TQ8Tensor>(
            std::vector<size_t>{static_cast<size_t>(MAX_SEQ), static_cast<size_t>(KV_DIM)}, HEAD_DIM);
        auto out_v = std::make_shared<TQ4Tensor>(
            std::vector<size_t>{static_cast<size_t>(MAX_SEQ), static_cast<size_t>(KV_DIM)}, HEAD_DIM);
        std::vector<int> kv_lens;
        int max_kv = cache.gather_kv_batched(l, 1, out_k.get(), out_v.get(), kv_lens);
        ASSERT_EQ(max_kv, 2);

        out_k->set_turboquant_context(turboquant_ctx_.get());
        layer_k_data[l].resize(2 * KV_DIM);
        for (int r = 0; r < 2; ++r)
            out_k->to_fp32_row(r, layer_k_data[l].data() + r * KV_DIM);
    }

    double mse_01 = computeMSE(layer_k_data[0].data(), layer_k_data[1].data(), 2 * KV_DIM);
    double mse_02 = computeMSE(layer_k_data[0].data(), layer_k_data[2].data(), 2 * KV_DIM);
    EXPECT_GT(mse_01, 0.0) << "Layer 0 and 1 should have different data";
    EXPECT_GT(mse_02, 0.0) << "Layer 0 and 2 should have different data";
}

TEST_F(Test__CPURingKVCache_TQ8, SplitTQ_Clear_ResetsAllLayers)
{
    CPURingKVCacheTQ cache(mpi_ctx_, 2, 1, 8, N_KV_HEADS, HEAD_DIM, DeviceId::cpu());

    auto k = makeRandomFP32(3, 1400);
    auto v = makeRandomFP32(3, 1500);
    ASSERT_TRUE(cache.append_kv(0, 0, quantizeTQ8(*k).get(), quantizeTQ4(*v).get(), 3));
    ASSERT_TRUE(cache.append_kv(1, 0, quantizeTQ8(*k).get(), quantizeTQ4(*v).get(), 3));

    EXPECT_EQ(cache.ring_size(0, 0), 3);
    EXPECT_EQ(cache.ring_size(1, 0), 3);

    cache.clear();

    EXPECT_EQ(cache.ring_size(0, 0), 0);
    EXPECT_EQ(cache.ring_size(1, 0), 0);
}

// ─────────────────────────────────────────────────────────────────────
// Key test: K (TQ8) quality >> V (TQ4) quality
// ─────────────────────────────────────────────────────────────────────

TEST_F(Test__CPURingKVCache_TQ8, SplitTQ_KQuality_StrictlyBetterThan_V)
{
    constexpr int MAX_SEQ = 32;
    constexpr int N_TOKENS = 16;

    CPURingKVCacheTQ cache(mpi_ctx_, 1, 1, MAX_SEQ,
                           N_KV_HEADS, HEAD_DIM, DeviceId::cpu());

    auto fp32_k = makeRandomFP32(N_TOKENS, 5100);
    auto fp32_v = makeRandomFP32(N_TOKENS, 5200);
    ASSERT_TRUE(cache.append_kv(0, 0, quantizeTQ8(*fp32_k).get(),
                                quantizeTQ4(*fp32_v).get(), N_TOKENS));

    auto out_k = std::make_shared<TQ8Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), static_cast<size_t>(KV_DIM)}, HEAD_DIM);
    auto out_v = std::make_shared<TQ4Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), static_cast<size_t>(KV_DIM)}, HEAD_DIM);
    std::vector<int> kv_lens;
    cache.gather_kv_batched(0, 1, out_k.get(), out_v.get(), kv_lens);

    out_k->set_turboquant_context(turboquant_ctx_.get());
    out_v->set_turboquant_context(turboquant_ctx_.get());

    double total_mse_k = 0.0, total_mse_v = 0.0;
    double total_cos_k = 0.0, total_cos_v = 0.0;

    for (int r = 0; r < N_TOKENS; ++r)
    {
        std::vector<float> dk(KV_DIM), dv(KV_DIM);
        out_k->to_fp32_row(r, dk.data());
        out_v->to_fp32_row(r, dv.data());

        total_mse_k += computeMSE(fp32_k->data() + r * KV_DIM, dk.data(), KV_DIM);
        total_mse_v += computeMSE(fp32_v->data() + r * KV_DIM, dv.data(), KV_DIM);
        total_cos_k += computeCosine(fp32_k->data() + r * KV_DIM, dk.data(), KV_DIM);
        total_cos_v += computeCosine(fp32_v->data() + r * KV_DIM, dv.data(), KV_DIM);
    }

    double avg_mse_k = total_mse_k / N_TOKENS;
    double avg_mse_v = total_mse_v / N_TOKENS;
    double avg_cos_k = total_cos_k / N_TOKENS;
    double avg_cos_v = total_cos_v / N_TOKENS;

    std::cout << "Split TQ K (TQ8): avg_mse=" << avg_mse_k << " avg_cos=" << avg_cos_k << std::endl;
    std::cout << "Split TQ V (TQ4): avg_mse=" << avg_mse_v << " avg_cos=" << avg_cos_v << std::endl;
    std::cout << "K/V MSE ratio: " << avg_mse_k / avg_mse_v << std::endl;

    // TQ8 K must be strictly better than TQ4 V
    EXPECT_LT(avg_mse_k, avg_mse_v) << "TQ8 K MSE must be lower than TQ4 V MSE";
    EXPECT_GT(avg_cos_k, avg_cos_v) << "TQ8 K cosine must be higher than TQ4 V cosine";

    // TQ8 K should have >10× lower MSE than TQ4 V
    EXPECT_LT(avg_mse_k / avg_mse_v, 0.1) << "TQ8 K should have >10× lower MSE than TQ4 V";

    // Absolute quality bounds
    EXPECT_GT(avg_cos_k, 0.99) << "TQ8 K cosine too low";
    EXPECT_GT(avg_cos_v, 0.90) << "TQ4 V cosine too low";
}
