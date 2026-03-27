/**
 * @file Test__CPURingKVCache_SplitTQ.cpp
 * @brief Tests for split TQ8-K / TQ4-V KV cache precision mode in CPURingKVCache.
 *
 * Validates the full round-trip: FP32 → TQ8/TQ4 quantize → ring buffer append →
 * ring buffer gather → TQ8/TQ4 dequantize → FP32. Ensures the ring buffer mechanics
 * (wrap-around, eviction, multi-layer) work correctly with asymmetric block-quantized data.
 *
 * K uses TQ8 (8-bit, 256 centroids) for higher inner-product fidelity.
 * V uses TQ4 (4-bit, 16 centroids) — V tolerates more quantization noise.
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

class Test__CPURingKVCache_SplitTQ : public ::testing::Test
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

    /// Quantize FP32 tensor to TQ8 (for K).
    std::shared_ptr<TQ8Tensor> quantizeTQ8(const FP32Tensor &src)
    {
        return TQ8Tensor::quantize_from_fp32(
            src.data(), src.shape(), HEAD_DIM, *turboquant_ctx_);
    }

    /// Quantize FP32 tensor to TQ4 (for V).
    std::shared_ptr<TQ4Tensor> quantizeTQ4(const FP32Tensor &src)
    {
        return TQ4Tensor::quantize_from_fp32(
            src.data(), src.shape(), HEAD_DIM, *turboquant_ctx_);
    }

    /// Dequantize TQ8 tensor to a new FP32 buffer.
    std::vector<float> dequantizeTQ8(const TQ8Tensor &t)
    {
        std::vector<float> buf(t.rows() * KV_DIM);
        t.dequantize_to_fp32(buf.data(), *turboquant_ctx_);
        return buf;
    }

    /// Dequantize TQ4 tensor to a new FP32 buffer.
    std::vector<float> dequantizeTQ4(const TQ4Tensor &t)
    {
        std::vector<float> buf(t.rows() * KV_DIM);
        t.dequantize_to_fp32(buf.data(), *turboquant_ctx_);
        return buf;
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

// ─────────────────────────────────────────────────────────────────────
// Split TQ8-K / TQ4-V tests
// ─────────────────────────────────────────────────────────────────────

TEST_F(Test__CPURingKVCache_SplitTQ, SplitTQ_PositionMajor_AppendAndGather_RoundTrip)
{
    constexpr int MAX_SEQ = 8;
    constexpr int N_TOKENS = 3;

    CPURingKVCacheTQ cache(mpi_ctx_, /*n_layers=*/1, /*batch_size=*/1, MAX_SEQ,
                           N_KV_HEADS, HEAD_DIM, DeviceId::cpu(),
                           KVCacheLayoutMode::POSITION_MAJOR);

    // Create FP32 data, quantize, append to cache
    auto fp32_k = makeRandomFP32(N_TOKENS, 100);
    auto fp32_v = makeRandomFP32(N_TOKENS, 200);
    auto tq8_k = quantizeTQ8(*fp32_k);
    auto tq4_v = quantizeTQ4(*fp32_v);

    ASSERT_TRUE(cache.append_kv(/*layer=*/0, /*seq=*/0, tq8_k.get(), tq4_v.get(), N_TOKENS));
    EXPECT_EQ(cache.ring_size(0, 0), N_TOKENS);

    // Gather from cache into fresh TQ4 output tensors
    auto out_k = std::make_shared<TQ8Tensor>(
        std::vector<size_t>{MAX_SEQ, KV_DIM}, HEAD_DIM, DeviceId::cpu());
    auto out_v = std::make_shared<TQ4Tensor>(
        std::vector<size_t>{MAX_SEQ, KV_DIM}, HEAD_DIM, DeviceId::cpu());
    std::vector<int> kv_lens;

    int max_kv = cache.gather_kv_batched(/*layer=*/0, /*num_seq=*/1,
                                         out_k.get(), out_v.get(), kv_lens);
    ASSERT_EQ(max_kv, N_TOKENS);
    ASSERT_EQ(kv_lens.size(), 1u);
    EXPECT_EQ(kv_lens[0], N_TOKENS);

    EXPECT_TRUE(compareRawRows(*tq8_k, *out_k, N_TOKENS))
        << "TQ8 K raw blocks changed after cache round-trip";
    EXPECT_TRUE(compareRawRows(*tq4_v, *out_v, N_TOKENS))
        << "TQ4 V raw blocks changed after cache round-trip";
}

TEST_F(Test__CPURingKVCache_SplitTQ, SplitTQ_HeadMajor_AppendAndGather_RoundTrip)
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
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), KV_DIM}, HEAD_DIM);
    auto out_v = std::make_shared<TQ4Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), KV_DIM}, HEAD_DIM);
    std::vector<int> kv_lens;

    int max_kv = cache.gather_kv_batched(0, 1, out_k.get(), out_v.get(), kv_lens);
    ASSERT_EQ(max_kv, N_TOKENS);

    EXPECT_TRUE(compareRawRows(*tq8_k, *out_k, N_TOKENS))
        << "HEAD_MAJOR TQ8 K raw blocks changed after cache round-trip";
    EXPECT_TRUE(compareRawRows(*tq4_v, *out_v, N_TOKENS))
        << "HEAD_MAJOR TQ4 V raw blocks changed after cache round-trip";
}

TEST_F(Test__CPURingKVCache_SplitTQ, SplitTQ_RingWrap_PreservesNewestTokens)
{
    constexpr int MAX_SEQ = 4;

    CPURingKVCacheTQ cache(mpi_ctx_, 1, 1, MAX_SEQ,
                           N_KV_HEADS, HEAD_DIM, DeviceId::cpu());

    // Append 6 tokens (overflows a capacity-4 ring → oldest 2 evicted)
    auto fp32_k = makeRandomFP32(6, 500);
    auto fp32_v = makeRandomFP32(6, 600);
    auto tq8_k = quantizeTQ8(*fp32_k);
    auto tq4_v = quantizeTQ4(*fp32_v);

    ASSERT_TRUE(cache.append_kv(0, 0, tq8_k.get(), tq4_v.get(), 6));
    EXPECT_EQ(cache.ring_size(0, 0), MAX_SEQ);

    // Gather — should contain tokens 2,3,4,5 in logical order
    auto out_k = std::make_shared<TQ8Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), KV_DIM}, HEAD_DIM);
    auto out_v = std::make_shared<TQ4Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), KV_DIM}, HEAD_DIM);
    std::vector<int> kv_lens;

    int max_kv = cache.gather_kv_batched(0, 1, out_k.get(), out_v.get(), kv_lens);
    ASSERT_EQ(max_kv, MAX_SEQ);

    // Directly quantize only the last 4 tokens as reference
    auto ref_k_slice = std::make_shared<FP32Tensor>(std::vector<size_t>{4, KV_DIM});
    auto ref_v_slice = std::make_shared<FP32Tensor>(std::vector<size_t>{4, KV_DIM});
    std::memcpy(ref_k_slice->mutable_data(), fp32_k->data() + 2 * KV_DIM, 4 * KV_DIM * sizeof(float));
    std::memcpy(ref_v_slice->mutable_data(), fp32_v->data() + 2 * KV_DIM, 4 * KV_DIM * sizeof(float));
    auto ref_tq8_k = quantizeTQ8(*ref_k_slice);
    auto ref_tq4_v = quantizeTQ4(*ref_v_slice);
    EXPECT_TRUE(compareRawRows(*ref_tq8_k, *out_k, MAX_SEQ))
        << "After ring wrap, gathered TQ8 K raw blocks don't match expected newest tokens";
    EXPECT_TRUE(compareRawRows(*ref_tq4_v, *out_v, MAX_SEQ))
        << "After ring wrap, gathered TQ4 V raw blocks don't match expected newest tokens";
}

TEST_F(Test__CPURingKVCache_SplitTQ, SplitTQ_IncrementalAppend_DecodeLike)
{
    constexpr int MAX_SEQ = 16;

    CPURingKVCacheTQ cache(mpi_ctx_, 1, 1, MAX_SEQ,
                           N_KV_HEADS, HEAD_DIM, DeviceId::cpu());

    // Simulate prefill (5 tokens) + 3 decode steps (1 token each)
    auto prefill_k = makeRandomFP32(5, 700);
    auto prefill_v = makeRandomFP32(5, 800);
    ASSERT_TRUE(cache.append_kv(0, 0, quantizeTQ8(*prefill_k).get(),
                                quantizeTQ4(*prefill_v).get(), 5));
    EXPECT_EQ(cache.ring_size(0, 0), 5);

    for (int step = 0; step < 3; ++step)
    {
        auto dec_k = makeRandomFP32(1, 900 + step);
        auto dec_v = makeRandomFP32(1, 1000 + step);
        ASSERT_TRUE(cache.append_kv(0, 0, quantizeTQ8(*dec_k).get(),
                                    quantizeTQ4(*dec_v).get(), 1));
    }
    EXPECT_EQ(cache.ring_size(0, 0), 8); // 5 prefill + 3 decode

    // Gather and verify no NaN/Inf
    auto out_k = std::make_shared<TQ8Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), KV_DIM}, HEAD_DIM);
    auto out_v = std::make_shared<TQ4Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), KV_DIM}, HEAD_DIM);
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

TEST_F(Test__CPURingKVCache_SplitTQ, SplitTQ_MultiLayer_IndependentData)
{
    constexpr int MAX_SEQ = 8;
    constexpr int N_LAYERS = 3;

    CPURingKVCacheTQ cache(mpi_ctx_, N_LAYERS, 1, MAX_SEQ,
                           N_KV_HEADS, HEAD_DIM, DeviceId::cpu());

    // Append different data to each layer
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
            std::vector<size_t>{static_cast<size_t>(MAX_SEQ), KV_DIM}, HEAD_DIM);
        auto out_v = std::make_shared<TQ4Tensor>(
            std::vector<size_t>{static_cast<size_t>(MAX_SEQ), KV_DIM}, HEAD_DIM);
        std::vector<int> kv_lens;
        int max_kv = cache.gather_kv_batched(l, 1, out_k.get(), out_v.get(), kv_lens);
        ASSERT_EQ(max_kv, 2);

        out_k->set_turboquant_context(turboquant_ctx_.get());
        layer_k_data[l].resize(2 * KV_DIM);
        for (int r = 0; r < 2; ++r)
            out_k->to_fp32_row(r, layer_k_data[l].data() + r * KV_DIM);
    }

    // Verify layers contain different data (different seeds → different values)
    double mse_01 = computeMSE(layer_k_data[0].data(), layer_k_data[1].data(), 2 * KV_DIM);
    double mse_02 = computeMSE(layer_k_data[0].data(), layer_k_data[2].data(), 2 * KV_DIM);
    EXPECT_GT(mse_01, 0.0) << "Layer 0 and 1 should have different data";
    EXPECT_GT(mse_02, 0.0) << "Layer 0 and 2 should have different data";
}

TEST_F(Test__CPURingKVCache_SplitTQ, SplitTQ_Clear_ResetsAllLayers)
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
// Quantization error bounds
// ─────────────────────────────────────────────────────────────────────

TEST_F(Test__CPURingKVCache_SplitTQ, SplitTQ_QuantizationError_WithinBounds)
{
    // TQ8 (K) has 256 centroids → much lower MSE than TQ4 (V) with 16 centroids.
    constexpr int N_TOKENS = 32;
    auto fp32 = makeRandomFP32(N_TOKENS, 3200);

    // TQ8 round-trip quality (K path)
    auto tq8 = quantizeTQ8(*fp32);
    auto roundtrip_tq8 = dequantizeTQ8(*tq8);
    double mse_tq8 = computeMSE(fp32->data(), roundtrip_tq8.data(), N_TOKENS * KV_DIM);
    EXPECT_LT(mse_tq8, 0.01) << "TQ8 MSE " << mse_tq8 << " exceeds expected quality floor";

    // TQ4 round-trip quality (V path)
    auto tq4 = quantizeTQ4(*fp32);
    auto roundtrip_tq4 = dequantizeTQ4(*tq4);
    double mse_tq4 = computeMSE(fp32->data(), roundtrip_tq4.data(), N_TOKENS * KV_DIM);
    EXPECT_LT(mse_tq4, 0.06) << "TQ4 MSE " << mse_tq4 << " exceeds expected quality floor";

    // TQ8 should be strictly better than TQ4
    EXPECT_LT(mse_tq8, mse_tq4) << "TQ8 should have lower MSE than TQ4";
    std::cout << "Quantization MSE — TQ8 (K): " << mse_tq8 << ", TQ4 (V): " << mse_tq4 << std::endl;
}

// ─────────────────────────────────────────────────────────────────────
// Ring cache accuracy: full pipeline cosine similarity
// ─────────────────────────────────────────────────────────────────────

TEST_F(Test__CPURingKVCache_SplitTQ, SplitTQ_CacheRoundTrip_CosineSimilarity)
{
    constexpr int MAX_SEQ = 32;
    constexpr int N_TOKENS = 16;

    CPURingKVCacheTQ cache(mpi_ctx_, 1, 1, MAX_SEQ,
                           N_KV_HEADS, HEAD_DIM, DeviceId::cpu());

    auto fp32_k = makeRandomFP32(N_TOKENS, 4100);
    auto fp32_v = makeRandomFP32(N_TOKENS, 4200);
    auto tq8_k = quantizeTQ8(*fp32_k);
    auto tq4_v = quantizeTQ4(*fp32_v);

    ASSERT_TRUE(cache.append_kv(0, 0, tq8_k.get(), tq4_v.get(), N_TOKENS));

    // Gather from cache
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

    // Compute per-row cosine similarity against original FP32
    double min_cosine_k = 1.0, avg_cosine_k = 0.0;
    double min_cosine_v = 1.0, avg_cosine_v = 0.0;

    for (int r = 0; r < N_TOKENS; ++r)
    {
        const float *orig_k = fp32_k->data() + r * KV_DIM;
        const float *orig_v = fp32_v->data() + r * KV_DIM;
        const float *reco_k = gathered_k.data() + r * KV_DIM;
        const float *reco_v = gathered_v.data() + r * KV_DIM;

        // Compute cosine per-row
        double dot_k = 0, na_k = 0, nb_k = 0;
        double dot_v = 0, na_v = 0, nb_v = 0;
        for (int i = 0; i < KV_DIM; ++i)
        {
            dot_k += orig_k[i] * reco_k[i];
            na_k += orig_k[i] * orig_k[i];
            nb_k += reco_k[i] * reco_k[i];
            dot_v += orig_v[i] * reco_v[i];
            na_v += orig_v[i] * orig_v[i];
            nb_v += reco_v[i] * reco_v[i];
        }
        double cos_k = (na_k > 1e-30 && nb_k > 1e-30)
                           ? dot_k / std::sqrt(na_k * nb_k)
                           : 0.0;
        double cos_v = (na_v > 1e-30 && nb_v > 1e-30)
                           ? dot_v / std::sqrt(na_v * nb_v)
                           : 0.0;
        min_cosine_k = std::min(min_cosine_k, cos_k);
        min_cosine_v = std::min(min_cosine_v, cos_v);
        avg_cosine_k += cos_k;
        avg_cosine_v += cos_v;
    }
    avg_cosine_k /= N_TOKENS;
    avg_cosine_v /= N_TOKENS;

    std::cout << "Cache round-trip K (TQ8): avg_cosine=" << avg_cosine_k
              << " min_cosine=" << min_cosine_k << std::endl;
    std::cout << "Cache round-trip V (TQ4): avg_cosine=" << avg_cosine_v
              << " min_cosine=" << min_cosine_v << std::endl;

    // TQ8 K: tighter thresholds (256 centroids)
    EXPECT_GT(avg_cosine_k, 0.98) << "Average K (TQ8) cosine through cache too low";
    EXPECT_GT(min_cosine_k, 0.95) << "Worst-case K (TQ8) cosine through cache too low";
    // TQ4 V: relaxed thresholds (16 centroids)
    EXPECT_GT(avg_cosine_v, 0.90) << "Average V (TQ4) cosine through cache too low";
    EXPECT_GT(min_cosine_v, 0.80) << "Worst-case V (TQ4) cosine through cache too low";
}

// ─────────────────────────────────────────────────────────────────────
// One-hot vectors through the ring cache
// ─────────────────────────────────────────────────────────────────────

TEST_F(Test__CPURingKVCache_SplitTQ, SplitTQ_OneHot_ThroughCache)
{
    constexpr int MAX_SEQ = 8;

    CPURingKVCacheTQ cache(mpi_ctx_, 1, 1, MAX_SEQ,
                           N_KV_HEADS, HEAD_DIM, DeviceId::cpu());

    // Create one-hot K and V vectors (2 tokens, each head has one-hot at different pos)
    constexpr int N_TOKENS = 2;
    auto fp32_k = std::make_shared<FP32Tensor>(
        std::vector<size_t>{N_TOKENS, static_cast<size_t>(KV_DIM)});
    auto fp32_v = std::make_shared<FP32Tensor>(
        std::vector<size_t>{N_TOKENS, static_cast<size_t>(KV_DIM)});
    float *kd = fp32_k->mutable_data();
    float *vd = fp32_v->mutable_data();
    std::fill(kd, kd + N_TOKENS * KV_DIM, 0.0f);
    std::fill(vd, vd + N_TOKENS * KV_DIM, 0.0f);

    // Token 0: one-hot at position 0 in each head
    for (int h = 0; h < N_KV_HEADS; ++h)
    {
        kd[h * HEAD_DIM] = 1.0f;
        vd[h * HEAD_DIM] = 1.0f;
    }
    // Token 1: one-hot at position HEAD_DIM/2 in each head
    for (int h = 0; h < N_KV_HEADS; ++h)
    {
        kd[KV_DIM + h * HEAD_DIM + HEAD_DIM / 2] = 1.0f;
        vd[KV_DIM + h * HEAD_DIM + HEAD_DIM / 2] = 1.0f;
    }

    auto tq8_k = quantizeTQ8(*fp32_k);
    auto tq4_v = quantizeTQ4(*fp32_v);
    ASSERT_TRUE(cache.append_kv(0, 0, tq8_k.get(), tq4_v.get(), N_TOKENS));

    // Gather and dequantize
    auto out_k = std::make_shared<TQ8Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), KV_DIM}, HEAD_DIM);
    auto out_v = std::make_shared<TQ4Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), KV_DIM}, HEAD_DIM);
    std::vector<int> kv_lens;
    cache.gather_kv_batched(0, 1, out_k.get(), out_v.get(), kv_lens);

    out_k->set_turboquant_context(turboquant_ctx_.get());
    std::vector<float> reco_k(N_TOKENS * KV_DIM);
    for (int r = 0; r < N_TOKENS; ++r)
        out_k->to_fp32_row(r, reco_k.data() + r * KV_DIM);

    // Verify: no NaN/Inf, and cosine > 0.7 for one-hot input
    for (size_t i = 0; i < reco_k.size(); ++i)
    {
        ASSERT_FALSE(std::isnan(reco_k[i])) << "NaN at index " << i;
        ASSERT_FALSE(std::isinf(reco_k[i])) << "Inf at index " << i;
    }

    // Per-row cosine with original
    for (int r = 0; r < N_TOKENS; ++r)
    {
        const float *orig = fp32_k->data() + r * KV_DIM;
        const float *reco = reco_k.data() + r * KV_DIM;
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < KV_DIM; ++i)
        {
            dot += orig[i] * reco[i];
            na += orig[i] * orig[i];
            nb += reco[i] * reco[i];
        }
        double cosine = (na > 1e-30 && nb > 1e-30) ? dot / std::sqrt(na * nb) : 0.0;
        EXPECT_GT(cosine, 0.70)
            << "One-hot through cache: cosine too low at row " << r;
    }
}
