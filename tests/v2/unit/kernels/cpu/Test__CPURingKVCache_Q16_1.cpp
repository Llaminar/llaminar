/**
 * @file Test__CPURingKVCache_Q16_1.cpp
 * @brief Comprehensive tests for Q16_1 KV cache in CPURingKVCache.
 *
 * Q16_1 is unique among KV cache precisions in two ways:
 *   1. **Variable block sizes**: Q16_1 supports BLOCK_32, BLOCK_64, and BLOCK_128
 *      (72, 136, and 264 bytes per block). The block size is chosen based on head_dim.
 *   2. **HEAD_MAJOR layout**: Q16_1 caches use HEAD_MAJOR layout [n_kv_heads][position][head_dim]
 *      to optimize per-head attention access patterns.
 *
 * Tests cover:
 *  - Metadata, precision, and layout mode reporting
 *  - Variable block size selection (BLOCK_32, BLOCK_64, BLOCK_128)
 *  - Append and gather round-trip correctness (HEAD_MAJOR)
 *  - get_kv_converted (Q16_1 → FP32 shadow dequantization)
 *  - Ring buffer wrap-around with HEAD_MAJOR layout
 *  - Incremental decode (append-one, convert, repeat)
 *  - Multi-layer independence
 *  - Multi-sequence (batch) support
 *  - Shadow invalidation after clear / clear_sequence / clear_layer
 *  - Multiple complete ring wraps (stress)
 *  - Eviction + dequant correctness
 *  - Quantization error bounds (MSE, cosine similarity)
 *  - RoPE-on-read integration
 *  - K and V tensor value independence
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <numeric>
#include <random>

#include "kernels/cpu/CPURingKVCache.h"
#include "kernels/IKVCache.h"
#include "tensors/Tensors.h"
#include "utils/MPIContext.h"

using namespace llaminar2;

// =========================================================================
// Test fixture
// =========================================================================

class Test__CPURingKVCache_Q16_1 : public ::testing::Test
{
protected:
    MPIContext mpi_ctx_{0, 1, MPI_COMM_WORLD};

    /// Create a random FP32 tensor [num_tokens, kv_dim].
    static std::shared_ptr<FP32Tensor> makeRandomFP32(int num_tokens, int kv_dim, unsigned seed)
    {
        auto t = std::make_shared<FP32Tensor>(std::vector<size_t>{
            static_cast<size_t>(num_tokens), static_cast<size_t>(kv_dim)});
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        float *d = t->mutable_data();
        for (size_t i = 0; i < t->numel(); ++i)
            d[i] = dist(rng);
        return t;
    }

    /// Create a "tagged" FP32 tensor where row r has all elements = base + r.
    static std::shared_ptr<FP32Tensor> taggedFP32(int rows, int kv_dim, float base)
    {
        auto t = std::make_shared<FP32Tensor>(std::vector<size_t>{
            static_cast<size_t>(rows), static_cast<size_t>(kv_dim)});
        float *d = t->mutable_data();
        for (int r = 0; r < rows; ++r)
        {
            float v = base + static_cast<float>(r);
            for (int c = 0; c < kv_dim; ++c)
                d[r * kv_dim + c] = v;
        }
        return t;
    }

    /// Create a uniform FP32 tensor (all elements = val).
    static std::shared_ptr<FP32Tensor> uniformFP32(int rows, int kv_dim, float val)
    {
        auto t = std::make_shared<FP32Tensor>(std::vector<size_t>{
            static_cast<size_t>(rows), static_cast<size_t>(kv_dim)});
        std::fill(t->mutable_data(), t->mutable_data() + rows * kv_dim, val);
        return t;
    }

    /// Quantize FP32 to Q16_1 with fixed-scale VNNI-safe quantization (same as production path).
    static std::shared_ptr<Q16_1Tensor> quantizeQ16(const FP32Tensor &src, int head_dim, float scale = 256.0f)
    {
        const auto &shape = src.shape();
        Q16BlockSize bs = optimal_q16_block_size(head_dim);
        auto q = std::make_shared<Q16_1Tensor>(shape, bs, DeviceId::cpu());
        bool ok = q->copyFrom_fp32_fixed_scale(src.data(), scale, head_dim);
        if (!ok)
            throw std::runtime_error("quantizeQ16 failed");
        return q;
    }

    /// Check that an FP32 row (kv_dim wide) has all elements near expected_val.
    static bool rowNear(const float *row, int kv_dim, float expected, float tol = 0.05f)
    {
        for (int c = 0; c < kv_dim; ++c)
            if (std::abs(row[c] - expected) > tol)
                return false;
        return true;
    }

    /// Compute cosine similarity between two float buffers.
    static double cosineSimilarity(const float *a, const float *b, size_t n)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
            norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }
        if (norm_a < 1e-12 || norm_b < 1e-12)
            return 0.0;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
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
};

// =========================================================================
// 1. METADATA AND PRECISION
// =========================================================================

TEST_F(Test__CPURingKVCache_Q16_1, Metadata_PrecisionAndLayout)
{
    CPURingKVCacheQ16_1 cache(mpi_ctx_, 2, 1, 16, 2, 64, DeviceId::cpu(),
                              KVCacheLayoutMode::HEAD_MAJOR);

    EXPECT_EQ(cache.k_precision(), ActivationPrecision::Q16_1);
    EXPECT_EQ(cache.v_precision(), ActivationPrecision::Q16_1);
    EXPECT_EQ(cache.layout_mode(), KVCacheLayoutMode::HEAD_MAJOR);
    EXPECT_EQ(cache.max_seq_len(), 16);
    EXPECT_EQ(cache.n_layers(), 2);
    EXPECT_EQ(cache.n_kv_heads(), 2);
    EXPECT_FALSE(cache.is_sharded());
}

TEST_F(Test__CPURingKVCache_Q16_1, Metadata_PositionMajorLayout)
{
    // Q16_1 can also be created with POSITION_MAJOR
    CPURingKVCacheQ16_1 cache(mpi_ctx_, 1, 1, 8, 1, 32, DeviceId::cpu(),
                              KVCacheLayoutMode::POSITION_MAJOR);

    EXPECT_EQ(cache.layout_mode(), KVCacheLayoutMode::POSITION_MAJOR);
    EXPECT_EQ(cache.k_precision(), ActivationPrecision::Q16_1);
}

// =========================================================================
// 2. VARIABLE BLOCK SIZE SELECTION
// =========================================================================

TEST_F(Test__CPURingKVCache_Q16_1, BlockSize_HeadDim32_SelectsBlock32)
{
    EXPECT_EQ(optimal_q16_block_size(32), Q16BlockSize::BLOCK_32);
    EXPECT_EQ(q16_block_size_elements(Q16BlockSize::BLOCK_32), 32u);
    EXPECT_EQ(q16_block_size_bytes(Q16BlockSize::BLOCK_32), 72u);
}

TEST_F(Test__CPURingKVCache_Q16_1, BlockSize_HeadDim64_SelectsBlock64)
{
    EXPECT_EQ(optimal_q16_block_size(64), Q16BlockSize::BLOCK_64);
    EXPECT_EQ(q16_block_size_elements(Q16BlockSize::BLOCK_64), 64u);
    EXPECT_EQ(q16_block_size_bytes(Q16BlockSize::BLOCK_64), 136u);
}

TEST_F(Test__CPURingKVCache_Q16_1, BlockSize_HeadDim128_SelectsBlock128)
{
    EXPECT_EQ(optimal_q16_block_size(128), Q16BlockSize::BLOCK_128);
    EXPECT_EQ(q16_block_size_elements(Q16BlockSize::BLOCK_128), 128u);
    EXPECT_EQ(q16_block_size_bytes(Q16BlockSize::BLOCK_128), 264u);
}

// =========================================================================
// 3. APPEND AND GET_KV_CONVERTED — HEAD_MAJOR BLOCK_64 (Qwen2 shape)
// =========================================================================

TEST_F(Test__CPURingKVCache_Q16_1, HeadMajor_Block64_AppendAndConvert_NoWrap)
{
    // Qwen2-like: head_dim=64, n_kv_heads=2, kv_dim=128
    constexpr int HEAD_DIM = 64;
    constexpr int N_KV_HEADS = 2;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    constexpr int MAX_SEQ = 8;
    constexpr int N_TOKENS = 3;

    CPURingKVCacheQ16_1 cache(mpi_ctx_, 1, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM,
                              DeviceId::cpu(), KVCacheLayoutMode::HEAD_MAJOR);

    auto fp32_k = makeRandomFP32(N_TOKENS, KV_DIM, 100);
    auto fp32_v = makeRandomFP32(N_TOKENS, KV_DIM, 200);
    auto q16_k = quantizeQ16(*fp32_k, HEAD_DIM);
    auto q16_v = quantizeQ16(*fp32_v, HEAD_DIM);

    ASSERT_TRUE(cache.append_kv(0, 0, q16_k.get(), q16_v.get(), N_TOKENS));
    EXPECT_EQ(cache.ring_size(0, 0), N_TOKENS);

    // get_kv_converted → FP32
    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, N_TOKENS);

    // FP32 shadow should be close to the original FP32 data.
    // Q16_1 with fixed scale has quantization noise, but cosine should be very high.
    double cos_k = cosineSimilarity(fp32_k->data(), out_k->data(), N_TOKENS * KV_DIM);
    double cos_v = cosineSimilarity(fp32_v->data(), out_v->data(), N_TOKENS * KV_DIM);

    EXPECT_GT(cos_k, 0.99) << "K cosine similarity too low: " << cos_k;
    EXPECT_GT(cos_v, 0.99) << "V cosine similarity too low: " << cos_v;
}

// =========================================================================
// 4. APPEND AND GET_KV_CONVERTED — HEAD_MAJOR BLOCK_128 (Llama3 shape)
// =========================================================================

TEST_F(Test__CPURingKVCache_Q16_1, HeadMajor_Block128_AppendAndConvert_NoWrap)
{
    // Llama3-like: head_dim=128, n_kv_heads=2, kv_dim=256
    constexpr int HEAD_DIM = 128;
    constexpr int N_KV_HEADS = 2;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    constexpr int MAX_SEQ = 8;
    constexpr int N_TOKENS = 4;

    CPURingKVCacheQ16_1 cache(mpi_ctx_, 1, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM,
                              DeviceId::cpu(), KVCacheLayoutMode::HEAD_MAJOR);

    auto fp32_k = makeRandomFP32(N_TOKENS, KV_DIM, 300);
    auto fp32_v = makeRandomFP32(N_TOKENS, KV_DIM, 400);
    auto q16_k = quantizeQ16(*fp32_k, HEAD_DIM);
    auto q16_v = quantizeQ16(*fp32_v, HEAD_DIM);

    ASSERT_TRUE(cache.append_kv(0, 0, q16_k.get(), q16_v.get(), N_TOKENS));
    EXPECT_EQ(cache.ring_size(0, 0), N_TOKENS);

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, N_TOKENS);

    double cos_k = cosineSimilarity(fp32_k->data(), out_k->data(), N_TOKENS * KV_DIM);
    double cos_v = cosineSimilarity(fp32_v->data(), out_v->data(), N_TOKENS * KV_DIM);

    EXPECT_GT(cos_k, 0.99) << "K cosine (BLOCK_128) too low: " << cos_k;
    EXPECT_GT(cos_v, 0.99) << "V cosine (BLOCK_128) too low: " << cos_v;
}

// =========================================================================
// 5. APPEND AND GET_KV_CONVERTED — HEAD_MAJOR BLOCK_32
// =========================================================================

TEST_F(Test__CPURingKVCache_Q16_1, HeadMajor_Block32_AppendAndConvert_NoWrap)
{
    constexpr int HEAD_DIM = 32;
    constexpr int N_KV_HEADS = 2;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    constexpr int MAX_SEQ = 8;
    constexpr int N_TOKENS = 5;

    CPURingKVCacheQ16_1 cache(mpi_ctx_, 1, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM,
                              DeviceId::cpu(), KVCacheLayoutMode::HEAD_MAJOR);

    auto fp32_k = makeRandomFP32(N_TOKENS, KV_DIM, 500);
    auto fp32_v = makeRandomFP32(N_TOKENS, KV_DIM, 600);
    auto q16_k = quantizeQ16(*fp32_k, HEAD_DIM);
    auto q16_v = quantizeQ16(*fp32_v, HEAD_DIM);

    ASSERT_TRUE(cache.append_kv(0, 0, q16_k.get(), q16_v.get(), N_TOKENS));

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, N_TOKENS);

    double cos_k = cosineSimilarity(fp32_k->data(), out_k->data(), N_TOKENS * KV_DIM);
    EXPECT_GT(cos_k, 0.99) << "K cosine (BLOCK_32) too low: " << cos_k;
}

// =========================================================================
// 6. RING WRAP — HEAD_MAJOR BLOCK_64
// =========================================================================

TEST_F(Test__CPURingKVCache_Q16_1, HeadMajor_Block64_RingWrap_ConvertedCorrectly)
{
    constexpr int HEAD_DIM = 64;
    constexpr int N_KV_HEADS = 2;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    constexpr int MAX_SEQ = 4;

    CPURingKVCacheQ16_1 cache(mpi_ctx_, 1, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM,
                              DeviceId::cpu(), KVCacheLayoutMode::HEAD_MAJOR);

    // Fill ring: 4 tokens (seeds 10, 11, 12, 13)
    auto fp32_fill = makeRandomFP32(MAX_SEQ, KV_DIM, 10);
    auto q16_fill = quantizeQ16(*fp32_fill, HEAD_DIM);
    ASSERT_TRUE(cache.append_kv(0, 0, q16_fill.get(), q16_fill.get(), MAX_SEQ));
    EXPECT_EQ(cache.ring_size(0, 0), MAX_SEQ);
    EXPECT_EQ(cache.ring_head(0, 0), 0);

    // Wrap: append 2 more → overwrites oldest 2, head advances to 2
    auto fp32_new = makeRandomFP32(2, KV_DIM, 20);
    auto q16_new = quantizeQ16(*fp32_new, HEAD_DIM);
    ASSERT_TRUE(cache.append_kv(0, 0, q16_new.get(), q16_new.get(), 2));
    EXPECT_EQ(cache.ring_size(0, 0), MAX_SEQ);
    EXPECT_EQ(cache.ring_head(0, 0), 2);

    // get_kv_converted: should give logical order [old_2, old_3, new_0, new_1]
    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, MAX_SEQ);

    // The FP32 shadow row 2 and 3 should match the two new tokens
    const float *shadow = out_k->data();
    double cos_new0 = cosineSimilarity(fp32_new->data(), shadow + 2 * KV_DIM, KV_DIM);
    double cos_new1 = cosineSimilarity(fp32_new->data() + KV_DIM, shadow + 3 * KV_DIM, KV_DIM);
    EXPECT_GT(cos_new0, 0.99) << "Wrapped token 0 cosine: " << cos_new0;
    EXPECT_GT(cos_new1, 0.99) << "Wrapped token 1 cosine: " << cos_new1;

    // Rows 0-1 should match the old tokens 2-3 (indices 2,3 from original fill)
    double cos_old2 = cosineSimilarity(fp32_fill->data() + 2 * KV_DIM, shadow + 0 * KV_DIM, KV_DIM);
    double cos_old3 = cosineSimilarity(fp32_fill->data() + 3 * KV_DIM, shadow + 1 * KV_DIM, KV_DIM);
    EXPECT_GT(cos_old2, 0.99) << "Surviving old token 2 cosine: " << cos_old2;
    EXPECT_GT(cos_old3, 0.99) << "Surviving old token 3 cosine: " << cos_old3;
}

// =========================================================================
// 7. RING WRAP — HEAD_MAJOR BLOCK_128
// =========================================================================

TEST_F(Test__CPURingKVCache_Q16_1, HeadMajor_Block128_RingWrap_ConvertedCorrectly)
{
    constexpr int HEAD_DIM = 128;
    constexpr int N_KV_HEADS = 2;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    constexpr int MAX_SEQ = 4;

    CPURingKVCacheQ16_1 cache(mpi_ctx_, 1, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM,
                              DeviceId::cpu(), KVCacheLayoutMode::HEAD_MAJOR);

    auto fp32_fill = makeRandomFP32(MAX_SEQ, KV_DIM, 30);
    auto q16_fill = quantizeQ16(*fp32_fill, HEAD_DIM);
    ASSERT_TRUE(cache.append_kv(0, 0, q16_fill.get(), q16_fill.get(), MAX_SEQ));

    auto fp32_new = makeRandomFP32(2, KV_DIM, 40);
    auto q16_new = quantizeQ16(*fp32_new, HEAD_DIM);
    ASSERT_TRUE(cache.append_kv(0, 0, q16_new.get(), q16_new.get(), 2));
    EXPECT_EQ(cache.ring_head(0, 0), 2);

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, MAX_SEQ);

    const float *shadow = out_k->data();
    double cos_new0 = cosineSimilarity(fp32_new->data(), shadow + 2 * KV_DIM, KV_DIM);
    double cos_new1 = cosineSimilarity(fp32_new->data() + KV_DIM, shadow + 3 * KV_DIM, KV_DIM);
    EXPECT_GT(cos_new0, 0.99) << "BLOCK_128 wrapped token 0 cosine: " << cos_new0;
    EXPECT_GT(cos_new1, 0.99) << "BLOCK_128 wrapped token 1 cosine: " << cos_new1;
}

// =========================================================================
// 8. INCREMENTAL DECODE (APPEND-ONE, CONVERT, REPEAT)
// =========================================================================

TEST_F(Test__CPURingKVCache_Q16_1, HeadMajor_IncrementalDecode_Block64)
{
    constexpr int HEAD_DIM = 64;
    constexpr int N_KV_HEADS = 2;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    constexpr int MAX_SEQ = 16;
    constexpr int PREFILL_LEN = 5;
    constexpr int DECODE_STEPS = 6;

    CPURingKVCacheQ16_1 cache(mpi_ctx_, 1, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM,
                              DeviceId::cpu(), KVCacheLayoutMode::HEAD_MAJOR);

    // Prefill
    auto fp32_prefill = makeRandomFP32(PREFILL_LEN, KV_DIM, 1000);
    auto q16_prefill = quantizeQ16(*fp32_prefill, HEAD_DIM);
    ASSERT_TRUE(cache.append_kv(0, 0, q16_prefill.get(), q16_prefill.get(), PREFILL_LEN));

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, PREFILL_LEN);

    // Verify prefill round-trip quality
    double cos_prefill = cosineSimilarity(fp32_prefill->data(), out_k->data(),
                                          PREFILL_LEN * KV_DIM);
    EXPECT_GT(cos_prefill, 0.99) << "Prefill cosine: " << cos_prefill;

    // Incremental decode: append 1 token at a time
    for (int step = 0; step < DECODE_STEPS; ++step)
    {
        auto fp32_tok = makeRandomFP32(1, KV_DIM, 2000 + step);
        auto q16_tok = quantizeQ16(*fp32_tok, HEAD_DIM);
        ASSERT_TRUE(cache.append_kv(0, 0, q16_tok.get(), q16_tok.get(), 1));

        ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                           &out_k, &out_v, &len));
        EXPECT_EQ(len, PREFILL_LEN + step + 1);

        // The newest token should be the last row in the shadow
        const int last_row = len - 1;
        double cos_last = cosineSimilarity(fp32_tok->data(),
                                           out_k->data() + last_row * KV_DIM, KV_DIM);
        EXPECT_GT(cos_last, 0.99)
            << "Decode step " << step << " newest token cosine: " << cos_last;
    }
}

TEST_F(Test__CPURingKVCache_Q16_1, HeadMajor_IncrementalDecode_Block128)
{
    constexpr int HEAD_DIM = 128;
    constexpr int N_KV_HEADS = 2;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    constexpr int MAX_SEQ = 16;
    constexpr int PREFILL_LEN = 4;
    constexpr int DECODE_STEPS = 5;

    CPURingKVCacheQ16_1 cache(mpi_ctx_, 1, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM,
                              DeviceId::cpu(), KVCacheLayoutMode::HEAD_MAJOR);

    auto fp32_prefill = makeRandomFP32(PREFILL_LEN, KV_DIM, 3000);
    auto q16_prefill = quantizeQ16(*fp32_prefill, HEAD_DIM);
    ASSERT_TRUE(cache.append_kv(0, 0, q16_prefill.get(), q16_prefill.get(), PREFILL_LEN));

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;

    for (int step = 0; step < DECODE_STEPS; ++step)
    {
        auto fp32_tok = makeRandomFP32(1, KV_DIM, 4000 + step);
        auto q16_tok = quantizeQ16(*fp32_tok, HEAD_DIM);
        ASSERT_TRUE(cache.append_kv(0, 0, q16_tok.get(), q16_tok.get(), 1));

        ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                           &out_k, &out_v, &len));
        EXPECT_EQ(len, PREFILL_LEN + step + 1);

        const int last_row = len - 1;
        double cos_last = cosineSimilarity(fp32_tok->data(),
                                           out_k->data() + last_row * KV_DIM, KV_DIM);
        EXPECT_GT(cos_last, 0.99)
            << "BLOCK_128 decode step " << step << " cosine: " << cos_last;
    }
}

// =========================================================================
// 9. INCREMENTAL DECODE WITH RING WRAP
// =========================================================================

TEST_F(Test__CPURingKVCache_Q16_1, HeadMajor_IncrementalDecode_WithWrap)
{
    constexpr int HEAD_DIM = 64;
    constexpr int N_KV_HEADS = 2;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    constexpr int MAX_SEQ = 6;

    CPURingKVCacheQ16_1 cache(mpi_ctx_, 1, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM,
                              DeviceId::cpu(), KVCacheLayoutMode::HEAD_MAJOR);

    // Fill to capacity
    auto fp32_fill = makeRandomFP32(MAX_SEQ, KV_DIM, 5000);
    auto q16_fill = quantizeQ16(*fp32_fill, HEAD_DIM);
    ASSERT_TRUE(cache.append_kv(0, 0, q16_fill.get(), q16_fill.get(), MAX_SEQ));
    EXPECT_EQ(cache.ring_head(0, 0), 0);

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, MAX_SEQ);

    // Decode + wrap: append 4 tokens one at a time (wraps over oldest 4)
    std::vector<std::shared_ptr<FP32Tensor>> decode_tokens;
    for (int step = 0; step < 4; ++step)
    {
        auto fp32_tok = makeRandomFP32(1, KV_DIM, 6000 + step);
        decode_tokens.push_back(fp32_tok);
        auto q16_tok = quantizeQ16(*fp32_tok, HEAD_DIM);
        ASSERT_TRUE(cache.append_kv(0, 0, q16_tok.get(), q16_tok.get(), 1));

        ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                           &out_k, &out_v, &len));
        EXPECT_EQ(len, MAX_SEQ) << "Ring should stay at capacity";

        // Newest token should be at the end of the shadow's logical order
        const int last_row = len - 1;
        double cos_last = cosineSimilarity(fp32_tok->data(),
                                           out_k->data() + last_row * KV_DIM, KV_DIM);
        EXPECT_GT(cos_last, 0.99)
            << "Wrap decode step " << step << " newest token cosine: " << cos_last;
    }

    // Final state: 2 surviving old tokens + 4 new decode tokens
    // Verify the surviving old tokens (indices 4, 5 from fill)
    double cos_surv0 = cosineSimilarity(fp32_fill->data() + 4 * KV_DIM,
                                        out_k->data() + 0 * KV_DIM, KV_DIM);
    double cos_surv1 = cosineSimilarity(fp32_fill->data() + 5 * KV_DIM,
                                        out_k->data() + 1 * KV_DIM, KV_DIM);
    EXPECT_GT(cos_surv0, 0.99) << "Surviving fill token 4 cosine: " << cos_surv0;
    EXPECT_GT(cos_surv1, 0.99) << "Surviving fill token 5 cosine: " << cos_surv1;
}

// =========================================================================
// 10. MULTI-LAYER INDEPENDENCE
// =========================================================================

TEST_F(Test__CPURingKVCache_Q16_1, MultiLayer_IndependentData)
{
    constexpr int HEAD_DIM = 64;
    constexpr int N_KV_HEADS = 2;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    constexpr int MAX_SEQ = 8;
    constexpr int N_LAYERS = 3;
    constexpr int N_TOKENS = 3;

    CPURingKVCacheQ16_1 cache(mpi_ctx_, N_LAYERS, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM,
                              DeviceId::cpu(), KVCacheLayoutMode::HEAD_MAJOR);

    // Append different data to each layer
    std::vector<std::shared_ptr<FP32Tensor>> layer_fp32;
    for (int l = 0; l < N_LAYERS; ++l)
    {
        auto fp32 = makeRandomFP32(N_TOKENS, KV_DIM, 7000 + l * 100);
        layer_fp32.push_back(fp32);
        auto q16 = quantizeQ16(*fp32, HEAD_DIM);
        ASSERT_TRUE(cache.append_kv(l, 0, q16.get(), q16.get(), N_TOKENS));
    }

    // Verify each layer independently
    for (int l = 0; l < N_LAYERS; ++l)
    {
        ITensor *out_k = nullptr;
        ITensor *out_v = nullptr;
        int len = 0;
        ASSERT_TRUE(cache.get_kv_converted(l, 0, ActivationPrecision::FP32,
                                           &out_k, &out_v, &len));
        EXPECT_EQ(len, N_TOKENS);

        double cos = cosineSimilarity(layer_fp32[l]->data(), out_k->data(), N_TOKENS * KV_DIM);
        EXPECT_GT(cos, 0.99)
            << "Layer " << l << " cosine similarity: " << cos;
    }

    // Verify layers are different from each other
    ITensor *k0 = nullptr, *k1 = nullptr, *v0 = nullptr, *v1 = nullptr;
    int len0 = 0, len1 = 0;
    cache.get_kv_converted(0, 0, ActivationPrecision::FP32, &k0, &v0, &len0);
    cache.get_kv_converted(1, 0, ActivationPrecision::FP32, &k1, &v1, &len1);
    double cross_cos = cosineSimilarity(k0->data(), k1->data(), N_TOKENS * KV_DIM);
    // With different random seeds, cross-layer cosine should be low
    EXPECT_LT(cross_cos, 0.5) << "Layers should have different data, cosine: " << cross_cos;
}

// =========================================================================
// 11. MULTI-SEQUENCE (BATCH)
// =========================================================================

TEST_F(Test__CPURingKVCache_Q16_1, MultiSequence_IndependentConversion)
{
    constexpr int HEAD_DIM = 64;
    constexpr int N_KV_HEADS = 2;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    constexpr int MAX_SEQ = 8;

    CPURingKVCacheQ16_1 cache(mpi_ctx_, 1, 2, MAX_SEQ, N_KV_HEADS, HEAD_DIM,
                              DeviceId::cpu(), KVCacheLayoutMode::HEAD_MAJOR);

    // Seq 0: 3 tokens
    auto fp32_s0 = makeRandomFP32(3, KV_DIM, 8000);
    auto q16_s0 = quantizeQ16(*fp32_s0, HEAD_DIM);
    ASSERT_TRUE(cache.append_kv(0, 0, q16_s0.get(), q16_s0.get(), 3));

    // Seq 1: 5 tokens
    auto fp32_s1 = makeRandomFP32(5, KV_DIM, 9000);
    auto q16_s1 = quantizeQ16(*fp32_s1, HEAD_DIM);
    ASSERT_TRUE(cache.append_kv(0, 1, q16_s1.get(), q16_s1.get(), 5));

    // Verify independently
    ITensor *k0 = nullptr, *v0 = nullptr;
    int len0 = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &k0, &v0, &len0));
    EXPECT_EQ(len0, 3);
    double cos_s0 = cosineSimilarity(fp32_s0->data(), k0->data(), 3 * KV_DIM);
    EXPECT_GT(cos_s0, 0.99);

    ITensor *k1 = nullptr, *v1 = nullptr;
    int len1 = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 1, ActivationPrecision::FP32,
                                       &k1, &v1, &len1));
    EXPECT_EQ(len1, 5);
    double cos_s1 = cosineSimilarity(fp32_s1->data(), k1->data(), 5 * KV_DIM);
    EXPECT_GT(cos_s1, 0.99);
}

// =========================================================================
// 12. SHADOW INVALIDATION — CLEAR
// =========================================================================

TEST_F(Test__CPURingKVCache_Q16_1, Clear_ThenReappend_ShadowReflectsNewData)
{
    constexpr int HEAD_DIM = 64;
    constexpr int N_KV_HEADS = 2;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    constexpr int MAX_SEQ = 8;

    CPURingKVCacheQ16_1 cache(mpi_ctx_, 1, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM,
                              DeviceId::cpu(), KVCacheLayoutMode::HEAD_MAJOR);

    // Append + convert
    auto fp32_old = makeRandomFP32(4, KV_DIM, 10000);
    auto q16_old = quantizeQ16(*fp32_old, HEAD_DIM);
    ASSERT_TRUE(cache.append_kv(0, 0, q16_old.get(), q16_old.get(), 4));

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 4);

    // Clear
    cache.clear();
    EXPECT_EQ(cache.ring_size(0, 0), 0);

    // Re-append different data
    auto fp32_new = makeRandomFP32(2, KV_DIM, 11000);
    auto q16_new = quantizeQ16(*fp32_new, HEAD_DIM);
    ASSERT_TRUE(cache.append_kv(0, 0, q16_new.get(), q16_new.get(), 2));

    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 2);

    double cos = cosineSimilarity(fp32_new->data(), out_k->data(), 2 * KV_DIM);
    EXPECT_GT(cos, 0.99) << "After clear+re-append, shadow should have new data. Cosine: " << cos;
}

TEST_F(Test__CPURingKVCache_Q16_1, ClearSequence_ShadowReflectsNewData)
{
    constexpr int HEAD_DIM = 64;
    constexpr int N_KV_HEADS = 2;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    constexpr int MAX_SEQ = 8;

    CPURingKVCacheQ16_1 cache(mpi_ctx_, 1, 2, MAX_SEQ, N_KV_HEADS, HEAD_DIM,
                              DeviceId::cpu(), KVCacheLayoutMode::HEAD_MAJOR);

    auto fp32_s0 = makeRandomFP32(3, KV_DIM, 12000);
    auto q16_s0 = quantizeQ16(*fp32_s0, HEAD_DIM);
    ASSERT_TRUE(cache.append_kv(0, 0, q16_s0.get(), q16_s0.get(), 3));

    auto fp32_s1 = makeRandomFP32(3, KV_DIM, 13000);
    auto q16_s1 = quantizeQ16(*fp32_s1, HEAD_DIM);
    ASSERT_TRUE(cache.append_kv(0, 1, q16_s1.get(), q16_s1.get(), 3));

    // Convert both
    ITensor *k0 = nullptr, *v0 = nullptr, *k1 = nullptr, *v1 = nullptr;
    int len0 = 0, len1 = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32, &k0, &v0, &len0));
    ASSERT_TRUE(cache.get_kv_converted(0, 1, ActivationPrecision::FP32, &k1, &v1, &len1));

    // Clear only seq 0
    cache.clear_sequence(0, 0);
    EXPECT_EQ(cache.ring_size(0, 0), 0);
    EXPECT_EQ(cache.ring_size(0, 1), 3); // Untouched

    // Re-append to seq 0
    auto fp32_new = makeRandomFP32(2, KV_DIM, 14000);
    auto q16_new = quantizeQ16(*fp32_new, HEAD_DIM);
    ASSERT_TRUE(cache.append_kv(0, 0, q16_new.get(), q16_new.get(), 2));

    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32, &k0, &v0, &len0));
    EXPECT_EQ(len0, 2);
    double cos_new = cosineSimilarity(fp32_new->data(), k0->data(), 2 * KV_DIM);
    EXPECT_GT(cos_new, 0.99);

    // Seq 1 should still have old data
    ASSERT_TRUE(cache.get_kv_converted(0, 1, ActivationPrecision::FP32, &k1, &v1, &len1));
    EXPECT_EQ(len1, 3);
    double cos_s1 = cosineSimilarity(fp32_s1->data(), k1->data(), 3 * KV_DIM);
    EXPECT_GT(cos_s1, 0.99);
}

TEST_F(Test__CPURingKVCache_Q16_1, ClearLayer_ShadowReflectsNewData)
{
    constexpr int HEAD_DIM = 64;
    constexpr int N_KV_HEADS = 2;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    constexpr int MAX_SEQ = 8;

    CPURingKVCacheQ16_1 cache(mpi_ctx_, 2, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM,
                              DeviceId::cpu(), KVCacheLayoutMode::HEAD_MAJOR);

    auto fp32_l0 = makeRandomFP32(3, KV_DIM, 15000);
    auto q16_l0 = quantizeQ16(*fp32_l0, HEAD_DIM);
    ASSERT_TRUE(cache.append_kv(0, 0, q16_l0.get(), q16_l0.get(), 3));

    auto fp32_l1 = makeRandomFP32(3, KV_DIM, 16000);
    auto q16_l1 = quantizeQ16(*fp32_l1, HEAD_DIM);
    ASSERT_TRUE(cache.append_kv(1, 0, q16_l1.get(), q16_l1.get(), 3));

    // Convert both
    ITensor *k0 = nullptr, *v0 = nullptr, *k1 = nullptr, *v1 = nullptr;
    int len0 = 0, len1 = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32, &k0, &v0, &len0));
    ASSERT_TRUE(cache.get_kv_converted(1, 0, ActivationPrecision::FP32, &k1, &v1, &len1));

    // Clear layer 0
    cache.clear_layer(0);
    EXPECT_EQ(cache.ring_size(0, 0), 0);
    EXPECT_EQ(cache.ring_size(1, 0), 3);

    // Re-append to layer 0
    auto fp32_new = makeRandomFP32(2, KV_DIM, 17000);
    auto q16_new = quantizeQ16(*fp32_new, HEAD_DIM);
    ASSERT_TRUE(cache.append_kv(0, 0, q16_new.get(), q16_new.get(), 2));

    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32, &k0, &v0, &len0));
    EXPECT_EQ(len0, 2);
    double cos = cosineSimilarity(fp32_new->data(), k0->data(), 2 * KV_DIM);
    EXPECT_GT(cos, 0.99);

    // Layer 1 still has old data
    ASSERT_TRUE(cache.get_kv_converted(1, 0, ActivationPrecision::FP32, &k1, &v1, &len1));
    EXPECT_EQ(len1, 3);
    double cos_l1 = cosineSimilarity(fp32_l1->data(), k1->data(), 3 * KV_DIM);
    EXPECT_GT(cos_l1, 0.99);
}

// =========================================================================
// 13. MULTIPLE COMPLETE RING WRAPS (STRESS)
// =========================================================================

TEST_F(Test__CPURingKVCache_Q16_1, HeadMajor_MultiWrap_Stress_Block64)
{
    constexpr int HEAD_DIM = 64;
    constexpr int N_KV_HEADS = 2;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    constexpr int MAX_SEQ = 4;

    CPURingKVCacheQ16_1 cache(mpi_ctx_, 1, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM,
                              DeviceId::cpu(), KVCacheLayoutMode::HEAD_MAJOR);

    // Fill the ring
    auto fp32_fill = makeRandomFP32(MAX_SEQ, KV_DIM, 20000);
    auto q16_fill = quantizeQ16(*fp32_fill, HEAD_DIM);
    ASSERT_TRUE(cache.append_kv(0, 0, q16_fill.get(), q16_fill.get(), MAX_SEQ));

    // Wrap around 3 full cycles (12 tokens, each overwrites the oldest)
    for (int i = 0; i < 12; ++i)
    {
        auto fp32_tok = makeRandomFP32(1, KV_DIM, 21000 + i);
        auto q16_tok = quantizeQ16(*fp32_tok, HEAD_DIM);
        ASSERT_TRUE(cache.append_kv(0, 0, q16_tok.get(), q16_tok.get(), 1));

        ITensor *out_k = nullptr;
        ITensor *out_v = nullptr;
        int len = 0;
        ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                           &out_k, &out_v, &len));
        EXPECT_EQ(len, MAX_SEQ);

        // Newest token at end
        const int last = len - 1;
        double cos_last = cosineSimilarity(fp32_tok->data(),
                                           out_k->data() + last * KV_DIM, KV_DIM);
        EXPECT_GT(cos_last, 0.99)
            << "Multi-wrap step " << i << " cosine: " << cos_last;
    }
}

TEST_F(Test__CPURingKVCache_Q16_1, HeadMajor_MultiWrap_Stress_Block128)
{
    constexpr int HEAD_DIM = 128;
    constexpr int N_KV_HEADS = 2;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    constexpr int MAX_SEQ = 4;

    CPURingKVCacheQ16_1 cache(mpi_ctx_, 1, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM,
                              DeviceId::cpu(), KVCacheLayoutMode::HEAD_MAJOR);

    auto fp32_fill = makeRandomFP32(MAX_SEQ, KV_DIM, 22000);
    auto q16_fill = quantizeQ16(*fp32_fill, HEAD_DIM);
    ASSERT_TRUE(cache.append_kv(0, 0, q16_fill.get(), q16_fill.get(), MAX_SEQ));

    for (int i = 0; i < 12; ++i)
    {
        auto fp32_tok = makeRandomFP32(1, KV_DIM, 23000 + i);
        auto q16_tok = quantizeQ16(*fp32_tok, HEAD_DIM);
        ASSERT_TRUE(cache.append_kv(0, 0, q16_tok.get(), q16_tok.get(), 1));

        ITensor *out_k = nullptr;
        ITensor *out_v = nullptr;
        int len = 0;
        ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                           &out_k, &out_v, &len));
        EXPECT_EQ(len, MAX_SEQ);

        const int last = len - 1;
        double cos_last = cosineSimilarity(fp32_tok->data(),
                                           out_k->data() + last * KV_DIM, KV_DIM);
        EXPECT_GT(cos_last, 0.99)
            << "BLOCK_128 multi-wrap step " << i << " cosine: " << cos_last;
    }
}

// =========================================================================
// 14. EVICTION + DEQUANT CORRECTNESS
// =========================================================================

TEST_F(Test__CPURingKVCache_Q16_1, HeadMajor_Evict_ThenConvert)
{
    constexpr int HEAD_DIM = 64;
    constexpr int N_KV_HEADS = 2;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    constexpr int MAX_SEQ = 8;

    CPURingKVCacheQ16_1 cache(mpi_ctx_, 1, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM,
                              DeviceId::cpu(), KVCacheLayoutMode::HEAD_MAJOR);

    // Append 5 tokens
    auto fp32_k = makeRandomFP32(5, KV_DIM, 25000);
    auto fp32_v = makeRandomFP32(5, KV_DIM, 26000);
    auto q16_k = quantizeQ16(*fp32_k, HEAD_DIM);
    auto q16_v = quantizeQ16(*fp32_v, HEAD_DIM);
    ASSERT_TRUE(cache.append_kv(0, 0, q16_k.get(), q16_v.get(), 5));

    // Evict 2 → remaining: tokens 2, 3, 4
    cache.evict_oldest_from_sequence(0, 2);
    EXPECT_EQ(cache.ring_size(0, 0), 3);
    EXPECT_EQ(cache.ring_head(0, 0), 2);

    // Convert
    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 3);

    // Verify surviving tokens match original indices 2, 3, 4
    for (int i = 0; i < 3; ++i)
    {
        double cos_k = cosineSimilarity(fp32_k->data() + (i + 2) * KV_DIM,
                                        out_k->data() + i * KV_DIM, KV_DIM);
        double cos_v = cosineSimilarity(fp32_v->data() + (i + 2) * KV_DIM,
                                        out_v->data() + i * KV_DIM, KV_DIM);
        EXPECT_GT(cos_k, 0.99) << "Evicted K row " << i << " cosine: " << cos_k;
        EXPECT_GT(cos_v, 0.99) << "Evicted V row " << i << " cosine: " << cos_v;
    }
}

TEST_F(Test__CPURingKVCache_Q16_1, HeadMajor_Evict_ThenAppend_ThenConvert)
{
    constexpr int HEAD_DIM = 64;
    constexpr int N_KV_HEADS = 2;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    constexpr int MAX_SEQ = 4;

    CPURingKVCacheQ16_1 cache(mpi_ctx_, 1, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM,
                              DeviceId::cpu(), KVCacheLayoutMode::HEAD_MAJOR);

    // Fill 3 tokens
    auto fp32_fill = makeRandomFP32(3, KV_DIM, 27000);
    auto q16_fill = quantizeQ16(*fp32_fill, HEAD_DIM);
    ASSERT_TRUE(cache.append_kv(0, 0, q16_fill.get(), q16_fill.get(), 3));

    // Evict 2 → remaining: token 2 only
    cache.evict_oldest_from_sequence(0, 2);
    EXPECT_EQ(cache.ring_size(0, 0), 1);

    // Append 2 new tokens
    auto fp32_new = makeRandomFP32(2, KV_DIM, 28000);
    auto q16_new = quantizeQ16(*fp32_new, HEAD_DIM);
    ASSERT_TRUE(cache.append_kv(0, 0, q16_new.get(), q16_new.get(), 2));
    EXPECT_EQ(cache.ring_size(0, 0), 3);

    // Convert
    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 3);

    // Row 0: old token 2
    double cos0 = cosineSimilarity(fp32_fill->data() + 2 * KV_DIM,
                                   out_k->data() + 0 * KV_DIM, KV_DIM);
    EXPECT_GT(cos0, 0.99) << "Surviving token cosine: " << cos0;

    // Row 1-2: new tokens
    double cos1 = cosineSimilarity(fp32_new->data() + 0 * KV_DIM,
                                   out_k->data() + 1 * KV_DIM, KV_DIM);
    double cos2 = cosineSimilarity(fp32_new->data() + 1 * KV_DIM,
                                   out_k->data() + 2 * KV_DIM, KV_DIM);
    EXPECT_GT(cos1, 0.99) << "New token 0 cosine: " << cos1;
    EXPECT_GT(cos2, 0.99) << "New token 1 cosine: " << cos2;
}

// =========================================================================
// 15. QUANTIZATION ERROR BOUNDS
// =========================================================================

TEST_F(Test__CPURingKVCache_Q16_1, QuantizationErrorBounds_AllBlockSizes)
{
    // Q16_1 with fixed scale should have very low quantization error.
    // The fixed-scale formula is qs = round(fp32 / d) where d = scale / 32767,
    // so error ≈ d/2 per element = scale/(2*32767) ≈ 0.0039 for scale=256.
    constexpr int N_TOKENS = 32;

    auto testBlockSize = [&](int head_dim, const char *label)
    {
        const int kv_dim = 2 * head_dim; // 2 heads
        auto fp32 = makeRandomFP32(N_TOKENS, kv_dim, 30000 + head_dim);
        auto q16 = quantizeQ16(*fp32, head_dim);

        // Dequant by element
        std::vector<float> dequant(N_TOKENS * kv_dim);
        for (int r = 0; r < N_TOKENS; ++r)
            for (int c = 0; c < kv_dim; ++c)
                dequant[r * kv_dim + c] = q16->dequant_element(r, c);

        double mse = computeMSE(fp32->data(), dequant.data(), N_TOKENS * kv_dim);
        double cos = cosineSimilarity(fp32->data(), dequant.data(), N_TOKENS * kv_dim);

        EXPECT_LT(mse, 0.001) << label << " MSE too high: " << mse;
        EXPECT_GT(cos, 0.999) << label << " cosine too low: " << cos;
    };

    testBlockSize(32, "BLOCK_32");
    testBlockSize(64, "BLOCK_64");
    testBlockSize(128, "BLOCK_128");
}

// =========================================================================
// 16. K AND V VALUE INDEPENDENCE
// =========================================================================

TEST_F(Test__CPURingKVCache_Q16_1, KAndV_IndependentValues)
{
    constexpr int HEAD_DIM = 64;
    constexpr int N_KV_HEADS = 2;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    constexpr int MAX_SEQ = 8;
    constexpr int N_TOKENS = 4;

    CPURingKVCacheQ16_1 cache(mpi_ctx_, 1, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM,
                              DeviceId::cpu(), KVCacheLayoutMode::HEAD_MAJOR);

    auto fp32_k = makeRandomFP32(N_TOKENS, KV_DIM, 40000);
    auto fp32_v = makeRandomFP32(N_TOKENS, KV_DIM, 41000);
    auto q16_k = quantizeQ16(*fp32_k, HEAD_DIM);
    auto q16_v = quantizeQ16(*fp32_v, HEAD_DIM);

    ASSERT_TRUE(cache.append_kv(0, 0, q16_k.get(), q16_v.get(), N_TOKENS));

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, N_TOKENS);

    // K should match fp32_k
    double cos_k = cosineSimilarity(fp32_k->data(), out_k->data(), N_TOKENS * KV_DIM);
    EXPECT_GT(cos_k, 0.99) << "K cosine: " << cos_k;

    // V should match fp32_v
    double cos_v = cosineSimilarity(fp32_v->data(), out_v->data(), N_TOKENS * KV_DIM);
    EXPECT_GT(cos_v, 0.99) << "V cosine: " << cos_v;

    // K and V should NOT match each other (different seeds)
    double cross = cosineSimilarity(out_k->data(), out_v->data(), N_TOKENS * KV_DIM);
    EXPECT_LT(cross, 0.5) << "K/V should be different data, cross-cosine: " << cross;
}

// =========================================================================
// 17. ROPE-ON-READ INTEGRATION
// =========================================================================

TEST_F(Test__CPURingKVCache_Q16_1, HeadMajor_RoPE_NoWrap)
{
    constexpr int HEAD_DIM = 64;
    constexpr int N_KV_HEADS = 2;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    constexpr int MAX_SEQ = 8;
    constexpr int N_TOKENS = 3;

    // Use two separate caches to get clean no-RoPE vs with-RoPE conversions.
    // (Sharing a single cache requires careful shadow invalidation sequencing.)
    CPURingKVCacheQ16_1 cache_noRoPE(mpi_ctx_, 1, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM,
                                     DeviceId::cpu(), KVCacheLayoutMode::HEAD_MAJOR);
    CPURingKVCacheQ16_1 cache_rope(mpi_ctx_, 1, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM,
                                   DeviceId::cpu(), KVCacheLayoutMode::HEAD_MAJOR);

    // Use uniform data so we can verify RoPE modifies it
    auto fp32 = uniformFP32(N_TOKENS, KV_DIM, 1.0f);
    auto q16 = quantizeQ16(*fp32, HEAD_DIM);
    ASSERT_TRUE(cache_noRoPE.append_kv(0, 0, q16.get(), q16.get(), N_TOKENS));
    ASSERT_TRUE(cache_rope.append_kv(0, 0, q16.get(), q16.get(), N_TOKENS));

    // Convert WITHOUT RoPE
    ITensor *k_no_rope = nullptr;
    ITensor *v_no_rope = nullptr;
    int len_no_rope = 0;
    ASSERT_TRUE(cache_noRoPE.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                              &k_no_rope, &v_no_rope, &len_no_rope));
    // Save a copy
    std::vector<float> k_before(k_no_rope->data(), k_no_rope->data() + N_TOKENS * KV_DIM);

    // Convert WITH RoPE
    IKVCache::KVReadParams rope;
    rope.rope_theta = 10000.0f;
    rope.position_start = 0;
    rope.n_kv_heads = N_KV_HEADS;
    rope.head_dim = HEAD_DIM;

    ITensor *k_with_rope = nullptr;
    ITensor *v_with_rope = nullptr;
    int len_rope = 0;
    ASSERT_TRUE(cache_rope.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                            &k_with_rope, &v_with_rope, &len_rope,
                                            &rope));
    EXPECT_EQ(len_rope, N_TOKENS);

    // Position 0 with cos(0)=1, sin(0)=0: K values should be unchanged
    bool pos0_matches = true;
    for (int c = 0; c < KV_DIM; ++c)
    {
        if (std::abs(k_before[c] - k_with_rope->data()[c]) > 0.01f)
        {
            pos0_matches = false;
            break;
        }
    }
    EXPECT_TRUE(pos0_matches) << "Position 0 with RoPE should be identity (cos(0)=1, sin(0)=0)";

    // Position > 0: K values should differ from non-RoPE version
    bool pos1_differs = false;
    for (int c = 0; c < KV_DIM; ++c)
    {
        if (std::abs(k_before[KV_DIM + c] - k_with_rope->data()[KV_DIM + c]) > 0.001f)
        {
            pos1_differs = true;
            break;
        }
    }
    EXPECT_TRUE(pos1_differs) << "Position 1 with RoPE should differ from non-RoPE";

    // V should NOT be affected by RoPE — should still be ~1.0
    for (int i = 0; i < N_TOKENS * KV_DIM; ++i)
    {
        EXPECT_NEAR(v_with_rope->data()[i], 1.0f, 0.02f)
            << "V should be ~1.0 (not RoPE-modified), index " << i;
    }
}

TEST_F(Test__CPURingKVCache_Q16_1, HeadMajor_RoPE_WithWrap)
{
    constexpr int HEAD_DIM = 64;
    constexpr int N_KV_HEADS = 2;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    constexpr int MAX_SEQ = 4;

    CPURingKVCacheQ16_1 cache(mpi_ctx_, 1, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM,
                              DeviceId::cpu(), KVCacheLayoutMode::HEAD_MAJOR);

    // Fill + wrap
    auto fp32_fill = uniformFP32(MAX_SEQ, KV_DIM, 1.0f);
    auto q16_fill = quantizeQ16(*fp32_fill, HEAD_DIM);
    ASSERT_TRUE(cache.append_kv(0, 0, q16_fill.get(), q16_fill.get(), MAX_SEQ));

    auto fp32_new = uniformFP32(2, KV_DIM, 1.0f);
    auto q16_new = quantizeQ16(*fp32_new, HEAD_DIM);
    ASSERT_TRUE(cache.append_kv(0, 0, q16_new.get(), q16_new.get(), 2));
    EXPECT_EQ(cache.ring_head(0, 0), 2);

    IKVCache::KVReadParams rope;
    rope.rope_theta = 10000.0f;
    rope.position_start = 0;
    rope.n_kv_heads = N_KV_HEADS;
    rope.head_dim = HEAD_DIM;

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len, &rope));
    EXPECT_EQ(len, MAX_SEQ);

    // Verify no NaN/Inf
    for (int i = 0; i < MAX_SEQ * KV_DIM; ++i)
    {
        ASSERT_FALSE(std::isnan(out_k->data()[i])) << "NaN at K[" << i << "]";
        ASSERT_FALSE(std::isinf(out_k->data()[i])) << "Inf at K[" << i << "]";
    }

    // Position 0 should be identity (cos(0)=1, sin(0)=0)
    for (int c = 0; c < KV_DIM; ++c)
        EXPECT_NEAR(out_k->data()[c], 1.0f, 0.02f)
            << "RoPE pos0 should be identity, got " << out_k->data()[c] << " at col " << c;
}

// =========================================================================
// 18. EMPTY CACHE AND EDGE CASES
// =========================================================================

TEST_F(Test__CPURingKVCache_Q16_1, GetKvConverted_EmptyCache_ReturnsZeroLen)
{
    CPURingKVCacheQ16_1 cache(mpi_ctx_, 1, 1, 8, 2, 64, DeviceId::cpu(),
                              KVCacheLayoutMode::HEAD_MAJOR);

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = -1;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 0);
}

TEST_F(Test__CPURingKVCache_Q16_1, GetKvConverted_OutOfBounds_ReturnsFalse)
{
    CPURingKVCacheQ16_1 cache(mpi_ctx_, 1, 1, 8, 2, 64, DeviceId::cpu(),
                              KVCacheLayoutMode::HEAD_MAJOR);

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 99;
    EXPECT_FALSE(cache.get_kv_converted(-1, 0, ActivationPrecision::FP32,
                                        &out_k, &out_v, &len));
    EXPECT_EQ(len, 0);
    EXPECT_EQ(out_k, nullptr);
}

TEST_F(Test__CPURingKVCache_Q16_1, GetKvConverted_UnsupportedTarget_ReturnsFalse)
{
    constexpr int HEAD_DIM = 64;
    constexpr int N_KV_HEADS = 2;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;

    CPURingKVCacheQ16_1 cache(mpi_ctx_, 1, 1, 8, N_KV_HEADS, HEAD_DIM,
                              DeviceId::cpu(), KVCacheLayoutMode::HEAD_MAJOR);

    auto fp32 = makeRandomFP32(2, KV_DIM, 50000);
    auto q16 = quantizeQ16(*fp32, HEAD_DIM);
    ASSERT_TRUE(cache.append_kv(0, 0, q16.get(), q16.get(), 2));

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    EXPECT_FALSE(cache.get_kv_converted(0, 0, ActivationPrecision::BF16,
                                        &out_k, &out_v, &len));
}

// =========================================================================
// 19. SINGLE TOKEN (M=1 DECODE PATH)
// =========================================================================

TEST_F(Test__CPURingKVCache_Q16_1, SingleToken_AppendAndConvert)
{
    constexpr int HEAD_DIM = 64;
    constexpr int N_KV_HEADS = 2;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    constexpr int MAX_SEQ = 8;

    CPURingKVCacheQ16_1 cache(mpi_ctx_, 1, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM,
                              DeviceId::cpu(), KVCacheLayoutMode::HEAD_MAJOR);

    auto fp32 = makeRandomFP32(1, KV_DIM, 51000);
    auto q16 = quantizeQ16(*fp32, HEAD_DIM);
    ASSERT_TRUE(cache.append_kv(0, 0, q16.get(), q16.get(), 1));
    EXPECT_EQ(cache.ring_size(0, 0), 1);

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 1);

    double cos = cosineSimilarity(fp32->data(), out_k->data(), KV_DIM);
    EXPECT_GT(cos, 0.99) << "Single token cosine: " << cos;
}

// =========================================================================
// 20. HEAD_MAJOR HEAD COUNT > 2 (MULTI-HEAD)
// =========================================================================

TEST_F(Test__CPURingKVCache_Q16_1, HeadMajor_MultipleKVHeads)
{
    // Test with more KV heads (e.g. 4 heads, common in larger models)
    constexpr int HEAD_DIM = 64;
    constexpr int N_KV_HEADS = 4;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    constexpr int MAX_SEQ = 8;
    constexpr int N_TOKENS = 3;

    CPURingKVCacheQ16_1 cache(mpi_ctx_, 1, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM,
                              DeviceId::cpu(), KVCacheLayoutMode::HEAD_MAJOR);

    auto fp32_k = makeRandomFP32(N_TOKENS, KV_DIM, 52000);
    auto fp32_v = makeRandomFP32(N_TOKENS, KV_DIM, 53000);
    auto q16_k = quantizeQ16(*fp32_k, HEAD_DIM);
    auto q16_v = quantizeQ16(*fp32_v, HEAD_DIM);

    ASSERT_TRUE(cache.append_kv(0, 0, q16_k.get(), q16_v.get(), N_TOKENS));

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, N_TOKENS);

    double cos_k = cosineSimilarity(fp32_k->data(), out_k->data(), N_TOKENS * KV_DIM);
    double cos_v = cosineSimilarity(fp32_v->data(), out_v->data(), N_TOKENS * KV_DIM);
    EXPECT_GT(cos_k, 0.99) << "4-head K cosine: " << cos_k;
    EXPECT_GT(cos_v, 0.99) << "4-head V cosine: " << cos_v;
}

TEST_F(Test__CPURingKVCache_Q16_1, HeadMajor_MultipleKVHeads_WithWrap)
{
    constexpr int HEAD_DIM = 64;
    constexpr int N_KV_HEADS = 4;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    constexpr int MAX_SEQ = 4;

    CPURingKVCacheQ16_1 cache(mpi_ctx_, 1, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM,
                              DeviceId::cpu(), KVCacheLayoutMode::HEAD_MAJOR);

    auto fp32_fill = makeRandomFP32(MAX_SEQ, KV_DIM, 54000);
    auto q16_fill = quantizeQ16(*fp32_fill, HEAD_DIM);
    ASSERT_TRUE(cache.append_kv(0, 0, q16_fill.get(), q16_fill.get(), MAX_SEQ));

    auto fp32_new = makeRandomFP32(1, KV_DIM, 55000);
    auto q16_new = quantizeQ16(*fp32_new, HEAD_DIM);
    ASSERT_TRUE(cache.append_kv(0, 0, q16_new.get(), q16_new.get(), 1));

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, MAX_SEQ);

    // New token is at the end (row 3)
    double cos_new = cosineSimilarity(fp32_new->data(),
                                      out_k->data() + 3 * KV_DIM, KV_DIM);
    EXPECT_GT(cos_new, 0.99) << "4-head wrap new token cosine: " << cos_new;

    // Old surviving tokens (indices 1, 2, 3 from fill) at rows 0, 1, 2
    for (int i = 0; i < 3; ++i)
    {
        double cos = cosineSimilarity(fp32_fill->data() + (i + 1) * KV_DIM,
                                      out_k->data() + i * KV_DIM, KV_DIM);
        EXPECT_GT(cos, 0.99) << "4-head wrap surviving row " << i << " cosine: " << cos;
    }
}

// =========================================================================
// 21. RING HEAD RETURNS TO ZERO
// =========================================================================

TEST_F(Test__CPURingKVCache_Q16_1, HeadMajor_HeadReturnsToZero_WithIntermediateQuery)
{
    // When the ring head wraps back to exactly 0, the shadow invalidation
    // detects the change because intermediate appends move the head to
    // non-zero positions first.
    constexpr int HEAD_DIM = 64;
    constexpr int N_KV_HEADS = 2;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    constexpr int MAX_SEQ = 4;

    CPURingKVCacheQ16_1 cache(mpi_ctx_, 1, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM,
                              DeviceId::cpu(), KVCacheLayoutMode::HEAD_MAJOR);

    // Fill
    auto fp32_fill = makeRandomFP32(MAX_SEQ, KV_DIM, 56000);
    auto q16_fill = quantizeQ16(*fp32_fill, HEAD_DIM);
    ASSERT_TRUE(cache.append_kv(0, 0, q16_fill.get(), q16_fill.get(), MAX_SEQ));

    // Force shadow creation
    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));

    // Wrap with 2 tokens first (head → 2), then query to update shadow.last_head
    auto fp32_half = makeRandomFP32(2, KV_DIM, 56500);
    auto q16_half = quantizeQ16(*fp32_half, HEAD_DIM);
    ASSERT_TRUE(cache.append_kv(0, 0, q16_half.get(), q16_half.get(), 2));
    EXPECT_EQ(cache.ring_head(0, 0), 2);
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));

    // Wrap 2 more → head returns to 0, but shadow.last_head is 2, so reconversion fires
    auto fp32_wrap = makeRandomFP32(2, KV_DIM, 57000);
    auto q16_wrap = quantizeQ16(*fp32_wrap, HEAD_DIM);
    ASSERT_TRUE(cache.append_kv(0, 0, q16_wrap.get(), q16_wrap.get(), 2));
    EXPECT_EQ(cache.ring_head(0, 0), 0);

    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, MAX_SEQ);

    // The last 2 rows should match fp32_wrap
    double cos_new0 = cosineSimilarity(fp32_wrap->data(),
                                       out_k->data() + 2 * KV_DIM, KV_DIM);
    double cos_new1 = cosineSimilarity(fp32_wrap->data() + KV_DIM,
                                       out_k->data() + 3 * KV_DIM, KV_DIM);
    EXPECT_GT(cos_new0, 0.99) << "After head returns to 0, new token 0 cosine: " << cos_new0;
    EXPECT_GT(cos_new1, 0.99) << "After head returns to 0, new token 1 cosine: " << cos_new1;
}

// =========================================================================
// 22. APPEND MORE THAN CAPACITY
// =========================================================================

TEST_F(Test__CPURingKVCache_Q16_1, AppendMoreThanCapacity_KeepsNewest)
{
    constexpr int HEAD_DIM = 64;
    constexpr int N_KV_HEADS = 2;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    constexpr int MAX_SEQ = 4;

    CPURingKVCacheQ16_1 cache(mpi_ctx_, 1, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM,
                              DeviceId::cpu(), KVCacheLayoutMode::HEAD_MAJOR);

    // Append 7 tokens into a cache of capacity 4
    auto fp32 = makeRandomFP32(7, KV_DIM, 58000);
    auto q16 = quantizeQ16(*fp32, HEAD_DIM);
    ASSERT_TRUE(cache.append_kv(0, 0, q16.get(), q16.get(), 7));
    EXPECT_EQ(cache.ring_size(0, 0), MAX_SEQ);

    // Should keep only the last 4 tokens (indices 3-6)
    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, MAX_SEQ);

    for (int i = 0; i < MAX_SEQ; ++i)
    {
        double cos = cosineSimilarity(fp32->data() + (i + 3) * KV_DIM,
                                      out_k->data() + i * KV_DIM, KV_DIM);
        EXPECT_GT(cos, 0.99)
            << "After over-capacity append, row " << i << " should match token " << (i + 3)
            << ". Cosine: " << cos;
    }
}

// =========================================================================
// 23. POSITION_MAJOR: Metadata and append work, but convertNewRows is
//     HEAD_MAJOR-only for Q16_1. Verify creation and append succeed.
// =========================================================================

TEST_F(Test__CPURingKVCache_Q16_1, PositionMajor_CreationAndAppend)
{
    // Q16_1 in POSITION_MAJOR can be created and appended to.
    // However, the convertNewRows dequant path only handles HEAD_MAJOR.
    constexpr int HEAD_DIM = 64;
    constexpr int N_KV_HEADS = 2;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
    constexpr int MAX_SEQ = 8;
    constexpr int N_TOKENS = 3;

    CPURingKVCacheQ16_1 cache(mpi_ctx_, 1, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM,
                              DeviceId::cpu(), KVCacheLayoutMode::POSITION_MAJOR);

    EXPECT_EQ(cache.layout_mode(), KVCacheLayoutMode::POSITION_MAJOR);

    auto fp32 = makeRandomFP32(N_TOKENS, KV_DIM, 59000);
    auto q16 = quantizeQ16(*fp32, HEAD_DIM);
    ASSERT_TRUE(cache.append_kv(0, 0, q16.get(), q16.get(), N_TOKENS));
    EXPECT_EQ(cache.ring_size(0, 0), N_TOKENS);
}

// =========================================================================
// 24. FACTORY CREATION
// =========================================================================

TEST_F(Test__CPURingKVCache_Q16_1, Factory_CreatesQ16_1HeadMajor)
{
    auto cache = createCPURingKVCache(
        ActivationPrecision::Q16_1, mpi_ctx_,
        1, 1, 8, 2, 64, DeviceId::cpu(),
        KVCacheLayoutMode::HEAD_MAJOR);

    ASSERT_NE(cache, nullptr);
    EXPECT_EQ(cache->k_precision(), ActivationPrecision::Q16_1);
    EXPECT_EQ(cache->layout_mode(), KVCacheLayoutMode::HEAD_MAJOR);
}

TEST_F(Test__CPURingKVCache_Q16_1, Factory_Sharded_Q16_1)
{
    auto cache = createShardedCPURingKVCache(
        ActivationPrecision::Q16_1, mpi_ctx_,
        2, 1, 16,
        4, 2, 0,
        64, DeviceId::cpu(),
        KVCacheLayoutMode::HEAD_MAJOR);

    ASSERT_NE(cache, nullptr);
    EXPECT_TRUE(cache->is_sharded());
    EXPECT_EQ(cache->local_n_kv_heads(), 2);
    EXPECT_EQ(cache->kv_head_start(), 0);
    EXPECT_EQ(cache->k_precision(), ActivationPrecision::Q16_1);
}
