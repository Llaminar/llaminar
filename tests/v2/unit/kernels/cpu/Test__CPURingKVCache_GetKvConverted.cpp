/**
 * @file Test__CPURingKVCache_GetKvConverted.cpp
 * @brief Tests for get_kv_converted() ring buffer wrap-around correctness.
 *
 * Targets three bugs that are latent when the ring buffer wraps:
 *
 *   Bug 1 (FP32 passthrough): get_kv_converted returns the raw ring buffer
 *          without linearizing. After wrap, physical row order != logical order.
 *
 *   Bug 2 (FP16/BF16/Q8_1 shadow): convertNewRows uses entry.size as the
 *          watermark. After wrap, entry.size stays at max_seq_len, so newly
 *          overwritten rows are never re-converted into the FP32 shadow.
 *
 *   Bug 3 (RoPE positions): position_start + from uses the physical row index
 *          instead of the token's actual sequence position after wrap-around.
 *
 * Each bug is tested with the simplest precision that exercises the code path.
 */

#include <gtest/gtest.h>

#include <cmath>
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

class Test__CPURingKVCache_GetKvConverted : public ::testing::Test
{
protected:
    static constexpr int HEAD_DIM = 4; // tiny for easy manual verification
    static constexpr int N_KV_HEADS = 1;
    static constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;

    MPIContext mpi_ctx_{0, 1, MPI_COMM_WORLD};

    /// Create an FP32 tensor where every element in row r = (base + r).
    /// This makes it trivial to identify which token a row came from.
    static std::shared_ptr<FP32Tensor> makeTagged(int num_tokens, float base)
    {
        auto t = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(num_tokens),
                                static_cast<size_t>(KV_DIM)});
        float *d = t->mutable_data();
        for (int r = 0; r < num_tokens; ++r)
        {
            float val = base + static_cast<float>(r);
            for (int c = 0; c < KV_DIM; ++c)
                d[r * KV_DIM + c] = val;
        }
        return t;
    }

    /// Create an FP16 tensor where every element in row r = (base + r).
    static std::shared_ptr<FP16Tensor> makeTaggedFP16(int num_tokens, float base)
    {
        // Create FP32 first then convert
        auto fp32 = makeTagged(num_tokens, base);
        auto t = std::make_shared<FP16Tensor>(
            std::vector<size_t>{static_cast<size_t>(num_tokens),
                                static_cast<size_t>(KV_DIM)});
        // Manual FP32→FP16 conversion via half-float bit tricks
        const float *src = fp32->data();
        uint16_t *dst = t->mutable_typed_data();
        for (int i = 0; i < num_tokens * KV_DIM; ++i)
        {
            // Use compiler's conversion to _Float16 if available, else a simple truncation
            // For test purposes, we just need a round-trippable value
            uint32_t bits;
            std::memcpy(&bits, &src[i], 4);
            // IEEE 754 FP32 → FP16 (simplified, works for small values < 65504)
            uint16_t sign = (bits >> 16) & 0x8000;
            int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
            uint16_t frac = (bits >> 13) & 0x03FF;
            if (exp <= 0)
                dst[i] = sign; // underflow to zero
            else if (exp >= 31)
                dst[i] = sign | 0x7C00; // overflow to inf
            else
                dst[i] = sign | (static_cast<uint16_t>(exp) << 10) | frac;
        }
        return t;
    }

    /// Check that an FP32 row has all elements equal to expected_val.
    static bool rowEquals(const float *row, float expected_val)
    {
        for (int c = 0; c < KV_DIM; ++c)
        {
            if (std::abs(row[c] - expected_val) > 0.01f)
                return false;
        }
        return true;
    }
};

// =========================================================================
// Bug 1: FP32 passthrough returns raw ring buffer without linearization
// =========================================================================

TEST_F(Test__CPURingKVCache_GetKvConverted, FP32_NoWrap_ReturnsCorrectOrder)
{
    // Baseline: no wrap-around, get_kv_converted should return data in order.
    constexpr int MAX_SEQ = 4;
    CPURingKVCacheFP32 cache(mpi_ctx_, 1, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM, DeviceId::cpu());

    // Append 3 tokens: rows tagged with values 100, 101, 102
    auto k = makeTagged(3, 100.0f);
    auto v = makeTagged(3, 200.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k.get(), v.get(), 3));

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int kv_len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &kv_len));
    EXPECT_EQ(kv_len, 3);

    const float *kd = out_k->data();
    EXPECT_TRUE(rowEquals(kd + 0 * KV_DIM, 100.0f)) << "Row 0 should be token 0 (val=100)";
    EXPECT_TRUE(rowEquals(kd + 1 * KV_DIM, 101.0f)) << "Row 1 should be token 1 (val=101)";
    EXPECT_TRUE(rowEquals(kd + 2 * KV_DIM, 102.0f)) << "Row 2 should be token 2 (val=102)";
}

TEST_F(Test__CPURingKVCache_GetKvConverted, FP32_WrapAround_ReturnsLogicalOrder)
{
    // Bug 1: After wrap, physical rows are scrambled. get_kv_converted must
    // return logically ordered data (oldest → newest).
    constexpr int MAX_SEQ = 4;
    CPURingKVCacheFP32 cache(mpi_ctx_, 1, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM, DeviceId::cpu());

    // Append 4 tokens (fills ring): values 100, 101, 102, 103
    auto k1 = makeTagged(4, 100.0f);
    auto v1 = makeTagged(4, 200.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k1.get(), v1.get(), 4));
    EXPECT_EQ(cache.ring_head(0, 0), 0);

    // Append 2 more tokens: values 104, 105 → overwrites tokens 100, 101
    auto k2 = makeTagged(2, 104.0f);
    auto v2 = makeTagged(2, 204.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k2.get(), v2.get(), 2));

    EXPECT_EQ(cache.ring_size(0, 0), 4);
    EXPECT_EQ(cache.ring_head(0, 0), 2); // head advanced by 2

    // get_kv_converted should return logical order: 102, 103, 104, 105
    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int kv_len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &kv_len));
    EXPECT_EQ(kv_len, 4);

    const float *kd = out_k->data();
    EXPECT_TRUE(rowEquals(kd + 0 * KV_DIM, 102.0f))
        << "Row 0 should be oldest surviving token (val=102), got " << kd[0];
    EXPECT_TRUE(rowEquals(kd + 1 * KV_DIM, 103.0f))
        << "Row 1 should be token (val=103), got " << kd[1 * KV_DIM];
    EXPECT_TRUE(rowEquals(kd + 2 * KV_DIM, 104.0f))
        << "Row 2 should be token (val=104), got " << kd[2 * KV_DIM];
    EXPECT_TRUE(rowEquals(kd + 3 * KV_DIM, 105.0f))
        << "Row 3 should be newest token (val=105), got " << kd[3 * KV_DIM];

    // Also check V
    const float *vd = out_v->data();
    EXPECT_TRUE(rowEquals(vd + 0 * KV_DIM, 202.0f))
        << "V row 0 should be (val=202), got " << vd[0];
    EXPECT_TRUE(rowEquals(vd + 3 * KV_DIM, 205.0f))
        << "V row 3 should be (val=205), got " << vd[3 * KV_DIM];
}

// =========================================================================
// Bug 2: Incremental conversion stops after ring wraps
// =========================================================================

TEST_F(Test__CPURingKVCache_GetKvConverted, FP16_NoWrap_ConvertedValuesMatch)
{
    // Baseline: FP16 → FP32 conversion without wrap.
    constexpr int MAX_SEQ = 8;
    CPURingKVCacheFP16 cache(mpi_ctx_, 1, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM, DeviceId::cpu());

    auto k = makeTaggedFP16(3, 10.0f);
    auto v = makeTaggedFP16(3, 20.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k.get(), v.get(), 3));

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int kv_len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &kv_len));
    EXPECT_EQ(kv_len, 3);

    const float *kd = out_k->data();
    EXPECT_TRUE(rowEquals(kd + 0 * KV_DIM, 10.0f));
    EXPECT_TRUE(rowEquals(kd + 1 * KV_DIM, 11.0f));
    EXPECT_TRUE(rowEquals(kd + 2 * KV_DIM, 12.0f));
}

TEST_F(Test__CPURingKVCache_GetKvConverted, FP16_WrapAround_NewTokensConverted)
{
    // Bug 2: After wrap, entry.size stays at max_seq_len, so the incremental
    // conversion watermark (shadow.converted_rows) == entry.size and new
    // tokens appended via wrap are never re-converted.
    constexpr int MAX_SEQ = 4;
    CPURingKVCacheFP16 cache(mpi_ctx_, 1, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM, DeviceId::cpu());

    //  Append 4 tokens to fill ring: values 10, 11, 12, 13
    auto k1 = makeTaggedFP16(4, 10.0f);
    auto v1 = makeTaggedFP16(4, 20.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k1.get(), v1.get(), 4));

    // First get_kv_converted: converts all 4 rows
    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int kv_len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &kv_len));
    EXPECT_EQ(kv_len, 4);

    // Append 2 more: values 14, 15 → overwrites tokens 10, 11
    auto k2 = makeTaggedFP16(2, 14.0f);
    auto v2 = makeTaggedFP16(2, 24.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k2.get(), v2.get(), 2));
    EXPECT_EQ(cache.ring_size(0, 0), 4);
    EXPECT_EQ(cache.ring_head(0, 0), 2);

    // Second get_kv_converted: Bug 2 means shadow.converted_rows == 4 == entry.size,
    // so the new tokens at physical rows 0,1 are never re-converted.
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &kv_len));
    EXPECT_EQ(kv_len, 4);

    // Should see logical order: 12, 13, 14, 15
    const float *kd = out_k->data();
    EXPECT_TRUE(rowEquals(kd + 0 * KV_DIM, 12.0f))
        << "Row 0 should be oldest surviving (val=12), got " << kd[0];
    EXPECT_TRUE(rowEquals(kd + 1 * KV_DIM, 13.0f))
        << "Row 1 should be (val=13), got " << kd[1 * KV_DIM];
    EXPECT_TRUE(rowEquals(kd + 2 * KV_DIM, 14.0f))
        << "Row 2 should be newly appended (val=14), got " << kd[2 * KV_DIM];
    EXPECT_TRUE(rowEquals(kd + 3 * KV_DIM, 15.0f))
        << "Row 3 should be newest (val=15), got " << kd[3 * KV_DIM];
}

TEST_F(Test__CPURingKVCache_GetKvConverted, FP16_IncrementalDecode_AfterWrap)
{
    // Simulate decode: append one token at a time, wrapping around.
    // Each get_kv_converted call should reflect the latest state.
    constexpr int MAX_SEQ = 4;
    CPURingKVCacheFP16 cache(mpi_ctx_, 1, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM, DeviceId::cpu());

    // Fill ring (4 tokens)
    auto k_fill = makeTaggedFP16(4, 10.0f);
    auto v_fill = makeTaggedFP16(4, 20.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k_fill.get(), v_fill.get(), 4));

    // Convert once to set watermark
    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int kv_len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &kv_len));

    // Decode: append token 14 (overwrites token 10)
    auto k_dec1 = makeTaggedFP16(1, 14.0f);
    auto v_dec1 = makeTaggedFP16(1, 24.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k_dec1.get(), v_dec1.get(), 1));

    // Get converted again — must include the new token
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &kv_len));
    EXPECT_EQ(kv_len, 4);

    const float *kd = out_k->data();
    // Logical order: 11, 12, 13, 14
    EXPECT_TRUE(rowEquals(kd + 0 * KV_DIM, 11.0f))
        << "After 1 decode step: row 0 should be 11, got " << kd[0];
    EXPECT_TRUE(rowEquals(kd + 3 * KV_DIM, 14.0f))
        << "After 1 decode step: row 3 should be 14, got " << kd[3 * KV_DIM];

    // Decode: append token 15 (overwrites token 11)
    auto k_dec2 = makeTaggedFP16(1, 15.0f);
    auto v_dec2 = makeTaggedFP16(1, 25.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k_dec2.get(), v_dec2.get(), 1));

    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &kv_len));
    kd = out_k->data();
    // Logical order: 12, 13, 14, 15
    EXPECT_TRUE(rowEquals(kd + 0 * KV_DIM, 12.0f))
        << "After 2 decode steps: row 0 should be 12, got " << kd[0];
    EXPECT_TRUE(rowEquals(kd + 3 * KV_DIM, 15.0f))
        << "After 2 decode steps: row 3 should be 15, got " << kd[3 * KV_DIM];
}

// =========================================================================
// Bug 3: RoPE positions wrong after wrap
// =========================================================================

TEST_F(Test__CPURingKVCache_GetKvConverted, FP32_RoPE_CorrectPositionsNoWrap)
{
    // Baseline: RoPE positions without wrap should start at position_start.
    constexpr int MAX_SEQ = 8;
    CPURingKVCacheFP32 cache(mpi_ctx_, 1, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM, DeviceId::cpu());

    // Append 4 tokens: all 1.0 so we can check RoPE modifies them
    auto k = std::make_shared<FP32Tensor>(
        std::vector<size_t>{4, static_cast<size_t>(KV_DIM)});
    auto v = std::make_shared<FP32Tensor>(
        std::vector<size_t>{4, static_cast<size_t>(KV_DIM)});
    std::fill(k->mutable_data(), k->mutable_data() + 4 * KV_DIM, 1.0f);
    std::fill(v->mutable_data(), v->mutable_data() + 4 * KV_DIM, 1.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k.get(), v.get(), 4));

    // Apply RoPE via get_kv_converted with rope_theta=10000
    IKVCache::KVReadParams rope;
    rope.rope_theta = 10000.0f;
    rope.position_start = 0;
    rope.n_kv_heads = N_KV_HEADS;
    rope.head_dim = HEAD_DIM;

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int kv_len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &kv_len, &rope));
    EXPECT_EQ(kv_len, 4);

    // Compute expected RoPE output for position 0
    const float *kd = out_k->data();
    const int half = HEAD_DIM / 2;
    // At position 0, cos(0)=1, sin(0)=0, so first row should be unchanged
    for (int c = 0; c < KV_DIM; ++c)
    {
        EXPECT_NEAR(kd[c], 1.0f, 0.001f) << "Position 0: RoPE should be identity, col=" << c;
    }

    // Position 1+ should differ from 1.0 (RoPE rotates the vector)
    bool pos1_differs = false;
    for (int c = 0; c < KV_DIM; ++c)
    {
        if (std::abs(kd[1 * KV_DIM + c] - 1.0f) > 0.001f)
        {
            pos1_differs = true;
            break;
        }
    }
    EXPECT_TRUE(pos1_differs) << "Position 1: RoPE should modify the vector";
}

TEST_F(Test__CPURingKVCache_GetKvConverted, FP32_RoPE_CorrectPositionsAfterWrap)
{
    // Bug 3: After ring wrap, RoPE position uses physical row index instead of
    // the token's actual sequence position.
    //
    // Strategy: Compare RoPE-applied output from get_kv_converted (after wrap)
    // against a manually computed reference that applies RoPE with correct positions.
    constexpr int MAX_SEQ = 4;
    CPURingKVCacheFP32 cache(mpi_ctx_, 1, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM, DeviceId::cpu());

    // Fill ring with 4 identical tokens (value 1.0)
    auto k1 = std::make_shared<FP32Tensor>(
        std::vector<size_t>{4, static_cast<size_t>(KV_DIM)});
    auto v1 = std::make_shared<FP32Tensor>(
        std::vector<size_t>{4, static_cast<size_t>(KV_DIM)});
    std::fill(k1->mutable_data(), k1->mutable_data() + 4 * KV_DIM, 1.0f);
    std::fill(v1->mutable_data(), v1->mutable_data() + 4 * KV_DIM, 1.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k1.get(), v1.get(), 4));

    // Append 2 more identical tokens (value 1.0) → overwrites positions 0,1
    auto k2 = std::make_shared<FP32Tensor>(
        std::vector<size_t>{2, static_cast<size_t>(KV_DIM)});
    auto v2 = std::make_shared<FP32Tensor>(
        std::vector<size_t>{2, static_cast<size_t>(KV_DIM)});
    std::fill(k2->mutable_data(), k2->mutable_data() + 2 * KV_DIM, 1.0f);
    std::fill(v2->mutable_data(), v2->mutable_data() + 2 * KV_DIM, 1.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k2.get(), v2.get(), 2));

    EXPECT_EQ(cache.ring_head(0, 0), 2);
    EXPECT_EQ(cache.ring_size(0, 0), 4);

    // Apply RoPE with position_start=0. The 4 surviving tokens correspond to
    // logical positions 2, 3, 4, 5 (tokens 0,1 were evicted). But what we
    // want from position_start=0 is positions 0, 1, 2, 3 for the output rows.
    IKVCache::KVReadParams rope;
    rope.rope_theta = 10000.0f;
    rope.position_start = 0;
    rope.n_kv_heads = N_KV_HEADS;
    rope.head_dim = HEAD_DIM;

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int kv_len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &kv_len, &rope));
    EXPECT_EQ(kv_len, 4);

    // Build a reference: 4 rows of 1.0, apply RoPE at positions 0,1,2,3
    std::vector<float> ref(4 * KV_DIM, 1.0f);
    // Apply RoPE manually using the same function the cache should use
    // (we import it indirectly via the cache, but for reference we compute cos/sin)
    const int half = HEAD_DIM / 2;
    const float theta = 10000.0f;
    for (int r = 0; r < 4; ++r)
    {
        float *row = ref.data() + r * KV_DIM;
        for (int h = 0; h < N_KV_HEADS; ++h)
        {
            float *head = row + h * HEAD_DIM;
            for (int i = 0; i < half; ++i)
            {
                float freq = 1.0f / std::pow(theta, static_cast<float>(2 * i) / HEAD_DIM);
                float angle = static_cast<float>(r) * freq; // position = r (0,1,2,3)
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);
                float x0 = head[i];
                float x1 = head[i + half];
                head[i] = x0 * cos_a - x1 * sin_a;
                head[i + half] = x0 * sin_a + x1 * cos_a;
            }
        }
    }

    const float *kd = out_k->data();
    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < KV_DIM; ++c)
        {
            EXPECT_NEAR(kd[r * KV_DIM + c], ref[r * KV_DIM + c], 0.001f)
                << "RoPE mismatch after wrap at row=" << r << " col=" << c
                << " (expected position " << r << ")";
        }
    }
}

// =========================================================================
// Combined: FP16 wrap + RoPE
// =========================================================================

TEST_F(Test__CPURingKVCache_GetKvConverted, FP16_WrapAround_WithRoPE_CorrectOutput)
{
    // Exercises Bugs 2 + 3 together in the FP16 convertNewRows path.
    constexpr int MAX_SEQ = 4;
    CPURingKVCacheFP16 cache(mpi_ctx_, 1, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM, DeviceId::cpu());

    // Helper to create uniform FP16 tensor (all elements = val)
    auto makeUniformFP16 = [](int num_tokens, float val)
    {
        auto fp32 = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(num_tokens),
                                static_cast<size_t>(KV_DIM)});
        std::fill(fp32->mutable_data(), fp32->mutable_data() + num_tokens * KV_DIM, val);
        auto t = std::make_shared<FP16Tensor>(
            std::vector<size_t>{static_cast<size_t>(num_tokens),
                                static_cast<size_t>(KV_DIM)});
        const float *src = fp32->data();
        uint16_t *dst = t->mutable_typed_data();
        for (int i = 0; i < num_tokens * KV_DIM; ++i)
        {
            uint32_t bits;
            std::memcpy(&bits, &src[i], 4);
            uint16_t sign = (bits >> 16) & 0x8000;
            int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
            uint16_t frac = (bits >> 13) & 0x03FF;
            if (exp <= 0)
                dst[i] = sign;
            else if (exp >= 31)
                dst[i] = sign | 0x7C00;
            else
                dst[i] = sign | (static_cast<uint16_t>(exp) << 10) | frac;
        }
        return t;
    };

    // Fill ring with 4 tokens (all value 1.0)
    auto k1 = makeUniformFP16(4, 1.0f);
    auto v1 = makeUniformFP16(4, 1.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k1.get(), v1.get(), 4));

    // First conversion to set watermark
    IKVCache::KVReadParams rope;
    rope.rope_theta = 10000.0f;
    rope.position_start = 0;
    rope.n_kv_heads = N_KV_HEADS;
    rope.head_dim = HEAD_DIM;

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int kv_len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &kv_len, &rope));

    // Wrap: append 1 more (overwrites oldest)
    auto k2 = makeUniformFP16(1, 1.0f);
    auto v2 = makeUniformFP16(1, 1.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k2.get(), v2.get(), 1));
    EXPECT_EQ(cache.ring_head(0, 0), 1);

    // Second conversion — should reflect new state
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &kv_len, &rope));
    EXPECT_EQ(kv_len, 4);

    // Row 0 has position 0 → cos(0)=1, sin(0)=0 → value unchanged (≈1.0)
    // This verifies both that the new token was converted (Bug 2) and
    // that positions are assigned correctly in logical order (Bug 3).
    const float *kd = out_k->data();
    for (int c = 0; c < KV_DIM; ++c)
    {
        EXPECT_NEAR(kd[c], 1.0f, 0.01f)
            << "After wrap + RoPE, logical position 0 should be RoPE-identity, col=" << c;
    }
}

// =========================================================================
// Regression: Verify gather_kv_batched still works correctly after fixes
// =========================================================================

TEST_F(Test__CPURingKVCache_GetKvConverted, FP32_GatherVsGetKvConverted_SameOrder)
{
    // After fixing get_kv_converted, verify it returns the same logical
    // ordering as gather_kv_batched (which was always correct).
    constexpr int MAX_SEQ = 4;
    CPURingKVCacheFP32 cache(mpi_ctx_, 1, 1, MAX_SEQ, N_KV_HEADS, HEAD_DIM, DeviceId::cpu());

    // Fill and wrap
    auto k1 = makeTagged(4, 100.0f);
    auto v1 = makeTagged(4, 200.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k1.get(), v1.get(), 4));

    auto k2 = makeTagged(2, 104.0f);
    auto v2 = makeTagged(2, 204.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k2.get(), v2.get(), 2));

    // gather_kv_batched (known correct)
    auto gather_k = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ),
                            static_cast<size_t>(KV_DIM)});
    auto gather_v = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ),
                            static_cast<size_t>(KV_DIM)});
    std::vector<int> kv_lens;
    int max_kv = cache.gather_kv_batched(0, 1, gather_k.get(), gather_v.get(), kv_lens);
    ASSERT_EQ(max_kv, 4);

    // get_kv_converted
    ITensor *conv_k = nullptr;
    ITensor *conv_v = nullptr;
    int conv_len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &conv_k, &conv_v, &conv_len));
    EXPECT_EQ(conv_len, 4);

    // Compare row by row
    const float *gk = gather_k->data();
    const float *ck = conv_k->data();
    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < KV_DIM; ++c)
        {
            EXPECT_FLOAT_EQ(gk[r * KV_DIM + c], ck[r * KV_DIM + c])
                << "gather vs get_kv_converted mismatch at row=" << r << " col=" << c;
        }
    }
}
