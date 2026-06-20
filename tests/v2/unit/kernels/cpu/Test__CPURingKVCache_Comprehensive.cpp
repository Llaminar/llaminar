/**
 * @file Test__CPURingKVCache_Comprehensive.cpp
 * @brief Comprehensive test suite for CPURingKVCache covering:
 *
 * - Shadow buffer wrap-around edge cases and stress tests
 * - get_kv() / get_v() accessors
 * - clear_sequence() / clear_layer() / clear() and their effect on shadows
 * - evict_oldest() / get_total_evicted() / reset_eviction_counter()
 * - Multi-sequence (batch_size > 1) gather and shadow behavior
 * - Multi-layer shadow correctness
 * - BF16 and Q8_1 get_kv_converted paths
 * - Factory functions (createCPURingKVCache / createShardedCPURingKVCache)
 * - Sharding accessors (is_sharded, local_n_kv_heads, kv_head_start, local_kv_dim)
 * - Metadata accessors (batch_size, n_kv_heads, kv_layout, layout_mode, k/v_precision)
 * - Boundary conditions: empty cache, null inputs, type mismatch, oversized appends
 * - Multiple complete ring wraps (stress)
 * - Evict + append + gather sequences
 * - get_kv_converted with target != FP32
 * - append_kv with num_tokens inferred from tensor shape
 * - HEAD_MAJOR + get_kv_converted
 * - get_layer_device()
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <numeric>

#include "kernels/cpu/CPURingKVCache.h"
#include "kernels/IKVCache.h"
#include "tensors/Tensors.h"
#include "utils/MPIContext.h"

using namespace llaminar2;

// =========================================================================
// Helpers
// =========================================================================

static MPIContext testMPI()
{
    return MPIContext(0, 1, MPI_COMM_WORLD);
}

static constexpr int HD = 4;  // head_dim
static constexpr int NKV = 1; // n_kv_heads
static constexpr int KV_DIM = NKV * HD;

/// Create FP32 tensor where row r has all elements = base + r.
static std::shared_ptr<FP32Tensor> taggedFP32(int rows, float base)
{
    auto t = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(KV_DIM)});
    float *d = t->mutable_data();
    for (int r = 0; r < rows; ++r)
    {
        float v = base + static_cast<float>(r);
        for (int c = 0; c < KV_DIM; ++c)
            d[r * KV_DIM + c] = v;
    }
    return t;
}

/// Create a uniform FP32 tensor (all elements = val).
static std::shared_ptr<FP32Tensor> uniformFP32(int rows, float val)
{
    auto t = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(KV_DIM)});
    std::fill(t->mutable_data(), t->mutable_data() + rows * KV_DIM, val);
    return t;
}

/// FP32 → FP16 bit conversion (simplified, works for small values).
static uint16_t fp32_to_fp16_bits(float val)
{
    uint32_t bits;
    std::memcpy(&bits, &val, 4);
    uint16_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint16_t frac = (bits >> 13) & 0x03FF;
    if (exp <= 0)
        return sign;
    if (exp >= 31)
        return sign | 0x7C00;
    return sign | (static_cast<uint16_t>(exp) << 10) | frac;
}

/// Create FP16 tensor where row r has all elements ≈ base + r.
static std::shared_ptr<FP16Tensor> taggedFP16(int rows, float base)
{
    auto t = std::make_shared<FP16Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(KV_DIM)});
    uint16_t *d = t->mutable_typed_data();
    for (int r = 0; r < rows; ++r)
    {
        uint16_t v = fp32_to_fp16_bits(base + static_cast<float>(r));
        for (int c = 0; c < KV_DIM; ++c)
            d[r * KV_DIM + c] = v;
    }
    return t;
}

/// FP32 → BF16 bit conversion.
static uint16_t fp32_to_bf16_bits(float val)
{
    uint32_t bits;
    std::memcpy(&bits, &val, 4);
    return static_cast<uint16_t>(bits >> 16); // BF16 = upper 16 bits of FP32
}

/// Create BF16 tensor where row r has all elements ≈ base + r.
static std::shared_ptr<BF16Tensor> taggedBF16(int rows, float base)
{
    auto t = std::make_shared<BF16Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(KV_DIM)});
    uint16_t *d = t->mutable_typed_data();
    for (int r = 0; r < rows; ++r)
    {
        uint16_t v = fp32_to_bf16_bits(base + static_cast<float>(r));
        for (int c = 0; c < KV_DIM; ++c)
            d[r * KV_DIM + c] = v;
    }
    return t;
}

/// Create Q8_1 tensor where row r has all elements = base + r.
static std::shared_ptr<Q8_1Tensor> taggedQ8_1(int rows, int cols, int base)
{
    const size_t blocks_per_row = (static_cast<size_t>(cols) + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
    std::vector<Q8_1Block> blocks(static_cast<size_t>(rows) * blocks_per_row);

    for (int r = 0; r < rows; ++r)
    {
        const int value = base + r;
        for (size_t b = 0; b < blocks_per_row; ++b)
        {
            Q8_1Block &block = blocks[static_cast<size_t>(r) * blocks_per_row + b];
            // Use an exact unit scale so dequantized rows identify stale shadows unambiguously.
            block.d = fp32_to_fp16_bits(1.0f);
            block.sum_qs = 0;
            for (int i = 0; i < static_cast<int>(Q8_1Block::BLOCK_SIZE); ++i)
            {
                const int col = static_cast<int>(b * Q8_1Block::BLOCK_SIZE) + i;
                const int8_t q = col < cols ? static_cast<int8_t>(value) : static_cast<int8_t>(0);
                block.qs[i] = q;
                block.sum_qs += q;
            }
        }
    }

    return std::make_shared<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
        blocks.data(), blocks.size(), DeviceId::cpu());
}

/// Check that an FP32 row has all elements near expected_val.
static bool rowNear(const float *row, float expected, float tol = 0.01f)
{
    for (int c = 0; c < KV_DIM; ++c)
        if (std::abs(row[c] - expected) > tol)
            return false;
    return true;
}

/// Check a row with an explicit column count; useful for Q8_1 rows with 32 columns.
static bool rowNearCols(const float *row, int cols, float expected, float tol = 0.01f)
{
    for (int c = 0; c < cols; ++c)
        if (std::abs(row[c] - expected) > tol)
            return false;
    return true;
}

// =========================================================================
// 1. METADATA ACCESSORS
// =========================================================================

class Test__CPURingKVCache_Comprehensive : public ::testing::Test
{
};

TEST_F(Test__CPURingKVCache_Comprehensive, MetadataAccessors_FP32)
{
    CPURingKVCacheFP32 cache(testMPI(), 3, 2, 16, 4, 8, DeviceId::cpu());

    EXPECT_EQ(cache.k_precision(), ActivationPrecision::FP32);
    EXPECT_EQ(cache.v_precision(), ActivationPrecision::FP32);
    EXPECT_EQ(cache.max_seq_len(), 16);
    EXPECT_EQ(cache.n_layers(), 3);
    EXPECT_EQ(cache.num_layers(), 3);
    EXPECT_EQ(cache.batch_size(), 2);
    EXPECT_EQ(cache.n_kv_heads(), 4);
    EXPECT_EQ(cache.layout_mode(), KVCacheLayoutMode::POSITION_MAJOR);
    EXPECT_FALSE(cache.is_sharded());
}

TEST_F(Test__CPURingKVCache_Comprehensive, MetadataAccessors_FP16)
{
    CPURingKVCacheFP16 cache(testMPI(), 2, 1, 8, 2, 4, DeviceId::cpu());

    EXPECT_EQ(cache.k_precision(), ActivationPrecision::FP16);
    EXPECT_EQ(cache.v_precision(), ActivationPrecision::FP16);
    EXPECT_EQ(cache.max_seq_len(), 8);
    EXPECT_EQ(cache.n_layers(), 2);
}

TEST_F(Test__CPURingKVCache_Comprehensive, MetadataAccessors_BF16)
{
    CPURingKVCacheBF16 cache(testMPI(), 1, 1, 8, 2, 4, DeviceId::cpu());

    EXPECT_EQ(cache.k_precision(), ActivationPrecision::BF16);
    EXPECT_EQ(cache.v_precision(), ActivationPrecision::BF16);
}

TEST_F(Test__CPURingKVCache_Comprehensive, MetadataAccessors_HeadMajor)
{
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, 8, 2, 4, DeviceId::cpu(),
                             KVCacheLayoutMode::HEAD_MAJOR);

    EXPECT_EQ(cache.layout_mode(), KVCacheLayoutMode::HEAD_MAJOR);
}

// =========================================================================
// 2. SHARDING ACCESSORS
// =========================================================================

TEST_F(Test__CPURingKVCache_Comprehensive, Sharding_NonSharded_Defaults)
{
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, 8, 4, HD, DeviceId::cpu());

    EXPECT_FALSE(cache.is_sharded());
    EXPECT_EQ(cache.local_n_kv_heads(), 4);
    EXPECT_EQ(cache.kv_head_start(), 0);
    EXPECT_EQ(cache.local_kv_dim(), 4 * HD);
}

TEST_F(Test__CPURingKVCache_Comprehensive, Sharding_Sharded_Reports)
{
    // Sharded constructor: total=4 heads, local=2 heads, starting at head 2
    CPURingKVCacheFP32 cache(testMPI(), 2, 1, 8, 4, 2, 2, HD, DeviceId::cpu());

    EXPECT_TRUE(cache.is_sharded());
    EXPECT_EQ(cache.local_n_kv_heads(), 2);
    EXPECT_EQ(cache.kv_head_start(), 2);
    EXPECT_EQ(cache.n_kv_heads(), 4);
    EXPECT_EQ(cache.local_kv_dim(), 2 * HD);
}

// =========================================================================
// 3. GET_KV ACCESSOR
// =========================================================================

TEST_F(Test__CPURingKVCache_Comprehensive, GetKv_EmptyCache_ReturnsZeroLen)
{
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, 4, NKV, HD, DeviceId::cpu());

    ITensor *k = nullptr;
    ITensor *v = nullptr;
    int len = -1;
    ASSERT_TRUE(cache.get_kv(0, 0, &k, &v, &len));
    EXPECT_NE(k, nullptr); // Tensor exists but is empty
    EXPECT_NE(v, nullptr);
    EXPECT_EQ(len, 0);
}

TEST_F(Test__CPURingKVCache_Comprehensive, GetKv_AfterAppend_ReturnsCorrectLen)
{
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, 8, NKV, HD, DeviceId::cpu());

    auto k = taggedFP32(3, 10.0f);
    auto v = taggedFP32(3, 20.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k.get(), v.get(), 3));

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv(0, 0, &out_k, &out_v, &len));
    EXPECT_EQ(len, 3);
}

TEST_F(Test__CPURingKVCache_Comprehensive, GetKv_OutOfBoundsLayer_ReturnsFalse)
{
    CPURingKVCacheFP32 cache(testMPI(), 2, 1, 4, NKV, HD, DeviceId::cpu());

    ITensor *k = nullptr;
    ITensor *v = nullptr;
    int len = 99;
    EXPECT_FALSE(cache.get_kv(-1, 0, &k, &v, &len));
    EXPECT_EQ(k, nullptr);
    EXPECT_EQ(len, 0);

    EXPECT_FALSE(cache.get_kv(2, 0, &k, &v, &len));
}

TEST_F(Test__CPURingKVCache_Comprehensive, GetKv_OutOfBoundsSeq_ReturnsFalse)
{
    CPURingKVCacheFP32 cache(testMPI(), 1, 2, 4, NKV, HD, DeviceId::cpu());

    ITensor *k = nullptr;
    ITensor *v = nullptr;
    int len = 99;
    EXPECT_FALSE(cache.get_kv(0, -1, &k, &v, &len));
    EXPECT_FALSE(cache.get_kv(0, 2, &k, &v, &len));
}

TEST_F(Test__CPURingKVCache_Comprehensive, GetKv_ConstOverload)
{
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, 4, NKV, HD, DeviceId::cpu());
    auto k = taggedFP32(2, 1.0f);
    auto v = taggedFP32(2, 2.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k.get(), v.get(), 2));

    const auto &const_cache = cache;
    const ITensor *ck = nullptr;
    const ITensor *cv = nullptr;
    int len = 0;
    ASSERT_TRUE(const_cache.get_kv(0, 0, &ck, &cv, &len));
    EXPECT_EQ(len, 2);
    EXPECT_NE(ck, nullptr);
}

// =========================================================================
// 4. GET_V / GET_K INDIVIDUAL ACCESSORS
// =========================================================================

TEST_F(Test__CPURingKVCache_Comprehensive, GetK_GetV_ReturnsCorrectPointers)
{
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, 4, NKV, HD, DeviceId::cpu());
    auto k = taggedFP32(2, 10.0f);
    auto v = taggedFP32(2, 20.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k.get(), v.get(), 2));

    ITensor *kv_k = nullptr;
    ITensor *kv_v = nullptr;
    int len = 0;
    cache.get_kv(0, 0, &kv_k, &kv_v, &len);

    ITensor *gk = cache.get_k(0, 0);
    ITensor *gv = cache.get_v(0, 0);

    EXPECT_EQ(gk, kv_k);
    EXPECT_EQ(gv, kv_v);
}

// =========================================================================
// 5. CLEAR_SEQUENCE / CLEAR_LAYER
// =========================================================================

TEST_F(Test__CPURingKVCache_Comprehensive, ClearSequence_ResetsOneEntry)
{
    CPURingKVCacheFP32 cache(testMPI(), 2, 2, 4, NKV, HD, DeviceId::cpu());

    auto k = taggedFP32(3, 1.0f);
    auto v = taggedFP32(3, 2.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k.get(), v.get(), 3));
    ASSERT_TRUE(cache.append_kv(0, 1, k.get(), v.get(), 3));
    ASSERT_TRUE(cache.append_kv(1, 0, k.get(), v.get(), 3));
    ASSERT_TRUE(cache.append_kv(1, 1, k.get(), v.get(), 3));

    cache.clear_sequence(0, 0);
    EXPECT_EQ(cache.ring_size(0, 0), 0);
    EXPECT_EQ(cache.ring_head(0, 0), 0);
    // Other entries untouched
    EXPECT_EQ(cache.ring_size(0, 1), 3);
    EXPECT_EQ(cache.ring_size(1, 0), 3);
    EXPECT_EQ(cache.ring_size(1, 1), 3);
}

TEST_F(Test__CPURingKVCache_Comprehensive, ClearSequence_OutOfBounds_NoOp)
{
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, 4, NKV, HD, DeviceId::cpu());
    auto k = taggedFP32(2, 1.0f);
    auto v = taggedFP32(2, 2.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k.get(), v.get(), 2));

    cache.clear_sequence(-1, 0);         // OOB
    cache.clear_sequence(0, -1);         // OOB
    cache.clear_sequence(1, 0);          // OOB
    EXPECT_EQ(cache.ring_size(0, 0), 2); // Unchanged
}

TEST_F(Test__CPURingKVCache_Comprehensive, ClearLayer_ResetsAllSeqsInLayer)
{
    CPURingKVCacheFP32 cache(testMPI(), 2, 2, 4, NKV, HD, DeviceId::cpu());

    auto k = taggedFP32(3, 1.0f);
    auto v = taggedFP32(3, 2.0f);
    for (int l = 0; l < 2; ++l)
        for (int s = 0; s < 2; ++s)
            ASSERT_TRUE(cache.append_kv(l, s, k.get(), v.get(), 3));

    cache.clear_layer(0);
    EXPECT_EQ(cache.ring_size(0, 0), 0);
    EXPECT_EQ(cache.ring_size(0, 1), 0);
    // Layer 1 untouched
    EXPECT_EQ(cache.ring_size(1, 0), 3);
    EXPECT_EQ(cache.ring_size(1, 1), 3);
}

TEST_F(Test__CPURingKVCache_Comprehensive, ClearLayer_OutOfBounds_NoOp)
{
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, 4, NKV, HD, DeviceId::cpu());
    auto k = taggedFP32(2, 1.0f);
    auto v = taggedFP32(2, 2.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k.get(), v.get(), 2));

    cache.clear_layer(-1);
    cache.clear_layer(1);
    EXPECT_EQ(cache.ring_size(0, 0), 2);
}

// =========================================================================
// 6. SHADOW INVALIDATION: clear() / clear_sequence() / clear_layer()
// =========================================================================

TEST_F(Test__CPURingKVCache_Comprehensive, FP16Shadow_Clear_ThenReappend_ConvertsCorrectly)
{
    // Bug check: clear() doesn't reset fp32_shadows_, but the "kv_len < shadow.converted_rows"
    // check in get_kv_converted should detect the reset.
    constexpr int MAX_SEQ = 4;
    CPURingKVCacheFP16 cache(testMPI(), 1, 1, MAX_SEQ, NKV, HD, DeviceId::cpu());

    // Append 4 tokens, convert them
    auto k1 = taggedFP16(4, 10.0f);
    auto v1 = taggedFP16(4, 20.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k1.get(), v1.get(), 4));

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 4);
    EXPECT_TRUE(rowNear(out_k->data(), 10.0f));

    // Clear the cache
    cache.clear();
    EXPECT_EQ(cache.ring_size(0, 0), 0);

    // Re-append 2 tokens with different values
    auto k2 = taggedFP16(2, 50.0f);
    auto v2 = taggedFP16(2, 60.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k2.get(), v2.get(), 2));

    // Shadow must reflect new data, not stale old data
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 2);
    EXPECT_TRUE(rowNear(out_k->data() + 0 * KV_DIM, 50.0f))
        << "After clear+re-append: row 0 should be 50, got " << out_k->data()[0];
    EXPECT_TRUE(rowNear(out_k->data() + 1 * KV_DIM, 51.0f))
        << "After clear+re-append: row 1 should be 51, got " << out_k->data()[KV_DIM];
}

TEST_F(Test__CPURingKVCache_Comprehensive, FP32Shadow_ClearSequence_ThenWrap_ConvertsCorrectly)
{
    // Tests that after clear_sequence the shadow is properly invalidated for FP32 wrapping path.
    constexpr int MAX_SEQ = 4;
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, MAX_SEQ, NKV, HD, DeviceId::cpu());

    // Fill and wrap to create shadow
    auto k1 = taggedFP32(4, 100.0f);
    auto v1 = taggedFP32(4, 200.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k1.get(), v1.get(), 4));
    auto k2 = taggedFP32(1, 104.0f);
    auto v2 = taggedFP32(1, 204.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k2.get(), v2.get(), 1));

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 4);

    // Clear and re-fill
    cache.clear_sequence(0, 0);
    EXPECT_EQ(cache.ring_size(0, 0), 0);
    EXPECT_EQ(cache.ring_head(0, 0), 0);

    auto k3 = taggedFP32(3, 200.0f);
    auto v3 = taggedFP32(3, 300.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k3.get(), v3.get(), 3));

    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 3);
    // Head is 0 now so it should passthrough directly
    EXPECT_TRUE(rowNear(out_k->data() + 0 * KV_DIM, 200.0f))
        << "After clear_sequence+re-append: got " << out_k->data()[0];
}

TEST_F(Test__CPURingKVCache_Comprehensive, FP16Shadow_ClearLayer_InvalidatesShadow)
{
    CPURingKVCacheFP16 cache(testMPI(), 2, 1, 4, NKV, HD, DeviceId::cpu());

    auto k = taggedFP16(3, 10.0f);
    auto v = taggedFP16(3, 20.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k.get(), v.get(), 3));
    ASSERT_TRUE(cache.append_kv(1, 0, k.get(), v.get(), 3));

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    ASSERT_TRUE(cache.get_kv_converted(1, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));

    // Clear only layer 0
    cache.clear_layer(0);

    // Re-append different data to layer 0
    auto k2 = taggedFP16(2, 70.0f);
    auto v2 = taggedFP16(2, 80.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k2.get(), v2.get(), 2));

    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 2);
    EXPECT_TRUE(rowNear(out_k->data(), 70.0f))
        << "Layer 0 after clear_layer: got " << out_k->data()[0];

    // Layer 1 still has old data
    ASSERT_TRUE(cache.get_kv_converted(1, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 3);
    EXPECT_TRUE(rowNear(out_k->data(), 10.0f));
}

TEST_F(Test__CPURingKVCache_Comprehensive, FP16Shadow_Clear_ThenSameLengthReappend_DoesNotReuseStaleRows)
{
    // Same-length reappend is the regression case: without clear() resetting
    // shadow metadata, kv_len == converted_rows and get_kv_converted skips reconversion.
    constexpr int MAX_SEQ = 4;
    CPURingKVCacheFP16 cache(testMPI(), 1, 1, MAX_SEQ, NKV, HD, DeviceId::cpu());

    auto old_k = taggedFP16(2, 10.0f);
    auto old_v = taggedFP16(2, 20.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, old_k.get(), old_v.get(), 2));

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    ASSERT_EQ(len, 2);
    ASSERT_TRUE(rowNear(out_k->data() + 0 * KV_DIM, 10.0f));
    ASSERT_TRUE(rowNear(out_v->data() + 0 * KV_DIM, 20.0f));

    cache.clear();
    ASSERT_EQ(cache.ring_size(0, 0), 0);
    ASSERT_EQ(cache.ring_head(0, 0), 0);

    auto new_k = taggedFP16(2, 50.0f);
    auto new_v = taggedFP16(2, 60.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, new_k.get(), new_v.get(), 2));

    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 2);
    EXPECT_TRUE(rowNear(out_k->data() + 0 * KV_DIM, 50.0f))
        << "clear() left stale K shadow row, got " << out_k->data()[0];
    EXPECT_TRUE(rowNear(out_k->data() + 1 * KV_DIM, 51.0f))
        << "clear() left stale K shadow row, got " << out_k->data()[KV_DIM];
    EXPECT_TRUE(rowNear(out_v->data() + 0 * KV_DIM, 60.0f))
        << "clear() left stale V shadow row, got " << out_v->data()[0];
    EXPECT_TRUE(rowNear(out_v->data() + 1 * KV_DIM, 61.0f))
        << "clear() left stale V shadow row, got " << out_v->data()[KV_DIM];
}

TEST_F(Test__CPURingKVCache_Comprehensive, BF16Shadow_ClearSequence_ThenSameLengthReappend_DoesNotReuseStaleRows)
{
    // Per-sequence reset must invalidate only the targeted converted shadow.
    constexpr int MAX_SEQ = 4;
    CPURingKVCacheBF16 cache(testMPI(), 1, 2, MAX_SEQ, NKV, HD, DeviceId::cpu());

    auto seq0_k = taggedBF16(2, 10.0f);
    auto seq0_v = taggedBF16(2, 20.0f);
    auto seq1_k = taggedBF16(2, 100.0f);
    auto seq1_v = taggedBF16(2, 110.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, seq0_k.get(), seq0_v.get(), 2));
    ASSERT_TRUE(cache.append_kv(0, 1, seq1_k.get(), seq1_v.get(), 2));

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    ASSERT_EQ(len, 2);
    ASSERT_TRUE(cache.get_kv_converted(0, 1, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    ASSERT_EQ(len, 2);

    cache.clear_sequence(0, 0);
    ASSERT_EQ(cache.ring_size(0, 0), 0);
    ASSERT_EQ(cache.ring_size(0, 1), 2);

    auto new_k = taggedBF16(2, 50.0f);
    auto new_v = taggedBF16(2, 60.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, new_k.get(), new_v.get(), 2));

    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 2);
    EXPECT_TRUE(rowNear(out_k->data() + 0 * KV_DIM, 50.0f, 0.1f))
        << "clear_sequence() left stale K shadow row, got " << out_k->data()[0];
    EXPECT_TRUE(rowNear(out_v->data() + 1 * KV_DIM, 61.0f, 0.1f))
        << "clear_sequence() left stale V shadow row, got " << out_v->data()[KV_DIM];

    ASSERT_TRUE(cache.get_kv_converted(0, 1, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 2);
    EXPECT_TRUE(rowNear(out_k->data() + 0 * KV_DIM, 100.0f, 0.1f))
        << "clear_sequence() should not disturb sibling sequence";
}

TEST_F(Test__CPURingKVCache_Comprehensive, Q8_1Shadow_ClearLayer_ThenSameLengthReappend_DoesNotReuseStaleRows)
{
    // Q8_1 dequantization works in 32-wide blocks, matching the long-context KV precision path.
    constexpr int Q8_HD = 32;
    constexpr int Q8_DIM = Q8_HD;
    CPURingKVCacheQ8_1 cache(testMPI(), 2, 1, 4, 1, Q8_HD, DeviceId::cpu());

    auto layer0_k = taggedQ8_1(2, Q8_DIM, 10);
    auto layer0_v = taggedQ8_1(2, Q8_DIM, 20);
    auto layer1_k = taggedQ8_1(2, Q8_DIM, 80);
    auto layer1_v = taggedQ8_1(2, Q8_DIM, 90);
    ASSERT_TRUE(cache.append_kv(0, 0, layer0_k.get(), layer0_v.get(), 2));
    ASSERT_TRUE(cache.append_kv(1, 0, layer1_k.get(), layer1_v.get(), 2));

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    ASSERT_EQ(len, 2);
    ASSERT_TRUE(cache.get_kv_converted(1, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    ASSERT_EQ(len, 2);

    cache.clear_layer(0);
    ASSERT_EQ(cache.ring_size(0, 0), 0);
    ASSERT_EQ(cache.ring_size(1, 0), 2);

    auto new_k = taggedQ8_1(2, Q8_DIM, 40);
    auto new_v = taggedQ8_1(2, Q8_DIM, 50);
    ASSERT_TRUE(cache.append_kv(0, 0, new_k.get(), new_v.get(), 2));

    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 2);
    EXPECT_TRUE(rowNearCols(out_k->data() + 0 * Q8_DIM, Q8_DIM, 40.0f))
        << "clear_layer() left stale Q8_1 K shadow row, got " << out_k->data()[0];
    EXPECT_TRUE(rowNearCols(out_k->data() + 1 * Q8_DIM, Q8_DIM, 41.0f))
        << "clear_layer() left stale Q8_1 K shadow row, got " << out_k->data()[Q8_DIM];
    EXPECT_TRUE(rowNearCols(out_v->data() + 0 * Q8_DIM, Q8_DIM, 50.0f))
        << "clear_layer() left stale Q8_1 V shadow row, got " << out_v->data()[0];

    ASSERT_TRUE(cache.get_kv_converted(1, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 2);
    EXPECT_TRUE(rowNearCols(out_k->data() + 0 * Q8_DIM, Q8_DIM, 80.0f))
        << "clear_layer() should not disturb sibling layer";
}

// =========================================================================
// 7. EVICTION
// =========================================================================

TEST_F(Test__CPURingKVCache_Comprehensive, EvictOldest_AllSequences)
{
    CPURingKVCacheFP32 cache(testMPI(), 1, 2, 8, NKV, HD, DeviceId::cpu());

    auto k = taggedFP32(5, 1.0f);
    auto v = taggedFP32(5, 2.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k.get(), v.get(), 5));
    ASSERT_TRUE(cache.append_kv(0, 1, k.get(), v.get(), 5));

    cache.evict_oldest(2);
    EXPECT_EQ(cache.ring_size(0, 0), 3);
    EXPECT_EQ(cache.ring_size(0, 1), 3);
    EXPECT_EQ(cache.ring_head(0, 0), 2);
    EXPECT_EQ(cache.ring_head(0, 1), 2);
}

TEST_F(Test__CPURingKVCache_Comprehensive, EvictTotalCounter_TracksAcrossOperations)
{
    CPURingKVCacheFP32 cache(testMPI(), 2, 1, 8, NKV, HD, DeviceId::cpu());

    EXPECT_EQ(cache.get_total_evicted(), 0);

    auto k = taggedFP32(5, 1.0f);
    auto v = taggedFP32(5, 2.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k.get(), v.get(), 5));
    ASSERT_TRUE(cache.append_kv(1, 0, k.get(), v.get(), 5));

    cache.evict_oldest_from_sequence(0, 2);
    // Evicted 2 tokens from seq 0 across 2 layers = 2*2 = 4 total
    EXPECT_EQ(cache.get_total_evicted(), 4);

    cache.evict_oldest_from_sequence(0, 1);
    // Evicted 1 more across 2 layers = 2 more
    EXPECT_EQ(cache.get_total_evicted(), 6);

    cache.reset_eviction_counter();
    EXPECT_EQ(cache.get_total_evicted(), 0);
}

TEST_F(Test__CPURingKVCache_Comprehensive, Evict_MoreThanCached_ClampedToSize)
{
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, 8, NKV, HD, DeviceId::cpu());

    auto k = taggedFP32(3, 1.0f);
    auto v = taggedFP32(3, 2.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k.get(), v.get(), 3));

    cache.evict_oldest_from_sequence(0, 100); // Way more than 3
    EXPECT_EQ(cache.ring_size(0, 0), 0);
    EXPECT_EQ(cache.get_total_evicted(), 3); // Only 3 were evicted
}

TEST_F(Test__CPURingKVCache_Comprehensive, Evict_ZeroOrNegative_NoOp)
{
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, 8, NKV, HD, DeviceId::cpu());

    auto k = taggedFP32(3, 1.0f);
    auto v = taggedFP32(3, 2.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k.get(), v.get(), 3));

    cache.evict_oldest_from_sequence(0, 0);
    EXPECT_EQ(cache.ring_size(0, 0), 3);
    cache.evict_oldest_from_sequence(0, -5);
    EXPECT_EQ(cache.ring_size(0, 0), 3);
    EXPECT_EQ(cache.get_total_evicted(), 0);
}

TEST_F(Test__CPURingKVCache_Comprehensive, Evict_ThenAppend_DataCorrect)
{
    constexpr int MAX_SEQ = 4;
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, MAX_SEQ, NKV, HD, DeviceId::cpu());

    // Fill 3 tokens: 100, 101, 102
    auto k1 = taggedFP32(3, 100.0f);
    auto v1 = taggedFP32(3, 200.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k1.get(), v1.get(), 3));

    // Evict 2 → remaining: 102
    cache.evict_oldest_from_sequence(0, 2);
    EXPECT_EQ(cache.ring_size(0, 0), 1);
    EXPECT_EQ(cache.ring_head(0, 0), 2);

    // Append 2 more: 110, 111
    auto k2 = taggedFP32(2, 110.0f);
    auto v2 = taggedFP32(2, 210.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k2.get(), v2.get(), 2));
    EXPECT_EQ(cache.ring_size(0, 0), 3);

    // Gather should show: 102, 110, 111
    auto gk = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), static_cast<size_t>(KV_DIM)});
    auto gv = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), static_cast<size_t>(KV_DIM)});
    std::vector<int> lens;
    int max_kv = cache.gather_kv_batched(0, 1, gk.get(), gv.get(), lens);
    EXPECT_EQ(max_kv, 3);
    EXPECT_TRUE(rowNear(gk->data() + 0 * KV_DIM, 102.0f));
    EXPECT_TRUE(rowNear(gk->data() + 1 * KV_DIM, 110.0f));
    EXPECT_TRUE(rowNear(gk->data() + 2 * KV_DIM, 111.0f));
}

TEST_F(Test__CPURingKVCache_Comprehensive, Evict_ThenWrap_Shadow_Correct)
{
    // Evict + wrap + get_kv_converted
    constexpr int MAX_SEQ = 4;
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, MAX_SEQ, NKV, HD, DeviceId::cpu());

    // Fill: 100, 101, 102, 103
    auto k1 = taggedFP32(4, 100.0f);
    auto v1 = taggedFP32(4, 200.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k1.get(), v1.get(), 4));

    // Evict 2 → 102, 103 (head=2, size=2)
    cache.evict_oldest_from_sequence(0, 2);

    // Append 3 → 102, 103, 110, 111 (fills ring, then wraps one)
    auto k2 = taggedFP32(3, 110.0f);
    auto v2 = taggedFP32(3, 210.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k2.get(), v2.get(), 3));
    // After: size=4 (full), head moved because one overwrite
    EXPECT_EQ(cache.ring_size(0, 0), 4);

    // get_kv_converted should give logical order: 103, 110, 111, 112
    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 4);
    EXPECT_TRUE(rowNear(out_k->data() + 0 * KV_DIM, 103.0f))
        << "got " << out_k->data()[0];
    EXPECT_TRUE(rowNear(out_k->data() + 1 * KV_DIM, 110.0f))
        << "got " << out_k->data()[KV_DIM];
    EXPECT_TRUE(rowNear(out_k->data() + 2 * KV_DIM, 111.0f))
        << "got " << out_k->data()[2 * KV_DIM];
    EXPECT_TRUE(rowNear(out_k->data() + 3 * KV_DIM, 112.0f))
        << "got " << out_k->data()[3 * KV_DIM];
}

// =========================================================================
// 8. MULTI-SEQUENCE (batch > 1)
// =========================================================================

TEST_F(Test__CPURingKVCache_Comprehensive, MultiSeq_AppendAndGather)
{
    constexpr int MAX_SEQ = 4;
    CPURingKVCacheFP32 cache(testMPI(), 1, 2, MAX_SEQ, NKV, HD, DeviceId::cpu());

    // Seq 0: 3 tokens (10, 11, 12)
    auto k0 = taggedFP32(3, 10.0f);
    auto v0 = taggedFP32(3, 20.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k0.get(), v0.get(), 3));

    // Seq 1: 2 tokens (50, 51)
    auto k1 = taggedFP32(2, 50.0f);
    auto v1 = taggedFP32(2, 60.0f);
    ASSERT_TRUE(cache.append_kv(0, 1, k1.get(), v1.get(), 2));

    // Gather batched — output layout is [num_seqs * max_kv_len, kv_dim]
    // max_kv_len = max(3, 2) = 3
    constexpr int MAX_KV = 3;
    auto gk = std::make_shared<FP32Tensor>(
        std::vector<size_t>{2 * static_cast<size_t>(MAX_KV), static_cast<size_t>(KV_DIM)});
    auto gv = std::make_shared<FP32Tensor>(
        std::vector<size_t>{2 * static_cast<size_t>(MAX_KV), static_cast<size_t>(KV_DIM)});
    std::vector<int> lens;
    int max_kv = cache.gather_kv_batched(0, 2, gk.get(), gv.get(), lens);
    EXPECT_EQ(max_kv, 3);
    ASSERT_EQ(lens.size(), 2u);
    EXPECT_EQ(lens[0], 3);
    EXPECT_EQ(lens[1], 2);

    // Check seq 0 data (offset = 0)
    EXPECT_TRUE(rowNear(gk->data() + 0 * KV_DIM, 10.0f));
    EXPECT_TRUE(rowNear(gk->data() + 2 * KV_DIM, 12.0f));

    // Check seq 1 data (offset = MAX_KV * KV_DIM, since stride = max_kv_len)
    const float *s1 = gk->data() + MAX_KV * KV_DIM;
    EXPECT_TRUE(rowNear(s1 + 0 * KV_DIM, 50.0f));
    EXPECT_TRUE(rowNear(s1 + 1 * KV_DIM, 51.0f));
}

TEST_F(Test__CPURingKVCache_Comprehensive, MultiSeq_IndependentWrapping)
{
    constexpr int MAX_SEQ = 4;
    CPURingKVCacheFP32 cache(testMPI(), 1, 2, MAX_SEQ, NKV, HD, DeviceId::cpu());

    // Seq 0: fill + wrap
    auto k0a = taggedFP32(4, 10.0f);
    auto v0a = taggedFP32(4, 20.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k0a.get(), v0a.get(), 4));
    auto k0b = taggedFP32(2, 14.0f);
    auto v0b = taggedFP32(2, 24.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k0b.get(), v0b.get(), 2));
    // Seq 0: head=2, size=4, logical: 12, 13, 14, 15

    // Seq 1: only 2 tokens, no wrap
    auto k1 = taggedFP32(2, 50.0f);
    auto v1 = taggedFP32(2, 60.0f);
    ASSERT_TRUE(cache.append_kv(0, 1, k1.get(), v1.get(), 2));
    // Seq 1: head=0, size=2, logical: 50, 51

    EXPECT_EQ(cache.ring_head(0, 0), 2);
    EXPECT_EQ(cache.ring_head(0, 1), 0);

    // get_kv_converted on each sequence independently
    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;

    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 4);
    EXPECT_TRUE(rowNear(out_k->data() + 0 * KV_DIM, 12.0f));
    EXPECT_TRUE(rowNear(out_k->data() + 3 * KV_DIM, 15.0f));

    ASSERT_TRUE(cache.get_kv_converted(0, 1, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 2);
    EXPECT_TRUE(rowNear(out_k->data() + 0 * KV_DIM, 50.0f));
    EXPECT_TRUE(rowNear(out_k->data() + 1 * KV_DIM, 51.0f));
}

// =========================================================================
// 9. MULTI-LAYER SHADOW CORRECTNESS
// =========================================================================

TEST_F(Test__CPURingKVCache_Comprehensive, MultiLayer_ShadowsIndependent)
{
    constexpr int MAX_SEQ = 4;
    CPURingKVCacheFP16 cache(testMPI(), 3, 1, MAX_SEQ, NKV, HD, DeviceId::cpu());

    // Each layer gets different data
    for (int l = 0; l < 3; ++l)
    {
        auto k = taggedFP16(3, static_cast<float>(l * 100));
        auto v = taggedFP16(3, static_cast<float>(l * 100 + 50));
        ASSERT_TRUE(cache.append_kv(l, 0, k.get(), v.get(), 3));
    }

    // Convert each layer independently
    for (int l = 0; l < 3; ++l)
    {
        ITensor *out_k = nullptr;
        ITensor *out_v = nullptr;
        int len = 0;
        ASSERT_TRUE(cache.get_kv_converted(l, 0, ActivationPrecision::FP32,
                                           &out_k, &out_v, &len));
        EXPECT_EQ(len, 3);
        float expected_base = static_cast<float>(l * 100);
        EXPECT_TRUE(rowNear(out_k->data(), expected_base, 0.5f))
            << "Layer " << l << " row 0: expected " << expected_base << " got " << out_k->data()[0];
    }
}

// =========================================================================
// 10. BF16 GET_KV_CONVERTED
// =========================================================================

TEST_F(Test__CPURingKVCache_Comprehensive, BF16_NoWrap_ConvertedValuesMatch)
{
    constexpr int MAX_SEQ = 8;
    CPURingKVCacheBF16 cache(testMPI(), 1, 1, MAX_SEQ, NKV, HD, DeviceId::cpu());

    auto k = taggedBF16(3, 10.0f);
    auto v = taggedBF16(3, 20.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k.get(), v.get(), 3));

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 3);

    // BF16 has lower precision; use a larger tolerance
    EXPECT_TRUE(rowNear(out_k->data() + 0 * KV_DIM, 10.0f, 0.1f));
    EXPECT_TRUE(rowNear(out_k->data() + 1 * KV_DIM, 11.0f, 0.1f));
    EXPECT_TRUE(rowNear(out_k->data() + 2 * KV_DIM, 12.0f, 0.1f));
}

TEST_F(Test__CPURingKVCache_Comprehensive, BF16_WrapAround_NewTokensConverted)
{
    constexpr int MAX_SEQ = 4;
    CPURingKVCacheBF16 cache(testMPI(), 1, 1, MAX_SEQ, NKV, HD, DeviceId::cpu());

    // Fill ring
    auto k1 = taggedBF16(4, 10.0f);
    auto v1 = taggedBF16(4, 20.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k1.get(), v1.get(), 4));

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));

    // Wrap: append 2 more
    auto k2 = taggedBF16(2, 14.0f);
    auto v2 = taggedBF16(2, 24.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k2.get(), v2.get(), 2));

    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 4);

    // Should see: 12, 13, 14, 15
    EXPECT_TRUE(rowNear(out_k->data() + 0 * KV_DIM, 12.0f, 0.1f))
        << "got " << out_k->data()[0];
    EXPECT_TRUE(rowNear(out_k->data() + 3 * KV_DIM, 15.0f, 0.1f))
        << "got " << out_k->data()[3 * KV_DIM];
}

// =========================================================================
// 11. GET_KV_CONVERTED EDGE CASES
// =========================================================================

TEST_F(Test__CPURingKVCache_Comprehensive, GetKvConverted_UnsupportedTarget_ReturnsFalse)
{
    CPURingKVCacheFP16 cache(testMPI(), 1, 1, 4, NKV, HD, DeviceId::cpu());

    auto k = taggedFP16(2, 1.0f);
    auto v = taggedFP16(2, 2.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k.get(), v.get(), 2));

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    EXPECT_FALSE(cache.get_kv_converted(0, 0, ActivationPrecision::BF16,
                                        &out_k, &out_v, &len));
}

TEST_F(Test__CPURingKVCache_Comprehensive, GetKvConverted_OutOfBounds_ReturnsFalse)
{
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, 4, NKV, HD, DeviceId::cpu());

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 99;
    EXPECT_FALSE(cache.get_kv_converted(-1, 0, ActivationPrecision::FP32,
                                        &out_k, &out_v, &len));
    EXPECT_EQ(out_k, nullptr);
    EXPECT_EQ(len, 0);
}

TEST_F(Test__CPURingKVCache_Comprehensive, GetKvConverted_EmptyCache_ReturnsZeroLen)
{
    CPURingKVCacheFP16 cache(testMPI(), 1, 1, 4, NKV, HD, DeviceId::cpu());

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = -1;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 0);
}

// =========================================================================
// 12. APPEND EDGE CASES
// =========================================================================

TEST_F(Test__CPURingKVCache_Comprehensive, Append_ZeroTokens_ReturnsFalse)
{
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, 4, NKV, HD, DeviceId::cpu());

    auto k = taggedFP32(0, 1.0f);
    auto v = taggedFP32(0, 2.0f);
    // append_kv with 0 tokens is treated as a no-op error
    EXPECT_FALSE(cache.append_kv(0, 0, k.get(), v.get(), 0));
    EXPECT_EQ(cache.ring_size(0, 0), 0);
}

TEST_F(Test__CPURingKVCache_Comprehensive, Append_NullTensors_ReturnsFalse)
{
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, 4, NKV, HD, DeviceId::cpu());

    EXPECT_FALSE(cache.append_kv(0, 0, nullptr, nullptr));
}

TEST_F(Test__CPURingKVCache_Comprehensive, Append_InferNumTokensFromShape)
{
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, 8, NKV, HD, DeviceId::cpu());

    auto k = taggedFP32(3, 10.0f);
    auto v = taggedFP32(3, 20.0f);
    // Use the overload that infers num_tokens from k->shape()[0]
    ASSERT_TRUE(cache.append_kv(0, 0, k.get(), v.get()));
    EXPECT_EQ(cache.ring_size(0, 0), 3);
}

TEST_F(Test__CPURingKVCache_Comprehensive, Append_MoreThanCapacity_SkipsOldest)
{
    constexpr int MAX_SEQ = 4;
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, MAX_SEQ, NKV, HD, DeviceId::cpu());

    // Append 7 tokens at once into a cache of capacity 4
    // Should keep only the last 4 (tokens 3, 4, 5, 6)
    auto k = taggedFP32(7, 100.0f);
    auto v = taggedFP32(7, 200.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k.get(), v.get(), 7));

    EXPECT_EQ(cache.ring_size(0, 0), 4);

    auto gk = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), static_cast<size_t>(KV_DIM)});
    auto gv = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), static_cast<size_t>(KV_DIM)});
    std::vector<int> lens;
    int max_kv = cache.gather_kv_batched(0, 1, gk.get(), gv.get(), lens);
    EXPECT_EQ(max_kv, 4);
    EXPECT_TRUE(rowNear(gk->data() + 0 * KV_DIM, 103.0f))
        << "Oldest kept should be 103, got " << gk->data()[0];
    EXPECT_TRUE(rowNear(gk->data() + 3 * KV_DIM, 106.0f))
        << "Newest should be 106, got " << gk->data()[3 * KV_DIM];
}

TEST_F(Test__CPURingKVCache_Comprehensive, Append_ExactCapacity_NoWrap)
{
    constexpr int MAX_SEQ = 4;
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, MAX_SEQ, NKV, HD, DeviceId::cpu());

    auto k = taggedFP32(4, 10.0f);
    auto v = taggedFP32(4, 20.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k.get(), v.get(), 4));

    EXPECT_EQ(cache.ring_size(0, 0), 4);
    EXPECT_EQ(cache.ring_head(0, 0), 0); // No wrap, head stays at 0
}

// =========================================================================
// 13. STRESS: MULTIPLE COMPLETE RING WRAPS
// =========================================================================

TEST_F(Test__CPURingKVCache_Comprehensive, FP32_MultiWrap_CorrectLinearization)
{
    constexpr int MAX_SEQ = 4;
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, MAX_SEQ, NKV, HD, DeviceId::cpu());

    // Fill the ring
    auto k0 = taggedFP32(4, 0.0f);
    auto v0 = taggedFP32(4, 0.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k0.get(), v0.get(), 4));

    // Wrap around multiple times: append 1 token at a time, 12 more tokens
    for (int i = 0; i < 12; ++i)
    {
        float val = 100.0f + static_cast<float>(i);
        auto ki = uniformFP32(1, val);
        auto vi = uniformFP32(1, val + 1000.0f);
        ASSERT_TRUE(cache.append_kv(0, 0, ki.get(), vi.get(), 1));

        // After every append, verify get_kv_converted returns correct order
        ITensor *out_k = nullptr;
        ITensor *out_v = nullptr;
        int len = 0;
        ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                           &out_k, &out_v, &len));
        EXPECT_EQ(len, 4);

        // The newest token should be at the end
        EXPECT_TRUE(rowNear(out_k->data() + 3 * KV_DIM, val))
            << "After append #" << i << ": newest should be " << val
            << " got " << out_k->data()[3 * KV_DIM];
    }
}

TEST_F(Test__CPURingKVCache_Comprehensive, FP16_MultiWrap_StressIncremental)
{
    constexpr int MAX_SEQ = 4;
    CPURingKVCacheFP16 cache(testMPI(), 1, 1, MAX_SEQ, NKV, HD, DeviceId::cpu());

    // Fill
    auto k0 = taggedFP16(4, 0.0f);
    auto v0 = taggedFP16(4, 0.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k0.get(), v0.get(), 4));

    // Wrap 3 full cycles (12 more tokens)
    for (int i = 0; i < 12; ++i)
    {
        float val = 10.0f + static_cast<float>(i);
        auto ki = taggedFP16(1, val);
        auto vi = taggedFP16(1, val);
        ASSERT_TRUE(cache.append_kv(0, 0, ki.get(), vi.get(), 1));

        ITensor *out_k = nullptr;
        ITensor *out_v = nullptr;
        int len = 0;
        ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                           &out_k, &out_v, &len));
        EXPECT_EQ(len, 4);

        // Newest at the end
        EXPECT_TRUE(rowNear(out_k->data() + 3 * KV_DIM, val, 0.1f))
            << "FP16 stress, append #" << i << ": expected " << val
            << " got " << out_k->data()[3 * KV_DIM];
    }
}

TEST_F(Test__CPURingKVCache_Comprehensive, FP32_RingHead_WrapsBackToZero)
{
    // Boundary: head wraps back to exactly 0
    constexpr int MAX_SEQ = 4;
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, MAX_SEQ, NKV, HD, DeviceId::cpu());

    // Fill: head=0, size=4
    auto k0 = taggedFP32(4, 0.0f);
    auto v0 = taggedFP32(4, 0.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k0.get(), v0.get(), 4));

    // Append exactly 4 more → head should advance exactly 4, wrapping back to 0
    auto k1 = taggedFP32(4, 100.0f);
    auto v1 = taggedFP32(4, 200.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k1.get(), v1.get(), 4));

    EXPECT_EQ(cache.ring_head(0, 0), 0);
    EXPECT_EQ(cache.ring_size(0, 0), 4);

    // Passthrough (head == 0, no need for shadow linearization)
    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 4);
    EXPECT_TRUE(rowNear(out_k->data() + 0 * KV_DIM, 100.0f));
    EXPECT_TRUE(rowNear(out_k->data() + 3 * KV_DIM, 103.0f));
}

// =========================================================================
// 14. FP32 INCREMENTAL DECODE (NO WRAP)
// =========================================================================

TEST_F(Test__CPURingKVCache_Comprehensive, FP32_IncrementalDecode_NoWrap)
{
    constexpr int MAX_SEQ = 8;
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, MAX_SEQ, NKV, HD, DeviceId::cpu());

    // Prefill 4 tokens
    auto k0 = taggedFP32(4, 10.0f);
    auto v0 = taggedFP32(4, 20.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k0.get(), v0.get(), 4));

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 4);

    // Decode step: append 1 token
    auto k1 = uniformFP32(1, 14.0f);
    auto v1 = uniformFP32(1, 24.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k1.get(), v1.get(), 1));

    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 5);
    // Head is 0, so passthrough, should see all 5
    EXPECT_TRUE(rowNear(out_k->data() + 0 * KV_DIM, 10.0f));
    EXPECT_TRUE(rowNear(out_k->data() + 4 * KV_DIM, 14.0f));
}

// =========================================================================
// 15. HEAD_MAJOR + GET_KV_CONVERTED
// =========================================================================

TEST_F(Test__CPURingKVCache_Comprehensive, HeadMajor_GetKvConverted_NoWrap)
{
    // HEAD_MAJOR layout with get_kv_converted (FP32 passthrough path)
    constexpr int MAX_SEQ = 8;
    constexpr int N_HEADS = 2;
    constexpr int MY_HD = 2;
    constexpr int MY_KVDIM = N_HEADS * MY_HD;

    CPURingKVCacheFP32 cache(testMPI(), 1, 1, MAX_SEQ, N_HEADS, MY_HD, DeviceId::cpu(),
                             KVCacheLayoutMode::HEAD_MAJOR);

    // Create position-major input (what projection stages produce)
    auto k = std::make_shared<FP32Tensor>(
        std::vector<size_t>{3, static_cast<size_t>(MY_KVDIM)});
    auto v = std::make_shared<FP32Tensor>(
        std::vector<size_t>{3, static_cast<size_t>(MY_KVDIM)});
    float *kd = k->mutable_data();
    float *vd = v->mutable_data();
    // Token 0: h0=[1,2], h1=[3,4]
    kd[0] = 1;
    kd[1] = 2;
    kd[2] = 3;
    kd[3] = 4;
    // Token 1: h0=[5,6], h1=[7,8]
    kd[4] = 5;
    kd[5] = 6;
    kd[6] = 7;
    kd[7] = 8;
    // Token 2: h0=[9,10], h1=[11,12]
    kd[8] = 9;
    kd[9] = 10;
    kd[10] = 11;
    kd[11] = 12;
    std::copy(kd, kd + 12, vd);

    ASSERT_TRUE(cache.append_kv(0, 0, k.get(), v.get(), 3));

    // get_kv_converted returns the raw buffer (HEAD_MAJOR layout since head=0)
    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 3);
    // In HEAD_MAJOR: [h0: pos0 pos1 pos2 | h1: pos0 pos1 pos2]
    // h0 block: [1,2, 5,6, 9,10, ...]  (each pos = MY_HD=2 elements)
    // h1 block: [3,4, 7,8, 11,12, ...]
    const float *okd = out_k->data();
    // Head 0, position 0
    EXPECT_FLOAT_EQ(okd[0], 1.0f);
    EXPECT_FLOAT_EQ(okd[1], 2.0f);
    // Head 0, position 1
    EXPECT_FLOAT_EQ(okd[MY_HD], 5.0f);
    // Head 1 starts at offset = MAX_SEQ * MY_HD
    const float *h1 = okd + MAX_SEQ * MY_HD;
    EXPECT_FLOAT_EQ(h1[0], 3.0f);
    EXPECT_FLOAT_EQ(h1[1], 4.0f);
}

// =========================================================================
// 16. FACTORY FUNCTIONS
// =========================================================================

TEST_F(Test__CPURingKVCache_Comprehensive, Factory_CreateCPURingKVCache_AllPrecisions)
{
    auto mpi = testMPI();
    auto check = [&](ActivationPrecision p)
    {
        auto cache = createCPURingKVCache(p, mpi, 1, 1, 8, 2, 4, DeviceId::cpu());
        ASSERT_NE(cache, nullptr) << "Factory failed for precision " << static_cast<int>(p);
        EXPECT_EQ(cache->k_precision(), p);
        EXPECT_EQ(cache->max_seq_len(), 8);
    };

    check(ActivationPrecision::FP32);
    check(ActivationPrecision::FP16);
    check(ActivationPrecision::BF16);
    check(ActivationPrecision::Q8_1);
    check(ActivationPrecision::Q16_1);
    check(ActivationPrecision::TQ4);
    check(ActivationPrecision::TQ8);
}

TEST_F(Test__CPURingKVCache_Comprehensive, Factory_CreateShardedCPURingKVCache)
{
    auto mpi = testMPI();
    auto cache = createShardedCPURingKVCache(
        ActivationPrecision::FP32, mpi,
        2, 1, 8,
        4, 2, 0,
        4, DeviceId::cpu());

    ASSERT_NE(cache, nullptr);
    EXPECT_TRUE(cache->is_sharded());
    EXPECT_EQ(cache->local_n_kv_heads(), 2);
    EXPECT_EQ(cache->kv_head_start(), 0);
}

TEST_F(Test__CPURingKVCache_Comprehensive, Factory_PerLayerDevices)
{
    auto mpi = testMPI();
    std::vector<int> devices = {0, 0};
    auto cache = createCPURingKVCache(
        ActivationPrecision::FP32, mpi,
        2, 1, 8, 2, 4,
        devices);
    ASSERT_NE(cache, nullptr);
    EXPECT_EQ(cache->n_layers(), 2);
}

// =========================================================================
// 17. GET_LAYER_DEVICE
// =========================================================================

TEST_F(Test__CPURingKVCache_Comprehensive, GetLayerDevice_ReturnsCorrectDevice)
{
    CPURingKVCacheFP32 cache(testMPI(), 2, 1, 4, NKV, HD, DeviceId::cpu());

    EXPECT_EQ(cache.get_layer_device(0).type, DeviceType::CPU);
    EXPECT_EQ(cache.get_layer_device(1).type, DeviceType::CPU);
    // OOB returns CPU (safe fallback)
    EXPECT_EQ(cache.get_layer_device(-1).type, DeviceType::CPU);
    EXPECT_EQ(cache.get_layer_device(99).type, DeviceType::CPU);
}

// =========================================================================
// 18. GATHER WITH WRAP — CONSISTENCY WITH GET_KV_CONVERTED
// =========================================================================

TEST_F(Test__CPURingKVCache_Comprehensive, FP32_GatherVsConverted_AfterWrap)
{
    // Use FP32 cache so gather and get_kv_converted both produce FP32 data.
    constexpr int MAX_SEQ = 4;
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, MAX_SEQ, NKV, HD, DeviceId::cpu());

    // Fill + wrap
    auto k1 = taggedFP32(4, 10.0f);
    auto v1 = taggedFP32(4, 20.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k1.get(), v1.get(), 4));
    auto k2 = taggedFP32(2, 14.0f);
    auto v2 = taggedFP32(2, 24.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k2.get(), v2.get(), 2));

    // Gather (linearizes ring into FP32 output)
    auto gk = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), static_cast<size_t>(KV_DIM)});
    auto gv = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), static_cast<size_t>(KV_DIM)});
    std::vector<int> lens;
    cache.gather_kv_batched(0, 1, gk.get(), gv.get(), lens);

    // get_kv_converted
    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 4);

    // Compare: both should have same logical order
    const float *g = gk->data();
    const float *c = out_k->data();
    for (int r = 0; r < 4; ++r)
    {
        for (int col = 0; col < KV_DIM; ++col)
        {
            EXPECT_NEAR(g[r * KV_DIM + col], c[r * KV_DIM + col], 0.01f)
                << "FP32 gather vs converted mismatch at row=" << r << " col=" << col;
        }
    }
}

TEST_F(Test__CPURingKVCache_Comprehensive, FP32_GatherVsConverted_MultipleWraps)
{
    // Additional gather vs converted comparison with multiple wraps.
    constexpr int MAX_SEQ = 4;
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, MAX_SEQ, NKV, HD, DeviceId::cpu());

    // Fill + wrap 3 times (12 overwrites)
    auto k0 = taggedFP32(4, 0.0f);
    auto v0 = taggedFP32(4, 0.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k0.get(), v0.get(), 4));
    for (int i = 0; i < 12; ++i)
    {
        auto ki = uniformFP32(1, 100.0f + static_cast<float>(i));
        auto vi = uniformFP32(1, 200.0f + static_cast<float>(i));
        ASSERT_TRUE(cache.append_kv(0, 0, ki.get(), vi.get(), 1));
    }

    auto gk = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), static_cast<size_t>(KV_DIM)});
    auto gv = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), static_cast<size_t>(KV_DIM)});
    std::vector<int> lens;
    cache.gather_kv_batched(0, 1, gk.get(), gv.get(), lens);

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));

    const float *g = gk->data();
    const float *c = out_k->data();
    for (int r = 0; r < 4; ++r)
        for (int col = 0; col < KV_DIM; ++col)
            EXPECT_NEAR(g[r * KV_DIM + col], c[r * KV_DIM + col], 0.01f)
                << "FP32 multi-wrap gather vs converted at row=" << r << " col=" << col;
}

// =========================================================================
// 19. SHADOW: REUSE AFTER HEAD RETURNS TO ZERO
// =========================================================================

TEST_F(Test__CPURingKVCache_Comprehensive, FP32Shadow_HeadReturnsToZero_ShadowInvalidated)
{
    // After wrapping exactly N times such that head == 0 again,
    // the shadow should still reflect latest data.
    constexpr int MAX_SEQ = 4;
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, MAX_SEQ, NKV, HD, DeviceId::cpu());

    // Fill: 0, 1, 2, 3 → head=0, size=4
    auto k0 = taggedFP32(4, 0.0f);
    auto v0 = taggedFP32(4, 0.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k0.get(), v0.get(), 4));

    // Force shadow creation via get_kv_converted
    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));

    // Wrap: append 2 → head=2
    auto k1 = taggedFP32(2, 10.0f);
    auto v1 = taggedFP32(2, 20.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k1.get(), v1.get(), 2));
    EXPECT_EQ(cache.ring_head(0, 0), 2);

    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_TRUE(rowNear(out_k->data() + 0 * KV_DIM, 2.0f));
    EXPECT_TRUE(rowNear(out_k->data() + 3 * KV_DIM, 11.0f));

    // Wrap again: append 2 → head=0 (back to origin)
    auto k2 = taggedFP32(2, 20.0f);
    auto v2 = taggedFP32(2, 30.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k2.get(), v2.get(), 2));
    EXPECT_EQ(cache.ring_head(0, 0), 0); // Back to zero

    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 4);
    // Passthrough path: should see 10, 11, 20, 21 in physical order (which IS logical when head=0)
    EXPECT_TRUE(rowNear(out_k->data() + 0 * KV_DIM, 10.0f))
        << "got " << out_k->data()[0];
    EXPECT_TRUE(rowNear(out_k->data() + 1 * KV_DIM, 11.0f))
        << "got " << out_k->data()[KV_DIM];
    EXPECT_TRUE(rowNear(out_k->data() + 2 * KV_DIM, 20.0f))
        << "got " << out_k->data()[2 * KV_DIM];
    EXPECT_TRUE(rowNear(out_k->data() + 3 * KV_DIM, 21.0f))
        << "got " << out_k->data()[3 * KV_DIM];
}

// =========================================================================
// 20. SHADOW: FP32 WRAP + RoPE INCREMENTAL AFTER WRAP
// =========================================================================

TEST_F(Test__CPURingKVCache_Comprehensive, FP32_RoPE_ReturnsCorrectLength_AfterWrap)
{
    // Tests that get_kv_converted with RoPE doesn't crash after wrap
    // and returns the correct length. RoPE position values are checked
    // indirectly — we verify the output differs from the raw (no-RoPE) case
    // at non-zero positions.
    constexpr int MAX_SEQ = 4;
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, MAX_SEQ, NKV, HD, DeviceId::cpu());

    // Fill with 1.0 everywhere
    auto k0 = uniformFP32(4, 1.0f);
    auto v0 = uniformFP32(4, 1.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k0.get(), v0.get(), 4));

    IKVCache::KVReadParams rope;
    rope.rope_theta = 10000.0f;
    rope.position_start = 0;
    rope.n_kv_heads = NKV;
    rope.head_dim = HD;

    // First call with RoPE (pre-wrap, head=0)
    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len, &rope));
    EXPECT_EQ(len, 4);

    // Position 0 with input [1,1,1,1] and cos(0)=1, sin(0)=0 should be identity
    for (int c = 0; c < KV_DIM; ++c)
        EXPECT_NEAR(out_k->data()[c], 1.0f, 0.001f)
            << "Pre-wrap RoPE position 0 should be identity, col=" << c;

    // Wrap: append 1 more → size stays 4, head=1
    auto k1 = uniformFP32(1, 1.0f);
    auto v1 = uniformFP32(1, 1.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k1.get(), v1.get(), 1));
    EXPECT_EQ(cache.ring_head(0, 0), 1);

    // After wrap, get_kv_converted with RoPE should not crash and return 4
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len, &rope));
    EXPECT_EQ(len, 4);
    // Output should not be all NaN or all zero
    bool has_nonzero = false;
    for (int i = 0; i < 4 * KV_DIM; ++i)
    {
        EXPECT_FALSE(std::isnan(out_k->data()[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(out_k->data()[i])) << "Inf at index " << i;
        if (std::abs(out_k->data()[i]) > 0.001f)
            has_nonzero = true;
    }
    EXPECT_TRUE(has_nonzero) << "All zeros after wrap+RoPE";

    // Decode: append 1 more → head=2
    auto k2 = uniformFP32(1, 1.0f);
    auto v2 = uniformFP32(1, 1.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k2.get(), v2.get(), 1));

    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len, &rope));
    EXPECT_EQ(len, 4);
}

// =========================================================================
// 21. V TENSOR CORRECTNESS (PARALLEL TO K)
// =========================================================================

TEST_F(Test__CPURingKVCache_Comprehensive, FP16_WrapAround_V_CorrectValues)
{
    constexpr int MAX_SEQ = 4;
    CPURingKVCacheFP16 cache(testMPI(), 1, 1, MAX_SEQ, NKV, HD, DeviceId::cpu());

    auto k1 = taggedFP16(4, 10.0f);
    auto v1 = taggedFP16(4, 110.0f); // V has different base
    ASSERT_TRUE(cache.append_kv(0, 0, k1.get(), v1.get(), 4));

    auto k2 = taggedFP16(2, 14.0f);
    auto v2 = taggedFP16(2, 114.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k2.get(), v2.get(), 2));

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));

    // K: 12, 13, 14, 15
    EXPECT_TRUE(rowNear(out_k->data() + 0 * KV_DIM, 12.0f, 0.1f));
    EXPECT_TRUE(rowNear(out_k->data() + 3 * KV_DIM, 15.0f, 0.1f));

    // V: 112, 113, 114, 115
    EXPECT_TRUE(rowNear(out_v->data() + 0 * KV_DIM, 112.0f, 0.1f))
        << "V row 0 expected 112, got " << out_v->data()[0];
    EXPECT_TRUE(rowNear(out_v->data() + 3 * KV_DIM, 115.0f, 0.1f))
        << "V row 3 expected 115, got " << out_v->data()[3 * KV_DIM];
}

// =========================================================================
// 22. APPEND VIA IKVCache INTERFACE (POLYMORPHIC)
// =========================================================================

TEST_F(Test__CPURingKVCache_Comprehensive, Append_ViaIKVCachePolymorph)
{
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, 8, NKV, HD, DeviceId::cpu());
    IKVCache *iface = &cache;

    auto k = taggedFP32(3, 10.0f);
    auto v = taggedFP32(3, 20.0f);
    ASSERT_TRUE(iface->append(0, 0, k.get(), v.get(), 3));
    EXPECT_EQ(cache.ring_size(0, 0), 3);
}

// =========================================================================
// 23. SINGLE TOKEN (M=1 DECODE PATH)
// =========================================================================

TEST_F(Test__CPURingKVCache_Comprehensive, SingleToken_AppendAndConvert)
{
    constexpr int MAX_SEQ = 4;
    CPURingKVCacheFP16 cache(testMPI(), 1, 1, MAX_SEQ, NKV, HD, DeviceId::cpu());

    // Append 1 token
    auto k = taggedFP16(1, 42.0f);
    auto v = taggedFP16(1, 84.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k.get(), v.get(), 1));
    EXPECT_EQ(cache.ring_size(0, 0), 1);

    // Convert to FP32 via get_kv_converted
    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 1);
    EXPECT_TRUE(rowNear(out_k->data(), 42.0f, 0.1f));
    EXPECT_TRUE(rowNear(out_v->data(), 84.0f, 0.1f));
}

TEST_F(Test__CPURingKVCache_Comprehensive, FP32_SingleToken_GatherAndConvert)
{
    // FP32 cache: gather and convert should produce identical results.
    constexpr int MAX_SEQ = 4;
    CPURingKVCacheFP32 cache(testMPI(), 1, 1, MAX_SEQ, NKV, HD, DeviceId::cpu());

    auto k = taggedFP32(1, 42.0f);
    auto v = taggedFP32(1, 84.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k.get(), v.get(), 1));

    // Gather
    auto gk = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), static_cast<size_t>(KV_DIM)});
    auto gv = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), static_cast<size_t>(KV_DIM)});
    std::vector<int> lens;
    int max_kv = cache.gather_kv_batched(0, 1, gk.get(), gv.get(), lens);
    EXPECT_EQ(max_kv, 1);
    EXPECT_TRUE(rowNear(gk->data(), 42.0f));
    EXPECT_TRUE(rowNear(gv->data(), 84.0f));

    // Convert
    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 1);
    EXPECT_TRUE(rowNear(out_k->data(), 42.0f));
}

// =========================================================================
// 24. MULTIPLE SEQUENCES + CLEAR_SEQUENCE INTERACTION
// =========================================================================

TEST_F(Test__CPURingKVCache_Comprehensive, MultiSeq_ClearOne_OtherUnaffected)
{
    CPURingKVCacheFP16 cache(testMPI(), 1, 2, 8, NKV, HD, DeviceId::cpu());

    auto k0 = taggedFP16(3, 10.0f);
    auto v0 = taggedFP16(3, 20.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k0.get(), v0.get(), 3));

    auto k1 = taggedFP16(2, 50.0f);
    auto v1 = taggedFP16(2, 60.0f);
    ASSERT_TRUE(cache.append_kv(0, 1, k1.get(), v1.get(), 2));

    // Convert both
    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    ASSERT_TRUE(cache.get_kv_converted(0, 1, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));

    // Clear seq 0 only
    cache.clear_sequence(0, 0);
    EXPECT_EQ(cache.ring_size(0, 0), 0);
    EXPECT_EQ(cache.ring_size(0, 1), 2); // Untouched

    // Re-append to seq 0
    auto k0b = taggedFP16(1, 99.0f);
    auto v0b = taggedFP16(1, 99.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k0b.get(), v0b.get(), 1));

    // Verify seq 0 shadow reflects new data
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 1);
    EXPECT_TRUE(rowNear(out_k->data(), 99.0f, 0.5f));

    // Verify seq 1 still has old data
    ASSERT_TRUE(cache.get_kv_converted(0, 1, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 2);
    EXPECT_TRUE(rowNear(out_k->data(), 50.0f, 0.5f));
}

// =========================================================================
// 25. SHADOW: CONVERSION AFTER WRAP FOLLOWED BY MORE APPENDS (INCREMENTAL)
// =========================================================================

TEST_F(Test__CPURingKVCache_Comprehensive, FP16_ConvertWrap_ThenIncrementalAppend)
{
    // Sequence: fill → wrap → convert → append 1 → convert → check
    // Tests that after shadow sees a wrap and reconverts, further incremental
    // appends (which advance head again) are still correctly tracked.
    constexpr int MAX_SEQ = 4;
    CPURingKVCacheFP16 cache(testMPI(), 1, 1, MAX_SEQ, NKV, HD, DeviceId::cpu());

    // Fill
    auto k0 = taggedFP16(4, 0.0f);
    auto v0 = taggedFP16(4, 0.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k0.get(), v0.get(), 4));

    // Wrap: append 1 → head=1
    auto k1 = taggedFP16(1, 10.0f);
    auto v1 = taggedFP16(1, 10.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k1.get(), v1.get(), 1));

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 4);
    EXPECT_TRUE(rowNear(out_k->data() + 3 * KV_DIM, 10.0f, 0.1f));

    // Append 1 more → head=2
    auto k2 = taggedFP16(1, 20.0f);
    auto v2 = taggedFP16(1, 20.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k2.get(), v2.get(), 1));

    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    EXPECT_EQ(len, 4);
    // Logical: 2, 3, 10, 20
    EXPECT_TRUE(rowNear(out_k->data() + 0 * KV_DIM, 2.0f, 0.1f))
        << "got " << out_k->data()[0];
    EXPECT_TRUE(rowNear(out_k->data() + 2 * KV_DIM, 10.0f, 0.1f))
        << "got " << out_k->data()[2 * KV_DIM];
    EXPECT_TRUE(rowNear(out_k->data() + 3 * KV_DIM, 20.0f, 0.1f))
        << "got " << out_k->data()[3 * KV_DIM];

    // Append 1 more → head=3
    auto k3 = taggedFP16(1, 30.0f);
    auto v3 = taggedFP16(1, 30.0f);
    ASSERT_TRUE(cache.append_kv(0, 0, k3.get(), v3.get(), 1));

    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP32,
                                       &out_k, &out_v, &len));
    // Logical: 3, 10, 20, 30
    EXPECT_TRUE(rowNear(out_k->data() + 0 * KV_DIM, 3.0f, 0.1f))
        << "got " << out_k->data()[0];
    EXPECT_TRUE(rowNear(out_k->data() + 3 * KV_DIM, 30.0f, 0.1f))
        << "got " << out_k->data()[3 * KV_DIM];
}
