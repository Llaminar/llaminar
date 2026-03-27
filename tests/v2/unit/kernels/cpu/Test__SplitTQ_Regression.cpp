/**
 * @file Test__SplitTQ_Regression.cpp
 * @brief Regression tests for bugs discovered during split TQ8/TQ4 integration.
 *
 * Each test targets a specific bug that was found and fixed:
 *
 *   1. turboquant_dequantize_row_tq8 used the same context for all heads
 *      instead of ctx.for_layer(h) per head — heads > 0 got wrong rotation.
 *
 *   2. Row offset miscalculation: typed_data() returns uint8_t*, so pointer
 *      arithmetic `+ r * blocks_per_row()` advances by block-count not bytes.
 *      Correct offset requires `* block_bytes()`.
 *
 *   3. Buffer overflow: calling dequantize_to_fp32() on a cache-sized tensor
 *      [MAX_SEQ, KV_DIM] into a [KV_LEN, KV_DIM] buffer when MAX_SEQ > KV_LEN.
 *      Fix: use per-row to_fp32_row() for only the occupied rows.
 *
 *   4. Context mismatch: quantize with for_layer(L) but dequantize with root
 *      context — rotation matrices don't match, output is garbage.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "kernels/cpu/turboquant/TurboQuantContext.h"
#include "kernels/cpu/turboquant/TurboQuantDequantizeTQ8.h"
#include "kernels/cpu/turboquant/TurboQuantDequantizeTQ4.h"
#include "tensors/Tensors.h"

using namespace llaminar2;

// ─────────────────────────────────────────────────────────────────────
// Helper
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
// Fixture
// ─────────────────────────────────────────────────────────────────────
class Test__SplitTQ_Regression : public ::testing::Test
{
protected:
    static constexpr int HEAD_DIM = 128;
    static constexpr int N_KV_HEADS = 8;
    static constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM; // 1024

    std::unique_ptr<TurboQuantContext> ctx_;

    void SetUp() override
    {
        ctx_ = std::make_unique<TurboQuantContext>(
            HEAD_DIM, /*rotation_seed=*/42, /*projection_seed=*/42);
    }

    static std::vector<float> makeRandomData(size_t count, unsigned seed)
    {
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> data(count);
        for (auto &v : data)
            v = dist(rng);
        return data;
    }
};

// ─────────────────────────────────────────────────────────────────────
// Regression 1: turboquant_dequantize_row_tq8 must use for_layer(h) per head
//
// The original bug: all heads were dequantized with the same root context,
// so head 0 was correct but heads 1..N_KV_HEADS-1 used the wrong rotation
// matrix. This test verifies that dequantize_row_tq8 matches the
// TQ8Tensor::dequantize_to_fp32 reference (which iterates per-head correctly).
// ─────────────────────────────────────────────────────────────────────
TEST_F(Test__SplitTQ_Regression, RowDequant_PerHead_ContextDerivation)
{
    constexpr int N_ROWS = 4;
    auto data = makeRandomData(N_ROWS * KV_DIM, 100);

    // Quantize via TQ8Tensor (correct implementation)
    std::vector<size_t> shape = {N_ROWS, KV_DIM};
    auto tq8 = TQ8Tensor::quantize_from_fp32(data.data(), shape, HEAD_DIM, *ctx_);
    ASSERT_NE(tq8, nullptr);

    // Reference: dequant via TQ8Tensor::dequantize_to_fp32 (known correct)
    std::vector<float> ref(N_ROWS * KV_DIM);
    tq8->dequantize_to_fp32(ref.data(), *ctx_);

    // Test: dequant via turboquant_dequantize_row_tq8 (the fixed function)
    const size_t row_bytes = N_KV_HEADS * tq8->block_bytes();
    const auto *raw = reinterpret_cast<const uint8_t *>(tq8->raw_data());
    std::vector<float> test(N_ROWS * KV_DIM);
    alignas(64) float scratch[HEAD_DIM];

    for (int r = 0; r < N_ROWS; ++r)
    {
        const auto *row_blocks = reinterpret_cast<const TQ8Block_128 *>(
            raw + r * row_bytes);
        turboquant_dequantize_row_tq8<HEAD_DIM>(
            row_blocks, *ctx_, test.data() + r * KV_DIM, N_KV_HEADS, scratch);
    }

    // Verify each head separately — the pre-fix bug only affected heads > 0
    for (int r = 0; r < N_ROWS; ++r)
    {
        for (int h = 0; h < N_KV_HEADS; ++h)
        {
            size_t offset = r * KV_DIM + h * HEAD_DIM;
            double cos = cosine_similarity(ref.data() + offset, test.data() + offset, HEAD_DIM);
            EXPECT_GT(cos, 0.9999)
                << "Row " << r << " head " << h
                << " mismatch — dequantize_row_tq8 may not be using for_layer(h) per head";
        }
    }

    // Also verify the overall result is identical (float-exact)
    for (size_t i = 0; i < ref.size(); ++i)
    {
        EXPECT_FLOAT_EQ(ref[i], test[i])
            << "Mismatch at index " << i
            << " (row=" << (i / KV_DIM) << " head=" << ((i % KV_DIM) / HEAD_DIM) << ")";
    }
}

// ─────────────────────────────────────────────────────────────────────
// Regression 2: Row byte-offset calculation for typed_data() (uint8_t*)
//
// typed_data() returns uint8_t*, so advancing by `blocks_per_row()` counts
// blocks not bytes. Must multiply by block_bytes() to get the correct offset.
// ─────────────────────────────────────────────────────────────────────
TEST_F(Test__SplitTQ_Regression, RowByteOffset_TypedData_Pointer_Arithmetic)
{
    constexpr int N_ROWS = 8;
    auto data = makeRandomData(N_ROWS * KV_DIM, 200);

    auto tq8 = TQ8Tensor::quantize_from_fp32(data.data(),
                                             {static_cast<size_t>(N_ROWS), static_cast<size_t>(KV_DIM)},
                                             HEAD_DIM, *ctx_);

    const size_t bpr = tq8->blocks_per_row();
    const size_t bb = tq8->block_bytes();
    const auto *raw = reinterpret_cast<const uint8_t *>(tq8->raw_data());

    // Correct offset: row_index * blocks_per_row * block_bytes
    // Wrong offset would be: row_index * blocks_per_row (off by block_bytes factor)
    for (int r = 0; r < N_ROWS; ++r)
    {
        const auto *correct_ptr = raw + r * bpr * bb;
        // Verify this matches the reference dequant for this row
        std::vector<float> row_ref(KV_DIM);
        tq8->set_turboquant_context(ctx_.get());
        tq8->to_fp32_row(r, row_ref.data());

        // Dequant from the manually-computed pointer
        std::vector<float> row_test(KV_DIM);
        alignas(64) float scratch[HEAD_DIM];
        const auto *row_blocks = reinterpret_cast<const TQ8Block_128 *>(correct_ptr);
        turboquant_dequantize_row_tq8<HEAD_DIM>(
            row_blocks, *ctx_, row_test.data(), N_KV_HEADS, scratch);

        double cos = cosine_similarity(row_ref.data(), row_test.data(), KV_DIM);
        EXPECT_GT(cos, 0.9999)
            << "Row " << r << ": manual byte-offset doesn't match to_fp32_row reference";
    }

    // Verify that wrong offset (missing * block_bytes) gives nonsense for row > 0
    if (N_ROWS > 1)
    {
        // Wrong pointer for row 1: advance by blocks_per_row instead of blocks_per_row * block_bytes
        const auto *wrong_ptr = raw + 1 * bpr; // WRONG: missing * bb
        const auto *right_ptr = raw + 1 * bpr * bb;
        EXPECT_NE(wrong_ptr, right_ptr)
            << "block_bytes should not be 1 — offset calculation matters";

        // The wrong pointer should produce garbage compared to the correct row 1
        std::vector<float> row1_correct(KV_DIM);
        tq8->to_fp32_row(1, row1_correct.data());

        std::vector<float> row1_wrong(KV_DIM);
        alignas(64) float scratch[HEAD_DIM];
        // This reads from the wrong offset — should produce bad data
        const auto *wrong_blocks = reinterpret_cast<const TQ8Block_128 *>(wrong_ptr);
        turboquant_dequantize_row_tq8<HEAD_DIM>(
            wrong_blocks, *ctx_, row1_wrong.data(), N_KV_HEADS, scratch);

        double cos_wrong = cosine_similarity(row1_correct.data(), row1_wrong.data(), KV_DIM);
        // With wrong offset, cosine should be very low (essentially reading wrong data)
        EXPECT_LT(cos_wrong, 0.5)
            << "Wrong byte offset should produce garbage — test may not be detecting the bug";
    }
}

// ─────────────────────────────────────────────────────────────────────
// Regression 3: Buffer overflow when dequantizing cache-sized tensor
//
// A cache tensor has shape [MAX_SEQ, KV_DIM] but only KV_LEN rows are
// occupied. Calling dequantize_to_fp32() on the full tensor writes
// MAX_SEQ * KV_DIM floats, overflowing a buffer sized for KV_LEN rows.
// Fix: use to_fp32_row() for only the occupied rows.
// ─────────────────────────────────────────────────────────────────────
TEST_F(Test__SplitTQ_Regression, CacheTensor_DequantPartialRows_NoOverflow)
{
    constexpr int MAX_SEQ = 64;
    constexpr int KV_LEN = 16;

    // Allocate a cache-sized TQ8 tensor [MAX_SEQ, KV_DIM]
    auto cache_tensor = std::make_shared<TQ8Tensor>(
        std::vector<size_t>{MAX_SEQ, static_cast<size_t>(KV_DIM)}, HEAD_DIM);

    // Fill only KV_LEN rows with valid quantized data
    auto data = makeRandomData(KV_LEN * KV_DIM, 300);
    auto temp_tensor = TQ8Tensor::quantize_from_fp32(
        data.data(),
        {static_cast<size_t>(KV_LEN), static_cast<size_t>(KV_DIM)},
        HEAD_DIM, *ctx_);

    // Copy the quantized rows into the cache tensor (simulate CPURingKVCache::append)
    const size_t row_bytes = cache_tensor->blocks_per_row() * cache_tensor->block_bytes();
    auto *cache_raw = reinterpret_cast<uint8_t *>(cache_tensor->raw_mutable_data());
    const auto *src_raw = reinterpret_cast<const uint8_t *>(temp_tensor->raw_data());
    for (int r = 0; r < KV_LEN; ++r)
        std::memcpy(cache_raw + r * row_bytes, src_raw + r * row_bytes, row_bytes);

    cache_tensor->set_turboquant_context(ctx_.get());

    // SAFE approach: per-row dequant, writing only KV_LEN rows
    std::vector<float> safe_buffer(KV_LEN * KV_DIM, -999.0f);
    for (int r = 0; r < KV_LEN; ++r)
        cache_tensor->to_fp32_row(r, safe_buffer.data() + r * KV_DIM);

    // Verify the safe dequant matches the reference
    std::vector<float> ref(KV_LEN * KV_DIM);
    temp_tensor->dequantize_to_fp32(ref.data(), *ctx_);

    for (int r = 0; r < KV_LEN; ++r)
    {
        double cos = cosine_similarity(
            ref.data() + r * KV_DIM,
            safe_buffer.data() + r * KV_DIM, KV_DIM);
        EXPECT_GT(cos, 0.9999)
            << "Row " << r << ": per-row dequant differs from reference";
    }

    // Verify that a sentinel-protected buffer would detect the overflow:
    // dequantize_to_fp32 on the cache tensor writes MAX_SEQ rows,
    // but we only have KV_LEN valid rows + sentinel afterwards
    constexpr float SENTINEL = -12345.0f;
    std::vector<float> overflow_detect(MAX_SEQ * KV_DIM, SENTINEL);
    cache_tensor->dequantize_to_fp32(overflow_detect.data(), *ctx_);

    // The first KV_LEN rows should have real data
    for (int r = 0; r < KV_LEN; ++r)
    {
        EXPECT_NE(overflow_detect[r * KV_DIM], SENTINEL)
            << "Row " << r << " should have been written";
    }

    // Rows KV_LEN..MAX_SEQ-1 will have been written too (that's the overflow).
    // This demonstrates why dequantize_to_fp32 on a cache tensor is dangerous.
    bool sentinel_overwritten = false;
    for (int r = KV_LEN; r < MAX_SEQ; ++r)
    {
        if (overflow_detect[r * KV_DIM] != SENTINEL)
        {
            sentinel_overwritten = true;
            break;
        }
    }
    EXPECT_TRUE(sentinel_overwritten)
        << "dequantize_to_fp32 on a MAX_SEQ tensor should write beyond KV_LEN — "
        << "this test proves why per-row dequant is necessary for partial cache reads";
}

// ─────────────────────────────────────────────────────────────────────
// Regression 4: Context mismatch — quantize and dequantize must use
// the same layer-derived context
//
// The bug: quantize with for_layer(LAYER_IDX) but dequant with root ctx.
// Since each for_layer(L) generates a different rotation matrix, the inverse
// rotation in dequant doesn't undo the forward rotation from quant.
// ─────────────────────────────────────────────────────────────────────
TEST_F(Test__SplitTQ_Regression, ContextMismatch_WrongLayerCtx_ProducesGarbage)
{
    constexpr int N_ROWS = 4;
    auto data = makeRandomData(N_ROWS * KV_DIM, 400);

    // Quantize with layer context derived from layer 5
    const auto &layer5_ctx = ctx_->for_layer(5);
    auto tq8 = TQ8Tensor::quantize_from_fp32(
        data.data(),
        {static_cast<size_t>(N_ROWS), static_cast<size_t>(KV_DIM)},
        HEAD_DIM, layer5_ctx);
    ASSERT_NE(tq8, nullptr);

    // Correct: dequant with the SAME layer 5 context
    std::vector<float> correct(N_ROWS * KV_DIM);
    tq8->dequantize_to_fp32(correct.data(), layer5_ctx);

    double correct_cos = cosine_similarity(data.data(), correct.data(), data.size());
    EXPECT_GT(correct_cos, 0.999)
        << "Matching context should give high round-trip cosine";

    // Wrong: dequant with root context (different rotation matrix)
    std::vector<float> wrong_root(N_ROWS * KV_DIM);
    tq8->dequantize_to_fp32(wrong_root.data(), *ctx_);

    double wrong_cos = cosine_similarity(data.data(), wrong_root.data(), data.size());
    EXPECT_LT(wrong_cos, 0.5)
        << "Mismatched context (root vs layer5) should give garbage — "
        << "got cosine " << wrong_cos;

    // Wrong: dequant with a different layer context (layer 3 instead of 5)
    const auto &layer3_ctx = ctx_->for_layer(3);
    std::vector<float> wrong_layer(N_ROWS * KV_DIM);
    tq8->dequantize_to_fp32(wrong_layer.data(), layer3_ctx);

    double wrong_layer_cos = cosine_similarity(data.data(), wrong_layer.data(), data.size());
    EXPECT_LT(wrong_layer_cos, 0.5)
        << "Mismatched context (layer3 vs layer5) should give garbage — "
        << "got cosine " << wrong_layer_cos;

    // Same test for TQ4 (V path)
    auto tq4 = TQ4Tensor::quantize_from_fp32(
        data.data(),
        {static_cast<size_t>(N_ROWS), static_cast<size_t>(KV_DIM)},
        HEAD_DIM, layer5_ctx);
    ASSERT_NE(tq4, nullptr);

    std::vector<float> tq4_correct(N_ROWS * KV_DIM);
    tq4->dequantize_to_fp32(tq4_correct.data(), layer5_ctx);
    double tq4_correct_cos = cosine_similarity(data.data(), tq4_correct.data(), data.size());
    EXPECT_GT(tq4_correct_cos, 0.98)
        << "TQ4 matching context should give reasonable round-trip cosine";

    std::vector<float> tq4_wrong(N_ROWS * KV_DIM);
    tq4->dequantize_to_fp32(tq4_wrong.data(), *ctx_);
    double tq4_wrong_cos = cosine_similarity(data.data(), tq4_wrong.data(), data.size());
    EXPECT_LT(tq4_wrong_cos, 0.5)
        << "TQ4 mismatched context should give garbage — got cosine " << tq4_wrong_cos;
}
