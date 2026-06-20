/**
 * @file Test__Q16DecodeHeadMajor.cpp
 * @brief Regression tests for Q16_1 decode attention with HEAD_MAJOR KV layout
 *
 * These tests exercise the compute_decode_q16kv() fast path with Q16_1
 * tensors arranged in HEAD_MAJOR layout [head][position][head_dim], which
 * is the layout used by CPURingKVCache for Q16_1 KV caches.
 *
 * Regression context: A bug in HEAD_MAJOR addressing caused the per-head
 * block offset to use POSITION_MAJOR formula, producing garbage output
 * for models with n_kv_heads > 1 (e.g., head_dim=64, Qwen2.5-0.5B).
 *
 * Test strategy:
 *   1. Generate random FP32 Q/K/V data in POSITION_MAJOR format
 *   2. Rearrange K/V into HEAD_MAJOR and quantize to Q16_1
 *   3. Also quantize K/V in POSITION_MAJOR Q16_1 for comparison
 *   4. Call compute_tensor() which dispatches to compute_decode_q16kv()
 *   5. Compare both against scalar FP32 reference
 *   6. Compare HEAD_MAJOR vs POSITION_MAJOR to ensure layout equivalence
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "v2/kernels/cpu/attention/CPUFlashAttentionKernelT.h"
#include "v2/tensors/Tensors.h"
#include "v2/utils/CPUFeatures.h"

using namespace llaminar2;

// ---------------------------------------------------------------------------
// Scalar reference (same as Test__CPUFlashAttentionKernelT.cpp)
// ---------------------------------------------------------------------------
namespace ref
{
    static void attention(
        const float *Q, const float *K, const float *V, float *O,
        int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int position_offset)
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
                const float *q_ptr = Q + static_cast<size_t>(q_pos) * q_stride + static_cast<size_t>(h) * head_dim;
                float *out = O + static_cast<size_t>(q_pos) * q_stride + static_cast<size_t>(h) * head_dim;
                const int q_abs = position_offset + q_pos;

                std::vector<float> scores(kv_len);
                for (int k = 0; k < kv_len; ++k)
                {
                    const float *k_ptr = K + static_cast<size_t>(k) * kv_stride + static_cast<size_t>(kv_h) * head_dim;
                    bool masked = causal && k > q_abs;
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
                    for (int k = 0; k < kv_len; ++k)
                        scores[k] /= sum_exp;

                std::fill(out, out + head_dim, 0.0f);
                for (int k = 0; k < kv_len; ++k)
                {
                    if (scores[k] == 0.0f)
                        continue;
                    const float *v_ptr = V + static_cast<size_t>(k) * kv_stride + static_cast<size_t>(kv_h) * head_dim;
                    for (int d = 0; d < head_dim; ++d)
                        out[d] += scores[k] * v_ptr[d];
                }
            }
        }
    }
} // namespace ref

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
namespace
{
    std::mt19937 &rng()
    {
        static std::mt19937 gen(12345);
        return gen;
    }

    void fill_random(float *buf, size_t n, float lo = -1.0f, float hi = 1.0f)
    {
        std::uniform_real_distribution<float> dist(lo, hi);
        auto &g = rng();
        for (size_t i = 0; i < n; ++i)
            buf[i] = dist(g);
    }

    float cosine_similarity(const float *a, const float *b, size_t n)
    {
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
            nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }
        double denom = std::sqrt(na) * std::sqrt(nb);
        if (denom < 1e-12)
            return 1.0f;
        return static_cast<float>(dot / denom);
    }

    float max_abs_error(const float *a, const float *b, size_t n)
    {
        float mx = 0.0f;
        for (size_t i = 0; i < n; ++i)
            mx = std::max(mx, std::abs(a[i] - b[i]));
        return mx;
    }

    bool is_finite(const float *buf, size_t n)
    {
        for (size_t i = 0; i < n; ++i)
            if (!std::isfinite(buf[i]))
                return false;
        return true;
    }

    /// Q16_1 quantisation introduces bounded error. Tolerances are looser than FP32
    /// but tight enough to catch addressing bugs (which produce cosine < 0.7).
    constexpr float Q16_COSINE_THRESHOLD = 0.995f;
    constexpr float Q16_MAX_ABS_TOLERANCE = 0.05f;

    /// Scale for Q16_1 quantisation — typical KV cache scale
    constexpr float KV_CACHE_SCALE = 8.0f;

    /**
     * @brief Rearrange FP32 K/V from POSITION_MAJOR to HEAD_MAJOR layout
     *
     * POSITION_MAJOR: [kv_len, n_kv_heads * head_dim]
     *   - Row k contains all heads' data: [head0_d0..head0_dN, head1_d0..head1_dN, ...]
     *
     * HEAD_MAJOR: [n_kv_heads * kv_len, head_dim]
     *   - First kv_len rows are head 0's positions
     *   - Next kv_len rows are head 1's positions, etc.
     *   - Row (h * kv_len + k) = head h, position k, [d0..dN]
     */
    std::vector<float> rearrange_to_head_major(
        const float *pos_major, int kv_len, int n_kv_heads, int head_dim)
    {
        const size_t total = static_cast<size_t>(n_kv_heads) * kv_len * head_dim;
        std::vector<float> head_major(total);

        for (int h = 0; h < n_kv_heads; ++h)
        {
            for (int k = 0; k < kv_len; ++k)
            {
                // Source: row k, head h at offset h * head_dim
                const float *src = pos_major + static_cast<size_t>(k) * n_kv_heads * head_dim +
                                   static_cast<size_t>(h) * head_dim;
                // Dest: row (h * kv_len + k), full row is head_dim wide
                float *dst = head_major.data() + (static_cast<size_t>(h) * kv_len + k) * head_dim;
                std::copy(src, src + head_dim, dst);
            }
        }
        return head_major;
    }

    /**
     * @brief Create a Q16_1Tensor from FP32 data in HEAD_MAJOR layout
     *
     * Shape: [n_kv_heads * kv_len, head_dim]
     * Block size chosen to match head_dim (64 → BLOCK_64, 128 → BLOCK_128)
     */
    std::unique_ptr<Q16_1Tensor> create_q16_head_major(
        const float *fp32_pos_major, int kv_len, int n_kv_heads, int head_dim)
    {
        // Rearrange to head-major
        auto hm_data = rearrange_to_head_major(fp32_pos_major, kv_len, n_kv_heads, head_dim);

        // Choose block size matching head_dim
        Q16BlockSize blk_size;
        if (head_dim <= 32)
            blk_size = Q16BlockSize::BLOCK_32;
        else if (head_dim <= 64)
            blk_size = Q16BlockSize::BLOCK_64;
        else
            blk_size = Q16BlockSize::BLOCK_128;

        const size_t rows = static_cast<size_t>(n_kv_heads) * kv_len;
        const size_t cols = static_cast<size_t>(head_dim);
        auto tensor = std::make_unique<Q16_1Tensor>(
            std::vector<size_t>{rows, cols}, blk_size);

        bool ok = tensor->copyFrom_fp32_fixed_scale(hm_data.data(), KV_CACHE_SCALE, head_dim);
        if (!ok)
            return nullptr;
        return tensor;
    }

    /**
     * @brief Create a HEAD_MAJOR Q16_1Tensor with extra physical rows per head.
     *
     * Runtime Q16 KV caches are allocated as [kv_head][max_seq_len][head_dim],
     * while decode attention receives a smaller logical kv_len. This helper
     * mirrors that physical ring-cache shape so tests can prove future rows do
     * not influence the currently active prefix.
     */
    std::unique_ptr<Q16_1Tensor> create_q16_head_major_physical(
        const float *fp32_pos_major, int physical_rows_per_head, int n_kv_heads, int head_dim)
    {
        auto hm_data = rearrange_to_head_major(fp32_pos_major, physical_rows_per_head, n_kv_heads, head_dim);

        Q16BlockSize blk_size;
        if (head_dim <= 32)
            blk_size = Q16BlockSize::BLOCK_32;
        else if (head_dim <= 64)
            blk_size = Q16BlockSize::BLOCK_64;
        else
            blk_size = Q16BlockSize::BLOCK_128;

        const size_t rows = static_cast<size_t>(n_kv_heads) * physical_rows_per_head;
        const size_t cols = static_cast<size_t>(head_dim);
        auto tensor = std::make_unique<Q16_1Tensor>(
            std::vector<size_t>{rows, cols}, blk_size);

        bool ok = tensor->copyFrom_fp32_fixed_scale(hm_data.data(), KV_CACHE_SCALE, head_dim);
        if (!ok)
            return nullptr;
        return tensor;
    }

    /**
     * @brief Create a Q16_1Tensor from FP32 data in POSITION_MAJOR layout
     *
     * Shape: [kv_len, n_kv_heads * head_dim]
     * Block size chosen to match head_dim
     */
    std::unique_ptr<Q16_1Tensor> create_q16_position_major(
        const float *fp32_pos_major, int kv_len, int n_kv_heads, int head_dim)
    {
        Q16BlockSize blk_size;
        if (head_dim <= 32)
            blk_size = Q16BlockSize::BLOCK_32;
        else if (head_dim <= 64)
            blk_size = Q16BlockSize::BLOCK_64;
        else
            blk_size = Q16BlockSize::BLOCK_128;

        const size_t rows = static_cast<size_t>(kv_len);
        const size_t cols = static_cast<size_t>(n_kv_heads) * head_dim;
        auto tensor = std::make_unique<Q16_1Tensor>(
            std::vector<size_t>{rows, cols}, blk_size);

        bool ok = tensor->copyFrom_fp32_fixed_scale(fp32_pos_major, KV_CACHE_SCALE, head_dim);
        if (!ok)
            return nullptr;
        return tensor;
    }

} // anonymous namespace

// ===========================================================================
// Test fixture
// ===========================================================================
class Test__Q16DecodeHeadMajor : public ::testing::Test
{
protected:
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> kernel_;

    void SetUp() override
    {
        rng().seed(12345);
    }

    /**
     * @brief Call compute_tensor() with Q16_1 K/V tensors for decode
     *
     * Wraps Q/output in FP32Tensor, passes Q16_1 K/V, and dispatches through
     * the public compute_tensor() API which routes to compute_decode_q16kv().
     * position_offset is computed internally as kv_len - seq_len (= kv_len - 1).
     */
    bool callQ16Decode(
        const float *Q_data, float *out_data,
        const Q16_1Tensor *K_q16, const Q16_1Tensor *V_q16,
        int kv_len, int n_heads, int n_kv_heads, int head_dim,
        bool causal = true)
    {
        const size_t q_size = static_cast<size_t>(n_heads) * head_dim;

        // Wrap Q in FP32Tensor — shape [1, n_heads * head_dim] (seq_len=1 for decode)
        FP32Tensor Q_tensor({1, q_size});
        std::memcpy(Q_tensor.mutable_data(), Q_data, q_size * sizeof(float));

        // Wrap output in FP32Tensor
        FP32Tensor O_tensor({1, q_size});
        std::memset(O_tensor.mutable_data(), 0, q_size * sizeof(float));

        // seq_len=1 (decode), kv_len > seq_len triggers Q16 decode path
        bool ok = kernel_.compute_tensor(
            &Q_tensor,
            K_q16,
            V_q16,
            &O_tensor,
            /*batch_size=*/1,
            /*seq_len=*/1,
            /*kv_len=*/kv_len,
            n_heads, n_kv_heads, head_dim,
            causal);

        if (ok)
        {
            std::memcpy(out_data, O_tensor.data(), q_size * sizeof(float));
        }
        return ok;
    }

    /**
     * @brief Call the explicit grouped verifier attention hook.
     *
     * MTP verifier graphs use this API directly.  Comparing it to repeated
     * one-row decode calls at this boundary catches grouped-kernel drift before
     * it can hide inside a larger graph parity failure.
     */
    bool callQ16GroupedVerifier(
        const float *Q_data, float *out_data,
        const Q16_1Tensor *K_q16, const Q16_1Tensor *V_q16,
        int verifier_rows, int kv_len, int n_heads, int n_kv_heads, int head_dim)
    {
        const size_t q_size = static_cast<size_t>(verifier_rows) *
                              static_cast<size_t>(n_heads) *
                              static_cast<size_t>(head_dim);

        FP32Tensor Q_tensor({static_cast<size_t>(verifier_rows),
                             static_cast<size_t>(n_heads) * head_dim});
        std::memcpy(Q_tensor.mutable_data(), Q_data, q_size * sizeof(float));

        FP32Tensor O_tensor({static_cast<size_t>(verifier_rows),
                             static_cast<size_t>(n_heads) * head_dim});
        std::memset(O_tensor.mutable_data(), 0, q_size * sizeof(float));

        const bool ok = kernel_.compute_verifier_rows_decode_equivalent(
            &Q_tensor,
            K_q16,
            V_q16,
            &O_tensor,
            verifier_rows,
            kv_len,
            n_heads,
            n_kv_heads,
            head_dim,
            /*causal=*/true);

        if (ok)
        {
            std::memcpy(out_data, O_tensor.data(), q_size * sizeof(float));
        }
        return ok;
    }

    /**
     * @brief Core test: run Q16 decode with HEAD_MAJOR layout, compare to FP32 ref
     */
    void runHeadMajorDecodeTest(
        int kv_len, int n_heads, int n_kv_heads, int head_dim,
        const char *label)
    {
        const size_t q_size = static_cast<size_t>(n_heads) * head_dim;
        const size_t kv_size = static_cast<size_t>(kv_len) * n_kv_heads * head_dim;

        std::vector<float> Q(q_size), K_fp32(kv_size), V_fp32(kv_size);
        fill_random(Q.data(), q_size);
        fill_random(K_fp32.data(), kv_size);
        fill_random(V_fp32.data(), kv_size);

        auto K_q16_hm = create_q16_head_major(K_fp32.data(), kv_len, n_kv_heads, head_dim);
        auto V_q16_hm = create_q16_head_major(V_fp32.data(), kv_len, n_kv_heads, head_dim);
        ASSERT_NE(K_q16_hm, nullptr) << label << ": failed to create HEAD_MAJOR K tensor";
        ASSERT_NE(V_q16_hm, nullptr) << label << ": failed to create HEAD_MAJOR V tensor";

        ASSERT_EQ(K_q16_hm->rows(), static_cast<size_t>(n_kv_heads) * kv_len)
            << label << ": K rows mismatch";
        ASSERT_EQ(K_q16_hm->cols(), static_cast<size_t>(head_dim))
            << label << ": K cols mismatch";

        std::vector<float> out_q16(q_size, 0.0f);

        bool ok = callQ16Decode(
            Q.data(), out_q16.data(),
            K_q16_hm.get(), V_q16_hm.get(),
            kv_len, n_heads, n_kv_heads, head_dim);
        ASSERT_TRUE(ok) << label << ": compute_tensor() failed";
        ASSERT_TRUE(is_finite(out_q16.data(), q_size))
            << label << ": Q16 HEAD_MAJOR output has NaN/Inf";

        // Reference uses position_offset = kv_len - 1 (same as compute_tensor internally)
        std::vector<float> out_ref(q_size, 0.0f);
        ref::attention(Q.data(), K_fp32.data(), V_fp32.data(), out_ref.data(),
                       1, kv_len, n_heads, n_kv_heads, head_dim,
                       true, kv_len - 1);
        ASSERT_TRUE(is_finite(out_ref.data(), q_size))
            << label << ": reference output has NaN/Inf";

        float cos = cosine_similarity(out_q16.data(), out_ref.data(), q_size);
        float mae = max_abs_error(out_q16.data(), out_ref.data(), q_size);

        EXPECT_GE(cos, Q16_COSINE_THRESHOLD)
            << label << ": HEAD_MAJOR cosine " << cos << " < " << Q16_COSINE_THRESHOLD
            << " (addressing bug likely)";
        EXPECT_LE(mae, Q16_MAX_ABS_TOLERANCE)
            << label << ": HEAD_MAJOR max abs error " << mae << " > " << Q16_MAX_ABS_TOLERANCE;
    }

    /**
     * @brief Compare HEAD_MAJOR vs POSITION_MAJOR Q16 decode outputs
     *
     * Both layouts should produce identical results if addressing is correct.
     */
    void runLayoutEquivalenceTest(
        int kv_len, int n_heads, int n_kv_heads, int head_dim,
        const char *label)
    {
        const size_t q_size = static_cast<size_t>(n_heads) * head_dim;
        const size_t kv_size = static_cast<size_t>(kv_len) * n_kv_heads * head_dim;

        std::vector<float> Q(q_size), K_fp32(kv_size), V_fp32(kv_size);
        fill_random(Q.data(), q_size);
        fill_random(K_fp32.data(), kv_size);
        fill_random(V_fp32.data(), kv_size);

        auto K_q16_hm = create_q16_head_major(K_fp32.data(), kv_len, n_kv_heads, head_dim);
        auto V_q16_hm = create_q16_head_major(V_fp32.data(), kv_len, n_kv_heads, head_dim);
        auto K_q16_pm = create_q16_position_major(K_fp32.data(), kv_len, n_kv_heads, head_dim);
        auto V_q16_pm = create_q16_position_major(V_fp32.data(), kv_len, n_kv_heads, head_dim);
        ASSERT_NE(K_q16_hm, nullptr);
        ASSERT_NE(V_q16_hm, nullptr);
        ASSERT_NE(K_q16_pm, nullptr);
        ASSERT_NE(V_q16_pm, nullptr);

        const int position_offset = kv_len - 1;

        std::vector<float> out_hm(q_size, 0.0f);
        bool ok_hm = callQ16Decode(
            Q.data(), out_hm.data(),
            K_q16_hm.get(), V_q16_hm.get(),
            kv_len, n_heads, n_kv_heads, head_dim);
        ASSERT_TRUE(ok_hm) << label << ": HEAD_MAJOR compute failed";

        std::vector<float> out_pm(q_size, 0.0f);
        bool ok_pm = callQ16Decode(
            Q.data(), out_pm.data(),
            K_q16_pm.get(), V_q16_pm.get(),
            kv_len, n_heads, n_kv_heads, head_dim);
        ASSERT_TRUE(ok_pm) << label << ": POSITION_MAJOR compute failed";

        ASSERT_TRUE(is_finite(out_hm.data(), q_size)) << label << ": HEAD_MAJOR has NaN/Inf";
        ASSERT_TRUE(is_finite(out_pm.data(), q_size)) << label << ": POSITION_MAJOR has NaN/Inf";

        float cos = cosine_similarity(out_hm.data(), out_pm.data(), q_size);
        float mae = max_abs_error(out_hm.data(), out_pm.data(), q_size);

        EXPECT_GE(cos, 0.9999f)
            << label << ": HEAD_MAJOR vs POSITION_MAJOR cosine " << cos
            << " — layouts not equivalent (addressing bug)";
        EXPECT_LE(mae, 0.001f)
            << label << ": HEAD_MAJOR vs POSITION_MAJOR max abs error " << mae;
    }
};

// ===========================================================================
// HEAD_MAJOR decode accuracy vs FP32 reference
// ===========================================================================

// --- head_dim=64 (BLOCK_64) — the configuration that triggered the regression ---

TEST_F(Test__Q16DecodeHeadMajor, HeadDim64_2KVHeads_Short)
{
    // Qwen2.5-0.5B config: n_heads=14, n_kv_heads=2, head_dim=64
    runHeadMajorDecodeTest(16, 14, 2, 64, "HD64_2KV_Short");
}

TEST_F(Test__Q16DecodeHeadMajor, HeadDim64_2KVHeads_Medium)
{
    runHeadMajorDecodeTest(128, 14, 2, 64, "HD64_2KV_Medium");
}

TEST_F(Test__Q16DecodeHeadMajor, HeadDim64_2KVHeads_Long)
{
    runHeadMajorDecodeTest(512, 14, 2, 64, "HD64_2KV_Long");
}

TEST_F(Test__Q16DecodeHeadMajor, HeadDim64_4KVHeads)
{
    // 4 KV heads, 8 query heads — exercises GQA with head_dim=64
    runHeadMajorDecodeTest(64, 8, 4, 64, "HD64_4KV");
}

TEST_F(Test__Q16DecodeHeadMajor, HeadDim64_8KVHeads)
{
    // 8 KV heads, 16 query heads — no GQA
    runHeadMajorDecodeTest(64, 16, 8, 64, "HD64_8KV");
}

TEST_F(Test__Q16DecodeHeadMajor, HeadDim64_OddKVLen)
{
    // Non-power-of-2 kv_len to test tile boundary handling
    runHeadMajorDecodeTest(37, 14, 2, 64, "HD64_OddKV");
}

// --- head_dim=128 (BLOCK_128) — standard Llama/Qwen3 config ---

TEST_F(Test__Q16DecodeHeadMajor, HeadDim128_4KVHeads_Short)
{
    // Qwen2.5-7B config: n_heads=28, n_kv_heads=4, head_dim=128
    runHeadMajorDecodeTest(16, 28, 4, 128, "HD128_4KV_Short");
}

TEST_F(Test__Q16DecodeHeadMajor, HeadDim128_4KVHeads_Medium)
{
    runHeadMajorDecodeTest(128, 28, 4, 128, "HD128_4KV_Medium");
}

TEST_F(Test__Q16DecodeHeadMajor, HeadDim128_4KVHeads_Long)
{
    runHeadMajorDecodeTest(512, 28, 4, 128, "HD128_4KV_Long");
}

TEST_F(Test__Q16DecodeHeadMajor, HeadDim128_8KVHeads)
{
    // Qwen3-0.6B: n_heads=16, n_kv_heads=8, head_dim=128
    runHeadMajorDecodeTest(64, 16, 8, 128, "HD128_8KV");
}

TEST_F(Test__Q16DecodeHeadMajor, HeadDim128_OddKVLen)
{
    runHeadMajorDecodeTest(37, 28, 4, 128, "HD128_OddKV");
}

// ===========================================================================
// HEAD_MAJOR vs POSITION_MAJOR equivalence
// ===========================================================================

TEST_F(Test__Q16DecodeHeadMajor, LayoutEquivalence_HeadDim64_2KV)
{
    runLayoutEquivalenceTest(64, 14, 2, 64, "Equiv_HD64_2KV");
}

TEST_F(Test__Q16DecodeHeadMajor, LayoutEquivalence_HeadDim64_4KV)
{
    runLayoutEquivalenceTest(64, 8, 4, 64, "Equiv_HD64_4KV");
}

TEST_F(Test__Q16DecodeHeadMajor, LayoutEquivalence_HeadDim128_4KV)
{
    runLayoutEquivalenceTest(64, 28, 4, 128, "Equiv_HD128_4KV");
}

TEST_F(Test__Q16DecodeHeadMajor, LayoutEquivalence_HeadDim128_8KV)
{
    runLayoutEquivalenceTest(64, 16, 8, 128, "Equiv_HD128_8KV");
}

TEST_F(Test__Q16DecodeHeadMajor, LayoutEquivalence_LongContext)
{
    // Longer context to exercise multiple KV tiles
    runLayoutEquivalenceTest(256, 14, 2, 64, "Equiv_Long_HD64");
}

TEST_F(Test__Q16DecodeHeadMajor, HeadMajorPhysicalRowsIgnoreFuturePrefix)
{
    const int logical_kv_len = 37;
    const int physical_rows_per_head = 64;
    const int n_heads = 16;
    const int n_kv_heads = 8;
    const int head_dim = 128;
    const size_t q_size = static_cast<size_t>(n_heads) * head_dim;
    const size_t logical_kv_size = static_cast<size_t>(logical_kv_len) * n_kv_heads * head_dim;
    const size_t physical_kv_size = static_cast<size_t>(physical_rows_per_head) * n_kv_heads * head_dim;

    std::vector<float> Q(q_size), K_logical(logical_kv_size), V_logical(logical_kv_size);
    std::vector<float> K_physical(physical_kv_size), V_physical(physical_kv_size);
    fill_random(Q.data(), q_size);
    fill_random(K_logical.data(), logical_kv_size);
    fill_random(V_logical.data(), logical_kv_size);

    const size_t row_width = static_cast<size_t>(n_kv_heads) * head_dim;
    for (int row = 0; row < physical_rows_per_head; ++row)
    {
        float *k_row = K_physical.data() + static_cast<size_t>(row) * row_width;
        float *v_row = V_physical.data() + static_cast<size_t>(row) * row_width;
        if (row < logical_kv_len)
        {
            std::copy(K_logical.data() + static_cast<size_t>(row) * row_width,
                      K_logical.data() + static_cast<size_t>(row + 1) * row_width,
                      k_row);
            std::copy(V_logical.data() + static_cast<size_t>(row) * row_width,
                      V_logical.data() + static_cast<size_t>(row + 1) * row_width,
                      v_row);
        }
        else
        {
            // Future physical rows should be completely invisible while the
            // logical kv_len is still inside the prefix. Large values make any
            // accidental read show up as a large numerical mismatch.
            fill_random(k_row, row_width, 80.0f, 120.0f);
            fill_random(v_row, row_width, -120.0f, -80.0f);
        }
    }

    auto K_compact = create_q16_head_major(K_logical.data(), logical_kv_len, n_kv_heads, head_dim);
    auto V_compact = create_q16_head_major(V_logical.data(), logical_kv_len, n_kv_heads, head_dim);
    auto K_physical_q16 = create_q16_head_major_physical(
        K_physical.data(), physical_rows_per_head, n_kv_heads, head_dim);
    auto V_physical_q16 = create_q16_head_major_physical(
        V_physical.data(), physical_rows_per_head, n_kv_heads, head_dim);
    ASSERT_NE(K_compact, nullptr);
    ASSERT_NE(V_compact, nullptr);
    ASSERT_NE(K_physical_q16, nullptr);
    ASSERT_NE(V_physical_q16, nullptr);

    std::vector<float> out_compact(q_size, 0.0f);
    std::vector<float> out_physical(q_size, 0.0f);
    ASSERT_TRUE(callQ16Decode(
        Q.data(), out_compact.data(),
        K_compact.get(), V_compact.get(),
        logical_kv_len, n_heads, n_kv_heads, head_dim));
    ASSERT_TRUE(callQ16Decode(
        Q.data(), out_physical.data(),
        K_physical_q16.get(), V_physical_q16.get(),
        logical_kv_len, n_heads, n_kv_heads, head_dim));

    const float cos = cosine_similarity(out_physical.data(), out_compact.data(), q_size);
    const float mae = max_abs_error(out_physical.data(), out_compact.data(), q_size);
    EXPECT_GE(cos, 0.9999f) << "Physical HEAD_MAJOR cache rows leaked into logical prefix";
    EXPECT_LE(mae, 0.001f) << "Physical HEAD_MAJOR cache rows changed logical prefix decode";
}

// ===========================================================================
// Edge cases
// ===========================================================================

TEST_F(Test__Q16DecodeHeadMajor, SingleKVPosition)
{
    // kv_len=1: degenerate case, attention should output V directly (scaled)
    runHeadMajorDecodeTest(1, 4, 2, 64, "SingleKV_HD64");
}

TEST_F(Test__Q16DecodeHeadMajor, SingleKVPosition_HD128)
{
    runHeadMajorDecodeTest(1, 4, 2, 128, "SingleKV_HD128");
}

TEST_F(Test__Q16DecodeHeadMajor, GroupedVerifierRowsMatchSerialDecode_M2ToM4)
{
    /*
     * Production Q16 KV caches are head-major, while the MTP verifier presents
     * a compact group of candidate rows.  Row r in the group must match serial
     * decode at BASE_KV + r + 1 visible KV positions.
     */
    constexpr int BASE_KV = 19;
    constexpr int MAX_M = 4;
    constexpr int FULL_KV = BASE_KV + MAX_M;
    constexpr int N_HEADS = 8;
    constexpr int N_KV_HEADS = 4;
    constexpr int HEAD_DIM = 64;
    constexpr int Q_DIM = N_HEADS * HEAD_DIM;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;

    rng().seed(88001);
    std::vector<float> Q_all(static_cast<size_t>(MAX_M) * Q_DIM);
    std::vector<float> K_fp32(static_cast<size_t>(FULL_KV) * KV_DIM);
    std::vector<float> V_fp32(static_cast<size_t>(FULL_KV) * KV_DIM);
    fill_random(Q_all.data(), Q_all.size(), -0.75f, 0.75f);
    fill_random(K_fp32.data(), K_fp32.size(), -0.5f, 0.5f);
    fill_random(V_fp32.data(), V_fp32.size(), -0.5f, 0.5f);

    auto K_q16 = create_q16_head_major(K_fp32.data(), FULL_KV, N_KV_HEADS, HEAD_DIM);
    auto V_q16 = create_q16_head_major(V_fp32.data(), FULL_KV, N_KV_HEADS, HEAD_DIM);
    ASSERT_NE(K_q16, nullptr);
    ASSERT_NE(V_q16, nullptr);

    for (int m = 2; m <= MAX_M; ++m)
    {
        const int kv_len = BASE_KV + m;
        std::vector<float> grouped(static_cast<size_t>(m) * Q_DIM, 0.0f);
        ASSERT_TRUE(callQ16GroupedVerifier(
            Q_all.data(),
            grouped.data(),
            K_q16.get(),
            V_q16.get(),
            m,
            kv_len,
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM))
            << "M=" << m;

        for (int row = 0; row < m; ++row)
        {
            std::vector<float> serial(Q_DIM, 0.0f);
            ASSERT_TRUE(callQ16Decode(
                Q_all.data() + static_cast<size_t>(row) * Q_DIM,
                serial.data(),
                K_q16.get(),
                V_q16.get(),
                BASE_KV + row + 1,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM))
                << "M=" << m << " row=" << row;

            const float *grouped_row = grouped.data() + static_cast<size_t>(row) * Q_DIM;
            const float max_diff = max_abs_error(grouped_row, serial.data(), Q_DIM);
            const float cos = cosine_similarity(grouped_row, serial.data(), Q_DIM);
            EXPECT_LT(max_diff, 1e-5f) << "M=" << m << " row=" << row;
            EXPECT_GT(cos, 0.999999f) << "M=" << m << " row=" << row;
        }
    }
}

TEST_F(Test__Q16DecodeHeadMajor, GroupedVerifierRowsMatchSerialDecode_Qwen36Shape_M2ToM4)
{
    /*
     * Qwen3.6 dense attention uses a wider head layout than the small unit
     * case above.  This regression keeps the grouped verifier path honest for
     * the model shape that exposed the first real parity drift.
     */
    constexpr int BASE_KV = 37;
    constexpr int MAX_M = 4;
    constexpr int FULL_KV = BASE_KV + MAX_M;
    constexpr int N_HEADS = 28;
    constexpr int N_KV_HEADS = 4;
    constexpr int HEAD_DIM = 128;
    constexpr int Q_DIM = N_HEADS * HEAD_DIM;
    constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;

    rng().seed(88002);
    std::vector<float> Q_all(static_cast<size_t>(MAX_M) * Q_DIM);
    std::vector<float> K_fp32(static_cast<size_t>(FULL_KV) * KV_DIM);
    std::vector<float> V_fp32(static_cast<size_t>(FULL_KV) * KV_DIM);
    fill_random(Q_all.data(), Q_all.size(), -0.25f, 0.25f);
    fill_random(K_fp32.data(), K_fp32.size(), -0.25f, 0.25f);
    fill_random(V_fp32.data(), V_fp32.size(), -0.25f, 0.25f);

    auto K_q16 = create_q16_head_major(K_fp32.data(), FULL_KV, N_KV_HEADS, HEAD_DIM);
    auto V_q16 = create_q16_head_major(V_fp32.data(), FULL_KV, N_KV_HEADS, HEAD_DIM);
    ASSERT_NE(K_q16, nullptr);
    ASSERT_NE(V_q16, nullptr);

    for (int m = 2; m <= MAX_M; ++m)
    {
        const int kv_len = BASE_KV + m;
        std::vector<float> grouped(static_cast<size_t>(m) * Q_DIM, 0.0f);
        ASSERT_TRUE(callQ16GroupedVerifier(
            Q_all.data(),
            grouped.data(),
            K_q16.get(),
            V_q16.get(),
            m,
            kv_len,
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM))
            << "M=" << m;

        for (int row = 0; row < m; ++row)
        {
            std::vector<float> serial(Q_DIM, 0.0f);
            ASSERT_TRUE(callQ16Decode(
                Q_all.data() + static_cast<size_t>(row) * Q_DIM,
                serial.data(),
                K_q16.get(),
                V_q16.get(),
                BASE_KV + row + 1,
                N_HEADS,
                N_KV_HEADS,
                HEAD_DIM))
                << "M=" << m << " row=" << row;

            const float *grouped_row = grouped.data() + static_cast<size_t>(row) * Q_DIM;
            const float max_diff = max_abs_error(grouped_row, serial.data(), Q_DIM);
            const float cos = cosine_similarity(grouped_row, serial.data(), Q_DIM);
            EXPECT_LT(max_diff, 1e-5f) << "M=" << m << " row=" << row;
            EXPECT_GT(cos, 0.999999f) << "M=" << m << " row=" << row;
        }
    }
}

TEST_F(Test__Q16DecodeHeadMajor, NoCausal)
{
    // Non-causal attention with HEAD_MAJOR layout
    const int kv_len = 64;
    const int n_heads = 14;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const size_t q_size = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_size = static_cast<size_t>(kv_len) * n_kv_heads * head_dim;

    std::vector<float> Q(q_size), K_fp32(kv_size), V_fp32(kv_size);
    fill_random(Q.data(), q_size);
    fill_random(K_fp32.data(), kv_size);
    fill_random(V_fp32.data(), kv_size);

    auto K_q16 = create_q16_head_major(K_fp32.data(), kv_len, n_kv_heads, head_dim);
    auto V_q16 = create_q16_head_major(V_fp32.data(), kv_len, n_kv_heads, head_dim);
    ASSERT_NE(K_q16, nullptr);
    ASSERT_NE(V_q16, nullptr);

    std::vector<float> out_q16(q_size, 0.0f);
    bool ok = callQ16Decode(
        Q.data(), out_q16.data(),
        K_q16.get(), V_q16.get(),
        kv_len, n_heads, n_kv_heads, head_dim,
        /*causal=*/false);
    ASSERT_TRUE(ok) << "NoCausal: compute_tensor() failed";
    ASSERT_TRUE(is_finite(out_q16.data(), q_size)) << "NoCausal: output has NaN/Inf";

    // For non-causal, position_offset doesn't affect results
    std::vector<float> out_ref(q_size, 0.0f);
    ref::attention(Q.data(), K_fp32.data(), V_fp32.data(), out_ref.data(),
                   1, kv_len, n_heads, n_kv_heads, head_dim,
                   false, kv_len - 1);

    float cos = cosine_similarity(out_q16.data(), out_ref.data(), q_size);
    EXPECT_GE(cos, Q16_COSINE_THRESHOLD)
        << "NoCausal: cosine " << cos << " below threshold";
}

// ===========================================================================
// MQA (Multi-Query Attention) — single KV head
// Note: n_kv_heads=1 does not trigger HEAD_MAJOR detection (is_head_major
// requires n_kv_heads > 1), but it's important to verify it still works.
// ===========================================================================

TEST_F(Test__Q16DecodeHeadMajor, MQA_SingleKVHead_HD64)
{
    // MQA: 8 query heads, 1 KV head
    // With n_kv_heads=1, blocks_per_kv_row == blocks_per_head regardless of
    // layout, so the is_head_major heuristic returns false. The addressing
    // is identical for both layouts when n_kv_heads=1.
    const int kv_len = 64;
    const int n_heads = 8;
    const int n_kv_heads = 1;
    const int head_dim = 64;
    const size_t q_size = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_size = static_cast<size_t>(kv_len) * n_kv_heads * head_dim;

    std::vector<float> Q(q_size), K_fp32(kv_size), V_fp32(kv_size);
    fill_random(Q.data(), q_size);
    fill_random(K_fp32.data(), kv_size);
    fill_random(V_fp32.data(), kv_size);

    // For n_kv_heads=1, HEAD_MAJOR and POSITION_MAJOR are the same shape
    auto K_q16 = create_q16_position_major(K_fp32.data(), kv_len, n_kv_heads, head_dim);
    auto V_q16 = create_q16_position_major(V_fp32.data(), kv_len, n_kv_heads, head_dim);
    ASSERT_NE(K_q16, nullptr);
    ASSERT_NE(V_q16, nullptr);

    std::vector<float> out_q16(q_size, 0.0f);
    bool ok = callQ16Decode(
        Q.data(), out_q16.data(),
        K_q16.get(), V_q16.get(),
        kv_len, n_heads, n_kv_heads, head_dim);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(is_finite(out_q16.data(), q_size));

    std::vector<float> out_ref(q_size, 0.0f);
    ref::attention(Q.data(), K_fp32.data(), V_fp32.data(), out_ref.data(),
                   1, kv_len, n_heads, n_kv_heads, head_dim,
                   true, kv_len - 1);

    float cos = cosine_similarity(out_q16.data(), out_ref.data(), q_size);
    EXPECT_GE(cos, Q16_COSINE_THRESHOLD)
        << "MQA_HD64: cosine " << cos << " below threshold";
}

TEST_F(Test__Q16DecodeHeadMajor, MQA_SingleKVHead_HD128)
{
    const int kv_len = 64;
    const int n_heads = 8;
    const int n_kv_heads = 1;
    const int head_dim = 128;
    const size_t q_size = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_size = static_cast<size_t>(kv_len) * n_kv_heads * head_dim;

    std::vector<float> Q(q_size), K_fp32(kv_size), V_fp32(kv_size);
    fill_random(Q.data(), q_size);
    fill_random(K_fp32.data(), kv_size);
    fill_random(V_fp32.data(), kv_size);

    auto K_q16 = create_q16_position_major(K_fp32.data(), kv_len, n_kv_heads, head_dim);
    auto V_q16 = create_q16_position_major(V_fp32.data(), kv_len, n_kv_heads, head_dim);
    ASSERT_NE(K_q16, nullptr);
    ASSERT_NE(V_q16, nullptr);

    std::vector<float> out_q16(q_size, 0.0f);
    bool ok = callQ16Decode(
        Q.data(), out_q16.data(),
        K_q16.get(), V_q16.get(),
        kv_len, n_heads, n_kv_heads, head_dim);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(is_finite(out_q16.data(), q_size));

    std::vector<float> out_ref(q_size, 0.0f);
    ref::attention(Q.data(), K_fp32.data(), V_fp32.data(), out_ref.data(),
                   1, kv_len, n_heads, n_kv_heads, head_dim,
                   true, kv_len - 1);

    float cos = cosine_similarity(out_q16.data(), out_ref.data(), q_size);
    EXPECT_GE(cos, Q16_COSINE_THRESHOLD)
        << "MQA_HD128: cosine " << cos << " below threshold";
}
