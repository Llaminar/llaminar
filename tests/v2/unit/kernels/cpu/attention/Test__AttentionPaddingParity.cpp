/**
 * @file Test__AttentionPaddingParity.cpp
 * @brief CPU attention parity tests for padded prefill bucket rows.
 *
 * These tests validate the first safety assumption behind Phase 6 prefill graph
 * buckets: real prompt rows must not attend to future padding rows when causal
 * attention is enabled. The test calls the CPU FP32 attention kernel directly,
 * avoiding graph capture and bucket execution so failures isolate masking logic.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "v2/kernels/cpu/attention/CPUFlashAttentionKernelT.h"
#include "v2/utils/DebugEnv.h"

using namespace llaminar2;

namespace
{
    /**
     * @brief Fill real Q/K/V rows with deterministic low-magnitude values.
     *
     * Keeping values small makes the padded and unpadded causal outputs compare
     * tightly while still producing non-trivial attention distributions.
     */
    void fillRealRows(
        std::vector<float> &q,
        std::vector<float> &k,
        std::vector<float> &v,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim)
    {
        const size_t q_cols = static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim);
        const size_t kv_cols = static_cast<size_t>(n_kv_heads) * static_cast<size_t>(head_dim);

        for (int row = 0; row < seq_len; ++row)
        {
            for (size_t col = 0; col < q_cols; ++col)
                q[static_cast<size_t>(row) * q_cols + col] = 0.01f * static_cast<float>((row + static_cast<int>(col)) % 17 + 1);
            for (size_t col = 0; col < kv_cols; ++col)
            {
                k[static_cast<size_t>(row) * kv_cols + col] = 0.02f * static_cast<float>((row * 3 + static_cast<int>(col)) % 13 + 1);
                v[static_cast<size_t>(row) * kv_cols + col] = 0.03f * static_cast<float>((row * 5 + static_cast<int>(col)) % 19 + 1);
            }
        }
    }

    /**
     * @brief Fill bucket-tail padding rows with values that would corrupt output if visible.
     */
    void fillHostilePadRows(
        std::vector<float> &q,
        std::vector<float> &k,
        std::vector<float> &v,
        int real_seq_len,
        int bucket_seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim)
    {
        const size_t q_cols = static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim);
        const size_t kv_cols = static_cast<size_t>(n_kv_heads) * static_cast<size_t>(head_dim);

        for (int row = real_seq_len; row < bucket_seq_len; ++row)
        {
            for (size_t col = 0; col < q_cols; ++col)
                q[static_cast<size_t>(row) * q_cols + col] = 7.0f;
            for (size_t col = 0; col < kv_cols; ++col)
            {
                k[static_cast<size_t>(row) * kv_cols + col] = 8.0f;
                v[static_cast<size_t>(row) * kv_cols + col] = (col % 2 == 0) ? 11.0f : -11.0f;
            }
        }
    }

    /// @brief Returns the largest absolute difference over the real output prefix.
    float maxRealRowDiff(
        const std::vector<float> &a,
        const std::vector<float> &b,
        int real_seq_len,
        int n_heads,
        int head_dim)
    {
        const size_t real_count = static_cast<size_t>(real_seq_len) * static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim);
        float max_diff = 0.0f;
        for (size_t i = 0; i < real_count; ++i)
            max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
        return max_diff;
    }
} // namespace

class Test__AttentionPaddingParity : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Force the FP32 path so this test validates causal masking, not VNNI
        // quantization or CPU feature dispatch differences.
        setenv("LLAMINAR_FLASH_PREFILL_I16_I12", "0", 1);
        mutableDebugEnv().attention.reload();
    }

    void TearDown() override
    {
        unsetenv("LLAMINAR_FLASH_PREFILL_I16_I12");
        mutableDebugEnv().attention.reload();
    }

    CPUFlashAttentionKernelT<ActivationPrecision::FP32> kernel_;
};

TEST_F(Test__AttentionPaddingParity, CausalMaskPreventsFuturePaddingFromChangingRealRows)
{
    constexpr int real_seq_len = 48;
    constexpr int bucket_seq_len = 80;
    constexpr int n_heads = 4;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 16;
    const size_t q_cols = static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim);
    const size_t kv_cols = static_cast<size_t>(n_kv_heads) * static_cast<size_t>(head_dim);

    std::vector<float> q_real(static_cast<size_t>(real_seq_len) * q_cols);
    std::vector<float> k_real(static_cast<size_t>(real_seq_len) * kv_cols);
    std::vector<float> v_real(static_cast<size_t>(real_seq_len) * kv_cols);
    fillRealRows(q_real, k_real, v_real, real_seq_len, n_heads, n_kv_heads, head_dim);

    std::vector<float> q_padded(static_cast<size_t>(bucket_seq_len) * q_cols, 0.0f);
    std::vector<float> k_padded(static_cast<size_t>(bucket_seq_len) * kv_cols, 0.0f);
    std::vector<float> v_padded(static_cast<size_t>(bucket_seq_len) * kv_cols, 0.0f);

    for (int row = 0; row < real_seq_len; ++row)
    {
        std::copy_n(q_real.begin() + static_cast<size_t>(row) * q_cols, q_cols,
                    q_padded.begin() + static_cast<size_t>(row) * q_cols);
        std::copy_n(k_real.begin() + static_cast<size_t>(row) * kv_cols, kv_cols,
                    k_padded.begin() + static_cast<size_t>(row) * kv_cols);
        std::copy_n(v_real.begin() + static_cast<size_t>(row) * kv_cols, kv_cols,
                    v_padded.begin() + static_cast<size_t>(row) * kv_cols);
    }
    fillHostilePadRows(q_padded, k_padded, v_padded, real_seq_len, bucket_seq_len, n_heads, n_kv_heads, head_dim);

    std::vector<float> out_real(q_real.size(), 0.0f);
    std::vector<float> out_padded(q_padded.size(), 0.0f);

    ASSERT_TRUE(kernel_.compute(q_real.data(), k_real.data(), v_real.data(), out_real.data(),
                                real_seq_len, n_heads, n_kv_heads, head_dim,
                                /*causal=*/true, /*window_size=*/-1));
    ASSERT_TRUE(kernel_.compute(q_padded.data(), k_padded.data(), v_padded.data(), out_padded.data(),
                                bucket_seq_len, n_heads, n_kv_heads, head_dim,
                                /*causal=*/true, /*window_size=*/-1));

    EXPECT_LT(maxRealRowDiff(out_real, out_padded, real_seq_len, n_heads, head_dim), 1e-5f);
}

TEST_F(Test__AttentionPaddingParity, NonCausalAttentionWouldExposeHostilePadding)
{
    constexpr int real_seq_len = 16;
    constexpr int bucket_seq_len = 24;
    constexpr int n_heads = 2;
    constexpr int n_kv_heads = 1;
    constexpr int head_dim = 8;
    const size_t q_cols = static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim);
    const size_t kv_cols = static_cast<size_t>(n_kv_heads) * static_cast<size_t>(head_dim);

    std::vector<float> q_real(static_cast<size_t>(real_seq_len) * q_cols);
    std::vector<float> k_real(static_cast<size_t>(real_seq_len) * kv_cols);
    std::vector<float> v_real(static_cast<size_t>(real_seq_len) * kv_cols);
    fillRealRows(q_real, k_real, v_real, real_seq_len, n_heads, n_kv_heads, head_dim);

    std::vector<float> q_padded(static_cast<size_t>(bucket_seq_len) * q_cols, 0.0f);
    std::vector<float> k_padded(static_cast<size_t>(bucket_seq_len) * kv_cols, 0.0f);
    std::vector<float> v_padded(static_cast<size_t>(bucket_seq_len) * kv_cols, 0.0f);
    for (int row = 0; row < real_seq_len; ++row)
    {
        std::copy_n(q_real.begin() + static_cast<size_t>(row) * q_cols, q_cols,
                    q_padded.begin() + static_cast<size_t>(row) * q_cols);
        std::copy_n(k_real.begin() + static_cast<size_t>(row) * kv_cols, kv_cols,
                    k_padded.begin() + static_cast<size_t>(row) * kv_cols);
        std::copy_n(v_real.begin() + static_cast<size_t>(row) * kv_cols, kv_cols,
                    v_padded.begin() + static_cast<size_t>(row) * kv_cols);
    }
    fillHostilePadRows(q_padded, k_padded, v_padded, real_seq_len, bucket_seq_len, n_heads, n_kv_heads, head_dim);

    std::vector<float> out_real(q_real.size(), 0.0f);
    std::vector<float> out_padded(q_padded.size(), 0.0f);

    ASSERT_TRUE(kernel_.compute(q_real.data(), k_real.data(), v_real.data(), out_real.data(),
                                real_seq_len, n_heads, n_kv_heads, head_dim,
                                /*causal=*/false, /*window_size=*/-1));
    ASSERT_TRUE(kernel_.compute(q_padded.data(), k_padded.data(), v_padded.data(), out_padded.data(),
                                bucket_seq_len, n_heads, n_kv_heads, head_dim,
                                /*causal=*/false, /*window_size=*/-1));

    EXPECT_GT(maxRealRowDiff(out_real, out_padded, real_seq_len, n_heads, head_dim), 1.0f);
}
