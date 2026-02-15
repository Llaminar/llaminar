/**
 * @file Test__CPUFlashAttentionKernel.cpp
 * @brief Integration tests for CPUFlashAttentionKernelT mixed-precision tensor path
 */

#include <gtest/gtest.h>

#include "kernels/cpu/attention/CPUAttentionKernelT.h"
#include "kernels/cpu/attention/CPUFlashAttentionKernelT.h"
#include "tensors/Tensors.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using namespace llaminar2;

namespace
{
    static std::vector<float> makeRandom(size_t count, uint32_t seed)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> v(count);
        for (size_t i = 0; i < count; ++i)
        {
            v[i] = dist(rng);
        }
        return v;
    }

    static float cosineSimilarity(const float *a, const float *b, size_t count)
    {
        double dot = 0.0;
        double na = 0.0;
        double nb = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            const double av = static_cast<double>(a[i]);
            const double bv = static_cast<double>(b[i]);
            dot += av * bv;
            na += av * av;
            nb += bv * bv;
        }
        if (na < 1e-20 || nb < 1e-20)
        {
            return 0.0f;
        }
        return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
    }

    static float maxAbsDiff(const float *a, const float *b, size_t count)
    {
        float m = 0.0f;
        for (size_t i = 0; i < count; ++i)
        {
            m = std::max(m, std::abs(a[i] - b[i]));
        }
        return m;
    }
}

TEST(Test__CPUFlashAttentionKernel, ComputeTensor_Prefill_FP32Q_Q81KV_MatchesReference)
{
    constexpr int batch_size = 1;
    constexpr int seq_len = 8;
    constexpr int kv_len = 8;
    constexpr int n_heads = 8;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    constexpr bool causal = true;

    const size_t q_cols = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_cols = static_cast<size_t>(n_kv_heads) * head_dim;
    const size_t out_count = static_cast<size_t>(seq_len) * q_cols;

    auto q_data = makeRandom(static_cast<size_t>(seq_len) * q_cols, 1001);
    auto k_data = makeRandom(static_cast<size_t>(kv_len) * kv_cols, 1002);
    auto v_data = makeRandom(static_cast<size_t>(kv_len) * kv_cols, 1003);

    FP32Tensor q_tensor({static_cast<size_t>(seq_len), q_cols});
    FP32Tensor out_tensor({static_cast<size_t>(seq_len), q_cols});
    FP32Tensor ref_tensor({static_cast<size_t>(seq_len), q_cols});

    std::copy(q_data.begin(), q_data.end(), q_tensor.mutable_data());

    auto k_q81 = Q8_1Tensor::quantize_from_fp32(k_data.data(), {static_cast<size_t>(kv_len), kv_cols});
    auto v_q81 = Q8_1Tensor::quantize_from_fp32(v_data.data(), {static_cast<size_t>(kv_len), kv_cols});
    ASSERT_NE(k_q81, nullptr);
    ASSERT_NE(v_q81, nullptr);

    CPUFlashAttentionKernelT<ActivationPrecision::FP32> flash_kernel;
    ASSERT_TRUE(flash_kernel.compute_tensor(
        &q_tensor,
        k_q81.get(),
        v_q81.get(),
        &out_tensor,
        batch_size,
        seq_len,
        kv_len,
        n_heads,
        n_kv_heads,
        head_dim,
        causal));

    CPUAttentionKernelT<ActivationPrecision::FP32> ref_kernel;
    const float *k_deq = k_q81->fp32_data();
    const float *v_deq = v_q81->fp32_data();
    ASSERT_NE(k_deq, nullptr);
    ASSERT_NE(v_deq, nullptr);
    ASSERT_TRUE(ref_kernel.compute(
        q_tensor.data(),
        k_deq,
        v_deq,
        ref_tensor.mutable_data(),
        seq_len,
        n_heads,
        n_kv_heads,
        head_dim,
        causal));

    const float cos = cosineSimilarity(out_tensor.data(), ref_tensor.data(), out_count);
    const float max_diff = maxAbsDiff(out_tensor.data(), ref_tensor.data(), out_count);

    EXPECT_GE(cos, 0.999f);
    EXPECT_LE(max_diff, 1e-3f);
}

TEST(Test__CPUFlashAttentionKernel, ComputeTensor_Decode_FP32Q_Q81KV_MatchesReference)
{
    constexpr int batch_size = 1;
    constexpr int seq_len = 1;
    constexpr int kv_len = 64;
    constexpr int n_heads = 14;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    constexpr bool causal = true;

    const size_t q_cols = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_cols = static_cast<size_t>(n_kv_heads) * head_dim;
    const size_t out_count = static_cast<size_t>(seq_len) * q_cols;

    auto q_data = makeRandom(static_cast<size_t>(seq_len) * q_cols, 2001);
    auto k_data = makeRandom(static_cast<size_t>(kv_len) * kv_cols, 2002);
    auto v_data = makeRandom(static_cast<size_t>(kv_len) * kv_cols, 2003);

    FP32Tensor q_tensor({static_cast<size_t>(seq_len), q_cols});
    FP32Tensor out_tensor({static_cast<size_t>(seq_len), q_cols});
    FP32Tensor ref_tensor({static_cast<size_t>(seq_len), q_cols});

    std::copy(q_data.begin(), q_data.end(), q_tensor.mutable_data());

    auto k_q81 = Q8_1Tensor::quantize_from_fp32(k_data.data(), {static_cast<size_t>(kv_len), kv_cols});
    auto v_q81 = Q8_1Tensor::quantize_from_fp32(v_data.data(), {static_cast<size_t>(kv_len), kv_cols});
    ASSERT_NE(k_q81, nullptr);
    ASSERT_NE(v_q81, nullptr);

    CPUFlashAttentionKernelT<ActivationPrecision::FP32> flash_kernel;
    ASSERT_TRUE(flash_kernel.compute_tensor(
        &q_tensor,
        k_q81.get(),
        v_q81.get(),
        &out_tensor,
        batch_size,
        seq_len,
        kv_len,
        n_heads,
        n_kv_heads,
        head_dim,
        causal));

    CPUAttentionKernelT<ActivationPrecision::FP32> ref_kernel;
    const float *k_deq = k_q81->fp32_data();
    const float *v_deq = v_q81->fp32_data();
    ASSERT_NE(k_deq, nullptr);
    ASSERT_NE(v_deq, nullptr);
    ASSERT_TRUE(ref_kernel.compute_decode(
        q_tensor.data(),
        k_deq,
        v_deq,
        ref_tensor.mutable_data(),
        seq_len,
        kv_len,
        n_heads,
        n_kv_heads,
        head_dim,
        causal,
        kv_len - seq_len));

    const float cos = cosineSimilarity(out_tensor.data(), ref_tensor.data(), out_count);
    const float max_diff = maxAbsDiff(out_tensor.data(), ref_tensor.data(), out_count);

    EXPECT_GE(cos, 0.999f);
    EXPECT_LE(max_diff, 1e-3f);
}
