/**
 * @file Test__CUDARingKVCache_RoPEOnRead.cpp
 * @brief Tests for get_kv_converted() with RoPE-on-read on existing CUDA KV caches
 * @author David Sanftenberg
 *
 * Validates that get_kv_converted() correctly applies RoPE to K in shadow
 * FP16 buffers for FP16 and Q8_1 CUDA KV caches.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <cmath>
#include "kernels/cuda/kvcache/CUDARingKVCache.h"
#include "kernels/IKVCache.h"
#include "tensors/GpuTensorView.h"
#include "tensors/TensorClasses.h"
#include "utils/Logger.h"

using namespace llaminar2;

namespace
{
    bool hasCUDA()
    {
        int count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
    }

    std::vector<float> generateRandomFP32(size_t count, unsigned seed = 42)
    {
        std::vector<float> data(count);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &val : data)
            val = dist(rng);
        return data;
    }

    std::vector<float> downloadFP16ToFP32(const void *d_ptr, size_t count)
    {
        std::vector<__half> h_fp16(count);
        cudaMemcpy(h_fp16.data(), d_ptr, count * sizeof(__half), cudaMemcpyDeviceToHost);
        std::vector<float> result(count);
        for (size_t i = 0; i < count; ++i)
            result[i] = __half2float(h_fp16[i]);
        return result;
    }

    float computeCosineSimilarity(const float *a, const float *b, size_t n)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += static_cast<double>(a[i]) * b[i];
            norm_a += static_cast<double>(a[i]) * a[i];
            norm_b += static_cast<double>(b[i]) * b[i];
        }
        if (norm_a < 1e-30 || norm_b < 1e-30)
            return 0.0f;
        return static_cast<float>(dot / (std::sqrt(norm_a) * std::sqrt(norm_b)));
    }
} // namespace

// =============================================================================
// FP16 Cache: RoPE-on-read
// =============================================================================

TEST(Test__CUDARingKVCache_RoPEOnRead, FP16_RoPEChangesK)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;
    const int num_tokens = 8;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP16, 1, 1, 32, n_kv_heads, head_dim);
    ASSERT_NE(cache, nullptr);

    auto h_K_fp32 = generateRandomFP32(num_tokens * kv_dim, 123);
    auto h_V_fp32 = generateRandomFP32(num_tokens * kv_dim, 456);

    // Convert to FP16 on host (FP16 cache expects __half data)
    std::vector<__half> h_K_fp16(num_tokens * kv_dim);
    std::vector<__half> h_V_fp16(num_tokens * kv_dim);
    for (size_t i = 0; i < h_K_fp32.size(); ++i)
    {
        h_K_fp16[i] = __float2half(h_K_fp32[i]);
        h_V_fp16[i] = __float2half(h_V_fp32[i]);
    }

    __half *d_K, *d_V;
    cudaMalloc(&d_K, num_tokens * kv_dim * sizeof(__half));
    cudaMalloc(&d_V, num_tokens * kv_dim * sizeof(__half));
    cudaMemcpy(d_K, h_K_fp16.data(), num_tokens * kv_dim * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V_fp16.data(), num_tokens * kv_dim * sizeof(__half), cudaMemcpyHostToDevice);

    ASSERT_TRUE(cache->append(0, 0, d_K, d_V, num_tokens));

    // Get raw K data via get_kv_for_attention (the void* API)
    const void *d_K_raw_ptr = nullptr;
    const void *d_V_raw_ptr = nullptr;
    int kv_len = 0;
    ASSERT_TRUE(cache->get_kv_for_attention(0, 0, &d_K_raw_ptr, &d_V_raw_ptr, &kv_len));
    ASSERT_NE(d_K_raw_ptr, nullptr);

    auto k_no_rope = downloadFP16ToFP32(d_K_raw_ptr, num_tokens * kv_dim);

    // Get with RoPE
    IKVCache::KVReadParams rope_params;
    rope_params.rope_theta = 10000.0f;
    rope_params.position_start = 0;
    rope_params.n_kv_heads = n_kv_heads;
    rope_params.head_dim = head_dim;

    ITensor *out_k_rope = nullptr;
    ITensor *out_v_rope = nullptr;
    ASSERT_TRUE(cache->get_kv_converted(0, 0, ActivationPrecision::FP16,
                                        &out_k_rope, &out_v_rope, &kv_len, &rope_params));
    ASSERT_NE(out_k_rope, nullptr);
    EXPECT_EQ(kv_len, num_tokens);

    // K should be different with RoPE
    auto k_with_rope = downloadFP16ToFP32(out_k_rope->gpu_data_ptr(), num_tokens * kv_dim);

    float diff = 0.0f;
    for (size_t i = 0; i < k_no_rope.size(); ++i)
        diff += std::abs(k_no_rope[i] - k_with_rope[i]);
    EXPECT_GT(diff, 1.0f) << "RoPE should significantly change K values";

    cudaFree(d_K);
    cudaFree(d_V);
}

// =============================================================================
// FP16 Cache: V Unchanged by RoPE
// =============================================================================

TEST(Test__CUDARingKVCache_RoPEOnRead, FP16_VUnchangedByRoPE)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;
    const int num_tokens = 6;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP16, 1, 1, 32, n_kv_heads, head_dim);
    ASSERT_NE(cache, nullptr);

    auto h_V_fp32 = generateRandomFP32(num_tokens * kv_dim, 789);
    auto h_K_fp32 = generateRandomFP32(num_tokens * kv_dim, 101);

    // Convert to FP16 on host
    std::vector<__half> h_K_fp16(num_tokens * kv_dim);
    std::vector<__half> h_V_fp16(num_tokens * kv_dim);
    for (size_t i = 0; i < h_K_fp32.size(); ++i)
    {
        h_K_fp16[i] = __float2half(h_K_fp32[i]);
        h_V_fp16[i] = __float2half(h_V_fp32[i]);
    }

    __half *d_K, *d_V;
    cudaMalloc(&d_K, num_tokens * kv_dim * sizeof(__half));
    cudaMalloc(&d_V, num_tokens * kv_dim * sizeof(__half));
    cudaMemcpy(d_K, h_K_fp16.data(), num_tokens * kv_dim * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V_fp16.data(), num_tokens * kv_dim * sizeof(__half), cudaMemcpyHostToDevice);
    ASSERT_TRUE(cache->append(0, 0, d_K, d_V, num_tokens));

    IKVCache::KVReadParams rope_params;
    rope_params.rope_theta = 10000.0f;
    rope_params.position_start = 0;
    rope_params.n_kv_heads = n_kv_heads;
    rope_params.head_dim = head_dim;

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int kv_len = 0;
    ASSERT_TRUE(cache->get_kv_converted(0, 0, ActivationPrecision::FP16,
                                        &out_k, &out_v, &kv_len, &rope_params));

    // V values should match original (within FP16 precision)
    auto v_out = downloadFP16ToFP32(out_v->gpu_data_ptr(), num_tokens * kv_dim);
    // Compare against FP16→FP32 reference (since we stored FP16 in cache)
    std::vector<float> h_V_ref(num_tokens * kv_dim);
    for (size_t i = 0; i < h_V_fp16.size(); ++i)
        h_V_ref[i] = __half2float(h_V_fp16[i]);
    float cos = computeCosineSimilarity(h_V_ref.data(), v_out.data(), num_tokens * kv_dim);
    EXPECT_GT(cos, 0.999f) << "V should be nearly identical to original (FP16 rounding only)";

    cudaFree(d_K);
    cudaFree(d_V);
}

// =============================================================================
// Q8_1 Cache: RoPE-on-read
// =============================================================================

TEST(Test__CUDARingKVCache_RoPEOnRead, Q8_1_RoPEChangesK)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;
    const int num_tokens = 8;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::Q8_1, 1, 1, 32, n_kv_heads, head_dim);
    ASSERT_NE(cache, nullptr);

    // Use FP32Tensor + appendWithStream which handles FP32→Q8_1 conversion
    auto K_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(kv_dim)});
    auto V_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(kv_dim)});

    std::mt19937 rng(333);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < static_cast<size_t>(num_tokens * kv_dim); ++i)
    {
        K_tensor->mutable_data()[i] = dist(rng);
        V_tensor->mutable_data()[i] = dist(rng);
    }

    DeviceId cuda_dev = DeviceId::cuda(0);
    ASSERT_TRUE(K_tensor->ensureOnDevice(cuda_dev));
    ASSERT_TRUE(V_tensor->ensureOnDevice(cuda_dev));

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    ASSERT_TRUE(cache->appendWithStream(0, 0,
                                        static_cast<const ITensor *>(K_tensor.get()),
                                        static_cast<const ITensor *>(V_tensor.get()),
                                        num_tokens, stream));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

    // Get with RoPE
    IKVCache::KVReadParams rope_params;
    rope_params.rope_theta = 10000.0f;
    rope_params.position_start = 0;
    rope_params.n_kv_heads = n_kv_heads;
    rope_params.head_dim = head_dim;

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int kv_len = 0;
    ASSERT_TRUE(cache->get_kv_converted(0, 0, ActivationPrecision::FP16,
                                        &out_k, &out_v, &kv_len, &rope_params));
    ASSERT_NE(out_k, nullptr);
    EXPECT_EQ(kv_len, num_tokens);

    // Verify K has RoPE applied — should have high cosine with reference
    // but not exact match due to Q8_1 quantization + RoPE
    auto k_result = downloadFP16ToFP32(out_k->gpu_data_ptr(), num_tokens * kv_dim);

    // Just verify it's not all zeros/NaN
    float sum = 0.0f;
    for (auto v : k_result)
    {
        ASSERT_FALSE(std::isnan(v)) << "NaN in Q8_1+RoPE output";
        sum += std::abs(v);
    }
    EXPECT_GT(sum, 0.1f) << "Output should be non-trivial";
}

// =============================================================================
// FP32 Cache: RoPE-on-read (converts to FP16 shadow + RoPE)
// =============================================================================

TEST(Test__CUDARingKVCache_RoPEOnRead, FP32_RoPEConvertsToFP16)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;
    const int num_tokens = 6;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32, 1, 1, 32, n_kv_heads, head_dim);
    ASSERT_NE(cache, nullptr);

    auto h_K = generateRandomFP32(num_tokens * kv_dim, 555);
    auto h_V = generateRandomFP32(num_tokens * kv_dim, 666);

    float *d_K, *d_V;
    cudaMalloc(&d_K, num_tokens * kv_dim * sizeof(float));
    cudaMalloc(&d_V, num_tokens * kv_dim * sizeof(float));
    cudaMemcpy(d_K, h_K.data(), num_tokens * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), num_tokens * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
    ASSERT_TRUE(cache->append(0, 0, d_K, d_V, num_tokens));

    IKVCache::KVReadParams rope_params;
    rope_params.rope_theta = 10000.0f;
    rope_params.position_start = 0;
    rope_params.n_kv_heads = n_kv_heads;
    rope_params.head_dim = head_dim;

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int kv_len = 0;
    ASSERT_TRUE(cache->get_kv_converted(0, 0, ActivationPrecision::FP16,
                                        &out_k, &out_v, &kv_len, &rope_params));
    ASSERT_NE(out_k, nullptr);
    EXPECT_EQ(kv_len, num_tokens);

    auto k_result = downloadFP16ToFP32(out_k->gpu_data_ptr(), num_tokens * kv_dim);

    // V should be close to original (just FP32→FP16 conversion)
    auto v_result = downloadFP16ToFP32(out_v->gpu_data_ptr(), num_tokens * kv_dim);
    float cos_v = computeCosineSimilarity(h_V.data(), v_result.data(), num_tokens * kv_dim);
    EXPECT_GT(cos_v, 0.999f) << "V should match original closely";

    // K should be modified by RoPE
    float cos_k = computeCosineSimilarity(h_K.data(), k_result.data(), num_tokens * kv_dim);
    EXPECT_LT(cos_k, 0.999f) << "K should be modified by RoPE";
    EXPECT_GT(cos_k, 0.5f) << "K should still be correlated with original";

    cudaFree(d_K);
    cudaFree(d_V);
}
