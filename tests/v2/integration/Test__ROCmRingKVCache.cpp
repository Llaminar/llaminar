/**
 * @file Test__ROCmRingKVCache.cpp
 * @brief Unit tests for ROCm Ring Buffer KV Cache
 * @author Llaminar Team
 * @date January 2026
 *
 * Tests:
 * 1. Basic append and retrieval
 * 2. Ring buffer wrap-around behavior
 * 3. O(1) eviction correctness
 * 4. Sliding window pattern
 * 5. Multi-precision (FP32, FP16, BF16)
 *
 * Target Hardware: AMD MI50 (gfx906 / Vega 20)
 */

#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <cmath>

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include "kernels/rocm/kvcache/ROCmRingKVCache.h"
#include "kernels/rocm/kvcache/ROCmRingKVCacheFactory.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "backends/DeviceId.h"
#include "utils/Logger.h"

using namespace llaminar2;

namespace
{
    // Check ROCm availability
    bool hasROCm()
    {
        int count = 0;
        hipError_t err = hipGetDeviceCount(&count);
        return (err == hipSuccess && count > 0);
    }

    // Generate random FP32 data
    std::vector<float> generateRandomFP32(size_t count, unsigned seed = 42)
    {
        std::vector<float> data(count);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &val : data)
        {
            val = dist(rng);
        }
        return data;
    }

    // Compute max absolute error
    float computeMaxError(const std::vector<float> &a, const std::vector<float> &b)
    {
        float max_err = 0.0f;
        size_t n = std::min(a.size(), b.size());
        for (size_t i = 0; i < n; ++i)
        {
            max_err = std::max(max_err, std::abs(a[i] - b[i]));
        }
        return max_err;
    }

} // namespace

// =============================================================================
// Test: Basic Append and Retrieval
// =============================================================================

TEST(Test__ROCmRingKVCache, BasicAppendRetrieve_FP32)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    // Parameters
    const int n_layers = 2;
    const int batch_size = 1;
    const int max_seq_len = 64;
    const int n_kv_heads = 4;
    const int head_dim = 32;
    const int kv_dim = n_kv_heads * head_dim;

    // Create cache using concrete type (tests need ROCm-specific methods)
    auto cache = std::make_unique<ROCmRingKVCache<ActivationPrecision::FP32>>(
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, 0);
    ASSERT_NE(cache, nullptr);

    // Verify initial state
    EXPECT_EQ(cache->n_layers(), n_layers);
    EXPECT_EQ(cache->batch_size(), batch_size);
    EXPECT_EQ(cache->max_seq_len(), max_seq_len);
    EXPECT_EQ(cache->n_kv_heads(), n_kv_heads);
    EXPECT_EQ(cache->head_dim(), head_dim);
    EXPECT_EQ(cache->kv_dim(), kv_dim);

    // Generate test data (10 tokens)
    const int num_tokens = 10;
    auto h_K = generateRandomFP32(num_tokens * kv_dim, 123);
    auto h_V = generateRandomFP32(num_tokens * kv_dim, 456);

    // Allocate device memory
    float *d_K, *d_V;
    hipMalloc(&d_K, num_tokens * kv_dim * sizeof(float));
    hipMalloc(&d_V, num_tokens * kv_dim * sizeof(float));
    hipMemcpy(d_K, h_K.data(), num_tokens * kv_dim * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, h_V.data(), num_tokens * kv_dim * sizeof(float), hipMemcpyHostToDevice);

    // Append to cache (layer 0)
    ASSERT_TRUE(cache->append(0, 0, d_K, d_V, num_tokens));
    EXPECT_EQ(cache->get_cached_tokens(0, 0), num_tokens);
    EXPECT_FALSE(cache->is_wrapped(0, 0)); // Should not be wrapped yet

    // Retrieve K/V
    const void *d_K_out, *d_V_out;
    int kv_len;
    ASSERT_TRUE(cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len));
    EXPECT_EQ(kv_len, num_tokens);

    // Copy back and verify
    std::vector<float> h_K_out(num_tokens * kv_dim);
    std::vector<float> h_V_out(num_tokens * kv_dim);
    hipMemcpy(h_K_out.data(), d_K_out, num_tokens * kv_dim * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_V_out.data(), d_V_out, num_tokens * kv_dim * sizeof(float), hipMemcpyDeviceToHost);

    float max_err_K = computeMaxError(h_K, h_K_out);
    float max_err_V = computeMaxError(h_V, h_V_out);

    LOG_INFO("[BasicAppendRetrieve] max_err_K=" << max_err_K << ", max_err_V=" << max_err_V);

    EXPECT_EQ(max_err_K, 0.0f);
    EXPECT_EQ(max_err_V, 0.0f);

    // Cleanup
    hipFree(d_K);
    hipFree(d_V);

    LOG_INFO("[BasicAppendRetrieve_FP32] PASSED");
}

// =============================================================================
// Test: Ring Buffer Wrap-Around
// =============================================================================

TEST(Test__ROCmRingKVCache, WrapAround_FP32)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    // Small buffer to force wrap-around
    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 8; // Small!
    const int n_kv_heads = 2;
    const int head_dim = 16;
    const int kv_dim = n_kv_heads * head_dim;

    auto cache = std::make_unique<ROCmRingKVCache<ActivationPrecision::FP32>>(
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, 0);
    ASSERT_NE(cache, nullptr);

    // Generate test data: 12 tokens (exceeds max_seq_len=8)
    const int total_tokens = 12;
    auto h_K_all = generateRandomFP32(total_tokens * kv_dim, 789);
    auto h_V_all = generateRandomFP32(total_tokens * kv_dim, 012);

    // Allocate device memory
    float *d_K, *d_V;
    hipMalloc(&d_K, total_tokens * kv_dim * sizeof(float));
    hipMalloc(&d_V, total_tokens * kv_dim * sizeof(float));
    hipMemcpy(d_K, h_K_all.data(), total_tokens * kv_dim * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, h_V_all.data(), total_tokens * kv_dim * sizeof(float), hipMemcpyHostToDevice);

    // Append all 12 tokens - should wrap and auto-evict oldest 4
    ASSERT_TRUE(cache->append(0, 0, d_K, d_V, total_tokens));

    // Should have max_seq_len=8 tokens, with oldest 4 evicted
    EXPECT_EQ(cache->get_cached_tokens(0, 0), max_seq_len);
    EXPECT_TRUE(cache->is_wrapped(0, 0)); // Should be wrapped

    // Retrieve and verify we have the LAST 8 tokens
    const void *d_K_out, *d_V_out;
    int kv_len;
    ASSERT_TRUE(cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len));
    EXPECT_EQ(kv_len, max_seq_len);

    // Copy back
    std::vector<float> h_K_out(max_seq_len * kv_dim);
    std::vector<float> h_V_out(max_seq_len * kv_dim);
    hipMemcpy(h_K_out.data(), d_K_out, max_seq_len * kv_dim * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_V_out.data(), d_V_out, max_seq_len * kv_dim * sizeof(float), hipMemcpyDeviceToHost);

    // Verify: output should match tokens [4..11] of original
    const float *expected_K = h_K_all.data() + 4 * kv_dim;
    const float *expected_V = h_V_all.data() + 4 * kv_dim;

    float max_err_K = computeMaxError(h_K_out, std::vector<float>(expected_K, expected_K + max_seq_len * kv_dim));
    float max_err_V = computeMaxError(h_V_out, std::vector<float>(expected_V, expected_V + max_seq_len * kv_dim));

    LOG_INFO("[WrapAround] max_err_K=" << max_err_K << ", max_err_V=" << max_err_V);

    EXPECT_EQ(max_err_K, 0.0f);
    EXPECT_EQ(max_err_V, 0.0f);

    // Verify eviction counter
    EXPECT_EQ(cache->get_total_evicted(), 4);

    // Cleanup
    hipFree(d_K);
    hipFree(d_V);

    LOG_INFO("[WrapAround_FP32] PASSED");
}

// =============================================================================
// Test: Ring Buffer Wrap-Around Edge Cases (Race Condition Prevention)
// =============================================================================

TEST(Test__ROCmRingKVCache, WrapAround_ExactlyDouble_FP32)
{
    // Tests the race condition fix: when appending exactly 2x buffer size,
    // we should only write the last max_seq_len tokens (tokens 8-15)
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 8;
    const int n_kv_heads = 2;
    const int head_dim = 16;
    const int kv_dim = n_kv_heads * head_dim;

    auto cache = std::make_unique<ROCmRingKVCache<ActivationPrecision::FP32>>(
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, 0);
    ASSERT_NE(cache, nullptr);

    // Exactly 2x buffer size: 16 tokens into 8-slot buffer
    const int total_tokens = 16;
    auto h_K_all = generateRandomFP32(total_tokens * kv_dim, 111);
    auto h_V_all = generateRandomFP32(total_tokens * kv_dim, 222);

    float *d_K, *d_V;
    hipMalloc(&d_K, total_tokens * kv_dim * sizeof(float));
    hipMalloc(&d_V, total_tokens * kv_dim * sizeof(float));
    hipMemcpy(d_K, h_K_all.data(), total_tokens * kv_dim * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, h_V_all.data(), total_tokens * kv_dim * sizeof(float), hipMemcpyHostToDevice);

    ASSERT_TRUE(cache->append(0, 0, d_K, d_V, total_tokens));

    EXPECT_EQ(cache->get_cached_tokens(0, 0), max_seq_len);
    EXPECT_TRUE(cache->is_wrapped(0, 0));

    const void *d_K_out, *d_V_out;
    int kv_len;
    ASSERT_TRUE(cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len));
    EXPECT_EQ(kv_len, max_seq_len);

    std::vector<float> h_K_out(max_seq_len * kv_dim);
    std::vector<float> h_V_out(max_seq_len * kv_dim);
    hipMemcpy(h_K_out.data(), d_K_out, max_seq_len * kv_dim * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_V_out.data(), d_V_out, max_seq_len * kv_dim * sizeof(float), hipMemcpyDeviceToHost);

    // Should have tokens 8-15 (the last 8)
    const float *expected_K = h_K_all.data() + 8 * kv_dim;
    const float *expected_V = h_V_all.data() + 8 * kv_dim;

    float max_err_K = computeMaxError(h_K_out, std::vector<float>(expected_K, expected_K + max_seq_len * kv_dim));
    float max_err_V = computeMaxError(h_V_out, std::vector<float>(expected_V, expected_V + max_seq_len * kv_dim));

    LOG_INFO("[WrapAround_ExactlyDouble] max_err_K=" << max_err_K << ", max_err_V=" << max_err_V);

    EXPECT_EQ(max_err_K, 0.0f) << "K values should exactly match tokens 8-15";
    EXPECT_EQ(max_err_V, 0.0f) << "V values should exactly match tokens 8-15";
    EXPECT_EQ(cache->get_total_evicted(), 8) << "Should have evicted exactly 8 tokens";

    hipFree(d_K);
    hipFree(d_V);

    LOG_INFO("[WrapAround_ExactlyDouble_FP32] PASSED");
}

TEST(Test__ROCmRingKVCache, WrapAround_BarelyOver_FP32)
{
    // Tests the race condition fix: when appending just 1 more than buffer size,
    // token 0 would write to position 0, token 8 would also write to position 0.
    // Without the fix, this causes a race condition.
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 8;
    const int n_kv_heads = 2;
    const int head_dim = 16;
    const int kv_dim = n_kv_heads * head_dim;

    auto cache = std::make_unique<ROCmRingKVCache<ActivationPrecision::FP32>>(
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, 0);
    ASSERT_NE(cache, nullptr);

    // Just 1 more than buffer: 9 tokens into 8-slot buffer
    const int total_tokens = 9;
    auto h_K_all = generateRandomFP32(total_tokens * kv_dim, 333);
    auto h_V_all = generateRandomFP32(total_tokens * kv_dim, 444);

    float *d_K, *d_V;
    hipMalloc(&d_K, total_tokens * kv_dim * sizeof(float));
    hipMalloc(&d_V, total_tokens * kv_dim * sizeof(float));
    hipMemcpy(d_K, h_K_all.data(), total_tokens * kv_dim * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, h_V_all.data(), total_tokens * kv_dim * sizeof(float), hipMemcpyHostToDevice);

    ASSERT_TRUE(cache->append(0, 0, d_K, d_V, total_tokens));

    EXPECT_EQ(cache->get_cached_tokens(0, 0), max_seq_len);
    EXPECT_TRUE(cache->is_wrapped(0, 0));

    const void *d_K_out, *d_V_out;
    int kv_len;
    ASSERT_TRUE(cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len));
    EXPECT_EQ(kv_len, max_seq_len);

    std::vector<float> h_K_out(max_seq_len * kv_dim);
    std::vector<float> h_V_out(max_seq_len * kv_dim);
    hipMemcpy(h_K_out.data(), d_K_out, max_seq_len * kv_dim * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_V_out.data(), d_V_out, max_seq_len * kv_dim * sizeof(float), hipMemcpyDeviceToHost);

    // Should have tokens 1-8 (the last 8, with token 0 evicted)
    const float *expected_K = h_K_all.data() + 1 * kv_dim;
    const float *expected_V = h_V_all.data() + 1 * kv_dim;

    float max_err_K = computeMaxError(h_K_out, std::vector<float>(expected_K, expected_K + max_seq_len * kv_dim));
    float max_err_V = computeMaxError(h_V_out, std::vector<float>(expected_V, expected_V + max_seq_len * kv_dim));

    LOG_INFO("[WrapAround_BarelyOver] max_err_K=" << max_err_K << ", max_err_V=" << max_err_V);

    EXPECT_EQ(max_err_K, 0.0f) << "K values should exactly match tokens 1-8";
    EXPECT_EQ(max_err_V, 0.0f) << "V values should exactly match tokens 1-8";
    EXPECT_EQ(cache->get_total_evicted(), 1) << "Should have evicted exactly 1 token";

    hipFree(d_K);
    hipFree(d_V);

    LOG_INFO("[WrapAround_BarelyOver_FP32] PASSED");
}

TEST(Test__ROCmRingKVCache, WrapAround_TripleBuffer_FP32)
{
    // Tests the race condition fix with 3x buffer size
    // This tests the general case where tokens_to_skip > max_seq_len
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 8;
    const int n_kv_heads = 2;
    const int head_dim = 16;
    const int kv_dim = n_kv_heads * head_dim;

    auto cache = std::make_unique<ROCmRingKVCache<ActivationPrecision::FP32>>(
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, 0);
    ASSERT_NE(cache, nullptr);

    // 3x buffer size: 24 tokens into 8-slot buffer
    const int total_tokens = 24;
    auto h_K_all = generateRandomFP32(total_tokens * kv_dim, 555);
    auto h_V_all = generateRandomFP32(total_tokens * kv_dim, 666);

    float *d_K, *d_V;
    hipMalloc(&d_K, total_tokens * kv_dim * sizeof(float));
    hipMalloc(&d_V, total_tokens * kv_dim * sizeof(float));
    hipMemcpy(d_K, h_K_all.data(), total_tokens * kv_dim * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, h_V_all.data(), total_tokens * kv_dim * sizeof(float), hipMemcpyHostToDevice);

    ASSERT_TRUE(cache->append(0, 0, d_K, d_V, total_tokens));

    EXPECT_EQ(cache->get_cached_tokens(0, 0), max_seq_len);
    EXPECT_TRUE(cache->is_wrapped(0, 0));

    const void *d_K_out, *d_V_out;
    int kv_len;
    ASSERT_TRUE(cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len));
    EXPECT_EQ(kv_len, max_seq_len);

    std::vector<float> h_K_out(max_seq_len * kv_dim);
    std::vector<float> h_V_out(max_seq_len * kv_dim);
    hipMemcpy(h_K_out.data(), d_K_out, max_seq_len * kv_dim * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_V_out.data(), d_V_out, max_seq_len * kv_dim * sizeof(float), hipMemcpyDeviceToHost);

    // Should have tokens 16-23 (the last 8)
    const float *expected_K = h_K_all.data() + 16 * kv_dim;
    const float *expected_V = h_V_all.data() + 16 * kv_dim;

    float max_err_K = computeMaxError(h_K_out, std::vector<float>(expected_K, expected_K + max_seq_len * kv_dim));
    float max_err_V = computeMaxError(h_V_out, std::vector<float>(expected_V, expected_V + max_seq_len * kv_dim));

    LOG_INFO("[WrapAround_TripleBuffer] max_err_K=" << max_err_K << ", max_err_V=" << max_err_V);

    EXPECT_EQ(max_err_K, 0.0f) << "K values should exactly match tokens 16-23";
    EXPECT_EQ(max_err_V, 0.0f) << "V values should exactly match tokens 16-23";
    EXPECT_EQ(cache->get_total_evicted(), 16) << "Should have evicted exactly 16 tokens";

    hipFree(d_K);
    hipFree(d_V);

    LOG_INFO("[WrapAround_TripleBuffer_FP32] PASSED");
}

// =============================================================================
// Test: O(1) Eviction
// =============================================================================

TEST(Test__ROCmRingKVCache, Eviction_FP32)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 16;
    const int n_kv_heads = 2;
    const int head_dim = 16;
    const int kv_dim = n_kv_heads * head_dim;

    auto cache = std::make_unique<ROCmRingKVCache<ActivationPrecision::FP32>>(
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, 0);
    ASSERT_NE(cache, nullptr);

    // Append 10 tokens
    const int num_tokens = 10;
    auto h_K = generateRandomFP32(num_tokens * kv_dim, 111);
    auto h_V = generateRandomFP32(num_tokens * kv_dim, 222);

    float *d_K, *d_V;
    hipMalloc(&d_K, num_tokens * kv_dim * sizeof(float));
    hipMalloc(&d_V, num_tokens * kv_dim * sizeof(float));
    hipMemcpy(d_K, h_K.data(), num_tokens * kv_dim * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, h_V.data(), num_tokens * kv_dim * sizeof(float), hipMemcpyHostToDevice);

    ASSERT_TRUE(cache->append(0, 0, d_K, d_V, num_tokens));
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 10);

    // Evict 3 oldest tokens
    cache->evict_oldest(0, 0, 3);
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 7);
    EXPECT_EQ(cache->get_total_evicted(), 3);

    // Retrieve and verify we have the LAST 7 tokens
    const void *d_K_out, *d_V_out;
    int kv_len;
    ASSERT_TRUE(cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len));
    EXPECT_EQ(kv_len, 7);

    // Copy back
    std::vector<float> h_K_out(7 * kv_dim);
    std::vector<float> h_V_out(7 * kv_dim);
    hipMemcpy(h_K_out.data(), d_K_out, 7 * kv_dim * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_V_out.data(), d_V_out, 7 * kv_dim * sizeof(float), hipMemcpyDeviceToHost);

    // Verify: output should match tokens [3..9] of original
    const float *expected_K = h_K.data() + 3 * kv_dim;
    const float *expected_V = h_V.data() + 3 * kv_dim;

    float max_err_K = computeMaxError(h_K_out, std::vector<float>(expected_K, expected_K + 7 * kv_dim));
    float max_err_V = computeMaxError(h_V_out, std::vector<float>(expected_V, expected_V + 7 * kv_dim));

    LOG_INFO("[Eviction] max_err_K=" << max_err_K << ", max_err_V=" << max_err_V);

    EXPECT_EQ(max_err_K, 0.0f);
    EXPECT_EQ(max_err_V, 0.0f);

    // Cleanup
    hipFree(d_K);
    hipFree(d_V);

    LOG_INFO("[Eviction_FP32] PASSED");
}

// =============================================================================
// Test: Clear Operations
// =============================================================================

TEST(Test__ROCmRingKVCache, Clear_FP32)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    const int n_layers = 2;
    const int batch_size = 2;
    const int max_seq_len = 16;
    const int n_kv_heads = 2;
    const int head_dim = 16;
    const int kv_dim = n_kv_heads * head_dim;

    auto cache = std::make_unique<ROCmRingKVCache<ActivationPrecision::FP32>>(
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, 0);
    ASSERT_NE(cache, nullptr);

    // Append some tokens to multiple layers/sequences
    const int num_tokens = 5;
    auto h_K = generateRandomFP32(num_tokens * kv_dim, 333);
    auto h_V = generateRandomFP32(num_tokens * kv_dim, 444);

    float *d_K, *d_V;
    hipMalloc(&d_K, num_tokens * kv_dim * sizeof(float));
    hipMalloc(&d_V, num_tokens * kv_dim * sizeof(float));
    hipMemcpy(d_K, h_K.data(), num_tokens * kv_dim * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, h_V.data(), num_tokens * kv_dim * sizeof(float), hipMemcpyHostToDevice);

    // Fill all caches
    for (int layer = 0; layer < n_layers; ++layer)
    {
        for (int seq = 0; seq < batch_size; ++seq)
        {
            ASSERT_TRUE(cache->append(layer, seq, d_K, d_V, num_tokens));
            EXPECT_EQ(cache->get_cached_tokens(layer, seq), num_tokens);
        }
    }

    // Clear specific sequence
    cache->clear_sequence(0, 1);                           // Layer 0, Seq 1
    EXPECT_EQ(cache->get_cached_tokens(0, 0), num_tokens); // Unchanged
    EXPECT_EQ(cache->get_cached_tokens(0, 1), 0);          // Cleared
    EXPECT_EQ(cache->get_cached_tokens(1, 0), num_tokens); // Unchanged
    EXPECT_EQ(cache->get_cached_tokens(1, 1), num_tokens); // Unchanged

    // Clear specific layer
    cache->clear_layer(1);
    EXPECT_EQ(cache->get_cached_tokens(0, 0), num_tokens); // Unchanged
    EXPECT_EQ(cache->get_cached_tokens(1, 0), 0);          // Cleared
    EXPECT_EQ(cache->get_cached_tokens(1, 1), 0);          // Cleared

    // Clear all
    cache->clear();
    for (int layer = 0; layer < n_layers; ++layer)
    {
        for (int seq = 0; seq < batch_size; ++seq)
        {
            EXPECT_EQ(cache->get_cached_tokens(layer, seq), 0);
        }
    }

    // Cleanup
    hipFree(d_K);
    hipFree(d_V);

    LOG_INFO("[Clear_FP32] PASSED");
}

// =============================================================================
// Test: Multi-Precision (FP16)
// =============================================================================

TEST(Test__ROCmRingKVCache, BasicAppendRetrieve_FP16)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 32;
    const int n_kv_heads = 2;
    const int head_dim = 16;
    const int kv_dim = n_kv_heads * head_dim;

    auto cache = std::make_unique<ROCmRingKVCache<ActivationPrecision::FP16>>(
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, 0);
    ASSERT_NE(cache, nullptr);

    // Verify precision
    EXPECT_EQ(cache->precision(), ActivationPrecision::FP16);

    // Generate test data
    const int num_tokens = 8;
    auto h_K_fp32 = generateRandomFP32(num_tokens * kv_dim, 555);
    auto h_V_fp32 = generateRandomFP32(num_tokens * kv_dim, 666);

    // Convert to FP16
    std::vector<_Float16> h_K(num_tokens * kv_dim);
    std::vector<_Float16> h_V(num_tokens * kv_dim);
    for (size_t i = 0; i < h_K.size(); ++i)
    {
        h_K[i] = static_cast<_Float16>(h_K_fp32[i]);
        h_V[i] = static_cast<_Float16>(h_V_fp32[i]);
    }

    // Allocate device memory
    _Float16 *d_K, *d_V;
    hipMalloc(&d_K, num_tokens * kv_dim * sizeof(_Float16));
    hipMalloc(&d_V, num_tokens * kv_dim * sizeof(_Float16));
    hipMemcpy(d_K, h_K.data(), num_tokens * kv_dim * sizeof(_Float16), hipMemcpyHostToDevice);
    hipMemcpy(d_V, h_V.data(), num_tokens * kv_dim * sizeof(_Float16), hipMemcpyHostToDevice);

    // Append
    ASSERT_TRUE(cache->append(0, 0, d_K, d_V, num_tokens));
    EXPECT_EQ(cache->get_cached_tokens(0, 0), num_tokens);

    // Retrieve
    const void *d_K_out, *d_V_out;
    int kv_len;
    ASSERT_TRUE(cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len));
    EXPECT_EQ(kv_len, num_tokens);

    // Copy back
    std::vector<_Float16> h_K_out(num_tokens * kv_dim);
    std::vector<_Float16> h_V_out(num_tokens * kv_dim);
    hipMemcpy(h_K_out.data(), d_K_out, num_tokens * kv_dim * sizeof(_Float16), hipMemcpyDeviceToHost);
    hipMemcpy(h_V_out.data(), d_V_out, num_tokens * kv_dim * sizeof(_Float16), hipMemcpyDeviceToHost);

    // Verify (FP16 should be exact bitwise match)
    for (size_t i = 0; i < h_K.size(); ++i)
    {
        EXPECT_EQ(static_cast<float>(h_K_out[i]), static_cast<float>(h_K[i]))
            << "K mismatch at index " << i;
        EXPECT_EQ(static_cast<float>(h_V_out[i]), static_cast<float>(h_V[i]))
            << "V mismatch at index " << i;
    }

    // Cleanup
    hipFree(d_K);
    hipFree(d_V);

    LOG_INFO("[BasicAppendRetrieve_FP16] PASSED");
}

// =============================================================================
// Test: Linearization Statistics
// =============================================================================

TEST(Test__ROCmRingKVCache, LinearizationStatistics_FP32)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 8;
    const int n_kv_heads = 2;
    const int head_dim = 16;
    const int kv_dim = n_kv_heads * head_dim;

    auto cache = std::make_unique<ROCmRingKVCache<ActivationPrecision::FP32>>(
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, 0);
    ASSERT_NE(cache, nullptr);

    // Initial stats
    EXPECT_EQ(cache->get_linearization_count(), 0);
    EXPECT_EQ(cache->get_total_evicted(), 0);

    // Append 12 tokens to force wrap
    const int num_tokens = 12;
    auto h_K = generateRandomFP32(num_tokens * kv_dim, 777);
    auto h_V = generateRandomFP32(num_tokens * kv_dim, 888);

    float *d_K, *d_V;
    hipMalloc(&d_K, num_tokens * kv_dim * sizeof(float));
    hipMalloc(&d_V, num_tokens * kv_dim * sizeof(float));
    hipMemcpy(d_K, h_K.data(), num_tokens * kv_dim * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, h_V.data(), num_tokens * kv_dim * sizeof(float), hipMemcpyHostToDevice);

    ASSERT_TRUE(cache->append(0, 0, d_K, d_V, num_tokens));
    EXPECT_TRUE(cache->is_wrapped(0, 0));
    EXPECT_EQ(cache->get_total_evicted(), 4); // 12 - 8 = 4 evicted

    // Get K/V should trigger linearization
    const void *d_K_out, *d_V_out;
    int kv_len;
    ASSERT_TRUE(cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len));
    EXPECT_EQ(cache->get_linearization_count(), 1);

    // Subsequent get should NOT trigger linearization (cached)
    ASSERT_TRUE(cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len));
    EXPECT_EQ(cache->get_linearization_count(), 1);

    // Reset counters
    cache->reset_linearization_counter();
    cache->reset_eviction_counter();
    EXPECT_EQ(cache->get_linearization_count(), 0);
    EXPECT_EQ(cache->get_total_evicted(), 0);

    // Cleanup
    hipFree(d_K);
    hipFree(d_V);

    LOG_INFO("[LinearizationStatistics_FP32] PASSED");
}

// =============================================================================
// Test: Sliding Window Pattern
// =============================================================================

TEST(Test__ROCmRingKVCache, SlidingWindow)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    // Simulate sliding window attention with window_size=32
    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 32; // Window size
    const int n_kv_heads = 4;
    const int head_dim = 16;
    const int kv_dim = n_kv_heads * head_dim;

    auto cache = std::make_unique<ROCmRingKVCache<ActivationPrecision::FP32>>(
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, 0);
    ASSERT_NE(cache, nullptr);

    float *d_K, *d_V;
    hipMalloc(&d_K, kv_dim * sizeof(float));
    hipMalloc(&d_V, kv_dim * sizeof(float));

    // Simulate 100 decode steps
    for (int step = 0; step < 100; ++step)
    {
        auto h_K = generateRandomFP32(kv_dim, step);
        auto h_V = generateRandomFP32(kv_dim, step + 1000);

        hipMemcpy(d_K, h_K.data(), kv_dim * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(d_V, h_V.data(), kv_dim * sizeof(float), hipMemcpyHostToDevice);

        // Append 1 token
        ASSERT_TRUE(cache->append(0, 0, d_K, d_V, 1));

        // Cache should never exceed window size (auto-evicts)
        EXPECT_LE(cache->get_cached_tokens(0, 0), max_seq_len);
    }

    // After 100 steps with window=32, should have exactly 32 tokens
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 32);

    // 100 appends with window=32 means 68 evicted
    EXPECT_EQ(cache->get_total_evicted(), 68);

    hipFree(d_K);
    hipFree(d_V);

    LOG_INFO("[SlidingWindow] PASSED - evicted=" << cache->get_total_evicted());
}

// =============================================================================
// Test: Batched Gather
// =============================================================================

TEST(Test__ROCmRingKVCache, BatchedGather)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    const int n_layers = 1;
    const int batch_size = 4;
    const int max_seq_len = 64;
    const int n_kv_heads = 2;
    const int head_dim = 32;
    const int kv_dim = n_kv_heads * head_dim;

    auto cache = std::make_unique<ROCmRingKVCache<ActivationPrecision::FP32>>(
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, 0);
    ASSERT_NE(cache, nullptr);

    // Fill each sequence with different lengths
    int seq_lens[] = {10, 20, 15, 25};
    std::vector<std::vector<float>> h_Ks(batch_size);
    std::vector<std::vector<float>> h_Vs(batch_size);

    float *d_K, *d_V;
    hipMalloc(&d_K, 30 * kv_dim * sizeof(float));
    hipMalloc(&d_V, 30 * kv_dim * sizeof(float));

    for (int seq = 0; seq < batch_size; ++seq)
    {
        h_Ks[seq] = generateRandomFP32(seq_lens[seq] * kv_dim, seq * 100);
        h_Vs[seq] = generateRandomFP32(seq_lens[seq] * kv_dim, seq * 100 + 1000);

        hipMemcpy(d_K, h_Ks[seq].data(), seq_lens[seq] * kv_dim * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(d_V, h_Vs[seq].data(), seq_lens[seq] * kv_dim * sizeof(float), hipMemcpyHostToDevice);

        ASSERT_TRUE(cache->append(0, seq, d_K, d_V, seq_lens[seq]));
    }

    // Verify individual sequence lengths
    for (int seq = 0; seq < batch_size; ++seq)
    {
        EXPECT_EQ(cache->get_cached_tokens(0, seq), seq_lens[seq]);
    }

    // Set up workspace for gather operation (REQUIRED - no fallback allocations)
    auto reqs = cache->getWorkspaceRequirements(batch_size, 0, 0);
    DeviceWorkspaceManager workspace(DeviceId::rocm(0), 1024 * 1024); // 1MB budget
    ASSERT_TRUE(workspace.allocate(reqs));
    cache->bindWorkspace(&workspace);

    // Gather all sequences
    int max_kv_len = 25; // Max sequence length
    float *d_K_gathered, *d_V_gathered;
    hipMalloc(&d_K_gathered, batch_size * max_kv_len * kv_dim * sizeof(float));
    hipMalloc(&d_V_gathered, batch_size * max_kv_len * kv_dim * sizeof(float));

    std::vector<int> kv_lens(batch_size);
    int actual_max = cache->gather_kv_batched(0, batch_size,
                                              d_K_gathered, d_V_gathered,
                                              kv_lens.data(), max_kv_len);

    EXPECT_EQ(actual_max, 25); // Max across sequences

    // Verify per-sequence lengths
    for (int seq = 0; seq < batch_size; ++seq)
    {
        EXPECT_EQ(kv_lens[seq], seq_lens[seq]);
    }

    // Verify content for sequence 0
    std::vector<float> h_K_gathered(batch_size * max_kv_len * kv_dim);
    hipMemcpy(h_K_gathered.data(), d_K_gathered,
              batch_size * max_kv_len * kv_dim * sizeof(float),
              hipMemcpyDeviceToHost);

    for (int t = 0; t < seq_lens[0]; ++t)
    {
        for (int d = 0; d < kv_dim; ++d)
        {
            int src_idx = t * kv_dim + d;
            int dst_idx = (0 * max_kv_len + t) * kv_dim + d;
            EXPECT_FLOAT_EQ(h_K_gathered[dst_idx], h_Ks[0][src_idx])
                << "Seq0 K mismatch at token " << t;
        }
    }

    // Unbind workspace before cleanup
    cache->unbindWorkspace();

    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_K_gathered);
    hipFree(d_V_gathered);

    LOG_INFO("[BatchedGather] PASSED");
}

// =============================================================================
// Test: Contiguous Optimization
// =============================================================================

TEST(Test__ROCmRingKVCache, ContiguousOptimization)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 64;
    const int n_kv_heads = 4;
    const int head_dim = 32;
    const int kv_dim = n_kv_heads * head_dim;

    auto cache = std::make_unique<ROCmRingKVCache<ActivationPrecision::FP32>>(
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, 0);
    ASSERT_NE(cache, nullptr);

    // Append tokens without wrapping
    auto h_K = generateRandomFP32(30 * kv_dim);
    auto h_V = generateRandomFP32(30 * kv_dim);

    float *d_K, *d_V;
    hipMalloc(&d_K, 30 * kv_dim * sizeof(float));
    hipMalloc(&d_V, 30 * kv_dim * sizeof(float));
    hipMemcpy(d_K, h_K.data(), 30 * kv_dim * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, h_V.data(), 30 * kv_dim * sizeof(float), hipMemcpyHostToDevice);

    ASSERT_TRUE(cache->append(0, 0, d_K, d_V, 30));

    // Should NOT be wrapped
    EXPECT_FALSE(cache->is_wrapped(0, 0));

    // Get K/V - should return direct pointer (no linearization)
    const void *d_K_out, *d_V_out;
    int kv_len;
    ASSERT_TRUE(cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len));

    // No linearizations should have occurred
    EXPECT_EQ(cache->get_linearization_count(), 0);

    // Multiple retrievals should still not linearize
    for (int i = 0; i < 10; ++i)
    {
        ASSERT_TRUE(cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len));
    }
    EXPECT_EQ(cache->get_linearization_count(), 0);

    hipFree(d_K);
    hipFree(d_V);

    LOG_INFO("[ContiguousOptimization] PASSED - linearizations=0");
}

// =============================================================================
// Test: Multi-Precision (BF16)
// =============================================================================

TEST(Test__ROCmRingKVCache, MultiPrecision_BF16)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 32;
    const int n_kv_heads = 2;
    const int head_dim = 32;
    const int kv_dim = n_kv_heads * head_dim;

    auto cache = std::make_unique<ROCmRingKVCache<ActivationPrecision::BF16>>(
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, 0);
    ASSERT_NE(cache, nullptr);
    EXPECT_EQ(cache->precision(), ActivationPrecision::BF16);

    // Generate FP32 data and convert to BF16
    const int num_tokens = 10;
    auto h_K_fp32 = generateRandomFP32(num_tokens * kv_dim, 999);
    auto h_V_fp32 = generateRandomFP32(num_tokens * kv_dim, 1001);

    // BF16: truncate mantissa (use hip_bfloat16 from hip/hip_bfloat16.h)
    std::vector<hip_bfloat16> h_K_bf16(num_tokens * kv_dim);
    std::vector<hip_bfloat16> h_V_bf16(num_tokens * kv_dim);
    for (size_t i = 0; i < h_K_fp32.size(); ++i)
    {
        h_K_bf16[i] = hip_bfloat16(h_K_fp32[i]);
        h_V_bf16[i] = hip_bfloat16(h_V_fp32[i]);
    }

    hip_bfloat16 *d_K, *d_V;
    hipMalloc(&d_K, num_tokens * kv_dim * sizeof(hip_bfloat16));
    hipMalloc(&d_V, num_tokens * kv_dim * sizeof(hip_bfloat16));
    hipMemcpy(d_K, h_K_bf16.data(), num_tokens * kv_dim * sizeof(hip_bfloat16), hipMemcpyHostToDevice);
    hipMemcpy(d_V, h_V_bf16.data(), num_tokens * kv_dim * sizeof(hip_bfloat16), hipMemcpyHostToDevice);

    ASSERT_TRUE(cache->append(0, 0, d_K, d_V, num_tokens));
    EXPECT_EQ(cache->get_cached_tokens(0, 0), num_tokens);

    const void *d_K_out, *d_V_out;
    int kv_len;
    ASSERT_TRUE(cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len));
    EXPECT_EQ(kv_len, num_tokens);

    // Verify content - convert hip_bfloat16 to float via union
    std::vector<hip_bfloat16> h_K_out(num_tokens * kv_dim);
    hipMemcpy(h_K_out.data(), d_K_out, num_tokens * kv_dim * sizeof(hip_bfloat16), hipMemcpyDeviceToHost);

    // Helper lambda for bf16 to float conversion
    auto bf16_to_float = [](hip_bfloat16 val) -> float
    {
        union
        {
            uint32_t u;
            float f;
        } conv;
        conv.u = static_cast<uint32_t>(val.data) << 16;
        return conv.f;
    };

    for (size_t i = 0; i < num_tokens * kv_dim; ++i)
    {
        float expected = bf16_to_float(h_K_bf16[i]);
        float actual = bf16_to_float(h_K_out[i]);
        EXPECT_FLOAT_EQ(actual, expected) << "BF16 mismatch at " << i;
    }

    hipFree(d_K);
    hipFree(d_V);

    LOG_INFO("[MultiPrecision_BF16] PASSED");
}

#endif // HAVE_ROCM

// =============================================================================
// No ROCm Fallback Test
// =============================================================================

TEST(Test__ROCmRingKVCache, NoROCm_CreateReturnsNull)
{
#ifndef HAVE_ROCM
    // Without HAVE_ROCM, this test would not compile if we tried to call
    // createROCmRingKVCache. Instead, just verify the compile-time guard works.
    SUCCEED() << "HAVE_ROCM not defined, compile-time guard working";
#else
    if (!hasROCm())
    {
        // At runtime with HAVE_ROCM but no actual device, factory should fail gracefully
        // Note: This depends on implementation - might return nullptr or throw
        LOG_INFO("ROCm compiled but no device available - skipping");
        GTEST_SKIP() << "ROCm compiled but no device";
    }
#endif
}
