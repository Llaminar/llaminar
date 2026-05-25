/**
 * @file Test__CUDARingKVCacheParity.cpp
 * @brief Parity tests for CUDA Ring Buffer KV Cache
 * @author David Sanftenberg
 * @date January 2026
 *
 * Tests:
 * 1. Basic append and retrieval
 * 2. Ring buffer wrap-around behavior
 * 3. O(1) eviction correctness
 * 4. Sliding window pattern
 * 5. Batched gather
 * 6. Multi-precision (FP32, FP16, BF16)
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <cmath>
#include "kernels/cuda/kvcache/CUDARingKVCache.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "execution/local_execution/device/WorkspaceDescriptor.h"
#include "tensors/Tensors.h"
#include "utils/Logger.h"

using namespace llaminar2;

namespace
{
    // Check CUDA availability
    bool hasCUDA()
    {
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        return (err == cudaSuccess && count > 0);
    }

    /**
     * @brief Owns a CUDA stream for integration tests that exercise stream-aware KV APIs.
     */
    class ScopedCudaStream
    {
    public:
        ScopedCudaStream()
        {
            EXPECT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
        }

        ~ScopedCudaStream()
        {
            if (stream_)
                cudaStreamDestroy(stream_);
        }

        void *opaque() const { return static_cast<void *>(stream_); }

        void synchronize() const
        {
            ASSERT_NE(stream_, nullptr);
            ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        }

    private:
        cudaStream_t stream_ = nullptr;
    };

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

TEST(Test__CUDARingKVCache, BasicAppendRetrieve_FP32)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    // Parameters
    const int n_layers = 2;
    const int batch_size = 1;
    const int max_seq_len = 64;
    const int n_kv_heads = 4;
    const int head_dim = 32;
    const int kv_dim = n_kv_heads * head_dim;

    // Create cache
    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ASSERT_NE(cache, nullptr);

    // Generate test data (10 tokens)
    const int num_tokens = 10;
    auto h_K = generateRandomFP32(num_tokens * kv_dim, 123);
    auto h_V = generateRandomFP32(num_tokens * kv_dim, 456);

    // Allocate device memory
    float *d_K, *d_V;
    cudaMalloc(&d_K, num_tokens * kv_dim * sizeof(float));
    cudaMalloc(&d_V, num_tokens * kv_dim * sizeof(float));
    cudaMemcpy(d_K, h_K.data(), num_tokens * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), num_tokens * kv_dim * sizeof(float), cudaMemcpyHostToDevice);

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
    cudaMemcpy(h_K_out.data(), d_K_out, num_tokens * kv_dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_V_out.data(), d_V_out, num_tokens * kv_dim * sizeof(float), cudaMemcpyDeviceToHost);

    float max_err_K = computeMaxError(h_K, h_K_out);
    float max_err_V = computeMaxError(h_V, h_V_out);

    LOG_INFO("[BasicAppendRetrieve] max_err_K=" << max_err_K << ", max_err_V=" << max_err_V);

    EXPECT_EQ(max_err_K, 0.0f);
    EXPECT_EQ(max_err_V, 0.0f);

    // Cleanup
    cudaFree(d_K);
    cudaFree(d_V);

    LOG_INFO("[BasicAppendRetrieve_FP32] PASSED");
}

// =============================================================================
// Test: Ring Buffer Wrap-Around
// =============================================================================

TEST(Test__CUDARingKVCache, WrapAround_FP32)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    // Small buffer to force wrap-around
    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 8; // Small!
    const int n_kv_heads = 2;
    const int head_dim = 16;
    const int kv_dim = n_kv_heads * head_dim;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ASSERT_NE(cache, nullptr);

    // Phase 1: Fill buffer with 6 tokens [T0..T5]
    const int phase1_tokens = 6;
    auto h_K1 = generateRandomFP32(phase1_tokens * kv_dim, 100);
    auto h_V1 = generateRandomFP32(phase1_tokens * kv_dim, 200);

    float *d_K, *d_V;
    cudaMalloc(&d_K, max_seq_len * kv_dim * sizeof(float));
    cudaMalloc(&d_V, max_seq_len * kv_dim * sizeof(float));

    cudaMemcpy(d_K, h_K1.data(), phase1_tokens * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V1.data(), phase1_tokens * kv_dim * sizeof(float), cudaMemcpyHostToDevice);

    ASSERT_TRUE(cache->append(0, 0, d_K, d_V, phase1_tokens));
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 6);
    EXPECT_EQ(cache->get_head_position(0, 0), 6);
    EXPECT_FALSE(cache->is_wrapped(0, 0));

    // Phase 2: Append 4 more tokens [T6..T9] - causes wrap!
    // Buffer state: [T6,T7,T8,T9,T4,T5,_,_] after auto-evict
    // Actually since max=8 and we have 6, adding 4 means 10 > 8
    // So we evict 2 oldest (T0,T1), leaving T2..T9 in the buffer
    const int phase2_tokens = 4;
    auto h_K2 = generateRandomFP32(phase2_tokens * kv_dim, 300);
    auto h_V2 = generateRandomFP32(phase2_tokens * kv_dim, 400);

    cudaMemcpy(d_K, h_K2.data(), phase2_tokens * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V2.data(), phase2_tokens * kv_dim * sizeof(float), cudaMemcpyHostToDevice);

    ASSERT_TRUE(cache->append(0, 0, d_K, d_V, phase2_tokens));

    // Should have evicted 2 tokens (T0, T1), keeping 8
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 8);
    EXPECT_EQ(cache->get_total_evicted(), 2);

    // Head should have wrapped: 6 + 4 = 10 % 8 = 2
    EXPECT_EQ(cache->get_head_position(0, 0), 2);

    // Now buffer IS wrapped (tail=2, head=2, but count=8 so tail=(2-8+8)%8=2)
    // Actually: tail = (head - count + max) % max = (2 - 8 + 8) % 8 = 2
    // So tail=2, head=2, but count=8 means whole buffer is used
    // is_wrapped check: tail >= head && count > 0 --> 2 >= 2 && 8 > 0 --> TRUE
    EXPECT_TRUE(cache->is_wrapped(0, 0));

    // Retrieve and verify linearization happens
    const void *d_K_out, *d_V_out;
    int kv_len;
    ASSERT_TRUE(cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len));
    EXPECT_EQ(kv_len, 8);
    EXPECT_EQ(cache->get_linearization_count(), 1); // Should have linearized

    // Verify content: should have T2,T3,T4,T5 from phase1 and T6,T7,T8,T9 from phase2
    std::vector<float> h_K_out(8 * kv_dim);
    std::vector<float> h_V_out(8 * kv_dim);
    cudaMemcpy(h_K_out.data(), d_K_out, 8 * kv_dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_V_out.data(), d_V_out, 8 * kv_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Check T2-T5 (indices 2-5 from phase1 data)
    for (int t = 0; t < 4; ++t)
    {
        for (int d = 0; d < kv_dim; ++d)
        {
            int src_idx = (t + 2) * kv_dim + d; // T2-T5 in original
            int dst_idx = t * kv_dim + d;       // T0-T3 in output
            EXPECT_FLOAT_EQ(h_K_out[dst_idx], h_K1[src_idx])
                << "K mismatch at output token " << t << " dim " << d;
        }
    }

    // Check T6-T9 (indices 0-3 from phase2 data)
    for (int t = 0; t < 4; ++t)
    {
        for (int d = 0; d < kv_dim; ++d)
        {
            int src_idx = t * kv_dim + d;       // T6-T9 in phase2
            int dst_idx = (t + 4) * kv_dim + d; // T4-T7 in output
            EXPECT_FLOAT_EQ(h_K_out[dst_idx], h_K2[src_idx])
                << "K mismatch at output token " << (t + 4) << " dim " << d;
        }
    }

    cudaFree(d_K);
    cudaFree(d_V);

    LOG_INFO("[WrapAround_FP32] PASSED - linearization_count=" << cache->get_linearization_count());
}

// =============================================================================
// Test: O(1) Eviction
// =============================================================================

TEST(Test__CUDARingKVCache, Eviction_O1)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 100;
    const int n_kv_heads = 4;
    const int head_dim = 32;
    const int kv_dim = n_kv_heads * head_dim;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);

    // Fill with 50 tokens
    auto h_K = generateRandomFP32(50 * kv_dim);
    auto h_V = generateRandomFP32(50 * kv_dim);

    float *d_K, *d_V;
    cudaMalloc(&d_K, 50 * kv_dim * sizeof(float));
    cudaMalloc(&d_V, 50 * kv_dim * sizeof(float));
    cudaMemcpy(d_K, h_K.data(), 50 * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), 50 * kv_dim * sizeof(float), cudaMemcpyHostToDevice);

    cache->append(0, 0, d_K, d_V, 50);
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 50);

    // Evict 20 tokens - should be O(1), no kernel launch
    cache->evict_oldest(0, 0, 20);

    EXPECT_EQ(cache->get_cached_tokens(0, 0), 30);
    EXPECT_EQ(cache->get_total_evicted(), 20);

    // Head position unchanged (eviction only affects tail)
    EXPECT_EQ(cache->get_head_position(0, 0), 50);

    // Retrieve remaining 30 tokens
    const void *d_K_out, *d_V_out;
    int kv_len;
    cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len);
    EXPECT_EQ(kv_len, 30);

    // Verify content: should have T20-T49
    std::vector<float> h_K_out(30 * kv_dim);
    cudaMemcpy(h_K_out.data(), d_K_out, 30 * kv_dim * sizeof(float), cudaMemcpyDeviceToHost);

    for (int t = 0; t < 30; ++t)
    {
        for (int d = 0; d < kv_dim; ++d)
        {
            int src_idx = (t + 20) * kv_dim + d; // T20-T49 in original
            int dst_idx = t * kv_dim + d;
            EXPECT_FLOAT_EQ(h_K_out[dst_idx], h_K[src_idx])
                << "K mismatch at token " << t;
        }
    }

    cudaFree(d_K);
    cudaFree(d_V);

    LOG_INFO("[Eviction_O1] PASSED");
}

// =============================================================================
// Test: Sliding Window Pattern
// =============================================================================

TEST(Test__CUDARingKVCache, SlidingWindow)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    // Simulate sliding window attention with window_size=32
    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 32; // Window size
    const int n_kv_heads = 4;
    const int head_dim = 16;
    const int kv_dim = n_kv_heads * head_dim;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);

    float *d_K, *d_V;
    cudaMalloc(&d_K, kv_dim * sizeof(float));
    cudaMalloc(&d_V, kv_dim * sizeof(float));

    // Simulate 100 decode steps
    for (int step = 0; step < 100; ++step)
    {
        auto h_K = generateRandomFP32(kv_dim, step);
        auto h_V = generateRandomFP32(kv_dim, step + 1000);

        cudaMemcpy(d_K, h_K.data(), kv_dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, h_V.data(), kv_dim * sizeof(float), cudaMemcpyHostToDevice);

        // Append 1 token
        cache->append(0, 0, d_K, d_V, 1);

        // Cache should never exceed window size (auto-evicts)
        EXPECT_LE(cache->get_cached_tokens(0, 0), max_seq_len);
    }

    // After 100 steps with window=32, should have exactly 32 tokens
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 32);

    // 100 appends with window=32 means 68 evicted
    EXPECT_EQ(cache->get_total_evicted(), 68);

    cudaFree(d_K);
    cudaFree(d_V);

    LOG_INFO("[SlidingWindow] PASSED - evicted=" << cache->get_total_evicted());
}

// =============================================================================
// Test: Batched Gather
// =============================================================================

TEST(Test__CUDARingKVCache, BatchedGather)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    const int n_layers = 1;
    const int batch_size = 4;
    const int max_seq_len = 64;
    const int n_kv_heads = 2;
    const int head_dim = 32;
    const int kv_dim = n_kv_heads * head_dim;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);

    // Fill each sequence with different lengths
    int seq_lens[] = {10, 20, 15, 25};
    std::vector<std::vector<float>> h_Ks(batch_size);
    std::vector<std::vector<float>> h_Vs(batch_size);

    float *d_K, *d_V;
    cudaMalloc(&d_K, 30 * kv_dim * sizeof(float));
    cudaMalloc(&d_V, 30 * kv_dim * sizeof(float));

    for (int seq = 0; seq < batch_size; ++seq)
    {
        h_Ks[seq] = generateRandomFP32(seq_lens[seq] * kv_dim, seq * 100);
        h_Vs[seq] = generateRandomFP32(seq_lens[seq] * kv_dim, seq * 100 + 1000);

        cudaMemcpy(d_K, h_Ks[seq].data(), seq_lens[seq] * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, h_Vs[seq].data(), seq_lens[seq] * kv_dim * sizeof(float), cudaMemcpyHostToDevice);

        cache->append(0, seq, d_K, d_V, seq_lens[seq]);
    }

    // Verify individual sequence lengths
    for (int seq = 0; seq < batch_size; ++seq)
    {
        EXPECT_EQ(cache->get_cached_tokens(0, seq), seq_lens[seq]);
    }

    // Gather all sequences
    int max_kv_len = 25; // Max sequence length
    float *d_K_gathered, *d_V_gathered;
    cudaMalloc(&d_K_gathered, batch_size * max_kv_len * kv_dim * sizeof(float));
    cudaMalloc(&d_V_gathered, batch_size * max_kv_len * kv_dim * sizeof(float));

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
    cudaMemcpy(h_K_gathered.data(), d_K_gathered,
               batch_size * max_kv_len * kv_dim * sizeof(float),
               cudaMemcpyDeviceToHost);

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

    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_K_gathered);
    cudaFree(d_V_gathered);

    LOG_INFO("[BatchedGather] PASSED");
}

// =============================================================================
// Test: Contiguous Optimization
// =============================================================================

TEST(Test__CUDARingKVCache, ContiguousOptimization)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 64;
    const int n_kv_heads = 4;
    const int head_dim = 32;
    const int kv_dim = n_kv_heads * head_dim;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);

    // Append tokens without wrapping
    auto h_K = generateRandomFP32(30 * kv_dim);
    auto h_V = generateRandomFP32(30 * kv_dim);

    float *d_K, *d_V;
    cudaMalloc(&d_K, 30 * kv_dim * sizeof(float));
    cudaMalloc(&d_V, 30 * kv_dim * sizeof(float));
    cudaMemcpy(d_K, h_K.data(), 30 * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), 30 * kv_dim * sizeof(float), cudaMemcpyHostToDevice);

    cache->append(0, 0, d_K, d_V, 30);

    // Should NOT be wrapped
    EXPECT_FALSE(cache->is_wrapped(0, 0));

    // Get K/V - should return direct pointer (no linearization)
    const void *d_K_out, *d_V_out;
    int kv_len;
    cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len);

    // No linearizations should have occurred
    EXPECT_EQ(cache->get_linearization_count(), 0);

    // Multiple retrievals should still not linearize
    for (int i = 0; i < 10; ++i)
    {
        cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len);
    }
    EXPECT_EQ(cache->get_linearization_count(), 0);

    cudaFree(d_K);
    cudaFree(d_V);

    LOG_INFO("[ContiguousOptimization] PASSED - linearizations=0");
}

// =============================================================================
// Test: Clear Operations
// =============================================================================

TEST(Test__CUDARingKVCache, ClearOperations)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    const int n_layers = 3;
    const int batch_size = 2;
    const int max_seq_len = 32;
    const int n_kv_heads = 2;
    const int head_dim = 16;
    const int kv_dim = n_kv_heads * head_dim;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);

    // Fill all layers and sequences
    auto h_K = generateRandomFP32(10 * kv_dim);
    auto h_V = generateRandomFP32(10 * kv_dim);

    float *d_K, *d_V;
    cudaMalloc(&d_K, 10 * kv_dim * sizeof(float));
    cudaMalloc(&d_V, 10 * kv_dim * sizeof(float));
    cudaMemcpy(d_K, h_K.data(), 10 * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), 10 * kv_dim * sizeof(float), cudaMemcpyHostToDevice);

    for (int layer = 0; layer < n_layers; ++layer)
    {
        for (int seq = 0; seq < batch_size; ++seq)
        {
            cache->append(layer, seq, d_K, d_V, 10);
        }
    }

    // Verify all filled
    for (int layer = 0; layer < n_layers; ++layer)
    {
        for (int seq = 0; seq < batch_size; ++seq)
        {
            EXPECT_EQ(cache->get_cached_tokens(layer, seq), 10);
        }
    }

    // Clear single sequence
    cache->clear_sequence(1, 0);
    EXPECT_EQ(cache->get_cached_tokens(1, 0), 0);
    EXPECT_EQ(cache->get_cached_tokens(1, 1), 10); // Other sequence unchanged

    // Clear entire layer
    cache->clear_layer(2);
    EXPECT_EQ(cache->get_cached_tokens(2, 0), 0);
    EXPECT_EQ(cache->get_cached_tokens(2, 1), 0);

    // Clear all
    cache->clear();
    for (int layer = 0; layer < n_layers; ++layer)
    {
        for (int seq = 0; seq < batch_size; ++seq)
        {
            EXPECT_EQ(cache->get_cached_tokens(layer, seq), 0);
        }
    }

    cudaFree(d_K);
    cudaFree(d_V);

    LOG_INFO("[ClearOperations] PASSED");
}

// =============================================================================
// Test: Multi-Precision (FP16)
// =============================================================================

TEST(Test__CUDARingKVCache, MultiPrecision_FP16)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 32;
    const int n_kv_heads = 2;
    const int head_dim = 32;
    const int kv_dim = n_kv_heads * head_dim;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP16,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ASSERT_NE(cache, nullptr);
    EXPECT_EQ(cache->precision(), ActivationPrecision::FP16);

    // Generate FP32 data and convert to FP16
    auto h_K_fp32 = generateRandomFP32(10 * kv_dim);
    auto h_V_fp32 = generateRandomFP32(10 * kv_dim);

    std::vector<__half> h_K_fp16(10 * kv_dim);
    std::vector<__half> h_V_fp16(10 * kv_dim);
    for (size_t i = 0; i < h_K_fp32.size(); ++i)
    {
        h_K_fp16[i] = __float2half(h_K_fp32[i]);
        h_V_fp16[i] = __float2half(h_V_fp32[i]);
    }

    __half *d_K, *d_V;
    cudaMalloc(&d_K, 10 * kv_dim * sizeof(__half));
    cudaMalloc(&d_V, 10 * kv_dim * sizeof(__half));
    cudaMemcpy(d_K, h_K_fp16.data(), 10 * kv_dim * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V_fp16.data(), 10 * kv_dim * sizeof(__half), cudaMemcpyHostToDevice);

    ASSERT_TRUE(cache->append(0, 0, d_K, d_V, 10));
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 10);

    const void *d_K_out, *d_V_out;
    int kv_len;
    ASSERT_TRUE(cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len));
    EXPECT_EQ(kv_len, 10);

    // Verify content
    std::vector<__half> h_K_out(10 * kv_dim);
    cudaMemcpy(h_K_out.data(), d_K_out, 10 * kv_dim * sizeof(__half), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < 10 * kv_dim; ++i)
    {
        float expected = __half2float(h_K_fp16[i]);
        float actual = __half2float(h_K_out[i]);
        EXPECT_FLOAT_EQ(actual, expected) << "FP16 mismatch at " << i;
    }

    cudaFree(d_K);
    cudaFree(d_V);

    LOG_INFO("[MultiPrecision_FP16] PASSED");
}

// =============================================================================
// REGRESSION TEST: FP32 ITensor → FP16 cache via appendWithStream
//
// Locks in the fix for the bug where appendWithStream() was never called
// because KVCacheAppendStage::Params.device_id defaulted to CPU.
//
// The bug path was: append(ITensor*) gets raw FP32 GPU pointer → passes it
// directly to ring buffer's append(void*) → raw FP32 bytes (4 bytes/elem)
// interpreted as __half (2 bytes/elem) → complete data corruption.
//
// The correct path: appendWithStream(ITensor*) detects FP32→FP16 mismatch →
// calls cuda_convert_tensor_to_fp16() on GPU → correct FP16 in ring buffer.
//
// This test validates the appendWithStream conversion path directly.
// =============================================================================

TEST(Test__CUDARingKVCache, AppendWithStream_FP32_to_FP16_Conversion)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 32;
    const int n_kv_heads = 2;
    const int head_dim = 32;
    const int kv_dim = n_kv_heads * head_dim;
    const int num_tokens = 8;

    // Create FP16 precision cache
    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP16,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ASSERT_NE(cache, nullptr);
    EXPECT_EQ(cache->precision(), ActivationPrecision::FP16);

    // Create FP32 tensors with known data (simulating projected K/V from GEMM)
    auto K_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(kv_dim)});
    auto V_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(kv_dim)});

    // Fill with recognizable values in [-1, 1] range
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < num_tokens * kv_dim; ++i)
    {
        K_tensor->mutable_data()[i] = dist(rng);
        V_tensor->mutable_data()[i] = dist(rng);
    }

    // Upload to GPU (this populates gpu_data_ptr())
    DeviceId cuda_dev = DeviceId::cuda(0);
    ASSERT_TRUE(K_tensor->ensureOnDevice(cuda_dev));
    ASSERT_TRUE(V_tensor->ensureOnDevice(cuda_dev));
    ASSERT_NE(K_tensor->gpu_data_ptr(), nullptr);
    ASSERT_NE(V_tensor->gpu_data_ptr(), nullptr);

    // Use appendWithStream (the correct GPU path). This should detect
    // FP32→FP16 mismatch and convert on the caller's explicit stream.
    ScopedCudaStream stream;
    ASSERT_TRUE(cache->appendWithStream(0, 0,
                                        static_cast<const ITensor *>(K_tensor.get()),
                                        static_cast<const ITensor *>(V_tensor.get()),
                                        num_tokens, stream.opaque()));
    stream.synchronize();

    EXPECT_EQ(cache->get_cached_tokens(0, 0), num_tokens);

    // Retrieve cached data
    const void *d_K_out, *d_V_out;
    int kv_len;
    ASSERT_TRUE(cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len));
    EXPECT_EQ(kv_len, num_tokens);

    // Read back FP16 data from cache
    std::vector<uint16_t> h_K_out_fp16(num_tokens * kv_dim);
    std::vector<uint16_t> h_V_out_fp16(num_tokens * kv_dim);
    cudaMemcpy(h_K_out_fp16.data(), d_K_out,
               num_tokens * kv_dim * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_V_out_fp16.data(), d_V_out,
               num_tokens * kv_dim * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    // Verify: FP16 round-trip should match FP32→FP16→FP32 within tolerance
    const float *k_src = K_tensor->data();
    const float *v_src = V_tensor->data();
    float max_k_err = 0.0f;
    float max_v_err = 0.0f;
    int k_zero_count = 0;

    for (size_t i = 0; i < num_tokens * kv_dim; ++i)
    {
        float k_cached = fp16_to_fp32(h_K_out_fp16[i]);
        float v_cached = fp16_to_fp32(h_V_out_fp16[i]);

        // FP16 has ~3 decimal digits of precision; error should be < 1e-3 for [-1,1]
        float k_err = std::abs(k_cached - k_src[i]);
        float v_err = std::abs(v_cached - v_src[i]);
        max_k_err = std::max(max_k_err, k_err);
        max_v_err = std::max(max_v_err, v_err);

        if (h_K_out_fp16[i] == 0 && k_src[i] != 0.0f)
            ++k_zero_count;
    }

    LOG_INFO("[AppendWithStream_FP32_to_FP16] max_k_err=" << max_k_err
                                                          << " max_v_err=" << max_v_err
                                                          << " k_zero_count=" << k_zero_count);

    // FP16 in [-1,1] range should have error < 0.001
    EXPECT_LT(max_k_err, 0.001f)
        << "REGRESSION: FP16 K data in cache doesn't match FP32 source. "
           "If errors are very large (>1.0), appendWithStream() likely "
           "wasn't called and raw FP32 bytes were written to FP16 buffer.";
    EXPECT_LT(max_v_err, 0.001f)
        << "REGRESSION: FP16 V data in cache doesn't match FP32 source.";

    // Verify no spurious zeros (sign that raw FP32 bytes were misinterpreted)
    EXPECT_EQ(k_zero_count, 0)
        << "REGRESSION: Found " << k_zero_count << " unexpected zero values. "
                                                   "This suggests FP32→FP16 conversion was not performed.";

    LOG_INFO("[AppendWithStream_FP32_to_FP16] PASSED");
}

// =============================================================================
// REGRESSION TEST: FP32 ITensor → FP16 cache via append (non-stream)
//
// Locks in the fail-fast behavior for the legacy non-stream GPU append path.
// GPU cache writes must use appendWithStream() so residency, format conversion,
// and graph-capture ordering all happen on the caller's explicit stream.
// =============================================================================

TEST(Test__CUDARingKVCache, Append_ITensor_FP32_to_FP16_RequiresExplicitStream)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 32;
    const int n_kv_heads = 2;
    const int head_dim = 32;
    const int kv_dim = n_kv_heads * head_dim;
    const int num_tokens = 4;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP16,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ASSERT_NE(cache, nullptr);

    // Create FP32 tensor with known data on GPU
    auto K_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(kv_dim)});
    auto V_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(kv_dim)});

    for (size_t i = 0; i < num_tokens * kv_dim; ++i)
    {
        K_tensor->mutable_data()[i] = 0.5f; // Easy to verify
        V_tensor->mutable_data()[i] = -0.25f;
    }

    DeviceId cuda_dev = DeviceId::cuda(0);
    ASSERT_TRUE(K_tensor->ensureOnDevice(cuda_dev));
    ASSERT_TRUE(V_tensor->ensureOnDevice(cuda_dev));

    // The non-stream ITensor append path used to reinterpret FP32 bytes as
    // FP16 and write corrupted cache rows. It now fails before touching cache
    // state, forcing callers onto appendWithStream().
    bool append_ok = cache->append(0, 0,
                                   static_cast<const ITensor *>(K_tensor.get()),
                                   static_cast<const ITensor *>(V_tensor.get()),
                                   num_tokens);
    ASSERT_FALSE(append_ok) << "GPU append(ITensor) must require appendWithStream()";
}

TEST(Test__CUDARingKVCache, AppendWithStream_RejectsNullAndAcceptsExplicitStream)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 32;
    const int n_kv_heads = 2;
    const int head_dim = 32;
    const int kv_dim = n_kv_heads * head_dim;
    const int num_tokens = 4;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP16,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ASSERT_NE(cache, nullptr);

    auto K_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(kv_dim)});
    auto V_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(kv_dim)});
    for (size_t i = 0; i < num_tokens * kv_dim; ++i)
    {
        K_tensor->mutable_data()[i] = 0.125f * static_cast<float>((i % 7) - 3);
        V_tensor->mutable_data()[i] = -0.0625f * static_cast<float>((i % 5) - 2);
    }

    EXPECT_FALSE(cache->appendWithStream(0, 0,
                                         static_cast<const ITensor *>(K_tensor.get()),
                                         static_cast<const ITensor *>(V_tensor.get()),
                                         num_tokens, nullptr));
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 0);

    ScopedCudaStream stream;
    ASSERT_TRUE(cache->appendWithStream(0, 0,
                                        static_cast<const ITensor *>(K_tensor.get()),
                                        static_cast<const ITensor *>(V_tensor.get()),
                                        num_tokens, stream.opaque()));
    stream.synchronize();
    EXPECT_EQ(cache->get_cached_tokens(0, 0), num_tokens);
}

// =============================================================================
// Test: Multi-Precision (BF16)
// =============================================================================

TEST(Test__CUDARingKVCache, MultiPrecision_BF16)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 32;
    const int n_kv_heads = 2;
    const int head_dim = 32;
    const int kv_dim = n_kv_heads * head_dim;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::BF16,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ASSERT_NE(cache, nullptr);
    EXPECT_EQ(cache->precision(), ActivationPrecision::BF16);

    // Generate FP32 data and convert to BF16
    auto h_K_fp32 = generateRandomFP32(10 * kv_dim);
    auto h_V_fp32 = generateRandomFP32(10 * kv_dim);

    std::vector<__nv_bfloat16> h_K_bf16(10 * kv_dim);
    std::vector<__nv_bfloat16> h_V_bf16(10 * kv_dim);
    for (size_t i = 0; i < h_K_fp32.size(); ++i)
    {
        h_K_bf16[i] = __float2bfloat16(h_K_fp32[i]);
        h_V_bf16[i] = __float2bfloat16(h_V_fp32[i]);
    }

    __nv_bfloat16 *d_K, *d_V;
    cudaMalloc(&d_K, 10 * kv_dim * sizeof(__nv_bfloat16));
    cudaMalloc(&d_V, 10 * kv_dim * sizeof(__nv_bfloat16));
    cudaMemcpy(d_K, h_K_bf16.data(), 10 * kv_dim * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V_bf16.data(), 10 * kv_dim * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    ASSERT_TRUE(cache->append(0, 0, d_K, d_V, 10));
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 10);

    const void *d_K_out, *d_V_out;
    int kv_len;
    ASSERT_TRUE(cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len));
    EXPECT_EQ(kv_len, 10);

    // Verify content
    std::vector<__nv_bfloat16> h_K_out(10 * kv_dim);
    cudaMemcpy(h_K_out.data(), d_K_out, 10 * kv_dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < 10 * kv_dim; ++i)
    {
        float expected = __bfloat162float(h_K_bf16[i]);
        float actual = __bfloat162float(h_K_out[i]);
        EXPECT_FLOAT_EQ(actual, expected) << "BF16 mismatch at " << i;
    }

    cudaFree(d_K);
    cudaFree(d_V);

    LOG_INFO("[MultiPrecision_BF16] PASSED");
}

// =============================================================================
// Test: IWorkspaceConsumer Interface - getWorkspaceRequirements
// =============================================================================

TEST(Test__CUDARingKVCache, WorkspaceRequirements)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    const int n_layers = 1;
    const int batch_size = 4;
    const int max_seq_len = 64;
    const int n_kv_heads = 2;
    const int head_dim = 32;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ASSERT_NE(cache, nullptr);

    // Cast to IWorkspaceConsumer to test the interface
    auto *workspace_consumer = dynamic_cast<IWorkspaceConsumer *>(cache.get());
    ASSERT_NE(workspace_consumer, nullptr);

    // Get workspace requirements with default batch size
    auto reqs = workspace_consumer->getWorkspaceRequirements(0);
    EXPECT_EQ(reqs.buffers.size(), 4u); // K_PTRS, V_PTRS, TAILS, COUNTS

    // Verify buffer names and sizes
    bool found_k_ptrs = false, found_v_ptrs = false;
    bool found_tails = false, found_counts = false;

    for (const auto &buf : reqs.buffers)
    {
        if (buf.name == KVCacheWorkspaceBuffers::BATCH_K_PTRS)
        {
            found_k_ptrs = true;
            EXPECT_EQ(buf.size_bytes, batch_size * sizeof(void *));
            EXPECT_FALSE(buf.required); // Optional buffer
        }
        else if (buf.name == KVCacheWorkspaceBuffers::BATCH_V_PTRS)
        {
            found_v_ptrs = true;
            EXPECT_EQ(buf.size_bytes, batch_size * sizeof(void *));
            EXPECT_FALSE(buf.required);
        }
        else if (buf.name == KVCacheWorkspaceBuffers::BATCH_TAILS)
        {
            found_tails = true;
            EXPECT_EQ(buf.size_bytes, batch_size * sizeof(int));
            EXPECT_FALSE(buf.required);
        }
        else if (buf.name == KVCacheWorkspaceBuffers::BATCH_COUNTS)
        {
            found_counts = true;
            EXPECT_EQ(buf.size_bytes, batch_size * sizeof(int));
            EXPECT_FALSE(buf.required);
        }
    }

    EXPECT_TRUE(found_k_ptrs) << "Missing BATCH_K_PTRS buffer";
    EXPECT_TRUE(found_v_ptrs) << "Missing BATCH_V_PTRS buffer";
    EXPECT_TRUE(found_tails) << "Missing BATCH_TAILS buffer";
    EXPECT_TRUE(found_counts) << "Missing BATCH_COUNTS buffer";

    // Test with explicit batch size
    auto reqs2 = workspace_consumer->getWorkspaceRequirements(8);
    for (const auto &buf : reqs2.buffers)
    {
        if (buf.name == KVCacheWorkspaceBuffers::BATCH_K_PTRS ||
            buf.name == KVCacheWorkspaceBuffers::BATCH_V_PTRS)
        {
            EXPECT_EQ(buf.size_bytes, 8u * sizeof(void *));
        }
        else if (buf.name == KVCacheWorkspaceBuffers::BATCH_TAILS ||
                 buf.name == KVCacheWorkspaceBuffers::BATCH_COUNTS)
        {
            EXPECT_EQ(buf.size_bytes, 8u * sizeof(int));
        }
    }

    LOG_INFO("[WorkspaceRequirements] PASSED");
}

// =============================================================================
// Test: IWorkspaceConsumer Interface - bindWorkspace/hasWorkspace
// =============================================================================

TEST(Test__CUDARingKVCache, WorkspaceBinding)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    const int n_layers = 1;
    const int batch_size = 4;
    const int max_seq_len = 64;
    const int n_kv_heads = 2;
    const int head_dim = 32;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ASSERT_NE(cache, nullptr);

    auto *workspace_consumer = dynamic_cast<IWorkspaceConsumer *>(cache.get());
    ASSERT_NE(workspace_consumer, nullptr);

    // Initially no workspace bound
    EXPECT_FALSE(workspace_consumer->hasWorkspace());
    EXPECT_EQ(workspace_consumer->getWorkspace(), nullptr);

    // Bind nullptr (unbind)
    workspace_consumer->bindWorkspace(nullptr);
    EXPECT_FALSE(workspace_consumer->hasWorkspace());

    LOG_INFO("[WorkspaceBinding] PASSED");
}

// =============================================================================
// Test: BatchedGather works without workspace (backward compatibility)
// =============================================================================

TEST(Test__CUDARingKVCache, BatchedGatherWithoutWorkspace)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    const int n_layers = 1;
    const int batch_size = 2;
    const int max_seq_len = 32;
    const int n_kv_heads = 2;
    const int head_dim = 16;
    const int kv_dim = n_kv_heads * head_dim;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ASSERT_NE(cache, nullptr);

    // Verify no workspace bound
    auto *workspace_consumer = dynamic_cast<IWorkspaceConsumer *>(cache.get());
    ASSERT_NE(workspace_consumer, nullptr);
    EXPECT_FALSE(workspace_consumer->hasWorkspace());

    // Fill sequences
    float *d_K, *d_V;
    cudaMalloc(&d_K, 10 * kv_dim * sizeof(float));
    cudaMalloc(&d_V, 10 * kv_dim * sizeof(float));

    auto h_K = generateRandomFP32(10 * kv_dim, 42);
    auto h_V = generateRandomFP32(10 * kv_dim, 43);

    for (int seq = 0; seq < batch_size; ++seq)
    {
        int seq_len = 5 + seq * 3; // 5 and 8 tokens
        cudaMemcpy(d_K, h_K.data(), seq_len * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, h_V.data(), seq_len * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
        cache->append(0, seq, d_K, d_V, seq_len);
    }

    // Gather without workspace - should fall back to per-call allocation
    int max_kv_len = 10;
    float *d_K_gathered, *d_V_gathered;
    cudaMalloc(&d_K_gathered, batch_size * max_kv_len * kv_dim * sizeof(float));
    cudaMalloc(&d_V_gathered, batch_size * max_kv_len * kv_dim * sizeof(float));

    std::vector<int> kv_lens(batch_size);
    int actual_max = cache->gather_kv_batched(0, batch_size,
                                              d_K_gathered, d_V_gathered,
                                              kv_lens.data(), max_kv_len);

    EXPECT_GT(actual_max, 0);
    EXPECT_EQ(kv_lens[0], 5);
    EXPECT_EQ(kv_lens[1], 8);

    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_K_gathered);
    cudaFree(d_V_gathered);

    LOG_INFO("[BatchedGatherWithoutWorkspace] PASSED - backward compatibility verified");
}
