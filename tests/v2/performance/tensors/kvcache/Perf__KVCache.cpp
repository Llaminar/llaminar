/**
 * @file Perf__KVCache.cpp
 * @brief Performance benchmarks for KV cache operations
 * @author David Sanftenberg
 *
 * Benchmarks:
 * - append_kv() throughput (single token decode, batch prefill)
 * - evict_oldest() latency for sliding window
 * - Memory bandwidth utilization
 * - Scaling with sequence length and number of layers
 */

#include <gtest/gtest.h>
#include "kernels/cpu/CPUKVCache.h"
#include "tensors/Tensors.h"
#include "utils/Logger.h"
#include "utils/MPIContext.h"
#include <chrono>
#include <vector>
#include <numeric>
#include <iomanip>
#include <omp.h>

using namespace llaminar2;

// Alias for backward compatibility with tests
using KVCache = CPUKVCache<ActivationPrecision::FP32>;

// Single-rank MPI context for performance tests
static MPIContext getTestMPIContext()
{
    return MPIContext(0, 1, MPI_COMM_WORLD);
}

/**
 * @class KVCachePerformanceTest
 * @brief Performance test fixture for KV cache operations
 */
class KVCachePerformanceTest : public ::testing::Test
{
protected:
    // Qwen2.5-0.5B parameters
    static constexpr int N_LAYERS = 24;
    static constexpr int N_KV_HEADS = 2;
    static constexpr int HEAD_DIM = 64;
    static constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM; // 128

    // Larger model parameters (Qwen2.5-7B style)
    static constexpr int N_LAYERS_7B = 32;
    static constexpr int N_KV_HEADS_7B = 4;
    static constexpr int HEAD_DIM_7B = 128;
    static constexpr int KV_DIM_7B = N_KV_HEADS_7B * HEAD_DIM_7B; // 512

    void SetUp() override
    {
        // Warm up memory allocator
        std::vector<float> warmup(1024 * 1024);
        std::fill(warmup.begin(), warmup.end(), 0.0f);
    }

    /**
     * @brief Create random K/V tensors for testing
     */
    std::pair<std::shared_ptr<FP32Tensor>, std::shared_ptr<FP32Tensor>>
    create_kv_tensors(int seq_len, int kv_dim, int seed = 42)
    {
        auto K = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)}, DeviceId::cpu());
        auto V = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)}, DeviceId::cpu());

        // Fill with deterministic random values
        float *k_data = K->mutable_data();
        float *v_data = V->mutable_data();
        for (int i = 0; i < seq_len * kv_dim; ++i)
        {
            k_data[i] = static_cast<float>((seed * 1103515245 + i) % 1000) / 500.0f - 1.0f;
            v_data[i] = static_cast<float>((seed * 1103515245 + i + 1) % 1000) / 500.0f - 1.0f;
        }

        return {K, V};
    }

    /**
     * @brief Compute memory bandwidth in GB/s
     */
    double compute_bandwidth_gbps(size_t bytes, double time_ms)
    {
        return (bytes / 1e9) / (time_ms / 1000.0);
    }

    /**
     * @brief Format bytes as human-readable string
     */
    std::string format_bytes(size_t bytes)
    {
        if (bytes >= 1024 * 1024 * 1024)
            return std::to_string(bytes / (1024 * 1024 * 1024)) + " GB";
        if (bytes >= 1024 * 1024)
            return std::to_string(bytes / (1024 * 1024)) + " MB";
        if (bytes >= 1024)
            return std::to_string(bytes / 1024) + " KB";
        return std::to_string(bytes) + " B";
    }
};

/**
 * @brief Benchmark single-token decode append throughput
 *
 * Measures how fast we can append 1 token to the cache per layer.
 * This is the critical path for autoregressive decode.
 */
TEST_F(KVCachePerformanceTest, SingleTokenAppend)
{
    const int max_seq_len = 2048;
    const int num_tokens = 256; // Simulate 256 decode steps
    const int warmup_steps = 10;

    KVCache cache(getTestMPIContext(), N_LAYERS, /*batch_size=*/1, max_seq_len, N_KV_HEADS, HEAD_DIM);

    // Pre-create K/V tensors for single token
    auto [K, V] = create_kv_tensors(1, KV_DIM);

    // Warmup
    for (int i = 0; i < warmup_steps; ++i)
    {
        for (int layer = 0; layer < N_LAYERS; ++layer)
        {
            cache.append_kv(layer, K.get(), V.get());
        }
    }
    cache.clear();

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();

    for (int token = 0; token < num_tokens; ++token)
    {
        for (int layer = 0; layer < N_LAYERS; ++layer)
        {
            cache.append_kv(layer, K.get(), V.get());
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Calculate metrics
    double time_per_token_us = (time_ms * 1000.0) / num_tokens;
    size_t bytes_per_token = N_LAYERS * 2 * KV_DIM * sizeof(float); // K + V per layer
    double bandwidth_gbps = compute_bandwidth_gbps(bytes_per_token * num_tokens, time_ms);

    std::cout << "\n=== Single Token Append (0.5B model) ===" << std::endl;
    std::cout << "  Tokens appended: " << num_tokens << std::endl;
    std::cout << "  Total time: " << std::fixed << std::setprecision(2) << time_ms << " ms" << std::endl;
    std::cout << "  Time per token: " << std::setprecision(2) << time_per_token_us << " µs" << std::endl;
    std::cout << "  Bytes per token: " << format_bytes(bytes_per_token) << std::endl;
    std::cout << "  Memory bandwidth: " << std::setprecision(2) << bandwidth_gbps << " GB/s" << std::endl;

    // Sanity check: should be able to append at least 10000 tokens/sec
    EXPECT_LT(time_per_token_us, 100.0) << "Single token append too slow (>100µs)";
}

/**
 * @brief Benchmark prefill append (many tokens at once)
 *
 * Measures throughput when appending a batch of tokens (e.g., prompt processing).
 */
TEST_F(KVCachePerformanceTest, BatchPrefill)
{
    const int max_seq_len = 4096;
    const std::vector<int> prefill_sizes = {64, 256, 512, 1024, 2048};

    std::cout << "\n=== Batch Prefill Performance ===" << std::endl;
    std::cout << std::setw(12) << "Tokens"
              << std::setw(12) << "Time (ms)"
              << std::setw(15) << "Tokens/sec"
              << std::setw(12) << "BW (GB/s)" << std::endl;
    std::cout << std::string(51, '-') << std::endl;

    for (int prefill_len : prefill_sizes)
    {
        KVCache cache(getTestMPIContext(), N_LAYERS, /*batch_size=*/1, max_seq_len, N_KV_HEADS, HEAD_DIM);
        auto [K, V] = create_kv_tensors(prefill_len, KV_DIM);

        // Warmup
        for (int layer = 0; layer < N_LAYERS; ++layer)
        {
            cache.append_kv(layer, K.get(), V.get());
        }
        cache.clear();

        // Benchmark
        const int iterations = 10;
        auto start = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < iterations; ++iter)
        {
            for (int layer = 0; layer < N_LAYERS; ++layer)
            {
                cache.append_kv(layer, K.get(), V.get());
            }
            cache.clear();
        }

        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

        size_t bytes_total = N_LAYERS * 2 * prefill_len * KV_DIM * sizeof(float);
        double tokens_per_sec = (prefill_len * N_LAYERS) / (time_ms / 1000.0);
        double bandwidth_gbps = compute_bandwidth_gbps(bytes_total, time_ms);

        std::cout << std::setw(12) << prefill_len
                  << std::setw(12) << std::fixed << std::setprecision(2) << time_ms
                  << std::setw(15) << std::setprecision(0) << tokens_per_sec
                  << std::setw(12) << std::setprecision(2) << bandwidth_gbps << std::endl;
    }
}

/**
 * @brief Benchmark eviction performance
 *
 * Measures latency of sliding window eviction at various cache fill levels.
 */
TEST_F(KVCachePerformanceTest, EvictionLatency)
{
    const int max_seq_len = 4096;
    const std::vector<int> fill_levels = {1024, 2048, 3072, 4096};
    const std::vector<int> evict_amounts = {1, 16, 64, 256, 512};

    std::cout << "\n=== Eviction Latency ===" << std::endl;
    std::cout << std::setw(12) << "Fill"
              << std::setw(12) << "Evict"
              << std::setw(15) << "Time (µs)"
              << std::setw(15) << "BW (GB/s)" << std::endl;
    std::cout << std::string(54, '-') << std::endl;

    for (int fill : fill_levels)
    {
        for (int evict : evict_amounts)
        {
            if (evict >= fill)
                continue;

            KVCache cache(getTestMPIContext(), N_LAYERS, /*batch_size=*/1, max_seq_len, N_KV_HEADS, HEAD_DIM);

            // Fill cache to specified level
            auto [K, V] = create_kv_tensors(fill, KV_DIM);
            for (int layer = 0; layer < N_LAYERS; ++layer)
            {
                cache.append_kv(layer, K.get(), V.get());
            }

            // Benchmark eviction
            const int iterations = 100;
            auto start = std::chrono::high_resolution_clock::now();

            for (int iter = 0; iter < iterations; ++iter)
            {
                cache.evict_oldest(evict);
                // Re-fill to maintain fill level
                auto [K_small, V_small] = create_kv_tensors(evict, KV_DIM);
                for (int layer = 0; layer < N_LAYERS; ++layer)
                {
                    cache.append_kv(layer, K_small.get(), V_small.get());
                }
            }

            auto end = std::chrono::high_resolution_clock::now();
            double time_us = std::chrono::duration<double, std::micro>(end - start).count() / iterations;

            // Eviction moves (fill - evict) tokens per layer, for K and V
            size_t bytes_moved = N_LAYERS * 2 * (fill - evict) * KV_DIM * sizeof(float);
            double bandwidth_gbps = compute_bandwidth_gbps(bytes_moved, time_us / 1000.0);

            std::cout << std::setw(12) << fill
                      << std::setw(12) << evict
                      << std::setw(15) << std::fixed << std::setprecision(1) << time_us
                      << std::setw(15) << std::setprecision(2) << bandwidth_gbps << std::endl;
        }
    }
}

/**
 * @brief Benchmark 7B model scale KV cache
 *
 * Tests performance with larger model parameters.
 */
TEST_F(KVCachePerformanceTest, LargeModelScale)
{
    const int max_seq_len = 8192;
    const int prefill_len = 1024;

    KVCache cache(getTestMPIContext(), N_LAYERS_7B, /*batch_size=*/1, max_seq_len, N_KV_HEADS_7B, HEAD_DIM_7B);
    auto [K, V] = create_kv_tensors(prefill_len, KV_DIM_7B);

    // Warmup
    for (int layer = 0; layer < N_LAYERS_7B; ++layer)
    {
        cache.append_kv(layer, K.get(), V.get());
    }
    cache.clear();

    // Benchmark prefill
    const int iterations = 10;
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter)
    {
        for (int layer = 0; layer < N_LAYERS_7B; ++layer)
        {
            cache.append_kv(layer, K.get(), V.get());
        }
        cache.clear();
    }

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

    size_t bytes_total = N_LAYERS_7B * 2 * prefill_len * KV_DIM_7B * sizeof(float);
    double bandwidth_gbps = compute_bandwidth_gbps(bytes_total, time_ms);

    std::cout << "\n=== 7B Model Scale (32 layers, 4 KV heads, 128 head_dim) ===" << std::endl;
    std::cout << "  Max seq len: " << max_seq_len << std::endl;
    std::cout << "  Prefill tokens: " << prefill_len << std::endl;
    std::cout << "  Cache size: " << format_bytes(N_LAYERS_7B * 2 * max_seq_len * KV_DIM_7B * sizeof(float)) << std::endl;
    std::cout << "  Prefill time: " << std::fixed << std::setprecision(2) << time_ms << " ms" << std::endl;
    std::cout << "  Memory bandwidth: " << bandwidth_gbps << " GB/s" << std::endl;
}

/**
 * @brief Test parallel vs sequential layer processing
 *
 * Simulates what would happen if evict_oldest was parallelized across layers.
 */
TEST_F(KVCachePerformanceTest, ParallelLayerPotential)
{
    const int max_seq_len = 2048;
    const int fill_level = 1024;
    const int evict_amount = 256;
    const int iterations = 50;

    // Create cache and fill it
    KVCache cache(getTestMPIContext(), N_LAYERS, /*batch_size=*/1, max_seq_len, N_KV_HEADS, HEAD_DIM);
    auto [K, V] = create_kv_tensors(fill_level, KV_DIM);
    for (int layer = 0; layer < N_LAYERS; ++layer)
    {
        cache.append_kv(layer, K.get(), V.get());
    }

    // Benchmark current sequential eviction
    auto start_seq = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
    {
        cache.evict_oldest(evict_amount);
        // Re-fill
        auto [K_small, V_small] = create_kv_tensors(evict_amount, KV_DIM);
        for (int layer = 0; layer < N_LAYERS; ++layer)
        {
            cache.append_kv(layer, K_small.get(), V_small.get());
        }
    }
    auto end_seq = std::chrono::high_resolution_clock::now();
    double time_seq_ms = std::chrono::duration<double, std::milli>(end_seq - start_seq).count();

    // Simulate parallelized eviction using raw memmove
    std::vector<std::vector<float>> k_buffers(N_LAYERS, std::vector<float>(max_seq_len * KV_DIM));
    std::vector<std::vector<float>> v_buffers(N_LAYERS, std::vector<float>(max_seq_len * KV_DIM));

    // Initialize buffers
    for (int layer = 0; layer < N_LAYERS; ++layer)
    {
        std::fill(k_buffers[layer].begin(), k_buffers[layer].end(), 1.0f);
        std::fill(v_buffers[layer].begin(), v_buffers[layer].end(), 1.0f);
    }

    int tokens_to_keep = fill_level - evict_amount;
    size_t shift_offset = evict_amount * KV_DIM;
    size_t keep_size = tokens_to_keep * KV_DIM * sizeof(float);

    auto start_par = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
    {
#pragma omp parallel for
        for (int layer = 0; layer < N_LAYERS; ++layer)
        {
            std::memmove(k_buffers[layer].data(), k_buffers[layer].data() + shift_offset, keep_size);
            std::memmove(v_buffers[layer].data(), v_buffers[layer].data() + shift_offset, keep_size);
        }
    }
    auto end_par = std::chrono::high_resolution_clock::now();
    double time_par_ms = std::chrono::duration<double, std::milli>(end_par - start_par).count();

    std::cout << "\n=== Parallel Eviction Potential ===" << std::endl;
    std::cout << "  Fill level: " << fill_level << " tokens" << std::endl;
    std::cout << "  Evict amount: " << evict_amount << " tokens" << std::endl;
    std::cout << "  Iterations: " << iterations << std::endl;
    std::cout << "  Sequential time: " << std::fixed << std::setprecision(2) << time_seq_ms << " ms" << std::endl;
    std::cout << "  Parallel time:   " << time_par_ms << " ms" << std::endl;
    std::cout << "  Potential speedup: " << std::setprecision(2) << time_seq_ms / time_par_ms << "x" << std::endl;
    std::cout << "  OMP threads: " << omp_get_max_threads() << std::endl;
}
