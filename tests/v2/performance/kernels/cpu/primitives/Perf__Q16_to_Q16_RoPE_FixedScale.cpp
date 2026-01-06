/**
 * @file Perf__Q16_to_Q16_RoPE_FixedScale.cpp
 * @brief Performance benchmarks for Q16→Q16 fixed-scale RoPE SIMD implementations
 * @author David Sanftenberg
 *
 * Measures throughput for fixed-scale Q16_1→Q16_1 RoPE operations:
 *   - Scalar implementation (fallback)
 *   - AVX2 implementation (4 chunks × 8 elements)
 *   - AVX512 implementation (2 chunks × 16 elements)
 *
 * The fixed-scale RoPE is used in HybridQ16 pipeline where K projection outputs
 * Q16_1 directly from GEMM, and we need to rescale to a fixed kv_cache_scale
 * for efficient integer attention.
 *
 * Key difference from standard Q16 RoPE: output scale is FIXED, eliminating
 * the expensive horizontal max reduction required for dynamic scaling.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>

#include "kernels/cpu/primitives/RoPEPrimitives.h"
#include "tensors/BlockStructures.h"

using namespace llaminar2;
using namespace llaminar2::primitives;

// ============================================================================
// Performance Test Fixture
// ============================================================================

class Q16_to_Q16_RoPE_FixedScale_Perf : public ::testing::Test
{
protected:
    // Benchmark configuration
    static constexpr size_t WARMUP_ITERATIONS = 20;
    static constexpr size_t BENCHMARK_ITERATIONS = 1000;

    // Model parameters (Qwen2.5-0.5B style)
    static constexpr int HEAD_DIM = 64;
    static constexpr int N_KV_HEADS = 2;
    static constexpr float ROPE_THETA = 1000000.0f;
    static constexpr float KV_CACHE_SCALE = 8.0f;

    // Larger model parameters (Qwen2.5-7B style)
    static constexpr int LARGE_HEAD_DIM = 128;
    static constexpr int LARGE_N_KV_HEADS = 8;

    std::mt19937 rng_{42};

    void SetUp() override
    {
        rng_.seed(42);
    }

    // =========================================================================
    // Data Generation Helpers
    // =========================================================================

    /**
     * @brief Generate random Q16_1 blocks with varying per-block scales
     *
     * This simulates GEMM output where each block has different magnitude.
     */
    std::vector<Q16_1Block> generate_random_q16_blocks(size_t num_blocks)
    {
        std::vector<Q16_1Block> blocks(num_blocks);
        std::uniform_real_distribution<float> scale_dist(0.001f, 2.0f);
        std::uniform_int_distribution<int16_t> qs_dist(-16383, 16383);

        for (auto &blk : blocks)
        {
            blk.d = scale_dist(rng_);
            int32_t sum = 0;
            for (int i = 0; i < 32; ++i)
            {
                blk.qs[i] = qs_dist(rng_);
                sum += blk.qs[i];
            }
            blk.sum_qs = sum;
        }
        return blocks;
    }

    /**
     * @brief Generate Q15-quantized sin/cos tables
     */
    void generate_sincos_tables(int head_dim, int position,
                                std::vector<int16_t> &cos_q15,
                                std::vector<int16_t> &sin_q15)
    {
        const int half_dim = head_dim / 2;
        cos_q15.resize(half_dim);
        sin_q15.resize(half_dim);

        for (int i = 0; i < half_dim; ++i)
        {
            float inv_freq = 1.0f / std::pow(ROPE_THETA, static_cast<float>(2 * i) / head_dim);
            float angle = static_cast<float>(position) * inv_freq;
            cos_q15[i] = static_cast<int16_t>(std::round(std::cos(angle) * 32767.0f));
            sin_q15[i] = static_cast<int16_t>(std::round(std::sin(angle) * 32767.0f));
        }
    }

    // =========================================================================
    // Benchmark Result Structure
    // =========================================================================

    struct BenchmarkResult
    {
        std::string name;
        double elapsed_ms;
        double heads_per_sec;
        double bandwidth_gbps;
        size_t iterations;
    };

    // =========================================================================
    // Output Formatting
    // =========================================================================

    void print_header(const std::string &test_name, int head_dim, int n_heads, int seq_len)
    {
        const int blocks_per_head = head_dim / 32;
        size_t total_blocks = static_cast<size_t>(seq_len) * n_heads * blocks_per_head;
        double input_mb = total_blocks * sizeof(Q16_1Block) / (1024.0 * 1024.0);

        std::cout << "\n╔══════════════════════════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║ " << std::setw(82) << std::left << test_name << "║" << std::endl;
        std::cout << "╠══════════════════════════════════════════════════════════════════════════════════╣" << std::endl;
        std::cout << "║ head_dim=" << std::setw(3) << head_dim
                  << "  n_kv_heads=" << std::setw(2) << n_heads
                  << "  seq_len=" << std::setw(4) << seq_len
                  << "  blocks/head=" << std::setw(2) << blocks_per_head
                  << "  input=" << std::fixed << std::setprecision(2) << std::setw(6) << input_mb << " MB"
                  << "    ║" << std::endl;
        std::cout << "╠══════════════════════════════════════════════════════════════════════════════════╣" << std::endl;
        std::cout << "│   Implementation    │  Time (ms)  │   Heads/sec   │  Bandwidth GB/s  │  Speedup  │" << std::endl;
        std::cout << "├─────────────────────┼─────────────┼───────────────┼──────────────────┼───────────┤" << std::endl;
    }

    void print_result(const BenchmarkResult &result, double baseline_ms = 0.0)
    {
        double speedup = (baseline_ms > 0) ? (baseline_ms / result.elapsed_ms) : 1.0;
        std::string speedup_str;
        if (baseline_ms > 0)
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << speedup << "x";
            speedup_str = oss.str();
        }
        else
        {
            speedup_str = "baseline";
        }

        std::cout << "│ " << std::setw(19) << std::left << result.name
                  << " │ " << std::setw(11) << std::right << std::fixed << std::setprecision(3) << result.elapsed_ms
                  << " │ " << std::setw(13) << std::right << std::fixed << std::setprecision(0) << result.heads_per_sec
                  << " │ " << std::setw(16) << std::right << std::fixed << std::setprecision(2) << result.bandwidth_gbps
                  << " │ " << std::setw(9) << std::right << speedup_str
                  << " │" << std::endl;
    }

    void print_footer()
    {
        std::cout << "╚══════════════════════════════════════════════════════════════════════════════════╝" << std::endl;
    }

    // =========================================================================
    // Benchmark Helpers
    // =========================================================================

    /**
     * @brief Benchmark the per-head fixed-scale RoPE function
     *
     * @param impl_name Name for display
     * @param head_dim Head dimension
     * @param n_heads Number of heads to process
     * @param iterations Number of benchmark iterations
     * @return BenchmarkResult
     */
    template <typename BenchFunc>
    BenchmarkResult benchmark_rope(
        const std::string &impl_name,
        BenchFunc &&bench_fn,
        int head_dim,
        int n_heads,
        size_t iterations)
    {
        const int blocks_per_head = head_dim / 32;
        const size_t total_blocks = static_cast<size_t>(n_heads) * blocks_per_head;

        // Generate input data
        auto q16_in = generate_random_q16_blocks(total_blocks);
        std::vector<Q16_1Block> q16_out(total_blocks);

        // Generate sin/cos for position 0 (representative)
        std::vector<int16_t> cos_q15, sin_q15;
        generate_sincos_tables(head_dim, 42, cos_q15, sin_q15);

        // Warmup
        for (size_t i = 0; i < WARMUP_ITERATIONS; ++i)
        {
            for (int h = 0; h < n_heads; ++h)
            {
                bench_fn(
                    q16_in.data() + h * blocks_per_head,
                    q16_out.data() + h * blocks_per_head,
                    head_dim, cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);
            }
        }

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < iterations; ++i)
        {
            for (int h = 0; h < n_heads; ++h)
            {
                bench_fn(
                    q16_in.data() + h * blocks_per_head,
                    q16_out.data() + h * blocks_per_head,
                    head_dim, cos_q15.data(), sin_q15.data(), KV_CACHE_SCALE);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double total_heads = static_cast<double>(n_heads) * iterations;
        double heads_per_sec = total_heads / (elapsed_ms / 1000.0);

        // Bandwidth: input + output Q16_1 blocks
        double total_bytes = 2.0 * total_blocks * sizeof(Q16_1Block) * iterations;
        double bandwidth_gbps = total_bytes / (1024.0 * 1024.0 * 1024.0) / (elapsed_ms / 1000.0);

        return {impl_name, elapsed_ms, heads_per_sec, bandwidth_gbps, iterations};
    }
};

// ============================================================================
// Scalar Implementation Wrapper (for explicit benchmarking)
// ============================================================================

// Force scalar path by calling the scalar template directly
static void rope_q16_scalar(
    const Q16_1Block *in, Q16_1Block *out, int head_dim,
    const int16_t *cos, const int16_t *sin, float scale)
{
    apply_rope_q16_to_q16_head_fixed_scale_scalar<Q16_1Block>(in, out, head_dim, cos, sin, scale);
}

// ============================================================================
// AVX2 Implementation Wrapper
// ============================================================================

#if defined(__AVX2__)
static void rope_q16_avx2(
    const Q16_1Block *in, Q16_1Block *out, int head_dim,
    const int16_t *cos, const int16_t *sin, float scale)
{
    apply_rope_q16_to_q16_head_fixed_scale_avx2(in, out, head_dim, cos, sin, scale);
}
#endif

// ============================================================================
// AVX512 Implementation Wrapper
// ============================================================================

#if defined(__AVX512F__)
static void rope_q16_avx512(
    const Q16_1Block *in, Q16_1Block *out, int head_dim,
    const int16_t *cos, const int16_t *sin, float scale)
{
    apply_rope_q16_to_q16_head_fixed_scale_avx512(in, out, head_dim, cos, sin, scale);
}
#endif

// ============================================================================
// Dispatcher Wrapper (uses auto-detected best path)
// ============================================================================

static void rope_q16_dispatch(
    const Q16_1Block *in, Q16_1Block *out, int head_dim,
    const int16_t *cos, const int16_t *sin, float scale)
{
    apply_rope_q16_to_q16_head_fixed_scale<Q16_1Block>(in, out, head_dim, cos, sin, scale);
}

// ============================================================================
// Performance Tests
// ============================================================================

/**
 * @test Small head dimension (64), single head - baseline latency test
 */
TEST_F(Q16_to_Q16_RoPE_FixedScale_Perf, SingleHead_HeadDim64)
{
    print_header("Q16→Q16 Fixed-Scale RoPE: Single Head Latency", HEAD_DIM, 1, 1);

    auto scalar_result = benchmark_rope("Scalar", rope_q16_scalar, HEAD_DIM, 1, BENCHMARK_ITERATIONS * 10);
    print_result(scalar_result);
    double baseline_ms = scalar_result.elapsed_ms;

#if defined(__AVX2__)
    auto avx2_result = benchmark_rope("AVX2", rope_q16_avx2, HEAD_DIM, 1, BENCHMARK_ITERATIONS * 10);
    print_result(avx2_result, baseline_ms);
#endif

#if defined(__AVX512F__)
    auto avx512_result = benchmark_rope("AVX512", rope_q16_avx512, HEAD_DIM, 1, BENCHMARK_ITERATIONS * 10);
    print_result(avx512_result, baseline_ms);
#endif

    auto dispatch_result = benchmark_rope("Auto-Dispatch", rope_q16_dispatch, HEAD_DIM, 1, BENCHMARK_ITERATIONS * 10);
    print_result(dispatch_result, baseline_ms);

    print_footer();

    // Verify SIMD versions are faster than scalar
    // Note: Integer-only implementation makes scalar much faster, so SIMD advantage is smaller
#if defined(__AVX512F__)
    EXPECT_LT(avx512_result.elapsed_ms, scalar_result.elapsed_ms * 0.95)
        << "AVX512 should be at least slightly faster than scalar";
#elif defined(__AVX2__)
    EXPECT_LT(avx2_result.elapsed_ms, scalar_result.elapsed_ms * 1.1)
        << "AVX2 should be comparable to scalar";
#endif
}

/**
 * @test Typical decode scenario: 2 KV heads, head_dim=64
 */
TEST_F(Q16_to_Q16_RoPE_FixedScale_Perf, Decode_2KVHeads_HeadDim64)
{
    print_header("Q16→Q16 Fixed-Scale RoPE: Decode (2 KV Heads)", HEAD_DIM, N_KV_HEADS, 1);

    auto scalar_result = benchmark_rope("Scalar", rope_q16_scalar, HEAD_DIM, N_KV_HEADS, BENCHMARK_ITERATIONS * 5);
    print_result(scalar_result);
    double baseline_ms = scalar_result.elapsed_ms;

#if defined(__AVX2__)
    auto avx2_result = benchmark_rope("AVX2", rope_q16_avx2, HEAD_DIM, N_KV_HEADS, BENCHMARK_ITERATIONS * 5);
    print_result(avx2_result, baseline_ms);
#endif

#if defined(__AVX512F__)
    auto avx512_result = benchmark_rope("AVX512", rope_q16_avx512, HEAD_DIM, N_KV_HEADS, BENCHMARK_ITERATIONS * 5);
    print_result(avx512_result, baseline_ms);
#endif

    auto dispatch_result = benchmark_rope("Auto-Dispatch", rope_q16_dispatch, HEAD_DIM, N_KV_HEADS, BENCHMARK_ITERATIONS * 5);
    print_result(dispatch_result, baseline_ms);

    print_footer();
}

/**
 * @test Larger model: 8 KV heads, head_dim=128
 */
TEST_F(Q16_to_Q16_RoPE_FixedScale_Perf, Decode_8KVHeads_HeadDim128)
{
    print_header("Q16→Q16 Fixed-Scale RoPE: Decode (8 KV Heads, Large)", LARGE_HEAD_DIM, LARGE_N_KV_HEADS, 1);

    auto scalar_result = benchmark_rope("Scalar", rope_q16_scalar, LARGE_HEAD_DIM, LARGE_N_KV_HEADS, BENCHMARK_ITERATIONS * 2);
    print_result(scalar_result);
    double baseline_ms = scalar_result.elapsed_ms;

#if defined(__AVX2__)
    auto avx2_result = benchmark_rope("AVX2", rope_q16_avx2, LARGE_HEAD_DIM, LARGE_N_KV_HEADS, BENCHMARK_ITERATIONS * 2);
    print_result(avx2_result, baseline_ms);
#endif

#if defined(__AVX512F__)
    auto avx512_result = benchmark_rope("AVX512", rope_q16_avx512, LARGE_HEAD_DIM, LARGE_N_KV_HEADS, BENCHMARK_ITERATIONS * 2);
    print_result(avx512_result, baseline_ms);
#endif

    auto dispatch_result = benchmark_rope("Auto-Dispatch", rope_q16_dispatch, LARGE_HEAD_DIM, LARGE_N_KV_HEADS, BENCHMARK_ITERATIONS * 2);
    print_result(dispatch_result, baseline_ms);

    print_footer();
}

/**
 * @test Prefill scenario: many positions (seq_len=128) with 2 KV heads
 */
TEST_F(Q16_to_Q16_RoPE_FixedScale_Perf, Prefill_SeqLen128_2KVHeads)
{
    constexpr int SEQ_LEN = 128;
    const int total_heads = SEQ_LEN * N_KV_HEADS;

    print_header("Q16→Q16 Fixed-Scale RoPE: Prefill (seq=128)", HEAD_DIM, N_KV_HEADS, SEQ_LEN);

    auto scalar_result = benchmark_rope("Scalar", rope_q16_scalar, HEAD_DIM, total_heads, BENCHMARK_ITERATIONS);
    print_result(scalar_result);
    double baseline_ms = scalar_result.elapsed_ms;

#if defined(__AVX2__)
    auto avx2_result = benchmark_rope("AVX2", rope_q16_avx2, HEAD_DIM, total_heads, BENCHMARK_ITERATIONS);
    print_result(avx2_result, baseline_ms);
#endif

#if defined(__AVX512F__)
    auto avx512_result = benchmark_rope("AVX512", rope_q16_avx512, HEAD_DIM, total_heads, BENCHMARK_ITERATIONS);
    print_result(avx512_result, baseline_ms);
#endif

    auto dispatch_result = benchmark_rope("Auto-Dispatch", rope_q16_dispatch, HEAD_DIM, total_heads, BENCHMARK_ITERATIONS);
    print_result(dispatch_result, baseline_ms);

    print_footer();
}

/**
 * @test Long prefill: seq_len=512 with 8 KV heads (large model)
 */
TEST_F(Q16_to_Q16_RoPE_FixedScale_Perf, Prefill_SeqLen512_LargeModel)
{
    constexpr int SEQ_LEN = 512;
    const int total_heads = SEQ_LEN * LARGE_N_KV_HEADS;

    print_header("Q16→Q16 Fixed-Scale RoPE: Long Prefill (seq=512, large)", LARGE_HEAD_DIM, LARGE_N_KV_HEADS, SEQ_LEN);

    // Fewer iterations for large data
    auto scalar_result = benchmark_rope("Scalar", rope_q16_scalar, LARGE_HEAD_DIM, total_heads, BENCHMARK_ITERATIONS / 10);
    print_result(scalar_result);
    double baseline_ms = scalar_result.elapsed_ms;

#if defined(__AVX2__)
    auto avx2_result = benchmark_rope("AVX2", rope_q16_avx2, LARGE_HEAD_DIM, total_heads, BENCHMARK_ITERATIONS / 10);
    print_result(avx2_result, baseline_ms);
#endif

#if defined(__AVX512F__)
    auto avx512_result = benchmark_rope("AVX512", rope_q16_avx512, LARGE_HEAD_DIM, total_heads, BENCHMARK_ITERATIONS / 10);
    print_result(avx512_result, baseline_ms);
#endif

    auto dispatch_result = benchmark_rope("Auto-Dispatch", rope_q16_dispatch, LARGE_HEAD_DIM, total_heads, BENCHMARK_ITERATIONS / 10);
    print_result(dispatch_result, baseline_ms);

    print_footer();

    // For large data, SIMD should show speedup but integer scalar is now fast
#if defined(__AVX512F__)
    EXPECT_LT(avx512_result.elapsed_ms, scalar_result.elapsed_ms * 0.85)
        << "AVX512 should be at least 1.2x faster than scalar for large data";
#endif
}

/**
 * @test Throughput comparison summary
 */
TEST_F(Q16_to_Q16_RoPE_FixedScale_Perf, ThroughputSummary)
{
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                Q16→Q16 Fixed-Scale RoPE: Throughput Summary                      ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  Scenario               │  Scalar M-heads/s  │   AVX2   │  AVX512  │ Best Speedup ║" << std::endl;
    std::cout << "├─────────────────────────┼────────────────────┼──────────┼──────────┼──────────────┤" << std::endl;

    struct ScenarioResult
    {
        std::string name;
        double scalar_rate;
        double avx2_rate;
        double avx512_rate;
    };
    std::vector<ScenarioResult> results;

    // Test scenarios
    auto test_scenario = [&](const std::string &name, int head_dim, int n_heads, size_t iters)
    {
        ScenarioResult r;
        r.name = name;
        r.scalar_rate = benchmark_rope("", rope_q16_scalar, head_dim, n_heads, iters).heads_per_sec / 1e6;
#if defined(__AVX2__)
        r.avx2_rate = benchmark_rope("", rope_q16_avx2, head_dim, n_heads, iters).heads_per_sec / 1e6;
#else
        r.avx2_rate = 0;
#endif
#if defined(__AVX512F__)
        r.avx512_rate = benchmark_rope("", rope_q16_avx512, head_dim, n_heads, iters).heads_per_sec / 1e6;
#else
        r.avx512_rate = 0;
#endif
        results.push_back(r);
    };

    test_scenario("Decode 1-head", HEAD_DIM, 1, 5000);
    test_scenario("Decode 2-head", HEAD_DIM, 2, 5000);
    test_scenario("Decode 8-head", LARGE_HEAD_DIM, 8, 2000);
    test_scenario("Prefill seq=128", HEAD_DIM, 128 * 2, 500);
    test_scenario("Prefill seq=512", LARGE_HEAD_DIM, 512 * 8, 50);

    for (const auto &r : results)
    {
        double best_rate = std::max({r.scalar_rate, r.avx2_rate, r.avx512_rate});
        double speedup = best_rate / r.scalar_rate;

        std::cout << "║  " << std::setw(22) << std::left << r.name
                  << " │ " << std::setw(18) << std::right << std::fixed << std::setprecision(2) << r.scalar_rate
                  << " │ " << std::setw(8) << std::right << std::fixed << std::setprecision(2) << r.avx2_rate
                  << " │ " << std::setw(8) << std::right << std::fixed << std::setprecision(2) << r.avx512_rate
                  << " │ " << std::setw(10) << std::right << std::fixed << std::setprecision(2) << speedup << "x"
                  << "  ║" << std::endl;
    }

    std::cout << "╚══════════════════════════════════════════════════════════════════════════════════╝" << std::endl;
}
