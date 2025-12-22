/**
 * @file Perf__CPURoPEKernel.cpp
 * @brief Performance benchmarks for CPURoPEKernelT
 * @author David Sanftenberg
 *
 * Measures throughput (tokens/sec, GB/s) for RoPE operations across:
 *   - Typed:  CPURoPEKernelT<FP32>::apply_typed()
 *   - Typed:  CPURoPEKernelT<BF16>::apply_typed()
 *   - Typed:  CPURoPEKernelT<FP16>::apply_typed()
 *   - Typed:  CPURoPEKernelT<Q8_1>::apply_typed()
 *
 * Test scenarios:
 *   - Single token (decode mode)
 *   - Small batch (prefill, seq_len=32)
 *   - Medium batch (prefill, seq_len=128)
 *   - Large batch (prefill, seq_len=512)
 *   - Long context (prefill, seq_len=2048)
 *
 * Expected results:
 *   1. BF16/FP16 should be faster than FP32 (memory bandwidth savings)
 *   2. Q8_1 should be fastest, especially at larger sequence lengths
 */

#include <gtest/gtest.h>
#include <mpi.h>

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "kernels/cpu/ops/CPURoPEKernelT.h"
#include "tensors/Tensors.h"
#include "tensors/SIMDHelpers.h"
#include "tensors/BlockStructures.h"
#include "execution/RuntimeConfig.h"

using namespace llaminar2;

// ============================================================================
// Performance Test Fixture
// ============================================================================

class CPURoPEKernel_Perf : public ::testing::Test
{
protected:
    // Benchmark configuration
    static constexpr size_t WARMUP_ITERATIONS = 10;
    static constexpr size_t BENCHMARK_ITERATIONS = 100;

    // Qwen2.5 model parameters
    static constexpr int N_HEADS = 14;              // Qwen2.5-0.5B query heads
    static constexpr int N_KV_HEADS = 2;            // Qwen2.5-0.5B KV heads
    static constexpr int HEAD_DIM = 64;             // Qwen2.5 head dimension
    static constexpr float ROPE_THETA = 1000000.0f; // Qwen2.5 RoPE theta

    // Larger model parameters for scaling tests
    static constexpr int LARGE_N_HEADS = 32;   // Qwen2.5-7B query heads
    static constexpr int LARGE_N_KV_HEADS = 8; // Qwen2.5-7B KV heads
    static constexpr int LARGE_HEAD_DIM = 128; // Qwen2.5-7B head dimension

    std::mt19937 rng_{42};
    int rank_ = 0;

    void SetUp() override
    {
        rng_.seed(42);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    }

    // =========================================================================
    // Data Generation Helpers
    // =========================================================================

    std::vector<float> generate_random_fp32(size_t count)
    {
        std::vector<float> data(count);
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        for (auto &v : data)
        {
            v = dist(rng_);
        }
        return data;
    }

    std::vector<int> generate_position_ids(int seq_len, int start_pos = 0)
    {
        std::vector<int> pos(seq_len);
        std::iota(pos.begin(), pos.end(), start_pos);
        return pos;
    }

    // =========================================================================
    // Benchmark Result Structures
    // =========================================================================

    struct BenchmarkResult
    {
        std::string name;
        double elapsed_ms;
        double tokens_per_sec;
        double bandwidth_gbps; // GB/s of Q+K tensor data
        size_t iterations;
        int seq_len;
    };

    // =========================================================================
    // Output Formatting
    // =========================================================================

    void print_header(const std::string &test_name, int seq_len, int n_heads, int n_kv_heads, int head_dim)
    {
        if (rank_ != 0)
            return;

        size_t q_size = static_cast<size_t>(seq_len) * n_heads * head_dim;
        size_t k_size = static_cast<size_t>(seq_len) * n_kv_heads * head_dim;
        double fp32_mb = (q_size + k_size) * sizeof(float) / (1024.0 * 1024.0);

        std::cout << "\n╔══════════════════════════════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║ " << std::setw(86) << std::left << test_name << "║" << std::endl;
        std::cout << "╠══════════════════════════════════════════════════════════════════════════════════════╣" << std::endl;
        std::cout << "║ seq_len=" << std::setw(5) << seq_len
                  << "  n_heads=" << std::setw(3) << n_heads
                  << "  n_kv_heads=" << std::setw(2) << n_kv_heads
                  << "  head_dim=" << std::setw(3) << head_dim
                  << "  FP32 Q+K=" << std::fixed << std::setprecision(2) << std::setw(7) << fp32_mb << " MB"
                  << "      ║" << std::endl;
        std::cout << "╠══════════════════════════════════════════════════════════════════════════════════════╣" << std::endl;
        std::cout << "│   Implementation    │  Time (ms)  │  Tokens/sec  │  Bandwidth GB/s  │    Speedup    │" << std::endl;
        std::cout << "├─────────────────────┼─────────────┼──────────────┼──────────────────┼───────────────┤" << std::endl;
    }

    void print_result(const BenchmarkResult &result, double baseline_ms = 0.0)
    {
        if (rank_ != 0)
            return;

        double speedup = (baseline_ms > 0) ? (baseline_ms / result.elapsed_ms) : 1.0;
        std::string speedup_str;
        if (baseline_ms > 0)
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << speedup << "x";
            if (speedup >= 1.0)
                oss << " ✓";
            speedup_str = oss.str();
        }
        else
        {
            speedup_str = "(baseline)";
        }

        std::cout << "│ " << std::setw(19) << std::left << result.name
                  << " │ " << std::setw(11) << std::right << std::fixed << std::setprecision(3) << result.elapsed_ms
                  << " │ " << std::setw(12) << std::right << std::fixed << std::setprecision(0) << result.tokens_per_sec
                  << " │ " << std::setw(16) << std::right << std::fixed << std::setprecision(2) << result.bandwidth_gbps
                  << " │ " << std::setw(13) << std::right << speedup_str
                  << " │" << std::endl;
    }

    void print_footer()
    {
        if (rank_ != 0)
            return;
        std::cout << "╚══════════════════════════════════════════════════════════════════════════════════════╝" << std::endl;
    }

    // =========================================================================
    // Benchmark Functions
    // =========================================================================

    BenchmarkResult benchmark_fp32_typed(
        int seq_len, int n_heads, int n_kv_heads, int head_dim)
    {
        size_t q_size = static_cast<size_t>(seq_len) * n_heads * head_dim;
        size_t k_size = static_cast<size_t>(seq_len) * n_kv_heads * head_dim;

        auto q_data = generate_random_fp32(q_size);
        auto k_data = generate_random_fp32(k_size);
        auto position_ids = generate_position_ids(seq_len);

        CPURoPEKernelT<ActivationPrecision::FP32> kernel;

        // Warmup
        for (size_t w = 0; w < WARMUP_ITERATIONS; ++w)
        {
            kernel.apply_typed(
                q_data.data(), k_data.data(),
                position_ids.data(),
                seq_len, n_heads, n_kv_heads, head_dim,
                ROPE_THETA);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < BENCHMARK_ITERATIONS; ++iter)
        {
            kernel.apply_typed(
                q_data.data(), k_data.data(),
                position_ids.data(),
                seq_len, n_heads, n_kv_heads, head_dim,
                ROPE_THETA);
        }
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double total_tokens = static_cast<double>(seq_len) * BENCHMARK_ITERATIONS;
        double tokens_per_sec = total_tokens / (elapsed_ms / 1000.0);
        double total_bytes = static_cast<double>(q_size + k_size) * sizeof(float) * BENCHMARK_ITERATIONS;
        double bandwidth_gbps = total_bytes / (1024.0 * 1024.0 * 1024.0) / (elapsed_ms / 1000.0);

        return {"FP32 Typed", elapsed_ms, tokens_per_sec, bandwidth_gbps, BENCHMARK_ITERATIONS, seq_len};
    }

    BenchmarkResult benchmark_bf16_typed(
        int seq_len, int n_heads, int n_kv_heads, int head_dim)
    {
        size_t q_size = static_cast<size_t>(seq_len) * n_heads * head_dim;
        size_t k_size = static_cast<size_t>(seq_len) * n_kv_heads * head_dim;

        // Generate FP32 and convert to BF16
        auto fp32_q = generate_random_fp32(q_size);
        auto fp32_k = generate_random_fp32(k_size);

        std::vector<uint16_t> bf16_q(q_size);
        std::vector<uint16_t> bf16_k(k_size);
        simd::convert_fp32_to_bf16(fp32_q.data(), bf16_q.data(), q_size);
        simd::convert_fp32_to_bf16(fp32_k.data(), bf16_k.data(), k_size);

        auto position_ids = generate_position_ids(seq_len);

        CPURoPEKernelT<ActivationPrecision::BF16> kernel;

        // Warmup
        for (size_t w = 0; w < WARMUP_ITERATIONS; ++w)
        {
            kernel.apply_typed(
                bf16_q.data(), bf16_k.data(),
                position_ids.data(),
                seq_len, n_heads, n_kv_heads, head_dim,
                ROPE_THETA);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < BENCHMARK_ITERATIONS; ++iter)
        {
            kernel.apply_typed(
                bf16_q.data(), bf16_k.data(),
                position_ids.data(),
                seq_len, n_heads, n_kv_heads, head_dim,
                ROPE_THETA);
        }
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double total_tokens = static_cast<double>(seq_len) * BENCHMARK_ITERATIONS;
        double tokens_per_sec = total_tokens / (elapsed_ms / 1000.0);
        // BF16 uses 2 bytes per element
        double total_bytes = static_cast<double>(q_size + k_size) * sizeof(uint16_t) * BENCHMARK_ITERATIONS;
        double bandwidth_gbps = total_bytes / (1024.0 * 1024.0 * 1024.0) / (elapsed_ms / 1000.0);

        return {"BF16 Typed", elapsed_ms, tokens_per_sec, bandwidth_gbps, BENCHMARK_ITERATIONS, seq_len};
    }

    BenchmarkResult benchmark_fp16_typed(
        int seq_len, int n_heads, int n_kv_heads, int head_dim)
    {
        size_t q_size = static_cast<size_t>(seq_len) * n_heads * head_dim;
        size_t k_size = static_cast<size_t>(seq_len) * n_kv_heads * head_dim;

        // Generate FP32 and convert to FP16
        auto fp32_q = generate_random_fp32(q_size);
        auto fp32_k = generate_random_fp32(k_size);

        std::vector<uint16_t> fp16_q(q_size);
        std::vector<uint16_t> fp16_k(k_size);
        simd::convert_fp32_to_fp16(fp32_q.data(), fp16_q.data(), q_size);
        simd::convert_fp32_to_fp16(fp32_k.data(), fp16_k.data(), k_size);

        auto position_ids = generate_position_ids(seq_len);

        CPURoPEKernelT<ActivationPrecision::FP16> kernel;

        // Warmup
        for (size_t w = 0; w < WARMUP_ITERATIONS; ++w)
        {
            kernel.apply_typed(
                fp16_q.data(), fp16_k.data(),
                position_ids.data(),
                seq_len, n_heads, n_kv_heads, head_dim,
                ROPE_THETA);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < BENCHMARK_ITERATIONS; ++iter)
        {
            kernel.apply_typed(
                fp16_q.data(), fp16_k.data(),
                position_ids.data(),
                seq_len, n_heads, n_kv_heads, head_dim,
                ROPE_THETA);
        }
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double total_tokens = static_cast<double>(seq_len) * BENCHMARK_ITERATIONS;
        double tokens_per_sec = total_tokens / (elapsed_ms / 1000.0);
        // FP16 uses 2 bytes per element
        double total_bytes = static_cast<double>(q_size + k_size) * sizeof(uint16_t) * BENCHMARK_ITERATIONS;
        double bandwidth_gbps = total_bytes / (1024.0 * 1024.0 * 1024.0) / (elapsed_ms / 1000.0);

        return {"FP16 Typed", elapsed_ms, tokens_per_sec, bandwidth_gbps, BENCHMARK_ITERATIONS, seq_len};
    }

    BenchmarkResult benchmark_q8_1_typed(
        int seq_len, int n_heads, int n_kv_heads, int head_dim)
    {
        // Q8_1 requires head_dim to be multiple of 32
        if (head_dim % 32 != 0)
        {
            return {"Q8_1 Typed", 0.0, 0.0, 0.0, 0, seq_len};
        }

        size_t q_size = static_cast<size_t>(seq_len) * n_heads * head_dim;
        size_t k_size = static_cast<size_t>(seq_len) * n_kv_heads * head_dim;
        size_t q_blocks = q_size / 32;
        size_t k_blocks = k_size / 32;

        // Generate FP32 and quantize to Q8_1
        auto fp32_q = generate_random_fp32(q_size);
        auto fp32_k = generate_random_fp32(k_size);

        std::vector<Q8_1Block> q8_q(q_blocks);
        std::vector<Q8_1Block> q8_k(k_blocks);
        simd::quantize_fp32_to_q8_1_blocks(fp32_q.data(), q8_q.data(), q_size);
        simd::quantize_fp32_to_q8_1_blocks(fp32_k.data(), q8_k.data(), k_size);

        auto position_ids = generate_position_ids(seq_len);

        CPURoPEKernelT<ActivationPrecision::Q8_1> kernel;

        // Warmup
        for (size_t w = 0; w < WARMUP_ITERATIONS; ++w)
        {
            kernel.apply_typed(
                q8_q.data(), q8_k.data(),
                position_ids.data(),
                seq_len, n_heads, n_kv_heads, head_dim,
                ROPE_THETA);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < BENCHMARK_ITERATIONS; ++iter)
        {
            kernel.apply_typed(
                q8_q.data(), q8_k.data(),
                position_ids.data(),
                seq_len, n_heads, n_kv_heads, head_dim,
                ROPE_THETA);
        }
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double total_tokens = static_cast<double>(seq_len) * BENCHMARK_ITERATIONS;
        double tokens_per_sec = total_tokens / (elapsed_ms / 1000.0);
        // Q8_1 block: 32 int8 + fp16 scale + int16 sum = 36 bytes per 32 elements = 1.125 bytes/element
        double total_bytes = static_cast<double>(q_blocks + k_blocks) * sizeof(Q8_1Block) * BENCHMARK_ITERATIONS;
        double bandwidth_gbps = total_bytes / (1024.0 * 1024.0 * 1024.0) / (elapsed_ms / 1000.0);

        return {"Q8_1 Typed", elapsed_ms, tokens_per_sec, bandwidth_gbps, BENCHMARK_ITERATIONS, seq_len};
    }

    // =========================================================================
    // Run All Benchmarks for a Given Configuration
    // =========================================================================

    void run_benchmark_suite(
        const std::string &test_name,
        int seq_len, int n_heads, int n_kv_heads, int head_dim)
    {
        print_header(test_name, seq_len, n_heads, n_kv_heads, head_dim);

        // Run benchmarks (FP32 Typed is now the baseline)
        auto fp32_typed = benchmark_fp32_typed(seq_len, n_heads, n_kv_heads, head_dim);
        auto bf16_typed = benchmark_bf16_typed(seq_len, n_heads, n_kv_heads, head_dim);
        auto fp16_typed = benchmark_fp16_typed(seq_len, n_heads, n_kv_heads, head_dim);
        auto q8_1_typed = benchmark_q8_1_typed(seq_len, n_heads, n_kv_heads, head_dim);

        // Print results with speedup relative to FP32 Typed (baseline)
        print_result(fp32_typed, 0.0); // baseline
        print_result(bf16_typed, fp32_typed.elapsed_ms);
        print_result(fp16_typed, fp32_typed.elapsed_ms);
        if (q8_1_typed.elapsed_ms > 0)
        {
            print_result(q8_1_typed, fp32_typed.elapsed_ms);
        }
        else
        {
            if (rank_ == 0)
            {
                std::cout << "│ " << std::setw(19) << std::left << "Q8_1 Typed"
                          << " │   (head_dim must be multiple of 32)                               │" << std::endl;
            }
        }

        print_footer();
    }
};

// ============================================================================
// Test Cases: Small Model (Qwen2.5-0.5B dimensions)
// ============================================================================

TEST_F(CPURoPEKernel_Perf, SmallModel_SingleToken)
{
    // Decode mode: single token
    run_benchmark_suite(
        "RoPE Performance: Single Token (Decode Mode) - Qwen2.5-0.5B",
        1, N_HEADS, N_KV_HEADS, HEAD_DIM);
}

TEST_F(CPURoPEKernel_Perf, SmallModel_SmallBatch)
{
    // Small prefill batch
    run_benchmark_suite(
        "RoPE Performance: seq_len=32 (Small Prefill) - Qwen2.5-0.5B",
        32, N_HEADS, N_KV_HEADS, HEAD_DIM);
}

TEST_F(CPURoPEKernel_Perf, SmallModel_MediumBatch)
{
    // Medium prefill batch
    run_benchmark_suite(
        "RoPE Performance: seq_len=128 (Medium Prefill) - Qwen2.5-0.5B",
        128, N_HEADS, N_KV_HEADS, HEAD_DIM);
}

TEST_F(CPURoPEKernel_Perf, SmallModel_LargeBatch)
{
    // Large prefill batch
    run_benchmark_suite(
        "RoPE Performance: seq_len=512 (Large Prefill) - Qwen2.5-0.5B",
        512, N_HEADS, N_KV_HEADS, HEAD_DIM);
}

TEST_F(CPURoPEKernel_Perf, SmallModel_LongContext)
{
    // Long context (2K tokens)
    run_benchmark_suite(
        "RoPE Performance: seq_len=2048 (Long Context) - Qwen2.5-0.5B",
        2048, N_HEADS, N_KV_HEADS, HEAD_DIM);
}

// ============================================================================
// Test Cases: Large Model (Qwen2.5-7B dimensions)
// ============================================================================

TEST_F(CPURoPEKernel_Perf, LargeModel_SingleToken)
{
    // Decode mode with larger model
    run_benchmark_suite(
        "RoPE Performance: Single Token (Decode Mode) - Qwen2.5-7B dims",
        1, LARGE_N_HEADS, LARGE_N_KV_HEADS, LARGE_HEAD_DIM);
}

TEST_F(CPURoPEKernel_Perf, LargeModel_MediumBatch)
{
    // Medium prefill with larger model
    run_benchmark_suite(
        "RoPE Performance: seq_len=128 (Medium Prefill) - Qwen2.5-7B dims",
        128, LARGE_N_HEADS, LARGE_N_KV_HEADS, LARGE_HEAD_DIM);
}

TEST_F(CPURoPEKernel_Perf, LargeModel_LargeBatch)
{
    // Large prefill with larger model
    run_benchmark_suite(
        "RoPE Performance: seq_len=512 (Large Prefill) - Qwen2.5-7B dims",
        512, LARGE_N_HEADS, LARGE_N_KV_HEADS, LARGE_HEAD_DIM);
}

TEST_F(CPURoPEKernel_Perf, LargeModel_LongContext)
{
    // Long context with larger model
    run_benchmark_suite(
        "RoPE Performance: seq_len=2048 (Long Context) - Qwen2.5-7B dims",
        2048, LARGE_N_HEADS, LARGE_N_KV_HEADS, LARGE_HEAD_DIM);
}

// ============================================================================
// Scaling Analysis
// ============================================================================

TEST_F(CPURoPEKernel_Perf, ScalingAnalysis)
{
    if (rank_ == 0)
    {
        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║                               SCALING ANALYSIS: Speedup vs FP32 Typed                                ║" << std::endl;
        std::cout << "╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣" << std::endl;
        std::cout << "║  seq_len  │  FP32 Typed  │  BF16 Typed  │  FP16 Typed  │  Q8_1 Typed  │  Best Speedup                 ║" << std::endl;
        std::cout << "├───────────┼──────────────┼──────────────┼──────────────┼──────────────┼───────────────────────────────┤" << std::endl;
    }

    std::vector<int> seq_lengths = {1, 8, 32, 128, 512, 1024, 2048};

    for (int seq_len : seq_lengths)
    {
        auto fp32_typed = benchmark_fp32_typed(seq_len, N_HEADS, N_KV_HEADS, HEAD_DIM);
        auto bf16_typed = benchmark_bf16_typed(seq_len, N_HEADS, N_KV_HEADS, HEAD_DIM);
        auto fp16_typed = benchmark_fp16_typed(seq_len, N_HEADS, N_KV_HEADS, HEAD_DIM);
        auto q8_1_typed = benchmark_q8_1_typed(seq_len, N_HEADS, N_KV_HEADS, HEAD_DIM);

        double fp32_speedup = 1.0; // baseline
        double bf16_speedup = fp32_typed.elapsed_ms / bf16_typed.elapsed_ms;
        double fp16_speedup = fp32_typed.elapsed_ms / fp16_typed.elapsed_ms;
        double q8_1_speedup = (q8_1_typed.elapsed_ms > 0)
                                  ? fp32_typed.elapsed_ms / q8_1_typed.elapsed_ms
                                  : 0.0;

        // Find best
        std::string best_name = "FP32 Typed";
        double best_speedup = fp32_speedup;
        if (bf16_speedup > best_speedup)
        {
            best_speedup = bf16_speedup;
            best_name = "BF16 Typed";
        }
        if (fp16_speedup > best_speedup)
        {
            best_speedup = fp16_speedup;
            best_name = "FP16 Typed";
        }
        if (q8_1_speedup > best_speedup)
        {
            best_speedup = q8_1_speedup;
            best_name = "Q8_1 Typed";
        }

        if (rank_ == 0)
        {
            std::ostringstream best_str;
            best_str << best_name << " (" << std::fixed << std::setprecision(2) << best_speedup << "x)";

            std::cout << "║ " << std::setw(9) << std::right << seq_len
                      << " │ " << std::setw(10) << std::right << std::fixed << std::setprecision(2) << fp32_speedup << "x "
                      << " │ " << std::setw(10) << std::right << std::fixed << std::setprecision(2) << bf16_speedup << "x "
                      << " │ " << std::setw(10) << std::right << std::fixed << std::setprecision(2) << fp16_speedup << "x "
                      << " │ " << std::setw(10) << std::right;

            if (q8_1_speedup > 0)
            {
                std::cout << std::fixed << std::setprecision(2) << q8_1_speedup << "x ";
            }
            else
            {
                std::cout << "N/A  ";
            }

            std::cout << " │ " << std::setw(29) << std::left << best_str.str()
                      << " ║" << std::endl;
        }
    }

    if (rank_ == 0)
    {
        std::cout << "╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝" << std::endl;
        std::cout << "\nExpected patterns:" << std::endl;
        std::cout << "  • FP32 Typed is baseline (1.0x)" << std::endl;
        std::cout << "  • BF16/FP16 should be faster than FP32 due to memory bandwidth (>1.0x)" << std::endl;
        std::cout << "  • Q8_1 should show best scaling at larger seq_len due to integer ops" << std::endl;
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
