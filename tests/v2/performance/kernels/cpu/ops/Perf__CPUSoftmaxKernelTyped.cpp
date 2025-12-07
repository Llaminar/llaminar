/**
 * @file Perf__CPUSoftmaxKernelTyped.cpp
 * @brief Performance benchmark for CPUSoftmaxKernelTyped (typed softmax kernels)
 *
 * This test benchmarks the typed Softmax kernel performance comparing:
 * 1. CPUSoftmaxKernelTyped<FP32> (baseline)
 * 2. CPUSoftmaxKernelTyped<BF16> (2x memory compression)
 * 3. CPUSoftmaxKernelTyped<FP16> (2x memory compression)
 * 4. CPUSoftmaxKernelTyped<Q8_1> (3.5x compression, integer-aware softmax)
 *
 * Softmax computational complexity:
 *   - Pass 1: Find max (N comparisons per row)
 *   - Pass 2: Compute exp and sum (N exp + N adds per row)
 *   - Pass 3: Normalize (N divides per row)
 *   - Total: ~3N operations per row (dominated by exp())
 *
 * For M rows x N columns:
 *   - FLOPs estimate: 3 * M * N (simplified)
 *   - Memory: M * N * bytes_per_element (read + write)
 *
 * The Q8_1 kernel uses integer-aware operations:
 *   - Integer max-finding within blocks
 *   - Batched scale operations
 *   - Direct requantization
 *
 * @author David Sanftenberg
 * @date 2025-12-07
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>

// V2 includes
#include "tensors/Tensors.h"
#include "tensors/BlockStructures.h"
#include "kernels/cpu/ops/CPUSoftmaxKernelTyped.h"
#include "tensors/SIMDHelpers.h"
#include "utils/Logger.h"

using namespace llaminar2;

// ============================================================================
// Benchmark Configuration
// ============================================================================

struct BenchmarkConfig
{
    int rows;         ///< Number of rows (e.g., num_heads or batch * num_heads)
    int cols;         ///< Number of columns (e.g., seq_len for attention)
    int warmup_iters; ///< Number of warmup iterations
    int bench_iters;  ///< Number of timed benchmark iterations
    bool use_causal;  ///< Whether to use causal masking
    float scale;      ///< Scale factor (typically 1/sqrt(d_k))
    std::string description;
};

struct BenchmarkStats
{
    double mean_ms;          ///< Mean time per iteration (ms)
    double stddev_ms;        ///< Standard deviation (ms)
    double min_ms;           ///< Minimum time (ms)
    double max_ms;           ///< Maximum time (ms)
    double bandwidth_gbps;   ///< Memory bandwidth (GB/s)
    double gflops;           ///< Throughput (GFLOPS)
    double elements_per_sec; ///< Elements processed per second
};

// ============================================================================
// Test Fixture
// ============================================================================

class CPUSoftmaxKernelTyped_Perf : public ::testing::Test
{
protected:
    int rank_ = 0;
    int world_size_ = 1;
    std::mt19937 rng_{42};

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

        if (rank_ == 0)
        {
            std::cout << "\n[Performance Test] CPUSoftmaxKernelTyped Benchmark" << std::endl;
            std::cout << "[Performance Test] OpenMP threads: " << omp_get_max_threads() << std::endl;
        }
    }

    /**
     * @brief Calculate statistics from timing samples
     *
     * @param times_ms Vector of timing samples in milliseconds
     * @param rows Number of rows processed
     * @param cols Number of columns per row
     * @param bytes_per_element Memory footprint per element
     * @return BenchmarkStats with computed metrics
     */
    BenchmarkStats calculate_stats(
        const std::vector<double> &times_ms,
        int rows,
        int cols,
        double bytes_per_element)
    {
        double sum = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
        double mean = sum / times_ms.size();
        double sq_sum = std::inner_product(times_ms.begin(), times_ms.end(), times_ms.begin(), 0.0);
        double stddev = std::sqrt(sq_sum / times_ms.size() - mean * mean);
        double min_val = *std::min_element(times_ms.begin(), times_ms.end());
        double max_val = *std::max_element(times_ms.begin(), times_ms.end());

        // Memory bandwidth: read input + write output
        size_t total_elements = static_cast<size_t>(rows) * cols;
        double total_bytes = total_elements * bytes_per_element * 2.0; // read + write (in-place)
        double bandwidth_gbps = (total_bytes / (mean * 1e-3)) / 1e9;

        // FLOPS estimate for softmax:
        // - Find max: N comparisons per row
        // - Exp + sum: N exp operations + N additions per row
        // - Normalize: N divisions per row
        // Simplified: ~4N operations per row (exp dominates)
        double flops = static_cast<double>(rows) * cols * 4.0;
        double gflops = (flops / (mean * 1e-3)) / 1e9;

        // Elements per second
        double elements_per_sec = total_elements / (mean * 1e-3);

        return {mean, stddev, min_val, max_val, bandwidth_gbps, gflops, elements_per_sec};
    }

    void print_header(const BenchmarkConfig &config)
    {
        if (rank_ == 0)
        {
            std::cout << "\n════════════════════════════════════════════════════════════════" << std::endl;
            std::cout << "Benchmark: " << config.description << std::endl;
            std::cout << "  Rows: " << config.rows << ", Cols: " << config.cols
                      << " (" << (static_cast<size_t>(config.rows) * config.cols / 1e6) << "M elements)" << std::endl;
            std::cout << "  Causal: " << (config.use_causal ? "Yes" : "No")
                      << ", Scale: " << config.scale << std::endl;
            std::cout << "  Warmup: " << config.warmup_iters << ", Iterations: " << config.bench_iters << std::endl;
            std::cout << "════════════════════════════════════════════════════════════════" << std::endl;
        }
    }

    void print_result(const std::string &kernel_name, const BenchmarkStats &stats)
    {
        if (rank_ == 0)
        {
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "  " << std::left << std::setw(20) << kernel_name
                      << " Time: " << std::setw(8) << stats.mean_ms << " ms"
                      << "  ±" << std::setw(6) << stats.stddev_ms << " ms"
                      << "  BW: " << std::setw(7) << stats.bandwidth_gbps << " GB/s"
                      << "  GFLOPS: " << std::setw(7) << stats.gflops
                      << std::endl;
        }
    }

    void print_comparison(const std::string &kernel_name, const BenchmarkStats &stats,
                          const BenchmarkStats &baseline)
    {
        if (rank_ == 0)
        {
            double speedup = baseline.mean_ms / stats.mean_ms;
            std::string speedup_str;
            if (speedup >= 1.0)
            {
                speedup_str = std::to_string(speedup).substr(0, 4) + "x faster";
            }
            else
            {
                speedup_str = std::to_string(1.0 / speedup).substr(0, 4) + "x slower";
            }

            std::cout << std::fixed << std::setprecision(3);
            std::cout << "  " << std::left << std::setw(20) << kernel_name
                      << " Time: " << std::setw(8) << stats.mean_ms << " ms"
                      << "  BW: " << std::setw(7) << stats.bandwidth_gbps << " GB/s"
                      << "  GFLOPS: " << std::setw(7) << stats.gflops
                      << "  vs FP32: " << speedup_str
                      << std::endl;
        }
    }

    // ========================================================================
    // Benchmark Functions for Each Kernel Type
    // ========================================================================

    /**
     * @brief Benchmark FP32 softmax (baseline)
     */
    BenchmarkStats bench_fp32(const BenchmarkConfig &config)
    {
        CPUSoftmaxKernelTyped<ActivationPrecision::FP32> kernel;

        size_t size = static_cast<size_t>(config.rows) * config.cols;
        std::vector<float> data(size);

        // Initialize with random data in reasonable range for softmax
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        for (auto &v : data)
            v = dist(rng_);

        // Warmup
        for (int i = 0; i < config.warmup_iters; ++i)
        {
            // Re-initialize data for each warmup (softmax is in-place)
            for (auto &v : data)
                v = dist(rng_);
            kernel.apply_typed(data.data(), config.rows, config.cols,
                               config.use_causal, config.scale);
        }

        // Benchmark
        std::vector<double> times_ms;
        times_ms.reserve(config.bench_iters);

        for (int i = 0; i < config.bench_iters; ++i)
        {
            // Re-initialize data (softmax is destructive)
            for (auto &v : data)
                v = dist(rng_);

            MPI_Barrier(MPI_COMM_WORLD);
            auto start = std::chrono::high_resolution_clock::now();

            kernel.apply_typed(data.data(), config.rows, config.cols,
                               config.use_causal, config.scale);

            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e6;
            times_ms.push_back(ms);
        }

        return calculate_stats(times_ms, config.rows, config.cols, sizeof(float));
    }

    /**
     * @brief Benchmark BF16 softmax (2x compression)
     */
    BenchmarkStats bench_bf16(const BenchmarkConfig &config)
    {
        CPUSoftmaxKernelTyped<ActivationPrecision::BF16> kernel;

        size_t size = static_cast<size_t>(config.rows) * config.cols;
        std::vector<uint16_t> data(size);
        std::vector<float> fp32_temp(size);

        // Initialize with random data
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

        auto reinit = [&]()
        {
            for (auto &v : fp32_temp)
                v = dist(rng_);
            simd::convert_fp32_to_bf16(fp32_temp.data(), data.data(), size);
        };

        // Warmup
        for (int i = 0; i < config.warmup_iters; ++i)
        {
            reinit();
            kernel.apply_typed(data.data(), config.rows, config.cols,
                               config.use_causal, config.scale);
        }

        // Benchmark
        std::vector<double> times_ms;
        times_ms.reserve(config.bench_iters);

        for (int i = 0; i < config.bench_iters; ++i)
        {
            reinit();

            MPI_Barrier(MPI_COMM_WORLD);
            auto start = std::chrono::high_resolution_clock::now();

            kernel.apply_typed(data.data(), config.rows, config.cols,
                               config.use_causal, config.scale);

            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e6;
            times_ms.push_back(ms);
        }

        return calculate_stats(times_ms, config.rows, config.cols, sizeof(uint16_t));
    }

    /**
     * @brief Benchmark FP16 softmax (2x compression)
     */
    BenchmarkStats bench_fp16(const BenchmarkConfig &config)
    {
        CPUSoftmaxKernelTyped<ActivationPrecision::FP16> kernel;

        size_t size = static_cast<size_t>(config.rows) * config.cols;
        std::vector<uint16_t> data(size);
        std::vector<float> fp32_temp(size);

        // Initialize with random data
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

        auto reinit = [&]()
        {
            for (auto &v : fp32_temp)
                v = dist(rng_);
            simd::convert_fp32_to_fp16(fp32_temp.data(), data.data(), size);
        };

        // Warmup
        for (int i = 0; i < config.warmup_iters; ++i)
        {
            reinit();
            kernel.apply_typed(data.data(), config.rows, config.cols,
                               config.use_causal, config.scale);
        }

        // Benchmark
        std::vector<double> times_ms;
        times_ms.reserve(config.bench_iters);

        for (int i = 0; i < config.bench_iters; ++i)
        {
            reinit();

            MPI_Barrier(MPI_COMM_WORLD);
            auto start = std::chrono::high_resolution_clock::now();

            kernel.apply_typed(data.data(), config.rows, config.cols,
                               config.use_causal, config.scale);

            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e6;
            times_ms.push_back(ms);
        }

        return calculate_stats(times_ms, config.rows, config.cols, sizeof(uint16_t));
    }

    /**
     * @brief Benchmark Q8_1 softmax (integer-aware, 3.5x compression)
     */
    BenchmarkStats bench_q8_1(const BenchmarkConfig &config)
    {
        CPUSoftmaxKernelTyped<ActivationPrecision::Q8_1> kernel;

        // Q8_1: 36 bytes per block of 32 elements
        // Number of blocks per row = ceil(cols / 32)
        size_t blocks_per_row = (config.cols + 31) / 32;
        size_t total_blocks = static_cast<size_t>(config.rows) * blocks_per_row;

        std::vector<Q8_1Block> data(total_blocks);

        // Initialize: generate FP32 random data, convert to Q8_1
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        size_t fp32_size = static_cast<size_t>(config.rows) * config.cols;
        std::vector<float> fp32_temp(fp32_size);

        auto reinit = [&]()
        {
            for (auto &v : fp32_temp)
                v = dist(rng_);
            simd::quantize_fp32_to_q8_1_blocks(fp32_temp.data(), data.data(), fp32_size);
        };

        // Warmup
        for (int i = 0; i < config.warmup_iters; ++i)
        {
            reinit();
            kernel.apply_typed(data.data(), config.rows, static_cast<int>(blocks_per_row),
                               config.use_causal, config.scale);
        }

        // Benchmark
        std::vector<double> times_ms;
        times_ms.reserve(config.bench_iters);

        for (int i = 0; i < config.bench_iters; ++i)
        {
            reinit();

            MPI_Barrier(MPI_COMM_WORLD);
            auto start = std::chrono::high_resolution_clock::now();

            kernel.apply_typed(data.data(), config.rows, static_cast<int>(blocks_per_row),
                               config.use_causal, config.scale);

            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e6;
            times_ms.push_back(ms);
        }

        // Q8_1: 36 bytes / 32 elements = 1.125 bytes per element
        double bytes_per_element = 36.0 / 32.0;
        return calculate_stats(times_ms, config.rows, config.cols, bytes_per_element);
    }

    // ========================================================================
    // Run Full Comparison Benchmark
    // ========================================================================

    void run_comparison_benchmark(const BenchmarkConfig &config)
    {
        print_header(config);

        // Run Q8_1 first to avoid cache pollution from FP32 benchmarks
        auto q8_1_stats = bench_q8_1(config);

        // Baseline: FP32 softmax
        auto fp32_stats = bench_fp32(config);
        print_result("FP32 (baseline)", fp32_stats);

        // BF16: 2x compression
        auto bf16_stats = bench_bf16(config);
        print_comparison("BF16 (2x)", bf16_stats, fp32_stats);

        // FP16: 2x compression
        auto fp16_stats = bench_fp16(config);
        print_comparison("FP16 (2x)", fp16_stats, fp32_stats);

        // Q8_1: 3.5x compression (print after running first)
        print_comparison("Q8_1 (3.5x)", q8_1_stats, fp32_stats);

        // Summary
        if (rank_ == 0)
        {
            std::cout << "\n--- Summary ---" << std::endl;
            std::cout << std::fixed << std::setprecision(2);

            double bf16_speedup = fp32_stats.mean_ms / bf16_stats.mean_ms;
            double fp16_speedup = fp32_stats.mean_ms / fp16_stats.mean_ms;
            double q8_1_speedup = fp32_stats.mean_ms / q8_1_stats.mean_ms;

            std::cout << "  BF16 vs FP32: " << bf16_speedup << "x"
                      << " (BW: " << bf16_stats.bandwidth_gbps << " vs " << fp32_stats.bandwidth_gbps << " GB/s)"
                      << std::endl;
            std::cout << "  FP16 vs FP32: " << fp16_speedup << "x"
                      << " (BW: " << fp16_stats.bandwidth_gbps << " vs " << fp32_stats.bandwidth_gbps << " GB/s)"
                      << std::endl;
            std::cout << "  Q8_1 vs FP32: " << q8_1_speedup << "x"
                      << " (BW: " << q8_1_stats.bandwidth_gbps << " vs " << fp32_stats.bandwidth_gbps << " GB/s)"
                      << std::endl;

            // Performance notes
            if (q8_1_speedup < 0.5)
            {
                std::cout << "\n  [NOTE] Q8_1 is slower due to dequant→exp→requant overhead." << std::endl;
                std::cout << "         Integer-domain softmax trades speed for memory compression." << std::endl;
            }
            else if (q8_1_speedup > 1.0)
            {
                std::cout << "\n  [NOTE] Q8_1 is faster due to reduced memory bandwidth pressure." << std::endl;
            }
        }
    }
};

// ============================================================================
// Test Cases - Attention-like Workloads
// ============================================================================

/**
 * @brief Single token decode attention (latency-critical)
 *
 * Typical attention softmax for single token decode:
 * - Rows: num_heads (e.g., 14 for Qwen 0.5B)
 * - Cols: current_seq_len (grows during generation)
 */
TEST_F(CPUSoftmaxKernelTyped_Perf, SingleTokenDecode_ShortContext)
{
    BenchmarkConfig config{
        .rows = 14,  // Qwen 0.5B num_heads
        .cols = 128, // Short context
        .warmup_iters = 10,
        .bench_iters = 100,
        .use_causal = true,
        .scale = 1.0f / std::sqrt(64.0f), // 1/sqrt(d_k)
        .description = "Single Token Decode - Short Context (14 heads × 128 seq)"};
    run_comparison_benchmark(config);
}

TEST_F(CPUSoftmaxKernelTyped_Perf, SingleTokenDecode_MediumContext)
{
    BenchmarkConfig config{
        .rows = 14,
        .cols = 512,
        .warmup_iters = 10,
        .bench_iters = 100,
        .use_causal = true,
        .scale = 1.0f / std::sqrt(64.0f),
        .description = "Single Token Decode - Medium Context (14 heads × 512 seq)"};
    run_comparison_benchmark(config);
}

TEST_F(CPUSoftmaxKernelTyped_Perf, SingleTokenDecode_LongContext)
{
    BenchmarkConfig config{
        .rows = 14,
        .cols = 2048,
        .warmup_iters = 5,
        .bench_iters = 50,
        .use_causal = true,
        .scale = 1.0f / std::sqrt(64.0f),
        .description = "Single Token Decode - Long Context (14 heads × 2048 seq)"};
    run_comparison_benchmark(config);
}

/**
 * @brief Prefill attention (batch processing)
 *
 * During prefill, we process the entire prompt at once:
 * - Rows: seq_len * num_heads (each position attends to all previous)
 * - Cols: seq_len (for square attention matrix pattern)
 */
TEST_F(CPUSoftmaxKernelTyped_Perf, Prefill_SmallPrompt)
{
    // 64 tokens × 14 heads = 896 rows, each attending to up to 64 positions
    BenchmarkConfig config{
        .rows = 64 * 14, // seq_len * num_heads
        .cols = 64,      // seq_len (attention width)
        .warmup_iters = 5,
        .bench_iters = 50,
        .use_causal = true,
        .scale = 1.0f / std::sqrt(64.0f),
        .description = "Prefill - Small Prompt (64 tokens × 14 heads)"};
    run_comparison_benchmark(config);
}

TEST_F(CPUSoftmaxKernelTyped_Perf, Prefill_MediumPrompt)
{
    // 256 tokens × 14 heads = 3584 rows
    BenchmarkConfig config{
        .rows = 256 * 14,
        .cols = 256,
        .warmup_iters = 3,
        .bench_iters = 20,
        .use_causal = true,
        .scale = 1.0f / std::sqrt(64.0f),
        .description = "Prefill - Medium Prompt (256 tokens × 14 heads)"};
    run_comparison_benchmark(config);
}

TEST_F(CPUSoftmaxKernelTyped_Perf, Prefill_LargePrompt)
{
    // 512 tokens × 14 heads = 7168 rows
    BenchmarkConfig config{
        .rows = 512 * 14,
        .cols = 512,
        .warmup_iters = 2,
        .bench_iters = 10,
        .use_causal = true,
        .scale = 1.0f / std::sqrt(64.0f),
        .description = "Prefill - Large Prompt (512 tokens × 14 heads)"};
    run_comparison_benchmark(config);
}

// ============================================================================
// Test Cases - Large Model Configurations
// ============================================================================

/**
 * @brief Qwen 7B-like model (larger head count)
 */
TEST_F(CPUSoftmaxKernelTyped_Perf, Qwen7B_SingleTokenDecode)
{
    // Qwen 7B: 32 heads, head_dim=128
    BenchmarkConfig config{
        .rows = 32,
        .cols = 512,
        .warmup_iters = 10,
        .bench_iters = 50,
        .use_causal = true,
        .scale = 1.0f / std::sqrt(128.0f),
        .description = "Qwen 7B - Single Token Decode (32 heads × 512 seq)"};
    run_comparison_benchmark(config);
}

TEST_F(CPUSoftmaxKernelTyped_Perf, Qwen7B_Prefill)
{
    // 128 tokens × 32 heads = 4096 rows
    BenchmarkConfig config{
        .rows = 128 * 32,
        .cols = 128,
        .warmup_iters = 3,
        .bench_iters = 20,
        .use_causal = true,
        .scale = 1.0f / std::sqrt(128.0f),
        .description = "Qwen 7B - Prefill (128 tokens × 32 heads)"};
    run_comparison_benchmark(config);
}

// ============================================================================
// Test Cases - Non-Causal (Encoder-style)
// ============================================================================

TEST_F(CPUSoftmaxKernelTyped_Perf, NonCausal_Bidirectional)
{
    // Encoder-style attention (no causal masking)
    BenchmarkConfig config{
        .rows = 256 * 12, // 256 tokens × 12 heads
        .cols = 256,
        .warmup_iters = 3,
        .bench_iters = 20,
        .use_causal = false,
        .scale = 1.0f / std::sqrt(64.0f),
        .description = "Non-Causal - Bidirectional Attention (256 × 12 heads)"};
    run_comparison_benchmark(config);
}

// ============================================================================
// Test Cases - Stress Tests
// ============================================================================

TEST_F(CPUSoftmaxKernelTyped_Perf, StressTest_VeryLongContext)
{
    // Very long context (32K-like)
    BenchmarkConfig config{
        .rows = 14,
        .cols = 8192,
        .warmup_iters = 2,
        .bench_iters = 10,
        .use_causal = true,
        .scale = 1.0f / std::sqrt(64.0f),
        .description = "Stress - Very Long Context (14 heads × 8192 seq)"};
    run_comparison_benchmark(config);
}

TEST_F(CPUSoftmaxKernelTyped_Perf, StressTest_ManyHeads)
{
    // Many heads (large model)
    BenchmarkConfig config{
        .rows = 128, // 128 heads
        .cols = 512,
        .warmup_iters = 5,
        .bench_iters = 30,
        .use_causal = true,
        .scale = 1.0f / std::sqrt(128.0f),
        .description = "Stress - Many Heads (128 heads × 512 seq)"};
    run_comparison_benchmark(config);
}

// ============================================================================
// Main with MPI initialization
// ============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
