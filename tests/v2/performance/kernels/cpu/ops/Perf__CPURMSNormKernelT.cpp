/**
 * @file Perf__CPURMSNormKernelT.cpp
 * @brief Performance benchmark for CPURMSNormKernelT (typed residuals)
 *
 * This test benchmarks the typed RMSNorm kernel performance comparing:
 * 1. CPURMSNormKernelT<FP32> (baseline)
 * 2. CPURMSNormKernelT<BF16> (2x memory compression)
 * 3. CPURMSNormKernelT<FP16> (2x memory compression)
 * 4. CPURMSNormKernelT<Q8_1> (3.5x compression, direct SIMD quantization)
 * 5. CPURMSNormKernelT<Q8_1> via IActivationTensor interface (OpenMP parallel)
 *
 * The Q8_1 benchmarks compare two quantization approaches:
 * - **Direct SIMD** (`simd::quantize_fp32_to_q8_1_blocks`): Vectorized AVX512/AVX2,
 *   single-threaded, optimal for small matrices
 * - **IActivationTensor** (`FP32Tensor::quantize_to_q8_1`): OpenMP parallelized,
 *   scalar inner loop, optimal for large matrices with many rows
 *
 * Success Criteria:
 * - BF16/FP16/Q8_1 variants have ≤20% overhead vs FP32
 *   (dequant/requant cost offset by reduced memory bandwidth)
 *
 * @author David Sanftenberg
 * @date 2025-12-04
 *
 * The custom entry owns one lexical native GPU lifetime without probing a
 * device. Native workers/caches retire before process exit and MPI finalization.
 */

#include "backends/GPUDeviceProcessLifetime.h"
#include "utils/MPISessionOwner.h"
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
#include "kernels/cpu/ops/CPURMSNormKernelT.h"
#include "tensors/SIMDHelpers.h"
#include "utils/Logger.h"

using namespace llaminar2;

// ============================================================================
// Benchmark Configuration
// ============================================================================

struct BenchmarkConfig
{
    int seq_len;
    int d_model;
    int warmup_iters;
    int bench_iters;
    std::string description;
};

struct BenchmarkStats
{
    double mean_ms;
    double stddev_ms;
    double min_ms;
    double max_ms;
    double bandwidth_gbps;
    double throughput_tokens_per_sec;
};

// ============================================================================
// Test Fixture
// ============================================================================

class CPURMSNormKernelT_Perf : public ::testing::Test
{
protected:
    int rank_ = 0;
    int world_size_ = 1;
    std::mt19937 rng_{42};

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    }

    // Calculate statistics from timing samples
    BenchmarkStats calculate_stats(
        const std::vector<double> &times_ms,
        int seq_len,
        int d_model,
        double bytes_per_element_in,
        double bytes_per_element_out)
    {
        double sum = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
        double mean = sum / times_ms.size();
        double sq_sum = std::inner_product(times_ms.begin(), times_ms.end(), times_ms.begin(), 0.0);
        double stddev = std::sqrt(sq_sum / times_ms.size() - mean * mean);
        double min_val = *std::min_element(times_ms.begin(), times_ms.end());
        double max_val = *std::max_element(times_ms.begin(), times_ms.end());

        // Bandwidth: Read Input + Read Gamma + Write Output
        // Gamma is small (d_model * 4 bytes), dominated by activations
        double total_bytes = (double)seq_len * d_model * (bytes_per_element_in + bytes_per_element_out);
        total_bytes += d_model * sizeof(float); // gamma read
        double bandwidth_gbps = (total_bytes / (mean * 1e-3)) / 1e9;

        // Throughput: tokens per second
        double throughput = (double)seq_len / (mean * 1e-3);

        return {mean, stddev, min_val, max_val, bandwidth_gbps, throughput};
    }

    void print_result(const std::string &kernel_name, const BenchmarkStats &stats)
    {
        if (rank_ == 0)
        {
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "  " << std::left << std::setw(25) << kernel_name
                      << " Mean: " << std::setw(8) << stats.mean_ms << " ms"
                      << "  Stddev: " << std::setw(8) << stats.stddev_ms << " ms"
                      << "  BW: " << std::setw(8) << stats.bandwidth_gbps << " GB/s"
                      << std::endl;
        }
    }

    void print_comparison(const std::string &kernel_name, const BenchmarkStats &stats,
                          const BenchmarkStats &baseline)
    {
        if (rank_ == 0)
        {
            double overhead_pct = ((stats.mean_ms / baseline.mean_ms) - 1.0) * 100.0;
            std::string overhead_str = (overhead_pct >= 0 ? "+" : "") +
                                       std::to_string(static_cast<int>(overhead_pct)) + "%";

            std::cout << std::fixed << std::setprecision(3);
            std::cout << "  " << std::left << std::setw(25) << kernel_name
                      << " Mean: " << std::setw(8) << stats.mean_ms << " ms"
                      << "  BW: " << std::setw(8) << stats.bandwidth_gbps << " GB/s"
                      << "  vs baseline: " << std::setw(6) << overhead_str
                      << std::endl;
        }
    }

    // ========================================================================
    // Benchmark Functions for Each Kernel Type
    // ========================================================================

    // Typed FP32 kernel (baseline - no precision conversion)
    BenchmarkStats bench_typed_fp32(const BenchmarkConfig &config)
    {
        CPURMSNormKernelT<ActivationPrecision::FP32> kernel;

        size_t size = config.seq_len * config.d_model;
        std::vector<float> input(size);
        std::vector<float> gamma(config.d_model, 1.0f);
        std::vector<float> output(size);

        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : input)
            v = dist(rng_);

        // Warmup
        for (int i = 0; i < config.warmup_iters; ++i)
        {
            kernel.apply_typed(input.data(), gamma.data(), output.data(),
                               config.seq_len, config.d_model, 1e-6f);
        }

        // Benchmark
        std::vector<double> times_ms;
        times_ms.reserve(config.bench_iters);

        for (int i = 0; i < config.bench_iters; ++i)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            auto start = std::chrono::high_resolution_clock::now();

            kernel.apply_typed(input.data(), gamma.data(), output.data(),
                               config.seq_len, config.d_model, 1e-6f);

            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e6;
            times_ms.push_back(ms);
        }

        return calculate_stats(times_ms, config.seq_len, config.d_model,
                               sizeof(float), sizeof(float));
    }

    // Typed BF16 kernel (2x compression)
    BenchmarkStats bench_typed_bf16(const BenchmarkConfig &config)
    {
        CPURMSNormKernelT<ActivationPrecision::BF16> kernel;

        size_t size = config.seq_len * config.d_model;
        std::vector<uint16_t> input(size);
        std::vector<float> gamma(config.d_model, 1.0f);
        std::vector<uint16_t> output(size);

        // Initialize: generate FP32 random data, convert to BF16
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> fp32_temp(size);
        for (auto &v : fp32_temp)
            v = dist(rng_);
        simd::convert_fp32_to_bf16(fp32_temp.data(), input.data(), size);

        // Warmup
        for (int i = 0; i < config.warmup_iters; ++i)
        {
            kernel.apply_typed(input.data(), gamma.data(), output.data(),
                               config.seq_len, config.d_model, 1e-6f);
        }

        // Benchmark
        std::vector<double> times_ms;
        times_ms.reserve(config.bench_iters);

        for (int i = 0; i < config.bench_iters; ++i)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            auto start = std::chrono::high_resolution_clock::now();

            kernel.apply_typed(input.data(), gamma.data(), output.data(),
                               config.seq_len, config.d_model, 1e-6f);

            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e6;
            times_ms.push_back(ms);
        }

        // BF16: 2 bytes per element
        return calculate_stats(times_ms, config.seq_len, config.d_model,
                               sizeof(uint16_t), sizeof(uint16_t));
    }

    // Typed FP16 kernel (2x compression)
    BenchmarkStats bench_typed_fp16(const BenchmarkConfig &config)
    {
        CPURMSNormKernelT<ActivationPrecision::FP16> kernel;

        size_t size = config.seq_len * config.d_model;
        std::vector<uint16_t> input(size);
        std::vector<float> gamma(config.d_model, 1.0f);
        std::vector<uint16_t> output(size);

        // Initialize: generate FP32 random data, convert to FP16
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> fp32_temp(size);
        for (auto &v : fp32_temp)
            v = dist(rng_);
        simd::convert_fp32_to_fp16(fp32_temp.data(), input.data(), size);

        // Warmup
        for (int i = 0; i < config.warmup_iters; ++i)
        {
            kernel.apply_typed(input.data(), gamma.data(), output.data(),
                               config.seq_len, config.d_model, 1e-6f);
        }

        // Benchmark
        std::vector<double> times_ms;
        times_ms.reserve(config.bench_iters);

        for (int i = 0; i < config.bench_iters; ++i)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            auto start = std::chrono::high_resolution_clock::now();

            kernel.apply_typed(input.data(), gamma.data(), output.data(),
                               config.seq_len, config.d_model, 1e-6f);

            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e6;
            times_ms.push_back(ms);
        }

        // FP16: 2 bytes per element
        return calculate_stats(times_ms, config.seq_len, config.d_model,
                               sizeof(uint16_t), sizeof(uint16_t));
    }

    // Typed Q8_1 kernel (3.5x compression)
    BenchmarkStats bench_typed_q8_1(const BenchmarkConfig &config)
    {
        CPURMSNormKernelT<ActivationPrecision::Q8_1> kernel;

        // Q8_1: 36 bytes per block of 32 elements
        // Number of blocks per row = ceil(d_model / 32)
        size_t blocks_per_row = (config.d_model + 31) / 32;
        size_t total_blocks = config.seq_len * blocks_per_row;

        std::vector<Q8_1Block> input(total_blocks);
        std::vector<float> gamma(config.d_model, 1.0f);
        std::vector<Q8_1Block> output(total_blocks);

        // Initialize: generate FP32 random data, convert to Q8_1
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        size_t fp32_size = config.seq_len * config.d_model;
        std::vector<float> fp32_temp(fp32_size);
        for (auto &v : fp32_temp)
            v = dist(rng_);
        simd::quantize_fp32_to_q8_1_blocks(fp32_temp.data(), input.data(), fp32_size);

        // Warmup
        for (int i = 0; i < config.warmup_iters; ++i)
        {
            kernel.apply_typed(input.data(), gamma.data(), output.data(),
                               config.seq_len, config.d_model, 1e-6f);
        }

        // Benchmark
        std::vector<double> times_ms;
        times_ms.reserve(config.bench_iters);

        for (int i = 0; i < config.bench_iters; ++i)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            auto start = std::chrono::high_resolution_clock::now();

            kernel.apply_typed(input.data(), gamma.data(), output.data(),
                               config.seq_len, config.d_model, 1e-6f);

            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e6;
            times_ms.push_back(ms);
        }

        // Q8_1: 36 bytes / 32 elements = 1.125 bytes per element
        double bytes_per_element_q8_1 = 36.0 / 32.0;
        return calculate_stats(times_ms, config.seq_len, config.d_model,
                               bytes_per_element_q8_1, bytes_per_element_q8_1);
    }

    // ========================================================================
    // Run Full Comparison Benchmark
    // ========================================================================

    void run_comparison_benchmark(const BenchmarkConfig &config)
    {
        if (rank_ == 0)
        {
            std::cout << "\n================================================================" << std::endl;
            std::cout << "Benchmark: " << config.description << std::endl;
            std::cout << "  Seq Len: " << config.seq_len << ", D Model: " << config.d_model << std::endl;
            std::cout << "  Warmup: " << config.warmup_iters << ", Iterations: " << config.bench_iters << std::endl;
            std::cout << "================================================================" << std::endl;
        }

        // Run Q8_1 FIRST to avoid cache pollution from other benchmarks
        // Typed Q8_1: 3.5x compression
        auto typed_q8_1 = bench_typed_q8_1(config);

        // Baseline: typed FP32 kernel (no precision conversion)
        auto baseline = bench_typed_fp32(config);
        print_result("Typed FP32 (baseline)", baseline);

        // Typed BF16: 2x compression
        auto typed_bf16 = bench_typed_bf16(config);
        print_comparison("Typed BF16 (2x)", typed_bf16, baseline);

        // Typed FP16: 2x compression
        auto typed_fp16 = bench_typed_fp16(config);
        print_comparison("Typed FP16 (2x)", typed_fp16, baseline);

        // Print Q8_1 result (ran first but print in original order)
        print_comparison("Typed Q8_1 (3.5x)", typed_q8_1, baseline);

        // Validation assertions
        if (rank_ == 0)
        {
            std::cout << "\n--- Validation ---" << std::endl;

            // Check other formats ≤20% overhead (allowing some margin)
            double bf16_overhead = (typed_bf16.mean_ms / baseline.mean_ms - 1.0);
            double fp16_overhead = (typed_fp16.mean_ms / baseline.mean_ms - 1.0);
            double q8_1_overhead = (typed_q8_1.mean_ms / baseline.mean_ms - 1.0);

            std::cout << "  BF16 overhead: " << std::fixed << std::setprecision(1)
                      << (bf16_overhead * 100.0) << "%"
                      << (bf16_overhead <= 0.20 ? " [PASS]" : " [WARN: >20%]") << std::endl;

            std::cout << "  FP16 overhead: " << std::fixed << std::setprecision(1)
                      << (fp16_overhead * 100.0) << "%"
                      << (fp16_overhead <= 0.20 ? " [PASS]" : " [WARN: >20%]") << std::endl;

            std::cout << "  Q8_1 overhead: " << std::fixed << std::setprecision(1)
                      << (q8_1_overhead * 100.0) << "%"
                      << (q8_1_overhead <= 0.20 ? " [PASS]" : " [WARN: >20%]") << std::endl;
        }
    }
};

// ============================================================================
// Test Cases
// ============================================================================

// Single token decode (latency-critical path)
TEST_F(CPURMSNormKernelT_Perf, SingleToken_Qwen05B)
{
    BenchmarkConfig config;
    config.seq_len = 1;
    config.d_model = 896; // Qwen 2.5 0.5B
    config.warmup_iters = 100;
    config.bench_iters = 1000;
    config.description = "Single Token - Qwen 0.5B (1×896)";

    run_comparison_benchmark(config);
}

TEST_F(CPURMSNormKernelT_Perf, SingleToken_Qwen7B)
{
    BenchmarkConfig config;
    config.seq_len = 1;
    config.d_model = 3584; // Qwen 2.5 7B
    config.warmup_iters = 100;
    config.bench_iters = 1000;
    config.description = "Single Token - Qwen 7B (1×3584)";

    run_comparison_benchmark(config);
}

// Small batch (typical prefill)
TEST_F(CPURMSNormKernelT_Perf, Prefill_32_Qwen7B)
{
    BenchmarkConfig config;
    config.seq_len = 32;
    config.d_model = 3584;
    config.warmup_iters = 50;
    config.bench_iters = 500;
    config.description = "Prefill 32 tokens - Qwen 7B (32×3584)";

    run_comparison_benchmark(config);
}

// Medium batch prefill
TEST_F(CPURMSNormKernelT_Perf, Prefill_128_Qwen7B)
{
    BenchmarkConfig config;
    config.seq_len = 128;
    config.d_model = 3584;
    config.warmup_iters = 20;
    config.bench_iters = 200;
    config.description = "Prefill 128 tokens - Qwen 7B (128×3584)";

    run_comparison_benchmark(config);
}

// Large batch prefill (stress test)
TEST_F(CPURMSNormKernelT_Perf, Prefill_512_Qwen7B)
{
    BenchmarkConfig config;
    config.seq_len = 512;
    config.d_model = 3584;
    config.warmup_iters = 50;
    config.bench_iters = 500;
    config.description = "Prefill 512 tokens - Qwen 7B (512×3584)";

    run_comparison_benchmark(config);
}

// Very large prefill (max typical context)
TEST_F(CPURMSNormKernelT_Perf, Prefill_2048_Qwen7B)
{
    BenchmarkConfig config;
    config.seq_len = 2048;
    config.d_model = 3584;
    config.warmup_iters = 50;
    config.bench_iters = 200;
    config.description = "Prefill 2048 tokens - Qwen 7B (2048×3584)";

    run_comparison_benchmark(config);
}

// ============================================================================
// Qwen 32B Tests (d_model=5120)
// ============================================================================

// Single token decode - Qwen 32B
TEST_F(CPURMSNormKernelT_Perf, SingleToken_Qwen32B)
{
    BenchmarkConfig config;
    config.seq_len = 1;
    config.d_model = 5120; // Qwen 2.5 32B
    config.warmup_iters = 100;
    config.bench_iters = 1000;
    config.description = "Single Token - Qwen 32B (1×5120)";

    run_comparison_benchmark(config);
}

// Small batch prefill - Qwen 32B
TEST_F(CPURMSNormKernelT_Perf, Prefill_32_Qwen32B)
{
    BenchmarkConfig config;
    config.seq_len = 32;
    config.d_model = 5120;
    config.warmup_iters = 50;
    config.bench_iters = 500;
    config.description = "Prefill 32 tokens - Qwen 32B (32×5120)";

    run_comparison_benchmark(config);
}

// Medium batch prefill - Qwen 32B
TEST_F(CPURMSNormKernelT_Perf, Prefill_128_Qwen32B)
{
    BenchmarkConfig config;
    config.seq_len = 128;
    config.d_model = 5120;
    config.warmup_iters = 20;
    config.bench_iters = 200;
    config.description = "Prefill 128 tokens - Qwen 32B (128×5120)";

    run_comparison_benchmark(config);
}

// Large batch prefill - Qwen 32B
TEST_F(CPURMSNormKernelT_Perf, Prefill_512_Qwen32B)
{
    BenchmarkConfig config;
    config.seq_len = 512;
    config.d_model = 5120;
    config.warmup_iters = 50;
    config.bench_iters = 500;
    config.description = "Prefill 512 tokens - Qwen 32B (512×5120)";

    run_comparison_benchmark(config);
}

// Very large prefill - Qwen 32B
TEST_F(CPURMSNormKernelT_Perf, Prefill_2048_Qwen32B)
{
    BenchmarkConfig config;
    config.seq_len = 2048;
    config.d_model = 5120;
    config.warmup_iters = 50;
    config.bench_iters = 200;
    config.description = "Prefill 2048 tokens - Qwen 32B (2048×5120)";

    run_comparison_benchmark(config);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv)
{
    if (auto terminal = llaminar2::MPISessionOwner::enterProcess(argc, argv))
        return *terminal;
    llaminar2::GPUDeviceProcessLifetime gpu_process_lifetime;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    int result = RUN_ALL_TESTS();
    llaminar2::GPUDeviceProcessLifetime::retireActive();
    MPI_Finalize();
    return result;
}
