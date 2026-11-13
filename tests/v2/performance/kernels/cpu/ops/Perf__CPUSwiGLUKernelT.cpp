/**
 * @file Perf__CPUSwiGLUKernelT.cpp
 * @brief Performance benchmark for CPUSwiGLUKernelT (typed SwiGLU kernels)
 *
 * This test benchmarks the typed SwiGLU kernel performance comparing:
 * 1. CPUSwiGLUKernelT<FP32> (baseline)
 * 2. CPUSwiGLUKernelT<BF16> (2x memory compression)
 * 3. CPUSwiGLUKernelT<FP16> (2x memory compression)
 * 4. CPUSwiGLUKernelT<Q8_1> (4x compression, integer-aware SwiGLU)
 *
 * SwiGLU computational complexity:
 *   - For each element: silu(gate) * up
 *   - silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *   - Per element: 1 exp, 1 add, 2 divides/muls
 *   - Total: ~5 FLOPs per element (dominated by exp)
 *
 * For N elements:
 *   - FLOPs estimate: 5 * N
 *   - Memory: 3 * N * bytes_per_element (2 inputs + 1 output)
 *
 * The Q8_1 kernel dequantizes per block, computes SwiGLU, and requantizes.
 *
 * @author David Sanftenberg
 * @date 2025-12-07
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
#include <cstring>

// V2 includes
#include "tensors/Tensors.h"
#include "tensors/BlockStructures.h"
#include "kernels/cpu/ops/CPUSwiGLUKernelT.h"
#include "tensors/SIMDHelpers.h"
#include "utils/Logger.h"

using namespace llaminar2;

// ============================================================================
// Benchmark Configuration
// ============================================================================

struct BenchmarkConfig
{
    int size;         ///< Number of elements (must be multiple of 32 for Q8_1)
    int warmup_iters; ///< Number of warmup iterations
    int bench_iters;  ///< Number of timed benchmark iterations
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
// Helper Functions
// ============================================================================

namespace
{
    // BF16 conversion (not provided by library, needed locally)
    inline uint16_t fp32_to_bf16(float val)
    {
        uint32_t bits;
        std::memcpy(&bits, &val, sizeof(float));
        return static_cast<uint16_t>(bits >> 16);
    }

    inline float bf16_to_fp32(uint16_t val)
    {
        uint32_t bits = static_cast<uint32_t>(val) << 16;
        float result;
        std::memcpy(&result, &bits, sizeof(float));
        return result;
    }

    // Use library FP16 conversions: llaminar2::fp32_to_fp16, llaminar2::fp16_to_fp32

    Q8_1Block create_q8_1_block(const float *values)
    {
        Q8_1Block block;
        float max_abs = 0.0f;
        for (int i = 0; i < 32; ++i)
        {
            max_abs = std::max(max_abs, std::abs(values[i]));
        }
        float d = max_abs / 127.0f;
        if (d < 1e-10f)
            d = 1e-10f;
        // Q8_1Block uses IEEE FP16 (not BF16!)
        block.d = llaminar2::fp32_to_fp16(d);
        float inv_d = 1.0f / d;
        int16_t sum_qs = 0;
        for (int i = 0; i < 32; ++i)
        {
            int32_t q = static_cast<int32_t>(std::round(values[i] * inv_d));
            q = std::max(-127, std::min(127, q));
            block.qs[i] = static_cast<int8_t>(q);
            sum_qs += block.qs[i];
        }
        block.sum_qs = sum_qs;
        return block;
    }

    std::vector<float> dequantize_q8_1_block(const Q8_1Block &block)
    {
        std::vector<float> result(32);
        // Q8_1Block uses IEEE FP16 (not BF16!)
        float d = llaminar2::fp16_to_fp32(block.d);
        for (int i = 0; i < 32; ++i)
        {
            result[i] = static_cast<float>(block.qs[i]) * d;
        }
        return result;
    }

    float compute_l2_error(const std::vector<float> &a, const std::vector<float> &b)
    {
        if (a.size() != b.size())
            return -1.0f;
        float sum_sq = 0.0f;
        for (size_t i = 0; i < a.size(); ++i)
        {
            float diff = a[i] - b[i];
            sum_sq += diff * diff;
        }
        return std::sqrt(sum_sq / a.size());
    }

    float compute_cosine_similarity(const std::vector<float> &a, const std::vector<float> &b)
    {
        if (a.size() != b.size() || a.empty())
            return 0.0f;
        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (size_t i = 0; i < a.size(); ++i)
        {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        if (norm_a < 1e-12f || norm_b < 1e-12f)
            return 1.0f;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }
} // namespace

// ============================================================================
// Test Fixture
// ============================================================================

class CPUSwiGLUKernelT_Perf : public ::testing::Test
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
            std::cout << "\n[Performance Test] CPUSwiGLUKernelT Benchmark" << std::endl;
            std::cout << "[Performance Test] OpenMP threads: " << omp_get_max_threads() << std::endl;
        }
    }

    BenchmarkStats calculate_stats(
        const std::vector<double> &times_ms,
        int size,
        double bytes_per_element)
    {
        double sum = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
        double mean = sum / times_ms.size();
        double sq_sum = std::inner_product(times_ms.begin(), times_ms.end(), times_ms.begin(), 0.0);
        double stddev = std::sqrt(sq_sum / times_ms.size() - mean * mean);
        double min_val = *std::min_element(times_ms.begin(), times_ms.end());
        double max_val = *std::max_element(times_ms.begin(), times_ms.end());

        // Memory bandwidth: 2 inputs + 1 output
        double total_bytes = static_cast<double>(size) * bytes_per_element * 3.0;
        double bandwidth_gbps = (total_bytes / (mean * 1e-3)) / 1e9;

        // FLOPS estimate for SwiGLU:
        // silu(x) = x / (1 + exp(-x)) = ~4 ops per element
        // silu(gate) * up = 1 mul
        // Total: ~5 ops per element
        double flops = static_cast<double>(size) * 5.0;
        double gflops = (flops / (mean * 1e-3)) / 1e9;

        double elements_per_sec = size / (mean * 1e-3);

        return {mean, stddev, min_val, max_val, bandwidth_gbps, gflops, elements_per_sec};
    }

    void print_header(const BenchmarkConfig &config)
    {
        if (rank_ == 0)
        {
            std::cout << "\n════════════════════════════════════════════════════════════════" << std::endl;
            std::cout << "Benchmark: " << config.description << std::endl;
            std::cout << "  Size: " << config.size << " elements (" << (config.size / 1e6) << "M)" << std::endl;
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
                          const BenchmarkStats &baseline, float l2_err = -1.0f, float cos_sim = -1.0f)
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
                      << "  vs FP32: " << speedup_str;

            if (l2_err >= 0.0f)
            {
                std::cout << std::scientific << std::setprecision(2)
                          << "  L2 Err: " << l2_err;
            }
            if (cos_sim >= 0.0f)
            {
                std::cout << std::fixed << std::setprecision(5)
                          << "  Cos: " << cos_sim;
            }
            std::cout << std::endl;
        }
    }

    void print_summary(const BenchmarkStats &fp32_stats,
                       const BenchmarkStats &bf16_stats,
                       const BenchmarkStats &fp16_stats,
                       const BenchmarkStats &q8_stats)
    {
        if (rank_ == 0)
        {
            std::cout << "\n--- Summary ---" << std::endl;
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "  BF16 vs FP32: " << (fp32_stats.mean_ms / bf16_stats.mean_ms) << "x"
                      << " (BW: " << bf16_stats.bandwidth_gbps << " vs " << fp32_stats.bandwidth_gbps << " GB/s)" << std::endl;
            std::cout << "  FP16 vs FP32: " << (fp32_stats.mean_ms / fp16_stats.mean_ms) << "x"
                      << " (BW: " << fp16_stats.bandwidth_gbps << " vs " << fp32_stats.bandwidth_gbps << " GB/s)" << std::endl;
            std::cout << "  Q8_1 vs FP32: " << (fp32_stats.mean_ms / q8_stats.mean_ms) << "x"
                      << " (BW: " << q8_stats.bandwidth_gbps << " vs " << fp32_stats.bandwidth_gbps << " GB/s)" << std::endl;

            if (q8_stats.mean_ms < fp32_stats.mean_ms)
            {
                std::cout << "\n  [NOTE] Q8_1 is faster due to reduced memory bandwidth pressure." << std::endl;
            }
        }
    }

    // ========================================================================
    // Benchmark Function for All Precisions
    // ========================================================================

    void run_benchmark(const BenchmarkConfig &config)
    {
        print_header(config);

        const int size = config.size;
        const int n_blocks = size / 32;

        // Generate random test data
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

        std::vector<float> gate_fp32(size), up_fp32(size), output_fp32(size);
        std::vector<uint16_t> gate_bf16(size), up_bf16(size), output_bf16(size);
        std::vector<uint16_t> gate_fp16(size), up_fp16(size), output_fp16(size);
        std::vector<Q8_1Block> gate_q8(n_blocks), up_q8(n_blocks), output_q8(n_blocks);

        for (int i = 0; i < size; ++i)
        {
            gate_fp32[i] = dist(rng_);
            up_fp32[i] = dist(rng_);
            gate_bf16[i] = fp32_to_bf16(gate_fp32[i]);
            up_bf16[i] = fp32_to_bf16(up_fp32[i]);
            gate_fp16[i] = llaminar2::fp32_to_fp16(gate_fp32[i]);
            up_fp16[i] = llaminar2::fp32_to_fp16(up_fp32[i]);
        }

        for (int b = 0; b < n_blocks; ++b)
        {
            gate_q8[b] = create_q8_1_block(&gate_fp32[b * 32]);
            up_q8[b] = create_q8_1_block(&up_fp32[b * 32]);
        }

        // Create kernels
        CPUSwiGLUKernelT<ActivationPrecision::FP32> kernel_fp32;
        CPUSwiGLUKernelT<ActivationPrecision::BF16> kernel_bf16;
        CPUSwiGLUKernelT<ActivationPrecision::FP16> kernel_fp16;
        CPUSwiGLUKernelT<ActivationPrecision::Q8_1> kernel_q8;

        // ================================================================
        // FP32 Benchmark (baseline)
        // ================================================================
        std::vector<double> times_fp32;
        for (int i = 0; i < config.warmup_iters; ++i)
        {
            kernel_fp32.apply_typed(gate_fp32.data(), up_fp32.data(), output_fp32.data(), size);
        }
        for (int i = 0; i < config.bench_iters; ++i)
        {
            auto start = std::chrono::high_resolution_clock::now();
            kernel_fp32.apply_typed(gate_fp32.data(), up_fp32.data(), output_fp32.data(), size);
            auto end = std::chrono::high_resolution_clock::now();
            times_fp32.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }
        auto stats_fp32 = calculate_stats(times_fp32, size, sizeof(float));
        print_result("FP32 (baseline)", stats_fp32);

        // ================================================================
        // BF16 Benchmark
        // ================================================================
        std::vector<double> times_bf16;
        for (int i = 0; i < config.warmup_iters; ++i)
        {
            kernel_bf16.apply_typed(gate_bf16.data(), up_bf16.data(), output_bf16.data(), size);
        }
        for (int i = 0; i < config.bench_iters; ++i)
        {
            auto start = std::chrono::high_resolution_clock::now();
            kernel_bf16.apply_typed(gate_bf16.data(), up_bf16.data(), output_bf16.data(), size);
            auto end = std::chrono::high_resolution_clock::now();
            times_bf16.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }
        auto stats_bf16 = calculate_stats(times_bf16, size, sizeof(uint16_t));

        // Convert BF16 output to FP32 for comparison
        std::vector<float> output_bf16_fp32(size);
        for (int i = 0; i < size; ++i)
        {
            output_bf16_fp32[i] = bf16_to_fp32(output_bf16[i]);
        }
        float l2_bf16 = compute_l2_error(output_fp32, output_bf16_fp32);
        float cos_bf16 = compute_cosine_similarity(output_fp32, output_bf16_fp32);
        print_comparison("BF16 (2x)", stats_bf16, stats_fp32, l2_bf16, cos_bf16);

        // ================================================================
        // FP16 Benchmark
        // ================================================================
        std::vector<double> times_fp16;
        for (int i = 0; i < config.warmup_iters; ++i)
        {
            kernel_fp16.apply_typed(gate_fp16.data(), up_fp16.data(), output_fp16.data(), size);
        }
        for (int i = 0; i < config.bench_iters; ++i)
        {
            auto start = std::chrono::high_resolution_clock::now();
            kernel_fp16.apply_typed(gate_fp16.data(), up_fp16.data(), output_fp16.data(), size);
            auto end = std::chrono::high_resolution_clock::now();
            times_fp16.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }
        auto stats_fp16 = calculate_stats(times_fp16, size, sizeof(uint16_t));

        // Convert FP16 output to FP32 for comparison
        std::vector<float> output_fp16_fp32(size);
        for (int i = 0; i < size; ++i)
        {
            output_fp16_fp32[i] = llaminar2::fp16_to_fp32(output_fp16[i]);
        }
        float l2_fp16 = compute_l2_error(output_fp32, output_fp16_fp32);
        float cos_fp16 = compute_cosine_similarity(output_fp32, output_fp16_fp32);
        print_comparison("FP16 (2x)", stats_fp16, stats_fp32, l2_fp16, cos_fp16);

        // ================================================================
        // Q8_1 Benchmark
        // ================================================================
        std::vector<double> times_q8;
        for (int i = 0; i < config.warmup_iters; ++i)
        {
            kernel_q8.apply_typed(gate_q8.data(), up_q8.data(), output_q8.data(), size);
        }
        for (int i = 0; i < config.bench_iters; ++i)
        {
            auto start = std::chrono::high_resolution_clock::now();
            kernel_q8.apply_typed(gate_q8.data(), up_q8.data(), output_q8.data(), size);
            auto end = std::chrono::high_resolution_clock::now();
            times_q8.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }
        // Q8_1 bytes per element: sizeof(Q8_1Block) / 32 = 36 / 32 ≈ 1.125 bytes
        auto stats_q8 = calculate_stats(times_q8, size, 36.0 / 32.0);

        // Dequantize Q8_1 output for comparison
        std::vector<float> output_q8_fp32;
        for (int b = 0; b < n_blocks; ++b)
        {
            auto block = dequantize_q8_1_block(output_q8[b]);
            output_q8_fp32.insert(output_q8_fp32.end(), block.begin(), block.end());
        }
        float l2_q8 = compute_l2_error(output_fp32, output_q8_fp32);
        float cos_q8 = compute_cosine_similarity(output_fp32, output_q8_fp32);
        print_comparison("Q8_1 (4x)", stats_q8, stats_fp32, l2_q8, cos_q8);

        print_summary(stats_fp32, stats_bf16, stats_fp16, stats_q8);
    }
};

// ============================================================================
// Benchmark Test Cases
// ============================================================================

TEST_F(CPUSwiGLUKernelT_Perf, SmallFFN_1K)
{
    run_benchmark({
        .size = 1024,
        .warmup_iters = 10,
        .bench_iters = 100,
        .description = "Small FFN (1K elements)",
    });
}

TEST_F(CPUSwiGLUKernelT_Perf, MediumFFN_32K)
{
    run_benchmark({
        .size = 32 * 1024,
        .warmup_iters = 10,
        .bench_iters = 100,
        .description = "Medium FFN (32K elements)",
    });
}

TEST_F(CPUSwiGLUKernelT_Perf, LargeFFN_256K)
{
    run_benchmark({
        .size = 256 * 1024,
        .warmup_iters = 5,
        .bench_iters = 50,
        .description = "Large FFN (256K elements)",
    });
}

TEST_F(CPUSwiGLUKernelT_Perf, VeryLargeFFN_1M)
{
    run_benchmark({
        .size = 1024 * 1024,
        .warmup_iters = 3,
        .bench_iters = 20,
        .description = "Very Large FFN (1M elements)",
    });
}

TEST_F(CPUSwiGLUKernelT_Perf, Qwen05B_FFN)
{
    // Qwen 0.5B: d_model=896, d_ff=4864
    // FFN intermediate size: batch * seq_len * d_ff
    // For seq_len=256: 256 * 4864 = 1,245,184
    run_benchmark({
        .size = 256 * 4864, // ~1.2M, rounded to multiple of 32
        .warmup_iters = 3,
        .bench_iters = 20,
        .description = "Qwen 0.5B FFN (seq=256, d_ff=4864)",
    });
}

TEST_F(CPUSwiGLUKernelT_Perf, Qwen7B_FFN)
{
    // Qwen 7B: d_model=4096, d_ff=11008
    // For seq_len=128: 128 * 11008 = 1,409,024
    run_benchmark({
        .size = 128 * 11008, // ~1.4M, rounded to multiple of 32
        .warmup_iters = 3,
        .bench_iters = 20,
        .description = "Qwen 7B FFN (seq=128, d_ff=11008)",
    });
}

TEST_F(CPUSwiGLUKernelT_Perf, BatchDecoding_SmallBatch)
{
    // Small batch decode: 8 sequences × d_ff
    // d_ff = 4864 (Qwen 0.5B)
    run_benchmark({
        .size = 8 * 4864,
        .warmup_iters = 10,
        .bench_iters = 100,
        .description = "Batch Decode (8 seq × 4864)",
    });
}

TEST_F(CPUSwiGLUKernelT_Perf, BatchDecoding_LargeBatch)
{
    // Large batch decode: 64 sequences × d_ff
    run_benchmark({
        .size = 64 * 4864,
        .warmup_iters = 5,
        .bench_iters = 50,
        .description = "Batch Decode (64 seq × 4864)",
    });
}

// ============================================================================
// Main with MPI initialization
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
