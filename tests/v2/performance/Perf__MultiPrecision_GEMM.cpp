/**
 * @file Perf__MultiPrecision_GEMM.cpp
 * @brief Performance benchmark for multi-precision GEMM operations
 *
 * This test benchmarks pure GEMM performance across all supported precision types:
 *   - FP32 × FP32 → FP32 (baseline)
 *   - BF16 × BF16 → FP32 (reduced precision activations + weights)
 *   - FP16 × FP16 → FP32 (IEEE half precision)
 *   - INT8 × INT8 → FP32 (quantized activations + weights)
 *
 * Measures:
 *   - Throughput (GFLOPS)
 *   - Memory bandwidth (GB/s)
 *   - Time per iteration (ms)
 *   - Efficiency vs FP32 baseline
 *
 * Test configuration:
 *   - Runs on Release builds (build_v2_release)
 *   - Uses optimal MPI/OpenMP settings
 *   - Pins to physical cores
 *   - Includes warmup iterations
 *   - Multiple trials for statistical significance
 *
 * Based on Perf__IQ4_NL_GEMM.cpp but tests symmetric precision GEMM
 * (both inputs same precision, not mixed quantized weights).
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <numeric>

// CBLAS for FP32 GEMM baseline
#include <cblas.h>

// V2 includes
#include "tensors/Tensors.h"
#include "tensors/SIMDHelpers.h"
#include "backends/ComputeBackend.h"
#include "kernels/cpu/gemm/GemmKernelTemplate.h"
#include "kernels/cpu/SimdTraits.h"
#include "kernels/cpu/gemm/GemmAutoTuner.h"
#include "kernels/cpu/gemm/GemmMicroKernelAdapter.h"

using namespace llaminar2;

/**
 * @brief Configuration for a single benchmark run
 */
struct BenchmarkConfig
{
    int m;                   ///< Number of rows in A (sequence length)
    int n;                   ///< Number of columns in B (output features)
    int k;                   ///< Number of columns in A / rows in B (input features)
    int warmup_iters;        ///< Number of warmup iterations
    int bench_iters;         ///< Number of timed benchmark iterations per trial
    int num_trials;          ///< Number of independent trials for statistics
    std::string description; ///< Human-readable description
};

/**
 * @brief Statistics for multiple benchmark trials
 */
struct BenchmarkStats
{
    double mean_ms;       ///< Mean time per iteration (ms)
    double stddev_ms;     ///< Standard deviation (ms)
    double min_ms;        ///< Minimum time (ms)
    double max_ms;        ///< Maximum time (ms)
    double mean_gflops;   ///< Mean throughput (GFLOPS)
    double stddev_gflops; ///< Standard deviation of throughput
    double bandwidth_gb;  ///< Memory bandwidth (GB/s)
};

/**
 * @brief Multi-Precision GEMM Performance Test Fixture
 *
 * Tests pure GEMM performance across all precision types.
 * No model loading required - uses synthetic data.
 */
class MultiPrecisionGemmPerf : public ::testing::Test
{
protected:
    int rank_ = 0;
    int world_size_ = 1;

    void SetUp() override
    {
        // Initialize MPI context
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

        // Initialize DeviceManager
        DeviceManager::instance().initialize(-1); // -1 = no NUMA filtering

        // Verify OpenMP is configured
        int max_threads = omp_get_max_threads();
        if (rank_ == 0)
        {
            std::cout << "[Multi-Precision GEMM] OpenMP max threads: " << max_threads << std::endl;
        }
        ASSERT_GT(max_threads, 1)
            << "OpenMP not configured! Expected OMP_NUM_THREADS > 1, got " << max_threads;
    }

    void TearDown() override
    {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /**
     * @brief Initialize FP32 matrix with realistic values
     */
    void initFP32Matrix(float *data, size_t size)
    {
        for (size_t i = 0; i < size; ++i)
        {
            // Realistic activation range [-0.5, 0.5]
            data[i] = (static_cast<float>(i % 1000) / 1000.0f) - 0.5f;
        }
    }

    /**
     * @brief Convert FP32 matrix to BF16
     */
    void convertToBF16(const float *src, uint16_t *dst, size_t size)
    {
        llaminar2::simd::convert_fp32_to_bf16(src, dst, size);
    }

    /**
     * @brief Convert FP32 matrix to FP16 (vectorized with AVX512/AVX2/F16C)
     */
    void convertToFP16(const float *src, uint16_t *dst, size_t size)
    {
        llaminar2::simd::convert_fp32_to_fp16(src, dst, size);
    }

    /**
     * @brief Convert FP32 matrix to INT8
     */
    void convertToINT8(const float *src, int8_t *dst, size_t size, float scale = 127.0f)
    {
        for (size_t i = 0; i < size; ++i)
        {
            float val = src[i] * scale;
            val = std::max(-128.0f, std::min(127.0f, val));
            dst[i] = static_cast<int8_t>(std::round(val));
        }
    }

    /**
     * @brief Calculate GFLOPS for GEMM
     *
     * GEMM: C[m×n] = A[m×k] × B[k×n]
     * FLOPs = 2*m*n*k (multiply + add per element)
     */
    double calculateGFLOPS(int m, int n, int k, double time_ms)
    {
        double flops = 2.0 * m * n * k;
        return (flops / time_ms) / 1e6; // GFLOPS
    }

    /**
     * @brief Calculate memory bandwidth
     *
     * Reads: A[m×k] + B[k×n]
     * Writes: C[m×n]
     */
    double calculateBandwidth(int m, int n, int k, size_t element_bytes, double time_ms)
    {
        size_t reads = (m * k + k * n) * element_bytes;
        size_t writes = m * n * sizeof(float); // Output always FP32
        double total_bytes = reads + writes;
        return (total_bytes / time_ms) / 1e6; // GB/s
    }

    /**
     * @brief Print benchmark results (rank 0 only)
     */
    void printResults(const BenchmarkConfig &config, const BenchmarkStats &stats, const std::string &precision)
    {
        if (rank_ != 0)
            return;

        double cv_percent = (stats.stddev_ms / stats.mean_ms) * 100.0;

        std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║ " << std::left << std::setw(62) << (precision + " GEMM - " + config.description) << " ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Configuration:                                                 ║\n";
        std::cout << "║   Matrix Dimensions: " << std::setw(5) << config.m << " × " << std::setw(5) << config.k
                  << " × " << std::setw(5) << config.n << "                        ║\n";
        std::cout << "║   MPI Ranks:         " << std::setw(10) << world_size_ << "                                      ║\n";
        std::cout << "║   Trials:            " << std::setw(10) << config.num_trials << "                                      ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Performance (mean ± stddev):                                   ║\n";
        std::cout << "║   Time per iter:    " << std::setw(7) << std::fixed << std::setprecision(2) << stats.mean_ms
                  << " ± " << std::setw(5) << stats.stddev_ms << " ms";
        std::cout << " (CV: " << std::setw(4) << std::setprecision(1) << cv_percent << "%)            ║\n";
        std::cout << "║   Throughput:       " << std::setw(7) << std::fixed << std::setprecision(2) << stats.mean_gflops
                  << " ± " << std::setw(5) << stats.stddev_gflops << " GFLOPS                      ║\n";
        std::cout << "║   Bandwidth:        " << std::setw(10) << std::fixed << std::setprecision(2) << stats.bandwidth_gb << " GB/s                                 ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Range:                                                         ║\n";
        std::cout << "║   Min time:         " << std::setw(10) << std::fixed << std::setprecision(2) << stats.min_ms << " ms                                   ║\n";
        std::cout << "║   Max time:         " << std::setw(10) << std::fixed << std::setprecision(2) << stats.max_ms << " ms                                   ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    }

    /**
     * @brief Benchmark FP32 × FP32 GEMM
     */
    BenchmarkStats benchmarkFP32(const BenchmarkConfig &config)
    {
        // Allocate matrices
        std::vector<float> A(config.m * config.k);
        std::vector<float> B(config.k * config.n);
        std::vector<float> C(config.m * config.n);

        initFP32Matrix(A.data(), A.size());
        initFP32Matrix(B.data(), B.size());

        // Use OpenBLAS via cblas_sgemm (standard CPU GEMM)
        // Layout: Row-major (CblasRowMajor)
        // C = alpha * A * B + beta * C
        auto gemm_fn = [&]()
        {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        config.m, config.n, config.k,
                        1.0f,                // alpha
                        A.data(), config.k,  // A (m×k)
                        B.data(), config.n,  // B (k×n)
                        0.0f,                // beta
                        C.data(), config.n); // C (m×n)
        };

        return runBenchmark(config, gemm_fn, sizeof(float));
    }

    /**
     * @brief Benchmark Custom FP32 × FP32 GEMM using GemmKernelTemplate with AutoTuner
     */
    BenchmarkStats benchmarkCustomFP32(const BenchmarkConfig &config)
    {
        // Allocate matrices
        std::vector<float> A(config.m * config.k);
        std::vector<float> B_data(config.k * config.n);
        std::vector<float> C(config.m * config.n);

        initFP32Matrix(A.data(), A.size());
        initFP32Matrix(B_data.data(), B_data.size());

        // Create FP32Tensor for B matrix (weights)
        // B is K×N in row-major format, FP32Tensor provides ITensorGemmTileDataProvider interface
        FP32Tensor B_tensor({static_cast<size_t>(config.k), static_cast<size_t>(config.n)});
        std::memcpy(B_tensor.mutable_data(), B_data.data(), B_data.size() * sizeof(float));

        // Use AutoTuner to select optimal kernel configuration for this matrix size
        auto &tuner = llaminar::v2::kernels::GemmAutoTuner::instance();

        // Register all micro-kernel variants (only done once per test run)
        static bool variants_registered = false;
        if (!variants_registered)
        {
            if (rank_ == 0)
            {
                std::cout << "[Custom GEMM] Registering micro-kernel variants...\n";
            }
            auto variants = llaminar2::kernels::gemm::registerMicroKernelVariants(&B_tensor);
            for (auto &variant : variants)
            {
                tuner.registerVariant(std::move(variant));
            }
            if (rank_ == 0)
            {
                std::cout << "[Custom GEMM] Registered " << variants.size() << " variants\n";
            }
            variants_registered = true;
        }

        // Get optimal kernel for this matrix size
        auto *kernel = tuner.getOptimalKernel(config.m, config.n, config.k);
        if (!kernel && rank_ == 0)
        {
            std::cerr << "[Custom GEMM] Failed to get optimal kernel for shape ["
                      << config.m << ", " << config.n << ", " << config.k << "]\n";
        }

        // Log selected configuration
        if (kernel && rank_ == 0)
        {
            auto kernel_config = kernel->config();
            std::cout << "[Custom GEMM] M=" << config.m
                      << " selected: tile=" << kernel_config.tile_m << "x" << kernel_config.tile_n
                      << ", unroll=" << kernel_config.unroll_factor
                      << ", prefetch=" << kernel_config.prefetch_blocks << "\n";
        }

        auto gemm_fn = [&]()
        {
            if (kernel)
            {
                // Use auto-tuned kernel with FP32Tensor as decoder
                bool success = kernel->multiply(
                    A.data(),
                    C.data(),
                    config.m, config.n, config.k,
                    static_cast<const ITensorGemmTileDataProvider *>(&B_tensor), // Cast FP32Tensor to decoder interface
                    false,                                                       // transpose_B
                    1.0f,                                                        // alpha
                    0.0f                                                         // beta
                );

                if (!success && rank_ == 0)
                {
                    std::cerr << "[Custom GEMM] Kernel execution failed!\n";
                }
            }
            else if (rank_ == 0)
            {
                std::cerr << "[Custom GEMM] No kernel available!\n";
            }
        };

        return runBenchmark(config, gemm_fn, sizeof(float));
    }

    /**
     * @brief Benchmark BF16 × BF16 GEMM → FP32
     */
    BenchmarkStats benchmarkBF16(const BenchmarkConfig &config)
    {
        // Allocate FP32 source matrices
        std::vector<float> A_fp32(config.m * config.k);
        std::vector<float> B_fp32(config.k * config.n);
        initFP32Matrix(A_fp32.data(), A_fp32.size());
        initFP32Matrix(B_fp32.data(), B_fp32.size());

        // Convert to BF16
        std::vector<uint16_t> A_bf16(config.m * config.k);
        std::vector<uint16_t> B_bf16(config.k * config.n);
        convertToBF16(A_fp32.data(), A_bf16.data(), A_fp32.size());
        convertToBF16(B_fp32.data(), B_bf16.data(), B_fp32.size());

        // Output in FP32
        std::vector<float> C(config.m * config.n);

        // BF16 GEMM: Convert to FP32, compute, store
        auto gemm_fn = [&]()
        {
            // Convert BF16 → FP32
            llaminar2::simd::convert_bf16_to_fp32(A_bf16.data(), A_fp32.data(), A_bf16.size());
            llaminar2::simd::convert_bf16_to_fp32(B_bf16.data(), B_fp32.data(), B_bf16.size());

            // FP32 GEMM
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        config.m, config.n, config.k,
                        1.0f,
                        A_fp32.data(), config.k,
                        B_fp32.data(), config.n,
                        0.0f,
                        C.data(), config.n);
        };

        return runBenchmark(config, gemm_fn, sizeof(uint16_t));
    }

    /**
     * @brief Benchmark FP16 × FP16 GEMM → FP32
     */
    BenchmarkStats benchmarkFP16(const BenchmarkConfig &config)
    {
        // Allocate FP32 source matrices
        std::vector<float> A_fp32(config.m * config.k);
        std::vector<float> B_fp32(config.k * config.n);
        initFP32Matrix(A_fp32.data(), A_fp32.size());
        initFP32Matrix(B_fp32.data(), B_fp32.size());

        // Convert to FP16
        std::vector<uint16_t> A_fp16(config.m * config.k);
        std::vector<uint16_t> B_fp16(config.k * config.n);
        convertToFP16(A_fp32.data(), A_fp16.data(), A_fp32.size());
        convertToFP16(B_fp32.data(), B_fp16.data(), B_fp32.size());

        // Output in FP32
        std::vector<float> C(config.m * config.n);

        // FP16 GEMM: Convert to FP32, compute, store
        auto gemm_fn = [&]()
        {
            // Convert FP16 → FP32 (vectorized with AVX512/AVX2/F16C)
            llaminar2::simd::convert_fp16_to_fp32(A_fp16.data(), A_fp32.data(), A_fp16.size());
            llaminar2::simd::convert_fp16_to_fp32(B_fp16.data(), B_fp32.data(), B_fp16.size());

            // FP32 GEMM
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        config.m, config.n, config.k,
                        1.0f,
                        A_fp32.data(), config.k,
                        B_fp32.data(), config.n,
                        0.0f,
                        C.data(), config.n);
        };

        return runBenchmark(config, gemm_fn, sizeof(uint16_t));
    }

    /**
     * @brief Benchmark INT8 × INT8 GEMM → FP32
     */
    BenchmarkStats benchmarkINT8(const BenchmarkConfig &config)
    {
        // Allocate FP32 source matrices
        std::vector<float> A_fp32(config.m * config.k);
        std::vector<float> B_fp32(config.k * config.n);
        initFP32Matrix(A_fp32.data(), A_fp32.size());
        initFP32Matrix(B_fp32.data(), B_fp32.size());

        // Convert to INT8
        std::vector<int8_t> A_int8(config.m * config.k);
        std::vector<int8_t> B_int8(config.k * config.n);
        float scale = 127.0f;
        convertToINT8(A_fp32.data(), A_int8.data(), A_fp32.size(), scale);
        convertToINT8(B_fp32.data(), B_int8.data(), B_fp32.size(), scale);

        // Output in FP32
        std::vector<float> C(config.m * config.n);

        // INT8 GEMM: Dequantize to FP32, compute, store
        auto gemm_fn = [&]()
        {
            // Dequantize INT8 → FP32
            for (size_t i = 0; i < A_int8.size(); ++i)
            {
                A_fp32[i] = static_cast<float>(A_int8[i]) / scale;
            }
            for (size_t i = 0; i < B_int8.size(); ++i)
            {
                B_fp32[i] = static_cast<float>(B_int8[i]) / scale;
            }

            // FP32 GEMM
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        config.m, config.n, config.k,
                        1.0f,
                        A_fp32.data(), config.k,
                        B_fp32.data(), config.n,
                        0.0f,
                        C.data(), config.n);
        };

        return runBenchmark(config, gemm_fn, sizeof(int8_t));
    }

private:
    /**
     * @brief Generic benchmark runner with multiple trials
     */
    template <typename GemmFunc>
    BenchmarkStats runBenchmark(const BenchmarkConfig &config, GemmFunc gemm_fn, size_t element_bytes)
    {
        // Global warmup
        for (int i = 0; i < config.warmup_iters; ++i)
        {
            gemm_fn();
        }

        // Run multiple trials
        std::vector<double> trial_times_ms;
        trial_times_ms.reserve(config.num_trials);

        for (int trial = 0; trial < config.num_trials; ++trial)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            auto start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < config.bench_iters; ++i)
            {
                gemm_fn();
            }

            MPI_Barrier(MPI_COMM_WORLD);
            auto end = std::chrono::high_resolution_clock::now();

            double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
            double avg_time_ms = elapsed_ms / config.bench_iters;
            trial_times_ms.push_back(avg_time_ms);
        }

        // Calculate statistics
        BenchmarkStats stats;

        // Mean
        double sum = std::accumulate(trial_times_ms.begin(), trial_times_ms.end(), 0.0);
        stats.mean_ms = sum / trial_times_ms.size();

        // Standard deviation
        double sq_diff_sum = 0.0;
        for (double t : trial_times_ms)
        {
            double diff = t - stats.mean_ms;
            sq_diff_sum += diff * diff;
        }
        stats.stddev_ms = std::sqrt(sq_diff_sum / trial_times_ms.size());

        // Min/Max
        stats.min_ms = *std::min_element(trial_times_ms.begin(), trial_times_ms.end());
        stats.max_ms = *std::max_element(trial_times_ms.begin(), trial_times_ms.end());

        // GFLOPS
        stats.mean_gflops = calculateGFLOPS(config.m, config.n, config.k, stats.mean_ms);
        stats.stddev_gflops = calculateGFLOPS(config.m, config.n, config.k, stats.mean_ms - stats.stddev_ms) - stats.mean_gflops;

        // Bandwidth
        stats.bandwidth_gb = calculateBandwidth(config.m, config.n, config.k, element_bytes, stats.mean_ms);

        return stats;
    }
};

// =============================================================================
// Multi-Precision Comparison Tests
// =============================================================================

/**
 * @brief Compare all precisions at small batch size (typical decode)
 */
TEST_F(MultiPrecisionGemmPerf, SmallBatch_AllPrecisions)
{
    BenchmarkConfig config{
        .m = 32,  // Small batch
        .n = 896, // Qwen 2.5 0.5B d_model
        .k = 896,
        .warmup_iters = 5,
        .bench_iters = 100,
        .num_trials = 5,
        .description = "Small Batch (32×896×896)"};

    if (rank_ == 0)
    {
        std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║       MULTI-PRECISION GEMM COMPARISON - SMALL BATCH           ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
    }

    auto fp32_stats = benchmarkFP32(config);
    printResults(config, fp32_stats, "FP32");

    auto bf16_stats = benchmarkBF16(config);
    printResults(config, bf16_stats, "BF16");

    auto fp16_stats = benchmarkFP16(config);
    printResults(config, fp16_stats, "FP16");

    auto int8_stats = benchmarkINT8(config);
    printResults(config, int8_stats, "INT8");

    // Print comparison table (rank 0 only)
    if (rank_ == 0)
    {
        std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                    COMPARISON SUMMARY                          ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Precision │   Time (ms) │   GFLOPS │  Speedup │  Efficiency   ║\n";
        std::cout << "╠═══════════╪═════════════╪══════════╪══════════╪═══════════════╣\n";

        auto print_row = [](const std::string &name, const BenchmarkStats &stats, double baseline_ms)
        {
            double speedup = baseline_ms / stats.mean_ms;
            double efficiency = (stats.mean_gflops / (2.0 * 32 * 896 * 896 / stats.mean_ms / 1e6)) * 100.0;
            std::cout << "║ " << std::left << std::setw(9) << name << " │ "
                      << std::right << std::fixed << std::setprecision(2) << std::setw(11) << stats.mean_ms << " │ "
                      << std::setw(8) << stats.mean_gflops << " │ "
                      << std::setw(8) << speedup << "× │ "
                      << std::setw(13) << (speedup * 100.0) << "% ║\n";
        };

        print_row("FP32", fp32_stats, fp32_stats.mean_ms);
        print_row("BF16", bf16_stats, fp32_stats.mean_ms);
        print_row("FP16", fp16_stats, fp32_stats.mean_ms);
        print_row("INT8", int8_stats, fp32_stats.mean_ms);

        std::cout << "╚═══════════╧═════════════╧══════════╧══════════╧═══════════════╝\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * @brief Compare all precisions at medium batch size
 */
TEST_F(MultiPrecisionGemmPerf, MediumBatch_AllPrecisions)
{
    BenchmarkConfig config{
        .m = 128,
        .n = 896,
        .k = 896,
        .warmup_iters = 5,
        .bench_iters = 100,
        .num_trials = 5,
        .description = "Medium Batch (128×896×896)"};

    if (rank_ == 0)
    {
        std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║      MULTI-PRECISION GEMM COMPARISON - MEDIUM BATCH           ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
    }

    auto fp32_stats = benchmarkFP32(config);
    printResults(config, fp32_stats, "FP32");

    auto bf16_stats = benchmarkBF16(config);
    printResults(config, bf16_stats, "BF16");

    auto fp16_stats = benchmarkFP16(config);
    printResults(config, fp16_stats, "FP16");

    auto int8_stats = benchmarkINT8(config);
    printResults(config, int8_stats, "INT8");

    if (rank_ == 0)
    {
        std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                    COMPARISON SUMMARY                          ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Precision │   Time (ms) │   GFLOPS │  Speedup vs FP32         ║\n";
        std::cout << "╠═══════════╪═════════════╪══════════╪══════════════════════════╣\n";

        auto print_row = [](const std::string &name, const BenchmarkStats &stats, double baseline_ms)
        {
            double speedup = baseline_ms / stats.mean_ms;
            std::cout << "║ " << std::left << std::setw(9) << name << " │ "
                      << std::right << std::fixed << std::setprecision(2) << std::setw(11) << stats.mean_ms << " │ "
                      << std::setw(8) << stats.mean_gflops << " │ "
                      << std::setw(24) << speedup << "× ║\n";
        };

        print_row("FP32", fp32_stats, fp32_stats.mean_ms);
        print_row("BF16", bf16_stats, fp32_stats.mean_ms);
        print_row("FP16", fp16_stats, fp32_stats.mean_ms);
        print_row("INT8", int8_stats, fp32_stats.mean_ms);

        std::cout << "╚═══════════╧═════════════╧══════════╧══════════════════════════╝\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * @brief Compare all precisions at large batch size (prefill)
 */
TEST_F(MultiPrecisionGemmPerf, LargeBatch_AllPrecisions)
{
    BenchmarkConfig config{
        .m = 512,
        .n = 896,
        .k = 896,
        .warmup_iters = 3,
        .bench_iters = 50,
        .num_trials = 5,
        .description = "Large Batch (512×896×896, Prefill)"};

    if (rank_ == 0)
    {
        std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║       MULTI-PRECISION GEMM COMPARISON - LARGE BATCH           ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
    }

    auto fp32_stats = benchmarkFP32(config);
    printResults(config, fp32_stats, "FP32");

    auto bf16_stats = benchmarkBF16(config);
    printResults(config, bf16_stats, "BF16");

    auto fp16_stats = benchmarkFP16(config);
    printResults(config, fp16_stats, "FP16");

    auto int8_stats = benchmarkINT8(config);
    printResults(config, int8_stats, "INT8");

    if (rank_ == 0)
    {
        std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                    COMPARISON SUMMARY                          ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Precision │   Time (ms) │   GFLOPS │  Speedup vs FP32         ║\n";
        std::cout << "╠═══════════╪═════════════╪══════════╪══════════════════════════╣\n";

        auto print_row = [](const std::string &name, const BenchmarkStats &stats, double baseline_ms)
        {
            double speedup = baseline_ms / stats.mean_ms;
            std::cout << "║ " << std::left << std::setw(9) << name << " │ "
                      << std::right << std::fixed << std::setprecision(2) << std::setw(11) << stats.mean_ms << " │ "
                      << std::setw(8) << stats.mean_gflops << " │ "
                      << std::setw(24) << speedup << "× ║\n";
        };

        print_row("FP32", fp32_stats, fp32_stats.mean_ms);
        print_row("BF16", bf16_stats, fp32_stats.mean_ms);
        print_row("FP16", fp16_stats, fp32_stats.mean_ms);
        print_row("INT8", int8_stats, fp32_stats.mean_ms);

        std::cout << "╚═══════════╧═════════════╧══════════╧══════════════════════════╝\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * @brief FP32 GEMM scaling test to find performance plateau
 *
 * Tests FP32 GEMM throughput at increasing matrix sizes to identify:
 * - When does cache blocking become effective?
 * - At what size do we hit peak GFLOPS?
 * - Memory bandwidth vs compute-bound transition
 */
TEST_F(MultiPrecisionGemmPerf, FP32_ScalingToPlateauPrefill)
{
    if (rank_ == 0)
    {
        std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║          FP32 GEMM SCALING - FINDING PERFORMANCE PLATEAU       ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
    }

    // Test matrix sizes (M × K × K for square K=N, prefill-like)
    // Start small, increase geometrically to find plateau
    std::vector<int> batch_sizes = {
        32,   // Tiny (fits in L1)
        64,   // Small (L1/L2)
        128,  // Medium (L2)
        256,  // Large (L2/L3)
        512,  // Very large (L3)
        1024, // Huge (exceeds L3)
        2048, // Massive (memory-bound)
        4096  // Extreme (definitely memory-bound)
    };

    const int k = 896; // Typical model dimension (Qwen 2.5 0.5B)
    const int n = 896;

    struct ScalingResult
    {
        int m;
        BenchmarkStats stats;
        size_t total_bytes;
        double bandwidth_gb_s;
    };

    std::vector<ScalingResult> results;

    for (int m : batch_sizes)
    {
        BenchmarkConfig config{
            .m = m,
            .n = n,
            .k = k,
            .warmup_iters = 3,
            .bench_iters = (m <= 512) ? 50 : (m <= 1024) ? 30
                                                         : 20, // Fewer iters for huge sizes
            .num_trials = 5,
            .description = "Scaling Test (M=" + std::to_string(m) + ")"};

        auto stats = benchmarkFP32(config);

        // Calculate total memory footprint
        size_t a_bytes = static_cast<size_t>(m) * k * sizeof(float);
        size_t b_bytes = static_cast<size_t>(k) * n * sizeof(float);
        size_t c_bytes = static_cast<size_t>(m) * n * sizeof(float);
        size_t total_bytes = a_bytes + b_bytes + c_bytes;

        double bandwidth_gb_s = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / (stats.mean_ms / 1000.0);

        results.push_back({m, stats, total_bytes, bandwidth_gb_s});

        if (rank_ == 0)
        {
            std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
            std::cout << "║ FP32 GEMM - Batch Size M=" << std::setw(4) << m << "                             ║\n";
            std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ Configuration:                                                 ║\n";
            std::cout << "║   Matrix Dimensions: " << std::setw(4) << m << "   × " << std::setw(4) << k << "   × " << std::setw(4) << n << "                          ║\n";
            std::cout << "║   Memory Footprint:  " << std::setw(7) << std::fixed << std::setprecision(2) << (total_bytes / (1024.0 * 1024.0)) << " MB                                     ║\n";
            std::cout << "║   MPI Ranks:         " << world_size_ << "                                               ║\n";
            std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ Performance (mean ± stddev):                                   ║\n";

            double cv_percent = (stats.stddev_ms / stats.mean_ms) * 100.0;
            std::cout << "║   Time per iter:    " << std::setw(6) << std::fixed << std::setprecision(2) << stats.mean_ms
                      << "  ± " << std::setw(4) << std::setprecision(2) << stats.stddev_ms
                      << "  ms (CV: " << std::setw(3) << std::setprecision(1) << cv_percent << "%)            ║\n";
            std::cout << "║   Throughput:       " << std::setw(6) << std::fixed << std::setprecision(2) << stats.mean_gflops
                      << "  ± " << std::setw(5) << std::setprecision(2) << stats.stddev_gflops << " GFLOPS                      ║\n";
            std::cout << "║   Bandwidth:        " << std::setw(6) << std::fixed << std::setprecision(2) << bandwidth_gb_s << "     GB/s                                 ║\n";
            std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ Range:                                                         ║\n";
            std::cout << "║   Min time:         " << std::setw(6) << std::fixed << std::setprecision(2) << stats.min_ms << "     ms                                   ║\n";
            std::cout << "║   Max time:         " << std::setw(6) << std::fixed << std::setprecision(2) << stats.max_ms << "     ms                                   ║\n";
            std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Print summary table
    if (rank_ == 0)
    {
        std::cout << "\n╔════════════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                         FP32 SCALING SUMMARY                                   ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║  Batch │   Time (ms) │   GFLOPS │  Bandwidth │  Memory  │  Efficiency      ║\n";
        std::cout << "║   (M)  │             │          │   (GB/s)   │   (MB)   │  vs Peak         ║\n";
        std::cout << "╠════════╪═════════════╪══════════╪════════════╪══════════╪══════════════════╣\n";

        // Find peak GFLOPS for efficiency calculation
        double peak_gflops = 0.0;
        for (const auto &result : results)
        {
            peak_gflops = std::max(peak_gflops, result.stats.mean_gflops);
        }

        for (const auto &result : results)
        {
            double efficiency = (result.stats.mean_gflops / peak_gflops) * 100.0;
            double mem_mb = result.total_bytes / (1024.0 * 1024.0);

            std::cout << "║ " << std::right << std::setw(6) << result.m << " │ "
                      << std::setw(11) << std::fixed << std::setprecision(2) << result.stats.mean_ms << " │ "
                      << std::setw(8) << std::setprecision(2) << result.stats.mean_gflops << " │ "
                      << std::setw(10) << std::setprecision(2) << result.bandwidth_gb_s << " │ "
                      << std::setw(8) << std::setprecision(2) << mem_mb << " │ "
                      << std::setw(15) << std::setprecision(1) << efficiency << "% ║\n";
        }

        std::cout << "╚════════╧═════════════╧══════════╧════════════╧══════════╧══════════════════╝\n";

        // Analysis notes
        std::cout << "\nPerformance Analysis:\n";
        std::cout << "  Peak GFLOPS:     " << std::fixed << std::setprecision(2) << peak_gflops << "\n";

        // Find where we hit 95% of peak
        for (size_t i = 0; i < results.size(); ++i)
        {
            if (results[i].stats.mean_gflops >= 0.95 * peak_gflops)
            {
                std::cout << "  Plateau starts:  M=" << results[i].m << " (≥95% of peak)\n";
                break;
            }
        }

        std::cout << "  Memory footprint at plateau: "
                  << std::fixed << std::setprecision(2)
                  << (results.back().total_bytes / (1024.0 * 1024.0)) << " MB\n";
        std::cout << "\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * @brief Custom FP32 GEMM vs OpenBLAS comparison
 *
 * Tests our GemmKernelTemplate implementation against OpenBLAS
 * to measure performance gap and identify optimization opportunities.
 */
TEST_F(MultiPrecisionGemmPerf, CustomFP32_vs_OpenBLAS)
{
    if (rank_ == 0)
    {
        std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║      CUSTOM FP32 GEMM vs OPENBLAS - PERFORMANCE COMPARISON     ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
    }

    // Test matrix sizes matching the scaling test
    std::vector<int> batch_sizes = {32, 64, 128, 256, 512, 1024, 2048, 4096};
    const int k = 896;
    const int n = 896;

    struct ComparisonResult
    {
        int m;
        BenchmarkStats openblas_stats;
        BenchmarkStats custom_stats;
        double speedup; // custom vs openblas
    };

    std::vector<ComparisonResult> results;

    for (int m : batch_sizes)
    {
        BenchmarkConfig config{
            .m = m,
            .n = n,
            .k = k,
            .warmup_iters = 3,
            .bench_iters = (m <= 512) ? 50 : (m <= 1024) ? 30
                                                         : 20,
            .num_trials = 5,
            .description = "Custom vs OpenBLAS (M=" + std::to_string(m) + ")"};

        // Benchmark OpenBLAS
        auto openblas_stats = benchmarkFP32(config);

        // Benchmark our custom kernel
        auto custom_stats = benchmarkCustomFP32(config);

        double speedup = openblas_stats.mean_gflops / custom_stats.mean_gflops;
        results.push_back({m, openblas_stats, custom_stats, speedup});

        if (rank_ == 0)
        {
            std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
            std::cout << "║ Batch Size M=" << std::setw(4) << m << " - COMPARISON                          ║\n";
            std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ OpenBLAS:                                                      ║\n";
            std::cout << "║   Throughput:       " << std::setw(7) << std::fixed << std::setprecision(2) << openblas_stats.mean_gflops << " GFLOPS                               ║\n";
            std::cout << "║   Time per iter:    " << std::setw(7) << std::fixed << std::setprecision(2) << openblas_stats.mean_ms << " ms                                    ║\n";
            std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ Custom Kernel:                                                 ║\n";
            std::cout << "║   Throughput:       " << std::setw(7) << std::fixed << std::setprecision(2) << custom_stats.mean_gflops << " GFLOPS                               ║\n";
            std::cout << "║   Time per iter:    " << std::setw(7) << std::fixed << std::setprecision(2) << custom_stats.mean_ms << " ms                                    ║\n";
            std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ Performance Gap:                                               ║\n";
            std::cout << "║   OpenBLAS / Custom: " << std::setw(6) << std::fixed << std::setprecision(2) << speedup << "×                                       ║\n";

            double efficiency = (custom_stats.mean_gflops / openblas_stats.mean_gflops) * 100.0;
            std::cout << "║   Custom Efficiency: " << std::setw(6) << std::fixed << std::setprecision(1) << efficiency << "%                                      ║\n";
            std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Print summary table
    if (rank_ == 0)
    {
        std::cout << "\n╔══════════════════════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                      CUSTOM vs OPENBLAS SUMMARY                                          ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║  Batch │  OpenBLAS  │   Custom   │  Speedup  │  Custom  │    Gap Analysis           ║\n";
        std::cout << "║   (M)  │  (GFLOPS)  │  (GFLOPS)  │  (Ratio)  │  Eff (%) │                           ║\n";
        std::cout << "╠════════╪════════════╪════════════╪═══════════╪══════════╪═══════════════════════════╣\n";

        for (const auto &result : results)
        {
            double efficiency = (result.custom_stats.mean_gflops / result.openblas_stats.mean_gflops) * 100.0;
            std::string analysis;

            if (efficiency >= 95.0)
                analysis = "Excellent (≥95%)     ";
            else if (efficiency >= 80.0)
                analysis = "Good (80-95%)        ";
            else if (efficiency >= 60.0)
                analysis = "Moderate (60-80%)    ";
            else if (efficiency >= 40.0)
                analysis = "Poor (40-60%)        ";
            else
                analysis = "Very Poor (<40%)     ";

            std::cout << "║ " << std::right << std::setw(6) << result.m << " │ "
                      << std::setw(10) << std::fixed << std::setprecision(2) << result.openblas_stats.mean_gflops << " │ "
                      << std::setw(10) << std::setprecision(2) << result.custom_stats.mean_gflops << " │ "
                      << std::setw(9) << std::setprecision(3) << result.speedup << " │ "
                      << std::setw(8) << std::setprecision(1) << efficiency << " │ "
                      << analysis << "║\n";
        }

        std::cout << "╚════════╧════════════╧════════════╧═══════════╧══════════╧═══════════════════════════╝\n";

        // Calculate overall statistics
        double avg_efficiency = 0.0;
        for (const auto &result : results)
        {
            avg_efficiency += (result.custom_stats.mean_gflops / result.openblas_stats.mean_gflops);
        }
        avg_efficiency = (avg_efficiency / results.size()) * 100.0;

        std::cout << "\nOverall Performance Analysis:\n";
        std::cout << "  Average Custom Efficiency: " << std::fixed << std::setprecision(1) << avg_efficiency << "%\n";
        std::cout << "  Best Case Efficiency:      "
                  << std::fixed << std::setprecision(1)
                  << (*std::max_element(results.begin(), results.end(),
                                        [](const auto &a, const auto &b)
                                        {
                                            return (a.custom_stats.mean_gflops / a.openblas_stats.mean_gflops) <
                                                   (b.custom_stats.mean_gflops / b.openblas_stats.mean_gflops);
                                        }))
                             .custom_stats.mean_gflops /
                         (*std::max_element(results.begin(), results.end(),
                                            [](const auto &a, const auto &b)
                                            {
                                                return (a.custom_stats.mean_gflops / a.openblas_stats.mean_gflops) <
                                                       (b.custom_stats.mean_gflops / b.openblas_stats.mean_gflops);
                                            }))
                             .openblas_stats.mean_gflops *
                         100.0
                  << "% (M="
                  << (*std::max_element(results.begin(), results.end(),
                                        [](const auto &a, const auto &b)
                                        {
                                            return (a.custom_stats.mean_gflops / a.openblas_stats.mean_gflops) <
                                                   (b.custom_stats.mean_gflops / b.openblas_stats.mean_gflops);
                                        }))
                         .m
                  << ")\n";
        std::cout << "  Worst Case Efficiency:     "
                  << std::fixed << std::setprecision(1)
                  << (*std::min_element(results.begin(), results.end(),
                                        [](const auto &a, const auto &b)
                                        {
                                            return (a.custom_stats.mean_gflops / a.openblas_stats.mean_gflops) <
                                                   (b.custom_stats.mean_gflops / b.openblas_stats.mean_gflops);
                                        }))
                             .custom_stats.mean_gflops /
                         (*std::min_element(results.begin(), results.end(),
                                            [](const auto &a, const auto &b)
                                            {
                                                return (a.custom_stats.mean_gflops / a.openblas_stats.mean_gflops) <
                                                       (b.custom_stats.mean_gflops / b.openblas_stats.mean_gflops);
                                            }))
                             .openblas_stats.mean_gflops *
                         100.0
                  << "% (M="
                  << (*std::min_element(results.begin(), results.end(),
                                        [](const auto &a, const auto &b)
                                        {
                                            return (a.custom_stats.mean_gflops / a.openblas_stats.mean_gflops) <
                                                   (b.custom_stats.mean_gflops / b.openblas_stats.mean_gflops);
                                        }))
                         .m
                  << ")\n";
        std::cout << "\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

// Main function for standalone execution
int main(int argc, char **argv)
{
    // Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    // Initialize GTest
    ::testing::InitGoogleTest(&argc, argv);

    // Run tests
    int result = RUN_ALL_TESTS();

    // Finalize MPI
    MPI_Finalize();

    return result;
}
