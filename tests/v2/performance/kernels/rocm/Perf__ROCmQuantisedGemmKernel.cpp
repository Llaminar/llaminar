/**
 * @file Perf__ROCmQuantisedGemmKernel.cpp
 * @brief Performance benchmark for ROCm INT8 Quantised GEMM (CK DeviceGemmMultipleD_Dl)
 *
 * This test benchmarks the ROCmQuantisedGemmKernel which uses AMD ComposableKernel (CK)
 * for INT8×INT8→INT32 GEMM operations on gfx906 (MI50/MI60) GPUs.
 *
 * It measures:
 *   - End-to-end throughput (TFLOPS) including quantization overhead
 *   - Kernel-only time (CK GEMM execution)
 *   - Activation quantization time
 *   - Weight packing time (amortized)
 *   - Cosine similarity accuracy vs FP32 reference
 *
 * Test configurations cover realistic Qwen model dimensions:
 *   - Qwen2.5-0.5B: hidden=896, intermediate=4864
 *   - Qwen2.5-7B:   hidden=3584, intermediate=18944
 *   - Qwen2.5-14B:  hidden=5120, intermediate=13824
 *   - Qwen2.5-32B:  hidden=5120, intermediate=27648
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>

#include "kernels/rocm/ROCmQuantisedGemmKernel.h"
#include "tensors/Tensors.h"
#include "../../../utils/TestTensorFactory.h"
#include "utils/Logger.h"

#ifdef HAVE_ONEDNN
#include "kernels/cpu/gemm_v4/FloatingPointGemmKernel.h"
#endif

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

using namespace llaminar2;
using namespace llaminar2::rocm;
using namespace llaminar2::test;

// ============================================================================
// Benchmark Configuration
// ============================================================================

/**
 * @brief Configuration for a single benchmark run
 */
struct ROCmBenchConfig
{
    std::string name; ///< Human-readable name (e.g., "Qwen-7B FFN Up")
    int M;            ///< Batch/sequence dimension
    int N;            ///< Output features (weight rows)
    int K;            ///< Input features (weight cols)
    int warmup_iters; ///< Warmup iterations (not timed)
    int bench_iters;  ///< Timed benchmark iterations
    int num_trials;   ///< Independent trials for statistics
};

/**
 * @brief Statistics from benchmark run
 */
struct ROCmBenchStats
{
    // Timing
    double mean_ms;   ///< Mean time per GEMM (ms)
    double min_ms;    ///< Minimum time (ms)
    double max_ms;    ///< Maximum time (ms)
    double stddev_ms; ///< Standard deviation (ms)

    // Performance
    double mean_tflops; ///< Mean throughput (TFLOPS)
    double peak_tflops; ///< Peak throughput (from min_ms)

    // Accuracy
    double cosine_sim; ///< Cosine similarity vs FP32 reference

    // Breakdown (optional)
    double quant_time_ms; ///< Activation quantization time (if measured)
    double gemm_time_ms;  ///< CK GEMM kernel time (if measured)
};

// ============================================================================
// Performance Test Fixture
// ============================================================================

class ROCmQuantisedGemmPerf : public ::testing::Test
{
protected:
    bool has_rocm_device_ = false;
    std::string device_name_;

    void SetUp() override
    {
#ifdef HAVE_ROCM
        int device_count = 0;
        hipError_t err = hipGetDeviceCount(&device_count);
        has_rocm_device_ = (err == hipSuccess && device_count > 0);

        if (has_rocm_device_)
        {
            hipDeviceProp_t props;
            (void)hipGetDeviceProperties(&props, 0);
            device_name_ = std::string(props.name) + " (" + props.gcnArchName + ")";
        }
#endif
    }

    /**
     * @brief Compute cosine similarity between two float arrays
     */
    double computeCosineSimilarity(const float *a, const float *b, size_t n)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += static_cast<double>(a[i]) * b[i];
            norm_a += static_cast<double>(a[i]) * a[i];
            norm_b += static_cast<double>(b[i]) * b[i];
        }
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-12);
    }

    /**
     * @brief Compute FP32 reference GEMM using OneDNN
     */
    void computeReference(const float *A, const float *B, float *C, int M, int N, int K)
    {
#ifdef HAVE_ONEDNN
        // OneDNN: A[M,K] × B^T[K,N] where B stored as [N,K]
        gemm_v4::run_onednn_fp32_matmul(A, B, C, M, N, K, true, 1.0f, 0.0f);
#else
        // Naive fallback
        for (int m = 0; m < M; ++m)
        {
            for (int n = 0; n < N; ++n)
            {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k)
                {
                    sum += A[m * K + k] * B[n * K + k];
                }
                C[m * N + n] = sum;
            }
        }
#endif
    }

    /**
     * @brief Run benchmark with given configuration
     */
    ROCmBenchStats runBenchmark(const ROCmBenchConfig &config)
    {
        const int M = config.M;
        const int N = config.N;
        const int K = config.K;

        // 1. Create random Q8_0 weights [N × K] and pack for ROCm
        auto weights = TestTensorFactory::createQ8_0Random({static_cast<size_t>(N), static_cast<size_t>(K)});
        ROCmPackedWeights packed;
        EXPECT_TRUE(packWeightsToROCm(weights.get(), packed));

        // 2. Create kernel
        ROCmQuantisedGemmKernel kernel(&packed, 0);

        // 3. Create random FP32 activations [M × K]
        auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});
        auto output = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});

        // 4. Compute reference for accuracy check
        std::vector<float> reference(M * N);
        computeReference(input->data(), weights->fp32_data(), reference.data(), M, N, K);

        // 5. Warmup
        for (int i = 0; i < config.warmup_iters; ++i)
        {
            kernel.multiply_tensor(input.get(), output.get(), M, N, K);
        }
        (void)hipDeviceSynchronize();

        // 6. Benchmark trials
        std::vector<double> trial_times_ms;
        trial_times_ms.reserve(config.num_trials);

        for (int t = 0; t < config.num_trials; ++t)
        {
            (void)hipDeviceSynchronize();
            auto start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < config.bench_iters; ++i)
            {
                kernel.multiply_tensor(input.get(), output.get(), M, N, K);
            }
            (void)hipDeviceSynchronize();

            auto end = std::chrono::high_resolution_clock::now();
            double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
            trial_times_ms.push_back(total_ms / config.bench_iters);
        }

        // 7. Compute accuracy
        double cosine_sim = computeCosineSimilarity(output->data(), reference.data(), M * N);

        // 8. Calculate statistics
        ROCmBenchStats stats;

        double sum = std::accumulate(trial_times_ms.begin(), trial_times_ms.end(), 0.0);
        stats.mean_ms = sum / config.num_trials;

        double sq_sum = std::inner_product(trial_times_ms.begin(), trial_times_ms.end(),
                                           trial_times_ms.begin(), 0.0);
        stats.stddev_ms = std::sqrt(sq_sum / config.num_trials - stats.mean_ms * stats.mean_ms);

        stats.min_ms = *std::min_element(trial_times_ms.begin(), trial_times_ms.end());
        stats.max_ms = *std::max_element(trial_times_ms.begin(), trial_times_ms.end());

        // TFLOPS = 2 * M * N * K / (time_s * 1e12)
        double ops = 2.0 * M * N * K;
        stats.mean_tflops = (ops / (stats.mean_ms * 1e-3)) / 1e12;
        stats.peak_tflops = (ops / (stats.min_ms * 1e-3)) / 1e12;

        stats.cosine_sim = cosine_sim;
        stats.quant_time_ms = 0.0; // Not measured separately in this version
        stats.gemm_time_ms = 0.0;

        return stats;
    }

    /**
     * @brief Print benchmark results in tabular format
     */
    void printResults(const ROCmBenchConfig &config, const ROCmBenchStats &stats)
    {
        std::cout << std::left << std::setw(28) << config.name
                  << " | M=" << std::setw(4) << config.M
                  << " N=" << std::setw(6) << config.N
                  << " K=" << std::setw(6) << config.K
                  << " | " << std::fixed << std::setprecision(3) << std::setw(7) << stats.mean_ms << " ms"
                  << " (±" << std::setprecision(3) << std::setw(5) << stats.stddev_ms << ")"
                  << " | " << std::setprecision(3) << std::setw(6) << stats.mean_tflops << " TFLOPS"
                  << " (pk " << std::setprecision(3) << stats.peak_tflops << ")"
                  << " | cos=" << std::setprecision(6) << stats.cosine_sim
                  << std::endl;
    }

    /**
     * @brief Print header for benchmark output
     */
    void printHeader(const std::string &section)
    {
        std::cout << "\n"
                  << "╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n"
                  << "║  " << std::left << std::setw(106) << section << "║\n"
                  << "║  Device: " << std::left << std::setw(98) << device_name_ << "║\n"
                  << "╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════╣\n"
                  << "║  " << std::left << std::setw(26) << "Workload"
                  << " | " << std::setw(25) << "Dimensions"
                  << " | " << std::setw(20) << "Time (ms)"
                  << " | " << std::setw(20) << "Throughput"
                  << " | " << std::setw(10) << "Accuracy"
                  << "║\n"
                  << "╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════╣"
                  << std::endl;
    }

    void printFooter()
    {
        std::cout << "╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n"
                  << std::endl;
    }

    /**
     * @brief Run a suite of benchmarks for a model size
     */
    void runModelBenchmarks(const std::string &model_name,
                            int hidden_size,
                            int intermediate_size,
                            const std::vector<int> &batch_sizes)
    {
        printHeader("ROCm INT8 GEMM: " + model_name);

        for (int M : batch_sizes)
        {
            // Attention Output Projection: [M, hidden] → [M, hidden]
            {
                ROCmBenchConfig cfg{
                    .name = "Attn Output",
                    .M = M,
                    .N = hidden_size,
                    .K = hidden_size,
                    .warmup_iters = 3,
                    .bench_iters = 10,
                    .num_trials = 5};
                auto stats = runBenchmark(cfg);
                printResults(cfg, stats);
                EXPECT_GT(stats.cosine_sim, 0.99);
            }

            // FFN Gate/Up Projection: [M, hidden] → [M, intermediate]
            {
                ROCmBenchConfig cfg{
                    .name = "FFN Gate/Up",
                    .M = M,
                    .N = intermediate_size,
                    .K = hidden_size,
                    .warmup_iters = 3,
                    .bench_iters = 10,
                    .num_trials = 5};
                auto stats = runBenchmark(cfg);
                printResults(cfg, stats);
                EXPECT_GT(stats.cosine_sim, 0.99);
            }

            // FFN Down Projection: [M, intermediate] → [M, hidden]
            {
                ROCmBenchConfig cfg{
                    .name = "FFN Down",
                    .M = M,
                    .N = hidden_size,
                    .K = intermediate_size,
                    .warmup_iters = 3,
                    .bench_iters = 10,
                    .num_trials = 5};
                auto stats = runBenchmark(cfg);
                printResults(cfg, stats);
                EXPECT_GT(stats.cosine_sim, 0.99);
            }
        }

        printFooter();
    }
};

// ============================================================================
// Performance Tests: Qwen Model Family (Individual Tests for gtest filtering)
// ============================================================================

// Model dimension constants
namespace QwenDims
{
    // Qwen2.5-0.5B: hidden=896, intermediate=4864
    constexpr int H_0_5B = 896;
    constexpr int I_0_5B = 4864;
    // Qwen2.5-7B: hidden=3584, intermediate=18944
    constexpr int H_7B = 3584;
    constexpr int I_7B = 18944;
    // Qwen2.5-14B: hidden=5120, intermediate=13824
    constexpr int H_14B = 5120;
    constexpr int I_14B = 13824;
    // Qwen2.5-32B: hidden=5120, intermediate=27648
    constexpr int H_32B = 5120;
    constexpr int I_32B = 27648;
}

// ----------------------------------------------------------------------------
// Qwen2.5-0.5B: Individual batch size tests
// Run with: --gtest_filter="*Qwen0_5B*" or "*Qwen0_5B*M1*" etc.
// ----------------------------------------------------------------------------

TEST_F(ROCmQuantisedGemmPerf, Qwen0_5B_M1)
{
    if (!has_rocm_device_)
        GTEST_SKIP() << "No ROCm device";
    runModelBenchmarks("Qwen2.5-0.5B", QwenDims::H_0_5B, QwenDims::I_0_5B, {1});
}

TEST_F(ROCmQuantisedGemmPerf, Qwen0_5B_M8)
{
    if (!has_rocm_device_)
        GTEST_SKIP() << "No ROCm device";
    runModelBenchmarks("Qwen2.5-0.5B", QwenDims::H_0_5B, QwenDims::I_0_5B, {8});
}

TEST_F(ROCmQuantisedGemmPerf, Qwen0_5B_M32)
{
    if (!has_rocm_device_)
        GTEST_SKIP() << "No ROCm device";
    runModelBenchmarks("Qwen2.5-0.5B", QwenDims::H_0_5B, QwenDims::I_0_5B, {32});
}

TEST_F(ROCmQuantisedGemmPerf, Qwen0_5B_M128)
{
    if (!has_rocm_device_)
        GTEST_SKIP() << "No ROCm device";
    runModelBenchmarks("Qwen2.5-0.5B", QwenDims::H_0_5B, QwenDims::I_0_5B, {128});
}

TEST_F(ROCmQuantisedGemmPerf, Qwen0_5B_M512)
{
    if (!has_rocm_device_)
        GTEST_SKIP() << "No ROCm device";
    runModelBenchmarks("Qwen2.5-0.5B", QwenDims::H_0_5B, QwenDims::I_0_5B, {512});
}

// ----------------------------------------------------------------------------
// Qwen2.5-7B: Individual batch size tests
// Run with: --gtest_filter="*Qwen7B*" or "*Qwen7B*M1*" etc.
// ----------------------------------------------------------------------------

TEST_F(ROCmQuantisedGemmPerf, Qwen7B_M1)
{
    if (!has_rocm_device_)
        GTEST_SKIP() << "No ROCm device";
    runModelBenchmarks("Qwen2.5-7B", QwenDims::H_7B, QwenDims::I_7B, {1});
}

TEST_F(ROCmQuantisedGemmPerf, Qwen7B_M8)
{
    if (!has_rocm_device_)
        GTEST_SKIP() << "No ROCm device";
    runModelBenchmarks("Qwen2.5-7B", QwenDims::H_7B, QwenDims::I_7B, {8});
}

TEST_F(ROCmQuantisedGemmPerf, Qwen7B_M32)
{
    if (!has_rocm_device_)
        GTEST_SKIP() << "No ROCm device";
    runModelBenchmarks("Qwen2.5-7B", QwenDims::H_7B, QwenDims::I_7B, {32});
}

TEST_F(ROCmQuantisedGemmPerf, Qwen7B_M128)
{
    if (!has_rocm_device_)
        GTEST_SKIP() << "No ROCm device";
    runModelBenchmarks("Qwen2.5-7B", QwenDims::H_7B, QwenDims::I_7B, {128});
}

TEST_F(ROCmQuantisedGemmPerf, Qwen7B_M512)
{
    if (!has_rocm_device_)
        GTEST_SKIP() << "No ROCm device";
    runModelBenchmarks("Qwen2.5-7B", QwenDims::H_7B, QwenDims::I_7B, {512});
}

// ----------------------------------------------------------------------------
// Qwen2.5-14B: Individual batch size tests
// Run with: --gtest_filter="*Qwen14B*" or "*Qwen14B*M1*" etc.
// ----------------------------------------------------------------------------

TEST_F(ROCmQuantisedGemmPerf, Qwen14B_M1)
{
    if (!has_rocm_device_)
        GTEST_SKIP() << "No ROCm device";
    runModelBenchmarks("Qwen2.5-14B", QwenDims::H_14B, QwenDims::I_14B, {1});
}

TEST_F(ROCmQuantisedGemmPerf, Qwen14B_M8)
{
    if (!has_rocm_device_)
        GTEST_SKIP() << "No ROCm device";
    runModelBenchmarks("Qwen2.5-14B", QwenDims::H_14B, QwenDims::I_14B, {8});
}

TEST_F(ROCmQuantisedGemmPerf, Qwen14B_M32)
{
    if (!has_rocm_device_)
        GTEST_SKIP() << "No ROCm device";
    runModelBenchmarks("Qwen2.5-14B", QwenDims::H_14B, QwenDims::I_14B, {32});
}

TEST_F(ROCmQuantisedGemmPerf, Qwen14B_M128)
{
    if (!has_rocm_device_)
        GTEST_SKIP() << "No ROCm device";
    runModelBenchmarks("Qwen2.5-14B", QwenDims::H_14B, QwenDims::I_14B, {128});
}

TEST_F(ROCmQuantisedGemmPerf, Qwen14B_M512)
{
    if (!has_rocm_device_)
        GTEST_SKIP() << "No ROCm device";
    runModelBenchmarks("Qwen2.5-14B", QwenDims::H_14B, QwenDims::I_14B, {512});
}

// ----------------------------------------------------------------------------
// Qwen2.5-32B: Individual batch size tests
// Run with: --gtest_filter="*Qwen32B*" or "*Qwen32B*M1*" etc.
// ----------------------------------------------------------------------------

TEST_F(ROCmQuantisedGemmPerf, Qwen32B_M1)
{
    if (!has_rocm_device_)
        GTEST_SKIP() << "No ROCm device";
    runModelBenchmarks("Qwen2.5-32B", QwenDims::H_32B, QwenDims::I_32B, {1});
}

TEST_F(ROCmQuantisedGemmPerf, Qwen32B_M8)
{
    if (!has_rocm_device_)
        GTEST_SKIP() << "No ROCm device";
    runModelBenchmarks("Qwen2.5-32B", QwenDims::H_32B, QwenDims::I_32B, {8});
}

TEST_F(ROCmQuantisedGemmPerf, Qwen32B_M32)
{
    if (!has_rocm_device_)
        GTEST_SKIP() << "No ROCm device";
    runModelBenchmarks("Qwen2.5-32B", QwenDims::H_32B, QwenDims::I_32B, {32});
}

TEST_F(ROCmQuantisedGemmPerf, Qwen32B_M128)
{
    if (!has_rocm_device_)
        GTEST_SKIP() << "No ROCm device";
    runModelBenchmarks("Qwen2.5-32B", QwenDims::H_32B, QwenDims::I_32B, {128});
}

// ----------------------------------------------------------------------------
// Aggregate tests (run all batch sizes for a model)
// Run with: --gtest_filter="*Qwen7B_All*" etc.
// ----------------------------------------------------------------------------

TEST_F(ROCmQuantisedGemmPerf, Qwen0_5B_All)
{
    if (!has_rocm_device_)
        GTEST_SKIP() << "No ROCm device";
    runModelBenchmarks("Qwen2.5-0.5B (hidden=896, inter=4864)",
                       QwenDims::H_0_5B, QwenDims::I_0_5B, {1, 8, 32, 128, 512});
}

TEST_F(ROCmQuantisedGemmPerf, Qwen7B_All)
{
    if (!has_rocm_device_)
        GTEST_SKIP() << "No ROCm device";
    runModelBenchmarks("Qwen2.5-7B (hidden=3584, inter=18944)",
                       QwenDims::H_7B, QwenDims::I_7B, {1, 8, 32, 128, 512});
}

TEST_F(ROCmQuantisedGemmPerf, Qwen14B_All)
{
    if (!has_rocm_device_)
        GTEST_SKIP() << "No ROCm device";
    runModelBenchmarks("Qwen2.5-14B (hidden=5120, inter=13824)",
                       QwenDims::H_14B, QwenDims::I_14B, {1, 8, 32, 128, 512});
}

TEST_F(ROCmQuantisedGemmPerf, Qwen32B_All)
{
    if (!has_rocm_device_)
        GTEST_SKIP() << "No ROCm device";
    runModelBenchmarks("Qwen2.5-32B (hidden=5120, inter=27648)",
                       QwenDims::H_32B, QwenDims::I_32B, {1, 8, 32, 128});
}

// ============================================================================
// Micro-benchmarks: Individual Operations
// ============================================================================

/**
 * @test Weight packing throughput
 *
 * Measures time to pack Q8_0 weights for ROCm CK GEMM.
 * This is typically done once per model load, so amortized cost is low.
 */
TEST_F(ROCmQuantisedGemmPerf, WeightPacking_Throughput)
{
    if (!has_rocm_device_)
    {
        GTEST_SKIP() << "No ROCm device available";
    }

    std::cout << "\n╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n"
              << "║  Weight Packing Throughput (Q8_0 → INT8 + scales)                                                             ║\n"
              << "╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════╣"
              << std::endl;

    struct PackingTest
    {
        std::string name;
        int N, K;
    };

    std::vector<PackingTest> tests = {
        {"0.5B Attn Out", 896, 896},
        {"0.5B FFN Up", 4864, 896},
        {"0.5B FFN Down", 896, 4864},
        {"7B Attn Out", 3584, 3584},
        {"7B FFN Up", 18944, 3584},
        {"7B FFN Down", 3584, 18944},
        {"14B Attn Out", 5120, 5120},
        {"14B FFN Up", 13824, 5120},
        {"32B FFN Up", 27648, 5120},
    };

    for (const auto &test : tests)
    {
        auto weights = TestTensorFactory::createQ8_0Random({static_cast<size_t>(test.N), static_cast<size_t>(test.K)});

        // Warmup
        ROCmPackedWeights packed_warmup;
        packWeightsToROCm(weights.get(), packed_warmup);

        // Timed run
        const int iters = 5;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i)
        {
            ROCmPackedWeights packed;
            packWeightsToROCm(weights.get(), packed);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;

        double bytes = test.N * test.K * sizeof(int8_t) + test.N * sizeof(float);
        double gb_s = (bytes / 1e9) / (ms / 1e3);

        std::cout << "║  " << std::left << std::setw(14) << test.name
                  << " | N=" << std::setw(6) << test.N << " K=" << std::setw(6) << test.K
                  << " | " << std::fixed << std::setprecision(3) << std::setw(7) << ms << " ms"
                  << " | " << std::setprecision(2) << std::setw(6) << gb_s << " GB/s"
                  << "                                   ║"
                  << std::endl;
    }

    std::cout << "╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n"
              << std::endl;
}

/**
 * @test Decode throughput sweep
 *
 * Measures GEMM throughput for decode (M=1) across various model sizes.
 * This is the critical path for autoregressive generation.
 */
TEST_F(ROCmQuantisedGemmPerf, Decode_M1_Sweep)
{
    if (!has_rocm_device_)
    {
        GTEST_SKIP() << "No ROCm device available";
    }

    printHeader("Decode (M=1) Throughput Sweep");

    struct DecodeTest
    {
        std::string name;
        int N, K;
    };

    std::vector<DecodeTest> tests = {
        // Qwen2.5-0.5B
        {"0.5B Attn Out", 896, 896},
        {"0.5B FFN Up", 4864, 896},
        {"0.5B FFN Down", 896, 4864},
        // Qwen2.5-7B
        {"7B Attn Out", 3584, 3584},
        {"7B FFN Up", 18944, 3584},
        {"7B FFN Down", 3584, 18944},
        // Qwen2.5-14B
        {"14B Attn Out", 5120, 5120},
        {"14B FFN Up", 13824, 5120},
        {"14B FFN Down", 5120, 13824},
        // Qwen2.5-32B
        {"32B Attn Out", 5120, 5120},
        {"32B FFN Up", 27648, 5120},
        {"32B FFN Down", 5120, 27648},
    };

    for (const auto &test : tests)
    {
        ROCmBenchConfig cfg{
            .name = test.name,
            .M = 1,
            .N = test.N,
            .K = test.K,
            .warmup_iters = 5,
            .bench_iters = 20,
            .num_trials = 5};
        auto stats = runBenchmark(cfg);
        printResults(cfg, stats);
        EXPECT_GT(stats.cosine_sim, 0.99);
    }

    printFooter();
}

/**
 * @test Prefill throughput sweep
 *
 * Measures GEMM throughput for prefill (M=128, M=512) across model sizes.
 * This is the critical path for prompt processing.
 */
TEST_F(ROCmQuantisedGemmPerf, Prefill_M128_M512_Sweep)
{
    if (!has_rocm_device_)
    {
        GTEST_SKIP() << "No ROCm device available";
    }

    printHeader("Prefill (M=128, M=512) Throughput Sweep");

    struct PrefillTest
    {
        std::string name;
        int M, N, K;
    };

    std::vector<PrefillTest> tests = {
        // Qwen2.5-0.5B - M=128
        {"0.5B FFN Up M=128", 128, 4864, 896},
        {"0.5B FFN Down M=128", 128, 896, 4864},
        // Qwen2.5-0.5B - M=512
        {"0.5B FFN Up M=512", 512, 4864, 896},
        {"0.5B FFN Down M=512", 512, 896, 4864},
        // Qwen2.5-7B - M=128
        {"7B FFN Up M=128", 128, 18944, 3584},
        {"7B FFN Down M=128", 128, 3584, 18944},
        // Qwen2.5-7B - M=512
        {"7B FFN Up M=512", 512, 18944, 3584},
        {"7B FFN Down M=512", 512, 3584, 18944},
        // Qwen2.5-14B - M=128
        {"14B FFN Up M=128", 128, 13824, 5120},
        {"14B FFN Down M=128", 128, 5120, 13824},
    };

    for (const auto &test : tests)
    {
        ROCmBenchConfig cfg{
            .name = test.name,
            .M = test.M,
            .N = test.N,
            .K = test.K,
            .warmup_iters = 3,
            .bench_iters = 10,
            .num_trials = 5};
        auto stats = runBenchmark(cfg);
        printResults(cfg, stats);
        EXPECT_GT(stats.cosine_sim, 0.99);
    }

    printFooter();
}

/**
 * @test Batch size scaling
 *
 * Measures how throughput scales with batch size for a fixed workload.
 * Useful for understanding GPU utilization characteristics.
 */
TEST_F(ROCmQuantisedGemmPerf, BatchSizeScaling)
{
    if (!has_rocm_device_)
    {
        GTEST_SKIP() << "No ROCm device available";
    }

    printHeader("Batch Size Scaling: Qwen-7B FFN Up (K=3584 → N=18944)");

    std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};

    for (int M : batch_sizes)
    {
        ROCmBenchConfig cfg{
            .name = "7B FFN Up",
            .M = M,
            .N = 18944,
            .K = 3584,
            .warmup_iters = 3,
            .bench_iters = 10,
            .num_trials = 5};
        auto stats = runBenchmark(cfg);
        printResults(cfg, stats);
        EXPECT_GT(stats.cosine_sim, 0.99);
    }

    printFooter();
}

// ============================================================================
// Accuracy Tests
// ============================================================================

/**
 * @test Accuracy sweep across dimensions
 *
 * Verifies cosine similarity remains high across all tested dimensions.
 */
TEST_F(ROCmQuantisedGemmPerf, AccuracySweep)
{
    if (!has_rocm_device_)
    {
        GTEST_SKIP() << "No ROCm device available";
    }

    std::cout << "\n╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n"
              << "║  Accuracy Sweep: Cosine Similarity vs FP32 Reference                                                          ║\n"
              << "╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════╣"
              << std::endl;

    struct AccuracyTest
    {
        std::string name;
        int M, N, K;
        double min_expected_cosine;
    };

    std::vector<AccuracyTest> tests = {
        // Small dimensions (potentially harder for INT8)
        {"Small 32×64×128", 32, 64, 128, 0.995},
        {"Small 64×128×256", 64, 128, 256, 0.995},
        // Model-realistic dimensions
        {"0.5B FFN Up", 64, 4864, 896, 0.995},
        {"0.5B FFN Down", 64, 896, 4864, 0.995},
        {"7B FFN Up", 64, 18944, 3584, 0.995},
        {"7B FFN Down", 64, 3584, 18944, 0.995},
        {"14B FFN Up", 64, 13824, 5120, 0.995},
        {"32B FFN Up", 32, 27648, 5120, 0.995},
    };

    for (const auto &test : tests)
    {
        ROCmBenchConfig cfg{
            .name = test.name,
            .M = test.M,
            .N = test.N,
            .K = test.K,
            .warmup_iters = 1,
            .bench_iters = 1,
            .num_trials = 1};
        auto stats = runBenchmark(cfg);

        std::cout << "║  " << std::left << std::setw(20) << test.name
                  << " | M=" << std::setw(4) << test.M
                  << " N=" << std::setw(6) << test.N
                  << " K=" << std::setw(6) << test.K
                  << " | cosine=" << std::fixed << std::setprecision(6) << stats.cosine_sim
                  << " (min=" << test.min_expected_cosine << ")"
                  << "                  ║"
                  << std::endl;

        EXPECT_GT(stats.cosine_sim, test.min_expected_cosine)
            << "Cosine similarity too low for " << test.name;
    }

    std::cout << "╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n"
              << std::endl;
}
