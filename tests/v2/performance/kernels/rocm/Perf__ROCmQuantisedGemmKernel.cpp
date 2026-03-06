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
#include <omp.h>

#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "tensors/Tensors.h"
#include "../../../utils/TestTensorFactory.h"
#include "utils/DebugEnv.h"
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
    std::string name;               ///< Human-readable name (e.g., "Qwen-7B FFN Up")
    int M;                          ///< Batch/sequence dimension
    int N;                          ///< Output features (weight rows)
    int K;                          ///< Input features (weight cols)
    int warmup_iters;               ///< Warmup iterations (not timed)
    int bench_iters;                ///< Timed benchmark iterations
    int num_trials;                 ///< Independent trials for statistics
    bool end_to_end_timing = false; ///< If true, time full multiply_tensor path (quantize + GEMM + scaling)
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

    class ScopedEnvOverride
    {
    public:
        ScopedEnvOverride(const char *name, const std::string &value)
            : name_(name)
        {
            const char *existing = std::getenv(name_);
            had_original_ = (existing != nullptr);
            if (had_original_)
            {
                original_value_ = existing;
            }
            ::setenv(name_, value.c_str(), 1);
            mutableDebugEnv().reload();
        }

        ~ScopedEnvOverride()
        {
            if (had_original_)
            {
                ::setenv(name_, original_value_.c_str(), 1);
            }
            else
            {
                ::unsetenv(name_);
            }
            mutableDebugEnv().reload();
        }

        ScopedEnvOverride(const ScopedEnvOverride &) = delete;
        ScopedEnvOverride &operator=(const ScopedEnvOverride &) = delete;

    private:
        const char *name_;
        bool had_original_ = false;
        std::string original_value_;
    };

    void SetUp() override
    {
        // Set OpenMP threads for CPU-side operations (quantization, weight packing)
        omp_set_num_threads(56);

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
     *
     * Uses HIP event timing to measure ONLY kernel execution, excluding PCIe transfers.
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

        // 2b. Bind required workspace (needed for full multiply_tensor path)
        auto workspace = std::make_unique<DeviceWorkspaceManager>(DeviceId::rocm(0), 1024ULL * 1024ULL * 1024ULL);
        const auto reqs = kernel.getWorkspaceRequirements(M, N, K);
        EXPECT_TRUE(workspace->allocate(reqs));
        kernel.bindWorkspace(workspace.get());

        // 3. Create random FP32 activations [M × K]
        auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});
        auto output = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});

        // 4. Compute reference for accuracy check
        std::vector<float> reference(M * N);
        computeReference(input->data(), weights->fp32_data(), reference.data(), M, N, K);

        // 5. Warmup
        bool use_kernel_timed_path = !config.end_to_end_timing;
        for (int i = 0; i < config.warmup_iters; ++i)
        {
            if (!use_kernel_timed_path)
            {
                kernel.multiply_tensor(input.get(), output.get(), M, N, K);
            }
            else
            {
                if (!kernel.multiply_tensor_timed(input.get(), output.get(), M, N, K, nullptr))
                {
                    use_kernel_timed_path = false;
                    kernel.multiply_tensor(input.get(), output.get(), M, N, K);
                }
            }
        }
        (void)hipDeviceSynchronize();

        // 6. Benchmark trials
        std::vector<double> trial_times_ms;
        trial_times_ms.reserve(config.num_trials * config.bench_iters);

        for (int t = 0; t < config.num_trials; ++t)
        {
            for (int i = 0; i < config.bench_iters; ++i)
            {
                if (!use_kernel_timed_path)
                {
                    auto start = std::chrono::high_resolution_clock::now();
                    kernel.multiply_tensor(input.get(), output.get(), M, N, K);
                    (void)hipDeviceSynchronize();
                    auto end = std::chrono::high_resolution_clock::now();
                    const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
                    trial_times_ms.push_back(elapsed_ms);
                }
                else
                {
                    float kernel_time_ms = 0.0f;
                    if (!kernel.multiply_tensor_timed(input.get(), output.get(), M, N, K, &kernel_time_ms))
                    {
                        use_kernel_timed_path = false;
                        auto start = std::chrono::high_resolution_clock::now();
                        kernel.multiply_tensor(input.get(), output.get(), M, N, K);
                        (void)hipDeviceSynchronize();
                        auto end = std::chrono::high_resolution_clock::now();
                        const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
                        trial_times_ms.push_back(elapsed_ms);
                        continue;
                    }

                    // Only record valid timings (>0 means HIP events worked)
                    if (kernel_time_ms > 0.0f)
                    {
                        trial_times_ms.push_back(kernel_time_ms);
                    }
                }
            }
        }

        // 7. Compute accuracy (after all benchmarks)
        double cosine_sim = computeCosineSimilarity(output->data(), reference.data(), M * N);

        // 8. Calculate statistics
        ROCmBenchStats stats;

        if (trial_times_ms.empty())
        {
            // Fallback if timed path didn't work
            stats.mean_ms = 0.0;
            stats.stddev_ms = 0.0;
            stats.min_ms = 0.0;
            stats.max_ms = 0.0;
            stats.mean_tflops = 0.0;
            stats.peak_tflops = 0.0;
        }
        else
        {
            double sum = std::accumulate(trial_times_ms.begin(), trial_times_ms.end(), 0.0);
            stats.mean_ms = sum / trial_times_ms.size();

            double sq_sum = std::inner_product(trial_times_ms.begin(), trial_times_ms.end(),
                                               trial_times_ms.begin(), 0.0);
            stats.stddev_ms = std::sqrt(sq_sum / trial_times_ms.size() - stats.mean_ms * stats.mean_ms);

            stats.min_ms = *std::min_element(trial_times_ms.begin(), trial_times_ms.end());
            stats.max_ms = *std::max_element(trial_times_ms.begin(), trial_times_ms.end());

            // TFLOPS = 2 * M * N * K / (time_s * 1e12)
            double ops = 2.0 * M * N * K;
            stats.mean_tflops = (ops / (stats.mean_ms * 1e-3)) / 1e12;
            stats.peak_tflops = (ops / (stats.min_ms * 1e-3)) / 1e12;
        }

        stats.cosine_sim = cosine_sim;
        stats.quant_time_ms = 0.0;
        stats.gemm_time_ms = stats.mean_ms;

        return stats;
    }

    void printFullPathSweepRow(const std::string &mode_name, const ROCmBenchConfig &config, const ROCmBenchStats &stats)
    {
        std::cout << "  mode=" << std::left << std::setw(16) << mode_name
                  << " M=" << std::setw(5) << config.M
                  << " N=" << std::setw(6) << config.N
                  << " K=" << std::setw(6) << config.K
                  << " | " << std::fixed << std::setprecision(3) << std::setw(8) << stats.mean_ms << " ms"
                  << " | " << std::setprecision(3) << std::setw(7) << stats.mean_tflops << " TFLOPS"
                  << " | cos=" << std::setprecision(6) << stats.cosine_sim
                  << std::endl;
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
// Performance Tests: Model-Based Benchmarks
// ============================================================================
//
// Each model test runs all GEMM shapes (Attention Output, FFN Gate/Up, FFN Down)
// across batch sizes from decode (M=1) through large prefill (M=16384).
//
// Run examples:
//   --gtest_filter="*Qwen0_5B*"     # All 0.5B tests
//   --gtest_filter="*Qwen7B*"       # All 7B tests
//   --gtest_filter="*WeightPacking*" # Weight packing throughput
//   --gtest_filter="*BatchSizeScaling*" # Scaling analysis
//
// ============================================================================

// Model dimension constants
namespace QwenDims
{
    // Qwen2.5-0.5B: hidden=896, intermediate=4864, num_heads=14, head_dim=64
    constexpr int H_0_5B = 896;
    constexpr int I_0_5B = 4864;
    // Qwen2.5-7B: hidden=3584, intermediate=18944, num_heads=28, head_dim=128
    constexpr int H_7B = 3584;
    constexpr int I_7B = 18944;
    // Qwen2.5-14B: hidden=5120, intermediate=13824, num_heads=40, head_dim=128
    constexpr int H_14B = 5120;
    constexpr int I_14B = 13824;
    // Qwen2.5-32B: hidden=5120, intermediate=27648, num_heads=40, head_dim=128
    constexpr int H_32B = 5120;
    constexpr int I_32B = 27648;
}

// Standard batch sizes: decode (M=1) through large prefill
static const std::vector<int> kBatchSizes = {1, 32, 128, 512, 1024, 4096, 8192, 16384};

// ----------------------------------------------------------------------------
// Qwen2.5-0.5B: hidden=896, intermediate=4864
// ----------------------------------------------------------------------------

TEST_F(ROCmQuantisedGemmPerf, Qwen0_5B)
{
    if (!has_rocm_device_)
        GTEST_SKIP() << "No ROCm device";
    runModelBenchmarks("Qwen2.5-0.5B (hidden=896, inter=4864)",
                       QwenDims::H_0_5B, QwenDims::I_0_5B, kBatchSizes);
}

// ----------------------------------------------------------------------------
// Qwen2.5-7B: hidden=3584, intermediate=18944
// ----------------------------------------------------------------------------

TEST_F(ROCmQuantisedGemmPerf, Qwen7B)
{
    if (!has_rocm_device_)
        GTEST_SKIP() << "No ROCm device";
    runModelBenchmarks("Qwen2.5-7B (hidden=3584, inter=18944)",
                       QwenDims::H_7B, QwenDims::I_7B, kBatchSizes);
}

// ----------------------------------------------------------------------------
// Qwen2.5-14B: hidden=5120, intermediate=13824
// ----------------------------------------------------------------------------

TEST_F(ROCmQuantisedGemmPerf, Qwen14B)
{
    if (!has_rocm_device_)
        GTEST_SKIP() << "No ROCm device";
    runModelBenchmarks("Qwen2.5-14B (hidden=5120, inter=13824)",
                       QwenDims::H_14B, QwenDims::I_14B, kBatchSizes);
}

// ----------------------------------------------------------------------------
// Qwen2.5-32B: hidden=5120, intermediate=27648
// ----------------------------------------------------------------------------

TEST_F(ROCmQuantisedGemmPerf, Qwen32B)
{
    if (!has_rocm_device_)
        GTEST_SKIP() << "No ROCm device";
    runModelBenchmarks("Qwen2.5-32B (hidden=5120, inter=27648)",
                       QwenDims::H_32B, QwenDims::I_32B, kBatchSizes);
}

// ============================================================================
// Micro-benchmarks: Weight Packing and Batch Size Scaling
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

    std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};

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

/**
 * @test Full production-path prefill sweep (quantize + prefill GEMM + scaling)
 *
 * Compares baseline prefill native INT8 path against grid-kpar variants while timing
 * full `multiply_tensor` execution to preserve interaction effects across stages.
 */
TEST_F(ROCmQuantisedGemmPerf, PrefillFullPath_GridKParSweep)
{
    if (!has_rocm_device_)
    {
        GTEST_SKIP() << "No ROCm device available";
    }

    struct ModeCfg
    {
        std::string name;
        std::string grid_kpar;
        std::string splits;
    };

    const std::vector<ModeCfg> modes = {
        {"baseline", "0", "0"},
        {"grid_auto", "1", "0"},
        {"grid_s4", "1", "4"},
        {"grid_s8", "1", "8"},
    };

    const std::vector<int> m_buckets = {64, 256};

    std::cout << "\n[Perf] ROCm prefill full-path sweep (Qwen-7B FFN Up, Mx3584 * 18944x3584^T)\n"
              << "[Perf] Path includes: activation quantization + prefill native GEMM + scaling/epilogue\n";

    for (int M : m_buckets)
    {
        std::cout << "\n[Perf] Shape bucket: M=" << M << " N=18944 K=3584" << std::endl;

        for (const auto &mode : modes)
        {
            ScopedEnvOverride grid_enabled("LLAMINAR_ROCM_VNNI_PREFILL_GRID_KPAR", mode.grid_kpar);
            ScopedEnvOverride grid_splits("LLAMINAR_ROCM_VNNI_PREFILL_GRID_KPAR_SPLITS", mode.splits);

            ROCmBenchConfig cfg{
                .name = "7B FFN Up FullPath",
                .M = M,
                .N = 18944,
                .K = 3584,
                .warmup_iters = 2,
                .bench_iters = 3,
                .num_trials = 2,
                .end_to_end_timing = true,
            };

            auto stats = runBenchmark(cfg);
            printFullPathSweepRow(mode.name, cfg, stats);
            EXPECT_GT(stats.cosine_sim, 0.99);
        }
    }
}

/**
 * @test Full production-path sweep across real model shapes and aspect ratios.
 *
 * Evaluates runtime auto prefill tile dispatch against forced micro-variants
 * on representative model projection matrices.
 */
TEST_F(ROCmQuantisedGemmPerf, PrefillFullPath_RealModelAspectSweep)
{
    if (!has_rocm_device_)
    {
        GTEST_SKIP() << "No ROCm device available";
    }

    struct ShapeCfg
    {
        std::string name;
        int N;
        int K;
    };

    const std::vector<ShapeCfg> shapes = {
        {"0.5B AttnOut", 896, 896},
        {"0.5B FFN Up", 4864, 896},
        {"0.5B FFN Down", 896, 4864},
        {"7B AttnOut", 3584, 3584},
        {"7B FFN Up", 18944, 3584},
        {"7B FFN Down", 3584, 18944},
        {"14B FFN Up", 13824, 5120},
        {"32B FFN Up", 27648, 5120},
    };

    struct VariantCfg
    {
        std::string name;
        std::string variant;
    };

    const std::vector<VariantCfg> variants = {
        {"auto", "-1"},
        {"tile16x16", "0"},
        {"tile32x8", "1"},
        {"tile8x32", "2"},
        {"tile8x8", "3"},
    };

    const std::vector<int> m_buckets = {32, 256};

    std::cout << "\n[Perf] ROCm prefill full-path real-model aspect sweep\n"
              << "[Perf] Path includes: activation quantization + prefill native GEMM + scaling/epilogue\n";

    for (const auto &shape : shapes)
    {
        for (int M : m_buckets)
        {
            std::cout << "\n[Perf] Shape=" << shape.name << " M=" << M << " N=" << shape.N << " K=" << shape.K << std::endl;

            for (const auto &variant : variants)
            {
                ScopedEnvOverride grid_enabled("LLAMINAR_ROCM_VNNI_PREFILL_GRID_KPAR", "0");
                ScopedEnvOverride variant_override("LLAMINAR_ROCM_VNNI_PREFILL_VARIANT", variant.variant);

                ROCmBenchConfig cfg{
                    .name = shape.name,
                    .M = M,
                    .N = shape.N,
                    .K = shape.K,
                    .warmup_iters = 1,
                    .bench_iters = 2,
                    .num_trials = 1,
                    .end_to_end_timing = true,
                };

                auto stats = runBenchmark(cfg);
                printFullPathSweepRow(variant.name, cfg, stats);
                EXPECT_GT(stats.cosine_sim, 0.99);
            }
        }
    }
}

TEST_F(ROCmQuantisedGemmPerf, PrefillFullPath_CptAndKbSweep)
{
    if (!has_rocm_device_)
    {
        GTEST_SKIP() << "No ROCm device available";
    }

    struct ShapeCfg
    {
        std::string name;
        int M;
        int N;
        int K;
    };

    const std::vector<ShapeCfg> shapes = {
        {"7B AttnOut", 128, 3584, 3584},
        {"7B FFN Up", 128, 18944, 3584},
        {"7B FFN Gate", 128, 18944, 3584},
        {"7B FFN Down", 128, 3584, 18944},
        {"14B AttnOut", 128, 5120, 5120},
        {"14B FFN Up", 128, 13824, 5120},
        {"14B FFN Gate", 128, 13824, 5120},
        {"14B FFN Down", 128, 5120, 13824},
    };

    const std::vector<std::string> cpts = {"1"};
    const std::vector<std::string> kbs = {"0", "4", "8", "16"};

    std::cout << "\n[Perf] ROCm prefill full-path KB sweep (CPT fixed to 1)\n"
              << "[Perf] Path includes: activation quantization + prefill native GEMM + scaling/epilogue\n";

    for (const auto &shape : shapes)
    {
        std::cout << "\n[Perf] Shape=" << shape.name << " M=" << shape.M << " N=" << shape.N << " K=" << shape.K << std::endl;

        for (const auto &cpt : cpts)
        {
            // Baseline (grid off) for this CPT
            {
                ScopedEnvOverride grid_enabled("LLAMINAR_ROCM_VNNI_PREFILL_GRID_KPAR", "0");
                ScopedEnvOverride prefill_cpt("LLAMINAR_ROCM_VNNI_PREFILL_CPT", cpt);

                ROCmBenchConfig cfg{
                    .name = shape.name + " baseline cpt=" + cpt,
                    .M = shape.M,
                    .N = shape.N,
                    .K = shape.K,
                    .warmup_iters = 1,
                    .bench_iters = 2,
                    .num_trials = 1,
                    .end_to_end_timing = true,
                };

                auto stats = runBenchmark(cfg);
                printFullPathSweepRow("baseline_cpt" + cpt, cfg, stats);
                EXPECT_GT(stats.cosine_sim, 0.99);
            }

            // Grid-kpar with KB sweep for this CPT
            for (const auto &kb : kbs)
            {
                ScopedEnvOverride grid_enabled("LLAMINAR_ROCM_VNNI_PREFILL_GRID_KPAR", "1");
                ScopedEnvOverride prefill_cpt("LLAMINAR_ROCM_VNNI_PREFILL_CPT", cpt);
                ScopedEnvOverride grid_kb("LLAMINAR_ROCM_VNNI_PREFILL_GRID_KPAR_KB", kb);
                ScopedEnvOverride grid_splits_auto("LLAMINAR_ROCM_VNNI_PREFILL_GRID_KPAR_SPLITS", "0");

                ROCmBenchConfig cfg{
                    .name = shape.name + " grid cpt=" + cpt + " kb=" + kb,
                    .M = shape.M,
                    .N = shape.N,
                    .K = shape.K,
                    .warmup_iters = 1,
                    .bench_iters = 2,
                    .num_trials = 1,
                    .end_to_end_timing = true,
                };

                auto stats = runBenchmark(cfg);
                printFullPathSweepRow("grid_cpt" + cpt + "_kb" + kb, cfg, stats);
                EXPECT_GT(stats.cosine_sim, 0.99);
            }
        }
    }
}

/**
 * @test Harvest canonical CK full-path prefill baselines for strategy-lab shapes
 *
 * This test emits parseable lines for the exact shape set used by
 * Microbench__ROCmPrefillStrategyLab.hip (Qwen2.5 0.5B and 3B projections).
 *
 * Output format:
 *   CANONICAL_CK_PREFILL,<shape_name>,M,N,K,mean_ms,mean_tflops,cosine
 */
TEST_F(ROCmQuantisedGemmPerf, PrefillFullPath_CKCanonicalHarvest_Qwen0_5B_3B)
{
    if (!has_rocm_device_)
    {
        GTEST_SKIP() << "No ROCm device available";
    }

    struct ShapeCfg
    {
        std::string name;
        int M;
        int N;
        int K;
    };

    const std::vector<ShapeCfg> shapes = {
        {"Qwen2.5-0.5B_AttnOut", 128, 896, 896},
        {"Qwen2.5-0.5B_FFN_Up", 128, 4864, 896},
        {"Qwen2.5-0.5B_FFN_Gate", 128, 4864, 896},
        {"Qwen2.5-0.5B_FFN_Down", 128, 896, 4864},
        {"Qwen2.5-0.5B_LM_Head", 128, 151936, 896},
        {"Qwen2.5-3B_AttnOut", 128, 2048, 2048},
        {"Qwen2.5-3B_FFN_Up", 128, 11008, 2048},
        {"Qwen2.5-3B_FFN_Gate", 128, 11008, 2048},
        {"Qwen2.5-3B_FFN_Down", 128, 2048, 11008},
        {"Qwen2.5-3B_LM_Head", 128, 151936, 2048},
    };

    // Canonical CK full-path: native dispatch now handles this automatically.
    ScopedEnvOverride grid_kpar("LLAMINAR_ROCM_VNNI_PREFILL_GRID_KPAR", "0");
    ScopedEnvOverride grid_splits("LLAMINAR_ROCM_VNNI_PREFILL_GRID_KPAR_SPLITS", "0");

    std::cout << "\n[Perf] Canonical CK full-path harvest (Qwen2.5 0.5B / 3B strategy-lab shapes)\n";
    std::cout << "[Perf] Emitting parseable lines prefixed with CANONICAL_CK_PREFILL\n";

    for (const auto &shape : shapes)
    {
        ROCmBenchConfig cfg{
            .name = shape.name,
            .M = shape.M,
            .N = shape.N,
            .K = shape.K,
            .warmup_iters = 2,
            .bench_iters = 4,
            .num_trials = 2,
            .end_to_end_timing = true,
        };

        const auto stats = runBenchmark(cfg);

        std::cout << "CANONICAL_CK_PREFILL,"
                  << shape.name << ","
                  << shape.M << ","
                  << shape.N << ","
                  << shape.K << ","
                  << std::fixed << std::setprecision(6) << stats.mean_ms << ","
                  << std::fixed << std::setprecision(6) << stats.mean_tflops << ","
                  << std::fixed << std::setprecision(6) << stats.cosine_sim
                  << std::endl;

        EXPECT_GT(stats.cosine_sim, 0.99);
    }
}
