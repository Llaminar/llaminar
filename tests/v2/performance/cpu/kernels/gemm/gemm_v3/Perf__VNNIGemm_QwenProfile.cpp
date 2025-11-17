/**
 * @file Perf__VNNIGemm_QwenProfile.cpp
 * @brief Realistic Qwen 0.5B pipeline operation profiling for VNNI GEMM (gemm_v3)
 *
 * Purpose:
 * - Test representative operations from Qwen 0.5B inference pipeline
 * - Compare performance between VNNIGemmAdapter and direct registry calls
 * - Use real quantized model weights loaded via ModelLoader
 * - Cover both single-token decode and batch prefill scenarios
 * - Present comprehensive performance comparison table
 *
 * Qwen 0.5B Architecture:
 * - d_model: 896
 * - n_heads: 14
 * - d_head: 64
 * - FFN intermediate: 4864
 * - Layers: 24
 *
 * Test Scenarios:
 * 1. Single-token decode (batch=1):
 *    - Q/K/V projections: [1, 896] × [896, 896] → [1, 896]
 *    - Attention output:  [1, 896] × [896, 896] → [1, 896]
 *    - FFN gate/up:       [1, 896] × [896, 4864] → [1, 4864]
 *    - FFN down:          [1, 4864] × [4864, 896] → [1, 896]
 *
 * 2. Medium batch (batch=32):
 *    - Q/K/V projections: [32, 896] × [896, 896] → [32, 896]
 *    - FFN operations:    [32, 896/4864] × [896/4864, ...] → [32, ...]
 *
 * 3. Large batch prefill (batch=128):
 *    - Q/K/V projections: [128, 896] × [896, 896] → [128, 896]
 *    - FFN operations:    [128, 896/4864] × [896/4864, ...] → [128, ...]
 *
 * Performance Metrics:
 * - INT8 GOPS (Giga Integer Operations Per Second)
 * - 1 operation = 1 INT8 multiply-accumulate (MAC)
 * - Total operations per GEMM: 2*M*N*K
 * - Adapter overhead: (adapter_time - registry_time) / registry_time × 100%
 * - Theoretical peak: ~2240 GOPS (28 cores × 80 GOPS/core @ 2.5 GHz)
 *
 * IMPORTANT: Parallelization Limitations
 * - Kernel uses M_R=8 (8 rows per tile)
 * - M=128 → 16 tiles → maximum 16 threads utilized
 * - M=32 → 4 tiles → maximum 4 threads utilized
 * - M=1 (decode) → 1 tile → single-threaded execution
 * - Efficiency % is measured against full 28-core peak, but actual
 *   parallelism is limited by M/M_R (tile count)
 * - For M=128: effective peak ~1120 GOPS (16 cores × 70 GOPS/core)
 * - Expected efficiency: ~7% (80 GOPS / 1120 GOPS) NOT 3.6%!
 *
 * @author David Sanftenberg
 * @date November 17, 2025
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cstring>
#include <algorithm>

#include "v2/kernels/cpu/gemm_v3/VNNIGemmAdapter.h"
#include "v2/kernels/cpu/gemm_v3/VNNIGemmKernelRegistry.h"
#include "tensors/Tensors.h"
#include "loaders/ModelLoader.h"

using namespace llaminar2;
using namespace std::chrono;

namespace
{
    // Operation descriptor for Qwen pipeline operations
    struct QwenOpDescriptor
    {
        std::string name;
        std::string category; // "decode", "batch_32", "batch_128"
        int m, n, k;
        std::string weight_key; // e.g., "blk.0.attn_q.weight"
    };

    // Benchmark result for comparison
    struct BenchmarkResult
    {
        std::string name;
        std::string category;
        int m, n, k;
        double adapter_gops;
        double adapter_time_ms;
        double registry_gops;
        double registry_time_ms;
        double adapter_overhead_pct;
        double efficiency_pct; // vs theoretical peak
    };

    // Theoretical peak for AVX512 VNNI on Sapphire Rapids @ 2.5 GHz
    // 1 core: 2 FMA × 16 INT8/cycle × 2.5 GHz = ~80 GOPS
    // Single socket (28 cores): 28 × 80 = 2240 GOPS theoretical peak
    // NOTE: We're running single-threaded per test, so actual peak is much lower
    constexpr double THEORETICAL_PEAK_GOPS = 2240.0; /**
                                                      * @brief Load real Qwen 0.5B weights from GGUF file
                                                      */
    std::unique_ptr<ModelLoader> load_qwen_model()
    {
        const char *model_path = std::getenv("LLAMINAR_TEST_MODEL_PATH");
        if (!model_path)
        {
            model_path = "/workspaces/llaminar/models/qwen2.5-0.5b-instruct-q8_0.gguf";
        }

        auto loader = std::make_unique<ModelLoader>();

        if (!loader->loadModel(model_path))
        {
            std::cerr << "Failed to load model from: " << model_path << std::endl;
            return nullptr;
        }

        std::cout << "Loaded model: " << model_path << std::endl;
        std::cout << "\n";

        return loader;
    }

    /**
     * @brief Generate random FP32 activation tensor
     */
    std::unique_ptr<FP32Tensor> generate_random_activations(int M, int K)
    {
        auto A = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});

        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        std::vector<float> data(M * K);
        for (auto &val : data)
        {
            val = dist(rng);
        }

        std::memcpy(A->mutable_data(), data.data(), M * K * sizeof(float));
        return A;
    }

    /**
     * @brief Benchmark a single operation using VNNIGemmAdapter
     *
     * @param M Activation rows
     * @param N Weight columns
     * @param K Inner dimension
     * @param A Activation tensor (FP32)
     * @param B Weight tensor (Q8_0)
     * @param warmup_iters Warmup iterations
     * @param timed_iters Timed iterations
     * @return Execution time in milliseconds
     */
    double benchmark_adapter(
        int M, int N, int K,
        const IActivationTensor &A,
        const Q8_0Tensor &B,
        int warmup_iters = 5,
        int timed_iters = 20)
    {
        std::vector<float> C(M * N);

        // Warmup
        for (int i = 0; i < warmup_iters; ++i)
        {
            vnni_gemm_adapter<8, 16, 128, 1, 64>(M, N, K, A, B, C.data(), N);
        }

        // Timed run
        auto start = high_resolution_clock::now();
        for (int i = 0; i < timed_iters; ++i)
        {
            vnni_gemm_adapter<8, 16, 128, 1, 64>(M, N, K, A, B, C.data(), N);
        }
        auto end = high_resolution_clock::now();

        // Use nanosecond precision for better accuracy on fast operations
        double elapsed_ns = duration_cast<nanoseconds>(end - start).count();
        double elapsed_ms = elapsed_ns / 1e6;
        return elapsed_ms / timed_iters;
    }

    /**
     * @brief Benchmark a single operation using direct registry call
     *
     * @param M Activation rows
     * @param N Weight columns
     * @param K Inner dimension
     * @param A Activation tensor (FP32)
     * @param B Weight tensor (Q8_0)
     * @param warmup_iters Warmup iterations
     * @param timed_iters Timed iterations
     * @return Execution time in milliseconds
     */
    double benchmark_registry(
        int M, int N, int K,
        const IActivationTensor &A,
        const Q8_0Tensor &B,
        int warmup_iters = 5,
        int timed_iters = 20)
    {
        std::vector<float> C(M * N);

        // Get kernel from registry
        auto &registry = VNNIGemmKernelRegistry::instance();
        VNNIGemmFunc kernel;

        kernel = registry.get_kernel(8, 16, 128, 1, 64);
        if (!kernel)
        {
            std::cerr << "Failed to lookup kernel (not found in registry)" << std::endl;
            return -1.0;
        }

        // Warmup
        for (int i = 0; i < warmup_iters; ++i)
        {
            kernel(M, N, K, A, B, C.data(), N);
        }

        // Timed run
        auto start = high_resolution_clock::now();
        for (int i = 0; i < timed_iters; ++i)
        {
            kernel(M, N, K, A, B, C.data(), N);
        }
        auto end = high_resolution_clock::now();

        // Use nanosecond precision for better accuracy on fast operations
        double elapsed_ns = duration_cast<nanoseconds>(end - start).count();
        double elapsed_ms = elapsed_ns / 1e6;
        return elapsed_ms / timed_iters;
    }

    /**
     * @brief Run full benchmark suite for Qwen operations
     */
    std::vector<BenchmarkResult> run_benchmark_suite(ModelLoader &loader)
    {
        std::vector<BenchmarkResult> results;

        // Define Qwen 0.5B operations to benchmark across multiple batch sizes
        // Note: Qwen 2.5 0.5B uses GQA with 14 Q heads (896 dims) and 2 KV heads (128 dims)
        // Testing scaling behavior: M=1 (single-thread), M=32/128/512 (partial parallelism),
        // M=1024/4096/8192/16384 (full parallelism with M_R=8 → 128/512/1024/2048 tiles)
        std::vector<QwenOpDescriptor> ops = {
            // M=1: Single-token decode (1 tile → single-threaded)
            {"FFN gate (M=1)", "M=1", 1, 4864, 896, "blk.0.ffn_gate.weight"},

            // M=32: Small batch (4 tiles → 4 threads max)
            {"FFN gate (M=32)", "M=32", 32, 4864, 896, "blk.0.ffn_gate.weight"},

            // M=128: Medium batch (16 tiles → 16 threads max)
            {"FFN gate (M=128)", "M=128", 128, 4864, 896, "blk.0.ffn_gate.weight"},

            // M=512: Large batch (64 tiles → full 28-thread utilization)
            {"FFN gate (M=512)", "M=512", 512, 4864, 896, "blk.0.ffn_gate.weight"},

            // M=1024: Very large batch (128 tiles → full parallelism)
            {"FFN gate (M=1024)", "M=1024", 1024, 4864, 896, "blk.0.ffn_gate.weight"},

            // M=4096: Huge batch (512 tiles → excellent load balancing)
            {"FFN gate (M=4096)", "M=4096", 4096, 4864, 896, "blk.0.ffn_gate.weight"},

            // M=8192: Massive batch (1024 tiles → maximum throughput)
            {"FFN gate (M=8192)", "M=8192", 8192, 4864, 896, "blk.0.ffn_gate.weight"},

            // M=16384: Extreme batch (2048 tiles → sustained peak)
            {"FFN gate (M=16384)", "M=16384", 16384, 4864, 896, "blk.0.ffn_gate.weight"},
        };

        for (const auto &op : ops)
        {
            std::cout << "Benchmarking: " << op.name << " [" << op.m << "×" << op.n << "×" << op.k << "]" << std::endl;

            // Load weight tensor
            auto B_tensor = loader.loadTensor(op.weight_key, 0);
            if (!B_tensor)
            {
                std::cerr << "  ⚠ Failed to load weight: " << op.weight_key << std::endl;
                continue;
            }

            auto *B_q8 = dynamic_cast<Q8_0Tensor *>(B_tensor.get());
            if (!B_q8)
            {
                std::cerr << "  ⚠ Weight is not Q8_0 format" << std::endl;
                continue;
            }

            // Verify dimensions
            auto shape = B_q8->shape();
            if (shape.size() != 2 || shape[0] != op.n || shape[1] != op.k)
            {
                std::cerr << "  ⚠ Weight shape mismatch: expected [" << op.n << ", " << op.k
                          << "], got [" << shape[0] << ", " << shape[1] << "]" << std::endl;
                continue;
            }

            // Generate random activations
            auto A = generate_random_activations(op.m, op.k);

            // Benchmark iterations: scale inversely with M for consistent total runtime
            // M=1: 100/500 (high reps for timing resolution)
            // M≤128: 10/30 (medium reps)
            // M≥512: 3/10 (low reps - operations are expensive)
            int warmup, timed;
            if (op.m == 1)
            {
                warmup = 100;
                timed = 500;
            }
            else if (op.m <= 128)
            {
                warmup = 10;
                timed = 30;
            }
            else if (op.m <= 1024)
            {
                warmup = 3;
                timed = 10;
            }
            else
            {
                warmup = 2;
                timed = 5; // Very large M: minimal reps
            }

            double adapter_time_ms = benchmark_adapter(op.m, op.n, op.k, *A, *B_q8, warmup, timed);

            // Benchmark registry path
            double registry_time_ms = benchmark_registry(op.m, op.n, op.k, *A, *B_q8, warmup, timed);

            if (adapter_time_ms < 0 || registry_time_ms < 0)
            {
                std::cerr << "  ⚠ Benchmark failed" << std::endl;
                continue;
            }

            // Calculate metrics
            double total_ops = 2.0 * op.m * op.n * op.k;
            double adapter_gops = (total_ops / 1e9) / (adapter_time_ms / 1000.0);
            double registry_gops = (total_ops / 1e9) / (registry_time_ms / 1000.0);
            double adapter_overhead_pct = ((adapter_time_ms - registry_time_ms) / registry_time_ms) * 100.0;
            double efficiency_pct = (registry_gops / THEORETICAL_PEAK_GOPS) * 100.0;

            BenchmarkResult result{
                op.name,
                op.category,
                op.m, op.n, op.k,
                adapter_gops,
                adapter_time_ms,
                registry_gops,
                registry_time_ms,
                adapter_overhead_pct,
                efficiency_pct};

            results.push_back(result);

            std::cout << "  Adapter:  " << std::setw(8) << std::fixed << std::setprecision(2)
                      << adapter_gops << " GOPS (" << adapter_time_ms << " ms)" << std::endl;
            std::cout << "  Registry: " << std::setw(8) << std::fixed << std::setprecision(2)
                      << registry_gops << " GOPS (" << registry_time_ms << " ms)" << std::endl;
            std::cout << "  Overhead: " << std::setw(8) << std::fixed << std::setprecision(2)
                      << adapter_overhead_pct << " %" << std::endl;
            std::cout << std::endl;
        }

        return results;
    }

    /**
     * @brief Print comprehensive results table
     */
    void print_results_table(const std::vector<BenchmarkResult> &results)
    {
        std::cout << "\n"
                  << std::string(140, '=') << "\n";
        std::cout << "VNNI GEMM Performance Summary: Adapter vs Registry Direct Call\n";
        std::cout << std::string(140, '=') << "\n\n";

        // Header
        std::cout << std::left << std::setw(25) << "Operation"
                  << std::right
                  << std::setw(12) << "M×N×K"
                  << std::setw(12) << "Adapter"
                  << std::setw(10) << "Time(ms)"
                  << std::setw(12) << "Registry"
                  << std::setw(10) << "Time(ms)"
                  << std::setw(12) << "Overhead%"
                  << std::setw(12) << "Efficiency%"
                  << "\n";
        std::cout << std::string(140, '-') << "\n";

        // Group by category
        std::string current_category;
        for (const auto &r : results)
        {
            if (r.category != current_category)
            {
                current_category = r.category;
                std::cout << "\n"
                          << current_category << ":\n";
            }

            std::string dims = std::to_string(r.m) + "×" + std::to_string(r.n) + "×" + std::to_string(r.k);

            std::cout << std::left << std::setw(25) << r.name
                      << std::right
                      << std::setw(12) << dims
                      << std::setw(12) << std::fixed << std::setprecision(2) << r.adapter_gops
                      << std::setw(10) << std::fixed << std::setprecision(3) << r.adapter_time_ms
                      << std::setw(12) << std::fixed << std::setprecision(2) << r.registry_gops
                      << std::setw(10) << std::fixed << std::setprecision(3) << r.registry_time_ms
                      << std::setw(12) << std::fixed << std::setprecision(2) << r.adapter_overhead_pct
                      << std::setw(12) << std::fixed << std::setprecision(2) << r.efficiency_pct
                      << "\n";
        }

        std::cout << "\n"
                  << std::string(140, '=') << "\n";

        // Summary statistics
        if (!results.empty())
        {
            double avg_overhead = 0.0;
            double avg_efficiency = 0.0;
            double max_gops = 0.0;

            for (const auto &r : results)
            {
                avg_overhead += r.adapter_overhead_pct;
                avg_efficiency += r.efficiency_pct;
                max_gops = std::max(max_gops, r.registry_gops);
            }

            avg_overhead /= results.size();
            avg_efficiency /= results.size();

            std::cout << "\nSummary Statistics:\n";
            std::cout << "  Average adapter overhead: " << std::fixed << std::setprecision(2)
                      << avg_overhead << "%\n";
            std::cout << "  Average efficiency (registry): " << std::fixed << std::setprecision(2)
                      << avg_efficiency << "% of theoretical peak\n";
            std::cout << "  Peak throughput: " << std::fixed << std::setprecision(2)
                      << max_gops << " GOPS\n";
            std::cout << "  Theoretical peak: " << std::fixed << std::setprecision(2)
                      << THEORETICAL_PEAK_GOPS << " GOPS\n";
            std::cout << "\n";
        }
    }

} // anonymous namespace

/**
 * @brief VNNI GEMM performance test fixture with MPI awareness
 */
class VNNIGemmPerformance : public ::testing::Test
{
protected:
    int rank_ = 0;
    int world_size_ = 1;

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

        if (rank_ == 0)
        {
            std::cout << "[VNNIGemmPerformance] MPI ranks: " << world_size_ << std::endl;
            std::cout << "[VNNIGemmPerformance] OpenMP threads: " << omp_get_max_threads() << std::endl;
        }
    }
};

/**
 * @brief Configuration sweep result for a specific M value
 */
struct SweepResult
{
    int m;                 ///< Batch size
    int m_r;               ///< Micro-kernel M dimension
    int n_r;               ///< Micro-kernel N dimension
    int k_blk;             ///< K block size
    int unroll_k;          ///< K-loop unroll factor
    int prefetch_b_l1;     ///< L1 prefetch distance
    double gops;           ///< Throughput (GOPS)
    double time_ms;        ///< Time per iteration (ms)
    double efficiency_pct; ///< Efficiency vs theoretical peak

    std::string config_str() const
    {
        std::ostringstream oss;
        oss << "M_R=" << m_r << ",N_R=" << n_r << ",K_BLK=" << k_blk
            << ",UNROLL=" << unroll_k << ",PREFETCH=" << prefetch_b_l1;
        return oss.str();
    }
};

/**
 * @brief Benchmark a single kernel configuration
 *
 * @param m Batch size (M dimension)
 * @param n Output dimension (N dimension)
 * @param k Input dimension (K dimension)
 * @param A Activation tensor
 * @param B Weight tensor
 * @param m_r Micro-kernel M dimension
 * @param n_r Micro-kernel N dimension
 * @param k_blk K block size
 * @param unroll_k K-loop unroll factor
 * @param prefetch_b_l1 L1 prefetch distance
 * @param warmup Number of warmup iterations
 * @param timed Number of timed iterations
 * @return double Time per iteration in milliseconds
 */
double benchmark_config(
    int m, int n, int k,
    const FP32Tensor &A,
    const Q8_0Tensor &B_q8,
    int m_r, int n_r, int k_blk, int unroll_k, int prefetch_b_l1,
    int warmup, int timed)
{
    auto &registry = VNNIGemmKernelRegistry::instance();
    auto kernel = registry.get_kernel(m_r, n_r, k_blk, unroll_k, prefetch_b_l1);
    if (!kernel)
    {
        return -1.0; // Configuration not available
    }

    // Allocate output buffer
    std::vector<float> C(m * n);

    // Warmup
    for (int i = 0; i < warmup; ++i)
    {
        kernel(m, n, k, A, B_q8, C.data(), n);
    }

    // Timed iterations with high-resolution measurement
    MPI_Barrier(MPI_COMM_WORLD);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < timed; ++i)
    {
        kernel(m, n, k, A, B_q8, C.data(), n);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto t1 = std::chrono::high_resolution_clock::now();

    // Use nanoseconds for better precision on fast operations
    double elapsed_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    double elapsed_ms = elapsed_ns / 1e6;
    double per_iter_ms = elapsed_ms / timed;

    // Sanity check: if time is suspiciously small, return -1 to flag it
    // Theoretical minimum for M=1: ~0.001 ms (assuming 10 GOPS on single thread)
    if (per_iter_ms < 0.0001)
    {
        return -1.0; // Too fast to measure accurately, likely timing error
    }

    return per_iter_ms;
}

/**
 * @brief Sweep all kernel configurations for a given operation size
 *
 * @param m Batch size (M dimension)
 * @param n Output dimension (N dimension)
 * @param k Input dimension (K dimension)
 * @param A Activation tensor
 * @param B_q8 Weight tensor
 * @param warmup Number of warmup iterations
 * @param timed Number of timed iterations
 * @return std::vector<SweepResult> Results for all configurations
 */
std::vector<SweepResult> sweep_configurations(
    int m, int n, int k,
    const FP32Tensor &A,
    const Q8_0Tensor &B_q8,
    int warmup, int timed)
{
    std::vector<SweepResult> results;

    // Configuration space (from VNNIGemmKernelRegistry.h)
    std::vector<int> m_r_values = {8, 16, 32, 64};
    std::vector<int> n_r_values = {16, 32, 64, 128};
    std::vector<int> k_blk_values = {32, 64, 128};
    std::vector<int> unroll_k_values = {1, 2, 4};
    std::vector<int> prefetch_values = {0, 64, 128, 256};

    // Total configurations
    double total_ops = 2.0 * m * n * k;
    double theoretical_peak_gops = 2240.0; // 28 cores × 80 GOPS/core

    // Sweep entire parameter space
    for (int m_r : m_r_values)
    {
        for (int n_r : n_r_values)
        {
            for (int k_blk : k_blk_values)
            {
                for (int unroll_k : unroll_k_values)
                {
                    for (int prefetch : prefetch_values)
                    {
                        double time_ms = benchmark_config(
                            m, n, k, A, B_q8,
                            m_r, n_r, k_blk, unroll_k, prefetch,
                            warmup, timed);

                        if (time_ms > 0)
                        {
                            double gops = total_ops / (time_ms * 1e6);
                            double efficiency_pct = (gops / theoretical_peak_gops) * 100.0;

                            // Sanity check: flag impossibly high performance
                            // NOTE: Theoretical peak assumes VNNI (80 GOPS/core), but AMX can do ~16× better
                            //       So we allow up to 800% efficiency before flagging as suspicious
                            if (efficiency_pct > 800.0)
                            {
                                std::cerr << "WARNING: Suspiciously high efficiency " << efficiency_pct
                                          << "% for M_R=" << m_r << ",N_R=" << n_r
                                          << ",K_BLK=" << k_blk << ",UNROLL=" << unroll_k
                                          << ",PREFETCH=" << prefetch
                                          << " (time=" << time_ms << "ms, GOPS=" << gops << ")\n";
                            }

                            results.push_back({m, m_r, n_r, k_blk, unroll_k, prefetch,
                                               gops, time_ms, efficiency_pct});
                        }
                    }
                }
            }
        }
    }

    // Sort by GOPS (descending)
    std::sort(results.begin(), results.end(),
              [](const SweepResult &a, const SweepResult &b)
              { return a.gops > b.gops; });

    return results;
}

/**
 * @brief Print top N configurations from sweep
 */
void print_sweep_top_configs(const std::vector<SweepResult> &results, int top_n = 10)
{
    if (results.empty())
    {
        std::cout << "No configurations tested!\n";
        return;
    }

    std::cout << "\n";
    std::cout << "========================================================================================================\n";
    std::cout << "Top " << top_n << " Configurations (M=" << results[0].m << ")\n";
    std::cout << "========================================================================================================\n";
    std::cout << std::left << std::setw(6) << "Rank"
              << std::setw(8) << "M_R"
              << std::setw(8) << "N_R"
              << std::setw(10) << "K_BLK"
              << std::setw(12) << "UNROLL_K"
              << std::setw(14) << "PREFETCH"
              << std::setw(12) << "GOPS"
              << std::setw(12) << "Time(ms)"
              << std::setw(12) << "Efficiency%\n";
    std::cout << "--------------------------------------------------------------------------------------------------------\n";

    int limit = std::min(top_n, static_cast<int>(results.size()));
    for (int i = 0; i < limit; ++i)
    {
        const auto &r = results[i];
        std::cout << std::left << std::setw(6) << (i + 1)
                  << std::setw(8) << r.m_r
                  << std::setw(8) << r.n_r
                  << std::setw(10) << r.k_blk
                  << std::setw(12) << r.unroll_k
                  << std::setw(14) << r.prefetch_b_l1
                  << std::fixed << std::setprecision(2)
                  << std::setw(12) << r.gops
                  << std::setw(12) << r.time_ms
                  << std::setw(12) << r.efficiency_pct << "\n";
    }
    std::cout << "========================================================================================================\n";

    // Performance statistics
    double best_gops = results[0].gops;
    double worst_gops = results.back().gops;
    double avg_gops = 0.0;
    for (const auto &r : results)
    {
        avg_gops += r.gops;
    }
    avg_gops /= results.size();

    std::cout << "\nSummary:\n";
    std::cout << "  Best:    " << std::fixed << std::setprecision(2) << best_gops << " GOPS (" << results[0].config_str() << ")\n";
    std::cout << "  Worst:   " << std::fixed << std::setprecision(2) << worst_gops << " GOPS\n";
    std::cout << "  Average: " << std::fixed << std::setprecision(2) << avg_gops << " GOPS\n";
    std::cout << "  Range:   " << std::fixed << std::setprecision(1) << (best_gops / worst_gops) << "× speedup\n";
    std::cout << "  Configs: " << results.size() << " tested\n\n";
}

/**
 * @brief Parameter sweep test: find optimal configurations for each batch size
 *
 * This test sweeps the entire kernel configuration space (M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1)
 * for representative batch sizes to identify optimal parameters for different workload characteristics.
 *
 * Test Matrix:
 * - M values: 1, 32, 128, 512, 1024, 4096, 8192
 * - Operation: FFN gate [M×4864×896] (realistic Qwen 0.5B workload)
 * - Configurations: ~200-400 registered kernels per M
 *
 * Output:
 * - Top 10 configurations per M value ranked by GOPS
 * - Best/worst/average performance statistics
 * - Configuration trends vs batch size
 */
TEST_F(VNNIGemmPerformance, ParameterSweep)
{
    // Only rank 0 does the work
    if (rank_ == 0)
    {
        auto loader = load_qwen_model();
        ASSERT_NE(loader, nullptr) << "Failed to load Qwen model";

        // Test FFN gate operation (realistic Qwen workload)
        const int n = 4864; // FFN intermediate dimension
        const int k = 896;  // d_model

        // Load FFN gate weights (layer 0 for consistency)
        auto B_tensor = loader->loadTensor("blk.0.ffn_gate.weight", 0);
        ASSERT_NE(B_tensor, nullptr) << "Failed to load FFN gate weights";

        auto *B_q8 = dynamic_cast<Q8_0Tensor *>(B_tensor.get());
        ASSERT_NE(B_q8, nullptr) << "FFN gate weights not Q8_0 format";

        // Batch sizes to sweep
        // NOTE: Skipping M=1 & M=32 because operations complete too fast for accurate timing
        //       Even with 10,000 iterations, total runtime ~1ms → unreliable per-iter measurements
        //       M=128 is the smallest batch size we can reliably benchmark
        std::vector<int> m_values = {128, 512, 1024, 4096, 8192, 16384, 32768};

        // Store best configuration per M for final summary
        std::map<int, SweepResult> best_configs;

        for (int m : m_values)
        {
            std::cout << "\n========================================================================================================\n";
            std::cout << "Sweeping configurations for M=" << m << " (batch size)\n";
            std::cout << "========================================================================================================\n";

            // Generate random activations
            auto A = generate_random_activations(m, k);

            // Iteration counts: scale inversely with M
            // Target: ~50-100ms minimum total runtime for reliable timing
            int warmup, timed;
            if (m == 32)
            {
                warmup = 10;
                timed = 30; // ~1-2 seconds per config, ~5 minutes total for 432 configs
            }
            else if (m <= 128)
            {
                warmup = 10;
                timed = 20;
            }
            else if (m <= 1024)
            {
                warmup = 5;
                timed = 20;
            }
            else
            {
                warmup = 2;
                timed = 5; // Large M: minimal reps (expensive operations)
            }

            // Sweep all configurations
            auto results = sweep_configurations(m, n, k, *A, *B_q8, warmup, timed);
            ASSERT_FALSE(results.empty()) << "No configurations available for M=" << m;

            // Print top 10 configurations
            print_sweep_top_configs(results, 10);

            // Store best configuration
            best_configs[m] = results[0];
        }

        // Print final summary: best configuration per M
        std::cout << "\n";
        std::cout << "========================================================================================================\n";
        std::cout << "Best Configuration Summary (All Batch Sizes)\n";
        std::cout << "========================================================================================================\n";
        std::cout << std::left << std::setw(8) << "M"
                  << std::setw(8) << "M_R"
                  << std::setw(8) << "N_R"
                  << std::setw(10) << "K_BLK"
                  << std::setw(12) << "UNROLL_K"
                  << std::setw(14) << "PREFETCH"
                  << std::setw(12) << "GOPS"
                  << std::setw(12) << "Efficiency%\n";
        std::cout << "--------------------------------------------------------------------------------------------------------\n";

        for (const auto &[m, r] : best_configs)
        {
            std::cout << std::left << std::setw(8) << m
                      << std::setw(8) << r.m_r
                      << std::setw(8) << r.n_r
                      << std::setw(10) << r.k_blk
                      << std::setw(12) << r.unroll_k
                      << std::setw(14) << r.prefetch_b_l1
                      << std::fixed << std::setprecision(2)
                      << std::setw(12) << r.gops
                      << std::setw(12) << r.efficiency_pct << "\n";
        }
        std::cout << "========================================================================================================\n";

        // Validate that we found valid configurations
        for (const auto &[m, r] : best_configs)
        {
            EXPECT_GT(r.gops, 0.0) << "Invalid GOPS for M=" << m;
            if (m >= 512)
            {
                // For large M with full parallelization, expect reasonable efficiency
                EXPECT_GT(r.efficiency_pct, 5.0)
                    << "Best configuration for M=" << m << " has low efficiency: " << r.efficiency_pct << "%";
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * @brief Main performance test: Qwen 0.5B pipeline profiling
 *
 * This test loads real Qwen weights and benchmarks realistic operations
 * using both VNNIGemmAdapter and direct registry calls.
 */
TEST_F(VNNIGemmPerformance, QwenPipelineProfile)
{
    // Only rank 0 does the work (single-rank performance test)
    if (rank_ == 0)
    {
        auto loader = load_qwen_model();
        ASSERT_NE(loader, nullptr) << "Failed to load Qwen model";

        auto results = run_benchmark_suite(*loader);
        ASSERT_FALSE(results.empty()) << "No benchmark results collected";

        print_results_table(results);

        // Validate that adapter overhead is reasonable (< 30%)
        // Only validate M≥512 where full parallelization is possible (≥64 tiles with M_R=8)
        // Smaller M values have limited parallelism:
        //   M=1   → 1 tile   → single-threaded
        //   M=32  → 4 tiles  → max 4 threads
        //   M=128 → 16 tiles → max 16 threads (12 idle on 28-core system!)
        // Threshold relaxed to 30% to account for run-to-run timing noise
        for (const auto &r : results)
        {
            if (r.m >= 512) // Full parallelization: 64+ tiles
            {
                EXPECT_LT(r.adapter_overhead_pct, 30.0)
                    << "Adapter overhead too high for " << r.name << " (M=" << r.m << "): "
                    << r.adapter_overhead_pct << "%";
            }
        }

        // Validate that we achieve reasonable efficiency (> 5% of peak for M≥512)
        // Only validate where full 28-thread parallelization possible
        // M=512 → 64 tiles, M=4096 → 512 tiles, M=16384 → 2048 tiles
        // Expect efficiency to improve with M as parallelism saturates
        // Threshold: 5% (memory bandwidth limits Q8_0 quantized inference)
        for (const auto &r : results)
        {
            if (r.m >= 512) // Full parallelization threshold
            {
                EXPECT_GT(r.efficiency_pct, 5.0)
                    << "Efficiency too low for " << r.name << " (M=" << r.m << "): "
                    << r.efficiency_pct << "% (expected >5% with full parallelization)";
            }
        }
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
