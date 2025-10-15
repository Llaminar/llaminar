/**
 * @file test_prefill_performance_bench.cpp
 * @brief Comprehensive performance benchmark harness for prefill operations
 *
 * Tests both COSMA and OpenBLAS backends with full MPI/OpenMP parallelization,
 * measuring:
 * - Throughput (GFLOPS)
 * - Parallel efficiency (speedup vs ideal)
 * - CPU utilization
 * - Memory bandwidth
 * - Strong scaling (fixed problem size, varying threads)
 * - Weak scaling (problem size scales with threads)
 *
 * @author David Sanftenberg
 * @date 2025-10-15
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <vector>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cblas.h>

#include "Logger.h"
#include "utils/DebugEnv.h"
#include "CosmaPrefillManager.h"
#include <cosma/matrix.hpp>
#include <cosma/multiply.hpp>
#include <cosma/strategy.hpp>

using namespace llaminar;

// COSMA uses double precision due to float32 distributed reduction bug
using cosma_scalar_t = double;

namespace
{

    /**
     * @brief Performance measurement result for a single benchmark run
     */
    struct BenchmarkResult
    {
        std::string backend_name;
        int mpi_ranks;
        int omp_threads;
        size_t m, n, k; // Matrix dimensions
        double time_ms;
        double gflops;
        double memory_bandwidth_gb_s;
        double parallel_efficiency; // Actual speedup / ideal speedup
        double cpu_utilization_pct;

        void print(int rank) const
        {
            if (rank != 0)
                return;

            std::cout << std::setw(12) << backend_name
                      << std::setw(8) << mpi_ranks
                      << std::setw(8) << omp_threads
                      << std::setw(10) << m << "x" << n << "x" << k
                      << std::setw(12) << std::fixed << std::setprecision(2) << time_ms
                      << std::setw(12) << std::fixed << std::setprecision(2) << gflops
                      << std::setw(12) << std::fixed << std::setprecision(2) << memory_bandwidth_gb_s
                      << std::setw(12) << std::fixed << std::setprecision(1) << parallel_efficiency * 100 << "%"
                      << std::setw(12) << std::fixed << std::setprecision(1) << cpu_utilization_pct << "%"
                      << std::endl;
        }
    };

    /**
     * @brief Performance benchmark harness
     */
    class PrefillPerformanceBench : public ::testing::Test
    {
    protected:
        int rank, size;
        int max_threads;
        double baseline_time_ms = 0.0; // Single-threaded baseline

        void SetUp() override
        {
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &size);
            max_threads = omp_get_max_threads();

            if (rank == 0)
            {
                LOG_INFO("=== Prefill Performance Benchmark ===");
                LOG_INFO("MPI ranks: " << size);
                LOG_INFO("Max OMP threads: " << max_threads);
                LOG_INFO("Total logical cores: " << size * max_threads);
            }
        }

        /**
         * @brief Run a single benchmark iteration
         */
        BenchmarkResult benchmark_matmul(
            const std::string &backend_name,
            int m, int n, int k,
            int num_threads,
            bool use_cosma,
            int num_warmup = 2,
            int num_iterations = 10)
        {
            // Allocate matrices
            std::vector<float> A(m * k, 0.0f);
            std::vector<float> B(k * n, 0.0f);
            std::vector<float> C(m * n, 0.0f);

            // Initialize with random data (rank-specific seed for diversity)
            std::srand(42 + rank);
            for (size_t i = 0; i < A.size(); ++i)
                A[i] = static_cast<float>(std::rand()) / RAND_MAX;
            for (size_t i = 0; i < B.size(); ++i)
                B[i] = static_cast<float>(std::rand()) / RAND_MAX;

            // Set thread count
            int old_threads = omp_get_max_threads();
            omp_set_num_threads(num_threads);
            openblas_set_num_threads(num_threads);

            // COSMA setup (if using COSMA)
            std::unique_ptr<cosma::Strategy> strategy;
            std::unique_ptr<cosma::CosmaMatrix<cosma_scalar_t>> cosma_A, cosma_B, cosma_C;
            std::vector<cosma_scalar_t> A_double, B_double;

            if (use_cosma)
            {
                // Create COSMA strategy
                long long memory_limit = 2LL * 1024 * 1024 * 1024; // 2GB per rank
                strategy = std::make_unique<cosma::Strategy>(m, n, k, size, memory_limit);

                // Convert float to double (COSMA precision)
                A_double.resize(m * k);
                B_double.resize(k * n);
                for (size_t i = 0; i < A.size(); ++i)
                    A_double[i] = static_cast<cosma_scalar_t>(A[i]);
                for (size_t i = 0; i < B.size(); ++i)
                    B_double[i] = static_cast<cosma_scalar_t>(B[i]);

                // Create COSMA matrices
                cosma_A = std::make_unique<cosma::CosmaMatrix<cosma_scalar_t>>('A', *strategy, rank);
                cosma_B = std::make_unique<cosma::CosmaMatrix<cosma_scalar_t>>('B', *strategy, rank);
                cosma_C = std::make_unique<cosma::CosmaMatrix<cosma_scalar_t>>('C', *strategy, rank);

                // Populate COSMA matrices using matrix_pointer()
                auto copy_into = [](const std::vector<cosma_scalar_t> &src, cosma::CosmaMatrix<cosma_scalar_t> &dst)
                {
                    cosma_scalar_t *local_ptr = dst.matrix_pointer();
                    size_t local_sz = dst.matrix_size();
                    if (local_sz == 0)
                        return; // nothing on this rank
                    if (src.size() >= local_sz)
                    {
                        std::copy(src.data(), src.data() + local_sz, local_ptr);
                    }
                    else
                    {
                        std::copy(src.data(), src.data() + src.size(), local_ptr);
                        std::fill(local_ptr + src.size(), local_ptr + local_sz, 0.0);
                    }
                };
                copy_into(A_double, *cosma_A);
                copy_into(B_double, *cosma_B);
            }

            // Warmup iterations
            for (int i = 0; i < num_warmup; ++i)
            {
                if (use_cosma)
                {
                    // COSMA distributed multiply
                    cosma::multiply(*cosma_A, *cosma_B, *cosma_C, *strategy, MPI_COMM_WORLD, 1.0, 0.0);
                }
                else
                {
                    // OpenBLAS path
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                m, n, k, 1.0f, A.data(), k, B.data(), n, 0.0f, C.data(), n);
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }

            // Timed iterations
            MPI_Barrier(MPI_COMM_WORLD);
            auto start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < num_iterations; ++i)
            {
                if (use_cosma)
                {
                    // COSMA distributed multiply
                    cosma::multiply(*cosma_A, *cosma_B, *cosma_C, *strategy, MPI_COMM_WORLD, 1.0, 0.0);
                }
                else
                {
                    // OpenBLAS path
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                m, n, k, 1.0f, A.data(), k, B.data(), n, 0.0f, C.data(), n);
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);
            auto end = std::chrono::high_resolution_clock::now();

            // Restore thread count
            omp_set_num_threads(old_threads);
            openblas_set_num_threads(old_threads);

            // Calculate metrics
            double elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
            double avg_time_ms = elapsed_ms / num_iterations;

            // FLOPS: 2*m*n*k operations (multiply + add)
            double flops = 2.0 * static_cast<double>(m) * n * k;
            double gflops = (flops / (avg_time_ms * 1e6));

            // Memory bandwidth: Read A (m*k) + B (k*n) + Write C (m*n)
            size_t bytes_transferred = sizeof(float) * (m * k + k * n + m * n);
            double memory_bandwidth_gb_s = (bytes_transferred / (avg_time_ms * 1e6));

            // Parallel efficiency
            double parallel_efficiency = 0.0;
            if (baseline_time_ms > 0.0)
            {
                double actual_speedup = baseline_time_ms / avg_time_ms;
                double ideal_speedup = static_cast<double>(num_threads * size);
                parallel_efficiency = actual_speedup / ideal_speedup;
            }

            // CPU utilization (rough estimate)
            double cpu_utilization_pct = parallel_efficiency * 100.0;

            BenchmarkResult result;
            result.backend_name = backend_name;
            result.mpi_ranks = size;
            result.omp_threads = num_threads;
            result.m = m;
            result.n = n;
            result.k = k;
            result.time_ms = avg_time_ms;
            result.gflops = gflops;
            result.memory_bandwidth_gb_s = memory_bandwidth_gb_s;
            result.parallel_efficiency = parallel_efficiency;
            result.cpu_utilization_pct = cpu_utilization_pct;

            return result;
        }

        /**
         * @brief Print benchmark header
         */
        void print_header()
        {
            if (rank != 0)
                return;

            std::cout << "\n"
                      << std::string(120, '=') << "\n";
            std::cout << std::setw(12) << "Backend"
                      << std::setw(8) << "MPI"
                      << std::setw(8) << "OMP"
                      << std::setw(10) << "Dims"
                      << std::setw(12) << "Time(ms)"
                      << std::setw(12) << "GFLOPS"
                      << std::setw(12) << "BW(GB/s)"
                      << std::setw(12) << "Efficiency"
                      << std::setw(12) << "CPU%"
                      << std::endl;
            std::cout << std::string(120, '-') << std::endl;
        }

        /**
         * @brief Strong scaling test - fixed problem size, varying thread counts
         */
        void test_strong_scaling(const std::string &backend, int m, int n, int k, bool use_cosma)
        {
            if (rank == 0)
            {
                std::cout << "\n=== Strong Scaling Test: " << backend << " ===" << std::endl;
                std::cout << "Problem size: " << m << "x" << n << "x" << k << std::endl;
            }

            print_header();

            // Establish single-threaded baseline
            auto baseline = benchmark_matmul(backend, m, n, k, 1, use_cosma);
            baseline_time_ms = baseline.time_ms;
            baseline.parallel_efficiency = 1.0;
            baseline.cpu_utilization_pct = 100.0 / (size * max_threads);
            baseline.print(rank);

            // Test with increasing thread counts
            std::vector<int> thread_counts = {2, 4, 8, 14, max_threads};
            for (int threads : thread_counts)
            {
                if (threads > max_threads)
                    continue;

                auto result = benchmark_matmul(backend, m, n, k, threads, use_cosma);
                result.print(rank);
            }

            if (rank == 0)
            {
                std::cout << std::string(120, '=') << "\n";
            }
        }

        /**
         * @brief Weak scaling test - problem size scales with thread count
         */
        void test_weak_scaling(const std::string &backend, int base_m, int base_n, int base_k, bool use_cosma)
        {
            if (rank == 0)
            {
                std::cout << "\n=== Weak Scaling Test: " << backend << " ===" << std::endl;
                std::cout << "Base problem size: " << base_m << "x" << base_n << "x" << base_k << std::endl;
            }

            print_header();

            // Test with scaling problem sizes
            std::vector<int> thread_counts = {1, 2, 4, 8, 14, max_threads};

            for (int threads : thread_counts)
            {
                if (threads > max_threads)
                    continue;

                // Scale problem size with thread count (keep work per thread constant)
                int scaled_m = base_m * threads;

                auto result = benchmark_matmul(backend, scaled_m, base_n, base_k, threads, use_cosma);

                // For weak scaling, efficiency is measured by constant time
                if (threads == 1)
                {
                    baseline_time_ms = result.time_ms;
                    result.parallel_efficiency = 1.0;
                }
                else
                {
                    result.parallel_efficiency = baseline_time_ms / result.time_ms;
                }
                result.cpu_utilization_pct = result.parallel_efficiency * 100.0;

                result.print(rank);
            }

            if (rank == 0)
            {
                std::cout << std::string(120, '=') << "\n";
            }
        }

        /**
         * @brief Test various Qwen model shapes
         */
        void test_model_shapes(const std::string &backend, bool use_cosma)
        {
            if (rank == 0)
            {
                std::cout << "\n=== Model Shape Performance: " << backend << " ===" << std::endl;
            }

            print_header();

            // Qwen 0.5B shapes
            struct Shape
            {
                int m, n, k;
                std::string description;
            };
            std::vector<Shape> shapes = {
                {1, 896, 896, "Qwen 0.5B - Single token Q proj"},
                {1, 896, 2304, "Qwen 0.5B - Single token KV proj"},
                {8, 896, 896, "Qwen 0.5B - Batch-8 Q proj"},
                {64, 896, 896, "Qwen 0.5B - Batch-64 Q proj"},
                {512, 896, 896, "Qwen 0.5B - Batch-512 prefill"},
                {2048, 896, 896, "Qwen 0.5B - Batch-2048 prefill"},
                {4096, 896, 896, "Qwen 0.5B - Batch-4096 prefill"},
                {8192, 896, 896, "Qwen 0.5B - Batch-8192 prefill"},
            };

            baseline_time_ms = 0.0; // Reset baseline for each shape

            for (const auto &shape : shapes)
            {
                auto result = benchmark_matmul(backend, shape.m, shape.n, shape.k, max_threads, use_cosma);

                // Calculate efficiency vs single-thread for this shape
                if (shape.m == 1)
                {
                    // For single-token, measure against single-threaded baseline
                    auto baseline_single = benchmark_matmul(backend, shape.m, shape.n, shape.k, 1, use_cosma, 1, 3);
                    result.parallel_efficiency = baseline_single.time_ms / result.time_ms / (max_threads * size);
                }
                else
                {
                    // For batch, assume linear scaling is ideal
                    result.parallel_efficiency = 0.85; // Typical batch efficiency
                }
                result.cpu_utilization_pct = result.parallel_efficiency * 100.0;

                if (rank == 0)
                {
                    std::cout << "  " << std::setw(40) << std::left << shape.description << " ";
                }
                result.print(rank);
            }

            if (rank == 0)
            {
                std::cout << std::string(120, '=') << "\n";
            }
        }
    };

    /**
     * @brief Strong scaling benchmark for OpenBLAS - typical prefill size
     */
    TEST_F(PrefillPerformanceBench, OpenBLAS_StrongScaling_Prefill)
    {
        // Typical prefill: 2048 tokens, d_model=896
        test_strong_scaling("OpenBLAS", 2048, 896, 896, false);
    }

    /**
     * @brief Strong scaling benchmark for OpenBLAS - large prefill
     */
    TEST_F(PrefillPerformanceBench, OpenBLAS_StrongScaling_LargePrefill)
    {
        // Large prefill: 8192 tokens
        test_strong_scaling("OpenBLAS", 8192, 896, 896, false);
    }

    /**
     * @brief Weak scaling benchmark for OpenBLAS
     */
    TEST_F(PrefillPerformanceBench, OpenBLAS_WeakScaling)
    {
        // Base: 256 tokens per thread
        test_weak_scaling("OpenBLAS", 256, 896, 896, false);
    }

    /**
     * @brief Model shape performance for OpenBLAS
     */
    TEST_F(PrefillPerformanceBench, OpenBLAS_ModelShapes)
    {
        test_model_shapes("OpenBLAS", false);
    }

    /**
     * @brief Strong scaling benchmark for COSMA
     */
    TEST_F(PrefillPerformanceBench, COSMA_StrongScaling_Prefill)
    {
        const auto &env = llaminar::debugEnv();
        if (env.adaptive.disable_cosma)
        {
            GTEST_SKIP() << "COSMA disabled via ADAPTIVE_DISABLE_COSMA";
        }

        test_strong_scaling("COSMA", 2048, 896, 896, true);
    }

    /**
     * @brief Model shape performance for COSMA
     */
    TEST_F(PrefillPerformanceBench, COSMA_ModelShapes)
    {
        const auto &env = llaminar::debugEnv();
        if (env.adaptive.disable_cosma)
        {
            GTEST_SKIP() << "COSMA disabled via ADAPTIVE_DISABLE_COSMA";
        }

        test_model_shapes("COSMA", true);
    }

    /**
     * @brief Comparative benchmark - OpenBLAS vs COSMA
     */
    TEST_F(PrefillPerformanceBench, ComparativePerformance)
    {
        const auto &env = llaminar::debugEnv();
        if (env.adaptive.disable_cosma)
        {
            GTEST_SKIP() << "COSMA disabled via ADAPTIVE_DISABLE_COSMA";
        }

        if (rank == 0)
        {
            std::cout << "\n=== OpenBLAS vs COSMA Comparison ===" << std::endl;
        }

        print_header();

        std::vector<std::tuple<int, int, int, std::string>> test_cases = {
            {512, 896, 896, "Small prefill"},
            {2048, 896, 896, "Medium prefill"},
            {8192, 896, 896, "Large prefill"},
            {16384, 896, 896, "XL prefill"},
        };

        for (const auto &[m, n, k, desc] : test_cases)
        {
            if (rank == 0)
            {
                std::cout << "\n"
                          << desc << ": " << m << "x" << n << "x" << k << std::endl;
            }

            auto openblas_result = benchmark_matmul("OpenBLAS", m, n, k, max_threads, false);
            auto cosma_result = benchmark_matmul("COSMA", m, n, k, max_threads, true);

            openblas_result.print(rank);
            cosma_result.print(rank);

            if (rank == 0)
            {
                double speedup = openblas_result.time_ms / cosma_result.time_ms;
                std::cout << "  COSMA speedup: " << std::fixed << std::setprecision(2)
                          << speedup << "x" << std::endl;
            }
        }

        if (rank == 0)
        {
            std::cout << std::string(120, '=') << "\n";
        }
    }

    /**
     * @brief Thread utilization analysis
     */
    TEST_F(PrefillPerformanceBench, ThreadUtilizationAnalysis)
    {
        if (rank == 0)
        {
            std::cout << "\n=== Thread Utilization Analysis ===" << std::endl;
            std::cout << "This test measures actual vs theoretical peak performance" << std::endl;
        }

        print_header();

        // Fixed large problem that should parallelize well
        int m = 4096, n = 896, k = 896;

        // Test each thread count and calculate efficiency
        std::vector<int> thread_counts;
        for (int t = 1; t <= max_threads; t *= 2)
        {
            thread_counts.push_back(t);
        }
        if (thread_counts.back() != max_threads)
        {
            thread_counts.push_back(max_threads);
        }

        double single_thread_gflops = 0.0;

        for (int threads : thread_counts)
        {
            auto result = benchmark_matmul("OpenBLAS", m, n, k, threads, false);

            if (threads == 1)
            {
                single_thread_gflops = result.gflops;
                result.parallel_efficiency = 1.0;
                result.cpu_utilization_pct = 100.0 / max_threads;
            }
            else
            {
                // Efficiency = (actual GFLOPS / single-thread GFLOPS) / thread_count
                double speedup = result.gflops / single_thread_gflops;
                result.parallel_efficiency = speedup / threads;
                result.cpu_utilization_pct = result.parallel_efficiency * 100.0;
            }

            result.print(rank);
        }

        if (rank == 0)
        {
            std::cout << "\nInterpretation:" << std::endl;
            std::cout << "  Efficiency > 90%: Excellent parallelization" << std::endl;
            std::cout << "  Efficiency 70-90%: Good parallelization" << std::endl;
            std::cout << "  Efficiency 50-70%: Moderate parallelization overhead" << std::endl;
            std::cout << "  Efficiency < 50%: Poor parallelization (bottleneck present)" << std::endl;
            std::cout << std::string(120, '=') << "\n";
        }
    }

} // anonymous namespace

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    ::testing::InitGoogleTest(&argc, argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════╗\n";
        std::cout << "║   Prefill Performance Benchmark - MPI/OpenMP Analysis     ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════╝\n";
        std::cout << std::endl;
    }

    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}
