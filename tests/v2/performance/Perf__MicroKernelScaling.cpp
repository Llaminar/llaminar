/**
 * @file Perf__MicroKernelScaling.cpp
 * @brief Performance test for micro-kernel GEMM scaling across MPI ranks
 *
 * Tests multi-socket performance by splitting GEMM work across MPI ranks.
 * Each rank processes a portion of the rows (m dimension) and we measure
 * the combined throughput.
 *
 * Test matrix: m ∈ {1, 8, 32, 128, 512, 1024, 2048, 4096}, n=896, k=896
 *
 * MPI Distribution Strategy:
 * - Rank 0 processes rows [0, m/2)
 * - Rank 1 processes rows [m/2, m)
 * - Each rank runs auto-tuner independently for its local shape
 * - Gather timing results to rank 0 for combined GFLOPS calculation
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <chrono>
#include <iomanip>
#include <vector>
#include <algorithm>

#include "kernels/cpu/gemm/GemmAutoTuner.h"
#include "kernels/cpu/gemm/GemmMicroKernelAdapter.h"
#include "tensors/TensorKernels.h"

using namespace llaminar2;
using namespace llaminar::v2::kernels;

namespace
{

    /**
     * @brief Test block decoder for performance testing
     *
     * Simple decoder that returns incrementing float values.
     * Used to test micro-kernel performance without loading a real model.
     */
    class TestBlockDecoder : public ITensorGemmTileDataProvider
    {
    public:
        TestBlockDecoder(int k, int n) : k_(k), n_(n) {}

        __attribute__((always_inline)) void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override
        {
            // Simple decode: return incrementing values
            for (int i = 0; i < 32; ++i)
            {
                output[i] = static_cast<float>(row_idx * 1000 + k_block_offset * 32 + i);
            }
        }

        const void *get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
        {
            // Not used in this test
            return nullptr;
        }

        size_t decoder_rows() const override { return n_; }
        size_t decoder_cols() const override { return k_; }
        size_t block_size() const override { return 32; }

    private:
        int k_;
        int n_;
    };

    /**
     * @brief Performance result for a single shape
     */
    struct PerfResult
    {
        int m, n, k;             // Problem shape
        int local_m;             // Rows per rank
        double local_time_ms;    // Time on this rank (ms)
        double global_time_ms;   // Max time across all ranks (ms)
        double local_gflops;     // GFLOPS on this rank
        double combined_gflops;  // Combined GFLOPS (sum of all ranks)
        std::string best_kernel; // Best kernel selected

        void print(int rank, int world_size) const
        {
            if (rank == 0)
            {
                std::cout << std::fixed << std::setprecision(2);
                std::cout << "  Shape: [" << std::setw(4) << m << " × " << std::setw(4) << n
                          << " × " << std::setw(4) << k << "]";
                std::cout << " | Local m: " << std::setw(4) << local_m;
                std::cout << " | Time: " << std::setw(8) << global_time_ms << " ms";
                std::cout << " | Single-rank: " << std::setw(7) << local_gflops << " GFLOPS";
                std::cout << " | Combined (" << world_size << " ranks): "
                          << std::setw(7) << combined_gflops << " GFLOPS";
                std::cout << " | " << best_kernel << "\n";
            }
        }
    };

    /**
     * @brief Run GEMM benchmark on local portion of rows
     */
    PerfResult benchmarkShape(int m_global, int n, int k, int rank, int world_size)
    {
        // Distribute rows across ranks
        int rows_per_rank = m_global / world_size;
        int local_m = rows_per_rank + (rank < m_global % world_size ? 1 : 0);

        // Create decoder
        TestBlockDecoder decoder(k, n);

        // Auto-tune for local shape
        auto &tuner = GemmAutoTuner::instance();
        tuner.clearCache();

        // Register variants (each rank does this independently)
        auto variants = kernels::gemm::registerMicroKernelVariants(&decoder);
        for (auto &variant : variants)
        {
            tuner.registerVariant(std::move(variant));
        }

        // Get optimal kernel for local shape
        auto *kernel = tuner.getOptimalKernel(local_m, n, k);
        EXPECT_NE(kernel, nullptr) << "Failed to get kernel for shape ["
                                   << local_m << ", " << n << ", " << k << "]";

        std::string kernel_name = "unknown";
        if (kernel)
        {
            auto config = kernel->config();
            std::ostringstream oss;
            oss << "tile" << config.tile_m << "x" << config.tile_n
                << "_u" << config.unroll_factor << "_p" << config.prefetch_blocks;
            kernel_name = oss.str();
        }

        // Allocate local matrices
        std::vector<float> A(local_m * k, 1.0f);
        std::vector<float> C(local_m * n, 0.0f);

        // Warmup
        if (kernel)
        {
            kernel->multiply(A.data(), C.data(), local_m, n, k, &decoder, 1.0f, 0.0f);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // Benchmark (5 iterations)
        constexpr int NUM_ITERS = 5;
        auto start = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < NUM_ITERS; ++iter)
        {
            if (kernel)
            {
                kernel->multiply(A.data(), C.data(), local_m, n, k, &decoder, 1.0f, 0.0f);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        MPI_Barrier(MPI_COMM_WORLD);

        // Compute local metrics
        double local_time_ms = std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERS;
        double local_flops = 2.0 * local_m * n * k; // multiply-add = 2 ops
        double local_gflops = (local_flops / (local_time_ms * 1e6));

        // Gather timing to rank 0
        double global_max_time_ms;
        MPI_Reduce(&local_time_ms, &global_max_time_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        // Gather GFLOPS to rank 0
        double total_gflops;
        MPI_Reduce(&local_gflops, &total_gflops, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // Gather kernel names (just use rank 0's kernel for reporting)

        PerfResult result;
        result.m = m_global;
        result.n = n;
        result.k = k;
        result.local_m = local_m;
        result.local_time_ms = local_time_ms;
        result.global_time_ms = global_max_time_ms;
        result.local_gflops = local_gflops;
        result.combined_gflops = total_gflops;
        result.best_kernel = kernel_name;

        return result;
    }

} // anonymous namespace

/**
 * @brief Test: Multi-socket scaling performance
 */
TEST(Perf__MicroKernelScaling, TwoSocketScaling)
{
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ASSERT_EQ(world_size, 2) << "This test requires exactly 2 MPI ranks (two sockets)";

    // Test matrix: typical Qwen inference shapes
    const std::vector<int> test_shapes_m = {
        1,    // Single token decode
        8,    // Small batch
        32,   // Medium batch
        128,  // Large batch
        512,  // Prefill
        1024, // Large prefill
        2048, // Very large prefill
        4096, // Extreme prefill
    };

    constexpr int N = 896;
    constexpr int K = 896;

    if (rank == 0)
    {
        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║ MICRO-KERNEL SCALING PERFORMANCE (2 MPI Ranks / 2 Sockets)                            ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Each rank processes half the rows (m/2), n=" << N << ", k=" << K << "                                ║\n";
        std::cout << "║ Combined GFLOPS = sum of both ranks' throughput                                       ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════════════════════════════╝\n\n";
    }

    std::vector<PerfResult> results;

    for (int m : test_shapes_m)
    {
        auto result = benchmarkShape(m, N, K, rank, world_size);
        results.push_back(result);
        result.print(rank, world_size);
    }

    if (rank == 0)
    {
        // Find peak performance
        auto peak = std::max_element(results.begin(), results.end(),
                                     [](const PerfResult &a, const PerfResult &b)
                                     {
                                         return a.combined_gflops < b.combined_gflops;
                                     });

        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║ PEAK PERFORMANCE                                                                       ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Shape: [" << std::setw(4) << peak->m << " × " << std::setw(4) << peak->n
                  << " × " << std::setw(4) << peak->k << "]";
        std::cout << " | Combined: " << std::fixed << std::setprecision(2)
                  << std::setw(7) << peak->combined_gflops << " GFLOPS";
        std::cout << " | Kernel: " << peak->best_kernel;
        std::cout << "                  ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════════════════════════════╝\n";

        // Scaling efficiency analysis
        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║ SCALING EFFICIENCY                                                                     ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════════════════════════════╣\n";

        for (const auto &r : results)
        {
            double scaling_efficiency = (r.combined_gflops / (r.local_gflops * world_size)) * 100.0;
            std::cout << "║ m=" << std::setw(4) << r.m
                      << " | Efficiency: " << std::setw(5) << std::fixed << std::setprecision(1)
                      << scaling_efficiency << "% ";

            if (scaling_efficiency >= 95.0)
            {
                std::cout << "(Excellent - Near-linear scaling)";
            }
            else if (scaling_efficiency >= 85.0)
            {
                std::cout << "(Good)";
            }
            else if (scaling_efficiency >= 70.0)
            {
                std::cout << "(Fair - Some overhead)";
            }
            else
            {
                std::cout << "(Poor - Significant overhead)";
            }

            // Pad to column width
            std::cout << std::string(std::max(0, 52 - (int)std::to_string((int)scaling_efficiency).length()), ' ') << "║\n";
        }

        std::cout << "╚════════════════════════════════════════════════════════════════════════════════════════╝\n";
    }
}

/**
 * @brief Test: Single-rank baseline (for comparison)
 */
TEST(Perf__MicroKernelScaling, SingleRankBaseline)
{
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Only run on rank 0
    if (rank != 0)
    {
        return;
    }

    const std::vector<int> test_shapes_m = {
        1, 8, 32, 128, 512, 1024, 2048, 4096};

    constexpr int N = 896;
    constexpr int K = 896;

    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ SINGLE-RANK BASELINE PERFORMANCE                                                       ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Full shape processed on single rank (for comparison)                                  ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════════════════╝\n\n";

    TestBlockDecoder decoder(K, N);
    auto &tuner = GemmAutoTuner::instance();

    for (int m : test_shapes_m)
    {
        tuner.clearCache();
        auto variants = kernels::gemm::registerMicroKernelVariants(&decoder);
        for (auto &variant : variants)
        {
            tuner.registerVariant(std::move(variant));
        }

        auto *kernel = tuner.getOptimalKernel(m, N, K);
        ASSERT_NE(kernel, nullptr);

        std::vector<float> A(m * K, 1.0f);
        std::vector<float> C(m * N, 0.0f);

        // Warmup
        kernel->multiply(A.data(), C.data(), m, N, K, &decoder, 1.0f, 0.0f);

        // Benchmark
        constexpr int NUM_ITERS = 5;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_ITERS; ++i)
        {
            kernel->multiply(A.data(), C.data(), m, N, K, &decoder, 1.0f, 0.0f);
        }
        auto end = std::chrono::high_resolution_clock::now();

        double time_ms = std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERS;
        double flops = 2.0 * m * N * K;
        double gflops = flops / (time_ms * 1e6);

        auto config = kernel->config();

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  m=" << std::setw(4) << m
                  << " | Time: " << std::setw(8) << time_ms << " ms"
                  << " | " << std::setw(7) << gflops << " GFLOPS"
                  << " | tile" << config.tile_m << "x" << config.tile_n
                  << "_u" << config.unroll_factor << "_p" << config.prefetch_blocks << "\n";
    }

    std::cout << "\n";
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);

    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}
