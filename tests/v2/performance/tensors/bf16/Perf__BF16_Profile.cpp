/**
 * @file Perf__BF16_Profile.cpp
 * @brief Focused BF16 profiling benchmark for perf analysis
 *
 * Purpose: Isolate BF16 conversion overhead for detailed profiling with linux perf.
 * Tests pure conversion time vs GEMM time to identify bottlenecks.
 *
 * Usage with perf:
 *   perf record -g --call-graph dwarf -- ./v2_perf_bf16_profile
 *   perf report --stdio
 *   perf annotate --stdio convert_fp32_to_bf16
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <chrono>
#include <vector>
#include <numeric>
#include <iomanip>
#include <cmath>

#include "tensors/SIMDHelpers.h"
#include "tensors/Tensors.h"
#include "backends/ComputeBackend.h"
#include "kernels/cpu/gemm/FloatingPointGemmKernel.h"

using namespace llaminar2;

namespace
{

    /**
     * @brief Profile BF16 conversion overhead in isolation
     */
    TEST(BF16Profile, ConversionOverhead)
    {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // Configuration: Medium batch (128×896×896)
        const int M = 128;
        const int K = 896;
        const int N = 896;
        const size_t A_size = M * K;
        const size_t B_size = K * N;
        const size_t C_size = M * N;

        // Allocate FP32 matrices
        std::vector<float> A_fp32(A_size, 1.0f);
        std::vector<float> B_fp32(B_size, 1.0f);
        std::vector<float> C_fp32(C_size, 0.0f);

        // Allocate BF16 matrices
        std::vector<uint16_t> A_bf16(A_size);
        std::vector<uint16_t> B_bf16(B_size);

        // Initialize with random-ish data
        for (size_t i = 0; i < A_size; ++i)
        {
            A_fp32[i] = static_cast<float>(i % 100) / 100.0f;
        }
        for (size_t i = 0; i < B_size; ++i)
        {
            B_fp32[i] = static_cast<float>(i % 100) / 100.0f;
        }

        // ========================================================================
        // TEST 1: Pure BF16 Conversion (FP32 → BF16)
        // ========================================================================

        const int warmup = 10;
        const int iterations = 100;

        if (rank == 0)
        {
            std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
            std::cout << "║         BF16 CONVERSION PROFILING (128×896×896)              ║\n";
            std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
        }

        // Warmup
        for (int i = 0; i < warmup; ++i)
        {
            simd::convert_fp32_to_bf16(A_fp32.data(), A_bf16.data(), A_size);
            simd::convert_fp32_to_bf16(B_fp32.data(), B_bf16.data(), B_size);
        }

        // Profile: FP32 → BF16 conversion
        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < iterations; ++iter)
        {
            simd::convert_fp32_to_bf16(A_fp32.data(), A_bf16.data(), A_size);
            simd::convert_fp32_to_bf16(B_fp32.data(), B_bf16.data(), B_size);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = std::chrono::high_resolution_clock::now();

        double conversion_fp32_to_bf16_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;

        // ========================================================================
        // TEST 2: Pure BF16 → FP32 Conversion
        // ========================================================================

        // Warmup
        for (int i = 0; i < warmup; ++i)
        {
            simd::convert_bf16_to_fp32(A_bf16.data(), A_fp32.data(), A_size);
            simd::convert_bf16_to_fp32(B_bf16.data(), B_fp32.data(), B_size);
        }

        // Profile: BF16 → FP32 conversion
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < iterations; ++iter)
        {
            simd::convert_bf16_to_fp32(A_bf16.data(), A_fp32.data(), A_size);
            simd::convert_bf16_to_fp32(B_bf16.data(), B_fp32.data(), B_size);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        t1 = std::chrono::high_resolution_clock::now();

        double conversion_bf16_to_fp32_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;

        // ========================================================================
        // TEST 3: Pure FP32 GEMM (Baseline)
        // ========================================================================

        // Warmup
        for (int i = 0; i < warmup; ++i)
        {
            llaminar2::gemm::run_onednn_fp32_matmul(
                A_fp32.data(), B_fp32.data(), C_fp32.data(),
                M, N, K, false);
        }

        // Profile: FP32 GEMM
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < iterations; ++iter)
        {
            llaminar2::gemm::run_onednn_fp32_matmul(
                A_fp32.data(), B_fp32.data(), C_fp32.data(),
                M, N, K, false);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        t1 = std::chrono::high_resolution_clock::now();

        double gemm_fp32_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;

        // ========================================================================
        // TEST 4: Full BF16 GEMM (Conversion + Compute)
        // ========================================================================

        // Warmup
        for (int i = 0; i < warmup; ++i)
        {
            simd::convert_bf16_to_fp32(A_bf16.data(), A_fp32.data(), A_size);
            simd::convert_bf16_to_fp32(B_bf16.data(), B_fp32.data(), B_size);
            llaminar2::gemm::run_onednn_fp32_matmul(
                A_fp32.data(), B_fp32.data(), C_fp32.data(),
                M, N, K, false);
        }

        // Profile: Full BF16 GEMM
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < iterations; ++iter)
        {
            // Convert BF16 → FP32
            simd::convert_bf16_to_fp32(A_bf16.data(), A_fp32.data(), A_size);
            simd::convert_bf16_to_fp32(B_bf16.data(), B_fp32.data(), B_size);

            // GEMM
            llaminar2::gemm::run_onednn_fp32_matmul(
                A_fp32.data(), B_fp32.data(), C_fp32.data(),
                M, N, K, false);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        t1 = std::chrono::high_resolution_clock::now();

        double gemm_bf16_full_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;

        // ========================================================================
        // REPORT RESULTS
        // ========================================================================

        if (rank == 0)
        {
            double total_elements = A_size + B_size;
            double bytes_fp32 = total_elements * sizeof(float);
            double bytes_bf16 = total_elements * sizeof(uint16_t);
            double flops = 2.0 * M * N * K; // multiply + add

            std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
            std::cout << "║                     PROFILING RESULTS                          ║\n";
            std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ Operation                          │ Time (ms) │ Overhead (%) ║\n";
            std::cout << "╠════════════════════════════════════╪═══════════╪══════════════╣\n";
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "║ FP32 → BF16 conversion (2 arrays)  │ "
                      << std::setw(9) << conversion_fp32_to_bf16_ms << " │ "
                      << std::setw(11) << (conversion_fp32_to_bf16_ms / gemm_fp32_ms * 100) << "% ║\n";
            std::cout << "║ BF16 → FP32 conversion (2 arrays)  │ "
                      << std::setw(9) << conversion_bf16_to_fp32_ms << " │ "
                      << std::setw(11) << (conversion_bf16_to_fp32_ms / gemm_fp32_ms * 100) << "% ║\n";
            std::cout << "║ Pure FP32 GEMM (baseline)          │ "
                      << std::setw(9) << gemm_fp32_ms << " │ "
                      << std::setw(11) << 100.0 << "% ║\n";
            std::cout << "║ BF16 GEMM (convert + compute)      │ "
                      << std::setw(9) << gemm_bf16_full_ms << " │ "
                      << std::setw(11) << (gemm_bf16_full_ms / gemm_fp32_ms * 100) << "% ║\n";
            std::cout << "╠════════════════════════════════════╪═══════════╪══════════════╣\n";
            std::cout << "║ Conversion overhead (BF16→FP32)    │ "
                      << std::setw(9) << conversion_bf16_to_fp32_ms << " │ "
                      << std::setw(11) << (conversion_bf16_to_fp32_ms / gemm_bf16_full_ms * 100) << "% ║\n";
            std::cout << "║ GEMM portion                       │ "
                      << std::setw(9) << gemm_fp32_ms << " │ "
                      << std::setw(11) << (gemm_fp32_ms / gemm_bf16_full_ms * 100) << "% ║\n";
            std::cout << "╚════════════════════════════════════╧═══════════╧══════════════╝\n\n";

            std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
            std::cout << "║                  BANDWIDTH ANALYSIS                            ║\n";
            std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ Conversion BF16→FP32 bandwidth: "
                      << std::setw(8) << std::setprecision(2)
                      << (bytes_bf16 / 1e9) / (conversion_bf16_to_fp32_ms / 1000.0)
                      << " GB/s        ║\n";
            std::cout << "║ Conversion FP32→BF16 bandwidth: "
                      << std::setw(8) << std::setprecision(2)
                      << (bytes_fp32 / 1e9) / (conversion_fp32_to_bf16_ms / 1000.0)
                      << " GB/s        ║\n";
            std::cout << "║ GEMM throughput:                "
                      << std::setw(8) << std::setprecision(2)
                      << (flops / 1e9) / (gemm_fp32_ms / 1000.0)
                      << " GFLOPS     ║\n";
            std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
        }

        EXPECT_GT(conversion_bf16_to_fp32_ms, 0.0);
        EXPECT_GT(gemm_fp32_ms, 0.0);
    }

    /**
     * @brief Hot loop for perf annotation - BF16 conversion only
     *
     * This test is designed to spend most time in conversion functions
     * so perf can show us the assembly-level hotspots.
     */
    TEST(BF16Profile, HotLoop_Conversion)
    {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // Large arrays for better perf sampling
        const size_t array_size = 128 * 896 * 896; // ~100M elements

        std::vector<float> fp32_data(array_size, 1.0f);
        std::vector<uint16_t> bf16_data(array_size);

        if (rank == 0)
        {
            std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
            std::cout << "║            BF16 CONVERSION HOT LOOP (for perf)               ║\n";
            std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ Array size: " << array_size << " elements (~"
                      << (array_size * sizeof(float) / 1e9) << " GB)              ║\n";
            std::cout << "║ Iterations: 1000 (for detailed perf sampling)                ║\n";
            std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
        }

        // Hot loop: Spend most time here for perf profiling
        const int iterations = 1000;

        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i)
        {
            // This is where perf should show most samples
            simd::convert_fp32_to_bf16(fp32_data.data(), bf16_data.data(), array_size);
            simd::convert_bf16_to_fp32(bf16_data.data(), fp32_data.data(), array_size);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (rank == 0)
        {
            double bytes_transferred = 2.0 * iterations * array_size * sizeof(float); // Read FP32 + write BF16 + read BF16 + write FP32
            double bandwidth_gb_s = (bytes_transferred / 1e9) / (elapsed_ms / 1000.0);

            std::cout << "Hot loop completed:\n";
            std::cout << "  Total time:  " << elapsed_ms << " ms\n";
            std::cout << "  Bandwidth:   " << std::setprecision(2) << bandwidth_gb_s << " GB/s\n";
            std::cout << "  Per iter:    " << elapsed_ms / iterations << " ms\n\n";
            std::cout << "Ready for perf analysis!\n\n";
        }

        EXPECT_GT(elapsed_ms, 0.0);
    }

} // anonymous namespace

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
