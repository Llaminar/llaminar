/**
 * @file Perf__CuTePipelining.cu
 * @brief Phase 2.7 Software Pipelining benchmark vs Phase 2.5 baseline
 *
 * @author David Sanftenberg
 * @date November 1, 2025
 *
 * PURPOSE:
 * Validate whether triple-buffered software pipelining provides measurable
 * performance benefit over sequential copy→compute execution.
 *
 * USER EXPECTATION:
 * - Single token (m=1): Minimal/no improvement (low parallelism)
 * - Batch 32+: Modest improvement (moderate parallelism)
 * - Batch 128+: Good improvement (1.5-2× target)
 * - FFN (large n): Good improvement (1.5-2× target)
 */

#include "../../src/v2/kernels/cuda/CudaGemmKernel.cuh"
#include "../../src/v2/kernels/cuda/CudaGemmKernelTensorCorePipelined.cuh"
#include "../../src/v2/kernels/cuda/IQ4_NL_BlockDecoder.h"
#include "../../src/v2/tensors/FP16Utils.h"
#include <gtest/gtest.h>
#include <vector>
#include <iomanip>
#include <chrono>

using namespace llaminar2::cuda;

/**
 * @brief Benchmark result for pipelining comparison
 */
struct PipeliningBenchmarkResult
{
    int m, n, k;
    double phase25_gflops;
    double phase27_gflops;
    double phase25_ms;
    double phase27_ms;
    double speedup;
};

/**
 * @brief Test fixture for pipelining benchmarks
 */
class CuTePipeliningBenchmark : public ::testing::Test
{
protected:
    void SetUp() override
    {
        cudaGetDevice(&device_id_);
        cudaGetDeviceProperties(&device_props_, device_id_);

        cudaStreamCreate(&stream_);
        cudaEventCreate(&start_event_);
        cudaEventCreate(&stop_event_);

        warmup_iterations_ = 10;
        benchmark_iterations_ = 100;
    }

    void TearDown() override
    {
        cudaGetLastError();

        if (stream_)
        {
            cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
        }
        if (start_event_)
            cudaEventDestroy(start_event_);
        if (stop_event_)
            cudaEventDestroy(stop_event_);

        if (test_A_device_)
            cudaFree(test_A_device_);
        if (test_B_device_)
            cudaFree(test_B_device_);
        if (test_C_phase25_device_)
            cudaFree(test_C_phase25_device_);
        if (test_C_phase27_device_)
            cudaFree(test_C_phase27_device_);

        cudaDeviceReset();
    }

    void allocateTestData(int m, int n, int k)
    {
        // Free existing allocations
        if (test_A_device_)
        {
            cudaFree(test_A_device_);
            test_A_device_ = nullptr;
        }
        if (test_B_device_)
        {
            cudaFree(test_B_device_);
            test_B_device_ = nullptr;
        }
        if (test_C_phase25_device_)
        {
            cudaFree(test_C_phase25_device_);
            test_C_phase25_device_ = nullptr;
        }
        if (test_C_phase27_device_)
        {
            cudaFree(test_C_phase27_device_);
            test_C_phase27_device_ = nullptr;
        }

        // Allocate device memory
        cudaMalloc(&test_A_device_, m * k * sizeof(cutlass::half_t));
        cudaMalloc(&test_B_device_, (k / 32) * n * sizeof(IQ4_NLBlock));
        cudaMalloc(&test_C_phase25_device_, m * n * sizeof(float));
        cudaMalloc(&test_C_phase27_device_, m * n * sizeof(float));

        // Initialize host data
        std::vector<cutlass::half_t> h_A(m * k);
        std::vector<IQ4_NLBlock> h_B((k / 32) * n);

        for (auto &val : h_A)
            val = cutlass::half_t(0.1f);
        for (auto &block : h_B)
        {
            block.d = __float2half_rn(0.1f);
            for (int i = 0; i < 16; ++i)
                block.qs[i] = 128;
        }

        // Copy to device
        cudaMemcpy(test_A_device_, h_A.data(), m * k * sizeof(cutlass::half_t), cudaMemcpyHostToDevice);
        cudaMemcpy(test_B_device_, h_B.data(), (k / 32) * n * sizeof(IQ4_NLBlock), cudaMemcpyHostToDevice);
        cudaMemset(test_C_phase25_device_, 0, m * n * sizeof(float));
        cudaMemset(test_C_phase27_device_, 0, m * n * sizeof(float));
    }

    PipeliningBenchmarkResult runComparison(int m, int n, int k, const std::string &label)
    {
        allocateTestData(m, n, k);

        IQ4_NL_Decoder<IQ4_NLBlock> decoder(test_B_device_, n, k / 32);

        // Warmup
        for (int i = 0; i < warmup_iterations_; ++i)
        {
            launchQuantizedGemmCuTe<cutlass::half_t, IQ4_NL_Decoder<IQ4_NLBlock>>(
                test_A_device_, test_C_phase25_device_, m, n, k, decoder, stream_);
            launchQuantizedGemmCuTePipelined<cutlass::half_t, IQ4_NL_Decoder<IQ4_NLBlock>>(
                test_A_device_, test_C_phase27_device_, m, n, k, decoder, stream_);
        }
        cudaStreamSynchronize(stream_);

        // Benchmark Phase 2.5 (sequential)
        cudaEventRecord(start_event_, stream_);
        for (int i = 0; i < benchmark_iterations_; ++i)
        {
            launchQuantizedGemmCuTe<cutlass::half_t, IQ4_NL_Decoder<IQ4_NLBlock>>(
                test_A_device_, test_C_phase25_device_, m, n, k, decoder, stream_);
        }
        cudaEventRecord(stop_event_, stream_);
        cudaEventSynchronize(stop_event_);

        float ms_phase25;
        cudaEventElapsedTime(&ms_phase25, start_event_, stop_event_);
        ms_phase25 /= benchmark_iterations_;

        // Benchmark Phase 2.7 (pipelined)
        cudaEventRecord(start_event_, stream_);
        for (int i = 0; i < benchmark_iterations_; ++i)
        {
            launchQuantizedGemmCuTePipelined<cutlass::half_t, IQ4_NL_Decoder<IQ4_NLBlock>>(
                test_A_device_, test_C_phase27_device_, m, n, k, decoder, stream_);
        }
        cudaEventRecord(stop_event_, stream_);
        cudaEventSynchronize(stop_event_);

        float ms_phase27;
        cudaEventElapsedTime(&ms_phase27, start_event_, stop_event_);
        ms_phase27 /= benchmark_iterations_;

        // Compute metrics
        double flops = 2.0 * m * n * k;
        double gflops_phase25 = (flops / 1e9) / (ms_phase25 / 1000.0);
        double gflops_phase27 = (flops / 1e9) / (ms_phase27 / 1000.0);
        double speedup = gflops_phase27 / gflops_phase25;

        // Validate correctness (outputs should match)
        std::vector<float> h_C_phase25(m * n);
        std::vector<float> h_C_phase27(m * n);
        cudaMemcpy(h_C_phase25.data(), test_C_phase25_device_, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_C_phase27.data(), test_C_phase27_device_, m * n * sizeof(float), cudaMemcpyDeviceToHost);

        float max_diff = 0.0f;
        for (size_t i = 0; i < h_C_phase25.size(); ++i)
        {
            float diff = std::abs(h_C_phase25[i] - h_C_phase27[i]);
            max_diff = std::max(max_diff, diff);
        }

        EXPECT_LT(max_diff, 1e-3f) << "Phase 2.5 and 2.7 outputs differ";

        // Print results
        std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║ " << std::left << std::setw(60) << label << " ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Shape: m=" << m << ", n=" << n << ", k=" << k << std::string(40 - std::to_string(m).size() - std::to_string(n).size() - std::to_string(k).size(), ' ') << "║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════╝\n";

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\n┌──────────────┬─────────────┬─────────────┬──────────┐\n";
        std::cout << "│ Kernel       │ Time (ms)   │ GFLOPS      │ Speedup  │\n";
        std::cout << "├──────────────┼─────────────┼─────────────┼──────────┤\n";
        std::cout << "│ Phase 2.5    │ " << std::setw(11) << ms_phase25
                  << " │ " << std::setw(11) << gflops_phase25 << " │ 1.00×    │\n";
        std::cout << "│ Phase 2.7    │ " << std::setw(11) << ms_phase27
                  << " │ " << std::setw(11) << gflops_phase27
                  << " │ " << std::setw(7) << speedup << "× │\n";
        std::cout << "└──────────────┴─────────────┴─────────────┴──────────┘\n";
        std::cout << "\nMax output difference: " << max_diff << " (tolerance: 1e-3)\n";

        // Interpretation
        if (speedup > 1.3)
        {
            std::cout << "✅ Significant improvement from pipelining (" << std::setprecision(1)
                      << (speedup - 1.0) * 100 << "% faster)\n";
        }
        else if (speedup > 1.05)
        {
            std::cout << "✓ Modest improvement from pipelining (" << std::setprecision(1)
                      << (speedup - 1.0) * 100 << "% faster)\n";
        }
        else
        {
            std::cout << "⚠ Minimal/no benefit from pipelining\n";
        }

        return {m, n, k, gflops_phase25, gflops_phase27, ms_phase25, ms_phase27, speedup};
    }

    int device_id_ = 0;
    cudaDeviceProp device_props_;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_event_ = nullptr;
    cudaEvent_t stop_event_ = nullptr;

    cutlass::half_t *test_A_device_ = nullptr;
    IQ4_NLBlock *test_B_device_ = nullptr;
    float *test_C_phase25_device_ = nullptr;
    float *test_C_phase27_device_ = nullptr;

    int warmup_iterations_ = 10;
    int benchmark_iterations_ = 100;
};

// ==================== Test Cases ====================

TEST_F(CuTePipeliningBenchmark, SingleToken_0_5B)
{
    runComparison(1, 896, 896, "0.5B Single Token (m=1) - Expect minimal improvement");
}

TEST_F(CuTePipeliningBenchmark, Batch32_7B)
{
    runComparison(32, 4096, 4096, "7B Batch 32 (m=32) - Expect modest improvement");
}

TEST_F(CuTePipeliningBenchmark, Batch128_7B)
{
    runComparison(128, 4096, 4096, "7B Batch 128 (m=128) - Expect good improvement");
}

TEST_F(CuTePipeliningBenchmark, FFN_14B)
{
    runComparison(128, 27648, 5120, "14B FFN Gate (m=128) - Expect good improvement");
}

// ==================== Main Entry Point ====================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║       Phase 2.7 Software Pipelining Benchmark Suite           ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Comparing:                                                     ║\n";
    std::cout << "║   Phase 2.5: Sequential copy→compute (baseline)                ║\n";
    std::cout << "║   Phase 2.7: Pipelined copy+compute (triple buffering)         ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Expected Results:                                              ║\n";
    std::cout << "║   • Single token (m=1):      Minimal improvement               ║\n";
    std::cout << "║   • Batch 32+ (m≥32):        Modest improvement                ║\n";
    std::cout << "║   • Batch 128+ (m≥128):      1.5-2× improvement                ║\n";
    std::cout << "║   • FFN (large n):           1.5-2× improvement                ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";

    return RUN_ALL_TESTS();
}
