/**
 * @file Perf__Q8_1Gemm_JR_Sweep.cpp
 * @brief JR_UNROLL parameter sweep for Q8_1 GEMM
 */

#include "kernels/cpu/gemm_v2/Q8_1GemmKernel.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"
#include <gtest/gtest.h>
#include <chrono>
#include <iomanip>
#include <vector>

using namespace llaminar2;

// Template to run benchmark for specific JR_UNROLL
template <int JR>
void benchmark_jr_unroll(const char *jr_name)
{
    using TestKernel = Q8_1GemmKernelTemplate<32, 128, 4, 0, 0, JR>;

    const int N = 4096;
    const int K = 4096;
    const int ldc = N;

    std::vector<int> m_sizes = {512, 1024, 2048, 4096, 8192, 16384};

    std::cout << "\n[" << jr_name << "]" << std::endl;
    std::cout << "    M   Time (ms)         GFLOPS        Speedup              Status" << std::endl;
    std::cout << "----------------------------------------------------------------------" << std::endl;

    double baseline_time = 0.0;
    for (int M : m_sizes)
    {
        // Create FP32 activation tensor
        auto A_fp32 = FP32Tensor(std::vector<size_t>{(size_t)M, (size_t)K});
        float *A_data = A_fp32.mutable_data();
        for (int i = 0; i < M * K; ++i)
        {
            A_data[i] = (i % 100) / 100.0f - 0.5f;
        }

        // Create Q8_0 weight tensor with random blocks
        auto B_q8_0 = Q8_0Tensor(std::vector<size_t>{(size_t)K, (size_t)N});
        // B_q8_0 is initialized with default values

        // Create output tensor
        std::vector<float> C(M * N, 0.0f);

        // Warmup
        TestKernel::gemm(M, N, K, A_fp32, B_q8_0, C.data(), ldc);

        // Benchmark
        auto t0 = std::chrono::high_resolution_clock::now();
        TestKernel::gemm(M, N, K, A_fp32, B_q8_0, C.data(), ldc);
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double gflops = (2.0 * M * N * K) / (ms * 1e6);

        if (M == m_sizes[0])
            baseline_time = ms;
        double speedup = baseline_time / ms;

        const char *status;
        if (gflops >= 420.0)
            status = "✅ Good";
        else if (gflops >= 350.0)
            status = "⚠️  Acceptable";
        else
            status = "❌ Needs work";

        std::cout << std::setw(8) << M
                  << std::fixed << std::setprecision(2)
                  << std::setw(14) << ms
                  << std::setw(15) << gflops
                  << std::setw(11) << speedup << "×"
                  << std::setw(20) << status
                  << std::endl;
    }
    std::cout << "----------------------------------------------------------------------" << std::endl;
}

TEST(Q8_1_JR_Sweep, AllConfigurations)
{
    std::cout << "\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  Q8_1 GEMM: JR_UNROLL Parameter Sweep\n";
    std::cout << "  N=4096, K=4096, sA-based compensation\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";

    benchmark_jr_unroll<1>("JR_UNROLL=1");
    benchmark_jr_unroll<2>("JR_UNROLL=2");
    benchmark_jr_unroll<4>("JR_UNROLL=4");
    benchmark_jr_unroll<6>("JR_UNROLL=6");
    benchmark_jr_unroll<8>("JR_UNROLL=8");

    std::cout << "\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  Sweep Complete\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
