#include <gtest/gtest.h>
#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <random>
#include <memory>
#include <algorithm>

#include "tensors/Tensors.h"
#include "kernels/cpu/native_vnni/CPUNativeVNNIGemmKernel.h"
#include "utils/DebugEnv.h"

using namespace llaminar2;

class GemmTuningSweep : public ::testing::Test
{
protected:
    int rank_ = 0;

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    }

    // Helper to create a dummy Q4_0 tensor
    std::shared_ptr<Q4_0Tensor> create_dummy_tensor(int rows, int cols)
    {
        size_t blocks_per_row = (cols + 31) / 32;
        size_t total_blocks = rows * blocks_per_row;
        size_t total_bytes = total_blocks * sizeof(Q4_0Block);

        std::vector<uint8_t> raw_data(total_bytes);
        // Fill with 0x11 to have valid packed nibbles
        std::fill(raw_data.begin(), raw_data.end(), 0x11);

        // Set scales to 1.0 (FP16 0x3C00)
        Q4_0Block *blocks = reinterpret_cast<Q4_0Block *>(raw_data.data());
        for (size_t i = 0; i < total_blocks; ++i)
        {
            blocks[i].d = 0x3C00;
        }

        return std::make_shared<Q4_0Tensor>(std::vector<size_t>{(size_t)rows, (size_t)cols}, raw_data);
    }

    double benchmark_kernel(int M, int N, int K, std::shared_ptr<Q4_0Tensor> tensor)
    {
        // Create input A (M x K)
        std::vector<float> A(M * K);
        std::fill(A.begin(), A.end(), 0.5f);

        // Create output C (M x N)
        std::vector<float> C(M * N);

        // Create kernel
        MPIContext mpi_ctx(rank_, 1, MPI_COMM_WORLD);
        auto kernel = tensor->createGemm();

        // Warmup
        for (int i = 0; i < 3; ++i)
        {
            kernel->multiply(A.data(), C.data(), M, N, K);
        }

        // Benchmark
        int iters = 5; // Keep it short for sweep
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i)
        {
            kernel->multiply(A.data(), C.data(), M, N, K);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        auto end = std::chrono::high_resolution_clock::now();

        double duration_sec = std::chrono::duration<double>(end - start).count();
        double gflops = (2.0 * M * N * K * iters) / (duration_sec * 1e9);

        return gflops;
    }

    void run_sweep(int M, int N, int K, const std::string &layer_name)
    {
        auto tensor = create_dummy_tensor(N, K);

        if (rank_ == 0)
        {
            std::cout << "Layer,M,N,K,L2_Limit_Pct,K_Tile_Threshold_Pct,Target_B_Size_Pct,GFLOPS" << std::endl;
        }

        std::vector<float> l2_limits = {0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
        std::vector<float> k_thresholds = {0.4f, 0.5f, 0.6f, 0.75f};
        std::vector<float> target_b_sizes = {0.25f, 0.5f, 0.75f, 0.9f};

        for (float l2 : l2_limits)
        {
            for (float k_thresh : k_thresholds)
            {
                for (float target_b : target_b_sizes)
                {
                    // Set env vars
                    setenv("LLAMINAR_GEMM_L2_LIMIT_PCT", std::to_string(l2).c_str(), 1);
                    setenv("LLAMINAR_GEMM_K_TILE_THRESHOLD_PCT", std::to_string(k_thresh).c_str(), 1);
                    setenv("LLAMINAR_GEMM_TARGET_B_SIZE_PCT", std::to_string(target_b).c_str(), 1);

                    // Reload DebugEnv
                    mutableDebugEnv().reload();

                    // Run benchmark
                    double gflops = benchmark_kernel(M, N, K, tensor);

                    if (rank_ == 0)
                    {
                        std::cout << layer_name << "," << M << "," << N << "," << K << ","
                                  << l2 << "," << k_thresh << "," << target_b << ","
                                  << gflops << std::endl;
                    }
                }
            }
        }
    }
};

TEST_F(GemmTuningSweep, Qwen32B_AttnOutput_M512)
{
    run_sweep(512, 5120, 5120, "Qwen32B_AttnOutput");
}

TEST_F(GemmTuningSweep, Qwen32B_FFNDown_M512)
{
    run_sweep(512, 5120, 27392, "Qwen32B_FFNDown");
}

TEST_F(GemmTuningSweep, Qwen32B_AttnOutput_M1)
{
    run_sweep(1, 5120, 5120, "Qwen32B_AttnOutput");
}

TEST_F(GemmTuningSweep, Qwen32B_FFNDown_M1)
{
    run_sweep(1, 5120, 27392, "Qwen32B_FFNDown");
}

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
