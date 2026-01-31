#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "kernels/cpu/gemm_v4/QuantisedGemmKernel.h"
#include <vector>
#include <random>
#include <cmath>
#include <mpi.h>

using namespace llaminar2;
using namespace llaminar2::gemm_v4;

void RunSoftmaxTest(int M)
{
    // Dimensions
    int N = 128; // Multiple blocks
    int K = 64;

    // Create random weights (N x K)
    std::vector<float> weights_fp32(N * K);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto &x : weights_fp32)
        x = dist(gen);

    // Quantize weights
    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});

    // Create kernel
    MPIContext ctx(0, 1, MPI_COMM_WORLD);
    auto generic_kernel = weights_tensor->createGemm();
    auto kernel = dynamic_cast<QuantisedGemmKernel *>(generic_kernel.get());
    ASSERT_NE(kernel, nullptr);

    // Create random input A (M x K)
    std::vector<float> A(M * K);
    for (auto &x : A)
        x = dist(gen);

    // Output buffers
    std::vector<float> C(M * N, 0.0f);

    // Softmax buffers
    // Size: M * (N / 64)
    int blocks_per_row = N / 64;
    std::vector<float> local_max(M * blocks_per_row, 0.0f);
    std::vector<float> local_sum(M * blocks_per_row, 0.0f);

    // Run fused multiply
    kernel->multiply_fused(A.data(), C.data(), M, N, K,
                           nullptr, nullptr, true, // bias, mask, do_softmax
                           local_max.data(), local_sum.data(),
                           false, 1.0f, 0.0f, &ctx, -1);

    // Verify Softmax stats
    // For each block, calculate expected max and sum(exp(x-max))
    for (int m = 0; m < M; ++m)
    {
        for (int blk = 0; blk < blocks_per_row; ++blk)
        {
            float expected_max = -1e9f;

            // Calculate expected C for this block
            for (int n_local = 0; n_local < 64; ++n_local)
            {
                int n = blk * 64 + n_local;
                // Use actual C for Softmax verification to isolate Softmax logic from Quantization error
                float val = C[m * N + n];
                if (val > expected_max)
                    expected_max = val;
            }

            float actual_max = local_max[m * blocks_per_row + blk];
            // Max should be exact if we use C values
            EXPECT_EQ(actual_max, expected_max) << "Max mismatch at M=" << m << " Blk=" << blk;

            // Verify Sum
            float expected_sum = 0.0f;
            for (int n_local = 0; n_local < 64; ++n_local)
            {
                int n = blk * 64 + n_local;
                float val = C[m * N + n];
                expected_sum += std::exp(val - actual_max);
            }

            float actual_sum = local_sum[m * blocks_per_row + blk];
            // Sum uses fast exp approximation in JIT, so expect some error
            // 5% tolerance is generous for fast exp
            EXPECT_NEAR(actual_sum, expected_sum, expected_sum * 0.05f) << "Sum mismatch at M=" << m << " Blk=" << blk;
        }
    }
}

TEST(Test__QuantisedGemmFused, SoftmaxCorrectness_M2)
{
    RunSoftmaxTest(2);
}

TEST(Test__QuantisedGemmFused, SoftmaxCorrectness_M1)
{
    RunSoftmaxTest(1);
}
