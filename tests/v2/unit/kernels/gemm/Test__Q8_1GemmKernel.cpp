#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "kernels/cpu/gemm_v4/Q8_1GemmKernel.h"
#include <vector>
#include <random>

using namespace llaminar2;
using namespace llaminar2::gemm_v4;

TEST(Test__Q8_1GemmKernel, BasicMatMul)
{
    // Dimensions
    int M = 1;
    int N = 64;
    int K = 64;

    // Create random weights (N x K)
    std::vector<float> weights_fp32(N * K);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : weights_fp32)
        x = dist(gen);

    // Quantize weights to Q8_1Tensor
    // Q8_1Tensor expects [N, K] shape
    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});

    // Create kernel
    auto kernel = weights_tensor->createGemm();
    ASSERT_NE(kernel, nullptr);

    // Create random input A (M x K)
    std::vector<float> A(M * K);
    for (auto &x : A)
        x = dist(gen);

    // Compute reference C (M x N)
    std::vector<float> C_ref(M * N, 0.0f);
    // A is M x K, Weights is N x K.
    // C = A * Weights^T
    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                sum += A[m * K + k] * weights_fp32[n * K + k];
            }
            C_ref[m * N + n] = sum;
        }
    }

    // Compute actual C
    std::vector<float> C_act(M * N, 0.0f);
    kernel->multiply(A.data(), C_act.data(), M, N, K);

    // Compare
    // Q8_1 quantization introduces error.
    // Tolerance depends on range.
    for (int i = 0; i < M * N; ++i)
    {
        EXPECT_NEAR(C_act[i], C_ref[i], 1.0f) << "Mismatch at index " << i;
    }
}
