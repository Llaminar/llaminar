/**
 * @file Test__CuBLASGemm.cpp
 * @brief Test cuBLAS GEMM kernel correctness
 *
 * **Purpose**: Validate CuBLASGemmKernel against CPU reference implementations.
 *
 * **Tests**:
 * - Small matrix correctness
 * - Various sizes (square, tall-skinny, short-wide)
 * - Transpose variants (NN, NT, TN, TT)
 * - Edge cases (single row for decode, large prefill)
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>

// Include project headers BEFORE CUDATestUtils.h (provides include paths)
#include "backends/ComputeBackend.h" // DeviceManager
#include "execution/local_execution/device/DeviceContext.h"
#ifdef HAVE_CUDA
#include "backends/cuda/CUDABackend.h"
#include "kernels/cuda/CuBLASGemmKernel.h"
#include <cuda_runtime.h>
#endif

// Now include test utils (uses headers above)
#include "../../../utils/CUDATestUtils.h"

using namespace llaminar2;
using namespace llaminar2::test::cuda;

// ============================================================================
// Test Fixture
// ============================================================================

class Test__CuBLASGemm : public CUDATestBase
{
protected:
#ifdef HAVE_CUDA
    void SetUp() override
    {
        CUDATestBase::SetUp();

        // Create cuBLAS kernel using detected GPU device
        // Note: CuBLASGemmKernel expects CUDA device index (0-based within CUDA devices)
        // gpu_idx_ is the DeviceManager index which includes CPU at index 0
        // For now, use 0 as the CUDA device since we only have one CUDA GPU
        if (gpu_idx_ >= 0)
        {
            kernel_ = std::make_unique<cuda::CuBLASGemmKernel>(0);
        }
    }

    void TearDown() override
    {
        kernel_.reset();
        CUDATestBase::TearDown();
    }

    std::unique_ptr<cuda::CuBLASGemmKernel> kernel_;
#endif
};

// ============================================================================
// Basic Correctness Tests
// ============================================================================

#ifdef HAVE_CUDA

TEST_F(Test__CuBLASGemm, SmallMatrix_NN)
{
    // Small test: C[32×64] = A[32×128] @ B[128×64]
    const int M = 32, N = 64, K = 128;

    // Generate test data
    auto A = generateRandomFP32(M * K, -1.0f, 1.0f, 100);
    auto B = generateRandomFP32(K * N, -1.0f, 1.0f, 200);
    std::vector<float> C_cuda(M * N, 0.0f);
    std::vector<float> C_cpu(M * N, 0.0f);

    // CPU reference (no transpose)
    cpuGemmNN(A.data(), B.data(), C_cpu.data(), M, N, K);

    // Allocate GPU memory
    float *d_A, *d_B, *d_C;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_A, M * K * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_B, K * N * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_C, M * N * sizeof(float)));

    // Upload
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemset(d_C, 0, M * N * sizeof(float)));

    // Execute cuBLAS GEMM (no transpose)
    ASSERT_TRUE(kernel_->execute(d_A, d_B, d_C, M, N, K,
                                 /*transA=*/false, /*transB=*/false));

    // Download result
    ASSERT_EQ(cudaSuccess, cudaMemcpy(C_cuda.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare
    auto result = compareArrays(C_cuda.data(), C_cpu.data(), M * N, GEMM_ABS_TOL, GEMM_REL_TOL);
    EXPECT_TRUE(result.passed) << "cuBLAS GEMM (NN) failed parity check";
    result.print();

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST_F(Test__CuBLASGemm, SmallMatrix_NT)
{
    // Test with B transposed (common for weights)
    // C[32×64] = A[32×128] @ B^T where B is stored as [64×128]
    const int M = 32, N = 64, K = 128;

    auto A = generateRandomFP32(M * K, -1.0f, 1.0f, 101);
    auto B = generateRandomFP32(N * K, -1.0f, 1.0f, 201); // B stored as [N×K]
    std::vector<float> C_cuda(M * N, 0.0f);
    std::vector<float> C_cpu(M * N, 0.0f);

    // CPU reference (B transposed - use cpuGemmNT)
    cpuGemmNT(A.data(), B.data(), C_cpu.data(), M, N, K);

    // GPU execution
    float *d_A, *d_B, *d_C;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_A, M * K * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_B, N * K * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_C, M * N * sizeof(float)));

    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_B, B.data(), N * K * sizeof(float), cudaMemcpyHostToDevice));

    // Execute with transB=true
    ASSERT_TRUE(kernel_->execute(d_A, d_B, d_C, M, N, K,
                                 /*transA=*/false, /*transB=*/true));

    ASSERT_EQ(cudaSuccess, cudaMemcpy(C_cuda.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    auto result = compareArrays(C_cuda.data(), C_cpu.data(), M * N, GEMM_ABS_TOL, GEMM_REL_TOL);
    EXPECT_TRUE(result.passed) << "cuBLAS GEMM (NT) failed parity check";
    result.print();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST_F(Test__CuBLASGemm, SquareMatrix_256)
{
    const int M = 256, N = 256, K = 256;

    auto A = generateRandomFP32(M * K);
    auto B = generateRandomFP32(K * N);
    std::vector<float> C_cuda(M * N, 0.0f);
    std::vector<float> C_cpu(M * N, 0.0f);

    cpuGemmNN(A.data(), B.data(), C_cpu.data(), M, N, K);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    ASSERT_TRUE(kernel_->execute(d_A, d_B, d_C, M, N, K, false, false));

    cudaMemcpy(C_cuda.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    auto result = compareArrays(C_cuda.data(), C_cpu.data(), M * N, GEMM_ABS_TOL, GEMM_REL_TOL);
    EXPECT_TRUE(result.passed) << "256×256 GEMM failed";
    result.print();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============================================================================
// LLM-Relevant Size Tests
// ============================================================================

TEST_F(Test__CuBLASGemm, DecodeSize_SingleToken)
{
    // Decode: 1 token through FFN
    // C[1×3584] = A[1×896] @ B[896×3584]  (Qwen2.5-0.5B FFN up)
    const int M = 1, N = 3584, K = 896;

    auto A = generateRandomFP32(M * K);
    auto B = generateRandomFP32(K * N);
    std::vector<float> C_cuda(M * N, 0.0f);
    std::vector<float> C_cpu(M * N, 0.0f);

    cpuGemmNN(A.data(), B.data(), C_cpu.data(), M, N, K);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    ASSERT_TRUE(kernel_->execute(d_A, d_B, d_C, M, N, K, false, false));

    cudaMemcpy(C_cuda.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    auto result = compareArrays(C_cuda.data(), C_cpu.data(), M * N, GEMM_ABS_TOL, GEMM_REL_TOL);
    EXPECT_TRUE(result.passed) << "Single token decode GEMM failed";
    result.print();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST_F(Test__CuBLASGemm, PrefillSize_MediumBatch)
{
    // Prefill: 128 tokens through attention projection
    // C[128×896] = A[128×896] @ B[896×896]
    const int M = 128, N = 896, K = 896;

    auto A = generateRandomFP32(M * K);
    auto B = generateRandomFP32(K * N);
    std::vector<float> C_cuda(M * N, 0.0f);
    std::vector<float> C_cpu(M * N, 0.0f);

    cpuGemmNN(A.data(), B.data(), C_cpu.data(), M, N, K);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    ASSERT_TRUE(kernel_->execute(d_A, d_B, d_C, M, N, K, false, false));

    cudaMemcpy(C_cuda.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    auto result = compareArrays(C_cuda.data(), C_cpu.data(), M * N, GEMM_ABS_TOL, GEMM_REL_TOL);
    EXPECT_TRUE(result.passed) << "Prefill GEMM failed";
    result.print();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST_F(Test__CuBLASGemm, LMHeadSize)
{
    // LM head: project to vocab
    // C[1×151936] = A[1×896] @ B[896×151936]  (Qwen2.5 vocab size)
    // Using smaller vocab for speed
    const int M = 1, N = 32000, K = 896;

    auto A = generateRandomFP32(M * K);
    auto B = generateRandomFP32(K * N);
    std::vector<float> C_cuda(M * N, 0.0f);
    std::vector<float> C_cpu(M * N, 0.0f);

    cpuGemmNN(A.data(), B.data(), C_cpu.data(), M, N, K);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    ASSERT_TRUE(kernel_->execute(d_A, d_B, d_C, M, N, K, false, false));

    cudaMemcpy(C_cuda.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    auto result = compareArrays(C_cuda.data(), C_cpu.data(), M * N, GEMM_ABS_TOL, GEMM_REL_TOL);
    EXPECT_TRUE(result.passed) << "LM head GEMM failed";
    result.print();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============================================================================
// Alpha/Beta Tests
// ============================================================================

TEST_F(Test__CuBLASGemm, AlphaBetaScaling)
{
    const int M = 32, N = 32, K = 64;
    const float alpha = 0.5f;
    const float beta = 0.25f;

    auto A = generateRandomFP32(M * K);
    auto B = generateRandomFP32(K * N);
    auto C_init = generateRandomFP32(M * N, 0.0f, 1.0f, 300);
    std::vector<float> C_cuda = C_init;
    std::vector<float> C_cpu = C_init;

    // CPU reference with alpha/beta
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int p = 0; p < K; ++p)
            {
                sum += A[i * K + p] * B[p * N + j];
            }
            C_cpu[i * N + j] = alpha * sum + beta * C_cpu[i * N + j];
        }
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C_init.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

    ASSERT_TRUE(kernel_->execute(d_A, d_B, d_C, M, N, K, false, false, alpha, beta));

    cudaMemcpy(C_cuda.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    auto result = compareArrays(C_cuda.data(), C_cpu.data(), M * N, GEMM_ABS_TOL, GEMM_REL_TOL);
    EXPECT_TRUE(result.passed) << "Alpha/Beta scaling failed";
    result.print();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(Test__CuBLASGemm, TinyMatrix_4x4)
{
    const int M = 4, N = 4, K = 4;

    auto A = generateRandomFP32(M * K);
    auto B = generateRandomFP32(K * N);
    std::vector<float> C_cuda(M * N, 0.0f);
    std::vector<float> C_cpu(M * N, 0.0f);

    cpuGemmNN(A.data(), B.data(), C_cpu.data(), M, N, K);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    ASSERT_TRUE(kernel_->execute(d_A, d_B, d_C, M, N, K, false, false));

    cudaMemcpy(C_cuda.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    auto result = compareArrays(C_cuda.data(), C_cpu.data(), M * N, GEMM_ABS_TOL, GEMM_REL_TOL);
    EXPECT_TRUE(result.passed) << "4x4 GEMM failed";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST_F(Test__CuBLASGemm, NonSquareAspectRatios)
{
    // Test various aspect ratios

    struct TestCase
    {
        int M, N, K;
        const char *name;
    };

    std::vector<TestCase> cases = {
        {16, 512, 64, "Wide output"},
        {512, 16, 64, "Tall output"},
        {64, 64, 1024, "Deep K"},
        {1024, 64, 64, "Many rows"},
    };

    for (const auto &tc : cases)
    {
        auto A = generateRandomFP32(tc.M * tc.K);
        auto B = generateRandomFP32(tc.K * tc.N);
        std::vector<float> C_cuda(tc.M * tc.N, 0.0f);
        std::vector<float> C_cpu(tc.M * tc.N, 0.0f);

        cpuGemmNN(A.data(), B.data(), C_cpu.data(), tc.M, tc.N, tc.K);

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, tc.M * tc.K * sizeof(float));
        cudaMalloc(&d_B, tc.K * tc.N * sizeof(float));
        cudaMalloc(&d_C, tc.M * tc.N * sizeof(float));

        cudaMemcpy(d_A, A.data(), tc.M * tc.K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B.data(), tc.K * tc.N * sizeof(float), cudaMemcpyHostToDevice);

        ASSERT_TRUE(kernel_->execute(d_A, d_B, d_C, tc.M, tc.N, tc.K, false, false))
            << "Failed for case: " << tc.name;

        cudaMemcpy(C_cuda.data(), d_C, tc.M * tc.N * sizeof(float), cudaMemcpyDeviceToHost);

        auto result = compareArrays(C_cuda.data(), C_cpu.data(), tc.M * tc.N, GEMM_ABS_TOL, GEMM_REL_TOL);
        EXPECT_TRUE(result.passed) << "Failed for case: " << tc.name;

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
}

#endif // HAVE_CUDA

// ============================================================================
// Non-CUDA Tests (always run)
// ============================================================================

TEST(Test__CuBLASGemm_NoCUDA, SkipsGracefully)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA not available in this build";
#else
    // This test only runs if CUDA is available
    SUCCEED() << "CUDA is available";
#endif
}
