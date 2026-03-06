/**
 * @file Test__GemmKernelContextIntegration.cpp
 * @brief Integration tests for GEMM kernels with device context support
 *
 * Tests the Phase 4 GPU Device Context Refactor for CuBLASGemmKernel and HipBLASGemmKernel:
 * - Context-provided handle vs own handle
 * - Both paths produce same results
 * - Backward compatibility with legacy constructor
 *
 * These tests require actual GPU hardware.
 *
 * NOTE: This file cannot test both CUDA and ROCm in the same compilation unit
 * because CUDA and HIP vector types conflict. Build with either HAVE_CUDA or
 * HAVE_ROCM (not both) to enable the respective tests.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>

#include "backends/ComputeBackend.h"
#include "backends/DeviceId.h"

// NOTE: Due to CUDA/HIP header conflicts, we can only include one GPU backend
// per compilation unit. When both HAVE_CUDA and HAVE_ROCM are defined,
// we prefer CUDA tests and skip ROCm tests.

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
#warning "Both HAVE_CUDA and HAVE_ROCM are defined. Only CUDA tests will be built due to header conflicts."
#undef HAVE_ROCM
#endif

#ifdef HAVE_CUDA
#include "kernels/cuda/CuBLASGemmKernel.h"
#include "backends/GPUDeviceContextPool.h"
#include <cuda_runtime.h>
#endif

#ifdef HAVE_ROCM
#include "kernels/rocm/gemm/HipBLASGemmKernel.h"
#include "backends/GPUDeviceContextPool.h"
#include <hip/hip_runtime.h>
#endif

#include <cmath>
#include <random>
#include <vector>

using namespace llaminar2;

// ============================================================================
// Helper Functions
// ============================================================================

namespace
{
    /**
     * @brief Generate random FP32 data
     */
    std::vector<float> generateRandomFP32(size_t count, float min_val, float max_val, unsigned seed)
    {
        std::vector<float> data(count);
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(min_val, max_val);
        for (size_t i = 0; i < count; ++i)
        {
            data[i] = dist(gen);
        }
        return data;
    }

    /**
     * @brief Compute max absolute difference between two arrays
     */
    float maxAbsDiff(const std::vector<float> &a, const std::vector<float> &b)
    {
        if (a.size() != b.size())
            return std::numeric_limits<float>::max();
        float max_diff = 0.0f;
        for (size_t i = 0; i < a.size(); ++i)
        {
            max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
        }
        return max_diff;
    }

    /**
     * @brief CPU reference GEMM for verification
     * C = A @ B (row-major)
     */
    void cpuGemmNN(const float *A, const float *B, float *C, int M, int N, int K)
    {
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k)
                {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }
} // namespace

// ============================================================================
// CUDA Integration Tests
// ============================================================================

#ifdef HAVE_CUDA

class Test__CuBLASGemmContextIntegration : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Check if CUDA is available
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0)
        {
            GTEST_SKIP() << "No CUDA devices available";
        }

        // Use device 0
        device_id_ = 0;
        ASSERT_EQ(cudaSuccess, cudaSetDevice(device_id_));
    }

    void TearDown() override
    {
        // NOTE: Do NOT call cudaDeviceReset() here!
        // The GPUDeviceContextPool is a singleton that persists across tests.
        // Calling cudaDeviceReset() would invalidate the pool's CUDA context
        // and cause subsequent tests using the pool to fail with CUBLAS_STATUS_INTERNAL_ERROR.
        // The pool handles its own cleanup during process shutdown.
        cudaDeviceSynchronize();
    }

    int device_id_ = 0;
};

TEST_F(Test__CuBLASGemmContextIntegration, LegacyConstructor_OwnsHandle)
{
    auto kernel = std::make_unique<cuda::CuBLASGemmKernel>(device_id_);

    EXPECT_TRUE(kernel->ownsHandle());
    EXPECT_EQ(kernel->device_id(), device_id_);
    EXPECT_FALSE(kernel->hasDeviceContext());
}

TEST_F(Test__CuBLASGemmContextIntegration, LegacyConstructor_ProducesCorrectResults)
{
    const int M = 64, N = 128, K = 256;

    // Generate test data
    auto A = generateRandomFP32(M * K, -1.0f, 1.0f, 100);
    auto B = generateRandomFP32(K * N, -1.0f, 1.0f, 200);
    std::vector<float> C_gpu(M * N, 0.0f);
    std::vector<float> C_cpu(M * N, 0.0f);

    // CPU reference
    cpuGemmNN(A.data(), B.data(), C_cpu.data(), M, N, K);

    // Allocate GPU memory
    float *d_A, *d_B, *d_C;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_A, M * K * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_B, K * N * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_C, M * N * sizeof(float)));

    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemset(d_C, 0, M * N * sizeof(float)));

    // Create kernel with legacy constructor
    auto kernel = std::make_unique<cuda::CuBLASGemmKernel>(device_id_);
    ASSERT_TRUE(kernel->ownsHandle());

    // Execute GEMM
    ASSERT_TRUE(kernel->execute(d_A, d_B, d_C, M, N, K, false, false));

    // Copy back and verify
    ASSERT_EQ(cudaSuccess, cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    float max_diff = maxAbsDiff(C_cpu, C_gpu);
    EXPECT_LT(max_diff, 1e-2f) << "Max absolute difference: " << max_diff;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST_F(Test__CuBLASGemmContextIntegration, ContextConstructor_DoesNotOwnHandle)
{
    // Get device context from pool
    auto &pool = GPUDeviceContextPool::instance();
    IWorkerGPUContext *ctx = &pool.getContext("cuda", device_id_);
    if (!ctx || !ctx->isInitialized())
    {
        GTEST_SKIP() << "GPUDeviceContextPool not initialized for CUDA device " << device_id_;
    }

    auto kernel = std::make_unique<cuda::CuBLASGemmKernel>(ctx);

    EXPECT_FALSE(kernel->ownsHandle());
    EXPECT_EQ(kernel->device_id(), device_id_);
    EXPECT_TRUE(kernel->hasDeviceContext());
    EXPECT_EQ(kernel->deviceContext(), ctx);
}

TEST_F(Test__CuBLASGemmContextIntegration, ContextConstructor_ProducesCorrectResults)
{
    // Get device context from pool
    auto &pool = GPUDeviceContextPool::instance();
    IWorkerGPUContext *ctx = &pool.getContext("cuda", device_id_);
    if (!ctx || !ctx->isInitialized())
    {
        GTEST_SKIP() << "GPUDeviceContextPool not initialized for CUDA device " << device_id_;
    }

    const int M = 64, N = 128, K = 256;

    // Generate test data
    auto A = generateRandomFP32(M * K, -1.0f, 1.0f, 100);
    auto B = generateRandomFP32(K * N, -1.0f, 1.0f, 200);
    std::vector<float> C_gpu(M * N, 0.0f);
    std::vector<float> C_cpu(M * N, 0.0f);

    // CPU reference
    cpuGemmNN(A.data(), B.data(), C_cpu.data(), M, N, K);

    // Allocate GPU memory
    float *d_A, *d_B, *d_C;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_A, M * K * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_B, K * N * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_C, M * N * sizeof(float)));

    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemset(d_C, 0, M * N * sizeof(float)));

    // Create kernel with context constructor
    auto kernel = std::make_unique<cuda::CuBLASGemmKernel>(ctx);
    ASSERT_FALSE(kernel->ownsHandle());

    // Execute GEMM through submitAndWait (required for context-based kernels)
    // The cuBLAS handle from context is bound to the worker thread
    bool exec_result = false;
    ctx->submitAndWait([&]()
                       { exec_result = kernel->execute(d_A, d_B, d_C, M, N, K, false, false); });
    ASSERT_TRUE(exec_result);

    // Copy back and verify
    ASSERT_EQ(cudaSuccess, cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    float max_diff = maxAbsDiff(C_cpu, C_gpu);
    EXPECT_LT(max_diff, 1e-2f) << "Max absolute difference: " << max_diff;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST_F(Test__CuBLASGemmContextIntegration, BothConstructors_ProduceSameResults)
{
    // Get device context from pool
    auto &pool = GPUDeviceContextPool::instance();
    IWorkerGPUContext *ctx = &pool.getContext("cuda", device_id_);
    if (!ctx || !ctx->isInitialized())
    {
        GTEST_SKIP() << "GPUDeviceContextPool not initialized for CUDA device " << device_id_;
    }

    const int M = 32, N = 64, K = 128;

    // Generate test data
    auto A = generateRandomFP32(M * K, -1.0f, 1.0f, 300);
    auto B = generateRandomFP32(K * N, -1.0f, 1.0f, 400);
    std::vector<float> C_legacy(M * N, 0.0f);
    std::vector<float> C_context(M * N, 0.0f);

    // Allocate GPU memory
    float *d_A, *d_B, *d_C;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_A, M * K * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_B, K * N * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_C, M * N * sizeof(float)));

    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    // Execute with legacy constructor
    {
        auto kernel = std::make_unique<cuda::CuBLASGemmKernel>(device_id_);
        ASSERT_EQ(cudaSuccess, cudaMemset(d_C, 0, M * N * sizeof(float)));
        ASSERT_TRUE(kernel->execute(d_A, d_B, d_C, M, N, K, false, false));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(C_legacy.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // Execute with context constructor - must use submitAndWait for context-bound handle
    {
        auto kernel = std::make_unique<cuda::CuBLASGemmKernel>(ctx);
        ASSERT_EQ(cudaSuccess, cudaMemset(d_C, 0, M * N * sizeof(float)));
        bool exec_result = false;
        ctx->submitAndWait([&]()
                           { exec_result = kernel->execute(d_A, d_B, d_C, M, N, K, false, false); });
        ASSERT_TRUE(exec_result);
        ASSERT_EQ(cudaSuccess, cudaMemcpy(C_context.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // Results should be identical (same cuBLAS implementation)
    float max_diff = maxAbsDiff(C_legacy, C_context);
    EXPECT_EQ(max_diff, 0.0f) << "Legacy and context kernels should produce identical results";

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST_F(Test__CuBLASGemmContextIntegration, ContextConstructor_ThrowsOnNull)
{
    EXPECT_THROW(
        {
            auto kernel = std::make_unique<cuda::CuBLASGemmKernel>(nullptr);
        },
        std::runtime_error);
}

#endif // HAVE_CUDA

// ============================================================================
// ROCm Integration Tests
// ============================================================================

#ifdef HAVE_ROCM

class Test__HipBLASGemmContextIntegration : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Check if ROCm is available
        int device_count = 0;
        hipError_t err = hipGetDeviceCount(&device_count);
        if (err != hipSuccess || device_count == 0)
        {
            GTEST_SKIP() << "No ROCm devices available";
        }

        // Use device 0
        device_id_ = DeviceId::rocm(0);
        ASSERT_EQ(hipSuccess, hipSetDevice(device_id_.ordinal));
    }

    void TearDown() override
    {
        // NOTE: Do NOT call hipDeviceReset() here!
        // The GPUDeviceContextPool is a singleton that persists across tests.
        // Calling hipDeviceReset() would invalidate the pool's HIP context
        // and cause subsequent tests using the pool to fail.
        // The pool handles its own cleanup during process shutdown.
        hipDeviceSynchronize();
    }

    DeviceId device_id_;
};

TEST_F(Test__HipBLASGemmContextIntegration, LegacyConstructor_OwnsHandle)
{
    auto kernel = std::make_unique<rocm::HipBLASGemmKernel>(device_id_);

    EXPECT_TRUE(kernel->ownsHandle());
    EXPECT_EQ(kernel->device_ordinal(), device_id_.ordinal);
    EXPECT_FALSE(kernel->hasDeviceContext());
}

TEST_F(Test__HipBLASGemmContextIntegration, LegacyConstructor_ProducesCorrectResults)
{
    const int M = 64, N = 128, K = 256;

    // Generate test data
    auto A = generateRandomFP32(M * K, -1.0f, 1.0f, 100);
    auto B = generateRandomFP32(K * N, -1.0f, 1.0f, 200);
    std::vector<float> C_gpu(M * N, 0.0f);
    std::vector<float> C_cpu(M * N, 0.0f);

    // CPU reference
    cpuGemmNN(A.data(), B.data(), C_cpu.data(), M, N, K);

    // Allocate GPU memory
    float *d_A, *d_B, *d_C;
    ASSERT_EQ(hipSuccess, hipMalloc(&d_A, M * K * sizeof(float)));
    ASSERT_EQ(hipSuccess, hipMalloc(&d_B, K * N * sizeof(float)));
    ASSERT_EQ(hipSuccess, hipMalloc(&d_C, M * N * sizeof(float)));

    ASSERT_EQ(hipSuccess, hipMemcpy(d_A, A.data(), M * K * sizeof(float), hipMemcpyHostToDevice));
    ASSERT_EQ(hipSuccess, hipMemcpy(d_B, B.data(), K * N * sizeof(float), hipMemcpyHostToDevice));
    ASSERT_EQ(hipSuccess, hipMemset(d_C, 0, M * N * sizeof(float)));

    // Create kernel with legacy constructor
    auto kernel = std::make_unique<rocm::HipBLASGemmKernel>(device_id_);
    ASSERT_TRUE(kernel->ownsHandle());

    // Execute GEMM
    ASSERT_TRUE(kernel->execute(d_A, d_B, d_C, M, N, K, false, false));

    // Copy back and verify
    ASSERT_EQ(hipSuccess, hipMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), hipMemcpyDeviceToHost));

    float max_diff = maxAbsDiff(C_cpu, C_gpu);
    EXPECT_LT(max_diff, 1e-2f) << "Max absolute difference: " << max_diff;

    // Cleanup
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
}

TEST_F(Test__HipBLASGemmContextIntegration, ContextConstructor_DoesNotOwnHandle)
{
    // Get device context from pool
    auto &pool = GPUDeviceContextPool::instance();
    IWorkerGPUContext *ctx = &pool.getContext("rocm", device_id_.ordinal);
    if (!ctx || !ctx->isInitialized())
    {
        GTEST_SKIP() << "GPUDeviceContextPool not initialized for ROCm device " << device_id_.ordinal;
    }

    auto kernel = std::make_unique<rocm::HipBLASGemmKernel>(ctx);

    EXPECT_FALSE(kernel->ownsHandle());
    EXPECT_EQ(kernel->device_ordinal(), device_id_.ordinal);
    EXPECT_TRUE(kernel->hasDeviceContext());
    EXPECT_EQ(kernel->deviceContext(), ctx);
}

TEST_F(Test__HipBLASGemmContextIntegration, ContextConstructor_ProducesCorrectResults)
{
    // Get device context from pool
    auto &pool = GPUDeviceContextPool::instance();
    IWorkerGPUContext *ctx = &pool.getContext("rocm", device_id_.ordinal);
    if (!ctx || !ctx->isInitialized())
    {
        GTEST_SKIP() << "GPUDeviceContextPool not initialized for ROCm device " << device_id_.ordinal;
    }

    const int M = 64, N = 128, K = 256;

    // Generate test data
    auto A = generateRandomFP32(M * K, -1.0f, 1.0f, 100);
    auto B = generateRandomFP32(K * N, -1.0f, 1.0f, 200);
    std::vector<float> C_gpu(M * N, 0.0f);
    std::vector<float> C_cpu(M * N, 0.0f);

    // CPU reference
    cpuGemmNN(A.data(), B.data(), C_cpu.data(), M, N, K);

    // Allocate GPU memory
    float *d_A, *d_B, *d_C;
    ASSERT_EQ(hipSuccess, hipMalloc(&d_A, M * K * sizeof(float)));
    ASSERT_EQ(hipSuccess, hipMalloc(&d_B, K * N * sizeof(float)));
    ASSERT_EQ(hipSuccess, hipMalloc(&d_C, M * N * sizeof(float)));

    ASSERT_EQ(hipSuccess, hipMemcpy(d_A, A.data(), M * K * sizeof(float), hipMemcpyHostToDevice));
    ASSERT_EQ(hipSuccess, hipMemcpy(d_B, B.data(), K * N * sizeof(float), hipMemcpyHostToDevice));
    ASSERT_EQ(hipSuccess, hipMemset(d_C, 0, M * N * sizeof(float)));

    // Create kernel with context constructor
    auto kernel = std::make_unique<rocm::HipBLASGemmKernel>(ctx);
    ASSERT_FALSE(kernel->ownsHandle());

    // Execute GEMM through submitAndWait (required for context-based kernels)
    // The hipBLAS handle from context is bound to the worker thread
    bool exec_result = false;
    ctx->submitAndWait([&]()
                       { exec_result = kernel->execute(d_A, d_B, d_C, M, N, K, false, false); });
    ASSERT_TRUE(exec_result);

    // Copy back and verify
    ASSERT_EQ(hipSuccess, hipMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), hipMemcpyDeviceToHost));

    float max_diff = maxAbsDiff(C_cpu, C_gpu);
    EXPECT_LT(max_diff, 1e-2f) << "Max absolute difference: " << max_diff;

    // Cleanup
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
}

TEST_F(Test__HipBLASGemmContextIntegration, BothConstructors_ProduceSameResults)
{
    // Get device context from pool
    auto &pool = GPUDeviceContextPool::instance();
    IWorkerGPUContext *ctx = &pool.getContext("rocm", device_id_.ordinal);
    if (!ctx || !ctx->isInitialized())
    {
        GTEST_SKIP() << "GPUDeviceContextPool not initialized for ROCm device " << device_id_.ordinal;
    }

    const int M = 32, N = 64, K = 128;

    // Generate test data
    auto A = generateRandomFP32(M * K, -1.0f, 1.0f, 300);
    auto B = generateRandomFP32(K * N, -1.0f, 1.0f, 400);
    std::vector<float> C_legacy(M * N, 0.0f);
    std::vector<float> C_context(M * N, 0.0f);

    // Allocate GPU memory
    float *d_A, *d_B, *d_C;
    ASSERT_EQ(hipSuccess, hipMalloc(&d_A, M * K * sizeof(float)));
    ASSERT_EQ(hipSuccess, hipMalloc(&d_B, K * N * sizeof(float)));
    ASSERT_EQ(hipSuccess, hipMalloc(&d_C, M * N * sizeof(float)));

    ASSERT_EQ(hipSuccess, hipMemcpy(d_A, A.data(), M * K * sizeof(float), hipMemcpyHostToDevice));
    ASSERT_EQ(hipSuccess, hipMemcpy(d_B, B.data(), K * N * sizeof(float), hipMemcpyHostToDevice));

    // Execute with legacy constructor
    {
        auto kernel = std::make_unique<rocm::HipBLASGemmKernel>(device_id_);
        ASSERT_EQ(hipSuccess, hipMemset(d_C, 0, M * N * sizeof(float)));
        ASSERT_TRUE(kernel->execute(d_A, d_B, d_C, M, N, K, false, false));
        ASSERT_EQ(hipSuccess, hipMemcpy(C_legacy.data(), d_C, M * N * sizeof(float), hipMemcpyDeviceToHost));
    }

    // Execute with context constructor - must use submitAndWait for context-bound handle
    {
        auto kernel = std::make_unique<rocm::HipBLASGemmKernel>(ctx);
        ASSERT_EQ(hipSuccess, hipMemset(d_C, 0, M * N * sizeof(float)));
        bool exec_result = false;
        ctx->submitAndWait([&]()
                           { exec_result = kernel->execute(d_A, d_B, d_C, M, N, K, false, false); });
        ASSERT_TRUE(exec_result);
        ASSERT_EQ(hipSuccess, hipMemcpy(C_context.data(), d_C, M * N * sizeof(float), hipMemcpyDeviceToHost));
    }

    // Results should be identical (same hipBLAS implementation)
    float max_diff = maxAbsDiff(C_legacy, C_context);
    EXPECT_EQ(max_diff, 0.0f) << "Legacy and context kernels should produce identical results";

    // Cleanup
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
}

TEST_F(Test__HipBLASGemmContextIntegration, ContextConstructor_ThrowsOnNull)
{
    EXPECT_THROW(
        {
            auto kernel = std::make_unique<rocm::HipBLASGemmKernel>(nullptr);
        },
        std::runtime_error);
}

#endif // HAVE_ROCM
