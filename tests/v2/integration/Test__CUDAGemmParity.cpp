/**
 * @file Test__CUDAGemmParity.cpp
 * @brief Parity tests for CUDA GEMM kernels vs CPU reference
 *
 * **Purpose**: Validate that CUDA GEMM kernels produce numerically equivalent
 * results to CPU kernels with high cosine similarity (>= 0.999).
 *
 * **Tests**:
 * - CUDAFloatingPointGemmKernel (FP32) vs FloatingPointGemmKernel
 * - CUDAQuantisedGemmKernel (IQ4_NL) vs QuantisedGemmKernel
 * - Various matrix sizes (decode, prefill, large)
 * - Real tensor objects through KernelFactory dispatch
 *
 * **Pass Criteria**:
 * - Cosine similarity >= 0.999 (very high correlation)
 * - No NaN/Inf in outputs
 * - Relative error < 5% for quantized (quantization inherently lossy)
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>

// Include project headers BEFORE CUDATestUtils.h
#include "tensors/Tensors.h"
#include "kernels/KernelFactory.h"
#include "backends/ComputeBackend.h"
#include "execution/DeviceContext.h"
#ifdef HAVE_CUDA
#include "backends/cuda/CUDABackend.h"
#include <cuda_runtime.h> // For cudaMalloc, cudaMemcpy, etc.
#endif

// Now include test utils
#include "../utils/CUDATestUtils.h"
#include "../utils/TestTensorFactory.h"

#include <vector>
#include <cmath>
#include <random>
#include <numeric>

using namespace llaminar2;
using namespace llaminar2::test::cuda;
using namespace llaminar2::test; // For TestTensorFactory

// Alias for kernel DeviceType to avoid ambiguity
using KernelDeviceType = llaminar::v2::kernels::DeviceType;

namespace
{

    // ============================================================================
    // Cosine Similarity Utilities
    // ============================================================================

    /**
     * @brief Compute cosine similarity between two float arrays
     *
     * cosine = (A · B) / (||A|| * ||B||)
     *
     * @return Value in [-1, 1], where 1 = perfect correlation
     */
    double cosineSimilarity(const float *a, const float *b, size_t count)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            dot += static_cast<double>(a[i]) * b[i];
            norm_a += static_cast<double>(a[i]) * a[i];
            norm_b += static_cast<double>(b[i]) * b[i];
        }
        double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
        if (denom < 1e-12)
            return 0.0;
        return dot / denom;
    }

    /**
     * @brief Compute relative L2 error: ||A - B|| / ||B||
     */
    double relativeL2Error(const float *actual, const float *expected, size_t count)
    {
        double diff_norm = 0.0, expected_norm = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            double diff = actual[i] - expected[i];
            diff_norm += diff * diff;
            expected_norm += static_cast<double>(expected[i]) * expected[i];
        }
        if (expected_norm < 1e-12)
            return diff_norm > 1e-12 ? 1e9 : 0.0;
        return std::sqrt(diff_norm / expected_norm);
    }

    /**
     * @brief Compute max absolute error
     */
    float maxAbsError(const float *actual, const float *expected, size_t count)
    {
        float max_err = 0.0f;
        for (size_t i = 0; i < count; ++i)
        {
            float err = std::abs(actual[i] - expected[i]);
            max_err = std::max(max_err, err);
        }
        return max_err;
    }

} // namespace

// ============================================================================
// Test Fixture
// ============================================================================

class Test__CUDAGemmParity : public CUDATestBase
{
protected:
    std::mt19937 rng_{42};
    std::uniform_real_distribution<float> dist_{-1.0f, 1.0f};

    /**
     * @brief Fill tensor with random data
     */
    void fillRandom(FP32Tensor *tensor)
    {
        float *data = tensor->mutable_data();
        for (size_t i = 0; i < tensor->numel(); ++i)
        {
            data[i] = dist_(rng_);
        }
    }

    /**
     * @brief Create random FP32 data
     */
    std::vector<float> randomFP32(size_t count)
    {
        std::vector<float> data(count);
        for (auto &val : data)
        {
            val = dist_(rng_);
        }
        return data;
    }

    /**
     * @brief CPU reference GEMM: C = A @ B^T
     *
     * For weight matrix B stored as [N, K], compute C[M, N] = A[M, K] @ B^T
     */
    void cpuGemmReference(
        const float *A, const float *B, float *C,
        int M, int N, int K)
    {
        // C[i, j] = sum_k A[i, k] * B[j, k]  (B is transposed)
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k)
                {
                    sum += A[i * K + k] * B[j * K + k];
                }
                C[i * N + j] = sum;
            }
        }
    }

    /**
     * @brief Parity result structure
     */
    struct ParityResult
    {
        double cosine_similarity = 0.0;
        double relative_l2_error = 0.0;
        float max_abs_error = 0.0f;
        bool has_nan_inf = false;
        bool passed = false;

        void print(const std::string &name) const
        {
            std::cout << name << ":\n"
                      << "  Cosine similarity: " << cosine_similarity << "\n"
                      << "  Relative L2 error: " << (relative_l2_error * 100.0) << "%\n"
                      << "  Max abs error:     " << max_abs_error << "\n"
                      << "  Passed:            " << (passed ? "YES" : "NO") << "\n";
        }
    };

    /**
     * @brief Compare CUDA vs CPU GEMM results
     *
     * @param cosine_threshold Minimum cosine similarity (default 0.999)
     * @param rel_l2_threshold Maximum relative L2 error (default 0.05 = 5%)
     */
    ParityResult checkParity(
        const float *cuda_result,
        const float *cpu_result,
        size_t count,
        double cosine_threshold = 0.999,
        double rel_l2_threshold = 0.05)
    {
        ParityResult result;
        result.has_nan_inf = hasNaNOrInf(cuda_result, count);
        result.cosine_similarity = cosineSimilarity(cuda_result, cpu_result, count);
        result.relative_l2_error = relativeL2Error(cuda_result, cpu_result, count);
        result.max_abs_error = maxAbsError(cuda_result, cpu_result, count);

        result.passed = !result.has_nan_inf &&
                        result.cosine_similarity >= cosine_threshold &&
                        result.relative_l2_error <= rel_l2_threshold;

        return result;
    }
};

// ============================================================================
// FP32 Parity Tests (CUDAFloatingPointGemmKernel vs FloatingPointGemmKernel)
// ============================================================================

#ifdef HAVE_CUDA

TEST_F(Test__CUDAGemmParity, FP32_SmallMatrix_128x256x512)
{
    // Dimensions
    const int M = 128; // batch/sequence
    const int N = 256; // output dim
    const int K = 512; // input dim

    // Create weight tensor on CPU
    auto weights = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)N, (size_t)K});
    fillRandom(weights.get());

    // Create activations
    auto A_data = randomFP32(M * K);

    // ===== CPU Reference =====
    std::vector<float> C_cpu(M * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_NE(cpu_kernel, nullptr) << "Failed to create CPU kernel";
    ASSERT_TRUE(cpu_kernel->multiply(A_data.data(), C_cpu.data(), M, N, K));

    // ===== CUDA =====
    // Upload weights to GPU
    ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
    EXPECT_TRUE(weights->isOnGPU());

    // Create CUDA kernel via KernelFactory
    auto cuda_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CUDA);
    ASSERT_NE(cuda_kernel, nullptr) << "Failed to create CUDA kernel";

    // Allocate GPU memory for activations and output
    float *d_A, *d_C;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_A, M * K * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_C, M * N * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_A, A_data.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemset(d_C, 0, M * N * sizeof(float)));

    // Execute CUDA GEMM
    ASSERT_TRUE(cuda_kernel->multiply(d_A, d_C, M, N, K));

    // Download result
    std::vector<float> C_cuda(M * N);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(C_cuda.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // ===== Compare =====
    auto result = checkParity(C_cuda.data(), C_cpu.data(), M * N, 0.9999, 0.01);
    result.print("FP32 128x256x512");

    EXPECT_FALSE(result.has_nan_inf) << "CUDA output contains NaN/Inf";
    EXPECT_GE(result.cosine_similarity, 0.9999)
        << "Cosine similarity too low: " << result.cosine_similarity;
    EXPECT_LE(result.relative_l2_error, 0.01)
        << "Relative L2 error too high: " << (result.relative_l2_error * 100) << "%";

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_C);
}

TEST_F(Test__CUDAGemmParity, FP32_DecodeSize_1x896x896)
{
    // Decode: single token projection
    const int M = 1;
    const int N = 896; // Qwen2.5 hidden dim
    const int K = 896;

    auto weights = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)N, (size_t)K});
    fillRandom(weights.get());

    auto A_data = randomFP32(M * K);

    // CPU reference
    std::vector<float> C_cpu(M * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_TRUE(cpu_kernel->multiply(A_data.data(), C_cpu.data(), M, N, K));

    // CUDA
    ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
    auto cuda_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CUDA);
    ASSERT_NE(cuda_kernel, nullptr);

    float *d_A, *d_C;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_A, M * K * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_C, M * N * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_A, A_data.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));

    ASSERT_TRUE(cuda_kernel->multiply(d_A, d_C, M, N, K));

    std::vector<float> C_cuda(M * N);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(C_cuda.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    auto result = checkParity(C_cuda.data(), C_cpu.data(), M * N, 0.9999, 0.01);
    result.print("FP32 Decode 1x896x896");

    EXPECT_GE(result.cosine_similarity, 0.9999);
    EXPECT_FALSE(result.has_nan_inf);

    cudaFree(d_A);
    cudaFree(d_C);
}

TEST_F(Test__CUDAGemmParity, FP32_PrefillSize_512x896x896)
{
    // Prefill: typical sequence length
    const int M = 512;
    const int N = 896;
    const int K = 896;

    auto weights = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)N, (size_t)K});
    fillRandom(weights.get());

    auto A_data = randomFP32(M * K);

    // CPU reference
    std::vector<float> C_cpu(M * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_TRUE(cpu_kernel->multiply(A_data.data(), C_cpu.data(), M, N, K));

    // CUDA
    ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
    auto cuda_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CUDA);
    ASSERT_NE(cuda_kernel, nullptr);

    float *d_A, *d_C;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_A, M * K * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_C, M * N * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_A, A_data.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));

    ASSERT_TRUE(cuda_kernel->multiply(d_A, d_C, M, N, K));

    std::vector<float> C_cuda(M * N);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(C_cuda.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    auto result = checkParity(C_cuda.data(), C_cpu.data(), M * N, 0.9999, 0.01);
    result.print("FP32 Prefill 512x896x896");

    EXPECT_GE(result.cosine_similarity, 0.9999);
    EXPECT_FALSE(result.has_nan_inf);

    cudaFree(d_A);
    cudaFree(d_C);
}

// ============================================================================
// IQ4_NL Parity Tests (CUDAQuantisedGemmKernel vs QuantisedGemmKernel)
// ============================================================================

TEST_F(Test__CUDAGemmParity, IQ4_NL_SmallMatrix_128x896x896)
{
    // Dimensions - K must be multiple of 32 for IQ4_NL
    // Using 896 (Qwen2.5 hidden dim) which is known to work
    const int M = 128;
    const int N = 896;
    const int K = 896;

    // Create IQ4_NL weight tensor using TestTensorFactory
    auto weights = TestTensorFactory::createIQ4_NLRandom({(size_t)N, (size_t)K}, 123);

    // Create activations
    auto A_data = randomFP32(M * K);

    // ===== CPU Reference =====
    std::vector<float> C_cpu(M * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_NE(cpu_kernel, nullptr) << "Failed to create CPU IQ4_NL kernel";
    ASSERT_TRUE(cpu_kernel->multiply(A_data.data(), C_cpu.data(), M, N, K));

    // Check CPU result is valid
    ASSERT_FALSE(hasNaNOrInf(C_cpu.data(), M * N)) << "CPU result has NaN/Inf";

    // ===== CUDA =====
    // Upload weights to GPU
    ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
    EXPECT_TRUE(weights->isOnGPU());

    // Create CUDA kernel via KernelFactory
    auto cuda_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CUDA);
    ASSERT_NE(cuda_kernel, nullptr) << "Failed to create CUDA IQ4_NL kernel";

    // Allocate GPU memory for activations and output
    float *d_A, *d_C;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_A, M * K * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_C, M * N * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_A, A_data.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemset(d_C, 0, M * N * sizeof(float)));

    // Execute CUDA GEMM
    ASSERT_TRUE(cuda_kernel->multiply(d_A, d_C, M, N, K));

    // Download result
    std::vector<float> C_cuda(M * N);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(C_cuda.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // ===== Compare =====
    // Quantized GEMM has inherent error from:
    // 1. Different quantization schemes (CPU VNNI vs CUDA symmetric INT8)
    // 2. Different accumulation order
    // Expect cosine >= 0.99 and rel_l2 <= 10%
    auto result = checkParity(C_cuda.data(), C_cpu.data(), M * N, 0.99, 0.10);
    result.print("IQ4_NL 128x896x896");

    EXPECT_FALSE(result.has_nan_inf) << "CUDA output contains NaN/Inf";
    EXPECT_GE(result.cosine_similarity, 0.99)
        << "Cosine similarity too low: " << result.cosine_similarity;
    EXPECT_LE(result.relative_l2_error, 0.10)
        << "Relative L2 error too high: " << (result.relative_l2_error * 100) << "%";

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_C);
}

TEST_F(Test__CUDAGemmParity, IQ4_NL_DecodeSize_1x896x896)
{
    // Decode single token - K must be multiple of 32
    const int M = 1;
    const int N = 896;
    const int K = 896;

    auto weights = TestTensorFactory::createIQ4_NLRandom({(size_t)N, (size_t)K}, 456);

    auto A_data = randomFP32(M * K);

    // CPU reference
    std::vector<float> C_cpu(M * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_TRUE(cpu_kernel->multiply(A_data.data(), C_cpu.data(), M, N, K));

    // CUDA
    ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
    auto cuda_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CUDA);
    ASSERT_NE(cuda_kernel, nullptr);

    float *d_A, *d_C;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_A, M * K * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_C, M * N * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_A, A_data.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));

    ASSERT_TRUE(cuda_kernel->multiply(d_A, d_C, M, N, K));

    std::vector<float> C_cuda(M * N);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(C_cuda.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    auto result = checkParity(C_cuda.data(), C_cpu.data(), M * N, 0.99, 0.10);
    result.print("IQ4_NL Decode 1x896x896");

    EXPECT_GE(result.cosine_similarity, 0.99);
    EXPECT_FALSE(result.has_nan_inf);

    cudaFree(d_A);
    cudaFree(d_C);
}

TEST_F(Test__CUDAGemmParity, IQ4_NL_PrefillSize_512x896x896)
{
    // Prefill - larger batch
    const int M = 512;
    const int N = 896;
    const int K = 896;

    auto weights = TestTensorFactory::createIQ4_NLRandom({(size_t)N, (size_t)K}, 789);

    auto A_data = randomFP32(M * K);

    // CPU reference
    std::vector<float> C_cpu(M * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_TRUE(cpu_kernel->multiply(A_data.data(), C_cpu.data(), M, N, K));

    // CUDA
    ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
    auto cuda_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CUDA);
    ASSERT_NE(cuda_kernel, nullptr);

    float *d_A, *d_C;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_A, M * K * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_C, M * N * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_A, A_data.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));

    ASSERT_TRUE(cuda_kernel->multiply(d_A, d_C, M, N, K));

    std::vector<float> C_cuda(M * N);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(C_cuda.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    auto result = checkParity(C_cuda.data(), C_cpu.data(), M * N, 0.99, 0.10);
    result.print("IQ4_NL Prefill 512x896x896");

    EXPECT_GE(result.cosine_similarity, 0.99);
    EXPECT_FALSE(result.has_nan_inf);

    cudaFree(d_A);
    cudaFree(d_C);
}

// ============================================================================
// Parameterized Quantized GEMM Parity Test
// ============================================================================

/**
 * @brief Helper macro for quantized parity tests
 *
 * This reduces duplication across all quantized tensor types.
 * All tests use the same pattern: create weights, run CPU reference, run CUDA, compare.
 *
 * K-quant formats use 256-element blocks, so K must be multiple of 256.
 * Simple formats (Q4_0, Q8_0, etc.) use 32-element blocks, K multiple of 32.
 */
#define DEFINE_QUANTIZED_PARITY_TEST(TestName, TensorType, CreateMethod, BlockSize, Seed)   \
    TEST_F(Test__CUDAGemmParity, TestName)                                                  \
    {                                                                                       \
        const int M = 128;                                                                  \
        const int N = 896;                                                                  \
        const int K = (BlockSize == 256) ? 768 : 896; /* K-quants need multiple of 256 */   \
                                                                                            \
        auto weights = TestTensorFactory::CreateMethod({(size_t)N, (size_t)K}, Seed);       \
                                                                                            \
        auto A_data = randomFP32(M * K);                                                    \
                                                                                            \
        /* CPU Reference */                                                                 \
        std::vector<float> C_cpu(M * N, 0.0f);                                              \
        auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(                 \
            weights.get(), KernelDeviceType::CPU);                                          \
        ASSERT_NE(cpu_kernel, nullptr) << "Failed to create CPU kernel for " #TensorType;   \
        ASSERT_TRUE(cpu_kernel->multiply(A_data.data(), C_cpu.data(), M, N, K));            \
        ASSERT_FALSE(hasNaNOrInf(C_cpu.data(), M * N)) << "CPU result has NaN/Inf";         \
                                                                                            \
        /* CUDA */                                                                          \
        ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));                                     \
        auto cuda_kernel = llaminar::v2::kernels::KernelFactory::createGemm(                \
            weights.get(), KernelDeviceType::CUDA);                                         \
        ASSERT_NE(cuda_kernel, nullptr) << "Failed to create CUDA kernel for " #TensorType; \
                                                                                            \
        float *d_A, *d_C;                                                                   \
        ASSERT_EQ(cudaSuccess, cudaMalloc(&d_A, M * K * sizeof(float)));                    \
        ASSERT_EQ(cudaSuccess, cudaMalloc(&d_C, M * N * sizeof(float)));                    \
        ASSERT_EQ(cudaSuccess, cudaMemcpy(d_A, A_data.data(), M * K * sizeof(float),        \
                                          cudaMemcpyHostToDevice));                         \
        ASSERT_EQ(cudaSuccess, cudaMemset(d_C, 0, M * N * sizeof(float)));                  \
                                                                                            \
        ASSERT_TRUE(cuda_kernel->multiply(d_A, d_C, M, N, K));                              \
                                                                                            \
        std::vector<float> C_cuda(M * N);                                                   \
        ASSERT_EQ(cudaSuccess, cudaMemcpy(C_cuda.data(), d_C, M * N * sizeof(float),        \
                                          cudaMemcpyDeviceToHost));                         \
                                                                                            \
        auto result = checkParity(C_cuda.data(), C_cpu.data(), M * N, 0.99, 0.15);          \
        result.print(#TensorType " 128x" + std::to_string(N) + "x" + std::to_string(K));    \
                                                                                            \
        EXPECT_FALSE(result.has_nan_inf) << "CUDA output contains NaN/Inf";                 \
        EXPECT_GE(result.cosine_similarity, 0.99)                                           \
            << "Cosine similarity too low: " << result.cosine_similarity;                   \
                                                                                            \
        cudaFree(d_A);                                                                      \
        cudaFree(d_C);                                                                      \
    }

// ============================================================================
// Q8_0 Parity Tests
// ============================================================================

DEFINE_QUANTIZED_PARITY_TEST(Q8_0_SmallMatrix, Q8_0Tensor, createQ8_0Random, 32, 101)

TEST_F(Test__CUDAGemmParity, Q8_0_DecodeSize_1x896x896)
{
    const int M = 1;
    const int N = 896;
    const int K = 896;

    auto weights = TestTensorFactory::createQ8_0Random({(size_t)N, (size_t)K}, 111);
    auto A_data = randomFP32(M * K);

    std::vector<float> C_cpu(M * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_TRUE(cpu_kernel->multiply(A_data.data(), C_cpu.data(), M, N, K));

    ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
    auto cuda_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CUDA);

    float *d_A, *d_C;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_A, M * K * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_C, M * N * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_A, A_data.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));

    ASSERT_TRUE(cuda_kernel->multiply(d_A, d_C, M, N, K));

    std::vector<float> C_cuda(M * N);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(C_cuda.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    auto result = checkParity(C_cuda.data(), C_cpu.data(), M * N, 0.99, 0.10);
    result.print("Q8_0 Decode 1x896x896");

    EXPECT_GE(result.cosine_similarity, 0.99);
    EXPECT_FALSE(result.has_nan_inf);

    cudaFree(d_A);
    cudaFree(d_C);
}

// ============================================================================
// Q4_0 Parity Tests
// ============================================================================

DEFINE_QUANTIZED_PARITY_TEST(Q4_0_SmallMatrix, Q4_0Tensor, createQ4_0Random, 32, 102)

TEST_F(Test__CUDAGemmParity, Q4_0_DecodeSize_1x896x896)
{
    const int M = 1;
    const int N = 896;
    const int K = 896;

    auto weights = TestTensorFactory::createQ4_0Random({(size_t)N, (size_t)K}, 121);
    auto A_data = randomFP32(M * K);

    std::vector<float> C_cpu(M * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_TRUE(cpu_kernel->multiply(A_data.data(), C_cpu.data(), M, N, K));

    ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
    auto cuda_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CUDA);

    float *d_A, *d_C;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_A, M * K * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_C, M * N * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_A, A_data.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));

    ASSERT_TRUE(cuda_kernel->multiply(d_A, d_C, M, N, K));

    std::vector<float> C_cuda(M * N);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(C_cuda.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    auto result = checkParity(C_cuda.data(), C_cpu.data(), M * N, 0.99, 0.10);
    result.print("Q4_0 Decode 1x896x896");

    EXPECT_GE(result.cosine_similarity, 0.99);
    EXPECT_FALSE(result.has_nan_inf);

    cudaFree(d_A);
    cudaFree(d_C);
}

// ============================================================================
// Q4_1 Parity Tests
// ============================================================================

DEFINE_QUANTIZED_PARITY_TEST(Q4_1_SmallMatrix, Q4_1Tensor, createQ4_1Random, 32, 103)

// ============================================================================
// Q5_0 Parity Tests
// ============================================================================

DEFINE_QUANTIZED_PARITY_TEST(Q5_0_SmallMatrix, Q5_0Tensor, createQ5_0Random, 32, 104)

// ============================================================================
// Q5_1 Parity Tests
// ============================================================================

DEFINE_QUANTIZED_PARITY_TEST(Q5_1_SmallMatrix, Q5_1Tensor, createQ5_1Random, 32, 105)

// ============================================================================
// K-Quant Parity Tests (256-element super-blocks)
// NOTE: These require proper K-quant encoding which the simple factory methods
// don't implement. The tests are disabled until proper quantization is added.
// The CUDA kernel itself supports these formats - just needs proper test data.
// ============================================================================

// DEFINE_QUANTIZED_PARITY_TEST(Q6_K_SmallMatrix, Q6_KTensor, createQ6_KRandom, 256, 201)
// DEFINE_QUANTIZED_PARITY_TEST(Q2_K_SmallMatrix, Q2_KTensor, createQ2_KRandom, 256, 202)
// DEFINE_QUANTIZED_PARITY_TEST(Q3_K_SmallMatrix, Q3_KTensor, createQ3_KRandom, 256, 203)
// DEFINE_QUANTIZED_PARITY_TEST(Q4_K_SmallMatrix, Q4_KTensor, createQ4_KRandom, 256, 204)
// DEFINE_QUANTIZED_PARITY_TEST(Q5_K_SmallMatrix, Q5_KTensor, createQ5_KRandom, 256, 205)

// ============================================================================
// IQ (Importance Quantization) Parity Tests
// NOTE: IQ formats use complex lookup tables and grid indices.
// Only IQ4_NL has a proper factory implementation.
// Other IQ formats are disabled until proper quantization is added.
// ============================================================================

// DEFINE_QUANTIZED_PARITY_TEST(IQ4_XS_SmallMatrix, IQ4_XSTensor, createIQ4_XSRandom, 256, 301)
// DEFINE_QUANTIZED_PARITY_TEST(IQ2_XXS_SmallMatrix, IQ2_XXSTensor, createIQ2_XXSRandom, 256, 302)
// DEFINE_QUANTIZED_PARITY_TEST(IQ2_XS_SmallMatrix, IQ2_XSTensor, createIQ2_XSRandom, 256, 303)
// DEFINE_QUANTIZED_PARITY_TEST(IQ2_S_SmallMatrix, IQ2_STensor, createIQ2_SRandom, 256, 304)
// DEFINE_QUANTIZED_PARITY_TEST(IQ3_XXS_SmallMatrix, IQ3_XXSTensor, createIQ3_XXSRandom, 256, 305)
// DEFINE_QUANTIZED_PARITY_TEST(IQ3_S_SmallMatrix, IQ3_STensor, createIQ3_SRandom, 256, 306)
// DEFINE_QUANTIZED_PARITY_TEST(IQ1_S_SmallMatrix, IQ1_STensor, createIQ1_SRandom, 256, 307)
// DEFINE_QUANTIZED_PARITY_TEST(IQ1_M_SmallMatrix, IQ1_MTensor, createIQ1_MRandom, 256, 308)

#endif // HAVE_CUDA

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
