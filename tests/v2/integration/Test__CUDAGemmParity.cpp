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
#include "tensors/TensorKernels.h" // For TensorProjectionDesc
#include "kernels/KernelFactory.h"
#include "backends/ComputeBackend.h"
#include "execution/DeviceContext.h"
#include "execution/GpuCoherence.h"           // For gpu_output(), with_gpu_coherence()
#include "execution/DeviceWorkspaceManager.h" // For workspace binding
#include "loaders/ModelLoader.h"
#include "tensors/TensorFactory.h"
#include "utils/MPIContext.h"
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
#include <filesystem>

using namespace llaminar2;
using namespace llaminar2::test::cuda;
using namespace llaminar2::test; // For TestTensorFactory

// Alias for kernel DeviceType to avoid ambiguity
using KernelDeviceType = llaminar::v2::kernels::DeviceType;

// Alias for TensorProjectionDesc
using TensorProjectionDesc = llaminar2::ITensorGemm::TensorProjectionDesc;

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

    // =========================================================================
    // Workspace Management Helpers
    // =========================================================================

    /**
     * @brief Set up workspace for a CUDA GEMM kernel if it supports it
     *
     * CUDA quantized GEMM kernels require pre-allocated workspace buffers for:
     * - Quantized activations (INT8)
     * - Per-row scales
     * - INT32 intermediate accumulator
     *
     * FP32 kernels don't need workspace, so this is a no-op for them.
     *
     * @param kernel The GEMM kernel
     * @param M Maximum batch/sequence length
     * @param N Output dimension
     * @param K Input dimension
     * @return true on success or if kernel doesn't need workspace, false on allocation failure
     */
    bool setupWorkspaceIfNeeded(ITensorGemm *kernel, int M, int N, int K)
    {
        auto *ws_consumer = dynamic_cast<IWorkspaceConsumer *>(kernel);
        if (!ws_consumer)
        {
            // Kernel doesn't implement IWorkspaceConsumer (e.g., FP32 kernel)
            return true;
        }

        auto reqs = ws_consumer->getWorkspaceRequirements(M, N, K);
        workspace_ = std::make_unique<DeviceWorkspaceManager>(gpu_device_, 64 * 1024 * 1024); // 64MB
        if (!workspace_->allocate(reqs))
        {
            LOG_ERROR("Failed to allocate workspace for CUDA GEMM kernel");
            return false;
        }
        ws_consumer->bindWorkspace(workspace_.get());
        return true;
    }

    /**
     * @brief Clean up workspace after test
     */
    void cleanupWorkspaceIfNeeded(ITensorGemm *kernel)
    {
        if (workspace_)
        {
            auto *ws_consumer = dynamic_cast<IWorkspaceConsumer *>(kernel);
            if (ws_consumer)
            {
                ws_consumer->unbindWorkspace();
            }
            workspace_.reset();
        }
    }

    /**
     * @brief Set up a shared workspace for multiple CUDA GEMM kernels
     *
     * Used for fused QKV operations where multiple kernels share workspace.
     * Computes the maximum requirements across all kernels.
     *
     * @param kernels Vector of kernels that need workspace
     * @param M Maximum batch/sequence length
     * @param Ns Output dimensions for each kernel
     * @param K Input dimension (shared)
     * @return true on success, false on allocation failure
     */
    bool setupSharedWorkspace(
        const std::vector<ITensorGemm *> &kernels,
        int M,
        const std::vector<int> &Ns,
        int K)
    {
        // Find max requirements across all kernels
        size_t max_quant_a = 0, max_scales_a = 0, max_acc = 0;

        for (size_t i = 0; i < kernels.size(); ++i)
        {
            auto *ws_consumer = dynamic_cast<IWorkspaceConsumer *>(kernels[i]);
            if (ws_consumer)
            {
                auto reqs = ws_consumer->getWorkspaceRequirements(M, Ns[i], K);
                for (const auto &buf : reqs.buffers)
                {
                    if (buf.name == "gemm_quant_a")
                        max_quant_a = std::max(max_quant_a, buf.size_bytes);
                    else if (buf.name == "gemm_scales_a")
                        max_scales_a = std::max(max_scales_a, buf.size_bytes);
                    else if (buf.name == "gemm_acc_int32")
                        max_acc = std::max(max_acc, buf.size_bytes);
                }
            }
        }

        // Create shared workspace with max requirements
        WorkspaceRequirements shared_reqs;
        shared_reqs.buffers.push_back({"gemm_quant_a", max_quant_a, 256, true});
        shared_reqs.buffers.push_back({"gemm_scales_a", max_scales_a, 256, true});
        shared_reqs.buffers.push_back({"gemm_acc_int32", max_acc, 256, true});

        workspace_ = std::make_unique<DeviceWorkspaceManager>(gpu_device_, 64 * 1024 * 1024); // 64MB
        if (!workspace_->allocate(shared_reqs))
        {
            LOG_ERROR("Failed to allocate shared workspace");
            return false;
        }

        // Bind shared workspace to ALL kernels
        for (auto *kernel : kernels)
        {
            auto *ws_consumer = dynamic_cast<IWorkspaceConsumer *>(kernel);
            if (ws_consumer)
            {
                ws_consumer->bindWorkspace(workspace_.get());
            }
        }

        return true;
    }

    /**
     * @brief Clean up shared workspace from multiple kernels
     */
    void cleanupSharedWorkspace(const std::vector<ITensorGemm *> &kernels)
    {
        for (auto *kernel : kernels)
        {
            auto *ws_consumer = dynamic_cast<IWorkspaceConsumer *>(kernel);
            if (ws_consumer && ws_consumer->hasWorkspace())
            {
                ws_consumer->unbindWorkspace();
            }
        }
        workspace_.reset();
    }

    std::unique_ptr<DeviceWorkspaceManager> workspace_;
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

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel.get(), M, N, K));

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
    cleanupWorkspaceIfNeeded(cuda_kernel.get());
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

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel.get(), M, N, K));

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

    cleanupWorkspaceIfNeeded(cuda_kernel.get());
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

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel.get(), M, N, K));

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

    cleanupWorkspaceIfNeeded(cuda_kernel.get());
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
        ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));                                  \
        auto cuda_kernel = llaminar::v2::kernels::KernelFactory::createGemm(                \
            weights.get(), KernelDeviceType::CUDA);                                         \
        ASSERT_NE(cuda_kernel, nullptr) << "Failed to create CUDA kernel for " #TensorType; \
                                                                                            \
        /* Set up workspace for quantized kernel */                                         \
        ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel.get(), M, N, K));                    \
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
        cleanupWorkspaceIfNeeded(cuda_kernel.get());                                        \
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

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel.get(), M, N, K));

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

    cleanupWorkspaceIfNeeded(cuda_kernel.get());
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

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel.get(), M, N, K));

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

    cleanupWorkspaceIfNeeded(cuda_kernel.get());
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

// ============================================================================
// Real Model Weight Parity Tests
// ============================================================================
// These tests load actual Q4_0 weights from a GGUF model file to verify
// that CUDA GEMM produces correct results with real-world weight distributions.
// This is critical because synthetic random weights may not expose issues
// that occur with the specific value patterns in trained models.

namespace
{
    constexpr const char *REAL_MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
}

/**
 * @test Q4_0 parity with real model weights: attn_q.weight (layer 0)
 *
 * Tests Q projection weight matrix which is critical for attention.
 * This is one of the weights that showed massive divergence in full inference
 * (cosine=0.098 vs expected ~1.0).
 */
TEST_F(Test__CUDAGemmParity, RealModel_Q4_0_AttnQ_Layer0)
{
    if (!std::filesystem::exists(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Model file not found: " << REAL_MODEL_PATH;
    }

    // Create MPI context (single rank for this test)
    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    TensorFactory factory(*mpi_ctx);
    ModelLoader loader(&factory);

    if (!loader.loadModel(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Failed to load model: " << REAL_MODEL_PATH;
    }

    // Load real Q projection weight
    auto weights = loader.loadTensor("blk.0.attn_q.weight", DeviceId::cpu());
    ASSERT_NE(weights, nullptr) << "Failed to load blk.0.attn_q.weight";

    auto *q4_tensor = dynamic_cast<Q4_0Tensor *>(weights.get());
    ASSERT_NE(q4_tensor, nullptr) << "Expected Q4_0Tensor, got different type";

    // Dimensions: [N, K] where N=896 (output), K=896 (input) for Qwen2.5-0.5B
    const int M = 4; // Small batch for testing
    const int N = static_cast<int>(q4_tensor->shape()[0]);
    const int K = static_cast<int>(q4_tensor->shape()[1]);

    std::cout << "Real model attn_q weight: " << N << "x" << K << " (Q4_0)\n";

    // Create random activations
    auto A_data = randomFP32(M * K);

    // ===== CPU Reference =====
    std::vector<float> C_cpu(M * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        q4_tensor, KernelDeviceType::CPU);
    ASSERT_NE(cpu_kernel, nullptr) << "Failed to create CPU kernel";
    ASSERT_TRUE(cpu_kernel->multiply(A_data.data(), C_cpu.data(), M, N, K));

    // ===== CUDA =====
    // Upload weights to GPU
    ASSERT_TRUE(q4_tensor->ensureOnDevice(gpu_device_));
    EXPECT_TRUE(q4_tensor->isOnGPU());

    auto cuda_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        q4_tensor, KernelDeviceType::CUDA);
    ASSERT_NE(cuda_kernel, nullptr) << "Failed to create CUDA kernel";

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel.get(), M, N, K));

    float *d_A, *d_C;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_A, M * K * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_C, M * N * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_A, A_data.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemset(d_C, 0, M * N * sizeof(float)));

    ASSERT_TRUE(cuda_kernel->multiply(d_A, d_C, M, N, K));

    std::vector<float> C_cuda(M * N);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(C_cuda.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // ===== Compare =====
    auto result = checkParity(C_cuda.data(), C_cpu.data(), M * N, 0.99, 0.10);
    result.print("Real Model Q4_0 attn_q (layer 0)");

    EXPECT_FALSE(result.has_nan_inf) << "CUDA output contains NaN/Inf";
    EXPECT_GE(result.cosine_similarity, 0.99)
        << "Cosine similarity too low: " << result.cosine_similarity;

    cleanupWorkspaceIfNeeded(cuda_kernel.get());
    cudaFree(d_A);
    cudaFree(d_C);
}

/**
 * @test Q4_0 parity with real model weights: attn_k.weight (layer 0)
 *
 * Tests K projection weight matrix. K projection showed even worse divergence
 * than Q in full inference (cosine=0.031).
 */
TEST_F(Test__CUDAGemmParity, RealModel_Q4_0_AttnK_Layer0)
{
    if (!std::filesystem::exists(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Model file not found: " << REAL_MODEL_PATH;
    }

    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    TensorFactory factory(*mpi_ctx);
    ModelLoader loader(&factory);

    if (!loader.loadModel(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Failed to load model: " << REAL_MODEL_PATH;
    }

    auto weights = loader.loadTensor("blk.0.attn_k.weight", DeviceId::cpu());
    ASSERT_NE(weights, nullptr) << "Failed to load blk.0.attn_k.weight";

    auto *q4_tensor = dynamic_cast<Q4_0Tensor *>(weights.get());
    ASSERT_NE(q4_tensor, nullptr) << "Expected Q4_0Tensor";

    const int M = 4;
    const int N = static_cast<int>(q4_tensor->shape()[0]);
    const int K = static_cast<int>(q4_tensor->shape()[1]);

    std::cout << "Real model attn_k weight: " << N << "x" << K << " (Q4_0)\n";

    auto A_data = randomFP32(M * K);

    // CPU
    std::vector<float> C_cpu(M * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        q4_tensor, KernelDeviceType::CPU);
    ASSERT_TRUE(cpu_kernel->multiply(A_data.data(), C_cpu.data(), M, N, K));

    // CUDA
    ASSERT_TRUE(q4_tensor->ensureOnDevice(gpu_device_));
    auto cuda_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        q4_tensor, KernelDeviceType::CUDA);

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel.get(), M, N, K));

    float *d_A, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMemcpy(d_A, A_data.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);

    ASSERT_TRUE(cuda_kernel->multiply(d_A, d_C, M, N, K));

    std::vector<float> C_cuda(M * N);
    cudaMemcpy(C_cuda.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    auto result = checkParity(C_cuda.data(), C_cpu.data(), M * N, 0.99, 0.10);
    result.print("Real Model Q4_0 attn_k (layer 0)");

    EXPECT_GE(result.cosine_similarity, 0.99);
    EXPECT_FALSE(result.has_nan_inf);

    cleanupWorkspaceIfNeeded(cuda_kernel.get());
    cudaFree(d_A);
    cudaFree(d_C);
}

/**
 * @test Q4_0 parity with real model weights: attn_v.weight (layer 0)
 *
 * Tests V projection. V showed less divergence (cosine=0.84) but still
 * significantly off in full inference.
 */
TEST_F(Test__CUDAGemmParity, RealModel_Q4_0_AttnV_Layer0)
{
    if (!std::filesystem::exists(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Model file not found: " << REAL_MODEL_PATH;
    }

    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    TensorFactory factory(*mpi_ctx);
    ModelLoader loader(&factory);

    if (!loader.loadModel(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Failed to load model: " << REAL_MODEL_PATH;
    }

    auto weights = loader.loadTensor("blk.0.attn_v.weight", DeviceId::cpu());
    ASSERT_NE(weights, nullptr) << "Failed to load blk.0.attn_v.weight";

    auto *q4_tensor = dynamic_cast<Q4_0Tensor *>(weights.get());
    ASSERT_NE(q4_tensor, nullptr) << "Expected Q4_0Tensor";

    const int M = 4;
    const int N = static_cast<int>(q4_tensor->shape()[0]);
    const int K = static_cast<int>(q4_tensor->shape()[1]);

    std::cout << "Real model attn_v weight: " << N << "x" << K << " (Q4_0)\n";

    auto A_data = randomFP32(M * K);

    // CPU
    std::vector<float> C_cpu(M * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        q4_tensor, KernelDeviceType::CPU);
    ASSERT_TRUE(cpu_kernel->multiply(A_data.data(), C_cpu.data(), M, N, K));

    // CUDA
    ASSERT_TRUE(q4_tensor->ensureOnDevice(gpu_device_));
    auto cuda_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        q4_tensor, KernelDeviceType::CUDA);

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel.get(), M, N, K));

    float *d_A, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMemcpy(d_A, A_data.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);

    ASSERT_TRUE(cuda_kernel->multiply(d_A, d_C, M, N, K));

    std::vector<float> C_cuda(M * N);
    cudaMemcpy(C_cuda.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    auto result = checkParity(C_cuda.data(), C_cpu.data(), M * N, 0.99, 0.10);
    result.print("Real Model Q4_0 attn_v (layer 0)");

    EXPECT_GE(result.cosine_similarity, 0.99);
    EXPECT_FALSE(result.has_nan_inf);

    cleanupWorkspaceIfNeeded(cuda_kernel.get());
    cudaFree(d_A);
    cudaFree(d_C);
}

/**
 * @test Q4_0 parity with real model weights: ffn_gate.weight (layer 0)
 *
 * Tests FFN gate weight which is a larger matrix (4864x896 for Qwen2.5-0.5B).
 */
TEST_F(Test__CUDAGemmParity, RealModel_Q4_0_FFNGate_Layer0)
{
    if (!std::filesystem::exists(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Model file not found: " << REAL_MODEL_PATH;
    }

    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    TensorFactory factory(*mpi_ctx);
    ModelLoader loader(&factory);

    if (!loader.loadModel(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Failed to load model: " << REAL_MODEL_PATH;
    }

    auto weights = loader.loadTensor("blk.0.ffn_gate.weight", DeviceId::cpu());
    ASSERT_NE(weights, nullptr) << "Failed to load blk.0.ffn_gate.weight";

    auto *q4_tensor = dynamic_cast<Q4_0Tensor *>(weights.get());
    ASSERT_NE(q4_tensor, nullptr) << "Expected Q4_0Tensor";

    const int M = 4;
    const int N = static_cast<int>(q4_tensor->shape()[0]);
    const int K = static_cast<int>(q4_tensor->shape()[1]);

    std::cout << "Real model ffn_gate weight: " << N << "x" << K << " (Q4_0)\n";

    auto A_data = randomFP32(M * K);

    // CPU
    std::vector<float> C_cpu(M * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        q4_tensor, KernelDeviceType::CPU);
    ASSERT_TRUE(cpu_kernel->multiply(A_data.data(), C_cpu.data(), M, N, K));

    // CUDA
    ASSERT_TRUE(q4_tensor->ensureOnDevice(gpu_device_));
    auto cuda_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        q4_tensor, KernelDeviceType::CUDA);

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel.get(), M, N, K));

    float *d_A, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMemcpy(d_A, A_data.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);

    ASSERT_TRUE(cuda_kernel->multiply(d_A, d_C, M, N, K));

    std::vector<float> C_cuda(M * N);
    cudaMemcpy(C_cuda.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    auto result = checkParity(C_cuda.data(), C_cpu.data(), M * N, 0.99, 0.10);
    result.print("Real Model Q4_0 ffn_gate (layer 0)");

    EXPECT_GE(result.cosine_similarity, 0.99);
    EXPECT_FALSE(result.has_nan_inf);

    cleanupWorkspaceIfNeeded(cuda_kernel.get());
    cudaFree(d_A);
    cudaFree(d_C);
}

/**
 * @test Q4_0 parity with real model weights: output/lm_head.weight
 *
 * Tests vocabulary projection (LM head) which is the final layer.
 * Shape: [vocab_size, hidden_dim] = [151936, 896] for Qwen2.5
 */
TEST_F(Test__CUDAGemmParity, RealModel_Q4_0_LMHead)
{
    if (!std::filesystem::exists(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Model file not found: " << REAL_MODEL_PATH;
    }

    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    TensorFactory factory(*mpi_ctx);
    ModelLoader loader(&factory);

    if (!loader.loadModel(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Failed to load model: " << REAL_MODEL_PATH;
    }

    auto weights = loader.loadTensor("output.weight", DeviceId::cpu());
    ASSERT_NE(weights, nullptr) << "Failed to load output.weight (LM head)";

    auto *q4_tensor = dynamic_cast<Q4_0Tensor *>(weights.get());
    if (!q4_tensor)
    {
        // LM head might be in a different format (e.g., FP32/FP16)
        std::cout << "LM head is not Q4_0, tensor type: "
                  << static_cast<int>(weights->native_type()) << "\n";
        GTEST_SKIP() << "LM head is not Q4_0 format";
    }

    const int M = 2;                                       // Smaller batch for large vocab
    const int N = static_cast<int>(q4_tensor->shape()[0]); // vocab_size
    const int K = static_cast<int>(q4_tensor->shape()[1]); // hidden_dim

    std::cout << "Real model LM head weight: " << N << "x" << K << " (Q4_0)\n";

    auto A_data = randomFP32(M * K);

    // CPU
    std::vector<float> C_cpu(M * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        q4_tensor, KernelDeviceType::CPU);
    ASSERT_TRUE(cpu_kernel->multiply(A_data.data(), C_cpu.data(), M, N, K));

    // CUDA
    ASSERT_TRUE(q4_tensor->ensureOnDevice(gpu_device_));
    auto cuda_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        q4_tensor, KernelDeviceType::CUDA);

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel.get(), M, N, K));

    float *d_A, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_C, static_cast<size_t>(M) * N * sizeof(float));
    cudaMemcpy(d_A, A_data.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);

    ASSERT_TRUE(cuda_kernel->multiply(d_A, d_C, M, N, K));

    std::vector<float> C_cuda(static_cast<size_t>(M) * N);
    cudaMemcpy(C_cuda.data(), d_C, static_cast<size_t>(M) * N * sizeof(float), cudaMemcpyDeviceToHost);

    auto result = checkParity(C_cuda.data(), C_cpu.data(), static_cast<size_t>(M) * N, 0.99, 0.10);
    result.print("Real Model Q4_0 LM Head");

    EXPECT_GE(result.cosine_similarity, 0.99);
    EXPECT_FALSE(result.has_nan_inf);

    cleanupWorkspaceIfNeeded(cuda_kernel.get());
    cudaFree(d_A);
    cudaFree(d_C);
}

/**
 * @test Q4_0 parity with real model weights using TENSOR API
 *
 * This test uses the multiply_tensor() API (same as full inference)
 * instead of the raw multiply() API to see if the issue is in the
 * tensor-based code path.
 */
TEST_F(Test__CUDAGemmParity, RealModel_Q4_0_AttnQ_TensorAPI)
{
    if (!std::filesystem::exists(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Model file not found: " << REAL_MODEL_PATH;
    }

    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    TensorFactory factory(*mpi_ctx);
    ModelLoader loader(&factory);

    if (!loader.loadModel(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Failed to load model: " << REAL_MODEL_PATH;
    }

    auto weights = loader.loadTensor("blk.0.attn_q.weight", DeviceId::cpu());
    ASSERT_NE(weights, nullptr);

    auto *q4_tensor = dynamic_cast<Q4_0Tensor *>(weights.get());
    ASSERT_NE(q4_tensor, nullptr);

    const int M = 4;
    const int N = static_cast<int>(q4_tensor->shape()[0]);
    const int K = static_cast<int>(q4_tensor->shape()[1]);

    std::cout << "Testing multiply_tensor() API with attn_q: " << N << "x" << K << "\n";

    // Create FP32 input and output tensors
    auto input_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    auto output_cpu = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});
    auto output_cuda = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});

    // Fill input with random data
    float *input_data = input_tensor->mutable_data();
    for (int i = 0; i < M * K; ++i)
    {
        input_data[i] = dist_(rng_);
    }

    // ===== CPU: multiply_tensor() =====
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        q4_tensor, KernelDeviceType::CPU);
    ASSERT_NE(cpu_kernel, nullptr);
    ASSERT_TRUE(cpu_kernel->multiply_tensor(
        input_tensor.get(), output_cpu.get(),
        M, N, K, true, 1.0f, 0.0f, nullptr, nullptr, -1));

    // ===== CUDA: multiply_tensor() =====
    // First ensure weights are on GPU
    ASSERT_TRUE(q4_tensor->ensureOnDevice(gpu_device_));

    auto cuda_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        q4_tensor, KernelDeviceType::CUDA);
    ASSERT_NE(cuda_kernel, nullptr);

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel.get(), M, N, K));

    // Use with_gpu_coherence for automatic input/output coherence management
    ASSERT_TRUE(with_gpu_coherence(
        gpu_device_,
        {input_tensor.get()}, // inputs
        {output_cuda.get()},  // outputs (will be marked dirty after kernel)
        [&]
        {
            return cuda_kernel->multiply_tensor(
                input_tensor.get(), output_cuda.get(),
                M, N, K, true, 1.0f, 0.0f, nullptr, nullptr, -1);
        }));

    // Clean up workspace
    cleanupWorkspaceIfNeeded(cuda_kernel.get());

    // ===== Compare =====
    // data() will automatically sync to host if needed
    const float *cpu_data = output_cpu->data();
    const float *cuda_data = output_cuda->data();

    auto result = checkParity(cuda_data, cpu_data, M * N, 0.99, 0.10);
    result.print("Real Model Q4_0 attn_q (multiply_tensor API)");

    EXPECT_GE(result.cosine_similarity, 0.99)
        << "multiply_tensor() API shows divergence!";
    EXPECT_FALSE(result.has_nan_inf);
}

/**
 * @test Fused QKV GEMM parity: multiply_fused_tensor() vs 3x multiply_tensor()
 *
 * This test validates the new multiply_fused_tensor() method which:
 * 1. Uploads input to GPU once
 * 2. Quantizes activations to INT8 once (shared across all projections)
 * 3. Runs all Q/K/V projections using the shared quantized activations
 *
 * **Key validation**: The fused path should produce identical results to
 * calling multiply_tensor() three times separately, since the quantization
 * of activations should be deterministic.
 *
 * This test was written to verify the fix for CUDA full model divergence
 * where the default multiply_fused_tensor() implementation was calling
 * multiply_tensor() in a loop, which requantized activations for each
 * projection (inefficient and potentially inconsistent).
 */
TEST_F(Test__CUDAGemmParity, FusedQKV_TensorAPI_vs_Separate)
{
    if (!std::filesystem::exists(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Model file not found: " << REAL_MODEL_PATH;
    }

    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    TensorFactory factory(*mpi_ctx);
    ModelLoader loader(&factory);

    if (!loader.loadModel(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Failed to load model: " << REAL_MODEL_PATH;
    }

    // Load Q, K, V projection weights
    auto weights_q_base = loader.loadTensor("blk.0.attn_q.weight", DeviceId::cpu());
    auto weights_k_base = loader.loadTensor("blk.0.attn_k.weight", DeviceId::cpu());
    auto weights_v_base = loader.loadTensor("blk.0.attn_v.weight", DeviceId::cpu());
    ASSERT_NE(weights_q_base, nullptr) << "Failed to load attn_q.weight";
    ASSERT_NE(weights_k_base, nullptr) << "Failed to load attn_k.weight";
    ASSERT_NE(weights_v_base, nullptr) << "Failed to load attn_v.weight";

    // Cast to concrete Q4_0 type
    auto *weights_q = dynamic_cast<Q4_0Tensor *>(weights_q_base.get());
    auto *weights_k = dynamic_cast<Q4_0Tensor *>(weights_k_base.get());
    auto *weights_v = dynamic_cast<Q4_0Tensor *>(weights_v_base.get());
    ASSERT_NE(weights_q, nullptr) << "Expected Q4_0Tensor for Q weights";
    ASSERT_NE(weights_k, nullptr) << "Expected Q4_0Tensor for K weights";
    ASSERT_NE(weights_v, nullptr) << "Expected Q4_0Tensor for V weights";

    // Get dimensions
    const int M = 4; // Small batch for testing (could also test M=1 for decode)
    const int N_q = static_cast<int>(weights_q->shape()[0]);
    const int N_k = static_cast<int>(weights_k->shape()[0]);
    const int N_v = static_cast<int>(weights_v->shape()[0]);
    const int K = static_cast<int>(weights_q->shape()[1]);

    std::cout << "FusedQKV test: M=" << M << " K=" << K
              << " N_q=" << N_q << " N_k=" << N_k << " N_v=" << N_v << "\n";

    // Ensure all weights are on GPU
    ASSERT_TRUE(weights_q->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(weights_k->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(weights_v->ensureOnDevice(gpu_device_));

    // Create CUDA kernels for each projection
    auto cuda_kernel_q = llaminar::v2::kernels::KernelFactory::createGemm(
        weights_q, KernelDeviceType::CUDA);
    auto cuda_kernel_k = llaminar::v2::kernels::KernelFactory::createGemm(
        weights_k, KernelDeviceType::CUDA);
    auto cuda_kernel_v = llaminar::v2::kernels::KernelFactory::createGemm(
        weights_v, KernelDeviceType::CUDA);
    ASSERT_NE(cuda_kernel_q, nullptr) << "Failed to create CUDA kernel for Q";
    ASSERT_NE(cuda_kernel_k, nullptr) << "Failed to create CUDA kernel for K";
    ASSERT_NE(cuda_kernel_v, nullptr) << "Failed to create CUDA kernel for V";

    // Set up SHARED workspace for fused QKV (all kernels share workspace)
    ASSERT_TRUE(setupSharedWorkspace(
        {cuda_kernel_q.get(), cuda_kernel_k.get(), cuda_kernel_v.get()},
        M, {N_q, N_k, N_v}, K));

    // Create input tensor with random data
    auto input_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    float *input_data = input_tensor->mutable_data();
    for (int i = 0; i < M * K; ++i)
    {
        input_data[i] = dist_(rng_);
    }

    // Create output tensors for SEPARATE path
    auto output_q_separate = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_q)});
    auto output_k_separate = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_k)});
    auto output_v_separate = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_v)});

    // Create output tensors for FUSED path
    auto output_q_fused = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_q)});
    auto output_k_fused = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_k)});
    auto output_v_fused = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_v)});

    // ===== SEPARATE PATH: 3x multiply_tensor() =====
    std::cout << "Running SEPARATE path (3x multiply_tensor)...\n";
    ASSERT_TRUE(with_gpu_coherence(
        gpu_device_,
        {input_tensor.get()},
        {output_q_separate.get(), output_k_separate.get(), output_v_separate.get()},
        [&]
        {
            return cuda_kernel_q->multiply_tensor(
                       input_tensor.get(), output_q_separate.get(),
                       M, N_q, K, true, 1.0f, 0.0f, nullptr, nullptr, -1) &&
                   cuda_kernel_k->multiply_tensor(
                       input_tensor.get(), output_k_separate.get(),
                       M, N_k, K, true, 1.0f, 0.0f, nullptr, nullptr, -1) &&
                   cuda_kernel_v->multiply_tensor(
                       input_tensor.get(), output_v_separate.get(),
                       M, N_v, K, true, 1.0f, 0.0f, nullptr, nullptr, -1);
        }));

    // ===== FUSED PATH: multiply_fused_tensor() =====
    std::cout << "Running FUSED path (multiply_fused_tensor)...\n";

    // Build projection descriptors
    std::vector<TensorProjectionDesc> projections;
    projections.emplace_back(cuda_kernel_q.get(), output_q_fused.get(), N_q,
                             nullptr, nullptr, false, "Q");
    projections.emplace_back(cuda_kernel_k.get(), output_k_fused.get(), N_k,
                             nullptr, nullptr, false, "K");
    projections.emplace_back(cuda_kernel_v.get(), output_v_fused.get(), N_v,
                             nullptr, nullptr, false, "V");

    // Call fused method with coherence wrapper
    ASSERT_TRUE(with_gpu_coherence(
        gpu_device_,
        {input_tensor.get()},
        {output_q_fused.get(), output_k_fused.get(), output_v_fused.get()},
        [&]
        {
            return cuda_kernel_q->multiply_fused_tensor(
                input_tensor.get(), projections, M, K, nullptr);
        }));

    // ===== COMPARE: Fused vs Separate =====
    std::cout << "\nComparing FUSED vs SEPARATE results:\n";

    // Sync back to host for comparison
    const float *q_separate = output_q_separate->data();
    const float *k_separate = output_k_separate->data();
    const float *v_separate = output_v_separate->data();
    const float *q_fused = output_q_fused->data();
    const float *k_fused = output_k_fused->data();
    const float *v_fused = output_v_fused->data();

    // Q projection parity
    auto result_q = checkParity(q_fused, q_separate, M * N_q, 0.9999, 0.001);
    result_q.print("Q projection (fused vs separate)");
    EXPECT_GE(result_q.cosine_similarity, 0.9999)
        << "Q projection: fused and separate should be nearly identical";
    EXPECT_FALSE(result_q.has_nan_inf);

    // K projection parity
    auto result_k = checkParity(k_fused, k_separate, M * N_k, 0.9999, 0.001);
    result_k.print("K projection (fused vs separate)");
    EXPECT_GE(result_k.cosine_similarity, 0.9999)
        << "K projection: fused and separate should be nearly identical";
    EXPECT_FALSE(result_k.has_nan_inf);

    // V projection parity
    auto result_v = checkParity(v_fused, v_separate, M * N_v, 0.9999, 0.001);
    result_v.print("V projection (fused vs separate)");
    EXPECT_GE(result_v.cosine_similarity, 0.9999)
        << "V projection: fused and separate should be nearly identical";
    EXPECT_FALSE(result_v.has_nan_inf);

    // Also compare against CPU to ensure correctness
    std::cout << "\nComparing FUSED vs CPU reference:\n";

    auto cpu_kernel_q = llaminar::v2::kernels::KernelFactory::createGemm(
        weights_q, KernelDeviceType::CPU);
    auto cpu_kernel_k = llaminar::v2::kernels::KernelFactory::createGemm(
        weights_k, KernelDeviceType::CPU);
    auto cpu_kernel_v = llaminar::v2::kernels::KernelFactory::createGemm(
        weights_v, KernelDeviceType::CPU);

    std::vector<float> q_cpu(M * N_q), k_cpu(M * N_k), v_cpu(M * N_v);
    const float *h_input = input_tensor->data(); // Sync input to host
    ASSERT_TRUE(cpu_kernel_q->multiply(h_input, q_cpu.data(), M, N_q, K));
    ASSERT_TRUE(cpu_kernel_k->multiply(h_input, k_cpu.data(), M, N_k, K));
    ASSERT_TRUE(cpu_kernel_v->multiply(h_input, v_cpu.data(), M, N_v, K));

    auto result_q_cpu = checkParity(q_fused, q_cpu.data(), M * N_q, 0.99, 0.10);
    result_q_cpu.print("Q projection (CUDA fused vs CPU)");
    EXPECT_GE(result_q_cpu.cosine_similarity, 0.99);

    auto result_k_cpu = checkParity(k_fused, k_cpu.data(), M * N_k, 0.99, 0.10);
    result_k_cpu.print("K projection (CUDA fused vs CPU)");
    EXPECT_GE(result_k_cpu.cosine_similarity, 0.99);

    auto result_v_cpu = checkParity(v_fused, v_cpu.data(), M * N_v, 0.99, 0.10);
    result_v_cpu.print("V projection (CUDA fused vs CPU)");
    EXPECT_GE(result_v_cpu.cosine_similarity, 0.99);

    // Clean up shared workspace
    cleanupSharedWorkspace({cuda_kernel_q.get(), cuda_kernel_k.get(), cuda_kernel_v.get()});
}

/**
 * @test Fused QKV with decode-size batch (M=1)
 *
 * Tests the fused path with M=1 which is the common case during autoregressive
 * decoding. This is important because the quantization behavior might differ
 * with single-row inputs.
 */
TEST_F(Test__CUDAGemmParity, FusedQKV_DecodeSize_M1)
{
    if (!std::filesystem::exists(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Model file not found: " << REAL_MODEL_PATH;
    }

    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    TensorFactory factory(*mpi_ctx);
    ModelLoader loader(&factory);

    if (!loader.loadModel(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Failed to load model: " << REAL_MODEL_PATH;
    }

    auto weights_q_base = loader.loadTensor("blk.0.attn_q.weight", DeviceId::cpu());
    auto weights_k_base = loader.loadTensor("blk.0.attn_k.weight", DeviceId::cpu());
    auto weights_v_base = loader.loadTensor("blk.0.attn_v.weight", DeviceId::cpu());
    ASSERT_NE(weights_q_base, nullptr);
    ASSERT_NE(weights_k_base, nullptr);
    ASSERT_NE(weights_v_base, nullptr);

    // Cast to concrete Q4_0 type
    auto *weights_q = dynamic_cast<Q4_0Tensor *>(weights_q_base.get());
    auto *weights_k = dynamic_cast<Q4_0Tensor *>(weights_k_base.get());
    auto *weights_v = dynamic_cast<Q4_0Tensor *>(weights_v_base.get());
    ASSERT_NE(weights_q, nullptr) << "Expected Q4_0Tensor for Q weights";
    ASSERT_NE(weights_k, nullptr) << "Expected Q4_0Tensor for K weights";
    ASSERT_NE(weights_v, nullptr) << "Expected Q4_0Tensor for V weights";

    const int M = 1; // Decode size!
    const int N_q = static_cast<int>(weights_q->shape()[0]);
    const int N_k = static_cast<int>(weights_k->shape()[0]);
    const int N_v = static_cast<int>(weights_v->shape()[0]);
    const int K = static_cast<int>(weights_q->shape()[1]);

    std::cout << "FusedQKV DECODE test: M=" << M << " K=" << K
              << " N_q=" << N_q << " N_k=" << N_k << " N_v=" << N_v << "\n";

    ASSERT_TRUE(weights_q->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(weights_k->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(weights_v->ensureOnDevice(gpu_device_));

    auto cuda_kernel_q = llaminar::v2::kernels::KernelFactory::createGemm(
        weights_q, KernelDeviceType::CUDA);
    auto cuda_kernel_k = llaminar::v2::kernels::KernelFactory::createGemm(
        weights_k, KernelDeviceType::CUDA);
    auto cuda_kernel_v = llaminar::v2::kernels::KernelFactory::createGemm(
        weights_v, KernelDeviceType::CUDA);

    // Set up SHARED workspace for fused QKV (all kernels share workspace)
    ASSERT_TRUE(setupSharedWorkspace(
        {cuda_kernel_q.get(), cuda_kernel_k.get(), cuda_kernel_v.get()},
        M, {N_q, N_k, N_v}, K));

    auto input_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    float *input_data = input_tensor->mutable_data();
    for (int i = 0; i < M * K; ++i)
    {
        input_data[i] = dist_(rng_);
    }

    // Fused outputs
    auto output_q_fused = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_q)});
    auto output_k_fused = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_k)});
    auto output_v_fused = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_v)});

    // Run fused with coherence wrapper
    std::vector<TensorProjectionDesc> projections;
    projections.emplace_back(cuda_kernel_q.get(), output_q_fused.get(), N_q,
                             nullptr, nullptr, false, "Q");
    projections.emplace_back(cuda_kernel_k.get(), output_k_fused.get(), N_k,
                             nullptr, nullptr, false, "K");
    projections.emplace_back(cuda_kernel_v.get(), output_v_fused.get(), N_v,
                             nullptr, nullptr, false, "V");

    ASSERT_TRUE(with_gpu_coherence(
        gpu_device_,
        {input_tensor.get()},
        {output_q_fused.get(), output_k_fused.get(), output_v_fused.get()},
        [&]
        {
            return cuda_kernel_q->multiply_fused_tensor(
                input_tensor.get(), projections, M, K, nullptr);
        }));

    // Compare against CPU
    auto cpu_kernel_q = llaminar::v2::kernels::KernelFactory::createGemm(
        weights_q, KernelDeviceType::CPU);
    auto cpu_kernel_k = llaminar::v2::kernels::KernelFactory::createGemm(
        weights_k, KernelDeviceType::CPU);
    auto cpu_kernel_v = llaminar::v2::kernels::KernelFactory::createGemm(
        weights_v, KernelDeviceType::CPU);

    std::vector<float> q_cpu(M * N_q), k_cpu(M * N_k), v_cpu(M * N_v);
    const float *h_input = input_tensor->data();
    ASSERT_TRUE(cpu_kernel_q->multiply(h_input, q_cpu.data(), M, N_q, K));
    ASSERT_TRUE(cpu_kernel_k->multiply(h_input, k_cpu.data(), M, N_k, K));
    ASSERT_TRUE(cpu_kernel_v->multiply(h_input, v_cpu.data(), M, N_v, K));

    const float *q_fused = output_q_fused->data();
    const float *k_fused = output_k_fused->data();
    const float *v_fused = output_v_fused->data();

    auto result_q = checkParity(q_fused, q_cpu.data(), M * N_q, 0.99, 0.10);
    result_q.print("Q decode (CUDA fused vs CPU)");
    EXPECT_GE(result_q.cosine_similarity, 0.99);
    EXPECT_FALSE(result_q.has_nan_inf);

    auto result_k = checkParity(k_fused, k_cpu.data(), M * N_k, 0.99, 0.10);
    result_k.print("K decode (CUDA fused vs CPU)");
    EXPECT_GE(result_k.cosine_similarity, 0.99);
    EXPECT_FALSE(result_k.has_nan_inf);

    auto result_v = checkParity(v_fused, v_cpu.data(), M * N_v, 0.99, 0.10);
    result_v.print("V decode (CUDA fused vs CPU)");
    EXPECT_GE(result_v.cosine_similarity, 0.99);
    EXPECT_FALSE(result_v.has_nan_inf);

    // Clean up shared workspace
    cleanupSharedWorkspace({cuda_kernel_q.get(), cuda_kernel_k.get(), cuda_kernel_v.get()});
}

// ============================================================================
// Cached Kernel Tests
// ============================================================================

/**
 * @test Cached kernel parity: getOrCreateGemm() vs createGemm()
 *
 * This test verifies that kernels obtained via getOrCreateGemm() (the caching API
 * used by the full pipeline) produce identical results to kernels created via
 * createGemm() (fresh kernel creation).
 *
 * **Why this matters**: If there's a bug in kernel caching (stale weights, incorrect
 * scale factors), this will catch it. The full pipeline uses getOrCreateGemm().
 */
TEST_F(Test__CUDAGemmParity, CachedKernel_vs_FreshKernel)
{
    if (!std::filesystem::exists(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Model file not found: " << REAL_MODEL_PATH;
    }

    // Clear any existing cached kernels
    llaminar::v2::kernels::KernelFactory::clearCache();

    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    TensorFactory factory(*mpi_ctx);
    ModelLoader loader(&factory);

    if (!loader.loadModel(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Failed to load model: " << REAL_MODEL_PATH;
    }

    auto weights = loader.loadTensor("blk.0.attn_q.weight", DeviceId::cpu());
    ASSERT_NE(weights, nullptr);

    auto *q4_tensor = dynamic_cast<Q4_0Tensor *>(weights.get());
    ASSERT_NE(q4_tensor, nullptr);

    const int M = 4;
    const int N = static_cast<int>(q4_tensor->shape()[0]);
    const int K = static_cast<int>(q4_tensor->shape()[1]);

    std::cout << "Testing cached kernel parity: attn_q " << N << "x" << K << "\n";

    // Ensure on GPU
    ASSERT_TRUE(q4_tensor->ensureOnDevice(gpu_device_));

    // Create fresh kernel (not cached)
    auto fresh_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        q4_tensor, KernelDeviceType::CUDA);
    ASSERT_NE(fresh_kernel, nullptr);

    // Get cached kernel (this is what the pipeline uses)
    auto *cached_kernel = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(
        q4_tensor);
    ASSERT_NE(cached_kernel, nullptr);

    // Set up SHARED workspace for both kernels
    ASSERT_TRUE(setupSharedWorkspace(
        {fresh_kernel.get(), cached_kernel},
        M, {N, N}, K));

    // Create input and outputs
    auto input_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    auto output_fresh = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});
    auto output_cached = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});

    // Fill input
    float *input_data = input_tensor->mutable_data();
    for (int i = 0; i < M * K; ++i)
    {
        input_data[i] = dist_(rng_);
    }

    // Run fresh kernel with coherence wrapper
    ASSERT_TRUE(with_gpu_coherence(
        gpu_device_,
        {input_tensor.get()},
        {output_fresh.get()},
        [&]
        {
            return fresh_kernel->multiply_tensor(
                input_tensor.get(), output_fresh.get(),
                M, N, K, true, 1.0f, 0.0f, nullptr, nullptr, -1);
        }));

    // Run cached kernel with coherence wrapper
    ASSERT_TRUE(with_gpu_coherence(
        gpu_device_,
        {input_tensor.get()},
        {output_cached.get()},
        [&]
        {
            return cached_kernel->multiply_tensor(
                input_tensor.get(), output_cached.get(),
                M, N, K, true, 1.0f, 0.0f, nullptr, nullptr, -1);
        }));

    // Compare: should be EXACTLY the same (same kernel, same weights)
    const float *fresh_data = output_fresh->data();
    const float *cached_data = output_cached->data();

    auto result = checkParity(cached_data, fresh_data, M * N, 0.9999, 0.001);
    result.print("Cached kernel vs Fresh kernel");

    EXPECT_GE(result.cosine_similarity, 0.9999)
        << "Cached and fresh kernels should produce nearly identical results";
    EXPECT_FALSE(result.has_nan_inf);

    // Clean up shared workspace
    cleanupSharedWorkspace({fresh_kernel.get(), cached_kernel});

    // Clear cache for next test
    llaminar::v2::kernels::KernelFactory::clearCache();
}

/**
 * @test Cached kernel reuse: multiple calls with same kernel
 *
 * Verifies that cached kernels produce consistent results across multiple calls.
 * This catches issues like stale GPU state or incorrect memory management.
 */
TEST_F(Test__CUDAGemmParity, CachedKernel_MultipleCallsConsistent)
{
    if (!std::filesystem::exists(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Model file not found: " << REAL_MODEL_PATH;
    }

    llaminar::v2::kernels::KernelFactory::clearCache();

    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    TensorFactory factory(*mpi_ctx);
    ModelLoader loader(&factory);

    if (!loader.loadModel(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Failed to load model: " << REAL_MODEL_PATH;
    }

    auto weights = loader.loadTensor("blk.0.attn_q.weight", DeviceId::cpu());
    ASSERT_NE(weights, nullptr);

    auto *q4_tensor = dynamic_cast<Q4_0Tensor *>(weights.get());
    ASSERT_NE(q4_tensor, nullptr);

    const int M = 1; // Decode size
    const int N = static_cast<int>(q4_tensor->shape()[0]);
    const int K = static_cast<int>(q4_tensor->shape()[1]);

    ASSERT_TRUE(q4_tensor->ensureOnDevice(gpu_device_));

    // Get cached kernel
    auto *kernel = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(q4_tensor);
    ASSERT_NE(kernel, nullptr);

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(kernel, M, N, K));

    // Create input
    auto input = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    float *input_data = input->mutable_data();
    for (int i = 0; i < M * K; ++i)
    {
        input_data[i] = dist_(rng_);
    }

    // Run 3 times with same input, compare all outputs
    std::vector<std::vector<float>> outputs(3);
    for (int run = 0; run < 3; ++run)
    {
        auto output = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});

        // Use with_gpu_coherence for clean coherence handling
        ASSERT_TRUE(with_gpu_coherence(
            gpu_device_,
            {input.get()},
            {output.get()},
            [&]
            {
                return kernel->multiply_tensor(
                    input.get(), output.get(),
                    M, N, K, true, 1.0f, 0.0f, nullptr, nullptr, -1);
            }));

        outputs[run].resize(M * N);
        const float *data = output->data();
        std::copy(data, data + M * N, outputs[run].begin());
    }

    // All runs should be identical
    auto result_1_vs_2 = checkParity(outputs[0].data(), outputs[1].data(), M * N, 0.99999, 0.0001);
    auto result_1_vs_3 = checkParity(outputs[0].data(), outputs[2].data(), M * N, 0.99999, 0.0001);

    std::cout << "Run 1 vs Run 2: cosine=" << result_1_vs_2.cosine_similarity << "\n";
    std::cout << "Run 1 vs Run 3: cosine=" << result_1_vs_3.cosine_similarity << "\n";

    EXPECT_GE(result_1_vs_2.cosine_similarity, 0.99999)
        << "Multiple runs with same input should be identical";
    EXPECT_GE(result_1_vs_3.cosine_similarity, 0.99999)
        << "Multiple runs with same input should be identical";

    // Clean up workspace
    cleanupWorkspaceIfNeeded(kernel);

    llaminar::v2::kernels::KernelFactory::clearCache();
}

/**
 * @test Cached kernel across different input sizes
 *
 * The full pipeline may use the same cached kernel for both prefill (large M)
 * and decode (M=1). This test verifies the cached kernel works correctly
 * across different input sizes.
 */
TEST_F(Test__CUDAGemmParity, CachedKernel_VaryingBatchSizes)
{
    if (!std::filesystem::exists(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Model file not found: " << REAL_MODEL_PATH;
    }

    llaminar::v2::kernels::KernelFactory::clearCache();

    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    TensorFactory factory(*mpi_ctx);
    ModelLoader loader(&factory);

    if (!loader.loadModel(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Failed to load model: " << REAL_MODEL_PATH;
    }

    auto weights = loader.loadTensor("blk.0.attn_q.weight", DeviceId::cpu());
    ASSERT_NE(weights, nullptr);

    auto *q4_tensor = dynamic_cast<Q4_0Tensor *>(weights.get());
    ASSERT_NE(q4_tensor, nullptr);

    const int N = static_cast<int>(q4_tensor->shape()[0]);
    const int K = static_cast<int>(q4_tensor->shape()[1]);

    ASSERT_TRUE(q4_tensor->ensureOnDevice(gpu_device_));

    // Get ONE cached kernel
    auto *kernel = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(q4_tensor);
    ASSERT_NE(kernel, nullptr);

    // Set up workspace for quantized kernel (with largest batch size = 128)
    ASSERT_TRUE(setupWorkspaceIfNeeded(kernel, 128, N, K));

    // Also create a CPU kernel for reference
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        q4_tensor, KernelDeviceType::CPU);
    ASSERT_NE(cpu_kernel, nullptr);

    // Test multiple batch sizes
    std::vector<int> batch_sizes = {1, 4, 16, 64, 128};

    for (int M : batch_sizes)
    {
        std::cout << "Testing M=" << M << "...\n";

        // Create input with fresh random data
        auto input = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
        float *input_data = input->mutable_data();
        for (int i = 0; i < M * K; ++i)
        {
            input_data[i] = dist_(rng_);
        }

        // CUDA output
        auto output_cuda = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});

        // Use with_gpu_coherence for clean coherence handling
        ASSERT_TRUE(with_gpu_coherence(
            gpu_device_,
            {input.get()},
            {output_cuda.get()},
            [&]
            {
                return kernel->multiply_tensor(
                    input.get(), output_cuda.get(),
                    M, N, K, true, 1.0f, 0.0f, nullptr, nullptr, -1);
            }));

        // CPU reference
        std::vector<float> cpu_output(M * N);
        ASSERT_TRUE(cpu_kernel->multiply(input->data(), cpu_output.data(), M, N, K));

        // Compare
        const float *cuda_data = output_cuda->data();
        auto result = checkParity(cuda_data, cpu_output.data(), M * N, 0.99, 0.10);
        result.print(("Cached kernel M=" + std::to_string(M)).c_str());

        EXPECT_GE(result.cosine_similarity, 0.99)
            << "Cached kernel should work for M=" << M;
        EXPECT_FALSE(result.has_nan_inf);
    }

    // Clean up workspace
    cleanupWorkspaceIfNeeded(kernel);

    llaminar::v2::kernels::KernelFactory::clearCache();
}

// ============================================================================
// Bias Tests (if model has biases)
// ============================================================================

/**
 * @test QKV projection with biases from GGUF
 *
 * Qwen models have biases for Q, K, V projections. This test verifies that
 * CUDA GEMM + bias produces correct results compared to CPU.
 */
TEST_F(Test__CUDAGemmParity, FusedQKV_WithBias)
{
    if (!std::filesystem::exists(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Model file not found: " << REAL_MODEL_PATH;
    }

    llaminar::v2::kernels::KernelFactory::clearCache();

    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    TensorFactory factory(*mpi_ctx);
    ModelLoader loader(&factory);

    if (!loader.loadModel(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Failed to load model: " << REAL_MODEL_PATH;
    }

    // Load weights
    auto weights_q = loader.loadTensor("blk.0.attn_q.weight", DeviceId::cpu());
    auto weights_k = loader.loadTensor("blk.0.attn_k.weight", DeviceId::cpu());
    auto weights_v = loader.loadTensor("blk.0.attn_v.weight", DeviceId::cpu());
    ASSERT_NE(weights_q, nullptr);
    ASSERT_NE(weights_k, nullptr);
    ASSERT_NE(weights_v, nullptr);

    // Try to load biases (may not exist in all models)
    auto bias_q = loader.loadTensor("blk.0.attn_q.bias", DeviceId::cpu());
    auto bias_k = loader.loadTensor("blk.0.attn_k.bias", DeviceId::cpu());
    auto bias_v = loader.loadTensor("blk.0.attn_v.bias", DeviceId::cpu());

    if (!bias_q && !bias_k && !bias_v)
    {
        GTEST_SKIP() << "Model has no QKV biases";
    }

    std::cout << "Biases found: Q=" << (bias_q ? "yes" : "no")
              << " K=" << (bias_k ? "yes" : "no")
              << " V=" << (bias_v ? "yes" : "no") << "\n";

    // Cast weights to Q4_0
    auto *wq = dynamic_cast<Q4_0Tensor *>(weights_q.get());
    auto *wk = dynamic_cast<Q4_0Tensor *>(weights_k.get());
    auto *wv = dynamic_cast<Q4_0Tensor *>(weights_v.get());
    ASSERT_NE(wq, nullptr);
    ASSERT_NE(wk, nullptr);
    ASSERT_NE(wv, nullptr);

    const int M = 1; // Decode
    const int N_q = static_cast<int>(wq->shape()[0]);
    const int N_k = static_cast<int>(wk->shape()[0]);
    const int N_v = static_cast<int>(wv->shape()[0]);
    const int K = static_cast<int>(wq->shape()[1]);

    std::cout << "FusedQKV with bias: M=" << M << " K=" << K
              << " N_q=" << N_q << " N_k=" << N_k << " N_v=" << N_v << "\n";

    // Upload weights to GPU
    ASSERT_TRUE(wq->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(wk->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(wv->ensureOnDevice(gpu_device_));

    // Upload biases to GPU if present
    if (bias_q)
        ASSERT_TRUE(bias_q->ensureOnDevice(gpu_device_));
    if (bias_k)
        ASSERT_TRUE(bias_k->ensureOnDevice(gpu_device_));
    if (bias_v)
        ASSERT_TRUE(bias_v->ensureOnDevice(gpu_device_));

    // Create kernels
    auto cuda_kernel_q = llaminar::v2::kernels::KernelFactory::createGemm(
        wq, KernelDeviceType::CUDA);
    auto cuda_kernel_k = llaminar::v2::kernels::KernelFactory::createGemm(
        wk, KernelDeviceType::CUDA);
    auto cuda_kernel_v = llaminar::v2::kernels::KernelFactory::createGemm(
        wv, KernelDeviceType::CUDA);

    // Set up SHARED workspace for fused QKV (all kernels share workspace)
    ASSERT_TRUE(setupSharedWorkspace(
        {cuda_kernel_q.get(), cuda_kernel_k.get(), cuda_kernel_v.get()},
        M, {N_q, N_k, N_v}, K));

    // Create input
    auto input = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    float *input_data = input->mutable_data();
    for (int i = 0; i < M * K; ++i)
    {
        input_data[i] = dist_(rng_);
    }
    ASSERT_TRUE(input->ensureOnDevice(gpu_device_));

    // Create outputs
    auto output_q = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_q)});
    auto output_k = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_k)});
    auto output_v = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_v)});
    ASSERT_TRUE(output_q->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(output_k->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(output_v->ensureOnDevice(gpu_device_));

    // Run fused GEMM with biases
    std::vector<TensorProjectionDesc> projections;
    projections.emplace_back(cuda_kernel_q.get(), output_q.get(), N_q,
                             bias_q.get(), nullptr, false, "Q");
    projections.emplace_back(cuda_kernel_k.get(), output_k.get(), N_k,
                             bias_k.get(), nullptr, false, "K");
    projections.emplace_back(cuda_kernel_v.get(), output_v.get(), N_v,
                             bias_v.get(), nullptr, false, "V");

    ASSERT_TRUE(cuda_kernel_q->multiply_fused_tensor(
        input.get(), projections, M, K, nullptr));

    // Mark outputs as device-dirty (tests bypass GraphExecutor auto-coherence)
    output_q->mark_device_dirty();
    output_k->mark_device_dirty();
    output_v->mark_device_dirty();

    // ===== CPU reference (GEMM + manual bias add) =====
    auto cpu_kernel_q = llaminar::v2::kernels::KernelFactory::createGemm(
        wq, KernelDeviceType::CPU);
    auto cpu_kernel_k = llaminar::v2::kernels::KernelFactory::createGemm(
        wk, KernelDeviceType::CPU);
    auto cpu_kernel_v = llaminar::v2::kernels::KernelFactory::createGemm(
        wv, KernelDeviceType::CPU);

    std::vector<float> q_cpu(M * N_q), k_cpu(M * N_k), v_cpu(M * N_v);
    const float *h_input = input->data();
    ASSERT_TRUE(cpu_kernel_q->multiply(h_input, q_cpu.data(), M, N_q, K));
    ASSERT_TRUE(cpu_kernel_k->multiply(h_input, k_cpu.data(), M, N_k, K));
    ASSERT_TRUE(cpu_kernel_v->multiply(h_input, v_cpu.data(), M, N_v, K));

    // Add biases (CPU side)
    if (bias_q)
    {
        const float *bq = static_cast<const FP32Tensor *>(bias_q.get())->data();
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N_q; ++j)
            {
                q_cpu[i * N_q + j] += bq[j];
            }
        }
    }
    if (bias_k)
    {
        const float *bk = static_cast<const FP32Tensor *>(bias_k.get())->data();
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N_k; ++j)
            {
                k_cpu[i * N_k + j] += bk[j];
            }
        }
    }
    if (bias_v)
    {
        const float *bv = static_cast<const FP32Tensor *>(bias_v.get())->data();
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N_v; ++j)
            {
                v_cpu[i * N_v + j] += bv[j];
            }
        }
    }

    // Compare
    const float *q_cuda = output_q->data();
    const float *k_cuda = output_k->data();
    const float *v_cuda = output_v->data();

    auto result_q = checkParity(q_cuda, q_cpu.data(), M * N_q, 0.99, 0.10);
    result_q.print("Q with bias (CUDA vs CPU)");
    EXPECT_GE(result_q.cosine_similarity, 0.99);
    EXPECT_FALSE(result_q.has_nan_inf);

    auto result_k = checkParity(k_cuda, k_cpu.data(), M * N_k, 0.99, 0.10);
    result_k.print("K with bias (CUDA vs CPU)");
    EXPECT_GE(result_k.cosine_similarity, 0.99);
    EXPECT_FALSE(result_k.has_nan_inf);

    auto result_v = checkParity(v_cuda, v_cpu.data(), M * N_v, 0.99, 0.10);
    result_v.print("V with bias (CUDA vs CPU)");
    EXPECT_GE(result_v.cosine_similarity, 0.99);
    EXPECT_FALSE(result_v.has_nan_inf);

    // Clean up shared workspace
    cleanupSharedWorkspace({cuda_kernel_q.get(), cuda_kernel_k.get(), cuda_kernel_v.get()});

    llaminar::v2::kernels::KernelFactory::clearCache();
}

/**
 * @test Fused QKV using cached kernels (simulating full pipeline)
 *
 * This test mimics the full pipeline's kernel usage:
 * 1. Uses getOrCreateGemm() to get cached kernels
 * 2. Uses real model weights and biases
 * 3. Runs multiple iterations like decode
 */
TEST_F(Test__CUDAGemmParity, FusedQKV_CachedKernels_MultipleIterations)
{
    if (!std::filesystem::exists(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Model file not found: " << REAL_MODEL_PATH;
    }

    llaminar::v2::kernels::KernelFactory::clearCache();

    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    TensorFactory factory(*mpi_ctx);
    ModelLoader loader(&factory);

    if (!loader.loadModel(REAL_MODEL_PATH))
    {
        GTEST_SKIP() << "Failed to load model: " << REAL_MODEL_PATH;
    }

    // Load weights
    auto weights_q = loader.loadTensor("blk.0.attn_q.weight", DeviceId::cpu());
    auto weights_k = loader.loadTensor("blk.0.attn_k.weight", DeviceId::cpu());
    auto weights_v = loader.loadTensor("blk.0.attn_v.weight", DeviceId::cpu());
    ASSERT_NE(weights_q, nullptr);
    ASSERT_NE(weights_k, nullptr);
    ASSERT_NE(weights_v, nullptr);

    auto *wq = dynamic_cast<Q4_0Tensor *>(weights_q.get());
    auto *wk = dynamic_cast<Q4_0Tensor *>(weights_k.get());
    auto *wv = dynamic_cast<Q4_0Tensor *>(weights_v.get());
    ASSERT_NE(wq, nullptr);
    ASSERT_NE(wk, nullptr);
    ASSERT_NE(wv, nullptr);

    const int M = 1;
    const int N_q = static_cast<int>(wq->shape()[0]);
    const int N_k = static_cast<int>(wk->shape()[0]);
    const int N_v = static_cast<int>(wv->shape()[0]);
    const int K = static_cast<int>(wq->shape()[1]);

    // Upload to GPU
    ASSERT_TRUE(wq->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(wk->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(wv->ensureOnDevice(gpu_device_));

    // Get CACHED kernels (this is what the pipeline does)
    auto *kernel_q = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(wq);
    auto *kernel_k = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(wk);
    auto *kernel_v = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(wv);
    ASSERT_NE(kernel_q, nullptr);
    ASSERT_NE(kernel_k, nullptr);
    ASSERT_NE(kernel_v, nullptr);

    // Set up SHARED workspace for fused QKV (all kernels share workspace)
    ASSERT_TRUE(setupSharedWorkspace(
        {kernel_q, kernel_k, kernel_v},
        M, {N_q, N_k, N_v}, K));

    // CPU kernels for reference
    auto cpu_kernel_q = llaminar::v2::kernels::KernelFactory::createGemm(
        wq, KernelDeviceType::CPU);
    auto cpu_kernel_k = llaminar::v2::kernels::KernelFactory::createGemm(
        wk, KernelDeviceType::CPU);
    auto cpu_kernel_v = llaminar::v2::kernels::KernelFactory::createGemm(
        wv, KernelDeviceType::CPU);

    std::cout << "Testing cached kernels over 5 iterations (simulating decode)...\n";

    // Run 5 iterations with different inputs
    for (int iter = 0; iter < 5; ++iter)
    {
        // Create fresh input for each iteration
        auto input = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
        float *input_data = input->mutable_data();
        for (int i = 0; i < M * K; ++i)
        {
            input_data[i] = dist_(rng_);
        }
        ASSERT_TRUE(input->ensureOnDevice(gpu_device_));

        // Create outputs
        auto out_q = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_q)});
        auto out_k = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_k)});
        auto out_v = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_v)});
        ASSERT_TRUE(out_q->ensureOnDevice(gpu_device_));
        ASSERT_TRUE(out_k->ensureOnDevice(gpu_device_));
        ASSERT_TRUE(out_v->ensureOnDevice(gpu_device_));

        // Run fused with cached kernels
        std::vector<TensorProjectionDesc> projections;
        projections.emplace_back(kernel_q, out_q.get(), N_q,
                                 nullptr, nullptr, false, "Q");
        projections.emplace_back(kernel_k, out_k.get(), N_k,
                                 nullptr, nullptr, false, "K");
        projections.emplace_back(kernel_v, out_v.get(), N_v,
                                 nullptr, nullptr, false, "V");

        ASSERT_TRUE(kernel_q->multiply_fused_tensor(
            input.get(), projections, M, K, nullptr));

        // Mark outputs as device-dirty (tests bypass GraphExecutor auto-coherence)
        out_q->mark_device_dirty();
        out_k->mark_device_dirty();
        out_v->mark_device_dirty();

        // CPU reference
        std::vector<float> q_cpu(M * N_q), k_cpu(M * N_k), v_cpu(M * N_v);
        const float *h_input = input->data();
        ASSERT_TRUE(cpu_kernel_q->multiply(h_input, q_cpu.data(), M, N_q, K));
        ASSERT_TRUE(cpu_kernel_k->multiply(h_input, k_cpu.data(), M, N_k, K));
        ASSERT_TRUE(cpu_kernel_v->multiply(h_input, v_cpu.data(), M, N_v, K));

        // Compare
        auto result_q = checkParity(out_q->data(), q_cpu.data(), M * N_q, 0.99, 0.10);
        auto result_k = checkParity(out_k->data(), k_cpu.data(), M * N_k, 0.99, 0.10);
        auto result_v = checkParity(out_v->data(), v_cpu.data(), M * N_v, 0.99, 0.10);

        std::cout << "  Iter " << iter << ": Q=" << result_q.cosine_similarity
                  << " K=" << result_k.cosine_similarity
                  << " V=" << result_v.cosine_similarity << "\n";

        EXPECT_GE(result_q.cosine_similarity, 0.99)
            << "Q failed at iteration " << iter;
        EXPECT_GE(result_k.cosine_similarity, 0.99)
            << "K failed at iteration " << iter;
        EXPECT_GE(result_v.cosine_similarity, 0.99)
            << "V failed at iteration " << iter;
        EXPECT_FALSE(result_q.has_nan_inf);
        EXPECT_FALSE(result_k.has_nan_inf);
        EXPECT_FALSE(result_v.has_nan_inf);
    }

    // Clean up shared workspace
    cleanupSharedWorkspace({kernel_q, kernel_k, kernel_v});

    llaminar::v2::kernels::KernelFactory::clearCache();
}

#endif // HAVE_CUDA

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
