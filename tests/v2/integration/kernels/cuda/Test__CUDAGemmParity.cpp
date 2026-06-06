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
#include "kernels/cuda/gemm/CUDAWeightPacker.h"
#include "backends/ComputeBackend.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/coherence/GpuCoherence.h"        // For gpu_output(), with_gpu_coherence()
#include "execution/local_execution/device/DeviceWorkspaceManager.h" // For workspace binding
#include "execution/compute_stages/stages/GDNProjectionStage.h"
#include "loaders/ModelLoader.h"
#include "tensors/TensorFactory.h"
#include "utils/PerfStatsCollector.h"
#include "utils/MPIContext.h"
#include "../../../utils/TestModelHelper.h"
#ifdef HAVE_CUDA
#include "backends/cuda/CUDABackend.h"
#include <cuda_runtime.h>
#endif

// Now include test utils
#include "../../../utils/CUDATestUtils.h"
#include "../../../utils/TestTensorFactory.h"
#include "../../../utils/GpuPreparedGemmHarness.h"

#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <functional>
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

#ifdef HAVE_CUDA
extern "C"
{
    void cudaNativeVNNIPrefill_setStreamKMode(int mode);
    int cudaNativeVNNIPrefill_getStreamKMode();
    void cudaNativeVNNIPrefill_setBK256Mode(int mode);
    int cudaNativeVNNIPrefill_getBK256Mode();
    void cudaNativeVNNIPrefill_setDeterministicMode(bool enabled);
    bool cudaNativeVNNIPrefill_getDeterministicMode();
    void cudaNativeVNNIPrefill_setForceTile(int tile_id, int split_k);
    void cudaNativeVNNIPrefill_getForceTile(int *tile_id, int *split_k);
    void cudaNativeVNNIPrefill_getLastLaunchSelection(
        int *tile_id,
        int *split_k,
        int *used_bk256,
        int *used_streamk);
}
#endif

namespace
{
    class ScopedEnv
    {
    public:
        ScopedEnv(const char *name, const char *value)
            : name_(name)
        {
            const char *old = std::getenv(name);
            if (old)
            {
                had_old_ = true;
                old_value_ = old;
            }
            setenv(name, value, 1);
        }

        ~ScopedEnv()
        {
            if (had_old_)
                setenv(name_.c_str(), old_value_.c_str(), 1);
            else
                unsetenv(name_.c_str());
        }

        ScopedEnv(const ScopedEnv &) = delete;
        ScopedEnv &operator=(const ScopedEnv &) = delete;

    private:
        std::string name_;
        std::string old_value_;
        bool had_old_ = false;
    };

#ifdef HAVE_CUDA
    class ScopedCudaPrefillModes
    {
    public:
        ScopedCudaPrefillModes()
            : streamk_(cudaNativeVNNIPrefill_getStreamKMode()),
              bk256_(cudaNativeVNNIPrefill_getBK256Mode()),
              deterministic_(cudaNativeVNNIPrefill_getDeterministicMode())
        {
            cudaNativeVNNIPrefill_getForceTile(&force_tile_, &force_split_k_);
        }

        ~ScopedCudaPrefillModes()
        {
            cudaNativeVNNIPrefill_setForceTile(force_tile_, force_split_k_);
            cudaNativeVNNIPrefill_setStreamKMode(streamk_);
            cudaNativeVNNIPrefill_setBK256Mode(bk256_);
            cudaNativeVNNIPrefill_setDeterministicMode(deterministic_);
        }

        ScopedCudaPrefillModes(const ScopedCudaPrefillModes &) = delete;
        ScopedCudaPrefillModes &operator=(const ScopedCudaPrefillModes &) = delete;

    private:
        int force_tile_ = -1;
        int force_split_k_ = 0;
        int streamk_ = 0;
        int bk256_ = 0;
        bool deterministic_ = false;
    };
#endif

    struct CUDASmallMFormatSpec
    {
        const char *name;
        double cosine_threshold;
        std::function<std::unique_ptr<TensorBase>(size_t, size_t)> create;
    };

    const std::vector<CUDASmallMFormatSpec> &cudaSmallMNativeFormats()
    {
        static const std::vector<CUDASmallMFormatSpec> formats = {
            {"Q4_0", 0.990, [](size_t n, size_t k) { return TestTensorFactory::createQ4_0Random({n, k}); }},
            {"IQ4_NL", 0.985, [](size_t n, size_t k) { return TestTensorFactory::createIQ4_NLRandom({n, k}); }},
            {"Q4_1", 0.990, [](size_t n, size_t k) { return TestTensorFactory::createQ4_1Random({n, k}); }},
            {"IQ4_XS", 0.985, [](size_t n, size_t k) { return TestTensorFactory::createIQ4_XSRandom({n, k}); }},
            {"Q5_0", 0.990, [](size_t n, size_t k) { return TestTensorFactory::createQ5_0Random({n, k}); }},
            {"Q5_1", 0.990, [](size_t n, size_t k) { return TestTensorFactory::createQ5_1Random({n, k}); }},
            {"Q4_K", 0.990, [](size_t n, size_t k) { return TestTensorFactory::createQ4_KRandom({n, k}); }},
            {"Q5_K", 0.990, [](size_t n, size_t k) { return TestTensorFactory::createQ5_KRandom({n, k}); }},
            {"Q6_K", 0.990, [](size_t n, size_t k) { return TestTensorFactory::createQ6_KRandom({n, k}); }},
            {"Q3_K", 0.980, [](size_t n, size_t k) { return TestTensorFactory::createQ3_KRandom({n, k}); }},
            {"Q2_K", 0.960, [](size_t n, size_t k) { return TestTensorFactory::createQ2_KRandom({n, k}); }},
            {"IQ3_S", 0.970, [](size_t n, size_t k) { return TestTensorFactory::createIQ3_SRandom({n, k}); }},
            {"IQ3_XXS", 0.960, [](size_t n, size_t k) { return TestTensorFactory::createIQ3_XXSRandom({n, k}); }},
            {"IQ2_S", 0.920, [](size_t n, size_t k) { return TestTensorFactory::createIQ2_SRandom({n, k}); }},
            {"IQ2_XS", 0.900, [](size_t n, size_t k) { return TestTensorFactory::createIQ2_XSRandom({n, k}); }},
            {"IQ2_XXS", 0.880, [](size_t n, size_t k) { return TestTensorFactory::createIQ2_XXSRandom({n, k}); }},
            {"IQ1_S", 0.800, [](size_t n, size_t k) { return TestTensorFactory::createIQ1_SRandom({n, k}); }},
            {"IQ1_M", 0.800, [](size_t n, size_t k) { return TestTensorFactory::createIQ1_MRandom({n, k}); }},
            {"Q8_0", 0.999, [](size_t n, size_t k) { return TestTensorFactory::createQ8_0Random({n, k}); }},
        };
        return formats;
    }


    ITensorGemm *getPreparedKernel(const TensorBase *tensor, DeviceId device_id)
    {
        // GPU INT8-packed (quantized native-VNNI) weights cannot go through
        // KernelFactory::prepareGemmHandleLocal() — that path is guarded and throws
        // because such kernels must be built from VRAM-resident, repacked payloads
        // owned by a WeightVRAMPool. Route those through the shared production-pipeline
        // helper instead, and keep the returned lifetime owners alive in a static list.
        auto *unpackable = dynamic_cast<const IINT8Unpackable *>(tensor);
        const bool is_gpu_quantized =
            (device_id.is_cuda() || device_id.is_rocm()) &&
            unpackable != nullptr &&
            const_cast<IINT8Unpackable *>(unpackable)->vnniFormatInfo() != nullptr;

        if (is_gpu_quantized)
        {
            static std::vector<llaminar2::test::GpuPreparedGemm> gpu_prepared;
            const uint64_t prepared_id = static_cast<uint64_t>(gpu_prepared.size());
            gpu_prepared.push_back(
                llaminar2::test::makeGpuPreparedGemm(
                    const_cast<TensorBase *>(tensor),
                    device_id,
                    "test.gpu_prepared_gemm.weight." + std::to_string(prepared_id),
                    ModelContextId{9900 + prepared_id}));
            return gpu_prepared.back().kernel;
        }

        static std::vector<std::shared_ptr<llaminar::v2::kernels::KernelFactory::PreparedGemmHandle>> handles;
        auto prepared = llaminar::v2::kernels::KernelFactory::prepareGemmHandleLocal(tensor, device_id);
        if (!prepared)
        {
            return nullptr;
        }
        handles.push_back(std::move(prepared));
        return llaminar::v2::kernels::KernelFactory::getOrCreateGemmEngine(handles.back().get());
    }

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

    // =========================================================================
    // Helpers: multiply via tensor interface (multiply() removed from ITensorGemm)
    // =========================================================================

    /**
     * @brief CPU multiply via tensor interface — wraps raw float* in FP32Tensors.
     */
    bool cpuMultiplyToVector(ITensorGemm *kernel, const float *A_data,
                             float *C_data, int M, int N, int K,
                             bool transpose_B = true, float alpha = 1.0f, float beta = 0.0f)
    {
        auto A_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)K});
        std::memcpy(A_tensor->mutable_data(), A_data, (size_t)M * K * sizeof(float));
        auto C_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)N});
        if (beta != 0.0f)
            std::memcpy(C_tensor->mutable_data(), C_data, (size_t)M * N * sizeof(float));
        bool ok = kernel->multiply_tensor(A_tensor.get(), C_tensor.get(), M, N, K, transpose_B, alpha, beta);
        if (ok)
            std::memcpy(C_data, C_tensor->data(), (size_t)M * N * sizeof(float));
        return ok;
    }

    /**
     * @brief CUDA multiply via tensor coherence — creates FP32Tensors, uploads, runs, downloads.
     */
    bool cudaMultiplyViaTensor(ITensorGemm *kernel,
                               const float *A_host, float *C_host,
                               int M, int N, int K, DeviceId gpu_device,
                               bool transpose_B = true, float alpha = 1.0f, float beta = 0.0f)
    {
        auto A_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)K});
        std::memcpy(A_tensor->mutable_data(), A_host, (size_t)M * K * sizeof(float));
        auto C_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)N});
        if (beta != 0.0f)
            std::memcpy(C_tensor->mutable_data(), C_host, (size_t)M * N * sizeof(float));
        bool ok = with_gpu_coherence(gpu_device, {A_tensor.get()}, {C_tensor.get()},
                                     [&]
                                     { return kernel->multiply_tensor(A_tensor.get(), C_tensor.get(), M, N, K, transpose_B, alpha, beta); });
        if (ok)
            std::memcpy(C_host, C_tensor->data(), (size_t)M * N * sizeof(float));
        return ok;
    }

    bool cpuFusedSwiGLUDownToVector(ITensorGemm *kernel,
                                    const float *gate_host,
                                    const float *up_host,
                                    float *C_host,
                                    int M, int N, int K,
                                    float alpha = 1.0f, float beta = 0.0f)
    {
        auto gate_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)K});
        auto up_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)K});
        std::memcpy(gate_tensor->mutable_data(), gate_host, (size_t)M * K * sizeof(float));
        std::memcpy(up_tensor->mutable_data(), up_host, (size_t)M * K * sizeof(float));
        auto C_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)N});
        if (beta != 0.0f)
            std::memcpy(C_tensor->mutable_data(), C_host, (size_t)M * N * sizeof(float));

        bool ok = kernel->multiply_tensor_with_fused_swiglu(
            gate_tensor.get(), up_tensor.get(), C_tensor.get(),
            M, N, K, alpha, beta);
        if (ok)
            std::memcpy(C_host, C_tensor->data(), (size_t)M * N * sizeof(float));
        return ok;
    }

    bool cudaFusedSwiGLUDownViaTensor(ITensorGemm *kernel,
                                      const float *gate_host,
                                      const float *up_host,
                                      float *C_host,
                                      int M, int N, int K,
                                      DeviceId gpu_device,
                                      float alpha = 1.0f, float beta = 0.0f)
    {
        auto gate_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)K});
        auto up_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)K});
        std::memcpy(gate_tensor->mutable_data(), gate_host, (size_t)M * K * sizeof(float));
        std::memcpy(up_tensor->mutable_data(), up_host, (size_t)M * K * sizeof(float));
        auto C_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)N});
        if (beta != 0.0f)
            std::memcpy(C_tensor->mutable_data(), C_host, (size_t)M * N * sizeof(float));

        bool ok = with_gpu_coherence(gpu_device, {gate_tensor.get(), up_tensor.get()}, {C_tensor.get()},
                                     [&]
                                     {
                                         return kernel->multiply_tensor_with_fused_swiglu(
                                             gate_tensor.get(), up_tensor.get(), C_tensor.get(),
                                             M, N, K, alpha, beta);
                                     });
        if (ok)
            std::memcpy(C_host, C_tensor->data(), (size_t)M * N * sizeof(float));
        return ok;
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
#ifdef HAVE_CUDA
            if (gpu_device_.is_cuda())
            {
                ASSERT_EQ(cudaSetDevice(gpu_device_.ordinal), cudaSuccess);
                ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess)
                    << "CUDA GEMM test cleanup must wait for workspace users before freeing buffers";
            }
#endif
            auto *ws_consumer = dynamic_cast<IWorkspaceConsumer *>(kernel);
            if (ws_consumer)
            {
                ws_consumer->unbindWorkspace();
            }
            kernel->resetDynamicState();
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
        // Merge requirements across all kernels by buffer name so fused paths
        // keep pace with the evolving CUDA workspace contract.
        WorkspaceRequirements shared_reqs;

        for (size_t i = 0; i < kernels.size(); ++i)
        {
            auto *ws_consumer = dynamic_cast<IWorkspaceConsumer *>(kernels[i]);
            if (ws_consumer)
            {
                auto reqs = ws_consumer->getWorkspaceRequirements(M, Ns[i], K);
                for (const auto &buf : reqs.buffers)
                {
                    auto it = std::find_if(
                        shared_reqs.buffers.begin(),
                        shared_reqs.buffers.end(),
                        [&](const WorkspaceDescriptor &existing)
                        {
                            return existing.name == buf.name;
                        });

                    if (it == shared_reqs.buffers.end())
                    {
                        shared_reqs.buffers.push_back(buf);
                        continue;
                    }

                    it->size_bytes = std::max(it->size_bytes, buf.size_bytes);
                    it->alignment = std::max(it->alignment, buf.alignment);
                    it->required = it->required || buf.required;
                }
            }
        }

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
#ifdef HAVE_CUDA
        if (workspace_ && gpu_device_.is_cuda())
        {
            ASSERT_EQ(cudaSetDevice(gpu_device_.ordinal), cudaSuccess);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess)
                << "CUDA GEMM shared-workspace cleanup must wait for all projection kernels before freeing buffers";
        }
#endif
        for (auto *kernel : kernels)
        {
            auto *ws_consumer = dynamic_cast<IWorkspaceConsumer *>(kernel);
            if (ws_consumer && ws_consumer->hasWorkspace())
            {
                ws_consumer->unbindWorkspace();
            }
            if (kernel)
            {
                kernel->resetDynamicState();
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
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));

    // ===== CUDA =====
    // Upload weights to GPU
    ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
    EXPECT_TRUE(weights->isOnGPU());

    // Create CUDA kernel via KernelFactory
    auto *cuda_kernel = getPreparedKernel(
        weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr) << "Failed to create CUDA kernel";

    // Execute CUDA GEMM via tensor coherence
    std::vector<float> C_cuda(M * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    // ===== Compare =====
    auto result = checkParity(C_cuda.data(), C_cpu.data(), M * N, 0.9999, 0.01);
    result.print("FP32 128x256x512");

    EXPECT_FALSE(result.has_nan_inf) << "CUDA output contains NaN/Inf";
    EXPECT_GE(result.cosine_similarity, 0.9999)
        << "Cosine similarity too low: " << result.cosine_similarity;
    EXPECT_LE(result.relative_l2_error, 0.01)
        << "Relative L2 error too high: " << (result.relative_l2_error * 100) << "%";

    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
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
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));

    // CUDA
    ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
    auto *cuda_kernel = getPreparedKernel(
        weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);

    float *d_A, *d_C;
    std::vector<float> C_cuda(M * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
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
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));

    // CUDA
    ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
    auto *cuda_kernel = getPreparedKernel(
        weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);

    float *d_A, *d_C;
    std::vector<float> C_cuda(M * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
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
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));

    // Check CPU result is valid
    ASSERT_FALSE(hasNaNOrInf(C_cpu.data(), M * N)) << "CPU result has NaN/Inf";

    // ===== CUDA =====
    // Upload weights to GPU
    ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
    EXPECT_TRUE(weights->isOnGPU());

    // Create CUDA kernel via KernelFactory
    auto *cuda_kernel = getPreparedKernel(
        weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr) << "Failed to create CUDA IQ4_NL kernel";

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    // Allocate GPU memory for activations and output
    float *d_A, *d_C;
    std::vector<float> C_cuda(M * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
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
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));

    // CUDA
    ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
    auto *cuda_kernel = getPreparedKernel(
        weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    float *d_A, *d_C;
    std::vector<float> C_cuda(M * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
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
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));

    // CUDA
    ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
    auto *cuda_kernel = getPreparedKernel(
        weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    float *d_A, *d_C;
    std::vector<float> C_cuda(M * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
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
#define DEFINE_QUANTIZED_PARITY_TEST(TestName, TensorType, CreateMethod, BlockSize, Seed)         \
    TEST_F(Test__CUDAGemmParity, TestName)                                                        \
    {                                                                                             \
        const int M = 128;                                                                        \
        const int N = 896;                                                                        \
        const int K = (BlockSize == 256) ? 768 : 896; /* K-quants need multiple of 256 */         \
                                                                                                  \
        auto weights = TestTensorFactory::CreateMethod({(size_t)N, (size_t)K}, Seed);             \
                                                                                                  \
        auto A_data = randomFP32(M * K);                                                          \
                                                                                                  \
        /* CPU Reference */                                                                       \
        std::vector<float> C_cpu(M * N, 0.0f);                                                    \
        auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(                       \
            weights.get(), KernelDeviceType::CPU);                                                \
        ASSERT_NE(cpu_kernel, nullptr) << "Failed to create CPU kernel for " #TensorType;         \
        ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K)); \
        ASSERT_FALSE(hasNaNOrInf(C_cpu.data(), M * N)) << "CPU result has NaN/Inf";               \
                                                                                                  \
        /* CUDA */                                                                                \
        ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));                                        \
        auto *cuda_kernel = getPreparedKernel(                                                    \
            weights.get(), gpu_device_);                                                          \
        ASSERT_NE(cuda_kernel, nullptr) << "Failed to create CUDA kernel for " #TensorType;       \
                                                                                                  \
        /* Set up workspace for quantized kernel */                                               \
        ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));                                \
                                                                                                  \
        std::vector<float> C_cuda(M * N, 0.0f);                                                   \
        ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(),                             \
                                          C_cuda.data(), M, N, K, gpu_device_));                  \
                                                                                                  \
        auto result = checkParity(C_cuda.data(), C_cpu.data(), M * N, 0.99, 0.15);                \
        result.print(#TensorType " 128x" + std::to_string(N) + "x" + std::to_string(K));          \
                                                                                                  \
        EXPECT_FALSE(result.has_nan_inf) << "CUDA output contains NaN/Inf";                       \
        EXPECT_GE(result.cosine_similarity, 0.99)                                                 \
            << "Cosine similarity too low: " << result.cosine_similarity;                         \
                                                                                                  \
        cleanupWorkspaceIfNeeded(cuda_kernel);                                                    \
        llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());                       \
    }

/**
 * @brief Helper macro for decode-size (M=1) quantized GEMV parity tests
 *
 * Exercises the NativeVNNI GEMV path (M=1 decode) for each codebook.
 * K-quant formats use 256-element blocks, so K must be multiple of 256.
 */
#define DEFINE_QUANTIZED_DECODE_PARITY_TEST(TestName, TensorType, CreateMethod, BlockSize, Seed)  \
    TEST_F(Test__CUDAGemmParity, TestName)                                                        \
    {                                                                                             \
        const int M = 1;                                                                          \
        const int N = 896;                                                                        \
        const int K = (BlockSize == 256) ? 768 : 896;                                             \
                                                                                                  \
        auto weights = TestTensorFactory::CreateMethod({(size_t)N, (size_t)K}, Seed);             \
                                                                                                  \
        auto A_data = randomFP32(M * K);                                                          \
                                                                                                  \
        /* CPU Reference */                                                                       \
        std::vector<float> C_cpu(M * N, 0.0f);                                                    \
        auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(                       \
            weights.get(), KernelDeviceType::CPU);                                                \
        ASSERT_NE(cpu_kernel, nullptr) << "Failed to create CPU kernel for " #TensorType;         \
        ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K)); \
        ASSERT_FALSE(hasNaNOrInf(C_cpu.data(), M * N)) << "CPU result has NaN/Inf";               \
                                                                                                  \
        /* CUDA */                                                                                \
        ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));                                        \
        auto *cuda_kernel = getPreparedKernel(                                                    \
            weights.get(), gpu_device_);                                                          \
        ASSERT_NE(cuda_kernel, nullptr) << "Failed to create CUDA kernel for " #TensorType;       \
                                                                                                  \
        ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));                                \
                                                                                                  \
        std::vector<float> C_cuda(M * N, 0.0f);                                                   \
        ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(),                             \
                                          C_cuda.data(), M, N, K, gpu_device_));                  \
                                                                                                  \
        auto result = checkParity(C_cuda.data(), C_cpu.data(), M * N, 0.99, 0.15);                \
        result.print(#TensorType " Decode 1x" + std::to_string(N) + "x" + std::to_string(K));     \
                                                                                                  \
        EXPECT_FALSE(result.has_nan_inf) << "CUDA output contains NaN/Inf";                       \
        EXPECT_GE(result.cosine_similarity, 0.99)                                                 \
            << "Cosine similarity too low: " << result.cosine_similarity;                         \
                                                                                                  \
        cleanupWorkspaceIfNeeded(cuda_kernel);                                                    \
        llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());                       \
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
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));

    ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
    auto *cuda_kernel = getPreparedKernel(
        weights.get(), gpu_device_);

    auto *ws_consumer = dynamic_cast<IWorkspaceConsumer *>(cuda_kernel);
    ASSERT_NE(ws_consumer, nullptr);
    const auto reqs = ws_consumer->getWorkspaceRequirements(M, N, K);
    const auto kpar_req = std::find_if(
        reqs.buffers.begin(),
        reqs.buffers.end(),
        [](const WorkspaceDescriptor &buffer)
        {
            return buffer.name == GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS;
        });
    ASSERT_NE(kpar_req, reqs.buffers.end())
        << "CUDA decode native-VNNI KPAR route must declare its partials workspace";
    EXPECT_TRUE(kpar_req->required);
    EXPECT_GE(kpar_req->size_bytes,
              static_cast<size_t>((K + 31) / 32) * static_cast<size_t>(N) * sizeof(float));

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    float *d_A, *d_C;
    std::vector<float> C_cuda(M * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
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
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));

    ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
    auto *cuda_kernel = getPreparedKernel(
        weights.get(), gpu_device_);

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    float *d_A, *d_C;
    std::vector<float> C_cuda(M * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

TEST_F(Test__CUDAGemmParity, Q4_0_SmallPrefillM4_UsesNativeSmallMRoute)
{
    const int M = 4;
    const int N = 896;
    const int K = 896;

    auto weights = TestTensorFactory::createQ4_0Random({(size_t)N, (size_t)K}, 231);
    auto A_data = randomFP32(static_cast<size_t>(M) * K);

    std::vector<float> C_cpu(static_cast<size_t>(M) * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_NE(cpu_kernel, nullptr);
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));
    ASSERT_FALSE(hasNaNOrInf(C_cpu.data(), C_cpu.size()));

    ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
    auto *cuda_kernel = getPreparedKernel(weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    std::vector<float> C_cuda(static_cast<size_t>(M) * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    auto result = checkParity(C_cuda.data(), C_cpu.data(), C_cpu.size(), 0.99, 0.15);
    result.print("Q4_0 small-prefill native GEMV route 4x896x896");
    EXPECT_FALSE(result.has_nan_inf)
        << "CUDA small-prefill M=4 route must not emit NaN/Inf";
    EXPECT_GE(result.cosine_similarity, 0.99)
        << "Cosine similarity too low: " << result.cosine_similarity;

    cleanupWorkspaceIfNeeded(cuda_kernel);
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

TEST_F(Test__CUDAGemmParity, Q4_0_PrefillGraphReplaySurvivesKernelDynamicReset)
{
    const int M = 512;
    const int N = 896;
    const int K = 896;

    ScopedCudaPrefillModes prefill_modes;
    cudaNativeVNNIPrefill_setBK256Mode(-1);
    cudaNativeVNNIPrefill_setDeterministicMode(false);
    cudaNativeVNNIPrefill_setForceTile(1, 1); // T64x128_w2x2, single split-K.
    cudaNativeVNNIPrefill_setStreamKMode(2);  // Force two-pass Stream-K fixup buffer path.

    auto weights = TestTensorFactory::createQ4_0Random({static_cast<size_t>(N), static_cast<size_t>(K)}, 232);
    auto A_data = randomFP32(static_cast<size_t>(M) * K);

    std::vector<float> C_cpu(static_cast<size_t>(M) * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_NE(cpu_kernel, nullptr);
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));
    ASSERT_FALSE(hasNaNOrInf(C_cpu.data(), C_cpu.size()));

    auto *cuda_kernel = getPreparedKernel(weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    FP32Tensor A_tensor({static_cast<size_t>(M), static_cast<size_t>(K)});
    std::memcpy(A_tensor.mutable_data(), A_data.data(), A_data.size() * sizeof(float));
    FP32Tensor C_tensor({static_cast<size_t>(M), static_cast<size_t>(N)});

    ASSERT_EQ(cudaSetDevice(gpu_device_.ordinal), cudaSuccess);
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    ASSERT_TRUE(A_tensor.ensureOnDevice(gpu_device_, stream));
    ASSERT_TRUE(C_tensor.ensureOnDevice(gpu_device_, stream));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    cuda_kernel->setGPUStream(stream);
    ASSERT_TRUE(cuda_kernel->multiply_tensor(
        &A_tensor, &C_tensor, M, N, K, true, 1.0f, 0.0f, nullptr, nullptr,
        gpu_device_.ordinal, workspace_.get()));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    int tile_id = -1;
    int split_k = 0;
    int used_bk256 = 0;
    int used_streamk = 0;
    cudaNativeVNNIPrefill_getLastLaunchSelection(&tile_id, &split_k, &used_bk256, &used_streamk);
    ASSERT_EQ(used_bk256, 0) << "Regression must exercise BK64 Stream-K, not BK256";
    ASSERT_EQ(used_streamk, 2) << "Regression must exercise two-pass Stream-K context scratch";

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;

    ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed), cudaSuccess);
    ASSERT_TRUE(cuda_kernel->multiply_tensor(
        &A_tensor, &C_tensor, M, N, K, true, 1.0f, 0.0f, nullptr, nullptr,
        gpu_device_.ordinal, workspace_.get()));
    ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0), cudaSuccess);
    ASSERT_NE(graph_exec, nullptr);

    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    cuda_kernel->resetDynamicState();
    EXPECT_FALSE(cuda_kernel->hasDynamicStateActive())
        << "resetDynamicState should clear request-scoped stream binding";

    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess)
        << "Captured graph must survive request reset without dangling context scratch";
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<float> C_cuda(static_cast<size_t>(M) * N, 0.0f);
    ASSERT_EQ(cudaMemcpyAsync(
                  C_cuda.data(),
                  C_tensor.gpu_data_ptr(),
                  C_cuda.size() * sizeof(float),
                  cudaMemcpyDeviceToHost,
                  stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    auto result = checkParity(C_cuda.data(), C_cpu.data(), C_cpu.size(), 0.99, 0.15);
    result.print("Q4_0 prefill graph replay after CUDA GEMM dynamic reset");
    EXPECT_FALSE(result.has_nan_inf);
    EXPECT_GE(result.cosine_similarity, 0.99)
        << "Cosine similarity too low after graph replay: " << result.cosine_similarity;

    if (graph_exec)
        cudaGraphExecDestroy(graph_exec);
    if (graph)
        cudaGraphDestroy(graph);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

    cleanupWorkspaceIfNeeded(cuda_kernel);
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

TEST_F(Test__CUDAGemmParity, Q4_1_DecodeSize_1x896x896)
{
    const int M = 1;
    const int N = 896;
    const int K = 896;

    auto weights = TestTensorFactory::createQ4_1Random({(size_t)N, (size_t)K}, 122);
    auto A_data = randomFP32(M * K);

    std::vector<float> C_cpu(M * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_NE(cpu_kernel, nullptr);
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));

    ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
    auto *cuda_kernel = getPreparedKernel(
        weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);

    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    float *d_A = nullptr;
    float *d_C = nullptr;
    std::vector<float> C_cuda(M * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

// ============================================================================
// Q4_1 Parity Tests
// ============================================================================

DEFINE_QUANTIZED_PARITY_TEST(Q4_1_SmallMatrix, Q4_1Tensor, createQ4_1Random, 32, 103)

// ============================================================================
// Q5_0 Parity Tests
// ============================================================================

DEFINE_QUANTIZED_PARITY_TEST(Q5_0_SmallMatrix, Q5_0Tensor, createQ5_0Random, 32, 104)

TEST_F(Test__CUDAGemmParity, Q5_0_DecodeSize_1x896x896)
{
    const int M = 1;
    const int N = 896;
    const int K = 896;

    auto weights = TestTensorFactory::createQ5_0Random({(size_t)N, (size_t)K}, 124);
    auto A_data = randomFP32(M * K);

    std::vector<float> C_cpu(M * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_NE(cpu_kernel, nullptr);
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));

    ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
    auto *cuda_kernel = getPreparedKernel(
        weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);

    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    float *d_A = nullptr;
    float *d_C = nullptr;
    std::vector<float> C_cuda(M * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

// ============================================================================
// Q5_1 Parity Tests
// ============================================================================

DEFINE_QUANTIZED_PARITY_TEST(Q5_1_SmallMatrix, Q5_1Tensor, createQ5_1Random, 32, 105)

TEST_F(Test__CUDAGemmParity, Q5_1_DecodeSize_1x896x896)
{
    const int M = 1;
    const int N = 896;
    const int K = 896;

    auto weights = TestTensorFactory::createQ5_1Random({(size_t)N, (size_t)K}, 125);
    auto A_data = randomFP32(M * K);

    std::vector<float> C_cpu(M * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_NE(cpu_kernel, nullptr);
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));

    ASSERT_TRUE(weights->ensureOnDevice(gpu_device_));
    auto *cuda_kernel = getPreparedKernel(
        weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);

    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    float *d_A = nullptr;
    float *d_C = nullptr;
    std::vector<float> C_cuda(M * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

// ============================================================================
// K-Quant Parity Tests (256-element super-blocks)
// ============================================================================

DEFINE_QUANTIZED_PARITY_TEST(Q6_K_SmallMatrix, Q6_KTensor, createQ6_KRandom, 256, 201)
DEFINE_QUANTIZED_PARITY_TEST(Q2_K_SmallMatrix, Q2_KTensor, createQ2_KRandom, 256, 202)
DEFINE_QUANTIZED_PARITY_TEST(Q3_K_SmallMatrix, Q3_KTensor, createQ3_KRandom, 256, 203)
DEFINE_QUANTIZED_PARITY_TEST(Q4_K_SmallMatrix, Q4_KTensor, createQ4_KRandom, 256, 204)
DEFINE_QUANTIZED_PARITY_TEST(Q5_K_SmallMatrix, Q5_KTensor, createQ5_KRandom, 256, 205)

DEFINE_QUANTIZED_DECODE_PARITY_TEST(Q6_K_DecodeSize, Q6_KTensor, createQ6_KRandom, 256, 211)
DEFINE_QUANTIZED_DECODE_PARITY_TEST(Q2_K_DecodeSize, Q2_KTensor, createQ2_KRandom, 256, 212)
DEFINE_QUANTIZED_DECODE_PARITY_TEST(Q3_K_DecodeSize, Q3_KTensor, createQ3_KRandom, 256, 213)
DEFINE_QUANTIZED_DECODE_PARITY_TEST(Q4_K_DecodeSize, Q4_KTensor, createQ4_KRandom, 256, 214)
DEFINE_QUANTIZED_DECODE_PARITY_TEST(Q5_K_DecodeSize, Q5_KTensor, createQ5_KRandom, 256, 215)

TEST_F(Test__CUDAGemmParity, Q4_K_VerifierSmallM_2x896x768)
{
    const int M = 2;
    const int N = 896;
    const int K = 768;

    auto weights = TestTensorFactory::createQ4_KRandom({(size_t)N, (size_t)K}, 224);
    auto A_data = randomFP32(static_cast<size_t>(M) * K);

    std::vector<float> C_cpu(static_cast<size_t>(M) * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_NE(cpu_kernel, nullptr);
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));
    ASSERT_FALSE(hasNaNOrInf(C_cpu.data(), C_cpu.size()));

    auto *cuda_kernel = getPreparedKernel(weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    std::vector<float> C_cuda(static_cast<size_t>(M) * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    auto result = checkParity(C_cuda.data(), C_cpu.data(), C_cpu.size(), 0.99, 0.15);
    result.print("Q4_K verifier-small-M 2x896x768");
    EXPECT_FALSE(result.has_nan_inf);
    EXPECT_GE(result.cosine_similarity, 0.99)
        << "Cosine similarity too low: " << result.cosine_similarity;
    EXPECT_LE(result.relative_l2_error, 0.15)
        << "Relative L2 error too high: " << (result.relative_l2_error * 100.0) << "%";

    cleanupWorkspaceIfNeeded(cuda_kernel);
    EXPECT_FALSE(cuda_kernel->hasDynamicStateActive())
        << "Small-M verifier GEMV test must not leak CUDA dynamic state after workspace cleanup";
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

TEST_F(Test__CUDAGemmParity, Q4_K_VerifierSmallM_2x512x768)
{
    const int M = 2;
    const int N = 512;
    const int K = 768;

    auto weights = TestTensorFactory::createQ4_KRandom({(size_t)N, (size_t)K}, 227);
    auto A_data = randomFP32(static_cast<size_t>(M) * K);

    std::vector<float> C_cpu(static_cast<size_t>(M) * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_NE(cpu_kernel, nullptr);
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));
    ASSERT_FALSE(hasNaNOrInf(C_cpu.data(), C_cpu.size()));

    auto *cuda_kernel = getPreparedKernel(weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    std::vector<float> C_cuda(static_cast<size_t>(M) * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    auto result = checkParity(C_cuda.data(), C_cpu.data(), C_cpu.size(), 0.99, 0.15);
    result.print("Q4_K verifier-small-M 2x512x768");
    EXPECT_FALSE(result.has_nan_inf);
    EXPECT_GE(result.cosine_similarity, 0.99)
        << "Cosine similarity too low: " << result.cosine_similarity;
    EXPECT_LE(result.relative_l2_error, 0.15)
        << "Relative L2 error too high: " << (result.relative_l2_error * 100.0) << "%";

    cleanupWorkspaceIfNeeded(cuda_kernel);
    EXPECT_FALSE(cuda_kernel->hasDynamicStateActive())
        << "Small-M verifier GEMV test must not leak CUDA dynamic state after workspace cleanup";
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

TEST_F(Test__CUDAGemmParity, Q4_K_VerifierSmallM_UsesGraphNativeRowwiseRoute)
{
    const int M = 2;
    const int N = 896;
    const int K = 768;

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    auto weights = TestTensorFactory::createQ4_KRandom({(size_t)N, (size_t)K}, 229);
    auto A_data = randomFP32(static_cast<size_t>(M) * K);

    auto *cuda_kernel = getPreparedKernel(weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    std::vector<float> C_cuda(static_cast<size_t>(M) * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    const auto records = PerfStatsCollector::snapshot({"kernel.cuda_native_vnni_m2_calls"});
    auto route_record = std::find_if(
        records.begin(),
        records.end(),
        [](const PerfStatRecord &record)
        {
            return record.domain == "kernel" &&
                   record.name == "cuda_native_vnni_m2_calls" &&
                   record.kind == PerfStatRecord::Kind::Counter;
        });

    ASSERT_NE(route_record, records.end())
        << "Q4_K M=2 verifier GEMM must use the graph-native rowwise native route";
    EXPECT_GE(route_record->value, 1.0);
    EXPECT_EQ(route_record->device, "cuda:" + std::to_string(gpu_device_.ordinal));
    EXPECT_EQ(route_record->tags.at("codebook"), "5");
    EXPECT_EQ(route_record->tags.at("n"), std::to_string(N));
    EXPECT_EQ(route_record->tags.at("k"), std::to_string(K));
    EXPECT_EQ(route_record->tags.at("route"), "rowwise");

    PerfStatsCollector::reset();
    cleanupWorkspaceIfNeeded(cuda_kernel);
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

TEST_F(Test__CUDAGemmParity, Q4_K_VerifierSmallM_M2MatchesTwoSingleRowDecodeGEMVs)
{
    const int M = 2;
    const int N = 896;
    const int K = 768;

    auto weights = TestTensorFactory::createQ4_KRandom({static_cast<size_t>(N), static_cast<size_t>(K)}, 230);
    auto A_data = randomFP32(static_cast<size_t>(M) * K);

    auto *cuda_kernel = getPreparedKernel(weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    std::vector<float> C_m2(static_cast<size_t>(M) * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_m2.data(), M, N, K, gpu_device_))
        << "M=2 verifier GEMM failed";

    for (int row = 0; row < M; ++row)
    {
        std::vector<float> C_m1(static_cast<size_t>(N), 0.0f);
        ASSERT_TRUE(cudaMultiplyViaTensor(
            cuda_kernel,
            A_data.data() + static_cast<size_t>(row) * K,
            C_m1.data(),
            1,
            N,
            K,
            gpu_device_))
            << "M=1 decode GEMV failed for row " << row;

        const float *verifier_row = C_m2.data() + static_cast<size_t>(row) * N;
        const auto result = checkParity(verifier_row, C_m1.data(), C_m1.size(), 0.999999, 1e-5);
        EXPECT_FALSE(result.has_nan_inf)
            << "M=2 verifier row " << row << " produced non-finite output";
        EXPECT_GE(result.cosine_similarity, 0.999999)
            << "M=2 verifier row " << row << " diverges from single-row decode GEMV";
        EXPECT_LE(result.relative_l2_error, 1e-5)
            << "M=2 verifier row " << row << " relative L2 differs from single-row decode GEMV";
    }

    cleanupWorkspaceIfNeeded(cuda_kernel);
    EXPECT_FALSE(cuda_kernel->hasDynamicStateActive())
        << "M=2 row-equivalence test must not leak CUDA dynamic state after workspace cleanup";
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

TEST_F(Test__CUDAGemmParity, Q4_K_Qwen36GDNOut_M4MatchesFourSingleRowDecodeGEMVs)
{
    const int M = 4;
    const int N = 5120;
    const int K = 6144;

    auto weights = TestTensorFactory::createQ4_KRandom({static_cast<size_t>(N), static_cast<size_t>(K)}, 231);
    auto A_data = randomFP32(static_cast<size_t>(M) * K);

    auto *cuda_kernel = getPreparedKernel(weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    std::vector<float> C_m4(static_cast<size_t>(M) * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_m4.data(), M, N, K, gpu_device_))
        << "M=4 Qwen3.6 GDN output projection GEMM failed";

    for (int row = 0; row < M; ++row)
    {
        std::vector<float> C_m1(static_cast<size_t>(N), 0.0f);
        ASSERT_TRUE(cudaMultiplyViaTensor(
            cuda_kernel,
            A_data.data() + static_cast<size_t>(row) * K,
            C_m1.data(),
            1,
            N,
            K,
            gpu_device_))
            << "M=1 Qwen3.6 GDN output projection GEMV failed for row " << row;

        const float *verifier_row = C_m4.data() + static_cast<size_t>(row) * N;
        const auto result = checkParity(verifier_row, C_m1.data(), C_m1.size(), 0.999999, 1e-5);
        EXPECT_FALSE(result.has_nan_inf)
            << "M=4 Qwen3.6 GDN output projection row " << row
            << " produced non-finite output";
        EXPECT_GE(result.cosine_similarity, 0.999999)
            << "M=4 Qwen3.6 GDN output projection row " << row
            << " diverges from single-row decode GEMV";
        EXPECT_LE(result.relative_l2_error, 1e-5)
            << "M=4 Qwen3.6 GDN output projection row " << row
            << " relative L2 differs from single-row decode GEMV";
    }

    cleanupWorkspaceIfNeeded(cuda_kernel);
    EXPECT_FALSE(cuda_kernel->hasDynamicStateActive())
        << "M=4 Qwen3.6 row-equivalence test must not leak CUDA dynamic state";
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

TEST_F(Test__CUDAGemmParity, Q4_K_FusedVerifierSmallM_2RowsTwoProjections)
{
    const int M = 2;
    const int N0 = 896;
    const int N1 = 512;
    const int K = 768;

    auto weights0 = TestTensorFactory::createQ4_KRandom({(size_t)N0, (size_t)K}, 225);
    auto weights1 = TestTensorFactory::createQ4_KRandom({(size_t)N1, (size_t)K}, 226);
    auto A_data = randomFP32(static_cast<size_t>(M) * K);

    std::vector<float> C0_cpu(static_cast<size_t>(M) * N0, 0.0f);
    std::vector<float> C1_cpu(static_cast<size_t>(M) * N1, 0.0f);
    auto cpu_kernel0 = llaminar::v2::kernels::KernelFactory::createGemm(
        weights0.get(), KernelDeviceType::CPU);
    auto cpu_kernel1 = llaminar::v2::kernels::KernelFactory::createGemm(
        weights1.get(), KernelDeviceType::CPU);
    ASSERT_NE(cpu_kernel0, nullptr);
    ASSERT_NE(cpu_kernel1, nullptr);
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel0.get(), A_data.data(), C0_cpu.data(), M, N0, K));
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel1.get(), A_data.data(), C1_cpu.data(), M, N1, K));

    auto *cuda_kernel0 = getPreparedKernel(weights0.get(), gpu_device_);
    auto *cuda_kernel1 = getPreparedKernel(weights1.get(), gpu_device_);
    ASSERT_NE(cuda_kernel0, nullptr);
    ASSERT_NE(cuda_kernel1, nullptr);
    ASSERT_TRUE(setupSharedWorkspace({cuda_kernel0, cuda_kernel1}, M, {N0, N1}, K));

    auto A_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)K});
    std::memcpy(A_tensor->mutable_data(), A_data.data(), A_data.size() * sizeof(float));
    auto C0_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)N0});
    auto C1_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)N1});

    std::vector<TensorProjectionDesc> projections = {
        {cuda_kernel0, C0_tensor.get(), N0, nullptr, "q4k_proj0"},
        {cuda_kernel1, C1_tensor.get(), N1, nullptr, "q4k_proj1"}};

    ASSERT_TRUE(with_gpu_coherence(
        gpu_device_,
        {A_tensor.get()},
        {C0_tensor.get(), C1_tensor.get()},
        [&]
        {
            return cuda_kernel0->multiply_fused_tensor(
                A_tensor.get(), projections, M, K);
        }));

    const float *C0_cuda = C0_tensor->data();
    const float *C1_cuda = C1_tensor->data();
    auto result0 = checkParity(C0_cuda, C0_cpu.data(), C0_cpu.size(), 0.99, 0.15);
    auto result1 = checkParity(C1_cuda, C1_cpu.data(), C1_cpu.size(), 0.99, 0.15);
    result0.print("Q4_K fused verifier-small-M projection 0");
    result1.print("Q4_K fused verifier-small-M projection 1");
    EXPECT_FALSE(result0.has_nan_inf);
    EXPECT_FALSE(result1.has_nan_inf);
    EXPECT_GE(result0.cosine_similarity, 0.99);
    EXPECT_GE(result1.cosine_similarity, 0.99);
    EXPECT_LE(result0.relative_l2_error, 0.15);
    EXPECT_LE(result1.relative_l2_error, 0.15);

    cleanupSharedWorkspace({cuda_kernel0, cuda_kernel1});
    EXPECT_FALSE(cuda_kernel0->hasDynamicStateActive())
        << "Small-M fused verifier projection 0 must not leak CUDA dynamic state after workspace cleanup";
    EXPECT_FALSE(cuda_kernel1->hasDynamicStateActive())
        << "Small-M fused verifier projection 1 must not leak CUDA dynamic state after workspace cleanup";
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights0.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights1.get());
}

TEST_F(Test__CUDAGemmParity, GDNProjectionStageFusesCUDAQuantizedQKVAndZSmallM)
{
    const int M = 2;
    const int K = 768;
    const int N_QKV = 896;
    const int N_Z = 512;
    const int N_ALPHA = 16;
    const int N_BETA = 16;

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    auto input = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    auto input_data = randomFP32(static_cast<size_t>(M) * K);
    std::memcpy(input->mutable_data(), input_data.data(), input_data.size() * sizeof(float));

    auto weights_qkv = TestTensorFactory::createQ4_KRandom(
        {static_cast<size_t>(N_QKV), static_cast<size_t>(K)}, 232);
    auto weights_z = TestTensorFactory::createQ4_KRandom(
        {static_cast<size_t>(N_Z), static_cast<size_t>(K)}, 233);
    auto weights_alpha = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(N_ALPHA), static_cast<size_t>(K)}, -0.1f, 0.1f, 234);
    auto weights_beta = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(N_BETA), static_cast<size_t>(K)}, -0.1f, 0.1f, 235);
    ASSERT_TRUE(weights_alpha->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(weights_beta->ensureOnDevice(gpu_device_));

    std::vector<float> alpha_cpu(static_cast<size_t>(M) * N_ALPHA, 0.0f);
    std::vector<float> beta_cpu(static_cast<size_t>(M) * N_BETA, 0.0f);
    auto cpu_alpha = llaminar::v2::kernels::KernelFactory::createGemm(
        weights_alpha.get(), KernelDeviceType::CPU);
    auto cpu_beta = llaminar::v2::kernels::KernelFactory::createGemm(
        weights_beta.get(), KernelDeviceType::CPU);
    ASSERT_NE(cpu_alpha, nullptr);
    ASSERT_NE(cpu_beta, nullptr);
    ASSERT_TRUE(cpuMultiplyToVector(cpu_alpha.get(), input_data.data(), alpha_cpu.data(), M, N_ALPHA, K));
    ASSERT_TRUE(cpuMultiplyToVector(cpu_beta.get(), input_data.data(), beta_cpu.data(), M, N_BETA, K));

    auto output_qkv = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_QKV)});
    auto output_z = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_Z)});
    auto output_alpha = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_ALPHA)});
    auto output_beta = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_BETA)});

    auto *kernel_qkv = getPreparedKernel(weights_qkv.get(), gpu_device_);
    auto *kernel_z = getPreparedKernel(weights_z.get(), gpu_device_);
    auto *kernel_alpha = getPreparedKernel(weights_alpha.get(), gpu_device_);
    auto *kernel_beta = getPreparedKernel(weights_beta.get(), gpu_device_);
    ASSERT_NE(kernel_qkv, nullptr);
    ASSERT_NE(kernel_z, nullptr);
    ASSERT_NE(kernel_alpha, nullptr);
    ASSERT_NE(kernel_beta, nullptr);
    ASSERT_TRUE(kernel_qkv->supports_fused_projection())
        << "CUDA quantized GEMM must advertise fused projection support so GDN can batch qkv+z";
    ASSERT_TRUE(kernel_z->supports_fused_projection())
        << "CUDA quantized GEMM must advertise fused projection support so GDN can batch qkv+z";
    ASSERT_TRUE(kernel_alpha->supports_fused_projection())
        << "CUDA FP32 GEMM must advertise fused projection support so GDN can batch alpha+beta";
    ASSERT_TRUE(kernel_beta->supports_fused_projection())
        << "CUDA FP32 GEMM must advertise fused projection support so GDN can batch alpha+beta";

    GDNProjectionStage::Params params;
    params.device_id = gpu_device_;
    params.input = input.get();
    params.m = M;
    params.k = K;
    params.w_qkv = weights_qkv.get();
    params.output_qkv = output_qkv.get();
    params.n_qkv = N_QKV;
    params.w_z = weights_z.get();
    params.output_z = output_z.get();
    params.n_z = N_Z;
    params.w_a = weights_alpha.get();
    params.output_a = output_alpha.get();
    params.n_a = N_ALPHA;
    params.w_b = weights_beta.get();
    params.output_b = output_beta.get();
    params.n_b = N_BETA;
    params.gemm_qkv = kernel_qkv;
    params.gemm_z = kernel_z;
    params.gemm_a = kernel_alpha;
    params.gemm_b = kernel_beta;

    GDNProjectionStage stage(params);
    WorkspaceRequirements reqs = stage.getWorkspaceRequirements(M, 0, K);
    workspace_ = std::make_unique<DeviceWorkspaceManager>(gpu_device_, 64 * 1024 * 1024);
    ASSERT_TRUE(workspace_->allocate(reqs));
    stage.bindWorkspace(workspace_.get());

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    stage.setGPUStream(static_cast<void *>(stream));
    CUDADeviceContext ctx(gpu_device_, gpu_device_.ordinal);

    const bool ok = with_gpu_coherence(
        gpu_device_,
        {input.get()},
        {output_qkv.get(), output_z.get(), output_alpha.get(), output_beta.get()},
        [&]
        {
            return stage.execute(&ctx);
        });
    ASSERT_TRUE(ok);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    ASSERT_NE(output_qkv->gpu_data_ptr(), nullptr);
    ASSERT_NE(output_z->gpu_data_ptr(), nullptr);
    ASSERT_NE(output_alpha->gpu_data_ptr(), nullptr);
    ASSERT_NE(output_beta->gpu_data_ptr(), nullptr);
    ASSERT_EQ(cudaMemsetAsync(output_qkv->gpu_data_ptr(), 0,
                              static_cast<size_t>(M) * N_QKV * sizeof(float), stream),
              cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(output_z->gpu_data_ptr(), 0,
                              static_cast<size_t>(M) * N_Z * sizeof(float), stream),
              cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(output_alpha->gpu_data_ptr(), 0,
                              static_cast<size_t>(M) * N_ALPHA * sizeof(float), stream),
              cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(output_beta->gpu_data_ptr(), 0,
                              static_cast<size_t>(M) * N_BETA * sizeof(float), stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), cudaSuccess);
    ASSERT_TRUE(stage.execute(&ctx))
        << "CUDA GDN projection stage should be graph-capturable with fused qkv+z and alpha+beta subgroups";
    ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0), cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaGraphExecDestroy(graph_exec), cudaSuccess);
    ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);

    output_qkv->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_z->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_alpha->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_beta->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

    stage.unbindWorkspace();
    workspace_.reset();
    kernel_qkv->resetDynamicState();
    kernel_z->resetDynamicState();
    kernel_alpha->resetDynamicState();
    kernel_beta->resetDynamicState();

    const auto route_records = PerfStatsCollector::snapshot({"kernel.gdn_projection_route"});
    const auto qkv_z_route = std::find_if(
        route_records.begin(),
        route_records.end(),
        [&](const PerfStatRecord &record)
        {
            return record.domain == "kernel" &&
                   record.name == "gdn_projection_route" &&
                   record.kind == PerfStatRecord::Kind::Counter &&
                   record.tags.at("route") == "native_subgroup" &&
                   record.tags.at("m") == std::to_string(M) &&
                   record.tags.at("k") == std::to_string(K) &&
                   record.tags.at("projections") == "2" &&
                   record.tags.at("names") == "qkv+z";
        });
    ASSERT_NE(qkv_z_route, route_records.end())
        << "CUDA GDN verifier projection must fuse quantized qkv+z instead of routing both as single fallbacks";

    const auto alpha_beta_route = std::find_if(
        route_records.begin(),
        route_records.end(),
        [&](const PerfStatRecord &record)
        {
            return record.domain == "kernel" &&
                   record.name == "gdn_projection_route" &&
                   record.kind == PerfStatRecord::Kind::Counter &&
                   record.tags.at("route") == "same_kernel_mixed_codebook_subgroup" &&
                   record.tags.at("m") == std::to_string(M) &&
                   record.tags.at("k") == std::to_string(K) &&
                   record.tags.at("projections") == "2" &&
                   record.tags.at("names") == "alpha+beta";
        });
    ASSERT_NE(alpha_beta_route, route_records.end())
        << "CUDA GDN verifier projection must batch FP32 alpha+beta instead of routing them as single fallbacks";

    const auto fused_records =
        PerfStatsCollector::snapshot({"kernel.cuda_native_vnni_small_m_fused_projection_calls"});
    const auto fused_kernel_record = std::find_if(
        fused_records.begin(),
        fused_records.end(),
        [&](const PerfStatRecord &record)
        {
            return record.domain == "kernel" &&
                   record.name == "cuda_native_vnni_small_m_fused_projection_calls" &&
                   record.kind == PerfStatRecord::Kind::Counter &&
                   record.tags.at("m") == std::to_string(M) &&
                   record.tags.at("k") == std::to_string(K) &&
                   record.tags.at("projections") == "2";
        });
    ASSERT_NE(fused_kernel_record, fused_records.end())
        << "CUDA qkv+z GDN subgroup should execute the graph-native small-M fused projection kernel";

    const auto fp32_fused_records =
        PerfStatsCollector::snapshot({"kernel.cuda_fp32_batched_fused_projection_calls"});
    const auto fp32_fused_record = std::find_if(
        fp32_fused_records.begin(),
        fp32_fused_records.end(),
        [&](const PerfStatRecord &record)
        {
            return record.domain == "kernel" &&
                   record.name == "cuda_fp32_batched_fused_projection_calls" &&
                   record.kind == PerfStatRecord::Kind::Counter &&
                   record.tags.at("m") == std::to_string(M) &&
                   record.tags.at("k") == std::to_string(K) &&
                   record.tags.at("n") == std::to_string(N_ALPHA) &&
                   record.tags.at("projections") == "2" &&
                   record.tags.at("route") == "cublas_batched_same_a";
        });
    ASSERT_NE(fp32_fused_record, fp32_fused_records.end())
        << "CUDA alpha+beta GDN subgroup should execute the graph-capturable cuBLAS batched projection route";

    const auto alpha_result =
        checkParity(output_alpha->data(), alpha_cpu.data(), alpha_cpu.size(), 0.9999, 0.01);
    const auto beta_result =
        checkParity(output_beta->data(), beta_cpu.data(), beta_cpu.size(), 0.9999, 0.01);
    EXPECT_FALSE(alpha_result.has_nan_inf);
    EXPECT_FALSE(beta_result.has_nan_inf);
    EXPECT_GE(alpha_result.cosine_similarity, 0.9999);
    EXPECT_GE(beta_result.cosine_similarity, 0.9999);
    EXPECT_LE(alpha_result.relative_l2_error, 0.01);
    EXPECT_LE(beta_result.relative_l2_error, 0.01);

    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_qkv.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_z.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_alpha.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_beta.get());
    PerfStatsCollector::reset();
}

TEST_F(Test__CUDAGemmParity, MTP_SmallM_FusedProjection_AllNativeFormats)
{
    const int K = 256;
    const int N0 = 192;
    const int N1 = 128;
    const std::array<int, 3> verifier_rows = {2, 3, 4};

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    for (const auto &fmt : cudaSmallMNativeFormats())
    {
        auto weights0 = fmt.create(N0, K);
        auto weights1 = fmt.create(N1, K);
        ASSERT_NE(weights0, nullptr) << "Failed to create " << fmt.name << " first projection";
        ASSERT_NE(weights1, nullptr) << "Failed to create " << fmt.name << " second projection";

        auto cpu_kernel0 = llaminar::v2::kernels::KernelFactory::createGemm(
            weights0.get(), KernelDeviceType::CPU);
        auto cpu_kernel1 = llaminar::v2::kernels::KernelFactory::createGemm(
            weights1.get(), KernelDeviceType::CPU);
        ASSERT_NE(cpu_kernel0, nullptr) << fmt.name << " CPU projection 0 kernel";
        ASSERT_NE(cpu_kernel1, nullptr) << fmt.name << " CPU projection 1 kernel";

        auto *cuda_kernel0 = getPreparedKernel(weights0.get(), gpu_device_);
        auto *cuda_kernel1 = getPreparedKernel(weights1.get(), gpu_device_);
        ASSERT_NE(cuda_kernel0, nullptr) << fmt.name << " CUDA projection 0 kernel";
        ASSERT_NE(cuda_kernel1, nullptr) << fmt.name << " CUDA projection 1 kernel";

        ASSERT_TRUE(setupSharedWorkspace({cuda_kernel0, cuda_kernel1}, 4, {N0, N1}, K))
            << fmt.name << " shared workspace";

        for (int M : verifier_rows)
        {
            auto A_data = randomFP32(static_cast<size_t>(M) * K);

            std::vector<float> C0_cpu(static_cast<size_t>(M) * N0, 0.0f);
            std::vector<float> C1_cpu(static_cast<size_t>(M) * N1, 0.0f);
            ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel0.get(), A_data.data(), C0_cpu.data(), M, N0, K))
                << fmt.name << " CPU projection 0 at M=" << M;
            ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel1.get(), A_data.data(), C1_cpu.data(), M, N1, K))
                << fmt.name << " CPU projection 1 at M=" << M;

            auto A_tensor = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
            std::memcpy(A_tensor->mutable_data(), A_data.data(), A_data.size() * sizeof(float));
            auto C0_tensor = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N0)});
            auto C1_tensor = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N1)});

            std::vector<TensorProjectionDesc> projections = {
                {cuda_kernel0, C0_tensor.get(), N0, nullptr, "qkv"},
                {cuda_kernel1, C1_tensor.get(), N1, nullptr, "z"}};

            ASSERT_TRUE(with_gpu_coherence(
                gpu_device_,
                {A_tensor.get()},
                {C0_tensor.get(), C1_tensor.get()},
                [&]
                {
                    return cuda_kernel0->multiply_fused_tensor(
                        A_tensor.get(), projections, M, K);
                }))
                << fmt.name << " CUDA fused projection failed at M=" << M;

            const float *C0_cuda = C0_tensor->data();
            const float *C1_cuda = C1_tensor->data();
            const auto result0 =
                checkParity(C0_cuda, C0_cpu.data(), C0_cpu.size(), fmt.cosine_threshold, 0.20);
            const auto result1 =
                checkParity(C1_cuda, C1_cpu.data(), C1_cpu.size(), fmt.cosine_threshold, 0.20);
            EXPECT_FALSE(result0.has_nan_inf)
                << fmt.name << " projection 0 produced non-finite output at M=" << M;
            EXPECT_FALSE(result1.has_nan_inf)
                << fmt.name << " projection 1 produced non-finite output at M=" << M;
            EXPECT_GE(result0.cosine_similarity, fmt.cosine_threshold)
                << fmt.name << " projection 0 cosine too low at M=" << M
                << " rel_l2=" << result0.relative_l2_error
                << " max_abs=" << result0.max_abs_error;
            EXPECT_GE(result1.cosine_similarity, fmt.cosine_threshold)
                << fmt.name << " projection 1 cosine too low at M=" << M
                << " rel_l2=" << result1.relative_l2_error
                << " max_abs=" << result1.max_abs_error;
            EXPECT_LE(result0.relative_l2_error, 0.20)
                << fmt.name << " projection 0 relative L2 too high at M=" << M;
            EXPECT_LE(result1.relative_l2_error, 0.20)
                << fmt.name << " projection 1 relative L2 too high at M=" << M;
        }

        cleanupSharedWorkspace({cuda_kernel0, cuda_kernel1});
        EXPECT_FALSE(cuda_kernel0->hasDynamicStateActive())
            << fmt.name << " projection 0 leaked CUDA dynamic state";
        EXPECT_FALSE(cuda_kernel1->hasDynamicStateActive())
            << fmt.name << " projection 1 leaked CUDA dynamic state";
        llaminar::v2::kernels::KernelFactory::clearCacheFor(weights0.get());
        llaminar::v2::kernels::KernelFactory::clearCacheFor(weights1.get());
    }

    const auto records =
        PerfStatsCollector::snapshot({"kernel.cuda_native_vnni_small_m_fused_projection_calls"});
    uint64_t total_count = 0;
    for (const auto &record : records)
    {
        if (record.domain == "kernel" &&
            record.name == "cuda_native_vnni_small_m_fused_projection_calls" &&
            record.kind == PerfStatRecord::Kind::Counter)
        {
            total_count += record.count;
            EXPECT_EQ(record.tags.at("k"), std::to_string(K));
            EXPECT_EQ(record.tags.at("projections"), "2");
            const std::string &m_tag = record.tags.at("m");
            const std::string &route_tag = record.tags.at("route");
            if (m_tag == "3")
            {
                EXPECT_EQ(route_tag, "rowwise")
                    << "M=3 CUDA verifier projections should stay on the known-good rowwise route until M=3 batching proves faster";
            }
            else if (m_tag == "4")
            {
                EXPECT_EQ(route_tag, "rowwise")
                    << "M=4 CUDA verifier projections should stay on the known-good rowwise route until M=4 batching proves faster";
            }
            else if (m_tag == "2")
            {
                EXPECT_EQ(route_tag, "m2_or_rowwise")
                    << "M=2 CUDA verifier projections should use the native verifier route";
            }
        }
    }
    EXPECT_EQ(total_count, cudaSmallMNativeFormats().size() * verifier_rows.size())
        << "Every CUDA native format and M=2/3/4 verifier shape should use the fused small-M route";

    PerfStatsCollector::reset();
}

TEST_F(Test__CUDAGemmParity, MTP_SmallM_MoEProjectionNamesStayRowwiseUntilBatchedRouteIsProven)
{
    const int K = 256;
    const int N0 = 192;
    const int N1 = 128;
    const std::array<int, 2> verifier_rows = {3, 4};

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    auto weights0 = TestTensorFactory::createQ4_KRandom({static_cast<size_t>(N0), static_cast<size_t>(K)}, 230);
    auto weights1 = TestTensorFactory::createQ4_KRandom({static_cast<size_t>(N1), static_cast<size_t>(K)}, 231);
    ASSERT_NE(weights0, nullptr);
    ASSERT_NE(weights1, nullptr);

    auto cpu_kernel0 = llaminar::v2::kernels::KernelFactory::createGemm(
        weights0.get(), KernelDeviceType::CPU);
    auto cpu_kernel1 = llaminar::v2::kernels::KernelFactory::createGemm(
        weights1.get(), KernelDeviceType::CPU);
    ASSERT_NE(cpu_kernel0, nullptr);
    ASSERT_NE(cpu_kernel1, nullptr);

    auto *cuda_kernel0 = getPreparedKernel(weights0.get(), gpu_device_);
    auto *cuda_kernel1 = getPreparedKernel(weights1.get(), gpu_device_);
    ASSERT_NE(cuda_kernel0, nullptr);
    ASSERT_NE(cuda_kernel1, nullptr);
    ASSERT_TRUE(setupSharedWorkspace({cuda_kernel0, cuda_kernel1}, 4, {N0, N1}, K));

    for (int M : verifier_rows)
    {
        auto A_data = randomFP32(static_cast<size_t>(M) * K);

        std::vector<float> C0_cpu(static_cast<size_t>(M) * N0, 0.0f);
        std::vector<float> C1_cpu(static_cast<size_t>(M) * N1, 0.0f);
        ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel0.get(), A_data.data(), C0_cpu.data(), M, N0, K));
        ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel1.get(), A_data.data(), C1_cpu.data(), M, N1, K));

        auto A_tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
        std::memcpy(A_tensor->mutable_data(), A_data.data(), A_data.size() * sizeof(float));
        auto C0_tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N0)});
        auto C1_tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N1)});

        std::vector<TensorProjectionDesc> projections = {
            {cuda_kernel0, C0_tensor.get(), N0, nullptr, "shared_gate"},
            {cuda_kernel1, C1_tensor.get(), N1, nullptr, "shared_up"}};

        ASSERT_TRUE(with_gpu_coherence(
            gpu_device_,
            {A_tensor.get()},
            {C0_tensor.get(), C1_tensor.get()},
            [&]
            {
                return cuda_kernel0->multiply_fused_tensor(
                    A_tensor.get(), projections, M, K);
            }))
            << "MoE-style CUDA fused projection failed at M=" << M;

        const float *C0_cuda = C0_tensor->data();
        const float *C1_cuda = C1_tensor->data();
        const auto result0 =
            checkParity(C0_cuda, C0_cpu.data(), C0_cpu.size(), 0.990, 0.20);
        const auto result1 =
            checkParity(C1_cuda, C1_cpu.data(), C1_cpu.size(), 0.990, 0.20);
        EXPECT_FALSE(result0.has_nan_inf);
        EXPECT_FALSE(result1.has_nan_inf);
        EXPECT_GE(result0.cosine_similarity, 0.990)
            << "projection 0 cosine too low at M=" << M;
        EXPECT_GE(result1.cosine_similarity, 0.990)
            << "projection 1 cosine too low at M=" << M;
        EXPECT_LE(result0.relative_l2_error, 0.20);
        EXPECT_LE(result1.relative_l2_error, 0.20);
    }

    cleanupSharedWorkspace({cuda_kernel0, cuda_kernel1});
    EXPECT_FALSE(cuda_kernel0->hasDynamicStateActive());
    EXPECT_FALSE(cuda_kernel1->hasDynamicStateActive());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights0.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights1.get());

    const auto records =
        PerfStatsCollector::snapshot({"kernel.cuda_native_vnni_small_m_fused_projection_calls"});
    uint64_t total_count = 0;
    for (const auto &record : records)
    {
        if (record.domain == "kernel" &&
            record.name == "cuda_native_vnni_small_m_fused_projection_calls" &&
            record.kind == PerfStatRecord::Kind::Counter)
        {
            total_count += record.count;
            EXPECT_EQ(record.tags.at("k"), std::to_string(K));
            EXPECT_EQ(record.tags.at("projections"), "2");
            EXPECT_EQ(record.tags.at("route"), "rowwise")
                << "MoE projection groups must not use the GDN chunked route until that path has direct parity coverage";
        }
    }
    EXPECT_EQ(total_count, verifier_rows.size());

    PerfStatsCollector::reset();
}

TEST_F(Test__CUDAGemmParity, Q4_K_FusedSwiGLUDownSmallM_2x896x768)
{
    const int M = 2;
    const int N = 896;
    const int K = 768;

    auto weights = TestTensorFactory::createQ4_KRandom({(size_t)N, (size_t)K}, 228);
    auto gate_data = randomFP32(static_cast<size_t>(M) * K);
    auto up_data = randomFP32(static_cast<size_t>(M) * K);

    std::vector<float> C_cpu(static_cast<size_t>(M) * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_NE(cpu_kernel, nullptr);
    ASSERT_TRUE(cpuFusedSwiGLUDownToVector(
        cpu_kernel.get(), gate_data.data(), up_data.data(), C_cpu.data(), M, N, K));
    ASSERT_FALSE(hasNaNOrInf(C_cpu.data(), C_cpu.size()));

    auto *cuda_kernel = getPreparedKernel(weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    std::vector<float> C_cuda(static_cast<size_t>(M) * N, 0.0f);
    ASSERT_TRUE(cudaFusedSwiGLUDownViaTensor(
        cuda_kernel, gate_data.data(), up_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    auto result = checkParity(C_cuda.data(), C_cpu.data(), C_cpu.size(), 0.99, 0.15);
    result.print("Q4_K fused-SwiGLU down small-M 2x896x768");
    EXPECT_FALSE(result.has_nan_inf);
    EXPECT_GE(result.cosine_similarity, 0.99)
        << "Cosine similarity too low: " << result.cosine_similarity;
    EXPECT_LE(result.relative_l2_error, 0.15)
        << "Relative L2 error too high: " << (result.relative_l2_error * 100.0) << "%";

    cleanupWorkspaceIfNeeded(cuda_kernel);
    EXPECT_FALSE(cuda_kernel->hasDynamicStateActive())
        << "Small-M fused-SwiGLU down test must not leak CUDA dynamic state after workspace cleanup";
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

TEST_F(Test__CUDAGemmParity, Q4_K_Qwen36FFNGateUp_M4FusedProjectionMatchesFourSingleRowDecodeGEMVs)
{
    const int M = 4;
    const int N = 17408;
    const int K = 5120;

    auto weights_gate = TestTensorFactory::createQ4_KRandom(
        {static_cast<size_t>(N), static_cast<size_t>(K)}, 331);
    auto weights_up = TestTensorFactory::createQ4_KRandom(
        {static_cast<size_t>(N), static_cast<size_t>(K)}, 332);
    auto A_data = randomFP32(static_cast<size_t>(M) * K);

    auto *cuda_gate = getPreparedKernel(weights_gate.get(), gpu_device_);
    auto *cuda_up = getPreparedKernel(weights_up.get(), gpu_device_);
    ASSERT_NE(cuda_gate, nullptr);
    ASSERT_NE(cuda_up, nullptr);
    ASSERT_TRUE(setupSharedWorkspace({cuda_gate, cuda_up}, M, {N, N}, K));

    auto A_m4 = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    std::memcpy(A_m4->mutable_data(), A_data.data(), A_data.size() * sizeof(float));
    auto gate_m4 = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});
    auto up_m4 = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});
    std::vector<TensorProjectionDesc> m4_projections = {
        {cuda_gate, gate_m4.get(), N, nullptr, "ffn_gate"},
        {cuda_up, up_m4.get(), N, nullptr, "ffn_up"}};

    ASSERT_TRUE(with_gpu_coherence(
        gpu_device_,
        {A_m4.get()},
        {gate_m4.get(), up_m4.get()},
        [&]
        {
            return cuda_gate->multiply_fused_tensor(
                A_m4.get(), m4_projections, M, K);
        }))
        << "M=4 Qwen3.6 FFN gate/up fused projection failed";

    for (int row = 0; row < M; ++row)
    {
        auto A_m1 = std::make_unique<FP32Tensor>(
            std::vector<size_t>{size_t{1}, static_cast<size_t>(K)});
        std::memcpy(
            A_m1->mutable_data(),
            A_data.data() + static_cast<size_t>(row) * K,
            static_cast<size_t>(K) * sizeof(float));
        auto gate_m1 = std::make_unique<FP32Tensor>(
            std::vector<size_t>{size_t{1}, static_cast<size_t>(N)});
        auto up_m1 = std::make_unique<FP32Tensor>(
            std::vector<size_t>{size_t{1}, static_cast<size_t>(N)});
        std::vector<TensorProjectionDesc> m1_projections = {
            {cuda_gate, gate_m1.get(), N, nullptr, "ffn_gate"},
            {cuda_up, up_m1.get(), N, nullptr, "ffn_up"}};

        ASSERT_TRUE(with_gpu_coherence(
            gpu_device_,
            {A_m1.get()},
            {gate_m1.get(), up_m1.get()},
            [&]
            {
                return cuda_gate->multiply_fused_tensor(
                    A_m1.get(), m1_projections, 1, K);
            }))
            << "M=1 Qwen3.6 FFN gate/up projection failed for row " << row;

        const float *gate_row =
            gate_m4->data() + static_cast<size_t>(row) * N;
        const float *up_row =
            up_m4->data() + static_cast<size_t>(row) * N;
        const auto gate_result =
            checkParity(gate_row, gate_m1->data(), static_cast<size_t>(N), 0.999999, 1.0e-5);
        const auto up_result =
            checkParity(up_row, up_m1->data(), static_cast<size_t>(N), 0.999999, 1.0e-5);
        EXPECT_FALSE(gate_result.has_nan_inf)
            << "M=4 Qwen3.6 FFN gate row " << row
            << " produced non-finite output";
        EXPECT_FALSE(up_result.has_nan_inf)
            << "M=4 Qwen3.6 FFN up row " << row
            << " produced non-finite output";
        EXPECT_GE(gate_result.cosine_similarity, 0.999999)
            << "M=4 Qwen3.6 FFN gate row " << row
            << " diverges from single-row decode GEMV";
        EXPECT_GE(up_result.cosine_similarity, 0.999999)
            << "M=4 Qwen3.6 FFN up row " << row
            << " diverges from single-row decode GEMV";
        EXPECT_LE(gate_result.relative_l2_error, 1.0e-5)
            << "M=4 Qwen3.6 FFN gate row " << row
            << " relative L2 differs from single-row decode GEMV";
        EXPECT_LE(up_result.relative_l2_error, 1.0e-5)
            << "M=4 Qwen3.6 FFN up row " << row
            << " relative L2 differs from single-row decode GEMV";
    }

    cleanupSharedWorkspace({cuda_gate, cuda_up});
    EXPECT_FALSE(cuda_gate->hasDynamicStateActive());
    EXPECT_FALSE(cuda_up->hasDynamicStateActive());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_gate.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_up.get());
}

TEST_F(Test__CUDAGemmParity, Q5Native_Qwen36FFNGateUp_M4FusedProjectionMatchesFourSingleRowDecodeGEMVs)
{
    const int M = 4;
    const int N = 17408;
    const int K = 5120;

    struct FormatCase
    {
        const char *name;
        std::function<std::unique_ptr<TensorBase>(size_t, size_t, int)> create;
    };

    const std::array<FormatCase, 2> formats = {
        FormatCase{
            "Q5_1",
            [](size_t n, size_t k, int seed)
            {
                return TestTensorFactory::createQ5_1Random({n, k}, seed);
            }},
        FormatCase{
            "Q5_K",
            [](size_t n, size_t k, int seed)
            {
                return TestTensorFactory::createQ5_KRandom({n, k}, seed);
            }}};

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");

    for (size_t fmt_index = 0; fmt_index < formats.size(); ++fmt_index)
    {
        const auto &fmt = formats[fmt_index];
        PerfStatsCollector::reset();

        auto weights_gate = fmt.create(static_cast<size_t>(N), static_cast<size_t>(K),
                                       341 + static_cast<int>(fmt_index) * 10);
        auto weights_up = fmt.create(static_cast<size_t>(N), static_cast<size_t>(K),
                                     342 + static_cast<int>(fmt_index) * 10);
        auto A_data = randomFP32(static_cast<size_t>(M) * K);

        auto *cuda_gate = getPreparedKernel(weights_gate.get(), gpu_device_);
        auto *cuda_up = getPreparedKernel(weights_up.get(), gpu_device_);
        ASSERT_NE(cuda_gate, nullptr) << fmt.name << " gate kernel";
        ASSERT_NE(cuda_up, nullptr) << fmt.name << " up kernel";
        ASSERT_TRUE(setupSharedWorkspace({cuda_gate, cuda_up}, M, {N, N}, K))
            << fmt.name << " shared workspace";

        auto A_m4 = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
        std::memcpy(A_m4->mutable_data(), A_data.data(), A_data.size() * sizeof(float));
        auto gate_m4 = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});
        auto up_m4 = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});
        std::vector<TensorProjectionDesc> m4_projections = {
            {cuda_gate, gate_m4.get(), N, nullptr, "ffn_gate"},
            {cuda_up, up_m4.get(), N, nullptr, "ffn_up"}};

        ASSERT_TRUE(with_gpu_coherence(
            gpu_device_,
            {A_m4.get()},
            {gate_m4.get(), up_m4.get()},
            [&]
            {
                return cuda_gate->multiply_fused_tensor(
                    A_m4.get(), m4_projections, M, K);
            }))
            << fmt.name << " M=4 Qwen3.6 FFN gate/up fused projection failed";

        for (int row = 0; row < M; ++row)
        {
            auto A_m1 = std::make_unique<FP32Tensor>(
                std::vector<size_t>{size_t{1}, static_cast<size_t>(K)});
            std::memcpy(
                A_m1->mutable_data(),
                A_data.data() + static_cast<size_t>(row) * K,
                static_cast<size_t>(K) * sizeof(float));
            auto gate_m1 = std::make_unique<FP32Tensor>(
                std::vector<size_t>{size_t{1}, static_cast<size_t>(N)});
            auto up_m1 = std::make_unique<FP32Tensor>(
                std::vector<size_t>{size_t{1}, static_cast<size_t>(N)});
            std::vector<TensorProjectionDesc> m1_projections = {
                {cuda_gate, gate_m1.get(), N, nullptr, "ffn_gate"},
                {cuda_up, up_m1.get(), N, nullptr, "ffn_up"}};

            ASSERT_TRUE(with_gpu_coherence(
                gpu_device_,
                {A_m1.get()},
                {gate_m1.get(), up_m1.get()},
                [&]
                {
                    return cuda_gate->multiply_fused_tensor(
                        A_m1.get(), m1_projections, 1, K);
                }))
                << fmt.name << " M=1 Qwen3.6 FFN gate/up projection failed for row " << row;

            const float *gate_row =
                gate_m4->data() + static_cast<size_t>(row) * N;
            const float *up_row =
                up_m4->data() + static_cast<size_t>(row) * N;
            const auto gate_result =
                checkParity(gate_row, gate_m1->data(), static_cast<size_t>(N), 0.999999, 1.0e-5);
            const auto up_result =
                checkParity(up_row, up_m1->data(), static_cast<size_t>(N), 0.999999, 1.0e-5);
            EXPECT_FALSE(gate_result.has_nan_inf)
                << fmt.name << " M=4 Qwen3.6 FFN gate row " << row
                << " produced non-finite output";
            EXPECT_FALSE(up_result.has_nan_inf)
                << fmt.name << " M=4 Qwen3.6 FFN up row " << row
                << " produced non-finite output";
            EXPECT_GE(gate_result.cosine_similarity, 0.999999)
                << fmt.name << " M=4 Qwen3.6 FFN gate row " << row
                << " diverges from single-row decode GEMV";
            EXPECT_GE(up_result.cosine_similarity, 0.999999)
                << fmt.name << " M=4 Qwen3.6 FFN up row " << row
                << " diverges from single-row decode GEMV";
            EXPECT_LE(gate_result.relative_l2_error, 1.0e-5)
                << fmt.name << " M=4 Qwen3.6 FFN gate row " << row
                << " relative L2 differs from single-row decode GEMV";
            EXPECT_LE(up_result.relative_l2_error, 1.0e-5)
                << fmt.name << " M=4 Qwen3.6 FFN up row " << row
                << " relative L2 differs from single-row decode GEMV";
        }

        const auto route_records =
            PerfStatsCollector::snapshot({"kernel.cuda_native_vnni_small_m_calls"});
        uint64_t codebook7_m4_rows = 0;
        for (const auto &record : route_records)
        {
            if (record.domain == "kernel" &&
                record.name == "cuda_native_vnni_small_m_calls" &&
                record.kind == PerfStatRecord::Kind::Counter &&
                record.tags.at("m") == std::to_string(M) &&
                record.tags.at("codebook") == "7" &&
                record.tags.at("n") == std::to_string(N) &&
                record.tags.at("k") == std::to_string(K) &&
                record.tags.at("route") == "rowwise")
            {
                codebook7_m4_rows += record.count;
            }
        }
        EXPECT_EQ(codebook7_m4_rows, 2u)
            << fmt.name << " M=4 gate/up must route through the graph-native "
            << "rowwise codebook-7 verifier path";

        cleanupSharedWorkspace({cuda_gate, cuda_up});
        EXPECT_FALSE(cuda_gate->hasDynamicStateActive());
        EXPECT_FALSE(cuda_up->hasDynamicStateActive());
        llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_gate.get());
        llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_up.get());
        PerfStatsCollector::reset();
    }
}

TEST_F(Test__CUDAGemmParity, Q4_K_Qwen36FFNDown_M4FusedSwiGLUMatchesFourSingleRowDecodeGEMVs)
{
    const int M = 4;
    const int N = 5120;
    const int K = 17408;

    auto weights = TestTensorFactory::createQ4_KRandom({static_cast<size_t>(N), static_cast<size_t>(K)}, 233);
    auto gate_data = randomFP32(static_cast<size_t>(M) * K);
    auto up_data = randomFP32(static_cast<size_t>(M) * K);

    auto *cuda_kernel = getPreparedKernel(weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    std::vector<float> C_m4(static_cast<size_t>(M) * N, 0.0f);
    ASSERT_TRUE(cudaFusedSwiGLUDownViaTensor(
        cuda_kernel,
        gate_data.data(),
        up_data.data(),
        C_m4.data(),
        M,
        N,
        K,
        gpu_device_))
        << "M=4 Qwen3.6 fused-SwiGLU down projection failed";

    for (int row = 0; row < M; ++row)
    {
        std::vector<float> C_m1(static_cast<size_t>(N), 0.0f);
        ASSERT_TRUE(cudaFusedSwiGLUDownViaTensor(
            cuda_kernel,
            gate_data.data() + static_cast<size_t>(row) * K,
            up_data.data() + static_cast<size_t>(row) * K,
            C_m1.data(),
            1,
            N,
            K,
            gpu_device_))
            << "M=1 Qwen3.6 fused-SwiGLU down projection failed for row " << row;

        const float *verifier_row = C_m4.data() + static_cast<size_t>(row) * N;
        const auto result = checkParity(verifier_row, C_m1.data(), C_m1.size(), 0.999999, 1e-5);
        EXPECT_FALSE(result.has_nan_inf)
            << "M=4 Qwen3.6 fused-SwiGLU down row " << row
            << " produced non-finite output";
        EXPECT_GE(result.cosine_similarity, 0.999999)
            << "M=4 Qwen3.6 fused-SwiGLU down row " << row
            << " diverges from single-row decode GEMV";
        EXPECT_LE(result.relative_l2_error, 1e-5)
            << "M=4 Qwen3.6 fused-SwiGLU down row " << row
            << " relative L2 differs from single-row decode GEMV";
    }

    cleanupWorkspaceIfNeeded(cuda_kernel);
    EXPECT_FALSE(cuda_kernel->hasDynamicStateActive())
        << "M=4 Qwen3.6 fused-SwiGLU row-equivalence test must not leak CUDA dynamic state";
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

TEST_F(Test__CUDAGemmParity, Q5Native_Qwen36FFNDown_M4FusedSwiGLUMatchesFourSingleRowDecodeGEMVs)
{
    const int M = 4;
    const int N = 5120;
    const int K = 17408;

    struct FormatCase
    {
        const char *name;
        std::function<std::unique_ptr<TensorBase>(size_t, size_t, int)> create;
    };

    const std::array<FormatCase, 2> formats = {
        FormatCase{
            "Q5_1",
            [](size_t n, size_t k, int seed)
            {
                return TestTensorFactory::createQ5_1Random({n, k}, seed);
            }},
        FormatCase{
            "Q5_K",
            [](size_t n, size_t k, int seed)
            {
                return TestTensorFactory::createQ5_KRandom({n, k}, seed);
            }}};

    for (size_t fmt_index = 0; fmt_index < formats.size(); ++fmt_index)
    {
        const auto &fmt = formats[fmt_index];

        auto weights = fmt.create(
            static_cast<size_t>(N),
            static_cast<size_t>(K),
            243 + static_cast<int>(fmt_index) * 10);
        auto gate_data = randomFP32(static_cast<size_t>(M) * K);
        auto up_data = randomFP32(static_cast<size_t>(M) * K);

        auto *cuda_kernel = getPreparedKernel(weights.get(), gpu_device_);
        ASSERT_NE(cuda_kernel, nullptr) << fmt.name << " CUDA kernel";
        ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K))
            << fmt.name << " workspace";

        std::vector<float> C_m4(static_cast<size_t>(M) * N, 0.0f);
        ASSERT_TRUE(cudaFusedSwiGLUDownViaTensor(
            cuda_kernel,
            gate_data.data(),
            up_data.data(),
            C_m4.data(),
            M,
            N,
            K,
            gpu_device_))
            << fmt.name << " M=4 Qwen3.6 fused-SwiGLU down projection failed";

        for (int row = 0; row < M; ++row)
        {
            std::vector<float> C_m1(static_cast<size_t>(N), 0.0f);
            ASSERT_TRUE(cudaFusedSwiGLUDownViaTensor(
                cuda_kernel,
                gate_data.data() + static_cast<size_t>(row) * K,
                up_data.data() + static_cast<size_t>(row) * K,
                C_m1.data(),
                1,
                N,
                K,
                gpu_device_))
                << fmt.name << " M=1 Qwen3.6 fused-SwiGLU down projection failed for row " << row;

            const float *verifier_row = C_m4.data() + static_cast<size_t>(row) * N;
            const auto result =
                checkParity(verifier_row, C_m1.data(), C_m1.size(), 0.999999, 1e-5);
            EXPECT_FALSE(result.has_nan_inf)
                << fmt.name << " M=4 Qwen3.6 fused-SwiGLU down row " << row
                << " produced non-finite output";
            EXPECT_GE(result.cosine_similarity, 0.999999)
                << fmt.name << " M=4 Qwen3.6 fused-SwiGLU down row " << row
                << " diverges from single-row decode GEMV";
            EXPECT_LE(result.relative_l2_error, 1e-5)
                << fmt.name << " M=4 Qwen3.6 fused-SwiGLU down row " << row
                << " relative L2 differs from single-row decode GEMV";
        }

        cleanupWorkspaceIfNeeded(cuda_kernel);
        EXPECT_FALSE(cuda_kernel->hasDynamicStateActive())
            << fmt.name << " M=4 Qwen3.6 fused-SwiGLU row-equivalence test "
            << "must not leak CUDA dynamic state";
        llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
    }
}

TEST_F(Test__CUDAGemmParity, Q4_K_FusedSwiGLUDownDecodeM1_1x896x768)
{
    const int M = 1;
    const int N = 896;
    const int K = 768;

    auto weights = TestTensorFactory::createQ4_KRandom({(size_t)N, (size_t)K}, 232);
    auto gate_data = randomFP32(static_cast<size_t>(M) * K);
    auto up_data = randomFP32(static_cast<size_t>(M) * K);

    std::vector<float> C_cpu(static_cast<size_t>(M) * N, 0.0f);
    auto cpu_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        weights.get(), KernelDeviceType::CPU);
    ASSERT_NE(cpu_kernel, nullptr);
    ASSERT_TRUE(cpuFusedSwiGLUDownToVector(
        cpu_kernel.get(), gate_data.data(), up_data.data(), C_cpu.data(), M, N, K));
    ASSERT_FALSE(hasNaNOrInf(C_cpu.data(), C_cpu.size()));

    auto *cuda_kernel = getPreparedKernel(weights.get(), gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    std::vector<float> C_cuda(static_cast<size_t>(M) * N, 0.0f);
    ASSERT_TRUE(cudaFusedSwiGLUDownViaTensor(
        cuda_kernel, gate_data.data(), up_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    auto result = checkParity(C_cuda.data(), C_cpu.data(), C_cpu.size(), 0.99, 0.15);
    result.print("Q4_K fused-SwiGLU down decode M=1 1x896x768");
    EXPECT_FALSE(result.has_nan_inf);
    EXPECT_GE(result.cosine_similarity, 0.99)
        << "Cosine similarity too low: " << result.cosine_similarity;
    EXPECT_LE(result.relative_l2_error, 0.15)
        << "Relative L2 error too high: " << (result.relative_l2_error * 100.0) << "%";

    cleanupWorkspaceIfNeeded(cuda_kernel);
    EXPECT_FALSE(cuda_kernel->hasDynamicStateActive())
        << "Decode fused-SwiGLU down test must not leak CUDA dynamic state after workspace cleanup";
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
}

// ============================================================================
// IQ (Importance Quantization) Parity Tests
// ============================================================================

DEFINE_QUANTIZED_PARITY_TEST(IQ4_XS_SmallMatrix, IQ4_XSTensor, createIQ4_XSRandom, 256, 301)
DEFINE_QUANTIZED_PARITY_TEST(IQ2_XXS_SmallMatrix, IQ2_XXSTensor, createIQ2_XXSRandom, 256, 302)

DEFINE_QUANTIZED_PARITY_TEST(IQ2_XS_SmallMatrix, IQ2_XSTensor, createIQ2_XSRandom, 256, 303)
DEFINE_QUANTIZED_PARITY_TEST(IQ2_S_SmallMatrix, IQ2_STensor, createIQ2_SRandom, 256, 304)
DEFINE_QUANTIZED_PARITY_TEST(IQ3_XXS_SmallMatrix, IQ3_XXSTensor, createIQ3_XXSRandom, 256, 305)
DEFINE_QUANTIZED_PARITY_TEST(IQ3_S_SmallMatrix, IQ3_STensor, createIQ3_SRandom, 256, 306)
DEFINE_QUANTIZED_PARITY_TEST(IQ1_S_SmallMatrix, IQ1_STensor, createIQ1_SRandom, 256, 307)
DEFINE_QUANTIZED_PARITY_TEST(IQ1_M_SmallMatrix, IQ1_MTensor, createIQ1_MRandom, 256, 308)

DEFINE_QUANTIZED_DECODE_PARITY_TEST(IQ4_XS_DecodeSize, IQ4_XSTensor, createIQ4_XSRandom, 256, 311)
DEFINE_QUANTIZED_DECODE_PARITY_TEST(IQ2_XXS_DecodeSize, IQ2_XXSTensor, createIQ2_XXSRandom, 256, 312)
DEFINE_QUANTIZED_DECODE_PARITY_TEST(IQ2_XS_DecodeSize, IQ2_XSTensor, createIQ2_XSRandom, 256, 313)
DEFINE_QUANTIZED_DECODE_PARITY_TEST(IQ2_S_DecodeSize, IQ2_STensor, createIQ2_SRandom, 256, 314)
DEFINE_QUANTIZED_DECODE_PARITY_TEST(IQ3_XXS_DecodeSize, IQ3_XXSTensor, createIQ3_XXSRandom, 256, 315)
DEFINE_QUANTIZED_DECODE_PARITY_TEST(IQ3_S_DecodeSize, IQ3_STensor, createIQ3_SRandom, 256, 316)
DEFINE_QUANTIZED_DECODE_PARITY_TEST(IQ1_S_DecodeSize, IQ1_STensor, createIQ1_SRandom, 256, 317)
DEFINE_QUANTIZED_DECODE_PARITY_TEST(IQ1_M_DecodeSize, IQ1_MTensor, createIQ1_MRandom, 256, 318)

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
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));

    // ===== CUDA =====
    // Upload weights to GPU
    ASSERT_TRUE(q4_tensor->ensureOnDevice(gpu_device_));
    EXPECT_TRUE(q4_tensor->isOnGPU());

    auto *cuda_kernel = getPreparedKernel(
        q4_tensor, gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr) << "Failed to create CUDA kernel";

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    float *d_A, *d_C;
    std::vector<float> C_cuda(M * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    // Clean up cache entry while tensor is still alive
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
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
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));

    // CUDA
    ASSERT_TRUE(q4_tensor->ensureOnDevice(gpu_device_));
    auto *cuda_kernel = getPreparedKernel(
        q4_tensor, gpu_device_);

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    float *d_A, *d_C;
    std::vector<float> C_cuda(M * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    // Clean up cache entry while tensor is still alive
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
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
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));

    // CUDA
    ASSERT_TRUE(q4_tensor->ensureOnDevice(gpu_device_));
    auto *cuda_kernel = getPreparedKernel(
        q4_tensor, gpu_device_);

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    float *d_A, *d_C;
    std::vector<float> C_cuda(M * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    // Clean up cache entry while tensor is still alive
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
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
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));

    // CUDA
    ASSERT_TRUE(q4_tensor->ensureOnDevice(gpu_device_));
    auto *cuda_kernel = getPreparedKernel(
        q4_tensor, gpu_device_);

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    float *d_A, *d_C;
    std::vector<float> C_cuda(M * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    // Clean up cache entry while tensor is still alive
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
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
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), A_data.data(), C_cpu.data(), M, N, K));

    // CUDA
    ASSERT_TRUE(q4_tensor->ensureOnDevice(gpu_device_));
    auto *cuda_kernel = getPreparedKernel(
        q4_tensor, gpu_device_);

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    float *d_A, *d_C;
    std::vector<float> C_cuda(static_cast<size_t>(M) * N, 0.0f);
    ASSERT_TRUE(cudaMultiplyViaTensor(cuda_kernel, A_data.data(), C_cuda.data(), M, N, K, gpu_device_));

    // Clean up cache entry while tensor is still alive
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
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

    auto *cuda_kernel = getPreparedKernel(
        q4_tensor, gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);

    // Set up workspace for quantized kernel
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

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
    cleanupWorkspaceIfNeeded(cuda_kernel);

    // ===== Compare =====
    // data() will automatically sync to host if needed
    const float *cpu_data = output_cpu->data();
    const float *cuda_data = output_cuda->data();

    auto result = checkParity(cuda_data, cpu_data, M * N, 0.99, 0.10);
    result.print("Real Model Q4_0 attn_q (multiply_tensor API)");

    EXPECT_GE(result.cosine_similarity, 0.99)
        << "multiply_tensor() API shows divergence!";
    EXPECT_FALSE(result.has_nan_inf);

    // Clean up cache entry while tensor is still alive
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
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

    // Note: ensureOnDevice is NOT needed for CUDA quantized GEMM path -
    // prepareGpuGemmOnDemand handles the upload via WeightVRAMPool internally

    // Create CUDA kernels for each projection
    auto *cuda_kernel_q = getPreparedKernel(
        weights_q, gpu_device_);
    auto *cuda_kernel_k = getPreparedKernel(
        weights_k, gpu_device_);
    auto *cuda_kernel_v = getPreparedKernel(
        weights_v, gpu_device_);
    ASSERT_NE(cuda_kernel_q, nullptr) << "Failed to create CUDA kernel for Q";
    ASSERT_NE(cuda_kernel_k, nullptr) << "Failed to create CUDA kernel for K";
    ASSERT_NE(cuda_kernel_v, nullptr) << "Failed to create CUDA kernel for V";

    // Set up SHARED workspace for fused QKV (all kernels share workspace)
    ASSERT_TRUE(setupSharedWorkspace(
        {cuda_kernel_q, cuda_kernel_k, cuda_kernel_v},
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
    projections.emplace_back(cuda_kernel_q, output_q_fused.get(), N_q,
                             nullptr, "Q");
    projections.emplace_back(cuda_kernel_k, output_k_fused.get(), N_k,
                             nullptr, "K");
    projections.emplace_back(cuda_kernel_v, output_v_fused.get(), N_v,
                             nullptr, "V");

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
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel_q.get(), h_input, q_cpu.data(), M, N_q, K));
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel_k.get(), h_input, k_cpu.data(), M, N_k, K));
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel_v.get(), h_input, v_cpu.data(), M, N_v, K));

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
    cleanupSharedWorkspace({cuda_kernel_q, cuda_kernel_k, cuda_kernel_v});

    // Clean up cache entries while tensors are still alive
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_q);
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_k);
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_v);
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

    auto *cuda_kernel_q = getPreparedKernel(
        weights_q, gpu_device_);
    auto *cuda_kernel_k = getPreparedKernel(
        weights_k, gpu_device_);
    auto *cuda_kernel_v = getPreparedKernel(
        weights_v, gpu_device_);

    // Set up SHARED workspace for fused QKV (all kernels share workspace)
    ASSERT_TRUE(setupSharedWorkspace(
        {cuda_kernel_q, cuda_kernel_k, cuda_kernel_v},
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
    projections.emplace_back(cuda_kernel_q, output_q_fused.get(), N_q,
                             nullptr, "Q");
    projections.emplace_back(cuda_kernel_k, output_k_fused.get(), N_k,
                             nullptr, "K");
    projections.emplace_back(cuda_kernel_v, output_v_fused.get(), N_v,
                             nullptr, "V");

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
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel_q.get(), h_input, q_cpu.data(), M, N_q, K));
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel_k.get(), h_input, k_cpu.data(), M, N_k, K));
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel_v.get(), h_input, v_cpu.data(), M, N_v, K));

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
    cleanupSharedWorkspace({cuda_kernel_q, cuda_kernel_k, cuda_kernel_v});

    // Clean up cache entries while tensors are still alive
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_q_base.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_k_base.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_v_base.get());
}

// ============================================================================
// Cached Kernel Tests
// ============================================================================

/**
 * @test Cached kernel parity: prepared GEMM engine vs createGemm()
 *
 * This test verifies that kernels obtained via prepared GEMM engine lookup (the caching path
 * used by the full pipeline) produce identical results to kernels created via
 * createGemm() (fresh kernel creation).
 *
 * **Why this matters**: If there's a bug in kernel caching (stale weights, incorrect
 * scale factors), this will catch it. The full pipeline uses the prepared GEMM engine path.
 */
TEST_F(Test__CUDAGemmParity, CachedKernel_vs_FreshKernel)
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

    std::cout << "Testing cached kernel parity: attn_q " << N << "x" << K << "\n";

    // Ensure on GPU
    ASSERT_TRUE(q4_tensor->ensureOnDevice(gpu_device_));

    // Get kernel via prepared GEMM path (legacy createGemm GPU path removed)
    auto *cuda_kernel = getPreparedKernel(q4_tensor, gpu_device_);
    ASSERT_NE(cuda_kernel, nullptr);

    // Set up workspace
    ASSERT_TRUE(setupWorkspaceIfNeeded(cuda_kernel, M, N, K));

    // Create input and outputs
    auto input_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    auto output_a = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});
    auto output_b = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});

    // Fill input
    float *input_data = input_tensor->mutable_data();
    for (int i = 0; i < M * K; ++i)
    {
        input_data[i] = dist_(rng_);
    }

    // Run kernel twice — should produce identical results
    ASSERT_TRUE(with_gpu_coherence(
        gpu_device_,
        {input_tensor.get()},
        {output_a.get()},
        [&]
        {
            return cuda_kernel->multiply_tensor(
                input_tensor.get(), output_a.get(),
                M, N, K, true, 1.0f, 0.0f, nullptr, nullptr, -1);
        }));

    ASSERT_TRUE(with_gpu_coherence(
        gpu_device_,
        {input_tensor.get()},
        {output_b.get()},
        [&]
        {
            return cuda_kernel->multiply_tensor(
                input_tensor.get(), output_b.get(),
                M, N, K, true, 1.0f, 0.0f, nullptr, nullptr, -1);
        }));

    // Compare: should be EXACTLY the same (same kernel, same weights, same input)
    const float *data_a = output_a->data();
    const float *data_b = output_b->data();

    auto result = checkParity(data_b, data_a, M * N, 0.9999, 0.001);
    result.print("Kernel run A vs run B (same prepared kernel)");

    EXPECT_GE(result.cosine_similarity, 0.9999)
        << "Same prepared kernel should produce nearly identical results";
    EXPECT_FALSE(result.has_nan_inf);

    // Clean up workspace
    cleanupWorkspaceIfNeeded(cuda_kernel);

    // Clean up cache entry while tensor is still alive
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
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
    auto *kernel = getPreparedKernel(q4_tensor, gpu_device_);
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

    // Clean up cache entry while tensor is still alive
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
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
    auto *kernel = getPreparedKernel(q4_tensor, gpu_device_);
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
        ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel.get(), input->data(), cpu_output.data(), M, N, K));

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

    // Clean up cache entry while tensor is still alive
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights.get());
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
    auto *cuda_kernel_q = getPreparedKernel(
        wq, gpu_device_);
    auto *cuda_kernel_k = getPreparedKernel(
        wk, gpu_device_);
    auto *cuda_kernel_v = getPreparedKernel(
        wv, gpu_device_);

    // Set up SHARED workspace for fused QKV (all kernels share workspace)
    ASSERT_TRUE(setupSharedWorkspace(
        {cuda_kernel_q, cuda_kernel_k, cuda_kernel_v},
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
    projections.emplace_back(cuda_kernel_q, output_q.get(), N_q,
                             bias_q.get(), "Q");
    projections.emplace_back(cuda_kernel_k, output_k.get(), N_k,
                             bias_k.get(), "K");
    projections.emplace_back(cuda_kernel_v, output_v.get(), N_v,
                             bias_v.get(), "V");

    ASSERT_TRUE(cuda_kernel_q->multiply_fused_tensor(
        input.get(), projections, M, K, nullptr));

    // Mark outputs as device-dirty (tests bypass DeviceGraphExecutor auto-coherence)
    output_q->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_k->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_v->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

    // ===== CPU reference (GEMM + manual bias add) =====
    auto cpu_kernel_q = llaminar::v2::kernels::KernelFactory::createGemm(
        wq, KernelDeviceType::CPU);
    auto cpu_kernel_k = llaminar::v2::kernels::KernelFactory::createGemm(
        wk, KernelDeviceType::CPU);
    auto cpu_kernel_v = llaminar::v2::kernels::KernelFactory::createGemm(
        wv, KernelDeviceType::CPU);

    std::vector<float> q_cpu(M * N_q), k_cpu(M * N_k), v_cpu(M * N_v);
    const float *h_input = input->data();
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel_q.get(), h_input, q_cpu.data(), M, N_q, K));
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel_k.get(), h_input, k_cpu.data(), M, N_k, K));
    ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel_v.get(), h_input, v_cpu.data(), M, N_v, K));

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
    cleanupSharedWorkspace({cuda_kernel_q, cuda_kernel_k, cuda_kernel_v});

    // Clean up cache entries while tensors are still alive
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_q.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_k.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_v.get());
}

/**
 * @test Fused QKV using cached kernels (simulating full pipeline)
 *
 * This test mimics the full pipeline's kernel usage:
 * 1. Uses prepared-handle + GEMM engine lookup to get cached kernels
 * 2. Uses real model weights and biases
 * 3. Runs multiple iterations like decode
 */
TEST_F(Test__CUDAGemmParity, FusedQKV_CachedKernels_MultipleIterations)
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
    auto *kernel_q = getPreparedKernel(wq, gpu_device_);
    auto *kernel_k = getPreparedKernel(wk, gpu_device_);
    auto *kernel_v = getPreparedKernel(wv, gpu_device_);
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
                                 nullptr, "Q");
        projections.emplace_back(kernel_k, out_k.get(), N_k,
                                 nullptr, "K");
        projections.emplace_back(kernel_v, out_v.get(), N_v,
                                 nullptr, "V");

        ASSERT_TRUE(kernel_q->multiply_fused_tensor(
            input.get(), projections, M, K, nullptr));

        // Mark outputs as device-dirty (tests bypass DeviceGraphExecutor auto-coherence)
        out_q->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        out_k->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        out_v->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

        // CPU reference
        std::vector<float> q_cpu(M * N_q), k_cpu(M * N_k), v_cpu(M * N_v);
        const float *h_input = input->data();
        ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel_q.get(), h_input, q_cpu.data(), M, N_q, K));
        ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel_k.get(), h_input, k_cpu.data(), M, N_k, K));
        ASSERT_TRUE(cpuMultiplyToVector(cpu_kernel_v.get(), h_input, v_cpu.data(), M, N_v, K));

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

    // Clean up cache entries while tensors are still alive
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_q.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_k.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(weights_v.get());
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
