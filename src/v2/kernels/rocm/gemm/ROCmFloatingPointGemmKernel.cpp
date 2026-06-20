/**
 * @file ROCmFloatingPointGemmKernel.cpp
 * @brief ITensorGemm adapter implementation for hipBLAS FP32/FP16/BF16 GEMM
 *
 * This is the C++ adapter that wraps HipBLASGemmKernel. It implements the full
 * ITensorGemm interface and can be compiled with the regular C++ compiler
 * (not hipcc), avoiding MPI/TensorKernels.h compilation issues.
 *
 * **Design**: The adapter:
 * 1. Implements ITensorGemm (includes IMPIContext, etc.)
 * 2. Uses shared HipBLASGemmKernel* from DeviceKernelCache (avoids JIT overhead)
 * 3. Handles tensor type introspection in multiply_tensor()
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "ROCmFloatingPointGemmKernel.h"
#include "HipBLASGemmKernel.h"
#include "backends/ComputeBackend.h"   // DeviceManager
#include "backends/DeviceId.h"         // DeviceId for cache lookup
#include "kernels/DeviceKernelCache.h" // Universal kernel cache
#include "kernels/rocm/ROCmKernelBase.h"
#include "tensors/Tensors.h"           // FP32Tensor, BF16Tensor, FP16Tensor
#include "tensors/KernelSnapshotInfo.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"
#include "utils/ROCmKernelProfiler.h"

#include <stdexcept>
#include <mutex>
#include <algorithm>
#include <atomic>
#include <hip/hip_runtime.h>

extern "C" bool rocmFp32_tiny_batched_projection(
    const float *const *d_A_array,
    const float *const *d_B_array,
    float *const *d_C_array,
    int M,
    int N,
    int K,
    int batch_count,
    int device_id,
    void *stream);

extern "C" bool rocmFp32_stage_batched_projection_pointers(
    const float **d_A_array,
    const float **d_B_array,
    float **d_C_array,
    const float *const *h_A_ptrs,
    const float *const *h_B_ptrs,
    float *const *h_C_ptrs,
    int batch_count,
    int device_id,
    void *stream);

namespace llaminar2
{
    namespace rocm
    {
        namespace
        {
            constexpr size_t MAX_FP32_BATCHED_PROJECTIONS = 8;
            std::atomic<uint32_t> g_rocm_fp32_gemm_workspace_slice_counter{0};
        }

        // =====================================================================
        // Constructor / Destructor
        // =====================================================================

        ROCmFloatingPointGemmKernel::ROCmFloatingPointGemmKernel(
            const TensorBase *weights,
            int rocm_device_id,
            Precision precision)
            : weights_(weights),
              d_weights_(nullptr),
              rocm_device_id_(rocm_device_id),
              precision_(precision),
              N_(0),
              K_(0),
              hipblas_kernel_(nullptr),
              slice_id_(g_rocm_fp32_gemm_workspace_slice_counter.fetch_add(1, std::memory_order_relaxed))
        {
            if (!weights)
            {
                throw std::runtime_error("[ROCmFloatingPointGemmKernel] Null weight tensor");
            }

            // Validate weight tensor type
            TensorType wt = weights->native_type();
            if (wt != TensorType::FP32 && wt != TensorType::FP16 && wt != TensorType::BF16)
            {
                throw std::runtime_error(
                    "[ROCmFloatingPointGemmKernel] Weight tensor must be FP32, FP16, or BF16, got: " +
                    std::to_string(static_cast<int>(wt)));
            }

            // Validate precision matches tensor type (or warn about emulation)
            if ((precision == Precision::FP32 && wt != TensorType::FP32) ||
                (precision == Precision::FP16 && wt != TensorType::FP16))
            {
                LOG_WARN("[ROCmFloatingPointGemmKernel] Precision mismatch: requested "
                         << static_cast<int>(precision) << " but tensor is "
                         << static_cast<int>(wt));
            }

            // Warn about BF16 emulation on MI50
            if (precision == Precision::BF16 || wt == TensorType::BF16)
            {
                LOG_WARN("[ROCmFloatingPointGemmKernel] BF16 will be emulated via FP32 - "
                         "MI50 (gfx906) has no native BF16 support");
            }

            // Get dimensions
            N_ = weights->rows(); // Output features
            K_ = weights->cols(); // Input features

            // Get device pointer (weights must already be on GPU)
            d_weights_ = weights->gpu_data_ptr();
            if (!d_weights_)
            {
                throw std::runtime_error(
                    "[ROCmFloatingPointGemmKernel] Weight tensor must be on GPU (call ensureOnDevice() first)");
            }

            // Get shared hipBLAS kernel from DeviceKernelCache (avoids per-tensor JIT overhead)
            DeviceId device = DeviceId::rocm(rocm_device_id_);
            hipblas_kernel_ = DeviceKernelCache::getKernel<HipBLASGemmKernel>(device, KernelType::BLAS_GEMM);

            LOG_DEBUG("[ROCmFloatingPointGemmKernel] Created for " << N_ << "x" << K_
                                                                   << " weights on ROCm device " << rocm_device_id_
                                                                   << " (using cached hipBLAS kernel)");
        }

        ROCmFloatingPointGemmKernel::ROCmFloatingPointGemmKernel(
            const void *d_weights,
            int N, int K,
            int rocm_device_id,
            Precision precision,
            std::shared_ptr<void> lifetime_owner)
            : weights_(nullptr),
              d_weights_(d_weights),
              rocm_device_id_(rocm_device_id),
              precision_(precision),
              N_(static_cast<size_t>(N)),
              K_(static_cast<size_t>(K)),
              hipblas_kernel_(nullptr),
              lifetime_owner_(std::move(lifetime_owner)),
              slice_id_(g_rocm_fp32_gemm_workspace_slice_counter.fetch_add(1, std::memory_order_relaxed))
        {
            if (!d_weights)
            {
                throw std::runtime_error("[ROCmFloatingPointGemmKernel] Null device weight pointer");
            }

            // Warn about BF16 emulation on MI50
            if (precision == Precision::BF16)
            {
                LOG_WARN("[ROCmFloatingPointGemmKernel] BF16 will be emulated via FP32 - "
                         "MI50 (gfx906) has no native BF16 support");
            }

            // Get shared hipBLAS kernel from DeviceKernelCache
            DeviceId device = DeviceId::rocm(rocm_device_id_);
            hipblas_kernel_ = DeviceKernelCache::getKernel<HipBLASGemmKernel>(device, KernelType::BLAS_GEMM);

            LOG_DEBUG("[ROCmFloatingPointGemmKernel] Created (raw ptr) for " << N_ << "x" << K_
                      << " weights on ROCm device " << rocm_device_id_
                      << " (using cached hipBLAS kernel)");
        }

        ROCmFloatingPointGemmKernel::~ROCmFloatingPointGemmKernel()
        {
            d_batch_A_ptrs_ = nullptr;
            d_batch_B_ptrs_ = nullptr;
            d_batch_C_ptrs_ = nullptr;
        }

        ROCmFloatingPointGemmKernel::ROCmFloatingPointGemmKernel(ROCmFloatingPointGemmKernel &&other) noexcept
            : weights_(other.weights_),
              d_weights_(other.d_weights_),
              rocm_device_id_(other.rocm_device_id_),
              precision_(other.precision_),
              N_(other.N_),
              K_(other.K_),
              hipblas_kernel_(other.hipblas_kernel_), // Just copy the shared pointer
              lifetime_owner_(std::move(other.lifetime_owner_)),
              workspace_(other.workspace_),
              slice_id_(other.slice_id_),
              d_batch_A_ptrs_(other.d_batch_A_ptrs_),
              d_batch_B_ptrs_(other.d_batch_B_ptrs_),
              d_batch_C_ptrs_(other.d_batch_C_ptrs_)
        {
            other.weights_ = nullptr;
            other.d_weights_ = nullptr;
            other.d_batch_A_ptrs_ = nullptr;
            other.d_batch_B_ptrs_ = nullptr;
            other.d_batch_C_ptrs_ = nullptr;
            other.workspace_ = nullptr;
            // Note: don't null other.hipblas_kernel_ - it's shared, not owned
        }

        ROCmFloatingPointGemmKernel &ROCmFloatingPointGemmKernel::operator=(ROCmFloatingPointGemmKernel &&other) noexcept
        {
            if (this != &other)
            {
                weights_ = other.weights_;
                d_weights_ = other.d_weights_;
                rocm_device_id_ = other.rocm_device_id_;
                precision_ = other.precision_;
                N_ = other.N_;
                K_ = other.K_;
                hipblas_kernel_ = other.hipblas_kernel_; // Just copy the shared pointer

                other.weights_ = nullptr;
                other.d_weights_ = nullptr;
                // Note: don't null other.hipblas_kernel_ - it's shared, not owned

                workspace_ = other.workspace_;
                slice_id_ = other.slice_id_;
                d_batch_A_ptrs_ = other.d_batch_A_ptrs_;
                d_batch_B_ptrs_ = other.d_batch_B_ptrs_;
                d_batch_C_ptrs_ = other.d_batch_C_ptrs_;
                other.d_batch_A_ptrs_ = nullptr;
                other.d_batch_B_ptrs_ = nullptr;
                other.d_batch_C_ptrs_ = nullptr;
                other.workspace_ = nullptr;
            }
            return *this;
        }

        // =====================================================================
        // ITensorGemm interface - multiply_tensor() PRIMARY ENTRY POINT
        // =====================================================================

        bool ROCmFloatingPointGemmKernel::multiply_tensor(
            const TensorBase *A, TensorBase *C,
            bool transpose_B,
            float alpha, float beta,
            const TensorBase *bias,
            const IMPIContext * /*mpi_ctx*/,
            int /*device_idx*/,
            DeviceWorkspaceManager *workspace,
            int activation_row_offset)
        {
            if (!A || !C)
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel::multiply_tensor] Null input or output tensor");
                return false;
            }

            // Get dimensions from tensors
            int m = static_cast<int>(A->rows());
            int n = static_cast<int>(N_);
            int k = static_cast<int>(K_);

            return multiply_tensor(A, C, m, n, k, transpose_B, alpha, beta, bias, nullptr, -1, workspace, activation_row_offset);
        }

        bool ROCmFloatingPointGemmKernel::multiply_tensor(
            const TensorBase *A, TensorBase *C,
            int m, int n, int k,
            bool transpose_B,
            float alpha, float beta,
            const TensorBase *bias,
            const IMPIContext * /*mpi_ctx*/,
            int /*device_idx*/,
            DeviceWorkspaceManager *workspace,
            int activation_row_offset)
        {
            ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::GEMM_ROCBLAS, static_cast<hipStream_t>(gpu_stream_));
            if (!A || !C)
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel::multiply_tensor] Null input or output tensor");
                return false;
            }

            // For now, only support FP32 I/O
            // TODO: Add BF16/FP16 activation support
            if (A->native_type() != TensorType::FP32)
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel::multiply_tensor] Only FP32 activations supported, got: "
                          << static_cast<int>(A->native_type()));
                return false;
            }

            if (C->native_type() != TensorType::FP32)
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel::multiply_tensor] Only FP32 output supported, got: "
                          << static_cast<int>(C->native_type()));
                return false;
            }

            // Get device pointers (caller must have data on GPU)
            const float *d_A = static_cast<const float *>(A->gpu_data_ptr());
            float *d_C = static_cast<float *>(C->gpu_data_ptr());

            if (!d_A || !d_C)
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel::multiply_tensor] A and C must be on GPU");
                return false;
            }

            // =================================================================
            // MAPPED OUTPUT REDIRECT: Detect host-mapped FP32 output memory.
            // Mapped memory (used for logits) causes PCIe-speed scattered writes
            // instead of HBM-speed writes. Redirect to HBM buffer, then bulk DMA.
            // =================================================================
            float *d_mapped_output = nullptr;
            if (C->isMapped())
            {
                const size_t needed = static_cast<size_t>(m) * n;
                const size_t needed_bytes = needed * sizeof(float);
                DeviceWorkspaceManager *effective_workspace = workspace ? workspace : workspace_;
                if (!validateROCmWorkspaceBinding(
                        effective_workspace,
                        rocm_device_id_,
                        "ROCmFloatingPointGemmKernel::multiply_tensor"))
                {
                    return false;
                }
                if (!effective_workspace->hasBuffer(GemmWorkspaceBuffers::ROCM_FP32_MAPPED_REDIRECT) ||
                    effective_workspace->getBufferSize(GemmWorkspaceBuffers::ROCM_FP32_MAPPED_REDIRECT) < needed_bytes)
                {
                    LOG_ERROR("[ROCmFloatingPointGemmKernel::multiply_tensor] Missing or undersized "
                              << "declared graph workspace buffer '"
                              << GemmWorkspaceBuffers::ROCM_FP32_MAPPED_REDIRECT
                              << "' for mapped-output redirect. required_bytes="
                              << needed_bytes);
                    return false;
                }
                d_mapped_output = d_C;
                d_C = static_cast<float *>(
                    effective_workspace->getBuffer(GemmWorkspaceBuffers::ROCM_FP32_MAPPED_REDIRECT));
                if (!d_C)
                {
                    LOG_ERROR("[ROCmFloatingPointGemmKernel::multiply_tensor] Mapped-output redirect workspace resolved to null");
                    return false;
                }
                static std::once_flag fp32gemm_mapped_once;
                std::call_once(fp32gemm_mapped_once, [&]()
                               { LOG_WARN("[ROCmFloatingPointGemmKernel] MAPPED REDIRECT: M=" << m << " N=" << n
                                                                                              << " mapped_ptr=" << d_mapped_output << " -> hbm=" << d_C
                                                                                              << " (" << (needed * 4 / 1024) << " KB)"); });
            }

            // Apply activation row offset
            if (activation_row_offset > 0)
            {
                d_A += static_cast<size_t>(activation_row_offset) * k;
            }

            // Extract bias pointer if provided
            const float *d_bias = nullptr;
            if (bias)
            {
                if (bias->native_type() != TensorType::FP32)
                {
                    LOG_ERROR("[ROCmFloatingPointGemmKernel::multiply_tensor] Bias must be FP32, got: "
                              << static_cast<int>(bias->native_type()));
                    return false;
                }
                d_bias = static_cast<const float *>(bias->gpu_data_ptr());
                if (!d_bias)
                {
                    LOG_ERROR("[ROCmFloatingPointGemmKernel::multiply_tensor] Bias tensor must be on GPU");
                    return false;
                }
            }

            /*
             * Qwen3.6 GDN alpha/beta verifier projections are tiny FP32 GEMMs
             * (M<=4, N<=64).  Grouped verifier rows use the local fixed-tree
             * tiny kernel so their reduction order is stable and graph
             * capturable.  Serial decode rows must use the same contract when a
             * graph workspace is bound; otherwise a hipBLAS M=1 reference can
             * differ by a few FP32 ULPs and the recurrent state amplifies that
             * into a token-level mismatch.
             */
            DeviceWorkspaceManager *effective_workspace = workspace ? workspace : workspace_;
            const bool can_use_tiny_decode_equivalent =
                precision_ == Precision::FP32 &&
                !d_bias &&
                transpose_B &&
                alpha == 1.0f &&
                beta == 0.0f &&
                m > 0 && m <= 4 &&
                n > 0 && n <= 64 &&
                k > 0 &&
                d_weights_ &&
                gpu_stream_ &&
                effective_workspace &&
                effective_workspace->hasBuffer(GemmWorkspaceBuffers::ROCM_FP32_BATCH_A_PTRS) &&
                effective_workspace->hasBuffer(GemmWorkspaceBuffers::ROCM_FP32_BATCH_B_PTRS) &&
                effective_workspace->hasBuffer(GemmWorkspaceBuffers::ROCM_FP32_BATCH_C_PTRS);

            if (can_use_tiny_decode_equivalent)
            {
                std::vector<const float *> a_ptrs{d_A};
                std::vector<const float *> b_ptrs{static_cast<const float *>(d_weights_)};
                std::vector<float *> c_ptrs{d_C};
                if (!stageBatchedPointers(a_ptrs, b_ptrs, c_ptrs, effective_workspace))
                    return false;

                bool success = rocmFp32_tiny_batched_projection(
                    d_batch_A_ptrs_,
                    d_batch_B_ptrs_,
                    d_batch_C_ptrs_,
                    m,
                    n,
                    k,
                    1,
                    rocm_device_id_,
                    gpu_stream_);
                if (!success)
                {
                    LOG_ERROR("[ROCmFloatingPointGemmKernel::multiply_tensor] Tiny FP32 single projection failed"
                              << " M=" << m << " N=" << n << " K=" << k);
                    return false;
                }

                if (d_mapped_output)
                {
                    hipError_t copy_status = hipMemcpyAsync(
                        d_mapped_output,
                        d_C,
                        static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float),
                        hipMemcpyDeviceToDevice,
                        static_cast<hipStream_t>(gpu_stream_));
                    if (copy_status != hipSuccess)
                    {
                        LOG_ERROR("[ROCmFloatingPointGemmKernel::multiply_tensor] Tiny FP32 mapped-output copy failed: "
                                  << hipGetErrorString(copy_status));
                        return false;
                    }
                }

                if (PerfStatsCollector::isEnabled())
                {
                    PerfStatsCollector::addCounter(
                        "kernel",
                        "rocm_fp32_tiny_single_projection_calls",
                        1.0,
                        "gemm",
                        "rocm:" + std::to_string(rocm_device_id_),
                        PerfStatsCollector::Tags{
                            {"m", std::to_string(m)},
                            {"n", std::to_string(n)},
                            {"k", std::to_string(k)}});
                }
                return true;
            }

            // Use fused GEMM+bias when bias is provided, otherwise use regular GEMM
            if (d_bias)
            {
                bool success = hipblas_kernel_->execute_with_bias(
                    d_A,                                    // d_A
                    static_cast<const float *>(d_weights_), // d_B
                    d_C,                                    // d_C
                    d_bias,                                 // d_bias
                    m, n, k,
                    false,       // transA = false
                    transpose_B, // transB
                    alpha, beta);
                // Bulk DMA from HBM redirect buffer to mapped output
                if (success && d_mapped_output)
                {
                    hipMemcpyAsync(d_mapped_output, d_C,
                                   static_cast<size_t>(m) * n * sizeof(float),
                                   hipMemcpyDeviceToDevice,
                                   static_cast<hipStream_t>(gpu_stream_));
                }
                return success;
            }
            else
            {
                bool success = hipblas_kernel_->execute(
                    d_A,                                    // d_A
                    static_cast<const float *>(d_weights_), // d_B
                    d_C,                                    // d_C
                    m, n, k,
                    false,       // transA = false
                    transpose_B, // transB
                    alpha, beta);
                // Bulk DMA from HBM redirect buffer to mapped output
                if (success && d_mapped_output)
                {
                    hipMemcpyAsync(d_mapped_output, d_C,
                                   static_cast<size_t>(m) * n * sizeof(float),
                                   hipMemcpyDeviceToDevice,
                                   static_cast<hipStream_t>(gpu_stream_));
                }
                return success;
            }
        }

        std::string ROCmFloatingPointGemmKernel::batchAPtrsBufferName() const
        {
            return GemmWorkspaceBuffers::ROCM_FP32_BATCH_A_PTRS;
        }

        std::string ROCmFloatingPointGemmKernel::batchBPtrsBufferName() const
        {
            return GemmWorkspaceBuffers::ROCM_FP32_BATCH_B_PTRS;
        }

        std::string ROCmFloatingPointGemmKernel::batchCPtrsBufferName() const
        {
            return GemmWorkspaceBuffers::ROCM_FP32_BATCH_C_PTRS;
        }

        bool ROCmFloatingPointGemmKernel::validateBatchedPointerWorkspace(
            DeviceWorkspaceManager *workspace,
            size_t required_count)
        {
            if (!validateROCmWorkspaceBinding(
                    workspace,
                    rocm_device_id_,
                    "ROCmFloatingPointGemmKernel::multiply_fused_tensor"))
            {
                return false;
            }

            const std::string a_name = batchAPtrsBufferName();
            const std::string b_name = batchBPtrsBufferName();
            const std::string c_name = batchCPtrsBufferName();
            const size_t required_bytes = required_count * sizeof(float *);
            if (!workspace->hasBuffer(a_name) || workspace->getBufferSize(a_name) < required_bytes ||
                !workspace->hasBuffer(b_name) || workspace->getBufferSize(b_name) < required_bytes ||
                !workspace->hasBuffer(c_name) || workspace->getBufferSize(c_name) < required_bytes)
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel::multiply_fused_tensor] Missing or undersized "
                          << "workspace pointer-array buffers. required_count=" << required_count
                          << " A=" << a_name << " B=" << b_name << " C=" << c_name);
                return false;
            }

            d_batch_A_ptrs_ = static_cast<const float **>(workspace->getBuffer(a_name));
            d_batch_B_ptrs_ = static_cast<const float **>(workspace->getBuffer(b_name));
            d_batch_C_ptrs_ = static_cast<float **>(workspace->getBuffer(c_name));
            return d_batch_A_ptrs_ && d_batch_B_ptrs_ && d_batch_C_ptrs_;
        }

        bool ROCmFloatingPointGemmKernel::stageBatchedPointers(
            const std::vector<const float *> &a_ptrs,
            const std::vector<const float *> &b_ptrs,
            const std::vector<float *> &c_ptrs,
            DeviceWorkspaceManager *workspace)
        {
            const size_t count = a_ptrs.size();
            if (count == 0 || b_ptrs.size() != count || c_ptrs.size() != count)
                return false;
            if (count > MAX_FP32_BATCHED_PROJECTIONS)
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel] Batched FP32 projection group exceeds workspace capacity: "
                          << count << " > " << MAX_FP32_BATCHED_PROJECTIONS);
                return false;
            }
            if (!validateBatchedPointerWorkspace(workspace, count))
                return false;

            if (!gpu_stream_)
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel] Batched FP32 projection requires an explicit non-null ROCm stream");
                return false;
            }

            if (!rocmFp32_stage_batched_projection_pointers(
                    d_batch_A_ptrs_,
                    d_batch_B_ptrs_,
                    d_batch_C_ptrs_,
                    a_ptrs.data(),
                    b_ptrs.data(),
                    c_ptrs.data(),
                    static_cast<int>(count),
                    rocm_device_id_,
                    gpu_stream_))
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel] Failed to stage batched FP32 projection pointers");
                return false;
            }
            return true;
        }

        bool ROCmFloatingPointGemmKernel::multiply_fused_tensor(
            const TensorBase *input,
            const std::vector<TensorProjectionDesc> &projections,
            int m, int k,
            const IMPIContext *mpi_ctx,
            DeviceWorkspaceManager *workspace)
        {
            if (!input || projections.empty())
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel::multiply_fused_tensor] Null input or empty projections");
                return false;
            }
            if (precision_ != Precision::FP32 || input->native_type() != TensorType::FP32)
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel::multiply_fused_tensor] Only FP32 activations/weights are supported");
                return false;
            }

            const float *d_input = static_cast<const float *>(input->gpu_data_ptr());
            if (!d_input)
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel::multiply_fused_tensor] Input has no GPU data");
                return false;
            }
            DeviceWorkspaceManager *effective_workspace = workspace ? workspace : workspace_;

            std::vector<bool> completed(projections.size(), false);
            for (size_t i = 0; i < projections.size(); ++i)
            {
                if (completed[i])
                    continue;

                const auto &seed = projections[i];
                if (!seed.kernel || !seed.output || seed.bias)
                    continue;

                auto *seed_kernel = dynamic_cast<ROCmFloatingPointGemmKernel *>(seed.kernel);
                auto *seed_output = dynamic_cast<FP32Tensor *>(seed.output);
                if (!seed_kernel || seed_kernel->precision_ != Precision::FP32 ||
                    seed_kernel->rocm_device_id_ != rocm_device_id_ ||
                    static_cast<int>(seed_kernel->K_) != k ||
                    !seed_output || seed_output->isMapped())
                {
                    continue;
                }

                std::vector<size_t> group_indices;
                group_indices.push_back(i);
                for (size_t j = i + 1; j < projections.size(); ++j)
                {
                    const auto &candidate = projections[j];
                    if (completed[j] || !candidate.kernel || !candidate.output || candidate.bias ||
                        candidate.n != seed.n)
                    {
                        continue;
                    }

                    auto *candidate_kernel = dynamic_cast<ROCmFloatingPointGemmKernel *>(candidate.kernel);
                    auto *candidate_output = dynamic_cast<FP32Tensor *>(candidate.output);
                    if (!candidate_kernel || candidate_kernel->precision_ != Precision::FP32 ||
                        candidate_kernel->rocm_device_id_ != rocm_device_id_ ||
                        static_cast<int>(candidate_kernel->K_) != k ||
                        !candidate_output || candidate_output->isMapped())
                    {
                        continue;
                    }
                    group_indices.push_back(j);
                }

                if (group_indices.size() < 2)
                    continue;

                std::vector<const float *> a_ptrs;
                std::vector<const float *> b_ptrs;
                std::vector<float *> c_ptrs;
                a_ptrs.reserve(group_indices.size());
                b_ptrs.reserve(group_indices.size());
                c_ptrs.reserve(group_indices.size());

                bool group_valid = true;
                for (size_t index : group_indices)
                {
                    const auto &projection = projections[index];
                    auto *projection_kernel = dynamic_cast<ROCmFloatingPointGemmKernel *>(projection.kernel);
                    auto *fp32_output = dynamic_cast<FP32Tensor *>(projection.output);
                    float *d_output = fp32_output
                                          ? static_cast<float *>(fp32_output->gpu_data_ptr())
                                          : nullptr;
                    if (!projection_kernel || !projection_kernel->d_weights_ || !d_output)
                    {
                        group_valid = false;
                        break;
                    }
                    a_ptrs.push_back(d_input);
                    b_ptrs.push_back(static_cast<const float *>(projection_kernel->d_weights_));
                    c_ptrs.push_back(d_output);
                }

                if (!group_valid)
                {
                    LOG_ERROR("[ROCmFloatingPointGemmKernel::multiply_fused_tensor] Invalid batched projection group");
                    return false;
                }

                if (!stageBatchedPointers(a_ptrs, b_ptrs, c_ptrs, effective_workspace))
                    return false;

                const int batch_count = static_cast<int>(group_indices.size());
                /*
                 * Verifier-sized GDN alpha/beta projections must use the same
                 * fixed reduction tree for M=1 and grouped M=2..4 rows.
                 * hipBLAS may legally choose different reduction schedules for
                 * those shapes; sub-ULP alpha/beta drift is then amplified by
                 * quantized projections and recurrence state.  The tiny kernel
                 * is workspace-backed, graph-capturable, and mirrors the CUDA
                 * decode-equivalent contract for these small projections.
                 */
                constexpr bool kTinyFP32ProjectionDecodeEquivalent = true;
                const bool use_tiny_fp32 =
                    kTinyFP32ProjectionDecodeEquivalent &&
                    m > 0 && m <= 4 &&
                    seed.n > 0 && seed.n <= 64 &&
                    k > 0 &&
                    batch_count > 0 &&
                    batch_count <= static_cast<int>(MAX_FP32_BATCHED_PROJECTIONS);

                if (use_tiny_fp32)
                {
                    if (!rocmFp32_tiny_batched_projection(
                            d_batch_A_ptrs_,
                            d_batch_B_ptrs_,
                            d_batch_C_ptrs_,
                            m,
                            seed.n,
                            k,
                            batch_count,
                            rocm_device_id_,
                            gpu_stream_))
                    {
                        LOG_ERROR("[ROCmFloatingPointGemmKernel::multiply_fused_tensor] Tiny FP32 batched projection failed"
                                  << " group_size=" << batch_count
                                  << " M=" << m << " N=" << seed.n << " K=" << k);
                        return false;
                    }

                    if (PerfStatsCollector::isEnabled())
                    {
                        PerfStatsCollector::addCounter(
                            "kernel",
                            "rocm_fp32_tiny_batched_projection_calls",
                            1.0,
                            "gemm",
                            "rocm:" + std::to_string(rocm_device_id_),
                            PerfStatsCollector::Tags{
                                {"m", std::to_string(m)},
                                {"n", std::to_string(seed.n)},
                                {"k", std::to_string(k)},
                                {"batch", std::to_string(batch_count)}});
                    }
                }
                else
                {
                    if (!hipblas_kernel_->execute_batched(
                            d_batch_A_ptrs_,
                            d_batch_B_ptrs_,
                            d_batch_C_ptrs_,
                            m,
                            seed.n,
                            k,
                            batch_count,
                            false,
                            true,
                            1.0f,
                            0.0f))
                    {
                        LOG_ERROR("[ROCmFloatingPointGemmKernel::multiply_fused_tensor] hipBLAS batched SGEMM failed"
                                  << " group_size=" << batch_count
                                  << " M=" << m << " N=" << seed.n << " K=" << k);
                        return false;
                    }

                    if (PerfStatsCollector::isEnabled())
                    {
                        PerfStatsCollector::addCounter(
                            "kernel",
                            "rocm_fp32_batched_projection_calls",
                            1.0,
                            "gemm",
                            "rocm:" + std::to_string(rocm_device_id_),
                            PerfStatsCollector::Tags{
                                {"m", std::to_string(m)},
                                {"n", std::to_string(seed.n)},
                                {"k", std::to_string(k)},
                                {"batch", std::to_string(batch_count)}});
                    }
                }

                for (size_t index : group_indices)
                    completed[index] = true;
            }

            for (size_t i = 0; i < projections.size(); ++i)
            {
                if (completed[i])
                    continue;

                const auto &projection = projections[i];
                if (!projection.kernel || !projection.output)
                    return false;

                if (!projection.kernel->multiply_tensor(
                        input,
                        projection.output,
                        m,
                        projection.n,
                        k,
                        true,
                        1.0f,
                        0.0f,
                        projection.bias,
                        mpi_ctx,
                        -1,
                        workspace))
                {
                    LOG_ERROR("[ROCmFloatingPointGemmKernel::multiply_fused_tensor] Projection failed for "
                              << (projection.name ? projection.name : "unnamed"));
                    return false;
                }
            }

            return true;
        }

        bool ROCmFloatingPointGemmKernel::multiply_fused_verifier_rows_decode_equivalent(
            const TensorBase *input,
            const std::vector<TensorProjectionDesc> &projections,
            int m, int k,
            const IMPIContext *mpi_ctx,
            DeviceWorkspaceManager *workspace)
        {
            if (m <= 1 || m > 4)
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel] grouped verifier projection requires M=2..4, got M="
                          << m);
                return false;
            }
            return multiply_fused_tensor(input, projections, m, k, mpi_ctx, workspace);
        }

        // =====================================================================
        // ITensorKernel interface
        // =====================================================================

        bool ROCmFloatingPointGemmKernel::supports_device(int device_idx) const
        {
            // This kernel only supports ROCm devices
            if (device_idx < 0)
            {
                return false; // CPU not supported
            }

            // Check against DeviceManager
            const auto &dm = DeviceManager::instance();
            if (static_cast<size_t>(device_idx) >= dm.devices().size())
            {
                return false;
            }

            const auto &dev = dm.devices()[device_idx];
            return (dev.type == ComputeBackendType::GPU_ROCM && dev.device_id == rocm_device_id_);
        }

        void ROCmFloatingPointGemmKernel::setGPUStream(void *stream)
        {
            gpu_stream_ = stream;
            if (hipblas_kernel_)
            {
                hipblas_kernel_->setStream(stream);
            }
        }

        // =====================================================================
        // IWorkspaceConsumer interface
        // =====================================================================

        WorkspaceRequirements ROCmFloatingPointGemmKernel::getWorkspaceRequirements(
            [[maybe_unused]] int m,
            [[maybe_unused]] int n,
            [[maybe_unused]] int k) const
        {
            WorkspaceRequirements reqs;
            if (precision_ != Precision::FP32)
            {
                return reqs;
            }

            const size_t pointer_array_bytes = MAX_FP32_BATCHED_PROJECTIONS * sizeof(float *);
            reqs.buffers.push_back({batchAPtrsBufferName(), pointer_array_bytes, 256, true});
            reqs.buffers.push_back({batchBPtrsBufferName(), pointer_array_bytes, 256, true});
            reqs.buffers.push_back({batchCPtrsBufferName(), pointer_array_bytes, 256, true});
            return reqs;
        }

        void ROCmFloatingPointGemmKernel::bindWorkspace(DeviceWorkspaceManager *workspace)
        {
            workspace_ = workspace;
            d_batch_A_ptrs_ = nullptr;
            d_batch_B_ptrs_ = nullptr;
            d_batch_C_ptrs_ = nullptr;
        }

        bool ROCmFloatingPointGemmKernel::hasWorkspace() const
        {
            return workspace_ != nullptr;
        }

        DeviceWorkspaceManager *ROCmFloatingPointGemmKernel::getWorkspace() const
        {
            return workspace_;
        }

        // =====================================================================
        // Activation-activation GEMM (not supported)
        // =====================================================================

        bool ROCmFloatingPointGemmKernel::multiply_activations(
            const float * /*A*/, const float * /*B*/, float * /*C*/,
            int /*m*/, int /*n*/, int /*k*/,
            bool /*transpose_B*/,
            float /*alpha*/, float /*beta*/,
            const IMPIContext * /*mpi_ctx*/,
            int /*device_idx*/)
        {
            LOG_ERROR("[ROCmFloatingPointGemmKernel] multiply_activations not supported - use dedicated attention kernel");
            return false;
        }

        bool ROCmFloatingPointGemmKernel::multiply_activations_strided(
            const float * /*A*/, const float * /*B*/, float * /*C*/,
            int /*m*/, int /*n*/, int /*k*/,
            int /*lda*/, int /*ldb*/, int /*ldc*/,
            bool /*transpose_B*/,
            float /*alpha*/, float /*beta*/,
            const IMPIContext * /*mpi_ctx*/,
            int /*device_idx*/)
        {
            LOG_ERROR("[ROCmFloatingPointGemmKernel] multiply_activations_strided not supported - use dedicated attention kernel");
            return false;
        }

        // =====================================================================
        // IKernelSnapshotCapable interface
        // =====================================================================

        KernelSnapshotInfo ROCmFloatingPointGemmKernel::getKernelSnapshotInfo() const
        {
            return KernelSnapshotInfo::gemm()
                .withInput("A", "input activations [m, k]", KernelBufferDtype::FP32)
                .withWeight("B", "weight matrix [n, k]", KernelBufferDtype::FP32)
                .withOutput("C", "output matrix [m, n]", KernelBufferDtype::FP32)
                .withScalar("precision", "computation precision (FP32/FP16/BF16)", KernelBufferDtype::INT32)
                .withScalar("N", "output features", KernelBufferDtype::INT32)
                .withScalar("K", "input features", KernelBufferDtype::INT32)
                .withScalar("rocm_device_id", "ROCm device ID", KernelBufferDtype::INT32);
        }

    } // namespace rocm
} // namespace llaminar2
