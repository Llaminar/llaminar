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
              d_batch_C_ptrs_(other.d_batch_C_ptrs_),
              cached_batch_A_ptrs_(std::move(other.cached_batch_A_ptrs_)),
              cached_batch_B_ptrs_(std::move(other.cached_batch_B_ptrs_)),
              cached_batch_C_ptrs_(std::move(other.cached_batch_C_ptrs_))
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
                cached_batch_A_ptrs_ = std::move(other.cached_batch_A_ptrs_);
                cached_batch_B_ptrs_ = std::move(other.cached_batch_B_ptrs_);
                cached_batch_C_ptrs_ = std::move(other.cached_batch_C_ptrs_);
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

        bool ROCmFloatingPointGemmKernel::uploadBatchedPointersIfChanged(
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

            if (cached_batch_A_ptrs_ == a_ptrs &&
                cached_batch_B_ptrs_ == b_ptrs &&
                cached_batch_C_ptrs_ == c_ptrs)
            {
                return true;
            }

            auto *stream = static_cast<hipStream_t>(gpu_stream_);
            hipError_t err = hipMemcpyAsync(
                d_batch_A_ptrs_, a_ptrs.data(), count * sizeof(float *),
                hipMemcpyHostToDevice, stream);
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel] Failed to upload batched A pointers: "
                          << hipGetErrorString(err));
                return false;
            }
            err = hipMemcpyAsync(
                d_batch_B_ptrs_, b_ptrs.data(), count * sizeof(float *),
                hipMemcpyHostToDevice, stream);
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel] Failed to upload batched B pointers: "
                          << hipGetErrorString(err));
                return false;
            }
            err = hipMemcpyAsync(
                d_batch_C_ptrs_, c_ptrs.data(), count * sizeof(float *),
                hipMemcpyHostToDevice, stream);
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel] Failed to upload batched C pointers: "
                          << hipGetErrorString(err));
                return false;
            }

            cached_batch_A_ptrs_ = a_ptrs;
            cached_batch_B_ptrs_ = b_ptrs;
            cached_batch_C_ptrs_ = c_ptrs;
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

                if (!uploadBatchedPointersIfChanged(a_ptrs, b_ptrs, c_ptrs, effective_workspace))
                    return false;

                if (!hipblas_kernel_->execute_batched(
                        d_batch_A_ptrs_,
                        d_batch_B_ptrs_,
                        d_batch_C_ptrs_,
                        m,
                        seed.n,
                        k,
                        static_cast<int>(group_indices.size()),
                        false,
                        true,
                        1.0f,
                        0.0f))
                {
                    LOG_ERROR("[ROCmFloatingPointGemmKernel::multiply_fused_tensor] hipBLAS batched SGEMM failed"
                              << " group_size=" << group_indices.size()
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
                            {"batch", std::to_string(group_indices.size())}});
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
            cached_batch_A_ptrs_.clear();
            cached_batch_B_ptrs_.clear();
            cached_batch_C_ptrs_.clear();
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
