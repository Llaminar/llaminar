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
 * 2. Retains a submission view borrowing context-owned hipBLAS handles, so
 *    expert retirement never tears down a library or drains unrelated streams.
 * 3. Handles tensor type introspection in multiply_tensor()
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "ROCmFloatingPointGemmKernel.h"
#include "kernels/common/FloatingPointVerifierLaunch.h"
#include "kernels/common/FloatingPointGemmWorkspaceABI.h"
#include "HipBLASGemmKernel.h"
#include "backends/ComputeBackend.h"   // DeviceManager
#include "backends/DeviceId.h"         // DeviceId for cache lookup
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

            // The lightweight view owns only bindings; the context owns BLAS.
            DeviceId device = DeviceId::rocm(rocm_device_id_);
            hipblas_kernel_ = std::make_unique<HipBLASGemmKernel>(device);

            LOG_DEBUG("[ROCmFloatingPointGemmKernel] Created for " << N_ << "x" << K_
                                                                   << " weights on ROCm device " << rocm_device_id_
                                                                   << " (borrowing device-context hipBLAS handles)");
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

            // Moving this view later must not move or destroy the library.
            DeviceId device = DeviceId::rocm(rocm_device_id_);
            hipblas_kernel_ = std::make_unique<HipBLASGemmKernel>(device);

            LOG_TRACE("[ROCmFloatingPointGemmKernel] Created (raw ptr) for " << N_ << "x" << K_
                      << " weights on ROCm device " << rocm_device_id_
                      << " (borrowing device-context hipBLAS handles)");
        }

        ROCmFloatingPointGemmKernel::~ROCmFloatingPointGemmKernel()
        {
            d_batch_A_ptrs_ = nullptr;
            d_batch_B_ptrs_ = nullptr;
            d_batch_C_ptrs_ = nullptr;
        }

        bool ROCmFloatingPointGemmKernel::exportContiguousFloatingPointWeights(
            ContiguousFloatingPointWeightDescriptor &out) const
        {
            TensorType type = TensorType::FP32;
            std::size_t element_bytes = sizeof(float);
            switch (precision_)
            {
            case Precision::FP16:
                type = TensorType::FP16;
                element_bytes = sizeof(std::uint16_t);
                break;
            case Precision::BF16:
                type = TensorType::BF16;
                element_bytes = sizeof(std::uint16_t);
                break;
            case Precision::FP32:
                break;
            }
            out = {
                .data = d_weights_,
                .type = type,
                .n = static_cast<int>(N_),
                .k = static_cast<int>(K_),
                .bytes = N_ * K_ * element_bytes,
            };
            return out.valid();
        }

        ROCmFloatingPointGemmKernel::ROCmFloatingPointGemmKernel(ROCmFloatingPointGemmKernel &&other) noexcept
            : weights_(other.weights_),
              d_weights_(other.d_weights_),
              rocm_device_id_(other.rocm_device_id_),
              precision_(other.precision_),
              N_(other.N_),
              K_(other.K_),
              hipblas_kernel_(std::move(other.hipblas_kernel_)),
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
                hipblas_kernel_ = std::move(other.hipblas_kernel_);

                other.weights_ = nullptr;
                other.d_weights_ = nullptr;

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
            if (!gpu_stream_)
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel::multiply_tensor] No explicit HIP stream is bound");
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
            if (!A || !C)
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel::multiply_tensor] Null input or output tensor");
                return false;
            }
            if (!gpu_stream_)
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel::multiply_tensor] No explicit HIP stream is bound");
                return false;
            }
            ROCM_KERNEL_PROFILE_SCOPE_STREAM(
                ROCmKernelType::GEMM_ROCBLAS,
                static_cast<hipStream_t>(gpu_stream_));

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
             * FP16/BF16 persistent floating weights still consume FP32 hidden
             * rows and produce FP32 projection outputs in the current graph
             * pipeline.  Runtime verifier rows must use the same fixed-order
             * device path as grouped publication; otherwise serial decode and
             * grouped MTP rows could diverge through different hipBLAS choices.
             */
            if (precision_ != Precision::FP32)
            {
                DeviceWorkspaceManager *effective_workspace = workspace ? workspace : workspace_;
                if (!transpose_B || alpha != 1.0f || beta != 0.0f || d_bias ||
                    m < 1 || n <= 0 || k <= 0 || !d_weights_ ||
                    !gpu_stream_ || !effective_workspace)
                {
                    LOG_ERROR("[ROCmFloatingPointGemmKernel::multiply_tensor] FP32x16 verifier projection requires "
                              << "transpose_B, alpha=1, beta=0, no bias, M>=1, stream, and workspace"
                              << " M=" << m << " N=" << n << " K=" << k
                              << " precision=" << static_cast<int>(precision_)
                              << " stream=" << gpu_stream_
                              << " workspace=" << effective_workspace);
                    return false;
                }

                std::vector<const float *> a_ptrs{d_A};
                std::vector<const float *> b_ptrs{reinterpret_cast<const float *>(d_weights_)};
                std::vector<float *> c_ptrs{d_C};
                if (!stageBatchedPointers(a_ptrs, b_ptrs, c_ptrs, effective_workspace))
                    return false;

                const int weight_dtype = (precision_ == Precision::BF16) ? 1 : 0;
                bool success = rocmFp32x16_batched_projection(
                    d_batch_A_ptrs_,
                    d_batch_B_ptrs_,
                    d_batch_C_ptrs_,
                    m,
                    n,
                    k,
                    1,
                    weight_dtype,
                    rocm_device_id_,
                    gpu_stream_,
                    VerifierKernelModeScope::rowsFor(m));
                if (success && d_mapped_output)
                {
                    const hipError_t copy_status = hipMemcpyAsync(
                        d_mapped_output,
                        d_C,
                        static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float),
                        hipMemcpyDeviceToDevice,
                        static_cast<hipStream_t>(gpu_stream_));
                    if (copy_status != hipSuccess)
                    {
                        LOG_ERROR("[ROCmFloatingPointGemmKernel::multiply_tensor] FP32x16 mapped-output copy failed: "
                                  << hipGetErrorString(copy_status));
                        return false;
                    }
                }
                if (success && PerfStatsCollector::isDomainEnabled("kernel"))
                {
                    PerfStatsCollector::addCounter(
                        "kernel",
                        "rocm_fp32x16_single_verifier_projection_calls",
                        1.0,
                        "gemm",
                        "rocm:" + std::to_string(rocm_device_id_),
                        PerfStatsCollector::Tags{
                            {"dtype", precision_ == Precision::BF16 ? "bf16" : "fp16"},
                            {"m", std::to_string(m)},
                            {"n", std::to_string(n)},
                            {"k", std::to_string(k)},
                            {"route", "fixed_order_fp32x16_single_projection"}});
                }
                return success;
            }

            /*
             * Qwen3.6 GDN alpha/beta projections are small-output FP32 GEMMs
             * (N<=64).  Use the local fixed-tree kernel for both decode and
             * prefill so their reduction order is stable and graph-capturable.
             * If this explicit route is requested, missing stream/workspace is
             * a hard failure instead of a quiet hipBLAS detour.
             */
            DeviceWorkspaceManager *effective_workspace = workspace ? workspace : workspace_;
            const bool requires_small_n_projection =
                precision_ == Precision::FP32 &&
                !d_bias &&
                transpose_B &&
                alpha == 1.0f &&
                beta == 0.0f &&
                m > 0 &&
                n > 0 && n <= 64 &&
                k > 0;

            const bool can_use_small_n_projection =
                requires_small_n_projection &&
                d_weights_ &&
                gpu_stream_ &&
                effective_workspace &&
                effective_workspace->hasBuffer(GemmWorkspaceBuffers::ROCM_FP32_BATCH_A_PTRS) &&
                effective_workspace->hasBuffer(GemmWorkspaceBuffers::ROCM_FP32_BATCH_B_PTRS) &&
                effective_workspace->hasBuffer(GemmWorkspaceBuffers::ROCM_FP32_BATCH_C_PTRS);

            if (can_use_small_n_projection)
            {
                std::vector<const float *> a_ptrs{d_A};
                std::vector<const float *> b_ptrs{static_cast<const float *>(d_weights_)};
                std::vector<float *> c_ptrs{d_C};
                if (!stageBatchedPointers(a_ptrs, b_ptrs, c_ptrs, effective_workspace))
                    return false;

                bool success = rocmFp32_small_n_batched_projection(
                    d_batch_A_ptrs_,
                    d_batch_B_ptrs_,
                    d_batch_C_ptrs_,
                    m,
                    n,
                    k,
                    1,
                    rocm_device_id_,
                    gpu_stream_,
                    VerifierKernelModeScope::rowsFor(m));
                if (!success)
                {
                    LOG_ERROR("[ROCmFloatingPointGemmKernel::multiply_tensor] Small-N FP32 single projection failed"
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
                        LOG_ERROR("[ROCmFloatingPointGemmKernel::multiply_tensor] Small-N FP32 mapped-output copy failed: "
                                  << hipGetErrorString(copy_status));
                        return false;
                    }
                }

                if (PerfStatsCollector::isDomainEnabled("kernel"))
                {
                    PerfStatsCollector::addCounter(
                        "kernel",
                        "rocm_fp32_small_n_single_projection_calls",
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
            if (requires_small_n_projection)
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel::multiply_tensor] Small-N FP32 projection requires "
                          << "an explicit stream, device weights, and declared batched pointer workspace"
                          << " M=" << m << " N=" << n << " K=" << k
                          << " stream=" << gpu_stream_
                          << " has_workspace=" << (effective_workspace != nullptr));
                return false;
            }

            // Use fused GEMM+bias when bias is provided, otherwise use regular GEMM
            hipblas_kernel_->bindWorkspace(effective_workspace);
            if (d_bias)
            {
                bool success = hipblas_kernel_->executeWithBiasOnStream(
                    ExplicitGPUStream{gpu_stream_},
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
                    (void)hipMemcpyAsync(d_mapped_output, d_C,
                                   static_cast<size_t>(m) * n * sizeof(float),
                                   hipMemcpyDeviceToDevice,
                                   static_cast<hipStream_t>(gpu_stream_));
                }
                return success;
            }
            else
            {
                bool success = hipblas_kernel_->executeOnStream(
                    ExplicitGPUStream{gpu_stream_},
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
                    (void)hipMemcpyAsync(d_mapped_output, d_C,
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
            if (count > floating_gemm_abi::kMaxBatchedProjections)
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel] Batched FP32 projection group exceeds workspace capacity: "
                          << count << " > "
                          << floating_gemm_abi::kMaxBatchedProjections);
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
            /*
             * FP16/BF16 weights use the fixed-order grouped projection kernel
             * for ordinary prefill as well as verifier rows. That kernel keeps
             * the K traversal and 16-bit conversion points independent of M,
             * which makes request padding and grouped publication byte-stable
             * without giving up one-launch grouped execution.
             */
            if (precision_ != Precision::FP32)
            {
                return multiply_fused_verifier_rows_decode_equivalent(
                    input, projections, m, k, mpi_ctx, workspace);
            }
            if (input->native_type() != TensorType::FP32)
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel::multiply_fused_tensor] FP32 activations are required");
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
                 * GDN alpha/beta projections use the same fixed reduction tree
                 * for decode and prefill.  hipBLAS may legally choose different
                 * reduction schedules across shapes, and ROCm graph capture has
                 * stricter library-call constraints after RCCL.  The small-N
                 * kernel is workspace-backed and graph-capturable.
                 */
                const bool use_small_n_fp32 =
                    m > 0 &&
                    seed.n > 0 && seed.n <= 64 &&
                    k > 0 &&
                    batch_count > 0 &&
                    batch_count <= static_cast<int>(
                        floating_gemm_abi::kMaxBatchedProjections);

                if (use_small_n_fp32)
                {
                    if (!rocmFp32_small_n_batched_projection(
                            d_batch_A_ptrs_,
                            d_batch_B_ptrs_,
                            d_batch_C_ptrs_,
                            m,
                            seed.n,
                            k,
                            batch_count,
                            rocm_device_id_,
                            gpu_stream_,
                    VerifierKernelModeScope::rowsFor(m)))
                    {
                        LOG_ERROR("[ROCmFloatingPointGemmKernel::multiply_fused_tensor] Small-N FP32 batched projection failed"
                                  << " group_size=" << batch_count
                                  << " M=" << m << " N=" << seed.n << " K=" << k);
                        return false;
                    }

                    if (PerfStatsCollector::isDomainEnabled("kernel"))
                    {
                        PerfStatsCollector::addCounter(
                            "kernel",
                            "rocm_fp32_small_n_batched_projection_calls",
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
                    hipblas_kernel_->bindWorkspace(effective_workspace);
                    if (!hipblas_kernel_->executeBatchedOnStream(
                            ExplicitGPUStream{gpu_stream_},
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

                if (PerfStatsCollector::isDomainEnabled("kernel"))
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
            if (m < 1)
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel] grouped verifier projection requires M>=1, got M="
                          << m);
                return false;
            }
            if (precision_ == Precision::FP32)
            {
                return multiply_fused_tensor(input, projections, m, k, mpi_ctx, workspace);
            }

            (void)mpi_ctx;
            if (!input || input->native_type() != TensorType::FP32 || projections.empty() || k <= 0)
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel] FP32x16 grouped verifier projection rejected: input="
                          << (input != nullptr)
                          << " input_type=" << (input ? static_cast<int>(input->native_type()) : -1)
                          << " projections=" << projections.size()
                          << " k=" << k);
                return false;
            }
            if (!gpu_stream_)
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel] FP32x16 grouped verifier projection requires an explicit ROCm stream");
                return false;
            }

            DeviceWorkspaceManager *effective_workspace = workspace ? workspace : workspace_;
            if (!validateROCmWorkspaceBinding(
                    effective_workspace,
                    rocm_device_id_,
                    "ROCmFloatingPointGemmKernel::multiply_fused_verifier_rows_decode_equivalent"))
            {
                return false;
            }

            const float *d_input = static_cast<const float *>(input->gpu_data_ptr());
            if (!d_input)
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel] FP32x16 grouped verifier projection input has no ROCm device data");
                return false;
            }

            const int weight_dtype = (precision_ == Precision::BF16) ? 1 : 0;
            const char *dtype_tag = (precision_ == Precision::BF16) ? "bf16" : "fp16";
            std::vector<bool> completed(projections.size(), false);

            for (size_t seed_index = 0; seed_index < projections.size(); ++seed_index)
            {
                if (completed[seed_index])
                    continue;

                const auto &seed = projections[seed_index];
                if (!seed.kernel || !seed.output || seed.bias || seed.n <= 0)
                {
                    LOG_ERROR("[ROCmFloatingPointGemmKernel] FP32x16 grouped verifier projection invalid seed at "
                              << seed_index);
                    return false;
                }
                if (seed.output->native_type() != TensorType::FP32 || seed.output->isMapped())
                {
                    LOG_ERROR("[ROCmFloatingPointGemmKernel] FP32x16 grouped verifier projection output must be unmapped FP32 at "
                              << seed_index);
                    return false;
                }

                std::vector<size_t> group_indices;
                group_indices.push_back(seed_index);
                for (size_t candidate_index = seed_index + 1; candidate_index < projections.size(); ++candidate_index)
                {
                    if (completed[candidate_index])
                        continue;
                    const auto &candidate = projections[candidate_index];
                    if (candidate.n == seed.n)
                        group_indices.push_back(candidate_index);
                }

                size_t group_offset = 0;
                while (group_offset < group_indices.size())
                {
                    const size_t group_count =
                        std::min(
                            floating_gemm_abi::kMaxBatchedProjections,
                            group_indices.size() - group_offset);
                    std::vector<const float *> a_ptrs(group_count, d_input);
                    std::vector<const float *> b_ptrs;
                    std::vector<float *> c_ptrs;
                    b_ptrs.reserve(group_count);
                    c_ptrs.reserve(group_count);

                    for (size_t local = 0; local < group_count; ++local)
                    {
                        const size_t projection_index = group_indices[group_offset + local];
                        const auto &projection = projections[projection_index];
                        auto *projection_kernel = dynamic_cast<ROCmFloatingPointGemmKernel *>(projection.kernel);
                        if (!projection_kernel ||
                            projection_kernel->precision_ != precision_ ||
                            projection_kernel->rocm_device_id_ != rocm_device_id_ ||
                            static_cast<int>(projection_kernel->K_) != k ||
                            projection_kernel->N_ != static_cast<size_t>(projection.n) ||
                            !projection_kernel->d_weights_)
                        {
                            LOG_ERROR("[ROCmFloatingPointGemmKernel] FP32x16 grouped verifier projection incompatible kernel at "
                                      << projection_index);
                            return false;
                        }

                        float *d_output = static_cast<float *>(projection.output->gpu_data_ptr());
                        if (!d_output)
                        {
                            LOG_ERROR("[ROCmFloatingPointGemmKernel] FP32x16 grouped verifier projection output has no ROCm data at "
                                      << projection_index);
                            return false;
                        }
                        b_ptrs.push_back(reinterpret_cast<const float *>(projection_kernel->d_weights_));
                        c_ptrs.push_back(d_output);
                    }

                    if (!stageBatchedPointers(a_ptrs, b_ptrs, c_ptrs, effective_workspace))
                        return false;

                    if (!rocmFp32x16_batched_projection(
                            d_batch_A_ptrs_,
                            d_batch_B_ptrs_,
                            d_batch_C_ptrs_,
                            m,
                            seed.n,
                            k,
                            static_cast<int>(group_count),
                            weight_dtype,
                            rocm_device_id_,
                            gpu_stream_,
                    VerifierKernelModeScope::rowsFor(m)))
                    {
                        LOG_ERROR("[ROCmFloatingPointGemmKernel] FP32x16 grouped verifier projection kernel failed"
                                  << " dtype=" << dtype_tag
                                  << " M=" << m
                                  << " N=" << seed.n
                                  << " K=" << k
                                  << " batch=" << group_count);
                        return false;
                    }

                    for (size_t local = 0; local < group_count; ++local)
                        completed[group_indices[group_offset + local]] = true;

                if (PerfStatsCollector::isDomainEnabled("kernel"))
                    {
                        PerfStatsCollector::addCounter(
                            "kernel",
                            "rocm_fp32x16_grouped_verifier_projection_calls",
                            1.0,
                            "gemm",
                            "rocm:" + std::to_string(rocm_device_id_),
                            PerfStatsCollector::Tags{
                                {"dtype", dtype_tag},
                                {"m", std::to_string(m)},
                                {"n", std::to_string(seed.n)},
                                {"k", std::to_string(k)},
                                {"projections", std::to_string(group_count)},
                                {"route", "fixed_order_fp32x16_batched_projection"}});
                    }

                    group_offset += group_count;
                }
            }

            return true;
        }

        bool ROCmFloatingPointGemmKernel::multiply_tensor_with_fused_swiglu(
            const TensorBase *gate,
            const TensorBase *up,
            TensorBase *output,
            int m, int n, int k,
            float alpha,
            float beta,
            DeviceWorkspaceManager *workspace)
        {
            return run_fixed_order_swiglu_down(
                gate,
                up,
                output,
                m,
                n,
                k,
                alpha,
                beta,
                workspace,
                /*verifier_grouped_call=*/false);
        }

        bool ROCmFloatingPointGemmKernel::multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
            const TensorBase *gate,
            const TensorBase *up,
            TensorBase *output,
            int m, int n, int k,
            float alpha,
            float beta,
            DeviceWorkspaceManager *workspace)
        {
            return run_fixed_order_swiglu_down(
                gate,
                up,
                output,
                m,
                n,
                k,
                alpha,
                beta,
                workspace,
                /*verifier_grouped_call=*/true);
        }

        bool ROCmFloatingPointGemmKernel::run_fixed_order_swiglu_down(
            const TensorBase *gate,
            const TensorBase *up,
            TensorBase *output,
            int m,
            int n,
            int k,
            float alpha,
            float beta,
            DeviceWorkspaceManager *workspace,
            bool verifier_grouped_call)
        {
            if (!gate || !up || !output || !d_weights_)
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel::run_fixed_order_swiglu_down] Null tensor or weights"
                          << " gate=" << (gate != nullptr)
                          << " up=" << (up != nullptr)
                          << " output=" << (output != nullptr)
                          << " weights=" << (d_weights_ != nullptr));
                return false;
            }
            if (m < 1 || n <= 0 || k <= 0 ||
                static_cast<size_t>(n) != N_ ||
                static_cast<size_t>(k) != K_)
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel::run_fixed_order_swiglu_down] "
                          << "requires M>=1 and dimensions matching the down weights"
                          << " M=" << m << " N=" << n << " K=" << k
                          << " weight_N=" << N_ << " weight_K=" << K_);
                return false;
            }
            if (alpha != 1.0f || beta != 0.0f)
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel::run_fixed_order_swiglu_down] "
                          << "only alpha=1,beta=0 is supported for decode-equivalent verifier rows"
                          << " alpha=" << alpha << " beta=" << beta);
                return false;
            }
            if (!gpu_stream_)
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel::run_fixed_order_swiglu_down] "
                          << "an explicit non-default ROCm stream is required");
                return false;
            }
            if (gate->native_type() != TensorType::FP32 ||
                up->native_type() != TensorType::FP32 ||
                output->native_type() != TensorType::FP32)
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel::run_fixed_order_swiglu_down] "
                          << "requires FP32 gate/up/output tensors"
                          << " gate_type=" << static_cast<int>(gate->native_type())
                          << " up_type=" << static_cast<int>(up->native_type())
                          << " output_type=" << static_cast<int>(output->native_type()));
                return false;
            }

            const float *d_gate = static_cast<const float *>(gate->gpu_data_ptr());
            const float *d_up = static_cast<const float *>(up->gpu_data_ptr());
            float *d_output = static_cast<float *>(output->gpu_data_ptr());
            if (!d_gate || !d_up || !d_output)
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel::run_fixed_order_swiglu_down] "
                          << "gate/up/output must be resident on the ROCm device");
                return false;
            }

            float *d_mapped_output = nullptr;
            if (output->isMapped())
            {
                const size_t needed_bytes =
                    static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float);
                DeviceWorkspaceManager *effective_workspace = workspace ? workspace : workspace_;
                if (!validateROCmWorkspaceBinding(
                        effective_workspace,
                        rocm_device_id_,
                        "ROCmFloatingPointGemmKernel::run_fixed_order_swiglu_down"))
                {
                    return false;
                }
                if (!effective_workspace->hasBuffer(GemmWorkspaceBuffers::ROCM_FP32_MAPPED_REDIRECT) ||
                    effective_workspace->getBufferSize(GemmWorkspaceBuffers::ROCM_FP32_MAPPED_REDIRECT) < needed_bytes)
                {
                    LOG_ERROR("[ROCmFloatingPointGemmKernel::run_fixed_order_swiglu_down] "
                              << "mapped output requires declared redirect workspace bytes="
                              << needed_bytes);
                    return false;
                }
                d_mapped_output = d_output;
                d_output = static_cast<float *>(
                    effective_workspace->getBuffer(GemmWorkspaceBuffers::ROCM_FP32_MAPPED_REDIRECT));
                if (!d_output)
                {
                    LOG_ERROR("[ROCmFloatingPointGemmKernel::run_fixed_order_swiglu_down] "
                              << "redirect workspace resolved to null");
                    return false;
                }
            }

            int weight_dtype = 0;
            const char *dtype_tag = "fp32";
            switch (precision_)
            {
            case Precision::FP32:
                weight_dtype = 0;
                dtype_tag = "fp32";
                break;
            case Precision::FP16:
                weight_dtype = 1;
                dtype_tag = "fp16";
                break;
            case Precision::BF16:
                weight_dtype = 2;
                dtype_tag = "bf16";
                break;
            }

            ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::GEMM_ROCBLAS, static_cast<hipStream_t>(gpu_stream_));
            bool success = rocmFloating_swiglu_down_projection(
                d_gate,
                d_up,
                d_weights_,
                d_output,
                m,
                n,
                k,
                weight_dtype,
                rocm_device_id_,
                gpu_stream_,
                    VerifierKernelModeScope::rowsFor(m));

            if (success && d_mapped_output)
            {
                const hipError_t copy_status = hipMemcpyAsync(
                    d_mapped_output,
                    d_output,
                    static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float),
                    hipMemcpyDeviceToDevice,
                    static_cast<hipStream_t>(gpu_stream_));
                if (copy_status != hipSuccess)
                {
                    LOG_ERROR("[ROCmFloatingPointGemmKernel::run_fixed_order_swiglu_down] "
                              << "mapped-output copy failed: " << hipGetErrorString(copy_status));
                    return false;
                }
            }

            if (success && PerfStatsCollector::isDomainEnabled("kernel"))
            {
                PerfStatsCollector::addCounter(
                    "kernel",
                    verifier_grouped_call
                        ? "rocm_floating_grouped_verifier_swiglu_down_calls"
                        : "rocm_floating_fused_swiglu_down_calls",
                    1.0,
                    "gemm",
                    "rocm:" + std::to_string(rocm_device_id_),
                    PerfStatsCollector::Tags{
                        {"dtype", dtype_tag},
                        {"m", std::to_string(m)},
                        {"n", std::to_string(n)},
                        {"k", std::to_string(k)},
                        {"route", "fixed_order_floating_swiglu_down"},
                        {"verifier", verifier_grouped_call ? "1" : "0"}});
            }
            return success;
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

        void ROCmFloatingPointGemmKernel::bindGPUStream(ExplicitGPUStream stream)
        {
            gpu_stream_ = stream.get();
            /* The shared hipBLAS handle is bound atomically with each launch. */
        }

        void ROCmFloatingPointGemmKernel::clearGPUStreamBinding()
        {
            gpu_stream_ = nullptr;
        }

        // =====================================================================
        // IWorkspaceConsumer interface
        // =====================================================================

        WorkspaceRequirements ROCmFloatingPointGemmKernel::getWorkspaceRequirements(
            int m, int n, int k) const
        {
            // The adapter must publish the complete low-level BOM, not merely
            // its pointer tables. The memory authority admits/materializes it.
            WorkspaceRequirements reqs = hipblas_kernel_->getWorkspaceRequirements(m, n, k);

            const size_t pointer_array_bytes =
                floating_gemm_abi::kMaxBatchedProjections *
                sizeof(float *);
            reqs.buffers.push_back({batchAPtrsBufferName(), pointer_array_bytes, 256, true});
            reqs.buffers.push_back({batchBPtrsBufferName(), pointer_array_bytes, 256, true});
            reqs.buffers.push_back({batchCPtrsBufferName(), pointer_array_bytes, 256, true});
            return reqs;
        }

        void ROCmFloatingPointGemmKernel::bindWorkspace(DeviceWorkspaceManager *workspace)
        {
            workspace_ = workspace;
            hipblas_kernel_->bindWorkspace(workspace);
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
