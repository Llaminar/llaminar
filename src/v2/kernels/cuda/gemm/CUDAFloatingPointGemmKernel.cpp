/**
 * @file CUDAFloatingPointGemmKernel.cpp
 * @brief ITensorGemm adapter implementation for cuBLAS FP32/FP16/BF16 GEMM
 *
 * This is the C++ adapter that wraps CuBLASGemmKernel. It implements the full
 * ITensorGemm interface and can be compiled with the regular C++ compiler
 * (not nvcc), avoiding MPI/TensorKernels.h compilation issues.
 *
 * **Design**: The adapter:
 * 1. Implements ITensorGemm (includes IMPIContext, etc.)
 * 2. Holds a CuBLASGemmKernel* that does the actual CUDA work
 * 3. Handles tensor type introspection in multiply_tensor()
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "CUDAFloatingPointGemmKernel.h"
#include "CuBLASGemmKernel.h"
#include "backends/ComputeBackend.h" // DeviceManager
#include "tensors/Tensors.h"         // FP32Tensor, BF16Tensor, FP16Tensor
#include "tensors/KernelSnapshotInfo.h"
#include "utils/Logger.h"
#include "utils/CUDAKernelProfiler.h"
#include "utils/PerfStatsCollector.h"

#include <stdexcept>
#include <mutex>
#include <vector>
#include <algorithm>

// CUDA memory operations (implemented in CUDAQuantisedGemmKernel_CUTLASS.cu)
extern "C"
{
    bool cudaQuantGemm_copyDeviceToDeviceAsync(float *d_dst, const float *d_src, size_t count, int cuda_device_id, void *stream);
    bool cudaFp32_stage_batched_projection_pointers(
        const float **d_A_array,
        const float **d_B_array,
        float **d_C_array,
        const float *const *h_A_ptrs,
        const float *const *h_B_ptrs,
        float *const *h_C_ptrs,
        int batch_count,
        int device_id,
        void *stream);
    bool cudaFp32_tiny_batched_projection(
        const float *const *d_A_array,
        const float *const *d_B_array,
        float *const *d_C_array,
        int M,
        int N,
        int K,
        int batch_count,
        int device_id,
        void *stream);
    bool cudaFp32x16_tiny_batched_projection(
        const float *const *d_A_array,
        const float *const *d_B_array,
        float *const *d_C_array,
        int M,
        int N,
        int K,
        int batch_count,
        int weight_dtype,
        int device_id,
        void *stream);
    bool cudaFloating_swiglu_down_projection(
        const float *d_gate,
        const float *d_up,
        const void *d_weights,
        float *d_output,
        int M,
        int N,
        int K,
        int weight_dtype,
        int device_id,
        void *stream);
}

namespace llaminar2
{
    namespace cuda
    {

        // =====================================================================
        // Constructor / Destructor
        // =====================================================================

        CUDAFloatingPointGemmKernel::CUDAFloatingPointGemmKernel(
            const TensorBase *weights,
            int cuda_device_id,
            Precision precision)
            : weights_(weights),
              d_weights_(nullptr),
              cuda_device_id_(cuda_device_id),
              precision_(precision),
              N_(0),
              K_(0),
              cublas_kernel_(nullptr)
        {
            if (!weights)
            {
                throw std::runtime_error("[CUDAFloatingPointGemmKernel] Null weight tensor");
            }

            // Validate weight tensor type
            TensorType wt = weights->native_type();
            if (wt != TensorType::FP32 && wt != TensorType::FP16 && wt != TensorType::BF16)
            {
                throw std::runtime_error(
                    "[CUDAFloatingPointGemmKernel] Weight tensor must be FP32, FP16, or BF16, got: " +
                    std::to_string(static_cast<int>(wt)));
            }

            // Validate precision matches tensor type
            if ((precision == Precision::FP32 && wt != TensorType::FP32) ||
                (precision == Precision::FP16 && wt != TensorType::FP16) ||
                (precision == Precision::BF16 && wt != TensorType::BF16))
            {
                LOG_WARN("[CUDAFloatingPointGemmKernel] Precision mismatch: requested "
                         << static_cast<int>(precision) << " but tensor is "
                         << static_cast<int>(wt));
            }

            // Get dimensions
            N_ = weights->rows(); // Output features
            K_ = weights->cols(); // Input features

            // Get device pointer (weights must already be on GPU)
            // Note: Use gpu_data_ptr() or current_home_dm_device_index() to check actual location
            // home_dm_device_index() returns creation-time device, not current location
            d_weights_ = weights->gpu_data_ptr();
            if (!d_weights_)
            {
                throw std::runtime_error(
                    "[CUDAFloatingPointGemmKernel] Weight tensor must be on GPU (call ensureOnDevice() first)");
            }

            // Create underlying cuBLAS kernel
            CuBLASGemmKernel::Precision cublas_precision;
            switch (precision)
            {
            case Precision::FP32:
                cublas_precision = CuBLASGemmKernel::Precision::FP32;
                break;
            case Precision::FP16:
                cublas_precision = CuBLASGemmKernel::Precision::FP16;
                break;
            case Precision::BF16:
                cublas_precision = CuBLASGemmKernel::Precision::BF16;
                break;
            default:
                cublas_precision = CuBLASGemmKernel::Precision::FP32;
            }

            cublas_kernel_ = std::make_unique<CuBLASGemmKernel>(cuda_device_id_, cublas_precision);

            LOG_DEBUG("[CUDAFloatingPointGemmKernel] Created for " << N_ << "x" << K_
                                                                   << " weights on CUDA device " << cuda_device_id_);
        }

        CUDAFloatingPointGemmKernel::CUDAFloatingPointGemmKernel(
            const void *d_weights,
            int N, int K,
            int cuda_device_id,
            Precision precision,
            std::shared_ptr<void> lifetime_owner)
            : weights_(nullptr),
              d_weights_(d_weights),
              cuda_device_id_(cuda_device_id),
              precision_(precision),
              N_(static_cast<size_t>(N)),
              K_(static_cast<size_t>(K)),
              cublas_kernel_(nullptr),
              lifetime_owner_(std::move(lifetime_owner))
        {
            if (!d_weights)
            {
                throw std::runtime_error("[CUDAFloatingPointGemmKernel] Null device weight pointer");
            }

            // Create underlying cuBLAS kernel
            CuBLASGemmKernel::Precision cublas_precision;
            switch (precision)
            {
            case Precision::FP32:
                cublas_precision = CuBLASGemmKernel::Precision::FP32;
                break;
            case Precision::FP16:
                cublas_precision = CuBLASGemmKernel::Precision::FP16;
                break;
            case Precision::BF16:
                cublas_precision = CuBLASGemmKernel::Precision::BF16;
                break;
            default:
                cublas_precision = CuBLASGemmKernel::Precision::FP32;
            }

            cublas_kernel_ = std::make_unique<CuBLASGemmKernel>(cuda_device_id_, cublas_precision);

            LOG_TRACE("[CUDAFloatingPointGemmKernel] Created (raw ptr) for " << N_ << "x" << K_
                      << " weights on CUDA device " << cuda_device_id_);
        }

        CUDAFloatingPointGemmKernel::~CUDAFloatingPointGemmKernel()
        {
        }

        CUDAFloatingPointGemmKernel::CUDAFloatingPointGemmKernel(CUDAFloatingPointGemmKernel &&other) noexcept
            : weights_(other.weights_),
              d_weights_(other.d_weights_),
              cuda_device_id_(other.cuda_device_id_),
              precision_(other.precision_),
              N_(other.N_),
              K_(other.K_),
              cublas_kernel_(std::move(other.cublas_kernel_)),
              lifetime_owner_(std::move(other.lifetime_owner_)),
              bound_workspace_(other.bound_workspace_)
        {
            other.weights_ = nullptr;
            other.d_weights_ = nullptr;
            other.bound_workspace_ = nullptr;
        }

        CUDAFloatingPointGemmKernel &CUDAFloatingPointGemmKernel::operator=(CUDAFloatingPointGemmKernel &&other) noexcept
        {
            if (this != &other)
            {
                weights_ = other.weights_;
                d_weights_ = other.d_weights_;
                cuda_device_id_ = other.cuda_device_id_;
                precision_ = other.precision_;
                N_ = other.N_;
                K_ = other.K_;
                cublas_kernel_ = std::move(other.cublas_kernel_);
                bound_workspace_ = other.bound_workspace_;

                other.weights_ = nullptr;
                other.d_weights_ = nullptr;
                other.bound_workspace_ = nullptr;
            }
            return *this;
        }

        // =====================================================================
        // ITensorGemm interface - multiply_tensor() PRIMARY ENTRY POINT
        // =====================================================================

        bool CUDAFloatingPointGemmKernel::multiply_tensor(
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
                LOG_ERROR("[CUDAFloatingPointGemmKernel::multiply_tensor] Null input or output tensor");
                return false;
            }

            // Get dimensions from tensors
            int m = static_cast<int>(A->rows());
            int n = static_cast<int>(N_);
            int k = static_cast<int>(K_);

            return multiply_tensor(A, C, m, n, k, transpose_B, alpha, beta, bias, nullptr, -1, workspace, activation_row_offset);
        }

        bool CUDAFloatingPointGemmKernel::multiply_tensor(
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
            (void)workspace; // TODO: Use workspace for intermediate allocations
            if (!A || !C)
            {
                LOG_ERROR("[CUDAFloatingPointGemmKernel::multiply_tensor] Null input or output tensor");
                return false;
            }

            // For now, only support FP32 I/O
            // TODO: Add BF16/FP16 activation support
            if (A->native_type() != TensorType::FP32)
            {
                LOG_ERROR("[CUDAFloatingPointGemmKernel::multiply_tensor] Only FP32 activations supported, got: "
                          << static_cast<int>(A->native_type()));
                return false;
            }

            if (C->native_type() != TensorType::FP32)
            {
                LOG_ERROR("[CUDAFloatingPointGemmKernel::multiply_tensor] Only FP32 output supported, got: "
                          << static_cast<int>(C->native_type()));
                return false;
            }

            // Get device pointers (caller must have data on GPU)
            const float *d_A = static_cast<const float *>(A->gpu_data_ptr());
            float *d_C = static_cast<float *>(C->gpu_data_ptr());

            if (!d_A || !d_C)
            {
                LOG_ERROR("[CUDAFloatingPointGemmKernel::multiply_tensor] A and C must be on GPU");
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
                DeviceWorkspaceManager *effective_workspace = workspace ? workspace : bound_workspace_;
                if (!effective_workspace ||
                    !effective_workspace->hasBuffer(GemmWorkspaceBuffers::CUDA_FP32_MAPPED_REDIRECT) ||
                    effective_workspace->getBufferSize(GemmWorkspaceBuffers::CUDA_FP32_MAPPED_REDIRECT) < needed_bytes)
                {
                    LOG_ERROR("[CUDAFloatingPointGemmKernel::multiply_tensor] Missing or undersized "
                              << "declared graph workspace buffer '"
                              << GemmWorkspaceBuffers::CUDA_FP32_MAPPED_REDIRECT
                              << "' for mapped-output redirect. required_bytes="
                              << needed_bytes);
                    return false;
                }
                d_mapped_output = d_C;
                d_C = static_cast<float *>(
                    effective_workspace->getBuffer(GemmWorkspaceBuffers::CUDA_FP32_MAPPED_REDIRECT));
                if (!d_C)
                {
                    LOG_ERROR("[CUDAFloatingPointGemmKernel::multiply_tensor] Mapped-output redirect workspace resolved to null");
                    return false;
                }
                static std::once_flag fp32gemm_mapped_once;
                std::call_once(fp32gemm_mapped_once, [&]()
                               { LOG_WARN("[CUDAFloatingPointGemmKernel] MAPPED REDIRECT: M=" << m << " N=" << n
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
                    LOG_ERROR("[CUDAFloatingPointGemmKernel::multiply_tensor] Bias must be FP32, got: "
                              << static_cast<int>(bias->native_type()));
                    return false;
                }
                d_bias = static_cast<const float *>(bias->gpu_data_ptr());
                if (!d_bias)
                {
                    LOG_ERROR("[CUDAFloatingPointGemmKernel::multiply_tensor] Bias tensor must be on GPU");
                    return false;
                }
            }

            /*
             * FP16/BF16 floating weights currently pair with FP32 hidden rows
             * and FP32 projection outputs in the graph pipeline.  For verifier
             * runtime-M rows, use the same fixed-order custom kernel as grouped
             * publication so serial decode and grouped verifier rows share one
             * device-resident numerical contract.  Larger non-verifier FP16/BF16
             * GEMMs are intentionally not routed through the FP32 cuBLAS adapter:
             * using a float* cast for 16-bit weights would be silent corruption.
             */
            if (precision_ != Precision::FP32)
            {
                DeviceWorkspaceManager *effective_workspace = workspace ? workspace : bound_workspace_;
                if (!transpose_B || alpha != 1.0f || beta != 0.0f || d_bias ||
                    m < 1 || n <= 0 || k <= 0 || !d_weights_ ||
                    !gpu_stream_ || !effective_workspace)
                {
                    LOG_ERROR("[CUDAFloatingPointGemmKernel::multiply_tensor] FP32x16 verifier projection requires "
                              << "transpose_B, alpha=1, beta=0, no bias, M>=1, stream, and workspace"
                              << " M=" << m << " N=" << n << " K=" << k
                              << " precision=" << static_cast<int>(precision_)
                              << " stream=" << gpu_stream_
                              << " workspace=" << effective_workspace);
                    return false;
                }

                auto *d_A_array = static_cast<const float **>(
                    effective_workspace->getBuffer(GemmWorkspaceBuffers::CUDA_FP32_BATCH_A_PTRS));
                auto *d_B_array = static_cast<const float **>(
                    effective_workspace->getBuffer(GemmWorkspaceBuffers::CUDA_FP32_BATCH_B_PTRS));
                auto *d_C_array = static_cast<float **>(
                    effective_workspace->getBuffer(GemmWorkspaceBuffers::CUDA_FP32_BATCH_C_PTRS));
                if (!d_A_array || !d_B_array || !d_C_array)
                {
                    LOG_ERROR("[CUDAFloatingPointGemmKernel::multiply_tensor] FP32x16 verifier projection missing pointer workspace");
                    return false;
                }

                const float *a_ptrs[1] = {d_A};
                const float *b_ptrs[1] = {reinterpret_cast<const float *>(d_weights_)};
                float *c_ptrs[1] = {d_C};
                if (!cudaFp32_stage_batched_projection_pointers(
                        d_A_array,
                        d_B_array,
                        d_C_array,
                        a_ptrs,
                        b_ptrs,
                        c_ptrs,
                        1,
                        cuda_device_id_,
                        gpu_stream_))
                {
                    LOG_ERROR("[CUDAFloatingPointGemmKernel::multiply_tensor] FP32x16 verifier projection failed to stage pointers");
                    return false;
                }

                const int weight_dtype = (precision_ == Precision::BF16) ? 1 : 0;
                bool success = cudaFp32x16_tiny_batched_projection(
                    d_A_array,
                    d_B_array,
                    d_C_array,
                    m,
                    n,
                    k,
                    1,
                    weight_dtype,
                    cuda_device_id_,
                    gpu_stream_);
                if (success && d_mapped_output)
                {
                    success = cudaQuantGemm_copyDeviceToDeviceAsync(
                        d_mapped_output,
                        d_C,
                        static_cast<size_t>(m) * static_cast<size_t>(n),
                        cuda_device_id_,
                        gpu_stream_);
                }
                if (success && PerfStatsCollector::isEnabled())
                {
                    PerfStatsCollector::addCounter(
                        "kernel",
                        "cuda_fp32x16_single_verifier_projection_calls",
                        1.0,
                        "gemm",
                        "cuda:" + std::to_string(cuda_device_id_),
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
             * Decode-sized FP32 stage projections need the same treatment as
             * FP16/BF16 above: cuBLAS is free to choose different reduction
             * schedules for serial and grouped shapes, which is legal GEMM behavior but
             * not legal for MTP verifier rows that may publish live state.
             *
             * When a graph/stage workspace is bound, route small FP32 rows
             * through the fixed-order grouped projection kernel even for the
             * serial M=1 decode call.  The grouped verifier entry point below
             * uses the same kernel, so row grouping changes launch geometry but
             * not per-output accumulation order.  Standalone tests that call
             * this adapter without graph workspace still exercise the ordinary
             * cuBLAS path; production stage execution always binds workspace.
             */
            DeviceWorkspaceManager *effective_workspace = workspace ? workspace : bound_workspace_;
            const bool can_use_fixed_order_fp32_decode =
                transpose_B &&
                alpha == 1.0f &&
                beta == 0.0f &&
                !d_bias &&
                m == 1 &&
                n > 0 &&
                k > 0 &&
                d_weights_ &&
                gpu_stream_ &&
                effective_workspace;
            if (can_use_fixed_order_fp32_decode)
            {
                auto *d_A_array = static_cast<const float **>(
                    effective_workspace->getBuffer(GemmWorkspaceBuffers::CUDA_FP32_BATCH_A_PTRS));
                auto *d_B_array = static_cast<const float **>(
                    effective_workspace->getBuffer(GemmWorkspaceBuffers::CUDA_FP32_BATCH_B_PTRS));
                auto *d_C_array = static_cast<float **>(
                    effective_workspace->getBuffer(GemmWorkspaceBuffers::CUDA_FP32_BATCH_C_PTRS));
                if (!d_A_array || !d_B_array || !d_C_array)
                {
                    LOG_ERROR("[CUDAFloatingPointGemmKernel::multiply_tensor] FP32 decode-equivalent projection missing pointer workspace");
                    return false;
                }

                const float *a_ptrs[1] = {d_A};
                const float *b_ptrs[1] = {static_cast<const float *>(d_weights_)};
                float *c_ptrs[1] = {d_C};
                if (!cudaFp32_stage_batched_projection_pointers(
                        d_A_array,
                        d_B_array,
                        d_C_array,
                        a_ptrs,
                        b_ptrs,
                        c_ptrs,
                        1,
                        cuda_device_id_,
                        gpu_stream_))
                {
                    LOG_ERROR("[CUDAFloatingPointGemmKernel::multiply_tensor] FP32 decode-equivalent projection failed to stage pointers");
                    return false;
                }

                CUDA_KERNEL_PROFILE_SCOPE_STREAM(CUDAKernelType::GEMM_CUBLAS, gpu_stream_);
                bool success = cudaFp32_tiny_batched_projection(
                    d_A_array,
                    d_B_array,
                    d_C_array,
                    m,
                    n,
                    k,
                    1,
                    cuda_device_id_,
                    gpu_stream_);
                if (success && d_mapped_output)
                {
                    success = cudaQuantGemm_copyDeviceToDeviceAsync(
                        d_mapped_output,
                        d_C,
                        static_cast<size_t>(m) * static_cast<size_t>(n),
                        cuda_device_id_,
                        gpu_stream_);
                }
                if (success && PerfStatsCollector::isEnabled())
                {
                    PerfStatsCollector::addCounter(
                        "kernel",
                        "cuda_fp32_single_verifier_projection_calls",
                        1.0,
                        "gemm",
                        "cuda:" + std::to_string(cuda_device_id_),
                        PerfStatsCollector::Tags{
                            {"dtype", "fp32"},
                            {"m", std::to_string(m)},
                            {"n", std::to_string(n)},
                            {"k", std::to_string(k)},
                            {"route", "fixed_order_fp32_single_projection"}});
                }
                return success;
            }

            // Use fused GEMM+bias when bias is provided, otherwise use regular GEMM
            if (d_bias)
            {
                CUDA_KERNEL_PROFILE_SCOPE_STREAM(CUDAKernelType::GEMM_CUBLAS, gpu_stream_);
                bool success = cublas_kernel_->execute_with_bias(
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
                    success = cudaQuantGemm_copyDeviceToDeviceAsync(
                        d_mapped_output, d_C,
                        static_cast<size_t>(m) * n,
                        cuda_device_id_, gpu_stream_);
                }
                return success;
            }
            else
            {
                CUDA_KERNEL_PROFILE_SCOPE_STREAM(CUDAKernelType::GEMM_CUBLAS, gpu_stream_);
                bool success = cublas_kernel_->execute(
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
                    success = cudaQuantGemm_copyDeviceToDeviceAsync(
                        d_mapped_output, d_C,
                        static_cast<size_t>(m) * n,
                        cuda_device_id_, gpu_stream_);
                }
                return success;
            }
        }

        bool CUDAFloatingPointGemmKernel::multiply_fused_tensor(
            const TensorBase *input,
            const std::vector<TensorProjectionDesc> &projections,
            int m, int k,
            const IMPIContext * /*mpi_ctx*/,
            DeviceWorkspaceManager *workspace)
        {
            if (!input || projections.empty())
            {
                LOG_ERROR("[CUDAFloatingPointGemmKernel::multiply_fused_tensor] Null input or empty projections");
                return false;
            }
            /*
             * FP16/BF16 weights use the fixed-order grouped projection kernel
             * for ordinary prefill as well as verifier rows. This is the
             * production batch-invariant implementation, not a serial replay:
             * one launch evaluates the projection group with an M-independent
             * reduction order.
             */
            if (precision_ != Precision::FP32)
            {
                return multiply_fused_verifier_rows_decode_equivalent(
                    input, projections, m, k, nullptr, workspace);
            }
            if (input->native_type() != TensorType::FP32)
            {
                LOG_ERROR("[CUDAFloatingPointGemmKernel::multiply_fused_tensor] Only FP32 activations are supported");
                return false;
            }
            if (m <= 0 || k <= 0 || static_cast<size_t>(k) != K_)
            {
                LOG_ERROR("[CUDAFloatingPointGemmKernel::multiply_fused_tensor] Invalid dimensions: m="
                          << m << " k=" << k << " expected_k=" << K_);
                return false;
            }
            if (!gpu_stream_)
            {
                LOG_ERROR("[CUDAFloatingPointGemmKernel::multiply_fused_tensor] No explicit CUDA stream is bound");
                return false;
            }
            DeviceWorkspaceManager *effective_workspace = workspace ? workspace : bound_workspace_;
            if (!effective_workspace)
            {
                LOG_ERROR("[CUDAFloatingPointGemmKernel::multiply_fused_tensor] Batched FP32 projection workspace is not bound");
                return false;
            }

            const float *d_A = static_cast<const float *>(input->gpu_data_ptr());
            if (!d_A)
            {
                LOG_ERROR("[CUDAFloatingPointGemmKernel::multiply_fused_tensor] Input tensor is not on CUDA device");
                return false;
            }

            const int n = projections.front().n;
            if (n <= 0)
            {
                LOG_ERROR("[CUDAFloatingPointGemmKernel::multiply_fused_tensor] Invalid projection width");
                return false;
            }

            bool can_use_homogeneous_batched_path = true;
            for (size_t i = 0; i < projections.size(); ++i)
            {
                const auto &proj = projections[i];
                auto *fp_kernel = dynamic_cast<CUDAFloatingPointGemmKernel *>(proj.kernel);
                if (!fp_kernel || !proj.output)
                {
                    LOG_ERROR("[CUDAFloatingPointGemmKernel::multiply_fused_tensor] Projection "
                              << i << " is not a CUDA FP32 GEMM/output pair");
                    return false;
                }
                if (proj.n <= 0 ||
                    fp_kernel->N_ != static_cast<size_t>(proj.n) ||
                    fp_kernel->K_ != K_ ||
                    fp_kernel->precision_ != Precision::FP32 ||
                    fp_kernel->cuda_device_id_ != cuda_device_id_)
                {
                    LOG_ERROR("[CUDAFloatingPointGemmKernel::multiply_fused_tensor] Projection "
                              << i << " is not compatible with the FP32 projection group"
                              << " n=" << proj.n
                              << " kernel_n=" << fp_kernel->N_
                              << " k=" << fp_kernel->K_
                              << " expected_k=" << K_);
                    return false;
                }
                if (proj.output->native_type() != TensorType::FP32)
                {
                    LOG_ERROR("[CUDAFloatingPointGemmKernel::multiply_fused_tensor] Projection "
                              << i << " output must be FP32");
                    return false;
                }
                fp_kernel->setGPUStream(gpu_stream_);

                // The cublas batched-same-A path requires identical N and no
                // per-projection bias. QKV projection groups often have
                // n_q != n_k == n_v, so those groups use the same explicit-stream
                // single-projection GEMM path below instead of failing.
                if (proj.n != n || proj.bias)
                    can_use_homogeneous_batched_path = false;
            }

            if (!can_use_homogeneous_batched_path)
            {
                for (size_t i = 0; i < projections.size(); ++i)
                {
                    const auto &proj = projections[i];
                    auto *fp_kernel = static_cast<CUDAFloatingPointGemmKernel *>(proj.kernel);
                    auto *output = static_cast<TensorBase *>(proj.output);
                    if (!fp_kernel->multiply_tensor(
                            input,
                            output,
                            m,
                            proj.n,
                            k,
                            /*transpose_B=*/true,
                            /*alpha=*/1.0f,
                            /*beta=*/0.0f,
                            proj.bias,
                            nullptr,
                            -1,
                            bound_workspace_,
                            /*activation_row_offset=*/0))
                    {
                        LOG_ERROR("[CUDAFloatingPointGemmKernel::multiply_fused_tensor] Heterogeneous FP32 projection "
                                  << i << " failed");
                        return false;
                    }
                }

                if (PerfStatsCollector::isEnabled())
                {
                    PerfStatsCollector::addCounter(
                        "kernel",
                        "cuda_fp32_batched_fused_projection_calls",
                        1.0,
                        "gemm",
                        "cuda:" + std::to_string(cuda_device_id_),
                        PerfStatsCollector::Tags{
                            {"m", std::to_string(m)},
                            {"k", std::to_string(k)},
                            {"n", "mixed"},
                            {"projections", std::to_string(projections.size())},
                            {"mapped_redirect", "per_projection"},
                            {"route", "cublas_sequential_heterogeneous"}});
                }
                return true;
            }

            std::vector<const float *> d_B_matrices;
            std::vector<float *> d_C_matrices;
            std::vector<float *> d_mapped_outputs;
            d_B_matrices.reserve(projections.size());
            d_C_matrices.reserve(projections.size());
            d_mapped_outputs.reserve(projections.size());

            float *d_mapped_redirect_base = nullptr;
            const size_t mapped_redirect_stride = static_cast<size_t>(m) * static_cast<size_t>(n);
            const size_t mapped_redirect_bytes =
                mapped_redirect_stride * projections.size() * sizeof(float);

            for (size_t i = 0; i < projections.size(); ++i)
            {
                const auto &proj = projections[i];
                auto *fp_kernel = dynamic_cast<CUDAFloatingPointGemmKernel *>(proj.kernel);
                auto *d_C = static_cast<float *>(proj.output->gpu_data_ptr());
                if (!fp_kernel->d_weights_ || !d_C)
                {
                    LOG_ERROR("[CUDAFloatingPointGemmKernel::multiply_fused_tensor] Projection "
                              << i << " weight/output is not on CUDA device");
                    return false;
                }

                float *d_mapped_output = nullptr;
                if (proj.output->isMapped())
                {
                    if (!d_mapped_redirect_base)
                    {
                        if (!effective_workspace->hasBuffer(GemmWorkspaceBuffers::CUDA_FP32_MAPPED_REDIRECT) ||
                            effective_workspace->getBufferSize(GemmWorkspaceBuffers::CUDA_FP32_MAPPED_REDIRECT) < mapped_redirect_bytes)
                        {
                            LOG_ERROR("[CUDAFloatingPointGemmKernel::multiply_fused_tensor] Missing or undersized "
                                      << "declared graph workspace buffer '"
                                      << GemmWorkspaceBuffers::CUDA_FP32_MAPPED_REDIRECT
                                      << "' for batched mapped-output redirect. required_bytes="
                                      << mapped_redirect_bytes);
                            return false;
                        }
                        d_mapped_redirect_base = static_cast<float *>(
                            effective_workspace->getBuffer(GemmWorkspaceBuffers::CUDA_FP32_MAPPED_REDIRECT));
                        if (!d_mapped_redirect_base)
                        {
                            LOG_ERROR("[CUDAFloatingPointGemmKernel::multiply_fused_tensor] Batched mapped-output redirect workspace resolved to null");
                            return false;
                        }
                    }
                    d_mapped_output = d_C;
                    d_C = d_mapped_redirect_base + mapped_redirect_stride * i;
                }

                d_B_matrices.push_back(static_cast<const float *>(fp_kernel->d_weights_));
                d_C_matrices.push_back(d_C);
                d_mapped_outputs.push_back(d_mapped_output);
            }

            const int batch_count = static_cast<int>(projections.size());
            /*
             * The small-N FP32 projection kernel owns the GDN alpha/beta
             * publication contract for CUDA. cuBLAS can legally choose
             * different reduction schedules as M changes, and even sub-ULP
             * alpha/beta drift is amplified by recurrent state and prefix
             * cache restore. Keep CUDA aligned with ROCm: small-N FP32 groups
             * use one fixed per-row reduction tree for decode, short cached
             * prefixes, and longer full prefill blocks.
             */
            constexpr bool kSmallNFP32ProjectionPublicationEquivalent = true;
            const bool use_tiny_fp32 =
                kSmallNFP32ProjectionPublicationEquivalent &&
                m > 0 &&
                n > 0 && n <= 64 &&
                k > 0 &&
                batch_count > 0 &&
                batch_count <= 8;

            bool success = false;
            bool used_tiny_fp32 = false;
            if (use_tiny_fp32)
            {
                const size_t pointer_array_bytes = static_cast<size_t>(batch_count) * sizeof(float *);
                auto *d_A_array = static_cast<const float **>(
                    effective_workspace->getBuffer(GemmWorkspaceBuffers::CUDA_FP32_BATCH_A_PTRS));
                auto *d_B_array = static_cast<const float **>(
                    effective_workspace->getBuffer(GemmWorkspaceBuffers::CUDA_FP32_BATCH_B_PTRS));
                auto *d_C_array = static_cast<float **>(
                    effective_workspace->getBuffer(GemmWorkspaceBuffers::CUDA_FP32_BATCH_C_PTRS));
                if (!d_A_array || !d_B_array || !d_C_array ||
                    effective_workspace->getBufferSize(GemmWorkspaceBuffers::CUDA_FP32_BATCH_A_PTRS) < pointer_array_bytes ||
                    effective_workspace->getBufferSize(GemmWorkspaceBuffers::CUDA_FP32_BATCH_B_PTRS) < pointer_array_bytes ||
                    effective_workspace->getBufferSize(GemmWorkspaceBuffers::CUDA_FP32_BATCH_C_PTRS) < pointer_array_bytes)
                {
                    LOG_ERROR("[CUDAFloatingPointGemmKernel::multiply_fused_tensor] Missing or undersized "
                              << "workspace pointer-array buffers for tiny FP32 projection. required_count="
                              << batch_count);
                    return false;
                }

                std::vector<const float *> d_A_matrices(static_cast<size_t>(batch_count), d_A);
                if (!cudaFp32_stage_batched_projection_pointers(
                        d_A_array,
                        d_B_array,
                        d_C_array,
                        d_A_matrices.data(),
                        d_B_matrices.data(),
                        d_C_matrices.data(),
                        batch_count,
                        cuda_device_id_,
                        gpu_stream_))
                {
                    LOG_ERROR("[CUDAFloatingPointGemmKernel::multiply_fused_tensor] Failed to stage tiny FP32 projection pointers");
                    return false;
                }

                CUDA_KERNEL_PROFILE_SCOPE_STREAM(CUDAKernelType::GEMM_CUBLAS, gpu_stream_);
                success = cudaFp32_tiny_batched_projection(
                    d_A_array,
                    d_B_array,
                    d_C_array,
                    m,
                    n,
                    k,
                    batch_count,
                    cuda_device_id_,
                    gpu_stream_);
                used_tiny_fp32 = success;
            }
            else
            {
                CUDA_KERNEL_PROFILE_SCOPE_STREAM(CUDAKernelType::GEMM_CUBLAS, gpu_stream_);
                success = cublas_kernel_->execute_batched_same_a(
                    d_A,
                    d_B_matrices,
                    d_C_matrices,
                    m,
                    n,
                    k,
                    false,
                    true,
                    1.0f,
                    0.0f,
                    effective_workspace);
            }

            if (success)
            {
                for (size_t i = 0; i < d_mapped_outputs.size(); ++i)
                {
                    if (!d_mapped_outputs[i])
                        continue;
                    if (!cudaQuantGemm_copyDeviceToDeviceAsync(
                            d_mapped_outputs[i],
                            d_C_matrices[i],
                            mapped_redirect_stride,
                            cuda_device_id_,
                            gpu_stream_))
                    {
                        LOG_ERROR("[CUDAFloatingPointGemmKernel::multiply_fused_tensor] Failed to copy batched mapped output "
                                  << i << " from redirect workspace");
                        success = false;
                        break;
                    }
                }
            }

            if (success && PerfStatsCollector::isEnabled())
            {
                PerfStatsCollector::addCounter(
                    "kernel",
                    "cuda_fp32_batched_fused_projection_calls",
                    1.0,
                    "gemm",
                    "cuda:" + std::to_string(cuda_device_id_),
                    PerfStatsCollector::Tags{
                        {"m", std::to_string(m)},
                        {"k", std::to_string(k)},
                        {"n", std::to_string(n)},
                        {"projections", std::to_string(projections.size())},
                        {"mapped_redirect", d_mapped_redirect_base ? "1" : "0"},
                        {"route", used_tiny_fp32 ? "small_n_fp32_batched_projection" : "cublas_batched_same_a"}});
            }

            return success;
        }

        bool CUDAFloatingPointGemmKernel::multiply_fused_verifier_rows_decode_equivalent(
            const TensorBase *input,
            const std::vector<TensorProjectionDesc> &projections,
            int m, int k,
            const IMPIContext *mpi_ctx,
            DeviceWorkspaceManager *workspace)
        {
            if (m < 1)
            {
                LOG_ERROR("[CUDAFloatingPointGemmKernel] grouped verifier projection requires M>=1, got M="
                          << m);
                return false;
            }
            (void)mpi_ctx;

            if (!input || projections.empty() || k <= 0)
            {
                LOG_ERROR("[CUDAFloatingPointGemmKernel] floating grouped verifier projection rejected: input="
                          << (input != nullptr)
                          << " projections=" << projections.size()
                          << " k=" << k);
                return false;
            }
            if (input->native_type() != TensorType::FP32)
            {
                LOG_ERROR("[CUDAFloatingPointGemmKernel] floating grouped verifier projection requires FP32 activations, got "
                          << static_cast<int>(input->native_type()));
                return false;
            }
            if (!gpu_stream_)
            {
                LOG_ERROR("[CUDAFloatingPointGemmKernel] floating grouped verifier projection requires an explicit CUDA stream");
                return false;
            }

            DeviceWorkspaceManager *effective_workspace = workspace ? workspace : bound_workspace_;
            if (!effective_workspace)
            {
                LOG_ERROR("[CUDAFloatingPointGemmKernel] floating grouped verifier projection requires declared graph workspace");
                return false;
            }

            const float *d_input = static_cast<const float *>(input->gpu_data_ptr());
            if (!d_input)
            {
                LOG_ERROR("[CUDAFloatingPointGemmKernel] floating grouped verifier projection input has no CUDA device data");
                return false;
            }

            constexpr size_t kMaxBatchedFP32x16Projections = 8;
            auto *d_A_array = static_cast<const float **>(
                effective_workspace->getBuffer(GemmWorkspaceBuffers::CUDA_FP32_BATCH_A_PTRS));
            auto *d_B_array = static_cast<const float **>(
                effective_workspace->getBuffer(GemmWorkspaceBuffers::CUDA_FP32_BATCH_B_PTRS));
            auto *d_C_array = static_cast<float **>(
                effective_workspace->getBuffer(GemmWorkspaceBuffers::CUDA_FP32_BATCH_C_PTRS));
            const size_t pointer_array_bytes = kMaxBatchedFP32x16Projections * sizeof(float *);
            if (!d_A_array || !d_B_array || !d_C_array ||
                effective_workspace->getBufferSize(GemmWorkspaceBuffers::CUDA_FP32_BATCH_A_PTRS) < pointer_array_bytes ||
                effective_workspace->getBufferSize(GemmWorkspaceBuffers::CUDA_FP32_BATCH_B_PTRS) < pointer_array_bytes ||
                effective_workspace->getBufferSize(GemmWorkspaceBuffers::CUDA_FP32_BATCH_C_PTRS) < pointer_array_bytes)
            {
                LOG_ERROR("[CUDAFloatingPointGemmKernel] floating grouped verifier projection missing pointer-array workspace");
                return false;
            }

            const bool use_fp32_weights = precision_ == Precision::FP32;
            const int weight_dtype = (precision_ == Precision::BF16) ? 1 : 0;
            const char *dtype_tag =
                precision_ == Precision::FP32 ? "fp32" :
                (precision_ == Precision::BF16 ? "bf16" : "fp16");
            std::vector<bool> completed(projections.size(), false);

            for (size_t seed_index = 0; seed_index < projections.size(); ++seed_index)
            {
                if (completed[seed_index])
                    continue;

                const auto &seed = projections[seed_index];
                if (!seed.kernel || !seed.output || seed.bias || seed.n <= 0)
                {
                    LOG_ERROR("[CUDAFloatingPointGemmKernel] floating grouped verifier projection invalid seed at "
                              << seed_index);
                    return false;
                }
                if (seed.output->native_type() != TensorType::FP32)
                {
                    LOG_ERROR("[CUDAFloatingPointGemmKernel] floating grouped verifier projection output must be FP32 at "
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
                        std::min(kMaxBatchedFP32x16Projections, group_indices.size() - group_offset);
                    std::vector<const float *> a_ptrs(group_count, d_input);
                    std::vector<const float *> b_ptrs;
                    std::vector<float *> c_ptrs;
                    std::vector<float *> mapped_outputs(group_count, nullptr);
                    b_ptrs.reserve(group_count);
                    c_ptrs.reserve(group_count);

                    float *d_redirect_base = nullptr;
                    const size_t output_values =
                        static_cast<size_t>(m) * static_cast<size_t>(seed.n);
                    const size_t redirect_bytes = output_values * group_count * sizeof(float);

                    for (size_t local = 0; local < group_count; ++local)
                    {
                        const size_t projection_index = group_indices[group_offset + local];
                        const auto &projection = projections[projection_index];
                        auto *projection_kernel = dynamic_cast<CUDAFloatingPointGemmKernel *>(projection.kernel);
                        if (!projection_kernel ||
                            projection_kernel->precision_ != precision_ ||
                            projection_kernel->cuda_device_id_ != cuda_device_id_ ||
                            static_cast<int>(projection_kernel->K_) != k ||
                            projection_kernel->N_ != static_cast<size_t>(projection.n) ||
                            !projection_kernel->d_weights_)
                        {
                            LOG_ERROR("[CUDAFloatingPointGemmKernel] floating grouped verifier projection incompatible kernel at "
                                      << projection_index);
                            return false;
                        }

                        float *d_output = static_cast<float *>(projection.output->gpu_data_ptr());
                        if (!d_output)
                        {
                            LOG_ERROR("[CUDAFloatingPointGemmKernel] floating grouped verifier projection output has no CUDA data at "
                                      << projection_index);
                            return false;
                        }

                        if (projection.output->isMapped())
                        {
                            if (!d_redirect_base)
                            {
                                if (!effective_workspace->hasBuffer(GemmWorkspaceBuffers::CUDA_FP32_MAPPED_REDIRECT) ||
                                    effective_workspace->getBufferSize(GemmWorkspaceBuffers::CUDA_FP32_MAPPED_REDIRECT) < redirect_bytes)
                                {
                                    LOG_ERROR("[CUDAFloatingPointGemmKernel] floating grouped verifier projection mapped output "
                                              << "requires redirect workspace bytes=" << redirect_bytes);
                                    return false;
                                }
                                d_redirect_base = static_cast<float *>(
                                    effective_workspace->getBuffer(GemmWorkspaceBuffers::CUDA_FP32_MAPPED_REDIRECT));
                                if (!d_redirect_base)
                                {
                                    LOG_ERROR("[CUDAFloatingPointGemmKernel] floating grouped verifier projection redirect workspace is null");
                                    return false;
                                }
                            }
                            mapped_outputs[local] = d_output;
                            d_output = d_redirect_base + output_values * local;
                        }

                        b_ptrs.push_back(reinterpret_cast<const float *>(projection_kernel->d_weights_));
                        c_ptrs.push_back(d_output);
                    }

                    if (!cudaFp32_stage_batched_projection_pointers(
                            d_A_array,
                            d_B_array,
                            d_C_array,
                            a_ptrs.data(),
                            b_ptrs.data(),
                            c_ptrs.data(),
                            static_cast<int>(group_count),
                            cuda_device_id_,
                            gpu_stream_))
                    {
                        LOG_ERROR("[CUDAFloatingPointGemmKernel] floating grouped verifier projection failed to stage pointers");
                        return false;
                    }

                    CUDA_KERNEL_PROFILE_SCOPE_STREAM(CUDAKernelType::GEMM_CUBLAS, gpu_stream_);
                    const bool projection_ok = use_fp32_weights
                                                   ? cudaFp32_tiny_batched_projection(
                                                         d_A_array,
                                                         d_B_array,
                                                         d_C_array,
                                                         m,
                                                         seed.n,
                                                         k,
                                                         static_cast<int>(group_count),
                                                         cuda_device_id_,
                                                         gpu_stream_)
                                                   : cudaFp32x16_tiny_batched_projection(
                                                         d_A_array,
                                                         d_B_array,
                                                         d_C_array,
                                                         m,
                                                         seed.n,
                                                         k,
                                                         static_cast<int>(group_count),
                                                         weight_dtype,
                                                         cuda_device_id_,
                                                         gpu_stream_);
                    if (!projection_ok)
                    {
                        LOG_ERROR("[CUDAFloatingPointGemmKernel] floating grouped verifier projection kernel failed"
                                  << " dtype=" << dtype_tag
                                  << " M=" << m
                                  << " N=" << seed.n
                                  << " K=" << k
                                  << " batch=" << group_count);
                        return false;
                    }

                    for (size_t local = 0; local < group_count; ++local)
                    {
                        if (!mapped_outputs[local])
                            continue;
                        if (!cudaQuantGemm_copyDeviceToDeviceAsync(
                                mapped_outputs[local],
                                c_ptrs[local],
                                output_values,
                                cuda_device_id_,
                                gpu_stream_))
                        {
                            LOG_ERROR("[CUDAFloatingPointGemmKernel] floating grouped verifier projection failed "
                                      << "to copy mapped output for local projection " << local);
                            return false;
                        }
                    }

                    for (size_t local = 0; local < group_count; ++local)
                        completed[group_indices[group_offset + local]] = true;

                    if (PerfStatsCollector::isEnabled())
                    {
                        PerfStatsCollector::addCounter(
                            "kernel",
                            use_fp32_weights
                                ? "cuda_fp32_grouped_verifier_projection_calls"
                                : "cuda_fp32x16_grouped_verifier_projection_calls",
                            1.0,
                            "gemm",
                            "cuda:" + std::to_string(cuda_device_id_),
                            PerfStatsCollector::Tags{
                                {"dtype", dtype_tag},
                                {"m", std::to_string(m)},
                                {"n", std::to_string(seed.n)},
                                {"k", std::to_string(k)},
                                {"projections", std::to_string(group_count)},
                                {"route", use_fp32_weights
                                              ? "fixed_order_fp32_batched_projection"
                                              : "fixed_order_fp32x16_batched_projection"}});
                    }

                    group_offset += group_count;
                }
            }

            return true;
        }

        bool CUDAFloatingPointGemmKernel::multiply_tensor_with_fused_swiglu(
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

        bool CUDAFloatingPointGemmKernel::multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
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

        bool CUDAFloatingPointGemmKernel::run_fixed_order_swiglu_down(
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
                LOG_ERROR("[CUDAFloatingPointGemmKernel::run_fixed_order_swiglu_down] Null tensor or weights"
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
                LOG_ERROR("[CUDAFloatingPointGemmKernel::run_fixed_order_swiglu_down] "
                          << "requires M>=1 and dimensions matching the down weights"
                          << " M=" << m << " N=" << n << " K=" << k
                          << " weight_N=" << N_ << " weight_K=" << K_);
                return false;
            }
            if (alpha != 1.0f || beta != 0.0f)
            {
                LOG_ERROR("[CUDAFloatingPointGemmKernel::run_fixed_order_swiglu_down] "
                          << "only alpha=1,beta=0 is supported for decode-equivalent verifier rows"
                          << " alpha=" << alpha << " beta=" << beta);
                return false;
            }
            if (!gpu_stream_)
            {
                LOG_ERROR("[CUDAFloatingPointGemmKernel::run_fixed_order_swiglu_down] "
                          << "an explicit non-default CUDA stream is required");
                return false;
            }
            if (gate->native_type() != TensorType::FP32 ||
                up->native_type() != TensorType::FP32 ||
                output->native_type() != TensorType::FP32)
            {
                LOG_ERROR("[CUDAFloatingPointGemmKernel::run_fixed_order_swiglu_down] "
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
                LOG_ERROR("[CUDAFloatingPointGemmKernel::run_fixed_order_swiglu_down] "
                          << "gate/up/output must be resident on the CUDA device");
                return false;
            }

            float *d_mapped_output = nullptr;
            if (output->isMapped())
            {
                const size_t needed_bytes =
                    static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float);
                DeviceWorkspaceManager *effective_workspace = workspace ? workspace : bound_workspace_;
                if (!effective_workspace ||
                    !effective_workspace->hasBuffer(GemmWorkspaceBuffers::CUDA_FP32_MAPPED_REDIRECT) ||
                    effective_workspace->getBufferSize(GemmWorkspaceBuffers::CUDA_FP32_MAPPED_REDIRECT) < needed_bytes)
                {
                    LOG_ERROR("[CUDAFloatingPointGemmKernel::run_fixed_order_swiglu_down] "
                              << "mapped output requires declared redirect workspace bytes="
                              << needed_bytes);
                    return false;
                }
                d_mapped_output = d_output;
                d_output = static_cast<float *>(
                    effective_workspace->getBuffer(GemmWorkspaceBuffers::CUDA_FP32_MAPPED_REDIRECT));
                if (!d_output)
                {
                    LOG_ERROR("[CUDAFloatingPointGemmKernel::run_fixed_order_swiglu_down] "
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

            CUDA_KERNEL_PROFILE_SCOPE_STREAM(CUDAKernelType::GEMM_CUBLAS, gpu_stream_);
            bool success = cudaFloating_swiglu_down_projection(
                d_gate,
                d_up,
                d_weights_,
                d_output,
                m,
                n,
                k,
                weight_dtype,
                cuda_device_id_,
                gpu_stream_);

            if (success && d_mapped_output)
            {
                success = cudaQuantGemm_copyDeviceToDeviceAsync(
                    d_mapped_output,
                    d_output,
                    static_cast<size_t>(m) * static_cast<size_t>(n),
                    cuda_device_id_,
                    gpu_stream_);
            }

            if (success && PerfStatsCollector::isEnabled())
            {
                PerfStatsCollector::addCounter(
                    "kernel",
                    verifier_grouped_call
                        ? "cuda_floating_grouped_verifier_swiglu_down_calls"
                        : "cuda_floating_fused_swiglu_down_calls",
                    1.0,
                    "gemm",
                    "cuda:" + std::to_string(cuda_device_id_),
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

        bool CUDAFloatingPointGemmKernel::supports_device(int device_idx) const
        {
            // This kernel only supports CUDA devices
            // device_idx >= 0 indicates GPU
            // We also check if it matches our specific CUDA device
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
            return (dev.type == ComputeBackendType::GPU_CUDA && dev.device_id == cuda_device_id_);
        }

        void CUDAFloatingPointGemmKernel::bindGPUStream(ExplicitGPUStream stream)
        {
            gpu_stream_ = stream.get();
            if (cublas_kernel_)
            {
                cublas_kernel_->bindStream(stream);
            }
        }

        void CUDAFloatingPointGemmKernel::clearGPUStreamBinding()
        {
            gpu_stream_ = nullptr;
            if (cublas_kernel_)
            {
                cublas_kernel_->clearStreamBinding();
            }
        }

        WorkspaceRequirements CUDAFloatingPointGemmKernel::getWorkspaceRequirements(int m, int n, int k) const
        {
            if (!cublas_kernel_)
                return WorkspaceRequirements{};
            WorkspaceRequirements reqs = cublas_kernel_->getWorkspaceRequirements(m, n, k);
            constexpr size_t kMaxBatchedFP32Projections = 8;
            const size_t pointer_array_bytes = kMaxBatchedFP32Projections * sizeof(float *);
            reqs.buffers.push_back({GemmWorkspaceBuffers::CUDA_FP32_BATCH_A_PTRS, pointer_array_bytes, 256, true});
            reqs.buffers.push_back({GemmWorkspaceBuffers::CUDA_FP32_BATCH_B_PTRS, pointer_array_bytes, 256, true});
            reqs.buffers.push_back({GemmWorkspaceBuffers::CUDA_FP32_BATCH_C_PTRS, pointer_array_bytes, 256, true});
            if (n == 0)
                n = static_cast<int>(N_);
            if (m > 0 && n > 0)
            {
                const size_t redirect_bytes =
                    kMaxBatchedFP32Projections *
                    static_cast<size_t>(m) *
                    static_cast<size_t>(n) *
                    sizeof(float);
                reqs.buffers.push_back({GemmWorkspaceBuffers::CUDA_FP32_MAPPED_REDIRECT,
                                        redirect_bytes,
                                        256,
                                        true});
            }
            return reqs;
        }

        void CUDAFloatingPointGemmKernel::bindWorkspace(DeviceWorkspaceManager *workspace)
        {
            bound_workspace_ = workspace;
            if (cublas_kernel_)
                cublas_kernel_->bindWorkspace(workspace);
        }

        void CUDAFloatingPointGemmKernel::unbindWorkspace()
        {
            if (cublas_kernel_)
                cublas_kernel_->unbindWorkspace();
            bound_workspace_ = nullptr;
        }

        // =====================================================================
        // Activation-activation GEMM (not supported)
        // =====================================================================

        bool CUDAFloatingPointGemmKernel::multiply_activations(
            const float * /*A*/, const float * /*B*/, float * /*C*/,
            int /*m*/, int /*n*/, int /*k*/,
            bool /*transpose_B*/,
            float /*alpha*/, float /*beta*/,
            const IMPIContext * /*mpi_ctx*/,
            int /*device_idx*/)
        {
            LOG_ERROR("[CUDAFloatingPointGemmKernel] multiply_activations not supported - use dedicated attention kernel");
            return false;
        }

        bool CUDAFloatingPointGemmKernel::multiply_activations_strided(
            const float * /*A*/, const float * /*B*/, float * /*C*/,
            int /*m*/, int /*n*/, int /*k*/,
            int /*lda*/, int /*ldb*/, int /*ldc*/,
            bool /*transpose_B*/,
            float /*alpha*/, float /*beta*/,
            const IMPIContext * /*mpi_ctx*/,
            int /*device_idx*/)
        {
            LOG_ERROR("[CUDAFloatingPointGemmKernel] multiply_activations_strided not supported - use dedicated attention kernel");
            return false;
        }

        // =====================================================================
        // IKernelSnapshotCapable interface
        // =====================================================================

        KernelSnapshotInfo CUDAFloatingPointGemmKernel::getKernelSnapshotInfo() const
        {
            return KernelSnapshotInfo::gemm()
                .withInput("A", "input activations [m, k]", KernelBufferDtype::FP32)
                .withWeight("B", "weight matrix [n, k]", KernelBufferDtype::FP32)
                .withOutput("C", "output matrix [m, n]", KernelBufferDtype::FP32)
                .withScalar("precision", "computation precision (FP32/FP16/BF16)", KernelBufferDtype::INT32)
                .withScalar("N", "output features", KernelBufferDtype::INT32)
                .withScalar("K", "input features", KernelBufferDtype::INT32)
                .withScalar("cuda_device_id", "CUDA device ID", KernelBufferDtype::INT32);
        }

    } // namespace cuda
} // namespace llaminar2
