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

            LOG_DEBUG("[CUDAFloatingPointGemmKernel] Created (raw ptr) for " << N_ << "x" << K_
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
            if (precision_ != Precision::FP32)
            {
                LOG_ERROR("[CUDAFloatingPointGemmKernel::multiply_fused_tensor] Only FP32 projection groups are supported");
                return false;
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
             * The tiny FP32 projection kernel owns the verifier-sized alpha /
             * beta contract for CUDA.  cuBLAS can legally choose different
             * reduction schedules for M=1 and M=2..4, and even sub-ULP GDN
             * alpha/beta drift is amplified by later quantized projections.
             *
             * The local kernel uses the same fixed reduction tree for every
             * row, so grouped verifier rows are decode-equivalent to repeated
             * single-row decode while staying device-resident and graph
             * capturable.
             */
            constexpr bool kTinyFP32ProjectionDecodeEquivalent = true;
            const bool use_tiny_fp32 =
                kTinyFP32ProjectionDecodeEquivalent &&
                m > 0 && m <= 4 &&
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
                    0.0f);
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
                        {"route", used_tiny_fp32 ? "tiny_fp32_batched_projection" : "cublas_batched_same_a"}});
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
            if (m <= 1 || m > 4)
            {
                LOG_ERROR("[CUDAFloatingPointGemmKernel] grouped verifier projection requires M=2..4, got M="
                          << m);
                return false;
            }
            return multiply_fused_tensor(input, projections, m, k, mpi_ctx, workspace);
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

        void CUDAFloatingPointGemmKernel::setGPUStream(void *stream)
        {
            gpu_stream_ = stream;
            if (cublas_kernel_)
            {
                cublas_kernel_->setStream(stream);
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
