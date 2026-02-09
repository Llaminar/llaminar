/**
 * @file ROCmFloatingPointGemmKernel.cpp
 * @brief ITensorGemm adapter implementation for hipBLAS FP32/FP16/BF16 GEMM
 *
 * This is the C++ adapter that wraps HipBLASGemmKernel. It implements the full
 * ITensorGemm interface and can be compiled with the regular C++ compiler
 * (not hipcc), avoiding MPI/TensorKernels.h compilation issues.
 *
 * **Design**: The adapter:
 * 1. Implements ITensorGemm (includes MPIContext, etc.)
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
#include "tensors/Tensors.h"           // FP32Tensor, BF16Tensor, FP16Tensor
#include "tensors/KernelSnapshotInfo.h"
#include "utils/Logger.h"
#include "utils/ROCmKernelProfiler.h"

#include <stdexcept>

namespace llaminar2
{
    namespace rocm
    {

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
              hipblas_kernel_(nullptr)
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

        ROCmFloatingPointGemmKernel::~ROCmFloatingPointGemmKernel() = default;

        ROCmFloatingPointGemmKernel::ROCmFloatingPointGemmKernel(ROCmFloatingPointGemmKernel &&other) noexcept
            : weights_(other.weights_),
              d_weights_(other.d_weights_),
              rocm_device_id_(other.rocm_device_id_),
              precision_(other.precision_),
              N_(other.N_),
              K_(other.K_),
              hipblas_kernel_(other.hipblas_kernel_) // Just copy the shared pointer
        {
            other.weights_ = nullptr;
            other.d_weights_ = nullptr;
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
            const MPIContext * /*mpi_ctx*/,
            int /*device_idx*/,
            DeviceWorkspaceManager *workspace)
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

            return multiply_tensor(A, C, m, n, k, transpose_B, alpha, beta, bias, nullptr, -1, workspace);
        }

        bool ROCmFloatingPointGemmKernel::multiply_tensor(
            const TensorBase *A, TensorBase *C,
            int m, int n, int k,
            bool transpose_B,
            float alpha, float beta,
            const TensorBase *bias,
            const MPIContext * /*mpi_ctx*/,
            int /*device_idx*/,
            DeviceWorkspaceManager *workspace)
        {
            ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::GEMM_ROCBLAS, static_cast<hipStream_t>(gpu_stream_));
            (void)workspace; // TODO: Use workspace for intermediate allocations
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
            // IMPORTANT: For BAR-backed tensors, use rocm_data_ptr() (HIP pointer)
            const float *d_A = nullptr;
            float *d_C = nullptr;

            if (A->isBARBacked() && A->rocm_data_ptr() != nullptr)
            {
                d_A = static_cast<const float *>(A->rocm_data_ptr());
                LOG_DEBUG("[ROCmFloatingPointGemmKernel::multiply_tensor] Using BAR rocm_data_ptr for A: " << d_A);
            }
            else
            {
                d_A = static_cast<const float *>(A->gpu_data_ptr());
            }

            if (C->isBARBacked() && C->rocm_data_ptr() != nullptr)
            {
                d_C = static_cast<float *>(C->rocm_data_ptr());
                LOG_DEBUG("[ROCmFloatingPointGemmKernel::multiply_tensor] Using BAR rocm_data_ptr for C: " << d_C);
            }
            else
            {
                d_C = static_cast<float *>(C->gpu_data_ptr());
            }

            if (!d_A || !d_C)
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel::multiply_tensor] A and C must be on GPU");
                return false;
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
                // Check BAR-backed status for bias
                if (bias->isBARBacked() && bias->rocm_data_ptr() != nullptr)
                {
                    d_bias = static_cast<const float *>(bias->rocm_data_ptr());
                }
                else
                {
                    d_bias = static_cast<const float *>(bias->gpu_data_ptr());
                }
                if (!d_bias)
                {
                    LOG_ERROR("[ROCmFloatingPointGemmKernel::multiply_tensor] Bias tensor must be on GPU");
                    return false;
                }
            }

            // Use fused GEMM+bias when bias is provided, otherwise use regular GEMM
            if (d_bias)
            {
                return hipblas_kernel_->execute_with_bias(
                    d_A,                                    // d_A
                    static_cast<const float *>(d_weights_), // d_B
                    d_C,                                    // d_C
                    d_bias,                                 // d_bias
                    m, n, k,
                    false,       // transA = false
                    transpose_B, // transB
                    alpha, beta);
            }
            else
            {
                return hipblas_kernel_->execute(
                    d_A,                                    // d_A
                    static_cast<const float *>(d_weights_), // d_B
                    d_C,                                    // d_C
                    m, n, k,
                    false,       // transA = false
                    transpose_B, // transB
                    alpha, beta);
            }
        }

        // =====================================================================
        // ITensorGemm interface - multiply() raw pointers
        // =====================================================================

        bool ROCmFloatingPointGemmKernel::multiply(
            const float *A, float *C,
            int m, int n, int k,
            bool transpose_B,
            float alpha, float beta,
            const MPIContext * /*mpi_ctx*/,
            int /*device_idx*/,
            DeviceWorkspaceManager *workspace)
        {
            ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::GEMM_ROCBLAS, static_cast<hipStream_t>(gpu_stream_));
            (void)workspace; // TODO: Use workspace for intermediate allocations
            if (!hipblas_kernel_)
            {
                LOG_ERROR("[ROCmFloatingPointGemmKernel::multiply] hipBLAS kernel not initialized");
                return false;
            }

            // A and C are device pointers
            // Weight (B) is stored in d_weights_
            // Execute: C = alpha * A @ B^T + beta * C (transpose_B is typically true)

            return hipblas_kernel_->execute(
                A,                                      // d_A
                static_cast<const float *>(d_weights_), // d_B
                C,                                      // d_C
                m, n, k,
                false,       // transA = false
                transpose_B, // transB
                alpha, beta);
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
        // Activation-activation GEMM (not supported)
        // =====================================================================

        bool ROCmFloatingPointGemmKernel::multiply_activations(
            const float * /*A*/, const float * /*B*/, float * /*C*/,
            int /*m*/, int /*n*/, int /*k*/,
            bool /*transpose_B*/,
            float /*alpha*/, float /*beta*/,
            const MPIContext * /*mpi_ctx*/,
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
            const MPIContext * /*mpi_ctx*/,
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
