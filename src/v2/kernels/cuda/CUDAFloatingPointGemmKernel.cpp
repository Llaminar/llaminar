/**
 * @file CUDAFloatingPointGemmKernel.cpp
 * @brief ITensorGemm adapter implementation for cuBLAS FP32/FP16/BF16 GEMM
 *
 * This is the C++ adapter that wraps CuBLASGemmKernel. It implements the full
 * ITensorGemm interface and can be compiled with the regular C++ compiler
 * (not nvcc), avoiding MPI/TensorKernels.h compilation issues.
 *
 * **Design**: The adapter:
 * 1. Implements ITensorGemm (includes MPIContext, etc.)
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

#include <stdexcept>

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

        CUDAFloatingPointGemmKernel::~CUDAFloatingPointGemmKernel() = default;

        CUDAFloatingPointGemmKernel::CUDAFloatingPointGemmKernel(CUDAFloatingPointGemmKernel &&other) noexcept
            : weights_(other.weights_),
              d_weights_(other.d_weights_),
              cuda_device_id_(other.cuda_device_id_),
              precision_(other.precision_),
              N_(other.N_),
              K_(other.K_),
              cublas_kernel_(std::move(other.cublas_kernel_))
        {
            other.weights_ = nullptr;
            other.d_weights_ = nullptr;
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

                other.weights_ = nullptr;
                other.d_weights_ = nullptr;
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
            const MPIContext * /*mpi_ctx*/,
            int /*device_idx*/,
            DeviceWorkspaceManager *workspace)
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

            return multiply_tensor(A, C, m, n, k, transpose_B, alpha, beta, nullptr, -1, workspace);
        }

        bool CUDAFloatingPointGemmKernel::multiply_tensor(
            const TensorBase *A, TensorBase *C,
            int m, int n, int k,
            bool transpose_B,
            float alpha, float beta,
            const MPIContext * /*mpi_ctx*/,
            int /*device_idx*/,
            DeviceWorkspaceManager *workspace)
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

            bool success = multiply(d_A, d_C, m, n, k, transpose_B, alpha, beta, nullptr, -1, workspace);
            return success;
        }

        // =====================================================================
        // ITensorGemm interface - multiply() raw pointers
        // =====================================================================

        bool CUDAFloatingPointGemmKernel::multiply(
            const float *A, float *C,
            int m, int n, int k,
            bool transpose_B,
            float alpha, float beta,
            const MPIContext * /*mpi_ctx*/,
            int /*device_idx*/,
            DeviceWorkspaceManager *workspace)
        {
            (void)workspace; // TODO: Use workspace for intermediate allocations
            if (!cublas_kernel_)
            {
                LOG_ERROR("[CUDAFloatingPointGemmKernel::multiply] cuBLAS kernel not initialized");
                return false;
            }

            // A and C are device pointers
            // Weight (B) is stored in d_weights_
            // Execute: C = alpha * A @ B^T + beta * C (transpose_B is typically true)

            return cublas_kernel_->execute(
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

        // =====================================================================
        // Activation-activation GEMM (not supported)
        // =====================================================================

        bool CUDAFloatingPointGemmKernel::multiply_activations(
            const float * /*A*/, const float * /*B*/, float * /*C*/,
            int /*m*/, int /*n*/, int /*k*/,
            bool /*transpose_B*/,
            float /*alpha*/, float /*beta*/,
            const MPIContext * /*mpi_ctx*/,
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
            const MPIContext * /*mpi_ctx*/,
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
