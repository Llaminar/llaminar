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
#include "utils/CUDAKernelProfiler.h"

#include <stdexcept>
#include <mutex>

// CUDA memory operations (implemented in CUDAQuantisedGemmKernel_CUTLASS.cu)
extern "C"
{
    bool cudaQuantGemm_allocFloat(float **d_ptr, size_t count, int cuda_device_id);
    bool cudaQuantGemm_copyDeviceToDeviceAsync(float *d_dst, const float *d_src, size_t count, int cuda_device_id, void *stream);
    void cudaQuantGemm_freeDevice(void *d_ptr);
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

        CUDAFloatingPointGemmKernel::~CUDAFloatingPointGemmKernel()
        {
            if (d_mapped_redirect_)
            {
                cudaQuantGemm_freeDevice(d_mapped_redirect_);
                d_mapped_redirect_ = nullptr;
                mapped_redirect_capacity_ = 0;
            }
        }

        CUDAFloatingPointGemmKernel::CUDAFloatingPointGemmKernel(CUDAFloatingPointGemmKernel &&other) noexcept
            : weights_(other.weights_),
              d_weights_(other.d_weights_),
              cuda_device_id_(other.cuda_device_id_),
              precision_(other.precision_),
              N_(other.N_),
              K_(other.K_),
              cublas_kernel_(std::move(other.cublas_kernel_)),
              d_mapped_redirect_(other.d_mapped_redirect_),
              mapped_redirect_capacity_(other.mapped_redirect_capacity_)
        {
            other.weights_ = nullptr;
            other.d_weights_ = nullptr;
            other.d_mapped_redirect_ = nullptr;
            other.mapped_redirect_capacity_ = 0;
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

                // Transfer redirect buffer ownership
                if (d_mapped_redirect_)
                    cudaQuantGemm_freeDevice(d_mapped_redirect_);
                d_mapped_redirect_ = other.d_mapped_redirect_;
                mapped_redirect_capacity_ = other.mapped_redirect_capacity_;
                other.d_mapped_redirect_ = nullptr;
                other.mapped_redirect_capacity_ = 0;
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
            const MPIContext * /*mpi_ctx*/,
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
            const MPIContext * /*mpi_ctx*/,
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
                if (needed > mapped_redirect_capacity_)
                {
                    if (d_mapped_redirect_)
                        cudaQuantGemm_freeDevice(d_mapped_redirect_);
                    cudaQuantGemm_allocFloat(&d_mapped_redirect_, needed, cuda_device_id_);
                    mapped_redirect_capacity_ = needed;
                }
                d_mapped_output = d_C;
                d_C = d_mapped_redirect_;
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
                CUDA_KERNEL_PROFILE_SCOPE(CUDAKernelType::GEMM_CUBLAS);
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
                    cudaQuantGemm_copyDeviceToDeviceAsync(
                        d_mapped_output, d_C,
                        static_cast<size_t>(m) * n,
                        cuda_device_id_, gpu_stream_);
                }
                return success;
            }
            else
            {
                CUDA_KERNEL_PROFILE_SCOPE(CUDAKernelType::GEMM_CUBLAS);
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
                    cudaQuantGemm_copyDeviceToDeviceAsync(
                        d_mapped_output, d_C,
                        static_cast<size_t>(m) * n,
                        cuda_device_id_, gpu_stream_);
                }
                return success;
            }
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
