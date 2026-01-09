/**
 * @file CUDAQuantisedGemmKernel.cpp
 * @brief ITensorGemm adapter implementation for CUTLASS INT8 quantized GEMM
 *
 * This is the C++ adapter that wraps the CUTLASS INT8 GEMM kernel. It implements
 * the full ITensorGemm interface and can be compiled with the regular C++ compiler
 * (not nvcc), avoiding MPI/TensorKernels.h compilation issues.
 *
 * **Design**:
 * 1. Implements ITensorGemm (includes MPIContext, TensorBase, etc.)
 * 2. Delegates CUDA work to CUDAQuantisedGemmKernel_Impl (in .cu file)
 * 3. Handles tensor type introspection in multiply_tensor()
 * 4. Manages lazy weight conversion to INT8 + scales
 *
 * **Weight Conversion Pipeline**:
 * 1. Dequantize original quantized weights to FP32
 * 2. Per-column symmetric quantization to INT8
 * 3. Store INT8 weights in ColumnMajor layout (CUTLASS requirement)
 * 4. Store per-column scales for output rescaling
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "CUDAQuantisedGemmKernel.h"
#include "backends/ComputeBackend.h" // DeviceManager
#include "tensors/Tensors.h"         // Q8_1Tensor, FP32Tensor, etc.
#include "tensors/BlockStructures.h" // Q8_1Block
#include "tensors/KernelSnapshotInfo.h"
#include "utils/Logger.h"

#include <stdexcept>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace llaminar2
{
    namespace cuda
    {

        // =====================================================================
        // Forward declarations for CUDA implementation (defined in .cu file)
        // =====================================================================

        // These functions are implemented in CUDAQuantisedGemmKernel_CUTLASS.cu
        extern "C"
        {
            // Upload converted INT8 weights to device
            bool cudaQuantGemm_uploadWeights(
                const int8_t *h_weights_int8, // [K x N] ColumnMajor
                const float *h_scales_B,      // [N] per-column scales
                int8_t **d_weights_int8,      // Output device pointer
                float **d_scales_B,           // Output device pointer
                int K, int N,
                int cuda_device_id);

            // Upload work buffers for activation quantization
            bool cudaQuantGemm_ensureWorkBuffers(
                int8_t **d_A_int8,   // [M x K] quantized activations
                float **d_scales_A,  // [M] per-row scales
                int32_t **d_C_int32, // [M x N] INT32 accumulator
                int *work_buffer_M,  // Current capacity
                int M, int K, int N,
                int cuda_device_id);

            // Execute CUTLASS INT8 GEMM
            bool cudaQuantGemm_execute(
                const int8_t *d_A_int8,       // [M x K] RowMajor
                const int8_t *d_weights_int8, // [K x N] ColumnMajor
                int32_t *d_C_int32,           // [M x N] RowMajor
                int M, int N, int K,
                int cuda_device_id);

            // Apply output scaling: C_fp32 = C_int32 * scales_A * scales_B
            bool cudaQuantGemm_applyScaling(
                const int32_t *d_C_int32, // [M x N]
                float *d_C_fp32,          // [M x N] output
                const float *d_scales_A,  // [M]
                const float *d_scales_B,  // [N]
                int M, int N,
                float alpha, float beta,
                const float *d_C_existing, // For beta != 0
                int cuda_device_id);

            // Quantize FP32 activations to INT8
            bool cudaQuantGemm_quantizeActivations(
                const float *d_A_fp32, // [M x K]
                int8_t *d_A_int8,      // [M x K] output
                float *d_scales_A,     // [M] output
                int M, int K,
                int cuda_device_id);

            // Free device memory
            void cudaQuantGemm_freeDevice(void *d_ptr);
        }

        // =====================================================================
        // PIMPL implementation struct
        // =====================================================================

        struct CUDAQuantisedGemmKernel::Impl
        {
            // Device memory for converted weights (only used when owns_weight_memory_ = true)
            int8_t *d_weights_int8 = nullptr; // [K x N] ColumnMajor
            float *d_scales_B = nullptr;      // [N] per-column scales

            // Work buffers for activation quantization
            int8_t *d_A_int8 = nullptr;   // [M x K]
            float *d_scales_A = nullptr;  // [M]
            int32_t *d_C_int32 = nullptr; // [M x N]
            int work_buffer_M = 0;

            // Flag to track if we own weight memory
            bool owns_weight_memory = false;

            ~Impl()
            {
                // Only free weight memory if we own it (not from CUDAPackedWeights cache)
                if (owns_weight_memory)
                {
                    if (d_weights_int8)
                        cudaQuantGemm_freeDevice(d_weights_int8);
                    if (d_scales_B)
                        cudaQuantGemm_freeDevice(d_scales_B);
                }
                // Always free work buffers (we always own these)
                if (d_A_int8)
                    cudaQuantGemm_freeDevice(d_A_int8);
                if (d_scales_A)
                    cudaQuantGemm_freeDevice(d_scales_A);
                if (d_C_int32)
                    cudaQuantGemm_freeDevice(d_C_int32);
            }
        };

        // =====================================================================
        // CUDAPackedWeights destructor
        // =====================================================================

        CUDAPackedWeights::~CUDAPackedWeights()
        {
            if (d_int8_data)
                cudaQuantGemm_freeDevice(d_int8_data);
            if (d_scales)
                cudaQuantGemm_freeDevice(d_scales);
        }

        // =====================================================================
        // packWeightsToCUDA: Convert any quantized tensor to INT8 + scales
        // =====================================================================

        bool packWeightsToCUDA(const TensorBase *tensor, CUDAPackedWeights &out)
        {
            if (!tensor)
            {
                LOG_ERROR("[packWeightsToCUDA] Null tensor");
                return false;
            }

            const int N = static_cast<int>(tensor->rows()); // Output features
            const int K = static_cast<int>(tensor->cols()); // Input features

            // Get dequantized FP32 data
            const float *h_weights_fp32 = tensor->data();
            if (!h_weights_fp32)
            {
                LOG_ERROR("[packWeightsToCUDA] Failed to get FP32 data from tensor");
                return false;
            }

            // Allocate output vectors
            out.int8_data.resize(static_cast<size_t>(K) * N);
            out.scales.resize(N);
            out.K = K;
            out.N = N;

            // Per-column (per-output-feature) symmetric quantization
            // Weight tensor is [N x K] row-major (output_features × input_features)
            // CUTLASS wants [K × N] column-major, but memory layout is same!
            for (int n = 0; n < N; ++n)
            {
                // Find max_abs for this output feature
                float max_abs = 0.0f;
                for (int k = 0; k < K; ++k)
                {
                    float val = h_weights_fp32[n * K + k];
                    max_abs = std::max(max_abs, std::abs(val));
                }

                float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
                float inv_scale = 1.0f / scale;
                out.scales[n] = scale;

                // Quantize: column-major layout [K × N] means element (k, n) at offset n * K + k
                for (int k = 0; k < K; ++k)
                {
                    float val = h_weights_fp32[n * K + k];
                    int8_t quantized = static_cast<int8_t>(
                        std::round(std::clamp(val * inv_scale, -127.0f, 127.0f)));
                    out.int8_data[n * K + k] = quantized;
                }
            }

            LOG_DEBUG("[packWeightsToCUDA] Packed " << N << "x" << K << " weights to INT8");
            return true;
        }

        // =====================================================================
        // Constructor / Destructor
        // =====================================================================

        CUDAQuantisedGemmKernel::CUDAQuantisedGemmKernel(const TensorBase *weights, int cuda_device_id)
            : weights_(weights),
              packed_(nullptr),
              cuda_device_id_(cuda_device_id),
              N_(0),
              K_(0),
              weights_converted_(false),
              owns_weight_memory_(true), // Legacy path owns weight memory
              impl_(std::make_unique<Impl>())
        {
            if (!weights)
            {
                throw std::runtime_error("[CUDAQuantisedGemmKernel] Null weight tensor");
            }

            // Get dimensions
            N_ = weights->rows(); // Output features
            K_ = weights->cols(); // Input features

            // Validate it's a quantized type
            TensorType wt = weights->native_type();
            bool is_quantized = (wt == TensorType::IQ4_NL ||
                                 wt == TensorType::Q8_0 ||
                                 wt == TensorType::Q4_0 ||
                                 wt == TensorType::Q4_1 ||
                                 wt == TensorType::Q5_0 ||
                                 wt == TensorType::Q5_1 ||
                                 wt == TensorType::Q4_K ||
                                 wt == TensorType::Q5_K ||
                                 wt == TensorType::Q6_K ||
                                 wt == TensorType::Q8_K ||
                                 wt == TensorType::Q2_K ||
                                 wt == TensorType::Q3_K ||
                                 wt == TensorType::Q8_1 ||
                                 wt == TensorType::IQ4_XS ||
                                 wt == TensorType::IQ2_XXS ||
                                 wt == TensorType::IQ2_XS ||
                                 wt == TensorType::IQ3_XXS ||
                                 wt == TensorType::IQ2_S ||
                                 wt == TensorType::IQ3_S ||
                                 wt == TensorType::IQ1_S ||
                                 wt == TensorType::IQ1_M);

            if (!is_quantized)
            {
                throw std::runtime_error(
                    "[CUDAQuantisedGemmKernel] Weight tensor must be quantized type, got: " +
                    std::to_string(static_cast<int>(wt)));
            }

            impl_->owns_weight_memory = true; // Legacy constructor owns weight memory

            LOG_DEBUG("[CUDAQuantisedGemmKernel] Created (legacy) for " << N_ << "x" << K_
                                                                        << " quantized weights (type=" << static_cast<int>(wt)
                                                                        << ") on CUDA device " << cuda_device_id_);
        }

        CUDAQuantisedGemmKernel::CUDAQuantisedGemmKernel(CUDAPackedWeights *packed, int cuda_device_id)
            : weights_(nullptr),
              packed_(packed),
              cuda_device_id_(cuda_device_id),
              N_(0),
              K_(0),
              weights_converted_(false),  // Not yet uploaded to device
              owns_weight_memory_(false), // CUDAPackedWeights owns the memory
              impl_(std::make_unique<Impl>())
        {
            if (!packed)
            {
                throw std::runtime_error("[CUDAQuantisedGemmKernel] Null packed weights");
            }

            N_ = static_cast<size_t>(packed->N);
            K_ = static_cast<size_t>(packed->K);

            impl_->owns_weight_memory = false; // Packed cache owns weight memory

            LOG_DEBUG("[CUDAQuantisedGemmKernel] Created (pre-packed) for " << N_ << "x" << K_
                                                                            << " INT8 weights on CUDA device " << cuda_device_id_);
        }

        CUDAQuantisedGemmKernel::~CUDAQuantisedGemmKernel() = default;

        CUDAQuantisedGemmKernel::CUDAQuantisedGemmKernel(CUDAQuantisedGemmKernel &&other) noexcept
            : weights_(other.weights_),
              packed_(other.packed_),
              cuda_device_id_(other.cuda_device_id_),
              N_(other.N_),
              K_(other.K_),
              weights_converted_(other.weights_converted_),
              owns_weight_memory_(other.owns_weight_memory_),
              impl_(std::move(other.impl_))
        {
            other.weights_ = nullptr;
            other.packed_ = nullptr;
            other.weights_converted_ = false;
            other.owns_weight_memory_ = false;
        }

        CUDAQuantisedGemmKernel &CUDAQuantisedGemmKernel::operator=(CUDAQuantisedGemmKernel &&other) noexcept
        {
            if (this != &other)
            {
                weights_ = other.weights_;
                packed_ = other.packed_;
                cuda_device_id_ = other.cuda_device_id_;
                N_ = other.N_;
                K_ = other.K_;
                weights_converted_ = other.weights_converted_;
                owns_weight_memory_ = other.owns_weight_memory_;
                impl_ = std::move(other.impl_);

                other.weights_ = nullptr;
                other.packed_ = nullptr;
                other.weights_converted_ = false;
                other.owns_weight_memory_ = false;
            }
            return *this;
        }

        // =====================================================================
        // Weight conversion: Any quantized format → INT8 + scales
        // =====================================================================

        void CUDAQuantisedGemmKernel::ensureWeightsConverted()
        {
            if (weights_converted_)
            {
                return;
            }

            // Pre-packed path: upload from CUDAPackedWeights
            if (packed_)
            {
                // Check if already uploaded to this device
                if (packed_->uploaded && packed_->cuda_device_id == cuda_device_id_)
                {
                    // Already uploaded - just reference the device pointers
                    impl_->d_weights_int8 = packed_->d_int8_data;
                    impl_->d_scales_B = packed_->d_scales;
                    weights_converted_ = true;
                    return;
                }

                // Upload to device and cache in packed_
                if (!cudaQuantGemm_uploadWeights(
                        packed_->int8_data.data(),
                        packed_->scales.data(),
                        &packed_->d_int8_data,
                        &packed_->d_scales,
                        packed_->K,
                        packed_->N,
                        cuda_device_id_))
                {
                    throw std::runtime_error("[CUDAQuantisedGemmKernel] Failed to upload pre-packed weights");
                }

                packed_->cuda_device_id = cuda_device_id_;
                packed_->uploaded = true;

                // Reference device pointers from packed cache
                impl_->d_weights_int8 = packed_->d_int8_data;
                impl_->d_scales_B = packed_->d_scales;
                weights_converted_ = true;

                LOG_DEBUG("[CUDAQuantisedGemmKernel] Uploaded pre-packed weights to device " << cuda_device_id_);
                return;
            }

            // Legacy path: convert from raw tensor
            LOG_DEBUG("[CUDAQuantisedGemmKernel] Converting weights to INT8 (legacy path)...");

            if (!weights_)
            {
                throw std::runtime_error("[CUDAQuantisedGemmKernel] No weights or packed data available");
            }

            // Get dequantized FP32 from tensor
            const float *h_weights_fp32 = weights_->data();
            if (!h_weights_fp32)
            {
                throw std::runtime_error(
                    "[CUDAQuantisedGemmKernel] Failed to get FP32 data from weight tensor");
            }

            // Per-column symmetric quantization to INT8
            std::vector<int8_t> h_weights_int8(K_ * N_);
            std::vector<float> h_scales_B(N_);

            for (size_t n = 0; n < N_; ++n) // For each output feature
            {
                // Find max_abs for this output feature
                float max_abs = 0.0f;
                for (size_t k = 0; k < K_; ++k)
                {
                    float val = h_weights_fp32[n * K_ + k];
                    max_abs = std::max(max_abs, std::abs(val));
                }

                float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
                float inv_scale = 1.0f / scale;
                h_scales_B[n] = scale;

                // Quantize: column-major layout means element (k, n) at offset n * K + k
                for (size_t k = 0; k < K_; ++k)
                {
                    float val = h_weights_fp32[n * K_ + k];
                    int8_t quantized = static_cast<int8_t>(
                        std::round(std::clamp(val * inv_scale, -127.0f, 127.0f)));
                    h_weights_int8[n * K_ + k] = quantized;
                }
            }

            // Upload to device
            if (!cudaQuantGemm_uploadWeights(
                    h_weights_int8.data(),
                    h_scales_B.data(),
                    &impl_->d_weights_int8,
                    &impl_->d_scales_B,
                    static_cast<int>(K_),
                    static_cast<int>(N_),
                    cuda_device_id_))
            {
                throw std::runtime_error("[CUDAQuantisedGemmKernel] Failed to upload converted weights");
            }

            weights_converted_ = true;
            LOG_DEBUG("[CUDAQuantisedGemmKernel] Weight conversion complete (legacy)");
        }

        void CUDAQuantisedGemmKernel::ensureWorkBuffers(int m)
        {
            if (m <= impl_->work_buffer_M)
            {
                return; // Already have enough capacity
            }

            if (!cudaQuantGemm_ensureWorkBuffers(
                    &impl_->d_A_int8,
                    &impl_->d_scales_A,
                    &impl_->d_C_int32,
                    &impl_->work_buffer_M,
                    m,
                    static_cast<int>(K_),
                    static_cast<int>(N_),
                    cuda_device_id_))
            {
                throw std::runtime_error("[CUDAQuantisedGemmKernel] Failed to allocate work buffers");
            }
        }

        // =====================================================================
        // ITensorGemm interface - multiply_tensor() PRIMARY ENTRY POINT
        // =====================================================================

        bool CUDAQuantisedGemmKernel::multiply_tensor(
            const TensorBase *A, TensorBase *C,
            bool transpose_B,
            float alpha, float beta,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            if (!A || !C)
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_tensor] Null input or output tensor");
                return false;
            }

            int m = static_cast<int>(A->rows());
            int n = static_cast<int>(N_);
            int k = static_cast<int>(K_);

            return multiply_tensor(A, C, m, n, k, transpose_B, alpha, beta, mpi_ctx, device_idx);
        }

        bool CUDAQuantisedGemmKernel::multiply_tensor(
            const TensorBase *A, TensorBase *C,
            int m, int n, int k,
            bool /*transpose_B*/,
            float alpha, float beta,
            const MPIContext * /*mpi_ctx*/,
            int /*device_idx*/)
        {
            if (!A || !C)
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_tensor] Null input or output tensor");
                return false;
            }

            // Ensure weights are converted
            ensureWeightsConverted();

            // Type dispatch based on A and C types
            TensorType a_type = A->native_type();
            TensorType c_type = C->native_type();

            if (a_type == TensorType::Q8_1 && c_type == TensorType::FP32)
            {
                // Q8_1 → FP32: Use Q8_1 blocks directly
                auto *q8_tensor = dynamic_cast<const Q8_1Tensor *>(A);
                if (!q8_tensor)
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel] Failed to cast A to Q8_1Tensor");
                    return false;
                }

                auto *fp32_tensor = dynamic_cast<FP32Tensor *>(C);
                if (!fp32_tensor)
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel] Failed to cast C to FP32Tensor");
                    return false;
                }

                // Get device pointers
                const Q8_1Block *d_A_q8 = static_cast<const Q8_1Block *>(q8_tensor->gpu_data_ptr());
                float *d_C = static_cast<float *>(fp32_tensor->gpu_data_ptr());

                if (!d_A_q8 || !d_C)
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel] A and C must be on GPU");
                    return false;
                }

                bool success = multiply_q8_to_fp32(d_A_q8, d_C, m, n, k, alpha, beta);
                if (success)
                    C->mark_device_dirty();
                return success;
            }
            else if (a_type == TensorType::FP32 && c_type == TensorType::FP32)
            {
                // FP32 → FP32: Quantize activations on-the-fly
                const float *d_A = static_cast<const float *>(A->gpu_data_ptr());
                float *d_C = static_cast<float *>(C->gpu_data_ptr());

                if (!d_A || !d_C)
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel] A and C must be on GPU");
                    return false;
                }

                bool success = multiply_fp32_to_fp32(d_A, d_C, m, n, k, alpha, beta);
                if (success)
                    C->mark_device_dirty();
                return success;
            }
            else if (a_type == TensorType::Q8_1 && c_type == TensorType::Q8_1)
            {
                // Q8_1 → Q8_1: Fused requantization
                auto *q8_A = dynamic_cast<const Q8_1Tensor *>(A);
                auto *q8_C = dynamic_cast<Q8_1Tensor *>(C);
                if (!q8_A || !q8_C)
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel] Failed to cast tensors");
                    return false;
                }

                const Q8_1Block *d_A_q8 = static_cast<const Q8_1Block *>(q8_A->gpu_data_ptr());
                Q8_1Block *d_C_q8 = static_cast<Q8_1Block *>(q8_C->gpu_data_ptr());

                bool success = multiply_q8_to_q8(d_A_q8, d_C_q8, m, n, k);
                if (success)
                    C->mark_device_dirty();
                return success;
            }
            else if (a_type == TensorType::FP32 && c_type == TensorType::Q8_1)
            {
                // FP32 → Q8_1: Quantize input, fused requant output
                const float *d_A = static_cast<const float *>(A->gpu_data_ptr());
                auto *q8_C = dynamic_cast<Q8_1Tensor *>(C);
                if (!q8_C)
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel] Failed to cast C to Q8_1Tensor");
                    return false;
                }

                Q8_1Block *d_C_q8 = static_cast<Q8_1Block *>(q8_C->gpu_data_ptr());

                bool success = multiply_fp32_to_q8(d_A, d_C_q8, m, n, k);
                if (success)
                    C->mark_device_dirty();
                return success;
            }
            else
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_tensor] Unsupported type combination: A="
                          << static_cast<int>(a_type) << ", C=" << static_cast<int>(c_type));
                return false;
            }
        }

        // =====================================================================
        // ITensorGemm interface - multiply() raw pointers (FP32 fallback)
        // =====================================================================

        bool CUDAQuantisedGemmKernel::multiply(
            const float *A, float *C,
            int m, int n, int k,
            bool /*transpose_B*/,
            float alpha, float beta,
            const MPIContext * /*mpi_ctx*/,
            int /*device_idx*/)
        {
            return multiply_fp32_to_fp32(A, C, m, n, k, alpha, beta);
        }

        // =====================================================================
        // Internal dispatch methods
        // =====================================================================

        bool CUDAQuantisedGemmKernel::multiply_q8_to_fp32(
            const Q8_1Block * /*d_A_q8*/, float * /*d_C*/,
            int /*m*/, int /*n*/, int /*k*/,
            float /*alpha*/, float /*beta*/)
        {
            // TODO: Implement Q8_1 direct path
            // For now, this would need to extract int8 data from Q8_1 blocks
            LOG_ERROR("[CUDAQuantisedGemmKernel] Q8_1→FP32 path not yet implemented");
            return false;
        }

        bool CUDAQuantisedGemmKernel::multiply_q8_to_q8(
            const Q8_1Block * /*d_A_q8*/, Q8_1Block * /*d_C_q8*/,
            int /*m*/, int /*n*/, int /*k*/)
        {
            // TODO: Implement Q8_1→Q8_1 fused requant path
            LOG_ERROR("[CUDAQuantisedGemmKernel] Q8_1→Q8_1 path not yet implemented");
            return false;
        }

        bool CUDAQuantisedGemmKernel::multiply_fp32_to_fp32(
            const float *d_A, float *d_C,
            int m, int n, int k,
            float alpha, float beta)
        {
            // Ensure weights converted and work buffers allocated
            ensureWeightsConverted();
            ensureWorkBuffers(m);

            // Step 1: Quantize activations
            if (!cudaQuantGemm_quantizeActivations(
                    d_A, impl_->d_A_int8, impl_->d_scales_A, m, k, cuda_device_id_))
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel] Activation quantization failed");
                return false;
            }

            // Step 2: Execute CUTLASS INT8 GEMM
            if (!cudaQuantGemm_execute(
                    impl_->d_A_int8, impl_->d_weights_int8, impl_->d_C_int32,
                    m, n, k, cuda_device_id_))
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel] CUTLASS GEMM failed");
                return false;
            }

            // Step 3: Apply scaling and output to FP32
            const float *d_C_existing = (beta != 0.0f) ? d_C : nullptr;
            if (!cudaQuantGemm_applyScaling(
                    impl_->d_C_int32, d_C, impl_->d_scales_A, impl_->d_scales_B,
                    m, n, alpha, beta, d_C_existing, cuda_device_id_))
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel] Scaling failed");
                return false;
            }

            return true;
        }

        bool CUDAQuantisedGemmKernel::multiply_fp32_to_q8(
            const float * /*d_A*/, Q8_1Block * /*d_C_q8*/,
            int /*m*/, int /*n*/, int /*k*/)
        {
            // TODO: Implement FP32→Q8_1 fused requant path
            LOG_ERROR("[CUDAQuantisedGemmKernel] FP32→Q8_1 path not yet implemented");
            return false;
        }

        // =====================================================================
        // Activation-activation GEMM (not supported)
        // =====================================================================

        bool CUDAQuantisedGemmKernel::multiply_activations(
            const float * /*A*/, const float * /*B*/, float * /*C*/,
            int /*m*/, int /*n*/, int /*k*/,
            bool /*transpose_B*/,
            float /*alpha*/, float /*beta*/,
            const MPIContext * /*mpi_ctx*/,
            int /*device_idx*/)
        {
            LOG_ERROR("[CUDAQuantisedGemmKernel] multiply_activations not supported - use dedicated attention kernel");
            return false;
        }

        bool CUDAQuantisedGemmKernel::multiply_activations_strided(
            const float * /*A*/, const float * /*B*/, float * /*C*/,
            int /*m*/, int /*n*/, int /*k*/,
            int /*lda*/, int /*ldb*/, int /*ldc*/,
            bool /*transpose_B*/,
            float /*alpha*/, float /*beta*/,
            const MPIContext * /*mpi_ctx*/,
            int /*device_idx*/)
        {
            LOG_ERROR("[CUDAQuantisedGemmKernel] multiply_activations_strided not supported - use dedicated attention kernel");
            return false;
        }

        // =====================================================================
        // ITensorKernel interface
        // =====================================================================

        bool CUDAQuantisedGemmKernel::supports_device(int device_idx) const
        {
            if (device_idx < 0)
            {
                return false; // CPU not supported
            }

            const auto &dm = DeviceManager::instance();
            if (static_cast<size_t>(device_idx) >= dm.devices().size())
            {
                return false;
            }

            const auto &dev = dm.devices()[device_idx];
            return (dev.type == ComputeBackendType::GPU_CUDA && dev.device_id == cuda_device_id_);
        }

        // =====================================================================
        // IKernelSnapshotCapable interface
        // =====================================================================

        KernelSnapshotInfo CUDAQuantisedGemmKernel::getKernelSnapshotInfo() const
        {
            return KernelSnapshotInfo::gemm()
                .withInput("A", "input activations [m, k]", KernelBufferDtype::FP32)
                .withWeight("B", "quantized weight matrix [n, k] (converted to INT8)", KernelBufferDtype::INT8)
                .withOutput("C", "output matrix [m, n]", KernelBufferDtype::FP32)
                .withScalar("N", "output features", KernelBufferDtype::INT32)
                .withScalar("K", "input features", KernelBufferDtype::INT32)
                .withScalar("cuda_device_id", "CUDA device ID", KernelBufferDtype::INT32)
                .withScalar("weights_converted", "whether weights are converted to INT8", KernelBufferDtype::INT32);
        }

    } // namespace cuda
} // namespace llaminar2
