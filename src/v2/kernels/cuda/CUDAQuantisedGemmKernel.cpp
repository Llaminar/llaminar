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
#include "backends/DeviceId.h"       // DeviceId
#include "tensors/Tensors.h"         // Q8_1Tensor, FP32Tensor, etc.
#include "tensors/TensorSlice.h"     // TensorSlice - for unwrapping sliced biases
#include "tensors/BlockStructures.h" // Q8_1Block
#include "tensors/KernelSnapshotInfo.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/device/WorkspaceDescriptor.h"
#include "utils/Logger.h"
#include "utils/CUDAKernelProfiler.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <atomic>
#include <unordered_set>

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
                int cuda_device_id,
                void *stream = nullptr);

            // Apply output scaling: C_fp32 = C_int32 * scales_A * scales_B + bias
            bool cudaQuantGemm_applyScaling(
                const int32_t *d_C_int32, // [M x N]
                float *d_C_fp32,          // [M x N] output
                const float *d_scales_A,  // [M]
                const float *d_scales_B,  // [N]
                int M, int N,
                float alpha, float beta,
                const float *d_C_existing, // For beta != 0
                const float *d_bias,       // [N] optional bias
                int cuda_device_id,
                void *stream = nullptr);

            // Quantize FP32 activations to INT8
            bool cudaQuantGemm_quantizeActivations(
                const float *d_A_fp32, // [M x K]
                int8_t *d_A_int8,      // [M x K] output
                float *d_scales_A,     // [M] output
                int M, int K,
                int cuda_device_id,
                void *stream = nullptr);

            // Quantize FP32 activations to INT8 with per-block-of-32 scales
            bool cudaQuantGemm_quantizeActivationsBlockwise(
                const float *d_A_fp32,          // [M x K]
                int8_t *d_A_int8,               // [M x K] output
                float *d_scales_A_blockwise,    // [M x (K/32)] output
                int M, int K,
                int cuda_device_id,
                void *stream = nullptr);

            // Execute blockwise INT8 GEMM with dp4a and FP32 accumulation
            bool cudaQuantGemm_blockwiseGemm(
                const int8_t *d_A_int8,             // [M x K] row-major
                const int8_t *d_weights_int8,       // [K x N] col-major
                float *d_C_fp32,                    // [M x N] output
                const float *d_scales_A_blockwise,  // [M x (K/32)]
                const float *d_scales_B,            // [N]
                int M, int N, int K,
                float alpha, float beta,
                const float *d_C_existing,
                const float *d_bias,
                int cuda_device_id,
                void *stream = nullptr);

            bool cudaQuantGemm_prepareTensorCoreBlockedWeights(
                const int8_t *d_weights_int8,
                int8_t **d_weights_int8_tc_blocked,
                int K, int N,
                int cuda_device_id,
                void *stream = nullptr);

            bool cudaQuantGemm_blockwiseTensorCoreGemm(
                const int8_t *d_A_int8,
                const int8_t *d_weights_int8_tc_blocked,
                int32_t *d_partial_int32,
                float *d_C_fp32,
                const float *d_scales_A_blockwise,
                const float *d_scales_B,
                int M, int N, int K,
                float alpha, float beta,
                const float *d_C_existing,
                const float *d_bias,
                int cuda_device_id,
                void *stream = nullptr);

            // Free device memory
            void cudaQuantGemm_freeDevice(void *d_ptr);

            // Memory management helpers for multiply_fused
            bool cudaQuantGemm_allocFloat(float **d_ptr, size_t count, int cuda_device_id);
            bool cudaQuantGemm_copyHostToDevice(float *d_dst, const float *h_src, size_t count, int cuda_device_id);
            bool cudaQuantGemm_copyDeviceToHost(float *h_dst, const float *d_src, size_t count, int cuda_device_id);
            bool cudaQuantGemm_copyInt32DeviceToHost(int32_t *h_dst, const int32_t *d_src, size_t count, int cuda_device_id);
            bool cudaQuantGemm_copyDeviceToDeviceAsync(float *d_dst, const float *d_src, size_t count, int cuda_device_id, void *stream);
            bool cudaQuantGemm_setDevice(int cuda_device_id);
            bool cudaQuantGemm_streamSync(int cuda_device_id, void *stream);

            // -----------------------------------------------------------------
            // Per-shape tensor-core GEMV kernel family
            // (CUDATensorCoreGemvKernels.cu)
            // -----------------------------------------------------------------
            void cudaTCGemv_setTuningOverrides(int tn, int cpt, int kb, int wk);

            bool cudaTCGemv_blockwiseGemv(
                const int8_t *d_A_int8,
                const int8_t *d_B_int8,
                float *d_C_fp32,
                const float *d_scales_A_block,
                const float *d_scales_B,
                int N, int K,
                float alpha, float beta,
                const float *d_C_existing,
                const float *d_bias,
                int cuda_device_id,
                void *stream);

            bool cudaNativePayloadGemv_supportsCodebook(uint8_t codebook_id);

            bool cudaNativePayloadInitIQGridTables();

            bool cudaNativePayloadGemv_fp32(
                const int8_t *d_A_int8,
                const uint8_t *d_payload,
                const uint16_t *d_scales,
                const uint16_t *d_mins,
                const uint32_t *d_emins,
                float *d_C_fp32,
                const float *d_scales_A_block,
                int N, int K,
                float alpha, float beta,
                const float *d_C_existing,
                const float *d_bias,
                uint8_t codebook_id,
                int cuda_device_id,
                void *stream);

            bool cudaNativePayloadInitIQGridTables_gemm();

            bool cudaNativePayloadGemm_fp32(
                const int8_t *d_A_int8,
                const uint8_t *d_payload,
                const uint16_t *d_scales,
                const uint16_t *d_mins,
                const uint32_t *d_emins,
                float *d_C_fp32,
                const float *d_scales_A_block,
                int M, int N, int K,
                float alpha, float beta,
                const float *d_C_existing,
                const float *d_bias,
                uint8_t codebook_id,
                int cuda_device_id,
                void *stream);

            // -----------------------------------------------------------------
            // Per-shape tensor-core GEMM kernel family
            // (CUDATensorCoreGemmKernels.cu)
            // -----------------------------------------------------------------
            void cudaTCGemm_setTuningOverrides(int force_v3, int force_v7, int mt, int kt, int splitk);

            bool cudaTCGemm_blockwiseGemm(
                const int8_t *d_A_int8,
                const int8_t *d_B_int8_tc_blocked,
                int32_t *d_partial_int32,
                float *d_C_fp32,
                const float *d_scales_A_block,
                const float *d_scales_B,
                int M, int N, int K,
                float alpha, float beta,
                const float *d_C_existing,
                const float *d_bias,
                int cuda_device_id,
                void *stream);
        }

        // =====================================================================
        // PIMPL implementation struct
        // =====================================================================

        struct CUDAQuantisedGemmKernel::Impl
        {
            // Device memory for converted weights (only used when owns_weight_memory_ = true)
            int8_t *d_weights_int8 = nullptr; // [K x N] ColumnMajor
            float *d_scales_B = nullptr;      // [N] per-column scales
            int8_t *d_weights_int8_tc_blocked = nullptr; // [K/32][N][32] tensor-core layout
            uint8_t *d_weights_native_payload = nullptr;
            uint16_t *d_weights_native_scales = nullptr;
            uint16_t *d_weights_native_mins = nullptr;
            uint32_t *d_weights_native_emins = nullptr;
            uint8_t native_codebook_id = 0;
            uint32_t native_blocks_per_row = 0;

            // Work buffers - ALWAYS from workspace (never owned by kernel)
            // These pointers are set from workspace in validateWorkspace()
            int8_t *d_A_int8 = nullptr;   // [M x K] - from workspace
            float *d_scales_A = nullptr;  // [M] - from workspace
            int32_t *d_C_int32 = nullptr; // [M x N] - from workspace

            // Flag to track if we own weight memory
            bool owns_weight_memory = false;
            bool owns_tc_blocked_weight_memory = false;

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
                if (owns_tc_blocked_weight_memory && d_weights_int8_tc_blocked)
                {
                    cudaQuantGemm_freeDevice(d_weights_int8_tc_blocked);
                }
                if (owns_weight_memory)
                {
                    if (d_weights_native_payload)
                        cudaQuantGemm_freeDevice(d_weights_native_payload);
                    if (d_weights_native_scales)
                        cudaQuantGemm_freeDevice(d_weights_native_scales);
                    if (d_weights_native_mins)
                        cudaQuantGemm_freeDevice(d_weights_native_mins);
                    if (d_weights_native_emins)
                        cudaQuantGemm_freeDevice(d_weights_native_emins);
                }
                // Work buffers are NEVER owned by kernel - they come from workspace
            }
        };

        namespace
        {
            std::atomic<CUDABlockwiseExecutionBackend> g_blockwise_backend{CUDABlockwiseExecutionBackend::LegacyDP4A};

            template <typename T>
            bool uploadHostArrayToDevice(
                const std::vector<T> &host,
                T **device,
                int cuda_device_id)
            {
                *device = nullptr;
                if (host.empty())
                {
                    return true;
                }

                if (cudaSetDevice(cuda_device_id) != cudaSuccess)
                {
                    return false;
                }

                const size_t bytes = host.size() * sizeof(T);
                if (cudaMalloc(reinterpret_cast<void **>(device), bytes) != cudaSuccess)
                {
                    *device = nullptr;
                    return false;
                }

                if (cudaMemcpy(*device, host.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess)
                {
                    cudaFree(*device);
                    *device = nullptr;
                    return false;
                }

                return true;
            }

            void freeDeviceUploadNativeBuffers(CUDAPackedWeights::DeviceUpload &upload)
            {
                if (upload.d_native_payload)
                    cudaQuantGemm_freeDevice(upload.d_native_payload);
                if (upload.d_native_scales)
                    cudaQuantGemm_freeDevice(upload.d_native_scales);
                if (upload.d_native_mins)
                    cudaQuantGemm_freeDevice(upload.d_native_mins);
                if (upload.d_native_emins)
                    cudaQuantGemm_freeDevice(upload.d_native_emins);

                upload.d_native_payload = nullptr;
                upload.d_native_scales = nullptr;
                upload.d_native_mins = nullptr;
                upload.d_native_emins = nullptr;
            }

            bool uploadNativePackedWeights(
                const CUDAPackedWeights &packed,
                CUDAPackedWeights::DeviceUpload &upload,
                int cuda_device_id)
            {
                if (!uploadHostArrayToDevice(packed.native_payload, &upload.d_native_payload, cuda_device_id) ||
                    !uploadHostArrayToDevice(packed.native_scales, &upload.d_native_scales, cuda_device_id) ||
                    !uploadHostArrayToDevice(packed.native_mins, &upload.d_native_mins, cuda_device_id) ||
                    !uploadHostArrayToDevice(packed.native_emins, &upload.d_native_emins, cuda_device_id))
                {
                    freeDeviceUploadNativeBuffers(upload);
                    return false;
                }

                return true;
            }

            bool runSelectedBlockwiseBackend(
                const int8_t *d_A_int8,
                const int8_t *d_weights_int8,
                const int8_t *d_weights_int8_tc_blocked,
                int32_t *d_partial_int32,
                float *d_C_fp32,
                const float *d_scales_A_blockwise,
                const float *d_scales_B,
                int m, int n, int k,
                float alpha, float beta,
                const float *d_C_existing,
                const float *d_bias,
                int cuda_device_id,
                void *stream)
            {
                const auto backend = g_blockwise_backend.load(std::memory_order_relaxed);

                // ── New per-shape kernel family (GEMV M=1, GEMM M>1) ──
                if (backend == CUDABlockwiseExecutionBackend::SpecializedBlockwise)
                {
                    bool ok = false;
                    if (m <= 1)
                    {
                        // GEMV path: d_A_int8 is [K], d_scales_A_blockwise is [K/32]
                        ok = cudaTCGemv_blockwiseGemv(
                            d_A_int8, d_weights_int8, d_C_fp32,
                            d_scales_A_blockwise, d_scales_B,
                            n, k,
                            alpha, beta, d_C_existing, d_bias,
                            cuda_device_id, stream);
                    }
                    else
                    {
                        // GEMM path: blocked weights + INT32 partial workspace.
                        ok = d_weights_int8_tc_blocked && cudaTCGemm_blockwiseGemm(
                            d_A_int8, d_weights_int8_tc_blocked, d_partial_int32, d_C_fp32,
                            d_scales_A_blockwise, d_scales_B,
                            m, n, k,
                            alpha, beta, d_C_existing, d_bias,
                            cuda_device_id, stream);
                    }

                    if (ok) return true;

                    LOG_WARN("[CUDAQuantisedGemmKernel] SpecializedBlockwise path failed (M=" << m
                             << "), falling back to legacy dp4a");
                }

                // ── Original tensor-core scaffold ──
                if (backend == CUDABlockwiseExecutionBackend::TensorCoreScaffold && d_weights_int8_tc_blocked)
                {
                    if (cudaQuantGemm_blockwiseTensorCoreGemm(
                            d_A_int8,
                            d_weights_int8_tc_blocked,
                            d_partial_int32,
                            d_C_fp32,
                            d_scales_A_blockwise,
                            d_scales_B,
                            m, n, k,
                            alpha, beta,
                            d_C_existing,
                            d_bias,
                            cuda_device_id,
                            stream))
                    {
                        return true;
                    }

                    LOG_WARN("[CUDAQuantisedGemmKernel] TensorCore scaffold blockwise path failed, falling back to legacy dp4a");
                }

                return cudaQuantGemm_blockwiseGemm(
                    d_A_int8,
                    d_weights_int8,
                    d_C_fp32,
                    d_scales_A_blockwise,
                    d_scales_B,
                    m, n, k,
                    alpha, beta,
                    d_C_existing,
                    d_bias,
                    cuda_device_id,
                    stream);
            }

            template <typename ImplT>
            bool canUseNativePayloadBlockwise(const ImplT *impl, int m, int k)
            {
                return impl &&
                       m > 0 &&
                       k > 0 &&
                       (k % 32) == 0 &&
                       impl->d_weights_native_payload &&
                       impl->d_weights_native_scales &&
                       cudaNativePayloadGemv_supportsCodebook(impl->native_codebook_id);
            }

            template <typename ImplT>
            bool runNativePayloadBlockwiseIfSupported(
                const ImplT *impl,
                const int8_t *d_A_int8,
                float *d_C_fp32,
                const float *d_scales_A_blockwise,
                int m, int n, int k,
                float alpha, float beta,
                const float *d_C_existing,
                const float *d_bias,
                int cuda_device_id,
                void *stream)
            {
                if (!canUseNativePayloadBlockwise(impl, m, k))
                {
                    return false;
                }

                const bool needs_iq_tables = impl->native_codebook_id >= 11 && impl->native_codebook_id <= 17;
                if (needs_iq_tables)
                {
                    static std::mutex iq_table_mutex;
                    static std::unordered_set<int> gemv_init_devices;
                    static std::unordered_set<int> gemm_init_devices;

                    std::lock_guard<std::mutex> lock(iq_table_mutex);
                    auto &initialized = (m == 1) ? gemv_init_devices : gemm_init_devices;
                    if (!initialized.count(cuda_device_id))
                    {
                        if (!cudaQuantGemm_setDevice(cuda_device_id))
                        {
                            LOG_ERROR("[CUDAQuantisedGemmKernel] Failed to set CUDA device " << cuda_device_id
                                                                                               << " before IQ grid table initialization");
                            return false;
                        }
                        const bool ok = (m == 1)
                            ? cudaNativePayloadInitIQGridTables()
                            : cudaNativePayloadInitIQGridTables_gemm();
                        if (!ok)
                        {
                            LOG_ERROR("[CUDAQuantisedGemmKernel] Failed to initialize IQ grid tables for CUDA device " << cuda_device_id);
                            return false;
                        }
                        initialized.insert(cuda_device_id);
                    }
                }

                if (m == 1)
                {
                    static std::once_flag native_payload_decode_once;
                    std::call_once(native_payload_decode_once, [&]()
                    {
                        LOG_INFO("[CUDAQuantisedGemmKernel] NativePayload GEMV decode enabled for supported CUDA codebooks");
                    });

                    return cudaNativePayloadGemv_fp32(
                        d_A_int8,
                        impl->d_weights_native_payload,
                        impl->d_weights_native_scales,
                        impl->d_weights_native_mins,
                        impl->d_weights_native_emins,
                        d_C_fp32,
                        d_scales_A_blockwise,
                        n, k,
                        alpha, beta,
                        d_C_existing,
                        d_bias,
                        impl->native_codebook_id,
                        cuda_device_id,
                        stream);
                }

                static std::once_flag native_payload_prefill_once;
                std::call_once(native_payload_prefill_once, [&]()
                {
                    LOG_INFO("[CUDAQuantisedGemmKernel] NativePayload GEMM prefill enabled for supported CUDA codebooks");
                });

                return cudaNativePayloadGemm_fp32(
                    d_A_int8,
                    impl->d_weights_native_payload,
                    impl->d_weights_native_scales,
                    impl->d_weights_native_mins,
                    impl->d_weights_native_emins,
                    d_C_fp32,
                    d_scales_A_blockwise,
                    m, n, k,
                    alpha, beta,
                    d_C_existing,
                    d_bias,
                    impl->native_codebook_id,
                    cuda_device_id,
                    stream);
            }
        }

        void CUDAQuantisedGemmKernel::setBlockwiseExecutionBackend(CUDABlockwiseExecutionBackend backend)
        {
            g_blockwise_backend.store(backend, std::memory_order_relaxed);
        }

        CUDABlockwiseExecutionBackend CUDAQuantisedGemmKernel::getBlockwiseExecutionBackend()
        {
            return g_blockwise_backend.load(std::memory_order_relaxed);
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
            LOG_DEBUG("[CUDAQuantisedGemmKernel::ensureWeightsConverted] Entry: N_=" << N_ << " K_=" << K_
                                                                                     << " weights_converted_=" << weights_converted_
                                                                                     << " d_scales_B=" << (impl_ ? (void *)impl_->d_scales_B : nullptr)
                                                                                     << " d_weights_int8=" << (impl_ ? (void *)impl_->d_weights_int8 : nullptr));
            if (weights_converted_)
            {
                return;
            }

            // Pre-packed path: upload from CUDAPackedWeights
            if (packed_)
            {
                std::lock_guard<std::mutex> lock(packed_->upload_mutex);

                auto upload_it = packed_->device_uploads.find(cuda_device_id_);
                if (upload_it == packed_->device_uploads.end())
                {
                    CUDAPackedWeights::DeviceUpload upload;

                    LOG_DEBUG("[CUDAQuantisedGemmKernel::ensureWeightsConverted] Uploading packed weights to CUDA:"
                              << cuda_device_id_ << " K=" << packed_->K << " N=" << packed_->N);
                    if (!cudaQuantGemm_uploadWeights(
                            packed_->int8_data.data(),
                            packed_->scales.data(),
                            &upload.d_int8_data,
                            &upload.d_scales,
                            packed_->K,
                            packed_->N,
                            cuda_device_id_))
                    {
                        if (upload.d_int8_data)
                            cudaQuantGemm_freeDevice(upload.d_int8_data);
                        if (upload.d_scales)
                            cudaQuantGemm_freeDevice(upload.d_scales);
                        throw std::runtime_error("[CUDAQuantisedGemmKernel] Failed to upload pre-packed weights");
                    }

                    if (!uploadNativePackedWeights(*packed_, upload, cuda_device_id_))
                    {
                        if (upload.d_int8_data)
                            cudaQuantGemm_freeDevice(upload.d_int8_data);
                        if (upload.d_scales)
                            cudaQuantGemm_freeDevice(upload.d_scales);
                        throw std::runtime_error("[CUDAQuantisedGemmKernel] Failed to upload pre-packed native buffers");
                    }

                    if (!cudaQuantGemm_prepareTensorCoreBlockedWeights(
                            upload.d_int8_data,
                            &upload.d_int8_data_tc_blocked,
                            packed_->K,
                            packed_->N,
                            cuda_device_id_))
                    {
                        LOG_WARN("[CUDAQuantisedGemmKernel] Failed to prepare tensor-core blocked weights; legacy blockwise path remains available");
                    }

                    auto emplaced = packed_->device_uploads.emplace(cuda_device_id_, upload);
                    upload_it = emplaced.first;
                }

                const auto &upload = upload_it->second;
                packed_->d_int8_data = upload.d_int8_data;
                packed_->d_scales = upload.d_scales;
                packed_->d_int8_data_tc_blocked = upload.d_int8_data_tc_blocked;
                packed_->d_native_payload = upload.d_native_payload;
                packed_->d_native_scales = upload.d_native_scales;
                packed_->d_native_mins = upload.d_native_mins;
                packed_->d_native_emins = upload.d_native_emins;
                packed_->cuda_device_id = cuda_device_id_;
                packed_->uploaded = true;

                impl_->d_weights_int8 = upload.d_int8_data;
                impl_->d_scales_B = upload.d_scales;
                impl_->d_weights_int8_tc_blocked = upload.d_int8_data_tc_blocked;
                impl_->d_weights_native_payload = upload.d_native_payload;
                impl_->d_weights_native_scales = upload.d_native_scales;
                impl_->d_weights_native_mins = upload.d_native_mins;
                impl_->d_weights_native_emins = upload.d_native_emins;
                impl_->native_codebook_id = packed_->native_codebook_id;
                impl_->native_blocks_per_row = packed_->native_blocks_per_row;
                weights_converted_ = true;

                // Release host-side packing buffers — data is now on GPU.
                // This saves ~2× the quantized weight size of host memory.
                // Only safe when this packed_ won't be uploaded to additional devices;
                // we check device_uploads.size() == 1 as a proxy (TP shards have
                // separate packed_ per shard, so this is typically the only device).
                if (packed_->device_uploads.size() <= 1)
                {
                    const size_t freed_bytes =
                        packed_->int8_data.capacity() +
                        packed_->scales.capacity() * sizeof(float) +
                        packed_->native_payload.capacity() +
                        packed_->native_scales.capacity() * sizeof(uint16_t) +
                        packed_->native_mins.capacity() * sizeof(uint16_t) +
                        packed_->native_emins.capacity() * sizeof(uint32_t);
                    packed_->int8_data.clear();
                    packed_->int8_data.shrink_to_fit();
                    packed_->scales.clear();
                    packed_->scales.shrink_to_fit();
                    packed_->native_payload.clear();
                    packed_->native_payload.shrink_to_fit();
                    packed_->native_scales.clear();
                    packed_->native_scales.shrink_to_fit();
                    packed_->native_mins.clear();
                    packed_->native_mins.shrink_to_fit();
                    packed_->native_emins.clear();
                    packed_->native_emins.shrink_to_fit();
                    if (freed_bytes > 0)
                    {
                        LOG_DEBUG("[CUDAQuantisedGemmKernel] Released host packing buffers: "
                                  << (freed_bytes / (1024 * 1024)) << " MB");
                    }
                }

                LOG_DEBUG("[CUDAQuantisedGemmKernel] Using cached pre-packed weights on CUDA:" << cuda_device_id_);
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

            // DEBUG: Print first few weight values to verify slicing
            if (N_ == 64 && K_ == 896) // This is the K weight shape for LOCAL TP
            {
                LOG_INFO("[CUDAQuantisedGemmKernel DEBUG] K weight N=" << N_ << " K=" << K_
                                                                       << " device=" << cuda_device_id_
                                                                       << " first 5 weights[0]: " << h_weights_fp32[0] << ", "
                                                                       << h_weights_fp32[1] << ", " << h_weights_fp32[2] << ", "
                                                                       << h_weights_fp32[3] << ", " << h_weights_fp32[4]);
            }

            CUDAPackedWeights legacy_packed;
            if (!packWeightsToCUDA(weights_, legacy_packed))
            {
                throw std::runtime_error(
                    "[CUDAQuantisedGemmKernel] Failed to pack converted weights");
            }

            // Upload to device
            if (!cudaQuantGemm_uploadWeights(
                    legacy_packed.int8_data.data(),
                    legacy_packed.scales.data(),
                    &impl_->d_weights_int8,
                    &impl_->d_scales_B,
                    static_cast<int>(K_),
                    static_cast<int>(N_),
                    cuda_device_id_))
            {
                throw std::runtime_error("[CUDAQuantisedGemmKernel] Failed to upload converted weights");
            }

            if (!uploadNativePackedWeights(legacy_packed, legacy_packed.device_uploads[cuda_device_id_], cuda_device_id_))
            {
                throw std::runtime_error("[CUDAQuantisedGemmKernel] Failed to upload converted native buffers");
            }

            legacy_packed.device_uploads[cuda_device_id_].d_int8_data = impl_->d_weights_int8;
            legacy_packed.device_uploads[cuda_device_id_].d_scales = impl_->d_scales_B;

            if (cudaQuantGemm_prepareTensorCoreBlockedWeights(
                    impl_->d_weights_int8,
                    &impl_->d_weights_int8_tc_blocked,
                    static_cast<int>(K_),
                    static_cast<int>(N_),
                    cuda_device_id_))
            {
                impl_->owns_tc_blocked_weight_memory = true;
            }
            else
            {
                LOG_WARN("[CUDAQuantisedGemmKernel] Failed to prepare tensor-core blocked weights in legacy path; legacy blockwise path remains available");
            }

            legacy_packed.device_uploads[cuda_device_id_].d_int8_data_tc_blocked = impl_->d_weights_int8_tc_blocked;
            impl_->d_weights_native_payload = legacy_packed.device_uploads[cuda_device_id_].d_native_payload;
            impl_->d_weights_native_scales = legacy_packed.device_uploads[cuda_device_id_].d_native_scales;
            impl_->d_weights_native_mins = legacy_packed.device_uploads[cuda_device_id_].d_native_mins;
            impl_->d_weights_native_emins = legacy_packed.device_uploads[cuda_device_id_].d_native_emins;
            impl_->native_codebook_id = legacy_packed.native_codebook_id;
            impl_->native_blocks_per_row = legacy_packed.native_blocks_per_row;

            weights_converted_ = true;
            LOG_DEBUG("[CUDAQuantisedGemmKernel] Weight conversion complete (legacy)");
        }

        void CUDAQuantisedGemmKernel::validateWorkspace() const
        {
            // Kernels REQUIRE workspace - no internal buffer allocation
            if (!hasWorkspace())
            {
                throw std::runtime_error(
                    "[CUDAQuantisedGemmKernel] Workspace not bound. Kernels require pre-allocated "
                    "workspace buffers via bindWorkspace(). Call bindWorkspace() with a "
                    "DeviceWorkspaceManager that has allocated the buffers from getWorkspaceRequirements().");
            }

            // Validate required buffers exist
            if (!workspace_->hasBuffer(GemmWorkspaceBuffers::QUANT_A))
            {
                throw std::runtime_error(
                    "[CUDAQuantisedGemmKernel] Workspace missing required buffer: " +
                    std::string(GemmWorkspaceBuffers::QUANT_A));
            }
            if (!workspace_->hasBuffer(GemmWorkspaceBuffers::SCALES_A))
            {
                throw std::runtime_error(
                    "[CUDAQuantisedGemmKernel] Workspace missing required buffer: " +
                    std::string(GemmWorkspaceBuffers::SCALES_A));
            }
            if (!workspace_->hasBuffer(GemmWorkspaceBuffers::ACC_INT32))
            {
                throw std::runtime_error(
                    "[CUDAQuantisedGemmKernel] Workspace missing required buffer: " +
                    std::string(GemmWorkspaceBuffers::ACC_INT32));
            }
            if (!workspace_->hasBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE))
            {
                throw std::runtime_error(
                    "[CUDAQuantisedGemmKernel] Workspace missing required buffer: " +
                    std::string(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE));
            }

            LOG_TRACE("[CUDAQuantisedGemmKernel::validateWorkspace] Workspace validated"
                      << " A_int8=" << workspace_->getBuffer(GemmWorkspaceBuffers::QUANT_A)
                      << " scales_A=" << workspace_->getBuffer(GemmWorkspaceBuffers::SCALES_A)
                      << " scales_A_blockwise=" << workspace_->getBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE)
                      << " C_int32=" << workspace_->getBuffer(GemmWorkspaceBuffers::ACC_INT32));
        }

        // =====================================================================
        // ITensorGemm interface - multiply_tensor() PRIMARY ENTRY POINT
        // =====================================================================

        bool CUDAQuantisedGemmKernel::multiply_tensor(
            const TensorBase *A, TensorBase *C,
            bool transpose_B,
            float alpha, float beta,
            const TensorBase *bias,
            const MPIContext *mpi_ctx,
            int device_idx,
            DeviceWorkspaceManager *workspace,
            int activation_row_offset)
        {
            (void)bias; // TODO: Implement bias support
            if (!A || !C)
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_tensor] Null input or output tensor");
                return false;
            }

            int m = static_cast<int>(A->rows());
            int n = static_cast<int>(N_);
            int k = static_cast<int>(K_);

            return multiply_tensor(A, C, m, n, k, transpose_B, alpha, beta, bias, mpi_ctx, device_idx, workspace, activation_row_offset);
        }

        bool CUDAQuantisedGemmKernel::multiply_tensor(
            const TensorBase *A, TensorBase *C,
            int m, int n, int k,
            bool /*transpose_B*/,
            float alpha, float beta,
            const TensorBase *bias,
            const MPIContext * /*mpi_ctx*/,
            int /*device_idx*/,
            DeviceWorkspaceManager *workspace,
            int activation_row_offset)
        {
            // Use passed workspace if provided, otherwise fall back to bound workspace
            DeviceWorkspaceManager *effective_ws = workspace ? workspace : workspace_;
            if (effective_ws != workspace_)
            {
                // Temporarily use passed workspace for this call
                DeviceWorkspaceManager *saved_ws = workspace_;
                workspace_ = effective_ws;
                bool result = multiply_tensor_impl(A, C, m, n, k, alpha, beta, bias, activation_row_offset);
                workspace_ = saved_ws;
                return result;
            }
            return multiply_tensor_impl(A, C, m, n, k, alpha, beta, bias, activation_row_offset);
        }

        bool CUDAQuantisedGemmKernel::multiply_tensor_impl(
            const TensorBase *A, TensorBase *C,
            int m, int n, int k,
            float alpha, float beta,
            const TensorBase *bias,
            int activation_row_offset)
        {
            if (!A || !C)
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_tensor] Null input or output tensor");
                return false;
            }

            // Coherence handled automatically by DeviceGraphExecutor

            // Ensure weights are converted
            ensureWeightsConverted();

            // Type dispatch based on A and C types
            TensorType a_type = A->native_type();
            TensorType c_type = C->native_type();

            // =================================================================
            // MAPPED OUTPUT REDIRECT: Detect host-mapped FP32 output memory.
            // Mapped memory (used for logits) causes PCIe-speed scattered writes
            // (~12 GB/s) instead of HBM-speed writes (~900 GB/s on RTX 3090).
            // Fix: redirect kernel output to HBM workspace, then bulk DMA.
            // =================================================================
            const bool output_is_mapped = (c_type == TensorType::FP32) && C->isMapped();
            float *d_mapped_output = nullptr; // original mapped pointer for DMA copy

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

                // Redirect mapped output to HBM workspace
                if (output_is_mapped)
                {
                    validateWorkspace();
                    d_mapped_output = d_C;
                    d_C = static_cast<float *>(workspace_->getBuffer(GemmWorkspaceBuffers::TEMP_C_FP32));
                    static std::once_flag q8_mapped_once;
                    std::call_once(q8_mapped_once, [&]()
                                   { LOG_WARN("[CUDAQuantisedGemmKernel] Q8→FP32 MAPPED REDIRECT: M=" << m << " N=" << n
                                                                                                      << " mapped_ptr=" << d_mapped_output << " -> workspace=" << d_C
                                                                                                      << " (" << (static_cast<size_t>(m) * n * 4 / 1024) << " KB)"); });
                }

                bool success = multiply_q8_to_fp32(d_A_q8, d_C, m, n, k, alpha, beta);

                // Bulk DMA from HBM workspace to mapped output
                if (success && output_is_mapped)
                {
                    cudaQuantGemm_copyDeviceToDeviceAsync(
                        d_mapped_output, d_C,
                        static_cast<size_t>(m) * n,
                        cuda_device_id_, gpu_stream_);
                }
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

                // Redirect mapped output to HBM workspace
                if (output_is_mapped)
                {
                    validateWorkspace();
                    d_mapped_output = d_C;
                    d_C = static_cast<float *>(workspace_->getBuffer(GemmWorkspaceBuffers::TEMP_C_FP32));
                    static std::once_flag fp32_mapped_once;
                    std::call_once(fp32_mapped_once, [&]()
                                   { LOG_WARN("[CUDAQuantisedGemmKernel] FP32→FP32 MAPPED REDIRECT: M=" << m << " N=" << n
                                                                                                        << " mapped_ptr=" << d_mapped_output << " -> workspace=" << d_C
                                                                                                        << " (" << (static_cast<size_t>(m) * n * 4 / 1024) << " KB)"); });
                }

                // Apply activation row offset
                if (activation_row_offset > 0)
                {
                    d_A += static_cast<size_t>(activation_row_offset) * k;
                }

                // Extract bias pointer if present
                const float *d_bias = bias ? static_cast<const float *>(bias->gpu_data_ptr()) : nullptr;

                bool success;
                if (d_bias)
                {
                    success = multiply_fp32_to_fp32_with_bias(d_A, d_C, d_bias, m, n, k, alpha, beta);
                }
                else
                {
                    success = multiply_fp32_to_fp32(d_A, d_C, m, n, k, alpha, beta);
                }

                // Bulk DMA from HBM workspace to mapped output
                if (success && output_is_mapped)
                {
                    cudaQuantGemm_copyDeviceToDeviceAsync(
                        d_mapped_output, d_C,
                        static_cast<size_t>(m) * n,
                        cuda_device_id_, gpu_stream_);
                }
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
            int /*device_idx*/,
            DeviceWorkspaceManager *workspace)
        {
            // Use passed workspace if provided, otherwise fall back to bound workspace
            DeviceWorkspaceManager *effective_ws = workspace ? workspace : workspace_;
            if (effective_ws != workspace_)
            {
                // Temporarily use passed workspace for this call
                DeviceWorkspaceManager *saved_ws = workspace_;
                workspace_ = effective_ws;
                bool result = multiply_fp32_to_fp32(A, C, m, n, k, alpha, beta);
                workspace_ = saved_ws;
                return result;
            }
            return multiply_fp32_to_fp32(A, C, m, n, k, alpha, beta);
        }

        // =====================================================================
        // ITensorGemm interface - multiply_fused() for fused projection stages
        // =====================================================================

        bool CUDAQuantisedGemmKernel::multiply_fused(
            const float *input,
            const std::vector<FusedProjectionDesc> &projections,
            int m, int k,
            const MPIContext * /*mpi_ctx*/,
            int /*device_idx*/,
            DeviceWorkspaceManager *workspace)
        {
            // Use passed workspace if provided, otherwise fall back to bound workspace
            DeviceWorkspaceManager *effective_ws = workspace ? workspace : workspace_;
            if (effective_ws != workspace_)
            {
                // Temporarily use passed workspace for this call
                DeviceWorkspaceManager *saved_ws = workspace_;
                workspace_ = effective_ws;
                bool result = multiply_fused_impl(input, projections, m, k);
                workspace_ = saved_ws;
                return result;
            }
            return multiply_fused_impl(input, projections, m, k);
        }

        bool CUDAQuantisedGemmKernel::multiply_fused_impl(
            const float *input,
            const std::vector<FusedProjectionDesc> &projections,
            int m, int k)
        {
            if (!input || projections.empty())
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused] Null input or empty projections");
                return false;
            }

            if (m <= 0 || k <= 0)
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused] Invalid dimensions: m=" << m << " k=" << k);
                return false;
            }

            if (!gpu_stream_)
                cudaQuantGemm_setDevice(cuda_device_id_);

            // Step 1: Copy input from host to device
            const size_t input_count = static_cast<size_t>(m) * k;
            float *d_input = nullptr;
            if (!cudaQuantGemm_allocFloat(&d_input, input_count, cuda_device_id_))
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused] Failed to allocate device input");
                return false;
            }

            if (!cudaQuantGemm_copyHostToDevice(d_input, input, input_count, cuda_device_id_))
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused] Failed to copy input to device");
                cudaQuantGemm_freeDevice(d_input);
                return false;
            }

            // Step 2: Execute each projection
            bool all_success = true;
            for (size_t i = 0; i < projections.size() && all_success; ++i)
            {
                const auto &proj = projections[i];
                if (!proj.kernel || !proj.output)
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused] Projection " << i << " has null kernel or output");
                    all_success = false;
                    break;
                }

                const int n = proj.n;
                const size_t output_count = static_cast<size_t>(m) * n;

                // Allocate device output
                float *d_output = nullptr;
                if (!cudaQuantGemm_allocFloat(&d_output, output_count, cuda_device_id_))
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused] Failed to allocate device output for projection " << i);
                    all_success = false;
                    break;
                }

                // Get the CUDA kernel for this projection
                auto *cuda_kernel = dynamic_cast<CUDAQuantisedGemmKernel *>(proj.kernel);
                if (!cuda_kernel)
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused] Projection " << i
                                                                                      << " kernel is not a CUDAQuantisedGemmKernel");
                    cudaQuantGemm_freeDevice(d_output);
                    all_success = false;
                    break;
                }

                // Upload bias to device if present
                float *d_bias = nullptr;
                if (proj.bias)
                {
                    if (!cudaQuantGemm_allocFloat(&d_bias, n, cuda_device_id_))
                    {
                        LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused] Failed to allocate device bias for projection " << i);
                        cudaQuantGemm_freeDevice(d_output);
                        all_success = false;
                        break;
                    }
                    if (!cudaQuantGemm_copyHostToDevice(d_bias, proj.bias->data(), n, cuda_device_id_))
                    {
                        LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused] Failed to copy bias to device for projection " << i);
                        cudaQuantGemm_freeDevice(d_bias);
                        cudaQuantGemm_freeDevice(d_output);
                        all_success = false;
                        break;
                    }
                }

                // Run the GEMM with device pointers (with fused bias)
                bool success = cuda_kernel->multiply_fp32_to_fp32_with_bias(d_input, d_output, d_bias, m, n, k, 1.0f, 0.0f);
                if (d_bias)
                {
                    cudaQuantGemm_freeDevice(d_bias);
                }
                if (!success)
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused] GEMM failed for projection " << i);
                    cudaQuantGemm_freeDevice(d_output);
                    all_success = false;
                    break;
                }

                // Copy output back to host
                if (!cudaQuantGemm_copyDeviceToHost(proj.output, d_output, output_count, cuda_device_id_))
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused] Failed to copy output from device for projection " << i);
                    cudaQuantGemm_freeDevice(d_output);
                    all_success = false;
                    break;
                }
                cudaQuantGemm_freeDevice(d_output);
            }

            cudaQuantGemm_freeDevice(d_input);
            return all_success;
        }

        // =====================================================================
        // ITensorGemm interface - multiply_fused_tensor() for TensorBase API
        // =====================================================================

        bool CUDAQuantisedGemmKernel::multiply_fused_tensor(
            const TensorBase *input,
            const std::vector<TensorProjectionDesc> &projections,
            int m, int k,
            const MPIContext * /*mpi_ctx*/,
            DeviceWorkspaceManager *workspace)
        {
            // Use passed workspace if provided, otherwise fall back to bound workspace
            DeviceWorkspaceManager *effective_ws = workspace ? workspace : workspace_;
            if (effective_ws != workspace_)
            {
                // Temporarily use passed workspace for this call
                DeviceWorkspaceManager *saved_ws = workspace_;
                workspace_ = effective_ws;
                bool result = multiply_fused_tensor_impl(input, projections, m, k);
                workspace_ = saved_ws;
                return result;
            }
            return multiply_fused_tensor_impl(input, projections, m, k);
        }

        bool CUDAQuantisedGemmKernel::multiply_fused_tensor_impl(
            const TensorBase *input,
            const std::vector<TensorProjectionDesc> &projections,
            int m, int k)
        {
            if (!input || projections.empty())
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Null input or empty projections");
                return false;
            }

            if (m <= 0 || k <= 0)
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Invalid dimensions: m=" << m << " k=" << k);
                return false;
            }

            if (!gpu_stream_)
                cudaQuantGemm_setDevice(cuda_device_id_);
            DeviceId target_device = DeviceId::cuda(cuda_device_id_);

            // Step 1: Ensure input is on the GPU
            const float *d_input = nullptr;
            if (input->native_type() == TensorType::FP32)
            {
                auto *fp32_input = dynamic_cast<FP32Tensor *>(const_cast<TensorBase *>(input));
                if (!fp32_input)
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Failed to cast input to FP32Tensor");
                    return false;
                }
                // Coherence handled automatically by DeviceGraphExecutor
                d_input = static_cast<const float *>(fp32_input->gpu_data_ptr());
                // NOTE: Don't log fp32_input->data() here - it triggers D2H transfer!
                LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Input GPU ptr=" << d_input);
            }
            else
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Unsupported input type: "
                          << static_cast<int>(input->native_type()));
                return false;
            }

            if (!d_input)
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Input has no GPU data");
                return false;
            }

            // Step 2: Validate workspace and get buffer pointers
            // Use this kernel's workspace for quantized activations (shared across all projections)
            validateWorkspace();
            int8_t *d_A_int8 = static_cast<int8_t *>(workspace_->getBuffer(GemmWorkspaceBuffers::QUANT_A));

            // Use blockwise quantization for prefill and for decode when a native
            // payload GEMV path is available.
            const bool use_blockwise = (k % 32 == 0) && (m > 1 || canUseNativePayloadBlockwise(impl_.get(), m, k));
            float *d_scales_A = nullptr;
            float *d_scales_A_blockwise = nullptr;

            if (use_blockwise)
            {
                d_scales_A_blockwise = static_cast<float *>(workspace_->getBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE));
                LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Blockwise quantizing activations once, m=" << m << " k=" << k);

                // Step 3a: Blockwise quantize activations ONCE (shared across all projections)
                if (!cudaQuantGemm_quantizeActivationsBlockwise(
                        d_input, d_A_int8, d_scales_A_blockwise, m, k, cuda_device_id_, gpu_stream_))
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Blockwise activation quantization failed");
                    return false;
                }
            }
            else
            {
                d_scales_A = static_cast<float *>(workspace_->getBuffer(GemmWorkspaceBuffers::SCALES_A));
                LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Row-wise quantizing activations once, m=" << m << " k=" << k);

                // Step 3b: Row-wise quantize activations ONCE (shared across all projections)
                if (!cudaQuantGemm_quantizeActivations(
                        d_input, d_A_int8, d_scales_A, m, k, cuda_device_id_, gpu_stream_))
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Activation quantization failed");
                    return false;
                }
            }

            // Step 4: Execute each projection using the SHARED quantized activations
            bool all_success = true;
            for (size_t i = 0; i < projections.size() && all_success; ++i)
            {
                const auto &proj = projections[i];
                if (!proj.kernel || !proj.output)
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Projection " << i << " has null kernel or output");
                    all_success = false;
                    break;
                }

                // Get the CUDA kernel for this projection
                auto *cuda_kernel = dynamic_cast<CUDAQuantisedGemmKernel *>(proj.kernel);
                if (!cuda_kernel)
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                             << " kernel is not a CUDAQuantisedGemmKernel");
                    all_success = false;
                    break;
                }

                const int n = proj.n;
                LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                         << " (" << (proj.name ? proj.name : "unnamed") << "): m=" << m << " n=" << n << " k=" << k);

                // Ensure the projection's weights are converted
                cuda_kernel->ensureWeightsConverted();

                // Validate this projection's workspace is bound and get its d_C_int32 buffer
                cuda_kernel->validateWorkspace();
                int32_t *proj_d_C_int32 = static_cast<int32_t *>(
                    cuda_kernel->workspace_->getBuffer(GemmWorkspaceBuffers::ACC_INT32));

                // Ensure output tensor is on device
                if (proj.output->native_type() != TensorType::FP32)
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                             << " output must be FP32, got " << static_cast<int>(proj.output->native_type()));
                    all_success = false;
                    break;
                }

                auto *fp32_output = dynamic_cast<FP32Tensor *>(proj.output);
                if (!fp32_output)
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Failed to cast output to FP32Tensor");
                    all_success = false;
                    break;
                }

                // Coherence handled automatically by DeviceGraphExecutor
                float *d_output = static_cast<float *>(fp32_output->gpu_data_ptr());
                if (!d_output)
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Output has no GPU data for projection " << i);
                    all_success = false;
                    break;
                }

                // Get bias pointer if present (needed for both CUTLASS and blockwise paths)
                const float *d_bias = nullptr;
                if (proj.bias)
                {
                    if (proj.bias->native_type() != TensorType::FP32)
                    {
                        LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                                 << " bias must be FP32, got " << static_cast<int>(proj.bias->native_type()));
                        all_success = false;
                        break;
                    }

                    // Handle TensorSlice - unwrap to get inner FP32Tensor
                    const TensorBase *bias_tensor = proj.bias;
                    bool was_slice = false;
                    const void *slice_ptr = nullptr;
                    if (auto *slice = dynamic_cast<const TensorSlice *>(proj.bias))
                    {
                        slice_ptr = slice;
                        bias_tensor = slice->inner();
                        was_slice = true;
                    }

                    auto *fp32_bias = dynamic_cast<FP32Tensor *>(const_cast<TensorBase *>(bias_tensor));
                    if (!fp32_bias)
                    {
                        LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Failed to cast bias to FP32Tensor"
                                  << " | was_slice=" << was_slice
                                  << " | bias_tensor=" << bias_tensor
                                  << " | native_type=" << static_cast<int>(bias_tensor->native_type()));
                        all_success = false;
                        break;
                    }

                    // Check if bias is already on the correct CUDA device
                    // In multi-GPU scenarios, each device should have its own bias tensor clone
                    // (created during weight preloading via WeightPreloader::uploadNonGemmWeights)
                    DeviceId target_device = DeviceId::cuda(cuda_device_id_);
                    auto current_dev = fp32_bias->current_device();

                    if (current_dev.has_value() && current_dev.value() == target_device)
                    {
                        // Already on correct device - use directly
                        d_bias = static_cast<const float *>(fp32_bias->gpu_data_ptr());
                    }
                    else if (current_dev.has_value() && current_dev->is_gpu())
                    {
                        // Tensor is on a DIFFERENT GPU - this is a multi-GPU race condition!
                        // Do NOT call ensureOnDevice() as it would free the other GPU's memory.
                        // The correct fix is to ensure each device has its own bias tensor clone.
                        LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] MULTI-GPU CONFLICT: Bias tensor is on "
                                  << current_dev->to_string() << " but we need CUDA:" << cuda_device_id_
                                  << ". Ensure WeightPreloader::uploadNonGemmWeights() was called for this device.");
                        all_success = false;
                        break;
                    }
                    else
                    {
                        // Tensor is on CPU or not uploaded yet - safe to upload to this device
                        if (!fp32_bias->ensureOnDevice(target_device))
                        {
                            LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Failed to upload bias to CUDA:" << cuda_device_id_);
                            all_success = false;
                            break;
                        }
                        d_bias = static_cast<const float *>(fp32_bias->gpu_data_ptr());
                    }

                    if (!d_bias)
                    {
                        LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Bias has no GPU data for projection " << i
                                                                                                                          << " | was_slice=" << was_slice
                                                                                                                          << " | slice_ptr=" << slice_ptr
                                                                                                                          << " | fp32_bias=" << fp32_bias
                                                                                                                          << " | numel=" << fp32_bias->numel()
                                                                                                                          << " | host_data=" << fp32_bias->data()
                                                                                                                          << " | device_valid=" << fp32_bias->isDeviceValid()
                                                                                                                          << " | device=" << (fp32_bias->current_device().has_value() ? fp32_bias->current_device()->to_string() : "none"));
                        all_success = false;
                        break;
                    }
                    LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                             << " using bias ptr=" << static_cast<const void *>(d_bias));
                }

                if (use_blockwise)
                {
                        if (runNativePayloadBlockwiseIfSupported(
                            cuda_kernel->impl_.get(),
                            d_A_int8,
                            d_output,
                            d_scales_A_blockwise,
                            m, n, k,
                            1.0f, 0.0f,
                            nullptr,
                            d_bias,
                            cuda_device_id_, gpu_stream_))
                    {
                        continue;
                    }

                    // Blockwise GEMM: produces final FP32 output directly (includes per-block scales, weight scales, bias)
                    CUDA_KERNEL_PROFILE_SCOPE(CUDAKernelType::GEMM);
                        if (!runSelectedBlockwiseBackend(
                            d_A_int8,
                            cuda_kernel->impl_->d_weights_int8,
                            cuda_kernel->impl_->d_weights_int8_tc_blocked,
                            proj_d_C_int32,
                            d_output,
                            d_scales_A_blockwise,
                            cuda_kernel->impl_->d_scales_B,
                            m, n, k,
                            1.0f, 0.0f,
                            nullptr,
                            d_bias,
                            cuda_device_id_, gpu_stream_))
                    {
                        LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Blockwise GEMM failed for projection " << i);
                        all_success = false;
                        break;
                    }
                }
                else
                {
                    // CUTLASS INT8 GEMM → INT32 accumulator, then separate scaling epilogue
                    {
                        CUDA_KERNEL_PROFILE_SCOPE(CUDAKernelType::GEMM);
                        if (!cudaQuantGemm_execute(
                                d_A_int8,                           // SHARED quantized activations (from this kernel's workspace)
                                cuda_kernel->impl_->d_weights_int8, // This projection's weights
                                proj_d_C_int32,                     // This projection's INT32 work buffer (from its workspace)
                                m, n, k, cuda_device_id_, gpu_stream_))
                        {
                            LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] CUTLASS GEMM failed for projection " << i);
                            all_success = false;
                            break;
                        }
                    }

                    // Apply scaling: output = int32_accum * scales_A * scales_B + bias
                    // Note: Use the SHARED scales_A from the quantized activations (from this kernel's workspace)
                    if (!cudaQuantGemm_applyScaling(
                            proj_d_C_int32,                 // This projection's INT32 result
                            d_output,                       // Output FP32
                            d_scales_A,                     // SHARED activation scales (from this kernel's workspace)
                            cuda_kernel->impl_->d_scales_B, // This projection's weight scales
                            m, n, 1.0f, 0.0f, nullptr, d_bias, cuda_device_id_, gpu_stream_))
                    {
                        LOG_ERROR("[CUDAQuantisedGemmKernel::multiply_fused_tensor] Scaling failed for projection " << i);
                        all_success = false;
                        break;
                    }
                }

#ifdef LLAMINAR_DEBUG_GEMM_VALUES
                // Debug: Log output values after scaling - EXPENSIVE, guarded by compile flag
                // CRITICAL: cudaMemcpy(D2H) on default stream is illegal during graph capture
                if (d_output && m > 0 && n > 0)
                {
                    size_t copy_count = std::min(static_cast<size_t>(m) * static_cast<size_t>(n), static_cast<size_t>(8));
                    std::vector<float> h_output(copy_count);
                    cudaQuantGemm_copyDeviceToHost(h_output.data(), d_output, h_output.size(), cuda_device_id_);
                    LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_fused_tensor] " << (proj.name ? proj.name : "unnamed")
                                                                                  << " output[0:4]=" << h_output[0] << "," << (h_output.size() > 1 ? h_output[1] : 0.f) << ","
                                                                                  << (h_output.size() > 2 ? h_output[2] : 0.f) << "," << (h_output.size() > 3 ? h_output[3] : 0.f));
                }
#endif
            }

            return all_success;
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
            LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_fp32_to_fp32] m=" << m << " n=" << n << " k=" << k
                                                                            << " alpha=" << alpha << " beta=" << beta
                                                                            << " d_A=" << static_cast<const void *>(d_A)
                                                                            << " d_C=" << static_cast<void *>(d_C));

            // Validate workspace is bound (REQUIRED - kernels don't do internal allocation)
            validateWorkspace();

            // Get workspace buffer pointers
            int8_t *d_A_int8 = static_cast<int8_t *>(workspace_->getBuffer(GemmWorkspaceBuffers::QUANT_A));
            int32_t *d_C_int32 = static_cast<int32_t *>(workspace_->getBuffer(GemmWorkspaceBuffers::ACC_INT32));

            // Ensure weights converted
            ensureWeightsConverted();

            // Use blockwise quantization for prefill and for decode when a native
            // payload GEMV path is available.
            const bool use_blockwise = (k % 32 == 0) && (m > 1 || canUseNativePayloadBlockwise(impl_.get(), m, k));

            if (use_blockwise)
            {
                float *d_scales_A_blockwise = static_cast<float *>(workspace_->getBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE));

                // Step 1: Blockwise quantize activations
                if (!cudaQuantGemm_quantizeActivationsBlockwise(
                        d_A, d_A_int8, d_scales_A_blockwise, m, k, cuda_device_id_, gpu_stream_))
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel] Blockwise activation quantization failed");
                    return false;
                }

                const float *d_C_existing = (beta != 0.0f) ? d_C : nullptr;
                if (runNativePayloadBlockwiseIfSupported(
                        impl_.get(),
                        d_A_int8,
                        d_C,
                        d_scales_A_blockwise,
                        m, n, k,
                        alpha, beta,
                        d_C_existing,
                        nullptr,
                        cuda_device_id_, gpu_stream_))
                {
                    LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_fp32_to_fp32] Complete (native payload GEMV)");
                    return true;
                }

                // Step 2: Blockwise GEMM (produces FP32 directly, includes scaling)
                {
                    CUDA_KERNEL_PROFILE_SCOPE(CUDAKernelType::GEMM);
                        if (!runSelectedBlockwiseBackend(
                            d_A_int8,
                            impl_->d_weights_int8,
                            impl_->d_weights_int8_tc_blocked,
                            d_C_int32,
                            d_C,
                            d_scales_A_blockwise,
                            impl_->d_scales_B,
                            m, n, k,
                            alpha, beta,
                            d_C_existing,
                            nullptr,
                            cuda_device_id_, gpu_stream_))
                    {
                        LOG_ERROR("[CUDAQuantisedGemmKernel] Blockwise GEMM failed");
                        return false;
                    }
                }

                LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_fp32_to_fp32] Complete (blockwise)");
                return true;
            }

            // Row-wise CUTLASS path (fallback when K not divisible by 32)
            float *d_scales_A = static_cast<float *>(workspace_->getBuffer(GemmWorkspaceBuffers::SCALES_A));

#ifdef LLAMINAR_DEBUG_GEMM_VALUES
            // Debug: Sample scales_B (weight scales) - EXPENSIVE, guarded by compile flag
            LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_fp32_to_fp32] d_scales_B="
                      << static_cast<const void *>(impl_->d_scales_B)
                      << " d_weights_int8=" << static_cast<const void *>(impl_->d_weights_int8));
            if (impl_->d_scales_B && n > 0)
            {
                std::vector<float> h_scales_B(std::min(n, 8));
                cudaQuantGemm_copyDeviceToHost(h_scales_B.data(), impl_->d_scales_B, h_scales_B.size(), cuda_device_id_);
                LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_fp32_to_fp32] scales_B[0:4]="
                          << h_scales_B[0] << "," << (h_scales_B.size() > 1 ? h_scales_B[1] : 0.f) << ","
                          << (h_scales_B.size() > 2 ? h_scales_B[2] : 0.f) << "," << (h_scales_B.size() > 3 ? h_scales_B[3] : 0.f));
            }
#endif

            // Step 1: Quantize activations
            if (!cudaQuantGemm_quantizeActivations(
                    d_A, d_A_int8, d_scales_A, m, k, cuda_device_id_, gpu_stream_))
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel] Activation quantization failed");
                return false;
            }

#ifdef LLAMINAR_DEBUG_GEMM_VALUES
            // Debug: dump scales_A (activation row scales) - EXPENSIVE, guarded by compile flag
            if (d_scales_A && m > 0)
            {
                std::vector<float> h_scales_A(std::min(m, 8));
                cudaQuantGemm_copyDeviceToHost(h_scales_A.data(), d_scales_A, h_scales_A.size(), cuda_device_id_);
                LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_fp32_to_fp32] scales_A[0:4]="
                          << h_scales_A[0] << "," << (h_scales_A.size() > 1 ? h_scales_A[1] : 0.f) << ","
                          << (h_scales_A.size() > 2 ? h_scales_A[2] : 0.f) << "," << (h_scales_A.size() > 3 ? h_scales_A[3] : 0.f));
            }
#endif

            // Step 2: Execute CUTLASS INT8 GEMM
            {
                CUDA_KERNEL_PROFILE_SCOPE(CUDAKernelType::GEMM);
                if (!cudaQuantGemm_execute(
                        d_A_int8, impl_->d_weights_int8, d_C_int32,
                        m, n, k, cuda_device_id_, gpu_stream_))
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel] CUTLASS GEMM failed");
                    return false;
                }
            }

#ifdef LLAMINAR_DEBUG_GEMM_VALUES
            // Debug: dump some int32 outputs - EXPENSIVE, guarded by compile flag
            if (d_C_int32 && m > 0 && n > 0)
            {
                size_t copy_count = std::min(static_cast<size_t>(m) * static_cast<size_t>(n), static_cast<size_t>(8));
                std::vector<int32_t> h_C_int32(copy_count);
                cudaQuantGemm_copyInt32DeviceToHost(h_C_int32.data(), d_C_int32, h_C_int32.size(), cuda_device_id_);
                LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_fp32_to_fp32] C_int32[0:4]="
                          << h_C_int32[0] << "," << (h_C_int32.size() > 1 ? h_C_int32[1] : 0) << ","
                          << (h_C_int32.size() > 2 ? h_C_int32[2] : 0) << "," << (h_C_int32.size() > 3 ? h_C_int32[3] : 0));

                // For LMHead (large n), dump row 1 as well
                if (m == 2 && n > 1000)
                {
                    std::vector<int32_t> h_C_int32_row1(8);
                    cudaQuantGemm_copyInt32DeviceToHost(h_C_int32_row1.data(), d_C_int32 + n, 8, cuda_device_id_);
                    LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_fp32_to_fp32] C_int32[row1,0:4]="
                              << h_C_int32_row1[0] << "," << h_C_int32_row1[1] << ","
                              << h_C_int32_row1[2] << "," << h_C_int32_row1[3]);
                }
            }
#endif

            // Step 3: Apply scaling and output to FP32 (no bias)
            const float *d_C_existing = (beta != 0.0f) ? d_C : nullptr;
            if (!cudaQuantGemm_applyScaling(
                    d_C_int32, d_C, d_scales_A, impl_->d_scales_B,
                    m, n, alpha, beta, d_C_existing, nullptr, cuda_device_id_, gpu_stream_))
            {
                LOG_ERROR("[CUDAQuantisedGemmKernel] Scaling failed");
                return false;
            }

#ifdef LLAMINAR_DEBUG_GEMM_VALUES
            // Debug: dump final FP32 outputs - EXPENSIVE, guarded by compile flag
            if (d_C && m > 0 && n > 0)
            {
                size_t copy_count = std::min(static_cast<size_t>(m) * static_cast<size_t>(n), static_cast<size_t>(8));
                std::vector<float> h_C_fp32(copy_count);
                cudaQuantGemm_copyDeviceToHost(h_C_fp32.data(), d_C, h_C_fp32.size(), cuda_device_id_);
                LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_fp32_to_fp32] C_fp32[0:4]="
                          << h_C_fp32[0] << "," << (h_C_fp32.size() > 1 ? h_C_fp32[1] : 0.f) << ","
                          << (h_C_fp32.size() > 2 ? h_C_fp32[2] : 0.f) << "," << (h_C_fp32.size() > 3 ? h_C_fp32[3] : 0.f));

                // For LMHead (large n), dump row 1 as well
                if (m == 2 && n > 1000)
                {
                    std::vector<float> h_C_fp32_row1(8);
                    cudaQuantGemm_copyDeviceToHost(h_C_fp32_row1.data(), d_C + n, 8, cuda_device_id_);
                    LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_fp32_to_fp32] C_fp32[row1,0:4]="
                              << h_C_fp32_row1[0] << "," << h_C_fp32_row1[1] << ","
                              << h_C_fp32_row1[2] << "," << h_C_fp32_row1[3]);
                }
            }
#endif

            LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_fp32_to_fp32] Complete");
            return true;
        }

        bool CUDAQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias(
            const float *d_A, float *d_C, const float *d_bias,
            int m, int n, int k,
            float alpha, float beta)
        {
            // Validate workspace is bound (REQUIRED - kernels don't do internal allocation)
            validateWorkspace();

            // Get workspace buffer pointers
            int8_t *d_A_int8 = static_cast<int8_t *>(workspace_->getBuffer(GemmWorkspaceBuffers::QUANT_A));
            int32_t *d_C_int32 = static_cast<int32_t *>(workspace_->getBuffer(GemmWorkspaceBuffers::ACC_INT32));

            // Ensure weights converted
            ensureWeightsConverted();

            // Use blockwise quantization whenever K is block-aligned so decode can
            // share one quantization pass across projections and choose either the
            // NativePayload or Int8Expanded GEMV path per projection.
            const bool use_blockwise = (k % 32 == 0);

            if (use_blockwise)
            {
                float *d_scales_A_blockwise = static_cast<float *>(workspace_->getBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE));

                // Step 1: Blockwise quantize activations
                {
                    CUDA_KERNEL_PROFILE_SCOPE(CUDAKernelType::QUANTIZE_ACTIVATIONS);
                    if (!cudaQuantGemm_quantizeActivationsBlockwise(
                            d_A, d_A_int8, d_scales_A_blockwise, m, k, cuda_device_id_, gpu_stream_))
                    {
                        LOG_ERROR("[CUDAQuantisedGemmKernel] Blockwise activation quantization failed");
                        return false;
                    }
                }

                const float *d_C_existing = (beta != 0.0f) ? d_C : nullptr;
                if (runNativePayloadBlockwiseIfSupported(
                        impl_.get(),
                        d_A_int8,
                        d_C,
                        d_scales_A_blockwise,
                        m, n, k,
                        alpha, beta,
                        d_C_existing,
                        d_bias,
                        cuda_device_id_, gpu_stream_))
                {
                    LOG_DEBUG("[CUDAQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] Complete (native payload GEMV)");
                    return true;
                }

                // Step 2: Blockwise GEMM (produces FP32 directly, includes scaling + bias)
                {
                    CUDA_KERNEL_PROFILE_SCOPE(CUDAKernelType::GEMM);
                        if (!runSelectedBlockwiseBackend(
                            d_A_int8,
                            impl_->d_weights_int8,
                            impl_->d_weights_int8_tc_blocked,
                            d_C_int32,
                            d_C,
                            d_scales_A_blockwise,
                            impl_->d_scales_B,
                            m, n, k,
                            alpha, beta,
                            d_C_existing,
                            d_bias,
                            cuda_device_id_, gpu_stream_))
                    {
                        LOG_ERROR("[CUDAQuantisedGemmKernel] Blockwise GEMM with bias failed");
                        return false;
                    }
                }

                return true;
            }

            // Row-wise CUTLASS path (fallback when K not divisible by 32)
            float *d_scales_A = static_cast<float *>(workspace_->getBuffer(GemmWorkspaceBuffers::SCALES_A));

            // Step 1: Quantize activations
            {
                CUDA_KERNEL_PROFILE_SCOPE(CUDAKernelType::QUANTIZE_ACTIVATIONS);
                if (!cudaQuantGemm_quantizeActivations(
                        d_A, d_A_int8, d_scales_A, m, k, cuda_device_id_, gpu_stream_))
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel] Activation quantization failed");
                    return false;
                }
            }

            // Step 2: Execute CUTLASS INT8 GEMM
            {
                CUDA_KERNEL_PROFILE_SCOPE(CUDAKernelType::GEMM);
                if (!cudaQuantGemm_execute(
                        d_A_int8, impl_->d_weights_int8, d_C_int32,
                        m, n, k, cuda_device_id_, gpu_stream_))
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel] CUTLASS GEMM failed");
                    return false;
                }
            }

            // Step 3: Apply scaling, bias, and output to FP32
            const float *d_C_existing = (beta != 0.0f) ? d_C : nullptr;
            {
                CUDA_KERNEL_PROFILE_SCOPE(CUDAKernelType::GEMM_SCALE_OUTPUT);
                if (!cudaQuantGemm_applyScaling(
                        d_C_int32, d_C, d_scales_A, impl_->d_scales_B,
                        m, n, alpha, beta, d_C_existing, d_bias, cuda_device_id_, gpu_stream_))
                {
                    LOG_ERROR("[CUDAQuantisedGemmKernel] Scaling with bias failed");
                    return false;
                }
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

        // =====================================================================
        // IWorkspaceConsumer Interface Implementation
        // =====================================================================

        WorkspaceRequirements CUDAQuantisedGemmKernel::getWorkspaceRequirements(
            int m, int n, int k) const
        {
            WorkspaceRequirements reqs;

            // Use internal dimensions if not specified
            if (n == 0)
                n = static_cast<int>(N_);
            if (k == 0)
                k = static_cast<int>(K_);

            // INT8 path needs quantization + accumulator buffers
            size_t quant_a_bytes = static_cast<size_t>(m) * k * sizeof(int8_t);
            size_t scales_a_bytes = static_cast<size_t>(m) * sizeof(float);
            size_t acc_int32_bytes = static_cast<size_t>(m) * n * sizeof(int32_t);

            reqs.buffers.push_back({GemmWorkspaceBuffers::QUANT_A, quant_a_bytes, 256, true});
            reqs.buffers.push_back({GemmWorkspaceBuffers::SCALES_A, scales_a_bytes, 256, true});
            reqs.buffers.push_back({GemmWorkspaceBuffers::ACC_INT32, acc_int32_bytes, 256, true});

            // Blockwise activation quantization scales: one float per 32-element block
            size_t num_blocks_per_row = static_cast<size_t>((k + 31) / 32);
            size_t scales_a_blockwise_bytes = static_cast<size_t>(m) * num_blocks_per_row * sizeof(float);
            reqs.buffers.push_back({GemmWorkspaceBuffers::SCALES_A_BLOCKWISE, scales_a_blockwise_bytes, 256, true});

            // FP32 output workspace for mapped memory redirect
            // When output is host-mapped (e.g., logits), scattered GPU writes go over PCIe.
            // This buffer provides an HBM target; we bulk-DMA to mapped memory after.
            size_t temp_c_fp32_bytes = static_cast<size_t>(m) * n * sizeof(float);
            reqs.buffers.push_back({GemmWorkspaceBuffers::TEMP_C_FP32, temp_c_fp32_bytes, 256, true});

            LOG_DEBUG("[CUDAQuantisedGemmKernel::getWorkspaceRequirements] INT8 path: "
                      << "quant_a=" << (quant_a_bytes / 1024) << "KB, "
                      << "scales_a=" << (scales_a_bytes) << "B, "
                      << "scales_a_blockwise=" << (scales_a_blockwise_bytes) << "B, "
                      << "acc=" << (acc_int32_bytes / 1024) << "KB"
                      << ", temp_c_fp32=" << (temp_c_fp32_bytes / 1024) << "KB");

            return reqs;
        }

        void CUDAQuantisedGemmKernel::bindWorkspace(DeviceWorkspaceManager *workspace)
        {
            workspace_ = workspace;
            if (workspace)
            {
                LOG_DEBUG("[CUDAQuantisedGemmKernel] Bound workspace manager at " << (void *)workspace
                                                                                  << ", entering managed mode");
            }
            else
            {
                LOG_DEBUG("[CUDAQuantisedGemmKernel] Unbound workspace, returning to legacy mode");
            }
        }

        bool CUDAQuantisedGemmKernel::hasWorkspace() const
        {
            return workspace_ != nullptr;
        }

        DeviceWorkspaceManager *CUDAQuantisedGemmKernel::getWorkspace() const
        {
            return workspace_;
        }

    } // namespace cuda
} // namespace llaminar2
