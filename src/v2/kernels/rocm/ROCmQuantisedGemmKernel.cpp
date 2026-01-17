/**
 * @file ROCmQuantisedGemmKernel.cpp
 * @brief ITensorGemm adapter implementation for ComposableKernel INT8 quantized GEMM
 *
 * This is the C++ adapter that wraps the CK INT8 GEMM kernel. It implements
 * the full ITensorGemm interface and can be compiled with the regular C++ compiler
 * (not hipcc), avoiding MPI/TensorKernels.h compilation issues with HIP headers.
 *
 * ## Architecture
 *
 * ```
 * ROCmQuantisedGemmKernel.cpp (this file, compiled with g++)
 *       │
 *       │ extern "C" function calls
 *       ▼
 * ROCmQuantisedGemmKernel_CK.hip (compiled with hipcc)
 *       │
 *       │ CK template instantiation (3-way dispatch)
 *       ▼
 * ComposableKernel DeviceGemmMultipleD_Dl (mk_nk_mn layout)
 *   - 32×32 kernel  for M ≤ 32  (decode: M=1 handled natively)
 *   - 64×64 kernel  for 32 < M < 128
 *   - 128×128 kernel for M ≥ 128 (prefill: peak throughput)
 * ```
 *
 * ## Weight Conversion Pipeline (packWeightsToROCm) - ONE-TIME AT LOAD
 *
 * Model weights are stored in various quantized formats (IQ4_NL, Q8_0, Q4_K, etc.).
 * This kernel requires symmetric INT8 quantization with per-output-feature scales.
 *
 * Weight conversion happens ONCE at model load time, entirely on CPU:
 *
 * 1. **Dequantize (CPU)**: Original quantized weights → FP32 via fp32_data()
 * 2. **Requantize (CPU)**: FP32 → INT8 with per-row scales (symmetric, [-127,127])
 * 3. **Upload (H2D)**: Copy INT8 weights + scales to GPU memory
 *
 * After loading, GPU only sees INT8 data - no format-specific decode on device.
 * This keeps VRAM usage minimal and avoids needing GPU kernels for each quant format.
 *
 * ## Memory Layout Convention (mk_nk_mn - optimized for 128x128 tiles)
 *
 * | Matrix        | Shape    | Memory Layout | Element Access          | CK View         |
 * |---------------|----------|---------------|-------------------------|-----------------|
 * | Model Weights | [N × K]  | Row-Major     | W[n,k] = data[n*K + k]  | ColMajor [K×N]  |
 * | Activations   | [M × K]  | Row-Major     | A[m,k] = data[m*K + k]  | RowMajor [M×K]  |
 * | Output        | [M × N]  | Row-Major     | C[m,n] = data[m*N + n]  | RowMajor [M×N]  |
 *
 * Key insight: Model weights [N×K] Row-Major == Column-Major [K×N]!
 * No transpose needed - we just reinterpret the layout for CK's mk_nk_mn convention.
 * This enables 128×128 tile sizes (4x larger than mk_kn_mn) for ~2-4x speedup.
 *
 * ## Two-Kernel Execution Path (PER-INFERENCE)
 *
 * Production path uses rocmQuantGemm_executeTwoKernel_cached():
 *   1. Upload FP32 activations to GPU (H2D)
 *   2. Quantize activations FP32→INT8 on GPU (rocmQuantGemm_quantizeActivations)
 *   3. CK INT8×INT8→INT32 GEMM (no scaling in kernel)
 *   4. Separate applyScales_kernel: E[m,n] = C_int32[m,n] * scale_A[m] * scale_B[n]
 *   5. Download FP32 output to host (D2H)
 *
 * Activation quantization happens ON GPU every inference call - this is the
 * per-row symmetric quantization that produces INT8 activations + scales.
 *
 * The two-kernel GEMM approach is required because CK's fused D-tensor scaling
 * doesn't support the per-row × per-column broadcast pattern we need.
 *
 * @see ROCmQuantisedGemmKernel_CK.hip for HIP kernel implementation
 * @see ROCmQuantisedGemmKernel.h for class documentation
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "ROCmQuantisedGemmKernel.h"
#include "backends/ComputeBackend.h" // DeviceManager
#include "backends/DeviceId.h"       // DeviceId
#include "tensors/Tensors.h"         // Q8_1Tensor, FP32Tensor, etc.
#include "tensors/BlockStructures.h" // Q8_1Block
#include "tensors/KernelSnapshotInfo.h"
#include "execution/DeviceWorkspaceManager.h"
#include "execution/WorkspaceDescriptor.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/SlabGemmConfig.h"
#include "utils/Logger.h"

#include <stdexcept>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace llaminar2
{
    namespace rocm
    {

        // =====================================================================
        // Forward declarations for HIP implementation (defined in .hip file)
        // =====================================================================

        // These functions are implemented in ROCmQuantisedGemmKernel_CK.hip
        extern "C"
        {
            // Upload converted INT8 weights to device
            bool rocmQuantGemm_uploadWeights(
                const int8_t *h_weights_int8, // [K x N] ColumnMajor
                const float *h_scales_B,      // [N] per-column scales
                int8_t **d_weights_int8,      // Output device pointer
                float **d_scales_B,           // Output device pointer
                int K, int N,
                int rocm_device_id);

            // Upload work buffers for activation quantization
            bool rocmQuantGemm_ensureWorkBuffers(
                int8_t **d_A_int8,   // [M x K] quantized activations
                float **d_scales_A,  // [M] per-row scales
                int32_t **d_C_int32, // [M x N] INT32 accumulator
                int *work_buffer_M,  // Current capacity
                int M, int K, int N,
                int rocm_device_id);

            // Quantize FP32 activations to INT8
            bool rocmQuantGemm_quantizeActivations(
                const float *d_A_fp32, // [M x K]
                int8_t *d_A_int8,      // [M x K] output
                float *d_scales_A,     // [M] output
                int M, int K,
                int rocm_device_id);

            // Execute INT8 GEMM using CK with SEPARATE scaling (two-kernel approach)
            // This runs CK for INT8→INT32 GEMM, then a separate kernel for scaling.
            bool rocmQuantGemm_executeTwoKernel(
                const int8_t *d_A_int8, // [M x K] RowMajor quantized activations
                const int8_t *d_B_int8, // [K x N] RowMajor transposed weights
                float *d_E_fp32,        // [M x N] RowMajor FP32 output
                const float *d_scale_A, // [M] per-row activation scales
                const float *d_scale_B, // [N] per-column weight scales
                int M, int N, int K,
                int rocm_device_id);

            // Execute INT8 GEMM using CK two-kernel with PRE-ALLOCATED buffer (allocation-free)
            // This is the preferred variant for hot-path execution.
            bool rocmQuantGemm_executeTwoKernel_cached(
                const int8_t *d_A_int8, // [M x K] RowMajor quantized activations
                const int8_t *d_B_int8, // [K x N] RowMajor transposed weights
                float *d_E_fp32,        // [M x N] RowMajor FP32 output
                const float *d_scale_A, // [M] per-row activation scales
                const float *d_scale_B, // [N] per-column weight scales
                int32_t *d_C_int32,     // [M x N] Pre-allocated INT32 accumulator
                int M, int N, int K,
                int rocm_device_id);

            // Execute INT8 GEMM using CK two-kernel with M-PADDING for decode (M < 8)
            // Pads activations to padded_m, runs CK, extracts first M rows of output.
            // Note: With the 32×32 kernel, only M < 8 needs explicit padding.
            bool rocmQuantGemm_executeTwoKernel_padded(
                const int8_t *d_A_int8, // [M x K] RowMajor quantized activations
                const int8_t *d_B_int8, // [N x K] RowMajor weights (viewed as [K x N] col-major)
                float *d_E_fp32,        // [M x N] RowMajor FP32 output (only first M rows written)
                const float *d_scale_A, // [M] per-row activation scales
                const float *d_scale_B, // [N] per-column weight scales
                int32_t *d_C_int32,     // [padded_m x N] Pre-allocated INT32 accumulator
                int M,                  // Actual M (output rows needed)
                int padded_m,           // Padded M for CK
                int N, int K,
                int rocm_device_id);

            // Get CK minimum dimension requirements
            int rocmQuantGemm_getMinM();
            int rocmQuantGemm_getMinN();
            int rocmQuantGemm_getMinK();

            // Free device memory
            void rocmQuantGemm_freeDevice(void *d_ptr);

            // Memory management helpers
            bool rocmQuantGemm_allocFloat(float **d_ptr, size_t count, int rocm_device_id);
            bool rocmQuantGemm_allocInt32(int32_t **d_ptr, size_t count, int rocm_device_id);
            bool rocmQuantGemm_copyHostToDevice(float *d_dst, const float *h_src, size_t count, int rocm_device_id);
            bool rocmQuantGemm_copyDeviceToHost(float *h_dst, const float *d_src, size_t count, int rocm_device_id);
            bool rocmQuantGemm_copyInt32DeviceToHost(int32_t *h_dst, const int32_t *d_src, size_t count, int rocm_device_id);
            bool rocmQuantGemm_setDevice(int rocm_device_id);
        }

        // =====================================================================
        // 3-Way Dispatch - Optimized kernel selection based on M
        // =====================================================================
        //
        // The .hip file implements 3-way dispatch using GemmMNPadding-enabled kernels:
        //
        //   ┌─────────────────────────────────────────────────────────────────┐
        //   │ M ≤ 32?      ───YES───> 32×32 kernel (handles M=1 natively!)   │
        //   │         │                                                       │
        //   │         NO                                                      │
        //   │         │                                                       │
        //   │ 32 < M < 128? ─YES───> 64×64 kernel (good for small batches)   │
        //   │         │                                                       │
        //   │         NO                                                      │
        //   │         │                                                       │
        //   │ M >= 128  ───────────> 128×128 kernel (peak throughput)        │
        //   └─────────────────────────────────────────────────────────────────┘
        //
        // Performance (FFN Up: K=3584, N=18944, MI50 gfx906):
        //   M=1:   0.46ms, ~2.4 TFLOPS  (32×32 kernel, 2.6x faster than 128-pad)
        //   M=32:  0.53ms, ~8.2 TFLOPS  (32×32 kernel)
        //   M=64:  0.51ms, ~17 TFLOPS   (64×64 kernel, 1.3x faster than 128-pad)
        //   M=128: 0.82ms, ~21 TFLOPS   (128×128 kernel)
        //
        // =====================================================================

        // Minimum M that the 32×32 kernel can handle without explicit padding
        // With GemmMNPadding, any M >= 1 works, but we keep 8 as the threshold
        // for explicit padding to match the old code path (for safety)
        constexpr int CK_MIN_M_FOR_EXPLICIT_PADDING = 8;
        constexpr int CK_MIN_N = 32; // All kernels: minimum N
        constexpr int CK_MIN_K = 32; // All kernels: minimum K

        // Returns the padded M value needed (only for M < 8)
        // The 32×32 kernel with GemmMNPadding handles most M values natively
        inline int getPaddedM(int M)
        {
            return (M < CK_MIN_M_FOR_EXPLICIT_PADDING) ? CK_MIN_M_FOR_EXPLICIT_PADDING : M;
        }

        // Check if M needs explicit padding (rare - only M < 8)
        inline bool needsMPadding(int M)
        {
            return M < CK_MIN_M_FOR_EXPLICIT_PADDING;
        }

        // =====================================================================
        // PIMPL implementation struct
        // =====================================================================

        struct ROCmQuantisedGemmKernel::Impl
        {
            // Device memory for converted weights (only used when owns_weight_memory_ = true)
            int8_t *d_weights_int8 = nullptr; // [K x N] ColumnMajor
            float *d_scales_B = nullptr;      // [N] per-column scales

            // Work buffers for activation quantization
            int8_t *d_A_int8 = nullptr;   // [M x K]
            float *d_scales_A = nullptr;  // [M]
            int32_t *d_C_int32 = nullptr; // [M x N]
            int work_buffer_M = 0;

            // Cached INT32 accumulator for CK two-kernel path (avoids hot-path allocations)
            int32_t *d_CK_int32 = nullptr;  // [M x N] accumulator for CK NoScale + applyScales
            size_t d_CK_int32_capacity = 0; // Current allocation size (elements)

            // Cached temporary buffers (to avoid hipMalloc/hipFree per call)
            float *d_A_fp32 = nullptr;    // [M x K] input activations FP32
            float *d_C_fp32 = nullptr;    // [M x N] output FP32
            size_t d_A_fp32_capacity = 0; // Current allocation size (elements)
            size_t d_C_fp32_capacity = 0; // Current allocation size (elements)

            // Flag to track if we own weight memory
            bool owns_weight_memory = false;

            ~Impl()
            {
                // Only free weight memory if we own it (not from ROCmPackedWeights cache)
                if (owns_weight_memory)
                {
                    if (d_weights_int8)
                        rocmQuantGemm_freeDevice(d_weights_int8);
                    if (d_scales_B)
                        rocmQuantGemm_freeDevice(d_scales_B);
                }
                // Always free work buffers (we always own these)
                if (d_A_int8)
                    rocmQuantGemm_freeDevice(d_A_int8);
                if (d_scales_A)
                    rocmQuantGemm_freeDevice(d_scales_A);
                if (d_C_int32)
                    rocmQuantGemm_freeDevice(d_C_int32);
                // Free cached CK INT32 buffer
                if (d_CK_int32)
                    rocmQuantGemm_freeDevice(d_CK_int32);
                // Free cached temporary buffers
                if (d_A_fp32)
                    rocmQuantGemm_freeDevice(d_A_fp32);
                if (d_C_fp32)
                    rocmQuantGemm_freeDevice(d_C_fp32);
            }
        };

        // =====================================================================
        // ROCmPackedWeights destructor
        // =====================================================================

        ROCmPackedWeights::~ROCmPackedWeights()
        {
            if (d_int8_data)
                rocmQuantGemm_freeDevice(d_int8_data);
            if (d_scales)
                rocmQuantGemm_freeDevice(d_scales);
        }

        // =====================================================================
        // packWeightsToROCm: Convert any quantized tensor to INT8 + scales
        // =====================================================================
        //
        // This function quantizes weights for CK GEMM using mk_nk_mn layout.
        //
        // ## Input Layout (Model Weights)
        //
        // Model weights are stored as [N × K] row-major:
        //   - N = output_features (rows)
        //   - K = input_features (columns)
        //   - Element W[n,k] at offset: n * K + k
        //
        // ## Output Layout (CK GEMM Weights - mk_nk_mn convention)
        //
        // CK's mk_nk_mn layout expects B as [N × K] stored as Column-Major [K × N]:
        //   - When viewed as B[k,n], element at offset: k + n * K
        //   - This is EXACTLY the same memory layout as [N × K] Row-Major!
        //   - W[n,k] at n*K + k == B[k,n] at k + n*K (same offset!)
        //
        // ## Key Insight: NO TRANSPOSE NEEDED!
        //
        // Model weights [N×K] Row-Major can be directly reinterpreted as
        // Column-Major [K×N] for CK's mk_nk_mn layout. We just quantize in place.
        //
        // ## Quantization
        //
        // Per-output-feature (per-row of model weights) symmetric quantization:
        //   scale[n] = max(|W[n,:]|) / 127.0
        //   int8[n,k] = round(W[n,k] / scale[n])
        //
        // This allows efficient output scaling: output = int32_result * scale_A * scale_B
        //
        // =====================================================================

        bool packWeightsToROCm(const TensorBase *tensor, ROCmPackedWeights &out)
        {
            if (!tensor)
            {
                LOG_ERROR("[packWeightsToROCm] Null tensor");
                return false;
            }

            const int N = static_cast<int>(tensor->rows()); // Output features (model weight rows)
            const int K = static_cast<int>(tensor->cols()); // Input features (model weight cols)

            // Get dequantized FP32 data - use fp32_data() which explicitly dequantizes
            const float *h_weights_fp32 = tensor->fp32_data();
            if (!h_weights_fp32)
            {
                LOG_ERROR("[packWeightsToROCm] Failed to get FP32 data from tensor");
                return false;
            }

            // Allocate output vectors:
            //   int8_data: [N × K] row-major (same layout as model weights!)
            //             This is Column-Major [K × N] for CK's mk_nk_mn convention
            //   scales: [N] per-output-feature scales
            out.int8_data.resize(static_cast<size_t>(N) * K);
            out.scales.resize(N);
            out.K = K;
            out.N = N;

            // Per-row (per-output-feature) symmetric quantization
            // NO transpose - keep same layout as model weights
            //
            // Input (h_weights_fp32):  [N × K] row-major, W[n,k] at n*K + k
            // Output (out.int8_data):  [N × K] row-major, same layout!
            //                          (viewed as Column-Major [K×N] for CK)
            for (int n = 0; n < N; ++n)
            {
                // Find max_abs for this output feature (row n of model weights)
                float max_abs = 0.0f;
                for (int k = 0; k < K; ++k)
                {
                    // W[n,k] = h_weights_fp32[n * K + k]
                    float val = h_weights_fp32[n * K + k];
                    max_abs = std::max(max_abs, std::abs(val));
                }

                // Symmetric quantization: scale = max_abs / 127
                float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
                float inv_scale = 1.0f / scale;
                out.scales[n] = scale; // One scale per output feature

                // Quantize in place - NO transpose!
                // Both source and destination use offset n*K + k
                for (int k = 0; k < K; ++k)
                {
                    float val = h_weights_fp32[n * K + k]; // Read W[n,k]
                    int8_t quantized = static_cast<int8_t>(
                        std::round(std::clamp(val * inv_scale, -127.0f, 127.0f)));
                    out.int8_data[n * K + k] = quantized; // Write at same offset!
                }
            }

            LOG_DEBUG("[packWeightsToROCm] Packed " << N << "x" << K << " weights to INT8 (mk_nk_mn layout)");
            return true;
        }

        // =====================================================================
        // Constructor / Destructor
        // =====================================================================

        ROCmQuantisedGemmKernel::ROCmQuantisedGemmKernel(const TensorBase *weights, int rocm_device_id)
            : weights_(weights),
              packed_(nullptr),
              rocm_device_id_(rocm_device_id),
              N_(0),
              K_(0),
              weights_converted_(false),
              owns_weight_memory_(true), // Legacy path owns weight memory
              impl_(std::make_unique<Impl>())
        {
            if (!weights)
            {
                throw std::runtime_error("[ROCmQuantisedGemmKernel] Null weight tensor");
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
                    "[ROCmQuantisedGemmKernel] Weight tensor must be quantized type, got: " +
                    std::to_string(static_cast<int>(wt)));
            }

            impl_->owns_weight_memory = true; // Legacy constructor owns weight memory

            LOG_DEBUG("[ROCmQuantisedGemmKernel] Created (legacy) for " << N_ << "x" << K_
                                                                        << " quantized weights (type=" << static_cast<int>(wt)
                                                                        << ") on ROCm device " << rocm_device_id_);
        }

        ROCmQuantisedGemmKernel::ROCmQuantisedGemmKernel(ROCmPackedWeights *packed, int rocm_device_id)
            : weights_(nullptr),
              packed_(packed),
              rocm_device_id_(rocm_device_id),
              N_(0),
              K_(0),
              weights_converted_(false),  // Not yet uploaded to device
              owns_weight_memory_(false), // ROCmPackedWeights owns the memory
              impl_(std::make_unique<Impl>())
        {
            if (!packed)
            {
                throw std::runtime_error("[ROCmQuantisedGemmKernel] Null packed weights");
            }

            N_ = static_cast<size_t>(packed->N);
            K_ = static_cast<size_t>(packed->K);

            impl_->owns_weight_memory = false; // Pre-packed path doesn't own weight memory

            LOG_DEBUG("[ROCmQuantisedGemmKernel] Created (pre-packed) for " << N_ << "x" << K_
                                                                            << " INT8 weights on ROCm device " << rocm_device_id_);
        }

        ROCmQuantisedGemmKernel::~ROCmQuantisedGemmKernel() = default;

        ROCmQuantisedGemmKernel::ROCmQuantisedGemmKernel(ROCmQuantisedGemmKernel &&) noexcept = default;
        ROCmQuantisedGemmKernel &ROCmQuantisedGemmKernel::operator=(ROCmQuantisedGemmKernel &&) noexcept = default;

        // =====================================================================
        // ITensorGemm interface - Implementation
        // =====================================================================

        bool ROCmQuantisedGemmKernel::multiply_tensor(
            const TensorBase *A, TensorBase *C,
            bool transpose_B,
            float alpha, float beta,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            (void)mpi_ctx;
            (void)device_idx;
            (void)transpose_B; // Weights are always transposed for this kernel

            if (!A || !C)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Null tensor");
                return false;
            }

            // Get dimensions
            int m = static_cast<int>(A->rows());
            int k = static_cast<int>(A->cols());
            int n = static_cast<int>(N_);

            if (static_cast<size_t>(k) != K_)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] K mismatch: A.cols=" << k << " vs weight K=" << K_);
                return false;
            }

            return multiply_tensor(A, C, m, n, k, transpose_B, alpha, beta, mpi_ctx, device_idx);
        }

        bool ROCmQuantisedGemmKernel::multiply_tensor(
            const TensorBase *A, TensorBase *C,
            int m, int n, int k,
            bool transpose_B,
            float alpha, float beta,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            (void)mpi_ctx;
            (void)device_idx;
            (void)transpose_B;

            if (!A || !C)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Null tensor");
                return false;
            }

            if (m <= 0 || n <= 0 || k <= 0)
            {
                LOG_WARN("[ROCmQuantisedGemmKernel::multiply_tensor] Zero dimensions");
                return true;
            }

            // Cast to FP32Tensor (only supported type for now)
            auto *A_fp32 = dynamic_cast<const FP32Tensor *>(A);
            auto *C_fp32 = dynamic_cast<FP32Tensor *>(C);

            if (!A_fp32)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] A must be FP32Tensor (Q8_1 not yet supported)");
                return false;
            }
            if (!C_fp32)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] C must be FP32Tensor");
                return false;
            }

            // Ensure weights are uploaded to device
            ensureWeightsConverted();

            // Get weight device pointers
            int8_t *d_weights_int8 = nullptr;
            float *d_scales_B = nullptr;

            if (packed_)
            {
                d_weights_int8 = packed_->d_int8_data;
                d_scales_B = packed_->d_scales;
            }
            else if (impl_)
            {
                d_weights_int8 = impl_->d_weights_int8;
                d_scales_B = impl_->d_scales_B;
            }

            if (!d_weights_int8 || !d_scales_B)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Weights not uploaded to device");
                return false;
            }

            LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_tensor] Weight ptrs: int8=" << (void *)d_weights_int8 << " scales=" << (void *)d_scales_B);

            // Ensure work buffers are allocated
            ensureWorkBuffers(m);

            int8_t *d_A_int8 = impl_->d_A_int8;
            float *d_scales_A = impl_->d_scales_A;
            int32_t *d_C_int32 = impl_->d_C_int32;

            LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_tensor] Work buffers: A_int8=" << (void *)d_A_int8
                                                                                         << " scales_A=" << (void *)d_scales_A << " C_int32=" << (void *)d_C_int32);

            // Ensure cached d_A_fp32 buffer is large enough (reuse across calls)
            const size_t a_fp32_size = static_cast<size_t>(m) * k;
            if (a_fp32_size > impl_->d_A_fp32_capacity)
            {
                if (impl_->d_A_fp32)
                    rocmQuantGemm_freeDevice(impl_->d_A_fp32);
                impl_->d_A_fp32 = nullptr;
                impl_->d_A_fp32_capacity = 0;

                if (!rocmQuantGemm_allocFloat(&impl_->d_A_fp32, a_fp32_size, rocm_device_id_))
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Failed to allocate activation buffer");
                    return false;
                }
                impl_->d_A_fp32_capacity = a_fp32_size;
            }
            float *d_A_fp32 = impl_->d_A_fp32;

            LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_tensor] Using cached d_A_fp32=" << (void *)d_A_fp32);

            if (!rocmQuantGemm_copyHostToDevice(d_A_fp32, A_fp32->data(), a_fp32_size, rocm_device_id_))
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Failed to copy activations to device");
                return false;
            }

            LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_tensor] Copied activations to device, now quantizing");

            // Quantize activations FP32 → INT8
            if (!rocmQuantGemm_quantizeActivations(d_A_fp32, d_A_int8, d_scales_A, m, k, rocm_device_id_))
            {
                rocmQuantGemm_freeDevice(d_A_fp32);
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Failed to quantize activations");
                return false;
            }

            LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_tensor] Quantized activations, now executing CK GEMM");

#if 0 // Debug dump code disabled - was causing heap corruption with non-standard sizes
      // DEBUG: Dump INT8 inputs before GEMM
      // Note: We need to pad buffer sizes to 4-byte boundaries for the copy function
            {
                size_t a_bytes = static_cast<size_t>(m) * k;
                size_t b_bytes = static_cast<size_t>(k) * n;
                std::vector<int8_t> h_A_int8((a_bytes + 3) & ~3);  // Round up to multiple of 4
                std::vector<int8_t> h_B_int8((b_bytes + 3) & ~3);  // Round up to multiple of 4
                rocmQuantGemm_copyDeviceToHost(reinterpret_cast<float*>(h_A_int8.data()), 
                                               reinterpret_cast<float*>(d_A_int8), 
                                               (a_bytes + 3) / 4, rocm_device_id_);
                rocmQuantGemm_copyDeviceToHost(reinterpret_cast<float*>(h_B_int8.data()), 
                                               reinterpret_cast<float*>(d_weights_int8), 
                                               (b_bytes + 3) / 4, rocm_device_id_);
                
                // Only print if we have at least 8 elements
                if (k >= 8) {
                    LOG_INFO("[DEBUG] A_int8 row0 first 8: " 
                             << (int)h_A_int8[0] << ", " << (int)h_A_int8[1] << ", " 
                             << (int)h_A_int8[2] << ", " << (int)h_A_int8[3] << ", "
                             << (int)h_A_int8[4] << ", " << (int)h_A_int8[5] << ", "
                             << (int)h_A_int8[6] << ", " << (int)h_A_int8[7]);
                }
                if (n >= 8) {
                    // B is now [K×N] RowMajor: B[k_idx, n_idx] = h_B_int8[k_idx * n + n_idx]
                    LOG_INFO("[DEBUG] B_int8 [K×N] RowMajor row0 (n=0..7): " 
                             << (int)h_B_int8[0*n + 0] << ", " << (int)h_B_int8[0*n + 1] << ", " 
                             << (int)h_B_int8[0*n + 2] << ", " << (int)h_B_int8[0*n + 3] << ", "
                             << (int)h_B_int8[0*n + 4] << ", " << (int)h_B_int8[0*n + 5] << ", "
                             << (int)h_B_int8[0*n + 6] << ", " << (int)h_B_int8[0*n + 7]);
                }
                if (k >= 2 && n >= 8) {
                    LOG_INFO("[DEBUG] B_int8 [K×N] RowMajor row1 (n=0..7): " 
                             << (int)h_B_int8[1*n + 0] << ", " << (int)h_B_int8[1*n + 1] << ", " 
                             << (int)h_B_int8[1*n + 2] << ", " << (int)h_B_int8[1*n + 3] << ", "
                             << (int)h_B_int8[1*n + 4] << ", " << (int)h_B_int8[1*n + 5] << ", "
                             << (int)h_B_int8[1*n + 6] << ", " << (int)h_B_int8[1*n + 7]);
                }
                if (k >= 8) {
                    // Show column 0 of B (the elements that will dot-product with A row 0)
                    LOG_INFO("[DEBUG] B_int8 col0 (B[k,0] for k=0..7): " 
                             << (int)h_B_int8[0*n + 0] << ", " << (int)h_B_int8[1*n + 0] << ", " 
                             << (int)h_B_int8[2*n + 0] << ", " << (int)h_B_int8[3*n + 0] << ", "
                             << (int)h_B_int8[4*n + 0] << ", " << (int)h_B_int8[5*n + 0] << ", "
                             << (int)h_B_int8[6*n + 0] << ", " << (int)h_B_int8[7*n + 0]);
                }

                // Manual INT32 dot product for C[0,0] = A[0,:] · B[:,0]
                // A is [M×K] row-major: A[m,k] = h_A_int8[m*K + k]
                // B is [K×N] row-major: B[k,n] = h_B_int8[k*N + n]
                // C[0,0] = sum_k { A[0,k] * B[k,0] }
                int32_t manual_dot = 0;
                for (int kk = 0; kk < k; ++kk) {
                    int8_t a_val = h_A_int8[0 * k + kk];  // A[0,kk] row-major
                    int8_t b_val = h_B_int8[kk * n + 0];  // B[kk,0] row-major [K×N]
                    manual_dot += static_cast<int32_t>(a_val) * static_cast<int32_t>(b_val);
                }
                LOG_INFO("[DEBUG] Manual dot(A_row0, B_col0) for [K×N] layout = " << manual_dot);
            }
#endif

            // Ensure cached d_C_fp32 buffer is large enough (reuse across calls)
            const size_t c_fp32_size = static_cast<size_t>(m) * n;
            if (c_fp32_size > impl_->d_C_fp32_capacity)
            {
                if (impl_->d_C_fp32)
                    rocmQuantGemm_freeDevice(impl_->d_C_fp32);
                impl_->d_C_fp32 = nullptr;
                impl_->d_C_fp32_capacity = 0;

                if (!rocmQuantGemm_allocFloat(&impl_->d_C_fp32, c_fp32_size, rocm_device_id_))
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Failed to allocate output buffer");
                    return false;
                }
                impl_->d_C_fp32_capacity = c_fp32_size;
            }
            float *d_C_fp32 = impl_->d_C_fp32;

            // =========================================================================
            // CK TWO-KERNEL DISPATCH (with M-padding for decode)
            // =========================================================================
            //
            // CK INT8 Two-Kernel is the ONLY path. For small M (decode), we pad
            // activations to CK_MIN_M (128), run CK, then extract first M rows.
            //
            // NOTE: hipBLAS INT8 on gfx906 has N <= K limitation, breaking FFN.
            //       M-padding for CK is more efficient and universally supported.
            //
            const int padded_m = getPaddedM(m);
            const bool needs_padding = needsMPadding(m);
            bool success = false;

            if (needs_padding)
            {
                LOG_DEBUG("[ROCmQuantisedGemmKernel] Using CK TWO-KERNEL with M-padding (M=" << m << " -> " << padded_m << ", N=" << n << ", K=" << k << ")");
            }
            else
            {
                LOG_DEBUG("[ROCmQuantisedGemmKernel] Using CK TWO-KERNEL (M=" << m << ", N=" << n << ", K=" << k << ")");
            }

            // Ensure cached INT32 accumulator buffer is large enough (use padded size)
            const size_t ck_int32_size = static_cast<size_t>(padded_m) * n;
            if (ck_int32_size > impl_->d_CK_int32_capacity)
            {
                if (impl_->d_CK_int32)
                    rocmQuantGemm_freeDevice(impl_->d_CK_int32);
                impl_->d_CK_int32 = nullptr;
                impl_->d_CK_int32_capacity = 0;

                if (!rocmQuantGemm_allocInt32(&impl_->d_CK_int32, ck_int32_size, rocm_device_id_))
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to allocate CK INT32 buffer");
                    return false;
                }
                impl_->d_CK_int32_capacity = ck_int32_size;
            }

            if (needs_padding)
            {
                // Padded execution: allocate padded buffers, run CK, extract first M rows
                success = rocmQuantGemm_executeTwoKernel_padded(
                    d_A_int8, d_weights_int8, d_C_fp32,
                    d_scales_A, d_scales_B,
                    impl_->d_CK_int32,
                    m, padded_m, n, k, rocm_device_id_);
            }
            else
            {
                // Direct execution: no padding needed
                success = rocmQuantGemm_executeTwoKernel_cached(
                    d_A_int8, d_weights_int8, d_C_fp32,
                    d_scales_A, d_scales_B,
                    impl_->d_CK_int32,
                    m, n, k, rocm_device_id_);
            }

            if (!success)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] All GEMM paths failed");
                return false;
            }

            // Copy result back to host
            if (!rocmQuantGemm_copyDeviceToHost(C_fp32->mutable_data(), d_C_fp32, c_fp32_size, rocm_device_id_))
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Failed to copy output to host");
                return false;
            }

            LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_tensor] Completed " << m << "x" << n << "x" << k);
            return true;
        }

        bool ROCmQuantisedGemmKernel::multiply_tensor_timed(
            const TensorBase *A, TensorBase *C,
            int m, int n, int k,
            float *kernel_time_ms)
        {
            if (kernel_time_ms)
                *kernel_time_ms = 0.0f;

            if (!A || !C)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor_timed] Null tensor");
                return false;
            }

            auto *A_fp32 = dynamic_cast<const FP32Tensor *>(A);
            auto *C_fp32 = dynamic_cast<FP32Tensor *>(C);
            if (!A_fp32)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor_timed] A must be FP32Tensor");
                return false;
            }
            if (!C_fp32)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor_timed] C must be FP32Tensor");
                return false;
            }

            // Get weight device pointers
            int8_t *d_weights_int8 = impl_->d_weights_int8;
            float *d_scales_B = impl_->d_scales_B;
            if (!d_weights_int8 || !d_scales_B)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor_timed] Weights not uploaded");
                return false;
            }

            // Ensure work buffers
            ensureWorkBuffers(m);

            int8_t *d_A_int8 = impl_->d_A_int8;
            float *d_scales_A = impl_->d_scales_A;
            int32_t *d_C_int32 = impl_->d_C_int32;

            // Ensure FP32 activation buffer
            const size_t a_fp32_size = static_cast<size_t>(m) * k;
            if (a_fp32_size > impl_->d_A_fp32_capacity)
            {
                if (impl_->d_A_fp32)
                    rocmQuantGemm_freeDevice(impl_->d_A_fp32);
                if (!rocmQuantGemm_allocFloat(&impl_->d_A_fp32, a_fp32_size, rocm_device_id_))
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor_timed] Failed to allocate A buffer");
                    return false;
                }
                impl_->d_A_fp32_capacity = a_fp32_size;
            }
            float *d_A_fp32 = impl_->d_A_fp32;

            // Copy activations H2D and quantize (OUTSIDE timing)
            if (!rocmQuantGemm_copyHostToDevice(d_A_fp32, A_fp32->data(), a_fp32_size, rocm_device_id_))
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor_timed] Failed to copy A");
                return false;
            }
            if (!rocmQuantGemm_quantizeActivations(d_A_fp32, d_A_int8, d_scales_A, m, k, rocm_device_id_))
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor_timed] Failed to quantize A");
                return false;
            }

            // Ensure output buffer
            const size_t c_fp32_size = static_cast<size_t>(m) * n;
            if (c_fp32_size > impl_->d_C_fp32_capacity)
            {
                if (impl_->d_C_fp32)
                    rocmQuantGemm_freeDevice(impl_->d_C_fp32);
                if (!rocmQuantGemm_allocFloat(&impl_->d_C_fp32, c_fp32_size, rocm_device_id_))
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor_timed] Failed to allocate C buffer");
                    return false;
                }
                impl_->d_C_fp32_capacity = c_fp32_size;
            }
            float *d_C_fp32 = impl_->d_C_fp32;

            // Calculate padding
            int padded_m = getPaddedM(m);
            bool needs_padding = (padded_m > m);

            // Ensure INT32 accumulator buffer
            const size_t ck_int32_size = static_cast<size_t>(padded_m) * n;
            if (ck_int32_size > impl_->d_CK_int32_capacity)
            {
                if (impl_->d_CK_int32)
                    rocmQuantGemm_freeDevice(impl_->d_CK_int32);
                impl_->d_CK_int32 = nullptr;
                if (!rocmQuantGemm_allocInt32(&impl_->d_CK_int32, ck_int32_size, rocm_device_id_))
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor_timed] Failed to allocate INT32 buffer");
                    return false;
                }
                impl_->d_CK_int32_capacity = ck_int32_size;
            }

            // Execute GEMM with HIP event timing (ONLY this is timed)
            bool success = false;
            if (needs_padding)
            {
                // Padded path doesn't have timed version yet, fall back to cached
                success = rocmQuantGemm_executeTwoKernel_padded(
                    d_A_int8, d_weights_int8, d_C_fp32,
                    d_scales_A, d_scales_B,
                    impl_->d_CK_int32,
                    m, padded_m, n, k, rocm_device_id_);
                // Can't get accurate timing for padded path without modifying it
                if (kernel_time_ms)
                    *kernel_time_ms = -1.0f; // Indicate timing not available
            }
            else
            {
                // Direct execution with timing
                success = rocmQuantGemm_executeTwoKernel_timed(
                    d_A_int8, d_weights_int8, d_C_fp32,
                    d_scales_A, d_scales_B,
                    impl_->d_CK_int32,
                    m, n, k, rocm_device_id_,
                    kernel_time_ms);
            }

            if (!success)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor_timed] GEMM failed");
                return false;
            }

            // Copy result D2H (OUTSIDE timing)
            if (!rocmQuantGemm_copyDeviceToHost(C_fp32->mutable_data(), d_C_fp32, c_fp32_size, rocm_device_id_))
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor_timed] Failed to copy C");
                return false;
            }

            return true;
        }

        bool ROCmQuantisedGemmKernel::multiply(
            const float *A, float *C,
            int m, int n, int k,
            bool transpose_B,
            float alpha, float beta,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            // TODO: Implement in Phase 5
            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply] Not yet implemented");
            return false;
        }

        bool ROCmQuantisedGemmKernel::multiply_fused(
            const float *input,
            const std::vector<FusedProjectionDesc> &projections,
            int m, int k,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            // TODO: Implement in Phase 5
            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused] Not yet implemented");
            return false;
        }

        bool ROCmQuantisedGemmKernel::multiply_fused_tensor(
            const TensorBase *input,
            const std::vector<TensorProjectionDesc> &projections,
            int m, int k,
            const MPIContext *mpi_ctx)
        {
            // TODO: Implement in Phase 5
            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Not yet implemented");
            return false;
        }

        bool ROCmQuantisedGemmKernel::multiply_activations(
            const float *A, const float *B, float *C,
            int m, int n, int k,
            bool transpose_B,
            float alpha, float beta,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            // Activation-activation GEMM is not supported by quantized kernel
            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_activations] Not supported - use dedicated attention kernel");
            return false;
        }

        bool ROCmQuantisedGemmKernel::multiply_activations_strided(
            const float *A, const float *B, float *C,
            int m, int n, int k,
            int lda, int ldb, int ldc,
            bool transpose_B,
            float alpha, float beta,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            // Strided activation GEMM is not supported
            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_activations_strided] Not supported");
            return false;
        }

        bool ROCmQuantisedGemmKernel::supports_device(int device_idx) const
        {
            // Supports any ROCm GPU device
            return device_idx >= 0;
        }

        KernelSnapshotInfo ROCmQuantisedGemmKernel::getKernelSnapshotInfo() const
        {
            return KernelSnapshotInfo::gemm()
                .withInput("A", "input activations [m, k]", KernelBufferDtype::FP32)
                .withWeight("B", "quantized weight matrix [n, k] (converted to INT8)", KernelBufferDtype::INT8)
                .withOutput("C", "output matrix [m, n]", KernelBufferDtype::FP32)
                .withScalar("N", "output features", KernelBufferDtype::INT32)
                .withScalar("K", "input features", KernelBufferDtype::INT32)
                .withScalar("rocm_device_id", "ROCm device ID", KernelBufferDtype::INT32)
                .withScalar("weights_converted", "whether weights are converted to INT8", KernelBufferDtype::INT32);
        }

        // =====================================================================
        // IWorkspaceConsumer Interface Implementation
        // =====================================================================

        WorkspaceRequirements ROCmQuantisedGemmKernel::getWorkspaceRequirements(
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

            // Also need FP32 temp buffers for host→device transfer
            size_t temp_a_fp32_bytes = static_cast<size_t>(m) * k * sizeof(float);
            size_t temp_c_fp32_bytes = static_cast<size_t>(m) * n * sizeof(float);

            reqs.buffers.push_back({GemmWorkspaceBuffers::QUANT_A, quant_a_bytes, 256, true});
            reqs.buffers.push_back({GemmWorkspaceBuffers::SCALES_A, scales_a_bytes, 256, true});
            reqs.buffers.push_back({GemmWorkspaceBuffers::ACC_INT32, acc_int32_bytes, 256, true});
            reqs.buffers.push_back({GemmWorkspaceBuffers::TEMP_A_FP32, temp_a_fp32_bytes, 256, true});
            reqs.buffers.push_back({GemmWorkspaceBuffers::TEMP_C_FP32, temp_c_fp32_bytes, 256, true});

            LOG_DEBUG("[ROCmQuantisedGemmKernel::getWorkspaceRequirements] INT8 path: "
                      << "quant_a=" << (quant_a_bytes / 1024) << "KB, "
                      << "scales_a=" << (scales_a_bytes) << "B, "
                      << "acc=" << (acc_int32_bytes / 1024) << "KB");

            return reqs;
        }

        void ROCmQuantisedGemmKernel::bindWorkspace(DeviceWorkspaceManager *workspace)
        {
            workspace_ = workspace;
            if (workspace)
            {
                LOG_DEBUG("[ROCmQuantisedGemmKernel] Bound workspace manager at " << (void *)workspace
                                                                                  << ", entering managed mode");
            }
            else
            {
                LOG_DEBUG("[ROCmQuantisedGemmKernel] Unbound workspace, returning to legacy mode");
            }
        }

        bool ROCmQuantisedGemmKernel::hasWorkspace() const
        {
            return workspace_ != nullptr;
        }

        DeviceWorkspaceManager *ROCmQuantisedGemmKernel::getWorkspace() const
        {
            return workspace_;
        }

        // =====================================================================
        // Internal methods - STUBS
        // =====================================================================

        void ROCmQuantisedGemmKernel::ensureWeightsConverted()
        {
            if (weights_converted_)
            {
                return; // Already converted and uploaded
            }

            // Path 1: Pre-packed weights (ROCmPackedWeights* passed to constructor)
            if (packed_)
            {
                if (!packed_->uploaded)
                {
                    // Upload to device
                    bool ok = rocmQuantGemm_uploadWeights(
                        packed_->int8_data.data(),
                        packed_->scales.data(),
                        &packed_->d_int8_data,
                        &packed_->d_scales,
                        packed_->K,
                        packed_->N,
                        rocm_device_id_);

                    if (!ok)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to upload pre-packed weights");
                        return;
                    }

                    packed_->uploaded = true;
                    packed_->rocm_device_id = rocm_device_id_;
                    LOG_DEBUG("[ROCmQuantisedGemmKernel] Uploaded pre-packed weights: "
                              << packed_->N << "x" << packed_->K);
                }

                // Point impl_ to packed_ device pointers
                impl_->d_weights_int8 = packed_->d_int8_data;
                impl_->d_scales_B = packed_->d_scales;
                weights_converted_ = true;
                return;
            }

            // Path 2: Legacy path - convert from TensorBase*
            if (!weights_)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel] No weights tensor or packed weights!");
                return;
            }

            // Pack on host
            ROCmPackedWeights host_packed;
            if (!packWeightsToROCm(weights_, host_packed))
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to pack weights");
                return;
            }

            // Upload to device
            bool ok = rocmQuantGemm_uploadWeights(
                host_packed.int8_data.data(),
                host_packed.scales.data(),
                &impl_->d_weights_int8,
                &impl_->d_scales_B,
                host_packed.K,
                host_packed.N,
                rocm_device_id_);

            if (!ok)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to upload weights to device");
                return;
            }

            impl_->owns_weight_memory = true; // We now own the device memory
            weights_converted_ = true;

            LOG_DEBUG("[ROCmQuantisedGemmKernel] Converted and uploaded weights: "
                      << N_ << "x" << K_);
        }

        void ROCmQuantisedGemmKernel::ensureWorkBuffers(int m)
        {
            // Check if we already have enough capacity
            if (impl_->work_buffer_M >= m && impl_->d_A_int8 && impl_->d_scales_A && impl_->d_C_int32)
            {
                return;
            }

            // Call the HIP function to allocate/resize buffers
            int k = static_cast<int>(K_);
            int n = static_cast<int>(N_);

            bool ok = rocmQuantGemm_ensureWorkBuffers(
                &impl_->d_A_int8,
                &impl_->d_scales_A,
                &impl_->d_C_int32,
                &impl_->work_buffer_M,
                m, k, n,
                rocm_device_id_);

            if (!ok)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::ensureWorkBuffers] Failed to allocate work buffers");
            }
        }

        bool ROCmQuantisedGemmKernel::multiply_q8_to_fp32(
            const Q8_1Block *d_A_q8, float *d_C,
            int m, int n, int k,
            float alpha, float beta)
        {
            // TODO: Implement in Phase 5
            return false;
        }

        bool ROCmQuantisedGemmKernel::multiply_q8_to_q8(
            const Q8_1Block *d_A_q8, Q8_1Block *d_C_q8,
            int m, int n, int k)
        {
            // TODO: Implement in Phase 5
            return false;
        }

        bool ROCmQuantisedGemmKernel::multiply_fp32_to_fp32(
            const float *d_A, float *d_C,
            int m, int n, int k,
            float alpha, float beta)
        {
            // TODO: Implement in Phase 5
            return false;
        }

        bool ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias(
            const float *d_A, float *d_C, const float *d_bias,
            int m, int n, int k,
            float alpha, float beta)
        {
            // TODO: Implement in Phase 5
            return false;
        }

        bool ROCmQuantisedGemmKernel::multiply_fp32_to_q8(
            const float *d_A, Q8_1Block *d_C_q8,
            int m, int n, int k)
        {
            // TODO: Implement in Phase 5
            return false;
        }

    } // namespace rocm
} // namespace llaminar2
