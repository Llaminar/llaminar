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
#include <chrono>

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

            // Execute INT8 GEMM without scaling (INT8→INT32 only)
            // Used when you want to apply custom scaling/bias separately.
            bool rocmQuantGemm_executeNoScale(
                const int8_t *d_A_int8, // [M x K] RowMajor quantized activations
                const int8_t *d_B_int8, // [K x N] RowMajor weights
                int32_t *d_C_int32,     // [M x N] RowMajor INT32 output
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

            // Execute INT8 GEMM using CK two-kernel with M-PADDING and PRE-ALLOCATED buffers (allocation-free)
            // This is the preferred variant for hot-path decode execution.
            bool rocmQuantGemm_executeTwoKernel_padded_cached(
                const int8_t *d_A_int8,  // [M x K] RowMajor quantized activations
                const int8_t *d_B_int8,  // [N x K] RowMajor weights (viewed as [K x N] col-major)
                float *d_E_fp32,         // [M x N] RowMajor FP32 output (only first M rows written)
                const float *d_scale_A,  // [M] per-row activation scales
                const float *d_scale_B,  // [N] per-column weight scales
                int32_t *d_C_int32,      // [padded_m x N] Pre-allocated INT32 accumulator
                int8_t *d_A_padded,      // [padded_m x K] Pre-allocated padded activations
                float *d_scale_A_padded, // [padded_m] Pre-allocated padded scales
                float *d_E_padded,       // [padded_m x N] Pre-allocated padded output
                int M,                   // Actual M (output rows needed)
                int padded_m,            // Padded M for CK
                int N, int K,
                int rocm_device_id);

            // Allocate INT8 buffer
            bool rocmQuantGemm_allocInt8(int8_t **d_ptr, size_t count, int rocm_device_id);

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

            // Apply scaling with full epilogue (alpha, beta, bias) - matches CUDA's cudaQuantGemm_applyScaling
            bool rocmQuantGemm_applyScaling(
                const int32_t *d_C_int32, // [M×N] INT32 GEMM output
                float *d_C_fp32,          // [M×N] FP32 output
                const float *d_scales_A,  // [M] per-row scales
                const float *d_scales_B,  // [N] per-column scales
                int M, int N,
                float alpha, float beta,
                const float *d_C_existing, // For beta != 0 (nullable)
                const float *d_bias,       // [N] optional bias (nullable)
                int rocm_device_id);

            // In-place bias addition: output[m,n] += bias[n]
            bool rocmQuantGemm_biasAdd(
                float *d_output,     // [M × N] FP32 output (modified in-place)
                const float *d_bias, // [N] bias vector
                int M, int N,
                int rocm_device_id);
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

            // Work buffer pointers - obtained from workspace at execution time
            // These are NOT owned by the kernel - they point into workspace-managed memory
            int8_t *d_A_int8 = nullptr;   // [M x K] quantized activations
            float *d_scales_A = nullptr;  // [M] per-row scales
            int32_t *d_C_int32 = nullptr; // [M x N] INT32 accumulator

            // CK-specific work buffers - also from workspace
            int32_t *d_CK_int32 = nullptr;     // [M x N] CK accumulator
            float *d_A_fp32 = nullptr;         // [M x K] input FP32
            float *d_C_fp32 = nullptr;         // [M x N] output FP32
            int8_t *d_A_padded = nullptr;      // [padded_m x K] padded activations
            float *d_scale_A_padded = nullptr; // [padded_m] padded scales
            float *d_E_padded = nullptr;       // [padded_m x N] padded output

            // Capacity tracking for workspace buffers (set during validateWorkspace)
            size_t d_CK_int32_capacity = 0;
            size_t d_A_fp32_capacity = 0;
            size_t d_C_fp32_capacity = 0;
            size_t d_A_padded_capacity = 0;
            size_t d_scale_A_padded_capacity = 0;
            size_t d_E_padded_capacity = 0;

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
                // Work buffers are NOT freed - they are owned by workspace
            }
        };

        // =====================================================================
        // ROCmPackedWeights destructor
        // =====================================================================

        ROCmPackedWeights::~ROCmPackedWeights()
        {
            if (d_int8_data || d_scales)
            {
                LOG_DEBUG("[ROCmPackedWeights::~] Freeing device memory: "
                          << "d_int8_data=" << (void *)d_int8_data
                          << " d_scales=" << (void *)d_scales
                          << " rocm_device_id=" << rocm_device_id);
            }
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
            const TensorBase *bias,
            const MPIContext *mpi_ctx,
            int device_idx,
            DeviceWorkspaceManager *workspace)
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

            return multiply_tensor(A, C, m, n, k, transpose_B, alpha, beta, bias, mpi_ctx, device_idx, workspace);
        }

        bool ROCmQuantisedGemmKernel::multiply_tensor(
            const TensorBase *A, TensorBase *C,
            int m, int n, int k,
            bool transpose_B,
            float alpha, float beta,
            const TensorBase *bias,
            const MPIContext *mpi_ctx,
            int device_idx,
            DeviceWorkspaceManager *workspace)
        {
            (void)mpi_ctx;
            (void)device_idx;
            (void)transpose_B;

            // Use passed workspace if provided, otherwise fall back to bound workspace
            DeviceWorkspaceManager *ws = workspace ? workspace : workspace_;
            DeviceWorkspaceManager *saved_workspace = workspace_;
            if (ws && ws != workspace_)
            {
                // Temporarily bind passed workspace for this call
                workspace_ = ws;
            }

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

            // Check if tensors are on GPU
            const float *d_input = static_cast<const float *>(A_fp32->gpu_data_ptr());
            float *d_output = static_cast<float *>(C_fp32->gpu_data_ptr());
            const bool use_gpu_path = (d_input != nullptr) && (d_output != nullptr);

            auto phase_start = std::chrono::high_resolution_clock::now();
            auto phase_end = phase_start;

            // Fast path: if bias is present and we're on GPU, use multiply_fp32_to_fp32_with_bias
            if (bias && use_gpu_path)
            {
                const float *d_bias = static_cast<const float *>(bias->gpu_data_ptr());
                if (d_bias)
                {
                    LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_tensor] Using bias path (d_input="
                              << d_input << ", d_output=" << d_output << ", d_bias=" << d_bias << ")");
                    bool result = multiply_fp32_to_fp32_with_bias(d_input, d_output, d_bias, m, n, k, alpha, beta);
                    // Restore original workspace binding
                    if (ws && ws != saved_workspace)
                    {
                        workspace_ = saved_workspace;
                    }
                    return result;
                }
            }

            if (use_gpu_path)
            {
                LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_tensor] Using GPU-to-GPU path (d_input="
                          << d_input << ", d_output=" << d_output << ")");
            }

            // Ensure weights are uploaded to device
            phase_start = std::chrono::high_resolution_clock::now();
            ensureWeightsConverted();
            phase_end = std::chrono::high_resolution_clock::now();
            double weights_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
            if (weights_ms > 1.0)
                LOG_TRACE("[ROCmGEMM::PHASES] ensureWeightsConverted: " << weights_ms << "ms");

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

            // Validate and populate workspace pointers
            phase_start = std::chrono::high_resolution_clock::now();
            validateWorkspace();
            phase_end = std::chrono::high_resolution_clock::now();
            double workbuf_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
            if (workbuf_ms > 1.0)
                LOG_TRACE("[ROCmGEMM::PHASES] validateWorkspace: " << workbuf_ms << "ms");

            int8_t *d_A_int8 = impl_->d_A_int8;
            float *d_scales_A = impl_->d_scales_A;
            int32_t *d_C_int32 = impl_->d_C_int32;

            LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_tensor] Work buffers: A_int8=" << (void *)d_A_int8
                                                                                         << " scales_A=" << (void *)d_scales_A << " C_int32=" << (void *)d_C_int32);

            const size_t a_fp32_size = static_cast<size_t>(m) * k;
            float *d_A_fp32_src = nullptr;

            if (use_gpu_path)
            {
                // GPU path: Use input directly from GPU memory
                d_A_fp32_src = const_cast<float *>(d_input);
                LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_tensor] Using GPU input directly: " << d_A_fp32_src);
            }
            else
            {
                // CPU path: Copy from host to device
                phase_start = std::chrono::high_resolution_clock::now();

                // Workspace is required - d_A_fp32 buffer is pre-allocated
                d_A_fp32_src = impl_->d_A_fp32;
                phase_end = std::chrono::high_resolution_clock::now();
                double alloc_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
                if (alloc_ms > 1.0)
                    LOG_TRACE("[ROCmGEMM::PHASES] d_A_fp32 setup: " << alloc_ms << "ms");

                LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_tensor] Using workspace d_A_fp32=" << (void *)d_A_fp32_src);

                phase_start = std::chrono::high_resolution_clock::now();
                if (!rocmQuantGemm_copyHostToDevice(d_A_fp32_src, A_fp32->data(), a_fp32_size, rocm_device_id_))
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Failed to copy activations to device");
                    return false;
                }
                phase_end = std::chrono::high_resolution_clock::now();
                double h2d_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
                if (h2d_ms > 1.0)
                    LOG_TRACE("[ROCmGEMM::PHASES] H2D copy (A_fp32): " << h2d_ms << "ms");
                LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_tensor] Copied activations to device");
            }

            LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_tensor] Now quantizing activations");

            // Quantize activations FP32 → INT8
            phase_start = std::chrono::high_resolution_clock::now();
            if (!rocmQuantGemm_quantizeActivations(d_A_fp32_src, d_A_int8, d_scales_A, m, k, rocm_device_id_))
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Failed to quantize activations");
                return false;
            }
            phase_end = std::chrono::high_resolution_clock::now();
            double quant_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
            if (quant_ms > 1.0)
                LOG_TRACE("[ROCmGEMM::PHASES] quantizeActivations: " << quant_ms << "ms");

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

            // Determine output buffer based on GPU vs CPU path
            const size_t c_fp32_size = static_cast<size_t>(m) * n;
            float *d_C_fp32_dst = nullptr;

            if (use_gpu_path)
            {
                // GPU path: Write directly to output tensor's GPU buffer
                d_C_fp32_dst = d_output;
                LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_tensor] Using GPU output directly: " << d_C_fp32_dst);
            }
            else
            {
                // CPU path: Use temp buffer from workspace, will copy to host later
                d_C_fp32_dst = impl_->d_C_fp32;
            }

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

            // Use workspace INT32 accumulator buffer (pre-allocated for padded size)
            // No allocation needed - workspace provides all buffers

            if (needs_padding)
            {
                // Padding buffers come from workspace - no allocation needed
                // Padded execution with cached buffers (no hot-path allocations!)
                auto gemm_start = std::chrono::high_resolution_clock::now();
                success = rocmQuantGemm_executeTwoKernel_padded_cached(
                    d_A_int8, d_weights_int8, d_C_fp32_dst,
                    d_scales_A, d_scales_B,
                    impl_->d_CK_int32,
                    impl_->d_A_padded, impl_->d_scale_A_padded, impl_->d_E_padded,
                    m, padded_m, n, k, rocm_device_id_);
                auto gemm_end = std::chrono::high_resolution_clock::now();
                double gemm_ms = std::chrono::duration<double, std::milli>(gemm_end - gemm_start).count();
                LOG_TRACE("[ROCmGEMM::TIMING] executeTwoKernel_padded_cached M=" << m << " N=" << n << " K=" << k << " took " << gemm_ms << "ms");
            }
            else
            {
                // Direct execution: no padding needed
                auto gemm_start = std::chrono::high_resolution_clock::now();
                success = rocmQuantGemm_executeTwoKernel_cached(
                    d_A_int8, d_weights_int8, d_C_fp32_dst,
                    d_scales_A, d_scales_B,
                    impl_->d_CK_int32,
                    m, n, k, rocm_device_id_);
                auto gemm_end = std::chrono::high_resolution_clock::now();
                double gemm_ms = std::chrono::duration<double, std::milli>(gemm_end - gemm_start).count();
                LOG_TRACE("[ROCmGEMM::TIMING] executeTwoKernel_cached M=" << m << " N=" << n << " K=" << k << " took " << gemm_ms << "ms");
            }

            if (!success)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] All GEMM paths failed");
                return false;
            }

            // Copy result back to host (only if not in GPU path)
            if (!use_gpu_path)
            {
                phase_start = std::chrono::high_resolution_clock::now();
                LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_tensor] CPU path: copying to host"
                          << " d_C_fp32_dst=" << d_C_fp32_dst
                          << " h_dst=" << (void *)C_fp32->mutable_data()
                          << " c_fp32_size=" << c_fp32_size);
                if (!rocmQuantGemm_copyDeviceToHost(C_fp32->mutable_data(), d_C_fp32_dst, c_fp32_size, rocm_device_id_))
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Failed to copy output to host");
                    return false;
                }
                phase_end = std::chrono::high_resolution_clock::now();
                double d2h_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
                if (d2h_ms > 1.0)
                    LOG_TRACE("[ROCmGEMM::PHASES] D2H copy (C_fp32): " << d2h_ms << "ms");
            }
            LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_tensor] Completed " << m << "x" << n << "x" << k);

            // Restore original workspace binding
            if (ws && ws != saved_workspace)
            {
                workspace_ = saved_workspace;
            }
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

            // Validate and populate workspace pointers
            validateWorkspace();

            int8_t *d_A_int8 = impl_->d_A_int8;
            float *d_scales_A = impl_->d_scales_A;
            float *d_A_fp32 = impl_->d_A_fp32;
            float *d_C_fp32 = impl_->d_C_fp32;

            // Copy activations H2D and quantize (OUTSIDE timing)
            const size_t a_fp32_size = static_cast<size_t>(m) * k;
            const size_t c_fp32_size = static_cast<size_t>(m) * n;
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

            // Calculate padding
            int padded_m = getPaddedM(m);
            bool needs_padding = (padded_m > m);

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
            int device_idx,
            DeviceWorkspaceManager *workspace)
        {
            (void)mpi_ctx;
            (void)device_idx;
            (void)transpose_B;

            // Use passed workspace if provided, otherwise fall back to bound workspace
            DeviceWorkspaceManager *ws = workspace ? workspace : workspace_;
            DeviceWorkspaceManager *saved_workspace = workspace_;
            if (ws && ws != workspace_)
            {
                // Temporarily bind passed workspace for this call
                workspace_ = ws;
            }

            // TODO: Implement in Phase 5
            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply] Not yet implemented");

            // Restore original workspace binding
            if (ws && ws != saved_workspace)
            {
                workspace_ = saved_workspace;
            }
            return false;
        }

        bool ROCmQuantisedGemmKernel::multiply_fused(
            const float *input,
            const std::vector<FusedProjectionDesc> &projections,
            int m, int k,
            const MPIContext *mpi_ctx,
            int device_idx,
            DeviceWorkspaceManager *workspace)
        {
            (void)mpi_ctx;
            (void)device_idx;

            // Use passed workspace if provided, otherwise fall back to bound workspace
            DeviceWorkspaceManager *ws = workspace ? workspace : workspace_;
            DeviceWorkspaceManager *saved_workspace = workspace_;
            if (ws && ws != workspace_)
            {
                // Temporarily bind passed workspace for this call
                workspace_ = ws;
            }

            // TODO: Implement in Phase 5
            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused] Not yet implemented");

            // Restore original workspace binding
            if (ws && ws != saved_workspace)
            {
                workspace_ = saved_workspace;
            }
            return false;
        }

        bool ROCmQuantisedGemmKernel::multiply_fused_tensor(
            const TensorBase *input,
            const std::vector<TensorProjectionDesc> &projections,
            int m, int k,
            const MPIContext * /*mpi_ctx*/,
            DeviceWorkspaceManager *workspace)
        {
            // Use passed workspace if provided, otherwise fall back to bound workspace
            DeviceWorkspaceManager *ws = workspace ? workspace : workspace_;
            DeviceWorkspaceManager *saved_workspace = workspace_;
            if (ws && ws != workspace_)
            {
                // Temporarily bind passed workspace for this call
                workspace_ = ws;
            }

            if (!input || projections.empty())
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Null input or empty projections");
                return false;
            }

            if (m <= 0 || k <= 0)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Invalid dimensions: m=" << m << " k=" << k);
                return false;
            }

            rocmQuantGemm_setDevice(rocm_device_id_);

            // Step 1: Ensure input is on the GPU
            const float *d_input = nullptr;
            if (input->native_type() == TensorType::FP32)
            {
                auto *fp32_input = dynamic_cast<FP32Tensor *>(const_cast<TensorBase *>(input));
                if (!fp32_input)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Failed to cast input to FP32Tensor");
                    return false;
                }
                // Coherence handled automatically by GraphExecutor
                d_input = static_cast<const float *>(fp32_input->gpu_data_ptr());
                LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Input GPU ptr=" << d_input << " cpu_ptr=" << fp32_input->data());
            }
            else
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Unsupported input type: "
                          << static_cast<int>(input->native_type()));
                return false;
            }

            if (!d_input)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Input has no GPU data");
                return false;
            }

            // Step 2: Validate and populate workspace pointers (shared across all projections)
            validateWorkspace();

            LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Quantizing activations once, m=" << m << " k=" << k);

            // Step 3: Quantize activations ONCE (shared across all projections)
            // Note: activations are already on device (d_input), so we can quantize directly
            if (!rocmQuantGemm_quantizeActivations(
                    const_cast<float *>(d_input), // Source FP32 on device
                    impl_->d_A_int8,              // Destination INT8 on device
                    impl_->d_scales_A,            // Scales on device
                    m, k, rocm_device_id_))
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Activation quantization failed");
                return false;
            }

            // Step 4: Execute each projection using the SHARED quantized activations
            bool all_success = true;
            for (size_t i = 0; i < projections.size() && all_success; ++i)
            {
                const auto &proj = projections[i];
                if (!proj.kernel || !proj.output)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i << " has null kernel or output");
                    all_success = false;
                    break;
                }

                // Get the ROCm kernel for this projection
                auto *rocm_kernel = dynamic_cast<ROCmQuantisedGemmKernel *>(proj.kernel);
                if (!rocm_kernel)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                             << " kernel is not a ROCmQuantisedGemmKernel");
                    all_success = false;
                    break;
                }

                const int n = proj.n;
                LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                         << " (" << (proj.name ? proj.name : "unnamed") << "): m=" << m << " n=" << n << " k=" << k);

                // Ensure the projection's weights are converted
                rocm_kernel->ensureWeightsConverted();

                // Validate this projection's workspace is bound and populated
                rocm_kernel->validateWorkspace();

                // Ensure output tensor is on device
                if (proj.output->native_type() != TensorType::FP32)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                             << " output must be FP32, got " << static_cast<int>(proj.output->native_type()));
                    all_success = false;
                    break;
                }

                auto *fp32_output = dynamic_cast<FP32Tensor *>(proj.output);
                if (!fp32_output)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Failed to cast output to FP32Tensor");
                    all_success = false;
                    break;
                }

                // Coherence handled automatically by GraphExecutor
                float *d_output = static_cast<float *>(fp32_output->gpu_data_ptr());
                if (!d_output)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Output has no GPU data for projection " << i);
                    all_success = false;
                    break;
                }

                // Get weight device pointers from this projection's kernel
                int8_t *d_weights_int8 = nullptr;
                float *d_scales_B = nullptr;

                if (rocm_kernel->packed_)
                {
                    d_weights_int8 = rocm_kernel->packed_->d_int8_data;
                    d_scales_B = rocm_kernel->packed_->d_scales;
                }
                else if (rocm_kernel->impl_)
                {
                    d_weights_int8 = rocm_kernel->impl_->d_weights_int8;
                    d_scales_B = rocm_kernel->impl_->d_scales_B;
                }

                if (!d_weights_int8 || !d_scales_B)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i << " weights not on device");
                    all_success = false;
                    break;
                }

                // M-padding logic for small batch sizes
                const int padded_m = getPaddedM(m);
                const bool needs_padding = needsMPadding(m);

                // Ensure CK INT32 buffer
                const size_t ck_int32_size = static_cast<size_t>(padded_m) * n;
                if (ck_int32_size > rocm_kernel->impl_->d_CK_int32_capacity)
                {
                    if (rocm_kernel->impl_->d_CK_int32)
                        rocmQuantGemm_freeDevice(rocm_kernel->impl_->d_CK_int32);
                    rocm_kernel->impl_->d_CK_int32 = nullptr;
                    rocm_kernel->impl_->d_CK_int32_capacity = 0;

                    if (!rocmQuantGemm_allocInt32(&rocm_kernel->impl_->d_CK_int32, ck_int32_size, rocm_device_id_))
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Failed to allocate CK INT32 buffer for projection " << i);
                        all_success = false;
                        break;
                    }
                    rocm_kernel->impl_->d_CK_int32_capacity = ck_int32_size;
                }

                // Get bias pointer if present
                const float *d_bias = nullptr;
                if (proj.bias)
                {
                    if (proj.bias->native_type() != TensorType::FP32)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                                 << " bias must be FP32, got " << static_cast<int>(proj.bias->native_type()));
                        all_success = false;
                        break;
                    }

                    // Use TensorBase interface - works with both FP32Tensor and TensorSlice
                    // Cast away const to access gpu_data_ptr() (coherence is managed by GraphExecutor)
                    auto *bias_tensor = const_cast<TensorBase *>(proj.bias);
                    d_bias = static_cast<const float *>(bias_tensor->gpu_data_ptr());
                    if (!d_bias)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Bias has no GPU data for projection " << i);
                        all_success = false;
                        break;
                    }
                    LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                             << " using bias ptr=" << static_cast<const void *>(d_bias));
                }

                // Execute CK GEMM using SHARED quantized activations
                bool success = false;

                if (d_bias)
                {
                    // When bias is present, use two-stage approach:
                    // 1. executeNoScale → INT32
                    // 2. applyScaling with bias → FP32

                    if (needs_padding)
                    {
                        // Padding buffers come from workspace - no allocation needed
                        // Use CACHED version to avoid hipMalloc/hipFree per call!
                        success = rocmQuantGemm_executeTwoKernel_padded_cached(
                            impl_->d_A_int8,
                            d_weights_int8,
                            d_output,
                            impl_->d_scales_A,
                            d_scales_B,
                            rocm_kernel->impl_->d_CK_int32,
                            impl_->d_A_padded, impl_->d_scale_A_padded, impl_->d_E_padded,
                            m, padded_m, n, k, rocm_device_id_);

                        if (success)
                        {
                            // Apply bias using GPU kernel (fast path)
                            success = rocmQuantGemm_biasAdd(d_output, d_bias, m, n, rocm_device_id_);
                            if (!success)
                            {
                                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Bias add failed for projection " << i);
                            }
                        }
                    }
                    else
                    {
                        // Non-padded case: use executeNoScale + applyScaling with bias
                        success = rocmQuantGemm_executeNoScale(
                            impl_->d_A_int8,
                            d_weights_int8,
                            rocm_kernel->impl_->d_CK_int32,
                            m, n, k, rocm_device_id_);

                        if (success)
                        {
                            // Apply scaling with bias: output = C_int32 * scale_A * scale_B + bias
                            success = rocmQuantGemm_applyScaling(
                                rocm_kernel->impl_->d_CK_int32,
                                d_output,
                                impl_->d_scales_A,
                                d_scales_B,
                                m, n,
                                1.0f, 0.0f,
                                nullptr, // No existing C
                                d_bias,  // Add bias
                                rocm_device_id_);
                        }
                    }
                }
                else
                {
                    // No bias: use fast path
                    if (needs_padding)
                    {
                        success = rocmQuantGemm_executeTwoKernel_padded(
                            impl_->d_A_int8,
                            d_weights_int8,
                            d_output,
                            impl_->d_scales_A,
                            d_scales_B,
                            rocm_kernel->impl_->d_CK_int32,
                            m, padded_m, n, k, rocm_device_id_);
                    }
                    else
                    {
                        success = rocmQuantGemm_executeTwoKernel_cached(
                            impl_->d_A_int8,
                            d_weights_int8,
                            d_output,
                            impl_->d_scales_A,
                            d_scales_B,
                            rocm_kernel->impl_->d_CK_int32,
                            m, n, k, rocm_device_id_);
                    }
                }

                if (!success)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] CK GEMM failed for projection " << i);
                    all_success = false;
                    break;
                }

                LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i << " complete");
            }

            // Restore original workspace binding
            if (ws && ws != saved_workspace)
            {
                workspace_ = saved_workspace;
            }

            return all_success;
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

            // ROCm-specific: M-padding buffers for CK when M < 8
            // For CK INT32 accumulator, we need max(m, CK_MIN_M_FOR_EXPLICIT_PADDING) * n
            // because direct execution uses m, padded uses CK_MIN_M_FOR_EXPLICIT_PADDING
            const int effective_m_for_ck = std::max(m, CK_MIN_M_FOR_EXPLICIT_PADDING);
            size_t ck_int32_bytes = static_cast<size_t>(effective_m_for_ck) * n * sizeof(int32_t);

            // Padding buffers only need to fit padded_m=8 since padding is only used for M < 8
            const int padded_m = CK_MIN_M_FOR_EXPLICIT_PADDING;
            size_t a_padded_bytes = static_cast<size_t>(padded_m) * k * sizeof(int8_t);
            size_t scale_a_padded_bytes = static_cast<size_t>(padded_m) * sizeof(float);
            size_t e_padded_bytes = static_cast<size_t>(padded_m) * n * sizeof(float);

            reqs.buffers.push_back({GemmWorkspaceBuffers::ROCM_CK_INT32, ck_int32_bytes, 256, true});
            reqs.buffers.push_back({GemmWorkspaceBuffers::ROCM_A_PADDED, a_padded_bytes, 256, true});
            reqs.buffers.push_back({GemmWorkspaceBuffers::ROCM_SCALE_A_PADDED, scale_a_padded_bytes, 256, true});
            reqs.buffers.push_back({GemmWorkspaceBuffers::ROCM_E_PADDED, e_padded_bytes, 256, true});

            LOG_DEBUG("[ROCmQuantisedGemmKernel::getWorkspaceRequirements] INT8 path: "
                      << "quant_a=" << (quant_a_bytes / 1024) << "KB, "
                      << "scales_a=" << (scales_a_bytes) << "B, "
                      << "acc=" << (acc_int32_bytes / 1024) << "KB, "
                      << "ck_int32=" << (ck_int32_bytes / 1024) << "KB (padded)");

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

        void ROCmQuantisedGemmKernel::validateWorkspace() const
        {
            // =========================================================================
            // Workspace is REQUIRED - no fallback allocation
            // =========================================================================
            if (!hasWorkspace())
            {
                throw std::runtime_error(
                    "[ROCmQuantisedGemmKernel] Workspace not bound. Kernels require pre-allocated "
                    "workspace buffers via bindWorkspace(). This is not optional.");
            }

            // Validate required buffers exist
            if (!workspace_->hasBuffer(GemmWorkspaceBuffers::QUANT_A))
            {
                throw std::runtime_error(
                    "[ROCmQuantisedGemmKernel] Workspace missing required buffer: QUANT_A");
            }
            if (!workspace_->hasBuffer(GemmWorkspaceBuffers::SCALES_A))
            {
                throw std::runtime_error(
                    "[ROCmQuantisedGemmKernel] Workspace missing required buffer: SCALES_A");
            }
            if (!workspace_->hasBuffer(GemmWorkspaceBuffers::ACC_INT32))
            {
                throw std::runtime_error(
                    "[ROCmQuantisedGemmKernel] Workspace missing required buffer: ACC_INT32");
            }
            // ROCm-specific buffers
            if (!workspace_->hasBuffer(GemmWorkspaceBuffers::TEMP_A_FP32))
            {
                throw std::runtime_error(
                    "[ROCmQuantisedGemmKernel] Workspace missing required buffer: TEMP_A_FP32");
            }
            if (!workspace_->hasBuffer(GemmWorkspaceBuffers::TEMP_C_FP32))
            {
                throw std::runtime_error(
                    "[ROCmQuantisedGemmKernel] Workspace missing required buffer: TEMP_C_FP32");
            }
            if (!workspace_->hasBuffer(GemmWorkspaceBuffers::ROCM_CK_INT32))
            {
                throw std::runtime_error(
                    "[ROCmQuantisedGemmKernel] Workspace missing required buffer: ROCM_CK_INT32");
            }
            if (!workspace_->hasBuffer(GemmWorkspaceBuffers::ROCM_A_PADDED))
            {
                throw std::runtime_error(
                    "[ROCmQuantisedGemmKernel] Workspace missing required buffer: ROCM_A_PADDED");
            }
            if (!workspace_->hasBuffer(GemmWorkspaceBuffers::ROCM_SCALE_A_PADDED))
            {
                throw std::runtime_error(
                    "[ROCmQuantisedGemmKernel] Workspace missing required buffer: ROCM_SCALE_A_PADDED");
            }
            if (!workspace_->hasBuffer(GemmWorkspaceBuffers::ROCM_E_PADDED))
            {
                throw std::runtime_error(
                    "[ROCmQuantisedGemmKernel] Workspace missing required buffer: ROCM_E_PADDED");
            }

            // Populate impl_ pointers from workspace
            impl_->d_A_int8 = static_cast<int8_t *>(workspace_->getBuffer(GemmWorkspaceBuffers::QUANT_A));
            impl_->d_scales_A = static_cast<float *>(workspace_->getBuffer(GemmWorkspaceBuffers::SCALES_A));
            impl_->d_C_int32 = static_cast<int32_t *>(workspace_->getBuffer(GemmWorkspaceBuffers::ACC_INT32));
            impl_->d_A_fp32 = static_cast<float *>(workspace_->getBuffer(GemmWorkspaceBuffers::TEMP_A_FP32));
            impl_->d_C_fp32 = static_cast<float *>(workspace_->getBuffer(GemmWorkspaceBuffers::TEMP_C_FP32));
            impl_->d_CK_int32 = static_cast<int32_t *>(workspace_->getBuffer(GemmWorkspaceBuffers::ROCM_CK_INT32));
            impl_->d_A_padded = static_cast<int8_t *>(workspace_->getBuffer(GemmWorkspaceBuffers::ROCM_A_PADDED));
            impl_->d_scale_A_padded = static_cast<float *>(workspace_->getBuffer(GemmWorkspaceBuffers::ROCM_SCALE_A_PADDED));
            impl_->d_E_padded = static_cast<float *>(workspace_->getBuffer(GemmWorkspaceBuffers::ROCM_E_PADDED));

            // Set capacity values to max (workspace is pre-sized for maximum dimensions)
            // These are used by code paths that check capacity before use
            impl_->d_A_fp32_capacity = SIZE_MAX;
            impl_->d_C_fp32_capacity = SIZE_MAX;
            impl_->d_CK_int32_capacity = SIZE_MAX;
            impl_->d_A_padded_capacity = SIZE_MAX;
            impl_->d_scale_A_padded_capacity = SIZE_MAX;
            impl_->d_E_padded_capacity = SIZE_MAX;

            LOG_TRACE("[ROCmQuantisedGemmKernel::validateWorkspace] Workspace validated, pointers populated"
                      << " A_int8=" << (void *)impl_->d_A_int8
                      << " scales_A=" << (void *)impl_->d_scales_A
                      << " C_int32=" << (void *)impl_->d_C_int32);
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
            LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32] m=" << m << " n=" << n << " k=" << k
                                                                            << " alpha=" << alpha << " beta=" << beta
                                                                            << " d_A=" << static_cast<const void *>(d_A)
                                                                            << " d_C=" << static_cast<void *>(d_C));

            // Ensure weights converted and validate workspace
            ensureWeightsConverted();
            validateWorkspace();

            // Step 1: Quantize FP32 activations to INT8
            if (!rocmQuantGemm_quantizeActivations(
                    d_A, impl_->d_A_int8, impl_->d_scales_A, m, k, rocm_device_id_))
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel] Activation quantization failed");
                return false;
            }

            // Step 2: Execute CK INT8 GEMM (two-kernel approach: GEMM + scaling separately)
            // The two-kernel cached version takes a pre-allocated INT32 buffer
            if (!rocmQuantGemm_executeTwoKernel_cached(
                    impl_->d_A_int8, impl_->d_weights_int8,
                    d_C, // Output FP32
                    impl_->d_scales_A, impl_->d_scales_B,
                    impl_->d_C_int32, // Pre-allocated INT32 accumulator
                    m, n, k, rocm_device_id_))
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel] CK two-kernel GEMM failed");
                return false;
            }

            // Note: The two-kernel approach already applies scaling with alpha=1.0, beta=0.0
            // If we need different alpha/beta, we'd need to use the full rocmQuantGemm_applyScaling
            // For now, the simple path works since alpha=1.0 and beta=0.0 is the common case
            if (alpha != 1.0f || beta != 0.0f)
            {
                LOG_WARN("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32] "
                         "Non-trivial alpha/beta not yet supported (alpha="
                         << alpha << " beta=" << beta << ")");
            }

            LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32] Complete");
            return true;
        }

        bool ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias(
            const float *d_A, float *d_C, const float *d_bias,
            int m, int n, int k,
            float alpha, float beta)
        {
            LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] m=" << m << " n=" << n << " k=" << k
                                                                                      << " alpha=" << alpha << " beta=" << beta
                                                                                      << " d_A=" << static_cast<const void *>(d_A)
                                                                                      << " d_C=" << static_cast<void *>(d_C)
                                                                                      << " d_bias=" << static_cast<const void *>(d_bias));

            // Ensure weights converted and validate workspace
            ensureWeightsConverted();
            validateWorkspace();

            // Step 1: Quantize FP32 activations to INT8
            if (!rocmQuantGemm_quantizeActivations(
                    d_A, impl_->d_A_int8, impl_->d_scales_A, m, k, rocm_device_id_))
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel] Activation quantization failed");
                return false;
            }

            // Step 2: Execute CK INT8 GEMM → INT32 (no scaling)
            if (!rocmQuantGemm_executeNoScale(
                    impl_->d_A_int8, impl_->d_weights_int8, impl_->d_C_int32,
                    m, n, k, rocm_device_id_))
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel] CK NoScale GEMM failed");
                return false;
            }

            // Step 3: Apply scaling with alpha, beta, and bias
            const float *d_C_existing = (beta != 0.0f) ? d_C : nullptr;
            if (!rocmQuantGemm_applyScaling(
                    impl_->d_C_int32, d_C, impl_->d_scales_A, impl_->d_scales_B,
                    m, n, alpha, beta, d_C_existing, d_bias, rocm_device_id_))
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel] Scaling with bias failed");
                return false;
            }

            LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] Complete");
            return true;
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
