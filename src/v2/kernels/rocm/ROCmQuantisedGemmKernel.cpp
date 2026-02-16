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
#include "tensors/FP16Utils.h"
#include "tensors/TensorKernels.h"
#include "tensors/KernelSnapshotInfo.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/device/WorkspaceDescriptor.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/SlabGemmConfig.h"
#include "utils/Logger.h"
#include "utils/ROCmKernelProfiler.h"
#include "utils/DebugEnv.h"

#include <stdexcept>
#include <vector>
#include <cmath>
#include <algorithm>

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif
#include <cstring>
#include <chrono>

namespace llaminar2
{
    namespace rocm
    {
        // =====================================================================
        // File-level helper: is the fused GEMV path enabled?
        // Default OFF (performance regression on gfx906 due to FP32 atomicAdd).
        // Set LLAMINAR_FUSED_GEMV=1 to enable for testing/development.
        // =====================================================================
        static bool isFusedGemvEnabled()
        {
            const bool enabled = debugEnv().rocm.fused_gemv;
            static bool logged_enabled_once = false;
            if (enabled && !logged_enabled_once)
            {
                LOG_INFO("[ROCmQuantisedGemmKernel] Fused GEMV enabled via DebugEnv (LLAMINAR_FUSED_GEMV=1)");
                logged_enabled_once = true;
            }
            return enabled;
        }

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
                int rocm_device_id, void *stream);

            // Execute INT8 GEMM using CK with SEPARATE scaling (two-kernel approach)
            // This runs CK for INT8→INT32 GEMM, then a separate kernel for scaling.
            bool rocmQuantGemm_executeTwoKernel(
                const int8_t *d_A_int8, // [M x K] RowMajor quantized activations
                const int8_t *d_B_int8, // [K x N] RowMajor transposed weights
                float *d_E_fp32,        // [M x N] RowMajor FP32 output
                const float *d_scale_A, // [M] per-row activation scales
                const float *d_scale_B, // [N] per-column weight scales
                int M, int N, int K,
                int rocm_device_id, void *stream);

            // Execute INT8 GEMM without scaling (INT8→INT32 only)
            // Used when you want to apply custom scaling/bias separately.
            bool rocmQuantGemm_executeNoScale(
                const int8_t *d_A_int8, // [M x K] RowMajor quantized activations
                const int8_t *d_B_int8, // [K x N] RowMajor weights
                int32_t *d_C_int32,     // [M x N] RowMajor INT32 output
                int M, int N, int K,
                int rocm_device_id, void *stream);

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
                int rocm_device_id, void *stream);

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
                int rocm_device_id, void *stream);

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
                int rocm_device_id, void *stream);

            // Execute INT8 GEMM using CK two-kernel with HIP event timing (for benchmarking)
            // Same as _cached but returns kernel time via kernel_time_ms parameter.
            bool rocmQuantGemm_executeTwoKernel_timed(
                const int8_t *d_A_int8, // [M × K] INT8 activations
                const int8_t *d_B_int8, // [K × N] INT8 weights
                float *d_E_fp32,        // [M × N] FP32 output
                const float *d_scale_A, // [M] per-row scales
                const float *d_scale_B, // [N] per-column scales
                int32_t *d_C_int32,     // Pre-allocated [M × N] INT32 accumulator
                int M, int N, int K,
                int rocm_device_id,
                float *kernel_time_ms, void *stream);

            // Allocate INT8 buffer
            bool rocmQuantGemm_allocInt8(int8_t **d_ptr, size_t count, int rocm_device_id);

            // Get CK minimum dimension requirements
            int rocmQuantGemm_getMinM();
            int rocmQuantGemm_getMinN();
            int rocmQuantGemm_getMinK();

            // Free device memory (must set device before freeing)
            void rocmQuantGemm_freeDevice(void *d_ptr, int rocm_device_id);

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
                int rocm_device_id, void *stream);

            // In-place bias addition: output[m,n] += bias[n]
            bool rocmQuantGemm_biasAdd(
                float *d_output,     // [M × N] FP32 output (modified in-place)
                const float *d_bias, // [N] bias vector
                int M, int N,
                int rocm_device_id, void *stream);

            // =========================================================================
            // GEMV kernel for decode (M=1) - bypasses CK INT8 GEMM entirely
            // Defined in ROCmGemvKernel.hip
            // =========================================================================

            // INT8×INT8 GEMV with VNNI-packed weights: B is [K/4 × N × 4]
            bool rocmGemv_int8_int8_int32_vnni(
                const int8_t *d_A_int8,      // [K] INT8 activation vector
                const int8_t *d_B_int8_vnni, // [K/4 × N × 4] VNNI-packed weights
                int32_t *d_C_int32,          // [N] INT32 output vector
                int N, int K,
                int device_id, void *stream);

            bool rocmGemv_int8_int8_fp32_vnni_scaled(
                const int8_t *d_A_int8,
                const int8_t *d_B_int8_vnni,
                float *d_C_fp32,
                const float *d_scale_A,
                const float *d_scale_B,
                int N, int K,
                float alpha,
                float beta,
                const float *d_C_existing,
                const float *d_bias,
                int device_id, void *stream);

            // Repack VNNI [K/4][N][4] → row-major [N×K] into scratch buffer
            // Used for CK GEMM prefill and legacy fp16/fp32 GEMV modes.
            // Cost: ~65-200μs for 18944×3584 (amortized over prefill).
            bool rocmGemv_repackVNNI_to_rowmajor(
                const int8_t *d_B_vnni, // [K/4][N][4] VNNI input
                int8_t *d_B_rowmajor,   // [N×K] row-major output (scratch buffer)
                int N, int K,
                int device_id, void *stream);

            // Ratio-VNNI decode GEMV (Phase-1: IQ4_NL + Q4_0, 4-bit)
            bool rocmGemv_ratio_vnni_int8_int32(
                const int8_t *d_A_int8,   // [K]
                const uint8_t *d_payload, // [blocks × N × payload_bytes]
                const int8_t *d_ratio,    // [blocks × N]
                int32_t *d_C_int32,       // [N]
                int N, int K,
                uint8_t bitwidth,
                uint8_t codebook_id,
                uint8_t has_min,
                uint8_t block_size,
                uint16_t payload_bytes,
                int device_id, void *stream);

            // Ratio-VNNI -> row-major INT8 expansion for CK prefill (Phase-1)
            bool rocmGemv_expandRatioVNNI_to_rowmajor(
                const uint8_t *d_payload,
                const int8_t *d_ratio,
                int8_t *d_B_rowmajor,
                int N, int K,
                uint8_t bitwidth,
                uint8_t codebook_id,
                uint8_t has_min,
                uint8_t block_size,
                uint16_t payload_bytes,
                int device_id, void *stream);

            // =========================================================================
            // Fused FP32→INT8 quantize + GEMV + scale kernel for decode (M=1)
            // Eliminates 3-kernel pipeline: quantize → GEMV → applyScaling
            // Single kernel does per-tile quantization in shared memory, INT8 dot
            // products via v_dot4_i32_i8, and FP32 atomicAdd with scale application.
            // Bias is applied separately via rocmQuantGemm_biasAdd (atomicAdd race).
            // Defined in ROCmGemvKernel.hip
            // =========================================================================
            bool rocmGemv_fused_fp32_int8_vnni(
                const float *d_A_fp32,       // [K] FP32 activation vector (unquantized)
                const int8_t *d_B_int8_vnni, // [K/4 × N × 4] VNNI-packed INT8 weights
                float *d_C_fp32,             // [N] FP32 output vector (zeroed by kernel)
                const float *d_scales_B,     // [N] per-column weight scales
                const float *d_bias,         // [N] optional bias (nullptr if none; applied via biasAdd)
                int N, int K,
                int device_id, void *stream);
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
            // Option B: Only VNNI layout is persistent on device. Row-major is repacked
            // on-demand from VNNI into d_B_rowmajor_scratch (workspace buffer).
            int8_t *d_weights_int8_vnni = nullptr;      // [K/4 x N x 4] VNNI layout (sole device copy)
            uint8_t *d_weights_ratio_payload = nullptr; // [blocks × N × payload_bytes]
            int8_t *d_weights_ratio = nullptr;          // [blocks × N]
            float *d_scales_B = nullptr;                // [N] per-column scales

            uint8_t ratio_vnni_bitwidth = 0;
            uint8_t ratio_vnni_codebook_id = 0;
            uint8_t ratio_vnni_has_min = 0;
            uint8_t ratio_vnni_block_size = 0;
            uint16_t ratio_vnni_payload_bytes = 0;

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

            // Option B: shared scratch buffer for VNNI→row-major repacking (from workspace)
            int8_t *d_B_rowmajor_scratch = nullptr; // [N x K] temporary row-major weights

            // Repack cache metadata (valid only when source pointers, dims, and scratch match)
            bool repack_cache_valid = false;
            int repack_cached_n = 0;
            int repack_cached_k = 0;
            int8_t *repack_cached_src_vnni = nullptr;
            uint8_t *repack_cached_src_ratio_payload = nullptr;
            int8_t *repack_cached_src_ratio = nullptr;
            int8_t *repack_cached_dst = nullptr;

            // Capacity tracking for workspace buffers (set during validateWorkspace)
            size_t d_CK_int32_capacity = 0;
            size_t d_A_fp32_capacity = 0;
            size_t d_C_fp32_capacity = 0;
            size_t d_A_padded_capacity = 0;
            size_t d_scale_A_padded_capacity = 0;
            size_t d_E_padded_capacity = 0;

            // Flag to track if we own weight memory
            bool owns_weight_memory = false;

            // ROCm device ID for proper cleanup
            int rocm_device_id = 0;

            ~Impl()
            {
                // Only free weight memory if we own it (not from ROCmPackedWeights cache)
                if (owns_weight_memory)
                {
                    if (d_weights_int8_vnni)
                        rocmQuantGemm_freeDevice(d_weights_int8_vnni, rocm_device_id);
                    if (d_weights_ratio_payload)
                        rocmQuantGemm_freeDevice(d_weights_ratio_payload, rocm_device_id);
                    if (d_weights_ratio)
                        rocmQuantGemm_freeDevice(d_weights_ratio, rocm_device_id);
                    if (d_scales_B)
                        rocmQuantGemm_freeDevice(d_scales_B, rocm_device_id);
                }
                // Work buffers (including d_B_rowmajor_scratch) are NOT freed -
                // they are owned by workspace
            }
        };

        namespace
        {
            template <typename ImplT>
            inline bool ensureRepackedWeightsForCK(
                ImplT *impl,
                int n, int k,
                int rocm_device_id,
                void *gpu_stream,
                const char *log_scope)
            {
                if (!impl || !impl->d_B_rowmajor_scratch)
                {
                    LOG_ERROR("[" << log_scope << "] Missing repack scratch buffer");
                    return false;
                }

                if (!impl->d_weights_int8_vnni && !(impl->d_weights_ratio_payload && impl->d_weights_ratio))
                {
                    LOG_ERROR("[" << log_scope << "] Missing VNNI/ratio source weights");
                    return false;
                }

                const bool cache_hit = impl->repack_cache_valid &&
                                       impl->repack_cached_n == n &&
                                       impl->repack_cached_k == k &&
                                       impl->repack_cached_src_vnni == impl->d_weights_int8_vnni &&
                                       impl->repack_cached_src_ratio_payload == impl->d_weights_ratio_payload &&
                                       impl->repack_cached_src_ratio == impl->d_weights_ratio &&
                                       impl->repack_cached_dst == impl->d_B_rowmajor_scratch;

                if (cache_hit)
                {
                    LOG_TRACE("[" << log_scope << "] Repack cache hit (N=" << n << ", K=" << k << ")");
                    return true;
                }

                bool repack_ok = false;
                if (impl->d_weights_ratio_payload && impl->d_weights_ratio &&
                    impl->ratio_vnni_bitwidth == 4 &&
                    impl->ratio_vnni_block_size == 32 &&
                    impl->ratio_vnni_payload_bytes == 16)
                {
                    repack_ok = rocmGemv_expandRatioVNNI_to_rowmajor(
                        impl->d_weights_ratio_payload,
                        impl->d_weights_ratio,
                        impl->d_B_rowmajor_scratch,
                        n, k,
                        impl->ratio_vnni_bitwidth,
                        impl->ratio_vnni_codebook_id,
                        impl->ratio_vnni_has_min,
                        impl->ratio_vnni_block_size,
                        impl->ratio_vnni_payload_bytes,
                        rocm_device_id, gpu_stream);
                }
                else if (impl->d_weights_int8_vnni)
                {
                    repack_ok = rocmGemv_repackVNNI_to_rowmajor(
                        impl->d_weights_int8_vnni,
                        impl->d_B_rowmajor_scratch,
                        n, k,
                        rocm_device_id, gpu_stream);
                }

                if (!repack_ok)
                {
                    LOG_ERROR("[" << log_scope << "] ratio/VNNI→row-major repack failed");
                    impl->repack_cache_valid = false;
                    return false;
                }

                impl->repack_cache_valid = true;
                impl->repack_cached_n = n;
                impl->repack_cached_k = k;
                impl->repack_cached_src_vnni = impl->d_weights_int8_vnni;
                impl->repack_cached_src_ratio_payload = impl->d_weights_ratio_payload;
                impl->repack_cached_src_ratio = impl->d_weights_ratio;
                impl->repack_cached_dst = impl->d_B_rowmajor_scratch;
                return true;
            }
        }

        // =====================================================================
        // ROCmPackedWeights destructor
        // =====================================================================

        ROCmPackedWeights::~ROCmPackedWeights()
        {
            if (d_int8_data_vnni || d_ratio_vnni_payload || d_ratio_vnni_ratio || d_scales)
            {
                LOG_TRACE("[ROCmPackedWeights::~] Freeing device memory: "
                          << "d_int8_data_vnni=" << (void *)d_int8_data_vnni
                          << " d_ratio_vnni_payload=" << (void *)d_ratio_vnni_payload
                          << " d_ratio_vnni_ratio=" << (void *)d_ratio_vnni_ratio
                          << " d_scales=" << (void *)d_scales
                          << " rocm_device_id=" << rocm_device_id);
            }
            if (d_int8_data_vnni)
                rocmQuantGemm_freeDevice(d_int8_data_vnni, rocm_device_id);
            if (d_ratio_vnni_payload)
                rocmQuantGemm_freeDevice(d_ratio_vnni_payload, rocm_device_id);
            if (d_ratio_vnni_ratio)
                rocmQuantGemm_freeDevice(d_ratio_vnni_ratio, rocm_device_id);
            if (d_scales)
                rocmQuantGemm_freeDevice(d_scales, rocm_device_id);
        }

        // =====================================================================
        // packWeightsToROCm: Convert any quantized tensor to INT8 + scales
        // =====================================================================

        namespace
        {
            constexpr uint8_t RATIO_VNNI_CODEBOOK_LINEAR = 0;
            constexpr uint8_t RATIO_VNNI_CODEBOOK_IQ4 = 4;

            bool packRatioVNNIPhase1(const TensorBase *tensor, ROCmPackedWeights &out)
            {
                if (!tensor)
                {
                    return false;
                }

                const TensorType wt = tensor->native_type();
                if (wt != TensorType::Q4_0 && wt != TensorType::IQ4_NL)
                {
                    return false;
                }

                auto *accessor = dynamic_cast<const IQuantizedTileAccessor *>(tensor);
                if (!accessor)
                {
                    LOG_DEBUG("[packRatioVNNIPhase1] Tensor does not implement IQuantizedTileAccessor");
                    return false;
                }

                const int N = static_cast<int>(tensor->rows());
                const int K = static_cast<int>(tensor->cols());
                if ((K % 32) != 0)
                {
                    return false;
                }
                if (accessor->block_size() != 32)
                {
                    return false;
                }

                const int blocks_per_row = K / 32;
                constexpr int payload_bytes = 16;

                out.ratio_vnni_payload.resize(static_cast<size_t>(blocks_per_row) * N * payload_bytes);
                out.ratio_vnni_ratio.resize(static_cast<size_t>(blocks_per_row) * N);
                out.ratio_vnni_bitwidth = 4;
                out.ratio_vnni_codebook_id = (wt == TensorType::IQ4_NL) ? RATIO_VNNI_CODEBOOK_IQ4 : RATIO_VNNI_CODEBOOK_LINEAR;
                out.ratio_vnni_has_min = 0;
                out.ratio_vnni_block_size = 32;
                out.ratio_vnni_payload_bytes = payload_bytes;

                // Pass 1: per-column max scale + mandatory 128/127 compensation
                for (int n = 0; n < N; ++n)
                {
                    float max_abs = 0.0f;
                    for (int b = 0; b < blocks_per_row; ++b)
                    {
                        float scale_b = accessor->get_block_scale(static_cast<size_t>(n), static_cast<size_t>(b));
                        max_abs = std::max(max_abs, std::abs(scale_b));
                    }
                    const float base_col = (max_abs > 0.0f) ? max_abs : 1.0f;
                    out.scales[n] = base_col * (128.0f / 127.0f);
                }

                // Pass 2: payload + ratio interleaved by N
                for (int n = 0; n < N; ++n)
                {
                    const float inv_col = 128.0f / out.scales[n];
                    for (int b = 0; b < blocks_per_row; ++b)
                    {
                        const void *raw_block = accessor->get_raw_block(static_cast<size_t>(n), static_cast<size_t>(b));
                        if (!raw_block)
                        {
                            LOG_ERROR("[packRatioVNNIPhase1] get_raw_block returned null at n=" << n << " b=" << b);
                            return false;
                        }

                        const uint8_t *payload_src = nullptr;
                        if (wt == TensorType::IQ4_NL)
                        {
                            const auto *blk = static_cast<const IQ4_NLBlock *>(raw_block);
                            payload_src = blk->qs;
                        }
                        else
                        {
                            const auto *blk = static_cast<const Q4_0Block *>(raw_block);
                            payload_src = blk->qs;
                        }

                        const size_t linear = static_cast<size_t>(b) * N + static_cast<size_t>(n);
                        std::memcpy(out.ratio_vnni_payload.data() + linear * payload_bytes, payload_src, payload_bytes);

                        const float scale_b = accessor->get_block_scale(static_cast<size_t>(n), static_cast<size_t>(b));
                        const float r_f = std::round(std::clamp(scale_b * inv_col, -127.0f, 127.0f));
                        out.ratio_vnni_ratio[linear] = static_cast<int8_t>(r_f);
                    }
                }

                LOG_DEBUG("[packRatioVNNIPhase1] Built Phase-1 ratio container for "
                          << N << "x" << K
                          << " payload=" << (out.ratio_vnni_payload.size() / 1024) << " KB"
                          << " ratio=" << (out.ratio_vnni_ratio.size() / 1024) << " KB");
                return true;
            }
        }

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

            out.scales.resize(N);
            out.K = K;
            out.N = N;

            // Phase-1 compact path: IQ4_NL / Q4_0 -> ratio-VNNI payload + ratio
            // Uses native block access (no FP32 dequant -> INT8 requant round-trip).
            if (packRatioVNNIPhase1(tensor, out))
            {
                out.int8_data.clear();
                out.int8_data_vnni.clear();
                LOG_DEBUG("[packWeightsToROCm] Packed " << N << "x" << K
                                                        << " weights to ratio-VNNI (phase1)");
                return true;
            }

            // Legacy generic path for all other quant formats:
            //   int8_data: [N × K] row-major (same layout as model weights!)
            //             This is Column-Major [K × N] for CK's mk_nk_mn convention
            //   scales: [N] per-output-feature scales
            out.int8_data.resize(static_cast<size_t>(N) * K);

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

            // Optional VNNI layout: [K/4][N][4] for GEMV experimentation
            out.int8_data_vnni.clear();
            if ((K % 4) == 0)
            {
                const size_t k_groups = static_cast<size_t>(K) / 4;
                out.int8_data_vnni.resize(k_groups * static_cast<size_t>(N) * 4);
                for (int n = 0; n < N; ++n)
                {
                    const size_t row_base = static_cast<size_t>(n) * K;
                    for (size_t kg = 0; kg < k_groups; ++kg)
                    {
                        const size_t src = row_base + kg * 4;
                        const size_t dst = (kg * static_cast<size_t>(N) + static_cast<size_t>(n)) * 4;
                        out.int8_data_vnni[dst + 0] = out.int8_data[src + 0];
                        out.int8_data_vnni[dst + 1] = out.int8_data[src + 1];
                        out.int8_data_vnni[dst + 2] = out.int8_data[src + 2];
                        out.int8_data_vnni[dst + 3] = out.int8_data[src + 3];
                    }
                }
            }

            LOG_DEBUG("[packWeightsToROCm] Packed " << N << "x" << K << " weights to INT8 (mk_nk_mn layout)"
                                                    << (out.int8_data_vnni.empty() ? "" : " + VNNI"));
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

            impl_->owns_weight_memory = true;        // Legacy constructor owns weight memory
            impl_->rocm_device_id = rocm_device_id_; // Store device ID for cleanup

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

            impl_->owns_weight_memory = false;       // Pre-packed path doesn't own weight memory
            impl_->rocm_device_id = rocm_device_id_; // Store device ID for cleanup

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
            ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::GEMM_CK, static_cast<hipStream_t>(gpu_stream_));
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
            // IMPORTANT: For BAR-backed tensors, use rocm_data_ptr() (HIP pointer) instead of
            // gpu_data_ptr() (which returns CUDA pointer for BAR-backed). This enables zero-copy
            // where ROCm kernels write directly to BAR memory that CUDA can then read.
            const float *d_input = nullptr;
            float *d_output = nullptr;

            if (A_fp32->isBARBacked() && A_fp32->rocm_data_ptr() != nullptr)
            {
                d_input = static_cast<const float *>(A_fp32->rocm_data_ptr());
                LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_tensor] Using BAR rocm_data_ptr for input A: " << d_input);
            }
            else
            {
                d_input = static_cast<const float *>(A_fp32->gpu_data_ptr());
            }

            if (C_fp32->isBARBacked() && C_fp32->rocm_data_ptr() != nullptr)
            {
                d_output = static_cast<float *>(C_fp32->rocm_data_ptr());
                LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_tensor] Using BAR rocm_data_ptr for output C: " << d_output);
            }
            else
            {
                d_output = static_cast<float *>(C_fp32->gpu_data_ptr());
            }

            const bool use_gpu_path = (d_input != nullptr) && (d_output != nullptr);

            auto phase_start = std::chrono::high_resolution_clock::now();
            auto phase_end = phase_start;

            if (use_gpu_path)
            {
                LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_tensor] Using GPU-to-GPU path (d_input="
                          << d_input << ", d_output=" << d_output << ")");
            }

            // =========================================================================
            // DECODE FAST PATH: M=1 GEMV (skips activation quantization + CK GEMM)
            //
            // For M=1 (single-token decode), the CK INT8 GEMM is catastrophically
            // inefficient (2.4 TFLOPS = 8% of MI60 peak) because decode is
            // bandwidth-bound, not compute-bound. Our custom GEMV kernel reads
            // INT8 weights directly and multiplies with FP32 activations,
            // eliminating:
            //   1. FP32→INT8 activation quantization kernel
            //   2. M-padding (hipMemset + hipMemcpy)
            //   3. CK 32×32 tile overhead (wastes 31 of 32 rows)
            //   4. INT32→FP32 scale application kernel
            //
            // Handles optional bias in a single fused kernel launch.
            // =========================================================================
            if (use_gpu_path && m == 1)
            {
                // Ensure weights are on device
                ensureWeightsConverted();

                // Option B: weights are stored as VNNI and/or ratio-VNNI on device
                int8_t *d_weights_vnni = impl_ ? impl_->d_weights_int8_vnni : nullptr;
                uint8_t *d_weights_ratio_payload = impl_ ? impl_->d_weights_ratio_payload : nullptr;
                int8_t *d_weights_ratio = impl_ ? impl_->d_weights_ratio : nullptr;
                float *d_scales_B = nullptr;
                if (packed_)
                {
                    d_scales_B = packed_->d_scales;
                }
                else if (impl_)
                {
                    d_scales_B = impl_->d_scales_B;
                }

                if ((d_weights_vnni || (d_weights_ratio_payload && d_weights_ratio)) && d_scales_B)
                {
                    // Resolve bias device pointer if present
                    const float *d_bias = nullptr;
                    if (bias)
                    {
                        if (bias->isBARBacked())
                            d_bias = static_cast<const float *>(bias->rocm_data_ptr());
                        else
                            d_bias = static_cast<const float *>(bias->gpu_data_ptr());
                    }

                    LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_tensor] GEMV fast path M=1 N=" << n << " K=" << k
                                                                                                 << (d_bias ? " +bias" : ""));

                    // INT8 VNNI GEMV (only path — fp16/fp32 modes removed)
                    if (!impl_)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] INT8 GEMV requires impl_ buffers");
                        return false;
                    }

                    validateWorkspace();

                    // Use fused FP32→INT8+GEMV+scale kernel when alpha=1, beta=0 AND fused enabled
                    if (alpha == 1.0f && beta == 0.0f && isFusedGemvEnabled())
                    {
                        if (!rocmGemv_fused_fp32_int8_vnni(
                                d_input, d_weights_vnni, d_output, d_scales_B, d_bias,
                                n, k, rocm_device_id_, gpu_stream_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Fused GEMV failed");
                            return false;
                        }
                    }
                    else
                    {
                        // Fallback: separate quantize → GEMV → applyScaling for non-standard alpha/beta
                        if (!rocmQuantGemm_quantizeActivations(
                                d_input, impl_->d_A_int8, impl_->d_scales_A, m, k, rocm_device_id_, gpu_stream_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Activation quantization failed");
                            return false;
                        }

                        const float *d_existing = (beta != 0.0f) ? d_output : nullptr;
                        bool fused_scale = false;
                        bool gemv_ok = false;
                        if (d_weights_ratio_payload && d_weights_ratio &&
                            impl_->ratio_vnni_bitwidth == 4 &&
                            impl_->ratio_vnni_block_size == 32 &&
                            impl_->ratio_vnni_payload_bytes == 16)
                        {
                            gemv_ok = rocmGemv_ratio_vnni_int8_int32(
                                impl_->d_A_int8,
                                d_weights_ratio_payload,
                                d_weights_ratio,
                                impl_->d_C_int32,
                                n, k,
                                impl_->ratio_vnni_bitwidth,
                                impl_->ratio_vnni_codebook_id,
                                impl_->ratio_vnni_has_min,
                                impl_->ratio_vnni_block_size,
                                impl_->ratio_vnni_payload_bytes,
                                rocm_device_id_, gpu_stream_);
                        }
                        else if (d_weights_vnni)
                        {
                            fused_scale = rocmGemv_int8_int8_fp32_vnni_scaled(
                                impl_->d_A_int8, d_weights_vnni, d_output,
                                impl_->d_scales_A, d_scales_B,
                                n, k,
                                alpha, beta,
                                d_existing, d_bias,
                                rocm_device_id_, gpu_stream_);

                            if (!fused_scale)
                            {
                                gemv_ok = rocmGemv_int8_int8_int32_vnni(
                                    impl_->d_A_int8, d_weights_vnni, impl_->d_C_int32,
                                    n, k, rocm_device_id_, gpu_stream_);
                            }
                        }

                        if (!fused_scale && !gemv_ok)
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] INT8 GEMV (ratio/VNNI) failed");
                            return false;
                        }

                        if (!fused_scale && !rocmQuantGemm_applyScaling(
                                                impl_->d_C_int32, d_output, impl_->d_scales_A, d_scales_B,
                                                m, n, alpha, beta, d_existing, d_bias, rocm_device_id_, gpu_stream_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] INT8 GEMV scaling failed");
                            return false;
                        }
                    }

                    if (ws && ws != saved_workspace)
                        workspace_ = saved_workspace;
                    return true;
                }
                // Fall through to CK path if weight pointers unavailable
            }

            // CK path: if bias is present and we're on GPU, use multiply_fp32_to_fp32_with_bias
            if (bias && use_gpu_path)
            {
                // Get bias device pointer - check BAR-backed status
                const float *d_bias = nullptr;
                if (bias->isBARBacked())
                {
                    d_bias = static_cast<const float *>(bias->rocm_data_ptr());
                }
                else
                {
                    d_bias = static_cast<const float *>(bias->gpu_data_ptr());
                }

                if (d_bias)
                {
                    LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_tensor] Using CK bias path (d_input="
                              << d_input << ", d_output=" << d_output << ", d_bias=" << d_bias << ")");
                    bool result = multiply_fp32_to_fp32_with_bias(d_input, d_output, d_bias, m, n, k, alpha, beta);
                    if (ws && ws != saved_workspace)
                    {
                        workspace_ = saved_workspace;
                    }
                    return result;
                }
            }

            // Ensure weights are uploaded to device
            phase_start = std::chrono::high_resolution_clock::now();
            ensureWeightsConverted();
            phase_end = std::chrono::high_resolution_clock::now();
            double weights_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
            if (weights_ms > 1.0)
                LOG_TRACE("[ROCmGEMM::PHASES] ensureWeightsConverted: " << weights_ms << "ms");

            // Get weight device pointers
            // Option B: weights are stored as VNNI only. For CK GEMM (needs row-major),
            // we repack VNNI→row-major into a shared workspace scratch buffer.
            float *d_scales_B = nullptr;

            if (packed_)
            {
                d_scales_B = packed_->d_scales;
            }
            else if (impl_)
            {
                d_scales_B = impl_->d_scales_B;
            }

            if (!d_scales_B)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Weight scales not uploaded to device");
                return false;
            }

            // Validate and populate workspace pointers (includes d_B_rowmajor_scratch)
            phase_start = std::chrono::high_resolution_clock::now();
            validateWorkspace();
            phase_end = std::chrono::high_resolution_clock::now();
            double workbuf_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
            if (workbuf_ms > 1.0)
                LOG_TRACE("[ROCmGEMM::PHASES] validateWorkspace: " << workbuf_ms << "ms");

            // Option B: Repack VNNI→row-major into workspace scratch for CK GEMM
            int8_t *d_weights_int8 = nullptr;
            if (impl_->d_B_rowmajor_scratch)
            {
                phase_start = std::chrono::high_resolution_clock::now();
                if (!ensureRepackedWeightsForCK(
                        impl_.get(), n, k, rocm_device_id_, gpu_stream_,
                        "ROCmQuantisedGemmKernel::multiply_tensor"))
                {
                    return false;
                }
                d_weights_int8 = impl_->d_B_rowmajor_scratch;
                phase_end = std::chrono::high_resolution_clock::now();
                double repack_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
                LOG_TRACE("[ROCmGEMM::PHASES] VNNI→rowmajor repack: " << repack_ms << "ms");
            }

            if (!d_weights_int8)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] No VNNI weights for CK GEMM repack");
                return false;
            }

            LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_tensor] Weight ptrs: int8(scratch)=" << (void *)d_weights_int8 << " scales=" << (void *)d_scales_B);

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
            if (!rocmQuantGemm_quantizeActivations(d_A_fp32_src, d_A_int8, d_scales_A, m, k, rocm_device_id_, gpu_stream_))
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
                    m, padded_m, n, k, rocm_device_id_, gpu_stream_);
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
                    m, n, k, rocm_device_id_, gpu_stream_);
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

            // Get weight device pointers — Option B: VNNI and/or ratio-VNNI, repack to scratch
            validateWorkspace();
            if ((!impl_->d_weights_int8_vnni && !(impl_->d_weights_ratio_payload && impl_->d_weights_ratio)) ||
                !impl_->d_B_rowmajor_scratch)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor_timed] No VNNI/ratio weights or repack scratch");
                return false;
            }
            float *d_scales_B = impl_->d_scales_B;
            if (!d_scales_B)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor_timed] Weights not uploaded");
                return false;
            }
            if (!ensureRepackedWeightsForCK(
                    impl_.get(), n, k, rocm_device_id_, gpu_stream_,
                    "ROCmQuantisedGemmKernel::multiply_tensor_timed"))
            {
                return false;
            }
            int8_t *d_weights_int8 = impl_->d_B_rowmajor_scratch;

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
            if (!rocmQuantGemm_quantizeActivations(d_A_fp32, d_A_int8, d_scales_A, m, k, rocm_device_id_, gpu_stream_))
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
                    m, padded_m, n, k, rocm_device_id_, gpu_stream_);
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
                    kernel_time_ms, gpu_stream_);
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
                // Coherence handled automatically by DeviceGraphExecutor
                // IMPORTANT: For BAR-backed tensors, use rocm_data_ptr() (HIP pointer)
                if (fp32_input->isBARBacked() && fp32_input->rocm_data_ptr() != nullptr)
                {
                    d_input = static_cast<const float *>(fp32_input->rocm_data_ptr());
                    LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Using BAR rocm_data_ptr for input: " << d_input);
                }
                else
                {
                    d_input = static_cast<const float *>(fp32_input->gpu_data_ptr());
                }
                // NOTE: Don't log fp32_input->data() here - it triggers D2H transfer!
                LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Input GPU ptr=" << d_input);
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

            const bool use_fused_m1 = (m == 1 && isFusedGemvEnabled());

            // Step 3: Quantize activations ONCE (shared across all projections)
            // For M=1 with fused GEMV, quantization is done in-kernel (shared memory).
            // For M>1 or M=1 fallback, use the separate quantization kernel.
            if (!use_fused_m1)
            {
                LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Quantizing activations once, m=" << m << " k=" << k);

                // Note: activations are already on device (d_input), so we can quantize directly
                if (!rocmQuantGemm_quantizeActivations(
                        const_cast<float *>(d_input), // Source FP32 on device
                        impl_->d_A_int8,              // Destination INT8 on device
                        impl_->d_scales_A,            // Scales on device
                        m, k, rocm_device_id_, gpu_stream_))
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Activation quantization failed");
                    return false;
                }
            }
            else
            {
                LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fused_tensor] M=1 fused path: skipping separate quantize kernel");
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

                // Coherence handled automatically by DeviceGraphExecutor
                // IMPORTANT: For BAR-backed tensors, use rocm_data_ptr() (HIP pointer)
                float *d_output = nullptr;
                if (fp32_output->isBARBacked() && fp32_output->rocm_data_ptr() != nullptr)
                {
                    d_output = static_cast<float *>(fp32_output->rocm_data_ptr());
                    LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Using BAR rocm_data_ptr for output: " << d_output);
                }
                else
                {
                    d_output = static_cast<float *>(fp32_output->gpu_data_ptr());
                }
                if (!d_output)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Output has no GPU data for projection " << i);
                    all_success = false;
                    break;
                }

                // Get weight scales from this projection's kernel
                float *d_scales_B = nullptr;

                if (rocm_kernel->packed_)
                {
                    d_scales_B = rocm_kernel->packed_->d_scales;
                }
                else if (rocm_kernel->impl_)
                {
                    d_scales_B = rocm_kernel->impl_->d_scales_B;
                }

                if (!d_scales_B)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i << " weight scales not on device");
                    all_success = false;
                    break;
                }

                // Resolve bias pointer early (needed by both GEMV and CK paths)
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

                    auto *bias_tensor = const_cast<TensorBase *>(proj.bias);
                    DeviceId target_device = DeviceId::rocm(rocm_device_id_);
                    auto current_dev = bias_tensor->current_device();

                    LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Proj " << i
                                                                                       << " bias tensor=" << bias_tensor
                                                                                       << " current_dev=" << (current_dev.has_value() ? current_dev->to_string() : "(none)")
                                                                                       << " target_device=" << target_device.to_string()
                                                                                       << " gpu_data_ptr=" << bias_tensor->gpu_data_ptr()
                                                                                       << " rocm_device_id_=" << rocm_device_id_);

                    if (current_dev.has_value() && current_dev.value() == target_device)
                    {
                        if (bias_tensor->isBARBacked() && bias_tensor->rocm_data_ptr() != nullptr)
                        {
                            d_bias = static_cast<const float *>(bias_tensor->rocm_data_ptr());
                            LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Using BAR rocm_data_ptr for bias: " << d_bias);
                        }
                        else
                        {
                            d_bias = static_cast<const float *>(bias_tensor->gpu_data_ptr());
                        }
                    }
                    else if (current_dev.has_value() && current_dev->is_gpu())
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] MULTI-GPU CONFLICT: Bias tensor is on "
                                  << current_dev->to_string() << " but we need ROCm:" << rocm_device_id_
                                  << ". Ensure WeightPreloader::uploadNonGemmWeights() was called for this device.");
                        all_success = false;
                        break;
                    }
                    else
                    {
                        if (!bias_tensor->ensureOnDevice(target_device))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Failed to upload bias to ROCm:" << rocm_device_id_);
                            all_success = false;
                            break;
                        }
                        if (bias_tensor->isBARBacked() && bias_tensor->rocm_data_ptr() != nullptr)
                        {
                            d_bias = static_cast<const float *>(bias_tensor->rocm_data_ptr());
                        }
                        else
                        {
                            d_bias = static_cast<const float *>(bias_tensor->gpu_data_ptr());
                        }
                    }

                    if (!d_bias)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Bias has no GPU data for projection " << i);
                        all_success = false;
                        break;
                    }
                    LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                             << " using bias ptr=" << static_cast<const void *>(d_bias));
                }

                // =========================================================================
                // DECODE FAST PATH: M=1 GEMV (skips CK GEMM entirely)
                //
                // For M=1 (single-token decode), we use a FUSED kernel that combines
                // FP32→INT8 quantization + INT8×INT8 GEMV + FP32 scaling into a single
                // kernel launch. This eliminates the entire 3-kernel pipeline:
                //   1. quantizeActivationsQ8_kernel (eliminated - done in shared memory)
                //   2. GEMV INT8×INT8→INT32 (fused)
                //   3. applyScaling INT32→FP32 (eliminated - scale applied in-kernel)
                //
                // The fused kernel:
                //   - Reads FP32 activations directly (no pre-quantization needed)
                //   - Quantizes per-tile (256 elements) in shared memory via warp reduction
                //   - Computes INT8 dot products using v_dot4_i32_i8
                //   - Applies scale_A * scale_B and atomicAdd's FP32 result to output
                //   - Bias applied separately (atomicAdd race prevents fusion)
                //
                // Result: 1 kernel instead of 3, eliminating ~3.6ms/token pipeline overhead.
                // =========================================================================
                if (m == 1)
                {
                    int8_t *d_vnni = rocm_kernel->impl_ ? rocm_kernel->impl_->d_weights_int8_vnni : nullptr;
                    uint8_t *d_ratio_payload = rocm_kernel->impl_ ? rocm_kernel->impl_->d_weights_ratio_payload : nullptr;
                    int8_t *d_ratio = rocm_kernel->impl_ ? rocm_kernel->impl_->d_weights_ratio : nullptr;
                    if (!d_vnni && !(d_ratio_payload && d_ratio))
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i << " VNNI/ratio weights not on device");
                        all_success = false;
                        break;
                    }

                    if (use_fused_m1 && d_vnni)
                    {
                        // FUSED PATH: FP32→INT8 quantize + GEMV + scale in one kernel launch
                        LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                                 << " FUSED GEMV M=1 N=" << n << " K=" << k
                                                                                                 << (d_bias ? " +bias" : ""));
                        if (!rocmGemv_fused_fp32_int8_vnni(
                                d_input, d_vnni, d_output, d_scales_B, d_bias,
                                n, k, rocm_device_id_, gpu_stream_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Fused GEMV failed for projection " << i);
                            all_success = false;
                            break;
                        }
                    }
                    else
                    {
                        // OLD PATH: separate quantize (already done above) → GEMV → applyScaling
                        const size_t gemv_int32_size = static_cast<size_t>(n);
                        if (gemv_int32_size > rocm_kernel->impl_->d_CK_int32_capacity)
                        {
                            if (rocm_kernel->impl_->d_CK_int32)
                                rocmQuantGemm_freeDevice(rocm_kernel->impl_->d_CK_int32, rocm_device_id_);
                            rocm_kernel->impl_->d_CK_int32 = nullptr;
                            rocm_kernel->impl_->d_CK_int32_capacity = 0;

                            if (!rocmQuantGemm_allocInt32(&rocm_kernel->impl_->d_CK_int32, gemv_int32_size, rocm_device_id_))
                            {
                                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Failed to allocate GEMV INT32 buffer");
                                all_success = false;
                                break;
                            }
                            rocm_kernel->impl_->d_CK_int32_capacity = gemv_int32_size;
                        }

                        LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                                 << " OLD GEMV M=1 N=" << n << " K=" << k
                                                                                                 << (d_bias ? " +bias" : ""));

                        bool fused_scale = false;
                        bool gemv_ok = false;
                        if (d_ratio_payload && d_ratio &&
                            rocm_kernel->impl_->ratio_vnni_bitwidth == 4 &&
                            rocm_kernel->impl_->ratio_vnni_block_size == 32 &&
                            rocm_kernel->impl_->ratio_vnni_payload_bytes == 16)
                        {
                            gemv_ok = rocmGemv_ratio_vnni_int8_int32(
                                impl_->d_A_int8,
                                d_ratio_payload,
                                d_ratio,
                                rocm_kernel->impl_->d_CK_int32,
                                n, k,
                                rocm_kernel->impl_->ratio_vnni_bitwidth,
                                rocm_kernel->impl_->ratio_vnni_codebook_id,
                                rocm_kernel->impl_->ratio_vnni_has_min,
                                rocm_kernel->impl_->ratio_vnni_block_size,
                                rocm_kernel->impl_->ratio_vnni_payload_bytes,
                                rocm_device_id_, gpu_stream_);
                        }
                        else if (d_vnni)
                        {
                            fused_scale = rocmGemv_int8_int8_fp32_vnni_scaled(
                                impl_->d_A_int8, d_vnni, d_output,
                                impl_->d_scales_A, d_scales_B,
                                n, k,
                                1.0f, 0.0f,
                                nullptr, d_bias,
                                rocm_device_id_, gpu_stream_);

                            if (!fused_scale)
                            {
                                gemv_ok = rocmGemv_int8_int8_int32_vnni(
                                    impl_->d_A_int8, d_vnni, rocm_kernel->impl_->d_CK_int32,
                                    n, k, rocm_device_id_, gpu_stream_);
                            }
                        }

                        if (!fused_scale && !gemv_ok)
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] GEMV failed for projection " << i);
                            all_success = false;
                            break;
                        }

                        if (!fused_scale && !rocmQuantGemm_applyScaling(
                                                rocm_kernel->impl_->d_CK_int32, d_output, impl_->d_scales_A, d_scales_B,
                                                m, n, 1.0f, 0.0f, nullptr, d_bias, rocm_device_id_, gpu_stream_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] GEMV scaling failed for projection " << i);
                            all_success = false;
                            break;
                        }
                    }

                    LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i << " GEMV complete");
                    continue; // Skip CK path below
                }

                // =========================================================================
                // PREFILL PATH: M>1 CK GEMM
                // =========================================================================

                // Repack VNNI→row-major into this kernel's workspace scratch for CK
                int8_t *d_weights_int8 = nullptr;
                int8_t *d_vnni = rocm_kernel->impl_ ? rocm_kernel->impl_->d_weights_int8_vnni : nullptr;
                uint8_t *d_ratio_payload = rocm_kernel->impl_ ? rocm_kernel->impl_->d_weights_ratio_payload : nullptr;
                int8_t *d_ratio = rocm_kernel->impl_ ? rocm_kernel->impl_->d_weights_ratio : nullptr;
                int8_t *d_scratch = rocm_kernel->impl_ ? rocm_kernel->impl_->d_B_rowmajor_scratch : nullptr;

                if (d_scratch)
                {
                    if (!ensureRepackedWeightsForCK(
                            rocm_kernel->impl_.get(), n, k, rocm_device_id_, gpu_stream_,
                            "ROCmQuantisedGemmKernel::multiply_fused_tensor"))
                    {
                        all_success = false;
                        break;
                    }
                    d_weights_int8 = d_scratch;
                }

                if (!d_weights_int8)
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
                        rocmQuantGemm_freeDevice(rocm_kernel->impl_->d_CK_int32, rocm_device_id_);
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

                // Execute CK GEMM using SHARED quantized activations (M>1 only)
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
                            m, padded_m, n, k, rocm_device_id_, gpu_stream_);

                        if (success)
                        {
                            // Apply bias using GPU kernel (fast path)
                            success = rocmQuantGemm_biasAdd(d_output, d_bias, m, n, rocm_device_id_, gpu_stream_);
                            if (!success)
                            {
                                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Bias add failed for projection " << i);
                            }
                        }
                    }
                    else
                    {
                        // Non-padded case: use executeNoScale + applyScaling with bias
                        LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                                 << " BEFORE executeNoScale m=" << m << " n=" << n << " k=" << k
                                                                                                 << " device=" << rocm_device_id_);
                        success = rocmQuantGemm_executeNoScale(
                            impl_->d_A_int8,
                            d_weights_int8,
                            rocm_kernel->impl_->d_CK_int32,
                            m, n, k, rocm_device_id_, gpu_stream_);
                        LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                                 << " AFTER executeNoScale success=" << success);

                        if (success)
                        {
                            // Apply scaling with bias: output = C_int32 * scale_A * scale_B + bias
                            LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                                     << " BEFORE applyScaling device=" << rocm_device_id_);
                            success = rocmQuantGemm_applyScaling(
                                rocm_kernel->impl_->d_CK_int32,
                                d_output,
                                impl_->d_scales_A,
                                d_scales_B,
                                m, n,
                                1.0f, 0.0f,
                                nullptr, // No existing C
                                d_bias,  // Add bias
                                rocm_device_id_, gpu_stream_);
                            LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                                     << " AFTER applyScaling success=" << success);
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
                            m, padded_m, n, k, rocm_device_id_, gpu_stream_);
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
                            m, n, k, rocm_device_id_, gpu_stream_);
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

            // Option B: shared scratch buffer for VNNI→row-major repacking
            // Sized to N×K for this kernel's weight matrix. The workspace manager
            // allocates max(N×K) across all kernel instances, so the largest weight
            // matrix (Gate/Up: 18944×3584 ≈ 65MB) determines the actual allocation.
            // This buffer is reused across layers since they execute sequentially.
            size_t repack_bytes = static_cast<size_t>(n) * k * sizeof(int8_t);
            reqs.buffers.push_back({GemmWorkspaceBuffers::ROCM_B_REPACK, repack_bytes, 256, true});

            LOG_DEBUG("[ROCmQuantisedGemmKernel::getWorkspaceRequirements] INT8 path: "
                      << "quant_a=" << (quant_a_bytes / 1024) << "KB, "
                      << "scales_a=" << (scales_a_bytes) << "B, "
                      << "acc=" << (acc_int32_bytes / 1024) << "KB, "
                      << "ck_int32=" << (ck_int32_bytes / 1024) << "KB (padded), "
                      << "repack=" << (repack_bytes / 1024) << "KB");

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
                    // Option B: Upload ONLY VNNI layout + scales to device.
                    // Row-major weights are repacked on-demand from VNNI into a
                    // shared workspace scratch buffer for CK GEMM prefill.

                    // Upload scales
                    rocmQuantGemm_setDevice(rocm_device_id_);
                    if (!rocmQuantGemm_allocFloat(&packed_->d_scales,
                                                  static_cast<size_t>(packed_->N),
                                                  rocm_device_id_))
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to alloc scales");
                        return;
                    }

                    hipError_t err = hipMemcpy(packed_->d_scales,
                                               packed_->scales.data(),
                                               static_cast<size_t>(packed_->N) * sizeof(float),
                                               hipMemcpyHostToDevice);
                    if (err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to upload scales: "
                                  << hipGetErrorString(err));
                        return;
                    }

                    // Upload VNNI layout (sole weight copy on device)
                    if (!packed_->int8_data_vnni.empty())
                    {
                        if (!rocmQuantGemm_allocInt8(&packed_->d_int8_data_vnni,
                                                     packed_->int8_data_vnni.size(),
                                                     rocm_device_id_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to alloc VNNI weights");
                            return;
                        }

                        err = hipMemcpy(packed_->d_int8_data_vnni,
                                        packed_->int8_data_vnni.data(),
                                        packed_->int8_data_vnni.size() * sizeof(int8_t),
                                        hipMemcpyHostToDevice);
                        if (err != hipSuccess)
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to upload VNNI weights: "
                                      << hipGetErrorString(err));
                            return;
                        }

                        LOG_DEBUG("[ROCmQuantisedGemmKernel] Uploaded VNNI weights (Option B, sole device copy): "
                                  << (packed_->int8_data_vnni.size() / 1024) << " KB");
                    }

                    // Upload Phase-1 ratio-VNNI compact buffers if available
                    if (!packed_->ratio_vnni_payload.empty() && !packed_->ratio_vnni_ratio.empty())
                    {
                        if (!rocmQuantGemm_allocInt8(reinterpret_cast<int8_t **>(&packed_->d_ratio_vnni_payload),
                                                     packed_->ratio_vnni_payload.size(),
                                                     rocm_device_id_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to alloc ratio payload");
                            return;
                        }
                        if (!rocmQuantGemm_allocInt8(&packed_->d_ratio_vnni_ratio,
                                                     packed_->ratio_vnni_ratio.size(),
                                                     rocm_device_id_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to alloc ratio bytes");
                            return;
                        }

                        err = hipMemcpy(packed_->d_ratio_vnni_payload,
                                        packed_->ratio_vnni_payload.data(),
                                        packed_->ratio_vnni_payload.size() * sizeof(uint8_t),
                                        hipMemcpyHostToDevice);
                        if (err != hipSuccess)
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to upload ratio payload: "
                                      << hipGetErrorString(err));
                            return;
                        }

                        err = hipMemcpy(packed_->d_ratio_vnni_ratio,
                                        packed_->ratio_vnni_ratio.data(),
                                        packed_->ratio_vnni_ratio.size() * sizeof(int8_t),
                                        hipMemcpyHostToDevice);
                        if (err != hipSuccess)
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to upload ratio bytes: "
                                      << hipGetErrorString(err));
                            return;
                        }

                        LOG_DEBUG("[ROCmQuantisedGemmKernel] Uploaded ratio-VNNI payload="
                                  << (packed_->ratio_vnni_payload.size() / 1024) << " KB ratio="
                                  << (packed_->ratio_vnni_ratio.size() / 1024) << " KB");
                    }

                    if (packed_->int8_data_vnni.empty() && packed_->ratio_vnni_payload.empty())
                    {
                        LOG_WARN("[ROCmQuantisedGemmKernel] No VNNI or ratio-VNNI layout available. "
                                 "ROCm GEMV/CK prefill paths may not work.");
                    }

                    packed_->uploaded = true;
                    packed_->rocm_device_id = rocm_device_id_;
                    LOG_DEBUG("[ROCmQuantisedGemmKernel] Uploaded pre-packed weights (Option B): "
                              << packed_->N << "x" << packed_->K
                              << " VRAM: " << (packed_->int8_data_vnni.size() / 1024) << " KB"
                              << " (was " << (packed_->int8_data_vnni.size() * 2 / 1024) << " KB with row-major)");
                }

                // Point impl_ to packed_ device pointers
                impl_->d_weights_int8_vnni = packed_->d_int8_data_vnni;
                impl_->d_weights_ratio_payload = packed_->d_ratio_vnni_payload;
                impl_->d_weights_ratio = packed_->d_ratio_vnni_ratio;
                impl_->d_scales_B = packed_->d_scales;
                impl_->ratio_vnni_bitwidth = packed_->ratio_vnni_bitwidth;
                impl_->ratio_vnni_codebook_id = packed_->ratio_vnni_codebook_id;
                impl_->ratio_vnni_has_min = packed_->ratio_vnni_has_min;
                impl_->ratio_vnni_block_size = packed_->ratio_vnni_block_size;
                impl_->ratio_vnni_payload_bytes = packed_->ratio_vnni_payload_bytes;
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

            // Option B (legacy path): Upload only VNNI + scales
            rocmQuantGemm_setDevice(rocm_device_id_);

            // Upload scales
            if (!rocmQuantGemm_allocFloat(&impl_->d_scales_B,
                                          static_cast<size_t>(host_packed.N),
                                          rocm_device_id_))
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to alloc scales");
                return;
            }

            hipError_t err = hipMemcpy(impl_->d_scales_B,
                                       host_packed.scales.data(),
                                       static_cast<size_t>(host_packed.N) * sizeof(float),
                                       hipMemcpyHostToDevice);
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to upload scales: "
                          << hipGetErrorString(err));
                return;
            }

            // Upload VNNI layout
            if (!host_packed.int8_data_vnni.empty())
            {
                if (!rocmQuantGemm_allocInt8(&impl_->d_weights_int8_vnni,
                                             host_packed.int8_data_vnni.size(),
                                             rocm_device_id_))
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to alloc VNNI weights");
                    return;
                }

                err = hipMemcpy(impl_->d_weights_int8_vnni,
                                host_packed.int8_data_vnni.data(),
                                host_packed.int8_data_vnni.size() * sizeof(int8_t),
                                hipMemcpyHostToDevice);
                if (err != hipSuccess)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to upload VNNI weights: "
                              << hipGetErrorString(err));
                    return;
                }
            }

            if (!host_packed.ratio_vnni_payload.empty() && !host_packed.ratio_vnni_ratio.empty())
            {
                if (!rocmQuantGemm_allocInt8(reinterpret_cast<int8_t **>(&impl_->d_weights_ratio_payload),
                                             host_packed.ratio_vnni_payload.size(),
                                             rocm_device_id_))
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to alloc ratio payload");
                    return;
                }
                if (!rocmQuantGemm_allocInt8(&impl_->d_weights_ratio,
                                             host_packed.ratio_vnni_ratio.size(),
                                             rocm_device_id_))
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to alloc ratio bytes");
                    return;
                }

                err = hipMemcpy(impl_->d_weights_ratio_payload,
                                host_packed.ratio_vnni_payload.data(),
                                host_packed.ratio_vnni_payload.size() * sizeof(uint8_t),
                                hipMemcpyHostToDevice);
                if (err != hipSuccess)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to upload ratio payload: "
                              << hipGetErrorString(err));
                    return;
                }

                err = hipMemcpy(impl_->d_weights_ratio,
                                host_packed.ratio_vnni_ratio.data(),
                                host_packed.ratio_vnni_ratio.size() * sizeof(int8_t),
                                hipMemcpyHostToDevice);
                if (err != hipSuccess)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to upload ratio bytes: "
                              << hipGetErrorString(err));
                    return;
                }

                impl_->ratio_vnni_bitwidth = host_packed.ratio_vnni_bitwidth;
                impl_->ratio_vnni_codebook_id = host_packed.ratio_vnni_codebook_id;
                impl_->ratio_vnni_has_min = host_packed.ratio_vnni_has_min;
                impl_->ratio_vnni_block_size = host_packed.ratio_vnni_block_size;
                impl_->ratio_vnni_payload_bytes = host_packed.ratio_vnni_payload_bytes;
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
            if (!workspace_->hasBuffer(GemmWorkspaceBuffers::ROCM_B_REPACK))
            {
                throw std::runtime_error(
                    "[ROCmQuantisedGemmKernel] Workspace missing required buffer: ROCM_B_REPACK");
            }

            // Populate impl_ pointers from workspace
            const auto prev_repack_scratch = impl_->d_B_rowmajor_scratch;
            impl_->d_A_int8 = static_cast<int8_t *>(workspace_->getBuffer(GemmWorkspaceBuffers::QUANT_A));
            impl_->d_scales_A = static_cast<float *>(workspace_->getBuffer(GemmWorkspaceBuffers::SCALES_A));
            impl_->d_C_int32 = static_cast<int32_t *>(workspace_->getBuffer(GemmWorkspaceBuffers::ACC_INT32));
            impl_->d_A_fp32 = static_cast<float *>(workspace_->getBuffer(GemmWorkspaceBuffers::TEMP_A_FP32));
            impl_->d_C_fp32 = static_cast<float *>(workspace_->getBuffer(GemmWorkspaceBuffers::TEMP_C_FP32));
            impl_->d_CK_int32 = static_cast<int32_t *>(workspace_->getBuffer(GemmWorkspaceBuffers::ROCM_CK_INT32));
            impl_->d_A_padded = static_cast<int8_t *>(workspace_->getBuffer(GemmWorkspaceBuffers::ROCM_A_PADDED));
            impl_->d_scale_A_padded = static_cast<float *>(workspace_->getBuffer(GemmWorkspaceBuffers::ROCM_SCALE_A_PADDED));
            impl_->d_E_padded = static_cast<float *>(workspace_->getBuffer(GemmWorkspaceBuffers::ROCM_E_PADDED));
            impl_->d_B_rowmajor_scratch = static_cast<int8_t *>(workspace_->getBuffer(GemmWorkspaceBuffers::ROCM_B_REPACK));

            if (impl_->d_B_rowmajor_scratch != prev_repack_scratch)
            {
                impl_->repack_cache_valid = false;
            }

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

            // DECODE FAST PATH: M=1 GEMV
            if (m == 1)
            {
                ensureWeightsConverted();
                // Option B: weights are VNNI and/or ratio-VNNI on device
                int8_t *d_vnni = impl_ ? impl_->d_weights_int8_vnni : nullptr;
                uint8_t *d_ratio_payload = impl_ ? impl_->d_weights_ratio_payload : nullptr;
                int8_t *d_ratio = impl_ ? impl_->d_weights_ratio : nullptr;
                float *d_s = packed_ ? packed_->d_scales : (impl_ ? impl_->d_scales_B : nullptr);
                if ((d_vnni || (d_ratio_payload && d_ratio)) && d_s)
                {
                    LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32] GEMV fast path M=1");

                    if (!impl_)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32] INT8 GEMV requires impl_ buffers");
                        return false;
                    }

                    validateWorkspace();

                    // Use fused kernel when alpha=1, beta=0 AND fused enabled
                    if (alpha == 1.0f && beta == 0.0f && isFusedGemvEnabled() && d_vnni)
                    {
                        if (!rocmGemv_fused_fp32_int8_vnni(
                                d_A, d_vnni, d_C, d_s, nullptr,
                                n, k, rocm_device_id_, gpu_stream_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32] Fused GEMV failed");
                            return false;
                        }
                        return true;
                    }

                    // Fallback: separate quantize → GEMV → applyScaling for non-standard alpha/beta
                    if (!rocmQuantGemm_quantizeActivations(
                            d_A, impl_->d_A_int8, impl_->d_scales_A, m, k, rocm_device_id_, gpu_stream_))
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32] Activation quantization failed");
                        return false;
                    }

                    const float *d_existing = (beta != 0.0f) ? d_C : nullptr;
                    bool fused_scale = false;
                    bool gemv_ok = false;
                    if (d_ratio_payload && d_ratio &&
                        impl_->ratio_vnni_bitwidth == 4 &&
                        impl_->ratio_vnni_block_size == 32 &&
                        impl_->ratio_vnni_payload_bytes == 16)
                    {
                        gemv_ok = rocmGemv_ratio_vnni_int8_int32(
                            impl_->d_A_int8,
                            d_ratio_payload,
                            d_ratio,
                            impl_->d_C_int32,
                            n, k,
                            impl_->ratio_vnni_bitwidth,
                            impl_->ratio_vnni_codebook_id,
                            impl_->ratio_vnni_has_min,
                            impl_->ratio_vnni_block_size,
                            impl_->ratio_vnni_payload_bytes,
                            rocm_device_id_, gpu_stream_);
                    }
                    else if (d_vnni)
                    {
                        fused_scale = rocmGemv_int8_int8_fp32_vnni_scaled(
                            impl_->d_A_int8, d_vnni, d_C,
                            impl_->d_scales_A, d_s,
                            n, k,
                            alpha, beta,
                            d_existing, nullptr,
                            rocm_device_id_, gpu_stream_);

                        if (!fused_scale)
                        {
                            gemv_ok = rocmGemv_int8_int8_int32_vnni(
                                impl_->d_A_int8, d_vnni, impl_->d_C_int32,
                                n, k, rocm_device_id_, gpu_stream_);
                        }
                    }

                    if (!fused_scale && !gemv_ok)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32] INT8 GEMV (ratio/VNNI) failed");
                        return false;
                    }

                    if (fused_scale)
                    {
                        return true;
                    }

                    return rocmQuantGemm_applyScaling(
                        impl_->d_C_int32, d_C, impl_->d_scales_A, d_s,
                        m, n, alpha, beta, d_existing, nullptr, rocm_device_id_, gpu_stream_);
                }
            }

            // Ensure weights converted and validate workspace
            ensureWeightsConverted();
            validateWorkspace();

            // Option B: Repack VNNI→row-major into workspace scratch for CK GEMM
            if ((!impl_->d_weights_int8_vnni && !(impl_->d_weights_ratio_payload && impl_->d_weights_ratio)) ||
                !impl_->d_B_rowmajor_scratch)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32] No VNNI/ratio weights or repack scratch for CK GEMM");
                return false;
            }
            if (!ensureRepackedWeightsForCK(
                    impl_.get(), n, k, rocm_device_id_, gpu_stream_,
                    "ROCmQuantisedGemmKernel::multiply_fp32_to_fp32"))
            {
                return false;
            }

            // Step 1: Quantize FP32 activations to INT8
            if (!rocmQuantGemm_quantizeActivations(
                    d_A, impl_->d_A_int8, impl_->d_scales_A, m, k, rocm_device_id_, gpu_stream_))
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel] Activation quantization failed");
                return false;
            }

            // Step 2: Execute CK INT8 GEMM (two-kernel approach: GEMM + scaling separately)
            // The two-kernel cached version takes a pre-allocated INT32 buffer
            if (!rocmQuantGemm_executeTwoKernel_cached(
                    impl_->d_A_int8, impl_->d_B_rowmajor_scratch,
                    d_C, // Output FP32
                    impl_->d_scales_A, impl_->d_scales_B,
                    impl_->d_C_int32, // Pre-allocated INT32 accumulator
                    m, n, k, rocm_device_id_, gpu_stream_))
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

            // DECODE FAST PATH: M=1 GEMV with fused bias
            if (m == 1)
            {
                ensureWeightsConverted();
                // Option B: weights are VNNI and/or ratio-VNNI on device
                int8_t *d_vnni = impl_ ? impl_->d_weights_int8_vnni : nullptr;
                uint8_t *d_ratio_payload = impl_ ? impl_->d_weights_ratio_payload : nullptr;
                int8_t *d_ratio = impl_ ? impl_->d_weights_ratio : nullptr;
                float *d_s = packed_ ? packed_->d_scales : (impl_ ? impl_->d_scales_B : nullptr);
                if ((d_vnni || (d_ratio_payload && d_ratio)) && d_s)
                {
                    LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] GEMV fast path M=1 +bias");

                    if (!impl_)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] INT8 GEMV requires impl_ buffers");
                        return false;
                    }

                    validateWorkspace();

                    // Use fused kernel when alpha=1, beta=0 AND fused enabled
                    // Bias handled inside dispatch function via biasAdd
                    if (alpha == 1.0f && beta == 0.0f && isFusedGemvEnabled() && d_vnni)
                    {
                        return rocmGemv_fused_fp32_int8_vnni(
                            d_A, d_vnni, d_C, d_s, d_bias,
                            n, k, rocm_device_id_, gpu_stream_);
                    }

                    // Fallback: separate quantize → GEMV → applyScaling for non-standard alpha/beta
                    if (!rocmQuantGemm_quantizeActivations(
                            d_A, impl_->d_A_int8, impl_->d_scales_A, m, k, rocm_device_id_, gpu_stream_))
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] Activation quantization failed");
                        return false;
                    }

                    const float *d_existing = (beta != 0.0f) ? d_C : nullptr;
                    bool fused_scale = false;
                    bool gemv_ok = false;
                    if (d_ratio_payload && d_ratio &&
                        impl_->ratio_vnni_bitwidth == 4 &&
                        impl_->ratio_vnni_block_size == 32 &&
                        impl_->ratio_vnni_payload_bytes == 16)
                    {
                        gemv_ok = rocmGemv_ratio_vnni_int8_int32(
                            impl_->d_A_int8,
                            d_ratio_payload,
                            d_ratio,
                            impl_->d_C_int32,
                            n, k,
                            impl_->ratio_vnni_bitwidth,
                            impl_->ratio_vnni_codebook_id,
                            impl_->ratio_vnni_has_min,
                            impl_->ratio_vnni_block_size,
                            impl_->ratio_vnni_payload_bytes,
                            rocm_device_id_, gpu_stream_);
                    }
                    else if (d_vnni)
                    {
                        fused_scale = rocmGemv_int8_int8_fp32_vnni_scaled(
                            impl_->d_A_int8, d_vnni, d_C,
                            impl_->d_scales_A, d_s,
                            n, k,
                            alpha, beta,
                            d_existing, d_bias,
                            rocm_device_id_, gpu_stream_);

                        if (!fused_scale)
                        {
                            gemv_ok = rocmGemv_int8_int8_int32_vnni(
                                impl_->d_A_int8, d_vnni, impl_->d_C_int32,
                                n, k, rocm_device_id_, gpu_stream_);
                        }
                    }

                    if (!fused_scale && !gemv_ok)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] INT8 GEMV (ratio/VNNI) failed");
                        return false;
                    }

                    if (fused_scale)
                    {
                        return true;
                    }

                    return rocmQuantGemm_applyScaling(
                        impl_->d_C_int32, d_C, impl_->d_scales_A, d_s,
                        m, n, alpha, beta, d_existing, d_bias, rocm_device_id_, gpu_stream_);
                }
            }

            // Ensure weights converted and validate workspace
            ensureWeightsConverted();
            validateWorkspace();

            // Option B: Repack VNNI→row-major into workspace scratch for CK GEMM
            if ((!impl_->d_weights_int8_vnni && !(impl_->d_weights_ratio_payload && impl_->d_weights_ratio)) ||
                !impl_->d_B_rowmajor_scratch)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] No VNNI/ratio weights or repack scratch for CK GEMM");
                return false;
            }
            if (!ensureRepackedWeightsForCK(
                    impl_.get(), n, k, rocm_device_id_, gpu_stream_,
                    "ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias"))
            {
                return false;
            }

            // Step 1: Quantize FP32 activations to INT8
            if (!rocmQuantGemm_quantizeActivations(
                    d_A, impl_->d_A_int8, impl_->d_scales_A, m, k, rocm_device_id_, gpu_stream_))
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel] Activation quantization failed");
                return false;
            }

            // Step 2: Execute CK INT8 GEMM → INT32 (no scaling)
            if (!rocmQuantGemm_executeNoScale(
                    impl_->d_A_int8, impl_->d_B_rowmajor_scratch, impl_->d_C_int32,
                    m, n, k, rocm_device_id_, gpu_stream_))
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel] CK NoScale GEMM failed");
                return false;
            }

            // Step 3: Apply scaling with alpha, beta, and bias
            const float *d_C_existing = (beta != 0.0f) ? d_C : nullptr;
            if (!rocmQuantGemm_applyScaling(
                    impl_->d_C_int32, d_C, impl_->d_scales_A, impl_->d_scales_B,
                    m, n, alpha, beta, d_C_existing, d_bias, rocm_device_id_, gpu_stream_))
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
