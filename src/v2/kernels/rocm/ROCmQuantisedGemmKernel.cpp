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
#include "ROCmKernelBase.h"
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
#include "utils/Assertions.h"
#include "utils/WeightLoadingProfiler.h"

#include <stdexcept>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <mutex>

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
        // Forward declarations for HIP implementation
        // =====================================================================
        //
        // Functions are defined across three .hip files:
        //   - ROCmQuantisedGemmKernel.hip      : Common kernels + utilities (quantization, scaling, memory)
        //   - ROCmQuantisedGemmKernel_CK.hip   : CK INT8 GEMM dispatch + kernel context
        //   - ROCmQuantisedGemmKernel_INT8_VNNI.hip : VNNI prefill/decode kernels
        //

        // These functions are implemented in ROCmQuantisedGemmKernel_CK.hip
        extern "C"
        {
            // Create/destroy per-instance CK kernel context (eliminates global/static cache state)
            void *rocmQuantGemm_createKernelContext(int rocm_device_id);                   // CK.hip
            void rocmQuantGemm_destroyKernelContext(void *kernel_ctx, int rocm_device_id); // CK.hip

            // Upload converted INT8 weights to device (common .hip)
            bool rocmQuantGemm_uploadWeights(
                const int8_t *h_weights_int8, // [K x N] ColumnMajor
                const float *h_scales_B,      // [N] per-column scales
                int8_t **d_weights_int8,      // Output device pointer
                float **d_scales_B,           // Output device pointer
                int K, int N,
                int rocm_device_id);

            // Upload work buffers for activation quantization (common .hip)
            bool rocmQuantGemm_ensureWorkBuffers(
                int8_t **d_A_int8,   // [M x K] quantized activations
                float **d_scales_A,  // [M] per-row scales
                int32_t **d_C_int32, // [M x N] INT32 accumulator
                int *work_buffer_M,  // Current capacity
                int M, int K, int N,
                int rocm_device_id);

            // Quantize FP32 activations to INT8 (common .hip)
            bool rocmQuantGemm_quantizeActivations(
                const float *d_A_fp32, // [M x K]
                int8_t *d_A_int8,      // [M x K] output
                float *d_scales_A,     // [M] output
                int M, int K,
                int rocm_device_id, void *stream);

            // Execute INT8 GEMM using CK with SEPARATE scaling (two-kernel approach)
            // This runs CK for INT8→INT32 GEMM, then a separate kernel for scaling. (common .hip)
            bool rocmQuantGemm_executeTwoKernel(
                const int8_t *d_A_int8, // [M x K] RowMajor quantized activations
                const int8_t *d_B_int8, // [K x N] RowMajor transposed weights
                float *d_E_fp32,        // [M x N] RowMajor FP32 output
                const float *d_scale_A, // [M] per-row activation scales
                const float *d_scale_B, // [N] per-column weight scales
                int M, int N, int K,
                int rocm_device_id, void *stream,
                void *kernel_ctx);

            // Execute INT8 GEMM without scaling (INT8→INT32 only, CK.hip)
            // Used when you want to apply custom scaling/bias separately.
            bool rocmQuantGemm_executeNoScale(
                const int8_t *d_A_int8, // [M x K] RowMajor quantized activations
                const int8_t *d_B_int8, // [K x N] RowMajor weights
                int32_t *d_C_int32,     // [M x N] RowMajor INT32 output
                int M, int N, int K,
                int rocm_device_id, void *stream,
                void *kernel_ctx);

            // Execute INT8 GEMM using CK two-kernel with PRE-ALLOCATED buffer (common .hip)
            // This is the preferred variant for hot-path execution.
            bool rocmQuantGemm_executeTwoKernel_cached(
                const int8_t *d_A_int8, // [M x K] RowMajor quantized activations
                const int8_t *d_B_int8, // [K x N] RowMajor transposed weights
                float *d_E_fp32,        // [M x N] RowMajor FP32 output
                const float *d_scale_A, // [M] per-row activation scales
                const float *d_scale_B, // [N] per-column weight scales
                int32_t *d_C_int32,     // [M x N] Pre-allocated INT32 accumulator
                int M, int N, int K,
                int rocm_device_id, void *stream,
                void *kernel_ctx);

            // Execute INT8 GEMM using CK two-kernel with M-PADDING for decode (M < 8, common .hip)
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
                int rocm_device_id, void *stream,
                void *kernel_ctx);

            // Execute INT8 GEMM using CK two-kernel with M-PADDING and PRE-ALLOCATED buffers (common .hip)
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
                int rocm_device_id, void *stream,
                void *kernel_ctx);

            // Execute INT8 GEMM using CK two-kernel with HIP event timing (common .hip)
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
                float *kernel_time_ms, void *stream,
                void *kernel_ctx);

            // Allocate INT8 buffer (common .hip)
            bool rocmQuantGemm_allocInt8(int8_t **d_ptr, size_t count, int rocm_device_id);

            // Get CK minimum dimension requirements (CK.hip)
            int rocmQuantGemm_getMinM();
            int rocmQuantGemm_getMinN();
            int rocmQuantGemm_getMinK();

            // Free device memory (common .hip)
            void rocmQuantGemm_freeDevice(void *d_ptr, int rocm_device_id);

            // Memory management helpers (all common .hip)
            bool rocmQuantGemm_allocFloat(float **d_ptr, size_t count, int rocm_device_id);
            bool rocmQuantGemm_allocInt32(int32_t **d_ptr, size_t count, int rocm_device_id);
            bool rocmQuantGemm_copyHostToDevice(float *d_dst, const float *h_src, size_t count, int rocm_device_id);
            bool rocmQuantGemm_copyDeviceToHost(float *h_dst, const float *d_src, size_t count, int rocm_device_id);
            bool rocmQuantGemm_copyInt32DeviceToHost(int32_t *h_dst, const int32_t *d_src, size_t count, int rocm_device_id);
            bool rocmQuantGemm_setDevice(int rocm_device_id);

            // Apply scaling with full epilogue (common .hip) - matches CUDA's cudaQuantGemm_applyScaling
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

            // Apply scale-A only (for grouped ratio-VNNI path where B-scales
            // are already applied per-group in the GEMV kernel).
            bool rocmQuantGemm_applyScaleA_fp32(
                const float *d_C_fp32_grouped, // [M×N] FP32 with B-scales applied
                float *d_output,               // [M×N] FP32 output
                const float *d_scales_A,       // [M] per-row activation scales
                int M, int N,
                float alpha, float beta,
                const float *d_C_existing,
                const float *d_bias,
                int rocm_device_id, void *stream);

            // In-place bias addition: output[m,n] += bias[n] (common .hip)
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

            // Native-VNNI GEMV: lossless Q4/IQ4 decode with FP16 per-block scales.
            // Output is FP32 with scale_A applied inline — no epilogue kernel needed.
            bool rocmGemv_native_vnni_fp32(
                const int8_t *d_A_int8,
                const uint8_t *d_payload,
                const void *d_block_scales, // __half* (FP16)
                float *d_C_fp32,
                const float *d_scale_A,
                int N, int K,
                uint8_t codebook_id,
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

            // =========================================================================
            // Fused scatter+reduce GEMV: 2-kernel pipeline replacing the 3-kernel
            // (quantize → GEMV → scale) pipeline. Scatter kernel fuses FP32→INT8
            // quantization + INT8 GEMV and writes unscaled FP32 partials.
            // Reduce kernel sums partials and applies scale_B + bias.
            // Eliminates memset, separate quant launch, and one kernel launch.
            // For KB=1, falls back to single-kernel fused path internally.
            // Defined in ROCmGemvKernel.hip
            // =========================================================================
            bool rocmGemv_fused_scatter_fp32_int8_vnni(
                const float *d_A_fp32,       // [K] FP32 activations
                const int8_t *d_B_int8_vnni, // [K/4 × N × 4] VNNI-packed weights
                float *d_C_fp32,             // [N] FP32 output
                const float *d_scales_B,     // [N] per-column weight scales
                const float *d_bias,         // [N] optional bias (nullable)
                float *d_partial_buf,        // [KB_MAX × N] pre-allocated partial buffer
                int N, int K,
                float alpha, float beta,
                const float *d_C_existing, // [N] used when beta != 0
                int device_id, void *stream);

            // =========================================================================
            // INT8-input scatter GEMV: pre-quantized activations + hybrid self-reduce.
            // Combines "quantize once" efficiency with scatter pipeline benefits
            // (no atomicAdd, no memset). Uses self-reduce for small N (K/V),
            // 2-kernel scatter+reduce for large N (Q/Wo/FFN).
            // Defined in ROCmGemvKernel.hip
            // =========================================================================
            bool rocmGemv_int8_scatter_vnni(
                const int8_t *d_A_int8,      // [K] pre-quantized INT8 activations
                const int8_t *d_B_int8_vnni, // [K/4 × N × 4] VNNI-packed weights
                float *d_C_fp32,             // [N] FP32 output
                const float *d_scale_A,      // [1] activation scale (device pointer)
                const float *d_scales_B,     // [N] per-column weight scales
                const float *d_bias,         // [N] optional bias (nullable)
                float *d_partial_buf,        // [KB_MAX × N] pre-allocated partial buffer
                int N, int K,
                float alpha, float beta,
                const float *d_C_existing, // [N] used when beta != 0
                int device_id, void *stream);

            // =========================================================================
            // Batched INT8 scatter GEMV: multiple projections in 2 kernel launches.
            // QKV: 6→2 launches, Gate/Up: 4→2 launches.
            // Better reduce occupancy from batched grid.
            // Defined in ROCmGemvKernel.hip
            // =========================================================================
            bool rocmGemv_int8_scatter_batched_vnni(
                const int8_t *d_A_int8,              // [K] pre-quantized INT8 activations
                const float *d_scale_A,              // [1] activation scale (device pointer)
                float *d_partial_buf,                // [KB_MAX × max_N] partial buffer
                int num_projections,                 // Number of projections (1..8)
                const int8_t *const *d_B_ptrs,       // [num_proj] VNNI weight pointers
                float *const *d_C_ptrs,              // [num_proj] FP32 output pointers
                const float *const *d_scales_B_ptrs, // [num_proj] weight scale pointers
                const float *const *d_bias_ptrs,     // [num_proj] bias pointers
                const int *N_per_proj,               // [num_proj] output dimensions
                int K,
                float alpha, float beta,
                int device_id, void *stream);

            // =========================================================================
            // PREFILL GEMM scaffold entrypoints (M>1)
            //
            // These functions are intentionally scaffold-first in early phases. They
            // let C++ dispatch code route into a stable C API now, while HIP kernel
            // internals are implemented incrementally behind the same ABI.
            // =========================================================================

            // INT8 VNNI native prefill GEMM scaffold: INT8 x INT8 -> INT32
            // Defined in ROCmQuantisedGemmKernel_INT8_VNNI.hip.
            bool rocmQuantGemm_int8_int8_int32_vnni_prefill(
                const int8_t *d_A_int8,      // [M x K] row-major INT8 activations
                const int8_t *d_B_int8_vnni, // [K/4 x N x 4] VNNI-packed INT8 weights
                int32_t *d_C_int32,          // [M x N] row-major INT32 accumulators
                int M, int N, int K,
                int tile_variant,
                int cpt,
                int device_id, void *stream);

            // INT8 VNNI native prefill GEMM split-K/grid-kpar variant.
            // Defined in ROCmQuantisedGemmKernel_INT8_VNNI.hip.
            bool rocmQuantGemm_int8_int8_int32_vnni_prefill_grid_kpar(
                const int8_t *d_A_int8,
                const int8_t *d_B_int8_vnni,
                int32_t *d_C_int32,
                int M, int N, int K,
                int split_k_slices,
                int tile_variant,
                int cpt,
                int kernel_body_variant,
                int grid_swizzle_variant,
                int device_id, void *stream);

            // INT8 VNNI wide-tile V3 prefill GEMM (LDS double-buffered pipeline).
            // kt_select: 8 or 16.  Auto-dispatched for K-heavy shapes (k >= n).
            // Defined in ROCmQuantisedGemmKernel_INT8_VNNI.hip.
            bool rocmQuantGemm_int8_int8_int32_vnni_prefill_wide_tile_v3(
                const int8_t *d_A_int8,
                const int8_t *d_B_int8_vnni,
                int32_t *d_C_int32,
                int M, int N, int K,
                int kt_select,
                int device_id, void *stream);

            // INT8 VNNI wide-tile V7 prefill GEMM (N128 safe-tile split, branchless interior).
            // kt_select: 8 or 16.  Auto-dispatched for N-heavy shapes (n > k).
            // Defined in ROCmQuantisedGemmKernel_INT8_VNNI.hip.
            bool rocmQuantGemm_int8_int8_int32_vnni_prefill_wide_tile_v7(
                const int8_t *d_A_int8,
                const int8_t *d_B_int8_vnni,
                int32_t *d_C_int32,
                int M, int N, int K,
                int kt_select,
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
            void *ck_kernel_context = nullptr;

            // Device memory for converted weights (only used when owns_weight_memory_ = true)
            // Option B: Only VNNI layout is persistent on device. Row-major is repacked
            // on-demand from VNNI into d_B_rowmajor_scratch (workspace buffer).
            int8_t *d_weights_int8_vnni = nullptr;       // [K/4 x N x 4] VNNI layout (sole device copy)
            int8_t *d_weights_int8_rowmajor = nullptr;   // [N x K] optional persistent CK row-major buffer (startup repack)
            float *d_scales_B = nullptr;                 // [N] per-column scales
            uint8_t *d_weights_native_payload = nullptr; // [blocks_per_row × N × 16]
            void *d_weights_native_scales = nullptr;     // [blocks_per_row × N] __half*
            uint8_t native_vnni_codebook_id = 0;
            uint32_t native_vnni_blocks_per_row = 0;
            bool has_native_vnni = false;
            void *startup_repack_ready_event = nullptr; // hipEvent_t* as opaque pointer
            bool startup_repack_event_pending = false;
            void *startup_commit_ready_event = nullptr; // hipEvent_t* as opaque pointer
            bool startup_commit_event_pending = false;
            void *startup_h2d_pinned_scales = nullptr;
            void *startup_h2d_pinned_vnni = nullptr;
            void *startup_h2d_pinned_native_payload = nullptr;
            void *startup_h2d_pinned_native_scales = nullptr;

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

            // Scatter+reduce partial buffer (from workspace)
            float *d_scatter_partial = nullptr; // [KB_MAX × N] FP32 scatter partials

            // Repack cache metadata (valid only when source pointers, dims, and scratch match)
            bool repack_cache_valid = false;
            int repack_cached_n = 0;
            int repack_cached_k = 0;
            int8_t *repack_cached_src_vnni = nullptr;
            int8_t *repack_cached_dst = nullptr;

            // Capacity tracking for workspace buffers (set during validateWorkspace)
            size_t d_CK_int32_capacity = 0;
            size_t d_A_fp32_capacity = 0;
            size_t d_C_fp32_capacity = 0;
            size_t d_A_padded_capacity = 0;
            size_t d_scale_A_padded_capacity = 0;
            size_t d_E_padded_capacity = 0;

            // Workspace validation cache: skip redundant hash map lookups when
            // the same workspace is re-validated across GEMM calls.
            const void *validated_workspace = nullptr;

            // Flag to track if we own weight memory
            bool owns_weight_memory = false;

            // ROCm device ID for proper cleanup
            int rocm_device_id = 0;

            ~Impl()
            {
                if (ck_kernel_context)
                {
                    rocmQuantGemm_destroyKernelContext(ck_kernel_context, rocm_device_id);
                    ck_kernel_context = nullptr;
                }

                // Only free weight memory if we own it (not from ROCmPackedWeights cache)
                if (owns_weight_memory)
                {
                    if (d_weights_int8_vnni)
                        rocmQuantGemm_freeDevice(d_weights_int8_vnni, rocm_device_id);
                    if (d_weights_int8_rowmajor)
                        rocmQuantGemm_freeDevice(d_weights_int8_rowmajor, rocm_device_id);
                    if (d_scales_B)
                        rocmQuantGemm_freeDevice(d_scales_B, rocm_device_id);
                    if (d_weights_native_payload)
                        rocmQuantGemm_freeDevice(d_weights_native_payload, rocm_device_id);
                    if (d_weights_native_scales)
                        rocmQuantGemm_freeDevice(d_weights_native_scales, rocm_device_id);
#ifdef HAVE_ROCM
                    if (startup_repack_ready_event)
                    {
                        hipEventDestroy(reinterpret_cast<hipEvent_t>(startup_repack_ready_event));
                        startup_repack_ready_event = nullptr;
                    }
                    if (startup_commit_ready_event)
                    {
                        hipEventDestroy(reinterpret_cast<hipEvent_t>(startup_commit_ready_event));
                        startup_commit_ready_event = nullptr;
                    }
                    auto free_pinned = [](void *&ptr)
                    {
                        if (ptr)
                        {
                            hipHostFree(ptr);
                            ptr = nullptr;
                        }
                    };
                    free_pinned(startup_h2d_pinned_scales);
                    free_pinned(startup_h2d_pinned_vnni);
                    free_pinned(startup_h2d_pinned_native_payload);
                    free_pinned(startup_h2d_pinned_native_scales);
#endif
                }
                // Work buffers (including d_B_rowmajor_scratch) are NOT freed -
                // they are owned by workspace
            }
        };

        namespace
        {
            inline bool validatePointerDeviceOrLog(
                const void *ptr,
                int expected_device,
                const char *pointer_name,
                const char *scope)
            {
#if LLAMINAR_ASSERTIONS_ACTIVE
                if (!ptr)
                {
                    LOG_ERROR("[" << scope << "] " << pointer_name << " is null");
                    return false;
                }

                hipPointerAttribute_t attr{};
                hipError_t err = hipPointerGetAttributes(&attr, ptr);
                if (err == hipSuccess)
                {
                    if (attr.device != expected_device)
                    {
                        LOG_ERROR("[" << scope << "] " << pointer_name
                                      << " on wrong device: ptr=" << ptr
                                      << " attr.device=" << attr.device
                                      << " expected=" << expected_device);
                        return false;
                    }
                    return true;
                }

                LOG_WARN("[" << scope << "] hipPointerGetAttributes failed for " << pointer_name
                             << " ptr=" << ptr << ": " << hipGetErrorString(err));
                return true;
#else
                // Release builds: skip HIP runtime validation for performance.
                // Each hipPointerGetAttributes() call costs ~1-5μs, and with ~8 calls
                // per GEMM × ~280 GEMMs per decode token, this adds ~2-10ms per token.
                (void)ptr;
                (void)expected_device;
                (void)pointer_name;
                (void)scope;
                return true;
#endif
            }

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

                if (!impl->d_weights_int8_vnni)
                {
                    LOG_ERROR("[" << log_scope << "] Missing VNNI source weights");
                    return false;
                }

                const bool cache_hit = impl->repack_cache_valid &&
                                       impl->repack_cached_n == n &&
                                       impl->repack_cached_k == k &&
                                       impl->repack_cached_src_vnni == impl->d_weights_int8_vnni &&
                                       impl->repack_cached_dst == impl->d_B_rowmajor_scratch;

                if (cache_hit)
                {
                    LOG_TRACE("[" << log_scope << "] Repack cache hit (N=" << n << ", K=" << k << ")");
                    return true;
                }

                bool repack_ok = false;
                if (impl->d_weights_int8_vnni)
                {
                    repack_ok = rocmGemv_repackVNNI_to_rowmajor(
                        impl->d_weights_int8_vnni,
                        impl->d_B_rowmajor_scratch,
                        n, k,
                        rocm_device_id, gpu_stream);
                }

                if (!repack_ok)
                {
                    LOG_ERROR("[" << log_scope << "] VNNI→row-major repack failed");
                    impl->repack_cache_valid = false;
                    return false;
                }

                impl->repack_cache_valid = true;
                impl->repack_cached_n = n;
                impl->repack_cached_k = k;
                impl->repack_cached_src_vnni = impl->d_weights_int8_vnni;
                impl->repack_cached_dst = impl->d_B_rowmajor_scratch;
                return true;
            }

            template <typename ImplT>
            inline bool waitForStartupRepackIfNeeded(
                ImplT *impl,
                int rocm_device_id,
                void *gpu_stream,
                const char *log_scope)
            {
#ifdef HAVE_ROCM
                if (!impl || !impl->startup_repack_ready_event || !impl->startup_repack_event_pending)
                {
                    if (!impl || !impl->startup_commit_ready_event || !impl->startup_commit_event_pending)
                    {
                        return true;
                    }
                }

                hipStream_t target_stream = reinterpret_cast<hipStream_t>(gpu_stream);
                if (target_stream == nullptr)
                {
                    target_stream = nullptr; // default stream
                }

                void *event_to_wait = nullptr;
                bool *pending_flag = nullptr;
                if (impl->startup_commit_ready_event && impl->startup_commit_event_pending)
                {
                    event_to_wait = impl->startup_commit_ready_event;
                    pending_flag = &impl->startup_commit_event_pending;
                }
                else if (impl->startup_repack_ready_event && impl->startup_repack_event_pending)
                {
                    event_to_wait = impl->startup_repack_ready_event;
                    pending_flag = &impl->startup_repack_event_pending;
                }
                if (!event_to_wait || !pending_flag)
                {
                    return true;
                }

                const hipError_t wait_err = hipStreamWaitEvent(
                    target_stream,
                    reinterpret_cast<hipEvent_t>(event_to_wait),
                    0);
                if (wait_err != hipSuccess)
                {
                    LOG_ERROR("[" << log_scope << "] hipStreamWaitEvent failed: " << hipGetErrorString(wait_err)
                                  << " device=" << rocm_device_id);
                    return false;
                }

                *pending_flag = false;

                auto free_pinned = [](void *&ptr)
                {
                    if (ptr)
                    {
                        const hipError_t err = hipHostFree(ptr);
                        if (err != hipSuccess)
                        {
                            LOG_WARN("[ROCmQuantisedGemmKernel] hipHostFree failed after startup readiness wait: "
                                     << hipGetErrorString(err));
                        }
                        ptr = nullptr;
                    }
                };
                free_pinned(impl->startup_h2d_pinned_scales);
                free_pinned(impl->startup_h2d_pinned_vnni);
                return true;
#else
                (void)impl;
                (void)rocm_device_id;
                (void)gpu_stream;
                (void)log_scope;
                return true;
#endif
            }

            template <typename UploadT>
            inline bool ensureStartupStreamsAndEvents(
                UploadT &upload,
                const ROCmStartupRepackPipelineConfig &cfg,
                const char *log_scope)
            {
#ifdef HAVE_ROCM
                if (!cfg.enabled)
                {
                    return true;
                }

                if (!upload.startup_h2d_stream)
                {
                    hipStream_t stream = nullptr;
                    const hipError_t create_err = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
                    if (create_err != hipSuccess)
                    {
                        LOG_WARN("[" << log_scope << "] Failed to create startup H2D stream: "
                                     << hipGetErrorString(create_err));
                        return false;
                    }
                    upload.startup_h2d_stream = reinterpret_cast<void *>(stream);
                }

                if (!upload.startup_repack_stream)
                {
                    if (cfg.stream_count >= 2)
                    {
                        hipStream_t stream = nullptr;
                        const hipError_t create_err = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
                        if (create_err != hipSuccess)
                        {
                            LOG_WARN("[" << log_scope << "] Failed to create startup repack stream: "
                                         << hipGetErrorString(create_err));
                            return false;
                        }
                        upload.startup_repack_stream = reinterpret_cast<void *>(stream);
                    }
                    else
                    {
                        upload.startup_repack_stream = upload.startup_h2d_stream;
                    }
                }

                if (!upload.startup_commit_stream)
                {
                    if (cfg.stream_count >= 3)
                    {
                        hipStream_t stream = nullptr;
                        const hipError_t create_err = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
                        if (create_err != hipSuccess)
                        {
                            LOG_WARN("[" << log_scope << "] Failed to create startup commit stream: "
                                         << hipGetErrorString(create_err));
                            return false;
                        }
                        upload.startup_commit_stream = reinterpret_cast<void *>(stream);
                    }
                    else
                    {
                        upload.startup_commit_stream = upload.startup_repack_stream;
                    }
                }

                if (!upload.startup_h2d_done_event)
                {
                    hipEvent_t event;
                    const hipError_t create_err = hipEventCreateWithFlags(&event, hipEventDisableTiming);
                    if (create_err != hipSuccess)
                    {
                        LOG_WARN("[" << log_scope << "] Failed to create startup H2D event: "
                                     << hipGetErrorString(create_err));
                        return false;
                    }
                    upload.startup_h2d_done_event = reinterpret_cast<void *>(event);
                }

                if (!upload.startup_repack_ready_event)
                {
                    hipEvent_t event;
                    const hipError_t create_err = hipEventCreateWithFlags(&event, hipEventDisableTiming);
                    if (create_err != hipSuccess)
                    {
                        LOG_WARN("[" << log_scope << "] Failed to create startup repack event: "
                                     << hipGetErrorString(create_err));
                        return false;
                    }
                    upload.startup_repack_ready_event = reinterpret_cast<void *>(event);
                }

                if (!upload.startup_commit_ready_event)
                {
                    hipEvent_t event;
                    const hipError_t create_err = hipEventCreateWithFlags(&event, hipEventDisableTiming);
                    if (create_err != hipSuccess)
                    {
                        LOG_WARN("[" << log_scope << "] Failed to create startup commit event: "
                                     << hipGetErrorString(create_err));
                        return false;
                    }
                    upload.startup_commit_ready_event = reinterpret_cast<void *>(event);
                }

                return true;
#else
                (void)upload;
                (void)cfg;
                (void)log_scope;
                return true;
#endif
            }

            template <typename UploadT>
            inline void freeStartupPinnedStaging(UploadT &upload)
            {
#ifdef HAVE_ROCM
                auto free_if_set = [](void *&ptr)
                {
                    if (ptr)
                    {
                        const hipError_t err = hipHostFree(ptr);
                        if (err != hipSuccess)
                        {
                            LOG_WARN("[ROCmQuantisedGemmKernel] hipHostFree failed for startup staging buffer: "
                                     << hipGetErrorString(err));
                        }
                        ptr = nullptr;
                    }
                };

                free_if_set(upload.startup_h2d_pinned_scales);
                free_if_set(upload.startup_h2d_pinned_vnni);
#else
                (void)upload;
#endif
            }

            inline hipError_t startupMemcpyAsyncOrSync(
                void *dst,
                const void *src,
                size_t bytes,
                bool async_enabled,
                void *stream,
                void **pinned_staging_slot,
                const char *log_scope,
                const char *tensor_name)
            {
#ifdef HAVE_ROCM
                if (pinned_staging_slot)
                {
                    *pinned_staging_slot = nullptr;
                }

                if (async_enabled)
                {
                    const void *copy_src = src;
                    if (bytes > 0 && pinned_staging_slot)
                    {
                        void *pinned_buf = nullptr;
                        const hipError_t alloc_err = hipHostMalloc(&pinned_buf, bytes, hipHostMallocDefault);
                        if (alloc_err == hipSuccess && pinned_buf)
                        {
                            std::memcpy(pinned_buf, src, bytes);
                            *pinned_staging_slot = pinned_buf;
                            copy_src = pinned_buf;
                        }
                        else
                        {
                            LOG_WARN("[" << log_scope << "] hipHostMalloc failed for startup staging (" << tensor_name
                                         << ", bytes=" << bytes << "): " << hipGetErrorString(alloc_err)
                                         << ". Falling back to pageable host source.");
                        }
                    }
                    return hipMemcpyAsync(dst, copy_src, bytes, hipMemcpyHostToDevice, reinterpret_cast<hipStream_t>(stream));
                }
                return hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice);
#else
                (void)dst;
                (void)src;
                (void)bytes;
                (void)async_enabled;
                (void)stream;
                (void)pinned_staging_slot;
                (void)log_scope;
                (void)tensor_name;
                return hipSuccess;
#endif
            }

            template <typename UploadT>
            inline bool launchStartupRowmajorRepackIfEnabled(
                UploadT &upload,
                int N,
                int K,
                int rocm_device_id,
                const ROCmStartupRepackPipelineConfig &cfg,
                const char *log_scope)
            {
#ifdef HAVE_ROCM
                // Architectural direction: startup pipeline is VNNI-only.
                // Do not build CK row-major weights during startup.
                (void)upload;
                (void)N;
                (void)K;
                (void)rocm_device_id;
                (void)cfg;
                (void)log_scope;
                return true;

                if (!cfg.enabled)
                {
                    return true;
                }

                if (!upload.d_int8_data_vnni)
                {
                    return true;
                }

                const size_t rowmajor_elems = static_cast<size_t>(N) * static_cast<size_t>(K);
                if (rowmajor_elems == 0)
                {
                    return true;
                }

                const size_t bytes_needed = rowmajor_elems * sizeof(int8_t);
                const size_t budget_bytes = static_cast<size_t>(cfg.budget_mb) * 1024ull * 1024ull;
                if (bytes_needed > budget_bytes)
                {
                    LOG_DEBUG("[" << log_scope << "] Skip startup GPU repack: row-major buffer "
                                  << (bytes_needed / (1024 * 1024)) << " MB exceeds budget "
                                  << cfg.budget_mb << " MB");
                    return true;
                }

                if (!upload.d_int8_data_rowmajor)
                {
                    if (!rocmQuantGemm_allocInt8(&upload.d_int8_data_rowmajor, rowmajor_elems, rocm_device_id))
                    {
                        LOG_WARN("[" << log_scope << "] Failed to allocate startup row-major repack buffer; falling back to runtime repack");
                        upload.d_int8_data_rowmajor = nullptr;
                        return true;
                    }
                }

                if (!ensureStartupStreamsAndEvents(upload, cfg, log_scope))
                {
                    return true;
                }

                hipStream_t repack_stream = reinterpret_cast<hipStream_t>(upload.startup_repack_stream);
                if (upload.startup_h2d_event_pending && upload.startup_h2d_done_event)
                {
                    const hipError_t wait_err = hipStreamWaitEvent(
                        repack_stream,
                        reinterpret_cast<hipEvent_t>(upload.startup_h2d_done_event),
                        0);
                    if (wait_err != hipSuccess)
                    {
                        LOG_WARN("[" << log_scope << "] Failed waiting for startup H2D event: "
                                     << hipGetErrorString(wait_err)
                                     << "; falling back to runtime repack");
                        return true;
                    }
                    upload.startup_h2d_event_pending = false;
                }

                bool repack_ok = false;
                if (upload.d_int8_data_vnni)
                {
                    repack_ok = rocmGemv_repackVNNI_to_rowmajor(
                        upload.d_int8_data_vnni,
                        upload.d_int8_data_rowmajor,
                        N, K,
                        rocm_device_id,
                        reinterpret_cast<void *>(repack_stream));
                }

                if (!repack_ok)
                {
                    LOG_WARN("[" << log_scope << "] Startup GPU repack kernel launch failed; falling back to runtime repack");
                    return true;
                }

                const hipError_t record_err = hipEventRecord(
                    reinterpret_cast<hipEvent_t>(upload.startup_repack_ready_event),
                    repack_stream);
                if (record_err != hipSuccess)
                {
                    LOG_WARN("[" << log_scope << "] Failed to record startup repack event: "
                                 << hipGetErrorString(record_err)
                                 << "; falling back to runtime repack");
                    return true;
                }

                upload.startup_repack_event_pending = true;
                LOG_DEBUG("[" << log_scope << "] Startup GPU row-major repack launched (event-driven): "
                              << N << "x" << K << " bytes=" << bytes_needed);
                return true;
#else
                (void)upload;
                (void)N;
                (void)K;
                (void)rocm_device_id;
                (void)cfg;
                (void)log_scope;
                return true;
#endif
            }

            template <typename UploadT>
            inline bool launchStartupCommitIfEnabled(
                UploadT &upload,
                const ROCmStartupRepackPipelineConfig &cfg,
                const char *log_scope)
            {
#ifdef HAVE_ROCM
                // Architectural direction: startup pipeline is VNNI-only.
                // With CK startup repack disabled, commit stage is a no-op.
                (void)upload;
                (void)cfg;
                (void)log_scope;
                return true;

                if (!cfg.enabled)
                {
                    return true;
                }

                if (!upload.startup_repack_ready_event || !upload.startup_repack_event_pending)
                {
                    return true;
                }

                if (!upload.startup_commit_stream || !upload.startup_commit_ready_event)
                {
                    return true;
                }

                hipStream_t commit_stream = reinterpret_cast<hipStream_t>(upload.startup_commit_stream);
                const hipError_t wait_err = hipStreamWaitEvent(
                    commit_stream,
                    reinterpret_cast<hipEvent_t>(upload.startup_repack_ready_event),
                    0);
                if (wait_err != hipSuccess)
                {
                    LOG_WARN("[" << log_scope << "] Failed waiting on startup repack event in commit stage: "
                                 << hipGetErrorString(wait_err));
                    return true;
                }

                const hipError_t rec_err = hipEventRecord(
                    reinterpret_cast<hipEvent_t>(upload.startup_commit_ready_event),
                    commit_stream);
                if (rec_err != hipSuccess)
                {
                    LOG_WARN("[" << log_scope << "] Failed recording startup commit event: "
                                 << hipGetErrorString(rec_err));
                    return true;
                }

                upload.startup_commit_event_pending = true;
                return true;
#else
                (void)upload;
                (void)cfg;
                (void)log_scope;
                return true;
#endif
            }
        }

        ROCmStartupRepackPipelineConfig getROCmStartupRepackPipelineConfig()
        {
            const auto &env = debugEnv();
            ROCmStartupRepackPipelineConfig cfg;
            cfg.enabled = env.rocm.startup_gpu_repack;
            cfg.slots = std::max(1, env.rocm.repack_slots);
            cfg.budget_mb = std::max(128, env.rocm.repack_budget_mb);
            cfg.stream_count = std::max(1, env.rocm.repack_streams);
            return cfg;
        }

        // =====================================================================
        // ROCmPackedWeights destructor
        // =====================================================================

        ROCmPackedWeights::~ROCmPackedWeights()
        {
            if (!device_uploads.empty())
            {
                for (auto &[device_id, upload] : device_uploads)
                {
                    if (upload.d_int8_data_vnni)
                        rocmQuantGemm_freeDevice(upload.d_int8_data_vnni, device_id);
                    if (upload.d_int8_data_rowmajor)
                        rocmQuantGemm_freeDevice(upload.d_int8_data_rowmajor, device_id);
                    if (upload.d_scales)
                        rocmQuantGemm_freeDevice(upload.d_scales, device_id);
#ifdef HAVE_ROCM
                    freeStartupPinnedStaging(upload);
                    if (upload.startup_h2d_done_event)
                        hipEventDestroy(reinterpret_cast<hipEvent_t>(upload.startup_h2d_done_event));
                    if (upload.startup_repack_ready_event)
                        hipEventDestroy(reinterpret_cast<hipEvent_t>(upload.startup_repack_ready_event));
                    if (upload.startup_commit_ready_event)
                        hipEventDestroy(reinterpret_cast<hipEvent_t>(upload.startup_commit_ready_event));
                    if (upload.startup_commit_stream && upload.startup_commit_stream != upload.startup_repack_stream)
                        hipStreamDestroy(reinterpret_cast<hipStream_t>(upload.startup_commit_stream));
                    if (upload.startup_repack_stream && upload.startup_repack_stream != upload.startup_h2d_stream)
                        hipStreamDestroy(reinterpret_cast<hipStream_t>(upload.startup_repack_stream));
                    if (upload.startup_h2d_stream)
                        hipStreamDestroy(reinterpret_cast<hipStream_t>(upload.startup_h2d_stream));
#endif
                }
            }
            else
            {
                if (d_int8_data_vnni)
                    rocmQuantGemm_freeDevice(d_int8_data_vnni, rocm_device_id);
                if (d_int8_data_rowmajor)
                    rocmQuantGemm_freeDevice(d_int8_data_rowmajor, rocm_device_id);
                if (d_scales)
                    rocmQuantGemm_freeDevice(d_scales, rocm_device_id);
#ifdef HAVE_ROCM
                if (startup_repack_ready_event)
                    hipEventDestroy(reinterpret_cast<hipEvent_t>(startup_repack_ready_event));
                if (startup_commit_ready_event)
                    hipEventDestroy(reinterpret_cast<hipEvent_t>(startup_commit_ready_event));
#endif
            }
        }

        // =====================================================================
        // packWeightsToROCm: Convert any quantized tensor to INT8 + scales
        // =====================================================================

        bool packNativeVNNI(const TensorBase *tensor, ROCmPackedWeights &out)
        {
            if (!tensor)
                return false;

            const TensorType wt = tensor->native_type();
            if (wt != TensorType::Q4_0 && wt != TensorType::IQ4_NL)
                return false;

            auto *raw_accessor = dynamic_cast<const ITensorGemmTileDataProvider *>(tensor);
            auto *quant_accessor = dynamic_cast<const IINT8Unpackable *>(tensor);
            if (!raw_accessor || !quant_accessor)
                return false;

            const int N = static_cast<int>(tensor->rows());
            const int K = static_cast<int>(tensor->cols());
            if ((K % 32) != 0 || raw_accessor->block_size() != 32)
                return false;

            const int blocks_per_row = K / 32;
            const int payload_bytes = 16; // 32 nibbles = 16 bytes

            out.native_vnni_payload.resize(static_cast<size_t>(blocks_per_row) * N * payload_bytes);
            out.native_vnni_scales.resize(static_cast<size_t>(blocks_per_row) * N);
            out.native_vnni_codebook_id = (wt == TensorType::IQ4_NL) ? 4 : 0;
            out.native_vnni_blocks_per_row = static_cast<uint32_t>(blocks_per_row);

            // Also need per-row scales for the INT8 activation scaling (used by CK expand and activation quant).
            // For native-VNNI we set per-row scale = 1.0 (scale is per-block, not per-row),
            // but we still need the scales array for activation quantization compatibility.
            // We compute per-row max-abs from dequantized weights for CK prefill compat.
            out.scales.resize(N);
            for (int n = 0; n < N; ++n)
            {
                float max_abs = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b)
                {
                    const float scale_b = std::abs(quant_accessor->get_block_scale(
                        static_cast<size_t>(n), static_cast<size_t>(b)));
                    // Conservative: scale * 8 covers full Q4_0 range
                    max_abs = std::max(max_abs, scale_b * 8.0f);
                }
                out.scales[n] = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
            }

            // Interleave payload and scales by N for coalesced GPU access
            for (int n = 0; n < N; ++n)
            {
                for (int b = 0; b < blocks_per_row; ++b)
                {
                    const void *raw_block = raw_accessor->get_raw_block_at(
                        static_cast<size_t>(n), static_cast<size_t>(b));
                    if (!raw_block)
                    {
                        LOG_ERROR("[packNativeVNNI] get_raw_block returned null at n=" << n << " b=" << b);
                        return false;
                    }

                    const uint8_t *payload_src = nullptr;
                    uint16_t scale_fp16 = 0;

                    if (wt == TensorType::Q4_0)
                    {
                        const auto *blk = static_cast<const Q4_0Block *>(raw_block);
                        payload_src = blk->qs;
                        scale_fp16 = blk->d;
                    }
                    else // IQ4_NL
                    {
                        const auto *blk = static_cast<const IQ4_NLBlock *>(raw_block);
                        payload_src = blk->qs;
                        scale_fp16 = blk->d;
                    }

                    // Interleave by N: block (b, n) stored at linear index b*N + n
                    const size_t linear = static_cast<size_t>(b) * N + static_cast<size_t>(n);
                    std::memcpy(out.native_vnni_payload.data() + linear * payload_bytes,
                                payload_src, payload_bytes);
                    out.native_vnni_scales[linear] = scale_fp16;
                }
            }

            LOG_DEBUG("[packNativeVNNI] Built native-VNNI container for " << N << "x" << K
                                                                          << " (codebook=" << static_cast<int>(out.native_vnni_codebook_id) << ")"
                                                                          << " payload=" << (out.native_vnni_payload.size() / 1024) << " KB"
                                                                          << " scales=" << (out.native_vnni_scales.size() * 2 / 1024) << " KB");
            return true;
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

        // =====================================================================
        // Fast Q8_0 direct path: skip FP32 round-trip entirely.
        // Q8_0 blocks are {fp16 scale, int8_t qs[32]}. We extract INT8 values
        // directly and convert per-block scales to per-row scales.
        // Produces both row-major and VNNI layouts in a single fused pass.
        // =====================================================================
        static bool packWeightsToROCm_Q8_0_fast(const TensorBase *tensor, ROCmPackedWeights &out)
        {
            const auto *q8_tensor = dynamic_cast<const Q8_0Tensor *>(tensor);
            if (!q8_tensor)
                return false;

            const int N = static_cast<int>(tensor->rows());
            const int K = static_cast<int>(tensor->cols());
            if (K % 32 != 0)
                return false; // Q8_0 blocks are 32 elements; non-aligned means partial blocks

            const size_t blocks_per_row = static_cast<size_t>(K) / 32;
            const Q8_0Block *blocks = q8_tensor->typed_data();
            if (!blocks)
                return false;

            out.K = K;
            out.N = N;
            out.scales.resize(N);
            out.int8_data.resize(static_cast<size_t>(N) * K);

            const bool build_vnni = (K % 4) == 0;
            const size_t k_groups = build_vnni ? (static_cast<size_t>(K) / 4) : 0;
            if (build_vnni)
                out.int8_data_vnni.resize(k_groups * static_cast<size_t>(N) * 4);
            else
                out.int8_data_vnni.clear();

            // Fused pass: compute per-row scale, rescale INT8 values, emit row-major + VNNI
#pragma omp parallel for schedule(static)
            for (int n = 0; n < N; ++n)
            {
                const Q8_0Block *row_blocks = blocks + static_cast<size_t>(n) * blocks_per_row;

                // Pass 1: find per-row max absolute dequantized value from block metadata.
                // For Q8_0, actual value = block_scale * qs[i], and |qs[i]| <= 127.
                // Per-row max_abs = max over blocks of (block_scale * max(|qs[i]|)).
                // We can simply find max(block_scale * 127) across all blocks as a
                // conservative upper bound, OR scan the actual qs values for tighter fit.
                // For speed, we use the tight path: scan blocks for actual max.
                float max_abs = 0.0f;
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    const float block_scale = fp16_to_fp32(row_blocks[b].d);
                    // Find max |qs| in this block
                    int max_qs = 0;
                    for (int i = 0; i < 32; ++i)
                    {
                        int abs_qs = row_blocks[b].qs[i] < 0 ? -row_blocks[b].qs[i] : row_blocks[b].qs[i];
                        if (abs_qs > max_qs)
                            max_qs = abs_qs;
                    }
                    float val = block_scale * static_cast<float>(max_qs);
                    if (val > max_abs)
                        max_abs = val;
                }

                const float row_scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
                out.scales[n] = row_scale;
                const float inv_row_scale = 1.0f / row_scale;

                // Pass 2: rescale and emit both row-major and VNNI in one pass over blocks
                int8_t *row_dst = out.int8_data.data() + static_cast<size_t>(n) * K;

                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    const Q8_0Block &block = row_blocks[b];
                    const float rescale = fp16_to_fp32(block.d) * inv_row_scale;
                    const size_t col_base = b * 32;

                    // Rescale INT8 values: new_qs = round(old_qs * block_scale / row_scale)
                    for (int i = 0; i < 32; ++i)
                    {
                        float val = static_cast<float>(block.qs[i]) * rescale;
                        int32_t q = static_cast<int32_t>(val + (val >= 0.0f ? 0.5f : -0.5f));
                        q = q < -127 ? -127 : (q > 127 ? 127 : q);
                        row_dst[col_base + i] = static_cast<int8_t>(q);
                    }

                    // Emit VNNI layout inline: [K/4][N][4]
                    // Each block of 32 elements produces 8 VNNI groups of 4
                    if (build_vnni)
                    {
                        for (int g = 0; g < 8; ++g)
                        {
                            const size_t kg = (col_base / 4) + g; // k_group index
                            const size_t dst_offset = (kg * static_cast<size_t>(N) + static_cast<size_t>(n)) * 4;
                            const size_t src_offset = col_base + g * 4;
                            out.int8_data_vnni[dst_offset + 0] = row_dst[src_offset + 0];
                            out.int8_data_vnni[dst_offset + 1] = row_dst[src_offset + 1];
                            out.int8_data_vnni[dst_offset + 2] = row_dst[src_offset + 2];
                            out.int8_data_vnni[dst_offset + 3] = row_dst[src_offset + 3];
                        }
                    }
                }
            }

            // Phase 3 (opt-in): keep only VNNI host buffer and drop row-major host copy.
            if (debugEnv().rocm.pack_vnni_only_host && !out.int8_data_vnni.empty())
            {
                out.int8_data.clear();
                out.int8_data.shrink_to_fit();
            }

            LOG_DEBUG("[packWeightsToROCm] Q8_0 fast-path packed " << N << "x" << K
                                                                   << " weights (direct INT8, no FP32 round-trip)"
                                                                   << (out.int8_data_vnni.empty() ? "" : " + VNNI"));
            return true;
        }

        bool packWeightsToROCm(const TensorBase *tensor, ROCmPackedWeights &out)
        {
            if (!tensor)
            {
                LOG_ERROR("[packWeightsToROCm] Null tensor");
                return false;
            }

            const int N = static_cast<int>(tensor->rows()); // Output features (model weight rows)
            const int K = static_cast<int>(tensor->cols()); // Input features (model weight cols)

            out.K = K;
            out.N = N;

            // Native-VNNI path: Q4_0 and IQ4_NL get lossless native-VNNI packing.
            // This runs in addition to (not instead of) other paths during transition.
            if (packNativeVNNI(tensor, out))
            {
                LOG_DEBUG("[packWeightsToROCm] Packed " << N << "x" << K
                                                        << " weights to native-VNNI");
            }

            // === Fast path: Q8_0 direct INT8 extraction (no FP32 round-trip) ===
            if (tensor->native_type() == TensorType::Q8_0 && packWeightsToROCm_Q8_0_fast(tensor, out))
            {
                return true;
            }

            // === Generic fallback: dequantize to FP32 then requantize to INT8 ===
            const float *h_weights_fp32 = tensor->fp32_data();
            if (!h_weights_fp32)
            {
                LOG_ERROR("[packWeightsToROCm] Failed to get FP32 data from tensor");
                return false;
            }

            out.scales.resize(N);

            // Legacy generic path for all other quant formats:
            out.int8_data.resize(static_cast<size_t>(N) * K);

            // Per-row (per-output-feature) symmetric quantization
#pragma omp parallel for schedule(static)
            for (int n = 0; n < N; ++n)
            {
                // Find max_abs for this output feature (row n of model weights)
                float max_abs = 0.0f;
                for (int k = 0; k < K; ++k)
                {
                    float val = h_weights_fp32[n * K + k];
                    max_abs = std::max(max_abs, std::abs(val));
                }

                // Symmetric quantization: scale = max_abs / 127
                float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
                float inv_scale = 1.0f / scale;
                out.scales[n] = scale;

                for (int k = 0; k < K; ++k)
                {
                    float val = h_weights_fp32[n * K + k];
                    int8_t quantized = static_cast<int8_t>(
                        std::round(std::clamp(val * inv_scale, -127.0f, 127.0f)));
                    out.int8_data[n * K + k] = quantized;
                }
            }

            // Optional VNNI layout: [K/4][N][4] for GEMV experimentation
            out.int8_data_vnni.clear();
            if ((K % 4) == 0)
            {
                const size_t k_groups = static_cast<size_t>(K) / 4;
                out.int8_data_vnni.resize(k_groups * static_cast<size_t>(N) * 4);
#pragma omp parallel for schedule(static)
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

            // Phase 3 (opt-in): keep only VNNI host buffer and drop row-major host copy.
            if (debugEnv().rocm.pack_vnni_only_host)
            {
                if (!out.int8_data_vnni.empty())
                {
                    out.int8_data.clear();
                    out.int8_data.shrink_to_fit();
                    LOG_DEBUG("[packWeightsToROCm] VNNI-only host pack enabled; released row-major host copy for "
                              << N << "x" << K << " weights");
                }
                else
                {
                    LOG_WARN("[packWeightsToROCm] LLAMINAR_ROCM_PACK_VNNI_ONLY=1 requested but VNNI layout unavailable "
                             << "(K=" << K << " not divisible by 4). Falling back to row-major host pack.");
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
              ck_dispatch_mutex_(std::make_unique<std::mutex>()),
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
            impl_->ck_kernel_context = rocmQuantGemm_createKernelContext(rocm_device_id_);
            if (!impl_->ck_kernel_context)
            {
                throw std::runtime_error("[ROCmQuantisedGemmKernel] Failed to create CK kernel context");
            }

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
              ck_dispatch_mutex_(std::make_unique<std::mutex>()),
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
            impl_->ck_kernel_context = rocmQuantGemm_createKernelContext(rocm_device_id_);
            if (!impl_->ck_kernel_context)
            {
                throw std::runtime_error("[ROCmQuantisedGemmKernel] Failed to create CK kernel context");
            }

            LOG_DEBUG("[ROCmQuantisedGemmKernel] Created (pre-packed) for " << N_ << "x" << K_
                                                                            << " INT8 weights on ROCm device " << rocm_device_id_);
        }

        ROCmQuantisedGemmKernel::~ROCmQuantisedGemmKernel() = default;

        ROCmQuantisedGemmKernel::ROCmQuantisedGemmKernel(ROCmQuantisedGemmKernel &&) noexcept = default;
        ROCmQuantisedGemmKernel &ROCmQuantisedGemmKernel::operator=(ROCmQuantisedGemmKernel &&) noexcept = default;

        /**
         * @brief Classify the best prefill route for the current shape and weight format.
         *
         * This is the single source of truth for Phase-0/1 routing decisions. Keeping
         * this logic centralized prevents subtle callsite drift where one entrypoint
         * might treat the same shape or metadata differently than another.
         */
        ROCmQuantisedGemmKernel::PrefillDispatchPath ROCmQuantisedGemmKernel::selectPrefillDispatchPath(int m, int n, int k) const
        {
            (void)n;

            // This helper is intentionally strict and conservative: if anything is
            // unclear, we keep the existing CK path as the safety net.
            if (m <= 1 || k <= 0 || (k % 4) != 0)
            {
                return PrefillDispatchPath::CK_FALLBACK;
            }

            if (!impl_)
            {
                return PrefillDispatchPath::CK_FALLBACK;
            }

            if (impl_->d_weights_int8_vnni != nullptr)
            {
                return PrefillDispatchPath::INT8_VNNI_NATIVE;
            }

            return PrefillDispatchPath::CK_FALLBACK;
        }

        /**
         * @brief Try native M>1 prefill execution for INT8 VNNI or ratio-VNNI formats.
         *
         * The helper follows a strict "native first, CK fallback always available"
         * contract. A false return means the caller should continue with existing CK
         * logic; it does not represent a fatal error by itself.
         */
        bool ROCmQuantisedGemmKernel::tryPrefillNativeGemm(
            const int8_t *d_A_int8,
            float *d_output,
            const float *d_scales_A,
            const float *d_scales_B,
            const float *d_bias,
            int m, int n, int k,
            float alpha, float beta,
            const char *callsite)
        {
            const auto path_label = [&](PrefillDispatchPath p) -> const char *
            {
                switch (p)
                {
                case PrefillDispatchPath::INT8_VNNI_NATIVE:
                    return "int8_vnni_native";
                case PrefillDispatchPath::CK_FALLBACK:
                default:
                    return "ck_fallback";
                }
            };

            const auto record_path_selected = [&](PrefillDispatchPath p)
            {
                if (!WeightLoadingProfiler::isEnabled())
                    return;
                WeightLoadingProfiler::addDetail(
                    std::string("prefill_gemm.path_selected.") + path_label(p),
                    1.0);
            };

            const auto record_fallback_reason = [&](const char *reason)
            {
                if (!WeightLoadingProfiler::isEnabled())
                    return;
                WeightLoadingProfiler::addDetail(
                    std::string("prefill_gemm.fallback_reason.") + reason,
                    1.0);
            };

            const auto record_kernel_ms = [&](PrefillDispatchPath p, double ms)
            {
                if (!WeightLoadingProfiler::isEnabled())
                    return;
                WeightLoadingProfiler::addDetail(
                    std::string("prefill_gemm.kernel_ms.") + path_label(p),
                    ms);
            };

            const auto record_epilogue_ms = [&](PrefillDispatchPath p, double ms)
            {
                if (!WeightLoadingProfiler::isEnabled())
                    return;
                WeightLoadingProfiler::addDetail(
                    std::string("prefill_gemm.epilogue_ms.") + path_label(p),
                    ms);
            };

            // This helper is for prefill only. Decode (M=1) is handled
            // by the dedicated GEMV fast path and should never come through here.
            if (m <= 1)
            {
                return false;
            }

            auto logFallback = [&](const char *reason)
            {
                static std::once_flag unsupported_shape_once;
                static std::once_flag missing_buffers_once;
                static std::once_flag launch_failure_once;

                std::once_flag *selected_once = &launch_failure_once;
                if (std::strcmp(reason, "shape") == 0)
                {
                    selected_once = &unsupported_shape_once;
                }
                else if (std::strcmp(reason, "buffers") == 0)
                {
                    selected_once = &missing_buffers_once;
                }

                record_path_selected(PrefillDispatchPath::CK_FALLBACK);
                record_fallback_reason(reason);

                std::call_once(*selected_once, [&]()
                               { LOG_INFO("[" << callsite << "] Native prefill path falling back to CK (reason="
                                              << reason << ")"); });
            };

            if (!impl_ || !d_A_int8 || !d_output || !d_scales_A || !d_scales_B || !impl_->d_C_int32)
            {
                logFallback("buffers");
                return false;
            }

            if ((k % 4) != 0)
            {
                logFallback("shape");
                return false;
            }

            const PrefillDispatchPath path = selectPrefillDispatchPath(m, n, k);
            record_path_selected(path);
            bool native_ok = false;
            const bool profiling_enabled = debugEnv().profile.enabled;
            std::chrono::high_resolution_clock::time_point kernel_start{};
            std::chrono::high_resolution_clock::time_point kernel_end{};
            if (profiling_enabled)
                kernel_start = std::chrono::high_resolution_clock::now();

            if (path == PrefillDispatchPath::INT8_VNNI_NATIVE)
            {
                LOG_TRACE("[" << callsite << "] Trying INT8 VNNI native prefill (M=" << m
                              << " N=" << n << " K=" << k << ")");
                struct VnniPrefillLaunchConfig
                {
                    bool use_grid_kpar = false;
                    int split_k_slices = 1;
                    int tile_variant = -1; // -1 => HIP-side auto tile select
                    int cpt = 1;
                    int kernel_body_variant = 0;
                    int grid_swizzle_variant = 0;
                    const char *profile = "default";
                    bool use_wide_tile = false; // Wide-tile kernel for extreme-wide shapes (M≤128, N>>K)
                };

                const auto &rocm_env = debugEnv().rocm;
                auto select_vnni_prefill_launch = [&rocm_env](int M, int N, int K) -> VnniPrefillLaunchConfig
                {
                    const double safe_k = static_cast<double>(std::max(K, 1));
                    const double aspect_n_over_k = static_cast<double>(N) / safe_k;
                    const int64_t work = static_cast<int64_t>(M) * static_cast<int64_t>(N) * static_cast<int64_t>(K);
                    const int k_groups = std::max(1, K / 4);

                    constexpr int64_t kSmallWorkThreshold = 1200000000LL; // 0.5B-like vs 3B-like split
                    constexpr int64_t kGridKparMinWork = 300000000LL;

                    const bool is_ffn_up_like = (aspect_n_over_k >= 2.0 && aspect_n_over_k < 16.0);
                    if (is_ffn_up_like && M >= 64 && rocm_env.vnni_prefill_ffn_override)
                    {
                        const bool use_grid_kpar = (rocm_env.vnni_prefill_ffn_override_grid_kpar < 0)
                                                       ? true
                                                       : (rocm_env.vnni_prefill_ffn_override_grid_kpar != 0);
                        const int split_k_slices = (rocm_env.vnni_prefill_ffn_override_splits > 0)
                                                       ? rocm_env.vnni_prefill_ffn_override_splits
                                                       : 2;
                        const int tile_variant = (rocm_env.vnni_prefill_ffn_override_variant >= 0)
                                                     ? rocm_env.vnni_prefill_ffn_override_variant
                                                     : 1;
                        const int cpt = (rocm_env.vnni_prefill_ffn_override_cpt > 0)
                                            ? rocm_env.vnni_prefill_ffn_override_cpt
                                            : 4;
                        const int kernel_body_variant = rocm_env.vnni_prefill_ffn_override_kernel_body;
                        const int grid_swizzle_variant = (rocm_env.vnni_prefill_ffn_override_grid_swizzle >= 0)
                                                             ? rocm_env.vnni_prefill_ffn_override_grid_swizzle
                                                             : 1;

                        if (use_grid_kpar)
                        {
                            return VnniPrefillLaunchConfig{true, split_k_slices, tile_variant, cpt, kernel_body_variant, grid_swizzle_variant, "ffn_override_env_gridkpar"};
                        }
                        return VnniPrefillLaunchConfig{false, split_k_slices, tile_variant, cpt, kernel_body_variant, grid_swizzle_variant, "ffn_override_env_baseline"};
                    }

                    // Shape classes (ratio-first, then work-size split).

                    // ── Wide-tile kernel for ALL prefill shapes ──
                    // V3/V7 kernels use a 2D grid: (ceil(N/N_TILE), ceil(M/128)).
                    // Each workgroup covers up to 128 M-rows × N_TILE N-cols.
                    // Key advantages over grid-kpar:
                    //   - No hipMemsetAsync (no output zeroing needed)
                    //   - No atomicAdd (each output element written by exactly one WG)
                    //   - No extra applyScaling kernel launch
                    // V3/V7 selection (K-heavy vs N-heavy) happens in the dispatch code.
                    {
                        return VnniPrefillLaunchConfig{false, 1, 1, 1, 0, 0, "wide_tile", true};
                    }

                    // ── Legacy grid-kpar paths below are unreachable in default policy ──
                    // Kept for env-var override compatibility (vnni_prefill_ffn_override, etc.)

                    if (aspect_n_over_k >= 32.0)
                    {
                        // LM-head/extreme-wide: fallback grid-kpar (unreachable in default path)
                        return VnniPrefillLaunchConfig{true, 1, 1, 4, 2, 1, "extreme_wide_gridkpar_32x8_cpt4"};
                    }

                    if (aspect_n_over_k >= 2.0)
                    {
                        // FFN up/gate (wide): strategy-lab favored split-k style behavior.
                        if (work < kGridKparMinWork || M < 16 || k_groups < 64)
                        {
                            return VnniPrefillLaunchConfig{false, 1, 1, 1, 0, 0, "wide_guardrail_baseline_32x8_cpt1"};
                        }

                        const int slices = (work <= kSmallWorkThreshold) ? 4 : 6;
                        return VnniPrefillLaunchConfig{true, slices, 1, 4, 2, 1, "wide_gridkpar_32x8_cpt4"};
                    }

                    if (aspect_n_over_k < 0.75)
                    {
                        // FFN down (tall): keep grid-kpar with moderate split count.
                        if (work < kGridKparMinWork || M < 16 || k_groups < 64)
                        {
                            return VnniPrefillLaunchConfig{false, 1, 1, 1, 0, 0, "tall_guardrail_baseline_32x8_cpt1"};
                        }

                        const int slices = (work <= kSmallWorkThreshold) ? 4 : 6;
                        return VnniPrefillLaunchConfig{true, slices, 1, 4, 2, 1, "tall_gridkpar_32x8_cpt4"};
                    }

                    // Attention-ish / near-square.
                    {
                        const int slices = (work <= kSmallWorkThreshold) ? 4 : 6;
                        return VnniPrefillLaunchConfig{true, slices, 1, 4, 2, 1, "balanced_gridkpar_32x8_cpt4"};
                    }
                };

                const VnniPrefillLaunchConfig policy_cfg = select_vnni_prefill_launch(m, n, k);
                if (WeightLoadingProfiler::isEnabled())
                {
                    WeightLoadingProfiler::addDetail(
                        std::string("prefill_gemm.int8_policy.") + policy_cfg.profile,
                        1.0);
                    if (std::strncmp(policy_cfg.profile, "ffn_override_env_", 17) == 0)
                    {
                        WeightLoadingProfiler::addDetail("prefill_gemm.int8_policy.ffn_up_override_applied", 1.0);
                    }
                }

                // ── Fast path: wide-tile (default policy, no env overrides) ──
                // Check wide-tile eligibility BEFORE computing grid-kpar variables.
                // In the default path (no manual overrides), policy always returns
                // use_wide_tile=true, so we skip ~100ns of dead grid-kpar arithmetic.
                const bool has_manual_override =
                    (rocm_env.vnni_prefill_variant >= 0) ||
                    (rocm_env.vnni_prefill_grid_variant >= 0) ||
                    (rocm_env.vnni_prefill_grid_kpar_splits > 0) ||
                    (rocm_env.vnni_prefill_grid_kpar_kb > 0) ||
                    (rocm_env.vnni_prefill_grid_swizzle >= 0) ||
                    (rocm_env.vnni_prefill_cpt != 1) ||
                    rocm_env.vnni_prefill_grid_kpar;

                if (!has_manual_override)
                {
                    static std::once_flag vnni_prefill_policy_once;
                    std::call_once(vnni_prefill_policy_once, [&]()
                                   { LOG_INFO("[" << callsite << "] INT8 prefill policy enabled (ratio+work map): "
                                                  << policy_cfg.profile
                                                  << ", M=" << m << ", N=" << n << ", K=" << k
                                                  << ", N/K=" << std::fixed << std::setprecision(2)
                                                  << (static_cast<double>(n) / static_cast<double>(std::max(k, 1)))
                                                  << ")"); });
                }

                // Wide-tile path: covers all M-rows in one block (extreme-wide shapes)
                if (!has_manual_override && policy_cfg.use_wide_tile)
                {
                    if (rocm_env.wide_tile_v7)
                    {
                        // V7 forced via env var: safe-tile split, branchless interior loop
                        native_ok = rocmQuantGemm_int8_int8_int32_vnni_prefill_wide_tile_v7(
                            d_A_int8,
                            impl_->d_weights_int8_vnni,
                            impl_->d_C_int32,
                            m, n, k,
                            rocm_env.wide_tile_kt,
                            rocm_device_id_, gpu_stream_);
                    }
                    else if (rocm_env.wide_tile_v3)
                    {
                        // V3 forced via env var: LDS double-buffered, N64
                        native_ok = rocmQuantGemm_int8_int8_int32_vnni_prefill_wide_tile_v3(
                            d_A_int8,
                            impl_->d_weights_int8_vnni,
                            impl_->d_C_int32,
                            m, n, k,
                            rocm_env.wide_tile_kt,
                            rocm_device_id_, gpu_stream_);
                    }
                    else if (k >= n)
                    {
                        // Auto V3 for K-heavy shapes (Attention: K==N, FFN_Down: K>>N).
                        // V3's LDS double-buffering hides global load latency in the long
                        // K-loop, giving +2-5% over V1 on these shapes.
                        // N-heavy shapes (FFN_Up/Gate, LM_Head: N > K) use V5 below.
                        native_ok = rocmQuantGemm_int8_int8_int32_vnni_prefill_wide_tile_v3(
                            d_A_int8,
                            impl_->d_weights_int8_vnni,
                            impl_->d_C_int32,
                            m, n, k,
                            16, // KT=16: best throughput for V3 auto-dispatch (perf sweep shows KT=16 wins 10/12 shapes)
                            rocm_device_id_, gpu_stream_);
                    }
                    else
                    {
                        // Auto V7/KT16 for N-heavy shapes (FFN_Up/Gate, LM_Head: N > K).
                        // V7's safe-tile split eliminates boundary branches from the hot
                        // loop — for aligned shapes (M=128, N%128==0, K%32==0) the boundary
                        // loop never executes, producing maximally tight ISA.
                        // Benchmark results (wallclock, 7B FFN_Up M=128 N=18944 K=3584):
                        //   V7/KT16: 7.300ms  V5/KT8: 7.397ms  CK: 7.323ms
                        // V7/KT16 matches or beats CK on all FFN_Up and LM_Head shapes.
                        native_ok = rocmQuantGemm_int8_int8_int32_vnni_prefill_wide_tile_v7(
                            d_A_int8,
                            impl_->d_weights_int8_vnni,
                            impl_->d_C_int32,
                            m, n, k,
                            16, // KT=16: best for V7 (safe-tile split, 2 waves/SIMD)
                            rocm_device_id_, gpu_stream_);
                    }
                    if (!native_ok)
                    {
                        static std::once_flag wide_tile_fallback_once;
                        std::call_once(wide_tile_fallback_once, [&]()
                                       { LOG_WARN("[" << callsite << "] Wide-tile kernel failed; falling back to grid-kpar/baseline"); });
                    }
                }

                // Grid-kpar / baseline variables — only computed when wide-tile
                // didn't fire or didn't succeed, avoiding ~100ns of dead arithmetic
                // on the hot path.
                if (!native_ok)
                {
                    const bool try_grid_kpar = has_manual_override
                                                   ? rocm_env.vnni_prefill_grid_kpar
                                                   : policy_cfg.use_grid_kpar;

                    const int baseline_variant = has_manual_override ? rocm_env.vnni_prefill_variant : policy_cfg.tile_variant;
                    const int grid_variant = has_manual_override ? rocm_env.vnni_prefill_grid_variant : policy_cfg.tile_variant;
                    const int cpt = has_manual_override ? rocm_env.vnni_prefill_cpt : policy_cfg.cpt;
                    const int kernel_body_variant = policy_cfg.kernel_body_variant;
                    const int grid_swizzle_variant = (rocm_env.vnni_prefill_grid_swizzle >= 0)
                                                         ? rocm_env.vnni_prefill_grid_swizzle
                                                         : policy_cfg.grid_swizzle_variant;

                    const int resolved_variant = [&]()
                    {
                        if (try_grid_kpar)
                        {
                            return grid_variant;
                        }
                        return baseline_variant;
                    }();

                    const int tile_x = [&]()
                    {
                        if (resolved_variant == 1)
                            return 32;
                        if (resolved_variant == 2 || resolved_variant == 3)
                            return 8;
                        return 16;
                    }();

                    const int logical_tile_n = std::max(1, tile_x * std::max(1, cpt));
                    const int grid_n = std::max(1, (n + logical_tile_n - 1) / logical_tile_n);
                    const int64_t logical_tiles = static_cast<int64_t>(m) * static_cast<int64_t>(grid_n);
                    const int k_groups = std::max(1, k / 4);
                    const bool low_parallelism_guardrail = (!has_manual_override) && try_grid_kpar &&
                                                           (logical_tiles < 512 || m < 16 || k_groups < 64);

                    const bool use_grid_kpar = low_parallelism_guardrail ? false : try_grid_kpar;

                    if (low_parallelism_guardrail)
                    {
                        if (WeightLoadingProfiler::isEnabled())
                        {
                            WeightLoadingProfiler::addDetail(
                                "prefill_gemm.int8_policy.guardrail_disable_grid_kpar",
                                1.0);
                        }
                    }

                    if (use_grid_kpar)
                    {
                        const int requested_slices_env = rocm_env.vnni_prefill_grid_kpar_splits;
                        const int requested_slices = [&]()
                        {
                            const int kb_override = rocm_env.vnni_prefill_grid_kpar_kb;
                            if (kb_override > 0)
                            {
                                return kb_override;
                            }

                            if (requested_slices_env > 0)
                            {
                                return requested_slices_env;
                            }

                            if (!has_manual_override)
                            {
                                return policy_cfg.split_k_slices;
                            }

                            // Auto heuristic (Phase 4+): target enough K-groups per split
                            // while keeping a minimum level of wavefront availability.
                            const int k_groups = std::max(1, k / 4);
                            constexpr int target_groups_per_slice = 64;
                            constexpr int target_min_waves_per_cu = 8;
                            constexpr int num_cus = 60;

                            int auto_slices = std::max(1, k_groups / target_groups_per_slice);

                            const int cpt_auto = std::max(1, cpt);
                            int tile_x = 16;
                            const int variant = grid_variant;
                            if (variant == 1)
                                tile_x = 32;
                            else if (variant == 2 || variant == 3)
                                tile_x = 8;

                            const int threads_x = std::max(1, tile_x);
                            const int logical_tile_n = std::max(1, threads_x * cpt_auto);
                            const int grid_n = std::max(1, (n + logical_tile_n - 1) / logical_tile_n);
                            const int min_slices_for_waves = std::max(1, (target_min_waves_per_cu * num_cus + grid_n - 1) / grid_n);
                            auto_slices = std::max(auto_slices, min_slices_for_waves);

                            // Prefer fewer atomics for very small M while still enabling
                            // parallel K reduction for wider prefill cases.
                            if (m < 8)
                            {
                                auto_slices = std::min(auto_slices, 2);
                            }
                            else if (m >= 24 && k >= 1024)
                            {
                                auto_slices = std::max(auto_slices, 4);
                            }

                            const int n_tiles = std::max(1, (n + 127) / 128);
                            const int hard_cap = std::min(8, std::max(1, n_tiles * 2));
                            return std::clamp(auto_slices, 1, hard_cap);
                        }();
                        native_ok = rocmQuantGemm_int8_int8_int32_vnni_prefill_grid_kpar(
                            d_A_int8,
                            impl_->d_weights_int8_vnni,
                            impl_->d_C_int32,
                            m, n, k,
                            requested_slices,
                            grid_variant,
                            cpt,
                            kernel_body_variant,
                            grid_swizzle_variant,
                            rocm_device_id_, gpu_stream_);

                        if (!native_ok)
                        {
                            static std::once_flag grid_kpar_fallback_once;
                            std::call_once(grid_kpar_fallback_once, [&]()
                                           { LOG_WARN("[" << callsite << "] INT8 prefill grid-kpar launch failed once; falling back to baseline prefill kernel"); });
                        }
                    }

                    if (!native_ok)
                    {
                        native_ok = rocmQuantGemm_int8_int8_int32_vnni_prefill(
                            d_A_int8,
                            impl_->d_weights_int8_vnni,
                            impl_->d_C_int32,
                            m, n, k,
                            baseline_variant,
                            cpt,
                            rocm_device_id_, gpu_stream_);
                    }
                } // end if (!native_ok) — grid-kpar / baseline fallback scope
                if (profiling_enabled)
                    kernel_end = std::chrono::high_resolution_clock::now();
            }
            else
            {
                logFallback("shape");
                return false;
            }

            if (profiling_enabled)
            {
                record_kernel_ms(
                    path,
                    std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count());
            }

            if (!native_ok)
            {
                logFallback("launch_error");
                return false;
            }

            // Phase-1 scaffold reuses the existing epilogue implementation so alpha,
            // beta, and optional bias semantics exactly match the CK fallback path.
            const float *d_existing = (beta != 0.0f) ? d_output : nullptr;

            std::chrono::high_resolution_clock::time_point epilogue_start{};
            if (profiling_enabled)
                epilogue_start = std::chrono::high_resolution_clock::now();
            if (!rocmQuantGemm_applyScaling(
                    impl_->d_C_int32,
                    d_output,
                    d_scales_A,
                    d_scales_B,
                    m, n,
                    alpha, beta,
                    d_existing,
                    d_bias,
                    rocm_device_id_, gpu_stream_))
            {
                logFallback("launch_error");
                return false;
            }
            if (profiling_enabled)
            {
                const auto epilogue_end = std::chrono::high_resolution_clock::now();
                record_epilogue_ms(
                    path,
                    std::chrono::duration<double, std::milli>(epilogue_end - epilogue_start).count());
            }

            return true;
        }

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
            DeviceWorkspaceManager *workspace,
            int activation_row_offset)
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

            return multiply_tensor(A, C, m, n, k, transpose_B, alpha, beta, bias, mpi_ctx, device_idx, workspace, activation_row_offset);
        }

        bool ROCmQuantisedGemmKernel::multiply_tensor(
            const TensorBase *A, TensorBase *C,
            int m, int n, int k,
            bool transpose_B,
            float alpha, float beta,
            const TensorBase *bias,
            const MPIContext *mpi_ctx,
            int device_idx,
            DeviceWorkspaceManager *workspace,
            int activation_row_offset)
        {
            ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::GEMM, static_cast<hipStream_t>(gpu_stream_));
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
            auto restore_workspace = [&](void *)
            {
                if (ws && ws != saved_workspace)
                {
                    workspace_ = saved_workspace;
                }
            };
            std::unique_ptr<void, decltype(restore_workspace)> workspace_restore_guard(nullptr, restore_workspace);

            if (!A || !C)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Null tensor");
                return false;
            }

            if (!validateROCmWorkspaceBinding(ws, rocm_device_id_, "ROCmQuantisedGemmKernel::multiply_tensor"))
            {
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

            // Apply activation row offset (used by LMHeadStage to process only last token)
            if (activation_row_offset > 0 && d_input != nullptr)
            {
                d_input += static_cast<size_t>(activation_row_offset) * k;
                LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_tensor] Applied activation_row_offset="
                          << activation_row_offset << " (offset " << (static_cast<size_t>(activation_row_offset) * k) << " floats)");
            }

            const bool use_gpu_path = (d_input != nullptr) && (d_output != nullptr);

            // Gate all phase timing behind LLAMINAR_PROFILING to eliminate ~18 chrono
            // calls per GEMM (~630ns × 140 GEMMs/token = ~88μs/token).
            const bool phase_timing = debugEnv().profile.enabled;
            std::chrono::high_resolution_clock::time_point phase_start{};
            std::chrono::high_resolution_clock::time_point phase_end{};

            if (use_gpu_path)
            {
                LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_tensor] Using GPU-to-GPU path (d_input="
                          << d_input << ", d_output=" << d_output << ")");

                if (!validatePointerDeviceOrLog(
                        d_input, rocm_device_id_, "d_input", "ROCmQuantisedGemmKernel::multiply_tensor") ||
                    !validatePointerDeviceOrLog(
                        d_output, rocm_device_id_, "d_output", "ROCmQuantisedGemmKernel::multiply_tensor"))
                {
                    return false;
                }
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

                // Option B: weights are stored as VNNI on device
                int8_t *d_weights_vnni = impl_ ? impl_->d_weights_int8_vnni : nullptr;
                float *d_scales_B = nullptr;
                if (packed_)
                {
                    d_scales_B = packed_->d_scales;
                }
                else if (impl_)
                {
                    d_scales_B = impl_->d_scales_B;
                }

                if ((impl_ && impl_->has_native_vnni) || (d_weights_vnni && d_scales_B))
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

                    // =================================================================
                    // CRITICAL: Detect mapped output memory (same fix as prefill).
                    // Mapped memory is used for logits (CPU sampling needs host access).
                    // GPU scattered writes to mapped memory go over PCIe (~1.6 GB/s)
                    // instead of HBM (~1000 GB/s). For LM_Head M=1 N=152064 → 608KB,
                    // this causes ~2700µs vs ~520µs expected from HBM bandwidth.
                    // Fix: redirect kernel output to HBM workspace, then bulk DMA.
                    // =================================================================
                    const bool gemv_output_is_mapped = C_fp32->isMapped();
                    float *d_gemv_output = d_output;
                    if (gemv_output_is_mapped)
                    {
                        d_gemv_output = impl_->d_C_fp32;
                        static std::once_flag gemv_mapped_once;
                        std::call_once(gemv_mapped_once, [&]()
                                       { LOG_WARN("[multiply_tensor] GEMV MAPPED OUTPUT REDIRECT: M=1 N=" << n
                                                                                                          << " mapped_ptr=" << d_output
                                                                                                          << " -> d_C_fp32=" << impl_->d_C_fp32
                                                                                                          << " (" << (n * 4 / 1024) << " KB)"); });
                    }

                    if (d_weights_vnni &&
                        !validatePointerDeviceOrLog(
                            d_weights_vnni, rocm_device_id_, "d_weights_vnni", "ROCmQuantisedGemmKernel::multiply_tensor"))
                    {
                        return false;
                    }
                    if (!validatePointerDeviceOrLog(
                            d_scales_B, rocm_device_id_, "d_scales_B", "ROCmQuantisedGemmKernel::multiply_tensor") ||
                        !validatePointerDeviceOrLog(
                            impl_->d_A_int8, rocm_device_id_, "workspace::QUANT_A", "ROCmQuantisedGemmKernel::multiply_tensor") ||
                        !validatePointerDeviceOrLog(
                            impl_->d_scales_A, rocm_device_id_, "workspace::SCALES_A", "ROCmQuantisedGemmKernel::multiply_tensor") ||
                        !validatePointerDeviceOrLog(
                            impl_->d_C_int32, rocm_device_id_, "workspace::ACC_INT32", "ROCmQuantisedGemmKernel::multiply_tensor"))
                    {
                        return false;
                    }

                    // =====================================================================
                    // Native-VNNI path: lossless Q4/IQ4 decode, FP16 block scales, scale_A inline
                    // Output is final FP32 — no epilogue kernel needed.
                    // =====================================================================
                    if (impl_->has_native_vnni && alpha == 1.0f && beta == 0.0f)
                    {
                        if (!rocmQuantGemm_quantizeActivations(
                                d_input, impl_->d_A_int8, impl_->d_scales_A, m, k, rocm_device_id_, gpu_stream_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Native-VNNI activation quantization failed");
                            return false;
                        }

                        if (!rocmGemv_native_vnni_fp32(
                                impl_->d_A_int8,
                                impl_->d_weights_native_payload,
                                impl_->d_weights_native_scales,
                                d_gemv_output,
                                impl_->d_scales_A,
                                n, k,
                                impl_->native_vnni_codebook_id,
                                rocm_device_id_, gpu_stream_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Native-VNNI GEMV failed");
                            return false;
                        }

                        if (d_bias)
                        {
                            if (!rocmQuantGemm_biasAdd(d_gemv_output, d_bias, m, n, rocm_device_id_, gpu_stream_))
                            {
                                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Native-VNNI bias add failed");
                                return false;
                            }
                        }
                        // Bulk DMA from HBM workspace to mapped output
                        if (gemv_output_is_mapped)
                        {
                            hipMemcpyAsync(d_output, impl_->d_C_fp32,
                                           static_cast<size_t>(n) * sizeof(float),
                                           hipMemcpyDeviceToDevice,
                                           static_cast<hipStream_t>(gpu_stream_));
                        }
                        return true;
                    }

                    // Use INT8 scatter+reduce GEMV when alpha=1, beta=0 AND VNNI weights available
                    if (alpha == 1.0f && beta == 0.0f && d_weights_vnni)
                    {
                        // Quantize activations (uses shared INT8 buffer)
                        if (!rocmQuantGemm_quantizeActivations(
                                d_input, impl_->d_A_int8, impl_->d_scales_A, m, k, rocm_device_id_, gpu_stream_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] INT8 scatter activation quantization failed");
                            return false;
                        }
                        if (!rocmGemv_int8_scatter_vnni(
                                impl_->d_A_int8, d_weights_vnni, d_gemv_output, impl_->d_scales_A, d_scales_B, d_bias,
                                impl_->d_scatter_partial,
                                n, k, alpha, beta, nullptr,
                                rocm_device_id_, gpu_stream_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] INT8 scatter GEMV failed");
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

                        const float *d_existing = (beta != 0.0f) ? d_gemv_output : nullptr;
                        bool fused_scale = false;
                        bool gemv_ok = false;
                        if (d_weights_vnni)
                        {
                            fused_scale = rocmGemv_int8_int8_fp32_vnni_scaled(
                                impl_->d_A_int8, d_weights_vnni, d_gemv_output,
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
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] INT8 GEMV failed");
                            return false;
                        }

                        if (!fused_scale && !rocmQuantGemm_applyScaling(
                                                impl_->d_C_int32, d_gemv_output, impl_->d_scales_A, d_scales_B,
                                                m, n, alpha, beta, d_existing, d_bias, rocm_device_id_, gpu_stream_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] INT8 GEMV scaling failed");
                            return false;
                        }
                    }

                    // Bulk DMA from HBM workspace to mapped output
                    if (gemv_output_is_mapped)
                    {
                        hipMemcpyAsync(d_output, impl_->d_C_fp32,
                                       static_cast<size_t>(n) * sizeof(float),
                                       hipMemcpyDeviceToDevice,
                                       static_cast<hipStream_t>(gpu_stream_));
                    }
                    return true;
                }
                // Fall through to CK path if weight pointers unavailable
            }

            // =========================================================================
            // NATIVE VNNI PREFILL ROUTE (M>1)
            //
            // We defer actual native dispatch until after activation quantization,
            // because both native prefill formats consume INT8 activations.
            // =========================================================================

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
                    // =================================================================
                    // MAPPED OUTPUT REDIRECT for bias path (same fix as GEMV path).
                    // Without this, biased LM head writes scatter to mapped memory
                    // over PCIe instead of HBM.
                    // =================================================================
                    const bool bias_output_is_mapped = C_fp32->isMapped();
                    float *d_bias_output = d_output;
                    if (bias_output_is_mapped && impl_)
                    {
                        d_bias_output = impl_->d_C_fp32;
                        static std::once_flag bias_mapped_once;
                        std::call_once(bias_mapped_once, [&]()
                                       { LOG_WARN("[multiply_tensor] BIAS PATH MAPPED REDIRECT: M=" << m << " N=" << n
                                                                                                    << " mapped_ptr=" << d_output
                                                                                                    << " -> d_C_fp32=" << impl_->d_C_fp32
                                                                                                    << " (" << (static_cast<size_t>(m) * n * 4 / 1024) << " KB)"); });
                    }

                    LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_tensor] Using CK bias path (d_input="
                              << d_input << ", d_output=" << d_bias_output << ", d_bias=" << d_bias << ")");
                    bool result = multiply_fp32_to_fp32_with_bias(d_input, d_bias_output, d_bias, m, n, k, alpha, beta);

                    // Bulk DMA from HBM workspace to mapped output
                    if (result && bias_output_is_mapped)
                    {
                        hipMemcpyAsync(d_output, impl_->d_C_fp32,
                                       static_cast<size_t>(m) * n * sizeof(float),
                                       hipMemcpyDeviceToDevice,
                                       static_cast<hipStream_t>(gpu_stream_));
                    }

                    if (ws && ws != saved_workspace)
                    {
                        workspace_ = saved_workspace;
                    }
                    return result;
                }
            }

            // Ensure weights are uploaded to device
            if (phase_timing)
                phase_start = std::chrono::high_resolution_clock::now();
            ensureWeightsConverted();
            if (phase_timing)
            {
                phase_end = std::chrono::high_resolution_clock::now();
                double weights_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
                if (weights_ms > 1.0)
                    LOG_TRACE("[ROCmGEMM::PHASES] ensureWeightsConverted: " << weights_ms << "ms");
            }

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

            if (!validatePointerDeviceOrLog(
                    d_scales_B, rocm_device_id_, "d_scales_B", "ROCmQuantisedGemmKernel::multiply_tensor"))
            {
                return false;
            }

            // Validate and populate workspace pointers (includes d_B_rowmajor_scratch)
            if (phase_timing)
                phase_start = std::chrono::high_resolution_clock::now();
            validateWorkspace();
            if (phase_timing)
            {
                phase_end = std::chrono::high_resolution_clock::now();
                double workbuf_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
                if (workbuf_ms > 1.0)
                    LOG_TRACE("[ROCmGEMM::PHASES] validateWorkspace: " << workbuf_ms << "ms");
            }

            // Option B: Repack VNNI→row-major into workspace scratch for CK GEMM.
            // Skip CK row-major materialization entirely when native VNNI prefill will
            // be attempted (M>1 and VNNI/ratio-VNNI weights exist). The native path
            // uses d_weights_int8_vnni directly and never touches d_weights_int8.
            // This avoids ~150ns/GEMM of cache checks + potential GPU alloc/repack on
            // first call for qualifying FFN shapes.
            const bool native_prefill_likely = (m > 1) &&
                                               selectPrefillDispatchPath(m, n, k) != PrefillDispatchPath::CK_FALLBACK;
            int8_t *d_weights_int8 = nullptr;

            const auto should_materialize_persistent_rowmajor = [&](int mm, int nn, int kk) -> bool
            {
                if (mm <= 1 || nn <= 0 || kk <= 0)
                {
                    return false;
                }

                const double n_over_k = static_cast<double>(nn) / static_cast<double>(std::max(1, kk));
                const bool ffn_like = (n_over_k >= 1.5) || (n_over_k <= 0.75);
                if (!ffn_like)
                {
                    return false;
                }

                const int64_t work = static_cast<int64_t>(mm) * static_cast<int64_t>(nn) * static_cast<int64_t>(kk);
                return work >= 300000000LL;
            };

            const auto try_materialize_persistent_rowmajor = [&](int nn, int kk) -> bool
            {
                if (!impl_ || impl_->d_weights_int8_rowmajor)
                {
                    return true;
                }

                if (!should_materialize_persistent_rowmajor(m, nn, kk))
                {
                    return false;
                }

                if (!impl_->d_weights_int8_vnni)
                {
                    return false;
                }

                int8_t **rowmajor_target = nullptr;
                if (packed_)
                {
                    auto upload_it = packed_->device_uploads.find(rocm_device_id_);
                    if (upload_it != packed_->device_uploads.end())
                    {
                        rowmajor_target = &upload_it->second.d_int8_data_rowmajor;
                    }
                }

                if (!rowmajor_target)
                {
                    rowmajor_target = &impl_->d_weights_int8_rowmajor;
                }

                const bool had_existing_rowmajor = (*rowmajor_target != nullptr);
                if (!(*rowmajor_target))
                {
                    const size_t rowmajor_elems = static_cast<size_t>(nn) * static_cast<size_t>(kk);
                    if (!rocmQuantGemm_allocInt8(rowmajor_target, rowmajor_elems, rocm_device_id_))
                    {
                        return false;
                    }
                }
                const bool allocated_now = (!had_existing_rowmajor && *rowmajor_target != nullptr);

                bool repack_ok = false;
                if (impl_->d_weights_int8_vnni)
                {
                    repack_ok = rocmGemv_repackVNNI_to_rowmajor(
                        impl_->d_weights_int8_vnni,
                        *rowmajor_target,
                        nn, kk,
                        rocm_device_id_, gpu_stream_);
                }

                if (!repack_ok)
                {
                    if (allocated_now && rowmajor_target && *rowmajor_target)
                    {
                        rocmQuantGemm_freeDevice(*rowmajor_target, rocm_device_id_);
                        *rowmajor_target = nullptr;
                    }
                    return false;
                }

                impl_->d_weights_int8_rowmajor = *rowmajor_target;
                if (packed_)
                {
                    packed_->d_int8_data_rowmajor = *rowmajor_target;
                }

                if (WeightLoadingProfiler::isEnabled())
                    WeightLoadingProfiler::addDetail("prefill_gemm.ck_rowmajor_persistent.materialized", 1.0);
                return true;
            };

            // CK row-major resolve — skip entirely when native VNNI prefill will be
            // attempted. This avoids per-GEMM overhead of persistent-rowmajor cache
            // checks, waitForStartupRepackIfNeeded, and ensureRepackedWeightsForCK.
            // On first call for qualifying FFN shapes, also avoids a GPU alloc + repack
            // kernel launch that would be immediately wasted.
            const auto resolve_ck_rowmajor_weights = [&]() -> bool
            {
                if (impl_->d_weights_int8_rowmajor)
                {
                    if (!waitForStartupRepackIfNeeded(
                            impl_.get(),
                            rocm_device_id_,
                            gpu_stream_,
                            "ROCmQuantisedGemmKernel::multiply_tensor"))
                    {
                        return false;
                    }
                    d_weights_int8 = impl_->d_weights_int8_rowmajor;
                }

                if (!d_weights_int8)
                {
                    (void)try_materialize_persistent_rowmajor(n, k);
                    if (impl_->d_weights_int8_rowmajor)
                    {
                        d_weights_int8 = impl_->d_weights_int8_rowmajor;
                        if (WeightLoadingProfiler::isEnabled())
                            WeightLoadingProfiler::addDetail("prefill_gemm.ck_rowmajor_persistent.reuse", 1.0);
                    }
                }

                if (!d_weights_int8 && impl_->d_B_rowmajor_scratch)
                {
                    if (phase_timing)
                        phase_start = std::chrono::high_resolution_clock::now();
                    if (!ensureRepackedWeightsForCK(
                            impl_.get(), n, k, rocm_device_id_, gpu_stream_,
                            "ROCmQuantisedGemmKernel::multiply_tensor"))
                    {
                        return false;
                    }
                    d_weights_int8 = impl_->d_B_rowmajor_scratch;
                    if (phase_timing)
                    {
                        phase_end = std::chrono::high_resolution_clock::now();
                        double repack_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
                        LOG_TRACE("[ROCmGEMM::PHASES] VNNI→rowmajor repack: " << repack_ms << "ms");
                    }
                }

                if (!d_weights_int8)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] No VNNI weights for CK GEMM repack");
                    return false;
                }

                if (!validatePointerDeviceOrLog(
                        d_weights_int8, rocm_device_id_, "workspace::ROCM_B_REPACK", "ROCmQuantisedGemmKernel::multiply_tensor"))
                {
                    return false;
                }
                return true;
            };

            // Only resolve CK weights eagerly when native prefill won't be tried.
            // Otherwise we defer to after the native attempt (if it falls through).
            if (!native_prefill_likely)
            {
                if (!resolve_ck_rowmajor_weights())
                    return false;
            }

            LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_tensor] Weight ptrs: int8(scratch)=" << (void *)d_weights_int8 << " scales=" << (void *)d_scales_B);

            int8_t *d_A_int8 = impl_->d_A_int8;
            float *d_scales_A = impl_->d_scales_A;
            int32_t *d_C_int32 = impl_->d_C_int32;

            if (!validatePointerDeviceOrLog(
                    d_A_int8, rocm_device_id_, "workspace::QUANT_A", "ROCmQuantisedGemmKernel::multiply_tensor") ||
                !validatePointerDeviceOrLog(
                    d_scales_A, rocm_device_id_, "workspace::SCALES_A", "ROCmQuantisedGemmKernel::multiply_tensor") ||
                !validatePointerDeviceOrLog(
                    d_C_int32, rocm_device_id_, "workspace::ACC_INT32", "ROCmQuantisedGemmKernel::multiply_tensor"))
            {
                return false;
            }

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

                // Workspace is required - d_A_fp32 buffer is pre-allocated
                d_A_fp32_src = impl_->d_A_fp32;

                LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_tensor] Using workspace d_A_fp32=" << (void *)d_A_fp32_src);

                if (phase_timing)
                    phase_start = std::chrono::high_resolution_clock::now();
                const float *h_A_src = A_fp32->data() + static_cast<size_t>(activation_row_offset) * k;
                if (!rocmQuantGemm_copyHostToDevice(d_A_fp32_src, h_A_src, a_fp32_size, rocm_device_id_))
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Failed to copy activations to device");
                    return false;
                }
                if (phase_timing)
                {
                    phase_end = std::chrono::high_resolution_clock::now();
                    double h2d_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
                    if (h2d_ms > 1.0)
                        LOG_TRACE("[ROCmGEMM::PHASES] H2D copy (A_fp32): " << h2d_ms << "ms");
                }
                LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_tensor] Copied activations to device");
            }

            LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_tensor] Now quantizing activations");

            // Quantize activations FP32 → INT8
            if (phase_timing)
                phase_start = std::chrono::high_resolution_clock::now();
            if (!rocmQuantGemm_quantizeActivations(d_A_fp32_src, d_A_int8, d_scales_A, m, k, rocm_device_id_, gpu_stream_))
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Failed to quantize activations");
                return false;
            }
            if (phase_timing)
            {
                phase_end = std::chrono::high_resolution_clock::now();
                double quant_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
                if (quant_ms > 1.0)
                    LOG_TRACE("[ROCmGEMM::PHASES] quantizeActivations: " << quant_ms << "ms");
            }

            // Attempt native dual-format prefill path first (INT8 VNNI or ratio-VNNI).
            // If this returns false, we continue to the CK path below.
            if (m > 1)
            {
                // CRITICAL OPTIMIZATION: Detect mapped (host-pinned) output memory.
                // Mapped memory is used for logits (CPU sampling needs host-side access).
                // GPU writes to mapped memory go over PCIe (~1.6 GB/s) instead of HBM
                // (~1024 GB/s). For LM_Head (M=596, N=152064 → 362MB), this causes
                // 226ms of PCIe writes vs. 0.35ms HBM + 30ms streaming D2H = 30ms.
                //
                // Fix: redirect scaling output to device workspace (d_C_fp32), then
                // do a bulk hipMemcpyAsync to the mapped output.
                const bool output_is_mapped = use_gpu_path && C_fp32->isMapped();
                float *d_prefill_output = (use_gpu_path && !output_is_mapped) ? d_output : impl_->d_C_fp32;

                // One-time diagnostic for mapped output redirect
                if (output_is_mapped && n > 100000)
                {
                    static std::once_flag mapped_redirect_once;
                    std::call_once(mapped_redirect_once, [&]()
                                   { LOG_WARN("[multiply_tensor] MAPPED OUTPUT REDIRECT: LM_Head M=" << m << " N=" << n
                                                                                                     << " gpu_ptr=" << d_output
                                                                                                     << " host_ptr=" << static_cast<void *>(C_fp32->mutable_data())
                                                                                                     << " -> d_C_fp32=" << impl_->d_C_fp32
                                                                                                     << " (" << (static_cast<size_t>(m) * n * 4 / (1024 * 1024)) << " MB)"
                                                                                                     << " ptrs_same=" << (static_cast<void *>(C_fp32->mutable_data()) == static_cast<void *>(d_output) ? "YES" : "NO")); });
                }

                const float *d_prefill_bias = nullptr;
                if (bias && use_gpu_path)
                {
                    d_prefill_bias = bias->isBARBacked()
                                         ? static_cast<const float *>(bias->rocm_data_ptr())
                                         : static_cast<const float *>(bias->gpu_data_ptr());
                }

                if (tryPrefillNativeGemm(
                        d_A_int8,
                        d_prefill_output,
                        d_scales_A,
                        d_scales_B,
                        d_prefill_bias,
                        m, n, k,
                        alpha, beta,
                        "ROCmQuantisedGemmKernel::multiply_tensor"))
                {
                    // Copy from device workspace to final destination when scaling
                    // was redirected away from d_output.
                    if (!use_gpu_path || output_is_mapped)
                    {
                        const size_t prefill_c_size = static_cast<size_t>(m) * n;
                        if (output_is_mapped)
                        {
                            // Mapped output: copy from device workspace to mapped host.
                            // First sync the stream to ensure scaling kernel is done,
                            // then use synchronous hipMemcpy which should use the SDMA engine.
                            float *host_dst = C_fp32->mutable_data();
                            const size_t copy_bytes = prefill_c_size * sizeof(float);

                            hipStreamSynchronize(static_cast<hipStream_t>(gpu_stream_));

                            auto copy_start = std::chrono::high_resolution_clock::now();
                            hipError_t err = hipMemcpy(
                                host_dst, impl_->d_C_fp32,
                                copy_bytes,
                                hipMemcpyDeviceToHost);
                            auto copy_end = std::chrono::high_resolution_clock::now();

                            static std::once_flag copy_timing_once;
                            std::call_once(copy_timing_once, [&]()
                                           {
                                double copy_ms = std::chrono::duration<double, std::milli>(copy_end - copy_start).count();
                                double bw_gbs = (copy_bytes / 1e9) / (copy_ms / 1e3);
                                LOG_WARN("[multiply_tensor] D2H COPY TIMING: " << (copy_bytes / (1024*1024)) << " MB in "
                                         << copy_ms << " ms = " << bw_gbs << " GB/s"); });

                            if (err != hipSuccess)
                            {
                                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Mapped output D2H copy failed: "
                                          << hipGetErrorString(err));
                                return false;
                            }
                        }
                        else
                        {
                            // CPU path: copy from workspace to host tensor
                            if (!rocmQuantGemm_copyDeviceToHost(C_fp32->mutable_data(), impl_->d_C_fp32, prefill_c_size, rocm_device_id_))
                            {
                                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Native prefill path succeeded but D2H copy failed");
                                return false;
                            }
                        }
                    }
                    return true;
                }

                // Native prefill path failed — lazily resolve CK row-major weights
                // now (was deferred because we expected native to succeed).
                if (native_prefill_likely && !d_weights_int8)
                {
                    if (!resolve_ck_rowmajor_weights())
                        return false;
                }
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

            // Determine output buffer based on GPU vs CPU path
            // CRITICAL: For mapped (host-pinned) output, redirect to device workspace
            // to avoid catastrophic PCIe write bandwidth (~1.6 GB/s vs 1024 GB/s HBM).
            const bool ck_output_is_mapped = use_gpu_path && C_fp32->isMapped();
            const size_t c_fp32_size = static_cast<size_t>(m) * n;
            float *d_C_fp32_dst = nullptr;

            if (use_gpu_path && !ck_output_is_mapped)
            {
                // GPU path: Write directly to output tensor's GPU buffer
                d_C_fp32_dst = d_output;
                LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_tensor] Using GPU output directly: " << d_C_fp32_dst);
            }
            else
            {
                // CPU path or mapped output: Use temp buffer from workspace
                d_C_fp32_dst = impl_->d_C_fp32;
            }

            if (!validatePointerDeviceOrLog(
                    d_C_fp32_dst, rocm_device_id_, "d_C_fp32_dst", "ROCmQuantisedGemmKernel::multiply_tensor"))
            {
                return false;
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
            const bool serialize_rocm_gemm = debugEnv().validation.serialize_rocm_gemm_stage;

            auto runCKDispatch = [&](auto &&dispatch_fn, const char *op_name) -> bool
            {
                if (!serialize_rocm_gemm)
                {
                    return dispatch_fn();
                }

                std::lock_guard<std::mutex> lock(*ck_dispatch_mutex_);
                bool dispatch_success = dispatch_fn();
                if (!dispatch_success)
                {
                    return false;
                }

#ifdef HAVE_ROCM
                const hipError_t sync_err = hipStreamSynchronize(reinterpret_cast<hipStream_t>(gpu_stream_));
                if (sync_err != hipSuccess)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] " << op_name
                                                                            << " stream sync failed during serialized CK dispatch: "
                                                                            << hipGetErrorString(sync_err));
                    return false;
                }
#endif
                return true;
            };

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
                success = runCKDispatch(
                    [&]()
                    {
                        return rocmQuantGemm_executeTwoKernel_padded_cached(
                            d_A_int8, d_weights_int8, d_C_fp32_dst,
                            d_scales_A, d_scales_B,
                            impl_->d_CK_int32,
                            impl_->d_A_padded, impl_->d_scale_A_padded, impl_->d_E_padded,
                            m, padded_m, n, k, rocm_device_id_, gpu_stream_, impl_->ck_kernel_context);
                    },
                    "executeTwoKernel_padded_cached");
                auto gemm_end = std::chrono::high_resolution_clock::now();
                double gemm_ms = std::chrono::duration<double, std::milli>(gemm_end - gemm_start).count();
                LOG_TRACE("[ROCmGEMM::TIMING] executeTwoKernel_padded_cached M=" << m << " N=" << n << " K=" << k << " took " << gemm_ms << "ms");
            }
            else
            {
                // Direct execution: no padding needed
                auto gemm_start = std::chrono::high_resolution_clock::now();
                success = runCKDispatch(
                    [&]()
                    {
                        return rocmQuantGemm_executeTwoKernel_cached(
                            d_A_int8, d_weights_int8, d_C_fp32_dst,
                            d_scales_A, d_scales_B,
                            impl_->d_CK_int32,
                            m, n, k, rocm_device_id_, gpu_stream_, impl_->ck_kernel_context);
                    },
                    "executeTwoKernel_cached");
                auto gemm_end = std::chrono::high_resolution_clock::now();
                double gemm_ms = std::chrono::duration<double, std::milli>(gemm_end - gemm_start).count();
                LOG_TRACE("[ROCmGEMM::TIMING] executeTwoKernel_cached M=" << m << " N=" << n << " K=" << k << " took " << gemm_ms << "ms");
            }

            if (!success)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] All GEMM paths failed");
                return false;
            }

            // Copy result back to host (only if not in GPU path, or mapped output)
            if (!use_gpu_path || ck_output_is_mapped)
            {
                phase_start = std::chrono::high_resolution_clock::now();
                if (ck_output_is_mapped)
                {
                    // Mapped output: streaming D2H from device workspace to mapped host.
                    // Use HOST pointer, not device-visible mapped pointer.
                    float *host_dst = C_fp32->mutable_data();
                    hipError_t err = hipMemcpyAsync(
                        host_dst, d_C_fp32_dst,
                        c_fp32_size * sizeof(float),
                        hipMemcpyDeviceToHost,
                        static_cast<hipStream_t>(gpu_stream_));
                    if (err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] CK mapped output D2H copy failed: "
                                  << hipGetErrorString(err));
                        return false;
                    }
                    LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_tensor] CK path: copied to mapped output");
                }
                else
                {
                    LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_tensor] CPU path: copying to host"
                              << " d_C_fp32_dst=" << d_C_fp32_dst
                              << " h_dst=" << (void *)C_fp32->mutable_data()
                              << " c_fp32_size=" << c_fp32_size);
                    if (!rocmQuantGemm_copyDeviceToHost(C_fp32->mutable_data(), d_C_fp32_dst, c_fp32_size, rocm_device_id_))
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Failed to copy output to host");
                        return false;
                    }
                }
                phase_end = std::chrono::high_resolution_clock::now();
                double d2h_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
                if (d2h_ms > 1.0)
                    LOG_TRACE("[ROCmGEMM::PHASES] D2H copy (C_fp32): " << d2h_ms << "ms");
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

            // Get weight device pointers — Option B: VNNI and/or ratio-VNNI, repack to scratch
            validateWorkspace();
            if ((!impl_->d_weights_int8_vnni && !impl_->d_weights_int8_rowmajor) ||
                (!impl_->d_B_rowmajor_scratch && !impl_->d_weights_int8_rowmajor))
            {
                static std::once_flag timed_path_unavailable_once;
                std::call_once(timed_path_unavailable_once, []()
                               { LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_tensor_timed] Timed path unavailable for this weight layout (missing VNNI/ratio buffers or repack scratch); caller may fallback to multiply_tensor()"); });
                return false;
            }
            float *d_scales_B = impl_->d_scales_B;
            if (!d_scales_B)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor_timed] Weights not uploaded");
                return false;
            }
            int8_t *d_weights_int8 = nullptr;
            if (impl_->d_weights_int8_rowmajor)
            {
                if (!waitForStartupRepackIfNeeded(
                        impl_.get(),
                        rocm_device_id_,
                        gpu_stream_,
                        "ROCmQuantisedGemmKernel::multiply_tensor_timed"))
                {
                    return false;
                }
                d_weights_int8 = impl_->d_weights_int8_rowmajor;
            }
            else
            {
                if (!ensureRepackedWeightsForCK(
                        impl_.get(), n, k, rocm_device_id_, gpu_stream_,
                        "ROCmQuantisedGemmKernel::multiply_tensor_timed"))
                {
                    return false;
                }
                d_weights_int8 = impl_->d_B_rowmajor_scratch;
            }

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
                    m, padded_m, n, k, rocm_device_id_, gpu_stream_, impl_->ck_kernel_context);
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
                    kernel_time_ms, gpu_stream_, impl_->ck_kernel_context);
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
            ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::GEMM, static_cast<hipStream_t>(gpu_stream_));

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

            // Step 3: Quantize activations ONCE (shared across all projections)
            // All M=1 paths now use INT8 scatter kernels which require pre-quantized
            // activations. For M>1, CK GEMM also uses pre-quantized activations.
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

            // Step 4: Execute projections using the SHARED quantized activations
            // For M=1, VNNI projections are batched into a single kernel dispatch.
            // Native-VNNI (Q4/IQ4) and M>1 projections are dispatched individually.
            bool all_success = true;
            const bool serialize_rocm_gemm = debugEnv().validation.serialize_rocm_gemm_stage;

            // Batched INT8 scatter GEMV collection arrays (for M=1 VNNI projections)
            const int8_t *batch_B_ptrs[8] = {};
            float *batch_C_ptrs[8] = {};
            const float *batch_scales_B_ptrs[8] = {};
            const float *batch_bias_ptrs[8] = {};
            int batch_N[8] = {};
            int batch_count = 0;

            auto runCKDispatch = [&](auto &&dispatch_fn, const char *op_name) -> bool
            {
                if (!serialize_rocm_gemm)
                {
                    return dispatch_fn();
                }

                std::lock_guard<std::mutex> lock(*ck_dispatch_mutex_);
                bool dispatch_success = dispatch_fn();
                if (!dispatch_success)
                {
                    return false;
                }

#ifdef HAVE_ROCM
                const hipError_t sync_err = hipStreamSynchronize(reinterpret_cast<hipStream_t>(gpu_stream_));
                if (sync_err != hipSuccess)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] " << op_name
                                                                                  << " stream sync failed during serialized CK dispatch: "
                                                                                  << hipGetErrorString(sync_err));
                    return false;
                }
#endif
                return true;
            };
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
                    if (!rocm_kernel->impl_->has_native_vnni && !d_vnni)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i << " VNNI weights not on device");
                        all_success = false;
                        break;
                    }

                    // Native-VNNI path: lossless Q4/IQ4 decode with FP16 block scales
                    if (rocm_kernel->impl_->has_native_vnni)
                    {
                        // Activations are always pre-quantized above (Step 3)
                        LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                                 << " NATIVE-VNNI GEMV M=1 N=" << n << " K=" << k
                                                                                                 << (d_bias ? " +bias" : ""));

                        if (!rocmGemv_native_vnni_fp32(
                                impl_->d_A_int8,
                                rocm_kernel->impl_->d_weights_native_payload,
                                rocm_kernel->impl_->d_weights_native_scales,
                                d_output,
                                impl_->d_scales_A,
                                n, k,
                                rocm_kernel->impl_->native_vnni_codebook_id,
                                rocm_device_id_, gpu_stream_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Native-VNNI GEMV failed for projection " << i);
                            all_success = false;
                            break;
                        }

                        if (d_bias)
                        {
                            if (!rocmQuantGemm_biasAdd(d_output, d_bias, m, n, rocm_device_id_, gpu_stream_))
                            {
                                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Native-VNNI bias add failed for projection " << i);
                                all_success = false;
                                break;
                            }
                        }

                        LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i << " native-VNNI GEMV complete");
                        continue;
                    }

                    // BATCHED INT8 SCATTER PATH: collect pointers for single batched dispatch
                    // All VNNI projections with the same K are batched into 2 kernel launches
                    // (1 batched scatter + 1 batched reduce) instead of 2N individual launches.
                    if (d_vnni && batch_count < 8)
                    {
                        LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                                 << " (" << (proj.name ? proj.name : "unnamed")
                                                                                                 << ") BATCHED SCATTER collect: N=" << n << " K=" << k
                                                                                                 << (d_bias ? " +bias" : ""));
                        batch_B_ptrs[batch_count] = d_vnni;
                        batch_C_ptrs[batch_count] = d_output;
                        batch_scales_B_ptrs[batch_count] = d_scales_B;
                        batch_bias_ptrs[batch_count] = d_bias;
                        batch_N[batch_count] = n;
                        batch_count++;
                        continue; // Dispatch happens after the loop
                    }

                    // Fallback: single-projection INT8 scatter (batch overflow or no VNNI)
                    if (d_vnni)
                    {
                        LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                                 << " SINGLE INT8 SCATTER M=1 N=" << n << " K=" << k);
                        if (!rocmGemv_int8_scatter_vnni(
                                impl_->d_A_int8, d_vnni, d_output,
                                impl_->d_scales_A, d_scales_B, d_bias,
                                rocm_kernel->impl_->d_scatter_partial,
                                n, k, 1.0f, 0.0f, nullptr,
                                rocm_device_id_, gpu_stream_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] INT8 scatter GEMV failed for projection " << i);
                            all_success = false;
                            break;
                        }
                    }
                    else
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                                 << " has no VNNI weights and is not native-VNNI");
                        all_success = false;
                        break;
                    }

                    LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i << " GEMV complete");
                    continue; // Skip CK path below
                }

                // =========================================================================
                // PREFILL PATH: M>1 CK GEMM
                // =========================================================================

                if (rocm_kernel->tryPrefillNativeGemm(
                        impl_->d_A_int8,
                        d_output,
                        impl_->d_scales_A,
                        d_scales_B,
                        d_bias,
                        m, n, k,
                        1.0f, 0.0f,
                        "ROCmQuantisedGemmKernel::multiply_fused_tensor"))
                {
                    LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                             << " native prefill complete");
                    continue;
                }

                // Repack VNNI→row-major into this kernel's workspace scratch for CK
                int8_t *d_weights_int8 = nullptr;
                int8_t *d_vnni = rocm_kernel->impl_ ? rocm_kernel->impl_->d_weights_int8_vnni : nullptr;
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
                        success = runCKDispatch(
                            [&]()
                            {
                                return rocmQuantGemm_executeTwoKernel_padded_cached(
                                    impl_->d_A_int8,
                                    d_weights_int8,
                                    d_output,
                                    impl_->d_scales_A,
                                    d_scales_B,
                                    rocm_kernel->impl_->d_CK_int32,
                                    impl_->d_A_padded, impl_->d_scale_A_padded, impl_->d_E_padded,
                                    m, padded_m, n, k, rocm_device_id_, gpu_stream_, rocm_kernel->impl_->ck_kernel_context);
                            },
                            "executeTwoKernel_padded_cached");

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
                        success = runCKDispatch(
                            [&]()
                            {
                                return rocmQuantGemm_executeNoScale(
                                    impl_->d_A_int8,
                                    d_weights_int8,
                                    rocm_kernel->impl_->d_CK_int32,
                                    m, n, k, rocm_device_id_, gpu_stream_, rocm_kernel->impl_->ck_kernel_context);
                            },
                            "executeNoScale");
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
                        success = runCKDispatch(
                            [&]()
                            {
                                return rocmQuantGemm_executeTwoKernel_padded(
                                    impl_->d_A_int8,
                                    d_weights_int8,
                                    d_output,
                                    impl_->d_scales_A,
                                    d_scales_B,
                                    rocm_kernel->impl_->d_CK_int32,
                                    m, padded_m, n, k, rocm_device_id_, gpu_stream_, rocm_kernel->impl_->ck_kernel_context);
                            },
                            "executeTwoKernel_padded");
                    }
                    else
                    {
                        success = runCKDispatch(
                            [&]()
                            {
                                return rocmQuantGemm_executeTwoKernel_cached(
                                    impl_->d_A_int8,
                                    d_weights_int8,
                                    d_output,
                                    impl_->d_scales_A,
                                    d_scales_B,
                                    rocm_kernel->impl_->d_CK_int32,
                                    m, n, k, rocm_device_id_, gpu_stream_, rocm_kernel->impl_->ck_kernel_context);
                            },
                            "executeTwoKernel_cached");
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

            // =========================================================================
            // BATCHED INT8 SCATTER DISPATCH: process all collected VNNI projections
            // in a single 2-kernel launch (1 batched scatter + 1 batched reduce).
            //
            // This replaces N individual kernel launches with 2, saving launch overhead
            // and improving GPU utilization (especially for small-N projections like K/V
            // which individually have only 4 n-blocks but batched with Q get 36).
            // =========================================================================
            if (batch_count > 0 && all_success)
            {
                LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Dispatching BATCHED INT8 SCATTER: "
                          << batch_count << " projections, K=" << k);

                if (!rocmGemv_int8_scatter_batched_vnni(
                        impl_->d_A_int8,          // Pre-quantized INT8 activations
                        impl_->d_scales_A,        // Activation scale
                        impl_->d_scatter_partial, // Shared partial buffer (workspace-managed)
                        batch_count,
                        batch_B_ptrs,
                        batch_C_ptrs,
                        batch_scales_B_ptrs,
                        batch_bias_ptrs,
                        batch_N,
                        k,
                        1.0f, 0.0f,
                        rocm_device_id_, gpu_stream_))
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Batched INT8 scatter dispatch failed");
                    all_success = false;
                }
                else
                {
                    LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Batched INT8 scatter complete: "
                              << batch_count << " projections in 2 kernel launches");
                }
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

            // Scatter+reduce partial buffer: KB_MAX × N × sizeof(float)
            // KB_MAX=64 is the maximum k-blocks the scatter dispatch can produce.
            // The workspace manager takes max across all kernel instances, so
            // the largest N (LM Head: 152064) determines the actual allocation.
            // Size: 64 × 152064 × 4 ≈ 37 MB — reused across layers.
            constexpr int SCATTER_KB_MAX = 64;
            size_t scatter_partial_bytes = static_cast<size_t>(SCATTER_KB_MAX) * n * sizeof(float);
            reqs.buffers.push_back({GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL, scatter_partial_bytes, 256, true});

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
            // Invalidate workspace validation cache so next validateWorkspace() re-populates
            if (impl_)
            {
                impl_->validated_workspace = nullptr;
            }
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
                std::lock_guard<std::mutex> lock(packed_->upload_mutex);
                auto upload_it = packed_->device_uploads.find(rocm_device_id_);

                if (upload_it == packed_->device_uploads.end())
                {
                    ROCmPackedWeights::DeviceUpload upload;
                    const auto startup_repack_cfg = getROCmStartupRepackPipelineConfig();
                    // Architectural direction: startup preparation is VNNI-only.
                    // Disable CK startup row-major repack orchestration (async streams/events)
                    // while preserving runtime fallback behavior.
                    const bool startup_repack_enabled = false;

                    // Option B: Upload ONLY VNNI layout + scales to device.
                    // Row-major weights are repacked on-demand from VNNI into a
                    // shared workspace scratch buffer for CK GEMM prefill.

                    // Upload scales
                    rocmQuantGemm_setDevice(rocm_device_id_);

                    bool async_upload_enabled = false;
                    if (startup_repack_enabled)
                    {
                        async_upload_enabled = ensureStartupStreamsAndEvents(
                            upload,
                            startup_repack_cfg,
                            "ROCmQuantisedGemmKernel::ensureWeightsConverted");
                    }

                    auto cleanup_startup_async_resources = [&upload]()
                    {
#ifdef HAVE_ROCM
                        upload.startup_h2d_event_pending = false;
                        upload.startup_repack_event_pending = false;
                        upload.startup_commit_event_pending = false;
                        freeStartupPinnedStaging(upload);
                        if (upload.startup_h2d_done_event)
                        {
                            hipEventDestroy(reinterpret_cast<hipEvent_t>(upload.startup_h2d_done_event));
                            upload.startup_h2d_done_event = nullptr;
                        }
                        if (upload.startup_repack_ready_event)
                        {
                            hipEventDestroy(reinterpret_cast<hipEvent_t>(upload.startup_repack_ready_event));
                            upload.startup_repack_ready_event = nullptr;
                        }
                        if (upload.startup_commit_ready_event)
                        {
                            hipEventDestroy(reinterpret_cast<hipEvent_t>(upload.startup_commit_ready_event));
                            upload.startup_commit_ready_event = nullptr;
                        }
                        if (upload.startup_commit_stream && upload.startup_commit_stream != upload.startup_repack_stream)
                        {
                            hipStreamDestroy(reinterpret_cast<hipStream_t>(upload.startup_commit_stream));
                        }
                        if (upload.startup_repack_stream && upload.startup_repack_stream != upload.startup_h2d_stream)
                        {
                            hipStreamDestroy(reinterpret_cast<hipStream_t>(upload.startup_repack_stream));
                        }
                        if (upload.startup_h2d_stream)
                        {
                            hipStreamDestroy(reinterpret_cast<hipStream_t>(upload.startup_h2d_stream));
                        }
                        upload.startup_h2d_stream = nullptr;
                        upload.startup_repack_stream = nullptr;
                        upload.startup_commit_stream = nullptr;
#endif
                    };

                    {
                        ScopedWeightLoadDetailTimer h2d_timer("weights.gemm_pack.h2d_stage");

                        if (!rocmQuantGemm_allocFloat(&upload.d_scales,
                                                      static_cast<size_t>(packed_->N),
                                                      rocm_device_id_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to alloc scales");
                            cleanup_startup_async_resources();
                            return;
                        }

                        hipError_t err = startupMemcpyAsyncOrSync(
                            upload.d_scales,
                            packed_->scales.data(),
                            static_cast<size_t>(packed_->N) * sizeof(float),
                            async_upload_enabled,
                            upload.startup_h2d_stream,
                            &upload.startup_h2d_pinned_scales,
                            "ROCmQuantisedGemmKernel::ensureWeightsConverted",
                            "scales");
                        if (err != hipSuccess)
                        {
                            rocmQuantGemm_freeDevice(upload.d_scales, rocm_device_id_);
                            upload.d_scales = nullptr;
                            cleanup_startup_async_resources();
                            LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to upload scales: "
                                      << hipGetErrorString(err));
                            return;
                        }

                        // Upload VNNI layout (sole weight copy on device)
                        if (!packed_->int8_data_vnni.empty())
                        {
                            if (!rocmQuantGemm_allocInt8(&upload.d_int8_data_vnni,
                                                         packed_->int8_data_vnni.size(),
                                                         rocm_device_id_))
                            {
                                rocmQuantGemm_freeDevice(upload.d_scales, rocm_device_id_);
                                upload.d_scales = nullptr;
                                cleanup_startup_async_resources();
                                LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to alloc VNNI weights");
                                return;
                            }

                            err = startupMemcpyAsyncOrSync(
                                upload.d_int8_data_vnni,
                                packed_->int8_data_vnni.data(),
                                packed_->int8_data_vnni.size() * sizeof(int8_t),
                                async_upload_enabled,
                                upload.startup_h2d_stream,
                                &upload.startup_h2d_pinned_vnni,
                                "ROCmQuantisedGemmKernel::ensureWeightsConverted",
                                "vnni");
                            if (err != hipSuccess)
                            {
                                rocmQuantGemm_freeDevice(upload.d_int8_data_vnni, rocm_device_id_);
                                upload.d_int8_data_vnni = nullptr;
                                rocmQuantGemm_freeDevice(upload.d_scales, rocm_device_id_);
                                upload.d_scales = nullptr;
                                cleanup_startup_async_resources();
                                LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to upload VNNI weights: "
                                          << hipGetErrorString(err));
                                return;
                            }

                            LOG_DEBUG("[ROCmQuantisedGemmKernel] Uploaded VNNI weights (Option B, sole device copy): "
                                      << (packed_->int8_data_vnni.size() / 1024) << " KB");
                        }

                        // Optional persistent row-major upload for CK prefill reuse.
                        // When host row-major pack is available, keep a device copy so
                        // repeated prefill calls can bypass VNNI/ratio->row-major repack.
                        if (!packed_->int8_data.empty())
                        {
                            if (!rocmQuantGemm_allocInt8(&upload.d_int8_data_rowmajor,
                                                         packed_->int8_data.size(),
                                                         rocm_device_id_))
                            {
                                if (upload.d_int8_data_vnni)
                                    rocmQuantGemm_freeDevice(upload.d_int8_data_vnni, rocm_device_id_);
                                if (upload.d_scales)
                                    rocmQuantGemm_freeDevice(upload.d_scales, rocm_device_id_);
                                upload.d_int8_data_vnni = nullptr;
                                upload.d_scales = nullptr;
                                cleanup_startup_async_resources();
                                LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to alloc row-major weights");
                                return;
                            }

                            err = startupMemcpyAsyncOrSync(
                                upload.d_int8_data_rowmajor,
                                packed_->int8_data.data(),
                                packed_->int8_data.size() * sizeof(int8_t),
                                async_upload_enabled,
                                upload.startup_h2d_stream,
                                nullptr,
                                "ROCmQuantisedGemmKernel::ensureWeightsConverted",
                                "rowmajor");
                            if (err != hipSuccess)
                            {
                                rocmQuantGemm_freeDevice(upload.d_int8_data_rowmajor, rocm_device_id_);
                                upload.d_int8_data_rowmajor = nullptr;
                                if (upload.d_int8_data_vnni)
                                    rocmQuantGemm_freeDevice(upload.d_int8_data_vnni, rocm_device_id_);
                                if (upload.d_scales)
                                    rocmQuantGemm_freeDevice(upload.d_scales, rocm_device_id_);
                                upload.d_int8_data_vnni = nullptr;
                                upload.d_scales = nullptr;
                                cleanup_startup_async_resources();
                                LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to upload row-major weights: "
                                          << hipGetErrorString(err));
                                return;
                            }

                            LOG_DEBUG("[ROCmQuantisedGemmKernel] Uploaded persistent row-major weights for CK prefill reuse: "
                                      << (packed_->int8_data.size() / 1024) << " KB");
                        }

                        // Upload native-VNNI payload + scales (lossless Q4_0/IQ4_NL)
                        if (!packed_->native_vnni_payload.empty() && !packed_->native_vnni_scales.empty())
                        {
                            // Allocate payload buffer
                            if (!rocmQuantGemm_allocInt8(reinterpret_cast<int8_t **>(&upload.d_native_vnni_payload),
                                                         packed_->native_vnni_payload.size(),
                                                         rocm_device_id_))
                            {
                                LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to alloc native-VNNI payload");
                                // Don't return — fall through, GEMV will use INT8-VNNI fallback
                            }
                            else
                            {
                                err = startupMemcpyAsyncOrSync(
                                    upload.d_native_vnni_payload,
                                    packed_->native_vnni_payload.data(),
                                    packed_->native_vnni_payload.size() * sizeof(uint8_t),
                                    async_upload_enabled,
                                    upload.startup_h2d_stream,
                                    &upload.startup_h2d_pinned_native_payload,
                                    "ROCmQuantisedGemmKernel::ensureWeightsConverted",
                                    "native_vnni_payload");
                                if (err != hipSuccess)
                                {
                                    rocmQuantGemm_freeDevice(upload.d_native_vnni_payload, rocm_device_id_);
                                    upload.d_native_vnni_payload = nullptr;
                                    LOG_WARN("[ROCmQuantisedGemmKernel] Failed to upload native-VNNI payload: "
                                             << hipGetErrorString(err));
                                }
                            }

                            // Allocate FP16 scales buffer
                            if (upload.d_native_vnni_payload)
                            {
                                const size_t scales_bytes = packed_->native_vnni_scales.size() * sizeof(uint16_t);
                                void *d_scales_tmp = nullptr;
#ifdef HAVE_ROCM
                                hipError_t alloc_err = hipMalloc(&d_scales_tmp, scales_bytes);
                                if (alloc_err != hipSuccess)
                                {
                                    LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to alloc native-VNNI scales: "
                                              << hipGetErrorString(alloc_err));
                                    d_scales_tmp = nullptr;
                                }
#endif
                                upload.d_native_vnni_scales = d_scales_tmp;
                                if (d_scales_tmp)
                                {
                                    err = startupMemcpyAsyncOrSync(
                                        d_scales_tmp,
                                        packed_->native_vnni_scales.data(),
                                        scales_bytes,
                                        async_upload_enabled,
                                        upload.startup_h2d_stream,
                                        &upload.startup_h2d_pinned_native_scales,
                                        "ROCmQuantisedGemmKernel::ensureWeightsConverted",
                                        "native_vnni_scales");
                                    if (err != hipSuccess)
                                    {
#ifdef HAVE_ROCM
                                        hipFree(d_scales_tmp);
#endif
                                        upload.d_native_vnni_scales = nullptr;
                                        LOG_WARN("[ROCmQuantisedGemmKernel] Failed to upload native-VNNI scales: "
                                                 << hipGetErrorString(err));
                                    }
                                }
                            }

                            if (upload.d_native_vnni_payload && upload.d_native_vnni_scales)
                            {
                                LOG_DEBUG("[ROCmQuantisedGemmKernel] Uploaded native-VNNI payload="
                                          << (packed_->native_vnni_payload.size() / 1024) << " KB scales="
                                          << (packed_->native_vnni_scales.size() * 2 / 1024) << " KB");
                            }
                        }

                        if (packed_->int8_data_vnni.empty())
                        {
                            LOG_WARN("[ROCmQuantisedGemmKernel] No VNNI layout available. "
                                     "ROCm GEMV/CK prefill paths may not work.");
                        }

                        if (async_upload_enabled && upload.startup_h2d_done_event)
                        {
#ifdef HAVE_ROCM
                            const hipError_t h2d_record_err = hipEventRecord(
                                reinterpret_cast<hipEvent_t>(upload.startup_h2d_done_event),
                                reinterpret_cast<hipStream_t>(upload.startup_h2d_stream));
                            if (h2d_record_err == hipSuccess)
                            {
                                upload.startup_h2d_event_pending = true;
                            }
                            else
                            {
                                LOG_WARN("[ROCmQuantisedGemmKernel] Failed to record startup H2D event: "
                                         << hipGetErrorString(h2d_record_err)
                                         << "; falling back to synchronous copy semantics");
                                upload.startup_h2d_event_pending = false;
                            }
#endif
                        }
                    }

                    {
                        ScopedWeightLoadDetailTimer repack_timer("weights.gemm_pack.gpu_repack_stage");
                        if (!launchStartupRowmajorRepackIfEnabled(
                                upload,
                                packed_->N,
                                packed_->K,
                                rocm_device_id_,
                                startup_repack_cfg,
                                "ROCmQuantisedGemmKernel::ensureWeightsConverted"))
                        {
                            LOG_WARN("[ROCmQuantisedGemmKernel] Startup GPU row-major repack launch failed; continuing with runtime repack fallback");
                        }
                    }

                    {
                        ScopedWeightLoadDetailTimer commit_timer("weights.gemm_pack.commit_stage");
                        if (!launchStartupCommitIfEnabled(
                                upload,
                                startup_repack_cfg,
                                "ROCmQuantisedGemmKernel::ensureWeightsConverted"))
                        {
                            LOG_WARN("[ROCmQuantisedGemmKernel] Startup GPU commit stage launch failed; continuing with repack readiness event fallback");
                        }
                    }

                    {
                        ScopedWeightLoadDetailTimer commit_timer("weights.gemm_pack.commit_publish");
                        auto emplaced = packed_->device_uploads.emplace(rocm_device_id_, upload);
                        upload_it = emplaced.first;
                    }

                    LOG_DEBUG("[ROCmQuantisedGemmKernel] Uploaded pre-packed weights to ROCm:" << rocm_device_id_
                                                                                               << " " << packed_->N << "x" << packed_->K
                                                                                               << " vnni=" << (packed_->int8_data_vnni.size() / 1024) << " KB"
                                                                                               << " rowmajor=" << (packed_->int8_data.size() / 1024) << " KB");
                }

                {
                    ScopedWeightLoadDetailTimer commit_timer("weights.gemm_pack.commit_publish");
                    auto &upload = upload_it->second;
                    packed_->d_int8_data_vnni = upload.d_int8_data_vnni;
                    packed_->d_int8_data_rowmajor = upload.d_int8_data_rowmajor;
                    packed_->d_scales = upload.d_scales;
                    packed_->d_native_vnni_payload = upload.d_native_vnni_payload;
                    packed_->d_native_vnni_scales = upload.d_native_vnni_scales;
                    packed_->startup_repack_ready_event = upload.startup_repack_ready_event;
                    packed_->startup_repack_event_pending = upload.startup_repack_event_pending;
                    packed_->startup_commit_ready_event = upload.startup_commit_ready_event;
                    packed_->startup_commit_event_pending = upload.startup_commit_event_pending;
                    packed_->uploaded = true;
                    packed_->rocm_device_id = rocm_device_id_;

                    // Point impl_ to packed_ device pointers
                    impl_->d_weights_int8_vnni = upload.d_int8_data_vnni;
                    impl_->d_weights_int8_rowmajor = upload.d_int8_data_rowmajor;
                    impl_->d_scales_B = upload.d_scales;
                    impl_->d_weights_native_payload = upload.d_native_vnni_payload;
                    impl_->d_weights_native_scales = upload.d_native_vnni_scales;
                    impl_->native_vnni_codebook_id = packed_->native_vnni_codebook_id;
                    impl_->native_vnni_blocks_per_row = packed_->native_vnni_blocks_per_row;
                    impl_->has_native_vnni = (upload.d_native_vnni_payload != nullptr && upload.d_native_vnni_scales != nullptr);
                    impl_->startup_repack_ready_event = upload.startup_repack_ready_event;
                    impl_->startup_repack_event_pending = upload.startup_repack_event_pending;
                    impl_->startup_commit_ready_event = upload.startup_commit_ready_event;
                    impl_->startup_commit_event_pending = upload.startup_commit_event_pending;
                    impl_->startup_h2d_pinned_scales = upload.startup_h2d_pinned_scales;
                    impl_->startup_h2d_pinned_vnni = upload.startup_h2d_pinned_vnni;
                    impl_->startup_h2d_pinned_native_payload = upload.startup_h2d_pinned_native_payload;
                    impl_->startup_h2d_pinned_native_scales = upload.startup_h2d_pinned_native_scales;
                }
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

            impl_->owns_weight_memory = true; // We now own the device memory
            weights_converted_ = true;

            LOG_DEBUG("[ROCmQuantisedGemmKernel] Converted and uploaded weights: "
                      << N_ << "x" << K_);
        }

        void ROCmQuantisedGemmKernel::validateWorkspace() const
        {
            // =========================================================================
            // Fast path: skip re-validation if same workspace was already validated.
            // During steady-state inference, the workspace pointer never changes,
            // so this avoids ~20 unordered_map<string> lookups per GEMM call.
            // (~2260 lookups/token × ~100ns each = ~0.2ms/token savings)
            // =========================================================================
            if (impl_->validated_workspace == workspace_)
            {
                return;
            }

            // =========================================================================
            // Workspace is REQUIRED - no fallback allocation
            // =========================================================================
            if (!validateROCmWorkspaceBinding(workspace_, rocm_device_id_, "ROCmQuantisedGemmKernel"))
            {
                throw std::runtime_error(
                    "[ROCmQuantisedGemmKernel] Invalid workspace binding");
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
            if (!workspace_->hasBuffer(GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL))
            {
                throw std::runtime_error(
                    "[ROCmQuantisedGemmKernel] Workspace missing required buffer: ROCM_SCATTER_PARTIAL");
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
            impl_->d_scatter_partial = static_cast<float *>(workspace_->getBuffer(GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL));

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

            // Mark workspace as validated to skip re-validation on next call
            impl_->validated_workspace = workspace_;

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
                float *d_s = packed_ ? packed_->d_scales : (impl_ ? impl_->d_scales_B : nullptr);
                if ((impl_ && impl_->has_native_vnni) || (d_vnni && d_s))
                {
                    LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32] GEMV fast path M=1");

                    if (!impl_)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32] INT8 GEMV requires impl_ buffers");
                        return false;
                    }

                    validateWorkspace();

                    // Native-VNNI path: lossless Q4/IQ4 decode, FP16 block scales, scale_A inline
                    if (impl_->has_native_vnni && alpha == 1.0f && beta == 0.0f)
                    {
                        if (!rocmQuantGemm_quantizeActivations(
                                d_A, impl_->d_A_int8, impl_->d_scales_A, m, k, rocm_device_id_, gpu_stream_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32] Native-VNNI activation quantization failed");
                            return false;
                        }

                        if (!rocmGemv_native_vnni_fp32(
                                impl_->d_A_int8,
                                impl_->d_weights_native_payload,
                                impl_->d_weights_native_scales,
                                d_C,
                                impl_->d_scales_A,
                                n, k,
                                impl_->native_vnni_codebook_id,
                                rocm_device_id_, gpu_stream_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32] Native-VNNI GEMV failed");
                            return false;
                        }
                        return true;
                    }

                    // Use INT8 scatter+reduce GEMV when alpha=1, beta=0 AND VNNI weights available
                    if (alpha == 1.0f && beta == 0.0f && d_vnni)
                    {
                        // Quantize activations (uses shared INT8 buffer)
                        if (!rocmQuantGemm_quantizeActivations(
                                d_A, impl_->d_A_int8, impl_->d_scales_A, m, k, rocm_device_id_, gpu_stream_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32] INT8 scatter activation quantization failed");
                            return false;
                        }
                        if (!rocmGemv_int8_scatter_vnni(
                                impl_->d_A_int8, d_vnni, d_C, impl_->d_scales_A, d_s, nullptr,
                                impl_->d_scatter_partial,
                                n, k, alpha, beta, nullptr,
                                rocm_device_id_, gpu_stream_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32] INT8 scatter GEMV failed");
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
                    if (d_vnni)
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
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32] INT8 GEMV (VNNI) failed");
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
            if (!impl_->d_weights_int8_vnni ||
                !impl_->d_B_rowmajor_scratch)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32] No VNNI weights or repack scratch for CK GEMM");
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

            if (m > 1 && tryPrefillNativeGemm(
                             impl_->d_A_int8,
                             d_C,
                             impl_->d_scales_A,
                             impl_->d_scales_B,
                             nullptr,
                             m, n, k,
                             alpha, beta,
                             "ROCmQuantisedGemmKernel::multiply_fp32_to_fp32"))
            {
                return true;
            }

            // Step 2: Execute CK INT8 GEMM (two-kernel approach: GEMM + scaling separately)
            // The two-kernel cached version takes a pre-allocated INT32 buffer
            if (!rocmQuantGemm_executeTwoKernel_cached(
                    impl_->d_A_int8, impl_->d_B_rowmajor_scratch,
                    d_C, // Output FP32
                    impl_->d_scales_A, impl_->d_scales_B,
                    impl_->d_C_int32, // Pre-allocated INT32 accumulator
                    m, n, k, rocm_device_id_, gpu_stream_, impl_->ck_kernel_context))
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
                float *d_s = packed_ ? packed_->d_scales : (impl_ ? impl_->d_scales_B : nullptr);
                if ((impl_ && impl_->has_native_vnni) || (d_vnni && d_s))
                {
                    LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] GEMV fast path M=1 +bias");

                    if (!impl_)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] INT8 GEMV requires impl_ buffers");
                        return false;
                    }

                    validateWorkspace();

                    // Native-VNNI path: lossless Q4/IQ4 decode, FP16 block scales, scale_A inline
                    if (impl_->has_native_vnni && alpha == 1.0f && beta == 0.0f)
                    {
                        if (!rocmQuantGemm_quantizeActivations(
                                d_A, impl_->d_A_int8, impl_->d_scales_A, m, k, rocm_device_id_, gpu_stream_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] Native-VNNI activation quantization failed");
                            return false;
                        }

                        if (!rocmGemv_native_vnni_fp32(
                                impl_->d_A_int8,
                                impl_->d_weights_native_payload,
                                impl_->d_weights_native_scales,
                                d_C,
                                impl_->d_scales_A,
                                n, k,
                                impl_->native_vnni_codebook_id,
                                rocm_device_id_, gpu_stream_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] Native-VNNI GEMV failed");
                            return false;
                        }

                        if (d_bias)
                        {
                            if (!rocmQuantGemm_biasAdd(d_C, d_bias, m, n, rocm_device_id_, gpu_stream_))
                            {
                                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] Native-VNNI bias add failed");
                                return false;
                            }
                        }
                        return true;
                    }

                    // Use INT8 scatter+reduce GEMV when alpha=1, beta=0 AND VNNI weights available
                    // Bias handled inside scatter reduce kernel
                    if (alpha == 1.0f && beta == 0.0f && d_vnni)
                    {
                        // Quantize activations (uses shared INT8 buffer)
                        if (!rocmQuantGemm_quantizeActivations(
                                d_A, impl_->d_A_int8, impl_->d_scales_A, m, k, rocm_device_id_, gpu_stream_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] INT8 scatter activation quantization failed");
                            return false;
                        }
                        return rocmGemv_int8_scatter_vnni(
                            impl_->d_A_int8, d_vnni, d_C, impl_->d_scales_A, d_s, d_bias,
                            impl_->d_scatter_partial,
                            n, k, alpha, beta, nullptr,
                            rocm_device_id_, gpu_stream_);
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
                    if (d_vnni)
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
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] INT8 GEMV (VNNI) failed");
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
            if (!impl_->d_weights_int8_vnni ||
                !impl_->d_B_rowmajor_scratch)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] No VNNI weights or repack scratch for CK GEMM");
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

            if (m > 1 && tryPrefillNativeGemm(
                             impl_->d_A_int8,
                             d_C,
                             impl_->d_scales_A,
                             impl_->d_scales_B,
                             d_bias,
                             m, n, k,
                             alpha, beta,
                             "ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias"))
            {
                return true;
            }

            // Step 2: Execute CK INT8 GEMM → INT32 (no scaling)
            if (!rocmQuantGemm_executeNoScale(
                    impl_->d_A_int8, impl_->d_B_rowmajor_scratch, impl_->d_C_int32,
                    m, n, k, rocm_device_id_, gpu_stream_, impl_->ck_kernel_context))
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
