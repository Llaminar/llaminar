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
 *   2. Quantize activations FP32→INT8 on GPU (rocmQuantGemm_quantizeActivationsBlockwise)
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
#include "../ROCmKernelBase.h"
#include "../ROCmWeightPacker.h"     // packWeightsToROCm, packNativeVNNI
#include "backends/ComputeBackend.h" // DeviceManager
#include "backends/DeviceId.h"       // DeviceId
#include "tensors/Tensors.h"         // Q8_1Tensor, FP32Tensor, etc.
#include "tensors/BlockStructures.h" // Q8_1Block
#include "tensors/FP16Utils.h"
#include "tensors/IQQuantTables.h" // iq3s_grid, iq2xs_grid, ksigns_iq2xs etc.
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
#include "utils/PerfStatsCollector.h"
#include "utils/WeightLoadingProfiler.h"

#include <stdexcept>
#include <vector>
#include <cmath>
#include <algorithm>
#include <atomic>
#include <string>
#include <array>
#include <mutex>
#include <set>
#include <unordered_map>

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif
#include <cstring>
#include <chrono>

namespace llaminar2
{
    namespace rocm
    {
        namespace
        {
            std::atomic<uint32_t> g_rocm_gemm_workspace_slice_counter{1};
        }

        // =====================================================================
        // Forward declarations for HIP implementation
        // =====================================================================
        //
        // Functions are defined across three .hip files:
        //   - ROCmQuantisedGemmKernel.hip      : Common kernels + utilities (quantization, scaling, memory)
        //   - ROCmQuantisedGemmKernel_CK.hip   : CK INT8 GEMM dispatch + kernel context
        //   - ROCmQuantisedGemmKernel_INT8_VNNI.hip : VNNI prefill/decode kernels
        //

        // These functions are implemented in ROCmQuantisedGemmKernel_*.hip
        extern "C"
        {
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

            // Blockwise quantize FP32 activations to INT8 with per-block scales (common .hip)
            bool rocmQuantGemm_quantizeActivationsBlockwise(
                const float *d_A_fp32,     // [M x K]
                int8_t *d_A_int8,          // [M x K] output
                float *d_scales_blockwise, // [M x blocks_per_row] output
                int M, int K,
                int rocm_device_id, void *stream,
                int block_size = 32);

            // Allocate INT8 buffer (common .hip)
            bool rocmQuantGemm_allocInt8(int8_t **d_ptr, size_t count, int rocm_device_id);

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

            bool rocmQuantGemm_applyFp32Epilogue(
                float *d_output,
                const float *d_bias,
                int M, int N,
                float alpha,
                int rocm_device_id, void *stream);

            bool rocmQuantGemm_applyFp32CombineEpilogue(
                const float *d_computed,
                float *d_output,
                const float *d_existing,
                const float *d_bias,
                int M, int N,
                float alpha, float beta,
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

            bool rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
                const int8_t *d_A_int8,
                const int8_t *d_B_int8_vnni,
                float *d_C_fp32,
                const float *d_scales_A_blockwise,
                const float *d_scale_B,
                int N, int K,
                float alpha,
                float beta,
                const float *d_C_existing,
                const float *d_bias,
                int device_id, void *stream);

            // Get current activation block size for blockwise quantization (32, 64, or 128)
            int rocmGemv_int8_vnni_get_act_block_k();

            // Native-VNNI GEMM: lossless decode with FP16 per-block scales (M>1 prefill).
            // Supports all native-VNNI formats via codebook_id.
            // Output is FP32 with scale_A applied inline — no separate epilogue needed.
            // Halved HBM bandwidth vs INT8 GEMM (4.5 bpw vs 8 bpw for Q4_0/IQ4_NL).
            // Defined in ROCmQuantisedGemmKernel_native_VNNI.hip
            bool rocmGemm_native_vnni_fp32(
                const int8_t *d_A_int8,
                const uint8_t *d_payload,
                const void *d_block_scales, // __half* (FP16 d)
                const void *d_block_mins,   // __half* (FP16 m, nullable)
                const void *d_block_emins,  // uint32_t* (packed FP16 emins, Q2_K only, else NULL)
                float *d_output,
                const float *d_scales_A,
                const float *d_scales_A_blockwise, // [M × blocks_per_row] per-block scales (nullptr = row-wise)
                int M, int N, int K,
                uint8_t codebook_id,
                int device_id, void *stream);

            // Native-VNNI GEMV: lossless decode with FP16 per-block scales.
            // Supports all ≤6-bit quantized formats via codebook_id:
            //   Q-quants: Q4_0, Q4_1, Q5_0, Q5_1
            //   K-quants: Q2_K, Q3_K, Q4_K (via Q4_1 codebook), Q5_K (via Q5_1 codebook), Q6_K
            //   IQ-quants: IQ4_NL, IQ4_XS (via IQ4_NL codebook), IQ3_S, IQ3_XXS,
            //              IQ2_S, IQ2_XS, IQ2_XXS, IQ1_S, IQ1_M
            // Q8_0/Q8_1/Q8_K use the INT8-VNNI path instead (already 1 byte/element).
            // Output is FP32 with scale_A applied inline — no epilogue kernel needed.
            // Uses scatter+reduce pattern: KB=1 → direct, KB>1 → scatter+reduce.
            bool rocmGemv_native_vnni_fp32(
                const int8_t *d_A_int8,
                const uint8_t *d_payload,
                const void *d_block_scales, // __half* (FP16 d)
                const void *d_block_mins,   // __half* (FP16 m), NULL for symmetric formats
                const void *d_block_emins,  // uint32_t* (packed FP16 emins, Q2_K only, else NULL)
                float *d_C_fp32,
                const float *d_scale_A,
                float *d_partial_fp32, // [KB_MAX × N] partial buffer (nullable when KB=1)
                int N, int K,
                uint8_t codebook_id,
                int device_id, void *stream,
                const float *d_scale_A_blockwise = nullptr);

            bool rocmGemv_native_vnni_q4_m2_fp32(
                const int8_t *d_A_int8,
                const uint8_t *d_payload,
                const void *d_block_scales, // __half* (FP16 d)
                const void *d_block_mins,   // __half* (FP16 m), Q4_1/Q4_K only
                float *d_C_fp32,
                const float *d_scale_A_blockwise, // [2 × blocks_per_row]
                float *d_partial_fp32,            // [KB_MAX × 2 × N]
                int N, int K,
                uint8_t codebook_id,
                int device_id, void *stream);

            bool rocmGemv_native_vnni_small_m_fp32(
                const int8_t *d_A_int8,
                const uint8_t *d_payload,
                const void *d_block_scales, // __half* (FP16 d)
                const void *d_block_mins,   // __half* auxiliary mins/scales, nullable for symmetric formats
                const void *d_block_emins,  // uint32_t* packed emins, Q2_K only
                float *d_C_fp32,
                const float *d_scale_A_blockwise, // [M × blocks_per_row]
                float *d_partial_fp32,            // [KB_MAX × M × N]
                int M, int N, int K,
                uint8_t codebook_id,
                int device_id, void *stream);

            bool rocmGemv_native_vnni_small_m_batched_fp32(
                const int8_t *d_A_int8,
                const uint8_t *const *d_payloads,
                const uint16_t *const *d_block_scales,
                const uint16_t *const *d_block_mins,
                const uint32_t *const *d_block_emins,
                const float *const *d_biases,
                float *const *d_outputs,
                const float *d_scale_A_blockwise, // [M × blocks_per_row]
                float *const *d_partials,         // per-projection [KB_MAX × M × N]
                const int *Ns,
                int num_projections,
                int M, int K,
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
            bool rocmGemv_int8_scatter_vnni_blockwise(
                const int8_t *d_A_int8,
                const int8_t *d_B_int8_vnni,
                float *d_C_fp32,
                const float *d_scales_A_blockwise,
                const float *d_scales_B,
                const float *d_bias,
                float *d_partial_buf,
                int N, int K,
                float alpha, float beta,
                const float *d_C_existing,
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

            bool rocmGemv_int8_scatter_batched_vnni_blockwise(
                const int8_t *d_A_int8,
                const float *d_scales_A_blockwise,
                float *d_partial_buf,
                int num_projections,
                const int8_t *const *d_B_ptrs,
                float *const *d_C_ptrs,
                const float *const *d_scales_B_ptrs,
                const float *const *d_bias_ptrs,
                const int *N_per_proj,
                int K,
                float alpha, float beta,
                int device_id, void *stream);

            bool rocmGemv_int8_int8_fp32_vnni_blockwise_scaled_batched(
                const int8_t *d_A_int8,
                const float *d_scales_A_blockwise,
                int num_projections,
                const int8_t *const *d_B_ptrs,
                float *const *d_C_ptrs,
                const float *const *d_scales_B_ptrs,
                const float *const *d_bias_ptrs,
                const int *N_per_proj,
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

            // INT8 VNNI wide-tile V3 blockwise GEMM (activation scales baked in, FP32 output).
            // KT=8 forced (1 block per tile). Caller applies weight-only epilogue.
            // m_tile_override: -1=auto, or 16/32 to force specific M_TILE.
            // Defined in ROCmQuantisedGemmKernel_INT8_VNNI.hip.
            bool rocmQuantGemm_int8_int8_fp32_vnni_prefill_wide_tile_v3_blockwise(
                const int8_t *d_A_int8,
                const int8_t *d_B_int8_vnni,
                float *d_C_fp32,
                const float *d_scales_A_blockwise,
                int M, int N, int K,
                int device_id, void *stream,
                int m_tile_override = -1,
                int unroll_kk = -1);

            // INT8 VNNI wide-tile V7 blockwise GEMM (activation scales baked in, FP32 output).
            // KT=8 forced (1 block per tile). Caller applies weight-only epilogue.
            // m_tile_override: -1=auto, or 16/32/64 to force specific M_TILE.
            // Defined in ROCmQuantisedGemmKernel_INT8_VNNI.hip.
            bool rocmQuantGemm_int8_int8_fp32_vnni_prefill_wide_tile_v7_blockwise(
                const int8_t *d_A_int8,
                const int8_t *d_B_int8_vnni,
                float *d_C_fp32,
                const float *d_scales_A_blockwise,
                int M, int N, int K,
                int device_id, void *stream,
                int m_tile_override = -1,
                int unroll_kk = -1);

            // Weight-only scaling epilogue for blockwise GEMM output.
            // Input is FP32 (activation scales already applied by blockwise kernel).
            // Applies: output = alpha * input * scale_B[n] + beta*existing + bias
            // Defined in ROCmQuantisedGemmKernel.hip.
            bool rocmQuantGemm_applyScaling_weightOnly(
                const float *d_C_fp32_in,
                float *d_output,
                const float *d_scales_B,
                int M, int N,
                float alpha, float beta,
                const float *d_C_existing,
                const float *d_bias,
                int rocm_device_id, void *stream);
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
            // VNNI-only on device: native-VNNI (≤6-bit) or INT8-VNNI (8-bit).
            int8_t *d_weights_int8_vnni = nullptr;    // [K/4 x N x 4] VNNI layout (sole INT8 device copy)
            float *d_scales_B = nullptr;              // [N] per-column scales
            uint8_t *d_weights_native_vnni = nullptr; // [blocks_per_row × N × payload_bytes]
            void *d_weights_native_scales = nullptr;  // [blocks_per_row × N] __half*
            void *d_weights_native_mins = nullptr;    // [blocks_per_row × N] __half* (asymmetric only, else NULL)
            void *d_weights_native_emins = nullptr;   // [blocks_per_row × N] uint32_t* (Q2_K only, packed {lo,hi} FP16 emins)
            uint8_t native_vnni_codebook_id = 0;
            uint32_t native_vnni_blocks_per_row = 0;
            bool has_native_vnni = false;
            void *startup_h2d_pinned_scales = nullptr;
            void *startup_h2d_pinned_vnni = nullptr;
            void *startup_h2d_pinned_native_vnni = nullptr;
            void *startup_h2d_pinned_native_scales = nullptr;
            void *startup_h2d_pinned_native_mins = nullptr;
            void *startup_h2d_pinned_native_emins = nullptr;

            // Work buffer pointers - obtained from workspace at execution time
            // These are NOT owned by the kernel - they point into workspace-managed memory
            int8_t *d_A_int8 = nullptr;            // [M x K] quantized activations
            float *d_scales_A = nullptr;           // [M] per-row scales (row-wise mode)
            float *d_scales_A_blockwise = nullptr; // [M x blocks_per_row] per-block scales (blockwise mode)
            int32_t *d_C_int32 = nullptr;          // [M x N] INT32 accumulator

            // Transfer work buffers - also from workspace
            float *d_A_fp32 = nullptr; // [M x K] input FP32
            float *d_C_fp32 = nullptr; // [M x N] output FP32

            // Scatter+reduce partial buffer (from workspace)
            float *d_scatter_partial = nullptr; // [KB_MAX × N] FP32 scatter partials

            // Capacity tracking for workspace buffers (set during validateWorkspace)
            size_t d_A_fp32_capacity = 0;
            size_t d_C_fp32_capacity = 0;

            // Workspace validation cache: skip redundant hash map lookups when
            // the same workspace is re-validated across GEMM calls.
            const void *validated_workspace = nullptr;

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
                    if (d_scales_B)
                        rocmQuantGemm_freeDevice(d_scales_B, rocm_device_id);
                    if (d_weights_native_vnni)
                        rocmQuantGemm_freeDevice(d_weights_native_vnni, rocm_device_id);
                    if (d_weights_native_scales)
                        rocmQuantGemm_freeDevice(d_weights_native_scales, rocm_device_id);
                    if (d_weights_native_mins)
                        rocmQuantGemm_freeDevice(d_weights_native_mins, rocm_device_id);
                    if (d_weights_native_emins)
                        rocmQuantGemm_freeDevice(d_weights_native_emins, rocm_device_id);
#ifdef HAVE_ROCM
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
                    free_pinned(startup_h2d_pinned_native_vnni);
                    free_pinned(startup_h2d_pinned_native_scales);
                    free_pinned(startup_h2d_pinned_native_mins);
                    free_pinned(startup_h2d_pinned_native_emins);
#endif
                }
                // Work buffers are NOT freed - they are owned by workspace
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
                free_if_set(upload.startup_h2d_pinned_native_vnni);
                free_if_set(upload.startup_h2d_pinned_native_scales);
                free_if_set(upload.startup_h2d_pinned_native_mins);
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
                    // Async path: stage through pinned host memory to avoid
                    // ROCm hipMemcpy failures on pageable brk-heap addresses,
                    // and to enable DMA overlap with other GPU work.
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

                // Synchronous fallback (only used when stream creation fails)
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
                    if (upload.d_scales)
                        rocmQuantGemm_freeDevice(upload.d_scales, device_id);
                    if (upload.d_native_vnni_payload)
                        rocmQuantGemm_freeDevice(upload.d_native_vnni_payload, device_id);
#ifdef HAVE_ROCM
                    if (upload.d_native_vnni_scales)
                        hipFree(upload.d_native_vnni_scales);
                    if (upload.d_native_vnni_mins)
                        hipFree(upload.d_native_vnni_mins);
                    freeStartupPinnedStaging(upload);
                    if (upload.startup_h2d_stream)
                        hipStreamDestroy(reinterpret_cast<hipStream_t>(upload.startup_h2d_stream));
#endif
                }
            }
            else
            {
                if (d_int8_data_vnni)
                    rocmQuantGemm_freeDevice(d_int8_data_vnni, rocm_device_id);
                if (d_scales)
                    rocmQuantGemm_freeDevice(d_scales, rocm_device_id);
            }
        }

        // =====================================================================
        // Weight packing: see ROCmWeightPacker.cpp
        // (packNativeVNNI, packWeightsToROCm_Q8_0_fast, packWeightsToROCm)
        // =====================================================================

        // =================================================================
        // ConcurrentPrefillPool: lightweight stream + scratch pool for
        // overlapping fused GEMM projections during prefill (M>1).
        //
        // Each stream gets its own scratch buffer so blockwise GEMM kernels
        // can write intermediate results without conflicting with projections
        // on other streams.  The pool is lazily initialized on first use
        // (per-kernel instance, not static).
        // =================================================================
        struct ConcurrentPrefillPool
        {
            static constexpr int MAX_STREAMS = 8;

            hipStream_t streams[MAX_STREAMS] = {};
            hipEvent_t completion[MAX_STREAMS] = {};
            hipEvent_t quant_ready = nullptr;
            int32_t *scratch[MAX_STREAMS] = {};
            size_t scratch_capacity[MAX_STREAMS] = {}; // in elements (M*N)
            float *scatter_partial[MAX_STREAMS] = {};
            size_t scatter_partial_capacity[MAX_STREAMS] = {}; // in elements (KB_MAX*N)
            int count = 0;
            int device_id = -1;
            bool initialized = false;

            void init(int dev_id, int num_streams)
            {
                if (initialized)
                    return;
                device_id = dev_id;
                count = std::min(num_streams, MAX_STREAMS);
                hipSetDevice(dev_id);
                for (int i = 0; i < count; ++i)
                {
                    hipStreamCreateWithFlags(&streams[i], hipStreamNonBlocking);
                    hipEventCreateWithFlags(&completion[i], hipEventDisableTiming);
                }
                hipEventCreateWithFlags(&quant_ready, hipEventDisableTiming);
                initialized = true;
                LOG_DEBUG("[ConcurrentPrefillPool] Initialized " << count
                                                                 << " streams on device " << dev_id);
            }

            // Ensure scratch buffer i has at least `elements` int32s
            bool ensureScratch(int idx, size_t elements)
            {
                if (idx < 0 || idx >= count)
                    return false;
                if (scratch_capacity[idx] >= elements)
                    return true; // Already big enough

                // Free old
                if (scratch[idx])
                {
                    hipSetDevice(device_id);
                    hipFree(scratch[idx]);
                    scratch[idx] = nullptr;
                    scratch_capacity[idx] = 0;
                }

                hipSetDevice(device_id);
                hipError_t err = hipMalloc(&scratch[idx], elements * sizeof(int32_t));
                if (err != hipSuccess)
                {
                    LOG_ERROR("[ConcurrentPrefillPool] Failed to allocate scratch["
                              << idx << "] (" << (elements * 4 / 1024) << " KB): "
                              << hipGetErrorString(err));
                    return false;
                }
                scratch_capacity[idx] = elements;
                LOG_DEBUG("[ConcurrentPrefillPool] Allocated scratch[" << idx
                                                                       << "] = " << (elements * 4 / 1024) << " KB");
                return true;
            }

            // Ensure scatter partial buffer i has at least `elements` floats
            bool ensureScatterPartial(int idx, size_t elements)
            {
                if (idx < 0 || idx >= count)
                    return false;
                if (scatter_partial_capacity[idx] >= elements)
                    return true;

                if (scatter_partial[idx])
                {
                    hipSetDevice(device_id);
                    hipFree(scatter_partial[idx]);
                    scatter_partial[idx] = nullptr;
                    scatter_partial_capacity[idx] = 0;
                }

                hipSetDevice(device_id);
                hipError_t err = hipMalloc(&scatter_partial[idx], elements * sizeof(float));
                if (err != hipSuccess)
                {
                    LOG_ERROR("[ConcurrentPool] Failed to allocate scatter_partial["
                              << idx << "] (" << (elements * 4 / 1024) << " KB): "
                              << hipGetErrorString(err));
                    return false;
                }
                scatter_partial_capacity[idx] = elements;
                LOG_DEBUG("[ConcurrentPool] Allocated scatter_partial[" << idx
                                                                        << "] = " << (elements * 4 / 1024) << " KB");
                return true;
            }

            void destroy()
            {
                if (!initialized)
                    return;
                hipSetDevice(device_id);
                for (int i = 0; i < count; ++i)
                {
                    if (streams[i])
                    {
                        hipStreamDestroy(streams[i]);
                        streams[i] = nullptr;
                    }
                    if (completion[i])
                    {
                        hipEventDestroy(completion[i]);
                        completion[i] = nullptr;
                    }
                    if (scratch[i])
                    {
                        hipFree(scratch[i]);
                        scratch[i] = nullptr;
                        scratch_capacity[i] = 0;
                    }
                    if (scatter_partial[i])
                    {
                        hipFree(scatter_partial[i]);
                        scatter_partial[i] = nullptr;
                        scatter_partial_capacity[i] = 0;
                    }
                }
                if (quant_ready)
                {
                    hipEventDestroy(quant_ready);
                    quant_ready = nullptr;
                }
                initialized = false;
                count = 0;
            }

            ~ConcurrentPrefillPool() { destroy(); }
        };

        // =====================================================================
        // Per-device shared ConcurrentPrefillPool singleton.
        //
        // Previously each ROCmQuantisedGemmKernel instance owned its own pool,
        // which for a 64-layer model meant ~80 separate pools each with its own
        // scratch (M*N int32) and scatter_partial (64*N float) buffers —
        // several GB of duplicated scratch on a single device. The pool is
        // purely infrastructure (streams + events + scratch), so a single
        // shared per-device instance is sufficient and dramatically cheaper.
        // =====================================================================
        static std::mutex &sharedPrefillPoolMutex()
        {
            static std::mutex m;
            return m;
        }

        static std::unordered_map<int, std::unique_ptr<ConcurrentPrefillPool>> &sharedPrefillPools()
        {
            static std::unordered_map<int, std::unique_ptr<ConcurrentPrefillPool>> pools;
            return pools;
        }

        static ConcurrentPrefillPool &getSharedPrefillPool(int device_id, int requested_streams)
        {
            std::lock_guard<std::mutex> lock(sharedPrefillPoolMutex());
            auto &pools = sharedPrefillPools();
            auto it = pools.find(device_id);
            if (it == pools.end())
            {
                auto pool = std::make_unique<ConcurrentPrefillPool>();
                // Initialize with MAX_STREAMS so the pool can serve any
                // caller's concurrency request without re-initialization.
                pool->init(device_id, ConcurrentPrefillPool::MAX_STREAMS);
                it = pools.emplace(device_id, std::move(pool)).first;
            }
            (void)requested_streams; // preserved for API compat
            return *it->second;
        }

        // =====================================================================
        // Constructor / Destructor
        // =====================================================================

        void ROCmQuantisedGemmKernel::clearSharedPrefillPools()
        {
            std::lock_guard<std::mutex> lock(sharedPrefillPoolMutex());
            auto &pools = sharedPrefillPools();
            // Pool destructor calls destroy() which frees streams, events, and
            // all scratch/scatter GPU allocations per device.
            pools.clear();
        }

        ROCmQuantisedGemmKernel::ROCmQuantisedGemmKernel(const TensorBase *weights, int rocm_device_id)
            : weights_(weights),
              packed_(nullptr),
              rocm_device_id_(rocm_device_id),
              N_(0),
              K_(0),
              weights_converted_(false),
              owns_weight_memory_(true), // Legacy path owns weight memory
              slice_id_(g_rocm_gemm_workspace_slice_counter.fetch_add(1, std::memory_order_relaxed)),
              impl_(std::make_unique<Impl>())
        {
            if (!weights)
            {
                throw std::runtime_error("[ROCmQuantisedGemmKernel] Null weight tensor");
            }

            // Get dimensions
            N_ = weights->rows(); // Output features
            K_ = weights->cols(); // Input features

            // Validate it's a quantized type supported by this kernel
            TensorType wt = weights->native_type();
            bool is_quantized = isNativeVnniFormat(wt) || isInt8VnniFormat(wt);

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
              slice_id_(g_rocm_gemm_workspace_slice_counter.fetch_add(1, std::memory_order_relaxed)),
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

        ROCmQuantisedGemmKernel::ROCmQuantisedGemmKernel(
            int N, int K, int rocm_device_id,
            uint8_t *d_native_vnni, void *d_native_scales,
            void *d_native_mins, void *d_native_emins,
            uint8_t codebook_id, uint32_t blocks_per_row,
            std::shared_ptr<void> lifetime_owner)
            : weights_(nullptr),
              packed_(nullptr),
              lifetime_owner_(std::move(lifetime_owner)),
              rocm_device_id_(rocm_device_id),
              N_(static_cast<size_t>(N)),
              K_(static_cast<size_t>(K)),
              weights_converted_(true),   // Already on device
              owns_weight_memory_(false), // Batch allocation owns the memory
              slice_id_(g_rocm_gemm_workspace_slice_counter.fetch_add(1, std::memory_order_relaxed)),
              impl_(std::make_unique<Impl>())
        {
            impl_->d_weights_native_vnni = d_native_vnni;
            impl_->d_weights_native_scales = d_native_scales;
            impl_->d_weights_native_mins = d_native_mins;
            impl_->d_weights_native_emins = d_native_emins;
            impl_->native_vnni_codebook_id = codebook_id;
            impl_->native_vnni_blocks_per_row = blocks_per_row;
            impl_->has_native_vnni = true;
            impl_->owns_weight_memory = false;
            impl_->rocm_device_id = rocm_device_id;

            LOG_DEBUG("[ROCmQuantisedGemmKernel] Created (MoE batch device ptrs) for " << N_ << "x" << K_
                                                                                       << " on ROCm device " << rocm_device_id_);
        }

        ROCmQuantisedGemmKernel::~ROCmQuantisedGemmKernel() = default;

        ROCmQuantisedGemmKernel::ROCmQuantisedGemmKernel(ROCmQuantisedGemmKernel &&) noexcept = default;
        ROCmQuantisedGemmKernel &ROCmQuantisedGemmKernel::operator=(ROCmQuantisedGemmKernel &&) noexcept = default;

        bool ROCmQuantisedGemmKernel::exportNativeVNNIMatrixDesc(DeviceNativeVNNIMatrixDesc &out)
        {
            out = {};
            ensureWeightsConverted();

            if (!impl_ || !impl_->has_native_vnni ||
                !impl_->d_weights_native_vnni || !impl_->d_weights_native_scales)
            {
                return false;
            }

            out.payload = impl_->d_weights_native_vnni;
            out.scales = impl_->d_weights_native_scales;
            out.mins = impl_->d_weights_native_mins;
            out.emins = impl_->d_weights_native_emins;
            out.n = static_cast<int>(N_);
            out.k = static_cast<int>(K_);
            out.blocks_per_row = impl_->native_vnni_blocks_per_row;
            out.codebook_id = impl_->native_vnni_codebook_id;
            return out.valid();
        }

        /**
         * @brief Classify the best prefill route for the current shape and weight format.
         *
         * This is the single source of truth for dispatch routing decisions.
         * The dispatch policy is:
         *   - ≤6-bit weights (has_native_vnni) → NATIVE_VNNI
         *   - 8-bit weights (d_weights_int8_vnni) → INT8_VNNI_NATIVE
         *   - Otherwise → UNSUPPORTED (no viable path)
         *
         * UNSUPPORTED is returned when impl_ is missing, dimensions are invalid,
         * or no VNNI weights are available.
         */
        ROCmQuantisedGemmKernel::PrefillDispatchPath ROCmQuantisedGemmKernel::selectPrefillDispatchPath(int m, int n, int k) const
        {
            (void)n;

            if (m <= 1 || k <= 0 || (k % 4) != 0)
            {
                return PrefillDispatchPath::UNSUPPORTED;
            }

            if (!impl_)
            {
                return PrefillDispatchPath::UNSUPPORTED;
            }

            // CK path retired. No fallback path available.

            // ≤6-bit formats use native-VNNI (lossless decode, FP16 block scales)
            if (impl_->has_native_vnni)
            {
                return PrefillDispatchPath::NATIVE_VNNI;
            }

            // 8-bit formats use INT8-VNNI (requantized symmetric INT8)
            if (impl_->d_weights_int8_vnni != nullptr)
            {
                return PrefillDispatchPath::INT8_VNNI_NATIVE;
            }

            LOG_WARN("[ROCmQuantisedGemmKernel] No viable prefill dispatch path:"
                     " has_native_vnni="
                     << impl_->has_native_vnni
                     << " d_weights_int8_vnni=" << (const void *)impl_->d_weights_int8_vnni
                     << " M=" << m << " N=" << n << " K=" << k);
            return PrefillDispatchPath::UNSUPPORTED;
        }

        /**
         * @brief Try native M>1 prefill execution for INT8 VNNI or ratio-VNNI formats.
         *
         * A false return means neither INT8-VNNI nor native-VNNI prefill could
         * execute. The caller will error unless LLAMINAR_ROCM_FORCE_CK=1 is set.
         */
        bool ROCmQuantisedGemmKernel::tryPrefillNativeGemm(
            const int8_t *d_A_int8,
            float *d_output,
            const float *d_scales_A,
            const float *d_scales_A_blockwise,
            const float *d_scales_B,
            const float *d_bias,
            int m, int n, int k,
            float alpha, float beta,
            const char *callsite,
            void *stream_override,
            int32_t *scratch_int32_override)
        {
            // Use overrides when provided (concurrent prefill dispatch)
            void *effective_stream = stream_override ? stream_override : gpu_stream_;
            int32_t *effective_scratch_int32 = scratch_int32_override ? scratch_int32_override : (impl_ ? impl_->d_C_int32 : nullptr);
            const auto path_label = [&](PrefillDispatchPath p) -> const char *
            {
                switch (p)
                {
                case PrefillDispatchPath::NATIVE_VNNI:
                    return "native_vnni";
                case PrefillDispatchPath::INT8_VNNI_NATIVE:
                    return "int8_vnni_native";
                case PrefillDispatchPath::UNSUPPORTED:
                default:
                    return "unsupported";
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

                record_path_selected(PrefillDispatchPath::UNSUPPORTED);
                record_fallback_reason(reason);

                std::call_once(*selected_once, [&]()
                               { LOG_WARN("[" << callsite << "] Prefill GEMM unavailable (reason="
                                              << reason << ")"); });
            };

            // Native-VNNI path uses impl_->d_weights_native_scales (not d_scales_B),
            // so d_scales_B may be null when only native-VNNI weights are present
            // (e.g. weights loaded via GPU pipeline / MoE batch constructor).
            const bool native_vnni_available = impl_ && impl_->has_native_vnni;
            if (!impl_ || !d_A_int8 || !d_output || !d_scales_A || !effective_scratch_int32 ||
                (!d_scales_B && !native_vnni_available))
            {
                LOG_WARN("[" << callsite << "] Prefill GEMM null pointer diagnostic:"
                                            " impl="
                             << (const void *)impl_.get()
                             << " d_A_int8=" << (const void *)d_A_int8
                             << " d_output=" << (const void *)d_output
                             << " d_scales_A=" << (const void *)d_scales_A
                             << " d_scales_B=" << (const void *)d_scales_B
                             << " scratch=" << (const void *)effective_scratch_int32
                             << " workspace=" << (const void *)workspace_);
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

            if (path == PrefillDispatchPath::NATIVE_VNNI)
            {
                LOG_TRACE("[" << callsite << "] Trying native-VNNI prefill (M=" << m
                              << " N=" << n << " K=" << k << ")");

                if (!impl_->d_weights_native_vnni || !impl_->d_weights_native_scales)
                {
                    LOG_WARN("[" << callsite << "] Native-VNNI prefill missing weight buffers:"
                                                " d_weights_native_vnni="
                                 << (const void *)impl_->d_weights_native_vnni
                                 << " d_weights_native_scales=" << (const void *)impl_->d_weights_native_scales);
                    logFallback("buffers");
                    return false;
                }

                if (beta != 0.0f && d_output == impl_->d_C_fp32)
                {
                    LOG_WARN("[" << callsite << "] Native-VNNI prefill beta path needs a distinct existing-output buffer");
                    logFallback("native_beta_alias");
                    return false;
                }

                float *d_native_output = (beta != 0.0f) ? impl_->d_C_fp32 : d_output;
                native_ok = rocmGemm_native_vnni_fp32(
                    d_A_int8,
                    impl_->d_weights_native_vnni,
                    impl_->d_weights_native_scales,
                    impl_->d_weights_native_mins,
                    impl_->d_weights_native_emins,
                    d_native_output,
                    d_scales_A,
                    d_scales_A_blockwise,
                    m, n, k,
                    impl_->native_vnni_codebook_id,
                    rocm_device_id_, effective_stream);

                if (profiling_enabled)
                    kernel_end = std::chrono::high_resolution_clock::now();

                if (!native_ok)
                {
                    logFallback("launch_error");
                    return false;
                }

                if (profiling_enabled)
                {
                    record_kernel_ms(
                        path,
                        std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count());
                }

                if (beta != 0.0f)
                {
                    std::chrono::high_resolution_clock::time_point epilogue_start{};
                    if (profiling_enabled)
                        epilogue_start = std::chrono::high_resolution_clock::now();

                    if (!rocmQuantGemm_applyFp32CombineEpilogue(
                            d_native_output,
                            d_output,
                            d_output,
                            d_bias,
                            m,
                            n,
                            alpha,
                            beta,
                            rocm_device_id_,
                            effective_stream))
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
                }
                else if (alpha != 1.0f || d_bias)
                {
                    std::chrono::high_resolution_clock::time_point epilogue_start{};
                    if (profiling_enabled)
                        epilogue_start = std::chrono::high_resolution_clock::now();

                    if (!rocmQuantGemm_applyFp32Epilogue(
                            d_output,
                            d_bias,
                            m,
                            n,
                            alpha,
                            rocm_device_id_,
                            effective_stream))
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
                }

                return true;
            }

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
                                   { LOG_DEBUG("[" << callsite << "] INT8 prefill policy enabled (ratio+work map): "
                                                   << policy_cfg.profile
                                                   << ", M=" << m << ", N=" << n << ", K=" << k
                                                   << ", N/K=" << std::fixed << std::setprecision(2)
                                                   << (static_cast<double>(n) / static_cast<double>(std::max(k, 1)))
                                                   << ")"); });
                }

                // Wide-tile path: covers all M-rows in one block (extreme-wide shapes)
                //
                // BLOCKWISE ACTIVATION SCALES: when d_scales_A_blockwise is provided,
                // use blockwise V3/V7 kernels that apply per-block activation scales
                // during accumulation, outputting FP32 directly. Then use weight-only
                // epilogue (scale_B only). K must be a multiple of 32 for block alignment.
                if (d_scales_A_blockwise != nullptr && (k % 32) == 0)
                {
                    // Reinterpret d_C_int32 buffer as float* — same 4-byte-per-element
                    // layout, device memory, no reallocation needed.
                    float *d_C_fp32_blockwise = reinterpret_cast<float *>(effective_scratch_int32);

                    const auto &bw_env = debugEnv().rocm;
                    // Data-driven dispatch (tuned on gfx906 MI50/MI60, UNROLL_KK sweep):
                    //   V7/MT64/U2 wins every shape×M in the perf sweep (1.02–1.45x
                    //   over best V3).  The old k>=n→V3 heuristic is retired.
                    //   Force overrides still take priority for experimentation.
                    const bool use_v3 = bw_env.blockwise_force_v3   ? true
                                        : bw_env.blockwise_force_v7 ? false
                                                                    : false;

                    if (use_v3)
                    {
                        // V3 M_TILE scales with M for K>=N region (tuning data):
                        //   M <= 128 → MT16, M <= 256 → MT32, M > 256 → MT64
                        int v3_mt = bw_env.blockwise_v3_mt;
                        if (v3_mt == 0)
                        {
                            if (m <= 128)
                                v3_mt = 16;
                            else if (m <= 256)
                                v3_mt = 32;
                            else
                                v3_mt = 64;
                        }
                        native_ok = rocmQuantGemm_int8_int8_fp32_vnni_prefill_wide_tile_v3_blockwise(
                            d_A_int8,
                            impl_->d_weights_int8_vnni,
                            d_C_fp32_blockwise,
                            d_scales_A_blockwise,
                            m, n, k,
                            rocm_device_id_, effective_stream,
                            v3_mt,
                            bw_env.blockwise_v3_unroll);
                    }
                    else
                    {
                        native_ok = rocmQuantGemm_int8_int8_fp32_vnni_prefill_wide_tile_v7_blockwise(
                            d_A_int8,
                            impl_->d_weights_int8_vnni,
                            d_C_fp32_blockwise,
                            d_scales_A_blockwise,
                            m, n, k,
                            rocm_device_id_, effective_stream,
                            bw_env.blockwise_v7_mt,
                            bw_env.blockwise_v7_unroll);
                    }

                    if (native_ok)
                    {
                        if (profiling_enabled)
                            kernel_end = std::chrono::high_resolution_clock::now();

                        if (profiling_enabled)
                        {
                            record_kernel_ms(
                                path,
                                std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count());
                        }

                        const float *d_existing = nullptr;
                        if (beta != 0.0f)
                        {
                            if (d_output == impl_->d_C_fp32)
                            {
                                logFallback("beta_alias");
                                return false;
                            }
                            hipError_t copy_err = hipMemcpyAsync(
                                impl_->d_C_fp32,
                                d_output,
                                static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float),
                                hipMemcpyDeviceToDevice,
                                static_cast<hipStream_t>(effective_stream));
                            if (copy_err != hipSuccess)
                            {
                                LOG_ERROR("[" << callsite << "] Failed to snapshot beta existing output: "
                                              << hipGetErrorString(copy_err));
                                logFallback("launch_error");
                                return false;
                            }
                            d_existing = impl_->d_C_fp32;
                        }

                        std::chrono::high_resolution_clock::time_point epilogue_start{};
                        if (profiling_enabled)
                            epilogue_start = std::chrono::high_resolution_clock::now();

                        if (!rocmQuantGemm_applyScaling_weightOnly(
                                d_C_fp32_blockwise,
                                d_output,
                                d_scales_B,
                                m, n,
                                alpha, beta,
                                d_existing,
                                d_bias,
                                rocm_device_id_, effective_stream))
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
                    else
                    {
                        // Blockwise kernel launch failed.
                        static std::once_flag bw_fallback_once;
                        std::call_once(bw_fallback_once, [&]()
                                       { LOG_ERROR("[" << callsite << "] Blockwise INT8 kernel failed (M=" << m
                                                       << " N=" << n << " K=" << k << ")"); });
                        logFallback("launch_error");
                        return false;
                    }
                }

                if (!has_manual_override && policy_cfg.use_wide_tile)
                {
                    if (rocm_env.wide_tile_v7)
                    {
                        // V7 forced via env var: safe-tile split, branchless interior loop
                        native_ok = rocmQuantGemm_int8_int8_int32_vnni_prefill_wide_tile_v7(
                            d_A_int8,
                            impl_->d_weights_int8_vnni,
                            effective_scratch_int32,
                            m, n, k,
                            rocm_env.wide_tile_kt,
                            rocm_device_id_, effective_stream);
                    }
                    else if (rocm_env.wide_tile_v3)
                    {
                        // V3 forced via env var: LDS double-buffered, N64
                        native_ok = rocmQuantGemm_int8_int8_int32_vnni_prefill_wide_tile_v3(
                            d_A_int8,
                            impl_->d_weights_int8_vnni,
                            effective_scratch_int32,
                            m, n, k,
                            rocm_env.wide_tile_kt,
                            rocm_device_id_, effective_stream);
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
                            effective_scratch_int32,
                            m, n, k,
                            16, // KT=16: best throughput for V3 auto-dispatch (perf sweep shows KT=16 wins 10/12 shapes)
                            rocm_device_id_, effective_stream);
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
                            effective_scratch_int32,
                            m, n, k,
                            16, // KT=16: best for V7 (safe-tile split, 2 waves/SIMD)
                            rocm_device_id_, effective_stream);
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
                            effective_scratch_int32,
                            m, n, k,
                            requested_slices,
                            grid_variant,
                            cpt,
                            kernel_body_variant,
                            grid_swizzle_variant,
                            rocm_device_id_, effective_stream);

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
                            effective_scratch_int32,
                            m, n, k,
                            baseline_variant,
                            cpt,
                            rocm_device_id_, effective_stream);
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

            // Epilogue: apply activation/weight scales, alpha/beta, and optional bias.
            const float *d_existing = nullptr;
            if (beta != 0.0f)
            {
                if (d_output == impl_->d_C_fp32)
                {
                    logFallback("beta_alias");
                    return false;
                }
                hipError_t copy_err = hipMemcpyAsync(
                    impl_->d_C_fp32,
                    d_output,
                    static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float),
                    hipMemcpyDeviceToDevice,
                    static_cast<hipStream_t>(effective_stream));
                if (copy_err != hipSuccess)
                {
                    LOG_ERROR("[" << callsite << "] Failed to snapshot beta existing output: "
                                  << hipGetErrorString(copy_err));
                    logFallback("launch_error");
                    return false;
                }
                d_existing = impl_->d_C_fp32;
            }

            std::chrono::high_resolution_clock::time_point epilogue_start{};
            if (profiling_enabled)
                epilogue_start = std::chrono::high_resolution_clock::now();
            if (!rocmQuantGemm_applyScaling(
                    effective_scratch_int32,
                    d_output,
                    d_scales_A,
                    d_scales_B,
                    m, n,
                    alpha, beta,
                    d_existing,
                    d_bias,
                    rocm_device_id_, effective_stream))
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
            const IMPIContext *mpi_ctx,
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
            const IMPIContext *mpi_ctx,
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
            std::unique_ptr<void, decltype(restore_workspace)> workspace_restore_guard(
                (ws && ws != saved_workspace) ? static_cast<void *>(this) : nullptr,
                restore_workspace);

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
            const float *d_input = static_cast<const float *>(A_fp32->gpu_data_ptr());
            float *d_output = static_cast<float *>(C_fp32->gpu_data_ptr());

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
            // DISPATCH POLICY (unified):
            //
            //   ≤6-bit weights (has_native_vnni) → native-VNNI GEMV/GEMM
            //   8-bit weights (d_weights_int8_vnni) → INT8-VNNI GEMV/GEMM
            //
            // If both VNNI paths fail, that is a hard error.
            // =========================================================================

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
            if (use_gpu_path && m > 1 && m <= 4)
            {
                ensureWeightsConverted();

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

                if (!impl_ || (!impl_->has_native_vnni && !(d_weights_vnni && d_scales_B)))
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Small-M verifier path has no supported GEMV weights");
                    return false;
                }

                const float *d_bias = nullptr;
                if (bias)
                {
                    auto *bias_tensor = const_cast<TensorBase *>(bias);
                    const auto target_device = DeviceId::rocm(rocm_device_id_);
                    if (!bias_tensor->ensureOnDevice(target_device))
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Failed to upload small-M bias tensor to ROCm device");
                        return false;
                    }

                    d_bias = static_cast<const float *>(bias->gpu_data_ptr());
                }

                validateWorkspace();

                if (d_weights_vnni &&
                    !validatePointerDeviceOrLog(
                        d_weights_vnni, rocm_device_id_, "d_weights_vnni", "ROCmQuantisedGemmKernel::multiply_tensor"))
                {
                    return false;
                }
                if (d_scales_B && !validatePointerDeviceOrLog(
                                      d_scales_B, rocm_device_id_, "d_scales_B", "ROCmQuantisedGemmKernel::multiply_tensor"))
                {
                    return false;
                }
                if (!validatePointerDeviceOrLog(
                        impl_->d_A_int8, rocm_device_id_, "workspace::QUANT_A", "ROCmQuantisedGemmKernel::multiply_tensor") ||
                    !validatePointerDeviceOrLog(
                        impl_->d_scales_A, rocm_device_id_, "workspace::SCALES_A", "ROCmQuantisedGemmKernel::multiply_tensor") ||
                    !validatePointerDeviceOrLog(
                        impl_->d_C_int32, rocm_device_id_, "workspace::ACC_INT32", "ROCmQuantisedGemmKernel::multiply_tensor"))
                {
                    return false;
                }

                const bool gemv_output_is_mapped = C_fp32->isMapped();
                const bool gemv_output_needs_copyout = gemv_output_is_mapped || beta != 0.0f;

                LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_tensor] Small-M GEMV verifier path M="
                          << m << " N=" << n << " K=" << k
                          << (d_bias ? " +bias" : ""));

#ifdef HAVE_ROCM
                if (m == 2 && impl_->has_native_vnni &&
                    !C_fp32->isMapped() && beta == 0.0f &&
                    debugEnv().rocm.concurrent_m2_rows)
                {
                    if ((k % 32) != 0)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Concurrent native-VNNI M=2 requires K multiple of 32, got K="
                                  << k);
                        return false;
                    }

                    if (!rocmQuantGemm_quantizeActivationsBlockwise(
                            d_input,
                            impl_->d_A_int8,
                            impl_->d_scales_A_blockwise,
                            m,
                            k,
                            rocm_device_id_,
                            gpu_stream_))
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Concurrent small-M blockwise activation quantization failed");
                        return false;
                    }

                    constexpr int SCATTER_KB_MAX = 64;
                    const int scale_blocks_per_row = k / 32;
                    auto &pool = getSharedPrefillPool(rocm_device_id_, m);
                    const auto check_hip = [&](hipError_t err, const char *operation) -> bool
                    {
                        if (err == hipSuccess)
                            return true;
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Concurrent small-M native-VNNI "
                                  << operation << " failed: " << hipGetErrorString(err));
                        return false;
                    };

                    if (!check_hip(
                            hipEventRecord(pool.quant_ready,
                                           static_cast<hipStream_t>(gpu_stream_)),
                            "record quant-ready event"))
                    {
                        return false;
                    }

                    for (int row = 0; row < m; ++row)
                    {
                        const int stream_idx = row % pool.count;
                        const size_t partial_elements = static_cast<size_t>(SCATTER_KB_MAX) * n;
                        if (!pool.ensureScatterPartial(stream_idx, partial_elements))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Failed to allocate concurrent native-VNNI scatter partial for row "
                                      << row << " (" << (partial_elements * sizeof(float)) << " bytes)");
                            return false;
                        }

                        if (!check_hip(
                                hipStreamWaitEvent(pool.streams[stream_idx], pool.quant_ready, 0),
                                "wait for quant-ready event"))
                        {
                            return false;
                        }

                        const int8_t *d_A_row = impl_->d_A_int8 + static_cast<size_t>(row) * k;
                        const float *d_scale_row = impl_->d_scales_A + row;
                        const float *d_scale_block_row = impl_->d_scales_A_blockwise + static_cast<size_t>(row) * scale_blocks_per_row;
                        float *d_output_row = d_output + static_cast<size_t>(row) * n;

                        if (!rocmGemv_native_vnni_fp32(
                                d_A_row,
                                impl_->d_weights_native_vnni,
                                impl_->d_weights_native_scales,
                                impl_->d_weights_native_mins,
                                impl_->d_weights_native_emins,
                                d_output_row,
                                d_scale_row,
                                pool.scatter_partial[stream_idx],
                                n, k,
                                impl_->native_vnni_codebook_id,
                                rocm_device_id_, pool.streams[stream_idx],
                                d_scale_block_row))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Concurrent small-M native-VNNI GEMV failed at row "
                                      << row);
                            return false;
                        }

                        if (alpha != 1.0f || d_bias)
                        {
                            if (!rocmQuantGemm_applyFp32Epilogue(
                                    d_output_row,
                                    d_bias,
                                    1,
                                    n,
                                    alpha,
                                    rocm_device_id_,
                                    pool.streams[stream_idx]))
                            {
                                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Concurrent small-M native-VNNI epilogue failed at row "
                                          << row);
                                return false;
                            }
                        }

                        if (!check_hip(
                                hipEventRecord(pool.completion[stream_idx],
                                               pool.streams[stream_idx]),
                                "record row-completion event"))
                        {
                            return false;
                        }
                    }

                    for (int row = 0; row < m; ++row)
                    {
                        if (!check_hip(
                                hipStreamWaitEvent(
                                    static_cast<hipStream_t>(gpu_stream_),
                                    pool.completion[row % pool.count], 0),
                                "wait for row-completion event"))
                        {
                            return false;
                        }
                    }

                    return true;
                }
#endif

                const bool supports_native_small_m =
                    impl_->has_native_vnni && m >= 2 && m <= 4;
                if (supports_native_small_m &&
                    !gemv_output_needs_copyout && beta == 0.0f)
                {
                    if ((k % 32) != 0)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Native-VNNI small-M verifier requires K multiple of 32, got M="
                                  << m << " K=" << k);
                        return false;
                    }

                    if (!rocmQuantGemm_quantizeActivationsBlockwise(
                            d_input,
                            impl_->d_A_int8,
                            impl_->d_scales_A_blockwise,
                            m,
                            k,
                            rocm_device_id_,
                            gpu_stream_))
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Native-VNNI small-M blockwise activation quantization failed");
                        return false;
                    }

                    if (!rocmGemv_native_vnni_small_m_fp32(
                            impl_->d_A_int8,
                            impl_->d_weights_native_vnni,
                            impl_->d_weights_native_scales,
                            impl_->d_weights_native_mins,
                            impl_->d_weights_native_emins,
                            d_output,
                            impl_->d_scales_A_blockwise,
                            impl_->d_scatter_partial,
                            m, n, k,
                            impl_->native_vnni_codebook_id,
                            rocm_device_id_, gpu_stream_))
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Native-VNNI small-M verifier GEMV failed");
                        return false;
                    }

                    if (alpha != 1.0f || d_bias)
                    {
                        if (!rocmQuantGemm_applyFp32Epilogue(
                                d_output,
                                d_bias,
                                m,
                                n,
                                alpha,
                                rocm_device_id_,
                                gpu_stream_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Native-VNNI small-M epilogue failed");
                            return false;
                        }
                    }

                    if (PerfStatsCollector::isEnabled())
                    {
                        PerfStatsCollector::addCounter(
                            "kernel",
                            "rocm_native_vnni_small_m_calls",
                            1.0,
                            "gemm",
                            "rocm:" + std::to_string(rocm_device_id_),
                            PerfStatsCollector::Tags{
                                {"m", std::to_string(m)},
                                {"codebook", std::to_string(static_cast<int>(impl_->native_vnni_codebook_id))},
                                {"n", std::to_string(n)},
                                {"k", std::to_string(k)}});

                        if (m == 2)
                        {
                            PerfStatsCollector::addCounter(
                                "kernel",
                                "rocm_native_vnni_m2_calls",
                                1.0,
                                "gemm",
                                "rocm:" + std::to_string(rocm_device_id_),
                                PerfStatsCollector::Tags{
                                    {"codebook", std::to_string(static_cast<int>(impl_->native_vnni_codebook_id))},
                                    {"n", std::to_string(n)},
                                    {"k", std::to_string(k)}});
                        }
                    }

                    if (PerfStatsCollector::isEnabled())
                    {
                        PerfStatsCollector::addCounter(
                            "kernel",
                            "rocm_native_vnni_small_m_rows",
                            static_cast<double>(m),
                            "gemm",
                            "rocm:" + std::to_string(rocm_device_id_),
                            PerfStatsCollector::Tags{
                                {"m", std::to_string(m)},
                                {"codebook", std::to_string(static_cast<int>(impl_->native_vnni_codebook_id))},
                                {"n", std::to_string(n)},
                                {"k", std::to_string(k)}});
                    }

                    return true;
                }

                for (int row = 0; row < m; ++row)
                {
                    const float *d_input_row = d_input + static_cast<size_t>(row) * k;
                    float *d_output_row = d_output + static_cast<size_t>(row) * n;
                    float *d_gemv_output = gemv_output_needs_copyout ? impl_->d_C_fp32 : d_output_row;

                    if (!rocmQuantGemm_quantizeActivationsBlockwise(
                            d_input_row,
                            impl_->d_A_int8,
                            impl_->d_scales_A_blockwise,
                            1,
                            k,
                            rocm_device_id_,
                            gpu_stream_))
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Small-M blockwise activation quantization failed at row "
                                  << row);
                        return false;
                    }

                    if (impl_->has_native_vnni)
                    {
                        if (!rocmGemv_native_vnni_fp32(
                                impl_->d_A_int8,
                                impl_->d_weights_native_vnni,
                                impl_->d_weights_native_scales,
                                impl_->d_weights_native_mins,
                                impl_->d_weights_native_emins,
                                d_gemv_output,
                                impl_->d_scales_A,
                                impl_->d_scatter_partial,
                                n, k,
                                impl_->native_vnni_codebook_id,
                                rocm_device_id_, gpu_stream_,
                                impl_->d_scales_A_blockwise))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Small-M native-VNNI GEMV failed at row "
                                      << row);
                            return false;
                        }

                        if (beta != 0.0f)
                        {
                            if (!rocmQuantGemm_applyFp32CombineEpilogue(
                                    d_gemv_output,
                                    d_gemv_output,
                                    d_output_row,
                                    d_bias,
                                    1,
                                    n,
                                    alpha,
                                    beta,
                                    rocm_device_id_,
                                    gpu_stream_))
                            {
                                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Small-M native-VNNI beta epilogue failed at row "
                                          << row);
                                return false;
                            }
                        }
                        else if (alpha != 1.0f || d_bias)
                        {
                            if (!rocmQuantGemm_applyFp32Epilogue(
                                    d_gemv_output,
                                    d_bias,
                                    1,
                                    n,
                                    alpha,
                                    rocm_device_id_,
                                    gpu_stream_))
                            {
                                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Small-M native-VNNI epilogue failed at row "
                                          << row);
                                return false;
                            }
                        }
                    }
                    else
                    {
                        const float *d_existing = (beta != 0.0f) ? d_output_row : nullptr;
                        bool int8_decode_ok = rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
                            impl_->d_A_int8, d_weights_vnni, d_gemv_output,
                            impl_->d_scales_A_blockwise, d_scales_B,
                            n, k,
                            alpha, beta,
                            d_existing, d_bias,
                            rocm_device_id_, gpu_stream_);

                        if (!int8_decode_ok)
                        {
                            int8_decode_ok = rocmGemv_int8_scatter_vnni_blockwise(
                                impl_->d_A_int8, d_weights_vnni, d_gemv_output, impl_->d_scales_A_blockwise, d_scales_B, d_bias,
                                impl_->d_scatter_partial,
                                n, k, alpha, beta, d_existing,
                                rocm_device_id_, gpu_stream_);
                        }

                        if (!int8_decode_ok)
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Small-M INT8 GEMV failed at row "
                                      << row);
                            return false;
                        }
                    }

                    if (gemv_output_needs_copyout)
                    {
                        const hipError_t copy_err = hipMemcpyAsync(
                            d_output_row,
                            impl_->d_C_fp32,
                            static_cast<size_t>(n) * sizeof(float),
                            hipMemcpyDeviceToDevice,
                            static_cast<hipStream_t>(gpu_stream_));
                        if (copy_err != hipSuccess)
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Small-M GEMV output copy failed at row "
                                      << row << ": " << hipGetErrorString(copy_err));
                            return false;
                        }
                    }
                }
                return true;
            }

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
                        auto *bias_tensor = const_cast<TensorBase *>(bias);
                        const auto target_device = DeviceId::rocm(rocm_device_id_);
                        if (!bias_tensor->ensureOnDevice(target_device))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Failed to upload decode bias tensor to ROCm device");
                            return false;
                        }

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
                    const bool gemv_output_needs_copyout = gemv_output_is_mapped || beta != 0.0f;
                    float *d_gemv_output = gemv_output_needs_copyout ? impl_->d_C_fp32 : d_output;
                    if (gemv_output_is_mapped)
                    {
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
                    // d_scales_B is only used by the INT8-VNNI path, not native-VNNI.
                    // Native-VNNI weights use impl_->d_weights_native_scales instead.
                    if (d_scales_B && !validatePointerDeviceOrLog(
                                          d_scales_B, rocm_device_id_, "d_scales_B", "ROCmQuantisedGemmKernel::multiply_tensor"))
                    {
                        return false;
                    }
                    if (!validatePointerDeviceOrLog(
                            impl_->d_A_int8, rocm_device_id_, "workspace::QUANT_A", "ROCmQuantisedGemmKernel::multiply_tensor") ||
                        !validatePointerDeviceOrLog(
                            impl_->d_scales_A, rocm_device_id_, "workspace::SCALES_A", "ROCmQuantisedGemmKernel::multiply_tensor") ||
                        !validatePointerDeviceOrLog(
                            impl_->d_C_int32, rocm_device_id_, "workspace::ACC_INT32", "ROCmQuantisedGemmKernel::multiply_tensor"))
                    {
                        return false;
                    }

                    // =====================================================================
                    // Native-VNNI path: lossless Q4/IQ4 decode, FP16 block scales, scale_A inline.
                    // Native GEMV produces FP32, then applies alpha/beta/bias with the shared epilogue when needed.
                    // =====================================================================
                    if (impl_->has_native_vnni)
                    {
                        if (!rocmQuantGemm_quantizeActivationsBlockwise(
                                d_input, impl_->d_A_int8, impl_->d_scales_A_blockwise, m, k, rocm_device_id_, gpu_stream_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Native-VNNI blockwise activation quantization failed");
                            return false;
                        }

                        if (!rocmGemv_native_vnni_fp32(
                                impl_->d_A_int8,
                                impl_->d_weights_native_vnni,
                                impl_->d_weights_native_scales,
                                impl_->d_weights_native_mins,
                                impl_->d_weights_native_emins,
                                d_gemv_output,
                                impl_->d_scales_A,
                                impl_->d_scatter_partial,
                                n, k,
                                impl_->native_vnni_codebook_id,
                                rocm_device_id_, gpu_stream_,
                                impl_->d_scales_A_blockwise))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Native-VNNI GEMV failed");
                            return false;
                        }

                        if (beta != 0.0f)
                        {
                            if (!rocmQuantGemm_applyFp32CombineEpilogue(
                                    d_gemv_output,
                                    d_gemv_output,
                                    d_output,
                                    d_bias,
                                    m,
                                    n,
                                    alpha,
                                    beta,
                                    rocm_device_id_,
                                    gpu_stream_))
                            {
                                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Native-VNNI beta epilogue failed");
                                return false;
                            }
                        }
                        else if (alpha != 1.0f || d_bias)
                        {
                            if (!rocmQuantGemm_applyFp32Epilogue(
                                    d_gemv_output,
                                    d_bias,
                                    m,
                                    n,
                                    alpha,
                                    rocm_device_id_,
                                    gpu_stream_))
                            {
                                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Native-VNNI epilogue failed");
                                return false;
                            }
                        }
                        // Bulk DMA from HBM workspace to output
                        if (gemv_output_needs_copyout)
                        {
                            hipMemcpyAsync(d_output, impl_->d_C_fp32,
                                           static_cast<size_t>(n) * sizeof(float),
                                           hipMemcpyDeviceToDevice,
                                           static_cast<hipStream_t>(gpu_stream_));
                        }
                        return true;
                    }

                    // Use INT8 scatter+reduce GEMV when VNNI weights are available.
                    if (d_weights_vnni)
                    {
                        if (!rocmQuantGemm_quantizeActivationsBlockwise(
                                d_input, impl_->d_A_int8, impl_->d_scales_A_blockwise, m, k, rocm_device_id_, gpu_stream_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] INT8 blockwise activation quantization failed");
                            return false;
                        }

                        const float *d_existing = (beta != 0.0f) ? d_output : nullptr;
                        bool int8_decode_ok = rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
                            impl_->d_A_int8, d_weights_vnni, d_gemv_output,
                            impl_->d_scales_A_blockwise, d_scales_B,
                            n, k,
                            alpha, beta,
                            d_existing, d_bias,
                            rocm_device_id_, gpu_stream_);

                        if (!int8_decode_ok)
                        {
                            int8_decode_ok = rocmGemv_int8_scatter_vnni_blockwise(
                                impl_->d_A_int8, d_weights_vnni, d_gemv_output, impl_->d_scales_A_blockwise, d_scales_B, d_bias,
                                impl_->d_scatter_partial,
                                n, k, alpha, beta, d_existing,
                                rocm_device_id_, gpu_stream_);
                        }

                        if (!int8_decode_ok)
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] INT8 scatter GEMV failed");
                            return false;
                        }

                        // Bulk DMA from HBM workspace to output
                        if (gemv_output_needs_copyout)
                        {
                            hipMemcpyAsync(d_output, impl_->d_C_fp32,
                                           static_cast<size_t>(n) * sizeof(float),
                                           hipMemcpyDeviceToDevice,
                                           static_cast<hipStream_t>(gpu_stream_));
                        }
                        return true;
                    }

                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] M=1 decode has no supported GEMV path for alpha="
                              << alpha << " beta=" << beta
                              << " native_vnni=" << (impl_->has_native_vnni ? "true" : "false")
                              << " int8_vnni=" << (d_weights_vnni ? "true" : "false"));
                    return false;
                }
                // No VNNI weight pointers available for M=1 decode — this should
                // not happen unless weights failed to upload.
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] M=1 decode: "
                          "no native-VNNI or INT8-VNNI weights available");
                return false;
            }

            // =========================================================================
            // NATIVE VNNI PREFILL ROUTE (M>1)
            //
            // We defer actual native dispatch until after activation quantization,
            // because both native prefill formats consume INT8 activations.
            // =========================================================================

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

            // Native-VNNI path uses impl_->d_weights_native_scales (not d_scales_B),
            // so d_scales_B may be null when only native-VNNI weights are present
            // (e.g. weights loaded via GPU pipeline / MoE batch constructor).
            const bool native_vnni_available_mt = impl_ && impl_->has_native_vnni;
            if (!d_scales_B && !native_vnni_available_mt)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Weight scales not uploaded to device");
                return false;
            }

            if (d_scales_B && !validatePointerDeviceOrLog(
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

            // Option B: Repack VNNI→row-major was used only by the CK fallback.
            // CK is retired; the M>1 path dispatches native-VNNI / INT8-VNNI
            // directly against `impl_->d_weights_int8_vnni` and never needs a
            // row-major copy.

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

            // Quantize activations FP32 → INT8 (blockwise)
            // M=1 decode uses the GEMV path above with its own blockwise quantization.
            if (phase_timing)
                phase_start = std::chrono::high_resolution_clock::now();
            {
                const int act_block_k = rocmGemv_int8_vnni_get_act_block_k();
                if (!rocmQuantGemm_quantizeActivationsBlockwise(d_A_fp32_src, d_A_int8, impl_->d_scales_A_blockwise, m, k, rocm_device_id_, gpu_stream_, act_block_k))
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Failed to blockwise-quantize activations");
                    return false;
                }
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
                const bool output_needs_copyout = output_is_mapped;
                float *d_prefill_output = (use_gpu_path && !output_needs_copyout) ? d_output : impl_->d_C_fp32;

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
                    auto *bias_tensor = const_cast<TensorBase *>(bias);
                    const auto target_device = DeviceId::rocm(rocm_device_id_);
                    if (!bias_tensor->ensureOnDevice(target_device))
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Failed to upload prefill bias tensor to ROCm device");
                        return false;
                    }

                    d_prefill_bias = static_cast<const float *>(bias_tensor->gpu_data_ptr());
                }

                // Try native-VNNI GEMM first (halved HBM bandwidth for Q4_0/IQ4_NL)
                // This path reads compact native-VNNI payloads (4.5 bpw) directly,
                // decodes to INT8 in-register, and produces FP32 output with per-block
                // FP16 scales applied inline — no separate scaling epilogue needed.
                // Bias is applied via a lightweight biasAdd kernel after the GEMM.
                if (impl_->has_native_vnni && alpha == 1.0f && beta == 0.0f)
                {
                    const uint8_t cb_id = impl_->native_vnni_codebook_id;
                    {
                        if (rocmGemm_native_vnni_fp32(
                                d_A_int8,
                                impl_->d_weights_native_vnni,
                                impl_->d_weights_native_scales,
                                impl_->d_weights_native_mins,
                                impl_->d_weights_native_emins,
                                d_prefill_output,
                                d_scales_A,
                                impl_->d_scales_A_blockwise,
                                m, n, k,
                                cb_id,
                                rocm_device_id_, gpu_stream_))
                        {
                            LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_tensor] "
                                      "Native-VNNI GEMM succeeded (M="
                                      << m << " N=" << n << " K=" << k
                                      << " codebook=" << static_cast<int>(cb_id)
                                      << (d_prefill_bias ? " +bias" : "") << ")");

                            // Apply bias if present (same post-kernel pattern as GEMV)
                            if (d_prefill_bias)
                            {
                                if (!rocmQuantGemm_biasAdd(d_prefill_output, d_prefill_bias, m, n, rocm_device_id_, gpu_stream_))
                                {
                                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Native-VNNI GEMM bias add failed");
                                    return false;
                                }
                            }

                            // Copy from workspace to output if needed.
                            if (!use_gpu_path || output_needs_copyout)
                            {
                                const size_t nvnni_c_size = static_cast<size_t>(m) * n;
                                if (!use_gpu_path)
                                {
                                    float *host_dst = C_fp32->mutable_data();
                                    hipMemcpyAsync(host_dst, d_prefill_output,
                                                   nvnni_c_size * sizeof(float),
                                                   hipMemcpyDeviceToHost,
                                                   static_cast<hipStream_t>(gpu_stream_));
                                    hipStreamSynchronize(static_cast<hipStream_t>(gpu_stream_));
                                }
                                else if (output_is_mapped)
                                {
                                    float *host_dst = C_fp32->mutable_data();
                                    hipMemcpyAsync(host_dst, d_prefill_output,
                                                   nvnni_c_size * sizeof(float),
                                                   hipMemcpyDeviceToHost,
                                                   static_cast<hipStream_t>(gpu_stream_));
                                    hipStreamSynchronize(static_cast<hipStream_t>(gpu_stream_));
                                }
                                else
                                {
                                    hipMemcpyAsync(d_output, d_prefill_output,
                                                   nvnni_c_size * sizeof(float),
                                                   hipMemcpyDeviceToDevice,
                                                   static_cast<hipStream_t>(gpu_stream_));
                                }
                            }
                            return true;
                        }
                        static std::once_flag nvnni_gemm_tensor_fallback;
                        std::call_once(nvnni_gemm_tensor_fallback, [&]()
                                       { LOG_WARN("[ROCmQuantisedGemmKernel::multiply_tensor] "
                                                  "Native-VNNI GEMM failed; falling back to INT8 GEMM"); });
                    }
                }

                if (tryPrefillNativeGemm(
                        d_A_int8,
                        d_prefill_output,
                        d_scales_A,
                        impl_->d_scales_A_blockwise,
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

                // Both VNNI paths failed for M>1 prefill — hard error.
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] M>1 prefill: "
                          "both native-VNNI and INT8-VNNI paths failed (M="
                          << m << " N=" << n << " K=" << k << ").");
                return false;
            }

            // Unreachable: M>1 path always returns above via native/INT8-VNNI or hard error.
            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Reached unreachable CK fallback sentinel");
            return false;
        }

        bool ROCmQuantisedGemmKernel::multiply_tensor_timed(
            const TensorBase * /*A*/, TensorBase * /*C*/,
            int /*m*/, int /*n*/, int /*k*/,
            float *kernel_time_ms)
        {
            // CK timed path retired. The timed variant existed solely to measure
            // CK kernel execution time for benchmarking. Use multiply_tensor() for
            // functional execution via native-VNNI / INT8-VNNI paths.
            if (kernel_time_ms)
                *kernel_time_ms = -1.0f;
            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor_timed] CK timed path retired");
            return false;
        }

        bool ROCmQuantisedGemmKernel::multiply_fused_tensor(
            const TensorBase *input,
            const std::vector<TensorProjectionDesc> &projections,
            int m, int k,
            const IMPIContext *mpi_ctx,
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
            auto restore_workspace = [&](void *)
            {
                if (ws && ws != saved_workspace)
                {
                    workspace_ = saved_workspace;
                }
            };
            std::unique_ptr<void, decltype(restore_workspace)> workspace_restore_guard(
                (ws && ws != saved_workspace) ? static_cast<void *>(this) : nullptr,
                restore_workspace);

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
                d_input = static_cast<const float *>(fp32_input->gpu_data_ptr());
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

            // Step 3: Ensure all projection weights are on GPU before the native-VNNI
            // pre-scan. has_native_vnni is only set by ensureWeightsConverted(), so we
            // must call it on every projection kernel before checking the flag.
            for (size_t pi = 0; pi < projections.size(); ++pi)
            {
                auto *rk = dynamic_cast<ROCmQuantisedGemmKernel *>(projections[pi].kernel);
                if (rk)
                    rk->ensureWeightsConverted();
            }

            // Step 4: Quantize activations ONCE (shared across all projections)
            // All M=1 paths now use INT8 scatter kernels which require pre-quantized
            // activations. For M>1, CK GEMM also uses pre-quantized activations.
            //
            // Blockwise quantization is supported when ALL projection kernels support it.
            // - Native-VNNI (Q4_0/IQ4_NL): blockwise at any M.
            // - INT8-VNNI (Q8_0/Q8_1): blockwise for both decode and prefill.
            bool all_projections_native_vnni = true;
            bool all_projections_have_vnni_weights = true;
            for (size_t pi = 0; pi < projections.size(); ++pi)
            {
                auto *rk = dynamic_cast<ROCmQuantisedGemmKernel *>(projections[pi].kernel);
                if (!rk || !rk->impl_ || !rk->impl_->has_native_vnni)
                {
                    all_projections_native_vnni = false;
                }
                if (!rk || !rk->impl_ || (!rk->impl_->has_native_vnni && !rk->impl_->d_weights_int8_vnni))
                {
                    all_projections_have_vnni_weights = false;
                }
                if (!all_projections_native_vnni && !all_projections_have_vnni_weights)
                    break;
            }
            bool fused_uses_blockwise_shared_quant = false;
            {
                LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Quantizing activations once, m=" << m << " k=" << k);

                if (!rocmQuantGemm_quantizeActivationsBlockwise(
                        const_cast<float *>(d_input),
                        impl_->d_A_int8,
                        impl_->d_scales_A_blockwise,
                        m, k, rocm_device_id_, gpu_stream_))
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Blockwise activation quantization failed");
                    return false;
                }

                fused_uses_blockwise_shared_quant = all_projections_have_vnni_weights;

                if (m >= 2 && m <= 4 && projections.size() >= 2 && PerfStatsCollector::isEnabled())
                {
                    PerfStatsCollector::addCounter(
                        "kernel",
                        "rocm_fused_small_m_shared_quant_calls",
                        1.0,
                        "gemm",
                        "rocm:" + std::to_string(rocm_device_id_),
                        PerfStatsCollector::Tags{
                            {"m", std::to_string(m)},
                            {"k", std::to_string(k)},
                            {"projections", std::to_string(projections.size())}});

                    if (m == 2)
                    {
                        PerfStatsCollector::addCounter(
                            "kernel",
                            "rocm_fused_m2_shared_quant_calls",
                            1.0,
                            "gemm",
                            "rocm:" + std::to_string(rocm_device_id_),
                            PerfStatsCollector::Tags{
                                {"k", std::to_string(k)},
                                {"projections", std::to_string(projections.size())}});
                    }
                }
            }

            // Step 5: Execute projections using the SHARED quantized activations
            // For M=1, VNNI projections are batched into a single kernel dispatch.
            // Native-VNNI (Q4/IQ4) and M>1 projections are dispatched individually.
            bool all_success = true;

            // Batched INT8 scatter GEMV collection arrays (for M=1 VNNI projections)
            const int8_t *batch_B_ptrs[8] = {};
            float *batch_C_ptrs[8] = {};
            const float *batch_scales_B_ptrs[8] = {};
            const float *batch_bias_ptrs[8] = {};
            int batch_N[8] = {};
            int batch_count = 0;

            auto isGDNDecodeProjectionSet = [&]() -> bool
            {
                if (m != 1 || projections.size() != 4)
                    return false;

                constexpr std::array<const char *, 4> expected_names = {"qkv", "z", "alpha", "beta"};
                for (size_t i = 0; i < expected_names.size(); ++i)
                {
                    const char *name = projections[i].name;
                    if (!name || std::strcmp(name, expected_names[i]) != 0)
                        return false;
                }
                return true;
            };

            // =========================================================================
            // CONCURRENT PREFILL PATH: Multi-stream dispatch for M>1 projections
            //
            // When enabled (LLAMINAR_ROCM_CONCURRENT_PREFILL=1), overlaps fused GEMM
            // projections on separate HIP streams.  Each projection gets its own
            // stream + scratch buffer so blockwise V7 kernels can write intermediate
            // results without conflicting.
            //
            // Prerequisites:
            //   - M > 1 (prefill, not decode)
            //   - >= 2 projections
            //   - All projections have VNNI weights (tryPrefillNativeGemm path)
            //   - No mapped outputs (those need shared d_C_fp32)
            //
            // DAG: quantization (main stream)
            //        ├──> projection 0 (stream 0, scratch 0)
            //        ├──> projection 1 (stream 1, scratch 1)
            //        └──> ...
            //      main stream waits on all completion events
            // =========================================================================
#ifdef HAVE_ROCM
            const bool small_m_native_verifier_fused =
                all_projections_native_vnni && m >= 2 && m <= 4;
            if (m > 2 && projections.size() >= 2 &&
                fused_uses_blockwise_shared_quant &&
                !small_m_native_verifier_fused &&
                debugEnv().rocm.concurrent_prefill)
            {
                // Check eligibility: all outputs must be direct device memory (not BAR/mapped)
                bool concurrent_eligible = true;
                for (size_t i = 0; i < projections.size(); ++i)
                {
                    const auto &proj = projections[i];
                    if (!proj.kernel || !proj.output)
                    {
                        concurrent_eligible = false;
                        break;
                    }
                    auto *fp32_out = dynamic_cast<FP32Tensor *>(proj.output);
                    if (!fp32_out)
                    {
                        concurrent_eligible = false;
                        break;
                    }
                    if (fp32_out->isMapped())
                    {
                        concurrent_eligible = false;
                        break;
                    }
                    auto *rk = dynamic_cast<ROCmQuantisedGemmKernel *>(proj.kernel);
                    if (!rk || !rk->impl_)
                    {
                        concurrent_eligible = false;
                        break;
                    }
                }

                if (concurrent_eligible)
                {
                    const int num_proj = static_cast<int>(projections.size());
                    auto &pool = getSharedPrefillPool(rocm_device_id_, num_proj);

                    // Record event after quantization completes on main stream
                    hipEventRecord(pool.quant_ready,
                                   static_cast<hipStream_t>(gpu_stream_));

                    for (int pi = 0; pi < num_proj; ++pi)
                    {
                        const auto &proj = projections[pi];
                        auto *rocm_kernel = dynamic_cast<ROCmQuantisedGemmKernel *>(proj.kernel);
                        const int n = proj.n;

                        // Validate and prepare this projection's workspace
                        rocm_kernel->validateWorkspace();

                        // Get output pointer
                        auto *fp32_output = dynamic_cast<FP32Tensor *>(proj.output);
                        float *d_output = static_cast<float *>(fp32_output->gpu_data_ptr());

                        if (!d_output)
                        {
                            throw std::runtime_error(
                                "[ConcurrentPrefill] Projection " + std::to_string(pi) +
                                " output has no GPU data (shape=" +
                                std::to_string(fp32_output->rows()) + "x" +
                                std::to_string(fp32_output->cols()) +
                                ", coherence=" +
                                std::to_string(static_cast<int>(fp32_output->coherenceState())) +
                                ") — ensure scratch tensors call allocateOnDevice()");
                        }

                        // Get per-projection weight scales and bias
                        const float *d_scales_B = nullptr;
                        if (rocm_kernel->packed_)
                        {
                            d_scales_B = rocm_kernel->packed_->d_scales;
                        }
                        else if (rocm_kernel->impl_)
                        {
                            d_scales_B = rocm_kernel->impl_->d_scales_B;
                        }
                        if (!d_scales_B && !(rocm_kernel->impl_ && rocm_kernel->impl_->has_native_vnni))
                        {
                            LOG_WARN("[ConcurrentPrefill] Projection " << pi
                                                                       << " (" << (proj.name ? proj.name : "?")
                                                                       << ") d_scales_B=0 — packed_=" << (const void *)rocm_kernel->packed_
                                                                       << " packed_->d_scales=" << (rocm_kernel->packed_ ? (const void *)rocm_kernel->packed_->d_scales : nullptr)
                                                                       << " impl_=" << (const void *)rocm_kernel->impl_.get()
                                                                       << " impl_->d_scales_B=" << (rocm_kernel->impl_ ? (const void *)rocm_kernel->impl_->d_scales_B : nullptr)
                                                                       << " weights_converted=" << rocm_kernel->weights_converted_
                                                                       << " N=" << rocm_kernel->N_ << " K=" << rocm_kernel->K_
                                                                       << " device=" << rocm_kernel->rocm_device_id_);
                        }
                        const float *d_bias = nullptr;
                        if (proj.bias)
                        {
                            auto *bias_fp32 = dynamic_cast<FP32Tensor *>(const_cast<TensorBase *>(proj.bias));
                            if (bias_fp32)
                            {
                                d_bias = static_cast<const float *>(bias_fp32->gpu_data_ptr());
                            }
                        }

                        // Ensure scratch buffer for this stream
                        const size_t scratch_elements = static_cast<size_t>(m) * n;
                        int stream_idx = pi % pool.count;
                        if (!pool.ensureScratch(stream_idx, scratch_elements))
                        {
                            throw std::runtime_error(
                                "[ConcurrentPrefill] Failed to allocate scratch for projection " +
                                std::to_string(pi) + " (" + std::to_string(scratch_elements * sizeof(int32_t)) +
                                " bytes) — GPU OOM");
                        }

                        // This stream waits for quantization to complete
                        hipStreamWaitEvent(pool.streams[stream_idx], pool.quant_ready, 0);

                        // If projections > streams, wait for previous projection on this stream
                        if (pi >= pool.count)
                        {
                            hipStreamWaitEvent(pool.streams[stream_idx],
                                               pool.completion[stream_idx], 0);
                        }

                        // Dispatch with stream + scratch overrides
                        LOG_DEBUG("[ConcurrentPrefill] Projection " << pi
                                                                    << " (" << (proj.name ? proj.name : "?")
                                                                    << ") M=" << m << " N=" << n << " K=" << k
                                                                    << " on stream " << stream_idx);

                        bool proj_ok = rocm_kernel->tryPrefillNativeGemm(
                            impl_->d_A_int8,
                            d_output,
                            impl_->d_scales_A,
                            impl_->d_scales_A_blockwise,
                            d_scales_B,
                            d_bias,
                            m, n, k,
                            1.0f, 0.0f,
                            "ConcurrentPrefill",
                            pool.streams[stream_idx],
                            pool.scratch[stream_idx]);

                        if (!proj_ok)
                        {
                            throw std::runtime_error(
                                "[ConcurrentPrefill] Projection " + std::to_string(pi) +
                                " (" + std::string(proj.name ? proj.name : "?") +
                                ") kernel launch failed on stream " + std::to_string(stream_idx) +
                                " — cannot continue inference");
                        }

                        // Record completion event for this stream
                        hipEventRecord(pool.completion[stream_idx],
                                       pool.streams[stream_idx]);
                    }

                    // All projections dispatched — main stream waits for completion
                    for (int si = 0; si < std::min(num_proj, pool.count); ++si)
                    {
                        hipStreamWaitEvent(
                            static_cast<hipStream_t>(gpu_stream_),
                            pool.completion[si], 0);
                    }

                    LOG_DEBUG("[ConcurrentPrefill] All " << num_proj
                                                         << " projections dispatched concurrently");

                    // Restore workspace and return success
                    if (ws && ws != saved_workspace)
                        workspace_ = saved_workspace;
                    return true;
                }
            }
#endif // HAVE_ROCM

            // =========================================================================
            // CONCURRENT DECODE PATH: Multi-stream dispatch for M=1 GEMV projections
            //
            // When enabled globally (LLAMINAR_ROCM_CONCURRENT_DECODE=1) or for a
            // scoped projection set (GDN), overlaps fused GEMV
            // projections on separate HIP streams.  At small TP-sharded N dimensions,
            // individual GEMVs may not saturate all CUs, so overlapping them can
            // improve utilization.
            //
            // INT8-VNNI projections use rocmGemv_int8_int8_fp32_vnni_blockwise_scaled
            //   which has no shared scratch dependency (atomicAdd to output).
            // Native-VNNI projections use rocmGemv_native_vnni_fp32 which needs a
            //   per-stream scatter_partial buffer from the pool.
            //
            // DAG: quantization (main stream)
            //        ├──> projection 0 GEMV (stream 0)
            //        ├──> projection 1 GEMV (stream 1)
            //        └──> ...
            //      main stream waits on all completion events
            // =========================================================================
#ifdef HAVE_ROCM
            const bool use_concurrent_decode =
                debugEnv().rocm.concurrent_decode ||
                (debugEnv().rocm.gdn_concurrent_decode && isGDNDecodeProjectionSet());

            if (m == 1 && projections.size() >= 2 &&
                fused_uses_blockwise_shared_quant &&
                use_concurrent_decode)
            {
                bool decode_concurrent_eligible = true;
                for (size_t i = 0; i < projections.size(); ++i)
                {
                    const auto &proj = projections[i];
                    if (!proj.kernel || !proj.output)
                    {
                        decode_concurrent_eligible = false;
                        break;
                    }
                    auto *fp32_out = dynamic_cast<FP32Tensor *>(proj.output);
                    if (!fp32_out)
                    {
                        decode_concurrent_eligible = false;
                        break;
                    }
                    // Mapped outputs use shared d_C_fp32 — cannot overlap
                    if (fp32_out->isMapped())
                    {
                        decode_concurrent_eligible = false;
                        break;
                    }
                    auto *rk = dynamic_cast<ROCmQuantisedGemmKernel *>(proj.kernel);
                    if (!rk || !rk->impl_)
                    {
                        decode_concurrent_eligible = false;
                        break;
                    }
                }

                if (decode_concurrent_eligible)
                {
                    const int num_proj = static_cast<int>(projections.size());
                    auto &pool = getSharedPrefillPool(rocm_device_id_, num_proj);

                    // Record event after quantization completes on main stream
                    hipEventRecord(pool.quant_ready,
                                   static_cast<hipStream_t>(gpu_stream_));

                    for (int pi = 0; pi < num_proj; ++pi)
                    {
                        const auto &proj = projections[pi];
                        auto *rocm_kernel = dynamic_cast<ROCmQuantisedGemmKernel *>(proj.kernel);
                        const int n = proj.n;

                        auto *fp32_output = dynamic_cast<FP32Tensor *>(proj.output);
                        float *d_output = static_cast<float *>(fp32_output->gpu_data_ptr());

                        if (!d_output)
                        {
                            throw std::runtime_error(
                                "[ConcurrentDecode] Projection " + std::to_string(pi) +
                                " output has no GPU data — cannot continue inference");
                        }

                        const float *d_scales_B = rocm_kernel->impl_->d_scales_B;
                        const float *d_bias = nullptr;
                        if (proj.bias)
                        {
                            auto *bias_fp32 = dynamic_cast<FP32Tensor *>(const_cast<TensorBase *>(proj.bias));
                            if (bias_fp32)
                            {
                                d_bias = static_cast<const float *>(bias_fp32->gpu_data_ptr());
                            }
                        }

                        int stream_idx = pi % pool.count;

                        // This stream waits for quantization
                        hipStreamWaitEvent(pool.streams[stream_idx], pool.quant_ready, 0);

                        // If reusing a stream, wait for its previous work
                        if (pi >= pool.count)
                        {
                            hipStreamWaitEvent(pool.streams[stream_idx],
                                               pool.completion[stream_idx], 0);
                        }

                        bool proj_ok = false;

                        if (rocm_kernel->impl_->has_native_vnni)
                        {
                            // Native-VNNI GEMV: needs per-stream scatter_partial
                            constexpr int SCATTER_KB_MAX = 64;
                            const size_t partial_elements = static_cast<size_t>(SCATTER_KB_MAX) * n;
                            if (!pool.ensureScatterPartial(stream_idx, partial_elements))
                            {
                                throw std::runtime_error(
                                    "[ConcurrentDecode] Failed to allocate scatter_partial for projection " +
                                    std::to_string(pi) + " (" + std::to_string(partial_elements * sizeof(float)) +
                                    " bytes) — GPU OOM");
                            }

                            proj_ok = rocmGemv_native_vnni_fp32(
                                impl_->d_A_int8,
                                rocm_kernel->impl_->d_weights_native_vnni,
                                rocm_kernel->impl_->d_weights_native_scales,
                                rocm_kernel->impl_->d_weights_native_mins,
                                rocm_kernel->impl_->d_weights_native_emins,
                                d_output,
                                impl_->d_scales_A,
                                pool.scatter_partial[stream_idx],
                                n, k,
                                rocm_kernel->impl_->native_vnni_codebook_id,
                                rocm_device_id_, pool.streams[stream_idx],
                                fused_uses_blockwise_shared_quant ? impl_->d_scales_A_blockwise : nullptr);
                        }
                        else
                        {
                            // INT8-VNNI GEMV: no shared scratch needed
                            int8_t *d_vnni = rocm_kernel->impl_->d_weights_int8_vnni;
                            if (!d_vnni)
                            {
                                throw std::runtime_error(
                                    "[ConcurrentDecode] Projection " + std::to_string(pi) +
                                    " has no INT8-VNNI weights — kernel not prepared");
                            }

                            proj_ok = rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
                                impl_->d_A_int8, d_vnni, d_output,
                                impl_->d_scales_A_blockwise, d_scales_B,
                                n, k,
                                1.0f, 0.0f,
                                nullptr, d_bias,
                                rocm_device_id_, pool.streams[stream_idx]);
                        }

                        if (!proj_ok)
                        {
                            throw std::runtime_error(
                                "[ConcurrentDecode] Projection " + std::to_string(pi) +
                                " GEMV kernel launch failed on stream " + std::to_string(pi % pool.count) +
                                " — cannot continue inference");
                        }

                        if (rocm_kernel->impl_->has_native_vnni && d_bias)
                        {
                            if (!rocmQuantGemm_biasAdd(d_output, d_bias, m, n,
                                                       rocm_device_id_, pool.streams[stream_idx]))
                            {
                                throw std::runtime_error(
                                    "[ConcurrentDecode] Bias add failed for projection " +
                                    std::to_string(pi) + " on stream " + std::to_string(stream_idx));
                            }
                        }

                        hipEventRecord(pool.completion[stream_idx],
                                       pool.streams[stream_idx]);
                    }

                    // All projections dispatched — main stream waits for completion
                    for (int si = 0; si < std::min(num_proj, pool.count); ++si)
                    {
                        hipStreamWaitEvent(
                            static_cast<hipStream_t>(gpu_stream_),
                            pool.completion[si], 0);
                    }

                    LOG_DEBUG("[ConcurrentDecode] All " << num_proj
                                                        << " projections dispatched concurrently");

                    if (ws && ws != saved_workspace)
                        workspace_ = saved_workspace;
                    return true;
                }
            }
#endif // HAVE_ROCM

#ifdef HAVE_ROCM
            if (small_m_native_verifier_fused &&
                projections.size() >= 2 &&
                projections.size() <= 8)
            {
                std::array<const uint8_t *, 8> payloads{};
                std::array<const uint16_t *, 8> scales{};
                std::array<const uint16_t *, 8> mins{};
                std::array<const uint32_t *, 8> emins{};
                std::array<float *, 8> outputs{};
                std::array<float *, 8> partials{};
                std::array<const float *, 8> biases{};
                std::array<int, 8> Ns{};

                bool batched_eligible = true;
                std::string batched_bypass_reason;
                uint8_t common_codebook = 0;
                bool have_common_codebook = false;

                auto mark_batched_bypass = [&](std::string reason)
                {
                    if (batched_bypass_reason.empty())
                        batched_bypass_reason = std::move(reason);
                    batched_eligible = false;
                };

                for (size_t i = 0; i < projections.size(); ++i)
                {
                    const auto &proj = projections[i];
                    auto *rocm_kernel = dynamic_cast<ROCmQuantisedGemmKernel *>(proj.kernel);
                    auto *fp32_output = dynamic_cast<FP32Tensor *>(proj.output);
                    if (!rocm_kernel || !rocm_kernel->impl_ || !rocm_kernel->impl_->has_native_vnni)
                    {
                        mark_batched_bypass("non_native_projection");
                        break;
                    }
                    if (!fp32_output)
                    {
                        mark_batched_bypass("non_fp32_output");
                        break;
                    }
                    if (fp32_output->isMapped())
                    {
                        mark_batched_bypass("mapped_output");
                        break;
                    }
                    if (!rocm_kernel->impl_->d_weights_native_vnni ||
                        !rocm_kernel->impl_->d_weights_native_scales ||
                        !rocm_kernel->impl_->d_scatter_partial)
                    {
                        mark_batched_bypass("missing_native_buffers");
                        break;
                    }

                    const uint8_t codebook = rocm_kernel->impl_->native_vnni_codebook_id;
                    if (!have_common_codebook)
                    {
                        common_codebook = codebook;
                        have_common_codebook = true;
                    }
                    else if (common_codebook != codebook)
                    {
                        mark_batched_bypass("mixed_codebook");
                        break;
                    }

                    float *d_output = static_cast<float *>(fp32_output->gpu_data_ptr());
                    if (!d_output)
                    {
                        mark_batched_bypass("missing_output_device_ptr");
                        break;
                    }

                    const float *d_bias = nullptr;
                    if (proj.bias)
                    {
                        if (proj.bias->native_type() != TensorType::FP32)
                        {
                            mark_batched_bypass("non_fp32_bias");
                            break;
                        }

                        auto *bias_tensor = const_cast<TensorBase *>(proj.bias);
                        const DeviceId target_device = DeviceId::rocm(rocm_device_id_);
                        const auto current_dev = bias_tensor->current_device();
                        if (current_dev.has_value() && current_dev.value() == target_device)
                        {
                            d_bias = static_cast<const float *>(bias_tensor->gpu_data_ptr());
                        }
                        else if (current_dev.has_value() && current_dev->is_gpu())
                        {
                            mark_batched_bypass("bias_wrong_gpu");
                            break;
                        }
                        else
                        {
                            if (!bias_tensor->ensureOnDevice(target_device))
                            {
                                mark_batched_bypass("bias_upload_failed");
                                break;
                            }
                            d_bias = static_cast<const float *>(bias_tensor->gpu_data_ptr());
                        }

                        if (!d_bias)
                        {
                            mark_batched_bypass("missing_bias_device_ptr");
                            break;
                        }
                    }

                    payloads[i] = rocm_kernel->impl_->d_weights_native_vnni;
                    scales[i] = static_cast<const uint16_t *>(rocm_kernel->impl_->d_weights_native_scales);
                    mins[i] = static_cast<const uint16_t *>(rocm_kernel->impl_->d_weights_native_mins);
                    emins[i] = static_cast<const uint32_t *>(rocm_kernel->impl_->d_weights_native_emins);
                    outputs[i] = d_output;
                    partials[i] = rocm_kernel->impl_->d_scatter_partial;
                    biases[i] = d_bias;
                    Ns[i] = proj.n;
                }

                if (batched_eligible)
                {
                    if (!rocmGemv_native_vnni_small_m_batched_fp32(
                            impl_->d_A_int8,
                            payloads.data(),
                            scales.data(),
                            mins.data(),
                            emins.data(),
                            biases.data(),
                            outputs.data(),
                            impl_->d_scales_A_blockwise,
                            partials.data(),
                            Ns.data(),
                            static_cast<int>(projections.size()),
                            m, k,
                            common_codebook,
                            rocm_device_id_,
                            gpu_stream_))
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Native-VNNI fused small-M batched GEMV failed"
                                  << " m=" << m << " k=" << k
                                  << " projections=" << projections.size()
                                  << " codebook=" << static_cast<int>(common_codebook));
                        return false;
                    }

                    if (PerfStatsCollector::isEnabled())
                    {
                        PerfStatsCollector::addCounter(
                            "kernel",
                            "rocm_native_vnni_small_m_batched_calls",
                            1.0,
                            "gemm",
                            "rocm:" + std::to_string(rocm_device_id_),
                            PerfStatsCollector::Tags{
                                {"m", std::to_string(m)},
                                {"codebook", std::to_string(static_cast<int>(common_codebook))},
                                {"k", std::to_string(k)},
                                {"projections", std::to_string(projections.size())}});

                        for (const int n_value : Ns)
                        {
                            if (n_value <= 0)
                                continue;
                            PerfStatsCollector::addCounter(
                                "kernel",
                                "rocm_native_vnni_small_m_batched_projection_calls",
                                1.0,
                                "gemm",
                                "rocm:" + std::to_string(rocm_device_id_),
                                PerfStatsCollector::Tags{
                                    {"m", std::to_string(m)},
                                    {"codebook", std::to_string(static_cast<int>(common_codebook))},
                                    {"n", std::to_string(n_value)},
                                    {"k", std::to_string(k)}});
                        }
                    }

                    LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Batched native small-M verifier complete"
                              << " M=" << m << " K=" << k
                              << " projections=" << projections.size());
                    return true;
                }

                if (PerfStatsCollector::isEnabled() && !batched_bypass_reason.empty())
                {
                    PerfStatsCollector::addCounter(
                        "kernel",
                        "rocm_native_vnni_small_m_batched_bypasses",
                        1.0,
                        "gemm",
                        "rocm:" + std::to_string(rocm_device_id_),
                        PerfStatsCollector::Tags{
                            {"m", std::to_string(m)},
                            {"k", std::to_string(k)},
                            {"projections", std::to_string(projections.size())},
                            {"reason", batched_bypass_reason}});
                }
            }
#endif // HAVE_ROCM

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

                // Weights already converted in Step 3 (ensureWeightsConverted loop).
                // Validate this projection's workspace is bound and populated.
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
                float *d_output = static_cast<float *>(fp32_output->gpu_data_ptr());
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

                // Native-VNNI path uses impl_->d_weights_native_scales (not d_scales_B),
                // so d_scales_B may be null when only native-VNNI weights are present.
                const bool native_vnni_fused = rocm_kernel->impl_ && rocm_kernel->impl_->has_native_vnni;
                if (!d_scales_B && !native_vnni_fused)
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
                        d_bias = static_cast<const float *>(bias_tensor->gpu_data_ptr());
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
                        d_bias = static_cast<const float *>(bias_tensor->gpu_data_ptr());
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
                        const bool output_is_mapped = fp32_output->isMapped();
                        const bool output_needs_copyout = output_is_mapped;
                        float *d_native_output = output_needs_copyout ? impl_->d_C_fp32 : d_output;

                        // Activations are always pre-quantized above (Step 3)
                        LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                                 << " NATIVE-VNNI GEMV M=1 N=" << n << " K=" << k
                                                                                                 << (d_bias ? " +bias" : ""));

                        if (!rocmGemv_native_vnni_fp32(
                                impl_->d_A_int8,
                                rocm_kernel->impl_->d_weights_native_vnni,
                                rocm_kernel->impl_->d_weights_native_scales,
                                rocm_kernel->impl_->d_weights_native_mins,
                                rocm_kernel->impl_->d_weights_native_emins,
                                d_native_output,
                                impl_->d_scales_A,
                                impl_->d_scatter_partial,
                                n, k,
                                rocm_kernel->impl_->native_vnni_codebook_id,
                                rocm_device_id_, gpu_stream_,
                                fused_uses_blockwise_shared_quant ? impl_->d_scales_A_blockwise : nullptr))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Native-VNNI GEMV failed for projection " << i);
                            all_success = false;
                            break;
                        }

                        if (d_bias)
                        {
                            if (!rocmQuantGemm_biasAdd(d_native_output, d_bias, m, n, rocm_device_id_, gpu_stream_))
                            {
                                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Native-VNNI bias add failed for projection " << i);
                                all_success = false;
                                break;
                            }
                        }

                        // Copy back from workspace to d_output
                        if (output_needs_copyout)
                        {
                            const size_t output_bytes = static_cast<size_t>(m) * n * sizeof(float);
                            if (output_is_mapped)
                            {
                                float *host_dst = fp32_output->mutable_data();
                                hipMemcpyAsync(host_dst, d_native_output,
                                               output_bytes,
                                               hipMemcpyDeviceToHost,
                                               static_cast<hipStream_t>(gpu_stream_));
                                hipStreamSynchronize(static_cast<hipStream_t>(gpu_stream_));
                            }
                            else
                            {
                                hipMemcpyAsync(d_output, d_native_output,
                                               output_bytes,
                                               hipMemcpyDeviceToDevice,
                                               static_cast<hipStream_t>(gpu_stream_));
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
                        bool projection_ok = rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
                            impl_->d_A_int8, d_vnni, d_output,
                            impl_->d_scales_A_blockwise, d_scales_B,
                            n, k,
                            1.0f, 0.0f,
                            nullptr, d_bias,
                            rocm_device_id_, gpu_stream_);

                        if (!projection_ok)
                        {
                            projection_ok = rocmGemv_int8_scatter_vnni_blockwise(
                                impl_->d_A_int8, d_vnni, d_output,
                                impl_->d_scales_A_blockwise, d_scales_B, d_bias,
                                rocm_kernel->impl_->d_scatter_partial,
                                n, k, 1.0f, 0.0f, nullptr,
                                rocm_device_id_, gpu_stream_);
                        }

                        if (!projection_ok)
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

                const bool output_is_mapped = fp32_output->isMapped();
                const bool output_needs_copyout = output_is_mapped;
                float *d_prefill_output = output_needs_copyout ? impl_->d_C_fp32 : d_output;

                const bool supports_native_small_m =
                    native_vnni_fused && m >= 2 && m <= 4;
                if (supports_native_small_m)
                {
                    if ((k % 32) != 0)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Native-VNNI fused small-M verifier requires K multiple of 32, got M="
                                  << m << " K=" << k);
                        all_success = false;
                        break;
                    }
                    if (!impl_->d_A_int8 || !impl_->d_scales_A_blockwise ||
                        !rocm_kernel->impl_->d_weights_native_vnni ||
                        !rocm_kernel->impl_->d_weights_native_scales ||
                        !rocm_kernel->impl_->d_scatter_partial)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Native-VNNI fused small-M verifier missing graph-native buffers"
                                  << " A=" << static_cast<const void *>(impl_->d_A_int8)
                                  << " scale_A_blockwise=" << static_cast<const void *>(impl_->d_scales_A_blockwise)
                                  << " weights=" << static_cast<const void *>(rocm_kernel->impl_->d_weights_native_vnni)
                                  << " scales=" << static_cast<const void *>(rocm_kernel->impl_->d_weights_native_scales)
                                  << " partial=" << static_cast<const void *>(rocm_kernel->impl_->d_scatter_partial));
                        all_success = false;
                        break;
                    }

                    if (!rocmGemv_native_vnni_small_m_fp32(
                            impl_->d_A_int8,
                            rocm_kernel->impl_->d_weights_native_vnni,
                            rocm_kernel->impl_->d_weights_native_scales,
                            rocm_kernel->impl_->d_weights_native_mins,
                            rocm_kernel->impl_->d_weights_native_emins,
                            d_prefill_output,
                            impl_->d_scales_A_blockwise,
                            rocm_kernel->impl_->d_scatter_partial,
                            m, n, k,
                            rocm_kernel->impl_->native_vnni_codebook_id,
                            rocm_device_id_, gpu_stream_))
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Native-VNNI fused small-M verifier GEMV failed for projection "
                                  << i);
                        all_success = false;
                        break;
                    }

                    if (d_bias)
                    {
                        if (!rocmQuantGemm_applyFp32Epilogue(
                                d_prefill_output,
                                d_bias,
                                m,
                                n,
                                1.0f,
                                rocm_device_id_,
                                gpu_stream_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Native-VNNI fused small-M epilogue failed for projection "
                                      << i);
                            all_success = false;
                            break;
                        }
                    }

                    if (output_needs_copyout)
                    {
                        const size_t output_bytes = static_cast<size_t>(m) * n * sizeof(float);
                        if (output_is_mapped)
                        {
                            float *host_dst = fp32_output->mutable_data();
                            hipMemcpyAsync(host_dst, d_prefill_output,
                                           output_bytes,
                                           hipMemcpyDeviceToHost,
                                           static_cast<hipStream_t>(gpu_stream_));
                            hipStreamSynchronize(static_cast<hipStream_t>(gpu_stream_));
                        }
                        else
                        {
                            hipMemcpyAsync(d_output, d_prefill_output,
                                           output_bytes,
                                           hipMemcpyDeviceToDevice,
                                           static_cast<hipStream_t>(gpu_stream_));
                        }
                    }

                    if (PerfStatsCollector::isEnabled())
                    {
                        PerfStatsCollector::addCounter(
                            "kernel",
                            "rocm_native_vnni_small_m_calls",
                            1.0,
                            "gemm",
                            "rocm:" + std::to_string(rocm_device_id_),
                            PerfStatsCollector::Tags{
                                {"m", std::to_string(m)},
                                {"codebook", std::to_string(static_cast<int>(rocm_kernel->impl_->native_vnni_codebook_id))},
                                {"n", std::to_string(n)},
                                {"k", std::to_string(k)},
                                {"shared_quant", "true"}});

                        if (m == 2)
                        {
                            PerfStatsCollector::addCounter(
                                "kernel",
                                "rocm_native_vnni_m2_calls",
                                1.0,
                                "gemm",
                                "rocm:" + std::to_string(rocm_device_id_),
                                PerfStatsCollector::Tags{
                                    {"codebook", std::to_string(static_cast<int>(rocm_kernel->impl_->native_vnni_codebook_id))},
                                    {"n", std::to_string(n)},
                                    {"k", std::to_string(k)},
                                    {"shared_quant", "true"}});
                        }
                    }

                    LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                             << " fused native small-M verifier complete");
                    continue;
                }

                if (rocm_kernel->tryPrefillNativeGemm(
                        impl_->d_A_int8,
                        d_prefill_output,
                        impl_->d_scales_A,
                        impl_->d_scales_A_blockwise,
                        d_scales_B,
                        d_bias,
                        m, n, k,
                        1.0f, 0.0f,
                        "ROCmQuantisedGemmKernel::multiply_fused_tensor"))
                {
                    // Copy back from workspace to d_output
                    if (output_needs_copyout)
                    {
                        const size_t output_bytes = static_cast<size_t>(m) * n * sizeof(float);
                        if (output_is_mapped)
                        {
                            float *host_dst = fp32_output->mutable_data();
                            hipMemcpyAsync(host_dst, d_prefill_output,
                                           output_bytes,
                                           hipMemcpyDeviceToHost,
                                           static_cast<hipStream_t>(gpu_stream_));
                            hipStreamSynchronize(static_cast<hipStream_t>(gpu_stream_));
                        }
                        else
                        {
                            hipMemcpyAsync(d_output, d_prefill_output,
                                           output_bytes,
                                           hipMemcpyDeviceToDevice,
                                           static_cast<hipStream_t>(gpu_stream_));
                        }
                    }

                    LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                             << " native prefill complete");
                    continue;
                }

                // Native prefill failed and CK is retired — hard error.
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                         << " native-VNNI prefill failed (m=" << m << " n=" << n << " k=" << k
                                                                                         << "); CK fallback has been removed.");
                all_success = false;
                break;
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

                bool batched_ok = false;
                if (fused_uses_blockwise_shared_quant)
                {
                    batched_ok = rocmGemv_int8_int8_fp32_vnni_blockwise_scaled_batched(
                        impl_->d_A_int8,
                        impl_->d_scales_A_blockwise,
                        batch_count,
                        batch_B_ptrs,
                        batch_C_ptrs,
                        batch_scales_B_ptrs,
                        batch_bias_ptrs,
                        batch_N,
                        k,
                        1.0f, 0.0f,
                        rocm_device_id_, gpu_stream_);

                    if (!batched_ok)
                    {
                        batched_ok = rocmGemv_int8_scatter_batched_vnni_blockwise(
                            impl_->d_A_int8,
                            impl_->d_scales_A_blockwise,
                            impl_->d_scatter_partial,
                            batch_count,
                            batch_B_ptrs,
                            batch_C_ptrs,
                            batch_scales_B_ptrs,
                            batch_bias_ptrs,
                            batch_N,
                            k,
                            1.0f, 0.0f,
                            rocm_device_id_, gpu_stream_);
                    }
                }
                else
                {
                    batched_ok = rocmGemv_int8_scatter_batched_vnni(
                        impl_->d_A_int8,
                        impl_->d_scales_A,
                        impl_->d_scatter_partial,
                        batch_count,
                        batch_B_ptrs,
                        batch_C_ptrs,
                        batch_scales_B_ptrs,
                        batch_bias_ptrs,
                        batch_N,
                        k,
                        1.0f, 0.0f,
                        rocm_device_id_, gpu_stream_);
                }

                if (!batched_ok)
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
            const IMPIContext *mpi_ctx,
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
            const IMPIContext *mpi_ctx,
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

            // Blockwise activation scales: [M × ceil(K/32)] for blockwise quantization mode
            const int blocks_per_row = (k + 31) / 32;
            size_t scales_a_blockwise_bytes = static_cast<size_t>(m) * blocks_per_row * sizeof(float);

            // Also need FP32 temp buffers for host→device transfer
            size_t temp_a_fp32_bytes = static_cast<size_t>(m) * k * sizeof(float);
            size_t temp_c_fp32_bytes = static_cast<size_t>(m) * n * sizeof(float);

            reqs.buffers.push_back({GemmWorkspaceBuffers::QUANT_A, quant_a_bytes, 256, true});
            reqs.buffers.push_back({GemmWorkspaceBuffers::SCALES_A, scales_a_bytes, 256, true});
            reqs.buffers.push_back({GemmWorkspaceBuffers::SCALES_A_BLOCKWISE, scales_a_blockwise_bytes, 256, true});
            reqs.buffers.push_back({GemmWorkspaceBuffers::ACC_INT32, acc_int32_bytes, 256, true});
            // Shared buffer names (matching CUDA).  Concurrent paths use
            // ConcurrentPrefillPool's per-stream scratch — these workspace
            // buffers are only used in serial execution paths.
            reqs.buffers.push_back({GemmWorkspaceBuffers::TEMP_A_FP32, temp_a_fp32_bytes, 256, true});
            reqs.buffers.push_back({GemmWorkspaceBuffers::TEMP_C_FP32, temp_c_fp32_bytes, 256, true});

            // NOTE: CK (ComposableKernel) workspace buffers (ROCM_CK_INT32,
            // ROCM_A_PADDED, ROCM_SCALE_A_PADDED, ROCM_E_PADDED, ROCM_B_REPACK)
            // are intentionally NOT requested. The CK dispatch path is being
            // retired — all quantized prefill now routes through native-VNNI
            // (≤6-bit) or INT8-VNNI (8-bit) kernels, neither of which needs
            // the CK scratch. The legacy CK dispatch code still exists in this
            // file but is no longer reachable in normal execution. This saves
            // up to N×K bytes (≈1.27 GB for a Q4_K LM head).

            // Scatter+reduce partial buffer: KB_MAX × rows × N × sizeof(float).
            // The graph-safe native-VNNI M=2/3/4 verifier route stores every
            // verifier row in one split-K partial buffer; serial M=1 paths use one row.
            // KB_MAX=64 is the maximum k-blocks the scatter dispatch can produce.
            // The workspace manager takes max across all kernel instances, so
            // the largest N (LM Head: 152064) determines the actual allocation.
            // M=4 LM head size: 64 × 4 × 152064 × 4 ≈ 149 MB — reused across layers.
            constexpr int SCATTER_KB_MAX = 64;
            const int scatter_rows = (m >= 2 && m <= 4) ? m : 1;
            size_t scatter_partial_bytes = static_cast<size_t>(SCATTER_KB_MAX) *
                                           static_cast<size_t>(scatter_rows) *
                                           static_cast<size_t>(n) * sizeof(float);
            reqs.buffers.push_back({scatterPartialBufferName(), scatter_partial_bytes, 256, true});

            LOG_DEBUG("[ROCmQuantisedGemmKernel::getWorkspaceRequirements] INT8 path: "
                      << "quant_a=" << (quant_a_bytes / 1024) << "KB, "
                      << "scales_a=" << (scales_a_bytes) << "B, "
                      << "scales_a_blockwise=" << (scales_a_blockwise_bytes) << "B"
                      << " (blocks_per_row=" << blocks_per_row << "), "
                      << "acc=" << (acc_int32_bytes / 1024) << "KB");

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

                    // Upload only VNNI layout + scales to device.
                    rocmQuantGemm_setDevice(rocm_device_id_);

                    // Create a dedicated H2D upload stream for this weight.
                    // Using an explicit stream instead of the NULL stream avoids
                    // contention when multiple upload threads work on the same
                    // device concurrently, and the async path's pinned staging
                    // (hipHostMalloc + memcpy + hipMemcpyAsync) naturally works
                    // around ROCm hipMemcpy failures on pageable brk-heap memory.
                    bool async_upload_enabled = false;
                    hipStream_t h2d_upload_stream = nullptr;
#ifdef HAVE_ROCM
                    {
                        hipError_t stream_err = hipStreamCreateWithFlags(&h2d_upload_stream, hipStreamNonBlocking);
                        if (stream_err == hipSuccess)
                        {
                            upload.startup_h2d_stream = reinterpret_cast<void *>(h2d_upload_stream);
                            async_upload_enabled = true;
                        }
                        else
                        {
                            LOG_WARN("[ROCmQuantisedGemmKernel] Failed to create H2D stream: "
                                     << hipGetErrorString(stream_err) << "; falling back to sync uploads");
                        }
                    }
#endif

                    auto cleanup_startup_async_resources = [&upload, &h2d_upload_stream]()
                    {
#ifdef HAVE_ROCM
                        freeStartupPinnedStaging(upload);
                        if (upload.startup_h2d_stream)
                        {
                            hipStreamDestroy(reinterpret_cast<hipStream_t>(upload.startup_h2d_stream));
                        }
                        upload.startup_h2d_stream = nullptr;
                        h2d_upload_stream = nullptr;
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
                                          << hipGetErrorString(err)
                                          << " (device=" << rocm_device_id_
                                          << " N=" << packed_->N << " K=" << packed_->K
                                          << " vnni_size=" << packed_->int8_data_vnni.size()
                                          << " src=" << (void *)packed_->int8_data_vnni.data()
                                          << " uploads=" << packed_->device_uploads.size() << ")");
                                return;
                            }

                            LOG_DEBUG("[ROCmQuantisedGemmKernel] Uploaded VNNI weights (Option B, sole device copy): "
                                      << (packed_->int8_data_vnni.size() / 1024) << " KB");
                        }

                        // Row-major weights are NOT uploaded here.  The CK prefill
                        // path repacks VNNI→row-major on-demand into the shared
                        // workspace scratch buffer (d_B_rowmajor_scratch), which
                        // reuses a single allocation for all GEMM calls.  Uploading
                        // a persistent row-major copy per weight would double GPU
                        // memory usage (VNNI + row-major) and OOM on large models.

                        // Upload native-VNNI payload + scales + mins (Q4_0, Q4_1, Q5_0, Q5_1,
                        // Q6_K, Q3_K, Q2_K, Q4_K, Q5_K, IQ4_NL, IQ4_XS, IQ3_S, IQ3_XXS,
                        // IQ2_S, IQ2_XS, IQ2_XXS, IQ1_S, IQ1_M)
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
                                    &upload.startup_h2d_pinned_native_vnni,
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

                            // Allocate FP16 mins buffer (asymmetric formats: Q4_1, Q5_1)
                            if (upload.d_native_vnni_payload && !packed_->native_vnni_mins.empty())
                            {
                                const size_t mins_bytes = packed_->native_vnni_mins.size() * sizeof(uint16_t);
                                void *d_mins_tmp = nullptr;
#ifdef HAVE_ROCM
                                hipError_t alloc_err = hipMalloc(&d_mins_tmp, mins_bytes);
                                if (alloc_err != hipSuccess)
                                {
                                    LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to alloc native-VNNI mins: "
                                              << hipGetErrorString(alloc_err));
                                    d_mins_tmp = nullptr;
                                }
#endif
                                upload.d_native_vnni_mins = d_mins_tmp;
                                if (d_mins_tmp)
                                {
                                    err = startupMemcpyAsyncOrSync(
                                        d_mins_tmp,
                                        packed_->native_vnni_mins.data(),
                                        mins_bytes,
                                        async_upload_enabled,
                                        upload.startup_h2d_stream,
                                        &upload.startup_h2d_pinned_native_mins,
                                        "ROCmQuantisedGemmKernel::ensureWeightsConverted",
                                        "native_vnni_mins");
                                    if (err != hipSuccess)
                                    {
#ifdef HAVE_ROCM
                                        hipFree(d_mins_tmp);
#endif
                                        upload.d_native_vnni_mins = nullptr;
                                        LOG_WARN("[ROCmQuantisedGemmKernel] Failed to upload native-VNNI mins: "
                                                 << hipGetErrorString(err));
                                    }
                                    else
                                    {
                                        LOG_DEBUG("[ROCmQuantisedGemmKernel] Uploaded native-VNNI mins="
                                                  << (mins_bytes / 1024) << " KB");
                                    }
                                }
                            }

                            // Allocate uint32 emins buffer (Q2_K only: packed {emin_lo, emin_hi} FP16)
                            if (upload.d_native_vnni_payload && !packed_->native_vnni_emins.empty())
                            {
                                const size_t emins_bytes = packed_->native_vnni_emins.size() * sizeof(uint32_t);
                                void *d_emins_tmp = nullptr;
#ifdef HAVE_ROCM
                                hipError_t alloc_err = hipMalloc(&d_emins_tmp, emins_bytes);
                                if (alloc_err != hipSuccess)
                                {
                                    LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to alloc native-VNNI emins: "
                                              << hipGetErrorString(alloc_err));
                                    d_emins_tmp = nullptr;
                                }
#endif
                                upload.d_native_vnni_emins = d_emins_tmp;
                                if (d_emins_tmp)
                                {
                                    err = startupMemcpyAsyncOrSync(
                                        d_emins_tmp,
                                        packed_->native_vnni_emins.data(),
                                        emins_bytes,
                                        async_upload_enabled,
                                        upload.startup_h2d_stream,
                                        &upload.startup_h2d_pinned_native_emins,
                                        "ROCmQuantisedGemmKernel::ensureWeightsConverted",
                                        "native_vnni_emins");
                                    if (err != hipSuccess)
                                    {
#ifdef HAVE_ROCM
                                        hipFree(d_emins_tmp);
#endif
                                        upload.d_native_vnni_emins = nullptr;
                                        LOG_WARN("[ROCmQuantisedGemmKernel] Failed to upload native-VNNI emins: "
                                                 << hipGetErrorString(err));
                                    }
                                    else
                                    {
                                        LOG_DEBUG("[ROCmQuantisedGemmKernel] Uploaded native-VNNI emins="
                                                  << (emins_bytes / 1024) << " KB");
                                    }
                                }
                            }
                        }

                        if (packed_->int8_data_vnni.empty() && packed_->native_vnni_payload.empty())
                        {
                            LOG_WARN("[ROCmQuantisedGemmKernel] No VNNI layout available. "
                                     "ROCm GEMV prefill paths may not work.");
                        }

                        // Synchronize and destroy the per-weight H2D stream.
                        // All hipMemcpyAsync calls on this stream must complete
                        // before we commit the upload and release pinned staging.
#ifdef HAVE_ROCM
                        if (h2d_upload_stream)
                        {
                            hipStreamSynchronize(h2d_upload_stream);
                            hipStreamDestroy(h2d_upload_stream);
                            h2d_upload_stream = nullptr;
                            upload.startup_h2d_stream = nullptr;
                        }
#endif
                        // Free pinned staging buffers now that the stream is synced
                        freeStartupPinnedStaging(upload);
                    }

                    {
                        ScopedWeightLoadDetailTimer commit_timer("weights.gemm_pack.commit_publish");
                        auto emplaced = packed_->device_uploads.emplace(rocm_device_id_, upload);
                        upload_it = emplaced.first;
                    }

                    LOG_DEBUG("[ROCmQuantisedGemmKernel] Uploaded pre-packed weights to ROCm:" << rocm_device_id_
                                                                                               << " " << packed_->N << "x" << packed_->K
                                                                                               << " vnni=" << (packed_->int8_data_vnni.size() / 1024) << " KB"
                                                                                               << " (VNNI-only)");
                }

                {
                    ScopedWeightLoadDetailTimer commit_timer("weights.gemm_pack.commit_publish");
                    auto &upload = upload_it->second;
                    packed_->d_int8_data_vnni = upload.d_int8_data_vnni;
                    packed_->d_scales = upload.d_scales;
                    packed_->d_native_vnni_payload = upload.d_native_vnni_payload;
                    packed_->d_native_vnni_scales = upload.d_native_vnni_scales;
                    packed_->d_native_vnni_mins = upload.d_native_vnni_mins;
                    packed_->d_native_vnni_emins = upload.d_native_vnni_emins;
                    packed_->uploaded = true;
                    packed_->rocm_device_id = rocm_device_id_;

                    // Point impl_ to packed_ device pointers
                    impl_->d_weights_int8_vnni = upload.d_int8_data_vnni;
                    impl_->d_scales_B = upload.d_scales;
                    impl_->d_weights_native_vnni = upload.d_native_vnni_payload;
                    impl_->d_weights_native_scales = upload.d_native_vnni_scales;
                    impl_->d_weights_native_mins = upload.d_native_vnni_mins;
                    impl_->d_weights_native_emins = upload.d_native_vnni_emins;
                    impl_->native_vnni_codebook_id = packed_->native_vnni_codebook_id;
                    impl_->native_vnni_blocks_per_row = packed_->native_vnni_blocks_per_row;
                    impl_->has_native_vnni = (upload.d_native_vnni_payload != nullptr && upload.d_native_vnni_scales != nullptr);
                    impl_->startup_h2d_pinned_scales = upload.startup_h2d_pinned_scales;
                    impl_->startup_h2d_pinned_vnni = upload.startup_h2d_pinned_vnni;
                    impl_->startup_h2d_pinned_native_vnni = upload.startup_h2d_pinned_native_vnni;
                    impl_->startup_h2d_pinned_native_scales = upload.startup_h2d_pinned_native_scales;
                    impl_->startup_h2d_pinned_native_mins = upload.startup_h2d_pinned_native_mins;
                    impl_->startup_h2d_pinned_native_emins = upload.startup_h2d_pinned_native_emins;
                }
                weights_converted_ = true;

                // Release host-side packing buffers — data is now on GPU.
                // This saves ~2× model_size of host memory for large models.
                // Only safe when this packed_ won't be uploaded to additional devices;
                // we check device_uploads.size() == 1 as a proxy (TP shards have
                // separate packed_ per shard, so this is typically the only device).
                if (packed_->device_uploads.size() <= 1)
                {
                    const size_t freed_bytes =
                        packed_->int8_data.capacity() +
                        packed_->int8_data_vnni.capacity() +
                        packed_->scales.capacity() * sizeof(float) +
                        packed_->native_vnni_payload.capacity() +
                        packed_->native_vnni_scales.capacity() * sizeof(uint16_t) +
                        packed_->native_vnni_mins.capacity() * sizeof(uint16_t);
                    packed_->int8_data.clear();
                    packed_->int8_data.shrink_to_fit();
                    packed_->int8_data_vnni.clear();
                    packed_->int8_data_vnni.shrink_to_fit();
                    packed_->scales.clear();
                    packed_->scales.shrink_to_fit();
                    packed_->native_vnni_payload.clear();
                    packed_->native_vnni_payload.shrink_to_fit();
                    packed_->native_vnni_scales.clear();
                    packed_->native_vnni_scales.shrink_to_fit();
                    packed_->native_vnni_mins.clear();
                    packed_->native_vnni_mins.shrink_to_fit();
                    if (freed_bytes > 0)
                    {
                        LOG_DEBUG("[ROCmQuantisedGemmKernel] Released host packing buffers: "
                                  << (freed_bytes / (1024 * 1024)) << " MB");
                    }
                }
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

            // Legacy path: upload the packed layouts produced by packWeightsToROCm.
            // INT8 formats provide VNNI + scales; native-VNNI formats provide
            // payload + per-block metadata for the dedicated native kernels.
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

            // Upload native-VNNI payload + metadata when present.
            if (!host_packed.native_vnni_payload.empty() && !host_packed.native_vnni_scales.empty())
            {
                if (!rocmQuantGemm_allocInt8(reinterpret_cast<int8_t **>(&impl_->d_weights_native_vnni),
                                             host_packed.native_vnni_payload.size(),
                                             rocm_device_id_))
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to alloc native-VNNI payload");
                    return;
                }

                err = hipMemcpy(impl_->d_weights_native_vnni,
                                host_packed.native_vnni_payload.data(),
                                host_packed.native_vnni_payload.size() * sizeof(uint8_t),
                                hipMemcpyHostToDevice);
                if (err != hipSuccess)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to upload native-VNNI payload: "
                              << hipGetErrorString(err));
                    return;
                }

                const size_t native_scales_bytes = host_packed.native_vnni_scales.size() * sizeof(uint16_t);
                err = hipMalloc(&impl_->d_weights_native_scales, native_scales_bytes);
                if (err != hipSuccess)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to alloc native-VNNI scales: "
                              << hipGetErrorString(err));
                    return;
                }

                err = hipMemcpy(impl_->d_weights_native_scales,
                                host_packed.native_vnni_scales.data(),
                                native_scales_bytes,
                                hipMemcpyHostToDevice);
                if (err != hipSuccess)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to upload native-VNNI scales: "
                              << hipGetErrorString(err));
                    return;
                }

                if (!host_packed.native_vnni_mins.empty())
                {
                    const size_t native_mins_bytes = host_packed.native_vnni_mins.size() * sizeof(uint16_t);
                    err = hipMalloc(&impl_->d_weights_native_mins, native_mins_bytes);
                    if (err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to alloc native-VNNI mins: "
                                  << hipGetErrorString(err));
                        return;
                    }

                    err = hipMemcpy(impl_->d_weights_native_mins,
                                    host_packed.native_vnni_mins.data(),
                                    native_mins_bytes,
                                    hipMemcpyHostToDevice);
                    if (err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to upload native-VNNI mins: "
                                  << hipGetErrorString(err));
                        return;
                    }
                }

                if (!host_packed.native_vnni_emins.empty())
                {
                    const size_t native_emins_bytes = host_packed.native_vnni_emins.size() * sizeof(uint32_t);
                    err = hipMalloc(&impl_->d_weights_native_emins, native_emins_bytes);
                    if (err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to alloc native-VNNI emins: "
                                  << hipGetErrorString(err));
                        return;
                    }

                    err = hipMemcpy(impl_->d_weights_native_emins,
                                    host_packed.native_vnni_emins.data(),
                                    native_emins_bytes,
                                    hipMemcpyHostToDevice);
                    if (err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to upload native-VNNI emins: "
                                  << hipGetErrorString(err));
                        return;
                    }
                }

                impl_->native_vnni_codebook_id = host_packed.native_vnni_codebook_id;
                impl_->native_vnni_blocks_per_row = host_packed.native_vnni_blocks_per_row;
            }

            impl_->has_native_vnni = (impl_->d_weights_native_vnni != nullptr &&
                                      impl_->d_weights_native_scales != nullptr);

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
            if (workspace_ && impl_->validated_workspace == workspace_)
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
            if (!workspace_->hasBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE))
            {
                throw std::runtime_error(
                    "[ROCmQuantisedGemmKernel] Workspace missing required buffer: SCALES_A_BLOCKWISE");
            }
            if (!workspace_->hasBuffer(GemmWorkspaceBuffers::ACC_INT32))
            {
                throw std::runtime_error(
                    "[ROCmQuantisedGemmKernel] Workspace missing required buffer: ACC_INT32");
            }
            // Shared FP32 temp buffers (used by serial paths only)
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
            // NOTE: CK-specific buffers (ROCM_CK_INT32, ROCM_A_PADDED,
            // ROCM_SCALE_A_PADDED, ROCM_E_PADDED, ROCM_B_REPACK) are no
            // longer requested — the CK path is retired. Skipped here.
            const std::string scatter_partial_name = scatterPartialBufferName();
            if (!workspace_->hasBuffer(scatter_partial_name))
            {
                throw std::runtime_error(
                    "[ROCmQuantisedGemmKernel] Workspace missing required buffer: " +
                    scatter_partial_name);
            }

            // Populate impl_ pointers from workspace
            impl_->d_A_int8 = static_cast<int8_t *>(workspace_->getBuffer(GemmWorkspaceBuffers::QUANT_A));
            impl_->d_scales_A = static_cast<float *>(workspace_->getBuffer(GemmWorkspaceBuffers::SCALES_A));
            impl_->d_scales_A_blockwise = static_cast<float *>(workspace_->getBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE));
            impl_->d_C_int32 = static_cast<int32_t *>(workspace_->getBuffer(GemmWorkspaceBuffers::ACC_INT32));
            impl_->d_A_fp32 = static_cast<float *>(workspace_->getBuffer(GemmWorkspaceBuffers::TEMP_A_FP32));
            impl_->d_C_fp32 = static_cast<float *>(workspace_->getBuffer(GemmWorkspaceBuffers::TEMP_C_FP32));
            impl_->d_scatter_partial = static_cast<float *>(workspace_->getBuffer(scatter_partial_name));

            // Set capacity values to max (workspace is pre-sized for maximum dimensions)
            // These are used by code paths that check capacity before use
            impl_->d_A_fp32_capacity = SIZE_MAX;
            impl_->d_C_fp32_capacity = SIZE_MAX;

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

        // =====================================================================
        // Fused SwiGLU + GEMM
        // =====================================================================

        extern "C"
        {
            bool hipOps_swiglu_fp32(
                const float *gate, const float *up, float *output,
                int size, int device_idx, void *stream);
            bool hipOps_fused_swiglu_quantize_blockwise(
                const float *gate, const float *up,
                int8_t *A_int8, float *scales_A_blockwise,
                int M, int K, int device_idx, void *stream);
        }

        bool ROCmQuantisedGemmKernel::multiply_tensor_with_fused_swiglu(
            const TensorBase *gate,
            const TensorBase *up,
            TensorBase *output,
            int m, int n, int k,
            float alpha, float beta)
        {
            // Get GPU data pointers (tensors must already be on device via coherence)
            const float *d_gate = static_cast<const float *>(gate->gpu_data_ptr());
            const float *d_up = static_cast<const float *>(up->gpu_data_ptr());
            float *d_C = static_cast<float *>(output->gpu_data_ptr());

            if (!d_gate || !d_up || !d_C)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor_with_fused_swiglu] "
                          "Null GPU pointer: gate="
                          << (void *)d_gate
                          << " up=" << (void *)d_up << " C=" << (void *)d_C);
                return false;
            }

            ensureWeightsConverted();
            validateWorkspace();

            if (!impl_)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor_with_fused_swiglu] "
                          "No impl_ (workspace buffers not initialized)");
                return false;
            }

            if (m > 1)
            {
                // M>1 PREFILL: Fused SwiGLU + blockwise quantize → INT8 GEMM
                if (!impl_->d_A_int8 || !impl_->d_scales_A_blockwise)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor_with_fused_swiglu] "
                              "Missing workspace buffers for fused SwiGLU+quant (d_A_int8 or d_scales_A_blockwise)");
                    return false;
                }

                // Step 1: Fused SwiGLU + blockwise quantize → d_A_int8 + d_scales_A_blockwise
                if (!hipOps_fused_swiglu_quantize_blockwise(
                        d_gate, d_up,
                        impl_->d_A_int8, impl_->d_scales_A_blockwise,
                        m, k, rocm_device_id_, gpu_stream_))
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor_with_fused_swiglu] "
                              "Fused SwiGLU+quantize kernel failed");
                    return false;
                }

                // Step 2: Native-VNNI GEMM (same dispatch as multiply_tensor M>1)
                if (impl_->has_native_vnni && alpha == 1.0f && beta == 0.0f)
                {
                    const uint8_t cb_id = impl_->native_vnni_codebook_id;
                    if (rocmGemm_native_vnni_fp32(
                            impl_->d_A_int8,
                            impl_->d_weights_native_vnni,
                            impl_->d_weights_native_scales,
                            impl_->d_weights_native_mins,
                            impl_->d_weights_native_emins,
                            d_C,
                            impl_->d_scales_A,
                            impl_->d_scales_A_blockwise,
                            m, n, k,
                            cb_id,
                            rocm_device_id_, gpu_stream_))
                    {
                        return true;
                    }
                }

                // Resolve d_scales_B (same logic as multiply_tensor)
                float *d_scales_B = nullptr;
                if (packed_)
                    d_scales_B = packed_->d_scales;
                else if (impl_)
                    d_scales_B = impl_->d_scales_B;

                // Fallback: tryPrefillNativeGemm (INT8 VNNI path)
                if (tryPrefillNativeGemm(
                        impl_->d_A_int8, d_C,
                        impl_->d_scales_A, impl_->d_scales_A_blockwise,
                        d_scales_B, nullptr,
                        m, n, k, alpha, beta,
                        "ROCmQuantisedGemmKernel::multiply_tensor_with_fused_swiglu"))
                {
                    return true;
                }

                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor_with_fused_swiglu] "
                          "No GEMM path available for M>1 (m="
                          << m << " n=" << n << " k=" << k << ")");
                return false;
            }
            else
            {
                // M=1 DECODE: SwiGLU → FP32 temp → standard FP32→FP32 GEMV path
                if (!impl_->d_A_fp32)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor_with_fused_swiglu] "
                              "No temp FP32 workspace buffer for M=1 decode path");
                    return false;
                }

                // Step 1: Compute SwiGLU on GPU: temp = silu(gate) * up
                if (!hipOps_swiglu_fp32(d_gate, d_up, impl_->d_A_fp32, m * k,
                                        rocm_device_id_, gpu_stream_))
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor_with_fused_swiglu] "
                              "SwiGLU kernel failed for decode");
                    return false;
                }

                // Step 2: Quantize + GEMV via existing FP32→FP32 path
                return multiply_fp32_to_fp32(impl_->d_A_fp32, d_C, m, n, k, alpha, beta);
            }
        }

        bool ROCmQuantisedGemmKernel::multiply_fp32_to_fp32(
            const float *d_A, float *d_C,
            int m, int n, int k,
            float alpha, float beta)
        {
            return multiply_fp32_to_fp32_with_bias(d_A, d_C, nullptr, m, n, k, alpha, beta);
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

                    // Native-VNNI path: lossless Q4/IQ4 decode, FP16 block scales, then optional alpha/beta/bias epilogue.
                    if (impl_->has_native_vnni)
                    {
                        if (!rocmQuantGemm_quantizeActivationsBlockwise(
                                d_A, impl_->d_A_int8, impl_->d_scales_A_blockwise, m, k, rocm_device_id_, gpu_stream_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] Native-VNNI blockwise activation quantization failed");
                            return false;
                        }

                        float *d_native_output = (beta != 0.0f) ? impl_->d_C_fp32 : d_C;
                        if (!rocmGemv_native_vnni_fp32(
                                impl_->d_A_int8,
                                impl_->d_weights_native_vnni,
                                impl_->d_weights_native_scales,
                                impl_->d_weights_native_mins,
                                impl_->d_weights_native_emins,
                                d_native_output,
                                impl_->d_scales_A,
                                impl_->d_scatter_partial,
                                n, k,
                                impl_->native_vnni_codebook_id,
                                rocm_device_id_, gpu_stream_,
                                impl_->d_scales_A_blockwise))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] Native-VNNI GEMV failed");
                            return false;
                        }

                        if (beta != 0.0f)
                        {
                            if (!rocmQuantGemm_applyFp32CombineEpilogue(
                                    d_native_output,
                                    d_C,
                                    d_C,
                                    d_bias,
                                    m,
                                    n,
                                    alpha,
                                    beta,
                                    rocm_device_id_,
                                    gpu_stream_))
                            {
                                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] Native-VNNI beta epilogue failed");
                                return false;
                            }
                        }
                        else if (alpha != 1.0f || d_bias)
                        {
                            if (!rocmQuantGemm_applyFp32Epilogue(
                                    d_C,
                                    d_bias,
                                    m,
                                    n,
                                    alpha,
                                    rocm_device_id_,
                                    gpu_stream_))
                            {
                                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] Native-VNNI epilogue failed");
                                return false;
                            }
                        }
                        return true;
                    }

                    // Use INT8 scatter+reduce GEMV when alpha=1, beta=0 AND VNNI weights available
                    // Bias handled inside scatter reduce kernel
                    if (alpha == 1.0f && beta == 0.0f && d_vnni)
                    {
                        if (!rocmQuantGemm_quantizeActivationsBlockwise(
                                d_A, impl_->d_A_int8, impl_->d_scales_A_blockwise, m, k, rocm_device_id_, gpu_stream_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] INT8 blockwise activation quantization failed");
                            return false;
                        }
                        bool ok = rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
                            impl_->d_A_int8, d_vnni, d_C,
                            impl_->d_scales_A_blockwise, d_s,
                            n, k,
                            alpha, beta,
                            nullptr, d_bias,
                            rocm_device_id_, gpu_stream_);

                        if (!ok)
                        {
                            ok = rocmGemv_int8_scatter_vnni_blockwise(
                                impl_->d_A_int8, d_vnni, d_C, impl_->d_scales_A_blockwise, d_s, d_bias,
                                impl_->d_scatter_partial,
                                n, k, alpha, beta, nullptr,
                                rocm_device_id_, gpu_stream_);
                        }

                        return ok;
                    }

                    // Fallback: blockwise quantize → blockwise GEMV for non-standard alpha/beta
                    if (!rocmQuantGemm_quantizeActivationsBlockwise(
                            d_A, impl_->d_A_int8, impl_->d_scales_A_blockwise, m, k, rocm_device_id_, gpu_stream_))
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] Activation quantization failed");
                        return false;
                    }

                    if (d_vnni)
                    {
                        float *d_int8_output = (beta != 0.0f) ? impl_->d_C_fp32 : d_C;
                        const float *d_existing = (beta != 0.0f) ? d_C : nullptr;
                        bool ok = rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
                            impl_->d_A_int8, d_vnni, d_int8_output,
                            impl_->d_scales_A_blockwise, d_s,
                            n, k,
                            alpha, beta,
                            d_existing, d_bias,
                            rocm_device_id_, gpu_stream_);

                        if (!ok)
                        {
                            ok = rocmGemv_int8_scatter_vnni_blockwise(
                                impl_->d_A_int8, d_vnni, d_int8_output, impl_->d_scales_A_blockwise, d_s, d_bias,
                                impl_->d_scatter_partial,
                                n, k, alpha, beta, d_existing,
                                rocm_device_id_, gpu_stream_);
                        }

                        if (!ok)
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] INT8 GEMV (VNNI blockwise) failed");
                            return false;
                        }
                        if (beta != 0.0f)
                        {
                            hipError_t copy_err = hipMemcpyAsync(
                                d_C,
                                d_int8_output,
                                static_cast<size_t>(n) * sizeof(float),
                                hipMemcpyDeviceToDevice,
                                static_cast<hipStream_t>(gpu_stream_));
                            if (copy_err != hipSuccess)
                            {
                                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] INT8 beta output copy failed: "
                                          << hipGetErrorString(copy_err));
                                return false;
                            }
                        }
                        return true;
                    }

                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] No VNNI weights for M=1 decode GEMV");
                    return false;
                }
            }

            // Ensure weights converted and validate workspace
            ensureWeightsConverted();
            validateWorkspace();

            // Step 1: Quantize FP32 activations to INT8 (blockwise)
            if (!rocmQuantGemm_quantizeActivationsBlockwise(
                    d_A, impl_->d_A_int8, impl_->d_scales_A_blockwise, m, k, rocm_device_id_, gpu_stream_))
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] Blockwise activation quantization failed");
                return false;
            }

            // Step 1b: Try native-VNNI GEMM (halved HBM bandwidth, no epilogue)
            // Same pattern as multiply_tensor and multiply_fp32_to_fp32: native-VNNI
            // produces final FP32 output with per-block scales applied inline.
            // Bias is applied via a lightweight biasAdd kernel after the GEMM.
            if (m > 1 && impl_->has_native_vnni && alpha == 1.0f && beta == 0.0f)
            {
                const uint8_t cb_id = impl_->native_vnni_codebook_id;
                if (rocmGemm_native_vnni_fp32(
                        impl_->d_A_int8,
                        impl_->d_weights_native_vnni,
                        impl_->d_weights_native_scales,
                        impl_->d_weights_native_mins,
                        impl_->d_weights_native_emins,
                        d_C,
                        impl_->d_scales_A,
                        impl_->d_scales_A_blockwise,
                        m, n, k,
                        cb_id,
                        rocm_device_id_, gpu_stream_))
                {
                    LOG_DEBUG("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] "
                              "Native-VNNI GEMM succeeded (M="
                              << m << " N=" << n << " K=" << k
                              << " codebook=" << static_cast<int>(cb_id) << " +bias)");

                    if (d_bias)
                    {
                        if (!rocmQuantGemm_biasAdd(d_C, d_bias, m, n, rocm_device_id_, gpu_stream_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] Native-VNNI GEMM bias add failed");
                            return false;
                        }
                    }
                    return true;
                }
                static std::once_flag nvnni_gemm_bias_fallback_once;
                std::call_once(nvnni_gemm_bias_fallback_once, [&]()
                               { LOG_WARN("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] "
                                          "Native-VNNI GEMM failed; falling back to INT8 GEMM"); });
            }

            if (m > 1 && tryPrefillNativeGemm(
                             impl_->d_A_int8,
                             d_C,
                             impl_->d_scales_A,
                             impl_->d_scales_A_blockwise,
                             impl_->d_scales_B,
                             d_bias,
                             m, n, k,
                             alpha, beta,
                             "ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias"))
            {
                return true;
            }

            // CK fallback retired — native-VNNI / INT8-VNNI are the only supported paths.
            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] "
                      "Native-VNNI and INT8-VNNI prefill both failed (m="
                      << m
                      << " n=" << n << " k=" << k << "); CK fallback has been removed.");
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
