#ifndef LLAMINAR2_KERNELS_ROCM_ROCMQUANTISEDGEMMKERNEL_H
#define LLAMINAR2_KERNELS_ROCM_ROCMQUANTISEDGEMMKERNEL_H

/**
 * @file ROCmQuantisedGemmKernel.h
 * @brief ROCm INT8 GEMM kernel for quantized tensors using AMD ComposableKernel (CK)
 *
 * Implements ITensorGemm using ComposableKernel (CK) INT8 GEMM for any quantized weight tensor.
 * This is the ROCm counterpart to CUDAQuantisedGemmKernel (which uses CUTLASS).
 *
 * ## Design Overview
 *
 * - **Primary Entry Point**: multiply_tensor() with automatic type introspection
 * - **Supported Weight Types**: IQ4_NL, Q8_0, Q4_0, Q4_K, and all GGUF quantized formats
 * - **Weight Conversion**: Dequantize → re-quantize to symmetric INT8 with per-column scales
 * - **Activation Handling**: FP32 activations quantized on-the-fly with per-row scales
 * - **CK Backend**: DeviceGemmMultipleD_Dl for gfx906 (MI50/MI60)
 *
 * ## Type Dispatch Matrix
 *
 * | A (input) | C (output) | Compute Path                           |
 * |-----------|------------|----------------------------------------|
 * | Q8_1      | FP32       | Direct INT8×INT8→INT32 + scale to FP32 |
 * | Q8_1      | Q8_1       | INT8×INT8→INT32 + fused requantize     |
 * | FP32      | FP32       | Quantize A → INT8×INT8→FP32            |
 * | FP32      | Q8_1       | Quantize A → INT8×INT8→Q8_1            |
 *
 * ## Memory Layout Convention
 *
 * This kernel uses **Row-Major layout for all matrices**, following CK's `mk_kn_mn` convention:
 *
 * - **A (activations)**: [M × K] row-major, element A[m,k] at offset `m * K + k`
 * - **B (weights)**: [K × N] row-major, element B[k,n] at offset `k * N + n`
 * - **C (output)**: [M × N] row-major, element C[m,n] at offset `m * N + n`
 *
 * The original model weights are [N × K] (output_features × input_features).
 * During packing, we transpose to [K × N] for efficient GEMM computation.
 *
 * ## Architecture Support
 *
 * | GPU Family | Architecture | CK Template Used       | Notes                    |
 * |------------|--------------|------------------------|-------------------------|
 * | MI50/MI60  | gfx906       | DeviceGemmMultipleD_Dl | DL instructions (4-way) |
 * | MI100      | gfx908       | DeviceGemmXdl          | MFMA (future)           |
 * | MI200/MI300| gfx90a/940a  | DeviceGemmXdl          | MFMA (future)           |
 *
 * ## References
 *
 * This implementation was developed using the following CK resources:
 *
 * - **Instance configuration**: device_gemm_dl_i8_i8_i8_mk_kn_mn_instance.cpp
 *   https://github.com/ROCm/composable_kernel/blob/develop/library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_dl_i8_i8_i8_mk_kn_mn_instance.cpp
 *
 * - **INT8 quantization example**: gemm_dl_quantization_int8.cpp
 *   https://github.com/ROCm/composable_kernel/blob/develop/example/14_gemm_quantization/gemm_dl_quantization_int8.cpp
 *
 * - **Key insight**: Tile parameters (ABlockTransfer*, BBlockTransfer*) are layout-specific.
 *   The `mk_kn_mn` suffix indicates Row,Row,Row layout; `km_kn_mn` indicates Col,Row,Row.
 *   Using wrong tile parameters causes incorrect numerical results without any error messages.
 *
 * ## Usage Example
 *
 * ```cpp
 * // Create kernel for quantized weights
 * auto kernel = std::make_unique<ROCmQuantisedGemmKernel>(weights, rocm_device_id);
 *
 * // Run GEMM: output = activations × weights^T
 * kernel->multiply_tensor(activations, output, m, n, k);
 * ```
 *
 * ## Workspace Management (Phase 2)
 *
 * This kernel implements IGpuWorkspaceConsumer for centralized workspace buffer
 * management. When a GpuWorkspaceManager is bound, the kernel uses pre-allocated
 * buffers instead of its internal allocations, enabling:
 *
 * - **Zero hot-path allocations**: No hipMalloc during GEMM execution
 * - **Memory budget control**: Workspace fits within overall VRAM budget
 * - **Buffer sharing**: Same workspace can be reused across kernel calls
 *
 * ```cpp
 * // Managed mode (Phase 2):
 * auto reqs = kernel->getWorkspaceRequirements(max_m, n, k);
 * workspaceManager->allocate(reqs);
 * kernel->bindWorkspace(workspaceManager);
 * kernel->multiply_tensor(A, C, m, n, k);  // Uses pre-allocated buffers
 * ```
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/BlockStructures.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include <memory>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <mutex>

namespace llaminar2
{
    // Forward declarations
    class TensorBase;
    class Q8_1Tensor;
    class FP32Tensor;
    class WeightPreloader; // For friend class access to ensureWeightsConverted

    namespace rocm
    {

        /**
         * @brief Startup GPU repack pipeline configuration scaffold (Phase 4 Step 1)
         *
         * This struct is intentionally plumbing-only in Step 1 and does not alter
         * current runtime behavior until the pipeline execution path is wired.
         */
        struct ROCmStartupRepackPipelineConfig
        {
            bool enabled = false;
            int slots = 8;
            int budget_mb = 1024;
            int stream_count = 3;
        };

        /**
         * @brief Build startup GPU repack pipeline configuration from DebugEnv
         *
         * Step 1 scope: expose env-driven configuration in one place, without
         * activating a new execution path yet.
         */
        ROCmStartupRepackPipelineConfig getROCmStartupRepackPipelineConfig();

        /**
         * @struct ROCmPackedWeights
         * @brief Pre-packed INT8 weights for ROCm CK GEMM
         *
         * Stores weights converted from any quantized format to symmetric INT8 with per-column scales.
         *
         * ## Memory Layout
         *
         * The packed weights use **Row-Major [K × N]** layout (matching CK's `mk_kn_mn` convention):
         *
         * - `int8_data[k * N + n]` = weight value for input feature `k`, output feature `n`
         * - `scales[n]` = scale factor for output feature (column) `n`
         *
         * This is the **transpose** of the original model weights which are [N × K].
         *
         * Optional VNNI layout is also stored for GEMV experiments:
         *   - int8_data_vnni: [K/4][N][4] where each group stores 4 contiguous K values per column.
         *
         * ## Quantization Formula
         *
         * Per-column symmetric quantization:
         * ```
         * max_abs = max(|weights[:, n]|)      // Per-column max absolute value
         * scale[n] = max_abs / 127.0          // Scale to [-127, 127] range
         * int8[k, n] = round(weights[n, k] / scale[n])  // Quantize with transpose
         * ```
         *
         * Dequantization (for output scaling):
         * ```
         * output[m, n] = (int32_result[m, n] * scale_A[m] * scale_B[n])
         * ```
         *
         * ## Caching
         *
         * This structure is cached in tensor->cache_ to avoid re-conversion on every kernel call.
         * Device upload happens lazily on first kernel execution.
         */
        struct ROCmPackedWeights
        {
            struct DeviceUpload
            {
                int8_t *d_int8_data_vnni = nullptr;
                int8_t *d_int8_data_rowmajor = nullptr;
                float *d_scales = nullptr;
                void *startup_h2d_pinned_scales = nullptr;
                void *startup_h2d_pinned_vnni = nullptr;
                uint8_t *d_native_vnni_payload = nullptr;
                void *d_native_vnni_scales = nullptr; // __half* on device; void* for header compatibility
                void *d_native_vnni_mins = nullptr;   // __half* on device; NULL for symmetric formats
                void *d_native_vnni_emins = nullptr;  // uint32_t* packed {lo,hi} emins; Q2_K only
                void *startup_h2d_pinned_native_payload = nullptr;
                void *startup_h2d_pinned_native_scales = nullptr;
                void *startup_h2d_pinned_native_mins = nullptr;
                void *startup_h2d_pinned_native_emins = nullptr;
                void *startup_h2d_stream = nullptr;
                void *startup_repack_stream = nullptr;
                void *startup_commit_stream = nullptr;
                void *startup_h2d_done_event = nullptr;
                bool startup_h2d_event_pending = false;
                void *startup_repack_ready_event = nullptr;
                bool startup_repack_event_pending = false;
                void *startup_commit_ready_event = nullptr;
                bool startup_commit_event_pending = false;
            };

            std::vector<int8_t> int8_data;      ///< [K × N] RowMajor INT8 weights (host only, not uploaded to device)
            std::vector<int8_t> int8_data_vnni; ///< [K/4 × N × 4] VNNI layout (the sole device layout)
            std::vector<float> scales;          ///< [N] per-column (per-output-feature) scale factors

            // Native-VNNI compact container (Q4_0, IQ4_NL)
            // Achieves lossless weight reconstruction by keeping per-block FP16 scales separate.
            std::vector<uint8_t> native_vnni_payload; ///< [blocks_per_row × N × 16] nibble payload interleaved by N
            std::vector<uint16_t> native_vnni_scales; ///< [blocks_per_row × N] FP16 per-block scales (raw uint16_t bits)
            std::vector<uint16_t> native_vnni_mins;   ///< [blocks_per_row × N] FP16 per-block mins (asymmetric formats only)
            std::vector<uint32_t> native_vnni_emins;  ///< [blocks_per_row × N] packed {lo,hi} emins (Q2_K only)
            uint8_t native_vnni_codebook_id = 0;      ///< NativeVNNIFormat: 0=Q4_0, 4=IQ4_NL, 5=Q4_1, 6=Q5_0, 7=Q5_1
            uint32_t native_vnni_blocks_per_row = 0;  ///< K / 32

            int K = 0; ///< Input features (rows in CK B matrix)
            int N = 0; ///< Output features (cols in CK B matrix)

            mutable std::mutex upload_mutex;
            std::unordered_map<int, DeviceUpload> device_uploads;

            // Device memory pointers (uploaded once, cached)
            // Legacy compatibility fields, mirrored from the active device upload.
            // Option B: Only VNNI layout is uploaded to device. Row-major is repacked
            // on-demand into a shared workspace scratch buffer for CK GEMM prefill.
            // Phase 4 pilot: optional persistent row-major buffer precomputed on GPU at startup.
            int8_t *d_int8_data_vnni = nullptr;         ///< Device pointer to VNNI-packed weights (sole device copy)
            int8_t *d_int8_data_rowmajor = nullptr;     ///< Optional persistent row-major CK buffer (startup GPU repack)
            float *d_scales = nullptr;                  ///< Device pointer to scales
            uint8_t *d_native_vnni_payload = nullptr;   ///< Device pointer to native-VNNI payload
            void *d_native_vnni_scales = nullptr;       ///< Device pointer to native-VNNI FP16 scales (__half*)
            void *d_native_vnni_mins = nullptr;         ///< Device pointer to native-VNNI FP16 mins (__half*, NULL for symmetric)
            void *d_native_vnni_emins = nullptr;        ///< Device pointer to native-VNNI packed emins (uint32_t*, Q2_K only)
            void *startup_repack_ready_event = nullptr; ///< Optional startup repack completion event (hipEvent_t*)
            bool startup_repack_event_pending = false;  ///< True until startup repack event is consumed by CK stream wait
            void *startup_commit_ready_event = nullptr; ///< Optional startup commit completion event (hipEvent_t*)
            bool startup_commit_event_pending = false;  ///< True until startup commit event is consumed by CK stream wait
            int rocm_device_id = -1;                    ///< Device where data is uploaded
            bool uploaded = false;                      ///< Whether device memory is allocated

            ROCmPackedWeights() = default;
            ROCmPackedWeights(const ROCmPackedWeights &) = delete;
            ROCmPackedWeights &operator=(const ROCmPackedWeights &) = delete;

            ROCmPackedWeights(ROCmPackedWeights &&other) noexcept
            {
                *this = std::move(other);
            }

            ROCmPackedWeights &operator=(ROCmPackedWeights &&other) noexcept
            {
                if (this != &other)
                {
                    std::scoped_lock guard(upload_mutex, other.upload_mutex);
                    int8_data = std::move(other.int8_data);
                    int8_data_vnni = std::move(other.int8_data_vnni);
                    scales = std::move(other.scales);
                    native_vnni_payload = std::move(other.native_vnni_payload);
                    native_vnni_scales = std::move(other.native_vnni_scales);
                    native_vnni_mins = std::move(other.native_vnni_mins);
                    native_vnni_emins = std::move(other.native_vnni_emins);
                    native_vnni_codebook_id = other.native_vnni_codebook_id;
                    native_vnni_blocks_per_row = other.native_vnni_blocks_per_row;
                    K = other.K;
                    N = other.N;
                    device_uploads = std::move(other.device_uploads);
                    d_int8_data_vnni = other.d_int8_data_vnni;
                    d_int8_data_rowmajor = other.d_int8_data_rowmajor;
                    d_scales = other.d_scales;
                    d_native_vnni_payload = other.d_native_vnni_payload;
                    d_native_vnni_scales = other.d_native_vnni_scales;
                    d_native_vnni_mins = other.d_native_vnni_mins;
                    d_native_vnni_emins = other.d_native_vnni_emins;
                    startup_repack_ready_event = other.startup_repack_ready_event;
                    startup_repack_event_pending = other.startup_repack_event_pending;
                    startup_commit_ready_event = other.startup_commit_ready_event;
                    startup_commit_event_pending = other.startup_commit_event_pending;
                    rocm_device_id = other.rocm_device_id;
                    uploaded = other.uploaded;

                    other.d_int8_data_vnni = nullptr;
                    other.d_int8_data_rowmajor = nullptr;
                    other.d_scales = nullptr;
                    other.d_native_vnni_payload = nullptr;
                    other.d_native_vnni_scales = nullptr;
                    other.d_native_vnni_mins = nullptr;
                    other.d_native_vnni_emins = nullptr;
                    other.native_vnni_codebook_id = 0;
                    other.native_vnni_blocks_per_row = 0;
                    other.startup_repack_ready_event = nullptr;
                    other.startup_repack_event_pending = false;
                    other.startup_commit_ready_event = nullptr;
                    other.startup_commit_event_pending = false;
                    other.rocm_device_id = -1;
                    other.uploaded = false;
                    other.K = 0;
                    other.N = 0;
                }
                return *this;
            }

            ~ROCmPackedWeights();
        };

        /**
         * @brief Pack any quantized tensor to ROCmPackedWeights (host-side)
         *
         * Dequantizes the tensor to FP32, then re-quantizes symmetrically to INT8.
         * Device upload happens separately when the kernel is first used.
         *
         * @param tensor Source quantized tensor
         * @param out Output packed weights structure
         * @return true on success
         */
        bool packWeightsToROCm(const TensorBase *tensor, ROCmPackedWeights &out);

        /**
         * @brief ROCm GEMM kernel for quantized weight tensors using ComposableKernel INT8
         *
         * Implements ITensorGemm for any quantized weight tensor type.
         *
         * ## Supported Weight Types
         *
         * - IQ4_NL, IQ4_XS, IQ2_*, IQ3_*, IQ1_* (imatrix quantized)
         * - Q8_0, Q8_1, Q8_K (8-bit quantized)
         * - Q4_0, Q4_1, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K (4-6 bit quantized)
         * - Q2_K, Q3_K (2-3 bit quantized)
         * - Any tensor implementing fp32_data() dequantization
         *
         * ## Compute Pipeline
         *
         * ```
         * ┌─────────────┐    ┌──────────────────┐    ┌─────────────┐
         * │ Quantized   │───▶│ Dequant to FP32  │───▶│ Requant to  │
         * │ Weights     │    │ (fp32_data())    │    │ INT8 [K×N]  │
         * └─────────────┘    └──────────────────┘    └─────────────┘
         *                                                   │
         * ┌─────────────┐    ┌──────────────────┐           │
         * │ FP32        │───▶│ Per-row INT8     │           │
         * │ Activations │    │ Quantization     │           │
         * └─────────────┘    └──────────────────┘           │
         *                           │                       │
         *                           ▼                       ▼
         *                    ┌──────────────────────────────────┐
         *                    │  CK DeviceGemmMultipleD_Dl       │
         *                    │  INT8 × INT8 → INT32 GEMM        │
         *                    └──────────────────────────────────┘
         *                                     │
         *                                     ▼
         *                    ┌──────────────────────────────────┐
         *                    │  Scale Output:                   │
         *                    │  FP32 = INT32 × scale_A × scale_B│
         *                    └──────────────────────────────────┘
         * ```
         *
         * ## Memory Layout (Row-Major Convention)
         *
         * All matrices use row-major layout matching CK's `mk_kn_mn` configuration:
         *
         * | Matrix       | Shape   | Layout    | Element Access           |
         * |--------------|---------|-----------|-------------------------|
         * | Activations  | [M × K] | RowMajor  | A[m,k] = data[m*K + k]  |
         * | Weights      | [K × N] | RowMajor  | B[k,n] = data[k*N + n]  |
         * | Output       | [M × N] | RowMajor  | C[m,n] = data[m*N + n]  |
         *
         * Note: Model weights are originally [N × K]; we transpose during packing.
         *
         * ## Architecture Support
         *
         * Currently implemented for gfx906 (MI50/MI60) using DeviceGemmMultipleD_Dl.
         * The DL (Data-Level) kernels use 4-way INT8 dot product instructions.
         * Future: DeviceGemmXdl for gfx908+ (MFMA instructions).
         */
        class ROCmQuantisedGemmKernel : public ITensorGemm, public IWorkspaceConsumer
        {
        public:
            /**
             * @brief Construct kernel for quantized weight tensor (lazy conversion)
             *
             * @param weights Any quantized tensor (must be on GPU)
             * @param rocm_device_id ROCm device ID (from hipGetDevice, NOT global index)
             *
             * @throws std::runtime_error if weight not quantized or not on GPU
             */
            ROCmQuantisedGemmKernel(const TensorBase *weights, int rocm_device_id);

            /**
             * @brief Construct kernel from pre-packed INT8 weights (PREFERRED)
             *
             * This constructor avoids redundant weight conversion by using pre-packed
             * ROCmPackedWeights that are cached in the tensor's cache_ field.
             *
             * @param packed Pre-packed INT8 weights with scales (from KernelFactory cache)
             * @param rocm_device_id ROCm device ID
             *
             * @throws std::runtime_error if packed is null or has invalid dimensions
             */
            ROCmQuantisedGemmKernel(ROCmPackedWeights *packed, int rocm_device_id);

            ~ROCmQuantisedGemmKernel() override;

            // Non-copyable
            ROCmQuantisedGemmKernel(const ROCmQuantisedGemmKernel &) = delete;
            ROCmQuantisedGemmKernel &operator=(const ROCmQuantisedGemmKernel &) = delete;

            // Movable
            ROCmQuantisedGemmKernel(ROCmQuantisedGemmKernel &&) noexcept;
            ROCmQuantisedGemmKernel &operator=(ROCmQuantisedGemmKernel &&) noexcept;

            // =========================================================================
            // ITensorGemm interface - Primary entry points
            // =========================================================================

            /**
             * @brief Tensor-based GEMM with type introspection (PRIMARY ENTRY POINT)
             *
             * Inspects A->native_type() and C->native_type() to select optimal path:
             * - Q8_1 → Q8_1: Zero-copy INT8 GEMM
             * - Q8_1 → FP32: INT8 GEMM + dequant
             * - FP32 → FP32: Quantize A + INT8 GEMM + dequant
             * - FP32 → Q8_1: Quantize A + INT8 GEMM + requant
             *
             * @param A Input activations tensor [m, k] (Q8_1 or FP32)
             * @param C Output tensor [m, n] (Q8_1 or FP32)
             * @param transpose_B Whether weights are transposed (ignored, always true)
             * @param alpha Scale factor for result
             * @param beta Scale for existing C (accumulate if != 0)
             * @param bias Optional bias tensor [n] to add after GEMM (nullptr = no bias)
             * @param mpi_ctx MPI context (unused for ROCm kernel)
             * @param device_idx Device index (unused, kernel bound to rocm_device_id_)
             * @param workspace Pre-allocated device workspace (nullptr = kernel allocates)
             *
             * @return true on success
             */
            bool multiply_tensor(
                const TensorBase *A, TensorBase *C,
                bool transpose_B = true,
                float alpha = 1.0f, float beta = 0.0f,
                const TensorBase *bias = nullptr,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                DeviceWorkspaceManager *workspace = nullptr,
                int activation_row_offset = 0) override;

            /**
             * @brief Tensor-based GEMM with explicit dimensions
             *
             * Same as above but with explicit m, n, k for pre-allocated buffers.
             */
            bool multiply_tensor(
                const TensorBase *A, TensorBase *C,
                int m, int n, int k,
                bool transpose_B = true,
                float alpha = 1.0f, float beta = 0.0f,
                const TensorBase *bias = nullptr,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                DeviceWorkspaceManager *workspace = nullptr,
                int activation_row_offset = 0) override;

            /**
             * @brief Raw FP32 pointer GEMM (fallback path)
             *
             * Quantizes FP32 activations → INT8, runs CK GEMM, outputs FP32.
             *
             * @param A FP32 activations [m, k] on device
             * @param C FP32 output [m, n] on device
             * @param workspace Pre-allocated device workspace (nullptr = kernel allocates)
             */
            bool multiply(
                const float *A, float *C,
                int m, int n, int k,
                bool transpose_B = true,
                float alpha = 1.0f, float beta = 0.0f,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                DeviceWorkspaceManager *workspace = nullptr) override;

            /**
             * @brief Fused multi-projection GEMM with automatic host/device transfer
             *
             * Handles the host-to-device transfer required for FusedGateUpGEMMStage
             * and FusedQKVGEMMStage which pass host pointers.
             *
             * @param input Host FP32 input [m, k]
             * @param projections Vector of projections (each with host output pointer)
             * @param m Number of rows
             * @param k Input dimension
             * @param mpi_ctx MPI context (unused for ROCm)
             * @param device_idx Device index (unused, kernel bound to rocm_device_id_)
             * @param workspace Pre-allocated device workspace (nullptr = kernel allocates)
             */
            bool multiply_fused(
                const float *input,
                const std::vector<FusedProjectionDesc> &projections,
                int m, int k,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                DeviceWorkspaceManager *workspace = nullptr) override;

            /**
             * @brief Tensor-aware fused multi-projection GEMM
             *
             * Optimized implementation that:
             * 1. Ensures input is on GPU (uploads if needed)
             * 2. Quantizes input to INT8 ONCE
             * 3. Runs all projections with the same quantized input
             * 4. Marks output tensors as device-dirty
             *
             * This is the preferred path for FusedQKVGEMMStage on GPU.
             *
             * @param input Input tensor [m, k] (will be uploaded to GPU if needed)
             * @param projections Vector of TensorProjectionDesc (kernels + output tensors)
             * @param m Number of rows (seq_len)
             * @param k Input dimension (d_model)
             * @param mpi_ctx MPI context (unused)
             * @param workspace Pre-allocated device workspace (nullptr = kernel allocates)
             * @return true on success
             */
            bool multiply_fused_tensor(
                const TensorBase *input,
                const std::vector<TensorProjectionDesc> &projections,
                int m, int k,
                const MPIContext *mpi_ctx = nullptr,
                DeviceWorkspaceManager *workspace = nullptr) override;

            /**
             * @brief Activation-activation GEMM (not supported for quantized kernel)
             *
             * ROCmQuantisedGemmKernel is for weight projections only.
             * For activation-activation GEMM, use a dedicated attention kernel.
             */
            bool multiply_activations(
                const float *A, const float *B, float *C,
                int m, int n, int k,
                bool transpose_B = true,
                float alpha = 1.0f, float beta = 0.0f,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override;

            /**
             * @brief Strided activation-activation GEMM (not supported)
             */
            bool multiply_activations_strided(
                const float *A, const float *B, float *C,
                int m, int n, int k,
                int lda, int ldb, int ldc,
                bool transpose_B = true,
                float alpha = 1.0f, float beta = 0.0f,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override;

            // =========================================================================
            // ITensorKernel interface
            // =========================================================================

            bool supports_device(int device_idx) const override;

            void setGPUStream(void *stream) override { gpu_stream_ = stream; }

            // =========================================================================
            // IKernelSnapshotCapable interface
            // =========================================================================

            KernelSnapshotInfo getKernelSnapshotInfo() const override;

            // =========================================================================
            // IWorkspaceConsumer Interface
            // =========================================================================

            /**
             * @brief Get workspace requirements for GEMM at given dimensions
             *
             * Returns buffers needed for INT8 GEMM:
             * - quant_a [M×K]: INT8 quantized activations
             * - scales_a [M]: per-row activation scales
             * - acc_int32 [M×N]: INT32 GEMM accumulator
             * - temp_a_fp32 [M×K]: FP32 activation input buffer
             * - temp_c_fp32 [M×N]: FP32 output buffer
             *
             * @param m Maximum sequence length (batch size)
             * @param n Number of output features (0 = use internal N_)
             * @param k Number of input features (0 = use internal K_)
             * @return WorkspaceRequirements describing all needed buffers
             *
             * @note Call with maximum expected M to allocate once and reuse.
             */
            WorkspaceRequirements getWorkspaceRequirements(
                int m, int n = 0, int k = 0) const override;

            /**
             * @brief Bind workspace manager for managed mode
             *
             * After binding, the kernel uses pre-allocated buffers from the
             * workspace manager instead of internal allocations.
             *
             * @param workspace Pointer to workspace manager (NOT owned, must outlive kernel)
             *                  Pass nullptr to return to legacy mode.
             */
            void bindWorkspace(DeviceWorkspaceManager *workspace) override;

            /**
             * @brief Check if a workspace is currently bound
             */
            bool hasWorkspace() const override;

            /**
             * @brief Get the currently bound workspace manager
             */
            DeviceWorkspaceManager *getWorkspace() const override;

            // =========================================================================
            // Benchmarking Support
            // =========================================================================

            /**
             * @brief Execute GEMM with kernel-only timing (excludes PCIe transfers)
             *
             * This method performs the full GEMM operation but returns the GPU kernel
             * execution time measured via HIP events, excluding:
             *   - Host-to-device activation transfer
             *   - Device-to-host output transfer
             *   - CPU-side quantization overhead
             *
             * Use this for accurate kernel performance measurement.
             *
             * @param A FP32 activations [m, k]
             * @param C FP32 output [m, n]
             * @param m Batch/sequence dimension
             * @param n Output features
             * @param k Input features
             * @param kernel_time_ms OUTPUT: GPU kernel time in milliseconds
             * @return true on success
             */
            bool multiply_tensor_timed(
                const TensorBase *A, TensorBase *C,
                int m, int n, int k,
                float *kernel_time_ms);

            // =========================================================================
            // Accessors
            // =========================================================================

            int rocm_device_id() const { return rocm_device_id_; }
            size_t weight_rows() const { return N_; }
            size_t weight_cols() const { return K_; }
            bool weights_converted() const { return weights_converted_; }

            /**
             * @brief Prepare weights for efficient execution (ITensorGemm interface)
             *
             * For ROCm: converts weights to INT8 + uploads to device memory.
             * Call this during weight preloading to avoid first-use overhead.
             */
            void prepareWeights() override { ensureWeightsConverted(); }

        private:
            /**
             * @brief Describes which native prefill path should be attempted for M>1.
             *
             * This enum is intentionally small and explicit so logs, counters, and
             * future telemetry can categorize prefill routing decisions in a stable
             * way. We keep `CK_FALLBACK` as an explicit value because fallback is a
             * first-class execution mode, not an error condition.
             */
            enum class PrefillDispatchPath
            {
                NATIVE_VNNI,      ///< Native-VNNI path (lossless ≤6-bit decode, FP16 scales)
                INT8_VNNI_NATIVE, ///< INT8-VNNI path (requantized 8-bit weights)
                CK_FALLBACK       ///< CK ComposableKernel (debug override only)
            };

            // =========================================================================
            // Internal dispatch methods
            // =========================================================================

            /**
             * @brief Select which prefill path should be attempted for this call.
             *
             * The selection policy is deliberately conservative:
             * - Decode (`m == 1`) never calls this helper.
             * - Unsupported metadata/shape immediately maps to `CK_FALLBACK`.
             * - Feature flag disablement also maps to `CK_FALLBACK`.
             *
             * @param m Number of rows in activation matrix.
             * @param n Number of output features.
             * @param k Number of input features.
             * @return Selected prefill dispatch path.
             */
            PrefillDispatchPath selectPrefillDispatchPath(int m, int n, int k) const;

            /**
             * @brief Attempt native INT8 VNNI prefill execution for M>1.
             *
             * This is the primary prefill dispatch path for VNNI-packed weights.
             * It centralizes all gate checks and one-time explanatory logging so
             * call sites stay small and junior developers can follow the
             * route/fallback logic in one place.
             *
             * @param d_A_int8 Quantized activations [M x K] in row-major INT8.
             * @param d_output FP32 output [M x N].
             * @param d_scales_A Per-row activation scales [M].
             * @param d_scales_B Per-column weight scales [N].
             * @param d_bias Optional bias [N], nullable.
             * @param m Number of rows.
             * @param n Number of output features.
             * @param k Number of input features.
             * @param alpha GEMM alpha.
             * @param beta GEMM beta.
             * @param callsite Friendly callsite name for logs.
             * @return true if native path succeeded and produced final FP32 output,
             *         false when caller should continue with CK fallback.
             */
            bool tryPrefillNativeGemm(
                const int8_t *d_A_int8,
                float *d_output,
                const float *d_scales_A,
                const float *d_scales_B,
                const float *d_bias,
                int m, int n, int k,
                float alpha, float beta,
                const char *callsite);

            /**
             * @brief Q8_1 activations → INT8 GEMM → FP32 output
             */
            bool multiply_q8_to_fp32(
                const Q8_1Block *d_A_q8, float *d_C,
                int m, int n, int k,
                float alpha, float beta);

            /**
             * @brief Q8_1 activations → INT8 GEMM → Q8_1 output (fused requant)
             */
            bool multiply_q8_to_q8(
                const Q8_1Block *d_A_q8, Q8_1Block *d_C_q8,
                int m, int n, int k);

            /**
             * @brief FP32 activations → quantize → INT8 GEMM → FP32 output
             */
            bool multiply_fp32_to_fp32(
                const float *d_A, float *d_C,
                int m, int n, int k,
                float alpha, float beta);

            /**
             * @brief FP32 activations → quantize → INT8 GEMM → FP32 output + bias
             * @param d_bias Optional bias vector [N], broadcasted across rows
             */
            bool multiply_fp32_to_fp32_with_bias(
                const float *d_A, float *d_C, const float *d_bias,
                int m, int n, int k,
                float alpha, float beta);

            /**
             * @brief FP32 activations → quantize → INT8 GEMM → Q8_1 output
             */
            bool multiply_fp32_to_q8(
                const float *d_A, Q8_1Block *d_C_q8,
                int m, int n, int k);

            // =========================================================================
            // Weight conversion
            // =========================================================================

            /**
             * @brief Ensure weights are converted to INT8 + scales
             *
             * Converts from native quantized format to:
             * - d_weights_int8_: INT8 [K × N] ColumnMajor
             * - d_scales_: float [N] per-column scales
             *
             * Conversion is done once and cached.
             */
            void ensureWeightsConverted();

            /**
             * @brief Validate workspace is bound and has required buffers
             *
             * @throws std::runtime_error if workspace not bound (kernels require workspace)
             */
            void validateWorkspace() const;

            // =========================================================================
            // Member data
            // =========================================================================

            const TensorBase *weights_ = nullptr; // Original weight tensor (null if using packed_)
            ROCmPackedWeights *packed_ = nullptr; // Pre-packed weights (owned by tensor cache)
            int rocm_device_id_;
            size_t N_; // Output features (weight rows)
            size_t K_; // Input features (weight cols)

            // Converted INT8 weight representation (cached) - only used with legacy constructor
            // When packed_ is set, these are unused (data comes from packed_->d_int8_data_vnni/d_scales)
            // Option B: Only VNNI layout stored on device; row-major repacked on-demand into workspace scratch
            int8_t *d_weights_int8_ = nullptr; // [K × N] RowMajor (legacy only, not uploaded to device with Option B)
            float *d_scales_B_ = nullptr;      // [N] per-column (per-output-feature) scales
            bool weights_converted_ = false;
            bool owns_weight_memory_ = false; // true if we allocated d_weights_int8_/d_scales_B_

            // IWorkspaceConsumer state - REQUIRED for execution
            // Kernels do not own any work buffers; all buffers come from workspace
            DeviceWorkspaceManager *workspace_ = nullptr; ///< Bound workspace manager (not owned, REQUIRED)

            // GPU stream for graph capture (nullptr = default stream)
            void *gpu_stream_ = nullptr;

            // Per-instance synchronization/logging state (no process-global mutable statics)
            std::unique_ptr<std::mutex> ck_dispatch_mutex_;
            // PIMPL for CK implementation (avoids CK headers in this header)
            struct Impl;
            std::unique_ptr<Impl> impl_;

            // Friend class for WeightPreloader (deprecated, will be removed)
            friend class llaminar2::WeightPreloader;
        };

    } // namespace rocm
} // namespace llaminar2

// =====================================================================
// Low-level CK GEMM Functions (exposed for benchmarking)
// =====================================================================
//
// NOTE: These are declared with C linkage to match the existing .cpp/.hip
// implementation pattern. They are global functions, not in any namespace.
// =====================================================================

extern "C"
{
    /**
     * @brief Pre-initialize all CK GEMM kernels to avoid first-call latency
     *
     * CK device objects are expensive to construct (5-10 seconds each on gfx906)
     * due to template instantiation and GPU ISA checks. Call this function
     * during backend initialization to avoid blocking on first inference call.
     *
     * Pre-initializes:
     *   - 32x32 kernel (for decode, M <= 32)
     *   - 64x64 kernel (for small batch, 32 < M < 128)
     *   - 128x128 kernel (for prefill, M >= 128)
     */
    void rocmQuantGemm_warmupKernels();

    /**
     * @brief Execute Two-Kernel INT8 GEMM (CK NoScale + applyScales_kernel)
     *
     * This is the DEFAULT and RECOMMENDED path. It uses:
     *   1. CK INT8×INT8→INT32 GEMM (executeNoScale)
     *   2. Custom scale application kernel (applyScales_kernel)
     *
     * Achieves ~0.9999 cosine similarity vs reference.
     */
    bool rocmQuantGemm_executeTwoKernel(
        const int8_t *d_A, const int8_t *d_B, float *d_E,
        const float *d_scaleA, const float *d_scaleB,
        int M, int N, int K,
        int rocm_device_id,
        void *stream,
        void *kernel_ctx);

    /**
     * @brief Execute Two-Kernel INT8 GEMM with HIP event timing (for benchmarking)
     *
     * Same as executeTwoKernel_cached but uses HIP events to measure ONLY
     * the GPU kernel execution time, excluding PCIe transfers.
     *
     * @param kernel_time_ms OUTPUT: Kernel execution time in milliseconds
     */
    bool rocmQuantGemm_executeTwoKernel_timed(
        const int8_t *d_A, const int8_t *d_B, float *d_E,
        const float *d_scaleA, const float *d_scaleB,
        int32_t *d_C_int32,
        int M, int N, int K,
        int rocm_device_id,
        float *kernel_time_ms, void *stream,
        void *kernel_ctx);

    /**
     * @brief Execute hipBLAS INT8 GEMM fallback
     *
     * Uses hipBLAS INT8 GEMM with FP32 accumulation. Useful as fallback
     * for dimensions CK doesn't support.
     */
    bool rocmQuantGemm_executeHipBLAS(
        const int8_t *d_A, const int8_t *d_B, float *d_E,
        const float *d_scaleA, const float *d_scaleB,
        int M, int N, int K,
        int rocm_device_id);
}

#endif // LLAMINAR2_KERNELS_ROCM_ROCMQUANTISEDGEMMKERNEL_H
