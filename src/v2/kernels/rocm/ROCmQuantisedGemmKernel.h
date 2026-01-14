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
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../../tensors/TensorKernels.h"
#include "../../tensors/BlockStructures.h"
#include <memory>
#include <cstdint>
#include <vector>

namespace llaminar2
{
    // Forward declarations
    class CPUTensorBase;
    using TensorBase = CPUTensorBase; // Backward compatibility alias
    class Q8_1Tensor;
    class FP32Tensor;

    namespace rocm
    {

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
            std::vector<int8_t> int8_data; ///< [K × N] RowMajor INT8 weights (transposed from model [N×K])
            std::vector<float> scales;     ///< [N] per-column (per-output-feature) scale factors
            int K = 0;                     ///< Input features (rows in CK B matrix)
            int N = 0;                     ///< Output features (cols in CK B matrix)

            // Device memory pointers (uploaded once, cached)
            int8_t *d_int8_data = nullptr; ///< Device pointer to INT8 weights
            float *d_scales = nullptr;     ///< Device pointer to scales
            int rocm_device_id = -1;       ///< Device where data is uploaded
            bool uploaded = false;         ///< Whether device memory is allocated

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
        class ROCmQuantisedGemmKernel : public ITensorGemm
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
             * @param mpi_ctx MPI context (unused for ROCm kernel)
             * @param device_idx Device index (unused, kernel bound to rocm_device_id_)
             *
             * @return true on success
             */
            bool multiply_tensor(
                const TensorBase *A, TensorBase *C,
                bool transpose_B = true,
                float alpha = 1.0f, float beta = 0.0f,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override;

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
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override;

            /**
             * @brief Raw FP32 pointer GEMM (fallback path)
             *
             * Quantizes FP32 activations → INT8, runs CK GEMM, outputs FP32.
             *
             * @param A FP32 activations [m, k] on device
             * @param C FP32 output [m, n] on device
             */
            bool multiply(
                const float *A, float *C,
                int m, int n, int k,
                bool transpose_B = true,
                float alpha = 1.0f, float beta = 0.0f,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override;

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
             */
            bool multiply_fused(
                const float *input,
                const std::vector<FusedProjectionDesc> &projections,
                int m, int k,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override;

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
             * @return true on success
             */
            bool multiply_fused_tensor(
                const TensorBase *input,
                const std::vector<TensorProjectionDesc> &projections,
                int m, int k,
                const MPIContext *mpi_ctx = nullptr) override;

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

            // =========================================================================
            // IKernelSnapshotCapable interface
            // =========================================================================

            KernelSnapshotInfo getKernelSnapshotInfo() const override;

            // =========================================================================
            // Accessors
            // =========================================================================

            int rocm_device_id() const { return rocm_device_id_; }
            size_t weight_rows() const { return N_; }
            size_t weight_cols() const { return K_; }
            bool weights_converted() const { return weights_converted_; }

        private:
            // =========================================================================
            // Internal dispatch methods
            // =========================================================================

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
             * @brief Ensure work buffers are allocated for given M
             */
            void ensureWorkBuffers(int m);

            // =========================================================================
            // Member data
            // =========================================================================

            const TensorBase *weights_ = nullptr; // Original weight tensor (null if using packed_)
            ROCmPackedWeights *packed_ = nullptr; // Pre-packed weights (owned by tensor cache)
            int rocm_device_id_;
            size_t N_; // Output features (weight rows)
            size_t K_; // Input features (weight cols)

            // Converted INT8 weight representation (cached) - only used with legacy constructor
            // When packed_ is set, these are unused (data comes from packed_->d_int8_data/d_scales)
            int8_t *d_weights_int8_ = nullptr; // [K × N] RowMajor (transposed from model [N×K])
            float *d_scales_B_ = nullptr;      // [N] per-column (per-output-feature) scales
            bool weights_converted_ = false;
            bool owns_weight_memory_ = false; // true if we allocated d_weights_int8_/d_scales_B_

            // Work buffers
            int8_t *d_A_int8_ = nullptr;   // [M × K] quantized activations
            float *d_scales_A_ = nullptr;  // [M] per-row scales
            int32_t *d_C_int32_ = nullptr; // [M × N] INT32 accumulator
            int work_buffer_M_ = 0;        // Current work buffer capacity

            // PIMPL for CK implementation (avoids CK headers in this header)
            struct Impl;
            std::unique_ptr<Impl> impl_;
        };

    } // namespace rocm
} // namespace llaminar2
