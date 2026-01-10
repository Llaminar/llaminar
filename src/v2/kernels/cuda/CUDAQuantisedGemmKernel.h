/**
 * @file CUDAQuantisedGemmKernel.h
 * @brief CUDA INT8 Tensor Core GEMM kernel for quantized tensors
 *
 * Implements ITensorGemm using CUTLASS INT8 GEMM for any quantized weight tensor.
 * This is the CUDA counterpart to CPU QuantisedGemmKernel.
 *
 * **Design**:
 * - Primary entry point: multiply_tensor() with type introspection
 * - Supports any quantized weight type (IQ4_NL, Q8_0, Q4_0, Q4_K, etc.)
 * - Weights pre-converted to INT8 + per-column scales on first use
 * - Activations quantized on-the-fly or used directly if already Q8_1
 * - Uses CUTLASS INT8 Tensor Core GEMM (SM 8.0+ Ampere)
 *
 * **Type Dispatch** (in multiply_tensor):
 * | A type | C type | Path |
 * |--------|--------|------|
 * | Q8_1   | FP32   | INT8×INT8→FP32 |
 * | Q8_1   | Q8_1   | INT8×INT8→Q8_1 (fused requant) |
 * | FP32   | FP32   | quant A→INT8×INT8→FP32 |
 * | FP32   | Q8_1   | quant A→INT8×INT8→Q8_1 |
 *
 * **Usage**:
 * ```cpp
 * auto kernel = std::make_unique<CUDAQuantisedGemmKernel>(weights, cuda_device_id);
 * kernel->multiply_tensor(activations, output, ...);
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

    namespace cuda
    {

        /**
         * @struct CUDAPackedWeights
         * @brief Pre-packed INT8 weights for CUDA CUTLASS GEMM (analogous to CPU QuantisedPackedWeights)
         *
         * Stores weights converted from any quantized format to symmetric INT8 with per-column scales.
         * Unlike CPU's VNNI packing, this is simple column-major INT8 data suitable for CUTLASS.
         *
         * **Memory Layout**:
         * - `int8_data`: [K × N] ColumnMajor INT8 weights (CUTLASS Tensor Core requirement)
         * - `scales`: [N] per-column FP32 scale factors
         *
         * **Conversion**: max_abs = max(abs(col)), scale = max_abs / 127, int8 = round(fp32 / scale * 127)
         *
         * This structure is cached in tensor->cache_ to avoid re-conversion on every kernel call.
         */
        struct CUDAPackedWeights
        {
            std::vector<int8_t> int8_data; ///< [K × N] ColumnMajor INT8 weights
            std::vector<float> scales;     ///< [N] per-column scale factors
            int K = 0;                     ///< Input features (rows in CUTLASS B matrix)
            int N = 0;                     ///< Output features (cols in CUTLASS B matrix)

            // Device memory pointers (uploaded once, cached)
            int8_t *d_int8_data = nullptr; ///< Device pointer to INT8 weights
            float *d_scales = nullptr;     ///< Device pointer to scales
            int cuda_device_id = -1;       ///< Device where data is uploaded
            bool uploaded = false;         ///< Whether device memory is allocated

            ~CUDAPackedWeights();
        };

        /**
         * @brief Pack any quantized tensor to CUDAPackedWeights (host-side)
         *
         * Dequantizes the tensor to FP32, then re-quantizes symmetrically to INT8.
         * Device upload happens separately when the kernel is first used.
         *
         * @param tensor Source quantized tensor
         * @param out Output packed weights structure
         * @return true on success
         */
        bool packWeightsToCUDA(const TensorBase *tensor, CUDAPackedWeights &out);

        /**
         * @brief CUDA GEMM kernel for quantized weight tensors using CUTLASS INT8
         *
         * Implements ITensorGemm for any quantized weight tensor type.
         *
         * **Supported Weight Types**:
         * - IQ4_NLTensor
         * - Q8_0Tensor
         * - Q4_0Tensor
         * - Q4_KTensor, Q5_KTensor, Q6_KTensor
         * - Any tensor implementing dequantization
         *
         * **Compute Path**:
         * 1. Weights: Dequantize → Requantize symmetric INT8 + per-column scales
         * 2. Activations: Q8_1 blocks used directly, or FP32→INT8 quantized
         * 3. GEMM: CUTLASS INT8×INT8→INT32 (Tensor Core mma.sync.m16n8k32)
         * 4. Output: INT32 × scale_A × scale_B → FP32 (or requant to Q8_1)
         *
         * **Memory Layout**:
         * - Weights: INT8 [K × N] ColumnMajor (CUTLASS Tensor Core requirement)
         * - Activations: INT8 [M × K] RowMajor
         * - Output: FP32/INT32 [M × N] RowMajor
         *
         * **Performance**:
         * - Weight conversion: Once per tensor (cached)
         * - CUTLASS Tensor Core: 50-90 TFLOPS on RTX 3090
         * - Activation quantization: Per-row symmetric, fused with GEMM launch
         */
        class CUDAQuantisedGemmKernel : public ITensorGemm
        {
        public:
            /**
             * @brief Construct kernel for quantized weight tensor (lazy conversion)
             *
             * @param weights Any quantized tensor (must be on GPU)
             * @param cuda_device_id CUDA device ID (from cudaGetDevice, NOT global index)
             *
             * @throws std::runtime_error if weight not quantized or not on GPU
             */
            CUDAQuantisedGemmKernel(const TensorBase *weights, int cuda_device_id);

            /**
             * @brief Construct kernel from pre-packed INT8 weights (PREFERRED)
             *
             * This constructor avoids redundant weight conversion by using pre-packed
             * CUDAPackedWeights that are cached in the tensor's cache_ field.
             *
             * @param packed Pre-packed INT8 weights with scales (from KernelFactory cache)
             * @param cuda_device_id CUDA device ID
             *
             * @throws std::runtime_error if packed is null or has invalid dimensions
             */
            CUDAQuantisedGemmKernel(CUDAPackedWeights *packed, int cuda_device_id);

            ~CUDAQuantisedGemmKernel() override;

            // Non-copyable
            CUDAQuantisedGemmKernel(const CUDAQuantisedGemmKernel &) = delete;
            CUDAQuantisedGemmKernel &operator=(const CUDAQuantisedGemmKernel &) = delete;

            // Movable
            CUDAQuantisedGemmKernel(CUDAQuantisedGemmKernel &&) noexcept;
            CUDAQuantisedGemmKernel &operator=(CUDAQuantisedGemmKernel &&) noexcept;

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
             * @param mpi_ctx MPI context (unused for CUDA kernel)
             * @param device_idx Device index (unused, kernel bound to cuda_device_id_)
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
             * Quantizes FP32 activations → INT8, runs CUTLASS GEMM, outputs FP32.
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
             * @param mpi_ctx MPI context (unused for CUDA)
             * @param device_idx Device index (unused, kernel bound to cuda_device_id_)
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
             * CUDAQuantisedGemmKernel is for weight projections only.
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

            int cuda_device_id() const { return cuda_device_id_; }
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
            CUDAPackedWeights *packed_ = nullptr; // Pre-packed weights (owned by tensor cache)
            int cuda_device_id_;
            size_t N_; // Output features (weight rows)
            size_t K_; // Input features (weight cols)

            // Converted INT8 weight representation (cached) - only used with legacy constructor
            // When packed_ is set, these are unused (data comes from packed_->d_int8_data/d_scales)
            int8_t *d_weights_int8_ = nullptr; // [K × N] ColumnMajor
            float *d_scales_B_ = nullptr;      // [N] per-column scales
            bool weights_converted_ = false;
            bool owns_weight_memory_ = false; // true if we allocated d_weights_int8_/d_scales_B_
            int32_t *d_C_int32_ = nullptr;    // [M × N] INT32 accumulator
            int work_buffer_M_ = 0;           // Current work buffer capacity

            // PIMPL for CUTLASS implementation (avoids CUTLASS in header)
            struct Impl;
            std::unique_ptr<Impl> impl_;
        };

    } // namespace cuda
} // namespace llaminar2
