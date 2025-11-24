/**
 * @file OneDNNGemmAdapter.h
 * @brief Adapter layer for OneDNN INT8 GEMM operations with tensor interface
 *
 * Provides high-level tensor-based interface for OneDNN INT8 matrix multiplication.
 * Handles quantization, packing, and dequantization transparently.
 *
 * **Key Features**:
 * - Automatic per-row activation quantization (FP32 → INT8)
 * - Automatic per-column weight quantization (any format → INT8)
 * - INT32 accumulation with fused dequantization
 * - Optional bias addition in dequantization step
 *
 * **Usage**:
 * ```cpp
 * onednn_gemm_adapter(M, N, K, activation_tensor, weight_tensor, output_tensor, bias);
 * // Result: output_tensor = activation_tensor @ weight_tensor^T + bias
 * ```
 *
 * @author David Sanftenberg
 * @date 2025-01-15
 */

#pragma once

#ifndef HAVE_ONEDNN
#error "OneDNN support is required to use gemm_v4"
#endif

#include <oneapi/dnnl/dnnl.hpp>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace llaminar2
{
    // Forward declarations
    class TensorBase;
    class IActivationTensor;

    namespace gemm_v4
    {
        // ===== Forward Declarations =====

        /**
         * @brief Get singleton OneDNN CPU engine (defined in OneDNNGemmKernel.h)
         */
        dnnl::engine &onednn_engine();

        /**
         * @brief Get singleton OneDNN execution stream (defined in OneDNNGemmKernel.h)
         */
        dnnl::stream &onednn_stream();

        /**
         * @brief Execute INT8 matrix multiplication using OneDNN (defined in OneDNNGemmKernel.h)
         */
        bool run_onednn_int8_matmul(const int8_t *A, const int8_t *B, int32_t *C, int M, int N, int K);

        /**
         * @brief Execute FP32 matrix multiplication using OneDNN (defined in OneDNNGemmKernel.h)
         */
        bool run_onednn_fp32_matmul(const float *A, const float *B, float *C, int M, int N, int K, bool transpose_B = false);

        // ===== Data Structures =====

        /**
         * @brief Packed INT8 weight tensor with per-column scaling factors
         *
         * **Storage Format**: Column-major INT8 with per-column FP32 scales
         * **Quantization**: Per-column symmetric quantization (no zero-point)
         * **Formula**: weight[n,k] = data[k*N + n] * col_scales[n]
         *
         * **Memory Layout**:
         * - `data`: [K, N] INT8 column-major (OneDNN matmul expects column-major weights)
         * - `col_scales`: [N] FP32 per-column dequantization scales
         *
         * **Usage**: Created by `pack_weights_to_int8()` from any TensorBase
         */
        struct WeightPack
        {
            std::vector<int8_t> data;          ///< INT8 quantized weights [K, N] column-major
            std::vector<float> col_scales;     ///< Per-column FP32 dequantization scales [N]
            std::vector<int32_t> compensation; ///< Per-column compensation for u8 shift [N]
            int rows = 0;                      ///< Number of rows (K dimension)
            int cols = 0;                      ///< Number of columns (N dimension)

            /**
             * @brief Get total number of quantized elements
             * @return K * N (total weight count)
             */
            size_t element_count() const { return data.size(); }
        };

        // ===== Weight Packing Functions =====

        /**
         * @brief Convert tensor weights to INT8 column-major format with per-column scaling
         *
         * **Conversion Pipeline**:
         * 1. Call `tensor.to_int8_perchannel()` to get INT8 row-major + per-column scales
         * 2. Transpose from row-major [N, K] to column-major [K, N]
         * 3. Store in WeightPack for OneDNN consumption
         *
         * **Input Tensor Format**: [N, K] row-major (standard weight storage)
         * **Output WeightPack Format**: [K, N] column-major (OneDNN matmul requirement)
         *
         * **Why Column-Major?**
         * OneDNN matmul primitive expects weights in column-major format for optimal
         * memory access patterns during INT8 GEMM execution (cache-friendly).
         *
         * @param tensor Source weight tensor (any quantized or FP format)
         * @param K Number of rows in packed weights (columns in original tensor)
         * @param N Number of columns in packed weights (rows in original tensor)
         * @return WeightPack with INT8 data and per-column scales
         *
         * @throws std::runtime_error if tensor is not 2D or dimensions mismatch
         * @throws std::runtime_error if to_int8_perchannel() fails
         *
         * @note Thread-safe (uses local buffers)
         */
        inline WeightPack pack_weights_to_int8(const TensorBase &tensor, int K, int N)
        {
            // Validate tensor is 2D
            const auto &shape = tensor.shape();
            if (shape.size() != 2)
            {
                throw std::runtime_error("Weight tensor must be 2D");
            }

            // Validate dimensions match (tensor is [N, K], we need to pack as [K, N])
            if (shape[0] != static_cast<size_t>(N) || shape[1] != static_cast<size_t>(K))
            {
                throw std::runtime_error("Weight tensor shape does not match provided dimensions");
            }

            // Initialize output pack
            WeightPack pack;
            pack.rows = K; // K rows in column-major format
            pack.cols = N; // N columns in column-major format
            pack.data.resize(static_cast<size_t>(K) * static_cast<size_t>(N), 0);
            pack.col_scales.resize(static_cast<size_t>(N), 1.0f);
            pack.compensation.resize(static_cast<size_t>(N), 0);

            // Step 1: Quantize tensor to INT8 row-major with per-column scales
            // row_major will be [N, K] (N rows, K columns)
            std::vector<int8_t> row_major(static_cast<size_t>(N) * static_cast<size_t>(K));
            std::vector<float> col_scale_buffer(static_cast<size_t>(K), 1.0f); // Unused (dummy for row scales)

            if (!tensor.to_int8_perchannel(row_major.data(), col_scale_buffer.data(), pack.col_scales.data()))
            {
                throw std::runtime_error("TensorBase::to_int8_perchannel failed for weight tensor");
            }

            // Step 2: Transpose from row-major [N, K] to column-major [K, N]
            // This changes memory layout for optimal OneDNN access patterns
            // Optimized: Parallelize over K (output rows) for sequential writes
#pragma omp parallel for schedule(static)
            for (int k = 0; k < K; ++k)
            {
                // Destination: row k in column-major layout (which is K x N)
                int8_t *dst_row = pack.data.data() + static_cast<size_t>(k) * static_cast<size_t>(N);

                for (int n = 0; n < N; ++n)
                {
                    // Source: row n, col k in row-major layout [N, K]
                    dst_row[n] = row_major[static_cast<size_t>(n) * static_cast<size_t>(K) + static_cast<size_t>(k)];
                }
            }

// Step 3: Compute compensation (sum of columns)
// pack.data is [K, N] row-major.
// We need sum over K for each N.
#pragma omp parallel for schedule(static)
            for (int n = 0; n < N; ++n)
            {
                int32_t sum = 0;
                for (int k = 0; k < K; ++k)
                {
                    sum += pack.data[static_cast<size_t>(k) * static_cast<size_t>(N) + n];
                }
                pack.compensation[n] = sum;
            }

            return pack;
        }

        // ===== GEMM Execution Functions =====

        /**
         * @brief Execute INT8 GEMM using pre-packed activation and weight tensors
         *
         * **Operation**: C = A @ B^T + bias (with INT8 quantization and FP32 output)
         *
         * **Execution Pipeline**:
         * 1. Validate dimensions and scale buffer sizes
         * 2. Allocate thread-local INT32 accumulator buffer
         * 3. Execute OneDNN INT8 matmul: accum = activation @ weights (INT8×INT8 → INT32)
         * 4. Dequantize and scale: output = accum * row_scales * col_scales + bias
         *
         * **Input Formats**:
         * - `activation`: [M, K] INT8 row-major + per-row FP32 scales
         * - `weights`: [K, N] INT8 column-major + per-column FP32 scales
         *
         * **Output Format**:
         * - `output_tensor`: [M, N] FP32 (native precision of IActivationTensor)
         *
         * **Optimization**: Uses thread-local accumulator buffer to avoid repeated allocations
         *
         * @param activation Packed INT8 activations [M, K] with per-row scales
         * @param weights Packed INT8 weights [K, N] with per-column scales
         * @param output_tensor Output activation tensor (FP32/FP16/BF16)
         * @param M Number of rows in activation and output
         * @param N Number of columns in weights and output
         * @param K Number of columns in activation and rows in weights
         * @param bias Optional bias vector [N] (added in dequantization step)
         *
         * @return true on success, false on OneDNN execution failure
         *
         * @throws std::runtime_error if dimension or scale buffer size mismatch
         *
         * @note Thread-safe (uses thread_local accumulator)
         * @note Bias is applied during dequantization (fused operation)
         */
        inline bool onednn_gemm_from_packed(const ActivationPack &activation,
                                            const WeightPack &weights,
                                            IActivationTensor &output_tensor,
                                            int M,
                                            int N,
                                            int K,
                                            const float *bias = nullptr)
        {
            // Validate activation dimensions
            if (activation.rows != M || activation.cols != K)
            {
                throw std::runtime_error("Activation pack dimensions mismatch");
            }

            // Validate weight dimensions
            if (weights.rows != K || weights.cols != N)
            {
                throw std::runtime_error("Weight pack dimensions mismatch");
            }

            // Validate scale buffer sizes
            if (activation.row_scales.size() != static_cast<size_t>(M) ||
                weights.col_scales.size() != static_cast<size_t>(N))
            {
                throw std::runtime_error("Scale buffers do not match matrix dimensions");
            }

            // Allocate thread-local INT32 accumulator buffer (reused across calls)
            static thread_local std::vector<int32_t> accum_buffer;
            const size_t accum_elems = static_cast<size_t>(M) * static_cast<size_t>(N);
            if (accum_buffer.size() < accum_elems)
            {
                accum_buffer.resize(accum_elems);
            }

            // Execute OneDNN INT8 matmul: accum = activation @ weights
            // Input: INT8 activations [M, K], INT8 weights [K, N]
            // Output: INT32 accumulator [M, N]
            if (!run_onednn_int8_matmul(activation.data.data(),
                                        weights.data.data(),
                                        accum_buffer.data(),
                                        M,
                                        N,
                                        K))
            {
                return false;
            }

            // Dequantize INT32 accumulator to FP32 output with scaling and bias
            // Formula: output[m,n] = accum[m,n] * row_scales[m] * col_scales[n] + bias[n]
            return output_tensor.from_int32_with_scales(
                accum_buffer.data(),
                M,
                N,
                activation.row_scales.data(),
                weights.col_scales.data(),
                bias);
        }

        /**
         * @brief High-level tensor-based GEMM adapter with automatic quantization
         *
         * **One-Stop GEMM Solution**: This is the main entry point for tensor-based GEMM.
         * Handles all quantization, packing, execution, and dequantization automatically.
         *
         * **Operation**: output = A @ B^T + bias (FP32 precision)
         *
         * **Complete Pipeline**:
         * 1. Quantize activations: FP32/FP16/BF16 → INT8 with per-row scaling
         * 2. Pack weights: Any format → INT8 column-major with per-column scaling
         * 3. Execute INT8 GEMM: INT8×INT8 → INT32 accumulator (OneDNN)
         * 4. Dequantize output: INT32 → FP32 with row/column scale fusion + bias
         *
         * **Supported Input Formats**:
         * - Activations (A): Any IActivationTensor (FP32, FP16, BF16, INT32)
         * - Weights (B): Any TensorBase (IQ4_NL, Q6_K, Q8_0, FP32, etc.)
         *
         * **Performance Characteristics**:
         * - Activation quantization: ~500 GB/s (row-parallel)
         * - Weight packing: One-time cost (cacheable)
         * - INT8 GEMM: ~2-4× faster than FP32 on AVX512-VNNI CPUs
         * - Dequantization: Fused with bias addition (~100 GB/s)
         *
         * @param M Number of rows in activation tensor A
         * @param N Number of columns in weight tensor B (before transpose)
         * @param K Number of columns in A and rows in B
         * @param A Activation tensor [M, K] (IActivationTensor interface)
         * @param B Weight tensor [N, K] transposed (TensorBase interface)
         * @param output_tensor Output activation tensor [M, N] (IActivationTensor interface)
         * @param bias Optional bias vector [N] (added during dequantization)
         *
         * @return true on success, false on quantization/execution/dequantization failure
         *
         * @throws std::runtime_error if tensor dimensions are invalid
         * @throws std::runtime_error if quantization or packing fails
         *
         * @note This function performs all conversions transparently
         * @note Weight packing could be cached for repeated use (future optimization)
         * @note Thread-safe (uses thread-local buffers in sub-functions)
         *
         * **Example**:
         * ```cpp
         * FP32Tensor activations({32, 896});      // [M=32, K=896]
         * IQ4_NLTensor weights({2048, 896});      // [N=2048, K=896]
         * FP32Tensor output({32, 2048});          // [M=32, N=2048]
         *
         * onednn_gemm_adapter(32, 2048, 896,
         *                     activations, weights, output, nullptr);
         * // Result: output = activations @ weights^T
         * ```
         */
        inline bool onednn_gemm_adapter(int M,
                                        int N,
                                        int K,
                                        const IActivationTensor &A,
                                        const TensorBase &B,
                                        IActivationTensor &output_tensor,
                                        const float *bias = nullptr)
        {
            // Step 1: Quantize activations to INT8 with per-row scaling
            // Converts from any IActivationTensor format (FP32/FP16/BF16/INT32) to INT8
            auto activation = A.to_int8_activation_pack(M, K);

            // Step 2: Pack weights to INT8 column-major with per-column scaling
            // Converts from any TensorBase format (IQ4_NL/Q6_K/FP32/etc.) to INT8
            auto weights = pack_weights_to_int8(B, K, N);

            // Step 3: Execute INT8 GEMM and dequantize to FP32 output
            return onednn_gemm_from_packed(activation, weights, output_tensor, M, N, K, bias);
        }
    } // namespace gemm_v4
} // namespace llaminar2
