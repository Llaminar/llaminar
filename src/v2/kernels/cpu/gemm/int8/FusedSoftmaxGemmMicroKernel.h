/**
 * @file FusedSoftmaxGemmMicroKernel.h
 * @brief Fused Softmax+GEMM micro-kernel for attention mechanism
 *
 * Implements a microkernel that fuses:
 * 1. INT8×INT8→INT32 GEMM (compute attention scores)
 * 2. Softmax normalization (convert to probabilities)
 * 3. Optional: Second GEMM with values (attend)
 *
 * This demonstrates the modularity of the microkernel architecture - we can
 * plug this into IntegerGemmKernel to get fused attention operations without
 * modifying the outer loop orchestration.
 *
 * Use case: Attention mechanism
 * ```
 * Standard (2 separate operations):
 *   scores = Q × K^T              // GEMM
 *   weights = softmax(scores)     // Softmax
 *
 * Fused (single operation):
 *   weights = softmax(Q × K^T)    // Fused Softmax+GEMM
 * ```
 *
 * Benefits:
 * - Single pass over data (better cache efficiency)
 * - Reduced memory bandwidth (no intermediate storage)
 * - Lower latency (no kernel launch overhead)
 *
 * @author David Sanftenberg
 * @date November 2025
 */

#pragma once

#include "GemmMicroKernelTemplateINT8.h"
#include "VectorizedSoftmax.h"                   // Vectorized softmax implementations
#include "../../../../tensors/BlockStructures.h" // For Q8_0Block
#include "../../../../tensors/FP16Utils.h"       // For fp16_to_fp32, fp32_to_fp16
#include <algorithm>
#include <cstring>
#include <cmath>

namespace llaminar2
{
    namespace kernels
    {
        namespace gemm
        {

            /**
             * @brief Fused Softmax+GEMM micro-kernel
             *
             * Computes: C[TILE_M × TILE_N] = softmax(A[TILE_M × K] * B[K × TILE_N])
             *
             * This microkernel matches the IntegerGemmMicroKernel interface but applies
             * softmax normalization during the reduction phase, enabling:
             * - Attention score computation (Q×K^T)
             * - Normalized dot products
             * - Probability distributions from logits
             *
             * @tparam ISA - SIMD ISA tag (must be AVX512VNNITag for INT8 support)
             * @tparam TILE_M - Number of output rows (typically 4, 8, 16, 32)
             * @tparam TILE_N - Number of output columns (typically 32 for Q8_0)
             * @tparam UNROLL_K - K-loop unroll factor (1, 2, 4, 8, 16)
             * @tparam PREFETCH_DIST - Prefetch distance in iterations (0, 1, 2, 3, 5)
             * @tparam MC - M-dimension cache block size (default 256)
             * @tparam KC - K-dimension cache block size (default 512, must be multiple of 4)
             * @tparam NC - N-dimension cache block size (default 128)
             *
             * Algorithm:
             * ```
             * zero():
             *     accumulators[TILE_M][TILE_N] = 0 (INT32)
             *
             * accumulate(A, B, k_panel, a_scales, b_scales):
             *     INT32_scores += INT8(A) × INT8(B)  // VNNI
             *     Track combined_scale for dequantization
             *
             * reduce(C):
             *     For each row i:
             *         FP32_scores[j] = INT32_scores[j] * combined_scale
             *         max_score = max(FP32_scores)
             *         exp_sum = sum(exp(FP32_scores - max_score))  // Stable softmax
             *         weights[j] = exp(FP32_scores[j] - max_score) / exp_sum
             *         Quantize weights → Q8_0 output
             * ```
             */
            template <
                typename ISA,
                int TILE_M,
                int TILE_N,
                int UNROLL_K = 4,
                int PREFETCH_DIST = 2,
                int MC = 256,
                int KC = 512,
                int NC = 128>
            class FusedSoftmaxGemmMicroKernel
            {
            public:
                using BaseKernel = MicroKernelTemplateINT8<ISA, TILE_M, TILE_N, UNROLL_K, PREFETCH_DIST, MC, KC, NC>;

                static constexpr int kTileM = TILE_M;
                static constexpr int kTileN = TILE_N;
                static constexpr int kBlockSize = 32; // Q8_0 block size

                /**
                 * @brief Constructor
                 */
                FusedSoftmaxGemmMicroKernel()
                {
                    zero();
                }

                /**
                 * @brief Zero all INT32 accumulators and scale trackers
                 */
                __attribute__((always_inline)) inline void zero()
                {
                    std::memset(accumulators_, 0, sizeof(accumulators_));
                    num_k_blocks_ = 0;
                    combined_scale_acc_ = 0.0;
                }

                /**
                 * @brief Accumulate INT8×INT8→INT32 outer product (GEMM phase)
                 *
                 * This phase is identical to standard integer GEMM - we accumulate
                 * attention scores as INT32 values. The softmax is applied later
                 * during reduce().
                 *
                 * @param A_panel Pointer to A panel (TILE_M rows × k_panel columns, INT8)
                 * @param B_panel Pointer to B panel (TILE_N columns × k_panel rows, INT8)
                 * @param k_panel Number of elements per row/column
                 * @param a_scales Scale factors for each A row (FP64, TILE_M elements)
                 * @param b_scales Scale factors for each B column (FP64, TILE_N elements)
                 */
                __attribute__((always_inline)) inline void accumulate(
                    const int8_t *__restrict__ A_panel,
                    const int8_t *__restrict__ B_panel,
                    int k_panel,
                    const double *__restrict__ a_scales,
                    const double *__restrict__ b_scales)
                {
                    // Temporary buffer for this K-block's contribution
                    alignas(64) int32_t block_result[TILE_M * TILE_N];
                    std::memset(block_result, 0, sizeof(block_result));

                    // Call INT8 micro-kernel (accumulates INT8×INT8→INT32)
                    BaseKernel::micro_kernel(
                        A_panel,
                        B_panel,
                        block_result,
                        TILE_N, // ldc (leading dimension of C tile)
                        k_panel,
                        1,      // alpha (INT32 scaling)
                        0,      // beta (overwrite, not accumulate)
                        TILE_M, // mr (actual rows)
                        TILE_N  // nr (actual columns)
                    );

                    // Accumulate INT32 results into persistent accumulators
                    for (int i = 0; i < TILE_M; ++i)
                    {
                        for (int j = 0; j < TILE_N; ++j)
                        {
                            accumulators_[i * TILE_N + j] += block_result[i * TILE_N + j];
                        }
                    }

                    // Accumulate combined scales for dequantization
                    for (int i = 0; i < TILE_M; ++i)
                    {
                        for (int j = 0; j < TILE_N; ++j)
                        {
                            combined_scale_acc_ += a_scales[i] * b_scales[j];
                        }
                    }

                    num_k_blocks_++;
                }

                /**
                 * @brief Reduce INT32 accumulators with fused softmax normalization
                 *
                 * This is where the fusion happens:
                 * 1. Dequantize INT32 → FP32 attention scores
                 * 2. Apply softmax row-wise (stable implementation with max subtraction)
                 * 3. Quantize normalized weights → Q8_0 output
                 *
                 * @param C_blocks Output Q8_0 blocks (TILE_M × (TILE_N/32) blocks)
                 *
                 * Softmax formula (stable version):
                 * ```
                 * max_i = max(scores[i, :])
                 * exp_scores[i, j] = exp(scores[i, j] - max_i)
                 * sum_i = sum(exp_scores[i, :])
                 * weights[i, j] = exp_scores[i, j] / sum_i
                 * ```
                 */
                __attribute__((always_inline)) inline void reduce(Q8_0Block *C_blocks)
                {
                    static_assert(TILE_N == 32, "reduce() requires TILE_N=32 for Q8_0 block alignment");

                    if (num_k_blocks_ == 0)
                    {
                        // No accumulation happened, output zeros
                        std::memset(C_blocks, 0, TILE_M * sizeof(Q8_0Block));
                        return;
                    }

                    // Compute average combined scale for dequantization
                    double avg_combined_scale = combined_scale_acc_ / (TILE_M * TILE_N * num_k_blocks_);

                    // Process each row independently (softmax is row-wise operation)
                    for (int i = 0; i < TILE_M; ++i)
                    {
                        // Step 1: Dequantize INT32 → FP32 attention scores
                        alignas(64) float fp32_scores[TILE_N];
                        for (int j = 0; j < TILE_N; ++j)
                        {
                            int32_t acc = accumulators_[i * TILE_N + j];
                            fp32_scores[j] = static_cast<float>(acc * avg_combined_scale);
                        }

                        // Step 2: Apply softmax (stable implementation)
                        alignas(64) float softmax_weights[TILE_N];
                        applySoftmax(fp32_scores, softmax_weights, TILE_N);

                        // Step 3: Quantize softmax weights → Q8_0
                        quantizeFP32ToQ8_0(softmax_weights, &C_blocks[i], TILE_N);
                    }
                }

                /**
                 * @brief Alternative reduce with temperature scaling
                 *
                 * Useful for attention with temperature parameter (common in transformers).
                 *
                 * @param C_blocks Output Q8_0 blocks
                 * @param temperature Temperature scaling factor (>0, typically 0.5-2.0)
                 *                    - temperature < 1.0: Sharper distribution (more confident)
                 *                    - temperature = 1.0: Standard softmax
                 *                    - temperature > 1.0: Smoother distribution (less confident)
                 */
                __attribute__((always_inline)) inline void reduce_with_temperature(Q8_0Block *C_blocks, float temperature)
                {
                    static_assert(TILE_N == 32, "reduce_with_temperature() requires TILE_N=32");

                    if (num_k_blocks_ == 0 || temperature <= 0.0f)
                    {
                        std::memset(C_blocks, 0, TILE_M * sizeof(Q8_0Block));
                        return;
                    }

                    double avg_combined_scale = combined_scale_acc_ / (TILE_M * TILE_N * num_k_blocks_);

                    for (int i = 0; i < TILE_M; ++i)
                    {
                        // Dequantize with temperature scaling
                        alignas(64) float fp32_scores[TILE_N];
                        for (int j = 0; j < TILE_N; ++j)
                        {
                            int32_t acc = accumulators_[i * TILE_N + j];
                            fp32_scores[j] = static_cast<float>(acc * avg_combined_scale) / temperature;
                        }

                        // Apply softmax
                        alignas(64) float softmax_weights[TILE_N];
                        applySoftmax(fp32_scores, softmax_weights, TILE_N);

                        // Quantize
                        quantizeFP32ToQ8_0(softmax_weights, &C_blocks[i], TILE_N);
                    }
                }

                /**
                 * @brief Get raw accumulator value (for debugging)
                 */
                inline int32_t accumulator(int i, int j) const
                {
                    return accumulators_[i * TILE_N + j];
                }

                /**
                 * @brief Get raw FP32 scores before softmax (for debugging)
                 */
                inline void get_scores(float *scores_out) const
                {
                    if (num_k_blocks_ == 0)
                    {
                        std::memset(scores_out, 0, TILE_M * TILE_N * sizeof(float));
                        return;
                    }

                    double avg_combined_scale = combined_scale_acc_ / (TILE_M * TILE_N * num_k_blocks_);
                    for (int i = 0; i < TILE_M; ++i)
                    {
                        for (int j = 0; j < TILE_N; ++j)
                        {
                            int32_t acc = accumulators_[i * TILE_N + j];
                            scores_out[i * TILE_N + j] = static_cast<float>(acc * avg_combined_scale);
                        }
                    }
                }

            private:
                /**
                 * @brief Apply stable softmax normalization
                 *
                 * Implements numerically stable softmax:
                 * ```
                 * max_x = max(x)
                 * exp_x[i] = exp(x[i] - max_x)
                 * sum_exp = sum(exp_x)
                 * softmax[i] = exp_x[i] / sum_exp
                 * ```
                 *
                 * Subtracting max_x prevents overflow in exp() while maintaining
                 * mathematical equivalence.
                 *
                 * Dispatches to ISA-specific vectorized implementations:
                 * - AVX512: 16-wide SIMD (fastest)
                 * - AVX2: 8-wide SIMD
                 * - Scalar: Fallback (reference implementation)
                 *
                 * @param x Input scores (FP32 array, n elements)
                 * @param softmax_out Output probabilities (FP32 array, n elements)
                 * @param n Number of elements
                 */
                static void applySoftmax(const float *x, float *softmax_out, int n)
                {
                    // Dispatch to vectorized softmax based on ISA tag
                    VectorizedSoftmax<ISA>::apply(x, softmax_out, n);
                }

                /**
                 * @brief Quantize FP32 array to Q8_0 block
                 *
                 * Same as IntegerGemmMicroKernel, provided here for completeness.
                 */
                static void quantizeFP32ToQ8_0(const float *x, Q8_0Block *block, int n)
                {
                    // Find max absolute value
                    float amax = 0.0f;
                    for (int i = 0; i < n; ++i)
                    {
                        float abs_val = std::fabs(x[i]);
                        if (abs_val > amax)
                            amax = abs_val;
                    }

                    // Compute scale (d = max / 127.0, stored in FP16)
                    float scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
                    float inv_scale = 1.0f / scale;
                    block->d = fp32_to_fp16(scale);

                    // Quantize to INT8 [-127, 127]
                    for (int i = 0; i < n; ++i)
                    {
                        float scaled = x[i] * inv_scale;
                        int32_t q = static_cast<int32_t>(std::round(scaled));
                        q = std::max(-127, std::min(127, q)); // Clamp to [-127, 127]
                        block->qs[i] = static_cast<int8_t>(q);
                    }
                }

                // INT32 accumulator array (TILE_M × TILE_N)
                alignas(64) int32_t accumulators_[TILE_M * TILE_N];

                // Scale tracking for dequantization
                double combined_scale_acc_;
                int num_k_blocks_;
            };

        } // namespace gemm
    } // namespace kernels
} // namespace llaminar2
