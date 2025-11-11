/**
 * @file IntegerGemmMicroKernel.h
 * @brief INT8 micro-kernel interface matching FP32 GEMM pattern
 *
 * Provides a clean, modular interface for INT8 micro-kernels:
 * - zero(): Initialize accumulators
 * - accumulate(): Accumulate A×B outer products
 * - reduce(): Reduce INT32 accumulators to output
 *
 * This mirrors the FP32 MicroKernel interface, enabling:
 * - Consistent API across precisions
 * - Modular micro-kernel swapping (e.g., fused softmax+gemm)
 * - Runtime tuning via template parameters
 *
 * @author David Sanftenberg
 * @date November 2025
 */

#pragma once

#include "GemmMicroKernelTemplateINT8.h"
#include <algorithm>
#include <cstring>

namespace llaminar2
{
    namespace kernels
    {
        namespace gemm
        {

            /**
             * @brief INT8 GEMM micro-kernel: Compute C[TILE_M × TILE_N] += A[TILE_M × K] * B[K × TILE_N]
             *
             * This class wraps MicroKernelTemplateINT8 to provide a simplified interface
             * matching the FP32 MicroKernel pattern, enabling modular kernel composition.
             *
             * @tparam ISA - SIMD ISA tag (must be AVX512VNNITag for INT8 support)
             * @tparam TILE_M - Number of output rows (typically 4, 8, 16, 32)
             * @tparam TILE_N - Number of output columns (typically 4, 8, 16, 32)
             * @tparam UNROLL_K - K-loop unroll factor (1, 2, 4, 8, 16)
             * @tparam PREFETCH_DIST - Prefetch distance in iterations (0, 1, 2, 3, 5)
             * @tparam MC - M-dimension cache block size (default 256)
             * @tparam KC - K-dimension cache block size (default 512, must be multiple of 4)
             * @tparam NC - N-dimension cache block size (default 128)
             *
             * Micro-kernel algorithm:
             * 1. zero() - Initialize TILE_M × TILE_N INT32 accumulators to zero
             * 2. For each K-block:
             *    - accumulate(A_panel, B_panel, k_panel) - Accumulate INT8×INT8→INT32
             * 3. reduce(C_tile, scales) - Reduce INT32 accumulators with scale factors
             *
             * Example (8×4 tile, 512 elements K):
             * ```
             * IntegerGemmMicroKernel<AVX512VNNITag, 8, 4> ukernel;
             * ukernel.zero();
             * for (int kb = 0; kb < 16; ++kb) {  // 16 blocks × 32 = 512
             *     load_panels(A, B, kb, A_panel, B_panel, a_scales, b_scales);
             *     ukernel.accumulate(A_panel, B_panel, 32);
             * }
             * ukernel.reduce(C_tile, a_scales, b_scales);
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
            class IntegerGemmMicroKernel
            {
            public:
                using BaseKernel = MicroKernelTemplateINT8<ISA, TILE_M, TILE_N, UNROLL_K, PREFETCH_DIST, MC, KC, NC>;

                static constexpr int kTileM = TILE_M;
                static constexpr int kTileN = TILE_N;
                static constexpr int kBlockSize = 32; // Q8_0 block size

                /**
                 * @brief Constructor - allocates accumulator storage
                 */
                IntegerGemmMicroKernel()
                {
                    zero();
                }

                /**
                 * @brief Zero all INT32 accumulators
                 *
                 * Call this once before processing all K-blocks for a tile.
                 */
                __attribute__((always_inline)) inline void zero()
                {
                    std::memset(accumulators_, 0, sizeof(accumulators_));
                    num_k_blocks_ = 0;
                    combined_scale_acc_ = 0.0;
                }

                /**
                 * @brief Accumulate INT8×INT8→INT32 outer product
                 *
                 * Computes: accumulators[i][j] += sum_k(A[i, k] * B[j, k]) (in INT32 domain)
                 *
                 * @param A_panel Pointer to A panel (TILE_M rows × k_panel columns, INT8, row-major)
                 * @param B_panel Pointer to B panel (TILE_N columns × k_panel rows, INT8, row-major per column)
                 * @param k_panel Number of elements per row in A_panel / per column in B_panel
                 * @param a_scales Scale factors for each A row (FP64, TILE_M elements)
                 * @param b_scales Scale factors for each B column (FP64, TILE_N elements)
                 *
                 * Layout assumptions:
                 * - A_panel[i * k_panel + k] is element (i, k)
                 * - B_panel[j * k_panel + k] is element (j, k)
                 *
                 * This method:
                 * 1. Calls BaseKernel::micro_kernel to accumulate INT8×INT8→INT32
                 * 2. Tracks combined scales for final requantization
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
                    // alpha=1, beta=0 means: block_result = 1 * A * B + 0 * block_result
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

                    // Accumulate combined scales for requantization
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
                 * @brief Reduce INT32 accumulators to Q8_0 output blocks
                 *
                 * Performs:
                 * 1. Compute average combined scale across all K-blocks
                 * 2. For each output element:
                 *    - FP32_value = INT32_accumulator * avg_combined_scale
                 *    - Quantize FP32_value → Q8_0 block
                 *
                 * @param C_blocks Output Q8_0 blocks (TILE_M × (TILE_N/32) blocks)
                 *                 Note: TILE_N must be 32 for this interface
                 *
                 * Layout: C_blocks[i * (TILE_N/32) + j_block] is block for row i, column block j_block
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

                    // Compute average combined scale
                    double avg_combined_scale = combined_scale_acc_ / (TILE_M * TILE_N * num_k_blocks_);

                    // Convert INT32 accumulators to FP32 with scale
                    alignas(64) float fp32_tile[TILE_M * TILE_N];
                    for (int i = 0; i < TILE_M; ++i)
                    {
                        for (int j = 0; j < TILE_N; ++j)
                        {
                            int32_t acc = accumulators_[i * TILE_N + j];
                            fp32_tile[i * TILE_N + j] = static_cast<float>(acc * avg_combined_scale);
                        }
                    }

                    // Quantize each row to Q8_0 (TILE_N must be 32)
                    for (int i = 0; i < TILE_M; ++i)
                    {
                        quantizeFP32ToQ8_0(&fp32_tile[i * TILE_N], &C_blocks[i], TILE_N);
                    }
                }

                /**
                 * @brief Alternative reduce() with explicit scale factors per element
                 *
                 * For advanced use cases where per-element scaling is needed.
                 *
                 * @param C_blocks Output Q8_0 blocks
                 * @param scales Scale factors per element (TILE_M × TILE_N array)
                 */
                __attribute__((always_inline)) inline void reduce_with_scales(Q8_0Block *C_blocks, const double *scales)
                {
                    static_assert(TILE_N == 32, "reduce_with_scales() requires TILE_N=32 for Q8_0 block alignment");

                    // Convert INT32 accumulators to FP32 with per-element scales
                    alignas(64) float fp32_tile[TILE_M * TILE_N];
                    for (int i = 0; i < TILE_M; ++i)
                    {
                        for (int j = 0; j < TILE_N; ++j)
                        {
                            int32_t acc = accumulators_[i * TILE_N + j];
                            fp32_tile[i * TILE_N + j] = static_cast<float>(acc * scales[i * TILE_N + j]);
                        }
                    }

                    // Quantize each row to Q8_0
                    for (int i = 0; i < TILE_M; ++i)
                    {
                        quantizeFP32ToQ8_0(&fp32_tile[i * TILE_N], &C_blocks[i], TILE_N);
                    }
                }

                /**
                 * @brief Get raw accumulator value (for debugging/diagnostics)
                 *
                 * @param i Row index (0 ≤ i < TILE_M)
                 * @param j Column index (0 ≤ j < TILE_N)
                 * @return INT32 accumulator value
                 */
                inline int32_t accumulator(int i, int j) const
                {
                    return accumulators_[i * TILE_N + j];
                }

            private:
                /**
                 * @brief Quantize FP32 array to Q8_0 block
                 *
                 * Computes:
                 * - scale = max(abs(x)) / 127.0
                 * - quant[i] = round(x[i] / scale)
                 *
                 * @param x Input FP32 array (n elements)
                 * @param block Output Q8_0 block
                 * @param n Number of elements (must be 32 for Q8_0)
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
                // Accumulates across K-blocks before final requantization
                alignas(64) int32_t accumulators_[TILE_M * TILE_N];

                // Scale tracking for requantization
                double combined_scale_acc_; // Accumulated sum of a_scale[i] * b_scale[j]
                int num_k_blocks_;          // Number of K-blocks accumulated
            };

        } // namespace gemm
    } // namespace kernels
} // namespace llaminar2
