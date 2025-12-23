/**
 * @file Q8DotProduct.h
 * @brief Microkernel μK1: Q8_1 dot product with proper scaling
 * @author David Sanftenberg
 *
 * Computes dot product between two Q8_1 quantized vectors, handling:
 * - Per-block FP16 scale factors
 * - Unsigned conversion for vpdpbusd (Q + 128)
 * - sum_qs bias correction
 */

#pragma once

#include <cstdint>
#include "tensors/BlockStructures.h" // Use canonical Q8_1Block definition

namespace llaminar::v2::kernels::microkernels
{

    // Use canonical Q8_1Block from tensors/BlockStructures.h
    using llaminar2::Q8_1Block;

    /**
     * @brief Parameters for Q8_1 dot product computation
     */
    struct Q8DotProductParams
    {
        const Q8_1Block *q_blocks = nullptr; ///< Q vector blocks [num_blocks]
        const Q8_1Block *k_blocks = nullptr; ///< K vector blocks [num_blocks]
        int num_blocks = 0;                  ///< Number of blocks (head_dim / 32)
        float global_scale = 0.0f;           ///< Pre-multiplied scale (e.g., 1/sqrt(d))
    };

    /**
     * @brief Result of Q8_1 dot product
     */
    struct Q8DotProductResult
    {
        float score = 0.0f; ///< Final scaled dot product
    };

    /**
     * @brief Reference implementation of Q8_1 dot product
     *
     * Algorithm:
     *   score = 0
     *   for each block b:
     *     d_q = fp16_to_fp32(q_blocks[b].d)
     *     d_k = fp16_to_fp32(k_blocks[b].d)
     *     block_scale = d_q * d_k
     *
     *     int32 dot = 0
     *     for i in [0, 32):
     *       dot += (q_qs[i] + 128) * k_qs[i]  // unsigned × signed
     *     dot -= 128 * k_blocks[b].sum_qs     // bias correction
     *
     *     score += dot * block_scale
     *   return score * global_scale
     *
     * @param params Input parameters
     * @return Dot product result
     */
    Q8DotProductResult q8_dot_product_ref(const Q8DotProductParams &params);

    /**
     * @brief AVX-512 VNNI implementation of Q8_1 dot product
     *
     * Uses vpdpbusd instruction for efficient unsigned × signed dot product.
     * Requires AVX-512 VNNI support.
     *
     * @param params Input parameters
     * @return Dot product result (matches reference within FP tolerance)
     */
    Q8DotProductResult q8_dot_product_avx512(const Q8DotProductParams &params);

    /**
     * @brief Dispatch to best available implementation
     *
     * @param params Input parameters
     * @return Dot product result
     */
    Q8DotProductResult q8_dot_product(const Q8DotProductParams &params);

} // namespace llaminar::v2::kernels::microkernels
