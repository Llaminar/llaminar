/**
 * @file VWeightedAccum.h
 * @brief Microkernel μK3: Weighted V vector accumulation with correction
 * @author David Sanftenberg
 * 
 * Accumulates weighted V vectors into FP32 context buffer, handling:
 * - Correction factor application when softmax max changes
 * - Q8_1 dequantization of V values
 * - Efficient vectorized accumulation
 */

#pragma once

#include "Q8DotProduct.h"  // For Q8_1Block definition
#include <cstdint>

namespace llaminar::v2::kernels::microkernels {

/**
 * @brief Parameters for weighted V accumulation
 */
struct VWeightedAccumParams {
    const Q8_1Block* v_blocks;  ///< V vector blocks for position n [num_blocks]
    float weight;               ///< Softmax weight for this position
    float correction;           ///< Rescaling factor (1.0 if no max update)
    float* context;             ///< FP32 accumulator [head_dim] (in/out)
    int num_blocks;             ///< Number of blocks (head_dim / 32)
};

/**
 * @brief Reference implementation of weighted V accumulation
 * 
 * Algorithm:
 *   // Apply correction to existing accumulation
 *   if (correction != 1.0):
 *     for d in [0, head_dim):
 *       context[d] *= correction
 *   
 *   // Add weighted V (dequantized)
 *   for each block b:
 *     d_v = fp16_to_fp32(v_blocks[b].d)
 *     for i in [0, 32):
 *       v_val = v_blocks[b].qs[i] * d_v
 *       context[b*32 + i] += weight * v_val
 * 
 * @param params Input parameters (context modified in place)
 */
void v_weighted_accum_ref(const VWeightedAccumParams& params);

/**
 * @brief AVX-512 implementation of weighted V accumulation
 * 
 * Vectorized version using:
 * - vbroadcastss for weight/correction
 * - vpmovsxbd for int8 → int32 extension
 * - vcvtdq2ps for int32 → float conversion
 * - vfmadd231ps for fused multiply-add
 * 
 * @param params Input parameters (context modified in place)
 */
void v_weighted_accum_avx512(const VWeightedAccumParams& params);

/**
 * @brief Dispatch to best available implementation
 * @param params Input parameters
 */
void v_weighted_accum(const VWeightedAccumParams& params);

/**
 * @brief Apply correction factor only (no V accumulation)
 * 
 * Useful when you need to apply correction before processing more K positions.
 * 
 * @param context FP32 accumulator [head_dim]
 * @param correction Rescaling factor
 * @param head_dim Dimension of context
 */
void apply_softmax_correction_ref(float* context, float correction, int head_dim);

/**
 * @brief AVX-512 correction factor application
 */
void apply_softmax_correction_avx512(float* context, float correction, int head_dim);

} // namespace llaminar::v2::kernels::microkernels
