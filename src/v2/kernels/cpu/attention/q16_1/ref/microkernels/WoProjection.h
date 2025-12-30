/**
 * @file WoProjection.h
 * @brief Integer Wo projection microkernel: INT32 context → Q16_1 output
 *
 * This microkernel projects the INT32 context (from P×V accumulation) through
 * the Wo weight matrix to produce Q16_1 output blocks. This is the final
 * computation before residual addition.
 *
 * ## Algorithm
 *
 * For each output row d in [0, d_model):
 *   proj[d] = sum_{h=0}^{n_heads-1} sum_{i=0}^{head_dim-1} context[h*head_dim+i] × Wo[d, h*head_dim+i]
 *
 * ## Integer Pipeline
 *
 * 1. INT32 context normalized (divide by softmax sum) → INT16 range
 * 2. Wo weights: Q16_1 format (INT16 quantized with scale)
 * 3. INT16 × INT16 → INT32 accumulation (VPDPWSSD in JIT)
 * 4. INT32 accumulators → Q16_1 output blocks (scale computed from max)
 *
 * ## Memory Patterns
 *
 * - **Flash Decode**: GEMV pattern, stream Wo rows, context in L2
 * - **FA2 Prefill**: Batched GEMM, amortize Wo loads across queries
 *
 * @see docs/v2/PROJECT_Q16_INTEGER_ATTENTION_V2.md
 */

#pragma once

#include "tensors/BlockStructures.h"
#include <cstdint>

namespace llaminar2::kernels::q16_1::microkernels
{

    // Import Q16 types from BlockStructures.h
    using llaminar2::Q16_1Block_128;
    using llaminar2::Q16_1Block_64;
    using llaminar2::Q16BlockSize;

    // ============================================================================
    // Normalization: INT32 → INT16 for Wo multiply
    // ============================================================================

    /**
     * @brief Normalize INT32 context to INT16 range for Wo multiplication.
     *
     * The P×V accumulation produces large INT32 values. Before Wo projection,
     * we need to normalize these to INT16 range [-32768, 32767] to maintain
     * precision in the VPDPWSSD multiply.
     *
     * @param context_int32 Input INT32 context [n_heads × head_dim]
     * @param context_int16 Output INT16 normalized context [n_heads × head_dim]
     * @param context_scale Output scale factor (context_int32 = context_int16 × scale)
     * @param num_elements Total elements (n_heads × head_dim)
     */
    void q16_context_normalize_to_int16(
        const int32_t *context_int32,
        int16_t *context_int16,
        float &context_scale,
        int num_elements);

    // ============================================================================
    // Flash Decode: Single Row Wo Projection (GEMV)
    // ============================================================================

    /**
     * @brief Single-row Wo projection for decode.
     *
     * Computes: output[d] = context_int16 · Wo[d, :] for one output dimension.
     *
     * @tparam BlockType Q16_1 block type (Q16_1Block_64 or Q16_1Block_128)
     *
     * @param context_int16 Normalized INT16 context [n_heads × head_dim]
     * @param Wo Single row of Wo weights in Q16_1 format
     * @param output_int32 Output INT32 accumulator (single value)
     * @param input_dim Total input dimension (n_heads × head_dim)
     * @param blocks_per_input Number of Q16_1 blocks per input dim
     */
    template <typename BlockType>
    void q16_wo_row_gemv(
        const int16_t *context_int16,
        const BlockType *Wo,
        int32_t &output_int32,
        int input_dim,
        int blocks_per_input);

    // ============================================================================
    // Full Wo Projection (GEMV for decode)
    // ============================================================================

    /**
     * @brief Full Wo projection for decode: context × Wo^T → output.
     *
     * Projects the concatenated context from all heads through Wo to produce
     * the final projection. Output is written as Q16_1 blocks.
     *
     * @tparam BlockType Q16_1 block type
     *
     * @param context_int32 INT32 context from P×V [n_heads × head_dim]
     * @param Wo Wo weight matrix in Q16_1 [d_model × n_heads × head_dim]
     * @param output Q16_1 output blocks [d_model / block_size blocks]
     * @param d_model Output dimension
     * @param input_dim Input dimension (n_heads × head_dim)
     * @param blocks_per_input Blocks per input row
     * @param blocks_per_output Blocks per output
     */
    template <typename BlockType>
    void q16_wo_projection(
        const int32_t *context_int32,
        const BlockType *Wo,
        BlockType *output,
        int d_model,
        int input_dim,
        int blocks_per_input,
        int blocks_per_output);

    // ============================================================================
    // FA2 Prefill: Batched Wo Projection (GEMM)
    // ============================================================================

    /**
     * @brief Batched Wo projection for prefill.
     *
     * Projects multiple queries through Wo simultaneously to amortize
     * weight matrix loads. Each query's context is normalized independently.
     *
     * @tparam BlockType Q16_1 block type
     *
     * @param context_int32 INT32 contexts [batch × n_heads × head_dim]
     * @param Wo Wo weight matrix [d_model × n_heads × head_dim]
     * @param output Q16_1 output [batch × d_model / block_size blocks]
     * @param batch_size Number of queries in batch
     * @param d_model Output dimension
     * @param input_dim Input dimension per query
     * @param context_stride Stride between query contexts
     * @param output_stride Stride between query outputs (in blocks)
     * @param blocks_per_input Blocks per input dimension
     * @param blocks_per_output Blocks per output dimension
     */
    template <typename BlockType>
    void q16_wo_projection_batched(
        const int32_t *context_int32,
        const BlockType *Wo,
        BlockType *output,
        int batch_size,
        int d_model,
        int input_dim,
        int context_stride,
        int output_stride,
        int blocks_per_input,
        int blocks_per_output);

    // ============================================================================
    // Output Quantization: INT32 → Q16_1
    // ============================================================================

    /**
     * @brief Quantize INT32 accumulators to Q16_1 blocks.
     *
     * Takes a contiguous buffer of INT32 values and packs them into Q16_1 blocks.
     * Computes the optimal scale per block based on max absolute value.
     *
     * @tparam BlockType Q16_1 block type
     *
     * @param accumulators INT32 input values
     * @param output Q16_1 output blocks
     * @param num_values Total number of values
     * @param input_scale Scale factor from context normalization
     * @param blocks_per_output Number of output blocks
     */
    template <typename BlockType>
    void q16_quantize_to_q16_1(
        const int32_t *accumulators,
        BlockType *output,
        int num_values,
        float input_scale,
        int blocks_per_output);

    // ============================================================================
    // Dispatch Functions (Runtime Block Size Selection)
    // ============================================================================

    /**
     * @brief Runtime dispatch for Wo projection based on block size.
     */
    void q16_wo_projection_dispatch(
        const int32_t *context_int32,
        const void *Wo,
        void *output,
        int d_model,
        int input_dim,
        Q16BlockSize block_size);

    /**
     * @brief Runtime dispatch for batched Wo projection.
     */
    void q16_wo_projection_batched_dispatch(
        const int32_t *context_int32,
        const void *Wo,
        void *output,
        int batch_size,
        int d_model,
        int input_dim,
        int context_stride,
        int output_stride,
        Q16BlockSize block_size);

} // namespace llaminar2::kernels::q16_1::microkernels
