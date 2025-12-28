/**
 * @file Q16FusedAttentionRef.h
 * @brief Reference implementation of Q16_1 fused attention (PURE INTEGER PIPELINE)
 *
 * This is the scalar C++ reference implementation for the Q16 integer-domain
 * attention kernel. It serves as:
 * 1. Ground truth for JIT kernel validation (bit-exact or tolerance match)
 * 2. Readable, debuggable implementation for algorithm development
 * 3. Fallback when JIT is not available
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * PURE INTEGER-DOMAIN PIPELINE (NO FP32 INTERMEDIATE!)
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * The entire pipeline stays in integer domain until final Q16_1 output:
 *
 * 1. Q×K^T dot products: INT16×INT16 → INT32 scores (via Int8Requant)
 * 2. Exp2FixedSoftmax: INT32 → INT16 attention weights [0, 32767]
 * 3. P×V accumulation: INT16×INT16 → INT32 accumulators (NOT FP32!)
 * 4. Wo projection (VPDPWSSD): INT16×INT16 → INT32 → Q16_1 output
 * 5. Native Q16_1 residual add: Q16_1 + Q16_1 → Q16_1
 *
 * No FP32 intermediate values anywhere in the attention-to-residual path!
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Two execution paths with DIFFERENT algorithms:
 *
 * FLASH DECODE (seq_len_q=1):
 * - Single query against full KV cache
 * - Parallel over KV positions, no tiling
 * - Single-pass softmax (no online state)
 * - Optimized for latency (token generation)
 *
 * FA2 PREFILL (seq_len_q>1):
 * - Batched queries, tiled processing
 * - Per-query softmax
 * - Optimized for throughput (prompt processing)
 *
 * @see PROJECT_Q16_INTEGER_ATTENTION.md for full design document
 * @see Test__Q16IntegerDomainWoProjection.cpp for end-to-end validation
 *
 * @author David Sanftenberg
 * @date December 2025
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <limits>
#include "tensors/BlockStructures.h"
#include "kernels/cpu/gemm_v4/QuantisedGemmJit_M1.h"

namespace llaminar2
{

    // Forward declare from BlockStructures.h
    using Q16_1Block = llaminar2::Q16_1Block;

    // Forward declare VNNI packed weights
    namespace gemm_v4
    {
        struct QuantisedPackedWeights;
    }

    namespace kernels::q16_1
    {

        // ============================================================================
        // Q16 Fused Attention + Wo + Residual Parameters (Full Fused Kernel)
        // ============================================================================

        /**
         * @brief Parameters for the full fused attention block (PURE INTEGER PIPELINE)
         *
         * This includes complete attention + Wo projection + residual add:
         * 1. Q×K^T dot products → INT32 scores
         * 2. Exp2FixedSoftmax → INT16 weights
         * 3. P×V accumulation → INT32 accumulators
         * 4. Wo projection (VPDPWSSD) → Q16_1 output
         * 5. Native Q16_1 residual add → Q16_1 output
         *
         * Execution paths:
         * - FLASH DECODE: Single-pass over KV, no tiling
         * - FA2 PREFILL: Per-query processing
         *
         * Input: Q, K, V as Q16_1, Wo as VNNI-packed, residual as Q16_1
         * Output: Updated residual as Q16_1 (in-place or out-of-place)
         */
        struct Q16FusedAttentionWoResidualParams
        {
            // ============== Attention tensors (Q16_1) ==============

            /** Query tensor: Q16_1 blocks [seq_len_q × (num_heads * head_dim)/32 blocks] */
            const Q16_1Block *Q = nullptr;

            /** Key tensor: Q16_1 blocks [kv_len × (num_kv_heads * head_dim)/32 blocks] */
            const Q16_1Block *K = nullptr;

            /** Value tensor: Q16_1 blocks [kv_len × (num_kv_heads * head_dim)/32 blocks] */
            const Q16_1Block *V = nullptr;

            // ============== Wo projection weights (VNNI-packed for VPDPWSSD) ==============

            /** VNNI-packed Wo weights for VPDPWSSD (INT16×INT16→INT32)
             *  Created from Q8_1 via KernelFactory::ensurePackedWeightsInTensorCache()
             */
            const gemm_v4::QuantisedPackedWeights *Wo_packed = nullptr;

            // ============== Residual tensors (Q16_1) ==============

            /** Input residual: Q16_1 blocks [seq_len_q × d_model/32 blocks] */
            const Q16_1Block *residual_in = nullptr;

            /** Output residual: Q16_1 blocks [seq_len_q × d_model/32 blocks]
             *  Can be same as residual_in for in-place update */
            Q16_1Block *residual_out = nullptr;

            // ============== Dimensions ==============

            int seq_len_q = 0;    ///< Number of query positions
            int kv_len = 0;       ///< Number of KV positions (cache length)
            int num_heads = 0;    ///< Number of query heads
            int num_kv_heads = 0; ///< Number of KV heads (for GQA/MQA)
            int head_dim = 0;     ///< Dimension per head (typically 64 or 128)
            int d_model = 0;      ///< Model dimension (num_heads * head_dim for most models)

            // ============== FA2 Prefill Tiling (ignored for Flash Decode) ==============

            int Bc = 256; ///< KV tile size for FA2 prefill (decode ignores, processes all KV)
            int Br = 16;  ///< Query tile size for FA2 prefill

            // ============== Attention config ==============

            float scale = 0.0f;      ///< 1/sqrt(head_dim), computed if 0
            bool causal = true;      ///< Apply causal masking
            int position_offset = 0; ///< Position offset for causal mask

            // ============== Snapshot/Debug config ==============

            /** Optional FP32 buffer for attention context snapshot [seq_len_q × num_heads × head_dim]
             *  If provided, INT32 context will be converted to FP32 and stored here before Wo projection.
             *  This enables debugging and parity testing with FP32 reference implementations.
             */
            float *context_snapshot = nullptr;

            // ============== Helper methods ==============

            float get_scale() const
            {
                if (scale > 0.0f)
                    return scale;
                return 1.0f / std::sqrt(static_cast<float>(head_dim));
            }

            bool is_decode() const { return seq_len_q == 1; }

            int get_kv_head(int query_head) const
            {
                return query_head / (num_heads / num_kv_heads);
            }

            int q_blocks_per_row() const
            {
                return (num_heads * head_dim + Q16_1Block::BLOCK_SIZE - 1) / Q16_1Block::BLOCK_SIZE;
            }

            int kv_blocks_per_row() const
            {
                return (num_kv_heads * head_dim + Q16_1Block::BLOCK_SIZE - 1) / Q16_1Block::BLOCK_SIZE;
            }

            int residual_blocks_per_row() const
            {
                return (d_model + Q16_1Block::BLOCK_SIZE - 1) / Q16_1Block::BLOCK_SIZE;
            }
        };

        // ============================================================================
        // Reference Implementation API - Full Fused Kernel (PURE INTEGER)
        // ============================================================================

        /**
         * @brief Execute full fused attention block (FLASH DECODE path, PURE INTEGER)
         *
         * FLASH DECODE: Single query against full KV cache.
         * - Q×K^T → INT32 scores (via Int8Requant)
         * - Exp2FixedSoftmax → INT16 weights
         * - P×V → INT32 accumulators (NOT FP32!)
         * - VPDPWSSD Wo projection → Q16_1 output
         * - Native Q16_1 residual add
         *
         * Optimized for latency in autoregressive token generation.
         *
         * @param params Full kernel parameters (seq_len_q must be 1)
         * @return true on success, false on validation error
         */
        bool q16_fused_attention_wo_residual_decode(const Q16FusedAttentionWoResidualParams &params);

        /**
         * @brief Execute full fused attention block (FA2 PREFILL path, PURE INTEGER)
         *
         * FA2 PREFILL: Batched queries with per-query processing.
         * - Per-query: Q×K^T → Softmax → P×V (all INT32 domain)
         * - VPDPWSSD Wo projection → Q16_1 output
         * - Native Q16_1 residual add
         *
         * Optimized for throughput in prompt processing.
         *
         * @param params Full kernel parameters (seq_len_q > 1)
         * @return true on success, false on validation error
         */
        bool q16_fused_attention_wo_residual_prefill(const Q16FusedAttentionWoResidualParams &params);

        /**
         * @brief Auto-dispatch full fused kernel based on seq_len_q
         */
        inline bool q16_fused_attention_wo_residual_reference(const Q16FusedAttentionWoResidualParams &params)
        {
            if (params.is_decode())
            {
                return q16_fused_attention_wo_residual_decode(params);
            }
            else
            {
                return q16_fused_attention_wo_residual_prefill(params);
            }
        }

        // ============================================================================
        // Validation
        // ============================================================================

        /**
         * @brief Validate full fused kernel parameters
         */
        bool q16_validate_full_params(const Q16FusedAttentionWoResidualParams &params);

        // ============================================================================
        // Utility Functions (exposed for testing)
        // ============================================================================

        /**
         * @brief Get INT16 value from Q16_1Block at specified position
         */
        inline int16_t q16_block_value(
            const Q16_1Block *blocks,
            int row,
            int col,
            int blocks_per_row)
        {
            constexpr int BLOCK_SIZE = Q16_1Block::BLOCK_SIZE;

            int block_idx = col / BLOCK_SIZE;
            int elem_idx = col % BLOCK_SIZE;

            const Q16_1Block &block = blocks[row * blocks_per_row + block_idx];
            return block.qs[elem_idx];
        }

        /**
         * @brief Get scale factor from Q16_1Block at specified row/column
         */
        inline float q16_block_scale(
            const Q16_1Block *blocks,
            int row,
            int col,
            int blocks_per_row)
        {
            constexpr int BLOCK_SIZE = Q16_1Block::BLOCK_SIZE;

            int block_idx = col / BLOCK_SIZE;
            const Q16_1Block &block = blocks[row * blocks_per_row + block_idx];
            return block.d;
        }

    } // namespace kernels::q16_1
} // namespace llaminar2
