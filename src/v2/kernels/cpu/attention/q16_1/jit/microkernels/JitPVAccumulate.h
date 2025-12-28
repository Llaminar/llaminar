/**
 * @file JitPVAccumulate.h
 * @brief JIT Microkernel for P×V weighted accumulation
 * @author David Sanftenberg
 * @date December 2025
 *
 * JIT-generated P×V weighted accumulation for attention output.
 * Accumulates weighted V vectors into INT64 context accumulators.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * ALGORITHM: Weighted V Accumulation
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * For each dimension d in [0, head_dim):
 *   O_acc[d] += Σ_kv (weight[kv] × V[kv, d])
 *
 * Where:
 *   - weight[kv]: INT16 attention weight from softmax [0, 32767]
 *   - V[kv, d]:   INT16 quantized value from Q16_1 block
 *   - O_acc[d]:   INT64 accumulator (prevents overflow for long sequences)
 *
 * The accumulation is: INT16 weight × INT16 V_qs → INT32 → accumulate to INT64
 *
 * After all KV positions processed, O_acc contains the weighted sum.
 * Normalization (÷ weight_sum × scale) happens in finalization phase.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * REGISTER CONTRACT
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * INPUTS:
 *   Scratch0-3 (zmm20-23): INT16 weights for 4 KV positions
 *                          (only low 16 bits of each 32-bit lane used)
 *
 * STREAMING:
 *   Input4-7 (zmm12-15): V vectors for 4 KV positions (block-by-block)
 *
 * OUTPUTS (accumulate):
 *   Accum0-7 (zmm0-7): INT64 accumulators for head_dim
 *                      8 regs × 8 INT64 = 64 elements per pass
 *                      For head_dim=128: process in 2 passes
 *
 * SCRATCH:
 *   Scratch4-5 (zmm24-25): Broadcast weight, intermediate products
 *   Input0-3 (zmm8-11):    Widened V values (INT16 → INT32)
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * VECTORIZATION STRATEGY
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * For each KV position:
 *   1. Broadcast weight[kv] to all 16 lanes of zmm (vpbroadcastd)
 *   2. Load V[kv, block] into Input4-7 (32 INT16 values)
 *   3. Widen V to INT32 (vpmovzxwd or vpmovsxwd)
 *   4. Multiply: weight_broadcast × V_widened → INT32 products
 *   5. Widen products to INT64 (vpmovsxdq)
 *   6. Accumulate: O_acc += products (vpaddq)
 *
 * This processes 8 INT64 values per ZMM register.
 * For 32 V values (one Q16_1 block), we need 4 passes through 8 values each.
 *
 * @see Q16RegisterAllocation.h for register zone definitions
 */

#pragma once

#include "../Q16RegisterAllocation.h"
#include "tensors/BlockStructures.h"
#include "../../../../jit/RegisterGuard.h"

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>

namespace llaminar2::kernels::q16_1::jit
{

    // ============================================================================
    // JIT Code Emitter
    // ============================================================================

    /**
     * @brief JIT code emitter for P×V accumulation microkernel
     *
     * Emits AVX-512 code for weighted V vector accumulation.
     */
    class JitPVAccumulateEmitter
    {
    public:
        /**
         * @brief Configuration for the P×V emitter
         */
        struct Config
        {
            int head_dim = 128;    ///< Head dimension (must be multiple of 32)
            int kv_micro_tile = 4; ///< KV positions per micro-iteration
            bool enable_prefetch = true;
            int prefetch_distance = 4;

            int blocks_per_head() const { return head_dim / 32; }
        };

        explicit JitPVAccumulateEmitter(const Config &config) : config_(config) {}

        // ─────────────────────────────────────────────────────────────────────────
        // JIT Code Emission (TODO: Implement with Xbyak)
        // ─────────────────────────────────────────────────────────────────────────

        /**
         * @brief Emit code to clear INT64 accumulators
         * @param gen Xbyak code generator
         *
         * Clears Accum0-7 (zmm0-7) to zero.
         */
        template <typename CodeGen>
        void emit_clear_accumulators(CodeGen &gen)
        {
            (void)gen;
            // TODO: vpxorq for all accumulator registers
            // gen.vpxorq(zmm0, zmm0, zmm0);
            // gen.vpxorq(zmm1, zmm1, zmm1);
            // ... etc
        }

        /**
         * @brief Emit code to load V vectors for 4 KV positions
         * @param gen Xbyak code generator
         * @param v_base Register holding pointer to V blocks
         * @param kv_offset KV position offset
         * @param v_stride Stride between KV positions (in bytes)
         * @param block_idx Which block within head (0 to blocks_per_head-1)
         */
        template <typename CodeGen>
        void emit_load_v_vectors(CodeGen &gen, const Xbyak::Reg64 &v_base,
                                 int kv_offset, const Xbyak::Reg64 &v_stride,
                                 int block_idx)
        {
            (void)gen;
            (void)v_base;
            (void)kv_offset;
            (void)v_stride;
            (void)block_idx;
            // TODO: Load 4 V blocks into Input4-7
        }

        /**
         * @brief Emit weighted accumulation for one KV position
         * @param gen Xbyak code generator
         * @param weight_reg Scratch register containing broadcast weight
         * @param v_reg Input register containing V vector (32 INT16)
         * @param accum_base_idx Starting accumulator index (0 or 4)
         *
         * Performs: Accum[accum_base_idx:accum_base_idx+4] += weight × V
         *
         * Processing 32 INT16 V values into 32 INT64 accumulators requires:
         *   - 4 passes of 8 INT64 values each
         *   - Each pass: widen 8 INT16 → 8 INT32 → 8 INT64, multiply, accumulate
         */
        template <typename CodeGen>
        void emit_weighted_accum_block(CodeGen &gen,
                                       const Xbyak::Zmm &weight_reg,
                                       const Xbyak::Zmm &v_reg,
                                       int accum_base_idx)
        {
            (void)gen;
            (void)weight_reg;
            (void)v_reg;
            (void)accum_base_idx;
            // TODO: Implement INT16 × INT16 → INT64 accumulation
            //
            // For each 8-element chunk of V:
            //   1. Extract 8 INT16 values (vextracti128 + vpmovzxwd)
            //   2. Multiply by broadcast weight (vpmulld)
            //   3. Widen to INT64 (vpmovsxdq)
            //   4. Accumulate (vpaddq)
        }

        /**
         * @brief Emit full P×V for 4 KV positions, one block
         * @param gen Xbyak code generator
         *
         * INPUT:  Scratch0-3 (INT16 weights), Input4-7 (V blocks for 4 KV positions)
         * OUTPUT: Accum0-7 (INT64 accumulators, accumulated)
         */
        template <typename CodeGen>
        void emit_pv_4kv_block(CodeGen &gen)
        {
            (void)gen;
            // TODO: For each of 4 KV positions:
            //   1. Broadcast weight from Scratch[kv] to Scratch4
            //   2. Load V block from Input[4+kv]
            //   3. Call emit_weighted_accum_block for each 8-element chunk
        }

        /**
         * @brief Emit prefetch for next V blocks
         */
        template <typename CodeGen>
        void emit_prefetch_v(CodeGen &gen, const Xbyak::Reg64 &v_base,
                             int prefetch_offset, int hint = 0)
        {
            (void)gen;
            (void)v_base;
            (void)prefetch_offset;
            (void)hint;
        }

        // ─────────────────────────────────────────────────────────────────────────
        // Reference Implementation (for testing)
        // ─────────────────────────────────────────────────────────────────────────

        /**
         * @brief Accumulate weighted V vectors (reference)
         *
         * @param weights INT16 attention weights [kv_count]
         * @param v_blocks V blocks [kv_count × blocks_per_head]
         * @param blocks_per_head Number of Q16_1 blocks per head
         * @param kv_count Number of KV positions
         * @param output INT64 accumulators [head_dim] (accumulated, not zeroed)
         */
        static void compute_reference(
            const int16_t *weights,
            const Q16_1Block *v_blocks,
            int blocks_per_head,
            int kv_count,
            int64_t *output)
        {
            const int head_dim = blocks_per_head * 32;

            for (int kv = 0; kv < kv_count; ++kv)
            {
                int64_t w = static_cast<int64_t>(weights[kv]);

                for (int b = 0; b < blocks_per_head; ++b)
                {
                    const Q16_1Block &block = v_blocks[kv * blocks_per_head + b];

                    for (int i = 0; i < 32; ++i)
                    {
                        int d_idx = b * 32 + i;
                        // INT16 weight × INT16 V_qs → INT32 → accumulate to INT64
                        output[d_idx] += w * static_cast<int64_t>(block.qs[i]);
                    }
                }
            }
        }

        /**
         * @brief Accumulate weighted V vectors for 4 KV positions (reference)
         *
         * @param weights INT16 attention weights [4]
         * @param v_blocks V blocks [4 × blocks_per_head]
         * @param blocks_per_head Number of Q16_1 blocks per head
         * @param output INT64 accumulators [head_dim] (accumulated)
         */
        static void compute_4kv_reference(
            const int16_t *weights,
            const Q16_1Block *v_blocks,
            int blocks_per_head,
            int64_t *output)
        {
            compute_reference(weights, v_blocks, blocks_per_head, 4, output);
        }

        /**
         * @brief Normalize INT64 accumulators to INT32 context
         *
         * After all KV positions processed:
         *   context[d] = round(O_acc[d] / weight_sum × some_scale)
         *
         * @param accumulators INT64 accumulators [head_dim]
         * @param weight_sum Sum of all weights
         * @param avg_v_scale Average V block scale (for dequantization)
         * @param output INT32 context [head_dim]
         * @param head_dim Head dimension
         */
        static void normalize_to_int32(
            const int64_t *accumulators,
            int64_t weight_sum,
            float avg_v_scale,
            int32_t *output,
            int head_dim)
        {
            if (weight_sum == 0)
            {
                std::fill(output, output + head_dim, 0);
                return;
            }

            // The accumulator contains: Σ(weight × V_qs)
            // Weight is in [0, 32767], V_qs is INT16
            // We need to normalize by weight_sum and apply V scale

            // Scale factor: avg_v_scale / weight_sum × scale_to_int32_range
            // For INT32 output, we target a reasonable range
            float norm = avg_v_scale / static_cast<float>(weight_sum);

            for (int d = 0; d < head_dim; ++d)
            {
                float val = static_cast<float>(accumulators[d]) * norm;
                // Clamp to INT32 range
                val = std::clamp(val, static_cast<float>(INT32_MIN), static_cast<float>(INT32_MAX));
                output[d] = static_cast<int32_t>(std::round(val));
            }
        }

        /**
         * @brief Compute average V scale for a set of blocks
         */
        static float compute_avg_v_scale(
            const Q16_1Block *v_blocks,
            int blocks_per_head,
            int kv_count)
        {
            double sum = 0.0;
            int count = blocks_per_head * kv_count;
            for (int i = 0; i < count; ++i)
            {
                sum += v_blocks[i].d;
            }
            return count > 0 ? static_cast<float>(sum / count) : 1.0f;
        }

    private:
        Config config_;
    };

    // ============================================================================
    // Accumulator Management
    // ============================================================================

    /**
     * @brief INT64 accumulator buffer for attention context
     *
     * Manages the INT64 accumulators used during P×V accumulation.
     * Provides clear, accumulate, and normalize operations.
     */
    class Int64AccumulatorBuffer
    {
    public:
        explicit Int64AccumulatorBuffer(int head_dim)
            : head_dim_(head_dim), data_(head_dim, 0)
        {
        }

        void clear()
        {
            std::fill(data_.begin(), data_.end(), 0);
        }

        int64_t *data() { return data_.data(); }
        const int64_t *data() const { return data_.data(); }

        int head_dim() const { return head_dim_; }

        /**
         * @brief Normalize accumulators to INT32 context
         */
        void normalize_to_int32(int64_t weight_sum, float avg_v_scale, int32_t *output) const
        {
            JitPVAccumulateEmitter::normalize_to_int32(
                data_.data(), weight_sum, avg_v_scale, output, head_dim_);
        }

    private:
        int head_dim_;
        std::vector<int64_t> data_;
    };

} // namespace llaminar2::kernels::q16_1::jit
