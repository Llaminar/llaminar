/**
 * @file JitQ16DotProduct.h
 * @brief JIT Microkernel for Q16_1 dot product (Q×K^T)
 * @author David Sanftenberg
 * @date December 2025
 *
 * JIT-generated Q16_1 dot product using AVX-512 `vpmaddwd` (INT16×INT16→INT32).
 * Processes 4 KV positions per micro-iteration for optimal SIMD utilization.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * ALGORITHM: Pure Integer Q×K^T
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * For each KV position kv in [0, 4):
 *   For each block b in [0, blocks_per_head):
 *     scale_q = Q[b].d           // FP32 scale (tracked separately)
 *     scale_k = K[kv, b].d       // FP32 scale (tracked separately)
 *     raw_dot = Σ(Q.qs[i] × K.qs[i]) for i in [0, 31]  // vpmaddwd
 *     combined_scale_product += scale_q × scale_k
 *   score[kv] = raw_dot × combined_scale × attention_scale
 *
 * INSTRUCTION: vpmaddwd (Packed Multiply and Add Word to Doubleword)
 * ─────────────────────────────────────────────────────────────────────
 * For each pair of adjacent INT16 values:
 *   INT32 result[i] = src1[2i] × src2[2i] + src1[2i+1] × src2[2i+1]
 *
 * For 32 INT16 values (512 bits), produces 16 INT32 values.
 * We then horizontal-sum the 16 INT32 values to get the block dot product.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * REGISTER CONTRACT
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * INPUTS (read-only, persistent during decode):
 *   Input0-3 (zmm8-11): Q vector (4 blocks for head_dim=128)
 *
 * STREAMING (loaded per micro-iteration):
 *   Input4-7 (zmm12-15): K vectors for 4 KV positions
 *
 * OUTPUTS:
 *   Scratch0-3 (zmm20-23): INT32 scores for 4 KV positions
 *
 * SCRATCH:
 *   Scratch4-5 (zmm24-25): Intermediate horizontal sums
 *
 * CONSTANTS:
 *   Const128 (zmm26): 128.0f for Q16 dequant reference
 *   ConstScale (zmm27): attention_scale = 1/sqrt(head_dim)
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

namespace llaminar2::kernels::q16_1::jit
{

    // ============================================================================
    // JIT Code Emitter
    // ============================================================================

    /**
     * @brief JIT code emitter for Q16_1 dot product microkernel
     *
     * Emits AVX-512 code for computing Q×K^T using vpmaddwd.
     * Designed to be composed with other microkernels in the fused attention kernel.
     */
    class JitQ16DotProductEmitter
    {
    public:
        /**
         * @brief Configuration for the dot product emitter
         */
        struct Config
        {
            int head_dim = 128;           ///< Head dimension (must be multiple of 32)
            int kv_micro_tile = 4;        ///< KV positions per micro-iteration
            float attention_scale = 0.0f; ///< 1/sqrt(head_dim), computed if 0
            bool enable_prefetch = true;  ///< Emit prefetch instructions
            int prefetch_distance = 4;    ///< KV positions ahead for prefetch

            int blocks_per_head() const { return head_dim / 32; }
            float get_attention_scale() const
            {
                if (attention_scale > 0.0f)
                    return attention_scale;
                return 1.0f / std::sqrt(static_cast<float>(head_dim));
            }
        };

        explicit JitQ16DotProductEmitter(const Config &config) : config_(config) {}

        // ─────────────────────────────────────────────────────────────────────────
        // JIT Code Emission (TODO: Implement with Xbyak)
        // ─────────────────────────────────────────────────────────────────────────

        /**
         * @brief Emit JIT code for loading Q vector into Input0-3
         * @param gen Xbyak code generator
         * @param q_ptr Register holding pointer to Q blocks
         *
         * Loads blocks_per_head Q16_1 blocks into Input0-3 (up to 4 blocks).
         * For head_dim=128, loads 4 blocks × 72 bytes = 288 bytes.
         */
        template <typename CodeGen>
        void emit_load_q_vector(CodeGen &gen, const Xbyak::Reg64 &q_ptr)
        {
            // TODO: Implement with borrow<>() pattern
            // auto input0 = gen.borrow<Q16Input0>();
            // auto input1 = gen.borrow<Q16Input1>();
            // ...
            // Load Q blocks: each Q16_1Block is 72 bytes (4 + 4 + 64)
            // But we only load the qs[32] INT16 values (64 bytes) for VPMADDWD
            (void)gen;
            (void)q_ptr;
        }

        /**
         * @brief Emit JIT code for streaming K vectors into Input4-7
         * @param gen Xbyak code generator
         * @param k_base Register holding pointer to K blocks
         * @param kv_offset Immediate offset (in KV positions) from k_base
         * @param kv_stride Stride between KV positions (in bytes)
         *
         * Loads 4 KV positions worth of K data into Input4-7.
         */
        template <typename CodeGen>
        void emit_load_k_vectors(CodeGen &gen, const Xbyak::Reg64 &k_base,
                                 int kv_offset, const Xbyak::Reg64 &kv_stride)
        {
            (void)gen;
            (void)k_base;
            (void)kv_offset;
            (void)kv_stride;
        }

        /**
         * @brief Emit JIT code for computing Q×K^T dot products
         * @param gen Xbyak code generator
         *
         * Computes 4 dot products (Q dot K[kv] for kv in [0,4))
         * using vpmaddwd and horizontal sum.
         *
         * INPUT:  Input0-3 (Q), Input4-7 (K[kv:kv+4])
         * OUTPUT: Scratch0-3 (INT32 scores for 4 KV positions)
         */
        template <typename CodeGen>
        void emit_dot_product_4kv(CodeGen &gen)
        {
            // TODO: Implement the core vpmaddwd + horizontal sum logic
            //
            // For each KV position kv in [0, 4):
            //   zmm_acc = 0
            //   For each block b in [0, blocks_per_head):
            //     zmm_tmp = vpmaddwd(Input[b].zmm(), K[kv, b].zmm())
            //     zmm_acc = vpaddd(zmm_acc, zmm_tmp)
            //   Scratch[kv] = horizontal_sum(zmm_acc)
            //
            // The horizontal sum uses:
            //   vextracti64x4 (extract high 256 bits)
            //   vpaddd (add high and low halves)
            //   vextracti128 (extract high 128 bits)
            //   vpaddd
            //   vphaddd (horizontal add pairs)
            //   vphaddd
            //   Final scalar in low 32 bits
            (void)gen;
        }

        /**
         * @brief Emit JIT code for applying attention scale to scores
         * @param gen Xbyak code generator
         *
         * Multiplies INT32 scores by attention_scale (from ConstScale zmm27).
         * Note: This converts INT32 → FP32 for the scale multiply, but the
         * result is immediately consumed by softmax which outputs INT16.
         */
        template <typename CodeGen>
        void emit_apply_scale(CodeGen &gen)
        {
            (void)gen;
        }

        /**
         * @brief Emit prefetch instruction for next K block
         * @param gen Xbyak code generator
         * @param k_base Register holding pointer to K blocks
         * @param prefetch_offset Offset (in bytes) from k_base
         * @param hint Prefetch hint (0=T0, 1=T1, 2=T2)
         */
        template <typename CodeGen>
        void emit_prefetch_k(CodeGen &gen, const Xbyak::Reg64 &k_base,
                             int prefetch_offset, int hint = 0)
        {
            (void)gen;
            (void)k_base;
            (void)prefetch_offset;
            (void)hint;
        }

        // ─────────────────────────────────────────────────────────────────────────
        // Reference Implementation (for testing)
        // ─────────────────────────────────────────────────────────────────────────

        /**
         * @brief Compute scaled dot product (reference implementation)
         *
         * @param q_data Pointer to Q blocks
         * @param k_data Pointer to K blocks
         * @param num_blocks Number of 32-element blocks per head
         * @param attention_scale 1/sqrt(head_dim)
         * @return Scaled dot product as FP32
         */
        static float compute_reference(
            const Q16_1Block *q_data,
            const Q16_1Block *k_data,
            int num_blocks,
            float attention_scale)
        {
            float total = 0.0f;
            for (int b = 0; b < num_blocks; ++b)
            {
                int32_t block_sum = 0;
                for (int i = 0; i < 32; ++i)
                {
                    block_sum += static_cast<int32_t>(q_data[b].qs[i]) *
                                 static_cast<int32_t>(k_data[b].qs[i]);
                }
                // Scale by both Q and K block scales
                total += static_cast<float>(block_sum) * q_data[b].d * k_data[b].d;
            }
            return total * attention_scale;
        }

        /**
         * @brief Compute raw INT32 dot product (no scale factors)
         *
         * Useful for testing vpmaddwd logic in isolation.
         */
        static int32_t compute_raw_int32(
            const Q16_1Block *q_data,
            const Q16_1Block *k_data,
            int num_blocks)
        {
            int32_t total = 0;
            for (int b = 0; b < num_blocks; ++b)
            {
                for (int i = 0; i < 32; ++i)
                {
                    total += static_cast<int32_t>(q_data[b].qs[i]) *
                             static_cast<int32_t>(k_data[b].qs[i]);
                }
            }
            return total;
        }

        /**
         * @brief Compute dot products for 4 KV positions (reference)
         *
         * @param q_data Pointer to Q blocks
         * @param k_data Pointer to K blocks (4 KV positions)
         * @param k_stride Stride between KV positions (in blocks)
         * @param num_blocks Number of blocks per head
         * @param attention_scale 1/sqrt(head_dim)
         * @param scores_out Output array for 4 INT32 scores
         */
        static void compute_4kv_reference(
            const Q16_1Block *q_data,
            const Q16_1Block *k_data,
            int k_stride,
            int num_blocks,
            float attention_scale,
            int32_t *scores_out)
        {
            for (int kv = 0; kv < 4; ++kv)
            {
                const Q16_1Block *k_row = k_data + kv * k_stride;
                float scaled = compute_reference(q_data, k_row, num_blocks, attention_scale);
                // Convert to INT32 for softmax input
                scores_out[kv] = static_cast<int32_t>(scaled);
            }
        }

    private:
        Config config_;
    };

    // ============================================================================
    // Horizontal Sum Helper (for reference and future JIT use)
    // ============================================================================

    /**
     * @brief Compute horizontal sum of 16 INT32 values in a ZMM register
     *
     * This is the algorithm we'll emit in JIT:
     *   1. vextracti64x4: split 512→256 (high half)
     *   2. vpaddd: add high and low 256-bit halves
     *   3. vextracti128: split 256→128 (high half)
     *   4. vpaddd: add high and low 128-bit halves (4 INT32)
     *   5. vphaddd: horizontal add pairs → 2 INT32
     *   6. vphaddd: horizontal add pairs → 1 INT32
     *   7. vmovd: extract final scalar
     */
    inline int32_t horizontal_sum_int32x16(const int32_t *values)
    {
        int32_t sum = 0;
        for (int i = 0; i < 16; ++i)
        {
            sum += values[i];
        }
        return sum;
    }

} // namespace llaminar2::kernels::q16_1::jit
