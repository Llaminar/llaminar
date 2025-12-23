/**
 * @file JitQ8DotProduct.h
 * @brief JIT Microkernel μK1: Q8_1 dot product with proper scaling
 * @author David Sanftenberg
 * @date December 2025
 *
 * JIT-generated Q8_1 dot product using AVX-512 VNNI (vpdpbusd).
 * Supports both single dot product (FA1) and 4x batched (FA2) modes.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * ALGORITHM
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * For each block b in [0, num_blocks):
 *   d_q = fp16_to_fp32(Q[b].d)
 *   d_k = fp16_to_fp32(K[b].d)
 *   sum_qs_k = K[b].sum_qs
 *
 *   raw_dot = vpdpbusd(Q_unsigned, K_signed)  // (Q+128) * K
 *   corrected_dot = raw_dot - 128 * sum_qs_k   // Undo unsigned bias
 *   scaled_dot = corrected_dot * d_q * d_k
 *
 *   score += scaled_dot
 *
 * return score * global_scale
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * REGISTER USAGE (FA2 4x Mode)
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Output scores (ScoreZone - xmm20-23):
 *   - Score0 (xmm20): Q·K[kv+0]
 *   - Score1 (xmm21): Q·K[kv+1]
 *   - Score2 (xmm22): Q·K[kv+2]
 *   - Score3 (xmm23): Q·K[kv+3]
 *
 * vpdpbusd accumulators (borrowed via RegisterGuard):
 *   - Accum4-7 (ymm4-7): Temporary accumulators for 4 parallel dot products
 *
 * K data loading:
 *   - Input0 (zmm8): K data loading and horizontal reduction temp
 *   - Input1 (zmm9): K scales (d_K, sum_qs_K)
 *
 * Correction term:
 *   - Scratch5 (zmm25): 128.0f * sum_qs_K correction
 *
 * Q data (caller-owned, pre-loaded):
 *   - zmm10-13: Unsigned Q data (persists across KV loop)
 *   - d_Q loaded on-demand from memory to save registers
 */

#pragma once

#include "JitMicrokernelBase.h"
#include "../../../jit/RegisterAllocation.h"
#include "../../../jit/RegisterGuard.h"
#include "../../../jit/RegisterEnforcement.h"
#include "../microkernels/Q8DotProduct.h" // For Q8_1Block struct

// Import typed register aliases and C++20 concepts
using namespace llaminar2::jit;

namespace llaminar::v2::kernels::jit
{
    // ============================================================================
    // Q8DotProduct Register Manifest (compile-time documentation)
    // ============================================================================
    // Documents the exact registers borrowed by emit_q8_dot_product().
    // Uses Accum4-7 for vpdpbusd, Input0-1 for K data/scales, Scratch5 for correction.

    using Q8DotProductManifest = KernelRegisterManifest<
        Accum4, Accum5, Accum6, Accum7, // vpdpbusd accumulators
        Input0, Input1,                 // K data and scales
        Scratch5                        // Q8_1 correction term
        >;
    static_assert(!Q8DotProductManifest::has_conflicts(),
                  "Q8DotProduct register manifest has conflicts!");

    /**
     * @brief JIT code emitter for Q8_1 dot product
     *
     * Emits inline code for computing dot(Q, K) * scale.
     * Designed to be embedded within larger JIT kernels (attention).
     */
    class JitQ8DotProductEmitter
    {
    public:
        /**
         * @brief Emit Q8_1 dot product: score = dot(Q, K) * global_scale
         *
         * Prerequisites:
         * - zmm_128() initialized to 0x80808080 for unsigned conversion
         * - Q blocks loaded to stack at q_stack_offset
         *
         * Clobbers:
         * - ymm4, ymm5, ymm6 (data loading and vpdpbusd)
         * - xmm6, xmm7 (horizontal sum)
         * - xmm8, xmm9 (scales d_Q, d_K)
         * - xmm14, xmm15 (correction, sum_qs)
         * - rax (constant loading)
         *
         * @param gen Code generator
         * @param dst_xmm XMM register to receive scalar score result
         * @param reg_K_ptr GP register pointing to K blocks
         * @param reg_rsp RSP register for stack access
         * @param q_stack_base_offset Stack offset where Q blocks are stored
         * @param num_blocks Number of Q8_1 blocks (head_dim / 32)
         * @param scale_xmm XMM containing global scale (element 0)
         */
        void emit_dot_product(
            JitMicrokernelBase &gen,
            const Xbyak::Xmm &dst_xmm,
            const Xbyak::Reg64 &reg_K_ptr,
            const Xbyak::Reg64 &reg_rsp,
            int q_stack_base_offset,
            int num_blocks,
            const Xbyak::Xmm &scale_xmm)
        {
            using namespace Xbyak;

            gen.debug_emit("emit_dot_product (" + std::to_string(num_blocks) + " blocks)");

            // Zero accumulator
            gen.vxorps(dst_xmm, dst_xmm, dst_xmm);

            // YMM registers for Q8_1 dot product (explicit construction)
            // Use registers that avoid:
            // - zmm0-7 (Accumulators in fused kernel)
            // - zmm10-13 (Q data in fused kernel)
            // - zmm16-19 (Softmax state)
            // - zmm21-22 (Passed as args: dst_xmm, scale_xmm)
            // - zmm26-31 (Constants)
            Ymm ymm_q(14);   // Safe input zone
            Ymm ymm_k(15);   // Safe input zone
            Ymm ymm_dot(20); // Scratch zone

            // XMM registers for scales and correction (explicit construction)
            Xmm xmm_d_q(8);           // Safe input zone
            Xmm xmm_d_k(9);           // Safe input zone
            Xmm xmm_sum_qs_k(23);     // Scratch zone
            Xmm xmm_correction(24);   // Scratch zone
            Xmm xmm_block_result(20); // Alias of ymm_dot
            Xmm xmm_tmp(25);          // Scratch zone

            for (int b = 0; b < num_blocks; ++b)
            {
                gen.debug_emit("  Block " + std::to_string(b));

                // Stack layout for Q block: [d (2B)][sum_qs (2B)][qs (32B)] = 36B
                // But we store padded to 64B on stack for alignment
                int q_offset = q_stack_base_offset + b * 64;
                int k_offset = b * 36; // K is at original Q8_1Block layout (36 bytes)

                // Load Q scale: d_Q (FP16 at offset 0)
                gen.vpbroadcastw(xmm_d_q, gen.ptr[reg_rsp + q_offset]);
                gen.vcvtph2ps(xmm_d_q, xmm_d_q);

                // Load Q data (32 int8 values at offset 4 in our padded layout)
                gen.vmovdqu8(ymm_q, gen.ptr[reg_rsp + q_offset + 4]);

                // Convert Q from signed to unsigned by XOR with 0x80
                // vpdpbusd: src1 (Q) must be unsigned, src2 (K) is signed
                gen.vpxord(ymm_q, ymm_q, gen.const_128().ymm());

                // Load K scale: d_K (FP16)
                gen.vpbroadcastw(xmm_d_k, gen.ptr[reg_K_ptr + k_offset]);
                gen.vcvtph2ps(xmm_d_k, xmm_d_k);

                // Load K sum_qs (INT16 at offset 2)
                gen.vpbroadcastw(xmm_sum_qs_k, gen.ptr[reg_K_ptr + k_offset + 2]);
                gen.vpmovsxwd(xmm_sum_qs_k, xmm_sum_qs_k); // Sign-extend to int32
                gen.vcvtdq2ps(xmm_sum_qs_k, xmm_sum_qs_k); // Convert to float

                // Load K data (32 int8 values at offset 4)
                // K remains SIGNED - do NOT XOR with 0x80
                gen.vmovdqu8(ymm_k, gen.ptr[reg_K_ptr + k_offset + 4]);

                // vpdpbusd: unsigned(Q+128) × signed(K)
                // Result = sum((Q+128) * K) = sum(Q*K) + 128*sum(K)
                gen.vxorps(ymm_dot, ymm_dot, ymm_dot);
                gen.vpdpbusd(ymm_dot, ymm_q, ymm_k);

                // Horizontal sum of dot product (8×int32 → scalar int32)
                gen.vextracti32x4(xmm_tmp, ymm_dot, 1);
                gen.vpaddd(xmm_block_result, Xmm(ymm_dot.getIdx()), xmm_tmp);
                gen.vpshufd(xmm_tmp, xmm_block_result, 0x4E);
                gen.vpaddd(xmm_block_result, xmm_block_result, xmm_tmp);
                gen.vpshufd(xmm_tmp, xmm_block_result, 0xB1);
                gen.vpaddd(xmm_block_result, xmm_block_result, xmm_tmp);

                // Convert to float
                gen.vcvtdq2ps(xmm_block_result, xmm_block_result);

                // Compute correction: 128.0f * sum_qs_K
                gen.mov(gen.eax, 0x43000000); // 128.0f in IEEE 754
                gen.vmovd(xmm_correction, gen.eax);
                gen.vbroadcastss(xmm_correction, xmm_correction);
                gen.vmulps(xmm_correction, xmm_correction, xmm_sum_qs_k);

                // Subtract correction: raw_dot - 128*sum_qs_K
                gen.vsubps(xmm_block_result, xmm_block_result, xmm_correction);

                // Compute combined scale: d_Q * d_K
                gen.vmulps(xmm_d_q, xmm_d_q, xmm_d_k);

                // Apply block scale
                gen.vmulps(xmm_block_result, xmm_block_result, xmm_d_q);

                // Accumulate
                gen.vaddps(dst_xmm, dst_xmm, xmm_block_result);
            }

            // Apply global scale (attention scale = 1/sqrt(d))
            gen.vmulss(dst_xmm, dst_xmm, scale_xmm);
        }

        /**
         * @brief Emit Q8_1 dot product using pre-loaded Q registers and d_Q from memory
         *
         * Optimized for Decode mode where Q data is in registers, but d_Q is loaded from memory
         * to save registers.
         *
         * @param gen Code generator
         * @param dst_xmm XMM register to receive scalar score result
         * @param reg_K_ptr GP register pointing to K blocks
         * @param reg_Q_ptr GP register pointing to Q blocks (for d_Q)
         * @param num_blocks Number of Q8_1 blocks
         * @param scale_xmm XMM containing global scale
         * @param q_reg_base_idx Base index of ZMM registers holding Q data (unsigned)
         */
        void emit_dot_product_register_q_mem_dq(
            JitMicrokernelBase &gen,
            const Xbyak::Xmm &dst_xmm,
            const Xbyak::Reg64 &reg_K_ptr,
            const Xbyak::Reg64 &reg_Q_ptr,
            int q_head_offset,
            int num_blocks,
            const Xbyak::Xmm &scale_xmm,
            int q_reg_base_idx)
        {
            using namespace Xbyak;

            // Zero accumulator
            gen.vxorps(dst_xmm, dst_xmm, dst_xmm);

            // Registers for K and computation
            // Use registers that avoid:
            // - zmm0-7 (Accumulators in fused kernel)
            // - zmm10-13 (Q data in fused kernel)
            // - zmm16-19 (Softmax state)
            // - zmm21-22 (Passed as args: dst_xmm, scale_xmm)
            // - zmm26-31 (Constants)
            Ymm ymm_k(15);   // Safe input zone
            Ymm ymm_dot(20); // Scratch zone

            // XMM registers for scales and correction
            Xmm xmm_d_q(8); // Scratch for d_Q
            Xmm xmm_d_k(9);
            Xmm xmm_sum_qs_k(23);     // Scratch zone
            Xmm xmm_correction(24);   // Scratch zone
            Xmm xmm_block_result(20); // Alias of ymm_dot
            Xmm xmm_tmp(25);          // Scratch zone

            for (int b = 0; b < num_blocks; ++b)
            {
                // Use pre-loaded Q registers
                Ymm ymm_q(q_reg_base_idx + b);

                int k_offset = b * 36;
                int q_offset = q_head_offset + b * 36;

                // Load Q scale: d_Q (FP16) from memory
                gen.vpbroadcastw(xmm_d_q, gen.ptr[reg_Q_ptr + q_offset]);
                gen.vcvtph2ps(xmm_d_q, xmm_d_q);

                // Load K scale: d_K (FP16)
                gen.vpbroadcastw(xmm_d_k, gen.ptr[reg_K_ptr + k_offset]);
                gen.vcvtph2ps(xmm_d_k, xmm_d_k);

                // Load K sum_qs (INT16 at offset 2)
                gen.vpbroadcastw(xmm_sum_qs_k, gen.ptr[reg_K_ptr + k_offset + 2]);
                gen.vpmovsxwd(xmm_sum_qs_k, xmm_sum_qs_k); // Sign-extend to int32
                gen.vcvtdq2ps(xmm_sum_qs_k, xmm_sum_qs_k); // Convert to float

                // Load K data (32 int8 values at offset 4)
                gen.vmovdqu8(ymm_k, gen.ptr[reg_K_ptr + k_offset + 4]);

                // vpdpbusd: unsigned(Q+128) × signed(K)
                gen.vxorps(ymm_dot, ymm_dot, ymm_dot);
                gen.vpdpbusd(ymm_dot, ymm_q, ymm_k);

                // Horizontal sum of dot product
                gen.vextracti32x4(xmm_tmp, ymm_dot, 1);
                gen.vpaddd(xmm_block_result, Xmm(ymm_dot.getIdx()), xmm_tmp);
                gen.vpshufd(xmm_tmp, xmm_block_result, 0x4E);
                gen.vpaddd(xmm_block_result, xmm_block_result, xmm_tmp);
                gen.vpshufd(xmm_tmp, xmm_block_result, 0xB1);
                gen.vpaddd(xmm_block_result, xmm_block_result, xmm_tmp);

                // Convert to float
                gen.vcvtdq2ps(xmm_block_result, xmm_block_result);

                // Compute correction: 128.0f * sum_qs_K
                gen.mov(gen.eax, 0x43000000); // 128.0f
                gen.vmovd(xmm_correction, gen.eax);
                gen.vbroadcastss(xmm_correction, xmm_correction);
                gen.vmulps(xmm_correction, xmm_correction, xmm_sum_qs_k);

                // Subtract correction: raw_dot - 128*sum_qs_K
                gen.vsubps(xmm_block_result, xmm_block_result, xmm_correction);

                // Compute combined scale: d_Q * d_K
                gen.vmulps(xmm_d_q, xmm_d_q, xmm_d_k);

                // Apply block scale
                gen.vmulps(xmm_block_result, xmm_block_result, xmm_d_q);

                // Accumulate
                gen.vaddps(dst_xmm, dst_xmm, xmm_block_result);
            }

            // Apply global scale (attention scale = 1/sqrt(d))
            gen.vmulss(dst_xmm, dst_xmm, scale_xmm);
        }

        /**
         * @brief Emit Q8_1 dot product using pre-loaded Q registers
         *
         * Optimized for Decode mode where Q is constant across KV loop.
         * Q data and d_Q scales must be pre-loaded into registers.
         *
         * @param gen Code generator
         * @param dst_xmm XMM register to receive scalar score result
         * @param reg_K_ptr GP register pointing to K blocks
         * @param num_blocks Number of Q8_1 blocks
         * @param scale_xmm XMM containing global scale
         * @param q_reg_base_idx Base index of ZMM registers holding Q data (unsigned)
         * @param dq_reg_base_idx Base index of XMM registers holding d_Q scales
         */
        void emit_dot_product_register_q(
            JitMicrokernelBase &gen,
            const Xbyak::Xmm &dst_xmm,
            const Xbyak::Reg64 &reg_K_ptr,
            int num_blocks,
            const Xbyak::Xmm &scale_xmm,
            int q_reg_base_idx,
            const Xbyak::Reg64 &reg_Q_ptr,
            int q_head_offset)
        {
            using namespace Xbyak;

            // Zero accumulator
            gen.vxorps(dst_xmm, dst_xmm, dst_xmm);

            // Registers for K and computation
            // Use registers that avoid:
            // - zmm0-7 (Accumulators in fused kernel)
            // - zmm10-13 (Q data in fused kernel)
            // - zmm16-19 (Softmax state)
            // - zmm21-22 (Passed as args: dst_xmm, scale_xmm)
            // - zmm26-31 (Constants)
            Ymm ymm_k(15);   // Safe input zone (Input7)
            Ymm ymm_dot(20); // Scratch zone

            // XMM registers for scales and correction
            Xmm xmm_d_q(8);           // Input0 - used for d_Q
            Xmm xmm_d_k(9);           // Input1 - used for d_K
            Xmm xmm_sum_qs_k(23);     // Scratch zone
            Xmm xmm_correction(24);   // Scratch zone
            Xmm xmm_block_result(20); // Alias of ymm_dot
            Xmm xmm_tmp(25);          // Scratch zone

            for (int b = 0; b < num_blocks; ++b)
            {
                // Use pre-loaded Q registers
                Ymm ymm_q(q_reg_base_idx + b);

                int k_offset = b * 36;
                int q_offset = q_head_offset + b * 36;

                // Load Q scale: d_Q (FP16) from memory
                gen.vpbroadcastw(xmm_d_q, gen.ptr[reg_Q_ptr + q_offset]);
                gen.vcvtph2ps(xmm_d_q, xmm_d_q);

                // Load K scale: d_K (FP16)
                gen.vpbroadcastw(xmm_d_k, gen.ptr[reg_K_ptr + k_offset]);
                gen.vcvtph2ps(xmm_d_k, xmm_d_k);

                // Load K sum_qs (INT16 at offset 2)
                gen.vpbroadcastw(xmm_sum_qs_k, gen.ptr[reg_K_ptr + k_offset + 2]);
                gen.vpmovsxwd(xmm_sum_qs_k, xmm_sum_qs_k); // Sign-extend to int32
                gen.vcvtdq2ps(xmm_sum_qs_k, xmm_sum_qs_k); // Convert to float

                // Load K data (32 int8 values at offset 4)
                gen.vmovdqu8(ymm_k, gen.ptr[reg_K_ptr + k_offset + 4]);

                // vpdpbusd: unsigned(Q+128) × signed(K)
                gen.vxorps(ymm_dot, ymm_dot, ymm_dot);
                gen.vpdpbusd(ymm_dot, ymm_q, ymm_k);

                // Horizontal sum of dot product
                gen.vextracti32x4(xmm_tmp, ymm_dot, 1);
                gen.vpaddd(xmm_block_result, Xmm(ymm_dot.getIdx()), xmm_tmp);
                gen.vpshufd(xmm_tmp, xmm_block_result, 0x4E);
                gen.vpaddd(xmm_block_result, xmm_block_result, xmm_tmp);
                gen.vpshufd(xmm_tmp, xmm_block_result, 0xB1);
                gen.vpaddd(xmm_block_result, xmm_block_result, xmm_tmp);

                // Convert to float
                gen.vcvtdq2ps(xmm_block_result, xmm_block_result);

                // Compute correction: 128.0f * sum_qs_K
                gen.mov(gen.eax, 0x43000000); // 128.0f
                gen.vmovd(xmm_correction, gen.eax);
                gen.vbroadcastss(xmm_correction, xmm_correction);
                gen.vmulps(xmm_correction, xmm_correction, xmm_sum_qs_k);

                // Subtract correction: raw_dot - 128*sum_qs_K
                gen.vsubps(xmm_block_result, xmm_block_result, xmm_correction);

                // Compute combined scale: d_Q * d_K
                gen.vmulps(xmm_d_q, xmm_d_q, xmm_d_k);

                // Apply block scale
                gen.vmulps(xmm_block_result, xmm_block_result, xmm_d_q);

                // Accumulate
                gen.vaddps(dst_xmm, dst_xmm, xmm_block_result);
            }

            // Apply global scale
            gen.vmulss(dst_xmm, dst_xmm, scale_xmm);
        }

        /**
         * @brief Emit 4x vectorized Q8_1 dot products for FA2 tile processing
         *
         * Computes 4 Q·K dot products simultaneously for better ILP:
         *   scores[0..3] = Q · K[kv+0..3] * scale
         *
         * FA2 Optimization: Instead of computing scores one-at-a-time and
         * immediately updating softmax state, we batch 4 scores together.
         * This allows:
         *   - Better instruction-level parallelism (4 independent dot products)
         *   - Amortized Q register loading (Q stays in registers)
         *   - Reduced branch mispredictions (batched softmax update)
         *
         * Register usage:
         *   - Q data: reused from q_reg_base_idx (pre-loaded unsigned Q)
         *   - K data: loaded for each of 4 KV positions
         *   - Output: 4 scalar scores in dst_xmm0..dst_xmm3
         *
         * @param gen Code generator
         * @param dst_xmm0..3 Output XMM registers for 4 scores (scalar in element 0)
         * @param reg_K_ptr Pointer to first K row
         * @param k_stride Bytes between consecutive K rows (typically num_kv_heads * num_blocks * 36)
         * @param num_blocks Number of Q8_1 blocks per head
         * @param scale_xmm Global scale (1/sqrt(head_dim))
         * @param q_reg_base_idx Base ZMM index for Q data (unsigned, persisted)
         * @param dq_reg_base_idx Base XMM index for d_Q scales
         */
        void emit_dot_product_4x(
            JitMicrokernelBase &gen,
            const Xbyak::Xmm &dst_xmm0,
            const Xbyak::Xmm &dst_xmm1,
            const Xbyak::Xmm &dst_xmm2,
            const Xbyak::Xmm &dst_xmm3,
            const Xbyak::Reg64 &reg_K_ptr,
            int64_t k_stride,
            int num_blocks,
            const Xbyak::Xmm &scale_xmm,
            int q_reg_base_idx,
            const Xbyak::Reg64 &reg_Q_ptr,
            int q_head_offset)
        {
            using namespace Xbyak;
            using namespace llaminar2::jit;

            gen.debug_emit("emit_dot_product_4x (" + std::to_string(num_blocks) + " blocks, stride=" + std::to_string(k_stride) + ")");

            // ═══════════════════════════════════════════════════════════════════
            // REGISTER GUARDS: Borrow ALL registers used by this emitter
            // ═══════════════════════════════════════════════════════════════════
            // This ensures we catch conflicts with caller's registers at runtime.
            // If a register is already borrowed by the caller, we'll get an assert.

            // vpdpbusd accumulators - Accum4-7 (zmm4-7)
            auto guard_acc4 = gen.borrow<Accum4>();
            auto guard_acc5 = gen.borrow<Accum5>();
            auto guard_acc6 = gen.borrow<Accum6>();
            auto guard_acc7 = gen.borrow<Accum7>();

            Ymm ymm_acc0 = guard_acc4.ymm();
            Ymm ymm_acc1 = guard_acc5.ymm();
            Ymm ymm_acc2 = guard_acc6.ymm();
            Ymm ymm_acc3 = guard_acc7.ymm();

            // K data loading - use Input0 (zmm8)
            // NOTE: This overlaps with xmm_d_k below, which is intentional!
            //       We load K data into ymm8, process it, then reuse xmm8 for d_k.
            auto guard_k_data = gen.borrow<Input0>();
            Ymm ymm_k = guard_k_data.ymm();

            // Temporaries for horizontal reduction - reuse Input0 (zmm8)
            // NOTE: We reuse Input0 because it's free after K data processing
            //       and before d_K loading.
            //       Previously used StateMax (zmm16) which CLOBBERED softmax state!
            Xmm xmm_tmp = guard_k_data.xmm();

            // d_K scale and sum_qs_K - use Input1 (zmm9) for both (sequential use)
            // NOTE: xmm_d_k and xmm_sum_qs_k can share Input0/Input1 since they're
            //       used sequentially within the reduce_and_scale lambda.
            auto guard_k_scales = gen.borrow<Input1>();
            Xmm xmm_d_k = guard_k_data.xmm();        // Reuse Input0 (after K data loaded)
            Xmm xmm_sum_qs_k = guard_k_scales.xmm(); // Input1

            // Correction term - use Scratch5 (zmm25)
            // CRITICAL: This was previously hardcoded to xmm14, which conflicted
            //           with d_Q registers when dq_reg_base_idx=14!
            auto guard_correction = gen.borrow<Scratch5>();
            Xmm xmm_correction = guard_correction.xmm();

            // Zero all 4 output accumulators
            gen.vxorps(dst_xmm0, dst_xmm0, dst_xmm0);
            gen.vxorps(dst_xmm1, dst_xmm1, dst_xmm1);
            gen.vxorps(dst_xmm2, dst_xmm2, dst_xmm2);
            gen.vxorps(dst_xmm3, dst_xmm3, dst_xmm3);

            for (int b = 0; b < num_blocks; ++b)
            {
                gen.debug_emit("  Block " + std::to_string(b) + " (4x)");

                // Get pre-loaded Q for this block (caller owns these registers)
                Ymm ymm_q(q_reg_base_idx + b);
                // d_Q is loaded on-demand inside reduce_and_scale to save registers

                int k_block_offset = b * 36;

                // Zero vpdpbusd accumulators for this block
                gen.vxorps(ymm_acc0, ymm_acc0, ymm_acc0);
                gen.vxorps(ymm_acc1, ymm_acc1, ymm_acc1);
                gen.vxorps(ymm_acc2, ymm_acc2, ymm_acc2);
                gen.vxorps(ymm_acc3, ymm_acc3, ymm_acc3);

                // Process K[0]: load K data, vpdpbusd
                gen.vmovdqu8(ymm_k, gen.ptr[reg_K_ptr + k_block_offset + 4]);
                gen.vpdpbusd(ymm_acc0, ymm_q, ymm_k);

                // Process K[1]
                gen.vmovdqu8(ymm_k, gen.ptr[reg_K_ptr + k_stride + k_block_offset + 4]);
                gen.vpdpbusd(ymm_acc1, ymm_q, ymm_k);

                // Process K[2]
                gen.vmovdqu8(ymm_k, gen.ptr[reg_K_ptr + 2 * k_stride + k_block_offset + 4]);
                gen.vpdpbusd(ymm_acc2, ymm_q, ymm_k);

                // Process K[3]
                gen.vmovdqu8(ymm_k, gen.ptr[reg_K_ptr + 3 * k_stride + k_block_offset + 4]);
                gen.vpdpbusd(ymm_acc3, ymm_q, ymm_k);

                // Horizontal sum and scale each accumulator
                // Helper lambda to reduce and scale one accumulator
                auto reduce_and_scale = [&](Ymm ymm_acc, Xmm dst_xmm, int k_idx)
                {
                    // Horizontal sum: 8×int32 → scalar int32
                    gen.vextracti32x4(xmm_tmp, ymm_acc, 1);
                    gen.vpaddd(Xmm(ymm_acc.getIdx()), Xmm(ymm_acc.getIdx()), xmm_tmp);
                    gen.vpshufd(xmm_tmp, Xmm(ymm_acc.getIdx()), 0x4E);
                    gen.vpaddd(Xmm(ymm_acc.getIdx()), Xmm(ymm_acc.getIdx()), xmm_tmp);
                    gen.vpshufd(xmm_tmp, Xmm(ymm_acc.getIdx()), 0xB1);
                    gen.vpaddd(Xmm(ymm_acc.getIdx()), Xmm(ymm_acc.getIdx()), xmm_tmp);

                    // Convert to float
                    gen.vcvtdq2ps(Xmm(ymm_acc.getIdx()), Xmm(ymm_acc.getIdx()));

                    // Load K[k_idx] scales: d_K and sum_qs_K
                    int64_t k_offset = k_idx * k_stride + k_block_offset;
                    gen.vpbroadcastw(xmm_d_k, gen.ptr[reg_K_ptr + k_offset]);
                    gen.vcvtph2ps(xmm_d_k, xmm_d_k);

                    gen.vpbroadcastw(xmm_sum_qs_k, gen.ptr[reg_K_ptr + k_offset + 2]);
                    gen.vpmovsxwd(xmm_sum_qs_k, xmm_sum_qs_k);
                    gen.vcvtdq2ps(xmm_sum_qs_k, xmm_sum_qs_k);

                    // Correction: 128.0f * sum_qs_K
                    gen.mov(gen.eax, 0x43000000); // 128.0f
                    gen.vmovd(xmm_correction, gen.eax);
                    gen.vbroadcastss(xmm_correction, xmm_correction);
                    gen.vmulps(xmm_correction, xmm_correction, xmm_sum_qs_k);

                    // raw_dot - correction
                    gen.vsubps(Xmm(ymm_acc.getIdx()), Xmm(ymm_acc.getIdx()), xmm_correction);

                    // Load d_Q on-demand into xmm_sum_qs_k (zmm9) - reusing register!
                    // d_Q is at reg_Q_ptr + q_head_offset + b * 36
                    gen.vpbroadcastw(xmm_sum_qs_k, gen.ptr[reg_Q_ptr + q_head_offset + b * 36]);
                    gen.vcvtph2ps(xmm_sum_qs_k, xmm_sum_qs_k);

                    // Scale: result * d_Q * d_K
                    gen.vmulps(Xmm(ymm_acc.getIdx()), Xmm(ymm_acc.getIdx()), xmm_sum_qs_k); // * d_Q
                    gen.vmulps(Xmm(ymm_acc.getIdx()), Xmm(ymm_acc.getIdx()), xmm_d_k);      // * d_K

                    // Accumulate into output
                    gen.vaddps(dst_xmm, dst_xmm, Xmm(ymm_acc.getIdx()));
                };

                reduce_and_scale(ymm_acc0, dst_xmm0, 0);
                reduce_and_scale(ymm_acc1, dst_xmm1, 1);
                reduce_and_scale(ymm_acc2, dst_xmm2, 2);
                reduce_and_scale(ymm_acc3, dst_xmm3, 3);
            }

            // Apply global scale to all 4 outputs
            gen.vmulss(dst_xmm0, dst_xmm0, scale_xmm);
            gen.vmulss(dst_xmm1, dst_xmm1, scale_xmm);
            gen.vmulss(dst_xmm2, dst_xmm2, scale_xmm);
            gen.vmulss(dst_xmm3, dst_xmm3, scale_xmm);
        }
    };

    /**
     * @brief Standalone JIT kernel for Q8_1 dot product (for testing)
     *
     * Generates a callable function that computes dot product between
     * two Q8_1 vectors.
     *
     * Function signature:
     *   float kernel(const Q8_1Block* q, const Q8_1Block* k, int num_blocks, float scale)
     */
    class JitQ8DotProductKernel : public JitMicrokernelBase
    {
    public:
        using kernel_func_t = float (*)(
            const microkernels::Q8_1Block *q,
            const microkernels::Q8_1Block *k,
            int num_blocks,
            float scale);

        explicit JitQ8DotProductKernel(int max_blocks = 8, bool debug = false)
            : JitMicrokernelBase(8 * 1024, debug), max_blocks_(max_blocks)
        {
            generate();
        }

        kernel_func_t get_kernel()
        {
            return getCode<kernel_func_t>();
        }

    private:
        int max_blocks_;
        JitQ8DotProductEmitter dot_emitter_;

        void generate()
        {
            using namespace Xbyak;

            debug_emit("JitQ8DotProductKernel::generate()");

            // Function: float kernel(const Q8_1Block* q, const Q8_1Block* k, int num_blocks, float scale)
            // Args (System V ABI): rdi = q, rsi = k, edx = num_blocks, xmm0 = scale
            const Reg64 &reg_q = rdi;
            const Reg64 &reg_k = rsi;
            const Reg64 &reg_num_blocks = rdx;
            const Xmm &xmm_scale = xmm0;

            // Save callee-saved registers
            push(rbx);
            push(rbp);

            // Initialize constants
            debug_emit("  Init constants");
            emit_broadcast_i32_const(const_128().zmm(), 0x80808080, rax);

            // Allocate stack for Q blocks (padded to 64 bytes each)
            // Max stack = max_blocks * 64 + alignment
            int stack_size = (max_blocks_ * 64 + 63) & ~63;
            sub(rsp, stack_size);

            // Copy Q blocks to stack (for aligned access)
            debug_emit("  Copy Q to stack");
            mov(rcx, reg_num_blocks);
            xor_(rbx, rbx); // block index

            Label copy_loop, copy_done;
            L(copy_loop);
            cmp(rbx, rcx);
            jge(copy_done, T_NEAR);

            // Load 36-byte Q8_1Block, store padded to 64 bytes on stack
            // Q block at [reg_q + rbx * 36]
            lea(rbp, ptr[reg_q + rbx * 8]); // rbp = q + rbx * 8
            lea(rbp, ptr[rbp + rbx * 4]);   // rbp = q + rbx * 8 + rbx * 4 = q + rbx * 12
            lea(rbp, ptr[rbp + rbx * 8]);   // rbp = q + rbx * 12 + rbx * 8 = q + rbx * 20
            lea(rbp, ptr[rbp + rbx * 16]);  // rbp = q + rbx * 20 + rbx * 16 = q + rbx * 36

            // Actually, simpler: rbp = reg_q + rbx * 36
            // Use imul for non-power-of-2 multiply
            mov(rbp, rbx);
            imul(rbp, rbp, 36);
            add(rbp, reg_q);

            // Copy 32 bytes + 4 bytes
            vmovdqu(ymm0, ptr[rbp]); // First 32 bytes
            vmovdqu(ptr[rsp + rbx * 64], ymm0);
            mov(eax, ptr[rbp + 32]); // Last 4 bytes
            mov(ptr[rsp + rbx * 64 + 32], eax);

            inc(rbx);
            jmp(copy_loop, T_NEAR);
            L(copy_done);

            // Now emit the dot product (hardcoded for num_blocks, or use jump table)
            // For simplicity, support num_blocks = 2 (head_dim=64)
            // TODO: Support variable num_blocks with jump table
            debug_emit("  Emit dot product (2 blocks)");

            // For this test kernel, assume num_blocks=2
            // A production version would use a jump table
            Xmm xmm_result = xmm1;
            dot_emitter_.emit_dot_product(*this, xmm_result, reg_k, rsp, 0, 2, xmm_scale);

            // Return value in xmm0
            vmovss(xmm0, xmm_result);

            // Restore stack and registers
            add(rsp, stack_size);
            pop(rbp);
            pop(rbx);
            ret();
        }
    };

} // namespace llaminar::v2::kernels::jit
