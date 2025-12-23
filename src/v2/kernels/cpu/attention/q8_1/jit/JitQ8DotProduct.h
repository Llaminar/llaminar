/**
 * @file JitQ8DotProduct.h
 * @brief JIT Microkernel μK1: Q8_1 dot product with proper scaling
 * @author David Sanftenberg
 * @date December 2025
 *
 * JIT-generated Q8_1 dot product using AVX-512 VNNI (vpdpbusd).
 * Mirrors the reference Q8DotProduct.h microkernel.
 *
 * Algorithm:
 *   For each block b in [0, num_blocks):
 *     d_q = fp16_to_fp32(Q[b].d)
 *     d_k = fp16_to_fp32(K[b].d)
 *     sum_qs_k = K[b].sum_qs
 *
 *     raw_dot = vpdpbusd(Q_unsigned, K_signed)  // (Q+128) * K
 *     corrected_dot = raw_dot - 128 * sum_qs_k   // Undo unsigned bias
 *     scaled_dot = corrected_dot * d_q * d_k
 *
 *     score += scaled_dot
 *
 *   return score * global_scale
 *
 * Register conventions:
 * - Input: Q/K block pointers, num_blocks, global_scale in GP regs
 * - Output: score in XMM (scalar)
 * - Constants: zmm_128() must be initialized to 0x80808080
 * - Scratch: zmm20-25, xmm4-9 freely clobbered
 */

#pragma once

#include "JitMicrokernelBase.h"
#include "../microkernels/Q8DotProduct.h" // For Q8_1Block struct

namespace llaminar::v2::kernels::jit
{

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
                gen.vpxord(ymm_q, ymm_q, Ymm(gen.zmm_128().getIdx()));

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
            int dq_reg_base_idx)
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

            // XMM registers for K scales and correction
            Xmm xmm_d_k(9);
            Xmm xmm_sum_qs_k(23);     // Scratch zone
            Xmm xmm_correction(24);   // Scratch zone
            Xmm xmm_block_result(20); // Alias of ymm_dot
            Xmm xmm_tmp(25);          // Scratch zone

            for (int b = 0; b < num_blocks; ++b)
            {
                // Use pre-loaded Q registers
                Ymm ymm_q(q_reg_base_idx + b);
                Xmm xmm_d_q(dq_reg_base_idx + b);

                int k_offset = b * 36;

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

                // Apply correction
                gen.vsubps(xmm_block_result, xmm_block_result, xmm_correction);

                // Apply scales: result * d_Q * d_K
                gen.vmulps(xmm_block_result, xmm_block_result, xmm_d_q);
                gen.vmulps(xmm_block_result, xmm_block_result, xmm_d_k);

                // Accumulate
                gen.vaddps(dst_xmm, dst_xmm, xmm_block_result);
            }

            // Apply global scale
            gen.vmulss(dst_xmm, dst_xmm, scale_xmm);
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
            emit_broadcast_i32_const(zmm_128(), 0x80808080, rax);

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
