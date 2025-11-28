#pragma once

#include "../../../../../external/onednn/third_party/xbyak/xbyak.h"
#include "../../../tensors/Tensors.h"
#include <vector>
#include <cstdint>
#include <iostream>

namespace llaminar2
{
    namespace gemm_v4
    {

        struct QuantisedPackedWeights
        {
            // Packed data: [K/4][N][4] (int8_t)
            // But we flatten it to [K/4 * N * 4]
            std::vector<int8_t> packed_data;

            // Compensation: [K/32][N] (int32_t)
            std::vector<int32_t> compensation;

            // Scales: [K/32][N] (float)
            std::vector<float> scales;

            // Mins: [K/32][N] (float) - for asymmetric quantization
            std::vector<float> mins;

            int K;
            int N;
        };

        struct QuantisedGemmParams
        {
            const void *A;
            const void *B_packed;
            const void *comp;
            const void *scales;
            const void *mins; // Added for asymmetric quantization
            float *C;
            int K_blocks;
            int N;
            int ldc;
            const float *bias;
            const float *mask;
            // Optional: Online Softmax
            float *local_max; // Output: Local max for each 64-col block
            float *local_sum; // Output: Local sum for each 64-col block
            bool do_softmax;
            int A_stride;
            const float *gate_input; // For SwiGLU
            bool do_swiglu;
        };

        class QuantisedGemmJit_M1 : public Xbyak::CodeGenerator
        {
        public:
            QuantisedGemmJit_M1() : Xbyak::CodeGenerator(4096 * 32)
            { // Allocate enough space
                generate();
            }

            // Signature:
            using kernel_func_t = void (*)(const QuantisedGemmParams *params);

            kernel_func_t get_kernel()
            {
                return getCode<kernel_func_t>();
            }

            static void pack_weights(const int8_t *B, int K, int N, QuantisedPackedWeights &packed)
            {
                // Dummy implementation for now, real one in QuantisedGemmKernel
                packed.K = K;
                packed.N = N;
                packed.packed_data.resize(K * N);
                packed.compensation.resize((K / 32) * N);
                packed.scales.resize((K / 32) * N);
            }

        private:
            void generate()
            {
                using namespace Xbyak;

                // Registers
                const Reg64 &reg_A = rdi;
                const Reg64 &reg_B = rsi;
                const Reg64 &reg_Comp = rdx;
                const Reg64 &reg_Scales = rcx;
                const Reg64 &reg_C = r8;
                const Reg64 &reg_K_blocks = r9;
                // N is on stack

                const Reg64 &reg_tmp = rax;
                const Reg64 &reg_stride = rbx; // Stride N*4
                const Reg64 &reg_loop_N = r10;
                const Reg64 &reg_loop_K = r11;
                const Reg64 &reg_B_cursor = r12;
                const Reg64 &reg_Comp_cursor = r13;
                const Reg64 &reg_C_cursor = r14;
                const Reg64 &reg_A_cursor = r15;
                const Reg64 &reg_Scales_cursor = rbp;

                const Reg32 &reg_tmp_32 = eax;

                // ZMMs
                // Accumulators for C (4x16 = 64 elements)
                const Zmm &zmm_c0 = zmm0;
                const Zmm &zmm_c1 = zmm1;
                const Zmm &zmm_c2 = zmm2;
                const Zmm &zmm_c3 = zmm3;

                // Temp accumulators (int32)
                const Zmm &zmm_acc0 = zmm4;
                const Zmm &zmm_acc1 = zmm5;
                const Zmm &zmm_acc2 = zmm6;
                const Zmm &zmm_acc3 = zmm7;

                // Constants
                const Zmm &zmm_scale = zmm8;
                const Ymm &ymm_scale = ymm8;
                const Zmm &zmm_128 = zmm9;
                const Zmm &zmm_neg_128 = zmm10;
                const Zmm &zmm_a = zmm11;
                const Zmm &zmm_b0 = zmm12;
                const Zmm &zmm_b1 = zmm13;
                const Zmm &zmm_b2 = zmm14;
                const Zmm &zmm_b3 = zmm15;
                const Ymm &ymm_tmp = ymm16;     // Temp for broadcast
                const Zmm &zmm_scale_b = zmm17; // Scale for B
                const Zmm &zmm_bias = zmm18;
                const Zmm &zmm_mask = zmm19;

                // Prologue
                push(rbx);
                push(rbp);
                push(r12);
                push(r13);
                push(r14);
                push(r15);

                // Save params pointer to stack (we need it for bias/mask)
                const Reg64 &reg_params = rdi;
                push(reg_params);

                // Load arguments from params struct
                // rdi is params
                mov(reg_B, ptr[reg_params + 8]);       // B_packed
                mov(reg_Comp, ptr[reg_params + 16]);   // comp
                mov(reg_Scales, ptr[reg_params + 24]); // scales
                // mins is at +32
                mov(reg_C, ptr[reg_params + 40]);                // C
                mov(reg_K_blocks.cvt32(), ptr[reg_params + 48]); // K_blocks

                // Load N into reg_loop_N temporarily to push it
                mov(reg_loop_N.cvt32(), ptr[reg_params + 52]); // N
                push(reg_loop_N);                              // Push N to stack (now at rsp)

                // Load ldc into reg_stride temporarily
                mov(reg_stride.cvt32(), ptr[reg_params + 56]); // ldc
                shl(reg_stride, 2);                            // ldc * 4 (stride in bytes)

                // Load A last (overwrites params pointer in rdi)
                mov(reg_A, ptr[reg_params + 0]); // A

                // Setup constants
                mov(reg_tmp_32, 0x80808080);
                vpbroadcastd(zmm_128, reg_tmp_32);

                mov(reg_tmp_32, -128);
                vpbroadcastd(zmm_neg_128, reg_tmp_32);

                // Loop over N (step 64)
                xor_(reg_loop_N, reg_loop_N);

                Label loop_N_label;
                L(loop_N_label);
                {
                    // Check if we are done
                    cmp(reg_loop_N, ptr[rsp]); // N is at rsp
                    jge("end_kernel", T_NEAR);

                    // Setup cursors
                    mov(reg_A_cursor, reg_A);

                    // Reload params to rax to restore clobbered registers (rdx, rcx)
                    mov(rax, ptr[rsp + 8]);
                    mov(reg_Comp, ptr[rax + 16]);   // Restore comp
                    mov(reg_Scales, ptr[rax + 24]); // Restore scales

                    // Offset calculation for Comp, Scales, C: loop_N * 4
                    mov(reg_tmp, reg_loop_N);
                    shl(reg_tmp, 2); // loop_N * 4

                    // Comp cursor: reg_Comp + offset
                    mov(reg_Comp_cursor, reg_Comp);
                    add(reg_Comp_cursor, reg_tmp);

                    // Scales cursor: reg_Scales + offset
                    mov(reg_Scales_cursor, reg_Scales);
                    add(reg_Scales_cursor, reg_tmp);

                    // C cursor: reg_C + offset
                    mov(reg_C_cursor, reg_C);
                    add(reg_C_cursor, reg_tmp);

                    // Initialize C accumulators
                    vmovups(zmm_c0, ptr[reg_C_cursor + 0 * 64]);
                    vmovups(zmm_c1, ptr[reg_C_cursor + 1 * 64]);
                    vmovups(zmm_c2, ptr[reg_C_cursor + 2 * 64]);
                    vmovups(zmm_c3, ptr[reg_C_cursor + 3 * 64]);

                    // Overwrite reg_C_cursor (r14) with Mins cursor
                    mov(rax, ptr[rsp + 8]);
                    mov(r14, ptr[rax + 32]); // mins ptr
                    mov(reg_tmp, reg_loop_N);
                    shl(reg_tmp, 2);
                    add(r14, reg_tmp);
                    // Now r14 is reg_Mins_cursor

                    // B cursor calculation: (loop_N / 64) * (K_blocks * 2048)
                    mov(reg_tmp, reg_loop_N);
                    shr(reg_tmp, 6);             // loop_N / 64
                    imul(reg_tmp, reg_K_blocks); // * K_blocks
                    shl(reg_tmp, 11);            // * 2048

                    mov(reg_B_cursor, reg_B);
                    add(reg_B_cursor, reg_tmp);

                    // Loop over K blocks
                    mov(reg_loop_K, reg_K_blocks);

                    Label loop_K_label;
                    L(loop_K_label);
                    {
                        // Load scale A (half) -> float broadcast
                        vpbroadcastw(ymm_tmp, ptr[reg_A_cursor]);
                        vcvtph2ps(zmm_scale, ymm_tmp);

                        // --- Asymmetric Weights Correction ---
                        // Load sum_qs (offset 2)
                        vpbroadcastw(ymm_tmp, ptr[reg_A_cursor + 2]);
                        vpmovsxwd(zmm20, ymm_tmp); // int16 -> int32 (zmm20 = sum_qs)
                        vcvtdq2ps(zmm20, zmm20);   // int32 -> float

                        // Multiply by scale A (zmm_scale)
                        vmulps(zmm20, zmm20, zmm_scale);

                        // Load mins (using r14)
                        vmovups(zmm21, ptr[r14 + 0 * 64]);
                        vmovups(zmm22, ptr[r14 + 1 * 64]);
                        vmovups(zmm23, ptr[r14 + 2 * 64]);
                        vmovups(zmm24, ptr[r14 + 3 * 64]);

                        // Multiply by sum_qs * scale_A
                        vmulps(zmm21, zmm21, zmm20);
                        vmulps(zmm22, zmm22, zmm20);
                        vmulps(zmm23, zmm23, zmm20);
                        vmulps(zmm24, zmm24, zmm20);

                        // Add to C
                        vaddps(zmm_c0, zmm_c0, zmm21);
                        vaddps(zmm_c1, zmm_c1, zmm22);
                        vaddps(zmm_c2, zmm_c2, zmm23);
                        vaddps(zmm_c3, zmm_c3, zmm24);

                        // Advance Mins cursor
                        add(r14, reg_stride);

                        vmovups(zmm_acc0, ptr[reg_Comp_cursor + 0 * 64]);
                        vmovups(zmm_acc1, ptr[reg_Comp_cursor + 1 * 64]);
                        vmovups(zmm_acc2, ptr[reg_Comp_cursor + 2 * 64]);
                        vmovups(zmm_acc3, ptr[reg_Comp_cursor + 3 * 64]);

                        // Multiply by -128
                        vpmulld(zmm_acc0, zmm_acc0, zmm_neg_128);
                        vpmulld(zmm_acc1, zmm_acc1, zmm_neg_128);
                        vpmulld(zmm_acc2, zmm_acc2, zmm_neg_128);
                        vpmulld(zmm_acc3, zmm_acc3, zmm_neg_128);

                        // Inner loop over 32 elements (8 steps of 4)
                        for (int i = 0; i < 8; ++i)
                        {
                            // Load A (4 bytes) -> broadcast
                            vpbroadcastd(zmm_a, ptr[reg_A_cursor + 4 + i * 4]);

                            // Convert s8 to u8 (xor 0x80)
                            vpxord(zmm_a, zmm_a, zmm_128);

                            // Load B and accumulate
                            vmovups(zmm_b0, ptr[reg_B_cursor + 0 * 64]);
                            vpdpbusd(zmm_acc0, zmm_a, zmm_b0);

                            vmovups(zmm_b1, ptr[reg_B_cursor + 1 * 64]);
                            vpdpbusd(zmm_acc1, zmm_a, zmm_b1);

                            vmovups(zmm_b2, ptr[reg_B_cursor + 2 * 64]);
                            vpdpbusd(zmm_acc2, zmm_a, zmm_b2);

                            vmovups(zmm_b3, ptr[reg_B_cursor + 3 * 64]);
                            vpdpbusd(zmm_acc3, zmm_a, zmm_b3);

                            // Advance B cursor by 256 bytes (64 columns * 4 rows)
                            add(reg_B_cursor, 256);
                        }

                        // Convert acc to float
                        vcvtdq2ps(zmm_acc0, zmm_acc0);
                        vcvtdq2ps(zmm_acc1, zmm_acc1);
                        vcvtdq2ps(zmm_acc2, zmm_acc2);
                        vcvtdq2ps(zmm_acc3, zmm_acc3);

                        // Load scale B
                        vmovups(zmm_scale_b, ptr[reg_Scales_cursor + 0 * 64]);
                        vmulps(zmm_acc0, zmm_acc0, zmm_scale_b);

                        vmovups(zmm_scale_b, ptr[reg_Scales_cursor + 1 * 64]);
                        vmulps(zmm_acc1, zmm_acc1, zmm_scale_b);

                        vmovups(zmm_scale_b, ptr[reg_Scales_cursor + 2 * 64]);
                        vmulps(zmm_acc2, zmm_acc2, zmm_scale_b);

                        vmovups(zmm_scale_b, ptr[reg_Scales_cursor + 3 * 64]);
                        vmulps(zmm_acc3, zmm_acc3, zmm_scale_b);

                        // Multiply by scale A
                        vmulps(zmm_acc0, zmm_acc0, zmm_scale);
                        vmulps(zmm_acc1, zmm_acc1, zmm_scale);
                        vmulps(zmm_acc2, zmm_acc2, zmm_scale);
                        vmulps(zmm_acc3, zmm_acc3, zmm_scale);

                        // Accumulate to C
                        vaddps(zmm_c0, zmm_c0, zmm_acc0);
                        vaddps(zmm_c1, zmm_c1, zmm_acc1);
                        vaddps(zmm_c2, zmm_c2, zmm_acc2);
                        vaddps(zmm_c3, zmm_c3, zmm_acc3);

                        // Advance pointers
                        add(reg_A_cursor, 36); // sizeof(Q8_1Block)

                        // Advance Comp and Scales cursor by stride (N*4)
                        add(reg_Comp_cursor, reg_stride);
                        add(reg_Scales_cursor, reg_stride);

                        dec(reg_loop_K);
                        jnz(loop_K_label, T_NEAR);
                    }

                    // Restore reg_C_cursor (r14)
                    mov(rax, ptr[rsp + 8]);           // params
                    mov(reg_C_cursor, ptr[rax + 40]); // C
                    mov(reg_tmp, reg_loop_N);
                    shl(reg_tmp, 2);
                    add(reg_C_cursor, reg_tmp);

                    // --- Bias & Mask ---
                    // Retrieve params pointer from stack (rsp + 8)
                    // rsp points to N. rsp+8 points to params.
                    mov(rax, ptr[rsp + 8]);

                    // 1. Bias
                    mov(rdx, ptr[rax + 64]); // bias ptr
                    test(rdx, rdx);
                    Label skip_bias;
                    jz(skip_bias, T_NEAR);
                    {
                        // Bias is vector of size N.
                        // We are at reg_loop_N offset.
                        // Address = bias_ptr + reg_loop_N * 4
                        lea(rdx, ptr[rdx + reg_loop_N * 4]);

                        vmovups(zmm_bias, ptr[rdx + 0 * 64]);
                        vaddps(zmm_c0, zmm_c0, zmm_bias);

                        vmovups(zmm_bias, ptr[rdx + 1 * 64]);
                        vaddps(zmm_c1, zmm_c1, zmm_bias);

                        vmovups(zmm_bias, ptr[rdx + 2 * 64]);
                        vaddps(zmm_c2, zmm_c2, zmm_bias);

                        vmovups(zmm_bias, ptr[rdx + 3 * 64]);
                        vaddps(zmm_c3, zmm_c3, zmm_bias);
                    }
                    L(skip_bias);

                    // 2. Mask
                    mov(rdx, ptr[rax + 72]); // mask ptr
                    test(rdx, rdx);
                    Label skip_mask;
                    jz(skip_mask, T_NEAR);
                    {
                        // Mask is M x N. For M=1, it's 1 x N.
                        // Same indexing as bias.
                        lea(rdx, ptr[rdx + reg_loop_N * 4]);

                        vmovups(zmm_mask, ptr[rdx + 0 * 64]);
                        vaddps(zmm_c0, zmm_c0, zmm_mask);

                        vmovups(zmm_mask, ptr[rdx + 1 * 64]);
                        vaddps(zmm_c1, zmm_c1, zmm_mask);

                        vmovups(zmm_mask, ptr[rdx + 2 * 64]);
                        vaddps(zmm_c2, zmm_c2, zmm_mask);

                        vmovups(zmm_mask, ptr[rdx + 3 * 64]);
                        vaddps(zmm_c3, zmm_c3, zmm_mask);
                    }
                    L(skip_mask);

                    // 3. Softmax
                    mov(al, ptr[rax + 96]); // do_softmax
                    test(al, al);
                    Label skip_softmax;
                    jz(skip_softmax, T_NEAR);
                    {
                        // Registers for Softmax
                        const Zmm &zmm_max = zmm20;
                        const Zmm &zmm_sum = zmm21;
                        const Zmm &zmm_tmp = zmm22;
                        const Zmm &zmm_c1_const = zmm23;
                        const Zmm &zmm_c2_const = zmm24;
                        const Zmm &zmm_c3_coeff = zmm25;
                        const Zmm &zmm_c4_const = zmm26;
                        const Zmm &zmm_c5_const = zmm27;
                        const Zmm &zmm_inv_ln2 = zmm28;
                        const Zmm &zmm_max_clip = zmm29;
                        const Zmm &zmm_min_clip = zmm30;
                        const Zmm &zmm_127 = zmm31;

                        // Load constants
                        mov(reg_tmp_32, 0x3fb8aa3b); // 1.44269504 (inv_ln2)
                        vpbroadcastd(zmm_inv_ln2, reg_tmp_32);

                        mov(reg_tmp_32, 0x3f7ffffe); // 0.99999994 (c1)
                        vpbroadcastd(zmm_c1_const, reg_tmp_32);

                        mov(reg_tmp_32, 0x3f317218); // 0.69314718 (c2)
                        vpbroadcastd(zmm_c2_const, reg_tmp_32);

                        mov(reg_tmp_32, 0x3e75fdf1); // 0.24022651 (c3)
                        vpbroadcastd(zmm_c3_coeff, reg_tmp_32);

                        mov(reg_tmp_32, 0x3d6356eb); // 0.05550411 (c4)
                        vpbroadcastd(zmm_c4_const, reg_tmp_32);

                        mov(reg_tmp_32, 0x3c1d9422); // 0.00961813 (c5)
                        vpbroadcastd(zmm_c5_const, reg_tmp_32);

                        mov(reg_tmp_32, 0x41200000); // 10.0f (max_clip)
                        vpbroadcastd(zmm_max_clip, reg_tmp_32);

                        mov(reg_tmp_32, 0xc1a00000); // -20.0f (min_clip)
                        vpbroadcastd(zmm_min_clip, reg_tmp_32);

                        mov(reg_tmp_32, 127);
                        vpbroadcastd(zmm_127, reg_tmp_32);

                        // Reload params because reg_tmp_32 (eax) clobbered rax
                        mov(rax, ptr[rsp + 8]);

                        // 1. Compute Max
                        vmaxps(zmm_max, zmm_c0, zmm_c1);
                        vmaxps(zmm_max, zmm_max, zmm_c2);
                        vmaxps(zmm_max, zmm_max, zmm_c3);

                        // Horizontal reduction of zmm_max
                        // Reduce 512 -> 256
                        vextractf64x4(ymm4, zmm_max, 1);           // Use ymm4 as tmp
                        vmaxps(ymm4, ymm4, Ymm(zmm_max.getIdx())); // ymm_max is low part
                        // Reduce 256 -> 128
                        vextractf128(xmm5, ymm4, 1);
                        vmaxps(xmm4, xmm4, xmm5);
                        // Reduce 128 -> 64
                        vpermilps(xmm5, xmm4, 0b01001110);
                        vmaxps(xmm4, xmm4, xmm5);
                        // Reduce 64 -> 32
                        vpermilps(xmm5, xmm4, 0b10110001);
                        vmaxps(xmm4, xmm4, xmm5);

                        // Broadcast max back to zmm_max
                        vpbroadcastd(zmm_max, xmm4);

                        // Store max to local_max
                        // local_max ptr is at offset 80
                        mov(rdx, ptr[rax + 80]);
                        // Offset: (reg_loop_N / 64) * 4
                        mov(r15, reg_loop_N); // Use r15 as temp
                        shr(r15, 4);          // / 64 * 4 = / 16
                        vmovss(ptr[rdx + r15], xmm4);

                        // 2. Compute Sum of Exp
                        vpxord(zmm_sum, zmm_sum, zmm_sum);

                        auto compute_exp_accumulate = [&](const Zmm &zmm_val)
                        {
                            // tmp = val - max
                            vsubps(zmm_tmp, zmm_val, zmm_max);

                            // Clip
                            vminps(zmm_tmp, zmm_tmp, zmm_max_clip);
                            vmaxps(zmm_tmp, zmm_tmp, zmm_min_clip);

                            // xf = x * inv_ln2
                            vmulps(zmm_tmp, zmm_tmp, zmm_inv_ln2);

                            // fx = floor(xf)
                            // Use zmm18 as fx (zmm_fx)
                            const Zmm &zmm_fx = zmm18;
                            vrndscaleps(zmm_fx, zmm_tmp, 0x01);

                            // fpart = xf - fx
                            vsubps(zmm_tmp, zmm_tmp, zmm_fx);

                            // Polynomial
                            // p = c5
                            const Zmm &zmm_p = zmm19; // Use zmm19 as p (was mask, free now)
                            vmovaps(zmm_p, zmm_c5_const);
                            vfmadd213ps(zmm_p, zmm_tmp, zmm_c4_const);
                            vfmadd213ps(zmm_p, zmm_tmp, zmm_c3_coeff);
                            vfmadd213ps(zmm_p, zmm_tmp, zmm_c2_const);
                            vfmadd213ps(zmm_p, zmm_tmp, zmm_c1_const);

                            // 2^floor(x/ln2)
                            vcvtps2dq(zmm_fx, zmm_fx); // Convert fx to int
                            vpaddd(zmm_fx, zmm_fx, zmm_127);
                            vpslld(zmm_fx, zmm_fx, 23);

                            // Result = p * 2^fx
                            vmulps(zmm_p, zmm_p, zmm_fx); // zmm_p is exp result

                            // Accumulate
                            vaddps(zmm_sum, zmm_sum, zmm_p);
                        };

                        compute_exp_accumulate(zmm_c0);
                        compute_exp_accumulate(zmm_c1);
                        compute_exp_accumulate(zmm_c2);
                        compute_exp_accumulate(zmm_c3);

                        // Horizontal reduction of zmm_sum
                        vextractf64x4(ymm4, zmm_sum, 1);
                        vaddps(ymm4, ymm4, Ymm(zmm_sum.getIdx()));
                        vextractf128(xmm5, ymm4, 1);
                        vaddps(xmm4, xmm4, xmm5);
                        vpermilps(xmm5, xmm4, 0b01001110);
                        vaddps(xmm4, xmm4, xmm5);
                        vpermilps(xmm5, xmm4, 0b10110001);
                        vaddps(xmm4, xmm4, xmm5);

                        // Store sum to local_sum
                        // local_sum ptr is at offset 88
                        mov(rdx, ptr[rax + 88]);
                        // Offset: r15 already has it
                        vmovss(ptr[rdx + r15], xmm4);
                    }
                    L(skip_softmax);

                    // 4. SwiGLU
                    // Reload params (rax might be clobbered)
                    mov(rax, ptr[rsp + 8]);  // params is at rsp + 8
                    cmp(byte[rax + 112], 1); // do_swiglu
                    Label skip_swiglu;
                    jne(skip_swiglu, T_NEAR);
                    {
                        // Registers for SwiGLU (reuse Softmax regs since they are free now)
                        const Zmm &zmm_gate = zmm20;
                        const Zmm &zmm_tmp = zmm21;
                        const Zmm &zmm_c1_const = zmm22;
                        const Zmm &zmm_c2_const = zmm23;
                        const Zmm &zmm_c3_coeff = zmm24;
                        const Zmm &zmm_c4_const = zmm25;
                        const Zmm &zmm_c5_const = zmm26;
                        const Zmm &zmm_inv_ln2 = zmm27;
                        const Zmm &zmm_max_clip = zmm28;
                        const Zmm &zmm_min_clip = zmm29;
                        const Zmm &zmm_127 = zmm30;
                        const Zmm &zmm_one = zmm31;

                        // Load constants
                        mov(reg_tmp_32, 0x3fb8aa3b); // 1.44269504 (inv_ln2)
                        vpbroadcastd(zmm_inv_ln2, reg_tmp_32);

                        mov(reg_tmp_32, 0x3f7ffffe); // 0.99999994 (c1)
                        vpbroadcastd(zmm_c1_const, reg_tmp_32);

                        mov(reg_tmp_32, 0x3f317218); // 0.69314718 (c2)
                        vpbroadcastd(zmm_c2_const, reg_tmp_32);

                        mov(reg_tmp_32, 0x3e75fdf1); // 0.24022651 (c3)
                        vpbroadcastd(zmm_c3_coeff, reg_tmp_32);

                        mov(reg_tmp_32, 0x3d6356eb); // 0.05550411 (c4)
                        vpbroadcastd(zmm_c4_const, reg_tmp_32);

                        mov(reg_tmp_32, 0x3c1d9422); // 0.00961813 (c5)
                        vpbroadcastd(zmm_c5_const, reg_tmp_32);

                        mov(reg_tmp_32, 0x42b00000); // 88.0f (max_clip)
                        vpbroadcastd(zmm_max_clip, reg_tmp_32);

                        mov(reg_tmp_32, 0xc2b00000); // -88.0f (min_clip)
                        vpbroadcastd(zmm_min_clip, reg_tmp_32);

                        mov(reg_tmp_32, 0x3f800000); // 1.0f
                        vpbroadcastd(zmm_one, reg_tmp_32);

                        mov(reg_tmp_32, 127);
                        vpbroadcastd(zmm_127, reg_tmp_32);

                        // Reload params because rax was clobbered by constants
                        mov(rax, ptr[rsp + 8]);

                        // Load gate_input pointer
                        mov(rdx, ptr[rax + 104]);

                        auto compute_swish = [&](const Zmm &zmm_val, int offset)
                        {
                            // Load gate
                            vmovups(zmm_gate, ptr[rdx + offset * 64]);

                            // Compute sigmoid(gate)
                            // y = -gate
                            vxorps(zmm_tmp, zmm_tmp, zmm_tmp);
                            vsubps(zmm_tmp, zmm_tmp, zmm_gate); // zmm_tmp = -gate

                            // Clip y to [-88, 88]
                            vminps(zmm_tmp, zmm_tmp, zmm_max_clip);
                            vmaxps(zmm_tmp, zmm_tmp, zmm_min_clip);

                            // exp(y)
                            // xf = x * inv_ln2
                            vmulps(zmm_tmp, zmm_tmp, zmm_inv_ln2);

                            // fx = floor(xf)
                            // Use zmm18 as fx (zmm_fx) - zmm_bias (zmm18) is free
                            const Zmm &zmm_fx = zmm18;
                            vrndscaleps(zmm_fx, zmm_tmp, 0x01);

                            // fpart = xf - fx
                            vsubps(zmm_tmp, zmm_tmp, zmm_fx);

                            // Polynomial
                            // p = c5
                            const Zmm &zmm_p = zmm19; // zmm_mask (zmm19) is free
                            vmovaps(zmm_p, zmm_c5_const);
                            vfmadd213ps(zmm_p, zmm_tmp, zmm_c4_const);
                            vfmadd213ps(zmm_p, zmm_tmp, zmm_c3_coeff);
                            vfmadd213ps(zmm_p, zmm_tmp, zmm_c2_const);
                            vfmadd213ps(zmm_p, zmm_tmp, zmm_c1_const);

                            // 2^floor(x/ln2)
                            vcvtps2dq(zmm_fx, zmm_fx);
                            vpaddd(zmm_fx, zmm_fx, zmm_127);
                            vpslld(zmm_fx, zmm_fx, 23);

                            // Result = p * 2^fx
                            vmulps(zmm_p, zmm_p, zmm_fx); // zmm_p is exp(-gate)

                            // den = 1 + exp
                            vaddps(zmm_p, zmm_p, zmm_one);

                            // res = gate / den
                            vdivps(zmm_p, zmm_gate, zmm_p); // zmm_p = swish(gate)

                            // val = val * res
                            vmulps(zmm_val, zmm_val, zmm_p);
                        };

                        compute_swish(zmm_c0, 0);
                        compute_swish(zmm_c1, 1);
                        compute_swish(zmm_c2, 2);
                        compute_swish(zmm_c3, 3);
                    }
                    L(skip_swiglu);

                    // Store C
                    vmovups(ptr[reg_C_cursor + 0 * 64], zmm_c0);
                    vmovups(ptr[reg_C_cursor + 1 * 64], zmm_c1);
                    vmovups(ptr[reg_C_cursor + 2 * 64], zmm_c2);
                    vmovups(ptr[reg_C_cursor + 3 * 64], zmm_c3);

                    // Advance N loop
                    add(reg_loop_N, 64);
                    jmp(loop_N_label, T_NEAR);
                }

                L("end_kernel");
                add(rsp, 16); // Pop N and params
                pop(r15);
                pop(r14);
                pop(r13);
                pop(r12);
                pop(rbp);
                pop(rbx);
                ret();
            }
        };

    }
}
