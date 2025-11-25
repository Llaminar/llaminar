#pragma once

#include "../../../../../external/onednn/third_party/xbyak/xbyak.h"
#include "Q8_1GemmJit_M1.h" // For Q8_1GemmParams
#include <vector>
#include <cstdint>

namespace llaminar2
{
    namespace gemm_v4
    {

        class Q8_1GemmJit_M2 : public Xbyak::CodeGenerator
        {
        public:
            Q8_1GemmJit_M2() : Xbyak::CodeGenerator(4096 * 64)
            {
                generate();
            }

            // Signature:
            using kernel_func_t = void (*)(const Q8_1GemmParams *params);

            kernel_func_t get_kernel()
            {
                return getCode<kernel_func_t>();
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
                // N is on stack (arg 7)
                // ldc is on stack (arg 8)

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
                // C Accumulators (2 rows x 4 cols = 8 regs)
                // Row 0: zmm0..3
                // Row 1: zmm4..7

                // Temp Accumulators (int32) (2 rows x 4 cols = 8 regs)
                // Row 0: zmm8..11
                // Row 1: zmm12..15

                // B registers (4 regs)
                const Zmm &zmm_b0 = zmm16;
                const Zmm &zmm_b1 = zmm17;
                const Zmm &zmm_b2 = zmm18;
                const Zmm &zmm_b3 = zmm19;

                // A registers (2 regs)
                const Zmm &zmm_a0 = zmm20;
                const Zmm &zmm_a1 = zmm21;

                // Constants
                const Zmm &zmm_scale = zmm22;
                const Zmm &zmm_128 = zmm23;
                const Zmm &zmm_neg_128 = zmm24;
                const Ymm &ymm_tmp = ymm25;
                const Zmm &zmm_scale_b0 = zmm26;
                const Zmm &zmm_scale_b1 = zmm27;
                const Zmm &zmm_scale_b2 = zmm28;
                const Zmm &zmm_scale_b3 = zmm29;
                const Zmm &zmm_bias = zmm30;
                const Zmm &zmm_mask = zmm31;

                // Prologue
                push(rbx);
                push(rbp);
                push(r12);
                push(r13);
                push(r14);
                push(r15);

                // Save params pointer
                const Reg64 &reg_params = rdi;
                push(reg_params);

                // Load args
                mov(reg_B, ptr[reg_params + 8]);
                mov(reg_Comp, ptr[reg_params + 16]);
                mov(reg_Scales, ptr[reg_params + 24]);
                mov(reg_C, ptr[reg_params + 32]);
                mov(reg_K_blocks.cvt32(), ptr[reg_params + 40]);

                // Load N and push
                mov(reg_loop_N.cvt32(), ptr[reg_params + 44]);
                push(reg_loop_N);

                // Load ldc and shift
                mov(reg_stride.cvt32(), ptr[reg_params + 48]);
                shl(reg_stride, 2);

                // Load A
                mov(reg_A, ptr[reg_params + 0]);

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
                    cmp(reg_loop_N, ptr[rsp]);
                    jge("end_kernel", T_NEAR);

                    // Setup cursors
                    mov(reg_A_cursor, reg_A);

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

                    // B cursor calculation: (loop_N / 64) * (K_blocks * 2048)
                    mov(reg_tmp, reg_loop_N);
                    shr(reg_tmp, 6);             // loop_N / 64
                    imul(reg_tmp, reg_K_blocks); // * K_blocks
                    shl(reg_tmp, 11);            // * 2048

                    mov(reg_B_cursor, reg_B);
                    add(reg_B_cursor, reg_tmp);

                    // Initialize C Accumulators (zmm0..7) from memory
                    // Row 0
                    vmovups(zmm0, ptr[reg_C_cursor + 0 * 64]);
                    vmovups(zmm1, ptr[reg_C_cursor + 1 * 64]);
                    vmovups(zmm2, ptr[reg_C_cursor + 2 * 64]);
                    vmovups(zmm3, ptr[reg_C_cursor + 3 * 64]);

                    // Row 1
                    // Load ldc
                    mov(rax, ptr[rsp + 8]);              // params
                    mov(reg_tmp.cvt32(), ptr[rax + 48]); // ldc
                    shl(reg_tmp, 2);                     // ldc * 4 bytes

                    vmovups(zmm4, ptr[reg_C_cursor + reg_tmp + 0 * 64]);
                    vmovups(zmm5, ptr[reg_C_cursor + reg_tmp + 1 * 64]);
                    vmovups(zmm6, ptr[reg_C_cursor + reg_tmp + 2 * 64]);
                    vmovups(zmm7, ptr[reg_C_cursor + reg_tmp + 3 * 64]);

                    // Loop over K blocks
                    mov(reg_loop_K, reg_K_blocks);

                    Label loop_K_label;
                    L(loop_K_label);
                    {
                        // 1. Load Comp (4 regs) and init Temp Accumulators
                        vmovups(zmm_b0, ptr[reg_Comp_cursor + 0 * 64]);
                        vmovups(zmm_b1, ptr[reg_Comp_cursor + 1 * 64]);
                        vmovups(zmm_b2, ptr[reg_Comp_cursor + 2 * 64]);
                        vmovups(zmm_b3, ptr[reg_Comp_cursor + 3 * 64]);

                        vpmulld(zmm_b0, zmm_b0, zmm_neg_128);
                        vpmulld(zmm_b1, zmm_b1, zmm_neg_128);
                        vpmulld(zmm_b2, zmm_b2, zmm_neg_128);
                        vpmulld(zmm_b3, zmm_b3, zmm_neg_128);

                        // Init Row 0 Accs
                        vmovdqa64(zmm8, zmm_b0);
                        vmovdqa64(zmm9, zmm_b1);
                        vmovdqa64(zmm10, zmm_b2);
                        vmovdqa64(zmm11, zmm_b3);

                        // Init Row 1 Accs
                        vmovdqa64(zmm12, zmm_b0);
                        vmovdqa64(zmm13, zmm_b1);
                        vmovdqa64(zmm14, zmm_b2);
                        vmovdqa64(zmm15, zmm_b3);

                        // 2. Inner loop over 32 elements (8 steps of 4)
                        for (int i = 0; i < 8; ++i)
                        {
                            // Load A Row 0
                            vpbroadcastd(zmm_a0, ptr[reg_A_cursor + 4 + i * 4]);
                            vpxord(zmm_a0, zmm_a0, zmm_128); // Convert to uint8

                            // Load A Row 1
                            // Offset: A_stride
                            mov(reg_tmp, ptr[rsp + 8]);              // Load params ptr
                            mov(reg_tmp.cvt32(), ptr[reg_tmp + 92]); // Load A_stride
                            vpbroadcastd(zmm_a1, ptr[reg_A_cursor + reg_tmp + 4 + i * 4]);
                            vpxord(zmm_a1, zmm_a1, zmm_128);

                            // Load B (4 regs)
                            vmovups(zmm_b0, ptr[reg_B_cursor + 0 * 64]);
                            vmovups(zmm_b1, ptr[reg_B_cursor + 1 * 64]);
                            vmovups(zmm_b2, ptr[reg_B_cursor + 2 * 64]);
                            vmovups(zmm_b3, ptr[reg_B_cursor + 3 * 64]);

                            // Accumulate Row 0
                            vpdpbusd(zmm8, zmm_a0, zmm_b0);
                            vpdpbusd(zmm9, zmm_a0, zmm_b1);
                            vpdpbusd(zmm10, zmm_a0, zmm_b2);
                            vpdpbusd(zmm11, zmm_a0, zmm_b3);

                            // Accumulate Row 1
                            vpdpbusd(zmm12, zmm_a1, zmm_b0);
                            vpdpbusd(zmm13, zmm_a1, zmm_b1);
                            vpdpbusd(zmm14, zmm_a1, zmm_b2);
                            vpdpbusd(zmm15, zmm_a1, zmm_b3);

                            // Prefetch B (4 steps ahead)
                            prefetcht0(ptr[reg_B_cursor + 1024]);
                            // Prefetch A (4 steps ahead)
                            prefetcht0(ptr[reg_A_cursor + 144]);

                            // Advance B cursor
                            add(reg_B_cursor, 256);
                        }

                        // 3. Convert Acc to float
                        vcvtdq2ps(zmm8, zmm8);
                        vcvtdq2ps(zmm9, zmm9);
                        vcvtdq2ps(zmm10, zmm10);
                        vcvtdq2ps(zmm11, zmm11);
                        vcvtdq2ps(zmm12, zmm12);
                        vcvtdq2ps(zmm13, zmm13);
                        vcvtdq2ps(zmm14, zmm14);
                        vcvtdq2ps(zmm15, zmm15);

                        // 4. Load Scale B (4 regs)
                        vmovups(zmm_scale_b0, ptr[reg_Scales_cursor + 0 * 64]);
                        vmovups(zmm_scale_b1, ptr[reg_Scales_cursor + 1 * 64]);
                        vmovups(zmm_scale_b2, ptr[reg_Scales_cursor + 2 * 64]);
                        vmovups(zmm_scale_b3, ptr[reg_Scales_cursor + 3 * 64]);

                        // 5. Load Scale A Row 0 and multiply
                        vpbroadcastw(ymm_tmp, ptr[reg_A_cursor]);
                        vcvtph2ps(zmm_scale, ymm_tmp);

                        vmulps(zmm8, zmm8, zmm_scale_b0);
                        vmulps(zmm9, zmm9, zmm_scale_b1);
                        vmulps(zmm10, zmm10, zmm_scale_b2);
                        vmulps(zmm11, zmm11, zmm_scale_b3);

                        vmulps(zmm8, zmm8, zmm_scale);
                        vmulps(zmm9, zmm9, zmm_scale);
                        vmulps(zmm10, zmm10, zmm_scale);
                        vmulps(zmm11, zmm11, zmm_scale);

                        // Add to C Acc Row 0
                        vaddps(zmm0, zmm0, zmm8);
                        vaddps(zmm1, zmm1, zmm9);
                        vaddps(zmm2, zmm2, zmm10);
                        vaddps(zmm3, zmm3, zmm11);

                        // 6. Load Scale A Row 1 and multiply
                        mov(reg_tmp, ptr[rsp + 8]);              // Load params ptr
                        mov(reg_tmp.cvt32(), ptr[reg_tmp + 92]); // Load A_stride
                        vpbroadcastw(ymm_tmp, ptr[reg_A_cursor + reg_tmp]);
                        vcvtph2ps(zmm_scale, ymm_tmp);

                        vmulps(zmm12, zmm12, zmm_scale_b0);
                        vmulps(zmm13, zmm13, zmm_scale_b1);
                        vmulps(zmm14, zmm14, zmm_scale_b2);
                        vmulps(zmm15, zmm15, zmm_scale_b3);

                        vmulps(zmm12, zmm12, zmm_scale);
                        vmulps(zmm13, zmm13, zmm_scale);
                        vmulps(zmm14, zmm14, zmm_scale);
                        vmulps(zmm15, zmm15, zmm_scale);

                        // Add to C Acc Row 1
                        vaddps(zmm4, zmm4, zmm12);
                        vaddps(zmm5, zmm5, zmm13);
                        vaddps(zmm6, zmm6, zmm14);
                        vaddps(zmm7, zmm7, zmm15);

                        // Advance pointers
                        add(reg_A_cursor, 36); // sizeof(Q8_1Block)
                        add(reg_Comp_cursor, reg_stride);
                        add(reg_Scales_cursor, reg_stride);

                        dec(reg_loop_K);
                        jnz(loop_K_label, T_NEAR);
                    }

                    // --- Bias & Mask ---
                    mov(rax, ptr[rsp + 8]); // params

                    // 1. Bias
                    mov(rdx, ptr[rax + 56]); // bias ptr
                    test(rdx, rdx);
                    Label skip_bias;
                    jz(skip_bias, T_NEAR);
                    {
                        lea(rdx, ptr[rdx + reg_loop_N * 4]);

                        vmovups(zmm_bias, ptr[rdx + 0 * 64]);
                        vaddps(zmm0, zmm0, zmm_bias);
                        vaddps(zmm4, zmm4, zmm_bias); // Row 1

                        vmovups(zmm_bias, ptr[rdx + 1 * 64]);
                        vaddps(zmm1, zmm1, zmm_bias);
                        vaddps(zmm5, zmm5, zmm_bias);

                        vmovups(zmm_bias, ptr[rdx + 2 * 64]);
                        vaddps(zmm2, zmm2, zmm_bias);
                        vaddps(zmm6, zmm6, zmm_bias);

                        vmovups(zmm_bias, ptr[rdx + 3 * 64]);
                        vaddps(zmm3, zmm3, zmm_bias);
                        vaddps(zmm7, zmm7, zmm_bias);
                    }
                    L(skip_bias);

                    // 2. Mask
                    mov(rdx, ptr[rax + 64]); // mask ptr
                    test(rdx, rdx);
                    Label skip_mask;
                    jz(skip_mask, T_NEAR);
                    {
                        // Row 0 mask
                        lea(rcx, ptr[rdx + reg_loop_N * 4]);

                        vmovups(zmm_mask, ptr[rcx + 0 * 64]);
                        vaddps(zmm0, zmm0, zmm_mask);

                        vmovups(zmm_mask, ptr[rcx + 1 * 64]);
                        vaddps(zmm1, zmm1, zmm_mask);

                        vmovups(zmm_mask, ptr[rcx + 2 * 64]);
                        vaddps(zmm2, zmm2, zmm_mask);

                        vmovups(zmm_mask, ptr[rcx + 3 * 64]);
                        vaddps(zmm3, zmm3, zmm_mask);

                        // Row 1 mask
                        // Offset = ldc * 4 (stride)
                        mov(rsi, ptr[rax + 48]); // ldc
                        shl(rsi, 2);
                        add(rcx, rsi); // rcx points to Row 1 mask

                        vmovups(zmm_mask, ptr[rcx + 0 * 64]);
                        vaddps(zmm4, zmm4, zmm_mask);

                        vmovups(zmm_mask, ptr[rcx + 1 * 64]);
                        vaddps(zmm5, zmm5, zmm_mask);

                        vmovups(zmm_mask, ptr[rcx + 2 * 64]);
                        vaddps(zmm6, zmm6, zmm_mask);

                        vmovups(zmm_mask, ptr[rcx + 3 * 64]);
                        vaddps(zmm7, zmm7, zmm_mask);
                    }
                    L(skip_mask);

                    // 3. Softmax
                    mov(al, ptr[rax + 88]); // do_softmax
                    test(al, al);
                    Label skip_softmax;
                    jz(skip_softmax, T_NEAR);
                    {
                        // Registers
                        const Zmm &zmm_max = zmm22;
                        const Zmm &zmm_sum = zmm23;
                        const Zmm &zmm_tmp = zmm24;
                        const Zmm &zmm_c1_const = zmm25;
                        const Zmm &zmm_c2_const = zmm26;
                        const Zmm &zmm_c3_coeff = zmm27;
                        const Zmm &zmm_c4_const = zmm28;
                        const Zmm &zmm_c5_const = zmm29;
                        const Zmm &zmm_inv_ln2 = zmm30;
                        const Zmm &zmm_max_clip = zmm31;
                        const Zmm &zmm_min_clip = zmm8;
                        const Zmm &zmm_127 = zmm9;
                        const Zmm &zmm_tmp2 = zmm10;
                        const Zmm &zmm_p = zmm11;
                        const Zmm &zmm_fx = zmm12;

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

                        auto compute_exp_accumulate = [&](const Zmm &zmm_val)
                        {
                            vsubps(zmm_tmp, zmm_val, zmm_max);
                            vminps(zmm_tmp, zmm_tmp, zmm_max_clip);
                            vmaxps(zmm_tmp, zmm_tmp, zmm_min_clip);
                            vmulps(zmm_tmp, zmm_tmp, zmm_inv_ln2);
                            vrndscaleps(zmm_fx, zmm_tmp, 0x01);
                            vsubps(zmm_tmp, zmm_tmp, zmm_fx);
                            vmovaps(zmm_p, zmm_c5_const);
                            vfmadd213ps(zmm_p, zmm_tmp, zmm_c4_const);
                            vfmadd213ps(zmm_p, zmm_tmp, zmm_c3_coeff);
                            vfmadd213ps(zmm_p, zmm_tmp, zmm_c2_const);
                            vfmadd213ps(zmm_p, zmm_tmp, zmm_c1_const);
                            vcvtps2dq(zmm_fx, zmm_fx);
                            vpaddd(zmm_fx, zmm_fx, zmm_127);
                            vpslld(zmm_fx, zmm_fx, 23);
                            vmulps(zmm_p, zmm_p, zmm_fx);
                            vaddps(zmm_sum, zmm_sum, zmm_p);
                        };

                        // --- Row 0 ---
                        vmaxps(zmm_max, zmm0, zmm1);
                        vmaxps(zmm_max, zmm_max, zmm2);
                        vmaxps(zmm_max, zmm_max, zmm3);

                        vextractf64x4(ymm10, zmm_max, 1);
                        vmaxps(ymm10, ymm10, Ymm(zmm_max.getIdx()));
                        vextractf128(xmm11, ymm10, 1);
                        vmaxps(xmm10, xmm10, xmm11);
                        vpermilps(xmm11, xmm10, 0b01001110);
                        vmaxps(xmm10, xmm10, xmm11);
                        vpermilps(xmm11, xmm10, 0b10110001);
                        vmaxps(xmm10, xmm10, xmm11);

                        vpbroadcastd(zmm_max, xmm10);

                        mov(rdx, ptr[rax + 72]); // local_max ptr
                        mov(r15, reg_loop_N);
                        shr(r15, 4); // / 16
                        vmovss(ptr[rdx + r15], xmm10);

                        vpxord(zmm_sum, zmm_sum, zmm_sum);
                        compute_exp_accumulate(zmm0);
                        compute_exp_accumulate(zmm1);
                        compute_exp_accumulate(zmm2);
                        compute_exp_accumulate(zmm3);

                        vextractf64x4(ymm10, zmm_sum, 1);
                        vaddps(ymm10, ymm10, Ymm(zmm_sum.getIdx()));
                        vextractf128(xmm11, ymm10, 1);
                        vaddps(xmm10, xmm10, xmm11);
                        vpermilps(xmm11, xmm10, 0b01001110);
                        vaddps(xmm10, xmm10, xmm11);
                        vpermilps(xmm11, xmm10, 0b10110001);
                        vaddps(xmm10, xmm10, xmm11);

                        mov(rdx, ptr[rax + 80]); // local_sum ptr
                        vmovss(ptr[rdx + r15], xmm10);

                        // --- Row 1 ---
                        vmaxps(zmm_max, zmm4, zmm5);
                        vmaxps(zmm_max, zmm_max, zmm6);
                        vmaxps(zmm_max, zmm_max, zmm7);

                        vextractf64x4(ymm10, zmm_max, 1);
                        vmaxps(ymm10, ymm10, Ymm(zmm_max.getIdx()));
                        vextractf128(xmm11, ymm10, 1);
                        vmaxps(xmm10, xmm10, xmm11);
                        vpermilps(xmm11, xmm10, 0b01001110);
                        vmaxps(xmm10, xmm10, xmm11);
                        vpermilps(xmm11, xmm10, 0b10110001);
                        vmaxps(xmm10, xmm10, xmm11);

                        vpbroadcastd(zmm_max, xmm10);

                        mov(rdx.cvt32(), ptr[rax + 44]); // N (32-bit load)
                        shr(rdx, 4);                     // N / 16
                        add(rdx, r15);                   // offset_row1

                        mov(rcx, ptr[rax + 72]); // local_max ptr
                        vmovss(ptr[rcx + rdx], xmm10);

                        vpxord(zmm_sum, zmm_sum, zmm_sum);
                        compute_exp_accumulate(zmm4);
                        compute_exp_accumulate(zmm5);
                        compute_exp_accumulate(zmm6);
                        compute_exp_accumulate(zmm7);

                        vextractf64x4(ymm10, zmm_sum, 1);
                        vaddps(ymm10, ymm10, Ymm(zmm_sum.getIdx()));
                        vextractf128(xmm11, ymm10, 1);
                        vaddps(xmm10, xmm10, xmm11);
                        vpermilps(xmm11, xmm10, 0b01001110);
                        vaddps(xmm10, xmm10, xmm11);
                        vpermilps(xmm11, xmm10, 0b10110001);
                        vaddps(xmm10, xmm10, xmm11);

                        mov(rcx, ptr[rax + 80]); // local_sum ptr
                        vmovss(ptr[rcx + rdx], xmm10);
                    }
                    L(skip_softmax);

                    // Store C
                    // Load ldc
                    mov(rax, ptr[rsp + 8]);              // params
                    mov(reg_tmp.cvt32(), ptr[rax + 48]); // ldc
                    shl(reg_tmp, 2);                     // ldc * 4 bytes

                    // Row 0
                    vmovups(ptr[reg_C_cursor + 0 * 64], zmm0);
                    vmovups(ptr[reg_C_cursor + 1 * 64], zmm1);
                    vmovups(ptr[reg_C_cursor + 2 * 64], zmm2);
                    vmovups(ptr[reg_C_cursor + 3 * 64], zmm3);

                    // Row 1
                    add(reg_C_cursor, reg_tmp); // + ldc
                    vmovups(ptr[reg_C_cursor + 0 * 64], zmm4);
                    vmovups(ptr[reg_C_cursor + 1 * 64], zmm5);
                    vmovups(ptr[reg_C_cursor + 2 * 64], zmm6);
                    vmovups(ptr[reg_C_cursor + 3 * 64], zmm7);

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
