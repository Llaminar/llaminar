#pragma once

#include "../../../../../external/onednn/third_party/xbyak/xbyak.h"
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
            // void kernel(const void* A, const void* B_packed, const void* comp, const void* scales, float* C, int K_blocks, int N, int ldc);
            using kernel_func_t = void (*)(const void *A, const void *B_packed, const void *comp, const void *scales, float *C, int K_blocks, int N, int ldc);

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

                // Prologue
                push(rbx);
                push(rbp);
                push(r12);
                push(r13);
                push(r14);
                push(r15);

                int stack_offset_N = 56;
                int stack_offset_ldc = 64;

                // Setup constants
                mov(reg_tmp_32, 0x80808080);
                vpbroadcastd(zmm_128, reg_tmp_32);

                mov(reg_tmp_32, -128);
                vpbroadcastd(zmm_neg_128, reg_tmp_32);

                // Loop over N (step 64)
                xor_(reg_loop_N, reg_loop_N);

                // Precompute stride = ldc * 4
                mov(reg_stride, ptr[rsp + stack_offset_ldc]);
                shl(reg_stride, 2); // ldc * 4

                Label loop_N_label;
                L(loop_N_label);
                {
                    // Check if we are done
                    cmp(reg_loop_N, ptr[rsp + stack_offset_N]);
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

                    // Initialize C Accumulators (zmm0..7) to 0
                    for (int i = 0; i < 8; ++i)
                    {
                        vpxord(Zmm(i), Zmm(i), Zmm(i));
                    }

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
                            // Offset: K_blocks * 36
                            mov(reg_tmp, reg_K_blocks);
                            imul(reg_tmp, reg_tmp, 36);
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
                        mov(reg_tmp, reg_K_blocks);
                        imul(reg_tmp, reg_tmp, 36);
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

                    // Store C
                    // Load ldc
                    mov(reg_tmp, ptr[rsp + stack_offset_ldc]);
                    shl(reg_tmp, 2); // ldc * 4 bytes

                    // Row 0
                    vaddps(zmm0, zmm0, ptr[reg_C_cursor + 0 * 64]);
                    vaddps(zmm1, zmm1, ptr[reg_C_cursor + 1 * 64]);
                    vaddps(zmm2, zmm2, ptr[reg_C_cursor + 2 * 64]);
                    vaddps(zmm3, zmm3, ptr[reg_C_cursor + 3 * 64]);
                    vmovups(ptr[reg_C_cursor + 0 * 64], zmm0);
                    vmovups(ptr[reg_C_cursor + 1 * 64], zmm1);
                    vmovups(ptr[reg_C_cursor + 2 * 64], zmm2);
                    vmovups(ptr[reg_C_cursor + 3 * 64], zmm3);

                    // Row 1
                    add(reg_C_cursor, reg_tmp); // + ldc
                    vaddps(zmm4, zmm4, ptr[reg_C_cursor + 0 * 64]);
                    vaddps(zmm5, zmm5, ptr[reg_C_cursor + 1 * 64]);
                    vaddps(zmm6, zmm6, ptr[reg_C_cursor + 2 * 64]);
                    vaddps(zmm7, zmm7, ptr[reg_C_cursor + 3 * 64]);
                    vmovups(ptr[reg_C_cursor + 0 * 64], zmm4);
                    vmovups(ptr[reg_C_cursor + 1 * 64], zmm5);
                    vmovups(ptr[reg_C_cursor + 2 * 64], zmm6);
                    vmovups(ptr[reg_C_cursor + 3 * 64], zmm7);

                    // Advance N loop
                    add(reg_loop_N, 64);
                    jmp(loop_N_label, T_NEAR);
                }

                L("end_kernel");
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
