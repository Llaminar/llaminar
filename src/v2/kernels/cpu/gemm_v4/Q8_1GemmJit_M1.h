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

        struct Q8_1PackedWeights
        {
            // Packed data: [K/4][N][4] (int8_t)
            // But we flatten it to [K/4 * N * 4]
            std::vector<int8_t> packed_data;

            // Compensation: [K/32][N] (int32_t)
            std::vector<int32_t> compensation;

            // Scales: [K/32][N] (float)
            std::vector<float> scales;

            int K;
            int N;
        };

        class Q8_1GemmJit_M1 : public Xbyak::CodeGenerator
        {
        public:
            Q8_1GemmJit_M1() : Xbyak::CodeGenerator(4096 * 32)
            { // Allocate enough space
                generate();
            }

            // Signature:
            // void kernel(const Q8_1Block* A, const int8_t* B_packed, const int32_t* comp, const float* scales, float* C, int K_blocks, int N, int ldc);
            using kernel_func_t = void (*)(const void *A, const void *B_packed, const void *comp, const void *scales, float *C, int K_blocks, int N, int ldc);

            kernel_func_t get_kernel()
            {
                return getCode<kernel_func_t>();
            }

            static void pack_weights(const int8_t *B, int K, int N, Q8_1PackedWeights &packed)
            {
                // Dummy implementation for now, real one in Q8_1GemmKernel
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

                // Prologue
                push(rbx);
                push(rbp);
                push(r12);
                push(r13);
                push(r14);
                push(r15);

                // Stack offset for N: 6 pushes * 8 = 48. + 8 (ret addr) = 56.
                // But wait, arguments 7+ are at rsp + 8 + (6*8) = rsp + 56.
                // Arg 7 is N.
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
                // Load ldc from stack
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

                    // Initialize C accumulators
                    vpxord(zmm_c0, zmm_c0, zmm_c0);
                    vpxord(zmm_c1, zmm_c1, zmm_c1);
                    vpxord(zmm_c2, zmm_c2, zmm_c2);
                    vpxord(zmm_c3, zmm_c3, zmm_c3);

                    // Loop over K blocks
                    mov(reg_loop_K, reg_K_blocks);

                    Label loop_K_label;
                    L(loop_K_label);
                    {
                        // Load scale A (half) -> float broadcast
                        vpbroadcastw(ymm_tmp, ptr[reg_A_cursor]);
                        vcvtph2ps(zmm_scale, ymm_tmp);

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
