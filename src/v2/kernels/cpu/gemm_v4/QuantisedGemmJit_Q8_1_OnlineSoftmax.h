/**
 * @file QuantisedGemmJit_Q8_1_OnlineSoftmax.h
 * @brief JIT-compiled AVX-512 VNNI GEMM kernel for Q8_1 x Q8_1 matrix multiplication with Online Softmax.
 * @author David Sanftenberg
 */

#pragma once

#include "../../../../../external/onednn/third_party/xbyak/xbyak.h"
#include "../../../tensors/BlockStructures.h"
#include <cstdint>
#include <cmath>

namespace llaminar2
{
    namespace gemm_v4
    {

        struct OnlineSoftmaxParams
        {
            const void *Q;         // Pointer to Q start (Q8_1 blocks)
            const void *K;         // Pointer to K start (Q8_1 blocks)
            float *C;              // Pointer to C start (scores)
            int M;                 // Number of Q rows
            int N;                 // Number of K rows (kv_len)
            int K_blocks;          // Number of blocks in head_dim
            int Q_stride_bytes;    // Stride between Q rows
            int K_stride_bytes;    // Stride between K rows
            int C_stride_bytes;    // Stride between C rows (usually N * sizeof(float))
            float scale;           // Attention scale
            const float *mask;     // Mask pointer
            int mask_stride_bytes; // Stride between mask rows
        };

        /**
         * @class QuantisedGemmJit_Q8_1_OnlineSoftmax
         * @brief JIT kernel for Q8_1 matrix * Q8_1 matrix (Q * K^T) with Online Softmax
         *
         * Computes dot product of M Q rows against N K rows.
         * Updates running Max and Sum for Online Softmax.
         * Stores raw scores to C (to be normalized in a second pass).
         *
         * Supports M-blocking for high arithmetic intensity (K-reuse).
         */
        class QuantisedGemmJit_Q8_1_OnlineSoftmax : public Xbyak::CodeGenerator
        {
        public:
            QuantisedGemmJit_Q8_1_OnlineSoftmax(int m_blocking = 1)
                : Xbyak::CodeGenerator(4096 * 32), m_blocking_(m_blocking)
            {
                generate();
            }

            using kernel_func_t = void (*)(const OnlineSoftmaxParams *params);

            kernel_func_t get_kernel()
            {
                return getCode<kernel_func_t>();
            }

        private:
            int m_blocking_;

            void generate()
            {
                using namespace Xbyak;

                // Registers
                const Reg64 &reg_params = rdi;
                const Reg64 &reg_Q_base = rsi;
                const Reg64 &reg_K_base = rdx;
                const Reg64 &reg_C_base = rcx;
                const Reg64 &reg_N = r8;
                const Reg64 &reg_K_blocks = r9;

                const Reg64 &reg_Q_stride = r10;
                const Reg64 &reg_K_stride = r11;
                const Reg64 &reg_C_stride = r12;
                const Reg64 &reg_mask_base = r13;
                const Reg64 &reg_mask_stride = r14;

                const Reg64 &reg_n = r15;
                const Reg64 &reg_k = rax;
                const Reg64 &reg_tmp = rbx;
                const Reg64 &reg_Q_cursor = rbp;

                // YMM Allocation (using YMM to match 8-lane VNNI output)
                // We use m_blocking_ rows of Q.
                // We use unroll_n_ columns of K.
                // IMPORTANT: Correction accumulators start at Xmm(4), so for m_blocking_ * unroll_n_
                // we need indices 4..4+m*u-1. These must NOT collide with Q data at Ymm(8+i).
                // For m=4, max safe unroll_n is 1 (4*1=4 corrections at Xmm4-7, Q at Ymm8-11 OK)
                // For m=1, unroll_n=2 is OK (1*2=2 corrections at Xmm4-5, Q at Ymm8 OK)
                int unroll_n_ = (m_blocking_ > 1) ? 1 : 2;

                // Vectorized Correction Registers
                // We use XMM registers starting after the main accumulators to store vectorized corrections.
                // VecAccCorr: Stores accumulated corrections for each K column (vectorized over M rows).
                // VecDQ: Stores packed d_Q values for M rows.
                // Base index: m_blocking_ * unroll_n_ (start of free regs after Acc)
                // We need ceil(m/4) registers for VecDQ and ceil(m/4) * unroll_n_ for VecAccCorr.
                // For m=4, n=1: Acc=0-3. VecAccCorr=4. VecDQ=5. Q=8. Safe.
                // For m=1, n=2: Acc=0-1. VecAccCorr=2,3. VecDQ=4. Q=8. Safe.
                int vec_acc_corr_base = m_blocking_ * unroll_n_;
                int vec_dq_base = vec_acc_corr_base + unroll_n_ * ((m_blocking_ + 3) / 4);

                // Accumulators: m_blocking_ * unroll_n_ (Ymm0-7)
                // Q values: m_blocking_ regs (Ymm8-11)
                // K values: unroll_n_ regs (Ymm12-13)

                // Max/Sum: m_blocking_ * 2 regs (Ymm14-21) - storing scalars broadcasted

                // Constants/Scales: Ymm22-31

                const Ymm ymm_128 = ymm22;
                const Ymm ymm_scale = ymm23;
                const Ymm ymm_one = ymm24; // 1.0f
                const Ymm ymm_neg_inf = ymm25;
                const Ymm ymm_128f = ymm30; // 128.0f for Q8_1 unsigned correction

                // Prologue
                push(rbx);
                push(rbp);
                push(r12);
                push(r13);
                push(r14);
                push(r15);

                // Load params
                mov(reg_Q_base, ptr[reg_params + 0]);
                mov(reg_K_base, ptr[reg_params + 8]);
                mov(reg_C_base, ptr[reg_params + 16]);
                mov(reg_N.cvt32(), ptr[reg_params + 28]);
                mov(reg_K_blocks.cvt32(), ptr[reg_params + 32]);
                mov(reg_Q_stride.cvt32(), ptr[reg_params + 36]);
                mov(reg_K_stride.cvt32(), ptr[reg_params + 40]);
                mov(reg_C_stride.cvt32(), ptr[reg_params + 44]);

                vbroadcastss(ymm_scale, ptr[reg_params + 48]);
                mov(reg_mask_base, ptr[reg_params + 56]);
                mov(reg_mask_stride.cvt32(), ptr[reg_params + 64]);

                // Constants
                mov(reg_tmp.cvt32(), 0x80808080);
                vmovd(xmm31, reg_tmp.cvt32());
                vpbroadcastd(ymm_128, xmm31);

                mov(reg_tmp.cvt32(), 0x3F800000); // 1.0f
                vmovd(xmm31, reg_tmp.cvt32());
                vpbroadcastd(ymm_one, xmm31);

                mov(reg_tmp.cvt32(), 0xFF800000); // -inf
                vmovd(xmm31, reg_tmp.cvt32());
                vpbroadcastd(ymm_neg_inf, xmm31);

                mov(reg_tmp.cvt32(), 0x43000000); // 128.0f
                vmovd(xmm31, reg_tmp.cvt32());
                vpbroadcastd(ymm_128f, xmm31);

                // Initialize Max/Sum
                for (int i = 0; i < m_blocking_; ++i)
                {
                    vmovaps(Ymm(14 + i), ymm_neg_inf);                                                       // Max = -inf
                    vxorps(Ymm(14 + m_blocking_ + i), Ymm(14 + m_blocking_ + i), Ymm(14 + m_blocking_ + i)); // Sum = 0
                }

                // ============================================================
                // PASS 1: GEMM + Online Softmax Update
                // ============================================================
                xor_(reg_n, reg_n);

                Label loop_N;
                L(loop_N);
                {
                    cmp(reg_n, reg_N);
                    jge("end_pass1", T_NEAR);

                    // Check tail N
                    mov(reg_tmp, reg_N);
                    sub(reg_tmp, reg_n);
                    cmp(reg_tmp, unroll_n_);
                    jl("tail_N", T_NEAR);

                    // Clear Accumulators (Ymm0..m_blocking_*unroll_n_-1 for main accumulator)
                    for (int i = 0; i < m_blocking_ * unroll_n_; ++i)
                    {
                        vxorps(Ymm(i), Ymm(i), Ymm(i));
                    }
                    // Clear Vector Correction Accumulators
                    for (int j = 0; j < unroll_n_; ++j)
                    {
                        for (int ii = 0; ii < m_blocking_; ii += 4)
                        {
                            int reg_idx = vec_acc_corr_base + j * ((m_blocking_ + 3) / 4) + (ii / 4);
                            vxorps(Xmm(reg_idx), Xmm(reg_idx), Xmm(reg_idx));
                        }
                    }

                    // Inner Loop K
                    xor_(reg_k, reg_k);
                    Label loop_K;
                    L(loop_K);
                    {
                        // Load Q rows
                        for (int i = 0; i < m_blocking_; ++i)
                        {
                            // Q_ptr = Q_base + i * Q_stride + k * 36
                            mov(reg_tmp, reg_Q_stride);
                            imul(reg_tmp, reg_tmp, i);
                            add(reg_tmp, reg_Q_base);

                            mov(reg_Q_cursor, reg_k);
                            imul(reg_Q_cursor, reg_Q_cursor, 36);
                            add(reg_Q_cursor, reg_tmp);

                            vmovdqu8(Ymm(8 + i), ptr[reg_Q_cursor + 4]); // Load Q8_1 (32 bytes)
                            vpxord(Ymm(8 + i), Ymm(8 + i), ymm_128);     // Unsigned
                        }

                        // Construct VecDQ (Packed d_Q values)
                        for (int ii = 0; ii < m_blocking_; ii += 4)
                        {
                            int reg_idx = vec_dq_base + (ii / 4);
                            // No need to clear if we fill all 4, but safe to clear
                            vxorps(Xmm(reg_idx), Xmm(reg_idx), Xmm(reg_idx));

                            for (int r = 0; r < 4 && (ii + r) < m_blocking_; ++r)
                            {
                                int i = ii + r;
                                // Load d_Q[i] address
                                mov(reg_tmp, reg_Q_stride);
                                imul(reg_tmp, reg_tmp, i);
                                add(reg_tmp, reg_Q_base);
                                mov(reg_Q_cursor, reg_k);
                                imul(reg_Q_cursor, reg_Q_cursor, 36);
                                add(reg_Q_cursor, reg_tmp);

                                // Insert word
                                vpinsrw(Xmm(reg_idx), Xmm(reg_idx), ptr[reg_Q_cursor], r);
                            }
                            vcvtph2ps(Xmm(reg_idx), Xmm(reg_idx)); // Convert 4 halfs to 4 floats
                        }

                        // Load K cols
                        for (int j = 0; j < unroll_n_; ++j)
                        {
                            // K_ptr = K_base + (n+j) * K_stride + k * 36
                            mov(reg_tmp, reg_K_stride);
                            mov(reg_Q_cursor, reg_n);
                            add(reg_Q_cursor, j);
                            imul(reg_tmp, reg_Q_cursor);
                            add(reg_tmp, reg_K_base);

                            mov(reg_Q_cursor, reg_k);
                            imul(reg_Q_cursor, reg_Q_cursor, 36);
                            add(reg_tmp, reg_Q_cursor);

                            vmovdqu8(Ymm(12 + j), ptr[reg_tmp + 4]); // Load K8_1 (32 bytes)
                        }

                        // Compute Dot Products and Corrections
                        // Loop over K columns (j)
                        for (int j = 0; j < unroll_n_; ++j)
                        {
                            // Address for K block metadata
                            mov(reg_tmp, reg_K_stride);
                            mov(reg_Q_cursor, reg_n);
                            add(reg_Q_cursor, j);
                            imul(reg_tmp, reg_Q_cursor);
                            add(reg_tmp, reg_K_base);
                            mov(reg_Q_cursor, reg_k);
                            imul(reg_Q_cursor, reg_Q_cursor, 36);
                            add(reg_tmp, reg_Q_cursor);

                            // Load d_K (FP16) -> Float -> Broadcast
                            vpbroadcastw(xmm31, ptr[reg_tmp]);
                            vcvtph2ps(Ymm(27), xmm31); // Ymm27 holds d_K broadcasted

                            // Load S_K (INT16) -> Float -> Broadcast
                            vpbroadcastw(xmm31, ptr[reg_tmp + 2]);
                            vpmovsxwd(xmm29, xmm31);
                            vcvtdq2ps(xmm29, xmm29); // xmm29 holds S_K broadcasted

                            // Compute T_K = d_K * S_K
                            vmulps(xmm29, xmm29, Xmm(27)); // xmm29 = T_K broadcasted

                            // Accumulate Correction: VecAccCorr[j] += VecDQ * T_K
                            for (int ii = 0; ii < m_blocking_; ii += 4)
                            {
                                int acc_idx = vec_acc_corr_base + j * ((m_blocking_ + 3) / 4) + (ii / 4);
                                int dq_idx = vec_dq_base + (ii / 4);
                                vfmadd231ps(Xmm(acc_idx), Xmm(dq_idx), xmm29);
                            }

                            // Dot Products for all M rows
                            for (int i = 0; i < m_blocking_; ++i)
                            {
                                // Load d_Q[i] (reload for scaling)
                                mov(reg_tmp, reg_Q_stride);
                                imul(reg_tmp, reg_tmp, i);
                                add(reg_tmp, reg_Q_base);
                                mov(reg_Q_cursor, reg_k);
                                imul(reg_Q_cursor, reg_Q_cursor, 36);
                                add(reg_Q_cursor, reg_tmp);

                                vpbroadcastw(xmm31, ptr[reg_Q_cursor]);
                                vcvtph2ps(Ymm(26), xmm31); // Ymm26 = d_Q[i] broadcasted

                                // Dot product
                                vxorps(Ymm(28), Ymm(28), Ymm(28));
                                vpdpbusd(Ymm(28), Ymm(8 + i), Ymm(12 + j)); // 8 int32 results
                                vcvtdq2ps(Ymm(28), Ymm(28));                // 8 floats

                                // Apply scale: d_Q * d_K
                                vmulps(Ymm(28), Ymm(28), Ymm(26)); // * d_Q
                                vmulps(Ymm(28), Ymm(28), Ymm(27)); // * d_K (already in Ymm27)

                                // Accumulate
                                vaddps(Ymm(i * unroll_n_ + j), Ymm(i * unroll_n_ + j), Ymm(28));
                            }
                        }

                        inc(reg_k);
                        cmp(reg_k, reg_K_blocks);
                        jl(loop_K, T_NEAR);
                    }

                    // Horizontal Sum and Update
                    for (int i = 0; i < m_blocking_; ++i)
                    {
                        for (int j = 0; j < unroll_n_; ++j)
                        {
                            Ymm ymm_acc = Ymm(i * unroll_n_ + j);

                            // HSum YMM (8 floats) -> Scalar XMM
                            vextractf32x4(xmm28, ymm_acc, 1);
                            vaddps(xmm28, xmm28, Xmm(ymm_acc.getIdx()));
                            vpermilps(xmm29, xmm28, 0b01001110); // Swap pairs
                            vaddps(xmm28, xmm28, xmm29);
                            vpermilps(xmm29, xmm28, 0b10110001); // Swap odd/even
                            vaddps(xmm28, xmm28, xmm29);         // xmm28[0] = raw dot product (with unsigned bias)

                            // Apply correction: subtract 128 × Σ_k(sum_qs_K[k] × d_Q[k] × d_K[k])
                            // Extract correction from VecAccCorr
                            int acc_reg_idx = vec_acc_corr_base + j * ((m_blocking_ + 3) / 4) + (i / 4);
                            int elem_idx = i % 4;

                            if (elem_idx == 0)
                            {
                                vmovss(xmm29, Xmm(acc_reg_idx));
                            }
                            else
                            {
                                vpermilps(xmm29, Xmm(acc_reg_idx), elem_idx);
                            }

                            // NOTE: Must reload 128.0f because softmax code below clobbers ymm30
                            mov(eax, 0x43000000);        // 128.0f in IEEE754
                            vmovd(xmm30, eax);           // Use xmm30 for 128.0f
                            vmulss(xmm29, xmm29, xmm30); // 128 * correction
                            vsubss(xmm28, xmm28, xmm29); // Subtract correction

                            // Apply Attention Scale

                            vmulss(xmm28, xmm28, Xmm(ymm_scale.getIdx()));

                            // Apply Mask
                            /*
                            Label skip_mask_scalar;
                            cmp(reg_mask_base, 0);
                            je(skip_mask_scalar, T_NEAR);

                            mov(reg_tmp, reg_mask_stride);
                            imul(reg_tmp, reg_tmp, i);
                            add(reg_tmp, reg_mask_base);

                            mov(reg_Q_cursor, reg_n);
                            add(reg_Q_cursor, j);
                            shl(reg_Q_cursor, 2);
                            add(reg_tmp, reg_Q_cursor);

                            vaddss(xmm28, xmm28, ptr[reg_tmp]);
                            L(skip_mask_scalar);
                            */

                            // Store Raw Score
                            mov(reg_tmp, reg_C_stride);
                            imul(reg_tmp, reg_tmp, i);
                            add(reg_tmp, reg_C_base);

                            mov(reg_Q_cursor, reg_n);
                            add(reg_Q_cursor, j);
                            shl(reg_Q_cursor, 2);
                            add(reg_tmp, reg_Q_cursor);

                            vmovss(ptr[reg_tmp], xmm28);

                            // Online Softmax Update
                            // m_local = xmm28[0]

                            // m_new = max(m_global, m_local)
                            vmovss(xmm31, Xmm(14 + i));  // Max[i]
                            vmaxss(xmm30, xmm31, xmm28); // m_new

                            // Save m_new and score because exp_ss clobbers xmm28-xmm31
                            vmovss(xmm27, xmm30); // Save m_new
                            vmovss(xmm26, xmm28); // Save score

                            // correction = exp(m_global - m_new)
                            vsubss(xmm29, xmm31, xmm30);
                            exp_ss(xmm29, xmm29);
                            vmulss(Xmm(14 + m_blocking_ + i), Xmm(14 + m_blocking_ + i), xmm29); // Sum[i] *= correction

                            // exp_val = exp(score - m_new)
                            vsubss(xmm28, xmm26, xmm27); // Restore score and m_new
                            exp_ss(xmm28, xmm28);

                            vaddss(Xmm(14 + m_blocking_ + i), Xmm(14 + m_blocking_ + i), xmm28); // Sum[i] += exp_val
                            vmovss(Xmm(14 + i), xmm27);                                          // Max[i] = m_new
                        }
                    }

                    add(reg_n, unroll_n_);
                    jmp(loop_N, T_NEAR);
                }

                L("tail_N");
                // Simple scalar tail loop
                Label loop_tail;
                L(loop_tail);
                {
                    cmp(reg_n, reg_N);
                    jge("end_pass1", T_NEAR);

                    // Clear Accumulators (one per M row)
                    for (int i = 0; i < m_blocking_; ++i)
                    {
                        vxorps(Ymm(i), Ymm(i), Ymm(i));
                    }
                    // Clear scalar correction accumulators (one per M row)
                    for (int i = 0; i < m_blocking_; ++i)
                    {
                        vxorps(Xmm(4 + i), Xmm(4 + i), Xmm(4 + i));
                    }

                    // Inner Loop K
                    xor_(reg_k, reg_k);
                    Label loop_K_tail;
                    L(loop_K_tail);
                    {
                        // Process all M rows for single N column

                        for (int i = 0; i < m_blocking_; ++i)
                        {
                            // Load Q
                            mov(reg_tmp, reg_Q_stride);
                            imul(reg_tmp, reg_tmp, i);
                            add(reg_tmp, reg_Q_base);
                            mov(reg_Q_cursor, reg_k);
                            imul(reg_Q_cursor, reg_Q_cursor, 36);
                            add(reg_Q_cursor, reg_tmp);
                            vmovdqu8(Ymm(8 + i), ptr[reg_Q_cursor + 4]);
                            vpxord(Ymm(8 + i), Ymm(8 + i), ymm_128);

                            // Load K (col n)
                            mov(reg_tmp, reg_K_stride);
                            imul(reg_tmp, reg_n);
                            add(reg_tmp, reg_K_base);
                            mov(reg_Q_cursor, reg_k);
                            imul(reg_Q_cursor, reg_Q_cursor, 36);
                            add(reg_tmp, reg_Q_cursor);
                            vmovdqu8(Ymm(12), ptr[reg_tmp + 4]);

                            // Dot
                            vxorps(Ymm(28), Ymm(28), Ymm(28));
                            vpdpbusd(Ymm(28), Ymm(8 + i), Ymm(12));
                            vcvtdq2ps(Ymm(28), Ymm(28));

                            // Scales
                            // Q scale
                            mov(reg_tmp, reg_Q_stride);
                            imul(reg_tmp, reg_tmp, i);
                            add(reg_tmp, reg_Q_base);
                            mov(reg_Q_cursor, reg_k);
                            imul(reg_Q_cursor, reg_Q_cursor, 36);
                            add(reg_Q_cursor, reg_tmp);
                            vpbroadcastw(xmm31, ptr[reg_Q_cursor]);
                            vcvtph2ps(Ymm(26), xmm31);

                            // K scale
                            mov(reg_tmp, reg_K_stride);
                            imul(reg_tmp, reg_n);
                            add(reg_tmp, reg_K_base);
                            mov(reg_Q_cursor, reg_k);
                            imul(reg_Q_cursor, reg_Q_cursor, 36);
                            add(reg_tmp, reg_Q_cursor);
                            vpbroadcastw(xmm31, ptr[reg_tmp]);
                            vcvtph2ps(Ymm(27), xmm31);

                            // Accumulate correction term: sum_qs_K × d_Q × d_K
                            vpbroadcastw(xmm31, ptr[reg_tmp + 2]);
                            vpmovsxwd(xmm29, xmm31);
                            vcvtdq2ps(xmm29, xmm29);
                            vmulss(xmm29, xmm29, Xmm(26));         // * d_Q
                            vmulss(xmm29, xmm29, Xmm(27));         // * d_K
                            vaddss(Xmm(4 + i), Xmm(4 + i), xmm29); // Accumulate correction per M row

                            vmulps(Ymm(28), Ymm(28), Ymm(26));
                            vmulps(Ymm(28), Ymm(28), Ymm(27));

                            // Accumulate
                            vaddps(Ymm(i), Ymm(i), Ymm(28)); // Use Ymm(i) as acc
                        }

                        inc(reg_k);
                        cmp(reg_k, reg_K_blocks);
                        jl(loop_K_tail, T_NEAR);
                    }

                    // HSum and Update
                    for (int i = 0; i < m_blocking_; ++i)
                    {
                        Ymm ymm_acc = Ymm(i);
                        vextractf32x4(xmm28, ymm_acc, 1);
                        vaddps(xmm28, xmm28, Xmm(ymm_acc.getIdx()));
                        vpermilps(xmm29, xmm28, 0b01001110);
                        vaddps(xmm28, xmm28, xmm29);
                        vpermilps(xmm29, xmm28, 0b10110001);
                        vaddps(xmm28, xmm28, xmm29);

                        // Apply unsigned conversion correction: subtract 128 * correction
                        // xmm(4+i) has accumulated sum_qs_K * d_Q * d_K as scalar
                        // NOTE: Must reload 128.0f because softmax code below clobbers ymm30
                        mov(eax, 0x43000000); // 128.0f in IEEE754
                        vmovd(xmm29, eax);
                        vmulss(xmm29, xmm29, Xmm(4 + i)); // 128 * correction
                        vsubss(xmm28, xmm28, xmm29);      // result - 128 * correction

                        vmulss(xmm28, xmm28, Xmm(ymm_scale.getIdx()));

                        // Mask
                        Label skip_mask_tail;
                        cmp(reg_mask_base, 0);
                        je(skip_mask_tail, T_NEAR);
                        mov(reg_tmp, reg_mask_stride);
                        imul(reg_tmp, reg_tmp, i);
                        add(reg_tmp, reg_mask_base);
                        mov(reg_Q_cursor, reg_n);
                        shl(reg_Q_cursor, 2);
                        add(reg_tmp, reg_Q_cursor);
                        vaddss(xmm28, xmm28, ptr[reg_tmp]);
                        L(skip_mask_tail);

                        // Store
                        mov(reg_tmp, reg_C_stride);
                        imul(reg_tmp, reg_tmp, i);
                        add(reg_tmp, reg_C_base);
                        mov(reg_Q_cursor, reg_n);
                        shl(reg_Q_cursor, 2);
                        add(reg_tmp, reg_Q_cursor);
                        vmovss(ptr[reg_tmp], xmm28);

                        // Softmax
                        vmovss(xmm31, Xmm(14 + i));
                        vmaxss(xmm30, xmm31, xmm28);
                        vsubss(xmm29, xmm31, xmm30);
                        exp_ss(xmm29, xmm29);
                        vmulss(Xmm(14 + m_blocking_ + i), Xmm(14 + m_blocking_ + i), xmm29);
                        vsubss(xmm28, xmm28, xmm30);
                        exp_ss(xmm28, xmm28);
                        vaddss(Xmm(14 + m_blocking_ + i), Xmm(14 + m_blocking_ + i), xmm28);
                        vmovss(Xmm(14 + i), xmm30);
                    }

                    inc(reg_n);
                    jmp(loop_tail, T_NEAR);
                }

                L("end_pass1");
                jmp("end_kernel", T_NEAR); // DEBUG: Skip pass2 to test raw GEMM

                // ============================================================
                // PASS 2: Normalize
                // ============================================================
                for (int i = 0; i < m_blocking_; ++i)
                {
                    xor_(reg_n, reg_n);

                    mov(reg_tmp, reg_C_stride);
                    imul(reg_tmp, reg_tmp, i);
                    add(reg_tmp, reg_C_base);
                    mov(reg_Q_cursor, reg_tmp);

                    Label tail_pass2;
                    Label end_pass2_row;

                    Label loop_pass2;
                    L(loop_pass2);
                    {
                        cmp(reg_n, reg_N);
                        jge(end_pass2_row, T_NEAR);

                        // Check tail
                        mov(reg_tmp, reg_N);
                        sub(reg_tmp, reg_n);
                        cmp(reg_tmp, 8); // Process 8 floats (YMM)
                        jl(tail_pass2, T_NEAR);

                        vmovups(ymm0, ptr[reg_Q_cursor]);

                        vbroadcastss(ymm1, Xmm(14 + i)); // Max[i]
                        vsubps(ymm0, ymm0, ymm1);
                        exp_ps(ymm0, ymm0);

                        vbroadcastss(ymm2, Xmm(14 + m_blocking_ + i)); // Sum[i]
                        vdivps(ymm0, ymm0, ymm2);

                        vmovups(ptr[reg_Q_cursor], ymm0);

                        add(reg_Q_cursor, 32);
                        add(reg_n, 8);
                        jmp(loop_pass2, T_NEAR);
                    }

                    L(tail_pass2);
                    Label loop_tail_pass2;
                    L(loop_tail_pass2);
                    {
                        cmp(reg_n, reg_N);
                        jge(end_pass2_row, T_NEAR);

                        vmovss(xmm0, ptr[reg_Q_cursor]);
                        vsubss(xmm0, xmm0, Xmm(14 + i));
                        exp_ss(xmm0, xmm0);
                        vdivss(xmm0, xmm0, Xmm(14 + m_blocking_ + i));
                        vmovss(ptr[reg_Q_cursor], xmm0);

                        add(reg_Q_cursor, 4);
                        inc(reg_n);
                        jmp(loop_tail_pass2, T_NEAR);
                    }

                    L(end_pass2_row);
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

            // Helpers
            void exp_ps(const Xbyak::Ymm &dst, const Xbyak::Ymm &src)
            {
                using namespace Xbyak;

                // Clamp input to -87.0f to avoid denormals
                mov(rbx, 0xC2AE0000); // -87.0f
                vmovd(xmm30, ebx);
                vpbroadcastd(ymm30, xmm30);
                vmaxps(dst, src, ymm30);

                // e^x = 2^(x * log2(e))
                mov(rbx, 0x3FB8AA3B); // log2(e)
                vmovd(xmm30, ebx);
                vpbroadcastd(ymm30, xmm30);

                vmulps(dst, dst, ymm30);

                vrndscaleps(ymm29, dst, 1); // floor
                vsubps(ymm28, dst, ymm29);  // frac

                // Horner's method (Taylor coefficients for 2^x on [0, 1))
                mov(rbx, 0x3D635847); // c3 = 0.0555041...
                vmovd(xmm30, ebx);
                vpbroadcastd(ymm30, xmm30);

                mov(rbx, 0x3E75FDDE); // c2 = 0.2402265...
                vmovd(xmm31, ebx);
                vpbroadcastd(ymm31, xmm31);
                vfmadd213ps(ymm30, ymm28, ymm31);

                mov(rbx, 0x3F317218); // c1 = 0.6931472...
                vmovd(xmm31, ebx);
                vpbroadcastd(ymm31, xmm31);
                vfmadd213ps(ymm30, ymm28, ymm31);

                mov(rbx, 0x3F800000); // c0 = 1.0
                vmovd(xmm31, ebx);
                vpbroadcastd(ymm31, xmm31);
                vfmadd213ps(ymm30, ymm28, ymm31);

                vscalefps(dst, ymm30, ymm29);
            }

            void exp_ss(const Xbyak::Xmm &dst, const Xbyak::Xmm &src)
            {
                using namespace Xbyak;

                // Clamp input to -87.0f
                mov(rbx, 0xC2AE0000); // -87.0f
                vmovd(xmm30, ebx);
                vmaxss(dst, src, xmm30);

                // Scalar version
                mov(rbx, 0x3FB8AA3B);
                vmovd(xmm30, ebx);

                vmulss(dst, dst, xmm30);

                vrndscaless(xmm29, dst, dst, 1);
                vsubss(xmm28, dst, xmm29);

                // Taylor coefficients
                mov(rbx, 0x3D635847);
                vmovd(xmm30, ebx);

                mov(rbx, 0x3E75FDDE);
                vmovd(xmm31, ebx);
                vfmadd213ss(xmm30, xmm28, xmm31);

                mov(rbx, 0x3F317218);
                vmovd(xmm31, ebx);
                vfmadd213ss(xmm30, xmm28, xmm31);

                mov(rbx, 0x3F800000);
                vmovd(xmm31, ebx);
                vfmadd213ss(xmm30, xmm28, xmm31);

                vscalefss(dst, xmm30, xmm29);
            }
        };

    }
}
