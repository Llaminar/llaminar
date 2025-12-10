/**
 * @file QuantisedAttentionJit_Q8_1_Fused.h
 * @brief Fully fused Q8_1 attention kernel: Q×K^T + OnlineSoftmax + V×Scores + Requant
 * @author David Sanftenberg
 *
 * This kernel computes the ENTIRE attention operation in a single pass:
 *   output[m] = softmax(Q[m] @ K^T * scale) @ V  →  Q8_1
 *
 * Key innovations:
 * 1. NO intermediate score matrix storage (O(seq×kv) memory saved)
 * 2. Online softmax with streaming V accumulation
 * 3. On-the-fly V dequantization while cache-hot
 * 4. Fused requantization to Q8_1 output
 *
 * Memory access pattern (for each Q row):
 *   - Read Q[m] once (head_dim/32 blocks)
 *   - Stream K[0..kv_len] (each row: head_dim/32 blocks)
 *   - Stream V[0..kv_len] (each row: head_dim/32 blocks) - fused with K read
 *   - Write output[m] once (head_dim/32 Q8_1 blocks)
 *
 * Algorithm:
 *   For each Q row m:
 *     max_m = -inf, sum_m = 0, context_accum[head_dim] = 0
 *     For each K/V position n:
 *       s = dot(Q[m], K[n]) * scale           // Q8_1 × Q8_1 → FP32
 *       if s > max_m:
 *         correction = exp(max_m - s)
 *         sum_m *= correction
 *         context_accum *= correction          // Rescale all head_dim values
 *         max_m = s
 *       weight = exp(s - max_m)
 *       sum_m += weight
 *       context_accum += weight * dequant(V[n]) // On-the-fly dequant
 *     context[m] = context_accum / sum_m
 *     output[m] = quantize_to_q8_1(context[m])
 */

#pragma once

#include "../../../../../external/onednn/third_party/xbyak/xbyak.h"
#include "../../../tensors/BlockStructures.h"
#include "../../../tensors/SIMDHelpers.h"
#include <cstdint>
#include <cmath>
#include <cstring>

namespace llaminar2
{
    namespace gemm_v4
    {
        /**
         * @brief Parameters for fully fused Q8_1 attention
         */
        struct FusedQ8_1AttentionParams
        {
            const void *Q;           ///< Q8_1 blocks [seq_len, head_dim] (row-major, head_dim/32 blocks per row)
            const void *K;           ///< Q8_1 blocks [kv_len, head_dim]
            const void *V;           ///< Q8_1 blocks [kv_len, head_dim]
            void *output;            ///< Q8_1 blocks [seq_len, head_dim] output
            int M;                   ///< Number of Q rows (seq_len for this head)
            int N;                   ///< Number of K/V rows (kv_len)
            int head_dim;            ///< Head dimension (must be multiple of 32)
            int Q_stride_bytes;      ///< Stride between Q rows in bytes
            int K_stride_bytes;      ///< Stride between K rows in bytes
            int V_stride_bytes;      ///< Stride between V rows in bytes
            int output_stride_bytes; ///< Stride between output rows in bytes
            float scale;             ///< Attention scale (1/sqrt(head_dim))
            const float *mask;       ///< Optional attention mask [M, N] (nullptr if none)
            int mask_stride;         ///< Stride between mask rows (in floats)
        };

        /**
         * @class QuantisedAttentionJit_Q8_1_Fused
         * @brief JIT kernel for fully fused Q8_1 attention with online softmax and requantization
         *
         * Register allocation strategy (AVX-512):
         *   ZMM0-3:   Context accumulators (head_dim=64: 4×16 floats)
         *   ZMM4-5:   Q blocks (2 blocks for head_dim=64)
         *   ZMM6-7:   K blocks (loaded per N iteration)
         *   ZMM8-9:   V blocks (loaded per N iteration, dequantized)
         *   ZMM10:    Current score (scalar broadcast)
         *   ZMM11:    Current weight (exp(s - max))
         *   ZMM12:    Running max
         *   ZMM13:    Running sum
         *   ZMM14:    Correction factor (for rescaling)
         *   ZMM15:    Temp for dequant scale
         *   ZMM16-19: More context accumulators if head_dim > 64
         *   ZMM20-25: Scratch registers
         *   ZMM26:    Constant: 128 (for unsigned conversion)
         *   ZMM27:    Constant: attention scale
         *   ZMM28:    Constant: -inf
         *   ZMM29:    Constant: 1.0f
         *   ZMM30-31: Scratch
         */
        class QuantisedAttentionJit_Q8_1_Fused : public Xbyak::CodeGenerator
        {
        public:
            /**
             * @brief Construct JIT kernel for specific head dimension
             * @param head_dim Head dimension (must be multiple of 32, typically 64, 128, 256, etc.)
             */
            explicit QuantisedAttentionJit_Q8_1_Fused(int head_dim = 64, bool debug_gen = false)
                : Xbyak::CodeGenerator(64 * 1024), head_dim_(head_dim), debug_generate_(debug_gen)
            {
                if (head_dim_ % 32 != 0)
                {
                    throw std::invalid_argument("head_dim must be multiple of 32");
                }
                num_blocks_ = head_dim_ / 32;

                // Compute dynamic stack layout offsets
                // Q blocks area: num_blocks * 64 bytes, rounded up to 64-byte alignment
                q_area_size_ = ((num_blocks_ * 64) + 63) & ~63;
                // Metadata area immediately after Q blocks: 64 bytes (K_base, V_base, reg_n, Q_norm_sq, padding)
                k_base_offset_ = q_area_size_;
                v_base_offset_ = q_area_size_ + 8;
                reg_n_offset_ = q_area_size_ + 16;
                q_norm_sq_offset_ = q_area_size_ + 24; // |Q|² for online normalization
                // q_stride_offset_ will be calculated later to ensure it's in a safe zone
                // Spill area starts at q_area_size_ + 64 (metadata area is 64 bytes)
                spill_base_ = q_area_size_ + 64;

                if (debug_generate_)
                    std::cerr << "[JIT_DEBUG] Starting generate() for head_dim=" << head_dim_
                              << " (" << num_blocks_ << " blocks), q_area=" << q_area_size_
                              << ", spill_base=" << spill_base_ << std::endl;
                generate();
                if (debug_generate_)
                    std::cerr << "[JIT_DEBUG] generate() completed successfully" << std::endl;
            }

            /**
             * @brief Get stack offset for low half of spilled context block
             * @param block_idx Block index (must be >= 2)
             * @return Stack offset for ctx_lo (16 floats = 64 bytes)
             *
             * Stack layout for spilled blocks:
             *   [rsp + spill_base_ + (b-2)*128]      : ctx_lo (64 bytes)
             *   [rsp + spill_base_ + (b-2)*128 + 64] : ctx_hi (64 bytes)
             */
            int spill_offset_lo(int block_idx) const
            {
                return spill_base_ + (block_idx - 2) * 128;
            }

            /**
             * @brief Get stack offset for high half of spilled context block
             * @param block_idx Block index (must be >= 2)
             * @return Stack offset for ctx_hi (16 floats = 64 bytes)
             */
            int spill_offset_hi(int block_idx) const
            {
                return spill_base_ + (block_idx - 2) * 128 + 64;
            }

            using kernel_func_t = void (*)(const FusedQ8_1AttentionParams *params);

            kernel_func_t get_kernel()
            {
                return getCode<kernel_func_t>();
            }

            size_t get_code_size() const
            {
                return getSize();
            }

        private:
            int head_dim_;
            int num_blocks_;       // head_dim / 32
            int stack_alloc_size_; // Actual stack allocation size (for epilogue)
            bool debug_generate_ = false;

            // Dynamic stack layout offsets (computed in constructor)
            int q_area_size_;      // Size of Q blocks area (rounded to 64)
            int k_base_offset_;    // Offset for K_base pointer storage
            int v_base_offset_;    // Offset for V_base pointer storage
            int reg_n_offset_;     // Offset for reg_n save slot
            int q_norm_sq_offset_; // Offset for |Q|² (online normalization)
            int q_stride_offset_;  // Offset for Q_stride save slot
            int spill_base_;       // Base offset for spilled context accumulators

            // Helper: Fast exp approximation using vexp2ps (2^x) and log2(e) scaling
            // exp(x) = 2^(x * log2(e))
            void emit_fast_exp(const Xbyak::Zmm &dst, const Xbyak::Zmm &src,
                               const Xbyak::Zmm &log2e, const Xbyak::Zmm &clamp_min)
            {
                using namespace Xbyak;
                if (debug_generate_)
                    std::cerr << "[JIT_DEBUG]   emit_fast_exp: vmaxps" << std::endl;
                // Clamp to avoid overflow/underflow
                vmaxps(dst, src, clamp_min);
                if (debug_generate_)
                    std::cerr << "[JIT_DEBUG]   emit_fast_exp: vmulps" << std::endl;
                // x * log2(e)
                vmulps(dst, dst, log2e);
                // 2^(x * log2(e)) using vscalefps trick or polynomial
                // For simplicity, use vexp2ps if available (AVX-512ER) or polynomial
                // Most CPUs don't have vexp2ps, so use polynomial approximation
                emit_exp2_poly(dst, dst);
            }

            // Polynomial approximation for 2^x (good for x in [-0.5, 0.5])
            // Uses Taylor series centered at 0
            void emit_exp2_poly(const Xbyak::Zmm &dst, const Xbyak::Zmm &src)
            {
                using namespace Xbyak;
                if (debug_generate_)
                    std::cerr << "[JIT_DEBUG]     emit_exp2_poly: vrndscaleps" << std::endl;
                // Split into integer and fractional parts
                // n = round(x), f = x - n
                // 2^x = 2^n * 2^f
                // Use round-to-nearest-even (imm=8) to keep f in [-0.5, 0.5] for better precision
                vrndscaleps(zmm30, src, 8); // round(x) -> zmm30
                vsubps(zmm31, src, zmm30);  // f = x - round(x) -> zmm31

                // 2^f using polynomial (for f in [-0.5, 0.5])
                // p(f) = 1 + f*(ln2 + f*(ln2^2/2 + f*(ln2^3/6 + ...)))

                // 5th order Taylor coefficients for 2^x
                static const float c0 = 1.0f;
                static const float c1 = 0.6931472f; // ln(2)
                static const float c2 = 0.2402265f; // ln(2)^2/2
                static const float c3 = 0.0555041f; // ln(2)^3/6
                static const float c4 = 0.0096181f; // ln(2)^4/24
                static const float c5 = 0.0013334f; // ln(2)^5/120

                if (debug_generate_)
                    std::cerr << "[JIT_DEBUG]     emit_exp2_poly: load c5 and start Horner" << std::endl;

                // Horner's method: c0 + f*(c1 + f*(c2 + f*(c3 + f*(c4 + f*c5))))

                // Load c5
                mov(rax, reinterpret_cast<uintptr_t>(&c5));
                vbroadcastss(dst, ptr[rax]);

                // dst = c5*f + c4
                mov(rax, reinterpret_cast<uintptr_t>(&c4));
                vbroadcastss(zmm25, ptr[rax]);
                vfmadd213ps(dst, zmm31, zmm25);

                // dst = (c5*f + c4)*f + c3
                mov(rax, reinterpret_cast<uintptr_t>(&c3));
                vbroadcastss(zmm25, ptr[rax]);
                vfmadd213ps(dst, zmm31, zmm25);

                // dst = ((...)*f + c3)*f + c2
                mov(rax, reinterpret_cast<uintptr_t>(&c2));
                vbroadcastss(zmm25, ptr[rax]);
                vfmadd213ps(dst, zmm31, zmm25);

                // dst = ((...)*f + c2)*f + c1
                mov(rax, reinterpret_cast<uintptr_t>(&c1));
                vbroadcastss(zmm25, ptr[rax]);
                vfmadd213ps(dst, zmm31, zmm25);

                // dst = ((...)*f + c1)*f + c0
                mov(rax, reinterpret_cast<uintptr_t>(&c0));
                vbroadcastss(zmm25, ptr[rax]);
                vfmadd213ps(dst, zmm31, zmm25);

                if (debug_generate_)
                    std::cerr << "[JIT_DEBUG]     emit_exp2_poly: vscalefps" << std::endl;
                // Scale by 2^n using vscalefps
                vscalefps(dst, dst, zmm30);
            }

            void generate()
            {
                using namespace Xbyak;

                // Function signature: void kernel(const FusedQ8_1AttentionParams* params)
                // params in rdi (System V ABI)

                const Reg64 &reg_params = rdi;
                const Reg64 &reg_Q = rsi;
                const Reg64 &reg_K = rdx;
                const Reg64 &reg_V = rcx;
                const Reg64 &reg_output = r8;
                const Reg64 &reg_M = r9;
                const Reg64 &reg_N = r10;
                const Reg64 &reg_Q_stride = r11;
                const Reg64 &reg_K_stride = r12;
                const Reg64 &reg_V_stride = r13;
                const Reg64 &reg_out_stride = r14;
                const Reg64 &reg_mask = r15;
                const Reg64 &reg_mask_stride = rbx;

                const Reg64 &reg_m = rbp; // Outer loop counter
                const Reg64 &reg_n = rax; // Inner loop counter
                // Use rdi as temp - it's free after params are loaded
                // DO NOT use r9 as temp - it holds M for loop comparison!
                const Reg64 &reg_tmp = rdi;
                // NOTE: K_ptr and V_ptr are computed into scratch registers
                // We save K/V base addresses to stack to preserve them across inner loop iterations
                // Stack layout: [rsp+0] = Q data, [rsp+256] = K_base, [rsp+264] = V_base, [rsp+272] = reg_n
                const Reg64 &reg_K_ptr = rcx; // Current K row (scratch)
                const Reg64 &reg_V_ptr = rdx; // Current V row (scratch)

                // ZMM register assignments
                const Zmm zmm_ctx0 = zmm0; // Context accumulator 0 (floats 0-15)
                const Zmm zmm_ctx1 = zmm1; // Context accumulator 1 (floats 16-31)
                const Zmm zmm_ctx2 = zmm2; // Context accumulator 2 (floats 32-47) if head_dim >= 64
                const Zmm zmm_ctx3 = zmm3; // Context accumulator 3 (floats 48-63) if head_dim >= 64

                const Zmm zmm_max = zmm12;    // Running max (scalar broadcast)
                const Zmm zmm_sum = zmm13;    // Running sum (scalar broadcast)
                const Zmm zmm_score = zmm10;  // Current score
                const Zmm zmm_weight = zmm11; // Current weight = exp(score - max)
                const Zmm zmm_corr = zmm14;   // Correction factor

                const Zmm zmm_128 = zmm26;     // Constant: 0x80 for unsigned conversion
                const Zmm zmm_scale = zmm27;   // Attention scale
                const Zmm zmm_neg_inf = zmm28; // -infinity
                const Zmm zmm_one = zmm29;     // 1.0f

                const Zmm zmm_log2e = zmm22;   // log2(e) for fast exp
                const Zmm zmm_exp_min = zmm23; // Min exp input (-87)

                // Prologue - save callee-saved registers
                if (debug_generate_)
                    std::cerr << "[JIT_DEBUG] Section 1: Prologue" << std::endl;
                push(rbx);
                push(rbp);
                push(r12);
                push(r13);
                push(r14);
                push(r15);

                // Load parameters
                if (debug_generate_)
                    std::cerr << "[JIT_DEBUG] Section 2: Load parameters" << std::endl;
                mov(reg_Q, ptr[reg_params + offsetof(FusedQ8_1AttentionParams, Q)]);
                mov(reg_K, ptr[reg_params + offsetof(FusedQ8_1AttentionParams, K)]);
                mov(reg_V, ptr[reg_params + offsetof(FusedQ8_1AttentionParams, V)]);
                mov(reg_output, ptr[reg_params + offsetof(FusedQ8_1AttentionParams, output)]);
                mov(reg_M.cvt32(), ptr[reg_params + offsetof(FusedQ8_1AttentionParams, M)]);
                mov(reg_N.cvt32(), ptr[reg_params + offsetof(FusedQ8_1AttentionParams, N)]);
                mov(reg_Q_stride.cvt32(), ptr[reg_params + offsetof(FusedQ8_1AttentionParams, Q_stride_bytes)]);
                mov(reg_K_stride.cvt32(), ptr[reg_params + offsetof(FusedQ8_1AttentionParams, K_stride_bytes)]);
                mov(reg_V_stride.cvt32(), ptr[reg_params + offsetof(FusedQ8_1AttentionParams, V_stride_bytes)]);
                mov(reg_out_stride.cvt32(), ptr[reg_params + offsetof(FusedQ8_1AttentionParams, output_stride_bytes)]);
                mov(reg_mask, ptr[reg_params + offsetof(FusedQ8_1AttentionParams, mask)]);
                mov(reg_mask_stride.cvt32(), ptr[reg_params + offsetof(FusedQ8_1AttentionParams, mask_stride)]);

                // Load constants
                if (debug_generate_)
                    std::cerr << "[JIT_DEBUG] Section 3: Load constants" << std::endl;
                vbroadcastss(zmm_scale, ptr[reg_params + offsetof(FusedQ8_1AttentionParams, scale)]);

                // 0x80808080 for unsigned conversion
                mov(reg_tmp.cvt32(), 0x80808080);
                vpbroadcastd(zmm_128, reg_tmp.cvt32());

                // -infinity
                mov(reg_tmp.cvt32(), 0xFF800000);
                vpbroadcastd(zmm_neg_inf, reg_tmp.cvt32());

                // 1.0f
                mov(reg_tmp.cvt32(), 0x3F800000);
                vpbroadcastd(zmm_one, reg_tmp.cvt32());

                // log2(e) = 1.4426950408889634
                mov(reg_tmp.cvt32(), 0x3FB8AA3B);
                vpbroadcastd(zmm_log2e, reg_tmp.cvt32());

                // -87.0f (min exp input to avoid underflow)
                mov(reg_tmp.cvt32(), 0xC2AE0000);
                vpbroadcastd(zmm_exp_min, reg_tmp.cvt32());

                // Allocate stack space for Q blocks AND preserved values AND spilled context accumulators
                // Dynamic stack layout (64-byte aligned for ZMM operations):
                //   [rsp+0 .. q_area_size_-1]        = Q blocks (num_blocks × 64 bytes, rounded to 64)
                //   [rsp+q_area_size_ .. +7]         = K_base pointer
                //   [rsp+q_area_size_+8 .. +15]      = V_base pointer
                //   [rsp+q_area_size_+16 .. +23]     = reg_n save slot
                //   [rsp+q_area_size_+24 .. +63]     = padding for 64-byte alignment
                //   [rsp+spill_base_ .. ]            = Spilled context accumulators (128 bytes per block for blocks >= 2)
                if (debug_generate_)
                    std::cerr << "[JIT_DEBUG] Section 4: Stack allocation (q_area=" << q_area_size_
                              << ", spill_base=" << spill_base_ << ")" << std::endl;
                // spill_base_ + (num_blocks - 2) * 128 bytes for spilled context accumulators
                const int spill_size = (num_blocks_ > 2) ? (num_blocks_ - 2) * 128 : 0;
                const int stack_size = spill_base_ + spill_size;
                // Round up to 64-byte alignment for ZMM
                stack_alloc_size_ = (stack_size + 63) & ~63;

                // Place Q_stride at the end of the stack frame to avoid any potential overwrites
                q_stride_offset_ = stack_alloc_size_;
                stack_alloc_size_ += 64; // Add extra space for Q_stride and alignment padding

                sub(rsp, stack_alloc_size_);

                // Save K and V base addresses to stack (they'll be clobbered in inner loop)
                mov(ptr[rsp + k_base_offset_], reg_K);          // Save K base
                mov(ptr[rsp + v_base_offset_], reg_V);          // Save V base
                mov(ptr[rsp + q_stride_offset_], reg_Q_stride); // Save Q stride

                // ============================================================
                // OUTER LOOP: For each Q row (m = 0..M-1)
                // ============================================================
                if (debug_generate_)
                    std::cerr << "[JIT_DEBUG] Section 5: Outer loop setup" << std::endl;

                xor_(reg_m, reg_m);

                Label loop_M, end_M;
                L(loop_M);
                cmp(reg_m, reg_M);
                jge(end_M, T_NEAR);

                // CRITICAL: Re-initialize constants that get clobbered by output quantization
                // at the end of each M iteration. Must be done BEFORE Section 6b (Compute |Q|²)
                // because |Q|² computation uses zmm_128!
                //   zmm26 (zmm_128): clobbered by sum_qs calculation
                //   zmm22 (zmm_log2e): clobbered by xmm22 used for scale calculation
                //   zmm23 (zmm_exp_min): clobbered by xmm23 used for FP16 conversion
                if (debug_generate_)
                    std::cerr << "[JIT_DEBUG] Section 5b: Re-init constants" << std::endl;
                mov(reg_tmp.cvt32(), 0x80808080);
                vpbroadcastd(zmm_128, reg_tmp.cvt32());
                mov(reg_tmp.cvt32(), 0x3FB8AA3B); // log2(e) ≈ 1.4427f
                vpbroadcastd(zmm_log2e, reg_tmp.cvt32());
                mov(reg_tmp.cvt32(), 0xC2AE0000); // -87.0f (exp min clamp)
                vpbroadcastd(zmm_exp_min, reg_tmp.cvt32());

                // --------------------------------------------------------
                // Load Q[m] blocks into stack (reused across all N iterations)
                // --------------------------------------------------------
                // Q_ptr = Q + m * Q_stride
                if (debug_generate_)
                    std::cerr << "[JIT_DEBUG] Section 6: Load Q blocks to stack" << std::endl;
                mov(reg_tmp, reg_m);
                mov(reg_Q_stride, ptr[rsp + q_stride_offset_]); // Reload Q stride
                imul(reg_tmp, reg_Q_stride);
                add(reg_tmp, reg_Q);

                // Copy Q blocks to stack for fast access
                for (int b = 0; b < num_blocks_; ++b)
                {
                    // Load 36 bytes per block, store to stack
                    // Use vmovdqu32 for extended registers (ymm16-31) which require EVEX encoding
                    vmovdqu32(ymm30, ptr[reg_tmp + b * 36]); // 32 bytes

                    vmovdqu32(ptr[rsp + b * 64], ymm30);  // Store to stack (padded)
                    mov(eax, ptr[reg_tmp + b * 36 + 32]); // Last 4 bytes
                    mov(ptr[rsp + b * 64 + 32], eax);
                }

                // --------------------------------------------------------
                // Section 6b: [REMOVED] Q/K normalization removed for standard attention
                // Standard attention uses: score = (Q · K) * scale
                // NOT: score = (Q · K) * scale / (|Q| * |K|)
                // --------------------------------------------------------
                if (debug_generate_)
                    std::cerr << "[JIT_DEBUG] Section 6b: SKIPPED (Q/K normalization removed)" << std::endl;

                // --------------------------------------------------------
                // Initialize accumulators for this Q row
                // --------------------------------------------------------
                if (debug_generate_)
                    std::cerr << "[JIT_DEBUG] Section 7: Initialize accumulators" << std::endl;
                vmovaps(zmm_max, zmm_neg_inf);     // max = -inf
                vxorps(zmm_sum, zmm_sum, zmm_sum); // sum = 0

                // Zero context accumulators (in registers for blocks 0-1)
                vxorps(zmm_ctx0, zmm_ctx0, zmm_ctx0);
                vxorps(zmm_ctx1, zmm_ctx1, zmm_ctx1);
                if (num_blocks_ >= 2)
                {
                    vxorps(zmm_ctx2, zmm_ctx2, zmm_ctx2);
                    vxorps(zmm_ctx3, zmm_ctx3, zmm_ctx3);
                }

                // Zero spilled context accumulators on stack (for blocks 2+)
                if (num_blocks_ > 2)
                {
                    // Use zmm30 as zero source
                    vxorps(zmm30, zmm30, zmm30);
                    for (int b = 2; b < num_blocks_; ++b)
                    {
                        vmovups(ptr[rsp + spill_offset_lo(b)], zmm30);
                        vmovups(ptr[rsp + spill_offset_hi(b)], zmm30);
                    }
                }

                // CRITICAL: Re-initialize constants moved to start of loop

                // --------------------------------------------------------
                // INNER LOOP: For each K/V position (n = 0..N-1)
                // --------------------------------------------------------
                if (debug_generate_)
                    std::cerr << "[JIT_DEBUG] Section 8: Inner loop setup" << std::endl;
                xor_(reg_n, reg_n);

                Label loop_N, end_N;
                L(loop_N);
                cmp(reg_n, reg_N);
                jge(end_N, T_NEAR);

                // K_ptr = K_base + n * K_stride (load K_base from stack)
                if (debug_generate_)
                    std::cerr << "[JIT_DEBUG] Section 9: K/V ptr calc" << std::endl;
                mov(reg_K_ptr, reg_n);
                imul(reg_K_ptr, reg_K_stride);
                add(reg_K_ptr, ptr[rsp + k_base_offset_]); // Load K_base from stack

                // V_ptr = V_base + n * V_stride (load V_base from stack)
                mov(reg_V_ptr, reg_n);
                imul(reg_V_ptr, reg_V_stride);
                add(reg_V_ptr, ptr[rsp + v_base_offset_]); // Load V_base from stack

                // IMPORTANT: Save reg_n (rax) now - it will be clobbered by eax uses and emit_fast_exp
                mov(ptr[rsp + reg_n_offset_], reg_n);

                // ========================================
                // Step 1: Compute score = dot(Q[m], K[n]) * scale
                // ========================================
                // Accumulate dot product across all blocks
                // Standard attention: NO Q/K normalization
                // Use xmm_score_acc for accumulation (lower part of zmm_score)
                if (debug_generate_)
                    std::cerr << "[JIT_DEBUG] Section 10: Score computation loop" << std::endl;
                const Xmm xmm_score_acc = Xmm(zmm_score.getIdx());
                vxorps(xmm_score_acc, xmm_score_acc, xmm_score_acc);

                for (int b = 0; b < num_blocks_; ++b)
                {
                    if (debug_generate_)
                        std::cerr << "[JIT_DEBUG]   Block " << b << ": Load Q from stack" << std::endl;
                    // Load Q block from stack (scales at offset 0, 2, data at offset 4)
                    // d_Q is a scalar - load as FP16, convert to FP32
                    // Use xmm4-9 for scratch (NOT xmm20-25 which overlap with zmm constants!)
                    // Register allocation for score loop:
                    //   xmm4/ymm4: Q data
                    //   xmm5/ymm5: K data
                    //   ymm6: vpdpbusd result
                    //   xmm6, xmm7: horizontal sum scratch
                    //   xmm8: d_Q (FP32)
                    //   xmm9: d_K (FP32)
                    //   xmm15: sum_qs_K (float), correction
                    //   xmm14: scratch for 128.0f constant
                    //   xmm31: |K|² accumulator
                    vpbroadcastw(xmm8, ptr[rsp + b * 64]); // d_Q (FP16) - broadcast to xmm
                    vcvtph2ps(xmm8, xmm8);                 // Convert FP16->FP32 (xmm->xmm)
                    if (debug_generate_)
                        std::cerr << "[JIT_DEBUG]   Block " << b << ": vmovdqu8 ymm4" << std::endl;
                    vmovdqu8(ymm4, ptr[rsp + b * 64 + 4]); // Q data (32 int8)
                    // XOR with 0x80 to convert signed->unsigned (only Q, not K!)
                    // vpdpbusd: src1 (Q) is unsigned, src2 (K) is signed
                    vpxord(ymm4, ymm4, Ymm(zmm_128.getIdx()));

                    if (debug_generate_)
                        std::cerr << "[JIT_DEBUG]   Block " << b << ": Load K" << std::endl;
                    // Load K block (scales + data)
                    vpbroadcastw(xmm9, ptr[reg_K_ptr + b * 36]);      // d_K (FP16)
                    vcvtph2ps(xmm9, xmm9);                            // Convert FP16->FP32
                    vpbroadcastw(xmm15, ptr[reg_K_ptr + b * 36 + 2]); // sum_qs_K (INT16)
                    vpmovsxwd(xmm15, xmm15);                          // Sign extend word->dword
                    vcvtdq2ps(xmm15, xmm15);                          // Convert int->float
                    vmovdqu8(ymm5, ptr[reg_K_ptr + b * 36 + 4]);      // K data (32 int8, SIGNED)
                    // K remains SIGNED - do NOT XOR with 0x80!

                    if (debug_generate_)
                        std::cerr << "[JIT_DEBUG]   Block " << b << ": VPDPBUSD for Q*K" << std::endl;
                    // vpdpbusd: src1=unsigned (Q+128), src2=signed (K)
                    // Computes: acc += (Q+128) * K = Q*K + 128*K
                    // Need correction: subtract 128 * sum_qs_K
                    vxorps(ymm6, ymm6, ymm6);
                    vpdpbusd(ymm6, ymm4, ymm5);

                    if (debug_generate_)
                        std::cerr << "[JIT_DEBUG]   Block " << b << ": Horizontal sum" << std::endl;
                    // Horizontal sum of dot product
                    vextracti128(xmm7, ymm6, 1);
                    vpaddd(xmm6, xmm6, xmm7);
                    vpshufd(xmm7, xmm6, 0x4E);
                    vpaddd(xmm6, xmm6, xmm7);
                    vpshufd(xmm7, xmm6, 0xB1);
                    vpaddd(xmm6, xmm6, xmm7);
                    // xmm6[0] = raw_dot = sum((Q+128) * K) = sum(Q*K) + 128*sum(K)

                    if (debug_generate_)
                        std::cerr << "[JIT_DEBUG]   Block " << b << ": Convert to float" << std::endl;
                    // Convert to float
                    vcvtdq2ps(xmm6, xmm6);

                    // Apply correction for unsigned conversion:
                    // raw_dot = Q*K + 128*sum_qs_K (because vpdpbusd used Q+128)
                    // We want: Q*K = raw_dot - 128*sum_qs_K
                    // Then multiply by d_Q * d_K: result = (raw_dot - 128*sum_qs_K) * d_Q * d_K
                    if (debug_generate_)
                        std::cerr << "[JIT_DEBUG]   Block " << b << ": Compute correction" << std::endl;
                    // Combined scale: d_Q * d_K (using xmm8 for d_Q, xmm9 for d_K)
                    vmulps(xmm8, xmm8, xmm9); // d_Q * d_K -> xmm8

                    // Correction = 128.0f * sum_qs_K (xmm15 has sum_qs_K as float)
                    mov(reg_tmp.cvt32(), 0x43000000); // 128.0f in IEEE 754
                    vmovd(xmm14, reg_tmp.cvt32());
                    vbroadcastss(xmm14, xmm14);
                    vmulps(xmm15, xmm15, xmm14); // 128 * sum_qs_K -> xmm15

                    if (debug_generate_)
                        std::cerr << "[JIT_DEBUG]   Block " << b << ": Final dot" << std::endl;
                    // Subtract correction from raw_dot, then scale
                    vsubps(xmm6, xmm6, xmm15); // raw_dot - 128*sum_qs_K
                    vmulps(xmm6, xmm6, xmm8);  // * d_Q * d_K

                    // Accumulate
                    vaddps(xmm_score_acc, xmm_score_acc, xmm6);
                }

                // Apply attention scale (scalar)
                // Standard attention: score = (Q · K) * scale (no Q/K normalization)
                if (debug_generate_)
                    std::cerr << "[JIT_DEBUG] Section 11: Apply scale and mask" << std::endl;
                vmulss(xmm_score_acc, xmm_score_acc, Xmm(zmm_scale.getIdx()));

                // Apply mask if present
                test(reg_mask, reg_mask);
                jz("no_mask", T_NEAR);
                {
                    // mask_val = mask[m * mask_stride + n]
                    // We use reg_tmp (rdi) for the calculation to avoid clobbering reg_n (rax).
                    // reg_n (rax) holds the current n loop counter.

                    mov(reg_tmp.cvt32(), reg_m.cvt32());            // edi = m (zero extends rdi)
                    imul(reg_tmp.cvt32(), reg_mask_stride.cvt32()); // edi = m * stride
                    add(reg_tmp.cvt32(), reg_n.cvt32());            // edi = m * stride + n

                    // Add mask value to accumulator
                    vaddss(xmm_score_acc, xmm_score_acc, ptr[reg_mask + reg_tmp * 4]);
                }
                L("no_mask");

                if (debug_generate_)
                    std::cerr << "[JIT_DEBUG] Section 12: Broadcast score" << std::endl;
                // Broadcast score to all lanes for comparison and later use
                vbroadcastss(zmm_score, xmm_score_acc);

                // ========================================
                // Step 2: Online softmax update
                // ========================================
                // if score > max:
                //   correction = exp(max - score)
                //   sum *= correction
                //   context_accum *= correction
                //   max = score
                // weight = exp(score - max)
                // sum += weight

                if (debug_generate_)
                    std::cerr << "[JIT_DEBUG] Section 13: Compare with max" << std::endl;
                // Compare score with current max (scalar comparison)
                vcomiss(xmm_score_acc, Xmm(zmm_max.getIdx()));
                jbe("score_le_max", T_NEAR);

                // score > max: need to rescale
                if (debug_generate_)
                    std::cerr << "[JIT_DEBUG] Section 14: Rescale branch" << std::endl;
                {
                    // correction = exp(max - score)
                    // NOTE: reg_n was already saved to [rsp+272] before score computation
                    vsubps(zmm_corr, zmm_max, zmm_score);
                    emit_fast_exp(zmm_corr, zmm_corr, zmm_log2e, zmm_exp_min);

                    // sum *= correction
                    vmulps(zmm_sum, zmm_sum, zmm_corr);

                    // context_accum *= correction (for all blocks)
                    vmulps(zmm_ctx0, zmm_ctx0, zmm_corr);
                    vmulps(zmm_ctx1, zmm_ctx1, zmm_corr);
                    if (num_blocks_ >= 2)
                    {
                        vmulps(zmm_ctx2, zmm_ctx2, zmm_corr);
                        vmulps(zmm_ctx3, zmm_ctx3, zmm_corr);
                    }

                    // Rescale spilled context accumulators (blocks 2+)
                    if (num_blocks_ > 2)
                    {
                        for (int b = 2; b < num_blocks_; ++b)
                        {
                            // Load from stack, multiply by correction, store back
                            vmovups(zmm30, ptr[rsp + spill_offset_lo(b)]);
                            vmulps(zmm30, zmm30, zmm_corr);
                            vmovups(ptr[rsp + spill_offset_lo(b)], zmm30);
                            vmovups(zmm30, ptr[rsp + spill_offset_hi(b)]);
                            vmulps(zmm30, zmm30, zmm_corr);
                            vmovups(ptr[rsp + spill_offset_hi(b)], zmm30);
                        }
                    }

                    // max = score
                    vmovaps(zmm_max, zmm_score);
                }

                L("score_le_max");

                // weight = exp(score - max)
                if (debug_generate_)
                    std::cerr << "[JIT_DEBUG] Section 15: Compute weight" << std::endl;
                // NOTE: reg_n was already saved to [rsp+272] before score computation
                vsubps(zmm_weight, zmm_score, zmm_max);
                emit_fast_exp(zmm_weight, zmm_weight, zmm_log2e, zmm_exp_min);

                // sum += weight
                vaddps(zmm_sum, zmm_sum, zmm_weight);

                // ========================================
                // Step 3: Accumulate weighted V
                // context_accum += weight * dequant(V[n])
                // ========================================
                if (debug_generate_)
                    std::cerr << "[JIT_DEBUG] Section 16: Accumulate weighted V" << std::endl;
                for (int b = 0; b < num_blocks_; ++b)
                {
                    // Load V block and dequantize
                    // d_V is a single FP16 scalar - load, convert to FP32, then broadcast to zmm
                    vpbroadcastw(xmm15, ptr[reg_V_ptr + b * 36]); // d_V (FP16) broadcast to xmm
                    vcvtph2ps(xmm15, xmm15);                      // Convert FP16->FP32 in xmm (element 0)
                    vbroadcastss(zmm15, xmm15);                   // Broadcast scalar to all zmm lanes

                    // Load V data (32 int8 values)
                    vpmovsxbd(zmm8, ptr[reg_V_ptr + b * 36 + 4]);      // First 16 int8 -> int32
                    vpmovsxbd(zmm9, ptr[reg_V_ptr + b * 36 + 4 + 16]); // Next 16 int8 -> int32

                    // Convert to float
                    vcvtdq2ps(zmm8, zmm8);
                    vcvtdq2ps(zmm9, zmm9);

                    // Scale by d_V
                    vmulps(zmm8, zmm8, zmm15);
                    vmulps(zmm9, zmm9, zmm15);

                    // Multiply by weight and accumulate
                    // context_accum[b*32 + 0..15] += weight * V_dequant[0..15]
                    // context_accum[b*32 + 16..31] += weight * V_dequant[16..31]
                    if (b == 0)
                    {
                        vfmadd231ps(zmm_ctx0, zmm8, zmm_weight);
                        vfmadd231ps(zmm_ctx1, zmm9, zmm_weight);
                    }
                    else if (b == 1)
                    {
                        vfmadd231ps(zmm_ctx2, zmm8, zmm_weight);
                        vfmadd231ps(zmm_ctx3, zmm9, zmm_weight);
                    }
                    else
                    {
                        // Spilled blocks (b >= 2): load from stack, FMA, store back
                        vmovups(zmm30, ptr[rsp + spill_offset_lo(b)]);
                        vfmadd231ps(zmm30, zmm8, zmm_weight);
                        vmovups(ptr[rsp + spill_offset_lo(b)], zmm30);
                        vmovups(zmm30, ptr[rsp + spill_offset_hi(b)]);
                        vfmadd231ps(zmm30, zmm9, zmm_weight);
                        vmovups(ptr[rsp + spill_offset_hi(b)], zmm30);
                    }
                }

                // Next N - restore reg_n (rax) from stack first (it was clobbered during score computation and exp calls)
                mov(reg_n, ptr[rsp + reg_n_offset_]);
                inc(reg_n);
                jmp(loop_N, T_NEAR);

                L(end_N);

                // ========================================
                // Step 4: Normalize and quantize to Q8_1
                // context[m] = context_accum / sum
                // output[m] = quantize_to_q8_1(context[m])
                // ========================================

                // Broadcast 1/sum for normalization
                // NOTE: vrcp14ps is approximate (error < 2^-14), which causes divergence.
                // We use full precision division: 1.0f / sum
                mov(reg_tmp.cvt32(), 0x3f800000);     // 1.0f as int32
                vpbroadcastd(zmm14, reg_tmp.cvt32()); // Broadcast 1.0f
                vdivps(zmm14, zmm14, zmm_sum);        // 1.0f / sum

                // Normalize context accumulators (blocks 0-1 in registers)
                vmulps(zmm_ctx0, zmm_ctx0, zmm14);
                vmulps(zmm_ctx1, zmm_ctx1, zmm14);
                if (num_blocks_ >= 2)
                {
                    vmulps(zmm_ctx2, zmm_ctx2, zmm14);
                    vmulps(zmm_ctx3, zmm_ctx3, zmm14);
                }

                // Normalize spilled context accumulators (blocks 2+)
                if (num_blocks_ > 2)
                {
                    for (int b = 2; b < num_blocks_; ++b)
                    {
                        vmovups(zmm30, ptr[rsp + spill_offset_lo(b)]);
                        vmulps(zmm30, zmm30, zmm14);
                        vmovups(ptr[rsp + spill_offset_lo(b)], zmm30);
                        vmovups(zmm30, ptr[rsp + spill_offset_hi(b)]);
                        vmulps(zmm30, zmm30, zmm14);
                        vmovups(ptr[rsp + spill_offset_hi(b)], zmm30);
                    }
                }

                // Output pointer for this row
                mov(reg_tmp, reg_m);
                imul(reg_tmp, reg_out_stride);
                add(reg_tmp, reg_output);

                // Quantize each 32-element block to Q8_1
                for (int b = 0; b < num_blocks_; ++b)
                {
                    // Get the two ZMM registers for this block (32 floats)
                    // For blocks 0-1: use zmm_ctx0-3 directly
                    // For blocks 2+: load from stack spill area first
                    Zmm ctx_lo, ctx_hi;
                    if (b == 0)
                    {
                        ctx_lo = zmm_ctx0;
                        ctx_hi = zmm_ctx1;
                    }
                    else if (b == 1)
                    {
                        ctx_lo = zmm_ctx2;
                        ctx_hi = zmm_ctx3;
                    }
                    else
                    {
                        // Load spilled block into zmm4/zmm5
                        vmovups(zmm4, ptr[rsp + spill_offset_lo(b)]);
                        vmovups(zmm5, ptr[rsp + spill_offset_hi(b)]);
                        ctx_lo = zmm4;
                        ctx_hi = zmm5;
                    }

                    // Find max absolute value across 32 floats
                    vrangeps(zmm20, ctx_lo, ctx_lo, 0x0B); // abs(ctx_lo)
                    vrangeps(zmm21, ctx_hi, ctx_hi, 0x0B); // abs(ctx_hi)
                    vmaxps(zmm20, zmm20, zmm21);

                    // Horizontal max reduction
                    vextractf32x8(ymm21, zmm20, 1);
                    vmaxps(ymm20, ymm20, ymm21);
                    // Use EVEX version for extended registers
                    vextractf32x4(xmm21, ymm20, 1); // EVEX: vextractf32x4 instead of vextractf128
                    vmaxps(xmm20, xmm20, xmm21);
                    vshufps(xmm21, xmm20, xmm20, 0x4E);
                    vmaxps(xmm20, xmm20, xmm21);
                    vshufps(xmm21, xmm20, xmm20, 0xB1);
                    vmaxps(xmm20, xmm20, xmm21);
                    // xmm20[0] now has max_abs

                    // Compute scale: d = max_abs / 127.0
                    mov(eax, 0x42FE0000); // 127.0f
                    vmovd(xmm21, eax);
                    vdivss(xmm22, xmm20, xmm21); // d = max_abs / 127

                    // Convert scale to FP16
                    vcvtps2ph(xmm23, xmm22, 0);
                    vpextrw(eax, xmm23, 0);
                    mov(ptr[reg_tmp + b * 36], ax); // Store d (FP16)

                    // Convert back to FP32 to match what the next layer will see
                    vcvtph2ps(xmm22, xmm23);

                    // Compute inverse scale for quantization using precise division
                    // inv_d = 1.0f / d_recon
                    // Handle d_recon == 0 case to avoid Inf/NaN -> -128 artifact

                    // Check if d_recon (xmm22) is not zero
                    vxorps(xmm6, xmm6, xmm6);   // 0.0f
                    vcmpss(k1, xmm22, xmm6, 4); // 4 = _CMP_NEQ_UQ (Not Equal)

                    // Initialize result to 0.0f
                    vxorps(xmm24, xmm24, xmm24);

                    // Compute division only where d_recon != 0
                    vmovaps(xmm6, Xmm(zmm_one.getIdx())); // 1.0f
                    vdivss(xmm24 | k1, xmm6, xmm22);      // xmm24 = 1.0f / d_recon (masked)

                    vbroadcastss(zmm24, xmm24); // Broadcast to zmm24

                    // Quantize: qs[i] = round(ctx[i] * inv_d)
                    vmulps(zmm20, ctx_lo, zmm24);
                    vmulps(zmm21, ctx_hi, zmm24);

                    // Round to nearest and convert to int32
                    vcvtps2dq(zmm20, zmm20);
                    vcvtps2dq(zmm21, zmm21);

                    // Pack to int8 with saturation
                    vpmovdb(xmm20, zmm20); // 16 int32 -> 16 int8
                    vpmovdb(xmm21, zmm21); // 16 int32 -> 16 int8
                    // Use EVEX version for extended registers
                    vinserti32x4(ymm20, ymm20, xmm21, 1); // EVEX: vinserti32x4 instead of vinserti128

                    // Compute sum_qs (sum of int8 values)
                    vpmovsxbd(zmm25, xmm20);
                    // Use EVEX version for extended registers
                    vextracti32x4(xmm26, ymm20, 1); // EVEX: vextracti32x4 instead of vextracti128
                    vpmovsxbd(zmm26, xmm26);
                    vpaddd(zmm25, zmm25, zmm26);

                    // Horizontal sum
                    vextracti32x8(ymm26, zmm25, 1);
                    vpaddd(ymm25, ymm25, ymm26);
                    // Use EVEX version for extended registers
                    vextracti32x4(xmm26, ymm25, 1); // EVEX: vextracti32x4 instead of vextracti128
                    vpaddd(xmm25, xmm25, xmm26);
                    vpshufd(xmm26, xmm25, 0x4E);
                    vpaddd(xmm25, xmm25, xmm26);
                    vpshufd(xmm26, xmm25, 0xB1);
                    vpaddd(xmm25, xmm25, xmm26);
                    vmovd(eax, xmm25);
                    mov(ptr[reg_tmp + b * 36 + 2], ax); // Store sum_qs (INT16)

                    // Store quantized values (use vmovdqu32 for extended register ymm20)
                    vmovdqu32(ptr[reg_tmp + b * 36 + 4], ymm20);
                }

                // Next M
                inc(reg_m);
                jmp(loop_M, T_NEAR);

                L(end_M);

                // Restore stack (use saved allocation size)
                add(rsp, stack_alloc_size_);

                // Epilogue
                pop(r15);
                pop(r14);
                pop(r13);
                pop(r12);
                pop(rbp);
                pop(rbx);

                ret();
            }
        };

        /**
         * @brief C++ reference implementation for testing
         */
        inline void fused_q8_1_attention_reference(
            const Q8_1Block *Q, const Q8_1Block *K, const Q8_1Block *V,
            Q8_1Block *output,
            int M, int N, int head_dim,
            int Q_stride_blocks, int K_stride_blocks, int V_stride_blocks, int out_stride_blocks,
            float scale, const float *mask = nullptr, int mask_stride = 0)
        {
            const int num_blocks = head_dim / 32;

            for (int m = 0; m < M; ++m)
            {
                const Q8_1Block *Q_row = Q + m * Q_stride_blocks;

                // Initialize online softmax state
                float max_score = -std::numeric_limits<float>::infinity();
                float sum_exp = 0.0f;
                std::vector<float> context(head_dim, 0.0f);

                for (int n = 0; n < N; ++n)
                {
                    const Q8_1Block *K_row = K + n * K_stride_blocks;
                    const Q8_1Block *V_row = V + n * V_stride_blocks;

                    // Compute dot product Q[m] · K[n]
                    float dot = 0.0f;
                    for (int b = 0; b < num_blocks; ++b)
                    {
                        float d_q = simd::fp16_to_fp32(Q_row[b].d);
                        float d_k = simd::fp16_to_fp32(K_row[b].d);

                        if (m == 0 && n == 0 && b == 0)
                        {
                            printf("[REF] m=0 n=0 b=0 d_q=%.6f d_k=%.6f\n", d_q, d_k);
                            printf("[REF] Q[0:8]: ");
                            for (int i = 0; i < 8; ++i)
                                printf("%.6f ", d_q * Q_row[b].qs[i]);
                            printf("\n");
                            printf("[REF] K[0:8]: ");
                            for (int i = 0; i < 8; ++i)
                                printf("%.6f ", d_k * K_row[b].qs[i]);
                            printf("\n");
                        }

                        int32_t acc = 0;
                        for (int i = 0; i < 32; ++i)
                        {
                            acc += static_cast<int32_t>(Q_row[b].qs[i]) * static_cast<int32_t>(K_row[b].qs[i]);
                        }
                        dot += static_cast<float>(acc) * d_q * d_k;
                    }

                    // Apply attention scale (standard attention, no Q/K normalization)
                    // score = (Q · K) * scale
                    float score = dot * scale;

                    if (m == 0 && n < 5)
                    {
                        printf("[REF] m=%d n=%d dot=%.4f scale=%.4f score=%.4f mask=%.4f\n",
                               m, n, dot, scale, score, mask ? mask[m * mask_stride + n] : 0.0f);
                    }

                    if (mask)
                    {
                        score += mask[m * mask_stride + n];
                    }

                    // Online softmax update
                    if (score > max_score)
                    {
                        float correction = std::exp(max_score - score);
                        sum_exp *= correction;
                        for (int d = 0; d < head_dim; ++d)
                        {
                            context[d] *= correction;
                        }
                        max_score = score;
                    }

                    float weight = std::exp(score - max_score);
                    sum_exp += weight;

                    if (m == 0 && n < 5)
                    {
                        printf("[REF] m=%d n=%d max_score=%.4f weight=%.4f sum_exp=%.4f\n",
                               m, n, max_score, weight, sum_exp);
                    }

                    // Accumulate weighted V
                    for (int b = 0; b < num_blocks; ++b)
                    {
                        float d_v = simd::fp16_to_fp32(V_row[b].d);
                        for (int i = 0; i < 32; ++i)
                        {
                            float v_val = static_cast<float>(V_row[b].qs[i]) * d_v;
                            context[b * 32 + i] += weight * v_val;
                        }
                    }
                }

                // Normalize
                float inv_sum = 1.0f / sum_exp;
                for (int d = 0; d < head_dim; ++d)
                {
                    context[d] *= inv_sum;
                }

                // Quantize to Q8_1
                Q8_1Block *out_row = output + m * out_stride_blocks;
                for (int b = 0; b < num_blocks; ++b)
                {
                    float max_abs = 0.0f;
                    for (int i = 0; i < 32; ++i)
                    {
                        max_abs = std::max(max_abs, std::abs(context[b * 32 + i]));
                    }

                    float d = max_abs / 127.0f;
                    float inv_d = (d > 1e-10f) ? (1.0f / d) : 0.0f;

                    out_row[b].d = simd::fp32_to_fp16(d);

                    int16_t sum_qs = 0;
                    for (int i = 0; i < 32; ++i)
                    {
                        int8_t q = static_cast<int8_t>(std::round(std::clamp(context[b * 32 + i] * inv_d, -127.0f, 127.0f)));
                        out_row[b].qs[i] = q;
                        sum_qs += q;
                    }
                    out_row[b].sum_qs = sum_qs;
                }
            }
        }

    } // namespace gemm_v4
} // namespace llaminar2
