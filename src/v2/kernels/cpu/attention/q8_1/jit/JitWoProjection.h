/**
 * @file JitWoProjection.h
 * @brief JIT Microkernel μK4: Output projection (Wo)
 * @author David Sanftenberg
 * @date December 2025
 *
 * JIT-generated output projection for fused attention-Wo operation.
 * Supports multiple Wo weight formats:
 *   - FP32: Direct matmul
 *   - Q8_1: Quantized matmul with Q8_1 dequantization
 *   - FP16: Half-precision matmul
 *   - BF16: BFloat16 matmul
 *
 * Algorithm:
 *   output[q, h*head_dim : (h+1)*head_dim] += context[h, :] * Wo[d_out_start:d_out_end, h*head_dim:(h+1)*head_dim]
 *
 * The projection uses tiled matmul for cache efficiency:
 * - Context tile: head_dim floats (usually 64)
 * - Wo tile: head_dim × d_out_tile
 */

#pragma once

#include "../../../jit/JitMicrokernelBase.h"
#include "../../../jit/RegisterAllocation.h"
#include "../../../jit/RegisterEnforcement.h"
#include <cstdint>

// Import typed register aliases and C++20 concepts
using namespace llaminar2::jit;

namespace llaminar::v2::kernels::jit
{
    // ============================================================================
    // WoProjection Register Manifest (compile-time documentation)
    // ============================================================================
    // Documents the registers borrowed by emit_wo_projection_* methods.
    // Uses Accum zone for FMA accumulation during projection.

    using WoProjectionManifest = KernelRegisterManifest<
        Accum0, Accum1, Accum2, Accum3, // FMA accumulators
        Scratch4, Scratch5              // Intermediate scratch
        >;
    static_assert(!WoProjectionManifest::has_conflicts(),
                  "WoProjection register manifest has conflicts!");

    /**
     * @brief Weight format for Wo projection
     */
    enum class WoFormat
    {
        FP32,
        Q8_1,
        Q8_1_VNNI_PACKED,
        FP16,
        BF16,
        // Hybrid mode: VNNI-packed weights with FP32 streaming dequantization
        // This gives highest precision by avoiding activation quantization
        FP32_STREAMING_DEQUANT
    };

    /**
     * @brief JIT code emitter for Wo projection
     *
     * Emits code for: output += context * Wo
     * Handles different weight formats with specialized code paths.
     */
    class JitWoProjectionEmitter
    {
    public:
        /**
         * @brief Emit FP32 Wo projection
         *
         * Simple dot product of context with Wo row.
         *
         * @param gen Code generator
         * @param reg_context_ptr Pointer to context vector (head_dim floats)
         * @param reg_wo_ptr Pointer to Wo row (head_dim floats)
         * @param reg_output_ptr Pointer to output location (1 float)
         * @param head_dim Dimension of attention head
         */
        void emit_project_fp32(
            JitMicrokernelBase &gen,
            const Xbyak::Reg64 &reg_context_ptr,
            const Xbyak::Reg64 &reg_wo_ptr,
            const Xbyak::Reg64 &reg_output_ptr,
            int head_dim)
        {
            using namespace Xbyak;

            gen.debug_emit("emit_project_fp32 (head_dim=" + std::to_string(head_dim) + ")");

            // Use typed scratch registers for accumulation
            Zmm zmm_acc = gen.borrow<llaminar2::jit::Scratch0>().zmm();
            gen.vxorps(zmm_acc, zmm_acc, zmm_acc);

            // Process 16 floats at a time
            int i = 0;
            for (; i + 16 <= head_dim; i += 16)
            {
                Zmm zmm_ctx = gen.borrow<llaminar2::jit::Scratch1>().zmm();
                Zmm zmm_wo = gen.borrow<llaminar2::jit::Scratch2>().zmm();

                gen.vmovups(zmm_ctx, gen.ptr[reg_context_ptr + i * 4]);
                gen.vmovups(zmm_wo, gen.ptr[reg_wo_ptr + i * 4]);
                gen.vfmadd231ps(zmm_acc, zmm_ctx, zmm_wo);
            }

            // Horizontal sum
            emit_horizontal_sum_zmm(gen, zmm_acc);

            // Store result (accumulate to output)
            gen.vaddss(Xmm(zmm_acc.getIdx()), Xmm(zmm_acc.getIdx()), gen.ptr[reg_output_ptr]);
            gen.vmovss(gen.ptr[reg_output_ptr], Xmm(zmm_acc.getIdx()));
        }

        /**
         * @brief Emit Q8_1 Wo projection
         *
         * Quantizes context to Q8_1, then performs integer dot product with Q8_1 Wo.
         *
         * @param gen Code generator
         * @param reg_context_ptr Pointer to context vector (head_dim floats)
         * @param reg_wo_ptr Pointer to Wo Q8_1 blocks (head_dim/32 blocks)
         * @param reg_output_ptr Pointer to output location (1 float)
         * @param head_dim Dimension of attention head (must be multiple of 32)
         */
        void emit_project_q8_1(
            JitMicrokernelBase &gen,
            const Xbyak::Reg64 &reg_context_ptr,
            const Xbyak::Reg64 &reg_wo_ptr,
            const Xbyak::Reg64 &reg_output_ptr,
            int head_dim)
        {
            using namespace Xbyak;

            gen.debug_emit("emit_project_q8_1 (head_dim=" + std::to_string(head_dim) + ")");

            int num_blocks = head_dim / 32;

            // Accumulator for final result
            Zmm zmm_acc_int = gen.borrow<llaminar2::jit::Scratch0>().zmm(); // Integer accumulator
            gen.vxorps(zmm_acc_int, zmm_acc_int, zmm_acc_int);

            Zmm zmm_scale_sum = gen.borrow<llaminar2::jit::Scratch1>().zmm(); // Scale * sum correction
            gen.vxorps(zmm_scale_sum, zmm_scale_sum, zmm_scale_sum);

            for (int b = 0; b < num_blocks; ++b)
            {
                emit_q8_1_block_projection(gen, reg_context_ptr, reg_wo_ptr,
                                           zmm_acc_int, zmm_scale_sum, b);
            }

            // Final dequantization:
            // result = acc_int * d_ctx * d_wo - scale_sum
            // (simplified - full implementation needs proper scaling)

            // For now, use simpler approach: just horizontal sum the integer accum
            // and convert to float
            emit_horizontal_sum_zmm(gen, zmm_acc_int);

            // Store result
            gen.vaddss(Xmm(zmm_acc_int.getIdx()), Xmm(zmm_acc_int.getIdx()), gen.ptr[reg_output_ptr]);
            gen.vmovss(gen.ptr[reg_output_ptr], Xmm(zmm_acc_int.getIdx()));
        }

        /**
         * @brief Emit FP16 Wo projection
         *
         * Converts FP16 weights to FP32, then dot product.
         *
         * @param gen Code generator
         * @param reg_context_ptr Pointer to context vector (head_dim floats)
         * @param reg_wo_ptr Pointer to Wo row (head_dim FP16 values)
         * @param reg_output_ptr Pointer to output location (1 float)
         * @param head_dim Dimension of attention head
         */
        void emit_project_fp16(
            JitMicrokernelBase &gen,
            const Xbyak::Reg64 &reg_context_ptr,
            const Xbyak::Reg64 &reg_wo_ptr,
            const Xbyak::Reg64 &reg_output_ptr,
            int head_dim)
        {
            using namespace Xbyak;

            gen.debug_emit("emit_project_fp16 (head_dim=" + std::to_string(head_dim) + ")");

            Zmm zmm_acc = gen.borrow<llaminar2::jit::Scratch0>().zmm();
            gen.vxorps(zmm_acc, zmm_acc, zmm_acc);

            // Process 16 floats at a time
            for (int i = 0; i + 16 <= head_dim; i += 16)
            {
                Zmm zmm_ctx = gen.borrow<llaminar2::jit::Scratch1>().zmm();
                Zmm zmm_wo = gen.borrow<llaminar2::jit::Scratch2>().zmm();
                Ymm ymm_fp16 = Ymm(zmm_wo.getIdx());

                // Load context (FP32)
                gen.vmovups(zmm_ctx, gen.ptr[reg_context_ptr + i * 4]);

                // Load Wo (FP16), convert to FP32
                gen.vmovups(ymm_fp16, gen.ptr[reg_wo_ptr + i * 2]);
                gen.vcvtph2ps(zmm_wo, ymm_fp16);

                // FMA
                gen.vfmadd231ps(zmm_acc, zmm_ctx, zmm_wo);
            }

            // Horizontal sum
            emit_horizontal_sum_zmm(gen, zmm_acc);

            // Store result
            gen.vaddss(Xmm(zmm_acc.getIdx()), Xmm(zmm_acc.getIdx()), gen.ptr[reg_output_ptr]);
            gen.vmovss(gen.ptr[reg_output_ptr], Xmm(zmm_acc.getIdx()));
        }

        /**
         * @brief Emit BF16 Wo projection
         *
         * Converts BF16 weights to FP32 (left-shift by 16), then dot product.
         *
         * @param gen Code generator
         * @param reg_context_ptr Pointer to context vector (head_dim floats)
         * @param reg_wo_ptr Pointer to Wo row (head_dim BF16 values)
         * @param reg_output_ptr Pointer to output location (1 float)
         * @param head_dim Dimension of attention head
         */
        void emit_project_bf16(
            JitMicrokernelBase &gen,
            const Xbyak::Reg64 &reg_context_ptr,
            const Xbyak::Reg64 &reg_wo_ptr,
            const Xbyak::Reg64 &reg_output_ptr,
            int head_dim)
        {
            using namespace Xbyak;

            gen.debug_emit("emit_project_bf16 (head_dim=" + std::to_string(head_dim) + ")");

            Zmm zmm_acc = gen.borrow<llaminar2::jit::Scratch0>().zmm();
            gen.vxorps(zmm_acc, zmm_acc, zmm_acc);

            // Process 16 floats at a time
            for (int i = 0; i + 16 <= head_dim; i += 16)
            {
                Zmm zmm_ctx = gen.borrow<llaminar2::jit::Scratch1>().zmm();
                Zmm zmm_wo = gen.borrow<llaminar2::jit::Scratch2>().zmm();
                Ymm ymm_bf16 = Ymm(zmm_wo.getIdx());

                // Load context (FP32)
                gen.vmovups(zmm_ctx, gen.ptr[reg_context_ptr + i * 4]);

                // Load Wo (BF16 stored as uint16), convert to FP32
                // BF16 -> FP32: zero-extend to 32-bit, then shift left by 16
                gen.vpmovzxwd(zmm_wo, gen.ptr[reg_wo_ptr + i * 2]); // Zero-extend u16 -> u32
                gen.vpslld(zmm_wo, zmm_wo, 16);                     // Shift left by 16 bits

                // FMA
                gen.vfmadd231ps(zmm_acc, zmm_ctx, zmm_wo);
            }

            // Horizontal sum
            emit_horizontal_sum_zmm(gen, zmm_acc);

            // Store result
            gen.vaddss(Xmm(zmm_acc.getIdx()), Xmm(zmm_acc.getIdx()), gen.ptr[reg_output_ptr]);
            gen.vmovss(gen.ptr[reg_output_ptr], Xmm(zmm_acc.getIdx()));
        }

        /**
         * @brief Emit batch Wo projection row
         *
         * Processes one output row (d_model dimension) for all heads in batch.
         * This is the inner kernel for the output projection matrix multiply.
         *
         * @param gen Code generator
         * @param format Weight format
         * @param reg_context_ptr Pointer to context buffer (batch × num_heads × head_dim)
         * @param reg_wo_ptr Pointer to Wo row
         * @param reg_output_ptr Pointer to output row
         * @param batch_size Number of sequences
         * @param num_heads Number of attention heads
         * @param head_dim Dimension per head
         */
        void emit_batch_project_row(
            JitMicrokernelBase &gen,
            WoFormat format,
            const Xbyak::Reg64 &reg_context_ptr,
            const Xbyak::Reg64 &reg_wo_ptr,
            const Xbyak::Reg64 &reg_output_ptr,
            int batch_size,
            int num_heads,
            int head_dim)
        {
            using namespace Xbyak;

            gen.debug_emit("emit_batch_project_row (format=" + std::to_string(static_cast<int>(format)) + ")");

            // For each batch element
            for (int b = 0; b < batch_size; ++b)
            {
                // For each head
                for (int h = 0; h < num_heads; ++h)
                {
                    int ctx_offset = (b * num_heads + h) * head_dim * 4; // FP32 context
                    int wo_offset = h * head_dim;                        // Wo layout depends on format
                    int out_offset = b * sizeof(float);                  // Output accumulator

                    Reg64 reg_ctx_tmp = gen.r14;
                    Reg64 reg_wo_tmp = gen.r15;

                    gen.lea(reg_ctx_tmp, gen.ptr[reg_context_ptr + ctx_offset]);

                    switch (format)
                    {
                    case WoFormat::FP32:
                        gen.lea(reg_wo_tmp, gen.ptr[reg_wo_ptr + wo_offset * 4]);
                        emit_project_fp32(gen, reg_ctx_tmp, reg_wo_tmp, reg_output_ptr, head_dim);
                        break;
                    case WoFormat::Q8_1:
                        gen.lea(reg_wo_tmp, gen.ptr[reg_wo_ptr + (wo_offset / 32) * 36]);
                        emit_project_q8_1(gen, reg_ctx_tmp, reg_wo_tmp, reg_output_ptr, head_dim);
                        break;
                    case WoFormat::FP16:
                        gen.lea(reg_wo_tmp, gen.ptr[reg_wo_ptr + wo_offset * 2]);
                        emit_project_fp16(gen, reg_ctx_tmp, reg_wo_tmp, reg_output_ptr, head_dim);
                        break;
                    case WoFormat::BF16:
                        gen.lea(reg_wo_tmp, gen.ptr[reg_wo_ptr + wo_offset * 2]);
                        emit_project_bf16(gen, reg_ctx_tmp, reg_wo_tmp, reg_output_ptr, head_dim);
                        break;
                    }
                }
            }
        }

    private:
        /**
         * @brief Emit horizontal sum of ZMM register
         *
         * Reduces 16 floats in ZMM to single scalar in element 0.
         * Result is in XMM[0] of the same register.
         *
         * Algorithm:
         * 1. Extract upper 256 bits, add to lower (8 floats in YMM)
         * 2. Extract upper 128 bits, add to lower (4 floats in XMM)
         * 3. haddps twice (2 floats, then 1 float)
         */
        void emit_horizontal_sum_zmm(JitMicrokernelBase &gen, const Xbyak::Zmm &zmm)
        {
            using namespace Xbyak;

            // Move input to zmm0 if it's not already there to avoid EVEX issues with high regs
            // We use zmm0 as the accumulator for reduction
            Zmm zmm_target = gen.zmm0;
            bool moved = false;
            if (zmm.getIdx() != 0)
            {
                gen.vmovaps(zmm_target, zmm);
                moved = true;
            }

            // Step 1: zmm0 -> ymm0 (add upper 256 to lower 256)
            Ymm ymm = Ymm(zmm_target.getIdx());                                      // ymm0
            Ymm ymm_hi = Ymm(gen.borrow<llaminar2::jit::Scratch5>().zmm().getIdx()); // ymm25

            gen.vextractf32x8(ymm_hi, zmm_target, 1); // Extract upper 256 bits
            gen.vaddps(ymm, ymm, ymm_hi);             // Add

            // Step 2: ymm0 -> xmm0 (add upper 128 to lower 128)
            Xmm xmm = Xmm(ymm.getIdx());       // xmm0
            Xmm xmm_hi = Xmm(ymm_hi.getIdx()); // xmm25

            // vextractf128 is VEX-only, so cannot use xmm25. Use vextractf32x4 (EVEX) instead.
            gen.vextractf32x4(xmm_hi, ymm, 1); // Extract upper 128 bits
            gen.vaddps(xmm, xmm, xmm_hi);      // Add

            // Step 3: xmm horizontal add (4 floats -> 1)
            // Now xmm is xmm0, so vhaddps is safe (VEX)
            gen.vhaddps(xmm, xmm, xmm); // a+b, c+d, a+b, c+d
            gen.vhaddps(xmm, xmm, xmm); // (a+b)+(c+d), ...

            // Move result back to zmm (xmm part) if needed
            if (moved)
            {
                gen.vmovaps(Xmm(zmm.getIdx()), xmm);
            }
        }

        /**
         * @brief Emit Q8_1 block projection
         *
         * Process one Q8_1 block (32 elements) of context × Wo.
         */
        void emit_q8_1_block_projection(
            JitMicrokernelBase &gen,
            const Xbyak::Reg64 &reg_context_ptr,
            const Xbyak::Reg64 &reg_wo_ptr,
            const Xbyak::Zmm &zmm_acc_int,
            const Xbyak::Zmm &zmm_scale_sum,
            int block_idx)
        {
            using namespace Xbyak;

            // This is a simplified Q8_1 dot product
            // Full implementation would quantize context on-the-fly and use vpdpbusd

            int ctx_offset = block_idx * 32 * 4; // 32 floats
            int wo_offset = block_idx * 36;      // Q8_1Block: 36 bytes

            // Load context block (32 floats)
            Zmm zmm_ctx_lo = gen.borrow<llaminar2::jit::Scratch2>().zmm();
            Zmm zmm_ctx_hi = gen.borrow<llaminar2::jit::Scratch3>().zmm();
            gen.vmovups(zmm_ctx_lo, gen.ptr[reg_context_ptr + ctx_offset]);
            gen.vmovups(zmm_ctx_hi, gen.ptr[reg_context_ptr + ctx_offset + 64]);

            // Load Wo scale (FP16 at offset 0)
            Zmm zmm_d_wo = gen.borrow<llaminar2::jit::Scratch4>().zmm();

            // Use xmm0 as temp for loading/conversion to avoid potential EVEX issues with high regs
            // on some instructions or Xbyak edge cases
            gen.vpbroadcastw(gen.xmm0, gen.ptr[reg_wo_ptr + wo_offset]);
            gen.vcvtph2ps(gen.xmm0, gen.xmm0);
            gen.vbroadcastss(zmm_d_wo, gen.xmm0);

            // Load Wo data (32 int8 at offset 4)
            Zmm zmm_wo_lo = gen.borrow<llaminar2::jit::Scratch5>().zmm();
            Zmm zmm_wo_hi = gen.zmm_accum0(); // Reuse accum0 since we need more regs
            gen.vpmovsxbd(zmm_wo_lo, gen.ptr[reg_wo_ptr + wo_offset + 4]);
            gen.vpmovsxbd(zmm_wo_hi, gen.ptr[reg_wo_ptr + wo_offset + 4 + 16]);

            // Convert to float
            gen.vcvtdq2ps(zmm_wo_lo, zmm_wo_lo);
            gen.vcvtdq2ps(zmm_wo_hi, zmm_wo_hi);

            // Scale by d_wo
            gen.vmulps(zmm_wo_lo, zmm_wo_lo, zmm_d_wo);
            gen.vmulps(zmm_wo_hi, zmm_wo_hi, zmm_d_wo);

            // Accumulate dot product
            gen.vfmadd231ps(zmm_acc_int, zmm_ctx_lo, zmm_wo_lo);
            gen.vfmadd231ps(zmm_acc_int, zmm_ctx_hi, zmm_wo_hi);
        }
    };

} // namespace llaminar::v2::kernels::jit
