/**
 * @file JitVWeightedAccum.h
 * @brief JIT Microkernel μK3: Weighted V accumulation
 * @author David Sanftenberg
 * @date December 2025
 *
 * JIT-generated weighted V accumulation for attention context computation.
 * Mirrors the reference VWeightedAccum.h microkernel.
 *
 * Algorithm:
 *   For each block b in [0, num_blocks):
 *     d_v = fp16_to_fp32(V[b].d)
 *     v_dequant[0:31] = V[b].qs[0:31] * d_v
 *     context[b*32 : (b+1)*32] += weight * v_dequant
 *
 * Register conventions:
 * - Input: V block pointer in GP reg, weight in zmm_weight()
 * - Output: Context accumulators in zmm_accum(0..N) or stack
 * - Scratch: zmm8-9 for V dequantization, zmm15 for scale
 */

#pragma once

#include "JitMicrokernelBase.h"

namespace llaminar::v2::kernels::jit {

/**
 * @brief JIT code emitter for weighted V accumulation
 *
 * Emits code for: context += weight * dequant(V)
 * Called once per K/V position in the attention inner loop.
 */
class JitVWeightedAccumEmitter {
public:
    /**
     * @brief Emit weighted V accumulation for one V row
     *
     * For each block, dequantizes V values and accumulates weighted sum
     * into context registers/stack.
     *
     * Prerequisites:
     * - zmm_weight() contains softmax weight (broadcast)
     *
     * Context storage:
     * - Blocks 0-1: zmm_accum(0-3) for 64 floats
     * - Blocks 2+: Spilled to stack
     *
     * Clobbers:
     * - zmm8, zmm9 (V dequantization)
     * - zmm15 (V scale)
     * - zmm_scratch(0-1) for spilled block load/store
     *
     * @param gen Code generator
     * @param reg_V_ptr GP register pointing to V blocks for this position
     * @param num_blocks Number of Q8_1 blocks (head_dim / 32)
     * @param spill_base_offset Stack offset for spilled context accumulators
     */
    void emit_weighted_accum(
        JitMicrokernelBase& gen,
        const Xbyak::Reg64& reg_V_ptr,
        int num_blocks,
        int spill_base_offset
    ) {
        using namespace Xbyak;
        
        gen.debug_emit("emit_v_weighted_accum (" + std::to_string(num_blocks) + " blocks)");
        
        // Register aliases
        Zmm zmm_v_lo = gen.zmm_input(0);    // zmm8: V values 0-15 (dequantized)
        Zmm zmm_v_hi = gen.zmm_input(1);    // zmm9: V values 16-31 (dequantized)
        Zmm zmm_d_v = gen.zmm_input(7);     // zmm15: V scale
        
        for (int b = 0; b < num_blocks; ++b) {
            gen.debug_emit("  V block " + std::to_string(b));
            
            int v_offset = b * 36;  // Q8_1Block layout: 36 bytes per block
            
            // Load V scale: d_V (FP16 at offset 0)
            gen.vpbroadcastw(Xmm(zmm_d_v.getIdx()), gen.ptr[reg_V_ptr + v_offset]);
            gen.vcvtph2ps(Xmm(zmm_d_v.getIdx()), Xmm(zmm_d_v.getIdx()));
            gen.vbroadcastss(zmm_d_v, Xmm(zmm_d_v.getIdx()));
            
            // Load V data (32 int8 values at offset 4)
            // Sign-extend to int32, convert to float
            gen.vpmovsxbd(zmm_v_lo, gen.ptr[reg_V_ptr + v_offset + 4]);       // 16 int8 -> int32
            gen.vpmovsxbd(zmm_v_hi, gen.ptr[reg_V_ptr + v_offset + 4 + 16]);  // Next 16
            
            gen.vcvtdq2ps(zmm_v_lo, zmm_v_lo);  // int32 -> float
            gen.vcvtdq2ps(zmm_v_hi, zmm_v_hi);
            
            // Scale by d_V
            gen.vmulps(zmm_v_lo, zmm_v_lo, zmm_d_v);
            gen.vmulps(zmm_v_hi, zmm_v_hi, zmm_d_v);
            
            // Accumulate: context += weight * v_dequant
            // Use FMA: context = context + weight * v
            if (b == 0) {
                // Block 0: context floats 0-31 in zmm_accum(0), zmm_accum(1)
                gen.vfmadd231ps(gen.zmm_accum(0), zmm_v_lo, gen.zmm_weight());
                gen.vfmadd231ps(gen.zmm_accum(1), zmm_v_hi, gen.zmm_weight());
            } else if (b == 1) {
                // Block 1: context floats 32-63 in zmm_accum(2), zmm_accum(3)
                gen.vfmadd231ps(gen.zmm_accum(2), zmm_v_lo, gen.zmm_weight());
                gen.vfmadd231ps(gen.zmm_accum(3), zmm_v_hi, gen.zmm_weight());
            } else {
                // Spilled blocks: load from stack, FMA, store back
                int spill_lo = spill_base_offset + (b - 2) * 128;
                int spill_hi = spill_lo + 64;
                
                Zmm zmm_tmp = gen.zmm_scratch(0);
                
                // Low half
                gen.vmovups(zmm_tmp, gen.ptr[gen.rsp + spill_lo]);
                gen.vfmadd231ps(zmm_tmp, zmm_v_lo, gen.zmm_weight());
                gen.vmovups(gen.ptr[gen.rsp + spill_lo], zmm_tmp);
                
                // High half
                gen.vmovups(zmm_tmp, gen.ptr[gen.rsp + spill_hi]);
                gen.vfmadd231ps(zmm_tmp, zmm_v_hi, gen.zmm_weight());
                gen.vmovups(gen.ptr[gen.rsp + spill_hi], zmm_tmp);
            }
        }
    }
    
    /**
     * @brief Emit context rescaling by correction factor
     *
     * When softmax max increases, all previous context values must be
     * rescaled by exp(old_max - new_max).
     *
     * @param gen Code generator
     * @param num_blocks Number of blocks
     * @param spill_base_offset Stack offset for spilled accumulators
     */
    void emit_rescale_context(
        JitMicrokernelBase& gen,
        int num_blocks,
        int spill_base_offset
    ) {
        using namespace Xbyak;
        
        gen.debug_emit("emit_rescale_context");
        
        // Rescale register-resident blocks
        gen.vmulps(gen.zmm_accum(0), gen.zmm_accum(0), gen.zmm_corr());
        gen.vmulps(gen.zmm_accum(1), gen.zmm_accum(1), gen.zmm_corr());
        
        if (num_blocks >= 2) {
            gen.vmulps(gen.zmm_accum(2), gen.zmm_accum(2), gen.zmm_corr());
            gen.vmulps(gen.zmm_accum(3), gen.zmm_accum(3), gen.zmm_corr());
        }
        
        // Rescale spilled blocks
        if (num_blocks > 2) {
            Zmm zmm_tmp = gen.zmm_scratch(0);
            
            for (int b = 2; b < num_blocks; ++b) {
                int spill_lo = spill_base_offset + (b - 2) * 128;
                int spill_hi = spill_lo + 64;
                
                gen.vmovups(zmm_tmp, gen.ptr[gen.rsp + spill_lo]);
                gen.vmulps(zmm_tmp, zmm_tmp, gen.zmm_corr());
                gen.vmovups(gen.ptr[gen.rsp + spill_lo], zmm_tmp);
                
                gen.vmovups(zmm_tmp, gen.ptr[gen.rsp + spill_hi]);
                gen.vmulps(zmm_tmp, zmm_tmp, gen.zmm_corr());
                gen.vmovups(gen.ptr[gen.rsp + spill_hi], zmm_tmp);
            }
        }
    }
    
    /**
     * @brief Emit context initialization (zero all accumulators)
     *
     * @param gen Code generator
     * @param num_blocks Number of blocks
     * @param spill_base_offset Stack offset for spilled accumulators
     */
    void emit_init_context(
        JitMicrokernelBase& gen,
        int num_blocks,
        int spill_base_offset
    ) {
        using namespace Xbyak;
        
        gen.debug_emit("emit_init_context");
        
        // Zero register-resident blocks
        gen.vxorps(gen.zmm_accum(0), gen.zmm_accum(0), gen.zmm_accum(0));
        gen.vxorps(gen.zmm_accum(1), gen.zmm_accum(1), gen.zmm_accum(1));
        
        if (num_blocks >= 2) {
            gen.vxorps(gen.zmm_accum(2), gen.zmm_accum(2), gen.zmm_accum(2));
            gen.vxorps(gen.zmm_accum(3), gen.zmm_accum(3), gen.zmm_accum(3));
        }
        
        // Zero spilled blocks
        if (num_blocks > 2) {
            Zmm zmm_zero = gen.zmm_scratch(0);
            gen.vxorps(zmm_zero, zmm_zero, zmm_zero);
            
            for (int b = 2; b < num_blocks; ++b) {
                int spill_lo = spill_base_offset + (b - 2) * 128;
                int spill_hi = spill_lo + 64;
                
                gen.vmovups(gen.ptr[gen.rsp + spill_lo], zmm_zero);
                gen.vmovups(gen.ptr[gen.rsp + spill_hi], zmm_zero);
            }
        }
    }
    
    /**
     * @brief Emit context normalization by 1/sum
     *
     * @param gen Code generator
     * @param inv_sum_zmm ZMM containing 1/sum (broadcast)
     * @param num_blocks Number of blocks
     * @param spill_base_offset Stack offset for spilled accumulators
     */
    void emit_normalize_context(
        JitMicrokernelBase& gen,
        const Xbyak::Zmm& inv_sum_zmm,
        int num_blocks,
        int spill_base_offset
    ) {
        using namespace Xbyak;
        
        gen.debug_emit("emit_normalize_context");
        
        // Normalize register-resident blocks
        gen.vmulps(gen.zmm_accum(0), gen.zmm_accum(0), inv_sum_zmm);
        gen.vmulps(gen.zmm_accum(1), gen.zmm_accum(1), inv_sum_zmm);
        
        if (num_blocks >= 2) {
            gen.vmulps(gen.zmm_accum(2), gen.zmm_accum(2), inv_sum_zmm);
            gen.vmulps(gen.zmm_accum(3), gen.zmm_accum(3), inv_sum_zmm);
        }
        
        // Normalize spilled blocks
        if (num_blocks > 2) {
            Zmm zmm_tmp = gen.zmm_scratch(0);
            
            for (int b = 2; b < num_blocks; ++b) {
                int spill_lo = spill_base_offset + (b - 2) * 128;
                int spill_hi = spill_lo + 64;
                
                gen.vmovups(zmm_tmp, gen.ptr[gen.rsp + spill_lo]);
                gen.vmulps(zmm_tmp, zmm_tmp, inv_sum_zmm);
                gen.vmovups(gen.ptr[gen.rsp + spill_lo], zmm_tmp);
                
                gen.vmovups(zmm_tmp, gen.ptr[gen.rsp + spill_hi]);
                gen.vmulps(zmm_tmp, zmm_tmp, inv_sum_zmm);
                gen.vmovups(gen.ptr[gen.rsp + spill_hi], zmm_tmp);
            }
        }
    }
};

}  // namespace llaminar::v2::kernels::jit
