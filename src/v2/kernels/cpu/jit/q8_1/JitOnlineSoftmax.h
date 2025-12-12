/**
 * @file JitOnlineSoftmax.h
 * @brief JIT Microkernel μK2: Online softmax state management
 * @author David Sanftenberg
 * @date December 2025
 *
 * JIT-generated online softmax for streaming attention computation.
 * Mirrors the reference OnlineSoftmax.h microkernel.
 *
 * Algorithm (from "Online normalizer calculation for softmax"):
 *   State: (max, sum_exp)
 *   
 *   update(score):
 *     if score > max:
 *       correction = exp(max - score)
 *       sum_exp *= correction
 *       max = score
 *     weight = exp(score - max)
 *     sum_exp += weight
 *     return weight
 *
 * This enables single-pass attention without materializing the full NxN score matrix.
 *
 * Register conventions:
 * - State: zmm_max(), zmm_sum() hold running state (broadcast scalars)
 * - Input: score in XMM (scalar)
 * - Output: weight in zmm_weight() (broadcast)
 * - Scratch: zmm20-25 freely clobbered
 */

#pragma once

#include "JitMicrokernelBase.h"
#include "JitFastExp.h"

namespace llaminar::v2::kernels::jit {

/**
 * @brief JIT code emitter for online softmax update
 *
 * Emits code for a single online softmax update step.
 * Maintains running (max, sum) state across iterations.
 */
class JitOnlineSoftmaxEmitter {
public:
    JitOnlineSoftmaxEmitter() = default;
    
    /**
     * @brief Emit code to initialize softmax state
     *
     * Sets:
     * - zmm_max() = -infinity
     * - zmm_sum() = 0
     *
     * Prerequisites:
     * - zmm_neg_inf() initialized to -infinity
     *
     * @param gen Code generator
     */
    void emit_init_state(JitMicrokernelBase& gen) {
        using namespace Xbyak;
        
        gen.debug_emit("emit_init_softmax_state");
        
        // max = -inf
        gen.vmovaps(gen.zmm_max(), gen.zmm_neg_inf());
        
        // sum = 0
        gen.vxorps(gen.zmm_sum(), gen.zmm_sum(), gen.zmm_sum());
    }
    
    /**
     * @brief Emit online softmax update: update state with new score, return weight
     *
     * This is the core online softmax operation:
     *   if score > max:
     *     correction = exp(max - score)
     *     sum *= correction
     *     [caller rescales context by correction]
     *     max = score
     *   weight = exp(score - max)
     *   sum += weight
     *
     * Prerequisites:
     * - zmm_max(), zmm_sum() contain current state
     * - zmm_log2e(), zmm_exp_min() initialized for fast_exp
     * - zmm_neg_inf() initialized
     *
     * Output:
     * - zmm_weight() contains weight = exp(score - max) (broadcast)
     * - zmm_corr() contains correction factor (valid only if rescale_needed)
     * - zmm_max(), zmm_sum() updated
     *
     * @param gen Code generator
     * @param score_xmm XMM containing scalar score (element 0)
     * @param label_prefix Unique prefix for jump labels
     * @param rescale_callback Label to jump to for context rescaling (optional)
     */
    void emit_update(
        JitMicrokernelBase& gen,
        const Xbyak::Xmm& score_xmm,
        const std::string& label_prefix
    ) {
        using namespace Xbyak;
        
        gen.debug_emit("emit_softmax_update");
        
        // Scratch ZMM for score broadcast
        Zmm zmm_score = gen.zmm_scratch(0);
        
        // Broadcast score to all lanes (needed for comparison and subtraction)
        gen.vbroadcastss(zmm_score, score_xmm);
        
        // Compare score with current max
        gen.vcomiss(score_xmm, Xmm(gen.zmm_max().getIdx()));
        
        std::string label_score_le_max = label_prefix + "_score_le_max";
        gen.jbe(label_score_le_max.c_str(), Xbyak::CodeGenerator::T_NEAR);
        
        // score > max: need to rescale
        gen.debug_emit("  score > max branch");
        {
            // correction = exp(max - score)
            gen.vsubps(gen.zmm_corr(), gen.zmm_max(), zmm_score);
            exp_emitter_.emit_fast_exp(gen, gen.zmm_corr(), gen.zmm_corr());
            
            // sum *= correction
            gen.vmulps(gen.zmm_sum(), gen.zmm_sum(), gen.zmm_corr());
            
            // max = score
            gen.vmovaps(gen.zmm_max(), zmm_score);
            
            // Note: Caller must rescale context accumulators by zmm_corr()
            // This is done in the composed kernel, not here
        }
        
        gen.L(label_score_le_max.c_str());
        
        // weight = exp(score - max)
        gen.vsubps(gen.zmm_weight(), zmm_score, gen.zmm_max());
        exp_emitter_.emit_fast_exp(gen, gen.zmm_weight(), gen.zmm_weight());
        
        // sum += weight
        gen.vaddps(gen.zmm_sum(), gen.zmm_sum(), gen.zmm_weight());
    }
    
    /**
     * @brief Emit code to compute final normalization factor
     *
     * Computes 1/sum for final context normalization.
     *
     * @param gen Code generator
     * @param dst_zmm ZMM to receive 1/sum (broadcast)
     */
    void emit_finalize(
        JitMicrokernelBase& gen,
        const Xbyak::Zmm& dst_zmm
    ) {
        using namespace Xbyak;
        
        gen.debug_emit("emit_softmax_finalize");
        
        // inv_sum = 1.0 / sum
        // Use precise division (not approximate vrcp14ps)
        gen.vdivps(dst_zmm, gen.zmm_one(), gen.zmm_sum());
    }
    
private:
    JitFastExpEmitter exp_emitter_;
};

/**
 * @brief Standalone JIT kernel for online softmax (for testing)
 *
 * Processes an array of scores and outputs softmax weights.
 *
 * Function signature:
 *   void kernel(const float* scores, float* weights, int n)
 */
class JitOnlineSoftmaxKernel : public JitMicrokernelBase {
public:
    using kernel_func_t = void (*)(const float* scores, float* weights, int n);
    
    explicit JitOnlineSoftmaxKernel(bool debug = false)
        : JitMicrokernelBase(8 * 1024, debug)
    {
        generate();
    }
    
    kernel_func_t get_kernel() {
        return getCode<kernel_func_t>();
    }
    
private:
    JitOnlineSoftmaxEmitter softmax_emitter_;
    
    void generate() {
        using namespace Xbyak;
        
        debug_emit("JitOnlineSoftmaxKernel::generate()");
        
        // Function: void kernel(const float* scores, float* weights, int n)
        // Args: rdi = scores, rsi = weights, edx = n
        const Reg64& reg_scores = rdi;
        const Reg64& reg_weights = rsi;
        const Reg64& reg_n = rdx;
        const Reg64& reg_i = rcx;
        
        // Save callee-saved (none needed for this simple kernel)
        
        // Initialize constants
        debug_emit("  Init constants");
        emit_broadcast_fp32_const(zmm_neg_inf(), -std::numeric_limits<float>::infinity(), rax);
        emit_broadcast_fp32_const(zmm_one(), 1.0f, rax);
        emit_broadcast_fp32_const(zmm_log2e(), 1.4426950408889634f, rax);
        emit_broadcast_fp32_const(zmm_exp_min(), -87.0f, rax);
        
        // Allocate stack for temporary weight storage
        // We need to store weights during first pass, then normalize
        // Stack: [weights_tmp: n * 4 bytes]
        // For simplicity, cap at 1024 elements
        const int max_n = 1024;
        sub(rsp, max_n * 4 + 64);  // +64 for alignment
        
        // Initialize softmax state
        softmax_emitter_.emit_init_state(*this);
        
        // Pass 1: Compute online softmax weights (unnormalized)
        debug_emit("  Pass 1: Compute weights");
        xor_(reg_i, reg_i);
        
        Label loop1_start, loop1_end;
        L(loop1_start);
        cmp(reg_i, reg_n);
        jge(loop1_end, T_NEAR);
        
        // Load score
        vmovss(xmm0, ptr[reg_scores + reg_i * 4]);
        
        // Update softmax state (unique label per iteration not needed, use index)
        // For loop, we need unique labels - use a counter approach
        // Actually, the label is inside emit_update, so we pass a unique prefix
        // Use reg_i value for uniqueness is tricky in JIT...
        // Simpler: use a single label scheme with je/jne flags
        // For this test kernel, inline the update logic
        
        // Inline softmax update (without label issues)
        // This duplicates emit_update but avoids label collision in loop
        {
            Zmm zmm_score = zmm_scratch(0);
            vbroadcastss(zmm_score, xmm0);
            
            // Compare score with max
            vcomiss(xmm0, Xmm(zmm_max().getIdx()));
            
            Label score_le_max;
            jbe(score_le_max, T_NEAR);
            
            // score > max: rescale
            vsubps(zmm_corr(), zmm_max(), zmm_score);
            JitFastExpEmitter().emit_fast_exp(*this, zmm_corr(), zmm_corr());
            vmulps(zmm_sum(), zmm_sum(), zmm_corr());
            vmovaps(zmm_max(), zmm_score);
            
            L(score_le_max);
            
            // weight = exp(score - max)
            vsubps(zmm_weight(), zmm_score, zmm_max());
            JitFastExpEmitter().emit_fast_exp(*this, zmm_weight(), zmm_weight());
            
            // sum += weight
            vaddps(zmm_sum(), zmm_sum(), zmm_weight());
        }
        
        // Store unnormalized weight to stack
        vmovss(ptr[rsp + reg_i * 4], Xmm(zmm_weight().getIdx()));
        
        inc(reg_i);
        jmp(loop1_start, T_NEAR);
        
        L(loop1_end);
        
        // Compute 1/sum
        debug_emit("  Compute 1/sum");
        Zmm zmm_inv_sum = zmm_scratch(3);
        softmax_emitter_.emit_finalize(*this, zmm_inv_sum);
        
        // Pass 2: Normalize weights
        debug_emit("  Pass 2: Normalize");
        xor_(reg_i, reg_i);
        
        Label loop2_start, loop2_end;
        L(loop2_start);
        cmp(reg_i, reg_n);
        jge(loop2_end, T_NEAR);
        
        // Load unnormalized weight from stack
        vmovss(xmm0, ptr[rsp + reg_i * 4]);
        
        // Normalize
        vmulss(xmm0, xmm0, Xmm(zmm_inv_sum.getIdx()));
        
        // Store to output
        vmovss(ptr[reg_weights + reg_i * 4], xmm0);
        
        inc(reg_i);
        jmp(loop2_start, T_NEAR);
        
        L(loop2_end);
        
        // Restore stack
        add(rsp, max_n * 4 + 64);
        ret();
    }
};

}  // namespace llaminar::v2::kernels::jit
