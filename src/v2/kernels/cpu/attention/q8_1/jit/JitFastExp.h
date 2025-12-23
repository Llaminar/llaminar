/**
 * @file JitFastExp.h
 * @brief JIT Microkernel μK5: Fast exponential approximation
 * @author David Sanftenberg
 * @date December 2025
 *
 * JIT-generated fast exp() for softmax computation.
 * Mirrors the reference FastExp.h microkernel but generates AVX-512 code at runtime.
 *
 * Algorithm: exp(x) = 2^(x * log2(e))
 * - Split into integer part n = floor(x * log2(e))
 * - Fractional part f = (x * log2(e)) - n
 * - Compute 2^f using 5th-order Taylor polynomial
 * - Scale by 2^n using vscalefps
 *
 * Register conventions (from JitMicrokernelBase.h):
 * - Input: zmm specified by caller
 * - Output: zmm specified by caller (can be same as input)
 * - Constants: zmm_log2e(), zmm_exp_min() must be initialized
 * - Scratch: zmm20-25 freely clobbered
 */

#pragma once

#include "JitMicrokernelBase.h"

namespace llaminar::v2::kernels::jit {

/**
 * @brief JIT code emitter for fast exp() approximation
 *
 * This class provides methods to emit fast exp code inline within
 * a larger JIT kernel. It does NOT generate a standalone callable function.
 *
 * Usage:
 * @code
 *   class MyKernel : public JitMicrokernelBase {
 *       JitFastExpEmitter exp_emitter_;
 *       
 *       void generate() {
 *           // ... setup code ...
 *           
 *           // Emit exp(zmm_input) -> zmm_output
 *           exp_emitter_.emit_fast_exp(*this, zmm_output, zmm_input);
 *           
 *           // ... more code ...
 *       }
 *   };
 * @endcode
 */
class JitFastExpEmitter {
public:
    /**
     * @brief Emit fast exp code: dst = exp(src)
     *
     * Prerequisites:
     * - zmm_log2e() initialized to log2(e) ≈ 1.4427
     * - zmm_exp_min() initialized to -87.0f
     *
     * Clobbers:
     * - zmm_scratch(0-5) (zmm20-25)
     * - rax (for constant loading)
     *
     * @param gen Code generator to emit into
     * @param dst Output ZMM register
     * @param src Input ZMM register (can be same as dst)
     */
    void emit_fast_exp(
        JitMicrokernelBase& gen,
        const Xbyak::Zmm& dst,
        const Xbyak::Zmm& src
    ) {
        using namespace Xbyak;
        
        gen.debug_emit("emit_fast_exp");
        
        // Clamp input to avoid overflow/underflow
        // x_clamped = max(src, -87.0f)
        gen.vmaxps(dst, src, gen.zmm_exp_min());
        
        // x * log2(e)
        gen.vmulps(dst, dst, gen.zmm_log2e());
        
        // Compute 2^(x * log2e) using polynomial
        emit_exp2_poly(gen, dst, dst);
    }
    
    /**
     * @brief Emit 2^x polynomial approximation
     *
     * Uses range reduction: 2^x = 2^n * 2^f where n = floor(x), f = x - n
     * Then computes 2^f using 5th-order Taylor series.
     *
     * @param gen Code generator
     * @param dst Output ZMM
     * @param src Input ZMM (can be same as dst)
     */
    void emit_exp2_poly(
        JitMicrokernelBase& gen,
        const Xbyak::Zmm& dst,
        const Xbyak::Zmm& src
    ) {
        using namespace Xbyak;
        
        gen.debug_emit("  emit_exp2_poly");
        
        // Scratch registers
        Zmm zmm_n = gen.zmm_scratch(0);      // Integer part
        Zmm zmm_f = gen.zmm_scratch(1);      // Fractional part
        Zmm zmm_tmp = gen.zmm_scratch(2);    // Temp for Horner evaluation
        
        // Split into integer and fractional parts
        // n = floor(x), f = x - n
        gen.vrndscaleps(zmm_n, src, 1);      // floor(x)
        gen.vsubps(zmm_f, src, zmm_n);       // f = x - floor(x)
        
        // 2^f using 5th-order Taylor series (Horner's method)
        // p(f) = c5*f^5 + c4*f^4 + c3*f^3 + c2*f^2 + c1*f + c0
        //      = ((((c5*f + c4)*f + c3)*f + c2)*f + c1)*f + c0
        
        // Taylor coefficients for 2^x around x=0:
        // c0 = 1.0, c1 = ln(2), c2 = ln(2)^2/2, c3 = ln(2)^3/6, ...
        static constexpr float c0 = 1.0f;
        static constexpr float c1 = 0.6931472f;    // ln(2)
        static constexpr float c2 = 0.2402265f;    // ln(2)^2/2
        static constexpr float c3 = 0.0555041f;    // ln(2)^3/6
        static constexpr float c4 = 0.0096181f;    // ln(2)^4/24
        static constexpr float c5 = 0.0013334f;    // ln(2)^5/120
        
        // Start Horner: dst = c5
        gen.emit_broadcast_fp32_const(dst, c5, gen.rax);
        
        // dst = c5*f + c4
        gen.emit_broadcast_fp32_const(zmm_tmp, c4, gen.rax);
        gen.vfmadd213ps(dst, zmm_f, zmm_tmp);
        
        // dst = (c5*f + c4)*f + c3
        gen.emit_broadcast_fp32_const(zmm_tmp, c3, gen.rax);
        gen.vfmadd213ps(dst, zmm_f, zmm_tmp);
        
        // dst = ((c5*f + c4)*f + c3)*f + c2
        gen.emit_broadcast_fp32_const(zmm_tmp, c2, gen.rax);
        gen.vfmadd213ps(dst, zmm_f, zmm_tmp);
        
        // dst = (((c5*f + c4)*f + c3)*f + c2)*f + c1
        gen.emit_broadcast_fp32_const(zmm_tmp, c1, gen.rax);
        gen.vfmadd213ps(dst, zmm_f, zmm_tmp);
        
        // dst = ((((c5*f + c4)*f + c3)*f + c2)*f + c1)*f + c0
        gen.emit_broadcast_fp32_const(zmm_tmp, c0, gen.rax);
        gen.vfmadd213ps(dst, zmm_f, zmm_tmp);
        
        // Scale by 2^n using vscalefps
        // vscalefps: dst[i] = dst[i] * 2^floor(zmm_n[i])
        gen.vscalefps(dst, dst, zmm_n);
    }
};

/**
 * @brief Standalone JIT kernel for fast exp (for testing)
 *
 * Generates a callable function: void kernel(const float* in, float* out, int n)
 * Processes n floats (must be multiple of 16).
 */
class JitFastExpKernel : public JitMicrokernelBase {
public:
    using kernel_func_t = void (*)(const float* in, float* out, int n);
    
    explicit JitFastExpKernel(bool debug = false)
        : JitMicrokernelBase(4 * 1024, debug)
    {
        generate();
    }
    
    kernel_func_t get_kernel() {
        return getCode<kernel_func_t>();
    }
    
private:
    JitFastExpEmitter exp_emitter_;
    
    void generate() {
        using namespace Xbyak;
        
        debug_emit("JitFastExpKernel::generate()");
        
        // Function: void kernel(const float* in, float* out, int n)
        // Args: rdi = in, rsi = out, rdx = n
        const Reg64& reg_in = rdi;
        const Reg64& reg_out = rsi;
        const Reg64& reg_n = rdx;
        const Reg64& reg_i = rcx;
        
        // Initialize constants
        debug_emit("  Init constants");
        emit_broadcast_fp32_const(zmm_log2e(), 1.4426950408889634f, rax);
        emit_broadcast_fp32_const(zmm_exp_min(), -87.0f, rax);
        
        // Loop: for (i = 0; i < n; i += 16)
        xor_(reg_i, reg_i);
        
        Label loop_start, loop_end;
        L(loop_start);
        cmp(reg_i, reg_n);
        jge(loop_end, T_NEAR);
        
        // Load 16 floats
        vmovups(zmm_input(0), ptr[reg_in + reg_i * 4]);
        
        // Compute exp
        exp_emitter_.emit_fast_exp(*this, zmm_accum(0), zmm_input(0));
        
        // Store 16 floats
        vmovups(ptr[reg_out + reg_i * 4], zmm_accum(0));
        
        // i += 16
        add(reg_i, 16);
        jmp(loop_start, T_NEAR);
        
        L(loop_end);
        ret();
    }
};

}  // namespace llaminar::v2::kernels::jit
