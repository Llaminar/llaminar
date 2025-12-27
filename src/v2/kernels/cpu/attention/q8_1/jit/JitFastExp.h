/**
 * @file JitFastExp.h
 * @brief JIT Microkernel μK5: Fast exponential approximation
 * @author David Sanftenberg
 * @date December 2025
 *
 * JIT-generated fast exp() for softmax computation.
 * Used for computing attention weights: weight[i] = exp(score[i] - max)
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * ALGORITHM
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * exp(x) = 2^(x * log2(e))
 *
 * Range reduction:
 *   n = floor(x * log2(e))        // Integer part
 *   f = (x * log2(e)) - n         // Fractional part, f ∈ [0, 1)
 *
 * 2^f approximation (5th-order Taylor polynomial in Horner form):
 *   2^f ≈ c0 + f*(c1 + f*(c2 + f*(c3 + f*(c4 + f*c5))))
 *
 * Final result:
 *   exp(x) = 2^n * 2^f           // vscalefps instruction
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * REGISTER USAGE (via RegisterGuard)
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Borrowed registers (InputZone):
 *   - Input0 (zmm8):  Horner temp for loading polynomial constants
 *   - Input6 (zmm14): Integer part n = floor(x * log2e)
 *   - Input7 (zmm15): Fractional part f = (x * log2e) - n
 *
 * FA2 SAFETY:
 *   ✓ Does NOT clobber Score0-3 (xmm20-23) - may hold scores during weight compute
 *   ✓ Does NOT clobber Scratch4-5 (zmm24-25) - may hold tile_max
 *   ✓ Does NOT clobber State (zmm16-19) - softmax state must persist
 *   ✓ Does NOT clobber Constants (zmm26-31)
 *   ✓ Does NOT clobber Q data (zmm10-13) - Q persists across KV loop
 *   ! CLOBBERS zmm8 - overlaps V accum, but exp called BEFORE V accumulation
 *
 * Required constants (must be initialized in zmm26-31 via JitMicrokernelBase):
 *   - zmm_log2e():   log2(e) ≈ 1.4427
 *   - zmm_exp_min(): -87.0f (clamp to prevent underflow)
 */

#pragma once

#include "../../../jit/JitMicrokernelBase.h"
#include "../../../jit/RegisterAllocation.h"
#include "../../../jit/RegisterGuard.h"
#include "../../../jit/RegisterEnforcement.h"

namespace llaminar::v2::kernels::jit
{
    // Import typed register types and C++20 concepts for compile-time checking
    using namespace llaminar2::jit;

    // ============================================================================
    // FastExp Register Manifest (compile-time documentation)
    // ============================================================================
    // Documents the exact registers borrowed by emit_exp2_poly().
    // The static_assert verifies no conflicts at compile time.

    using FastExpManifest = KernelRegisterManifest<Input0, Input6, Input7>;
    static_assert(!FastExpManifest::has_conflicts(),
                  "FastExp register manifest has conflicts!");
    static_assert(FastExpManifest::all_from_zones<QVectorZone>(),
                  "FastExp should only use InputZone (QVector) registers");

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
    class JitFastExpEmitter
    {
    public:
        /**
         * @brief Emit fast exp code: dst = exp(src)
         *
         * Prerequisites:
         * - zmm_log2e() initialized to log2(e) ≈ 1.4427
         * - zmm_exp_min() initialized to -87.0f
         *
         * Clobbers (InputZone - safe after dot products):
         * - zmm8: Integer part for range reduction
         * - zmm9: Fractional part for range reduction
         * - zmm10: Horner polynomial temp
         * - rax: For constant loading
         *
         * SAFE for FA2:
         * - Does NOT clobber Score0-3 (xmm20-23)
         * - Does NOT clobber Scratch4-5 (zmm24-25) - may hold tile_max!
         * - Does NOT clobber constants (zmm26-31)
         *
         * @param gen Code generator to emit into
         * @param dst Output ZMM register
         * @param src Input ZMM register (can be same as dst)
         */
        void emit_fast_exp(
            JitMicrokernelBase &gen,
            const Xbyak::Zmm &dst,
            const Xbyak::Zmm &src)
        {
            using namespace Xbyak;

            gen.debug_emit("emit_fast_exp");

            // Clamp input to avoid overflow/underflow
            // x_clamped = max(src, -87.0f)
            gen.vmaxps(dst, src, gen.zmm_const_exp_min());

            // x * log2(e)
            gen.vmulps(dst, dst, gen.zmm_const_log2e());

            // Compute 2^(x * log2e) using polynomial
            emit_exp2_poly(gen, dst, dst);
        }

#if __cplusplus >= 202002L
        /**
         * @brief Emit fast exp code using typed registers (C++20 concept-constrained)
         *
         * Preferred overload when using typed registers from RegisterAllocation.h.
         * Provides compile-time safety via TypedZmmRegister concept.
         *
         * @tparam DstReg Typed ZMM register for output (must satisfy TypedZmmRegister)
         * @tparam SrcReg Typed ZMM register for input (must satisfy TypedZmmRegister)
         * @param gen Code generator to emit into
         * @param dst Output register (typed)
         * @param src Input register (typed, can be same as dst)
         */
        template <TypedZmmRegister DstReg, TypedZmmRegister SrcReg>
        void emit_fast_exp(
            JitMicrokernelBase &gen,
            const DstReg &dst,
            const SrcReg &src)
        {
            // Delegate to the raw version, extracting Xbyak::Zmm
            emit_fast_exp(gen, dst.zmm(), src.zmm());
        }
#endif // C++20

        /**
         * @brief Emit 2^x polynomial approximation
         *
         * Uses range reduction: 2^x = 2^n * 2^f where n = floor(x), f = x - n
         * Then computes 2^f using 5th-order Taylor series.
         *
         * REGISTER USAGE:
         * - zmm8: Horner temp for loading constants during FMA
         * - zmm14: Integer part n (floor of input) - preserved for final vscalefps
         * - zmm15: Fractional part f - needed throughout polynomial evaluation
         * - dst: Polynomial accumulator (output)
         *
         * Uses zmm8, zmm14, zmm15 to avoid conflicts with:
         * - Q data (zmm10-13): Persists across KV loop - PRESERVED
         * - V accum (zmm8-9): V accumulation happens AFTER exp, so zmm8 overlap is OK
         * - Score0-3 (xmm20-23): FA2 scores still live during weight computation
         * - Scratch4-5 (zmm24-25): May hold tile_max - PRESERVED
         * - Constants (zmm26-31): Must be preserved
         * - State (zmm16-19): Softmax max/sum/weight/corr
         *
         * @param gen Code generator
         * @param dst Output ZMM
         * @param src Input ZMM (can be same as dst)
         */
        void emit_exp2_poly(
            JitMicrokernelBase &gen,
            const Xbyak::Zmm &dst,
            const Xbyak::Zmm &src)
        {
            using namespace Xbyak;
            using namespace llaminar2::jit;

            gen.debug_emit("  emit_exp2_poly");

            // === REGISTER GUARDS ===
            // Borrow registers to detect conflicts at JIT compile time.
            // Uses Input0 (zmm8), Input6 (zmm14), Input7 (zmm15).
            auto guard_tmp = gen.borrow<Input0>(); // zmm8 for Horner temp
            auto guard_n = gen.borrow<Input6>();   // zmm14 for integer part
            auto guard_f = gen.borrow<Input7>();   // zmm15 for fractional part

            Zmm zmm_n = guard_n.zmm();     // Integer part n = floor(x)
            Zmm zmm_f = guard_f.zmm();     // Fractional part f
            Zmm zmm_tmp = guard_tmp.zmm(); // Temp for loading constants

            // Split into integer and fractional parts
            // n = floor(x), f = x - n
            gen.vrndscaleps(zmm_n, src, 1); // floor(x)
            gen.vsubps(zmm_f, src, zmm_n);  // f = x - floor(x)

            // 2^f using 5th-order Taylor series (Horner's method)
            // p(f) = c5*f^5 + c4*f^4 + c3*f^3 + c2*f^2 + c1*f + c0
            //      = ((((c5*f + c4)*f + c3)*f + c2)*f + c1)*f + c0

            // Taylor coefficients for 2^x around x=0:
            // c0 = 1.0, c1 = ln(2), c2 = ln(2)^2/2, c3 = ln(2)^3/6, ...
            static constexpr float c0 = 1.0f;
            static constexpr float c1 = 0.6931472f; // ln(2)
            static constexpr float c2 = 0.2402265f; // ln(2)^2/2
            static constexpr float c3 = 0.0555041f; // ln(2)^3/6
            static constexpr float c4 = 0.0096181f; // ln(2)^4/24
            static constexpr float c5 = 0.0013334f; // ln(2)^5/120

            // Horner's method: p(f) = ((((c5*f + c4)*f + c3)*f + c2)*f + c1)*f + c0
            // dst accumulates the polynomial, zmm_f holds f, zmm_tmp loads constants

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
    class JitFastExpKernel : public JitMicrokernelBase
    {
    public:
        using kernel_func_t = void (*)(const float *in, float *out, int n);

        explicit JitFastExpKernel(bool debug = false)
            : JitMicrokernelBase(4 * 1024, debug)
        {
            generate();
        }

        kernel_func_t get_kernel()
        {
            return getCode<kernel_func_t>();
        }

    private:
        JitFastExpEmitter exp_emitter_;

        void generate()
        {
            using namespace Xbyak;

            debug_emit("JitFastExpKernel::generate()");

            // Function: void kernel(const float* in, float* out, int n)
            // Args: rdi = in, rsi = out, rdx = n
            const Reg64 &reg_in = rdi;
            const Reg64 &reg_out = rsi;
            const Reg64 &reg_n = rdx;
            const Reg64 &reg_i = rcx;

            // Initialize constants
            debug_emit("  Init constants");
            emit_broadcast_fp32_const(zmm_const_log2e(), 1.4426950408889634f, rax);
            emit_broadcast_fp32_const(zmm_const_exp_min(), -87.0f, rax);

            // Loop: for (i = 0; i < n; i += 16)
            xor_(reg_i, reg_i);

            Label loop_start, loop_end;
            L(loop_start);
            cmp(reg_i, reg_n);
            jge(loop_end, T_NEAR);

            // Load 16 floats
            vmovups(zmm_accum0(), ptr[reg_in + reg_i * 4]);

            // Compute exp
            exp_emitter_.emit_fast_exp(*this, zmm_accum0(), zmm_accum0());

            // Store 16 floats
            vmovups(ptr[reg_out + reg_i * 4], zmm_accum0());

            // i += 16
            add(reg_i, 16);
            jmp(loop_start, T_NEAR);

            L(loop_end);
            ret();
        }
    };

} // namespace llaminar::v2::kernels::jit
