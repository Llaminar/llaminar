/**
 * @file JitMicrokernelBase.h
 * @brief Base class for JIT microkernels with shared register conventions and utilities
 * @author David Sanftenberg
 * @date December 2025
 *
 * Provides common infrastructure for JIT microkernel development:
 * - Standardized register allocation conventions
 * - Common SIMD patterns (horizontal sums, broadcasts)
 * - Debug emission support
 * - Stack management helpers
 *
 * Design Philosophy:
 * - Each JIT microkernel is independently testable
 * - Microkernels can be composed into fused kernels
 * - Register conventions ensure composability without conflicts
 */

#pragma once

#include "../../../../../../external/onednn/third_party/xbyak/xbyak.h"
#include <array>
#include <cstdint>
#include <string>
#include <iostream>

namespace llaminar::v2::kernels::jit
{

    /**
     * @brief Register allocation zones for JIT microkernels
     *
     * To enable composition, we partition ZMM registers into zones:
     * - Constants (zmm26-31): Shared constants, preserved across microkernels
     * - Scratch (zmm20-25): Temporary values, freely clobbered
     * - Accumulator (zmm0-7): Results that persist across microkernel calls
     * - Input (zmm8-15): Input operands, may be clobbered by microkernel
     * - State (zmm16-19): Running state (max, sum, etc.) for online algorithms
     */
    struct ZmmZones
    {
        // Accumulator zone: Results that persist (e.g., context accumulators)
        static constexpr int ACCUM_START = 0;
        static constexpr int ACCUM_END = 7; // zmm0-7

        // Input zone: Loaded inputs (Q, K, V blocks)
        static constexpr int INPUT_START = 8;
        static constexpr int INPUT_END = 15; // zmm8-15

        // State zone: Running algorithm state (max, sum, weight)
        static constexpr int STATE_START = 16;
        static constexpr int STATE_END = 19; // zmm16-19

        // Scratch zone: Temporaries, freely clobbered
        static constexpr int SCRATCH_START = 20;
        static constexpr int SCRATCH_END = 25; // zmm20-25

        // Constants zone: Preserved across calls
        static constexpr int CONST_START = 26;
        static constexpr int CONST_END = 31; // zmm26-31
    };

    /**
     * @brief Well-known constant register assignments
     *
     * These registers hold constants used across multiple microkernels.
     * Once initialized, they should not be modified.
     *
     * Note: ZMM_NEG_INF (zmm28) is dual-purposed:
     *   - Decode mode: -infinity for softmax initialization
     *   - Prefill mode: 16.0f for Q8_1 correction factor
     */
    struct ConstRegs
    {
        static constexpr int ZMM_128 = 26;     ///< 0x80808080 for unsigned conversion
        static constexpr int ZMM_SCALE = 27;   ///< Attention scale (1/sqrt(d))
        static constexpr int ZMM_NEG_INF = 28; ///< -infinity for softmax init (decode)
        static constexpr int ZMM_16 = 28;      ///< 16.0f for Q8_1 correction (prefill) - shares zmm28
        static constexpr int ZMM_ONE = 29;     ///< 1.0f
        static constexpr int ZMM_LOG2E = 30;   ///< log2(e) for fast exp
        static constexpr int ZMM_EXP_MIN = 31; ///< -87.0f (exp underflow clamp)
    };

    /**
     * @brief Well-known state register assignments
     */
    struct StateRegs
    {
        static constexpr int ZMM_MAX = 16;    ///< Running softmax maximum
        static constexpr int ZMM_SUM = 17;    ///< Running softmax sum
        static constexpr int ZMM_WEIGHT = 18; ///< Current softmax weight
        static constexpr int ZMM_CORR = 19;   ///< Correction factor
        static constexpr int ZMM_SCORE = 15;  ///< Current attention score (shared with scratch)
    };

    /**
     * @brief Base class for JIT microkernel code generators
     *
     * Provides common utilities for JIT microkernel implementation:
     * - Debug emission with consistent formatting
     * - Register zone accessors
     * - Common SIMD patterns
     */
    class JitMicrokernelBase : public Xbyak::CodeGenerator
    {
    public:
        explicit JitMicrokernelBase(size_t max_code_size = 8 * 1024, bool debug = false)
            : Xbyak::CodeGenerator(max_code_size), debug_(debug) {}

        virtual ~JitMicrokernelBase() = default;

        /**
         * @brief Get generated code size
         */
        size_t code_size() const { return getSize(); }

        /**
         * @brief Check if debug output is enabled
         */
        bool debug_enabled() const { return debug_; }

        // ========================================================================
        // Debug and Utility Methods (public for emitters)
        // ========================================================================

        /**
         * @brief Emit debug message during code generation
         */
        void debug_emit(const std::string &msg)
        {
            if (debug_)
            {
                std::cerr << "[JIT] " << msg << std::endl;
            }
        }

        // ========================================================================
        // Common ZMM Register Accessors (public for emitters)
        // ========================================================================

        Xbyak::Zmm zmm_128() const { return Xbyak::Zmm(ConstRegs::ZMM_128); }
        Xbyak::Zmm zmm_scale() const { return Xbyak::Zmm(ConstRegs::ZMM_SCALE); }
        Xbyak::Zmm zmm_neg_inf() const { return Xbyak::Zmm(ConstRegs::ZMM_NEG_INF); }
        Xbyak::Zmm zmm_16() const { return Xbyak::Zmm(ConstRegs::ZMM_16); }  ///< 16.0f for Q8_1 correction
        Xbyak::Zmm zmm_one() const { return Xbyak::Zmm(ConstRegs::ZMM_ONE); }
        Xbyak::Zmm zmm_log2e() const { return Xbyak::Zmm(ConstRegs::ZMM_LOG2E); }
        Xbyak::Zmm zmm_exp_min() const { return Xbyak::Zmm(ConstRegs::ZMM_EXP_MIN); }

        Xbyak::Zmm zmm_max() const { return Xbyak::Zmm(StateRegs::ZMM_MAX); }
        Xbyak::Zmm zmm_sum() const { return Xbyak::Zmm(StateRegs::ZMM_SUM); }
        Xbyak::Zmm zmm_weight() const { return Xbyak::Zmm(StateRegs::ZMM_WEIGHT); }
        Xbyak::Zmm zmm_corr() const { return Xbyak::Zmm(StateRegs::ZMM_CORR); }
        Xbyak::Zmm zmm_score() const { return Xbyak::Zmm(StateRegs::ZMM_SCORE); }

        // Scratch registers
        Xbyak::Zmm zmm_scratch(int idx) const
        {
            return Xbyak::Zmm(ZmmZones::SCRATCH_START + idx);
        }

        // Accumulator registers
        Xbyak::Zmm zmm_accum(int idx) const
        {
            return Xbyak::Zmm(ZmmZones::ACCUM_START + idx);
        }

        // Input registers
        Xbyak::Zmm zmm_input(int idx) const
        {
            return Xbyak::Zmm(ZmmZones::INPUT_START + idx);
        }

        // ========================================================================
        // Common SIMD Patterns (public for use by Emitter classes)
        // ========================================================================

        bool debug_ = false;

        // ========================================================================
        // Common SIMD Patterns
        // ========================================================================

        /**
         * @brief Emit horizontal sum of ZMM register to scalar in XMM
         *
         * Result is in element 0 of dst_xmm.
         *
         * @param dst_xmm Destination XMM (will contain scalar result)
         * @param src_zmm Source ZMM to reduce
         * @param tmp_ymm Temporary YMM register
         * @param tmp_xmm Temporary XMM register
         */
        void emit_hsum_zmm_to_scalar(
            const Xbyak::Xmm &dst_xmm,
            const Xbyak::Zmm &src_zmm,
            const Xbyak::Ymm &tmp_ymm,
            const Xbyak::Xmm &tmp_xmm)
        {
            using namespace Xbyak;
            debug_emit("  hsum_zmm_to_scalar");

            // Extract high 256 bits, add to low 256 bits
            vextractf32x8(tmp_ymm, src_zmm, 1);
            vaddps(Ymm(dst_xmm.getIdx()), Ymm(src_zmm.getIdx()), tmp_ymm);

            // Extract high 128 bits of result, add to low 128 bits
            vextractf32x4(tmp_xmm, Ymm(dst_xmm.getIdx()), 1);
            vaddps(dst_xmm, dst_xmm, tmp_xmm);

            // Horizontal add within 128 bits
            vshufps(tmp_xmm, dst_xmm, dst_xmm, 0x4E); // swap pairs
            vaddps(dst_xmm, dst_xmm, tmp_xmm);
            vshufps(tmp_xmm, dst_xmm, dst_xmm, 0xB1); // swap elements
            vaddps(dst_xmm, dst_xmm, tmp_xmm);
            // Result in dst_xmm[0]
        }

        /**
         * @brief Emit horizontal max of ZMM register to scalar in XMM
         *
         * @param dst_xmm Destination XMM (will contain scalar max)
         * @param src_zmm Source ZMM to reduce
         * @param tmp_ymm Temporary YMM register
         * @param tmp_xmm Temporary XMM register
         */
        void emit_hmax_zmm_to_scalar(
            const Xbyak::Xmm &dst_xmm,
            const Xbyak::Zmm &src_zmm,
            const Xbyak::Ymm &tmp_ymm,
            const Xbyak::Xmm &tmp_xmm)
        {
            using namespace Xbyak;
            debug_emit("  hmax_zmm_to_scalar");

            vextractf32x8(tmp_ymm, src_zmm, 1);
            vmaxps(Ymm(dst_xmm.getIdx()), Ymm(src_zmm.getIdx()), tmp_ymm);

            vextractf32x4(tmp_xmm, Ymm(dst_xmm.getIdx()), 1);
            vmaxps(dst_xmm, dst_xmm, tmp_xmm);

            vshufps(tmp_xmm, dst_xmm, dst_xmm, 0x4E);
            vmaxps(dst_xmm, dst_xmm, tmp_xmm);
            vshufps(tmp_xmm, dst_xmm, dst_xmm, 0xB1);
            vmaxps(dst_xmm, dst_xmm, tmp_xmm);
        }

        /**
         * @brief Emit horizontal sum of int32 ZMM to scalar
         */
        void emit_hsum_epi32_zmm_to_scalar(
            const Xbyak::Xmm &dst_xmm,
            const Xbyak::Zmm &src_zmm,
            const Xbyak::Ymm &tmp_ymm,
            const Xbyak::Xmm &tmp_xmm)
        {
            using namespace Xbyak;
            debug_emit("  hsum_epi32_zmm_to_scalar");

            vextracti32x8(tmp_ymm, src_zmm, 1);
            vpaddd(Ymm(dst_xmm.getIdx()), Ymm(src_zmm.getIdx()), tmp_ymm);

            vextracti32x4(tmp_xmm, Ymm(dst_xmm.getIdx()), 1);
            vpaddd(dst_xmm, dst_xmm, tmp_xmm);

            vpshufd(tmp_xmm, dst_xmm, 0x4E);
            vpaddd(dst_xmm, dst_xmm, tmp_xmm);
            vpshufd(tmp_xmm, dst_xmm, 0xB1);
            vpaddd(dst_xmm, dst_xmm, tmp_xmm);
        }

        /**
         * @brief Load FP32 constant into XMM via integer mov
         *
         * @param dst XMM destination
         * @param value Float value to load
         * @param tmp_reg 64-bit GP register for intermediate
         */
        void emit_load_fp32_const(
            const Xbyak::Xmm &dst,
            float value,
            const Xbyak::Reg64 &tmp_reg)
        {
            union
            {
                float f;
                uint32_t i;
            } u;
            u.f = value;
            mov(tmp_reg.cvt32(), u.i);
            vmovd(dst, tmp_reg.cvt32());
        }

        /**
         * @brief Broadcast FP32 constant to ZMM
         *
         * @param dst ZMM destination
         * @param value Float value to broadcast
         * @param tmp_reg 64-bit GP register for intermediate
         */
        void emit_broadcast_fp32_const(
            const Xbyak::Zmm &dst,
            float value,
            const Xbyak::Reg64 &tmp_reg)
        {
            union
            {
                float f;
                uint32_t i;
            } u;
            u.f = value;
            mov(tmp_reg.cvt32(), u.i);
            vpbroadcastd(dst, tmp_reg.cvt32());
        }

        /**
         * @brief Broadcast int32 constant to ZMM
         */
        void emit_broadcast_i32_const(
            const Xbyak::Zmm &dst,
            uint32_t value,
            const Xbyak::Reg64 &tmp_reg)
        {
            mov(tmp_reg.cvt32(), value);
            vpbroadcastd(dst, tmp_reg.cvt32());
        }

        // ========================================================================
        // Stack Frame Management (for standalone kernels)
        // ========================================================================

        /**
         * @brief List of callee-saved registers we preserve
         */
        static constexpr std::array<Xbyak::Reg64, 6> callee_saved_regs()
        {
            return {Xbyak::util::rbx, Xbyak::util::rbp, Xbyak::util::r12,
                    Xbyak::util::r13, Xbyak::util::r14, Xbyak::util::r15};
        }

        /**
         * @brief Size of the stack frame for callee-saved registers
         *
         * @return Size in bytes (6 registers × 8 bytes = 48 bytes)
         */
        static constexpr size_t stack_frame_size()
        {
            return 6 * 8; // 6 callee-saved registers
        }

        /**
         * @brief Push all callee-saved registers
         *
         * Call at the start of a function that uses rbx, rbp, r12-r15.
         */
        void push_callee_saved()
        {
            debug_emit("push_callee_saved");
            push(rbx);
            push(rbp);
            push(r12);
            push(r13);
            push(r14);
            push(r15);
        }

        /**
         * @brief Pop all callee-saved registers (in reverse order)
         *
         * Call at the end of a function before ret.
         */
        void pop_callee_saved()
        {
            debug_emit("pop_callee_saved");
            pop(r15);
            pop(r14);
            pop(r13);
            pop(r12);
            pop(rbp);
            pop(rbx);
        }

        /**
         * @brief Load a 32-bit float constant into a ZMM register (broadcasted)
         *
         * @param dst Destination ZMM register
         * @param value Float value to load
         *
         * Uses rax as a temporary register for the integer representation.
         */
        void load_constant_f32(const Xbyak::Zmm &dst, float value)
        {
            emit_broadcast_fp32_const(dst, value, rax);
        }
    };

} // namespace llaminar::v2::kernels::jit
