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
 * ═══════════════════════════════════════════════════════════════════════════════
 * DESIGN PHILOSOPHY
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * - Each JIT microkernel (μK1-μK5) is independently testable
 * - Microkernels compose into fused kernels via emitters
 * - Register conventions ensure composability without conflicts
 * - RegisterGuard system catches conflicts at JIT compile time
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * REGISTER ZONES (32 ZMM registers partitioned by purpose)
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * AccumZone (zmm0-7): Context accumulators - persist across KV loop
 *   - 8 × 16 floats = 128 floats max (supports head_dim up to 128)
 *   - For head_dim=64: zmm0-3 (64 floats)
 *   - For head_dim=128: zmm0-7 (128 floats)
 *
 * InputZone (zmm8-15): Q/K/V data loading
 *   - zmm8-9: V accum output, exp temp
 *   - zmm10-13: Pre-loaded unsigned Q data (2-4 blocks)
 *   - zmm14-15: exp2_poly (integer/fractional parts)
 *
 * StateZone (zmm16-19): Online softmax state
 *   - zmm16: StateMax - running maximum
 *   - zmm17: StateSum - running sum of exp weights
 *   - zmm18: StateWeight - current weight (FA1 single mode)
 *   - zmm19: StateCorr - correction factor for rescaling
 *
 * ScratchZone (zmm20-25): Temporaries + FA2 scores
 *   - zmm20-23: Score0-3 (xmm low 128-bits) / weight broadcast (full zmm)
 *   - zmm24: Scratch4 - safe scratch, tile_max storage
 *   - zmm25: Scratch5 - safe scratch, correction term
 *
 * ConstZone (zmm26-31): Preloaded constants - never modified after init
 *   - zmm26: 0x80808080 (Q8 unsigned bias)
 *   - zmm27: attention scale (1/sqrt(head_dim))
 *   - zmm28: -infinity (softmax init) or 16.0f (Q8_1 correction)
 *   - zmm29: 1.0f
 *   - zmm30: log2(e) for exp approximation
 *   - zmm31: -87.0f (exp underflow clamp)
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * REGISTER ALIASING WARNING
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * XMM, YMM, and ZMM registers with the same index share physical storage:
 *   - xmm0 = lower 128 bits of ymm0 = lower 128 bits of zmm0
 *   - Writing to ymm0 zeroes the upper 384 bits of zmm0
 *   - Writing to xmm0 zeroes the upper 384 bits of zmm0
 *
 * FA2 Score/Weight aliasing (intentional):
 *   - Scores computed into Score0-3 (xmm20-23)
 *   - After scores consumed, weights broadcast into Scratch0-3 (zmm20-23)
 *   - This reuses the same physical registers for different phases
 *
 * CRITICAL: zmm20 holds broadcasted weight during V accumulation!
 *   - Cannot use zmm20 as scratch during emit_weighted_accum
 *   - Bug fix (Dec 2025): Changed temp registers to zmm21+ for spills
 *
 * Use RegisterAllocation.h typed registers to prevent aliasing bugs.
 */

#pragma once

#include "../../../../external/onednn/third_party/xbyak/xbyak.h"
#include "RegisterAllocation.h"
#include "RegisterGuard.h"
#include <array>
#include <cstdint>
#include <string>
#include <iostream>
#include <memory>

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
     * - Register tracking for conflict detection during JIT compilation
     */
    class JitMicrokernelBase : public Xbyak::CodeGenerator
    {
    public:
        explicit JitMicrokernelBase(size_t max_code_size = 8 * 1024, bool debug = false)
            : Xbyak::CodeGenerator(max_code_size), debug_(debug), tracker_(std::make_unique<llaminar2::jit::RegisterTracker>()) // Always enable tracking
        {
        }

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
        // Legacy Raw ZMM Accessors (DEPRECATED - use typed accessors below)
        // ========================================================================
        // These return raw Xbyak::Zmm which bypasses compile-time type checking.
        // Prefer the typed accessors (const_128(), state_max(), etc.) for new code.

        [[deprecated("Use const_128() instead")]]
        Xbyak::Zmm zmm_128() const
        {
            return Xbyak::Zmm(ConstRegs::ZMM_128);
        }
        [[deprecated("Use const_scale() instead")]]
        Xbyak::Zmm zmm_scale() const
        {
            return Xbyak::Zmm(ConstRegs::ZMM_SCALE);
        }
        [[deprecated("Use const_neg_inf() instead")]]
        Xbyak::Zmm zmm_neg_inf() const
        {
            return Xbyak::Zmm(ConstRegs::ZMM_NEG_INF);
        }
        [[deprecated("Use const_16() instead")]]
        Xbyak::Zmm zmm_16() const
        {
            return Xbyak::Zmm(ConstRegs::ZMM_16);
        }
        [[deprecated("Use const_one() instead")]]
        Xbyak::Zmm zmm_one() const
        {
            return Xbyak::Zmm(ConstRegs::ZMM_ONE);
        }
        [[deprecated("Use const_log2e() instead")]]
        Xbyak::Zmm zmm_log2e() const
        {
            return Xbyak::Zmm(ConstRegs::ZMM_LOG2E);
        }
        [[deprecated("Use const_exp_min() instead")]]
        Xbyak::Zmm zmm_exp_min() const
        {
            return Xbyak::Zmm(ConstRegs::ZMM_EXP_MIN);
        }

        [[deprecated("Use state_max() instead")]]
        Xbyak::Zmm zmm_max() const
        {
            return Xbyak::Zmm(StateRegs::ZMM_MAX);
        }
        [[deprecated("Use state_sum() instead")]]
        Xbyak::Zmm zmm_sum() const
        {
            return Xbyak::Zmm(StateRegs::ZMM_SUM);
        }
        [[deprecated("Use state_weight() instead")]]
        Xbyak::Zmm zmm_weight() const
        {
            return Xbyak::Zmm(StateRegs::ZMM_WEIGHT);
        }
        [[deprecated("Use state_corr() instead")]]
        Xbyak::Zmm zmm_corr() const
        {
            return Xbyak::Zmm(StateRegs::ZMM_CORR);
        }
        [[deprecated("Use Input7 from RegisterAllocation.h instead")]]
        Xbyak::Zmm zmm_score() const
        {
            return Xbyak::Zmm(StateRegs::ZMM_SCORE);
        }

        // Scratch registers - use scratchN() typed accessors
        [[deprecated("Use scratch0()-scratch5() typed accessors")]]
        Xbyak::Zmm zmm_scratch(int idx) const
        {
            return Xbyak::Zmm(ZmmZones::SCRATCH_START + idx);
        }

        // Accumulator registers - use accumN() typed accessors
        [[deprecated("Use accum0()-accum7() typed accessors")]]
        Xbyak::Zmm zmm_accum(int idx) const
        {
            return Xbyak::Zmm(ZmmZones::ACCUM_START + idx);
        }

        // Input registers - use InputN from RegisterAllocation.h
        [[deprecated("Use Input0-Input7 from RegisterAllocation.h")]]
        Xbyak::Zmm zmm_input(int idx) const
        {
            return Xbyak::Zmm(ZmmZones::INPUT_START + idx);
        }

        // ========================================================================
        // Typed Register Accessors (compile-time safe)
        // ========================================================================
        // These return typed registers from RegisterAllocation.h, enabling
        // compile-time checking of register conflicts.
        // NOTE: These are TAG TYPES only - use zmm_*() accessors below or borrow<>()

        // Constant registers (zmm26-31) - preloaded at kernel init
        static constexpr llaminar2::jit::Const128 const_128() { return {}; }
        static constexpr llaminar2::jit::ConstScale const_scale() { return {}; }
        static constexpr llaminar2::jit::ConstNegInf const_neg_inf() { return {}; }
        static constexpr llaminar2::jit::Const16 const_16() { return {}; } // Note: aliases const_neg_inf()
        static constexpr llaminar2::jit::ConstOne const_one() { return {}; }
        static constexpr llaminar2::jit::ConstLog2e const_log2e() { return {}; }
        static constexpr llaminar2::jit::ConstExpMin const_exp_min() { return {}; }

        // State registers (zmm16-19) - online softmax state
        static constexpr llaminar2::jit::StateMax state_max() { return {}; }
        static constexpr llaminar2::jit::StateSum state_sum() { return {}; }
        static constexpr llaminar2::jit::StateWeight state_weight() { return {}; }
        static constexpr llaminar2::jit::StateCorr state_corr() { return {}; }

        // Score registers (xmm20-23) - FA2 tile scores
        // WARNING: These alias Scratch0-3 (zmm20-23)!
        static constexpr llaminar2::jit::Score0 score0() { return {}; }
        static constexpr llaminar2::jit::Score1 score1() { return {}; }
        static constexpr llaminar2::jit::Score2 score2() { return {}; }
        static constexpr llaminar2::jit::Score3 score3() { return {}; }

        // Scratch registers (zmm20-25) - temporaries
        // WARNING: Scratch0-3 alias Score0-3!
        static constexpr llaminar2::jit::Scratch0 scratch0() { return {}; }
        static constexpr llaminar2::jit::Scratch1 scratch1() { return {}; }
        static constexpr llaminar2::jit::Scratch2 scratch2() { return {}; }
        static constexpr llaminar2::jit::Scratch3 scratch3() { return {}; }
        static constexpr llaminar2::jit::Scratch4 scratch4() { return {}; } // Safe during FA2
        static constexpr llaminar2::jit::Scratch5 scratch5() { return {}; } // Safe during FA2

        // Accumulator registers (zmm0-7) - context accumulators
        static constexpr llaminar2::jit::Accum0 accum0() { return {}; }
        static constexpr llaminar2::jit::Accum1 accum1() { return {}; }
        static constexpr llaminar2::jit::Accum2 accum2() { return {}; }
        static constexpr llaminar2::jit::Accum3 accum3() { return {}; }
        static constexpr llaminar2::jit::Accum4 accum4() { return {}; }
        static constexpr llaminar2::jit::Accum5 accum5() { return {}; }
        static constexpr llaminar2::jit::Accum6 accum6() { return {}; }
        static constexpr llaminar2::jit::Accum7 accum7() { return {}; }

        // ========================================================================
        // Direct Xbyak Register Accessors
        // ========================================================================
        // These return raw Xbyak registers directly for "owned" registers that
        // don't need borrowing (constants, state). For scratch/accum, prefer borrow<>().

        // Constants (zmm26-31) - preloaded once, read-only throughout kernel
        static constexpr Xbyak::Zmm zmm_const_128() { return Xbyak::Zmm(llaminar2::jit::Const128::absolute_index); }
        static constexpr Xbyak::Ymm ymm_const_128() { return Xbyak::Ymm(llaminar2::jit::Const128::absolute_index); }
        static constexpr Xbyak::Zmm zmm_const_scale() { return Xbyak::Zmm(llaminar2::jit::ConstScale::absolute_index); }
        static constexpr Xbyak::Zmm zmm_const_neg_inf() { return Xbyak::Zmm(llaminar2::jit::ConstNegInf::absolute_index); }
        static constexpr Xbyak::Zmm zmm_const_16() { return Xbyak::Zmm(llaminar2::jit::Const16::absolute_index); }
        static constexpr Xbyak::Zmm zmm_const_one() { return Xbyak::Zmm(llaminar2::jit::ConstOne::absolute_index); }
        static constexpr Xbyak::Zmm zmm_const_log2e() { return Xbyak::Zmm(llaminar2::jit::ConstLog2e::absolute_index); }
        static constexpr Xbyak::Zmm zmm_const_exp_min() { return Xbyak::Zmm(llaminar2::jit::ConstExpMin::absolute_index); }

        // State (zmm16-19) - persist across kernel, managed at higher level
        static constexpr Xbyak::Zmm zmm_state_max() { return Xbyak::Zmm(llaminar2::jit::StateMax::absolute_index); }
        static constexpr Xbyak::Xmm xmm_state_max() { return Xbyak::Xmm(llaminar2::jit::StateMax::absolute_index); }
        static constexpr Xbyak::Zmm zmm_state_sum() { return Xbyak::Zmm(llaminar2::jit::StateSum::absolute_index); }
        static constexpr Xbyak::Zmm zmm_state_weight() { return Xbyak::Zmm(llaminar2::jit::StateWeight::absolute_index); }
        static constexpr Xbyak::Zmm zmm_state_corr() { return Xbyak::Zmm(llaminar2::jit::StateCorr::absolute_index); }

        // Scores (xmm20-23) - alias scratch, but used in specific phases
        static constexpr Xbyak::Xmm xmm_score0() { return Xbyak::Xmm(llaminar2::jit::Score0::absolute_index); }
        static constexpr Xbyak::Xmm xmm_score1() { return Xbyak::Xmm(llaminar2::jit::Score1::absolute_index); }
        static constexpr Xbyak::Xmm xmm_score2() { return Xbyak::Xmm(llaminar2::jit::Score2::absolute_index); }
        static constexpr Xbyak::Xmm xmm_score3() { return Xbyak::Xmm(llaminar2::jit::Score3::absolute_index); }

        // Accumulators (zmm0-7) - for direct access when not using borrow system
        static constexpr Xbyak::Zmm zmm_accum0() { return Xbyak::Zmm(llaminar2::jit::Accum0::absolute_index); }
        static constexpr Xbyak::Zmm zmm_accum1() { return Xbyak::Zmm(llaminar2::jit::Accum1::absolute_index); }
        static constexpr Xbyak::Zmm zmm_accum2() { return Xbyak::Zmm(llaminar2::jit::Accum2::absolute_index); }
        static constexpr Xbyak::Zmm zmm_accum3() { return Xbyak::Zmm(llaminar2::jit::Accum3::absolute_index); }
        static constexpr Xbyak::Zmm zmm_accum4() { return Xbyak::Zmm(llaminar2::jit::Accum4::absolute_index); }
        static constexpr Xbyak::Zmm zmm_accum5() { return Xbyak::Zmm(llaminar2::jit::Accum5::absolute_index); }
        static constexpr Xbyak::Zmm zmm_accum6() { return Xbyak::Zmm(llaminar2::jit::Accum6::absolute_index); }
        static constexpr Xbyak::Zmm zmm_accum7() { return Xbyak::Zmm(llaminar2::jit::Accum7::absolute_index); }

        // ========================================================================
        // Type-to-Register Converters (for template parameters)
        // ========================================================================
        // These convert typed register parameters to raw Xbyak registers.
        // Use when a template function receives a typed register argument and
        // needs to emit code with it.

        /**
         * @brief Convert a typed ZMM register to its Xbyak equivalent
         * @tparam T Typed register (e.g., Scratch4, Accum0)
         */
        template <typename T>
        static constexpr Xbyak::Zmm to_zmm(const T &)
        {
            return Xbyak::Zmm(T::absolute_index);
        }

        /**
         * @brief Convert a typed YMM register to its Xbyak equivalent
         * @tparam T Typed register (e.g., Scratch4, Accum0)
         */
        template <typename T>
        static constexpr Xbyak::Ymm to_ymm(const T &)
        {
            return Xbyak::Ymm(T::absolute_index);
        }

        /**
         * @brief Convert a typed XMM register to its Xbyak equivalent
         * @tparam T Typed register (e.g., Score0, Score1)
         */
        template <typename T>
        static constexpr Xbyak::Xmm to_xmm(const T &)
        {
            return Xbyak::Xmm(T::absolute_index);
        }

        // ========================================================================
        // Register Tracking (Runtime Conflict Detection)
        // ========================================================================

        /**
         * @brief Check if register tracking is enabled (always true)
         * @note Register tracking is always enabled; this method exists for API compatibility
         */
        bool tracking_enabled() const { return true; }

        /**
         * @brief Get the register tracker
         * @return Pointer to tracker (never null)
         */
        llaminar2::jit::RegisterTracker *tracker() { return tracker_.get(); }

        /**
         * @brief Borrow a typed register with RAII tracking
         *
         * Returns an RAII guard that will:
         * - Assert if the register is already borrowed
         * - Automatically release when the guard goes out of scope
         *
         * @tparam RegType Typed register (e.g., Score0, Scratch4, Accum0)
         */
        template <typename RegType>
        [[nodiscard]] auto borrow()
        {
            return tracker_->borrow<RegType>();
        }

        /**
         * @brief Reset all register borrows (for starting a new phase)
         */
        void reset_borrows()
        {
            tracker_->reset();
        }

        /**
         * @brief Get debug string of currently borrowed registers
         */
        std::string borrowed_registers_debug() const
        {
            return tracker_->debug_string();
        }

        /**
         * @brief Get a register pool for a zone
         *
         * Enables pool-based borrowing with encoding constraints:
         * @code
         * auto scratch_pool = pool<ScratchZone>();
         * auto guard = scratch_pool.borrow_any();  // Any available scratch
         *
         * // For VEX-safe registers, use Accumulator or QVector zones:
         * auto accum_pool = pool<AccumulatorZone>();
         * auto vex_guard = accum_pool.borrow_any();  // Guaranteed VEX-safe
         * @endcode
         *
         * @tparam Zone The register zone (AccumulatorZone, ScratchZone, etc.)
         * @return RegisterPool<Zone> for borrowing registers
         */
        template <typename Zone>
        [[nodiscard]] llaminar2::jit::RegisterPool<Zone> pool()
        {
            return tracker_->pool<Zone>();
        }

        // ========================================================================
        // Common SIMD Patterns (public for use by Emitter classes)
        // ========================================================================

        bool debug_ = false;
        std::unique_ptr<llaminar2::jit::RegisterTracker> tracker_;

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
