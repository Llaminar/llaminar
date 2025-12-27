/**
 * @file RegisterAllocation.h
 * @brief Compile-time safe register allocation framework for JIT code generation
 *
 * This framework provides type-safe wrappers around Xbyak registers to prevent
 * register clobbering bugs at compile time. Instead of relying on comments and
 * conventions, register assignments are encoded in the C++ type system.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * KEY CONCEPTS
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * 1. **Register Zones**: Disjoint sets of registers for different purposes
 *    - AccumulatorZone (zmm0-7): Context accumulators
 *    - QVectorZone (zmm8-15): Input data (Q, K, V, temps)
 *    - StateZone (zmm16-19): Online softmax state
 *    - ScratchZone (zmm20-25): Temporaries
 *    - ScoreZone (xmm20-23): FA2 scalar scores (aliases low Scratch)
 *    - ReservedZone (zmm26-31): Constants
 *
 * 2. **Typed Registers**: Template wrappers encoding register index in the type
 *    - TypedZmm<Zone, index> → wraps specific zmm register
 *    - TypedXmm<Zone, index> → wraps specific xmm register
 *
 * 3. **Zone Membership**: SFINAE/static_assert verify zone membership at compile time
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * FA2 SCORE/SCRATCH ALIASING
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * ScoreZone (xmm20-23) physically overlaps ScratchZone (zmm20-23):
 *   - Phase 1: Scores written to Score0-3 (xmm20-23)
 *   - Phase 2: After scores consumed, weights broadcast to Scratch0-3 (zmm20-23)
 *
 * CRITICAL CONSTRAINT: During V accumulation (emit_weighted_accum):
 *   - zmm20 holds the broadcasted attention weight
 *   - CANNOT use zmm20 as scratch (would clobber weight!)
 *   - Use zmm21+ or Scratch4-5 (zmm24-25) for any temps needed
 *
 * Safe scratch during FA2: Scratch4 (zmm24), Scratch5 (zmm25)
 *   - These do NOT alias ScoreZone
 *   - Use require_safe_scratch_for_fa2<T> to enforce at compile time
 *
 * Example usage:
 * @code
 * // Define your register layout
 * using MyAccum0 = TypedZmm<AccumulatorZone, 0>;  // zmm0
 * using MyAccum1 = TypedZmm<AccumulatorZone, 1>;  // zmm1
 * using MyScratch0 = TypedZmm<ScratchZone, 0>;    // zmm20
 *
 * // Functions can enforce zone membership via static_assert
 * template<typename ScratchReg>
 * void emit_something(const ScratchReg& scratch) {
 *     static_assert(std::is_same_v<typename ScratchReg::zone_type, ScratchZone>,
 *                   "Must pass a scratch register");
 * }
 *
 * // FA2-safe scratch constraint
 * template<typename T, require_safe_scratch_for_fa2<T> = true>
 * void emit_tile_max_reduction(const T& output, ...) {
 *     // output guaranteed to be Scratch4 or Scratch5 (zmm24/25)
 * }
 * @endcode
 */

#pragma once

#include <type_traits>
#include "../../../../external/onednn/third_party/xbyak/xbyak.h"

namespace llaminar2::jit
{

    // ============================================================================
    // VEX/EVEX Encoding Constraint Tags
    // ============================================================================

    /**
     * @brief Registers 0-15 can use VEX encoding (legacy SSE/AVX compatibility)
     *
     * VEX-encoded instructions work with xmm0-15/ymm0-15 but NOT xmm16-31/ymm16-31.
     * Examples: vmovdqu, vshufps (non-512-bit), scalar ops
     */
    struct LowRegisterTag
    {
        static constexpr bool is_vex_safe = true;
        static constexpr bool is_evex_only = false;
        static constexpr const char *name = "LOW(0-15)";
    };

    /**
     * @brief Registers 16-31 require EVEX encoding
     *
     * These cannot use VEX-encoded instructions. Must use EVEX equivalents:
     *   - vmovdqu → vmovdqu16/vmovdqu32/vmovdqu64
     *   - vshufps → vshufps (with EVEX prefix, automatic for zmm)
     */
    struct HighRegisterTag
    {
        static constexpr bool is_vex_safe = false;
        static constexpr bool is_evex_only = true;
        static constexpr const char *name = "HIGH(16-31)";
    };

    // ============================================================================
    // Register Zone Definitions
    // ============================================================================

    /**
     * @brief Base zone tag for CRTP pattern
     */
    struct RegisterZoneBase
    {
    };

    /**
     * @brief Accumulator zone: zmm0-zmm7 (8 registers for V accumulation)
     *
     * All LOW registers - VEX-safe for xmm/ymm operations.
     */
    struct AccumulatorZone : RegisterZoneBase
    {
        static constexpr int base_index = 0;
        static constexpr int count = 8;
        static constexpr const char *name = "Accumulator";

        // Encoding constraints
        static constexpr int low_count = 8;  // All 8 are LOW (0-7)
        static constexpr int high_count = 0; // None are HIGH
        static constexpr bool has_low = true;
        static constexpr bool has_high = false;
        static constexpr bool all_low = true;
        static constexpr bool all_high = false;
    };

    /**
     * @brief Q vector zone: zmm8-zmm15 (8 registers for Q tiles)
     *
     * All LOW registers - VEX-safe for xmm/ymm operations.
     */
    struct QVectorZone : RegisterZoneBase
    {
        static constexpr int base_index = 8;
        static constexpr int count = 8;
        static constexpr const char *name = "QVector";

        // Encoding constraints
        static constexpr int low_count = 8;  // All 8 are LOW (8-15)
        static constexpr int high_count = 0; // None are HIGH
        static constexpr bool has_low = true;
        static constexpr bool has_high = false;
        static constexpr bool all_low = true;
        static constexpr bool all_high = false;
    };

    /**
     * @brief State zone: zmm16-zmm19 (online softmax state)
     *
     * All HIGH registers - require EVEX encoding for xmm/ymm operations.
     *
     * Layout:
     *   zmm16 = max (running maximum for numerical stability)
     *   zmm17 = sum (running sum of exp weights)
     *   zmm18 = weight (current attention weight)
     *   zmm19 = corr (correction factor for rescaling)
     */
    struct StateZone : RegisterZoneBase
    {
        static constexpr int base_index = 16;
        static constexpr int count = 4;
        static constexpr const char *name = "State";

        // Named indices within zone
        static constexpr int MAX_IDX = 0;    // zmm16
        static constexpr int SUM_IDX = 1;    // zmm17
        static constexpr int WEIGHT_IDX = 2; // zmm18
        static constexpr int CORR_IDX = 3;   // zmm19

        // Encoding constraints
        static constexpr int low_count = 0;  // None are LOW
        static constexpr int high_count = 4; // All 4 are HIGH (16-19)
        static constexpr bool has_low = false;
        static constexpr bool has_high = true;
        static constexpr bool all_low = false;
        static constexpr bool all_high = true;
    };

    /**
     * @brief Scratch zone: zmm20-zmm25 (6 temporary registers)
     *
     * All HIGH registers - require EVEX encoding for xmm/ymm operations.
     *
     * CRITICAL: When used as XMM/YMM overlays, these share the same physical registers!
     *   xmm20 overlays zmm20, etc.
     *
     * For FA2 tiling, xmm20-xmm23 hold scores, so zmm20-zmm23 cannot be used
     * simultaneously for other scratch purposes.
     */
    struct ScratchZone : RegisterZoneBase
    {
        static constexpr int base_index = 20;
        static constexpr int count = 6;
        static constexpr const char *name = "Scratch";

        // Encoding constraints
        static constexpr int low_count = 0;  // None are LOW
        static constexpr int high_count = 6; // All 6 are HIGH (20-25)
        static constexpr bool has_low = false;
        static constexpr bool has_high = true;
        static constexpr bool all_low = false;
        static constexpr bool all_high = true;
    };

    /**
     * @brief Score zone: xmm20-xmm23 (4 scalar scores for FA2 tiling)
     *
     * All HIGH registers - require EVEX encoding.
     *
     * WARNING: Physically overlaps ScratchZone zmm20-zmm23!
     * Using ScoreZone marks those scratch registers as unavailable.
     */
    struct ScoreZone : RegisterZoneBase
    {
        static constexpr int base_index = 20;
        static constexpr int count = 4;
        static constexpr const char *name = "Score";

        // Encoding constraints
        static constexpr int low_count = 0;  // None are LOW
        static constexpr int high_count = 4; // All 4 are HIGH (20-23)
        static constexpr bool has_low = false;
        static constexpr bool has_high = true;
        static constexpr bool all_low = false;
        static constexpr bool all_high = true;
    };

    /**
     * @brief Reserved zone: zmm26-zmm31 (for future use / callee-saved)
     *
     * All HIGH registers - require EVEX encoding.
     */
    struct ReservedZone : RegisterZoneBase
    {
        static constexpr int base_index = 26;
        static constexpr int count = 6;
        static constexpr const char *name = "Reserved";

        // Encoding constraints
        static constexpr int low_count = 0;  // None are LOW
        static constexpr int high_count = 6; // All 6 are HIGH (26-31)
        static constexpr bool has_low = false;
        static constexpr bool has_high = true;
        static constexpr bool all_low = false;
        static constexpr bool all_high = true;
    };

    // ============================================================================
    // Zone Conflict Detection (Compile-Time)
    // ============================================================================

    /**
     * @brief Check if two zones overlap (share physical registers)
     */
    template <typename Zone1, typename Zone2>
    struct ZonesOverlap
    {
        static constexpr int z1_start = Zone1::base_index;
        static constexpr int z1_end = Zone1::base_index + Zone1::count;
        static constexpr int z2_start = Zone2::base_index;
        static constexpr int z2_end = Zone2::base_index + Zone2::count;

        static constexpr bool value = (z1_start < z2_end) && (z2_start < z1_end);
    };

    template <typename Zone1, typename Zone2>
    inline constexpr bool zones_overlap_v = ZonesOverlap<Zone1, Zone2>::value;

    // Static assertion: ScoreZone overlaps ScratchZone (by design, they alias)
    static_assert(zones_overlap_v<ScoreZone, ScratchZone>,
                  "ScoreZone and ScratchZone should overlap (XMM/ZMM aliasing)");

    // Static assertion: Core zones don't overlap
    static_assert(!zones_overlap_v<AccumulatorZone, StateZone>,
                  "Accumulator and State zones must not overlap");
    static_assert(!zones_overlap_v<StateZone, ScratchZone>,
                  "State and Scratch zones must not overlap");
    static_assert(!zones_overlap_v<AccumulatorZone, QVectorZone>,
                  "Accumulator and QVector zones must not overlap");

    // ============================================================================
    // Typed Register Wrappers
    // ============================================================================

    /**
     * @brief Compile-time typed ZMM register
     *
     * The zone and index are encoded in the type, enabling compile-time checks.
     *
     * @tparam Zone The register zone (AccumulatorZone, ScratchZone, etc.)
     * @tparam LocalIdx Index within the zone (0 to Zone::count-1)
     */
    /**
     * @brief Compile-time typed ZMM register - TAG TYPE ONLY
     *
     * IMPORTANT: This is a tag type with NO direct register accessors.
     * To get the underlying Xbyak::Zmm/Ymm/Xmm, you MUST borrow via RegisterGuard:
     *
     * @code
     * // WRONG - won't compile:
     * Xbyak::Zmm reg = Scratch0{}.zmm();  // ERROR: no .zmm() method
     *
     * // CORRECT - use RegisterGuard:
     * auto guard = borrow<Scratch0>();
     * gen.vmovaps(guard.zmm(), ...);
     * @endcode
     *
     * This design ENFORCES register tracking - you cannot accidentally
     * use a register without the guard system knowing about it.
     *
     * @tparam Zone The register zone (AccumulatorZone, ScratchZone, etc.)
     * @tparam LocalIdx Index within the zone (0 to Zone::count-1)
     */
    template <typename Zone, int LocalIdx>
    struct TypedZmm
    {
        static_assert(std::is_base_of_v<RegisterZoneBase, Zone>,
                      "Zone must derive from RegisterZoneBase");
        static_assert(LocalIdx >= 0 && LocalIdx < Zone::count,
                      "LocalIdx out of range for this zone");

        using zone_type = Zone;
        static constexpr int local_index = LocalIdx;
        static constexpr int absolute_index = Zone::base_index + LocalIdx;

        // Type marker for trait detection (since we removed .zmm()/.ymm() accessors)
        static constexpr bool is_zmm_type = true;

        // Encoding constraints - determined by absolute register index
        static constexpr bool is_low_register = absolute_index < 16;
        static constexpr bool is_high_register = absolute_index >= 16;
        static constexpr bool is_vex_safe = is_low_register;
        static constexpr bool is_evex_only = is_high_register;
        using encoding_tag = std::conditional_t<is_low_register, LowRegisterTag, HighRegisterTag>;

        // NO .zmm(), .ymm(), .xmm() accessors!
        // Use borrow<ThisType>() to get a RegisterGuard, then call .zmm() on that.
    };

    /**
     * @brief Compile-time typed XMM register
     *
     * Similar to TypedZmm but for scalar/128-bit operations.
     * IMPORTANT: XMM registers alias the low bits of ZMM registers!
     */
    /**
     * @brief Compile-time typed XMM register - TAG TYPE ONLY
     *
     * Similar to TypedZmm - this is a tag type with NO direct accessors.
     * You MUST use RegisterGuard to access the underlying Xbyak register.
     *
     * IMPORTANT: XMM registers alias the low bits of ZMM registers!
     *
     * ENCODING WARNING: XMM16-31 (HIGH) require EVEX encoding.
     * If you need VEX-safe XMM registers, use AccumulatorZone or QVectorZone.
     */
    template <typename Zone, int LocalIdx>
    struct TypedXmm
    {
        static_assert(std::is_base_of_v<RegisterZoneBase, Zone>,
                      "Zone must derive from RegisterZoneBase");
        static_assert(LocalIdx >= 0 && LocalIdx < Zone::count,
                      "LocalIdx out of range for this zone");

        using zone_type = Zone;
        static constexpr int local_index = LocalIdx;
        static constexpr int absolute_index = Zone::base_index + LocalIdx;

        // Type marker for trait detection (since we removed .xmm()/.as_zmm() accessors)
        static constexpr bool is_xmm_type = true;

        // Encoding constraints - determined by absolute register index
        static constexpr bool is_low_register = absolute_index < 16;
        static constexpr bool is_high_register = absolute_index >= 16;
        static constexpr bool is_vex_safe = is_low_register;
        static constexpr bool is_evex_only = is_high_register;
        using encoding_tag = std::conditional_t<is_low_register, LowRegisterTag, HighRegisterTag>;

        // NO .xmm() or .as_zmm() accessors!
        // Use borrow<ThisType>() to get a RegisterGuard, then call .xmm() on that.
    };

    // ============================================================================
    // Type Traits for Zone-Based Constraints (C++17 compatible)
    // ============================================================================

    /**
     * @brief Type trait: Register belongs to a specific zone
     */
    template <typename Reg, typename Zone>
    struct is_zone : std::is_same<typename Reg::zone_type, Zone>
    {
    };

    template <typename Reg, typename Zone>
    inline constexpr bool is_zone_v = is_zone<Reg, Zone>::value;

    /**
     * @brief Type trait: Register is NOT from a specific zone
     */
    template <typename Reg, typename Zone>
    inline constexpr bool is_not_zone_v = !is_zone_v<Reg, Zone>;

    /**
     * @brief Type trait: Register is from any of the listed zones
     */
    template <typename Reg, typename... Zones>
    inline constexpr bool is_any_zone_v = (is_zone_v<Reg, Zones> || ...);

    // ============================================================================
    // Encoding Constraint Type Traits
    // ============================================================================

    /**
     * @brief Type trait: Register is LOW (0-15), VEX-safe
     */
    template <typename Reg>
    inline constexpr bool is_low_register_v = Reg::is_low_register;

    /**
     * @brief Type trait: Register is HIGH (16-31), EVEX-only
     */
    template <typename Reg>
    inline constexpr bool is_high_register_v = Reg::is_high_register;

    /**
     * @brief Type trait: Register can use VEX-encoded instructions (xmm/ymm)
     */
    template <typename Reg>
    inline constexpr bool is_vex_safe_v = Reg::is_vex_safe;

    /**
     * @brief Type trait: Register requires EVEX encoding for xmm/ymm ops
     */
    template <typename Reg>
    inline constexpr bool is_evex_only_v = Reg::is_evex_only;

    // ============================================================================
    // Convenient Type Aliases
    // ============================================================================

    // Accumulator registers (zmm0-zmm7)
    template <int N>
    using AccumZmm = TypedZmm<AccumulatorZone, N>;
    using Accum0 = AccumZmm<0>;
    using Accum1 = AccumZmm<1>;
    using Accum2 = AccumZmm<2>;
    using Accum3 = AccumZmm<3>;
    using Accum4 = AccumZmm<4>;
    using Accum5 = AccumZmm<5>;
    using Accum6 = AccumZmm<6>;
    using Accum7 = AccumZmm<7>;

    // Q vector registers (zmm8-zmm15) - also called "Input" zone
    template <int N>
    using QVecZmm = TypedZmm<QVectorZone, N>;
    using Input0 = QVecZmm<0>; // zmm8
    using Input1 = QVecZmm<1>; // zmm9
    using Input2 = QVecZmm<2>; // zmm10
    using Input3 = QVecZmm<3>; // zmm11
    using Input4 = QVecZmm<4>; // zmm12
    using Input5 = QVecZmm<5>; // zmm13
    using Input6 = QVecZmm<6>; // zmm14
    using Input7 = QVecZmm<7>; // zmm15

    // State registers (zmm16-zmm19)
    using StateMax = TypedZmm<StateZone, StateZone::MAX_IDX>;
    using StateSum = TypedZmm<StateZone, StateZone::SUM_IDX>;
    using StateWeight = TypedZmm<StateZone, StateZone::WEIGHT_IDX>;
    using StateCorr = TypedZmm<StateZone, StateZone::CORR_IDX>;

    // Scratch registers (zmm20-zmm25)
    template <int N>
    using ScratchZmm = TypedZmm<ScratchZone, N>;
    using Scratch0 = ScratchZmm<0>; // zmm20 - WARNING: aliases xmm20 (Score0)
    using Scratch1 = ScratchZmm<1>; // zmm21 - WARNING: aliases xmm21 (Score1)
    using Scratch2 = ScratchZmm<2>; // zmm22 - WARNING: aliases xmm22 (Score2)
    using Scratch3 = ScratchZmm<3>; // zmm23 - WARNING: aliases xmm23 (Score3)
    using Scratch4 = ScratchZmm<4>; // zmm24 - Safe during FA2 scoring
    using Scratch5 = ScratchZmm<5>; // zmm25 - Safe during FA2 scoring

    // Score registers (xmm20-xmm23) - for FA2 4-way tile processing
    template <int N>
    using ScoreXmm = TypedXmm<ScoreZone, N>;
    using Score0 = ScoreXmm<0>; // xmm20
    using Score1 = ScoreXmm<1>; // xmm21
    using Score2 = ScoreXmm<2>; // xmm22
    using Score3 = ScoreXmm<3>; // xmm23

    // Reserved/Constant registers (zmm26-zmm31) - preloaded constants
    // These are loaded once at kernel init and never modified during execution
    template <int N>
    using ReservedZmm = TypedZmm<ReservedZone, N>;
    using Const128 = ReservedZmm<0>;    // zmm26: 128.0f (Q8 dequant scale) or 0x80808080
    using ConstScale = ReservedZmm<1>;  // zmm27: attention scale (rsqrt(head_dim))
    using ConstNegInf = ReservedZmm<2>; // zmm28: -inf (decode) OR 16.0f (prefill Q8_1 correction)
    using Const16 = ReservedZmm<2>;     // zmm28: alias - shares with ConstNegInf (mode-dependent)
    using ConstOne = ReservedZmm<3>;    // zmm29: 1.0f
    using ConstLog2e = ReservedZmm<4>;  // zmm30: log2(e) for fast exp
    using ConstExpMin = ReservedZmm<5>; // zmm31: -87.0f (exp underflow clamp)

    // ============================================================================
    // Register Set for Function Signatures
    // ============================================================================

    /**
     * @brief Declare what registers a function uses (for documentation + future checking)
     *
     * This is primarily for documentation now, but could be extended to
     * do runtime or compile-time conflict detection.
     */
    template <typename... Regs>
    struct UsesRegisters
    {
        static constexpr size_t count = sizeof...(Regs);

        // Check if any pair conflicts (same absolute index)
        static constexpr bool has_conflicts()
        {
            if constexpr (count < 2)
            {
                return false;
            }
            else
            {
                constexpr int indices[] = {Regs::absolute_index...};
                for (size_t i = 0; i < count; ++i)
                {
                    for (size_t j = i + 1; j < count; ++j)
                    {
                        if (indices[i] == indices[j])
                            return true;
                    }
                }
                return false;
            }
        }

        static_assert(!has_conflicts(), "Register set contains conflicting registers!");
    };

    // ============================================================================
    // SFINAE helpers for constrained function templates (C++17)
    // ============================================================================

    /**
     * @brief Enable function if Reg is from Zone
     */
    template <typename Reg, typename Zone>
    using require_zone = std::enable_if_t<is_zone_v<Reg, Zone>, bool>;

    /**
     * @brief Enable function if Reg is from ScratchZone and is safe during FA2
     * (i.e., local_index >= 4, meaning zmm24 or zmm25)
     */
    template <typename Reg>
    using require_safe_scratch_for_fa2 = std::enable_if_t<
        is_zone_v<Reg, ScratchZone> && (Reg::local_index >= 4), bool>;

    /**
     * @brief Enable function if Reg is VEX-safe (LOW register, 0-15)
     *
     * Use this to constrain functions that emit VEX-encoded instructions
     * on xmm/ymm registers.
     */
    template <typename Reg>
    using require_vex_safe = std::enable_if_t<is_vex_safe_v<Reg>, bool>;

    /**
     * @brief Enable function if Reg is a LOW register (0-15)
     */
    template <typename Reg>
    using require_low_register = std::enable_if_t<is_low_register_v<Reg>, bool>;

    /**
     * @brief Enable function if Reg is a HIGH register (16-31)
     */
    template <typename Reg>
    using require_high_register = std::enable_if_t<is_high_register_v<Reg>, bool>;

} // namespace llaminar2::jit
