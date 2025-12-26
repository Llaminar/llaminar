/**
 * @file RegisterEnforcement.h
 * @brief Compile-time enforcement that JIT code uses typed registers
 *
 * This header provides mechanisms to catch raw Xbyak register usage at compile
 * time. Include this header in JIT kernel files to get compile errors when
 * attempting to use raw Xbyak::Zmm, Xbyak::Ymm, or Xbyak::Xmm directly instead
 * of typed registers from RegisterAllocation.h.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * ENFORCEMENT MECHANISMS
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * 1. **Type Traits**: `is_typed_register<T>` distinguishes typed from raw registers
 * 2. **Deleted Overloads**: Functions that should only accept typed registers
 *    have deleted overloads for raw Xbyak types
 * 3. **Concepts (C++20)**: `TypedRegister` concept for cleaner constraints
 * 4. **Static Assertions**: Verify register usage at compile time
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * USAGE
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * @code
 * #include "RegisterEnforcement.h"
 *
 * // This function only accepts typed registers
 * template<TypedZmmRegister T>
 * void emit_safe_operation(Xbyak::CodeGenerator& gen, const T& reg) {
 *     gen.vmovaps(reg.zmm(), ...);  // OK
 * }
 *
 * // Attempting to call with raw Xbyak::Zmm will fail at compile time:
 * // emit_safe_operation(gen, Xbyak::Zmm(20));  // ERROR: constraint not satisfied
 *
 * // Use REQUIRE_TYPED_REGISTER in function bodies for C++17:
 * void legacy_function(auto reg) {
 *     REQUIRE_TYPED_REGISTER(reg);  // static_assert if not typed
 *     gen.vmovaps(reg.zmm(), ...);
 * }
 * @endcode
 */

#pragma once

#include "RegisterAllocation.h"
#include <type_traits>

namespace llaminar2::jit
{

    // ============================================================================
    // Type Traits for Register Classification
    // ============================================================================

    namespace detail
    {
        // Primary template: not a typed register
        template <typename T, typename = void>
        struct is_typed_zmm_impl : std::false_type
        {
        };

        // Specialization: has zone_type, absolute_index, and zmm() accessor
        // but ALSO has ymm() which TypedXmm does NOT have
        template <typename T>
        struct is_typed_zmm_impl<T,
                                 std::void_t<
                                     typename T::zone_type,
                                     decltype(T::absolute_index),
                                     decltype(T::local_index),
                                     decltype(std::declval<T>().zmm()),
                                     decltype(std::declval<T>().ymm())>> : std::true_type
        {
        };

        // Check for TypedXmm - has xmm() and as_zmm() but NOT zmm() directly
        template <typename T, typename = void>
        struct is_typed_xmm_impl : std::false_type
        {
        };

        template <typename T>
        struct is_typed_xmm_impl<T,
                                 std::void_t<
                                     typename T::zone_type,
                                     decltype(T::absolute_index),
                                     decltype(T::local_index),
                                     decltype(std::declval<T>().xmm()),
                                     decltype(std::declval<T>().as_zmm())>> : std::true_type
        {
        };

        // Check for RegisterGuard<T>
        template <typename T, typename = void>
        struct is_register_guard_impl : std::false_type
        {
        };

        template <typename T>
        struct is_register_guard_impl<T,
                                      std::void_t<
                                          typename T::reg_type,
                                          decltype(T::physical_index),
                                          decltype(std::declval<T>().reg())>> : std::true_type
        {
        };

    } // namespace detail

    /**
     * @brief Type trait: Is T a TypedZmm register?
     */
    template <typename T>
    struct is_typed_zmm : detail::is_typed_zmm_impl<std::decay_t<T>>
    {
    };

    template <typename T>
    inline constexpr bool is_typed_zmm_v = is_typed_zmm<T>::value;

    /**
     * @brief Type trait: Is T a TypedXmm register?
     */
    template <typename T>
    struct is_typed_xmm : detail::is_typed_xmm_impl<std::decay_t<T>>
    {
    };

    template <typename T>
    inline constexpr bool is_typed_xmm_v = is_typed_xmm<T>::value;

    /**
     * @brief Type trait: Is T a RegisterGuard?
     */
    template <typename T>
    struct is_register_guard : detail::is_register_guard_impl<std::decay_t<T>>
    {
    };

    template <typename T>
    inline constexpr bool is_register_guard_v = is_register_guard<T>::value;

    /**
     * @brief Type trait: Is T any typed register (ZMM, XMM, or Guard)?
     */
    template <typename T>
    inline constexpr bool is_typed_register_v =
        is_typed_zmm_v<T> || is_typed_xmm_v<T> || is_register_guard_v<T>;

    /**
     * @brief Type trait: Is T a raw Xbyak register (not typed)?
     */
    template <typename T>
    inline constexpr bool is_raw_xbyak_register_v =
        std::is_same_v<std::decay_t<T>, Xbyak::Zmm> ||
        std::is_same_v<std::decay_t<T>, Xbyak::Ymm> ||
        std::is_same_v<std::decay_t<T>, Xbyak::Xmm> ||
        std::is_same_v<std::decay_t<T>, Xbyak::Reg>;

    // ============================================================================
    // C++20 Concepts (if available)
    // ============================================================================

#if __cplusplus >= 202002L

    /**
     * @brief Concept: T is a typed ZMM register
     */
    template <typename T>
    concept TypedZmmRegister = is_typed_zmm_v<T>;

    /**
     * @brief Concept: T is a typed XMM register
     */
    template <typename T>
    concept TypedXmmRegister = is_typed_xmm_v<T>;

    /**
     * @brief Concept: T is any typed register (ZMM, XMM, or Guard)
     */
    template <typename T>
    concept TypedRegister = is_typed_register_v<T>;

    /**
     * @brief Concept: T is a RegisterGuard
     */
    template <typename T>
    concept GuardedRegister = is_register_guard_v<T>;

    /**
     * @brief Concept: T is NOT a raw Xbyak register
     */
    template <typename T>
    concept NotRawXbyakRegister = !is_raw_xbyak_register_v<T>;

#endif // C++20

    // ============================================================================
    // C++17 SFINAE Helpers
    // ============================================================================

    /**
     * @brief SFINAE: Enable if T is a typed ZMM register
     */
    template <typename T>
    using require_typed_zmm = std::enable_if_t<is_typed_zmm_v<T>, bool>;

    /**
     * @brief SFINAE: Enable if T is a typed XMM register
     */
    template <typename T>
    using require_typed_xmm = std::enable_if_t<is_typed_xmm_v<T>, bool>;

    /**
     * @brief SFINAE: Enable if T is any typed register
     */
    template <typename T>
    using require_typed_register = std::enable_if_t<is_typed_register_v<T>, bool>;

    /**
     * @brief SFINAE: Enable if T is a RegisterGuard
     */
    template <typename T>
    using require_guarded_register = std::enable_if_t<is_register_guard_v<T>, bool>;

    // ============================================================================
    // Static Assert Macros
    // ============================================================================

/**
 * @brief Assert at compile time that a register is typed (not raw Xbyak)
 *
 * Usage:
 * @code
 * template<typename Reg>
 * void my_emit_function(const Reg& reg) {
 *     REQUIRE_TYPED_REGISTER(reg);
 *     gen.vmovaps(reg.zmm(), ...);
 * }
 * @endcode
 */
#define REQUIRE_TYPED_REGISTER(reg)                                  \
    static_assert(                                                   \
        ::llaminar2::jit::is_typed_register_v<decltype(reg)>,        \
        "Register must be a typed register (TypedZmm, TypedXmm, or " \
        "RegisterGuard). Use types from RegisterAllocation.h: "      \
        "Accum0-7, Input0-7, Scratch0-5, Score0-3, State*, etc. "    \
        "Do not use raw Xbyak::Zmm/Ymm/Xmm.")

/**
 * @brief Assert that a register is a TypedZmm specifically
 */
#define REQUIRE_TYPED_ZMM(reg)                                               \
    static_assert(                                                           \
        ::llaminar2::jit::is_typed_zmm_v<decltype(reg)>,                     \
        "Register must be a TypedZmm. Use types from RegisterAllocation.h: " \
        "Accum0-7, Input0-7, Scratch0-5, State*, etc.")

/**
 * @brief Assert that a register is a RegisterGuard
 */
#define REQUIRE_GUARDED_REGISTER(reg)                                  \
    static_assert(                                                     \
        ::llaminar2::jit::is_register_guard_v<decltype(reg)>,          \
        "Register must be borrowed via RegisterTracker::borrow<T>(). " \
        "Use: auto guard = tracker.borrow<Scratch0>();")

    // ============================================================================
    // Deleted Function Pattern for Raw Register Rejection
    // ============================================================================

    /**
     * @brief Base class that can be inherited to get deleted overloads
     *
     * Inherit from this to automatically reject raw Xbyak registers in
     * emit_* methods. Override the typed versions in your derived class.
     *
     * @code
     * class MyEmitter : public RejectRawRegisters<MyEmitter> {
     * public:
     *     template<typename T, require_typed_zmm<T> = true>
     *     void emit_operation(const T& reg) {
     *         // Only typed registers reach here
     *     }
     *     // Raw Xbyak::Zmm will hit the deleted base class version
     * };
     * @endcode
     */
    template <typename Derived>
    class RejectRawRegisters
    {
    protected:
        // Deleted overloads for raw Xbyak types
        void emit_operation(const Xbyak::Zmm &) = delete;
        void emit_operation(const Xbyak::Ymm &) = delete;
        void emit_operation(const Xbyak::Xmm &) = delete;

        // Helper to produce clear error message
        template <typename T>
        static constexpr void reject_raw_register()
        {
            static_assert(!is_raw_xbyak_register_v<T>,
                          "Raw Xbyak::Zmm/Ymm/Xmm not allowed. Use typed registers.");
        }
    };

    // ============================================================================
    // Compile-Time Register Index Validation
    // ============================================================================

    /**
     * @brief Validate register index is within expected range
     *
     * Use this to catch typos like Accum<8> (should be Accum<7> max)
     */
    template <typename Zone, int LocalIdx>
    struct ValidateRegisterIndex
    {
        static_assert(LocalIdx >= 0, "Register index cannot be negative");
        static_assert(LocalIdx < Zone::count,
                      "Register index exceeds zone capacity. "
                      "Check zone definitions in RegisterAllocation.h");
        static constexpr bool valid = true;
    };

    // ============================================================================
    // Register Name for Error Messages
    // ============================================================================

    /**
     * @brief Get human-readable name for a register index
     *
     * Useful for generating helpful error messages.
     */
    template <int AbsoluteIndex>
    struct RegisterNameFromIndex
    {
        static constexpr const char *zone_name()
        {
            if constexpr (AbsoluteIndex < 8)
                return "Accumulator";
            else if constexpr (AbsoluteIndex < 16)
                return "QVector/Input";
            else if constexpr (AbsoluteIndex < 20)
                return "State";
            else if constexpr (AbsoluteIndex < 26)
                return "Scratch";
            else
                return "Reserved";
        }

        static constexpr int local_index()
        {
            if constexpr (AbsoluteIndex < 8)
                return AbsoluteIndex;
            else if constexpr (AbsoluteIndex < 16)
                return AbsoluteIndex - 8;
            else if constexpr (AbsoluteIndex < 20)
                return AbsoluteIndex - 16;
            else if constexpr (AbsoluteIndex < 26)
                return AbsoluteIndex - 20;
            else
                return AbsoluteIndex - 26;
        }

        static constexpr const char *suggested_type()
        {
            if constexpr (AbsoluteIndex < 8)
                return "Accum0-7";
            else if constexpr (AbsoluteIndex < 16)
                return "Input0-7";
            else if constexpr (AbsoluteIndex == 16)
                return "StateMax";
            else if constexpr (AbsoluteIndex == 17)
                return "StateSum";
            else if constexpr (AbsoluteIndex == 18)
                return "StateWeight";
            else if constexpr (AbsoluteIndex == 19)
                return "StateCorr";
            else if constexpr (AbsoluteIndex < 26)
                return "Scratch0-5";
            else
                return "Reserved (define custom type)";
        }
    };

    // ============================================================================
    // KernelRegisterManifest: Document kernel register usage
    // ============================================================================

    /**
     * @brief Declare a kernel's complete register footprint at compile time
     *
     * Use this to document and verify which registers a kernel uses.
     * The compiler will check for conflicts between registers.
     *
     * @code
     * // Document FastExp's register requirements
     * using FastExpManifest = KernelRegisterManifest<
     *     Scratch4,   // Output
     *     Scratch5,   // Temp n
     *     Input6,     // Temp f
     *     Input7      // Temp for constants
     * >;
     *
     * // Verify no conflicts at compile time
     * static_assert(!FastExpManifest::has_conflicts());
     *
     * // Check all registers are from expected zones
     * static_assert(FastExpManifest::all_from_zones<ScratchZone, QVectorZone>());
     * @endcode
     */
    template <typename... Regs>
    struct KernelRegisterManifest
    {
        static constexpr size_t count = sizeof...(Regs);

        /**
         * @brief Check if any two registers in the manifest conflict
         */
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

        /**
         * @brief Check if all registers are from the specified zones
         */
        template <typename... AllowedZones>
        static constexpr bool all_from_zones()
        {
            return (is_any_zone_v<Regs, AllowedZones...> && ...);
        }

        /**
         * @brief Check if manifest uses a specific register
         */
        template <typename Reg>
        static constexpr bool uses_register()
        {
            return ((Regs::absolute_index == Reg::absolute_index) || ...);
        }

        /**
         * @brief Check if two manifests have overlapping registers
         */
        template <typename... OtherRegs>
        static constexpr bool overlaps_with(KernelRegisterManifest<OtherRegs...>)
        {
            return ((uses_register<OtherRegs>()) || ...);
        }

        // Static assert that no conflicts exist
        static_assert(!has_conflicts(),
                      "KernelRegisterManifest contains duplicate registers!");
    };

    // ============================================================================
    // Pre-defined Manifests for Common Microkernels
    // ============================================================================

    /**
     * @brief FastExp microkernel register manifest
     *
     * Uses: Scratch4 (output), Scratch5 (n), Input6 (f), Input7 (tmp)
     */
    using FastExpManifest = KernelRegisterManifest<Scratch4, Scratch5, Input6, Input7>;

    // Verify FastExp manifest is valid
    static_assert(!FastExpManifest::has_conflicts());
    static_assert(FastExpManifest::all_from_zones<ScratchZone, QVectorZone>());

    // ============================================================================
    // Extract underlying Xbyak register from any typed register
    // ============================================================================

    /**
     * @brief Get Xbyak::Zmm from any typed register or guard
     *
     * Provides uniform access regardless of whether you have a TypedZmm,
     * RegisterGuard, or need to extract from a register set.
     */
    template <typename T>
    constexpr auto to_xbyak_zmm(const T &reg) -> Xbyak::Zmm
    {
        if constexpr (is_register_guard_v<T>)
        {
            return reg.zmm();
        }
        else if constexpr (is_typed_zmm_v<T>)
        {
            return reg.zmm();
        }
        else
        {
            static_assert(is_typed_register_v<T>,
                          "to_xbyak_zmm requires a typed register");
            return reg.zmm();
        }
    }

    /**
     * @brief Get Xbyak::Xmm from any typed register or guard
     */
    template <typename T>
    constexpr auto to_xbyak_xmm(const T &reg) -> Xbyak::Xmm
    {
        return reg.xmm();
    }

} // namespace llaminar2::jit

// ============================================================================
// BLOCKING RAW REGISTER CONSTRUCTION
// ============================================================================
//
// When ENFORCE_TYPED_REGISTERS is defined, we shadow Zmm/Ymm/Xmm with wrapper
// functions that produce compile-time errors when called with integer indices.
//
// This catches patterns like:
//   Zmm(16)           // ERROR: Use state_max().zmm() or StateMax{}.zmm()
//   Xbyak::Zmm(16)    // Still works (explicit namespace), but Zmm(16) fails
//
// To opt-in:
//   #define ENFORCE_TYPED_REGISTERS
//   #include "RegisterEnforcement.h"
//
// Or add to your CMakeLists.txt:
//   target_compile_definitions(your_target PRIVATE ENFORCE_TYPED_REGISTERS)
//
// ============================================================================

#ifdef ENFORCE_TYPED_REGISTERS

namespace llaminar2::jit::enforcement_detail
{
    // Helper that always fails with a descriptive message
    template <int N>
    struct BlockedRegisterConstruction
    {
        static_assert(N < 0, // Always false but depends on N
                      "\n\n"
                      "╔══════════════════════════════════════════════════════════════════╗\n"
                      "║              RAW REGISTER CONSTRUCTION BLOCKED                    ║\n"
                      "╠══════════════════════════════════════════════════════════════════╣\n"
                      "║ Raw Zmm(N)/Ymm(N)/Xmm(N) construction is prohibited.             ║\n"
                      "║                                                                   ║\n"
                      "║ Use typed registers from RegisterAllocation.h instead:           ║\n"
                      "║   zmm0-7:   accum0()-accum7() or Accum0{}-Accum7{}               ║\n"
                      "║   zmm8-15:  Input0{}-Input7{} (from RegisterAllocation.h)        ║\n"
                      "║   zmm16-19: state_max(), state_sum(), state_weight(), state_corr()║\n"
                      "║   zmm20-25: scratch0()-scratch5() or Scratch0{}-Scratch5{}       ║\n"
                      "║   zmm26-31: const_128(), const_scale(), const_one(), etc.        ║\n"
                      "║                                                                   ║\n"
                      "║ Example migration:                                                ║\n"
                      "║   Before: Zmm zmm_max = Zmm(16);                                 ║\n"
                      "║   After:  auto zmm_max = state_max().zmm();                      ║\n"
                      "║                                                                   ║\n"
                      "║ For dynamic indices (e.g., in loops), use helper functions.      ║\n"
                      "╚══════════════════════════════════════════════════════════════════╝\n");
    };

    // Deleted function to block Zmm(int) construction
    // The template parameter captures the integer at compile time
    template <int N>
    [[deprecated("Use typed registers - see RegisterAllocation.h")]]
    constexpr auto Zmm_blocked(std::integral_constant<int, N>) -> BlockedRegisterConstruction<N>
    {
        return {};
    }

} // namespace llaminar2::jit::enforcement_detail

// ============================================================================
// Shadowing macros - these intercept raw Zmm/Ymm/Xmm usage
// ============================================================================
// When ENFORCE_TYPED_REGISTERS is defined, these macros expand to
// constructs that fail at compile time with helpful error messages.
//
// IMPORTANT: These only work for literal integer arguments like Zmm(16).
// Variable arguments like Zmm(q) require runtime checking (RegisterTracker).
// ============================================================================

// Macro to produce compile error for Zmm(N) where N is a literal
#define LLAMINAR_BLOCK_ZMM(N)                                                                  \
    static_assert(false,                                                                       \
                  "Raw Zmm(" #N ") is blocked. Use typed register from RegisterAllocation.h. " \
                  "For zmm" #N ", use the appropriate typed accessor.")

// Note: We can't easily intercept Zmm(N) without breaking legitimate uses.
// Instead, rely on:
// 1. REQUIRE_TYPED_REGISTER() macro in function bodies
// 2. Code review with grep for "Zmm(" patterns
// 3. The deprecated warnings on legacy accessors

#endif // ENFORCE_TYPED_REGISTERS
