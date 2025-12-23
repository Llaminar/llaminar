/**
 * @file RegisterGuard.h
 * @brief RAII-based register tracking to detect conflicts at JIT-compile time
 *
 * This framework builds on RegisterAllocation.h to add RUNTIME checks during
 * JIT code generation. When a register is "borrowed", it's marked as in-use.
 * If another piece of code tries to borrow the same register (or an aliased
 * register), an assertion fires immediately.
 *
 * Key insight: XMM/YMM/ZMM registers all share the same physical register file.
 * If you borrow zmm20, you cannot simultaneously use xmm20 or ymm20.
 *
 * Usage:
 * @code
 * RegisterTracker tracker;
 *
 * void emit_phase1() {
 *     auto score0 = tracker.borrow<Score0>();  // Marks physical reg 20 as borrowed
 *     gen.vmovss(score0.xmm(), ...);
 *     // ~RegisterGuard releases when scope ends
 * }
 *
 * void emit_phase2() {
 *     auto scratch0 = tracker.borrow<Scratch0>();  // FAILS if Score0 still borrowed!
 *     gen.vmovaps(scratch0.zmm(), ...);
 * }
 * @endcode
 *
 * @note This adds overhead during JIT compilation, not during JIT execution.
 *       The generated assembly is identical whether or not guards are used.
 */

#pragma once

#include "RegisterAllocation.h"
#include <bitset>
#include <cassert>
#include <iostream>
#include <string>
#include <array>
#include <utility>

namespace llaminar2::jit
{

    // Forward declarations
    class RegisterTracker;

    template <typename... RegTypes>
    class ScopedRegisterSet;

    // ============================================================================
    // RegisterGuard: RAII wrapper for borrowed registers
    // ============================================================================

    /**
     * @brief RAII guard for a borrowed register
     *
     * When constructed, marks the register as borrowed in the tracker.
     * When destroyed, releases the register back to the tracker.
     * Move-only to prevent double-release.
     *
     * @tparam RegType The typed register (e.g., Score0, Scratch4, Accum0)
     */
    template <typename RegType>
    class RegisterGuard
    {
    public:
        using reg_type = RegType;
        static constexpr int physical_index = RegType::absolute_index;

        // Constructor: takes ownership of the borrow
        RegisterGuard(RegisterTracker *tracker, RegType reg);

        // Destructor: releases the register
        ~RegisterGuard();

        // Move constructor
        RegisterGuard(RegisterGuard &&other) noexcept
            : tracker_(other.tracker_), reg_(other.reg_)
        {
            other.tracker_ = nullptr; // Prevent release in moved-from object
        }

        // Move assignment
        RegisterGuard &operator=(RegisterGuard &&other) noexcept
        {
            if (this != &other)
            {
                release_if_owned();
                tracker_ = other.tracker_;
                reg_ = other.reg_;
                other.tracker_ = nullptr;
            }
            return *this;
        }

        // No copy
        RegisterGuard(const RegisterGuard &) = delete;
        RegisterGuard &operator=(const RegisterGuard &) = delete;

        // Access the underlying typed register
        RegType reg() const { return reg_; }

        // Convenience accessors - use SFINAE to only enable what the underlying type supports
        template <typename R = RegType>
        auto zmm() const -> decltype(std::declval<R>().zmm()) { return reg_.zmm(); }

        template <typename R = RegType>
        auto ymm() const -> decltype(std::declval<R>().ymm()) { return reg_.ymm(); }

        Xbyak::Xmm xmm() const { return reg_.xmm(); }

        // Explicit release (rarely needed, destructor handles it)
        void release();

    private:
        void release_if_owned();

        RegisterTracker *tracker_;
        RegType reg_;
    };

    // ============================================================================
    // RegisterTracker: Tracks which registers are currently borrowed
    // ============================================================================

    /**
     * @brief Tracks register borrows and detects conflicts
     *
     * Maintains a bitset of borrowed physical registers. Since XMM/YMM/ZMM
     * all map to the same physical register file (0-31), we only need one bitset.
     *
     * Debug mode provides detailed conflict messages.
     */
    class RegisterTracker
    {
    public:
        static constexpr int NUM_REGS = 32;

        RegisterTracker() = default;

        // Non-copyable, non-movable (tracking state shouldn't be shared)
        RegisterTracker(const RegisterTracker &) = delete;
        RegisterTracker &operator=(const RegisterTracker &) = delete;

        /**
         * @brief Borrow a typed register
         *
         * @tparam RegType Typed register (e.g., Score0, Scratch4)
         * @return RegisterGuard<RegType> RAII guard that releases on destruction
         * @throws Assertion failure if register is already borrowed
         */
        template <typename RegType>
        [[nodiscard]] RegisterGuard<RegType> borrow()
        {
            return RegisterGuard<RegType>(this, RegType{});
        }

        /**
         * @brief Borrow a register with explicit instance (for interface compatibility)
         */
        template <typename RegType>
        [[nodiscard]] RegisterGuard<RegType> borrow(RegType reg)
        {
            return RegisterGuard<RegType>(this, reg);
        }

        /**
         * @brief Borrow multiple registers at once as a ScopedRegisterSet
         *
         * @code
         * auto regs = tracker.borrow_set<Scratch4, Scratch5>();
         * gen.vmovaps(regs.get<0>().zmm(), ...);
         * gen.vmovaps(regs.get<1>().zmm(), ...);
         * @endcode
         */
        template <typename... RegTypes>
        [[nodiscard]] ScopedRegisterSet<RegTypes...> borrow_set()
        {
            return ScopedRegisterSet<RegTypes...>(*this);
        }

        /**
         * @brief Check if a physical register is currently borrowed
         */
        bool is_borrowed(int physical_index) const
        {
            assert(physical_index >= 0 && physical_index < NUM_REGS);
            return borrowed_[physical_index];
        }

        /**
         * @brief Check if a typed register is currently borrowed
         */
        template <typename RegType>
        bool is_borrowed() const
        {
            return is_borrowed(RegType::absolute_index);
        }

        /**
         * @brief Get the name of the register that borrowed a physical index
         * @return Empty string if not borrowed, otherwise the borrower's name
         */
        const std::string &borrower_name(int physical_index) const
        {
            assert(physical_index >= 0 && physical_index < NUM_REGS);
            return borrower_names_[physical_index];
        }

        /**
         * @brief Reset all borrows (use with caution, mainly for testing)
         */
        void reset()
        {
            borrowed_.reset();
            for (auto &name : borrower_names_)
                name.clear();
        }

        /**
         * @brief Get count of currently borrowed registers
         */
        size_t borrowed_count() const
        {
            return borrowed_.count();
        }

        /**
         * @brief Assert that a physical register is NOT borrowed
         *
         * Use this to detect raw accessor usage conflicts:
         * @code
         * // In emit code that uses gen.accum4().zmm() directly:
         * if (gen.tracker()) gen.tracker()->assert_available<Accum4>("emit_foo context");
         * auto zmm = gen.accum4().zmm();
         * @endcode
         *
         * @tparam RegType The typed register to check
         * @param context Description of where this access is happening
         * @throws Assertion failure if register is borrowed
         */
        template <typename RegType>
        void assert_available(const char *context = "")
        {
            constexpr int idx = RegType::absolute_index;
            if (borrowed_[idx])
            {
                std::cerr << "\n╔══════════════════════════════════════════════════════════════════╗\n";
                std::cerr << "║              RAW ACCESSOR CONFLICT DETECTED                       ║\n";
                std::cerr << "╠══════════════════════════════════════════════════════════════════╣\n";
                std::cerr << "║ Physical register: zmm/ymm/xmm" << idx << "\n";
                std::cerr << "║ Currently borrowed by: '" << borrower_names_[idx] << "'\n";
                std::cerr << "║ Raw access attempted";
                if (context && context[0])
                {
                    std::cerr << " in: " << context;
                }
                std::cerr << "\n";
                std::cerr << "║\n";
                std::cerr << "║ FIX: Either:\n";
                std::cerr << "║   1. Use borrow<" << RegType::zone_type::name << "["
                          << RegType::local_index << "]>() instead of raw accessor\n";
                std::cerr << "║   2. Use a different register that isn't borrowed\n";
                std::cerr << "║   3. Release the borrower before this access\n";
                std::cerr << "╠══════════════════════════════════════════════════════════════════╣\n";
                std::cerr << debug_string() << "\n";
                std::cerr << "╚══════════════════════════════════════════════════════════════════╝\n";
                std::cerr.flush();
                assert(false && "Raw accessor used on borrowed register - see error above");
            }
        }

        /**
         * @brief Assert availability using physical index (for non-templated contexts)
         */
        void assert_available(int physical_index, const char *reg_name, const char *context = "")
        {
            if (borrowed_[physical_index])
            {
                std::cerr << "\n╔══════════════════════════════════════════════════════════════════╗\n";
                std::cerr << "║              RAW ACCESSOR CONFLICT DETECTED                       ║\n";
                std::cerr << "╠══════════════════════════════════════════════════════════════════╣\n";
                std::cerr << "║ Physical register: zmm/ymm/xmm" << physical_index << " (" << reg_name << ")\n";
                std::cerr << "║ Currently borrowed by: '" << borrower_names_[physical_index] << "'\n";
                std::cerr << "║ Raw access attempted";
                if (context && context[0])
                {
                    std::cerr << " in: " << context;
                }
                std::cerr << "\n";
                std::cerr << "╠══════════════════════════════════════════════════════════════════╣\n";
                std::cerr << debug_string() << "\n";
                std::cerr << "╚══════════════════════════════════════════════════════════════════╝\n";
                std::cerr.flush();
                assert(false && "Raw accessor used on borrowed register - see error above");
            }
        }

        /**
         * @brief Print borrowed registers (for debugging)
         */
        std::string debug_string() const
        {
            std::string result = "Borrowed registers: ";
            bool first = true;
            for (int i = 0; i < NUM_REGS; ++i)
            {
                if (borrowed_[i])
                {
                    if (!first)
                        result += ", ";
                    result += "r" + std::to_string(i);
                    if (!borrower_names_[i].empty())
                    {
                        result += "(" + borrower_names_[i] + ")";
                    }
                    first = false;
                }
            }
            if (first)
                result += "(none)";
            return result;
        }

    private:
        template <typename RegType>
        friend class RegisterGuard;

        void mark_borrowed(int physical_index, const std::string &name)
        {
            assert(physical_index >= 0 && physical_index < NUM_REGS);
            if (borrowed_[physical_index])
            {
                // Conflict! Print helpful error message BEFORE asserting
                std::cerr << "\n╔══════════════════════════════════════════════════════════════════╗\n";
                std::cerr << "║                    REGISTER CONFLICT DETECTED                     ║\n";
                std::cerr << "╠══════════════════════════════════════════════════════════════════╣\n";
                std::cerr << "║ Physical register: zmm/ymm/xmm" << physical_index << "\n";
                if (!borrower_names_[physical_index].empty())
                {
                    std::cerr << "║ Already borrowed by: '" << borrower_names_[physical_index] << "'\n";
                }
                std::cerr << "║ Attempted borrow by: '" << name << "'\n";
                std::cerr << "╠══════════════════════════════════════════════════════════════════╣\n";
                std::cerr << debug_string();
                std::cerr << "╚══════════════════════════════════════════════════════════════════╝\n";
                std::cerr.flush();
                assert(false && "Register conflict detected - see error message above");
            }
            borrowed_.set(physical_index);
            borrower_names_[physical_index] = name;
        }

        void mark_released(int physical_index)
        {
            assert(physical_index >= 0 && physical_index < NUM_REGS);
            assert(borrowed_[physical_index] && "Releasing unborrowed register!");
            borrowed_.reset(physical_index);
            borrower_names_[physical_index].clear();
        }

        std::bitset<NUM_REGS> borrowed_;
        std::array<std::string, NUM_REGS> borrower_names_;
    };

    // ============================================================================
    // RegisterGuard Implementation
    // ============================================================================

    template <typename RegType>
    RegisterGuard<RegType>::RegisterGuard(RegisterTracker *tracker, RegType reg)
        : tracker_(tracker), reg_(reg)
    {
        if (tracker_)
        {
            // Generate a name for error messages
            std::string name = std::string(RegType::zone_type::name) +
                               "[" + std::to_string(RegType::local_index) + "]";
            tracker_->mark_borrowed(physical_index, name);
        }
    }

    template <typename RegType>
    RegisterGuard<RegType>::~RegisterGuard()
    {
        release_if_owned();
    }

    template <typename RegType>
    void RegisterGuard<RegType>::release()
    {
        release_if_owned();
        tracker_ = nullptr;
    }

    template <typename RegType>
    void RegisterGuard<RegType>::release_if_owned()
    {
        if (tracker_)
        {
            tracker_->mark_released(physical_index);
        }
    }

    // ============================================================================
    // ScopedRegisterSet: Borrow multiple registers at once
    // ============================================================================

    /**
     * @brief Borrow a set of registers that must be held together
     *
     * Useful when a function needs multiple scratch registers and wants
     * to ensure they're all available at once.
     *
     * @code
     * auto regs = tracker.borrow_set<Scratch4, Scratch5>();
     * gen.vmovaps(regs.get<0>().zmm(), ...);
     * gen.vmovaps(regs.get<1>().zmm(), ...);
     * @endcode
     */
    template <typename... RegTypes>
    class ScopedRegisterSet
    {
    public:
        explicit ScopedRegisterSet(RegisterTracker &tracker)
            : guards_(tracker.borrow<RegTypes>()...) {}

        template <size_t N>
        auto &get()
        {
            return std::get<N>(guards_);
        }

        template <size_t N>
        const auto &get() const
        {
            return std::get<N>(guards_);
        }

    private:
        std::tuple<RegisterGuard<RegTypes>...> guards_;
    };

    // ============================================================================
    // Optional: ENABLE_REGISTER_GUARDS macro for conditional compilation
    // ============================================================================

#ifdef LLAMINAR_DISABLE_REGISTER_GUARDS

// No-op versions when guards are disabled (for production if overhead matters)
#define BORROW_REG(tracker, RegType) \
    RegType {}
#define BORROW_REG_NAMED(tracker, RegType, name) \
    RegType {}

#else

// Active versions with full tracking
#define BORROW_REG(tracker, RegType) (tracker).borrow<RegType>()
#define BORROW_REG_NAMED(tracker, RegType, name) (tracker).borrow<RegType>()

#endif

    // ============================================================================
    // Conflict Detection Utilities
    // ============================================================================

    /**
     * @brief Static check: Do two register types conflict (same physical register)?
     */
    template <typename Reg1, typename Reg2>
    struct registers_conflict : std::bool_constant<
                                    Reg1::absolute_index == Reg2::absolute_index>
    {
    };

    template <typename Reg1, typename Reg2>
    inline constexpr bool registers_conflict_v = registers_conflict<Reg1, Reg2>::value;

    // Compile-time assertions for known conflicts
    static_assert(registers_conflict_v<Score0, Scratch0>,
                  "Score0 and Scratch0 must conflict (both map to physical reg 20)");
    static_assert(registers_conflict_v<Score1, Scratch1>,
                  "Score1 and Scratch1 must conflict (both map to physical reg 21)");
    static_assert(!registers_conflict_v<Scratch4, Score0>,
                  "Scratch4 (reg 24) should not conflict with Score0 (reg 20)");

} // namespace llaminar2::jit
