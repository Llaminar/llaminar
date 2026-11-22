/**
 * @file PhysicalModeClassification.h
 * @brief Typed physical-mode classification (design §9.1-§9.3).
 *
 * Fixed-slot full-context backing is the qualified mode; paged KV is a
 * SEPARATE unqualified physical mode (not an automatic fallback), and state
 * must be classified before reset/restore is implemented. This typed
 * classification makes an unqualified mode fail CLOSED at admission: a
 * derived classification is refused unless it is the qualified mode, with a
 * precise diagnostic — never silently downgraded.
 */

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace llaminar2
{
    /** @brief Physical state modes (design §9.1-§9.2). */
    enum class PhysicalMode : std::uint8_t
    {
        FixedSlotFullContext = 0, ///< Qualified: fixed slot, full-context backing.
        PagedKV,                  ///< Separate unqualified mode (never automatic).
    };

    /** @brief Typed classification result. */
    struct PhysicalModeClassification
    {
        PhysicalMode mode = PhysicalMode::FixedSlotFullContext;

        [[nodiscard]] bool isQualified() const noexcept
        {
            return mode == PhysicalMode::FixedSlotFullContext;
        }

        /** @brief Admission gate: refuse unqualified modes (fail closed). */
        void requireQualified() const
        {
            if (!isQualified())
                throw std::logic_error(
                    "Unqualified physical mode (paged KV) is refused before state mutation; "
                    "request fixed-slot full-context backing or a separately qualified mode");
        }
    };

    /** @brief Classify a requested physical mode (explicit; never inferred). */
    [[nodiscard]] inline PhysicalModeClassification classifyPhysicalMode(PhysicalMode requested) noexcept
    {
        return PhysicalModeClassification{requested};
    }
} // namespace llaminar2
