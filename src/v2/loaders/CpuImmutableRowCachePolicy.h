/**
 * @file CpuImmutableRowCachePolicy.h
 * @brief Frozen setup values for experimental host native-row cache admission/eviction.
 *
 * Frequency controls candidate admission; probation/protected logical recency
 * controls eviction. This value owns neither rows nor ordering. The actual cache
 * mints its policy order; no GPU channel sequence is a shared-cache authority.
 */
#pragma once
#include <cstdint>
namespace llaminar2 {
/** @brief Explicit experimental constants, never reread from the environment during execution. */
struct CpuImmutableRowCachePolicy final {
    std::uint64_t aging_accesses = 65536; ///< Logical accesses per lazy frequency epoch.
    std::uint32_t protected_numerator = 3, protected_denominator = 4; ///< Bounded protected fraction.
    bool operator==(const CpuImmutableRowCachePolicy &) const = default;
};
} // namespace llaminar2
