/**
 * @file NativeLayoutBarrierLayout.h
 * @brief One canonical stored-byte contract for the two-join router/FFN closure.
 *
 * Planner, backend and replay owner share this exact slot extent. Each join
 * owns an arrival and sense word, retained until captured graph retirement.
 */
#pragma once
#include <cstddef>
#include <cstdint>
namespace llaminar2 {
/** @brief Immutable per-instance barrier storage, never a separate allocation ledger. */
struct NativeLayoutBarrierLayout {
    static constexpr unsigned phases=2; ///< Router/shared ready, then routed gate/up ready.
    static constexpr unsigned words_per_phase=2; ///< Arrival and persistent sense.
    static constexpr unsigned words=phases*words_per_phase;
    static constexpr std::size_t bytes=words*sizeof(std::uint32_t);
};
}
