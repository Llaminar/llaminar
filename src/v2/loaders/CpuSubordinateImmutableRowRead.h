/**
 * @file CpuSubordinateImmutableRowRead.h
 * @brief Exact live metadata for a GPU-authorized subordinate native source read.
 *
 * These records name operations, never GPU locations or mutable GPU directory
 * state. The existing host processor alone acquires the authenticated request;
 * no public integer constructor can manufacture its cache read capability.
 */
#pragma once
#include <cstdint>
#include <type_traits>
namespace llaminar2 {
/** @brief Source service operation; exclusive promotion must detach new RAM reads. */
enum class ImmutableRowReadIntent : std::uint64_t { Read = 0, ReadAndInvalidate = 1 };
/** @brief One live compact-ID operation with an opaque GPU reservation generation. */
struct ImmutableRowSubordinateIntent final {
    ImmutableRowReadIntent intent = ImmutableRowReadIntent::Read;
    std::uint64_t destination_grant = 0; ///< Zero for Read, nonzero for exclusive promotion; CPU never decodes it.
};
/** @brief Published only after native fill and generation-matched RAM invalidation. */
struct ImmutableRowInvalidationAcknowledgement final {
    std::uint64_t destination_grant = 0; ///< Exact authenticated promotion token, zero for ordinary source reads.
};
static_assert(sizeof(ImmutableRowSubordinateIntent)==16 && sizeof(ImmutableRowInvalidationAcknowledgement)==8);
static_assert(std::is_trivially_copyable_v<ImmutableRowSubordinateIntent> &&
    std::is_trivially_copyable_v<ImmutableRowInvalidationAcknowledgement>);
} // namespace llaminar2
