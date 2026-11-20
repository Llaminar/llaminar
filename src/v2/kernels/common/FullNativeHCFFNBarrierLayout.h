/**
 * @file FullNativeHCFFNBarrierLayout.h
 * @brief Stored-byte layout for distinct canonical HC-prefix and FFN-suffix joins.
 *
 * These types describe instance-owned arena bytes, never allocation or residency
 * authority. The existing PMA/workspace owner admits and retains the complete
 * extent through captured retirement. Prefix and suffix use independent sense
 * slots once per invocation; stream/event retirement orders subsequent reuse.
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include "NativeLayoutBarrierLayout.h"

namespace llaminar2 {
/**
 * @brief Positive compile-time count of arrival/sense pairs.
 * @tparam PhaseCount Number of joins reached exactly once by every admitted CTA.
 * @details This is an extent only. It does not admit a grid, reserve occupancy,
 * reset generations or permit concurrent users of the same retained bytes.
 */
template<unsigned PhaseCount>
struct ResidentGridBarrierLayout {
    static_assert(PhaseCount>0,"Resident grid barriers require a positive phase count");
    static_assert(PhaseCount<=std::numeric_limits<unsigned>::max()/NativeLayoutBarrierLayout::words_per_phase,
        "Resident grid barrier word extent overflows");
    static constexpr unsigned phases=PhaseCount; ///< Distinct once-per-invocation joins.
    static constexpr unsigned words_per_phase=NativeLayoutBarrierLayout::words_per_phase; ///< Arrival and persistent sense.
    static constexpr unsigned words=phases*words_per_phase; ///< Complete stored-word extent.
    static_assert(words<=std::numeric_limits<std::size_t>::max()/sizeof(std::uint32_t),
        "Resident grid barrier byte extent overflows");
    static constexpr std::size_t bytes=std::size_t(words)*sizeof(std::uint32_t); ///< PMA/BOM extent.
};

/**
 * @brief One retained full-region bank, with nonoverlapping prefix/suffix epochs.
 * @details The tuned HC prefix has three joins; the unchanged router/FFN
 * suffix has its existing two joins. The original five-slot prefix reservation
 * remains the stored ABI: the two retired slots are never read or written by
 * execution. Each phase object starts at phase zero
 * using its own offset. Reusing one phase object across both regions would
 * violate this contract. No end-of-kernel reset is required or permitted here.
 */
struct FullNativeHCFFNBarrierLayout {
    using Prefix=ResidentGridBarrierLayout<3>; ///< Live HC publication/retirement joins.
    using PrefixStorage=ResidentGridBarrierLayout<5>; ///< Retained prefix extent, including two unused slots.
    using Suffix=ResidentGridBarrierLayout<NativeLayoutBarrierLayout::phases>; ///< Existing FFN joins.
    static constexpr unsigned prefix_phases=Prefix::phases; ///< HC join count.
    static constexpr unsigned suffix_phases=Suffix::phases; ///< FFN join count.
    static constexpr unsigned phases=prefix_phases+suffix_phases; ///< Complete live join count.
    static constexpr unsigned storage_phases=PrefixStorage::phases+Suffix::phases; ///< Retained slot count, not executable work.
    static constexpr unsigned words_per_phase=Prefix::words_per_phase; ///< Shared arrival/sense ABI.
    static constexpr unsigned prefix_word_offset=0; ///< First HC word within the retained bank.
    static constexpr unsigned suffix_word_offset=PrefixStorage::words; ///< Stable first FFN word; never aliases HC state.
    static constexpr std::size_t prefix_byte_offset=0; ///< HC byte offset for owner bindings.
    static constexpr std::size_t suffix_byte_offset=PrefixStorage::bytes; ///< Stable FFN byte offset for owner bindings.
    static constexpr unsigned prefix_words=Prefix::words; ///< Active HC subextent.
    static constexpr unsigned prefix_storage_words=PrefixStorage::words; ///< Stored HC subextent, including unused words.
    static constexpr unsigned suffix_words=Suffix::words; ///< FFN subextent.
    static constexpr unsigned words=PrefixStorage::words+Suffix::words; ///< Full retained extent.
    static constexpr std::size_t bytes=PrefixStorage::bytes+Suffix::bytes; ///< Full PMA/BOM byte extent.
};
static_assert(FullNativeHCFFNBarrierLayout::prefix_words<=FullNativeHCFFNBarrierLayout::suffix_word_offset,
    "Live prefix joins must remain inside their retained storage before the suffix");
static_assert(FullNativeHCFFNBarrierLayout::Suffix::words==NativeLayoutBarrierLayout::words&&
    FullNativeHCFFNBarrierLayout::Suffix::bytes==NativeLayoutBarrierLayout::bytes,
    "Standalone FFN barrier ABI must remain unchanged");
}
