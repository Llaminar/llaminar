/**
 * @file MoEOverlayEpochTestStorage.h
 * @brief Aligned model-free fixtures for the production variable epoch record.
 *
 * Tail storage follows the production byte recipe exactly. Participant IDs
 * remain immutable metadata and arrivals remain the sole mutable member lanes.
 */
#pragma once
#include "execution/moe/MoEOverlayDeviceControllerABI.h"
#include <array>
#include <cstddef>
#include <cstdint>

namespace llaminar2::test
{
    /** @brief Aligned storage whose tail layout matches the production extent recipe.
     * @tparam Members Actual fixture membership, independent of global IDs. */
    template <std::size_t Members>
    struct alignas(64) MoEOverlayEpochTestStorage
    {
        static_assert(Members > 0u && Members <= UINT32_MAX);
        MoEOverlayDeviceControllerInferenceEpochRecord record;
        std::array<std::uint32_t, Members> ids{};
        alignas(std::uint64_t) std::array<std::uint64_t, Members> arrivals{};

        /** @brief Bind the same exact admitted extent consumed by CPU/CUDA/HIP.
         * @param members Ascending original global IDs; no bit encoding.
         * @param publisher Original sole epoch publisher.
         * @param fingerprint Immutable topology identity. */
        void initialize(std::array<std::uint32_t, Members> members,
            std::uint32_t publisher, std::uint64_t fingerprint = 0xabc123u)
        {
            record = {};
            record.bindMemberExtent(static_cast<std::uint32_t>(Members));
            record.publisher_participant_id = publisher;
            record.topology_fingerprint = fingerprint;
            ids = members;
            arrivals.fill(0u);
        }

        /** @return Exact admitted bytes; trailing C++ object alignment is not work/storage extent. */
        static constexpr std::uint64_t bytes() noexcept
        {
            return MoEOverlayDeviceControllerInferenceEpochRecord::bytesForMembers(static_cast<std::uint32_t>(Members));
        }
    };
}
