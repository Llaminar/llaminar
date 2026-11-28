/**
 * @file ROCmLiveArrivalGrid.h
 * @brief Admit the hardware-bounded native HIP unpack node.
 *
 * HIP graphs do not expose device-updatable geometry or conditional launch
 * nodes. Its retained work node covers at most one resident hardware wave, capped
 * by the most distinct arrivals that can be represented by the plan and directory.
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>

namespace llaminar2
{
    /** @brief One resident grid bounded by native hardware and representable arrivals. */
    class ROCmLiveArrivalGrid final
    {
    public:
        static constexpr unsigned threads = 256; ///< Exact compiled block width used in the occupancy query.

        struct Hardware
        {
            int active_cus; ///< Device inventory, not compile-time architecture defaults.
            int blocks_per_cu; ///< Occupancy of this exact unpack kernel.
        };

        struct ArrivalCapacity
        {
            std::uint32_t plan_entries; ///< Commands in one admitted wave.
            std::uint32_t destination_slots; ///< Distinct destinations in the local directory.
        };

        /** @return The resident grid capped by the distinct-arrival bound; no work node for zero capacity. */
        [[nodiscard]] static constexpr std::optional<ROCmLiveArrivalGrid> admit(
            Hardware hardware, ArrivalCapacity capacity) noexcept
        {
            if (hardware.active_cus <= 0 || hardware.blocks_per_cu <= 0)
                return std::nullopt;
            const auto resident = std::uint64_t(hardware.active_cus) * std::uint64_t(hardware.blocks_per_cu);
            const auto blocks = std::min({resident, std::uint64_t(capacity.plan_entries),
                std::uint64_t(capacity.destination_slots)});
            if (blocks > std::uint64_t(std::numeric_limits<std::int32_t>::max()))
                return std::nullopt;
            return ROCmLiveArrivalGrid(static_cast<std::uint32_t>(blocks));
        }

        /** @return Native retained node width, or zero when no arrival is representable. */
        [[nodiscard]] constexpr std::uint32_t blocks() const noexcept { return blocks_; }

    private:
        explicit constexpr ROCmLiveArrivalGrid(std::uint32_t blocks) : blocks_(blocks) {}
        std::uint32_t blocks_;
    };
}
