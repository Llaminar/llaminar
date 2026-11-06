/**
 * @file PhysicalMemoryBacking.h
 * @brief Typed physical backing of one accelerator allocator.
 *
 * A discrete accelerator owns dedicated memory. An integrated accelerator
 * (for example NVIDIA GB10 or an APU) allocates from the same physical pool as
 * its host. Discovery records this from the driver's integrated-device fact;
 * PhysicalMemoryAuthority uses it to admit the host and accelerator demand
 * jointly instead of crediting the same bytes to two independent allocators.
 */
#pragma once
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace llaminar2
{
    /** @brief Where an allocator's physical bytes come from. */
    enum class PhysicalMemoryBacking : std::uint8_t
    {
        Dedicated = 0,    ///< Separate physical memory (discrete VRAM, or the host itself).
        RankHostPool = 1, ///< Accelerator allocations consume the same rank host pool.
    };

    /** @return Stable diagnostic and wire spelling. */
    [[nodiscard]] constexpr std::string_view toString(PhysicalMemoryBacking backing) noexcept
    {
        switch (backing)
        {
        case PhysicalMemoryBacking::Dedicated:
            return "dedicated";
        case PhysicalMemoryBacking::RankHostPool:
            return "rank_host_pool";
        }
        return "invalid";
    }

    /**
     * @brief Decode a wire value without admitting unknown backings.
     * @throws std::invalid_argument for an unknown value.
     */
    [[nodiscard]] inline PhysicalMemoryBacking physicalMemoryBackingFromWire(std::uint8_t value)
    {
        switch (value)
        {
        case static_cast<std::uint8_t>(PhysicalMemoryBacking::Dedicated):
            return PhysicalMemoryBacking::Dedicated;
        case static_cast<std::uint8_t>(PhysicalMemoryBacking::RankHostPool):
            return PhysicalMemoryBacking::RankHostPool;
        default:
            throw std::invalid_argument("Unknown physical memory backing on the wire");
        }
    }
}
