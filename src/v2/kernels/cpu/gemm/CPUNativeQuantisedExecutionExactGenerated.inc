/**
 * @file CPUNativeQuantisedExecutionExactGenerated.inc
 * @brief Generated exact CPU execution overlays; no generic policy is fitted.
 * Full typed keys preserve source/prepared arithmetic and execution ownership.
 * An exact miss leaves the existing certified policy program in authority.
 */
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

namespace llaminar2::cpu::native_quantised::generated
{
struct CPUExecutionExactGeneratedEntry
{
    std::array<std::uint32_t, 18> key;
    CPUExecutionExactChoice choice;
};
inline constexpr std::array<CPUExecutionExactGeneratedEntry, 2> kCPUExecutionExactEntries{{
    {{1u, 1u, 5u, 1u, 0u, 1u, 1u, 1u, 28u, 3u, 320u, 320u, 2560u, 16u, 1u, 1u, 1u, 1u}, CPUExecutionExactChoice::WideRows},
    {{1u, 1u, 7u, 0u, 1u, 1u, 1u, 1u, 28u, 3u, 2560u, 2560u, 320u, 10u, 1u, 1u, 1u, 1u}, CPUExecutionExactChoice::Pairwise},
}};

/** Resolve one immutable execution key without allocation or policy mutation. */
inline bool selectCPUExecutionExactPolicy(
    const CPUExecutionExactKey &key, CPUExecutionExactChoice &choice) noexcept
{
    const auto scalars = key.scalars();
    std::size_t first = 0u;
    std::size_t last = kCPUExecutionExactEntries.size();
    while (first < last)
    {
        const std::size_t middle = first + (last - first) / 2u;
        if (kCPUExecutionExactEntries[middle].key < scalars)
            first = middle + 1u;
        else
            last = middle;
    }
    if (first == kCPUExecutionExactEntries.size() || kCPUExecutionExactEntries[first].key != scalars)
        return false;
    choice = kCPUExecutionExactEntries[first].choice;
    return true;
}
}
