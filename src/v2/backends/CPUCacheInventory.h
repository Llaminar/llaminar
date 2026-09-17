/**
 * @file CPUCacheInventory.h
 * @brief Immutable Linux cache-domain observation for canonical CPU inventory.
 *
 * CPUID reports one sharing domain, not the sum across a socket's chiplets.
 * This observer deduplicates the highest data/unified cache of the supplied
 * physical cores. It reports hardware facts only, never memory admission or
 * a throughput estimate; consumers reuse HardwareInventory's published value.
 */
#pragma once
#include <cstddef>
#include <filesystem>
#include <span>

namespace llaminar2
{
    /**
     * @brief Sum distinct last-level cache domains covering every supplied core.
     * @param cores Linux logical CPU IDs, one representative per physical core.
     * @param cpu_root Linux CPU sysfs root; injectable for device-free fixtures.
     * @return Aggregate bytes, or zero if any core lacks cache metadata.
     * @throws std::invalid_argument for contradictory/malformed observations.
     * @throws std::overflow_error when the observed sum is unrepresentable.
     */
    size_t observeCPUSharedCacheBytes(std::span<const int> cores,
        const std::filesystem::path &cpu_root = "/sys/devices/system/cpu");
}
