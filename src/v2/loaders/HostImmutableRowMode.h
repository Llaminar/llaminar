/**
 * @file HostImmutableRowMode.h
 * @brief Actual immutable host-provider implementation, never an unresolved tier preference.
 * The adaptive alternative cannot be reported as whole mmap merely because its
 * private source service maps files. Observations retain the actual mode.
 */
#pragma once
#include <stdexcept>
#include <string_view>
namespace llaminar2 {
/** @brief Closed host serving alternatives. */
enum class HostImmutableRowMode { WholeRAM, WholeMemoryMapped, AdaptiveRAMSSD };
/** @return Stable production evidence label, rejecting an unknown actual mode. */
inline std::string_view hostImmutableRowModeName(HostImmutableRowMode mode)
{
    switch(mode) {
    case HostImmutableRowMode::WholeRAM:return "ram";
    case HostImmutableRowMode::WholeMemoryMapped:return "mmap";
    case HostImmutableRowMode::AdaptiveRAMSSD:return "host_adaptive_ram_ssd";
    }
    throw std::invalid_argument("Unknown actual host native row implementation");
}
} // namespace llaminar2
