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
enum class HostImmutableRowMode { WholeRAM, WholeMemoryMapped, AdaptiveRAMSSD, AdaptiveFullRAM };
/**
 * @brief Classify actual adaptive owner implementations without conflating requested storage modes.
 * @param mode Original concrete provider mode.
 * @return True for adaptive SSD/full-RAM owners, false for ordinary whole-source owners.
 * @throws std::invalid_argument For an unrecognized actual mode.
 */
inline bool hostImmutableRowModeIsAdaptive(HostImmutableRowMode mode){
    switch(mode){
    case HostImmutableRowMode::WholeRAM:case HostImmutableRowMode::WholeMemoryMapped:return false;
    case HostImmutableRowMode::AdaptiveRAMSSD:case HostImmutableRowMode::AdaptiveFullRAM:return true;
    }
    throw std::invalid_argument("Unknown actual host native row implementation");
}
/** @return Stable production evidence label, rejecting an unknown actual mode. */
inline std::string_view hostImmutableRowModeName(HostImmutableRowMode mode)
{
    switch(mode) {
    case HostImmutableRowMode::WholeRAM:return "ram";
    case HostImmutableRowMode::WholeMemoryMapped:return "mmap";
    case HostImmutableRowMode::AdaptiveRAMSSD:return "host_adaptive_ram_ssd";
    case HostImmutableRowMode::AdaptiveFullRAM:return "gpu_subordinate_full_ram";
    }
    throw std::invalid_argument("Unknown actual host native row implementation");
}
} // namespace llaminar2
