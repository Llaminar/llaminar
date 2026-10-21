/**
 * @file ParityHardwareRequirements.h
 * @brief Metadata-only rank-local accelerator requirements from exact topology addresses.
 *
 * A collective tag is not a device inventory. Sparse GPU ordinals require the
 * corresponding visible ordinal range; CPU-only participants require neither
 * GPU runtime. This value is a frozen test prerequisite, not an allocation plan.
 */
#pragma once
#include "backends/GlobalDeviceAddress.h"
#include <algorithm>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>

namespace llaminar2::test::parity
{
/** @brief Exact minimum visible GPU ordinal ranges for one local topology. */
struct ParityHardwareRequirements
{
    int cuda_devices = 0;
    int rocm_devices = 0;

    /**
     * @brief Derive visible ordinal ranges without probing either backend.
     * @param addresses Exact rank-local participant addresses.
     * @return Minimum CUDA/ROCm device counts covering all requested ordinals.
     * @throws std::invalid_argument For an empty list, unknown device type or negative GPU ordinal.
     * @throws std::overflow_error When ordinal+1 cannot be represented.
     */
    static ParityHardwareRequirements fromParticipants(std::span<const GlobalDeviceAddress> addresses)
    {
        if (addresses.empty()) throw std::invalid_argument("Parity hardware requires actual participants");
        ParityHardwareRequirements result;
        for (const auto &address : addresses)
        {
            if (address.device_type == DeviceType::CPU) continue;
            if (address.device_type != DeviceType::CUDA && address.device_type != DeviceType::ROCm)
                throw std::invalid_argument("Parity hardware participant has unknown device type");
            if (address.device_ordinal < 0)
                throw std::invalid_argument("Parity hardware participant has negative GPU ordinal");
            if (address.device_ordinal == std::numeric_limits<int>::max())
                throw std::overflow_error("Parity hardware visible ordinal count overflows int");
            auto &count = address.device_type == DeviceType::CUDA ? result.cuda_devices : result.rocm_devices;
            count = std::max(count, address.device_ordinal + 1);
        }
        return result;
    }
};

/**
 * @brief Validate observed inventory against the already-derived exact requirements.
 * @param required Frozen topology requirements.
 * @param available Counts observed only for the backends required by this topology.
 * @return Precise missing-inventory reason or no failure.
 * @throws std::invalid_argument For negative required or observed counts.
 */
inline std::optional<std::string> parityHardwareInventoryFailure(
    ParityHardwareRequirements required, ParityHardwareRequirements available)
{
    if (required.cuda_devices < 0 || required.rocm_devices < 0 ||
        available.cuda_devices < 0 || available.rocm_devices < 0)
        throw std::invalid_argument("Parity hardware device counts must be nonnegative");
    if (required.cuda_devices > available.cuda_devices)
        return "Need " + std::to_string(required.cuda_devices) + " CUDA devices, found " + std::to_string(available.cuda_devices);
    if (required.rocm_devices > available.rocm_devices)
        return "Need " + std::to_string(required.rocm_devices) + " ROCm devices, found " + std::to_string(available.rocm_devices);
    return std::nullopt;
}
} // namespace llaminar2::test::parity
