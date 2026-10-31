/**
 * @file CUDAHardwareMetadata.h
 * @brief Process-lifetime physical CUDA facts, separate from native resource generations.
 *
 * Visible ordinals index an admitted immutable vector and never alias another
 * device through a masked index or default ordinal. UUIDs identify allocations;
 * distinct visible CUDA instances may legitimately share a parent PCI endpoint.
 * Runtime resets retire handles, loaded-function state and storage, not facts.
 */
#pragma once
#include "backends/ImmutableHardwarePublication.h"
#include "backends/DevicePCIAddress.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2
{
    /** @brief Exact physical limits; precision support and implemented programs remain separate. */
    struct CUDADeviceHardware
    {
        int ordinal = -1;
        std::string name, pci_address;
        std::array<std::uint8_t, 16> uuid{};
        int pci_domain = 0, pci_bus = 0, pci_device = 0;
        int sm_major = 0, sm_minor = 0, sm_count = 0, native_warp_width = 0;
        int maximum_threads_per_block = 0, maximum_threads_per_sm = 0;
        int maximum_blocks_per_sm = 0, registers_per_block = 0, registers_per_sm = 0;
        std::size_t shared_memory_per_block = 0, shared_memory_optin_per_block = 0;
        std::size_t shared_memory_per_sm = 0, total_memory = 0, last_level_cache = 0;
    };

    /** @brief Complete checked visibility with no fixed card-count ceiling or PCI-only alias. */
    class CUDAHardwareInventory final
    {
    public:
        /**
         * @brief Admit an ordered complete discovery result before any reader can see it.
         * @param devices Exact visible records; genuinely empty visibility is valid.
         * @throws std::invalid_argument On absent, conflicting or malformed physical facts.
         */
        explicit CUDAHardwareInventory(std::vector<CUDADeviceHardware> devices)
            : devices_(std::move(devices))
        {
            if (devices_.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw std::invalid_argument("CUDA physical inventory exceeds native ordinal representation");
            std::set<std::array<std::uint8_t, 16>> uuids;
            for (std::size_t index = 0; index < devices_.size(); ++index)
            {
                const auto &device = devices_[index];
                if (device.ordinal != static_cast<int>(index) || device.name.empty() ||
                    device.pci_address.empty() || device.pci_domain < 0 || device.pci_bus < 0 ||
                    device.pci_device < 0 || device.sm_major <= 0 || device.sm_minor < 0 ||
                    device.sm_count <= 0 || device.native_warp_width <= 0 ||
                    device.maximum_threads_per_block <= 0 || device.maximum_threads_per_sm <= 0 ||
                    device.maximum_blocks_per_sm <= 0 || device.registers_per_block <= 0 ||
                    device.registers_per_sm <= 0 || device.shared_memory_per_block == 0 ||
                    device.shared_memory_optin_per_block < device.shared_memory_per_block ||
                    device.shared_memory_per_sm < device.shared_memory_per_block || device.total_memory == 0)
                    throw std::invalid_argument("Incomplete CUDA physical metadata at ordinal " + std::to_string(index));
                const auto pci = DevicePCIAddress::parse(device.pci_address).toString();
                const auto colon = pci.find(':');
                const auto component = [&pci](std::size_t offset, std::size_t length) {
                    std::uint32_t value = 0;
                    const auto *begin = pci.data() + offset;
                    std::from_chars(begin, begin + length, value, 16);
                    return value;
                };
                // The validated full BDF owns function identity. Native property
                // components must agree, but shared parent BDFs are legal (MIG).
                if (component(0, colon) != static_cast<std::uint32_t>(device.pci_domain) ||
                    component(colon + 1, 2) != static_cast<std::uint32_t>(device.pci_bus) ||
                    component(colon + 4, 2) != static_cast<std::uint32_t>(device.pci_device))
                    throw std::invalid_argument("Conflicting CUDA PCI components at ordinal " + std::to_string(index));
                if (std::all_of(device.uuid.begin(), device.uuid.end(), [](auto byte) { return byte == 0; }))
                    throw std::invalid_argument("Absent CUDA physical UUID at ordinal " + std::to_string(index));
                if (!uuids.insert(device.uuid).second)
                    throw std::invalid_argument("Duplicate CUDA physical UUID at ordinal " + std::to_string(index));
            }
        }
        /** @return Exact frozen visible count; no backend or driver is queried. */
        [[nodiscard]] std::size_t size() const noexcept { return devices_.size(); }
        /**
         * @brief Borrow one exact visible record, independently of runtime reset.
         * @param ordinal Backend-local ordinal in this process's frozen visibility.
         * @return Process-lifetime physical facts.
         * @throws std::out_of_range For negative or absent ordinals, without substitution.
         */
        [[nodiscard]] const CUDADeviceHardware &at(int ordinal) const
        {
            if (ordinal < 0 || static_cast<std::size_t>(ordinal) >= devices_.size())
                throw std::out_of_range("Invalid CUDA physical device ordinal " + std::to_string(ordinal));
            return devices_[static_cast<std::size_t>(ordinal)];
        }
    private:
        std::vector<CUDADeviceHardware> devices_;
    };

    /** @brief Explicit once-admission authority with an injectable device-free source. */
    class CUDAHardwareAuthority final
    {
    public:
        /**
         * @brief Publish one source result, retaining any original failure permanently.
         * @param source Native startup reader or recorded device-free reader.
         * @throws Discovery or complete-inventory validation failures.
         */
        void initialize(const std::function<std::vector<CUDADeviceHardware>()> &source)
        {
            publication_.initialize(source);
        }
        /**
         * @brief Acquire already admitted physical facts without native work.
         * @return Stable immutable inventory.
         * @throws std::logic_error Before explicit successful startup admission.
         */
        [[nodiscard]] const CUDAHardwareInventory &snapshot() const
        {
            return publication_.snapshot("CUDA physical metadata was not admitted before execution");
        }
    private:
        ImmutableHardwarePublication<CUDAHardwareInventory> publication_;
    };

    /** @brief Admit CUDA physical facts only at an explicitly enabled startup boundary. */
    void initializeCUDAHardwareMetadata();
    /** @return Already admitted inventory; no lazy driver initialization or generation refresh. */
    const CUDAHardwareInventory &cudaHardwareMetadata();
}
