/**
 * @file DevicePCIAddress.h
 * @brief Checked physical endpoint identity shared by driver discovery and OS evidence.
 *
 * Driver UUIDs remain the allocation identity, including GPU partitions. A PCI
 * address is the additional join to Linux DRM counters; neither enumeration
 * order nor a PCIe bridge's bottleneck address identifies that endpoint. This
 * value normalizes native driver spelling without discovering any hardware.
 */
#pragma once

#include <array>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace llaminar2
{
    /** @brief An authenticated complete PCI domain:bus:device.function address. */
    class DevicePCIAddress
    {
    public:
        /**
         * @brief Validate a native driver address and normalize its hexadecimal spelling.
         * @param value Four- through eight-digit domain followed by the complete BDF.
         * @return Lowercase Linux spelling, with at least four domain digits.
         * @throws std::invalid_argument For missing fields, padding, or out-of-range components.
         */
        static DevicePCIAddress parse(std::string_view value)
        {
            const auto colon = value.find(':');
            if ((colon < 4 || colon > 8) || value.size() != colon + 8 ||
                value[colon + 3] != ':' || value[colon + 6] != '.')
                throw std::invalid_argument("GPU discovery requires a complete native PCI address");
            const auto component = [](std::string_view text, uint32_t maximum) {
                uint32_t result = 0;
                const auto converted = std::from_chars(text.data(), text.data() + text.size(), result, 16);
                if (converted.ec != std::errc{} || converted.ptr != text.data() + text.size() || result > maximum)
                    throw std::invalid_argument("GPU discovery has an invalid PCI address component");
                return result;
            };
            const auto domain = component(value.substr(0, colon), UINT32_MAX);
            const auto bus = component(value.substr(colon + 1, 2), 255);
            const auto device = component(value.substr(colon + 4, 2), 31);
            const auto function = component(value.substr(colon + 7, 1), 7);
            std::array<char, 17> result{};
            std::snprintf(result.data(), result.size(), "%04x:%02x:%02x.%x", domain, bus, device, function);
            return DevicePCIAddress(result.data());
        }

        /** @return Stable complete physical endpoint spelling; performs no OS or driver calls. */
        const std::string &toString() const noexcept { return value_; }

    private:
        /** @brief Retain only spelling produced by the validating factory. @param value Validated BDF. */
        explicit DevicePCIAddress(std::string value) : value_(std::move(value)) {}
        std::string value_; ///< Immutable physical endpoint; never an allocation or capacity ledger.
    };
}
