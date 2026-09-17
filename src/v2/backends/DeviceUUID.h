/**
 * @file DeviceUUID.h
 * @brief Backend-independent encoding of a driver's immutable GPU identity.
 *
 * A backend ordinal identifies an accessible device only inside one process.
 * Cluster discovery must preserve the driver's UUID to recognize shared views
 * of the same allocation resource, including partitioned GPUs sharing a BDF.
 * This pure formatter performs no discovery and owns no capacity accounting.
 */
#pragma once

#include <algorithm>
#include <span>
#include <stdexcept>
#include <string>

namespace llaminar2
{
    /**
     * @brief Encode an observed 16-byte CUDA/HIP UUID without sign extension.
     * @param bytes Exact driver UUID bytes, never a rank or ordinal substitute.
     * @return Stable lowercase hexadecimal identity used by the MPI inventory.
     * @throws std::invalid_argument if the driver supplied no usable identity.
     */
    inline std::string formatDeviceUUID(std::span<const char, 16> bytes)
    {
        if (std::all_of(bytes.begin(), bytes.end(), [](char byte) { return byte == 0; }))
            throw std::invalid_argument("GPU discovery returned an empty device UUID");
        constexpr char hex[] = "0123456789abcdef";
        std::string identity(32, '0');
        for (std::size_t i = 0; i < bytes.size(); ++i)
        {
            const auto byte = static_cast<unsigned char>(bytes[i]);
            identity[2 * i] = hex[byte >> 4];
            identity[2 * i + 1] = hex[byte & 15];
        }
        return identity;
    }
}
