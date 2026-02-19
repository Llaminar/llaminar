/**
 * @file GlobalDeviceAddress.cpp
 * @brief Implementation of GlobalDeviceAddress
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "GlobalDeviceAddress.h"
#include "utils/Logger.h"
#include <sstream>
#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace llaminar2
{

    // =========================================================================
    // Factory Methods
    // =========================================================================

    GlobalDeviceAddress GlobalDeviceAddress::cpu(int numa, const std::string &hostname)
    {
        GlobalDeviceAddress addr;
        addr.hostname = hostname;
        addr.numa_node = numa;
        addr.device_type = DeviceType::CPU;
        addr.device_ordinal = 0;
        return addr;
    }

    GlobalDeviceAddress GlobalDeviceAddress::cuda(int ordinal, int numa, const std::string &hostname)
    {
        GlobalDeviceAddress addr;
        addr.hostname = hostname;
        addr.numa_node = numa;
        addr.device_type = DeviceType::CUDA;
        addr.device_ordinal = ordinal;
        return addr;
    }

    GlobalDeviceAddress GlobalDeviceAddress::rocm(int ordinal, int numa, const std::string &hostname)
    {
        GlobalDeviceAddress addr;
        addr.hostname = hostname;
        addr.numa_node = numa;
        addr.device_type = DeviceType::ROCm;
        addr.device_ordinal = ordinal;
        return addr;
    }

    // =========================================================================
    // Parsing Helpers
    // =========================================================================

    DeviceType GlobalDeviceAddress::parseDeviceType(const std::string &type_str)
    {
        // Convert to lowercase for case-insensitive comparison
        std::string lower = type_str;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char c)
                       { return std::tolower(c); });

        if (lower == "cpu")
        {
            return DeviceType::CPU;
        }
        else if (lower == "cuda")
        {
            return DeviceType::CUDA;
        }
        else if (lower == "rocm" || lower == "hip")
        {
            return DeviceType::ROCm;
        }
        else if (lower == "vulkan")
        {
            return DeviceType::Vulkan;
        }
        else if (lower == "metal")
        {
            return DeviceType::Metal;
        }

        throw std::invalid_argument("Unknown device type: '" + type_str +
                                    "'. Valid types: cpu, cuda, rocm, hip, vulkan, metal");
    }

    std::string GlobalDeviceAddress::deviceTypeToString(DeviceType type)
    {
        switch (type)
        {
        case DeviceType::CPU:
            return "cpu";
        case DeviceType::CUDA:
            return "cuda";
        case DeviceType::ROCm:
            return "rocm";
        case DeviceType::Vulkan:
            return "vulkan";
        case DeviceType::Metal:
            return "metal";
        default:
            return "unknown";
        }
    }

    // =========================================================================
    // Parsing
    // =========================================================================

    GlobalDeviceAddress GlobalDeviceAddress::parse(const std::string &spec, int current_numa)
    {
        auto result = tryParse(spec, current_numa);
        if (!result)
        {
            throw std::invalid_argument("Failed to parse device address: '" + spec +
                                        "'. Expected format: [hostname:]numa:type:ordinal or type:ordinal");
        }
        return *result;
    }

    std::optional<GlobalDeviceAddress> GlobalDeviceAddress::tryParse(const std::string &spec, int current_numa)
    {
        if (spec.empty())
        {
            return std::nullopt;
        }

        // Split by ':'
        std::vector<std::string> parts;
        std::stringstream ss(spec);
        std::string part;
        while (std::getline(ss, part, ':'))
        {
            parts.push_back(part);
        }

        GlobalDeviceAddress addr;
        addr.hostname = "localhost";
        addr.numa_node = current_numa;

        try
        {
            if (parts.size() == 2)
            {
                // Short form: "type:ordinal"
                // e.g., "cuda:0" -> localhost:<current_numa>:cuda:0
                addr.device_type = parseDeviceType(parts[0]);
                addr.device_ordinal = std::stoi(parts[1]);
            }
            else if (parts.size() == 3)
            {
                // Medium form: "numa:type:ordinal"
                // e.g., "0:cuda:0" -> localhost:0:cuda:0
                addr.numa_node = std::stoi(parts[0]);
                addr.device_type = parseDeviceType(parts[1]);
                addr.device_ordinal = std::stoi(parts[2]);
            }
            else if (parts.size() == 4)
            {
                // Full form: "hostname:numa:type:ordinal"
                // e.g., "node1:0:cuda:0"
                addr.hostname = parts[0];
                addr.numa_node = std::stoi(parts[1]);
                addr.device_type = parseDeviceType(parts[2]);
                addr.device_ordinal = std::stoi(parts[3]);
            }
            else
            {
                return std::nullopt;
            }

            // Validate ordinal is non-negative
            if (addr.device_ordinal < 0)
            {
                return std::nullopt;
            }

            // Validate NUMA node: reject truly invalid negatives
            // (NUMA_NODE_UNKNOWN/-1 is allowed as sentinel value)
            if (addr.numa_node < -1)
            {
                return std::nullopt;
            }

            return addr;
        }
        catch (const std::exception &)
        {
            // stoi or parseDeviceType failed
            return std::nullopt;
        }
    }

    // =========================================================================
    // Serialization
    // =========================================================================

    std::string GlobalDeviceAddress::toString() const
    {
        std::ostringstream oss;
        oss << hostname << ":" << numa_node << ":"
            << deviceTypeToString(device_type) << ":" << device_ordinal;
        return oss.str();
    }

    std::string GlobalDeviceAddress::toShortString() const
    {
        std::ostringstream oss;

        // Omit localhost and numa_node 0 for brevity
        if (hostname == "localhost" && (numa_node == 0 || numa_node < 0))
        {
            oss << deviceTypeToString(device_type) << ":" << device_ordinal;
        }
        else if (hostname == "localhost")
        {
            oss << numa_node << ":" << deviceTypeToString(device_type) << ":" << device_ordinal;
        }
        else
        {
            oss << hostname << ":" << numa_node << ":"
                << deviceTypeToString(device_type) << ":" << device_ordinal;
        }

        return oss.str();
    }

    // =========================================================================
    // Conversion to/from DeviceId
    // =========================================================================

    DeviceId GlobalDeviceAddress::toLocalDeviceId() const
    {
        switch (device_type)
        {
        case DeviceType::CPU:
            return DeviceId::cpu();
        case DeviceType::CUDA:
            return DeviceId::cuda(device_ordinal);
        case DeviceType::ROCm:
            return DeviceId::rocm(device_ordinal);
        default:
            // Vulkan/Metal not yet supported in DeviceId
            LOG_WARN("Unsupported device type for DeviceId conversion: "
                     << deviceTypeToString(device_type));
            return DeviceId::cpu();
        }
    }

    GlobalDeviceAddress GlobalDeviceAddress::fromLocalDeviceId(
        const DeviceId &local_id,
        const std::string &hostname,
        int numa_node)
    {
        GlobalDeviceAddress addr;
        addr.hostname = hostname;
        addr.numa_node = numa_node;
        addr.device_type = local_id.type;
        addr.device_ordinal = local_id.ordinal;
        return addr;
    }

    // =========================================================================
    // Predicates
    // =========================================================================

    bool GlobalDeviceAddress::isLocal() const
    {
        return hostname.empty() || hostname == "localhost";
    }

    bool GlobalDeviceAddress::sameNuma(const GlobalDeviceAddress &other) const
    {
        // If either side has unknown NUMA, we cannot confirm same-NUMA
        if (numa_node < 0 || other.numa_node < 0)
            return false;
        return hostname == other.hostname && numa_node == other.numa_node;
    }

    bool GlobalDeviceAddress::sameHost(const GlobalDeviceAddress &other) const
    {
        return hostname == other.hostname;
    }

    // =========================================================================
    // Comparison Operators
    // =========================================================================

    bool GlobalDeviceAddress::operator==(const GlobalDeviceAddress &o) const
    {
        return hostname == o.hostname &&
               numa_node == o.numa_node &&
               device_type == o.device_type &&
               device_ordinal == o.device_ordinal;
    }

    bool GlobalDeviceAddress::operator!=(const GlobalDeviceAddress &o) const
    {
        return !(*this == o);
    }

    bool GlobalDeviceAddress::operator<(const GlobalDeviceAddress &o) const
    {
        // Lexicographic comparison for consistent ordering
        if (hostname != o.hostname)
            return hostname < o.hostname;
        if (numa_node != o.numa_node)
            return numa_node < o.numa_node;
        if (device_type != o.device_type)
            return static_cast<int>(device_type) < static_cast<int>(o.device_type);
        return device_ordinal < o.device_ordinal;
    }

    // =========================================================================
    // Stream Output
    // =========================================================================

    std::ostream &operator<<(std::ostream &os, const GlobalDeviceAddress &addr)
    {
        os << addr.toString();
        return os;
    }

} // namespace llaminar2
