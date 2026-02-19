/**
 * @file GlobalDeviceAddress.h
 * @brief Fully-qualified global device address for heterogeneous orchestration
 *
 * Provides a complete addressing scheme that includes:
 * - Hostname (for multi-node deployments)
 * - NUMA node (for memory affinity)
 * - Device type (CPU, CUDA, ROCm)
 * - Device ordinal (0-indexed)
 *
 * Format: hostname:numa_node:device_type:device_ordinal
 * Examples:
 *   - "localhost:0:cuda:0" - First CUDA GPU on NUMA node 0
 *   - "node1:1:rocm:0" - First ROCm GPU on node1, NUMA node 1
 *   - "localhost:0:cpu:0" - CPU execution on NUMA node 0
 *
 * Shorthand forms supported by parse():
 *   - "cuda:0" -> localhost:<current_numa>:cuda:0
 *   - "0:cuda:0" -> localhost:0:cuda:0
 *   - "node1:0:cuda:0" -> full form
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "DeviceId.h"
#include "DeviceType.h"
#include <string>
#include <optional>
#include <vector>
#include <ostream>
#include <functional>

namespace llaminar2
{

    /**
     * @brief Fully-qualified global device address
     *
     * Unlike DeviceId (which is rank-local), GlobalDeviceAddress provides
     * complete device identification across a distributed system:
     * - Multi-node support via hostname
     * - NUMA-aware placement
     * - Type-safe device identification
     *
     * Thread safety: Immutable value type, safe to copy and share.
     */
    /// Sentinel value indicating NUMA node has not been resolved from hardware.
    static constexpr int NUMA_NODE_UNKNOWN = -1;

    struct GlobalDeviceAddress
    {
        std::string hostname = "localhost";
        int numa_node = NUMA_NODE_UNKNOWN;
        DeviceType device_type = DeviceType::CPU;
        int device_ordinal = 0;

        /// @brief Check whether numa_node has been explicitly set
        bool hasValidNuma() const { return numa_node >= 0; }

        // =========================================================================
        // Factory Methods
        // =========================================================================

        /**
         * @brief Create a CPU device address
         * @param numa NUMA node (NUMA_NODE_UNKNOWN if not known)
         * @param hostname Hostname (default "localhost")
         * @return GlobalDeviceAddress for CPU
         */
        static GlobalDeviceAddress cpu(int numa = NUMA_NODE_UNKNOWN, const std::string &hostname = "localhost");

        /**
         * @brief Create a CUDA device address
         * @param ordinal GPU ordinal (0-indexed)
         * @param numa NUMA node (NUMA_NODE_UNKNOWN if not known)
         * @param hostname Hostname (default "localhost")
         * @return GlobalDeviceAddress for CUDA GPU
         */
        static GlobalDeviceAddress cuda(int ordinal, int numa = NUMA_NODE_UNKNOWN, const std::string &hostname = "localhost");

        /**
         * @brief Create a ROCm device address
         * @param ordinal GPU ordinal (0-indexed)
         * @param numa NUMA node (NUMA_NODE_UNKNOWN if not known)
         * @param hostname Hostname (default "localhost")
         * @return GlobalDeviceAddress for ROCm GPU
         */
        static GlobalDeviceAddress rocm(int ordinal, int numa = NUMA_NODE_UNKNOWN, const std::string &hostname = "localhost");

        // =========================================================================
        // Parsing
        // =========================================================================

        /**
         * @brief Parse address from string (throws on error)
         *
         * Supports multiple formats:
         *   - Full: "hostname:numa:type:ordinal" (e.g., "node1:0:cuda:0")
         *   - No host: "numa:type:ordinal" (e.g., "0:cuda:0" -> localhost:0:cuda:0)
         *   - Short: "type:ordinal" (e.g., "cuda:0" -> localhost:<current_numa>:cuda:0)
         *
         * @param spec String specification
         * @param current_numa NUMA node to use for short form (NUMA_NODE_UNKNOWN if not known)
         * @return Parsed GlobalDeviceAddress
         * @throws std::invalid_argument if parsing fails
         */
        static GlobalDeviceAddress parse(const std::string &spec, int current_numa = NUMA_NODE_UNKNOWN);

        /**
         * @brief Try to parse address from string (returns nullopt on error)
         * @param spec String specification
         * @param current_numa NUMA node to use for short form (NUMA_NODE_UNKNOWN if not known)
         * @return Parsed GlobalDeviceAddress or nullopt if invalid
         */
        static std::optional<GlobalDeviceAddress> tryParse(const std::string &spec, int current_numa = NUMA_NODE_UNKNOWN);

        // =========================================================================
        // Serialization
        // =========================================================================

        /**
         * @brief Convert to full string representation
         * @return String in format "hostname:numa:type:ordinal"
         */
        std::string toString() const;

        /**
         * @brief Convert to shortest unambiguous string
         *
         * Omits localhost if hostname is "localhost".
         * Example: "cuda:0" instead of "localhost:0:cuda:0" for local GPU.
         *
         * @return Shortened string representation
         */
        std::string toShortString() const;

        // =========================================================================
        // Conversion to/from DeviceId
        // =========================================================================

        /**
         * @brief Convert to local DeviceId (for kernel calls)
         *
         * Note: This discards hostname and NUMA information. Use when
         * you need to make kernel calls on the local node.
         *
         * @return DeviceId with type and ordinal
         */
        DeviceId toLocalDeviceId() const;

        /**
         * @brief Create from local DeviceId with additional context
         * @param local_id Local DeviceId
         * @param hostname Hostname (default "localhost")
         * @param numa_node NUMA node (NUMA_NODE_UNKNOWN if not known)
         * @return GlobalDeviceAddress
         */
        static GlobalDeviceAddress fromLocalDeviceId(
            const DeviceId &local_id,
            const std::string &hostname = "localhost",
            int numa_node = NUMA_NODE_UNKNOWN);

        // =========================================================================
        // Predicates
        // =========================================================================

        /**
         * @brief Check if this is a local device (hostname is "localhost")
         * @return true if hostname is "localhost" or empty
         */
        bool isLocal() const;

        /**
         * @brief Check if this is a CPU device
         * @return true if device_type is CPU
         */
        bool isCPU() const { return device_type == DeviceType::CPU; }

        /**
         * @brief Check if this is a GPU device (CUDA or ROCm)
         * @return true if device_type is CUDA or ROCm
         */
        bool isGPU() const { return device_type == DeviceType::CUDA || device_type == DeviceType::ROCm; }

        /**
         * @brief Check if this is a CUDA device
         * @return true if device_type is CUDA
         */
        bool isCUDA() const { return device_type == DeviceType::CUDA; }

        /**
         * @brief Check if this is a ROCm device
         * @return true if device_type is ROCm
         */
        bool isROCm() const { return device_type == DeviceType::ROCm; }

        /**
         * @brief Check if two addresses share the same NUMA node
         * @param other Other address to compare
         * @return true if same hostname and NUMA node
         */
        bool sameNuma(const GlobalDeviceAddress &other) const;

        /**
         * @brief Check if two addresses are on the same host
         * @param other Other address to compare
         * @return true if same hostname
         */
        bool sameHost(const GlobalDeviceAddress &other) const;

        // =========================================================================
        // Comparison Operators
        // =========================================================================

        bool operator==(const GlobalDeviceAddress &o) const;
        bool operator!=(const GlobalDeviceAddress &o) const;
        bool operator<(const GlobalDeviceAddress &o) const;

    private:
        /**
         * @brief Parse device type string
         * @param type_str Type string (e.g., "cuda", "rocm", "cpu")
         * @return Parsed DeviceType
         * @throws std::invalid_argument if type_str is not recognized
         */
        static DeviceType parseDeviceType(const std::string &type_str);

        /**
         * @brief Convert DeviceType to string
         * @param type DeviceType enum value
         * @return Lowercase string representation
         */
        static std::string deviceTypeToString(DeviceType type);
    };

    /**
     * @brief Stream output operator for GlobalDeviceAddress
     */
    std::ostream &operator<<(std::ostream &os, const GlobalDeviceAddress &addr);

} // namespace llaminar2

// ============================================================================
// std::hash specialization for GlobalDeviceAddress
// ============================================================================
namespace std
{
    template <>
    struct hash<llaminar2::GlobalDeviceAddress>
    {
        size_t operator()(const llaminar2::GlobalDeviceAddress &addr) const noexcept
        {
            // Combine hashes using FNV-1a style mixing
            size_t h = 14695981039346656037ULL; // FNV offset basis
            const size_t prime = 1099511628211ULL;

            // Hash hostname
            for (char c : addr.hostname)
            {
                h ^= static_cast<size_t>(c);
                h *= prime;
            }

            // Hash numa_node
            h ^= static_cast<size_t>(addr.numa_node);
            h *= prime;

            // Hash device_type
            h ^= static_cast<size_t>(addr.device_type);
            h *= prime;

            // Hash device_ordinal
            h ^= static_cast<size_t>(addr.device_ordinal);
            h *= prime;

            return h;
        }
    };
} // namespace std
