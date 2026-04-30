/**
 * @file DeviceRegistry.h
 * @brief Singleton device registry for heterogeneous compute management
 *
 * Provides centralized device discovery and management using GlobalDeviceAddress
 * as the canonical identifier. Supports CPU NUMA nodes, CUDA GPUs, and ROCm GPUs.
 *
 * **Usage**:
 * ```cpp
 * auto& registry = DeviceRegistry::instance();
 * registry.discover();
 *
 * // Get all CUDA devices
 * auto cudas = registry.devicesByType(DeviceType::CUDA);
 * for (const auto& addr : cudas) {
 *     LOG_INFO("Found " << addr.toString()
 *              << " with " << registry.memoryCapacity(addr) / 1e9 << " GB");
 * }
 *
 * // Get backend for a device
 * IBackend* backend = registry.backendFor(cudas[0]);
 * ```
 *
 * **Thread Safety**: All methods are thread-safe via internal mutex.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "IDeviceRegistry.h"
#include "IBackend.h"
#include <mutex>
#include <unordered_map>
#include <memory>
#include <optional>

namespace llaminar2
{

    /**
     * @brief Singleton device registry implementation
     *
     * Discovers and manages all compute devices available on the system.
     * Uses GlobalDeviceAddress for consistent device identification.
     */
    class DeviceRegistry : public IDeviceRegistry
    {
    public:
        /**
         * @brief Get the singleton instance
         * @return Reference to the global DeviceRegistry
         */
        static DeviceRegistry &instance();

        // =========================================================================
        // IDeviceRegistry Implementation
        // =========================================================================

        void discover() override;
        bool isDiscovered() const override;

        std::vector<GlobalDeviceAddress> allDevices() const override;
        std::vector<GlobalDeviceAddress> devicesByType(DeviceType type) const override;
        std::vector<GlobalDeviceAddress> devicesByNuma(int numa_node) const override;

        bool isValid(const GlobalDeviceAddress &addr) const override;
        bool isAvailable(const GlobalDeviceAddress &addr) const override;

        size_t memoryCapacity(const GlobalDeviceAddress &addr) const override;
        size_t memoryAvailable(const GlobalDeviceAddress &addr) const override;
        std::pair<int, int> computeCapability(const GlobalDeviceAddress &addr) const override;
        std::string deviceName(const GlobalDeviceAddress &addr) const override;

        IBackend *backendFor(const GlobalDeviceAddress &addr) override;

        int numaAffinity(const GlobalDeviceAddress &addr) const override;
        std::string pcieBusId(const GlobalDeviceAddress &addr) const override;
        bool canP2P(const GlobalDeviceAddress &a, const GlobalDeviceAddress &b) const override;

        size_t deviceCount(DeviceType type) const override;
        size_t totalDeviceCount() const override;

        // =========================================================================
        // Additional Methods
        // =========================================================================

        /**
         * @brief Clear cached device info and re-discover
         *
         * Useful after hot-plug events or for testing.
         */
        void refresh();

        /**
         * @brief Get the default device for a type
         *
         * Returns the first available device of the specified type.
         * For CPU: NUMA node 0
         * For CUDA/ROCm: Ordinal 0
         *
         * @param type Device type
         * @return Default device address, or empty optional if none available
         */
        std::optional<GlobalDeviceAddress> defaultDevice(DeviceType type) const;

        /**
         * @brief Return architecture metadata for a local DeviceId if discovered.
         *
         * Metadata is descriptive only; DeviceId identity remains (type, ordinal).
         */
        std::optional<DeviceArchInfo> archInfo(DeviceId device) const;

        /**
         * @brief Return a copy of DeviceId enriched with discovered metadata.
         *
         * If the registry has no metadata for the device, returns the original
         * DeviceId unchanged.
         */
        DeviceId resolve(DeviceId device) const;

    private:
        DeviceRegistry();
        ~DeviceRegistry() = default;

        // Non-copyable, non-movable
        DeviceRegistry(const DeviceRegistry &) = delete;
        DeviceRegistry &operator=(const DeviceRegistry &) = delete;
        DeviceRegistry(DeviceRegistry &&) = delete;
        DeviceRegistry &operator=(DeviceRegistry &&) = delete;

        // =========================================================================
        // Internal State
        // =========================================================================

        mutable std::mutex mutex_;
        bool discovered_ = false;

        // All discovered devices
        std::vector<GlobalDeviceAddress> devices_;

        // Index from canonical string to vector index (for fast lookup)
        std::unordered_map<std::string, size_t> device_index_;

        // =========================================================================
        // Cached Device Info
        // =========================================================================

        struct DeviceInfo
        {
            std::string name;
            size_t memory_capacity = 0;
            int numa_affinity = -1;
            std::string pcie_bus_id;
            std::pair<int, int> compute_capability{0, 0};
            std::optional<DeviceArchInfo> arch_info;
            bool available = true;
        };
        std::unordered_map<std::string, DeviceInfo> device_info_;

        // =========================================================================
        // Discovery Helpers
        // =========================================================================

        void discoverCpuDevices();
        void discoverCudaDevices();
        void discoverRocmDevices();

        /**
         * @brief Get the canonical key for a device address
         * @param addr Device address
         * @return Canonical string key for map lookup
         */
        std::string getKey(const GlobalDeviceAddress &addr) const;

        /**
         * @brief Find DeviceInfo for an address
         * @param addr Device address
         * @return Pointer to DeviceInfo or nullptr if not found
         */
        const DeviceInfo *findInfo(const GlobalDeviceAddress &addr) const;
    };

} // namespace llaminar2
