/**
 * @file ProfilingContext.h
 * @brief Thread-local device context for multi-device profiling attribution
 * @author David Sanftenberg
 *
 * Provides thread-local storage for tracking which device is currently executing,
 * allowing profilers to attribute timing data to specific devices. This is essential
 * for multi-device tensor parallelism where multiple devices execute in parallel.
 *
 * Usage:
 *   // In device runner thread:
 *   ProfilingContext::setCurrentDevice(DeviceId::cuda(0));
 *   // ... execute kernels - profiling automatically attributed to cuda:0 ...
 *   ProfilingContext::clearCurrentDevice();
 *
 *   // Or use RAII:
 *   {
 *       ProfilingDeviceScope scope(DeviceId::cuda(0));
 *       // ... execute kernels ...
 *   } // Device context auto-cleared
 */
#pragma once

#include "../backends/DeviceId.h"
#include <optional>
#include <string>

namespace llaminar2
{

    /**
     * @brief Thread-local device context for profiling attribution
     *
     * Each thread maintains its own current device context. When kernels execute
     * and record profiling data, the profilers use this context to attribute
     * the timing to the correct device.
     */
    class ProfilingContext
    {
    public:
        /**
         * @brief Set the current device for this thread
         * @param device The device executing on this thread
         */
        static void setCurrentDevice(DeviceId device)
        {
            current_device() = device;
        }

        /**
         * @brief Clear the current device context for this thread
         */
        static void clearCurrentDevice()
        {
            current_device() = std::nullopt;
        }

        /**
         * @brief Get the current device for this thread
         * @return Device ID if set, or std::nullopt if not in a device context
         */
        static std::optional<DeviceId> getCurrentDevice()
        {
            return current_device();
        }

        /**
         * @brief Get a string key for the current device (for map indexing)
         * @return Device string like "cuda:0", "rocm:1", "cpu", or "unknown"
         */
        static std::string getCurrentDeviceKey()
        {
            auto device = current_device();
            if (!device.has_value())
            {
                return "unknown";
            }
            return device->toString();
        }

        /**
         * @brief Check if we're in a device context
         */
        static bool hasDeviceContext()
        {
            return current_device().has_value();
        }

    private:
        static std::optional<DeviceId> &current_device()
        {
            thread_local std::optional<DeviceId> device;
            return device;
        }
    };

    /**
     * @brief RAII helper to set/clear device context
     *
     * Automatically sets device context on construction and clears on destruction.
     * Use this when entering device-specific execution scope.
     */
    class ProfilingDeviceScope
    {
    public:
        explicit ProfilingDeviceScope(DeviceId device)
        {
            ProfilingContext::setCurrentDevice(device);
        }

        ~ProfilingDeviceScope()
        {
            ProfilingContext::clearCurrentDevice();
        }

        // Non-copyable, non-movable
        ProfilingDeviceScope(const ProfilingDeviceScope &) = delete;
        ProfilingDeviceScope &operator=(const ProfilingDeviceScope &) = delete;
    };

} // namespace llaminar2
