#pragma once
#include "backends/DeviceId.h"
#include <vector>
#include <string>
#include <cstddef>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <iomanip>

namespace llaminar2
{

struct DeviceMemoryPlan
{
    DeviceId device;
    size_t weight_bytes = 0;
    size_t kv_cache_bytes = 0;
    size_t activation_bytes = 0;
    size_t workspace_bytes = 0;

    size_t device_total_bytes = 0;   // From DeviceInfo.memory_bytes
    size_t device_free_bytes = 0;    // From DeviceInfo.free_memory_bytes
    size_t headroom_bytes = 128ULL * 1024 * 1024;  // 128 MB default

    size_t total_bytes() const
    {
        return weight_bytes + kv_cache_bytes + activation_bytes + workspace_bytes;
    }

    bool fits() const
    {
        return total_bytes() + headroom_bytes <= device_free_bytes;
    }

    size_t deficit() const
    {
        auto needed = total_bytes() + headroom_bytes;
        return needed > device_free_bytes ? needed - device_free_bytes : 0;
    }

    size_t remaining() const
    {
        auto needed = total_bytes() + headroom_bytes;
        return device_free_bytes > needed ? device_free_bytes - needed : 0;
    }

    std::string summary() const
    {
        auto mb = [](size_t b) { return static_cast<double>(b) / (1024.0 * 1024.0); };
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(0);
        ss << device.to_string() << ": "
           << "weights=" << mb(weight_bytes) << " MB, "
           << "kv_cache=" << mb(kv_cache_bytes) << " MB, "
           << "activations=" << mb(activation_bytes) << " MB, "
           << "workspace=" << mb(workspace_bytes) << " MB, "
           << "total=" << mb(total_bytes()) << "/" << mb(device_free_bytes) << " MB"
           << (fits() ? " [OK]" : " [OVER by " + std::to_string(static_cast<int>(mb(deficit()))) + " MB]");
        return ss.str();
    }
};

struct MemoryPlan
{
    std::vector<DeviceMemoryPlan> devices;
    std::vector<std::string> diagnostics;  // Warnings/errors

    bool fits() const
    {
        return std::all_of(devices.begin(), devices.end(),
            [](const DeviceMemoryPlan& d) { return d.fits(); });
    }

    size_t total_bytes() const
    {
        return std::accumulate(devices.begin(), devices.end(), size_t{0},
            [](size_t acc, const DeviceMemoryPlan& d) { return acc + d.total_bytes(); });
    }

    size_t total_available() const
    {
        return std::accumulate(devices.begin(), devices.end(), size_t{0},
            [](size_t acc, const DeviceMemoryPlan& d) { return acc + d.device_free_bytes; });
    }

    /// Render a libfort-formatted table. Declared here, defined in MemoryPlanner.cpp.
    std::string renderTable() const;
};

} // namespace llaminar2
