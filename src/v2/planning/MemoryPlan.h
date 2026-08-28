/**
 * @file MemoryPlan.h
 * @brief Final and incremental device-memory admission result types.
 *
 * A plan reports the complete post-initialization footprint while separately
 * tracking bytes that are already resident and reflected in the device's live
 * free-memory reading. This distinction lets a fresh graph adopt immutable
 * prepared weights without either hiding them from the BOM or charging their
 * allocation twice.
 */

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
    int max_seq_len = 0;
    int activation_seq_len = 0;
    /** Primary graph-view model weights. */
    size_t weight_bytes = 0;
    /** Concurrent alternate views such as replicated dense decode weights. */
    size_t additional_weight_bytes = 0;
    /** Planned weight bytes already resident before this runner's admission. */
    size_t retained_weight_bytes = 0;
    size_t kv_cache_bytes = 0;
    size_t persistent_state_bytes = 0;
    size_t live_recurrent_state_bytes = 0;
    size_t checkpoint_state_bytes = 0;
    /** Persistent device slots used to archive/restore one prefix payload. */
    size_t prefix_cache_staging_bytes = 0;
    /** Configured bounded capacity of the accelerator-hot prefix tier. */
    size_t prefix_cache_device_hot_bytes = 0;
    /** Opaque native-driver storage retained by the complete graph family. */
    size_t captured_graph_bytes = 0;
    /** Persistent backend and FP16 scratch allocations owned by LocalTP. */
    size_t collective_bytes = 0;
    size_t activation_bytes = 0;
    size_t workspace_bytes = 0;
    /** Workspace bytes already owned by a sealed model-context backing lease. */
    size_t retained_workspace_bytes = 0;

    size_t device_total_bytes = 0;   // From DeviceInfo.memory_bytes
    size_t device_free_bytes = 0;    // From DeviceInfo.free_memory_bytes
    size_t total_bytes() const
    {
        return weight_bytes + additional_weight_bytes + kv_cache_bytes +
               persistent_state_bytes + prefix_cache_staging_bytes +
               prefix_cache_device_hot_bytes + captured_graph_bytes +
               collective_bytes +
               activation_bytes + workspace_bytes;
    }

    /** @return Complete persistent weight footprint across every physical view. */
    size_t total_weight_bytes() const
    {
        return weight_bytes + additional_weight_bytes;
    }

    /** @return Weight bytes this runner must newly allocate. */
    size_t incremental_weight_bytes() const
    {
        const size_t total = total_weight_bytes();
        return total - std::min(total, retained_weight_bytes);
    }

    /**
     * @return Bytes newly required from the currently free device capacity.
     *
     * Retained weights remain part of total_bytes(), but the device allocator
     * has already removed them from device_free_bytes.
     */
    size_t incremental_bytes() const
    {
        const size_t incremental_workspace =
            workspace_bytes -
            std::min(workspace_bytes, retained_workspace_bytes);
        return incremental_weight_bytes() + kv_cache_bytes +
               persistent_state_bytes + prefix_cache_staging_bytes +
               prefix_cache_device_hot_bytes + captured_graph_bytes +
               collective_bytes +
               activation_bytes + incremental_workspace;
    }

    bool fits() const
    {
        return incremental_bytes() <= device_free_bytes;
    }

    size_t deficit() const
    {
        auto needed = incremental_bytes();
        return needed > device_free_bytes ? needed - device_free_bytes : 0;
    }

    size_t remaining() const
    {
        auto needed = incremental_bytes();
        return device_free_bytes > needed ? device_free_bytes - needed : 0;
    }

    std::string summary() const
    {
        auto mb = [](size_t b) { return static_cast<double>(b) / (1024.0 * 1024.0); };
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(0);
        ss << device.to_string() << ": "
           << "weights=" << mb(weight_bytes) << " MB, "
           << "additional_weights=" << mb(additional_weight_bytes) << " MB, "
           << "retained_weights=" << mb(retained_weight_bytes) << " MB, "
           << "kv_cache=" << mb(kv_cache_bytes) << " MB, "
           << "state=" << mb(persistent_state_bytes) << " MB, "
           << "prefix_staging=" << mb(prefix_cache_staging_bytes) << " MB, "
           << "prefix_hot=" << mb(prefix_cache_device_hot_bytes) << " MB, "
           << "captured_graphs=" << mb(captured_graph_bytes) << " MB, "
           << "collective=" << mb(collective_bytes) << " MB, "
           << "activations=" << mb(activation_bytes) << " MB, "
           << "workspace=" << mb(workspace_bytes) << " MB, "
           << "retained_workspace=" << mb(retained_workspace_bytes) << " MB, "
           << "total=" << mb(total_bytes()) << " MB, "
           << "new=" << mb(incremental_bytes()) << "/" << mb(device_free_bytes) << " MB"
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
