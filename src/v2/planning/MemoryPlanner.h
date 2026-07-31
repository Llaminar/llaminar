#pragma once
#include "planning/MemoryPlan.h"
#include "planning/ModelMemoryProfile.h"
#include "backends/DeviceId.h"

#include <vector>
#include <string>

namespace llaminar2
{

/// Configuration for a single device in a memory plan.
struct DevicePlanConfig
{
    DeviceId device;
    size_t device_total_bytes = 0;
    size_t device_free_bytes = 0;

    // TP configuration for this device
    int shard_index = 0;
    int total_shards = 1;

    // PP configuration: layer range
    int first_layer = 0;
    int last_layer = -1;  // -1 = all

    // KV cache configuration
    std::string kv_precision = "fp16";
    int local_kv_heads = 0;   // After TP sharding, 0 = use profile.n_kv_heads

    // Runtime parameters
    int batch_size = 1;
    int max_seq_len = 0;  // 0 = use profile.max_seq_len
    int activation_seq_len = 0;  // 0 = use max_seq_len for activation/workspace
    bool mtp_enabled = false;

    // Headroom
    size_t headroom_bytes = 128ULL * 1024 * 1024;
};

/**
 * @brief Result of selecting one common resident graph-row capacity.
 *
 * GPU participants in one forward graph family must agree on the largest
 * prefill shape that can be resident at once.  The full context remains owned
 * by the KV cache; only activation and kernel-workspace rows are bounded by
 * this value.  Long prompts are represented by serial replays of captured
 * bucket graphs.
 */
struct ResidentGraphMemoryPlan
{
    int resident_graph_rows = 0;
    MemoryPlan memory_plan;

    /// @brief True when the selected row capacity and full context both fit.
    bool fits() const { return resident_graph_rows > 0 && memory_plan.fits(); }
};

class MemoryPlanner
{
public:
    /// Plan memory for a set of devices with the given model profile.
    static MemoryPlan plan(
        const ModelMemoryProfile& profile,
        const std::vector<DevicePlanConfig>& device_configs
    );

    /**
     * @brief Select the largest configured graph row bucket that fits.
     *
     * Every GPU config is evaluated with the same candidate so LocalTP and
     * other symmetric graph domains cannot silently construct incompatible
     * resident shapes. CPU configs retain their requested activation length.
     * When no candidate fits, the returned plan contains the smallest
     * candidate's diagnostics and `fits()` is false.
     */
    static ResidentGraphMemoryPlan planLargestFittingResidentGraphRows(
        const ModelMemoryProfile& profile,
        const std::vector<DevicePlanConfig>& device_configs,
        const std::vector<int>& candidate_rows
    );
};

} // namespace llaminar2
