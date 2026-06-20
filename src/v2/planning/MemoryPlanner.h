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

    // Headroom
    size_t headroom_bytes = 128ULL * 1024 * 1024;
};

class MemoryPlanner
{
public:
    /// Plan memory for a set of devices with the given model profile.
    static MemoryPlan plan(
        const ModelMemoryProfile& profile,
        const std::vector<DevicePlanConfig>& device_configs
    );
};

} // namespace llaminar2
