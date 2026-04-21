#pragma once
#include "backends/DeviceId.h"
#include <cstddef>
#include <string>

namespace llaminar2
{

class KVCacheMemoryEstimator
{
public:
    /// Estimate KV cache memory in bytes for a single device.
    /// @param n_layers       Number of transformer layers on this device (PP slice)
    /// @param batch_size     Batch size
    /// @param max_seq_len    Maximum sequence length (context window)
    /// @param n_kv_heads     Number of KV heads (local, after TP sharding)
    /// @param head_dim       Dimension per head
    /// @param kv_precision   "fp32", "fp16", "q8_1", "tq", "auto"
    /// @param device         Target device (GPU vs CPU affects TQ availability)
    static size_t estimate(
        int n_layers,
        int batch_size,
        int max_seq_len,
        int n_kv_heads,
        int head_dim,
        const std::string& kv_precision,
        DeviceId device
    );

    /// Bytes per KV element for a given precision.
    static float getBytesPerElement(const std::string& kv_precision);
};

} // namespace llaminar2
