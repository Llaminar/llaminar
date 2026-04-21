#pragma once
#include "backends/DeviceId.h"
#include <cstddef>
#include <string>

namespace llaminar2
{

struct ModelMemoryProfile;

struct WeightEstimate
{
    size_t native_bytes = 0;    // As stored in GGUF
    size_t device_bytes = 0;    // After device-specific packing/repacking
};

class WeightMemoryEstimator
{
public:
    /// Estimate weight memory for a device, accounting for TP sharding and PP layer range.
    static WeightEstimate estimate(
        const ModelMemoryProfile& profile,
        DeviceId device,
        int shard_index = 0,
        int total_shards = 1,
        int first_layer = 0,
        int last_layer = -1  // -1 = all layers
    );

    /// Bytes per weight element for native (GGUF on-disk) format.
    static float getNativeBytesPerWeight(const std::string& quant_type);

    /// Bytes per weight element after CUDA/ROCm repacking (Q8_0 → int8 packed).
    static float getCUDAPackedBytesPerWeight(size_t K);

    /// Bytes per weight element after CPU VNNI packing.
    static float getCPUPackedBytesPerWeight(const std::string& quant_type);

private:
    /// Is this tensor sharded (column or row parallel) in TP mode?
    static bool isShardedTensor(const std::string& tensor_name);

    /// Is this tensor replicated across all TP ranks (norms, embeddings)?
    static bool isReplicatedTensor(const std::string& tensor_name);
};

} // namespace llaminar2
