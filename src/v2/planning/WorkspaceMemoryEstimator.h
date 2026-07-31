#pragma once
#include "backends/DeviceId.h"
#include "planning/ModelMemoryProfile.h"
#include <cstddef>

namespace llaminar2
{

class WorkspaceMemoryEstimator
{
public:
    /// Estimate kernel workspace memory for a device.
    /// GPU devices need workspace for GEMM, LM head, and other kernels.
    /// CPU devices typically have zero workspace (use stack/heap).
    static size_t estimate(
        int batch_size,
        int max_seq_len,
        int d_model,
        int d_ff,
        int vocab_size,
        DeviceId device
    );

    /**
     * @brief Estimate the production graph-family workspace from GGUF geometry.
     */
    static size_t estimate(
        const ModelMemoryProfile& profile,
        int batch_size,
        int resident_graph_rows,
        int local_d_ff,
        int first_layer,
        int last_layer,
        int total_shards,
        DeviceId device
    );
};

} // namespace llaminar2
