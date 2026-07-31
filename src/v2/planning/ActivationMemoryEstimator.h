#pragma once
#include "backends/DeviceId.h"
#include "planning/ModelMemoryProfile.h"
#include <cstddef>

namespace llaminar2
{

class ActivationMemoryEstimator
{
public:
    /// Estimate activation buffer memory for a single device.
    /// This accounts for the BufferArena allocation (hidden states, QKV projections,
    /// FFN intermediates, logits, residuals).
    static size_t estimate(
        int batch_size,
        int max_seq_len,
        int d_model,
        int d_ff,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int vocab_size,
        DeviceId device
    );

    /**
     * @brief Estimate the logical BufferArena ownership for a model profile.
     *
     * This overload includes architecture-specific projection buffers inferred
     * from the GGUF tensor inventory. It is the production planning entrypoint;
     * the scalar overload remains useful for focused generic-model tests.
     */
    static size_t estimate(
        const ModelMemoryProfile& profile,
        int batch_size,
        int resident_graph_rows,
        int local_d_ff,
        int local_n_heads,
        int local_n_kv_heads,
        int first_layer,
        int last_layer,
        int total_shards,
        DeviceId device
    );
};

} // namespace llaminar2
