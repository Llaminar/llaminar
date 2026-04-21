#pragma once
#include "backends/DeviceId.h"
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
};

} // namespace llaminar2
