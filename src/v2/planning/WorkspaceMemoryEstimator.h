#pragma once
#include "backends/DeviceId.h"
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
};

} // namespace llaminar2
