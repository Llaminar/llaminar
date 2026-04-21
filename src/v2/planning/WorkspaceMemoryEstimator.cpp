#include "planning/WorkspaceMemoryEstimator.h"

#include <algorithm>

namespace llaminar2
{

size_t WorkspaceMemoryEstimator::estimate(
    int batch_size,
    int max_seq_len,
    int d_model,
    int d_ff,
    int vocab_size,
    DeviceId device)
{
    // CPU kernels don't use a GPU workspace manager
    if (device.is_cpu())
    {
        return 0;
    }

    if (batch_size <= 0 || d_model <= 0 || vocab_size <= 0)
    {
        return 0;
    }

    size_t B = static_cast<size_t>(batch_size);
    size_t S = static_cast<size_t>(max_seq_len);
    size_t D = static_cast<size_t>(d_model);
    size_t V = static_cast<size_t>(vocab_size);

    constexpr size_t FP32 = 4;

    // Formula from WorkspaceAllocator::computeModelAwareBudgetFloor:
    // LM head workspace: 3 × batch × vocab × 4 bytes (output + temp + padded)
    size_t lm_head_workspace = 3 * B * V * FP32;

    // Per-layer GEMM overhead: 2 × seq × d_model × 4 bytes
    size_t gemm_overhead = 2 * S * D * FP32;

    // Padded N buffer: 8 × vocab × 4 bytes
    size_t padded_n = 8 * V * FP32;

    // Embed table temp: vocab × d_model × 4 bytes
    size_t embed_temp = V * D * FP32;

    // Sum with 10% safety margin
    size_t raw = lm_head_workspace + gemm_overhead + padded_n + embed_temp;
    size_t with_margin = static_cast<size_t>(static_cast<double>(raw) * 1.1);

    // Floor: 768 MB (matches WorkspaceAllocator)
    constexpr size_t FLOOR = 768ULL * 1024 * 1024;

    return std::max(FLOOR, with_margin);
}

} // namespace llaminar2
