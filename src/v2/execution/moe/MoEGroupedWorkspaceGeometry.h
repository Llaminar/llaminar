/**
 * @file MoEGroupedWorkspaceGeometry.h
 * @brief Canonical lightweight layout extent for the ROCm grouped work bank.
 *
 * The workspace BOM and HIP launch admission share this one geometry authority.
 * It describes storage only, without importing tensors, host SIMD code, or any
 * allocation policy into device compilation. Population workers reuse its bank
 * with a compact payload and a trailing live-count publication.
 */
#pragma once
#include <algorithm>
#include <cstddef>
namespace llaminar2::MoEWorkspaceBuffers
{
/**
 * @brief Return persistent words for the adaptive ROCm prefill planner.
 *
 * The directory owns two device-published count words followed by
 * independent conservative TM12 and TM16 spans. Each populated expert
 * contributes at most one partially filled tile, which bounds a span by
 * `active_experts + ceil(total_slots / tile_rows)`. Clamping each span
 * to total route slots keeps the contract valid for very small models
 * without reserving a dense expert-by-row rectangle. The fixed-TM4 planner
 * can instead initialize its complete conservative span. These layouts never execute
 * concurrently, so they share a bank sized for their maximum, not their sum.
 */
inline std::size_t rocmAdaptivePrefillDirectoryWords(
    std::size_t total_slots,
    std::size_t num_experts)
{
    total_slots = std::max<std::size_t>(1, total_slots);
    num_experts = std::max<std::size_t>(1, num_experts);
    const std::size_t active_experts =
        std::min(total_slots, num_experts);
    const auto capacity = [=](std::size_t tile_rows)
    {
        return std::min(
            total_slots,
            active_experts +
                (total_slots + tile_rows - 1) / tile_rows);
    };
    return std::max(2u + capacity(12u) + capacity(16u), capacity(4u));
}
}
