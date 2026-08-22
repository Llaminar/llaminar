/**
 * @file GPUAllocationPolicy.h
 * @brief Canonical free-memory invariant for CUDA and ROCm allocations.
 *
 * Backend allocation, model preflight, and ExpertOverlay capacity admission
 * must reserve the same terminal allocator headroom. Keeping this value here
 * prevents a planner from filling VRAM to a point at which the backend rejects
 * even a tiny setup allocation despite an otherwise valid memory BOM.
 */

#pragma once

#include <cstddef>

namespace llaminar2::gpu_allocation_policy
{
    /**
     * @brief Bytes that must remain free after any individual GPU allocation.
     *
     * This is additive to workload/runtime safety reserves: it is the concrete
     * allocator contract enforced immediately before cudaMalloc/hipMalloc.
     */
    inline constexpr std::size_t kMinimumFreeHeadroomBytes =
        64ULL * 1024ULL * 1024ULL;
} // namespace llaminar2::gpu_allocation_policy
