/**
 * @file ROCmMoEKernelResources.h
 * @brief Read-only compiler resource evidence for ROCm runtime route grouping.
 *
 * Tests and profiling tools inspect the actual loaded specialization. These
 * values are not allocation estimates or a second physical-memory ledger, and
 * querying them is an initialization/diagnostic operation, never a graph node.
 */
#pragma once

#include <cstddef>
#include <optional>

namespace llaminar2
{
    /** @brief The four retained small-route grouping transactions. */
    enum class ROCmRuntimeGroupVariant
    {
        RouterRoutes,
        AssignedRoutes,
        RouterCompletePlan,
        AssignedCompletePlan,
    };

    /** @brief Immutable compiler resources and launch occupancy for one kernel. */
    struct ROCmRuntimeGroupResources
    {
        int registers_per_thread;
        std::size_t private_bytes_per_thread;
        std::size_t shared_bytes_per_block;
        int launch_threads;
        int maximum_threads_per_block;
        int resident_blocks_per_multiprocessor;
    };

    /**
     * @brief Inspect a compiled specialization on the current ROCm device.
     * @param variant Exact transaction whose resources are requested.
     * @return Loaded kernel resources, or no value for an invalid variant/HIP
     *         query failure. Callers must report failure, not substitute values.
     * @pre The caller owns an initialized ROCm worker context and is outside
     *      graph capture. No launch, allocation, or synchronization occurs.
     */
    [[nodiscard]] std::optional<ROCmRuntimeGroupResources>
    queryROCmRuntimeGroupResources(ROCmRuntimeGroupVariant variant);
}
