/**
 * @file MoEOverlayHostRowWork.h
 * @brief Socket-local worksharing for independent host sparse-payload rows.
 *
 * These operations address already admitted host storage, not tensor coherence
 * or device transfers. The graph owns source lifetime and publication ordering.
 * Large prefill packets use the inference socket's existing OpenMP worker budget;
 * decode packets stay on the caller. A call from one worker of an existing team
 * must not introduce an orphaned workshare or a nested team.
 */
#pragma once

#include "utils/OpenMPUtils.h"

#include <cstddef>

namespace llaminar2
{
    /**
     * @brief Visit disjoint payload rows and join before returning to their publisher.
     * @param rows Number of live rows, never the allocation's unused capacity.
     * @param row_bytes Bytes copied or packed per row; used only for work granularity.
     * @param work One row's independent operation, which must not throw or publish state.
     *
     * The 128-KiB minimum amortizes a worker-team entry and keeps small decode
     * packets serial. Division avoids overflowing a row-count/stride product.
     * The caller publishes counts/events only after this function has joined.
     */
    template <typename RowWork>
    inline void forEachMoEOverlayHostRow(size_t rows, size_t row_bytes, const RowWork &work)
    {
        constexpr size_t kMinimumParallelBytes = 128u * 1024u;
        const bool parallel = !omp_in_parallel() && rows > 1u && row_bytes != 0u &&
            rows >= kMinimumParallelBytes / row_bytes +
                        (kMinimumParallelBytes % row_bytes != 0u);
        if (parallel)
        {
            const auto workshare = [&]() {
#pragma omp for schedule(static)
                for (size_t row = 0; row < rows; ++row)
                    work(row);
            };
            OMP_WORKSHARE_REGION(workshare);
        }
        else
        {
            for (size_t row = 0; row < rows; ++row)
                work(row);
        }
    }
}
