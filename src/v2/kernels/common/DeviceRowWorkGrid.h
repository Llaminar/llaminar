/**
 * @file DeviceRowWorkGrid.h
 * @brief Capture-time worker admission for device-counted row-tile loops.
 *
 * Physical matrix capacity owns storage, not the amount of mandatory scheduling
 * work on each replay. A counted kernel may use a bounded row grid and visit
 * further live tiles by grid stride. The existing launch policy supplies its
 * desired parallelism; this helper neither invents a second policy nor reads
 * the device count. Rectangular callers retain their original full row grid.
 */
#pragma once

#include "kernels/common/DeviceRowRange.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace llaminar2
{
    /**
     * @brief Bound row workers after accounting for independent column/K tasks.
     * @param rows Captured physical tile and optional borrowed device count.
     * @param rows_per_task Fixed arithmetic reuse width; one for reducers.
     * @param independent_tasks Column tiles times K partitions, without rows.
     * @param target_tasks Existing backend policy's desired parallel task count.
     * @return Positive row-grid dimension; device iteration strides by this grid.
     * @throws std::invalid_argument If any launch extent is nonpositive.
     *
     * A wide projection already supplies enough independent work to occupy the
     * device. Extra physical row capacity must then add loop iterations only
     * when those rows are live. Narrow projections retain enough row workers to
     * meet the same parallelism policy. No allocation or live-state shadow is
     * involved, and wider row loops never change per-row arithmetic ordering.
     */
    [[nodiscard]] inline int deviceRowWorkerGroups(
        const DeviceRowRange &rows, int rows_per_task,
        std::int64_t independent_tasks, std::int64_t target_tasks)
    {
        if (rows_per_task <= 0 || independent_tasks <= 0 || target_tasks <= 0)
            throw std::invalid_argument("Device row grid requires positive launch geometry");
        const int physical_groups = 1 + (rows.physicalRows() - 1) / rows_per_task;
        if (!rows.countOwner())
            return physical_groups;
        // Subtract before ceiling division: valid large grids must not overflow.
        const auto needed_groups = 1 + (target_tasks - 1) / independent_tasks;
        return static_cast<int>(std::min<std::int64_t>(physical_groups, needed_groups));
    }
}
