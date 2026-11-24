/**
 * @file CPUNativeX86PackedPrefill.h
 * @brief x86 decode-once prefill over admitted participant workspace.
 *
 * The caller selects this route only for ordinary prefill. Verifiers and
 * explicit alternative schedules retain their original arithmetic leaves.
 */
#pragma once

#include "CPUNativeQuantisedWeightView.h"
#include "CPUNativeX86PanelWorkspace.h"
#include "CPUProjectionRowGrid.h"

#include <span>

namespace llaminar2::cpu::native_quantised::x86_panel
{
    /**
     * @brief Consume file-density weights once per N strip and physical K panel.
     * @param weights Borrowed admitted resident weights with original factor planes.
     * @param format Exact decoder for those weights.
     * @param activation Persistent Q8 rows in original coordinates.
     * @param output Caller-owned output; only live rows/columns are written.
     * @param rows Physical row count (including original inactive positions).
     * @param output_stride Caller-defined row stride, in floats.
     * @param partitions Independent M1 arithmetic partitions, in original ascending order.
     * @param workers Current participant workshare width.
     * @param lanes Runtime x86 vector width expressed in output columns.
     * @param grid Admitted original-row workshare grid; no compaction.
     * @param work Original optional per-request extents, retained for alias proof.
     * @param scratch Existing projection slab, disjoint from all live data.
     */
    void project(const CPUNativeQuantisedWeightView &weights, Format format, const Q8_1Block *activation, float *output,
                 int rows, int output_stride, int partitions, int workers, int lanes, const CPUProjectionRowGrid &grid,
                 const ProjectionRowWork *work, std::span<float> scratch);
} // namespace llaminar2::cpu::native_quantised::x86_panel
