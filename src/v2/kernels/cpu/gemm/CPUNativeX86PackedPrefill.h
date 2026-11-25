/**
 * @file CPUNativeX86PackedPrefill.h
 * @brief x86 decode-once prefill over admitted participant workspace.
 *
 * The caller selects this route only for ordinary prefill. Verifiers and
 * explicit alternative schedules retain their original arithmetic leaves.
 * Dense projections workshare their own strips (project()); pooled members
 * of the batched expert route contribute one task per 64-column chunk to
 * their launch's workshare (projectPooledChunk()).
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

    /**
     * @brief Execute one 64-column chunk task of a pooled member on the calling worker's slot.
     * @param weights Borrowed admitted resident weights of the member.
     * @param format Exact decoder for those weights (formatFor() of their encoding).
     * @param activation The member's Q8 rows; every row is live.
     * @param output The member's output; only this chunk's columns of its rows are written.
     * @param rows The member's rows (at least min_pooled_rows).
     * @param output_stride The member's output row stride, in floats.
     * @param partitions The member's serial K partition count (at least one).
     * @param lanes Runtime x86 vector width (8 AVX2, 16 AVX-512); the strip is pooledStrip(lanes).
     * @param chunk Physical 64-column chunk owned by this task.
     * @param slot The calling worker's slot of the launch's admitted slab, at least
     *        workerFloats(format, pooledStrip(lanes), rows, blocks, partitions) floats.
     * @details The launcher admits geometry, slab extent, team width and
     * aliasing for every member before its team writes anything. Alignment is
     * not part of the contract: slots are whole 64-byte multiples apart
     * relative to the slab base, every slot access is unaligned-safe, and
     * production workspace slabs are 64-byte aligned so that slots never share
     * a cache line. A task has
     * no synchronization and writes only its chunk and its own slot, so tasks
     * of different members and chunks interleave freely in one workshare.
     * Each strip decodes every K panel once and reuses it for all rows, with
     * the dense panel program's exact serial M=1 arithmetic.
     */
    void projectPooledChunk(const CPUNativeQuantisedWeightView &weights, Format format, const Q8_1Block *activation,
                            float *output, int rows, int output_stride, int partitions, int lanes, int chunk,
                            float *slot);
} // namespace llaminar2::cpu::native_quantised::x86_panel
