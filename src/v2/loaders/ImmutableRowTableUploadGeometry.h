/**
 * @file ImmutableRowTableUploadGeometry.h
 * @brief Source-owned eligibility and zero-increment BOM for reused loader staging.
 *
 * Planning supplies the original canonical ordinary-loader geometry; table extent
 * never sizes staging. Materialization consumes its finalized typed receipt and
 * repeats this same native-row fit check. Reusing one drained host slot adds no
 * capacity or device staging to the already-admitted higher-priority load BOM.
 */
#pragma once
#include "loaders/GPUVramPreflight.h"
#include "planning/PhysicalMemoryAuthority.h"
namespace llaminar2
{
/** @brief Check one complete original native row against canonical loader geometry.
 * @param row_bytes Exact positive original native row stride.
 * @param ordinary Canonical resolveGPUWeightLoadMemoryGeometry output for ordinary weights.
 * @return Whether the existing actual host slot can hold one row; no resize is performed.
 * @throws std::invalid_argument For malformed geometry rather than an ineligible row.
 */
[[nodiscard]] inline bool residentImmutableRowUploadFits(std::size_t row_bytes,
    const GPUWeightLoadMemoryGeometry &ordinary)
{
    if (!row_bytes) throw std::invalid_argument("Resident immutable upload requires positive native row bytes");
    if (!ordinary.staging_slot_bytes || ordinary.staging_stream_count <= 0)
        return false;
    const auto count = static_cast<std::size_t>(ordinary.staging_stream_count);
    if (ordinary.staging_slot_bytes > std::numeric_limits<std::size_t>::max() / count ||
        ordinary.host_staging_bytes != ordinary.staging_slot_bytes * count ||
        ordinary.staging_slot_stride_bytes != alignGPUWeightLoadAllocation(ordinary.staging_slot_bytes) ||
        ordinary.staging_slot_stride_bytes > std::numeric_limits<std::size_t>::max() / count ||
        ordinary.staging_bytes != ordinary.staging_slot_stride_bytes * count ||
        ordinary.staging_slot_bytes > ordinary.maximum_source_bytes)
        throw std::invalid_argument("Resident immutable upload received noncanonical ordinary-loader geometry");
    return row_bytes <= ordinary.staging_slot_bytes;
}
/** @return Exact incremental capacity: zero because only a drained original slot is expressible. */
[[nodiscard]] inline PhysicalMemoryPlan residentImmutableRowUploadIncrementalBOM()
{ return {}; }
}
