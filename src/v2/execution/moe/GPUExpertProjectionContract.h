/**
 * @file GPUExpertProjectionContract.h
 * @brief Metadata-only admission shared by GPU expert execution and planning.
 *
 * One payload arithmetic tag authenticates its projection bundle. Native
 * codebooks may differ; mixed native/floating or floating-precision bundles
 * need a distinct implementation and must be rejected before preparation.
 * This interface contributes no allocation or execution-state authority.
 */
#pragma once
#include "GpuExpertSlotPool.h"
#include <vector>

namespace llaminar2
{
    /**
     * @brief Admit one exact GPU expert projection bundle without payload I/O.
     * @param specs Authenticated gate/up or gate/up/down roles and source formats.
     * @param projection_set Frozen complete-expert or gate/up-only ownership.
     * @param layer_index Actual model layer for precise source diagnostics, or -1.
     * @throws std::invalid_argument For missing roles, invalid geometry or unsupported arithmetic.
     *
     * Live directory admission and captured expert cost probes use the same
     * format contract. A planner must not skip it and turn an implementation
     * limitation into a late, generic graph-capture failure.
     */
    void validateGPUExpertProjectionContract(
        const std::vector<GpuExpertSlotPool::ProjectionSpec> &specs,
        DeviceMoEProjectionSet projection_set, int layer_index = -1);
}
