/**
 * @file PreparedOverlaySourceRetirement.h
 * @brief Retire frozen ExpertOverlay upload sources at their preparation boundary.
 *
 * Explicit expert selections are owned by graph-frozen bindings, not by the
 * legacy WeightManager caches. Their prepared engines become the execution and
 * movement authority after upload. Retirement therefore follows those exact
 * bindings and registry identities; it must not depend on a later cache sweep.
 */
#pragma once

#include <cstddef>

namespace llaminar2
{
    class FrozenModelWeightSet;
    class MoEExpertOverlayPreparationPlan;
    class ExpertGemmRegistry;
    class WeightMetadataRegistry;

    /**
     * @brief Release owned GPU-only expert sources after complete preparation.
     *
     * Validates every selected expert's exact participant and domain ownership
     * before releasing any source. Borrowed views, CPU execution sources, and
     * graph-lifetime requirements are retained. No allocation or device wait is
     * introduced into inference: the completed setup transaction calls this
     * immediately after its upload pipeline has finalized.
     *
     * @param weights Frozen bindings consumed by the completed preparation.
     * @param plan Exact rank/device-scoped preparation request set.
     * @param engines Authoritative prepared engine identities and lifetimes.
     * @param metadata Monotonic host-consumer policy for each physical source.
     * @return Logical source bytes retired; duplicate bindings count once.
     * @throws std::logic_error If a candidate lacks exact complete ownership.
     */
    std::size_t retirePreparedOverlaySources(
        const FrozenModelWeightSet &weights,
        const MoEExpertOverlayPreparationPlan &plan,
        const ExpertGemmRegistry &engines,
        const WeightMetadataRegistry &metadata);
}
