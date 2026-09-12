/**
 * @file MoEOverlayCPUServiceMeasurement.h
 * @brief Bounded prepared-CPU-expert observations for startup economy readiness.
 *
 * Cold CPU experts need not receive any user route. This setup-only producer
 * measures an exact resident FFN without a router, histogram, KV cache, sampler
 * or placement writer. Its immutable observations supplement absent service
 * classes in the existing certifier; they never become live inference totals.
 */
#pragma once

#include "MoEOverlayEconomyCalibrationPlanner.h"
#include "planning/PhysicalMemoryAuthority.h"

namespace llaminar2
{
    /**
     * @brief Setup producer on the same rank-affined OpenMP execution as CPU serving.
     *
     * All calls finish before maintenance and request admission. One bank lease
     * spans each observation, and private floating execution bindings prevent
     * workspace mutation in already prepared serving engines. GPU observations
     * remain device-produced; this class cannot execute a GPU expert on CPU.
     */
    class MoEOverlayCPUServiceMeasurement final
    {
    public:
        /** @brief Complete model dimensions for a singleton full-expert observation. */
        struct Geometry
        {
            int d_model = 0;
            int intermediate = 0;
        };

        /**
         * @brief Maximum concurrently allocated setup payload on one CPU resource.
         * @param geometry Full expert dimensions, not a tensor-parallel shard.
         * @return Exact maximum of the serial phase scratch families.
         * @throws std::invalid_argument for nonpositive dimensions.
         */
        [[nodiscard]] static size_t allocationBytes(Geometry geometry);

        /**
         * @brief Measure missing CPU class/phase coordinates from the initial bank.
         * @param registry Local CPU/GPU residency endpoints in canonical ID order.
         * @param epoch Exact published startup epoch; no candidate bank is legal.
         * @param catalog Immutable exact gate/up/down format-equivalence classes.
         * @param topology Phases actually priced by the retained serving policy.
         * @param geometry Full expert FFN dimensions.
         * @param memory Sole rank-local physical admission/claim authority.
         * @return Complete local matrix with only measured CPU coordinates populated.
         * @throws std::exception for a missing bank, invalid source or failed FFN.
         *
         * Call on the production rank's affined setup thread, not an arbitrary
         * worker with a different OpenMP/NUMA policy. No new thread pool or CPU
         * worker budget is selected here. Live observations take precedence in
         * the existing certifier, and are never overwritten by these samples.
         */
        [[nodiscard]] static std::vector<MoEOverlayParticipantLayerServiceTotals> measure(
            const MoEOverlayParticipantResidencyRegistry &registry,
            uint64_t epoch,
            const MoEOverlayEconomyCalibrationLayerCatalog &catalog,
            const ExpertHistogramProductionTopology &topology,
            Geometry geometry,
            const std::shared_ptr<PhysicalMemoryAuthority> &memory);
    };
}
