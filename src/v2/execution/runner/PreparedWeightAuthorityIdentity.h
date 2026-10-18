/**
 * @file PreparedWeightAuthorityIdentity.h
 * @brief Exact metadata identities for retained expert placements and their memory owners.
 *
 * A reused model keeps the original PMA and prepared bytes. These identities
 * reject changes that would require different ownership or admission before any
 * new runner borrows those bytes; they are not checksums of weight/cache data.
 */
#pragma once
#include <memory>
#include <string>

namespace llaminar2
{
    struct MoERoutedExpertPlacementPlan;
    struct OrchestrationConfig;
    class ResolvedRankOrchestration;
    class AdmittedMoEPipelineMemory;
    struct MTPRuntimeConfig;

    /**
     * @brief Compare the physical MTP preparation contract for dense and MoE reuse.
     * @param retained Configuration that admitted and prepared the original weights.
     * @param requested New active policy borrowing the existing prepared model.
     * @return Whether retained depth/batch capacity and predictor/head placement agree.
     *
     * Active depth, verification and enabled state do not change this retained
     * physical contract. Predictor replication and terminal-head sharding do.
     */
    bool retainedMTPWeightAuthorityMatches(const MTPRuntimeConfig &retained, const MTPRuntimeConfig &requested);

    /**
     * @brief Serialize ordered placement and capacity-affecting policy.
     * @param plan Normalized model-wide or stage-local expert topology.
     * @param config Model-resolved request, including retained MTP geometry.
     * @return Collision-free metadata identity, or empty for a non-expert model.
     */
    std::string routedWeightAuthorityRequestIdentity(
        const std::shared_ptr<MoERoutedExpertPlacementPlan> &plan, const OrchestrationConfig &config);

    /** @return Ordered scopes and full prepared-weight/prefix-capacity intent of a pipeline request. */
    std::string pipelineRoutedWeightAuthorityRequestIdentity(const ResolvedRankOrchestration &topology);
    /** @return The same identity from the immutable original aggregate admission. */
    std::string pipelineRoutedWeightAuthorityRequestIdentity(const AdmittedMoEPipelineMemory &admission);
}
