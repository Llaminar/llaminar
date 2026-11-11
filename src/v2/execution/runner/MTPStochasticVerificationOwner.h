/**
 * @file MTPStochasticVerificationOwner.h
 * @brief Validate stochastic-verifier ownership independently of MPI size.
 *
 * Expert-only and original pipeline stage ranks follow graph transactions;
 * they do not own a vocabulary shard or participate in sampling. This admission check consumes the installed
 * runner and rank plan without creating another execution-state authority.
 */
#pragma once

#include <string_view>

namespace llaminar2
{
    class IInferenceRunner;
    struct RankExecutionPlan;

    /** @brief This rank's installed participation in MTP execution. */
    enum class MTPRankParticipation
    {
        SingleRank,
        ModelCollectivePeer,
        ExpertTransactionContinuation,
        ExpertTransactionFollower,
        OriginalPipelineContinuation, ///< Real vocabulary-tail owner; stage ranks follow retained transactions.
        OriginalPipelineFollower, ///< Original layer-stage owner; never a vocabulary/sampler peer.
    };

    /** @return Whether this installed role participates in full-model depth/token collectives.
     * Original pipeline and expert followers are ordered by their retained
     * transaction services; MPI world membership alone does not make them
     * vocabulary peers or participants in a scalar depth broadcast/barrier.
     */
    [[nodiscard]] inline constexpr bool mtpRankRequiresModelCollectives(MTPRankParticipation participation) noexcept
    {
        return participation == MTPRankParticipation::ModelCollectivePeer;
    }

    /**
     * @brief Reject incomplete stochastic-distribution/publication ownership.
     * @param plan Frozen continuation/model execution plan, not an expert tier.
     * @param runner Installed inference runner whose resources implement MTP.
     * @param participation Role projected from the installed transaction owner.
     * @return Empty on valid ownership, otherwise a fatal admission diagnostic.
     *
     * This function never enables a capability, changes execution policy, reads
     * live GPU state or coordinates peers. The selected runner must still prove
     * its full resident verifier and accepted-state publication implementation.
     */
    [[nodiscard]] std::string_view mtpStochasticVerificationOwnerFailure(
        const RankExecutionPlan &plan,
        const IInferenceRunner &runner,
        MTPRankParticipation participation);
}
