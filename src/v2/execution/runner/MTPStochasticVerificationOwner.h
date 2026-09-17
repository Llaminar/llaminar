/**
 * @file MTPStochasticVerificationOwner.h
 * @brief Validate stochastic-verifier ownership independently of MPI size.
 *
 * An expert-only rank follows graph transactions; it does not own a vocabulary
 * shard or participate in sampling. This admission check consumes the installed
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
    };

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
