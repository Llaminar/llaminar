/**
 * @file MTPStochasticVerificationOwner.cpp
 * @brief Production admission of full-vocabulary stochastic MTP owners.
 *
 * CPU global TP gathers complete vocabulary rows in its graph. GPU local TP
 * mirrors terminal heads and publishes a device-resident accepted outcome.
 * ExpertOverlay followers consume transactions, not vocabulary distributions.
 * None of these contracts is implied merely by a process count or backend name.
 */
#include "MTPStochasticVerificationOwner.h"

#include "execution/local_execution/orchestrators/IInferenceRunner.h"
#include "execution/mpi_orchestration/RankExecutionPlan.h"

namespace llaminar2
{
    std::string_view mtpStochasticVerificationOwnerFailure(
        const RankExecutionPlan &plan,
        const IInferenceRunner &runner,
        MTPRankParticipation participation)
    {
        // An expert service cannot become a sampler just because its backend
        // happens to implement the same reduction kernels as the continuation.
        if (participation == MTPRankParticipation::ExpertTransactionFollower)
            return "An ExpertOverlay transaction follower cannot own MTP verification";

        // Only full-model peers need a vocabulary collective. The installed
        // overlay coordinator already orders expert-only followers through
        // retained tickets; those followers never own a vocabulary shard.
        const bool distributed_model = plan.usesGlobalTP() ||
            participation == MTPRankParticipation::ModelCollectivePeer;
        const bool gathered_cpu = plan.usesGlobalTP() &&
            runner.primaryDeviceId().is_cpu() && runner.supportsMTPTokenCoordination();
        if (distributed_model && !gathered_cpu)
            return "MTP stochastic model peers require graph-gathered CPU GlobalTP logits";

        // Exempting expert followers from the vocabulary collective must not
        // exempt their GPU continuation from its actual resident outcome and
        // publication contract. These probes inspect installed resources only.
        if (participation == MTPRankParticipation::ExpertTransactionContinuation &&
            runner.primaryDeviceId().is_gpu() &&
            (!runner.supportsDeviceStochasticMTPVerification() ||
             !runner.supportsDeviceResidentMTPSpecStatePublication()))
            return "MTP ExpertOverlay continuation requires resident stochastic verification and state publication";

        // Each local GPU participant must own both its full distribution and
        // the accepted-state publisher; a full head alone is not sufficient.
        if (plan.usesLocalTP() &&
            (!runner.primaryDeviceId().is_gpu() ||
             !runner.usesMirroredMTPHeadForVerifier() ||
             !runner.supportsDeviceStochasticMTPVerification() ||
             !runner.supportsDeviceResidentMTPSpecStatePublication()))
            return "MTP stochastic LocalTP requires mirrored child-resident verifier outcomes and publication";

        return {};
    }
}
