#pragma once

#include "MTPDecodeCatchup.h"
#include "MTPSpecDecodeMetadata.h"
#include "MTPSpecStateContract.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Fully planned vLLM-style speculative decode transaction batch.
     *
     * This object is the CPU-visible contract between a verifier outcome and
     * live-state publication.  It intentionally owns all intermediate planning
     * artifacts so callers can inspect or upload the same metadata that was used
     * to produce the final per-request `MTPSpecStepPlan` list.
     */
    struct MTPSpecTransactionBatchPlan
    {
        bool ok = false;
        std::string error;

        MTPSpecDecodeMetadataShape shape;
        int request_count = 0;

        MTPSpecDecodeMetadataBatch metadata;
        MTPSpecDecodeStateCommitPlan commit_plan;
        MTPSpecDecodeStatePublicationPlan publication_plan;
        MTPSpecStepPlanBatch step_plans;
    };

    /**
     * @brief Build a publication-ready transaction plan from accepted outcomes.
     *
     * Device-resident stochastic verification may keep rejected draft tokens on
     * the accelerator and return only accepted counts plus emitted tokens.  This
     * helper converts those reduced outcomes into the same padded metadata and
     * publication plans used by the host-visible greedy path.
     */
    MTPSpecTransactionBatchPlan buildMTPSpecTransactionBatchPlanFromAcceptedOutcomes(
        const MTPSpecDecodeMetadataShape &shape,
        const std::vector<MTPSpecDecodeAcceptedOutcome> &outcomes,
        const std::vector<int32_t> &base_cached_tokens);

    /**
     * @brief Single-request compatibility wrapper for accepted outcomes.
     *
     * The runner still executes one user request at a time today.  Keeping that
     * path on top of the batched helper prevents the single-request code from
     * developing slightly different accepted-count semantics.
     */
    MTPSpecTransactionBatchPlan buildMTPSpecTransactionBatchPlanFromAcceptedOutcome(
        const MTPSpecDecodeMetadataShape &shape,
        const MTPSpecDecodeAcceptedOutcome &outcome,
        int32_t base_cached_tokens);

    /**
     * @brief Build a publication-ready transaction plan from greedy catch-up.
     *
     * This path is used when the host knows the draft tokens and target verifier
     * result.  It shares the final commit/publication checks with the accepted-
     * outcome path so greedy and stochastic verification cannot drift.
     */
    MTPSpecTransactionBatchPlan buildMTPSpecTransactionBatchPlanFromGreedyCatchup(
        const MTPSpecDecodeMetadataShape &shape,
        int request_id,
        int vocab_size,
        const MTPDecodeCatchupGreedyRequest &request,
        const MTPDecodeCatchupGreedyResult &result,
        int32_t base_cached_tokens);

} // namespace llaminar2
