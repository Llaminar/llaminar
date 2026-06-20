#pragma once

#include "MTPDecodeCatchup.h"
#include "MTPRejectionSampler.h"
#include "MTPSpecDecodeMetadata.h"
#include "MTPSpecStateContract.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llaminar2
{
    /**
     * @brief Describes how a transaction plan is allowed to publish live state.
     *
     * Direct plans may be handed to the normal accepted-state publisher.
     * Replay-required plans are useful Phase 9.8 evidence: they prove a grouped
     * verifier outcome is well formed, but live KV/GDN/short-conv state still
     * has to be advanced by the decode-equivalent replay path.  Keeping this
     * bit in the plan prevents future code from quietly treating a grouped
     * outcome proof as an economical vLLM-style publication.
     */
    enum class MTPSpecTransactionPublicationContract
    {
        DirectAcceptedStatePublication,
        DecodeEquivalentReplayPublicationRequired,
    };

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
        MTPSpecTransactionPublicationContract publication_contract =
            MTPSpecTransactionPublicationContract::DirectAcceptedStatePublication;
        std::string publication_contract_reason = "direct_accepted_state_publication";

        bool requiresDecodeEquivalentReplayPublication() const
        {
            return publication_contract ==
                   MTPSpecTransactionPublicationContract::
                       DecodeEquivalentReplayPublicationRequired;
        }
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
     * @brief Build a transaction plan from device-reduced stochastic outcomes.
     *
     * GPU stochastic verification should reduce each request to an accepted
     * count, committed output tokens, and an optional bonus-ready token before
     * the host sees the result.  This helper validates those compact outcomes
     * against the original request shapes, converts them to
     * `MTPSpecDecodeAcceptedOutcome`, and then uses the same padded
     * metadata/publication planner as every other vLLM-style path.
     */
    MTPSpecTransactionBatchPlan buildMTPSpecTransactionBatchPlanFromDeviceRejectionOutcomes(
        const MTPSpecDecodeMetadataShape &shape,
        const std::vector<int> &request_ids,
        int vocab_size,
        const std::vector<MTPDecodeCatchupGreedyRequest> &requests,
        const std::vector<MTPDeviceRejectionBatchOutcome> &device_outcomes,
        const std::vector<int32_t> &base_cached_tokens);

    /**
     * @brief Build a grouped-outcome plan that explicitly requires replay publication.
     *
     * Use this for the Phase 9.8 middle lane: the device produced a compact
     * grouped verifier outcome, but the backend has not yet proven direct
     * accepted-state publication for that model/backend.  The returned metadata
     * and step plans are still valuable for validation and telemetry, while the
     * publication contract hard-fails direct-publisher entry points.
     */
    MTPSpecTransactionBatchPlan
    buildMTPSpecTransactionBatchPlanFromDeviceRejectionOutcomesForReplayPublication(
        const MTPSpecDecodeMetadataShape &shape,
        const std::vector<int> &request_ids,
        int vocab_size,
        const std::vector<MTPDecodeCatchupGreedyRequest> &requests,
        const std::vector<MTPDeviceRejectionBatchOutcome> &device_outcomes,
        const std::vector<int32_t> &base_cached_tokens);

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

    /**
     * @brief Build a greedy grouped-outcome plan that requires replay publication.
     *
     * This is the greedy mirror of the stochastic grouped-outcome middle lane:
     * compact verifier rows have produced a valid accepted-token plan, but the
     * backend has not proven direct publication of the corresponding live
     * state.  Direct-publisher helpers must reject the returned plan.
     */
    MTPSpecTransactionBatchPlan
    buildMTPSpecTransactionBatchPlanFromGreedyCatchupForReplayPublication(
        const MTPSpecDecodeMetadataShape &shape,
        int request_id,
        int vocab_size,
        const MTPDecodeCatchupGreedyRequest &request,
        const MTPDecodeCatchupGreedyResult &result,
        int32_t base_cached_tokens);

    /**
     * @brief Batched greedy catch-up transaction planner.
     *
     * This is the vLLM-style all-position greedy path after verifier rows have
     * been sampled.  It keeps request ids, base-cache counts, metadata, commit
     * rows, and step plans aligned in one object so production batching can
     * publish every request atomically.
     */
    MTPSpecTransactionBatchPlan buildMTPSpecTransactionBatchPlanFromGreedyCatchups(
        const MTPSpecDecodeMetadataShape &shape,
        const std::vector<int> &request_ids,
        int vocab_size,
        const std::vector<MTPDecodeCatchupGreedyRequest> &requests,
        const std::vector<MTPDecodeCatchupGreedyResult> &results,
        const std::vector<int32_t> &base_cached_tokens);

    /**
     * @brief Batched greedy grouped-outcome plan requiring replay publication.
     */
    MTPSpecTransactionBatchPlan
    buildMTPSpecTransactionBatchPlanFromGreedyCatchupsForReplayPublication(
        const MTPSpecDecodeMetadataShape &shape,
        const std::vector<int> &request_ids,
        int vocab_size,
        const std::vector<MTPDecodeCatchupGreedyRequest> &requests,
        const std::vector<MTPDecodeCatchupGreedyResult> &results,
        const std::vector<int32_t> &base_cached_tokens);

} // namespace llaminar2
