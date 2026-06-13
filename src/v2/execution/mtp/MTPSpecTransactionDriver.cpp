#include "MTPSpecTransactionDriver.h"

#include <string>
#include <utility>

namespace llaminar2
{
    namespace
    {
        MTPSpecTransactionBatchPlan planFromMetadata(
            MTPSpecDecodeMetadataBatch metadata,
            const std::vector<int32_t> &base_cached_tokens)
        {
            MTPSpecTransactionBatchPlan plan;
            plan.shape = metadata.shape;
            plan.request_count = metadata.request_count;
            plan.metadata = std::move(metadata);

            if (!plan.metadata.ok)
            {
                plan.ok = false;
                plan.error = std::string("MTP spec transaction metadata failed: ") +
                             plan.metadata.error;
                return plan;
            }
            if (static_cast<int>(base_cached_tokens.size()) != plan.metadata.request_count)
            {
                plan.ok = false;
                plan.error = "MTP spec transaction base-cache vector does not match request count";
                return plan;
            }

            /*
             * Keep commit planning as a named artifact even though
             * buildMTPSpecStepPlans() can rebuild it internally.  Tests and
             * future scheduler code can then assert exactly which flattened
             * verifier slots will be published before any backend state mutates.
             */
            plan.commit_plan = buildMTPSpecDecodeStateCommitPlan(plan.metadata);
            if (!plan.commit_plan.ok)
            {
                plan.ok = false;
                plan.error = std::string("MTP spec transaction commit plan failed: ") +
                             plan.commit_plan.error;
                return plan;
            }

            plan.publication_plan =
                buildMTPSpecDecodeStatePublicationPlan(
                    plan.commit_plan,
                    base_cached_tokens);
            if (!plan.publication_plan.ok)
            {
                plan.ok = false;
                plan.error = std::string("MTP spec transaction publication plan failed: ") +
                             plan.publication_plan.error;
                return plan;
            }

            plan.step_plans =
                buildMTPSpecStepPlans(
                    plan.metadata,
                    plan.publication_plan);
            if (!plan.step_plans.ok)
            {
                plan.ok = false;
                plan.error = std::string("MTP spec transaction step plan failed: ") +
                             plan.step_plans.error;
                return plan;
            }

            plan.ok = true;
            return plan;
        }
    } // namespace

    MTPSpecTransactionBatchPlan buildMTPSpecTransactionBatchPlanFromAcceptedOutcomes(
        const MTPSpecDecodeMetadataShape &shape,
        const std::vector<MTPSpecDecodeAcceptedOutcome> &outcomes,
        const std::vector<int32_t> &base_cached_tokens)
    {
        return planFromMetadata(
            buildMTPSpecDecodeMetadataBatchFromAcceptedOutcomes(shape, outcomes),
            base_cached_tokens);
    }

    MTPSpecTransactionBatchPlan buildMTPSpecTransactionBatchPlanFromAcceptedOutcome(
        const MTPSpecDecodeMetadataShape &shape,
        const MTPSpecDecodeAcceptedOutcome &outcome,
        int32_t base_cached_tokens)
    {
        return buildMTPSpecTransactionBatchPlanFromAcceptedOutcomes(
            shape,
            std::vector<MTPSpecDecodeAcceptedOutcome>{outcome},
            std::vector<int32_t>{base_cached_tokens});
    }

    MTPSpecTransactionBatchPlan buildMTPSpecTransactionBatchPlanFromGreedyCatchup(
        const MTPSpecDecodeMetadataShape &shape,
        int request_id,
        int vocab_size,
        const MTPDecodeCatchupGreedyRequest &request,
        const MTPDecodeCatchupGreedyResult &result,
        int32_t base_cached_tokens)
    {
        return buildMTPSpecTransactionBatchPlanFromGreedyCatchups(
            shape,
            std::vector<int>{request_id},
            vocab_size,
            std::vector<MTPDecodeCatchupGreedyRequest>{request},
            std::vector<MTPDecodeCatchupGreedyResult>{result},
            std::vector<int32_t>{base_cached_tokens});
    }

    MTPSpecTransactionBatchPlan buildMTPSpecTransactionBatchPlanFromGreedyCatchups(
        const MTPSpecDecodeMetadataShape &shape,
        const std::vector<int> &request_ids,
        int vocab_size,
        const std::vector<MTPDecodeCatchupGreedyRequest> &requests,
        const std::vector<MTPDecodeCatchupGreedyResult> &results,
        const std::vector<int32_t> &base_cached_tokens)
    {
        return planFromMetadata(
            buildMTPSpecDecodeMetadataBatchFromGreedyCatchups(
                shape,
                request_ids,
                vocab_size,
                requests,
                results),
            base_cached_tokens);
    }

} // namespace llaminar2
