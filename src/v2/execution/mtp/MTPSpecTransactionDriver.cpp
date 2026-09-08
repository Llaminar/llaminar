#include "MTPSpecTransactionDriver.h"

#include <algorithm>
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

        MTPSpecTransactionBatchPlan transactionPlanFailure(std::string reason)
        {
            MTPSpecTransactionBatchPlan plan;
            plan.ok = false;
            plan.error = std::move(reason);
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

    MTPSpecTransactionBatchPlan buildDeviceRejectionOutcomePlan(
        const MTPSpecDecodeMetadataShape &shape,
        const std::vector<int> &request_ids,
        int vocab_size,
        const std::vector<MTPDecodeCatchupGreedyRequest> &requests,
        const std::vector<MTPDeviceRejectionBatchOutcome> &device_outcomes,
        const std::vector<int32_t> &base_cached_tokens)
    {
        if (!shape.valid())
            return transactionPlanFailure(
                "MTP device rejection transaction has invalid metadata shape");
        if (requests.empty())
            return transactionPlanFailure(
                "MTP device rejection transaction has no requests");
        if (request_ids.size() != requests.size())
            return transactionPlanFailure(
                "MTP device rejection transaction request-id vector mismatch");
        if (device_outcomes.size() != requests.size())
            return transactionPlanFailure(
                "MTP device rejection transaction outcome vector mismatch");
        if (base_cached_tokens.size() != requests.size())
            return transactionPlanFailure(
                "MTP device rejection transaction base-cache vector mismatch");
        if (static_cast<int>(requests.size()) > shape.max_requests)
            return transactionPlanFailure(
                "MTP device rejection transaction exceeds max_requests");

        std::vector<MTPSpecDecodeAcceptedOutcome> accepted_outcomes;
        accepted_outcomes.reserve(requests.size());
        for (size_t i = 0; i < requests.size(); ++i)
        {
            const MTPDecodeCatchupGreedyRequest &request = requests[i];
            if (request.draft_tokens.empty())
            {
                return transactionPlanFailure(
                    "MTP device rejection transaction request has no draft tokens");
            }
            if (static_cast<int>(request.draft_tokens.size()) >
                shape.max_draft_tokens)
            {
                return transactionPlanFailure(
                    "MTP device rejection transaction request exceeds max_draft_tokens");
            }

            /*
             * Device kernels already decide the stochastic rows.  Re-summarize
             * the compact result on the host so every backend gets the same
             * bounds checks before live state publication is planned.
             */
            MTPRejectionBatchOutcome outcome =
                summarizeDeviceMTPRejectionBatchOutcome(
                    request,
                    device_outcomes[i]);
            if (!outcome.ok)
            {
                return transactionPlanFailure(
                    std::string("MTP device rejection outcome ") +
                    std::to_string(i) + " failed: " + outcome.error);
            }

            const int declared_draft_capacity =
                static_cast<int>(request.draft_tokens.size());
            /*
             * A captured dynamic-depth verifier has one immutable maximum row
             * capacity, but its resident controller may select a smaller active
             * depth for a particular replay.  A full acceptance proves exactly
             * how many active rows were consumed: every comparison row accepted,
             * followed by one first-token state row.  Metadata must describe that
             * active transaction width, not the larger host diagnostic vector.
             *
             * Rejection and stop outcomes may terminate before the selected depth
             * is observable, so they retain the declared capacity and publish only
             * their accepted prefix.  This keeps graph capacity and transaction
             * semantics separate without introducing a host-side depth authority.
             */
            const int transaction_draft_count =
                outcome.all_speculative_accepted &&
                        !outcome.stopped_on_output
                    ? outcome.consumed_verifier_rows + 1
                    : declared_draft_capacity;
            if (transaction_draft_count <= 0 ||
                transaction_draft_count > declared_draft_capacity)
            {
                return transactionPlanFailure(
                    std::string("MTP device rejection outcome ") +
                    std::to_string(i) +
                    " selected an invalid active verifier width");
            }
            MTPSpecDecodeAcceptedOutcome accepted;
            accepted.request_id = request_ids[i];
            accepted.vocab_size = vocab_size;
            accepted.draft_count = transaction_draft_count;
            accepted.committed_output_tokens = std::move(outcome.output_tokens);
            if (!outcome.stopped_on_output &&
                outcome.all_speculative_accepted &&
                outcome.ready_token >= 0)
            {
                accepted.bonus_ready_token = outcome.ready_token;
            }
            if (outcome.commit_boundary_clipped)
            {
                accepted.commit_boundary_ready_token = outcome.ready_token;
            }
            accepted.accepted_verifier_input_prefix =
                std::min(
                    transaction_draft_count,
                    std::max(0, outcome.accepted_speculative_prefix) + 1);
            accepted.target_verifier_state_commit_count =
                outcome.target_verifier_state_commit_count;
            accepted.all_drafts_accepted = outcome.all_speculative_accepted;
            accepted.stopped_on_output = outcome.stopped_on_output;
            accepted.commit_boundary_clipped =
                outcome.commit_boundary_clipped;
            accepted_outcomes.push_back(std::move(accepted));
        }

        return planFromMetadata(
            buildMTPSpecDecodeMetadataBatchFromAcceptedOutcomes(
                shape,
                accepted_outcomes),
            base_cached_tokens);
    }

    MTPSpecTransactionBatchPlan buildMTPSpecTransactionBatchPlanFromDeviceRejectionOutcomes(
        const MTPSpecDecodeMetadataShape &shape,
        const std::vector<int> &request_ids,
        int vocab_size,
        const std::vector<MTPDecodeCatchupGreedyRequest> &requests,
        const std::vector<MTPDeviceRejectionBatchOutcome> &device_outcomes,
        const std::vector<int32_t> &base_cached_tokens)
    {
        return buildDeviceRejectionOutcomePlan(
            shape,
            request_ids,
            vocab_size,
            requests,
            device_outcomes,
            base_cached_tokens);
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

    namespace
    {
        MTPSpecTransactionBatchPlan buildGreedyCatchupPlan(
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
    } // namespace

    MTPSpecTransactionBatchPlan buildMTPSpecTransactionBatchPlanFromGreedyCatchups(
        const MTPSpecDecodeMetadataShape &shape,
        const std::vector<int> &request_ids,
        int vocab_size,
        const std::vector<MTPDecodeCatchupGreedyRequest> &requests,
        const std::vector<MTPDecodeCatchupGreedyResult> &results,
        const std::vector<int32_t> &base_cached_tokens)
    {
        return buildGreedyCatchupPlan(
            shape,
            request_ids,
            vocab_size,
            requests,
            results,
            base_cached_tokens);
    }

} // namespace llaminar2
