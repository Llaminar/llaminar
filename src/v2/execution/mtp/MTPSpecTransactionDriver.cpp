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

    MTPSpecTransactionBatchPlan buildMTPSpecTransactionBatchPlanFromDeviceRejectionOutcomes(
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

            const int draft_count =
                static_cast<int>(request.draft_tokens.size());
            MTPSpecDecodeAcceptedOutcome accepted;
            accepted.request_id = request_ids[i];
            accepted.vocab_size = vocab_size;
            accepted.draft_count = draft_count;
            accepted.committed_output_tokens = std::move(outcome.output_tokens);
            if (!outcome.stopped_on_output &&
                outcome.all_speculative_accepted &&
                outcome.ready_token >= 0)
            {
                accepted.bonus_ready_token = outcome.ready_token;
            }
            accepted.accepted_verifier_input_prefix =
                std::min(
                    draft_count,
                    std::max(0, outcome.accepted_speculative_prefix) + 1);
            accepted.target_verifier_state_commit_count =
                outcome.target_verifier_state_commit_count;
            accepted.all_drafts_accepted = outcome.all_speculative_accepted;
            accepted.stopped_on_output = outcome.stopped_on_output;
            accepted_outcomes.push_back(std::move(accepted));
        }

        return buildMTPSpecTransactionBatchPlanFromAcceptedOutcomes(
            shape,
            accepted_outcomes,
            base_cached_tokens);
    }

    MTPSpecTransactionBatchPlan buildMTPSpecTransactionBatchPlanFromDeviceRejectionMetadata(
        const MTPSpecDecodeMetadataShape &shape,
        const std::vector<int> &request_ids,
        int vocab_size,
        const std::vector<MTPDecodeCatchupGreedyRequest> &requests,
        const std::vector<int> &outcome_meta,
        int meta_stride,
        const std::vector<int32_t> &base_cached_tokens)
    {
        using namespace sampling_math;

        if (meta_stride < kSpeculativeBatchMetaCount)
        {
            return transactionPlanFailure(
                "MTP device rejection metadata transaction has an invalid meta stride");
        }
        if (requests.empty())
        {
            return transactionPlanFailure(
                "MTP device rejection metadata transaction has no requests");
        }
        if (outcome_meta.size() <
            requests.size() * static_cast<size_t>(meta_stride))
        {
            return transactionPlanFailure(
                "MTP device rejection metadata transaction meta vector is undersized");
        }

        std::vector<MTPDeviceRejectionBatchOutcome> outcomes;
        outcomes.reserve(requests.size());
        for (size_t i = 0; i < requests.size(); ++i)
        {
            const MTPDecodeCatchupGreedyRequest &request = requests[i];
            if (request.draft_tokens.empty())
            {
                return transactionPlanFailure(
                    "MTP device rejection metadata request has no draft tokens");
            }

            const int *meta =
                outcome_meta.data() + i * static_cast<size_t>(meta_stride);
            if (meta[kSpecBatchMetaOk] == 0)
            {
                return transactionPlanFailure(
                    "MTP device rejection metadata row is marked invalid");
            }

            const int output_count = meta[kSpecBatchMetaOutputCount];
            const int accepted_prefix =
                meta[kSpecBatchMetaAcceptedSpeculativePrefix];
            const int rejected_token =
                meta[kSpecBatchMetaRejectedVerifiedToken];
            if (output_count < 1 ||
                output_count > kSpeculativeBatchMaxOutputTokens)
            {
                return transactionPlanFailure(
                    "MTP device rejection metadata output count is invalid");
            }
            if (accepted_prefix < 0 ||
                accepted_prefix >
                    static_cast<int>(request.draft_tokens.size()) - 1)
            {
                return transactionPlanFailure(
                    "MTP device rejection metadata accepted prefix is invalid");
            }

            MTPDeviceRejectionBatchOutcome outcome;
            outcome.ok = true;
            outcome.output_token_count = output_count;

            int emitted = 0;
            outcome.output_tokens[static_cast<size_t>(emitted++)] =
                request.draft_tokens.front();
            for (int row = 0;
                 emitted < output_count && row < accepted_prefix;
                 ++row)
            {
                const size_t draft_index = static_cast<size_t>(row + 1);
                if (draft_index >= request.draft_tokens.size())
                {
                    return transactionPlanFailure(
                        "MTP device rejection metadata accepted draft index is out of range");
                }
                outcome.output_tokens[static_cast<size_t>(emitted++)] =
                    request.draft_tokens[draft_index];
            }
            if (emitted < output_count)
            {
                if (rejected_token < 0)
                {
                    return transactionPlanFailure(
                        "MTP device rejection metadata needs an unrecoverable output token");
                }
                outcome.output_tokens[static_cast<size_t>(emitted++)] =
                    rejected_token;
            }
            if (emitted != output_count)
            {
                return transactionPlanFailure(
                    "MTP device rejection metadata reconstructed token count mismatch");
            }

            outcome.accepted_speculative_prefix = accepted_prefix;
            outcome.target_verifier_state_commit_count =
                meta[kSpecBatchMetaTargetVerifierStateCommitCount];
            outcome.ready_token = meta[kSpecBatchMetaReadyToken];
            outcome.rejected_verified_token = rejected_token;
            outcome.stopped_on_output =
                meta[kSpecBatchMetaStoppedOnOutput] != 0;
            outcome.all_speculative_accepted =
                meta[kSpecBatchMetaAllSpeculativeAccepted] != 0;
            outcome.consumed_verifier_rows =
                meta[kSpecBatchMetaConsumedVerifierRows];
            outcome.sampled_terminal =
                meta[kSpecBatchMetaSampledTerminal] != 0;
            outcomes.push_back(std::move(outcome));
        }

        return buildMTPSpecTransactionBatchPlanFromDeviceRejectionOutcomes(
            shape,
            request_ids,
            vocab_size,
            requests,
            outcomes,
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
