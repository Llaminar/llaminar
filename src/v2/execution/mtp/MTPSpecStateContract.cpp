#include "MTPSpecStateContract.h"

#include <sstream>
#include <utility>

namespace llaminar2
{
    namespace
    {
        MTPSpecStepPlanBatch stepPlanFailure(
            const MTPSpecDecodeMetadataShape &shape,
            int request_count,
            std::string reason)
        {
            MTPSpecStepPlanBatch result;
            result.shape = shape;
            result.request_count = request_count;
            result.ok = false;
            result.error = std::move(reason);
            return result;
        }

        bool sameShape(
            const MTPSpecDecodeMetadataShape &a,
            const MTPSpecDecodeMetadataShape &b)
        {
            return a.max_requests == b.max_requests &&
                   a.max_draft_tokens == b.max_draft_tokens;
        }

        bool hasSize(const std::vector<int32_t> &values, int required)
        {
            return static_cast<int>(values.size()) >= required;
        }

        bool isInvalidIndex(int32_t value)
        {
            return value == kMTPSpecDecodeInvalidToken;
        }
    } // namespace

    MTPSpecStepPlanBatch buildMTPSpecStepPlans(
        const MTPSpecDecodeMetadataBatch &batch,
        const MTPSpecDecodeStatePublicationPlan &publication_plan)
    {
        const MTPSpecDecodeMetadataShape &shape = batch.shape;
        if (!batch.ok)
        {
            return stepPlanFailure(
                shape,
                batch.request_count,
                std::string("cannot build MTP spec-step plans from invalid metadata batch: ") +
                    batch.error);
        }
        if (!publication_plan.ok)
        {
            return stepPlanFailure(
                publication_plan.shape,
                publication_plan.request_count,
                std::string("cannot build MTP spec-step plans from invalid publication plan: ") +
                    publication_plan.error);
        }
        if (!shape.valid())
            return stepPlanFailure(shape, batch.request_count, "invalid MTP spec-step shape");
        if (!sameShape(shape, publication_plan.shape))
            return stepPlanFailure(shape, batch.request_count, "metadata and publication shapes differ");
        if (batch.request_count != publication_plan.request_count)
            return stepPlanFailure(shape, batch.request_count, "metadata and publication request counts differ");
        if (batch.request_count < 0 || batch.request_count > shape.max_requests)
            return stepPlanFailure(shape, batch.request_count, "request count exceeds metadata shape");

        const int requests = shape.max_requests;
        const int target_slots = requests * shape.maxTargetQueryLen();
        if (!hasSize(batch.draft_counts, requests) ||
            !hasSize(batch.target_query_lens, requests) ||
            !hasSize(batch.valid_sampled_counts, requests) ||
            !hasSize(batch.committed_output_counts, requests) ||
            !hasSize(batch.rejected_token_counts, requests) ||
            !hasSize(batch.token_indices_to_sample, requests) ||
            !hasSize(batch.next_condition_tokens, requests) ||
            !hasSize(batch.all_drafts_accepted_flags, requests) ||
            !hasSize(batch.stopped_flags, requests) ||
            !hasSize(batch.accepted_state_counts, requests) ||
            !hasSize(batch.accepted_state_slot_indices, requests) ||
            !hasSize(batch.correction_replay_start_indices, requests) ||
            !hasSize(batch.correction_replay_counts, requests) ||
            !hasSize(batch.bonus_ready_token_rows, requests) ||
            !hasSize(batch.bonus_ready_token_indices, requests) ||
            !hasSize(batch.bonus_ready_state_slot_indices, requests) ||
            !hasSize(publication_plan.base_cached_tokens, requests) ||
            !hasSize(publication_plan.target_cached_tokens, requests) ||
            !hasSize(publication_plan.accepted_state_counts, requests) ||
            !hasSize(publication_plan.accepted_state_slot_indices, requests) ||
            !hasSize(publication_plan.correction_replay_start_indices, requests) ||
            !hasSize(publication_plan.correction_replay_counts, requests) ||
            !hasSize(publication_plan.bonus_ready_token_rows, requests) ||
            !hasSize(publication_plan.bonus_ready_token_indices, requests) ||
            !hasSize(publication_plan.bonus_ready_state_slot_indices, requests))
        {
            return stepPlanFailure(shape, batch.request_count, "metadata or publication arrays are undersized");
        }
        if (static_cast<int>(batch.transactions.size()) < batch.request_count)
            return stepPlanFailure(shape, batch.request_count, "metadata transaction vector is undersized");

        MTPSpecStepPlanBatch result;
        result.ok = true;
        result.shape = shape;
        result.request_count = batch.request_count;
        result.steps.reserve(static_cast<size_t>(batch.request_count));

        for (int i = 0; i < batch.request_count; ++i)
        {
            auto fail_request = [&](const char *reason)
            {
                std::ostringstream msg;
                msg << "request " << i << ": " << reason;
                return stepPlanFailure(shape, batch.request_count, msg.str());
            };

            const MTPSpecDecodeTransaction &tx =
                batch.transactions[static_cast<size_t>(i)];
            if (!tx.ok)
                return fail_request("transaction is invalid");

            const int draft_count = batch.draft_counts[static_cast<size_t>(i)];
            const int target_rows = batch.target_query_lens[static_cast<size_t>(i)];
            const int valid_sampled_count =
                batch.valid_sampled_counts[static_cast<size_t>(i)];
            const int committed_output_count =
                batch.committed_output_counts[static_cast<size_t>(i)];
            const int rejected_count =
                batch.rejected_token_counts[static_cast<size_t>(i)];
            const int accepted_count =
                publication_plan.accepted_state_counts[static_cast<size_t>(i)];
            const int base_cached =
                publication_plan.base_cached_tokens[static_cast<size_t>(i)];
            const int target_cached =
                publication_plan.target_cached_tokens[static_cast<size_t>(i)];
            const int accepted_slot =
                publication_plan.accepted_state_slot_indices[static_cast<size_t>(i)];
            const int correction_start =
                publication_plan.correction_replay_start_indices[static_cast<size_t>(i)];
            const int correction_count =
                publication_plan.correction_replay_counts[static_cast<size_t>(i)];
            const int bonus_row =
                publication_plan.bonus_ready_token_rows[static_cast<size_t>(i)];
            const int bonus_index =
                publication_plan.bonus_ready_token_indices[static_cast<size_t>(i)];
            const int bonus_slot =
                publication_plan.bonus_ready_state_slot_indices[static_cast<size_t>(i)];
            const bool all_drafts_accepted =
                batch.all_drafts_accepted_flags[static_cast<size_t>(i)] != 0;
            const bool stopped =
                batch.stopped_flags[static_cast<size_t>(i)] != 0;

            if (draft_count < 0 || draft_count > shape.max_draft_tokens)
                return fail_request("draft count is outside metadata shape");
            if (target_rows != draft_count + 1)
                return fail_request("target verifier row count must be draft_count + 1");
            if (tx.draft_count != draft_count || tx.target_query_len != target_rows)
                return fail_request("transaction shape does not match metadata");
            if (valid_sampled_count < 0 || valid_sampled_count > target_rows)
                return fail_request("valid sampled count is outside target rows");
            if (committed_output_count < 0 ||
                committed_output_count > valid_sampled_count)
            {
                return fail_request("committed output count is outside sampled prefix");
            }
            if (rejected_count != target_rows - valid_sampled_count)
                return fail_request("rejected count does not match verifier suffix");
            if (accepted_count != batch.accepted_state_counts[static_cast<size_t>(i)])
                return fail_request("publication accepted state count differs from metadata");
            if (accepted_count < 0 || accepted_count > draft_count)
                return fail_request("accepted state count is outside draft prefix");
            if (target_cached != base_cached + accepted_count)
                return fail_request("target cached-token count must equal base plus accepted state count");

            if (accepted_count == 0)
            {
                if (!isInvalidIndex(accepted_slot))
                    return fail_request("zero accepted state count must not publish a state slot");
            }
            else if (accepted_slot < 0 || accepted_slot >= target_slots)
            {
                return fail_request("accepted state slot index is outside batch target slots");
            }

            if (correction_count < 0)
                return fail_request("correction replay count is negative");
            if (correction_count == 0)
            {
                if (!isInvalidIndex(correction_start))
                    return fail_request("zero correction replay count must not name a replay start");
            }
            else
            {
                if (stopped)
                    return fail_request("stopped requests must not require correction replay");
                if (correction_start != accepted_count)
                    return fail_request("correction replay must start after accepted state prefix");
                if (accepted_count + correction_count > committed_output_count)
                    return fail_request("correction replay exceeds committed output prefix");
            }

            const bool has_bonus = !isInvalidIndex(bonus_row);
            if (has_bonus)
            {
                if (!all_drafts_accepted ||
                    stopped ||
                    accepted_count != draft_count ||
                    committed_output_count != draft_count ||
                    valid_sampled_count != target_rows ||
                    bonus_row != draft_count ||
                    bonus_index < 0 ||
                    bonus_slot < 0 ||
                    bonus_slot >= target_slots)
                {
                    return fail_request("bonus-ready state is inconsistent with an all-accepted verifier step");
                }
            }
            else if (!isInvalidIndex(bonus_index) || !isInvalidIndex(bonus_slot))
            {
                return fail_request("bonus-ready fields must be all invalid or all present");
            }

            MTPSpecStepPlan step;
            step.request_index = i;
            step.request_id = tx.request_id;
            step.draft_count = draft_count;
            step.target_rows = target_rows;
            step.valid_sampled_count = valid_sampled_count;
            step.committed_output_count = committed_output_count;
            step.accepted_count = accepted_count;
            step.rejected_count = rejected_count;
            step.base_cached_tokens = base_cached;
            step.target_cached_tokens = target_cached;
            step.accepted_state_slot_index = accepted_slot;
            step.correction_replay_start_index = correction_start;
            step.correction_replay_count = correction_count;
            step.bonus_ready_token_row = bonus_row;
            step.bonus_ready_token_index = bonus_index;
            step.bonus_ready_state_slot_index = bonus_slot;
            step.next_condition_token =
                batch.next_condition_tokens[static_cast<size_t>(i)];
            step.all_drafts_accepted = all_drafts_accepted;
            step.stopped = stopped;
            result.steps.push_back(step);
        }

        return result;
    }

    MTPSpecStepPlanBatch buildMTPSpecStepPlans(
        const MTPSpecDecodeMetadataBatch &batch,
        const std::vector<int32_t> &base_cached_tokens)
    {
        MTPSpecDecodeStateCommitPlan commit_plan =
            buildMTPSpecDecodeStateCommitPlan(batch);
        if (!commit_plan.ok)
        {
            return stepPlanFailure(
                batch.shape,
                batch.request_count,
                std::string("cannot build MTP spec-step commit plan: ") +
                    commit_plan.error);
        }

        MTPSpecDecodeStatePublicationPlan publication_plan =
            buildMTPSpecDecodeStatePublicationPlan(
                commit_plan,
                base_cached_tokens);
        if (!publication_plan.ok)
        {
            return stepPlanFailure(
                batch.shape,
                batch.request_count,
                std::string("cannot build MTP spec-step publication plan: ") +
                    publication_plan.error);
        }

        return buildMTPSpecStepPlans(batch, publication_plan);
    }

} // namespace llaminar2
