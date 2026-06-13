#include "MTPSpecDecodeMetadata.h"

#include "MTPDecodeCatchup.h"
#include "../../backends/IBackend.h"
#include "../local_execution/device/DeviceWorkspaceManager.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <optional>
#include <sstream>
#include <utility>

namespace llaminar2
{
    namespace
    {
        WorkspaceDescriptor int32Buffer(const char *name, int count)
        {
            return WorkspaceDescriptor{
                name,
                static_cast<size_t>(std::max(0, count)) * sizeof(int32_t),
                256,
                true};
        }

        MTPSpecDecodeMetadataBatch metadataFailure(
            const MTPSpecDecodeMetadataShape &shape,
            std::string reason)
        {
            MTPSpecDecodeMetadataBatch batch;
            batch.shape = shape;
            batch.ok = false;
            batch.error = std::move(reason);
            return batch;
        }

        MTPSpecDecodeVerifierInputPlan verifierInputPlanFailure(
            const MTPSpecDecodeMetadataShape &shape,
            std::string reason)
        {
            MTPSpecDecodeVerifierInputPlan plan;
            plan.shape = shape;
            plan.ok = false;
            plan.error = std::move(reason);
            return plan;
        }

        MTPSpecDecodeVerifierGraphForwardPlan verifierGraphForwardPlanFailure(
            const MTPSpecDecodeVerifierInputPlan &input_plan,
            std::string reason)
        {
            MTPSpecDecodeVerifierGraphForwardPlan plan;
            plan.shape = input_plan.shape;
            plan.request_count = input_plan.request_count;
            plan.ok = false;
            plan.error = std::move(reason);
            return plan;
        }

        MTPSpecDecodeStateCommitPlan statePlanFailure(
            const MTPSpecDecodeMetadataShape &shape,
            int request_count,
            std::string reason)
        {
            MTPSpecDecodeStateCommitPlan plan;
            plan.shape = shape;
            plan.request_count = request_count;
            plan.ok = false;
            plan.error = std::move(reason);
            return plan;
        }

        MTPSpecDecodeStatePublicationPlan publicationPlanFailure(
            const MTPSpecDecodeMetadataShape &shape,
            int request_count,
            std::string reason)
        {
            MTPSpecDecodeStatePublicationPlan plan;
            plan.shape = shape;
            plan.request_count = request_count;
            plan.ok = false;
            plan.error = std::move(reason);
            return plan;
        }

        void fillInvalid(std::vector<int32_t> &values, int count)
        {
            values.assign(
                static_cast<size_t>(std::max(0, count)),
                kMTPSpecDecodeInvalidToken);
        }

        void fillZero(std::vector<int32_t> &values, int count)
        {
            values.assign(static_cast<size_t>(std::max(0, count)), 0);
        }

        bool tokenInVocab(int32_t token, int vocab_size)
        {
            return token >= 0 && (vocab_size <= 0 || token < vocab_size);
        }
    } // namespace

    WorkspaceRequirements buildMTPSpecDecodeWorkspaceRequirements(
        const MTPSpecDecodeMetadataShape &shape)
    {
        WorkspaceRequirements reqs;
        if (!shape.valid())
            return reqs;

        const int requests = shape.max_requests;
        const int draft_slots = requests * shape.max_draft_tokens;
        const int target_slots = requests * shape.maxTargetQueryLen();
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::DRAFT_COUNTS, requests));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::TARGET_QUERY_LENS, requests));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::VALID_SAMPLED_COUNTS, requests));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::ACCEPTED_DRAFT_PREFIXES, requests));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::COMMITTED_OUTPUT_COUNTS, requests));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::REJECTED_TOKEN_COUNTS, requests));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::TOKEN_INDICES_TO_SAMPLE, requests));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::NEXT_CONDITION_TOKENS, requests));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::ALL_DRAFTS_ACCEPTED_FLAGS, requests));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::STOPPED_FLAGS, requests));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::QUERY_START_LOCS, requests + 1));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::STATE_INDICES, target_slots));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::ACCEPTED_STATE_COUNTS, requests));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::SPECULATIVE_STATE_SLOT_INDICES, target_slots));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::COMMITTED_STATE_ROWS, requests));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::COMMITTED_STATE_INDICES, requests));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::ACCEPTED_STATE_SLOT_INDICES, requests));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::BONUS_READY_TOKEN_ROWS, requests));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::BONUS_READY_TOKEN_INDICES, requests));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::BONUS_READY_STATE_SLOT_INDICES, requests));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::DRAFT_TOKENS, draft_slots));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::SAMPLED_TOKENS, target_slots));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::VERIFIER_LOGIT_ROWS, target_slots));
        return reqs;
    }

    MTPSpecDecodeStateCommitPlan buildMTPSpecDecodeStateCommitPlan(
        const MTPSpecDecodeMetadataBatch &batch)
    {
        const MTPSpecDecodeMetadataShape &shape = batch.shape;
        if (!batch.ok)
            return statePlanFailure(
                shape,
                batch.request_count,
                std::string("cannot build state commit plan from invalid batch: ") +
                    batch.error);
        if (!shape.valid())
            return statePlanFailure(shape, batch.request_count, "invalid MTP spec-decode metadata shape");
        if (batch.request_count < 0 || batch.request_count > shape.max_requests)
            return statePlanFailure(shape, batch.request_count, "request count exceeds metadata shape");

        const int requests = shape.max_requests;
        const int target_slots = requests * shape.maxTargetQueryLen();
        auto has_size = [](const std::vector<int32_t> &values, int required)
        {
            return static_cast<int>(values.size()) >= required;
        };
        if (!has_size(batch.draft_counts, requests) ||
            !has_size(batch.target_query_lens, requests) ||
            !has_size(batch.valid_sampled_counts, requests) ||
            !has_size(batch.committed_output_counts, requests) ||
            (!batch.target_verifier_state_commit_counts.empty() &&
             !has_size(batch.target_verifier_state_commit_counts, requests)) ||
            !has_size(batch.rejected_token_counts, requests) ||
            !has_size(batch.all_drafts_accepted_flags, requests) ||
            !has_size(batch.stopped_flags, requests) ||
            !has_size(batch.state_indices, target_slots) ||
            !has_size(batch.accepted_state_counts, requests) ||
            !has_size(batch.speculative_state_slot_indices, target_slots))
        {
            return statePlanFailure(shape, batch.request_count, "metadata batch arrays are undersized");
        }

        MTPSpecDecodeStateCommitPlan plan;
        plan.ok = true;
        plan.shape = shape;
        plan.request_count = batch.request_count;
        fillInvalid(plan.committed_state_rows, requests);
        fillInvalid(plan.committed_state_indices, requests);
        fillZero(plan.accepted_state_counts, requests);
        fillInvalid(plan.accepted_state_slot_indices, requests);
        fillInvalid(plan.correction_replay_start_indices, requests);
        fillZero(plan.correction_replay_counts, requests);
        fillInvalid(plan.bonus_ready_token_rows, requests);
        fillInvalid(plan.bonus_ready_token_indices, requests);
        fillInvalid(plan.bonus_ready_state_slot_indices, requests);

        for (int i = 0; i < batch.request_count; ++i)
        {
            const int draft_count = batch.draft_counts[static_cast<size_t>(i)];
            const int target_query_len = batch.target_query_lens[static_cast<size_t>(i)];
            const int valid_sampled_count = batch.valid_sampled_counts[static_cast<size_t>(i)];
            const int committed_count = batch.committed_output_counts[static_cast<size_t>(i)];
            const int state_commit_count =
                batch.target_verifier_state_commit_counts.empty()
                    ? committed_count
                    : batch.target_verifier_state_commit_counts[static_cast<size_t>(i)];
            const int rejected_count = batch.rejected_token_counts[static_cast<size_t>(i)];
            const bool all_drafts_accepted =
                batch.all_drafts_accepted_flags[static_cast<size_t>(i)] != 0;
            const bool stopped =
                batch.stopped_flags[static_cast<size_t>(i)] != 0;
            const int target_offset = i * shape.maxTargetQueryLen();
            const int accepted_state_count =
                batch.accepted_state_counts[static_cast<size_t>(i)];

            auto fail_request = [&](const char *reason)
            {
                std::ostringstream msg;
                msg << "request " << i << ": " << reason;
                return statePlanFailure(shape, batch.request_count, msg.str());
            };

            if (draft_count < 0 || draft_count > shape.max_draft_tokens)
                return fail_request("draft count is outside metadata shape");
            if (target_query_len != draft_count + 1 ||
                target_query_len <= 0 ||
                target_query_len > shape.maxTargetQueryLen())
            {
                return fail_request("target query length does not match draft count");
            }
            if (valid_sampled_count < 0 ||
                valid_sampled_count > target_query_len)
            {
                return fail_request("valid sampled count is outside target query");
            }
            if (committed_count < 0 ||
                committed_count > valid_sampled_count ||
                committed_count > target_query_len)
            {
                return fail_request("committed output count is outside valid sampled prefix");
            }
            if (state_commit_count < 0 ||
                state_commit_count > committed_count ||
                state_commit_count > draft_count)
            {
                return fail_request("target verifier state commit count is outside committed verifier-input prefix");
            }
            if (accepted_state_count != state_commit_count)
                return fail_request("accepted state count must match target verifier state commit count");
            if (rejected_count != target_query_len - valid_sampled_count)
                return fail_request("rejected token count does not match valid sampled prefix");

            plan.accepted_state_counts[static_cast<size_t>(i)] =
                accepted_state_count;

            const int correction_replay_count =
                std::max(0, committed_count - accepted_state_count);
            if (correction_replay_count > 0 && !stopped)
            {
                plan.correction_replay_start_indices[static_cast<size_t>(i)] =
                    accepted_state_count;
                plan.correction_replay_counts[static_cast<size_t>(i)] =
                    correction_replay_count;
            }

            const int extra_valid_sampled = valid_sampled_count - committed_count;
            if (extra_valid_sampled > 0)
            {
                if (extra_valid_sampled != 1 ||
                    !all_drafts_accepted ||
                    stopped ||
                    committed_count != draft_count)
                {
                    return fail_request(
                        "valid sampled suffix beyond committed outputs is only allowed for an all-accepted bonus ready token");
                }
                const int bonus_row = valid_sampled_count - 1;
                const int32_t bonus_slot_index =
                    batch.speculative_state_slot_indices[static_cast<size_t>(target_offset + bonus_row)];
                const int32_t bonus_index =
                    batch.state_indices[static_cast<size_t>(target_offset + bonus_row)];
                if (bonus_index < 0 || bonus_slot_index < 0)
                    return fail_request("bonus ready-token row has no state index");
                plan.bonus_ready_token_rows[static_cast<size_t>(i)] = bonus_row;
                plan.bonus_ready_token_indices[static_cast<size_t>(i)] =
                    bonus_index;
                plan.bonus_ready_state_slot_indices[static_cast<size_t>(i)] =
                    bonus_slot_index;
            }

            if (state_commit_count > 0)
            {
                const int committed_row = state_commit_count - 1;
                const int32_t accepted_slot_index =
                    batch.speculative_state_slot_indices[static_cast<size_t>(target_offset + committed_row)];
                const int32_t committed_index =
                    batch.state_indices[static_cast<size_t>(target_offset + committed_row)];
                if (committed_index < 0 || accepted_slot_index < 0)
                    return fail_request("committed state row has no state index");
                plan.committed_state_rows[static_cast<size_t>(i)] =
                    committed_row;
                plan.committed_state_indices[static_cast<size_t>(i)] =
                    committed_index;
                plan.accepted_state_slot_indices[static_cast<size_t>(i)] =
                    accepted_slot_index;
            }
        }

        return plan;
    }

    MTPSpecDecodeStatePublicationPlan buildMTPSpecDecodeStatePublicationPlan(
        const MTPSpecDecodeStateCommitPlan &commit_plan,
        const std::vector<int32_t> &base_cached_tokens)
    {
        const MTPSpecDecodeMetadataShape &shape = commit_plan.shape;
        if (!commit_plan.ok)
        {
            return publicationPlanFailure(
                shape,
                commit_plan.request_count,
                std::string("cannot build publication plan from invalid commit plan: ") +
                    commit_plan.error);
        }
        if (!shape.valid())
            return publicationPlanFailure(shape, commit_plan.request_count, "invalid MTP spec-decode metadata shape");
        if (commit_plan.request_count < 0 ||
            commit_plan.request_count > shape.max_requests)
        {
            return publicationPlanFailure(shape, commit_plan.request_count, "request count exceeds metadata shape");
        }
        if (static_cast<int>(base_cached_tokens.size()) != commit_plan.request_count)
        {
            return publicationPlanFailure(
                shape,
                commit_plan.request_count,
                "base cached-token vector does not match request count");
        }

        auto has_size = [](const std::vector<int32_t> &values, int required)
        {
            return static_cast<int>(values.size()) >= required;
        };
        const int requests = shape.max_requests;
        const int target_slots = requests * shape.maxTargetQueryLen();
        if (!has_size(commit_plan.accepted_state_counts, requests) ||
            !has_size(commit_plan.accepted_state_slot_indices, requests) ||
            !has_size(commit_plan.correction_replay_start_indices, requests) ||
            !has_size(commit_plan.correction_replay_counts, requests) ||
            !has_size(commit_plan.bonus_ready_token_rows, requests) ||
            !has_size(commit_plan.bonus_ready_token_indices, requests) ||
            !has_size(commit_plan.bonus_ready_state_slot_indices, requests))
        {
            return publicationPlanFailure(
                shape,
                commit_plan.request_count,
                "commit plan arrays are undersized");
        }

        MTPSpecDecodeStatePublicationPlan plan;
        plan.ok = true;
        plan.shape = shape;
        plan.request_count = commit_plan.request_count;
        fillZero(plan.base_cached_tokens, requests);
        fillZero(plan.target_cached_tokens, requests);
        fillZero(plan.accepted_state_counts, requests);
        fillInvalid(plan.accepted_state_slot_indices, requests);
        fillInvalid(plan.correction_replay_start_indices, requests);
        fillZero(plan.correction_replay_counts, requests);
        fillInvalid(plan.bonus_ready_token_rows, requests);
        fillInvalid(plan.bonus_ready_token_indices, requests);
        fillInvalid(plan.bonus_ready_state_slot_indices, requests);

        for (int i = 0; i < commit_plan.request_count; ++i)
        {
            const int32_t base_cached = base_cached_tokens[static_cast<size_t>(i)];
            const int32_t accepted_count =
                commit_plan.accepted_state_counts[static_cast<size_t>(i)];
            if (base_cached < 0)
            {
                return publicationPlanFailure(
                    shape,
                    commit_plan.request_count,
                    "base cached-token count is negative");
            }
            if (accepted_count < 0 ||
                accepted_count > shape.max_draft_tokens)
            {
                return publicationPlanFailure(
                    shape,
                    commit_plan.request_count,
                    "accepted state count is outside metadata shape");
            }

            const int32_t accepted_slot =
                commit_plan.accepted_state_slot_indices[static_cast<size_t>(i)];
            if (accepted_count == 0)
            {
                if (accepted_slot != kMTPSpecDecodeInvalidToken)
                {
                    return publicationPlanFailure(
                        shape,
                        commit_plan.request_count,
                        "zero accepted state count must not name an accepted state slot");
                }
            }
            else if (accepted_slot < 0 ||
                     accepted_slot >= target_slots)
            {
                return publicationPlanFailure(
                    shape,
                    commit_plan.request_count,
                    "accepted state slot index is outside batch target slots");
            }

            const int32_t correction_start =
                commit_plan.correction_replay_start_indices[static_cast<size_t>(i)];
            const int32_t correction_count =
                commit_plan.correction_replay_counts[static_cast<size_t>(i)];
            if (correction_count < 0)
            {
                return publicationPlanFailure(
                    shape,
                    commit_plan.request_count,
                    "correction replay count is negative");
            }
            if (correction_count == 0 &&
                correction_start != kMTPSpecDecodeInvalidToken)
            {
                return publicationPlanFailure(
                    shape,
                    commit_plan.request_count,
                    "zero correction replay count must not name a replay start");
            }
            if (correction_count > 0 &&
                correction_start != accepted_count)
            {
                return publicationPlanFailure(
                    shape,
                    commit_plan.request_count,
                    "correction replay must start after the accepted state prefix");
            }

            const int32_t bonus_row =
                commit_plan.bonus_ready_token_rows[static_cast<size_t>(i)];
            const int32_t bonus_index =
                commit_plan.bonus_ready_token_indices[static_cast<size_t>(i)];
            const int32_t bonus_slot =
                commit_plan.bonus_ready_state_slot_indices[static_cast<size_t>(i)];
            if (bonus_row == kMTPSpecDecodeInvalidToken)
            {
                if (bonus_index != kMTPSpecDecodeInvalidToken ||
                    bonus_slot != kMTPSpecDecodeInvalidToken)
                {
                    return publicationPlanFailure(
                        shape,
                        commit_plan.request_count,
                        "bonus ready-token fields must be all invalid or all present");
                }
            }
            else if (bonus_row < 0 ||
                     bonus_row >= shape.maxTargetQueryLen() ||
                     bonus_index < 0 ||
                     bonus_slot < 0 ||
                     bonus_slot >= target_slots)
            {
                return publicationPlanFailure(
                    shape,
                    commit_plan.request_count,
                    "bonus ready-token state is outside batch target slots");
            }

            plan.base_cached_tokens[static_cast<size_t>(i)] = base_cached;
            plan.target_cached_tokens[static_cast<size_t>(i)] =
                base_cached + accepted_count;
            plan.accepted_state_counts[static_cast<size_t>(i)] = accepted_count;
            plan.accepted_state_slot_indices[static_cast<size_t>(i)] =
                accepted_slot;
            plan.correction_replay_start_indices[static_cast<size_t>(i)] =
                correction_start;
            plan.correction_replay_counts[static_cast<size_t>(i)] =
                correction_count;
            plan.bonus_ready_token_rows[static_cast<size_t>(i)] =
                commit_plan.bonus_ready_token_rows[static_cast<size_t>(i)];
            plan.bonus_ready_token_indices[static_cast<size_t>(i)] =
                commit_plan.bonus_ready_token_indices[static_cast<size_t>(i)];
            plan.bonus_ready_state_slot_indices[static_cast<size_t>(i)] =
                commit_plan.bonus_ready_state_slot_indices[static_cast<size_t>(i)];
        }

        return plan;
    }

    MTPSpecDecodeVerifierInputPlan buildMTPSpecDecodeVerifierInputPlan(
        const MTPSpecDecodeMetadataShape &shape,
        const std::vector<MTPSpecDecodeVerifierDraftRequest> &requests)
    {
        if (!shape.valid())
            return verifierInputPlanFailure(shape, "invalid MTP verifier input metadata shape");
        if (requests.empty())
            return verifierInputPlanFailure(shape, "MTP verifier input plan has no requests");
        if (static_cast<int>(requests.size()) > shape.max_requests)
            return verifierInputPlanFailure(shape, "MTP verifier input plan exceeds max_requests");

        MTPSpecDecodeVerifierInputPlan plan;
        plan.ok = true;
        plan.shape = shape;
        plan.request_count = static_cast<int>(requests.size());
        fillZero(plan.query_start_locs, shape.max_requests + 1);
        fillInvalid(plan.bonus_logit_rows, shape.max_requests);

        int query_cursor = 0;
        for (size_t request_index = 0; request_index < requests.size(); ++request_index)
        {
            const MTPSpecDecodeVerifierDraftRequest &request = requests[request_index];
            const int draft_count = static_cast<int>(request.draft_tokens.size());
            if (draft_count <= 0)
                return verifierInputPlanFailure(shape, "MTP verifier input request has no draft tokens");
            if (draft_count > shape.max_draft_tokens)
                return verifierInputPlanFailure(shape, "MTP verifier input request exceeds max_draft_tokens");

            plan.query_start_locs[request_index] = query_cursor;

            // Current Llaminar verifier rows are compact and contiguous for a
            // request: row 0 verifies the first sidecar draft, and the final
            // row becomes the bonus-ready distribution if every draft accepts.
            for (int row = 0; row < draft_count; ++row)
            {
                plan.verifier_input_tokens.push_back(
                    request.draft_tokens[static_cast<size_t>(row)]);
                plan.verifier_logit_rows.push_back(query_cursor + row);
            }
            plan.bonus_logit_rows[request_index] =
                query_cursor + draft_count - 1;
            query_cursor += draft_count;
        }

        plan.query_start_locs[requests.size()] = query_cursor;
        plan.total_verifier_input_tokens = query_cursor;
        plan.compact_logit_row_count =
            static_cast<int>(plan.verifier_logit_rows.size());
        return plan;
    }

    MTPSpecDecodeVerifierGraphForwardPlan buildMTPSpecDecodeVerifierGraphForwardPlan(
        const MTPSpecDecodeVerifierInputPlan &input_plan)
    {
        if (!input_plan.ok)
        {
            return verifierGraphForwardPlanFailure(
                input_plan,
                std::string("cannot materialize graph forward plan from invalid verifier input plan: ") +
                    input_plan.error);
        }
        if (!input_plan.shape.valid())
            return verifierGraphForwardPlanFailure(input_plan, "invalid verifier graph metadata shape");
        if (input_plan.request_count <= 0 ||
            input_plan.request_count > input_plan.shape.max_requests)
        {
            return verifierGraphForwardPlanFailure(input_plan, "request count exceeds verifier graph shape");
        }
        if (static_cast<int>(input_plan.query_start_locs.size()) <
            input_plan.request_count + 1)
        {
            return verifierGraphForwardPlanFailure(input_plan, "verifier query_start_locs are undersized");
        }
        if (input_plan.total_verifier_input_tokens < 0 ||
            static_cast<int>(input_plan.verifier_input_tokens.size()) <
                input_plan.total_verifier_input_tokens)
        {
            return verifierGraphForwardPlanFailure(input_plan, "verifier input tokens are undersized");
        }
        if (input_plan.compact_logit_row_count < 0 ||
            static_cast<int>(input_plan.verifier_logit_rows.size()) <
                input_plan.compact_logit_row_count)
        {
            return verifierGraphForwardPlanFailure(input_plan, "verifier logit rows are undersized");
        }

        MTPSpecDecodeVerifierGraphForwardPlan graph_plan;
        graph_plan.ok = true;
        graph_plan.shape = input_plan.shape;
        graph_plan.request_count = input_plan.request_count;
        graph_plan.token_batches.reserve(static_cast<size_t>(input_plan.request_count));
        graph_plan.sequence_lengths.reserve(static_cast<size_t>(input_plan.request_count));
        graph_plan.verifier_logit_rows.reserve(
            static_cast<size_t>(input_plan.compact_logit_row_count));
        fillInvalid(graph_plan.bonus_logit_rows, input_plan.shape.max_requests);

        int max_request_len = 0;
        for (int request = 0; request < input_plan.request_count; ++request)
        {
            const int start = input_plan.query_start_locs[static_cast<size_t>(request)];
            const int end = input_plan.query_start_locs[static_cast<size_t>(request + 1)];
            if (start < 0 || end < start ||
                end > input_plan.total_verifier_input_tokens)
            {
                return verifierGraphForwardPlanFailure(input_plan, "verifier query_start_locs are not monotonic");
            }
            const int len = end - start;
            if (len <= 0 || len > input_plan.shape.maxTargetQueryLen())
            {
                return verifierGraphForwardPlanFailure(input_plan, "verifier request length is outside shape");
            }

            std::vector<int> tokens;
            tokens.reserve(static_cast<size_t>(len));
            for (int row = start; row < end; ++row)
            {
                tokens.push_back(
                    static_cast<int>(
                        input_plan.verifier_input_tokens[static_cast<size_t>(row)]));
            }
            graph_plan.sequence_lengths.push_back(len);
            graph_plan.token_batches.push_back(std::move(tokens));
            max_request_len = std::max(max_request_len, len);
        }
        if (input_plan.query_start_locs[static_cast<size_t>(input_plan.request_count)] !=
            input_plan.total_verifier_input_tokens)
        {
            return verifierGraphForwardPlanFailure(input_plan, "verifier query_start_locs final value does not match input token count");
        }
        graph_plan.padded_seq_len = max_request_len;
        graph_plan.total_graph_tokens = input_plan.request_count * max_request_len;

        auto map_logical_row_to_graph_row = [&](int32_t logical_row)
            -> std::optional<int32_t>
        {
            if (logical_row < 0 ||
                logical_row >= input_plan.total_verifier_input_tokens)
            {
                return std::nullopt;
            }
            for (int request = 0; request < input_plan.request_count; ++request)
            {
                const int start = input_plan.query_start_locs[static_cast<size_t>(request)];
                const int end = input_plan.query_start_locs[static_cast<size_t>(request + 1)];
                if (logical_row >= start && logical_row < end)
                {
                    const int row_in_request = static_cast<int>(logical_row) - start;
                    return static_cast<int32_t>(
                        request * graph_plan.padded_seq_len + row_in_request);
                }
            }
            return std::nullopt;
        };

        for (int row = 0; row < input_plan.compact_logit_row_count; ++row)
        {
            const auto mapped = map_logical_row_to_graph_row(
                input_plan.verifier_logit_rows[static_cast<size_t>(row)]);
            if (!mapped)
                return verifierGraphForwardPlanFailure(input_plan, "verifier logit row is outside query ranges");
            graph_plan.verifier_logit_rows.push_back(*mapped);
        }

        const int bonus_count = std::min<int>(
            input_plan.request_count,
            static_cast<int>(input_plan.bonus_logit_rows.size()));
        for (int request = 0; request < bonus_count; ++request)
        {
            const int32_t logical_bonus =
                input_plan.bonus_logit_rows[static_cast<size_t>(request)];
            if (logical_bonus == kMTPSpecDecodeInvalidToken)
                continue;
            const auto mapped = map_logical_row_to_graph_row(logical_bonus);
            if (!mapped)
                return verifierGraphForwardPlanFailure(input_plan, "bonus logit row is outside query ranges");
            graph_plan.bonus_logit_rows[static_cast<size_t>(request)] = *mapped;
        }

        return graph_plan;
    }

    MTPSpecDecodeMetadataBatch buildMTPSpecDecodeMetadataBatch(
        const MTPSpecDecodeMetadataShape &shape,
        const std::vector<MTPSpecDecodeRequest> &requests,
        const std::vector<int32_t> &committed_output_counts,
        const std::vector<int32_t> &stopped_flags)
    {
        if (!shape.valid())
            return metadataFailure(shape, "invalid MTP spec-decode metadata shape");
        if (requests.empty())
            return metadataFailure(shape, "MTP spec-decode metadata batch has no requests");
        std::vector<int32_t> target_verifier_state_commit_counts;
        target_verifier_state_commit_counts.reserve(requests.size());
        for (const MTPSpecDecodeRequest &request : requests)
        {
            MTPSpecDecodeTransaction tx =
                buildMTPSpecDecodeTransaction(request);
            if (!tx.ok)
            {
                return metadataFailure(
                    shape,
                    std::string("request transaction metadata failed: ") +
                        tx.error);
            }
            target_verifier_state_commit_counts.push_back(
                static_cast<int32_t>(tx.accepted_speculative_prefix));
        }
        return buildMTPSpecDecodeMetadataBatchWithStateCommitCounts(
            shape,
            requests,
            committed_output_counts,
            target_verifier_state_commit_counts,
            stopped_flags);
    }

    MTPSpecDecodeMetadataBatch buildMTPSpecDecodeMetadataBatchWithStateCommitCounts(
        const MTPSpecDecodeMetadataShape &shape,
        const std::vector<MTPSpecDecodeRequest> &requests,
        const std::vector<int32_t> &committed_output_counts,
        const std::vector<int32_t> &target_verifier_state_commit_counts,
        const std::vector<int32_t> &stopped_flags)
    {
        if (!shape.valid())
            return metadataFailure(shape, "invalid MTP spec-decode metadata shape");
        if (requests.empty())
            return metadataFailure(shape, "MTP spec-decode metadata batch has no requests");
        if (static_cast<int>(requests.size()) > shape.max_requests)
            return metadataFailure(shape, "MTP spec-decode metadata batch exceeds max_requests");
        if (committed_output_counts.size() != requests.size())
            return metadataFailure(shape, "committed output count vector does not match request count");
        if (target_verifier_state_commit_counts.size() != requests.size())
            return metadataFailure(shape, "target verifier state commit count vector does not match request count");
        if (stopped_flags.size() != requests.size())
            return metadataFailure(shape, "stopped flag vector does not match request count");

        MTPSpecDecodeMetadataBatch batch;
        batch.ok = true;
        batch.shape = shape;
        batch.request_count = static_cast<int>(requests.size());
        batch.transactions.reserve(requests.size());

        const int requests_slots = shape.max_requests;
        const int draft_slots = shape.max_requests * shape.max_draft_tokens;
        const int target_slots = shape.max_requests * shape.maxTargetQueryLen();
        fillZero(batch.draft_counts, requests_slots);
        fillZero(batch.target_query_lens, requests_slots);
        fillZero(batch.valid_sampled_counts, requests_slots);
        fillZero(batch.accepted_draft_prefixes, requests_slots);
        fillZero(batch.committed_output_counts, requests_slots);
        fillZero(batch.target_verifier_state_commit_counts, requests_slots);
        fillZero(batch.rejected_token_counts, requests_slots);
        fillInvalid(batch.token_indices_to_sample, requests_slots);
        fillInvalid(batch.next_condition_tokens, requests_slots);
        fillZero(batch.all_drafts_accepted_flags, requests_slots);
        fillZero(batch.stopped_flags, requests_slots);
        fillZero(batch.query_start_locs, requests_slots + 1);
        fillInvalid(batch.state_indices, target_slots);
        fillZero(batch.accepted_state_counts, requests_slots);
        fillInvalid(batch.speculative_state_slot_indices, target_slots);
        fillInvalid(batch.committed_state_rows, requests_slots);
        fillInvalid(batch.committed_state_indices, requests_slots);
        fillInvalid(batch.accepted_state_slot_indices, requests_slots);
        fillInvalid(batch.correction_replay_start_indices, requests_slots);
        fillZero(batch.correction_replay_counts, requests_slots);
        fillInvalid(batch.bonus_ready_token_rows, requests_slots);
        fillInvalid(batch.bonus_ready_token_indices, requests_slots);
        fillInvalid(batch.bonus_ready_state_slot_indices, requests_slots);
        fillInvalid(batch.draft_tokens, draft_slots);
        fillInvalid(batch.sampled_tokens, target_slots);

        int query_cursor = 0;
        for (size_t request_index = 0; request_index < requests.size(); ++request_index)
        {
            const MTPSpecDecodeRequest &request = requests[request_index];
            if (static_cast<int>(request.draft_tokens.size()) > shape.max_draft_tokens)
            {
                return metadataFailure(shape, "request draft token count exceeds metadata shape");
            }
            if (static_cast<int>(request.sampled_tokens.size()) > shape.maxTargetQueryLen())
            {
                return metadataFailure(shape, "request sampled token count exceeds metadata shape");
            }

            MTPSpecDecodeTransaction tx =
                buildMTPSpecDecodeTransaction(request);
            if (!tx.ok)
            {
                return metadataFailure(
                    shape,
                    std::string("request transaction metadata failed: ") + tx.error);
            }
            const int32_t committed_count =
                committed_output_counts[request_index];
            if (committed_count < 0 ||
                committed_count > tx.valid_sampled_count)
            {
                return metadataFailure(
                    shape,
                    "committed output count is outside the valid sampled prefix");
            }
            const int32_t state_commit_count =
                target_verifier_state_commit_counts[request_index];
            if (state_commit_count < 0 ||
                state_commit_count > committed_count ||
                state_commit_count > tx.draft_count)
            {
                return metadataFailure(
                    shape,
                    "target verifier state commit count is outside committed verifier-input prefix");
            }

            const int i = static_cast<int>(request_index);
            const int draft_offset = i * shape.max_draft_tokens;
            const int target_offset = i * shape.maxTargetQueryLen();
            batch.draft_counts[request_index] = tx.draft_count;
            batch.target_query_lens[request_index] = tx.target_query_len;
            batch.valid_sampled_counts[request_index] = tx.valid_sampled_count;
            batch.accepted_draft_prefixes[request_index] = tx.accepted_speculative_prefix;
            batch.committed_output_counts[request_index] =
                committed_count;
            batch.target_verifier_state_commit_counts[request_index] =
                state_commit_count;
            batch.accepted_state_counts[request_index] =
                state_commit_count;
            batch.rejected_token_counts[request_index] = tx.rejected_token_count;
            batch.token_indices_to_sample[request_index] = tx.token_index_to_sample;
            batch.next_condition_tokens[request_index] = tx.next_condition_token;
            batch.all_drafts_accepted_flags[request_index] =
                tx.allDraftsAccepted() ? 1 : 0;
            batch.stopped_flags[request_index] =
                stopped_flags[request_index] != 0 ? 1 : 0;
            batch.query_start_locs[request_index] = query_cursor;

            for (int row = 0; row < tx.target_query_len; ++row)
            {
                batch.state_indices[static_cast<size_t>(target_offset + row)] =
                    query_cursor + row;
                batch.speculative_state_slot_indices[static_cast<size_t>(target_offset + row)] =
                    query_cursor + row;
            }
            query_cursor += tx.target_query_len;

            for (size_t j = 0; j < request.draft_tokens.size(); ++j)
            {
                batch.draft_tokens[static_cast<size_t>(draft_offset) + j] =
                    request.draft_tokens[j];
            }
            for (size_t j = 0; j < request.sampled_tokens.size(); ++j)
            {
                batch.sampled_tokens[static_cast<size_t>(target_offset) + j] =
                    request.sampled_tokens[j];
            }

            batch.transactions.push_back(std::move(tx));
        }

        batch.query_start_locs[requests.size()] = query_cursor;
        batch.total_target_query_tokens = query_cursor;

        MTPSpecDecodeStateCommitPlan commit_plan =
            buildMTPSpecDecodeStateCommitPlan(batch);
        if (!commit_plan.ok)
            return metadataFailure(shape, commit_plan.error);
        batch.committed_state_rows =
            std::move(commit_plan.committed_state_rows);
        batch.committed_state_indices =
            std::move(commit_plan.committed_state_indices);
        batch.accepted_state_counts =
            std::move(commit_plan.accepted_state_counts);
        batch.accepted_state_slot_indices =
            std::move(commit_plan.accepted_state_slot_indices);
        batch.correction_replay_start_indices =
            std::move(commit_plan.correction_replay_start_indices);
        batch.correction_replay_counts =
            std::move(commit_plan.correction_replay_counts);
        batch.bonus_ready_token_rows =
            std::move(commit_plan.bonus_ready_token_rows);
        batch.bonus_ready_token_indices =
            std::move(commit_plan.bonus_ready_token_indices);
        batch.bonus_ready_state_slot_indices =
            std::move(commit_plan.bonus_ready_state_slot_indices);
        return batch;
    }

    MTPSpecDecodeMetadataBatch buildMTPSpecDecodeMetadataBatchFromGreedyCatchup(
        const MTPSpecDecodeMetadataShape &shape,
        int request_id,
        int vocab_size,
        const MTPDecodeCatchupGreedyRequest &request,
        const MTPDecodeCatchupGreedyResult &result)
    {
        return buildMTPSpecDecodeMetadataBatchFromGreedyCatchups(
            shape,
            std::vector<int>{request_id},
            vocab_size,
            std::vector<MTPDecodeCatchupGreedyRequest>{request},
            std::vector<MTPDecodeCatchupGreedyResult>{result});
    }

    MTPSpecDecodeMetadataBatch buildMTPSpecDecodeMetadataBatchFromGreedyCatchups(
        const MTPSpecDecodeMetadataShape &shape,
        const std::vector<int> &request_ids,
        int vocab_size,
        const std::vector<MTPDecodeCatchupGreedyRequest> &requests,
        const std::vector<MTPDecodeCatchupGreedyResult> &results)
    {
        if (request_ids.size() != requests.size())
            return metadataFailure(shape, "MTP spec-decode request-id vector does not match request count");
        if (results.size() != requests.size())
            return metadataFailure(shape, "MTP spec-decode catch-up result vector does not match request count");

        std::vector<MTPSpecDecodeRequest> spec_requests;
        std::vector<int32_t> committed_output_counts;
        std::vector<int32_t> target_verifier_state_commit_counts;
        std::vector<int32_t> stopped_flags;
        std::vector<int> expected_accepted_verifier_prefixes;
        std::vector<bool> expected_all_drafts_accepted_flags;
        std::vector<int32_t> expected_next_condition_tokens;
        std::vector<int> expected_valid_sampled_counts;
        spec_requests.reserve(requests.size());
        committed_output_counts.reserve(requests.size());
        target_verifier_state_commit_counts.reserve(requests.size());
        stopped_flags.reserve(requests.size());
        expected_accepted_verifier_prefixes.reserve(requests.size());
        expected_all_drafts_accepted_flags.reserve(requests.size());
        expected_next_condition_tokens.reserve(requests.size());
        expected_valid_sampled_counts.reserve(requests.size());

        for (size_t i = 0; i < requests.size(); ++i)
        {
            const MTPDecodeCatchupGreedyRequest &request = requests[i];
            const MTPDecodeCatchupGreedyResult &result = results[i];
            auto fail_request = [&](const char *reason)
            {
                std::ostringstream msg;
                msg << "request " << i << ": " << reason;
                return metadataFailure(shape, msg.str());
            };

            if (!result.ok)
            {
                std::ostringstream msg;
                msg << "request " << i
                    << ": cannot build MTP spec-decode metadata from failed catch-up: "
                    << result.error;
                return metadataFailure(shape, msg.str());
            }
            if (request.draft_tokens.empty())
                return fail_request("MTP spec-decode catch-up request has no draft tokens");
            if (result.accepted_tokens.empty())
                return fail_request("MTP spec-decode catch-up result has no committed output tokens");
            if (result.accepted_speculative_prefix < 0)
                return fail_request("MTP spec-decode catch-up accepted prefix is negative");
            if (result.ready_token < 0 &&
                result.all_speculative_accepted &&
                !result.stopped_on_output)
            {
                return fail_request("MTP spec-decode catch-up accepted all drafts without a ready token");
            }

            const bool expected_all_drafts_accepted =
                result.all_speculative_accepted && !result.stopped_on_output;
            const std::optional<int32_t> bonus_ready_token =
                expected_all_drafts_accepted
                    ? std::optional<int32_t>{result.ready_token}
                    : std::optional<int32_t>{};
            const int expected_accepted_verifier_prefix =
                std::min<int>(
                    static_cast<int>(request.draft_tokens.size()),
                    result.accepted_speculative_prefix + 1);

            spec_requests.push_back(
                buildMTPSpecDecodeRequestFromVerifierOutput(
                    request_ids[i],
                    vocab_size,
                    request.draft_tokens,
                    result.accepted_tokens,
                    bonus_ready_token,
                    result.all_speculative_accepted,
                    result.stopped_on_output));
            committed_output_counts.push_back(
                static_cast<int32_t>(result.accepted_tokens.size()));
            target_verifier_state_commit_counts.push_back(
                static_cast<int32_t>(
                    result.target_verifier_state_commit_count >= 0
                        ? result.target_verifier_state_commit_count
                        : expected_accepted_verifier_prefix));
            stopped_flags.push_back(result.stopped_on_output ? 1 : 0);
            expected_accepted_verifier_prefixes.push_back(
                expected_accepted_verifier_prefix);
            expected_all_drafts_accepted_flags.push_back(
                expected_all_drafts_accepted);
            expected_valid_sampled_counts.push_back(
                static_cast<int>(result.accepted_tokens.size()) +
                (expected_all_drafts_accepted ? 1 : 0));
            expected_next_condition_tokens.push_back(
                expected_all_drafts_accepted
                    ? result.ready_token
                    : result.accepted_tokens.back());
        }

        MTPSpecDecodeMetadataBatch batch =
            buildMTPSpecDecodeMetadataBatchWithStateCommitCounts(
                shape,
                spec_requests,
                committed_output_counts,
                target_verifier_state_commit_counts,
                stopped_flags);
        if (!batch.ok)
            return batch;
        if (batch.transactions.size() != requests.size())
            return metadataFailure(shape, "MTP spec-decode metadata batch produced the wrong transaction count");

        for (size_t i = 0; i < requests.size(); ++i)
        {
            const MTPSpecDecodeTransaction &tx = batch.transactions[i];
            if (tx.accepted_speculative_prefix !=
                expected_accepted_verifier_prefixes[i])
            {
                return metadataFailure(
                    shape,
                    "MTP spec-decode accepted-prefix mismatch between catch-up result and transaction");
            }

            if (tx.allDraftsAccepted() != expected_all_drafts_accepted_flags[i])
            {
                return metadataFailure(
                    shape,
                    "MTP spec-decode all-drafts-accepted mismatch between catch-up result and transaction");
            }

            if (tx.valid_sampled_count != expected_valid_sampled_counts[i])
            {
                return metadataFailure(
                    shape,
                    "MTP spec-decode valid-sampled-count mismatch between catch-up result and transaction");
            }

            if (tx.next_condition_token != expected_next_condition_tokens[i])
            {
                return metadataFailure(
                    shape,
                    "MTP spec-decode next-condition-token mismatch between catch-up result and transaction");
            }
        }

        return batch;
    }

    MTPSpecDecodeMetadataBatch buildMTPSpecDecodeMetadataBatchFromAcceptedOutcome(
        const MTPSpecDecodeMetadataShape &shape,
        const MTPSpecDecodeAcceptedOutcome &outcome)
    {
        return buildMTPSpecDecodeMetadataBatchFromAcceptedOutcomes(
            shape,
            std::vector<MTPSpecDecodeAcceptedOutcome>{outcome});
    }

    MTPSpecDecodeMetadataBatch buildMTPSpecDecodeMetadataBatchFromAcceptedOutcomes(
        const MTPSpecDecodeMetadataShape &shape,
        const std::vector<MTPSpecDecodeAcceptedOutcome> &outcomes)
    {
        if (!shape.valid())
            return metadataFailure(shape, "invalid MTP spec-decode metadata shape");
        if (outcomes.empty())
            return metadataFailure(shape, "accepted outcome batch has no requests");
        if (static_cast<int>(outcomes.size()) > shape.max_requests)
            return metadataFailure(shape, "accepted outcome batch exceeds max_requests");

        MTPSpecDecodeMetadataBatch batch;
        batch.ok = true;
        batch.shape = shape;
        batch.request_count = static_cast<int>(outcomes.size());

        const int request_slots = shape.max_requests;
        const int draft_slots = shape.max_requests * shape.max_draft_tokens;
        const int target_slots = shape.max_requests * shape.maxTargetQueryLen();
        fillZero(batch.draft_counts, request_slots);
        fillZero(batch.target_query_lens, request_slots);
        fillZero(batch.valid_sampled_counts, request_slots);
        fillZero(batch.accepted_draft_prefixes, request_slots);
        fillZero(batch.committed_output_counts, request_slots);
        fillZero(batch.target_verifier_state_commit_counts, request_slots);
        fillZero(batch.rejected_token_counts, request_slots);
        fillInvalid(batch.token_indices_to_sample, request_slots);
        fillInvalid(batch.next_condition_tokens, request_slots);
        fillZero(batch.all_drafts_accepted_flags, request_slots);
        fillZero(batch.stopped_flags, request_slots);
        fillZero(batch.query_start_locs, request_slots + 1);
        fillInvalid(batch.state_indices, target_slots);
        fillZero(batch.accepted_state_counts, request_slots);
        fillInvalid(batch.speculative_state_slot_indices, target_slots);
        fillInvalid(batch.committed_state_rows, request_slots);
        fillInvalid(batch.committed_state_indices, request_slots);
        fillInvalid(batch.accepted_state_slot_indices, request_slots);
        fillInvalid(batch.correction_replay_start_indices, request_slots);
        fillZero(batch.correction_replay_counts, request_slots);
        fillInvalid(batch.bonus_ready_token_rows, request_slots);
        fillInvalid(batch.bonus_ready_token_indices, request_slots);
        fillInvalid(batch.bonus_ready_state_slot_indices, request_slots);
        fillInvalid(batch.draft_tokens, draft_slots);
        fillInvalid(batch.sampled_tokens, target_slots);

        int query_cursor = 0;
        batch.transactions.reserve(outcomes.size());
        for (size_t request_index = 0; request_index < outcomes.size(); ++request_index)
        {
            const MTPSpecDecodeAcceptedOutcome &outcome = outcomes[request_index];
            auto fail_request = [&](const char *reason)
            {
                std::ostringstream msg;
                msg << "request " << request_index << ": " << reason;
                return metadataFailure(shape, msg.str());
            };

            if (outcome.draft_count <= 0 ||
                outcome.draft_count > shape.max_draft_tokens)
            {
                return fail_request("accepted outcome draft count is outside metadata shape");
            }
            if (outcome.vocab_size < 0)
                return fail_request("accepted outcome has negative vocabulary size");
            if (outcome.committed_output_tokens.empty())
                return fail_request("accepted outcome has no committed output tokens");
            if (outcome.accepted_verifier_input_prefix < 0 ||
                outcome.accepted_verifier_input_prefix > outcome.draft_count)
            {
                return fail_request("accepted outcome verifier prefix is outside draft count");
            }

            const int target_query_len = outcome.draft_count + 1;
            const int committed_count =
                static_cast<int>(outcome.committed_output_tokens.size());
            if (committed_count > target_query_len)
                return fail_request("accepted outcome committed output count exceeds verifier rows");
            for (int32_t token : outcome.committed_output_tokens)
            {
                if (!tokenInVocab(token, outcome.vocab_size))
                    return fail_request("accepted outcome committed token is outside the vocabulary");
            }

            const bool has_bonus_ready =
                outcome.all_drafts_accepted &&
                !outcome.stopped_on_output &&
                outcome.bonus_ready_token.has_value();
            if (outcome.all_drafts_accepted)
            {
                if (outcome.accepted_verifier_input_prefix != outcome.draft_count)
                    return fail_request("all-accepted outcome must publish every verifier input row");
                if (!outcome.stopped_on_output &&
                    !outcome.bonus_ready_token.has_value())
                {
                    return fail_request("all-accepted outcome is missing its bonus ready token");
                }
            }
            if (outcome.bonus_ready_token.has_value() &&
                !tokenInVocab(*outcome.bonus_ready_token, outcome.vocab_size))
            {
                return fail_request("accepted outcome bonus ready token is outside the vocabulary");
            }

            const int valid_sampled_count =
                committed_count + (has_bonus_ready ? 1 : 0);
            if (valid_sampled_count <= 0 ||
                valid_sampled_count > target_query_len)
            {
                return fail_request("accepted outcome valid sampled count is outside verifier rows");
            }

            const int state_commit_count =
                outcome.target_verifier_state_commit_count >= 0
                    ? outcome.target_verifier_state_commit_count
                    : outcome.accepted_verifier_input_prefix;
            if (state_commit_count < 0 ||
                state_commit_count > outcome.accepted_verifier_input_prefix ||
                state_commit_count > committed_count ||
                state_commit_count > outcome.draft_count)
            {
                return fail_request("accepted outcome state commit count is outside committed prefix");
            }

            const int i = static_cast<int>(request_index);
            const int draft_offset = i * shape.max_draft_tokens;
            const int target_offset = i * shape.maxTargetQueryLen();
            batch.draft_counts[request_index] = outcome.draft_count;
            batch.target_query_lens[request_index] = target_query_len;
            batch.valid_sampled_counts[request_index] = valid_sampled_count;
            batch.accepted_draft_prefixes[request_index] =
                outcome.accepted_verifier_input_prefix;
            batch.committed_output_counts[request_index] = committed_count;
            batch.target_verifier_state_commit_counts[request_index] =
                state_commit_count;
            batch.accepted_state_counts[request_index] = state_commit_count;
            batch.rejected_token_counts[request_index] =
                target_query_len - valid_sampled_count;
            batch.token_indices_to_sample[request_index] =
                valid_sampled_count - 1;
            batch.next_condition_tokens[request_index] =
                has_bonus_ready ? *outcome.bonus_ready_token
                                : outcome.committed_output_tokens.back();
            batch.all_drafts_accepted_flags[request_index] =
                outcome.all_drafts_accepted ? 1 : 0;
            batch.stopped_flags[request_index] =
                outcome.stopped_on_output ? 1 : 0;
            batch.query_start_locs[request_index] = query_cursor;

            for (int row = 0; row < target_query_len; ++row)
            {
                batch.state_indices[static_cast<size_t>(target_offset + row)] =
                    query_cursor + row;
                batch.speculative_state_slot_indices[static_cast<size_t>(target_offset + row)] =
                    query_cursor + row;
            }

            /*
             * Draft tokens are diagnostic metadata here. Copy only verifier
             * rows whose token identity is known from committed output. Unknown
             * rejected device slots stay invalid; inventing token ids would
             * make the host view disagree with the device-resident contract.
             */
            for (int row = 0;
                 row < outcome.accepted_verifier_input_prefix &&
                 row < committed_count;
                 ++row)
            {
                batch.draft_tokens[static_cast<size_t>(draft_offset + row)] =
                    outcome.committed_output_tokens[static_cast<size_t>(row)];
            }
            for (int row = 0; row < committed_count; ++row)
            {
                batch.sampled_tokens[static_cast<size_t>(target_offset + row)] =
                    outcome.committed_output_tokens[static_cast<size_t>(row)];
            }
            if (has_bonus_ready)
            {
                batch.sampled_tokens[static_cast<size_t>(target_offset + committed_count)] =
                    *outcome.bonus_ready_token;
            }

            MTPSpecDecodeTransaction tx;
            tx.ok = true;
            tx.request_id = outcome.request_id;
            tx.draft_count = outcome.draft_count;
            tx.target_query_len = target_query_len;
            tx.valid_sampled_count = valid_sampled_count;
            tx.accepted_speculative_prefix =
                outcome.accepted_verifier_input_prefix;
            tx.rejected_token_count = target_query_len - valid_sampled_count;
            tx.token_index_to_sample = valid_sampled_count - 1;
            tx.next_condition_token =
                batch.next_condition_tokens[request_index];
            tx.committed_output_tokens = outcome.committed_output_tokens;
            tx.rejected_draft_tokens.assign(
                static_cast<size_t>(
                    std::max(
                        0,
                        outcome.draft_count -
                            outcome.accepted_verifier_input_prefix)),
                kMTPSpecDecodeInvalidToken);
            batch.transactions.push_back(std::move(tx));
            query_cursor += target_query_len;
        }

        batch.query_start_locs[outcomes.size()] = query_cursor;
        batch.total_target_query_tokens = query_cursor;

        MTPSpecDecodeStateCommitPlan commit_plan =
            buildMTPSpecDecodeStateCommitPlan(batch);
        if (!commit_plan.ok)
            return metadataFailure(shape, commit_plan.error);
        batch.committed_state_rows =
            std::move(commit_plan.committed_state_rows);
        batch.committed_state_indices =
            std::move(commit_plan.committed_state_indices);
        batch.accepted_state_counts =
            std::move(commit_plan.accepted_state_counts);
        batch.accepted_state_slot_indices =
            std::move(commit_plan.accepted_state_slot_indices);
        batch.correction_replay_start_indices =
            std::move(commit_plan.correction_replay_start_indices);
        batch.correction_replay_counts =
            std::move(commit_plan.correction_replay_counts);
        batch.bonus_ready_token_rows =
            std::move(commit_plan.bonus_ready_token_rows);
        batch.bonus_ready_token_indices =
            std::move(commit_plan.bonus_ready_token_indices);
        batch.bonus_ready_state_slot_indices =
            std::move(commit_plan.bonus_ready_state_slot_indices);
        return batch;
    }

    bool MTPSpecDecodeMetadataDevicePointers::complete() const
    {
        return draft_counts &&
               target_query_lens &&
               valid_sampled_counts &&
               accepted_draft_prefixes &&
               committed_output_counts &&
               rejected_token_counts &&
               token_indices_to_sample &&
               next_condition_tokens &&
               all_drafts_accepted_flags &&
               stopped_flags &&
               query_start_locs &&
               state_indices &&
               accepted_state_counts &&
               speculative_state_slot_indices &&
               accepted_state_slot_indices &&
               bonus_ready_token_rows &&
               bonus_ready_token_indices &&
               bonus_ready_state_slot_indices &&
               draft_tokens &&
               sampled_tokens &&
               verifier_logit_rows;
    }

    MTPSpecDecodeMetadataWorkspaceBinding::MTPSpecDecodeMetadataWorkspaceBinding(
        MTPSpecDecodeMetadataShape shape)
        : shape_(shape)
    {
    }

    void MTPSpecDecodeMetadataWorkspaceBinding::setShape(
        MTPSpecDecodeMetadataShape shape)
    {
        shape_ = shape;
        refreshDevicePointers();
    }

    MTPSpecDecodeMetadataShape MTPSpecDecodeMetadataWorkspaceBinding::effectiveShape(
        int m,
        int n,
        int k) const
    {
        MTPSpecDecodeMetadataShape effective = shape_;
        if (m > 0)
            effective.max_requests = std::max(effective.max_requests, m);
        if (n > 0)
            effective.max_draft_tokens =
                std::max(effective.max_draft_tokens, n);
        if (k > 0)
            effective.max_draft_tokens =
                std::max(effective.max_draft_tokens, k);
        return effective;
    }

    WorkspaceRequirements MTPSpecDecodeMetadataWorkspaceBinding::getWorkspaceRequirements(
        int m,
        int n,
        int k) const
    {
        return buildMTPSpecDecodeWorkspaceRequirements(effectiveShape(m, n, k));
    }

    void MTPSpecDecodeMetadataWorkspaceBinding::bindWorkspace(
        DeviceWorkspaceManager *workspace)
    {
        workspace_ = workspace;
        refreshDevicePointers();
    }

    bool MTPSpecDecodeMetadataWorkspaceBinding::hasWorkspace() const
    {
        return workspace_ != nullptr && device_pointers_.complete();
    }

    void MTPSpecDecodeMetadataWorkspaceBinding::refreshDevicePointers()
    {
        device_pointers_ = {};
        binding_error_.clear();
        if (!workspace_)
            return;

        const WorkspaceRequirements reqs =
            buildMTPSpecDecodeWorkspaceRequirements(shape_);
        if (reqs.buffers.empty())
        {
            binding_error_ = "invalid MTP spec-decode metadata workspace shape";
            return;
        }

        std::ostringstream error;
        auto bind_int32 = [&](const char *name, int32_t **out) -> bool
        {
            const WorkspaceDescriptor *desc = reqs.find(name);
            if (!desc)
            {
                error << "missing descriptor for " << name;
                return false;
            }
            if (!workspace_->hasBuffer(name))
            {
                error << "workspace missing buffer " << name;
                return false;
            }
            const size_t size = workspace_->getBufferSize(name);
            if (size < desc->size_bytes)
            {
                error << "workspace buffer " << name << " is too small: "
                      << size << " < " << desc->size_bytes;
                return false;
            }
            *out = static_cast<int32_t *>(workspace_->getBuffer(name));
            if (!*out)
            {
                error << "workspace buffer " << name << " returned null";
                return false;
            }
            return true;
        };

        auto bind_int32_optional = [&](const char *name, int32_t **out) -> bool
        {
            const WorkspaceDescriptor *desc = reqs.find(name);
            if (!desc)
            {
                *out = nullptr;
                return true;
            }
            if (!workspace_->hasBuffer(name))
            {
                *out = nullptr;
                return true;
            }
            const size_t size = workspace_->getBufferSize(name);
            if (size < desc->size_bytes)
            {
                error << "workspace buffer " << name << " is too small: "
                      << size << " < " << desc->size_bytes;
                return false;
            }
            *out = static_cast<int32_t *>(workspace_->getBuffer(name));
            if (!*out)
            {
                error << "workspace buffer " << name << " returned null";
                return false;
            }
            return true;
        };

        if (!bind_int32(
                MTPSpecDecodeWorkspaceBuffers::DRAFT_COUNTS,
                &device_pointers_.draft_counts) ||
            !bind_int32(
                MTPSpecDecodeWorkspaceBuffers::TARGET_QUERY_LENS,
                &device_pointers_.target_query_lens) ||
            !bind_int32(
                MTPSpecDecodeWorkspaceBuffers::VALID_SAMPLED_COUNTS,
                &device_pointers_.valid_sampled_counts) ||
            !bind_int32(
                MTPSpecDecodeWorkspaceBuffers::ACCEPTED_DRAFT_PREFIXES,
                &device_pointers_.accepted_draft_prefixes) ||
            !bind_int32(
                MTPSpecDecodeWorkspaceBuffers::COMMITTED_OUTPUT_COUNTS,
                &device_pointers_.committed_output_counts) ||
            !bind_int32(
                MTPSpecDecodeWorkspaceBuffers::REJECTED_TOKEN_COUNTS,
                &device_pointers_.rejected_token_counts) ||
            !bind_int32(
                MTPSpecDecodeWorkspaceBuffers::TOKEN_INDICES_TO_SAMPLE,
                &device_pointers_.token_indices_to_sample) ||
            !bind_int32(
                MTPSpecDecodeWorkspaceBuffers::NEXT_CONDITION_TOKENS,
                &device_pointers_.next_condition_tokens) ||
            !bind_int32(
                MTPSpecDecodeWorkspaceBuffers::ALL_DRAFTS_ACCEPTED_FLAGS,
                &device_pointers_.all_drafts_accepted_flags) ||
            !bind_int32(
                MTPSpecDecodeWorkspaceBuffers::STOPPED_FLAGS,
                &device_pointers_.stopped_flags) ||
            !bind_int32(
                MTPSpecDecodeWorkspaceBuffers::QUERY_START_LOCS,
                &device_pointers_.query_start_locs) ||
            !bind_int32(
                MTPSpecDecodeWorkspaceBuffers::STATE_INDICES,
                &device_pointers_.state_indices) ||
            !bind_int32(
                MTPSpecDecodeWorkspaceBuffers::ACCEPTED_STATE_COUNTS,
                &device_pointers_.accepted_state_counts) ||
            !bind_int32(
                MTPSpecDecodeWorkspaceBuffers::SPECULATIVE_STATE_SLOT_INDICES,
                &device_pointers_.speculative_state_slot_indices) ||
            !bind_int32_optional(
                MTPSpecDecodeWorkspaceBuffers::COMMITTED_STATE_ROWS,
                &device_pointers_.committed_state_rows) ||
            !bind_int32_optional(
                MTPSpecDecodeWorkspaceBuffers::COMMITTED_STATE_INDICES,
                &device_pointers_.committed_state_indices) ||
            !bind_int32(
                MTPSpecDecodeWorkspaceBuffers::ACCEPTED_STATE_SLOT_INDICES,
                &device_pointers_.accepted_state_slot_indices) ||
            !bind_int32(
                MTPSpecDecodeWorkspaceBuffers::BONUS_READY_TOKEN_ROWS,
                &device_pointers_.bonus_ready_token_rows) ||
            !bind_int32(
                MTPSpecDecodeWorkspaceBuffers::BONUS_READY_TOKEN_INDICES,
                &device_pointers_.bonus_ready_token_indices) ||
            !bind_int32(
                MTPSpecDecodeWorkspaceBuffers::BONUS_READY_STATE_SLOT_INDICES,
                &device_pointers_.bonus_ready_state_slot_indices) ||
            !bind_int32(
                MTPSpecDecodeWorkspaceBuffers::DRAFT_TOKENS,
                &device_pointers_.draft_tokens) ||
            !bind_int32(
                MTPSpecDecodeWorkspaceBuffers::SAMPLED_TOKENS,
                &device_pointers_.sampled_tokens) ||
            !bind_int32(
                MTPSpecDecodeWorkspaceBuffers::VERIFIER_LOGIT_ROWS,
                &device_pointers_.verifier_logit_rows))
        {
            binding_error_ = error.str();
            device_pointers_ = {};
        }
    }

    MTPSpecDecodeMetadataUploadResult uploadMTPSpecDecodeMetadataBatch(
        const MTPSpecDecodeMetadataBatch &batch,
        const MTPSpecDecodeMetadataWorkspaceBinding &binding,
        DeviceId device,
        IBackend *backend,
        void *stream)
    {
        MTPSpecDecodeMetadataUploadResult result;
        if (!batch.ok)
        {
            result.error = std::string("cannot upload invalid MTP metadata batch: ") +
                           batch.error;
            return result;
        }
        if (!binding.hasWorkspace())
        {
            result.error = std::string("MTP metadata workspace is not completely bound: ") +
                           binding.bindingError();
            return result;
        }
        if (device.is_gpu())
        {
            if (!stream)
            {
                result.error =
                    "MTP metadata GPU upload requires an explicit non-null stream";
                return result;
            }
            if (!backend)
            {
                result.error = "MTP metadata GPU upload requires a backend";
                return result;
            }
        }

        const auto &ptrs = binding.devicePointers();
        auto upload = [&](int32_t *dst,
                          const std::vector<int32_t> &src,
                          const char *name) -> bool
        {
            if (src.empty())
                return true;
            if (!dst)
            {
                result.error = std::string("MTP metadata destination is null for ") + name;
                return false;
            }

            const size_t bytes = src.size() * sizeof(int32_t);
            bool ok = true;
            if (device.is_gpu())
            {
                ok = backend->hostToDeviceOnStream(
                    dst,
                    src.data(),
                    bytes,
                    device.ordinal,
                    stream);
            }
            else
            {
                std::memcpy(dst, src.data(), bytes);
            }

            if (!ok)
            {
                result.error = std::string("MTP metadata upload failed for ") + name;
                return false;
            }

            result.bytes_uploaded += bytes;
            return true;
        };

        auto upload_optional = [&](int32_t *dst,
                                   const std::vector<int32_t> &src,
                                   const char *name) -> bool
        {
            if (!dst)
                return true;
            return upload(dst, src, name);
        };

        if (!upload(ptrs.draft_counts, batch.draft_counts,
                    MTPSpecDecodeWorkspaceBuffers::DRAFT_COUNTS) ||
            !upload(ptrs.target_query_lens, batch.target_query_lens,
                    MTPSpecDecodeWorkspaceBuffers::TARGET_QUERY_LENS) ||
            !upload(ptrs.valid_sampled_counts, batch.valid_sampled_counts,
                    MTPSpecDecodeWorkspaceBuffers::VALID_SAMPLED_COUNTS) ||
            !upload(ptrs.accepted_draft_prefixes, batch.accepted_draft_prefixes,
                    MTPSpecDecodeWorkspaceBuffers::ACCEPTED_DRAFT_PREFIXES) ||
            !upload(ptrs.committed_output_counts, batch.committed_output_counts,
                    MTPSpecDecodeWorkspaceBuffers::COMMITTED_OUTPUT_COUNTS) ||
            !upload(ptrs.rejected_token_counts, batch.rejected_token_counts,
                    MTPSpecDecodeWorkspaceBuffers::REJECTED_TOKEN_COUNTS) ||
            !upload(ptrs.token_indices_to_sample, batch.token_indices_to_sample,
                    MTPSpecDecodeWorkspaceBuffers::TOKEN_INDICES_TO_SAMPLE) ||
            !upload(ptrs.next_condition_tokens, batch.next_condition_tokens,
                    MTPSpecDecodeWorkspaceBuffers::NEXT_CONDITION_TOKENS) ||
            !upload(ptrs.all_drafts_accepted_flags, batch.all_drafts_accepted_flags,
                    MTPSpecDecodeWorkspaceBuffers::ALL_DRAFTS_ACCEPTED_FLAGS) ||
            !upload(ptrs.stopped_flags, batch.stopped_flags,
                    MTPSpecDecodeWorkspaceBuffers::STOPPED_FLAGS) ||
            !upload(ptrs.query_start_locs, batch.query_start_locs,
                    MTPSpecDecodeWorkspaceBuffers::QUERY_START_LOCS) ||
            !upload(ptrs.state_indices, batch.state_indices,
                    MTPSpecDecodeWorkspaceBuffers::STATE_INDICES) ||
            !upload(ptrs.accepted_state_counts, batch.accepted_state_counts,
                    MTPSpecDecodeWorkspaceBuffers::ACCEPTED_STATE_COUNTS) ||
            !upload(ptrs.speculative_state_slot_indices, batch.speculative_state_slot_indices,
                    MTPSpecDecodeWorkspaceBuffers::SPECULATIVE_STATE_SLOT_INDICES) ||
            !upload_optional(ptrs.committed_state_rows, batch.committed_state_rows,
                             MTPSpecDecodeWorkspaceBuffers::COMMITTED_STATE_ROWS) ||
            !upload_optional(ptrs.committed_state_indices, batch.committed_state_indices,
                             MTPSpecDecodeWorkspaceBuffers::COMMITTED_STATE_INDICES) ||
            !upload(ptrs.accepted_state_slot_indices, batch.accepted_state_slot_indices,
                    MTPSpecDecodeWorkspaceBuffers::ACCEPTED_STATE_SLOT_INDICES) ||
            !upload(ptrs.bonus_ready_token_rows, batch.bonus_ready_token_rows,
                    MTPSpecDecodeWorkspaceBuffers::BONUS_READY_TOKEN_ROWS) ||
            !upload(ptrs.bonus_ready_token_indices, batch.bonus_ready_token_indices,
                    MTPSpecDecodeWorkspaceBuffers::BONUS_READY_TOKEN_INDICES) ||
            !upload(ptrs.bonus_ready_state_slot_indices, batch.bonus_ready_state_slot_indices,
                    MTPSpecDecodeWorkspaceBuffers::BONUS_READY_STATE_SLOT_INDICES) ||
            !upload(ptrs.draft_tokens, batch.draft_tokens,
                    MTPSpecDecodeWorkspaceBuffers::DRAFT_TOKENS) ||
            !upload(ptrs.sampled_tokens, batch.sampled_tokens,
                    MTPSpecDecodeWorkspaceBuffers::SAMPLED_TOKENS))
        {
            return result;
        }

        result.ok = true;
        return result;
    }

    MTPSpecDecodeMetadataUploadResult uploadMTPSpecDecodeVerifierInputPlan(
        const MTPSpecDecodeVerifierInputPlan &plan,
        const MTPSpecDecodeMetadataWorkspaceBinding &binding,
        DeviceId device,
        IBackend *backend,
        void *stream)
    {
        MTPSpecDecodeMetadataUploadResult result;
        if (!plan.ok)
        {
            result.error = std::string("cannot upload invalid MTP verifier input plan: ") +
                           plan.error;
            return result;
        }
        if (!binding.hasWorkspace())
        {
            result.error = std::string("MTP verifier metadata workspace is not completely bound: ") +
                           binding.bindingError();
            return result;
        }
        if (plan.compact_logit_row_count < 0 ||
            plan.compact_logit_row_count >
                plan.shape.max_requests * plan.shape.maxTargetQueryLen() ||
            static_cast<int>(plan.verifier_logit_rows.size()) <
                plan.compact_logit_row_count)
        {
            result.error = "MTP verifier row plan is outside metadata shape";
            return result;
        }
        const MTPSpecDecodeVerifierGraphForwardPlan graph_plan =
            buildMTPSpecDecodeVerifierGraphForwardPlan(plan);
        if (!graph_plan.ok)
        {
            result.error = std::string("MTP verifier row graph materialization failed: ") +
                           graph_plan.error;
            return result;
        }
        if (static_cast<int>(graph_plan.verifier_logit_rows.size()) <
            plan.compact_logit_row_count)
        {
            result.error = "MTP verifier graph row plan is undersized";
            return result;
        }
        if (device.is_gpu())
        {
            if (!stream)
            {
                result.error =
                    "MTP verifier row metadata GPU upload requires an explicit non-null stream";
                return result;
            }
            if (!backend)
            {
                result.error = "MTP verifier row metadata GPU upload requires a backend";
                return result;
            }
        }

        const auto &ptrs = binding.devicePointers();
        if (!ptrs.verifier_logit_rows)
        {
            result.error = "MTP verifier row metadata workspace has no verifier row buffer";
            return result;
        }

        const size_t bytes =
            static_cast<size_t>(plan.compact_logit_row_count) * sizeof(int32_t);
        if (bytes == 0)
        {
            result.ok = true;
            return result;
        }

        bool ok = true;
        if (device.is_gpu())
        {
            ok = backend->hostToDeviceOnStream(
                ptrs.verifier_logit_rows,
                graph_plan.verifier_logit_rows.data(),
                bytes,
                device.ordinal,
                stream);
        }
        else
        {
            std::memcpy(
                ptrs.verifier_logit_rows,
                graph_plan.verifier_logit_rows.data(),
                bytes);
        }

        if (!ok)
        {
            result.error = "MTP verifier row metadata upload failed";
            return result;
        }

        result.ok = true;
        result.bytes_uploaded = bytes;
        return result;
    }

} // namespace llaminar2
