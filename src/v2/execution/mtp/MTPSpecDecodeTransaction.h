#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{

    constexpr int32_t kMTPSpecDecodeInvalidToken = -1;

    struct MTPSpecDecodeRequest
    {
        int request_id = 0;
        int32_t backup_next_token = kMTPSpecDecodeInvalidToken;
        std::vector<int32_t> draft_tokens;
        std::vector<int32_t> sampled_tokens;
        bool discarded = false;
        int vocab_size = 0;
    };

    struct MTPSpecDecodeTransaction
    {
        bool ok = false;
        std::string error;

        int request_id = 0;
        int draft_count = 0;
        int target_query_len = 0;
        int valid_sampled_count = 0;
        int accepted_speculative_prefix = 0;
        int rejected_token_count = 0;
        int token_index_to_sample = -1;
        int32_t next_condition_token = kMTPSpecDecodeInvalidToken;

        std::vector<int32_t> committed_output_tokens;
        std::vector<int32_t> rejected_draft_tokens;

        bool allDraftsAccepted() const
        {
            return draft_count > 0 && accepted_speculative_prefix == draft_count;
        }
    };

    struct MTPSpecDecodeBatchSummary
    {
        bool ok = false;
        std::string error;
        int total_target_query_tokens = 0;
        int total_valid_sampled_tokens = 0;
        int total_rejected_tokens = 0;
        std::vector<MTPSpecDecodeTransaction> transactions;
    };

    MTPSpecDecodeTransaction buildMTPSpecDecodeTransaction(
        const MTPSpecDecodeRequest &request);

    MTPSpecDecodeRequest buildMTPSpecDecodeRequestFromVerifierOutput(
        int request_id,
        int vocab_size,
        const std::vector<int32_t> &draft_tokens,
        const std::vector<int32_t> &committed_output_tokens,
        std::optional<int32_t> bonus_ready_token,
        bool all_drafts_accepted,
        bool stopped_on_output);

    MTPSpecDecodeTransaction buildMTPSpecDecodeTransactionFromVerifierOutput(
        int request_id,
        int vocab_size,
        const std::vector<int32_t> &draft_tokens,
        const std::vector<int32_t> &committed_output_tokens,
        std::optional<int32_t> bonus_ready_token,
        bool all_drafts_accepted,
        bool stopped_on_output);

    MTPSpecDecodeBatchSummary buildMTPSpecDecodeBatchSummary(
        const std::vector<MTPSpecDecodeRequest> &requests);

} // namespace llaminar2
