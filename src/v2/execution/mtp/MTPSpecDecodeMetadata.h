#pragma once

#include "MTPSpecDecodeTransaction.h"
#include "../local_execution/device/WorkspaceDescriptor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llaminar2
{
    namespace MTPSpecDecodeWorkspaceBuffers
    {
        constexpr const char *DRAFT_COUNTS = "mtp_spec_decode_draft_counts";
        constexpr const char *TARGET_QUERY_LENS = "mtp_spec_decode_target_query_lens";
        constexpr const char *VALID_SAMPLED_COUNTS = "mtp_spec_decode_valid_sampled_counts";
        constexpr const char *ACCEPTED_DRAFT_PREFIXES = "mtp_spec_decode_accepted_draft_prefixes";
        constexpr const char *COMMITTED_OUTPUT_COUNTS = "mtp_spec_decode_committed_output_counts";
        constexpr const char *REJECTED_TOKEN_COUNTS = "mtp_spec_decode_rejected_token_counts";
        constexpr const char *TOKEN_INDICES_TO_SAMPLE = "mtp_spec_decode_token_indices_to_sample";
        constexpr const char *NEXT_CONDITION_TOKENS = "mtp_spec_decode_next_condition_tokens";
        constexpr const char *ALL_DRAFTS_ACCEPTED_FLAGS = "mtp_spec_decode_all_drafts_accepted_flags";
        constexpr const char *STOPPED_FLAGS = "mtp_spec_decode_stopped_flags";
        constexpr const char *QUERY_START_LOCS = "mtp_spec_decode_query_start_locs";
        constexpr const char *STATE_INDICES = "mtp_spec_decode_state_indices";
        constexpr const char *DRAFT_TOKENS = "mtp_spec_decode_draft_tokens";
        constexpr const char *SAMPLED_TOKENS = "mtp_spec_decode_sampled_tokens";
    } // namespace MTPSpecDecodeWorkspaceBuffers

    struct MTPSpecDecodeMetadataShape
    {
        int max_requests = 1;
        int max_draft_tokens = 0;

        int maxTargetQueryLen() const { return max_draft_tokens + 1; }
        bool valid() const { return max_requests > 0 && max_draft_tokens > 0; }
    };

    struct MTPSpecDecodeMetadataBatch
    {
        bool ok = false;
        std::string error;

        MTPSpecDecodeMetadataShape shape;
        int request_count = 0;
        int total_target_query_tokens = 0;

        std::vector<int32_t> draft_counts;
        std::vector<int32_t> target_query_lens;
        std::vector<int32_t> valid_sampled_counts;
        std::vector<int32_t> accepted_draft_prefixes;
        std::vector<int32_t> committed_output_counts;
        std::vector<int32_t> rejected_token_counts;
        std::vector<int32_t> token_indices_to_sample;
        std::vector<int32_t> next_condition_tokens;
        std::vector<int32_t> all_drafts_accepted_flags;
        std::vector<int32_t> stopped_flags;
        std::vector<int32_t> query_start_locs;
        std::vector<int32_t> state_indices;
        std::vector<int32_t> draft_tokens;
        std::vector<int32_t> sampled_tokens;
        std::vector<MTPSpecDecodeTransaction> transactions;
    };

    WorkspaceRequirements buildMTPSpecDecodeWorkspaceRequirements(
        const MTPSpecDecodeMetadataShape &shape);

    MTPSpecDecodeMetadataBatch buildMTPSpecDecodeMetadataBatch(
        const MTPSpecDecodeMetadataShape &shape,
        const std::vector<MTPSpecDecodeRequest> &requests,
        const std::vector<int32_t> &committed_output_counts,
        const std::vector<int32_t> &stopped_flags);

} // namespace llaminar2
