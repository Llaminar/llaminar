#pragma once

#include "MTPSpecDecodeTransaction.h"
#include "../../backends/DeviceId.h"
#include "../local_execution/device/WorkspaceDescriptor.h"
#include "../../interfaces/IWorkspaceConsumer.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llaminar2
{
    class IBackend;
    struct MTPDecodeCatchupGreedyRequest;
    struct MTPDecodeCatchupGreedyResult;

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
        constexpr const char *COMMITTED_STATE_ROWS = "mtp_spec_decode_committed_state_rows";
        constexpr const char *COMMITTED_STATE_INDICES = "mtp_spec_decode_committed_state_indices";
        constexpr const char *BONUS_READY_TOKEN_ROWS = "mtp_spec_decode_bonus_ready_token_rows";
        constexpr const char *BONUS_READY_TOKEN_INDICES = "mtp_spec_decode_bonus_ready_token_indices";
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
        std::vector<int32_t> target_verifier_state_commit_counts;
        std::vector<int32_t> rejected_token_counts;
        std::vector<int32_t> token_indices_to_sample;
        std::vector<int32_t> next_condition_tokens;
        std::vector<int32_t> all_drafts_accepted_flags;
        std::vector<int32_t> stopped_flags;
        std::vector<int32_t> query_start_locs;
        std::vector<int32_t> state_indices;
        std::vector<int32_t> committed_state_rows;
        std::vector<int32_t> committed_state_indices;
        std::vector<int32_t> bonus_ready_token_rows;
        std::vector<int32_t> bonus_ready_token_indices;
        std::vector<int32_t> draft_tokens;
        std::vector<int32_t> sampled_tokens;
        std::vector<MTPSpecDecodeTransaction> transactions;
    };

    struct MTPSpecDecodeStateCommitPlan
    {
        bool ok = false;
        std::string error;

        MTPSpecDecodeMetadataShape shape;
        int request_count = 0;

        std::vector<int32_t> committed_state_rows;
        std::vector<int32_t> committed_state_indices;
        std::vector<int32_t> bonus_ready_token_rows;
        std::vector<int32_t> bonus_ready_token_indices;
    };

    WorkspaceRequirements buildMTPSpecDecodeWorkspaceRequirements(
        const MTPSpecDecodeMetadataShape &shape);

    MTPSpecDecodeMetadataBatch buildMTPSpecDecodeMetadataBatch(
        const MTPSpecDecodeMetadataShape &shape,
        const std::vector<MTPSpecDecodeRequest> &requests,
        const std::vector<int32_t> &committed_output_counts,
        const std::vector<int32_t> &stopped_flags);

    MTPSpecDecodeMetadataBatch buildMTPSpecDecodeMetadataBatchWithStateCommitCounts(
        const MTPSpecDecodeMetadataShape &shape,
        const std::vector<MTPSpecDecodeRequest> &requests,
        const std::vector<int32_t> &committed_output_counts,
        const std::vector<int32_t> &target_verifier_state_commit_counts,
        const std::vector<int32_t> &stopped_flags);

    MTPSpecDecodeMetadataBatch buildMTPSpecDecodeMetadataBatchFromGreedyCatchup(
        const MTPSpecDecodeMetadataShape &shape,
        int request_id,
        int vocab_size,
        const MTPDecodeCatchupGreedyRequest &request,
        const MTPDecodeCatchupGreedyResult &result);

    MTPSpecDecodeStateCommitPlan buildMTPSpecDecodeStateCommitPlan(
        const MTPSpecDecodeMetadataBatch &batch);

    struct MTPSpecDecodeMetadataDevicePointers
    {
        int32_t *draft_counts = nullptr;
        int32_t *target_query_lens = nullptr;
        int32_t *valid_sampled_counts = nullptr;
        int32_t *accepted_draft_prefixes = nullptr;
        int32_t *committed_output_counts = nullptr;
        int32_t *rejected_token_counts = nullptr;
        int32_t *token_indices_to_sample = nullptr;
        int32_t *next_condition_tokens = nullptr;
        int32_t *all_drafts_accepted_flags = nullptr;
        int32_t *stopped_flags = nullptr;
        int32_t *query_start_locs = nullptr;
        int32_t *state_indices = nullptr;
        int32_t *committed_state_rows = nullptr;
        int32_t *committed_state_indices = nullptr;
        int32_t *bonus_ready_token_rows = nullptr;
        int32_t *bonus_ready_token_indices = nullptr;
        int32_t *draft_tokens = nullptr;
        int32_t *sampled_tokens = nullptr;

        bool complete() const;
    };

    /**
     * Runner-owned workspace consumer for graph-facing spec-decode metadata.
     *
     * The buffers are not a graph stage scratch allocation. They are persistent
     * per-runner metadata slots that graph-captured MTP verifier/state stages
     * can read after the runner uploads a new batch on an explicit stream.
     */
    class MTPSpecDecodeMetadataWorkspaceBinding : public IWorkspaceConsumer
    {
    public:
        explicit MTPSpecDecodeMetadataWorkspaceBinding(
            MTPSpecDecodeMetadataShape shape = {});

        void setShape(MTPSpecDecodeMetadataShape shape);
        const MTPSpecDecodeMetadataShape &shape() const { return shape_; }

        WorkspaceRequirements getWorkspaceRequirements(
            int m, int n = 0, int k = 0) const override;

        void bindWorkspace(DeviceWorkspaceManager *workspace) override;
        bool hasWorkspace() const override;
        DeviceWorkspaceManager *getWorkspace() const override { return workspace_; }

        const MTPSpecDecodeMetadataDevicePointers &devicePointers() const
        {
            return device_pointers_;
        }

        const std::string &bindingError() const { return binding_error_; }

    private:
        MTPSpecDecodeMetadataShape effectiveShape(int m, int n, int k) const;
        void refreshDevicePointers();

        MTPSpecDecodeMetadataShape shape_;
        DeviceWorkspaceManager *workspace_ = nullptr;
        MTPSpecDecodeMetadataDevicePointers device_pointers_;
        std::string binding_error_;
    };

    struct MTPSpecDecodeMetadataUploadResult
    {
        bool ok = false;
        std::string error;
        size_t bytes_uploaded = 0;
    };

    MTPSpecDecodeMetadataUploadResult uploadMTPSpecDecodeMetadataBatch(
        const MTPSpecDecodeMetadataBatch &batch,
        const MTPSpecDecodeMetadataWorkspaceBinding &binding,
        DeviceId device,
        IBackend *backend,
        void *stream);

} // namespace llaminar2
