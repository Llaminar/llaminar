#pragma once

#include "MTPSpecDecodeTransaction.h"
#include "../../backends/DeviceId.h"
#include "../local_execution/device/WorkspaceDescriptor.h"
#include "../../interfaces/IWorkspaceConsumer.h"

#include <cstdint>
#include <optional>
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
        constexpr const char *BASE_CACHED_TOKENS = "mtp_spec_decode_base_cached_tokens";
        constexpr const char *TARGET_CACHED_TOKENS = "mtp_spec_decode_target_cached_tokens";
        constexpr const char *SHIFTED_TARGET_CACHED_TOKENS = "mtp_spec_decode_shifted_target_cached_tokens";
        constexpr const char *SHIFTED_ACCEPTED_STATE_COUNTS = "mtp_spec_decode_shifted_accepted_state_counts";
        constexpr const char *PUBLICATION_OK_FLAGS = "mtp_spec_decode_publication_ok_flags";
        constexpr const char *QUERY_START_LOCS = "mtp_spec_decode_query_start_locs";
        constexpr const char *STATE_INDICES = "mtp_spec_decode_state_indices";
        constexpr const char *ACCEPTED_STATE_COUNTS = "mtp_spec_decode_accepted_state_counts";
        constexpr const char *SPECULATIVE_STATE_SLOT_INDICES = "mtp_spec_decode_speculative_state_slot_indices";
        constexpr const char *COMMITTED_STATE_ROWS = "mtp_spec_decode_committed_state_rows";
        constexpr const char *COMMITTED_STATE_INDICES = "mtp_spec_decode_committed_state_indices";
        constexpr const char *ACCEPTED_STATE_SLOT_INDICES = "mtp_spec_decode_accepted_state_slot_indices";
        constexpr const char *BONUS_READY_TOKEN_ROWS = "mtp_spec_decode_bonus_ready_token_rows";
        constexpr const char *BONUS_READY_TOKEN_INDICES = "mtp_spec_decode_bonus_ready_token_indices";
        constexpr const char *BONUS_READY_STATE_SLOT_INDICES = "mtp_spec_decode_bonus_ready_state_slot_indices";
        constexpr const char *DRAFT_TOKENS = "mtp_spec_decode_draft_tokens";
        constexpr const char *SAMPLED_TOKENS = "mtp_spec_decode_sampled_tokens";
        constexpr const char *VERIFIER_LOGIT_ROWS = "mtp_spec_decode_verifier_logit_rows";
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
        std::vector<int32_t> accepted_state_counts;
        std::vector<int32_t> speculative_state_slot_indices;
        std::vector<int32_t> committed_state_rows;
        std::vector<int32_t> committed_state_indices;
        std::vector<int32_t> accepted_state_slot_indices;
        std::vector<int32_t> correction_replay_start_indices;
        std::vector<int32_t> correction_replay_counts;
        std::vector<int32_t> bonus_ready_token_rows;
        std::vector<int32_t> bonus_ready_token_indices;
        std::vector<int32_t> bonus_ready_state_slot_indices;
        std::vector<int32_t> draft_tokens;
        std::vector<int32_t> sampled_tokens;
        std::vector<MTPSpecDecodeTransaction> transactions;
    };

    /**
     * @brief Draft-token request used before target verifier logits exist.
     *
     * This is the pre-verification side of the vLLM-style metadata contract.
     * The normal `MTPSpecDecodeRequest` also contains sampled verifier tokens,
     * which are not available until after the target verifier graph runs.
     */
    struct MTPSpecDecodeVerifierDraftRequest
    {
        int request_id = 0;
        std::vector<int32_t> draft_tokens;
    };

    /**
     * @brief Flattened target-verifier input and row-selection plan.
     *
     * Llaminar's current SingleDevice verifier contract feeds the already
     * accepted main token followed by sidecar draft tokens into one target
     * forward. The logits rows produced by that forward verify the later draft
     * tokens and provide the bonus-ready row when every draft is accepted. This
     * plan makes that row mapping explicit so the graph builder, sampler, and
     * state publication path share the same compact-row semantics.
     */
    struct MTPSpecDecodeVerifierInputPlan
    {
        bool ok = false;
        std::string error;

        MTPSpecDecodeMetadataShape shape;
        int request_count = 0;
        int total_verifier_input_tokens = 0;
        int compact_logit_row_count = 0;

        std::vector<int32_t> verifier_input_tokens;
        std::vector<int32_t> query_start_locs;
        std::vector<int32_t> verifier_logit_rows;
        std::vector<int32_t> bonus_logit_rows;
    };

    /**
     * @brief Verifier input materialized in graph execution coordinates.
     *
     * `MTPSpecDecodeVerifierInputPlan` stores a compact logical sequence where
     * requests are concatenated without padding. Batched graph execution pads
     * each request to the maximum target length, so row-select stages must read
     * padded graph row indices instead of logical flat row indices. This result
     * keeps both concepts explicit: `token_batches` are suitable for the
     * existing `forward_batch()` path, while `verifier_logit_rows` and
     * `bonus_logit_rows` are the row coordinates seen by graph stages.
     */
    struct MTPSpecDecodeVerifierGraphForwardPlan
    {
        bool ok = false;
        std::string error;

        MTPSpecDecodeMetadataShape shape;
        int request_count = 0;
        int padded_seq_len = 0;
        int total_graph_tokens = 0;

        std::vector<std::vector<int>> token_batches;
        std::vector<int> sequence_lengths;
        std::vector<int32_t> verifier_logit_rows;
        std::vector<int32_t> bonus_logit_rows;
    };

    struct MTPSpecDecodeStateCommitPlan
    {
        bool ok = false;
        std::string error;

        MTPSpecDecodeMetadataShape shape;
        int request_count = 0;

        std::vector<int32_t> committed_state_rows;
        std::vector<int32_t> committed_state_indices;
        std::vector<int32_t> accepted_state_counts;
        std::vector<int32_t> accepted_state_slot_indices;
        std::vector<int32_t> correction_replay_start_indices;
        std::vector<int32_t> correction_replay_counts;
        std::vector<int32_t> bonus_ready_token_rows;
        std::vector<int32_t> bonus_ready_token_indices;
        std::vector<int32_t> bonus_ready_state_slot_indices;
    };

    struct MTPSpecDecodeStatePublicationPlan
    {
        bool ok = false;
        std::string error;

        MTPSpecDecodeMetadataShape shape;
        int request_count = 0;

        std::vector<int32_t> base_cached_tokens;
        std::vector<int32_t> target_cached_tokens;
        std::vector<int32_t> accepted_state_counts;
        std::vector<int32_t> accepted_state_slot_indices;
        std::vector<int32_t> correction_replay_start_indices;
        std::vector<int32_t> correction_replay_counts;
        std::vector<int32_t> bonus_ready_token_rows;
        std::vector<int32_t> bonus_ready_token_indices;
        std::vector<int32_t> bonus_ready_state_slot_indices;
    };

    /**
     * @brief Accepted-count view of one completed speculative decode step.
     *
     * This is the vLLM-shaped metadata input used when draft tokens stayed in
     * device slots and the host only knows the verifier outcome.  The
     * `accepted_verifier_input_prefix` count includes the first main-model
     * token at verifier row zero; it is therefore one larger than the number of
     * accepted sidecar draft tokens when at least the first output token was
     * produced.
     */
    struct MTPSpecDecodeAcceptedOutcome
    {
        int request_id = 0;
        int vocab_size = 0;
        int draft_count = 0;
        std::vector<int32_t> committed_output_tokens;
        std::optional<int32_t> bonus_ready_token;
        int accepted_verifier_input_prefix = 0;
        int target_verifier_state_commit_count = -1;
        bool all_drafts_accepted = false;
        bool stopped_on_output = false;
    };

    WorkspaceRequirements buildMTPSpecDecodeWorkspaceRequirements(
        const MTPSpecDecodeMetadataShape &shape);

    MTPSpecDecodeMetadataBatch buildMTPSpecDecodeMetadataBatch(
        const MTPSpecDecodeMetadataShape &shape,
        const std::vector<MTPSpecDecodeRequest> &requests,
        const std::vector<int32_t> &committed_output_counts,
        const std::vector<int32_t> &stopped_flags);

    /**
     * @brief Build pre-verifier row metadata from draft tokens.
     *
     * The result remains host-readable for CPU tests and diagnostics. GPU
     * runners upload the compact row-index subset through
     * `uploadMTPSpecDecodeVerifierInputPlan()` so graph-captured row-select
     * stages read stable workspace metadata instead of stage-local vectors.
     */
    MTPSpecDecodeVerifierInputPlan buildMTPSpecDecodeVerifierInputPlan(
        const MTPSpecDecodeMetadataShape &shape,
        const std::vector<MTPSpecDecodeVerifierDraftRequest> &requests);

    MTPSpecDecodeVerifierGraphForwardPlan buildMTPSpecDecodeVerifierGraphForwardPlan(
        const MTPSpecDecodeVerifierInputPlan &plan);

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

    /**
     * @brief Build metadata for a compact batch of greedy verifier results.
     *
     * This is the request-batched equivalent of
     * `buildMTPSpecDecodeMetadataBatchFromGreedyCatchup()`.  Every request
     * supplies its draft-token vector and all-position verifier result; the
     * returned batch uses one flattened verifier-state index space so state
     * publication can target the same rows sampled by the verifier graph.
     */
    MTPSpecDecodeMetadataBatch buildMTPSpecDecodeMetadataBatchFromGreedyCatchups(
        const MTPSpecDecodeMetadataShape &shape,
        const std::vector<int> &request_ids,
        int vocab_size,
        const std::vector<MTPDecodeCatchupGreedyRequest> &requests,
        const std::vector<MTPDecodeCatchupGreedyResult> &results);

    /**
     * @brief Build metadata from accepted counts instead of host draft tokens.
     *
     * Use this for device-resident stochastic MTP lanes where rejected draft
     * token slots intentionally never leave GPU memory.  The resulting metadata
     * preserves the same state/publication contract as the draft-token helper,
     * but its accepted prefix is provided by the verifier outcome rather than
     * reconstructed by comparing host-visible token vectors.
     */
    MTPSpecDecodeMetadataBatch buildMTPSpecDecodeMetadataBatchFromAcceptedOutcome(
        const MTPSpecDecodeMetadataShape &shape,
        const MTPSpecDecodeAcceptedOutcome &outcome);

    /**
     * @brief Build metadata for a batch of device-resident verifier outcomes.
     *
     * This is the shared vLLM-style path for amortized speculative decode: each
     * request contributes its accepted-count result, optional bonus-ready token,
     * and committed output tokens while rejected draft identities may remain on
     * device. The returned metadata uses padded per-request slots but assigns
     * state indices from one flattened target-verifier batch, which is the
     * layout the graph-side row selectors and state publishers consume.
     */
    MTPSpecDecodeMetadataBatch buildMTPSpecDecodeMetadataBatchFromAcceptedOutcomes(
        const MTPSpecDecodeMetadataShape &shape,
        const std::vector<MTPSpecDecodeAcceptedOutcome> &outcomes);

    MTPSpecDecodeStateCommitPlan buildMTPSpecDecodeStateCommitPlan(
        const MTPSpecDecodeMetadataBatch &batch);

    MTPSpecDecodeStatePublicationPlan buildMTPSpecDecodeStatePublicationPlan(
        const MTPSpecDecodeStateCommitPlan &commit_plan,
        const std::vector<int32_t> &base_cached_tokens);

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
        int32_t *base_cached_tokens = nullptr;
        int32_t *target_cached_tokens = nullptr;
        int32_t *shifted_target_cached_tokens = nullptr;
        int32_t *shifted_accepted_state_counts = nullptr;
        int32_t *publication_ok_flags = nullptr;
        int32_t *query_start_locs = nullptr;
        int32_t *state_indices = nullptr;
        int32_t *accepted_state_counts = nullptr;
        int32_t *speculative_state_slot_indices = nullptr;
        int32_t *committed_state_rows = nullptr;
        int32_t *committed_state_indices = nullptr;
        int32_t *accepted_state_slot_indices = nullptr;
        int32_t *bonus_ready_token_rows = nullptr;
        int32_t *bonus_ready_token_indices = nullptr;
        int32_t *bonus_ready_state_slot_indices = nullptr;
        int32_t *draft_tokens = nullptr;
        int32_t *sampled_tokens = nullptr;
        int32_t *verifier_logit_rows = nullptr;

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

    /**
     * @brief Upload row indices consumed by graph-captured verifier row select stages.
     *
     * This is the narrow primitive behind compact all-position logits. The
     * stochastic verifier path usually calls `uploadMTPSpecDecodeVerifierInputPlan()`,
     * which first maps logical verifier rows into padded graph rows. Other
     * graph-native users, such as request-batched prefill terminal-logit
     * selection, may already have graph row coordinates and can upload them
     * directly through this helper. GPU callers must pass the explicit stream
     * that will execute or replay the graph.
     */
    MTPSpecDecodeMetadataUploadResult uploadMTPSpecDecodeVerifierLogitRows(
        const std::vector<int32_t> &verifier_logit_rows,
        int row_count,
        const MTPSpecDecodeMetadataWorkspaceBinding &binding,
        DeviceId device,
        IBackend *backend,
        void *stream);

    /**
     * @brief Upload the compact target-verifier row plan for graph replay.
     *
     * The all-position verifier graph reads `VERIFIER_LOGIT_ROWS` from the same
     * runner-owned metadata workspace as the stochastic/spec-state kernels. GPU
     * callers must pass the exact explicit stream that will execute or replay
     * the graph so the row-select stage observes the freshly uploaded indices.
     */
    MTPSpecDecodeMetadataUploadResult uploadMTPSpecDecodeVerifierInputPlan(
        const MTPSpecDecodeVerifierInputPlan &plan,
        const MTPSpecDecodeMetadataWorkspaceBinding &binding,
        DeviceId device,
        IBackend *backend,
        void *stream);

} // namespace llaminar2
