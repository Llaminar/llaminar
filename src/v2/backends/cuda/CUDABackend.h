/**
 * @file CUDABackend.h
 * @brief CUDA backend public API (no CUDA headers exposed)
 *
 * **Purpose**: Public interface for CUDA backend. Implementation lives in .cu file
 * to avoid exposing cuda_runtime.h to other compilation units.
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../IBackend.h"
#include <cstdint>
#include <vector>

namespace llaminar2
{

    /**
     * @class CUDABackend
     * @brief CUDA compute backend implementation
     *
     * **Implementation**: See CUDABackend.cu
     * **Requirements**: NVIDIA GPU with CUDA Toolkit 11.0+
     * **Compilation**: Requires nvcc compiler, -DHAVE_CUDA=ON
     */
    class CUDABackend : public IBackend
    {
    public:
        CUDABackend();
        ~CUDABackend() override;

        // Memory transfer operations (see IBackend documentation)
        bool deviceToHost(void *dst, const void *src, size_t bytes, int device_id, void *stream) override;
        bool hostToDevice(void *dst, const void *src, size_t bytes, int device_id, void *stream) override;
        bool deviceToDevice(void *dst, const void *src, size_t bytes, int device_id, void *stream) override;
        bool synchronize(int device_id) override;
        bool streamSynchronize(int device_id) override;
        bool setDevice(int device_id) override;

        // Event operations (fine-grained synchronization)
        void *createEvent(int device_id) override;
        void *createTimingEvent(int device_id) override;
        void destroyEvent(void *event, int device_id) override;
        bool recordEvent(void *event, int device_id, void *stream) override;
        bool waitForEvent(void *event, int device_id) override;
        bool eventElapsedTimeMs(
            void *start_event,
            void *stop_event,
            int device_id,
            float *out_ms) override;

        // Memory allocation operations
        void *allocate(size_t bytes, int device_id) override;
        void free(void *ptr, int device_id) override;
        bool memset(void *ptr, int value, size_t bytes, int device_id, void *stream) override;

        // Zero-copy mapped memory operations
        void *allocateMapped(size_t bytes, int device_id, void **device_ptr) override;
        void freeMapped(void *host_ptr, int device_id) override;

        // Device query operations
        int deviceCount() const override;
        std::string backendName() const override;
        std::string deviceName(int device_id) const override;
        size_t deviceMemoryTotal(int device_id) const override;
        size_t deviceMemoryFree(int device_id) const override;

        // Host memory pinning for async DMA
        bool pinHostMemory(void *ptr, size_t bytes) override;
        bool unpinHostMemory(void *ptr) override;

        // GPU-side argmax for greedy sampling
        bool argmaxF32(const void *data_device, int n, int device_id,
                       float *out_value, int *out_index, void *stream,
                       void *partial_vals = nullptr, void *partial_idxs = nullptr,
                       int partial_capacity = 0) override;
        bool argmaxF32BatchedRows(const void *data_device, int rows, int cols, int device_id,
                                  float *out_values, int *out_indices, void *stream,
                                  void *partial_vals = nullptr, void *partial_idxs = nullptr,
                                  int partial_capacity = 0) override;
        bool enqueueArgmaxF32BatchedRowsDevice(
            const void *data_device,
            int rows,
            int cols,
            int device_id,
            void *stream,
            void *out_values_device,
            void *out_indices_device,
            void *partial_vals = nullptr,
            void *partial_idxs = nullptr,
            int partial_capacity = 0,
            int output_stride = 1) override;
        bool enqueueArgmaxF32BatchedRowsAndPublishMTPChainDevice(
            const void *data_device,
            int rows,
            int cols,
            int device_id,
            void *stream,
            void *out_values_device,
            void *out_indices_device,
            void *chain_condition_tokens_device,
            void *chain_position_ids_device,
            int chain_position_increment,
            void *partial_vals,
            void *partial_idxs,
            int partial_capacity,
            int output_stride = 1) override;
        bool enqueueRetainMTPFirstTransactionDraftBoundaryDevice(
            const void *data_words_device,
            int word_count,
            int boundary,
            int draft_slot,
            const void *condition_token_device,
            const void *position_id_device,
            const void *generation_control_device,
            int generation_control_stride,
            void *diagnostic_record_device,
            int device_id,
            void *stream) override;
        bool enqueueConfigureMTPGreedyPenaltyPolicyDevice(
            void *controls_device,
            float presence_penalty,
            float frequency_penalty,
            bool first_token_already_in_history,
            int device_id,
            void *stream) override;
        bool enqueueArgmaxF32BatchedRowsWithMTPPenaltiesDevice(
            const void *data_device,
            int rows,
            int cols,
            const void *verifier_input_tokens_device,
            const void *generated_token_counts_device,
            const void *penalty_policy_device,
            const void *active_rows_device,
            int device_id,
            void *stream,
            void *out_values_device,
            void *out_indices_device,
            void *partial_vals,
            void *partial_idxs,
            int partial_capacity,
            int output_stride = 1) override;
        bool enqueueApplyMTPPenaltiesToF32RowsDevice(
            void *data_device,
            int rows,
            int cols,
            int row_stride,
            const void *verifier_input_tokens_device,
            const void *generated_token_counts_device,
            const void *penalty_policy_device,
            int device_id,
            void *stream,
            const void *active_rows_device = nullptr) override;
        bool enqueueApplyMTPBranchPenaltiesToF32RowDevice(
            void *data_device,
            int cols,
            const void *first_condition_token_device,
            const void *prior_draft_tokens_device,
            int prior_draft_count,
            const void *generated_token_counts_device,
            const void *penalty_policy_device,
            int device_id,
            void *stream) override;
        bool enqueueCommitMTPGreedyPenaltyHistoryDevice(
            const void *output_tokens_device,
            const void *output_meta_device,
            void *penalty_policy_device,
            const void *accepted_state_counts_device,
            const void *stopped_flags_device,
            int output_token_capacity,
            int vocab_size,
            void *generated_token_counts_device,
            int device_id,
            void *stream) override;

        // GPU-side top-k selection for sampling
        bool topKF32(const void *data_device, int n, int k, int device_id,
                     float *out_values, int *out_indices, void *stream) override;
        bool sampleTopKTopPF32(const void *data_device, int n,
                               int top_k, float top_p, float temperature,
                               uint64_t rng_seed, uint64_t rng_offset,
                               int device_id, int *out_token,
                               void *stream) override;
        bool enqueueSampleTopKTopPF32Device(const void *data_device, int n,
                                            int top_k, float top_p, float temperature,
                                            uint64_t rng_seed, uint64_t rng_offset,
                                            int device_id, void *stream,
                                            void *out_token_device) override;
        bool enqueuePublishInt32ControlScalarDevice(
            int32_t value,
            void *out_value_device,
            int device_id,
            void *stream) override;
        bool enqueueBuildTopKTopPDistributionF32Device(const void *data_device, int n,
                                                       int top_k, float top_p, float temperature,
                                                       int device_id, void *stream,
                                                       void *out_token_ids_device,
                                                       void *out_probs_device,
                                                       void *scratch_values_device = nullptr,
                                                       void *scratch_indices_device = nullptr,
                                                       int scratch_capacity = 0) override;
        bool enqueueBuildTopKTopPDistributionsF32Device(
            const void *data_device,
            int row_count,
            int n,
            int row_stride,
            int top_k,
            float top_p,
            float temperature,
            int device_id,
            void *stream,
            void *out_token_ids_device,
            int out_stride,
            void *out_probs_device,
            void *scratch_values_device = nullptr,
            void *scratch_indices_device = nullptr,
            int scratch_capacity = 0,
            const void *active_rows_device = nullptr) override;
        bool enqueueBuildTopKTopPProcessedLogitsF32Device(
            const void *data_device,
            int row_count,
            int n,
            int row_stride,
            int top_k,
            float top_p,
            float temperature,
            int device_id,
            void *stream,
            void *out_logits_device,
            int out_row_stride,
            void *scratch_values_device = nullptr,
            void *scratch_indices_device = nullptr,
            int scratch_capacity = 0) override;
        bool enqueueSampleDistributionF32Device(
            const void *token_ids_device,
            const void *probs_device,
            int top_k,
            float threshold,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_probability_device = nullptr,
            uint64_t threshold_seed = 0,
            const void *threshold_position_device = nullptr,
            int threshold_position_offset = 0) override;
        bool enqueueSampleProcessedLogitsF32Device(
            const void *logits_device,
            int vocab_size,
            int row_stride,
            float threshold,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_probability_device = nullptr) override;
        bool enqueueSampleProcessedLogitsF32DeviceIfSpeculativeBatchNeedsBonus(
            const void *logits_device,
            int vocab_size,
            int row_stride,
            float threshold,
            const void *verify_tokens_device,
            const void *verify_accepted_device,
            int row_count,
            int first_token,
            const void *first_token_device,
            const int *stop_tokens_host,
            int stop_token_count,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_probability_device = nullptr,
            uint64_t threshold_seed = 0,
            const void *threshold_position_device = nullptr,
            int threshold_position_offset = 0) override;
        bool enqueueSoftmaxAndSampleTemperatureLogitsF32Device(
            const void *logits_device,
            int vocab_size,
            int row_stride,
            float temperature,
            float threshold,
            int device_id,
            void *stream,
            void *out_probabilities_device,
            int out_row_stride,
            void *out_token_device,
            void *out_probability_device = nullptr) override;
        bool enqueueScaleAndSampleTemperatureLogitsF32Device(
            const void *logits_device,
            int vocab_size,
            int row_stride,
            float temperature,
            float threshold,
            int device_id,
            void *stream,
            void *out_logits_device,
            int out_row_stride,
            void *out_token_device,
            void *out_probability_device = nullptr) override;
        bool enqueueSoftmaxProcessedLogitsF32Device(
            const void *logits_device,
            int row_count,
            int vocab_size,
            int row_stride,
            int device_id,
            void *stream,
            void *out_probabilities_device,
            int out_row_stride) override;
        bool enqueueFillInverseExponentialSamplesF32Device(
            void *out_samples_device,
            int row_count,
            int vocab_size,
            int row_stride,
            uint64_t seed,
            int first_logical_position,
            int device_id,
            void *stream) override;
        bool enqueueSpeculativeVerifyDistributionsF32Device(
            const void *target_token_ids_device,
            const void *target_probs_device,
            const void *draft_token_ids_device,
            const void *draft_probs_device,
            int top_k,
            int draft_token,
            uint64_t accept_seed,
            uint64_t accept_offset,
            uint64_t residual_seed,
            uint64_t residual_offset,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_accepted_device,
            void *out_accept_probability_device = nullptr,
            void *out_accept_threshold_device = nullptr) override;
        bool enqueueSpeculativeVerifyDistributionsF32DeviceThresholds(
            const void *target_token_ids_device,
            const void *target_probs_device,
            const void *draft_token_ids_device,
            const void *draft_probs_device,
            int top_k,
            int draft_token,
            float accept_threshold,
            float residual_threshold,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_accepted_device,
            void *out_accept_probability_device = nullptr,
            void *out_accept_threshold_device = nullptr) override;
        bool enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatch(
            const void *target_token_ids_device,
            const void *target_probs_device,
            const void *draft_token_ids_device,
            const void *draft_probs_device,
            int top_k,
            int distribution_stride,
            const int *draft_tokens_host,
            const float *accept_thresholds_host,
            const float *residual_thresholds_host,
            int row_count,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_accepted_device,
            void *out_accept_probability_device = nullptr,
            void *out_accept_threshold_device = nullptr) override;
        bool enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatchDeviceTokens(
            const void *target_token_ids_device,
            const void *target_probs_device,
            const void *draft_token_ids_device,
            const void *draft_probs_device,
            int top_k,
            int distribution_stride,
            const void *draft_tokens_device,
            const float *accept_thresholds_host,
            const float *residual_thresholds_host,
            int row_count,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_accepted_device,
            void *out_accept_probability_device = nullptr,
            void *out_accept_threshold_device = nullptr,
            const void *draft_token_probabilities_device = nullptr,
            uint64_t inverse_sample_seed = 0,
            int inverse_sample_first_logical_position = 0,
            int inverse_sample_vocab_size = 0,
            const void *threshold_base_position_device = nullptr,
            int threshold_position_offset = 0) override;
        bool enqueueSpeculativeVerifyProcessedLogitsF32DeviceThresholdsBatchDeviceTokens(
            const void *target_logits_device,
            const void *draft_logits_device,
            int row_count,
            int vocab_size,
            int target_row_stride,
            int draft_row_stride,
            const void *draft_tokens_device,
            const float *accept_thresholds_host,
            const float *residual_thresholds_host,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_accepted_device,
            void *out_accept_probability_device = nullptr,
            void *out_accept_threshold_device = nullptr,
            const void *draft_token_probabilities_device = nullptr) override;
        bool enqueueSpeculativeVerifyProcessedTargetDraftProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
            const void *target_logits_device,
            const void *draft_probabilities_device,
            int row_count,
            int vocab_size,
            int target_row_stride,
            int draft_row_stride,
            const void *draft_tokens_device,
            const float *accept_thresholds_host,
            uint64_t inverse_sample_seed,
            int inverse_sample_first_logical_position,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_accepted_device,
            void *out_accept_probability_device = nullptr,
            void *out_accept_threshold_device = nullptr,
            bool no_draft_probabilities = false,
            const void *threshold_base_position_device = nullptr,
            int threshold_position_offset = 0) override;
        bool enqueueSpeculativeVerifyProcessedTargetDraftLogitsF32DeviceThresholdsBatchDeviceTokens(
            const void *target_logits_device,
            const void *draft_logits_device,
            int row_count,
            int vocab_size,
            int target_row_stride,
            int draft_row_stride,
            const void *draft_tokens_device,
            const float *accept_thresholds_host,
            uint64_t inverse_sample_seed,
            int inverse_sample_first_logical_position,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_accepted_device,
            void *out_accept_probability_device = nullptr,
            void *out_accept_threshold_device = nullptr,
            const void *draft_token_probabilities_device = nullptr) override;
        bool enqueueSpeculativeVerifyProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
            const void *target_probabilities_device,
            const void *draft_probabilities_device,
            const void *inverse_rejection_samples_device,
            int row_count,
            int vocab_size,
            int target_row_stride,
            int draft_row_stride,
            int inverse_sample_row_stride,
            const void *draft_tokens_device,
            const float *accept_thresholds_host,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_accepted_device,
            void *out_accept_probability_device = nullptr,
            void *out_accept_threshold_device = nullptr,
            bool no_draft_probabilities = false) override;
        bool enqueueSummarizeSpeculativeVerifyBatch(
            const void *verify_tokens_device,
            const void *verify_accepted_device,
            int row_count,
            int first_token,
            const int *stop_tokens_host,
            int stop_token_count,
            const void *bonus_token_device,
            bool has_bonus_token,
            int device_id,
            void *stream,
            int out_token_capacity,
            void *out_tokens_device,
            void *out_meta_device,
            const void *max_state_commit_rows_device = nullptr,
            int leading_committed_output_count = 0) override;
        bool enqueueSummarizeSpeculativeVerifyBatchDeviceFirstToken(
            const void *verify_tokens_device,
            const void *verify_accepted_device,
            int row_count,
            const void *first_token_device,
            const int *stop_tokens_host,
            int stop_token_count,
            const void *bonus_token_device,
            bool has_bonus_token,
            int device_id,
            void *stream,
            int out_token_capacity,
            void *out_tokens_device,
            void *out_meta_device,
            const void *max_state_commit_rows_device = nullptr,
            int leading_committed_output_count = 0) override;
        bool enqueueSummarizeSpeculativeVerifyBatchDeviceGenerationControls(
            const void *verify_tokens_device,
            const void *verify_accepted_device,
            const void *greedy_draft_tokens_device,
            int row_count,
            const void *first_token_device,
            const void *stop_tokens_device,
            const void *bonus_token_device,
            bool has_bonus_token,
            const void *generation_control_device,
            int device_id,
            void *stream,
            int out_token_capacity,
            void *out_tokens_device,
            void *out_meta_device) override;
        bool enqueueSampleAndSummarizeSerialEquivalentSpeculativeBatchDeviceGenerationControls(
            const void *target_token_ids_device,
            const void *target_probs_device,
            int target_row_stride,
            int top_k,
            int row_count,
            uint64_t threshold_seed,
            const void *threshold_position_device,
            int threshold_position_offset,
            const void *verifier_input_tokens_device,
            const void *stop_tokens_device,
            const void *generation_control_device,
            int device_id,
            void *stream,
            int out_token_capacity,
            void *sampled_target_tokens_device,
            void *out_tokens_device,
            void *out_meta_device,
            void *first_transaction_diagnostic_device = nullptr) override;
        bool enqueueSummarizeGreedySpeculativeVerifyBatch(
            const void *verify_tokens_device,
            const void *draft_tokens_device,
            int compare_row_count,
            int first_token,
            const int *stop_tokens_host,
            int stop_token_count,
            int device_id,
            void *stream,
            int out_token_capacity,
            void *out_tokens_device,
            void *out_meta_device,
            const void *max_state_commit_rows_device = nullptr,
            int leading_committed_output_count = 0) override;
        bool enqueueSummarizeGreedySpeculativeVerifyBatchDeviceControls(
            const void *verify_tokens_device,
            const void *draft_tokens_device,
            int compare_row_count,
            const void *active_verifier_row_count_device,
            const void *stop_tokens_device,
            int device_id,
            void *stream,
            int out_token_capacity,
            void *out_tokens_device,
            void *out_meta_device,
            const void *max_state_commit_rows_device = nullptr,
            const void *penalty_policy_device = nullptr) override;
        bool enqueueAdvanceSpeculativeCommitBoundary(
            void *meta_device,
            int request_count,
            int meta_stride,
            void *decode_rounds_committed_device,
            void *decode_rounds_until_maintenance_device,
            void *maintenance_due_device,
            void *decode_boundary_advanced_device,
            int device_id,
            void *stream) override;
        bool enqueueInitializeDeviceGeneration(
            int request_count,
            int max_new_tokens,
            const sampling_math::DeviceGenerationDepthPolicy &depth_policy,
            int response_token_stride,
            void *response_tokens_device,
            int control_stride,
            void *control_device,
            int device_id,
            void *stream) override;
        bool enqueuePrepareDeviceGenerationTransactionBudget(
            void *control_device,
            int control_stride,
            int request_count,
            int verifier_row_capacity,
            const void *maintenance_rows_remaining_device,
            int device_id,
            void *stream) override;
        bool enqueueCommitDeviceGenerationAndDeriveSpeculativePublicationMetadata(
            const void *output_tokens_device,
            int output_token_stride,
            void *meta_device,
            int meta_stride,
            const void *base_cached_tokens_device,
            int request_count,
            int padded_state_rows_per_request,
            void *response_tokens_device,
            int response_token_stride,
            void *control_device,
            int control_stride,
            int device_id,
            void *stream,
            void *out_restore_rows_device,
            void *out_target_cached_tokens_device,
            void *out_accepted_state_counts_device,
            void *out_ok_device,
            void *out_next_condition_tokens_device = nullptr,
            void *out_all_drafts_accepted_flags_device = nullptr,
            void *out_stopped_flags_device = nullptr,
            void *out_next_sidecar_condition_tokens_device = nullptr,
            void *out_next_sidecar_position_ids_device = nullptr,
            void *out_next_verifier_condition_tokens_device = nullptr) override;
        bool enqueueDeriveSpeculativePublicationMetadata(
            const void *meta_device,
            int meta_stride,
            const void *base_cached_tokens_device,
            int request_count,
            int padded_state_rows_per_request,
            int max_state_commit_rows,
            int device_id,
            void *stream,
            void *out_restore_rows_device,
            void *out_target_cached_tokens_device,
            void *out_accepted_state_counts_device,
            void *out_ok_device,
            void *out_next_condition_tokens_device = nullptr,
            const void *output_tokens_device = nullptr,
            int output_token_stride = 0,
            void *out_all_drafts_accepted_flags_device = nullptr,
            void *out_stopped_flags_device = nullptr,
            void *out_next_verifier_condition_tokens_device = nullptr) override;
        bool enqueueDeriveShiftedSpeculativePublicationMetadataFromPrimary(
            const void *base_cached_tokens_device,
            const void *main_target_cached_tokens_device,
            const void *main_publication_ok_device,
            int request_count,
            int mtp_depth,
            int device_id,
            void *stream,
            void *out_target_cached_tokens_device,
            void *out_accepted_state_counts_device,
            void *out_ok_device) override;
        bool enqueuePrepareSpeculativeShiftedKVTokens(
            const void *meta_device,
            int meta_stride,
            const void *output_tokens_device,
            int output_token_stride,
            int request_index,
            int first_output_token_index,
            int row_count,
            int32_t filler_token,
            int device_id,
            void *stream,
            void *out_tokens_device,
            const void *base_positions_device,
            int position_offset,
            void *out_position_ids_device) override;
        bool enqueuePrepareMTPBatchedSidecarInputs(
            const void *condition_tokens_device,
            int condition_token_stride,
            const void *base_positions_device,
            int position_offset,
            int request_count,
            int device_id,
            void *stream,
            void *out_condition_tokens_device,
            void *out_position_ids_device) override;
        bool enqueuePrepareMTPVerifierPositionIds(
            const void *base_positions_device,
            int request_count,
            int padded_seq_len,
            int device_id,
            void *stream,
            void *out_position_ids_device) override;
        bool enqueuePrepareMTPVerifierGeometry(
            const void *base_positions_device,
            const void *valid_graph_rows_device,
            int valid_graph_row_count,
            void *generation_control_device,
            int generation_control_stride,
            int request_count,
            int padded_seq_len,
            int device_id,
            void *stream,
            void *out_position_ids_device,
            void *out_request_lengths_device) override;
        bool enqueuePrepareMTPVerifierControlledRow(
            const void *first_token_device,
            const void *draft_tokens_device,
            const void *base_position_device,
            void *generation_control_row_device,
            int generation_control_stride,
            int padded_seq_len,
            int device_id,
            void *stream,
            void *out_tokens_device,
            void *out_position_ids_device,
            void *out_request_length_device,
            void *out_base_position_snapshot_device) override;
        bool enqueueInitializeMTPDeviceLogicalState(
            const void *sampled_tokens_device,
            const void *target_positions_device,
            int request_count,
            int device_id,
            void *stream,
            void *out_base_cached_tokens_device,
            void *out_target_positions_device,
            void *out_accepted_state_counts_device,
            void *out_next_condition_tokens_device,
            void *out_all_drafts_accepted_flags_device,
            void *out_stopped_flags_device,
            void *out_publication_ok_flags_device) override;

        // GPU-side sparse logit penalty application
        bool prepareLogitPenaltyWorkspace(
            int vocab_size,
            int device_id) override;
        bool applyLogitPenaltiesF32(void *logits_device,
                                    const int *token_ids_host,
                                    const float *penalties_host,
                                    int num_penalties, int vocab_size,
                                    int device_id, void *stream) override;
        bool enqueueLogitPenaltiesF32Device(void *logits_device,
                                            const void *token_ids_device,
                                            const void *penalties_device,
                                            int num_penalties, int vocab_size,
                                            int device_id, void *stream) override;

        // Capability queries
        bool supportsBF16(int device_id) const override;
        bool supportsFP16(int device_id) const override;
        bool supportsINT8(int device_id) const override;

        // Compute operations
        bool gemmIQ4NL(
            const void *A_device,
            const void *B_device,
            void *C_device,
            int m,
            int n,
            int k,
            int device_id) override;

        // Stream management
        void *createStream(int device_id) override;
        void destroyStream(void *stream, int device_id) override;
        bool synchronizeStream(void *stream, int device_id) override;
        bool streamWaitEvent(void *stream, void *event, int device_id) override;

        // Async H2D without sync (for pipelined loading)
        bool hostToDeviceOnStream(void *dst, const void *src, size_t bytes,
                                  int device_id, void *stream) override;
        bool deviceToHostOnStream(void *dst, const void *src, size_t bytes,
                                  int device_id, void *stream) override;

        // Pinned host memory
        void *allocatePinned(size_t bytes, int device_id) override;
        void freePinned(void *ptr, int device_id) override;

        // Stream-aware memory operations
        bool deviceCopyAsync(void *dst, const void *src, size_t bytes,
                             int device_id, void *stream) override;

        // Collective reduction primitives
        bool vectorAddInplace(void *output, const void *input, size_t count,
                              int element_size, int device_id, void *stream) override;

        // Backend identity
        DeviceType backendDeviceType() const override { return DeviceType::CUDA; }

    private:
        int device_count_;

        // Per-device argmax result buffers (lazily allocated)
        struct ArgmaxDeviceBuffers
        {
            void *value_ptr = nullptr;
            void *index_ptr = nullptr;
            int allocated_count = 0;
        };
        std::vector<ArgmaxDeviceBuffers> argmax_buffers_;

        // Per-device top-k result buffers (lazily allocated)
        struct TopKDeviceBuffers
        {
            void *values_ptr = nullptr;
            void *indices_ptr = nullptr;
            int allocated_k = 0;
        };
        std::vector<TopKDeviceBuffers> topk_buffers_;

        // Per-device sampled-token result buffers (lazily allocated)
        struct SampleTokenDeviceBuffers
        {
            void *token_ptr = nullptr; // int on device
        };
        std::vector<SampleTokenDeviceBuffers> sample_token_buffers_;

        /**
         * @brief Persistent per-device sparse-penalty publication storage.
         *
         * The first call reserves a full-vocabulary token/value pair so later
         * decode steps cannot reallocate in the hot path. `ready_event` orders
         * reuse when successive logits producers use different streams.
         */
        struct PenaltyDeviceBuffers
        {
            void *token_ids_ptr = nullptr;  ///< int[allocated_count] on device.
            void *penalties_ptr = nullptr;  ///< float[allocated_count] on device.
            int allocated_count = 0;
            void *ready_event = nullptr;    ///< Event recorded after the penalty kernel.
            void *producer_stream = nullptr; ///< Stream owning the newest publication.
            bool publication_valid = false;
        };
        std::vector<PenaltyDeviceBuffers> penalty_buffers_;
    };

} // namespace llaminar2
