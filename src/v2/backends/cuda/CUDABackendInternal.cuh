/**
 * @file CUDABackendInternal.cuh
 * @brief Shared internal CUDA error-checking macros, explicit-stream validation, and sampling extern "C" declarations.
 */
#pragma once

#include "CUDABackend.h"
#include "CUDAGraphCapture.h"
#include "CUDAHardwareMetadata.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/moe/DeviceMoERebalanceABI.h"
#include "execution/mtp/MTPVerifierOutcomeGraph.h"
#include "kernels/common/GenerationLogicalState.h"
#include "kernels/common/SamplingMath.h"
#include "utils/Logger.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace llaminar2
{
using sampling_math::kSpeculativeBatchMaxStopTokens;
using sampling_math::kSpeculativeBatchMetaCount;
    // ====================================================================
    // Helper Macros for CUDA Error Checking
    // ====================================================================
    //
    // CUDA_CHECK_OR_THROW: Use for hot-path control flow (success-path API
    // calls in compute kernels) where silent failure causes silent miscompute
    // or delayed segfault. Logs at ERROR and throws std::runtime_error with
    // file/line context.
    //
    // CUDA_WARN_IF_FAIL: Use for cleanup/destructor/rollback paths (free,
    // destroy, error-recovery after upstream failure, resource-clear-before-
    // reuse). Logs at WARN and continues. Throwing here would call
    // std::terminate from destructors on stack unwind, or mask the real
    // upstream failure when called during error rollback.
    //
    // cudaErrorCudartUnloading is silenced because it is expected during
    // process exit when the CUDA runtime tears down before our cleanup runs.
#define CUDA_CHECK_OR_THROW(call)                                                          \
    do                                                                                     \
    {                                                                                      \
        cudaError_t _err = (call);                                                         \
        if (_err != cudaSuccess)                                                           \
        {                                                                                  \
            std::ostringstream _oss;                                                       \
            _oss << "[CUDABackend] " << #call << " failed: "                               \
                 << cudaGetErrorString(_err) << " (" << __FILE__ << ":" << __LINE__ << ")";\
            LOG_ERROR(_oss.str());                                                         \
            throw std::runtime_error(_oss.str());                                          \
        }                                                                                  \
    } while (0)

#define CUDA_WARN_IF_FAIL(call)                                                            \
    do                                                                                     \
    {                                                                                      \
        cudaError_t _err = (call);                                                         \
        if (_err != cudaSuccess)                                                           \
        {                                                                                  \
            if (_err == cudaErrorCudartUnloading)                                          \
            {                                                                              \
                LOG_TRACE("[CUDABackend] " << #call                                        \
                                           << " skipped: CUDA runtime shutting down");    \
            }                                                                              \
            else                                                                           \
            {                                                                              \
                LOG_WARN("[CUDABackend] " << #call << " failed: "                          \
                                          << cudaGetErrorString(_err) << " ("             \
                                          << __FILE__ << ":" << __LINE__ << ")");         \
            }                                                                              \
        }                                                                                  \
    } while (0)


    // ====================================================================
    // Stream Resolution Helper
    // ====================================================================

    /**
     * @brief Convert an opaque execution stream after enforcing explicit ownership.
     *
     * CUDA's null stream is process-global scheduling state, not a harmless
     * default. Accepting it here would erase the producer/consumer ordering
     * expressed by the graph. Every executable backend API therefore fails at
     * this common boundary before it can enqueue work ambiguously.
     */
    static cudaStream_t requireExplicitStream(void *stream, const char *operation)
    {
        if (!stream)
        {
            throw std::invalid_argument(
                std::string(operation ? operation : "CUDABackend operation") +
                " requires an explicit non-null CUDA stream");
        }
        return static_cast<cudaStream_t>(stream);
    }


    // Forward declarations for CUDA sampling kernels (CUDASamplingKernels.cu)
    extern "C" bool cudaOps_argmax_f32(
        const float *data, int n, float *out_value, int *out_index,
        float *partial_vals, int *partial_idxs, int partial_capacity,
        int device_idx, void *stream);
    extern "C" bool cudaOps_argmax_f32_batched_rows(
        const float *data, int rows, int cols, int row_stride,
        float *out_values, int *out_indices,
        float *partial_vals, int *partial_idxs, int partial_capacity,
        int device_idx, void *stream, int output_stride);
    extern "C" bool cudaOps_argmax_f32_batched_rows_publish_mtp_chain(
        const float *data, int rows, int cols, int row_stride,
        float *out_values, int *out_indices,
        int *chain_condition_tokens, int *chain_position_ids,
        int chain_position_increment,
        float *partial_vals, int *partial_idxs, int partial_capacity,
        int device_idx, void *stream, int output_stride);
    extern "C" bool cudaOps_retain_mtp_first_transaction_draft_boundary(
        const uint32_t *data_words,
        int word_count,
        int boundary,
        int draft_slot,
        const int *condition_token,
        const int *position_id,
        const int *generation_control,
        int generation_control_stride,
        void *diagnostic_record,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_configure_mtp_greedy_penalty_policy(
        MTPGreedyPenaltyPolicy *controls,
        float presence_penalty,
        float frequency_penalty,
        bool first_token_already_in_history,
        int device_idx,
        void *stream, float repetition_penalty);
    extern "C" bool cudaOps_argmax_f32_batched_rows_mtp_penalties(
        const float *data, int rows, int cols, int row_stride,
        const int *verifier_input_tokens,
        const int *generated_token_counts,
        const MTPGreedyPenaltyPolicy *policy,
        const int *active_rows,
        float *out_values, int *out_indices,
        float *partial_vals, int *partial_idxs, int partial_capacity,
        int device_idx, void *stream, int output_stride);
    extern "C" bool cudaOps_apply_mtp_penalties_f32_rows(
        float *data, int rows, int cols, int row_stride,
        const int *verifier_input_tokens,
        const int *generated_token_counts,
        const MTPGreedyPenaltyPolicy *policy,
        const int *active_rows,
        int device_idx, void *stream);
    extern "C" bool cudaOps_apply_mtp_branch_penalties_f32_row(
        float *data, int cols,
        const int *first_condition_token,
        const int *prior_draft_tokens,
        int prior_draft_count,
        const int *generated_token_counts,
        const MTPGreedyPenaltyPolicy *policy,
        int device_idx, void *stream);
    extern "C" bool cudaOps_commit_mtp_greedy_penalty_history(
        const int *output_tokens,
        const int *output_meta,
        MTPGreedyPenaltyPolicy *policy,
        const int *accepted_state_counts,
        const int *stopped_flags,
        int output_token_capacity,
        int vocab_size,
        int *generated_token_counts,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_commit_generation_token_history(
        const int *token,
        int vocab_size,
        int *generated_token_counts,
        MTPGreedyPenaltyPolicy *direct_output_policy,
        int device_idx,
        void *stream);

    extern "C" bool cudaOps_topk_f32(
        const float *data, int n, int k, float *out_values, int *out_indices,
        int device_idx, void *stream);
    extern "C" bool cudaOps_sample_topk_topp_f32(
        const float *data, int n, int k, float top_p, float temperature,
        unsigned long long rng_seed, unsigned long long rng_offset,
        int *out_token, int device_idx, void *stream);
    extern "C" bool cudaOps_publish_int32_control_scalar(
        int32_t value,
        int32_t *out_value,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_topk_topp_distribution_f32(
        const float *data, int n, int k, float top_p, float temperature,
        int *out_token_ids, float *out_probs,
        float *scratch_values, int *scratch_indices, int scratch_capacity,
        int device_idx, void *stream);
    extern "C" bool cudaOps_topk_topp_distributions_f32(
        const float *data, int row_count, int n, int row_stride, int k,
        float top_p, float temperature,
        int *out_token_ids, int out_stride, float *out_probs,
        float *scratch_values, int *scratch_indices, int scratch_capacity,
        const int *active_rows,
        int device_idx, void *stream);
    extern "C" bool cudaOps_topk_topp_distributions_f32_with_history(
        const float *data, int row_count, int n, int row_stride, int k,
        float top_p, float temperature,
        int *out_token_ids, int out_stride, float *out_probs,
        float *scratch_values, int *scratch_indices, int scratch_capacity,
        const int *active_rows,
        const int *counts, const MTPGreedyPenaltyPolicy *policy, const int *tokens,
        int device_idx, void *stream);
    extern "C" bool cudaOps_topk_topp_processed_logits_f32(
        const float *data, int row_count, int n, int row_stride, int k,
        float top_p, float temperature,
        float *out_logits, int out_row_stride,
        float *scratch_values, int *scratch_indices, int scratch_capacity,
        int device_idx, void *stream);
    extern "C" bool cudaOps_speculative_verify_distribution_f32(
        const int *target_token_ids, const float *target_probs,
        const int *draft_token_ids, const float *draft_probs,
        int k, int draft_token,
        unsigned long long accept_seed, unsigned long long accept_offset,
        unsigned long long residual_seed, unsigned long long residual_offset,
        int *out_token, int *out_accepted,
        float *out_accept_probability, float *out_accept_threshold,
        int device_idx, void *stream);
    extern "C" bool cudaOps_sample_distribution_f32(
        const int *token_ids, const float *probs,
        int k, float threshold,
        int *out_token, float *out_probability,
        unsigned long long threshold_seed,
        const int *threshold_position,
        int threshold_position_offset,
        int device_idx, void *stream, const uint64_t *threshold_seed_device);
    extern "C" bool cudaOps_sample_processed_logits_f32(
        const float *logits,
        int vocab_size,
        int row_stride,
        float threshold,
        int *out_token,
        float *out_probability,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_sample_processed_logits_if_speculative_batch_needs_bonus_f32(
        const float *logits,
        int vocab_size,
        int row_stride,
        float threshold,
        const int *verify_tokens,
        const int *verify_accepted,
        int row_count,
        int first_token,
        const int *first_token_device,
        int stop_token0,
        int stop_token1,
        int stop_token2,
        int stop_token3,
        int stop_token4,
        int stop_token5,
        int stop_token6,
        int stop_token7,
        int stop_token_count,
        int *out_token,
        float *out_probability,
        unsigned long long threshold_seed,
        const int *threshold_position,
        int threshold_position_offset,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_softmax_processed_logits_f32(
        const float *logits,
        int row_count,
        int vocab_size,
        int row_stride,
        float *out_probabilities,
        int out_row_stride,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_softmax_sample_temperature_logits_f32(
        const float *logits,
        int vocab_size,
        int row_stride,
        float temperature,
        float threshold,
        float *out_probabilities,
        int out_row_stride,
        int *out_token,
        float *out_probability,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_scale_sample_temperature_logits_f32(
        const float *logits,
        int vocab_size,
        int row_stride,
        float temperature,
        float threshold,
        float *out_logits,
        int out_row_stride,
        int *out_token,
        float *out_probability,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_fill_inverse_exponential_samples_f32(
        float *out_samples,
        int row_count,
        int vocab_size,
        int row_stride,
        unsigned long long seed,
        int first_logical_position,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_speculative_verify_distribution_threshold_f32(
        const int *target_token_ids, const float *target_probs,
        const int *draft_token_ids, const float *draft_probs,
        int k, int draft_token,
        float accept_threshold, float residual_threshold,
        int *out_token, int *out_accepted,
        float *out_accept_probability, float *out_accept_threshold,
        int device_idx, void *stream);
    extern "C" bool cudaOps_speculative_verify_distribution_thresholds_batch_f32(
        const int *target_token_ids, const float *target_probs,
        const int *draft_token_ids, const float *draft_probs,
        int k, int distribution_stride,
        const int *draft_tokens_host,
        const float *accept_thresholds_host,
        const float *residual_thresholds_host,
        int row_count,
        int *out_token,
        int *out_accepted,
        float *out_accept_probability,
        float *out_accept_threshold,
        int device_idx, void *stream);
    extern "C" bool cudaOps_speculative_verify_distribution_thresholds_batch_device_tokens_f32(
        const int *target_token_ids, const float *target_probs,
        const int *draft_token_ids, const float *draft_probs,
        int k, int distribution_stride,
        const int *sampled_draft_tokens,
        const float *sampled_draft_probabilities,
        const float *accept_thresholds_host,
        const float *residual_thresholds_host,
        int row_count,
        unsigned long long inverse_sample_seed,
        int inverse_sample_first_logical_position,
        int inverse_sample_vocab_size,
        unsigned long long threshold_seed,
        int threshold_first_logical_position,
        int thresholds_from_seed,
        const int *threshold_base_position,
        int threshold_position_offset,
        const int *generation_control,
        int *out_token,
        int *out_accepted,
        float *out_accept_probability,
        float *out_accept_threshold,
        int device_idx, void *stream, const uint64_t *threshold_seed_device);
    extern "C" bool cudaOps_speculative_verify_processed_logits_thresholds_batch_device_tokens_f32(
        const float *target_logits,
        const float *draft_logits,
        int row_count,
        int vocab_size,
        int target_row_stride,
        int draft_row_stride,
        const int *sampled_draft_tokens,
        const float *accept_thresholds_host,
        const float *residual_thresholds_host,
        int *out_token,
        int *out_accepted,
        float *out_accept_probability,
        float *out_accept_threshold,
        const float *draft_token_probabilities,
        int device_idx, void *stream);
    extern "C" bool cudaOps_speculative_verify_processed_target_draft_probabilities_thresholds_batch_device_tokens_f32(
        const float *target_logits,
        const float *draft_probabilities,
        int row_count,
        int vocab_size,
        int target_row_stride,
        int draft_row_stride,
        const int *sampled_draft_tokens,
        const float *accept_thresholds_host,
        unsigned long long inverse_sample_seed,
        int inverse_sample_first_logical_position,
        int thresholds_from_seed,
        const int *threshold_base_position,
        int threshold_position_offset,
        int *out_token,
        int *out_accepted,
        float *out_accept_probability,
        float *out_accept_threshold,
        int no_draft_probabilities,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_speculative_verify_processed_target_draft_logits_thresholds_batch_device_tokens_f32(
        const float *target_logits,
        const float *draft_logits,
        int row_count,
        int vocab_size,
        int target_row_stride,
        int draft_row_stride,
        const int *sampled_draft_tokens,
        const float *sampled_draft_probabilities,
        const float *accept_thresholds_host,
        unsigned long long inverse_sample_seed,
        int inverse_sample_first_logical_position,
        int *out_token,
        int *out_accepted,
        float *out_accept_probability,
        float *out_accept_threshold,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_speculative_verify_probabilities_thresholds_batch_device_tokens_f32(
        const float *target_probabilities,
        const float *draft_probabilities,
        const float *inverse_rejection_samples,
        int row_count,
        int vocab_size,
        int target_row_stride,
        int draft_row_stride,
        int inverse_sample_row_stride,
        const int *sampled_draft_tokens,
        const float *accept_thresholds_host,
        int no_draft_probabilities,
        int *out_token,
        int *out_accepted,
        float *out_accept_probability,
        float *out_accept_threshold,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_summarize_speculative_verify_batch(
        const int *verify_tokens,
        const int *verify_accepted,
        int row_count,
        int first_token,
        int stop_token0, int stop_token1, int stop_token2, int stop_token3,
        int stop_token4, int stop_token5, int stop_token6, int stop_token7,
        int stop_token_count,
        const int *bonus_token,
        int has_bonus_token,
        int *out_tokens,
        int out_token_capacity,
        int *out_meta,
        const uint32_t *max_state_commit_rows,
        int leading_committed_output_count,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_summarize_speculative_verify_batch_device_first_token(
        const int *verify_tokens,
        const int *verify_accepted,
        int row_count,
        const int *first_token,
        int stop_token0, int stop_token1, int stop_token2, int stop_token3,
        int stop_token4, int stop_token5, int stop_token6, int stop_token7,
        int stop_token_count,
        const int *bonus_token,
        int has_bonus_token,
        int *out_tokens,
        int out_token_capacity,
        int *out_meta,
        const uint32_t *max_state_commit_rows,
        int leading_committed_output_count,
        int device_idx,
        void *stream);
    extern "C" bool
    cudaOps_summarize_speculative_verify_batch_device_generation_controls(
        const int *verify_tokens,
        const int *verify_accepted,
        const int *greedy_draft_tokens,
        int row_count,
        const int *first_token,
        const int *stop_tokens,
        const int *bonus_token,
        int has_bonus_token,
        const int *generation_control,
        int *out_tokens,
        int out_token_capacity,
        int *out_meta,
        int device_idx,
        void *stream);
    extern "C" bool
    cudaOps_sample_and_summarize_serial_equivalent_speculative_batch_device_generation_controls(
        const int *target_token_ids,
        const float *target_probs,
        int target_row_stride,
        int top_k,
        int row_count,
        unsigned long long threshold_seed,
        const int *threshold_position,
        int threshold_position_offset,
        const int *verifier_input_tokens,
        const int *stop_tokens,
        const int *generation_control,
        int *sampled_target_tokens,
        int *out_tokens,
        int out_token_capacity,
        int *out_meta,
        void *first_transaction_diagnostic,
        int device_idx,
        void *stream, const uint64_t *threshold_seed_device);
    extern "C" bool cudaOps_summarize_greedy_speculative_verify_batch(
        const int *verify_tokens,
        const int *draft_tokens,
        int compare_row_count,
        int first_token,
        int stop_token0, int stop_token1, int stop_token2, int stop_token3,
        int stop_token4, int stop_token5, int stop_token6, int stop_token7,
        int stop_token_count,
        int *out_tokens,
        int out_token_capacity,
        int *out_meta,
        const uint32_t *max_state_commit_rows,
        int leading_committed_output_count,
        int device_idx,
        void *stream);
    extern "C" bool
    cudaOps_summarize_greedy_speculative_verify_batch_device_controls(
        const int *verify_tokens,
        const int *draft_tokens,
        int compare_row_count,
        const int *active_verifier_row_count,
        const int *stop_tokens,
        int *out_tokens,
        int out_token_capacity,
        int *out_meta,
        const uint32_t *max_state_commit_rows,
        const int *next_leading_committed_output_count,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_advance_speculative_commit_boundary(
        const int *meta,
        int request_count,
        int meta_stride,
        uint32_t *decode_rounds_committed,
        uint32_t *decode_rounds_until_maintenance,
        uint32_t *maintenance_due,
        uint32_t *decode_boundary_advanced,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_publish_serial_decode_commit_boundary(
        uint32_t *decode_rounds_committed,
        uint32_t *decode_rounds_until_maintenance,
        uint32_t *maintenance_due,
        uint32_t *decode_boundary_advanced,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_acknowledge_decode_commit_boundary(
        uint32_t *decode_rounds_until_maintenance,
        uint32_t *maintenance_due,
        uint32_t *decode_boundary_advanced,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_initialize_device_moe_rebalance_dispatch_ticket(
        uint64_t session_epoch,
        uint64_t workspace_generation,
        uint32_t participant_id,
        uint32_t participant_count,
        DeviceMoERebalanceDispatchTicket *ticket,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_publish_device_moe_rebalance_dispatch_ticket(
        const uint32_t *controller_magic,
        const uint32_t *controller_version,
        const uint32_t *controller_error,
        const uint32_t *decode_rounds_committed,
        const uint32_t *decode_rounds_until_maintenance,
        const uint32_t *maintenance_due,
        const uint32_t *decode_boundary_advanced,
        DeviceMoERebalanceDispatchTicket *ticket,
        int device_idx,
        void *stream);
    /** @brief Exact-stream ordinary publication bridge implemented by the sampling TU. */
    extern "C" bool cudaOps_publish_ordinary_generation_sample(
        const sampling_math::OrdinaryGenerationPublication &publication,
        int device_idx, void *stream);

    extern "C" bool cudaOps_initialize_device_generation(
        int request_count,
        int max_new_tokens,
        const sampling_math::DeviceGenerationPolicy &depth_policy,
        sampling_math::DeviceGenerationLeadingRowDisposition
            initial_leading_row_disposition,
        int response_token_stride,
        int control_stride,
        int *control,
        int device_idx,
        void *stream,
        const sampling_math::DeviceGenerationInitialization &initialization);
    extern "C" bool cudaOps_initialize_device_generation_dispatch_tickets(
        uint64_t session_epoch,
        uint64_t workspace_generation,
        int *control,
        int control_stride,
        int request_count,
        sampling_math::DeviceGenerationDispatchTicket *tickets,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_publish_device_generation_dispatch_tickets(
        int *control,
        int control_stride,
        int request_count,
        const uint32_t *maintenance_due,
        sampling_math::DeviceGenerationDispatchTicket *tickets,
        int device_idx,
        void *stream);
    extern "C" bool
    cudaMoE_publish_current_batch_llep_evidence_to_generation_control(
        const void *runtime_layers,
        int layer_count,
        int *generation_control,
        int generation_control_stride,
        int request_count,
        int movement_layer_count_index,
        int non_owner_assignment_layer_count_index,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_prepare_device_generation_transaction_budget(
        int *control,
        int control_stride,
        int request_count,
        int verifier_row_capacity,
        const uint32_t *maintenance_rows_remaining,
        const uint32_t *maintenance_due,
        uint32_t *decode_boundary_advanced,
        int device_idx,
        void *stream);
    extern "C" bool
    cudaOps_commit_device_generation_and_derive_speculative_publication_metadata(
        const int32_t *compact_tokens,
        int output_token_stride,
        int *compact_meta,
        int meta_stride,
        const int *base_cached_tokens,
        int request_count,
        int padded_state_rows_per_request,
        int32_t *response_tokens,
        int response_token_stride,
        int *control,
        int control_stride,
        int *out_restore_rows,
        int *out_target_cached_tokens,
        int *out_accepted_state_counts,
        int *out_ok,
        int *out_next_condition_tokens,
        int *out_all_drafts_accepted_flags,
        int *out_stopped_flags,
        int *out_next_sidecar_condition_tokens,
        int *out_next_sidecar_position_ids,
        int *out_next_verifier_condition_tokens,
        const int32_t *verifier_input_tokens,
        int verifier_input_token_stride,
        sampling_math::MTPCommittedVerifierIdentityRecord *
            out_committed_verifier_identity,
        sampling_math::MTPDepthTransactionTrace *depth_transaction_trace,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_derive_speculative_publication_metadata(
        const int *meta,
        int meta_stride,
        const int *base_cached_tokens,
        int request_count,
        int padded_state_rows_per_request,
        int max_state_commit_rows,
        int *out_restore_rows,
        int *out_target_cached_tokens,
        int *out_accepted_state_counts,
        int *out_ok,
        int *out_next_condition_tokens,
        const int32_t *output_tokens,
        int output_token_stride,
        int *out_all_drafts_accepted_flags,
        int *out_stopped_flags,
        int *out_next_verifier_condition_tokens,
        int device_idx,
        void *stream);
    extern "C" bool
    cudaOps_derive_shifted_speculative_publication_metadata_from_primary(
        const int *base_cached_tokens,
        const int *main_target_cached_tokens,
        const int *main_publication_ok,
        int request_count,
        int mtp_depth,
        int *out_target_cached_tokens,
        int *out_accepted_state_counts,
        int *out_ok,
        int device_idx,
        void *stream, llaminar2::MTPShiftedCacheOrigin origin);
    extern "C" bool cudaOps_prepare_speculative_shifted_kv_tokens(
        const int *meta,
        int meta_stride,
        const int32_t *output_tokens,
        int output_token_stride,
        int request_index,
        int first_output_token_index,
        int row_count,
        int32_t filler_token,
        int32_t *out_tokens,
        const int32_t *base_positions,
        int position_offset,
        int32_t *out_position_ids,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_prepare_mtp_batched_sidecar_inputs(
        const int32_t *condition_tokens,
        int condition_token_stride,
        const int32_t *base_positions,
        int position_offset,
        int request_count,
        int32_t *out_condition_tokens,
        int32_t *out_position_ids,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_prepare_mtp_verifier_position_ids(
        const int32_t *base_positions,
        int request_count,
        int padded_seq_len,
        int32_t *out_position_ids,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_prepare_mtp_verifier_geometry(
        const int32_t *base_positions,
        const int32_t *valid_graph_rows,
        int valid_graph_row_count,
        int *generation_control,
        int generation_control_stride,
        int request_count,
        int padded_seq_len,
        int32_t *out_position_ids,
        int32_t *out_request_lengths,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_prepare_mtp_verifier_controlled_row(
        const int32_t *first_token,
        const int32_t *draft_tokens,
        const int32_t *base_position,
        int *generation_control_row,
        int generation_control_stride,
        int padded_seq_len,
        int32_t *out_tokens,
        int32_t *out_position_ids,
        int32_t *out_request_length,
        int32_t *out_base_position_snapshot,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_initialize_generation_logical_state(
        const GenerationLogicalStateInitialization &initialization, int device_idx, void *stream);

} // namespace llaminar2
