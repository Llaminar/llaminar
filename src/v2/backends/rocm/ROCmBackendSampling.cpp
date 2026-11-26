/**
 * @file ROCmBackendSampling.cpp
 * @brief ROCm backend device-side sampling, speculative verification, generation control, and logit penalty wrappers.
 */
#include "ROCmBackendInternal.h"

namespace llaminar2
{
    bool ROCmBackend::enqueueBuildTopKTopPDistributionF32Device(
        const void *data_device,
        int n,
        int top_k,
        float top_p,
        float temperature,
        int device_id,
        void *stream,
        void *out_token_ids_device,
        void *out_probs_device,
        void *scratch_values_device,
        void *scratch_indices_device,
        int scratch_capacity)
    {
        if (device_id >= device_count_ || device_id < 0 || !data_device ||
            n <= 0 || top_k <= 0 || !stream || !out_token_ids_device || !out_probs_device)
        {
            return false;
        }

        if (top_k > 256)
            top_k = 256;
        if (top_k > n)
            top_k = n;

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_topk_topp_distribution_f32(
            static_cast<const float *>(data_device),
            n,
            top_k,
            top_p,
            temperature,
            static_cast<int *>(out_token_ids_device),
            static_cast<float *>(out_probs_device),
            static_cast<float *>(scratch_values_device),
            static_cast<int *>(scratch_indices_device),
            scratch_capacity,
            device_id,
            stream);
    }

    bool ROCmBackend::enqueueBuildTopKTopPDistributionsF32Device(
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
        void *scratch_values_device,
        void *scratch_indices_device,
        int scratch_capacity,
        const void *active_rows_device)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !data_device || row_count <= 0 || n <= 0 || row_stride < n ||
            top_k <= 0 || out_stride <= 0 || out_stride < top_k ||
            !stream || !out_token_ids_device || !out_probs_device)
        {
            return false;
        }

        if (top_k > 256)
            top_k = 256;
        if (top_k > n)
            top_k = n;

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_topk_topp_distributions_f32(
            static_cast<const float *>(data_device),
            row_count,
            n,
            row_stride,
            top_k,
            top_p,
            temperature,
            static_cast<int *>(out_token_ids_device),
            out_stride,
            static_cast<float *>(out_probs_device),
            static_cast<float *>(scratch_values_device),
            static_cast<int *>(scratch_indices_device),
            scratch_capacity,
            static_cast<const int *>(active_rows_device),
            device_id,
            stream);
    }

    bool ROCmBackend::enqueueBuildTopKTopPDistributionsF32WithHistoryDevice(
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
        void *scratch_values_device,
        void *scratch_indices_device,
        int scratch_capacity,
        const GenerationPenaltyHistory &history)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !history.admitsRows(row_count) ||
            !data_device || row_count <= 0 || n <= 0 || row_stride < n ||
            top_k <= 0 || top_k > 256 || top_k > n || out_stride <= 0 || out_stride < top_k ||
            !stream || !out_token_ids_device || !out_probs_device)
        {
            return false;
        }

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_topk_topp_distributions_f32_with_history(
            static_cast<const float *>(data_device),
            row_count,
            n,
            row_stride,
            top_k,
            top_p,
            temperature,
            static_cast<int *>(out_token_ids_device),
            out_stride,
            static_cast<float *>(out_probs_device),
            static_cast<float *>(scratch_values_device),
            static_cast<int *>(scratch_indices_device),
            scratch_capacity,
            static_cast<const int *>(history.activeRows()),
            static_cast<const int *>(history.counts()),
            static_cast<const MTPGreedyPenaltyPolicy *>(history.policy()),
            static_cast<const int *>(history.branchTokens()),
            device_id,
            stream);
    }

    bool ROCmBackend::enqueueBuildTopKTopPProcessedLogitsF32Device(
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
        void *scratch_values_device,
        void *scratch_indices_device,
        int scratch_capacity)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !data_device || row_count <= 0 || n <= 0 ||
            row_stride < n || out_row_stride < n ||
            top_k <= 0 || !stream || !out_logits_device)
        {
            return false;
        }

        if (top_k > 256)
            top_k = 256;
        if (top_k > n)
            top_k = n;

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_topk_topp_processed_logits_f32(
            static_cast<const float *>(data_device),
            row_count,
            n,
            row_stride,
            top_k,
            top_p,
            temperature,
            static_cast<float *>(out_logits_device),
            out_row_stride,
            static_cast<float *>(scratch_values_device),
            static_cast<int *>(scratch_indices_device),
            scratch_capacity,
            device_id,
            stream);
    }

    bool ROCmBackend::enqueueSpeculativeVerifyDistributionsF32Device(
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
        void *out_accept_probability_device,
        void *out_accept_threshold_device)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !target_token_ids_device || !target_probs_device ||
            !draft_token_ids_device || !draft_probs_device ||
            top_k <= 0 || top_k > 256 || !stream || !out_token_device || !out_accepted_device)
        {
            return false;
        }

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_speculative_verify_distribution_f32(
            static_cast<const int *>(target_token_ids_device),
            static_cast<const float *>(target_probs_device),
            static_cast<const int *>(draft_token_ids_device),
            static_cast<const float *>(draft_probs_device),
            top_k,
            draft_token,
            static_cast<unsigned long long>(accept_seed),
            static_cast<unsigned long long>(accept_offset),
            static_cast<unsigned long long>(residual_seed),
            static_cast<unsigned long long>(residual_offset),
            static_cast<int *>(out_token_device),
            static_cast<int *>(out_accepted_device),
            static_cast<float *>(out_accept_probability_device),
            static_cast<float *>(out_accept_threshold_device),
            device_id,
            stream);
    }

    bool ROCmBackend::enqueueSampleDistributionF32Device(
        const void *token_ids_device,
        const void *probs_device,
        int top_k,
        float threshold,
        int device_id,
        void *stream,
        void *out_token_device,
        void *out_probability_device,
        uint64_t threshold_seed,
        const void *threshold_position_device,
        int threshold_position_offset,
        const uint64_t *threshold_seed_device)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !token_ids_device || !probs_device ||
            top_k <= 0 || top_k > 256 || !stream || !out_token_device ||
            (threshold_position_device && threshold_seed == 0 && !threshold_seed_device) ||
            (threshold_seed_device && (!threshold_position_device || threshold_seed != 0)))
        {
            return false;
        }

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_sample_distribution_f32(
            static_cast<const int *>(token_ids_device),
            static_cast<const float *>(probs_device),
            top_k,
            threshold,
            static_cast<int *>(out_token_device),
            static_cast<float *>(out_probability_device),
            static_cast<unsigned long long>(threshold_seed),
            static_cast<const int *>(threshold_position_device),
            threshold_position_offset,
            device_id,
            stream,
            threshold_seed_device);
    }

    bool ROCmBackend::enqueueSampleProcessedLogitsF32Device(
        const void *logits_device,
        int vocab_size,
        int row_stride,
        float threshold,
        int device_id,
        void *stream,
        void *out_token_device,
        void *out_probability_device)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !logits_device || vocab_size <= 0 || row_stride < vocab_size ||
            !stream || !out_token_device)
        {
            return false;
        }

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_sample_processed_logits_f32(
            static_cast<const float *>(logits_device),
            vocab_size,
            row_stride,
            threshold,
            static_cast<int *>(out_token_device),
            static_cast<float *>(out_probability_device),
            device_id,
            stream);
    }

    bool ROCmBackend::enqueueSampleProcessedLogitsF32DeviceIfSpeculativeBatchNeedsBonus(
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
        void *out_probability_device,
        uint64_t threshold_seed,
        const void *threshold_position_device,
        int threshold_position_offset)
    {
        using namespace sampling_math;
        if (device_id >= device_count_ || device_id < 0 ||
            !logits_device || vocab_size <= 0 || row_stride < vocab_size ||
            !verify_tokens_device || !verify_accepted_device ||
            row_count < 0 ||
            (first_token < 0 && !first_token_device) ||
            stop_token_count < 0 ||
            stop_token_count > kSpeculativeBatchMaxStopTokens ||
            (stop_token_count > 0 && !stop_tokens_host) ||
            !stream || !out_token_device ||
            (threshold_position_device && threshold_seed == 0))
        {
            return false;
        }

        int stop_tokens[kSpeculativeBatchMaxStopTokens] =
            {-1, -1, -1, -1, -1, -1, -1, -1};
        for (int i = 0; i < stop_token_count; ++i)
            stop_tokens[i] = stop_tokens_host[i];

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_sample_processed_logits_if_speculative_batch_needs_bonus_f32(
            static_cast<const float *>(logits_device),
            vocab_size,
            row_stride,
            threshold,
            static_cast<const int *>(verify_tokens_device),
            static_cast<const int *>(verify_accepted_device),
            row_count,
            first_token,
            static_cast<const int *>(first_token_device),
            stop_tokens[0],
            stop_tokens[1],
            stop_tokens[2],
            stop_tokens[3],
            stop_tokens[4],
            stop_tokens[5],
            stop_tokens[6],
            stop_tokens[7],
            stop_token_count,
            static_cast<int *>(out_token_device),
            static_cast<float *>(out_probability_device),
            static_cast<unsigned long long>(threshold_seed),
            static_cast<const int *>(threshold_position_device),
            threshold_position_offset,
            device_id,
            stream);
    }

    bool ROCmBackend::enqueueSoftmaxAndSampleTemperatureLogitsF32Device(
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
        void *out_probability_device)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !logits_device || vocab_size <= 0 || row_stride < vocab_size ||
            out_row_stride < vocab_size || !stream ||
            !out_probabilities_device || !out_token_device)
        {
            return false;
        }

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_softmax_sample_temperature_logits_f32(
            static_cast<const float *>(logits_device),
            vocab_size,
            row_stride,
            temperature,
            threshold,
            static_cast<float *>(out_probabilities_device),
            out_row_stride,
            static_cast<int *>(out_token_device),
            static_cast<float *>(out_probability_device),
            device_id,
            stream);
    }

    bool ROCmBackend::enqueueScaleAndSampleTemperatureLogitsF32Device(
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
        void *out_probability_device)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !logits_device || vocab_size <= 0 || row_stride < vocab_size ||
            out_row_stride < vocab_size || !stream ||
            !out_logits_device || !out_token_device)
        {
            return false;
        }

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_scale_sample_temperature_logits_f32(
            static_cast<const float *>(logits_device),
            vocab_size,
            row_stride,
            temperature,
            threshold,
            static_cast<float *>(out_logits_device),
            out_row_stride,
            static_cast<int *>(out_token_device),
            static_cast<float *>(out_probability_device),
            device_id,
            stream);
    }

    bool ROCmBackend::enqueueSoftmaxProcessedLogitsF32Device(
        const void *logits_device,
        int row_count,
        int vocab_size,
        int row_stride,
        int device_id,
        void *stream,
        void *out_probabilities_device,
        int out_row_stride)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !logits_device || row_count <= 0 || vocab_size <= 0 ||
            row_stride < vocab_size || out_row_stride < vocab_size ||
            !stream || !out_probabilities_device)
        {
            return false;
        }

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_softmax_processed_logits_f32(
            static_cast<const float *>(logits_device),
            row_count,
            vocab_size,
            row_stride,
            static_cast<float *>(out_probabilities_device),
            out_row_stride,
            device_id,
            stream);
    }

    bool ROCmBackend::enqueueFillInverseExponentialSamplesF32Device(
        void *out_samples_device,
        int row_count,
        int vocab_size,
        int row_stride,
        uint64_t seed,
        int first_logical_position,
        int device_id,
        void *stream)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !out_samples_device || row_count <= 0 ||
            vocab_size <= 0 || row_stride < vocab_size || !stream)
        {
            return false;
        }

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_fill_inverse_exponential_samples_f32(
            static_cast<float *>(out_samples_device),
            row_count,
            vocab_size,
            row_stride,
            static_cast<unsigned long long>(seed),
            first_logical_position,
            device_id,
            stream);
    }

    bool ROCmBackend::enqueueSpeculativeVerifyDistributionsF32DeviceThresholds(
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
        void *out_accept_probability_device,
        void *out_accept_threshold_device)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !target_token_ids_device || !target_probs_device ||
            !draft_token_ids_device || !draft_probs_device ||
            top_k <= 0 || top_k > 256 || !stream || !out_token_device || !out_accepted_device)
        {
            return false;
        }

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_speculative_verify_distribution_threshold_f32(
            static_cast<const int *>(target_token_ids_device),
            static_cast<const float *>(target_probs_device),
            static_cast<const int *>(draft_token_ids_device),
            static_cast<const float *>(draft_probs_device),
            top_k,
            draft_token,
            accept_threshold,
            residual_threshold,
            static_cast<int *>(out_token_device),
            static_cast<int *>(out_accepted_device),
            static_cast<float *>(out_accept_probability_device),
            static_cast<float *>(out_accept_threshold_device),
            device_id,
            stream);
    }

    bool ROCmBackend::enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatch(
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
        void *out_accept_probability_device,
        void *out_accept_threshold_device)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !target_token_ids_device || !target_probs_device ||
            !draft_token_ids_device || !draft_probs_device ||
            top_k <= 0 || top_k > 256 ||
            distribution_stride < top_k ||
            row_count <= 0 ||
            !draft_tokens_host || !accept_thresholds_host ||
            !residual_thresholds_host ||
            !stream || !out_token_device || !out_accepted_device)
        {
            return false;
        }

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_speculative_verify_distribution_thresholds_batch_f32(
            static_cast<const int *>(target_token_ids_device),
            static_cast<const float *>(target_probs_device),
            static_cast<const int *>(draft_token_ids_device),
            static_cast<const float *>(draft_probs_device),
            top_k,
            distribution_stride,
            draft_tokens_host,
            accept_thresholds_host,
            residual_thresholds_host,
            row_count,
            static_cast<int *>(out_token_device),
            static_cast<int *>(out_accepted_device),
            static_cast<float *>(out_accept_probability_device),
            static_cast<float *>(out_accept_threshold_device),
            device_id,
            stream);
    }

    bool ROCmBackend::enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatchDeviceTokens(
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
        void *out_accept_probability_device,
        void *out_accept_threshold_device,
        const void *draft_token_probabilities_device,
        uint64_t inverse_sample_seed,
        int inverse_sample_first_logical_position,
        int inverse_sample_vocab_size,
        const void *threshold_base_position_device,
        int threshold_position_offset,
            const int *generation_control_device,
        const uint64_t *threshold_seed_device)
    {
        const bool has_draft_distribution =
            draft_token_ids_device != nullptr && draft_probs_device != nullptr;
        const bool has_one_hot_draft_distribution =
            draft_token_ids_device == nullptr && draft_probs_device == nullptr;
        const bool has_host_thresholds =
            accept_thresholds_host != nullptr &&
            residual_thresholds_host != nullptr;
        const int threshold_position_source_count =
            (inverse_sample_first_logical_position >= 0 ? 1 : 0) +
            (threshold_base_position_device != nullptr ? 1 : 0);
        const bool uses_seeded_device_thresholds =
            accept_thresholds_host == nullptr &&
            residual_thresholds_host == nullptr &&
            has_one_hot_draft_distribution &&
            ((inverse_sample_seed != 0) + (threshold_seed_device != nullptr) == 1) &&
            threshold_position_source_count == 1;
        if (device_id >= device_count_ || device_id < 0 ||
            !target_token_ids_device || !target_probs_device ||
            (!has_draft_distribution && !has_one_hot_draft_distribution) ||
            !draft_tokens_device ||
            top_k <= 0 || top_k > 256 ||
            distribution_stride < top_k ||
            row_count <= 0 ||
            (!has_host_thresholds && !uses_seeded_device_thresholds) ||
            !stream || !out_token_device || !out_accepted_device ||
            (threshold_seed_device &&
             (!uses_seeded_device_thresholds || !generation_control_device ||
              reinterpret_cast<uintptr_t>(threshold_seed_device) % alignof(uint64_t) != 0)) ||
            (generation_control_device &&
             (!uses_seeded_device_thresholds || !threshold_base_position_device ||
              !has_one_hot_draft_distribution || inverse_sample_vocab_size <= 0)))
        {
            return false;
        }

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_speculative_verify_distribution_thresholds_batch_device_tokens_f32(
            static_cast<const int *>(target_token_ids_device),
            static_cast<const float *>(target_probs_device),
            static_cast<const int *>(draft_token_ids_device),
            static_cast<const float *>(draft_probs_device),
            top_k,
            distribution_stride,
            static_cast<const int *>(draft_tokens_device),
            static_cast<const float *>(draft_token_probabilities_device),
            accept_thresholds_host,
            residual_thresholds_host,
            row_count,
            inverse_sample_seed,
            inverse_sample_first_logical_position,
            inverse_sample_vocab_size,
            uses_seeded_device_thresholds ? inverse_sample_seed : 0ull,
            inverse_sample_first_logical_position,
            uses_seeded_device_thresholds ? 1 : 0,
            static_cast<const int *>(threshold_base_position_device),
            threshold_position_offset,
            generation_control_device,
            static_cast<int *>(out_token_device),
            static_cast<int *>(out_accepted_device),
            static_cast<float *>(out_accept_probability_device),
            static_cast<float *>(out_accept_threshold_device),
            device_id,
            stream, threshold_seed_device);
    }

    bool ROCmBackend::enqueueSpeculativeVerifyProcessedLogitsF32DeviceThresholdsBatchDeviceTokens(
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
        void *out_accept_probability_device,
        void *out_accept_threshold_device,
        const void *draft_token_probabilities_device)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !target_logits_device || !draft_logits_device ||
            !draft_tokens_device ||
            row_count <= 0 ||
            vocab_size <= 0 ||
            target_row_stride < vocab_size ||
            draft_row_stride < vocab_size ||
            !accept_thresholds_host || !residual_thresholds_host ||
            !stream || !out_token_device || !out_accepted_device)
        {
            return false;
        }

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_speculative_verify_processed_logits_thresholds_batch_device_tokens_f32(
            static_cast<const float *>(target_logits_device),
            static_cast<const float *>(draft_logits_device),
            row_count,
            vocab_size,
            target_row_stride,
            draft_row_stride,
            static_cast<const int *>(draft_tokens_device),
            accept_thresholds_host,
            residual_thresholds_host,
            static_cast<int *>(out_token_device),
            static_cast<int *>(out_accepted_device),
            static_cast<float *>(out_accept_probability_device),
            static_cast<float *>(out_accept_threshold_device),
            static_cast<const float *>(draft_token_probabilities_device),
            device_id,
            stream);
    }

    bool ROCmBackend::enqueueSpeculativeVerifyProcessedTargetDraftProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
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
        void *out_accept_probability_device,
        void *out_accept_threshold_device,
        bool no_draft_probabilities,
        const void *threshold_base_position_device,
        int threshold_position_offset)
    {
        const bool has_host_thresholds = accept_thresholds_host != nullptr;
        const int threshold_position_source_count =
            (inverse_sample_first_logical_position >= 0 ? 1 : 0) +
            (threshold_base_position_device != nullptr ? 1 : 0);
        const bool uses_seeded_device_thresholds =
            !has_host_thresholds &&
            inverse_sample_seed != 0 &&
            threshold_position_source_count == 1;
        if (device_id >= device_count_ || device_id < 0 ||
            !target_logits_device ||
            (!no_draft_probabilities && !draft_probabilities_device) ||
            !draft_tokens_device ||
            row_count <= 0 ||
            vocab_size <= 0 ||
            target_row_stride < vocab_size ||
            (!no_draft_probabilities && draft_row_stride < vocab_size) ||
            (!has_host_thresholds && !uses_seeded_device_thresholds) ||
            (has_host_thresholds && threshold_base_position_device) ||
            !stream || !out_token_device || !out_accepted_device)
        {
            return false;
        }

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_speculative_verify_processed_target_draft_probabilities_thresholds_batch_device_tokens_f32(
            static_cast<const float *>(target_logits_device),
            static_cast<const float *>(draft_probabilities_device),
            row_count,
            vocab_size,
            target_row_stride,
            draft_row_stride,
            static_cast<const int *>(draft_tokens_device),
            accept_thresholds_host,
            static_cast<unsigned long long>(inverse_sample_seed),
            inverse_sample_first_logical_position,
            uses_seeded_device_thresholds ? 1 : 0,
            static_cast<const int *>(threshold_base_position_device),
            threshold_position_offset,
            static_cast<int *>(out_token_device),
            static_cast<int *>(out_accepted_device),
            static_cast<float *>(out_accept_probability_device),
            static_cast<float *>(out_accept_threshold_device),
            no_draft_probabilities ? 1 : 0,
            device_id,
            stream);
    }

    bool ROCmBackend::enqueueSpeculativeVerifyProcessedTargetDraftLogitsF32DeviceThresholdsBatchDeviceTokens(
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
        void *out_accept_probability_device,
        void *out_accept_threshold_device,
        const void *draft_token_probabilities_device)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !target_logits_device || !draft_logits_device ||
            !draft_tokens_device ||
            row_count <= 0 ||
            vocab_size <= 0 ||
            target_row_stride < vocab_size ||
            draft_row_stride < vocab_size ||
            !accept_thresholds_host ||
            !stream || !out_token_device || !out_accepted_device)
        {
            return false;
        }

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_speculative_verify_processed_target_draft_logits_thresholds_batch_device_tokens_f32(
            static_cast<const float *>(target_logits_device),
            static_cast<const float *>(draft_logits_device),
            row_count,
            vocab_size,
            target_row_stride,
            draft_row_stride,
            static_cast<const int *>(draft_tokens_device),
            static_cast<const float *>(draft_token_probabilities_device),
            accept_thresholds_host,
            static_cast<unsigned long long>(inverse_sample_seed),
            inverse_sample_first_logical_position,
            static_cast<int *>(out_token_device),
            static_cast<int *>(out_accepted_device),
            static_cast<float *>(out_accept_probability_device),
            static_cast<float *>(out_accept_threshold_device),
            device_id,
            stream);
    }

    bool ROCmBackend::enqueueSpeculativeVerifyProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
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
        void *out_accept_probability_device,
        void *out_accept_threshold_device,
        bool no_draft_probabilities)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !target_probabilities_device || !inverse_rejection_samples_device ||
            (!no_draft_probabilities && !draft_probabilities_device) ||
            !draft_tokens_device ||
            row_count <= 0 ||
            vocab_size <= 0 ||
            target_row_stride < vocab_size ||
            (!no_draft_probabilities && draft_row_stride < vocab_size) ||
            inverse_sample_row_stride < vocab_size ||
            !accept_thresholds_host ||
            !stream || !out_token_device || !out_accepted_device)
        {
            return false;
        }

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_speculative_verify_probabilities_thresholds_batch_device_tokens_f32(
            static_cast<const float *>(target_probabilities_device),
            static_cast<const float *>(draft_probabilities_device),
            static_cast<const float *>(inverse_rejection_samples_device),
            row_count,
            vocab_size,
            target_row_stride,
            draft_row_stride,
            inverse_sample_row_stride,
            static_cast<const int *>(draft_tokens_device),
            accept_thresholds_host,
            no_draft_probabilities ? 1 : 0,
            static_cast<int *>(out_token_device),
            static_cast<int *>(out_accepted_device),
            static_cast<float *>(out_accept_probability_device),
            static_cast<float *>(out_accept_threshold_device),
            device_id,
            stream);
    }

    bool ROCmBackend::enqueueSummarizeSpeculativeVerifyBatch(
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
        const void *max_state_commit_rows_device,
        int leading_committed_output_count)
    {
        using namespace sampling_math;
        if (device_id >= device_count_ || device_id < 0 ||
            !verify_tokens_device || !verify_accepted_device ||
            row_count < 0 || out_token_capacity < row_count + 1 ||
            stop_token_count < 0 ||
            stop_token_count > kSpeculativeBatchMaxStopTokens ||
            (stop_token_count > 0 && !stop_tokens_host) ||
            (has_bonus_token && !bonus_token_device) ||
            !stream || !out_tokens_device || !out_meta_device)
        {
            return false;
        }

        int stop_tokens[kSpeculativeBatchMaxStopTokens] =
            {-1, -1, -1, -1, -1, -1, -1, -1};
        for (int i = 0; i < stop_token_count; ++i)
            stop_tokens[i] = stop_tokens_host[i];

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_summarize_speculative_verify_batch(
            static_cast<const int *>(verify_tokens_device),
            static_cast<const int *>(verify_accepted_device),
            row_count,
            first_token,
            stop_tokens[0],
            stop_tokens[1],
            stop_tokens[2],
            stop_tokens[3],
            stop_tokens[4],
            stop_tokens[5],
            stop_tokens[6],
            stop_tokens[7],
            stop_token_count,
            static_cast<const int *>(bonus_token_device),
            has_bonus_token ? 1 : 0,
            static_cast<int *>(out_tokens_device),
            out_token_capacity,
            static_cast<int *>(out_meta_device),
            static_cast<const uint32_t *>(max_state_commit_rows_device),
            leading_committed_output_count,
            device_id,
            stream);
    }

    bool ROCmBackend::enqueueSummarizeSpeculativeVerifyBatchDeviceFirstToken(
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
        const void *max_state_commit_rows_device,
        int leading_committed_output_count)
    {
        using namespace sampling_math;
        if (device_id >= device_count_ || device_id < 0 ||
            !verify_tokens_device || !verify_accepted_device ||
            !first_token_device ||
            row_count < 0 || out_token_capacity < row_count + 1 ||
            stop_token_count < 0 ||
            stop_token_count > kSpeculativeBatchMaxStopTokens ||
            (stop_token_count > 0 && !stop_tokens_host) ||
            (has_bonus_token && !bonus_token_device) ||
            !stream || !out_tokens_device || !out_meta_device)
        {
            return false;
        }

        int stop_tokens[kSpeculativeBatchMaxStopTokens] =
            {-1, -1, -1, -1, -1, -1, -1, -1};
        for (int i = 0; i < stop_token_count; ++i)
            stop_tokens[i] = stop_tokens_host[i];

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_summarize_speculative_verify_batch_device_first_token(
            static_cast<const int *>(verify_tokens_device),
            static_cast<const int *>(verify_accepted_device),
            row_count,
            static_cast<const int *>(first_token_device),
            stop_tokens[0],
            stop_tokens[1],
            stop_tokens[2],
            stop_tokens[3],
            stop_tokens[4],
            stop_tokens[5],
            stop_tokens[6],
            stop_tokens[7],
            stop_token_count,
            static_cast<const int *>(bonus_token_device),
            has_bonus_token ? 1 : 0,
            static_cast<int *>(out_tokens_device),
            out_token_capacity,
            static_cast<int *>(out_meta_device),
            static_cast<const uint32_t *>(max_state_commit_rows_device),
            leading_committed_output_count,
            device_id,
            stream);
    }

    bool ROCmBackend::
        enqueueSummarizeSpeculativeVerifyBatchDeviceGenerationControls(
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
            void *out_meta_device)
    {
        const bool has_acceptance_rows = verify_accepted_device != nullptr;
        const bool has_greedy_rows = greedy_draft_tokens_device != nullptr;
        if (device_id >= device_count_ || device_id < 0 ||
            !verify_tokens_device || has_acceptance_rows == has_greedy_rows ||
            row_count < 0 || !first_token_device || !stop_tokens_device ||
            (has_bonus_token && !bonus_token_device) ||
            !generation_control_device ||
            out_token_capacity < row_count + 1 || !stream ||
            !out_tokens_device || !out_meta_device)
        {
            return false;
        }

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_summarize_speculative_verify_batch_device_generation_controls(
            static_cast<const int *>(verify_tokens_device),
            static_cast<const int *>(verify_accepted_device),
            static_cast<const int *>(greedy_draft_tokens_device),
            row_count,
            static_cast<const int *>(first_token_device),
            static_cast<const int *>(stop_tokens_device),
            static_cast<const int *>(bonus_token_device),
            has_bonus_token ? 1 : 0,
            static_cast<const int *>(generation_control_device),
            static_cast<int *>(out_tokens_device),
            out_token_capacity,
            static_cast<int *>(out_meta_device),
            device_id,
            stream);
    }

    bool ROCmBackend::
        enqueueSampleAndSummarizeSerialEquivalentSpeculativeBatchDeviceGenerationControls(
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
            void *first_transaction_diagnostic_device,
            const uint64_t *threshold_seed_device)
    {
        if (device_id < 0 || device_id >= device_count_)
            return false;

        HipDeviceGuard::setDevice(device_id);
        return rocmOps_sample_and_summarize_serial_equivalent_speculative_batch_device_generation_controls(
            static_cast<const int *>(target_token_ids_device),
            static_cast<const float *>(target_probs_device),
            target_row_stride,
            top_k,
            row_count,
            static_cast<unsigned long long>(threshold_seed),
            static_cast<const int *>(threshold_position_device),
            threshold_position_offset,
            static_cast<const int *>(verifier_input_tokens_device),
            static_cast<const int *>(stop_tokens_device),
            static_cast<const int *>(generation_control_device),
            static_cast<int *>(sampled_target_tokens_device),
            static_cast<int *>(out_tokens_device),
            out_token_capacity,
            static_cast<int *>(out_meta_device),
            first_transaction_diagnostic_device,
            device_id,
            stream, threshold_seed_device);
    }

    bool ROCmBackend::enqueueSummarizeGreedySpeculativeVerifyBatch(
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
        const void *max_state_commit_rows_device,
        int leading_committed_output_count)
    {
        using namespace sampling_math;
        if (device_id >= device_count_ || device_id < 0 ||
            !verify_tokens_device || !draft_tokens_device ||
            compare_row_count < 0 ||
            out_token_capacity < compare_row_count + 1 ||
            stop_token_count < 0 ||
            stop_token_count > kSpeculativeBatchMaxStopTokens ||
            (stop_token_count > 0 && !stop_tokens_host) ||
            !stream || !out_tokens_device || !out_meta_device)
        {
            return false;
        }

        int stop_tokens[kSpeculativeBatchMaxStopTokens] =
            {-1, -1, -1, -1, -1, -1, -1, -1};
        for (int i = 0; i < stop_token_count; ++i)
            stop_tokens[i] = stop_tokens_host[i];

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_summarize_greedy_speculative_verify_batch(
            static_cast<const int *>(verify_tokens_device),
            static_cast<const int *>(draft_tokens_device),
            compare_row_count,
            first_token,
            stop_tokens[0],
            stop_tokens[1],
            stop_tokens[2],
            stop_tokens[3],
            stop_tokens[4],
            stop_tokens[5],
            stop_tokens[6],
            stop_tokens[7],
            stop_token_count,
            static_cast<int *>(out_tokens_device),
            out_token_capacity,
            static_cast<int *>(out_meta_device),
            static_cast<const uint32_t *>(max_state_commit_rows_device),
            leading_committed_output_count,
            device_id,
            stream);
    }

    bool ROCmBackend::
        enqueueSummarizeGreedySpeculativeVerifyBatchDeviceControls(
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
            const void *max_state_commit_rows_device,
            const void *next_leading_committed_output_count_device)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !verify_tokens_device || !draft_tokens_device ||
            !active_verifier_row_count_device || !stop_tokens_device ||
            !next_leading_committed_output_count_device ||
            compare_row_count < 0 ||
            out_token_capacity < compare_row_count + 1 ||
            !stream || !out_tokens_device || !out_meta_device)
        {
            return false;
        }

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_summarize_greedy_speculative_verify_batch_device_controls(
            static_cast<const int *>(verify_tokens_device),
            static_cast<const int *>(draft_tokens_device),
            compare_row_count,
            static_cast<const int *>(active_verifier_row_count_device),
            static_cast<const int *>(stop_tokens_device),
            static_cast<int *>(out_tokens_device),
            out_token_capacity,
            static_cast<int *>(out_meta_device),
            static_cast<const uint32_t *>(max_state_commit_rows_device),
            static_cast<const int *>(
                next_leading_committed_output_count_device),
            device_id,
            stream);
    }

    bool ROCmBackend::enqueueAdvanceSpeculativeCommitBoundary(
        void *meta_device,
        int request_count,
        int meta_stride,
        void *decode_rounds_committed_device,
        void *decode_rounds_until_maintenance_device,
        void *maintenance_due_device,
        void *decode_boundary_advanced_device,
        int device_id,
        void *stream)
    {
        if (device_id < 0 || device_id >= device_count_ || !meta_device ||
            request_count <= 0 ||
            meta_stride < sampling_math::kSpeculativeBatchMetaCount ||
            !decode_rounds_committed_device ||
            !decode_rounds_until_maintenance_device ||
            !maintenance_due_device ||
            !decode_boundary_advanced_device || !stream)
        {
            return false;
        }

        HipDeviceGuard::setDevice(device_id);
        return rocmOps_advance_speculative_commit_boundary(
            static_cast<int *>(meta_device),
            request_count,
            meta_stride,
            static_cast<uint32_t *>(decode_rounds_committed_device),
            static_cast<uint32_t *>(decode_rounds_until_maintenance_device),
            static_cast<uint32_t *>(maintenance_due_device),
            static_cast<uint32_t *>(decode_boundary_advanced_device),
            device_id,
            stream);
    }

    bool ROCmBackend::enqueuePublishSerialDecodeCommitBoundary(
        void *decode_rounds_committed_device,
        void *decode_rounds_until_maintenance_device,
        void *maintenance_due_device,
        void *decode_boundary_advanced_device,
        int device_id,
        void *stream)
    {
        if (device_id < 0 || device_id >= device_count_ ||
            !decode_rounds_committed_device ||
            !decode_rounds_until_maintenance_device ||
            !maintenance_due_device ||
            !decode_boundary_advanced_device || !stream)
        {
            return false;
        }

        HipDeviceGuard::setDevice(device_id);
        return rocmOps_publish_serial_decode_commit_boundary(
            static_cast<uint32_t *>(decode_rounds_committed_device),
            static_cast<uint32_t *>(decode_rounds_until_maintenance_device),
            static_cast<uint32_t *>(maintenance_due_device),
            static_cast<uint32_t *>(decode_boundary_advanced_device),
            device_id,
            stream);
    }

    bool ROCmBackend::enqueueAcknowledgeDecodeCommitBoundary(
        void *decode_rounds_until_maintenance_device,
        void *maintenance_due_device,
        void *decode_boundary_advanced_device,
        int device_id,
        void *stream)
    {
        if (device_id < 0 || device_id >= device_count_ ||
            !decode_rounds_until_maintenance_device ||
            !maintenance_due_device ||
            !decode_boundary_advanced_device || !stream)
        {
            return false;
        }

        HipDeviceGuard::setDevice(device_id);
        return rocmOps_acknowledge_decode_commit_boundary(
            static_cast<uint32_t *>(decode_rounds_until_maintenance_device),
            static_cast<uint32_t *>(maintenance_due_device),
            static_cast<uint32_t *>(decode_boundary_advanced_device),
            device_id,
            stream);
    }

    bool ROCmBackend::enqueueInitializeDeviceMoERebalanceDispatchTicket(
        uint64_t session_epoch,
        uint64_t workspace_generation,
        uint32_t participant_id,
        uint32_t participant_count,
        void *ticket_device,
        int device_id,
        void *stream)
    {
        if (device_id < 0 || device_id >= device_count_ ||
            session_epoch == 0u || workspace_generation == 0u ||
            participant_count == 0u || participant_id >= participant_count ||
            !ticket_device || !stream)
        {
            return false;
        }

        HipDeviceGuard::setDevice(device_id);
        return rocmOps_initialize_device_moe_rebalance_dispatch_ticket(
            session_epoch,
            workspace_generation,
            participant_id,
            participant_count,
            static_cast<DeviceMoERebalanceDispatchTicket *>(ticket_device),
            device_id,
            stream);
    }

    bool ROCmBackend::enqueuePublishDeviceMoERebalanceDispatchTicket(
        const void *controller_magic_device,
        const void *controller_version_device,
        const void *controller_error_device,
        const void *decode_rounds_committed_device,
        const void *decode_rounds_until_maintenance_device,
        const void *maintenance_due_device,
        const void *decode_boundary_advanced_device,
        void *ticket_device,
        int device_id,
        void *stream)
    {
        if (device_id < 0 || device_id >= device_count_ ||
            !controller_magic_device || !controller_version_device ||
            !controller_error_device || !decode_rounds_committed_device ||
            !decode_rounds_until_maintenance_device ||
            !maintenance_due_device || !decode_boundary_advanced_device ||
            !ticket_device || !stream)
        {
            return false;
        }

        HipDeviceGuard::setDevice(device_id);
        return rocmOps_publish_device_moe_rebalance_dispatch_ticket(
            static_cast<const uint32_t *>(controller_magic_device),
            static_cast<const uint32_t *>(controller_version_device),
            static_cast<const uint32_t *>(controller_error_device),
            static_cast<const uint32_t *>(decode_rounds_committed_device),
            static_cast<const uint32_t *>(
                decode_rounds_until_maintenance_device),
            static_cast<const uint32_t *>(maintenance_due_device),
            static_cast<const uint32_t *>(decode_boundary_advanced_device),
            static_cast<DeviceMoERebalanceDispatchTicket *>(ticket_device),
            device_id,
            stream);
    }

    /** @copydoc IBackend::enqueueInitializeDeviceGeneration */
    bool ROCmBackend::enqueueInitializeDeviceGeneration(
        int request_count,
        int max_new_tokens,
        const sampling_math::DeviceGenerationPolicy &depth_policy,
        sampling_math::DeviceGenerationLeadingRowDisposition
            initial_leading_row_disposition,
        int response_token_stride,
        void *response_tokens_device,
        int control_stride,
        void *control_device,
        int device_id,
        void *stream,
        const sampling_math::DeviceGenerationInitialization &initialization)
    {
        if (device_id < 0 || device_id >= device_count_ ||
            request_count <= 0 || !initialization.valid() ||
            !sampling_math::valid_device_generation_admission(
                depth_policy, max_new_tokens, initial_leading_row_disposition) ||
            response_token_stride <= 0 || response_token_stride < max_new_tokens ||
            !response_tokens_device ||
            control_stride < sampling_math::kDeviceGenerationControlCount ||
            !control_device || !stream)
        {
            return false;
        }

        HipDeviceGuard::setDevice(device_id);
        return rocmOps_initialize_device_generation(
            request_count,
            max_new_tokens,
            depth_policy,
            initial_leading_row_disposition,
            response_token_stride,
            control_stride,
            static_cast<int *>(control_device),
            device_id,
            stream, initialization);
    }

    /** @copydoc IBackend::enqueuePublishOrdinaryGenerationSample */
    bool ROCmBackend::enqueuePublishOrdinaryGenerationSample(
        const sampling_math::OrdinaryGenerationPublication &publication,
        int device_id, void *stream)
    {
        if (device_id < 0 || device_id >= device_count_ ||
            !publication.valid() || !stream)
            return false;
        return rocmOps_publish_ordinary_generation_sample(
            publication, device_id, stream);
    }

    bool ROCmBackend::enqueueInitializeDeviceGenerationDispatchTicket(
        uint64_t session_epoch,
        uint64_t workspace_generation,
        void *control_device,
        int control_stride,
        int request_count,
        void *dispatch_tickets_device,
        int device_id,
        void *stream)
    {
        if (device_id < 0 || device_id >= device_count_ ||
            session_epoch == 0 || workspace_generation == 0 ||
            !control_device ||
            control_stride < sampling_math::kDeviceGenerationControlCount ||
            request_count <= 0 || !dispatch_tickets_device || !stream)
        {
            return false;
        }

        HipDeviceGuard::setDevice(device_id);
        return rocmOps_initialize_device_generation_dispatch_tickets(
            session_epoch,
            workspace_generation,
            static_cast<int *>(control_device),
            control_stride,
            request_count,
            static_cast<sampling_math::DeviceGenerationDispatchTicket *>(
                dispatch_tickets_device),
            device_id,
            stream);
    }

    bool ROCmBackend::enqueuePublishDeviceGenerationDispatchTickets(
        void *control_device,
        int control_stride,
        int request_count,
        const void *maintenance_due_device,
        void *dispatch_tickets_device,
        int device_id,
        void *stream)
    {
        if (device_id < 0 || device_id >= device_count_ || !control_device ||
            control_stride < sampling_math::kDeviceGenerationControlCount ||
            request_count <= 0 || !dispatch_tickets_device || !stream)
        {
            return false;
        }

        HipDeviceGuard::setDevice(device_id);
        return rocmOps_publish_device_generation_dispatch_tickets(
            static_cast<int *>(control_device),
            control_stride,
            request_count,
            static_cast<const uint32_t *>(maintenance_due_device),
            static_cast<sampling_math::DeviceGenerationDispatchTicket *>(
                dispatch_tickets_device),
            device_id,
            stream);
    }

    bool ROCmBackend::
        enqueuePublishMoECurrentBatchLLEPEvidenceToGenerationControl(
            const void *runtime_layers_device,
            int layer_count,
            void *generation_control_device,
            int generation_control_stride,
            int request_count,
            int device_id,
            void *stream)
    {
        if (device_id < 0 || device_id >= device_count_ ||
            !runtime_layers_device || layer_count <= 0 ||
            !generation_control_device ||
            generation_control_stride <
                sampling_math::kDeviceGenerationControlCount ||
            request_count <= 0 || !stream)
        {
            return false;
        }

        HipDeviceGuard::setDevice(device_id);
        return hipMoE_publish_current_batch_llep_evidence_to_generation_control(
            runtime_layers_device,
            layer_count,
            static_cast<int *>(generation_control_device),
            generation_control_stride,
            request_count,
            sampling_math::
                kDeviceGenerationControlCurrentBatchLLEPMovementLayerCount,
            sampling_math::
                kDeviceGenerationControlCurrentBatchLLEPNonOwnerAssignmentLayerCount,
            device_id,
            stream);
    }

    bool ROCmBackend::enqueuePrepareDeviceGenerationTransactionBudget(
        void *control_device,
        int control_stride,
        int request_count,
        int verifier_row_capacity,
        const void *maintenance_rows_remaining_device,
        const void *maintenance_due_device,
        void *decode_boundary_advanced_device,
        int device_id,
        void *stream)
    {
        const bool has_maintenance_boundary =
            maintenance_rows_remaining_device != nullptr ||
            maintenance_due_device != nullptr ||
            decode_boundary_advanced_device != nullptr;
        const bool has_complete_maintenance_boundary =
            maintenance_rows_remaining_device != nullptr &&
            maintenance_due_device != nullptr &&
            decode_boundary_advanced_device != nullptr;
        if (device_id < 0 || device_id >= device_count_ || !control_device ||
            control_stride < sampling_math::kDeviceGenerationControlCount ||
            request_count <= 0 || verifier_row_capacity <= 0 || !stream ||
            (has_maintenance_boundary &&
             !has_complete_maintenance_boundary))
        {
            return false;
        }

        HipDeviceGuard::setDevice(device_id);
        return rocmOps_prepare_device_generation_transaction_budget(
            static_cast<int *>(control_device),
            control_stride,
            request_count,
            verifier_row_capacity,
            static_cast<const uint32_t *>(maintenance_rows_remaining_device),
            static_cast<const uint32_t *>(maintenance_due_device),
            static_cast<uint32_t *>(decode_boundary_advanced_device),
            device_id,
            stream);
    }

    bool ROCmBackend::
        enqueueCommitDeviceGenerationAndDeriveSpeculativePublicationMetadata(
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
        void *out_next_condition_tokens_device,
        void *out_all_drafts_accepted_flags_device,
        void *out_stopped_flags_device,
        void *out_next_sidecar_condition_tokens_device,
        void *out_next_sidecar_position_ids_device,
        void *out_next_verifier_condition_tokens_device,
        const void *verifier_input_tokens_device,
        int verifier_input_token_stride,
        void *out_committed_verifier_identity_device,
        sampling_math::MTPDepthTransactionTrace *depth_transaction_trace_device)
    {
        const bool has_verifier_identity_binding =
            verifier_input_tokens_device != nullptr ||
            verifier_input_token_stride != 0 ||
            out_committed_verifier_identity_device != nullptr;
        const bool has_complete_verifier_identity_binding =
            verifier_input_tokens_device != nullptr &&
            verifier_input_token_stride > 0 &&
            out_committed_verifier_identity_device != nullptr;
        if (device_id < 0 || device_id >= device_count_ ||
            !output_tokens_device || output_token_stride <= 0 ||
            !meta_device || !base_cached_tokens_device ||
            meta_stride < sampling_math::kSpeculativeBatchMetaCount ||
            request_count <= 0 || (depth_transaction_trace_device && request_count != 1) || padded_state_rows_per_request <= 0 ||
            !response_tokens_device ||
            response_token_stride <= 0 || !control_device ||
            control_stride < sampling_math::kDeviceGenerationControlCount ||
            !out_restore_rows_device || !out_target_cached_tokens_device ||
            !out_accepted_state_counts_device || !out_ok_device ||
            ((out_next_sidecar_condition_tokens_device ||
              out_next_sidecar_position_ids_device) &&
             (!out_next_sidecar_condition_tokens_device ||
              !out_next_sidecar_position_ids_device ||
              !out_next_condition_tokens_device)) ||
            !out_next_verifier_condition_tokens_device ||
            (has_verifier_identity_binding &&
             !has_complete_verifier_identity_binding) ||
            !stream)
        {
            return false;
        }

        HipDeviceGuard::setDevice(device_id);
        return rocmOps_commit_device_generation_and_derive_speculative_publication_metadata(
            static_cast<const int32_t *>(output_tokens_device),
            output_token_stride,
            static_cast<int *>(meta_device),
            meta_stride,
            static_cast<const int *>(base_cached_tokens_device),
            request_count,
            padded_state_rows_per_request,
            static_cast<int32_t *>(response_tokens_device),
            response_token_stride,
            static_cast<int *>(control_device),
            control_stride,
            static_cast<int *>(out_restore_rows_device),
            static_cast<int *>(out_target_cached_tokens_device),
            static_cast<int *>(out_accepted_state_counts_device),
            static_cast<int *>(out_ok_device),
            static_cast<int *>(out_next_condition_tokens_device),
            static_cast<int *>(out_all_drafts_accepted_flags_device),
            static_cast<int *>(out_stopped_flags_device),
            static_cast<int *>(out_next_sidecar_condition_tokens_device),
            static_cast<int *>(out_next_sidecar_position_ids_device),
            static_cast<int *>(out_next_verifier_condition_tokens_device),
            static_cast<const int32_t *>(verifier_input_tokens_device),
            verifier_input_token_stride,
            static_cast<
                sampling_math::MTPCommittedVerifierIdentityRecord *>(
                out_committed_verifier_identity_device),
            depth_transaction_trace_device,
            device_id,
            stream);
    }

    bool ROCmBackend::enqueueDeriveSpeculativePublicationMetadata(
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
        void *out_next_condition_tokens_device,
        const void *output_tokens_device,
        int output_token_stride,
        void *out_all_drafts_accepted_flags_device,
        void *out_stopped_flags_device,
        void *out_next_verifier_condition_tokens_device)
    {
        using namespace sampling_math;
        if (device_id >= device_count_ || device_id < 0 ||
            !meta_device || !base_cached_tokens_device ||
            meta_stride < kSpeculativeBatchMetaCount ||
            request_count <= 0 ||
            padded_state_rows_per_request <= 0 ||
            max_state_commit_rows < 0 ||
            max_state_commit_rows > padded_state_rows_per_request ||
            !stream ||
            !out_restore_rows_device ||
            !out_target_cached_tokens_device ||
            !out_accepted_state_counts_device ||
            !out_ok_device ||
            ((out_next_condition_tokens_device || output_tokens_device) &&
             (!out_next_condition_tokens_device ||
              !output_tokens_device ||
              output_token_stride <= 0)))
        {
            return false;
        }

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_derive_speculative_publication_metadata(
            static_cast<const int *>(meta_device),
            meta_stride,
            static_cast<const int *>(base_cached_tokens_device),
            request_count,
            padded_state_rows_per_request,
            max_state_commit_rows,
            static_cast<int *>(out_restore_rows_device),
            static_cast<int *>(out_target_cached_tokens_device),
            static_cast<int *>(out_accepted_state_counts_device),
            static_cast<int *>(out_ok_device),
            static_cast<int *>(out_next_condition_tokens_device),
            static_cast<const int32_t *>(output_tokens_device),
            output_token_stride,
            static_cast<int *>(out_all_drafts_accepted_flags_device),
            static_cast<int *>(out_stopped_flags_device),
            static_cast<int *>(out_next_verifier_condition_tokens_device),
            device_id,
            stream);
    }

    bool ROCmBackend::
        enqueueDeriveShiftedSpeculativePublicationMetadataFromPrimary(
        const void *base_cached_tokens_device,
        const void *main_target_cached_tokens_device,
        const void *main_publication_ok_device,
        int request_count,
        int mtp_depth,
        int device_id,
        void *stream,
        void *out_target_cached_tokens_device,
        void *out_accepted_state_counts_device,
        void *out_ok_device, MTPShiftedCacheOrigin origin)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !base_cached_tokens_device ||
            !main_target_cached_tokens_device ||
            !main_publication_ok_device ||
            request_count <= 0 ||
            mtpShiftedPrefixOffset(origin,mtp_depth)<0 ||
            !stream ||
            !out_target_cached_tokens_device ||
            !out_accepted_state_counts_device ||
            !out_ok_device)
        {
            return false;
        }

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_derive_shifted_speculative_publication_metadata_from_primary(
            static_cast<const int *>(base_cached_tokens_device),
            static_cast<const int *>(main_target_cached_tokens_device),
            static_cast<const int *>(main_publication_ok_device),
            request_count,
            mtp_depth,
            static_cast<int *>(out_target_cached_tokens_device),
            static_cast<int *>(out_accepted_state_counts_device),
            static_cast<int *>(out_ok_device),
            device_id,
            stream,origin);
    }

    bool ROCmBackend::enqueuePrepareSpeculativeShiftedKVTokens(
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
        void *out_position_ids_device)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !meta_device ||
            !output_tokens_device ||
            !out_tokens_device ||
            !base_positions_device ||
            !out_position_ids_device ||
            !stream ||
            meta_stride < sampling_math::kSpeculativeBatchMetaCount ||
            output_token_stride <= first_output_token_index ||
            request_index < 0 ||
            first_output_token_index < 0 ||
            row_count <= 0)
        {
            return false;
        }

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_prepare_speculative_shifted_kv_tokens(
            static_cast<const int *>(meta_device),
            meta_stride,
            static_cast<const int32_t *>(output_tokens_device),
            output_token_stride,
            request_index,
            first_output_token_index,
            row_count,
            filler_token,
            static_cast<int32_t *>(out_tokens_device),
            static_cast<const int32_t *>(base_positions_device),
            position_offset,
            static_cast<int32_t *>(out_position_ids_device),
            device_id,
            stream);
    }

    bool ROCmBackend::enqueuePrepareMTPBatchedSidecarInputs(
        const void *condition_tokens_device,
        int condition_token_stride,
        const void *base_positions_device,
        int position_offset,
        int request_count,
        int device_id,
        void *stream,
        void *out_condition_tokens_device,
        void *out_position_ids_device)
    {
        if (device_id < 0 || device_id >= device_count_ ||
            !condition_tokens_device ||
            condition_token_stride <= 0 ||
            !base_positions_device ||
            request_count <= 0 ||
            !stream ||
            !out_condition_tokens_device ||
            !out_position_ids_device)
        {
            return false;
        }

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_prepare_mtp_batched_sidecar_inputs(
            static_cast<const int32_t *>(condition_tokens_device),
            condition_token_stride,
            static_cast<const int32_t *>(base_positions_device),
            position_offset,
            request_count,
            static_cast<int32_t *>(out_condition_tokens_device),
            static_cast<int32_t *>(out_position_ids_device),
            device_id,
            stream);
    }

    bool ROCmBackend::enqueuePrepareMTPVerifierPositionIds(
        const void *base_positions_device,
        int request_count,
        int padded_seq_len,
        int device_id,
        void *stream,
        void *out_position_ids_device)
    {
        if (device_id < 0 || device_id >= device_count_ ||
            !base_positions_device ||
            request_count <= 0 ||
            padded_seq_len <= 0 ||
            !stream ||
            !out_position_ids_device)
        {
            return false;
        }

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_prepare_mtp_verifier_position_ids(
            static_cast<const int32_t *>(base_positions_device),
            request_count,
            padded_seq_len,
            static_cast<int32_t *>(out_position_ids_device),
            device_id,
            stream);
    }

    bool ROCmBackend::enqueuePrepareMTPVerifierGeometry(
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
        void *out_request_lengths_device)
    {
        const bool has_generation_control =
            generation_control_device != nullptr;
        if (device_id < 0 || device_id >= device_count_ ||
            !base_positions_device || request_count <= 0 ||
            padded_seq_len <= 0 || !stream || !out_position_ids_device ||
            !out_request_lengths_device ||
            has_generation_control != (generation_control_stride > 0) ||
            (has_generation_control &&
             generation_control_stride <
                 sampling_math::kDeviceGenerationControlCount))
        {
            return false;
        }

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_prepare_mtp_verifier_geometry(
            static_cast<const int32_t *>(base_positions_device),
            static_cast<const int32_t *>(valid_graph_rows_device),
            valid_graph_row_count,
            static_cast<int *>(generation_control_device),
            generation_control_stride,
            request_count,
            padded_seq_len,
            static_cast<int32_t *>(out_position_ids_device),
            static_cast<int32_t *>(out_request_lengths_device),
            device_id,
            stream);
    }

    bool ROCmBackend::enqueuePrepareMTPVerifierControlledRow(
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
        void *out_base_position_snapshot_device)
    {
        if (device_id < 0 || device_id >= device_count_ ||
            !first_token_device || !draft_tokens_device ||
            !base_position_device || !generation_control_row_device ||
            generation_control_stride <
                sampling_math::kDeviceGenerationControlCount ||
            padded_seq_len <= 1 || !stream || !out_tokens_device ||
            !out_position_ids_device || !out_request_length_device ||
            !out_base_position_snapshot_device)
        {
            return false;
        }

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_prepare_mtp_verifier_controlled_row(
            static_cast<const int32_t *>(first_token_device),
            static_cast<const int32_t *>(draft_tokens_device),
            static_cast<const int32_t *>(base_position_device),
            static_cast<int *>(generation_control_row_device),
            generation_control_stride,
            padded_seq_len,
            static_cast<int32_t *>(out_tokens_device),
            static_cast<int32_t *>(out_position_ids_device),
            static_cast<int32_t *>(out_request_length_device),
            static_cast<int32_t *>(out_base_position_snapshot_device),
            device_id,
            stream);
    }

    bool ROCmBackend::enqueueInitializeGenerationLogicalState(
        const GenerationLogicalStateInitialization &initialization,
        int device_id, void *stream)
    {
        if (device_id < 0 || device_id >= device_count_ || !stream || !initialization.valid())
            return false;
        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_initialize_generation_logical_state(initialization, device_id, stream);
    }

    // Forward declaration for HIP penalty kernel (implemented in ROCmSamplingKernels.hip)
    extern "C" bool rocmOps_apply_logit_penalties_f32(
        float *logits, const int *token_ids, const float *penalties,
        int num_penalties, int vocab_size, int device_idx, void *stream, float repetition_penalty);

    bool ROCmBackend::prepareLogitPenaltyWorkspace(
        int vocab_size,
        int device_id)
    {
        if (device_id < 0 ||
            device_id >= device_count_ ||
            vocab_size <= 0 ||
            static_cast<size_t>(device_id) >= penalty_buffers_.size())
        {
            return false;
        }

        auto &bufs = penalty_buffers_[static_cast<size_t>(device_id)];
        if (bufs.allocated_count >= vocab_size &&
            bufs.token_ids_ptr &&
            bufs.penalties_ptr &&
            bufs.ready_event)
        {
            return true;
        }
        if (bufs.allocated_count != 0 ||
            bufs.token_ids_ptr ||
            bufs.penalties_ptr ||
            bufs.ready_event)
        {
            LOG_ERROR("[ROCmBackend] Refusing to resize an active logit-penalty workspace");
            return false;
        }

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        hipError_t err =
            hipMalloc(&bufs.token_ids_ptr, vocab_size * sizeof(int));
        if (err != hipSuccess)
            return false;
        err = hipMalloc(
            &bufs.penalties_ptr,
            vocab_size * sizeof(float));
        if (err != hipSuccess)
        {
            HIP_WARN_IF_FAIL(hipFree(bufs.token_ids_ptr));
            bufs.token_ids_ptr = nullptr;
            return false;
        }
        hipEvent_t ready_event = nullptr;
        err = hipEventCreateWithFlags(
            &ready_event,
            hipEventDisableTiming);
        if (err != hipSuccess)
        {
            HIP_WARN_IF_FAIL(hipFree(bufs.penalties_ptr));
            HIP_WARN_IF_FAIL(hipFree(bufs.token_ids_ptr));
            bufs.penalties_ptr = nullptr;
            bufs.token_ids_ptr = nullptr;
            return false;
        }
        bufs.ready_event = ready_event;
        bufs.allocated_count = vocab_size;
        return true;
    }

    extern "C" bool rocmOps_initialize_prompt_repetition_history(
        int32_t *counts, const int32_t *tokens, int token_count, int vocab_size, void *stream);

    bool ROCmBackend::initializePromptRepetitionHistory(
        void *counts_device, const int32_t *unique_tokens, int token_count,
        int vocab_size, int device_id, void *stream)
    {
        if (device_id < 0 || device_id >= device_count_ || !counts_device || !unique_tokens ||
            token_count <= 0 || token_count > vocab_size || !stream)
            return false;
        for (int i = 0; i < token_count; ++i)
            if (unique_tokens[i] < 0 || unique_tokens[i] >= vocab_size ||
                (i != 0 && unique_tokens[i - 1] >= unique_tokens[i]))
                return false;
        auto &bufs = penalty_buffers_[static_cast<size_t>(device_id)];
        if (token_count > bufs.allocated_count || !bufs.token_ids_ptr || !bufs.ready_event)
            return false;
        HipDeviceGuard::setDevice(device_id);
        auto s = requireExplicitStream(stream, "ROCmBackend::initializePromptRepetitionHistory");
        if (bufs.publication_valid && bufs.producer_stream != stream)
            HIP_CHECK_OR_THROW(hipStreamWaitEvent(s, static_cast<hipEvent_t>(bufs.ready_event), 0));
        HIP_CHECK_OR_THROW(hipMemcpyAsync(bufs.token_ids_ptr, unique_tokens,
            static_cast<size_t>(token_count) * sizeof(int32_t), hipMemcpyHostToDevice, s));
        if (!rocmOps_initialize_prompt_repetition_history(static_cast<int32_t *>(counts_device),
                static_cast<const int32_t *>(bufs.token_ids_ptr), token_count, vocab_size, s))
            return false;
        HIP_CHECK_OR_THROW(hipEventRecord(static_cast<hipEvent_t>(bufs.ready_event), s));
        bufs.producer_stream = stream;
        bufs.publication_valid = true;
        return true;
    }

    bool ROCmBackend::applyLogitPenaltiesF32(void *logits_device,
                                              const int *token_ids_host,
                                              const float *penalties_host,
                                              int num_penalties, int vocab_size,
                                              int device_id, void *stream, float repetition_penalty)
    {
        if (!std::isfinite(repetition_penalty) || repetition_penalty <= 0.0f)
            return false;
        if (device_id >= device_count_ || device_id < 0 || !logits_device ||
            !token_ids_host || !penalties_host || num_penalties <= 0)
            return false;

        auto &bufs = penalty_buffers_[static_cast<size_t>(device_id)];
        if (num_penalties > bufs.allocated_count ||
            !bufs.token_ids_ptr ||
            !bufs.penalties_ptr ||
            !bufs.ready_event)
            return false;

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        hipStream_t s =
            requireExplicitStream(stream, "ROCmBackend::applyLogitPenaltiesF32");

        if (bufs.publication_valid &&
            bufs.producer_stream != stream)
        {
            HIP_CHECK_OR_THROW(hipStreamWaitEvent(
                s,
                static_cast<hipEvent_t>(bufs.ready_event),
                0));
        }

        // Upload penalty data to device
        HIP_CHECK_OR_THROW(hipMemcpyAsync(bufs.token_ids_ptr, token_ids_host,
                                           num_penalties * sizeof(int),
                                           hipMemcpyHostToDevice, s));
        HIP_CHECK_OR_THROW(hipMemcpyAsync(bufs.penalties_ptr, penalties_host,
                                           num_penalties * sizeof(float),
                                           hipMemcpyHostToDevice, s));

        // Apply penalties in-place on device
        if (!rocmOps_apply_logit_penalties_f32(
                static_cast<float *>(logits_device),
                static_cast<const int *>(bufs.token_ids_ptr),
                static_cast<const float *>(bufs.penalties_ptr),
                num_penalties, vocab_size, device_id, s, repetition_penalty))
        {
            return false;
        }

        HIP_CHECK_OR_THROW(hipEventRecord(
            static_cast<hipEvent_t>(bufs.ready_event),
            s));
        bufs.producer_stream = stream;
        bufs.publication_valid = true;
        return true;
    }

    bool ROCmBackend::enqueueLogitPenaltiesF32Device(void *logits_device,
                                                     const void *token_ids_device,
                                                     const void *penalties_device,
                                                     int num_penalties,
                                                     int vocab_size,
                                                     int device_id,
                                                     void *stream, float repetition_penalty)
    {
        if (!std::isfinite(repetition_penalty) || repetition_penalty <= 0.0f)
            return false;
        if (device_id >= device_count_ || device_id < 0 || !logits_device ||
            !token_ids_device || !penalties_device || num_penalties <= 0 ||
            vocab_size <= 0 || !stream)
        {
            return false;
        }

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_apply_logit_penalties_f32(
            static_cast<float *>(logits_device),
            static_cast<const int *>(token_ids_device),
            static_cast<const float *>(penalties_device),
            num_penalties,
            vocab_size,
            device_id,
            stream, repetition_penalty);
    }
} // namespace llaminar2
