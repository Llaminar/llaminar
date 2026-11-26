/**
 * @file DeviceGraphOrchestratorOverlay.cpp
 * @brief DeviceGraphOrchestrator method shard (60110..65100).
 */
#include "DeviceGraphOrchestratorInternal.h"

namespace llaminar2
{

    bool DeviceGraphOrchestrator::verifyStochasticDistributionsBatchOutcomeOnDeviceCommon(
        int first_target_slot,
        int first_draft_slot,
        const int32_t *draft_tokens,
        const float *accept_thresholds,
        const float *residual_thresholds,
        const float *sample_thresholds,
        int row_count,
        int32_t first_token,
        int first_target_sample_slot,
        int first_token_row_offset,
        int first_token_row_stride,
        bool first_token_from_device,
        const int32_t *stop_tokens,
        int stop_token_count,
        int bonus_target_slot,
        float bonus_threshold,
        DeviceSpeculativeVerifyBatchOutcome *out,
        uint64_t inverse_sample_seed,
        int inverse_sample_first_logical_position,
        const int32_t *threshold_base_position_device,
        int threshold_position_offset,
        bool use_vllm_probability_rejection,
        bool serial_sample_equivalent,
        const int *generation_control_device,
        int output_request_slot,
        void *stream_override,
        bool copy_summary_to_host)
    {
        using namespace sampling_math;
        (void)draft_tokens;
        auto reject = [&](const std::string &reason) -> bool
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Stochastic compact outcome rejected on "
                      << state_.device_id.toString()
                      << ": " << reason
                      << " target_slot=" << first_target_slot
                      << " draft_slot=" << first_draft_slot
                      << " rows=" << row_count
                      << " bonus_slot=" << bonus_target_slot
                      << " output_slot=" << output_request_slot
                      << " first_token_from_device=" << boolTag(first_token_from_device)
                      << " first_target_sample_slot=" << first_target_sample_slot
                      << " token_row_offset=" << first_token_row_offset
                      << " token_row_stride=" << first_token_row_stride
                      << " materialized_batch_valid="
                      << boolTag(materialized_mtp_verifier_device_token_batch_.valid)
                      << " materialized_batch_requests="
                      << materialized_mtp_verifier_device_token_batch_.request_count
                      << " materialized_batch_logical_stride="
                      << materialized_mtp_verifier_device_token_batch_
                             .logical_padded_seq_len
                      << " materialized_batch_physical_stride="
                      << materialized_mtp_verifier_device_token_batch_
                             .physical_padded_seq_len
                      << " materialized_batch_capacity="
                      << materialized_mtp_verifier_device_token_batch_.total_token_capacity
                      << " serial_equivalent=" << boolTag(serial_sample_equivalent)
                      << " vllm_rejection=" << boolTag(use_vllm_probability_rejection)
                      << " host_position=" << inverse_sample_first_logical_position
                      << " device_position=" << boolTag(threshold_base_position_device != nullptr));
            return false;
        };
        if (!out)
            return reject("missing compact outcome destination");
        *out = DeviceSpeculativeVerifyBatchOutcome{};
        if (!generation_control_device || !mtp_verifier_stop_tokens_dev_)
        {
            return reject(
                "resident generation reducer has no device control row or request stop-policy row");
        }

        const bool has_bonus = bonus_target_slot >= 0;
        const bool derive_rejection_thresholds_from_seed =
            accept_thresholds == nullptr &&
            residual_thresholds == nullptr &&
            !serial_sample_equivalent &&
            use_vllm_probability_rejection &&
            inverse_sample_seed != 0 &&
            (inverse_sample_first_logical_position >= 0 ||
             threshold_base_position_device != nullptr);
        const bool has_host_thresholds =
            accept_thresholds != nullptr && residual_thresholds != nullptr;
        const bool has_sample_thresholds = sample_thresholds != nullptr;
        const bool derive_sample_thresholds_from_device_position =
            serial_sample_equivalent &&
            !has_sample_thresholds &&
            inverse_sample_seed != 0 &&
            inverse_sample_first_logical_position < 0 &&
            threshold_base_position_device != nullptr;
        const bool has_resident_threshold_position =
            threshold_base_position_device != nullptr;
        if (has_resident_threshold_position &&
            inverse_sample_first_logical_position >= 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Stochastic threshold position must be either device-owned or host-scalar, never both");
            return false;
        }
        const bool first_token_from_target_sample =
            first_token_from_device && first_target_sample_slot >= 0;
        const bool first_token_from_verifier_tokens =
            first_token_from_device && first_token_row_offset >= 0;
        if (!supportsDeviceStochasticMTPVerification())
            return reject("device stochastic verification workspace is unavailable");
        if (first_target_slot < 0 || first_draft_slot < 0)
            return reject("negative target or draft slot");
        if (row_count <= 0 || row_count > mtp_max_draft_depth_)
            return reject("row count is outside the compact verifier limit");
        const int verifier_token_count = row_count + 1;
        const bool materialized_batch_row_matches =
            mtp_verifier_input_tokens_dev_ &&
            materialized_mtp_verifier_device_token_batch_.valid &&
            first_token_row_offset >= 0 &&
            first_token_row_stride ==
                materialized_mtp_verifier_device_token_batch_
                    .logical_padded_seq_len &&
            materialized_mtp_verifier_device_token_batch_
                    .physical_padded_seq_len >= first_token_row_stride &&
            (materialized_mtp_verifier_device_token_batch_
                         .physical_padded_seq_len == first_token_row_stride ||
             (materialized_mtp_verifier_device_token_batch_.request_count == 1 &&
              first_token_row_offset == 0)) &&
            verifier_token_count <= first_token_row_stride &&
            first_token_row_offset + verifier_token_count <=
                materialized_mtp_verifier_device_token_batch_
                    .total_token_capacity;
        const bool materialized_scalar_row_matches =
            mtp_verifier_input_tokens_dev_ &&
            materialized_mtp_verifier_device_token_row_.valid &&
            first_token_row_offset == 0 &&
            first_token_row_stride == verifier_token_count &&
            materialized_mtp_verifier_device_token_row_
                    .total_verifier_input_tokens == verifier_token_count &&
            materialized_mtp_verifier_device_token_row_.draft_token_count ==
                row_count;
        if (first_target_slot + row_count > stochastic_target_row_capacity_)
            return reject("target rows exceed the allocated stochastic target capacity");
        if (first_draft_slot + row_count > stochastic_draft_row_capacity_)
            return reject("draft rows exceed the allocated stochastic draft capacity");
        if (stop_token_count < 0 ||
            stop_token_count > kSpeculativeBatchMaxStopTokens ||
            (stop_token_count > 0 && !stop_tokens))
        {
            return reject("stop-token descriptor is malformed");
        }
        if (first_token_from_device &&
            first_token_from_target_sample == first_token_from_verifier_tokens)
        {
            return reject("device first token must name exactly one resident source");
        }
        if (first_token_from_target_sample &&
            (first_target_sample_slot >= stochastic_target_row_capacity_ ||
             !stochastic_target_sample_tokens_dev_))
        {
            return reject("target-sample first-token source is outside the resident sample workspace");
        }
        if (first_token_from_verifier_tokens &&
            !materialized_batch_row_matches &&
            !materialized_scalar_row_matches)
        {
            return reject("verifier-token first-token source does not match a complete materialized verifier row");
        }
        if (has_bonus && bonus_target_slot >= stochastic_target_row_capacity_)
            return reject("bonus row exceeds the allocated stochastic target capacity");
        if (output_request_slot < 0 ||
            output_request_slot >= stochastic_batch_output_request_capacity_)
        {
            return reject("compact output request slot exceeds the allocated batch capacity");
        }
        if (serial_sample_equivalent &&
            !has_sample_thresholds &&
            !derive_sample_thresholds_from_device_position)
        {
            return reject("serial-equivalent sampling has no explicit or device-position threshold authority");
        }
        if (!serial_sample_equivalent &&
            !has_host_thresholds &&
            !derive_rejection_thresholds_from_seed)
        {
            return reject("probability rejection has no explicit or seeded threshold authority");
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return reject("backend lookup failed");
        /*
         * If the verifier replay was launch-only deferred, all target
         * distribution builds have already been enqueued on this stream. The
         * batch reducer and compact D2H copy consume the same stream and close
         * the handoff. Falling back to a fresh explicit stream would race the
         * verifier rows, so consume-and-clear the pending stream here.
         */
        void *stream = stream_override;
        if (!stream)
        {
            stream = consumePendingLogitsStream(
                PendingLogitsStreamRole::AllPositionVerifier,
                "verifyStochasticDistributionsBatchOutcomeOnDevice");
        }
        if (!stream)
        {
            stream = explicitGPUStreamForOperation(
                "verifyStochasticDistributionsBatchOutcomeOnDevice");
        }
        if (!stream)
            return reject("no explicit verifier stream is available");

        const int target_top_k =
            stochastic_target_top_k_[static_cast<size_t>(first_target_slot)];
        const StochasticRowFormat target_row_format =
            stochastic_target_row_formats_[static_cast<size_t>(first_target_slot)];
        const bool use_processed_target_rows =
            use_vllm_probability_rejection &&
            target_row_format == StochasticRowFormat::ProcessedLogits;
        const bool verifier_uses_one_hot_draft_tokens =
            use_vllm_probability_rejection || serial_sample_equivalent;
        if (target_top_k <= 0 ||
            target_top_k > static_cast<int>(kStochasticDistributionMaxK))
        {
            return reject("target distribution has an invalid top-k width");
        }
        if (target_row_format != StochasticRowFormat::CompactDistribution &&
            target_row_format != StochasticRowFormat::ProcessedLogits)
        {
            return reject("target distribution row format is empty or unsupported");
        }

        for (int row = 0; row < row_count; ++row)
        {
            const int target_slot = first_target_slot + row;
            const int draft_slot = first_draft_slot + row;
            const int draft_top_k =
                stochastic_draft_top_k_[static_cast<size_t>(draft_slot)];
            if (stochastic_target_top_k_[static_cast<size_t>(target_slot)] != target_top_k ||
                stochastic_target_row_formats_[static_cast<size_t>(target_slot)] != target_row_format ||
                (!verifier_uses_one_hot_draft_tokens &&
                 draft_top_k != target_top_k))
            {
                return reject(
                    "target/draft distribution rows disagree on top-k width or row format at row " +
                    std::to_string(row));
            }
        }
        if (has_bonus &&
            (stochastic_target_top_k_[static_cast<size_t>(bonus_target_slot)] != target_top_k ||
             stochastic_target_row_formats_[static_cast<size_t>(bonus_target_slot)] != target_row_format))
        {
            return reject("bonus target distribution disagrees with verifier row format");
        }

        const int full_vocab_size =
            graph_builder_ ? graph_builder_->config().vocab_size : state_.vocab_size;
        if (full_vocab_size <= 0)
            return reject("graph exposes no positive full-vocabulary size");
        auto *target_ids =
            static_cast<int *>(stochastic_target_token_ids_dev_) +
            static_cast<size_t>(first_target_slot) * kStochasticDistributionMaxK;
        auto *target_probs =
            static_cast<float *>(stochastic_target_probs_dev_) +
            static_cast<size_t>(first_target_slot) * kStochasticDistributionMaxK;
        auto *draft_ids =
            static_cast<int *>(stochastic_draft_token_ids_dev_) +
            static_cast<size_t>(first_draft_slot) * kStochasticDistributionMaxK;
        auto *draft_probs =
            static_cast<float *>(stochastic_draft_probs_dev_) +
            static_cast<size_t>(first_draft_slot) * kStochasticDistributionMaxK;
        auto *sampled_draft_tokens =
            static_cast<int *>(stochastic_draft_sample_tokens_dev_) + first_draft_slot;
        auto *sampled_draft_probs =
            (!verifier_uses_one_hot_draft_tokens && stochastic_draft_sample_probs_dev_)
                ? static_cast<float *>(stochastic_draft_sample_probs_dev_) + first_draft_slot
                : nullptr;
        auto *processed_target_logits =
            use_processed_target_rows
                ? static_cast<float *>(stochastic_processed_logits_dev_) +
                      static_cast<size_t>(first_target_slot) *
                          static_cast<size_t>(full_vocab_size)
                : nullptr;
        auto *out_token_dev =
            static_cast<int *>(stochastic_verify_tokens_dev_) + first_target_slot;
        auto *out_accepted_dev =
            static_cast<int *>(stochastic_verify_accepted_dev_) + first_target_slot;
        auto *out_accept_prob_dev =
            static_cast<float *>(stochastic_verify_accept_probs_dev_) + first_target_slot;
        auto *out_threshold_dev =
            static_cast<float *>(stochastic_verify_thresholds_dev_) + first_target_slot;
        auto *summary_tokens_dev =
            static_cast<int32_t *>(stochastic_batch_output_tokens_dev_) +
            static_cast<size_t>(output_request_slot) *
                stochastic_batch_output_token_stride_;
        auto *summary_meta_dev =
            static_cast<int *>(stochastic_batch_output_meta_dev_) +
            static_cast<size_t>(output_request_slot) * kSpeculativeBatchMetaCount;
        const int32_t *summary_draft_tokens_device = nullptr;
        if (materialized_batch_row_matches)
        {
            summary_draft_tokens_device =
                static_cast<const int32_t *>(mtp_verifier_input_tokens_dev_) +
                first_token_row_offset;
        }
        else if (materialized_scalar_row_matches)
        {
            summary_draft_tokens_device =
                static_cast<const int32_t *>(mtp_verifier_input_tokens_dev_);
        }
        if (serial_sample_equivalent && !summary_draft_tokens_device)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Serial-equivalent stochastic MTP outcome requires a materialized verifier-token row");
            return false;
        }

        if (!waitForRequiredStochasticDraftSampleReadyRange(
                first_draft_slot,
                row_count,
                stream,
                "stochastic_batch_verifier"))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to order stochastic batch verifier after deferred draft samples");
            return false;
        }

        if (debugEnv().validation.validate_buffers)
        {
            std::vector<int32_t> sampled_drafts(
                static_cast<size_t>(row_count));
            if (!backend->deviceToHostFast(
                    sampled_drafts.data(),
                    sampled_draft_tokens,
                    sizeof(int32_t) * static_cast<size_t>(row_count),
                    state_.device_id.gpu_ordinal(),
                    stream))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to inspect deferred stochastic draft tokens");
                return false;
            }
            for (int row = 0; row < row_count; ++row)
            {
                const int32_t token = sampled_drafts[static_cast<size_t>(row)];
                if (token < 0)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Deferred stochastic draft sample slot="
                              << (first_draft_slot + row)
                              << " is invalid before batch verifier: token="
                              << token);
                    return false;
                }
            }
        }

        if (serial_sample_equivalent)
        {
            if (!has_bonus ||
                target_row_format != StochasticRowFormat::CompactDistribution ||
                (!has_sample_thresholds &&
                 !derive_sample_thresholds_from_device_position) ||
                first_target_slot + row_count >= stochastic_target_row_capacity_)
            {
                return reject("serial-equivalent reducer requires compact target rows, a bonus row, and device-position thresholds");
            }

            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "stochastic_serial_equivalent_target_sample_enqueue",
                    "decode",
                    state_.device_id.toString(),
                    {{"rows", std::to_string(row_count)},
                     {"top_k", std::to_string(target_top_k)}});
                for (int row = 0; row < row_count; ++row)
                {
                    const int target_slot = first_target_slot + row;
                    auto *row_ids =
                        static_cast<int *>(stochastic_target_token_ids_dev_) +
                        static_cast<size_t>(target_slot) *
                            kStochasticDistributionMaxK;
                    auto *row_probs =
                        static_cast<float *>(stochastic_target_probs_dev_) +
                        static_cast<size_t>(target_slot) *
                            kStochasticDistributionMaxK;
                    if (!backend->enqueueSampleDistributionF32Device(
                            row_ids,
                            row_probs,
                            target_top_k,
                            has_sample_thresholds
                                ? sample_thresholds[row]
                                : 0.0f,
                            state_.device_id.gpu_ordinal(),
                            stream,
                            out_token_dev + row,
                            /*out_probability_device=*/nullptr,
                            derive_sample_thresholds_from_device_position
                                ? inverse_sample_seed
                                : 0,
                            derive_sample_thresholds_from_device_position
                                ? threshold_base_position_device
                                : nullptr,
                            threshold_position_offset + row))
                    {
                        return reject(
                            "backend rejected serial-equivalent target sample row " +
                            std::to_string(row));
                    }
                }

                auto *bonus_ids =
                    static_cast<int *>(stochastic_target_token_ids_dev_) +
                    static_cast<size_t>(bonus_target_slot) *
                        kStochasticDistributionMaxK;
                auto *bonus_probs =
                    static_cast<float *>(stochastic_target_probs_dev_) +
                    static_cast<size_t>(bonus_target_slot) *
                        kStochasticDistributionMaxK;
                if (!backend->enqueueSampleDistributionF32Device(
                        bonus_ids,
                        bonus_probs,
                        target_top_k,
                        derive_sample_thresholds_from_device_position
                            ? 0.0f
                            : bonus_threshold,
                        state_.device_id.gpu_ordinal(),
                        stream,
                        out_token_dev + row_count,
                        /*out_probability_device=*/nullptr,
                        derive_sample_thresholds_from_device_position
                            ? inverse_sample_seed
                            : 0,
                        derive_sample_thresholds_from_device_position
                            ? threshold_base_position_device
                            : nullptr,
                        threshold_position_offset + row_count))
                {
                    return reject("backend rejected the serial-equivalent bonus sample row");
                }
            }

            bool summary_enqueued = false;
            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "stochastic_serial_equivalent_summary_enqueue",
                    "decode",
                    state_.device_id.toString(),
                    {{"rows", std::to_string(row_count)},
                     {"top_k", std::to_string(target_top_k)}});
                summary_enqueued =
                    backend->enqueueSummarizeSpeculativeVerifyBatchDeviceGenerationControls(
                        out_token_dev,
                        /*verify_accepted_device=*/nullptr,
                        summary_draft_tokens_device,
                        row_count,
                        summary_draft_tokens_device,
                        mtp_verifier_stop_tokens_dev_,
                        out_token_dev + row_count,
                        /*has_bonus_token=*/true,
                        generation_control_device,
                        state_.device_id.gpu_ordinal(),
                        stream,
                        stochastic_batch_output_token_stride_,
                        summary_tokens_dev,
                        summary_meta_dev);
            }
            if (!summary_enqueued)
                return reject("backend rejected the serial-equivalent compact summary launch");

            logMTPSerialEquivalentStochasticDecisionDiagnostics(
                *backend,
                state_.device_id,
                stream,
                inverse_sample_seed,
                threshold_base_position_device,
                threshold_position_offset,
                target_ids,
                target_probs,
                static_cast<int>(kStochasticDistributionMaxK),
                target_top_k,
                row_count + 1,
                summary_draft_tokens_device,
                verifier_token_count,
                out_token_dev,
                summary_tokens_dev,
                stochastic_batch_output_token_stride_,
                reinterpret_cast<const int32_t *>(summary_meta_dev),
                kSpeculativeBatchMetaCount);

            PerfStatsCollector::addCounter(
                "mtp",
                "stochastic_serial_equivalent_verify_batch_device_token_rows",
                static_cast<double>(row_count),
                "decode",
                state_.device_id.toString(),
                {{"top_k", std::to_string(target_top_k)}});
            if (derive_sample_thresholds_from_device_position)
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "stochastic_serial_equivalent_resident_position_sample_rows",
                    static_cast<double>(row_count + 1),
                    "decode",
                    state_.device_id.toString(),
                    {{"position_owner", "verifier_base_snapshot"},
                     {"includes_bonus", "true"}});
            }

            if (!copy_summary_to_host)
                return true;

            std::vector<int32_t> output_tokens(
                static_cast<size_t>(stochastic_batch_output_token_stride_));
            std::array<int, kSpeculativeBatchMetaCount> meta{};
            if (!backend->deviceToHostFast(output_tokens.data(),
                                           summary_tokens_dev,
                                           sizeof(int32_t) * output_tokens.size(),
                                           state_.device_id.gpu_ordinal(),
                                           stream) ||
                !backend->deviceToHostFast(meta.data(),
                                           summary_meta_dev,
                                           sizeof(int) * meta.size(),
                                           state_.device_id.gpu_ordinal(),
                                           stream))
            {
                return false;
            }
            return fillSpeculativeVerifyOutcomeFromMeta(
                output_tokens.data(),
                stochastic_batch_output_token_stride_,
                meta.data(),
                out);
        }

        /*
         * First launch: row-local stochastic accept/reject decisions. The
         * sampled MTP draft tokens come from STOCHASTIC_DRAFT_SAMPLE_TOKENS,
         * which is written by sampleStochasticDistributionOnDevice(). The
         * compact outcome path never reads a host draft-token shadow.
         */
        bool verify_enqueued = false;
        if (sampled_draft_tokens)
        {
            if (use_vllm_probability_rejection)
            {
                if (first_target_slot + row_count > stochastic_target_row_capacity_ ||
                    first_draft_slot + row_count > stochastic_draft_row_capacity_)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] vLLM stochastic verifier requested invalid compact target rows");
                    return false;
                }

                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    use_processed_target_rows
                        ? "stochastic_vllm_processed_batch_verify_enqueue"
                        : "stochastic_vllm_compact_batch_verify_enqueue",
                    "decode",
                    state_.device_id.toString(),
                    {{"rows", std::to_string(row_count)},
                     {"top_k", std::to_string(target_top_k)},
                     {"draft", "one_hot"},
                     {"target_format", use_processed_target_rows
                                           ? "processed_logits"
                                           : "compact_distribution"}});
                if (use_processed_target_rows)
                {
                    verify_enqueued =
                        processed_target_logits &&
                        backend->enqueueSpeculativeVerifyProcessedTargetDraftProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
                            processed_target_logits,
                            /*draft_probabilities_device=*/nullptr,
                            row_count,
                            full_vocab_size,
                            full_vocab_size,
                            full_vocab_size,
                            sampled_draft_tokens,
                            accept_thresholds,
                            inverse_sample_seed,
                            inverse_sample_first_logical_position,
                            state_.device_id.gpu_ordinal(),
                            stream,
                            out_token_dev,
                            out_accepted_dev,
                            out_accept_prob_dev,
                            out_threshold_dev,
                            /*no_draft_probabilities=*/true,
                            threshold_base_position_device,
                            threshold_position_offset);
                }
                else
                {
                    verify_enqueued =
                        backend->enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatchDeviceTokens(
                            target_ids,
                            target_probs,
                            /*draft_token_ids_device=*/nullptr,
                            /*draft_probs_device=*/nullptr,
                            target_top_k,
                            static_cast<int>(kStochasticDistributionMaxK),
                            sampled_draft_tokens,
                            accept_thresholds,
                            residual_thresholds,
                            row_count,
                            state_.device_id.gpu_ordinal(),
                            stream,
                            out_token_dev,
                            out_accepted_dev,
                            out_accept_prob_dev,
                            out_threshold_dev,
                            /*draft_token_probabilities_device=*/nullptr,
                            inverse_sample_seed,
                            inverse_sample_first_logical_position,
                            full_vocab_size,
                            threshold_base_position_device,
                            threshold_position_offset);
                }
            }
            else
            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "stochastic_batch_verify_enqueue",
                    "decode",
                    state_.device_id.toString(),
                    {{"rows", std::to_string(row_count)},
                     {"top_k", std::to_string(target_top_k)}});
                verify_enqueued =
                    backend->enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatchDeviceTokens(
                        target_ids,
                        target_probs,
                        draft_ids,
                        draft_probs,
                        target_top_k,
                        static_cast<int>(kStochasticDistributionMaxK),
                        sampled_draft_tokens,
                        accept_thresholds,
                        residual_thresholds,
                        row_count,
                        state_.device_id.gpu_ordinal(),
                        stream,
                        out_token_dev,
                        out_accepted_dev,
                        out_accept_prob_dev,
                        out_threshold_dev,
                        sampled_draft_probs);
            }
        }
        if (!verify_enqueued)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] GPU stochastic batch verifier requires device-resident draft tokens");
            return false;
        }
        PerfStatsCollector::addCounter(
            "mtp",
            use_vllm_probability_rejection
                ? "stochastic_vllm_compact_one_hot_verify_batch_device_token_rows"
                : "stochastic_verify_batch_device_token_rows",
            static_cast<double>(row_count),
            "decode",
            state_.device_id.toString(),
            {{"top_k", std::to_string(target_top_k)}});
        if (has_resident_threshold_position)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "stochastic_request_batch_resident_position_threshold_rows",
                static_cast<double>(row_count),
                "decode",
                state_.device_id.toString(),
                {{"target_format",
                  use_processed_target_rows
                      ? "processed_logits"
                      : "compact_distribution"}});
        }

        if (debugEnv().validation.validate_buffers)
        {
            std::vector<int32_t> verifier_tokens(static_cast<size_t>(row_count));
            std::vector<int32_t> verifier_accepted(static_cast<size_t>(row_count));
            if (!backend->deviceToHostFast(
                    verifier_tokens.data(),
                    out_token_dev,
                    sizeof(int32_t) * static_cast<size_t>(row_count),
                    state_.device_id.gpu_ordinal(),
                    stream) ||
                !backend->deviceToHostFast(
                    verifier_accepted.data(),
                    out_accepted_dev,
                    sizeof(int32_t) * static_cast<size_t>(row_count),
                    state_.device_id.gpu_ordinal(),
                    stream))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to inspect stochastic verifier row outputs");
                return false;
            }
            for (int row = 0; row < row_count; ++row)
            {
                const int32_t token = verifier_tokens[static_cast<size_t>(row)];
                if (token < 0)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Stochastic verifier row="
                              << row << " produced invalid token=" << token
                              << " accepted="
                              << verifier_accepted[static_cast<size_t>(row)]);
                    return false;
                }
            }
        }

        std::array<int, kSpeculativeBatchMaxStopTokens> packed_stop_tokens =
            {-1, -1, -1, -1, -1, -1, -1, -1};
        for (int i = 0; i < stop_token_count; ++i)
            packed_stop_tokens[static_cast<size_t>(i)] = stop_tokens[i];

        const int *first_token_dev = nullptr;
        if (first_token_from_device)
        {
            if (first_token_from_verifier_tokens)
            {
                first_token_dev =
                    static_cast<const int *>(mtp_verifier_input_tokens_dev_) +
                    first_token_row_offset;
            }
            else
            {
                first_token_dev =
                    static_cast<const int *>(stochastic_target_sample_tokens_dev_) +
                    first_target_sample_slot;
                if (!waitForRequiredStochasticTargetSampleReady(
                        first_target_sample_slot,
                        stream,
                        "stochastic_batch_summary_first_token"))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Failed to order stochastic batch summary after deferred target sample");
                    return false;
                }
            }
        }

        const int *bonus_token_dev = nullptr;
        if (has_bonus)
        {
            auto *bonus_out =
                static_cast<int *>(stochastic_verify_tokens_dev_) + bonus_target_slot;
            bool bonus_enqueued = false;
            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    use_processed_target_rows
                        ? "stochastic_batch_processed_bonus_sample_enqueue"
                        : "stochastic_batch_bonus_sample_enqueue",
                    "decode",
                    state_.device_id.toString(),
                    {{"top_k", std::to_string(target_top_k)},
                     {"lazy", use_processed_target_rows ? "true" : "false"},
                     {"verifier", "compact_draft_table"}});
                if (use_processed_target_rows)
                {
                    auto *bonus_logits =
                        static_cast<float *>(stochastic_processed_logits_dev_) +
                        static_cast<size_t>(bonus_target_slot) *
                            static_cast<size_t>(full_vocab_size);
                    bonus_enqueued =
                        backend->enqueueSampleProcessedLogitsF32DeviceIfSpeculativeBatchNeedsBonus(
                            bonus_logits,
                            full_vocab_size,
                            full_vocab_size,
                            bonus_threshold,
                            out_token_dev,
                            out_accepted_dev,
                            row_count,
                            first_token_from_device ? -1 : first_token,
                            first_token_dev,
                            packed_stop_tokens.data(),
                            stop_token_count,
                            state_.device_id.gpu_ordinal(),
                            stream,
                            bonus_out,
                            /*out_probability_device=*/nullptr,
                            inverse_sample_seed,
                            threshold_base_position_device,
                            threshold_position_offset + row_count);
                }
                else
                {
                    auto *bonus_ids =
                        static_cast<int *>(stochastic_target_token_ids_dev_) +
                        static_cast<size_t>(bonus_target_slot) * kStochasticDistributionMaxK;
                    auto *bonus_probs =
                        static_cast<float *>(stochastic_target_probs_dev_) +
                        static_cast<size_t>(bonus_target_slot) * kStochasticDistributionMaxK;
                    bonus_enqueued = backend->enqueueSampleDistributionF32Device(
                        bonus_ids,
                        bonus_probs,
                        target_top_k,
                        bonus_threshold,
                        state_.device_id.gpu_ordinal(),
                        stream,
                        bonus_out,
                        /*out_probability_device=*/nullptr,
                        inverse_sample_seed,
                        threshold_base_position_device,
                        threshold_position_offset + row_count);
                }
            }
            if (!bonus_enqueued)
            {
                return false;
            }
            bonus_token_dev = bonus_out;
        }

        bool summary_enqueued = false;
        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "stochastic_batch_summary_enqueue",
                "decode",
                state_.device_id.toString(),
                {{"rows", std::to_string(row_count)},
                 {"top_k", std::to_string(target_top_k)},
                 {"first_token_source",
                  first_token_from_device
                      ? (first_token_from_verifier_tokens
                             ? "verifier_token_batch"
                             : "device")
                      : "host"},
                 {"has_bonus", has_bonus ? "true" : "false"}});
            if (!first_token_from_device)
            {
                return reject(
                    "resident generation summary requires a device-owned first token");
            }
            summary_enqueued =
                backend->enqueueSummarizeSpeculativeVerifyBatchDeviceGenerationControls(
                    out_token_dev,
                    out_accepted_dev,
                    /*greedy_draft_tokens_device=*/nullptr,
                    row_count,
                    first_token_dev,
                    mtp_verifier_stop_tokens_dev_,
                    bonus_token_dev,
                    has_bonus,
                    generation_control_device,
                    state_.device_id.gpu_ordinal(),
                    stream,
                    stochastic_batch_output_token_stride_,
                    summary_tokens_dev,
                    summary_meta_dev);
        }
        if (!summary_enqueued)
        {
            return false;
        }
        if (first_token_from_device)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "stochastic_batch_summary_device_first_tokens",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"source",
                  first_token_from_verifier_tokens
                      ? "verifier_token_batch"
                      : "target_sample_slot"},
                 {"slot", std::to_string(first_target_sample_slot)}});
        }

        std::vector<int32_t> output_tokens(
            static_cast<size_t>(stochastic_batch_output_token_stride_));
        std::array<int, kSpeculativeBatchMetaCount> meta{};
        const bool capture_row_debug =
            debugEnv().validation.validate_buffers ||
            Logger::getInstance().shouldLog(LogLevel::VERBOSITY_DEBUG);
        std::vector<float> debug_accept_probs(static_cast<size_t>(row_count));
        std::vector<float> debug_accept_thresholds(static_cast<size_t>(row_count));
        std::vector<int32_t> debug_draft_tokens(static_cast<size_t>(row_count));
        std::vector<float> debug_draft_probs(static_cast<size_t>(row_count));
        /*
         * The batch metadata is fixed-shape and the token payload is bounded by
         * the graph's configured verifier-row capacity. Fast D2H is safe here
         * and keeps ROCm from spending more time validating the pointer than
         * copying the result.
         */
        bool copied_summary = !copy_summary_to_host;
        if (copy_summary_to_host)
        {
            /*
             * This is the intentional host-visible boundary for the compact
             * stochastic outcome. Its time includes any previously enqueued
             * distribution or verifier kernels that have not yet completed.
             */
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "stochastic_batch_summary_d2h_sync",
                "decode",
                state_.device_id.toString(),
                {{"rows", std::to_string(row_count)},
                 {"top_k", std::to_string(target_top_k)}});
            copied_summary =
                backend->deviceToHostFast(output_tokens.data(),
                                          summary_tokens_dev,
                                          sizeof(int32_t) * output_tokens.size(),
                                          state_.device_id.gpu_ordinal(),
                                          stream) &&
                backend->deviceToHostFast(meta.data(),
                                          summary_meta_dev,
                                          sizeof(int) * meta.size(),
                                          state_.device_id.gpu_ordinal(),
                                          stream);
            /*
             * The row-detail arrays exist only for debug logging and buffer
             * validation.  Keep production verification to the two semantic
             * transfers above: output tokens and compact outcome metadata.
             * On ROCm each fast D2H copy synchronizes the stream, so copying
             * debug-only rows here would turn one intentional host boundary
             * into several extra synchronization points.
             */
            if (copy_summary_to_host && copied_summary && capture_row_debug)
            {
                copied_summary =
                    backend->deviceToHostFast(debug_accept_probs.data(),
                                              out_accept_prob_dev,
                                              sizeof(float) * static_cast<size_t>(row_count),
                                              state_.device_id.gpu_ordinal(),
                                              stream) &&
                    backend->deviceToHostFast(debug_accept_thresholds.data(),
                                              out_threshold_dev,
                                              sizeof(float) * static_cast<size_t>(row_count),
                                              state_.device_id.gpu_ordinal(),
                                              stream) &&
                    backend->deviceToHostFast(debug_draft_tokens.data(),
                                              sampled_draft_tokens,
                                              sizeof(int32_t) * static_cast<size_t>(row_count),
                                              state_.device_id.gpu_ordinal(),
                                              stream) &&
                    (!sampled_draft_probs ||
                     backend->deviceToHostFast(debug_draft_probs.data(),
                                               sampled_draft_probs,
                                               sizeof(float) * static_cast<size_t>(row_count),
                                               state_.device_id.gpu_ordinal(),
                                               stream));
            }
        }
        if (!copied_summary)
        {
            return false;
        }
        if (copy_summary_to_host)
        {
            drainPendingGpuTimingMeasurements(backend);
        }
        if (!copy_summary_to_host)
        {
            for (int row = 0; row < row_count; ++row)
            {
                stochastic_target_distribution_streams_[
                    static_cast<size_t>(first_target_slot + row)] = nullptr;
                stochastic_target_row_formats_[
                    static_cast<size_t>(first_target_slot + row)] =
                    StochasticRowFormat::Empty;
                stochastic_draft_distribution_streams_[
                    static_cast<size_t>(first_draft_slot + row)] = nullptr;
                stochastic_draft_row_formats_[
                    static_cast<size_t>(first_draft_slot + row)] =
                    StochasticRowFormat::Empty;
                stochastic_draft_top_k_[static_cast<size_t>(first_draft_slot + row)] = 0;
                clearStochasticDraftSampleReadySlot(
                    first_draft_slot + row,
                    StochasticSampleReadyClearMode::Force);
            }
            if (has_bonus)
            {
                stochastic_target_distribution_streams_[
                    static_cast<size_t>(bonus_target_slot)] = nullptr;
                stochastic_target_row_formats_[
                    static_cast<size_t>(bonus_target_slot)] =
                    StochasticRowFormat::Empty;
            }
            if (first_token_from_target_sample)
            {
                clearStochasticTargetSampleReadySlot(
                    first_target_sample_slot,
                    StochasticSampleReadyClearMode::Force);
            }

            PerfStatsCollector::addCounter(
                "mtp",
                "stochastic_verify_batch_outcome_rows",
                static_cast<double>(row_count),
                "decode",
                state_.device_id.toString(),
                {{"top_k", std::to_string(target_top_k)}});
            return true;
        }
        if (capture_row_debug && row_count > 0)
        {
            std::ostringstream row_debug;
            row_debug << "[DeviceGraphOrchestrator] stochastic batch outcome rows="
                      << row_count
                      << " output_count=" << meta[kSpecBatchMetaOutputCount]
                      << " accepted_prefix=" << meta[kSpecBatchMetaAcceptedSpeculativePrefix]
                      << " all_accepted=" << meta[kSpecBatchMetaAllSpeculativeAccepted]
                      << " sampled_terminal=" << meta[kSpecBatchMetaSampledTerminal]
                      << " commit_boundary_clipped="
                      << meta[kSpecBatchMetaCommitBoundaryClipped]
                      << " ready_token=" << meta[kSpecBatchMetaReadyToken]
                      << " rejected_token=" << meta[kSpecBatchMetaRejectedVerifiedToken]
                      << " per_row=";
            for (int row = 0; row < row_count; ++row)
            {
                if (row > 0)
                    row_debug << ",";
                row_debug << "{p=" << debug_accept_probs[static_cast<size_t>(row)]
                          << ",u=" << debug_accept_thresholds[static_cast<size_t>(row)]
                          << ",draft=" << debug_draft_tokens[static_cast<size_t>(row)]
                          << ",q=";
                if (sampled_draft_probs)
                    row_debug << debug_draft_probs[static_cast<size_t>(row)];
                else
                    row_debug << "one_hot";
                row_debug
                          << "}";
            }
            LOG_DEBUG(row_debug.str());
        }

        if (meta[kSpecBatchMetaOk] == 0 ||
            meta[kSpecBatchMetaOutputCount] < 0 ||
            meta[kSpecBatchMetaOutputCount] >
                stochastic_batch_output_token_stride_)
        {
            return false;
        }
        for (int i = 0; i < meta[kSpecBatchMetaOutputCount]; ++i)
        {
            if (output_tokens[static_cast<size_t>(i)] < 0)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Stochastic batch summary produced invalid output token="
                          << output_tokens[static_cast<size_t>(i)]
                          << " at index=" << i
                          << " output_count=" << meta[kSpecBatchMetaOutputCount]
                          << " accepted_prefix="
                          << meta[kSpecBatchMetaAcceptedSpeculativePrefix]
                          << " consumed_rows="
                          << meta[kSpecBatchMetaConsumedVerifierRows]);
                return false;
            }
        }

        for (int row = 0; row < row_count; ++row)
        {
            stochastic_target_distribution_streams_[
                static_cast<size_t>(first_target_slot + row)] = nullptr;
            stochastic_target_row_formats_[
                static_cast<size_t>(first_target_slot + row)] =
                StochasticRowFormat::Empty;
            stochastic_draft_distribution_streams_[
                static_cast<size_t>(first_draft_slot + row)] = nullptr;
            stochastic_draft_row_formats_[
                static_cast<size_t>(first_draft_slot + row)] =
                StochasticRowFormat::Empty;
            stochastic_draft_top_k_[static_cast<size_t>(first_draft_slot + row)] = 0;
            clearStochasticDraftSampleReadySlot(
                first_draft_slot + row,
                StochasticSampleReadyClearMode::Force);
        }
        if (has_bonus)
        {
            stochastic_target_distribution_streams_[
                static_cast<size_t>(bonus_target_slot)] = nullptr;
            stochastic_target_row_formats_[
                static_cast<size_t>(bonus_target_slot)] =
                StochasticRowFormat::Empty;
        }
        if (first_token_from_target_sample)
        {
            clearStochasticTargetSampleReadySlot(
                first_target_sample_slot,
                StochasticSampleReadyClearMode::Force);
        }

        out->ok = true;
        out->output_tokens = output_tokens;
        out->output_token_count = meta[kSpecBatchMetaOutputCount];
        out->accepted_speculative_prefix =
            meta[kSpecBatchMetaAcceptedSpeculativePrefix];
        out->target_verifier_state_commit_count =
            meta[kSpecBatchMetaTargetVerifierStateCommitCount];
        out->ready_token = meta[kSpecBatchMetaReadyToken];
        out->rejected_verified_token = meta[kSpecBatchMetaRejectedVerifiedToken];
        out->stopped_on_output = meta[kSpecBatchMetaStoppedOnOutput] != 0;
        out->all_speculative_accepted =
            meta[kSpecBatchMetaAllSpeculativeAccepted] != 0;
        out->consumed_verifier_rows = meta[kSpecBatchMetaConsumedVerifierRows];
        out->sampled_terminal = meta[kSpecBatchMetaSampledTerminal] != 0;
        out->commit_boundary_clipped =
            meta[kSpecBatchMetaCommitBoundaryClipped] != 0;

        PerfStatsCollector::addCounter(
            "mtp",
            "stochastic_verify_batch_outcome_rows",
            static_cast<double>(row_count),
            "decode",
            state_.device_id.toString(),
            {{"top_k", std::to_string(target_top_k)}});
        return true;
    }

    // =========================================================================
    // Batch Interface Implementation
    // =========================================================================

    bool DeviceGraphOrchestrator::forward_batch(
        const std::vector<std::vector<int>> &token_batches)
    {
        return forwardHostTokenBatchImpl(
            token_batches,
            ForwardExecutionRole::MainInference);
    }

    bool DeviceGraphOrchestrator::forwardHostTokenBatchImpl(
        const std::vector<std::vector<int>> &token_batches,
        ForwardExecutionRole execution_role)
    {
        // Enable device-scoped logging for this execution
        ScopedDeviceLog device_log(state_.device_id);

        const bool grouped_mtp_verifier =
            execution_role == ForwardExecutionRole::GroupedMTPVerifier;
        if (execution_role != ForwardExecutionRole::MainInference &&
            !grouped_mtp_verifier)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Host-token batch received an "
                "unsupported execution role");
            return false;
        }
        if (grouped_mtp_verifier && !state_.device_id.is_cpu())
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Host-token grouped MTP verification "
                "cannot execute on a GPU");
            return false;
        }

        const char *operation = grouped_mtp_verifier
                                    ? "forwardGroupedMTPVerifierWithHostTokenIds"
                                    : "forward_batch";

        if (token_batches.empty())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] " << operation
                                                    << " called with an empty batch");
            return false;
        }

        int batch_size = static_cast<int>(token_batches.size());
        if (batch_size > state_.batch_size)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] " << operation
                                                    << " batch size " << batch_size
                                                    << " exceeds initialized batch size "
                                                    << state_.batch_size);
            return false;
        }

        // Find max sequence length (for padding)
        int max_len = 0;
        for (const auto &seq : token_batches)
        {
            max_len = std::max(max_len, static_cast<int>(seq.size()));
        }
        if (max_len <= 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] " << operation
                                                    << " requires at least one token per batch row");
            return false;
        }
        if (grouped_mtp_verifier &&
            std::any_of(
                token_batches.begin(),
                token_batches.end(),
                [](const auto &row) { return row.empty(); }))
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Grouped MTP verifier batches cannot "
                "contain an empty request row");
            return false;
        }
        padded_seq_len_ = max_len;

        /*
         * Keep physical graph width and logical request width separate. The
         * explicit length row travels with this transaction into admission,
         * masking, KV append, and terminal-row selection. Mutable runner progress
         * is restored from old+actual after the rectangular graph is submitted.
         */
        std::vector<int> actual_lengths(batch_size);
        for (int i = 0; i < batch_size; ++i)
        {
            actual_lengths[i] = static_cast<int>(token_batches[i].size());
        }
        std::vector<int> old_positions = state_.positions;
        std::vector<int> old_sequence_lengths = state_.sequence_lengths;

        /*
         * Only ordinary request-batched prefill needs synthetic terminal-row
         * logits. Grouped MTP verification has its own typed role, keeps
         * all-position logits enabled, and carries explicit verifier-row
         * metadata. Role identity, rather than mutable logits mode, keeps the
         * two graph families disjoint.
         */
        const bool compact_prefill_logits =
            execution_role == ForwardExecutionRole::MainInference &&
            state_.device_id.is_gpu() &&
            batch_size > 1 &&
            !compute_all_position_logits_;
        if (compact_prefill_logits &&
            graph_builder_ &&
            graph_builder_->config().mtp.enabled)
        {
            /*
             * The first sampled token immediately completes a resident mailbox
             * row. Size its persistent metadata before graph construction so
             * admission-time position publication never reallocates after the
             * compact logits tensor has been produced.
             */
            MTPSpecDecodeMetadataShape metadata_shape =
                mtp_spec_decode_metadata_binding_.capacity();
            metadata_shape.max_requests =
                std::max(metadata_shape.max_requests, batch_size);
            metadata_shape.max_draft_tokens =
                std::max(
                    metadata_shape.max_draft_tokens,
                    std::max(1, mtp_max_draft_depth_));
            mtp_spec_decode_metadata_binding_.ensureCapacity(metadata_shape);
        }
        auto restore_prefill_logits_mode = [&]()
        {
            if (!compact_prefill_logits)
                return;
            request_batched_prefill_logits_row_count_ = 0;
            if (graph_builder_)
                graph_builder_->setRowIndexedAllPositionLogitRows({});
            setComputeRowIndexedAllPositionLogits(false, 0);
            setComputeAllPositionLogits(false);
        };
        if (compact_prefill_logits)
        {
            if (!setComputeAllPositionLogits(true) ||
                !setComputeRowIndexedAllPositionLogits(true, batch_size) ||
                !graph_builder_)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to configure "
                          "request-batched terminal logits row selection");
                restore_prefill_logits_mode();
                state_.sequence_lengths = old_sequence_lengths;
                return false;
            }
            request_batched_prefill_logits_row_count_ = batch_size;
        }

        // Create flattened, padded token array [batch_size * padded_seq_len]
        std::vector<int> flat_tokens(batch_size * padded_seq_len_, 0); // pad with 0
        for (int b = 0; b < batch_size; ++b)
        {
            const auto &seq = token_batches[b];
            for (size_t s = 0; s < seq.size(); ++s)
            {
                flat_tokens[b * padded_seq_len_ + s] = seq[s];
            }
        }

        const auto result = forwardImpl(
            flat_tokens.data(),
            /*token_ids_device=*/nullptr,
            padded_seq_len_,
            batch_size,
            execution_role,
            ForwardInvocationKind::Automatic,
            /*position_ids_device_override=*/nullptr,
            /*sequence_lengths_device_override=*/nullptr,
            actual_lengths);
        if (compact_prefill_logits)
        {
            /*
             * Keep request_batched_prefill_logits_row_count_ until the sampler
             * consumes the compact logits buffer, but restore graph-builder
             * knobs so the following decode/verifier graph starts from a clean
             * normal-logits configuration.
             */
            if (graph_builder_)
                graph_builder_->setRowIndexedAllPositionLogitRows({});
            setComputeRowIndexedAllPositionLogits(false, 0);
            setComputeAllPositionLogits(false);
            if (!result)
                request_batched_prefill_logits_row_count_ = 0;
        }

        if (!result)
        {
            state_.positions = old_positions;
            state_.sequence_lengths = old_sequence_lengths;
            return false;
        }

        // Restore actual request progress after the padded graph execution.
        // This is important for:
        // 1. Proper logits extraction (only extract non-padded logits)
        // 2. Snapshot comparison (shapes should match actual token count)
        // 3. KV cache position tracking (only actual tokens contribute to cache)
        state_.positions.resize(batch_size);
        state_.sequence_lengths.resize(batch_size);
        for (int i = 0; i < batch_size; ++i)
        {
            const int old_pos =
                i < static_cast<int>(old_positions.size()) ? old_positions[i] : 0;
            const int old_len =
                i < static_cast<int>(old_sequence_lengths.size()) ? old_sequence_lengths[i] : old_pos;
            state_.positions[i] = old_pos + actual_lengths[i];
            state_.sequence_lengths[i] = old_len + actual_lengths[i];
        }

        return true;
    }

    const float *DeviceGraphOrchestrator::getLogits(int seq_idx) const
    {
        if (!state_.logits)
        {
            return nullptr;
        }

        if (seq_idx < 0 || seq_idx >= state_.batch_size)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid sequence index " << seq_idx
                                                                          << " (batch_size=" << state_.batch_size << ")");
            return nullptr;
        }

        // Return pointer to logits for requested sequence
        // Layout: [batch_size, vocab_size] (LM head always computes M=1 per batch entry)
        // For sequence seq_idx, logits start at row seq_idx
        const float *base = state_.logits->fp32_data();
        if (!base)
        {
            return nullptr;
        }

        return base + (seq_idx * state_.vocab_size);
    }

    int DeviceGraphOrchestrator::getPosition(int seq_idx) const
    {
        if (seq_idx < 0 || static_cast<size_t>(seq_idx) >= state_.positions.size())
        {
            return 0;
        }
        return state_.positions[seq_idx];
    }

    // =========================================================================
    // Private Helpers
    // =========================================================================

    void DeviceGraphOrchestrator::updateCachedGraphParams(ComputeGraph &graph, int pos_offset, int seq_len)
    {
        // Update all stages in the graph that have dynamic parameters
        const auto &order = graph.getExecutionOrder();

        for (const auto &node_name : order)
        {
            ComputeNode *node = graph.getNode(node_name);
            if (!node || !node->stage)
                continue;

            // Update dynamic params (pos_offset, seq_len)
            // Only stages that override updateDynamicParams will actually do anything
            node->stage->updateDynamicParams(pos_offset, seq_len);
        }

        LOG_TRACE("[DeviceGraphOrchestrator] Updated cached graph params: pos_offset="
                  << pos_offset << " seq_len=" << seq_len);
    }

    bool DeviceGraphOrchestrator::canUseCachedGraph(int layer_idx, int seq_len) const
    {
        if (!cache_config_.enabled)
            return false;
        if (layer_idx < 0 || static_cast<size_t>(layer_idx) >= layer_graph_cache_.size())
            return false;

        const auto &cache = layer_graph_cache_[layer_idx];
        return cache.valid && cache.cached_seq_len == seq_len;
    }

    // =========================================================================
    // MoE Expert Rebalance Controller
    // =========================================================================

    void DeviceGraphOrchestrator::setMoERebalanceController(
        std::unique_ptr<MoERebalanceController> controller)
    {
        moe_rebalance_extra_controllers_.clear();
        moe_rebalance_controller_ = std::move(controller);
    }

    void DeviceGraphOrchestrator::addMoERebalanceController(
        std::unique_ptr<MoERebalanceController> controller)
    {
        if (!controller)
            return;
        if (!moe_rebalance_controller_)
        {
            moe_rebalance_controller_ = std::move(controller);
            return;
        }
        moe_rebalance_extra_controllers_.push_back(std::move(controller));
    }

    std::vector<MoERebalanceController *> DeviceGraphOrchestrator::moeRebalanceControllers() const
    {
        std::vector<MoERebalanceController *> controllers;
        if (moe_rebalance_controller_)
            controllers.push_back(moe_rebalance_controller_.get());
        for (const auto &controller : moe_rebalance_extra_controllers_)
        {
            if (controller)
                controllers.push_back(controller.get());
        }
        return controllers;
    }

    MoERebalanceController *DeviceGraphOrchestrator::moeRebalanceControllerForDomain(
        const std::string &domain_id) const
    {
        if (moe_rebalance_controller_ && moe_rebalance_controller_->domainId() == domain_id)
            return moe_rebalance_controller_.get();
        for (const auto &controller : moe_rebalance_extra_controllers_)
        {
            if (controller && controller->domainId() == domain_id)
                return controller.get();
        }
        return nullptr;
    }

    int DeviceGraphOrchestrator::moeRebalanceParticipantId() const
    {
        if (const auto *global_ctx = globalTPContextForMTPCoordination())
            return global_ctx->myIndex();
        if (graph_builder_ && graph_builder_->config().tp_ctx &&
            graph_builder_->config().tp_ctx->isLocal() &&
            graph_builder_->config().tp_device_idx >= 0)
        {
            return graph_builder_->config().tp_device_idx;
        }
        if (mpi_ctx_ && mpi_ctx_->world_size() > 1)
            return mpi_ctx_->rank();
        return 0;
    }

    void DeviceGraphOrchestrator::initializeExpertPayloadProvider()
    {
        // Count MoE stages to decide if a provider is needed
        int moe_stage_count = 0;
        if (forward_engine_)
        {
            forward_engine_->forEachCachedStage(
                ComputeStageType::MOE_EXPERT_FFN,
                [&](IComputeStage *)
                { ++moe_stage_count; });
        }

        if (moe_stage_count == 0)
        {
            LOG_DEBUG("[DGO] No MoE stages found — skipping payload provider initialization");
            return;
        }

        // Create provider
        expert_payload_provider_ = std::make_unique<ExpertWeightPayloadProvider>();

        // Wire to all cached MoE stages
        int wired = 0;
        forward_engine_->forEachCachedStage(
            ComputeStageType::MOE_EXPERT_FFN,
            [&](IComputeStage *s)
            {
                auto *moe = dynamic_cast<MoEExpertComputeStage *>(s);
                if (moe)
                {
                    moe->setPayloadProvider(expert_payload_provider_.get());
                    ++wired;
                }
            });

        // Wire to WeightManager for host retention decisions
        if (weight_manager_)
        {
            weight_manager_->setExpertPayloadProvider(expert_payload_provider_.get());
        }

        LOG_DEBUG("[DGO] Expert payload provider initialized and wired to "
                  << wired << " MoE stages");
    }

    void DeviceGraphOrchestrator::initializePreparedWeightStore(DeviceId device)
    {
        if (!weight_manager_)
        {
            LOG_DEBUG("[DGO] No weight manager — skipping prepared weight store");
            return;
        }

        ModelContextId requested_model_id{};
        requested_model_id.value = reinterpret_cast<uint64_t>(this); // Fallback only when no model-owned store exists
        if (frozen_weight_set_ && frozen_weight_set_->strategy().model_id.value != 0)
            requested_model_id = frozen_weight_set_->strategy().model_id;

        if (!prepared_weight_store_)
        {
            if (auto concrete_weight_manager = std::dynamic_pointer_cast<WeightManager>(weight_manager_))
            {
                prepared_weight_store_ = concrete_weight_manager->preparedWeightStoreIfInitialized();
            }
            if (!prepared_weight_store_)
            {
                prepared_weight_store_ = std::make_shared<PreparedWeightStore>(requested_model_id);
            }
            if (auto concrete_weight_manager = std::dynamic_pointer_cast<WeightManager>(weight_manager_))
            {
                concrete_weight_manager->setPreparedWeightStore(prepared_weight_store_);
            }
        }
        if (!frozen_weight_set_ && prepared_weight_store_->modelId().value != 0)
            requested_model_id = prepared_weight_store_->modelId();

        if (!prepared_weight_store_->bindModelIdIfUnset(requested_model_id))
        {
            throw std::runtime_error(
                "[DGO] Prepared weight store model id mismatch: store=" +
                std::to_string(prepared_weight_store_->modelId().value) +
                " requested=" + std::to_string(requested_model_id.value));
        }

        ModelContextId model_id = prepared_weight_store_->modelId();
        if (model_id.value == 0)
            model_id = requested_model_id;

        int registered = 0;
        uint64_t next_binding_id = 1;
        if (frozen_weight_set_)
        {
            for (const auto &binding : frozen_weight_set_->bindings())
                next_binding_id = std::max(next_binding_id, binding.binding_id + 1);
        }
        if (decode_replicated_dense_weight_set_)
        {
            for (const auto &binding : decode_replicated_dense_weight_set_->bindings())
                next_binding_id = std::max(next_binding_id, binding.binding_id + 1);
        }

        auto register_if_prepared = [&] (
            const WeightBinding &source_binding,
            PhysicalMemoryOwner memory_owner)
        {
            const std::string &name = source_binding.identity.canonical_name;
            TensorBase *tensor = source_binding.tensor;
            if (!tensor)
                return;

            WeightBinding binding = source_binding;

            if (!frozen_weight_set_)
                binding.identity.model_id = model_id;
            else if (binding.identity.model_id.value == 0)
                binding.identity.model_id = model_id;
            binding.residency.home_device = device;
            binding.residency.resident_device = device;

            /*
             * GPU graph construction is a consumer of prepared embeddings,
             * never their allocator. WeightManager must create the unique
             * device representation before it closes the device-preparation
             * gate. Enforcing that boundary here makes late, planner-invisible
             * allocations structurally impossible.
             */
            if (binding.identity.role == WeightRole::Embedding)
            {
                const auto *unpackable =
                    IINT8Unpackable::fromTensor(tensor);
                if (device.is_gpu() && unpackable &&
                    !prepared_weight_store_
                         ->preparedRefForBinding(
                             binding.binding_id,
                             device,
                             PreparedWeightKind::PreparedEmbedding)
                         .has_value())
                {
                    throw std::runtime_error(
                        "[DGO] Quantized GPU embedding reached graph "
                        "construction without a prepared device handle: " +
                        name);
                }
                return;
            }

            try
            {
                if (binding.identity.role == WeightRole::Norm ||
                    binding.identity.role == WeightRole::Bias ||
                    tensor->shape().size() != 2)
                {
                    return;
                }

                const PreparedWeightKind expected_kind =
                    binding.prepared.has_value()
                        ? binding.prepared->kind
                        : (device.is_cuda()
                               ? PreparedWeightKind::CudaInt8PackedGemm
                               : (device.is_rocm()
                                      ? PreparedWeightKind::RocmInt8PackedGemm
                                      : PreparedWeightKind::CpuPackedGemm));
                if (prepared_weight_store_->preparedRefForBinding(
                        binding.binding_id,
                        device,
                        expected_kind)
                        .has_value())
                {
                    return;
                }

                if (device.is_cpu())
                {
                    prepared_weight_store_->prepareGemm(
                        binding,
                        memory_owner);
                    ++registered;
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN("[DGO] Failed to register prepared weight '"
                         << name << "' in store: " << e.what());
            }
        };

        if (frozen_weight_set_)
        {
            // Iterate graph-frozen bindings rather than WeightManager names so
            // aliases such as tied output.weight -> token_embd.weight keep their
            // own binding ids and roles.
            for (const auto &binding : frozen_weight_set_->bindings())
                register_if_prepared(
                    binding,
                    frozen_weight_set_->physicalMemoryOwner());
        }
        if (decode_replicated_dense_weight_set_)
        {
            for (const auto &binding : decode_replicated_dense_weight_set_->bindings())
                register_if_prepared(
                    binding,
                    decode_replicated_dense_weight_set_
                        ->physicalMemoryOwner());
        }
        if (!frozen_weight_set_ && !decode_replicated_dense_weight_set_)
        {
            // Fallback for non-frozen legacy graph setup.
            weight_manager_->forEachPreparedWeight([&](const std::string &name, TensorBase *tensor)
                                                   {
                WeightBinding binding;
                binding.binding_id = next_binding_id++;
                binding.identity.canonical_name = name;
                binding.identity.role = inferWeightRole(name);
                binding.identity.layer = inferWeightLayer(name);
                binding.tensor = tensor;
                register_if_prepared(
                    binding,
                    PhysicalMemoryOwner::PrimaryModelWeights); });
        }

        LOG_DEBUG("[DGO] Prepared weight store initialized with "
                  << prepared_weight_store_->size() << " entries for device " << device.toString()
                  << " (new=" << registered << ")");
        if (auto concrete_weight_manager = std::dynamic_pointer_cast<WeightManager>(weight_manager_))
            concrete_weight_manager->logHostMemorySummary("after base preparation");

        // Phase 10: Wire prepared weight store to graph builder so stages
        // can resolve kernels through the store instead of KernelFactory fallbacks.
        if (graph_builder_)
        {
            graph_builder_->setPreparedWeightStore(prepared_weight_store_.get());
        }

        if (!prepareMTPMoEExpertSlabs(device))
        {
            throw std::runtime_error(
                "[DGO] Failed to prepare MTP MoE expert slabs before host weight release");
        }
    }

    bool DeviceGraphOrchestrator::prepareMTPMoEExpertSlabs(DeviceId device)
    {
        if (!device.is_cpu())
            return true;
        if (!prepared_weight_store_ || !frozen_weight_set_ || !graph_builder_)
            return true;

        const auto bindings = makeModelWeightBindings(
            *frozen_weight_set_, state_.device_id);
        if (bindings.mtp.empty() || bindings.mtp.depths.empty())
            return true;

        const GraphConfig &cfg = graph_builder_->config();
        int prepared_depths = 0;
        int reused_depths = 0;

        for (const auto &depth : bindings.mtp.depths)
        {
            const bool has_any_moe_weight =
                depth.fa_block.moe_gate ||
                depth.fa_block.moe_gate_exps ||
                depth.fa_block.moe_up_exps ||
                depth.fa_block.moe_down_exps;
            if (!has_any_moe_weight)
                continue;

            if (cfg.moe.usesExpertOverlayDurableResidencyAuthority())
            {
                const auto &residency = cfg.moe.expert_overlay_participant_residency;
                const auto manager = std::dynamic_pointer_cast<WeightManager>(weight_manager_);
                if (!cfg.moe.expert_overlay_residency_authority || !residency ||
                    !manager || residency->localParticipantIds().empty())
                {
                    LOG_ERROR("[DGO] MTP expert reuse has no complete model-owned overlay residency authority");
                    return false;
                }
                const int source_layer = depth.source_layer_index >= 0
                    ? depth.source_layer_index : cfg.n_layers + depth.depth_index;
                std::string error;
                if ((residency->originalModel().value && !residency->bindOriginalPreparedContext(
                        std::dynamic_pointer_cast<ModelContext>(injected_model_ctx_),residency->projectionPreparation(),&error)) ||
                    !residency->registerInitialLayerFromPreparedRegistry(
                        manager->expertGemmRegistry(), source_layer, &error))
                {
                    LOG_ERROR("[DGO] MTP expert residency reuse rejected source layer "
                        << source_layer << ": " << error);
                    return false;
                }
                ++reused_depths;
                continue;
            }

            TensorBase *gate_exps = legacyTensor(depth.fa_block.moe_gate_exps);
            TensorBase *up_exps = legacyTensor(depth.fa_block.moe_up_exps);
            TensorBase *down_exps = legacyTensor(depth.fa_block.moe_down_exps);
            /*
             * CPU model population commits every routed expert (including the
             * nextn predictor layer) as exact per-expert prepared projections
             * in the model's ExpertGemmRegistry, consumed through the frozen
             * static per-expert bindings; no 3D parent remains bound. The
             * prepared-registry path authenticates that bank instead of
             * reading a raw parent. A raw parent set is only valid complete.
             */
            const auto concrete_manager = std::dynamic_pointer_cast<WeightManager>(weight_manager_);
            const bool any_raw_parent = gate_exps || up_exps || down_exps;
            if (any_raw_parent && (!gate_exps || !up_exps || !down_exps))
            {
                LOG_ERROR("[DGO] MTP depth " << depth.depth_index
                                             << " has partial MoE expert tensor bindings; refusing lazy raw fallback");
                return false;
            }
            if (!any_raw_parent && (!concrete_manager || cfg.moe.intermediate_size <= 0))
            {
                LOG_ERROR("[DGO] MTP depth " << depth.depth_index
                                             << " has no raw MoE expert parents and no prepared CPU expert registry");
                return false;
            }
            if (any_raw_parent && (gate_exps->shape().size() != 3 || gate_exps->shape()[1] == 0))
            {
                LOG_ERROR("[DGO] MTP depth " << depth.depth_index
                                             << " MoE gate expert tensor must be 3D [cols, rows, experts]");
                return false;
            }

            MoEExpertComputeStage::Params params;
            params.device_id = device;
            params.num_experts = cfg.moe.num_experts;
            params.top_k = cfg.moe.top_k;
            params.d_model = cfg.d_model;
            params.expert_intermediate = cfg.moe.intermediate_size > 0
                                             ? cfg.moe.intermediate_size
                                             : static_cast<int>(gate_exps->shape()[1]);
            if (concrete_manager)
                params.expert_registry = &concrete_manager->expertGemmRegistry();
            params.layer_idx = depth.source_layer_index >= 0
                                   ? depth.source_layer_index
                                   : cfg.n_layers + depth.depth_index;
            params.gate_exps = gate_exps;
            params.up_exps = up_exps;
            params.down_exps = down_exps;
            params.prepared_store = prepared_weight_store_.get();
            params.persistent_weight_owner = frozen_weight_set_->physicalMemoryOwner();

            if (cfg.moe.routed_compute_policy == RoutedExpertComputePolicy::Apportioned)
            {
                params.local_expert_start = cfg.moe.local_expert_start;
                params.local_expert_count = cfg.moe.local_expert_count;
                if (cfg.moe.owner_participant_count > 1)
                {
                    params.expert_mask =
                        routed_expert_ownership::expertMaskForParticipant(
                            cfg.moe.num_experts,
                            cfg.moe.owner_participant_count,
                            cfg.moe.owner_participant_index,
                            params.layer_idx,
                            cfg.moe.owner_order);
                }
            }

            if (!MoEExpertComputeStage::extractExpertViews(params) ||
                !MoEExpertComputeStage::prepareExpertGemmEngines(params))
            {
                LOG_ERROR("[DGO] Failed to prepare MTP MoE expert slabs for depth "
                          << depth.depth_index << " source_layer=" << params.layer_idx
                          << " on " << device.to_string());
                return false;
            }

            ++prepared_depths;
        }

        if (reused_depths > 0)
            PerfStatsCollector::addCounter("weight_loading", "mtp_moe_expert_residency_reuses",
                static_cast<double>(reused_depths), "load", device.to_string());

        if (prepared_depths > 0)
        {
            LOG_DEBUG("[DGO] Prepared " << prepared_depths
                                        << " MTP MoE expert slab set(s) for "
                                        << device.to_string()
                                        << " before host raw weight release");
            PerfStatsCollector::addCounter(
                "weight_loading",
                "mtp_moe_expert_slab_preparations",
                static_cast<double>(prepared_depths),
                "load",
                device.to_string());
        }

        return true;
    }

    void DeviceGraphOrchestrator::applyExpertMasks(
        const std::vector<std::vector<bool>> &masks,
        const ReceivedWeightsMap &received_weights,
        const ReceivedPreparedExpertsMap &received_prepared_experts)
    {
        applyExpertMasksForDomain(
            {}, masks, received_weights, received_prepared_experts);
    }

    void DeviceGraphOrchestrator::applyExpertMasksForDomain(
        const std::string &domain_id,
        const std::vector<std::vector<bool>> &masks,
        const ReceivedWeightsMap &received_weights,
        const ReceivedPreparedExpertsMap &received_prepared_experts)
    {
        using Clock = std::chrono::steady_clock;
        const auto t_start = Clock::now();
        validateMoERebalanceDomain(*this, domain_id, "applyExpertMasks");

        // Lazily initialize payload provider on first mask application
        if (!expert_payload_provider_)
        {
            initializeExpertPayloadProvider();
        }

        // ── Step 1: Collect all MoE stages ──────────────────────────────
        const bool graph_stable_gpu_rebalance =
            usesGraphStableMoEOverlayResidency();
        if (graph_stable_gpu_rebalance && forward_engine_)
        {
            const size_t invalidated =
                forward_engine_->invalidateMoEPlacementSensitiveGraphsForStablePlacement(
                    [this] { retireMTPPublicationVerifierBorrows(); });
            if (invalidated > 0)
            {
                PerfStatsCollector::addCounter(
                    "moe_rebalance",
                    "graph_stable_placement_cache_invalidations",
                    static_cast<double>(invalidated),
                    "rebalance",
                    primaryDeviceId().is_valid() ? primaryDeviceId().toString() : std::string{},
                    {{"domain_id", domain_id.empty() ? "default" : domain_id}});
                LOG_DEBUG("[DGO] Invalidated " << invalidated
                                               << " MoE placement-sensitive cached graph(s); "
                                               << "fixed prefill and single-token decode remain graph-stable");
            }
        }

        struct StageInfo
        {
            MoEExpertComputeStage *stage;
            int layer;
        };
        std::vector<StageInfo> moe_stages;
        auto accepts_moe_rebalance_stage = [&](MoEExpertComputeStage *stage)
        {
            if (!stage)
                return false;
            return !graph_stable_gpu_rebalance ||
                   stage->usesGraphStableMoEPlacement();
        };

        if (forward_engine_)
            if (prepared_weight_store_)
            {
                forward_engine_->forEachCachedStage(
                    ComputeStageType::MOE_EXPERT_FFN,
                    [&](IComputeStage *s)
                    {
                        auto *moe = dynamic_cast<MoEExpertComputeStage *>(s);
                        if (!accepts_moe_rebalance_stage(moe))
                            return;
                        int layer = moe->layerIndex();
                        if (layer >= 0 && static_cast<size_t>(layer) < masks.size())
                            moe_stages.push_back({moe, layer});
                    });
            }

        if (graph_stable_gpu_rebalance && !masks.empty() && moe_stages.empty())
        {
            throw std::runtime_error(
                "GPU graph-stable MoE rebalance requires cached graph-stable MoE stages");
        }

        const size_t discovered_stage_count = moe_stages.size();
        if (!graph_stable_gpu_rebalance)
        {
            /*
             * CPU ownership waves are sparse.  A stage whose exact mask is
             * already current and which received no payload has no lifecycle
             * work to perform.  Excluding it here avoids releasing, scanning,
             * and rebinding all model layers for a handful of exact migrations.
             * GPU publication deliberately remains unchanged because its
             * persistent runtime tables have a separate graph-stable contract.
             */
            std::erase_if(
                moe_stages,
                [&](const StageInfo &info)
                {
                    if (!info.stage || !info.stage->device().is_cpu())
                        return false;
                    const auto received_layer = received_weights.find(info.layer);
                    const auto prepared_layer =
                        received_prepared_experts.find(info.layer);
                    const bool has_arrivals =
                        received_layer != received_weights.end() &&
                        !received_layer->second.empty();
                    const bool has_prepared_arrivals =
                        prepared_layer != received_prepared_experts.end() &&
                        !prepared_layer->second.empty();
                    return !has_arrivals && !has_prepared_arrivals &&
                           info.stage->expertMaskMatches(
                               masks[static_cast<size_t>(info.layer)]);
                });
        }

        const bool has_gpu_publish_stage =
            std::any_of(moe_stages.begin(),
                        moe_stages.end(),
                        [](const StageInfo &info)
                        {
                            return info.stage && info.stage->device().is_gpu();
                        });
        void *publication_stream = nullptr;
        if (has_gpu_publish_stage)
        {
            /*
             * Mask publication mutates graph-stable device tables outside a
             * normal stage execution pass. Cached stages may have had their
             * execution stream cleared at a request boundary, so bind an
             * orchestrator-owned explicit stream before any register/refresh path
             * reaches backend kernels.
             */
            publication_stream = explicitGPUStreamForOperation("moe_expert_mask_publish");
            if (!publication_stream)
            {
                throw std::runtime_error(
                    "MoE expert mask publication requires an explicit non-null GPU stream");
            }
        }

        auto bind_publication_stream = [&](MoEExpertComputeStage *stage)
        {
            if (stage && stage->device().is_gpu())
                stage->setGPUStream(publication_stream);
        };

        for (auto &[stage, layer] : moe_stages)
        {
            (void)layer;
            bind_publication_stream(stage);
        }

        const auto stage_discovery_end = Clock::now();

        auto release_departed = [&]()
        {
            for (auto &[stage, layer] : moe_stages)
            {
                (void)stage->releaseDepartedExperts(masks[layer]);
            }
        };

        /*
         * Graph-stable GPU rebalance may have several cached MoE stages for the
         * same layer: fixed-topology prefill, single-token decode, and any other
         * preserved graph-stable owners. GPU-direct activation publishes arrivals
         * to the shared PreparedWeightStore before mask application. If one stage
         * releases departures from that store before sibling stages adopt their
         * newly-arrived engines, those siblings can incorrectly fall through to
         * the forbidden host/raw fallback. Adopt first in graph-stable mode, then
         * release once every selected stage has local GEMM handles.
         */
        const auto release_start = Clock::now();
        if (!graph_stable_gpu_rebalance)
            release_departed();
        const auto release_end = Clock::now();

        // ── Step 2: register + prepare (parallel across stages) ─────────
        std::atomic<int> applied{0};
        const int n = static_cast<int>(moe_stages.size());

::llaminar::v2::noteOpenMPTeamFork();
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; ++i)
        {
            auto &[stage, layer] = moe_stages[i];
            const std::unordered_map<int, ExpertWeightBlobs> *layer_received = nullptr;
            const std::unordered_map<int, PreparedExpertEngines> *
                layer_prepared = nullptr;
            auto layer_it = received_weights.find(layer);
            if (layer_it != received_weights.end())
                layer_received = &layer_it->second;
            auto prepared_it = received_prepared_experts.find(layer);
            if (prepared_it != received_prepared_experts.end())
                layer_prepared = &prepared_it->second;

            if (stage->registerAndPrepareNewExperts(
                    masks[layer], layer_received, layer_prepared))
                applied.fetch_add(1, std::memory_order_relaxed);
        }

        const auto prepare_end = Clock::now();

        if (applied.load(std::memory_order_relaxed) != n)
        {
            LOG_ERROR("[DGO] Failed to prepare " << (n - applied.load(std::memory_order_relaxed))
                                                 << " MoEExpertComputeStages during expert mask application");
            throw std::runtime_error("MoE expert mask application failed: missing prepared expert weights");
        }

        if (graph_stable_gpu_rebalance)
            release_departed();

        // ── Step 3: apply masks (fast, no heavy ops) ───────────────────
        for (auto &[stage, layer] : moe_stages)
            stage->applyExpertMask(masks[layer]);
        const auto mask_publish_end = Clock::now();

        const bool mask_placement_changed = current_expert_masks_ != masks;
        if (mask_placement_changed)
        {
            current_expert_masks_ = masks;
            ++current_expert_mask_epoch_;
            if (graph_stable_gpu_rebalance)
            {
                /*
                 * Graph-stable GPU rebalance deliberately pins
                 * moePlacementEpoch() so decode/prefill graphs keep replaying
                 * through persistent runtime tables. Prefix-cache entries are
                 * different: they encode KV/logits produced under the current
                 * expert residency, so host-published graph-stable mask changes
                 * must still invalidate prefix fingerprints.
                 */
                ++moe_runtime_movement_epoch_;
                PerfStatsCollector::addCounter(
                    "moe_rebalance",
                    "expert_runtime_movement_epoch",
                    static_cast<double>(moe_runtime_movement_epoch_),
                    "rebalance",
                    primaryDeviceId().is_valid() ? primaryDeviceId().toString() : std::string{},
                    {{"domain_id", domain_id.empty() ? "default" : domain_id},
                     {"source", "mask_publish"}});
            }
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "expert_mask_placement_epoch",
                static_cast<double>(current_expert_mask_epoch_),
                "rebalance",
                primaryDeviceId().is_valid() ? primaryDeviceId().toString() : std::string{},
                {{"domain_id", domain_id.empty() ? "default" : domain_id}});
        }

        LOG_DEBUG("[DGO] Applied expert masks to " << applied.load()
                                                   << " MoEExpertComputeStages across " << masks.size()
                                                   << " layers mask_epoch=" << current_expert_mask_epoch_
                                                   << " changed=" << (mask_placement_changed ? "true" : "false"));

        size_t received_entries = 0;
        size_t received_bytes = 0;
        for (const auto &[layer, experts] : received_weights)
        {
            (void)layer;
            received_entries += experts.size();
            for (const auto &[expert, blobs] : experts)
            {
                (void)expert;
                received_bytes += blobs.totalBytes();
            }
        }
        for (const auto &[layer, experts] : received_prepared_experts)
        {
            (void)layer;
            received_entries += experts.size();
            for (const auto &[expert, prepared] : experts)
            {
                (void)expert;
                received_bytes += prepared.packed_bytes;
            }
        }

        const auto t_end = Clock::now();
        const std::string stats_device =
            primaryDeviceId().is_valid() ? primaryDeviceId().toString() : std::string{};
        const PerfStatsCollector::Tags phase_tags{
            {"domain_id", domain_id.empty() ? "default" : domain_id}};
        const auto record_phase = [&](const char *name, auto begin, auto end)
        {
            PerfStatsCollector::recordTimingNs(
                "moe_rebalance",
                name,
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()),
                "rebalance",
                stats_device,
                phase_tags);
        };
        record_phase("mask_apply_stage_discovery", t_start, stage_discovery_end);
        record_phase("mask_apply_departure_release", release_start, release_end);
        record_phase("mask_apply_arrival_prepare", release_end, prepare_end);
        record_phase("mask_apply_mask_publication", prepare_end, mask_publish_end);
        record_phase("mask_apply_total", t_start, t_end);
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "mask_apply_discovered_stages",
            static_cast<double>(discovered_stage_count),
            "rebalance",
            stats_device,
            phase_tags);
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "mask_apply_selected_stages",
            static_cast<double>(moe_stages.size()),
            "rebalance",
            stats_device,
            phase_tags);
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "mask_apply_unchanged_cpu_stage_skips",
            static_cast<double>(discovered_stage_count - moe_stages.size()),
            "rebalance",
            stats_device,
            phase_tags);
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "mask_apply_received_entries",
            static_cast<double>(received_entries),
            "rebalance",
            stats_device,
            phase_tags);
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "mask_apply_received_bytes",
            static_cast<double>(received_bytes),
            "rebalance",
            stats_device,
            phase_tags);
        double prep_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        if (moe_rebalance_controller_)
            moe_rebalance_controller_->recordPrepDuration(prep_ms);
        LOG_DEBUG("[DGO] Expert mask application + engine prep took "
                  << std::fixed << std::setprecision(1) << prep_ms << " ms");
    }

    ReceivedWeightsMap DeviceGraphOrchestrator::collectExpertWeightsForMasks(
        const std::vector<std::vector<bool>> &masks) const
    {
        ReceivedWeightsMap result;

        std::unordered_map<int, const MoEExpertComputeStage *> moe_by_layer;
        if (forward_engine_)
        {
            forward_engine_->forEachCachedStage(
                ComputeStageType::MOE_EXPERT_FFN,
                [&](IComputeStage *stage)
                {
                    const auto *moe_stage = dynamic_cast<const MoEExpertComputeStage *>(stage);
                    if (moe_stage && moe_stage->layerIndex() >= 0)
                        moe_by_layer[moe_stage->layerIndex()] = moe_stage;
                });
        }

        for (size_t layer_idx = 0; layer_idx < masks.size(); ++layer_idx)
        {
            auto stage_it = moe_by_layer.find(static_cast<int>(layer_idx));
            if (stage_it == moe_by_layer.end())
                continue;

            const auto &mask = masks[layer_idx];
            for (size_t expert_idx = 0; expert_idx < mask.size(); ++expert_idx)
            {
                if (!mask[expert_idx])
                    continue;
                auto blobs = stage_it->second->serializeExpert(static_cast<int>(expert_idx));
                if (!blobs.empty())
                    result[static_cast<int>(layer_idx)][static_cast<int>(expert_idx)] = std::move(blobs);
            }
        }

        return result;
    }

    DeviceGraphOrchestrator::PreparedDirectExpertTransfer
    DeviceGraphOrchestrator::prepareExpertWeightsDirectForMasksFrom(
        DeviceGraphOrchestrator &source,
        const std::vector<std::vector<bool>> &masks)
    {
        PreparedDirectExpertTransfer prepared;
        prepared.remaining_masks = masks;
        retireCompletedGpuDirectActivationCompletions();
        if (&source == this)
            return prepared;

        auto collect_moe_stages = [](DeviceGraphOrchestrator &orchestrator)
        {
            std::unordered_map<int, std::vector<MoEExpertComputeStage *>> by_layer;

            if (orchestrator.forward_engine_)
            {
                orchestrator.forward_engine_->forEachCachedStage(
                    ComputeStageType::MOE_EXPERT_FFN,
                    [&](IComputeStage *stage)
                    {
                        auto *moe_stage = dynamic_cast<MoEExpertComputeStage *>(stage);
                        if (moe_stage && moe_stage->layerIndex() >= 0)
                            by_layer[moe_stage->layerIndex()].push_back(moe_stage);
                    });
            }

            if (by_layer.empty())
            {
                for (size_t layer = 0; layer < orchestrator.layer_graph_cache_.size(); ++layer)
                {
                    auto &cache = orchestrator.layer_graph_cache_[layer];
                    if (!cache.valid || !cache.ffn_decode)
                        continue;
                    for (const auto &node_name : cache.ffn_decode->getExecutionOrder())
                    {
                        auto *node = cache.ffn_decode->getNode(node_name);
                        if (!node || !node->stage ||
                            node->stage->type() != ComputeStageType::MOE_EXPERT_FFN)
                        {
                            continue;
                        }
                        auto *moe_stage = dynamic_cast<MoEExpertComputeStage *>(node->stage.get());
                        if (moe_stage && moe_stage->layerIndex() >= 0)
                            by_layer[moe_stage->layerIndex()].push_back(moe_stage);
                    }
                }
            }

            return by_layer;
        };

        auto src_by_layer = collect_moe_stages(source);
        auto dst_by_layer = collect_moe_stages(*this);
        if (src_by_layer.empty() || dst_by_layer.empty())
            return prepared;

        const bool graph_stable_gpu_rebalance =
            usesGraphStableMoEOverlayResidency();
        auto select_rebalance_destination_stages =
            [&](const std::vector<MoEExpertComputeStage *> &candidates)
        {
            std::vector<MoEExpertComputeStage *> selected;
            selected.reserve(candidates.size());
            for (auto *stage : candidates)
            {
                if (!stage)
                    continue;
                if (graph_stable_gpu_rebalance &&
                    !stage->usesGraphStableMoEPlacement())
                {
                    continue;
                }
                selected.push_back(stage);
            }
            return selected;
        };

        const size_t wave_experts = static_cast<size_t>(
            std::max(1, debugEnv().moe_rebalance.gpu_direct_transfer_wave_experts));
        const size_t staging_buffers = static_cast<size_t>(
            std::max(1, debugEnv().moe_rebalance.gpu_direct_transfer_buffers));
        size_t requested_arrival_entries = 0;
        for (size_t layer_idx = 0; layer_idx < masks.size(); ++layer_idx)
        {
            auto dst_it = dst_by_layer.find(static_cast<int>(layer_idx));
            if (dst_it == dst_by_layer.end())
                continue;

            std::vector<int> expert_ids;
            const auto &layer_mask = masks[layer_idx];
            expert_ids.reserve(static_cast<size_t>(
                std::count(layer_mask.begin(), layer_mask.end(), true)));
            for (size_t expert_idx = 0; expert_idx < layer_mask.size(); ++expert_idx)
            {
                if (layer_mask[expert_idx])
                    expert_ids.push_back(static_cast<int>(expert_idx));
            }
            if (expert_ids.empty())
                continue;

            const auto dst_stages =
                select_rebalance_destination_stages(dst_it->second);
            for (auto *dst_stage : dst_stages)
            {
                if (!dst_stage)
                    continue;
                requested_arrival_entries +=
                    dst_stage->missingPreparedExpertIds(expert_ids).size();
            }
        }
        const size_t staging_pool_capacity =
            std::max(wave_experts * staging_buffers, requested_arrival_entries);

        auto abort_prepared = [&]() -> PreparedDirectExpertTransfer
        {
            prepared.remaining_masks = masks;
            prepared.staged_arrivals.clear();
            pending_gpu_direct_transfer_slot_arrivals_.clear();
            return prepared;
        };

        int staged_layers = 0;
        int staged_experts = 0;
        int staged_stage_arrivals = 0;
        for (size_t layer_idx = 0; layer_idx < prepared.remaining_masks.size(); ++layer_idx)
        {
            auto src_it = src_by_layer.find(static_cast<int>(layer_idx));
            auto dst_it = dst_by_layer.find(static_cast<int>(layer_idx));
            if (src_it == src_by_layer.end() || dst_it == dst_by_layer.end())
                continue;

            std::vector<int> expert_ids;
            const auto &layer_mask = prepared.remaining_masks[layer_idx];
            for (size_t expert_idx = 0; expert_idx < layer_mask.size(); ++expert_idx)
            {
                if (layer_mask[expert_idx])
                    expert_ids.push_back(static_cast<int>(expert_idx));
            }
            if (expert_ids.empty())
                continue;

            std::vector<int> satisfied_counts(prepared.remaining_masks[layer_idx].size(), 0);
            const auto dst_stages =
                select_rebalance_destination_stages(dst_it->second);
            if (dst_stages.empty())
                continue;

            int required_stage_count = 0;
            for (auto *dst_stage : dst_stages)
            {
                ++required_stage_count;
                std::vector<int> stage_pending = dst_stage->missingPreparedExpertIds(expert_ids);
                const size_t active_arrival_capacity = stage_pending.size();
                for (int expert_id : expert_ids)
                {
                    if (std::find(stage_pending.begin(), stage_pending.end(), expert_id) != stage_pending.end())
                        continue;
                    if (expert_id >= 0 &&
                        expert_id < static_cast<int>(satisfied_counts.size()))
                    {
                        ++satisfied_counts[static_cast<size_t>(expert_id)];
                    }
                }

                if (stage_pending.empty())
                    continue;

                for (auto *src_stage : src_it->second)
                {
                    if (!src_stage || stage_pending.empty())
                        continue;

                    void *source_producer_stream = src_stage->gpuStream();
                    if (!source_producer_stream && source.primaryDeviceId().is_gpu())
                    {
                        source_producer_stream =
                            source.explicitGPUStreamForOperation("moe_gpu_direct_transfer_source");
                    }
                    if (!source_producer_stream)
                        continue;

                    std::vector<int> source_satisfied;
                    while (!stage_pending.empty())
                    {
                        const size_t wave_count = std::min(stage_pending.size(), wave_experts);
                        std::vector<int> wave_expert_ids;
                        wave_expert_ids.reserve(wave_count);
                        for (size_t wave_idx = 0; wave_idx < wave_count; ++wave_idx)
                            wave_expert_ids.push_back(stage_pending[wave_idx]);
                        if (wave_expert_ids.empty())
                            continue;

                        GpuDirectTransferSlotArrivals stage_arrivals;
                        auto satisfied_experts =
                            dst_stage->stageExpertsGPUDirectToTransferSlotsFrom(
                                *src_stage,
                                wave_expert_ids,
                                source_producer_stream,
                                &stage_arrivals,
                                active_arrival_capacity,
                                staging_pool_capacity,
                                &gpu_direct_transfer_staging_pools_);
                        if (satisfied_experts.empty())
                            break;

                        if (!stage_arrivals.empty())
                        {
                            PendingGpuDirectTransferSlotArrival pending;
                            pending.destination_stage = dst_stage;
                            pending.arrivals = std::move(stage_arrivals);
                            prepared.staged_arrivals.push_back(pending);
                            pending_gpu_direct_transfer_slot_arrivals_.push_back(std::move(pending));
                            ++staged_stage_arrivals;
                        }

                        source_satisfied.insert(
                            source_satisfied.end(),
                            satisfied_experts.begin(),
                            satisfied_experts.end());

                        std::sort(satisfied_experts.begin(), satisfied_experts.end());
                        satisfied_experts.erase(
                            std::unique(satisfied_experts.begin(), satisfied_experts.end()),
                            satisfied_experts.end());
                        stage_pending.erase(
                            std::remove_if(stage_pending.begin(),
                                           stage_pending.end(),
                                           [&](int expert_id)
                                           {
                                               return std::find(satisfied_experts.begin(),
                                                                satisfied_experts.end(),
                                                                expert_id) != satisfied_experts.end();
                                           }),
                            stage_pending.end());
                    }

                    if (source_satisfied.empty())
                        continue;
                    std::sort(source_satisfied.begin(), source_satisfied.end());
                    source_satisfied.erase(
                        std::unique(source_satisfied.begin(), source_satisfied.end()),
                        source_satisfied.end());
                    for (int expert_id : source_satisfied)
                    {
                        if (expert_id >= 0 &&
                            expert_id < static_cast<int>(satisfied_counts.size()))
                        {
                            ++satisfied_counts[static_cast<size_t>(expert_id)];
                        }
                    }
                }
            }

            if (required_stage_count == 0)
                continue;

            bool any_layer_staged = false;
            for (int expert_id : expert_ids)
            {
                if (expert_id >= 0 &&
                    expert_id < static_cast<int>(prepared.remaining_masks[layer_idx].size()) &&
                    prepared.remaining_masks[layer_idx][static_cast<size_t>(expert_id)] &&
                    expert_id < static_cast<int>(satisfied_counts.size()) &&
                    satisfied_counts[static_cast<size_t>(expert_id)] >= required_stage_count)
                {
                    prepared.remaining_masks[layer_idx][static_cast<size_t>(expert_id)] = false;
                    ++staged_experts;
                    any_layer_staged = true;
                }
            }
            if (any_layer_staged)
                ++staged_layers;
        }

        if (staged_experts > 0)
        {
            LOG_DEBUG("[DGO] GPU-direct/local staged " << staged_experts
                                                       << " expert arrival mask entries across "
                                                       << staged_layers << " layer(s) from sibling orchestrator"
                                                       << " in " << staged_stage_arrivals
                                                       << " async transfer-slot wave(s); activation is deferred until publish");
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "gpu_direct_transfer_slot_prepare_pending_batches",
                static_cast<double>(staged_stage_arrivals),
                "rebalance",
                primaryDeviceId().to_string(),
                {});
        }

        return prepared;
    }

    bool DeviceGraphOrchestrator::activatePreparedGpuDirectExpertTransfers(
        const std::vector<PendingGpuDirectTransferSlotArrival> &staged_arrivals)
    {
        if (staged_arrivals.empty())
            return true;

        void *activation_stream = explicitGPUStreamForOperation("moe_gpu_direct_transfer_activation");
        if (!activation_stream)
        {
            LOG_ERROR("[DGO] GPU-direct staged expert activation requires an explicit stream on "
                      << primaryDeviceId().to_string());
            return false;
        }

        int activated_batches = 0;
        int activated_experts = 0;
        const bool graph_stable_gpu_rebalance =
            usesGraphStableMoEOverlayResidency();
        for (const auto &pending : staged_arrivals)
        {
            if (pending.empty())
                continue;

            auto expected_experts = pending.arrivals.expertIds();
            GpuDirectTransferCompletion activation_completion;
            auto activated =
                pending.destination_stage->activateGpuDirectTransferSlotArrivals(
                    pending.arrivals,
                    activation_stream,
                    graph_stable_gpu_rebalance ? &activation_completion : nullptr,
                    !graph_stable_gpu_rebalance);
            std::sort(activated.begin(), activated.end());
            activated.erase(std::unique(activated.begin(), activated.end()), activated.end());
            if (activated != expected_experts)
            {
                LOG_ERROR("[DGO] GPU-direct staged expert activation mismatch on "
                          << primaryDeviceId().to_string()
                          << " expected=" << expected_experts.size()
                          << " activated=" << activated.size());
                return false;
            }
            if (graph_stable_gpu_rebalance && activation_completion.valid())
                pending_gpu_direct_activation_completions_.push_back(std::move(activation_completion));
            ++activated_batches;
            activated_experts += static_cast<int>(activated.size());
        }

        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "gpu_direct_transfer_slot_publish_batches",
            static_cast<double>(activated_batches),
            "rebalance",
            primaryDeviceId().to_string(),
            {});
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "gpu_direct_transfer_slot_publish_experts",
            static_cast<double>(activated_experts),
            "rebalance",
            primaryDeviceId().to_string(),
            {});
        return true;
    }

    void DeviceGraphOrchestrator::clearPendingGpuDirectExpertTransfers()
    {
        pending_gpu_direct_transfer_slot_arrivals_.clear();
        pending_gpu_direct_activation_completions_.clear();
    }

    void DeviceGraphOrchestrator::retireCompletedGpuDirectActivationCompletions()
    {
        if (pending_gpu_direct_activation_completions_.empty())
            return;

        std::vector<GpuDirectTransferCompletion> still_pending;
        still_pending.reserve(pending_gpu_direct_activation_completions_.size());
        int retired = 0;
        for (auto &completion : pending_gpu_direct_activation_completions_)
        {
            if (!completion.valid())
                continue;

            bool ready = false;
            bool queried = false;
            try
            {
                auto &gpu_ctx = GPUDeviceContextPool::instance().getContext(completion.device_id);
                queried = gpu_ctx.queryEventChecked(completion.ready_event.get(), ready);
            }
            catch (const std::exception &e)
            {
                LOG_DEBUG("[DGO] Could not query graph-stable GPU-direct activation completion on "
                          << completion.device_id.to_string() << ": " << e.what());
            }

            if (queried && ready)
            {
                ++retired;
                continue;
            }
            still_pending.push_back(std::move(completion));
        }

        pending_gpu_direct_activation_completions_.swap(still_pending);
        if (retired > 0)
        {
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "gpu_direct_activation_completions_retired",
                static_cast<double>(retired),
                "rebalance",
                primaryDeviceId().to_string(),
                {});
        }
    }

    bool DeviceGraphOrchestrator::activatePendingGpuDirectExpertTransfers()
    {
        retireCompletedGpuDirectActivationCompletions();
        std::vector<PendingGpuDirectTransferSlotArrival> pending;
        pending.swap(pending_gpu_direct_transfer_slot_arrivals_);
        const bool ok = activatePreparedGpuDirectExpertTransfers(pending);
        if (!ok)
            pending_gpu_direct_transfer_slot_arrivals_.clear();
        return ok;
    }

    std::vector<std::vector<bool>> DeviceGraphOrchestrator::transferExpertWeightsDirectForMasksFrom(
        DeviceGraphOrchestrator &source,
        const std::vector<std::vector<bool>> &masks)
    {
        std::vector<std::vector<bool>> remaining = masks;
        if (&source == this)
            return remaining;

        auto collect_moe_stages = [](DeviceGraphOrchestrator &orchestrator)
        {
            std::unordered_map<int, std::vector<MoEExpertComputeStage *>> by_layer;

            if (orchestrator.forward_engine_)
            {
                orchestrator.forward_engine_->forEachCachedStage(
                    ComputeStageType::MOE_EXPERT_FFN,
                    [&](IComputeStage *stage)
                    {
                        auto *moe_stage = dynamic_cast<MoEExpertComputeStage *>(stage);
                        if (moe_stage && moe_stage->layerIndex() >= 0)
                            by_layer[moe_stage->layerIndex()].push_back(moe_stage);
                    });
            }

            if (by_layer.empty())
            {
                for (size_t layer = 0; layer < orchestrator.layer_graph_cache_.size(); ++layer)
                {
                    auto &cache = orchestrator.layer_graph_cache_[layer];
                    if (!cache.valid || !cache.ffn_decode)
                        continue;
                    for (const auto &node_name : cache.ffn_decode->getExecutionOrder())
                    {
                        auto *node = cache.ffn_decode->getNode(node_name);
                        if (!node || !node->stage ||
                            node->stage->type() != ComputeStageType::MOE_EXPERT_FFN)
                        {
                            continue;
                        }
                        auto *moe_stage = dynamic_cast<MoEExpertComputeStage *>(node->stage.get());
                        if (moe_stage && moe_stage->layerIndex() >= 0)
                            by_layer[moe_stage->layerIndex()].push_back(moe_stage);
                    }
                }
            }

            return by_layer;
        };

        auto src_by_layer = collect_moe_stages(source);
        auto dst_by_layer = collect_moe_stages(*this);
        if (src_by_layer.empty() || dst_by_layer.empty())
            return remaining;

        int copied_layers = 0;
        int copied_experts = 0;
        for (size_t layer_idx = 0; layer_idx < remaining.size(); ++layer_idx)
        {
            auto src_it = src_by_layer.find(static_cast<int>(layer_idx));
            auto dst_it = dst_by_layer.find(static_cast<int>(layer_idx));
            if (src_it == src_by_layer.end() || dst_it == dst_by_layer.end())
                continue;

            std::vector<int> expert_ids;
            const auto &layer_mask = remaining[layer_idx];
            for (size_t expert_idx = 0; expert_idx < layer_mask.size(); ++expert_idx)
            {
                if (layer_mask[expert_idx])
                    expert_ids.push_back(static_cast<int>(expert_idx));
            }
            if (expert_ids.empty())
                continue;

            std::vector<int> satisfied_counts(remaining[layer_idx].size(), 0);
            int required_stage_count = 0;
            for (auto *dst_stage : dst_it->second)
            {
                if (!dst_stage)
                    continue;
                ++required_stage_count;

                std::vector<int> stage_pending = expert_ids;
                for (auto *src_stage : src_it->second)
                {
                    if (!src_stage || stage_pending.empty())
                        continue;

                    void *source_producer_stream = src_stage->gpuStream();
                    if (!source_producer_stream && source.primaryDeviceId().is_gpu())
                    {
                        source_producer_stream =
                            source.explicitGPUStreamForOperation("moe_gpu_direct_transfer_source");
                    }
                    if (!source_producer_stream)
                        continue;

                    auto satisfied_experts =
                        dst_stage->transferExpertsGPUDirectFrom(
                            *src_stage,
                            stage_pending,
                            source_producer_stream);
                    if (satisfied_experts.empty())
                        continue;

                    for (int expert_id : satisfied_experts)
                    {
                        if (expert_id >= 0 &&
                            expert_id < static_cast<int>(satisfied_counts.size()))
                        {
                            ++satisfied_counts[static_cast<size_t>(expert_id)];
                        }
                    }

                    stage_pending.erase(
                        std::remove_if(stage_pending.begin(),
                                       stage_pending.end(),
                                       [&](int expert_id)
                                       {
                                           return std::find(satisfied_experts.begin(),
                                                            satisfied_experts.end(),
                                                            expert_id) != satisfied_experts.end();
                                       }),
                        stage_pending.end());
                }
            }

            if (required_stage_count == 0)
                continue;

            for (int expert_id : expert_ids)
            {
                if (expert_id >= 0 &&
                    expert_id < static_cast<int>(remaining[layer_idx].size()) &&
                    remaining[layer_idx][static_cast<size_t>(expert_id)] &&
                    expert_id < static_cast<int>(satisfied_counts.size()) &&
                    satisfied_counts[static_cast<size_t>(expert_id)] >= required_stage_count)
                {
                    remaining[layer_idx][static_cast<size_t>(expert_id)] = false;
                    ++copied_experts;
                }
            }
            ++copied_layers;
        }

        if (copied_experts > 0)
        {
            LOG_DEBUG("[DGO] GPU-direct/local satisfied " << copied_experts
                                                          << " expert arrival mask entries across "
                                                          << copied_layers << " layer(s) from sibling orchestrator");
        }

        return remaining;
    }

    void DeviceGraphOrchestrator::setExpertReplicaSetForParticipant(
        const ExpertReplicaSet &replicas, int participant_id)
    {
        ExpertReplicaSet normalized_replicas = replicas;
        normalized_replicas.rebuildAggregateReplicaFlags();
        validateMoERebalanceDomain(*this, normalized_replicas.domain_id, "setExpertReplicaSetForParticipant");

        const bool replica_placement_changed =
            current_expert_replica_participant_id_ != participant_id ||
            !normalized_replicas.sameReplicaPlacement(current_expert_replica_set_);
        current_expert_replica_set_ = normalized_replicas;
        current_expert_replica_participant_id_ = participant_id;
        if (replica_placement_changed)
            ++current_expert_replica_epoch_;

        int count = 0;
        void *publication_stream = nullptr;
        auto ensure_replica_publication_stream = [&]() -> void *
        {
            if (publication_stream)
                return publication_stream;
            publication_stream = explicitGPUStreamForOperation("moe_replica_set_publish");
            if (!publication_stream)
            {
                throw std::runtime_error(
                    "MoE replica set publication requires an explicit non-null GPU stream");
            }
            return publication_stream;
        };

        if (forward_engine_)
        {
            forward_engine_->forEachCachedStage(
                ComputeStageType::MOE_EXPERT_FFN,
                [&](IComputeStage *s)
                {
                    auto *moe = dynamic_cast<MoEExpertComputeStage *>(s);
                    if (moe)
                    {
                        if (moe->device().is_gpu())
                            moe->setGPUStream(ensure_replica_publication_stream());
                        moe->setReplicaSet(normalized_replicas, participant_id);
                        count++;
                    }
                });
        }

        const std::string device =
            graph_builder_ ? graph_builder_->config().default_device.toString() : std::string{};
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "replica_set_cached_stage_applications",
            static_cast<double>(count),
            "rebalance",
            device,
            {{"domain_id", normalized_replicas.domain_id},
             {"participant", std::to_string(participant_id)},
             {"replicas", std::to_string(normalized_replicas.num_replicated)},
             {"replica_epoch", std::to_string(current_expert_replica_epoch_)},
             {"placement_changed", replica_placement_changed ? "true" : "false"},
             {"forward_engine", forward_engine_ ? "true" : "false"}});

        LOG_DEBUG("[DGO] Set expert replica info (" << normalized_replicas.num_replicated
                                                    << " replicas) on " << count
                                                    << " MoE stages (participant " << participant_id << ")");
    }

    void DeviceGraphOrchestrator::setExpertReplicaSet(
        const ExpertReplicaSet &replicas, int socket_id)
    {
        setExpertReplicaSetForParticipant(replicas, socket_id);
    }

    size_t DeviceGraphOrchestrator::releaseRawExpertWeights()
    {
        if (raw_expert_weights_released_after_graph_build_)
        {
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "raw_expert_release_already_complete_skips",
                1.0,
                "rebalance",
                primaryDeviceId().is_valid() ? primaryDeviceId().toString() : std::string{});
            return 0;
        }

        size_t total_freed = 0;
        int stage_count = 0;
        PreparedWeightStore *summary_store = prepared_weight_store_.get();

        if (forward_engine_)
        {
            forward_engine_->forEachCachedStage(
                ComputeStageType::MOE_EXPERT_FFN,
                [&](IComputeStage *s)
                {
                    auto *moe = dynamic_cast<MoEExpertComputeStage *>(s);
                    if (moe)
                    {
                        if (!summary_store)
                            summary_store = moe->buildWeightContext().prepared_store;
                        total_freed += moe->releaseRawExpertWeights();
                        ++stage_count;
                    }
                });
        }

        LOG_DEBUG("[DGO] Released raw expert weights across " << stage_count
                                                              << " MoE stages: " << (total_freed >> 20) << " MB freed");
        if (summary_store)
            if (auto concrete_weight_manager = std::dynamic_pointer_cast<WeightManager>(weight_manager_))
                concrete_weight_manager->logHostMemorySummary("after rebalance raw release");
        return total_freed;
    }

    bool DeviceGraphOrchestrator::buildMTPWorkspaceFamilyManifest(
        std::vector<std::unique_ptr<ComputeGraph>> &
            owned_mtp_graphs)
    {
        if (!owned_mtp_graphs.empty())
        {
            LOG_ERROR("[DGO] MTP workspace-family manifest requires an empty dedicated owner");
            return false;
        }
        if (!graph_builder_ ||
            !retainsMTPGraphCapacity(graph_builder_->config().mtp))
            return true;

        /*
         * A PP follower participates in the main-model MTP condition and
         * grouped-verifier transactions, but it never owns the learned
         * predictor.  Those main-model graph roles were declared above by
         * materializeForwardGraphForShape().  The sidecar manifest below is
         * exclusively the predictor's shifted-KV/weight family, so asking a
         * follower to declare it would require a cache that the same typed
         * ownership policy deliberately did not allocate.
         *
         * Keep this projection beside the live KV-cache construction rather
         * than rediscovering it from a cache-null check.  The latter would
         * silently turn a missing terminal predictor into a follower-shaped
         * success and split the MTP lifecycle authority.
         */
        const MTPStateRole mtp_state_role = resolveMTPStateRole(
            /*retains_capacity=*/true,
            !pp_stage_config_ || pp_stage_config_->has_lm_head);
        if (mtp_state_role == MTPStateRole::MainModelFollower)
            return true;
        if (!state_.isInitialized() || !hasGlobalWeights())
        {
            LOG_ERROR("[DGO] Cannot declare the MTP workspace family before inference state and weights are initialized");
            return false;
        }
        if (state_.mtp_kv_caches.empty() || !state_.mtp_kv_caches[0])
        {
            LOG_ERROR("[DGO] Cannot declare the MTP workspace family without its depth-zero KV cache");
            return false;
        }
        if (mtp_sidecar_condition_token_slot_width_ <= 0 || !state_.hidden)
        {
            LOG_ERROR("[DGO] Cannot declare the MTP workspace family without its configured row capacity and hidden-state owner");
            return false;
        }
        if (state_.device_id.is_gpu() &&
            (!mtp_sidecar_condition_token_dev_ ||
             !mtp_sidecar_position_ids_dev_))
        {
            LOG_ERROR("[DGO] Cannot declare the GPU MTP workspace family without persistent device token/position slots");
            return false;
        }
        if (!ensureMTPTerminalHiddenBuffer(
                mtp_sidecar_condition_token_slot_width_))
        {
            LOG_ERROR("[DGO] Cannot declare the MTP workspace family without its maximum-capacity terminal-hidden mailbox");
            return false;
        }

        const FrozenModelWeightSet *mtp_weight_set =
            selectMTPDecodeWeightSet("MTP workspace family manifest");
        if (!mtp_weight_set)
            return false;
        const ModelWeightBindings bindings =
            makeModelWeightBindings(
                *mtp_weight_set, state_.device_id);
        if (bindings.mtp.empty() || bindings.mtp.depths.empty())
        {
            LOG_ERROR("[DGO] Cannot declare the GPU MTP workspace family without depth-zero weight bindings");
            return false;
        }

        auto extension = [&](BufferId id) -> TensorBase *
        {
            const auto it = state_.extension_buffers.find(id);
            return it == state_.extension_buffers.end()
                       ? nullptr
                       : it->second.get();
        };

        MTPForwardOutput output{
            .logits = extension(BufferId::MTP_LOGITS),
            .gathered_logits =
                mtpSidecarRequiresLogitsGather(
                    /*kv_cache_only=*/false)
                    ? extension(BufferId::MTP_LOGITS_GATHERED)
                    : nullptr,
            .hidden = extension(BufferId::MTP_HIDDEN),
            .embedding = extension(BufferId::MTP_EMBEDDING),
            .norm_hidden = extension(BufferId::MTP_NORM_HIDDEN),
            .norm_embedding = extension(BufferId::MTP_NORM_EMBEDDING),
            .concat = extension(BufferId::MTP_CONCAT),
            .projected = extension(BufferId::MTP_PROJECTED),
            .q = extension(BufferId::MTP_Q_PROJ),
            .k = extension(BufferId::MTP_K_PROJ),
            .v = extension(BufferId::MTP_V_PROJ),
            .k_full_prefill = extension(BufferId::MTP_K_FULL_PREFILL),
            .v_full_prefill = extension(BufferId::MTP_V_FULL_PREFILL),
            .q_raw = extension(BufferId::MTP_FA_Q_RAW),
            .q_gate = extension(BufferId::MTP_FA_GATE),
            .attn_output = extension(BufferId::MTP_ATTN_OUTPUT),
            .attn_proj = extension(BufferId::MTP_ATTN_PROJ),
            .gate = extension(BufferId::MTP_GATE_PROJ),
            .up = extension(BufferId::MTP_UP_PROJ),
            .ffn_output = extension(BufferId::MTP_FFN_OUTPUT),
        };

        void *const publication_stream = state_.device_id.is_gpu()
                                             ? explicitGPUStreamForOperation(
                                                   "mtp_workspace_family_manifest")
                                             : nullptr;
        if (!publication_stream)
        {
            if (state_.device_id.is_gpu())
            {
                LOG_ERROR("[DGO] GPU MTP workspace-family declaration requires an explicit publication stream");
                return false;
            }
        }

        const int maximum_rows =
            mtp_sidecar_capture_layout_.maximumRows();
        const auto &sidecar_weights = bindings.mtp.depths[0].fa_block;
        if (sidecar_weights.moe_gate || sidecar_weights.moe_gate_exps ||
            sidecar_weights.moe_up_exps || sidecar_weights.moe_down_exps)
            output.moe = MoEActivationBindings::bind(
                graph_builder_->config().moe.routed_compute_policy, maximum_rows, extension);
        std::vector<int32_t> host_tokens(
            static_cast<size_t>(maximum_rows),
            0);
        std::vector<int> host_positions(
            static_cast<size_t>(maximum_rows),
            0);

        auto append_sidecar =
            [&](MTPSidecarCaptureRole role,
                int rows,
                TensorBase *terminal_hidden,
                BufferId terminal_hidden_buffer_id,
                bool kv_cache_only) -> bool
        {
            if (rows <= 0 || rows > maximum_rows || !terminal_hidden)
                return false;

            const int condition_slot = state_.device_id.is_gpu()
                                           ? mtp_sidecar_capture_layout_
                                                 .conditionTokenSlot(role, rows)
                                           : -1;
            const int required_token_capacity =
                (condition_slot + 1) *
                mtp_sidecar_condition_token_slot_width_;
            if (state_.device_id.is_gpu() &&
                (condition_slot < 0 ||
                 required_token_capacity >
                     mtp_sidecar_condition_token_capacity_))
            {
                LOG_ERROR("[DGO] MTP workspace-family token slot exceeds the persistent arena: role="
                          << static_cast<int>(role)
                          << " rows=" << rows
                          << " slot=" << condition_slot
                          << " required=" << required_token_capacity
                          << " capacity="
                          << mtp_sidecar_condition_token_capacity_);
                return false;
            }

            const auto *condition_tokens = state_.device_id.is_gpu()
                                               ? static_cast<const int32_t *>(
                                                     mtp_sidecar_condition_token_dev_) +
                                                     condition_slot *
                                                         mtp_sidecar_condition_token_slot_width_
                                               : nullptr;
            MTPForwardInput input{
                .capture_role = role,
                .draft_token_ids = host_tokens.data(),
                .draft_token_ids_device = condition_tokens,
                .terminal_hidden = terminal_hidden,
                .kv_cache = state_.mtp_kv_caches[0].get(),
                .position_ids = host_positions.data(),
                .position_ids_device = state_.device_id.is_gpu()
                                           ? mtp_sidecar_position_ids_dev_
                                           : nullptr,
                .sequence_lengths = nullptr,
                .sequence_lengths_device = nullptr,
                .batch_size = 1,
                .seq_len = rows,
                .first_sequence_index = 0,
                .device_state_publication_stream = publication_stream,
                .device = state_.device_id,
                .terminal_hidden_buffer_id =
                    terminal_hidden_buffer_id,
                .kv_cache_only = kv_cache_only,
                .graph_submission_intent = ForwardGraphSubmissionIntent::WorkspaceDeclaration,
            };

            ComputeGraph sidecar = graph_builder_->buildMTPGraph(
                0,
                bindings.mtp.depths[0],
                input,
                output);
            if (!sidecar.isWorkspaceDeclaration())
                throw std::logic_error("MTP workspace manifest received an executable sidecar graph");
            if (sidecar.size() == 0)
            {
                LOG_ERROR("[DGO] Model graph builder returned an empty MTP workspace-family participant: role="
                          << static_cast<int>(role)
                          << " rows=" << rows);
                return false;
            }
            owned_mtp_graphs.push_back(
                std::make_unique<ComputeGraph>(
                    std::move(sidecar)));
            return true;
        };

        /*
         * Full and chained sidecars have the same mathematical depth but bind
         * different terminal-hidden owners. Declare both because BufferId and
         * stage policy can legitimately select different workspace consumers.
         */
        if (!append_sidecar(
                MTPSidecarCaptureRole::Full,
                1,
                state_.prefix_terminal_hidden.get(),
                BufferId::PREFIX_TERMINAL_HIDDEN,
                /*kv_cache_only=*/false) ||
            !append_sidecar(
                MTPSidecarCaptureRole::Chained,
                1,
                output.hidden,
                BufferId::MTP_HIDDEN,
                /*kv_cache_only=*/false))
        {
            return false;
        }

        /*
         * Row count is part of every captured KV-only graph signature. Build the
         * entire configured interval so kernel dispatch changes at any M also
         * contribute their exact workspace names and capacities.
         */
        for (int rows = 1; rows <= maximum_rows; ++rows)
        {
            if (!append_sidecar(
                    MTPSidecarCaptureRole::KVOnly,
                    rows,
                    state_.prefix_terminal_hidden.get(),
                    BufferId::PREFIX_TERMINAL_HIDDEN,
                    /*kv_cache_only=*/true))
            {
                return false;
            }
        }

        /*
         * This count authenticates family completeness at the setup boundary.
         * It is intentionally emitted for CPU too: an executable CPU stage
         * retains its bound workspace addresses, so omitting its sidecar graph
         * is the same lifetime defect as omitting a captured GPU participant.
         */
        PerfStatsCollector::addCounter(
            "mtp",
            "workspace_family_graph_participants",
            static_cast<double>(owned_mtp_graphs.size()),
            "materialize",
            state_.device_id.toString(),
            {{"backend", state_.device_id.is_gpu() ? "gpu" : "cpu"},
             {"maximum_rows", std::to_string(maximum_rows)},
             {"full_graphs", "2"},
             {"kv_only_graphs", std::to_string(maximum_rows)}});

        /*
         * Terminal-hidden publication no longer contributes private workspace
         * buffers to this family manifest. Fixed ranges encode their source
         * offset in the captured D2D node, request terminals consume the arena
         * length array, and accepted rows consume the existing speculative
         * metadata workspace. Materializing a host-owned scalar or row array
         * here would reserve dead memory and keep the retired upload path alive.
         */

        if (!state_.device_id.is_gpu())
            return true;

        publishGraphBuildDeviceStateReady(
            mtp_graph_build_device_state_ready_,
            publication_stream,
            "mtp_workspace_family_manifest");
        /*
         * Manifest graphs are declarations and will not execute. Consume their
         * publication edge on the same explicit stream now so the next real
         * sidecar graph may publish a fresh edge without depending on an
         * unrelated request-reset boundary. Stream order carries these
         * immutable device-state writes into that later publication; no device
         * synchronization or host observation is introduced.
         */
        return waitForPendingGraphBuildDeviceStateReady(
            mtp_graph_build_device_state_ready_,
            publication_stream,
            DeviceTimelineRole::MTPSidecarGraph,
            "mtp_workspace_family_manifest_complete");
    }

    ForwardInput DeviceGraphOrchestrator::groupedVerifierWorkspaceInput(
        const ForwardInput &ordinary, const MTPGraphOwnerPlan &owner)
    {
        if ((!ordinary.device.is_cpu() && !ordinary.device.is_gpu()) ||
            ordinary.batch_size <= 0 ||
            ordinary.graph_submission_intent != ForwardGraphSubmissionIntent::WorkspaceDeclaration)
            throw std::invalid_argument("Grouped verifier workspace requires an ordinary cold declaration");
        if (!owner.retainsGraphCapacity())
            throw std::logic_error("Grouped verifier workspace requires its retained generation owner");
        const auto rows = ordinary.device.is_gpu()
            ? owner.retainedVerifierRequestRows()
            : DeviceRequestRowRanges::fullyActiveRequests(
                  ordinary.batch_size, owner.verifierRowsPerRequest());
        ForwardInput grouped = ordinary;
        grouped.batch_size = rows.requests();
        grouped.seq_len = rows.rowsPerRequest();
        grouped.token_ids = nullptr;
        grouped.token_ids_device = nullptr;
        grouped.position_ids = nullptr;
        grouped.position_ids_device = nullptr;
        grouped.sequence_lengths = nullptr;
        grouped.sequence_lengths_device = nullptr;
        grouped.execution_role = ForwardExecutionRole::GroupedMTPVerifier;
        grouped.execution_phase = ForwardExecutionPhase::Decode;
        grouped.position_offset = 1;
        grouped.real_seq_len = 0;
        grouped.bucket_seq_len = 0;
        return grouped;
    }

    bool DeviceGraphOrchestrator::materializeForwardGraphForShape(
        int seq_len,
        int batch_size,
        TensorBase *pipeline_hidden_input,
        std::shared_ptr<const Qwen4ExpFixedHeadPipelinePort> fixed_head_port)
    {
        if (forward_workspace_family_materialized_)
        {
            const uint64_t current_generation =
                workspaceGeneration(state_.device_id);
            if (state_.device_id.is_gpu() &&
                (forward_workspace_family_generation_ == 0 ||
                 current_generation !=
                     forward_workspace_family_generation_))
            {
                LOG_ERROR(
                    "[DGO] Materialized forward workspace family changed generation from "
                    << forward_workspace_family_generation_ << " to "
                    << current_generation);
                return false;
            }
            return true;
        }

        if (!state_.isInitialized())
        {
            LOG_ERROR("[DGO] Cannot materialize forward graph before inference state initialization");
            return false;
        }
        if (!hasGlobalWeights())
        {
            LOG_ERROR("[DGO] Cannot materialize forward graph before weights are configured");
            return false;
        }
        if (seq_len <= 0 || batch_size <= 0)
        {
            LOG_ERROR("[DGO] Invalid materialization shape: seq_len=" << seq_len
                                                                      << " batch_size=" << batch_size);
            return false;
        }
        if (batch_size > state_.batch_size)
        {
            LOG_ERROR("[DGO] Materialization batch size " << batch_size
                                                          << " exceeds initialized batch size " << state_.batch_size);
            return false;
        }

        const int total_tokens = batch_size * seq_len;
        if (total_tokens > state_.batch_size * state_.max_seq_len)
        {
            LOG_ERROR("[DGO] Materialization token count " << total_tokens
                                                           << " exceeds buffer capacity "
                                                           << (state_.batch_size * state_.max_seq_len));
            return false;
        }
        const int activation_seq_len =
            state_.activation_seq_len > 0 ? state_.activation_seq_len : state_.max_seq_len;
        if (total_tokens > state_.batch_size * activation_seq_len)
        {
            LOG_ERROR("[DGO] Materialization token count " << total_tokens
                                                           << " exceeds activation graph buffer capacity "
                                                           << (state_.batch_size * activation_seq_len)
                                                           << " (context capacity is "
                                                           << (state_.batch_size * state_.max_seq_len)
                                                           << ")");
            return false;
        }

        if (pp_stage_config_.has_value() &&
            !pp_stage_config_->has_embedding &&
            !pipeline_hidden_input)
        {
            LOG_ERROR(
                "[DGO] Non-embedding PP workspace materialization requires its stable previous-stage activation owner");
            return false;
        }

        std::vector<int> token_ids(static_cast<size_t>(total_tokens), 0);
        std::vector<int> position_ids(static_cast<size_t>(total_tokens));
        for (int i = 0; i < total_tokens; ++i)
            position_ids[static_cast<size_t>(i)] = i % seq_len;

        ModelBuffers model_buffers = state_.toModelBuffers();
        setBuffers(model_buffers);

        ForwardInput input;
        input.graph_submission_intent=ForwardGraphSubmissionIntent::WorkspaceDeclaration;
        input.token_ids = token_ids.data();
        input.position_ids = position_ids.data();
        input.batch_size = batch_size;
        input.seq_len = seq_len;
        input.execution_phase =
            seq_len == 1
                ? ForwardExecutionPhase::Decode
                : ForwardExecutionPhase::Prefill;
        input.position_offset = 0;
        const auto sparse_request_identity =
            currentMoESparseRequestIdentity();
        input.moe_overlay_collective_generation_id =
            sparse_request_identity
                ? sparse_request_identity->generation()
                : 0;
        input.moe_overlay_collective_step_id = 0;
        input.device = state_.device_id;
        input.kv_cache = state_.kv_cache.get();
        input.external_hidden_state = pipeline_hidden_input;
        input.fixed_head_pipeline_port = std::move(fixed_head_port);

        void *const publication_stream =
            state_.device_id.is_gpu()
                ? explicitGPUStreamForOperation(
                      "forward_workspace_family_manifest")
                : nullptr;
        if (state_.device_id.is_gpu() && !publication_stream)
        {
            LOG_ERROR("[DGO] GPU forward workspace-family declaration requires an explicit publication stream");
            return false;
        }
        input.device_state_publication_stream = publication_stream;
        if(state_.device_id.is_gpu()) {
            if(!request_token_ids_dev_ || !request_position_ids_dev_ || total_tokens>request_input_row_capacity_) {
                LOG_ERROR("[DGO] GPU workspace manifest requires its admitted native request input owners");
                return false;
            }
            input.token_ids=nullptr;input.token_ids_device=request_token_ids_dev_;
            input.position_ids=nullptr;input.position_ids_device=request_position_ids_dev_;
        }

        ForwardInput initial_declaration_input = input;
        if (!bindMTPMainTerminalHiddenTransaction(
                initial_declaration_input,
                batch_size,
                MTPMainTerminalHiddenGraphBinding::Purpose::
                    WorkspaceFamilyDeclaration))
        {
            LOG_ERROR("[DGO] Forward workspace-family declaration could not bind its MTP terminal publication topology");
            return false;
        }

        auto result = buildForwardGraph(initial_declaration_input);
        if (!result)
        {
            LOG_ERROR("[DGO] Eager forward graph materialization failed: " << result.error());
            return false;
        }

        std::vector<std::unique_ptr<ComputeGraph>>
            owned_primary_lane_graphs;
        std::vector<WorkspaceGraphParticipantRole>
            primary_lane_graph_roles;
        /*
         * Decode and prefill are different graph topologies. A numeric
         * "largest M" hint cannot discover a stage that is absent from the
         * decode graph, such as prefill-only recurrent-state collectives or
         * the padded-bucket terminal-row selector. Build those exact
         * declarations before generation one is allocated so later graph
         * capture can never introduce a new workspace name or capacity.
         */
        auto append_forward_participant =
            [&](ForwardInput participant_input,
                WorkspaceGraphParticipantRole participant_role,
                const char *participant_name) -> bool
        {
            if (state_.device_id.is_gpu() &&
                !waitForPendingGraphBuildDeviceStateReady(
                    main_graph_build_device_state_ready_,
                    publication_stream,
                    DeviceTimelineRole::MainForwardGraph,
                    participant_name))
            {
                return false;
            }

            /*
             * The workspace manifest must bind the same all-position output
             * owners as the live verifier. A recurrent-only declaration with
             * serial logits omits its multi-row vocabulary gather, allowing
             * that gather bank to alias snapshots consumed by accepted-state
             * publication. Admission sees the complete participant before any
             * captured pointer is published; ordinary bindings are restored
             * when this declaration finishes, including failure exits.
             */
            ModelBuffers participant_buffers = managed_buffers_;
            if (participant_role == WorkspaceGraphParticipantRole::GroupedVerifier)
            {
                if (!bindAllPositionLogitsOutputs(
                        participant_input.execution_role,
                        static_cast<std::size_t>(participant_input.batch_size) *
                            static_cast<std::size_t>(participant_input.seq_len),
                        participant_buffers.logits,
                        participant_buffers.logits_local))
                {
                    LOG_ERROR("[DGO] Grouped-verifier workspace declaration lacks its complete logits owners");
                    return false;
                }
            }
            /** @brief Scope one setup participant's model-buffer selection. */
            struct ScopedParticipantBuffers final
            {
                DeviceGraphOrchestrator &host;
                const ModelBuffers previous;
                /**
                 * @brief Publish the participant's borrowed tensor owners.
                 * @param owner Participant-local graph configuration authority.
                 * @param prior Complete buffer selection restored on destruction.
                 * @param selected Complete output selection for this declaration.
                 */
                ScopedParticipantBuffers(DeviceGraphOrchestrator &owner,
                    const ModelBuffers &prior, const ModelBuffers &selected)
                    : host(owner), previous(prior)
                {
                    host.setBuffers(selected);
                }
                /** @brief Restore ordinary output selection on every exit. */
                ~ScopedParticipantBuffers() { host.setBuffers(previous); }
                ScopedParticipantBuffers(const ScopedParticipantBuffers &) = delete;
                ScopedParticipantBuffers &operator=(const ScopedParticipantBuffers &) = delete;
            } participant_buffer_scope(*this, managed_buffers_, participant_buffers);

            auto participant = buildForwardGraph(participant_input);
            if (!participant)
            {
                LOG_ERROR("[DGO] Failed to declare forward workspace-family participant "
                          << participant_name << ": "
                          << participant.error());
                return false;
            }
            owned_primary_lane_graphs.push_back(
                std::make_unique<ComputeGraph>(
                    participant.takeGraph()));
            primary_lane_graph_roles.push_back(
                participant_role);
            return true;
        };

        const int maximum_prefill_seq_len =
            state_.activation_seq_len > 0
                ? state_.activation_seq_len
                : state_.max_seq_len;
        std::vector<int> maximum_prefill_tokens;
        std::vector<int> maximum_prefill_positions;
        if (maximum_prefill_seq_len > 1)
        {
            const int maximum_prefill_tokens_count =
                batch_size * maximum_prefill_seq_len;
            maximum_prefill_tokens.assign(
                static_cast<size_t>(maximum_prefill_tokens_count),
                0);
            maximum_prefill_positions.resize(
                static_cast<size_t>(maximum_prefill_tokens_count));
            for (int request = 0; request < batch_size; ++request)
            {
                for (int row = 0;
                     row < maximum_prefill_seq_len;
                     ++row)
                {
                    maximum_prefill_positions[static_cast<size_t>(
                        request * maximum_prefill_seq_len + row)] =
                        row;
                }
            }

            ForwardInput maximum_prefill_input = input;
            maximum_prefill_input.token_ids =
                maximum_prefill_tokens.data();
            maximum_prefill_input.position_ids =
                maximum_prefill_positions.data();
            maximum_prefill_input.seq_len =
                maximum_prefill_seq_len;
            maximum_prefill_input.execution_phase =
                ForwardExecutionPhase::Prefill;
            maximum_prefill_input.real_seq_len = 0;
            maximum_prefill_input.bucket_seq_len = 0;
            /*
             * Workspace declaration precedes request admission, but the
             * device request-length address already has model lifetime. Any
             * graph-owned terminal-row selector must capture that permanent
             * owner now; readiness of its contents is established only when a
             * real request later publishes the admission event. This is the
             * same ownership contract used by bucketed prefill below.
             */
            if (state_.device_id.is_gpu())
            {
                if (!request_token_ids_dev_ ||
                    !request_position_ids_dev_ ||
                    !request_sequence_lengths_dev_ ||
                    request_sequence_lengths_capacity_ < batch_size)
                {
                    LOG_ERROR("[DGO] Maximum GPU prefill workspace-family declaration requires persistent token, position, and request-length owners");
                    return false;
                }
                maximum_prefill_input.token_ids = nullptr;
                maximum_prefill_input.position_ids = nullptr;
                maximum_prefill_input.token_ids_device =
                    request_token_ids_dev_;
                maximum_prefill_input.position_ids_device =
                    request_position_ids_dev_;
                maximum_prefill_input.sequence_lengths_device =
                    static_cast<const int32_t *>(
                        request_sequence_lengths_dev_);
            }
            /*
             * MTP-enabled GPU prefill appends a depth-zero KV transaction to
             * this very graph. Declare that exact maximum-row topology before
             * the first workspace generation is published. The typed setup
             * binding carries permanent arena/KV addresses but is rejected by
             * ForwardExecutionEngine, so it can size stages without becoming
             * an executable fallback.
             */
            if (!bindShiftedMTPPrefillTransaction(
                    maximum_prefill_input,
                    batch_size,
                    ShiftedMTPPrefillGraphBinding::Purpose::
                        WorkspaceFamilyDeclaration))
            {
                LOG_ERROR("[DGO] Maximum prefill workspace-family declaration could not bind its shifted-MTP topology");
                return false;
            }
            if (!append_forward_participant(
                    maximum_prefill_input,
                    WorkspaceGraphParticipantRole::Prefill,
                    "forward_workspace_family_maximum_prefill"))
            {
                return false;
            }
        }

        /*
         * Bucketed prefill adds graph-owned row selection even when the first
         * request exactly fills the bucket. Runtime cache construction marks
         * exact bucket requests this way specifically so shorter replays can
         * reuse the capture. Mirror that production input policy here.
         */
        const auto resident_prefill_buckets =
            prefillGraphBucketsAtOrBelowCapacity(
                debugEnv().execution.prefill_graph_bucket_sizes,
                maximum_prefill_seq_len);
        std::vector<int> bucket_tokens;
        std::vector<int> bucket_positions;
        if (state_.device_id.is_gpu() &&
            debugEnv().execution.gpu_graphs &&
            debugEnv().execution.prefill_graph_buckets &&
            batch_size == 1 &&
            !resident_prefill_buckets.empty())
        {
            const int bucket_seq_len =
                resident_prefill_buckets.back();
            bucket_tokens.assign(
                static_cast<size_t>(bucket_seq_len),
                0);
            bucket_positions.resize(
                static_cast<size_t>(bucket_seq_len));
            std::iota(
                bucket_positions.begin(),
                bucket_positions.end(),
                0);

            ForwardInput bucket_input = input;
            bucket_input.token_ids = nullptr;
            bucket_input.position_ids = nullptr;
            bucket_input.seq_len = bucket_seq_len;
            bucket_input.execution_phase =
                ForwardExecutionPhase::Prefill;
            bucket_input.real_seq_len = bucket_seq_len;
            bucket_input.bucket_seq_len = bucket_seq_len;
            /*
             * Workspace-family declaration happens before any request has
             * populated the arena request-length row.  The address is already
             * permanent, however, and the captured bucket graph must bind the
             * same owner that runtime admission will populate and event-publish.
             * Requiring an "active" value here would confuse data readiness with
             * pointer lifetime and would force graph declaration back toward a
             * host-owned replay scalar.
             */
            if (!request_sequence_lengths_dev_ ||
                request_sequence_lengths_capacity_ < 1)
            {
                LOG_ERROR("[DGO] Bucketed GPU prefill workspace-family declaration requires the persistent request-length owner");
                return false;
            }
            bucket_input.sequence_lengths_device =
                static_cast<const int32_t *>(
                    request_sequence_lengths_dev_);
            if (!append_forward_participant(
                    bucket_input,
                    WorkspaceGraphParticipantRole::Prefill,
                    "forward_workspace_family_maximum_prefill_bucket"))
            {
                return false;
            }
        }

        std::vector<int> grouped_verifier_tokens;
        std::vector<int> grouped_verifier_positions;
        std::vector<int> grouped_verifier_request_lengths;
        if (retainsMTPGraphCapacity(graph_builder_->config().mtp) &&
            mtp_max_verifier_rows_ > 1)
        {
            /*
             * The grouped verifier is a main-model graph, not an MTP sidecar.
             * It selects speculative recurrent-state banks and grouped
             * decode-equivalent kernels that ordinary decode/prefill graphs do
             * not contain. Declare its maximum configured M with the same
             * backend-specific authority used by production replay: CPU owns
             * host rows, while GPU owns persistent arena rows. Applying the GPU
             * pointer requirement to CPU makes the canonical host path
             * impossible to materialize; retaining host rows beside GPU rows
             * would instead create two competing authorities.
             */
            // GPU metadata has exactly the admitted generation R entries,
            // independently of the ordinary/prefill batch passed to this method.
            ForwardInput grouped_input = groupedVerifierWorkspaceInput(input, mtp_graph_owner_plan_);
            const int grouped_requests = grouped_input.batch_size;
            const int grouped_rows = grouped_input.seq_len;
            const int grouped_total_rows = DeviceRequestRowRanges::fullyActiveRequests(
                grouped_requests, grouped_rows).physicalRows();
            ScopedGroupedMTPVerifierGraphPolicy grouped_verifier_scope(
                graph_builder_.get(),
                &compute_all_position_logits_,
                &compute_row_indexed_all_position_logits_,
                &row_indexed_all_position_logits_row_count_,
                &mtp_verifier_outcome_graph_mode_,
                grouped_total_rows,
                MTPVerifierOutcomeGraphMode::Disabled);
            if (!grouped_verifier_scope.ready())
            {
                LOG_ERROR("[DGO] Graph builder rejected complete grouped-verifier workspace-family policy");
                return false;
            }
            grouped_verifier_tokens.assign(
                static_cast<size_t>(grouped_total_rows),
                0);
            grouped_verifier_positions.resize(
                static_cast<size_t>(grouped_total_rows));
            for (int request = 0; request < grouped_requests; ++request)
            {
                std::iota(
                    grouped_verifier_positions.begin() +
                        static_cast<ptrdiff_t>(request * grouped_rows),
                    grouped_verifier_positions.begin() +
                        static_cast<ptrdiff_t>((request + 1) * grouped_rows),
                    1);
            }

            if (state_.device_id.is_gpu())
            {
                if (!mtp_verifier_input_tokens_dev_ ||
                    !mtp_verifier_position_ids_dev_ ||
                    !mtp_verifier_request_lengths_dev_)
                {
                    LOG_ERROR("[DGO] GPU grouped-verifier workspace-family declaration requires persistent device token, position, and request-length rows");
                    return false;
                }
                grouped_input.token_ids_device =
                    mtp_verifier_input_tokens_dev_;
                grouped_input.position_ids_device =
                    mtp_verifier_position_ids_dev_;
                grouped_input.sequence_lengths_device =
                    static_cast<const int32_t *>(
                        mtp_verifier_request_lengths_dev_);
            }
            else
            {
                grouped_input.token_ids =
                    grouped_verifier_tokens.data();
                grouped_input.position_ids =
                    grouped_verifier_positions.data();
                if (grouped_requests > 1)
                {
                    grouped_verifier_request_lengths.assign(
                        static_cast<size_t>(grouped_requests),
                        grouped_rows);
                    grouped_input.sequence_lengths =
                        &grouped_verifier_request_lengths;
                }
            }
            if (!append_forward_participant(
                    grouped_input,
                    WorkspaceGraphParticipantRole::
                        GroupedVerifier,
                    "forward_workspace_family_grouped_verifier"))
            {
                return false;
            }
        }

        std::vector<int> mtp_condition_tokens;
        std::vector<int> mtp_condition_positions;
        std::vector<int> mtp_condition_sequence_lengths;
        if (state_.device_id.is_gpu() &&
            retainsMTPGraphCapacity(graph_builder_->config().mtp))
        {
            /*
             * A live MTP condition is not the ordinary scalar decode graph.
             * It owns one row per active request, binds device-resident token,
             * position, and sequence-length rows, and gives GDN/short-conv
             * stages a distinct workspace namespace. Materialize that exact
             * topology before the first request so graph capture can never
             * discover a new scratch name or move a published address.
             */
            const int condition_requests =
                stochastic_batch_output_request_capacity_;
            const auto &logical_state =
                device_resident_logical_sequence_state_storage_;
            if (!logical_state.validFor(condition_requests))
            {
                LOG_ERROR("[DGO] MTP-condition workspace-family declaration requires "
                          << condition_requests
                          << " persistent logical-state rows");
                return false;
            }

            ScopedLiveMTPRequestBatchConditionPolicy condition_scope(
                graph_builder_.get(),
                &live_mtp_request_batch_condition_,
                /*enable=*/true);
            if (!condition_scope.ready())
            {
                LOG_ERROR("[DGO] Graph builder rejected the MTP-condition workspace-family policy");
                return false;
            }

            mtp_condition_tokens.assign(
                static_cast<size_t>(condition_requests),
                0);
            mtp_condition_positions.assign(
                static_cast<size_t>(condition_requests),
                0);
            mtp_condition_sequence_lengths.assign(
                static_cast<size_t>(condition_requests),
                1);

            ForwardInput condition_input = input;
            condition_input.token_ids =
                mtp_condition_tokens.data();
            condition_input.token_ids_device =
                logical_state.next_condition_tokens_device;
            condition_input.position_ids =
                mtp_condition_positions.data();
            condition_input.position_ids_device =
                logical_state.target_cached_tokens_device;
            condition_input.sequence_lengths =
                &mtp_condition_sequence_lengths;
            condition_input.sequence_lengths_device =
                logical_state.target_cached_tokens_device;
            condition_input.execution_role =
                ForwardExecutionRole::MTPCondition;
            condition_input.execution_phase =
                ForwardExecutionPhase::Decode;
            condition_input.batch_size =
                condition_requests;
            condition_input.seq_len = 1;
            condition_input.position_offset = 0;
            condition_input.real_seq_len = 0;
            condition_input.bucket_seq_len = 0;
            if (!append_forward_participant(
                    condition_input,
                    WorkspaceGraphParticipantRole::
                        MTPCondition,
                    "forward_workspace_family_mtp_condition"))
            {
                return false;
            }
        }

        /*
         * buildForwardGraph() publishes graph-build device state for every
         * materialized participant. append_forward_participant() consumes the
         * previous participant before constructing the next one, leaving the
         * final forward declaration intentionally outstanding. Close that final
         * producer/consumer edge on the manifest's explicit stream before any
         * request can build a real graph. Without this terminal consume, the
         * first runtime prefill would attempt to publish over a still-owned
         * declaration event and correctly terminate for double publication.
         *
         * This wait is stream ordered. It neither synchronizes the device nor
         * exposes graph-build state to the host.
         */
        if (state_.device_id.is_gpu() &&
            !waitForPendingGraphBuildDeviceStateReady(
                main_graph_build_device_state_ready_,
                publication_stream,
                DeviceTimelineRole::MainForwardGraph,
                "forward_workspace_family_manifest_complete"))
        {
            LOG_ERROR("[DGO] Failed to close forward workspace-family device-state publication");
            return false;
        }

        std::vector<std::unique_ptr<ComputeGraph>>
            owned_mtp_family_graphs;
        if (!buildMTPWorkspaceFamilyManifest(
                owned_mtp_family_graphs))
        {
            LOG_ERROR("[DGO] Eager MTP workspace-family graph materialization failed");
            return false;
        }
        if (!materializeDeviceMoERebalanceMaintenanceGraphForFamily())
        {
            LOG_ERROR("[DGO] Eager device MoE maintenance graph-family materialization failed");
            return false;
        }

        std::vector<WorkspaceGraphParticipant>
            serial_family_graphs;
        serial_family_graphs.reserve(
            owned_primary_lane_graphs.size() +
            owned_mtp_family_graphs.size() +
            (device_moe_rebalance_maintenance_graph_.graph ? 1U : 0U));
        for (size_t participant_index = 0;
             participant_index <
             owned_primary_lane_graphs.size();
             ++participant_index)
        {
            serial_family_graphs.push_back(
                WorkspaceGraphParticipant{
                    .graph =
                        owned_primary_lane_graphs[
                            participant_index]
                            .get(),
                    .role =
                        primary_lane_graph_roles[
                            participant_index],
                });
        }
        for (const auto &participant :
             owned_mtp_family_graphs)
        {
            serial_family_graphs.push_back(
                WorkspaceGraphParticipant{
                    .graph = participant.get(),
                    .role =
                        WorkspaceGraphParticipantRole::
                            GroupedVerifier,
                });
        }
        if (device_moe_rebalance_maintenance_graph_.graph)
        {
            serial_family_graphs.push_back(
                WorkspaceGraphParticipant{
                    .graph =
                        device_moe_rebalance_maintenance_graph_.graph.get(),
                    .role =
                        WorkspaceGraphParticipantRole::
                            MoERebalanceMaintenance,
                    .lifetime =
                        WorkspaceGraphParticipantLifetime::
                            PersistentAcrossParticipants,
                });
        }

        if (!ensureDeviceWorkspaceAllocated(
                result.graph(),
                seq_len,
                WorkspaceGraphFamilyPolicy::
                    SerialDeviceFamilyLargestParticipant,
                WorkspaceGraphParticipantRole::Decode,
                serial_family_graphs))
        {
            LOG_ERROR("[DGO] Eager forward graph workspace materialization failed for "
                      << (graph_builder_ ? graph_builder_->architectureName() : std::string("unknown"))
                      << " shape=[batch=" << batch_size << ", seq=" << seq_len << "]");
            return false;
        }
        if (!materializeMTPTerminalHiddenPublicationGraphs())
        {
            LOG_ERROR("[DGO] Eager terminal-hidden publication graph materialization failed for "
                      << (graph_builder_ ? graph_builder_->architectureName() : std::string("unknown"))
                      << " shape=[batch=" << batch_size << ", seq=" << seq_len << "]");
            return false;
        }
        if (moe_overlay_epoch_execution_binding_ &&
            !materializeMoEOverlayEpochBoundaryGraphs())
        {
            LOG_ERROR("[DGO] Eager ExpertOverlay epoch-boundary graph materialization failed for "
                      << state_.device_id.toString());
            return false;
        }
        if (device_moe_rebalance_maintenance_graph_.graph)
        {
            const uint64_t published_generation =
                workspaceGeneration(state_.device_id);
            if (published_generation == 0)
            {
                LOG_ERROR("[DGO] Device MoE maintenance graph-family participant has no published GPU workspace generation on "
                          << state_.device_id.toString());
                return false;
            }
            device_moe_rebalance_maintenance_graph_.workspace_generation =
                published_generation;

            IBackend *const maintenance_backend =
                getBackendFor(state_.device_id);
            IWorkerGPUContext *maintenance_gpu_ctx = nullptr;
            try
            {
                maintenance_gpu_ctx =
                    &GPUDeviceContextPool::instance().getContext(
                        state_.device_id);
            }
            catch (const std::exception &exception)
            {
                LOG_ERROR("[DGO] Device MoE scheduler setup could not resolve its worker context on "
                          << state_.device_id.toString() << ": "
                          << exception.what());
                return false;
            }
            if (!maintenance_backend || !maintenance_gpu_ctx ||
                !device_moe_rebalance_maintenance_graph_.segment_cache
                     .ensureCaptureStream(
                         maintenance_gpu_ctx,
                         state_.device_id,
                         /*context_from_process_pool=*/true))
            {
                LOG_ERROR("[DGO] Device MoE scheduler setup could not preallocate its exact stream on "
                          << state_.device_id.toString());
                return false;
            }
            void *const maintenance_stream =
                device_moe_rebalance_maintenance_graph_.segment_cache
                    .capture_stream;
            if (!maintenance_stream)
            {
                LOG_ERROR("[DGO] Device MoE scheduler setup produced a null stream on "
                          << state_.device_id.toString());
                return false;
            }

            /*
             * A topology-wide controller performs this handoff while it
             * materializes each endpoint.  The homogeneous single-domain
             * special case has no external controller service: this retained
             * maintenance stream is the controller.  Publish the complete
             * physical runtime family here, after main, MTP, and maintenance
             * graph declaration has finished, so dormant retained layers
             * (for example a disabled NextN source layer) cannot remain as an
             * epoch-zero template when the first maintenance scan executes.
             *
             * finalizeMoEOverlayDeviceControllerRuntime() and the following
             * maintenance work share this exact stream.  Their ordering is
             * therefore entirely device-side; no setup or inference stream is
             * synchronized on the host.
             */
            if (!publishMoEOverlayDeviceInitialRuntime(
                    maintenance_stream))
            {
                LOG_ERROR("[DGO] Participant-local device MoE controller could not publish its complete initial runtime family on "
                          << state_.device_id.toString());
                return false;
            }
            if (!device_moe_rebalance_maintenance_graph_.completion_event)
            {
                void *const raw_event = maintenance_backend->createEvent(
                    state_.device_id.gpu_ordinal());
                if (!raw_event)
                {
                    LOG_ERROR("[DGO] Device MoE scheduler setup could not preallocate its completion event on "
                              << state_.device_id.toString());
                    return false;
                }
                const int ordinal = state_.device_id.gpu_ordinal();
                device_moe_rebalance_maintenance_graph_.completion_event.reset(
                    raw_event,
                    [maintenance_backend, ordinal](void *event)
                    {
                        if (event)
                            maintenance_backend->destroyEvent(event, ordinal);
                    });
            }

            /* HIP's scheduler observer and event are setup-owned resources.
             * Creating either at the first decode boundary would make ordinary
             * inference allocate host/device runtime objects in the hot path. */
            if (state_.device_id.is_rocm())
            {
                if (!device_moe_rebalance_dispatch_ticket_host_scratch_)
                {
                    device_moe_rebalance_dispatch_ticket_host_scratch_ =
                        std::make_unique<
                            PinnedMoERebalanceDispatchTicketScratch>();
                }
                if (!device_moe_rebalance_dispatch_ticket_host_scratch_
                         ->valid() &&
                    !device_moe_rebalance_dispatch_ticket_host_scratch_
                         ->allocate(
                             maintenance_backend,
                             state_.device_id.gpu_ordinal()))
                {
                    LOG_ERROR("[DGO] HIP Device MoE scheduler could not allocate its one-way pinned ticket observer on "
                              << state_.device_id.toString());
                    return false;
                }
                if (!device_moe_rebalance_dispatch_ticket_ready_event_)
                {
                    void *const raw_event = maintenance_backend->createEvent(
                        state_.device_id.gpu_ordinal());
                    if (!raw_event)
                    {
                        LOG_ERROR("[DGO] HIP Device MoE scheduler could not allocate its ticket-ready event on "
                                  << state_.device_id.toString());
                        return false;
                    }
                    const int ordinal = state_.device_id.gpu_ordinal();
                    device_moe_rebalance_dispatch_ticket_ready_event_.reset(
                        raw_event,
                        [maintenance_backend, ordinal](void *event)
                        {
                            if (event)
                                maintenance_backend->destroyEvent(
                                    event,
                                    ordinal);
                        });
                }
            }

            /*
             * The first request reset can precede eager workspace-family
             * materialization. At that point there is no maintenance graph and
             * therefore no typed transaction owner to reset. Initialize the
             * newly bound persistent controller here, on the same explicit
             * family stream, then publish the ordinary request-reset edge so the
             * first real forward/verifier stream cannot observe zeroed cadence
             * state. This is a one-time binding lifecycle action; every later
             * request uses resetInferenceState() and the same transaction owner.
             */
            auto &maintenance = device_moe_rebalance_maintenance_graph_;
            if (!maintenance.request_transaction_initialized)
            {
                if (!waitForPendingRequestStateReset(
                        publication_stream,
                        DeviceTimelineRole::RequestStateReset,
                        "workspace_family_device_moe_transaction_join"))
                {
                    LOG_ERROR("[DGO] First device MoE transaction initialization could not join prior request reset on "
                              << state_.device_id.toString());
                    return false;
                }
                if (!maintenance.resetRequestOwnedDeviceTransaction(
                        publication_stream) ||
                    !initializeDeviceMoERebalanceDispatchTicketOnStream(
                        publication_stream,
                        session_epoch_))
                {
                    LOG_ERROR("[DGO] First device MoE transaction initialization failed after workspace-family binding on "
                              << state_.device_id.toString());
                    return false;
                }
                publishRequestStateResetReady(
                    publication_stream,
                    "workspace_family_device_moe_transaction_init");
                PerfStatsCollector::addCounter(
                    "moe_rebalance",
                    "device_controller_first_request_initializations",
                    1.0,
                    "request_reset",
                    state_.device_id.toString(),
                    {{"ordering", "request_reset_event"},
                     {"workspace_generation",
                      std::to_string(published_generation)}});
            }

            std::string cadence_error;
            if (!prepareDeviceMoERebalanceCadenceTransactionSources(
                    maintenance_gpu_ctx,
                    maintenance_backend,
                    maintenance_stream,
                    &cadence_error))
            {
                LOG_ERROR("[DGO] Device MoE cadence source setup failed on "
                          << state_.device_id.toString() << ": "
                          << cadence_error);
                return false;
            }
        }

        const size_t stage_count = result.graph().size();
        forward_workspace_family_generation_ =
            workspaceGeneration(state_.device_id);
        if (state_.device_id.is_gpu() &&
            forward_workspace_family_generation_ == 0)
        {
            LOG_ERROR(
                "[DGO] Complete GPU workspace-family materialization published no generation");
            return false;
        }
        forward_workspace_family_materialized_ = true;
        LOG_DEBUG("[DGO] Eagerly materialized forward graph for "
                  << (graph_builder_ ? graph_builder_->architectureName() : std::string("unknown"))
                  << " shape=[batch=" << batch_size << ", seq=" << seq_len << "]"
                  << " stages=" << stage_count);
        return true;
    }

    ExpertTransferResult DeviceGraphOrchestrator::transferExpertWeights(
        const std::vector<ExpertMigration> &manifest)
    {
        if (manifest.empty() || !mpi_ctx_)
            return {};

        int my_rank = mpi_ctx_->rank();
        MPI_Comm comm = mpi_ctx_->communicator();

        // Collect MoE stages by layer from forward engine's graph cache.
        std::unordered_map<int, MoEExpertComputeStage *> moe_by_layer;
        if (forward_engine_)
        {
            forward_engine_->forEachCachedStage(
                ComputeStageType::MOE_EXPERT_FFN,
                [&](IComputeStage *stage)
                {
                    auto *moe_stage = dynamic_cast<MoEExpertComputeStage *>(stage);
                    if (moe_stage && moe_stage->layerIndex() >= 0)
                        moe_by_layer[moe_stage->layerIndex()] = moe_stage;
                });
        }

        if (moe_by_layer.empty())
        {
            throw std::runtime_error(
                "Dynamic expert transfer requires canonical ForwardExecutionEngine MoE stages");
        }

        LOG_DEBUG("[DGO] Found " << moe_by_layer.size() << " MoE stages for weight transfer"
                                 << " (forward_engine=" << (forward_engine_ ? "yes" : "no") << ")");

        if (!primaryDeviceId().is_cpu())
        {
            throw std::runtime_error(
                "Cross-rank GPU expert ownership movement requires the device-resident transfer protocol; host serialization is forbidden");
        }

        auto get_weights =
            [&](int layer_idx, int expert_id) -> ExpertPackedWeights
        {
            auto it = moe_by_layer.find(layer_idx);
            if (it == moe_by_layer.end())
            {
                throw std::runtime_error(
                    "Dynamic expert transfer cannot find the exact source layer stage");
            }
            return it->second->detachPreparedExpert(expert_id);
        };

        auto make_engine =
            [](std::unique_ptr<IPackedWeights> weights)
                -> std::shared_ptr<ITensorGemm>
        {
            return llaminar::v2::kernels::KernelFactory::
                createExpertGemmFromPackedWeights(std::move(weights));
        };

        ExpertTransferEvidence evidence;
        ExpertTransferResult result;
        const int target_numa_node = cpuBackendNUMANode();
        if (target_numa_node < 0)
        {
            throw std::runtime_error(
                "Cross-rank CPU expert movement requires a rank-local NUMA-bound CPU backend");
        }
        if (!frozen_weight_set_)
            throw std::logic_error("CPU ownership transfer requires its frozen persistent weight owner");
        result.prepared = ExpertWeightTransfer::transferLayeredPreparedCPU(
            manifest,
            get_weights,
            make_engine,
            my_rank,
            target_numa_node,
            comm,
            &evidence,
            CPUWeightStoragePlacement::accounted(physical_memory_authority_, primaryDeviceId(),
                frozen_weight_set_->physicalMemoryOwner(),
                CPUWeightStoragePlacement::onNode(target_numa_node)));

        const std::string device =
            primaryDeviceId().is_valid() ? primaryDeviceId().toString() : std::string{};
        const PerfStatsCollector::Tags tags{
            {"domain_id", moe_rebalance_controller_
                              ? moe_rebalance_controller_->domainId()
                              : std::string("default")},
            {"rank", std::to_string(my_rank)},
            {"kind", "ownership"}};
        const auto record_phase = [&](const char *name, uint64_t duration_ns)
        {
            PerfStatsCollector::recordTimingNs(
                "moe_rebalance", name, duration_ns,
                "rebalance", device, tags);
        };
        record_phase("weight_transfer_payload_prepare", evidence.payload_prepare_ns);
        record_phase("weight_transfer_metadata_exchange", evidence.metadata_exchange_ns);
        record_phase("weight_transfer_receive_allocation", evidence.receive_allocation_ns);
        record_phase("weight_transfer_payload_exchange", evidence.payload_exchange_ns);
        record_phase("weight_transfer_result_publication", evidence.result_publication_ns);
        record_phase("weight_transfer_total", evidence.total_ns);
        PerfStatsCollector::addCounter(
            "moe_rebalance", "weight_transfer_manifest_entries",
            static_cast<double>(evidence.manifest_entries),
            "rebalance", device, tags);
        PerfStatsCollector::addCounter(
            "moe_rebalance", "weight_transfer_outgoing_entries",
            static_cast<double>(evidence.outgoing_entries),
            "rebalance", device, tags);
        PerfStatsCollector::addCounter(
            "moe_rebalance", "weight_transfer_incoming_entries",
            static_cast<double>(evidence.incoming_entries),
            "rebalance", device, tags);
        PerfStatsCollector::addCounter(
            "moe_rebalance", "weight_transfer_outgoing_bytes",
            static_cast<double>(evidence.outgoing_bytes),
            "rebalance", device, tags);
        PerfStatsCollector::addCounter(
            "moe_rebalance", "weight_transfer_incoming_bytes",
            static_cast<double>(evidence.incoming_bytes),
            "rebalance", device, tags);
        return result;
    }

    bool DeviceGraphOrchestrator::materializeCPUCurrentBatchLLEPTransaction(
        CPUCurrentBatchLLEPTransactionState &state,
        ICPUCurrentBatchLLEPExpertConsumer &expert_consumer,
        IGlobalTPContext &tp_ctx)
    {
        if (!primaryDeviceId().is_cpu() || state.active ||
            state.layer_idx < 0 || state.num_experts <= 0 ||
            state.durable_parent_epoch == 0 ||
            state.durable_domain.empty() ||
            state.durable_epoch_lease.has_value() ||
            state.participant_count <= 1 ||
            state.participant_count != tp_ctx.degree() ||
            state.participant_id != tp_ctx.myIndex() ||
            state.owner_participants.size() !=
                static_cast<size_t>(state.num_experts) ||
            state.owner_mask.size() !=
                static_cast<size_t>(state.num_experts) ||
            state.transient_resident_mask.size() !=
                static_cast<size_t>(state.num_experts))
        {
            LOG_ERROR("[DGO] Invalid CPU current-batch LLEP begin contract"
                      << " layer=" << state.layer_idx
                      << " participant=" << state.participant_id
                      << " participants=" << state.participant_count);
            return false;
        }
        if (active_cpu_llep_state_ || active_cpu_llep_expert_consumer_ ||
            active_cpu_llep_tp_ctx_)
        {
            LOG_ERROR("[DGO] CPU current-batch LLEP transactions may not overlap"
                      << " requested_layer=" << state.layer_idx
                      << " active_layer="
                      << (active_cpu_llep_state_
                              ? active_cpu_llep_state_->layer_idx
                              : -1));
            return false;
        }
        if (expert_consumer.cpuCurrentBatchLLEPLayerIndex() !=
                state.layer_idx ||
            expert_consumer.cpuCurrentBatchLLEPParticipantId() !=
                state.participant_id ||
            !expert_consumer.cpuCurrentBatchLLEPUsesCPU())
        {
            LOG_ERROR("[DGO] CPU current-batch LLEP target stage identity mismatch"
                      << " transaction_layer=" << state.layer_idx
                      << " stage_layer="
                      << expert_consumer.cpuCurrentBatchLLEPLayerIndex()
                      << " transaction_participant="
                      << state.participant_id
                      << " stage_participant="
                      << expert_consumer.cpuCurrentBatchLLEPParticipantId());
            return false;
        }

        active_cpu_llep_state_ = &state;
        active_cpu_llep_expert_consumer_ = &expert_consumer;
        active_cpu_llep_tp_ctx_ = &tp_ctx;
        auto clear_active_publication = [&]() noexcept
        {
            active_cpu_llep_state_ = nullptr;
            active_cpu_llep_expert_consumer_ = nullptr;
            active_cpu_llep_tp_ctx_ = nullptr;
        };

        try
        {
            std::vector<ExpertMigration> manifest;
            manifest.reserve(state.status.weight_transfer_count);
            for (uint32_t transfer_index = 0;
                 transfer_index < state.status.weight_transfer_count;
                 ++transfer_index)
            {
                const auto &transfer = state.transfers[transfer_index];
                if (transfer.expert >=
                        static_cast<uint32_t>(state.num_experts) ||
                    transfer.source_participant >=
                        static_cast<uint32_t>(state.participant_count) ||
                    transfer.destination_participant >=
                        static_cast<uint32_t>(state.participant_count) ||
                    transfer.source_participant ==
                        transfer.destination_participant ||
                    state.owner_participants[transfer.expert] !=
                        transfer.source_participant)
                {
                    throw std::logic_error(
                        "CPU current-batch LLEP transfer does not match authoritative ownership");
                }
                manifest.push_back({
                    .layer_idx = state.layer_idx,
                    .expert_id = static_cast<int>(transfer.expert),
                    .src_rank =
                        static_cast<int>(transfer.source_participant),
                    .dst_rank =
                        static_cast<int>(transfer.destination_participant),
                });
            }

            ExpertTransferEvidence evidence;
            ReceivedPreparedExpertsMap prepared_arrivals;
            if (!manifest.empty())
            {
                auto get_weights =
                    [&](int layer_idx, int expert_id)
                        -> ExpertPackedWeights
                {
                    if (layer_idx != state.layer_idx)
                    {
                        throw std::logic_error(
                            "CPU current-batch LLEP requested a foreign layer payload");
                    }
                    return expert_consumer
                        .cloneCPUCurrentBatchLLEPPreparedExpert(
                            expert_id,
                            state.durable_parent_epoch,
                            CPUWeightStoragePlacement::accounted(physical_memory_authority_, primaryDeviceId(),
                                PhysicalMemoryOwner::ExpertMigrationStaging,
                                CPUWeightStoragePlacement::local()));
                };
                auto make_engine =
                    [](std::unique_ptr<IPackedWeights> weights)
                        -> std::shared_ptr<ITensorGemm>
                {
                    return llaminar::v2::kernels::KernelFactory::
                        createExpertGemmFromPackedWeights(
                            std::move(weights));
                };
                prepared_arrivals =
                    ExpertWeightTransfer::transferLayeredPreparedCPU(
                        manifest,
                        get_weights,
                        make_engine,
                        state.participant_id,
                        cpuBackendNUMANode(),
                        tp_ctx.communicator(),
                        &evidence,
                        CPUWeightStoragePlacement::accounted(physical_memory_authority_, primaryDeviceId(),
                            PhysicalMemoryOwner::ExpertShadowSlots,
                            CPUWeightStoragePlacement::onNode(cpuBackendNUMANode())));
            }

            const std::unordered_map<int, PreparedExpertEngines> *
                layer_arrivals = nullptr;
            const auto arrival_it =
                prepared_arrivals.find(state.layer_idx);
            if (arrival_it != prepared_arrivals.end())
                layer_arrivals = &arrival_it->second;

            if (!expert_consumer
                     .installCPUCurrentBatchLLEPTransientResidency(
                         state,
                         layer_arrivals))
            {
                throw std::runtime_error(
                    "CPU current-batch LLEP could not prepare every transient expert arrival");
            }

            ++state.transaction_epoch;
            state.active = true;

            const PerfStatsCollector::Tags tags{
                {"layer", std::to_string(state.layer_idx)},
                {"participant", std::to_string(state.participant_id)},
                {"participants", std::to_string(state.participant_count)},
                {"kind", "cpu_current_batch_llep"}};
            PerfStatsCollector::recordTimingNs(
                "moe_rebalance",
                "cpu_llep_weight_transfer_total",
                evidence.total_ns,
                "prefill",
                primaryDeviceId().toString(),
                tags);
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "cpu_llep_weight_transfer_outgoing_bytes",
                static_cast<double>(evidence.outgoing_bytes),
                "prefill",
                primaryDeviceId().toString(),
                tags);
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "cpu_llep_weight_transfer_incoming_bytes",
                static_cast<double>(evidence.incoming_bytes),
                "prefill",
                primaryDeviceId().toString(),
                tags);
            return true;
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[DGO] CPU current-batch LLEP begin failed"
                      << " layer=" << state.layer_idx
                      << " participant=" << state.participant_id
                      << " error=" << e.what());
            state.active = false;
            try
            {
                (void)expert_consumer
                    .discardCPUCurrentBatchLLEPTransientResidency(state);
            }
            catch (...)
            {
                LOG_ERROR("[DGO] CPU current-batch LLEP rollback also failed; "
                          "the runner is not reusable");
            }
            clear_active_publication();
            return false;
        }
    }

    bool DeviceGraphOrchestrator::restoreCPUCurrentBatchLLEPPhysicalState(
        CPUCurrentBatchLLEPTransactionState &state,
        ICPUCurrentBatchLLEPExpertConsumer &expert_consumer,
        IGlobalTPContext &tp_ctx)
    {
        if (!state.active || !state.durable_epoch_lease.has_value() ||
            state.durable_parent_epoch == 0 ||
            state.durable_domain.empty() ||
            state.durable_epoch_lease->purpose() !=
                MoEOverlayResidencyAuthority::TicketLeasePurpose::
                    CurrentBatchLLEP ||
            state.durable_epoch_lease->epoch() !=
                state.durable_parent_epoch ||
            active_cpu_llep_state_ != &state ||
            active_cpu_llep_expert_consumer_ != &expert_consumer ||
            active_cpu_llep_tp_ctx_ != &tp_ctx ||
            state.participant_count != tp_ctx.degree() ||
            state.participant_id != tp_ctx.myIndex())
        {
            LOG_ERROR("[DGO] CPU current-batch LLEP restore does not match "
                      "the active transaction"
                      << " layer=" << state.layer_idx
                      << " participant=" << state.participant_id);
            return false;
        }

        try
        {
            if (!expert_consumer
                     .discardCPUCurrentBatchLLEPTransientResidency(state))
            {
                throw std::runtime_error(
                    "transient prepared engines could not be discarded");
            }
            state.active = false;
            active_cpu_llep_state_ = nullptr;
            active_cpu_llep_expert_consumer_ = nullptr;
            active_cpu_llep_tp_ctx_ = nullptr;
            return true;
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[DGO] CPU current-batch LLEP restore failed fatally"
                      << " layer=" << state.layer_idx
                      << " participant=" << state.participant_id
                      << " error=" << e.what());
            return false;
        }
    }

    ExpertTransferResult DeviceGraphOrchestrator::transferReplicaWeights(
        const ExpertReplicaSet &replicas)
    {
        if (replicas.num_replicated == 0 || !mpi_ctx_)
            return {};

        int my_rank = mpi_ctx_->rank();
        int world_size = mpi_ctx_->world_size();
        MPI_Comm comm = mpi_ctx_->communicator();

        std::vector<ExpertMigration> manifest;
        for (int layer_idx = 0;
             layer_idx < replicas.base_ownership.layerCount();
             ++layer_idx)
        {
            for (int expert_id = 0;
                 expert_id < replicas.base_ownership.expertCount();
                 ++expert_id)
            {
                const int owner =
                    replicas.base_ownership.owner(layer_idx, expert_id);
                for (int participant = 0; participant < world_size; ++participant)
                {
                    if (participant == owner ||
                        !replicas.hasReplicaOnParticipant(
                            layer_idx, expert_id, participant))
                    {
                        continue;
                    }
                    manifest.push_back({
                        .layer_idx = layer_idx,
                        .expert_id = expert_id,
                        .src_rank = owner,
                        .dst_rank = participant,
                    });
                }
            }
        }

        if (manifest.empty())
            return {};

        LOG_DEBUG("[DGO] Transferring " << replicas.num_replicated
                                        << " exact layer/expert replica slots via MPI");

        // Collect MoE stages by layer
        std::unordered_map<int, MoEExpertComputeStage *> moe_by_layer;
        if (forward_engine_)
        {
            forward_engine_->forEachCachedStage(
                ComputeStageType::MOE_EXPERT_FFN,
                [&](IComputeStage *stage)
                {
                    auto *moe_stage = dynamic_cast<MoEExpertComputeStage *>(stage);
                    if (moe_stage && moe_stage->layerIndex() >= 0)
                        moe_by_layer[moe_stage->layerIndex()] = moe_stage;
                });
        }

        ExpertTransferEvidence evidence;
        ExpertTransferResult result;
        if (primaryDeviceId().is_cpu())
        {
            if (!frozen_weight_set_)
                throw std::logic_error("CPU replica transfer requires its frozen persistent weight owner");
            auto get_weights =
                [&](int layer_idx, int expert_id) -> ExpertPackedWeights
            {
                auto it = moe_by_layer.find(layer_idx);
                if (it == moe_by_layer.end())
                {
                    throw std::runtime_error(
                        "Replica transfer cannot find the exact source layer stage");
                }
                return it->second->clonePreparedExpert(expert_id);
            };
            auto make_engine =
                [](std::unique_ptr<IPackedWeights> weights)
                    -> std::shared_ptr<ITensorGemm>
            {
                return llaminar::v2::kernels::KernelFactory::
                    createExpertGemmFromPackedWeights(std::move(weights));
            };
            result.prepared =
                ExpertWeightTransfer::transferLayeredPreparedCPU(
                    manifest,
                    get_weights,
                    make_engine,
                    my_rank,
                    cpuBackendNUMANode(),
                    comm,
                    &evidence,
                    CPUWeightStoragePlacement::accounted(physical_memory_authority_, primaryDeviceId(),
                        frozen_weight_set_->physicalMemoryOwner(),
                        CPUWeightStoragePlacement::onNode(cpuBackendNUMANode())));
        }
        else
        {
            // Cross-backend conversion explicitly owns a serialized source
            // representation. It is not selected for homogeneous CPU movement.
            auto get_blobs =
                [&](int layer_idx, int expert_id) -> ExpertWeightBlobs
            {
                auto it = moe_by_layer.find(layer_idx);
                if (it == moe_by_layer.end())
                {
                    throw std::runtime_error(
                        "Replica transfer cannot find the exact source layer stage");
                }
                return it->second->serializeExpert(expert_id);
            };
            result.serialized = ExpertWeightTransfer::transferLayered(
                manifest, get_blobs, my_rank, comm, &evidence,
                CPUWeightStoragePlacement::accounted(physical_memory_authority_, DeviceId::cpu(),
                    PhysicalMemoryOwner::ExpertMigrationStaging,
                    CPUWeightStoragePlacement::local()));
        }

        const std::string device = primaryDeviceId().is_valid()
                                       ? primaryDeviceId().toString()
                                       : std::string{};
        const PerfStatsCollector::Tags tags{
            {"domain_id", moe_rebalance_controller_
                              ? moe_rebalance_controller_->domainId()
                              : std::string("default")},
            {"rank", std::to_string(my_rank)},
            {"kind", "replica"}};
        const auto record_phase = [&](const char *name, uint64_t duration_ns)
        {
            PerfStatsCollector::recordTimingNs(
                "moe_rebalance", name, duration_ns,
                "rebalance", device, tags);
        };
        record_phase(
            "weight_transfer_payload_prepare", evidence.payload_prepare_ns);
        record_phase(
            "weight_transfer_metadata_exchange", evidence.metadata_exchange_ns);
        record_phase(
            "weight_transfer_receive_allocation", evidence.receive_allocation_ns);
        record_phase(
            "weight_transfer_payload_exchange", evidence.payload_exchange_ns);
        record_phase(
            "weight_transfer_result_publication", evidence.result_publication_ns);
        record_phase("weight_transfer_total", evidence.total_ns);
        /*
         * Replica waves use the same packed MPI protocol as ownership waves,
         * so they must publish the same byte/entry evidence.  Omitting these
         * counters made a successful hot-replica transfer indistinguishable
         * from a policy-only mask update even though the phase timers ran and
         * the receiver installed final NUMA-local packed weights.
         */
        PerfStatsCollector::addCounter(
            "moe_rebalance", "weight_transfer_manifest_entries",
            static_cast<double>(evidence.manifest_entries),
            "rebalance", device, tags);
        PerfStatsCollector::addCounter(
            "moe_rebalance", "weight_transfer_outgoing_entries",
            static_cast<double>(evidence.outgoing_entries),
            "rebalance", device, tags);
        PerfStatsCollector::addCounter(
            "moe_rebalance", "weight_transfer_incoming_entries",
            static_cast<double>(evidence.incoming_entries),
            "rebalance", device, tags);
        PerfStatsCollector::addCounter(
            "moe_rebalance", "weight_transfer_outgoing_bytes",
            static_cast<double>(evidence.outgoing_bytes),
            "rebalance", device, tags);
        PerfStatsCollector::addCounter(
            "moe_rebalance", "weight_transfer_incoming_bytes",
            static_cast<double>(evidence.incoming_bytes),
            "rebalance", device, tags);
        return result;
    }

    // =========================================================================
    // Phase-Aware Weight Access (Gap 3 - CPU Decode Participation)
    // =========================================================================

    void DeviceGraphOrchestrator::setWeightManager(std::shared_ptr<IWeightManager> weight_manager)
    {
        weight_manager_ = std::move(weight_manager);
        LOG_DEBUG("[DeviceGraphOrchestrator] WeightManager set");
    }

    void DeviceGraphOrchestrator::setWeightPlacementMap(std::shared_ptr<IWeightPlacementMap> placement_map)
    {
        weight_placement_map_ = std::move(placement_map);
        LOG_DEBUG("[DeviceGraphOrchestrator] WeightPlacementMap set");
    }

    // =========================================================================
    // Tensor Parallel Configuration (Phase 1c: Proportional TP)
    // =========================================================================

    void DeviceGraphOrchestrator::setTensorParallelConfig(std::shared_ptr<TensorParallelConfig> config)
    {
        tp_config_ = std::move(config);

        if (tp_config_)
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] TensorParallelConfig set: "
                      << "world_size=" << tp_config_->worldSize()
                      << ", proportional=" << (tp_config_->isProportional() ? "yes" : "no"));

            // If we have a graph builder, propagate the config to it
            if (graph_builder_)
            {
                // Note: The graph builder's config is read-only after construction,
                // but we store the tp_config for use in buffer allocation and KV cache creation
                LOG_DEBUG("[DeviceGraphOrchestrator] TensorParallelConfig will be used for buffer sizing");
            }
        }
        else
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] TensorParallelConfig cleared");
        }
    }

    // =========================================================================
    // Multi-Domain Tensor Parallel Configuration (Phase 6.3: Heterogeneous TP)
    // =========================================================================

    void DeviceGraphOrchestrator::setDomainConfig(std::shared_ptr<MultiDomainTPConfig> config)
    {
        domain_config_ = std::move(config);

        if (domain_config_)
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] MultiDomainTPConfig set: "
                      << "domains=" << domain_config_->domains().size()
                      << ", has_gpu=" << (domain_config_->gpuDomain() ? "yes" : "no")
                      << ", has_cpu=" << (domain_config_->cpuDomain() ? "yes" : "no")
                      << ", cross_rank=" << (domain_config_->hasCrossRankTP() ? "yes" : "no"));

            // Propagate domain config to graph builder if present
            if (graph_builder_)
            {
                // Graph builder can access domain config through config_.multi_domain_tp_config
                // Note: The graph builder uses getDomainForLayer() which delegates to this config
                LOG_DEBUG("[DeviceGraphOrchestrator] Domain config available for AllreduceStage routing");
            }
        }
        else
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] MultiDomainTPConfig cleared (legacy MPI path)");
        }
    }

    // =========================================================================
    // Pipeline Parallelism Configuration
    // =========================================================================

    void DeviceGraphOrchestrator::setPPStageConfig(const FactoryPPStageConfig &config)
    {
        if (!config.isValid())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid FactoryPPStageConfig: "
                      << "first_layer=" << config.first_layer
                      << ", last_layer=" << config.last_layer);
            throw std::invalid_argument("Invalid FactoryPPStageConfig");
        }

        pp_stage_config_ = config;

        LOG_DEBUG("[DeviceGraphOrchestrator] PP stage configured: "
                  << "layers=[" << config.first_layer << ", " << config.last_layer << ") "
                  << "has_embedding=" << (config.has_embedding ? "yes" : "no")
                  << " has_lm_head=" << (config.has_lm_head ? "yes" : "no"));
    }

    // =========================================================================
    // Hidden State API (for Pipeline Parallelism)
    // =========================================================================

    TensorBase *DeviceGraphOrchestrator::getHiddenState()
    {
        if (!state_.hidden)
        {
            LOG_WARN("[DeviceGraphOrchestrator] getHiddenState: no hidden state available");
            return nullptr;
        }
        return state_.hidden.get();
    }

    const TensorBase *DeviceGraphOrchestrator::getHiddenState() const
    {
        return state_.hidden.get();
    }

    void DeviceGraphOrchestrator::setHiddenState(TensorBase *hidden_state)
    {
        external_hidden_state_input_ = hidden_state;
        LOG_DEBUG("[DeviceGraphOrchestrator] setHiddenState: "
                  << (hidden_state ? "set" : "cleared")
                  << " external hidden state input");
    }

    bool DeviceGraphOrchestrator::hasHiddenStateInput() const
    {
        return external_hidden_state_input_ != nullptr;
    }

    void DeviceGraphOrchestrator::clearHiddenStateInput()
    {
        external_hidden_state_input_ = nullptr;
    }

    const TPDomain *DeviceGraphOrchestrator::getDomainForLayer(int layer_idx, bool is_attention) const
    {
        if (!domain_config_)
        {
            return nullptr; // No domain config - use legacy MPI path
        }

        const TPDomain *domain = domain_config_->domainForLayer(layer_idx, is_attention);

        LOG_DEBUG("[DeviceGraphOrchestrator] getDomainForLayer: layer=" << layer_idx
                                                                        << ", is_attention=" << (is_attention ? "true" : "false")
                                                                        << " -> domain=" << (domain ? domain->name : "nullptr"));

        return domain;
    }

    void DeviceGraphOrchestrator::transitionToPhase(InferencePhase phase)
    {
        if (current_phase_ != phase)
        {
            InferencePhase old_phase = current_phase_;
            current_phase_ = phase;

            LOG_DEBUG("[DeviceGraphOrchestrator] Phase transition: " << toString(old_phase)
                                                                     << " -> " << toString(phase));

            // Notify weight streamer of phase transition (if streaming is enabled)
            if (weight_streamer_)
            {
                weight_streamer_->onPhaseTransition(old_phase, phase);
                LOG_DEBUG("[DeviceGraphOrchestrator] Weight streamer notified of phase transition");
            }
        }
    }

    // =========================================================================
    // Weight Streaming (Option B)
    // =========================================================================

    void DeviceGraphOrchestrator::setWeightStreamer(std::shared_ptr<IWeightStreamer> streamer)
    {
        weight_streamer_ = std::move(streamer);
        if (weight_streamer_)
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] Weight streaming enabled");
        }
        else
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] Weight streaming disabled");
        }
    }

    bool DeviceGraphOrchestrator::isWeightStreamingEnabled() const
    {
        return weight_streamer_ != nullptr;
    }

    void DeviceGraphOrchestrator::setCollectiveContext(std::shared_ptr<ICollectiveContext> collective_ctx)
    {
        injected_collective_ctx_ = std::move(collective_ctx);
        if (injected_collective_ctx_)
        {
            // Wire to executor for GPU-native collective interception
            executor_.setCollectiveContext(injected_collective_ctx_.get());
            LOG_DEBUG("[DeviceGraphOrchestrator] GPU-native collectives enabled via CollectiveContext");
        }
        else
        {
            executor_.setCollectiveContext(nullptr);
            LOG_DEBUG("[DeviceGraphOrchestrator] CollectiveContext cleared - using CPU MPI fallback");
        }
    }

    std::shared_ptr<TensorBase> DeviceGraphOrchestrator::getPhaseAwareWeight(
        const std::string &name,
        int layer_idx,
        InferencePhase phase) const
    {
        if (!weight_manager_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator::getPhaseAwareWeight] WeightManager not set");
            return nullptr;
        }

        // CRITICAL: Use getWeightForDevice() instead of getWeight() to get
        // device-isolated tensor instances. In multi-device (LOCAL TP) scenarios,
        // getWeight() returns the SAME shared tensor to all devices, which causes
        // a race condition: Device 0's ensureOnDevice(rocm:0) allocates GPU memory,
        // then Device 1's ensureOnDevice(rocm:1) frees Device 0's allocation and
        // reallocates on Device 1 — while Device 0's kernels are still using it.
        // getWeightForDevice() returns clones for non-primary devices.
        const DeviceId device = state_.device_id;

        // PREFILL phase: Always use full weight (GPU is primary, compute-bound)
        if (phase == InferencePhase::PREFILL)
        {
            LOG_TRACE("[DeviceGraphOrchestrator::getPhaseAwareWeight] PREFILL phase - returning full weight for " << name
                                                                                                                  << " on " << device.to_string());
            auto weight = weight_manager_->getWeightForDevice(name, device, layer_idx);
            if (!weight)
            {
                LOG_ERROR("[DeviceGraphOrchestrator::getPhaseAwareWeight] Failed to load weight: " << name);
            }
            return weight;
        }

        // DECODE phase: Check if CPU should participate
        if (!shouldUseCPUDecodeWeight(name, layer_idx))
        {
            // No CPU participation - use full weight (GPU handles it)
            LOG_TRACE("[DeviceGraphOrchestrator::getPhaseAwareWeight] DECODE phase, no CPU participation - returning full weight for " << name
                                                                                                                                       << " on " << device.to_string());
            return weight_manager_->getWeightForDevice(name, device, layer_idx);
        }

        // CPU decode participation enabled - get decode shard
        if (!weight_placement_map_)
        {
            // No placement map - fall back to full weight
            LOG_WARN("[DeviceGraphOrchestrator::getPhaseAwareWeight] CPU decode participation but no placement map - using full weight for " << name);
            return weight_manager_->getWeightForDevice(name, device, layer_idx);
        }

        // Get device info from placement map
        WeightDeviceInfo device_info = weight_placement_map_->getDeviceInfoForWeight(name, layer_idx);

        if (!device_info.cpu_decode_participation)
        {
            // This weight doesn't have CPU decode participation
            LOG_TRACE("[DeviceGraphOrchestrator::getPhaseAwareWeight] DECODE phase, weight " << name << " has no CPU participation - returning full weight");
            return weight_manager_->getWeightForDevice(name, device, layer_idx);
        }

        // Find the CPU device in decode_devices and get its fraction
        for (size_t i = 0; i < device_info.decode_devices.size(); ++i)
        {
            if (device_info.decode_devices[i].is_cpu())
            {
                float fraction = device_info.decode_fractions[i];
                LOG_DEBUG("[DeviceGraphOrchestrator::getPhaseAwareWeight] DECODE phase - returning CPU decode shard for "
                          << name << " (fraction=" << fraction << ")");
                return weight_manager_->getDecodeWeight(name, DeviceId::cpu(), fraction, layer_idx);
            }
        }

        // No CPU in decode devices - use full weight
        LOG_TRACE("[DeviceGraphOrchestrator::getPhaseAwareWeight] DECODE phase, CPU not in decode devices - returning full weight for " << name);
        return weight_manager_->getWeightForDevice(name, device, layer_idx);
    }
} // namespace llaminar2
