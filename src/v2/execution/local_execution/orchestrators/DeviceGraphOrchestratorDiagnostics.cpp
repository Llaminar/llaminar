/**
 * @file DeviceGraphOrchestratorDiagnostics.cpp
 * @brief DeviceGraphOrchestrator method shard (55993..60109).
 */
#include "DeviceGraphOrchestratorInternal.h"

namespace llaminar2
{

    bool DeviceGraphOrchestrator::consumeMTPRequestPenaltyPolicyOnDevice(
        const MTPRequestPenaltyPolicy &expected_policy,
        void *consumer_stream,
        const char *consumer)
    {
        if (!state_.device_id.is_gpu() || !consumer_stream || !arena_ ||
            !mtp_greedy_penalty_policy_dev_ ||
            !mtp_request_penalty_policy_published_session_epoch_ ||
            *mtp_request_penalty_policy_published_session_epoch_ !=
                session_epoch_ ||
            expected_policy != mtp_published_request_penalty_policy_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP penalty-policy consumer does not match request admission"
                      << " consumer="
                      << (consumer && consumer[0] != '\0'
                              ? consumer
                              : "unknown")
                      << " session=" << session_epoch_
                      << " published_session="
                      << (mtp_request_penalty_policy_published_session_epoch_
                              ? std::to_string(
                                    *mtp_request_penalty_policy_published_session_epoch_)
                              : "none")
                      << " expected_presence="
                      << expected_policy.presence_penalty
                      << " published_presence="
                      << mtp_published_request_penalty_policy_.presence_penalty
                      << " expected_frequency="
                      << expected_policy.frequency_penalty
                      << " published_frequency="
                      << mtp_published_request_penalty_policy_.frequency_penalty
                      << " expected_repetition=" << expected_policy.repetition_penalty
                      << " published_repetition=" << mtp_published_request_penalty_policy_.repetition_penalty);
            return false;
        }
        if (!arena_->prepareForRead(
                BufferId::MTP_GREEDY_PENALTY_POLICY,
                state_.device_id,
                consumer_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Could not order MTP penalty-policy consumer after its device producer"
                      << " consumer="
                      << (consumer && consumer[0] != '\0'
                              ? consumer
                              : "unknown"));
            return false;
        }
        return true;
    }

    bool DeviceGraphOrchestrator::waitForPendingRequestInputAdmission(
        void *consumer_stream,
        const char *consumer_name)
    {
        if (!state_.device_id.is_gpu())
            return true;
        auto &ready = request_input_admission_ready_;
        if (!ready.valid)
            return true;
        if (!consumer_stream || !ready.event || !ready.producer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] GPU request-input admission has incomplete stream/event ownership");
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;
        const bool same_stream = consumer_stream == ready.producer_stream;
        if (!DeviceEventEdge::at(DeviceTimelinePoint::RequestInputAdmission)
                 .from(DeviceTimelineRole::RequestAdmissionTransfer)
                 .to(DeviceTimelineRole::MainForwardGraph)
                 .enqueueWait(
                     *backend,
                     state_.device_id,
                     ready.event.get(),
                     ready.producer_stream,
                     consumer_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to order graph execution after request admission for consumer="
                      << (consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown"));
            return false;
        }

        const int admitted_token_count = ready.token_count;
        const int admitted_request_count = ready.request_count;
        PerfStatsCollector::addCounter(
            "request_admission",
            "device_input_event_waits",
            1.0,
            "prefill",
            state_.device_id.toString(),
            {{"same_stream", boolTag(same_stream)},
             {"tokens", std::to_string(ready.token_count)},
             {"requests", std::to_string(ready.request_count)}});
        ready.valid = false;
        ready.producer_stream = nullptr;
        ready.token_count = 0;
        ready.request_count = 0;
        beginRequestInputReaderTransaction(
            "host_request_admission",
            admitted_token_count,
            admitted_request_count);
        return true;
    }

    void DeviceGraphOrchestrator::beginRequestInputReaderTransaction(
        const char *admission_kind,
        int token_count,
        int request_count)
    {
        auto &reuse = request_input_reuse_ready_;
        if (!state_.device_id.is_gpu() ||
            !admission_kind ||
            admission_kind[0] == '\0' ||
            token_count <= 0 ||
            request_count <= 0 ||
            reuse.consumers_started ||
            reuse.valid)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Request-input reader transaction "
                "violated the reusable-bank lifecycle"
                << " device=" << state_.device_id.toString()
                << " admission="
                << (admission_kind && admission_kind[0] != '\0'
                        ? admission_kind
                        : "unknown")
                << " tokens=" << token_count
                << " requests=" << request_count
                << " consumers_started="
                << boolTag(reuse.consumers_started)
                << " prior_release_valid=" << boolTag(reuse.valid));
            std::terminate();
        }

        reuse.consumers_started = true;
        PerfStatsCollector::addCounter(
            "request_admission",
            "device_input_reader_admissions",
            1.0,
            "prefill",
            state_.device_id.toString(),
            {{"admission", admission_kind},
             {"tokens", std::to_string(token_count)},
             {"requests", std::to_string(request_count)},
             {"ordering", "writer_to_reader_event"}});
    }

    void DeviceGraphOrchestrator::publishRequestInputReuseReady(
        void *consumer_completion_stream,
        const char *producer_name)
    {
        if (!state_.device_id.is_gpu())
            return;

        auto &reuse = request_input_reuse_ready_;
        if (!consumer_completion_stream ||
            !reuse.event ||
            !reuse.consumers_started ||
            reuse.valid)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Request-input reuse publication violated the reusable-bank lifecycle"
                      << " stream=" << consumer_completion_stream
                      << " event=" << reuse.event.get()
                      << " consumers_started=" << boolTag(reuse.consumers_started)
                      << " prior_release_valid=" << boolTag(reuse.valid));
            std::terminate();
        }

        /*
         * Request-batched shifted prefill does not archive one terminal row after
         * its final sidecar. Join that side stream here without consuming its
         * live-state publication. For the single-request path the final terminal
         * archive already consumed the shifted-KV edge on this same stream, so
         * this observation is a no-op.
         */
        if (!waitForPendingShiftedMTPKVReadyForObservation(
                consumer_completion_stream,
                "request_input_reuse"))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Could not order request-input release after shifted MTP readers");
            std::terminate();
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend ||
            !DeviceEventEdge::at(DeviceTimelinePoint::RequestInputReuseReady)
                 .from(DeviceTimelineRole::MainForwardGraph)
                 .publish(
                     *backend,
                     state_.device_id,
                     reuse.event.get(),
                     consumer_completion_stream))
        {
            /*
             * GPU readers have already launched. Without a completion event the
             * reusable bank has no legal next writer, so continuing inference
             * would turn a backend publication failure into silent stale input.
             */
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to publish request-input reuse readiness");
            std::terminate();
        }

        reuse.producer_stream = consumer_completion_stream;
        reuse.valid = true;
        reuse.consumers_started = false;
        PerfStatsCollector::addCounter(
            "request_admission",
            "device_input_reuse_publications",
            1.0,
            "prefill",
            state_.device_id.toString(),
            {{"producer", producer_name && producer_name[0] != '\0'
                              ? producer_name
                              : "unknown"},
             {"ordering", "transitive_last_reader"}});
    }

    /**
     * @brief Seed the first device-owned request-batch publication after prefill.
     *
     * This is deliberately a device-to-device publication operation. Prompt
     * positions already live in the persistent request-length arena row, sampled
     * token identities live in target-sample arena slots, and every logical-state
     * field is completed before the mailbox event is recorded.
     */
    bool DeviceGraphOrchestrator::initializeDeviceResidentLogicalSequenceStateFromMainBatchSamples(
        int request_count,
        void *producer_stream)
    {
        if (!state_.device_id.is_gpu() ||
            request_count <= 0 ||
            request_count > stochastic_batch_output_request_capacity_ ||
            !producer_stream ||
            !stochastic_target_sample_tokens_dev_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Initial MTP device logical-state publication received invalid request geometry or scratch");
            return false;
        }

        const MTPSpecDecodeMetadataShape metadata_shape =
            mtp_spec_decode_metadata_binding_.capacity();
        if (!mtp_spec_decode_metadata_binding_.hasWorkspace() ||
            metadata_shape.max_requests < request_count)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Initial MTP device logical-state publication requires a pre-bound request-batch metadata workspace");
            return false;
        }

        const MTPSpecDecodeMetadataDevicePointers &workspace_ptrs =
            mtp_spec_decode_metadata_binding_.devicePointers();
        MTPSpecDecodeMetadataDevicePointers publication_ptrs = workspace_ptrs;
        if (!workspace_ptrs.base_cached_tokens ||
            !device_resident_logical_sequence_state_storage_
                 .bindPublicationOutputs(
                     &publication_ptrs,
                     request_count))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Initial MTP device logical-state publication has no complete arena-owned output storage");
            return false;
        }
        if (!request_sequence_lengths_dev_ ||
            request_sequence_lengths_active_count_ < request_count)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Initial MTP device logical-state publication is missing admitted request lengths");
            return false;
        }

        for (int request = 0; request < request_count; ++request)
        {
            if (!waitForRequiredStochasticTargetSampleReady(
                    request,
                    producer_stream,
                    "initialize_mtp_logical_state_after_prefill"))
            {
                return false;
            }
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;
        if (!waitForDeviceResidentLogicalSequenceStateRowReuse(
                producer_stream,
                "request_batch_prefill_logical_state_row_reuse"))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Initial MTP device logical-state publication could not order reusable rows");
            return false;
        }

        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "request_batch_prefill_device_logical_state_enqueue",
                "decode",
                state_.device_id.toString(),
                {{"requests", std::to_string(request_count)}});
            if (!backend->enqueueInitializeGenerationLogicalState({
                    .source = GenerationInitialFrontier::SampledCondition,
                    .positions = static_cast<const int32_t *>(request_sequence_lengths_dev_),
                    .sampled_tokens = static_cast<const int32_t *>(stochastic_target_sample_tokens_dev_),
                    .request_count = request_count,
                    .output = {.base_cached_tokens = workspace_ptrs.base_cached_tokens,
                        .target_positions = publication_ptrs.target_cached_tokens,
                        .accepted_state_counts = publication_ptrs.accepted_state_counts,
                        .next_condition_tokens = publication_ptrs.next_condition_tokens,
                        .all_drafts_accepted_flags = publication_ptrs.all_drafts_accepted_flags,
                        .stopped_flags = publication_ptrs.stopped_flags,
                        .publication_ok_flags = publication_ptrs.publication_ok_flags}},
                    state_.device_id.gpu_ordinal(), producer_stream))
            {
                return false;
            }
        }

        /*
         * Shifted-prefill graph work may be emitted as several economical
         * request-local kernel groups, but the serving transaction that follows
         * owns the complete logical request batch. Promote that earlier fence to
         * the canonical request count before exposing the logical-state mailbox.
         * Waiting on the prior shared transaction first preserves cross-stream
         * ordering; recordDeviceResidentMTPTransactionMutation() can then retain
         * the same shared identity on every subsequent grouped sidecar mutation.
         */
        const DeviceResidentMTPTransactionLease prior_transaction =
            currentDeviceResidentMTPTransactionLease();
        if (prior_transaction.valid() &&
            !waitForDeviceResidentMTPTransaction(
                prior_transaction,
                producer_stream,
                "request_batch_prefill_logical_state_transaction_promotion"))
        {
            return false;
        }
        if (!recordDeviceResidentMTPTransactionMutation(
                request_count,
                producer_stream,
                "request_batch_prefill_logical_state_transaction"))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Initial MTP device logical-state publication could not establish the canonical request-batch transaction");
            return false;
        }

        std::string mailbox_error;
        if (!recordDeviceResidentLogicalSequenceStateMailbox(
                request_count,
                producer_stream,
                DeviceResidentLogicalStatePublicationKind::MainBatchSampleInitialization,
                &mailbox_error))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Initial MTP device logical-state mailbox publication failed: "
                      << mailbox_error);
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "request_batch_prefill_device_mtp_transaction_publications",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"requests", std::to_string(request_count)},
             {"transaction_owner", "canonical_shifted_kv_counts"},
             {"logical_state_owner", "arena_persistent"}});

        PerfStatsCollector::addCounter(
            "mtp",
            "request_batch_prefill_device_logical_state_publications",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"requests", std::to_string(request_count)},
             {"logical_state_owner", "arena_persistent"},
             {"position_owner", "arena_persistent"}});
        return true;
    }

    bool DeviceGraphOrchestrator::publishMainLogitsBatchSamplesToDeviceResidentState(
        int request_count,
        const SamplingParams &params,
        const uint64_t *stochastic_position_seeds)
    {
        if (request_count <= 0)
        {
            return false;
        }
        if (!state_.device_id.is_gpu())
        {
            return false;
        }
        TensorBase *logits_tensor = requireCurrentMainLogits(
            DeviceLogitsSource::MainRequestBatch,
            "publishMainLogitsBatchSamplesToDeviceResidentState");
        if (current_main_logits_publication_.descriptor
                .isColumnParallelStorage())
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] Request-batched main-logits "
                      "sampling is not enabled for column-parallel LM heads");
            return false;
        }

        if (!logits_tensor || !logits_tensor->deviceValid())
        {
            return false;
        }

        void *gpu_ptr = logits_tensor->gpu_data_ptr();
        if (!gpu_ptr)
        {
            return false;
        }

        const auto &shape = logits_tensor->shape();
        if (shape.empty())
        {
            return false;
        }

        /*
         * Batched prefill stores exactly one terminal logits row per request.
         * This differs from the all-position verifier tensor: there is no
         * padded sequence dimension to skip here.
         */
        const size_t rows = shape.size() >= 2 ? shape[0] : 1;
        const size_t cols = shape.size() >= 2 ? shape[1] : shape[0];
        if (rows < static_cast<size_t>(request_count) ||
            cols == 0 ||
            cols > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] Request-batched logits shape "
                      "cannot satisfy sampling request: rows=" << rows
                      << " cols=" << cols
                      << " request_count=" << request_count);
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
        {
            return false;
        }

        const PendingLogitsStreamRole stream_role =
            current_main_logits_publication_.descriptor.storage_surface ==
                    ForwardLogitsStorageSurface::AllPositionFull ||
                current_main_logits_publication_.descriptor.storage_surface ==
                    ForwardLogitsStorageSurface::AllPositionLocal
                ? PendingLogitsStreamRole::AllPositionVerifier
                : PendingLogitsStreamRole::MainDecode;
        void *stream = peekPendingLogitsStream(stream_role);
        if (!stream)
            stream = explicitGPUStreamForOperation(
                "publishMainLogitsBatchSamplesToDeviceResidentState");
        if (!stream)
        {
            return false;
        }
        if (!waitForForwardGraphOutputReady(
                stream,
                DeviceTimelineRole::TargetSampler,
                ForwardGraphOutputKind::Any,
                "publishMainLogitsBatchSamplesToDeviceResidentState"))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Request-batched main-logits "
                      "sampling could not join ForwardGraphOutputReady");
            return false;
        }

        auto *base = static_cast<const float *>(gpu_ptr);
        const int vocab_cols = static_cast<int>(cols);
        if (!stochastic_target_sample_tokens_dev_ ||
            !stochastic_verify_accept_probs_dev_ ||
            !argmax_partial_vals_dev_ ||
            !argmax_partial_idxs_dev_ ||
            argmax_partial_capacity_ <= 0 ||
            request_count > stochastic_target_row_capacity_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Request-batched prefill sampling is missing persistent device scratch");
            return false;
        }

        if (params.is_greedy())
        {
            for (int row = 0; row < request_count; ++row)
            {
                clearStochasticTargetSampleReadySlot(
                    row,
                    StochasticSampleReadyClearMode::Force);
            }

            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "request_batch_prefill_argmax_enqueue",
                    "decode",
                    state_.device_id.toString(),
                    {{"requests", std::to_string(request_count)},
                     {"vocab", std::to_string(vocab_cols)}});
                if (!backend->enqueueArgmaxF32BatchedRowsDevice(
                        base,
                        request_count,
                        vocab_cols,
                        state_.device_id.gpu_ordinal(),
                        stream,
                        stochastic_verify_accept_probs_dev_,
                        stochastic_target_sample_tokens_dev_,
                        argmax_partial_vals_dev_,
                        argmax_partial_idxs_dev_,
                        argmax_partial_capacity_))
                {
                    return false;
                }
            }

            for (int row = 0; row < request_count; ++row)
            {
                if (!recordStochasticTargetSampleReady(
                        row,
                        stream,
                        /*verifier_consumer_pending=*/true, StochasticTargetPublication::NewValue))
                {
                    return false;
                }
            }
        }
        else
        {
            if (params.top_k <= 0 || params.top_k > 256)
            {
                LOG_DEBUG("[DeviceGraphOrchestrator] GPU stochastic batch-row "
                          "sampling requires 1 <= top_k <= 256; got top_k="
                          << params.top_k);
                return false;
            }
            if (!stochastic_position_seeds)
            {
                LOG_DEBUG("[DeviceGraphOrchestrator] GPU stochastic batch-row "
                          "sampling requires immutable per-request position seeds");
                return false;
            }
            if (logits_tensor != state_.all_position_logits.get() ||
                request_batched_prefill_logits_row_count_ < request_count)
            {
                LOG_DEBUG("[DeviceGraphOrchestrator] GPU stochastic batch-row "
                          "sampling requires compact request-batched prefill logits");
                return false;
            }
            if (!supportsDeviceStochasticMTPVerification())
            {
                LOG_DEBUG("[DeviceGraphOrchestrator] GPU stochastic batch-row "
                          "sampling requires compact device distribution support");
                return false;
            }

            if (!request_sequence_lengths_dev_ ||
                request_sequence_lengths_active_count_ < request_count)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] GPU stochastic prefill sampling is missing resident logical positions");
                return false;
            }
            for (int row = 0; row < request_count; ++row)
            {
                if (stochastic_position_seeds[row] == 0)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] GPU stochastic prefill sampling received an unresolved zero seed");
                    return false;
                }
            }

            /*
             * Build all request distributions in one launch, then leave every
             * sampled token in its target slot. The first grouped verifier and
             * the generation controller consume these slots directly; there is
             * no host response boundary between prefill and transaction zero.
             */
            if (!buildStochasticDistributionsOnDevice(
                    DeviceLogitsSource::AllPosition,
                    /*first_row=*/0,
                    DeviceDistributionBuffer::Target,
                    /*first_slot=*/0,
                    request_count,
                    params,
                    vocab_cols))
            {
                return false;
            }
            for (int row = 0; row < request_count; ++row)
            {
                if (!sampleStochasticDistributionOnDeviceImpl(
                        DeviceDistributionBuffer::Target,
                        row,
                        /*threshold=*/0.0f,
                        /*out_token_host=*/nullptr,
                        stochastic_position_seeds[row],
                        static_cast<const int32_t *>(request_sequence_lengths_dev_) + row,
                        /*threshold_position_offset=*/0))
                {
                    return false;
                }
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "request_batch_prefill_resident_position_threshold_rows",
                static_cast<double>(request_count),
                "decode",
                state_.device_id.toString(),
                {{"position_owner", "arena_request_sequence_lengths"}});
        }

        if (!initializeDeviceResidentLogicalSequenceStateFromMainBatchSamples(
                request_count,
                stream))
        {
            return false;
        }

        drainPendingGpuTimingMeasurements(backend);
        (void)consumePendingLogitsStream(
            stream_role,
            "publishMainLogitsBatchSamplesToDeviceResidentStateComplete");
        request_batched_prefill_logits_row_count_ = 0;
        PerfStatsCollector::addCounter(
            "mtp",
            "request_batch_prefill_samples_published_device_resident",
            static_cast<double>(request_count),
            "decode",
            state_.device_id.toString(),
            {{"host_token_materializations", "0"},
             {"consumer", "first_grouped_verifier_transaction"}});
        return true;
    }

    /**
     * @brief Apply sparse penalties to the exact published main-logits row.
     * @param penalties Global token IDs and additive logit penalties.
     * @param vocab_size Complete model vocabulary size.
     * @return True when every relevant penalty has been applied.
     * @throws std::logic_error if there is no current main-logits publication.
     *
     * GPU graph replay can leave logits pending on a capture stream. Penalty
     * application is an in-place mutation of those logits, so it is also a
     * logits producer. Republish its exact stream through the typed handoff for
     * the next sampler. CPU execution has no pending device edge, but must
     * select the same published tensor and global-token offset as the sampler.
     */
    bool DeviceGraphOrchestrator::applyPenaltiesOnDevice(const LogitPenaltyBatch &penalties,
                                                         int vocab_size)
    {
        if (penalties.empty())
            return true; // Nothing to apply, success

        TensorBase *main_logits = requireCurrentMainLogits(
            DeviceLogitsSource::Main, "applyPenaltiesOnDevice");
        const bool column_parallel =
            current_main_logits_publication_.descriptor.isColumnParallelStorage();
        if (!state_.device_id.is_gpu())
        {
            const int token_offset = column_parallel
                ? vocabOffsetForTPConfig(graph_builder_->config()) : 0;
            return applyPenaltiesToTensorRowOnHost(
                main_logits,
                penalties,
                vocab_size,
                0,
                token_offset,
                "applyPenaltiesOnDeviceHost");
        }

        void *stream = prepareMainLogitsDeviceConsumer(
            "applyPenaltiesOnDevice",
            MainLogitsHandoffMode::Observe);
        if (!stream)
            return false;

        bool ok = false;
        if (column_parallel)
        {
            const int token_offset = vocabOffsetForTPConfig(graph_builder_->config());
            ok = applyPenaltiesToTensorRowOnDevice(
                main_logits,
                state_.device_id,
                penalties,
                vocab_size,
                0,
                token_offset,
                stream,
                "applyPenaltiesOnDeviceLocal");
        }
        else
        {
            ok = applyPenaltiesToTensorRowOnDevice(
                main_logits,
                state_.device_id,
                penalties,
                vocab_size,
                0,
                0,
                stream,
                "applyPenaltiesOnDevice");
        }

        if (ok)
        {
            /*
             * The sampler owns clearing this pending stream. We intentionally
             * leave it set even if it came from explicitGPUStreamForOperation()
             * because the penalty kernel is now the newest logits writer.
             */
            publishPendingLogitsStream(
                PendingLogitsStreamRole::MainDecode,
                stream,
                "applyPenaltiesOnDevice");
        }
        return ok;
    }

    /**
     * @brief Apply sparse penalties to MTP sidecar logits on the logits stream.
     *
     * Sidecar logits often come from a captured MTP graph. The penalty kernel
     * must stay ordered with that graph replay and with the following draft
     * sampler, so this method republishes the stream for the next MTP logits
     * consumer instead of switching to an unrelated operation stream.
     */
    bool DeviceGraphOrchestrator::applyPenaltiesToMTPLogitsOnDevice(
        const LogitPenaltyBatch &penalties,
        int vocab_size)
    {
        if (penalties.empty())
            return true;

        auto it = state_.extension_buffers.find(mtpSamplingLogitsBufferId());
        if (it == state_.extension_buffers.end() || !it->second)
            return false;

        const int token_offset = mtpSidecarLogitsAreColumnParallel() && graph_builder_
                                     ? vocabOffsetForTPConfig(graph_builder_->config())
                                     : 0;
        if (!state_.device_id.is_gpu())
        {
            return applyPenaltiesToTensorRowOnHost(
                it->second.get(),
                penalties,
                vocab_size,
                0,
                token_offset,
                "applyPenaltiesToMTPLogitsOnDeviceHost");
        }

        void *stream = peekPendingLogitsStream(PendingLogitsStreamRole::MTPSidecar);
        if (!stream)
        {
            stream = explicitGPUStreamForOperation("applyPenaltiesToMTPLogitsOnDevice");
        }
        if (!stream)
            return false;

        const bool ok = applyPenaltiesToTensorRowOnDevice(
            it->second.get(),
            state_.device_id,
            penalties,
            vocab_size,
            0,
            token_offset,
            stream,
            "applyPenaltiesToMTPLogitsOnDevice");
        if (ok)
        {
            publishPendingLogitsStream(
                PendingLogitsStreamRole::MTPSidecar,
                stream,
                "applyPenaltiesToMTPLogitsOnDevice");
        }
        return ok;
    }

    /**
     * @brief Apply sparse penalties to one all-position verifier logits row.
     *
     * All-position verifier replay may build several target distributions from
     * one captured logits buffer. Penalties are row-local mutations of that
     * buffer, so they must be enqueued on the all-position producer stream and
     * leave that stream available for the next distribution build.
     */
    bool DeviceGraphOrchestrator::applyPenaltiesToAllPositionLogitsOnDeviceRow(
        int row,
        const LogitPenaltyBatch &penalties,
        int vocab_size)
    {
        if (penalties.empty())
            return true;
        if (row < 0)
            return false;

        TensorBase *tensor = nullptr;
        int token_offset = 0;
        if (activeAllPositionLogitsAreColumnParallel())
        {
            /*
             * Column-parallel LocalTP keeps two useful all-position views:
             * a local LM-head shard for cross-shard sampling, and sometimes a
             * gathered full-vocab tensor for diagnostics or host consumers.
             *
             * Penalties are part of the sampled distribution, so they must be
             * applied to the same producer buffer that LogitsLocalInfo exposes
             * to RankOrchestrator::sampleGreedyFromAllPositionLogitsOnDeviceRows().
             * Mutating the gathered tensor here would leave the local shard
             * unchanged and let the verifier sample stale row logits.
             */
            tensor = state_.all_position_logits_local.get();
            token_offset = localLogitsVocabOffset();
        }
        else if (state_.all_position_logits)
        {
            tensor = state_.all_position_logits.get();
        }
        if (!tensor)
            return false;

        if (!state_.device_id.is_gpu())
        {
            return applyPenaltiesToTensorRowOnHost(
                tensor,
                penalties,
                vocab_size,
                row,
                token_offset,
                "applyPenaltiesToAllPositionLogitsOnDeviceRowHost");
        }

        void *stream = peekPendingLogitsStream(PendingLogitsStreamRole::AllPositionVerifier);
        if (!stream)
        {
            stream = explicitGPUStreamForOperation(
                "applyPenaltiesToAllPositionLogitsOnDeviceRow");
        }
        if (!stream)
            return false;

        const bool ok = applyPenaltiesToTensorRowOnDevice(
            tensor,
            state_.device_id,
            penalties,
            vocab_size,
            row,
            token_offset,
            stream,
            "applyPenaltiesToAllPositionLogitsOnDeviceRow");
        if (ok)
        {
            publishPendingLogitsStream(
                PendingLogitsStreamRole::AllPositionVerifier,
                stream,
                "applyPenaltiesToAllPositionLogitsOnDeviceRow");
            if (!recordAllPositionVerifierStateReady(
                    stream,
                    "applyPenaltiesToAllPositionLogitsOnDeviceRow"))
            {
                throw std::runtime_error(
                    "Failed to publish all-position logits penalty readiness");
            }
        }
        return ok;
    }

    bool DeviceGraphOrchestrator::applyDeviceOwnedMTPPenaltiesToLogitRows(
        DeviceLogitsSource source,
        int row_count,
        const MTPRequestPenaltyPolicy &penalty_policy)
    {
        if (!state_.device_id.is_gpu() || row_count <= 0 ||
            (source != DeviceLogitsSource::Main &&
             source != DeviceLogitsSource::AllPosition))
        {
            return false;
        }
        if (!penalty_policy.enabled())
            return true;
        if (!arena_ || !mtp_generated_token_counts_dev_ ||
            !mtp_greedy_penalty_policy_dev_ ||
            mtp_generated_token_count_capacity_ != state_.vocab_size)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-owned MTP penalty state is not fully bound");
            return false;
        }

        const bool verifier_rows = source == DeviceLogitsSource::AllPosition;
        void *stream = verifier_rows
                           ? peekPendingLogitsStream(
                                 PendingLogitsStreamRole::AllPositionVerifier)
                           : prepareMainLogitsDeviceConsumer(
                                 "applyDeviceOwnedMTPPenaltiesToLogitRows",
                                 MainLogitsHandoffMode::Observe,
                                 /*require_main_forward=*/true);
        if (verifier_rows && !stream)
        {
            stream = explicitGPUStreamForOperation(
                "applyDeviceOwnedMTPPenaltiesToLogitRows");
        }
        if (!stream)
            return false;

        if (!consumeMTPRequestPenaltyPolicyOnDevice(
                penalty_policy,
                stream,
                "applyDeviceOwnedMTPPenaltiesToLogitRows"))
        {
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        TensorBase *tensor = verifier_rows
                                 ? state_.all_position_logits.get()
                                 : requireCurrentMainLogits(
                                       DeviceLogitsSource::Main,
                                       "applyDeviceOwnedMTPPenaltiesToLogitRows");
        if (!backend || !tensor || !tensor->deviceValid() ||
            !tensor->gpu_data_ptr())
        {
            return false;
        }
        const auto &shape = tensor->shape();
        if (shape.size() < 2 ||
            row_count > static_cast<int>(shape[0]) ||
            shape[1] != static_cast<size_t>(state_.vocab_size))
        {
            return false;
        }

        if (!arena_->prepareForRead(
                BufferId::MTP_GENERATED_TOKEN_COUNTS,
                state_.device_id,
                stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Could not order MTP penalty rows after generated-token history publication");
            return false;
        }
        const void *verifier_input_tokens = nullptr;
        if (verifier_rows)
        {
            if (!mtp_verifier_input_tokens_dev_ ||
                !arena_->prepareForRead(
                    BufferId::MTP_VERIFIER_INPUT_TOKENS,
                    state_.device_id,
                    stream))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Could not order MTP penalty rows after verifier-token publication");
                return false;
            }
            verifier_input_tokens = mtp_verifier_input_tokens_dev_;
        }

        if (!backend->enqueueApplyMTPPenaltiesToF32RowsDevice(
                tensor->gpu_data_ptr(),
                row_count,
                state_.vocab_size,
                state_.vocab_size,
                verifier_input_tokens,
                mtp_generated_token_counts_dev_,
                mtp_greedy_penalty_policy_dev_,
                state_.device_id.gpu_ordinal(),
                stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-owned MTP penalty row transform failed");
            return false;
        }

        const PendingLogitsStreamRole role =
            verifier_rows ? PendingLogitsStreamRole::AllPositionVerifier
                          : PendingLogitsStreamRole::MainDecode;
        publishPendingLogitsStream(
            role,
            stream,
            "applyDeviceOwnedMTPPenaltiesToLogitRows");
        if (verifier_rows &&
            !recordAllPositionVerifierStateReady(
                stream,
                "applyDeviceOwnedMTPPenaltiesToLogitRows"))
        {
            throw std::runtime_error(
                "Failed to publish device-owned MTP penalty-row readiness");
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "device_owned_mtp_penalty_logit_rows",
            static_cast<double>(row_count),
            "decode",
            state_.device_id.toString(),
            {{"source", verifier_rows ? "all_position" : "main"},
             {"history_owner", "device_generated_token_histogram"},
             {"verifier_prefix", verifier_rows ? "device_row" : "none"}});
        return true;
    }

    bool DeviceGraphOrchestrator::applyDeviceOwnedMTPBranchPenaltiesToLogits(
        int prior_draft_count,
        const MTPRequestPenaltyPolicy &penalty_policy)
    {
        if (!state_.device_id.is_gpu() || prior_draft_count < 0 ||
            prior_draft_count > mtp_max_draft_depth_)
        {
            return false;
        }
        if (!penalty_policy.enabled())
            return true;
        if (!arena_ || !mtp_generated_token_counts_dev_ ||
            !mtp_greedy_penalty_policy_dev_ ||
            !mtp_sidecar_condition_token_dev_ ||
            !stochastic_draft_sample_tokens_dev_ ||
            mtp_generated_token_count_capacity_ != state_.vocab_size)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-owned MTP branch penalty state is not fully bound");
            return false;
        }

        /*
         * A branch transform is a direct consumer of the current sidecar row.
         * Requiring its published stream makes a missing sidecar->penalty edge
         * a hard error instead of silently selecting an unrelated stream.
         */
        void *stream = peekPendingLogitsStream(
            PendingLogitsStreamRole::MTPSidecar);
        if (!stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP branch penalties require a published sidecar logits stream");
            return false;
        }
        if (!consumeMTPRequestPenaltyPolicyOnDevice(
                penalty_policy,
                stream,
                "applyDeviceOwnedMTPBranchPenaltiesToLogits"))
        {
            return false;
        }
        if (prior_draft_count > 0 &&
            !waitForRequiredStochasticDraftSampleReadyRange(
                /*first_slot=*/0,
                prior_draft_count,
                stream,
                "mtp_branch_penalty_history"))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Could not order MTP branch penalties after prior draft samples");
            return false;
        }
        if (!arena_->prepareForRead(
                BufferId::MTP_GENERATED_TOKEN_COUNTS,
                state_.device_id,
                stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Could not order MTP branch penalties after durable history publication");
            return false;
        }

        auto logits_it = state_.extension_buffers.find(mtpSamplingLogitsBufferId());
        TensorBase *mtp_logits =
            logits_it == state_.extension_buffers.end()
                ? nullptr
                : logits_it->second.get();
        if (!mtp_logits || !mtp_logits->deviceValid() ||
            !mtp_logits->gpu_data_ptr())
        {
            return false;
        }
        const auto &shape = mtp_logits->shape();
        if (shape.empty())
            return false;
        const size_t rows = shape.size() >= 2 ? shape[0] : 1;
        const size_t cols = shape.size() >= 2 ? shape[1] : shape[0];
        if (rows < 1 || cols != static_cast<size_t>(state_.vocab_size))
            return false;

        const int full_condition_slot =
            mtp_sidecar_capture_layout_.conditionTokenSlot(
                MTPSidecarCaptureRole::Full,
                /*total_rows=*/1);
        const int condition_offset =
            full_condition_slot * mtp_sidecar_condition_token_slot_width_;
        if (condition_offset < 0 ||
            condition_offset >= mtp_sidecar_condition_token_capacity_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Full-sidecar condition token is outside its persistent slot");
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend ||
            !backend->enqueueApplyMTPBranchPenaltiesToF32RowDevice(
                mtp_logits->gpu_data_ptr(),
                state_.vocab_size,
                static_cast<const int32_t *>(
                    mtp_sidecar_condition_token_dev_) + condition_offset,
                stochastic_draft_sample_tokens_dev_,
                prior_draft_count,
                mtp_generated_token_counts_dev_,
                mtp_greedy_penalty_policy_dev_,
                state_.device_id.gpu_ordinal(),
                stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-owned MTP branch penalty transform failed");
            return false;
        }

        publishPendingLogitsStream(
            PendingLogitsStreamRole::MTPSidecar,
            stream,
            "applyDeviceOwnedMTPBranchPenaltiesToLogits");
        PerfStatsCollector::addCounter(
            "mtp",
            "device_owned_mtp_branch_penalty_logit_rows",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"prior_drafts", std::to_string(prior_draft_count)},
             {"first_condition_owner", "device_penalty_policy"},
             {"history_owner", "device_branch"}});
        return true;
    }

    bool DeviceGraphOrchestrator::supportsRowLocalAllPositionPenaltyApplication() const
    {
        /*
         * CPU all-position rows are native host tensors, and the method above
         * applies the same sparse row-local penalty map directly to that row.
         * GPU rows use the explicit-stream sparse penalty kernel.  Both are
         * grouped-verifier implementations: the caller mutates each verifier
         * row in place, then samples/reduces the grouped outcome without
         * replaying rows through the serial decode loop.
         */
        return state_.device_id.is_cpu() || state_.device_id.is_gpu();
    }

    bool DeviceGraphOrchestrator::supportsDeviceStochasticMTPVerification() const
    {
        if (!state_.device_id.is_gpu())
            return false;
        /*
         * Ordinary column-parallel logits do not contain a full distribution on
         * one participant, so the stochastic verifier must consume either a
         * complete head or the full publication from an explicit native gather.
         * Projection weights remain sharded when the latter policy is active.
         */
        if (graph_builder_ &&
            graph_builder_->config().lm_head_column_parallel &&
            !publishesFullVocabularyMTPLogits())
        {
            return false;
        }
        return stochastic_target_token_ids_dev_ &&
               stochastic_target_probs_dev_ &&
               stochastic_draft_token_ids_dev_ &&
               stochastic_draft_probs_dev_ &&
               stochastic_processed_logits_dev_ &&
               stochastic_inverse_rejection_samples_dev_ &&
               stochastic_target_sample_tokens_dev_ &&
               stochastic_draft_sample_tokens_dev_ &&
               stochastic_draft_sample_probs_dev_ &&
               stochastic_topk_partial_vals_dev_ &&
               stochastic_topk_partial_idxs_dev_ &&
               stochastic_topk_partial_capacity_ > 0 &&
               stochastic_draft_topk_partial_vals_dev_ &&
               stochastic_draft_topk_partial_idxs_dev_ &&
               stochastic_draft_topk_partial_capacity_ > 0 &&
               argmax_partial_vals_dev_ &&
               argmax_partial_idxs_dev_ &&
               argmax_partial_capacity_ > 0 &&
               mtp_verifier_input_tokens_dev_ &&
               stochastic_verify_tokens_dev_ &&
               stochastic_verify_accepted_dev_ &&
               stochastic_verify_accept_probs_dev_ &&
               stochastic_verify_thresholds_dev_ &&
               stochastic_batch_output_tokens_dev_ &&
               stochastic_batch_output_meta_dev_;
    }

    bool DeviceGraphOrchestrator::buildStochasticProcessedLogitRowsOnDevice(
        DeviceLogitsSource source,
        int first_row,
        DeviceDistributionBuffer buffer,
        int first_slot,
        int row_count,
        const SamplingParams &params,
        int vocab_size,
        void *stream,
        const char *operation_name)
    {
        if (!supportsDeviceStochasticMTPVerification() ||
            !stochastic_processed_logits_dev_ ||
            first_row < 0 || first_slot < 0 || row_count <= 0 ||
            vocab_size <= 0 || params.top_k <= 0 ||
            stream == nullptr)
        {
            return false;
        }

        const int max_slots =
            buffer == DeviceDistributionBuffer::Target
                ? stochastic_target_row_capacity_
                : stochastic_draft_row_capacity_;
        if (first_slot + row_count > max_slots ||
            row_count > stochastic_target_row_capacity_)
        {
            return false;
        }

        TensorBase *tensor = resolveDeviceLogitsTensor(
            source,
            operation_name ? operation_name
                           : "buildStochasticProcessedLogitRowsOnDevice");

        if (!tensor || !tensor->deviceValid() || !tensor->gpu_data_ptr())
            return false;

        const auto &shape = tensor->shape();
        if (shape.empty())
            return false;
        const size_t rows = shape.size() >= 2 ? shape[0] : 1;
        const size_t cols = shape.size() >= 2 ? shape[1] : shape[0];
        if (cols == 0 ||
            cols > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            static_cast<size_t>(first_row + row_count) > rows ||
            static_cast<int>(cols) != vocab_size)
        {
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        const float *row_ptr =
            static_cast<const float *>(tensor->gpu_data_ptr()) +
            static_cast<size_t>(first_row) * cols;
        auto *processed_rows =
            static_cast<float *>(stochastic_processed_logits_dev_) +
            static_cast<size_t>(first_slot) * cols;
        /*
         * Target/verifier distributions and MTP draft distributions may be
         * produced on different explicit streams.  Keep their partial top-k
         * workspaces disjoint so graph-captured stochastic verification cannot
         * race through a shared scratch buffer.
         */
        void *partial_vals =
            buffer == DeviceDistributionBuffer::Target
                ? stochastic_topk_partial_vals_dev_
                : stochastic_draft_topk_partial_vals_dev_;
        void *partial_idxs =
            buffer == DeviceDistributionBuffer::Target
                ? stochastic_topk_partial_idxs_dev_
                : stochastic_draft_topk_partial_idxs_dev_;
        const int partial_capacity =
            buffer == DeviceDistributionBuffer::Target
                ? stochastic_topk_partial_capacity_
                : stochastic_draft_topk_partial_capacity_;
        if (!partial_vals || !partial_idxs || partial_capacity <= 0)
            return false;

        const std::map<std::string, std::string> processed_tags = {
            {"operation", operation_name ? operation_name : "unknown"},
            {"rows", std::to_string(row_count)},
            {"top_k", std::to_string(params.top_k)}};
        std::shared_ptr<void> gpu_timing_start;
        std::shared_ptr<void> gpu_timing_stop;
        if (!beginPendingGpuTimingMeasurement(
                "stochastic_processed_rows_build_gpu",
                stream,
                &gpu_timing_start,
                &gpu_timing_stop))
        {
            return false;
        }
        bool ok = false;
        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "stochastic_processed_rows_build_enqueue",
                "decode",
                state_.device_id.toString(),
                processed_tags);
            ok = backend->enqueueBuildTopKTopPProcessedLogitsF32Device(
                    row_ptr,
                    row_count,
                    static_cast<int>(cols),
                    static_cast<int>(cols),
                    params.top_k,
                    params.top_p,
                    params.temperature,
                    state_.device_id.gpu_ordinal(),
                    stream,
                    processed_rows,
                    static_cast<int>(cols),
                    partial_vals,
                    partial_idxs,
                    partial_capacity);
        }
        if (!ok)
            return false;
        finishPendingGpuTimingMeasurement(
            "stochastic_processed_rows_build_gpu",
            stream,
            processed_tags,
            std::move(gpu_timing_start),
            std::move(gpu_timing_stop));

        PerfStatsCollector::addCounter(
            "mtp",
            "stochastic_processed_rows",
            static_cast<double>(row_count),
            "decode",
            state_.device_id.toString(),
            {{"buffer", buffer == DeviceDistributionBuffer::Target ? "target" : "draft"}});
        for (int row = 0; row < row_count; ++row)
        {
            const int slot = first_slot + row;
            if (buffer == DeviceDistributionBuffer::Target)
            {
                stochastic_target_top_k_[static_cast<size_t>(slot)] = params.top_k;
                stochastic_target_row_formats_[static_cast<size_t>(slot)] =
                    StochasticRowFormat::ProcessedLogits;
                stochastic_target_distribution_streams_[static_cast<size_t>(slot)] = stream;
            }
            else
            {
                stochastic_draft_top_k_[static_cast<size_t>(slot)] = params.top_k;
                stochastic_draft_row_formats_[static_cast<size_t>(slot)] =
                    StochasticRowFormat::ProcessedLogits;
                stochastic_draft_distribution_streams_[static_cast<size_t>(slot)] = stream;
            }
        }
        return true;
    }

    bool DeviceGraphOrchestrator::buildStochasticProcessedLogitRowsOnDevice(
        DeviceLogitsSource source,
        int first_row,
        DeviceDistributionBuffer buffer,
        int first_slot,
        int row_count,
        const SamplingParams &params,
        int vocab_size)
    {
        void *stream = nullptr;
        switch (source)
        {
        case DeviceLogitsSource::Main:
            stream = prepareMainLogitsDeviceConsumer(
                "buildStochasticProcessedLogitRowsOnDevice(Main)",
                MainLogitsHandoffMode::Observe);
            break;
        case DeviceLogitsSource::MainRequestBatch:
            stream = prepareMainLogitsDeviceConsumer(
                "buildStochasticProcessedLogitRowsOnDevice(MainRequestBatch)",
                MainLogitsHandoffMode::Observe,
                /*require_main_forward=*/false);
            break;
        case DeviceLogitsSource::MTP:
            stream = peekPendingLogitsStream(PendingLogitsStreamRole::MTPSidecar);
            break;
        case DeviceLogitsSource::AllPosition:
            stream = peekPendingLogitsStream(PendingLogitsStreamRole::AllPositionVerifier);
            break;
        }
        if (!stream)
        {
            stream = explicitGPUStreamForOperation("buildStochasticProcessedLogitRowsOnDevice");
        }
        if (!stream)
            return false;

        return buildStochasticProcessedLogitRowsOnDevice(
            source,
            first_row,
            buffer,
            first_slot,
            row_count,
            params,
            vocab_size,
            stream,
            "buildStochasticProcessedLogitRowsOnDevice");
    }

    bool DeviceGraphOrchestrator::buildStochasticDistributionOnDevice(
        DeviceLogitsSource source,
        int row,
        DeviceDistributionBuffer buffer,
        int slot,
        const SamplingParams &params,
        int vocab_size)
    {
        if (!supportsDeviceStochasticMTPVerification() ||
            row < 0 || slot < 0 || vocab_size <= 0 ||
            params.top_k <= 0 ||
            params.top_k > static_cast<int>(kStochasticDistributionMaxK))
        {
            return false;
        }

        TensorBase *tensor = resolveDeviceLogitsTensor(
            source,
            "buildStochasticDistributionOnDevice");

        if (!tensor || !tensor->deviceValid())
            return false;

        const auto &shape = tensor->shape();
        if (shape.empty())
            return false;
        const size_t rows = shape.size() >= 2 ? shape[0] : 1;
        const size_t cols = shape.size() >= 2 ? shape[1] : shape[0];
        if (cols == 0 ||
            cols > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            static_cast<size_t>(row) >= rows ||
            static_cast<int>(cols) != vocab_size)
        {
            return false;
        }

        int *token_ids = nullptr;
        float *probs = nullptr;
        int max_slots = 0;
        if (buffer == DeviceDistributionBuffer::Target)
        {
            token_ids = static_cast<int *>(stochastic_target_token_ids_dev_);
            probs = static_cast<float *>(stochastic_target_probs_dev_);
            max_slots = stochastic_target_row_capacity_;
        }
        else
        {
            token_ids = static_cast<int *>(stochastic_draft_token_ids_dev_);
            probs = static_cast<float *>(stochastic_draft_probs_dev_);
            max_slots = stochastic_draft_row_capacity_;
        }
        if (!token_ids || !probs || slot >= max_slots)
            return false;

        if (buffer == DeviceDistributionBuffer::Target)
        {
            stochastic_target_top_k_[static_cast<size_t>(slot)] = 0;
            stochastic_target_row_formats_[static_cast<size_t>(slot)] =
                StochasticRowFormat::Empty;
            stochastic_target_distribution_streams_[static_cast<size_t>(slot)] = nullptr;
        }
        else
        {
            stochastic_draft_top_k_[static_cast<size_t>(slot)] = 0;
            stochastic_draft_row_formats_[static_cast<size_t>(slot)] =
                StochasticRowFormat::Empty;
            stochastic_draft_distribution_streams_[static_cast<size_t>(slot)] = nullptr;
            clearStochasticDraftSampleReadySlot(
                slot,
                StochasticSampleReadyClearMode::Force);
        }

        void *stream = nullptr;
        bool used_pending_main_stream = false;
        bool used_pending_mtp_stream = false;
        if (source == DeviceLogitsSource::Main ||
            source == DeviceLogitsSource::MainRequestBatch)
        {
            /*
             * MTP condition-forward replay may leave main logits pending on
             * the capture stream. Build the stochastic distribution on that
             * same stream so the CPU does not force a replay sync before the
             * sampler kernel has been queued.
             */
            used_pending_main_stream =
                peekPendingLogitsStream(
                    PendingLogitsStreamRole::MainDecode) != nullptr;
            stream = prepareMainLogitsDeviceConsumer(
                source == DeviceLogitsSource::Main
                    ? "buildStochasticDistributionOnDevice(Main)"
                    : "buildStochasticDistributionOnDevice(MainRequestBatch)",
                MainLogitsHandoffMode::Consume,
                /*require_main_forward=*/
                source == DeviceLogitsSource::Main);
        }
        else if (source == DeviceLogitsSource::MTP)
        {
            stream = consumePendingLogitsStream(
                PendingLogitsStreamRole::MTPSidecar,
                "buildStochasticDistributionOnDevice(MTP)");
            used_pending_mtp_stream = stream != nullptr;
        }
        else if (source == DeviceLogitsSource::AllPosition)
        {
            /*
             * A deferred all-position verifier replay leaves its capture stream
             * here for the next device-side consumer. Stochastic verification
             * may build several target/bonus distributions before the summary
             * kernel finally copies compact metadata to host, so the stream is
             * intentionally not cleared by individual distribution builds.
             */
            stream = peekPendingLogitsStream(PendingLogitsStreamRole::AllPositionVerifier);
        }
        if (!stream)
            stream = explicitGPUStreamForOperation("buildStochasticDistributionOnDevice");
        if (!stream)
            return false;

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        const void *gpu_ptr = tensor->gpu_data_ptr();
        if (!gpu_ptr)
            return false;
        const float *row_ptr =
            static_cast<const float *>(gpu_ptr) +
            static_cast<size_t>(row) * cols;

        const size_t slot_offset =
            static_cast<size_t>(slot) * kStochasticDistributionMaxK;
        void *partial_vals =
            buffer == DeviceDistributionBuffer::Target
                ? stochastic_topk_partial_vals_dev_
                : stochastic_draft_topk_partial_vals_dev_;
        void *partial_idxs =
            buffer == DeviceDistributionBuffer::Target
                ? stochastic_topk_partial_idxs_dev_
                : stochastic_draft_topk_partial_idxs_dev_;
        const int partial_capacity =
            buffer == DeviceDistributionBuffer::Target
                ? stochastic_topk_partial_capacity_
                : stochastic_draft_topk_partial_capacity_;
        if (!partial_vals || !partial_idxs || partial_capacity <= 0)
            return false;
        bool ok = false;
        const std::map<std::string, std::string> distribution_tags = {
            {"slot", std::to_string(slot)},
            {"source", source == DeviceLogitsSource::Main
                           ? "main"
                           : (source == DeviceLogitsSource::MainRequestBatch
                                  ? "main_request_batch"
                           : (source == DeviceLogitsSource::MTP
                                  ? "mtp"
                                  : "all_position"))},
            {"top_k", std::to_string(params.top_k)}};
        std::shared_ptr<void> gpu_timing_start;
        std::shared_ptr<void> gpu_timing_stop;
        if (!beginPendingGpuTimingMeasurement(
                "stochastic_distribution_build_gpu",
                stream,
                &gpu_timing_start,
                &gpu_timing_stop))
        {
            return false;
        }
        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "stochastic_distribution_build_enqueue",
                "decode",
                state_.device_id.toString(),
                distribution_tags);
            ok = backend->enqueueBuildTopKTopPDistributionF32Device(
                row_ptr,
                static_cast<int>(cols),
                params.top_k,
                params.top_p,
                params.temperature,
                state_.device_id.gpu_ordinal(),
                stream,
                token_ids + slot_offset,
                probs + slot_offset,
                partial_vals,
                partial_idxs,
                partial_capacity);
        }
        if (ok)
        {
            finishPendingGpuTimingMeasurement(
                "stochastic_distribution_build_gpu",
                stream,
                distribution_tags,
                std::move(gpu_timing_start),
                std::move(gpu_timing_stop));
        }
        if (ok)
        {
            if (used_pending_main_stream)
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "stochastic_main_distribution_stream_handoffs",
                    1.0,
                    "decode",
                    state_.device_id.toString(),
                    {{"slot", std::to_string(slot)}});
            }
            if (used_pending_mtp_stream)
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "stochastic_mtp_distribution_stream_handoffs",
                    1.0,
                    "decode",
                    state_.device_id.toString(),
                    {{"slot", std::to_string(slot)}});
            }
            const int effective_top_k =
                std::min(params.top_k, static_cast<int>(cols));
            size_t partial_blocks =
                (cols + kStochasticTopKSmallKThreads - 1) /
                kStochasticTopKSmallKThreads;
            partial_blocks = std::max<size_t>(partial_blocks, 1);
            partial_blocks = std::min(partial_blocks, kStochasticTopKPartialBlocks);
            const bool used_gpu_smallk_scratch =
                state_.device_id.is_gpu() &&
                effective_top_k > 0 &&
                effective_top_k <= static_cast<int>(kStochasticTopKSmallKCap) &&
                partial_vals &&
                partial_idxs &&
                partial_capacity >=
                    static_cast<int>(partial_blocks * static_cast<size_t>(effective_top_k));
            if (used_gpu_smallk_scratch)
            {
                const char *source_name = "unknown";
                switch (source)
                {
                case DeviceLogitsSource::Main:
                    source_name = "main";
                    break;
                case DeviceLogitsSource::MTP:
                    source_name = "mtp";
                    break;
                case DeviceLogitsSource::AllPosition:
                    source_name = "all_position";
                    break;
                case DeviceLogitsSource::MainRequestBatch:
                    source_name = "main_request_batch";
                    break;
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "stochastic_topk_smallk_scratch_distribution_builds",
                    1.0,
                    "decode",
                    state_.device_id.toString(),
                    {
                        {"source", source_name},
                        {"top_k", std::to_string(effective_top_k)},
                    });
            }
            if (buffer == DeviceDistributionBuffer::Target)
            {
                stochastic_target_top_k_[static_cast<size_t>(slot)] = params.top_k;
                stochastic_target_row_formats_[static_cast<size_t>(slot)] =
                    StochasticRowFormat::CompactDistribution;
                stochastic_target_distribution_streams_[static_cast<size_t>(slot)] = stream;
                /*
                 * Target distribution slots are reused for all-position
                 * verifier rows after the first target token has already been
                 * sampled.  Do not clear target sample readiness here: that
                 * readiness belongs to STOCHASTIC_TARGET_SAMPLE_TOKENS and is
                 * the event edge that lets the verifier summary read the
                 * deferred first token safely from another stream.
                 */
            }
            else
            {
                stochastic_draft_top_k_[static_cast<size_t>(slot)] = params.top_k;
                stochastic_draft_row_formats_[static_cast<size_t>(slot)] =
                    StochasticRowFormat::CompactDistribution;
                stochastic_draft_distribution_streams_[static_cast<size_t>(slot)] = stream;
                clearStochasticDraftSampleReadySlot(
                    slot,
                    StochasticSampleReadyClearMode::Force);
            }
        }
        return ok;
    }

    bool DeviceGraphOrchestrator::
        buildCapturedStochasticVerifierTargetDistributions(
            int row_count,
            const SamplingParams &params,
            const MTPRequestPenaltyPolicy &penalty_policy,
            int vocab_size)
    {
        if (!supportsDeviceStochasticMTPVerification() ||
            !state_.device_id.is_gpu() || row_count <= 0 ||
            row_count > stochastic_target_row_capacity_ ||
            params.top_k <= 0 ||
            params.top_k > static_cast<int>(kStochasticDistributionMaxK) ||
            vocab_size <= 0)
        {
            return false;
        }

        /*
         * A verifier forward is the sole legal producer for this transaction.
         * Replacing a missing handoff with an arbitrary operation stream would
         * make stale logits appear valid and would prevent the parent graph
         * from expressing its true producer edge.
         */
        void *const verifier_stream = peekPendingLogitsStream(
            PendingLogitsStreamRole::AllPositionVerifier);
        if (!verifier_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Captured stochastic target distributions require an exact all-position verifier producer stream");
            return false;
        }

        std::string captured_error;
        if (!materializeMTPStochasticTargetDistributionGraph(
                row_count,
                params,
                penalty_policy,
                vocab_size,
                &captured_error))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Could not materialize captured stochastic verifier target distributions: "
                      << captured_error);
            return false;
        }

        const bool uses_device_owned_active_rows =
            mtpStochasticTargetDistributionCache(row_count).stage &&
            mtpStochasticTargetDistributionCache(row_count).stage->getParams()
                    .active_rows_device != nullptr;
        if (uses_device_owned_active_rows &&
            (!arena_ ||
             !arena_->prepareForRead(
                 BufferId::MTP_VERIFIER_REQUEST_LENGTHS,
                 state_.device_id,
                 verifier_stream)))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Captured stochastic verifier target distributions could not join resident active-row readiness");
            return false;
        }

        const bool apply_penalties = penalty_policy.enabled();
        if (apply_penalties &&
            (!arena_ ||
             !consumeMTPRequestPenaltyPolicyOnDevice(
                 penalty_policy,
                 verifier_stream,
                 "captured_stochastic_verifier_target_distributions") ||
             !arena_->prepareForRead(
                 BufferId::MTP_GENERATED_TOKEN_COUNTS,
                 state_.device_id,
                 verifier_stream) ||
             !arena_->prepareForRead(
                 BufferId::MTP_VERIFIER_INPUT_TOKENS,
                 state_.device_id,
                 verifier_stream)))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Captured stochastic verifier target distributions could not join resident history readiness");
            return false;
        }

        for (int row = 0; row < row_count; ++row)
        {
            stochastic_target_top_k_[static_cast<size_t>(row)] = 0;
            stochastic_target_row_formats_[static_cast<size_t>(row)] =
                StochasticRowFormat::Empty;
            stochastic_target_distribution_streams_[static_cast<size_t>(row)] =
                nullptr;
        }

        if (!executeMTPStochasticTargetDistributionCaptured(
                verifier_stream,
                row_count,
                &captured_error))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Captured stochastic verifier target distributions failed: "
                      << captured_error);
            return false;
        }

        /*
         * The capture stream has handed completion back to verifier_stream at
         * this point. Publish the exact same stream as both the logits lineage
         * and compact-distribution lineage; the downstream outcome reducer
         * consumes that one owner exactly once.
         */
        publishPendingLogitsStream(
            PendingLogitsStreamRole::AllPositionVerifier,
            verifier_stream,
            "buildCapturedStochasticVerifierTargetDistributions");
        if (!recordAllPositionVerifierStateReady(
                verifier_stream,
                "buildCapturedStochasticVerifierTargetDistributions"))
        {
            throw std::runtime_error(
                "Failed to publish captured stochastic verifier target-distribution readiness");
        }

        for (int row = 0; row < row_count; ++row)
        {
            stochastic_target_top_k_[static_cast<size_t>(row)] = params.top_k;
            stochastic_target_row_formats_[static_cast<size_t>(row)] =
                StochasticRowFormat::CompactDistribution;
            stochastic_target_distribution_streams_[static_cast<size_t>(row)] =
                verifier_stream;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "captured_stochastic_verifier_target_distribution_rows",
            static_cast<double>(row_count),
            "decode",
            state_.device_id.toString(),
            {{"top_k", std::to_string(params.top_k)},
             {"penalties", apply_penalties ? "true" : "false"},
             {"implementation", "monolithic_graph"}});
        return true;
    }

    bool DeviceGraphOrchestrator::buildStochasticDistributionsOnDevice(
        DeviceLogitsSource source,
        int first_row,
        DeviceDistributionBuffer buffer,
        int first_slot,
        int row_count,
        const SamplingParams &params,
        int vocab_size)
    {
        if (!supportsDeviceStochasticMTPVerification() ||
            (source != DeviceLogitsSource::AllPosition &&
             source != DeviceLogitsSource::Main &&
             source != DeviceLogitsSource::MainRequestBatch) ||
            buffer != DeviceDistributionBuffer::Target ||
            first_row < 0 ||
            first_slot < 0 ||
            row_count <= 0 ||
            params.top_k <= 0 ||
            params.top_k > static_cast<int>(kStochasticDistributionMaxK))
        {
            return false;
        }

        if (first_slot + row_count > stochastic_target_row_capacity_)
            return false;

        auto *token_ids = static_cast<int *>(stochastic_target_token_ids_dev_);
        auto *probs = static_cast<float *>(stochastic_target_probs_dev_);
        if (!token_ids || !probs)
            return false;

        for (int row = 0; row < row_count; ++row)
        {
            const int slot = first_slot + row;
            stochastic_target_top_k_[static_cast<size_t>(slot)] = 0;
            stochastic_target_row_formats_[static_cast<size_t>(slot)] =
                StochasticRowFormat::Empty;
            stochastic_target_distribution_streams_[static_cast<size_t>(slot)] = nullptr;
        }

        /*
         * Batched all-position verifier distributions must stay ordered behind
         * the launch-only verifier replay. Individual rows do not consume the
         * pending stream; the final compact summary copy closes this handoff.
         */
        const PendingLogitsStreamRole stream_role =
            (source == DeviceLogitsSource::Main ||
             source == DeviceLogitsSource::MainRequestBatch)
                ? PendingLogitsStreamRole::MainDecode
                : PendingLogitsStreamRole::AllPositionVerifier;
        void *stream = nullptr;
        if (stream_role == PendingLogitsStreamRole::MainDecode)
        {
            stream = prepareMainLogitsDeviceConsumer(
                "buildStochasticDistributionsOnDevice",
                MainLogitsHandoffMode::Observe,
                /*require_main_forward=*/
                source == DeviceLogitsSource::Main);
        }
        else
        {
            stream = peekPendingLogitsStream(stream_role);
            if (!stream)
            {
                stream = explicitGPUStreamForOperation(
                    "buildStochasticDistributionsOnDevice");
            }
        }
        if (!stream)
            return false;

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        TensorBase *tensor = resolveDeviceLogitsTensor(
            source,
            "buildStochasticDistributionsOnDevice");
        if (!tensor || !tensor->gpu_data_ptr())
            return false;
        const auto &shape = tensor->shape();
        if (shape.size() < 2)
            return false;
        const size_t rows = shape[0];
        const size_t cols = shape[1];
        if (first_row + row_count > static_cast<int>(rows) ||
            (vocab_size > 0 && static_cast<size_t>(vocab_size) != cols))
        {
            return false;
        }

        const float *row_ptr =
            static_cast<const float *>(tensor->gpu_data_ptr()) +
            static_cast<size_t>(first_row) * cols;
        const size_t slot_offset =
            static_cast<size_t>(first_slot) * kStochasticDistributionMaxK;

        bool ok = false;
        const char *source_name =
            source == DeviceLogitsSource::Main
                ? "main"
                : source == DeviceLogitsSource::MainRequestBatch
                      ? "main_request_batch"
                      : "all_position";
        const std::map<std::string, std::string> distribution_tags = {
            {"first_slot", std::to_string(first_slot)},
            {"rows", std::to_string(row_count)},
            {"source", source_name},
            {"top_k", std::to_string(params.top_k)}};
        std::shared_ptr<void> gpu_timing_start;
        std::shared_ptr<void> gpu_timing_stop;
        if (!beginPendingGpuTimingMeasurement(
                "stochastic_distribution_batch_build_gpu",
                stream,
                &gpu_timing_start,
                &gpu_timing_stop))
        {
            return false;
        }
        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "stochastic_distribution_batch_build_enqueue",
                "decode",
                state_.device_id.toString(),
                distribution_tags);
            ok = backend->enqueueBuildTopKTopPDistributionsF32Device(
                row_ptr,
                row_count,
                static_cast<int>(cols),
                static_cast<int>(cols),
                params.top_k,
                params.top_p,
                params.temperature,
                state_.device_id.gpu_ordinal(),
                stream,
                token_ids + slot_offset,
                static_cast<int>(kStochasticDistributionMaxK),
                probs + slot_offset,
                stochastic_topk_partial_vals_dev_,
                stochastic_topk_partial_idxs_dev_,
                stochastic_topk_partial_capacity_);
        }
        if (ok)
        {
            finishPendingGpuTimingMeasurement(
                "stochastic_distribution_batch_build_gpu",
                stream,
                distribution_tags,
                std::move(gpu_timing_start),
                std::move(gpu_timing_stop));
        }

        if (!ok)
            return false;

        /*
         * This method launches the compact-distribution kernel directly rather
         * than through DeviceGraphExecutor, so no stage epilogue exists to
         * publish its arena outputs.  The following captured outcome graph is a
         * true consumer of token IDs and probabilities; stream ordering alone
         * cannot satisfy its graph-frontier coherence proof while the arena
         * still labels those tensors host-authoritative.  Publish every compact
         * workspace output written by this direct launch on the exact producer
         * stream.  (`ALL_POSITION_LOGITS` is only an output of the captured
         * penalty-enabled stage and is not mutated here.) This records events
         * only and does not synchronize or transfer data.
         */
        if (!arena_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Stochastic distribution build has no arena coherence owner");
            return false;
        }
        arena_->markWritten(
            BufferId::STOCHASTIC_TARGET_TOKEN_IDS,
            state_.device_id,
            stream);
        arena_->markWritten(
            BufferId::STOCHASTIC_TARGET_PROBS,
            state_.device_id,
            stream);
        arena_->markWritten(
            BufferId::STOCHASTIC_TOPK_PARTIAL_VALS,
            state_.device_id,
            stream);
        arena_->markWritten(
            BufferId::STOCHASTIC_TOPK_PARTIAL_IDXS,
            state_.device_id,
            stream);

        for (int row = 0; row < row_count; ++row)
        {
            const int slot = first_slot + row;
            stochastic_target_top_k_[static_cast<size_t>(slot)] = params.top_k;
            stochastic_target_row_formats_[static_cast<size_t>(slot)] =
                StochasticRowFormat::CompactDistribution;
            stochastic_target_distribution_streams_[static_cast<size_t>(slot)] = stream;
        }

        const int effective_top_k =
            std::min(params.top_k, static_cast<int>(cols));
        size_t partial_blocks =
            (cols + kStochasticTopKSmallKThreads - 1) /
            kStochasticTopKSmallKThreads;
        partial_blocks = std::max<size_t>(partial_blocks, 1);
        partial_blocks = std::min(partial_blocks, kStochasticTopKPartialBlocks);
        if (effective_top_k > 0 &&
            effective_top_k <= static_cast<int>(kStochasticTopKSmallKCap))
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "stochastic_topk_smallk_scratch_distribution_builds",
                static_cast<double>(row_count),
                "decode",
                state_.device_id.toString(),
                {
                    {"source", std::string(source_name) + "_batched"},
                    {"top_k", std::to_string(effective_top_k)},
                    {"partial_blocks", std::to_string(partial_blocks)},
                });
        }

        return true;
    }

    bool DeviceGraphOrchestrator::sampleStochasticDistributionOnDeviceImpl(
        DeviceDistributionBuffer buffer,
        int slot,
        float threshold,
        int32_t *out_token_host,
        uint64_t threshold_seed,
        const int32_t *threshold_position_device,
        int threshold_position_offset)
    {
        if (!supportsDeviceStochasticMTPVerification() ||
            slot < 0 ||
            (threshold_position_device && threshold_seed == 0) ||
            (!threshold_position_device && threshold_seed != 0))
            return false;

        int *token_ids = nullptr;
        float *probs = nullptr;
        int max_slots = 0;
        if (buffer == DeviceDistributionBuffer::Target)
        {
            token_ids = static_cast<int *>(stochastic_target_token_ids_dev_);
            probs = static_cast<float *>(stochastic_target_probs_dev_);
            max_slots = stochastic_target_row_capacity_;
        }
        else
        {
            token_ids = static_cast<int *>(stochastic_draft_token_ids_dev_);
            probs = static_cast<float *>(stochastic_draft_probs_dev_);
            max_slots = stochastic_draft_row_capacity_;
        }
        if (!token_ids || !probs || slot >= max_slots)
            return false;
        const int active_top_k =
            buffer == DeviceDistributionBuffer::Target
                ? stochastic_target_top_k_[static_cast<size_t>(slot)]
                : stochastic_draft_top_k_[static_cast<size_t>(slot)];
        if (active_top_k <= 0 ||
            active_top_k > static_cast<int>(kStochasticDistributionMaxK))
        {
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;
        void *stream = nullptr;
        if (buffer == DeviceDistributionBuffer::Target)
        {
            stream = stochastic_target_distribution_streams_[static_cast<size_t>(slot)];
            stochastic_target_distribution_streams_[static_cast<size_t>(slot)] = nullptr;
        }
        else
        {
            stream = stochastic_draft_distribution_streams_[static_cast<size_t>(slot)];
            stochastic_draft_distribution_streams_[static_cast<size_t>(slot)] = nullptr;
        }
        if (!stream)
        {
            // A distribution constructed by a caller without a dedicated
            // producer stream is ordered on the orchestrator's explicit
            // operation stream. The device-default/null stream is never
            // acceptable here.
            stream = explicitGPUStreamForOperation("sampleStochasticDistributionOnDevice");
        }
        if (!stream)
            return false;

        const char *buffer_name =
            buffer == DeviceDistributionBuffer::Draft ? "draft" : "target";

        /*
         * Sampled target and draft tokens both outlive the sampler kernel.
         * Keep them out of STOCHASTIC_VERIFY_TOKENS because verifier rows reuse
         * that buffer for row-local accept/reject outputs later in the same
         * speculative step.
         */
        int *out_token_dev =
            (buffer == DeviceDistributionBuffer::Draft)
                ? static_cast<int *>(stochastic_draft_sample_tokens_dev_) + slot
                : static_cast<int *>(stochastic_target_sample_tokens_dev_) + slot;
        if (!out_token_dev)
            return false;

        /*
         * The selected draft probability is small but valuable metadata for the
         * vLLM-style fused verifier: the verifier can read q(draft_token)
         * directly instead of rebuilding or rescanning the draft distribution.
         * Target samples do not currently need the same metadata, so only draft
         * slots write this companion buffer.
         */
        float *out_probability_dev =
            (buffer == DeviceDistributionBuffer::Draft && stochastic_draft_sample_probs_dev_)
                ? static_cast<float *>(stochastic_draft_sample_probs_dev_) + slot
                : nullptr;
        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "sample_stochastic_distribution_enqueue",
                "decode",
                state_.device_id.toString(),
                {{"buffer", buffer_name},
                 {"slot", std::to_string(slot)},
                 {"top_k", std::to_string(active_top_k)},
                 {"resident_position", boolTag(threshold_position_device != nullptr)}});
            if (!backend->enqueueSampleDistributionF32Device(
                    token_ids + static_cast<size_t>(slot) * kStochasticDistributionMaxK,
                    probs + static_cast<size_t>(slot) * kStochasticDistributionMaxK,
                    active_top_k,
                    threshold,
                    state_.device_id.gpu_ordinal(),
                    stream,
                    out_token_dev,
                    out_probability_dev,
                    threshold_seed,
                    threshold_position_device,
                    threshold_position_offset))
            {
                return false;
            }
        }

        if (buffer == DeviceDistributionBuffer::Draft)
        {
            if (!recordStochasticDraftSampleReady(
                    slot,
                    stream,
                    /*verifier_consumer_pending=*/out_token_host == nullptr))
                return false;
        }
        else
        {
            if (!recordStochasticTargetSampleReady(
                    slot,
                    stream,
                    /*verifier_consumer_pending=*/out_token_host == nullptr, StochasticTargetPublication::NewValue))
                return false;
        }

        if (!out_token_host)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "sample_stochastic_distribution_deferred_host_reads",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"buffer", buffer_name},
                 {"slot", std::to_string(slot)},
                 {"top_k", std::to_string(active_top_k)}});
            return true;
        }

        int32_t token = -1;
        /*
         * This read is the semantic point where the runner needs the sampled
         * draft token on the host. It drains all earlier distribution work on
         * `stream`, but the source is trusted arena scratch, so use the fast
         * D2H path and avoid per-token ROCm pointer-attribute validation.
         */
        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "sample_stochastic_distribution_d2h_sync",
                "decode",
                state_.device_id.toString(),
                {{"buffer", buffer_name},
                 {"slot", std::to_string(slot)},
                 {"top_k", std::to_string(active_top_k)}});
            if (!backend->deviceToHostFast(&token, out_token_dev, sizeof(int),
                                           state_.device_id.gpu_ordinal(), stream))
            {
                return false;
            }
        }
        drainPendingGpuTimingMeasurements(backend);
        *out_token_host = token;
        return true;
    }

    bool DeviceGraphOrchestrator::publishCapturedMTPDraftToken(
        int row,
        int slot,
        const MTPRequestPenaltyPolicy &penalty_policy)
    {
        if (!state_.device_id.is_gpu() || row < 0 || slot < 0 ||
            slot >= stochastic_draft_row_capacity_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Captured MTP draft publication received invalid GPU row/slot geometry");
            return false;
        }

        /*
         * A proposal graph has exactly one legal producer: the sidecar graph
         * that wrote MTP_LOGITS.  Do not invent another stream when that edge
         * is absent; doing so would turn a missing dependency into a race.
         */
        void *const producer_stream = consumePendingLogitsStream(
            PendingLogitsStreamRole::MTPSidecar,
            "publishCapturedMTPDraftToken");
        if (!producer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Captured MTP draft publication requires the exact sidecar-logit producer stream");
            return false;
        }
        if (penalty_policy.enabled() &&
            (!arena_ ||
             !consumeMTPRequestPenaltyPolicyOnDevice(
                 penalty_policy,
                 producer_stream,
                 "captured_mtp_draft_token_publication") ||
             (slot > 0 && !waitForRequiredStochasticDraftSampleReadyRange(
                 /*first_slot=*/0,
                 slot,
                 producer_stream,
                 "captured_mtp_draft_penalty_history")) ||
             !arena_->prepareForRead(
                 BufferId::MTP_GENERATED_TOKEN_COUNTS,
                 state_.device_id,
                 producer_stream) ||
             !arena_->prepareForRead(
                 BufferId::MTP_CONDITION_TOKEN,
                 state_.device_id,
                 producer_stream)))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Captured MTP draft publication could not consume request-owned penalty history");
            return false;
        }

        stochastic_draft_top_k_[static_cast<size_t>(slot)] = 0;
        stochastic_draft_row_formats_[static_cast<size_t>(slot)] =
            StochasticRowFormat::Empty;
        stochastic_draft_distribution_streams_[static_cast<size_t>(slot)] =
            nullptr;
        clearStochasticDraftSampleReadySlot(
            slot,
            StochasticSampleReadyClearMode::Force);

        std::string graph_error;
        if (!materializeMTPDraftTokenPublicationGraph(
                row,
                slot,
                penalty_policy,
                &graph_error) ||
            !executeMTPDraftTokenPublicationCaptured(
                producer_stream,
                row,
                slot,
                &graph_error) ||
            !recordStochasticDraftSampleReady(
                slot,
                producer_stream,
                /*verifier_consumer_pending=*/true))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Captured MTP draft publication failed: "
                      << graph_error);
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "captured_mtp_draft_token_publications",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"row", std::to_string(row)},
             {"slot", std::to_string(slot)},
             {"penalties",
              penalty_policy.enabled() ? "true" : "false"},
             {"host_observation", "false"}});
        return true;
    }

    bool DeviceGraphOrchestrator::sampleStochasticDraftProposalOnDeviceImpl(
        DeviceLogitsSource source,
        int row,
        int slot,
        const SamplingParams &params,
        int vocab_size,
        float threshold,
        int32_t *out_token_host,
        bool verifier_consumer_pending,
        const MTPRequestPenaltyPolicy &proposal_penalty_policy)
    {
        (void)params;
        (void)threshold;

        if (!supportsDeviceStochasticMTPVerification() ||
            source != DeviceLogitsSource::MTP ||
            row < 0 || slot < 0 ||
            slot >= stochastic_draft_row_capacity_ ||
            vocab_size <= 0 ||
            !stochastic_draft_sample_tokens_dev_ ||
            !stochastic_draft_sample_probs_dev_ ||
            !argmax_partial_vals_dev_ ||
            !argmax_partial_idxs_dev_ ||
            argmax_partial_capacity_ <= 0)
        {
            return false;
        }

        auto it = state_.extension_buffers.find(mtpSamplingLogitsBufferId());
        TensorBase *tensor =
            it == state_.extension_buffers.end() ? nullptr : it->second.get();
        if (!tensor || !tensor->deviceValid() || !tensor->gpu_data_ptr())
            return false;

        const auto &shape = tensor->shape();
        if (shape.empty())
            return false;
        const size_t rows = shape.size() >= 2 ? shape[0] : 1;
        const size_t cols = shape.size() >= 2 ? shape[1] : shape[0];
        if (cols == 0 ||
            cols > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            static_cast<size_t>(row) >= rows ||
            static_cast<int>(cols) != vocab_size)
        {
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;
        auto *out_token_dev =
            static_cast<int *>(stochastic_draft_sample_tokens_dev_) + slot;

        /*
         * vLLM's default MTP proposal is greedy. The verifier receives the
         * sampled token and treats the draft distribution q as one-hot at that
         * token, avoiding full draft probability/logit rows in runner scratch.
         * The proposal is always published through the mandatory captured
         * stage so the production method and parent-loop fragment have one
         * implementation. Its explicit proposal policy is neutral for stochastic
         * verification; greedy verification applies request branch penalties
         * inside that same captured stage, exactly once per fresh sidecar row.
         */
        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "stochastic_draft_greedy_proposal_enqueue",
                "decode",
                state_.device_id.toString(),
                {{"slot", std::to_string(slot)},
                 {"vocab", std::to_string(vocab_size)}});
            if (!publishCapturedMTPDraftToken(
                    row,
                    slot,
                    proposal_penalty_policy))
            {
                return false;
            }
        }

        auto &ready =
            stochastic_draft_sample_ready_[static_cast<size_t>(slot)];
        ready.verifier_consumer_pending =
            ready.verifier_consumer_pending || verifier_consumer_pending ||
            out_token_host == nullptr;
        void *const stream = ready.producer_stream;
        if (!ready.valid || !stream)
            return false;

        PerfStatsCollector::addCounter(
            "mtp",
            "stochastic_draft_greedy_proposals",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"slot", std::to_string(slot)}});

        if (!out_token_host)
            return true;

        int32_t token = -1;
        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "stochastic_draft_greedy_proposal_d2h_sync",
                "decode",
                state_.device_id.toString(),
                {{"slot", std::to_string(slot)}});
            if (!backend->deviceToHostFast(&token,
                                           out_token_dev,
                                           sizeof(int),
                                           state_.device_id.gpu_ordinal(),
                                           stream))
            {
                return false;
            }
        }
        *out_token_host = token;
        return token >= 0;
    }

    int DeviceGraphOrchestrator::sampleStochasticDraftProposalOnDevice(
        DeviceLogitsSource source,
        int row,
        int slot,
        const SamplingParams &params,
        int vocab_size,
        float threshold)
    {
        int32_t token = -1;
        if (!sampleStochasticDraftProposalOnDeviceImpl(
                source,
                row,
                slot,
                params,
                vocab_size,
                threshold,
                &token,
                /*verifier_consumer_pending=*/true,
                MTPRequestPenaltyPolicy{}))
        {
            return -1;
        }
        return token;
    }

    bool DeviceGraphOrchestrator::sampleStochasticDraftProposalOnDeviceDeferred(
        DeviceLogitsSource source,
        int row,
        int slot,
        const SamplingParams &params,
        int vocab_size,
        float threshold)
    {
        return sampleStochasticDraftProposalOnDeviceImpl(
            source,
            row,
            slot,
            params,
            vocab_size,
            threshold,
            /*out_token_host=*/nullptr,
            /*verifier_consumer_pending=*/true,
            MTPRequestPenaltyPolicy{});
    }

    DeviceStochasticTargetSampleSlotHandle
    DeviceGraphOrchestrator::deviceStochasticTargetSampleProducerSlot(int slot)
    {
        DeviceStochasticTargetSampleSlotHandle handle;
        handle.slot = slot;
        handle.device = state_.device_id;

        if (!state_.device_id.is_gpu() ||
            slot < 0 ||
            slot >= stochastic_target_row_capacity_ ||
            slot >= static_cast<int>(stochastic_target_sample_ready_.size()) ||
            !stochastic_target_sample_tokens_dev_)
        {
            return {};
        }

        const auto &ready =
            stochastic_target_sample_ready_[static_cast<size_t>(slot)];
        if (!ready.valid || !ready.event || !ready.producer_stream)
            return {};

        handle.token_device =
            static_cast<int32_t *>(stochastic_target_sample_tokens_dev_) + slot;
        handle.stream = ready.producer_stream;
        return handle.valid() ? handle : DeviceStochasticTargetSampleSlotHandle{};
    }

    DeviceStochasticTargetSampleSlotHandle
    DeviceGraphOrchestrator::
        deviceStochasticTargetSampleBroadcastDestinationSlot(int slot)
    {
        DeviceStochasticTargetSampleSlotHandle handle;
        handle.slot = slot;
        handle.device = state_.device_id;

        if (!state_.device_id.is_gpu() ||
            slot < 0 ||
            slot >= stochastic_target_row_capacity_ ||
            slot >= static_cast<int>(stochastic_target_sample_ready_.size()) ||
            !stochastic_target_sample_tokens_dev_)
        {
            return {};
        }

        /*
         * Acquire the peer's replicated main-forward publication through the
         * same durable handoff used by an ordinary sampler.  The returned
         * stream has consumed the optional logits token and joined
         * ForwardGraphOutputReady, so NCCL/RCCL cannot publish a target token
         * ahead of peer-local terminal hidden or live recurrent state.
         */
        void *stream = prepareMainLogitsDeviceConsumer(
            "deviceStochasticTargetSampleBroadcastDestinationSlot",
            MainLogitsHandoffMode::Consume);
        if (!stream)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Mirrored target broadcast destination "
                "could not join the main-forward producer for slot="
                << slot << " device=" << state_.device_id.toString());
            return {};
        }
        clearStochasticTargetSampleReadySlot(
            slot,
            StochasticSampleReadyClearMode::Force);

        handle.token_device =
            static_cast<int32_t *>(stochastic_target_sample_tokens_dev_) + slot;
        handle.stream = stream;
        return handle.valid() ? handle : DeviceStochasticTargetSampleSlotHandle{};
    }

    bool DeviceGraphOrchestrator::recordStochasticTargetSampleSlotReadyFromDevice(
        int slot,
        void *producer_stream,
        bool verifier_consumer_pending)
    {
        if (!state_.device_id.is_gpu() ||
            slot < 0 ||
            slot >= stochastic_target_row_capacity_ ||
            !stochastic_target_sample_tokens_dev_ ||
            !producer_stream)
        {
            return false;
        }

        return recordStochasticTargetSampleReady(
            slot,
            producer_stream,
            verifier_consumer_pending, StochasticTargetPublication::NewValue);
    }

    bool DeviceGraphOrchestrator::stageStochasticDraftTokensForDeviceVerification(
        const int32_t *draft_tokens,
        int draft_token_count,
        int first_draft_slot)
    {
        if (!state_.device_id.is_gpu() ||
            !draft_tokens ||
            first_draft_slot < 0 ||
            draft_token_count <= 0 ||
            first_draft_slot + draft_token_count > stochastic_draft_row_capacity_ ||
            !stochastic_draft_sample_tokens_dev_)
        {
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        void *stream = explicitGPUStreamForOperation(
            "stageStochasticDraftTokensForDeviceVerification");
        if (!backend || !stream)
            return false;

        for (int slot = 0; slot < draft_token_count; ++slot)
            clearStochasticDraftSampleReadySlot(
                first_draft_slot + slot,
                StochasticSampleReadyClearMode::Force);

        if (!backend->hostToDeviceOnStream(
                static_cast<int32_t *>(stochastic_draft_sample_tokens_dev_) +
                    first_draft_slot,
                draft_tokens,
                sizeof(int32_t) * static_cast<size_t>(draft_token_count),
                state_.device_id.gpu_ordinal(),
                stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to stage stochastic draft tokens for verifier");
            return false;
        }

        for (int slot = 0; slot < draft_token_count; ++slot)
        {
            if (!recordStochasticDraftSampleReady(
                    first_draft_slot + slot,
                    stream,
                    /*verifier_consumer_pending=*/true))
                return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "stochastic_request_batch_draft_token_stages",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"draft_tokens", std::to_string(draft_token_count)},
             {"first_slot", std::to_string(first_draft_slot)}});
        return true;
    }

    bool DeviceGraphOrchestrator::stageStochasticTargetTokenForDeviceSampling(
        int32_t target_token,
        int target_sample_slot)
    {
        if (!state_.device_id.is_gpu() ||
            target_token < 0 ||
            target_sample_slot < 0 ||
            target_sample_slot >= stochastic_target_row_capacity_ ||
            !stochastic_target_sample_tokens_dev_)
        {
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        // A forced token replaces a sampled token at the same live-state
        // boundary. Its consumers also read the current KV position and
        // terminal hidden row; publishing on an unrelated stream can race the
        // preceding forward even though the scalar itself is ready. Reuse the
        // sampler's canonical event-aware boundary without consuming its logits
        // handoff. Any output kind also covers diagnostic verifier successors.
        void *stream = prepareMainLogitsDeviceConsumer(
            "stageStochasticTargetTokenForDeviceSampling",
            MainLogitsHandoffMode::Observe,
            /*require_main_forward=*/false);
        if (!backend || !stream)
            return false;

        clearStochasticTargetSampleReadySlot(
            target_sample_slot,
            StochasticSampleReadyClearMode::Force);
        /*
         * `target_token` is a host policy decision, but the target slot becomes
         * its authoritative execution owner at this boundary. A scalar kernel
         * takes the value through launch parameters and avoids an asynchronous
         * H2D read from this method's stack. The readiness event below then names
         * the exact producer stream for every sidecar and verifier consumer.
         */
        if (!backend->enqueuePublishInt32ControlScalarDevice(
                target_token,
                static_cast<int32_t *>(stochastic_target_sample_tokens_dev_) +
                    target_sample_slot,
                state_.device_id.gpu_ordinal(),
                stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to publish host-resolved target token into device execution state");
            return false;
        }
        if (!recordStochasticTargetSampleReady(
                target_sample_slot,
                stream,
                /*verifier_consumer_pending=*/true, StochasticTargetPublication::NewValue))
        {
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "stochastic_target_token_device_stages",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"slot", std::to_string(target_sample_slot)},
             {"token", std::to_string(target_token)}});
        return true;
    }

    bool DeviceGraphOrchestrator::publishForcedDeviceResidentConditionToken(
        int32_t token,
        int target_sample_slot)
    {
        if (!state_.device_id.is_gpu() || target_sample_slot < 0 ||
            target_sample_slot >= stochastic_target_row_capacity_ ||
            !stochastic_target_sample_tokens_dev_ || !state_.kv_cache ||
            !state_.kv_cache->supportsDeviceResidentSequenceStatePublication())
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Forced policy token requires persistent GPU token storage and a device-owned live KV position");
            return false;
        }
        const int32_t *live_position_device =
            state_.kv_cache->deviceSequenceCachedTokenCountPtr(/*seq_idx=*/0);
        IBackend *backend = getBackendFor(state_.device_id);
        const int32_t *token_device =
            static_cast<const int32_t *>(stochastic_target_sample_tokens_dev_) +
            target_sample_slot;
        void *producer_stream = prepareMainLogitsDeviceConsumer(
            "publishForcedDeviceResidentConditionToken",
            MainLogitsHandoffMode::Observe,
            /*require_main_forward=*/false);
        if (!backend || !live_position_device || !token_device ||
            !producer_stream)
        {
            return false;
        }

        /*
         * Ordinary generation deliberately has no MTP target-sample readiness
         * events.  Publish the host policy scalar directly into the already
         * reserved device slot, then use the logical mailbox's own event as the
         * sole typed handoff. Reusing stageStochasticTargetTokenForDeviceSampling()
         * here would make a non-MTP request depend on an unallocated MTP event
         * and was the production crash behind bounded reasoning responses.
         */
        clearStochasticTargetSampleReadySlot(
            target_sample_slot,
            StochasticSampleReadyClearMode::Force);
        if (!backend->enqueuePublishInt32ControlScalarDevice(
                token,
                const_cast<int32_t *>(token_device),
                state_.device_id.gpu_ordinal(),
                producer_stream) ||
            !publishPreparedArenaGraphInput(
                BufferId::STOCHASTIC_TARGET_SAMPLE_TOKENS,
                stochastic_target_sample_tokens_dev_,
                producer_stream,
                state_.device_id,
                "forced_ordinary_condition_token"))
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Forced policy token could not enter persistent ordinary device state");
            return false;
        }

        /*
         * A forced token bypasses the sampler but remains ordinary generated
         * output. Update the same durable histogram on the token producer's
         * stream so later penalties see byte-identical serial history. With no
         * active penalties the histogram is deliberately untouched.
         */
        if (mtp_request_penalty_policy_.enabled())
        {
            if (!mtp_generated_token_counts_dev_ ||
                mtp_generated_token_count_capacity_ != state_.vocab_size ||
                !backend->enqueueCommitGenerationTokenHistoryDevice(
                    token_device,
                    state_.vocab_size,
                    mtp_generated_token_counts_dev_,
                    state_.device_id.gpu_ordinal(),
                    producer_stream) ||
                !publishPreparedArenaGraphInput(
                    BufferId::MTP_GENERATED_TOKEN_COUNTS,
                    mtp_generated_token_counts_dev_,
                    producer_stream,
                    state_.device_id,
                    "forced_generation_token_history"))
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Forced policy token could not advance device generation history");
                return false;
            }
        }

        if (!publishDeviceResidentLogicalSequenceStateFromDeviceRows(
                token_device,
                live_position_device,
                /*request_count=*/1,
                producer_stream,
                DeviceResidentLogicalStatePublicationKind::
                    TargetSampleInitialization,
                "forced_ordinary_condition"))
        {
            return false;
        }
        /*
         * The forced policy token is now the canonical next-condition owner.
         * Its scalar/history/mailbox writes were queued on the exact observed
         * logits producer and the mailbox published its own readiness event.
         * This is the final semantic consumer of the replaced logits row,
         * matching the target-sample MTP condition path. Retire the one-shot
         * handoff only after successful publication: the next retained forward
         * may have another capture stream, while every failure above leaves
         * the previous handoff owned and the durable output event intact.
         */
        (void)consumePendingLogitsStream(
            PendingLogitsStreamRole::MainDecode,
            "publishForcedDeviceResidentConditionToken");

        PerfStatsCollector::addCounter(
            "generation",
            "forced_device_resident_condition_publications",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"target_slot", std::to_string(target_sample_slot)},
             {"history", mtp_request_penalty_policy_.enabled()
                             ? "device_committed"
                             : "penalty_free"}});
        return true;
    }

    bool DeviceGraphOrchestrator::publishDeviceResidentConditionTokenToTargetSampleSlot(
        const DeviceResidentLogicalSequenceStateHandle &logical_state,
        int request_index,
        int target_sample_slot)
    {
        if (!state_.device_id.is_gpu() ||
            !logical_state.coversRequest(request_index) ||
            logical_state.device != state_.device_id ||
            target_sample_slot < 0 ||
            target_sample_slot >= stochastic_target_row_capacity_ ||
            !stochastic_target_sample_tokens_dev_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid resident condition-token target publication"
                      << " request=" << request_index
                      << " target_slot=" << target_sample_slot
                      << " runner=" << state_.device_id.toString());
            return false;
        }
        if (!device_resident_logical_sequence_state_mailbox_.ownsHandle(
                logical_state,
                live_replay_state_epoch_))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident condition-token target publication received a stale or foreign mailbox");
            return false;
        }

        const int32_t *condition_token_device =
            logical_state.nextConditionTokenDeviceForRequest(request_index);
        IBackend *backend = getBackendFor(state_.device_id);
        void *stream = explicitGPUStreamForOperation(
            "publishDeviceResidentConditionTokenToTargetSampleSlot");
        if (!condition_token_device || !backend || !stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident condition-token target publication requires a device token, backend, and explicit stream");
            return false;
        }

        /*
         * Snapshot the token before any shifted-cache consumer retargets the
         * logical-state transaction. The target sample arena is persistent, so
         * the later main graph can consume this exact token after the mailbox
         * itself has advanced to a new transaction epoch.
         */
        clearStochasticTargetSampleReadySlot(
            target_sample_slot,
            StochasticSampleReadyClearMode::Force);
        DeviceResidentLogicalStateReadScope logical_state_read(
            *this,
            stream,
            "resident_condition_token_target_publication");
        if (!logical_state_read.ready())
            return false;
        if (!backend->deviceCopyAsync(
                static_cast<int32_t *>(stochastic_target_sample_tokens_dev_) +
                    target_sample_slot,
                condition_token_device,
                sizeof(int32_t),
                state_.device_id.gpu_ordinal(),
                stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to copy resident condition token into target sample slot");
            return false;
        }
        if (!recordStochasticTargetSampleReady(
                target_sample_slot,
                stream,
                /*verifier_consumer_pending=*/true, StochasticTargetPublication::NewValue))
        {
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "resident_condition_token_target_publications",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"request_index", std::to_string(request_index)},
             {"target_slot", std::to_string(target_sample_slot)},
             {"transfer", "d2d"}});
        return true;
    }

    int DeviceGraphOrchestrator::sampleStochasticDistributionOnDevice(
        DeviceDistributionBuffer buffer,
        int slot,
        float threshold)
    {
        int32_t token = -1;
        if (!sampleStochasticDistributionOnDeviceImpl(
                buffer,
                slot,
                threshold,
                &token))
        {
            return -1;
        }
        return token;
    }

    bool DeviceGraphOrchestrator::sampleStochasticDistributionOnDeviceDeferred(
        DeviceDistributionBuffer buffer,
        int slot,
        float threshold)
    {
        return sampleStochasticDistributionOnDeviceImpl(
            buffer,
            slot,
            threshold,
            /*out_token_host=*/nullptr);
    }

    bool DeviceGraphOrchestrator::verifyStochasticDistributionsOnDevice(
        int target_slot,
        int draft_slot,
        int draft_token,
        float accept_threshold,
        float residual_threshold,
        DeviceSpeculativeVerifyResult *out)
    {
        if (!supportsDeviceStochasticMTPVerification() || !out ||
            target_slot < 0 || draft_slot < 0 ||
            target_slot >= stochastic_target_row_capacity_ ||
            draft_slot >= stochastic_draft_row_capacity_)
        {
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;
        void *stream = explicitGPUStreamForOperation("verifyStochasticDistributionsOnDevice");
        if (!stream)
            return false;

        int *target_ids = static_cast<int *>(stochastic_target_token_ids_dev_) +
                          static_cast<size_t>(target_slot) * kStochasticDistributionMaxK;
        float *target_probs = static_cast<float *>(stochastic_target_probs_dev_) +
                              static_cast<size_t>(target_slot) * kStochasticDistributionMaxK;
        int *draft_ids = static_cast<int *>(stochastic_draft_token_ids_dev_) +
                         static_cast<size_t>(draft_slot) * kStochasticDistributionMaxK;
        float *draft_probs = static_cast<float *>(stochastic_draft_probs_dev_) +
                             static_cast<size_t>(draft_slot) * kStochasticDistributionMaxK;
        const int target_top_k = stochastic_target_top_k_[static_cast<size_t>(target_slot)];
        const int draft_top_k = stochastic_draft_top_k_[static_cast<size_t>(draft_slot)];
        if (target_top_k <= 0 ||
            draft_top_k <= 0 ||
            target_top_k != draft_top_k ||
            target_top_k > static_cast<int>(kStochasticDistributionMaxK))
        {
            return false;
        }

        int *out_token_dev = static_cast<int *>(stochastic_verify_tokens_dev_) + target_slot;
        int *out_accepted_dev = static_cast<int *>(stochastic_verify_accepted_dev_) + target_slot;
        float *out_accept_prob_dev =
            static_cast<float *>(stochastic_verify_accept_probs_dev_) + target_slot;
        float *out_threshold_dev =
            static_cast<float *>(stochastic_verify_thresholds_dev_) + target_slot;

        if (!backend->enqueueSpeculativeVerifyDistributionsF32DeviceThresholds(
                target_ids,
                target_probs,
                draft_ids,
                draft_probs,
                target_top_k,
                draft_token,
                accept_threshold,
                residual_threshold,
                state_.device_id.gpu_ordinal(),
                stream,
                out_token_dev,
                out_accepted_dev,
                out_accept_prob_dev,
                out_threshold_dev))
        {
            return false;
        }

        int token = -1;
        int accepted = 0;
        float accept_probability = 0.0f;
        float threshold = 0.0f;
        /*
         * These compact result buffers are written by the verifier kernel on
         * the same stream. Fast D2H preserves the synchronization point while
         * skipping backend pointer validation that dominates tiny ROCm copies.
         */
        if (!backend->deviceToHostFast(&token, out_token_dev, sizeof(int),
                                       state_.device_id.gpu_ordinal(), stream) ||
            !backend->deviceToHostFast(&accepted, out_accepted_dev, sizeof(int),
                                       state_.device_id.gpu_ordinal(), stream) ||
            !backend->deviceToHostFast(&accept_probability, out_accept_prob_dev, sizeof(float),
                                       state_.device_id.gpu_ordinal(), stream) ||
            !backend->deviceToHostFast(&threshold, out_threshold_dev, sizeof(float),
                                       state_.device_id.gpu_ordinal(), stream))
        {
            return false;
        }

        out->token = token;
        out->accepted = accepted != 0;
        out->accept_probability = accept_probability;
        out->accept_threshold = threshold;
        return token >= 0;
    }

    bool DeviceGraphOrchestrator::verifyStochasticDistributionsBatchOnDevice(
        int first_target_slot,
        int first_draft_slot,
        const int32_t *draft_tokens,
        const float *accept_thresholds,
        const float *residual_thresholds,
        int row_count,
        DeviceSpeculativeVerifyResult *out)
    {
        if (!supportsDeviceStochasticMTPVerification() ||
            first_target_slot < 0 || first_draft_slot < 0 ||
            row_count <= 0 ||
            row_count > mtp_max_draft_depth_ ||
            first_target_slot + row_count > stochastic_target_row_capacity_ ||
            first_draft_slot + row_count > stochastic_draft_row_capacity_ ||
            !draft_tokens || !accept_thresholds || !residual_thresholds || !out)
        {
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;
        /*
         * The verifier consumes compact target distributions built immediately
         * after the main decode row.  Prefer that producer stream so the verifier
         * is naturally ordered behind the distribution build without a host sync.
         * A null producer stream means the row was either rebuilt through a
         * synchronized path or is being exercised by a non-graph unit runner.
         */
        void *stream =
            stochastic_target_distribution_streams_[static_cast<size_t>(first_target_slot)];
        if (!stream)
        {
            stream = explicitGPUStreamForOperation(
                "verifyStochasticDistributionsBatchOnDevice");
        }
        if (!stream)
            return false;

        const int target_top_k =
            stochastic_target_top_k_[static_cast<size_t>(first_target_slot)];
        if (target_top_k <= 0 ||
            target_top_k > static_cast<int>(kStochasticDistributionMaxK))
        {
            return false;
        }
        bool uses_one_hot_draft_tokens = false;
        bool uses_draft_distribution_rows = false;
        for (int row = 0; row < row_count; ++row)
        {
            const int target_slot = first_target_slot + row;
            const int draft_slot = first_draft_slot + row;
            const int draft_top_k =
                stochastic_draft_top_k_[static_cast<size_t>(draft_slot)];
            if (stochastic_target_top_k_[static_cast<size_t>(target_slot)] != target_top_k)
            {
                return false;
            }
            if (draft_top_k == target_top_k)
            {
                uses_draft_distribution_rows = true;
            }
            else if (draft_top_k <= 0)
            {
                uses_one_hot_draft_tokens = true;
            }
            else
            {
                return false;
            }
        }

        if (uses_one_hot_draft_tokens == uses_draft_distribution_rows)
        {
            return false;
        }

        if (uses_one_hot_draft_tokens &&
            !waitForRequiredStochasticDraftSampleReadyRange(
                first_draft_slot,
                row_count,
                stream,
                "decode_equivalent_stochastic_verifier"))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to order decode-equivalent stochastic verifier after sampled draft tokens");
            return false;
        }

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
            stochastic_draft_sample_tokens_dev_
                ? static_cast<int *>(stochastic_draft_sample_tokens_dev_) +
                      first_draft_slot
                : nullptr;
        auto *out_token_dev =
            static_cast<int *>(stochastic_verify_tokens_dev_) + first_target_slot;
        auto *out_accepted_dev =
            static_cast<int *>(stochastic_verify_accepted_dev_) + first_target_slot;
        auto *out_accept_prob_dev =
            static_cast<float *>(stochastic_verify_accept_probs_dev_) + first_target_slot;
        auto *out_threshold_dev =
            static_cast<float *>(stochastic_verify_thresholds_dev_) + first_target_slot;

        bool verify_enqueued = false;
        if (uses_one_hot_draft_tokens)
        {
            if (!sampled_draft_tokens)
                return false;
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
                    out_threshold_dev);
        }
        else
        {
            verify_enqueued =
                backend->enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatch(
                    target_ids,
                    target_probs,
                    draft_ids,
                    draft_probs,
                    target_top_k,
                    static_cast<int>(kStochasticDistributionMaxK),
                    draft_tokens,
                    accept_thresholds,
                    residual_thresholds,
                    row_count,
                    state_.device_id.gpu_ordinal(),
                    stream,
                    out_token_dev,
                    out_accepted_dev,
                    out_accept_prob_dev,
                    out_threshold_dev);
        }
        if (!verify_enqueued)
        {
            return false;
        }

        std::vector<int> tokens(static_cast<size_t>(row_count));
        std::vector<int> accepted(static_cast<size_t>(row_count));
        std::vector<float> accept_probabilities(
            static_cast<size_t>(row_count));
        std::vector<float> thresholds(static_cast<size_t>(row_count));
        if (!backend->deviceToHostFast(tokens.data(), out_token_dev,
                                       sizeof(int) * static_cast<size_t>(row_count),
                                       state_.device_id.gpu_ordinal(), stream) ||
            !backend->deviceToHostFast(accepted.data(), out_accepted_dev,
                                       sizeof(int) * static_cast<size_t>(row_count),
                                       state_.device_id.gpu_ordinal(), stream) ||
            !backend->deviceToHostFast(accept_probabilities.data(), out_accept_prob_dev,
                                       sizeof(float) * static_cast<size_t>(row_count),
                                       state_.device_id.gpu_ordinal(), stream) ||
            !backend->deviceToHostFast(thresholds.data(), out_threshold_dev,
                                       sizeof(float) * static_cast<size_t>(row_count),
                                       state_.device_id.gpu_ordinal(), stream))
        {
            return false;
        }

        for (int row = 0; row < row_count; ++row)
        {
            out[row].accepted = accepted[static_cast<size_t>(row)] != 0;
            out[row].token = tokens[static_cast<size_t>(row)];
            out[row].accept_probability =
                accept_probabilities[static_cast<size_t>(row)];
            out[row].accept_threshold = thresholds[static_cast<size_t>(row)];
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "stochastic_verify_batch_rows",
            static_cast<double>(row_count),
            "decode",
            state_.device_id.toString(),
            {{"top_k", std::to_string(target_top_k)},
             {"draft", uses_one_hot_draft_tokens ? "one_hot" : "distribution"}});
        return true;
    }

    bool DeviceGraphOrchestrator::verifyStochasticDistributionsBatchOutcomeOnDevice(
        int first_target_slot,
        int first_draft_slot,
        const int32_t *draft_tokens,
        const float *accept_thresholds,
        const float *residual_thresholds,
        int row_count,
        int32_t first_token,
        const int32_t *stop_tokens,
        int stop_token_count,
        int bonus_target_slot,
        float bonus_threshold,
        DeviceSpeculativeVerifyBatchOutcome *out,
        uint64_t inverse_sample_seed,
        int inverse_sample_first_logical_position,
        bool use_vllm_probability_rejection)
    {
        /*
         * Keep the old host-visible scalar API as a compatibility bridge over
         * the resident outcome contract.  This makes single-request decode and
         * request-batched decode use one ordering/stream path, and gives Phase
         * 10 publication work a single device-handle interface to consume.
         */
        DeviceSpeculativeOutcomeHandle handle;
        if (!verifyStochasticDistributionsBatchOutcomeOnDeviceResident(
                first_target_slot,
                first_draft_slot,
                draft_tokens,
                accept_thresholds,
                residual_thresholds,
                row_count,
                first_token,
                stop_tokens,
                stop_token_count,
                bonus_target_slot,
                bonus_threshold,
                &handle,
                inverse_sample_seed,
                inverse_sample_first_logical_position,
                use_vllm_probability_rejection))
        {
            return false;
        }
        return copyDeviceSpeculativeOutcomesToHostForDiagnostics(handle, out);
    }

    bool DeviceGraphOrchestrator::verifyStochasticDistributionsBatchOutcomeOnDeviceFirstToken(
        int first_target_slot,
        int first_draft_slot,
        const int32_t *draft_tokens,
        const float *accept_thresholds,
        const float *residual_thresholds,
        int row_count,
        int first_target_sample_slot,
        const int32_t *stop_tokens,
        int stop_token_count,
        int bonus_target_slot,
        float bonus_threshold,
        DeviceSpeculativeVerifyBatchOutcome *out,
        uint64_t inverse_sample_seed,
        int inverse_sample_first_logical_position,
        bool use_vllm_probability_rejection)
    {
        DeviceSpeculativeOutcomeHandle handle;
        if (!verifyStochasticDistributionsBatchOutcomeOnDeviceFirstTokenResident(
                first_target_slot,
                first_draft_slot,
                draft_tokens,
                accept_thresholds,
                residual_thresholds,
                row_count,
                first_target_sample_slot,
                stop_tokens,
                stop_token_count,
                bonus_target_slot,
                bonus_threshold,
                &handle,
                inverse_sample_seed,
                inverse_sample_first_logical_position,
                use_vllm_probability_rejection))
        {
            return false;
        }
        return copyDeviceSpeculativeOutcomesToHostForDiagnostics(handle, out);
    }

    bool DeviceGraphOrchestrator::verifyStochasticDistributionsRequestBatchOutcomesOnDevice(
        const DeviceStochasticBatchOutcomeRequest *requests,
        int request_count,
        DeviceSpeculativeVerifyBatchOutcome *outcomes)
    {
        DeviceSpeculativeOutcomeHandle handle;
        if (!verifyStochasticDistributionsRequestBatchOutcomesOnDeviceResident(
                requests,
                request_count,
                &handle))
        {
            return false;
        }
        return copyDeviceSpeculativeOutcomesToHostForDiagnostics(handle, outcomes);
    }

    bool DeviceGraphOrchestrator::verifyStochasticDistributionsRequestBatchOutcomesOnDeviceResident(
        const DeviceStochasticBatchOutcomeRequest *requests,
        int request_count,
        DeviceSpeculativeOutcomeHandle *out_handle)
    {
        using namespace sampling_math;
        if (out_handle)
            *out_handle = DeviceSpeculativeOutcomeHandle{};
        if (!requests || !out_handle ||
            request_count <= 0 ||
            request_count > stochastic_batch_output_request_capacity_ ||
            !supportsDeviceStochasticMTPVerification() ||
            !stochastic_batch_output_tokens_dev_ ||
            !stochastic_batch_output_meta_dev_)
        {
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        /*
         * All target distributions for the scheduled request batch have already
         * been queued on the verifier stream.  Consume that stream once and keep
         * every per-request verify/bonus/summary launch ordered behind it, then
         * perform exactly one compact D2H copy for all request outcomes.
         */
        void *stream = consumePendingLogitsStream(
            PendingLogitsStreamRole::AllPositionVerifier,
            "verifyStochasticDistributionsRequestBatchOutcomesOnDevice");
        if (!stream)
        {
            stream = explicitGPUStreamForOperation(
                "verifyStochasticDistributionsRequestBatchOutcomesOnDevice");
        }
        if (!stream)
            return false;

        bool uses_resident_threshold_positions = false;
        bool uses_verifier_base_snapshot_positions = false;
        bool all_requests_serial_sample_equivalent = true;
        bool all_requests_captured_rejection = true;
        for (int request_idx = 0; request_idx < request_count; ++request_idx)
        {
            const DeviceStochasticDrawPositionSource source =
                requests[request_idx].draw_position_source;
            uses_resident_threshold_positions =
                uses_resident_threshold_positions ||
                source == DeviceStochasticDrawPositionSource::ResidentLogicalState;
            uses_verifier_base_snapshot_positions =
                uses_verifier_base_snapshot_positions ||
                source == DeviceStochasticDrawPositionSource::VerifierBaseSnapshot;
            all_requests_serial_sample_equivalent =
                all_requests_serial_sample_equivalent &&
                requests[request_idx].serial_sample_equivalent;
            const auto &request = requests[request_idx];
            all_requests_captured_rejection = all_requests_captured_rejection &&
                request.use_vllm_probability_rejection && !request.serial_sample_equivalent &&
                request.derive_thresholds_from_seed && request.inverse_sample_seed != 0 &&
                source == DeviceStochasticDrawPositionSource::VerifierBaseSnapshot;
        }

        DeviceResidentLogicalSequenceStateHandle threshold_position_state;
        std::optional<DeviceResidentLogicalStateReadScope>
            threshold_position_read;
        if (uses_resident_threshold_positions)
        {
            threshold_position_state = deviceResidentLogicalSequenceState();
            if (!threshold_position_state.valid() ||
                threshold_position_state.device != state_.device_id ||
                threshold_position_state.request_count < request_count)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Stochastic request batch could not bind its resident threshold-position mailbox");
                return false;
            }
            threshold_position_read.emplace(
                *this,
                stream,
                "stochastic_request_batch_threshold_positions");
            if (!threshold_position_read->ready())
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Stochastic request batch could not order its resident threshold-position mailbox");
                return false;
            }
        }

        const MTPSpecDecodeMetadataDevicePointers &publication_metadata =
            mtp_spec_decode_metadata_binding_.devicePointers();
        if (uses_verifier_base_snapshot_positions &&
            (!mtp_publication_base_cache_snapshot_ready_ ||
             mtp_publication_base_cache_snapshot_request_count_ < request_count ||
             !publication_metadata.base_cached_tokens))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Seeded stochastic verifier could not bind the pre-verifier device base-count snapshot");
            return false;
        }

        /*
         * Outcome reduction advances a maintenance cadence only for Dynamic
         * homogeneous residency. Static overlay remains device-authoritative
         * for sampling and publication while carrying no movement clock.
         */
        const bool requires_rebalance_controller =
            usesParticipantLocalDeviceMoERebalanceController();
        std::string rebalance_controller_error;
        DeviceMoERebalanceGraphControllerState *rebalance_controller =
            requires_rebalance_controller
                ? deviceMoERebalanceControllerStateDevice(
                      &rebalance_controller_error)
                : nullptr;
        if (requires_rebalance_controller && !rebalance_controller)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-owned stochastic verifier could not bind the MoE maintenance controller: "
                      << rebalance_controller_error);
            return false;
        }
        auto *rebalance_controller_bytes =
            reinterpret_cast<std::byte *>(rebalance_controller);
        auto *decode_rounds_committed_device =
            rebalance_controller
                ? reinterpret_cast<uint32_t *>(
                      rebalance_controller_bytes +
                      offsetof(
                          DeviceMoERebalanceGraphControllerState,
                          decode_rounds_committed))
                : nullptr;
        auto *decode_rounds_until_maintenance_device =
            rebalance_controller
                ? reinterpret_cast<uint32_t *>(
                      rebalance_controller_bytes +
                      offsetof(
                          DeviceMoERebalanceGraphControllerState,
                          decode_rounds_until_maintenance))
                : nullptr;
        auto *maintenance_due_device =
            rebalance_controller
                ? reinterpret_cast<uint32_t *>(
                      rebalance_controller_bytes +
                      offsetof(
                          DeviceMoERebalanceGraphControllerState,
                          maintenance_due))
                : nullptr;
        auto *decode_boundary_advanced_device =
            rebalance_controller
                ? reinterpret_cast<uint32_t *>(
                      rebalance_controller_bytes +
                      offsetof(
                          DeviceMoERebalanceGraphControllerState,
                          decode_boundary_advanced))
                : nullptr;

        /*
         * Captured verifier preparation consumed the controller event and
         * published the exact transaction budget before any grouped forward
         * node ran. Reaching summary with another ready event would mean two
         * consumers claimed the same transaction, so fail rather than guessing
         * which controller version the verifier observed.
         */
        if (!device_generation_storage_.validFor(request_count) ||
            device_generation_storage_.active_request_count != request_count ||
            !device_generation_state_ready_.handoff.borrowed())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-owned stochastic verifier did not receive one exclusively consumed preparation transaction");
            return false;
        }

        std::shared_ptr<void> producer_start_timing_event;
        std::shared_ptr<void> producer_stop_timing_event;
        const bool collect_producer_gpu_timing =
            PerfStatsCollector::isDomainEnabled("mtp") &&
            state_.device_id.is_gpu();
        if (collect_producer_gpu_timing)
        {
            if (!acquirePersistentMTPGpuTimingEvents(
                    "stochastic_request_batch_compact_outcome_summary",
                    &producer_start_timing_event,
                    &producer_stop_timing_event))
            {
                return false;
            }
            if (!backend->recordEvent(
                    producer_start_timing_event.get(),
                    state_.device_id.gpu_ordinal(),
                    stream))
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "stochastic_request_batch_summary_gpu_timing_record_failures",
                    1.0,
                    "decode",
                    state_.device_id.toString(),
                    {{"event", "start"}});
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to record the preallocated stochastic outcome timing start event");
                return false;
            }
        }

        const bool captured_outcome =
            all_requests_serial_sample_equivalent || all_requests_captured_rejection;
        if (captured_outcome)
        {
            for (int request_idx = 0;
                 request_idx < request_count;
                 ++request_idx)
            {
                if (!waitForRequiredStochasticDraftSampleReadyRange(
                        requests[request_idx].first_draft_slot,
                        requests[request_idx].row_count,
                        stream,
                        "captured_stochastic_outcome"))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Captured stochastic outcome could not consume deferred draft readiness");
                    return false;
                }
            }

            std::string captured_error;
            if (!materializeMTPStochasticOutcomeGraph(
                    requests,
                    request_count,
                    &captured_error) ||
                !executeMTPStochasticOutcomeCaptured(
                    stream,
                    request_count,
                    requests[0].row_count,
                    &captured_error))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Captured stochastic outcome failed: "
                          << captured_error);
                return false;
            }
            PerfStatsCollector::addCounter(
                "mtp",
                all_requests_serial_sample_equivalent
                    ? "stochastic_serial_equivalent_captured_outcomes"
                    : "stochastic_probability_rejection_captured_outcomes",
                static_cast<double>(request_count),
                "decode",
                state_.device_id.toString(),
                {{"comparison_rows", std::to_string(requests[0].row_count)},
                 {"fused_sample_summary", all_requests_serial_sample_equivalent ? "true" : "false"}});
        }
        else
        {
            for (int request_idx = 0;
                 request_idx < request_count;
                 ++request_idx)
            {
                const DeviceStochasticBatchOutcomeRequest &request =
                    requests[request_idx];
            if (request.row_count <= 0 ||
                request.row_count > mtp_max_draft_depth_ ||
                request.stop_token_count < 0 ||
                request.stop_token_count > kSpeculativeBatchMaxStopTokens)
            {
                return false;
            }

            const int32_t *stop_tokens =
                request.stop_token_count > 0
                    ? request.stop_tokens.data()
                    : nullptr;
            const int32_t *draft_tokens = request.hostDraftTokensOrNull();
            const bool source_is_explicit =
                request.draw_position_source ==
                DeviceStochasticDrawPositionSource::ExplicitThresholds;
            const bool source_is_host_position =
                request.draw_position_source ==
                DeviceStochasticDrawPositionSource::HostLogicalPosition;
            const bool source_is_resident_state =
                request.draw_position_source ==
                DeviceStochasticDrawPositionSource::ResidentLogicalState;
            const bool source_is_verifier_snapshot =
                request.draw_position_source ==
                DeviceStochasticDrawPositionSource::VerifierBaseSnapshot;
            const bool derive_thresholds_from_seed =
                request.derive_thresholds_from_seed &&
                request.inverse_sample_seed != 0 &&
                (request.serial_sample_equivalent ||
                 request.use_vllm_probability_rejection) &&
                !source_is_explicit &&
                ((source_is_host_position &&
                  request.inverse_sample_first_logical_position >= 0) ||
                 ((source_is_resident_state || source_is_verifier_snapshot) &&
                  request.inverse_sample_first_logical_position < 0));
            if (request.derive_thresholds_from_seed !=
                    derive_thresholds_from_seed ||
                (!request.derive_thresholds_from_seed && !source_is_explicit))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Seeded stochastic descriptor has an inconsistent draw-position authority for request="
                          << request.request_id);
                return false;
            }
            const size_t required_rows =
                static_cast<size_t>(request.row_count);
            if ((!request.use_device_draft_tokens &&
                 request.draft_tokens.size() < required_rows) ||
                (source_is_explicit &&
                 request.serial_sample_equivalent &&
                 request.sample_thresholds.size() < required_rows) ||
                (source_is_explicit &&
                 !request.serial_sample_equivalent &&
                 (request.accept_thresholds.size() < required_rows ||
                  request.residual_thresholds.size() < required_rows)))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Stochastic descriptor host rows do not cover request="
                          << request.request_id
                          << " row_count=" << request.row_count);
                return false;
            }
            const int32_t *threshold_base_position_device = nullptr;
            if (source_is_resident_state)
            {
                if (!derive_thresholds_from_seed ||
                    !threshold_position_state.coversRequest(request.request_id))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Resident stochastic threshold descriptor is malformed for request="
                              << request.request_id);
                    return false;
                }
                threshold_base_position_device =
                    threshold_position_state.targetPositionDeviceForRequest(
                        request.request_id);
            }
            else if (source_is_verifier_snapshot)
            {
                if (!derive_thresholds_from_seed ||
                    request.request_id < 0 ||
                    request.request_id >=
                        mtp_publication_base_cache_snapshot_request_count_)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Verifier-snapshot stochastic threshold descriptor is malformed for request="
                              << request.request_id);
                    return false;
                }
                threshold_base_position_device =
                    publication_metadata.base_cached_tokens + request.request_id;
            }
            const float *accept_thresholds =
                derive_thresholds_from_seed || request.serial_sample_equivalent
                    ? nullptr
                    : request.accept_thresholds.data();
            const float *residual_thresholds =
                derive_thresholds_from_seed || request.serial_sample_equivalent
                    ? nullptr
                    : request.residual_thresholds.data();
            const float *sample_thresholds =
                request.serial_sample_equivalent
                    ? (derive_thresholds_from_seed
                           ? nullptr
                           : request.sample_thresholds.data())
                    : nullptr;
            DeviceSpeculativeVerifyBatchOutcome ignored_host_outcome;

            const bool ok =
                verifyStochasticDistributionsBatchOutcomeOnDeviceCommon(
                    request.first_target_slot,
                    request.first_draft_slot,
                    draft_tokens,
                    accept_thresholds,
                    residual_thresholds,
                    sample_thresholds,
                    request.row_count,
                    request.first_token,
                    request.first_target_sample_slot,
                    request.token_row_offset,
                    request.token_row_stride,
                    request.first_token_from_device,
                    stop_tokens,
                    request.stop_token_count,
                    request.bonus_target_slot,
                    request.bonus_threshold,
                    &ignored_host_outcome,
                    request.inverse_sample_seed,
                    request.inverse_sample_first_logical_position,
                    threshold_base_position_device,
                    /*threshold_position_offset=*/1,
                    request.use_vllm_probability_rejection,
                    request.serial_sample_equivalent,
                    device_generation_storage_.controlForRequest(request_idx),
                    /*output_request_slot=*/request_idx,
                    stream,
                    /*copy_summary_to_host=*/false);
                if (!ok)
                    return false;
            }
        }

        if (!captured_outcome &&
            rebalance_controller &&
            !backend->enqueueAdvanceSpeculativeCommitBoundary(
                stochastic_batch_output_meta_dev_,
                request_count,
                kSpeculativeBatchMetaCount,
                decode_rounds_committed_device,
                decode_rounds_until_maintenance_device,
                maintenance_due_device,
                decode_boundary_advanced_device,
                state_.device_id.gpu_ordinal(),
                stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-owned stochastic verifier could not advance the MoE maintenance boundary");
            return false;
        }

        if (producer_stop_timing_event)
        {
            if (!backend->recordEvent(
                    producer_stop_timing_event.get(),
                    state_.device_id.gpu_ordinal(),
                    stream))
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "stochastic_request_batch_summary_gpu_timing_record_failures",
                    1.0,
                    "decode",
                    state_.device_id.toString(),
                    {{"event", "stop"}});
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to record the preallocated stochastic outcome timing stop event");
                return false;
            }
        }

        std::shared_ptr<void> response_ready_event =
            acquirePersistentMTPOutcomeReadyEvent(
                "stochastic_request_batch_compact_outcome");
        if (!response_ready_event)
        {
            return false;
        }
        int logical_verifier_rows_per_request = 0;
        for (int request_index = 0;
             request_index < request_count;
             ++request_index)
        {
            logical_verifier_rows_per_request =
                std::max(
                    logical_verifier_rows_per_request,
                    requests[request_index].row_count + 1);
        }
        if (!sealDeviceSpeculativeOutcomeVerifierGeometry(
                out_handle,
                request_count,
                logical_verifier_rows_per_request,
                "stochastic_request_batch_device_outcome"))
        {
            return false;
        }
        /*
         * Record before any caller enqueues state publication on this producer
         * stream.  The compact outcome rows are immutable after the verifier
         * summary kernels above, so the response bridge can wait only for this
         * event and avoid draining later live-state publication work.
         */
        if (!DeviceEventEdge::at(
                 DeviceTimelinePoint::CompactSpeculativeResponseReady)
                 .from(DeviceTimelineRole::VerifierSummary)
                 .publish(
                     *backend,
                     state_.device_id,
                     response_ready_event.get(),
                     stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to record stochastic outcome response-ready event");
            return false;
        }

        materialized_mtp_verifier_device_token_batch_ = {};
        if (!recordDeviceResidentMTPTransactionMutation(
                request_count,
                stream,
                "stochastic_request_batch_device_outcome"))
        {
            return false;
        }
        out_handle->output_tokens_device =
            static_cast<const int32_t *>(stochastic_batch_output_tokens_dev_);
        out_handle->meta_device =
            static_cast<int *>(stochastic_batch_output_meta_dev_);
        out_handle->output_token_stride =
            stochastic_batch_output_token_stride_;
        out_handle->meta_stride = kSpeculativeBatchMetaCount;
        out_handle->device = state_.device_id;
        out_handle->stream = stream;
        out_handle->response_ready_event = std::move(response_ready_event);
        out_handle->mtp_transaction =
            currentDeviceResidentMTPTransactionLease();
        out_handle->device_generation_controller_owned = true;
        out_handle->sampling_mode = DeviceGenerationSamplingMode::Stochastic;
        out_handle->local_tp_participant_complete =
            publishesFullVocabularyMTPLogits() &&
            graph_builder_ &&
            graph_builder_->config().tp_ctx &&
            graph_builder_->config().tp_ctx->isLocal() &&
            graph_builder_->config().tp_ctx->degree() > 1;
        out_handle->producer_start_timing_event =
            std::move(producer_start_timing_event);
        out_handle->producer_stop_timing_event =
            std::move(producer_stop_timing_event);
        return out_handle->valid();
    }

    bool DeviceGraphOrchestrator::copyDeviceSpeculativeOutcomesToHostForDiagnostics(
        const DeviceSpeculativeOutcomeHandle &handle,
        DeviceSpeculativeVerifyBatchOutcome *outcomes)
    {
        using namespace sampling_math;
        if (!handle.valid() ||
            !outcomes ||
            handle.device != state_.device_id ||
            handle.request_count > stochastic_batch_output_request_capacity_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Compact response materialization rejected its input contract: device="
                      << state_.device_id.toString()
                      << " handle_valid=" << handle.valid()
                      << " outcomes=" << static_cast<const void *>(outcomes)
                      << " handle_device=" << handle.device.toString()
                      << " request_count=" << handle.request_count
                      << " request_capacity="
                      << stochastic_batch_output_request_capacity_
                      << " output_stride=" << handle.output_token_stride
                      << " meta_stride=" << handle.meta_stride
                      << " stream=" << handle.stream
                      << " ready_event=" << handle.response_ready_event.get());
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Compact response materialization found no backend for "
                      << state_.device_id.toString());
            return false;
        }

        const int request_count = handle.request_count;
        const size_t output_token_elements =
            static_cast<size_t>(request_count) *
            static_cast<size_t>(handle.output_token_stride);
        const size_t meta_elements =
            static_cast<size_t>(request_count) *
            static_cast<size_t>(handle.meta_stride);
        std::vector<int32_t> cpu_output_tokens_storage;
        std::vector<int> cpu_meta_storage;
        int32_t *output_tokens = nullptr;
        int *meta = nullptr;

        if (handle.device.is_gpu())
        {
            if (!stochastic_batch_output_host_scratch_ ||
                !stochastic_batch_output_host_scratch_->canServe(
                    request_count,
                    handle.output_token_stride,
                    handle.meta_stride))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Missing pinned stochastic outcome host scratch for "
                          << request_count << " requests on " << state_.device_id.toString());
                return false;
            }
            output_tokens = stochastic_batch_output_host_scratch_->output_tokens;
            meta = stochastic_batch_output_host_scratch_->meta;
        }
        else
        {
            cpu_output_tokens_storage.assign(output_token_elements, -1);
            cpu_meta_storage.assign(meta_elements, 0);
            output_tokens = cpu_output_tokens_storage.data();
            meta = cpu_meta_storage.data();
        }

        {
            PerfStatsCollector::ScopedTimer total_timer(
                "mtp",
                "stochastic_request_batch_summary_d2h_sync",
                "decode",
                state_.device_id.toString(),
                {{"requests", std::to_string(request_count)}});
            /*
             * Queue both compact copies on a dedicated response bridge stream.
             * Host response materialization is already the ownership handoff for
             * emitted tokens, so split the unavoidable wait into two pieces:
             * response-ready dependency wait and the actual tiny D2H copy wait.
             * Without this split, deferred verifier graph execution shows up as
             * "D2H wait" and hides the real kernel target.
             */
            void *copy_stream = handle.stream;
            if (handle.device.is_gpu())
            {
                if (!stochastic_outcome_response_bridge_stream_)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Compact response materialization has no preallocated result stream");
                    return false;
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "stochastic_request_batch_summary_bridge_stream_reuses",
                    1.0,
                    "decode",
                    state_.device_id.toString());
                copy_stream = stochastic_outcome_response_bridge_stream_.get();
                {
                    PerfStatsCollector::ScopedTimer dependency_timer(
                        "mtp",
                        "stochastic_request_batch_summary_response_dependency_enqueue",
                        "decode",
                        state_.device_id.toString(),
                        {{"requests", std::to_string(request_count)}});
                    /*
                     * The response bridge is a GPU consumer of the compact
                     * verifier result. Express that producer/consumer edge on
                     * device; a host event wait here would serialize launch
                     * submission and then make the final D2H stream wait below
                     * pay for the same work a second time.
                     */
                    if (!DeviceEventEdge::at(
                             DeviceTimelinePoint::CompactSpeculativeResponseReady)
                             .from(DeviceTimelineRole::VerifierSummary)
                             .to(DeviceTimelineRole::HostResultBridge)
                             .enqueueWait(
                                 *backend,
                                 state_.device_id,
                                 handle.response_ready_event.get(),
                                 handle.stream,
                                 copy_stream))
                    {
                        LOG_ERROR("[DeviceGraphOrchestrator] Failed to enqueue stochastic outcome response-ready dependency");
                        return false;
                    }
                }
            }
            {
                PerfStatsCollector::ScopedTimer enqueue_timer(
                    "mtp",
                    "stochastic_request_batch_summary_d2h_enqueue",
                    "decode",
                    state_.device_id.toString(),
                    {{"requests", std::to_string(request_count)}});
                if (!backend->deviceToHostOnStream(
                        output_tokens,
                        handle.output_tokens_device,
                        sizeof(int32_t) * output_token_elements,
                        state_.device_id.gpu_ordinal(),
                        copy_stream) ||
                    !backend->deviceToHostOnStream(
                        meta,
                        handle.meta_device,
                        sizeof(int) * meta_elements,
                        state_.device_id.gpu_ordinal(),
                        copy_stream))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Compact response materialization failed to enqueue token/meta D2H copies for "
                              << request_count << " requests on "
                              << state_.device_id.toString());
                    return false;
                }
            }
            {
                PerfStatsCollector::ScopedTimer wait_timer(
                    "mtp",
                    "stochastic_request_batch_summary_d2h_wait",
                    "decode",
                    state_.device_id.toString(),
                    {{"requests", std::to_string(request_count)}});
                if (!backend->synchronizeStream(
                        copy_stream,
                        state_.device_id.gpu_ordinal()))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Compact response materialization failed while waiting for the final response bridge on "
                              << state_.device_id.toString());
                    return false;
                }
            }
            drainPendingGpuTimingMeasurements(backend);
            if (handle.producer_start_timing_event &&
                handle.producer_stop_timing_event)
            {
                float elapsed_ms = 0.0f;
                if (backend->eventElapsedTimeMs(
                        handle.producer_start_timing_event.get(),
                        handle.producer_stop_timing_event.get(),
                        state_.device_id.gpu_ordinal(),
                        &elapsed_ms))
                {
                    const double clamped_ms =
                        std::max(0.0, static_cast<double>(elapsed_ms));
                    PerfStatsCollector::recordTimingNs(
                        "mtp",
                        "stochastic_request_batch_summary_gpu_reducer",
                        static_cast<uint64_t>(clamped_ms * 1000000.0),
                        "decode",
                        state_.device_id.toString(),
                        {{"requests", std::to_string(request_count)}});
                }
                else
                {
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "stochastic_request_batch_summary_gpu_timing_read_failures",
                        1.0,
                        "decode",
                        state_.device_id.toString(),
                        {{"requests", std::to_string(request_count)}});
                }
            }
        }

        for (int request_idx = 0; request_idx < request_count; ++request_idx)
        {
            const auto token_base =
                static_cast<size_t>(request_idx) *
                static_cast<size_t>(handle.output_token_stride);
            const auto meta_base =
                static_cast<size_t>(request_idx) *
                static_cast<size_t>(handle.meta_stride);
            const int *request_meta = meta + meta_base;
            if (request_meta[kSpecBatchMetaOk] == 0 ||
                request_meta[kSpecBatchMetaOutputCount] < 0 ||
                request_meta[kSpecBatchMetaOutputCount] >
                    handle.output_token_stride)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Compact response metadata is invalid for request="
                          << request_idx
                          << " ok=" << request_meta[kSpecBatchMetaOk]
                          << " output_count="
                          << request_meta[kSpecBatchMetaOutputCount]
                          << " output_stride=" << handle.output_token_stride
                          << " accepted_prefix="
                          << request_meta[kSpecBatchMetaAcceptedSpeculativePrefix]
                          << " state_commit_count="
                          << request_meta[kSpecBatchMetaTargetVerifierStateCommitCount]
                          << " ready_token="
                          << request_meta[kSpecBatchMetaReadyToken]);
                return false;
            }

            DeviceSpeculativeVerifyBatchOutcome &outcome =
                outcomes[request_idx];
            outcome = DeviceSpeculativeVerifyBatchOutcome(
                static_cast<size_t>(handle.output_token_stride));
            for (int output_idx = 0;
                 output_idx < request_meta[kSpecBatchMetaOutputCount];
                 ++output_idx)
            {
                const int32_t token =
                    output_tokens[token_base + static_cast<size_t>(output_idx)];
                if (token < 0)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Request-batched stochastic summary produced invalid token="
                              << token << " request=" << request_idx
                              << " output_index=" << output_idx);
                    return false;
                }
                outcome.output_tokens[static_cast<size_t>(output_idx)] = token;
            }
            outcome.ok = true;
            outcome.output_token_count =
                request_meta[kSpecBatchMetaOutputCount];
            outcome.accepted_speculative_prefix =
                request_meta[kSpecBatchMetaAcceptedSpeculativePrefix];
            outcome.target_verifier_state_commit_count =
                request_meta[kSpecBatchMetaTargetVerifierStateCommitCount];
            outcome.ready_token = request_meta[kSpecBatchMetaReadyToken];
            outcome.rejected_verified_token =
                request_meta[kSpecBatchMetaRejectedVerifiedToken];
            outcome.stopped_on_output =
                request_meta[kSpecBatchMetaStoppedOnOutput] != 0;
            outcome.all_speculative_accepted =
                request_meta[kSpecBatchMetaAllSpeculativeAccepted] != 0;
            outcome.consumed_verifier_rows =
                request_meta[kSpecBatchMetaConsumedVerifierRows];
            outcome.sampled_terminal =
                request_meta[kSpecBatchMetaSampledTerminal] != 0;
            outcome.commit_boundary_clipped =
                request_meta[kSpecBatchMetaCommitBoundaryClipped] != 0;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "stochastic_verify_request_batch_outcomes",
            static_cast<double>(request_count),
            "decode",
            state_.device_id.toString());
        return true;
    }
} // namespace llaminar2
