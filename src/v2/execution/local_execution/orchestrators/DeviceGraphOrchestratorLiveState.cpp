/**
 * @file DeviceGraphOrchestratorLiveState.cpp
 * @brief DeviceGraphOrchestrator method shard (37248..41845).
 */
#include "DeviceGraphOrchestratorInternal.h"

namespace llaminar2
{

    bool DeviceGraphOrchestrator::commitMTPShiftedRowsFromPartialForward(
        const int32_t *tokens,
        int token_count,
        int already_appended_tokens,
        int main_forward_token_count,
        bool allow_speculative_discard,
        int position_offset_override,
        int already_appended_shifted_kv_tokens)
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
            return true;
        if (token_count <= already_appended_tokens)
            return true;
        if (!tokens || token_count <= 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP shifted-row commit requires accepted tokens");
            return false;
        }
        if (already_appended_tokens < 1)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP shifted-row commit expects the sidecar to own the first accepted token");
            return false;
        }
        if (!state_.hidden)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP shifted-row commit requires hidden rows from the last main forward");
            return false;
        }
        if (state_.positions.empty())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP shifted-row commit requires position state");
            return false;
        }
        const int catchup_token_count = token_count - already_appended_tokens;
        const int resident_shifted_kv_tokens =
            already_appended_shifted_kv_tokens >= 0
                ? already_appended_shifted_kv_tokens
                : already_appended_tokens;
        if (resident_shifted_kv_tokens < 0 ||
            resident_shifted_kv_tokens > already_appended_tokens)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP shifted-row commit received invalid resident shifted-KV count="
                      << resident_shifted_kv_tokens
                      << " already_appended_tokens=" << already_appended_tokens);
            return false;
        }
        const int hidden_source_row_start = already_appended_tokens - 1;
        const int hidden_source_row_end = hidden_source_row_start + catchup_token_count;
        if (main_forward_token_count <= 0 ||
            hidden_source_row_start < 0 ||
            hidden_source_row_end > main_forward_token_count)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP shifted-row commit received invalid main_forward_token_count="
                      << main_forward_token_count << " token_count=" << token_count
                      << " already_appended_tokens=" << already_appended_tokens
                      << " catchup_token_count=" << catchup_token_count
                      << " hidden_source_row_start=" << hidden_source_row_start
                      << " hidden_source_row_end=" << hidden_source_row_end);
            return false;
        }

        const int position_offset =
            position_offset_override >= 0
                ? position_offset_override
                : state_.positions[0] - main_forward_token_count;
        if (position_offset < 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid MTP shifted-row commit position_offset="
                      << position_offset << " token_count=" << token_count
                      << " main_forward_token_count=" << main_forward_token_count
                      << " override=" << position_offset_override);
            return false;
        }

        IKVCache *cache = state_.mtp_kv_caches.empty() ? nullptr : state_.mtp_kv_caches[0].get();
        if (!cache)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP shifted-row commit requires an initialized MTP KV cache");
            return false;
        }

        void *stream = nullptr;
        void *hidden_producer_stream = nullptr;
        DeviceResidentMTPTransactionLease transaction;
        if (state_.device_id.is_gpu())
        {
            stream = explicitGPUStreamForOperation("commitMTPShiftedRowsFromPartialForward");
            if (!stream)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] MTP shifted-row commit requires an explicit GPU stream");
                return false;
            }
            if (!waitForPendingShiftedMTPKVReadyForObservation(
                    stream,
                    "shifted_row_partial_metadata"))
            {
                return false;
            }
            transaction = currentDeviceResidentMTPTransactionLease();
            if (!transaction.valid())
            {
                if (!recordDeviceResidentMTPTransactionMutation(
                        /*request_count=*/1,
                        stream,
                        "shifted_row_partial_forward_commit_admission"))
                {
                    return false;
                }
                transaction = currentDeviceResidentMTPTransactionLease();
            }
            hidden_producer_stream = stream;
            if (!waitForForwardGraphOutputReady(
                    hidden_producer_stream,
                    DeviceTimelineRole::AcceptedStatePublication,
                    ForwardGraphOutputKind::GroupedVerifier,
                    "commit_mtp_shifted_rows_from_partial_forward"))
            {
                return false;
            }
        }

        const int expected_cached_tokens =
            mtpShiftedAppendBoundaryRows(position_offset,resident_shifted_kv_tokens,cache->stateOwnership().shifted_origin);
        /*
         * This is a storage-safety question, not a request-policy decision.
         * A retained runner may admit a wider request after reset, so preserve
         * every speculative row permitted by its immutable graph envelope.
         */
        const bool may_discard_speculative_rows =
            allow_speculative_discard || mtp_max_draft_depth_ > 1;
        if (!prepareShiftedMTPKVCommitBoundary(
                *cache,
                /*depth=*/0,
                /*request_index=*/0,
                expected_cached_tokens,
                may_discard_speculative_rows,
                stream,
                state_.device_id.is_gpu() ? &transaction : nullptr,
                "shifted_row_partial_forward_commit"))
        {
            return false;
        }

        PerfStatsCollector::ScopedTimer timer(
            "mtp",
            "shifted_row_commit",
            perfPhaseName(),
            state_.device_id.toString());
        const uint64_t workspace_generation_before_commit =
            state_.device_id.is_gpu() ? workspaceGeneration(state_.device_id) : 0;

        if (catchup_token_count > 0)
        {
            if (catchup_token_count >
                mtp_sidecar_condition_token_slot_width_)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] MTP shifted-row batched catchup exceeds configured graph capacity: rows="
                          << catchup_token_count << " capacity="
                          << mtp_sidecar_condition_token_slot_width_);
                return false;
            }
            if (catchup_token_count > main_forward_token_count)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] MTP shifted-row batched catchup exceeds available verifier hidden rows: "
                          << catchup_token_count << " > " << main_forward_token_count);
                return false;
            }
            std::vector<int32_t> token_batch(
                static_cast<size_t>(catchup_token_count));
            for (int row = 0; row < catchup_token_count; ++row)
                token_batch[static_cast<size_t>(row)] = tokens[already_appended_tokens + row];
            if (!selectMTPTerminalHiddenRows(
                    hidden_source_row_start,
                    catchup_token_count,
                    main_forward_token_count,
                    hidden_producer_stream))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] MTP shifted-row catchup failed to select verifier hidden rows "
                          << hidden_source_row_start << ".."
                          << (hidden_source_row_start + catchup_token_count - 1)
                          << " of " << main_forward_token_count);
                return false;
            }
            if (!executeMTPDepth0Batched(token_batch.data(),
                                         catchup_token_count,
                                         state_.prefix_terminal_hidden.get(),
                                         position_offset + already_appended_tokens,
                                         kMTPDecodeCatchupContext,
                                         true,
                                         BufferId::PREFIX_TERMINAL_HIDDEN,
                                         /*defer_final_sync=*/state_.device_id.is_gpu()))
            {
                return false;
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "shifted_row_catchup_hidden_row_selects",
                static_cast<double>(catchup_token_count),
                perfPhaseName(),
                state_.device_id.toString(),
                {{"source_row_start", std::to_string(hidden_source_row_start)},
                 {"main_forward_token_count", std::to_string(main_forward_token_count)}});
            PerfStatsCollector::addCounter(
                "mtp",
                "shifted_row_catchup_sidecar_batches",
                1.0,
                perfPhaseName(),
                state_.device_id.toString(),
                {{"rows", std::to_string(catchup_token_count)}});
        }
        if (!refreshMTPTerminalHiddenState(
                main_forward_token_count,
                1,
                hidden_producer_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to restore terminal hidden after MTP shifted-row commit");
            return false;
        }
        PerfStatsCollector::addCounter(
            "mtp",
            "shifted_rows_committed",
            static_cast<double>(token_count - already_appended_tokens),
            perfPhaseName(),
            state_.device_id.toString());
        if (state_.device_id.is_gpu())
        {
            const uint64_t workspace_generation_after_commit =
                workspaceGeneration(state_.device_id);
            if (workspace_generation_after_commit != workspace_generation_before_commit)
            {
                handleLivePrefixReplayStateAfterMutation(
                    LivePrefixMutationReason::Unknown,
                    "mtp_shifted_row_commit_workspace_rebind");
            }
        }
        return true;
    }

    const float *DeviceGraphOrchestrator::mtpLogits() const
    {
        /*
         * CPU GlobalTP sidecars project a local vocabulary shard into
         * MTP_LOGITS, then finish with a graph-owned allgather into
         * MTP_LOGITS_GATHERED.  Host stochastic sampling must consume the latter;
         * greedy distributed sampling intentionally continues to use the local
         * shard through sampleGreedyFromMTPLogitsOnDevice().
         */
        const BufferId owner =
            mtpSidecarLogitsAreColumnParallel() &&
                    globalTPContextForMTPCoordination() != nullptr
                ? BufferId::MTP_LOGITS_GATHERED
                : mtpSamplingLogitsBufferId();
        auto it = state_.extension_buffers.find(owner);
        if (it == state_.extension_buffers.end() || !it->second)
        {
            return nullptr;
        }
        return publishLogitsTensorToHost(
            it->second.get(),
            HostLogitsSurface::MTPSidecar,
            "mtpLogits");
    }

    bool DeviceGraphOrchestrator::setComputeAllPositionLogits(bool enabled)
    {
        if (!graph_builder_)
        {
            return false;
        }
        if (!graph_builder_->setComputeAllPositionLogits(enabled))
        {
            return false;
        }
        if (!enabled &&
            !graph_builder_->setComputeRowIndexedAllPositionLogits(false, 0))
        {
            return false;
        }
        compute_all_position_logits_ = enabled;
        if (!enabled)
        {
            // Row-indexed logits are meaningful only while the verifier asks
            // for all-position logits. Clearing both knobs together prevents a
            // stale compact row count from shaping the next verifier graph.
            compute_row_indexed_all_position_logits_ = false;
            row_indexed_all_position_logits_row_count_ = 0;
        }
        return true;
    }

    bool DeviceGraphOrchestrator::setComputeRowIndexedAllPositionLogits(bool enabled, int row_count)
    {
        if (!graph_builder_)
        {
            return false;
        }

        int physical_row_count = row_count;
        if (enabled && state_.device_id.is_gpu() &&
            pending_mtp_spec_verifier_input_plan_ &&
            pending_mtp_spec_verifier_input_plan_->request_count == 1)
        {
            const int logical_row_count =
                pending_mtp_spec_verifier_input_plan_->compact_logit_row_count;
            std::string width_policy_error;
            const auto width_policy =
                mtpVerifierPhysicalWidthPolicyForRequest(
                    /*request_count=*/1,
                    &width_policy_error);
            const int policy_row_count =
                width_policy
                    ? mtpVerifierPhysicalPaddedSeqLen(
                          /*request_count=*/1,
                          logical_row_count,
                          mtp_max_verifier_rows_,
                          *width_policy)
                    : 0;
            if (policy_row_count <= 0 ||
                (row_count != logical_row_count &&
                 row_count != policy_row_count))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Row-indexed verifier mode does not match the installed logical/physical geometry"
                          << " requested_rows=" << row_count
                          << " logical_rows=" << logical_row_count
                          << " physical_rows=" << policy_row_count
                          << " policy_error=" << width_policy_error);
                return false;
            }
            physical_row_count = policy_row_count;
        }

        if (!graph_builder_->setComputeRowIndexedAllPositionLogits(
                enabled,
                physical_row_count))
        {
            return false;
        }
        compute_row_indexed_all_position_logits_ = enabled;
        row_indexed_all_position_logits_row_count_ =
            enabled ? physical_row_count : 0;
        return true;
    }

    bool DeviceGraphOrchestrator::setMTPSpecVerifierInputPlan(
        const MTPSpecDecodeVerifierInputPlan &plan)
    {
        if (!plan.ok)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Refusing invalid MTP verifier input plan: "
                      << plan.error);
            return false;
        }
        const int max_compact_rows =
            plan.shape.maxTargetQueryLen() * plan.shape.max_requests;
        if (plan.compact_logit_row_count <= 0 ||
            plan.compact_logit_row_count > max_compact_rows ||
            plan.compact_logit_row_count >
                static_cast<int>(plan.verifier_logit_rows.size()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Refusing malformed MTP verifier row plan: rows="
                      << plan.compact_logit_row_count
                      << " row_values=" << plan.verifier_logit_rows.size()
                      << " max_rows=" << max_compact_rows);
            return false;
        }
        const MTPSpecDecodeVerifierGraphForwardPlan graph_forward_plan =
            buildMTPSpecDecodeVerifierGraphForwardPlan(plan);
        if (!graph_forward_plan.ok)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Refusing MTP verifier row plan that cannot be materialized for graph execution: "
                      << graph_forward_plan.error);
            return false;
        }
        const bool scalar_gpu_verifier =
            state_.device_id.is_gpu() && plan.request_count == 1;
        std::string width_policy_error;
        const auto width_policy =
            scalar_gpu_verifier
                ? mtpVerifierPhysicalWidthPolicyForRequest(
                      plan.request_count,
                      &width_policy_error)
                : std::optional<MTPVerifierPhysicalWidthPolicy>{};
        const int physical_row_count =
            scalar_gpu_verifier && width_policy
                ? mtpVerifierPhysicalPaddedSeqLen(
                      plan.request_count,
                      graph_forward_plan.padded_seq_len,
                      mtp_max_verifier_rows_,
                      *width_policy)
                : plan.compact_logit_row_count;
        if (physical_row_count <= 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP verifier plan has no valid physical graph width"
                      << " logical_rows=" << graph_forward_plan.padded_seq_len
                      << " max_rows=" << mtp_max_verifier_rows_
                      << " policy_error=" << width_policy_error);
            return false;
        }

        const MTPSpecDecodeMetadataShape metadata_capacity =
            mtp_spec_decode_metadata_binding_.capacity();
        const int metadata_row_capacity =
            metadata_capacity.max_requests *
            metadata_capacity.maxTargetQueryLen();
        const bool metadata_capacity_is_insufficient =
            state_.device_id.is_gpu() &&
            (!mtp_spec_decode_metadata_binding_.hasWorkspace() ||
             metadata_capacity.max_requests < plan.request_count ||
             (scalar_gpu_verifier
                  ? metadata_capacity.max_draft_tokens < physical_row_count
                  : metadata_row_capacity < physical_row_count));
        if (metadata_capacity_is_insufficient)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP verifier plan exceeds its "
                      "pre-materialized metadata capacity"
                      << " plan_requests=" << plan.request_count
                      << " logical_rows=" << graph_forward_plan.padded_seq_len
                      << " physical_rows=" << physical_row_count
                      << " capacity_requests=" << metadata_capacity.max_requests
                      << " capacity_rows=" << metadata_capacity.max_draft_tokens
                      << " capacity_total_rows=" << metadata_row_capacity
                      << " binding_error="
                      << mtp_spec_decode_metadata_binding_.bindingError());
            return false;
        }

        std::vector<int> selected_rows;
        selected_rows.reserve(static_cast<size_t>(physical_row_count));
        if (scalar_gpu_verifier)
        {
            for (int row = 0; row < physical_row_count; ++row)
                selected_rows.push_back(row);
        }
        else
        {
            selected_rows.assign(
                graph_forward_plan.verifier_logit_rows.begin(),
                graph_forward_plan.verifier_logit_rows.begin() +
                    plan.compact_logit_row_count);
        }

        /*
         * Verifier callers historically installed row mode both before and
         * after the metadata plan. Publish the canonical physical width here,
         * at the plan boundary, so either order creates the same graph identity.
         * setComputeRowIndexedAllPositionLogits() preserves this width when a
         * later transaction step repeats the logical row count.
         */
        if (scalar_gpu_verifier &&
            !graph_builder_->setComputeRowIndexedAllPositionLogits(
                true,
                physical_row_count))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Graph builder rejected the policy-owned MTP verifier row family");
            return false;
        }
        if (!graph_builder_->setRowIndexedAllPositionLogitRows(selected_rows))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Graph builder rejected MTP verifier row plan");
            return false;
        }

        if (scalar_gpu_verifier)
        {
            compute_row_indexed_all_position_logits_ = true;
            row_indexed_all_position_logits_row_count_ = physical_row_count;
        }

        pending_mtp_spec_verifier_input_plan_ = plan;
        pending_mtp_verifier_device_token_plan_.reset();
        pending_mtp_verifier_device_token_batch_plan_.reset();
        mtp_publication_base_cache_snapshot_ready_ = false;
        mtp_publication_base_cache_snapshot_request_count_ = 0;
        mtp_publication_main_kv_base_checkpoints_ready_ = false;
        mtp_publication_main_kv_base_checkpoint_request_count_ = 0;
        materialized_mtp_verifier_device_token_row_ = {};
        materialized_mtp_verifier_device_token_batch_ = {};
        return true;
    }

    void DeviceGraphOrchestrator::clearMTPSpecVerifierInputPlan()
    {
        pending_mtp_spec_verifier_input_plan_.reset();
        pending_mtp_verifier_device_token_plan_.reset();
        pending_mtp_verifier_device_token_batch_plan_.reset();
        /*
         * The scoped verifier plan is cleared immediately after the verifier
         * forward, but direct publication still needs the pre-verifier
         * BASE_CACHED_TOKENS snapshot.  The next setMTPSpecVerifierInputPlan()
         * invalidates this marker before staging another verifier graph.
         * Keep materialized_mtp_verifier_device_token_row_ alive as well: the
         * greedy compact outcome reducer runs after this RAII cleanup and
         * consumes the device token row that the verifier graph just used.
         */
        if (graph_builder_)
            graph_builder_->setRowIndexedAllPositionLogitRows({});
    }

    bool DeviceGraphOrchestrator::publishPreparedArenaGraphInput(
        BufferId id,
        const void *expected_device_ptr,
        void *producer_stream,
        DeviceId device,
        const char *producer)
    {
        if (!arena_ ||
            !arena_->isRegistered(id) ||
            !device.is_gpu() ||
            !producer_stream ||
            !expected_device_ptr)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot publish prepared arena graph input"
                      << " buffer=" << bufferIdName(id)
                      << " producer=" << (producer ? producer : "unknown")
                      << " device=" << device.toString()
                      << " stream=" << producer_stream
                      << " expected_ptr=" << expected_device_ptr);
            return false;
        }

        const void *const arena_device_ptr =
            arena_->getDevicePtr(id, device);
        if (!arena_device_ptr ||
            arena_device_ptr != expected_device_ptr)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Raw graph-input producer no longer targets its arena binding"
                      << " buffer=" << bufferIdName(id)
                      << " producer=" << (producer ? producer : "unknown")
                      << " device=" << device.toString()
                      << " expected_ptr=" << expected_device_ptr
                      << " arena_ptr=" << arena_device_ptr);
            return false;
        }

        try
        {
            arena_->markWritten(id, device, producer_stream);
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to publish prepared arena graph input"
                      << " buffer=" << bufferIdName(id)
                      << " producer=" << (producer ? producer : "unknown")
                      << " device=" << device.toString()
                      << " error=" << e.what());
            return false;
        }
        return true;
    }

    bool DeviceGraphOrchestrator::zeroAndPublishGenerationSamplingInputsOnStream(
        void *producer_stream,
        const char *reason)
    {
        IBackend *const backend = getBackendFor(state_.device_id);
        // Use the arena's registered geometry, including inactive retained
        // rows. Clearing only the active vocabulary leaks history when the
        // next request batch is wider than its predecessor.
        const size_t request_rows = arena_
            ? arena_->getRows(BufferId::MTP_GENERATED_TOKEN_COUNTS) : 0;
        const size_t history_bytes =
            request_rows * static_cast<size_t>(mtp_generated_token_count_capacity_) *
            sizeof(int32_t);
        const auto seed_tensor = arena_
            ? arena_->getSharedTensor(BufferId::SAMPLING_REQUEST_SEEDS) : nullptr;
        if (!state_.device_id.is_gpu() ||
            !producer_stream ||
            !backend ||
            request_rows == 0 ||
            !seed_tensor || !seed_tensor->gpu_data_ptr() ||
            !backend->memset(seed_tensor->gpu_data_ptr(), 0, seed_tensor->size_bytes(),
                state_.device_id.gpu_ordinal(), producer_stream) ||
            !publishPreparedArenaGraphInput(BufferId::SAMPLING_REQUEST_SEEDS,
                seed_tensor->gpu_data_ptr(), producer_stream, state_.device_id,
                "zero_generation_request_seeds") ||
            !mtp_generated_token_counts_dev_ ||
            mtp_generated_token_count_capacity_ != state_.vocab_size ||
            !backend->memset(
                mtp_generated_token_counts_dev_,
                0,
                history_bytes,
                state_.device_id.gpu_ordinal(),
                producer_stream) ||
            !publishPreparedArenaGraphInput(
                BufferId::MTP_GENERATED_TOKEN_COUNTS,
                mtp_generated_token_counts_dev_,
                producer_stream,
                state_.device_id,
                "zero_mtp_generated_token_history"))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Could not establish empty device-owned MTP generated-token history"
                      << " reason=" << (reason ? reason : "unknown")
                      << " device=" << state_.device_id.toString()
                      << " stream=" << producer_stream
                      << " history=" << mtp_generated_token_counts_dev_
                      << " capacity="
                      << mtp_generated_token_count_capacity_
                      << " vocab=" << state_.vocab_size);
            return false;
        }
        return true;
    }

    bool DeviceGraphOrchestrator::initializeAndPublishMTPTerminalHiddenArchiveOnStream(
        void *producer_stream,
        const char *reason)
    {
        IBackend *const backend = getBackendFor(state_.device_id);
        TensorBase *const archive = state_.prefix_terminal_hidden.get();
        const size_t archive_bytes = archive ? archive->size_bytes() : 0;
        if (!state_.device_id.is_gpu() ||
            !producer_stream ||
            !backend ||
            !archive ||
            !archive->gpu_data_ptr() ||
            archive_bytes == 0 ||
            !backend->memset(
                archive->gpu_data_ptr(),
                0,
                archive_bytes,
                state_.device_id.gpu_ordinal(),
                producer_stream) ||
            !publishPreparedArenaGraphInput(
                BufferId::PREFIX_TERMINAL_HIDDEN,
                archive->gpu_data_ptr(),
                producer_stream,
                state_.device_id,
                "initialize_mtp_terminal_hidden_archive"))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Could not initialize the device-owned MTP terminal-hidden archive"
                      << " reason=" << (reason ? reason : "unknown")
                      << " stream=" << producer_stream
                      << " bytes=" << archive_bytes);
            return false;
        }

        /*
         * Coherent zero bytes satisfy the captured read/write contract, but no
         * main or verifier graph has produced a request terminal yet.
         */
        state_.mtp_terminal_hidden_publication.invalidate();
        return true;
    }

    bool DeviceGraphOrchestrator::initializeAndPublishMTPSampleBanksOnStream(
        void *producer_stream,
        const char *reason)
    {
        IBackend *const backend = getBackendFor(state_.device_id);
        const size_t target_token_bytes =
            static_cast<size_t>(stochastic_target_row_capacity_) *
            sizeof(int32_t);
        const size_t draft_token_bytes =
            static_cast<size_t>(stochastic_draft_row_capacity_) *
            sizeof(int32_t);
        const size_t draft_probability_bytes =
            static_cast<size_t>(stochastic_draft_row_capacity_) *
            sizeof(float);
        const size_t verifier_token_bytes =
            static_cast<size_t>(stochastic_target_row_capacity_) *
            sizeof(int32_t);
        const size_t committed_verifier_identity_bytes =
            static_cast<size_t>(
                stochastic_batch_output_request_capacity_) *
            sizeof(
                sampling_math::MTPCommittedVerifierIdentityRecord);

        /*
         * -1 is an intentionally invalid token sentinel. Per-slot readiness,
         * not these initial bytes, grants semantic read access to reusable
         * proposal and verifier-input scratch. The committed identity is a
         * separate persistent evidence record: zeroing it makes "no committed
         * verifier transaction" explicit without pretending scratch is
         * durable. These setup fills also create every arena completion event,
         * so later sampler and commit publication cannot allocate in decode.
         */
        if (!state_.device_id.is_gpu() ||
            !producer_stream ||
            !backend ||
            !stochastic_target_sample_tokens_dev_ ||
            !stochastic_draft_sample_tokens_dev_ ||
            !stochastic_draft_sample_probs_dev_ ||
            !mtp_verifier_input_tokens_dev_ ||
            !mtp_committed_verifier_identity_dev_ ||
            stochastic_target_row_capacity_ <= 0 ||
            stochastic_draft_row_capacity_ <= 0 ||
            stochastic_batch_output_request_capacity_ <= 0 ||
            !backend->memset(
                stochastic_target_sample_tokens_dev_,
                0xFF,
                target_token_bytes,
                state_.device_id.gpu_ordinal(),
                producer_stream) ||
            !backend->memset(
                stochastic_draft_sample_tokens_dev_,
                0xFF,
                draft_token_bytes,
                state_.device_id.gpu_ordinal(),
                producer_stream) ||
            !backend->memset(
                stochastic_draft_sample_probs_dev_,
                0,
                draft_probability_bytes,
                state_.device_id.gpu_ordinal(),
                producer_stream) ||
            !backend->memset(
                mtp_verifier_input_tokens_dev_,
                0xFF,
                verifier_token_bytes,
                state_.device_id.gpu_ordinal(),
                producer_stream) ||
            !backend->memset(
                mtp_committed_verifier_identity_dev_,
                0,
                committed_verifier_identity_bytes,
                state_.device_id.gpu_ordinal(),
                producer_stream) ||
            !publishPreparedArenaGraphInput(
                BufferId::STOCHASTIC_TARGET_SAMPLE_TOKENS,
                stochastic_target_sample_tokens_dev_,
                producer_stream,
                state_.device_id,
                "initialize_stochastic_target_sample_tokens") ||
            !publishPreparedArenaGraphInput(
                BufferId::STOCHASTIC_DRAFT_SAMPLE_TOKENS,
                stochastic_draft_sample_tokens_dev_,
                producer_stream,
                state_.device_id,
                "initialize_stochastic_draft_sample_tokens") ||
            !publishPreparedArenaGraphInput(
                BufferId::STOCHASTIC_DRAFT_SAMPLE_PROBS,
                stochastic_draft_sample_probs_dev_,
                producer_stream,
                state_.device_id,
                "initialize_stochastic_draft_sample_probs") ||
            !publishPreparedArenaGraphInput(
                BufferId::MTP_VERIFIER_INPUT_TOKENS,
                mtp_verifier_input_tokens_dev_,
                producer_stream,
                state_.device_id,
                "initialize_mtp_verifier_input_tokens") ||
            !publishPreparedArenaGraphInput(
                BufferId::MTP_COMMITTED_VERIFIER_IDENTITY,
                mtp_committed_verifier_identity_dev_,
                producer_stream,
                state_.device_id,
                "initialize_mtp_committed_verifier_identity"))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Could not initialize persistent MTP sample banks"
                      << " reason=" << (reason ? reason : "unknown")
                      << " device=" << state_.device_id.toString()
                      << " stream=" << producer_stream
                      << " target_capacity="
                      << stochastic_target_row_capacity_
                      << " draft_capacity="
                      << stochastic_draft_row_capacity_);
            return false;
        }
        return true;
    }

    bool DeviceGraphOrchestrator::initializeAndPublishDeviceGenerationStorageOnStream(
        void *producer_stream,
        const char *reason)
    {
        IBackend *const backend = getBackendFor(state_.device_id);
        const size_t response_bytes =
            static_cast<size_t>(device_generation_storage_.request_capacity) *
            static_cast<size_t>(device_generation_storage_.response_token_stride) *
            sizeof(int32_t);
        const size_t control_bytes =
            static_cast<size_t>(device_generation_storage_.request_capacity) *
            static_cast<size_t>(device_generation_storage_.control_stride) *
            sizeof(int);
        const size_t dispatch_ticket_bytes =
            static_cast<size_t>(device_generation_storage_.request_capacity) *
            sizeof(sampling_math::DeviceGenerationDispatchTicket);

        /*
         * The response ledger uses -1 as its empty-token sentinel while a zero
         * controller row is deliberately non-admitted.  Request admission
         * replaces both rows with enqueueInitializeDeviceGeneration(); these
         * setup bytes exist to establish real device authority and pre-create
         * each tensor's reusable completion event, not to admit generation.
         */
        if (!state_.device_id.is_gpu() ||
            !producer_stream ||
            !backend ||
            !device_generation_storage_.validFor(
                device_generation_storage_.request_capacity) ||
            device_generation_storage_.active_request_count != 0 ||
            !backend->memset(
                device_generation_storage_.response_tokens_device,
                0xFF,
                response_bytes,
                state_.device_id.gpu_ordinal(),
                producer_stream) ||
            !backend->memset(
                device_generation_storage_.control_device,
                0,
                control_bytes,
                state_.device_id.gpu_ordinal(),
                producer_stream) ||
            !backend->memset(
                device_generation_storage_.dispatch_tickets_device,
                0,
                dispatch_ticket_bytes,
                state_.device_id.gpu_ordinal(),
                producer_stream) ||
            !publishDeviceGenerationArenaState(
                producer_stream,
                "initialize_device_generation_storage"))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Could not initialize persistent device-generation storage"
                      << " reason=" << (reason ? reason : "unknown")
                      << " device=" << state_.device_id.toString()
                      << " stream=" << producer_stream
                      << " request_capacity="
                      << device_generation_storage_.request_capacity
                      << " response_stride="
                      << device_generation_storage_.response_token_stride
                      << " control_stride="
                      << device_generation_storage_.control_stride);
            return false;
        }
        return true;
    }

    bool DeviceGraphOrchestrator::resetAndPublishDeviceForwardStatusOnStream(
        void *producer_stream,const char *reason)
    {
        using Policy=sampling_math::DeviceGenerationForwardFailurePolicy;
        if (!state_.device_id.is_gpu()) return true;
        if (!graph_builder_) return false;
        const auto policy=graph_builder_->deviceGenerationForwardFailurePolicy();
        if (policy==Policy::ResetAtResponseAdmission) return true;
        if (policy!=Policy::PreserveUntilRequestReset || !producer_stream ||
            device_generation_storage_.active_request_count!=0 ||
            !device_generation_state_ready_.handoff.inactive() ||
            !device_generation_storage_.validFor(device_generation_storage_.request_capacity))
            return false;
        auto *backend=getBackendFor(state_.device_id);
        const auto bytes=static_cast<std::size_t>(device_generation_storage_.request_capacity)*
            static_cast<std::size_t>(device_generation_storage_.control_stride)*sizeof(int);
        if (!backend || !backend->memset(device_generation_storage_.control_device,0,bytes,
                state_.device_id.gpu_ordinal(),producer_stream) ||
            !publishPreparedArenaGraphInput(BufferId::MTP_GENERATION_CONTROL,
                device_generation_storage_.control_device,producer_stream,state_.device_id,
                "request_forward_status_reset"))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Request forward-status reset failed reason="
                <<(reason?reason:"unknown")<<" device="<<state_.device_id.toString());
            return false;
        }
        return true;
    }

    bool DeviceGraphOrchestrator::publishDeviceGenerationArenaState(
        void *producer_stream,
        const char *producer)
    {
        if (mtp_depth_transaction_trace_dev_ &&
            !publishPreparedArenaGraphInput(BufferId::MTP_FIRST_TRANSACTION_DIAGNOSTIC,
                mtp_depth_transaction_trace_dev_, producer_stream, state_.device_id, producer))
            return false;
        if (!device_generation_storage_.validFor(
                device_generation_storage_.request_capacity) ||
            !publishPreparedArenaGraphInput(
                BufferId::MTP_GENERATION_RESPONSE_TOKENS,
                device_generation_storage_.response_tokens_device,
                producer_stream,
                state_.device_id,
                producer) ||
            !publishPreparedArenaGraphInput(
                BufferId::MTP_GENERATION_CONTROL,
                device_generation_storage_.control_device,
                producer_stream,
                state_.device_id,
                producer) ||
            !publishPreparedArenaGraphInput(
                BufferId::MTP_GENERATION_DISPATCH_TICKETS,
                device_generation_storage_.dispatch_tickets_device,
                producer_stream,
                state_.device_id,
                producer))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to publish the complete device-generation arena transaction"
                      << " producer="
                      << (producer && producer[0] != '\0'
                              ? producer
                              : "unknown")
                      << " device=" << state_.device_id.toString()
                      << " stream=" << producer_stream);
            return false;
        }
        return true;
    }

    bool DeviceGraphOrchestrator::materializePendingMTPVerifierInputTokensOnDevice(
        void *execution_stream,
        DeviceId execution_device)
    {
        const bool has_single_row_plan =
            pending_mtp_verifier_device_token_plan_.has_value();
        const bool has_batch_plan =
            pending_mtp_verifier_device_token_batch_plan_.has_value();
        if (!has_single_row_plan && !has_batch_plan)
            return true;
        if (has_single_row_plan && has_batch_plan)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Conflicting MTP verifier device-token row and batch plans");
            return false;
        }
        if (!state_.device_id.is_gpu())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Pending MTP verifier device-token plan on non-GPU runner");
            return false;
        }
        if (!execution_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP verifier device-token upload requires an explicit stream");
            return false;
        }
        if (!mtp_verifier_input_tokens_dev_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP verifier device-token buffers are not allocated");
            return false;
        }

        const DeviceId token_device =
            execution_device.is_gpu() ? execution_device : state_.device_id;
        IBackend *backend = getBackendFor(token_device);
        if (!backend)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] No backend for MTP verifier device-token upload on "
                      << token_device.toString());
            return false;
        }

        auto *verifier_tokens =
            static_cast<int32_t *>(mtp_verifier_input_tokens_dev_);
        if (has_batch_plan)
        {
            if (!stochastic_draft_sample_tokens_dev_)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] MTP verifier device-token batch requires STOCHASTIC_DRAFT_SAMPLE_TOKENS");
                return false;
            }
            const auto &batch_plan =
                *pending_mtp_verifier_device_token_batch_plan_;
            if (batch_plan.request_count <= 0 ||
                batch_plan.request_count >
                    stochastic_batch_output_request_capacity_ ||
                batch_plan.logical_padded_seq_len <= 0 ||
                batch_plan.logical_padded_seq_len >
                    mtp_max_verifier_rows_ ||
                batch_plan.request_count *
                        batch_plan.logical_padded_seq_len >
                    stochastic_target_row_capacity_)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Invalid MTP verifier device-token batch plan: requests="
                          << batch_plan.request_count
                          << " logical_padded_seq_len="
                          << batch_plan.logical_padded_seq_len
                          << " capacity=" << stochastic_target_row_capacity_);
                return false;
            }

            for (int request = 0; request < batch_plan.request_count; ++request)
            {
                const DeviceMTPVerifierInputBatchRequest &row =
                    batch_plan.requests[static_cast<size_t>(request)];
                if (!row.first_token_from_device ||
                    row.first_target_sample_slot < 0 ||
                    row.first_target_sample_slot >=
                        stochastic_target_row_capacity_ ||
                    row.total_verifier_input_tokens <= 0 ||
                    row.total_verifier_input_tokens >
                        batch_plan.logical_padded_seq_len ||
                    row.draft_token_count < 0 ||
                    row.draft_token_count + 1 !=
                        row.total_verifier_input_tokens ||
                    row.first_draft_slot < 0 ||
                    row.first_draft_slot + row.draft_token_count >
                        stochastic_draft_row_capacity_)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Invalid MTP verifier device-token batch row "
                              << request);
                    return false;
                }

                int32_t *row_tokens =
                    verifier_tokens +
                    static_cast<size_t>(request) *
                        static_cast<size_t>(
                            batch_plan.logical_padded_seq_len);
                if (!waitForRequiredStochasticTargetSampleReady(
                        row.first_target_sample_slot,
                        execution_stream,
                        "mtp_verifier_batch_first_token"))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Failed to order verifier batch token copy after canonical target slot for row "
                              << request);
                    return false;
                }
                const int32_t *first_token_device =
                    static_cast<const int32_t *>(
                        stochastic_target_sample_tokens_dev_) +
                    row.first_target_sample_slot;
                if (!backend->deviceCopyAsync(
                        row_tokens,
                        first_token_device,
                        sizeof(int32_t),
                        token_device.gpu_ordinal(),
                        execution_stream))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Failed to copy canonical device first token for MTP verifier batch row "
                              << request);
                    return false;
                }

                if (row.draft_token_count > 0 &&
                    !waitForRequiredStochasticDraftSampleReadyRange(
                        row.first_draft_slot,
                        row.draft_token_count,
                        execution_stream,
                        "mtp_verifier_batch_input_tokens"))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Failed to order verifier batch token copy after draft samples");
                    return false;
                }
                if (row.draft_token_count > 0)
                {
                    const auto *draft_tokens =
                        static_cast<const int32_t *>(
                            stochastic_draft_sample_tokens_dev_) +
                        row.first_draft_slot;
                    if (!backend->deviceCopyAsync(
                            row_tokens + 1,
                            draft_tokens,
                            sizeof(int32_t) *
                                static_cast<size_t>(row.draft_token_count),
                            token_device.gpu_ordinal(),
                            execution_stream))
                    {
                        LOG_ERROR("[DeviceGraphOrchestrator] Failed to copy MTP verifier batch draft tokens for row "
                                  << request);
                        return false;
                    }
                }
            }

            PerfStatsCollector::addCounter(
                "mtp",
                "verifier_device_token_batch_input_prepares",
                1.0,
                "decode",
                token_device.toString(),
                {{"requests", std::to_string(batch_plan.request_count)},
                 {"logical_padded_seq_len",
                  std::to_string(
                      batch_plan.logical_padded_seq_len)},
                 {"physical_padded_seq_len",
                  std::to_string(
                      batch_plan.logical_padded_seq_len)}});
            materialized_mtp_verifier_device_token_batch_.valid = true;
            materialized_mtp_verifier_device_token_batch_.request_count =
                batch_plan.request_count;
            materialized_mtp_verifier_device_token_batch_
                .logical_padded_seq_len =
                batch_plan.logical_padded_seq_len;
            materialized_mtp_verifier_device_token_batch_
                .physical_padded_seq_len =
                batch_plan.logical_padded_seq_len;
            materialized_mtp_verifier_device_token_batch_.total_token_capacity =
                batch_plan.request_count *
                batch_plan.logical_padded_seq_len;
            materialized_mtp_verifier_device_token_row_ = {};
            pending_mtp_verifier_device_token_batch_plan_.reset();
            return true;
        }

        const auto &plan = *pending_mtp_verifier_device_token_plan_;
        if (!plan.all_tokens_from_host && !stochastic_draft_sample_tokens_dev_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP verifier device-token plan requires STOCHASTIC_DRAFT_SAMPLE_TOKENS");
            return false;
        }
        if (plan.first_token_from_device &&
            !stochastic_target_sample_tokens_dev_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP verifier device-token plan requires STOCHASTIC_TARGET_SAMPLE_TOKENS");
            return false;
        }
        if (plan.all_tokens_from_host)
        {
            if (!backend->hostToDeviceOnStream(
                    verifier_tokens,
                    plan.host_tokens.data(),
                    sizeof(int32_t) *
                        static_cast<size_t>(plan.total_verifier_input_tokens),
                    token_device.gpu_ordinal(),
                    execution_stream))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to upload host-known MTP verifier token row");
                return false;
            }
            materialized_mtp_verifier_device_token_row_.valid = true;
            materialized_mtp_verifier_device_token_row_.first_token_from_device = false;
            materialized_mtp_verifier_device_token_row_.first_target_sample_slot = -1;
            materialized_mtp_verifier_device_token_row_.total_verifier_input_tokens =
                plan.total_verifier_input_tokens;
            materialized_mtp_verifier_device_token_row_.draft_token_count =
                plan.draft_token_count;
            PerfStatsCollector::addCounter(
                "mtp",
                "verifier_device_token_input_prepares",
                1.0,
                "decode",
                token_device.toString(),
                {{"total_tokens", std::to_string(plan.total_verifier_input_tokens)},
                 {"draft_tokens", std::to_string(plan.draft_token_count)},
                 {"source", "host_fixture"}});
            pending_mtp_verifier_device_token_plan_.reset();
            pending_mtp_verifier_device_token_batch_plan_.reset();
            return true;
        }
        const auto *draft_tokens =
            static_cast<const int32_t *>(stochastic_draft_sample_tokens_dev_) +
            plan.first_draft_slot;

        /*
         * The first verifier input token may still be a host scalar on legacy
         * paths, or it may already be device-resident in the vLLM-style fast
         * lane.  Later entries are sampled draft tokens.  Every copy is queued
         * on the forward engine's stream so graph replay observes a coherent
         * token row without a null-stream dependency.
         */
        if (plan.first_token_from_device)
        {
            const auto *target_token =
                static_cast<const int32_t *>(stochastic_target_sample_tokens_dev_) +
                plan.first_target_sample_slot;
            if (!waitForRequiredStochasticTargetSampleReady(
                    plan.first_target_sample_slot,
                    execution_stream,
                    "mtp_verifier_first_token") ||
                !backend->deviceCopyAsync(
                    verifier_tokens,
                    target_token,
                    sizeof(int32_t),
                    token_device.gpu_ordinal(),
                    execution_stream))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to copy device-resident first MTP verifier token");
                return false;
            }
        }
        else
        {
            if (!backend->hostToDeviceOnStream(
                    verifier_tokens,
                    &pending_mtp_verifier_device_token_plan_->first_token,
                    sizeof(int32_t),
                    token_device.gpu_ordinal(),
                    execution_stream))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to upload first MTP verifier token");
                return false;
            }
        }
        if (plan.draft_token_count > 0 &&
            !waitForRequiredStochasticDraftSampleReadyRange(
                plan.first_draft_slot,
                plan.draft_token_count,
                execution_stream,
                "mtp_verifier_input_tokens"))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to order verifier token copy after deferred draft samples");
            return false;
        }
        if (plan.draft_token_count > 0 &&
            !backend->deviceCopyAsync(
                verifier_tokens + 1,
                draft_tokens,
                sizeof(int32_t) * static_cast<size_t>(plan.draft_token_count),
                token_device.gpu_ordinal(),
                execution_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to copy MTP draft verifier tokens");
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "verifier_device_token_input_prepares",
            1.0,
            "decode",
            token_device.toString(),
            {{"total_tokens", std::to_string(plan.total_verifier_input_tokens)},
             {"draft_tokens", std::to_string(plan.draft_token_count)}});
        materialized_mtp_verifier_device_token_row_.valid = true;
        materialized_mtp_verifier_device_token_row_.first_token_from_device =
            plan.first_token_from_device;
        materialized_mtp_verifier_device_token_row_.first_target_sample_slot =
            plan.first_target_sample_slot;
        materialized_mtp_verifier_device_token_row_.total_verifier_input_tokens =
            plan.total_verifier_input_tokens;
        materialized_mtp_verifier_device_token_row_.draft_token_count =
            plan.draft_token_count;
        pending_mtp_verifier_device_token_plan_.reset();
        pending_mtp_verifier_device_token_batch_plan_.reset();
        return true;
    }

    void DeviceGraphOrchestrator::clearStochasticDraftSampleReadySlot(
        int slot,
        StochasticSampleReadyClearMode mode)
    {
        if (slot < 0 ||
            slot >= static_cast<int>(stochastic_draft_sample_ready_.size()))
        {
            return;
        }

        auto &ready = stochastic_draft_sample_ready_[static_cast<size_t>(slot)];
        if (mode == StochasticSampleReadyClearMode::PreserveVerifierConsumer &&
            ready.verifier_consumer_pending)
        {
            return;
        }
        ready.valid = false;
        ready.producer_stream = nullptr;
        ready.verifier_consumer_pending = false;
    }

    void DeviceGraphOrchestrator::clearStochasticDraftSampleReadySlots(
        StochasticSampleReadyClearMode mode)
    {
        for (int slot = 0;
             slot < static_cast<int>(stochastic_draft_sample_ready_.size());
             ++slot)
        {
            clearStochasticDraftSampleReadySlot(slot, mode);
        }
    }

    void DeviceGraphOrchestrator::clearStochasticTargetSampleReadySlot(
        int slot,
        StochasticSampleReadyClearMode mode)
    {
        if (slot < 0 ||
            slot >= static_cast<int>(stochastic_target_sample_ready_.size()))
        {
            return;
        }

        auto &ready = stochastic_target_sample_ready_[static_cast<size_t>(slot)];
        if (mode == StochasticSampleReadyClearMode::PreserveVerifierConsumer &&
            ready.verifier_consumer_pending)
        {
            return;
        }
        ready.history_right = StochasticTargetHistoryRight::Absent;
        ready.valid = false;
        ready.producer_stream = nullptr;
        ready.verifier_consumer_pending = false;
    }

    void DeviceGraphOrchestrator::clearStochasticTargetSampleReadySlots(
        StochasticSampleReadyClearMode mode)
    {
        for (int slot = 0;
             slot < static_cast<int>(stochastic_target_sample_ready_.size());
             ++slot)
        {
            clearStochasticTargetSampleReadySlot(slot, mode);
        }
    }

    bool DeviceGraphOrchestrator::recordStochasticDraftSampleReady(
        int slot,
        void *producer_stream,
        bool verifier_consumer_pending)
    {
        if (!state_.device_id.is_gpu())
            return true;
        if (slot < 0 ||
            slot >= static_cast<int>(stochastic_draft_sample_ready_.size()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid deferred draft sample slot=" << slot);
            return false;
        }
        if (!producer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Deferred draft sample readiness requires an explicit producer stream");
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        auto &ready = stochastic_draft_sample_ready_[static_cast<size_t>(slot)];
        if (!ready.event)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Deferred draft sample event was not preallocated for slot="
                      << slot);
            return false;
        }

        if (!backend->recordEvent(
                ready.event.get(),
                state_.device_id.gpu_ordinal(),
                producer_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to record deferred draft sample event");
            ready.valid = false;
            ready.producer_stream = nullptr;
            return false;
        }
        if (!publishPreparedArenaGraphInput(
                BufferId::STOCHASTIC_DRAFT_SAMPLE_TOKENS,
                stochastic_draft_sample_tokens_dev_,
                producer_stream,
                state_.device_id,
                "stochastic_draft_sample_ready"))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Deferred draft sample could not publish arena authority for slot="
                      << slot);
            ready.valid = false;
            ready.producer_stream = nullptr;
            return false;
        }

        ready.valid = true;
        ready.producer_stream = producer_stream;
        ready.verifier_consumer_pending =
            ready.verifier_consumer_pending || verifier_consumer_pending;
        PerfStatsCollector::addCounter(
            "mtp",
            "stochastic_draft_sample_ready_events",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"slot", std::to_string(slot)}});
        return true;
    }

    bool DeviceGraphOrchestrator::recordStochasticTargetSampleReady(
        int slot,
        void *producer_stream,
        bool verifier_consumer_pending,
        StochasticTargetPublication publication)
    {
        if (!state_.device_id.is_gpu())
            return true;
        if (slot < 0 ||
            slot >= static_cast<int>(stochastic_target_sample_ready_.size()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid deferred target sample slot=" << slot);
            return false;
        }
        if (!producer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Deferred target sample readiness requires an explicit producer stream");
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        auto &ready = stochastic_target_sample_ready_[static_cast<size_t>(slot)];
        if ((publication != StochasticTargetPublication::NewValue &&
             publication != StochasticTargetPublication::PreserveHistoryRight) ||
            (publication == StochasticTargetPublication::PreserveHistoryRight &&
             (!ready.valid || ready.history_right == StochasticTargetHistoryRight::Absent)))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Target readiness publication has no exact history-consumer authority");
            return false;
        }
        const auto history_right = publication == StochasticTargetPublication::NewValue
            ? StochasticTargetHistoryRight::Available : ready.history_right;
        if (!ready.event)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Deferred target sample event was not preallocated for slot="
                      << slot);
            return false;
        }

        if (!backend->recordEvent(
                ready.event.get(),
                state_.device_id.gpu_ordinal(),
                producer_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to record deferred target sample event");
            ready.valid = false;
            ready.history_right = StochasticTargetHistoryRight::Absent;
            ready.producer_stream = nullptr;
            return false;
        }
        if (!publishPreparedArenaGraphInput(
                BufferId::STOCHASTIC_TARGET_SAMPLE_TOKENS,
                stochastic_target_sample_tokens_dev_,
                producer_stream,
                state_.device_id,
                "stochastic_target_sample_ready"))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Deferred target sample could not publish arena authority for slot="
                      << slot);
            ready.valid = false;
            ready.history_right = StochasticTargetHistoryRight::Absent;
            ready.producer_stream = nullptr;
            return false;
        }

        ready.history_right = history_right;
        ready.valid = true;
        ready.producer_stream = producer_stream;
        ready.verifier_consumer_pending =
            ready.verifier_consumer_pending || verifier_consumer_pending;
        PerfStatsCollector::addCounter(
            "mtp",
            "stochastic_target_sample_ready_events",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"slot", std::to_string(slot)}});
        return true;
    }

    bool DeviceGraphOrchestrator::beginPendingGpuTimingMeasurement(
        const std::string &name,
        void *stream,
        std::shared_ptr<void> *out_start_event,
        std::shared_ptr<void> *out_stop_event)
    {
        if (out_start_event)
            out_start_event->reset();
        if (out_stop_event)
            out_stop_event->reset();
        if (!PerfStatsCollector::isDomainEnabled("mtp") ||
            !state_.device_id.is_gpu())
            return true;
        if (!stream || !out_start_event || !out_stop_event)
            return false;

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        if (!acquirePersistentMTPGpuTimingEvents(
                name.c_str(),
                out_start_event,
                out_stop_event))
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "pending_gpu_timing_event_failures",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"name", name}});
            return false;
        }

        const int device_ordinal = state_.device_id.gpu_ordinal();
        if (!backend->recordEvent(
                out_start_event->get(),
                device_ordinal,
                stream))
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "pending_gpu_timing_record_failures",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"name", name}, {"event", "start"}});
            out_start_event->reset();
            out_stop_event->reset();
            return false;
        }

        return true;
    }

    void DeviceGraphOrchestrator::finishPendingGpuTimingMeasurement(
        const std::string &name,
        void *stream,
        std::map<std::string, std::string> tags,
        std::shared_ptr<void> start_event,
        std::shared_ptr<void> stop_event)
    {
        if (!start_event || !stop_event || !stream || !state_.device_id.is_gpu())
            return;

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return;

        const int device_ordinal = state_.device_id.gpu_ordinal();
        if (!backend->recordEvent(stop_event.get(), device_ordinal, stream))
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "pending_gpu_timing_record_failures",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"name", name}, {"event", "stop"}});
            return;
        }

        pending_gpu_timing_measurements_.push_back(
            PendingGpuTimingMeasurement{
                name,
                std::move(tags),
                std::move(start_event),
                std::move(stop_event)});
    }

    void DeviceGraphOrchestrator::drainPendingGpuTimingMeasurements(
        IBackend *backend)
    {
        if (!backend || !state_.device_id.is_gpu() ||
            pending_gpu_timing_measurements_.empty())
        {
            return;
        }

        std::vector<PendingGpuTimingMeasurement> pending;
        pending.swap(pending_gpu_timing_measurements_);
        const int device_ordinal = state_.device_id.gpu_ordinal();
        for (const auto &measurement : pending)
        {
            float elapsed_ms = 0.0f;
            if (backend->eventElapsedTimeMs(
                    measurement.start_event.get(),
                    measurement.stop_event.get(),
                    device_ordinal,
                    &elapsed_ms))
            {
                const double clamped_ms =
                    std::max(0.0, static_cast<double>(elapsed_ms));
                PerfStatsCollector::recordTimingNs(
                    "mtp",
                    measurement.name,
                    static_cast<uint64_t>(clamped_ms * 1000000.0),
                    "decode",
                    state_.device_id.toString(),
                    measurement.tags);
            }
            else
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "pending_gpu_timing_read_failures",
                    1.0,
                    "decode",
                    state_.device_id.toString(),
                    {{"name", measurement.name}});
            }
        }
    }

    bool
    DeviceGraphOrchestrator::reclaimCompletedGpuTimingMeasurementsNonblocking(
        IBackend *backend)
    {
        if (pending_gpu_timing_measurements_.empty())
            return true;
        if (!backend || !state_.device_id.is_gpu())
            return false;

        IWorkerGPUContext *gpu_context = nullptr;
        try
        {
            gpu_context =
                &GPUDeviceContextPool::instance().getContext(
                    state_.device_id);
        }
        catch (const std::exception &error)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Could not acquire the GPU context "
                "needed for nonblocking MTP timing-event reclamation on "
                << state_.device_id.toString() << ": " << error.what());
            return false;
        }

        /*
         * Compact in place so profiling cannot introduce allocator work into
         * decode. A completed entry releases its shared event owners when the
         * vector is resized, making the fixed pool pair immediately reusable.
         */
        size_t retained_count = 0;
        size_t reclaimed_count = 0;
        const int device_ordinal = state_.device_id.gpu_ordinal();
        for (size_t index = 0;
             index < pending_gpu_timing_measurements_.size();
             ++index)
        {
            auto &measurement =
                pending_gpu_timing_measurements_[index];
            bool ready = false;
            if (!gpu_context->queryEventChecked(
                    measurement.stop_event.get(),
                    ready))
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Failed to query pending MTP "
                    "GPU timing event name="
                    << measurement.name << " device="
                    << state_.device_id.toString());
                return false;
            }

            if (!ready)
            {
                if (retained_count != index)
                {
                    pending_gpu_timing_measurements_[retained_count] =
                        std::move(measurement);
                }
                ++retained_count;
                continue;
            }

            float elapsed_ms = 0.0f;
            if (!backend->eventElapsedTimeMs(
                    measurement.start_event.get(),
                    measurement.stop_event.get(),
                    device_ordinal,
                    &elapsed_ms))
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "pending_gpu_timing_read_failures",
                    1.0,
                    "decode",
                    state_.device_id.toString(),
                    {{"name", measurement.name},
                     {"reclaim", "nonblocking"}});
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Failed to read completed MTP "
                    "GPU timing events name="
                    << measurement.name << " device="
                    << state_.device_id.toString());
                return false;
            }

            const double clamped_ms =
                std::max(0.0, static_cast<double>(elapsed_ms));
            PerfStatsCollector::recordTimingNs(
                "mtp",
                measurement.name,
                static_cast<uint64_t>(clamped_ms * 1000000.0),
                "decode",
                state_.device_id.toString(),
                measurement.tags);
            ++reclaimed_count;
        }

        pending_gpu_timing_measurements_.resize(retained_count);
        if (reclaimed_count > 0)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "pending_gpu_timing_nonblocking_reclaims",
                static_cast<double>(reclaimed_count),
                "decode",
                state_.device_id.toString());
        }
        return true;
    }

    bool DeviceGraphOrchestrator::waitForStochasticDraftSampleReadyRange(
        int first_slot,
        int slot_count,
        void *consumer_stream,
        const char *consumer_name)
    {
        if (!state_.device_id.is_gpu() || slot_count <= 0)
            return true;
        if (!consumer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Deferred draft sample consumer requires an explicit stream");
            return false;
        }
        if (first_slot < 0 ||
            first_slot + slot_count >
                static_cast<int>(stochastic_draft_sample_ready_.size()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid deferred draft sample wait range first_slot="
                      << first_slot << " slot_count=" << slot_count);
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        const char *consumer = consumer_name && consumer_name[0] != '\0'
                                   ? consumer_name
                                   : "unknown";
        for (int i = 0; i < slot_count; ++i)
        {
            const int slot = first_slot + i;
            const auto &ready =
                stochastic_draft_sample_ready_[static_cast<size_t>(slot)];
            if (!ready.valid)
                continue;
            if (!ready.event)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Deferred draft sample slot="
                          << slot << " is marked valid without an event");
                return false;
            }
            if (ready.producer_stream != consumer_stream &&
                !backend->streamWaitEvent(
                    consumer_stream,
                    ready.event.get(),
                    state_.device_id.gpu_ordinal()))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to queue deferred draft sample wait for slot="
                          << slot << " consumer=" << consumer);
                return false;
            }

            PerfStatsCollector::addCounter(
                "mtp",
                "stochastic_draft_sample_ready_waits",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"slot", std::to_string(slot)},
                 {"consumer", consumer},
                 {"same_stream", boolTag(ready.producer_stream == consumer_stream)}});
        }
        return true;
    }

    bool DeviceGraphOrchestrator::waitForRequiredStochasticDraftSampleReadyRange(
        int first_slot,
        int slot_count,
        void *consumer_stream,
        const char *consumer_name)
    {
        if (!state_.device_id.is_gpu() || slot_count <= 0)
            return true;
        if (first_slot < 0 ||
            first_slot + slot_count >
                static_cast<int>(stochastic_draft_sample_ready_.size()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid required deferred draft sample wait range first_slot="
                      << first_slot << " slot_count=" << slot_count);
            return false;
        }

        const char *consumer = consumer_name && consumer_name[0] != '\0'
                                   ? consumer_name
                                   : "unknown";
        for (int i = 0; i < slot_count; ++i)
        {
            const int slot = first_slot + i;
            const auto &ready =
                stochastic_draft_sample_ready_[static_cast<size_t>(slot)];
            if (!ready.valid || !ready.event)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Required deferred draft sample slot="
                          << slot << " is not ready for consumer=" << consumer);
                return false;
            }
        }

        return waitForStochasticDraftSampleReadyRange(
            first_slot,
            slot_count,
            consumer_stream,
            consumer_name);
    }

    bool DeviceGraphOrchestrator::waitForStochasticTargetSampleReady(
        int slot,
        void *consumer_stream,
        const char *consumer_name)
    {
        if (!state_.device_id.is_gpu())
            return true;
        if (!consumer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Deferred target sample consumer requires an explicit stream");
            return false;
        }
        if (slot < 0 ||
            slot >= static_cast<int>(stochastic_target_sample_ready_.size()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid deferred target sample wait slot="
                      << slot);
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        const char *consumer = consumer_name && consumer_name[0] != '\0'
                                   ? consumer_name
                                   : "unknown";
        const auto &ready =
            stochastic_target_sample_ready_[static_cast<size_t>(slot)];
        if (!ready.valid)
            return true;
        if (!ready.event)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Deferred target sample slot="
                      << slot << " is marked valid without an event");
            return false;
        }
        if (ready.producer_stream != consumer_stream &&
            !backend->streamWaitEvent(
                consumer_stream,
                ready.event.get(),
                state_.device_id.gpu_ordinal()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to queue deferred target sample wait for slot="
                      << slot << " consumer=" << consumer);
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "stochastic_target_sample_ready_waits",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"slot", std::to_string(slot)},
             {"consumer", consumer},
             {"same_stream", boolTag(ready.producer_stream == consumer_stream)}});
        return true;
    }

    bool DeviceGraphOrchestrator::waitForRequiredStochasticTargetSampleReady(
        int slot,
        void *consumer_stream,
        const char *consumer_name)
    {
        if (!state_.device_id.is_gpu())
            return true;
        if (slot < 0 ||
            slot >= static_cast<int>(stochastic_target_sample_ready_.size()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid required deferred target sample wait slot="
                      << slot);
            return false;
        }

        const char *consumer = consumer_name && consumer_name[0] != '\0'
                                   ? consumer_name
                                   : "unknown";
        const auto &ready =
            stochastic_target_sample_ready_[static_cast<size_t>(slot)];
        if (!ready.valid || !ready.event)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Required deferred target sample slot="
                      << slot << " is not ready for consumer=" << consumer);
            return false;
        }

        return waitForStochasticTargetSampleReady(
            slot,
            consumer_stream,
            consumer_name);
    }

    bool DeviceGraphOrchestrator::initializeShiftedMTPKVReadyEvent()
    {
        if (!state_.device_id.is_gpu())
            return true;

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        auto &ready = shifted_mtp_kv_ready_;
        if (ready.event)
            return true;

        void *raw_event =
            backend->createEvent(state_.device_id.gpu_ordinal());
        if (!raw_event)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to preallocate shifted MTP KV readiness event");
            return false;
        }

        const int device_ordinal = state_.device_id.gpu_ordinal();
        ready.event = std::shared_ptr<void>(
            raw_event,
            [backend, device_ordinal](void *event)
            {
                if (event)
                    backend->destroyEvent(event, device_ordinal);
            });
        ready.producer_stream = nullptr;
        ready.mutation_generation = 0;
        ready.valid = false;
        return true;
    }

    bool DeviceGraphOrchestrator::initializePersistentDeviceGenerationEvents()
    {
        if (!state_.device_id.is_gpu())
            return true;

        auto fail = [&](const std::string &event_name) -> bool
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to preallocate persistent generation event '"
                      << event_name << "' for "
                      << state_.device_id.toString());
            return false;
        };
        auto ensure_event =
            [&](std::shared_ptr<void> &event,
                const std::string &event_name) -> bool
        {
            if (!event)
                event = allocatePersistentDeviceEvent(state_.device_id);
            return event ? true : fail(event_name);
        };

        // Ordinary and speculative generation publish the same frontier.
        // Its complete reader/writer event bank must exist even without a
        // predictor; request admission only records and waits these owners.
        if (!ensure_event(
                device_resident_logical_sequence_state_ready_event_,
                "device_resident_logical_state"))
        {
            return false;
        }

        /*
         * Every potential reader stream owns a distinct completion event. This
         * is the fixed-resource half of the logical-state fan-out/fan-in
         * protocol: decode only records and waits these events; it never creates
         * one or makes one reader wait another.
         */
        for (size_t lane = 0;
             lane <
                 device_resident_logical_state_reader_completion_events_.size();
             ++lane)
        {
            if (!ensure_event(
                    device_resident_logical_state_reader_completion_events_[lane],
                    "device_resident_logical_state_reader_" +
                        std::to_string(lane)))
            {
                return false;
            }
        }

        if (debugEnv().runtime_debug.mtp_device_phase_snapshots)
        {
            if (!device_resident_logical_state_diagnostic_storage_.validFor(1))
            {
                return fail("device_resident_logical_state_phase_storage");
            }
            for (size_t phase_index = 0;
                 phase_index <
                     device_resident_logical_state_diagnostic_records_.size();
                 ++phase_index)
            {
                const auto phase =
                    static_cast<DeviceResidentLogicalStateDiagnosticPhase>(
                        phase_index);
                auto &record =
                    device_resident_logical_state_diagnostic_records_[
                        phase_index];
                if (!ensure_event(
                        record.ready_event,
                        std::string("device_resident_logical_state_phase_") +
                            deviceResidentLogicalStateDiagnosticPhaseName(
                                phase)))
                {
                    return false;
                }
            }
        }

        // The remaining resources belong to speculative execution only. Keep
        // one setup authority, without allocating unused draft/verifier pools
        // for ordinary generation or creating them lazily in admission.
        if (!graph_builder_)
            return fail("generation_graph_owner");
        if (!graph_builder_->config().mtp.enabled)
            return true;

        /*
         * Sample slots are addressed directly by the verifier graph plan. Give
         * every slot a fixed event now so deferred draft/target publication can
         * never acquire backend resources while decode is active.
         */
        for (size_t slot = 0; slot < stochastic_target_sample_ready_.size(); ++slot)
            if (!ensure_event(stochastic_target_sample_ready_[slot].event,
                    "stochastic_target_sample_" + std::to_string(slot)))
                return false;
        for (size_t slot = 0; slot < stochastic_draft_sample_ready_.size(); ++slot)
            if (!ensure_event(stochastic_draft_sample_ready_[slot].event,
                    "stochastic_draft_sample_" + std::to_string(slot)))
                return false;
        if (!initializeShiftedMTPKVReadyEvent() ||
            !ensure_event(all_position_verifier_state_ready_.event, "all_position_verifier_state") ||
            !ensure_event(accepted_spec_publication_ready_.event, "accepted_spec_publication") ||
            !ensure_event(device_resident_mtp_transaction_ready_event_, "device_resident_transaction"))
            return false;

        /*
         * The response bridge can still hold the previous compact outcome while
         * the next transaction begins. Three fixed slots cover that overlap and
         * leave one diagnostic observer without making the steady-state path
         * depend on event allocation.
         */
        constexpr size_t kOutcomeReadyEventPoolSize = 3;
        if (mtp_outcome_response_ready_event_pool_.empty())
        {
            mtp_outcome_response_ready_event_pool_.reserve(
                kOutcomeReadyEventPoolSize);
            for (size_t slot = 0;
                 slot < kOutcomeReadyEventPoolSize;
                 ++slot)
            {
                auto event =
                    allocatePersistentDeviceEvent(state_.device_id);
                if (!event)
                    return fail(
                        "compact_outcome_response_" +
                        std::to_string(slot));
                mtp_outcome_response_ready_event_pool_.push_back(
                    std::move(event));
            }
        }

        /*
         * At most one measurement can be pending for each target/draft row
         * producer before the compact response boundary drains perfstats. HIP's
         * ticket-scheduled parent can additionally time every retained fragment
         * in the selected transaction while one enclosing transaction interval
         * remains live. A stochastic depth-D branch owns at most 2*D+7 source
         * fragments (including maintenance); five additional pairs cover the
         * enclosing interval, compact reducers, publication, and one diagnostic
         * observer. Deriving the pool from the configured maximum depth keeps
         * profiling allocation-free and total through depth 15.
         */
        if (PerfStatsCollector::isDomainEnabled("mtp") &&
            mtp_gpu_timing_event_pool_.empty())
        {
            const size_t hosted_transaction_timing_capacity =
                static_cast<size_t>(2 * mtp_max_draft_depth_ + 12);
            const size_t timing_pair_capacity =
                std::max(
                    {size_t{8},
                     stochastic_target_sample_ready_.size() +
                         stochastic_draft_sample_ready_.size() +
                         4,
                     hosted_transaction_timing_capacity});
            mtp_gpu_timing_event_pool_.reserve(timing_pair_capacity);
            for (size_t slot = 0;
                 slot < timing_pair_capacity;
                 ++slot)
            {
                PersistentGpuTimingEventPair pair;
                pair.start_event =
                    allocatePersistentDeviceTimingEvent(state_.device_id);
                pair.stop_event =
                    allocatePersistentDeviceTimingEvent(state_.device_id);
                if (!pair.start_event || !pair.stop_event)
                {
                    return fail(
                        "gpu_timing_pair_" + std::to_string(slot));
                }
                mtp_gpu_timing_event_pool_.push_back(std::move(pair));
            }
            pending_gpu_timing_measurements_.reserve(
                timing_pair_capacity);
        }

        return true;
    }

    std::shared_ptr<void>
    DeviceGraphOrchestrator::acquirePersistentMTPOutcomeReadyEvent(
        const char *consumer_name)
    {
        for (const auto &event : mtp_outcome_response_ready_event_pool_)
        {
            if (event && event.use_count() == 1)
                return event;
        }

        LOG_ERROR("[DeviceGraphOrchestrator] Persistent MTP compact-outcome event pool exhausted"
                  << " consumer="
                  << (consumer_name && consumer_name[0] != '\0'
                          ? consumer_name
                          : "unknown")
                  << " capacity="
                  << mtp_outcome_response_ready_event_pool_.size());
        return {};
    }

    bool DeviceGraphOrchestrator::acquirePersistentMTPGpuTimingEvents(
        const char *measurement_name,
        std::shared_ptr<void> *out_start_event,
        std::shared_ptr<void> *out_stop_event)
    {
        if (out_start_event)
            out_start_event->reset();
        if (out_stop_event)
            out_stop_event->reset();
        if (!PerfStatsCollector::isDomainEnabled("mtp") ||
            !state_.device_id.is_gpu())
            return true;
        if (!out_start_event || !out_stop_event)
            return false;

        IBackend *backend = getBackendFor(state_.device_id);
        if (!reclaimCompletedGpuTimingMeasurementsNonblocking(backend))
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Could not reclaim completed MTP "
                "GPU timing events before measurement="
                << (measurement_name && measurement_name[0] != '\0'
                        ? measurement_name
                        : "unknown"));
            return false;
        }

        for (const auto &pair : mtp_gpu_timing_event_pool_)
        {
            if (pair.start_event &&
                pair.stop_event &&
                pair.start_event.use_count() == 1 &&
                pair.stop_event.use_count() == 1)
            {
                *out_start_event = pair.start_event;
                *out_stop_event = pair.stop_event;
                return true;
            }
        }

        LOG_ERROR("[DeviceGraphOrchestrator] Persistent MTP GPU timing-event pool exhausted"
                  << " measurement="
                  << (measurement_name && measurement_name[0] != '\0'
                          ? measurement_name
                          : "unknown")
                  << " capacity="
                  << mtp_gpu_timing_event_pool_.size()
                  << " pending="
                  << pending_gpu_timing_measurements_.size());
        return false;
    }

    bool DeviceGraphOrchestrator::recordShiftedMTPKVReady(
        void *producer_stream,
        const char *producer_name)
    {
        if (!state_.device_id.is_gpu())
            return true;
        if (!producer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Deferred shifted MTP KV readiness requires an explicit producer stream");
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        auto &ready = shifted_mtp_kv_ready_;
        if (ready.valid && ready.producer_stream != producer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Deferred shifted MTP KV producer attempted to overwrite an unconsumed event from a different stream");
            return false;
        }
        if (!ready.event)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Shifted MTP KV readiness event was not initialized before publication");
            return false;
        }

        if (!backend->recordEvent(
                ready.event.get(),
                state_.device_id.gpu_ordinal(),
                producer_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to record shifted MTP KV readiness event");
            ready.valid = false;
            ready.producer_stream = nullptr;
            return false;
        }

        ready.valid = true;
        ready.producer_stream = producer_stream;
        /*
         * Event and generation are one publication transaction. Assigning the
         * generation here prevents a later host-side bookkeeping call from
         * racing a legitimate consumer that has already waited and retired the
         * ready slot. This auxiliary generation never changes the main
         * live-state epoch or invalidates a logical-state mailbox.
         */
        ready.mutation_generation =
            ++shifted_mtp_kv_mutation_generation_;
        PerfStatsCollector::addCounterWithSequence(
            "mtp",
            "shifted_mtp_kv_ready_events",
            1.0,
            {static_cast<uint64_t>(ready.mutation_generation), static_cast<uint64_t>(live_replay_state_epoch_)},
            perfPhaseName(),
            state_.device_id.toString(),
            {{"producer", producer_name && producer_name[0] != '\0'
                              ? producer_name
                              : "unknown"},
             {"mutation_domain", "shifted_mtp_kv"}});
        return true;
    }

    bool DeviceGraphOrchestrator::waitForPendingShiftedMTPKVReady(
        void *consumer_stream,
        const char *consumer_name)
    {
        if (!state_.device_id.is_gpu())
            return true;
        auto &ready = shifted_mtp_kv_ready_;
        if (!ready.valid)
            return true;
        if (!consumer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Deferred shifted MTP KV consumer requires an explicit stream");
            return false;
        }
        if (!ready.event)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Deferred shifted MTP KV readiness is marked valid without an event");
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        const bool raw_stream_handle_matches =
            ready.producer_stream == consumer_stream;
        if (!backend->streamWaitEvent(
                consumer_stream,
                ready.event.get(),
                state_.device_id.gpu_ordinal()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to queue shifted MTP KV wait for consumer="
                      << (consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown"));
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "shifted_mtp_kv_ready_waits",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"consumer", consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown"},
             {"raw_stream_handle_matches",
              boolTag(raw_stream_handle_matches)},
             {"wait_policy", "durable_event"}});
        ready.valid = false;
        ready.producer_stream = nullptr;
        ready.mutation_generation = 0;
        return true;
    }

    bool DeviceGraphOrchestrator::waitForPendingShiftedMTPKVReadyForObservation(
        void *consumer_stream,
        const char *consumer_name) const
    {
        if (!state_.device_id.is_gpu())
            return true;
        const auto &ready = shifted_mtp_kv_ready_;
        if (!ready.valid)
            return true;
        if (!consumer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Deferred shifted MTP KV observation requires an explicit stream");
            return false;
        }
        if (!ready.event)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Deferred shifted MTP KV observation is marked valid without an event");
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        const bool same_stream = ready.producer_stream == consumer_stream;
        if (!same_stream &&
            !backend->streamWaitEvent(
                consumer_stream,
                ready.event.get(),
                state_.device_id.gpu_ordinal()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to queue shifted MTP KV observation wait for consumer="
                      << (consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown"));
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "shifted_mtp_kv_observation_waits",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"consumer", consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown"},
             {"same_stream", boolTag(same_stream)}});
        return true;
    }

    bool DeviceGraphOrchestrator::recordAllPositionVerifierStateReady(
        void *producer_stream,
        const char *producer_name)
    {
        if (!state_.device_id.is_gpu())
            return true;
        if (!producer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Deferred all-position verifier state readiness requires an explicit producer stream");
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        auto &ready = all_position_verifier_state_ready_;
        if (!ready.event)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] All-position verifier state readiness event was not preallocated");
            return false;
        }

        if (!backend->recordEvent(
                ready.event.get(),
                state_.device_id.gpu_ordinal(),
                producer_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to record all-position verifier state readiness event");
            ready.valid = false;
            ready.producer_stream = nullptr;
            return false;
        }

        ready.valid = true;
        ready.producer_stream = producer_stream;
        PerfStatsCollector::addCounter(
            "mtp",
            "all_position_verifier_state_ready_events",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"producer", producer_name && producer_name[0] != '\0'
                              ? producer_name
                              : "unknown"}});
        return true;
    }

    bool DeviceGraphOrchestrator::waitForPendingAllPositionVerifierStateReady(
        void *consumer_stream,
        const char *consumer_name)
    {
        if (!state_.device_id.is_gpu())
            return true;

        auto &ready = all_position_verifier_state_ready_;
        if (!ready.valid)
            return true;
        if (!consumer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Deferred all-position verifier state consumer requires an explicit stream");
            return false;
        }
        if (!ready.event)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Deferred all-position verifier state is marked valid without an event");
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        const bool same_stream = ready.producer_stream == consumer_stream;
        if (mtpPublicationDiagnosticsEnabled())
        {
            LOG_INFO(
                "[MTPPublicationDiagnostics] phase=verifier_ready_consume"
                << " device=" << state_.device_id.toString()
                << " producer_stream=" << ready.producer_stream
                << " consumer_stream=" << consumer_stream
                << " event=" << ready.event.get()
                << " same_stream=" << boolTag(same_stream)
                << " consumer="
                << (consumer_name && consumer_name[0] != '\0'
                        ? consumer_name
                        : "unknown"));
        }
        if (!same_stream &&
            !backend->streamWaitEvent(
                consumer_stream,
                ready.event.get(),
                state_.device_id.gpu_ordinal()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to queue all-position verifier state wait for consumer="
                      << (consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown"));
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "all_position_verifier_state_ready_waits",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"consumer", consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown"},
             {"same_stream", boolTag(same_stream)}});
        clearPendingAllPositionVerifierStateReady();
        return true;
    }

    bool DeviceGraphOrchestrator::waitForPendingAllPositionVerifierStateReadyForObservation(
        void *consumer_stream,
        const char *consumer_name) const
    {
        if (!state_.device_id.is_gpu())
            return true;

        const auto &ready = all_position_verifier_state_ready_;
        if (!ready.valid)
            return true;
        if (!consumer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Deferred all-position verifier state observation requires an explicit stream");
            return false;
        }
        if (!ready.event)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Deferred all-position verifier state observation is marked valid without an event");
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        const bool same_stream = ready.producer_stream == consumer_stream;
        if (!same_stream &&
            !backend->streamWaitEvent(
                consumer_stream,
                ready.event.get(),
                state_.device_id.gpu_ordinal()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to queue all-position verifier state observation wait for consumer="
                      << (consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown"));
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "all_position_verifier_state_observation_waits",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"consumer", consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown"},
             {"same_stream", boolTag(same_stream)}});
        return true;
    }

    void DeviceGraphOrchestrator::clearPendingAllPositionVerifierStateReady()
    {
        all_position_verifier_state_ready_.valid = false;
        all_position_verifier_state_ready_.producer_stream = nullptr;
    }

    bool DeviceGraphOrchestrator::recordAcceptedSpecPublicationReady(
        void *producer_stream,
        const char *producer_name)
    {
        if (!state_.device_id.is_gpu())
            return true;
        if (!producer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Accepted spec-state publication readiness requires an explicit producer stream");
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        auto &ready = accepted_spec_publication_ready_;
        if (ready.valid)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Accepted spec-state publication "
                "attempted to overwrite an unconsumed event");
            return false;
        }
        if (!ready.event)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Accepted spec-state publication readiness event was not preallocated");
            return false;
        }

        if (!DeviceEventEdge::at(
                 DeviceTimelinePoint::AcceptedSpecPublicationReady)
                 .from(DeviceTimelineRole::AcceptedStatePublication)
                 .publish(
                     *backend,
                     state_.device_id,
                     ready.event.get(),
                     producer_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to record accepted spec-state publication readiness event");
            ready.valid = false;
            ready.producer_stream = nullptr;
            ready.live_state_epoch = 0;
            return false;
        }

        ready.valid = true;
        ready.producer_stream = producer_stream;
        ready.live_state_epoch = live_replay_state_epoch_;
        PerfStatsCollector::addCounterWithSequence(
            "mtp",
            "accepted_spec_publication_ready_events",
            1.0,
            {static_cast<uint64_t>(ready.live_state_epoch)},
            perfPhaseName(),
            state_.device_id.toString(),
            {{"producer", producer_name && producer_name[0] != '\0'
                              ? producer_name
                              : "unknown"}});
        return true;
    }

    void DeviceGraphOrchestrator::clearPendingAcceptedSpecPublicationReady() const
    {
        accepted_spec_publication_ready_.valid = false;
        accepted_spec_publication_ready_.producer_stream = nullptr;
        accepted_spec_publication_ready_.live_state_epoch = 0;
    }

    bool DeviceGraphOrchestrator::recordLivePrefixCheckpointReady(
        PrefixStateSnapshot *snapshot,
        void *producer_stream,
        const char *producer_name) const
    {
        if (!state_.device_id.is_gpu())
            return true;
        if (!snapshot || !snapshot->valid)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Live prefix checkpoint readiness requires a valid snapshot");
            return false;
        }
        if (!producer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Live prefix checkpoint readiness requires an explicit producer stream");
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        std::shared_ptr<void> event;
        for (const DeviceKVSequenceStateCheckpoint &checkpoint :
             snapshot->device_sequence_state_checkpoints)
        {
            if (!event && checkpoint.ready_event)
            {
                event = checkpoint.ready_event;
            }
        }
        auto adopt_preallocated_event =
            [&](const PrefixBlockHandle &handle)
        {
            /*
             * A logical checkpoint may retain one storage slot for the main
             * cache and one per shifted MTP cache. All copies are submitted to
             * this method's producer stream, so one event recorded after the
             * final copy proves the complete snapshot ready. Retaining the
             * selected event in PrefixStateSnapshot keeps its pool allocation
             * alive without requiring every independently owned storage slot
             * to share one event identity.
             */
            if (!event && handle.live_checkpoint_ready_event)
            {
                event = handle.live_checkpoint_ready_event;
            }
        };
        for (const PrefixBlockHandle &handle : snapshot->blocks)
        {
            adopt_preallocated_event(handle);
        }
        for (const PrefixBlockHandle &handle : snapshot->mtp_blocks)
        {
            adopt_preallocated_event(handle);
        }
        if (!event)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Live prefix checkpoint submitted asynchronous device work without a preallocated readiness event");
            std::terminate();
        }

        if (!backend->recordEvent(
                event.get(),
                state_.device_id.gpu_ordinal(),
                producer_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to record live prefix checkpoint readiness event");
            /*
             * Device copies have already been submitted and their storage is
             * owned by the snapshot being published. Returning false would
             * release that ownership without any completion proof, allowing a
             * later checkpoint to overwrite bytes still in flight. Event
             * recording is therefore part of the transaction commit, not an
             * optional notification with a synchronization fallback.
             */
            std::terminate();
        }

        snapshot->ready_event = event;
        snapshot->ready_producer_stream = producer_stream;
        snapshot->ready_device = state_.device_id;
        snapshot->ready_event_valid = true;

        live_prefix_checkpoint_ready_.event = std::move(event);
        live_prefix_checkpoint_ready_.producer_stream = producer_stream;
        live_prefix_checkpoint_ready_.valid = true;
        live_prefix_checkpoint_ready_.live_state_epoch = live_replay_state_epoch_;

        PerfStatsCollector::addCounterWithSequence(
            "mtp",
            "live_prefix_checkpoint_ready_events",
            1.0,
            {static_cast<uint64_t>(live_prefix_checkpoint_ready_.live_state_epoch)},
            perfPhaseName(),
            state_.device_id.toString(),
            {{"producer", producer_name && producer_name[0] != '\0'
                              ? producer_name
                              : "unknown"}});
        return true;
    }

    bool DeviceGraphOrchestrator::waitForSnapshotReady(
        const PrefixStateSnapshot &snapshot,
        void *consumer_stream,
        const char *consumer_name) const
    {
        if (!state_.device_id.is_gpu() || !snapshot.ready_event_valid)
            return true;
        if (!consumer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Prefix snapshot readiness wait requires an explicit consumer stream");
            return false;
        }
        if (!snapshot.ready_event)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Prefix snapshot readiness is marked valid without an event");
            return false;
        }
        if (snapshot.ready_device != state_.device_id)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Prefix snapshot readiness belongs to "
                      << snapshot.ready_device.toString()
                      << " but restore is running on " << state_.device_id.toString());
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        const bool same_stream = snapshot.ready_producer_stream == consumer_stream;
        if (!same_stream &&
            !backend->streamWaitEvent(
                consumer_stream,
                snapshot.ready_event.get(),
                state_.device_id.gpu_ordinal()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to queue prefix snapshot readiness wait for consumer="
                      << (consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown"));
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "live_prefix_checkpoint_snapshot_waits",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"consumer", consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown"},
             {"same_stream", boolTag(same_stream)}});
        return true;
    }

    bool DeviceGraphOrchestrator::
        waitForPendingLivePrefixCheckpointReadyForObservation(
            void *consumer_stream,
            const char *consumer_name) const
    {
        if (!state_.device_id.is_gpu())
            return true;
        const auto &ready = live_prefix_checkpoint_ready_;
        if (!ready.valid)
            return true;
        if (!consumer_stream || !ready.event)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Pending live prefix checkpoint observation is missing its stream or event");
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        const bool raw_stream_handle_matches =
            ready.producer_stream == consumer_stream;
        if (!backend->streamWaitEvent(
                consumer_stream,
                ready.event.get(),
                state_.device_id.gpu_ordinal()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to queue live prefix checkpoint observation wait for consumer="
                      << (consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown"));
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "live_prefix_checkpoint_observation_waits",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"consumer",
              consumer_name && consumer_name[0] != '\0'
                  ? consumer_name
                  : "unknown"},
             {"raw_stream_handle_matches",
              boolTag(raw_stream_handle_matches)},
             {"wait_policy", "durable_event"}});
        return true;
    }

    bool DeviceGraphOrchestrator::waitForPendingLivePrefixCheckpointReady(
        void *consumer_stream,
        const char *consumer_name)
    {
        if (!state_.device_id.is_gpu())
            return true;
        auto &ready = live_prefix_checkpoint_ready_;
        if (!ready.valid)
            return true;
        if (!consumer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Pending live prefix checkpoint consumer requires an explicit stream");
            return false;
        }
        if (!ready.event)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Pending live prefix checkpoint is marked valid without an event");
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        const bool raw_stream_handle_matches =
            ready.producer_stream == consumer_stream;
        if (!backend->streamWaitEvent(
                consumer_stream,
                ready.event.get(),
                state_.device_id.gpu_ordinal()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to queue live prefix checkpoint source wait for consumer="
                      << (consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown"));
            return false;
        }

        PerfStatsCollector::addCounterWithSequence(
            "mtp",
            "live_prefix_checkpoint_source_waits",
            1.0,
            {static_cast<uint64_t>(ready.live_state_epoch)},
            perfPhaseName(),
            state_.device_id.toString(),
            {{"consumer", consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown"},
             {"raw_stream_handle_matches",
              boolTag(raw_stream_handle_matches)},
             {"wait_policy", "durable_event"}});
        clearPendingLivePrefixCheckpointReady();
        return true;
    }

    void DeviceGraphOrchestrator::clearPendingLivePrefixCheckpointReady()
    {
        live_prefix_checkpoint_ready_.valid = false;
        live_prefix_checkpoint_ready_.producer_stream = nullptr;
        live_prefix_checkpoint_ready_.live_state_epoch = 0;
        live_prefix_checkpoint_ready_.event.reset();
    }

    bool DeviceGraphOrchestrator::initializeMTPPrefillTerminalArchiveReadyEvent()
    {
        if (!state_.device_id.is_gpu())
            return true;

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        auto &ready = mtp_prefill_terminal_archive_ready_;
        if (ready.event)
            return true;

        void *raw_event =
            backend->createEvent(state_.device_id.gpu_ordinal());
        if (!raw_event)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to preallocate MTP prefill hidden-selection event");
            return false;
        }

        const int device_ordinal = state_.device_id.gpu_ordinal();
        ready.event = std::shared_ptr<void>(
            raw_event,
            [backend, device_ordinal](void *event)
            {
                if (event)
                    backend->destroyEvent(event, device_ordinal);
            });
        ready.producer_stream = nullptr;
        ready.valid = false;
        return true;
    }

    bool DeviceGraphOrchestrator::beginMTPTerminalHiddenMailboxWrite(
        void *writer_stream,
        const char *writer_name)
    {
        if (!state_.device_id.is_gpu())
            return true;
        if (!writer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP terminal-hidden mailbox writer "
                      << (writer_name && writer_name[0] != '\0'
                              ? writer_name
                              : "unknown")
                      << " has no exact GPU producer stream");
            return false;
        }

        /*
         * The ready event protects an older producer write. The shifted-KV
         * event protects the sidecar read that consumed that payload. Queue
         * both edges on the new writer stream before any row-select or upload
         * can replace the reusable device mailbox.
         */
        return waitForPendingMTPPrefillTerminalArchiveReady(
                   writer_stream,
                   writer_name) &&
               waitForPendingShiftedMTPKVReady(
                   writer_stream,
                   writer_name);
    }

    bool DeviceGraphOrchestrator::publishMTPTerminalHiddenMailboxReady(
        void *producer_stream,
        const char *producer_name)
    {
        if (!state_.device_id.is_gpu())
            return true;
        if (!producer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP prefill terminal archive requires an explicit producer stream");
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        auto &ready = mtp_prefill_terminal_archive_ready_;
        if (ready.valid)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP prefill hidden selection attempted to replace an unconsumed readiness event"
                      << " previous_stream=" << ready.producer_stream
                      << " replacement_stream=" << producer_stream);
            return false;
        }
        if (!ready.event)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP prefill hidden-selection event was not initialized before publication");
            return false;
        }
        if (!backend->recordEvent(
                ready.event.get(),
                state_.device_id.gpu_ordinal(),
                producer_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to record MTP prefill terminal archive event");
            return false;
        }

        ready.producer_stream = producer_stream;
        ready.valid = true;
        PerfStatsCollector::addCounter(
            "mtp",
            "shifted_prefill_terminal_archive_ready_events",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"producer", producer_name && producer_name[0] != '\0'
                              ? producer_name
                              : "unknown"}});
        return true;
    }

    bool DeviceGraphOrchestrator::waitForPendingMTPPrefillTerminalArchiveReady(
        void *consumer_stream,
        const char *consumer_name)
    {
        if (!state_.device_id.is_gpu())
            return true;
        auto &ready = mtp_prefill_terminal_archive_ready_;
        if (!ready.valid)
            return true;
        if (!consumer_stream || !ready.event)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Pending MTP prefill terminal archive has no valid consumer stream or event");
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        const bool raw_stream_handle_matches =
            ready.producer_stream == consumer_stream;
        if (!backend->streamWaitEvent(
                consumer_stream,
                ready.event.get(),
                state_.device_id.gpu_ordinal()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to queue MTP prefill terminal archive wait for consumer="
                      << (consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown"));
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "shifted_prefill_terminal_archive_ready_waits",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"consumer", consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown"},
             {"raw_stream_handle_matches",
              boolTag(raw_stream_handle_matches)},
             {"wait_policy", "durable_event"}});
        ready.valid = false;
        ready.producer_stream = nullptr;
        return true;
    }

    bool DeviceGraphOrchestrator::
        waitForPendingMTPPrefillTerminalArchiveReadyForObservation(
            void *consumer_stream,
            const char *consumer_name) const
    {
        if (!state_.device_id.is_gpu())
            return true;
        const auto &ready = mtp_prefill_terminal_archive_ready_;
        if (!ready.valid)
            return true;
        if (!consumer_stream || !ready.event)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP prefill terminal archive observation has no valid stream or event");
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        const bool same_stream = ready.producer_stream == consumer_stream;
        if (!same_stream &&
            !backend->streamWaitEvent(
                consumer_stream,
                ready.event.get(),
                state_.device_id.gpu_ordinal()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to queue MTP prefill terminal archive observation wait for consumer="
                      << (consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown"));
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "shifted_prefill_terminal_archive_observation_waits",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"consumer", consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown"},
             {"same_stream", boolTag(same_stream)}});
        return true;
    }

    void DeviceGraphOrchestrator::clearPendingMTPPrefillTerminalArchiveReady()
    {
        mtp_prefill_terminal_archive_ready_.valid = false;
        mtp_prefill_terminal_archive_ready_.producer_stream = nullptr;
    }

    bool DeviceGraphOrchestrator::prepareLivePrefixMutationReadyEvent(
        void *producer_stream,
        const char *producer_name)
    {
        if (!state_.device_id.is_gpu())
            return true;
        if (!producer_stream)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Live prefix mutation event "
                "preflight requires an explicit stream for producer="
                << (producer_name && producer_name[0] != '\0'
                        ? producer_name
                        : "unknown"));
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        retirePendingPrefixPayloadUses(/*wait_for_all=*/false);

        auto &ready = live_prefix_mutation_ready_;
        if (ready.event)
        {
            if (ready.valid)
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Live prefix mutation event "
                    "preflight found an unconsumed publication");
                return false;
            }
            if (ready.producer_stream &&
                ready.producer_stream != producer_stream)
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Live prefix mutation event "
                    "preflight found an unconsumed event on another stream");
                return false;
            }
            ready.producer_stream = producer_stream;
            return true;
        }

        void *raw_event =
            backend->createEvent(state_.device_id.gpu_ordinal());
        if (!raw_event)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Failed to allocate live prefix "
                "mutation readiness event during preflight");
            return false;
        }
        const int device_ordinal = state_.device_id.gpu_ordinal();
        ready.event = std::shared_ptr<void>(
            raw_event,
            [backend, device_ordinal](void *event)
            {
                if (event)
                    backend->destroyEvent(event, device_ordinal);
            });
        ready.producer_stream = producer_stream;
        return true;
    }

    bool DeviceGraphOrchestrator::recordLivePrefixMutationReady(
        void *producer_stream,
        const char *producer_name,
        std::vector<PrefixPayloadReadLease> retained_payload_sources,
        std::vector<std::shared_ptr<void>> retained_device_sources)
    {
        if (!state_.device_id.is_gpu())
            return true;
        if (!producer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Live prefix mutation readiness requires an explicit producer stream");
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        auto &ready = live_prefix_mutation_ready_;
        if (ready.valid)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Live prefix mutation attempted to "
                "overwrite an unconsumed publication");
            if (!retained_payload_sources.empty() ||
                !retained_device_sources.empty() ||
                !ready.retained_payload_sources.empty() ||
                !ready.retained_device_sources.empty())
            {
                std::terminate();
            }
            return false;
        }
        if (!ready.event &&
            !prepareLivePrefixMutationReadyEvent(
                producer_stream,
                producer_name))
        {
            if (!retained_payload_sources.empty() ||
                !retained_device_sources.empty())
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Prefix restore reached commit "
                    "without a preflighted completion event");
                std::terminate();
            }
            return false;
        }
        if (ready.producer_stream != producer_stream)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Live prefix mutation commit stream "
                "does not match its preflighted event stream");
            if (!retained_payload_sources.empty() ||
                !retained_device_sources.empty())
            {
                std::terminate();
            }
            return false;
        }

        if (!DeviceEventEdge::at(
                 DeviceTimelinePoint::LivePrefixMutationReady)
                 .from(DeviceTimelineRole::PrefixRestoreMutation)
                 .publish(
                     *backend,
                     state_.device_id,
                     ready.event.get(),
                     producer_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to record live prefix mutation readiness event");
            if (!retained_payload_sources.empty() ||
                !retained_device_sources.empty() ||
                !ready.retained_payload_sources.empty() ||
                !ready.retained_device_sources.empty())
            {
                /*
                 * The backend accepted asynchronous reads from cache-owned
                 * bytes but cannot publish their completion edge. Releasing
                 * those owners or changing to synchronous execution would both
                 * violate the production contract, so stop immediately.
                 */
                std::terminate();
            }
            return false;
        }

        for (auto &source : retained_payload_sources)
        {
            ready.retained_payload_sources.push_back(std::move(source));
        }
        for (auto &source : retained_device_sources)
        {
            ready.retained_device_sources.push_back(std::move(source));
        }
        ready.valid = true;
        ready.producer_stream = producer_stream;
        ready.live_state_epoch = live_replay_state_epoch_;
        PerfStatsCollector::addCounterWithSequence(
            "mtp",
            "live_prefix_mutation_ready_events",
            1.0,
            {static_cast<uint64_t>(ready.live_state_epoch)},
            perfPhaseName(),
            state_.device_id.toString(),
            {{"producer", producer_name && producer_name[0] != '\0'
                              ? producer_name
                              : "unknown"}});
        return true;
    }

    bool DeviceGraphOrchestrator::joinPublishedLiveStateHandoffs(
        void *consumer_stream,
        DeviceTimelineRole consumer_role,
        const char *consumer_name) const
    {
        if (!state_.device_id.is_gpu())
            return true;
        if (!consumer_stream)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Published live-state handoff "
                "requires an explicit consumer stream");
            return false;
        }

        bool consumes_publications = false;
        switch (consumer_role)
        {
        case DeviceTimelineRole::DeviceGenerationController:
        case DeviceTimelineRole::MainForwardGraph:
        case DeviceTimelineRole::PrefixRestoreMutation:
        case DeviceTimelineRole::RequestStateReset:
            /*
             * The generation controller is the outer owner of the captured
             * transaction family. Once its parent stream has joined the last
             * externally scheduled state, every successor lives inside that
             * family and no host-side graph will consume the old one-shot
             * publication. Retire it here just as a main forward would.
             */
            consumes_publications = true;
            break;

        case DeviceTimelineRole::MTPSidecarGraph:
        case DeviceTimelineRole::PrefixCheckpointArchive:
        case DeviceTimelineRole::MoERebalanceMaintenance:
        case DeviceTimelineRole::TargetSampler:
        case DeviceTimelineRole::HostResultBridge:
        case DeviceTimelineRole::Diagnostics:
            break;

        default:
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Device timeline role "
                << deviceTimelineRoleName(consumer_role)
                << " is not permitted to join published live-state "
                   "handoffs");
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        const char *consumer =
            consumer_name && consumer_name[0] != '\0'
                ? consumer_name
                : "unknown";
        const bool accepted_publication_pending =
            accepted_spec_publication_ready_.valid;
        const bool live_mutation_pending =
            live_prefix_mutation_ready_.valid;
        const uint64_t accepted_epoch =
            accepted_spec_publication_ready_.live_state_epoch;
        const uint64_t mutation_epoch =
            live_prefix_mutation_ready_.live_state_epoch;

        if (accepted_publication_pending)
        {
            if (!accepted_spec_publication_ready_.event)
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Accepted spec-state "
                    "publication is valid without a durable event");
                return false;
            }
            if (!DeviceEventEdge::at(
                     DeviceTimelinePoint::AcceptedSpecPublicationReady)
                     .from(DeviceTimelineRole::AcceptedStatePublication)
                     .to(consumer_role)
                     .enqueuePublishedWait(
                         *backend,
                         state_.device_id,
                         accepted_spec_publication_ready_.event.get(),
                         consumer_stream))
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Failed to join accepted "
                    "spec-state publication for consumer="
                    << consumer);
                return false;
            }
        }

        if (live_mutation_pending)
        {
            if (!live_prefix_mutation_ready_.event)
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Live prefix mutation is valid "
                    "without a durable event");
                return false;
            }
            if (!DeviceEventEdge::at(
                     DeviceTimelinePoint::LivePrefixMutationReady)
                     .from(DeviceTimelineRole::PrefixRestoreMutation)
                     .to(consumer_role)
                     .enqueuePublishedWait(
                         *backend,
                         state_.device_id,
                         live_prefix_mutation_ready_.event.get(),
                         consumer_stream))
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Failed to join live prefix "
                    "mutation for consumer="
                    << consumer);
                return false;
            }
        }

        if (accepted_publication_pending)
        {
            PerfStatsCollector::addCounterWithSequence(
                "mtp",
                consumes_publications
                    ? "accepted_spec_publication_ready_waits"
                    : "accepted_spec_publication_observation_waits",
                1.0,
                {static_cast<uint64_t>(accepted_epoch)},
                perfPhaseName(),
                state_.device_id.toString(),
                {{"consumer", consumer},
                 {"consumer_role",
                  std::string(deviceTimelineRoleName(consumer_role))},
                 {"wait_policy", "durable_event"}});
        }
        if (live_mutation_pending)
        {
            PerfStatsCollector::addCounterWithSequence(
                "mtp",
                consumes_publications
                    ? "live_prefix_mutation_ready_waits"
                    : "live_prefix_mutation_observation_waits",
                1.0,
                {static_cast<uint64_t>(mutation_epoch)},
                perfPhaseName(),
                state_.device_id.toString(),
                {{"consumer", consumer},
                 {"consumer_role",
                  std::string(deviceTimelineRoleName(consumer_role))},
                 {"wait_policy", "durable_event"}});
        }

        if (consumes_publications)
        {
            if (accepted_publication_pending)
                clearPendingAcceptedSpecPublicationReady();
            if (live_mutation_pending)
                clearPendingLivePrefixMutationReady();
        }
        return true;
    }

    void DeviceGraphOrchestrator::clearPendingLivePrefixMutationReady() const
    {
        auto &ready = live_prefix_mutation_ready_;
        if (ready.event &&
            (ready.valid || !ready.retained_payload_sources.empty() ||
             !ready.retained_device_sources.empty()))
        {
            PendingPrefixPayloadUse pending;
            pending.completion_event = std::move(ready.event);
            pending.retained_payload_sources =
                std::move(ready.retained_payload_sources);
            pending.retained_device_sources =
                std::move(ready.retained_device_sources);
            pending_prefix_payload_uses_.push_back(std::move(pending));
        }

        ready.valid = false;
        ready.producer_stream = nullptr;
        ready.live_state_epoch = 0;
        ready.retained_payload_sources.clear();
        ready.retained_device_sources.clear();
        ready.event.reset();
        retirePendingPrefixPayloadUses(/*wait_for_all=*/false);
    }

    bool DeviceGraphOrchestrator::orderCheckpointMetadataPoolReuse(
        void *checkpoint_stream) const
    {
        if (!state_.device_id.is_gpu())
            return true;
        auto *backend = getBackendFor(state_.device_id);
        if (!backend || !checkpoint_stream)
            return false;

        const auto belongs_to_pool = [&](const std::shared_ptr<void> &source)
        {
            return source && std::any_of(live_device_sequence_state_storage_pool_.begin(),
                live_device_sequence_state_storage_pool_.end(), [&](const auto &slot)
                {
                    // Pointer equality alone is insufficient: an alias with a
                    // different deleter must never be released as pool-owned.
                    return slot.storage && source.get() == slot.storage.get() &&
                        !source.owner_before(slot.storage) && !slot.storage.owner_before(source);
                });
        };
        const auto order_and_release = [&](const std::shared_ptr<void> &event,
                                           std::vector<std::shared_ptr<void>> &sources)
        {
            if (std::none_of(sources.begin(), sources.end(), belongs_to_pool))
                return true;
            if (!event || !DeviceEventEdge::at(DeviceTimelinePoint::LivePrefixMutationReady)
                    .from(DeviceTimelineRole::PrefixRestoreMutation)
                    .to(DeviceTimelineRole::PrefixCheckpointArchive)
                    .enqueuePublishedWait(*backend, state_.device_id, event.get(), checkpoint_stream))
                return false;
            // The pool continues to own the physical allocation. This drops
            // only the restore's exclusion lease, after ordering the next
            // writer behind that read. A retained caller snapshot still keeps
            // use_count > 1, so it cannot be overwritten by another checkpoint.
            std::erase_if(sources, belongs_to_pool);
            return true;
        };

        auto &live = live_prefix_mutation_ready_;
        if (!order_and_release(live.event, live.retained_device_sources))
            return false;
        for (auto &pending : pending_prefix_payload_uses_)
            if (!order_and_release(pending.completion_event, pending.retained_device_sources))
                return false;
        return true;
    }

    void DeviceGraphOrchestrator::retireCompletedPrefixRestoreSources() const
    {
        if (state_.device_id.is_gpu())
        {
            const auto source_alias_count = [&]
            {
                size_t aliases = live_prefix_mutation_ready_.retained_payload_sources.size() +
                    live_prefix_mutation_ready_.retained_device_sources.size();
                for (const auto &pending : pending_prefix_payload_uses_)
                    aliases += pending.retained_payload_sources.size() + pending.retained_device_sources.size();
                return aliases;
            };
            const size_t earlier_aliases = source_alias_count();
            retirePendingPrefixPayloadUses(false);
            PerfStatsCollector::addCounter("prefix_cache", "restore_source_admission_polls",
                1.0, "admission", state_.device_id.toString());
            const size_t retired_aliases = earlier_aliases - source_alias_count();
            if (retired_aliases != 0u)
                PerfStatsCollector::addCounter("prefix_cache", "restore_source_admission_retired_aliases",
                    static_cast<double>(retired_aliases), "admission", state_.device_id.toString());
        }
    }

    void DeviceGraphOrchestrator::retirePendingPrefixPayloadUses(
        bool wait_for_all) const
    {
        IBackend *backend = getBackendFor(state_.device_id);
        const auto completion_is_ready = [&](const std::shared_ptr<void> &event)
        {
            if (!event || !state_.device_id.is_gpu() || !backend)
                throw std::logic_error("Prefix restore source retirement has no exact GPU completion authority");
            if (wait_for_all)
            {
                if (!backend->waitForEvent(event.get(), state_.device_id.gpu_ordinal()))
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Could not close an exact "
                        "prefix restore payload event during teardown");
                    std::terminate();
                }
                return true;
            }
            auto &gpu_context = GPUDeviceContextPool::instance().getContext(state_.device_id);
            bool complete = false;
            if (!gpu_context.queryEventChecked(event.get(), complete))
                throw std::runtime_error("Prefix restore source completion query failed on " + state_.device_id.toString());
            return complete;
        };

        auto &live = live_prefix_mutation_ready_;
        if (!live.retained_payload_sources.empty() || !live.retained_device_sources.empty())
        {
            if (!live.valid)
                throw std::logic_error("Prefix restore sources outlived their published completion authority");
            if (completion_is_ready(live.event))
            {
                // Source-byte ownership ends when the native reads complete.
                // The live-state ordering event has a separate consumer lifetime:
                // suffix harvest may need RAM before decode takes that handoff.
                // Retain its event, stream, epoch and valid bit for that consumer.
                live.retained_payload_sources.clear();
                live.retained_device_sources.clear();
                PerfStatsCollector::addCounter("prefix_cache",
                    "restore_sources_retired_before_handoff", 1.0, "restore",
                    state_.device_id.toString(), {{"waited", wait_for_all ? "true" : "false"}});
            }
        }
        if (pending_prefix_payload_uses_.empty())
            return;

        std::vector<PendingPrefixPayloadUse> still_pending;
        still_pending.reserve(pending_prefix_payload_uses_.size());
        size_t retired = 0;
        for (auto &pending : pending_prefix_payload_uses_)
        {
            if (!pending.valid())
                continue;

            if (completion_is_ready(pending.completion_event))
            {
                ++retired;
                continue;
            }
            still_pending.push_back(std::move(pending));
        }

        pending_prefix_payload_uses_.swap(still_pending);
        if (retired > 0)
        {
            PerfStatsCollector::addCounter(
                "prefix_cache",
                "restore_payload_uses_retired",
                static_cast<double>(retired),
                "restore",
                state_.device_id.toString(),
                {{"waited", wait_for_all ? "true" : "false"}});
        }
    }

    bool DeviceGraphOrchestrator::waitForLiveInferenceStateReadyForObservation(
        void *observation_stream,
        const char *observation_name,
        DeviceTimelineRole observation_role,
        bool logical_state_mailbox_already_joined) const
    {
        if (!state_.device_id.is_gpu())
            return true;
        if (!observation_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Live inference-state observation requires an explicit stream");
            return reportDeviceTimelineJoinFailure(
                state_.device_id,
                "live_inference_state_observation",
                "ExplicitObservationStream",
                observation_name);
        }

        const char *consumer =
            observation_name && observation_name[0] != '\0'
                ? observation_name
                : "live_inference_state_observation";

        const auto snapshot_logical_state =
            [&](DeviceResidentLogicalStateDiagnosticPhase phase)
        {
            const auto &mailbox =
                device_resident_logical_sequence_state_mailbox_;
            if (mailbox.valid() &&
                mailbox.live_state_epoch == live_replay_state_epoch_)
            {
                snapshotDeviceResidentLogicalStatePhase(
                    phase,
                    observation_stream,
                    mailbox.request_count,
                    mailbox.publication_generation);
            }
        };

        /*
         * This method is the read-only counterpart to the restore/truncate and
         * forward-consumer handoff paths.  Do not add ad hoc waits at probe or
         * prefix-cache call sites: if a new live-state owner is introduced, it
         * belongs here so every host-visible KV/GDN/MTP export observes the
         * same producer set.
         */
        if (!logical_state_mailbox_already_joined &&
            !waitForDeviceResidentLogicalSequenceStateMailboxForObservation(
                observation_stream,
                consumer))
        {
            return reportDeviceTimelineJoinFailure(
                state_.device_id,
                "live_inference_state_observation",
                "LogicalSequenceStateReady",
                consumer);
        }
        snapshot_logical_state(
            DeviceResidentLogicalStateDiagnosticPhase::ObservationMailboxReady);

        if (!waitForPendingLivePrefixCheckpointReadyForObservation(
                observation_stream,
                consumer))
        {
            return reportDeviceTimelineJoinFailure(
                state_.device_id,
                "live_inference_state_observation",
                "LivePrefixCheckpointReady",
                consumer);
        }
        snapshot_logical_state(
            DeviceResidentLogicalStateDiagnosticPhase::
                ObservationPrefixCheckpointJoined);

        if (!joinPublishedLiveStateHandoffs(
                observation_stream,
                observation_role,
                consumer))
        {
            return reportDeviceTimelineJoinFailure(
                state_.device_id,
                "live_inference_state_observation",
                "PublishedLiveStateHandoffs",
                consumer);
        }
        snapshot_logical_state(
            DeviceResidentLogicalStateDiagnosticPhase::
                ObservationPublishedHandoffsJoined);

        if (!waitForPendingLiveGraphProducersForObservation(
                observation_stream,
                consumer,
                observation_role))
        {
            return reportDeviceTimelineJoinFailure(
                state_.device_id,
                "live_inference_state_observation",
                "LiveGraphProducersReady",
                consumer);
        }
        snapshot_logical_state(
            DeviceResidentLogicalStateDiagnosticPhase::
                ObservationGraphProducersJoined);

        const DeviceResidentMTPTransactionLease transaction =
            currentDeviceResidentMTPTransactionLease();
        if (transaction.valid() &&
            !waitForDeviceResidentMTPTransaction(
                transaction,
                observation_stream,
                consumer))
        {
            return reportDeviceTimelineJoinFailure(
                state_.device_id,
                "live_inference_state_observation",
                "MTPTransactionReady",
                consumer);
        }
        snapshot_logical_state(
            DeviceResidentLogicalStateDiagnosticPhase::
                ObservationMTPTransactionJoined);
        return true;
    }

    bool DeviceGraphOrchestrator::validatePipelineFollowerMTPInput(
        const ForwardInput &input,
        void *execution_stream,
        DeviceId execution_device) const
    {
        const auto fail = [](const char *reason) {
            LOG_ERROR("[DeviceGraphOrchestrator] Pipeline follower MTP input rejected: " << reason);
            return false;
        };
        if (!state_.device_id.is_gpu() || !execution_stream ||
            execution_device != state_.device_id || input.device != state_.device_id ||
            !pp_stage_config_ || pp_stage_config_->has_lm_head || !pipeline_forward_edges_)
            return fail("missing exact device/stream or frozen follower transport");
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled ||
            input.kv_cache != state_.kv_cache.get() || !state_.kv_cache ||
            pending_mtp_verifier_device_token_plan_ || pending_mtp_verifier_device_token_batch_plan_ ||
            greedy_verifier_outcome_graph_transaction_.state != GreedyVerifierOutcomeGraphState::Idle)
            return fail("missing local MTP/cache ownership or conflicting token/outcome producer");
        try
        {
            pipeline_forward_edges_->validateFollowerMTPInput(input);
        }
        catch (const std::invalid_argument &error)
        {
            return fail(error.what());
        }

        // These are the only two arena-owned input families retained at serving
        // preparation: grouped verifier rows and the scalar logical mailbox.
        // A partial family, another participant's pointer or an interior slice
        // cannot borrow the native receive merely because its shape matches.
        const bool verifier_bank = input.token_ids_device == mtp_verifier_input_tokens_dev_ &&
            input.position_ids_device == mtp_verifier_position_ids_dev_ &&
            input.sequence_lengths_device == mtp_verifier_request_lengths_dev_ &&
            input.seq_len <= mtp_max_verifier_rows_;
        const auto &logical = device_resident_logical_sequence_state_storage_;
        const bool condition_mailbox = input.execution_role == ForwardExecutionRole::MTPCondition &&
            logical.validFor(1) && input.token_ids_device == logical.next_condition_tokens_device &&
            input.position_ids_device == logical.target_cached_tokens_device &&
            input.sequence_lengths_device == logical.target_cached_tokens_device;
        if (!verifier_bank && !condition_mailbox)
            return fail("input rows are not one complete local retained bank");
        return true;
    }

    bool DeviceGraphOrchestrator::prepareDeviceTokenInputsForForwardGraphExecution(
        const ForwardInput &input,
        void *execution_stream,
        DeviceId execution_device)
    {
        if (!state_.device_id.is_gpu())
            return true;

        if (pp_stage_config_ && !pp_stage_config_->has_lm_head &&
            (input.execution_role == ForwardExecutionRole::GroupedMTPVerifier ||
             input.execution_role == ForwardExecutionRole::MTPCondition))
            return validatePipelineFollowerMTPInput(input, execution_stream, execution_device);

        const bool has_single_row_plan =
            pending_mtp_verifier_device_token_plan_.has_value();
        const bool has_batch_plan =
            pending_mtp_verifier_device_token_batch_plan_.has_value();
        const bool has_pending_plan =
            has_single_row_plan || has_batch_plan;
        const bool consumes_persistent_token_bank =
            input.token_ids_device == mtp_verifier_input_tokens_dev_;

        /*
         * A pending plan and its stable arena pointer form one transaction.
         * Either side without the other indicates that a caller staged input
         * for one graph and launched another, or attempted to replay stale
         * bytes from a previous transaction.
         */
        if (has_pending_plan != consumes_persistent_token_bank)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-token forward plan/pointer ownership mismatch"
                      << " pending_single=" << (has_single_row_plan ? 1 : 0)
                      << " pending_batch=" << (has_batch_plan ? 1 : 0)
                      << " consumes_persistent_bank="
                      << (consumes_persistent_token_bank ? 1 : 0)
                      << " role="
                      << static_cast<int>(input.execution_role));
            return false;
        }
        if (!has_pending_plan)
            return true;

        if (input.execution_role !=
                ForwardExecutionRole::GroupedMTPVerifier &&
            input.execution_role != ForwardExecutionRole::MTPCondition)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Persistent MTP token rows may only feed an explicit condition or grouped-verifier graph");
            return false;
        }
        if (input.execution_role == ForwardExecutionRole::MTPCondition)
        {
            if (!has_single_row_plan ||
                input.batch_size != 1 ||
                input.seq_len != 1 ||
                pending_mtp_verifier_device_token_plan_->
                        total_verifier_input_tokens != 1 ||
                pending_mtp_verifier_device_token_plan_->draft_token_count !=
                    0)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Device-resident MTP condition requires exactly one staged target token and no draft rows");
                return false;
            }
        }

        if (input.execution_role ==
            ForwardExecutionRole::GroupedMTPVerifier)
        {
            if (!execution_stream)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Grouped MTP verifier token preparation requires the exact non-null forward stream");
                return false;
            }
            if (has_single_row_plan)
            {
                const auto &plan =
                    *pending_mtp_verifier_device_token_plan_;
                if (plan.all_tokens_from_host ||
                    !plan.first_token_from_device)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] GPU grouped MTP verifier tokens must be fully device-owned");
                    return false;
                }
            }
            else
            {
                for (const auto &request :
                     pending_mtp_verifier_device_token_batch_plan_->requests)
                {
                    if (!request.first_token_from_device)
                    {
                        LOG_ERROR("[DeviceGraphOrchestrator] GPU grouped MTP verifier batch contains a host-owned condition token");
                        return false;
                    }
                }
            }

            /*
             * The grouped verifier consumes token IDs, positions, lengths, and
             * the pre-append KV checkpoint as one transaction.  Leave the plan
             * pending here; prepareAllPositionVerifierGraphMetadata() binds and
             * replays the single typed preparation graph after all producer
             * events have joined this stream.
             */
            return true;
        }

        const DeviceId token_device =
            execution_device.is_gpu() ? execution_device : state_.device_id;
        if (!materializePendingMTPVerifierInputTokensOnDevice(
                execution_stream,
                token_device))
        {
            return false;
        }
        if (!publishPreparedArenaGraphInput(
                BufferId::MTP_VERIFIER_INPUT_TOKENS,
                mtp_verifier_input_tokens_dev_,
                execution_stream,
                token_device,
                input.execution_role == ForwardExecutionRole::MTPCondition
                    ? "mtp_condition_input_tokens"
                    : "mtp_verifier_input_tokens"))
        {
            return false;
        }
        return true;
    }

    bool DeviceGraphOrchestrator::prepareAllPositionVerifierGraphMetadata(
        const ForwardInput &input,
        void *execution_stream,
        DeviceId execution_device)
    {
        if (state_.device_id.is_gpu() && pp_stage_config_ && !pp_stage_config_->has_lm_head &&
            (input.execution_role == ForwardExecutionRole::GroupedMTPVerifier ||
             input.execution_role == ForwardExecutionRole::MTPCondition))
        {
            // Native row arrival and the follower's pre-append checkpoint are
            // already graph roots. An owner-style preparation graph here would
            // create a second producer and require a nonexistent local sampler.
            return validatePipelineFollowerMTPInput(input, execution_stream, execution_device);
        }
        if (!compute_all_position_logits_ ||
            !compute_row_indexed_all_position_logits_)
        {
            return true;
        }

        // CPU row selection consumes the row plan captured in the graph builder.
        // The metadata workspace is needed only when GPU cached graphs read row
        // indices from persistent device buffers during replay.
        if (!state_.device_id.is_gpu())
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "verifier_row_metadata_path",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"path", "cpu_builder_row_plan"},
                 {"rows", std::to_string(row_indexed_all_position_logits_row_count_)}});
            return true;
        }

        if (!execution_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Row-indexed GPU verifier metadata upload requires an explicit stream");
            return false;
        }

        /*
         * The all-position verifier reads both the main live state and the
         * shifted MTP KV cache prepared by the immediately preceding sidecar
         * catch-up step.  Full sidecar forwards publish a logits-stream
         * handoff, but KV-only catch-up intentionally publishes a narrower
         * shifted-KV event.  Consume that event here so cached verifier graph
         * replay cannot race a deferred shifted-cache append.
         */
        if (!waitForPendingShiftedMTPKVReady(
                execution_stream,
                "all_position_verifier_shifted_mtp_kv"))
        {
            return false;
        }

        const DeviceId metadata_device =
            execution_device.is_gpu() ? execution_device : state_.device_id;
        IBackend *backend = getBackendFor(metadata_device);
        if (!backend)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] No backend for verifier metadata upload on "
                      << metadata_device.toString());
            return false;
        }
        if (!waitForPendingLogitsStream(
                PendingLogitsStreamRole::MTPSidecar,
                execution_stream,
                "all_position_verifier_graph"))
        {
            return false;
        }

        const int expected_rows = row_indexed_all_position_logits_row_count_;
        if (!pending_mtp_spec_verifier_input_plan_)
        {
            /*
             * Request-batched GPU prefill is not a verifier transaction. Its
             * row-select stage reads request lengths directly from the arena
             * and derives flattened terminal rows inside the captured kernel.
             * Do not upload a host row shadow into the verifier workspace: that
             * buffer has a separate lifetime and remains exclusively owned by
             * actual verifier metadata producers.
             */
            if (request_batched_prefill_logits_row_count_ >= expected_rows &&
                input.sequence_lengths_device &&
                input.batch_size == expected_rows &&
                input.seq_len > 0)
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "verifier_row_metadata_path",
                    1.0,
                    "prefill",
                    metadata_device.toString(),
                    {{"path", "request_batched_prefill_resident_lengths"},
                     {"rows", std::to_string(expected_rows)}});
                return true;
            }

            LOG_ERROR("[DeviceGraphOrchestrator] Row-indexed GPU verifier graph has no pending MTP row plan or request-prefill row plan");
            return false;
        }

        const auto &plan = *pending_mtp_spec_verifier_input_plan_;
        if (plan.compact_logit_row_count <= 0 ||
            plan.compact_logit_row_count > expected_rows)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Verifier row plan count "
                      << plan.compact_logit_row_count
                      << " is outside physical graph row count "
                      << expected_rows);
            return false;
        }
        const MTPSpecDecodeVerifierGraphForwardPlan graph_forward_plan =
            buildMTPSpecDecodeVerifierGraphForwardPlan(plan);
        if (!graph_forward_plan.ok)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Verifier row plan cannot be materialized for metadata publication: "
                      << graph_forward_plan.error);
            return false;
        }
        const bool has_single_token_plan =
            pending_mtp_verifier_device_token_plan_.has_value();
        const bool has_batch_token_plan =
            pending_mtp_verifier_device_token_batch_plan_.has_value();
        if (has_single_token_plan == has_batch_token_plan)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] GPU grouped verifier preparation requires exactly one resident token plan"
                      << " single=" << (has_single_token_plan ? 1 : 0)
                      << " batch=" << (has_batch_token_plan ? 1 : 0));
            return false;
        }
        bool consume_greedy_penalty_policy = false;
        if (mtp_verifier_outcome_graph_mode_ ==
            MTPVerifierOutcomeGraphMode::Greedy)
        {
            const auto &transaction =
                greedy_verifier_outcome_graph_transaction_;
            if (transaction.state !=
                    GreedyVerifierOutcomeGraphState::Armed ||
                transaction.verifier_token_count !=
                    plan.compact_logit_row_count ||
                !mtp_verifier_stop_tokens_dev_)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Greedy verifier graph controls do not match the compact row plan"
                          << " state="
                          << static_cast<int>(transaction.state)
                          << " transaction_rows="
                          << transaction.verifier_token_count
                          << " logical_rows="
                          << plan.compact_logit_row_count
                          << " physical_rows=" << expected_rows
                          << " stop_buffer="
                          << mtp_verifier_stop_tokens_dev_);
                return false;
            }
            if (!mtp_request_stop_tokens_published_session_epoch_ ||
                *mtp_request_stop_tokens_published_session_epoch_ !=
                    session_epoch_)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Greedy verifier graph reached replay without request-admitted stop controls"
                          << " session=" << session_epoch_
                          << " published_session="
                          << (mtp_request_stop_tokens_published_session_epoch_
                                  ? std::to_string(
                                        *mtp_request_stop_tokens_published_session_epoch_)
                                  : "none"));
                return false;
            }
            if (!consumeMTPRequestPenaltyPolicyOnDevice(
                    transaction.penalty_policy,
                    execution_stream,
                    "greedy_verifier_graph"))
            {
                return false;
            }
            consume_greedy_penalty_policy = true;
        }
        else if (greedy_verifier_outcome_graph_transaction_.state !=
                 GreedyVerifierOutcomeGraphState::Idle)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Armed greedy verifier controls reached metadata preparation without the greedy graph policy");
            return false;
        }
        if (plan.request_count <= 0 ||
            plan.request_count != input.batch_size ||
            input.seq_len <= 0 ||
            input.batch_size * input.seq_len >
                stochastic_target_row_capacity_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Grouped verifier position geometry does not fit its persistent arena owner: plan_requests="
                      << plan.request_count
                      << " graph_batch=" << input.batch_size
                      << " graph_seq_len=" << input.seq_len
                      << " capacity=" << stochastic_target_row_capacity_);
            return false;
        }
        if (!mtp_verifier_position_ids_dev_ ||
            !mtp_verifier_request_lengths_dev_ ||
            input.position_ids_device != mtp_verifier_position_ids_dev_ ||
            input.sequence_lengths_device !=
                mtp_verifier_request_lengths_dev_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] GPU grouped verifier is not bound to its complete device-owned geometry");
            return false;
        }

        const auto upload = uploadMTPSpecDecodeVerifierInputPlan(
            plan,
            mtp_spec_decode_metadata_binding_,
            metadata_device,
            backend,
            execution_stream);
        if (!upload.ok)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to upload verifier row metadata: "
                      << upload.error);
            return false;
        }

        const auto &ptrs = mtp_spec_decode_metadata_binding_.devicePointers();
        if (!ptrs.base_cached_tokens)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Verifier metadata workspace has no base-cache snapshot buffer");
            return false;
        }
        if (!state_.kv_cache)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot snapshot verifier base cache count without a KV cache");
            return false;
        }
        const int request_count = plan.request_count;
        mtp_publication_main_kv_base_checkpoints_ready_ = false;
        mtp_publication_main_kv_base_checkpoint_request_count_ = 0;
        if (request_count >
            static_cast<int>(
                mtp_publication_main_kv_base_checkpoints_.size()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Verifier request count exceeds preallocated main-KV publication checkpoints: requests="
                      << request_count
                      << " checkpoints="
                      << mtp_publication_main_kv_base_checkpoints_.size());
            return false;
        }

        auto &checkpoint_bindings =
            mtp_verifier_preparation_checkpoint_scratch_;
        checkpoint_bindings.clear();
        if (checkpoint_bindings.capacity() <
            static_cast<size_t>(request_count))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Grouped-verifier checkpoint scratch was not preallocated for request count "
                      << request_count);
            return false;
        }
        for (int request_index = 0;
             request_index < request_count;
             ++request_index)
        {
            DeviceKVSequenceStateCheckpoint &checkpoint =
                mtp_publication_main_kv_base_checkpoints_[
                    static_cast<size_t>(request_index)];
            if (!checkpoint.valid() ||
                checkpoint.sequence_index != request_index ||
                checkpoint.device != metadata_device)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Verifier main-KV publication checkpoint identity mismatch for request "
                          << request_index);
                return false;
            }
            checkpoint_bindings.push_back(
                MTPVerifierPreparationStage::MainKVCheckpointBinding{
                    .cache = state_.kv_cache.get(),
                    .sequence_index = request_index,
                    .checkpoint_device = checkpoint.data(),
                    .checkpoint_bytes = checkpoint.bytes,
                });
        }
        const int32_t *base_count_device =
            state_.kv_cache->deviceSequenceCachedTokenCountPtr(/*seq_idx=*/0);
        if (!base_count_device)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] KV cache cannot expose a device-owned base cached-token count for verifier publication");
            return false;
        }

        auto &token_row_bindings =
            mtp_verifier_preparation_token_row_scratch_;
        token_row_bindings.clear();
        if (token_row_bindings.capacity() <
            static_cast<size_t>(request_count))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Grouped-verifier token-row scratch was not preallocated for request count "
                      << request_count);
            return false;
        }
        auto *verifier_tokens =
            static_cast<int32_t *>(mtp_verifier_input_tokens_dev_);
        if (!verifier_tokens || !stochastic_draft_sample_tokens_dev_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Captured grouped-verifier preparation has no resident token banks");
            return false;
        }

        if (has_single_token_plan)
        {
            const auto &token_plan =
                *pending_mtp_verifier_device_token_plan_;
            if (request_count != 1 || input.batch_size != 1 ||
                token_plan.all_tokens_from_host ||
                !token_plan.first_token_from_device ||
                token_plan.total_verifier_input_tokens !=
                    plan.compact_logit_row_count ||
                token_plan.draft_token_count + 1 !=
                    plan.compact_logit_row_count ||
                input.real_seq_len != plan.compact_logit_row_count ||
                input.seq_len < input.real_seq_len ||
                token_plan.first_target_sample_slot < 0 ||
                token_plan.first_target_sample_slot >=
                    stochastic_target_row_capacity_ ||
                token_plan.first_draft_slot < 0 ||
                token_plan.first_draft_slot +
                        token_plan.draft_token_count >
                    stochastic_draft_row_capacity_)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Scalar GPU grouped-verifier token plan is not fully device-owned or does not match graph geometry");
                return false;
            }
            if (!waitForRequiredStochasticTargetSampleReady(
                    token_plan.first_target_sample_slot,
                    execution_stream,
                    "captured_verifier_first_token") ||
                !waitForRequiredStochasticDraftSampleReadyRange(
                    token_plan.first_draft_slot,
                    token_plan.draft_token_count,
                    execution_stream,
                    "captured_verifier_draft_tokens"))
            {
                return false;
            }

            token_row_bindings.push_back(
                MTPVerifierPreparationStage::TokenRowBinding{
                    .first_token_device =
                        static_cast<const int32_t *>(
                            stochastic_target_sample_tokens_dev_) +
                        token_plan.first_target_sample_slot,
                    .draft_tokens_device =
                        token_plan.draft_token_count > 0
                            ? static_cast<const int32_t *>(
                                  stochastic_draft_sample_tokens_dev_) +
                                  token_plan.first_draft_slot
                            : nullptr,
                    .destination_device = verifier_tokens,
                    .draft_token_count = token_plan.draft_token_count,
                });
        }
        else
        {
            const auto &batch_plan =
                *pending_mtp_verifier_device_token_batch_plan_;
            std::string width_policy_error;
            const auto width_policy =
                mtpVerifierPhysicalWidthPolicyForRequest(
                    batch_plan.request_count,
                    &width_policy_error);
            const int physical_padded_seq_len =
                width_policy
                    ? mtpVerifierPhysicalPaddedSeqLen(
                          batch_plan.request_count,
                          batch_plan.logical_padded_seq_len,
                          mtp_max_verifier_rows_,
                          *width_policy)
                    : 0;
            if (batch_plan.request_count != request_count ||
                batch_plan.request_count != input.batch_size ||
                physical_padded_seq_len <= 0 ||
                physical_padded_seq_len != input.seq_len ||
                static_cast<int>(batch_plan.requests.size()) !=
                    request_count)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] GPU grouped-verifier token batch does not match logical/physical graph geometry"
                          << " requests=" << batch_plan.request_count
                          << " graph_batch=" << input.batch_size
                          << " logical_padded_seq_len="
                          << batch_plan.logical_padded_seq_len
                          << " physical_padded_seq_len="
                          << physical_padded_seq_len
                          << " graph_seq_len=" << input.seq_len
                          << " policy_error=" << width_policy_error);
                return false;
            }

            for (int request_index = 0;
                 request_index < request_count;
                 ++request_index)
            {
                const auto &token_request =
                    batch_plan.requests[
                        static_cast<size_t>(request_index)];
                if (!token_request.first_token_from_device ||
                    token_request.total_verifier_input_tokens <= 0 ||
                    token_request.total_verifier_input_tokens > input.seq_len ||
                    token_request.draft_token_count + 1 !=
                        token_request.total_verifier_input_tokens ||
                    token_request.first_target_sample_slot < 0 ||
                    token_request.first_target_sample_slot >=
                        stochastic_target_row_capacity_ ||
                    token_request.first_draft_slot < 0 ||
                    token_request.first_draft_slot +
                            token_request.draft_token_count >
                        stochastic_draft_row_capacity_)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] GPU grouped-verifier token batch row is invalid or host-owned request="
                              << request_index);
                    return false;
                }

                if (!waitForRequiredStochasticTargetSampleReady(
                        token_request.first_target_sample_slot,
                        execution_stream,
                        "captured_verifier_batch_first_token") ||
                    !waitForRequiredStochasticDraftSampleReadyRange(
                        token_request.first_draft_slot,
                        token_request.draft_token_count,
                        execution_stream,
                        "captured_verifier_batch_draft_tokens"))
                {
                    return false;
                }

                token_row_bindings.push_back(
                    MTPVerifierPreparationStage::TokenRowBinding{
                        .first_token_device =
                            static_cast<const int32_t *>(
                                stochastic_target_sample_tokens_dev_) +
                            token_request.first_target_sample_slot,
                        .draft_tokens_device =
                            token_request.draft_token_count > 0
                                ? static_cast<const int32_t *>(
                                      stochastic_draft_sample_tokens_dev_) +
                                      token_request.first_draft_slot
                                : nullptr,
                        .destination_device =
                            verifier_tokens +
                            static_cast<size_t>(request_index) *
                                static_cast<size_t>(input.seq_len),
                        .draft_token_count =
                            token_request.draft_token_count,
                    });
            }
        }

        /*
         * One captured graph now owns every pre-verifier mutation. The row-plan
         * upload above is a no-op for the scalar/full-identity production lane;
         * ragged request batches retain their resident workspace pointer as
         * graph replay data. Producer events have already joined
         * `execution_stream`, so one capture-stream edge orders token assembly,
         * complete KV checkpointing, position expansion, and the base-count
         * snapshot together. Request penalty state has its own arena event and is
         * consumed above; it is never rewritten by this graph.
         */
        const bool explicit_valid_rows =
            graph_forward_plan.logit_row_layout ==
            MTPSpecDecodeVerifierLogitRowLayout::ExplicitSelection;
        MTPVerifierPreparationStage::Params stage_params;
        stage_params.device_id = metadata_device;
        stage_params.backend = backend;
        stage_params.authority = MTPVerifierPreparationStage::Authority::VerifierInputOwner;
        stage_params.token_rows =
            std::span<const MTPVerifierPreparationStage::TokenRowBinding>(
                token_row_bindings);
        stage_params.request_count = request_count;
        stage_params.padded_seq_len = input.seq_len;
        stage_params.base_cached_tokens_device = base_count_device;
        stage_params.valid_graph_rows_device =
            explicit_valid_rows ? ptrs.verifier_logit_rows : nullptr;
        stage_params.valid_graph_row_count = plan.compact_logit_row_count;
        std::string control_policy_error;
        const auto control_policy =
            mtpVerifierPreparationControlPolicyForRequest(
                request_count,
                &control_policy_error);
        if (!control_policy)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] GPU grouped-verifier preparation has malformed controller admission: "
                      << control_policy_error);
            return false;
        }
        const bool generation_controller_owned =
            *control_policy == MTPVerifierPreparationControlPolicy::
                                   DeviceGenerationControlled;
        /*
         * Device generation and MoE maintenance are separate controllers.
         * Static ExpertOverlay owns the former and deliberately omits the
         * latter; Dynamic homogeneous execution owns both. Keep the optional
         * maintenance pointers null for Static so the budget kernel expresses
         * an unconstrained transaction without inventing movement state.
         */
        const bool requires_rebalance_controller =
            generation_controller_owned &&
            usesParticipantLocalDeviceMoERebalanceController();
        std::string rebalance_controller_error;
        DeviceMoERebalanceGraphControllerState *rebalance_controller =
            requires_rebalance_controller
                ? deviceMoERebalanceControllerStateDevice(
                      &rebalance_controller_error)
                : nullptr;
        if (requires_rebalance_controller && !rebalance_controller)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Controlled grouped-verifier preparation could not bind the MoE maintenance controller: "
                      << rebalance_controller_error);
            return false;
        }
        auto *rebalance_controller_bytes =
            reinterpret_cast<std::byte *>(rebalance_controller);
        stage_params.generation_control_device =
            generation_controller_owned
                ? device_generation_storage_.control_device
                : nullptr;
        stage_params.generation_control_stride =
            generation_controller_owned
                ? device_generation_storage_.control_stride
                : 0;
        stage_params.maintenance_rows_remaining_device =
            pipeline_maintenance_admission_ ? pipeline_maintenance_admission_->remaining(pipeline_maintenance_participant_) :
            rebalance_controller
                ? reinterpret_cast<const uint32_t *>(
                      rebalance_controller_bytes +
                      offsetof(
                          DeviceMoERebalanceGraphControllerState,
                          decode_rounds_until_maintenance))
                : nullptr;
        stage_params.maintenance_due_device =
            pipeline_maintenance_admission_ && !rebalance_controller ? pipeline_maintenance_admission_->inactiveStatus(pipeline_maintenance_participant_) :
            rebalance_controller
                ? reinterpret_cast<const uint32_t *>(
                      rebalance_controller_bytes +
                      offsetof(
                          DeviceMoERebalanceGraphControllerState,
                          maintenance_due))
                : nullptr;
        stage_params.decode_boundary_advanced_device =
            pipeline_maintenance_admission_ && !rebalance_controller ? pipeline_maintenance_admission_->inactiveMarker(pipeline_maintenance_participant_) :
            rebalance_controller
                ? reinterpret_cast<uint32_t *>(
                      rebalance_controller_bytes +
                      offsetof(
                          DeviceMoERebalanceGraphControllerState,
                          decode_boundary_advanced))
                : nullptr;
        stage_params.position_ids_device =
            static_cast<int32_t *>(mtp_verifier_position_ids_dev_);
        stage_params.request_lengths_device =
            static_cast<int32_t *>(
                mtp_verifier_request_lengths_dev_);
        stage_params.base_cached_tokens_snapshot_device =
            ptrs.base_cached_tokens;
        stage_params.main_kv_checkpoints =
            std::span<const MTPVerifierPreparationStage::MainKVCheckpointBinding>(
                checkpoint_bindings);
        MTPVerifierPreparationGraphKey preparation_key;
        std::string preparation_error;
        if (!materializeMTPVerifierPreparationGraph(
                stage_params,
                &preparation_key,
                &preparation_error) ||
            !executeMTPVerifierPreparationCaptured(
                execution_stream,
                preparation_key,
                &preparation_error))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Captured grouped-verifier preparation failed: "
                      << preparation_error);
            return false;
        }

        if (!publishPreparedArenaGraphInput(
                BufferId::MTP_VERIFIER_INPUT_TOKENS,
                mtp_verifier_input_tokens_dev_,
                execution_stream,
                metadata_device,
                "captured_mtp_verifier_input_tokens") ||
            !publishPreparedArenaGraphInput(
                BufferId::MTP_VERIFIER_POSITION_IDS,
                mtp_verifier_position_ids_dev_,
                execution_stream,
                metadata_device,
                "captured_mtp_verifier_position_ids") ||
            !publishPreparedArenaGraphInput(
                BufferId::MTP_VERIFIER_REQUEST_LENGTHS,
                mtp_verifier_request_lengths_dev_,
                execution_stream,
                metadata_device,
                "captured_mtp_verifier_request_lengths"))
        {
            return false;
        }
        if (generation_controller_owned &&
            !publishPreparedArenaGraphInput(
                BufferId::MTP_GENERATION_CONTROL,
                device_generation_storage_.control_device,
                execution_stream,
                metadata_device,
                "captured_mtp_verifier_transaction_budget"))
        {
            return false;
        }
        if (has_single_token_plan)
        {
            const auto &token_plan =
                *pending_mtp_verifier_device_token_plan_;
            materialized_mtp_verifier_device_token_row_.valid = true;
            materialized_mtp_verifier_device_token_row_
                .first_token_from_device = true;
            materialized_mtp_verifier_device_token_row_
                .first_target_sample_slot =
                token_plan.first_target_sample_slot;
            materialized_mtp_verifier_device_token_row_
                .total_verifier_input_tokens =
                token_plan.total_verifier_input_tokens;
            materialized_mtp_verifier_device_token_row_
                .draft_token_count = token_plan.draft_token_count;
            materialized_mtp_verifier_device_token_batch_ = {};
        }
        else
        {
            const auto &batch_plan =
                *pending_mtp_verifier_device_token_batch_plan_;
            materialized_mtp_verifier_device_token_batch_.valid = true;
            materialized_mtp_verifier_device_token_batch_.request_count =
                batch_plan.request_count;
            materialized_mtp_verifier_device_token_batch_
                .logical_padded_seq_len =
                batch_plan.logical_padded_seq_len;
            materialized_mtp_verifier_device_token_batch_
                .physical_padded_seq_len = input.seq_len;
            materialized_mtp_verifier_device_token_batch_
                .total_token_capacity =
                batch_plan.request_count * input.seq_len;
            materialized_mtp_verifier_device_token_row_ = {};
        }
        pending_mtp_verifier_device_token_plan_.reset();
        pending_mtp_verifier_device_token_batch_plan_.reset();
        mtp_publication_base_cache_snapshot_ready_ = true;
        mtp_publication_base_cache_snapshot_request_count_ = request_count;
        mtp_publication_main_kv_base_checkpoints_ready_ = true;
        mtp_publication_main_kv_base_checkpoint_request_count_ =
            request_count;

        PerfStatsCollector::addCounter(
            "mtp",
            has_single_token_plan
                ? "verifier_device_token_input_prepares"
                : "verifier_device_token_batch_input_prepares",
            1.0,
            "decode",
            metadata_device.toString(),
            {{"requests", std::to_string(request_count)},
             {"padded_seq_len", std::to_string(input.seq_len)},
             {"source", "captured_device_resident"}});
        if (consume_greedy_penalty_policy)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "graph_owned_greedy_penalty_policy_consumptions",
                1.0,
                "decode",
                metadata_device.toString(),
                {{"rows", std::to_string(expected_rows)},
                 {"owner", "request_admission_and_device_publication"}});
        }
        PerfStatsCollector::addCounter(
            "mtp",
            "verifier_row_metadata_upload_bytes",
            static_cast<double>(upload.bytes_uploaded),
            "decode",
            metadata_device.toString());
        PerfStatsCollector::addCounter(
            "mtp",
            "verifier_base_cache_device_snapshots",
            1.0,
            "decode",
            metadata_device.toString(),
            {{"requests", std::to_string(request_count)}});
        PerfStatsCollector::addCounter(
            "mtp",
            "verifier_base_kv_metadata_checkpoint_captures",
            static_cast<double>(request_count),
            "decode",
            metadata_device.toString(),
            {{"requests", std::to_string(request_count)},
             {"layers",
              std::to_string(state_.kv_cache->n_layers())},
             {"ownership", "persistent_device_checkpoint"}});
        PerfStatsCollector::addCounter(
            "mtp",
            "verifier_device_position_expansions",
            1.0,
            "decode",
            metadata_device.toString(),
            {{"requests", std::to_string(request_count)},
             {"padded_seq_len", std::to_string(input.seq_len)},
             {"rows", std::to_string(request_count * input.seq_len)}});
        PerfStatsCollector::addCounter(
            "mtp",
            "verifier_row_metadata_path",
            1.0,
            "decode",
            metadata_device.toString(),
            {{"path",
              graph_forward_plan.logit_row_layout ==
                      MTPSpecDecodeVerifierLogitRowLayout::
                          FullPhysicalIdentity
                  ? "direct_full_physical_rows"
                  : "persistent_workspace"},
             {"rows", std::to_string(expected_rows)}});

        return true;
    }
} // namespace llaminar2
