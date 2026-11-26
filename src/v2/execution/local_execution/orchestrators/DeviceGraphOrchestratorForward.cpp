/**
 * @file DeviceGraphOrchestratorForward.cpp
 * @brief DeviceGraphOrchestrator method shard (17631..22590).
 */
#include "DeviceGraphOrchestratorInternal.h"

namespace llaminar2
{

    std::optional<const float *> DeviceGraphOrchestrator::forwardImpl(
        const int *tokens,
        const void *token_ids_device,
        int seq_len,
        int batch_size,
        ForwardExecutionRole execution_role,
        ForwardInvocationKind invocation,
        const void *position_ids_device_override,
        const int32_t *sequence_lengths_device_override,
        std::span<const int> request_real_lengths,
        std::optional<RemotePrefillTransactionGeometry>
            remote_prefill_geometry)
    {
        // Enable device-scoped logging for this execution
        // All LOG_* calls from this thread will include the device ID
        ScopedDeviceLog device_log(state_.device_id);

        if (!state_.isInitialized())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] forward() called without initialized state");
            return std::nullopt;
        }

        if (!hasGlobalWeights())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] forward() called without global weights set");
            return std::nullopt;
        }

        if (batch_size > state_.batch_size)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Batch size " << batch_size
                                                              << " exceeds initialized batch size " << state_.batch_size);
            return std::nullopt;
        }
        if (invocation == ForwardInvocationKind::ExplicitPrefill &&
            execution_role != ForwardExecutionRole::MainInference)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Internal MTP forward roles are decode-equivalent and cannot force prefill semantics");
            return std::nullopt;
        }
        const bool restored_prefix_mtp_decode_bridge =
            invocation ==
            ForwardInvocationKind::RestoredPrefixMTPDecodeBridge;
        const bool committed_mtp_condition =
            invocation == ForwardInvocationKind::CommittedMTPCondition;
        if (committed_mtp_condition &&
            (execution_role != ForwardExecutionRole::MTPCondition ||
             batch_size != 1 || seq_len != 1 || !token_ids_device ||
             !position_ids_device_override || !sequence_lengths_device_override))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Committed MTP condition requires one complete resident token/position/length row");
            return std::nullopt;
        }
        const ForwardExecutionRole restored_prefix_bridge_role =
            state_.device_id.is_gpu()
                ? ForwardExecutionRole::MTPCondition
                : ForwardExecutionRole::MainInference;
        if (restored_prefix_mtp_decode_bridge &&
            (execution_role != restored_prefix_bridge_role ||
             batch_size != 1 || seq_len != 1 || !tokens || token_ids_device))
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Restored-prefix MTP bridge requires "
                "one host-admitted MTP-condition token");
            return std::nullopt;
        }
        if (position_ids_device_override && !state_.device_id.is_gpu())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-resident position rows require a GPU forward");
            return std::nullopt;
        }
        if (sequence_lengths_device_override && !state_.device_id.is_gpu())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-resident sequence-length rows require a GPU forward");
            return std::nullopt;
        }
        if (invocation == ForwardInvocationKind::ExplicitDecode &&
            batch_size > 1 &&
            (!position_ids_device_override ||
             !sequence_lengths_device_override))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident request-batched decode requires device-owned position and sequence-length rows");
            return std::nullopt;
        }

        // Reject already selected ordinary output demand before token admission/event edges.
        requireActiveNativeTerminalPublicationCapacity(batch_size, seq_len);

        /*
         * Normalize the immutable row geometry at the API boundary. Ordinary
         * rectangular forwards intentionally mean that every physical row is
         * real; ragged request batches must provide their lengths explicitly.
         * Nothing below this point derives admission data from mutable runner
         * progress, so submitting the graph cannot race the H2D source.
         */
        std::vector<int> rectangular_request_lengths;
        if (request_real_lengths.empty())
        {
            rectangular_request_lengths.assign(
                static_cast<size_t>(batch_size),
                seq_len);
            request_real_lengths = rectangular_request_lengths;
        }
        if (request_real_lengths.size() !=
            static_cast<size_t>(batch_size))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Forward request-length geometry does not match its request count"
                      << " lengths=" << request_real_lengths.size()
                      << " requests=" << batch_size);
            return std::nullopt;
        }
        for (int request = 0; request < batch_size; ++request)
        {
            const int real_tokens =
                request_real_lengths[static_cast<size_t>(request)];
            if (real_tokens <= 0 || real_tokens > seq_len)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Forward received an invalid logical request row"
                          << " request=" << request
                          << " real_tokens=" << real_tokens
                          << " physical_seq_len=" << seq_len);
                return std::nullopt;
            }
        }

        const bool mtp_condition =
            execution_role == ForwardExecutionRole::MTPCondition;
        const bool has_device_resident_condition_position =
            position_ids_device_override != nullptr;
        const bool has_device_resident_condition_length =
            sequence_lengths_device_override != nullptr;
        if (mtp_condition &&
            has_device_resident_condition_position !=
                has_device_resident_condition_length)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Resident MTP condition state must "
                "provide both device position and sequence-length rows");
            return std::nullopt;
        }
        const bool live_request_batch_condition =
            mtp_condition &&
            ((has_device_resident_condition_position &&
              has_device_resident_condition_length) ||
             (restored_prefix_mtp_decode_bridge &&
              state_.device_id.is_gpu()));
        if (mtp_condition &&
            !restored_prefix_mtp_decode_bridge &&
            ((invocation != ForwardInvocationKind::ExplicitDecode &&
              !committed_mtp_condition) ||
             batch_size <= 0 || !token_ids_device))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident MTP condition role "
                      "requires one or more device token rows and forced decode semantics");
            return std::nullopt;
        }
        /*
         * The live-condition policy is selected by state ownership, not by
         * request count. A one-request device-resident continuation needs the
         * same full-row logits surface and isolated GDN workspace as a larger
         * request batch. Gating this policy on `batch_size > 1` leaves scalar
         * resident advances sampling stale verifier logits.
         */
        if (!mtp_condition &&
            invocation == ForwardInvocationKind::ExplicitDecode &&
            batch_size > 1)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Multi-request forced decode must explicitly "
                      "declare the MTP request-batch condition execution role");
            return std::nullopt;
        }
        ScopedLiveMTPRequestBatchConditionPolicy live_condition_scope(
            graph_builder_.get(),
            &live_mtp_request_batch_condition_,
            live_request_batch_condition);
        if (!live_condition_scope.ready())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Graph builder rejected live MTP request-batch condition policy");
            return std::nullopt;
        }

        /*
         * All-position logits are an output policy, not an execution role:
         * both MTP prompt prefill and grouped verification use them. Publish
         * verifier ownership to the declarative graph explicitly for exactly
         * this execution, then restore the builder before any later graph is
         * constructed.
         */
        struct GroupedMTPVerifierScope
        {
            IGraphBuilder *builder = nullptr;
            bool active = false;
            bool enabled = true;

            GroupedMTPVerifierScope(
                IGraphBuilder *builder,
                ForwardExecutionRole execution_role)
                : builder(builder)
            {
                const bool grouped_verifier =
                    execution_role ==
                    ForwardExecutionRole::GroupedMTPVerifier;
                if (!grouped_verifier)
                    return;
                enabled =
                    builder &&
                    builder->setGroupedMTPVerifier(true);
                active = enabled;
            }

            ~GroupedMTPVerifierScope()
            {
                if (active && builder)
                    (void)builder->setGroupedMTPVerifier(false);
            }
        } grouped_verifier_scope(graph_builder_.get(), execution_role);
        if (!grouped_verifier_scope.enabled)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Graph builder rejected grouped MTP verifier policy");
            return std::nullopt;
        }

        struct MTPVerifierOutcomeGraphScope
        {
            IGraphBuilder *builder = nullptr;
            MTPVerifierOutcomeGraphMode mode =
                MTPVerifierOutcomeGraphMode::Disabled;
            bool installed = false;

            MTPVerifierOutcomeGraphScope(
                IGraphBuilder *graph_builder,
                ForwardExecutionRole role,
                MTPVerifierOutcomeGraphMode requested_mode)
                : builder(graph_builder),
                  mode(requested_mode)
            {
                if (mode != MTPVerifierOutcomeGraphMode::Disabled &&
                    role != ForwardExecutionRole::GroupedMTPVerifier)
                {
                    return;
                }
                installed =
                    builder &&
                    builder->setMTPVerifierOutcomeGraphMode(mode);
            }

            ~MTPVerifierOutcomeGraphScope()
            {
                if (installed && builder)
                {
                    (void)builder->setMTPVerifierOutcomeGraphMode(
                        MTPVerifierOutcomeGraphMode::Disabled);
                }
            }
        } verifier_outcome_scope(
            graph_builder_.get(),
            execution_role,
            mtp_verifier_outcome_graph_mode_);
        if (!verifier_outcome_scope.installed)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Graph builder rejected the MTP verifier outcome transaction"
                      << " mode="
                      << static_cast<int>(
                             mtp_verifier_outcome_graph_mode_)
                      << " grouped_role="
                      << (execution_role ==
                                  ForwardExecutionRole::GroupedMTPVerifier
                              ? "true"
                              : "false"));
            return std::nullopt;
        }

        int total_tokens = batch_size * seq_len;
        if (DebugEnv::isTruthyEnv(
                "LLAMINAR_MTP_MIRROR_LAYER_DIAGNOSTICS"))
        {
            const char *role_name = "unknown";
            switch (execution_role)
            {
            case ForwardExecutionRole::MainInference:
                role_name = "main_inference";
                break;
            case ForwardExecutionRole::GroupedMTPVerifier:
                role_name = "grouped_mtp_verifier";
                break;
            case ForwardExecutionRole::MTPCondition:
                role_name = "mtp_condition";
                break;
            }
            LOG_INFO(
                "[MirroredLayerDiagnostics] forward_role="
                << role_name
                << " device=" << state_.device_id.toString()
                << " batch_size=" << batch_size
                << " seq_len=" << seq_len
                << " total_tokens=" << total_tokens
                << " invocation=" << static_cast<int>(invocation)
                << " all_position_logits="
                << boolTag(compute_all_position_logits_)
                << " grouped_verifier_policy="
                << boolTag(
                       execution_role ==
                       ForwardExecutionRole::GroupedMTPVerifier)
                << " live_request_batch_condition="
                << boolTag(live_request_batch_condition));
        }
        if (total_tokens > state_.batch_size * state_.max_seq_len)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Total tokens " << total_tokens
                                                                << " exceeds buffer capacity "
                                                                << state_.batch_size * state_.max_seq_len);
            return std::nullopt;
        }
        const int activation_seq_len =
            state_.activation_seq_len > 0 ? state_.activation_seq_len : state_.max_seq_len;
        if (total_tokens > state_.batch_size * activation_seq_len)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Total tokens " << total_tokens
                                                                << " exceeds activation graph buffer capacity "
                                                                << state_.batch_size * activation_seq_len
                                                                << " (context capacity is "
                                                                << state_.batch_size * state_.max_seq_len
                                                                << "); run long prompts through prefill chunk scheduling "
                                                                   "or increase the activation arena length");
            return std::nullopt;
        }

        /*
         * Select phase from the typed execution role before considering row
         * geometry. Grouped verifiers are serial-decode-equivalent for every
         * M, so the ordinary MainInference continuation heuristic must never
         * switch M=5+ verifier graphs to tensor-parallel prefill. That old
         * crossover added one dense allreduce per layer plus GDN/KV state
         * allgathers and made dynamic MTP depth an accidental topology policy.
         *
         * MainInference still supports the compatibility heuristic for callers
         * that have not supplied an explicit phase: scalar work is decode, and
         * a bounded multi-row continuation with established history is decode.
         */
        const int decode_max_seq_len = std::max(1, cache_config_.decode_seq_len);
        const ForwardExecutionPhase resolved_phase =
            resolveForwardExecutionPhase({
                .role = execution_role,
                .invocation = invocation,
                .seq_len = seq_len,
                .batch_size = batch_size,
                .decode_max_seq_len = decode_max_seq_len,
                .logical_position = state_.positions[0],
            });
        const InferencePhase new_phase =
            resolved_phase == ForwardExecutionPhase::Decode
                ? InferencePhase::DECODE
                : InferencePhase::PREFILL;
        transitionToPhase(new_phase);

        /*
         * A GPU grouped verifier executes after accepted-state publication has
         * updated the canonical KV count entirely on device. The legacy
         * `state_.positions` vector is intentionally not adopted at that
         * boundary, so deriving verifier RoPE rows from it would replay the
         * previous transaction's positions. Bind the verifier graph to its
         * dedicated arena row instead; prepareAllPositionVerifierGraphMetadata()
         * fills that row from the live KV count on the graph's explicit stream.
         *
         * Token and position ownership move together. A pending verifier plan
         * with host token input remains the CPU/diagnostic contract and does not
         * silently enter this device-resident path.
         */
        const bool use_device_resident_mtp_verifier_positions =
            state_.device_id.is_gpu() &&
            token_ids_device != nullptr &&
            pending_mtp_spec_verifier_input_plan_.has_value();
        const void *effective_position_ids_device =
            position_ids_device_override;
        const int32_t *effective_sequence_lengths_device =
            sequence_lengths_device_override;
        bool materialize_serial_decode_position_from_device_kv = false;
        if (use_device_resident_mtp_verifier_positions)
        {
            if (!mtp_verifier_position_ids_dev_ ||
                !mtp_verifier_request_lengths_dev_ ||
                total_tokens > stochastic_target_row_capacity_)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Device-resident grouped verifier geometry is missing or exceeds persistent arena capacity: rows="
                          << total_tokens
                          << " capacity=" << stochastic_target_row_capacity_);
                return std::nullopt;
            }
            if (position_ids_device_override &&
                position_ids_device_override !=
                    mtp_verifier_position_ids_dev_)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Grouped verifier received a competing device position owner");
                return std::nullopt;
            }
            if (sequence_lengths_device_override &&
                sequence_lengths_device_override !=
                    mtp_verifier_request_lengths_dev_)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Grouped verifier received a competing device request-length owner");
                return std::nullopt;
            }
            effective_position_ids_device =
                mtp_verifier_position_ids_dev_;
        }
        else if (state_.device_id.is_gpu() &&
                 new_phase == InferencePhase::DECODE &&
                 execution_role == ForwardExecutionRole::MainInference &&
                 batch_size == 1 &&
                 seq_len == 1 &&
                 !effective_position_ids_device)
        {
            /*
             * The live KV count is the authoritative next position, but it is
             * mutable inside the forward graph. Bind the stable request arena
             * row into every serial-decode capture; its captured root takes a
             * device snapshot before attention can advance the cache. Retained
             * parent iterations therefore need no host prelude to refresh it.
             * RoPE and replicated MoE routing consequently consume
             * the same immutable logical position without a host upload.
             */
            if (!request_position_ids_dev_ ||
                !state_.kv_cache ||
                !state_.kv_cache
                     ->deviceSequenceCachedTokenCountPtr(/*seq_idx=*/0))
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] GPU serial decode requires a "
                    "stable arena position row and device-owned KV count");
                return std::nullopt;
            }
            effective_position_ids_device = request_position_ids_dev_;
            materialize_serial_decode_position_from_device_kv = true;
        }

        std::optional<SparseQueryPrefillProof> cpu_prefill_proof;
        if (new_phase == InferencePhase::PREFILL && state_.device_id.is_cpu() &&
            batch_size == 1 && sparseQueryProgramCount() > 1u)
            cpu_prefill_proof = SparseQueryPrefillProof::admittedRange(state_.positions[0], seq_len);

        // Build host position rows only for a host-owned forward transaction.
        // Device-resident MTP condition rows already carry their authoritative
        // positions in the logical-state mailbox; constructing a host shadow
        // here would create an allocation and a second, stale source of truth.
        std::vector<int> position_ids;
        if (restored_prefix_mtp_decode_bridge &&
            state_.device_id.is_gpu())
        {
            /*
             * Request admission copies the new token and a temporary position
             * as one immutable host-I/O transaction. The captured decode
             * prelude replaces that temporary row from canonical device KV
             * before either RoPE or the shifted-MTP bridge reads it.
             */
            position_ids.push_back(state_.positions[0]);
        }
        else if (!effective_position_ids_device)
        {
            position_ids.reserve(total_tokens);
            for (int b = 0; b < batch_size; ++b)
            {
                const int pos_offset = state_.positions[b];
                for (int s = 0; s < seq_len; ++s)
                    position_ids.push_back(pos_offset + s);
            }
        }

        /*
         * External prompt or teacher-forced token rows cross into the GPU once
         * as a single admission transaction. From this point onward, embedding, RoPE, recurrent
         * stages, and shifted-MTP publication all consume the same persistent
         * arena rows. Grouped verifiers and live request-batch decode already
         * arrive with their own device-owned token/position sources and must
         * never overwrite the request-admission bank.
        */
        const void *effective_token_ids_device = token_ids_device;
        const bool uses_request_input_bank =
            state_.device_id.is_gpu() &&
            (execution_role == ForwardExecutionRole::MainInference ||
             restored_prefix_mtp_decode_bridge) &&
            !effective_token_ids_device;
        if (uses_request_input_bank)
        {
            std::optional<DeviceDecodePositionBinding> admitted_decode_position;
            if(materialize_serial_decode_position_from_device_kv)
                admitted_decode_position=DeviceDecodePositionBinding{
                    .backend=getBackendFor(state_.device_id),
                    .cached_tokens=state_.kv_cache->deviceSequenceCachedTokenCountPtr(0),
                    .position=static_cast<int32_t *>(request_position_ids_dev_)};
            if (!admitRequestInputsOnDevice(
                    tokens,
                    admitted_decode_position ? nullptr : position_ids.data(),
                    request_real_lengths,
                    total_tokens,
                    batch_size,
                    seq_len,
                    admitted_decode_position))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Main forward could not admit its complete request input");
                return std::nullopt;
            }
            effective_token_ids_device = request_token_ids_dev_;
            effective_position_ids_device = request_position_ids_dev_;
        }

        if (restored_prefix_mtp_decode_bridge &&
            state_.device_id.is_gpu())
        {
            /*
             * The request bank is the immutable host-I/O ingress, not the
             * main graph's logical-state authority. Compose its admitted token
             * with the canonical device KV count into the same durable mailbox
             * used by every ordinary MTP-condition advance. The shifted
             * sidecar continues to consume the original one-row request bank,
             * while main attention, recurrent state, routing, and logits all
             * consume this event-published condition transaction. CPU owns the
             * same typed bridge as host state: it executes the ordinary
             * one-row decode below and then advances shifted MTP state through
             * populateCPUMTPShiftedCacheFromPrefill(). It must never be forced
             * through this device-mailbox publication transition.
             */
            if (!state_.kv_cache ||
                !request_input_admission_ready_.valid ||
                !request_input_admission_ready_.producer_stream)
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Restored-prefix MTP bridge "
                    "has no live request admission or canonical KV owner");
                return std::nullopt;
            }
            const int32_t *const live_position_device =
                state_.kv_cache->deviceSequenceCachedTokenCountPtr(
                    /*seq_idx=*/0);
            if (!live_position_device ||
                !publishDeviceResidentLogicalSequenceStateFromDeviceRows(
                    static_cast<const int32_t *>(request_token_ids_dev_),
                    live_position_device,
                    /*request_count=*/1,
                    request_input_admission_ready_.producer_stream,
                    DeviceResidentLogicalStatePublicationKind::
                        RestoredPrefixDecodeBridgeInitialization,
                    "restored_prefix_request_admission"))
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Restored-prefix MTP bridge "
                    "could not publish its device logical-state transaction");
                return std::nullopt;
            }
            const DeviceResidentLogicalSequenceStateHandle logical_state =
                deviceResidentLogicalSequenceState();
            if (!logical_state.coversRequest(/*request_index=*/0) ||
                !device_resident_logical_sequence_state_mailbox_.ownsHandle(
                    logical_state,
                    live_replay_state_epoch_))
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Restored-prefix MTP bridge "
                    "published an incomplete logical-state mailbox");
                return std::nullopt;
            }
            effective_token_ids_device =
                logical_state.nextConditionTokenDeviceForRequest(0);
            effective_position_ids_device =
                logical_state.targetPositionDeviceForRequest(0);
            effective_sequence_lengths_device =
                logical_state.targetSequenceLengthDeviceForRequest(0);
        }

        TensorBase *logits_output = state_.logits.get();
        TensorBase *logits_local_output = state_.logits_local.get();
        if (compute_all_position_logits_ ||
            live_mtp_request_batch_condition_)
        {
            const auto &config = graph_builder_->config();
            size_t rows = static_cast<size_t>(total_tokens);
            if (compute_all_position_logits_ &&
                compute_row_indexed_all_position_logits_)
            {
                if (batch_size > 1 &&
                    !pending_mtp_spec_verifier_input_plan_ &&
                    request_batched_prefill_logits_row_count_ <= 0)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Batched row-indexed logits require an explicit MTP verifier or request-prefill row plan");
                    return std::nullopt;
                }
                if (row_indexed_all_position_logits_row_count_ <= 0 ||
                    row_indexed_all_position_logits_row_count_ > total_tokens)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Invalid row-indexed all-position logit row count: rows="
                              << row_indexed_all_position_logits_row_count_
                              << " total_tokens=" << total_tokens);
                    return std::nullopt;
                }
                rows = static_cast<size_t>(row_indexed_all_position_logits_row_count_);
            }
            if (!bindAllPositionLogitsOutputs(
                    execution_role,
                    rows,
                    logits_output,
                    logits_local_output))
            {
                return std::nullopt;
            }
        }

        // Prepare model buffers from state
        ModelBuffers model_buffers = state_.toModelBuffers();
        model_buffers.logits = logits_output;
        model_buffers.logits_local = logits_local_output;

        setBuffers(model_buffers);

        // Build forward input
        ForwardInput input;
        input.token_ids = effective_token_ids_device ? nullptr : tokens;
        input.token_ids_device = effective_token_ids_device;
        input.position_ids = effective_position_ids_device
                                 ? nullptr
                                 : position_ids.data();
        input.position_ids_device = effective_position_ids_device;
        if (materialize_serial_decode_position_from_device_kv)
            input.device_decode_position = DeviceDecodePositionBinding{
                .backend = getBackendFor(state_.device_id),
                .cached_tokens = state_.kv_cache->deviceSequenceCachedTokenCountPtr(0),
                .position = static_cast<int32_t *>(request_position_ids_dev_)};
        input.batch_size = batch_size;
        input.seq_len = seq_len;
        const bool bounded_scalar_gpu_verifier =
            execution_role == ForwardExecutionRole::GroupedMTPVerifier &&
            state_.device_id.is_gpu() && batch_size == 1 &&
            !request_real_lengths.empty();
        if (bounded_scalar_gpu_verifier)
        {
            input.real_seq_len = request_real_lengths.front();
            input.bucket_seq_len =
                input.real_seq_len < seq_len ? seq_len : 0;
        }
        /*
         * The orchestrator owns the request-local logical cursor.  Publish it
         * through both fields while the lower graph APIs still carry the legacy
         * `position_offset` fallback: `token_offset` is the absolute prefill
         * range start used by bucket planning, replay metadata, and restored
         * prefix suffixes; `position_offset` remains the decode/RoPE fallback.
         */
        input.position_offset = live_request_batch_condition
                                    ? 0
                                    : state_.positions[0];
        input.token_offset = input.position_offset;
        const auto sparse_request_identity =
            currentMoESparseRequestIdentity();
        input.moe_overlay_collective_generation_id =
            sparse_request_identity
                ? sparse_request_identity->generation()
                : 0;
        input.moe_overlay_collective_step_id =
            static_cast<uint64_t>(std::max(input.token_offset, 0));
        if (execution_role == ForwardExecutionRole::GroupedMTPVerifier)
        {
            const int logical_verifier_rows = *std::max_element(
                request_real_lengths.begin(), request_real_lengths.end());
            input.moe_overlay_mtp_depth = logical_verifier_rows - 1;
            if (input.moe_overlay_mtp_depth <= 0)
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Grouped MTP verifier requires an admitted positive draft depth");
                return std::nullopt;
            }
        }
        input.execution_role = execution_role;
        input.execution_phase =
            new_phase == InferencePhase::DECODE
                ? ForwardExecutionPhase::Decode
                : ForwardExecutionPhase::Prefill;
        // CPU request positions are host-owned. GPU prefill uses the original
        // accepted ingress proof in forwardPrefillChunkSchedule instead.
        if (cpu_prefill_proof) {
            input.sparse_query_prefill_proof = cpu_prefill_proof;
            input.sparse_query_execution_program = input.sparse_query_prefill_proof->program(
                qwen4exp::QwenSparseFirstRankingLogicalRow);
        }
        input.state_transaction =
            restored_prefix_mtp_decode_bridge
                ? ForwardStateTransaction::RestoredPrefixMTPDecodeBridge
                : committed_mtp_condition
                      ? ForwardStateTransaction::CommittedMTPCondition
                      : ForwardStateTransaction::Ordinary;
        /*
         * The graph builder is the single owner of this transaction.  A full
         * prefix hit may restore portable MoE placement and return terminal
         * logits without launching a main forward graph.  In that case the
         * next request reset retires the pending model transaction.  Keeping a
         * second orchestrator flag made that legal lifecycle look pending
         * forever and caused the following request to disagree with the model.
         */
        input.rehydrate_prefix_runtime_on_device =
            (execution_role == ForwardExecutionRole::MainInference ||
             restored_prefix_mtp_decode_bridge) &&
            graph_builder_ &&
            graph_builder_->
                prefixCacheRuntimeStateRequiresDeviceRehydration();
        input.device_state_publication_stream =
            state_.device_id.is_gpu()
                ? explicitGPUStreamForOperation(
                      "forward_graph_device_state_publication")
                : nullptr;
        if (state_.device_id.is_gpu() &&
            !input.device_state_publication_stream)
        {
            throw std::runtime_error(
                "GPU forward input construction requires an explicit "
                "device-state publication stream");
        }
        input.device = state_.device_id;
        input.kv_cache = state_.kv_cache.get();

        /*
         * Pipeline prefill binds each non-embedding participant's persistent
         * receive bank as its activation input.  MTP condition and verifier
         * forwards traverse those same layers and native P2P edges, so they
         * must use the exact same binding rather than falling through to the
         * legacy setHiddenState() transport.  The bank is local arena state:
         * PipelineForwardGraphEdges records the preceding receive before this
         * graph's roots consume it, preserving device-only ordering without a
         * host activation mirror.
         *
         * An independently supplied activation is a different authority and
         * cannot be mixed into a frozen graph-native pipeline.  Reject it at
         * the construction boundary instead of allowing executeForward() to
         * replace a captured pointer after graph identity has been chosen.
         */
        const bool consumes_pipeline_activation =
            pp_stage_config_.has_value() &&
            !pp_stage_config_->has_embedding &&
            pipeline_forward_edges_ != nullptr;
        if (consumes_pipeline_activation)
        {
            if (!state_.hidden)
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Graph-native PP forward "
                    "requires its persistent activation receive bank");
                return std::nullopt;
            }
            if (external_hidden_state_input_ &&
                external_hidden_state_input_ != state_.hidden.get())
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Graph-native PP forward "
                    "received an activation outside its frozen receive "
                    "bank");
                return std::nullopt;
            }
            input.external_hidden_state = state_.hidden.get();
        }

        // Pass sequence_lengths for batch-aware attention masking
        // This enables proper separation of sequences in batched execution
        std::vector<int> transaction_request_lengths;
        if (invocation == ForwardInvocationKind::ExplicitDecode &&
            batch_size > 1)
        {
            /*
             * Request-batched decode appends one row to each independent KV
             * sequence. Persistent logical lengths live in the device mailbox;
             * this host vector describes only the fixed graph shape and must
             * never be populated from those mutable device values.
             */
            transaction_request_lengths.assign(
                static_cast<size_t>(batch_size),
                seq_len);
            input.sequence_lengths = &transaction_request_lengths;
        }
        else if (batch_size > 1)
        {
            transaction_request_lengths.assign(
                request_real_lengths.begin(),
                request_real_lengths.end());
            input.sequence_lengths = &transaction_request_lengths;
        }
        else
        {
            input.sequence_lengths = nullptr;
        }
        if (execution_role == ForwardExecutionRole::GroupedMTPVerifier &&
            state_.device_id.is_gpu())
        {
            input.sequence_lengths_device =
                static_cast<const int32_t *>(
                    mtp_verifier_request_lengths_dev_);
            // The admitted first transaction and its cold siblings borrow
            // the same count row and must name the same native graph key.
            if (use_device_resident_mtp_verifier_positions)
                input.captured_verifier_extent = retainedMTPVerifierExtent(input.seq_len);
        }
        else if (effective_sequence_lengths_device)
        {
            input.sequence_lengths_device =
                effective_sequence_lengths_device;
        }
        else if (new_phase == InferencePhase::PREFILL &&
                 request_sequence_lengths_active_count_ >= batch_size)
        {
            input.sequence_lengths_device =
                static_cast<const int32_t *>(
                    request_sequence_lengths_dev_);
        }
        else
        {
            input.sequence_lengths_device = nullptr;
        }

        if (!bindShiftedMTPPrefillTransaction(input, batch_size))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Main forward could not bind its graph-integrated shifted MTP prefill transaction");
            return std::nullopt;
        }
        if (!bindMTPMainTerminalHiddenTransaction(input, batch_size))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Main forward could not bind its graph-integrated MTP terminal-hidden publication");
            return std::nullopt;
        }

        // Build forward output
        ForwardOutput output;
        output.logits = logits_output;
        output.hidden = state_.hidden.get();

        std::vector<int> last_forward_request_lengths(
            request_real_lengths.begin(),
            request_real_lengths.end());

        const int logical_rows_per_request = *std::max_element(
            request_real_lengths.begin(), request_real_lengths.end());
        const MoEOverlayInferenceGraphRole overlay_graph_role =
            execution_role == ForwardExecutionRole::GroupedMTPVerifier
                ? MoEOverlayInferenceGraphRole::MTPGroupedVerifier
                : (input.execution_phase == ForwardExecutionPhase::Prefill
                       ? MoEOverlayInferenceGraphRole::MainPrefill
                       : MoEOverlayInferenceGraphRole::MainDecode);
        int overlay_physical_rows_per_request = seq_len;
        if (remote_prefill_geometry)
        {
            if (overlay_graph_role !=
                    MoEOverlayInferenceGraphRole::MainPrefill ||
                batch_size != 1 ||
                remote_prefill_geometry->physical_rows_per_request <
                    logical_rows_per_request)
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Remote prefill transaction "
                    "geometry is incompatible with the local exact-shape graph"
                    << " logical_rows=" << logical_rows_per_request
                    << " physical_rows="
                    << remote_prefill_geometry
                           ->physical_rows_per_request
                    << " requests=" << batch_size);
                return std::nullopt;
            }
            overlay_physical_rows_per_request =
                remote_prefill_geometry->physical_rows_per_request;
        }
        else if (moe_overlay_inference_transaction_coordinator_ &&
            overlay_graph_role ==
                MoEOverlayInferenceGraphRole::MainPrefill &&
            state_.device_id.is_gpu() && batch_size == 1 &&
            cache_config_.enabled &&
            debugEnv().execution.gpu_graphs &&
            debugEnv().execution.prefill_graph_buckets &&
            !pp_stage_config_.has_value())
        {
            /*
             * Transaction admission precedes ForwardExecutionEngine::execute,
             * so it must resolve the same physical capture bucket here. The
             * authenticated mapped-activation ABI intentionally rejects a
             * follower that guesses a larger executable after seeing only the
             * real row count.
             */
            const auto raw_buckets =
                rawPrefillGraphBucketsForResidentCapacity(
                    debugEnv().execution.prefill_graph_bucket_sizes,
                    residentGraphRows(),
                    debugEnv().execution.prefill_graph_min_seq);
            const auto selected = selectPrefillGraphBucket(
                seq_len, raw_buckets);
            if (!selected)
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] ExpertOverlay transaction "
                    "could not resolve the forward engine's physical prefill "
                    "bucket: "
                    << selected.error << " real_rows=" << seq_len
                    << " resident_rows=" << residentGraphRows());
                return std::nullopt;
            }
            overlay_physical_rows_per_request =
                selected.bucket_seq_len;
        }
        MoEOverlayInferenceExecutionDescriptor overlay_descriptor{
            .graph_role = overlay_graph_role,
            .logical_step_id = 0,
            .placement_epoch =
                moe_overlay_inference_transaction_coordinator_
                    ? moe_overlay_inference_transaction_coordinator_
                          ->currentPlacementEpoch()
                    : 0,
            .request_count = batch_size,
            .logical_rows_per_request = logical_rows_per_request,
            .physical_rows_per_request =
                overlay_physical_rows_per_request,
            .draft_depth =
                execution_role == ForwardExecutionRole::GroupedMTPVerifier
                    ? input.moe_overlay_mtp_depth
                    : -1,
            .sidecar_depth = -1,
        };
        if (moe_overlay_inference_transaction_coordinator_ &&
            overlay_descriptor.graph_role ==
                MoEOverlayInferenceGraphRole::MainPrefill)
        {
            std::string sequence_error;
            if (!moe_overlay_inference_transaction_coordinator_
                     ->admitSerialPrefillGraph(
                         moe_overlay_inference_transaction_participant_index_,
                         &sequence_error))
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] ExpertOverlay serial prefill "
                    "graph-sequence admission failed: "
                    << sequence_error);
                return std::nullopt;
            }
        }
        MoEOverlayInferenceParticipantGraphScope overlay_graph_scope(
            moe_overlay_inference_transaction_coordinator_,
            overlay_descriptor,
            moe_overlay_inference_transaction_participant_index_);
        if (!overlay_graph_scope.ready())
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] ExpertOverlay transaction "
                "authority rejected the forward graph: "
                << overlay_graph_scope.error());
            return std::nullopt;
        }
        if (overlay_graph_scope.active())
        {
            const auto &binding = overlay_graph_scope.binding();
            if (moe_overlay_collective_request_generation_ == 0 ||
                moe_overlay_collective_request_generation_ !=
                    binding.request_generation)
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Graph transaction generation "
                    "disagrees with the request generation installed at "
                    "admission");
                return std::nullopt;
            }
            input.moe_overlay_collective_generation_id =
                binding.request_generation;
            input.moe_overlay_collective_step_id =
                binding.descriptor.logical_step_id;
            input.moe_overlay_sequence_id = binding.sequence_id;
            input.moe_overlay_sequence_placement_epoch =
                binding.descriptor.placement_epoch;
            input.moe_overlay_sequence_graph_ordinal =
                binding.sequence_graph_ordinal;
            input.moe_overlay_sequence_graph_count =
                binding.sequence_graph_count;
            if (execution_role ==
                ForwardExecutionRole::GroupedMTPVerifier)
            {
                input.moe_overlay_mtp_depth =
                    binding.descriptor.draft_depth;
            }

            if (overlay_graph_scope.ownsTicketAuthority() &&
                input.device.is_gpu())
            {
                /*
                 * Reserve/stamp the sparse transaction before graph construction,
                 * but wake the remote follower only after native capture and
                 * instantiation have completed. The graph executor invokes this
                 * callback on the exact stream immediately before transaction
                 * zero or steady replay, closing the cold-capture timeout race.
                 */
                input.moe_overlay_graph_launch_dependency =
                    [&overlay_graph_scope, device = input.device](
                        DeviceGraphExecutor::GraphExecutableLaunchPhase,
                        void *execution_stream) -> bool
                {
                    std::string arm_error;
                    if (!overlay_graph_scope.armForExecutableLaunch(
                            execution_stream, &arm_error))
                    {
                        LOG_ERROR(
                            "[DeviceGraphOrchestrator] ExpertOverlay graph "
                            "could not arm its remote follower at the "
                            "executable launch edge on "
                            << device.toString() << ": " << arm_error);
                        return false;
                    }
                    return true;
                };
            }
            else if (overlay_graph_scope.ownsTicketAuthority())
            {
                std::string arm_error;
                if (!overlay_graph_scope.armForHostExecution(&arm_error))
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] CPU ExpertOverlay graph could "
                        "not arm its remote follower at the host execution edge: "
                        << arm_error);
                    return std::nullopt;
                }
            }
        }

        // Execute forward pass
        bool success = executeForward(input, output);

        std::string overlay_finish_error;
        if (!overlay_graph_scope.finish(
                success, &overlay_finish_error))
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] ExpertOverlay transaction "
                "authority could not finish the forward graph: "
                << overlay_finish_error);
            return std::nullopt;
        }

        if (!success)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] forward() execution failed");
            const auto *const moe = graph_builder_
                                        ? &graph_builder_->config().moe
                                        : nullptr;
            if (moe && moe->device_controller_fabric)
            {
                /*
                 * A retained parent can fail before its first sparse ticket is
                 * visible when a continuation epoch acquire is stalled. The
                 * authority rank owns the mapped LocalTP barrier, so report it
                 * here at the terminal forward boundary; follower ranks cannot
                 * inspect this process-local fabric.
                 */
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Fatal forward epoch evidence device="
                    << state_.device_id.toString() << ' '
                    << moe->device_controller_fabric
                           ->describeInferenceEpochBarrier());
            }
            return std::nullopt;
        }

        if (execution_role == ForwardExecutionRole::GroupedMTPVerifier &&
            state_.device_id.is_gpu())
        {
            std::string control_policy_error;
            const auto control_policy =
                mtpVerifierPreparationControlPolicyForRequest(
                    batch_size,
                    &control_policy_error);
            if (!control_policy)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Grouped verifier completed with malformed controller admission: "
                          << control_policy_error);
                return std::nullopt;
            }
            const MTPVerifierPreparationGraphKey preparation_key{
                .control_policy = *control_policy,
                .request_count = batch_size,
                .padded_seq_len = seq_len,
            };
            std::string pairing_error;
            if (!bindMTPVerifierForwardGraphPair(
                    preparation_key,
                    &pairing_error))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Grouped verifier completed without publishing an exact preparation/forward capture pair: "
                          << pairing_error);
                return std::nullopt;
            }
        }

        if (input.rehydrate_prefix_runtime_on_device)
        {
            graph_builder_->
                completePrefixCacheRuntimeStateDeviceRehydration();
        }

        /*
         * Bind failure diagnostics to the exact graph role that just
         * completed. A grouped verifier is not merely a target-logit
         * producer: accepted-state publication can select its hidden rows
         * into PREFIX_TERMINAL_HIDDEN for the next speculative step. Its
         * graph-captured checkpoint bank must therefore supersede the older
         * prefill bank when it is the latest successful hidden producer.
         */
        const char *graph_regime =
            execution_role == ForwardExecutionRole::GroupedMTPVerifier
                ? "grouped_verifier"
                : execution_role == ForwardExecutionRole::MTPCondition
                      ? batch_size > 1
                            ? "request_batch_decode"
                            : "scalar_condition_decode"
                      : batch_size * seq_len == 1
                            ? "serial_decode"
                            : "prefill";
        if (output.execution.graph_seq_len <= 0 ||
            output.execution.graph_batch_size <= 0)
        {
            throw std::runtime_error(
                "Successful forward omitted its captured graph geometry from "
                "execution provenance");
        }
        last_mirrored_layer_checkpoint_prefix_ =
            std::string(graph_regime) +
            "_m" +
            std::to_string(
                output.execution.graph_batch_size *
                output.execution.graph_seq_len) +
            "_";

        const bool serial_layer_diagnostics =
            DebugEnv::isTruthyEnv(
                "LLAMINAR_MTP_MIRROR_LAYER_DIAGNOSTICS");
        const bool serial_state_diagnostics =
            DebugEnv::isTruthyEnv(
                "LLAMINAR_MTP_TERMINAL_TRANSACTION_DIAGNOSTICS");
        if ((serial_layer_diagnostics || serial_state_diagnostics) &&
            std::string_view(graph_regime) == "serial_decode")
        {
            if (!output.execution.valid || !output.execution.stream)
            {
                throw std::runtime_error(
                    "Serial layer diagnostics require exact GPU producer "
                    "stream provenance");
            }

            /*
             * This intentionally expensive oracle runs only under the explicit
             * diagnostics switch. Enqueuing each read on the graph's own
             * producer stream makes the observed row an exact post-forward
             * boundary without adding an unordered cross-stream probe.
             */
            const int input_token = tokens ? tokens[0] : -1;
            const int input_position =
                state_.positions.empty() ? -1 : state_.positions.front();
            for (const auto &digest :
                 captureMirroredMTPDigests(output.execution.stream))
            {
                const bool layer_digest =
                    serial_layer_diagnostics &&
                    digest.name.starts_with("main_checkpoint_");
                const bool state_digest =
                    serial_state_diagnostics &&
                    (digest.name.starts_with("main_live_kv_") ||
                     digest.name.starts_with("main_live_gdn_") ||
                     digest.name == "main_live_position");
                if (!layer_digest && !state_digest)
                    continue;
                LOG_INFO(
                    (layer_digest
                         ? "[MTPSerialLayerDigest] device="
                         : "[MTPSerialStateDigest] device=")
                    << state_.device_id.toString()
                    << " input_token=" << input_token
                    << " input_position=" << input_position
                    << " name=" << digest.name
                    << " available="
                    << (digest.available ? "true" : "false")
                    << " bytes=" << digest.byte_count
                    << " hash=0x" << std::hex << digest.hash << std::dec
                    << (digest.value_preview.empty()
                            ? std::string{}
                            : " values=" + digest.value_preview));
            }
        }

        // A resident MTP condition advances canonical positions in its device
        // logical-state publication kernel. Never maintain a competing host
        // cursor for that transaction.
        if (!live_request_batch_condition)
        {
            for (int b = 0; b < batch_size; ++b)
            {
                const int logical_rows =
                    request_real_lengths[static_cast<size_t>(b)];
                state_.positions[b] += logical_rows;
                state_.sequence_lengths[b] += logical_rows;
            }
        }

        const bool main_prefill =
            new_phase == InferencePhase::PREFILL &&
            execution_role == ForwardExecutionRole::MainInference;
        const bool graph_integrated_shifted_mtp =
            isGraphIntegratedShiftedMTPTransaction(
                input.execution_role,
                input.execution_phase,
                input.state_transaction,
                input.batch_size,
                input.seq_len);
        // The restored-prefix scalar bridge traverses every PP stage, but
        // only a terminal participant owns shifted KV and its predictor.
        // Use the same runtime role as chunked prefill; enabled MTP alone
        // includes upstream verifier-only participants with no shifted bank.
        const auto mtp_runtime_role = resolveMTPRuntimeTransactionRole(
            graph_builder_ && graph_builder_->config().mtp.enabled,
            !pp_stage_config_ || pp_stage_config_->has_lm_head);
        if (graph_integrated_shifted_mtp &&
            state_.device_id.is_gpu() &&
            mtp_runtime_role == MTPRuntimeTransactionRole::PredictorOwner)
        {
            if (!input.shifted_mtp_prefill.has_value())
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Enabled GPU shifted-MTP "
                    "producer completed without its graph-integrated "
                    "shifted-KV transaction");
                return std::nullopt;
            }
            /*
             * HiddenStateRowsSelectStage archived the terminal main-model row
             * before the depth-zero KV append in this same captured graph.
             * Publish that fact for both legal producers.  Omitting the
             * restored-prefix edge makes harvest run refreshMTPTerminalHiddenState(),
             * whose sidecar projection can overwrite the main-logits surface
             * before the terminal cache block copies it.
             */
            state_.mtp_terminal_hidden_publication.publishMainForward();
            if (restored_prefix_mtp_decode_bridge)
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "restored_prefix_decode_bridge_transactions",
                    1.0,
                    "decode",
                    state_.device_id.toString(),
                    {{"history", std::to_string(state_.positions.front())},
                     {"state_owner", "captured_main_graph"},
                     {"terminal_hidden", "graph_integrated"}});
            }
            else
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "shifted_prefill_integrated_graph_transactions",
                    1.0,
                    perfPhaseName(),
                    state_.device_id.toString(),
                    {{"requests", std::to_string(batch_size)},
                     {"bucket_rows", std::to_string(seq_len)}});
            }
        }
        else if (main_prefill &&
                 !populateCPUMTPShiftedCacheFromPrefill(
                     tokens,
                     seq_len,
                     batch_size,
                     input.position_offset,
                     &last_forward_request_lengths,
                     &position_ids))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to populate MTP shifted prefill cache");
            return std::nullopt;
        }
        else if (restored_prefix_mtp_decode_bridge &&
                 state_.device_id.is_cpu() &&
                 !populateCPUMTPShiftedCacheFromPrefill(
                     tokens,
                     seq_len,
                     batch_size,
                     input.position_offset,
                     &last_forward_request_lengths,
                     &position_ids))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] CPU restored-prefix decode could not bridge shifted MTP state");
            return std::nullopt;
        }

        if (uses_request_input_bank ||
            materialize_serial_decode_position_from_device_kv)
        {
            publishRequestInputReuseReady(
                output.execution.stream,
                uses_request_input_bank
                    ? (restored_prefix_mtp_decode_bridge
                           ? "restored_prefix_mtp_decode_bridge"
                           : new_phase==InferencePhase::DECODE ? "main_decode_transaction" : "main_prefill_transaction")
                    : "serial_decode_position_transaction");
        }

        if (execution_role == ForwardExecutionRole::MainInference ||
            execution_role == ForwardExecutionRole::MTPCondition)
        {
            const bool terminal_hidden_archived =
                input.shifted_mtp_prefill.has_value() ||
                input.mtp_main_terminal_hidden.has_value() ||
                (main_prefill && state_.device_id.is_cpu() &&
                 graph_builder_ && graph_builder_->config().mtp.enabled &&
                 state_.mtp_terminal_hidden_publication.current()) ||
                (restored_prefix_mtp_decode_bridge &&
                 state_.device_id.is_cpu() && graph_builder_ &&
                 graph_builder_->config().mtp.enabled &&
                 state_.mtp_terminal_hidden_publication.current());
            noteMainForwardHiddenProducedForMTP(
                seq_len,
                batch_size,
                std::move(last_forward_request_lengths),
                terminal_hidden_archived
                    ? MTPTerminalHiddenPublication::Source::MainForward
                    : MTPTerminalHiddenPublication::Source::Unavailable);
        }

        // After first prefill, release host-resident weight data. Keep the host
        // bytes alive until MTP shifted-prefill has materialized its sidecar
        // weights through normal stage-contract coherence.
        if (release_host_resident_after_forward_ && !host_resident_released_ && seq_len > 1 && weight_manager_)
        {
            host_resident_released_ = true;
            scheduleMmapReclaimAfterFirstPrefill();
        }

        LOG_TRACE("[FORWARD_TRACE] seq_len=" << seq_len
                                             << " pos_offset=" << input.position_offset
                                             << " token_ids[0]=" << (tokens ? tokens[0] : -1)
                                             << " positions_after=" << state_.positions[0]);

        // Return logits pointer
        // GPU: logits remain on device (DEVICE_AUTHORITATIVE) — avoid massive D2H transfer.
        // Callers that need host data should call logits() explicitly.
        // CPU: logits are already on host, fp32_data() is essentially free.
        if (state_.device_id.is_gpu())
            return reinterpret_cast<const float *>(logits_output);
        return logits_output ? logits_output->fp32_data() : nullptr;
    }

    bool DeviceGraphOrchestrator::supportsPrefillChunkSchedule(int seq_len) const
    {
        const auto &env = debugEnv().execution;
        if (seq_len <= 0 ||
            !state_.isInitialized() ||
            compute_all_position_logits_)
        {
            return false;
        }

        /*
         * An eager CPU graph uses the same canonical segmentation boundaries
         * as a captured GPU graph, but executes only the live rows in each
         * segment.  The scheduler's bucket remains the authenticated remote
         * transaction capacity; it is not a request to pad CPU arithmetic.
         */
        if (state_.device_id.is_cpu())
        {
            return seq_len <= state_.max_seq_len &&
                   state_.activation_seq_len > 0;
        }
        if (!state_.device_id.is_gpu() ||
            !cache_config_.enabled ||
            !env.gpu_graphs ||
            !env.prefill_graph_buckets)
        {
            return false;
        }

        const auto buckets = prefillGraphBucketsAtOrBelowCapacity(
            env.prefill_graph_bucket_sizes,
            state_.activation_seq_len);
        return !buckets.empty();
    }

    std::optional<DevicePrefillChunkGraphBinding>
    DeviceGraphOrchestrator::makeDevicePrefillChunkGraphBinding(
        int bucket_seq_len,
        int pad_token_id)
    {
        IBackend *const backend = getBackendFor(state_.device_id);
        const int32_t *const cached_tokens_device =
            state_.kv_cache
                ? state_.kv_cache->deviceSequenceCachedTokenCountPtr(
                      /*seq_idx=*/0)
                : nullptr;
        const uint64_t generation = workspaceGeneration(state_.device_id);
        if (!state_.device_id.is_gpu() || !backend ||
            !cached_tokens_device || generation == 0 ||
            bucket_seq_len <= 0 ||
            bucket_seq_len > prefill_chunk_row_capacity_ ||
            !request_token_ids_dev_ || !request_position_ids_dev_ ||
            !request_sequence_lengths_dev_ ||
            !prefill_chunk_token_ids_dev_ ||
            !prefill_chunk_position_ids_dev_ ||
            !prefill_chunk_real_rows_dev_ ||
            !prefill_chunk_row_stride_dev_)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Captured prefill chunk binding "
                "lacks a permanent owner or exceeds admitted capacity"
                << " device=" << state_.device_id.toString()
                << " backend=" << backend
                << " cached_tokens=" << cached_tokens_device
                << " workspace_generation=" << generation
                << " bucket_rows=" << bucket_seq_len
                << " resident_rows=" << prefill_chunk_row_capacity_);
            return std::nullopt;
        }

        /*
         * FNV-1a is used only as a stable in-process cache identity.  Every
         * immediate graph parameter and every captured permanent pointer is
         * included.  Live request values are intentionally absent: they are
         * published into these allocations before each replay.
         */
        uint64_t capture_identity = UINT64_C(1469598103934665603);
        auto mix_u64 = [&](uint64_t value)
        {
            capture_identity ^= value;
            capture_identity *= UINT64_C(1099511628211);
        };
        auto mix_pointer = [&](const void *pointer)
        {
            mix_u64(static_cast<uint64_t>(
                reinterpret_cast<uintptr_t>(pointer)));
        };
        mix_u64(generation);
        const auto placement = graph_builder_ ? graph_builder_->config().ngram_table_placement : nullptr;
        const bool require_exact_rows = placement && placement->fixedHeadPipeline();
        mix_u64(static_cast<uint64_t>(bucket_seq_len));
        if (require_exact_rows) mix_u64(1);
        mix_u64(static_cast<uint64_t>(request_input_row_capacity_));
        mix_u64(static_cast<uint64_t>(
            static_cast<uint32_t>(pad_token_id)));
        mix_pointer(backend);
        mix_pointer(request_token_ids_dev_);
        mix_pointer(request_position_ids_dev_);
        mix_pointer(request_sequence_lengths_dev_);
        mix_pointer(cached_tokens_device);
        mix_pointer(prefill_chunk_token_ids_dev_);
        mix_pointer(prefill_chunk_position_ids_dev_);
        mix_pointer(prefill_chunk_real_rows_dev_);
        mix_pointer(prefill_chunk_row_stride_dev_);
        if (capture_identity == 0)
            capture_identity = UINT64_MAX;

        DevicePrefillChunkGraphBinding binding{
            .backend = backend,
            .request_token_ids_device = static_cast<const int32_t *>(
                request_token_ids_dev_),
            .request_position_ids_device = static_cast<const int32_t *>(
                request_position_ids_dev_),
            .request_total_rows_device = static_cast<const int32_t *>(
                request_sequence_lengths_dev_),
            .cached_tokens_device = cached_tokens_device,
            .chunk_token_ids_device = static_cast<int32_t *>(
                prefill_chunk_token_ids_dev_),
            .chunk_position_ids_device = static_cast<int32_t *>(
                prefill_chunk_position_ids_dev_),
            .chunk_real_rows_device = prefill_chunk_real_rows_dev_,
            .chunk_row_stride_device = prefill_chunk_row_stride_dev_,
            .request_row_capacity = request_input_row_capacity_,
            .bucket_seq_len = bucket_seq_len,
            .pad_token_id = pad_token_id,
            .require_exact_rows = require_exact_rows,
            .capture_identity = capture_identity,
        };
        if (!binding.valid())
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Constructed an incomplete captured "
                "prefill chunk binding");
            return std::nullopt;
        }
        return binding;
    }

    bool DeviceGraphOrchestrator::prepareServingGraphSetupInputs(
        int row_count,
        void *producer_stream)
    {
        if (!state_.device_id.is_gpu() || !producer_stream ||
            row_count <= 0 || row_count > prefill_chunk_row_capacity_ ||
            static_cast<size_t>(row_count) >
                request_padding_token_ids_host_.size() ||
            static_cast<size_t>(row_count) >
                request_setup_zero_rows_host_.size() ||
            request_batch_geometry_host_.size() !=
                request_batch_geometry_layout_.scalarCount() ||
            !request_token_ids_dev_ || !request_position_ids_dev_ ||
            !request_batch_geometry_dev_)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Setup capture input rows exceed their permanent arena/host owners"
                << " rows=" << row_count
                << " capacity=" << prefill_chunk_row_capacity_
                << " stream=" << producer_stream);
            return false;
        }

        std::fill(
            request_batch_geometry_host_.begin(),
            request_batch_geometry_host_.end(),
            int32_t{0});
        request_batch_geometry_host_.front() = row_count;
        request_batch_geometry_host_[
            request_batch_geometry_layout_.rowStrideIndex()] =
                row_count;

        IBackend *const backend = getBackendFor(state_.device_id);
        const size_t row_bytes =
            sizeof(int32_t) * static_cast<size_t>(row_count);
        const size_t geometry_bytes =
            sizeof(int32_t) * request_batch_geometry_host_.size();
        if (!backend ||
            !backend->hostToDeviceOnStream(
                request_token_ids_dev_,
                request_padding_token_ids_host_.data(),
                row_bytes,
                state_.device_id.gpu_ordinal(),
                producer_stream) ||
            !backend->hostToDeviceOnStream(
                request_position_ids_dev_,
                request_setup_zero_rows_host_.data(),
                row_bytes,
                state_.device_id.gpu_ordinal(),
                producer_stream) ||
            !backend->hostToDeviceOnStream(
                request_batch_geometry_dev_,
                request_batch_geometry_host_.data(),
                geometry_bytes,
                state_.device_id.gpu_ordinal(),
                producer_stream))
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Failed to initialize setup-only graph capture inputs on their exact stream");
            return false;
        }

        const bool published =
            publishPreparedArenaGraphInput(
                BufferId::REQUEST_TOKEN_IDS,
                request_token_ids_dev_,
                producer_stream,
                state_.device_id,
                "serving_graph_setup_input") &&
            publishPreparedArenaGraphInput(
                BufferId::REQUEST_POSITION_IDS,
                request_position_ids_dev_,
                producer_stream,
                state_.device_id,
                "serving_graph_setup_input") &&
            publishPreparedArenaGraphInput(
                BufferId::REQUEST_BATCH_GEOMETRY,
                request_batch_geometry_dev_,
                producer_stream,
                state_.device_id,
                "serving_graph_setup_input");
        if (published)
        {
            PerfStatsCollector::addCounter(
                "forward_graph",
                "setup_capture_input_publications",
                1.0,
                "setup",
                state_.device_id.toString(),
                {{"rows", std::to_string(row_count)},
                 {"request_admitted", "false"},
                 {"producer", "serving_family_stream"}});
        }
        return published;
    }

    /**
     * @brief Execute root-owned chunks without conflating payload capacity and tokens.
     * @param tokens Immutable real request token IDs.
     * @param seq_len Number of real tokens in this submitted schedule.
     * @param policy Root-owned chunk boundaries and physical bucket capacities.
     * @param pad_token_id GPU capture padding token, never a CPU inference token.
     * @param allow_padded_execution Admit initialized physical transport slack.
     * @return True after every logical chunk and its maintenance boundary complete.
     */
    bool DeviceGraphOrchestrator::forwardPrefillChunkSchedule(
        const int *tokens,
        int seq_len,
        const PrefillChunkSchedulerPolicy &policy,
        int pad_token_id,
        bool allow_padded_execution)
    {
        ScopedDeviceLog device_log(state_.device_id);

        if (!tokens || seq_len <= 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid prefill chunk schedule input");
            return false;
        }
        if (!supportsPrefillChunkSchedule(seq_len))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Prefill chunk schedule is not supported for current runner state");
            return false;
        }
        if (!hasGlobalWeights())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] prefill chunk schedule called without global weights set");
            return false;
        }
        const bool consumes_pipeline_hidden =
            pp_stage_config_.has_value() &&
            !pp_stage_config_->has_embedding;
        TensorBase *const pipeline_hidden_input =
            consumes_pipeline_hidden
                ? (pipeline_forward_edges_ ? state_.hidden.get() : external_hidden_state_input_)
                : nullptr;
        if (consumes_pipeline_hidden && !pipeline_hidden_input)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Non-embedding PP prefill chunk requires its previous-stage activation owner");
            return false;
        }
        if (seq_len > state_.max_seq_len)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Prefill chunk schedule length " << seq_len
                                                                                 << " exceeds buffer capacity "
                                                                                 << state_.max_seq_len);
            return false;
        }
        const int activation_seq_len =
            state_.activation_seq_len > 0 ? state_.activation_seq_len : state_.max_seq_len;
        const auto normalized_buckets = normalizePrefillGraphBuckets(policy.bucket_sizes);
        if (!normalized_buckets.empty() && normalized_buckets.back() > activation_seq_len)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Prefill chunk bucket " << normalized_buckets.back()
                                                                        << " exceeds activation graph buffer capacity "
                                                                        << activation_seq_len);
            return false;
        }

        const auto placement = graph_builder_ ? graph_builder_->config().ngram_table_placement : nullptr;
        const bool exact = placement && placement->fixedHeadPipeline();
        if (exact && (policy.width_policy != PrefillChunkWidthPolicy::ExactRetained ||
            !serving_graph_family_plan_ || policy.bucket_sizes != serving_graph_family_plan_->prefill_bucket_rows))
            throw std::invalid_argument("Fixed PLE pipeline requires its original retained exact prefill widths");
        transitionToPhase(InferencePhase::PREFILL);

        const auto planned_chunks = planPrefillChunkSchedule(policy);
        if (!planned_chunks || planned_chunks.chunks.empty())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid prefill chunk policy: "
                      << planned_chunks.error);
            return false;
        }
        if (consumes_pipeline_hidden && !pipeline_forward_edges_ &&
            planned_chunks.chunks.size() != 1u)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] A PP child accepts exactly one pipeline-wide chunk transaction at a time");
            return false;
        }

        if (state_.device_id.is_cpu())
        {
            if (policy.real_token_start != state_.positions[0])
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] CPU prefill schedule starts "
                    "outside the canonical request cursor"
                    << " schedule_start=" << policy.real_token_start
                    << " position=" << state_.positions[0]);
                return false;
            }

            std::uint64_t executed_rows = 0u;
            for (const PrefillChunkPlan &chunk : planned_chunks.chunks)
            {
                const int relative_offset =
                    chunk.token_offset - policy.real_token_start;
                if (chunk.real_count <= 0 || relative_offset < 0 ||
                    relative_offset > seq_len ||
                    chunk.real_count > seq_len - relative_offset ||
                    chunk.bucket_seq_len < chunk.real_count ||
                    chunk.bucket_seq_len > activation_seq_len)
                {
                    cancelPrefillChunkSnapshotDiagnostics();
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] CPU prefill chunk exceeds "
                        "its admitted logical or transaction capacity"
                        << " chunk=" << chunk.chunk_index
                        << " real_rows=" << chunk.real_count
                        << " transaction_rows=" << chunk.bucket_seq_len
                        << " capacity=" << activation_seq_len);
                    return false;
                }

                bool snapshot_scope_active = false;
                std::optional<const float *> chunk_completion;
                const int *execution_tokens = tokens + relative_offset;
                if (pp_stage_config_.has_value() &&
                    chunk.bucket_seq_len > chunk.real_count &&
                    !allow_padded_execution)
                {
                    cancelPrefillChunkSnapshotDiagnostics();
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] CPU PP prefill requires "
                        "explicit admission of padded transport capacity");
                    return false;
                }
                try
                {
                    beginPrefillChunkSnapshotDiagnostics(chunk);
                    snapshot_scope_active = true;
                    // CPU graphs have no fixed capture width. Execute the same
                    // exact logical rows as ordinary CPU inference; computing
                    // padding would append phantom KV rows and select a fake
                    // terminal token. Physical capacity belongs only to the
                    // cross-participant transaction, carried separately below.
                    chunk_completion = forwardImpl(
                        execution_tokens,
                        /*token_ids_device=*/nullptr,
                        chunk.real_count,
                        /*batch_size=*/1,
                        ForwardExecutionRole::MainInference,
                        ForwardInvocationKind::ExplicitPrefill,
                        /*position_ids_device_override=*/nullptr,
                        /*sequence_lengths_device_override=*/nullptr,
                        /*request_real_lengths=*/{},
                        RemotePrefillTransactionGeometry{
                            .physical_rows_per_request =
                                chunk.bucket_seq_len,
                        });
                    endPrefillChunkSnapshotDiagnostics(chunk);
                    snapshot_scope_active = false;
                }
                catch (...)
                {
                    if (snapshot_scope_active)
                        endPrefillChunkSnapshotDiagnostics(chunk);
                    cancelPrefillChunkSnapshotDiagnostics();
                    throw;
                }
                if (!chunk_completion)
                {
                    cancelPrefillChunkSnapshotDiagnostics();
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] CPU exact-shape prefill "
                        "chunk " << chunk.chunk_index << " failed");
                    return false;
                }

                if (pp_stage_config_.has_value() &&
                    !pp_stage_config_->has_lm_head &&
                    chunk.bucket_seq_len > chunk.real_count)
                {
                    // A following GPU consumes a complete physical bucket.
                    // Initialize only the outgoing slack, after real CPU work,
                    // without executing padding through attention, FFN or KV.
                    // CPU owns this host tensor; no GPU event or copy is added.
                    auto *hidden = dynamic_cast<FP32Tensor *>(getHiddenState());
                    const size_t width = static_cast<size_t>(state_.d_model);
                    const size_t live_elements =
                        static_cast<size_t>(chunk.real_count) * width;
                    const size_t physical_elements =
                        static_cast<size_t>(chunk.bucket_seq_len) * width;
                    if (!hidden || physical_elements > hidden->numel())
                    {
                        cancelPrefillChunkSnapshotDiagnostics();
                        LOG_ERROR(
                            "[DeviceGraphOrchestrator] CPU pipeline payload "
                            "requires its admitted FP32 activation capacity");
                        return false;
                    }
                    std::fill_n(hidden->mutable_data() + live_elements,
                                physical_elements - live_elements, 0.0f);
                }

                const PrefillChunkMaintenanceState maintenance_state =
                    prefillChunkMaintenanceState(chunk);
                const PrefillChunkMaintenanceDecision maintenance_decision =
                    evaluatePrefillChunkMaintenance(
                        chunk, maintenance_state);
                const bool maintenance_requested =
                    maintenance_state.rebalance_requested ||
                    chunk.rebalance_required_after;
                if ((maintenance_requested && !maintenance_decision) ||
                    (maintenance_decision.can_run &&
                     !onPrefillChunkMaintenance(
                         chunk, maintenance_decision)))
                {
                    cancelPrefillChunkSnapshotDiagnostics();
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] CPU prefill chunk "
                        "maintenance failed after chunk "
                        << chunk.chunk_index << ": "
                        << maintenance_decision.reason);
                    return false;
                }
                executed_rows +=
                    static_cast<std::uint64_t>(chunk.real_count);
            }

            if (executed_rows != static_cast<std::uint64_t>(seq_len) ||
                !finalizePrefillChunkSnapshotDiagnostics(planned_chunks))
            {
                cancelPrefillChunkSnapshotDiagnostics();
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] CPU prefill schedule did not "
                    "retire its complete logical row sequence");
                return false;
            }
            PerfStatsCollector::addCounter(
                "forward_graph",
                "host_eager_prefill_chunk_transactions",
                static_cast<double>(planned_chunks.chunks.size()),
                "prefill",
                state_.device_id.toString(),
                {{"logical_rows", std::to_string(seq_len)},
                 {"local_padding_rows", "0"},
                 {"transaction_capacity_rows",
                  std::to_string(
                      planned_chunks.chunks.front().bucket_seq_len)},
                 {"position_authority", "forward_impl"}});
            return true;
        }

        const int captured_bucket_seq_len =
            planned_chunks.chunks.front().bucket_seq_len;
        std::uint64_t physical_rows = 0;
        std::vector<DevicePrefillChunkGraphBinding> chunk_bindings;
        chunk_bindings.reserve(planned_chunks.chunks.size());
        for (const auto &chunk : planned_chunks.chunks)
        {
            // Setup and admission use this same binding authority. A smaller
            // retained tail changes only immutable graph width/identity, never
            // the token bank, producer stream or canonical device KV cursor.
            const auto binding = makeDevicePrefillChunkGraphBinding(
                chunk.bucket_seq_len, pad_token_id);
            if (!binding)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Device-resident prefill bucket exceeds its admitted family capacity"
                          << " bucket=" << chunk.bucket_seq_len
                          << " capacity=" << prefill_chunk_row_capacity_);
                return false;
            }
            chunk_bindings.push_back(*binding);
            physical_rows += static_cast<std::uint64_t>(binding->bucket_seq_len);
        }

        // Validate the original ingress extent before signed position-row fill.
        // The caller's schedule must name this same accepted logical range;
        // no live GPU KV count is observed to select a retained program.
        const auto request_range = SparseQueryPrefillProof::admittedRange(state_.positions[0], seq_len);
        if (policy.real_token_start != state_.positions[0] || policy.real_token_count != seq_len) {
            LOG_ERROR("[DeviceGraphOrchestrator] GPU prefill schedule disagrees with its original request ingress");
            return false;
        }
        std::vector<int> request_position_ids(static_cast<size_t>(seq_len));
        for (int row = 0; row < seq_len; ++row)
            request_position_ids[static_cast<size_t>(row)] =
                state_.positions[0] + row;
        const std::array<int, 1> request_real_lengths{seq_len};
        if (!admitRequestInputsOnDevice(
                tokens,
                request_position_ids.data(),
                request_real_lengths,
                seq_len,
                /*request_count=*/1,
                seq_len))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Chunked prefill could not admit its complete request input");
            return false;
        }

        TensorBase *logits_output = state_.logits.get();
        TensorBase *logits_local_output = state_.logits_local.get();
        ModelBuffers model_buffers = state_.toModelBuffers();
        model_buffers.logits = logits_output;
        model_buffers.logits_local = logits_local_output;
        setBuffers(model_buffers);

        ForwardInput input;
        input.token_ids = nullptr;
        input.token_ids_device = prefill_chunk_token_ids_dev_;
        input.position_ids = nullptr;
        input.position_ids_device = prefill_chunk_position_ids_dev_;
        input.position_policy = ForwardPositionPolicy::ExplicitRows;
        input.execution_role = ForwardExecutionRole::MainInference;
        input.execution_phase = ForwardExecutionPhase::Prefill;
        if (sparseQueryProgramCount() > 1u) {
            input.sparse_query_prefill_proof = request_range;
            input.sparse_query_execution_program = input.sparse_query_prefill_proof->program(
                qwen4exp::QwenSparseFirstRankingLogicalRow);
        }
        input.batch_size = 1;
        input.seq_len = seq_len;
        input.real_seq_len = seq_len;
        input.position_offset = state_.positions[0];
        input.token_offset = state_.positions[0];
        const auto sparse_request_identity =
            currentMoESparseRequestIdentity();
        input.moe_overlay_collective_generation_id =
            sparse_request_identity
                ? sparse_request_identity->generation()
                : 0;
        input.moe_overlay_collective_step_id =
            static_cast<uint64_t>(std::max(input.token_offset, 0));
        input.device_state_publication_stream =
            explicitGPUStreamForOperation(
                "device_prefill_chunk_graph_state_publication");
        input.device = state_.device_id;
        input.kv_cache = state_.kv_cache.get();
        input.sequence_lengths = nullptr;
        input.sequence_lengths_device = prefill_chunk_real_rows_dev_;
        input.rehydrate_prefix_runtime_on_device =
            graph_builder_ &&
            graph_builder_->prefixCacheRuntimeStateRequiresDeviceRehydration();
        input.external_hidden_state = pipeline_hidden_input;
        if (consumes_pipeline_hidden)
        {
            if (!pipeline_forward_edges_)
                input.host_pipeline_activation_rows = HostPipelineActivationRows::admitted(
                    pipeline_hidden_input, input.device, input.batch_size, seq_len);
            // The graph input now owns this stable pointer for the complete
            // one-chunk transaction; preserve ordinary setHiddenState()
            // single-use semantics for the next pipeline boundary.
            external_hidden_state_input_ = nullptr;
        }

        if (!input.device_state_publication_stream)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Device-resident prefill chunk "
                "graph lacks an explicit publication stream");
            return false;
        }
        input.device_prefill_chunk = chunk_bindings.front();
        if (!input.device_prefill_chunk)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Device-resident prefill could not "
                "construct its captured chunk binding");
            return false;
        }
        if (!bindShiftedMTPPrefillTransaction(input, /*request_count=*/1))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Chunked prefill could not bind its graph-integrated shifted MTP transaction");
            return false;
        }

        ensureForwardEngine();
        auto schedule = ForwardExecutionEngine::preparePrefillChunkRuntimeSchedule(
            input,
            policy,
            pad_token_id,
            allow_padded_execution);
        if (!schedule)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to prepare prefill chunk schedule: "
                      << schedule.error);
            return false;
        }

        ForwardOutput output;
        output.logits = logits_output;
        output.hidden = state_.hidden.get();

        std::vector<ForwardInput> chunk_inputs;
        chunk_inputs.reserve(chunk_bindings.size());
        for (const auto &binding : chunk_bindings)
        {
            ForwardInput chunk_input = input;
            chunk_input.device_prefill_chunk = binding;
            // Shifted-MTP identity embeds the materializer's physical width.
            // Seal it through its existing authority too; copying the largest
            // bucket's predictor identity would miss the retained smaller graph.
            if (!bindShiftedMTPPrefillTransaction(chunk_input, /*request_count=*/1))
                return false;
            chunk_inputs.push_back(std::move(chunk_input));
        }
        if (!forward_engine_->runPrefillChunkSchedule(input, schedule, output, *this,
                                                     chunk_inputs))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Prefill chunk schedule execution failed");
            return false;
        }
        // One completed request witness covers its actual retained chunk
        // composition (for example 8+1), never warmup or cold materialization.
        // Original request admission chooses the program, not observed outputs.
        if (sparseQueryProgramCount() > 1u && input.sparse_query_prefill_proof &&
            input.sparse_query_prefill_proof->namesRange(input.position_offset, seq_len))
            PerfStatsCollector::addCounter("qsa", "query_program_forward_submissions", 1.0,
                "prefill", state_.device_id.toString(),
                {{"program", input.sparse_query_prefill_proof->program(qwen4exp::QwenSparseFirstRankingLogicalRow)==
                    SparseQueryExecutionProgram::ProvenNoRanking?"proven_no_ranking":"ranking_possible"},
                 {"role", std::to_string(static_cast<int>(input.execution_role))},
                 {"physical_rows", std::to_string(physical_rows)},
                 {"logical_rows", std::to_string(seq_len)},
                 {"logical_begin", std::to_string(input.position_offset)},
                 {"width_policy", policy.width_policy==PrefillChunkWidthPolicy::ExactRetained?"exact_retained":"retained_padding"},
                 {"observation", "successful_forward_submission_not_native_replay_count"}});
        if (input.rehydrate_prefix_runtime_on_device)
        {
            graph_builder_->completePrefixCacheRuntimeStateDeviceRehydration();
        }

        state_.positions[0] += seq_len;
        state_.sequence_lengths[0] += seq_len;

        /*
         * A PP follower participates in the main captured prefill graph but
         * does not own the learned predictor, shifted KV family, or terminal
         * hidden archive.  Separately, an MTP-off request may retain the same
         * maximum-capacity graph family as a later MTP request.  Physical
         * sidecar ownership therefore cannot decide whether this *runtime*
         * prefill must carry a shifted transaction: only the typed runtime
         * role can.  Using the capacity role here previously made a retained
         * MTP-off graph demand a binding which bindShiftedMTPPrefillTransaction
         * correctly omitted.
         */
        const MTPRuntimeTransactionRole mtp_runtime_role =
            resolveMTPRuntimeTransactionRole(
                graph_builder_ && graph_builder_->config().mtp.enabled,
                !pp_stage_config_ || pp_stage_config_->has_lm_head);
        const bool executes_shifted_mtp_transaction =
            mtp_runtime_role == MTPRuntimeTransactionRole::PredictorOwner;
        if (executes_shifted_mtp_transaction &&
            !input.shifted_mtp_prefill.has_value())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] GPU chunked prefill completed without its graph-integrated shifted-MTP transaction");
            return false;
        }
        if (executes_shifted_mtp_transaction)
            state_.mtp_terminal_hidden_publication.publishMainForward();
        else
            state_.mtp_terminal_hidden_publication.invalidate();
        PerfStatsCollector::addCounter(
            "graph",
            "device_resident_prefill_chunk_transactions",
            static_cast<double>(schedule.chunks.size()),
            "prefill",
            state_.device_id.toString(),
            {{"logical_rows", std::to_string(seq_len)},
             {"bucket_rows", std::to_string(captured_bucket_seq_len)},
             {"terminal_bucket_rows", std::to_string(chunk_bindings.back().bucket_seq_len)},
             {"physical_rows", std::to_string(physical_rows)},
             {"request_admissions", "1"},
             {"source_cursor", "canonical_kv_count"},
             {"host_chunk_slices", "0"}});
        if (executes_shifted_mtp_transaction)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "shifted_prefill_integrated_graph_transactions",
                static_cast<double>(schedule.chunks.size()),
                "prefill",
                state_.device_id.toString(),
                {{"requests", "1"},
                 {"bucket_rows",
                  std::to_string(captured_bucket_seq_len)},
                 {"chunked", "true"}});
        }

        publishRequestInputReuseReady(
            output.execution.stream,
            "chunked_prefill_transaction");

        const int terminal_seq_len = schedule.chunks.empty()
                                         ? seq_len
                                         : schedule.chunks.back().chunk.real_count;
        noteMainForwardHiddenProducedForMTP(
            terminal_seq_len,
            1,
            {},
            executes_shifted_mtp_transaction &&
                    state_.mtp_terminal_hidden_publication.current()
                ? MTPTerminalHiddenPublication::Source::MainForward
                : MTPTerminalHiddenPublication::Source::Unavailable);

        // The main captured graph has now consumed both ordinary and MTP
        // prefill weights. Host payload release therefore follows the exact
        // final graph producer instead of an out-of-graph sidecar.
        if (release_host_resident_after_forward_ && !host_resident_released_ && weight_manager_)
        {
            host_resident_released_ = true;
            scheduleMmapReclaimAfterFirstPrefill();
        }

        LOG_TRACE("[FORWARD_TRACE] chunk_schedule seq_len=" << seq_len
                                                            << " pos_offset=" << input.position_offset
                                                            << " chunks=" << schedule.chunks.size()
                                                            << " positions_after=" << state_.positions[0]);
        return true;
    }

    bool DeviceGraphOrchestrator::ensureMTPTerminalHiddenBuffer(int min_rows)
    {
        if (!graph_builder_ ||
            !retainsMTPGraphCapacity(graph_builder_->config().mtp))
            return true;
        if (min_rows <= 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid MTP terminal hidden row capacity request: "
                      << min_rows);
            return false;
        }
        const int required_rows = std::max(1, min_rows);

        /*
         * CPU executable graphs and GPU captured graphs both retain the raw
         * mailbox address in their stages.  The owner is therefore fixed at
         * arena initialization on every backend; runtime resize/rebind would
         * make an already-cached graph observe freed storage.
         */
        const auto arena_owner =
            arena_ && arena_->isRegistered(
                          BufferId::PREFIX_TERMINAL_HIDDEN)
                ? arena_->getSharedTensor(
                      BufferId::PREFIX_TERMINAL_HIDDEN)
                : std::shared_ptr<TensorBase>{};
        const auto archive_device =
            state_.prefix_terminal_hidden
                ? state_.prefix_terminal_hidden->current_device()
                : std::optional<DeviceId>{};
        const bool storage_ready =
            state_.prefix_terminal_hidden &&
            (state_.device_id.is_gpu()
                 ? state_.prefix_terminal_hidden->gpu_data_ptr() != nullptr &&
                       archive_device.has_value() &&
                       *archive_device == state_.device_id
                 : state_.prefix_terminal_hidden->raw_data() != nullptr);
        if (!arena_owner ||
            state_.prefix_terminal_hidden != arena_owner ||
            state_.prefix_terminal_hidden->rows() <
                static_cast<size_t>(required_rows) ||
            state_.prefix_terminal_hidden->cols() !=
                static_cast<size_t>(graph_builder_->config().residualWidth()) ||
            state_.prefix_terminal_hidden->isMapped() ||
            !storage_ready)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP terminal-hidden archive was not installed as a capacity-complete immutable arena owner before graph construction"
                      << " required_rows=" << required_rows
                      << " d_model=" << graph_builder_->config().residualWidth()
                      << " device=" << state_.device_id.toString());
            return false;
        }
        return true;
    }

    bool DeviceGraphOrchestrator::executeMTPHiddenRowSelect(
        TensorBase *input,
        BufferId input_buffer_id,
        TensorBase *output,
        BufferId output_buffer_id,
            MTPTerminalHiddenRowSelectGraphCache &cache,
            const char *node_name,
            int row_idx,
        int seq_len,
        void *stream)
    {
        if (state_.device_id.is_gpu())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] GPU terminal-hidden publication forbids the host-authored scalar selector; use the fixed-range rows graph");
            return false;
        }
        if (row_idx < 0 || row_idx >= seq_len)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid MTP hidden row selection: row="
                      << row_idx << " seq_len=" << seq_len);
            return false;
        }
        if (!input || !output)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot select MTP hidden row without input/output buffer");
            return false;
        }

        if (graph_builder_->config().residualWidth() <= 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot select MTP hidden row with invalid d_model="
                      << graph_builder_->config().residualWidth());
            return false;
        }

        const ResidualRowGeometry residual_rows(graph_builder_->config().residualWidth());
        residual_rows.validate(input->shape());
        residual_rows.validate(output->shape());
        const size_t hidden_rows = input->rows();
        if (hidden_rows > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Hidden-state row capacity exceeds int range: rows="
                      << hidden_rows);
            return false;
        }
        const int seq_capacity = static_cast<int>(hidden_rows);
        if (seq_capacity < seq_len)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Hidden-state row-select capacity too small: capacity="
                      << seq_capacity << " seq_len=" << seq_len);
            return false;
        }

        const uint64_t current_workspace_generation =
            state_.device_id.is_gpu() ? workspaceGeneration(state_.device_id) : 0;
        const bool workspace_generation_changed =
            state_.device_id.is_gpu() &&
            cache.workspace_generation != 0 &&
            current_workspace_generation != cache.workspace_generation;

        const bool rebuild =
            !cache.valid ||
            !cache.graph ||
            !cache.stage ||
            workspace_generation_changed ||
            cache.input != input ||
            cache.output != output ||
            cache.device != state_.device_id ||
            cache.seq_capacity != seq_capacity ||
            cache.d_model != graph_builder_->config().residualWidth();

        if (rebuild)
        {
            HiddenStateRowSelectStage::Params params;
            params.device_id = state_.device_id;
            params.input = input;
            params.output = output;
            params.seq_len = seq_capacity;
            params.d_model = graph_builder_->config().residualWidth();
            params.selected_row_idx = row_idx;
            params.input_buffer_id = input_buffer_id;
            params.output_buffer_id = output_buffer_id;
            params.workspace_buffer_name =
                std::string(HiddenStateRowSelectStage::WS_SELECTED_ROW_SCALAR) +
                "_" + (node_name && node_name[0] != '\0' ? node_name : "mtp_hidden_row_select");

            auto stage = ComputeStageFactory::createHiddenStateRowSelect(params);
            auto *row_select_stage = dynamic_cast<HiddenStateRowSelectStage *>(stage.get());
            if (!row_select_stage)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] HiddenStateRowSelect factory returned an incompatible stage");
                return false;
            }

            auto graph = std::make_unique<ComputeGraph>();
            graph->addNode(node_name && node_name[0] != '\0' ? node_name : "mtp_hidden_row_select",
                           std::move(stage),
                           state_.device_id);

            cache.graph = std::move(graph);
            cache.stage = row_select_stage;
            cache.input = input;
            cache.output = output;
            cache.device = state_.device_id;
            cache.workspace_generation = 0;
            cache.seq_capacity = seq_capacity;
            cache.d_model = graph_builder_->config().residualWidth();
            cache.valid = true;
        }

        cache.stage->setSelectedRowForReplay(row_idx);
        if (state_.device_id.is_gpu())
        {
            if (!stream)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] MTP hidden row-select requires the exact GPU producer stream");
                return false;
            }
            cache.stage->setGPUStream(stream);
        }

        IDeviceContext *ctx = getDeviceContext(state_.device_id);
        if (!ctx)
            return false;

        if (!execute(*cache.graph, ctx))
            return false;

        if (state_.device_id.is_gpu())
            cache.workspace_generation = workspaceGeneration(state_.device_id);
        return true;
    }

    bool DeviceGraphOrchestrator::executeMTPTerminalHiddenRowSelect(int row_idx, int seq_len, void *stream)
    {
        if (!state_.hidden)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot select MTP hidden row without hidden-state buffer");
            return false;
        }
        if (!ensureMTPTerminalHiddenBuffer())
            return false;
        return executeMTPHiddenRowSelect(
            state_.hidden.get(),
            BufferId::HIDDEN_STATE,
            state_.prefix_terminal_hidden.get(),
            BufferId::PREFIX_TERMINAL_HIDDEN,
            mtp_terminal_hidden_row_select_cache_,
            "mtp_terminal_hidden_row_select",
            row_idx,
            seq_len,
            stream);
    }

    bool DeviceGraphOrchestrator::materializeMTPTerminalHiddenRowsSelectGraph(
        MTPTerminalHiddenRowsSelectGraphCache &cache,
        const char *node_name,
        HiddenStateRowsSelectStage::DeviceRowIndexSource row_index_source,
        int selected_row_count,
        int fixed_contiguous_row_start,
        const char *row_buffer_name,
        const int32_t *external_device_row_indices,
        int request_index)
    {
        if (!mtp_graph_owner_plan_.ownsControllerExecutables())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Pipeline follower cannot materialize terminal-hidden helper executables");
            return false;
        }
        if (!state_.device_id.is_gpu() || !state_.hidden ||
            !state_.prefix_terminal_hidden || selected_row_count <= 0 ||
            graph_builder_->config().residualWidth() <= 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot materialize a GPU terminal-hidden rows-select graph from incomplete model bindings");
            return false;
        }
        if (row_index_source !=
                HiddenStateRowsSelectStage::DeviceRowIndexSource::
                    FixedContiguousRange &&
            row_index_source !=
                HiddenStateRowsSelectStage::DeviceRowIndexSource::
                    ExternalDeviceIndices &&
            row_index_source !=
                HiddenStateRowsSelectStage::DeviceRowIndexSource::
                    RequestTerminalLengths &&
            row_index_source !=
                HiddenStateRowsSelectStage::DeviceRowIndexSource::
                    ShiftedPrefillKVProgress)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] GPU terminal-hidden publication requires fixed-contiguous, external-device, request-terminal, or shifted-prefill-progress row ownership");
            return false;
        }

        const ResidualRowGeometry residual_rows(graph_builder_->config().residualWidth());
        residual_rows.validate(state_.hidden->shape());
        residual_rows.validate(state_.prefix_terminal_hidden->shape());
        const size_t hidden_rows = state_.hidden->rows();
        if (hidden_rows == 0 ||
            hidden_rows > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid hidden-state capacity while materializing terminal-hidden publication: rows="
                      << hidden_rows);
            return false;
        }
        const bool consumes_request_geometry =
            row_index_source ==
                HiddenStateRowsSelectStage::DeviceRowIndexSource::
                    RequestTerminalLengths ||
            row_index_source ==
                HiddenStateRowsSelectStage::DeviceRowIndexSource::
                    ShiftedPrefillKVProgress;
        const auto *request_lengths_device =
            consumes_request_geometry
                ? static_cast<const int32_t *>(request_sequence_lengths_dev_)
                : nullptr;
        const auto *request_row_stride_device =
            consumes_request_geometry
                ? request_row_stride_dev_
                : nullptr;
        const int32_t *main_cached_tokens_device = nullptr;
        const int32_t *shifted_cached_tokens_device = nullptr;
        if (row_index_source ==
            HiddenStateRowsSelectStage::DeviceRowIndexSource::
                ShiftedPrefillKVProgress)
        {
            static_assert(
                sizeof(int) == sizeof(int32_t),
                "Device KV progress publication requires 32-bit cache counts");
            if (graph_builder_->config()
                    .mtp_shifted_prefill_hidden_publication !=
                MTPShiftedPrefillHiddenPublicationPolicy::
                    GraphIntegratedKVTransaction)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] GPU shifted-prefill publication requires an explicit graph-declared device-KV-progress policy");
                return false;
            }
            if (request_index < 0 ||
                request_index >= stochastic_batch_output_request_capacity_ ||
                !state_.kv_cache || state_.mtp_kv_caches.empty() ||
                !state_.mtp_kv_caches[0])
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Shifted-prefill publication cannot resolve its request-local KV owners: request="
                          << request_index
                          << " capacity="
                          << stochastic_batch_output_request_capacity_);
                return false;
            }
            main_cached_tokens_device =
                state_.kv_cache->deviceSequenceCachedTokenCountPtr(
                    request_index);
            shifted_cached_tokens_device =
                state_.mtp_kv_caches[0]
                    ->deviceSequenceCachedTokenCountPtr(request_index);
            if (!main_cached_tokens_device ||
                !shifted_cached_tokens_device || !request_lengths_device ||
                !request_row_stride_device)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Shifted-prefill publication requires canonical device KV counters and resident request geometry: request="
                          << request_index
                          << " main_count=" << main_cached_tokens_device
                          << " shifted_count=" << shifted_cached_tokens_device
                          << " lengths=" << request_lengths_device
                          << " stride=" << request_row_stride_device);
                return false;
            }
        }
        const int seq_capacity = static_cast<int>(hidden_rows);
        if (row_index_source ==
            HiddenStateRowsSelectStage::DeviceRowIndexSource::
                RequestTerminalLengths)
        {
            if (graph_builder_->config()
                    .mtp_request_terminal_hidden_publication !=
                MTPRequestTerminalHiddenPublicationPolicy::
                    GraphCapturedDeviceGeometry)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] GPU request-terminal publication requires an explicit graph-declared device-geometry policy");
                return false;
            }
            if (!request_lengths_device || !request_row_stride_device ||
                request_sequence_lengths_capacity_ < selected_row_count ||
                selected_row_count > seq_capacity)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Request-terminal publication requires complete resident geometry within the hidden-state activation capacity: requests="
                          << selected_row_count
                          << " length_capacity="
                          << request_sequence_lengths_capacity_
                          << " activation_capacity=" << seq_capacity);
                return false;
            }
        }
        if (row_index_source ==
                HiddenStateRowsSelectStage::DeviceRowIndexSource::
                    FixedContiguousRange &&
            (fixed_contiguous_row_start < 0 ||
             fixed_contiguous_row_start >
                 seq_capacity - selected_row_count))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Fixed terminal-hidden publication range exceeds hidden-state capacity: start="
                      << fixed_contiguous_row_start
                      << " rows=" << selected_row_count
                      << " capacity=" << seq_capacity);
            return false;
        }
        if (row_index_source ==
                HiddenStateRowsSelectStage::DeviceRowIndexSource::
                    ExternalDeviceIndices &&
            ((!row_buffer_name || row_buffer_name[0] == '\0') ||
             !external_device_row_indices))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] External terminal-hidden publication requires an exact producer-owned device row-index pointer");
            return false;
        }
        if (state_.prefix_terminal_hidden->rows() <
            static_cast<size_t>(selected_row_count))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Terminal-hidden mailbox is smaller than publication graph capacity: mailbox_rows="
                      << state_.prefix_terminal_hidden->rows()
                      << " graph_rows=" << selected_row_count);
            return false;
        }

        const uint64_t generation = workspaceGeneration(state_.device_id);
        if (generation == 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Terminal-hidden publication graphs require a finalized GPU workspace generation");
            return false;
        }
        const std::string stable_row_buffer =
            row_buffer_name ? std::string(row_buffer_name) : std::string{};
        // Only accepted-state selection owns a raw verifier-logit row. Other
        // hidden archival helpers retain their original one-output contract.
        const bool accepted_publication = row_index_source ==
            HiddenStateRowsSelectStage::DeviceRowIndexSource::ExternalDeviceIndices;
        TensorBase *logits_input = nullptr;
        TensorBase *logits_output = nullptr;
        int logits_columns = 0;
        bool local_logits = false;
        BufferId logits_source_id = BufferId::MTP_LOGITS;
        if (accepted_publication)
        {
            local_logits = allPositionVerifierGraphWritesLocalLogits(1) &&
                !graph_builder_->config().hasNativeLocalVocabularyGather();
            logits_source_id = local_logits || !graph_builder_->config().mtpParticipantLogitsAreVocabularySharded()
                ? BufferId::MTP_LOGITS : BufferId::MTP_LOGITS_GATHERED;
            const auto source = state_.extension_buffers.find(logits_source_id);
            logits_input = source == state_.extension_buffers.end() ? nullptr : source->second.get();
            logits_output = local_logits ? state_.logits_local.get() : state_.logits.get();
            logits_columns = local_logits ? graph_builder_->config().vocab_local
                : graph_builder_->config().vocab_size;
            if (!logits_input || !logits_output || logits_columns <= 0 ||
                logits_input->native_type() != TensorType::FP32 || logits_output->native_type() != TensorType::FP32 ||
                logits_input->cols() < logits_columns || logits_output->cols() < logits_columns ||
                logits_input->cols() > std::numeric_limits<int>::max() ||
                logits_output->cols() > std::numeric_limits<int>::max() ||
                logits_input->rows() <= 0 ||
                logits_input->rows() > std::numeric_limits<int>::max() ||
                logits_output->rows() < 1 ||
                !logits_input->gpu_data_ptr() || !logits_output->gpu_data_ptr())
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Accepted terminal logits require original admitted FP32 vocabulary roots"
                    << " source=" << bufferIdName(logits_source_id)
                    << " source_present=" << (logits_input != nullptr)
                    << " source_rows=" << (logits_input ? logits_input->rows() : 0)
                    << " source_cols=" << (logits_input ? logits_input->cols() : 0)
                    << " source_native_storage=" << (logits_input && logits_input->gpu_data_ptr())
                    << " destination=" << (local_logits ? "LOGITS_LOCAL" : "LOGITS")
                    << " destination_present=" << (logits_output != nullptr)
                    << " destination_rows=" << (logits_output ? logits_output->rows() : 0)
                    << " destination_cols=" << (logits_output ? logits_output->cols() : 0)
                    << " destination_native_storage=" << (logits_output && logits_output->gpu_data_ptr())
                    << " live_columns=" << logits_columns);
                return false;
            }
        }
        const bool bindings_match =
            cache.valid && cache.graph && cache.stage &&
            cache.logits_input == logits_input && cache.logits_output == logits_output &&
            cache.logits_columns == logits_columns &&
            cache.logits_source_rows == (logits_input ? logits_input->rows() : 0) &&
            cache.logits_input_device == (logits_input ? logits_input->gpu_data_ptr() : nullptr) &&
            cache.logits_output_device == (logits_output ? logits_output->gpu_data_ptr() : nullptr) &&
            cache.logits_source_stride == (logits_input ? logits_input->cols() : 0) &&
            cache.logits_destination_stride == (logits_output ? logits_output->cols() : 0) &&
            cache.input == state_.hidden.get() &&
            cache.output == state_.prefix_terminal_hidden.get() &&
            cache.device == state_.device_id &&
            cache.workspace_generation == generation &&
            cache.seq_capacity == seq_capacity &&
            cache.d_model == graph_builder_->config().residualWidth() &&
            cache.selected_row_count == selected_row_count &&
            cache.fixed_contiguous_row_start == fixed_contiguous_row_start &&
            cache.request_row_stride == 0 &&
            cache.request_sequence_lengths_device == request_lengths_device &&
            cache.request_row_stride_source ==
                (consumes_request_geometry
                     ? HiddenStateRowsSelectStage::RequestRowStrideSource::
                           ExternalDeviceScalar
                     : HiddenStateRowsSelectStage::RequestRowStrideSource::
                           StaticGraphGeometry) &&
            cache.request_row_stride_device == request_row_stride_device &&
            cache.main_cached_tokens_device == main_cached_tokens_device &&
            cache.shifted_cached_tokens_device ==
                shifted_cached_tokens_device &&
            cache.request_index == request_index &&
            cache.shifted_origin == graph_builder_->config().mtpShiftedCacheOrigin() &&
            cache.row_index_source == row_index_source &&
            cache.row_buffer_name == stable_row_buffer &&
            cache.external_device_row_indices ==
                external_device_row_indices;
        if (bindings_match)
            return true;

        HiddenStateRowsSelectStage::Params params;
        params.device_id = state_.device_id;
        params.shifted_cache_origin=graph_builder_->config().mtpShiftedCacheOrigin();
        params.input = state_.hidden.get();
        params.output = state_.prefix_terminal_hidden.get();
        params.seq_len = seq_capacity;
        params.d_model = graph_builder_->config().residualWidth();
        params.selected_row_count = selected_row_count;
        params.selected_row_indices.resize(
            static_cast<size_t>(selected_row_count));
        if (row_index_source ==
            HiddenStateRowsSelectStage::DeviceRowIndexSource::
                FixedContiguousRange)
        {
            std::iota(
                params.selected_row_indices.begin(),
                params.selected_row_indices.end(),
                fixed_contiguous_row_start);
        }
        else
        {
            std::fill(
                params.selected_row_indices.begin(),
                params.selected_row_indices.end(),
                0);
        }
        if (accepted_publication)
            params.row_copy_policy.negative_rows = NegativeRowSelectionPolicy::PreserveDestination;
        params.input_buffer_id = BufferId::HIDDEN_STATE;
        params.output_buffer_id = BufferId::PREFIX_TERMINAL_HIDDEN;
        params.device_row_index_source = row_index_source;
        params.fixed_contiguous_row_start = fixed_contiguous_row_start;
        if (row_index_source ==
            HiddenStateRowsSelectStage::DeviceRowIndexSource::
                ExternalDeviceIndices)
        {
            params.external_device_row_indices =
                external_device_row_indices;
        }
        if (row_index_source ==
            HiddenStateRowsSelectStage::DeviceRowIndexSource::
                RequestTerminalLengths)
        {
            params.request_sequence_lengths_device = request_lengths_device;
            params.request_row_stride_source =
                HiddenStateRowsSelectStage::RequestRowStrideSource::
                    ExternalDeviceScalar;
            params.request_row_stride_device = request_row_stride_device;
        }
        else if (row_index_source ==
                 HiddenStateRowsSelectStage::DeviceRowIndexSource::
                     ShiftedPrefillKVProgress)
        {
            params.request_sequence_lengths_device = request_lengths_device;
            params.request_row_stride_source =
                HiddenStateRowsSelectStage::RequestRowStrideSource::
                    ExternalDeviceScalar;
            params.request_row_stride_device = request_row_stride_device;
            params.main_cached_tokens_device = main_cached_tokens_device;
            params.shifted_cached_tokens_device =
                shifted_cached_tokens_device;
            params.request_index = request_index;
        }

        auto stage = ComputeStageFactory::createHiddenStateRowsSelect(params);
        auto *rows_select_stage =
            dynamic_cast<HiddenStateRowsSelectStage *>(stage.get());
        if (!rows_select_stage)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] HiddenStateRowsSelect factory returned an incompatible terminal-hidden publication stage");
            return false;
        }

        auto graph = std::make_unique<ComputeGraph>();
        const std::string stable_node_name =
            node_name && node_name[0] != '\0'
                ? std::string(node_name)
                : std::string("mtp_terminal_hidden_rows_select");
        // The same owner plan prices this selector as one bounded native
        // helper. Capture verifies its physical nodes before instantiation.
        graph->setExecutableMemoryClass(GPUGraphExecutableClass::BoundedFlatHelper);
        graph->addNode(
            stable_node_name,
            std::move(stage),
            state_.device_id);

        if (accepted_publication)
        {
            HiddenStateRowsSelectStage::Params logits;
            logits.device_id = state_.device_id;
            logits.input = logits_input;
            logits.output = logits_output;
            logits.seq_len = static_cast<int>(logits_input->rows());
            logits.d_model = logits_columns;
            // LOGITS is the existing scalar request-zero authority. Other
            // requests retain their original all-position batch consumers.
            logits.selected_row_count = 1;
            logits.input_buffer_id = logits_source_id;
            logits.output_buffer_id = local_logits ? BufferId::LOGITS_LOCAL : BufferId::LOGITS;
            logits.device_row_index_source = HiddenStateRowsSelectStage::DeviceRowIndexSource::ExternalDeviceIndices;
            logits.external_device_row_indices = external_device_row_indices;
            logits.row_copy_policy = {
                .negative_rows = NegativeRowSelectionPolicy::PreserveDestination,
                .source_stride = static_cast<int>(logits_input->cols()),
                .destination_stride = static_cast<int>(logits_output->cols())};
            const std::string logits_node = stable_node_name + "_canonical_logits";
            graph->addNode(logits_node, ComputeStageFactory::createHiddenStateRowsSelect(std::move(logits)), state_.device_id);
            graph->addDependency(logits_node, stable_node_name);
            graph->setTerminalNode(logits_node);
        }

        invalidateSparseQueryParents();
        cache.invalidate();
        cache.graph = std::move(graph);
        cache.stage = rows_select_stage;
        cache.logits_input = logits_input;
        cache.logits_output = logits_output;
        cache.logits_columns = logits_columns;
        cache.logits_source_rows = logits_input ? static_cast<int>(logits_input->rows()) : 0;
        cache.logits_input_device = logits_input ? logits_input->gpu_data_ptr() : nullptr;
        cache.logits_output_device = logits_output ? logits_output->gpu_data_ptr() : nullptr;
        cache.logits_source_stride = logits_input ? static_cast<int>(logits_input->cols()) : 0;
        cache.logits_destination_stride = logits_output ? static_cast<int>(logits_output->cols()) : 0;
        cache.input = state_.hidden.get();
        cache.output = state_.prefix_terminal_hidden.get();
        cache.device = state_.device_id;
        cache.workspace_generation = generation;
        cache.seq_capacity = seq_capacity;
        cache.d_model = graph_builder_->config().residualWidth();
        cache.selected_row_count = selected_row_count;
        cache.fixed_contiguous_row_start = fixed_contiguous_row_start;
        cache.request_row_stride = 0;
        cache.request_sequence_lengths_device = request_lengths_device;
        cache.request_row_stride_source =
            params.request_row_stride_source;
        cache.request_row_stride_device = request_row_stride_device;
        cache.main_cached_tokens_device = main_cached_tokens_device;
        cache.shifted_cached_tokens_device = shifted_cached_tokens_device;
        cache.shifted_origin=graph_builder_->config().mtpShiftedCacheOrigin();
        cache.request_index = request_index;
        cache.row_buffer_name = stable_row_buffer;
        cache.external_device_row_indices =
            external_device_row_indices;
        cache.row_index_source = row_index_source;
        cache.valid = true;

        PerfStatsCollector::addCounter(
            "mtp",
            "terminal_hidden_publication_graph_materializations",
            1.0,
            "graph_setup",
            state_.device_id.toString(),
            {{"rows", std::to_string(selected_row_count)},
             {"source",
              row_index_source ==
                      HiddenStateRowsSelectStage::DeviceRowIndexSource::
                          FixedContiguousRange
                  ? "fixed_contiguous"
                  : row_index_source ==
                            HiddenStateRowsSelectStage::DeviceRowIndexSource::
                                RequestTerminalLengths
                        ? "request_terminal"
                        : row_index_source ==
                                  HiddenStateRowsSelectStage::
                                      DeviceRowIndexSource::
                                          ShiftedPrefillKVProgress
                              ? "shifted_prefill_kv_progress"
                              : "device_accepted"},
             {"workspace_generation", std::to_string(generation)}});
        return true;
    }

    bool DeviceGraphOrchestrator::materializeMTPTerminalHiddenPublicationGraphs()
    {
        if (!graph_builder_ || !retainsMTPGraphCapacity(graph_builder_->config().mtp) ||
            !state_.device_id.is_gpu())
        {
            return true;
        }
        /*
         * These publication graphs copy terminal hidden rows into the learned
         * predictor's archive and, for prefill, bind both main and shifted KV
         * progress counters.  They are therefore predictor-owner work, not
         * generic main-model MTP work.  Pipeline followers already retain and
         * capture their MTP condition/verifier graphs through the primary
         * forward family; constructing this terminal family on them would
         * contradict initializeKVCaches() and manufacture a second ownership
         * rule from cache presence.
         */
        const MTPStateRole mtp_state_role = resolveMTPStateRole(
            retainsMTPGraphCapacity(graph_builder_->config().mtp),
            !pp_stage_config_ || pp_stage_config_->has_lm_head);
        if (mtp_state_role != MTPStateRole::PredictorOwner)
            return true;
        if (mtp_sidecar_condition_token_slot_width_ <= 0 ||
            static_cast<int>(
                mtp_terminal_hidden_contiguous_rows_select_caches_.size()) !=
                mtp_sidecar_condition_token_slot_width_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Terminal-hidden publication cache cardinality does not match MTP graph capacity");
            return false;
        }

        for (int row_count = 1;
             row_count <= mtp_sidecar_condition_token_slot_width_;
             ++row_count)
        {
            auto &cache =
                mtp_terminal_hidden_contiguous_rows_select_caches_[
                    static_cast<size_t>(row_count - 1)];
            if (!cache)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Missing terminal-hidden contiguous publication cache for rows="
                          << row_count);
                return false;
            }
            const std::string node_name =
                "mtp_terminal_hidden_contiguous_rows_" +
                std::to_string(row_count);
            if (!materializeMTPTerminalHiddenRowsSelectGraph(
                    *cache,
                    node_name.c_str(),
                    HiddenStateRowsSelectStage::DeviceRowIndexSource::
                        FixedContiguousRange,
                    row_count,
                    /*fixed_contiguous_row_start=*/0))
            {
                return false;
            }
        }

        const int accepted_row_capacity =
            stochastic_batch_output_request_capacity_;
        if (accepted_row_capacity <= 0 ||
            accepted_row_capacity >
                mtp_sidecar_condition_token_slot_width_ ||
            static_cast<int>(
                mtp_terminal_hidden_device_accepted_rows_select_caches_.size()) !=
                accepted_row_capacity)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Accepted-state terminal-hidden graph family does not match its mailbox: requests="
                      << accepted_row_capacity
                      << " mailbox_rows="
                      << mtp_sidecar_condition_token_slot_width_
                      << " caches="
                      << mtp_terminal_hidden_device_accepted_rows_select_caches_.size());
            return false;
        }
        for (int accepted_request_count = 1;
             accepted_request_count <= accepted_row_capacity;
             ++accepted_request_count)
        {
            auto &cache =
                mtp_terminal_hidden_device_accepted_rows_select_caches_[
                    static_cast<size_t>(accepted_request_count - 1)];
            if (!cache)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Missing accepted-state terminal-hidden publication cache for requests="
                          << accepted_request_count);
                return false;
            }
            const std::string node_name =
                "mtp_terminal_hidden_device_accepted_rows_" +
                std::to_string(accepted_request_count);
            if (!materializeMTPTerminalHiddenRowsSelectGraph(
                    *cache,
                    node_name.c_str(),
                    HiddenStateRowsSelectStage::DeviceRowIndexSource::
                        ExternalDeviceIndices,
                    accepted_request_count,
                    /*fixed_contiguous_row_start=*/0,
                    MTPSpecDecodeWorkspaceBuffers::ACCEPTED_STATE_SLOT_INDICES,
                    mtp_spec_decode_metadata_binding_
                        .devicePointers()
                        .accepted_state_slot_indices))
            {
                return false;
            }
        }

        if (static_cast<int>(
                mtp_terminal_hidden_request_rows_select_caches_.size()) !=
            stochastic_batch_output_request_capacity_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Request-terminal publication graph family does not match configured request capacity");
            return false;
        }
        for (int request_count = 1;
             request_count <= stochastic_batch_output_request_capacity_;
             ++request_count)
        {
            auto &cache =
                mtp_terminal_hidden_request_rows_select_caches_[
                    static_cast<size_t>(request_count - 1)];
            if (!cache)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Missing request-terminal publication cache for requests="
                          << request_count);
                return false;
            }

            const std::string node_name =
                "mtp_terminal_hidden_request_rows_" +
                std::to_string(request_count);
            if (!materializeMTPTerminalHiddenRowsSelectGraph(
                    *cache,
                    node_name.c_str(),
                    HiddenStateRowsSelectStage::DeviceRowIndexSource::
                        RequestTerminalLengths,
                    request_count))
            {
                return false;
            }
        }

        const size_t expected_shifted_prefill_graphs =
            static_cast<size_t>(stochastic_batch_output_request_capacity_) *
            static_cast<size_t>(mtp_sidecar_condition_token_slot_width_);
        if (mtp_shifted_prefill_hidden_rows_select_caches_.size() !=
            expected_shifted_prefill_graphs)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Shifted-prefill publication graph family does not match request/row capacity: expected="
                      << expected_shifted_prefill_graphs
                      << " actual="
                      << mtp_shifted_prefill_hidden_rows_select_caches_.size());
            return false;
        }
        for (int request_index = 0;
             request_index < stochastic_batch_output_request_capacity_;
             ++request_index)
        {
            for (int row_count = 1;
                 row_count <= mtp_sidecar_condition_token_slot_width_;
                 ++row_count)
            {
                const size_t cache_index =
                    static_cast<size_t>(request_index) *
                        static_cast<size_t>(
                            mtp_sidecar_condition_token_slot_width_) +
                    static_cast<size_t>(row_count - 1);
                auto &cache =
                    mtp_shifted_prefill_hidden_rows_select_caches_[cache_index];
                if (!cache)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Missing shifted-prefill terminal-hidden graph owner: request="
                              << request_index << " rows=" << row_count);
                    return false;
                }
                const std::string node_name =
                    "mtp_shifted_prefill_hidden_request_" +
                    std::to_string(request_index) + "_rows_" +
                    std::to_string(row_count);
                if (!materializeMTPTerminalHiddenRowsSelectGraph(
                        *cache,
                        node_name.c_str(),
                        HiddenStateRowsSelectStage::DeviceRowIndexSource::
                            ShiftedPrefillKVProgress,
                        row_count,
                        /*fixed_contiguous_row_start=*/0,
                        /*row_buffer_name=*/nullptr,
                        /*external_device_row_indices=*/nullptr,
                        request_index))
                {
                    return false;
                }
            }
        }
        return true;
    }

    bool DeviceGraphOrchestrator::executeMTPHiddenRowsSelect(
        TensorBase *input,
        BufferId input_buffer_id,
        TensorBase *output,
        BufferId output_buffer_id,
        MTPTerminalHiddenRowsSelectGraphCache &cache,
        const char *node_name,
        int row_start,
        int row_count,
        int seq_len,
        void *stream)
    {
        if (row_start < 0 || row_count <= 0 || row_start + row_count > seq_len)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid MTP hidden rows selection: start="
                      << row_start << " count=" << row_count << " seq_len=" << seq_len);
            return false;
        }
        if (row_count > mtp_sidecar_condition_token_slot_width_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP hidden rows selection exceeds the configured sidecar capacity: rows="
                      << row_count << " capacity="
                      << mtp_sidecar_condition_token_slot_width_);
            return false;
        }
        if (!input || !output)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot select MTP hidden rows without input/output buffer");
            return false;
        }
        if (graph_builder_->config().residualWidth() <= 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot select MTP hidden rows with invalid d_model="
                      << graph_builder_->config().residualWidth());
            return false;
        }

        const ResidualRowGeometry residual_rows(graph_builder_->config().residualWidth());
        residual_rows.validate(input->shape());
        residual_rows.validate(output->shape());
        const size_t hidden_rows = input->rows();
        if (hidden_rows > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Hidden-state rows-select capacity exceeds int range: rows="
                      << hidden_rows);
            return false;
        }
        const int seq_capacity = static_cast<int>(hidden_rows);
        if (seq_capacity < seq_len)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Hidden-state rows-select capacity too small: capacity="
                      << seq_capacity << " seq_len=" << seq_len);
            return false;
        }

        if (!state_.device_id.is_gpu())
        {
            std::vector<int> selected_rows;
            selected_rows.reserve(static_cast<size_t>(row_count));
            for (int i = 0; i < row_count; ++i)
                selected_rows.push_back(row_start + i);
            return executeMTPHiddenRowsSelect(
                input,
                input_buffer_id,
                output,
                output_buffer_id,
                cache,
                node_name,
                selected_rows,
                seq_len,
                stream);
        }

        if (input != state_.hidden.get() ||
            output != state_.prefix_terminal_hidden.get() ||
            input_buffer_id != BufferId::HIDDEN_STATE ||
            output_buffer_id != BufferId::PREFIX_TERMINAL_HIDDEN)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] GPU contiguous terminal-hidden publication must use the canonical arena bindings");
            return false;
        }

        const uint64_t current_workspace_generation =
            workspaceGeneration(state_.device_id);
        const bool prepared =
            cache.valid && cache.graph && cache.stage &&
            cache.input == input && cache.output == output &&
            cache.device == state_.device_id &&
            cache.workspace_generation == current_workspace_generation &&
            cache.seq_capacity == seq_capacity &&
            cache.d_model == graph_builder_->config().residualWidth() &&
            cache.selected_row_count == row_count &&
            cache.fixed_contiguous_row_start == row_start &&
            cache.row_index_source ==
                HiddenStateRowsSelectStage::DeviceRowIndexSource::
                    FixedContiguousRange;
        if (!prepared)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Fixed-contiguous MTP terminal-hidden publication reached execution without a current pre-materialized graph: node="
                      << (node_name ? node_name : "<unnamed>")
                      << " start=" << row_start
                      << " rows=" << row_count
                      << " seq_capacity=" << seq_capacity
                      << " workspace_generation="
                      << current_workspace_generation);
            return false;
        }
        if (!stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Fixed-contiguous MTP hidden rows-select requires the exact GPU producer stream");
            return false;
        }
        if (!executeMTPTerminalHiddenRowsCaptured(
                cache,
                stream,
                node_name && node_name[0] != '\0'
                    ? node_name
                    : "mtp_terminal_hidden_rows_fixed_contiguous"))
            return false;

        if (workspaceGeneration(state_.device_id) !=
            current_workspace_generation)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Terminal-hidden publication changed GPU workspace generation during execution");
            return false;
        }
        return true;
    }

    bool DeviceGraphOrchestrator::executeMTPHiddenRowsSelect(
        TensorBase *input,
        BufferId input_buffer_id,
        TensorBase *output,
        BufferId output_buffer_id,
        MTPTerminalHiddenRowsSelectGraphCache &cache,
        const char *node_name,
        const std::vector<int> &selected_rows,
        int seq_len,
        void *stream)
    {
        if (selected_rows.empty())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Empty MTP hidden row-index selection");
            return false;
        }
        if (selected_rows.size() >
            static_cast<size_t>(mtp_sidecar_condition_token_slot_width_))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP hidden row-index selection exceeds the configured sidecar capacity: rows="
                      << selected_rows.size() << " capacity="
                      << mtp_sidecar_condition_token_slot_width_);
            return false;
        }
        if (!input || !output)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot select MTP hidden rows without input/output buffer");
            return false;
        }
        if (graph_builder_->config().residualWidth() <= 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot select MTP hidden rows with invalid d_model="
                      << graph_builder_->config().residualWidth());
            return false;
        }

        const ResidualRowGeometry residual_rows(graph_builder_->config().residualWidth());
        residual_rows.validate(input->shape());
        residual_rows.validate(output->shape());
        const size_t hidden_rows = input->rows();
        if (hidden_rows > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Hidden-state rows-select capacity exceeds int range: rows="
                      << hidden_rows);
            return false;
        }
        const int seq_capacity = static_cast<int>(hidden_rows);
        if (seq_capacity < seq_len)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Hidden-state rows-select capacity too small: capacity="
                      << seq_capacity << " seq_len=" << seq_len);
            return false;
        }
        for (const int row : selected_rows)
        {
            if (row < 0 || row >= seq_len)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Invalid MTP hidden row index "
                          << row << " for seq_len=" << seq_len);
                return false;
            }
        }

        const uint64_t current_workspace_generation =
            state_.device_id.is_gpu() ? workspaceGeneration(state_.device_id) : 0;
        const bool workspace_generation_changed =
            state_.device_id.is_gpu() &&
            cache.workspace_generation != 0 &&
            current_workspace_generation != cache.workspace_generation;

        const bool rebuild =
            !cache.valid ||
            !cache.graph ||
            !cache.stage ||
            workspace_generation_changed ||
            cache.input != input ||
            cache.output != output ||
            cache.device != state_.device_id ||
            cache.seq_capacity != seq_capacity ||
            cache.d_model != graph_builder_->config().residualWidth() ||
            cache.selected_row_count != static_cast<int>(selected_rows.size());

        if (rebuild)
        {
            HiddenStateRowsSelectStage::Params params;
            params.device_id = state_.device_id;
            params.input = input;
            params.output = output;
            params.seq_len = seq_capacity;
            params.d_model = graph_builder_->config().residualWidth();
            params.selected_row_count = static_cast<int>(selected_rows.size());
            params.selected_row_indices = selected_rows;
            params.input_buffer_id = input_buffer_id;
            params.output_buffer_id = output_buffer_id;
            params.workspace_buffer_name =
                std::string(HiddenStateRowsSelectStage::WS_SELECTED_ROWS_ARRAY) +
                "_" + (node_name && node_name[0] != '\0' ? node_name : "mtp_hidden_rows_select");

            auto stage = ComputeStageFactory::createHiddenStateRowsSelect(params);
            auto *rows_select_stage = dynamic_cast<HiddenStateRowsSelectStage *>(stage.get());
            if (!rows_select_stage)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] HiddenStateRowsSelect factory returned an incompatible stage");
                return false;
            }

            auto graph = std::make_unique<ComputeGraph>();
            graph->addNode(node_name && node_name[0] != '\0' ? node_name : "mtp_hidden_rows_select",
                           std::move(stage),
                           state_.device_id);

            cache.graph = std::move(graph);
            cache.stage = rows_select_stage;
            cache.input = input;
            cache.output = output;
            cache.device = state_.device_id;
            cache.workspace_generation = 0;
            cache.seq_capacity = seq_capacity;
            cache.d_model = graph_builder_->config().residualWidth();
            cache.selected_row_count = static_cast<int>(selected_rows.size());
            cache.row_buffer_name = params.workspace_buffer_name;
            cache.fixed_contiguous_row_start = 0;
            cache.request_row_stride = 0;
            cache.request_sequence_lengths_device = nullptr;
            cache.row_index_source =
                HiddenStateRowsSelectStage::DeviceRowIndexSource::
                    StageOwnedIndices;
            cache.valid = true;
        }

        if (!cache.stage->setSelectedRowsForReplay(selected_rows))
            return false;

        if (state_.device_id.is_gpu())
        {
            if (!stream)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] MTP hidden rows-select requires the exact GPU producer stream");
                return false;
            }
            cache.stage->setGPUStream(stream);
        }

        IDeviceContext *ctx = getDeviceContext(state_.device_id);
        if (!ctx)
            return false;

        if (!execute(*cache.graph, ctx))
            return false;

        if (state_.device_id.is_gpu())
            cache.workspace_generation = workspaceGeneration(state_.device_id);
        return true;
    }

    bool DeviceGraphOrchestrator::executeMTPHiddenRowsSelectFromDeviceMetadata(
        TensorBase *input,
        TensorBase *output,
        MTPTerminalHiddenRowsSelectGraphCache &cache,
        const char *node_name,
        const char *row_buffer_name,
        const int32_t *row_indices_device,
        int row_count,
        int seq_len,
        void *stream)
    {
        if (!state_.device_id.is_gpu())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-metadata hidden row selection requires a GPU device");
            return false;
        }
        if (!row_buffer_name || row_buffer_name[0] == '\0' ||
            !row_indices_device)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-metadata hidden row selection requires an exact producer-owned device pointer");
            return false;
        }
        if (row_count <= 0 ||
            row_count > mtp_sidecar_condition_token_slot_width_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-metadata hidden rows selection exceeds the configured sidecar capacity: rows="
                      << row_count << " capacity="
                      << mtp_sidecar_condition_token_slot_width_);
            return false;
        }
        if (seq_len <= 0)
            return false;
        if (!input || !output)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot select device-metadata MTP hidden rows without input/output buffer");
            return false;
        }
        if (graph_builder_->config().residualWidth() <= 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot select device-metadata MTP hidden rows with invalid d_model="
                      << graph_builder_->config().residualWidth());
            return false;
        }

        const ResidualRowGeometry residual_rows(graph_builder_->config().residualWidth());
        residual_rows.validate(input->shape());
        residual_rows.validate(output->shape());
        const size_t hidden_rows = input->rows();
        if (hidden_rows > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Hidden-state rows-select capacity exceeds int range: rows="
                      << hidden_rows);
            return false;
        }
        const int seq_capacity = static_cast<int>(hidden_rows);
        if (seq_capacity < seq_len)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Hidden-state rows-select capacity too small: capacity="
                      << seq_capacity << " seq_len=" << seq_len);
            return false;
        }

        const uint64_t current_workspace_generation =
            workspaceGeneration(state_.device_id);
        if (!cache.valid || !cache.graph || !cache.stage ||
            current_workspace_generation == 0 ||
            cache.workspace_generation != current_workspace_generation ||
            cache.input != input || cache.output != output ||
            cache.device != state_.device_id ||
            cache.seq_capacity != seq_capacity ||
            cache.d_model != graph_builder_->config().residualWidth() ||
            cache.selected_row_count != row_count ||
            cache.row_index_source !=
                HiddenStateRowsSelectStage::DeviceRowIndexSource::
                    ExternalDeviceIndices ||
            cache.row_buffer_name != row_buffer_name ||
            cache.external_device_row_indices != row_indices_device)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-metadata terminal-hidden publication reached execution without its exact pre-materialized producer binding"
                      << " rows=" << row_count
                      << " workspace_generation="
                      << current_workspace_generation
                      << " cache_generation="
                      << cache.workspace_generation
                      << " producer_ptr=" << row_indices_device
                      << " cache_ptr="
                      << cache.external_device_row_indices);
            return false;
        }

        if (!stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-metadata MTP hidden rows-select requires the exact GPU producer stream");
            return false;
        }
        if (!executeMTPTerminalHiddenRowsCaptured(
                cache,
                stream,
                node_name && node_name[0] != '\0'
                    ? node_name
                    : "mtp_terminal_hidden_rows_device_metadata"))
            return false;

        return true;
    }

    bool DeviceGraphOrchestrator::executeMTPTerminalHiddenRowsCaptured(
        MTPTerminalHiddenRowsSelectGraphCache &cache,
        void *producer_consumer_stream,
        const char *perf_context)
    {
        if (!state_.device_id.is_gpu() || !producer_consumer_stream ||
            !cache.valid || !cache.graph || !cache.stage)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Captured terminal-hidden rows execution requires a complete GPU graph and an explicit producer/consumer stream");
            return false;
        }
        if (!debugEnv().execution.gpu_graphs)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] GPU terminal-hidden publication requires graph capture; eager execution is not a production alternative");
            return false;
        }

        IDeviceContext *ctx = getDeviceContext(state_.device_id);
        if (!ctx)
            return false;
        auto &gpu_ctx =
            GPUDeviceContextPool::instance().getContext(state_.device_id);
        if (!cache.segment_cache.ensureCaptureStream(
                &gpu_ctx,
                state_.device_id,
                /*context_from_process_pool=*/true) ||
            !cache.segment_cache.orderCaptureStreamAfter(
                &gpu_ctx,
                producer_consumer_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to order terminal-hidden publication onto its persistent capture stream");
            return false;
        }

        cache.segment_cache.perf_context =
            perf_context && perf_context[0] != '\0'
                ? perf_context
                : "mtp_terminal_hidden_rows";
        auto capture_policy = buildDecodeCapturePolicy(
            /*has_collectives=*/false,
            ctx);
        if (!capture_policy.allow_cached_graph_replay)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Terminal-hidden publication requires mandatory cached graph replay");
            return false;
        }
        capture_policy.defer_final_sync = true;
        capture_policy.force_recapture = false;
        capture_policy.graph_replay_plan_policy =
            DeviceGraphExecutor::GraphReplayPlanPolicy::RequireFullGraph;

        cache.graph->reset();
        bool used_graph_replay = false;
        if (!executor_.executeDecodeWithCapturePolicy(
                *cache.graph,
                ctx,
                &cache.segment_cache,
                cache.segment_cache.capture_stream,
                &gpu_ctx,
                /*collective_nodes=*/nullptr,
                capture_policy,
                &used_graph_replay))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Terminal-hidden publication graph execution failed under mandatory capture policy");
            return false;
        }

        if (!cache.segment_cache.orderStreamAfterCapture(
                &gpu_ctx,
                producer_consumer_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to publish terminal-hidden graph output to its consumer stream");
            return false;
        }
        return true;
    }

    bool DeviceGraphOrchestrator::selectMTPTerminalHiddenRows(
        int row_start,
        int row_count,
        int seq_len,
        void *stream)
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
            return true;
        PerfStatsCollector::ScopedTimer timer(
            "mtp",
            "terminal_hidden_rows_select",
            perfPhaseName(),
            state_.device_id.toString(),
            {{"rows", std::to_string(row_count)}});
        if (seq_len <= 0)
            return false;
        state_.mtp_terminal_hidden_publication.invalidate();
        if (row_count == 1 && !state_.device_id.is_gpu())
            return selectMTPTerminalHiddenRow(row_start, seq_len, stream);
        if (!state_.hidden)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot select MTP terminal hidden rows without hidden-state buffer");
            return false;
        }
        if (!ensureMTPTerminalHiddenBuffer(row_count))
            return false;
        if (!beginMTPTerminalHiddenMailboxWrite(
                stream,
                "mtp_terminal_hidden_rows_select"))
        {
            return false;
        }
        MTPTerminalHiddenRowsSelectGraphCache *selected_cache =
            &mtp_terminal_hidden_rows_select_cache_;
        if (state_.device_id.is_gpu() && row_start == 0)
        {
            if (row_count > static_cast<int>(
                                mtp_terminal_hidden_contiguous_rows_select_caches_.size()) ||
                !mtp_terminal_hidden_contiguous_rows_select_caches_[
                    static_cast<size_t>(row_count - 1)])
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Missing pre-materialized MTP terminal-hidden catchup graph for rows="
                          << row_count);
                return false;
            }
            selected_cache =
                mtp_terminal_hidden_contiguous_rows_select_caches_[
                    static_cast<size_t>(row_count - 1)]
                    .get();
            const uint64_t generation =
                workspaceGeneration(state_.device_id);
            if (!selected_cache->valid || !selected_cache->graph ||
                !selected_cache->stage ||
                selected_cache->workspace_generation != generation ||
                selected_cache->row_index_source !=
                    HiddenStateRowsSelectStage::DeviceRowIndexSource::
                        FixedContiguousRange ||
                selected_cache->fixed_contiguous_row_start != 0 ||
                selected_cache->selected_row_count != row_count)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] MTP terminal-hidden catchup graph reached decode without current pre-materialized bindings: rows="
                          << row_count
                          << " workspace_generation=" << generation);
                return false;
            }
        }
        if (!executeMTPHiddenRowsSelect(
                state_.hidden.get(),
                BufferId::HIDDEN_STATE,
                state_.prefix_terminal_hidden.get(),
                BufferId::PREFIX_TERMINAL_HIDDEN,
                *selected_cache,
                "mtp_terminal_hidden_rows_select",
                row_start,
                row_count,
                seq_len,
                stream))
        {
            return false;
        }
        if (!publishMTPTerminalHiddenMailboxReady(
                stream,
                "mtp_terminal_hidden_rows_select"))
        {
            return false;
        }
        state_.mtp_terminal_hidden_publication.publishMainForward();
        return true;
    }

    bool DeviceGraphOrchestrator::selectMTPTerminalHiddenRows(
        const std::vector<int> &row_indices,
        int seq_len,
        void *stream)
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
            return true;
        if (state_.device_id.is_gpu())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] GPU terminal-hidden publication forbids host-authored row vectors; use a fixed-range or device-metadata graph");
            return false;
        }
        state_.mtp_terminal_hidden_publication.invalidate();
        PerfStatsCollector::ScopedTimer timer(
            "mtp",
            "terminal_hidden_rows_select",
            perfPhaseName(),
            state_.device_id.toString(),
            {{"rows", std::to_string(row_indices.size())},
             {"selection", "explicit"}});
        if (seq_len <= 0 || row_indices.empty())
            return false;
        if (!state_.hidden)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot select MTP terminal hidden rows without hidden-state buffer");
            return false;
        }
        if (!ensureMTPTerminalHiddenBuffer(static_cast<int>(row_indices.size())))
            return false;
        if (!beginMTPTerminalHiddenMailboxWrite(
                stream,
                "mtp_terminal_hidden_rows_select_explicit"))
        {
            return false;
        }
        if (!executeMTPHiddenRowsSelect(
                state_.hidden.get(),
                BufferId::HIDDEN_STATE,
                state_.prefix_terminal_hidden.get(),
                BufferId::PREFIX_TERMINAL_HIDDEN,
                mtp_terminal_hidden_rows_select_cache_,
                "mtp_terminal_hidden_rows_select",
                row_indices,
                seq_len,
                stream))
        {
            return false;
        }
        if (!publishMTPTerminalHiddenMailboxReady(
                stream,
                "mtp_terminal_hidden_rows_select_explicit"))
        {
            return false;
        }
        state_.mtp_terminal_hidden_publication.publishMainForward();
        return true;
    }

    bool DeviceGraphOrchestrator::selectMTPTerminalHiddenRowsFromDeviceRequestLengths(
        int request_count,
        int request_row_stride,
        int total_rows,
        void *stream)
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
            return true;
        state_.mtp_terminal_hidden_publication.invalidate();
        PerfStatsCollector::ScopedTimer timer(
            "mtp",
            "terminal_hidden_rows_select",
            perfPhaseName(),
            state_.device_id.toString(),
            {{"rows", std::to_string(request_count)},
             {"selection", "device_request_lengths"}});

        if (!state_.device_id.is_gpu())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident request-length terminal-hidden selection requires a GPU runner");
            return false;
        }
        if (graph_builder_->config()
                .mtp_request_terminal_hidden_publication !=
            MTPRequestTerminalHiddenPublicationPolicy::
                GraphCapturedDeviceGeometry)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Request-terminal publication reached execution without the graph-declared device-geometry policy");
            return false;
        }
        if (request_count <= 0 || request_row_stride <= 0 ||
            request_count > stochastic_batch_output_request_capacity_ ||
            request_count > request_sequence_lengths_capacity_ ||
            request_count > static_cast<int>(
                                mtp_terminal_hidden_request_rows_select_caches_
                                    .size()) ||
            request_count >
                std::numeric_limits<int>::max() / request_row_stride ||
            total_rows != request_count * request_row_stride)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid resident request-terminal geometry: requests="
                      << request_count
                      << " stride=" << request_row_stride
                      << " total_rows=" << total_rows
                      << " request_capacity="
                      << request_sequence_lengths_capacity_
                      << " mailbox_capacity="
                      << mtp_sidecar_condition_token_slot_width_);
            return false;
        }
        if (!state_.hidden ||
            state_.hidden->rows() < static_cast<size_t>(total_rows))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident request-terminal publication exceeds the hidden-state tensor");
            return false;
        }
        if (!request_sequence_lengths_dev_ || !request_row_stride_dev_ ||
            request_sequence_lengths_active_count_ < request_count)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident request-terminal publication requires current device-owned request lengths: requests="
                      << request_count
                      << " active_lengths="
                      << request_sequence_lengths_active_count_);
            return false;
        }
        if (!stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident request-terminal publication requires the exact GPU producer stream");
            return false;
        }
        if (!ensureMTPTerminalHiddenBuffer(request_count))
            return false;

        const uint64_t generation = workspaceGeneration(state_.device_id);
        auto &request_cache_owner =
            mtp_terminal_hidden_request_rows_select_caches_[
                static_cast<size_t>(request_count - 1)];
        if (!request_cache_owner)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Request-terminal publication has no graph owner for requests="
                      << request_count);
            return false;
        }
        auto &request_cache = *request_cache_owner;
        if (!request_cache.valid || !request_cache.graph ||
            !request_cache.stage ||
            request_cache.workspace_generation != generation ||
            request_cache.input != state_.hidden.get() ||
            request_cache.output != state_.prefix_terminal_hidden.get() ||
            request_cache.row_index_source !=
                HiddenStateRowsSelectStage::DeviceRowIndexSource::
                    RequestTerminalLengths ||
            request_cache.selected_row_count != request_count ||
            request_cache.seq_capacity < total_rows ||
            request_cache.request_row_stride != 0 ||
            request_cache.request_sequence_lengths_device !=
                static_cast<const int32_t *>(request_sequence_lengths_dev_) ||
            request_cache.request_row_stride_source !=
                HiddenStateRowsSelectStage::RequestRowStrideSource::
                    ExternalDeviceScalar ||
            request_cache.request_row_stride_device !=
                request_row_stride_dev_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Request-terminal hidden publication reached execution without current pre-materialized bindings: requests="
                      << request_count
                      << " stride=" << request_row_stride
                      << " total_rows=" << total_rows
                      << " graph_capacity=" << request_cache.seq_capacity
                      << " workspace_generation=" << generation);
            return false;
        }
        if (!beginMTPTerminalHiddenMailboxWrite(
                stream,
                "mtp_terminal_hidden_rows_select_device_request_lengths"))
        {
            return false;
        }

        if (!executeMTPTerminalHiddenRowsCaptured(
                request_cache,
                stream,
                "mtp_terminal_hidden_rows_device_request_lengths"))
            return false;
        if (workspaceGeneration(state_.device_id) != generation)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Request-terminal publication changed GPU workspace generation during execution");
            return false;
        }
        if (!publishMTPTerminalHiddenMailboxReady(
                stream,
                "mtp_terminal_hidden_rows_select_device_request_lengths"))
        {
            return false;
        }

        state_.mtp_terminal_hidden_publication.publishMainForward();
        return true;
    }

    bool DeviceGraphOrchestrator::selectMTPTerminalHiddenRowsFromShiftedPrefillProgress(
        int request_index,
        int request_count,
        int row_count,
        int total_rows,
        void *stream)
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
            return true;
        state_.mtp_terminal_hidden_publication.invalidate();
        PerfStatsCollector::ScopedTimer timer(
            "mtp",
            "terminal_hidden_rows_select",
            perfPhaseName(),
            state_.device_id.toString(),
            {{"rows", std::to_string(row_count)},
             {"request", std::to_string(request_index)},
             {"selection", "device_shifted_kv_progress"}});

        if (!state_.device_id.is_gpu() ||
            graph_builder_->config()
                    .mtp_shifted_prefill_hidden_publication !=
                MTPShiftedPrefillHiddenPublicationPolicy::
                    GraphIntegratedKVTransaction)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Shifted-prefill progress selection reached execution without its graph-declared GPU policy");
            return false;
        }
        if (request_count <= 0 ||
            request_count > stochastic_batch_output_request_capacity_ ||
            request_index < 0 || request_index >= request_count ||
            row_count <= 0 ||
            row_count > mtp_sidecar_condition_token_slot_width_ ||
            total_rows <= 0 || total_rows % request_count != 0 ||
            request_sequence_lengths_active_count_ < request_count ||
            !request_sequence_lengths_dev_ || !request_row_stride_dev_ ||
            !state_.hidden ||
            state_.hidden->rows() < static_cast<size_t>(total_rows) ||
            !state_.kv_cache || state_.mtp_kv_caches.empty() ||
            !state_.mtp_kv_caches[0])
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Shifted-prefill progress selection received incomplete geometry or cache ownership: request="
                      << request_index << "/" << request_count
                      << " rows=" << row_count
                      << " total_rows=" << total_rows
                      << " active_geometry="
                      << request_sequence_lengths_active_count_);
            return false;
        }
        if (!stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Shifted-prefill progress selection requires the exact hidden-state producer stream");
            return false;
        }
        if (!ensureMTPTerminalHiddenBuffer(row_count))
            return false;

        /*
         * The previous KV-only sidecar is the producer of the shifted count
         * consumed by this selector. Queue that event on the main-forward
         * stream before replaying the selector. The selector then publishes the
         * terminal-hidden mailbox event consumed by the next sidecar, forming
         * one explicit S(KV) -> R(select) -> S(sidecar) chain with no host wait.
         */
        if (!waitForPendingShiftedMTPKVReady(
                stream,
                "mtp_shifted_prefill_hidden_selection"))
        {
            return false;
        }

        const size_t cache_index =
            static_cast<size_t>(request_index) *
                static_cast<size_t>(mtp_sidecar_condition_token_slot_width_) +
            static_cast<size_t>(row_count - 1);
        if (cache_index >=
                mtp_shifted_prefill_hidden_rows_select_caches_.size() ||
            !mtp_shifted_prefill_hidden_rows_select_caches_[cache_index])
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Shifted-prefill progress graph family has no owner: request="
                      << request_index << " rows=" << row_count);
            return false;
        }
        auto &cache =
            *mtp_shifted_prefill_hidden_rows_select_caches_[cache_index];
        const uint64_t generation = workspaceGeneration(state_.device_id);
        const int32_t *main_count =
            state_.kv_cache->deviceSequenceCachedTokenCountPtr(request_index);
        const int32_t *shifted_count =
            state_.mtp_kv_caches[0]
                ->deviceSequenceCachedTokenCountPtr(request_index);
        if (!cache.valid || !cache.graph || !cache.stage ||
            cache.workspace_generation != generation ||
            cache.input != state_.hidden.get() ||
            cache.output != state_.prefix_terminal_hidden.get() ||
            cache.row_index_source !=
                HiddenStateRowsSelectStage::DeviceRowIndexSource::
                    ShiftedPrefillKVProgress ||
            cache.selected_row_count != row_count ||
            cache.request_index != request_index ||
            cache.request_sequence_lengths_device !=
                static_cast<const int32_t *>(request_sequence_lengths_dev_) ||
            cache.request_row_stride_source !=
                HiddenStateRowsSelectStage::RequestRowStrideSource::
                    ExternalDeviceScalar ||
            cache.request_row_stride_device != request_row_stride_dev_ ||
            cache.main_cached_tokens_device != main_count ||
            cache.shifted_cached_tokens_device != shifted_count ||
            cache.shifted_origin != graph_builder_->config().mtpShiftedCacheOrigin() ||
            cache.seq_capacity < total_rows)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Shifted-prefill hidden publication reached execution without current pre-materialized bindings: request="
                      << request_index << " rows=" << row_count
                      << " total_rows=" << total_rows
                      << " graph_capacity=" << cache.seq_capacity
                      << " workspace_generation=" << generation);
            return false;
        }
        if (!beginMTPTerminalHiddenMailboxWrite(
                stream,
                "mtp_shifted_prefill_hidden_selection"))
        {
            return false;
        }

        if (!executeMTPTerminalHiddenRowsCaptured(
                cache,
                stream,
                "mtp_shifted_prefill_hidden_selection"))
            return false;
        if (workspaceGeneration(state_.device_id) != generation)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Shifted-prefill selector changed GPU workspace generation during replay");
            return false;
        }
        if (!publishMTPTerminalHiddenMailboxReady(
                stream,
                "mtp_shifted_prefill_hidden_selection"))
        {
            return false;
        }

        state_.mtp_terminal_hidden_publication.publishMainForward();
        PerfStatsCollector::addCounter(
            "mtp",
            "shifted_prefill_device_progress_hidden_publications",
            static_cast<double>(row_count),
            perfPhaseName(),
            state_.device_id.toString(),
            {{"request", std::to_string(request_index)},
             {"rows", std::to_string(row_count)},
             {"ownership", "canonical_kv_progress"}});
        return true;
    }

    bool DeviceGraphOrchestrator::selectMTPTerminalHiddenRowsFromDeviceAcceptedState(
        int row_count,
        int seq_len,
        void *stream)
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
            return true;
        state_.mtp_terminal_hidden_publication.invalidate();
        PerfStatsCollector::ScopedTimer timer(
            "mtp",
            "terminal_hidden_rows_select",
            perfPhaseName(),
            state_.device_id.toString(),
            {{"rows", std::to_string(row_count)},
             {"selection", "device_accepted_state"}});

        if (!state_.device_id.is_gpu())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device accepted-state terminal-hidden selection requires a GPU runner");
            return false;
        }
        if (seq_len <= 0 || row_count <= 0 ||
            row_count > mtp_sidecar_condition_token_slot_width_)
            return false;
        if (!state_.hidden)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot select device accepted-state terminal hidden without hidden-state buffer");
            return false;
        }
        if (!mtp_spec_decode_metadata_binding_.hasWorkspace() ||
            !mtp_spec_decode_metadata_binding_.devicePointers().accepted_state_slot_indices)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device accepted-state terminal-hidden selection requires bound MTP metadata workspace");
            return false;
        }
        const int publication_row_capacity =
            stochastic_batch_output_request_capacity_;
        if (row_count > publication_row_capacity ||
            row_count > static_cast<int>(
                            mtp_terminal_hidden_device_accepted_rows_select_caches_
                                .size()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device accepted-state terminal-hidden request count exceeds the pre-materialized graph capacity: requests="
                      << row_count
                      << " capacity=" << publication_row_capacity);
            return false;
        }
        if (!ensureMTPTerminalHiddenBuffer(row_count))
            return false;
        const uint64_t generation = workspaceGeneration(state_.device_id);
        auto &accepted_cache_owner =
            mtp_terminal_hidden_device_accepted_rows_select_caches_[
                static_cast<size_t>(row_count - 1)];
        if (!accepted_cache_owner)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device accepted-state terminal-hidden graph family has an empty request-count slot: requests="
                      << row_count);
            return false;
        }
        auto &accepted_cache = *accepted_cache_owner;
        if (!accepted_cache.valid || !accepted_cache.graph ||
            !accepted_cache.stage || !accepted_cache.logitsBindingsCurrent() ||
            accepted_cache.workspace_generation != generation ||
            accepted_cache.row_index_source !=
                HiddenStateRowsSelectStage::DeviceRowIndexSource::
                    ExternalDeviceIndices ||
            accepted_cache.selected_row_count != row_count ||
            accepted_cache.row_buffer_name !=
                MTPSpecDecodeWorkspaceBuffers::ACCEPTED_STATE_SLOT_INDICES ||
            accepted_cache.external_device_row_indices !=
                mtp_spec_decode_metadata_binding_
                    .devicePointers()
                    .accepted_state_slot_indices)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device accepted-state terminal-hidden graph reached publication without current pre-materialized bindings: requests="
                      << row_count
                      << " capacity=" << publication_row_capacity
                      << " workspace_generation=" << generation);
            return false;
        }
        if (!beginMTPTerminalHiddenMailboxWrite(
                stream,
                "mtp_terminal_hidden_rows_select_device_accepted_state"))
        {
            return false;
        }

        if (!executeMTPHiddenRowsSelectFromDeviceMetadata(
                state_.hidden.get(),
                state_.prefix_terminal_hidden.get(),
                accepted_cache,
                "mtp_terminal_hidden_rows_select_device_accepted_state",
                MTPSpecDecodeWorkspaceBuffers::ACCEPTED_STATE_SLOT_INDICES,
                mtp_spec_decode_metadata_binding_
                    .devicePointers()
                    .accepted_state_slot_indices,
                row_count,
                seq_len,
                stream))
        {
            return false;
        }
        if (!publishMTPTerminalHiddenMailboxReady(
                stream,
                "mtp_terminal_hidden_rows_select_device_accepted_state"))
        {
            return false;
        }

        state_.mtp_terminal_hidden_publication.publishAcceptedVerifier();
        // Batched samplers retain their original all-position publication
        // contract; the canonical rows above are still copied for every request.
        return row_count != 1 || publishCanonicalMTPMainLogits(stream, "device_accepted_verifier_row");
    }

    void DeviceGraphOrchestrator::noteMainForwardHiddenProducedForMTP(
        int seq_len,
        int batch_size,
        std::vector<int> request_lengths,
        MTPTerminalHiddenPublication::Source terminal_hidden_source)
    {
        state_.last_forward_seq_len = seq_len;
        state_.last_forward_batch_size = batch_size;
        if (request_lengths.empty())
        {
            request_lengths.assign(
                static_cast<size_t>(std::max(0, batch_size)),
                seq_len);
        }
        state_.last_forward_request_lengths = std::move(request_lengths);
        if (terminal_hidden_source ==
            MTPTerminalHiddenPublication::Source::MainForward)
            state_.mtp_terminal_hidden_publication.publishMainForward();
        else if (terminal_hidden_source ==
                 MTPTerminalHiddenPublication::Source::Unavailable)
            state_.mtp_terminal_hidden_publication.invalidate();
        else
            throw std::logic_error(
                "Main-forward terminal-hidden publication received a non-main producer");
    }

    bool DeviceGraphOrchestrator::resolveMTPTerminalHiddenInput(
        TensorBase **terminal_hidden,
        BufferId *terminal_hidden_buffer_id,
        const char *operation)
    {
        if (!terminal_hidden || !terminal_hidden_buffer_id)
            return false;

        *terminal_hidden = nullptr;
        *terminal_hidden_buffer_id = BufferId::HIDDEN_STATE;

        if (state_.prefix_terminal_hidden &&
            state_.mtp_terminal_hidden_publication.current())
        {
            *terminal_hidden = state_.prefix_terminal_hidden.get();
            *terminal_hidden_buffer_id = BufferId::PREFIX_TERMINAL_HIDDEN;
            return true;
        }

        if (state_.last_forward_batch_size == 1 && state_.last_forward_seq_len > 1)
        {
            if (!refreshMTPTerminalHiddenState(
                    state_.last_forward_seq_len,
                    state_.last_forward_batch_size))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to refresh terminal hidden for "
                          << (operation ? operation : "MTP sidecar"));
                return false;
            }
            if (!state_.prefix_terminal_hidden ||
                !state_.mtp_terminal_hidden_publication.current())
            {
                LOG_ERROR("[DeviceGraphOrchestrator] MTP terminal hidden refresh did not publish a current buffer for "
                          << (operation ? operation : "MTP sidecar"));
                return false;
            }
            *terminal_hidden = state_.prefix_terminal_hidden.get();
            *terminal_hidden_buffer_id = BufferId::PREFIX_TERMINAL_HIDDEN;
            return true;
        }

        if (state_.last_forward_batch_size > 1)
        {
            if (!refreshMTPTerminalHiddenState(
                    state_.last_forward_seq_len,
                    state_.last_forward_batch_size))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to refresh batched terminal hidden for "
                          << (operation ? operation : "MTP sidecar"));
                return false;
            }
            if (!state_.prefix_terminal_hidden ||
                !state_.mtp_terminal_hidden_publication.current())
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Batched MTP terminal hidden refresh did not publish"
                          << " a current prefix buffer for "
                          << (operation ? operation : "MTP sidecar"));
                return false;
            }
            *terminal_hidden = state_.prefix_terminal_hidden.get();
            *terminal_hidden_buffer_id = BufferId::PREFIX_TERMINAL_HIDDEN;
            return true;
        }

        if (state_.last_forward_seq_len == 1 && state_.last_forward_batch_size == 1 && state_.hidden)
        {
            /*
             * GPU MTP sidecar graphs are captured against concrete tensor
             * addresses.  Feeding HIDDEN_STATE after ordinary decode but
             * PREFIX_TERMINAL_HIDDEN after accepted-state publication makes
             * the first-sidecar graph alternate signatures and rebuild in the
             * hot loop.  Copy the single live row into the stable prefix
             * terminal-hidden buffer instead; this matches the vLLM-style
             * persistent-input-buffer contract and keeps graph replay reusable.
             */
            if (state_.device_id.is_gpu())
            {
                if (!refreshMTPTerminalHiddenState(
                        state_.last_forward_seq_len,
                        state_.last_forward_batch_size))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Failed to refresh single-row terminal hidden for "
                              << (operation ? operation : "MTP sidecar"));
                    return false;
                }
                if (!state_.prefix_terminal_hidden ||
                    !state_.mtp_terminal_hidden_publication.current())
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Single-row MTP terminal hidden refresh did not publish"
                              << " a current prefix buffer for "
                              << (operation ? operation : "MTP sidecar"));
                    return false;
                }
                *terminal_hidden = state_.prefix_terminal_hidden.get();
                *terminal_hidden_buffer_id = BufferId::PREFIX_TERMINAL_HIDDEN;
                return true;
            }
            *terminal_hidden = state_.hidden.get();
            *terminal_hidden_buffer_id = BufferId::HIDDEN_STATE;
            return true;
        }

        LOG_ERROR("[DeviceGraphOrchestrator] "
                  << (operation ? operation : "MTP sidecar")
                  << " requires a current terminal hidden row, but no prefix restore or main forward is current");
        return false;
    }

    bool DeviceGraphOrchestrator::refreshMTPTerminalHiddenState(
        int seq_len,
        int batch_size,
        void *hidden_producer_stream)
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
            return true;
        state_.mtp_terminal_hidden_publication.invalidate();
        PerfStatsCollector::ScopedTimer timer(
            "mtp",
            "terminal_hidden_refresh",
            perfPhaseName(),
            state_.device_id.toString());
        if (seq_len <= 0 || batch_size <= 0)
            return false;
        if (batch_size > std::numeric_limits<int>::max() / seq_len)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP terminal-hidden refresh geometry exceeds int range: batch="
                      << batch_size << " seq_len=" << seq_len);
            return false;
        }
        const int total_rows = seq_len * batch_size;
        void *producer_stream = nullptr;
        if (state_.device_id.is_gpu())
        {
            producer_stream = hidden_producer_stream;
            if (!producer_stream)
            {
                producer_stream = explicitGPUStreamForOperation(
                    "mtp_terminal_hidden_refresh");
                if (!producer_stream ||
                    !waitForForwardGraphOutputReady(
                        producer_stream,
                        DeviceTimelineRole::MTPSidecarGraph,
                        ForwardGraphOutputKind::Any,
                        "mtp_terminal_hidden_refresh"))
                {
                    return false;
                }
            }
        }

        if (batch_size == 1)
        {
            if (state_.device_id.is_gpu() && seq_len > 1)
            {
                /*
                 * Scalar prefill uses the same resident geometry contract as
                 * request-batched prefill. This keeps prompt width out of graph
                 * identity and removes the last seq_len-1 host row cursor from
                 * terminal publication. Single-row decode remains the immutable
                 * row-zero graph below because it has no padded prompt geometry.
                 */
                return selectMTPTerminalHiddenRowsFromDeviceRequestLengths(
                    /*request_count=*/1,
                    seq_len,
                    /*total_rows=*/seq_len,
                    producer_stream);
            }
            if (!selectMTPTerminalHiddenRows(
                    seq_len - 1,
                    /*row_count=*/1,
                    seq_len,
                    producer_stream))
                return false;
        }
        else
        {
            if (state_.device_id.is_gpu())
            {
                /*
                 * Request admission has already published true sequence
                 * lengths into an arena-resident INT32 array. The captured
                 * selector computes each flattened terminal row from those
                 * lengths on the producer stream, so no host vector can become
                 * stale relative to the prefill graph that wrote HIDDEN_STATE.
                 */
                return selectMTPTerminalHiddenRowsFromDeviceRequestLengths(
                    batch_size,
                    seq_len,
                    total_rows,
                    producer_stream);
            }
            if (static_cast<int>(state_.last_forward_request_lengths.size()) <
                batch_size)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Batched MTP terminal hidden capture lacks per-request lengths: batch_size="
                          << batch_size << " recorded="
                          << state_.last_forward_request_lengths.size());
                return false;
            }

            std::vector<int> terminal_rows;
            terminal_rows.reserve(static_cast<size_t>(batch_size));
            for (int request = 0; request < batch_size; ++request)
            {
                const int request_len =
                    state_.last_forward_request_lengths[static_cast<size_t>(request)];
                if (request_len <= 0 || request_len > seq_len)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Invalid batched MTP terminal hidden request length: request="
                              << request << " length=" << request_len
                              << " padded_seq_len=" << seq_len);
                    return false;
                }
                terminal_rows.push_back(request * seq_len + request_len - 1);
            }

            /*
             * Padded request batches are flattened as
             * [request, padded_seq_len, d_model]. Gather each request's true
             * terminal row into the stable sidecar input buffer.
             */
            if (!selectMTPTerminalHiddenRows(
                    terminal_rows,
                    total_rows,
                    producer_stream))
            {
                return false;
            }
        }

        /*
         * Each selector publishes the mailbox at its exact producer boundary.
         * Do not manufacture a second generation here: callers lease the
         * selector's publication, not an administrative wrapper transition.
         */
        return true;
    }

    bool DeviceGraphOrchestrator::ensureMTPCheckpointTerminalHidden()
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
            return true;
        if (state_.prefix_terminal_hidden &&
            state_.mtp_terminal_hidden_publication.current())
            return true;
        if (!state_.hidden ||
            state_.last_forward_seq_len <= 0 ||
            state_.last_forward_batch_size <= 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot materialize MTP checkpoint terminal hidden"
                      << " without a live main-forward hidden row");
            return false;
        }

        /*
         * Keep the capture contract explicit: live MTP checkpoints must carry
         * a terminal hidden row that was selected from the latest main forward
         * on the runner's own stream.  That prevents restore from combining
         * fresh KV/GDN with a stale PREFIX_TERMINAL_HIDDEN buffer.
         */
        return refreshMTPTerminalHiddenState(
            state_.last_forward_seq_len,
            state_.last_forward_batch_size);
    }

    bool DeviceGraphOrchestrator::restoreLiveCheckpointTerminalHidden(
        const PrefixBlockHandle &handle,
        void *stream,
        const char *consumer_name,
        bool publish_mailbox,
        int destination_row)
    {
        auto fail = [&](const std::string &reason) -> bool
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "live_checkpoint_terminal_hidden_restore_failures",
                1.0,
                perfPhaseName(),
                state_.device_id.toString(),
                {{"reason", reason},
                 {"consumer",
                  consumer_name && consumer_name[0] != '\0'
                      ? consumer_name
                      : "unknown"}});
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to restore live-checkpoint terminal hidden on "
                      << state_.device_id.toString()
                      << ": " << reason);
            return false;
        };

        if (!handle.layout.includes_terminal_hidden ||
            handle.layout.terminal_hidden_bytes == 0 ||
            !handle.has_terminal_hidden)
        {
            return fail("missing_terminal_hidden_payload");
        }
        if (!ensureMTPTerminalHiddenBuffer())
        {
            return fail("terminal_hidden_buffer_unavailable");
        }
        const ResidualRowGeometry residual_rows(graph_builder_->config().residualWidth());
        if (!state_.prefix_terminal_hidden || destination_row < 0 ||
            handle.layout.terminal_hidden_bytes != residual_rows.rowBytes())
        {
            return fail("terminal_hidden_payload_differs_from_complete_residual_row");
        }
        const auto destination_offset = residual_rows.elementOffset(
            state_.prefix_terminal_hidden->shape(), static_cast<size_t>(destination_row)) * sizeof(float);

        if (publish_mailbox &&
            !beginMTPTerminalHiddenMailboxWrite(
                stream,
                consumer_name && consumer_name[0] != '\0'
                    ? consumer_name
                    : "live_checkpoint_terminal_hidden_restore"))
        {
            return fail("terminal_hidden_mailbox_acquire_failed");
        }

        const size_t bytes = handle.layout.terminal_hidden_bytes;
        const char *implementation = "host_copy";
        if (state_.device_id.is_gpu())
        {
            if (!stream)
                return fail("missing_explicit_gpu_stream");

            /*
             * Transient rollback state is part of the live GPU transaction. A
             * host-only source is not a slower equivalent implementation: it is
             * an invalid ownership domain that can carry stale tensor state.
             */
            IBackend *backend = getBackendFor(state_.device_id);
            void *destination =
                state_.prefix_terminal_hidden->gpu_data_ptr();
            const void *source = handle.deviceTerminalHiddenData();
            if (!backend || !destination || !source)
            {
                return fail(
                    "gpu_live_checkpoint_requires_device_terminal_hidden");
            }
            destination = static_cast<std::byte *>(destination) + destination_offset;
            if (!backend->deviceCopyAsync(
                    destination,
                    source,
                    bytes,
                    state_.device_id.gpu_ordinal(),
                    stream))
            {
                return fail("terminal_hidden_device_to_device_copy_failed");
            }
            TransferEngine::publishDeviceWrite(
                state_.prefix_terminal_hidden.get(),
                state_.device_id,
                stream);
            implementation = "device_to_device";
        }
        else
        {
            void *destination =
                state_.prefix_terminal_hidden->raw_mutable_data();
            if (!destination || !handle.terminal_hidden)
            {
                return fail(
                    "cpu_live_checkpoint_requires_host_terminal_hidden");
            }
            destination = static_cast<std::byte *>(destination) + destination_offset;
            std::memcpy(destination, handle.terminal_hidden, bytes);
            TransferEngine::publishHostWrite(
                state_.prefix_terminal_hidden.get());
        }

        if (arena_ && arena_->isRegistered(BufferId::PREFIX_TERMINAL_HIDDEN))
        {
            arena_->markWrittenFlagsOnly(
                BufferId::PREFIX_TERMINAL_HIDDEN,
                state_.device_id.is_gpu()
                    ? state_.device_id
                    : DeviceId::cpu());
        }
        state_.mtp_terminal_hidden_publication.publishCheckpointRestore();

        if (publish_mailbox &&
            !publishMTPTerminalHiddenMailboxReady(
                stream,
                consumer_name && consumer_name[0] != '\0'
                    ? consumer_name
                    : "live_checkpoint_terminal_hidden_restore"))
        {
            return fail("terminal_hidden_mailbox_publication_failed");
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "live_checkpoint_terminal_hidden_restores",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"bytes", std::to_string(bytes)},
             {"implementation", implementation},
             {"consumer",
              consumer_name && consumer_name[0] != '\0'
                  ? consumer_name
                  : "unknown"}});
        return true;
    }

    bool DeviceGraphOrchestrator::importMTPCheckpointTerminalHidden(
        const PrefixStateSnapshot &snapshot,
        void *stream,
        const char *consumer_name)
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
            return true;

        auto fail = [&](const std::string &reason) -> bool
        {
            PerfStatsCollector::addCounterWithSequence(
                "mtp",
                "checkpoint_terminal_hidden_import_failures",
                1.0,
                {static_cast<uint64_t>(snapshot.cached_tokens)},
                perfPhaseName(),
                state_.device_id.toString(),
                {{"reason", reason}});
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to import MTP checkpoint terminal hidden on "
                      << state_.device_id.toString()
                      << ": " << reason
                      << " cached_tokens=" << snapshot.cached_tokens);
            return false;
        };

        if (!snapshot.valid || snapshot.cached_tokens <= 0)
            return fail("invalid_or_empty_checkpoint");
        if(snapshot.mtp_shifted_origin!=graph_builder_->config().mtpShiftedCacheOrigin())
            return fail("checkpoint_shifted_origin_mismatch");
        if (!waitForSnapshotReady(
                snapshot,
                stream,
                consumer_name
                    ? consumer_name
                    : "mtp_checkpoint_terminal_hidden_import"))
        {
            return fail("snapshot_ready_wait_failed");
        }

        const PrefixBlockHandle *terminal_handle = nullptr;
        if (snapshot.logical_checkpoint)
        {
            if (snapshot.blocks.size() > 1)
                return fail("logical_checkpoint_has_multiple_payload_blocks");
            if (!snapshot.blocks.empty())
                terminal_handle = &snapshot.blocks.front();
        }
        else if (!snapshot.blocks.empty())
        {
            terminal_handle = &snapshot.blocks.back();
        }
        if (!terminal_handle)
            return fail("missing_terminal_hidden_block");

        /*
         * Import exactly the checkpoint row. Verifier publication retains
         * ownership of KV, recurrent, position, and shifted-MTP state.
         */
        return restoreLiveCheckpointTerminalHidden(
            *terminal_handle,
            stream,
            consumer_name
                ? consumer_name
                : "mtp_checkpoint_terminal_hidden_import",
            /*publish_mailbox=*/true);
    }

    bool DeviceGraphOrchestrator::selectMTPTerminalHiddenRow(int row_idx, int seq_len, void *stream)
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
            return true;
        if (state_.device_id.is_gpu())
        {
            return selectMTPTerminalHiddenRows(
                row_idx,
                /*row_count=*/1,
                seq_len,
                stream);
        }
        state_.mtp_terminal_hidden_publication.invalidate();
        PerfStatsCollector::ScopedTimer timer(
            "mtp",
            "terminal_hidden_row_select",
            perfPhaseName(),
            state_.device_id.toString());
        if (seq_len <= 0)
            return false;

        if (!beginMTPTerminalHiddenMailboxWrite(
                stream,
                "mtp_terminal_hidden_row_select"))
        {
            return false;
        }
        if (!executeMTPTerminalHiddenRowSelect(row_idx, seq_len, stream))
            return false;
        if (!publishMTPTerminalHiddenMailboxReady(
                stream,
                "mtp_terminal_hidden_row_select"))
        {
            return false;
        }
        state_.mtp_terminal_hidden_publication.publishMainForward();
        return true;
    }

    std::optional<size_t>
    DeviceGraphOrchestrator::mtpVerifierPreparationRegistryIndex(
        const MTPVerifierPreparationGraphKey &key,
        std::string *error) const
    {
        auto reject = [&](const std::string &reason)
            -> std::optional<size_t>
        {
            if (error)
                *error = reason;
            return std::nullopt;
        };
        if (error)
            error->clear();

        if (!key.populated() ||
            key.request_count > stochastic_batch_output_request_capacity_ ||
            key.padded_seq_len > mtp_max_verifier_rows_)
        {
            return reject(
                "grouped-verifier preparation key is outside the configured registry geometry");
        }
        const bool exact_controlled_family = key.control_policy ==
                MTPVerifierPreparationControlPolicy::DeviceGenerationControlled &&
            mtpVerifierGraphFamily() == MTPVerifierGraphFamily::ExactRows;
        if (key.request_count == 1 &&
            (exact_controlled_family
                ? !mtpVerifierGraphFamilyContains(key.padded_seq_len, mtp_max_verifier_rows_, mtpVerifierGraphFamily())
                : mtpVerifierPhysicalRowBucket(key.padded_seq_len, mtp_max_verifier_rows_) != key.padded_seq_len))
        {
            return reject(
                "scalar grouped-verifier preparation key is not a canonical physical bucket");
        }

        const size_t expected_registry_size =
            kMTPVerifierPreparationControlPolicyCount *
            static_cast<size_t>(
                stochastic_batch_output_request_capacity_) *
            static_cast<size_t>(mtp_max_verifier_rows_);
        if (mtp_verifier_preparation_graphs_.size() !=
            expected_registry_size)
        {
            return reject(
                "grouped-verifier preparation registry capacity does not match its configured key space");
        }
        const size_t control_policy_index =
            static_cast<size_t>(key.control_policy);
        const size_t registry_index =
            (control_policy_index *
                 static_cast<size_t>(
                     stochastic_batch_output_request_capacity_) +
             static_cast<size_t>(key.request_count - 1)) *
                static_cast<size_t>(mtp_max_verifier_rows_) +
            static_cast<size_t>(key.padded_seq_len - 1);
        if (registry_index >= mtp_verifier_preparation_graphs_.size() ||
            !mtp_verifier_preparation_graphs_[registry_index])
        {
            return reject(
                "grouped-verifier preparation registry has no owner for the exact key");
        }
        return registry_index;
    }

    std::optional<
        DeviceGraphOrchestrator::MTPVerifierPreparationControlPolicy>
    DeviceGraphOrchestrator::mtpVerifierPreparationControlPolicyForRequest(
        int request_count,
        std::string *error) const
    {
        auto reject = [&](const std::string &reason)
            -> std::optional<MTPVerifierPreparationControlPolicy>
        {
            if (error)
                *error = reason;
            return std::nullopt;
        };
        if (error)
            error->clear();
        if (request_count <= 0)
            return reject("verifier preparation requires a positive request count");

        const int active_request_count =
            device_generation_storage_.active_request_count;
        if (active_request_count == 0)
            return MTPVerifierPreparationControlPolicy::Standalone;
        if (active_request_count != request_count ||
            !device_generation_storage_.validFor(request_count))
        {
            return reject(
                "verifier preparation found partial or stale device-generation admission: active=" +
                std::to_string(active_request_count) +
                " requested=" + std::to_string(request_count));
        }
        return MTPVerifierPreparationControlPolicy::
            DeviceGenerationControlled;
    }

    std::optional<MTPVerifierPhysicalWidthPolicy>
    DeviceGraphOrchestrator::mtpVerifierPhysicalWidthPolicyForRequest(
        int request_count,
        std::string *error) const
    {
        const auto control_policy =
            mtpVerifierPreparationControlPolicyForRequest(
                request_count,
                error);
        if (!control_policy)
            return std::nullopt;

        if (*control_policy == MTPVerifierPreparationControlPolicy::Standalone || request_count != 1)
            return MTPVerifierPhysicalWidthPolicy::BoundedLogicalBucket;
        if (!active_device_generation_admission_)
        {
            if (error) *error = "device-controlled verifier width has no admitted MTP depth policy";
            return std::nullopt;
        }
        const auto mode = active_device_generation_admission_->depth_policy.mode ==
                sampling_math::DeviceGenerationPolicyMode::Dynamic
            ? MTPDepthPolicyMode::Dynamic : MTPDepthPolicyMode::Fixed;
        return resolveMTPVerifierPhysicalWidthPolicy(mtpVerifierExecutionRole(), mode);
    }

    MTPVerifierExecutionRole DeviceGraphOrchestrator::mtpVerifierExecutionRole() const
    {
        if (!state_.device_id.is_gpu()) return MTPVerifierExecutionRole::Standalone;
        return llaminar2::mtpVerifierExecutionRole(declaredDeviceGenerationExecutionPolicy());
    }

    MTPVerifierGraphFamily DeviceGraphOrchestrator::mtpVerifierGraphFamily() const
    {
        return mtp_verifier_graph_family_;
    }

    CapturedVerifierExtent DeviceGraphOrchestrator::retainedMTPVerifierExtent(int rows) const
    {
        if (mtpVerifierGraphFamily() != MTPVerifierGraphFamily::ExactRows ||
            !mtp_graph_owner_plan_.ownsControllerExecutables())
            return {};

        return CapturedVerifierExtent::exactHostedRequest(state_.device_id,
            DeviceRowRange::deviceCounted(rows,
                static_cast<const int32_t *>(mtp_verifier_request_lengths_dev_)));
    }

    const ForwardGraphSignature &DeviceGraphOrchestrator::retainedMTPVerifierSignature(
        int rows, MTPVerifierOutcomeGraphMode outcome, SparseQueryExecutionProgram program) const
    {
        if (!forward_engine_ || !mtpVerifierGraphFamilyContains(rows, mtp_max_verifier_rows_, mtpVerifierGraphFamily()) ||
            (outcome != MTPVerifierOutcomeGraphMode::Greedy && outcome != MTPVerifierOutcomeGraphMode::Disabled))
            throw std::logic_error("Exact verifier selection requires an admitted row/outcome family");
        const auto &signature = mtp_retained_verifier_signatures_.at(sparseQueryExecutionProgramIndex(program))
            .at(static_cast<std::size_t>(rows - 2)).at(outcome == MTPVerifierOutcomeGraphMode::Greedy ? 0u : 1u);
        const auto expected_extent = retainedMTPVerifierExtent(rows);
        if (!signature || signature->captured_verifier_extent != expected_extent ||
            signature->seq_len != rows || signature->batch_size != 1 ||
            signature->sparse_query_execution_program != program ||
            signature->mtp_verifier_outcome_graph_mode != outcome ||
            !forward_engine_->retainedDecodeGraph(*signature))
            throw std::logic_error("Exact verifier selection has no original replay-ready setup signature");
        return *signature;
    }

    bool DeviceGraphOrchestrator::prepareMTPDirectTokenHistoryPublication(
        void *producer_stream, std::string *error)
    {
        const auto fail = [&](const char *why) { if (error) *error = why; return false; };
        const auto generation = workspaceGeneration(state_.device_id);
        auto &cache = mtp_direct_token_history_graph_;
        auto *backend = getBackendFor(state_.device_id);
        if (!state_.device_id.is_gpu() || !producer_stream || !backend || generation == 0 ||
            mtp_graph_owner_plan_.directTokenHistoryGraphSlots() != 1 ||
            !stochastic_target_sample_tokens_dev_ || !mtp_generated_token_counts_dev_ || !mtp_greedy_penalty_policy_dev_ ||
            mtp_generated_token_count_capacity_ != state_.vocab_size || sparseQueryParentLaunched())
            return fail("Direct MTP history helper lacks its canonical owner or original arena bindings");
        if (cache.graph || cache.stage)
        {
            if (!cache.graph || !cache.stage || cache.workspace_generation != generation)
                return fail("Direct MTP history helper has a partial or stale retained identity");
            const auto &params = cache.stage->getParams();
            if (params.device_id != state_.device_id || params.backend != backend ||
                params.token_device != stochastic_target_sample_tokens_dev_ ||
                params.counts_device != mtp_generated_token_counts_dev_ ||
                params.policy_device != mtp_greedy_penalty_policy_dev_ || params.vocab_size != state_.vocab_size)
                return fail("Direct MTP history helper cannot replace its original captured pointers");
            return cache.segment_cache.deviceLoopGraphTemplate(*cache.graph, error).has_value();
        }
        if (serving_graph_family_lifecycle_ == ServingGraphFamilyLifecycle::Sealed ||
            isGraphCaptureActive() || !debugEnv().execution.gpu_graphs)
            return fail("Direct MTP history helper must be captured during cold serving setup");
        MTPDirectTokenHistoryPublicationStage::Params params;
        params.device_id = state_.device_id;
        params.backend = backend;
        params.token_device = static_cast<const int32_t *>(stochastic_target_sample_tokens_dev_);
        params.counts_device = static_cast<int32_t *>(mtp_generated_token_counts_dev_);
        params.policy_device = static_cast<MTPGreedyPenaltyPolicy *>(mtp_greedy_penalty_policy_dev_);
        params.vocab_size = state_.vocab_size;
        auto stage = std::make_unique<MTPDirectTokenHistoryPublicationStage>(params);
        cache.stage = stage.get();
        cache.graph = std::make_unique<ComputeGraph>();
        cache.graph->setExecutableMemoryClass(GPUGraphExecutableClass::BoundedFlatHelper);
        cache.graph->addNode("mtp_direct_token_history_publication", std::move(stage), state_.device_id);
        cache.workspace_generation = generation;
        auto *ctx = getDeviceContext(state_.device_id);
        auto &gpu = GPUDeviceContextPool::instance().getContext(state_.device_id);
        if (!ctx || !cache.segment_cache.ensureCaptureStream(&gpu, state_.device_id, true) ||
            !cache.segment_cache.orderCaptureStreamAfter(&gpu, producer_stream))
            return fail("Direct MTP history helper cannot acquire its setup capture timeline");
        cache.segment_cache.perf_context = "mtp_direct_token_history_publication";
        cache.segment_cache.replay_workload = {.seq_len = 1, .batch_size = 1, .m = 1, .decode = true};
        auto policy = buildDecodeCapturePolicy(false, ctx);
        if (!policy.allow_cached_graph_replay)
            return fail("Direct MTP history publication requires retained native graph replay");
        policy.defer_final_sync = true;
        policy.force_recapture = false;
        policy.graph_replay_plan_policy = DeviceGraphExecutor::GraphReplayPlanPolicy::RequireFullGraph;
        cache.graph->reset();
        bool used_graph_replay = false;
        if (!executor_.executeDecodeWithCapturePolicy(*cache.graph, ctx, &cache.segment_cache,
                cache.segment_cache.capture_stream, &gpu, nullptr, policy, &used_graph_replay,
                DeviceGraphExecutor::GraphInitialSubmissionPolicy::MaterializeWithoutLaunch) ||
            !cache.segment_cache.deviceLoopGraphTemplate(*cache.graph, error))
            return fail("Direct MTP history publication did not produce a complete native executable");
        PerfStatsCollector::addCounter("mtp", "direct_token_history_graph_materializations", 1.0,
            "graph_setup", state_.device_id.toString(), {{"target_slot", "0"}, {"initial_submission", "materialize_without_launch"}});
        return true;
    }

    bool DeviceGraphOrchestrator::publishMTPDirectTokenHistory(int target_sample_slot,
        MTPDirectTokenHistoryDisposition disposition)
    {
        const auto fail = [&](const char *why) {
            LOG_ERROR("[DeviceGraphOrchestrator] Direct MTP history publication rejected: " << why);
            return false;
        };
        if (!state_.device_id.is_gpu() || target_sample_slot != 0 ||
            mtp_graph_owner_plan_.directTokenHistoryGraphSlots() != 1 ||
            (disposition != MTPDirectTokenHistoryDisposition::NewlyEmitted &&
             disposition != MTPDirectTokenHistoryDisposition::AlreadyPublished))
            return fail("requires scalar target slot zero and an explicit output disposition");
        auto &cache = mtp_direct_token_history_graph_;
        auto *backend = getBackendFor(state_.device_id);
        if (!arena_ || !backend || mtp_graph_owner_plan_.directTokenHistoryGraphSlots() != 1 ||
            !cache.graph || !cache.stage || cache.workspace_generation == 0 ||
            cache.workspace_generation != workspaceGeneration(state_.device_id) || sparseQueryParentLaunched())
            return fail("the pre-admitted native helper is missing, stale, or concurrently in use");
        const auto &params = cache.stage->getParams();
        if (params.device_id != state_.device_id || params.backend != backend ||
            params.token_device != stochastic_target_sample_tokens_dev_ ||
            params.counts_device != mtp_generated_token_counts_dev_ ||
            params.policy_device != mtp_greedy_penalty_policy_dev_ || params.vocab_size != state_.vocab_size ||
            arena_->getDevicePtr(BufferId::STOCHASTIC_TARGET_SAMPLE_TOKENS, state_.device_id) != params.token_device ||
            arena_->getDevicePtr(BufferId::MTP_GENERATED_TOKEN_COUNTS, state_.device_id) != params.counts_device ||
            arena_->getDevicePtr(BufferId::MTP_GREEDY_PENALTY_POLICY, state_.device_id) != params.policy_device)
            return fail("original target/history pointer or vocabulary identity changed");
        std::string error;
        const auto executable = cache.segment_cache.deviceLoopGraphTemplate(*cache.graph, &error);
        if (!executable || !executable->capture ||
            target_sample_slot >= static_cast<int>(stochastic_target_sample_ready_.size()))
            return fail("original native executable or target event owner is missing");
        auto &ready = stochastic_target_sample_ready_[static_cast<size_t>(target_sample_slot)];
        if (disposition == MTPDirectTokenHistoryDisposition::AlreadyPublished)
        {
            // Closing this consumer right cannot change a token/count. It also
            // prevents a later erroneous New call on the same publication.
            if (ready.valid) ready.history_right = StochasticTargetHistoryRight::Consumed;
            return true;
        }
        const auto source = deviceStochasticTargetSampleProducerSlot(target_sample_slot);
        if (ready.history_right != StochasticTargetHistoryRight::Available ||
            !source.valid() || source.token_device != params.token_device ||
            !waitForRequiredStochasticTargetSampleReady(target_sample_slot, source.stream, "direct_mtp_output_history") ||
            !arena_->prepareForRead(BufferId::STOCHASTIC_TARGET_SAMPLE_TOKENS, state_.device_id, source.stream) ||
            !arena_->prepareForRead(BufferId::MTP_GENERATED_TOKEN_COUNTS, state_.device_id, source.stream) ||
            !arena_->prepareForRead(BufferId::MTP_GREEDY_PENALTY_POLICY, state_.device_id, source.stream))
            return fail("exact sample/history publication is absent, already consumed, or unordered");
        // Retire the right before submission. An exceptional launch/publication
        // cannot be retried against a possibly already incremented histogram.
        ready.history_right = StochasticTargetHistoryRight::Consumed;
        if (!executable->capture->launchOnStream(source.stream) ||
            !publishPreparedArenaGraphInput(BufferId::MTP_GENERATED_TOKEN_COUNTS,
                mtp_generated_token_counts_dev_, source.stream, state_.device_id, "direct_mtp_output_history") ||
            !publishPreparedArenaGraphInput(BufferId::MTP_GREEDY_PENALTY_POLICY,
                mtp_greedy_penalty_policy_dev_, source.stream, state_.device_id, "direct_mtp_output_history"))
            return fail("exact sample/history event edge or retained executable submission failed");
        PerfStatsCollector::addCounter("mtp", "direct_token_history_publications", 1.0,
            "decode", state_.device_id.toString(), {{"target_slot", "0"}, {"source", "resident_sample"}});
        return true;
    }

    bool DeviceGraphOrchestrator::prepareHostedMTPDraftHelperFamily(
        DeviceGenerationSamplingMode sampling_mode, void *producer_stream, std::string *error)
    {
        const auto fail = [&](const std::string &why) { if (error) *error = why; return false; };
        const auto generation = workspaceGeneration(state_.device_id);
        if (!isValidDeviceGenerationSamplingMode(sampling_mode) ||
            !producer_stream || generation == 0 || sparseQueryParentLaunched() ||
            mtpVerifierGraphFamily() != MTPVerifierGraphFamily::ExactRows ||
            mtp_max_draft_depth_ <= 0 ||
            mtp_draft_token_publication_graphs_.size() < static_cast<std::size_t>(mtp_max_draft_depth_) ||
            !mtp_draft_token_publication_graphs_.front())
            return fail("Hosted draft helper family has no idle admitted capacity or preparation stream");
        const ComputeGraph *full_graph = nullptr;
        if (!validateHostedMTPSidecarReplay(MTPSidecarCaptureRole::Full, 1, &full_graph, error) ||
            !mtpDraftTokenPublicationDeviceLoopGraphTemplate(0, error))
            return false;
        const auto &full = mtp_sidecar_depth0_device_token_cache_;
        const auto &source = *mtp_draft_token_publication_graphs_.front();
        const auto original = source.stage->getParams();
        // Authenticate the law of the actual first-transaction producer.
        // Greedy proposals consume request branch penalties. Stochastic
        // verification uses an unpenalized one-hot greedy proposal even when
        // the target/verifier distribution has request-owned penalties; see
        // sampleStochasticDraftProposalOnDeviceImpl. Those are distinct laws,
        // not a changed buffer owner or permission to rebind a captured stage.
        const MTPRequestPenaltyPolicy draft_penalty_policy =
            sampling_mode == DeviceGenerationSamplingMode::Greedy
                ? mtp_request_penalty_policy_ : MTPRequestPenaltyPolicy{};
        const auto logits_it = state_.extension_buffers.find(mtpSamplingLogitsBufferId());
        const auto hidden_it = state_.extension_buffers.find(BufferId::MTP_HIDDEN);
        const int full_slot = mtp_sidecar_capture_layout_.conditionTokenSlot(MTPSidecarCaptureRole::Full, 1);
        const int chained_slot = mtp_sidecar_capture_layout_.conditionTokenSlot(MTPSidecarCaptureRole::Chained, 1);
        const auto condition_offset = [&](int slot) {
            return slot * mtp_sidecar_condition_token_slot_width_;
        };
        if (!full_graph || !source.valid || source.workspace_generation != generation ||
            !full.uses_device_token_ids || !full.uses_device_position_ids || full.seq_len != 1 ||
            full.batch_size != 1 || full.first_seq_idx != 0 || full.token_ids.size() != 1 ||
            !mtp_sidecar_condition_token_dev_ || !mtp_sidecar_position_ids_dev_ ||
            full_slot < 0 || chained_slot < 0 || condition_offset(full_slot) < 0 ||
            condition_offset(chained_slot) < 0 ||
            condition_offset(full_slot) >= mtp_sidecar_condition_token_capacity_ ||
            condition_offset(chained_slot) >= mtp_sidecar_condition_token_capacity_ ||
            full.condition_token_device != static_cast<int32_t *>(mtp_sidecar_condition_token_dev_) + condition_offset(full_slot) ||
            full.position_ids_device != mtp_sidecar_position_ids_dev_ ||
            logits_it == state_.extension_buffers.end() || !logits_it->second ||
            hidden_it == state_.extension_buffers.end() || !hidden_it->second ||
            original.device_id != state_.device_id || original.backend != getBackendFor(state_.device_id) ||
            original.vocab_size != state_.vocab_size || original.logits_row != 0 ||
            original.destination_slot != 0 || original.prior_draft_count != 0 ||
            original.logits_row_device != logits_it->second->gpu_data_ptr() ||
            original.draft_tokens_device != stochastic_draft_sample_tokens_dev_ ||
            original.draft_values_device != stochastic_draft_sample_probs_dev_ ||
            original.next_chain_condition_token_device != static_cast<int32_t *>(mtp_sidecar_condition_token_dev_) + condition_offset(chained_slot) ||
            original.next_chain_position_id_device != mtp_sidecar_position_ids_dev_ ||
            original.chain_position_increment != 1 ||
            original.argmax_partial_values_device != argmax_partial_vals_dev_ ||
            original.argmax_partial_indices_device != argmax_partial_idxs_dev_ ||
            original.argmax_partial_capacity != argmax_partial_capacity_ ||
            original.apply_penalties != draft_penalty_policy.enabled())
            return fail("Hosted draft helpers lost the exact original scalar sidecar/publication bindings");
        if (original.apply_penalties &&
            (original.first_condition_token_device != full.condition_token_device ||
             original.prior_draft_tokens_device != stochastic_draft_sample_tokens_dev_ ||
             original.generated_token_counts_device != mtp_generated_token_counts_dev_ ||
             original.penalty_policy_device != mtp_greedy_penalty_policy_dev_))
            return fail("Hosted draft helper source differs from the admitted device penalty history");

        // Full is already an actual successful producer even at initial depth1.
        // Chained is another scalar topology, not one graph per executed depth.
        // Reuse the canonical builder and original persistent input ownership,
        // stopping before token staging or model execution.
        if (mtp_max_draft_depth_ > 1)
        {
            const ComputeGraph *chained_graph = nullptr;
            std::string chained_error;
            if (!validateHostedMTPSidecarReplay(MTPSidecarCaptureRole::Chained, 1,
                    &chained_graph, &chained_error))
            {
                if (!executeMTPDepth0Batched(full.token_ids.data(), 1, hidden_it->second.get(),
                        full.position_id, "mtp_decode_chained", false, BufferId::MTP_HIDDEN, true,
                        full.condition_token_device, -1, false, 1, nullptr, full.position_ids_device,
                        nullptr, 0, nullptr, 0, -1, 0, nullptr, 1, 1, 1, 0, 0,
                        DeviceGraphExecutor::GraphInitialSubmissionPolicy::MaterializeWithoutLaunch))
                    return fail("Hosted chained sidecar failed canonical cold construction/capture on " + state_.device_id.toString());
                if (workspaceGeneration(state_.device_id) != generation)
                    return fail("Cold chained sidecar changed the admitted workspace generation");
                if (!validateHostedMTPSidecarReplay(MTPSidecarCaptureRole::Chained, 1, &chained_graph, error))
                    return false; // Preserve the actual failed readiness check, never the pre-capture lookup error.
            }
        }

        // Only immutable slot geometry differs between publications. Captured
        // diagnostics deliberately omit the terminal-hidden input after slot0,
        // matching the canonical live materializer rather than borrowing its row.
        for (int slot = 0; slot < mtp_max_draft_depth_; ++slot)
        {
            if (!mtp_draft_token_publication_graphs_[static_cast<std::size_t>(slot)])
                return fail("Hosted draft helper family is missing an admitted slot owner");
            auto params = original;
            params.destination_slot = slot;
            params.prior_draft_count = slot;
            if (slot != 0 && params.first_transaction_diagnostic_device)
            {
                params.diagnostic_condition_token_device = original.next_chain_condition_token_device;
                const auto terminal_input = static_cast<std::size_t>(
                    sampling_math::MTPFirstTransactionDraftBoundary::TerminalHiddenInput);
                params.diagnostic_boundary_words_device[terminal_input] = nullptr;
                params.diagnostic_boundary_word_counts[terminal_input] = 0;
            }
            auto &cache = *mtp_draft_token_publication_graphs_[static_cast<std::size_t>(slot)];
            if (!cache.valid || !cache.graph || !cache.stage || cache.workspace_generation != generation ||
                !cache.stage->hasSameCaptureIdentity(params))
            {
                invalidateSparseQueryParents();
                cache.invalidate();
                auto stage = std::make_unique<MTPDraftTokenPublicationStage>(std::move(params));
                cache.stage = stage.get();
                cache.graph = std::make_unique<ComputeGraph>();
                cache.graph->setExecutableMemoryClass(GPUGraphExecutableClass::BoundedFlatHelper);
                cache.graph->addNode("mtp_draft_token_publication_" + std::to_string(slot),
                    std::move(stage), state_.device_id);
                cache.workspace_generation = generation;
                cache.valid = true;
            }
            if (!executeMTPDraftTokenPublicationCaptured(producer_stream, 0, slot, error,
                    DeviceGraphExecutor::GraphInitialSubmissionPolicy::MaterializeWithoutLaunch) ||
                !mtpDraftTokenPublicationDeviceLoopGraphTemplate(slot, error))
                return false;
        }
        return true;
    }

    bool DeviceGraphOrchestrator::prepareHostedMTPVerifierBranch(int rows, int source_rows,
        DeviceGenerationSamplingMode sampling, void *stream, std::string *error)
    {
        const auto fail = [&](const std::string &why) { if (error) *error = why; return false; };
        if (mtpVerifierGraphFamily() != MTPVerifierGraphFamily::ExactRows || sparseQueryParentLaunched() || !stream)
            return fail("Hosted verifier preparation requires its admitted exact family and idle parents");
        const MTPVerifierPreparationGraphKey source_key{
            .control_policy = MTPVerifierPreparationControlPolicy::DeviceGenerationControlled,
            .request_count = 1, .padded_seq_len = source_rows};
        const auto source_index = mtpVerifierPreparationRegistryIndex(source_key, error);
        if (!source_index) return false;
        const auto &source = *mtp_verifier_preparation_graphs_[*source_index];
        if (!source.valid || !source.stage || source.workspace_generation != workspaceGeneration(state_.device_id) ||
            source.key != source_key || !source.stage->getParams().generation_control_device)
            return fail("Hosted verifier preparation lost its original controlled admission bindings");
        auto params = source.stage->getParams();
        if (params.request_count != 1 || params.valid_graph_rows_device || params.token_rows.size() != 1 ||
            params.generation_control_device != device_generation_storage_.control_device)
            return fail("Hosted exact-row family requires the canonical scalar device-controlled preparation");
        auto token_row = params.token_rows.front();
        token_row.draft_token_count = rows - 1; // Capture evidence; the controller owns live clipping.
        params.token_rows = std::span<const MTPVerifierPreparationStage::TokenRowBinding>(&token_row, 1);
        params.padded_seq_len = rows;
        params.valid_graph_row_count = rows;
        MTPVerifierPreparationGraphKey target_key;
        if (!materializeMTPVerifierPreparationGraph(params, &target_key, error) ||
            !executeMTPVerifierPreparationCaptured(stream, target_key, error,
                DeviceGraphExecutor::GraphInitialSubmissionPolicy::MaterializeWithoutLaunch))
            return false;
        const auto target_index = mtpVerifierPreparationRegistryIndex(target_key, error);
        if (!target_index) return false;
        auto &target = *mtp_verifier_preparation_graphs_[*target_index];
        const auto outcome = sampling == DeviceGenerationSamplingMode::Greedy
            ? MTPVerifierOutcomeGraphMode::Greedy : MTPVerifierOutcomeGraphMode::Disabled;
        try
        {
            for (std::size_t program = 0; program < sparseQueryProgramCount(); ++program)
                target.paired_forward_signatures[program] = retainedMTPVerifierSignature(rows, outcome,
                    program == 0 ? SparseQueryExecutionProgram::RankingPossible : SparseQueryExecutionProgram::ProvenNoRanking);
        }
        catch (const std::exception &exception) { return fail(exception.what()); }
        return true;
    }
} // namespace llaminar2
