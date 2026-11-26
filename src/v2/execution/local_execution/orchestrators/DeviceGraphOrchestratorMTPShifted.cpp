/**
 * @file DeviceGraphOrchestratorMTPShifted.cpp
 * @brief DeviceGraphOrchestrator method shard (27488..32477).
 */
#include "DeviceGraphOrchestratorInternal.h"

namespace llaminar2
{

    bool DeviceGraphOrchestrator::bindMTPMainTerminalHiddenTransaction(
        ForwardInput &input,
        int request_count,
        MTPMainTerminalHiddenGraphBinding::Purpose purpose)
    {
        input.mtp_main_terminal_hidden.reset();
        input.mtp_main_terminal_logits.reset();
        // Only the tail archives the terminal hidden row consumed by drafting.
        // Intermediate activation banks are already ordered by pipeline edges.
        if (pp_stage_config_ && !pp_stage_config_->has_lm_head)
            return true;

        const bool declaration_only =
            purpose == MTPMainTerminalHiddenGraphBinding::Purpose::
                           WorkspaceFamilyDeclaration;
        // Retained setup of an explicitly speculative graph must bind its
        // complete transaction. Ordinary Off prefill/decode stays ordinary;
        // physical capacity alone cannot change those executable identities.
        const bool mtp_role = input.execution_role == ForwardExecutionRole::MTPCondition ||
            input.execution_role == ForwardExecutionRole::GroupedMTPVerifier;
        const auto transaction_submission = declaration_only
            ? ForwardGraphSubmissionIntent::WorkspaceDeclaration
            : mtp_role ? input.graph_submission_intent : ForwardGraphSubmissionIntent::Execute;
        if (!graph_builder_ ||
            !requiresMTPGraphTransactionBinding(
                graph_builder_->config().mtp.enabled,
                retainsMTPGraphCapacity(graph_builder_->config().mtp), transaction_submission))
            return true;
        if (input.execution_role != ForwardExecutionRole::MainInference &&
            input.execution_role != ForwardExecutionRole::MTPCondition)
        {
            return true;
        }
        if (input.execution_phase != ForwardExecutionPhase::Decode)
            return true;
        if (state_.device_id.is_gpu())
        {
            if (input.seq_len <= 0 || (input.seq_len != 1 && request_count != 1) ||
                request_count <= 0 || request_count != input.batch_size ||
                !state_.logits || graph_builder_->config().vocab_size <= 0)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Canonical MTP logits require scalar main rows and original vocabulary storage");
                return false;
            }
            const uint64_t generation = workspaceGeneration(state_.device_id);
            if (!declaration_only && generation == 0)
                return false;
            uint64_t identity = UINT64_C(1469598103934665603);
            const auto mix = [&](uint64_t value) { identity = (identity ^ value) * UINT64_C(1099511628211); };
            mix(declaration_only ? UINT64_MAX : generation);
            mix(static_cast<uint64_t>(request_count));
            mix(static_cast<uint64_t>(input.seq_len));
            for (TensorBase *tensor : {state_.logits.get(), state_.logits_local.get()})
            {
                mix(reinterpret_cast<uintptr_t>(tensor));
                if (!tensor) continue;
                if (tensor->native_type() != TensorType::FP32 || tensor->rows() < 1 ||
                    tensor->cols() > std::numeric_limits<int>::max() || (!declaration_only && !tensor->gpu_data_ptr()))
                    return false;
                mix(reinterpret_cast<uintptr_t>(tensor->gpu_data_ptr()));
                mix(static_cast<uint64_t>(tensor->rows()));
                mix(static_cast<uint64_t>(tensor->cols()));
            }
            const int full_columns = graph_builder_->config().vocab_size;
            const int local_columns = graph_builder_->config().vocab_local;
            mix(static_cast<uint64_t>(full_columns));
            mix(static_cast<uint64_t>(local_columns));
            mix(static_cast<uint64_t>(localLogitsVocabOffset()));
            input.mtp_main_terminal_logits = MTPMainTerminalLogitsGraphBinding{
                .full = state_.logits.get(), .local = state_.logits_local.get(),
                .full_columns = full_columns, .local_columns = local_columns,
                .request_count = request_count, .capture_identity = identity ? identity : UINT64_MAX,
                .declaration_only = declaration_only};
        }
        if (input.state_transaction ==
            ForwardStateTransaction::RestoredPrefixMTPDecodeBridge)
        {
            const ForwardExecutionRole expected_role =
                state_.device_id.is_gpu()
                    ? ForwardExecutionRole::MTPCondition
                    : ForwardExecutionRole::MainInference;
            if (input.execution_role != expected_role ||
                input.seq_len != 1 || request_count != 1 ||
                input.batch_size != 1 ||
                (state_.device_id.is_gpu() &&
                 !input.shifted_mtp_prefill.has_value()))
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Restored-prefix MTP decode "
                    "bridge has an incomplete typed state transaction");
                return false;
            }
            /*
             * GPU shifted publication archives the new terminal row itself.
             * CPU publication is completed by the host-owned shifted bridge
             * after main decode, while the previous archive is still intact.
             */
            return true;
        }
        const bool contiguous_terminal_rows =
            input.seq_len == 1 || request_count == 1;
        if (input.shifted_mtp_prefill.has_value() || input.seq_len <= 0 ||
            request_count <= 0 || request_count != input.batch_size ||
            !contiguous_terminal_rows)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP main decode terminal publication requires one contiguous terminal row per request"
                      << " seq_len=" << input.seq_len
                      << " requests=" << request_count
                      << " batch=" << input.batch_size
                      << " shifted_prefill="
                      << input.shifted_mtp_prefill.has_value());
            return false;
        }
        if (!ensureMTPTerminalHiddenBuffer(request_count) ||
            !state_.prefix_terminal_hidden ||
            state_.prefix_terminal_hidden->native_type() != TensorType::FP32 ||
            state_.prefix_terminal_hidden->rows() <
                static_cast<size_t>(request_count) ||
            state_.prefix_terminal_hidden->cols() <
                static_cast<size_t>(state_.d_model))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP main decode terminal publication has no complete persistent archive");
            return false;
        }

        const bool gpu_archive_ready =
            !state_.device_id.is_gpu() ||
            state_.prefix_terminal_hidden->gpu_data_ptr() != nullptr;
        const bool cpu_archive_ready =
            state_.device_id.is_gpu() ||
            state_.prefix_terminal_hidden->raw_data() != nullptr;
        if (!gpu_archive_ready || !cpu_archive_ready)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP main decode terminal archive is not resident on its execution device");
            return false;
        }

        const uint64_t generation = workspaceGeneration(state_.device_id);
        if (state_.device_id.is_gpu() && !declaration_only && generation == 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] GPU MTP main decode terminal publication requires a finalized workspace generation");
            return false;
        }

        uint64_t identity = UINT64_C(1469598103934665603);
        const int source_row_start =
            input.seq_len == 1 ? 0 : input.seq_len - 1;
        auto mix = [&](uint64_t value)
        {
            identity ^= value;
            identity *= UINT64_C(1099511628211);
        };
        mix(declaration_only ? UINT64_MAX : generation);
        mix(static_cast<uint64_t>(request_count));
        mix(static_cast<uint64_t>(input.seq_len));
        mix(static_cast<uint64_t>(source_row_start));
        mix(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
            state_.prefix_terminal_hidden.get())));
        mix(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
            state_.device_id.is_gpu()
                ? state_.prefix_terminal_hidden->gpu_data_ptr()
                : state_.prefix_terminal_hidden->raw_data())));
        if (identity == 0)
            identity = UINT64_MAX;

        MTPMainTerminalHiddenGraphBinding binding{
            .terminal_hidden_archive =
                state_.prefix_terminal_hidden.get(),
            .request_count = request_count,
            .source_row_start = source_row_start,
            .capture_identity = identity,
            .purpose = purpose,
        };
        if (!binding.validForRequestCount(request_count))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP main decode terminal publication produced an incomplete graph binding");
            return false;
        }
        input.mtp_main_terminal_hidden = std::move(binding);
        return true;
    }

    bool DeviceGraphOrchestrator::populateCPUMTPShiftedCacheFromPrefill(
        const int *tokens,
        int seq_len,
        int batch_size,
        int position_offset,
        const std::vector<int> *request_lengths,
        const std::vector<int> *position_ids)
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled || state_.mtp_kv_caches.empty())
            return true;
        if (!state_.device_id.is_cpu())
        {
            throw std::runtime_error(
                "populateCPUMTPShiftedCacheFromPrefill is a CPU-only lifecycle boundary; GPU shifted prefill must be graph-integrated");
        }
        if (!tokens || seq_len <= 0)
            return false;

        requireMTPShiftedHostPrefill(graph_builder_->config().mtpShiftedCacheOrigin(),
            position_offset,seq_len,batch_size,
            request_lengths ? std::optional(std::span<const int>(*request_lengths)) : std::nullopt,
            position_ids ? std::optional(std::span<const int>(*position_ids)) : std::nullopt);
        const int logical_seq_len =
            request_lengths && !request_lengths->empty()
                ? std::clamp(request_lengths->front(), 1, seq_len)
                : seq_len;

        const FrozenModelWeightSet *mtp_weight_set =
            selectMTPDecodeWeightSet(
                "populateCPUMTPShiftedCacheFromPrefill");
        if (!mtp_weight_set)
        {
            return false;
        }

        auto bindings = makeModelWeightBindings(
            *mtp_weight_set, state_.device_id);
        if (bindings.mtp.empty() || bindings.mtp.depths.empty())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] populateCPUMTPShiftedCacheFromPrefill requested without MTP weight bindings");
            return false;
        }

        const int previous_tokens = std::max(0, position_offset);
        if (!state_.mtp_kv_caches[0])
            return true;

        if (batch_size > 1)
        {
            if (!request_lengths ||
                request_lengths->size() < static_cast<size_t>(batch_size) ||
                !position_ids ||
                position_ids->size() < static_cast<size_t>(batch_size * seq_len))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Batched shifted MTP prefill requires explicit request lengths and position rows");
                return false;
            }

            const int total_rows = batch_size * seq_len;
            std::vector<int32_t> token_group(
                static_cast<size_t>(mtp_sidecar_condition_token_slot_width_));
            for (int request = 0; request < batch_size; ++request)
            {
                const int real_tokens = std::clamp(
                    (*request_lengths)[static_cast<size_t>(request)],
                    0,
                    seq_len);
                const int shifted_rows = std::max(0, real_tokens - 1);
                for (int row = 0; row < shifted_rows;)
                {
                    const int grouped_rows = std::min(
                        mtp_sidecar_condition_token_slot_width_,
                        shifted_rows - row);
                    for (int grouped_row = 0; grouped_row < grouped_rows; ++grouped_row)
                    {
                        const int request_row = row + grouped_row;
                        const int flat_hidden_row = request * seq_len + request_row;
                        const int flat_token_row = flat_hidden_row + 1;
                        token_group[static_cast<size_t>(grouped_row)] =
                            static_cast<int32_t>(tokens[flat_token_row]);
                    }

                    const bool hidden_rows_ready =
                        selectMTPTerminalHiddenRows(
                            request * seq_len + row,
                            grouped_rows,
                            total_rows,
                            /*stream=*/nullptr);
                    if (!hidden_rows_ready)
                    {
                        return false;
                    }

                    const int first_token_flat_row =
                        request * seq_len + row + 1;
                    const int grouped_position =
                        (*position_ids)[static_cast<size_t>(first_token_flat_row)];
                    if (!executeMTPDepth0Batched(
                            token_group.data(),
                            grouped_rows,
                            state_.prefix_terminal_hidden.get(),
                            grouped_position,
                            "mtp_shifted_prefill_request_batch",
                            /*kv_cache_only=*/true,
                            BufferId::PREFIX_TERMINAL_HIDDEN,
                            /*defer_final_sync=*/false,
                            /*draft_condition_tokens_device=*/nullptr,
                            /*draft_condition_ready_slot=*/-1,
                            /*draft_condition_ready_is_target=*/false,
                            /*request_batch=*/1,
                            /*position_ids_override=*/nullptr,
                            /*position_ids_device_override=*/nullptr,
                            /*speculative_outcome_meta_device=*/nullptr,
                            /*speculative_outcome_meta_stride=*/0,
                            /*speculative_outcome_output_tokens_device=*/nullptr,
                            /*speculative_outcome_output_token_stride=*/0,
                            /*speculative_outcome_request_index=*/-1,
                            /*speculative_first_output_token_index=*/0,
                            /*speculative_outcome_ready_event=*/nullptr,
                            /*draft_condition_token_stride=*/1,
                            /*draft_condition_ready_count=*/1,
                            /*draft_condition_ready_stride=*/1,
                            /*device_position_offset=*/0,
                            /*first_seq_idx=*/request))
                    {
                        return false;
                    }

                    PerfStatsCollector::addCounter(
                        "mtp",
                        "shifted_prefill_request_group_payloads",
                        1.0,
                        perfPhaseName(),
                        state_.device_id.toString(),
                        {{"request", std::to_string(request)},
                         {"rows", std::to_string(grouped_rows)},
                         {"first_seq_idx", std::to_string(request)}});
                    row += grouped_rows;
                }
            }
            return true;
        }

        const auto origin=graph_builder_->config().mtpShiftedCacheOrigin();
        if(origin==MTPShiftedCacheOrigin::VisibleZeroHiddenTokenZero) {
            const int cached=state_.mtp_kv_caches[0]->get_cached_tokens(0,0);
            if(previous_tokens==0) {
                if(cached!=0)throw std::invalid_argument("QFN initial bootstrap requires a reset empty shifted cache");
                if(!publishCPUVisibleZeroHiddenMTPBootstrap())return false;
                if(!executeMTPDepth0(tokens[0],state_.prefix_terminal_hidden.get(),0,
                    "qfn_visible_zero_hidden_bootstrap",true,BufferId::PREFIX_TERMINAL_HIDDEN,false))return false;
            } else {
                if(cached!=previous_tokens)
                    throw std::invalid_argument("QFN resumed prefix must retain its visible seed and exact shifted frontier");
                if(!state_.prefix_terminal_hidden || !state_.mtp_terminal_hidden_publication.current())
                    throw std::invalid_argument("QFN resumed prefix requires its exact published prior4H hidden row before append");
            }
        }

        if (previous_tokens > 0 &&
            state_.prefix_terminal_hidden &&
            state_.mtp_terminal_hidden_publication.current())
        {
            const bool bridge_ok = executeMTPDepth0(
                tokens[0],
                state_.prefix_terminal_hidden.get(),
                previous_tokens,
                "mtp_shifted_prefill",
                true,
                BufferId::PREFIX_TERMINAL_HIDDEN,
                /*defer_final_sync=*/false);
            if (!bridge_ok)
            {
                return false;
            }
            PerfStatsCollector::addCounterWithSequence(
                "mtp",
                "shifted_prefill_segment_bridge_rows",
                1.0,
                {static_cast<uint64_t>(previous_tokens)},
                perfPhaseName(),
                state_.device_id.toString(),
                {{"segment_rows", std::to_string(seq_len)}});
        }

        std::vector<int32_t> token_batch(
            static_cast<size_t>(mtp_sidecar_condition_token_slot_width_));
        for (int row = 0; row + 1 < logical_seq_len;)
        {
            const int remaining_rows = (logical_seq_len - 1) - row;
            const int batch_rows = std::min(
                mtp_sidecar_condition_token_slot_width_,
                remaining_rows);
            const bool hidden_rows_ready =
                selectMTPTerminalHiddenRows(
                    row,
                    batch_rows,
                    seq_len,
                    /*stream=*/nullptr);
            if (!hidden_rows_ready)
            {
                return false;
            }
            for (int i = 0; i < batch_rows; ++i)
            {
                token_batch[static_cast<size_t>(i)] =
                    static_cast<int32_t>(tokens[row + 1 + i]);
            }
            if (!executeMTPDepth0Batched(
                                         token_batch.data(),
                                         batch_rows,
                                         state_.prefix_terminal_hidden.get(),
                                         position_offset + row + 1,
                                         "mtp_shifted_prefill",
                                         true,
                                         BufferId::PREFIX_TERMINAL_HIDDEN,
                                         /*defer_final_sync=*/false,
                                         /*draft_condition_tokens_device=*/nullptr,
                                         /*draft_condition_ready_slot=*/-1,
                                         /*draft_condition_ready_is_target=*/false,
                                         /*request_batch=*/1,
                                         /*position_ids_override=*/nullptr,
                                         /*position_ids_device_override=*/nullptr))
            {
                return false;
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "shifted_prefill_sidecar_batches",
                1.0,
                perfPhaseName(),
                state_.device_id.toString(),
                {{"rows", std::to_string(batch_rows)}});
            row += batch_rows;
        }

        /*
         * The grouped sidecar batches above deliberately stop at `seq_len - 1`:
         * row i predicts the shifted KV row conditioned by token i + 1.  Their
         * final row-select buffer therefore contains the last *conditionable*
         * hidden rows, not the segment's terminal hidden row.  Stable LLEP
         * prefill divides one request into multiple forward transactions.  The
         * next transaction needs exactly this terminal row to publish the pair
         * that crosses the segment boundary, so archive it explicitly after all
         * internal pairs have consumed the multi-row hidden tensor.
         */
        const bool terminal_hidden_archived = selectMTPTerminalHiddenRow(
            logical_seq_len - 1,
            seq_len,
            /*stream=*/nullptr);
        if (!terminal_hidden_archived)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to archive terminal hidden row after shifted MTP prefill segment"
                      << " position_offset=" << position_offset
                      << " seq_len=" << seq_len);
            return false;
        }
        PerfStatsCollector::addCounter(
            "mtp",
            "shifted_prefill_terminal_hidden_archives",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"position_offset", std::to_string(position_offset)},
             {"segment_rows", std::to_string(logical_seq_len)},
             {"physical_rows", std::to_string(seq_len)}});

        return true;
    }

    bool DeviceGraphOrchestrator::publishCPUVisibleZeroHiddenMTPBootstrap()
    {
        if (!state_.device_id.is_cpu() || !graph_builder_ ||
            !graph_builder_->config().mtp.enabled ||
            graph_builder_->config().mtpShiftedCacheOrigin() !=
                MTPShiftedCacheOrigin::VisibleZeroHiddenTokenZero)
            throw std::invalid_argument("Visible zero-hidden MTP bootstrap requires its original CPU model policy");
        if (!ensureMTPTerminalHiddenBuffer())
            return false;
        if (state_.prefix_terminal_hidden->native_type() != TensorType::FP32)
            throw std::invalid_argument("Visible zero-hidden MTP bootstrap requires the original FP32 mailbox");
        const auto bytes = ResidualRowGeometry(graph_builder_->config().residualWidth()).rowBytes();
        // This is the existing original arena mailbox, not normalized head
        // output or another transaction's source. Spare rows remain untouched.
        // Invalidate before the first write; publish the typed producer only
        // after TransferEngine has published the completed host mutation.
        state_.mtp_terminal_hidden_publication.invalidate();
        std::memset(state_.prefix_terminal_hidden->raw_mutable_data(), 0, bytes);
        TransferEngine::publishHostWrite(state_.prefix_terminal_hidden.get());
        state_.mtp_terminal_hidden_publication.publishVisibleZeroHiddenBootstrap();
        return true;
    }

    const float *DeviceGraphOrchestrator::logits() const
    {
        if (!state_.logits)
        {
            return nullptr;
        }
        if (state_.device_id.is_gpu())
        {
            PerfStatsCollector::addCounter(
                "sampling",
                "host_logits_access",
                1.0,
                perfPhaseName(),
                state_.device_id.toString(),
                {{"source", "device_graph_logits"},
                 {"device_valid", state_.logits->deviceValid() ? "true" : "false"},
                 {"host_valid", state_.logits->hostValid() ? "true" : "false"},
                 {"rows", std::to_string(state_.logits->rows())},
                 {"cols", std::to_string(state_.logits->cols())}});
        }
        return publishLogitsTensorToHost(
            state_.logits.get(),
            HostLogitsSurface::Main,
            "logits");
    }

    bool DeviceGraphOrchestrator::forwardMTP(int32_t draft_condition_token)
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
        {
            return false;
        }
        if (state_.mtp_kv_caches.empty() || !state_.mtp_kv_caches[0])
        {
            LOG_ERROR("[DeviceGraphOrchestrator] forwardMTP requires an initialized MTP KV cache");
            return false;
        }
        const FrozenModelWeightSet *mtp_weight_set =
            selectMTPDecodeWeightSet("forwardMTP");
        if (!mtp_weight_set)
        {
            return false;
        }

        auto bindings = makeModelWeightBindings(
            *mtp_weight_set, state_.device_id);
        if (bindings.mtp.empty() || bindings.mtp.depths.empty())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] forwardMTP requested without MTP weight bindings");
            return false;
        }

        int position_id = state_.positions.empty() ? 0 : state_.positions[0];
        TensorBase *terminal_hidden = nullptr;
        BufferId terminal_hidden_buffer_id = BufferId::HIDDEN_STATE;
        if (!resolveMTPTerminalHiddenInput(
                &terminal_hidden,
                &terminal_hidden_buffer_id,
                "forwardMTP"))
            return false;
        return executeMTPDepth0(
            draft_condition_token,
            terminal_hidden,
            position_id,
            "mtp_decode_sidecar",
            false,
            terminal_hidden_buffer_id);
    }

    bool DeviceGraphOrchestrator::forwardMTPForDeviceSampling(int32_t draft_condition_token)
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
        {
            return false;
        }
        if (state_.mtp_kv_caches.empty() || !state_.mtp_kv_caches[0])
        {
            LOG_ERROR("[DeviceGraphOrchestrator] forwardMTPForDeviceSampling requires an initialized MTP KV cache");
            return false;
        }
        const FrozenModelWeightSet *mtp_weight_set =
            selectMTPDecodeWeightSet("forwardMTPForDeviceSampling");
        if (!mtp_weight_set)
        {
            return false;
        }

        auto bindings = makeModelWeightBindings(
            *mtp_weight_set, state_.device_id);
        if (bindings.mtp.empty() || bindings.mtp.depths.empty())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] forwardMTPForDeviceSampling requested without MTP weight bindings");
            return false;
        }

        const int position_id = state_.positions.empty() ? 0 : state_.positions[0];
        TensorBase *terminal_hidden = nullptr;
        BufferId terminal_hidden_buffer_id = BufferId::HIDDEN_STATE;
        if (!resolveMTPTerminalHiddenInput(
                &terminal_hidden,
                &terminal_hidden_buffer_id,
                "forwardMTPForDeviceSampling"))
            return false;

        /*
         * Request a deferred final sync. If graph capture, stage timing, or
         * collectives make that impossible, executeMTPDepth0 synchronizes before
         * returning and leaves no pending stream, preserving correctness.
         */
        return executeMTPDepth0(
            draft_condition_token,
            terminal_hidden,
            position_id,
            "mtp_decode_sidecar",
            false,
            terminal_hidden_buffer_id,
            /*defer_final_sync=*/true);
    }

    bool DeviceGraphOrchestrator::forwardMTPFromLastDraft(int32_t draft_condition_token, int position_id)
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
        {
            return false;
        }
        if (position_id < 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Chained MTP sidecar requires a non-negative position_id");
            return false;
        }

        auto it = state_.extension_buffers.find(BufferId::MTP_HIDDEN);
        TensorBase *mtp_hidden =
            (it == state_.extension_buffers.end() || !it->second) ? nullptr : it->second.get();
        if (!mtp_hidden)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Chained MTP sidecar requires previous MTP hidden state");
            return false;
        }

        return executeMTPDepth0(
            draft_condition_token,
            mtp_hidden,
            position_id,
            "mtp_decode_sidecar_chain",
            false,
            BufferId::MTP_HIDDEN);
    }

    bool DeviceGraphOrchestrator::forwardMTPFromLastDraftForDeviceSampling(
        int32_t draft_condition_token,
        int position_id)
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
        {
            return false;
        }
        if (position_id < 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Chained MTP sidecar device-sampling path requires a non-negative position_id");
            return false;
        }

        auto it = state_.extension_buffers.find(BufferId::MTP_HIDDEN);
        TensorBase *mtp_hidden =
            (it == state_.extension_buffers.end() || !it->second) ? nullptr : it->second.get();
        if (!mtp_hidden)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Chained MTP sidecar device-sampling path requires previous MTP hidden state");
            return false;
        }

        return executeMTPDepth0(
            draft_condition_token,
            mtp_hidden,
            position_id,
            "mtp_decode_sidecar_chain",
            false,
            BufferId::MTP_HIDDEN,
            /*defer_final_sync=*/true);
    }

    bool DeviceGraphOrchestrator::forwardMTPFromDeviceDraftAtLivePositionForDeviceSampling(
        int draft_sample_slot,
        int position_offset)
    {
        if (!supportsMTPDeviceDraftTokenInput())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-token chained MTP sidecar is not supported by this runner");
            return false;
        }
        if (draft_sample_slot < 0 ||
            draft_sample_slot >= stochastic_draft_row_capacity_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-token chained MTP sidecar received invalid draft slot="
                      << draft_sample_slot);
            return false;
        }
        if (position_offset < 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-token chained MTP sidecar requires a non-negative live-position offset");
            return false;
        }

        auto it = state_.extension_buffers.find(BufferId::MTP_HIDDEN);
        TensorBase *mtp_hidden =
            (it == state_.extension_buffers.end() || !it->second) ? nullptr : it->second.get();
        if (!mtp_hidden)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-token chained MTP sidecar requires previous MTP hidden state");
            return false;
        }
        if (!state_.kv_cache ||
            !state_.kv_cache->supportsDeviceResidentSequenceStatePublication())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-token chained MTP sidecar requires a device-owned live KV position");
            return false;
        }
        const int32_t *live_position_device =
            state_.kv_cache->deviceSequenceCachedTokenCountPtr(/*seq_idx=*/0);
        if (!live_position_device)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-token chained MTP sidecar could not resolve the canonical live KV count");
            return false;
        }

        const int *draft_token_device =
            static_cast<const int *>(stochastic_draft_sample_tokens_dev_) + draft_sample_slot;

        /*
         * The draft-slot readiness event is recorded after the previous
         * sidecar and sampler. Waiting on it therefore orders both the token
         * and the earlier main-cache count before the dynamic-input composer.
         * A non-zero depth offset is added on device into MTP_POSITION_IDS, so
         * captured replay retains only stable arena/cache pointers.
         */
        return executeMTPDepth0Batched(
            /*draft_condition_tokens=*/nullptr,
            /*token_count=*/1,
            mtp_hidden,
            /*position_id=*/0,
            "mtp_decode_sidecar_chain_device_token_live_position",
            /*kv_cache_only=*/false,
            BufferId::MTP_HIDDEN,
            /*defer_final_sync=*/true,
            draft_token_device,
            draft_sample_slot,
            /*draft_condition_ready_is_target=*/false,
            /*request_batch=*/1,
            /*position_ids_override=*/nullptr,
            live_position_device,
            /*speculative_outcome_meta_device=*/nullptr,
            /*speculative_outcome_meta_stride=*/0,
            /*speculative_outcome_output_tokens_device=*/nullptr,
            /*speculative_outcome_output_token_stride=*/0,
            /*speculative_outcome_request_index=*/-1,
            /*speculative_first_output_token_index=*/0,
            /*speculative_outcome_ready_event=*/nullptr,
            /*draft_condition_token_stride=*/1,
            /*draft_condition_ready_count=*/1,
            /*draft_condition_ready_stride=*/1,
            position_offset);
    }

    bool DeviceGraphOrchestrator::forwardMTPFromDeviceTargetAtLivePositionForDeviceSampling(
        int target_sample_slot)
    {
        if (!supportsMTPDeviceDraftTokenInput())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-token first MTP sidecar is not supported by this runner");
            return false;
        }
        if (target_sample_slot < 0 ||
            target_sample_slot >= stochastic_target_row_capacity_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-token first MTP sidecar received invalid target slot="
                      << target_sample_slot);
            return false;
        }
        if (!stochastic_target_sample_tokens_dev_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-token first MTP sidecar requires STOCHASTIC_TARGET_SAMPLE_TOKENS");
            return false;
        }
        if (!state_.kv_cache ||
            !state_.kv_cache->supportsDeviceResidentSequenceStatePublication())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-token first MTP sidecar requires a device-owned live KV position");
            return false;
        }
        const int32_t *live_position_device =
            state_.kv_cache->deviceSequenceCachedTokenCountPtr(/*seq_idx=*/0);
        if (!live_position_device)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-token first MTP sidecar could not resolve the canonical live KV count");
            return false;
        }

        TensorBase *terminal_hidden = nullptr;
        BufferId terminal_hidden_buffer_id = BufferId::HIDDEN_STATE;
        if (!resolveMTPTerminalHiddenInput(
                &terminal_hidden,
                &terminal_hidden_buffer_id,
                "forwardMTPFromDeviceTargetAtLivePositionForDeviceSampling"))
            return false;

        const int *target_token_device =
            static_cast<const int *>(stochastic_target_sample_tokens_dev_) +
            target_sample_slot;

        /*
         * This is the scalar counterpart to
         * advanceMTPRequestBatchConditionOnDevice(). The sidecar itself can
         * consume the target slot and canonical KV count directly, but the
         * request-lifetime mailbox must advance at the same ownership boundary
         * or diagnostics and the next resident transaction retain the
         * pre-condition position. Publication is enqueued before the sidecar;
         * each consumer waits on the event that owns the rows it actually uses.
         */
        if (!publishDeviceResidentLogicalSequenceStateFromTargetSample(
                target_sample_slot,
                live_position_device))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-token first MTP sidecar could not publish its resident logical-state boundary");
            return false;
        }

        /*
         * Target sampling runs after the condition graph on the main-logits
         * stream. Its readiness event therefore fences the canonical device KV
         * count as well as the sampled token before the sidecar stream reads
         * either value. No host position mirror can become stale here.
         */
        return executeMTPDepth0Batched(
            /*draft_condition_tokens=*/nullptr,
            /*token_count=*/1,
            terminal_hidden,
            /*position_id=*/0,
            "mtp_decode_sidecar_device_target_token_live_position",
            /*kv_cache_only=*/false,
            terminal_hidden_buffer_id,
            /*defer_final_sync=*/true,
            target_token_device,
            target_sample_slot,
            /*draft_condition_ready_is_target=*/true,
            /*request_batch=*/1,
            /*position_ids_override=*/nullptr,
            live_position_device);
    }

    bool DeviceGraphOrchestrator::forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(
        const DeviceResidentLogicalSequenceStateHandle &logical_state,
        int request_index)
    {
        if (!supportsMTPDeviceDraftTokenInput())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident logical-state MTP sidecar is not supported by this runner");
            return false;
        }
        if (!logical_state.valid())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident logical-state MTP sidecar received an invalid handle");
            return false;
        }
        if (logical_state.device != state_.device_id)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident logical-state MTP sidecar device mismatch: handle="
                      << logical_state.device.toString()
                      << " runner=" << state_.device_id.toString());
            return false;
        }
        if (logical_state.live_state_epoch != live_replay_state_epoch_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident logical-state MTP sidecar received a stale live-state epoch");
            return false;
        }
        if (!logical_state.coversRequest(request_index))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident logical-state MTP sidecar received request_index="
                      << request_index << " for request_count="
                      << logical_state.request_count);
            return false;
        }

        const auto &mailbox = device_resident_logical_sequence_state_mailbox_;
        if (!mailbox.ownsHandle(logical_state, live_replay_state_epoch_))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident logical-state MTP sidecar refuses a non-owned mailbox handle");
            return false;
        }

        TensorBase *terminal_hidden = nullptr;
        BufferId terminal_hidden_buffer_id = BufferId::HIDDEN_STATE;
        if (!resolveMTPTerminalHiddenInput(
                &terminal_hidden,
                &terminal_hidden_buffer_id,
                "forwardMTPFromDeviceResidentLogicalStateForDeviceSampling"))
        {
            return false;
        }

        /*
         * These pointers name one row in the runner-owned mailbox.  The sidecar
         * executor will wait on the mailbox event after selecting its explicit
         * stream, then copy the token into the persistent condition-token slot
         * and bind the position row directly to device-aware RoPE stages.
         */
        const int32_t *next_condition_token_device =
            logical_state.nextConditionTokenDeviceForRequest(request_index);
        const int32_t *target_position_device =
            logical_state.targetPositionDeviceForRequest(request_index);
        if (!next_condition_token_device || !target_position_device)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident logical-state MTP sidecar could not derive request-local device rows");
            return false;
        }
        return executeMTPDepth0Batched(
            /*draft_condition_tokens=*/nullptr,
            /*token_count=*/1,
            terminal_hidden,
            /*position_id=*/0,
            "mtp_decode_sidecar_resident_logical_state",
            /*kv_cache_only=*/false,
            terminal_hidden_buffer_id,
            /*defer_final_sync=*/true,
            next_condition_token_device,
            /*draft_condition_ready_slot=*/-1,
            /*draft_condition_ready_is_target=*/false,
            /*request_batch=*/1,
            /*position_ids_override=*/nullptr,
            target_position_device);
    }

    bool DeviceGraphOrchestrator::forwardMTPAndSampleGreedy(
        int32_t draft_condition_token,
        int32_t *out_token)
    {
        if (!out_token)
            return false;
        *out_token = -1;
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
        {
            return false;
        }
        if (state_.mtp_kv_caches.empty() || !state_.mtp_kv_caches[0])
        {
            LOG_ERROR("[DeviceGraphOrchestrator] forwardMTPAndSampleGreedy requires an initialized MTP KV cache");
            return false;
        }
        if (!frozen_weight_set_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] forwardMTPAndSampleGreedy requires frozen MTP weight bindings");
            return false;
        }

        int position_id = state_.positions.empty() ? 0 : state_.positions[0];
        TensorBase *terminal_hidden = nullptr;
        BufferId terminal_hidden_buffer_id = BufferId::HIDDEN_STATE;
        if (!resolveMTPTerminalHiddenInput(
                &terminal_hidden,
                &terminal_hidden_buffer_id,
                "forwardMTPAndSampleGreedy"))
            return false;

        if (!executeMTPDepth0(
                draft_condition_token,
                terminal_hidden,
                position_id,
                "mtp_decode_sidecar",
                false,
                terminal_hidden_buffer_id,
                /*defer_final_sync=*/true))
        {
            return false;
        }

        const int token = sampleGreedyFromMTPLogitsOnDevice();
        if (token < 0)
            return false;
        *out_token = token;
        return true;
    }

    bool DeviceGraphOrchestrator::forwardMTPAndSampleGreedyToDeviceDraftSlot(
        int32_t draft_condition_token,
        int draft_sample_slot,
        int32_t *out_token)
    {
        if (out_token)
            *out_token = -1;
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
        {
            return false;
        }
        if (state_.mtp_kv_caches.empty() || !state_.mtp_kv_caches[0])
        {
            LOG_ERROR("[DeviceGraphOrchestrator] forwardMTPAndSampleGreedyToDeviceDraftSlot requires an initialized MTP KV cache");
            return false;
        }
        if (!frozen_weight_set_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] forwardMTPAndSampleGreedyToDeviceDraftSlot requires frozen MTP weight bindings");
            return false;
        }

        int position_id = state_.positions.empty() ? 0 : state_.positions[0];
        TensorBase *terminal_hidden = nullptr;
        BufferId terminal_hidden_buffer_id = BufferId::HIDDEN_STATE;
        if (!resolveMTPTerminalHiddenInput(
                &terminal_hidden,
                &terminal_hidden_buffer_id,
                "forwardMTPAndSampleGreedyToDeviceDraftSlot"))
            return false;

        if (!executeMTPDepth0(
                draft_condition_token,
                terminal_hidden,
                position_id,
                "mtp_decode_sidecar",
                false,
                terminal_hidden_buffer_id,
                /*defer_final_sync=*/true))
        {
            return false;
        }

        return sampleGreedyFromMTPLogitsToDeviceDraftSlot(
            draft_sample_slot,
            out_token);
    }

    bool DeviceGraphOrchestrator::forwardMTPBatchAndSampleGreedy(
        const int32_t *draft_condition_tokens,
        const int *position_ids,
        int request_batch,
        int32_t *out_tokens)
    {
        if (!draft_condition_tokens || !position_ids || !out_tokens ||
            request_batch <= 1 ||
            request_batch > stochastic_batch_output_request_capacity_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Batched MTP greedy sidecar exceeds configured request capacity: requests="
                      << request_batch << " capacity="
                      << stochastic_batch_output_request_capacity_);
            return false;
        }
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
            return false;
        if (state_.mtp_kv_caches.empty() || !state_.mtp_kv_caches[0])
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Batched MTP greedy sidecar requires an initialized MTP KV cache");
            return false;
        }
        if (static_cast<int>(state_.positions.size()) < request_batch)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Batched MTP greedy sidecar requires per-request live positions");
            return false;
        }

        TensorBase *terminal_hidden = nullptr;
        BufferId terminal_hidden_buffer_id = BufferId::HIDDEN_STATE;
        if (!resolveMTPTerminalHiddenInput(
                &terminal_hidden,
                &terminal_hidden_buffer_id,
                "forwardMTPBatchAndSampleGreedy"))
        {
            return false;
        }

        if (!executeMTPDepth0Batched(
                draft_condition_tokens,
                /*token_count=*/1,
                terminal_hidden,
                position_ids[0],
                "mtp_decode_sidecar_request_batch",
                /*kv_cache_only=*/false,
                terminal_hidden_buffer_id,
                /*defer_final_sync=*/true,
                /*draft_condition_tokens_device=*/nullptr,
                /*draft_condition_ready_slot=*/-1,
                /*draft_condition_ready_is_target=*/false,
                request_batch,
                position_ids))
        {
            return false;
        }

        auto it = state_.extension_buffers.find(mtpSamplingLogitsBufferId());
        TensorBase *mtp_logits =
            (it == state_.extension_buffers.end() || !it->second)
                ? nullptr
                : it->second.get();
        if (!mtp_logits)
            return false;

        const auto &shape = mtp_logits->shape();
        if (shape.empty())
            return false;
        const size_t rows = shape.size() >= 2 ? shape[0] : 1;
        const size_t cols = shape.size() >= 2 ? shape[1] : shape[0];
        if (cols == 0 ||
            rows < static_cast<size_t>(request_batch) ||
            cols > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            return false;
        }

        const int token_offset = mtpSidecarLogitsAreColumnParallel() && graph_builder_
                                     ? vocabOffsetForTPConfig(graph_builder_->config())
                                     : 0;
        auto device_opt = mtp_logits->current_device();
        if (device_opt.has_value() && device_opt->is_gpu() && mtp_logits->deviceValid())
        {
            IBackend *backend = getBackendFor(*device_opt);
            const void *gpu_ptr = mtp_logits->gpu_data_ptr();
            void *stream = consumePendingLogitsStream(
                PendingLogitsStreamRole::MTPSidecar,
                "forwardMTPBatchAndSampleGreedy");
            if (!stream)
                stream = explicitGPUStreamForOperation("forwardMTPBatchAndSampleGreedy");
            if (!backend || !gpu_ptr || !stream)
                return false;

            std::vector<float> values(static_cast<size_t>(request_batch));
            std::vector<int> indices(static_cast<size_t>(request_batch));
            if (!backend->argmaxF32BatchedRows(
                    gpu_ptr,
                    request_batch,
                    static_cast<int>(cols),
                    device_opt->gpu_ordinal(),
                    values.data(),
                    indices.data(),
                    stream,
                    argmax_partial_vals_dev_,
                    argmax_partial_idxs_dev_,
                    argmax_partial_capacity_))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Batched MTP greedy argmax failed");
                return false;
            }

            for (int row = 0; row < request_batch; ++row)
            {
                if (indices[static_cast<size_t>(row)] < 0)
                    return false;
                out_tokens[row] =
                    static_cast<int32_t>(token_offset + indices[static_cast<size_t>(row)]);
            }
            return true;
        }

        for (int row = 0; row < request_batch; ++row)
        {
            const int token = coordinateGreedyCandidate(
                sampleGreedyCandidateFromTensor(
                    mtp_logits,
                    row,
                    token_offset,
                    argmax_partial_vals_dev_,
                    argmax_partial_idxs_dev_,
                    argmax_partial_capacity_,
                    /*stream=*/nullptr,
                    "mtp_logits_request_batch"),
                globalTPContextForMTPCoordination());
            if (token < 0)
                return false;
            out_tokens[row] = static_cast<int32_t>(token);
        }
        return true;
    }

    bool DeviceGraphOrchestrator::sampleMTPBatchGreedyLogitsToDeviceDraftSlots(
        int request_batch,
        int first_draft_slot,
        int slot_stride,
        const char *context,
        int32_t *out_tokens)
    {
        const int last_draft_slot =
            first_draft_slot +
            (request_batch > 0 ? (request_batch - 1) * slot_stride : 0);
        if (!supportsDeviceStochasticMTPVerification() ||
            !state_.device_id.is_gpu() ||
            request_batch <= 0 ||
            request_batch > stochastic_batch_output_request_capacity_ ||
            first_draft_slot < 0 ||
            slot_stride <= 0 ||
            last_draft_slot >= stochastic_draft_row_capacity_ ||
            !stochastic_draft_sample_tokens_dev_ ||
            !stochastic_draft_sample_probs_dev_ ||
            !argmax_partial_vals_dev_ ||
            !argmax_partial_idxs_dev_ ||
            argmax_partial_capacity_ <= 0)
        {
            return false;
        }

        auto logits_it = state_.extension_buffers.find(mtpSamplingLogitsBufferId());
        TensorBase *mtp_logits =
            (logits_it == state_.extension_buffers.end() || !logits_it->second)
                ? nullptr
                : logits_it->second.get();
        if (!mtp_logits || !mtp_logits->deviceValid() || !mtp_logits->gpu_data_ptr())
            return false;

        const auto &shape = mtp_logits->shape();
        if (shape.empty())
            return false;
        const size_t rows = shape.size() >= 2 ? shape[0] : 1;
        const size_t cols = shape.size() >= 2 ? shape[1] : shape[0];
        if (cols == 0 ||
            rows < static_cast<size_t>(request_batch) ||
            cols > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            return false;
        }

        const int token_offset = mtpSidecarLogitsAreColumnParallel() && graph_builder_
                                     ? vocabOffsetForTPConfig(graph_builder_->config())
                                     : 0;
        if (token_offset != 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Batched MTP device draft-slot sampling currently requires unsharded logits");
            return false;
        }

        auto device_opt = mtp_logits->current_device();
        if (!device_opt.has_value() ||
            !device_opt->is_gpu() ||
            *device_opt != state_.device_id)
        {
            return false;
        }

        IBackend *backend = getBackendFor(*device_opt);
        void *stream = consumePendingLogitsStream(
            PendingLogitsStreamRole::MTPSidecar,
            context ? context : "sampleMTPBatchGreedyLogitsToDeviceDraftSlots");
        if (!stream)
        {
            stream = explicitGPUStreamForOperation(
                context ? context : "sampleMTPBatchGreedyLogitsToDeviceDraftSlots");
        }
        if (!backend || !stream)
            return false;

        for (int row = 0; row < request_batch; ++row)
            clearStochasticDraftSampleReadySlot(
                first_draft_slot + row * slot_stride,
                StochasticSampleReadyClearMode::Force);

        auto *out_values_dev =
            static_cast<float *>(stochastic_draft_sample_probs_dev_) +
            first_draft_slot;
        auto *out_tokens_dev =
            static_cast<int32_t *>(stochastic_draft_sample_tokens_dev_) +
            first_draft_slot;

        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "request_batch_sidecar_device_draft_argmax_enqueue",
                "decode",
                state_.device_id.toString(),
                {{"requests", std::to_string(request_batch)},
                 {"first_slot", std::to_string(first_draft_slot)},
                 {"slot_stride", std::to_string(slot_stride)}});
            if (!backend->enqueueArgmaxF32BatchedRowsDevice(
                    mtp_logits->gpu_data_ptr(),
                    request_batch,
                    static_cast<int>(cols),
                    state_.device_id.gpu_ordinal(),
                    stream,
                    out_values_dev,
                    out_tokens_dev,
                    argmax_partial_vals_dev_,
                    argmax_partial_idxs_dev_,
                    argmax_partial_capacity_,
                    slot_stride))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Batched MTP device draft-slot argmax failed");
                return false;
            }
        }

        for (int row = 0; row < request_batch; ++row)
        {
            if (!recordStochasticDraftSampleReady(
                    first_draft_slot + row * slot_stride,
                    stream,
                    /*verifier_consumer_pending=*/true))
                return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "request_batch_sidecar_device_draft_slots",
            static_cast<double>(request_batch),
            "decode",
            state_.device_id.toString(),
            {{"first_slot", std::to_string(first_draft_slot)},
             {"slot_stride", std::to_string(slot_stride)},
             {"host_shadow", out_tokens ? "true" : "false"}});

        /*
         * Publication diagnostics are intentionally sampled from the same
         * arena slots that the verifier consumes.  Reading MTP_LOGITS here
         * would only prove what the argmax should have published; reading the
         * slots proves what crossed the actual sidecar/verifier boundary.
         *
         * The diagnostic shadow is opt-in because deviceToHostFast() drains
         * the producer stream.  Normal serving therefore retains the fully
         * asynchronous device-owned path, while a failing parity run can show
         * every request-major proposal token without changing production
         * ownership or introducing a second sampling implementation.
         */
        std::vector<int32_t> diagnostic_tokens;
        int32_t *host_shadow = out_tokens;
        if (!host_shadow && debugEnv().runtime_debug.mtp_publication_diagnostics)
        {
            diagnostic_tokens.assign(static_cast<size_t>(request_batch), -1);
            host_shadow = diagnostic_tokens.data();
        }

        if (!host_shadow)
            return true;

        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "request_batch_sidecar_device_draft_shadow_d2h_sync",
                "decode",
                state_.device_id.toString(),
                {{"requests", std::to_string(request_batch)},
                 {"first_slot", std::to_string(first_draft_slot)},
                 {"slot_stride", std::to_string(slot_stride)}});
            const size_t shadow_span =
                static_cast<size_t>(last_draft_slot - first_draft_slot + 1);
            std::vector<int32_t> shadow_slots(shadow_span, -1);
            if (!backend->deviceToHostFast(
                    shadow_slots.data(),
                    out_tokens_dev,
                    sizeof(int32_t) * shadow_span,
                    state_.device_id.gpu_ordinal(),
                    stream))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Batched MTP device draft-slot shadow copy failed");
                return false;
            }
            for (int row = 0; row < request_batch; ++row)
            {
                host_shadow[row] =
                    shadow_slots[static_cast<size_t>(row * slot_stride)];
            }
        }

        for (int row = 0; row < request_batch; ++row)
        {
            if (host_shadow[row] < 0)
                return false;
        }

        if (debugEnv().runtime_debug.mtp_publication_diagnostics)
        {
            std::ostringstream token_list;
            token_list << '[';
            for (int row = 0; row < request_batch; ++row)
            {
                if (row > 0)
                    token_list << ',';
                token_list << host_shadow[row];
            }
            token_list << ']';
            LOG_INFO(
                "[MTPPublicationDiagnostics] phase=request_batch_sidecar_draft_publication"
                << " context=" << (context ? context : "<unnamed>")
                << " requests=" << request_batch
                << " first_slot=" << first_draft_slot
                << " slot_stride=" << slot_stride
                << " draft_tokens=" << token_list.str());
        }
        return true;
    }

    bool DeviceGraphOrchestrator::advanceMTPRequestBatchConditionOnDevice(
        const DeviceResidentLogicalSequenceStateHandle &logical_state,
        int request_batch,
        const SamplingParams &params,
        const uint64_t *stochastic_position_seeds)
    {
        ScopedDeviceLog device_log(state_.device_id);

        auto fail = [&](const std::string &reason) -> bool
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident request-batch condition advance failed: "
                      << reason);
            return false;
        };

        if (!state_.device_id.is_gpu())
            return fail("operation requires a GPU runner");
        if (request_batch <= 0 ||
            request_batch > stochastic_batch_output_request_capacity_)
        {
            return fail(
                "request count exceeds configured capacity " +
                std::to_string(stochastic_batch_output_request_capacity_));
        }
        if (!logical_state.coversRequest(request_batch - 1) ||
            logical_state.device != state_.device_id ||
            !device_resident_logical_sequence_state_mailbox_.ownsHandle(
                logical_state,
                live_replay_state_epoch_))
        {
            return fail("logical-state mailbox is stale, foreign, or undersized");
        }
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
            return fail("MTP is not enabled for this graph");
        const auto &graph_config = graph_builder_->config();
        const bool has_full_vocabulary_main_head =
            !graph_config.lm_head_column_parallel ||
            publishesFullVocabularyMTPLogits();
        if (!has_full_vocabulary_main_head)
        {
            return fail(
                "condition sampling requires a SingleDevice full-vocabulary "
                "main head or a mirrored LocalTP full-vocabulary main head");
        }
        if (!params.is_greedy())
        {
            if (params.top_k <= 0 ||
                params.top_k > static_cast<int>(kStochasticDistributionMaxK))
            {
                return fail("stochastic condition sampling requires 1 <= top_k <= 256");
            }
            if (!stochastic_position_seeds)
                return fail("stochastic condition sampling is missing request seeds");
            for (int request = 0; request < request_batch; ++request)
            {
                if (stochastic_position_seeds[request] == 0)
                    return fail("stochastic condition sampling received a zero request seed");
            }
        }

        IKVCache *shifted_cache =
            state_.mtp_kv_caches.empty() ? nullptr : state_.mtp_kv_caches[0].get();
        if (!shifted_cache)
            return fail("shifted MTP KV cache is not initialized");

        void *stream = explicitGPUStreamForOperation(
            "advanceMTPRequestBatchConditionOnDevice");
        if (!stream)
            return fail("no explicit GPU stream is available");
        if (!waitForPendingShiftedMTPKVReadyForObservation(
                stream,
                "request_batch_condition_shifted_kv"))
        {
            return false;
        }

        /*
         * Scalar decode repairs the shifted sidecar cache immediately before
         * consuming the last emitted token in the main graph. Do the same work
         * once for the complete request batch. prepareShiftedMTPKVCommitBoundary()
         * validates each independent sequence head; executeMTPDepth0Batched()
         * then launches one grouped sidecar graph rather than replaying rows.
         */
        for (int request = 0; request < request_batch; ++request)
        {
            if (!prepareShiftedMTPKVCommitBoundary(
                    *shifted_cache,
                    /*depth=*/0,
                    request,
                    /*expected_cached_tokens=*/-1,
                    /*allow_speculative_discard=*/true,
                    stream,
                    &logical_state.mtp_transaction,
                    "request_batch_condition_shifted_commit"))
            {
                return fail(
                    "shifted-cache boundary validation failed for request " +
                    std::to_string(request));
            }
        }

        TensorBase *terminal_hidden = nullptr;
        BufferId terminal_hidden_buffer_id = BufferId::HIDDEN_STATE;
        if (!resolveMTPTerminalHiddenInput(
                &terminal_hidden,
                &terminal_hidden_buffer_id,
                "advanceMTPRequestBatchConditionOnDevice"))
        {
            return false;
        }
        if (debugEnv().runtime_debug.mtp_condition_graph_contract_trace)
        {
            LOG_INFO(
                "[MTPConditionGraphContract] event=shifted_sidecar_enter"
                << " device=" << state_.device_id.toString()
                << " requests=" << request_batch);
        }
        if (!executeMTPDepth0Batched(
                /*draft_condition_tokens=*/nullptr,
                /*token_count=*/1,
                terminal_hidden,
                /*position_id=*/0,
                "mtp_request_batch_condition_shifted_commit",
                /*kv_cache_only=*/true,
                terminal_hidden_buffer_id,
                /*defer_final_sync=*/true,
                logical_state.next_condition_tokens_device,
                /*draft_condition_ready_slot=*/-1,
                /*draft_condition_ready_is_target=*/false,
                request_batch,
                /*position_ids_override=*/nullptr,
                logical_state.target_positions_device,
                /*speculative_outcome_meta_device=*/nullptr,
                /*speculative_outcome_meta_stride=*/0,
                /*speculative_outcome_output_tokens_device=*/nullptr,
                /*speculative_outcome_output_token_stride=*/0,
                /*speculative_outcome_request_index=*/-1,
                /*speculative_first_output_token_index=*/0,
                /*speculative_outcome_ready_event=*/nullptr,
                /*draft_condition_token_stride=*/1,
                /*draft_condition_ready_count=*/request_batch,
                /*draft_condition_ready_stride=*/1,
                /*device_position_offset=*/0,
                /*first_seq_idx=*/0))
        {
            return fail("grouped shifted-MTP condition append failed");
        }
        if (debugEnv().runtime_debug.mtp_condition_graph_contract_trace)
        {
            LOG_INFO(
                "[MTPConditionGraphContract] event=shifted_sidecar_leave"
                << " device=" << state_.device_id.toString()
                << " requests=" << request_batch);
        }

        /*
         * The main graph consumes the old mailbox directly. Token, position,
         * and sequence-length host pointers are null; embedding and every
         * dynamic position consumer bind to stable device rows. Forcing decode
         * semantics is essential: batch_size > 1 must not turn this one-token
         * continuation into padded prefill.
         */
        setMTPMainDecodeSyncDeferralEnabled(true);
        const float *main_logits = nullptr;
        if (debugEnv().runtime_debug.mtp_condition_graph_contract_trace)
        {
            LOG_INFO(
                "[MTPConditionGraphContract] event=main_condition_enter"
                << " device=" << state_.device_id.toString()
                << " requests=" << request_batch);
        }
        {
            /*
             * SnapshotCapture normally publishes bare semantic aliases, so a
             * later sidecar or verifier graph replaces the condition graph's
             * diagnostics.  Retain an integration-only namespaced copy of
             * every enabled stage here.  The callback still records device to
             * device snapshots at graph execution time; it never inserts a
             * host synchronization into the production transaction.
             */
            ScopedStringOverride snapshot_scope(
                snapshot_context_,
                "MTP_REQUEST_BATCH_CONDITION");
            main_logits = forwardImpl(
                /*tokens=*/nullptr,
                logical_state.next_condition_tokens_device,
                /*seq_len=*/1,
                request_batch,
                ForwardExecutionRole::MTPCondition,
                ForwardInvocationKind::ExplicitDecode,
                logical_state.target_positions_device,
                logical_state.target_sequence_lengths_device).value_or(nullptr);
        }
        if (!main_logits)
        {
            setMTPMainDecodeSyncDeferralEnabled(false);
            return fail("grouped main condition forward failed");
        }
        if (debugEnv().runtime_debug.mtp_condition_graph_contract_trace)
        {
            LOG_INFO(
                "[MTPConditionGraphContract] event=main_condition_leave"
                << " device=" << state_.device_id.toString()
                << " requests=" << request_batch);
        }

        TensorBase *logits_tensor = state_.all_position_logits.get();
        if (!logits_tensor || !logits_tensor->deviceValid() ||
            !logits_tensor->gpu_data_ptr())
        {
            return fail("grouped main condition forward produced no device logits");
        }
        const auto &logits_shape = logits_tensor->shape();
        if (logits_shape.empty())
            return fail("grouped main condition logits have no shape");
        const size_t logits_rows = logits_shape.size() >= 2 ? logits_shape[0] : 1;
        const size_t logits_cols = logits_shape.size() >= 2
                                       ? logits_shape[1]
                                       : logits_shape[0];
        if (logits_rows < static_cast<size_t>(request_batch) ||
            logits_cols == 0 ||
            logits_cols > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            return fail("grouped main condition logits do not cover the request batch");
        }

        stream = prepareMainLogitsDeviceConsumer(
            "request_batch_condition_target_sampling",
            MainLogitsHandoffMode::Observe);
        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend || !stream)
            return fail("condition sampler could not resolve its backend stream");
        if (!stochastic_target_sample_tokens_dev_ ||
            !stochastic_verify_accept_probs_dev_ ||
            !argmax_partial_vals_dev_ ||
            !argmax_partial_idxs_dev_ ||
            argmax_partial_capacity_ <= 0 ||
            request_batch > stochastic_target_row_capacity_)
        {
            return fail("condition sampler is missing persistent target scratch");
        }

        for (int request = 0; request < request_batch; ++request)
        {
            clearStochasticTargetSampleReadySlot(
                request,
                StochasticSampleReadyClearMode::Force);
        }

        if (params.is_greedy())
        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "request_batch_condition_target_argmax_enqueue",
                "decode",
                state_.device_id.toString(),
                {{"requests", std::to_string(request_batch)},
                 {"vocab", std::to_string(logits_cols)}});
            if (!backend->enqueueArgmaxF32BatchedRowsDevice(
                    logits_tensor->gpu_data_ptr(),
                    request_batch,
                    static_cast<int>(logits_cols),
                    state_.device_id.gpu_ordinal(),
                    stream,
                    stochastic_verify_accept_probs_dev_,
                    stochastic_target_sample_tokens_dev_,
                    argmax_partial_vals_dev_,
                    argmax_partial_idxs_dev_,
                    argmax_partial_capacity_))
            {
                return fail("grouped target argmax launch failed");
            }
            for (int request = 0; request < request_batch; ++request)
            {
                if (!recordStochasticTargetSampleReady(
                        request,
                        stream,
                        /*verifier_consumer_pending=*/true, StochasticTargetPublication::NewValue))
                {
                    return false;
                }
            }
        }
        else
        {
            if (!buildStochasticDistributionsOnDevice(
                    DeviceLogitsSource::MainRequestBatch,
                    /*first_row=*/0,
                    DeviceDistributionBuffer::Target,
                    /*first_slot=*/0,
                    request_batch,
                    params,
                    static_cast<int>(logits_cols)))
            {
                return fail("grouped target distribution build failed");
            }
            for (int request = 0; request < request_batch; ++request)
            {
                if (!sampleStochasticDistributionOnDeviceImpl(
                        DeviceDistributionBuffer::Target,
                        request,
                        /*threshold=*/0.0f,
                        /*out_token_host=*/nullptr,
                        stochastic_position_seeds[request],
                        logical_state.targetPositionDeviceForRequest(request),
                        /*threshold_position_offset=*/1))
                {
                    return fail(
                        "resident target sampling failed for request " +
                        std::to_string(request));
                }
            }
        }

        /*
         * Advance logical positions and publish sampled first tokens as one
         * mailbox mutation. enqueuePrepareMTPBatchedSidecarInputs() performs
         * the position + 1 transform on device; enqueueInitializeGenerationLogicalState()
         * then fills every mailbox field from those authoritative rows. Both
         * launches are allocation-free and graph-capturable.
         */
        if (!mtp_sidecar_condition_token_dev_ ||
            mtp_sidecar_condition_token_capacity_ < request_batch ||
            !mtp_sidecar_position_ids_dev_)
        {
            return fail("condition advance is missing persistent mailbox composition scratch");
        }
        {
            DeviceResidentLogicalStateReadScope logical_state_read(
                *this,
                stream,
                "request_batch_condition_publication_input");
            if (!logical_state_read.ready())
                return fail("logical-state publication input wait failed");
            if (!backend->enqueuePrepareMTPBatchedSidecarInputs(
                    stochastic_target_sample_tokens_dev_,
                    /*condition_token_stride=*/1,
                    logical_state.target_positions_device,
                    /*position_offset=*/1,
                    request_batch,
                    state_.device_id.gpu_ordinal(),
                    stream,
                    mtp_sidecar_condition_token_dev_,
                    mtp_sidecar_position_ids_dev_))
            {
                return fail("device logical-position advance launch failed");
            }
            if (!logical_state_read.complete())
                return fail("logical-state publication input completion failed");
        }

        if (!waitForDeviceResidentLogicalSequenceStateRowReuse(
                stream,
                "request_batch_condition_logical_state_row_reuse"))
        {
            return fail("advanced logical-state mailbox could not order reusable rows");
        }

        if (!mtp_spec_decode_metadata_binding_.hasWorkspace())
            return fail("condition advance has no bound MTP metadata workspace");
        const MTPSpecDecodeMetadataDevicePointers &workspace_ptrs =
            mtp_spec_decode_metadata_binding_.devicePointers();
        MTPSpecDecodeMetadataDevicePointers publication_ptrs = workspace_ptrs;
        if (!workspace_ptrs.base_cached_tokens ||
            !device_resident_logical_sequence_state_storage_.bindPublicationOutputs(
                &publication_ptrs,
                request_batch))
        {
            return fail("condition advance has incomplete arena publication outputs");
        }
        if (!backend->enqueueInitializeGenerationLogicalState({
                .source = GenerationInitialFrontier::SampledCondition,
                .positions = static_cast<const int32_t *>(mtp_sidecar_position_ids_dev_),
                .sampled_tokens = static_cast<const int32_t *>(stochastic_target_sample_tokens_dev_),
                .request_count = request_batch,
                .output = {.base_cached_tokens = workspace_ptrs.base_cached_tokens,
                    .target_positions = publication_ptrs.target_cached_tokens,
                    .accepted_state_counts = publication_ptrs.accepted_state_counts,
                    .next_condition_tokens = publication_ptrs.next_condition_tokens,
                    .all_drafts_accepted_flags = publication_ptrs.all_drafts_accepted_flags,
                    .stopped_flags = publication_ptrs.stopped_flags,
                    .publication_ok_flags = publication_ptrs.publication_ok_flags}},
                state_.device_id.gpu_ordinal(), stream))
        {
            return fail("advanced logical-state mailbox publication launch failed");
        }

        std::string mailbox_error;
        if (!recordDeviceResidentLogicalSequenceStateMailbox(
                request_batch,
                stream,
                DeviceResidentLogicalStatePublicationKind::RequestBatchConditionAdvance,
                &mailbox_error))
        {
            return fail(
                "advanced logical-state mailbox event failed: " + mailbox_error);
        }
        if (debugEnv().runtime_debug.mtp_condition_graph_contract_trace)
        {
            LOG_INFO(
                "[MTPConditionGraphContract] event=mailbox_published"
                << " device=" << state_.device_id.toString()
                << " requests=" << request_batch);
        }
        (void)consumePendingLogitsStream(
            PendingLogitsStreamRole::MainDecode,
            "advanceMTPRequestBatchConditionOnDeviceComplete");

        PerfStatsCollector::addCounter(
            "mtp",
            "request_batch_resident_condition_advances",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"requests", std::to_string(request_batch)},
             {"sampling", params.is_greedy() ? "greedy" : "stochastic"},
             {"condition_owner", "device_logical_state_mailbox"},
             {"target_owner", "device_target_sample_slots"}});
        return true;
    }

    bool DeviceGraphOrchestrator::forwardMTPBatchFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlots(
        const DeviceResidentLogicalSequenceStateHandle &logical_state,
        int request_batch,
        int first_draft_slot,
        int slot_stride)
    {
        if (request_batch <= 0 ||
            request_batch > stochastic_batch_output_request_capacity_ ||
            first_draft_slot < 0 || slot_stride <= 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident request-batched MTP sidecar exceeds configured request capacity or has invalid destination slots");
            return false;
        }
        if (!state_.device_id.is_gpu())
            return false;
        if (!logical_state.coversRequest(request_batch - 1) ||
            logical_state.device != state_.device_id ||
            !device_resident_logical_sequence_state_mailbox_.ownsHandle(
                logical_state,
                live_replay_state_epoch_))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident request-batched MTP sidecar received a stale or foreign logical-state mailbox");
            return false;
        }
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
            return false;
        if (state_.mtp_kv_caches.empty() || !state_.mtp_kv_caches[0])
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Batched MTP device-slot sidecar requires an initialized MTP KV cache");
            return false;
        }
        TensorBase *terminal_hidden = nullptr;
        BufferId terminal_hidden_buffer_id = BufferId::HIDDEN_STATE;
        if (!resolveMTPTerminalHiddenInput(
                &terminal_hidden,
                &terminal_hidden_buffer_id,
                "forwardMTPBatchFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlots"))
        {
            return false;
        }

        if (!executeMTPDepth0Batched(
                /*draft_condition_tokens=*/nullptr,
                /*token_count=*/1,
                terminal_hidden,
                /*position_id=*/0,
                "mtp_decode_sidecar_request_batch_resident_drafts",
                /*kv_cache_only=*/false,
                terminal_hidden_buffer_id,
                /*defer_final_sync=*/true,
                logical_state.next_condition_tokens_device,
                /*draft_condition_ready_slot=*/-1,
                /*draft_condition_ready_is_target=*/false,
                request_batch,
                /*position_ids_override=*/nullptr,
                logical_state.target_positions_device))
        {
            return false;
        }

        return sampleMTPBatchGreedyLogitsToDeviceDraftSlots(
            request_batch,
            first_draft_slot,
            slot_stride,
            "forwardMTPBatchFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlots",
            /*out_tokens=*/nullptr);
    }

    bool DeviceGraphOrchestrator::forwardMTPBatchFromLastDraftAndSampleGreedy(
        const int32_t *draft_condition_tokens,
        const int *position_ids,
        int request_batch,
        int32_t *out_tokens)
    {
        if (!draft_condition_tokens || !position_ids || !out_tokens ||
            request_batch <= 1 ||
            request_batch > stochastic_batch_output_request_capacity_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Batched chained MTP greedy sidecar exceeds configured request capacity");
            return false;
        }
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
            return false;

        auto hidden_it = state_.extension_buffers.find(BufferId::MTP_HIDDEN);
        TensorBase *mtp_hidden =
            (hidden_it == state_.extension_buffers.end() || !hidden_it->second)
                ? nullptr
                : hidden_it->second.get();
        if (!mtp_hidden)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Batched chained MTP greedy sidecar requires previous MTP hidden rows");
            return false;
        }

        if (!executeMTPDepth0Batched(
                draft_condition_tokens,
                /*token_count=*/1,
                mtp_hidden,
                position_ids[0],
                "mtp_decode_sidecar_request_batch_chain",
                /*kv_cache_only=*/false,
                BufferId::MTP_HIDDEN,
                /*defer_final_sync=*/true,
                /*draft_condition_tokens_device=*/nullptr,
                /*draft_condition_ready_slot=*/-1,
                /*draft_condition_ready_is_target=*/false,
                request_batch,
                position_ids))
        {
            return false;
        }

        auto logits_it = state_.extension_buffers.find(mtpSamplingLogitsBufferId());
        TensorBase *mtp_logits =
            (logits_it == state_.extension_buffers.end() || !logits_it->second)
                ? nullptr
                : logits_it->second.get();
        if (!mtp_logits)
            return false;

        const auto &shape = mtp_logits->shape();
        if (shape.empty())
            return false;
        const size_t rows = shape.size() >= 2 ? shape[0] : 1;
        const size_t cols = shape.size() >= 2 ? shape[1] : shape[0];
        if (cols == 0 ||
            rows < static_cast<size_t>(request_batch) ||
            cols > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            return false;
        }

        const int token_offset = mtpSidecarLogitsAreColumnParallel() && graph_builder_
                                     ? vocabOffsetForTPConfig(graph_builder_->config())
                                     : 0;
        auto device_opt = mtp_logits->current_device();
        if (device_opt.has_value() && device_opt->is_gpu() && mtp_logits->deviceValid())
        {
            IBackend *backend = getBackendFor(*device_opt);
            const void *gpu_ptr = mtp_logits->gpu_data_ptr();
            void *stream = consumePendingLogitsStream(
                PendingLogitsStreamRole::MTPSidecar,
                "forwardMTPBatchFromLastDraftAndSampleGreedy");
            if (!stream)
                stream = explicitGPUStreamForOperation(
                    "forwardMTPBatchFromLastDraftAndSampleGreedy");
            if (!backend || !gpu_ptr || !stream)
                return false;

            std::vector<float> values(static_cast<size_t>(request_batch));
            std::vector<int> indices(static_cast<size_t>(request_batch));
            if (!backend->argmaxF32BatchedRows(
                    gpu_ptr,
                    request_batch,
                    static_cast<int>(cols),
                    device_opt->gpu_ordinal(),
                    values.data(),
                    indices.data(),
                    stream,
                    argmax_partial_vals_dev_,
                    argmax_partial_idxs_dev_,
                    argmax_partial_capacity_))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Batched chained MTP greedy argmax failed");
                return false;
            }

            for (int row = 0; row < request_batch; ++row)
            {
                if (indices[static_cast<size_t>(row)] < 0)
                    return false;
                out_tokens[row] =
                    static_cast<int32_t>(token_offset + indices[static_cast<size_t>(row)]);
            }
            return true;
        }

        for (int row = 0; row < request_batch; ++row)
        {
            const int token = coordinateGreedyCandidate(
                sampleGreedyCandidateFromTensor(
                    mtp_logits,
                    row,
                    token_offset,
                    argmax_partial_vals_dev_,
                    argmax_partial_idxs_dev_,
                    argmax_partial_capacity_,
                    /*stream=*/nullptr,
                    "mtp_logits_request_batch_chain"),
                globalTPContextForMTPCoordination());
            if (token < 0)
                return false;
            out_tokens[row] = static_cast<int32_t>(token);
        }
        return true;
    }

    bool DeviceGraphOrchestrator::forwardMTPBatchFromDeviceDraftSlotsAndSampleGreedyToDeviceDraftSlots(
        const DeviceResidentLogicalSequenceStateHandle &logical_state,
        int request_batch,
        int first_condition_slot,
        int condition_slot_stride,
        int position_offset,
        int first_draft_slot,
        int draft_slot_stride)
    {
        const int last_condition_slot =
            first_condition_slot +
            (request_batch > 0
                 ? (request_batch - 1) * condition_slot_stride
                 : 0);
        if (request_batch <= 0 ||
            request_batch > stochastic_batch_output_request_capacity_ ||
            first_condition_slot < 0 || condition_slot_stride <= 0 ||
            last_condition_slot >= stochastic_draft_row_capacity_ ||
            position_offset <= 0 ||
            first_draft_slot < 0 || draft_slot_stride <= 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Chained device-resident request-batched MTP sidecar received invalid slot or depth geometry");
            return false;
        }
        if (!state_.device_id.is_gpu())
            return false;
        if (!logical_state.coversRequest(request_batch - 1) ||
            logical_state.device != state_.device_id ||
            !device_resident_logical_sequence_state_mailbox_.ownsHandle(
                logical_state,
                live_replay_state_epoch_))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Chained device-resident request-batched MTP sidecar received a stale or foreign logical-state mailbox");
            return false;
        }
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
            return false;
        if (!stochastic_draft_sample_tokens_dev_)
            return false;

        auto hidden_it = state_.extension_buffers.find(BufferId::MTP_HIDDEN);
        TensorBase *mtp_hidden =
            (hidden_it == state_.extension_buffers.end() || !hidden_it->second)
                ? nullptr
                : hidden_it->second.get();
        if (!mtp_hidden)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Batched chained MTP device-slot sidecar requires previous MTP hidden rows");
            return false;
        }

        if (!executeMTPDepth0Batched(
                /*draft_condition_tokens=*/nullptr,
                /*token_count=*/1,
                mtp_hidden,
                /*position_id=*/0,
                "mtp_decode_sidecar_request_batch_chain_resident_drafts",
                /*kv_cache_only=*/false,
                BufferId::MTP_HIDDEN,
                /*defer_final_sync=*/true,
                static_cast<const int32_t *>(
                    stochastic_draft_sample_tokens_dev_) +
                    first_condition_slot,
                first_condition_slot,
                /*draft_condition_ready_is_target=*/false,
                request_batch,
                /*position_ids_override=*/nullptr,
                logical_state.target_positions_device,
                /*speculative_outcome_meta_device=*/nullptr,
                /*speculative_outcome_meta_stride=*/0,
                /*speculative_outcome_output_tokens_device=*/nullptr,
                /*speculative_outcome_output_token_stride=*/0,
                /*speculative_outcome_request_index=*/-1,
                /*speculative_first_output_token_index=*/0,
                /*speculative_outcome_ready_event=*/nullptr,
                condition_slot_stride,
                request_batch,
                condition_slot_stride,
                position_offset))
        {
            return false;
        }

        return sampleMTPBatchGreedyLogitsToDeviceDraftSlots(
            request_batch,
            first_draft_slot,
            draft_slot_stride,
            "forwardMTPBatchFromDeviceDraftSlotsAndSampleGreedyToDeviceDraftSlots",
            /*out_tokens=*/nullptr);
    }

    bool DeviceGraphOrchestrator::forwardMTPFromLastDraftAndSampleGreedy(
        int32_t draft_condition_token,
        int position_id,
        int32_t *out_token)
    {
        if (!out_token)
            return false;
        *out_token = -1;
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
        {
            return false;
        }
        if (position_id < 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Chained MTP sidecar+sample requires a non-negative position_id");
            return false;
        }

        auto it = state_.extension_buffers.find(BufferId::MTP_HIDDEN);
        TensorBase *mtp_hidden =
            (it == state_.extension_buffers.end() || !it->second) ? nullptr : it->second.get();
        if (!mtp_hidden)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Chained MTP sidecar+sample requires previous MTP hidden state");
            return false;
        }

        if (!executeMTPDepth0(
                draft_condition_token,
                mtp_hidden,
                position_id,
                "mtp_decode_sidecar_chain",
                false,
                BufferId::MTP_HIDDEN,
                /*defer_final_sync=*/true))
        {
            return false;
        }

        const int token = sampleGreedyFromMTPLogitsOnDevice();
        if (token < 0)
            return false;
        *out_token = token;
        return true;
    }

    bool DeviceGraphOrchestrator::forwardMTPFromLastDraftAndSampleGreedyToDeviceDraftSlot(
        int32_t draft_condition_token,
        int position_id,
        int draft_sample_slot,
        int32_t *out_token)
    {
        if (out_token)
            *out_token = -1;
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
        {
            return false;
        }
        if (position_id < 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Chained MTP sidecar+device-slot sample requires a non-negative position_id");
            return false;
        }

        auto it = state_.extension_buffers.find(BufferId::MTP_HIDDEN);
        TensorBase *mtp_hidden =
            (it == state_.extension_buffers.end() || !it->second) ? nullptr : it->second.get();
        if (!mtp_hidden)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Chained MTP sidecar+device-slot sample requires previous MTP hidden state");
            return false;
        }

        if (!executeMTPDepth0(
                draft_condition_token,
                mtp_hidden,
                position_id,
                "mtp_decode_sidecar_chain",
                false,
                BufferId::MTP_HIDDEN,
                /*defer_final_sync=*/true))
        {
            return false;
        }

        return sampleGreedyFromMTPLogitsToDeviceDraftSlot(
            draft_sample_slot,
            out_token);
    }

    bool DeviceGraphOrchestrator::flushPendingMTPWork()
    {
        void *stream =
            peekPendingLogitsStream(PendingLogitsStreamRole::MTPSidecar);
        if (!stream)
        {
            return true;
        }

        if (state_.device_id.is_gpu())
        {
            /*
             * A GPU consumer must take ownership through
             * waitForPendingLogitsStream() or consume the producer stream
             * directly for an immediately chained kernel. Draining that stream
             * on the host would make device residency optional again and hide
             * a missing ordering edge, so reject the obsolete API outright.
             */
            LOG_ERROR("[DeviceGraphOrchestrator] flushPendingMTPWork cannot drain a GPU sidecar stream; "
                      "the next GPU consumer must publish an event wait");
            return false;
        }

        /*
         * CPU execution is synchronous and should not normally publish a
         * stream. Consume a defensive stale marker without creating a GPU
         * backend context.
         */
        (void)consumePendingLogitsStream(
            PendingLogitsStreamRole::MTPSidecar,
            "flushPendingMTPWork_cpu");
        return true;
    }

    const char *DeviceGraphOrchestrator::pendingLogitsStreamRoleName(
        PendingLogitsStreamRole role)
    {
        switch (role)
        {
        case PendingLogitsStreamRole::MTPSidecar:
            return "mtp_sidecar";
        case PendingLogitsStreamRole::MainDecode:
            return "main_decode";
        case PendingLogitsStreamRole::AllPositionVerifier:
            return "all_position_verifier";
        }
        return "unknown";
    }

    DeviceGraphOrchestrator::PendingLogitsStreamHandoff &
    DeviceGraphOrchestrator::pendingLogitsStreamHandoff(
        PendingLogitsStreamRole role)
    {
        switch (role)
        {
        case PendingLogitsStreamRole::MTPSidecar:
            return pending_logits_streams_[0];
        case PendingLogitsStreamRole::MainDecode:
            return pending_logits_streams_[1];
        case PendingLogitsStreamRole::AllPositionVerifier:
            return pending_logits_streams_[2];
        }
        throw std::logic_error("Unknown pending logits stream role");
    }

    const DeviceGraphOrchestrator::PendingLogitsStreamHandoff &
    DeviceGraphOrchestrator::pendingLogitsStreamHandoff(
        PendingLogitsStreamRole role) const
    {
        switch (role)
        {
        case PendingLogitsStreamRole::MTPSidecar:
            return pending_logits_streams_[0];
        case PendingLogitsStreamRole::MainDecode:
            return pending_logits_streams_[1];
        case PendingLogitsStreamRole::AllPositionVerifier:
            return pending_logits_streams_[2];
        }
        throw std::logic_error("Unknown pending logits stream role");
    }

    void *DeviceGraphOrchestrator::pendingLogitsStreamValue(
        PendingLogitsStreamRole role) const
    {
        return pendingLogitsStreamHandoff(role).peek();
    }

    void DeviceGraphOrchestrator::publishPendingLogitsStream(
        PendingLogitsStreamRole role,
        void *stream,
        const char *producer)
    {
        auto &handoff = pendingLogitsStreamHandoff(role);
        if (!stream)
        {
            clearPendingLogitsStream(
                role,
                producer ? producer : "publishPendingLogitsStream(nullptr)");
            return;
        }
        if (!handoff.canPublish(stream))
        {
            std::ostringstream message;
            message << "Cannot replace unconsumed pending logits stream for role "
                    << pendingLogitsStreamRoleName(role)
                    << " from producer "
                    << (producer ? producer : "unknown")
                    << ". Consume or explicitly clear the previous handoff first.";
            throw std::logic_error(message.str());
        }

        handoff.publish(stream);

        PerfStatsCollector::addCounter(
            "mtp",
            "pending_logits_stream_publishes",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"role", pendingLogitsStreamRoleName(role)},
             {"producer", producer ? producer : "unknown"}});
    }

    void *DeviceGraphOrchestrator::consumePendingLogitsStream(
        PendingLogitsStreamRole role,
        const char *consumer)
    {
        auto &handoff = pendingLogitsStreamHandoff(role);
        void *stream = handoff.consume();
        if (stream)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "pending_logits_stream_consumes",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"role", pendingLogitsStreamRoleName(role)},
                 {"consumer", consumer ? consumer : "unknown"}});
        }
        return stream;
    }

    void *DeviceGraphOrchestrator::prepareMainLogitsDeviceConsumer(
        const char *consumer,
        MainLogitsHandoffMode mode,
        bool require_main_forward)
    {
        if (!state_.device_id.is_gpu())
            return nullptr;

        const char *consumer_name =
            consumer && consumer[0] != '\0'
                ? consumer
                : "main_logits_device_consumer";
        void *stream =
            mode == MainLogitsHandoffMode::Consume
                ? consumePendingLogitsStream(
                      PendingLogitsStreamRole::MainDecode,
                      consumer_name)
                : peekPendingLogitsStream(
                      PendingLogitsStreamRole::MainDecode);
        if (!stream)
            stream = explicitGPUStreamForOperation(consumer_name);
        if (!stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Main-logits device consumer "
                      << consumer_name
                      << " could not acquire an explicit GPU stream");
            return nullptr;
        }

        /*
         * The role-specific stream is an optional latency optimization. The
         * durable event is the ownership proof and survives graph-cache reset,
         * prefix restore, and capture-stream rebinding. Always joining it here
         * makes stale-logits sampling structurally impossible even when no
         * deferred stream handoff was armed for this invocation.
         */
        if (!joinPublishedLiveStateHandoffs(
                stream,
                DeviceTimelineRole::TargetSampler,
                consumer_name))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Main-logits device consumer "
                      << consumer_name
                      << " could not join the live-state mutation timeline");
            return nullptr;
        }
        if (!waitForForwardGraphOutputReady(
                stream,
                DeviceTimelineRole::TargetSampler,
                require_main_forward
                    ? ForwardGraphOutputKind::MainForward
                    : ForwardGraphOutputKind::Any,
                consumer_name))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Main-logits device consumer "
                      << consumer_name
                      << " could not join ForwardGraphOutputReady");
            return nullptr;
        }

        PerfStatsCollector::addCounter(
            "sampling",
            "main_logits_durable_output_waits",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"consumer", consumer_name},
             {"handoff_mode",
              mode == MainLogitsHandoffMode::Consume
                  ? "consume"
                  : "observe"}});
        return stream;
    }

    bool DeviceGraphOrchestrator::waitForPendingLogitsStream(
        PendingLogitsStreamRole role,
        void *consumer_stream,
        const char *consumer)
    {
        void *producer_stream = consumePendingLogitsStream(role, consumer);
        if (!producer_stream)
            return true;
        if (!state_.device_id.is_gpu())
            return true;
        if (!consumer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Pending logits stream wait for role "
                      << pendingLogitsStreamRoleName(role)
                      << " requires an explicit consumer stream");
            return false;
        }

        const bool same_stream = producer_stream == consumer_stream;
        if (!same_stream)
        {
            /*
             * Queue a device-side dependency instead of synchronizing the
             * producer stream on the CPU.  This mirrors the cached graph-replay
             * collective handoff: the verifier graph can launch immediately,
             * but GPU execution waits until all sidecar work already enqueued
             * on the producer stream is complete.
             */
            if (!GPUDeviceContextPool::instance()
                     .getContext(state_.device_id)
                     .insertStreamDependency(
                         consumer_stream,
                         producer_stream))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to publish pending "
                          "logits stream dependency for role "
                          << pendingLogitsStreamRoleName(role)
                          << " consumer=" << (consumer ? consumer : "unknown"));
                return false;
            }
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "pending_logits_stream_waits",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"role", pendingLogitsStreamRoleName(role)},
             {"consumer", consumer ? consumer : "unknown"},
             {"same_stream", boolTag(same_stream)}});
        return true;
    }

    bool DeviceGraphOrchestrator::waitForPendingLogitsStreamForObservation(
        PendingLogitsStreamRole role,
        void *consumer_stream,
        const char *consumer) const
    {
        void *producer_stream = peekPendingLogitsStream(role);
        if (!producer_stream)
            return true;
        if (!state_.device_id.is_gpu())
            return true;
        if (!consumer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Pending logits stream observation wait for role "
                      << pendingLogitsStreamRoleName(role)
                      << " requires an explicit consumer stream");
            return false;
        }

        const bool same_stream = producer_stream == consumer_stream;
        if (!same_stream)
        {
            /*
             * Observations are read-only owners of the dependency, not owners
             * of the logits payload.  Queue the producer dependency exactly as
             * a consuming wait would, but leave the handoff published so the
             * sampler, reducer, or mutation boundary can still consume it.
             */
            if (!GPUDeviceContextPool::instance()
                     .getContext(state_.device_id)
                     .insertStreamDependency(
                         consumer_stream,
                         producer_stream))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to publish pending "
                          "logits observation dependency for role "
                          << pendingLogitsStreamRoleName(role)
                          << " consumer=" << (consumer ? consumer : "unknown"));
                return false;
            }
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "pending_logits_stream_observation_waits",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"role", pendingLogitsStreamRoleName(role)},
             {"consumer", consumer ? consumer : "unknown"},
             {"same_stream", boolTag(same_stream)}});
        return true;
    }

    void *DeviceGraphOrchestrator::peekPendingLogitsStream(
        PendingLogitsStreamRole role) const
    {
        return pendingLogitsStreamValue(role);
    }

    void DeviceGraphOrchestrator::clearPendingLogitsStream(
        PendingLogitsStreamRole role,
        const char *reason)
    {
        auto &handoff = pendingLogitsStreamHandoff(role);
        if (handoff.hasStream())
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "pending_logits_stream_clears",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"role", pendingLogitsStreamRoleName(role)},
                 {"reason", reason ? reason : "unknown"}});
        }
        handoff.clear();
    }

    void DeviceGraphOrchestrator::clearAllPendingLogitsStreams(const char *reason)
    {
        clearPendingLogitsStream(PendingLogitsStreamRole::MTPSidecar, reason);
        clearPendingLogitsStream(PendingLogitsStreamRole::MainDecode, reason);
        clearPendingLogitsStream(PendingLogitsStreamRole::AllPositionVerifier, reason);
    }

    bool DeviceGraphOrchestrator::waitForPendingLiveGraphProducersBeforePrefixMutation(
        void *mutation_stream,
        const char *mutation_name)
    {
        const char *consumer =
            mutation_name && mutation_name[0] != '\0'
                ? mutation_name
                : "live_prefix_mutation";

        /*
         * These handoffs are producer-order tokens, not semantic ownership of
         * logits.  Consuming them here is correct because a prefix restore or
         * truncate invalidates any logits from the abandoned timeline.
         */
        if (!waitForPendingLogitsStream(
                PendingLogitsStreamRole::MTPSidecar,
                mutation_stream,
                consumer))
        {
            return false;
        }
        if (!waitForPendingLogitsStream(
                PendingLogitsStreamRole::MainDecode,
                mutation_stream,
                consumer))
        {
            return false;
        }
        if (!waitForPendingLogitsStream(
                PendingLogitsStreamRole::AllPositionVerifier,
                mutation_stream,
                consumer))
        {
            return false;
        }
        if (!waitForPendingShiftedMTPKVReady(
                mutation_stream,
                consumer))
        {
            return false;
        }
        if (!waitForPendingAllPositionVerifierStateReady(
                mutation_stream,
                consumer))
        {
            return false;
        }
        /*
         * A live-state mutation may truncate or replace the request whose
         * terminal hidden row is still being archived.  Order the mutation
         * after that device copy before any destination buffer can be reused.
         * This consumes the handoff because the mutation establishes a new
         * logical timeline.
         */
        if (!waitForPendingMTPPrefillTerminalArchiveReady(
                mutation_stream,
                consumer))
        {
            return false;
        }
        return true;
    }

    bool DeviceGraphOrchestrator::initializeForwardGraphOutputReadyEvent()
    {
        if (!state_.device_id.is_gpu())
            return true;

        auto &ready_set = forward_graph_output_ready_;
        if (ready_set.main_forward.event &&
            ready_set.main_forward.device == state_.device_id &&
            ready_set.grouped_verifier.event &&
            ready_set.grouped_verifier.device == state_.device_id)
            return true;

        /*
         * Reinitialization is a binding-identity boundary. No inference work can
         * be active while initializeBuffers() replaces the runner device, so the
         * previous backend event may be retired before creating its replacement.
         */
        ready_set = {};
        clearCurrentMainLogitsPublication(
            "initialize_forward_graph_output_ready_event");
        ready_set.main_forward.event =
            allocatePersistentDeviceEvent(state_.device_id);
        ready_set.grouped_verifier.event =
            allocatePersistentDeviceEvent(state_.device_id);
        if (!ready_set.main_forward.event ||
            !ready_set.grouped_verifier.event)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to allocate typed ForwardGraphOutputReady events for "
                      << state_.device_id.toString());
            ready_set = {};
            return false;
        }
        ready_set.main_forward.device = state_.device_id;
        ready_set.grouped_verifier.device = state_.device_id;
        return true;
    }

    DeviceGraphOrchestrator::ForwardGraphOutputReadyState *
    DeviceGraphOrchestrator::forwardGraphOutputPublicationFor(
        ForwardExecutionRole role,
        bool all_position_logits)
    {
        switch (role)
        {
        case ForwardExecutionRole::MainInference:
        case ForwardExecutionRole::MTPCondition:
            return &forward_graph_output_ready_.main_forward;
        case ForwardExecutionRole::GroupedMTPVerifier:
            if (!all_position_logits)
                return nullptr;
            return &forward_graph_output_ready_.grouped_verifier;
        }
        return nullptr;
    }

    const DeviceGraphOrchestrator::ForwardGraphOutputReadyState *
    DeviceGraphOrchestrator::selectForwardGraphOutputPublication(
        ForwardGraphOutputKind required_kind) const
    {
        const auto &main = forward_graph_output_ready_.main_forward;
        const auto &grouped = forward_graph_output_ready_.grouped_verifier;
        switch (required_kind)
        {
        case ForwardGraphOutputKind::MainForward:
            return main.valid ? &main : nullptr;
        case ForwardGraphOutputKind::GroupedVerifier:
            return grouped.valid ? &grouped : nullptr;
        case ForwardGraphOutputKind::Any:
            if (!main.valid)
                return grouped.valid ? &grouped : nullptr;
            if (!grouped.valid)
                return &main;
            return grouped.publication_sequence > main.publication_sequence
                       ? &grouped
                       : &main;
        }
        return nullptr;
    }

    void DeviceGraphOrchestrator::clearForwardGraphOutputPublications() noexcept
    {
        auto clear = [](ForwardGraphOutputReadyState &ready)
        {
            ready.valid = false;
            ready.publication_sequence = 0;
            ready.execution_role = ForwardExecutionRole::MainInference;
            ready.is_decode = false;
            ready.all_position_logits = false;
            ready.completion_scope =
                ForwardCompletionScope::ModelForwardOnly;
            ready.graph_seq_len = 0;
            ready.graph_batch_size = 0;
        };
        clear(forward_graph_output_ready_.main_forward);
        clear(forward_graph_output_ready_.grouped_verifier);
        forward_graph_output_ready_.next_publication_sequence = 1;
    }

    ForwardLogitsStorageSurface
    DeviceGraphOrchestrator::classifyForwardLogitsSurface(
        const TensorBase *tensor) const noexcept
    {
        if (!tensor)
            return ForwardLogitsStorageSurface::None;
        if (tensor == state_.all_position_logits.get())
            return ForwardLogitsStorageSurface::AllPositionFull;
        if (tensor == state_.all_position_logits_local.get())
            return ForwardLogitsStorageSurface::AllPositionLocal;
        if (tensor == state_.logits.get())
            return ForwardLogitsStorageSurface::CanonicalFull;
        if (tensor == state_.logits_local.get())
            return ForwardLogitsStorageSurface::CanonicalLocal;
        return ForwardLogitsStorageSurface::Unknown;
    }

    bool DeviceGraphOrchestrator::publishCurrentMainLogits(
        TensorBase *tensor,
        const ForwardLogitsPublicationDescriptor &descriptor,
        MainLogitsPublicationSource source,
        const char *producer_name)
    {
        if (!descriptor.isMainModelOutput())
            return true;
        if (!tensor ||
            descriptor.storage_surface ==
                ForwardLogitsStorageSurface::None)
        {
            clearCurrentMainLogitsPublication(producer_name);
            return true;
        }
        if (descriptor.storage_surface ==
                ForwardLogitsStorageSurface::Unknown ||
            source == MainLogitsPublicationSource::None)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Main-logits producer does not name an owned storage surface"
                      << " producer="
                      << (producer_name && producer_name[0] != '\0'
                              ? producer_name
                              : "unknown")
                      << " role="
                      << static_cast<int>(descriptor.execution_role)
                      << " tensor=" << tensor
                      << " surface="
                      << static_cast<int>(descriptor.storage_surface));
            return false;
        }

        current_main_logits_publication_ = {
            .tensor = tensor,
            .descriptor = descriptor,
            .source = source,
            .session_epoch = session_epoch_,
            .valid = true,
        };
        PerfStatsCollector::addCounter(
            "sampling",
            "current_main_logits_publications",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"producer",
              producer_name && producer_name[0] != '\0'
                  ? producer_name
                  : "unknown"},
             {"surface",
              std::to_string(
                  static_cast<int>(descriptor.storage_surface))},
             {"execution_role",
              std::to_string(
                  static_cast<int>(descriptor.execution_role))}});
        return true;
    }

    bool DeviceGraphOrchestrator::publishCPUAcceptedVerifierLogits(
        const PrefixLogitsRowSelection &selection)
    {
        if (!state_.device_id.is_cpu())
            throw std::logic_error("Host accepted-logit publication requires CPU ownership");
        selection.requireCurrent(session_epoch_);
        if (pp_stage_config_ && !pp_stage_config_->has_lm_head)
            return true;
        if (!graph_builder_ || graph_builder_->config().vocab_size <= 0)
            return false;
        TensorBase *const source = state_.all_position_logits.get();
        TensorBase *const destination = state_.logits.get();
        const size_t vocabulary = static_cast<size_t>(graph_builder_->config().vocab_size);
        if (!source || !destination || source->native_type() != TensorType::FP32 ||
            destination->native_type() != TensorType::FP32 ||
            source->rows() <= 0 ||
            static_cast<size_t>(source->rows()) != selection.rows() ||
            source->cols() < static_cast<int64_t>(vocabulary) ||
            destination->cols() < static_cast<int64_t>(vocabulary) ||
            !source->raw_data() || !destination->raw_mutable_data())
            return false;
        const PrefixLogitsVocabularyRange full{0, vocabulary};
        const auto slice = PrefixTerminalLogitsSlice::resolveSelectedRow(
            full, full, fp32LogitsRowBytes(source), selection, session_epoch_);
        // Stateful publication selected this same row for KV/GDN and terminal
        // hidden. Copy only its live vocabulary, never the retained matrix or
        // its unused capacity. Canonical storage survives the next verifier's
        // reuse and keeps existing scalar samplers on a single-row authority.
        const auto *bytes = static_cast<const uint8_t *>(source->raw_data());
        std::memmove(destination->raw_mutable_data(), bytes + slice.byteOffset(), slice.byteCount());
        TransferEngine::publishHostWrite(destination);
        if (arena_ && arena_->isRegistered(BufferId::LOGITS))
            arena_->markWrittenFlagsOnly(BufferId::LOGITS, DeviceId::cpu());
        const bool published = publishCurrentMainLogits(destination,
            {.execution_role = ForwardExecutionRole::MainInference,
             .logical_all_position_logits = false,
             .storage_surface = ForwardLogitsStorageSurface::CanonicalFull},
            MainLogitsPublicationSource::AcceptedVerifierState,
            "accepted_cpu_verifier_row");
        if (published)
        {
            PerfStatsCollector::addCounter("mtp", "accepted_terminal_logits_rows", 1.0,
                "decode", state_.device_id.toString(),
                {{"selected_row", std::to_string(selection.row())},
                 {"producer_rows", std::to_string(selection.rows())}});
            PerfStatsCollector::addCounter("mtp", "accepted_terminal_logits_bytes",
                static_cast<double>(slice.byteCount()), "decode", state_.device_id.toString());
        }
        return published;
    }

    bool DeviceGraphOrchestrator::publishCanonicalMTPMainLogits(
        void *stream, const char *producer_name)
    {
        if (!state_.device_id.is_gpu() || !stream || !graph_builder_ || !arena_)
            return false;
        if (pp_stage_config_ && !pp_stage_config_->has_lm_head)
            return true;
        const bool local = allPositionVerifierGraphWritesLocalLogits(1) &&
            !graph_builder_->config().hasNativeLocalVocabularyGather();
        TensorBase *destination = local ? state_.logits_local.get() : state_.logits.get();
        if (!destination || !destination->gpu_data_ptr())
            return false;
        // Same canonical tensor/event authority as an ordinary forward output.
        // The caller has either enqueued its selector on this exact stream or
        // joined the original complete parent's terminal event on this stream.
        arena_->markWritten(local ? BufferId::LOGITS_LOCAL : BufferId::LOGITS, state_.device_id, stream);
        return publishCurrentMainLogits(destination,
            {.execution_role = ForwardExecutionRole::MainInference,
             .logical_all_position_logits = false,
             .storage_surface = local ? ForwardLogitsStorageSurface::CanonicalLocal : ForwardLogitsStorageSurface::CanonicalFull},
            MainLogitsPublicationSource::AcceptedVerifierState, producer_name);
    }

    void DeviceGraphOrchestrator::clearCurrentMainLogitsPublication(
        const char *reason) noexcept
    {
        (void)reason;
        current_main_logits_publication_ = {};
    }

    TensorBase *DeviceGraphOrchestrator::requireCurrentMainLogits(
        DeviceLogitsSource source,
        const char *consumer_name) const
    {
        if (source != DeviceLogitsSource::Main &&
            source != DeviceLogitsSource::MainRequestBatch)
        {
            throw std::invalid_argument(
                "Current-main logits resolver received a non-main source");
        }

        const auto &publication = current_main_logits_publication_;
        const bool source_supported =
            source == DeviceLogitsSource::Main
                ? publication.descriptor.supportsScalarMainConsumer()
                : publication.descriptor.supportsMainRequestBatchConsumer();
        if (!publication.valid || !publication.tensor ||
            publication.session_epoch != session_epoch_ ||
            !source_supported)
        {
            std::ostringstream message;
            message << "Main-logits consumer "
                    << (consumer_name && consumer_name[0] != '\0'
                            ? consumer_name
                            : "unknown")
                    << " has no compatible current publication"
                    << " source=" << static_cast<int>(source)
                    << " valid=" << publication.valid
                    << " publication_epoch=" << publication.session_epoch
                    << " session_epoch=" << session_epoch_
                    << " execution_role="
                    << static_cast<int>(
                           publication.descriptor.execution_role)
                    << " logical_all_position="
                    << publication.descriptor.logical_all_position_logits
                    << " storage_surface="
                    << static_cast<int>(
                           publication.descriptor.storage_surface);
            throw std::logic_error(message.str());
        }
        return publication.tensor;
    }

    std::optional<DeviceGraphOrchestrator::PrefixArchiveLogitsView>
    DeviceGraphOrchestrator::currentPrefixArchiveLogits() const
    {
        const auto &publication = current_main_logits_publication_;
        if (!publication.valid || !publication.tensor ||
            publication.session_epoch != session_epoch_ ||
            !publication.descriptor.supportsScalarMainConsumer() ||
            prefix_layout_.terminal_logits_bytes == 0)
        {
            return std::nullopt;
        }
        if (!graph_builder_ || publication.tensor->native_type() != TensorType::FP32 ||
            prefix_layout_.terminal_logits_bytes % sizeof(float) != 0)
            throw std::logic_error("Prefix terminal logits require an owned FP32 vocabulary row");

        const auto &config = graph_builder_->config();
        const int local_start = localLogitsVocabOffset();
        if (config.vocab_size <= 0 || local_start < 0)
            throw std::logic_error("Prefix terminal logits have invalid vocabulary ownership");
        const size_t full_columns = static_cast<size_t>(config.vocab_size);
        const size_t archive_columns = prefix_layout_.terminal_logits_bytes / sizeof(float);
        const PrefixLogitsVocabularyRange full{0, full_columns};
        const PrefixLogitsVocabularyRange local{static_cast<size_t>(local_start),
            config.vocab_local > 0 ? static_cast<size_t>(config.vocab_local) : 0};
        const PrefixLogitsVocabularyRange archive = archive_columns == full_columns ? full : local;
        if (archive_columns != archive.token_count)
            throw std::logic_error("Prefix terminal-logit layout disagrees with its vocabulary shard");

        PrefixLogitsVocabularyRange source = full;
        switch (publication.descriptor.storage_surface)
        {
        case ForwardLogitsStorageSurface::CanonicalFull:
        case ForwardLogitsStorageSurface::AllPositionFull:
            break;
        case ForwardLogitsStorageSurface::CanonicalLocal:
        case ForwardLogitsStorageSurface::AllPositionLocal:
            source = local;
            break;
        default:
            throw std::logic_error("Prefix terminal logits have no owned publication surface");
        }
        // The immutable archive width, not the current decode/prefill phase,
        // selects its vocabulary interval. The published surface selects the
        // physical source. A gathered CPU row therefore starts at this rank's
        // vocab offset; a local CPU/GPU row starts at byte zero.
        const size_t full_row_bytes = fp32LogitsRowBytes(state_.logits.get());
        (void)PrefixTerminalLogitsSlice::resolve(full, archive, full_row_bytes);
        return PrefixArchiveLogitsView{publication.tensor,
            PrefixTerminalLogitsSlice::resolve(source, archive, fp32LogitsRowBytes(publication.tensor))};
    }

    TensorBase *DeviceGraphOrchestrator::resolveDeviceLogitsTensor(
        DeviceLogitsSource source,
        const char *consumer_name) const
    {
        if (source == DeviceLogitsSource::Main ||
            source == DeviceLogitsSource::MainRequestBatch)
        {
            return requireCurrentMainLogits(source, consumer_name);
        }
        if (source == DeviceLogitsSource::AllPosition)
            return state_.all_position_logits.get();

        const auto it = state_.extension_buffers.find(mtpSamplingLogitsBufferId());
        return it == state_.extension_buffers.end()
                   ? nullptr
                   : it->second.get();
    }

    bool DeviceGraphOrchestrator::publishForwardGraphOutputReady(
        const ForwardOutput &output)
    {
        const auto &producer = output.execution;
        if (!producer.valid)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot publish invalid forward-output provenance");
            return false;
        }

        if (producer.execution_role ==
                ForwardExecutionRole::MainInference ||
            producer.execution_role ==
                ForwardExecutionRole::MTPCondition)
        {
            const ForwardLogitsPublicationDescriptor descriptor{
                .execution_role = producer.execution_role,
                .logical_all_position_logits =
                    producer.all_position_logits,
                .storage_surface =
                    classifyForwardLogitsSurface(output.logits),
            };
            if (!publishCurrentMainLogits(
                    output.logits,
                    descriptor,
                    MainLogitsPublicationSource::ForwardGraph,
                    "forward_graph"))
            {
                return false;
            }
        }

        if (!state_.device_id.is_gpu())
            return true;

        auto *ready = forwardGraphOutputPublicationFor(
            producer.execution_role,
            producer.all_position_logits);
        if (!producer.valid ||
            producer.device != state_.device_id ||
            !producer.stream ||
            !ready ||
            !ready->event ||
            ready->device != state_.device_id)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid durable forward-output publication"
                      << " expected_device=" << state_.device_id.toString()
                      << " producer_valid=" << producer.valid
                      << " producer_device=" << producer.device.toString()
                      << " producer_stream=" << producer.stream
                      << " event=" << (ready ? ready->event.get() : nullptr)
                      << " event_device="
                      << (ready ? ready->device.toString() : std::string("invalid")));
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend ||
            !DeviceEventEdge::at(DeviceTimelinePoint::ForwardGraphOutputReady)
                 .from(DeviceTimelineRole::MainForwardGraph)
                 .publish(
                     *backend,
                     state_.device_id,
                     ready->event.get(),
                     producer.stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to publish ForwardGraphOutputReady for "
                      << state_.device_id.toString());
            return false;
        }

        /*
         * Retain semantic metadata only. The producer stream is owned by the
         * forward graph cache and is invalid outside this publication call.
         */
        ready->valid = true;
        ready->publication_sequence =
            forward_graph_output_ready_.next_publication_sequence++;
        ready->execution_role = producer.execution_role;
        ready->is_decode = producer.is_decode;
        ready->all_position_logits = producer.all_position_logits;
        ready->completion_scope = producer.completion_scope;
        ready->graph_seq_len = producer.graph_seq_len;
        ready->graph_batch_size = producer.graph_batch_size;
        PerfStatsCollector::addCounter(
            "forward_graph",
            "durable_output_publications",
            1.0,
            producer.is_decode ? "decode" : "prefill",
            state_.device_id.toString(),
            {{"execution_role",
              std::to_string(static_cast<int>(producer.execution_role))},
             {"all_position_logits",
              boolTag(producer.all_position_logits)}});
        return true;
    }

    bool DeviceGraphOrchestrator::waitForLastInferenceCompletionForBenchmark()
    {
        if (!state_.device_id.is_gpu())
            return true;

        /**
         * A benchmark observes one of two mutually exclusive production
         * transactions.  Keep that authority typed here: a full prefix hit
         * intentionally launches no forward graph, while an ordinary prefill
         * intentionally owns no prefix-restore completion edge.
         */
        enum class CompletionSource
        {
            ForwardGraph,
            PrefixTerminalRestore,
        };
        struct CompletionPublication
        {
            CompletionSource source = CompletionSource::ForwardGraph;
            std::shared_ptr<void> event;
            const char *ordering = "invalid";
            const char *phase = "prefill";
            bool mtp_prefill_included = false;
            uint64_t publication_sequence = 0;
            int graph_seq_len = 0;
            int graph_batch_size = 0;
            ForwardExecutionRole execution_role =
                ForwardExecutionRole::MainInference;
        };

        CompletionPublication completion;
        const bool current_main_publication =
            current_main_logits_publication_.valid &&
            current_main_logits_publication_.session_epoch == session_epoch_;
        const auto current_main_source =
            current_main_publication
                ? current_main_logits_publication_.source
                : MainLogitsPublicationSource::None;

        const auto *ready = selectForwardGraphOutputPublication(
            ForwardGraphOutputKind::Any);
        if (current_main_source ==
            MainLogitsPublicationSource::PrefixTerminalRestore)
        {
            if (!live_prefix_mutation_ready_.valid ||
                !live_prefix_mutation_ready_.event)
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Benchmark completion boundary "
                    "found prefix-restored output without its durable mutation event"
                    << " device=" << state_.device_id.toString()
                    << " live_state_epoch="
                    << live_prefix_mutation_ready_.live_state_epoch
                    << " event="
                    << live_prefix_mutation_ready_.event.get());
                return false;
            }
            completion.source = CompletionSource::PrefixTerminalRestore;
            completion.event = live_prefix_mutation_ready_.event;
            completion.ordering = "prefix_terminal_restore_event";
            completion.phase = "prefill";
            completion.mtp_prefill_included =
                graph_builder_ && graph_builder_->config().mtp.enabled &&
                !state_.mtp_kv_caches.empty();
            completion.publication_sequence =
                live_prefix_mutation_ready_.live_state_epoch;
            completion.graph_seq_len = state_.positions.empty()
                                               ? 0
                                               : state_.positions.front();
            completion.graph_batch_size = state_.batch_size;
        }
        else
        {
            if (!ready || !ready->event ||
                ready->device != state_.device_id)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Benchmark completion boundary has no valid durable transaction event"
                          << " device=" << state_.device_id.toString()
                          << " output_source="
                          << static_cast<int>(current_main_source)
                          << " forward_valid=" << (ready != nullptr)
                          << " event="
                          << (ready ? ready->event.get() : nullptr)
                          << " event_device="
                          << (ready ? ready->device.toString()
                                    : std::string("invalid")));
                return false;
            }
            completion.source = CompletionSource::ForwardGraph;
            completion.event = ready->event;
            completion.ordering = "durable_forward_output_event";
            completion.phase = ready->is_decode ? "decode" : "prefill";
            completion.mtp_prefill_included =
                !ready->is_decode &&
                ready->execution_role ==
                    ForwardExecutionRole::MainInference &&
                graph_builder_ && graph_builder_->config().mtp.enabled &&
                !state_.mtp_kv_caches.empty();
            completion.publication_sequence =
                ready->publication_sequence;
            completion.graph_seq_len = ready->graph_seq_len;
            completion.graph_batch_size = ready->graph_batch_size;
            completion.execution_role = ready->execution_role;
        }

        if (completion.source == CompletionSource::ForwardGraph &&
            completion.mtp_prefill_included &&
            ready->completion_scope !=
                ForwardCompletionScope::GraphIntegratedShiftedMTPPrefill)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] GPU MTP prefill completed without graph-integrated shifted-KV scope"
                      << " device=" << state_.device_id.toString()
                      << " graph_seq_len=" << ready->graph_seq_len
                      << " graph_batch_size=" << ready->graph_batch_size
                      << " completion_scope="
                      << static_cast<int>(ready->completion_scope));
            return false;
        }
        if (completion.source == CompletionSource::ForwardGraph &&
            completion.mtp_prefill_included)
        {
            /*
             * Shifted MTP KV population and terminal archival are graph nodes,
             * so ForwardGraphOutputReady is the transitive terminal event for
             * the complete production prefill transaction. Require that exact
             * scope rather than inferring completeness from MTP configuration
             * or consulting the retired post-forward sidecar handoff. A GPU MTP
             * prefill that does not publish this scope is incomplete and must
             * fail instead of measuring only its main-model prefix.
             */
            completion.ordering =
                "graph_integrated_shifted_mtp_prefill_event";
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!completion.event || !backend)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Benchmark completion boundary has no queryable exact transaction event on "
                      << state_.device_id.toString()
                      << " ordering=" << completion.ordering);
            return false;
        }

        /*
         * A host benchmark must observe the exact terminal event, but an
         * unbounded cudaEventSynchronize()/hipEventSynchronize() turns a lost
         * graph edge into a permanently occupied accelerator. Poll the event
         * through the backend's non-blocking query contract and apply the same
         * canonical 30-second failure bound as graph/collective rendezvous.
         * This remains a benchmark result boundary only: production consumers
         * continue to chain the event on-device and never enter this loop.
         */
        const auto deadline =
            std::chrono::steady_clock::now() +
            std::chrono::milliseconds(
                collective_timeout_policy::kDefaultCollectiveTimeoutMs);
        bool event_ready = false;
        while (!event_ready && std::chrono::steady_clock::now() < deadline)
        {
            if (!backend->queryEvent(
                    completion.event.get(),
                    state_.device_id.gpu_ordinal(),
                    &event_ready))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Benchmark failed to query the exact transaction event on "
                          << state_.device_id.toString()
                          << " ordering=" << completion.ordering);
                return false;
            }
            if (!event_ready)
                std::this_thread::yield();
        }
        if (!event_ready)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Benchmark exact transaction event timed out after "
                      << collective_timeout_policy::kDefaultCollectiveTimeoutMs
                      << "ms on " << state_.device_id.toString()
                      << " ordering=" << completion.ordering
                      << " publication_sequence="
                      << completion.publication_sequence
                      << " graph_seq_len=" << completion.graph_seq_len
                      << " graph_batch_size="
                      << completion.graph_batch_size
                      << " execution_role="
                      << static_cast<int>(completion.execution_role));
            const auto *const moe = graph_builder_
                                        ? &graph_builder_->config().moe
                                        : nullptr;
            if (moe && moe->device_controller_fabric)
            {
                try
                {
                    const auto barrier =
                        moe_overlay_epoch_execution_binding_
                            ? moe_overlay_epoch_execution_binding_.arena
                                  ->admissionBarrier()
                            : DeviceMoEOverlayEpochAdmissionBarrierBinding{};
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Fatal forward timeout epoch evidence device="
                        << state_.device_id.toString()
                        << " local_participant="
                        << (barrier.valid()
                                ? std::to_string(barrier.participant_id)
                                : std::string("unbound"))
                        << ' '
                        << moe->device_controller_fabric
                               ->describeInferenceEpochBarrier());
                }
                catch (const std::exception &exception)
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Could not describe fatal forward epoch state: "
                        << exception.what());
                }
            }
            return false;
        }

        PerfStatsCollector::addCounter(
            "forward_graph",
            "benchmark_terminal_event_waits",
            1.0,
            completion.phase,
            state_.device_id.toString(),
            {{"ordering", completion.ordering},
             {"transaction_source",
              completion.source == CompletionSource::ForwardGraph
                  ? "forward_graph"
                  : "prefix_terminal_restore"},
             {"scope", "benchmark_result_boundary"},
             {"mtp_prefill_included",
              boolTag(completion.mtp_prefill_included)},
             {"device_sync", "false"},
             {"host_transfer", "false"}});
        return true;
    }

    bool DeviceGraphOrchestrator::waitForForwardGraphOutputReady(
        void *consumer_stream,
        DeviceTimelineRole consumer_role,
        ForwardGraphOutputKind required_kind,
        const char *consumer_name) const
    {
        if (!state_.device_id.is_gpu())
            return true;
        if (!consumer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] ForwardGraphOutputReady consumer requires an explicit GPU stream");
            return false;
        }

        const auto *ready =
            selectForwardGraphOutputPublication(required_kind);
        if (!ready)
            return true;
        if (!ready->event || ready->device != state_.device_id)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] ForwardGraphOutputReady publication has invalid durable ownership");
            return false;
        }

        const bool main_forward =
            ready == &forward_graph_output_ready_.main_forward &&
            (ready->execution_role ==
                 ForwardExecutionRole::MainInference ||
             ready->execution_role ==
                 ForwardExecutionRole::MTPCondition);
        const bool grouped_verifier =
            ready == &forward_graph_output_ready_.grouped_verifier &&
            ready->all_position_logits &&
            ready->execution_role ==
                ForwardExecutionRole::GroupedMTPVerifier;
        const bool kind_matches =
            required_kind == ForwardGraphOutputKind::Any ||
            (required_kind == ForwardGraphOutputKind::MainForward &&
             main_forward) ||
            (required_kind == ForwardGraphOutputKind::GroupedVerifier &&
             grouped_verifier);
        if (!kind_matches)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] ForwardGraphOutputReady kind mismatch for consumer="
                      << (consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown")
                      << " required="
                      << (required_kind == ForwardGraphOutputKind::MainForward
                              ? "main_forward"
                              : "grouped_verifier")
                      << " published_execution_role="
                      << static_cast<int>(ready->execution_role)
                      << " published_all_position_logits="
                      << ready->all_position_logits);
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend ||
            !DeviceEventEdge::at(DeviceTimelinePoint::ForwardGraphOutputReady)
                 .from(DeviceTimelineRole::MainForwardGraph)
                 .to(consumer_role)
                 .enqueuePublishedWait(
                     *backend,
                     state_.device_id,
                     ready->event.get(),
                     consumer_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to consume durable ForwardGraphOutputReady publication for "
                      << (consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown"));
            return false;
        }

        PerfStatsCollector::addCounter(
            "forward_graph",
            "durable_output_waits",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"consumer",
              consumer_name && consumer_name[0] != '\0'
                  ? consumer_name
                  : "unknown"},
             {"consumer_role",
              std::string(deviceTimelineRoleName(consumer_role))},
             {"decode", boolTag(ready->is_decode)},
             {"all_position_logits",
              boolTag(ready->all_position_logits)}});
        return true;
    }

    void *DeviceGraphOrchestrator::prepareLogitsHostObservation(
        HostLogitsSurface surface,
        const char *operation)
    {
        if (!state_.device_id.is_gpu())
            return nullptr;

        PendingLogitsStreamRole pending_role =
            PendingLogitsStreamRole::MainDecode;
        ForwardGraphOutputKind required_kind =
            ForwardGraphOutputKind::MainForward;
        switch (surface)
        {
        case HostLogitsSurface::Main:
            pending_role = PendingLogitsStreamRole::MainDecode;
            required_kind = ForwardGraphOutputKind::MainForward;
            break;
        case HostLogitsSurface::MTPSidecar:
            pending_role = PendingLogitsStreamRole::MTPSidecar;
            /*
             * Sidecar execution may be the latest ordinary forward or a
             * dedicated cached sidecar graph. The durable event covers the
             * former; the role-specific handoff covers the latter.
             */
            required_kind = ForwardGraphOutputKind::Any;
            break;
        case HostLogitsSurface::AllPositionVerifier:
            pending_role = PendingLogitsStreamRole::AllPositionVerifier;
            required_kind = ForwardGraphOutputKind::GroupedVerifier;
            break;
        }

        const bool current_main_is_prefix_restore =
            surface == HostLogitsSurface::Main &&
            current_main_logits_publication_.valid &&
            current_main_logits_publication_.session_epoch == session_epoch_ &&
            current_main_logits_publication_.source ==
                MainLogitsPublicationSource::PrefixTerminalRestore;
        const bool has_live_prefix_restore_publication =
            current_main_is_prefix_restore &&
            live_prefix_mutation_ready_.valid;
        const bool has_grouped_prefix_restore_successor =
            current_main_is_prefix_restore &&
            !has_live_prefix_restore_publication &&
            selectForwardGraphOutputPublication(
                ForwardGraphOutputKind::GroupedVerifier) != nullptr;
        if (has_grouped_prefix_restore_successor)
        {
            /*
             * Grouped-verifier admission consumes the live restore handoff,
             * making verifier completion its transitive successor. The graph
             * does not overwrite canonical LOGITS, so restored tensor identity
             * remains authoritative while the grouped event orders observation.
             */
            required_kind = ForwardGraphOutputKind::GroupedVerifier;
        }

        const bool has_durable_publication =
            selectForwardGraphOutputPublication(required_kind) != nullptr;
        const bool has_stream_publication =
            pendingLogitsStreamValue(pending_role) != nullptr;
        const bool has_verifier_event_publication =
            surface == HostLogitsSurface::AllPositionVerifier &&
            all_position_verifier_state_ready_.valid;
        if (!has_durable_publication &&
            !has_stream_publication &&
            !has_verifier_event_publication &&
            !has_live_prefix_restore_publication)
        {
            std::ostringstream message;
            message << "GPU logits host observation has no producer publication"
                    << " operation=" << (operation ? operation : "unknown")
                    << " device=" << state_.device_id.toString()
                    << " current_main_valid="
                    << current_main_logits_publication_.valid
                    << " current_main_source="
                    << static_cast<int>(
                           current_main_logits_publication_.source)
                    << " current_main_epoch="
                    << current_main_logits_publication_.session_epoch
                    << " session_epoch=" << session_epoch_
                    << " live_prefix_ready="
                    << live_prefix_mutation_ready_.valid
                    << " main_forward_ready="
                    << forward_graph_output_ready_.main_forward.valid
                    << " grouped_verifier_ready="
                    << forward_graph_output_ready_.grouped_verifier.valid;
            throw std::runtime_error(message.str());
        }

        void *host_bridge_stream =
            explicitGPUStreamForOperation(operation ? operation : "logits_host_observation");
        if (!host_bridge_stream)
        {
            throw std::runtime_error(
                "GPU logits host observation requires an explicit host-bridge stream");
        }

        if (has_durable_publication &&
            !waitForForwardGraphOutputReady(
                host_bridge_stream,
                DeviceTimelineRole::HostResultBridge,
                required_kind,
                operation))
        {
            throw std::runtime_error(
                "GPU logits host observation could not consume the durable forward event");
        }

        /*
         * A full prefix hit restores the canonical logits row without running
         * a forward graph. Its exact producer edge is therefore the live-state
         * mutation event recorded after the terminal logits/hidden copies.
         * Host observation is read-only: queue a wait on the bridge stream but
         * retain the publication for the target sampler or next main graph.
         */
        if (has_live_prefix_restore_publication &&
            !joinPublishedLiveStateHandoffs(
                host_bridge_stream,
                DeviceTimelineRole::HostResultBridge,
                operation))
        {
            throw std::runtime_error(
                "GPU logits host observation could not observe the prefix-restore event");
        }

        /*
         * Host publication is a read-only observation. It must not steal the
         * one-shot handoff from a sampler, reducer, or state mutation. Grouped
         * verifier replay additionally owns a durable event recorded on the
         * exact replay stream, so a sampler may consume its raw stream token
         * without making a later diagnostic/result D2H race the graph.
         */
        if (has_verifier_event_publication &&
            !waitForPendingAllPositionVerifierStateReadyForObservation(
                host_bridge_stream,
                operation))
        {
            throw std::runtime_error(
                "GPU logits host observation could not wait for verifier readiness");
        }
        if (!has_verifier_event_publication &&
            has_stream_publication &&
            !waitForPendingLogitsStreamForObservation(
                pending_role,
                host_bridge_stream,
                operation))
        {
            throw std::runtime_error(
                "GPU logits host observation could not observe the producer stream handoff");
        }
        return host_bridge_stream;
    }

    const float *DeviceGraphOrchestrator::publishLogitsTensorToHost(
        TensorBase *tensor,
        HostLogitsSurface surface,
        const char *operation) const
    {
        if (!tensor)
            return nullptr;

        void *host_bridge_stream = nullptr;
        if (state_.device_id.is_gpu())
        {
            if (!tensor->deviceValid() || !tensor->gpu_data_ptr())
            {
                std::ostringstream message;
                message << "GPU logits host observation requires device-authoritative storage"
                        << " operation=" << (operation ? operation : "unknown")
                        << " device=" << state_.device_id.toString();
                throw std::runtime_error(message.str());
            }

            auto tensor_device = tensor->current_device();
            if (!tensor_device.has_value() || !tensor_device->is_gpu())
            {
                throw std::runtime_error(
                    "GPU logits host observation has no unambiguous tensor device");
            }

            auto *self = const_cast<DeviceGraphOrchestrator *>(this);
            host_bridge_stream =
                self->prepareLogitsHostObservation(surface, operation);

            /*
             * The host bridge has now joined every producer edge. Publishing
             * the tensor on that stream records the exact completion event that
             * TensorBase::ensureOnHost() consumes for the ordered D2H.
             */
            TransferEngine::publishDeviceWrite(
                tensor,
                *tensor_device,
                host_bridge_stream);
        }

        if (!tensor->ensureOnHost(host_bridge_stream))
        {
            std::ostringstream message;
            message << "Failed to publish logits to host"
                    << " operation=" << (operation ? operation : "unknown")
                    << " device=" << state_.device_id.toString();
            throw std::runtime_error(message.str());
        }
        return tensor->fp32_data();
    }

    bool DeviceGraphOrchestrator::waitForPendingLiveGraphProducersForObservation(
        void *observation_stream,
        const char *observation_name,
        DeviceTimelineRole observation_role) const
    {
        const char *consumer =
            observation_name && observation_name[0] != '\0'
                ? observation_name
                : "live_state_observation";

        /*
         * Ordinary prefill/decode is also a live-state producer. Its event was
         * recorded while the exact graph stream was valid and is owned by this
         * orchestrator rather than the graph cache. Consumers therefore wait on
         * the durable publication directly; they never attempt to record a fresh
         * event on a stream that cache invalidation may already have destroyed.
         */
        if (!waitForForwardGraphOutputReady(
                observation_stream,
                observation_role,
                ForwardGraphOutputKind::Any,
                consumer))
        {
            return reportDeviceTimelineJoinFailure(
                state_.device_id,
                "live_graph_producer_observation",
                "ForwardGraphOutputReady",
                consumer);
        }

        /*
         * Host-visible exports read KV/GDN/MTP state, but they do not decide
         * the lifetime of deferred logits or verifier transactions.  Each wait
         * here therefore observes without consuming; mutation and sampler paths
         * remain the only consumers of their respective handoffs.
         */
        if (!waitForPendingLogitsStreamForObservation(
                PendingLogitsStreamRole::MTPSidecar,
                observation_stream,
                consumer))
        {
            return reportDeviceTimelineJoinFailure(
                state_.device_id,
                "live_graph_producer_observation",
                "MTPSidecarLogitsReady",
                consumer);
        }
        if (!waitForPendingLogitsStreamForObservation(
                PendingLogitsStreamRole::MainDecode,
                observation_stream,
                consumer))
        {
            return reportDeviceTimelineJoinFailure(
                state_.device_id,
                "live_graph_producer_observation",
                "MainDecodeLogitsReady",
                consumer);
        }
        if (!waitForPendingLogitsStreamForObservation(
                PendingLogitsStreamRole::AllPositionVerifier,
                observation_stream,
                consumer))
        {
            return reportDeviceTimelineJoinFailure(
                state_.device_id,
                "live_graph_producer_observation",
                "AllPositionVerifierLogitsReady",
                consumer);
        }
        if (!waitForPendingShiftedMTPKVReadyForObservation(
                observation_stream,
                consumer))
        {
            return reportDeviceTimelineJoinFailure(
                state_.device_id,
                "live_graph_producer_observation",
                "ShiftedMTPKVReady",
                consumer);
        }
        if (!waitForPendingAllPositionVerifierStateReadyForObservation(
                observation_stream,
                consumer))
        {
            return reportDeviceTimelineJoinFailure(
                state_.device_id,
                "live_graph_producer_observation",
                "AllPositionVerifierReady",
                consumer);
        }
        /*
         * Prefix harvest reads the archived terminal row, but observation does
         * not transfer lifetime ownership.  Leave the handoff pending so the
         * next forward or live-state mutation still consumes the same source
         * protection edge.
         */
        if (!waitForPendingMTPPrefillTerminalArchiveReadyForObservation(
                observation_stream,
                consumer))
        {
            return reportDeviceTimelineJoinFailure(
                state_.device_id,
                "live_graph_producer_observation",
                "MTPPrefillTerminalArchiveReady",
                consumer);
        }
        return true;
    }

    void DeviceGraphOrchestrator::clearMTPVerifierTransactionStateForBoundary(const char *reason)
    {
        const char *clear_reason =
            reason && reason[0] != '\0' ? reason : "mtp_verifier_transaction_boundary";

        pending_mtp_spec_verifier_input_plan_.reset();
        pending_mtp_verifier_device_token_plan_.reset();
        pending_mtp_verifier_device_token_batch_plan_.reset();
        materialized_mtp_verifier_device_token_row_ = {};
        materialized_mtp_verifier_device_token_batch_ = {};
        mtp_publication_base_cache_snapshot_ready_ = false;
        mtp_publication_base_cache_snapshot_request_count_ = 0;
        mtp_publication_main_kv_base_checkpoints_ready_ = false;
        mtp_publication_main_kv_base_checkpoint_request_count_ = 0;
        request_batched_prefill_logits_row_count_ = 0;
        greedy_verifier_outcome_graph_transaction_ = {};
        mtp_verifier_outcome_graph_mode_ =
            MTPVerifierOutcomeGraphMode::Disabled;
        std::fill(
            mtp_committed_verifier_identities_host_.begin(),
            mtp_committed_verifier_identities_host_.end(),
            sampling_math::MTPCommittedVerifierIdentityRecord{});

        if (graph_builder_)
        {
            /*
             * Boundary reset returns the runner to an ordinary continuation.
             * All-position and row-indexed verifier mode describe the abandoned
             * speculative verifier transaction, so clear both the builder-owned
             * row list and the local mode mirrors.
             */
            (void)graph_builder_->setRowIndexedAllPositionLogitRows({});
            (void)setComputeRowIndexedAllPositionLogits(false, 0);
            (void)setComputeAllPositionLogits(false);
        }
        else
        {
            compute_all_position_logits_ = false;
            compute_row_indexed_all_position_logits_ = false;
            row_indexed_all_position_logits_row_count_ = 0;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "verifier_transaction_boundary_clears",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"reason", clear_reason}});
    }

    void DeviceGraphOrchestrator::clearLivePrefixRestoreTransientHandoffs(const char *reason)
    {
        const char *clear_reason =
            reason && reason[0] != '\0' ? reason : "restore_live_prefix_state";

        clearAllPendingLogitsStreams(clear_reason);
        clearPendingAllPositionVerifierStateReady();
        clearPendingAcceptedSpecPublicationReady();
        clearPendingLivePrefixCheckpointReady();
        clearPendingMTPPrefillTerminalArchiveReady();

        shifted_mtp_kv_ready_.valid = false;
        shifted_mtp_kv_ready_.producer_stream = nullptr;
        shifted_mtp_kv_ready_.mutation_generation = 0;

        /*
         * A prefix restore replaces the storage contents described by the
         * previous forward publication. Preserve the event allocation, but
         * invalidate its semantic provenance before any restored terminal
         * logits may be sampled. The terminal restore publishes through
         * LivePrefixMutationReady, which the target sampler and diagnostic host
         * result bridge observe directly.
         */
        clearForwardGraphOutputPublications();
        clearCurrentMainLogitsPublication(clear_reason);

        clearMTPVerifierTransactionStateForBoundary(clear_reason);

        defer_next_mtp_main_decode_sync_ = false;
        defer_all_position_verifier_sync_ = false;
        std::fill(stochastic_target_distribution_streams_.begin(),
                  stochastic_target_distribution_streams_.end(),
                  nullptr);
        std::fill(stochastic_draft_distribution_streams_.begin(),
                  stochastic_draft_distribution_streams_.end(),
                  nullptr);
        /*
         * Restoring the verifier-base checkpoint happens between sidecar
         * sampling and verifier replay.  Drop stale sample slots, but preserve
         * slots that were explicitly marked as verifier-owned so the token-row
         * materializer can still order itself after their producer streams.
         */
        clearStochasticTargetSampleReadySlots(
            StochasticSampleReadyClearMode::PreserveVerifierConsumer);
        clearStochasticDraftSampleReadySlots(
            StochasticSampleReadyClearMode::PreserveVerifierConsumer);
        clearDeviceResidentLogicalSequenceStateMailbox();

        PerfStatsCollector::addCounter(
            "mtp",
            "live_prefix_restore_transient_handoff_clears",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"reason", clear_reason}});
    }

    void DeviceGraphOrchestrator::setMTPAllPositionVerifierSyncDeferralEnabled(bool enabled)
    {
        defer_all_position_verifier_sync_ = enabled;
        if (!enabled)
        {
            // Only the request-scoped logits stream token belongs to the
            // deferral toggle. Read-only samplers and host observations leave
            // it published; the transaction boundary clears it after every
            // observer has enqueued its work. Accepted-state publication may
            // still need the verifier-row event after deferral is disabled.
            clearPendingLogitsStream(
                PendingLogitsStreamRole::AllPositionVerifier,
                "setMTPAllPositionVerifierSyncDeferralEnabled(false)");
            std::fill(stochastic_target_distribution_streams_.begin(),
                      stochastic_target_distribution_streams_.end(),
                      nullptr);
            std::fill(stochastic_draft_distribution_streams_.begin(),
                      stochastic_draft_distribution_streams_.end(),
                      nullptr);
            clearStochasticTargetSampleReadySlots();
            clearStochasticDraftSampleReadySlots();
        }
    }

    void DeviceGraphOrchestrator::setMTPMainDecodeSyncDeferralEnabled(bool enabled)
    {
        defer_next_mtp_main_decode_sync_ = enabled;
        if (!enabled)
        {
            clearPendingLogitsStream(
                PendingLogitsStreamRole::MainDecode,
                "setMTPMainDecodeSyncDeferralEnabled(false)");
        }
    }

    bool DeviceGraphOrchestrator::consumeUnusedReplicatedMainLogitsPublication()
    {
        TensorBase *main_logits = requireCurrentMainLogits(
            DeviceLogitsSource::Main,
            "consumeUnusedReplicatedMainLogitsPublication");
        if (!state_.device_id.is_gpu() ||
            current_main_logits_publication_.descriptor
                .isColumnParallelStorage() ||
            !main_logits->deviceValid() ||
            !main_logits->gpu_data_ptr())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Only a GPU runner with replicated "
                      "full main logits may retire an unused main-decode publication");
            return false;
        }

        /*
         * A synchronized cache miss has no stream token to retire. A deferred
         * graph replay does, and consuming that token is the complete semantic
         * boundary because no kernel will read this participant's duplicate row.
         * The durable ForwardGraphOutputReady event still orders the next graph
         * or state mutation; there is deliberately no host wait or no-op kernel.
         */
        void *producer_stream = consumePendingLogitsStream(
            PendingLogitsStreamRole::MainDecode,
            "consumeUnusedReplicatedMainLogitsPublication");
        if (producer_stream)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "unused_replicated_main_logits_publication_consumptions",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"ordering_owner", "forward_graph_output_ready"},
                 {"consumer", "non_primary_rank_participant"}});
        }
        return true;
    }

    bool DeviceGraphOrchestrator::shouldDeferAllPositionVerifierFinalSync() const
    {
        return defer_all_position_verifier_sync_ &&
               compute_all_position_logits_ &&
               state_.device_id.is_gpu() &&
               debugEnv().execution.gpu_graphs &&
               !debugEnv().gpu_stage_timing &&
               !debugEnv().gpu_stage_timing_detail;
    }

    void DeviceGraphOrchestrator::setPendingAllPositionVerifierStream(void *stream)
    {
        if (mtpPublicationDiagnosticsEnabled())
        {
            LOG_INFO(
                "[MTPPublicationDiagnostics] phase=verifier_ready_publish"
                << " device=" << state_.device_id.toString()
                << " producer_stream=" << stream
                << " event="
                << all_position_verifier_state_ready_.event.get());
        }
        publishPendingLogitsStream(
            PendingLogitsStreamRole::AllPositionVerifier,
            stream,
            "setPendingAllPositionVerifierStream");
        if (!stream)
        {
            clearPendingAllPositionVerifierStateReady();
            return;
        }
        if (!recordAllPositionVerifierStateReady(
                stream,
                "setPendingAllPositionVerifierStream"))
        {
            throw std::runtime_error(
                "Failed to record all-position verifier state readiness");
        }
        if (stream)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "all_position_verifier_deferred_stream_handoffs",
                1.0,
                "decode",
                state_.device_id.toString());
        }
    }

    bool DeviceGraphOrchestrator::shouldDeferMainDecodeFinalSync() const
    {
        return defer_next_mtp_main_decode_sync_ &&
               !compute_all_position_logits_ &&
               state_.device_id.is_gpu() &&
               debugEnv().execution.gpu_graphs &&
               !debugEnv().gpu_stage_timing &&
               !debugEnv().gpu_stage_timing_detail;
    }

    void DeviceGraphOrchestrator::setPendingMainDecodeStream(void *stream)
    {
        /*
         * The main-decode stream handoff is one-shot by design.  MTP condition
         * forward is the producer, and the next GPU sampler/distribution build
         * is the only valid consumer.  Clearing the arm flag here prevents a
         * later unrelated decode from inheriting stale replay ordering.
        */
        defer_next_mtp_main_decode_sync_ = false;
        publishPendingLogitsStream(
            PendingLogitsStreamRole::MainDecode,
            stream,
            "setPendingMainDecodeStream");
        if (stream)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "main_decode_deferred_stream_handoffs",
                1.0,
                "decode",
                state_.device_id.toString());
        }
    }

    bool DeviceGraphOrchestrator::supportsMTPSpecStatePublication() const
    {
        if (!graph_builder_)
            return false;

        const auto &mtp = graph_builder_->config().mtp;
        const int requested_rows =
            resolveMTPMaximumExecutionDraftDepth(mtp);
        const MTPVerifierRowCapability capability =
            mtpVerifierRowCapability();
        const auto &config = graph_builder_->config();
        if (config.isMoE() || isPrefixCacheMoEModel())
        {
            return capability.supportsMoEDirectAllPositionRows(
                requested_rows,
                mtp.verify_mode == MTPVerifyMode::SpeculativeSampling);
        }
        return capability.supportsDenseDirectAllPositionRows(
            requested_rows,
            mtp.verify_mode == MTPVerifyMode::SpeculativeSampling);
    }

    MTPVerifierRowCapability DeviceGraphOrchestrator::mtpVerifierRowCapability() const
    {
        MTPVerifierRowCapability capability;
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled ||
            state_.kv_cache == nullptr || state_.mtp_kv_caches.empty())
        {
            return capability;
        }

        constexpr int kPhase97ProvenVerifierRows = 4;
        const auto &config = graph_builder_->config();
        const bool model_is_moe = config.isMoE() || isPrefixCacheMoEModel();
        if (model_is_moe)
        {
            capability.moe_decode_equivalent =
                MTPVerifierRowEquivalenceSpec::proven(kPhase97ProvenVerifierRows);
            /*
             * Direct all-position publication is stronger than grouped row
             * equivalence. The grouped MoE verifier can now produce correct
             * row-local outcomes on CUDA/ROCm, but publishing accepted KV/GDN,
             * shifted MTP KV, terminal hidden/logits, and logical positions is
             * still a separate transaction proof. Keep the direct publication
             * field empty so callers cannot accidentally skip replay-published
             * live-state adoption.
             */
            return capability;
        }

        capability.dense_decode_equivalent =
            MTPVerifierRowEquivalenceSpec::proven(kPhase97ProvenVerifierRows);

        /*
         * Dense direct all-position publication stays fail-closed for every
         * backend until it has a dedicated continuation-equivalence proof.
         * The Phase 9.7 M=1..4 tests prove the shared decode-equivalent path;
         * the broader Qwen3.6 benchmark-prompt regression caught a real
         * GDN/short-conv state mismatch when GPU runners published state
         * directly from all-position verifier rows.
         */
        return capability;
    }

    void DeviceGraphOrchestrator::clearDeviceResidentLogicalSequenceStateMailbox()
    {
        std::unique_lock<std::shared_mutex> admission_lock(
            device_resident_logical_state_admission_sync_->admission_mutex);
        if (pending_device_resident_logical_state_writer_.valid())
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Logical-state mailbox retirement "
                "attempted inside an unfinished replacement transaction"
                << " writer="
                << (pending_device_resident_logical_state_writer_.writer_name &&
                            pending_device_resident_logical_state_writer_
                                    .writer_name[0] != '\0'
                        ? pending_device_resident_logical_state_writer_.writer_name
                        : "unknown"));
            std::terminate();
        }
        /*
         * Logical outcome metadata and shifted-cache ownership overlap during
         * device-resident publication. Metadata preparation retires the previous
         * logical outcome before the initial/suffix shifted-row commits consume
         * the restored cache boundary. Clearing the shifted mailbox here would
         * silently destroy that still-live owner. Request/session reset sites
         * clear both mailboxes explicitly because only those boundaries retire
         * both transactions together.
         */
        if (mtpPublicationDiagnosticsEnabled())
        {
            const auto &mailbox = device_resident_logical_sequence_state_mailbox_;
            LOG_INFO("[MTPPublicationDiagnostics] phase=logical_state_mailbox_clear"
                     << " device=" << state_.device_id.toString()
                     << " valid=" << (mailbox.valid() ? "true" : "false")
                     << " requests=" << mailbox.request_count
                     << " mailbox_epoch=" << mailbox.live_state_epoch
                     << " live_state_epoch=" << live_replay_state_epoch_
                     << " stream=" << mailbox.producer_stream
                     << " event=" << mailbox.ready_event.get());
        }
        device_resident_logical_sequence_state_mailbox_.clear();
    }

    bool DeviceGraphOrchestrator::retireConsumedMTPConditionInputMailbox(
        void *completed_stream,
        uint64_t publication_generation)
    {
        if (!state_.device_id.is_gpu() || !completed_stream ||
            publication_generation == 0)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] MTP-condition mailbox retirement "
                "requires its exact GPU stream and publication generation");
            return false;
        }

        std::unique_lock<std::shared_mutex> admission_lock(
            device_resident_logical_state_admission_sync_->admission_mutex);
        if (pending_device_resident_logical_state_writer_.valid())
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] MTP-condition mailbox "
                "retirement overlapped a replacement writer");
            return false;
        }

        const auto &mailbox =
            device_resident_logical_sequence_state_mailbox_;
        const auto &access_epoch =
            device_resident_logical_state_access_epoch_;
        if (!mailbox.valid() || mailbox.request_count <= 0 ||
            mailbox.live_state_epoch != live_replay_state_epoch_ ||
            mailbox.publication_generation != publication_generation ||
            !access_epoch.valid() ||
            access_epoch.publication_generation !=
                publication_generation)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] MTP condition completed without "
                "its exact transient logical-state publication"
                << " mailbox_valid=" << boolTag(mailbox.valid())
                << " requests=" << mailbox.request_count
                << " mailbox_epoch=" << mailbox.live_state_epoch
                << " live_state_epoch=" << live_replay_state_epoch_
                << " publication_kind="
                << deviceResidentLogicalStatePublicationKindName(
                       mailbox.publication_kind)
                << " publication_generation="
                << mailbox.publication_generation
                << " expected_generation=" << publication_generation
                << " access_generation="
                << access_epoch.publication_generation);
            return false;
        }

        const bool exact_reader_completed = std::any_of(
            access_epoch.reader_completions.begin(),
            access_epoch.reader_completions.begin() +
                static_cast<std::ptrdiff_t>(
                    access_epoch.reader_stream_count),
            [&](const DeviceResidentLogicalStateReaderCompletion &completion)
            {
                return completion.stream == completed_stream &&
                       completion.validFor(
                           publication_generation);
            });
        if (!exact_reader_completed)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] MTP-condition mailbox "
                "retirement found no completion edge for its graph stream"
                << " stream=" << completed_stream
                << " readers=" << access_epoch.reader_stream_count
                << " publication_generation="
                << publication_generation);
            return false;
        }

        const std::string publication_kind =
            deviceResidentLogicalStatePublicationKindName(
                mailbox.publication_kind);
        device_resident_logical_sequence_state_mailbox_.clear();
        device_resident_logical_sequence_state_storage_.retirePublication();
        PerfStatsCollector::addCounterWithSequence(
            "mtp",
            "device_resident_mtp_condition_input_mailbox_retirements",
            1.0,
            {static_cast<uint64_t>(publication_generation)},
            "decode",
            state_.device_id.toString(),
            {{"publication_kind", publication_kind},
             {"next_position_owner", "canonical_device_kv_count"},
             {"ordering", "reader_event_preserved"}});
        return true;
    }

    void DeviceGraphOrchestrator::retireDeviceResidentMTPTransaction()
    {
        if (mtpPublicationDiagnosticsEnabled() &&
            device_resident_mtp_transaction_)
        {
            const auto &transaction = *device_resident_mtp_transaction_;
            LOG_INFO("[MTPPublicationDiagnostics] phase=mtp_transaction_retire"
                     << " device=" << state_.device_id.toString()
                     << " valid=" << (transaction.valid() ? "true" : "false")
                     << " requests=" << transaction.request_count
                     << " depth_count="
                     << transaction.shifted_cached_tokens_device_by_depth.size()
                     << " session_epoch=" << transaction.session_epoch
                     << " mutation_generation="
                     << transaction.mutation_generation
                     << " stream=" << transaction.producer_stream
                     << " event=" << transaction.ready_event.get());
        }
        device_resident_mtp_transaction_.reset();
        device_resident_logical_sequence_state_storage_.retirePublication();
    }

    DeviceResidentMTPTransactionLease
    DeviceGraphOrchestrator::currentDeviceResidentMTPTransactionLease() const
    {
        return DeviceResidentMTPTransactionLease{
            .state = device_resident_mtp_transaction_};
    }

    bool DeviceGraphOrchestrator::recordDeviceResidentMTPTransactionMutation(
        int request_count,
        void *producer_stream,
        const char *producer_name)
    {
        if (!state_.device_id.is_gpu())
            return true;

        const char *owner_name =
            producer_name && producer_name[0] != '\0'
                ? producer_name
                : "unknown";
        if (request_count <= 0 ||
            request_count > state_.batch_size ||
            state_.mtp_kv_caches.empty() ||
            !producer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device MTP transaction mutation has invalid request geometry or stream"
                      << " producer=" << owner_name
                      << " request_count=" << request_count
                      << " batch_size=" << state_.batch_size
                      << " depth_count=" << state_.mtp_kv_caches.size()
                      << " stream=" << producer_stream);
            return false;
        }

        std::vector<const int *> canonical_counts;
        canonical_counts.reserve(state_.mtp_kv_caches.size());
        for (size_t depth = 0; depth < state_.mtp_kv_caches.size(); ++depth)
        {
            IKVCache *cache = state_.mtp_kv_caches[depth].get();
            const int *canonical_count =
                cache ? cache->deviceSequenceCachedTokenCountPtr(0) : nullptr;
            if (!cache ||
                !cache->supportsDeviceResidentSequenceStatePublication() ||
                !canonical_count)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Device MTP transaction cannot bind a canonical shifted-cache count"
                          << " producer=" << owner_name
                          << " depth=" << depth
                          << " cache=" << cache
                          << " canonical_count=" << canonical_count);
                return false;
            }
            canonical_counts.push_back(canonical_count);
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        if (!device_resident_mtp_transaction_ready_event_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device MTP transaction event was not preallocated"
                      << " producer=" << owner_name);
            return false;
        }
        const int device_ordinal = state_.device_id.gpu_ordinal();
        if (!backend->recordEvent(
                device_resident_mtp_transaction_ready_event_.get(),
                device_ordinal,
                producer_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to record device MTP transaction event"
                      << " producer=" << owner_name);
            return false;
        }

        const bool owns_current_session =
            device_resident_mtp_transaction_ &&
            device_resident_mtp_transaction_->session_epoch == session_epoch_ &&
            device_resident_mtp_transaction_->device == state_.device_id &&
            device_resident_mtp_transaction_->request_count == request_count &&
            device_resident_mtp_transaction_
                    ->shifted_cached_tokens_device_by_depth == canonical_counts;
        if (!owns_current_session)
        {
            device_resident_mtp_transaction_ =
                std::make_shared<DeviceResidentMTPTransactionState>();
            device_resident_mtp_transaction_->device = state_.device_id;
            device_resident_mtp_transaction_->request_count = request_count;
            device_resident_mtp_transaction_
                ->shifted_cached_tokens_device_by_depth =
                std::move(canonical_counts);
            device_resident_mtp_transaction_->session_epoch = session_epoch_;
        }

        auto &transaction = *device_resident_mtp_transaction_;
        transaction.producer_stream = producer_stream;
        transaction.ready_event =
            device_resident_mtp_transaction_ready_event_;
        ++transaction.mutation_generation;

        PerfStatsCollector::addCounterWithSequence(
            "mtp",
            "device_resident_mtp_transaction_fences",
            1.0,
            {static_cast<uint64_t>(session_epoch_), static_cast<uint64_t>(transaction.mutation_generation)},
            perfPhaseName(),
            state_.device_id.toString(),
            {{"producer", owner_name},
             {"depth_count", std::to_string(
                                 transaction
                                     .shifted_cached_tokens_device_by_depth
                                     .size())},
             {"requests", std::to_string(request_count)},
             {"new_transaction", boolTag(!owns_current_session)}});
        if (mtpPublicationDiagnosticsEnabled())
        {
            LOG_INFO("[MTPPublicationDiagnostics] phase=mtp_transaction_fence"
                     << " device=" << state_.device_id.toString()
                     << " producer=" << owner_name
                     << " depth_count="
                     << transaction.shifted_cached_tokens_device_by_depth.size()
                     << " requests=" << request_count
                     << " session_epoch=" << transaction.session_epoch
                     << " mutation_generation="
                     << transaction.mutation_generation
                     << " stream=" << producer_stream);
        }
        return transaction.valid();
    }

    bool DeviceGraphOrchestrator::recordRestoredDeviceResidentMTPTransaction(
        void *producer_stream,
        const char *producer_name)
    {
        if (!state_.device_id.is_gpu() || state_.mtp_kv_caches.empty())
            return true;
        return recordDeviceResidentMTPTransactionMutation(
            state_.batch_size,
            producer_stream,
            producer_name && producer_name[0] != '\0'
                ? producer_name
                : "restore_live_prefix_state_shifted_kv");
    }

    bool DeviceGraphOrchestrator::waitForDeviceResidentMTPTransaction(
        const DeviceResidentMTPTransactionLease &lease,
        void *consumer_stream,
        const char *consumer_name) const
    {
        if (!state_.device_id.is_gpu())
            return true;
        const char *name =
            consumer_name && consumer_name[0] != '\0'
                ? consumer_name
                : "unknown";
        if (!lease.valid() ||
            lease.state != device_resident_mtp_transaction_ ||
            lease.state->session_epoch != session_epoch_ ||
            lease.state->device != state_.device_id)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device MTP transaction consumer received a stale or foreign lease"
                      << " consumer=" << name
                      << " lease=" << lease.state.get()
                      << " owner=" << device_resident_mtp_transaction_.get()
                      << " lease_session_epoch="
                      << (lease.state ? lease.state->session_epoch : 0)
                      << " runner_session_epoch=" << session_epoch_);
            return false;
        }
        if (!consumer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device MTP transaction consumer requires an explicit stream"
                      << " consumer=" << name);
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;
        const bool same_stream =
            lease.state->producer_stream == consumer_stream;
        if (!same_stream &&
            !backend->streamWaitEvent(
                consumer_stream,
                lease.state->ready_event.get(),
                state_.device_id.gpu_ordinal()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to queue device MTP transaction wait"
                      << " consumer=" << name);
            return false;
        }

        PerfStatsCollector::addCounterWithSequence(
            "mtp",
            "device_resident_mtp_transaction_waits",
            1.0,
            {static_cast<uint64_t>(lease.state->session_epoch), static_cast<uint64_t>(lease.state->mutation_generation)},
            perfPhaseName(),
            state_.device_id.toString(),
            {{"consumer", name},
             {"same_stream", boolTag(same_stream)},
             {"depth_count", std::to_string(
                                 lease.state
                                     ->shifted_cached_tokens_device_by_depth
                                     .size())}});
        return true;
    }

    std::optional<int> DeviceGraphOrchestrator::deviceResidentLogicalTokenCountForObservation(
        int request_index,
        void *consumer_stream,
        const char *consumer_name) const
    {
        if (!state_.device_id.is_gpu())
            return std::nullopt;

        if (!consumer_stream ||
            request_index < 0 ||
            request_index >= state_.batch_size)
        {
            return std::nullopt;
        }

        DeviceResidentLogicalStateReadScope logical_state_read(
            const_cast<DeviceGraphOrchestrator &>(*this),
            consumer_stream,
            consumer_name);
        if (!logical_state_read.ready())
            return std::nullopt;

        const auto &mailbox = device_resident_logical_sequence_state_mailbox_;
        const bool mailbox_covers_request =
            mailbox.valid() &&
            mailbox.live_state_epoch == live_replay_state_epoch_ &&
            request_index < mailbox.request_count;

        const int32_t *logical_token_count_device = nullptr;
        const char *state_owner = nullptr;
        if (mailbox_covers_request)
        {
            logical_token_count_device =
                mailbox.target_positions_device + request_index;
            state_owner = "logical_state_mailbox";
        }
        else
        {
            if (!logical_state_read.complete())
                return std::nullopt;
            /*
             * Prefix replacement and checkpoint restore deliberately retire
             * the prior outcome mailbox: its condition token and acceptance
             * fields belong to the discarded timeline. The canonical KV count,
             * however, is restored in device memory and remains the authority
             * for the new timeline. Host-visible observation boundaries must
             * read that device row rather than reviving state_.positions.
             *
             * Callers have already waited for the live-state mutation event on
             * consumer_stream. Reading the cache-owned row on that same stream
             * therefore observes the exact restored/truncated count without a
             * device-wide synchronization or a production-path host mirror.
             */
            if (!state_.kv_cache ||
                !state_.kv_cache
                     ->supportsDeviceResidentSequenceStatePublication())
            {
                return std::nullopt;
            }
            logical_token_count_device =
                state_.kv_cache
                    ->deviceSequenceCachedTokenCountPtr(request_index);
            if (!logical_token_count_device)
                return std::nullopt;
            state_owner = "canonical_kv_sequence_count";
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return std::nullopt;
        int32_t value = -1;
        if (!backend->deviceToHostFast(
                &value,
                logical_token_count_device,
                sizeof(value),
                state_.device_id.gpu_ordinal(),
                consumer_stream))
        {
            return std::nullopt;
        }
        if (mailbox_covers_request && !logical_state_read.complete())
            return std::nullopt;
        if (value < 0)
            return std::nullopt;
        if (mtpPublicationDiagnosticsEnabled())
        {
            LOG_INFO("[MTPPublicationDiagnostics] phase=logical_token_count_observation"
                     << " device=" << state_.device_id.toString()
                     << " request=" << request_index
                     << " tokens=" << value
                     << " state_owner=" << state_owner
                     << " consumer=" << (consumer_name && consumer_name[0] != '\0'
                                             ? consumer_name
                                             : "unknown"));
        }
        return static_cast<int>(value);
    }

    std::optional<int> DeviceGraphOrchestrator::deviceResidentShiftedMTPKVTokenCountForObservation(
        int depth,
        int request_index,
        void *consumer_stream,
        const char *consumer_name) const
    {
        if (!state_.device_id.is_gpu())
            return std::nullopt;
        const DeviceResidentMTPTransactionLease transaction =
            currentDeviceResidentMTPTransactionLease();
        if (!transaction.valid() ||
            !transaction.state->coversDepth(depth) ||
            request_index < 0 ||
            request_index >= transaction.state->request_count)
        {
            return std::nullopt;
        }
        if (!waitForDeviceResidentMTPTransaction(
                transaction,
                consumer_stream,
                consumer_name))
        {
            return std::nullopt;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return std::nullopt;
        int32_t value = -1;
        if (!backend->deviceToHostFast(
                &value,
                transaction.state->cachedTokensForDepth(depth) + request_index,
                sizeof(value),
                state_.device_id.gpu_ordinal(),
                consumer_stream))
        {
            return std::nullopt;
        }
        if (value < 0)
            return std::nullopt;
        if (mtpPublicationDiagnosticsEnabled())
        {
            LOG_INFO("[MTPPublicationDiagnostics] phase=shifted_mtp_kv_token_count_observation"
                     << " device=" << state_.device_id.toString()
                     << " request=" << request_index
                     << " depth=" << depth
                     << " tokens=" << value
                     << " consumer=" << (consumer_name && consumer_name[0] != '\0'
                                             ? consumer_name
                                             : "unknown"));
        }
        return static_cast<int>(value);
    }

    bool DeviceGraphOrchestrator::prepareShiftedMTPKVCommitBoundary(
        IKVCache &cache,
        int depth,
        int request_index,
        int expected_cached_tokens,
        bool allow_speculative_discard,
        void *stream,
        const DeviceResidentMTPTransactionLease *transaction,
        const char *operation)
    {
        const char *operation_name =
            operation && operation[0] != '\0' ? operation : "unknown";
        if (depth < 0 || request_index < 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Shifted MTP KV commit boundary received invalid geometry"
                      << " operation=" << operation_name
                      << " depth=" << depth
                      << " request=" << request_index
                      << " expected_cached_tokens=" << expected_cached_tokens);
            return false;
        }

        if (state_.device_id.is_gpu())
        {
            if (!stream)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Device-owned shifted MTP KV commit boundary requires an explicit stream"
                          << " operation=" << operation_name);
                return false;
            }

            if (!transaction ||
                !transaction->valid() ||
                !transaction->coversRequest(request_index))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] GPU shifted MTP KV commit requires an explicit request-scoped transaction lease"
                          << " operation=" << operation_name
                          << " depth=" << depth
                          << " request=" << request_index);
                return false;
            }
            if (!transaction->state->coversDepth(depth))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] GPU MTP transaction does not cover the requested shifted-cache depth"
                          << " operation=" << operation_name
                          << " depth=" << depth
                          << " request=" << request_index);
                return false;
            }
            const int *canonical_count =
                cache.deviceSequenceCachedTokenCountPtr(request_index);
            const int *transaction_count =
                transaction->state->cachedTokensForDepth(depth) +
                request_index;
            if (!cache.supportsDeviceResidentSequenceStatePublication() ||
                !canonical_count ||
                canonical_count != transaction_count)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] GPU MTP transaction does not name the cache's canonical device count"
                          << " operation=" << operation_name
                          << " depth=" << depth
                          << " request=" << request_index
                          << " canonical=" << canonical_count
                          << " transaction=" << transaction_count);
                return false;
            }
            if (!waitForDeviceResidentMTPTransaction(
                    *transaction,
                    stream,
                    operation_name))
            {
                return false;
            }

            PerfStatsCollector::addCounterWithSequence(
                "mtp",
                "device_resident_shifted_mtp_kv_commit_boundaries",
                1.0,
                {static_cast<uint64_t>(transaction->state->session_epoch), static_cast<uint64_t>(transaction->state->mutation_generation)},
                perfPhaseName(),
                state_.device_id.toString(),
                {{"operation", operation_name},
                 {"depth", std::to_string(depth)},
                 {"request", std::to_string(request_index)}});
            return true;
        }

        /* CPU sequence metadata remains synchronous and is validated exactly. */
        if (expected_cached_tokens < 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] CPU shifted MTP KV commit boundary requires an explicit expected count"
                      << " operation=" << operation_name
                      << " depth=" << depth
                      << " request=" << request_index);
            return false;
        }
        int current_cached_tokens =
            cache.get_cached_tokens(cache.first_layer_index(), request_index);
        if (current_cached_tokens > expected_cached_tokens)
        {
            if (!allow_speculative_discard)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Shifted MTP KV commit boundary has unexpected extra rows"
                          << " operation=" << operation_name
                          << " current=" << current_cached_tokens
                          << " expected=" << expected_cached_tokens
                          << " depth=" << depth
                          << " request=" << request_index);
                return false;
            }
            if (!cache.truncateSequence(
                    request_index,
                    expected_cached_tokens,
                    stream))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Shifted MTP KV commit boundary failed to discard speculative rows"
                          << " operation=" << operation_name
                          << " current=" << current_cached_tokens
                          << " expected=" << expected_cached_tokens
                          << " depth=" << depth
                          << " request=" << request_index);
                return false;
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "speculative_shifted_rows_discarded",
                static_cast<double>(current_cached_tokens - expected_cached_tokens),
                perfPhaseName(),
                state_.device_id.toString(),
                {{"operation", operation_name},
                 {"state_owner", "host_bootstrap"}});
            current_cached_tokens = expected_cached_tokens;
        }
        if (current_cached_tokens < expected_cached_tokens)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Shifted MTP KV commit boundary is missing rows"
                      << " operation=" << operation_name
                      << " current=" << current_cached_tokens
                      << " expected=" << expected_cached_tokens
                      << " depth=" << depth
                      << " request=" << request_index);
            return false;
        }
        return true;
    }

    DeviceResidentLogicalSequenceStateHandle
    DeviceGraphOrchestrator::deviceResidentLogicalSequenceState() const
    {
        DeviceResidentLogicalSequenceStateHandle handle;
        const auto &mailbox = device_resident_logical_sequence_state_mailbox_;
        if (!mailbox.valid())
            return handle;
        if (mailbox.live_state_epoch != live_replay_state_epoch_)
            return handle;

        handle.target_positions_device = mailbox.target_positions_device;
        handle.target_sequence_lengths_device =
            mailbox.target_sequence_lengths_device;
        handle.accepted_state_counts_device =
            mailbox.accepted_state_counts_device;
        handle.next_condition_tokens_device =
            mailbox.next_condition_tokens_device;
        handle.all_drafts_accepted_flags_device =
            mailbox.all_drafts_accepted_flags_device;
        handle.stopped_flags_device = mailbox.stopped_flags_device;
        handle.publication_ok_flags_device =
            mailbox.publication_ok_flags_device;
        handle.request_count = mailbox.request_count;
        handle.device = state_.device_id;
        handle.stream = mailbox.producer_stream;
        handle.ready_event = mailbox.ready_event.get();
        handle.live_state_epoch = mailbox.live_state_epoch;
        handle.publication_generation = mailbox.publication_generation;
        handle.mtp_transaction =
            currentDeviceResidentMTPTransactionLease();
        return handle;
    }

    bool DeviceGraphOrchestrator::observeDeviceResidentNextConditionTokens(
        const DeviceResidentLogicalSequenceStateHandle &logical_state,
        int request_count,
        int32_t *out_tokens)
    {
        if (!state_.device_id.is_gpu() ||
            request_count <= 0 ||
            !out_tokens ||
            !logical_state.coversRequest(request_count - 1) ||
            logical_state.device != state_.device_id)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Resident next-condition-token "
                      "observation received a stale, foreign, or incomplete mailbox");
            return false;
        }

        void *stream = explicitGPUStreamForOperation(
            "observeDeviceResidentNextConditionTokens");
        if (!stream)
            return false;
        DeviceResidentLogicalStateReadScope logical_state_read(
            *this,
            stream,
            "observe_device_resident_next_condition_tokens");
        if (!logical_state_read.ready() ||
            !device_resident_logical_sequence_state_mailbox_.ownsHandle(
                logical_state,
                live_replay_state_epoch_))
        {
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend ||
            !backend->deviceToHostFast(
                out_tokens,
                logical_state.next_condition_tokens_device,
                static_cast<size_t>(request_count) * sizeof(int32_t),
                state_.device_id.gpu_ordinal(),
                stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to publish resident "
                      "next-condition-token result rows to host");
            return false;
        }
        if (!logical_state_read.complete())
            return false;

        PerfStatsCollector::addCounter(
            "mtp",
            "device_resident_next_condition_token_host_observations",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"requests", std::to_string(request_count)},
             {"boundary", "host_result"}});
        return true;
    }

    bool DeviceGraphOrchestrator::
        publishDeviceResidentLogicalSequenceStateFromTargetSample(
            int target_sample_slot,
            const int32_t *live_position_device)
    {
        auto fail = [&](const std::string &reason) -> bool
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Device-target logical-state "
                "publication failed: "
                << reason);
            PerfStatsCollector::addCounter(
                "mtp",
                "device_target_logical_state_publication_failures",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"reason", reason}});
            return false;
        };

        if (!state_.device_id.is_gpu())
            return fail("operation requires a GPU runner");
        if (target_sample_slot < 0 ||
            target_sample_slot >= stochastic_target_row_capacity_ ||
            target_sample_slot >=
                static_cast<int>(stochastic_target_sample_ready_.size()))
        {
            return fail("target sample slot is outside the persistent row capacity");
        }
        if (!stochastic_target_sample_tokens_dev_)
            return fail("persistent target sample storage is not bound");
        if (!live_position_device)
            return fail("canonical device KV count is null");

        const auto &target_ready =
            stochastic_target_sample_ready_[
                static_cast<size_t>(target_sample_slot)];
        if (!target_ready.valid || !target_ready.event ||
            !target_ready.producer_stream)
        {
            return fail(
                "target sample has no exact event-backed producer stream");
        }
        void *const producer_stream = target_ready.producer_stream;
        if (!waitForRequiredStochasticTargetSampleReady(
                target_sample_slot,
                producer_stream,
                "device_target_logical_state_publication"))
        {
            return fail("target sample producer could not be ordered");
        }

        const auto *target_token_device =
            static_cast<const int32_t *>(
                stochastic_target_sample_tokens_dev_) +
            target_sample_slot;
        if (!publishDeviceResidentLogicalSequenceStateFromDeviceRows(
                target_token_device,
                live_position_device,
                /*request_count=*/1,
                producer_stream,
                DeviceResidentLogicalStatePublicationKind::
                    TargetSampleInitialization,
                "device_target_sample"))
        {
            return fail("typed device-row publication failed");
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "device_target_logical_state_publications",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"target_slot", std::to_string(target_sample_slot)},
             {"position_owner", "canonical_device_kv_count"},
             {"token_owner", "persistent_target_sample_slot"},
             {"storage_owner", "arena_persistent"}});
        return true;
    }

    bool DeviceGraphOrchestrator::
        publishDeviceResidentLogicalSequenceStateFromDeviceRows(
            const int32_t *condition_token_device,
            const int32_t *live_position_device,
            int request_count,
            void *producer_stream,
            DeviceResidentLogicalStatePublicationKind publication_kind,
            const char *source)
    {
        const char *const source_name =
            source && source[0] != '\0' ? source : "unknown";
        auto fail = [&](const std::string &reason) -> bool
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Device-row logical-state "
                "publication failed"
                << " source=" << source_name << " reason=" << reason);
            PerfStatsCollector::addCounter(
                "mtp",
                "device_row_logical_state_publication_failures",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"source", source_name}, {"reason", reason}});
            return false;
        };

        const auto &storage =
            device_resident_logical_sequence_state_storage_;
        if (!state_.device_id.is_gpu())
            return fail("operation requires a GPU runner");
        if (!condition_token_device || !live_position_device ||
            !producer_stream)
        {
            return fail("source token, position, or producer stream is null");
        }
        if (!storage.validFor(request_count))
            return fail("arena-owned publication rows are incomplete");

        IBackend *const backend = getBackendFor(state_.device_id);
        if (!backend)
            return fail("GPU backend is unavailable");

        /*
         * The mailbox is deliberately single-buffered. Every replacement is
         * ordered after all readers of the prior publication, then initialized
         * on the exact source stream. This is the sole writer transition for
         * sampled targets and restored-prefix request admission alike.
         */
        if (!waitForDeviceResidentLogicalSequenceStateRowReuse(
                producer_stream, source_name))
        {
            return fail("prior mailbox row reuse could not be ordered");
        }
        if (!backend->enqueueInitializeGenerationLogicalState({
                .source = GenerationInitialFrontier::SampledCondition,
                .positions = live_position_device,
                .sampled_tokens = condition_token_device,
                .request_count = request_count,
                .output = {
                    .base_cached_tokens = storage.initialization_base_cached_tokens_scratch_device,
                    .target_positions = storage.target_cached_tokens_device,
                    .accepted_state_counts = storage.accepted_state_counts_device,
                    .next_condition_tokens = storage.next_condition_tokens_device,
                    .all_drafts_accepted_flags = storage.all_drafts_accepted_flags_device,
                    .stopped_flags = storage.stopped_flags_device,
                    .publication_ok_flags = storage.publication_ok_flags_device}},
                state_.device_id.gpu_ordinal(), producer_stream))
        {
            return fail("logical-state initialization kernel launch failed");
        }

        std::string mailbox_error;
        if (!recordDeviceResidentLogicalSequenceStateMailbox(
                request_count,
                producer_stream,
                publication_kind,
                &mailbox_error))
        {
            return fail(
                mailbox_error.empty()
                    ? "mailbox readiness publication failed"
                    : mailbox_error);
        }

        logMTPDeviceTargetLogicalStatePublicationDiagnostics(
            *backend,
            state_.device_id,
            producer_stream,
            live_replay_state_epoch_,
            condition_token_device,
            live_position_device,
            storage.initialization_base_cached_tokens_scratch_device,
            storage.target_cached_tokens_device,
            storage.next_condition_tokens_device,
            storage.publication_ok_flags_device);
        PerfStatsCollector::addCounter(
            "mtp",
            "device_row_logical_state_publications",
            static_cast<double>(request_count),
            "decode",
            state_.device_id.toString(),
            {{"source", source_name},
             {"publication_kind",
              deviceResidentLogicalStatePublicationKindName(
                  publication_kind)},
             {"ordering", "event_published"},
             {"storage_owner", "arena_persistent"}});
        return true;
    }

    const char *DeviceGraphOrchestrator::deviceResidentLogicalStatePublicationKindName(
        DeviceResidentLogicalStatePublicationKind kind)
    {
        switch (kind)
        {
        case DeviceResidentLogicalStatePublicationKind::RequestBatchConditionAdvance:
            return "request_batch_condition_advance";
        case DeviceResidentLogicalStatePublicationKind::TargetSampleInitialization:
            return "target_sample_initialization";
        case DeviceResidentLogicalStatePublicationKind::RestoredPrefixDecodeBridgeInitialization:
            return "restored_prefix_decode_bridge_initialization";
        case DeviceResidentLogicalStatePublicationKind::AcceptedSpecState:
            return "accepted_spec_state";
        case DeviceResidentLogicalStatePublicationKind::MainBatchSampleInitialization:
            return "main_batch_sample_initialization";
        case DeviceResidentLogicalStatePublicationKind::UnsampledLogitsInitialization:
            return "unsampled_logits_initialization";
        case DeviceResidentLogicalStatePublicationKind::OrdinaryGenerationTransaction:
            return "ordinary_generation_transaction";
        case DeviceResidentLogicalStatePublicationKind::OrdinaryGenerationTerminal:
            return "ordinary_generation_terminal";
        case DeviceResidentLogicalStatePublicationKind::ForwardOnlyTerminal:
            return "forward_only_terminal";
        }
        return "unknown";
    }
} // namespace llaminar2
