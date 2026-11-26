/**
 * @file OrchestrationRunnerPrefillAndBatch.cpp
 * @brief OrchestrationRunner prefill, batched prefill, and batched decode execution.
 */

#include "OrchestrationRunnerInternal.h"

namespace llaminar2
{
    // =========================================================================
    // Inference
    // =========================================================================

    bool OrchestrationRunner::leastLoadedCurrentBatchPrefillUsesStableWindows() const
    {
        const auto &placement = config_.moe_routed_expert_plan;
        if (!placement || !placement->enabled)
            return false;

        return std::any_of(
            placement->domains.begin(),
            placement->domains.end(),
            [](const RoutedExpertDomain &domain)
            {
                return domain.routed_prefill_assignment_policy ==
                       RoutedExpertAssignmentPolicy::LeastLoadedResident;
            });
    }

    int OrchestrationRunner::stableRoutedPrefillAssignmentWindowTokens(
        int prefix_cache_block_size,
        bool prefix_cache_enabled) const
    {
        if (!leastLoadedCurrentBatchPrefillUsesStableWindows())
        {
            return 0;
        }

        const int plan_window =
            plan_.runtime.moe_routed_prefill.assignment_window_tokens;
        const int config_window =
            config_.moe_routed_prefill.assignment_window_tokens;
        if (plan_window < 0 || config_window < 0)
        {
            return -1;
        }
        if (plan_window > 0)
        {
            return plan_window;
        }
        if (config_window > 0)
        {
            return config_window;
        }
        if (prefix_cache_enabled)
        {
            return prefix_cache_block_size;
        }
        return 0;
    }

    void OrchestrationRunner::notifyHostMoEOverlayInferenceProgress(
        MoEOverlayInferenceProgressPhase phase,
        std::uint64_t completed_logical_tokens) noexcept
    {
        if (completed_logical_tokens == 0u) return;
        for (const auto &binding : moe_pipeline_stage_bindings_)
            if (const auto &owner = binding->owners().host_maintenance; owner)
                owner->notify({.phase = phase, .completed_logical_tokens = completed_logical_tokens});
        if (!moe_expert_overlay_maintenance_service_ ||
            completed_logical_tokens == 0u)
        {
            return;
        }
        moe_expert_overlay_maintenance_service_->notifyInferenceProgress({
            .phase = phase,
            .completed_logical_tokens = completed_logical_tokens,
        });
    }

    bool OrchestrationRunner::forwardSinglePrefillTransaction(
        const int *tokens,
        int token_count,
        const std::string &failure_message)
    {
        if (!runner_ || !tokens || token_count <= 0)
            return setError(failure_message);

        const auto &exec = debugEnv().execution;
        const auto &overlay_schedule =
            plan_.runtime.overlay_prefill_schedule;
        const ServingGraphPreparationKind serving_graph_kind =
            runner_->servingGraphPreparationKind();
        /*
         * Serving-graph preparation is a setup transition, not a request
         * transition. Distributed ExpertOverlay authenticates the resolved
         * kind once in
         * bindMoEOverlayTransferProgressAndMaterializeServingGraphFamily().
         * Ordinary eager runners need no physical bucket materialization, so
         * an unresolved kind is intentionally equivalent to the generic
         * forwardPrefill() path here. Rechecking setup state per transaction
         * would create a second lifecycle authority and reject otherwise valid
         * host runners before their inference contract can execute.
         */
        const bool local_executes_physical_buckets =
            serving_graph_kind ==
            ServingGraphPreparationKind::NativeDeviceExecutableFamily;
        /*
         * Ordinary runners may derive a bucket from their own graph arena.
         * ExpertOverlay cannot: its next sparse dispatch must be accepted by
         * every remote participant.  Initialization has already frozen a
         * root-authoritative schedule at the minimum admitted capacity, so the
         * request hot path only reads that typed contract.
         */
        const auto buckets = overlay_schedule.enabled()
                                 ? overlay_schedule.bucket_rows
                                 : rawPrefillGraphBucketsForResidentCapacity(
                                       exec.prefill_graph_bucket_sizes,
                                       plan_.runtime.resident_graph_rows,
                                       exec.prefill_graph_min_seq);
        const bool long_bucketed_prefill =
            exec.gpu_graphs &&
            exec.prefill_graph_buckets &&
            !buckets.empty() &&
            token_count > buckets.back();
        const bool overlay_scheduled_prefill =
            exec.gpu_graphs && exec.prefill_graph_buckets &&
            overlay_schedule.enabled() && !buckets.empty() &&
            token_count > 1;
        const bool native_captured_prefill =
            local_executes_physical_buckets && exec.gpu_graphs &&
            exec.prefill_graph_buckets && !buckets.empty();

        /*
         * A native serving family is captured with a device-resident chunk
         * materializer ahead of every prefill graph. Route every request that
         * is mathematically prefill through that same typed scheduler. The
         * caller handles a one-token suffix after a restored prefix as serial
         * decode, because padding it would change arithmetic order. Sending
         * any other short prefill through forwardPrefill() made the engine
         * independently choose the same bucket without the immutable
         * materializer binding, so setup executables could never match runtime.
         */
        if (long_bucketed_prefill || overlay_scheduled_prefill ||
            native_captured_prefill)
        {
            if (!runner_->supportsPrefillChunkSchedule(token_count))
            {
                ++prefill_chunk_stats_.failures;
                return setError(
                    failure_message +
                    " (prefill schedule required by the retained serving graph family, "
                    "but runner does not support prefill chunk scheduling)");
            }

            // Request length is not a fixed physical transaction contract.
            // Every bucket was retained during preparation, so a short
            // nonterminal remainder must use its smaller existing graph too.
            // For example, 448 live rows under a 384-row cap require 384+64,
            // not 384+384. This changes neither row authority nor KV extent.
            const auto policy = PrefillChunkSchedulerPolicy::forRetainedBucketFamily(
                buckets, runner_->get_position(), token_count,
                plan_.runtime.ngram_table_placement && plan_.runtime.ngram_table_placement->fixedHeadPipeline()
                    ? PrefillChunkWidthPolicy::ExactRetained : PrefillChunkWidthPolicy::RetainedPadding);

            PrefillChunkSchedule chunk_schedule = planPrefillChunkSchedule(policy);
            if (!chunk_schedule)
            {
                ++prefill_chunk_stats_.schedules;
                ++prefill_chunk_stats_.failures;
                return setError(failure_message + " (chunk planning failed: " +
                                chunk_schedule.error + ")");
            }

            uint64_t local_padding_rows = 0;
            for (const auto &chunk : chunk_schedule.chunks)
            {
                if (local_executes_physical_buckets)
                {
                    local_padding_rows += static_cast<uint64_t>(
                        std::max(
                            0,
                            chunk.bucket_seq_len - chunk.real_count));
                }
            }

            ++prefill_chunk_stats_.schedules;
            std::uint64_t schedule_fingerprint = 14695981039346656037ull;
            std::int64_t execution_rows = 0;
            for (const auto &chunk : chunk_schedule.chunks)
            {
                /*
                 * The aggregate workload names the continuation graph that
                 * owns inference. A native device graph executes the admitted
                 * bucket, whereas an eager host graph executes only live rows.
                 * The bucket remains in the fingerprint because a retained
                 * expert-only follower authenticates that remote capacity even
                 * when the continuation performs no padded CPU arithmetic.
                 */
                execution_rows += local_executes_physical_buckets
                                      ? chunk.bucket_seq_len
                                      : chunk.real_count;
                schedule_fingerprint = mixInferenceWorkloadScalar(
                    schedule_fingerprint,
                    static_cast<std::uint64_t>(chunk.real_count));
                schedule_fingerprint = mixInferenceWorkloadScalar(
                    schedule_fingerprint,
                    static_cast<std::uint64_t>(chunk.bucket_seq_len));
            }
            if (execution_rows <= 0 ||
                execution_rows > std::numeric_limits<int>::max())
            {
                ++prefill_chunk_stats_.failures;
                return setError(
                    failure_message +
                    " (chunked prefill execution-row identity overflowed)");
            }
            const auto workload = makeMoEOverlayInferenceWorkloadIdentity(
                    ExpertHistogramSource::PrefillChunk,
                    token_count,
                    static_cast<int>(execution_rows),
                    static_cast<int>(chunk_schedule.chunks.size()),
                    0,
                    schedule_fingerprint);
            if (moe_overlay_prefill_interference_owner_ ==
                MoEOverlayPrefillInterferenceOwner::TransactionCoordinator)
            {
                std::string declaration_error;
                if (!moe_overlay_inference_transaction_coordinator_ ||
                    !moe_overlay_inference_transaction_coordinator_
                         ->declarePrefillInterferenceSchedule(
                             workload, &declaration_error))
                {
                    ++prefill_chunk_stats_.failures;
                    return setError(
                        failure_message +
                        " (ExpertOverlay could not declare the complete "
                        "heterogeneous prefill interval: " +
                        declaration_error + ")");
                }
            }
            MoEOverlayInferenceInterferenceScope interference_scope(
                moe_overlay_prefill_interference_owner_ ==
                        MoEOverlayPrefillInterferenceOwner::
                            TransactionCoordinator
                    ? nullptr
                    : moe_expert_overlay_interference_probe_.get(),
                workload);
            if (runner_->forwardPrefillChunkSchedule(
                    tokens,
                    token_count,
                    policy,
                    exec.prefill_graph_pad_token_id,
                    /*allow_padded_execution=*/true))
            {
                ++prefill_chunk_stats_.successful_schedules;
                prefill_chunk_stats_.chunks +=
                    static_cast<uint64_t>(chunk_schedule.chunks.size());
                prefill_chunk_stats_.real_tokens += static_cast<uint64_t>(token_count);
                prefill_chunk_stats_.padded_tokens += local_padding_rows;
                notifyHostMoEOverlayInferenceProgress(
                    MoEOverlayInferenceProgressPhase::Prefill,
                    static_cast<std::uint64_t>(token_count));
                return true;
            }

            ++prefill_chunk_stats_.failures;
            return setError(failure_message + " (chunked prefill failed)");
        }

        int execution_rows = token_count;
        if (local_executes_physical_buckets && exec.gpu_graphs &&
            exec.prefill_graph_buckets && !buckets.empty())
        {
            const auto selected =
                selectPrefillGraphBucket(token_count, buckets);
            if (selected)
                execution_rows = selected.bucket_seq_len;
        }
        const auto workload = makeMoEOverlayInferenceWorkloadIdentity(
                ExpertHistogramSource::PrefillChunk,
                token_count,
                execution_rows,
                1,
                0);
        if (moe_overlay_prefill_interference_owner_ ==
            MoEOverlayPrefillInterferenceOwner::TransactionCoordinator)
        {
            std::string declaration_error;
            if (!moe_overlay_inference_transaction_coordinator_ ||
                !moe_overlay_inference_transaction_coordinator_
                     ->declarePrefillInterferenceSchedule(
                         workload, &declaration_error))
            {
                return setError(
                    failure_message +
                    " (ExpertOverlay could not declare the complete "
                    "heterogeneous prefill interval: " +
                    declaration_error + ")");
            }
        }
        MoEOverlayInferenceInterferenceScope interference_scope(
            moe_overlay_prefill_interference_owner_ ==
                    MoEOverlayPrefillInterferenceOwner::
                        TransactionCoordinator
                ? nullptr
                : moe_expert_overlay_interference_probe_.get(),
            workload);
        if (!runner_->forwardPrefill(tokens, token_count))
            return setError(failure_message);
        notifyHostMoEOverlayInferenceProgress(
            MoEOverlayInferenceProgressPhase::Prefill,
            static_cast<std::uint64_t>(token_count));
        return true;
    }

    bool OrchestrationRunner::forwardPrefillTokens(
        const int *tokens,
        int token_count,
        const std::string &failure_message,
        int stable_segment_tokens)
    {
        if (!runner_ || !tokens || token_count <= 0)
            return setError(failure_message);
        if (stable_segment_tokens < 0)
        {
            return setError(
                failure_message +
                " (stable prefill segment size cannot be negative)");
        }

        if (stable_segment_tokens > 0 && token_count > stable_segment_tokens)
        {
            for (int offset = 0; offset < token_count; offset += stable_segment_tokens)
            {
                const int segment_tokens =
                    std::min(stable_segment_tokens, token_count - offset);
                if (!forwardSinglePrefillTransaction(
                        tokens + offset,
                        segment_tokens,
                        failure_message))
                {
                    return false;
                }
            }
            return true;
        }

        return forwardSinglePrefillTransaction(tokens, token_count, failure_message);
    }

    void OrchestrationRunner::clearBatchedDecodeState()
    {
        batched_decode_active_ = false;
        batched_request_states_.clear();
    }

    bool OrchestrationRunner::activateMoEOverlayDemandAtRequestBoundary()
    {
        try
        {
            for (const auto &binding : moe_pipeline_stage_bindings_)
                if (binding->owners().host_maintenance)
                    (void)binding->owners().authority->activateOptimizationDemandAtRequestBoundary();
            if (!moe_expert_overlay_residency_authority_) return true;
            /*
             * The authority transition is deliberately idempotent. A normal
             * remote rank reaches this helper from its typed command handler
             * and then calls prefill(), whose public admission path reaches it
             * again. Expert-only transaction followers execute only the first
             * call, so both process roles still open exactly the same bank
             * before graph work begins.
             */
            (void)moe_expert_overlay_residency_authority_
                ->activateOptimizationDemandAtRequestBoundary();
            return true;
        }
        catch (const std::exception &error)
        {
            return setError(
                std::string(
                    "ExpertOverlay request-boundary demand activation failed: ") +
                error.what());
        }
    }

    bool OrchestrationRunner::prefill(const std::vector<int32_t> &prompt_tokens)
    {
        if (!initialization_lifecycle_.readyForInference())
        {
            setError("Runner not initialized");
            return false;
        }

        if (prompt_tokens.empty())
        {
            setError("Empty prompt tokens");
            return false;
        }

        if (debugEnv().runtime_debug.mtp_depth_transaction_trace &&
            (!validMTPDepthTransactionTracePolicy(activeMTPRequestConfig()) ||
            !sampling_math::validMTPDepthTransactionTraceAdmission(
                sampling_math::DeviceGenerationAdmissionKind::NewRequest,
                config_.batch_size, config_.n_predict < 0 ? 128 : config_.n_predict)))
            return setError("MTP depth-transaction trace rejected a disabled, zero-depth, unsupported or unbounded request policy");

        /*
         * Delayed ExpertOverlay economy certification may finish while the
         * preceding request still owns decode transactions. Installation
         * therefore leaves routing demand quarantined. `prefill()` is the
         * public request-admission edge shared by HTTP serving, benchmarks,
         * and parity; open the first live generation here before any prefix
         * lookup, follower command, or model graph for the new request runs.
         */
        if (!activateMoEOverlayDemandAtRequestBoundary())
            return false;

        const bool mpi_coordinated_world =
            mpi_coordinated_mode_ && mpi_ctx_ && mpi_ctx_->world_size() > 1;
        const bool mpi_root_command =
            mpi_coordinated_world &&
            mpi_ctx_->rank() == mpi_coordinated_root_rank_;
        ScopedMPIRootCommandFailureAbort prefill_command_failure_abort(
            mpi_ctx_.get(),
            mpi_root_command,
            "PREFILL",
            &last_error_);
        ensureFiniteDevicePlacementArmed();
        ScopedMoEOverlayRootCommand overlay_prefill_command(
            moe_overlay_inference_transaction_coordinator_);

        if (!runner_ ||
            !runner_->configureMTPRequestStopTokens(stop_tokens_))
        {
            return setError(
                "Inference runner rejected request stop-token policy at "
                "request admission");
        }
        if (!runner_->configureMTPRequestPenaltyPolicy(
                MTPRequestPenaltyPolicy{
                    .presence_penalty =
                        active_sampling_params_.presence_penalty,
                    .frequency_penalty =
                        active_sampling_params_.frequency_penalty,
                    .repetition_penalty = active_sampling_params_.repetition_penalty,
                }))
        {
            return setError(
                "Inference runner rejected request penalty policy at "
                "request admission");
        }

        mtp_bypassed_ = false;
        mtp_bypass_recorded_for_request_ = false;
        mtp_bypass_reason_.clear();
        mtp_stats_ = {};
        device_generation_terminal_ledger_authoritative_ = false;
        device_generation_embedded_moe_maintenance_pending_ack_ = false;
        hosted_device_generation_decode_progress_retired_.store(
            0u, std::memory_order_release);
        device_generation_admission_.reset();
        ready_mtp_condition_.reset();
        generation_request_seeds_.reset();
        pending_mtp_condition_token_.reset();
        pending_mtp_condition_params_.reset();
        pending_mtp_condition_resident_state_.reset();
        prelaunched_mtp_first_sidecar_resident_state_.reset();
        prelaunched_mtp_first_sidecar_params_.reset();
        decode_transaction_planning_position_.reset();
        clearBatchedDecodeState();
        last_token_ = prompt_tokens.back();

        const std::uint64_t retired_prefill_tokens =
            moe_overlay_pending_prefill_progress_tokens_.exchange(
                0u, std::memory_order_acq_rel);
        const std::uint64_t retired_decode_tokens =
            moe_overlay_pending_decode_progress_tokens_.exchange(
                0u, std::memory_order_acq_rel);
        if (retired_prefill_tokens >
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::int32_t>::max()) ||
            retired_decode_tokens >
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::int32_t>::max()))
        {
            return setError(
                "ExpertOverlay retired inference progress exceeds the typed prefill command payload");
        }

        /*
         * The next outer command may be another prefill, not necessarily decode.
         * Carry the continuation-authoritative retired progress in every command
         * header so follower controller graphs can join a device-authored epoch
         * before they block on its first inference ticket. Omitting these fields
         * creates a circular wait: the leader waits for a follower snapshot while
         * that follower waits for the new prefill's transaction ticket.
         */
        if (mpi_root_command)
        {
            if (!continuation_request_group_ ||
                continuation_request_group_->role() !=
                    ContinuationRequestGroup::Role::TokenConsumer ||
                prompt_tokens.size() > static_cast<std::size_t>(
                    std::numeric_limits<std::int32_t>::max()))
                return setError("PREFILL has no valid continuation recipient group or prompt wire extent");
            broadcastCommand(MPICommand::PREFILL);
            std::array<std::int32_t, 3> command_header{
                static_cast<std::int32_t>(prompt_tokens.size()),
                static_cast<std::int32_t>(retired_prefill_tokens),
                static_cast<std::int32_t>(retired_decode_tokens),
            };
            mpi_ctx_->broadcast_int32(
                command_header.data(),
                command_header.size(),
                mpi_coordinated_root_rank_);
            // All peers need the header to enter the command/ticket lifecycle.
            // Only continuation owners consume tokens; a sole owner sends none.
            continuation_request_group_->publishPrompt(
                prompt_tokens, mpi_coordinated_root_rank_);
            prefill_command_failure_abort.markPublished();
        }
        if (retired_prefill_tokens != 0u &&
            (!moe_overlay_device_controller_graph_service_ ||
             !moe_overlay_device_controller_graph_service_
                  ->notifyInferenceProgress(
                      MoEOverlayDeviceControllerGraphService::InferencePhase::Prefill,
                      retired_prefill_tokens)))
        {
            return setError(
                "Continuation authority could not publish retired-prefill progress before prefill execution");
        }
        if (retired_decode_tokens != 0u &&
            (!moe_overlay_device_controller_graph_service_ ||
             !moe_overlay_device_controller_graph_service_
                  ->notifyInferenceProgress(
                      MoEOverlayDeviceControllerGraphService::InferencePhase::Decode,
                      retired_decode_tokens)))
        {
            return setError(
                "Continuation authority could not publish retired-decode progress before prefill execution");
        }

        if (overlay_prefill_command.active())
        {
            const auto command_context = moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_ : mpi_ctx_;
            if (!command_context ||
                moe_overlay_collective_generation_id_ == 0 ||
                !moe_expert_overlay_residency_authority_)
            {
                return setError(
                    "ExpertOverlay prefill authority has incomplete root, "
                    "request-generation, or residency ownership");
            }
            const auto snapshot =
                moe_expert_overlay_residency_authority_->snapshot();
            if (!snapshot || !snapshot->valid() ||
                moe_overlay_inference_command_sequence_ ==
                    std::numeric_limits<std::uint64_t>::max())
            {
                return setError(
                    "ExpertOverlay prefill authority has no valid placement "
                    "epoch or command identity");
            }
            const MoEOverlayInferenceCommandIdentity command_identity{
                .request_generation =
                    moe_overlay_collective_generation_id_,
                .command_id =
                    ++moe_overlay_inference_command_sequence_,
                // Initial admitted lower bound only: mapped device packet kernels
                // authenticate their actual current placement ticket and bank.
                .initial_placement_epoch = snapshot->epoch,
            };
            std::string command_error;
            if (!overlay_prefill_command.begin(
                    command_context->rank(), command_identity, &command_error))
            {
                return setError(
                    "ExpertOverlay remote prefill command admission failed: " +
                    command_error);
            }
        }

        const auto complete_distributed_prefill = [&]() -> bool
        {
            if (active_sampling_params_.repetition_penalty != 1.0f &&
                !moe_overlay_inference_transaction_follower_)
            {
                if (runner_->primaryDeviceId().is_gpu())
                {
                    auto unique_tokens = prompt_tokens;
                    std::sort(unique_tokens.begin(), unique_tokens.end());
                    unique_tokens.erase(std::unique(unique_tokens.begin(), unique_tokens.end()), unique_tokens.end());
                    if (!runner_->initializePromptRepetitionHistory(unique_tokens))
                        return setError("Prompt repetition history admission failed");
                }
                else
                    for (int token : prompt_tokens) sampler_.record_prompt_token(token);
            }
            std::string command_error;
            if (!overlay_prefill_command.complete(&command_error))
            {
                return setError(
                    "ExpertOverlay prefill command completion failed: " +
                    command_error);
            }
            prefill_command_failure_abort.markCompleted();
            return true;
        };

        const auto &plan_prefix = plan_.runtime.prefix_cache;
        const auto &config_prefix = config_.prefix_cache;
        const MTPRuntimeConfig active_mtp = activeMTPRequestConfig();
        /*
         * The installed transaction role determines continuation ownership.
         * Expert followers and original layer-stage followers retain their
         * participant-local graph/state families, but only the vocabulary tail
         * owns the sampler and depth controller. Prefix admission still spans
         * every original pipeline rank because all stages must submit the same
         * restored frontier and suffix graph sequence.
         */
        const auto mtp_participation = mtpRankParticipation();
        const auto *original_pipeline =
            dynamic_cast<const GlobalOrchestrator *>(runner_.get());
        const bool coordinates_original_pipeline_prefix =
            original_pipeline && original_pipeline->originalRankPipelineReady();
        if (coordinates_original_pipeline_prefix && !mpi_ctx_)
            return setError("Original pipeline prefix admission has no original MPI context");
        const bool owns_mtp_continuation_authority =
            active_mtp.enabled &&
            mtp_participation != MTPRankParticipation::ExpertTransactionFollower &&
            mtp_participation != MTPRankParticipation::OriginalPipelineFollower;
        const bool prefix_cache_enabled =
            (plan_prefix.enabled || config_prefix.enabled) &&
            plan_prefix.storage_mode != PrefixCacheStorageMode::Disabled &&
            config_prefix.storage_mode != PrefixCacheStorageMode::Disabled;
        const bool mtp_full_hit_requires_terminal_hidden =
            active_mtp.enabled &&
            active_mtp.require_terminal_hidden_for_full_hit;
        prefix_request_summary_ = {};
        prefix_request_summary_.enabled = prefix_cache_enabled;
        prefix_request_summary_.requested_tokens = static_cast<int>(prompt_tokens.size());

        if (owns_mtp_continuation_authority && runner_)
        {
            if (!ensureMTPDepthController(active_mtp))
            {
                return false;
            }
            /*
             * prefill() is the public request-admission boundary even when the
             * prefix cache satisfies every prompt token and no model prefill
             * graph runs.  Adaptive depth observations belong to one request:
             * carrying a promotion, rejection memory, or partial hysteresis
             * window into the next generate() call makes identical seeded
             * requests select different graph depths.
             *
             * clearCache() also resets this controller, but callers are not
             * required to clear the persistent prefix cache between requests.
             * Reset here after configuration so both ordinary admission and
             * exact prefix restore begin from the configured initial depth.
             */
            mtp_depth_controller_->reset();
            const int effective_max_draft_tokens = effectiveMTPMaxDraftDepth(active_mtp);
            if (effective_max_draft_tokens < 1)
            {
                return setError(
                    "MTP decode requires --mtp-draft-tokens >= 1");
            }
            if (effective_max_draft_tokens > 1 && !runner_->supportsChainedMTPDrafts())
            {
                return setError(
                    "MTP decode with --mtp-draft-tokens > 1 requires runner support for chained MTP sidecars");
            }
        }
        else
        {
            mtp_depth_controller_.reset();
        }

        if (prefix_cache_enabled)
        {
            try
            {
                if (continuation_request_group_ &&
                    continuation_request_group_->role() !=
                        ContinuationRequestGroup::Role::TokenConsumer)
                {
                    return setError(
                        "Expert-only rank attempted to enter continuation prefix-cache admission");
                }
                PrefixLookupResult local_hit = runner_->lookupPrefix(prompt_tokens);
                PrefixParticipantLookup participant = makePrefixParticipantLookup(
                    mpi_ctx_ ? mpi_ctx_->rank() : 0,
                    runner_->primaryDeviceId(),
                    local_hit,
                    {},
                    plan_.usesGlobalTP() ||
                            plan_.usesPipelineParallel()
                        ? PrefixFingerprintCoordinationPolicy::
                              ValidateParticipantLocally
                        : PrefixFingerprintCoordinationPolicy::
                              RequireIdentical);

                PrefixCoordinationResult coordination;
                const MPI_Comm prefix_coordination_comm =
                    coordinates_original_pipeline_prefix
                        ? mpi_ctx_->communicator()
                        : (continuation_request_group_
                               ? continuation_request_group_->coordinationCommunicator()
                               : MPI_COMM_NULL);
                if (prefix_coordination_comm != MPI_COMM_NULL)
                {
                    MPIPrefixCollectiveCoordinator domain_coordinator(
                        prefix_coordination_comm);
                    coordination = coordinatePrefixLookups({participant}, &domain_coordinator);
                }
                else
                {
                    coordination = coordinatePrefixLookups({participant});
                }
                const int coordination_block_size =
                    local_hit.block_size > 0
                        ? local_hit.block_size
                        : (plan_prefix.block_size > 0
                               ? plan_prefix.block_size
                               : config_prefix.block_size);
                PrefixLookupResult coordinated_hit =
                    makePrefixLookupResult(coordination, coordination_block_size);
                const bool coordinated_cache_available =
                    coordinated_hit.cache_enabled && coordinated_hit.supported;
                const bool coordinated_prefix_hit =
                    coordinated_cache_available && coordinated_hit.cached_tokens > 0;
                int stable_prefix_prefill_segment_tokens =
                    stableRoutedPrefillAssignmentWindowTokens(
                        coordinated_hit.block_size,
                        coordinated_prefix_hit);
                if (leastLoadedCurrentBatchPrefillUsesStableWindows() &&
                    stable_prefix_prefill_segment_tokens < 0)
                {
                    std::ostringstream error;
                    error << "Routed-prefill assignment window cannot be negative"
                          << " plan_prefill_window="
                          << plan_.runtime.moe_routed_prefill.assignment_window_tokens
                          << " config_prefill_window="
                          << config_.moe_routed_prefill.assignment_window_tokens;
                    return setError(error.str());
                }
                if (coordinated_prefix_hit &&
                    leastLoadedCurrentBatchPrefillUsesStableWindows() &&
                    stable_prefix_prefill_segment_tokens <= 0)
                {
                    std::ostringstream error;
                    error << "Prefix cache least-loaded routed prefill requires a positive stable block size"
                          << " local_hit_block_size=" << local_hit.block_size
                          << " plan_block_size=" << plan_prefix.block_size
                          << " config_block_size=" << config_prefix.block_size
                          << " plan_prefill_window="
                          << plan_.runtime.moe_routed_prefill.assignment_window_tokens
                          << " config_prefill_window="
                          << config_.moe_routed_prefill.assignment_window_tokens;
                    return setError(error.str());
                }
                int matched_tokens = coordinated_hit.cached_tokens;
                LOG_DEBUG("[OrchestrationRunner] Prefix cache lookup summary: local_tokens="
                          << local_hit.cached_tokens
                          << " coordinated_tokens=" << coordinated_hit.cached_tokens
                          << " supported=" << coordinated_hit.supported
                          << " terminal_logits=" << coordinated_hit.has_terminal_logits
                          << " terminal_hidden=" << coordinated_hit.has_terminal_hidden
                          << " requires_terminal_logits=" << coordinated_hit.requires_terminal_logits
                          << " requires_terminal_hidden=" << coordinated_hit.requires_terminal_hidden
                          << " blocks=" << coordinated_hit.blocks.size()
                          << " bypass_reason=" << coordinated_hit.bypass_reason);

                prefill_logits_ready_ = false;
                ready_mtp_condition_.reset();
                pending_mtp_condition_token_.reset();
                pending_mtp_condition_params_.reset();
                pending_mtp_condition_resident_state_.reset();
                prelaunched_mtp_first_sidecar_resident_state_.reset();
                prelaunched_mtp_first_sidecar_params_.reset();
                decode_transaction_planning_position_.reset();

                auto make_common_hit = [&]()
                {
                    PrefixLookupResult hit = local_hit.clampedTo(matched_tokens);
                    hit.cache_enabled = coordinated_hit.cache_enabled;
                    hit.supported = coordinated_hit.supported;
                    hit.fingerprint_key = coordinated_hit.fingerprint_key != 0
                                              ? coordinated_hit.fingerprint_key
                                              : hit.fingerprint_key;
                    hit.placement_epochs = coordinated_hit.placement_epochs;
                    hit.bypass_reason = coordinated_hit.bypass_reason;
                    // Payload availability remains participant-local. Shared
                    // terminal requirements select the boundary below; they
                    // must never manufacture a terminal row on a PP follower.
                    hit.restore_model_runtime_state = matched_tokens > 0;
                    hit.restore_hybrid_state_for_suffix_prefill =
                        matched_tokens > 0 &&
                        matched_tokens < static_cast<int>(prompt_tokens.size());
                    return hit;
                };

                PrefixLookupResult common_hit = make_common_hit();
                prefix_request_summary_.bypassed = !coordinated_hit.supported;
                prefix_request_summary_.bypass_reason = coordinated_hit.bypass_reason;
                prefix_request_summary_.admission_placement_epochs =
                    coordinated_hit.placement_epochs;
                if (active_mtp.enabled &&
                    coordinated_hit.requires_terminal_hidden &&
                    matched_tokens > 0 &&
                    matched_tokens < static_cast<int>(prompt_tokens.size()) &&
                    !coordinated_hit.has_terminal_hidden)
                {
                    const int block_size =
                        common_hit.block_size > 0 ? common_hit.block_size : plan_prefix.block_size;
                    matched_tokens = std::max(0, matched_tokens - std::max(1, block_size));
                    common_hit = make_common_hit();
                }

                if (matched_tokens == static_cast<int>(prompt_tokens.size()) &&
                    !((!coordinated_hit.requires_terminal_logits ||
                       coordinated_hit.has_terminal_logits) &&
                      (!mtp_full_hit_requires_terminal_hidden ||
                       !coordinated_hit.requires_terminal_hidden ||
                       coordinated_hit.has_terminal_hidden)))
                {
                    /*
                     * A full prefix hit can serve decode directly only when the
                     * terminal block carries the logits and, for MTP, the
                     * verifier's terminal hidden row.  Decide that boundary
                     * before importing any cached state so a shortened restore
                     * is a single prefix transaction rather than a populate,
                     * request reset, and second populate.
                     */
                    const int block_size =
                        common_hit.block_size > 0 ? common_hit.block_size : plan_prefix.block_size;
                    matched_tokens = std::max(0, matched_tokens - std::max(1, block_size));
                    common_hit = make_common_hit();
                    LOG_DEBUG("[OrchestrationRunner] Prefix cache full hit lacks terminal state; "
                              "restoring matched_tokens="
                              << matched_tokens
                              << " and pre-filling the suffix");
                }

                auto full_terminal_hit_has_restorable_runtime = [&]() -> bool
                {
                    if (matched_tokens != static_cast<int>(prompt_tokens.size()) ||
                        (common_hit.requires_terminal_logits &&
                         !common_hit.has_terminal_logits) ||
                        (mtp_full_hit_requires_terminal_hidden &&
                         common_hit.requires_terminal_hidden &&
                         !common_hit.has_terminal_hidden) ||
                        common_hit.blocks.empty())
                    {
                        return false;
                    }

                    const auto &terminal_block = common_hit.blocks.back();
                    return terminal_block.has_model_runtime_state &&
                           terminal_block.total_bytes > terminal_block.layout.totalBytes();
                };

                bool full_terminal_runtime_restorable =
                    full_terminal_hit_has_restorable_runtime();
                if (coordinates_original_pipeline_prefix)
                {
                    // A full-hit exemption from stable-window alignment needs
                    // every original stage's own immutable runtime image. A
                    // tail image cannot authenticate an upstream stage image.
                    MPIPrefixCollectiveCoordinator domain_coordinator(
                        prefix_coordination_comm);
                    bool all_runtime_restorable = false;
                    if (!domain_coordinator.allAndBool(
                            full_terminal_runtime_restorable,
                            &all_runtime_restorable))
                    {
                        return setError(
                            "Original pipeline prefix runtime restore coordination failed");
                    }
                    full_terminal_runtime_restorable = all_runtime_restorable;
                }

                if (stable_prefix_prefill_segment_tokens > 0 &&
                    matched_tokens % stable_prefix_prefill_segment_tokens != 0 &&
                    !full_terminal_runtime_restorable)
                {
                    const int original_matched_tokens = matched_tokens;
                    matched_tokens =
                        (matched_tokens / stable_prefix_prefill_segment_tokens) *
                        stable_prefix_prefill_segment_tokens;
                    common_hit = make_common_hit();
                    LOG_INFO("[OrchestrationRunner] Prefix cache LLEP prefill aligned "
                             "matched_tokens="
                             << original_matched_tokens
                             << " down to stable boundary " << matched_tokens
                             << " (stable_block="
                             << stable_prefix_prefill_segment_tokens
                             << "); suffix prefill will recompute the remainder");
                }

                if (coordinates_original_pipeline_prefix)
                {
                    // Lookup already clamped every original stage to one
                    // frontier. Shared terminal/runtime obligations above and
                    // the admitted routing-window policy must preserve that
                    // agreement before any restore or suffix graph is submitted.
                    MPIPrefixCollectiveCoordinator domain_coordinator(
                        prefix_coordination_comm);
                    int common_restore_tokens = 0;
                    if (!domain_coordinator.allMinInt(
                            matched_tokens, &common_restore_tokens))
                    {
                        return setError(
                            "Original pipeline prefix restore frontier coordination failed");
                    }
                    const bool local_frontier_admitted =
                        common_restore_tokens >= 0 &&
                        common_restore_tokens == matched_tokens &&
                        common_hit.cached_tokens == matched_tokens;
                    bool all_frontiers_admitted = false;
                    if (!domain_coordinator.allAndBool(
                            local_frontier_admitted, &all_frontiers_admitted) ||
                        !all_frontiers_admitted)
                    {
                        return setError(
                            "Original pipeline prefix restore frontier disagrees with a stage admission");
                    }
                }

                if (matched_tokens == 0 && stable_prefix_prefill_segment_tokens > 0)
                {
                    stable_prefix_prefill_segment_tokens =
                        stableRoutedPrefillAssignmentWindowTokens(
                            coordinated_hit.block_size,
                            /*prefix_cache_enabled=*/false);
                }

                if (matched_tokens > 0)
                {
                    if (!runner_->populatePrefix(common_hit))
                    {
                        std::ostringstream error;
                        error << "Prefix cache populate failed at matched_tokens="
                              << matched_tokens
                              << "; refusing to downgrade the hit to a miss";
                        return setError(error.str());
                    }
                }
                else
                {
                    /*
                     * Misses with live request state still cross the ordinary
                     * request boundary before prefill.  A fresh runner at
                     * position zero has no KV/GDN/MTP rows to retire, so
                     * manufacturing an extra reset boundary would make the
                     * prefix-cache miss path semantically different from an
                     * uncached first request.  Hits do not reset here:
                     * populatePrefix() owns the prefix restore boundary and
                     * may preserve graph-owned model runtime state long enough
                     * to replay a portable snapshot.
                     */
                    const int current_position = runner_->get_position();
                    if (current_position < 0)
                    {
                        return setError(
                            "Prefix cache miss observed a negative runner position before request reset");
                    }
                    if (current_position > 0)
                    {
                        resetUnderlyingRunnerRequestState("prefix-cache-initial-reset");
                        if (!advanceMoEOverlayCollectiveRequestGeneration(
                                "prefix-cache-initial-reset"))
                        {
                            return false;
                        }
                    }
                }

                if (matched_tokens > 0 && prefixCacheTraceEnabled())
                {
                    const PrefixRuntimeStateSnapshot probe = runner_->prefixStateProbe();
                    LOG_INFO("[PREFIX_TRACE] post-populate prefix state "
                             << summarizePrefixProbeForTrace(probe));
                }

                const auto harvest_schedule = PrefixHarvestSchedule::forPrefill(
                    coordinated_hit, static_cast<int>(prompt_tokens.size()),
                    matched_tokens, stable_prefix_prefill_segment_tokens);
                // Preserve only diagnostic values from the consumed restore
                // chain. The later harvest witness deliberately owns fewer
                // blocks and must not redefine what this request restored.
                const int summary_block_size =
                    common_hit.block_size > 0 ? common_hit.block_size : plan_prefix.block_size;
                const int restored_block_count = !common_hit.blocks.empty()
                    ? static_cast<int>(common_hit.blocks.size())
                    : (summary_block_size > 0 ? matched_tokens / summary_block_size : 0);
                auto restored_metadata = runner_->prefixRestoreMetadata(common_hit);
                if (prefix_coordination_comm != MPI_COMM_NULL)
                {
                    MPIPrefixCollectiveCoordinator domain_coordinator(prefix_coordination_comm);
                    if (!domain_coordinator.allRestoreMetadata(restored_metadata, &restored_metadata))
                        return setError("Prefix restore metadata coordination failed");
                }
                // The restore producer now owns every DMA source through its
                // completion event. Retain just the terminal reuse witness in
                // both local admissions so LRU pressure can reclaim consumed
                // attention blocks and earlier recurrent snapshots.
                local_hit = local_hit.forHarvest(matched_tokens);
                common_hit = common_hit.forHarvest(matched_tokens);
                if (!runner_->preparePrefixHarvest(
                        local_hit, prompt_tokens, harvest_schedule))
                {
                    return setError("Prefix cache prefill archive preparation failed");
                }

                int suffix_start = matched_tokens;
                bool terminal_state_restored = false;

                // Both an imported checkpoint and a newly archived frontier
                // are live prefix state. Advance each interval once, through
                // the same native prefill or one-row decode transaction.
                const auto forward_prefix_interval = [&](int end) -> bool
                {
                    const int suffix_len = end - suffix_start;
                    if (suffix_start > 0 && suffix_len == 1)
                    {
                        /*
                         * A one-row suffix after a restored prefix is a serial
                         * decode boundary, irrespective of whether MTP is
                         * enabled.  Sending it through a padded prefill bucket
                         * changes reduction geometry and therefore violates
                         * byte equivalence with ordinary decode.  MTP needs
                         * the stronger typed bridge because the same graph
                         * must additionally advance depth-zero shifted KV;
                         * the non-MTP transaction owns only main-model state.
                         */
                        const int suffix_token = prompt_tokens[
                            static_cast<size_t>(suffix_start)];
                        std::string sequence_error;
                        if (!overlay_prefill_command.beginRestoredPrefixDecodeSequence(
                                &sequence_error))
                        {
                            return setError(
                                "ExpertOverlay restored-prefix suffix graph-sequence admission failed: " +
                                sequence_error);
                        }
                        const bool suffix_ok =
                            (owns_mtp_continuation_authority ||
                             (dynamic_cast<GlobalOrchestrator *>(runner_.get()) &&
                              static_cast<GlobalOrchestrator *>(runner_.get())->originalRankPipelineReady()))
                                ? runner_->forwardRestoredPrefixMTPDecodeBridge(
                                      {.token_id = suffix_token,
                                       .restored_prefix_tokens =
                                           suffix_start})
                                : runner_->forward(
                                      &suffix_token,
                                      /*seq_len=*/1);
                        if (!suffix_ok)
                        {
                            return setError(
                                owns_mtp_continuation_authority
                                    ? "Forward pass failed during the restored-prefix MTP decode bridge"
                                    : "Forward pass failed during the restored-prefix serial decode bridge");
                        }
                        notifyHostMoEOverlayInferenceProgress(
                            MoEOverlayInferenceProgressPhase::Decode,
                            1u);
                        PerfStatsCollector::addCounter(
                            "forward_graph",
                            "restored_prefix_suffix_decode_transactions",
                            1.0,
                            "decode",
                            runner_->primaryDeviceId().toString(),
                            {{"outer_command", "prefill"},
                             {"mathematical_phase", "decode"},
                             {"logical_rows", "1"},
                             {"state_transition",
                              owns_mtp_continuation_authority
                                  ? "main_and_shifted_mtp"
                                  : "main_only"}});
                    }
                    else if (!forwardPrefillTokens(
                                 prompt_tokens.data() + suffix_start,
                                 suffix_len,
                                 "Forward pass failed during prefix-cache suffix prefill",
                                 stable_prefix_prefill_segment_tokens))
                    {
                        return false;
                    }
                    suffix_start = end;
                    return true;
                };

                for (const int checkpoint : harvest_schedule.reusableCheckpoints())
                {
                    if (!forward_prefix_interval(checkpoint))
                        return false;
                    // Harvest authenticates exactly the tokens already run,
                    // never the still-unexecuted chat-template tail. Saving
                    // here is essential: recurrent state cannot be rewound
                    // from the final prompt checkpoint on the next request.
                    const std::vector<int32_t> checkpoint_tokens(
                        prompt_tokens.begin(), prompt_tokens.begin() + checkpoint);
                    if (!runner_->harvestPrefix(local_hit, checkpoint_tokens, checkpoint))
                        return setError("Prefix cache reusable-boundary harvest failed");
                }

                if (suffix_start < static_cast<int>(prompt_tokens.size()))
                {
                    if (!forward_prefix_interval(static_cast<int>(prompt_tokens.size())))
                        return false;
                    prefill_logits_ready_ = true;
                }
                else if ((!common_hit.requires_terminal_logits ||
                          common_hit.has_terminal_logits) &&
                         (!mtp_full_hit_requires_terminal_hidden ||
                          !common_hit.requires_terminal_hidden ||
                          common_hit.has_terminal_hidden) &&
                         runner_->restorePrefixTerminalState(common_hit))
                {
                    prefill_logits_ready_ = true;
                    terminal_state_restored = true;
                }
                else
                {
                    std::ostringstream error;
                    error << "Prefix cache terminal restore failed at matched_tokens="
                          << matched_tokens
                          << " terminal_logits="
                          << (common_hit.has_terminal_logits ? "yes" : "no")
                          << " terminal_hidden="
                          << (common_hit.has_terminal_hidden ? "yes" : "no")
                          << " mtp_requires_hidden="
                          << (mtp_full_hit_requires_terminal_hidden ? "yes" : "no");
                    return setError(error.str());
                }

                if (prefixCacheTraceEnabled())
                {
                    /*
                     * This is the exact source boundary captured by
                     * harvestPrefix(). Pair it with the post-populate trace
                     * above to prove that restore reproduced every probed
                     * persistent-state byte before suffix decode starts.
                     */
                    const PrefixRuntimeStateSnapshot probe =
                        runner_->prefixStateProbe();
                    LOG_INFO("[PREFIX_TRACE] pre-harvest prefix state "
                             << summarizePrefixProbeForTrace(probe));
                }
                if (!runner_->harvestPrefix(
                        local_hit,
                        prompt_tokens,
                        static_cast<int>(prompt_tokens.size())))
                {
                    return setError(
                        "Prefix cache harvest failed after successful request execution");
                }
                /*
                 * Sample the production movement authority only after every
                 * child has either archived its admitted state or deliberately
                 * discarded a stale admission.  This is request-result
                 * metadata, not a second placement authority: it records why
                 * the next identical request may correctly miss while
                 * background movement remains fully asynchronous.
                 */
                prefix_request_summary_.movement_stages = runner_->prefixMovementStages();
                if (prefix_request_summary_.movement_stages.empty())
                    prefix_request_summary_.completion_movement_epoch = runner_->moeRuntimeMovementEpoch();
                else
                {
                    // Independent stage epochs have no model-wide ordering.
                    // Only their scoped observations can explain this harvest.
                    prefix_request_summary_.admission_placement_epochs = {};
                    prefix_request_summary_.completion_movement_epoch = 0;
                }

                const bool full_hit = matched_tokens == static_cast<int>(prompt_tokens.size());
                prefix_request_summary_.hit = matched_tokens > 0 && full_hit;
                prefix_request_summary_.partial_hit = matched_tokens > 0 && !full_hit;
                prefix_request_summary_.matched_tokens = matched_tokens;
                prefix_request_summary_.matched_blocks = restored_block_count;
                prefix_request_summary_.terminal_logits_restored = terminal_state_restored;
                prefix_request_summary_.terminal_hidden_restored =
                    terminal_state_restored && common_hit.has_terminal_hidden;
                prefix_request_summary_.mtp_state_restored = restored_metadata.hasMTPState();
                prefix_request_summary_.hybrid_state_restored = restored_metadata.hasHybridState();
                prefix_request_summary_.storage_tier = restored_metadata.storageTierName();

                /*
                 * This is request-local diagnostic traffic, not a lifecycle
                 * transition. The typed request summary and PerfStats remain
                 * the production authorities; INFO logging every economy
                 * sample would perturb the workload being measured.
                 */
                LOG_DEBUG("[OrchestrationRunner] Prefix cache request: "
                          << (matched_tokens > 0
                                  ? (full_hit ? "hit" : "partial-hit")
                                  : "miss")
                          << " matched_tokens=" << matched_tokens
                          << " prompt_tokens=" << prompt_tokens.size()
                          << " terminal_logits="
                          << (common_hit.has_terminal_logits ? "yes" : "no"));
                if (!initializeDecodeTransactionPlanningPositionAfterPrefill(
                        static_cast<int>(prompt_tokens.size()),
                        terminal_state_restored
                            ? "prefix_terminal_restore"
                            : (matched_tokens > 0
                                   ? "prefix_suffix_prefill"
                                   : "prefix_cache_miss_prefill")))
                {
                    return false;
                }
                return complete_distributed_prefill();
            }
            catch (const std::exception &e)
            {
                return setError(std::string("Prefill with prefix cache failed: ") + e.what());
            }
        }

        // Run forward pass
        try
        {
            const int stable_llep_prefill_window =
                stableRoutedPrefillAssignmentWindowTokens(
                    /*prefix_cache_block_size=*/0,
                    /*prefix_cache_enabled=*/false);
            if (stable_llep_prefill_window < 0)
            {
                std::ostringstream error;
                error << "Routed-prefill assignment window cannot be negative"
                      << " plan_prefill_window="
                      << plan_.runtime.moe_routed_prefill.assignment_window_tokens
                      << " config_prefill_window="
                      << config_.moe_routed_prefill.assignment_window_tokens;
                return setError(error.str());
            }
            if (!forwardPrefillTokens(prompt_tokens.data(),
                                      static_cast<int>(prompt_tokens.size()),
                                      "Forward pass failed during prefill",
                                      stable_llep_prefill_window))
                return false;
        }
        catch (const std::exception &e)
        {
            return setError(std::string("Prefill failed: ") + e.what());
        }

        // Signal that prefill logits are ready for sampling.
        // The first decodeStep() will sample from these logits directly
        // instead of re-feeding the last prompt token (which would cause
        // the model to see it twice at consecutive positions, corrupting
        // GDN recurrence state and KV cache entries).
        prefill_logits_ready_ = true;

        if (!initializeDecodeTransactionPlanningPositionAfterPrefill(
                static_cast<int>(prompt_tokens.size()),
                "ordinary_prefill"))
        {
            return false;
        }

        return complete_distributed_prefill();
    }

    bool OrchestrationRunner::supportsPrefillBatch(int request_batch) const
    {
        if (!initialization_lifecycle_.readyForInference() || !runner_ || request_batch <= 1)
            return false;

        const MTPRuntimeConfig mtp = activeMTPRequestConfig();
        if (!mtp.enabled || request_batch > mtp.max_request_batch)
            return false;

        const auto &plan_prefix = plan_.runtime.prefix_cache;
        const auto &config_prefix = config_.prefix_cache;
        const bool prefix_cache_enabled =
            (plan_prefix.enabled || config_prefix.enabled) &&
            plan_prefix.storage_mode != PrefixCacheStorageMode::Disabled &&
            config_prefix.storage_mode != PrefixCacheStorageMode::Disabled;
        if (prefix_cache_enabled)
            return false;

        if (plan_.usesLocalPP() ||
            plan_.usesGlobalTP() ||
            plan_.usesPipelineParallel())
        {
            return false;
        }

        if (mpi_ctx_ && mpi_ctx_->world_size() > 1)
            return false;

        return runner_->batch_size() >= request_batch;
    }

    bool OrchestrationRunner::prefillBatch(
        const std::vector<std::vector<int32_t>> &token_batches)
    {
        if (debugEnv().runtime_debug.mtp_depth_transaction_trace)
            return setError("MTP depth-transaction trace requires exactly one logical request");
        if (!initialization_lifecycle_.readyForInference())
            return setError("Runner not initialized");
        if (!runner_)
            return setError("Runner unavailable");

        const int request_batch = static_cast<int>(token_batches.size());
        if (request_batch <= 1)
        {
            return setError(
                "Request-batched prefill requires at least two logical requests");
        }

        const MTPRuntimeConfig mtp = activeMTPRequestConfig();
        if (!mtp.enabled)
            return setError("Request-batched prefill requires MTP to be enabled");
        if (request_batch > mtp.max_request_batch)
        {
            return setError(
                "Request-batched prefill exceeds configured MTP max_request_batch");
        }

        const auto &plan_prefix = plan_.runtime.prefix_cache;
        const auto &config_prefix = config_.prefix_cache;
        const bool prefix_cache_enabled =
            (plan_prefix.enabled || config_prefix.enabled) &&
            plan_prefix.storage_mode != PrefixCacheStorageMode::Disabled &&
            config_prefix.storage_mode != PrefixCacheStorageMode::Disabled;
        if (prefix_cache_enabled)
        {
            return setError(
                "Request-batched prefill with prefix cache requires Phase 9 "
                "common-prefix coordination");
        }

        if (plan_.usesLocalPP() ||
            plan_.usesGlobalTP() ||
            plan_.usesPipelineParallel())
        {
            return setError(
                "Request-batched prefill is currently implemented for "
                "SingleDevice and LocalTP runners");
        }

        if (mpi_ctx_ && mpi_ctx_->world_size() > 1)
        {
            return setError(
                "Request-batched prefill is not enabled for MPI multi-rank runners");
        }

        if (runner_->batch_size() < request_batch)
        {
            return setError(
                "Request-batched prefill exceeds initialized runner batch capacity");
        }

        std::vector<std::vector<int>> converted;
        converted.reserve(token_batches.size());
        std::vector<BatchedDecodeRequestState> next_states;
        next_states.reserve(token_batches.size());
        for (size_t request_index = 0;
             request_index < token_batches.size();
             ++request_index)
        {
            const std::vector<int32_t> &tokens = token_batches[request_index];
            if (tokens.empty())
                return setError("Request-batched prefill received an empty prompt");

            converted.emplace_back(tokens.begin(), tokens.end());

            BatchedDecodeRequestState state;
            state.last_token = tokens.back();
            state.logical_tokens = static_cast<int>(tokens.size());
            state.stochastic_position_seed =
                resolveRequestBatchedStochasticSeed(
                    active_sampling_params_, request_index);
            state.prefill_logits_ready = true;
            state.sampler = Sampler(active_sampling_params_.seed);
            if (!runner_->primaryDeviceId().is_gpu() && active_sampling_params_.repetition_penalty != 1.0f)
                for (int token : tokens) state.sampler.record_prompt_token(token);
            next_states.push_back(std::move(state));
        }

        mtp_bypassed_ = false;
        mtp_bypass_recorded_for_request_ = false;
        mtp_bypass_reason_.clear();
        mtp_stats_ = {};
        device_generation_terminal_ledger_authoritative_ = false;
        device_generation_embedded_moe_maintenance_pending_ack_ = false;
        hosted_device_generation_decode_progress_retired_.store(
            0u, std::memory_order_release);
        device_generation_admission_.reset();
        prefix_request_summary_ = {};
        ready_mtp_condition_.reset();
        prefill_logits_ready_ = false;
        pending_mtp_condition_token_.reset();
        pending_mtp_condition_params_.reset();
        pending_mtp_condition_resident_state_.reset();
        prelaunched_mtp_first_sidecar_resident_state_.reset();
        prelaunched_mtp_first_sidecar_params_.reset();
        decode_transaction_planning_position_.reset();
        last_token_ = next_states.front().last_token;

        if (!runner_->forward_batch(converted))
        {
            device_generation_admission_.reset();
            clearBatchedDecodeState();
            return setError("Forward batch failed during request-batched prefill");
        }
        if (runner_->primaryDeviceId().is_gpu() && active_sampling_params_.repetition_penalty != 1.0f)
        {
            for (size_t request = 0; request < token_batches.size(); ++request)
            {
                auto unique_tokens = token_batches[request];
                std::sort(unique_tokens.begin(), unique_tokens.end());
                unique_tokens.erase(std::unique(unique_tokens.begin(), unique_tokens.end()), unique_tokens.end());
                if (!runner_->initializePromptRepetitionHistory(unique_tokens, static_cast<int>(request)))
                    return setError("Batched prompt repetition history admission failed");
            }
        }
        if (runner_->primaryDeviceId().is_gpu() &&
            !device_generation_admission_.openAfterPrefill())
        {
            clearBatchedDecodeState();
            return setError(
                "Request-batched GPU prefill could not open its unique device-generation admission boundary");
        }

        /*
         * From this point onward the scalar decode state is intentionally
         * invalid. decodeStepBatch() is the only API allowed to consume this
         * request set, because it must advance and publish every request slot
         * under the same ownership transaction.
         */
        batched_request_states_ = std::move(next_states);
        batched_decode_active_ = true;
        return true;
    }

    bool OrchestrationRunner::supportsDecodeStepBatch(int request_batch) const
    {
        if (!initialization_lifecycle_.readyForInference() || !runner_ || request_batch <= 1)
            return false;
        if (!batched_decode_active_)
            return false;
        if (static_cast<int>(batched_request_states_.size()) != request_batch)
            return false;
        if (runner_->vocab_size() <= 0)
            return false;

        bool has_ready_prefill_logits = false;
        bool has_verifier_continuation = false;
        bool has_completed_requests = false;
        for (const BatchedDecodeRequestState &state : batched_request_states_)
        {
            if (state.is_complete)
            {
                has_completed_requests = true;
                continue;
            }
            has_ready_prefill_logits =
                has_ready_prefill_logits || state.prefill_logits_ready;
            has_verifier_continuation =
                has_verifier_continuation || !state.prefill_logits_ready;
        }
        if (has_ready_prefill_logits && has_verifier_continuation)
        {
            /*
             * Mixed ownership is a supported transient transaction: this call
             * flushes each already sampled resident ready token without running
             * model work, then the following call rejoins grouped verification.
             * Advertising false here would make a healthy device-owned state
             * look like a missing implementation to the serving layer.
            */
            return true;
        }
        if (has_ready_prefill_logits)
            return true;

        const MTPRuntimeConfig mtp = activeMTPRequestConfig();
        const int draft_depth = effectiveMTPMaxDraftDepth(mtp);
        /*
         * Greedy sampling is the deterministic specialization of the
         * speculative verifier contract. A deployment may keep the verifier in
         * SpeculativeSampling mode while selecting temperature=0 for one
         * request; that does not require a different grouped graph or outcome
         * publisher. Key the greedy lane from the active sampling contract,
         * rather than rejecting an otherwise executable resident transaction
         * because of the configured stochastic superset.
         */
        const bool greedy_batch_verify =
            active_sampling_params_.is_greedy();
        const bool stochastic_batch_verify =
            mtp.verify_mode == MTPVerifyMode::SpeculativeSampling &&
            !active_sampling_params_.is_greedy() &&
            !active_sampling_params_.has_penalties() &&
            runner_->supportsDeviceStochasticMTPVerification();
        /*
         * CPU request batching may still publish the grouped transaction plan
         * directly.  GPU request batching must publish from compact resident
         * outcome rows and must feed verifier rows from device draft-token slots.
         */
        const bool gpu_request_batch =
            runner_->primaryDeviceId().is_gpu();
        const bool direct_publication_ok =
            !gpu_request_batch &&
            runner_->supportsMTPSpecStatePublication();
        const bool resident_publication_ok =
            gpu_request_batch &&
            runner_->supportsDeviceResidentMTPSpecStatePublication() &&
            runner_->supportsMTPDeviceDraftTokenInput();
        const bool chained_ok =
            draft_depth <= 1 || runner_->supportsChainedMTPDrafts();
        return has_verifier_continuation &&
               !has_completed_requests &&
               mtp.enabled &&
               (greedy_batch_verify || stochastic_batch_verify) &&
               draft_depth >= 1 &&
               chained_ok &&
               (direct_publication_ok || resident_publication_ok);
    }

    GenerationBatchResult OrchestrationRunner::decodeStepBatch(int request_batch)
    {
        GenerationBatchResult batch_result;

        if (!initialization_lifecycle_.readyForInference())
        {
            batch_result.error = "Runner not initialized";
            return batch_result;
        }
        if (!runner_)
        {
            batch_result.error = "Runner unavailable";
            return batch_result;
        }
        if (request_batch <= 1)
        {
            batch_result.error =
                "decodeStepBatch() requires at least two logical requests";
            return batch_result;
        }
        if (!batched_decode_active_)
        {
            batch_result.error =
                "decodeStepBatch() requires a preceding prefillBatch() call";
            return batch_result;
        }
        if (static_cast<int>(batched_request_states_.size()) != request_batch)
        {
            batch_result.error =
                "decodeStepBatch() request count does not match active "
                "request-batched prefill state";
            return batch_result;
        }

        const int vocab = vocabSize();
        if (vocab <= 0)
        {
            batch_result.error = "decodeStepBatch() requires a positive vocabulary size";
            return batch_result;
        }

        const bool gpu_request_batch = runner_->primaryDeviceId().is_gpu();
        const int padded_seq_len = runner_->padded_seq_len();
        if (padded_seq_len <= 0)
        {
            batch_result.error =
                "decodeStepBatch() requires positive request-batch geometry";
            return batch_result;
        }

        std::vector<int> planning_sequence_lengths(
            static_cast<size_t>(request_batch),
            0);
        if (gpu_request_batch)
        {
            for (int request = 0; request < request_batch; ++request)
            {
                const int logical_tokens =
                    batched_request_states_[static_cast<size_t>(request)]
                        .logical_tokens;
                if (logical_tokens <= 0)
                {
                    batch_result.error =
                        "decodeStepBatch() GPU request batch has invalid immutable prefill geometry";
                    return batch_result;
                }
                planning_sequence_lengths[static_cast<size_t>(request)] =
                    logical_tokens;
            }
        }
        else
        {
            const std::vector<int> &sequence_lengths =
                runner_->sequence_lengths();
            if (static_cast<int>(sequence_lengths.size()) < request_batch)
            {
                batch_result.error =
                    "decodeStepBatch() CPU request batch requires per-request sequence metadata";
                return batch_result;
            }
            std::copy_n(
                sequence_lengths.begin(),
                request_batch,
                planning_sequence_lengths.begin());
        }
        DeviceResidentLogicalSequenceStateHandle resident_batch_state =
            runner_->deviceResidentLogicalSequenceState();
        if (!gpu_request_batch && resident_batch_state.valid())
        {
            bool shadows_valid = true;
            for (int request = 0; request < request_batch; ++request)
            {
                const int logical_tokens =
                    batched_request_states_[static_cast<size_t>(request)]
                        .logical_tokens;
                if (logical_tokens < 0)
                {
                    shadows_valid = false;
                    break;
                }
                planning_sequence_lengths[static_cast<size_t>(request)] =
                    logical_tokens;
            }
            if (!shadows_valid)
            {
                batch_result.error =
                    "decodeStepBatch() has resident logical state but missing "
                    "transaction-derived request positions";
                return batch_result;
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "request_batch_planning_resident_shadow_reads",
                1.0,
                "decode",
                {},
                {{"request_count", std::to_string(request_batch)}});
        }

        batch_result.requests.resize(static_cast<size_t>(request_batch));
        bool initial_resident_prefill_transaction = false;

        bool has_ready_prefill_logits = false;
        bool has_verifier_continuation = false;
        bool has_completed_requests = false;
        for (int request = 0; request < request_batch; ++request)
        {
            const BatchedDecodeRequestState &state =
                batched_request_states_[static_cast<size_t>(request)];
            if (state.is_complete)
            {
                has_completed_requests = true;
                continue;
            }
            has_ready_prefill_logits =
                has_ready_prefill_logits || state.prefill_logits_ready;
            has_verifier_continuation =
                has_verifier_continuation || !state.prefill_logits_ready;
        }

        if (has_ready_prefill_logits && has_verifier_continuation)
        {
            if (gpu_request_batch)
            {
                batch_result.error =
                    "decodeStepBatch() GPU request batch reached mixed host-stepped state; complete resident generation must terminate in one parent launch";
                return batch_result;
            }
            /*
             * A mixed batch is a normal speculative-decode outcome.  A lane
             * that accepted every draft owns a token sampled from its terminal
             * verifier row, while a rejecting lane owns a correction condition
             * and must enter another verifier transaction.  Returning the ready
             * token at the end of the previous call made that lane produce one
             * more response token than scalar decode.
             *
             * Preserve scalar API boundaries by flushing only already-sampled
             * ready tokens on this call.  Lanes without a ready token remain
             * unchanged and return an empty response for this turn.  After the
             * flush every live lane owns an unforwarded condition token, so the
             * next call rejoins the single grouped, device-resident verifier
             * path without serial replay or host-side model execution.
             */
            for (int request = 0; request < request_batch; ++request)
            {
                BatchedDecodeRequestState &state =
                    batched_request_states_[static_cast<size_t>(request)];
                GenerationResult &request_result =
                    batch_result.requests[static_cast<size_t>(request)];
                if (state.is_complete)
                {
                    request_result.is_complete = true;
                    continue;
                }
                if (!state.prefill_logits_ready)
                    continue;
                if (!state.ready_sampled_token.has_value() ||
                    !state.ready_sampled_params.has_value())
                {
                    batch_result.error =
                        "decodeStepBatch() mixed ready lane is missing its "
                        "device-sampled token transaction";
                    return batch_result;
                }
                if (!samplingParamsEqual(
                        *state.ready_sampled_params,
                        active_sampling_params_))
                {
                    batch_result.error =
                        "decodeStepBatch() mixed ready token was sampled with "
                        "different sampling parameters";
                    return batch_result;
                }

                const int32_t token = *state.ready_sampled_token;
                state.prefill_logits_ready = false;
                state.ready_sampled_token.reset();
                state.ready_sampled_params.reset();
                state.last_token = token;
                state.sampler.record_token(token);

                request_result.tokens.push_back(token);
                request_result.is_complete =
                    std::find(stop_tokens_.begin(), stop_tokens_.end(), token) !=
                    stop_tokens_.end();
                state.is_complete = request_result.is_complete;
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "request_batch_mixed_ready_token_flushes",
                1.0,
                "decode",
                {},
                {{"request_count", std::to_string(request_batch)}});
            return batch_result;
        }

        if (has_ready_prefill_logits && gpu_request_batch)
        {
            if (has_verifier_continuation || has_completed_requests)
            {
                batch_result.error =
                    "decodeStepBatch() GPU prefill admission requires every request row to enter transaction zero together";
                return batch_result;
            }
            for (const BatchedDecodeRequestState &state :
                 batched_request_states_)
            {
                if (!state.prefill_logits_ready ||
                    state.ready_sampled_token.has_value() ||
                    state.ready_sampled_params.has_value())
                {
                    batch_result.error =
                        "decodeStepBatch() GPU prefill boundary contains a forbidden host-sampled token shadow";
                    return batch_result;
                }
            }
            if (!admitRequestBatchDeviceResidentGeneration(request_batch))
            {
                batch_result.error =
                    last_error_.empty()
                        ? "decodeStepBatch() could not admit request-batched device generation"
                        : last_error_;
                return batch_result;
            }

            std::vector<uint64_t> position_seeds;
            if (!active_sampling_params_.is_greedy())
            {
                position_seeds.reserve(static_cast<size_t>(request_batch));
                for (const BatchedDecodeRequestState &state :
                     batched_request_states_)
                {
                    position_seeds.push_back(state.stochastic_position_seed);
                }
            }
            if (!runner_
                     ->publishMainLogitsBatchSamplesToDeviceResidentState(
                         request_batch,
                         active_sampling_params_,
                         position_seeds.empty() ? nullptr
                                                : position_seeds.data()))
            {
                batch_result.error =
                    "decodeStepBatch() could not publish request-batched prefill samples into resident state";
                return batch_result;
            }

            resident_batch_state =
                runner_->deviceResidentLogicalSequenceState();
            if (!resident_batch_state.valid() ||
                resident_batch_state.request_count != request_batch)
            {
                batch_result.error =
                    "decodeStepBatch() request-batched prefill sampler produced no complete resident logical-state mailbox";
                return batch_result;
            }
            for (BatchedDecodeRequestState &state :
                 batched_request_states_)
            {
                state.prefill_logits_ready = false;
            }
            initial_resident_prefill_transaction = true;
            has_ready_prefill_logits = false;
            has_verifier_continuation = true;
            PerfStatsCollector::addCounter(
                "mtp",
                "request_batch_prefill_to_resident_transaction_zero",
                1.0,
                "decode",
                {},
                {{"request_count", std::to_string(request_batch)},
                 {"host_token_materializations", "0"}});
        }

        if (has_ready_prefill_logits)
        {
            for (int request = 0; request < request_batch; ++request)
            {
                BatchedDecodeRequestState &state =
                    batched_request_states_[static_cast<size_t>(request)];
                GenerationResult &request_result =
                    batch_result.requests[static_cast<size_t>(request)];

                if (state.is_complete)
                {
                    request_result.is_complete = true;
                    continue;
                }
                if (!state.prefill_logits_ready)
                {
                    batch_result.error =
                        "decodeStepBatch() expected every active request to "
                        "own terminal prefill logits";
                    return batch_result;
                }

                int token = kMTPSpecDecodeInvalidToken;
                if (state.ready_sampled_token.has_value())
                {
                    if (!state.ready_sampled_params.has_value())
                    {
                        batch_result.error =
                            "decodeStepBatch() ready token is missing the "
                            "sampling parameters that produced it";
                        return batch_result;
                    }
                    if (!samplingParamsEqual(
                            *state.ready_sampled_params,
                            active_sampling_params_))
                    {
                        batch_result.error =
                            "decodeStepBatch() ready token was sampled with "
                            "different sampling parameters";
                        return batch_result;
                    }
                    token = *state.ready_sampled_token;
                }
                else
                {
                    const int logical_length =
                        planning_sequence_lengths[static_cast<size_t>(request)];
                    if (logical_length <= 0 || logical_length > padded_seq_len)
                    {
                        batch_result.error =
                            "decodeStepBatch() received invalid per-request sequence length";
                        return batch_result;
                    }

                    const float *sequence_logits = runner_->getLogits(request);
                    if (!sequence_logits)
                    {
                        batch_result.error =
                            "decodeStepBatch() could not access per-request logits";
                        return batch_result;
                    }

                    const float *terminal_logits =
                        sequence_logits +
                        static_cast<size_t>(logical_length - 1) *
                            static_cast<size_t>(vocab);
                    token = sampleHostLogitsAtLogicalPosition(
                        state.sampler,
                        terminal_logits,
                        static_cast<size_t>(vocab),
                        active_sampling_params_, logical_length);
                }

                state.prefill_logits_ready = false;
                state.ready_sampled_token.reset();
                state.ready_sampled_params.reset();
                state.last_token = token;

                request_result.tokens.push_back(token);
                request_result.is_complete =
                    std::find(stop_tokens_.begin(), stop_tokens_.end(), token) !=
                    stop_tokens_.end();
                state.is_complete = request_result.is_complete;

                state.sampler.record_token(token);
            }

            return batch_result;
        }

        if (!has_verifier_continuation)
            return batch_result;
        if (has_completed_requests)
        {
            batch_result.error =
                "decodeStepBatch() request-batched verifier continuation "
                "requires every request lane to be active";
            return batch_result;
        }

        const MTPRuntimeConfig mtp = activeMTPRequestConfig();
        if (!mtp.enabled)
        {
            batch_result.error =
                "decodeStepBatch() MTP verifier continuation requires MTP";
            return batch_result;
        }
        /*
         * Keep execution symmetric with supportsDecodeStepBatch(): greedy
         * sampling uses the deterministic branch of either verifier mode.
         * SpeculativeSampling remains relevant only for non-greedy sampling,
         * where device-resident probability rejection is required.
         */
        const bool greedy_batch_verify =
            active_sampling_params_.is_greedy();
        const bool stochastic_batch_verify =
            mtp.verify_mode == MTPVerifyMode::SpeculativeSampling &&
            !active_sampling_params_.is_greedy();
        if (!greedy_batch_verify && !stochastic_batch_verify)
        {
            batch_result.error =
                "decodeStepBatch() request-batched verifier continuation "
                "requires greedy verification or stochastic speculative sampling";
            return batch_result;
        }
        if (stochastic_batch_verify && active_sampling_params_.has_penalties())
        {
            batch_result.error =
                "decodeStepBatch() request-batched verifier continuation "
                "does not yet support penalty-mutated stochastic sampling";
            return batch_result;
        }
        if (stochastic_batch_verify &&
            !runner_->supportsDeviceStochasticMTPVerification())
        {
            batch_result.error =
                "decodeStepBatch() stochastic request batching requires "
                "device-resident stochastic MTP verification";
            return batch_result;
        }
        if (greedy_batch_verify && active_sampling_params_.has_penalties())
        {
            batch_result.error =
                "decodeStepBatch() request-batched verifier continuation "
                "does not yet support penalty-mutated greedy sampling";
            return batch_result;
        }
        const int draft_depth = effectiveMTPMaxDraftDepth(mtp);
        if (draft_depth < 1)
        {
            batch_result.error =
                "decodeStepBatch() request-batched verifier continuation "
                "requires --mtp-draft-tokens >= 1";
            return batch_result;
        }
        if (draft_depth > 1 && !runner_->supportsChainedMTPDrafts())
        {
            batch_result.error =
                "decodeStepBatch() request-batched verifier continuation "
                "requires batched chained MTP draft support for depth > 1";
            return batch_result;
        }
        /*
         * CPU request batches can publish the grouped transaction plan directly.
         * GPU request batches publish compact resident verifier outcomes and feed
         * verifier inputs from device draft-token slots; the direct host-plan
         * publisher is intentionally not considered a GPU continuation path.
         */
        const bool request_batch_direct_publication_ok =
            !gpu_request_batch &&
            runner_->supportsMTPSpecStatePublication();
        const bool request_batch_resident_publication_ok =
            gpu_request_batch &&
            runner_->supportsDeviceResidentMTPSpecStatePublication() &&
            runner_->supportsMTPDeviceDraftTokenInput();
        if (!request_batch_direct_publication_ok &&
            !request_batch_resident_publication_ok)
        {
            batch_result.error =
                "decodeStepBatch() request-batched verifier continuation "
                "requires CPU grouped publication or GPU device-resident "
                "compact publication with device draft-token input";
            return batch_result;
        }

        if (!runner_->ensureMTPCheckpointTerminalHidden())
        {
            batch_result.error =
                "decodeStepBatch() could not materialize MTP checkpoint terminal hidden";
            return batch_result;
        }
        /*
         * Request-batched planning_sequence_lengths is the scheduler-owned
         * logical cursor set.  The current rollback API archives sequence zero;
         * pass that lane's exact cursor explicitly so a GPU child never consults
         * a lagging host mirror while sizing the device-resident checkpoint.
         */
        PrefixStateSnapshot checkpoint =
            runner_->captureLivePrefixCheckpoint(
                PrefixCheckpointCaptureRequest{
                    .sequence_index = 0,
                    .logical_cached_tokens =
                        planning_sequence_lengths.front(),
                    .maximum_main_append_tokens = draft_depth + 2,
                    .maximum_shifted_append_tokens = draft_depth + 1});
        if (!checkpoint.valid)
        {
            batch_result.error =
                "decodeStepBatch() could not capture live prefix checkpoint";
            return batch_result;
        }

        auto fail_after_checkpoint =
            [&](const std::string &message) -> GenerationBatchResult
        {
            runner_->setComputeAllPositionLogits(false);
            runner_->setComputeRowIndexedAllPositionLogits(false, 0);
            std::string restored_suffix;
            if (!runner_->restoreLivePrefixState(checkpoint))
                restored_suffix = "; checkpoint restore failed";
            batch_result.error = message + restored_suffix;
            return batch_result;
        };

        bool request_batch_condition_advanced = false;
        if (gpu_request_batch && !initial_resident_prefill_transaction)
        {
            std::vector<uint64_t> condition_position_seeds;
            if (stochastic_batch_verify)
            {
                condition_position_seeds.resize(
                    static_cast<size_t>(request_batch),
                    0);
                for (int request = 0; request < request_batch; ++request)
                {
                    condition_position_seeds[static_cast<size_t>(request)] =
                        batched_request_states_[static_cast<size_t>(request)]
                            .stochastic_position_seed;
                }
            }

            /*
             * Match scalar decode's transaction boundary before drafting. The
             * main graph consumes the token returned by the preceding call,
             * samples the first new target token, and republishes that token as
             * the MTP sidecar condition. Mutable token identities and logical
             * positions never cross the host while this transition executes.
             */
            if (!runner_->advanceMTPRequestBatchConditionOnDevice(
                    resident_batch_state,
                    request_batch,
                    active_sampling_params_,
                    condition_position_seeds.empty()
                        ? nullptr
                        : condition_position_seeds.data()))
            {
                return fail_after_checkpoint(
                    "decodeStepBatch() could not advance the resident main "
                    "condition batch before MTP drafting");
            }
            resident_batch_state =
                runner_->deviceResidentLogicalSequenceState();
            if (!resident_batch_state.valid() ||
                resident_batch_state.request_count < request_batch)
            {
                return fail_after_checkpoint(
                    "decodeStepBatch() condition advance did not publish a "
                    "complete resident logical-state mailbox");
            }
            for (int request = 0; request < request_batch; ++request)
            {
                ++planning_sequence_lengths[static_cast<size_t>(request)];
            }
            request_batch_condition_advanced = true;
            PerfStatsCollector::addCounter(
                "mtp",
                "request_batch_condition_forward_transactions",
                1.0,
                "decode",
                {},
                {{"requests", std::to_string(request_batch)},
                 {"sampling",
                  stochastic_batch_verify ? "stochastic" : "greedy"},
                 {"state_owner", "device_logical_state_mailbox"}});
        }

        std::vector<int32_t> condition_tokens;
        std::vector<int> position_ids;
        std::vector<int32_t> sidecar_drafts;
        std::vector<std::vector<int32_t>> request_drafts(
            static_cast<size_t>(request_batch));
        const bool use_request_batch_device_draft_slots =
            gpu_request_batch &&
            runner_->supportsMTPDeviceDraftTokenInput();
        const bool use_request_batch_resident_condition_tokens =
            gpu_request_batch &&
            resident_batch_state.valid() &&
            resident_batch_state.request_count >= request_batch;
        if (gpu_request_batch &&
            (!use_request_batch_device_draft_slots ||
             !use_request_batch_resident_condition_tokens))
        {
            return fail_after_checkpoint(
                "decodeStepBatch() GPU request-batched MTP requires a live "
                "device-resident logical-state mailbox and device draft slots");
        }
        if (!use_request_batch_device_draft_slots)
        {
            condition_tokens.assign(
                static_cast<size_t>(request_batch),
                kMTPSpecDecodeInvalidToken);
            position_ids.assign(
                static_cast<size_t>(request_batch),
                -1);
            sidecar_drafts.assign(
                static_cast<size_t>(request_batch),
                kMTPSpecDecodeInvalidToken);
            for (int request = 0; request < request_batch; ++request)
            {
                const BatchedDecodeRequestState &state =
                    batched_request_states_[static_cast<size_t>(request)];
                if (state.is_complete)
                {
                    batch_result.requests[static_cast<size_t>(request)].is_complete = true;
                    continue;
                }
                const int logical_length =
                    planning_sequence_lengths[static_cast<size_t>(request)];
                if (logical_length <= 0)
                {
                    batch_result.error =
                        "decodeStepBatch() received invalid per-request sequence length";
                    return batch_result;
                }
                condition_tokens[static_cast<size_t>(request)] = state.last_token;
                position_ids[static_cast<size_t>(request)] = logical_length;
            }

            /*
             * CPU grouped sidecars still consume explicit host rows. Completed
             * rows receive a harmless value solely to preserve the dense grouped
             * shape; the GPU lane has no corresponding host token or position row.
             */
            for (int request = 0; request < request_batch; ++request)
            {
                if (condition_tokens[static_cast<size_t>(request)] ==
                    kMTPSpecDecodeInvalidToken)
                {
                    condition_tokens[static_cast<size_t>(request)] = 0;
                    position_ids[static_cast<size_t>(request)] =
                        std::max(
                            0,
                            planning_sequence_lengths[static_cast<size_t>(request)]);
                }
            }
        }
        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                use_request_batch_device_draft_slots
                    ? "request_batch_sidecar_forward_device_drafts"
                    : "request_batch_sidecar_forward",
                "decode");
            const bool sidecar_ok =
                use_request_batch_device_draft_slots
                    ? runner_->forwardMTPBatchFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlots(
                          resident_batch_state,
                          request_batch,
                          /*first_draft_slot=*/0,
                          /*slot_stride=*/draft_depth)
                    : runner_->forwardMTPBatchAndSampleGreedy(
                          condition_tokens.data(),
                          position_ids.data(),
                          request_batch,
                          sidecar_drafts.data());
            if (!sidecar_ok)
            {
                return fail_after_checkpoint(
                    use_request_batch_device_draft_slots
                        ? "decodeStepBatch() request-batched MTP sidecar could not produce device draft slots"
                        : "decodeStepBatch() request-batched MTP sidecar failed");
            }
        }
        if (!use_request_batch_device_draft_slots)
        {
            for (int request = 0; request < request_batch; ++request)
            {
                const int32_t draft = sidecar_drafts[static_cast<size_t>(request)];
                if (draft < 0)
                {
                    return fail_after_checkpoint(
                        "decodeStepBatch() request-batched MTP sidecar produced "
                        "an invalid first draft token");
                }
                request_drafts[static_cast<size_t>(request)].push_back(draft);
            }
        }

        for (int draft_index = 1; draft_index < draft_depth; ++draft_index)
        {
            if (!use_request_batch_device_draft_slots)
            {
                for (int request = 0; request < request_batch; ++request)
                {
                    condition_tokens[static_cast<size_t>(request)] =
                        request_drafts[static_cast<size_t>(request)].back();
                    position_ids[static_cast<size_t>(request)] =
                        planning_sequence_lengths[static_cast<size_t>(request)] +
                        draft_index;
                    sidecar_drafts[static_cast<size_t>(request)] =
                        kMTPSpecDecodeInvalidToken;
                }
            }

            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "request_batch_chained_sidecar_forward",
                "decode",
                {},
                {{"draft_index", std::to_string(draft_index)}});
            const bool chained_sidecar_ok =
                use_request_batch_device_draft_slots
                    ? runner_->forwardMTPBatchFromDeviceDraftSlotsAndSampleGreedyToDeviceDraftSlots(
                          resident_batch_state,
                          request_batch,
                          /*first_condition_slot=*/draft_index - 1,
                          /*condition_slot_stride=*/draft_depth,
                          /*position_offset=*/draft_index,
                          /*first_draft_slot=*/draft_index,
                          /*draft_slot_stride=*/draft_depth)
                    : runner_->forwardMTPBatchFromLastDraftAndSampleGreedy(
                          condition_tokens.data(),
                          position_ids.data(),
                          request_batch,
                          sidecar_drafts.data());
            if (!chained_sidecar_ok)
            {
                return fail_after_checkpoint(
                    use_request_batch_device_draft_slots
                        ? "decodeStepBatch() request-batched chained MTP sidecar could not produce device draft slots"
                        : "decodeStepBatch() request-batched chained MTP sidecar failed");
            }
            if (!use_request_batch_device_draft_slots)
            {
                for (int request = 0; request < request_batch; ++request)
                {
                    const int32_t draft =
                        sidecar_drafts[static_cast<size_t>(request)];
                    if (draft < 0)
                    {
                        return fail_after_checkpoint(
                            "decodeStepBatch() request-batched chained MTP "
                            "sidecar produced an invalid draft token");
                    }
                    request_drafts[static_cast<size_t>(request)].push_back(draft);
                }
            }
        }

        if (!use_request_batch_device_draft_slots &&
            !runner_->flushPendingMTPWork())
        {
            return fail_after_checkpoint(
                "decodeStepBatch() request-batched MTP sidecar flush failed");
        }
        if (use_request_batch_device_draft_slots)
        {
            /*
             * Every proposal column records a readiness event. Chained sidecars
             * and verifier token materialization queue stream waits on those
             * events, so a CPU-visible stream flush would be both redundant and
             * a serialization bug in the fully device-resident lane.
             */
            PerfStatsCollector::addCounter(
                "mtp",
                "request_batch_sidecar_host_flushes_avoided",
                1.0,
                "decode",
                {},
                {{"request_count", std::to_string(request_batch)},
                 {"draft_depth", std::to_string(draft_depth)}});
        }

        MTPSpecRequestBatchOwner owner;
        const std::string compatibility_key =
            std::string("singledevice:") + runner_->architecture() +
            (stochastic_batch_verify ? ":stochastic:d" : ":greedy:d") +
            std::to_string(draft_depth);
        for (int request = 0; request < request_batch; ++request)
        {
            const BatchedDecodeRequestState &state =
                batched_request_states_[static_cast<size_t>(request)];
            if (state.is_complete)
                continue;
            if (!use_request_batch_device_draft_slots)
            {
                const int32_t draft =
                    sidecar_drafts[static_cast<size_t>(request)];
                if (draft < 0)
                {
                    return fail_after_checkpoint(
                        "decodeStepBatch() request-batched MTP sidecar produced "
                        "an invalid draft token");
                }
            }

            MTPSpecSchedulableRequest pending;
            pending.request_id = request;
            pending.ready = true;
            pending.mode = stochastic_batch_verify
                               ? MTPSpecRequestBatchMode::STOCHASTIC
                               : MTPSpecRequestBatchMode::GREEDY;
            pending.verifier_input =
                use_request_batch_device_draft_slots
                    ? MTPSpecVerifierInputPlacement::DEVICE_TOKEN_ROW
                    : MTPSpecVerifierInputPlacement::HOST_TOKENS;
            pending.compatibility_key = compatibility_key;
            pending.vocab_size = vocab;
            pending.base_cached_tokens =
                static_cast<int32_t>(
                    planning_sequence_lengths[static_cast<size_t>(request)]);
            pending.requires_shifted_kv_publication = true;
            pending.greedy_request.draft_tokens.clear();
            pending.greedy_request.draft_tokens.reserve(
                static_cast<size_t>(draft_depth) + 1u);
            pending.greedy_request.draft_tokens.push_back(
                use_request_batch_device_draft_slots
                    ? 0
                    : state.last_token);
            if (use_request_batch_device_draft_slots)
            {
                /*
                 * The scheduler needs only the verifier row shape. Proposal
                 * values stay exclusively in the request-major device matrix;
                 * zero is a valid non-semantic shadow that is never consumed by
                 * GPU embedding, comparison, or publication kernels.
                 */
                pending.greedy_request.draft_tokens.insert(
                    pending.greedy_request.draft_tokens.end(),
                    static_cast<size_t>(draft_depth),
                    0);
            }
            else
            {
                pending.greedy_request.draft_tokens.insert(
                    pending.greedy_request.draft_tokens.end(),
                    request_drafts[static_cast<size_t>(request)].begin(),
                    request_drafts[static_cast<size_t>(request)].end());
            }
            pending.greedy_request.stop_tokens = stop_tokens_;
            pending.greedy_request.base_sidecar_position =
                planning_sequence_lengths[static_cast<size_t>(request)];
            pending.greedy_request.verifier_path =
                "request_batched_grouped_decode_equivalent_publication";
            pending.greedy_request.implementation_name =
                stochastic_batch_verify
                    ? "request_batched_stochastic_d" + std::to_string(draft_depth)
                    : "request_batched_greedy_d" + std::to_string(draft_depth);

            std::string enqueue_error;
            if (!owner.enqueueRequest(std::move(pending), &enqueue_error))
            {
                return fail_after_checkpoint(
                    std::string("decodeStepBatch() could not enqueue MTP request: ") +
                    enqueue_error);
            }
        }

        MTPSpecRequestBatchScheduler scheduler(
            MTPSpecRequestBatchSchedulerConfig{
                .max_request_batch = request_batch,
                .max_draft_tokens = draft_depth + 1,
                .mode = stochastic_batch_verify
                            ? MTPSpecRequestBatchMode::STOCHASTIC
                            : MTPSpecRequestBatchMode::GREEDY});

        const bool use_device_resident_request_batch_publication =
            gpu_request_batch;
        if (use_device_resident_request_batch_publication &&
            !runner_->supportsDeviceResidentMTPSpecStatePublication())
        {
            return fail_after_checkpoint(
                "decodeStepBatch() GPU request batching requires "
                "device-resident MTP state publication");
        }
        if (use_device_resident_request_batch_publication &&
            !runner_->supportsMTPDeviceDraftTokenInput())
        {
            return fail_after_checkpoint(
                "decodeStepBatch() GPU request batching requires "
                "device draft-token verifier input");
        }

        DeviceSpeculativeOutcomeHandle resident_request_batch_outcome;
        bool resident_request_batch_outcome_ready = false;
        int resident_request_batch_verifier_rows = 0;
        std::vector<std::optional<Sampler>> sampled_terminal_samplers(
            static_cast<size_t>(request_batch));

        auto publish = [&](const MTPSpecTransactionBatchPlan &plan,
                           std::string *error) -> bool
        {
            return runner_->publishAcceptedMTPSpecStateBatch(
                plan.step_plans,
                error);
        };

        auto process_stochastic_host_outcomes =
            [&](const std::vector<DeviceStochasticBatchOutcomeRequest> &outcome_requests,
                const std::vector<Sampler> &bonus_samplers,
                const std::vector<MTPDeviceRejectionBatchOutcome> &outcomes,
                std::string *error) -> bool
        {
            auto set_error = [&](std::string message) -> bool
            {
                if (error)
                    *error = std::move(message);
                return false;
            };

            if (outcome_requests.size() != outcomes.size())
            {
                return set_error(
                    "stochastic request-batch outcome vector size mismatch");
            }

            for (size_t i = 0; i < outcome_requests.size(); ++i)
            {
                const int request_id = outcome_requests[i].request_id;
                if (request_id < 0 || request_id >= request_batch)
                {
                    return set_error(
                        "stochastic request-batch outcome descriptor has an out-of-range request id");
                }
                if (static_cast<size_t>(request_id) >= bonus_samplers.size())
                {
                    return set_error(
                        "stochastic request-batch bonus sampler vector is undersized");
                }
                if (outcomes[i].sampled_terminal)
                {
                    sampled_terminal_samplers[static_cast<size_t>(request_id)] =
                        bonus_samplers[static_cast<size_t>(request_id)];
                }

                const int physical_rows =
                    std::max(0, outcome_requests[i].row_count);
                const int semantic_rows =
                    std::min(
                        physical_rows,
                        std::max(0, outcomes[i].consumed_verifier_rows));
                const int post_reject_rows =
                    std::max(0, physical_rows - semantic_rows);

                /*
                 * The compact outcome is already materialized for the serving
                 * response at this point, so this opt-in trace adds no device
                 * transfer or synchronization. Keeping every publication
                 * decision on one line makes batch-invariance failures
                 * directly comparable with a scalar request: the trace records
                 * the RNG policy, emitted token bytes, accepted/committed rows,
                 * correction token, and terminal ownership together.
                 */
                if (debugEnv().runtime_debug.mtp_publication_diagnostics)
                {
                    std::ostringstream output_tokens;
                    const int output_count = std::clamp(
                        outcomes[i].output_token_count,
                        0,
                        static_cast<int>(outcomes[i].output_tokens.size()));
                    output_tokens << '[';
                    for (int token_index = 0;
                         token_index < output_count;
                         ++token_index)
                    {
                        if (token_index > 0)
                            output_tokens << ',';
                        output_tokens << outcomes[i].output_tokens[
                            static_cast<size_t>(token_index)];
                    }
                    output_tokens << ']';

                    LOG_INFO(
                        "[MTPPublicationDiagnostics] phase=request_batch_stochastic_compact_outcome"
                        << " request=" << request_id
                        << " rows=" << physical_rows
                        << " serial_sample_equivalent="
                        << (outcome_requests[i].serial_sample_equivalent
                                ? "true"
                                : "false")
                        << " vllm_probability_rejection="
                        << (outcome_requests[i].use_vllm_probability_rejection
                                ? "true"
                                : "false")
                        << " output_count=" << output_count
                        << " output_tokens=" << output_tokens.str()
                        << " accepted_prefix="
                        << outcomes[i].accepted_speculative_prefix
                        << " committed_states="
                        << outcomes[i].target_verifier_state_commit_count
                        << " consumed_rows="
                        << outcomes[i].consumed_verifier_rows
                        << " ready_token=" << outcomes[i].ready_token
                        << " rejected_token="
                        << outcomes[i].rejected_verified_token
                        << " all_accepted="
                        << (outcomes[i].all_speculative_accepted
                                ? "true"
                                : "false")
                        << " sampled_terminal="
                        << (outcomes[i].sampled_terminal ? "true" : "false")
                        << " stopped="
                        << (outcomes[i].stopped_on_output ? "true" : "false"));
                }

                PerfStatsCollector::addCounter(
                    "mtp",
                    "stochastic_device_physical_verify_rows",
                    static_cast<double>(physical_rows),
                    "decode",
                    {},
                    {{"implementation", "request_batch_device_outcome"},
                     {"request_batch", "true"}});
                PerfStatsCollector::addCounter(
                    "mtp",
                    "stochastic_device_semantic_verify_rows",
                    static_cast<double>(semantic_rows),
                    "decode",
                    {},
                    {{"implementation", "request_batch_device_outcome"},
                     {"request_batch", "true"}});
                PerfStatsCollector::addCounter(
                    "mtp",
                    "stochastic_device_post_reject_rows",
                    static_cast<double>(post_reject_rows),
                    "decode",
                    {},
                    {{"implementation", "request_batch_device_outcome"},
                     {"request_batch", "true"}});
            }
            return true;
        };

        std::vector<int> scheduled_request_ids;
        std::vector<int32_t> scheduled_base_cached_tokens;
        std::vector<MTPDecodeCatchupGreedyResult> catchup_results;

        if (!stochastic_batch_verify)
        {
            if (use_device_resident_request_batch_publication)
            {
                auto release_and_fail =
                    [&](MTPOwnedDeviceOutcomeBatchTransactionResult &tx,
                        std::string message) -> GenerationBatchResult
                {
                    if (owner.hasInFlightBatch())
                    {
                        std::string release_error;
                        tx.released =
                            owner.releaseInFlightBatch(&release_error);
                        if (!tx.released)
                        {
                            message += "; release failed: ";
                            message += release_error;
                        }
                    }
                    return fail_after_checkpoint(message);
                };

                auto produce_greedy_resident_outcomes =
                    [&](const MTPSpecRequestBatch &scheduled_batch,
                        DeviceSpeculativeOutcomeHandle *resident_handle,
                        std::string *error) -> bool
                {
                    auto set_producer_error = [&](std::string message) -> bool
                    {
                        if (error)
                            *error = std::move(message);
                        return false;
                    };

                    if (!resident_handle)
                        return set_producer_error("greedy resident outcome handle is null");
                    *resident_handle = {};
                    if (!scheduled_batch.ok)
                        return set_producer_error("scheduled greedy batch is invalid");
                    if (scheduled_batch.request_count <= 0)
                        return set_producer_error("scheduled greedy batch is empty");

                    std::vector<MTPSpecDecodeVerifierDraftRequest>
                        verifier_requests;
                    verifier_requests.reserve(
                        scheduled_batch.greedy_requests.size());
                    for (size_t i = 0;
                         i < scheduled_batch.greedy_requests.size();
                         ++i)
                    {
                        MTPSpecDecodeVerifierDraftRequest request;
                        request.request_id = scheduled_batch.request_ids[i];
                        request.draft_tokens =
                            scheduled_batch.greedy_requests[i].draft_tokens;
                        verifier_requests.push_back(std::move(request));
                    }

                    MTPSpecDecodeVerifierInputPlan verifier_input_plan =
                        buildMTPSpecDecodeVerifierInputPlan(
                            scheduled_batch.shape,
                            verifier_requests);
                    if (!verifier_input_plan.ok)
                    {
                        return set_producer_error(
                            std::string("greedy request-batch verifier input plan failed: ") +
                            verifier_input_plan.error);
                    }
                    if (!verifierInputPlanHasCompactRows(verifier_input_plan))
                    {
                        return set_producer_error(
                            "greedy request-batch verifier row metadata is malformed");
                    }

                    const int padded_seq_len =
                        scheduled_batch.shape.max_draft_tokens;
                    std::vector<DeviceMTPVerifierInputBatchRequest>
                        token_batch_requests;
                    token_batch_requests.reserve(
                        scheduled_batch.greedy_requests.size());
                    for (size_t i = 0;
                         i < scheduled_batch.greedy_requests.size();
                         ++i)
                    {
                        const int request_id = scheduled_batch.request_ids[i];
                        if (request_id < 0 || request_id >= request_batch)
                        {
                            return set_producer_error(
                                "greedy request-batch verifier returned an out-of-range request id");
                        }
                        const MTPDecodeCatchupGreedyRequest &request =
                            scheduled_batch.greedy_requests[i];
                        const int verifier_token_count =
                            static_cast<int>(request.draft_tokens.size());
                        if (verifier_token_count <= 1 ||
                            verifier_token_count > padded_seq_len)
                        {
                            return set_producer_error(
                                "greedy request-batch verifier request shape is invalid");
                        }

                        DeviceMTPVerifierInputBatchRequest descriptor;
                        descriptor.request_id = request_id;
                        descriptor.first_token = request.draft_tokens.front();
                        if (use_request_batch_resident_condition_tokens)
                        {
                            descriptor.first_token_from_device = true;
                            descriptor.first_target_sample_slot = request_id;
                            if (!resident_batch_state.coversRequest(request_id))
                            {
                                return set_producer_error(
                                    "greedy request-batch verifier has no resident condition-token row for request");
                            }
                        }
                        else
                        {
                            descriptor.first_token_from_device = false;
                            descriptor.first_target_sample_slot = -1;
                        }
                        descriptor.first_draft_slot =
                            request_id * draft_depth;
                        descriptor.draft_token_count =
                            verifier_token_count - 1;
                        descriptor.total_verifier_input_tokens =
                            verifier_token_count;
                        token_batch_requests.push_back(descriptor);
                    }

                    bool row_indexed_enabled = false;
                    bool all_position_enabled = false;
                    auto cleanup_row_modes = [&]() -> bool
                    {
                        bool ok = true;
                        runner_->clearMTPSpecVerifierInputPlan();
                        if (all_position_enabled)
                        {
                            all_position_enabled = false;
                            ok = runner_->setComputeAllPositionLogits(false) && ok;
                        }
                        if (row_indexed_enabled)
                        {
                            row_indexed_enabled = false;
                            ok = runner_->setComputeRowIndexedAllPositionLogits(false, 0) && ok;
                        }
                        return ok;
                    };

                    const int compact_row_count =
                        verifier_input_plan.compact_logit_row_count;
                    if (!runner_->setComputeRowIndexedAllPositionLogits(
                            true,
                            compact_row_count))
                    {
                        return set_producer_error(
                            "greedy request-batch verifier could not enable row-indexed logits");
                    }
                    row_indexed_enabled = true;
                    if (!runner_->setMTPSpecVerifierInputPlan(
                            verifier_input_plan))
                    {
                        const bool cleanup_ok = cleanup_row_modes();
                        return set_producer_error(
                            cleanup_ok
                                ? "greedy request-batch verifier could not install row plan"
                                : "greedy request-batch verifier could not install row plan and cleanup failed");
                    }
                    /*
                     * Installing a verifier plan defines a new graph transaction
                     * and deliberately invalidates any pending device-token
                     * composition. Stage the request-major token matrix only
                     * after that transaction exists, so the verifier forward can
                     * materialize the same rows later consumed by the compact
                     * outcome reducer.
                     */
                    const void *device_token_batch =
                        runner_->prepareMTPVerifierInputTokenBatchOnDevice(
                            token_batch_requests.data(),
                            static_cast<int>(token_batch_requests.size()),
                            padded_seq_len);
                    if (!device_token_batch)
                    {
                        const bool cleanup_ok = cleanup_row_modes();
                        return set_producer_error(
                            cleanup_ok
                                ? "greedy request-batch verifier could not prepare device token matrix"
                                : "greedy request-batch verifier could not prepare device token matrix and cleanup failed");
                    }
                    if (!runner_->setComputeAllPositionLogits(true))
                    {
                        const bool cleanup_ok = cleanup_row_modes();
                        return set_producer_error(
                            cleanup_ok
                                ? "greedy request-batch verifier could not enable all-position logits"
                                : "greedy request-batch verifier could not enable all-position logits and cleanup failed");
                    }
                    all_position_enabled = true;

                    MTPVerifierForwardExecutionOptions forward_options;
                    forward_options.device_token_ids = device_token_batch;
                    forward_options.allow_batched_host_forward = false;
                    {
                        ScopedMTPAllPositionVerifierSyncDeferral
                            verifier_sync_deferral(
                                runner_.get(),
                                runner_->primaryDeviceId().is_gpu());
                        PerfStatsCollector::ScopedTimer verifier_timer(
                            "mtp",
                            "request_batch_greedy_verifier_forward_device_tokens",
                            "decode",
                            {},
                            {{"requests",
                              std::to_string(
                                  scheduled_batch.request_count)},
                             {"draft_depth", std::to_string(draft_depth)}});
                        const MTPVerifierForwardExecutionResult forward =
                            executeMTPSpecVerifierForward(
                                *runner_,
                                verifier_input_plan,
                                forward_options);
                        if (!forward.ok)
                        {
                            const bool cleanup_ok = cleanup_row_modes();
                            return set_producer_error(
                                cleanup_ok
                                    ? std::string("greedy request-batch verifier forward failed: ") +
                                          forward.error
                                    : std::string("greedy request-batch verifier forward failed and cleanup failed: ") +
                                          forward.error);
                        }
                    }

                    if (!cleanup_row_modes())
                    {
                        return set_producer_error(
                            "greedy request-batch verifier could not disable row-indexed logits");
                    }

                    std::vector<DeviceGreedyBatchOutcomeRequest>
                        outcome_requests;
                    outcome_requests.reserve(
                        scheduled_batch.greedy_requests.size());
                    for (size_t i = 0;
                         i < scheduled_batch.greedy_requests.size();
                         ++i)
                    {
                        const MTPDecodeCatchupGreedyRequest &request =
                            scheduled_batch.greedy_requests[i];
                        const int verifier_token_count =
                            static_cast<int>(request.draft_tokens.size());
                        if (static_cast<int>(
                                verifier_input_plan.query_start_locs.size()) <=
                            static_cast<int>(i))
                        {
                            return set_producer_error(
                                "greedy request-batch verifier query row metadata is undersized");
                        }

                        DeviceGreedyBatchOutcomeRequest descriptor;
                        descriptor.request_id =
                            scheduled_batch.request_ids[i];
                        descriptor.first_target_row =
                            verifier_input_plan.query_start_locs[i];
                        descriptor.verifier_token_count =
                            verifier_token_count;
                        descriptor.token_row_stride = padded_seq_len;
                        descriptor.token_row_offset =
                            static_cast<int>(i) * padded_seq_len;
                        descriptor.first_token =
                            request.draft_tokens.front();
                        descriptor.stop_token_count =
                            static_cast<int>(stop_tokens_.size());
                        if (descriptor.stop_token_count >
                            static_cast<int>(
                                sampling_math::kSpeculativeBatchMaxStopTokens))
                        {
                            return set_producer_error(
                                "greedy request-batch stop-token count exceeds device summary capacity");
                        }
                        for (int stop_index = 0;
                             stop_index < descriptor.stop_token_count;
                             ++stop_index)
                        {
                            descriptor.stop_tokens[
                                static_cast<size_t>(stop_index)] =
                                stop_tokens_[static_cast<size_t>(stop_index)];
                        }
                        outcome_requests.push_back(descriptor);
                    }

                    if (!runner_->verifyGreedyAllPositionRequestBatchOutcomesOnDeviceResident(
                            outcome_requests.data(),
                            static_cast<int>(outcome_requests.size()),
                            resident_handle))
                    {
                        return set_producer_error(
                            "greedy request-batch resident device outcome verifier failed");
                    }
                    return resident_handle->valid();
                };

                MTPOwnedDeviceOutcomeBatchTransactionResult tx;
                tx.scheduled_batch = owner.scheduleNextBatch(scheduler);
                if (!tx.scheduled_batch.ok)
                {
                    return fail_after_checkpoint(
                        std::string("decodeStepBatch() request-batched greedy "
                                    "verifier scheduling failed: ") +
                        tx.scheduled_batch.error);
                }

                std::string produce_error;
                tx.produced = produce_greedy_resident_outcomes(
                    tx.scheduled_batch,
                    &resident_request_batch_outcome,
                    &produce_error);
                resident_request_batch_outcome_ready =
                    tx.produced && resident_request_batch_outcome.valid();
                resident_request_batch_verifier_rows =
                    tx.scheduled_batch.shape.max_draft_tokens;
                if (!tx.produced)
                {
                    std::string message =
                        "decodeStepBatch() request-batched greedy resident "
                        "verifier production failed";
                    if (!produce_error.empty())
                    {
                        message += ": ";
                        message += produce_error;
                    }
                    return release_and_fail(tx, std::move(message));
                }
                if (!resident_request_batch_outcome_ready ||
                    resident_request_batch_verifier_rows <= 0)
                {
                    return release_and_fail(
                        tx,
                        "decodeStepBatch() request-batched greedy resident "
                        "verifier produced no valid device outcome");
                }

                DeviceSpeculativePublicationRequest publication_request;
                publication_request.outcome = resident_request_batch_outcome;
                publication_request.max_state_commit_rows =
                    resident_request_batch_verifier_rows;
                publication_request.publish_mtp_shifted_kv =
                    tx.scheduled_batch.requires_shifted_kv_publication;

                std::string publication_error;
                {
                    PerfStatsCollector::ScopedTimer publication_timer(
                        "mtp",
                        "request_batch_publish_accepted_state_device_resident",
                        "decode",
                        {},
                        {{"sampling", "greedy"},
                         {"request_count",
                          std::to_string(publication_request.requestCount())},
                         {"logical_verifier_rows",
                          std::to_string(
                              publication_request
                                  .logicalVerifierRowsPerRequest())},
                         {"physical_verifier_rows",
                          std::to_string(
                              publication_request
                                  .physicalVerifierRowsPerRequest())}});
                    tx.published =
                        runner_->publishAcceptedMTPSpecStateBatchFromDeviceOutcome(
                            publication_request,
                            &publication_error);
                }
                if (!tx.published)
                {
                    std::string message =
                        "decodeStepBatch() request-batched greedy resident "
                        "state publication failed";
                    if (!publication_error.empty())
                    {
                        message += ": ";
                        message += publication_error;
                    }
                    return release_and_fail(tx, std::move(message));
                }

                const DeviceResidentLogicalSequenceStateHandle logical_state =
                    runner_->deviceResidentLogicalSequenceState();
                if (!logical_state.valid() ||
                    logical_state.request_count <
                        tx.scheduled_batch.request_count)
                {
                    return release_and_fail(
                        tx,
                        "decodeStepBatch() request-batched greedy resident "
                        "publication produced no valid logical-state mailbox");
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "request_batch_resident_plan_checks",
                    1.0,
                    "decode",
                    {},
                    {{"request_count",
                      std::to_string(tx.scheduled_batch.request_count)},
                     {"sampling", "greedy"}});

                std::string commit_error;
                tx.committed = owner.commitInFlightBatch(&commit_error);
                if (!tx.committed)
                {
                    return fail_after_checkpoint(
                        std::string("decodeStepBatch() request-batched greedy "
                                    "resident publication succeeded but owner "
                                    "commit failed: ") +
                        commit_error);
                }

                PerfStatsCollector::addCounter(
                    "mtp",
                    "request_batch_device_resident_state_publications",
                    1.0,
                    "decode",
                    {},
                    {{"request_count",
                      std::to_string(publication_request.requestCount())},
                     {"logical_verifier_rows",
                      std::to_string(
                          publication_request.logicalVerifierRowsPerRequest())},
                     {"sampling", "greedy"}});
                tx.ok = true;
                return completeDeviceResidentBatchGeneration(
                    publication_request,
                    DeviceGenerationSamplingMode::Greedy,
                    draft_depth,
                    draft_depth,
                    std::move(batch_result));
            }
            else
            {
                MTPOwnedGreedyVerifierBatchTransactionResult tx =
                    executeOwnedMTPGreedyVerifierScheduledBatchTransactionAndPublish(
                        *runner_,
                        owner,
                        scheduler,
                        publish);
                if (!tx.ok)
                {
                    return fail_after_checkpoint(
                        std::string("decodeStepBatch() request-batched verifier "
                                    "transaction failed: ") +
                        tx.error);
                }

                scheduled_request_ids = tx.scheduled_batch.request_ids;
                scheduled_base_cached_tokens =
                    tx.scheduled_batch.base_cached_tokens;
                catchup_results = tx.transaction.catchup.results;
            }
        }
        else
        {
            auto produce_stochastic_outcomes =
                [&](const MTPSpecRequestBatch &scheduled_batch,
                    std::vector<MTPDeviceRejectionBatchOutcome> *outcomes,
                    std::string *error) -> bool
            {
                auto set_producer_error = [&](std::string message) -> bool
                {
                    if (error)
                        *error = std::move(message);
                    return false;
                };

                if (!outcomes)
                    return set_producer_error("stochastic outcome output vector is null");
                if (!scheduled_batch.ok)
                    return set_producer_error("scheduled stochastic batch is invalid");
                if (scheduled_batch.request_count <= 0)
                    return set_producer_error("scheduled stochastic batch is empty");

                resident_request_batch_outcome = {};
                resident_request_batch_outcome_ready = false;
                resident_request_batch_verifier_rows = 0;

                std::vector<MTPSpecDecodeVerifierDraftRequest> verifier_requests;
                verifier_requests.reserve(scheduled_batch.greedy_requests.size());
                for (size_t i = 0; i < scheduled_batch.greedy_requests.size(); ++i)
                {
                    MTPSpecDecodeVerifierDraftRequest request;
                    request.request_id = scheduled_batch.request_ids[i];
                    request.draft_tokens =
                        scheduled_batch.greedy_requests[i].draft_tokens;
                    verifier_requests.push_back(std::move(request));
                }

                MTPSpecDecodeVerifierInputPlan verifier_input_plan =
                    buildMTPSpecDecodeVerifierInputPlan(
                        scheduled_batch.shape,
                        verifier_requests);
                if (!verifier_input_plan.ok)
                {
                    return set_producer_error(
                        std::string("stochastic request-batch verifier input plan failed: ") +
                        verifier_input_plan.error);
                }
                if (!verifierInputPlanHasCompactRows(verifier_input_plan))
                {
                    return set_producer_error(
                        "stochastic request-batch verifier row metadata is malformed");
                }

                const int padded_seq_len =
                    scheduled_batch.shape.max_draft_tokens;
                std::vector<DeviceMTPVerifierInputBatchRequest>
                    token_batch_requests;
                const void *device_token_batch = nullptr;
                if (use_device_resident_request_batch_publication)
                {
                    token_batch_requests.reserve(
                        scheduled_batch.greedy_requests.size());
                    for (size_t i = 0;
                         i < scheduled_batch.greedy_requests.size();
                         ++i)
                    {
                        const int request_id = scheduled_batch.request_ids[i];
                        if (request_id < 0 || request_id >= request_batch)
                        {
                            return set_producer_error(
                                "stochastic request-batch verifier returned an out-of-range request id");
                        }

                        const MTPDecodeCatchupGreedyRequest &request =
                            scheduled_batch.greedy_requests[i];
                        const int verifier_token_count =
                            static_cast<int>(request.draft_tokens.size());
                        if (verifier_token_count <= 1 ||
                            verifier_token_count > padded_seq_len)
                        {
                            return set_producer_error(
                                "stochastic request-batch verifier request shape is invalid");
                        }

                        DeviceMTPVerifierInputBatchRequest descriptor;
                        descriptor.request_id = request_id;
                        descriptor.first_token = request.draft_tokens.front();
                        if (use_request_batch_resident_condition_tokens)
                        {
                            descriptor.first_token_from_device = true;
                            descriptor.first_target_sample_slot = request_id;
                            if (!resident_batch_state.coversRequest(request_id))
                            {
                                return set_producer_error(
                                    "stochastic request-batch verifier has no resident condition-token row for request");
                            }
                        }
                        else
                        {
                            descriptor.first_token_from_device = false;
                            descriptor.first_target_sample_slot = -1;
                        }
                        descriptor.first_draft_slot =
                            request_id * draft_depth;
                        descriptor.draft_token_count =
                            verifier_token_count - 1;
                        descriptor.total_verifier_input_tokens =
                            verifier_token_count;
                        token_batch_requests.push_back(descriptor);
                    }
                }

                bool row_indexed_enabled = false;
                bool all_position_enabled = false;
                auto cleanup_row_modes = [&]() -> bool
                {
                    bool ok = true;
                    runner_->clearMTPSpecVerifierInputPlan();
                    if (all_position_enabled)
                    {
                        all_position_enabled = false;
                        ok = runner_->setComputeAllPositionLogits(false) && ok;
                    }
                    if (row_indexed_enabled)
                    {
                        row_indexed_enabled = false;
                        ok = runner_->setComputeRowIndexedAllPositionLogits(false, 0) && ok;
                    }
                    return ok;
                };

                const int compact_row_count =
                    verifier_input_plan.compact_logit_row_count;
                if (!runner_->setComputeRowIndexedAllPositionLogits(
                        true,
                        compact_row_count))
                {
                    return set_producer_error(
                        "stochastic request-batch verifier could not enable row-indexed logits");
                }
                row_indexed_enabled = true;
                if (!runner_->setMTPSpecVerifierInputPlan(verifier_input_plan))
                {
                    const bool cleanup_ok = cleanup_row_modes();
                    return set_producer_error(
                        cleanup_ok
                            ? "stochastic request-batch verifier could not install row plan"
                            : "stochastic request-batch verifier could not install row plan and cleanup failed");
                }
                if (use_device_resident_request_batch_publication)
                {
                    /*
                     * setMTPSpecVerifierInputPlan() starts the authoritative
                     * verifier transaction and clears stale token-composition
                     * work. Publish this transaction first, then stage the
                     * device-owned request rows that its forward will consume.
                     */
                    device_token_batch =
                        runner_->prepareMTPVerifierInputTokenBatchOnDevice(
                            token_batch_requests.data(),
                            static_cast<int>(token_batch_requests.size()),
                            padded_seq_len);
                    if (!device_token_batch)
                    {
                        const bool cleanup_ok = cleanup_row_modes();
                        return set_producer_error(
                            cleanup_ok
                                ? "stochastic request-batch verifier could not prepare device token matrix"
                                : "stochastic request-batch verifier could not prepare device token matrix and cleanup failed");
                    }
                }
                if (!runner_->setComputeAllPositionLogits(true))
                {
                    const bool cleanup_ok = cleanup_row_modes();
                    return set_producer_error(
                        cleanup_ok
                            ? "stochastic request-batch verifier could not enable all-position logits"
                            : "stochastic request-batch verifier could not enable all-position logits and cleanup failed");
                }
                all_position_enabled = true;

                {
                    ScopedMTPAllPositionVerifierSyncDeferral verifier_sync_deferral(
                        runner_.get(),
                        runner_->primaryDeviceId().is_gpu());
                    PerfStatsCollector::ScopedTimer verifier_timer(
                        "mtp",
                        use_device_resident_request_batch_publication
                            ? "request_batch_stochastic_verifier_forward_device_tokens"
                            : "request_batch_stochastic_verifier_forward",
                        "decode",
                        {},
                        {{"requests", std::to_string(scheduled_batch.request_count)},
                         {"draft_depth", std::to_string(draft_depth)}});
                    MTPVerifierForwardExecutionOptions forward_options;
                    forward_options.device_token_ids = device_token_batch;
                    forward_options.allow_batched_host_forward =
                        !use_device_resident_request_batch_publication;
                    const MTPVerifierForwardExecutionResult forward =
                        executeMTPSpecVerifierForward(
                            *runner_,
                            verifier_input_plan,
                            forward_options);
                    if (!forward.ok)
                    {
                        const bool cleanup_ok = cleanup_row_modes();
                        return set_producer_error(
                            cleanup_ok
                                ? std::string("stochastic request-batch verifier forward failed: ") +
                                      forward.error
                                : std::string("stochastic request-batch verifier forward failed and cleanup failed: ") +
                                      forward.error);
                    }
                }

                if (!cleanup_row_modes())
                {
                    return set_producer_error(
                        "stochastic request-batch verifier could not disable row-indexed logits");
                }

                std::vector<DeviceStochasticBatchOutcomeRequest>
                    outcome_requests;
                outcome_requests.reserve(
                    scheduled_batch.greedy_requests.size());
                std::vector<Sampler> bonus_samplers(
                    static_cast<size_t>(request_batch));
                int next_draft_slot = 0;
                for (size_t i = 0; i < scheduled_batch.greedy_requests.size(); ++i)
                {
                    const int request_id = scheduled_batch.request_ids[i];
                    if (request_id < 0 || request_id >= request_batch)
                    {
                        return set_producer_error(
                            "stochastic request-batch verifier returned an out-of-range request id");
                    }

                    const MTPDecodeCatchupGreedyRequest &request =
                        scheduled_batch.greedy_requests[i];
                    const int verifier_token_count =
                        static_cast<int>(request.draft_tokens.size());
                    const int compare_rows = verifier_token_count - 1;
                    if (compare_rows <= 0 ||
                        compare_rows > draft_depth ||
                        static_cast<int>(verifier_input_plan.query_start_locs.size()) <=
                            static_cast<int>(i))
                    {
                        return set_producer_error(
                            "stochastic request-batch verifier request shape is invalid");
                    }
                    if (stop_tokens_.size() >
                        static_cast<size_t>(
                            sampling_math::kSpeculativeBatchMaxStopTokens))
                    {
                        return set_producer_error(
                            "stochastic request-batch stop-token count exceeds device summary capacity");
                    }

                    const int first_compact_row =
                        verifier_input_plan.query_start_locs[i];
                    const int first_draft_slot =
                        use_request_batch_device_draft_slots
                            ? request_id * draft_depth
                            : next_draft_slot;
                    next_draft_slot += compare_rows;
                    const int bonus_row = compare_rows;
                    if (!runner_->buildStochasticDistributionsOnDevice(
                            DeviceLogitsSource::AllPosition,
                            first_compact_row,
                            DeviceDistributionBuffer::Target,
                            first_compact_row,
                            compare_rows + 1,
                            active_sampling_params_,
                            vocab))
                    {
                        return set_producer_error(
                            "stochastic request-batch compact target-row build failed");
                    }

                    if (!use_request_batch_device_draft_slots &&
                        !runner_->stageStochasticDraftTokensForDeviceVerification(
                            request.draft_tokens.data() + 1,
                            compare_rows,
                            first_draft_slot))
                    {
                        return set_producer_error(
                            "stochastic request-batch draft-token staging failed");
                    }

                    Sampler &request_sampler =
                        batched_request_states_[static_cast<size_t>(request_id)]
                            .sampler;
                    const uint64_t request_position_seed =
                        batched_request_states_[static_cast<size_t>(request_id)]
                            .stochastic_position_seed;
                    DeviceStochasticBatchOutcomeRequest descriptor;
                    if (!descriptor.ensureHostRowCapacity(compare_rows))
                    {
                        return set_producer_error(
                            "stochastic request-batch verifier could not size its host diagnostic descriptor");
                    }
                    descriptor.request_id = request_id;
                    descriptor.first_target_slot = first_compact_row;
                    descriptor.first_draft_slot = first_draft_slot;
                    descriptor.row_count = compare_rows;
                    descriptor.first_token = request.draft_tokens.front();
                    if (use_device_resident_request_batch_publication)
                    {
                        descriptor.first_token_from_device = true;
                        descriptor.first_target_sample_slot = -1;
                        descriptor.token_row_offset =
                            static_cast<int>(i) * padded_seq_len;
                        descriptor.token_row_stride = padded_seq_len;
                    }
                    else
                    {
                        descriptor.first_token_from_device = false;
                        descriptor.first_target_sample_slot = -1;
                        descriptor.token_row_offset = -1;
                        descriptor.token_row_stride = 0;
                    }
                    descriptor.bonus_target_slot = first_compact_row + bonus_row;
                    descriptor.use_device_draft_tokens = true;
                    /*
                     * A fixed seed is a pathwise reproducibility contract, not
                     * merely a distributional one. The grouped GPU verifier
                     * therefore samples every target row in parallel with the
                     * same position-keyed draw as serial decode, then compares
                     * those sampled tokens against the resident draft matrix.
                     * vLLM probability rejection remains appropriate for an
                     * unseeded request, where no cross-run token identity is
                     * promised, but it cannot make a seeded RB=N transaction
                     * byte-equivalent to RB=1.
                     */
                    const bool serial_sample_equivalent =
                        active_sampling_params_.seed != 0;
                    descriptor.serial_sample_equivalent =
                        serial_sample_equivalent;
                    descriptor.use_vllm_probability_rejection =
                        !serial_sample_equivalent;
                    const int base_cached_tokens =
                        scheduled_batch.base_cached_tokens[i];
                    if (use_device_resident_request_batch_publication)
                    {
                        /*
                         * GPU stochastic verification owns both the random draw
                         * and its mutable logical-position source. The seed is
                         * immutable request configuration; the position is read
                         * from the publication mailbox by the verifier kernel.
                         * Host arrays remain constructor sentinels and must never
                         * be adopted as launch arguments in this mode.
                         */
                        descriptor.inverse_sample_seed = request_position_seed;
                        descriptor.inverse_sample_first_logical_position = -1;
                        descriptor.derive_thresholds_from_seed = true;
                        descriptor.draw_position_source =
                            DeviceStochasticDrawPositionSource::ResidentLogicalState;
                        descriptor.bonus_threshold = 0.0f;
                    }
                    else
                    {
                        for (int row = 0; row < compare_rows; ++row)
                        {
                            const int logical_position =
                                base_cached_tokens + 1 + row;
                            descriptor.accept_thresholds[static_cast<size_t>(row)] =
                                mtpSpecStochasticThresholdForPosition(
                                    active_sampling_params_,
                                    request_sampler,
                                    logical_position,
                                    MTPSpecStochasticDrawPurpose::Accept);
                            descriptor.residual_thresholds[static_cast<size_t>(row)] =
                                mtpSpecStochasticThresholdForPosition(
                                    active_sampling_params_,
                                    request_sampler,
                                    logical_position,
                                    MTPSpecStochasticDrawPurpose::Residual);
                        }
                        Sampler bonus_sampler = request_sampler;
                        descriptor.bonus_threshold =
                            mtpSpecStochasticThresholdForPosition(
                                active_sampling_params_,
                                bonus_sampler,
                                base_cached_tokens + verifier_token_count,
                                MTPSpecStochasticDrawPurpose::Sample);
                        bonus_samplers[static_cast<size_t>(request_id)] =
                            std::move(bonus_sampler);
                        descriptor.inverse_sample_seed =
                            mtpSpecInverseSampleSeedForThresholds(
                                active_sampling_params_,
                                descriptor.residual_thresholds.data(),
                                static_cast<size_t>(compare_rows));
                        descriptor.inverse_sample_first_logical_position =
                            base_cached_tokens + 1;
                    }
                    for (int row = 0; row < compare_rows; ++row)
                    {
                        descriptor.draft_tokens[static_cast<size_t>(row)] =
                            request.draft_tokens[static_cast<size_t>(row + 1)];
                    }

                    if (use_device_resident_request_batch_publication)
                    {
                        bonus_samplers[static_cast<size_t>(request_id)] =
                            request_sampler;
                    }
                    descriptor.stop_token_count =
                        static_cast<int>(stop_tokens_.size());
                    for (int stop_index = 0;
                         stop_index < descriptor.stop_token_count;
                         ++stop_index)
                    {
                        descriptor.stop_tokens[static_cast<size_t>(stop_index)] =
                            stop_tokens_[static_cast<size_t>(stop_index)];
                    }

                    outcome_requests.push_back(std::move(descriptor));
                }

                outcomes->assign(
                    outcome_requests.size(),
                    MTPDeviceRejectionBatchOutcome{});
                if (use_device_resident_request_batch_publication)
                {
                    DeviceSpeculativeOutcomeHandle resident_handle;
                    if (!runner_->verifyStochasticDistributionsRequestBatchOutcomesOnDeviceResident(
                            outcome_requests.data(),
                            static_cast<int>(outcome_requests.size()),
                            &resident_handle))
                    {
                        return set_producer_error(
                            "stochastic request-batch resident device outcome verifier failed");
                    }

                    resident_request_batch_outcome = resident_handle;
                    resident_request_batch_outcome_ready = true;
                    resident_request_batch_verifier_rows =
                        scheduled_batch.shape.max_draft_tokens;
                    outcomes->clear();
                    return true;
                }
                else if (!runner_->verifyStochasticDistributionsRequestBatchOutcomesOnDevice(
                             outcome_requests.data(),
                             static_cast<int>(outcome_requests.size()),
                             outcomes->data()))
                {
                    return set_producer_error(
                        "stochastic request-batch device outcome verifier failed");
                }

                if (!process_stochastic_host_outcomes(
                        outcome_requests,
                        bonus_samplers,
                        *outcomes,
                        error))
                {
                    return false;
                }
                return true;
            };

            MTPOwnedDeviceOutcomeBatchTransactionResult tx;
            if (use_device_resident_request_batch_publication)
            {
                auto release_and_fail =
                    [&](std::string message) -> GenerationBatchResult
                {
                    if (owner.hasInFlightBatch())
                    {
                        std::string release_error;
                        tx.released =
                            owner.releaseInFlightBatch(&release_error);
                        if (!tx.released)
                        {
                            message += "; release failed: ";
                            message += release_error;
                        }
                    }
                    return fail_after_checkpoint(message);
                };

                tx.scheduled_batch = owner.scheduleNextBatch(scheduler);
                if (!tx.scheduled_batch.ok)
                {
                    return fail_after_checkpoint(
                        std::string("decodeStepBatch() request-batched stochastic "
                                    "verifier scheduling failed: ") +
                        tx.scheduled_batch.error);
                }

                std::string produce_error;
                tx.produced =
                    produce_stochastic_outcomes(
                        tx.scheduled_batch,
                        &tx.device_outcomes,
                        &produce_error);
                if (!tx.produced)
                {
                    std::string message =
                        "decodeStepBatch() request-batched stochastic "
                        "resident verifier production failed";
                    if (!produce_error.empty())
                    {
                        message += ": ";
                        message += produce_error;
                    }
                    return release_and_fail(std::move(message));
                }

                if (!resident_request_batch_outcome_ready ||
                    !resident_request_batch_outcome.valid() ||
                    resident_request_batch_verifier_rows <= 0)
                {
                    return release_and_fail(
                        "decodeStepBatch() request-batched stochastic "
                        "resident verifier produced no valid device outcome");
                }

                DeviceSpeculativePublicationRequest publication_request;
                publication_request.outcome = resident_request_batch_outcome;
                publication_request.max_state_commit_rows =
                    resident_request_batch_verifier_rows;
                publication_request.publish_mtp_shifted_kv =
                    tx.scheduled_batch.requires_shifted_kv_publication;

                std::string publication_error;
                {
                    PerfStatsCollector::ScopedTimer publication_timer(
                        "mtp",
                        "request_batch_publish_accepted_state_device_resident",
                        "decode",
                        {},
                        {{"sampling", "stochastic"},
                         {"request_count",
                          std::to_string(publication_request.requestCount())},
                         {"logical_verifier_rows",
                          std::to_string(
                              publication_request
                                  .logicalVerifierRowsPerRequest())},
                         {"physical_verifier_rows",
                          std::to_string(
                              publication_request
                                  .physicalVerifierRowsPerRequest())}});
                    tx.published =
                        runner_->publishAcceptedMTPSpecStateBatchFromDeviceOutcome(
                            publication_request,
                            &publication_error);
                }
                if (!tx.published)
                {
                    std::string message =
                        "decodeStepBatch() request-batched stochastic "
                        "resident state publication failed";
                    if (!publication_error.empty())
                    {
                        message += ": ";
                        message += publication_error;
                    }
                    return release_and_fail(std::move(message));
                }

                const DeviceResidentLogicalSequenceStateHandle logical_state =
                    runner_->deviceResidentLogicalSequenceState();
                if (!logical_state.valid() ||
                    logical_state.request_count <
                        tx.scheduled_batch.request_count)
                {
                    return release_and_fail(
                        "decodeStepBatch() request-batched stochastic resident "
                        "publication produced no valid logical-state mailbox");
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "request_batch_resident_plan_checks",
                    1.0,
                    "decode",
                    {},
                    {{"request_count",
                      std::to_string(tx.scheduled_batch.request_count)},
                     {"state_owner", "device_transaction"}});

                std::string commit_error;
                tx.committed = owner.commitInFlightBatch(&commit_error);
                if (!tx.committed)
                {
                    return fail_after_checkpoint(
                        std::string("decodeStepBatch() request-batched stochastic "
                                    "resident publication succeeded but owner "
                                    "commit failed: ") +
                        commit_error);
                }

                PerfStatsCollector::addCounter(
                    "mtp",
                    "request_batch_device_resident_state_publications",
                    1.0,
                    "decode",
                    {},
                    {{"request_count",
                      std::to_string(publication_request.requestCount())},
                     {"logical_verifier_rows",
                      std::to_string(
                          publication_request.logicalVerifierRowsPerRequest())}});
                tx.ok = true;
                return completeDeviceResidentBatchGeneration(
                    publication_request,
                    DeviceGenerationSamplingMode::Stochastic,
                    draft_depth,
                    draft_depth,
                    std::move(batch_result));
            }
            else
            {
                tx =
                    executeOwnedMTPDeviceOutcomeScheduledBatchTransactionAndPublish(
                        owner,
                        scheduler,
                        produce_stochastic_outcomes,
                        publish);
                if (!tx.ok)
                {
                    return fail_after_checkpoint(
                        std::string("decodeStepBatch() request-batched stochastic "
                                    "verifier transaction failed: ") +
                        tx.error);
                }
            }

            scheduled_request_ids = tx.scheduled_batch.request_ids;
            scheduled_base_cached_tokens =
                tx.scheduled_batch.base_cached_tokens;
            catchup_results.reserve(tx.device_outcomes.size());
            for (size_t i = 0; i < tx.device_outcomes.size(); ++i)
            {
                MTPDecodeCatchupGreedyResult catchup =
                    buildAllPositionMTPDecodeCatchupFromDeviceBatchOutcome(
                        tx.scheduled_batch.greedy_requests[i],
                        tx.device_outcomes[i]);
                if (!catchup.ok)
                {
                    return fail_after_checkpoint(
                        std::string("decodeStepBatch() stochastic catch-up "
                                    "summary failed: ") +
                        catchup.error);
                }
                catchup_results.push_back(std::move(catchup));
            }
        }

        if (static_cast<int>(scheduled_request_ids.size()) !=
                static_cast<int>(catchup_results.size()) ||
            scheduled_request_ids.empty())
        {
            return fail_after_checkpoint(
                "decodeStepBatch() request-batched verifier transaction "
                "returned inconsistent request vectors");
        }

        for (size_t i = 0; i < scheduled_request_ids.size(); ++i)
        {
            const int request = scheduled_request_ids[i];
            if (request < 0 || request >= request_batch)
            {
                return fail_after_checkpoint(
                    "decodeStepBatch() request-batched verifier returned an "
                    "out-of-range request id");
            }

            BatchedDecodeRequestState &state =
                batched_request_states_[static_cast<size_t>(request)];
            GenerationResult &request_result =
                batch_result.requests[static_cast<size_t>(request)];
            const MTPDecodeCatchupGreedyResult &catchup =
                catchup_results[i];
            if (!catchup.ok || catchup.accepted_tokens.empty())
            {
                return fail_after_checkpoint(
                    "decodeStepBatch() request-batched verifier produced an "
                    "invalid catch-up result");
            }
            if (i >= scheduled_base_cached_tokens.size() ||
                catchup.target_verifier_state_commit_count < 0)
            {
                return fail_after_checkpoint(
                    "decodeStepBatch() request-batched verifier produced "
                    "invalid accepted-state length metadata");
            }
            state.logical_tokens =
                scheduled_base_cached_tokens[i] +
                catchup.target_verifier_state_commit_count;
            PerfStatsCollector::addCounterWithSequence(
                "mtp",
                "request_batch_resident_planning_position_commits",
                1.0,
                {static_cast<uint64_t>(state.logical_tokens)},
                "decode",
                {},
                {{"request", std::to_string(request)}});

            /*
             * A resident GPU batch has already consumed the previously emitted
             * condition token in its grouped main forward. Verifier row zero is
             * therefore the first newly sampled target token and must be
             * returned. CPU's older host-batch lane still includes the prior
             * response token at row zero until it adopts the same grouped
             * condition-forward contract.
             */
            auto new_token_begin =
                catchup.accepted_tokens.begin() +
                (request_batch_condition_advanced ? 0 : 1);
            request_result.tokens.insert(
                request_result.tokens.end(),
                new_token_begin,
                catchup.accepted_tokens.end());
            if (sampled_terminal_samplers[static_cast<size_t>(request)].has_value())
            {
                state.sampler =
                    *sampled_terminal_samplers[static_cast<size_t>(request)];
            }

            const bool has_ready_token =
                !catchup.stopped_on_output &&
                catchup.all_speculative_accepted &&
                catchup.ready_token >= 0;
            request_result.is_complete =
                catchup.stopped_on_output;
            state.is_complete = request_result.is_complete;
            state.last_token = !request_result.tokens.empty()
                                   ? request_result.tokens.back()
                                   : catchup.accepted_tokens.back();

            if (has_ready_token)
            {
                state.prefill_logits_ready = true;
                state.ready_sampled_token = catchup.ready_token;
                state.ready_sampled_params = active_sampling_params_;
            }
            else
            {
                state.prefill_logits_ready = false;
                state.ready_sampled_token.reset();
                state.ready_sampled_params.reset();
            }

            for (auto it = new_token_begin; it != catchup.accepted_tokens.end(); ++it)
            {
                const int32_t token = *it;
                state.sampler.record_token(token);
            }
            ++mtp_stats_.verifier_runs;
            mtp_stats_.verifier_token_count +=
                static_cast<uint64_t>(catchup.main_forward_token_count);
            const int accepted_speculative =
                std::max(0, catchup.accepted_speculative_prefix);
            mtp_stats_.accepted_tokens +=
                static_cast<uint64_t>(accepted_speculative);
            const int rejected =
                catchup.all_speculative_accepted ? 0 : 1;
            mtp_stats_.rejected_tokens += static_cast<uint64_t>(rejected);
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "request_batched_verifier_transactions",
            1.0,
            "decode",
            {},
            {{"requests", std::to_string(scheduled_request_ids.size())},
             {"draft_depth", std::to_string(draft_depth)},
             {"mode", stochastic_batch_verify ? "stochastic" : "greedy"}});
        return batch_result;
    }

} // namespace llaminar2
