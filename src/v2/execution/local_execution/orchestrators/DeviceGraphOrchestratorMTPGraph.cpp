/**
 * @file DeviceGraphOrchestratorMTPGraph.cpp
 * @brief DeviceGraphOrchestrator method shard (22591..27487).
 */
#include "DeviceGraphOrchestratorInternal.h"

namespace llaminar2
{

    bool DeviceGraphOrchestrator::prepareHostedMTPStochasticBranch(
        int rows, int source_rows, void *stream, std::string *error)
    {
        const auto fail = [&](const std::string &why) { if (error) *error = why; return false; };
        const auto generation = workspaceGeneration(state_.device_id);
        if (mtpVerifierGraphFamily() != MTPVerifierGraphFamily::ExactRows || !stream || sparseQueryParentLaunched())
            return fail("Stochastic family preparation requires admitted exact widths and idle parents");
        const auto &source_target = mtpStochasticTargetDistributionCache(source_rows);
        const auto &source_outcome = mtpStochasticOutcomeCache(source_rows);
        if (generation == 0 || !source_target.valid || !source_target.graph || !source_target.stage ||
            source_target.workspace_generation != generation || !source_outcome.valid || !source_outcome.graph ||
            !source_outcome.stage || source_outcome.workspace_generation != generation ||
            !source_target.segment_cache.deviceLoopGraphTemplate(*source_target.graph, error) ||
            !source_outcome.segment_cache.deviceLoopGraphTemplate(*source_outcome.graph, error))
            return fail("Stochastic family lost its original completed initial-transaction producer");
        const auto &original_target = source_target.stage->getParams();
        const auto &original_outcome = source_outcome.stage->getParams();
        const auto seeds = arena_ ? arena_->getSharedTensor(BufferId::SAMPLING_REQUEST_SEEDS) : nullptr;
        const bool known_law = original_outcome.verification == MTPStochasticOutcomeStage::Verification::SerialEquivalent ||
            original_outcome.verification == MTPStochasticOutcomeStage::Verification::OneHotProbabilityRejection;
        if (!known_law || original_target.device_id != state_.device_id || original_outcome.device_id != state_.device_id ||
            !original_target.backend || original_target.backend != original_outcome.backend ||
            original_target.row_count != source_rows || original_target.first_logit_row != 0 || original_target.first_target_slot != 0 ||
            original_outcome.request_count != 1 || original_outcome.comparison_rows_per_request != source_rows - 1 ||
            original_outcome.verifier_row_capacity != source_rows || original_outcome.verifier_input_token_stride != source_rows ||
            original_outcome.sampled_target_token_stride != source_rows || original_outcome.accepted_row_stride != source_rows ||
            original_target.top_k != original_outcome.top_k ||
            original_target.target_row_stride != original_outcome.target_distribution_row_stride ||
            original_target.target_token_ids_device != original_outcome.target_token_ids_device ||
            original_target.target_probs_device != original_outcome.target_probs_device ||
            original_target.target_token_ids_device != stochastic_target_token_ids_dev_ ||
            original_target.target_probs_device != stochastic_target_probs_dev_ ||
            original_outcome.verifier_input_tokens_device != mtp_verifier_input_tokens_dev_ ||
            original_outcome.generation_control_device != device_generation_storage_.control_device ||
            original_outcome.generation_control_stride != device_generation_storage_.control_stride ||
            original_target.active_rows_device != mtp_verifier_request_lengths_dev_ || !seeds ||
            !original_outcome.threshold_seeds_device ||
            original_outcome.threshold_seeds_device != seeds->gpu_data_ptr())
            return fail("Stochastic exact-row source geometry, law, sampler rows or admitted controls disagree");
        // Validate the original join before copying immutable Params or recording
        // siblings. Pointer inspection reads arena metadata, never device values.
        auto target_params = original_target;
        auto outcome_params = original_outcome;
        target_params.row_count = rows;
        outcome_params.comparison_rows_per_request = rows - 1;
        outcome_params.accepted_row_stride = rows;
        outcome_params.verifier_input_token_stride = rows;
        outcome_params.verifier_row_capacity = rows;
        outcome_params.sampled_target_token_stride = rows;
        auto &target = mtpStochasticTargetDistributionCache(rows);
        auto &outcome = mtpStochasticOutcomeCache(rows);
        if (!target.valid || !target.stage || target.workspace_generation != generation ||
            !target.stage->hasSameCaptureIdentity(target_params))
        {
            auto stage = std::make_unique<MTPStochasticTargetDistributionStage>(std::move(target_params));
            auto *binding = stage.get();
            auto graph = std::make_unique<ComputeGraph>();
            graph->addNode("mtp_stochastic_target_distribution", std::move(stage), state_.device_id);
            invalidateSparseQueryParents();
            target.invalidate();
            target.stage = binding;
            target.graph = std::move(graph);
            target.workspace_generation = generation;
            target.valid = true;
        }
        if (!outcome.valid || !outcome.stage || outcome.workspace_generation != generation ||
            !outcome.stage->hasSameCaptureIdentity(outcome_params))
        {
            auto stage = std::make_unique<MTPStochasticOutcomeStage>(std::move(outcome_params));
            auto *binding = stage.get();
            auto graph = std::make_unique<ComputeGraph>();
            graph->addNode("mtp_stochastic_outcome", std::move(stage), state_.device_id);
            invalidateSparseQueryParents();
            outcome.invalidate();
            outcome.stage = binding;
            outcome.graph = std::move(graph);
            outcome.workspace_generation = generation;
            outcome.valid = true;
        }
        return executeMTPStochasticTargetDistributionCaptured(stream, rows, error,
                   DeviceGraphExecutor::GraphInitialSubmissionPolicy::MaterializeWithoutLaunch) &&
            executeMTPStochasticOutcomeCaptured(stream, 1, rows - 1, error,
                   DeviceGraphExecutor::GraphInitialSubmissionPolicy::MaterializeWithoutLaunch);
    }

    bool DeviceGraphOrchestrator::validateHostedMTPSidecarReplay(
        MTPSidecarCaptureRole role,
        int row_count,
        const ComputeGraph **out_graph,
        std::string *error) const
    {
        auto fail = [&](const std::string &reason) -> bool
        {
            if (error)
                *error = reason;
            return false;
        };
        if (error)
            error->clear();
        if (out_graph)
            *out_graph = nullptr;
        if (!state_.device_id.is_gpu())
            return fail("hosted MTP sidecar replay requires a GPU runner");

        const MTPSidecarGraphCache *cache = nullptr;
        try
        {
            (void)mtp_sidecar_capture_layout_.conditionTokenSlot(
                role, row_count);
            switch (role)
            {
            case MTPSidecarCaptureRole::Full:
                cache = &mtp_sidecar_depth0_device_token_cache_;
                break;
            case MTPSidecarCaptureRole::Chained:
                cache = &mtp_sidecar_depth0_chained_device_token_cache_;
                break;
            case MTPSidecarCaptureRole::KVOnly:
                return fail(
                    "KV-only sidecars are not members of a hosted MTP transaction branch");
            }
        }
        catch (const std::exception &exception)
        {
            return fail(exception.what());
        }

        const uint64_t generation = workspaceGeneration(state_.device_id);
        if (!cache || !cache->valid || !cache->graph || generation == 0 ||
            cache->workspace_generation != generation)
        {
            return fail("hosted MTP sidecar cache is stale or incomplete");
        }
        if (!cache->uses_device_token_ids ||
            !cache->uses_device_position_ids ||
            !cache->condition_token_device ||
            !cache->position_ids_device)
        {
            return fail(
                "hosted MTP sidecar does not own stable device token and position inputs");
        }
        const int64_t cached_rows =
            static_cast<int64_t>(cache->seq_len) *
            static_cast<int64_t>(cache->batch_size);
        if (row_count <= 0 || cached_rows != row_count)
        {
            return fail(
                "hosted MTP sidecar row geometry does not match the retained cache");
        }
        if (!cache->segment_cache.initialized ||
            cache->segment_cache.needs_capture ||
            !cache->segment_cache.capture_stream ||
            !cache->segment_cache.gpu_ctx_ref ||
            cache->segment_cache.segments.empty())
        {
            return fail(
                "hosted MTP sidecar has not reached steady retained replay state");
        }
        if (out_graph)
            *out_graph = cache->graph.get();
        return true;
    }

    std::optional<MoESparseRequestIdentity>
    DeviceGraphOrchestrator::currentMoESparseRequestIdentity() const noexcept
    {
        const auto plan = graph_builder_
                              ? graph_builder_->config()
                                    .moe.routed_expert_plan
                              : nullptr;
        if (plan && plan->usesExpertOverlayAuthority())
        {
            return MoESparseRequestIdentity::expertOverlayRoot(
                moe_overlay_collective_request_generation_);
        }
        return MoESparseRequestIdentity::localRequestState(session_epoch_);
    }

    std::optional<IComputeStage::MoEOverlayCollectiveRuntimeParams>
    DeviceGraphOrchestrator::resolveMTPMoEOverlayCollectiveRuntimeParams(
        const MoEOverlayInferenceParticipantGraphScope &graph_scope,
        IComputeStage::MoEOverlayCollectiveRuntimeParams::ExecutionSemantics
            execution_semantics,
        std::optional<int> expected_transaction_draft_depth,
        int sparse_graph_depth,
        std::string *error)
    {
        using RuntimeParams =
            IComputeStage::MoEOverlayCollectiveRuntimeParams;
        using Semantics = RuntimeParams::ExecutionSemantics;
        const auto reject = [&](std::string reason)
            -> std::optional<RuntimeParams>
        {
            if (error)
                *error = std::move(reason);
            return std::nullopt;
        };
        if (error)
            error->clear();

        if ((expected_transaction_draft_depth &&
             *expected_transaction_draft_depth <= 0) ||
            sparse_graph_depth < 0 ||
            (execution_semantics != Semantics::MTPDraft &&
             execution_semantics != Semantics::GroupedVerifier) ||
            (execution_semantics == Semantics::GroupedVerifier &&
             (!expected_transaction_draft_depth ||
              sparse_graph_depth !=
                  *expected_transaction_draft_depth)))
        {
            return reject(
                "MTP sparse runtime binding received an invalid typed role or depth");
        }
        if (!graph_scope.ready())
            return reject(graph_scope.error());

        const auto request_identity = currentMoESparseRequestIdentity();
        if (!request_identity)
        {
            return reject(
                "MTP sparse runtime binding has no admitted request identity");
        }

        RuntimeParams params{
            .generation_id = request_identity->generation(),
            .step_id = 0,
            .execution_semantics = execution_semantics,
            .mtp_depth = sparse_graph_depth,
            .placement_epoch = 0,
        };

        if (moe_overlay_inference_transaction_coordinator_)
        {
            /*
             * Once a cross-rank authority is installed, an inert scope is a
             * lifecycle error. Falling back to process-local identity here
             * would let a remote follower and continuation use different
             * operation ids while appearing to execute the same graph.
             */
            if (!graph_scope.active())
            {
                return reject(
                    "Distributed MTP sparse graph has no active transaction binding");
            }
            const auto &binding = graph_scope.binding();
            if (binding.request_generation == 0 ||
                binding.request_generation !=
                    moe_overlay_collective_request_generation_ ||
                binding.descriptor.draft_depth <= 0 ||
                (expected_transaction_draft_depth &&
                 binding.descriptor.draft_depth !=
                     *expected_transaction_draft_depth))
            {
                return reject(
                    "MTP sparse transaction binding disagrees with request generation or selected depth");
            }
            params.generation_id = binding.request_generation;
            params.step_id = binding.descriptor.logical_step_id;
            params.placement_epoch =
                binding.descriptor.placement_epoch;
        }
        else if (state_.device_id.is_cpu())
        {
            // CPU NodeTP has no heterogeneous transaction follower, but its
            // rank-batch stages still require an explicit operation identity.
            // Issue once at the graph owner, not independently in every stage
            // or cache variant. Corrections can revisit a token position.
            const auto operation = moe_sparse_host_operation_sequence_.issue();
            if (!operation)
                return reject("CPU MTP sparse operation identity exhausted");
            params.step_id = *operation;
        }

        /*
         * Native GPU epochs retain their device-owned invocation identity;
         * this host binding supplies only the admitted generation and phase.
         * CPU execution instead consumes the host operation issued above.
         */
        if (!params.valid() || !params.hasExecutionSemantics())
        {
            return reject(
                "MTP sparse runtime binding has no admitted request generation or execution semantics");
        }
        return params;
    }

    bool DeviceGraphOrchestrator::replayHostedMTPSidecar(
        MTPSidecarCaptureRole role,
        int draft_depth,
        void **out_producer_stream,
        std::string *error)
    {
        auto fail = [&](const std::string &reason) -> bool
        {
            if (error)
                *error = reason;
            return false;
        };
        if (error)
            error->clear();
        if (out_producer_stream)
            *out_producer_stream = nullptr;

        const ComputeGraph *validated_graph = nullptr;
        std::string validation_error;
        if (!validateHostedMTPSidecarReplay(
                role,
                /*row_count=*/1,
                &validated_graph,
                &validation_error))
        {
            return fail(validation_error);
        }

        MTPSidecarGraphCache *cache =
            role == MTPSidecarCaptureRole::Full
                ? &mtp_sidecar_depth0_device_token_cache_
                : role == MTPSidecarCaptureRole::Chained
                      ? &mtp_sidecar_depth0_chained_device_token_cache_
                      : nullptr;
        if (!cache || cache->graph.get() != validated_graph)
            return fail("hosted MTP sidecar cache identity changed after validation");
        void *const replay_stream = cache->segment_cache.capture_stream;
        auto &loop = deviceGenerationLoopGraph();
        IWorkerGPUContext *const replay_context =
            cache->segment_cache.gpu_ctx_ref;
        if (!loop.stream || !replay_context ||
            !cache->segment_cache.orderCaptureStreamAfter(
                replay_context, loop.stream.get()))
            return fail("hosted MTP sidecar scheduler handoff failed");
        if (moe_overlay_epoch_execution_binding_ &&
            moe_overlay_epoch_lease_lifecycle_->load() !=
                MoEOverlayEpochLeaseState::HostedParent)
        {
            return fail(
                "hosted MTP sidecar has no scheduler-owned ExpertOverlay reader");
        }

        MoEOverlayInferenceExecutionDescriptor descriptor{
            .graph_role = MoEOverlayInferenceGraphRole::MTPDraft,
            .logical_step_id = 0,
            .placement_epoch =
                moe_overlay_inference_transaction_coordinator_
                    ? moe_overlay_inference_transaction_coordinator_
                          ->currentPlacementEpoch()
                    : 0,
            .request_count = 1,
            .logical_rows_per_request = 1,
            .physical_rows_per_request = 1,
            .draft_depth = -1,
            .sidecar_depth = -1,
        };
        MoEOverlayInferenceParticipantGraphScope graph_scope(
            moe_overlay_inference_transaction_coordinator_,
            descriptor,
            moe_overlay_inference_transaction_participant_index_);
        if (!graph_scope.ready())
            return fail(graph_scope.error());

        const auto sparse_params =
            resolveMTPMoEOverlayCollectiveRuntimeParams(
                graph_scope,
                IComputeStage::MoEOverlayCollectiveRuntimeParams::
                    ExecutionSemantics::MTPDraft,
                std::optional<int>{draft_depth},
                /*sparse_graph_depth=*/0,
                &validation_error);
        if (!sparse_params)
            return fail(validation_error);
        for (IComputeStage *stage :
             cache->moe_overlay_collective_runtime_stages)
        {
            if (!stage)
                return fail("hosted MTP sidecar contains a null sparse stage");
            stage->updateMoEOverlayCollectiveRuntimeParams(*sparse_params);
        }

        IDeviceContext *const ctx = getDeviceContext(state_.device_id);
        auto capture_policy_result = buildDecodeCapturePolicyForGraph(
            *cache->graph,
            !cache->collective_nodes.empty(),
            ctx,
            "hosted MTP sidecar");
        if (!capture_policy_result)
            return fail("hosted MTP sidecar native capture policy is unavailable");
        auto capture_policy = std::move(*capture_policy_result);
        if (moe_overlay_epoch_execution_binding_ ||
            graph_scope.ownsTicketAuthority())
        {
            const auto existing =
                std::move(capture_policy.launch_dependency);
            capture_policy.launch_dependency =
                [this,
                 &graph_scope,
                 existing,
                 device = state_.device_id](
                    DeviceGraphExecutor::GraphExecutableLaunchPhase phase,
                    void *execution_stream) -> bool
            {
                if (existing && !existing(phase, execution_stream))
                    return false;
                if (!consumeMoEOverlayEpochForHostedChild(
                        execution_stream,
                        "hosted_mtp_sidecar_executable"))
                {
                    return false;
                }
                if (!graph_scope.ownsTicketAuthority())
                    return true;
                std::string arm_error;
                if (!graph_scope.armForExecutableLaunch(
                        execution_stream, &arm_error))
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Hosted MTP sidecar could "
                        "not arm its ExpertOverlay follower at the retained "
                        "executable launch edge on "
                        << device.toString() << ": " << arm_error);
                    return false;
                }
                return true;
            };
        }
        if (!ctx || !capture_policy.allow_cached_graph_replay ||
            cache->segment_cache.graph_replay_plan_policy !=
                capture_policy.graph_replay_plan_policy)
        {
            return fail(
                "hosted MTP sidecar replay authority changed after setup");
        }

        cache->graph->reset();
        const bool submitted = executor_.executeWithCachedGraphReplay(
            *cache->graph,
            ctx,
            cache->segment_cache,
            replay_stream,
            cache->segment_cache.gpu_ctx_ref,
            &cache->collective_nodes,
            capture_policy.collectives_graph_capturable,
            /*force_recapture=*/false,
            /*defer_final_sync=*/true,
            {},
            capture_policy.graph_replay_plan_policy,
            capture_policy.launch_dependency,
            /*event_published_outputs=*/{},
            capture_policy.retained_parent_composer);
        if (!submitted)
        {
            std::string finish_error;
            (void)graph_scope.finish(false, &finish_error);
            return fail("hosted MTP sidecar retained replay submission failed");
        }
        if (!cache->segment_cache.orderStreamAfterCapture(
                replay_context, loop.stream.get()))
        {
            std::string finish_error;
            (void)graph_scope.finish(false, &finish_error);
            return fail("hosted MTP sidecar completion handoff failed");
        }

        /*
         * The scheduler-stream wait above is the participant terminal. Publish
         * protocol completion only after that edge exists; finishing earlier
         * lets another rank retire the graph group while this participant's
         * retained executable is still using its immutable overlay ticket.
         */
        std::string finish_error;
        if (!graph_scope.finish(true, &finish_error))
            return fail(finish_error);

        if (out_producer_stream)
            *out_producer_stream = replay_stream;
        PerfStatsCollector::addCounter(
            "mtp",
            "hosted_retained_sidecar_replays",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"role",
              role == MTPSidecarCaptureRole::Full ? "full" : "chained"},
             {"draft_depth", std::to_string(draft_depth)},
             {"replay_units",
              std::to_string(cache->segment_cache.segments.size())}});
        return true;
    }

    bool DeviceGraphOrchestrator::replayHostedMTPGroupedVerifier(
        const ForwardGraphSignature &signature,
        int draft_depth,
        void **out_producer_stream,
        std::string *error)
    {
        auto fail = [&](const std::string &reason) -> bool
        {
            if (error)
                *error = reason;
            return false;
        };
        if (error)
            error->clear();
        if (out_producer_stream)
            *out_producer_stream = nullptr;
        if (!forward_engine_)
            return fail("hosted grouped verifier has no forward engine");
        if (signature.seq_len != draft_depth + 1 ||
            signature != retainedMTPVerifierSignature(signature.seq_len,
                signature.mtp_verifier_outcome_graph_mode, sparseQueryGraphProgram()))
            return fail("hosted grouped verifier ticket does not name its exact retained row family");
        if (moe_overlay_epoch_execution_binding_ &&
            moe_overlay_epoch_lease_lifecycle_->load() !=
                MoEOverlayEpochLeaseState::HostedParent)
        {
            return fail(
                "hosted grouped verifier has no active parent-owned ExpertOverlay reader");
        }

        std::string retained_error;
        const auto retained = forward_engine_->retainedDecodeGraph(
            signature, &retained_error);
        if (!retained)
            return fail(retained_error);

        MoEOverlayInferenceExecutionDescriptor descriptor{
            .graph_role = MoEOverlayInferenceGraphRole::MTPGroupedVerifier,
            .logical_step_id = 0,
            .placement_epoch =
                moe_overlay_inference_transaction_coordinator_
                    ? moe_overlay_inference_transaction_coordinator_
                          ->currentPlacementEpoch()
                    : 0,
            .request_count = signature.batch_size,
            .logical_rows_per_request = draft_depth + 1,
            .physical_rows_per_request = signature.seq_len,
            .draft_depth = draft_depth,
            .sidecar_depth = -1,
        };
        MoEOverlayInferenceParticipantGraphScope graph_scope(
            moe_overlay_inference_transaction_coordinator_,
            descriptor,
            moe_overlay_inference_transaction_participant_index_);
        if (!graph_scope.ready())
            return fail(graph_scope.error());

        const auto params =
            resolveMTPMoEOverlayCollectiveRuntimeParams(
                graph_scope,
                IComputeStage::MoEOverlayCollectiveRuntimeParams::
                    ExecutionSemantics::GroupedVerifier,
                std::optional<int>{draft_depth},
                /*sparse_graph_depth=*/draft_depth,
                &retained_error);
        if (!params)
            return fail(retained_error);

        void *producer_stream = nullptr;
        IDeviceContext *const ctx = getDeviceContext(state_.device_id);
        DeviceGraphExecutor::GraphLaunchDependencyHook launch_dependency;
        if (moe_overlay_epoch_execution_binding_ ||
            graph_scope.ownsTicketAuthority())
        {
            launch_dependency =
                [this, &graph_scope, device = state_.device_id](
                    DeviceGraphExecutor::GraphExecutableLaunchPhase,
                    void *execution_stream) -> bool
            {
                if (!consumeMoEOverlayEpochForHostedChild(
                        execution_stream,
                        "hosted_mtp_grouped_verifier_executable"))
                {
                    return false;
                }
                if (!graph_scope.ownsTicketAuthority())
                    return true;
                std::string arm_error;
                if (!graph_scope.armForExecutableLaunch(
                        execution_stream, &arm_error))
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Hosted grouped verifier "
                        "could not arm its ExpertOverlay follower at the "
                        "retained executable launch edge on "
                        << device.toString() << ": " << arm_error);
                    return false;
                }
                return true;
            };
        }
        const bool submitted = forward_engine_->replayRetainedDecodeGraph(
            signature,
            ctx,
            *params,
            launch_dependency,
            *this,
            deviceGenerationLoopGraph().stream.get(),
            &producer_stream,
            &retained_error);
        std::string finish_error;
        if (!graph_scope.finish(submitted, &finish_error))
            return fail(finish_error);
        if (!submitted)
            return fail(retained_error);
        if (producer_stream != retained->stream)
        {
            return fail(
                "hosted grouped verifier published an invalid completion stream");
        }
        if (out_producer_stream)
            *out_producer_stream = producer_stream;
        return true;
    }

    std::optional<
        DeviceGraphExecutor::GraphSegmentCache::DeviceLoopGraphTemplateView>
    DeviceGraphOrchestrator::mtpSidecarDeviceLoopGraphTemplate(
        MTPSidecarCaptureRole role,
        int row_count,
        std::string *error) const
    {
        using TemplateView = DeviceGraphExecutor::GraphSegmentCache::
            DeviceLoopGraphTemplateView;
        auto reject = [&](const std::string &reason)
            -> std::optional<TemplateView>
        {
            if (error)
                *error = reason;
            return std::nullopt;
        };
        if (error)
            error->clear();
        if (!state_.device_id.is_gpu())
            return reject("MTP sidecar loop composition requires a GPU runner");

        const MTPSidecarGraphCache *cache = nullptr;
        try
        {
            (void)mtp_sidecar_capture_layout_.conditionTokenSlot(
                role, row_count);
            switch (role)
            {
            case MTPSidecarCaptureRole::Full:
                cache = &mtp_sidecar_depth0_device_token_cache_;
                break;
            case MTPSidecarCaptureRole::Chained:
                cache = &mtp_sidecar_depth0_chained_device_token_cache_;
                break;
            case MTPSidecarCaptureRole::KVOnly:
                if (row_count == 1)
                {
                    cache = &mtp_sidecar_depth0_kv_only_device_token_cache_;
                    break;
                }
                {
                    const size_t cache_index =
                        mtp_sidecar_capture_layout_.kvOnlyBatchCacheIndex(
                            row_count);
                    if (cache_index >=
                            mtp_sidecar_depth0_kv_only_batch_caches_.size() ||
                        !mtp_sidecar_depth0_kv_only_batch_caches_[cache_index])
                    {
                        return reject(
                            "MTP KV-only sidecar graph cache is not allocated for row count " +
                            std::to_string(row_count));
                    }
                    cache =
                        mtp_sidecar_depth0_kv_only_batch_caches_[cache_index]
                            .get();
                }
                break;
            }
        }
        catch (const std::exception &exception)
        {
            return reject(exception.what());
        }

        if (!cache || !cache->valid || !cache->graph)
            return reject("MTP sidecar graph is not initialized");
        if (!cache->uses_device_token_ids ||
            !cache->uses_device_position_ids ||
            !cache->condition_token_device ||
            !cache->position_ids_device)
        {
            return reject(
                "MTP sidecar graph does not own device-resident token and position inputs");
        }
        const int64_t cached_rows =
            static_cast<int64_t>(cache->seq_len) *
            static_cast<int64_t>(cache->batch_size);
        if (cached_rows != row_count)
        {
            return reject(
                "MTP sidecar graph geometry does not match requested row count: cached=" +
                std::to_string(cached_rows) +
                " requested=" + std::to_string(row_count));
        }

        return cache->segment_cache.deviceLoopGraphTemplate(
            *cache->graph,
            error);
    }

    std::optional<
        DeviceGraphExecutor::GraphSegmentCache::DeviceLoopGraphTemplateView>
    DeviceGraphOrchestrator::
        mtpDraftTokenPublicationDeviceLoopGraphTemplate(
            int slot,
            std::string *error) const
    {
        using TemplateView = DeviceGraphExecutor::GraphSegmentCache::
            DeviceLoopGraphTemplateView;
        auto reject = [&](const std::string &reason)
            -> std::optional<TemplateView>
        {
            if (error)
                *error = reason;
            return std::nullopt;
        };
        if (error)
            error->clear();
        if (!state_.device_id.is_gpu())
        {
            return reject(
                "MTP draft-publication loop composition requires a GPU runner");
        }
        if (slot < 0 || slot >= mtp_max_draft_depth_ ||
            slot >= static_cast<int>(
                        mtp_draft_token_publication_graphs_.size()) ||
            !mtp_draft_token_publication_graphs_[
                static_cast<size_t>(slot)])
        {
            return reject(
                "MTP draft-publication slot is outside the preallocated graph family");
        }

        const auto &cache = *mtp_draft_token_publication_graphs_[
            static_cast<size_t>(slot)];
        if (!cache.valid || !cache.graph || !cache.stage)
            return reject("MTP draft-publication graph is not initialized");
        if (cache.workspace_generation == 0 ||
            cache.workspace_generation != workspaceGeneration(state_.device_id))
        {
            return reject(
                "MTP draft-publication graph belongs to a stale workspace generation");
        }
        const auto &params = cache.stage->getParams();
        if (params.destination_slot != slot ||
            params.prior_draft_count != slot ||
            params.logits_row != 0 || !params.logits_row_device ||
            !params.draft_tokens_device || !params.draft_values_device ||
            !params.next_chain_condition_token_device ||
            !params.next_chain_position_id_device ||
            params.chain_position_increment != 1)
        {
            return reject(
                "MTP draft-publication graph is not the canonical scalar fixed-depth producer-owned binding");
        }

        return cache.segment_cache.deviceLoopGraphTemplate(
            *cache.graph,
            error);
    }

    std::optional<
        DeviceGraphExecutor::GraphSegmentCache::DeviceLoopGraphTemplateView>
    DeviceGraphOrchestrator::
        mtpVerifierPreparationDeviceLoopGraphTemplate(
            int request_count,
            int padded_seq_len,
            std::string *error) const
    {
        using TemplateView = DeviceGraphExecutor::GraphSegmentCache::
            DeviceLoopGraphTemplateView;
        auto reject = [&](const std::string &reason)
            -> std::optional<TemplateView>
        {
            if (error)
                *error = reason;
            return std::nullopt;
        };
        if (error)
            error->clear();
        if (!state_.device_id.is_gpu())
        {
            return reject(
                "grouped-verifier preparation loop composition requires a GPU runner");
        }
        const MTPVerifierPreparationGraphKey key{
            .control_policy = MTPVerifierPreparationControlPolicy::
                DeviceGenerationControlled,
            .request_count = request_count,
            .padded_seq_len = padded_seq_len,
        };
        std::string registry_error;
        const auto registry_index =
            mtpVerifierPreparationRegistryIndex(key, &registry_error);
        if (!registry_index)
            return reject(registry_error);

        const uint64_t generation = workspaceGeneration(state_.device_id);
        const auto &cache =
            *mtp_verifier_preparation_graphs_[*registry_index];
        if (!cache.valid || cache.key != key || !cache.graph || !cache.stage ||
            generation == 0 || cache.workspace_generation != generation)
        {
            return reject(
                "exact grouped-verifier preparation registry slot is stale or incomplete");
        }

        const auto &params = cache.stage->getParams();
        if (params.device_id != state_.device_id || !params.backend ||
            params.request_count != request_count ||
            params.padded_seq_len != padded_seq_len ||
            static_cast<int>(params.token_rows.size()) != request_count ||
            static_cast<int>(params.main_kv_checkpoints.size()) !=
                request_count ||
            !params.base_cached_tokens_device ||
            !params.position_ids_device || !params.request_lengths_device ||
            !params.base_cached_tokens_snapshot_device ||
            params.generation_control_device !=
                device_generation_storage_.control_device ||
            params.generation_control_stride !=
                device_generation_storage_.control_stride ||
            params.valid_graph_row_count <= 0 ||
            params.valid_graph_row_count > request_count * padded_seq_len ||
            (!params.valid_graph_rows_device &&
             params.valid_graph_row_count % request_count != 0))
        {
            return reject(
                "exact grouped-verifier preparation has incomplete device-owned bindings");
        }
        for (const auto &row : params.token_rows)
        {
            if (!row.valid(padded_seq_len))
            {
                return reject(
                    "exact grouped-verifier preparation contains an invalid token-row binding");
            }
        }
        for (const auto &checkpoint : params.main_kv_checkpoints)
        {
            if (!checkpoint.valid())
            {
                return reject(
                    "exact grouped-verifier preparation contains an invalid KV checkpoint binding");
            }
        }

        return cache.segment_cache.deviceLoopGraphTemplate(
            *cache.graph,
            error);
    }

    std::optional<ForwardExecutionEngine::DeviceLoopGraphTemplateView>
    DeviceGraphOrchestrator::
        mtpAllPositionVerifierDeviceLoopGraphTemplate(
            int request_count,
            int padded_seq_len,
            DeviceGenerationSamplingMode sampling_mode,
            std::string *error) const
    {
        using TemplateView =
            ForwardExecutionEngine::DeviceLoopGraphTemplateView;
        auto reject = [&](const std::string &reason)
            -> std::optional<TemplateView>
        {
            if (error)
                *error = reason;
            return std::nullopt;
        };
        if (error)
            error->clear();
        if (!isValidDeviceGenerationSamplingMode(sampling_mode))
        {
            return reject(
                "all-position verifier loop composition received an invalid sampling topology");
        }

        const MTPVerifierPreparationGraphKey key{
            .control_policy = MTPVerifierPreparationControlPolicy::
                DeviceGenerationControlled,
            .request_count = request_count,
            .padded_seq_len = padded_seq_len,
        };
        std::string preparation_error;
        if (!mtpVerifierPreparationDeviceLoopGraphTemplate(
                request_count,
                padded_seq_len,
                &preparation_error))
        {
            return reject(
                "all-position verifier has no matching preparation registry capture: " +
                preparation_error);
        }
        if (!forward_engine_)
        {
            return reject(
                "all-position verifier loop composition has no forward engine");
        }

        std::string registry_error;
        const auto registry_index =
            mtpVerifierPreparationRegistryIndex(key, &registry_error);
        if (!registry_index)
            return reject(registry_error);
        auto &cache =
            *mtp_verifier_preparation_graphs_[*registry_index];
        if (!pairedVerifierForwardSignature(cache))
        {
            return reject(
                "exact grouped-verifier preparation has no paired forward signature");
        }

        std::string forward_error;
        auto forward = forward_engine_->deviceLoopGraphTemplate(
            *pairedVerifierForwardSignature(cache),
            &forward_error);
        if (!forward)
        {
            return reject(
                "all-position verifier capture is not exportable: " +
                forward_error);
        }

        const auto &signature = forward->signature;
        const MTPVerifierOutcomeGraphMode expected_outcome_mode =
            sampling_mode == DeviceGenerationSamplingMode::Greedy
                ? MTPVerifierOutcomeGraphMode::Greedy
                : MTPVerifierOutcomeGraphMode::Disabled;
        if (forward->device != state_.device_id ||
            signature.device != state_.device_id || !signature.decode ||
            !signature.all_position_logits ||
            signature.seq_len != padded_seq_len ||
            signature.batch_size != request_count ||
            signature.all_position_logit_rows !=
                request_count * padded_seq_len ||
            !signature.usesDeviceTokenIds() ||
            !signature.usesDevicePositionIds() ||
            !signature.usesDeviceSequenceLengths() ||
            signature.mtp_verifier_outcome_graph_mode !=
                expected_outcome_mode ||
            signature.is_bucketed_prefill)
        {
            return reject(
                std::string("retained all-position verifier signature does not consume the active device-owned ") +
                deviceGenerationSamplingModeName(sampling_mode) +
                " preparation/outcome topology");
        }
        return forward;
    }

    bool DeviceGraphOrchestrator::bindMTPVerifierForwardGraphPair(
        const MTPVerifierPreparationGraphKey &key,
        std::string *error)
    {
        auto fail = [&](const std::string &reason) -> bool
        {
            if (error)
                *error = reason;
            return false;
        };
        if (error)
            error->clear();
        if (!forward_engine_)
            return fail("grouped-verifier pairing has no forward engine");

        std::string registry_error;
        const auto registry_index =
            mtpVerifierPreparationRegistryIndex(key, &registry_error);
        if (!registry_index)
            return fail(registry_error);

        auto &cache = *mtp_verifier_preparation_graphs_[*registry_index];
        const uint64_t generation = workspaceGeneration(state_.device_id);
        if (!cache.valid || cache.key != key || !cache.graph ||
            !cache.stage || generation == 0 ||
            cache.workspace_generation != generation)
        {
            return fail(
                "grouped-verifier pairing registry owner is stale or incomplete");
        }

        const auto forward =
            forward_engine_->lastAllPositionVerifierForwardGraph();
        if (!forward || !*forward)
        {
            return fail(
                "grouped-verifier pairing has no just-completed verifier forward");
        }
        const auto &signature = forward->signature;
        if (forward->device != state_.device_id ||
            signature.device != state_.device_id || !signature.decode ||
            signature.sparse_query_execution_program != sparseQueryGraphProgram() ||
            !signature.all_position_logits ||
            signature.seq_len != key.padded_seq_len ||
            signature.batch_size != key.request_count ||
            signature.all_position_logit_rows !=
                key.request_count * key.padded_seq_len ||
            !signature.usesDeviceTokenIds() ||
            !signature.usesDevicePositionIds() ||
            signature.position_policy != ForwardPositionPolicy::ExplicitRows ||
            !signature.usesDeviceSequenceLengths() ||
            signature.is_bucketed_prefill)
        {
            const char *control_policy_name =
                key.control_policy ==
                        MTPVerifierPreparationControlPolicy::Standalone
                    ? "standalone"
                    : key.control_policy ==
                              MTPVerifierPreparationControlPolicy::
                                  DeviceGenerationControlled
                          ? "device_generation_controlled"
                          : "invalid";
            std::ostringstream diagnostic;
            diagnostic
                << "grouped-verifier forward signature does not match its "
                   "exact preparation registry key"
                << " expected={control_policy=" << control_policy_name
                << ",device=" << state_.device_id.toString()
                << ",request_count=" << key.request_count
                << ",physical_seq_len=" << key.padded_seq_len
                << ",all_position_logit_rows="
                << key.request_count * key.padded_seq_len
                << ",decode=true,all_position_logits=true"
                << ",uses_device_token_ids=true"
                << ",uses_device_position_ids=true"
                << ",position_policy="
                << static_cast<int>(ForwardPositionPolicy::ExplicitRows)
                << ",uses_device_sequence_lengths=true"
                << ",is_bucketed_prefill=false}"
                << " actual={view_device=" << forward->device.toString()
                << ",signature_device=" << signature.device.toString()
                << ",batch_size=" << signature.batch_size
                << ",seq_len=" << signature.seq_len
                << ",all_position_logit_rows="
                << signature.all_position_logit_rows
                << ",decode=" << signature.decode
                << ",all_position_logits="
                << signature.all_position_logits
                << ",uses_device_token_ids="
                << signature.usesDeviceTokenIds()
                << ",uses_device_position_ids="
                << signature.usesDevicePositionIds()
                << ",position_policy="
                << static_cast<int>(signature.position_policy)
                << ",uses_device_sequence_lengths="
                << signature.usesDeviceSequenceLengths()
                << ",is_bucketed_prefill="
                << signature.is_bucketed_prefill
                << ",outcome_graph_mode="
                << static_cast<int>(
                       signature.mtp_verifier_outcome_graph_mode)
                << '}';
            return fail(diagnostic.str());
        }

        if (key.control_policy == MTPVerifierPreparationControlPolicy::
                                      DeviceGenerationControlled &&
            !active_device_generation_admission_)
        {
            return fail(
                "grouped-verifier pairing has no admitted MTP depth policy");
        }
        const bool dynamic_depth =
            key.control_policy == MTPVerifierPreparationControlPolicy::
                                      DeviceGenerationControlled
                ? active_device_generation_admission_->depth_policy.mode ==
                      sampling_math::DeviceGenerationPolicyMode::Dynamic
                : graph_builder_ &&
                      graph_builder_->config().mtp.depth_policy.mode ==
                          MTPDepthPolicyMode::Dynamic;
        const DeviceGenerationLoopTopology topology =
            dynamic_depth
                ? DeviceGenerationLoopTopology::DynamicDepth
                : DeviceGenerationLoopTopology::FixedDepth;
        const DeviceGenerationExecutionPolicy execution_policy =
            deviceGenerationExecutionPolicy(topology);
        std::string replay_error;
        const bool replay_ready =
            execution_policy ==
                    DeviceGenerationExecutionPolicy::
                        HostScheduledCapturedTransactions
                ? static_cast<bool>(forward_engine_->retainedDecodeGraph(
                      signature, &replay_error))
                : static_cast<bool>(forward_engine_->deviceLoopGraphTemplate(
                      signature, &replay_error));
        if (!replay_ready)
        {
            return fail(
                "grouped-verifier forward is not replay-ready for exact pairing: " +
                replay_error);
        }

        std::array<std::optional<ForwardGraphSignature>, kSparseQueryExecutionProgramCount> paired;
        for (std::size_t program = 0; program < sparseQueryProgramCount(); ++program)
        {
            auto sibling = signature;
            sibling.sparse_query_execution_program = program == 0 ? SparseQueryExecutionProgram::RankingPossible :
                SparseQueryExecutionProgram::ProvenNoRanking;
            std::string sibling_error;
            const bool sibling_ready = execution_policy == DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions ?
                static_cast<bool>(forward_engine_->retainedDecodeGraph(sibling, &sibling_error)) :
                static_cast<bool>(forward_engine_->deviceLoopGraphTemplate(sibling, &sibling_error));
            if (!sibling_ready)
                return fail("Grouped verifier lacks its exact sealed query-program sibling: " + sibling_error);
            paired[program] = sibling;
        }
        bool changed = false;
        for (std::size_t program = 0; program < sparseQueryProgramCount(); ++program)
            changed = changed || (cache.paired_forward_signatures[program] &&
                cache.paired_forward_signatures[program] != paired[program]);
        if (changed) invalidateSparseQueryParents();
        cache.paired_forward_signatures = std::move(paired);
        return true;
    }

    std::optional<
        DeviceGraphExecutor::GraphSegmentCache::DeviceLoopGraphTemplateView>
    DeviceGraphOrchestrator::mtpAcceptedTerminalHiddenDeviceLoopGraphTemplate(
        int request_count,
        std::string *error) const
    {
        using TemplateView = DeviceGraphExecutor::GraphSegmentCache::
            DeviceLoopGraphTemplateView;
        auto reject = [&](const std::string &reason)
            -> std::optional<TemplateView>
        {
            if (error)
                *error = reason;
            return std::nullopt;
        };
        if (error)
            error->clear();
        if (!state_.device_id.is_gpu())
        {
            return reject(
                "accepted terminal-hidden loop composition requires a GPU runner");
        }
        if (request_count <= 0 ||
            request_count > stochastic_batch_output_request_capacity_ ||
            request_count > static_cast<int>(
                                mtp_terminal_hidden_device_accepted_rows_select_caches_
                                    .size()))
        {
            return reject(
                "accepted terminal-hidden request count is outside the pre-materialized graph family");
        }

        const auto &cache_owner =
            mtp_terminal_hidden_device_accepted_rows_select_caches_[
                static_cast<size_t>(request_count - 1)];
        if (!cache_owner)
            return reject("accepted terminal-hidden graph owner is missing");
        const MTPTerminalHiddenRowsSelectGraphCache &cache = *cache_owner;
        if (!cache.valid || !cache.graph || !cache.stage || !cache.logitsBindingsCurrent())
            return reject("accepted terminal-hidden/logit graph is not initialized with current allocations");
        if (cache.workspace_generation == 0 ||
            cache.workspace_generation != workspaceGeneration(state_.device_id))
        {
            return reject(
                "accepted terminal-hidden graph belongs to a stale workspace generation");
        }
        if (cache.selected_row_count != request_count ||
            cache.row_index_source !=
                HiddenStateRowsSelectStage::DeviceRowIndexSource::
                    ExternalDeviceIndices ||
            cache.row_buffer_name !=
                MTPSpecDecodeWorkspaceBuffers::ACCEPTED_STATE_SLOT_INDICES ||
            cache.external_device_row_indices !=
                mtp_spec_decode_metadata_binding_
                    .devicePointers()
                    .accepted_state_slot_indices)
        {
            return reject(
                "accepted terminal-hidden graph is not bound to compact device outcome indices");
        }

        return cache.segment_cache.deviceLoopGraphTemplate(
            *cache.graph,
            error);
    }

    std::optional<
        DeviceGraphExecutor::GraphSegmentCache::DeviceLoopGraphTemplateView>
    DeviceGraphOrchestrator::
        mtpSpeculativeStatePublicationDeviceLoopGraphTemplate(
            int request_count,
            int verifier_rows_per_request,
            std::string *error) const
    {
        using TemplateView = DeviceGraphExecutor::GraphSegmentCache::
            DeviceLoopGraphTemplateView;
        auto reject = [&](const std::string &reason)
            -> std::optional<TemplateView>
        {
            if (error)
                *error = reason;
            return std::nullopt;
        };
        if (error)
            error->clear();
        if (!state_.device_id.is_gpu())
        {
            return reject(
                "speculative state-publication loop composition requires a GPU runner");
        }

        if (request_count <= 0 || verifier_rows_per_request <= 0)
            return reject("speculative state-publication export requires positive request and physical-row geometry");
        // A const export names its physical family member directly. Cold setup
        // may leave another width selected for external execution; inspection
        // must neither depend on that incidental selection nor mutate it.
        const MTPSpeculativeStatePublicationGraphCache *member = nullptr;
        try
        {
            const auto program = sparseQueryExecutionProgramIndex(sparseQueryGraphProgram());
            const auto width = mtpPublicationWidthIndex(verifier_rows_per_request);
            member = &mtp_speculative_state_publication_graphs_.at(mtpPublicationCacheIndex(program, width));
        }
        catch (const std::exception &exception)
        {
            return reject(std::string("speculative state-publication export has no admitted physical-row member: ") + exception.what());
        }
        const auto &cache = *member;
        if (!cache.valid || !cache.graph || !cache.stage)
            return reject("speculative state-publication graph is not initialized");
        if (cache.workspace_generation == 0 ||
            cache.workspace_generation != workspaceGeneration(state_.device_id) ||
            cache.pipeline_edges != pipeline_forward_edges_)
        {
            return reject(
                "speculative state-publication graph belongs to a stale workspace generation");
        }
        const auto &params = cache.stage->getParams();
        if (!cache.verifier_signature ||
            cache.verifier_signature->sparse_query_execution_program != sparseQueryGraphProgram() ||
            cache.verifier_signature->device != state_.device_id ||
            !cache.verifier_signature->decode || !cache.verifier_signature->all_position_logits ||
            cache.verifier_signature->batch_size != request_count ||
            cache.verifier_signature->seq_len != verifier_rows_per_request ||
            params.request_count != request_count ||
            params.verifier_rows_per_request != verifier_rows_per_request)
        {
            return reject(
                "speculative state-publication graph geometry does not match the requested parent loop");
        }
        const bool follower = params.authority ==
            MTPSpeculativeStatePublicationStage::Authority::PipelineFollower;
        if (follower ? (!pipeline_forward_edges_ || !pp_stage_config_ || pp_stage_config_->has_lm_head)
                     : (params.authority != MTPSpeculativeStatePublicationStage::Authority::BoundedGeneration ||
                        !params.generation_response_tokens_device || !params.generation_control_device ||
                        !params.next_sidecar_condition_tokens_device || !params.next_sidecar_position_ids_device))
        {
            return reject(
                "speculative state-publication graph lacks its exact follower or terminal controller authority");
        }

        return cache.segment_cache.deviceLoopGraphTemplate(
            *cache.graph,
            error);
    }

    std::optional<
        DeviceGraphExecutor::GraphSegmentCache::DeviceLoopGraphTemplateView>
    DeviceGraphOrchestrator::
        mtpStochasticTargetDistributionDeviceLoopGraphTemplate(
            int row_count,
            std::string *error) const
    {
        using TemplateView = DeviceGraphExecutor::GraphSegmentCache::
            DeviceLoopGraphTemplateView;
        auto reject = [&](const std::string &reason)
            -> std::optional<TemplateView>
        {
            if (error)
                *error = reason;
            return std::nullopt;
        };
        if (error)
            error->clear();
        if (!state_.device_id.is_gpu())
        {
            return reject(
                "stochastic target-distribution loop composition requires a GPU runner");
        }

        const auto &cache = mtpStochasticTargetDistributionCache(row_count);
        if (!cache.valid || !cache.graph || !cache.stage)
        {
            return reject(
                "stochastic target-distribution graph is not initialized");
        }
        if (cache.workspace_generation == 0 ||
            cache.workspace_generation != workspaceGeneration(state_.device_id))
        {
            return reject(
                "stochastic target-distribution graph belongs to a stale workspace generation");
        }
        const auto &params = cache.stage->getParams();
        if (row_count <= 0 || params.row_count != row_count ||
            params.first_logit_row != 0 || params.first_target_slot != 0)
        {
            return reject(
                "stochastic target-distribution graph is not the canonical zero-based parent-loop geometry");
        }
        if (!params.logits_device || !params.target_token_ids_device ||
            !params.target_probs_device ||
            params.active_rows_device !=
                static_cast<const int32_t *>(
                    mtp_verifier_request_lengths_dev_) ||
            (params.apply_penalties &&
             (!params.verifier_input_tokens_device ||
              !params.generated_token_counts_device ||
              !params.penalty_policy_device)))
        {
            return reject(
                "stochastic target-distribution graph has incomplete persistent device bindings");
        }

        return cache.segment_cache.deviceLoopGraphTemplate(
            *cache.graph,
            error);
    }

    std::optional<
        DeviceGraphExecutor::GraphSegmentCache::DeviceLoopGraphTemplateView>
    DeviceGraphOrchestrator::
        mtpStochasticOutcomeDeviceLoopGraphTemplate(
            int request_count,
            int comparison_rows_per_request,
            std::string *error) const
    {
        using TemplateView = DeviceGraphExecutor::GraphSegmentCache::
            DeviceLoopGraphTemplateView;
        auto reject = [&](const std::string &reason)
            -> std::optional<TemplateView>
        {
            if (error)
                *error = reason;
            return std::nullopt;
        };
        if (error)
            error->clear();
        if (!state_.device_id.is_gpu())
        {
            return reject(
                "stochastic outcome loop composition requires a GPU runner");
        }

        const auto &cache = mtpStochasticOutcomeCache(comparison_rows_per_request + 1);
        if (!cache.valid || !cache.graph || !cache.stage)
            return reject("stochastic outcome graph is not initialized");
        if (cache.workspace_generation == 0 ||
            cache.workspace_generation != workspaceGeneration(state_.device_id))
        {
            return reject(
                "stochastic outcome graph belongs to a stale workspace generation");
        }
        const auto &params = cache.stage->getParams();
        if (request_count <= 0 || comparison_rows_per_request <= 0 ||
            params.request_count != request_count ||
            params.comparison_rows_per_request !=
                comparison_rows_per_request)
        {
            return reject(
                "stochastic outcome graph geometry does not match the requested parent loop");
        }
        if (!params.generation_control_device ||
            !params.threshold_base_positions_device ||
            !params.verifier_input_tokens_device)
        {
            return reject(
                "stochastic outcome graph lacks persistent controller or verifier bindings");
        }

        return cache.segment_cache.deviceLoopGraphTemplate(
            *cache.graph,
            error);
    }

    std::optional<
        DeviceGraphExecutor::GraphSegmentCache::DeviceLoopGraphTemplateView>
    DeviceGraphOrchestrator::
        moeOverlayEpochBoundaryDeviceLoopGraphTemplate(
            MoEOverlayEpochBoundaryStage::Operation operation,
            std::string *error) const
    {
        using TemplateView = DeviceGraphExecutor::GraphSegmentCache::
            DeviceLoopGraphTemplateView;
        auto reject = [&](const std::string &reason)
            -> std::optional<TemplateView>
        {
            if (error)
                *error = reason;
            return std::nullopt;
        };
        if (error)
            error->clear();
        if (!state_.device_id.is_gpu() ||
            !moe_overlay_epoch_execution_binding_)
        {
            return reject(
                "ExpertOverlay epoch boundary composition requires a bound GPU participant");
        }

        const auto &cache =
            operation == MoEOverlayEpochBoundaryStage::Operation::Acquire
                ? moe_overlay_epoch_acquire_graph_
                : moe_overlay_epoch_release_graph_;
        if (!cache.valid || !cache.graph || !cache.stage)
            return reject("ExpertOverlay epoch boundary graph is not initialized");
        if (cache.workspace_generation == 0 ||
            cache.workspace_generation != workspaceGeneration(state_.device_id))
        {
            return reject(
                "ExpertOverlay epoch boundary graph belongs to a stale workspace generation");
        }
        const auto &params = cache.stage->getParams();
        if (params.operation != operation ||
            params.arena != moe_overlay_epoch_execution_binding_.arena ||
            params.request_slot !=
                moe_overlay_epoch_execution_binding_.request_slot)
        {
            return reject(
                "ExpertOverlay epoch boundary graph has mismatched arena or request-slot identity");
        }

        return cache.segment_cache.deviceLoopGraphTemplate(
            *cache.graph,
            error);
    }

    bool DeviceGraphOrchestrator::materializeMTPDeviceGenerationLoopGraph(
        int request_count,
        int draft_depth,
        DeviceGenerationSamplingMode sampling_mode,
        std::string *error,
        MTPGenerationFamilyPreparation preparation)
    {
        using namespace sampling_math;

        auto &loop = deviceGenerationLoopGraph();
        auto fail = [&](const std::string &reason) -> bool
        {
            if (preparation == MTPGenerationFamilyPreparation::Materialize)
                loop.invalidateGraph();
            if (error)
                *error = reason;
            LOG_ERROR("[DeviceGraphOrchestrator] " << reason);
            return false;
        };
        if (error)
            error->clear();
        if (!isValidDeviceGenerationSamplingMode(sampling_mode))
        {
            return fail(
                "device-generation parent graph received an invalid sampling topology");
        }

        const uint64_t generation = workspaceGeneration(state_.device_id);
        if (!active_device_generation_admission_)
        {
            return fail(
                "device-generation parent graph has no admitted MTP depth policy");
        }
        const DeviceGenerationPolicy depth_policy =
            active_device_generation_admission_->depth_policy;
        const bool dynamic_depth =
            depth_policy.mode == DeviceGenerationPolicyMode::Dynamic;
        const int minimum_draft_depth =
            dynamic_depth ? depth_policy.minimum_depth : draft_depth;
        const int maximum_draft_depth =
            dynamic_depth ? depth_policy.maximum_depth : draft_depth;
        const int verifier_rows_per_request = draft_depth + 1;
        std::string width_policy_error;
        const auto width_policy =
            mtpVerifierPhysicalWidthPolicyForRequest(
                request_count,
                &width_policy_error);
        const int physical_verifier_rows_per_request =
            width_policy
                ? mtpVerifierPhysicalPaddedSeqLen(
                      request_count,
                      verifier_rows_per_request,
                      mtp_max_verifier_rows_,
                      *width_policy)
                : 0;
        const bool include_device_moe_maintenance =
            usesParticipantLocalDeviceMoERebalanceController();
        const bool include_overlay_epoch_boundaries =
            static_cast<bool>(moe_overlay_epoch_execution_binding_);
        const DeviceGenerationLoopTopology loop_topology =
            dynamic_depth ? DeviceGenerationLoopTopology::DynamicDepth
                          : DeviceGenerationLoopTopology::FixedDepth;
        const DeviceGenerationExecutionPolicy execution_policy =
            deviceGenerationExecutionPolicy(loop_topology);
        const bool hosted_execution =
            execution_policy ==
            DeviceGenerationExecutionPolicy::
                HostScheduledCapturedTransactions;
        const bool compact_native_dynamic =
            dynamic_depth && !hosted_execution;
        const auto desired_execution_kind =
            execution_policy ==
                    DeviceGenerationExecutionPolicy::NativeConditionalGraph
                ? MTPDeviceGenerationLoopGraphCache::ExecutionKind::
                      NativeConditionalParent
                : execution_policy ==
                          DeviceGenerationExecutionPolicy::
                              HostScheduledCapturedTransactions
                      ? MTPDeviceGenerationLoopGraphCache::ExecutionKind::
                            HostedDispatchTicketPublisher
                      : MTPDeviceGenerationLoopGraphCache::ExecutionKind::
                            Unmaterialized;
        const auto fragment_count_for_depth =
            [include_device_moe_maintenance,
             include_overlay_epoch_boundaries,
             hosted_execution,
             sampling_mode](int depth) -> size_t
        {
            const size_t common_fragment_count =
                static_cast<size_t>(2 * depth + 4);
            const size_t stochastic_fragment_count =
                sampling_mode == DeviceGenerationSamplingMode::Stochastic
                    ? 2u
                    : 0u;
            return common_fragment_count + stochastic_fragment_count +
                   (include_device_moe_maintenance ? 1u : 0u) +
                   (include_overlay_epoch_boundaries
                        ? 2u
                        : 0u);
        };
        size_t expected_fragment_count = compact_native_dynamic
            ? fragment_count_for_depth(maximum_draft_depth)
            : 0u;
        for (int depth = minimum_draft_depth;
             !compact_native_dynamic && depth <= maximum_draft_depth;
             ++depth)
        {
            expected_fragment_count += fragment_count_for_depth(depth);
        }
        if (!state_.device_id.is_gpu() || generation == 0 ||
            request_count != 1 || draft_depth <= 0 ||
            draft_depth > mtp_max_draft_depth_ ||
            physical_verifier_rows_per_request <= 0 ||
            !depth_policy.valid() || minimum_draft_depth <= 0 ||
            maximum_draft_depth < minimum_draft_depth ||
            maximum_draft_depth > mtp_max_draft_depth_ ||
            (dynamic_depth && draft_depth != maximum_draft_depth) ||
            (!dynamic_depth && draft_depth != depth_policy.initial_depth))
        {
            return fail(
                "device-generation parent graph requires one admitted GPU "
                "request and an exact supported depth-policy capture width: " +
                width_policy_error);
        }
        if (!loop.stream || !loop.capture ||
            loop.capture->executionStream() != loop.stream.get())
        {
            return fail(
                "device-generation parent graph has no exact preallocated stream/capture owner");
        }
        if (execution_policy ==
                DeviceGenerationExecutionPolicy::Unsupported ||
            desired_execution_kind ==
                MTPDeviceGenerationLoopGraphCache::ExecutionKind::
                    Unmaterialized)
        {
            return fail(
                "device-generation graph has no complete execution policy for its backend and depth topology");
        }
        // Condition and verifier children borrow this same whole-request
        // parent; their recording streams never become submission authority.
        if (!bindRetainedStorageSubmissionStream(loop.stream, error))
            return false;
        if (execution_policy ==
                DeviceGenerationExecutionPolicy::NativeConditionalGraph &&
            (!loop.capture->supportsDeviceControlledWhileLoop() ||
             (dynamic_depth &&
              !loop.capture->supportsDeviceControlledSelectorWhileLoop())))
        {
            return fail(
                dynamic_depth
                    ? "dynamic device-generation parent graph requires a native selector-gated WHILE implementation"
                    : "device-generation parent graph requires a native device-controlled WHILE implementation");
        }
        if (!device_generation_storage_.validFor(request_count) ||
            device_generation_storage_.active_request_count != request_count)
        {
            return fail(
                "device-generation parent graph has no exclusive admitted controller rows");
        }
        if (mtp_device_generation_loop_fragment_scratch_.capacity() <
                expected_fragment_count ||
            loop.source_fragments.capacity() < expected_fragment_count ||
            mtp_device_generation_loop_hosted_fragment_scratch_.capacity() <
                expected_fragment_count ||
            loop.hosted_fragments.capacity() < expected_fragment_count)
        {
            return fail(
                "device-generation parent graph branch/fragment storage was not preallocated for the complete configured policy");
        }

        mtp_device_generation_loop_fragment_scratch_.clear();
        mtp_device_generation_loop_hosted_fragment_scratch_.clear();
        std::array<size_t,
                   DeviceGenerationPolicy::kMaximumSupportedDraftDepth + 1>
            branch_offsets{};
        std::array<size_t,
                   DeviceGenerationPolicy::kMaximumSupportedDraftDepth + 1>
            branch_fragment_counts{};
        int assembling_depth = 0;
        const uint32_t *maintenance_due_device = nullptr;
        const auto active_fragment_count = [&]() -> size_t
        {
            return hosted_execution
                       ? mtp_device_generation_loop_hosted_fragment_scratch_
                             .size()
                       : mtp_device_generation_loop_fragment_scratch_.size();
        };
        auto append = [&](const char *fragment_name,
                          const auto &view,
                          const std::string &fragment_error,
                          DeviceControlledLoopFragmentExecution execution =
                              DeviceControlledLoopFragmentExecution::Always,
                          const uint32_t *condition_word_device = nullptr,
                          HostedDeviceGenerationFragment::Kind hosted_kind =
                              HostedDeviceGenerationFragment::Kind::
                                  CapturedLocal,
                          int minimum_selector = -1) -> bool
        {
            if (!view || !view->capture)
            {
                return fail(
                    std::string("device-generation parent graph rejected ") +
                    fragment_name + " for depth " +
                    std::to_string(assembling_depth) +
                    (fragment_error.empty()
                         ? ": no strict monolithic capture was exported"
                         : ": " + fragment_error));
            }
            if (hosted_execution)
            {
                if (hosted_kind !=
                        HostedDeviceGenerationFragment::Kind::CapturedLocal &&
                    hosted_kind != HostedDeviceGenerationFragment::Kind::
                                       MoEOverlayEpochAcquire &&
                    hosted_kind != HostedDeviceGenerationFragment::Kind::
                                       MoEOverlayEpochRelease)
                {
                    return fail(
                        std::string("device-generation hosted capture has an invalid typed kind for ") +
                        fragment_name);
                }
                mtp_device_generation_loop_hosted_fragment_scratch_.push_back(
                    HostedDeviceGenerationFragment{
                        .name = fragment_name,
                        .kind = hosted_kind,
                        .capture = view->capture,
                        .execution = execution,
                        .condition_word_device = condition_word_device,
                        .minimum_selector = minimum_selector,
                    });
            }
            else
            {
                mtp_device_generation_loop_fragment_scratch_.push_back(
                    DeviceControlledLoopFragment{
                        .name = fragment_name,
                        .capture = view->capture,
                        .execution = execution,
                        .condition_word_device = condition_word_device,
                        .minimum_selector = minimum_selector,
                    });
            }
            return true;
        };
        auto append_hosted_semantic =
            [&](const char *fragment_name,
                HostedDeviceGenerationFragment::Kind kind,
                const ComputeGraph *semantic_graph,
                const ForwardGraphSignature *forward_signature,
                const std::string &fragment_error) -> bool
        {
            if (!hosted_execution ||
                (kind != HostedDeviceGenerationFragment::Kind::
                             MTPGroupedVerifierReplay &&
                 !semantic_graph) ||
                (kind == HostedDeviceGenerationFragment::Kind::
                             MTPGroupedVerifierReplay &&
                 !forward_signature))
            {
                return fail(
                    std::string("device-generation hosted branch rejected ") +
                    fragment_name + " for depth " +
                    std::to_string(assembling_depth) +
                    (fragment_error.empty()
                         ? ": retained semantic identity is incomplete"
                         : ": " + fragment_error));
            }
            HostedDeviceGenerationFragment fragment;
            fragment.name = fragment_name;
            fragment.kind = kind;
            fragment.semantic_graph = semantic_graph;
            if (forward_signature)
                fragment.forward_signature = *forward_signature;
            mtp_device_generation_loop_hosted_fragment_scratch_.push_back(
                std::move(fragment));
            return true;
        };

        /*
         * Request orchestration executes the maintenance boundary for the
         * externally materialized first transaction before composing this
         * parent. That operation also leaves the exact maintenance child
         * replay-ready. Every branch therefore owns one complete transaction in
         * producer order: epoch acquire, sidecars, verifier, compact outcome,
         * state publication, terminal-hidden selection, epoch release, then
         * maintenance for the state it committed. Native and heterogeneous
         * hosted parents both own acquire as their first typed fragment. Hosted
         * sidecar streams borrow the resulting immutable event and return their
         * terminal to the scheduler stream before its final release fragment.
         * Acquire/release are absent together for an explicit non-overlay
         * topology.
         *
         * Native conditional execution retains one maximum physical envelope.
         * Hosted tickets instead select an immutable exact-row preparation,
         * model-forward and publication family. Mutable active rows remain owned
         * by the controller and can only shorten a selected branch for budget,
         * EOS or maintenance clipping; no host value rewrites its capture key.
         */
        if (hosted_execution && preparation == MTPGenerationFamilyPreparation::Materialize)
        {
            if (!mtp_publication_admission_authority_)
                return fail("Hosted exact-row family has no original first-transaction publication authority");
            if (!prepareHostedMTPDraftHelperFamily(sampling_mode, loop.stream.get(), error))
                return false;
            const int source_rows = mtp_publication_admission_authority_->seq_len;
            const auto widths = mtpVerifierGraphFamilyWidthCount(mtp_max_verifier_rows_, mtpVerifierGraphFamily());
            for (std::size_t index = 0; index < widths; ++index)
            {
                const int rows = mtpVerifierGraphFamilyWidthAt(index, mtp_max_verifier_rows_, mtpVerifierGraphFamily());
                if (!prepareHostedMTPVerifierBranch(rows, source_rows,
                        sampling_mode, loop.stream.get(), error))
                    return false;
                if (sampling_mode == DeviceGenerationSamplingMode::Stochastic &&
                    !prepareHostedMTPStochasticBranch(rows, source_rows, loop.stream.get(), error))
                    return false;
                const auto outcome = sampling_mode == DeviceGenerationSamplingMode::Greedy
                    ? MTPVerifierOutcomeGraphMode::Greedy : MTPVerifierOutcomeGraphMode::Disabled;
                if (!prepareMTPStatePublicationForVerifier(retainedMTPVerifierSignature(rows, outcome,
                        sparseQueryGraphProgram()), loop.stream.get(), error))
                    return false;
            }
        }
        auto assemble_transaction_branch = [&](int transaction_depth) -> bool
        {
            assembling_depth = transaction_depth;
            const int branch_physical_rows = hosted_execution ? transaction_depth + 1
                : physical_verifier_rows_per_request;
            const size_t branch_offset =
                active_fragment_count();
            std::string fragment_error;

            if (include_overlay_epoch_boundaries)
            {
                auto epoch_acquire =
                    moeOverlayEpochBoundaryDeviceLoopGraphTemplate(
                        MoEOverlayEpochBoundaryStage::Operation::Acquire,
                        &fragment_error);
                if (!append(
                        "ExpertOverlay epoch acquire",
                        epoch_acquire,
                        fragment_error,
                        DeviceControlledLoopFragmentExecution::Always,
                        nullptr,
                        hosted_execution
                            ? HostedDeviceGenerationFragment::Kind::
                                  MoEOverlayEpochAcquire
                            : HostedDeviceGenerationFragment::Kind::
                                  CapturedLocal))
                {
                    return false;
                }
            }

            if (hosted_execution)
            {
                const ComputeGraph *sidecar_graph = nullptr;
                if (!validateHostedMTPSidecarReplay(
                        MTPSidecarCaptureRole::Full,
                        request_count,
                        &sidecar_graph,
                        &fragment_error) ||
                    !append_hosted_semantic(
                        "full sidecar",
                        HostedDeviceGenerationFragment::Kind::
                            MTPFullSidecarReplay,
                        sidecar_graph,
                        nullptr,
                        fragment_error))
                {
                    return false;
                }
            }
            else
            {
                auto full_sidecar = mtpSidecarDeviceLoopGraphTemplate(
                    MTPSidecarCaptureRole::Full,
                    request_count,
                    &fragment_error);
                if (!append("full sidecar", full_sidecar, fragment_error))
                    return false;
            }

            for (int slot = 0; slot < transaction_depth; ++slot)
            {
                const int slot_depth = slot + 1;
                const bool selector_gated =
                    compact_native_dynamic &&
                    slot_depth > minimum_draft_depth;
                const auto slot_execution = selector_gated
                    ? DeviceControlledLoopFragmentExecution::
                          IfDeviceSelectorAtLeast
                    : DeviceControlledLoopFragmentExecution::Always;
                const int slot_minimum_selector =
                    selector_gated ? slot_depth : -1;
                if (slot > 0)
                {
                    fragment_error.clear();
                    if (hosted_execution)
                    {
                        const ComputeGraph *sidecar_graph = nullptr;
                        if (!validateHostedMTPSidecarReplay(
                                MTPSidecarCaptureRole::Chained,
                                request_count,
                                &sidecar_graph,
                                &fragment_error) ||
                            !append_hosted_semantic(
                                "chained sidecar",
                                HostedDeviceGenerationFragment::Kind::
                                    MTPChainedSidecarReplay,
                                sidecar_graph,
                                nullptr,
                                fragment_error))
                        {
                            return false;
                        }
                    }
                    else
                    {
                        auto chained_sidecar =
                            mtpSidecarDeviceLoopGraphTemplate(
                                MTPSidecarCaptureRole::Chained,
                                request_count,
                                &fragment_error);
                        if (!append(
                                "chained sidecar",
                                chained_sidecar,
                                fragment_error,
                                slot_execution,
                                nullptr,
                                HostedDeviceGenerationFragment::Kind::
                                    CapturedLocal,
                                slot_minimum_selector))
                        {
                            return false;
                        }
                    }
                }

                fragment_error.clear();
                auto draft_publication =
                    mtpDraftTokenPublicationDeviceLoopGraphTemplate(
                        slot,
                        &fragment_error);
                if (!append(
                        "draft-token publication",
                        draft_publication,
                        fragment_error,
                        slot_execution,
                        nullptr,
                        HostedDeviceGenerationFragment::Kind::CapturedLocal,
                        slot_minimum_selector))
                {
                    return false;
                }
            }

            fragment_error.clear();
            auto verifier_preparation =
                mtpVerifierPreparationDeviceLoopGraphTemplate(
                    request_count,
                    branch_physical_rows,
                    &fragment_error);
            if (!append(
                    "verifier preparation",
                    verifier_preparation,
                    fragment_error))
            {
                return false;
            }

            fragment_error.clear();
            if (hosted_execution)
            {
                const MTPVerifierPreparationGraphKey key{
                    .control_policy =
                        MTPVerifierPreparationControlPolicy::
                            DeviceGenerationControlled,
                    .request_count = request_count,
                    .padded_seq_len =
                        branch_physical_rows,
                };
                const auto registry_index =
                    mtpVerifierPreparationRegistryIndex(
                        key, &fragment_error);
                if (!registry_index)
                    return false;
                auto &preparation_cache =
                    *mtp_verifier_preparation_graphs_[*registry_index];
                if (!pairedVerifierForwardSignature(preparation_cache) ||
                    !forward_engine_)
                {
                    return fail(
                        "hosted grouped verifier has no exact paired forward signature");
                }
                const ForwardGraphSignature &signature =
                    *pairedVerifierForwardSignature(preparation_cache);
                const MTPVerifierOutcomeGraphMode expected_outcome_mode =
                    sampling_mode == DeviceGenerationSamplingMode::Greedy
                        ? MTPVerifierOutcomeGraphMode::Greedy
                        : MTPVerifierOutcomeGraphMode::Disabled;
                const auto retained = forward_engine_->retainedDecodeGraph(
                    signature, &fragment_error);
                if (!retained || retained->device != state_.device_id ||
                    signature.seq_len !=
                        branch_physical_rows ||
                    signature.batch_size != request_count ||
                    !signature.all_position_logits ||
                    signature.all_position_logit_rows !=
                        request_count *
                            branch_physical_rows ||
                    !signature.usesDeviceSequenceLengths() ||
                    signature.mtp_verifier_outcome_graph_mode !=
                        expected_outcome_mode ||
                    !append_hosted_semantic(
                        "all-position verifier forward",
                        HostedDeviceGenerationFragment::Kind::
                            MTPGroupedVerifierReplay,
                        nullptr,
                        &signature,
                        fragment_error))
                {
                    return false;
                }
            }
            else
            {
                auto verifier_forward =
                    mtpAllPositionVerifierDeviceLoopGraphTemplate(
                        request_count,
                        branch_physical_rows,
                        sampling_mode,
                        &fragment_error);
                if (!append(
                        "all-position verifier forward",
                        verifier_forward,
                        fragment_error))
                {
                    return false;
                }
            }

            if (sampling_mode == DeviceGenerationSamplingMode::Stochastic)
            {
                fragment_error.clear();
                auto target_distribution =
                    mtpStochasticTargetDistributionDeviceLoopGraphTemplate(
                        branch_physical_rows,
                        &fragment_error);
                if (!append(
                        "stochastic target distribution",
                        target_distribution,
                        fragment_error))
                {
                    return false;
                }

                fragment_error.clear();
                auto stochastic_outcome =
                    mtpStochasticOutcomeDeviceLoopGraphTemplate(
                        request_count,
                        hosted_execution ? branch_physical_rows - 1 : draft_depth,
                        &fragment_error);
                if (!append(
                        "stochastic outcome",
                        stochastic_outcome,
                        fragment_error))
                {
                    return false;
                }
            }

            fragment_error.clear();
            const MTPVerifierPreparationGraphKey publication_key{
                .control_policy = MTPVerifierPreparationControlPolicy::DeviceGenerationControlled,
                .request_count = request_count, .padded_seq_len = branch_physical_rows};
            const auto publication_index = mtpVerifierPreparationRegistryIndex(publication_key, &fragment_error);
            if (!publication_index || !mtp_verifier_preparation_graphs_[*publication_index] ||
                !pairedVerifierForwardSignature(*mtp_verifier_preparation_graphs_[*publication_index]) ||
                (preparation == MTPGenerationFamilyPreparation::Materialize &&
                 !prepareMTPStatePublicationForVerifier(
                    *pairedVerifierForwardSignature(*mtp_verifier_preparation_graphs_[*publication_index]),
                    loop.stream.get(), &fragment_error)))
                return fail("Exact verifier publication child is not prepared: " + fragment_error);
            auto state_publication =
                mtpSpeculativeStatePublicationDeviceLoopGraphTemplate(
                    request_count,
                    branch_physical_rows,
                    &fragment_error);
            if (!append(
                    "speculative state publication",
                    state_publication,
                    fragment_error))
            {
                return false;
            }

            fragment_error.clear();
            auto terminal_hidden =
                mtpAcceptedTerminalHiddenDeviceLoopGraphTemplate(
                    request_count,
                    &fragment_error);
            if (!append(
                    "accepted terminal-hidden publication",
                    terminal_hidden,
                    fragment_error))
            {
                return false;
            }

            if (include_overlay_epoch_boundaries)
            {
                fragment_error.clear();
                auto epoch_release =
                    moeOverlayEpochBoundaryDeviceLoopGraphTemplate(
                        MoEOverlayEpochBoundaryStage::Operation::Release,
                        &fragment_error);
                if (!append(
                        "ExpertOverlay epoch release",
                        epoch_release,
                        fragment_error,
                        DeviceControlledLoopFragmentExecution::Always,
                        nullptr,
                        hosted_execution
                            ? HostedDeviceGenerationFragment::Kind::
                                  MoEOverlayEpochRelease
                            : HostedDeviceGenerationFragment::Kind::
                                  CapturedLocal))
                {
                    return false;
                }
            }

            /*
             * Tail placement is exact: the unconditional epoch release must
             * precede even a skipped maintenance fragment, while cloning
             * a native maintenance head would consume the first transaction twice.
             * The serial-outcome fragment publishes `maintenance_due`; the typed
             * conditional skips the whole maintenance body, including its
             * collective, on ordinary iterations.
             * Hosted tickets are observed between outcome/release and this
             * tail. Their continuation selector retires the completed tail
             * before starting the next body; it does not reuse native body
             * ordering with a predicate sampled one transaction earlier.
             */
            if (include_device_moe_maintenance)
            {
                const auto &maintenance =
                    device_moe_rebalance_maintenance_graph_;
                const DeviceMoERebalanceGraphControllerState *const controller =
                    deviceMoERebalanceControllerStateDevice();
                if (!maintenance.graph ||
                    maintenance.workspace_generation == 0 ||
                    maintenance.workspace_generation != generation ||
                    !controller)
                {
                    return fail(
                        "device-generation parent graph has no current device MoE maintenance owner");
                }
                maintenance_due_device =
                    reinterpret_cast<const uint32_t *>(
                        reinterpret_cast<const std::byte *>(controller) +
                        offsetof(
                            DeviceMoERebalanceGraphControllerState,
                            maintenance_due));
                auto maintenance_fragment =
                    maintenance.segment_cache.deviceLoopGraphTemplate(
                        *maintenance.graph,
                        &fragment_error);
                if (!append(
                        "device MoE maintenance",
                        maintenance_fragment,
                        fragment_error,
                        DeviceControlledLoopFragmentExecution::
                            IfDeviceWordNonZero,
                        maintenance_due_device))
                {
                    return false;
                }
            }

            const size_t assembled_count =
                active_fragment_count() - branch_offset;
            if (assembled_count !=
                fragment_count_for_depth(transaction_depth))
            {
                return fail(
                    "device-generation parent graph assembled an invalid per-depth transaction body");
            }
            branch_offsets[static_cast<size_t>(transaction_depth)] =
                branch_offset;
            branch_fragment_counts[static_cast<size_t>(transaction_depth)] =
                assembled_count;
            return true;
        };

        if (compact_native_dynamic)
        {
            if (!assemble_transaction_branch(maximum_draft_depth))
                return false;
        }
        else
        {
            for (int depth = minimum_draft_depth;
                 depth <= maximum_draft_depth;
                 ++depth)
            {
                if (!assemble_transaction_branch(depth))
                    return false;
            }
        }

        if (active_fragment_count() != expected_fragment_count)
        {
            return fail(
                "device-generation parent graph assembled an unexpected fragment count");
        }

        /*
         * Keep semantic graph attribution beside the declarative branch table.
         * Walking source captures here gives each kernel its producer role and
         * depth before native WHILE/SWITCH composition erases that distinction.
         * It also runs only for a new executable: the reuse return above occurs
         * after source identity is checked, while publication is intentionally
         * placed below that check and immediately before graph construction.
         */
        const size_t ticket_conditioned_fragment_count = hosted_execution
            ? static_cast<size_t>(std::count_if(
                  mtp_device_generation_loop_hosted_fragment_scratch_.begin(),
                  mtp_device_generation_loop_hosted_fragment_scratch_.end(),
                  [](const HostedDeviceGenerationFragment &fragment)
                  {
                      return fragment.execution !=
                             DeviceControlledLoopFragmentExecution::Always;
                  }))
            : 0u;
        const size_t native_conditional_fragment_count = hosted_execution
            ? 0u
            : static_cast<size_t>(std::count_if(
                  mtp_device_generation_loop_fragment_scratch_.begin(),
                  mtp_device_generation_loop_fragment_scratch_.end(),
                  [](const DeviceControlledLoopFragment &fragment)
                  {
                      return fragment.execution !=
                             DeviceControlledLoopFragmentExecution::Always;
                  }));
        const size_t native_selector_gated_fragment_count = hosted_execution
            ? 0u
            : static_cast<size_t>(std::count_if(
                  mtp_device_generation_loop_fragment_scratch_.begin(),
                  mtp_device_generation_loop_fragment_scratch_.end(),
                  [](const DeviceControlledLoopFragment &fragment)
                  {
                      return fragment.execution ==
                             DeviceControlledLoopFragmentExecution::
                                 IfDeviceSelectorAtLeast;
                  }));

        const bool source_identity_matches = hosted_execution
            ? (loop.hosted_fragments.size() == expected_fragment_count &&
               std::equal(
                   loop.hosted_fragments.begin(),
                   loop.hosted_fragments.end(),
                   mtp_device_generation_loop_hosted_fragment_scratch_.begin(),
                   [](const HostedDeviceGenerationFragment &source,
                      const HostedDeviceGenerationFragment &candidate)
                   {
                       return source.hasSameExecutionIdentity(candidate);
                   }))
            : (loop.source_fragments.size() == expected_fragment_count &&
               std::equal(
                   loop.source_fragments.begin(),
                   loop.source_fragments.end(),
                   mtp_device_generation_loop_fragment_scratch_.begin(),
                   [](const DeviceControlledLoopFragment &source,
                      const DeviceControlledLoopFragment &candidate)
                   {
                       return source.hasSameExecutionIdentity(candidate);
                   }));
        if (loop.valid && loop.capture->hasExecutable() &&
            loop.workspace_generation == generation &&
            loop.request_count == request_count &&
            loop.draft_depth == draft_depth &&
            loop.verifier_rows_per_request == verifier_rows_per_request &&
            loop.minimum_draft_depth == minimum_draft_depth &&
            loop.maximum_draft_depth == maximum_draft_depth &&
            loop.depth_policy_mode == static_cast<int>(depth_policy.mode) &&
            loop.sampling_mode == sampling_mode &&
            loop.fragment_count == expected_fragment_count &&
            loop.native_conditional_fragment_count ==
                native_conditional_fragment_count &&
            loop.native_selector_gated_fragment_count ==
                native_selector_gated_fragment_count &&
            loop.ticket_conditioned_fragment_count ==
                ticket_conditioned_fragment_count &&
            loop.execution_kind == desired_execution_kind &&
            source_identity_matches)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "device_generation_loop_graph_reuses",
                1.0,
                "graph_setup",
                state_.device_id.toString(),
                {{"backend", loop.capture->backendName()},
                 {"requests", std::to_string(request_count)},
                 {"draft_depth", std::to_string(draft_depth)},
                 {"minimum_draft_depth",
                  std::to_string(minimum_draft_depth)},
                 {"maximum_draft_depth",
                  std::to_string(maximum_draft_depth)},
                 {"depth_policy",
                  dynamic_depth ? "dynamic" : "fixed_width"},
                 {"sampling_mode",
                  deviceGenerationSamplingModeName(sampling_mode)},
                 {"execution_policy",
                  deviceGenerationExecutionPolicyName(execution_policy)},
                 {"fragments", std::to_string(expected_fragment_count)},
                 {"conditional_fragments",
                  std::to_string(native_conditional_fragment_count)},
                 {"selector_gated_fragments",
                  std::to_string(native_selector_gated_fragment_count)},
                 {"ticket_conditioned_fragments",
                  std::to_string(ticket_conditioned_fragment_count)},
                 {"workspace_generation", std::to_string(generation)}});
            return true;
        }

        if (preparation == MTPGenerationFamilyPreparation::RequireRetained)
            return fail("ContinueResponse requires the unchanged complete retained hosted MTP family");

        loop.invalidateGraph();
        if (debugEnv().runtime_debug.gpu_graph_kernel_inventory)
        {
            for (int depth = minimum_draft_depth;
                 depth <= maximum_draft_depth;
                 ++depth)
            {
                const size_t branch_index = static_cast<size_t>(depth);
                const size_t branch_offset = branch_offsets[branch_index];
                const size_t branch_count =
                    branch_fragment_counts[branch_index];
                for (size_t fragment_index = 0;
                     fragment_index < branch_count;
                     ++fragment_index)
                {
                    std::string inventory_error;
                    if (hosted_execution)
                    {
                        const HostedDeviceGenerationFragment &fragment =
                            mtp_device_generation_loop_hosted_fragment_scratch_[
                                branch_offset + fragment_index];
                        if (fragment.capture &&
                            !DeviceGraphCaptureController::publishKernelInventory(
                                *fragment.capture,
                                fragment.name ? fragment.name : "unnamed",
                                depth,
                                state_.device_id,
                                inventory_error))
                        {
                            return fail(inventory_error);
                        }
                        if (!fragment.capture)
                        {
                            PerfStatsCollector::addCounter(
                                "mtp",
                                "device_generation_segmented_semantic_inventory",
                                1.0,
                                "graph_setup",
                                state_.device_id.toString(),
                                {{"fragment",
                                  fragment.name ? fragment.name : "unnamed"},
                                 {"depth", std::to_string(depth)}});
                        }
                    }
                    else
                    {
                        const DeviceControlledLoopFragment &fragment =
                            mtp_device_generation_loop_fragment_scratch_[
                                branch_offset + fragment_index];
                        if (!DeviceGraphCaptureController::publishKernelInventory(
                                *fragment.capture,
                                fragment.name ? fragment.name : "unnamed",
                                depth,
                                state_.device_id,
                                inventory_error))
                        {
                            return fail(inventory_error);
                        }
                    }
                }
            }
        }
        const DeviceGenerationGraphControl graph_control{
            .policy = depth_policy,
            .rows = device_generation_storage_.control_device,
            .stride = device_generation_storage_.control_stride,
            .requests = request_count,
        };
        std::string construction_error;
        bool built = false;
        if (execution_policy ==
            DeviceGenerationExecutionPolicy::NativeConditionalGraph)
        {
            built = DeviceGenerationGraphProgram::native(
                *loop.capture, graph_control,
                DeviceControlledLoopProgram{
                    .iteration = mtp_device_generation_loop_fragment_scratch_,
                    .terminal=retainedStorageTerminalFragments()},
                construction_error);
        }
        else
        {
            /*
             * A hosted graph owner captures only immutable ticket publication.
             * HIP uses this because it lacks conditional graph nodes;
             * heterogeneous CUDA uses it because a native parent cannot clone
             * a remote ROCm sparse boundary. The retained semantic branch table
             * still names the exact production sidecar/verifier replay plans.
             * The ticket capture accepts no response, cache, sampler, or
             * compact-outcome pointer.
             */
            IBackend *const backend = getBackendFor(state_.device_id);
            if (!backend)
            {
                return fail(
                    "host-scheduled device generation has no owning backend");
            }
            auto &context = GPUDeviceContextPool::instance().getContext(state_.device_id);
            built = DeviceGenerationGraphProgram::ticketPublisher(
                *loop.capture, context, *backend, state_.device_id, graph_control,
                maintenance_due_device, device_generation_storage_.dispatch_tickets_device,
                construction_error,retained_storage_boundary_.get());
        }
        if (!built)
        {
            return fail(construction_error);
        }

        loop.source_fragments.clear();
        loop.hosted_fragments.clear();
        if (hosted_execution)
        {
            for (const HostedDeviceGenerationFragment &fragment :
                 mtp_device_generation_loop_hosted_fragment_scratch_)
            {
                loop.hosted_fragments.push_back(fragment);
            }
        }
        else
        {
            for (const DeviceControlledLoopFragment &fragment :
                 mtp_device_generation_loop_fragment_scratch_)
            {
                loop.source_fragments.push_back(fragment);
            }
        }
        loop.workspace_generation = generation;
        loop.request_count = request_count;
        loop.draft_depth = draft_depth;
        loop.verifier_rows_per_request = verifier_rows_per_request;
        loop.minimum_draft_depth = minimum_draft_depth;
        loop.maximum_draft_depth = maximum_draft_depth;
        loop.depth_policy_mode = static_cast<int>(depth_policy.mode);
        loop.sampling_mode = sampling_mode;
        loop.fragment_count = expected_fragment_count;
        loop.native_conditional_fragment_count =
            native_conditional_fragment_count;
        loop.native_selector_gated_fragment_count =
            native_selector_gated_fragment_count;
        loop.ticket_conditioned_fragment_count =
            ticket_conditioned_fragment_count;
        loop.branch_offsets = branch_offsets;
        loop.branch_fragment_counts = branch_fragment_counts;
        loop.execution_kind = desired_execution_kind;
        loop.valid = true;
        loop.launched = false;

        PerfStatsCollector::addCounter(
            "mtp",
            "device_generation_loop_graph_materializations",
            1.0,
            "graph_setup",
            state_.device_id.toString(),
            {{"backend", loop.capture->backendName()},
             {"requests", std::to_string(request_count)},
             {"draft_depth", std::to_string(draft_depth)},
             {"minimum_draft_depth",
              std::to_string(minimum_draft_depth)},
             {"maximum_draft_depth",
              std::to_string(maximum_draft_depth)},
             {"depth_policy",
              dynamic_depth ? "dynamic" : "fixed_width"},
             {"sampling_mode",
              deviceGenerationSamplingModeName(sampling_mode)},
             {"terminal_head_policy", mtpTerminalHeadPolicyToString(
                  graph_builder_->config().mtp.terminal_head_policy)},
             {"native_local_vocabulary",
              graph_builder_->config().mtpPublishesNativeGatheredVocabulary() ? "true" : "false"},
             {"verifier_rows",
              std::to_string(verifier_rows_per_request)},
             {"physical_verifier_rows",
              std::to_string(physical_verifier_rows_per_request)},
             {"verifier_graph_family", hosted_execution ? "exact_rows" : "bounded_buckets"},
             {"retained_verifier_widths", std::to_string(mtpVerifierGraphFamilyWidthCount(
                  mtp_max_verifier_rows_, mtpVerifierGraphFamily()))},
             {"publication_graph_owners", std::to_string(mtp_graph_owner_plan_.speculativePublicationGraphSlots())},
             {"generation_parent_graph_owners", std::to_string(mtp_graph_owner_plan_.generationParentGraphSlots())},
             {"sampler_helper_graph_owners", std::to_string(mtp_graph_owner_plan_.samplerHelperGraphSlots())},
             {"fragments", std::to_string(expected_fragment_count)},
             {"conditional_fragments",
              std::to_string(native_conditional_fragment_count)},
             {"selector_gated_fragments",
              std::to_string(native_selector_gated_fragment_count)},
             {"ticket_conditioned_fragments",
              std::to_string(ticket_conditioned_fragment_count)},
             {"device_moe_maintenance",
              include_device_moe_maintenance ? "true" : "false"},
             {"workspace_generation", std::to_string(generation)},
             {"execution",
              execution_policy ==
                      DeviceGenerationExecutionPolicy::
                          NativeConditionalGraph
                  ? (dynamic_depth
                         ? "native_device_controlled_selector_while"
                         : "native_device_controlled_while")
                  : "hosted_captured_transactions_with_ticket_only_dispatch"}});
        return true;
    }

    bool DeviceGraphOrchestrator::executeMTPDepth0(
        int32_t draft_condition_token,
        TensorBase *terminal_hidden,
        int position_id,
        const char *sidecar_perf_context,
        bool kv_cache_only,
        BufferId terminal_hidden_buffer_id,
        bool defer_final_sync)
    {
        return executeMTPDepth0Batched(
            &draft_condition_token,
            1,
            terminal_hidden,
            position_id,
            sidecar_perf_context,
            kv_cache_only,
            terminal_hidden_buffer_id,
            defer_final_sync);
    }

    bool DeviceGraphOrchestrator::executeMTPDepth0Batched(
        const int32_t *draft_condition_tokens,
        int token_count,
        TensorBase *terminal_hidden,
        int position_id,
        const char *sidecar_perf_context,
        bool kv_cache_only,
        BufferId terminal_hidden_buffer_id,
        bool defer_final_sync,
        const void *draft_condition_tokens_device,
        int draft_condition_ready_slot,
        bool draft_condition_ready_is_target,
        int request_batch,
        const int *position_ids_override,
        const void *position_ids_device_override,
        const void *speculative_outcome_meta_device,
        int speculative_outcome_meta_stride,
        const void *speculative_outcome_output_tokens_device,
        int speculative_outcome_output_token_stride,
        int speculative_outcome_request_index,
        int speculative_first_output_token_index,
        void *speculative_outcome_ready_event,
        int draft_condition_token_stride,
        int draft_condition_ready_count,
        int draft_condition_ready_stride,
        int device_position_offset,
        int first_seq_idx,
        DeviceGraphExecutor::GraphInitialSubmissionPolicy submission)
    {
        if (state_.device_id.is_gpu() && !mtp_graph_owner_plan_.ownsControllerExecutables())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Pipeline follower cannot execute or materialize predictor sidecar graphs");
            return false;
        }

        const bool materialize_only = submission ==
            DeviceGraphExecutor::GraphInitialSubmissionPolicy::MaterializeWithoutLaunch;
        if (submission != DeviceGraphExecutor::GraphInitialSubmissionPolicy::CaptureInstantiateAndLaunch &&
            !materialize_only)
            throw std::invalid_argument("MTP sidecar has an unknown initial submission policy");
        if (materialize_only)
        {
            // Cold full/chained graphs borrow only their original retained
            // hidden roots. No live tokens or model arithmetic are submitted.
            const auto chained=state_.extension_buffers.find(BufferId::MTP_HIDDEN);
            const bool hidden_binding=
                (terminal_hidden_buffer_id==BufferId::PREFIX_TERMINAL_HIDDEN && terminal_hidden==state_.prefix_terminal_hidden.get()) ||
                (terminal_hidden_buffer_id==BufferId::MTP_HIDDEN && chained!=state_.extension_buffers.end() &&
                 terminal_hidden==chained->second.get());
            if (!state_.device_id.is_gpu() || kv_cache_only || !terminal_hidden || !hidden_binding ||
                token_count != 1 || request_batch != 1 || !draft_condition_tokens_device ||
                !position_ids_device_override || position_ids_override || speculative_outcome_meta_device ||
                speculative_outcome_output_tokens_device || speculative_outcome_ready_event || first_seq_idx != 0)
                throw std::invalid_argument("Cold sidecar capture requires its exact original scalar full/chained role");
        }
        const int seq_len = token_count;
        const int total_rows =
            (token_count > 0 && request_batch > 0)
                ? token_count * request_batch
                : 0;
        const std::string device_key = state_.device_id.toString();
        const std::string phase = materialize_only ? "graph_setup" : perfPhaseName();
        const std::string sidecar_context =
            (sidecar_perf_context && sidecar_perf_context[0] != '\0')
                ? sidecar_perf_context
                : ((phase == "prefill") ? "mtp_shifted_prefill" : "mtp_decode_sidecar");
        const bool external_device_condition_tokens = draft_condition_tokens_device != nullptr;
        const bool prepare_device_condition_tokens_from_speculative_outcome =
            speculative_outcome_meta_device != nullptr ||
            speculative_outcome_output_tokens_device != nullptr ||
            speculative_outcome_ready_event != nullptr;
        const bool compose_batched_device_inputs =
            (external_device_condition_tokens &&
             (draft_condition_token_stride != 1 || device_position_offset != 0)) ||
            (prepare_device_condition_tokens_from_speculative_outcome &&
             position_ids_device_override != nullptr);
        /*
         * A captured sidecar must never bind the caller's sliding source
         * pointer. Shifted prefill advances `position_ids_device_override`
         * through the prefill input one group at a time, so using that pointer
         * as graph identity rebuilt the same M-row graph for every group.
         *
         * Stage all device-owned position rows into the persistent sidecar
         * arena instead. The source may move; the captured consumer address
         * does not. Token and position rows are prepared by one backend launch,
         * preserving device ownership without adding a host observation.
         */
        const bool stage_device_position_rows =
            position_ids_device_override != nullptr &&
            (external_device_condition_tokens ||
             prepare_device_condition_tokens_from_speculative_outcome);
        const void *effective_position_ids_device =
            stage_device_position_rows
                ? mtp_sidecar_position_ids_dev_
                : position_ids_device_override;
        const bool use_device_position_ids =
            effective_position_ids_device != nullptr;
        const bool use_device_condition_tokens =
            external_device_condition_tokens ||
            prepare_device_condition_tokens_from_speculative_outcome;
        if (state_.device_id.is_gpu() && !defer_final_sync)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] GPU MTP sidecars require an event-published completion handoff; "
                      "host stream synchronization is not a supported execution mode");
            return false;
        }
        if ((!draft_condition_tokens &&
             !external_device_condition_tokens &&
            !prepare_device_condition_tokens_from_speculative_outcome) ||
            token_count <= 0 || request_batch <= 0 || total_rows <= 0 ||
            first_seq_idx < 0 ||
            total_rows > mtp_sidecar_condition_token_slot_width_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP sidecar received invalid shape: seq_len="
                      << token_count << " batch=" << request_batch);
            return false;
        }
        if (state_.device_id.is_gpu() && !use_device_condition_tokens)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] GPU MTP sidecars require a device-owned condition-token source");
            return false;
        }
        if (state_.device_id.is_gpu() &&
            stage_device_position_rows &&
            !mtp_sidecar_position_ids_dev_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] GPU MTP sidecars require persistent device position storage");
            return false;
        }
        if (external_device_condition_tokens &&
            (draft_condition_token_stride <= 0 ||
             draft_condition_ready_count <= 0 ||
             draft_condition_ready_stride <= 0))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-token MTP sidecar received invalid token-slot geometry");
            return false;
        }
        if (compose_batched_device_inputs &&
            (!position_ids_device_override || !mtp_sidecar_position_ids_dev_))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Batched device-token MTP sidecar input composition requires resident base positions and arena position storage");
            return false;
        }
        if (request_batch > 1 && token_count != 1)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Request-batched MTP sidecar supports exactly one token per request");
            return false;
        }
        if (request_batch > 1 &&
            !position_ids_override &&
            !use_device_position_ids)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Request-batched MTP sidecar requires per-request position ids");
            return false;
        }
        if (position_ids_override && use_device_position_ids)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP sidecar received both host and device position overrides");
            return false;
        }
        if (use_device_position_ids && !state_.device_id.is_gpu())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-position MTP sidecar requires a GPU device");
            return false;
        }
        if (prepare_device_condition_tokens_from_speculative_outcome &&
            (!state_.device_id.is_gpu() ||
             !speculative_outcome_meta_device ||
             speculative_outcome_meta_stride < sampling_math::kSpeculativeBatchMetaCount ||
             !speculative_outcome_output_tokens_device ||
             speculative_outcome_output_token_stride <
                 speculative_first_output_token_index + total_rows ||
             speculative_outcome_request_index < 0 ||
             speculative_first_output_token_index < 0 ||
             !speculative_outcome_ready_event ||
             !position_ids_device_override ||
             !mtp_sidecar_position_ids_dev_))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Speculative-outcome MTP sidecar token preparation received invalid resident metadata");
            return false;
        }
        if (!kv_cache_only && token_count != 1)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Batched MTP sidecar is only supported for kv_cache_only catchup");
            return false;
        }
        if (draft_condition_tokens)
        {
            const int vocab_size = graph_builder_
                                       ? graph_builder_->config().vocab_size
                                       : 0;
            for (int i = 0; i < total_rows; ++i)
            {
                const int32_t token = draft_condition_tokens[i];
                if (token < 0 || (vocab_size > 0 && token >= vocab_size))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] MTP sidecar received invalid host token="
                              << token << " at row=" << i
                              << " seq_len=" << token_count
                              << " batch=" << request_batch
                              << " context=" << sidecar_context
                              << " vocab=" << vocab_size);
                    return false;
                }
            }
        }
        PerfStatsCollector::ScopedTimer total_timer(
            "mtp",
            "sidecar_depth0_total",
            phase,
            device_key,
            {{"context", sidecar_context},
             {"depth", "0"},
             {"device_tokens", boolTag(use_device_condition_tokens)},
             {"device_positions", boolTag(use_device_position_ids)},
             {"kv_cache_only", boolTag(kv_cache_only)},
             {"seq_len", std::to_string(token_count)},
             {"batch", std::to_string(request_batch)}});
        PerfStatsCollector::addCounter(
            "mtp",
            materialize_only ? "sidecar_materializations_without_launch" : "sidecar_depth0_calls",
            1.0,
            phase,
            device_key,
            {{"context", sidecar_context},
             {"depth", "0"},
             {"device_tokens", boolTag(use_device_condition_tokens)},
             {"device_positions", boolTag(use_device_position_ids)},
             {"kv_cache_only", boolTag(kv_cache_only)},
             {"seq_len", std::to_string(token_count)},
             {"batch", std::to_string(request_batch)}});
        if (!graph_builder_ || (!graph_builder_->config().mtp.enabled &&
            !(materialize_only && retainsMTPGraphCapacity(graph_builder_->config().mtp))))
            return false;
        if (state_.mtp_kv_caches.empty() || !state_.mtp_kv_caches[0])
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP sidecar requires an initialized MTP KV cache");
            return false;
        }
        const FrozenModelWeightSet *mtp_weight_set =
            selectMTPDecodeWeightSet("MTP sidecar");
        if (!mtp_weight_set)
        {
            return false;
        }

        ModelWeightBindings bindings;
        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "sidecar_resolve_weight_bindings",
                phase,
                device_key,
                {{"depth", "0"}});
            bindings = makeModelWeightBindings(
                *mtp_weight_set, state_.device_id);
        }
        if (bindings.mtp.empty() || bindings.mtp.depths.empty())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP sidecar requested without MTP weight bindings");
            return false;
        }
        const auto &depth0_bindings = bindings.mtp.depths[0];
        const bool mtp_moe_sidecar =
            depth0_bindings.fa_block.moe_gate ||
            depth0_bindings.fa_block.moe_gate_exps ||
            depth0_bindings.fa_block.moe_up_exps ||
            depth0_bindings.fa_block.moe_down_exps;

        auto get_extension = [&](BufferId id) -> TensorBase *
        {
            auto it = state_.extension_buffers.find(id);
            return it == state_.extension_buffers.end() ? nullptr : it->second.get();
        };

        TensorBase *mtp_logits = get_extension(BufferId::MTP_LOGITS);
        TensorBase *mtp_logits_gathered =
            get_extension(BufferId::MTP_LOGITS_GATHERED);
        TensorBase *mtp_hidden = get_extension(BufferId::MTP_HIDDEN);
        TensorBase *mtp_embedding = get_extension(BufferId::MTP_EMBEDDING);
        TensorBase *mtp_norm_hidden = get_extension(BufferId::MTP_NORM_HIDDEN);
        TensorBase *mtp_norm_embedding = get_extension(BufferId::MTP_NORM_EMBEDDING);
        TensorBase *mtp_concat = get_extension(BufferId::MTP_CONCAT);
        TensorBase *mtp_projected = get_extension(BufferId::MTP_PROJECTED);
        TensorBase *mtp_q = get_extension(BufferId::MTP_Q_PROJ);
        TensorBase *mtp_k = get_extension(BufferId::MTP_K_PROJ);
        TensorBase *mtp_v = get_extension(BufferId::MTP_V_PROJ);
        TensorBase *mtp_k_full_prefill = get_extension(BufferId::MTP_K_FULL_PREFILL);
        TensorBase *mtp_v_full_prefill = get_extension(BufferId::MTP_V_FULL_PREFILL);
        TensorBase *mtp_q_raw = get_extension(BufferId::MTP_FA_Q_RAW);
        TensorBase *mtp_q_gate = get_extension(BufferId::MTP_FA_GATE);
        TensorBase *mtp_attn_output = get_extension(BufferId::MTP_ATTN_OUTPUT);
        TensorBase *mtp_attn_proj = get_extension(BufferId::MTP_ATTN_PROJ);
        TensorBase *mtp_gate = get_extension(BufferId::MTP_GATE_PROJ);
        TensorBase *mtp_up = get_extension(BufferId::MTP_UP_PROJ);
        TensorBase *mtp_ffn_output = get_extension(BufferId::MTP_FFN_OUTPUT);

        const bool missing_common_buffers =
            !terminal_hidden ||
            !mtp_embedding || !mtp_norm_hidden || !mtp_norm_embedding ||
            !mtp_concat || !mtp_projected ||
            !mtp_q || !mtp_k || !mtp_v || !mtp_q_raw || !mtp_q_gate;
        const bool missing_full_sidecar_buffers =
            !mtp_logits || !mtp_hidden ||
            !mtp_attn_output || !mtp_attn_proj || !mtp_gate || !mtp_up ||
            !mtp_ffn_output;
        const bool requires_mtp_logits_gather =
            mtpSidecarRequiresLogitsGather(kv_cache_only);
        if (missing_common_buffers || (!kv_cache_only && missing_full_sidecar_buffers))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP sidecar missing required buffers");
            return false;
        }
        if (requires_mtp_logits_gather && !mtp_logits_gathered)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP sidecar is missing its gathered full-vocabulary logits buffer");
            return false;
        }
        auto require_rows = [&](const char *name, const TensorBase *tensor) -> bool
        {
            if (!tensor)
                return true;
            if (tensor->rows() >= static_cast<size_t>(total_rows))
                return true;
            LOG_ERROR("[DeviceGraphOrchestrator] MTP sidecar buffer " << name
                                                                      << " has only " << tensor->rows()
                                                                      << " rows for total_rows=" << total_rows
                                                                      << " seq_len=" << token_count
                                                                      << " batch=" << request_batch);
            return false;
        };
        if (!require_rows("terminal_hidden", terminal_hidden) ||
            !require_rows("mtp_embedding", mtp_embedding) ||
            !require_rows("mtp_norm_hidden", mtp_norm_hidden) ||
            !require_rows("mtp_norm_embedding", mtp_norm_embedding) ||
            !require_rows("mtp_concat", mtp_concat) ||
            !require_rows("mtp_projected", mtp_projected) ||
            !require_rows("mtp_q", mtp_q) ||
            !require_rows("mtp_k", mtp_k) ||
            !require_rows("mtp_v", mtp_v) ||
            !require_rows("mtp_k_full_prefill", mtp_k_full_prefill) ||
            !require_rows("mtp_v_full_prefill", mtp_v_full_prefill) ||
            !require_rows("mtp_q_raw", mtp_q_raw) ||
            !require_rows("mtp_q_gate", mtp_q_gate) ||
            (!kv_cache_only &&
             (!require_rows("mtp_logits", mtp_logits) ||
              (requires_mtp_logits_gather &&
               !require_rows("mtp_logits_gathered", mtp_logits_gathered)) ||
              !require_rows("mtp_hidden", mtp_hidden) ||
              !require_rows("mtp_attn_output", mtp_attn_output) ||
              !require_rows("mtp_attn_proj", mtp_attn_proj) ||
              !require_rows("mtp_gate", mtp_gate) ||
              !require_rows("mtp_up", mtp_up) ||
              !require_rows("mtp_ffn_output", mtp_ffn_output))))
        {
            return false;
        }

        MTPForwardOutput output;
        output.logits = mtp_logits;
        output.gathered_logits = requires_mtp_logits_gather
                                     ? mtp_logits_gathered
                                     : nullptr;
        output.hidden = mtp_hidden;
        output.embedding = mtp_embedding;
        output.norm_hidden = mtp_norm_hidden;
        output.norm_embedding = mtp_norm_embedding;
        output.concat = mtp_concat;
        output.projected = mtp_projected;
        output.q = mtp_q;
        output.k = mtp_k;
        output.v = mtp_v;
        output.k_full_prefill = mtp_k_full_prefill;
        output.v_full_prefill = mtp_v_full_prefill;
        output.q_raw = mtp_q_raw;
        output.q_gate = mtp_q_gate;
        output.attn_output = mtp_attn_output;
        output.attn_proj = mtp_attn_proj;
        output.gate = mtp_gate;
        output.up = mtp_up;
        output.ffn_output = mtp_ffn_output;
        if (!kv_cache_only && mtp_moe_sidecar)
            output.moe = MoEActivationBindings::bind(
                graph_builder_->config().moe.routed_compute_policy, total_rows, get_extension);

        /*
         * Resolve cache identity and input publication from the same typed role.
         * A graph can no longer be selected with one row-index convention and
         * reverse-mapped to a token slot with another convention.
         */
        MTPSidecarCaptureRole sidecar_role = MTPSidecarCaptureRole::Full;
        MTPSidecarGraphCache *sidecar_cache_ptr = nullptr;
        if (kv_cache_only)
        {
            sidecar_role = MTPSidecarCaptureRole::KVOnly;
            if (total_rows == 1)
            {
                sidecar_cache_ptr =
                    use_device_condition_tokens
                        ? &mtp_sidecar_depth0_kv_only_device_token_cache_
                        : &mtp_sidecar_depth0_kv_only_cache_;
            }
            else
            {
                const size_t cache_index =
                    mtp_sidecar_capture_layout_.kvOnlyBatchCacheIndex(total_rows);
                if (cache_index >= mtp_sidecar_depth0_kv_only_batch_caches_.size() ||
                    !mtp_sidecar_depth0_kv_only_batch_caches_[cache_index])
                {
                    throw std::logic_error(
                        "Configured MTP sidecar row count has no preallocated KV-only graph cache");
                }
                sidecar_cache_ptr =
                    mtp_sidecar_depth0_kv_only_batch_caches_[cache_index].get();
            }
        }
        else if (terminal_hidden_buffer_id == BufferId::MTP_HIDDEN)
        {
            sidecar_role = MTPSidecarCaptureRole::Chained;
            sidecar_cache_ptr =
                use_device_condition_tokens
                    ? &mtp_sidecar_depth0_chained_device_token_cache_
                    : &mtp_sidecar_depth0_chained_cache_;
        }
        else
        {
            sidecar_cache_ptr =
                use_device_condition_tokens
                    ? &mtp_sidecar_depth0_device_token_cache_
                    : &mtp_sidecar_depth0_cache_;
        }

        const int condition_token_slot =
            use_device_condition_tokens
                ? mtp_sidecar_capture_layout_.conditionTokenSlot(
                      sidecar_role,
                      total_rows)
                : -1;
        MTPSidecarGraphCache &sidecar_cache = *sidecar_cache_ptr;
        int32_t *condition_token_device = nullptr;
        if (use_device_condition_tokens)
        {
            const int required_capacity =
                (condition_token_slot + 1) *
                mtp_sidecar_condition_token_slot_width_;
            if (condition_token_slot < 0 ||
                !mtp_sidecar_condition_token_dev_ ||
                mtp_sidecar_condition_token_capacity_ < required_capacity ||
                total_rows > mtp_sidecar_condition_token_slot_width_)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Device-token MTP sidecar requires a role-owned"
                          << " MTP_CONDITION_TOKEN slot: slot=" << condition_token_slot
                          << " capacity=" << mtp_sidecar_condition_token_capacity_
                          << " required=" << required_capacity
                          << " requested_rows=" << total_rows
                          << " context=" << sidecar_context);
                return false;
            }
            condition_token_device =
                static_cast<int32_t *>(mtp_sidecar_condition_token_dev_) +
                condition_token_slot *
                    mtp_sidecar_condition_token_slot_width_;
        }

        MTPForwardInput input;
        input.capture_role = sidecar_role;
        input.draft_token_ids = draft_condition_tokens;
        input.draft_token_ids_device =
            use_device_condition_tokens ? condition_token_device : nullptr;
        input.terminal_hidden = terminal_hidden;
        input.kv_cache = state_.mtp_kv_caches[0].get();
        input.position_ids = &position_id;
        input.position_ids_device = effective_position_ids_device;
        input.sequence_lengths = nullptr;
        input.batch_size = request_batch;
        input.seq_len = token_count;
        input.first_sequence_index = first_seq_idx;
        input.device_state_publication_stream =
            state_.device_id.is_gpu()
                ? explicitGPUStreamForOperation(
                      "mtp_graph_device_state_publication")
                : nullptr;
        if (state_.device_id.is_gpu() &&
            !input.device_state_publication_stream)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] MTP graph construction requires an "
                "explicit non-null device-state publication stream");
            return false;
        }
        input.device = state_.device_id;
        input.terminal_hidden_buffer_id = terminal_hidden_buffer_id;
        input.kv_cache_only = kv_cache_only;

        const uint64_t current_moe_placement_epoch = moePlacementEpoch();
        const bool sidecar_moe_epoch_sensitive = mtp_moe_sidecar && !kv_cache_only;
        const uint64_t sidecar_moe_epoch_key =
            sidecar_moe_epoch_sensitive ? current_moe_placement_epoch : 0;
        const bool missing_cached_graph =
            !sidecar_cache.valid || !sidecar_cache.graph;
        const bool terminal_hidden_binding_changed =
            sidecar_cache.terminal_hidden != terminal_hidden;
        const bool graph_shape_changed =
            sidecar_cache.seq_len != token_count ||
            sidecar_cache.batch_size != request_batch ||
            sidecar_cache.first_seq_idx != first_seq_idx;
        const bool token_binding_changed =
            sidecar_cache.uses_device_token_ids != use_device_condition_tokens ||
            sidecar_cache.condition_token_slot != condition_token_slot ||
            sidecar_cache.condition_token_device != condition_token_device;
        const bool position_binding_changed =
            sidecar_cache.uses_device_position_ids != use_device_position_ids ||
            sidecar_cache.position_ids_device != effective_position_ids_device;
        const bool moe_binding_changed =
            sidecar_cache.moe_epoch_sensitive != sidecar_moe_epoch_sensitive ||
            sidecar_cache.moe_placement_epoch != sidecar_moe_epoch_key;
        const bool needs_graph_rebuild =
            missing_cached_graph ||
            terminal_hidden_binding_changed ||
            graph_shape_changed ||
            token_binding_changed ||
            position_binding_changed ||
            moe_binding_changed;

        const bool rebuilt_graph = needs_graph_rebuild;
        if (needs_graph_rebuild)
        {
            std::string rebuild_reasons;
            const auto append_rebuild_reason =
                [&](bool active, const char *reason)
            {
                if (!active)
                    return;
                if (!rebuild_reasons.empty())
                    rebuild_reasons += ",";
                rebuild_reasons += reason;
            };
            append_rebuild_reason(missing_cached_graph, "missing_graph");
            append_rebuild_reason(
                terminal_hidden_binding_changed,
                "terminal_hidden_binding");
            append_rebuild_reason(graph_shape_changed, "graph_shape");
            append_rebuild_reason(token_binding_changed, "token_binding");
            append_rebuild_reason(
                position_binding_changed,
                "position_binding");
            append_rebuild_reason(moe_binding_changed, "moe_binding");
            PerfStatsCollector::addCounter(
                "mtp",
                "sidecar_graph_rebuild_reasons",
                1.0,
                phase,
                device_key,
                {{"context", sidecar_context},
                 {"reasons", rebuild_reasons},
                 {"seq_len", std::to_string(token_count)}});

            sidecar_cache.invalidate();
            sidecar_cache.token_id =
                use_device_condition_tokens ? -1 : draft_condition_tokens[0];
            sidecar_cache.token_ids.assign(static_cast<size_t>(total_rows), 0);
            sidecar_cache.position_ids.assign(static_cast<size_t>(total_rows), 0);
            sidecar_cache.position_id = position_id;
            sidecar_cache.seq_len = token_count;
            sidecar_cache.batch_size = request_batch;
            sidecar_cache.first_seq_idx = first_seq_idx;
            sidecar_cache.uses_device_token_ids = use_device_condition_tokens;
            sidecar_cache.uses_device_position_ids = use_device_position_ids;
            sidecar_cache.condition_token_slot = condition_token_slot;
            sidecar_cache.condition_token_device = condition_token_device;
            sidecar_cache.position_ids_device = effective_position_ids_device;
            sidecar_cache.terminal_hidden = terminal_hidden;
            sidecar_cache.moe_placement_epoch = sidecar_moe_epoch_key;
            sidecar_cache.moe_epoch_sensitive = sidecar_moe_epoch_sensitive;
            sidecar_cache.capture_perf_context = sidecar_context;
            sidecar_cache.segment_cache.perf_context =
                sidecar_cache.capture_perf_context;

            MTPForwardInput cached_input = input;
            cached_input.draft_token_ids = sidecar_cache.token_ids.data();
            cached_input.draft_token_ids_device =
                use_device_condition_tokens ? condition_token_device : nullptr;
            cached_input.position_ids = sidecar_cache.position_ids.data();
            cached_input.position_ids_device = effective_position_ids_device;

            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "sidecar_build_graph",
                phase,
                device_key,
                {{"depth", "0"}, {"seq_len", std::to_string(token_count)}});
            ComputeGraph graph = graph_builder_->buildMTPGraph(
                0,
                bindings.mtp.depths[0],
                cached_input,
                output);
            if (graph.size() == 0)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to build MTP sidecar graph");
                return false;
            }
            if (terminal_hidden_buffer_id ==
                BufferId::PREFIX_TERMINAL_HIDDEN)
            {
                /*
                 * A persistent terminal-hidden publication survives a sidecar
                 * launch only when every stage declares that mailbox as
                 * read-only.  Validate the production graph itself at the cold
                 * construction boundary; a later writer is an architecture
                 * error, not a reason to copy or reselect the row after every
                 * sidecar.
                 */
                for (const std::string &node_name : graph.getExecutionOrder())
                {
                    const ComputeNode *node = graph.getNode(node_name);
                    if (!node || !node->stage)
                    {
                        LOG_ERROR("[DeviceGraphOrchestrator] MTP sidecar graph contains an incomplete node while validating terminal-hidden ownership: node="
                                  << node_name);
                        return false;
                    }
                    const auto writes =
                        node->stage->bufferContract().allWrites();
                    const bool overwrites_terminal_hidden =
                        std::any_of(
                            writes.begin(),
                            writes.end(),
                            [](const BufferBinding &binding)
                            {
                                return binding.id ==
                                       BufferId::PREFIX_TERMINAL_HIDDEN;
                            });
                    if (overwrites_terminal_hidden)
                    {
                        LOG_ERROR("[DeviceGraphOrchestrator] MTP sidecar stage violates the persistent terminal-hidden read-only contract: node="
                                  << node_name
                                  << " stage=" << node->stage->name());
                        return false;
                    }
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "sidecar_terminal_hidden_read_only_contracts",
                    1.0,
                    "graph_setup",
                    device_key,
                    {{"context", sidecar_context},
                     {"nodes", std::to_string(graph.size())}});
            }
            if (state_.device_id.is_gpu())
            {
                publishGraphBuildDeviceStateReady(
                    mtp_graph_build_device_state_ready_,
                    cached_input.device_state_publication_stream,
                    "mtp_sidecar_graph_build");
            }

            sidecar_cache.graph = std::make_unique<ComputeGraph>(std::move(graph));
            sidecar_cache.dynamic_param_stages.clear();
            sidecar_cache.moe_overlay_collective_runtime_stages.clear();
            sidecar_cache.collective_nodes.clear();
            for (const auto &node_name : sidecar_cache.graph->getExecutionOrder())
            {
                ComputeNode *node = sidecar_cache.graph->getNode(node_name);
                if (!node || !node->stage)
                    continue;
                if (node->stage->hasDynamicParams())
                    sidecar_cache.dynamic_param_stages.push_back(node->stage.get());
                if (node->stage->hasMoEOverlayCollectiveRuntimeParams())
                {
                    sidecar_cache.moe_overlay_collective_runtime_stages.push_back(
                        node->stage.get());
                }

                if (node->stage->isCollectiveStage())
                {
                    sidecar_cache.collective_nodes.insert(node_name);
                }
            }
            sidecar_cache.valid = true;
            PerfStatsCollector::addCounter(
                "mtp",
                "sidecar_collective_node_scans",
                1.0,
                phase,
                device_key,
                {{"depth", "0"},
                 {"seq_len", std::to_string(token_count)},
                 {"has_collectives", boolTag(!sidecar_cache.collective_nodes.empty())},
                 {"node_count", std::to_string(sidecar_cache.collective_nodes.size())}});
            PerfStatsCollector::addCounter(
                "mtp",
                "sidecar_graph_cache_misses",
                1.0,
                phase,
                device_key,
                {{"context", sidecar_context},
                 {"depth", "0"},
                 {"device_tokens", boolTag(use_device_condition_tokens)},
                 {"device_positions", boolTag(use_device_position_ids)},
                 {"kv_cache_only", boolTag(kv_cache_only)},
                 {"seq_len", std::to_string(token_count)},
                 {"moe_placement_epoch", std::to_string(sidecar_moe_epoch_key)}});
        }
        else
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "sidecar_graph_cache_hits",
                1.0,
                phase,
                device_key,
                {{"context", sidecar_context},
                 {"depth", "0"},
                 {"device_tokens", boolTag(use_device_condition_tokens)},
                 {"device_positions", boolTag(use_device_position_ids)},
                 {"kv_cache_only", boolTag(kv_cache_only)},
                 {"seq_len", std::to_string(token_count)},
                 {"moe_placement_epoch", std::to_string(sidecar_moe_epoch_key)}});
        }

        if (materialize_only && sidecar_cache.segment_cache.initialized &&
            !sidecar_cache.segment_cache.needs_capture &&
            sidecar_cache.workspace_generation == workspaceGeneration(state_.device_id))
            return sidecar_cache.segment_cache.capture_stream && sidecar_cache.segment_cache.gpu_ctx_ref &&
                !sidecar_cache.segment_cache.segments.empty();

        IDeviceContext *ctx = getDeviceContext(state_.device_id);
        if (!ctx)
            return false;

        IWorkerGPUContext *sidecar_gpu_ctx = nullptr;
        void *sidecar_dynamic_stream = nullptr;
        const bool try_gpu_graph_capture =
            state_.device_id.is_gpu() &&
            debugEnv().execution.gpu_graphs;
        const auto sidecar_epoch_ownership = moeOverlaySidecarEpochOwnership(
            sidecar_role,
            static_cast<bool>(moe_overlay_inference_transaction_coordinator_));
        const bool participates_in_overlay_graph_group =
            sidecar_epoch_ownership ==
            MoEOverlaySidecarEpochOwnership::GraphSequence;
        if (state_.device_id.is_gpu() && !try_gpu_graph_capture)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] GPU MTP sidecars require mandatory "
                "graph capture; eager execution is forbidden");
            return false;
        }
        if (state_.device_id.is_gpu())
        {
            auto &pool = GPUDeviceContextPool::instance();
            sidecar_gpu_ctx = &pool.getContext(state_.device_id);
            sidecar_dynamic_stream = sidecar_gpu_ctx->defaultStream();
            if (!sidecar_dynamic_stream)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] MTP sidecar requires an explicit non-null GPU stream for "
                          << state_.device_id.toString());
                return false;
            }
        }
        if (try_gpu_graph_capture)
        {
            if (sidecar_cache.segment_cache.ensureCaptureStream(
                    sidecar_gpu_ctx,
                    state_.device_id,
                    /*context_from_process_pool=*/true) &&
                sidecar_cache.segment_cache.ensureCaptureInputEvent(
                    sidecar_gpu_ctx) &&
                sidecar_cache.segment_cache.ensureCaptureOutputEvent(
                    sidecar_gpu_ctx))
            {
                void *capture_stream = sidecar_cache.segment_cache.capture_stream;
                sidecar_dynamic_stream = capture_stream;
            }
            else
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to create MTP sidecar graph capture stream");
                return false;
            }
        }
        if (state_.device_id.is_gpu() &&
            !waitForPendingGraphBuildDeviceStateReady(
                mtp_graph_build_device_state_ready_,
                sidecar_dynamic_stream,
                DeviceTimelineRole::MTPSidecarGraph,
                "mtp_sidecar_graph_build_state"))
        {
            return false;
        }
        if (!materialize_only && state_.device_id.is_gpu() &&
            !waitForPendingDeviceMoERebalanceMaintenance(
                sidecar_dynamic_stream,
                DeviceTimelineRole::MTPSidecarGraph,
                "mtp_sidecar_graph"))
        {
            return false;
        }
        // Only a full standalone sidecar reads movable experts here. KV-only
        // work publishes shifted-KV readiness, not an ambient placement lease;
        // coordinated full sidecars acquire after graph-sequence admission.
        if (!materialize_only && state_.device_id.is_gpu() &&
            sidecar_epoch_ownership ==
                MoEOverlaySidecarEpochOwnership::ExternalReader &&
            !acquireMoEOverlayEpochForExternalTransaction(
                sidecar_dynamic_stream,
                DeviceTimelineRole::MTPSidecarGraph,
                "mtp_sidecar_graph"))
        {
            return false;
        }
        if (!materialize_only && state_.device_id.is_gpu() &&
            !waitForPendingShiftedMTPKVReady(
                sidecar_dynamic_stream,
                "mtp_sidecar_before_execute"))
        {
            return false;
        }
        /*
         * PREFIX_TERMINAL_HIDDEN is a reusable device mailbox. Shifted prefill
         * and main-forward refresh publish its exact row-selection stream; the
         * sidecar is the payload owner that consumes that handoff. This is the
         * R -> S edge in the forward -> row-select -> sidecar pipeline.
         */
        if (!materialize_only && state_.device_id.is_gpu() &&
            terminal_hidden_buffer_id == BufferId::PREFIX_TERMINAL_HIDDEN &&
            !waitForPendingMTPPrefillTerminalArchiveReady(
                sidecar_dynamic_stream,
                "mtp_sidecar_terminal_hidden_input"))
        {
            return false;
        }
        /*
         * The sidecar is an observer of main-model KV/GDN/short-conv state,
         * never its event owner. The typed dispatcher enforces that distinction
         * and leaves both publication handoffs for the next main forward graph.
         */
        if (!materialize_only && state_.device_id.is_gpu() &&
            !joinPublishedLiveStateHandoffs(
                sidecar_dynamic_stream,
                DeviceTimelineRole::MTPSidecarGraph,
                "mtp_sidecar_before_execute"))
        {
            return false;
        }
        if (!materialize_only && use_device_condition_tokens)
        {
            DeviceResidentLogicalStateReadScope logical_state_read(
                *this,
                sidecar_dynamic_stream,
                "mtp_sidecar_logical_state_input_staging");
            if (!logical_state_read.ready())
                return false;

            if (!sidecar_dynamic_stream)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Device-token MTP sidecar requires an explicit non-null stream");
                return false;
            }
            bool condition_source_ready = true;
            if (external_device_condition_tokens &&
                draft_condition_ready_slot >= 0)
            {
                if (draft_condition_ready_is_target)
                {
                    condition_source_ready =
                        draft_condition_ready_count == 1 &&
                        waitForRequiredStochasticTargetSampleReady(
                            draft_condition_ready_slot,
                            sidecar_dynamic_stream,
                            "mtp_sidecar_device_target_token");
                }
                else
                {
                    for (int row = 0;
                         row < draft_condition_ready_count && condition_source_ready;
                         ++row)
                    {
                        condition_source_ready =
                            waitForRequiredStochasticDraftSampleReadyRange(
                                draft_condition_ready_slot +
                                    row * draft_condition_ready_stride,
                                1,
                                sidecar_dynamic_stream,
                                "mtp_sidecar_device_token");
                    }
                }
            }
            if (!condition_source_ready)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to order MTP sidecar token copy after deferred sample");
                return false;
            }
            IBackend *backend = getBackendFor(state_.device_id);
            if (!backend)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to resolve backend for MTP sidecar token staging");
                return false;
            }

            const size_t staged_token_bytes =
                sizeof(int32_t) * static_cast<size_t>(total_rows);
            const bool stages_position_rows =
                prepare_device_condition_tokens_from_speculative_outcome ||
                (external_device_condition_tokens &&
                 position_ids_device_override);
            if (!arena_ ||
                !arena_->prepareForWrite(
                    BufferId::MTP_CONDITION_TOKEN,
                    state_.device_id,
                    sidecar_dynamic_stream) ||
                (stages_position_rows &&
                 !arena_->prepareForWrite(
                     BufferId::MTP_POSITION_IDS,
                     state_.device_id,
                     sidecar_dynamic_stream)))
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] MTP sidecar input staging "
                    "could not acquire its arena write transaction");
                return false;
            }
            bool staged_tokens_ok = false;
            if (prepare_device_condition_tokens_from_speculative_outcome)
            {
                /*
                 * The compact verifier reducer has already produced output
                 * tokens and accepted counts on its own stream.  Wait for that
                 * producer event from the sidecar stream, then prepare the
                 * bounded suffix directly into the persistent condition-token
                 * slot consumed by the captured KV-only MTP graph. The same
                 * launch expands the canonical verifier-base count into the
                 * persistent position row. Rows beyond the device-resident
                 * accepted count receive a harmless filler token and are
                 * removed by shifted-KV publication.
                 */
                if (!backend->streamWaitEvent(
                        sidecar_dynamic_stream,
                        speculative_outcome_ready_event,
                        state_.device_id.gpu_ordinal()))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Failed to order shifted MTP suffix token preparation after compact verifier outcome");
                    return false;
                }
                staged_tokens_ok =
                    backend->enqueuePrepareSpeculativeShiftedKVTokens(
                        speculative_outcome_meta_device,
                        speculative_outcome_meta_stride,
                        speculative_outcome_output_tokens_device,
                        speculative_outcome_output_token_stride,
                        speculative_outcome_request_index,
                        speculative_first_output_token_index,
                        total_rows,
                        /*filler_token=*/0,
                        state_.device_id.gpu_ordinal(),
                        sidecar_dynamic_stream,
                        condition_token_device,
                        position_ids_device_override,
                        device_position_offset,
                        mtp_sidecar_position_ids_dev_);
                if (staged_tokens_ok)
                {
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "speculative_shifted_kv_token_position_fusions",
                        static_cast<double>(total_rows),
                        phase,
                        device_key,
                        {{"rows", std::to_string(total_rows)},
                         {"position_offset", std::to_string(device_position_offset)}});
                }
            }
            else if (external_device_condition_tokens &&
                     position_ids_device_override)
            {
                /*
                 * Request-major proposal inputs expose one strided token per
                 * request and one base position per request. Shifted-prefill
                 * inputs instead expose contiguous token/position rows. Both
                 * layouts are normalized by the same allocation-free backend
                 * primitive; only the source geometry differs.
                 */
                const int preparation_rows =
                    compose_batched_device_inputs
                        ? request_batch
                        : total_rows;
                const int preparation_token_stride =
                    compose_batched_device_inputs
                        ? draft_condition_token_stride
                        : 1;
                const int preparation_position_offset =
                    compose_batched_device_inputs
                        ? device_position_offset
                        : 0;
                staged_tokens_ok =
                    backend->enqueuePrepareMTPBatchedSidecarInputs(
                        draft_condition_tokens_device,
                        preparation_token_stride,
                        position_ids_device_override,
                        preparation_position_offset,
                        preparation_rows,
                        state_.device_id.gpu_ordinal(),
                        sidecar_dynamic_stream,
                        condition_token_device,
                        mtp_sidecar_position_ids_dev_);
                if (staged_tokens_ok)
                {
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "device_resident_sidecar_input_normalizations",
                        static_cast<double>(preparation_rows),
                        phase,
                        device_key,
                        {{"layout",
                          compose_batched_device_inputs
                              ? "request_major"
                              : "contiguous_rows"},
                         {"rows", std::to_string(preparation_rows)}});
                }
            }
            else if (external_device_condition_tokens)
            {
                /*
                 * A token-only sidecar still copies into its role-owned slot
                 * because graph capture records the destination pointer, not
                 * the producer mailbox.
                 */
                staged_tokens_ok = backend->deviceCopyAsync(
                    condition_token_device,
                    draft_condition_tokens_device,
                    staged_token_bytes,
                    state_.device_id.gpu_ordinal(),
                    sidecar_dynamic_stream);
            }
            if (!staged_tokens_ok)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to stage MTP sidecar token rows"
                          << " source="
                          << (prepare_device_condition_tokens_from_speculative_outcome
                                  ? "resident_outcome"
                                  : (external_device_condition_tokens ? "device" : "host"))
                          << " rows=" << total_rows);
                return false;
            }
            try
            {
                /*
                 * These direct preparation kernels sit outside an ordinary
                 * ComputeStage epilogue. Publish their exact stream so the
                 * captured sidecar and optional diagnostic graphs consume one
                 * canonical arena authority instead of relying on pointer
                 * residency or same-stream coincidence.
                 */
                arena_->markWritten(
                    BufferId::MTP_CONDITION_TOKEN,
                    state_.device_id,
                    sidecar_dynamic_stream);
                if (stages_position_rows)
                {
                    arena_->markWritten(
                        BufferId::MTP_POSITION_IDS,
                        state_.device_id,
                        sidecar_dynamic_stream);
                }
            }
            catch (const std::exception &exception)
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] MTP sidecar input staging "
                    "could not publish arena authority: "
                    << exception.what());
                return false;
            }
            if (external_device_condition_tokens &&
                draft_condition_ready_slot >= 0 &&
                draft_condition_ready_is_target)
            {
                /*
                 * The deferred first target token has two consumers in the
                 * vLLM-style greedy lane: the first MTP sidecar and the later
                 * verifier-token row materializer.  This sidecar stream has
                 * already waited on the sampler event before staging the token,
                 * so record a fresh ready event here.  The verifier can then
                 * wait on the sidecar handoff even if the original sampler
                 * event was consumed or replaced by intermediate graph work.
                 */
                if (!recordStochasticTargetSampleReady(
                        draft_condition_ready_slot,
                        sidecar_dynamic_stream, false, StochasticTargetPublication::PreserveHistoryRight))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Failed to preserve deferred target token readiness after MTP sidecar staging");
                    return false;
                }
            }

            PerfStatsCollector::addCounter(
                "mtp",
                "mtp_sidecar_device_token_inputs",
                1.0,
                phase,
                device_key,
                {{"seq_len", std::to_string(token_count)},
                 {"slot", std::to_string(condition_token_slot)},
                 {"source", external_device_condition_tokens ? "device" : "host"},
                 {"source_stride", std::to_string(draft_condition_token_stride)},
                 {"position_offset", std::to_string(device_position_offset)}});
        }

        const bool sidecar_uses_gpu_workspace =
            state_.device_id.is_gpu() && sidecar_cache.graph;
        const uint64_t current_workspace_generation =
            sidecar_uses_gpu_workspace ? workspaceGeneration(state_.device_id) : 0;
        const bool sidecar_workspace_validated =
            sidecar_uses_gpu_workspace &&
            sidecar_cache.workspace_generation != 0 &&
            current_workspace_generation == sidecar_cache.workspace_generation;

        /*
         * Sparse ExpertOverlay consumers borrow route packets, so they need
         * not expose an arena input shape from which the allocator can infer M.
         * Supply this transaction's flattened request/token rows explicitly,
         * just as ForwardExecutionEngine does for main forward participants.
         * Omitting M substitutes the full KV context capacity and asks a
         * one-row predictor to grow the already published serial workspace.
         * Keep the exact mathematical role as well: a sidecar is not prefill,
         * even when a KV catch-up transaction carries several token rows.
         */
        if (!sidecar_workspace_validated &&
            !ensureDeviceWorkspaceAllocated(
                *sidecar_cache.graph,
                total_rows,
                WorkspaceGraphFamilyPolicy::SerialDeviceFamilyExactParticipant,
                WorkspaceGraphParticipantRole::MTPCondition))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to allocate MTP sidecar workspace before dynamic param update");
            return false;
        }
        if (sidecar_uses_gpu_workspace && !sidecar_workspace_validated)
        {
            const uint64_t new_workspace_generation = workspaceGeneration(state_.device_id);
            if (new_workspace_generation != sidecar_cache.workspace_generation)
            {
                if (sidecar_cache.workspace_generation != 0)
                {
                    LOG_DEBUG("[DeviceGraphOrchestrator] MTP sidecar workspace generation changed on "
                              << state_.device_id.toString()
                              << " from " << sidecar_cache.workspace_generation
                              << " to " << new_workspace_generation
                              << "; dropping captured sidecar replay state");
                    sidecar_cache.resetReplayStateAfterWorkspaceRebind();
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "sidecar_workspace_rebinds",
                        1.0,
                        phase,
                        device_key,
                        {{"depth", "0"}, {"seq_len", std::to_string(token_count)}});
                }
                sidecar_cache.workspace_generation = new_workspace_generation;
            }
        }

        sidecar_cache.token_id =
            use_device_condition_tokens ? -1 : draft_condition_tokens[0];
        if (sidecar_cache.token_ids.size() != static_cast<size_t>(total_rows) ||
            sidecar_cache.position_ids.size() != static_cast<size_t>(total_rows))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP sidecar cache has unstable token/position storage for rows="
                      << total_rows);
            return false;
        }
        if (external_device_condition_tokens ||
            prepare_device_condition_tokens_from_speculative_outcome)
        {
            std::fill(sidecar_cache.token_ids.begin(), sidecar_cache.token_ids.end(), 0);
        }
        else
        {
            std::copy(draft_condition_tokens,
                      draft_condition_tokens + total_rows,
                      sidecar_cache.token_ids.begin());
        }
        sidecar_cache.position_id = position_id;
        for (int request = 0; request < request_batch; ++request)
        {
            const int base_position =
                position_ids_override
                    ? position_ids_override[request]
                    : position_id + request * token_count;
            for (int i = 0; i < token_count; ++i)
            {
                const int row = request * token_count + i;
                sidecar_cache.position_ids[static_cast<size_t>(row)] =
                    base_position + i;
            }
        }
        if (sidecar_dynamic_stream)
        {
            std::string stream_binding_error;
            if (!mtp_sidecar::bindStagesToCaptureStream(*sidecar_cache.graph,
                                                         sidecar_dynamic_stream,
                                                         &stream_binding_error))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to bind MTP sidecar stages to explicit stream: "
                          << stream_binding_error);
                return false;
            }
        }
        for (auto *stage : sidecar_cache.dynamic_param_stages)
        {
            if (!stage)
                continue;

            if (use_device_position_ids)
            {
                if (!stage->supportsDeviceResidentDynamicPositionReplay())
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] MTP sidecar cannot replay with device-resident positions because stage "
                              << stage->name()
                              << " still requires host scalar dynamic position state");
                    return false;
                }
                stage->updateDynamicDevicePositionIds(
                    effective_position_ids_device,
                    total_rows);
                if (stage->type() == ComputeStageType::ROPE)
                    continue;
            }

            if (request_batch > 1 &&
                position_ids_override &&
                stage->type() == ComputeStageType::ROPE)
            {
                auto *rope_stage = dynamic_cast<RoPEStage *>(stage);
                if (!rope_stage)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] MTP sidecar found a ROPE stage"
                              << " that is not a RoPEStage");
                    return false;
                }
                /*
                 * Request-batched MTP sidecars use an explicit per-row
                 * position array: two logical requests can both be at
                 * position N, so the RoPE rows are [N, N], not the scalar
                 * contiguous range [N, N+1].  The stage pre-uploads these
                 * positions to its workspace-owned device buffer before graph
                 * capture/replay, which keeps captured RoPE execution free of
                 * host-to-device copies.
                 */
                rope_stage->updateDynamicPositionIds(
                    sidecar_cache.position_ids.data(),
                    total_rows);
                continue;
            }

            /*
             * When device position rows are active, every remaining dynamic
             * stage has explicitly declared that the scalar position argument
             * is ignored or replaced by device-resident metadata.  Use a
             * neutral value so resident replay cannot accidentally depend on
             * state_.positions while direct publication is being promoted.
             */
            const int scalar_position_for_dynamic_params =
                use_device_position_ids ? 0 : position_id;
            stage->updateDynamicParams(scalar_position_for_dynamic_params, token_count);
        }

        if (materialize_only)
        {
            // Stage construction, workspace admission and neutral device-position
            // binding above are shared with execution. Stop before diagnostics,
            // reader leases, input publication, sparse tickets or any model launch.
            const bool collectives = !sidecar_cache.collective_nodes.empty();
            auto policy = buildDecodeCapturePolicyForGraph(*sidecar_cache.graph, collectives,
                ctx, "cold chained MTP sidecar");
            if (!policy || !policy->allow_cached_graph_replay) return false;
            policy->defer_final_sync = true;
            policy->force_recapture = false;
            const DeviceId device = state_.device_id;
            policy->capture_boundary = [this, device](const std::string &boundary, void *stream) {
                ForwardInput input;
                input.device = device;
                return waitAtDecodeGraphCaptureBoundary(input, device, boundary, stream);
            };
            sidecar_cache.segment_cache.perf_context = sidecar_cache.capture_perf_context;
            ScopedStringOverride snapshot_scope(snapshot_context_, sidecar_cache.capture_perf_context);
            sidecar_cache.graph->reset();
            bool replay = false;
            if (!executor_.executeDecodeWithCapturePolicy(*sidecar_cache.graph, ctx,
                    &sidecar_cache.segment_cache, sidecar_dynamic_stream, sidecar_gpu_ctx,
                    collectives ? &sidecar_cache.collective_nodes : nullptr,
                    *policy, &replay, submission))
                return false;
            return sidecar_cache.segment_cache.initialized &&
                !sidecar_cache.segment_cache.needs_capture;
        }

        if (mtpGraphReuseDiagnosticsEnabledForContext(sidecar_context))
        {
            IBackend *backend = getBackendFor(state_.device_id);
            const void *terminal_hidden_device =
                terminal_hidden && terminal_hidden->deviceValid()
                    ? terminal_hidden->gpu_data_ptr()
                    : nullptr;
            const size_t terminal_hidden_row_bytes =
                mtpDiagnosticTensorRowBytes(terminal_hidden);
            const void *mtp_live_position =
                !state_.mtp_kv_caches.empty() && state_.mtp_kv_caches[0]
                    ? state_.mtp_kv_caches[0]
                          ->deviceSequenceCachedTokenCountPtr(first_seq_idx)
                    : nullptr;
            std::vector<MTPGraphReuseDiagnosticSource> diagnostic_sources{
                {"condition_token",
                 condition_token_device,
                 sizeof(int32_t) * static_cast<size_t>(total_rows)},
                {"position_id",
                 effective_position_ids_device,
                 sizeof(int32_t) * static_cast<size_t>(total_rows)},
                {"terminal_hidden",
                 terminal_hidden_device,
                 terminal_hidden_row_bytes},
                {"mtp_live_position",
                 mtp_live_position,
                 sizeof(int32_t)},
            };

            /*
             * Main-model checkpoints are inserted into the captured graph only
             * when mirrored-layer diagnostics were armed before construction.
             * Append the bank produced by the latest successful main forward
             * to this same stream observation.  Comparing attention-residual
             * and FFN-delta rows per layer identifies the first producer that
             * changes before the MTP sidecar sees its terminal hidden input.
             */
            if (terminal_hidden_buffer_id != BufferId::MTP_HIDDEN)
            {
                if (const auto *qwen_moe =
                        dynamic_cast<const Qwen35MoEGraph *>(
                            graph_builder_.get()))
                {
                    const auto checkpoints =
                        qwen_moe->mirroredLayerCheckpoints();
                    diagnostic_sources.reserve(
                        diagnostic_sources.size() + checkpoints.size());
                    for (const auto &checkpoint : checkpoints)
                    {
                        if (!checkpoint.tensor ||
                            last_mirrored_layer_checkpoint_prefix_.empty() ||
                            !checkpoint.name.starts_with(
                                last_mirrored_layer_checkpoint_prefix_))
                        {
                            continue;
                        }
                        diagnostic_sources.push_back({
                            checkpoint.name,
                            checkpoint.tensor->deviceValid()
                                ? checkpoint.tensor->gpu_data_ptr()
                                : nullptr,
                            mtpDiagnosticTensorRowBytes(checkpoint.tensor),
                        });
                    }
                }
            }

            /*
             * The sidecar stream has already inherited the main-forward
             * completion event at this boundary.  When a recurrent layer is
             * selected for graph-reuse diagnostics, observe its complete live
             * banks here as well.  This places the measurement after main
             * prefill but before the first MTP sidecar launch, which separates
             * a divergent prefill producer from a later speculative-state
             * publication error without introducing an unordered host probe.
             */
            if (const auto diagnostic_gdn_layer =
                    mtpGraphReuseDiagnosticGDNLayer())
            {
                auto *hybrid_cache =
                    dynamic_cast<IHybridKVCache *>(state_.kv_cache.get());
                const HybridGDNLayerState *gdn_state =
                    hybrid_cache
                        ? hybrid_cache->getGDNState(*diagnostic_gdn_layer)
                        : nullptr;
                if (!gdn_state)
                {
                    LOG_ERROR("[MTPGraphReuseDiagnostics] selected pre-sidecar diagnostic layer is not GDN"
                              << " layer=" << *diagnostic_gdn_layer
                              << " device=" << state_.device_id.toString());
                    return false;
                }

                const auto append_binding =
                    [&](const char *owner,
                        const GDNDeviceStateBinding &binding)
                {
                    const std::string prefix =
                        std::string{"layer"} +
                        std::to_string(*diagnostic_gdn_layer) +
                        "_" + owner;
                    if (binding.primary_state &&
                        binding.primary_state_floats > 0)
                    {
                        diagnostic_sources.push_back({
                            prefix + "_primary",
                            binding.primary_state,
                            static_cast<size_t>(
                                binding.primary_state_floats) *
                                sizeof(float),
                        });
                    }
                    if (binding.secondary_state &&
                        binding.secondary_state_floats > 0)
                    {
                        diagnostic_sources.push_back({
                            prefix + "_secondary",
                            binding.secondary_state,
                            static_cast<size_t>(
                                binding.secondary_state_floats) *
                                sizeof(float),
                        });
                    }
                    if (binding.request_state_bank &&
                        binding.request_state_bank_floats > 0)
                    {
                        diagnostic_sources.push_back({
                            prefix + "_requests",
                            binding.request_state_bank,
                            binding.request_state_bank_floats *
                                sizeof(float),
                        });
                    }
                };
                append_binding("conv", gdn_state->conv_device_state);
                append_binding(
                    "recurrence",
                    gdn_state->recurrence_device_state);
            }
            if (!backend ||
                !logMTPGraphReuseBoundaryDiagnostics(
                    "before_execute",
                    sidecar_context,
                    session_epoch_,
                    rebuilt_graph,
                    /*graph_replayed=*/false,
                    *backend,
                    state_.device_id,
                    sidecar_dynamic_stream,
                    diagnostic_sources))
            {
                return false;
            }
        }
        if (device_resident_logical_sequence_state_mailbox_.valid() &&
            device_resident_logical_sequence_state_mailbox_.live_state_epoch ==
                live_replay_state_epoch_)
        {
            snapshotDeviceResidentLogicalStatePhase(
                DeviceResidentLogicalStateDiagnosticPhase::SidecarEntry,
                sidecar_dynamic_stream,
                device_resident_logical_sequence_state_mailbox_.request_count,
                device_resident_logical_sequence_state_mailbox_
                    .publication_generation);
        }

        /*
         * KV-only shifted-cache sidecars contain no routed-expert stages and
         * therefore require no remote follower transaction.  A full MoE
         * sidecar publishes exactly once per rank-wide graph group, and every
         * LocalTP participant enters that group even when its current route set
         * has no remote work. Each child that does own sparse stages stamps the
         * coordinator-assigned operation id before capture or replay. Symmetric
         * scope entry prevents a zero-remote-work child from stranding the
         * group's completion fence.
         */
        const bool has_moe_overlay_sparse_boundary =
            !sidecar_cache.moe_overlay_collective_runtime_stages.empty();
        MoEOverlayInferenceExecutionDescriptor overlay_descriptor{
            .graph_role = MoEOverlayInferenceGraphRole::MTPDraft,
            .logical_step_id = 0,
            .placement_epoch =
                moe_overlay_inference_transaction_coordinator_
                    ? moe_overlay_inference_transaction_coordinator_
                          ->currentPlacementEpoch()
                    : 0,
            .request_count = request_batch,
            .logical_rows_per_request = token_count,
            .physical_rows_per_request = token_count,
            .draft_depth = -1,
            .sidecar_depth = -1,
        };
        MoEOverlayInferenceParticipantGraphScope overlay_graph_scope(
            participates_in_overlay_graph_group
                ? moe_overlay_inference_transaction_coordinator_
                : nullptr,
            overlay_descriptor,
            moe_overlay_inference_transaction_participant_index_);
        if (!overlay_graph_scope.ready())
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] ExpertOverlay transaction "
                "authority rejected the MTP sidecar graph"
                << " participant="
                << moe_overlay_inference_transaction_participant_index_
                << " device=" << state_.device_id.toString()
                << ": " << overlay_graph_scope.error());
            return false;
        }
        MoEOverlayExternalSequenceGraphLease overlay_sequence_graph_lease;
        if (state_.device_id.is_gpu() && overlay_graph_scope.active() &&
            !overlay_sequence_graph_lease.admit(
                *this,
                overlay_graph_scope.binding(),
                sidecar_dynamic_stream,
                DeviceTimelineRole::MTPSidecarGraph,
                sidecar_context.c_str()))
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] ExpertOverlay transaction "
                "could not admit the MTP sidecar into its complete "
                "execution-sequence residency lease"
                << " participant="
                << moe_overlay_inference_transaction_participant_index_
                << " device=" << state_.device_id.toString());
            return false;
        }
        const auto finish_overlay_graph =
            [&](bool execution_succeeded) -> bool
        {
            std::string overlay_finish_error;
            if (!overlay_graph_scope.finish(
                    execution_succeeded, &overlay_finish_error))
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] ExpertOverlay transaction "
                    "authority could not finish the MTP sidecar graph: "
                    << overlay_finish_error);
                return false;
            }
            if (!overlay_sequence_graph_lease.finish(execution_succeeded))
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] ExpertOverlay MTP sidecar "
                    "could not finish its execution-sequence residency edge");
                return false;
            }
            return true;
        };
        if (has_moe_overlay_sparse_boundary)
        {
            std::string runtime_binding_error;
            const auto params =
                resolveMTPMoEOverlayCollectiveRuntimeParams(
                    overlay_graph_scope,
                    IComputeStage::MoEOverlayCollectiveRuntimeParams::
                        ExecutionSemantics::MTPDraft,
                    /*expected_transaction_draft_depth=*/std::nullopt,
                    /*sparse_graph_depth=*/0,
                    &runtime_binding_error);
            if (!params)
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] MTP sidecar could not bind "
                    "its typed sparse execution phase: "
                    << runtime_binding_error);
                return false;
            }
            for (auto *stage :
                 sidecar_cache.moe_overlay_collective_runtime_stages)
            {
                if (!stage)
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] MTP sidecar contains a null sparse runtime stage");
                    return false;
                }
                stage->updateMoEOverlayCollectiveRuntimeParams(*params);
            }
        }

        /*
         * Every sidecar that binds the persistent terminal-hidden mailbox
         * acquires the same typed read lease here, immediately before graph
         * submission.  Keeping this at the common execution boundary covers
         * ordinary decode, resident logical-state correction, batched
         * device-outcome catch-up, and prefix-restore replay without asking
         * each caller to reproduce the ownership protocol.
         */
        std::optional<MTPTerminalHiddenPublication::ReadLease>
            terminal_hidden_read;
        if (terminal_hidden_buffer_id ==
            BufferId::PREFIX_TERMINAL_HIDDEN)
        {
            if (!state_.prefix_terminal_hidden ||
                terminal_hidden != state_.prefix_terminal_hidden.get())
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] MTP sidecar declared the "
                    "persistent terminal-hidden mailbox with a foreign "
                    "tensor binding");
                return false;
            }
            terminal_hidden_read =
                state_.mtp_terminal_hidden_publication.acquireReadLease();
            if (!terminal_hidden_read)
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] MTP sidecar could not lease a "
                    "current persistent terminal-hidden publication");
                return false;
            }
        }

        sidecar_cache.graph->reset();

        const bool has_sidecar_collectives = !sidecar_cache.collective_nodes.empty();

        bool ok = false;
        {
            ScopedStringOverride snapshot_scope(snapshot_context_, sidecar_context);
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "sidecar_execute_graph",
                phase,
                device_key,
                {{"context", sidecar_context},
                 {"depth", "0"},
                 {"device_tokens", boolTag(use_device_condition_tokens)},
                 {"kv_cache_only", boolTag(kv_cache_only)},
                 {"seq_len", std::to_string(token_count)}});

            bool used_graph_replay = false;
            const bool can_defer_sidecar_sync =
                defer_final_sync &&
                try_gpu_graph_capture;
            clearPendingLogitsStream(
                PendingLogitsStreamRole::MTPSidecar,
                "executeMTPDepth0_begin");
            if (try_gpu_graph_capture)
            {
                if (!sidecar_gpu_ctx)
                {
                    auto &pool = GPUDeviceContextPool::instance();
                    sidecar_gpu_ctx = &pool.getContext(state_.device_id);
                }
                if (sidecar_cache.capture_perf_context.empty())
                {
                    throw std::logic_error(
                        "MTP sidecar graph cache has no immutable capture identity");
                }
                sidecar_cache.segment_cache.perf_context =
                    sidecar_cache.capture_perf_context;
                auto capture_policy_result = buildDecodeCapturePolicyForGraph(
                    *sidecar_cache.graph,
                    has_sidecar_collectives,
                    ctx,
                    "ExpertOverlay MTP sidecar timeline");
                if (!capture_policy_result)
                    return false;
                auto capture_policy = std::move(*capture_policy_result);
                capture_policy.defer_final_sync = can_defer_sidecar_sync;
                if (overlay_graph_scope.ownsTicketAuthority())
                {
                    if (!capture_policy.allow_cached_graph_replay)
                    {
                        LOG_ERROR(
                            "[DeviceGraphOrchestrator] GPU ExpertOverlay MTP "
                            "sidecar requires retained graph execution; an "
                            "eager launch cannot provide the authenticated "
                            "pre-launch ticket edge on "
                            << state_.device_id.toString());
                        return false;
                    }
                    const auto existing =
                        std::move(capture_policy.launch_dependency);
                    capture_policy.launch_dependency =
                        [&overlay_graph_scope,
                         existing,
                         device = state_.device_id](
                            DeviceGraphExecutor::GraphExecutableLaunchPhase phase,
                            void *execution_stream) -> bool
                    {
                        if (existing && !existing(phase, execution_stream))
                            return false;
                        std::string arm_error;
                        if (!overlay_graph_scope.armForExecutableLaunch(
                                execution_stream, &arm_error))
                        {
                            LOG_ERROR(
                                "[DeviceGraphOrchestrator] MTP sidecar could "
                                "not arm its ExpertOverlay follower at the "
                                "executable launch edge on "
                                << device.toString() << ": " << arm_error);
                            return false;
                        }
                        return true;
                    };
                }
                if (capture_policy.allow_cached_graph_replay)
                {
                    /*
                     * MTP sidecar decode graphs use the same LocalTP capture
                     * lifecycle as the main decode graph.  Without this
                     * boundary, one TP participant can begin HIP/CUDA stream
                     * capture for a sidecar collective while a sibling is still
                     * draining or publishing the previous shifted-KV catch-up.
                     * ROCm reports that race as a capture-implicit stream
                     * dependency and poisons the RCCL graph-captured allreduce.
                     */
                    ForwardInput capture_boundary_input;
                    capture_boundary_input.device = state_.device_id;
                    const DeviceId capture_device = state_.device_id;
                    capture_policy.capture_boundary =
                        [this, capture_boundary_input, capture_device](
                            const std::string &boundary_name,
                            void *capture_stream) -> bool
                    {
                        return waitAtDecodeGraphCaptureBoundary(
                            capture_boundary_input,
                            capture_device,
                            boundary_name,
                            capture_stream);
                    };
                }
                const char *sidecar_recapture_context_filter =
                    DebugEnv::envValue("LLAMINAR_MTP_FORCE_SIDECAR_GRAPH_RECAPTURE_CONTEXT");
                const bool force_this_sidecar_context =
                    DebugEnv::isTruthyEnv("LLAMINAR_MTP_FORCE_SIDECAR_GRAPH_RECAPTURE") &&
                    (!sidecar_recapture_context_filter ||
                     sidecar_context == sidecar_recapture_context_filter);
                if (force_this_sidecar_context)
                {
                    /*
                     * Diagnostic-only recapture splitter: force just the MTP
                     * sidecar graph, or one named sidecar context, through
                     * recapture while leaving the target verifier graph on its
                     * normal replay path.  If acceptance recovers for only one
                     * context, that context owns stale persistent device inputs.
                     */
                    capture_policy.force_recapture = true;
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "sidecar_force_recapture_diagnostic",
                        1.0,
                        phase,
                        device_key,
                        {{"context", sidecar_context},
                         {"filter", sidecar_recapture_context_filter ? sidecar_recapture_context_filter : "<all>"},
                         {"seq_len", std::to_string(token_count)}});
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "sidecar_decode_capture_policy",
                    1.0,
                    phase,
                    device_key,
                    {{"context", sidecar_cache.segment_cache.perf_context},
                     {"invocation_context", sidecar_context},
                     {"seq_len", std::to_string(token_count)},
                     {"allow_graph_replay", boolTag(capture_policy.allow_cached_graph_replay)},
                     {"force_recapture", boolTag(capture_policy.force_recapture)},
                     {"defer_final_sync", boolTag(capture_policy.defer_final_sync)},
                     {"has_collectives", boolTag(has_sidecar_collectives)},
                     {"heterogeneous_segmented", boolTag(capture_policy.heterogeneous_segmented_enabled)},
                     {"collectives_graph_capturable", boolTag(capture_policy.collectives_graph_capturable)},
                     {"replay_plan_policy",
                      graphReplayPlanPolicyTag(
                          capture_policy.graph_replay_plan_policy)}});
                ok = executor_.executeDecodeWithCapturePolicy(
                    *sidecar_cache.graph,
                    ctx,
                    &sidecar_cache.segment_cache,
                    sidecar_dynamic_stream,
                    sidecar_gpu_ctx,
                    has_sidecar_collectives ? &sidecar_cache.collective_nodes : nullptr,
                    capture_policy,
                    &used_graph_replay);
                if (!kv_cache_only)
                {
                    void *sample_stream = nullptr;
                    std::string deferred_stream_error;
                    if (!mtp_sidecar::deferredSamplingStream(
                            ok,
                            capture_policy.defer_final_sync,
                            sidecar_cache.segment_cache.capture_stream,
                            &sample_stream,
                            &deferred_stream_error))
                    {
                        LOG_ERROR("[DeviceGraphOrchestrator] " << deferred_stream_error);
                        return false;
                    }
                    if (sample_stream)
                    {
                        // Full sidecar replay produces logits for the sampler.
                        // KV-only replay produces shifted MTP KV instead, so it
                        // must use the shifted-KV readiness event below rather
                        // than publishing an unrelated logits handoff.
                        publishPendingLogitsStream(
                            PendingLogitsStreamRole::MTPSidecar,
                            sample_stream,
                            "executeMTPDepth0_deferred_sampling_stream");
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "sidecar_forward_sample_sync_fusions",
                            1.0,
                            phase,
                            device_key,
                            {{"context", sidecar_context},
                             {"seq_len", std::to_string(token_count)}});
                    }
                }
            }
                else
                {
                    if (overlay_graph_scope.ownsTicketAuthority())
                    {
                        if (state_.device_id.is_gpu())
                        {
                            LOG_ERROR(
                                "[DeviceGraphOrchestrator] GPU ExpertOverlay "
                                "MTP sidecar cannot use uncaptured execution");
                            return false;
                        }
                        std::string arm_error;
                        if (!overlay_graph_scope.armForHostExecution(
                                &arm_error))
                        {
                            LOG_ERROR(
                                "[DeviceGraphOrchestrator] CPU MTP sidecar "
                                "could not arm its ExpertOverlay follower at "
                                "the host execution boundary: "
                                << arm_error);
                            return false;
                        }
                    }
                    // The common sidecar prelude already bound this exact
                    // workspace before dynamic parameters were published.
                    // Submit directly, as the captured GPU branch does. The
                    // generic execute() wrapper would scan/bind it again
                    // without the transaction's row geometry.
                    ok = executor_.execute(*sidecar_cache.graph, ctx);
                }

                if (ok &&
                    device_resident_logical_sequence_state_mailbox_.valid() &&
                    device_resident_logical_sequence_state_mailbox_
                            .live_state_epoch == live_replay_state_epoch_)
                {
                    snapshotDeviceResidentLogicalStatePhase(
                        DeviceResidentLogicalStateDiagnosticPhase::SidecarExit,
                        sidecar_dynamic_stream,
                        device_resident_logical_sequence_state_mailbox_
                            .request_count,
                        device_resident_logical_sequence_state_mailbox_
                            .publication_generation);
                }

                if (ok &&
                    mtpGraphReuseDiagnosticsEnabledForContext(sidecar_context))
                {
                    auto extension =
                        [&](BufferId id) -> const TensorBase *
                    {
                        const auto it = state_.extension_buffers.find(id);
                        return it == state_.extension_buffers.end()
                                   ? nullptr
                                   : it->second.get();
                    };
                    std::vector<MTPGraphReuseDiagnosticSource>
                        diagnostic_sources;
                    auto append_tensor =
                        [&](const char *name,
                            const TensorBase *tensor)
                    {
                        const MTPGraphReuseDiagnosticSource source{
                            name,
                            tensor && tensor->deviceValid()
                                ? tensor->gpu_data_ptr()
                                : nullptr,
                            mtpDiagnosticTensorRowBytes(tensor)};
                        /*
                         * Sidecar roles deliberately bind different subsets of
                         * the extension arena. An absent tensor is not a missing
                         * publication; it is outside this graph's contract.
                         * Observe every bound range, while the mandatory
                         * shifted-KV position below remains fail-closed.
                         */
                        if (source.device_data && source.byte_count > 0)
                            diagnostic_sources.push_back(source);
                    };
                    append_tensor(
                        "mtp_embedding",
                        extension(BufferId::MTP_EMBEDDING));
                    append_tensor(
                        "mtp_norm_hidden",
                        extension(BufferId::MTP_NORM_HIDDEN));
                    append_tensor(
                        "mtp_concat",
                        extension(BufferId::MTP_CONCAT));
                    append_tensor(
                        "mtp_projected",
                        extension(BufferId::MTP_PROJECTED));
                    append_tensor(
                        "mtp_q",
                        extension(BufferId::MTP_Q_PROJ));
                    append_tensor(
                        "mtp_k",
                        extension(BufferId::MTP_K_PROJ));
                    append_tensor(
                        "mtp_v",
                        extension(BufferId::MTP_V_PROJ));
                    append_tensor(
                        "mtp_attention_output",
                        extension(BufferId::MTP_ATTN_OUTPUT));
                    append_tensor(
                        "mtp_hidden",
                        extension(BufferId::MTP_HIDDEN));
                    append_tensor(
                        "mtp_logits",
                        extension(BufferId::MTP_LOGITS));
                    const void *mtp_live_position =
                        !state_.mtp_kv_caches.empty() &&
                                state_.mtp_kv_caches[0]
                            ? state_.mtp_kv_caches[0]
                                  ->deviceSequenceCachedTokenCountPtr(first_seq_idx)
                            : nullptr;
                    diagnostic_sources.push_back({
                        "mtp_live_position",
                        mtp_live_position,
                        sizeof(int32_t),
                    });
                    IBackend *backend = getBackendFor(state_.device_id);
                    if (!backend ||
                        !logMTPGraphReuseBoundaryDiagnostics(
                            "after_execute",
                            sidecar_context,
                            session_epoch_,
                            rebuilt_graph,
                            used_graph_replay,
                            *backend,
                            state_.device_id,
                            sidecar_dynamic_stream,
                            diagnostic_sources))
                    {
                        return false;
                    }
                }

                PerfStatsCollector::addCounter(
                    "mtp",
                "sidecar_graph_capture_path",
                1.0,
                phase,
                device_key,
                {{"context", sidecar_context},
                 {"graph_context", sidecar_cache.capture_perf_context},
                 {"depth", "0"},
                 {"device_tokens", boolTag(use_device_condition_tokens)},
                 {"kv_cache_only", boolTag(kv_cache_only)},
                 {"seq_len", std::to_string(token_count)},
                 {"path", used_graph_replay
                              ? DeviceGraphCaptureController::replayModeName(sidecar_cache.segment_cache)
                              : "plain"}});
        }
        if (ok && terminal_hidden_read)
        {
            /*
             * Graph execution above has submitted every read on the exact
             * sidecar stream.  A changed generation at this point means a
             * producer crossed that submission window, so continuing would
             * make the captured pointer's contents ambiguous.
             */
            if (!state_.mtp_terminal_hidden_publication.stillOwns(
                    *terminal_hidden_read))
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] MTP sidecar crossed a "
                    "terminal-hidden publication transition while submitting "
                    "its read-only graph");
                return false;
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "sidecar_terminal_hidden_read_leases",
                1.0,
                phase,
                device_key,
                {{"context", sidecar_context},
                 {"source",
                  MTPTerminalHiddenPublication::sourceName(
                      terminal_hidden_read->source)},
                 {"kv_cache_only", boolTag(kv_cache_only)},
                 {"rows", std::to_string(total_rows)}});
        }
        if (ok && state_.device_id.is_gpu() && sidecar_dynamic_stream &&
            !peekPendingLogitsStream(PendingLogitsStreamRole::MTPSidecar))
        {
            /*
             * KV-only sidecars have no host-visible result. When the caller
             * requests deferred completion, publish the persistent shifted-KV
             * event for both first-use execution and captured replay. A graph
             * cache miss is not a reason to block the host because every later
             * cache reader already consumes this event on its exact stream.
             */
            if (kv_cache_only && defer_final_sync)
            {
                if (!recordShiftedMTPKVReady(
                        sidecar_dynamic_stream,
                        sidecar_context.c_str()))
                {
                    return false;
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "shifted_mtp_kv_stream_syncs_deferred",
                    1.0,
                    phase,
                    device_key,
                    {{"context", sidecar_context},
                     {"seq_len", std::to_string(token_count)},
                     {"path", "capture"}});
            }
            else
            {
                LOG_ERROR("[DeviceGraphOrchestrator] GPU MTP sidecar completed without publishing "
                          "its required logits or shifted-KV event handoff"
                          << " context=" << sidecar_context
                          << " kv_cache_only=" << kv_cache_only
                          << " path=capture");
                return false;
            }
        }
        if (ok && state_.device_id.is_gpu() && sidecar_dynamic_stream)
        {
            /*
             * The sidecar has mutated canonical shifted-cache metadata on its
             * explicit stream. Advance the shared request transaction fence in
             * place. Every previously issued child-local lease observes this
             * new event and generation; no live-state epoch retarget or copied
             * count row is needed.
             */
            if (!recordDeviceResidentMTPTransactionMutation(
                    request_batch,
                    sidecar_dynamic_stream,
                    sidecar_context.c_str()))
            {
                return false;
            }
        }
        if (ok && requires_mtp_logits_gather)
        {
            /*
             * CPU GlobalTP keeps the large MTP projection column-sharded and
             * gathers only its compact verifier-row logits.  Publish this
             * counter after successful graph execution so integration tests
             * prove the terminal AllGatherStage actually ran; graph discovery
             * alone is insufficient because MoE sidecars contain unrelated
             * collectives as well.
             */
            PerfStatsCollector::addCounter(
                "mtp",
                graph_builder_->config().mtpPublishesNativeGatheredVocabulary()
                    ? "native_local_sidecar_logit_allgathers" : "global_tp_sidecar_logit_allgathers",
                1.0,
                phase,
                device_key,
                {{"context", sidecar_context},
                 {"rows", std::to_string(total_rows)},
                 {"local_vocab", std::to_string(mtp_logits->cols())},
                 {"gathered_vocab", std::to_string(mtp_logits_gathered->cols())},
                 {"implementation", "graph_allgather"}});
        }
        if (ok && state_.device_id.is_gpu() &&
            PerfStatsCollector::gpuStageEventTimingEnabled())
        {
            auto &timeline = executor_.stageTimeline();
            if (timeline.isInitialized())
            {
                // MTP sidecar graph capture/replay can leave a timeline initialized
                // without fresh events. Clear it here so stale warmup events are never queried.
                if (suppress_timeline_ || !timeline.hasValidRecords())
                {
                    timeline.resetTimings();
                    if (!finish_overlay_graph(ok))
                        return false;
                    return ok;
                }

                auto &pool = GPUDeviceContextPool::instance();
                auto &gpu_ctx = pool.getContext(state_.device_id);
                timeline.collect(&gpu_ctx);
                const std::string mtp_phase = "mtp_" + phase;
                timeline.recordPerfStats(
                    mtp_phase.c_str(),
                    device_key.c_str(),
                    "mtp_stage_gpu",
                    {{"context", sidecar_context},
                     {"depth", "0"}});
                timeline.resetTimings();
            }
        }
        if (!finish_overlay_graph(ok))
            return false;
        return ok;
    }

    bool DeviceGraphOrchestrator::bindShiftedMTPPrefillTransaction(
        ForwardInput &input,
        int request_count,
        ShiftedMTPPrefillGraphBinding::Purpose purpose)
    {
        input.shifted_mtp_prefill.reset();
        // The pipeline terminal alone owns the learned predictor. A follower
        // still executes its main-model rows, including a restored-prefix
        // condition, but has no shifted-cache transaction to bind.
        if (pp_stage_config_ && !pp_stage_config_->has_lm_head)
            return true;

        const bool declaration_only =
            purpose == ShiftedMTPPrefillGraphBinding::Purpose::
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
        /*
         * Logical execution-off and physical graph-capacity-off are different
         * states. A declaration-only call must expose the exact shifted-KV
         * stages retained by a later MTP-enabled runner, otherwise the first
         * process-campaign cell seals a reusable block that is smaller than
         * the model's already-admitted graph family. The purpose tag keeps
         * this topology out of ForwardExecutionEngine, so declaring it cannot
         * accidentally execute MTP for an MTPOff request.
         */
        const bool main_prefill =
            input.execution_phase == ForwardExecutionPhase::Prefill &&
            input.state_transaction == ForwardStateTransaction::Ordinary;
        const bool restored_prefix_decode_bridge =
            input.execution_phase == ForwardExecutionPhase::Decode &&
            input.state_transaction ==
                ForwardStateTransaction::RestoredPrefixMTPDecodeBridge;
        if (!state_.device_id.is_gpu() ||
            (input.execution_role != ForwardExecutionRole::MainInference &&
             !(restored_prefix_decode_bridge &&
               input.execution_role ==
                   ForwardExecutionRole::MTPCondition)) ||
            (!main_prefill && !restored_prefix_decode_bridge))
        {
            return true;
        }

        if (graph_builder_->config()
                .mtp_shifted_prefill_hidden_publication !=
            MTPShiftedPrefillHiddenPublicationPolicy::
                GraphIntegratedKVTransaction)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] GPU MTP prefill requires the graph-integrated shifted-KV transaction policy");
            return false;
        }
        const int32_t *request_row_stride_device =
            input.device_prefill_chunk
                ? input.device_prefill_chunk->chunk_row_stride_device
                : request_row_stride_dev_;
        const int32_t *request_token_ids_device =
            restored_prefix_decode_bridge
                ? static_cast<const int32_t *>(request_token_ids_dev_)
                : static_cast<const int32_t *>(input.token_ids_device);
        const int32_t *request_position_ids_device =
            restored_prefix_decode_bridge
                ? static_cast<const int32_t *>(request_position_ids_dev_)
                : static_cast<const int32_t *>(input.position_ids_device);
        const int32_t *request_segment_lengths_device =
            restored_prefix_decode_bridge
                ? static_cast<const int32_t *>(
                      request_sequence_lengths_dev_)
                : input.device_prefill_chunk
                ? input.device_prefill_chunk->chunk_real_rows_device
                : (input.token_ids_device == request_token_ids_dev_
                       ? static_cast<const int32_t *>(
                             request_sequence_lengths_dev_)
                       : input.sequence_lengths_device);
        const size_t captured_input_rows =
            request_count > 0
                ? static_cast<size_t>(request_count) *
                      static_cast<size_t>(std::max(
                          0,
                          input.device_prefill_chunk
                              ? input.device_prefill_chunk->bucket_seq_len
                              : input.seq_len))
                : size_t{0};
        if (request_count <= 0 ||
            request_count > request_sequence_lengths_capacity_ ||
            !request_token_ids_device || !request_position_ids_device ||
            !request_segment_lengths_device || !request_row_stride_device ||
            !mtp_shifted_prefill_token_ids_dev_ ||
            !mtp_shifted_prefill_position_ids_dev_ ||
            !mtp_shifted_prefill_append_lengths_dev_ ||
            captured_input_rows <= 0 ||
            captured_input_rows >
                static_cast<size_t>(
                    std::max(0, mtp_shifted_prefill_row_capacity_)))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] GPU MTP prefill cannot bind incomplete persistent request geometry"
                      << " requests=" << request_count
                      << " length_capacity="
                      << request_sequence_lengths_capacity_
                      << " tokens=" << request_token_ids_device
                      << " positions=" << request_position_ids_device
                      << " segment_lengths="
                      << request_segment_lengths_device
                      << " row_stride=" << request_row_stride_device
                      << " captured_rows=" << captured_input_rows
                      << " shifted_capacity="
                      << mtp_shifted_prefill_row_capacity_);
            return false;
        }
        if (!state_.kv_cache || state_.mtp_kv_caches.empty() ||
            !state_.mtp_kv_caches[0])
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Enabled GPU MTP prefill requires both main and depth-zero shifted KV caches");
            return false;
        }
        if (!ensureMTPTerminalHiddenBuffer(request_count) ||
            !state_.prefix_terminal_hidden ||
            !state_.prefix_terminal_hidden->gpu_data_ptr())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] GPU MTP prefill requires a device-resident request terminal archive");
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
            .gathered_logits = nullptr,
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
        const std::array<std::pair<const char *, TensorBase *>, 9>
            required_kv_only_scratch{{
                {"embedding", output.embedding},
                {"norm_hidden", output.norm_hidden},
                {"norm_embedding", output.norm_embedding},
                {"concat", output.concat},
                {"projected", output.projected},
                {"q", output.q},
                {"k", output.k},
                {"v", output.v},
                {"q_raw", output.q_raw},
            }};
        for (const auto &[name, tensor] : required_kv_only_scratch)
        {
            if (!tensor || !tensor->gpu_data_ptr())
            {
                LOG_ERROR("[DeviceGraphOrchestrator] GPU MTP prefill is missing device-resident KV-only scratch '"
                          << name << "'");
                return false;
            }
        }
        if (!output.q_gate || !output.q_gate->gpu_data_ptr())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] GPU MTP prefill is missing device-resident KV-only scratch 'q_gate'");
            return false;
        }

        ShiftedMTPPrefillGraphBinding binding{
            .kv_cache = state_.mtp_kv_caches[0].get(),
            .terminal_hidden_archive =
                state_.prefix_terminal_hidden.get(),
            .request_token_ids_device = request_token_ids_device,
            .request_position_ids_device = request_position_ids_device,
            .request_segment_lengths_device =
                request_segment_lengths_device,
            .shifted_token_ids_device = static_cast<int32_t *>(
                mtp_shifted_prefill_token_ids_dev_),
            .shifted_position_ids_device = static_cast<int32_t *>(
                mtp_shifted_prefill_position_ids_dev_),
            .append_lengths_device = static_cast<int32_t *>(
                mtp_shifted_prefill_append_lengths_dev_),
            .request_row_stride_device = request_row_stride_device,
            .output = output,
        };
        binding.purpose = purpose;
        binding.main_cached_tokens_device.reserve(
            static_cast<size_t>(request_count));
        binding.shifted_cached_tokens_device.reserve(
            static_cast<size_t>(request_count));
        for (int request = 0; request < request_count; ++request)
        {
            const int32_t *main_count =
                state_.kv_cache->deviceSequenceCachedTokenCountPtr(request);
            const int32_t *shifted_count = state_.mtp_kv_caches[0]
                                                ->deviceSequenceCachedTokenCountPtr(
                                                    request);
            if (!main_count || !shifted_count)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] GPU MTP prefill cannot bind canonical KV progress for request "
                          << request);
                return false;
            }
            binding.main_cached_tokens_device.push_back(main_count);
            binding.shifted_cached_tokens_device.push_back(shifted_count);
        }

        const uint64_t generation = workspaceGeneration(state_.device_id);
        const bool workspace_family_declaration = declaration_only;
        if (!workspace_family_declaration && generation == 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] GPU MTP prefill requires a finalized workspace generation");
            return false;
        }
        if (workspace_family_declaration)
        {
            /*
             * The setup graph needs a non-zero topology discriminator, but it
             * must not pretend that generation zero is executable capture
             * identity. Purpose remains declaration-only and the execution
             * engine rejects it before cache lookup or launch.
             */
            binding.capture_identity = UINT64_MAX;
        }
        else
        {
            uint64_t identity = UINT64_C(1469598103934665603);
            auto mix_u64 = [&](uint64_t value)
            {
                identity ^= value;
                identity *= UINT64_C(1099511628211);
            };
            auto mix_pointer = [&](const void *pointer)
            {
                mix_u64(static_cast<uint64_t>(
                    reinterpret_cast<uintptr_t>(pointer)));
            };
            mix_u64(generation);
            mix_u64(static_cast<uint64_t>(request_count));
            mix_u64(input.device_prefill_chunk
                        ? input.device_prefill_chunk->capture_identity
                        : uint64_t{0});
            mix_pointer(input.token_ids_device);
            mix_pointer(input.position_ids_device);
            mix_pointer(input.sequence_lengths_device);
            mix_pointer(binding.kv_cache);
            mix_pointer(state_.kv_cache.get());
            mix_pointer(binding.terminal_hidden_archive);
            mix_pointer(binding.terminal_hidden_archive->gpu_data_ptr());
            mix_pointer(binding.request_token_ids_device);
            mix_pointer(binding.request_position_ids_device);
            mix_pointer(binding.request_segment_lengths_device);
            mix_pointer(binding.shifted_token_ids_device);
            mix_pointer(binding.shifted_position_ids_device);
            mix_pointer(binding.append_lengths_device);
            mix_pointer(binding.request_row_stride_device);
            for (const int32_t *pointer : binding.main_cached_tokens_device)
                mix_pointer(pointer);
            for (const int32_t *pointer : binding.shifted_cached_tokens_device)
                mix_pointer(pointer);
            const std::array<TensorBase *, 10> captured_scratch{{
                output.embedding,
                output.norm_hidden,
                output.norm_embedding,
                output.concat,
                output.projected,
                output.q,
                output.k,
                output.v,
                output.q_raw,
                output.q_gate,
            }};
            for (TensorBase *tensor : captured_scratch)
            {
                mix_pointer(tensor);
                mix_pointer(tensor ? tensor->gpu_data_ptr() : nullptr);
            }
            binding.capture_identity =
                identity == 0 ? UINT64_MAX : identity;
        }
        if (!binding.validForRequestCount(request_count))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] GPU MTP prefill produced an incomplete graph binding");
            return false;
        }

        input.shifted_mtp_prefill = std::move(binding);
        return true;
    }
} // namespace llaminar2
