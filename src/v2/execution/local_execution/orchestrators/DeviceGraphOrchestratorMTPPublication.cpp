/**
 * @file DeviceGraphOrchestratorMTPPublication.cpp
 * @brief Participant-local MTP publication setup and native capture ownership.
 *
 * The terminal owns compact outcomes, response budgets and predictor state.
 * Pipeline followers bind only received committed metadata and their own main
 * model state. Both roles share checkpoint validation, histogram/recurrent
 * publication and the same event-ordered captured executable cache. Native
 * transport remains an explicit graph node owned by the rank's frozen cohort.
 */
#include "DeviceGraphOrchestrator.h"
#include "PipelineForwardGraphEdges.h"
#include "backends/BackendManager.h"
#include "execution/compute_stages/stages/MoERoutingStage.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/moe/MoEGroupedVerifierHistogramBoundarySet.h"
#include "utils/DebugEnv.h"
#include "utils/Logger.h"

#include <cstddef>

namespace llaminar2
{
    bool DeviceGraphOrchestrator::materializeMTPSpeculativeStatePublicationGraph(
        const DeviceSpeculativePublicationRequest &request,
        const MTPSpecDecodeMetadataDevicePointers &publication_metadata,
        ComputeGraph &verifier_graph,
        int verifier_rows_per_request,
        std::string *error)
    {
        auto fail = [&](const std::string &reason) -> bool
        {
            if (error)
                *error = reason;
            LOG_ERROR("[DeviceGraphOrchestrator] " << reason);
            return false;
        };

        const uint64_t generation = workspaceGeneration(state_.device_id);
        if (!state_.device_id.is_gpu() || generation == 0 ||
            request.requestCount() <= 0 ||
            verifier_rows_per_request <= 0)
        {
            return fail(
                "MTP speculative publication graph requires finalized GPU bindings and positive geometry");
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return fail("MTP speculative publication graph could not resolve its GPU backend");
        const int full_condition_slot =
            mtp_sidecar_capture_layout_.conditionTokenSlot(
                MTPSidecarCaptureRole::Full,
                request.requestCount());
        const int full_condition_offset =
            full_condition_slot *
            mtp_sidecar_condition_token_slot_width_;
        if (!mtp_sidecar_condition_token_dev_ ||
            !mtp_sidecar_position_ids_dev_ ||
            !stochastic_target_sample_tokens_dev_ ||
            request.requestCount() >
                mtp_sidecar_condition_token_slot_width_ ||
            static_cast<size_t>(request.requestCount()) *
                    static_cast<size_t>(verifier_rows_per_request) >
                static_cast<size_t>(stochastic_target_row_capacity_) ||
            full_condition_offset < 0 ||
            full_condition_offset + request.requestCount() >
                mtp_sidecar_condition_token_capacity_)
        {
            return fail(
                "MTP speculative publication graph has no persistent next-transaction sidecar mailbox");
        }
        if (!state_.kv_cache ||
            static_cast<int>(
                mtp_publication_main_kv_base_checkpoints_.size()) <
                request.requestCount())
        {
            return fail(
                "MTP speculative publication graph has incomplete primary KV checkpoint ownership");
        }

        MTPSpeculativeStatePublicationStage::Params params;
        params.device_id = state_.device_id;
        params.backend = backend;
        params.outcome_tokens_device =
            request.outcome.output_tokens_device;
        params.outcome_meta_device = request.outcome.meta_device;
        params.outcome_token_stride =
            request.outcome.output_token_stride;
        params.outcome_meta_stride = request.outcome.meta_stride;
        params.base_cached_tokens_device =
            publication_metadata.base_cached_tokens;
        params.accepted_restore_rows_device =
            publication_metadata.accepted_state_slot_indices;
        params.target_cached_tokens_device =
            publication_metadata.target_cached_tokens;
        params.accepted_state_counts_device =
            publication_metadata.accepted_state_counts;
        params.publication_ok_flags_device =
            publication_metadata.publication_ok_flags;
        params.next_condition_tokens_device =
            publication_metadata.next_condition_tokens;
        params.all_drafts_accepted_flags_device =
            publication_metadata.all_drafts_accepted_flags;
        params.stopped_flags_device = publication_metadata.stopped_flags;
        params.next_verifier_condition_tokens_device =
            static_cast<int32_t *>(stochastic_target_sample_tokens_dev_);
        params.next_sidecar_condition_tokens_device =
            static_cast<int32_t *>(mtp_sidecar_condition_token_dev_) +
            full_condition_offset;
        params.next_sidecar_position_ids_device =
            static_cast<int32_t *>(mtp_sidecar_position_ids_dev_);

        params.authority = request.outcome.device_generation_controller_owned
            ? MTPSpeculativeStatePublicationStage::Authority::BoundedGeneration
            : MTPSpeculativeStatePublicationStage::Authority::CompactOutcome;
        params.generation_response_tokens_device =
            device_generation_storage_.response_tokens_device;
        params.generation_response_token_stride =
            device_generation_storage_.response_token_stride;
        params.generation_control_device =
            device_generation_storage_.control_device;
        params.generation_control_stride =
            device_generation_storage_.control_stride;
        params.verifier_input_tokens_device =
            static_cast<const int32_t *>(
                mtp_verifier_input_tokens_dev_);
        params.verifier_input_token_stride =
            verifier_rows_per_request;
        params.committed_verifier_identity_device =
            mtp_committed_verifier_identity_dev_;

        /*
         * Greedy compact outcomes leave their accepted count in the verifier
         * graph; publication is therefore the sole point at which the Dynamic
         * ExpertOverlay cadence can observe a fully committed transaction.
         * Stochastic reducers already advance this exact device boundary in
         * their fused outcome graph, so binding them here would be a double
         * transition. The producer-owned typed sampling mode makes that split
         * explicit rather than inferring it from a mutable runner flag.
         */
        const bool owns_greedy_dynamic_moe_boundary =
            request.outcome.device_generation_controller_owned &&
            request.outcome.sampling_mode ==
                DeviceGenerationSamplingMode::Greedy &&
            usesParticipantLocalDeviceMoERebalanceController();
        if (owns_greedy_dynamic_moe_boundary)
        {
            std::string controller_error;
            DeviceMoERebalanceGraphControllerState *const controller =
                deviceMoERebalanceControllerStateDevice(&controller_error);
            if (!controller)
            {
                return fail(
                    "MTP speculative publication cannot bind the Dynamic MoE cadence controller: " +
                    controller_error);
            }
            auto *const controller_bytes =
                reinterpret_cast<std::byte *>(controller);
            params.dynamic_moe_commit_boundary =
                MTPSpeculativeStatePublicationStage::
                    DynamicMoECommitBoundaryBinding{
                        .decode_rounds_committed_device =
                            reinterpret_cast<uint32_t *>(
                                controller_bytes +
                                offsetof(
                                    DeviceMoERebalanceGraphControllerState,
                                    decode_rounds_committed)),
                        .decode_rounds_until_maintenance_device =
                            reinterpret_cast<uint32_t *>(
                                controller_bytes +
                                offsetof(
                                    DeviceMoERebalanceGraphControllerState,
                                    decode_rounds_until_maintenance)),
                        .maintenance_due_device =
                            reinterpret_cast<uint32_t *>(
                                controller_bytes +
                                offsetof(
                                    DeviceMoERebalanceGraphControllerState,
                                    maintenance_due)),
                        .decode_boundary_advanced_device =
                            reinterpret_cast<uint32_t *>(
                                controller_bytes +
                                offsetof(
                                    DeviceMoERebalanceGraphControllerState,
                                    decode_boundary_advanced)),
                    };
        }

        params.request_count = request.requestCount();
        params.verifier_rows_per_request = verifier_rows_per_request;
        params.max_state_commit_rows = request.max_state_commit_rows;

        params.publish_shifted_kv = request.publish_mtp_shifted_kv;
        params.shifted_target_cached_tokens_device =
            publication_metadata.shifted_target_cached_tokens;
        params.shifted_accepted_state_counts_device =
            publication_metadata.shifted_accepted_state_counts;
        if (request.publish_mtp_shifted_kv)
        {
            params.shifted_kv_caches.reserve(state_.mtp_kv_caches.size());
            for (size_t depth = 0;
                 depth < state_.mtp_kv_caches.size();
                 ++depth)
            {
                IKVCache *cache = state_.mtp_kv_caches[depth].get();
                if (!cache ||
                    !cache->supportsDeviceResidentSequenceStatePublication() ||
                    !cache->deviceSequenceCachedTokenCountPtr(0))
                {
                    return fail(
                        "MTP speculative publication graph has no complete shifted-KV device owner at depth " +
                        std::to_string(depth));
                }
                params.shifted_kv_caches.push_back(cache);
            }
        }

        params.penalty_policy_device = mtp_greedy_penalty_policy_dev_;
        params.generated_token_counts_device =
            static_cast<int32_t *>(mtp_generated_token_counts_dev_);
        params.vocab_size = state_.vocab_size;
        params.require_captured_verifier_state =
            mtpSpecStatePublicationRequiresCapturedStage();

        return installMTPStatePublicationGraph(std::move(params), verifier_graph, error);
    }

    bool DeviceGraphOrchestrator::installMTPStatePublicationGraph(
        MTPSpeculativeStatePublicationStage::Params params,
        ComputeGraph &verifier_graph, std::string *error)
    {
        const auto fail = [&](const std::string &reason) {
            if (error) *error = reason;
            LOG_ERROR("[DeviceGraphOrchestrator] " << reason);
            return false;
        };
        const auto generation = workspaceGeneration(state_.device_id);
        const int request_count = params.request_count;
        const int verifier_rows_per_request = params.verifier_rows_per_request;
        if (!state_.device_id.is_gpu() || generation == 0 || params.device_id != state_.device_id ||
            request_count <= 0 || verifier_rows_per_request <= 0 || !state_.kv_cache ||
            !params.main_kv_bindings.empty() ||
            mtp_publication_main_kv_base_checkpoints_.size() < static_cast<size_t>(request_count))
            return fail("MTP publication requires finalized local workspace and canonical checkpoint ownership");

        // Rollback storage is always contributed by the participant's state owner,
        // never by an outcome handle or another pipeline stage.
        params.main_kv_bindings.reserve(
            static_cast<size_t>(request_count));
        for (int request_index = 0;
             request_index < request_count;
             ++request_index)
        {
            const DeviceKVSequenceStateCheckpoint &checkpoint =
                mtp_publication_main_kv_base_checkpoints_[
                    static_cast<size_t>(request_index)];
            if (!checkpoint.valid() ||
                checkpoint.device != state_.device_id ||
                checkpoint.sequence_index != request_index)
            {
                return fail(
                    "MTP speculative publication graph received a stale or foreign main-KV checkpoint for request " +
                    std::to_string(request_index));
            }
            params.main_kv_bindings.push_back({
                .cache = state_.kv_cache.get(),
                .first_sequence_index = request_index,
                .base_checkpoint_device = checkpoint.data(),
                .base_checkpoint_bytes = checkpoint.bytes,
            });
        }


        MoEGroupedVerifierHistogramBoundarySet histogram_boundaries;
        std::string histogram_boundary_error;

        for (const std::string &node_name :
             verifier_graph.getExecutionOrder())
        {
            ComputeNode *node = verifier_graph.getNode(node_name);
            if (!node || !node->stage)
            {
                return fail(
                    "MTP speculative publication graph encountered an incomplete verifier node: " +
                    node_name);
            }
            IComputeStage *stage = node->stage.get();
            params.verifier_state_stages.push_back(stage);
            if (stage->type() == ComputeStageType::MOE_ROUTER)
            {
                auto *router = dynamic_cast<MoERoutingStage *>(stage);
                if (!router || router->layerIndex() < 0)
                {
                    return fail(
                        "MTP speculative publication graph found an invalid MoE router: " +
                        node_name);
                }
                if (!histogram_boundaries.registerRoutedLayer(
                        router->layerIndex(),
                        node_name,
                        &histogram_boundary_error))
                {
                    return fail(
                        "MTP speculative publication graph rejected a routed MoE layer: " +
                        histogram_boundary_error);
                }

                if (!histogram_boundaries.registerStage(
                        router,
                        node_name,
                        &histogram_boundary_error))
                {
                    return fail(
                        "MTP speculative publication graph rejected the overlay-router history boundary: " +
                        histogram_boundary_error);
                }
            }
            if (stage->type() == ComputeStageType::MOE_EXPERT_FFN)
            {
                auto *moe_stage =
                    dynamic_cast<MoEExpertComputeStage *>(stage);
                if (!moe_stage ||
                    !histogram_boundaries.registerStage(
                        moe_stage,
                        node_name,
                        &histogram_boundary_error))
                {
                    return fail(
                        "MTP speculative publication graph rejected an MoE verifier history boundary: " +
                        histogram_boundary_error);
                }
            }
        }

        MoEGroupedVerifierHistogramBoundaryResolution
            histogram_boundary_resolution;
        if (!histogram_boundaries.resolve(
                histogram_boundary_resolution,
                &histogram_boundary_error))
        {
            return fail(
                "MTP speculative publication graph has an invalid verifier-history lifecycle: " +
                histogram_boundary_error);
        }
        params.moe_histogram_publishers =
            std::move(histogram_boundary_resolution.publishers);
        params.moe_histogram_publication_stream =
            histogram_boundary_resolution.publication_stream;

        auto &cache = mtp_speculative_state_publication_graph_;
        if (cache.valid && cache.graph && cache.stage &&
            cache.workspace_generation == generation &&
            cache.pipeline_edges == pipeline_forward_edges_ &&
            cache.stage->hasSameCaptureIdentity(params))
        {
            return true;
        }

        auto graph = std::make_unique<ComputeGraph>();
        MTPSpeculativeStatePublicationStage *publication_stage = nullptr;
        try
        {
            if (pipeline_forward_edges_)
                pipeline_forward_edges_->validatePublication(params);
            auto stage = std::make_unique<MTPSpeculativeStatePublicationStage>(std::move(params));
            if (!stage->validate())
                return fail("MTP publication has incomplete persistent state bindings");
            publication_stage = stage.get();
            graph->addNode("mtp_speculative_state_publication", std::move(stage), state_.device_id);
            // A pipeline's local graph is a source child, not the complete
            // transaction. The mandatory topology composer adds its retained
            // native exchange; standalone publication is already complete.
            graph->setNativeCaptureEnvelope(pipeline_forward_edges_ ? GraphNativeCaptureEnvelope::Ordinary
                : GraphNativeCaptureEnvelope::DeviceOwnedTimelineTransaction);
        }
        catch (const std::exception &exception)
        {
            return fail("MTP publication graph rejected ownership: " + std::string(exception.what()));
        }

        mtp_device_generation_loop_graph_.invalidateGraph();
        /* All publication identities are sequential and share one exact MoE
         * histogram producer.  Retain its certified stream across geometry or
         * pointer-identity replacement; only final topology teardown destroys
         * that stream. */
        cache.replaceIdentityPreservingProducerStream();
        cache.graph = std::move(graph);
        cache.stage = publication_stage;
        cache.workspace_generation = generation;
        cache.pipeline_edges = pipeline_forward_edges_;
        cache.valid = true;

        PerfStatsCollector::addCounter(
            "mtp",
            "speculative_state_publication_graph_materializations",
            1.0,
            "graph_setup",
            state_.device_id.toString(),
            {{"requests", std::to_string(request_count)},
             {"verifier_rows",
              std::to_string(verifier_rows_per_request)},
             {"workspace_generation", std::to_string(generation)}});
        return true;
    }

    bool DeviceGraphOrchestrator::materializePipelineFollowerMTPPublicationGraph(
        ComputeGraph &verifier_graph, int verifier_rows_per_request, std::string *error)
    {
        const auto fail = [&](const char *reason) {
            if (error) *error = reason;
            LOG_ERROR("[DeviceGraphOrchestrator] " << reason);
            return false;
        };
        if (!state_.device_id.is_gpu() || !pp_stage_config_ || pp_stage_config_->has_lm_head ||
            !pipeline_forward_edges_ || verifier_rows_per_request < 2 ||
            verifier_rows_per_request > mtp_max_verifier_rows_ ||
            !mtp_spec_decode_metadata_binding_.hasWorkspace() ||
            !device_resident_logical_sequence_state_storage_.validFor(1))
            return fail("Pipeline follower publication requires local resident metadata and frozen native ownership");
        auto metadata = mtp_spec_decode_metadata_binding_.devicePointers();
        if (!device_resident_logical_sequence_state_storage_.bindPublicationOutputs(&metadata, 1))
            return fail("Pipeline follower publication cannot bind its durable local outputs");
        MTPSpeculativeStatePublicationStage::Params params;
        params.device_id = state_.device_id;
        params.backend = getBackendFor(state_.device_id);
        params.authority = MTPSpeculativeStatePublicationStage::Authority::PipelineFollower;
        params.request_count = 1;
        params.verifier_rows_per_request = verifier_rows_per_request;
        params.accepted_restore_rows_device = metadata.accepted_state_slot_indices;
        params.target_cached_tokens_device = metadata.target_cached_tokens;
        params.accepted_state_counts_device = metadata.accepted_state_counts;
        params.publication_ok_flags_device = metadata.publication_ok_flags;
        params.require_captured_verifier_state = mtpSpecStatePublicationRequiresCapturedStage();
        return installMTPStatePublicationGraph(std::move(params), verifier_graph, error);
    }

    bool DeviceGraphOrchestrator::executeMTPStatePublicationGraph(
        void *producer_stream,
        std::string *error,
        DeviceGraphExecutor::GraphInitialSubmissionPolicy submission)
    {
        auto fail = [&](const std::string &reason) -> bool
        {
            if (error)
                *error = reason;
            LOG_ERROR("[DeviceGraphOrchestrator] " << reason);
            return false;
        };

        auto &cache = mtp_speculative_state_publication_graph_;
        const uint64_t generation = workspaceGeneration(state_.device_id);
        if (!state_.device_id.is_gpu() || !producer_stream ||
            !cache.valid || !cache.graph || !cache.stage ||
            cache.workspace_generation == 0 ||
            cache.workspace_generation != generation ||
            cache.pipeline_edges != pipeline_forward_edges_)
        {
            return fail(
                "captured MTP state publication requires a current graph and the exact outcome stream");
        }
        if (!debugEnv().execution.gpu_graphs)
        {
            return fail(
                "GPU MTP state publication requires graph capture; eager publication is forbidden");
        }

        // The communicator records this immutable exchange once at serving
        // setup. Local policies (for example clipping the tail's final commit)
        // may change independently without making peers re-enter capture.
        const auto *transport = cache.pipeline_edges
            ? cache.pipeline_edges->publicationTransport(
                PipelineForwardGraphEdges::PublicationBanks::from(cache.stage->getParams()))
            : nullptr;
        if (cache.pipeline_edges && !transport)
            return fail("Pipeline MTP publication requires its frozen serving-time transport capture");
        if (submission == DeviceGraphExecutor::GraphInitialSubmissionPolicy::MaterializeWithoutLaunch &&
            cache.segment_cache.initialized)
            return cache.segment_cache.deviceLoopGraphTemplate(*cache.graph, error).has_value();

        IDeviceContext *ctx = getDeviceContext(state_.device_id);
        if (!ctx)
            return fail("captured MTP state publication could not resolve its device context");
        auto &gpu_ctx =
            GPUDeviceContextPool::instance().getContext(state_.device_id);
        void *const histogram_publication_stream =
            cache.stage->getParams().moe_histogram_publication_stream;
        const bool capture_stream_ready =
            histogram_publication_stream
                ? cache.segment_cache.bindBorrowedCaptureStream(
                      &gpu_ctx,
                      histogram_publication_stream,
                      state_.device_id,
                      /*context_from_process_pool=*/true)
                : cache.segment_cache.ensureCaptureStream(
                      &gpu_ctx,
                      state_.device_id,
                      /*context_from_process_pool=*/true);
        if (!capture_stream_ready ||
            !cache.segment_cache.orderCaptureStreamAfter(
                &gpu_ctx,
                producer_stream))
        {
            return fail(
                "captured MTP state publication could not consume compact outcome readiness");
        }

        cache.segment_cache.perf_context =
            "mtp_speculative_state_publication";
        cache.segment_cache.replay_workload = {
            .seq_len = cache.stage->getParams().verifier_rows_per_request,
            .batch_size = cache.stage->getParams().request_count,
            .m = cache.stage->getParams().verifier_rows_per_request *
                 cache.stage->getParams().request_count,
            .all_position_rows = cache.stage->getParams().verifier_rows_per_request,
            .decode = true,
            .all_position_logits = true,
        };
        auto capture_policy = buildDecodeCapturePolicy(
            /*has_collectives=*/false,
            ctx);
        if (!capture_policy.allow_cached_graph_replay)
        {
            return fail(
                "captured MTP state publication requires mandatory graph replay");
        }
        capture_policy.defer_final_sync = true;
        capture_policy.force_recapture = false;
        capture_policy.graph_replay_plan_policy =
            DeviceGraphExecutor::GraphReplayPlanPolicy::RequireFullGraph;

        if (cache.pipeline_edges)
        {
            capture_policy.graph_replay_plan_policy =
                DeviceGraphExecutor::GraphReplayPlanPolicy::RequireRetainedParentComposition;
            capture_policy.retained_parent_composer = [edges = cache.pipeline_edges, stage = cache.stage](
                IGPUGraphCapture &destination, const ComputeGraph &,
                std::span<const DeviceGraphExecutor::GraphSegmentCache::RetainedCaptureUnitTemplateView> units) {
                if (units.size() != 1 || !units.front().capture) return false;
                return edges->composePublication(destination, *units.front().capture, stage->getParams());
            };
        }

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
                &used_graph_replay,
                submission))
        {
            return fail(
                "captured MTP state publication failed under the strict monolithic capture policy");
        }
        if (submission == DeviceGraphExecutor::GraphInitialSubmissionPolicy::MaterializeWithoutLaunch)
            return true;
        if (!cache.segment_cache.orderStreamAfterCapture(
                &gpu_ctx,
                producer_stream))
        {
            return fail(
                "captured MTP state publication could not publish its completion edge to the outcome stream");
        }
        return true;
    }

}
