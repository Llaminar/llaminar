/**
 * @file Qwen35MoEGraphOverlay.cpp
 * @brief ExpertOverlay routed lowering and shared-expert/residual finalization for Qwen35MoEGraph.
 */

#include "Qwen35MoEGraphInternal.h"

namespace llaminar2
{
    void Qwen35MoEGraph::buildFFNExpertOverlayBranch(FFNBuildContext &ctx)
    {
        using SharedPrefillCPUInputFork = FFNBuildContext::SharedPrefillCPUInputFork;
        using DeferredOverlayCombinedPublication = FFNBuildContext::DeferredOverlayCombinedPublication;

        const LayerWeights &layer = ctx.layer;
        ActivationBuffers &buffers = ctx.buffers;
        const LayerGraphSourceIdentity &identity = ctx.identity;
        const int seq_len = ctx.seq_len;
        const int batch_size = ctx.batch_size;
        const DeviceId device = ctx.device;
        void *const device_state_publication_stream = ctx.device_state_publication_stream;
        const int32_t *const sequence_lengths_device = ctx.sequence_lengths_device;
        const Qwen4ExpPredictorResidualBinding *const predictor = ctx.predictor;
        (void)seq_len;
        (void)batch_size;
        (void)identity;
        (void)layer;

        const int layer_idx = ctx.layer_idx;
        ComputeGraph &graph = ctx.graph;
        const bool mtp_sidecar_context = ctx.mtp_sidecar_context;
        const int mtp_depth_idx = ctx.mtp_depth_idx;
        const std::string &prefix = ctx.prefix;
        std::string &ffn_terminal = ctx.ffn_terminal;
        const int total_tokens = ctx.total_tokens;
        const int32_t *const routed_live_rows = ctx.routed_live_rows;
        const MoEOverlayServicePhaseHint routed_service_phase = ctx.routed_service_phase;
        const ExpertHistogramSource routed_histogram_source = ctx.routed_histogram_source;
        const auto &overlay_runtime_plan = ctx.overlay_runtime_plan;
        const auto &overlay_plan = ctx.overlay_plan;
        const RoutedExpertLayerPlacement *const overlay_placement = ctx.overlay_placement;
        const bool mapped_intermediate_shards = ctx.mapped_intermediate_shards;
        const auto mapped_shard_phase = ctx.mapped_shard_phase;
        const bool mapped_shard_grouped = ctx.mapped_shard_grouped;
        const bool mapped_cuda_retained_m1 = ctx.mapped_cuda_retained_m1;
        const bool mapped_decode_shared_overlap = ctx.mapped_decode_shared_overlap;
        IMoERuntimeTable *const moe_runtime_table = ctx.moe_runtime_table;
        const bool ordinary_prefill_graph = ctx.ordinary_prefill_graph;
        const MoEOverlayNodeLocalRouteTransport node_local_route_transport = ctx.node_local_route_transport;
        ILocalTPContext *const local_tp_ctx = ctx.local_tp_ctx;
        IGlobalTPContext *const global_tp_ctx = ctx.global_tp_ctx;
        const bool distributed_expert_overlay = ctx.distributed_expert_overlay;
        const QwenRoutedContinuationTopology &routed_continuation_topology = ctx.routed_continuation_topology;
        const bool captured_overlay_continuation = ctx.captured_overlay_continuation;
        const bool captured_local_tp_continuation = ctx.captured_local_tp_continuation;
        const RoutedExpertDomain *const cpu_llep_overlay_domain = ctx.cpu_llep_overlay_domain;
        const bool cpu_llep_prefill_transport_supported = ctx.cpu_llep_prefill_transport_supported;
        const bool captured_distributed_overlay_runtime_table = ctx.captured_distributed_overlay_runtime_table;
        const bool captured_overlay_route_ledger_uses_grouped_publication = ctx.captured_overlay_route_ledger_uses_grouped_publication;
        const MoERuntimeRouteWeightProjection captured_overlay_route_weight_projection = ctx.captured_overlay_route_weight_projection;
        const auto &routed_pipeline_kernel_owner = ctx.routed_pipeline_kernel_owner;
        TensorBase *const routing_indices = ctx.routing_indices;
        TensorBase *const routing_weights = ctx.routing_weights;
        const int expert_intermediate = ctx.expert_intermediate;
        TensorBase *const moe_output = ctx.moe_output;
        TensorBase *const canonical_route_contributions = ctx.canonical_route_contributions;
        auto &shared_prefill_cpu_input_fork = ctx.shared_prefill_cpu_input_fork;
        auto &shared_prefill_mapped_dispatch = ctx.shared_prefill_mapped_dispatch;
        auto &shared_decode_cpu_packet = ctx.shared_decode_cpu_packet;
        auto &shared_decode_ticket_cutpoint = ctx.shared_decode_ticket_cutpoint;
        bool &routed_overlay_has_distributed_sparse_protocol = ctx.routed_overlay_has_distributed_sparse_protocol;
        const bool has_shared_expert_branch = ctx.has_shared_expert_branch;
        const DeviceId planned_shared_device = ctx.planned_shared_device;
        const bool shared_expert_requires_tp_allreduce = ctx.shared_expert_requires_tp_allreduce;
        auto &deferred_overlay_combined_publication = ctx.deferred_overlay_combined_publication;
        std::string &captured_overlay_routed_unit_terminal = ctx.captured_overlay_routed_unit_terminal;
        std::string &captured_overlay_pre_return_terminal = ctx.captured_overlay_pre_return_terminal;
        std::string &captured_overlay_first_return_node = ctx.captured_overlay_first_return_node;

        const auto &takeGraphRebalanceSidebandsForAllreduce = ctx.takeGraphRebalanceSidebandsForAllreduce;
        const auto &maybeAddGraphRebalancePayloadStageAfterSideband = ctx.maybeAddGraphRebalancePayloadStageAfterSideband;
        const auto &needsMoEParticipantAllreduce = ctx.needsMoEParticipantAllreduce;
        const auto &makeExpertParams = ctx.makeExpertParams;
        auto prepareExpertParams = [&](MoEExpertComputeStage::Params &expert_params,
                                       DeviceId stage_device,
                                       const std::string &placement_context = {},
                                       const std::string &registry_domain_name = {},
                                       const MoEExpertOwnerParticipant *registry_participant = nullptr)
        {
            return ctx.prepareExpertParams(
                expert_params,
                stage_device,
                placement_context,
                registry_domain_name,
                registry_participant);
        };
        const auto &registerInitialOverlayParticipantBank = ctx.registerInitialOverlayParticipantBank;
        (void)device_state_publication_stream;
        (void)sequence_lengths_device;
        (void)predictor;
        (void)mtp_depth_idx;
        (void)routed_histogram_source;
        (void)mapped_shard_phase;
        (void)mapped_shard_grouped;
        (void)mapped_cuda_retained_m1;
        (void)mapped_decode_shared_overlap;
        (void)captured_local_tp_continuation;
        (void)has_shared_expert_branch;
        (void)planned_shared_device;
        (void)shared_expert_requires_tp_allreduce;

                auto owner_map_lifetime = routedForwardRecipe(device).owners;
                const int continuation_root_participant =
                    continuationRootParticipant(*overlay_plan);
                const int participant_count =
                    participantCountForGraphNativeOverlay(
                        *owner_map_lifetime,
                        continuation_root_participant);
                const bool distributed_overlay =
                    distributed_expert_overlay;
                std::vector<int> local_overlay_participants;
                if (distributed_overlay)
                {
                    const int current_world_rank =
                        config_.moe.overlay_mpi_ctx->rank();
                    for (const auto &participant :
                         owner_map_lifetime->participants())
                    {
                        if (participant.world_rank_known &&
                            participant.world_rank == current_world_rank)
                        {
                            local_overlay_participants.push_back(
                                participant.participant_id);
                        }
                    }
                }
                else
                {
                    for (const auto &participant :
                         owner_map_lifetime->participants())
                    {
                        local_overlay_participants.push_back(
                            participant.participant_id);
                    }
                }
                std::sort(
                    local_overlay_participants.begin(),
                    local_overlay_participants.end());
                local_overlay_participants.erase(
                    std::unique(
                        local_overlay_participants.begin(),
                        local_overlay_participants.end()),
                    local_overlay_participants.end());
                const auto owns_overlay_participant =
                    [&](int participant_id)
                {
                    return std::binary_search(
                        local_overlay_participants.begin(),
                        local_overlay_participants.end(),
                        participant_id);
                };

                const int continuation_tier_index =
                    routed_continuation_topology.tier_index;
                const RoutedExpertDomain *continuation_expert_domain =
                    routed_continuation_topology.domain;
                const int graph_local_continuation_participant =
                    routed_continuation_topology
                        .graph_local_participant;
                /*
                 * Dynamic mapped continuations use the same complete captured
                 * sparse endpoint graph as Static. The mutable choice is not a
                 * host-side graph branch: every MoE runtime table is bound to
                 * DeviceMoEOverlayEpochArena, whose acquire kernel reads the
                 * topology-wide mapped admission epoch and pins one immutable
                 * bank for the complete request transaction. Maintenance may
                 * therefore prepare and publish the inactive bank on its own
                 * streams without changing this graph's topology or embedded
                 * descriptor addresses.
                 */
                const bool owns_continuation_root =
                    captured_overlay_continuation
                        ? graph_local_continuation_participant ==
                              continuation_root_participant
                        : owns_overlay_participant(
                              continuation_root_participant);
                const DistributedSparseGraphContract sparse_graph_contract =
                    resolveDistributedSparseGraphContract(
                        distributed_overlay,
                        captured_overlay_continuation,
                        owns_continuation_root);
                routed_overlay_has_distributed_sparse_protocol =
                    distributed_overlay;
                if (config_.moe.expert_overlay_residency_authority &&
                    !config_.moe.expert_overlay_participant_residency)
                {
                    throw std::runtime_error(
                        "Qwen35 MoE production ExpertOverlay graph has a "
                        "global residency authority but no process-local "
                        "epoch-indexed prepared banks");
                }

                std::string captured_local_expert_compute_node;
                std::vector<std::string> retained_pre_cpu_publications;
                std::string captured_local_expert_terminal;
                std::string captured_overlay_ticket_wave;
                std::string captured_overlay_outbound_wave;
                std::string captured_overlay_inbound_wave;
                int continuation_root_tp_index = -1;
                if (captured_overlay_continuation)
                {
                    /*
                     * Every continuation GPU first records the common router
                     * prefix. The logical root publishes all remote packets,
                     * then its continuation-local expert branch runs in the
                     * same retained device parent while follower devices work
                     * concurrently. Rank-local TP continuations additionally
                     * reduce their fixed route-slot banks and broadcast the
                     * completed row; a single-device continuation needs
                     * neither collective and retains its local output in
                     * MOE_COMBINED_OUTPUT for the ordered remote folds.
                     */
                    captured_overlay_ticket_wave =
                        prefix + "moe_overlay_ticket_capture";
                    captured_overlay_outbound_wave =
                        prefix + "moe_overlay_outbound_capture";
                    captured_overlay_inbound_wave =
                        prefix + "moe_overlay_inbound_capture";
                    // The CUDA program retires this original local wave.
                    // Its participant uses only the typed ticket-unit cutpoints.
                    if (!mapped_cuda_retained_m1)
                        graph.setGraphCaptureWaveContract(
                            prefix + "moe_routing",
                            GraphCaptureWaveContract{
                                .identity = captured_overlay_ticket_wave,
                            });
                    if (!canonical_route_contributions)
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE captured distributed continuation requires canonical route storage");
                    }
                    const auto *local_participant =
                        owner_map_lifetime->participantForId(
                            graph_local_continuation_participant);
                    const auto *root_participant =
                        owner_map_lifetime->participantForId(
                            continuation_root_participant);
                    if (!local_participant || !root_participant ||
                        local_participant->domain_name !=
                            overlay_plan->continuation_domain ||
                        root_participant->domain_name !=
                            overlay_plan->continuation_domain)
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE captured distributed continuation has an unaligned logical participant mapping");
                    }
                    if (captured_local_tp_continuation)
                    {
                        if (!local_tp_ctx ||
                            local_participant->domain_participant_index !=
                                config_.tp_device_idx ||
                            root_participant->domain_participant_index < 0 ||
                            root_participant->domain_participant_index >=
                                local_tp_ctx->degree())
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE captured distributed LocalTP continuation has an unaligned graph-device/participant mapping");
                        }
                        continuation_root_tp_index =
                            root_participant->domain_participant_index;
                    }
                    else if (graph_local_continuation_participant !=
                                 continuation_root_participant ||
                             !continuation_expert_domain ||
                             continuation_expert_domain->scope !=
                                 ExecutionDomainScope::SINGLE ||
                             continuation_expert_domain->participants.size() !=
                                 1u ||
                             local_participant->domain_participant_index != 0)
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE captured single-device continuation has an invalid root identity");
                    }

                    if (mapped_intermediate_shards)
                    {
                        auto *runtime = dynamic_cast<DeviceMoERuntimeTable *>(moe_runtime_table);
                        const auto &residency = config_.moe.expert_overlay_participant_residency;
                        const auto manager = model_ctx_ ? model_ctx_->concreteWeightManager() : nullptr;
                        if (!runtime || !residency || !manager || !prepared_weight_store_ ||
                            !canonical_route_contributions || !device_state_publication_stream)
                            throw std::logic_error("Mapped intermediate shards require the original live runtime and fragment preparation");
                        const auto declaration = adoptMoEOverlayJointGraphDeclaration(
                            *residency, manager->expertGemmRegistry(), prepared_weight_store_->modelId(),
                            graph_local_continuation_participant, layer_idx, config_.moe.num_experts,
                            config_.d_model, config_.moe.intermediate_size);
                        const auto initial = declaration->acquireInitialConstruction();
                        if (!initial)
                            throw std::logic_error("Mapped intermediate shards lost their original construction source");
                        const auto &source = initial->layerForModelLayer(layer_idx);
                        publishInitialMoEIntermediateShardRuntimeLayer(*runtime, *owner_map_lifetime,
                            source, layer_idx, device_state_publication_stream);
                        MoEIntermediateShardPipelineParams local;
                        local.family = declaration->family();
                        local.runtime = runtime;
                        local.kernel_owner = routed_pipeline_kernel_owner;
                        local.rows = total_tokens; local.top_k = config_.moe.top_k;
                        local.phase = mapped_shard_phase;
                        local.row_mode = mapped_shard_grouped
                            ? MoEIntermediateShardRowMode::Grouped : MoEIntermediateShardRowMode::Decode;
                        if (routed_live_rows)
                            local.live_rows = DeviceRowRange::deviceCounted(total_tokens, routed_live_rows);
                        local.prefix = prefix + "mapped_routed/";
                        local.router_node = prefix + "moe_routing";
                        local.hidden = {buffers.normalized, buffers.idFor(BufferId::NORMALIZED)};
                        local.routing_indices = {routing_indices, buffers.idFor(BufferId::MOE_EXPERT_INDICES)};
                        local.routing_weights = {routing_weights, buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS)};
                        local.route_partials = {canonical_route_contributions,
                            buffers.idFor(BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS)};
                        const auto slots = appendMoEIntermediateShardRouteSlots(graph, local, source);
                        captured_local_expert_compute_node = slots.producer;
                    }
                    else
                    {
                        auto participant_mask =
                            owner_map_lifetime->expertMaskForParticipant(
                                layer_idx,
                                graph_local_continuation_participant,
                                config_.moe.num_experts);
                        if (!hasActiveExpertMask(participant_mask))
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE captured distributed continuation produced an empty graph-local expert mask");
                        }

                        auto local_params = makeExpertParams(
                            moe_output,
                            buffers.idFor(BufferId::MOE_COMBINED_OUTPUT),
                            std::move(participant_mask),
                            device);
                        local_params.my_socket_id =
                            local_participant->domain_participant_index;
                        local_params.participant_count =
                            captured_local_tp_continuation
                                ? local_tp_ctx->degree()
                                : 1;
                        local_params.require_device_routing_tensor_decode =
                            device.is_gpu();
                        /* Every captured heterogeneous continuation publishes its
                         * graph-local routes in original router slots. LocalTP
                         * transports those slots between continuation devices;
                         * a single-device continuation folds them locally before
                         * mapped remote returns are joined. Keeping one producer
                         * contract makes the local arithmetic and diagnostics
                         * independent of transport topology. */
                        local_params.canonical_route_contributions =
                            canonical_route_contributions;
                        local_params.canonical_route_diagnostic_publication =
                            MoECanonicalRouteDiagnosticPublication::ContinuationLocalBank;
                        local_params.canonical_route_arithmetic =
                            MoECanonicalRouteArithmeticPolicy::
                                PreweightedContributionThenOrderedAdd;
                        local_params.canonical_route_layout =
                            MoECanonicalRoutePublicationLayout::
                                DenseOriginalRouteSlots;
                        local_params.canonical_route_contributions_buffer_id =
                            buffers.idFor(
                                BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS);

                        if (!prepareExpertParams(
                                local_params,
                                device,
                                "captured distributed continuation participant " +
                                    std::to_string(
                                        graph_local_continuation_participant),
                                overlay_plan->continuation_domain,
                                local_participant))
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE could not prepare captured continuation-local experts for layer " +
                                std::to_string(layer_idx) + " on " +
                                device.to_string());
                        }

                        if (captured_distributed_overlay_runtime_table)
                        {
                            const auto domain_local_owners =
                                domainLocalOwnerParticipantsFromMap(
                                    *owner_map_lifetime,
                                    layer_idx,
                                    config_.moe.num_experts,
                                    overlay_plan->continuation_domain);
                            const auto overlay_route_participants =
                                ownerParticipantsFromMap(
                                    *owner_map_lifetime,
                                    layer_idx,
                                    config_.moe.num_experts);
                            if (!initializeMaskedLocalDecodeRuntimeTable(
                                    moe_runtime_table,
                                    layer_idx,
                                    config_.moe.num_experts,
                                    config_.moe.top_k,
                                    config_.d_model,
                                    expert_intermediate,
                                    local_params.expert_mask,
                                    local_params.my_socket_id,
                                    local_params.participant_count,
                                    domain_local_owners,
                                    overlay_route_participants,
                                    local_params.prepared_gate_gemm,
                                    local_params.prepared_up_gemm,
                                    local_params.prepared_down_gemm,
                                    device_state_publication_stream,
                                    /*allow_existing_dynamic_bank=*/true,
                                    "captured distributed continuation decode runtime bank"))
                            {
                                throw std::runtime_error(
                                    "Qwen35 MoE could not initialize the captured continuation-local decode runtime table for layer " +
                                    std::to_string(layer_idx) + " on " +
                                    device.to_string());
                            }
                            if (total_tokens > 1 &&
                                captured_overlay_route_ledger_uses_grouped_publication)
                            {
                                local_params.use_runtime_row_grouping = true;
                            }
                        }

                        registerInitialOverlayParticipantBank(
                            *owner_map_lifetime,
                            graph_local_continuation_participant,
                            local_params,
                            "Qwen35 MoE captured distributed continuation");

                        captured_local_expert_compute_node =
                            prefix +
                            "moe_expert_ffn_overlay_continuation_local";
                        graph.addNode(
                            captured_local_expert_compute_node,
                            ComputeStageFactory::createMoEExpertCompute(
                                local_params),
                            device);
                        graph.addDependency(
                            captured_local_expert_compute_node,
                            prefix + "moe_routing");
                    }
                    if (!mapped_cuda_retained_m1)
                        graph.setGraphCaptureWaveContract(
                            captured_local_expert_compute_node,
                            GraphCaptureWaveContract{
                                .identity = captured_overlay_outbound_wave,
                            });

                    captured_local_expert_terminal =
                        captured_local_expert_compute_node;
                    /* Local and remote experts publish disjoint original route
                     * slots into one canonical bank. Do not reduce the local
                     * prefix here: the sole ordered reducer is installed only
                     * after every mapped return has materialized its slots. */
                }

                /*
                 * CPU least-loaded prefill is a transient child of the one
                 * durable ExpertOverlay epoch.  The state is graph-owned and
                 * shared by the begin planner, root descriptor, the one local
                 * sparse endpoint on each rank, and restore.  No child owner
                 * map is published: the durable lease remains the only
                 * placement authority until every sparse return completes.
                 */
                std::shared_ptr<CPUCurrentBatchLLEPTransactionState>
                    cpu_llep_state;
                if (cpu_llep_prefill_transport_supported)
                {
                    cpu_llep_state =
                        std::make_shared<
                            CPUCurrentBatchLLEPTransactionState>();
                }
                const auto host_dispatch_lease_owner =
                    cpu_llep_state
                        ? MoEOverlayHostDispatchLeaseOwner::CurrentBatchLLEP
                        : config_.moe.expert_overlay_residency_authority
                              ? MoEOverlayHostDispatchLeaseOwner::
                                    FinalSparseReturn
                              : MoEOverlayHostDispatchLeaseOwner::None;
                MoELocalExpertStage *cpu_llep_local_consumer = nullptr;
                std::vector<std::string> cpu_llep_sparse_dispatch_nodes;

                auto dispatch_output_lifetime = std::make_shared<MoEExpertDispatchOutput>();
                std::shared_ptr<MoEOverlayDispatchTicketStorage>
                    dispatch_ticket_storage;
                std::string dispatch_ticket_publish_name;
                const std::string dispatch_name =
                    prefix + "moe_expert_dispatch";
                std::string dispatch_dependency =
                    prefix + "moe_routing";
                bool host_dispatch_path_materialized = false;
                const auto ensure_host_dispatch_path = [&]()
                {
                    if (host_dispatch_path_materialized)
                        return;
                    if (captured_overlay_continuation &&
                        requiresSingleNativeExecutable(graph.nativeCaptureEnvelope()))
                        throw std::logic_error(
                            "Captured device-owned expert routing cannot create a host dispatch endpoint");
                    if (sparse_graph_contract.ownsDispatchAuthority() &&
                        device.is_gpu())
                    {
                        dispatch_ticket_storage = dispatchTicketForForward(
                            layer_idx, total_tokens, device,
                            moe_runtime_table
                                ? moe_runtime_table->hostLayerState(layer_idx).overlay_epoch_ticket
                                : nullptr);
                        if (config_.moe.expert_overlay_residency_authority &&
                            dispatch_ticket_storage->epochAuthority() !=
                                MoEOverlayDispatchEpochAuthority::CapturedRequest)
                        {
                            throw std::logic_error(
                                "GPU ExpertOverlay dispatch requires the captured request's placement epoch");
                        }
                        dispatch_output_lifetime->ticket_lifetime =
                            dispatch_ticket_storage;

                        const std::string ticket_publish_dependency =
                            prefix + "moe_routing";

                        MoEOverlayTicketPublishStage::Params ticket_params;
                        ticket_params.device_id = device;
                        ticket_params.hidden = buffers.normalized;
                        ticket_params.routing_indices = routing_indices;
                        ticket_params.routing_weights = routing_weights;
                        ticket_params.hidden_buffer_id =
                            buffers.idFor(BufferId::NORMALIZED);
                        ticket_params.routing_indices_buffer_id =
                            buffers.idFor(BufferId::MOE_EXPERT_INDICES);
                        ticket_params.routing_weights_buffer_id =
                            buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS);
                        /*
                         * A decode graph always owns one activation row even
                         * though sequence_lengths_device contains the growing
                         * KV position.  Only a padded multi-row prefill bucket
                         * needs that scalar to distinguish real prompt rows
                         * from physical padding.
                         */
                        ticket_params.active_row_count_device =
                            batch_size == 1 && total_tokens > 1
                                ? routed_live_rows
                                : nullptr;
                        ticket_params.layer_idx = layer_idx;
                        ticket_params.bucket_rows = total_tokens;
                        ticket_params.top_k = config_.moe.top_k;
                        ticket_params.d_model = config_.d_model;
                        ticket_params.ticket_storage = dispatch_ticket_storage;

                        dispatch_ticket_publish_name =
                            prefix + "moe_overlay_ticket_publish";
                        graph.addNode(
                            dispatch_ticket_publish_name,
                            ComputeStageFactory::createMoEOverlayTicketPublish(
                                ticket_params),
                            device);
                        graph.addDependency(
                            dispatch_ticket_publish_name,
                            ticket_publish_dependency);
                        if (captured_overlay_continuation)
                        {
                            graph.setGraphCaptureWaveContract(
                                dispatch_ticket_publish_name,
                                GraphCaptureWaveContract{
                                    .identity =
                                        captured_overlay_ticket_wave,
                                });
                        }
                    }

                    MoEExpertDispatchStage::Params dispatch_params;
                    dispatch_params.device_id = DeviceId::cpu();
                    dispatch_params.ticket_storage = dispatch_ticket_storage;
                    if (!dispatch_ticket_storage)
                    {
                        dispatch_params.routing_indices = routing_indices;
                        dispatch_params.routing_weights = routing_weights;
                        dispatch_params.hidden = buffers.normalized;
                        dispatch_params.routing_indices_buffer_id =
                            buffers.idFor(BufferId::MOE_EXPERT_INDICES);
                        dispatch_params.routing_weights_buffer_id =
                            buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS);
                        dispatch_params.hidden_buffer_id =
                            buffers.idFor(BufferId::NORMALIZED);
                    }
                    dispatch_params.seq_len = total_tokens;
                    dispatch_params.top_k = config_.moe.top_k;
                    dispatch_params.d_model = config_.d_model;
                    dispatch_params.continuation_domain = overlay_runtime_plan
                                                              ? overlay_runtime_plan->continuationDomain().name
                                                              : overlay_plan->continuation_domain;
                    dispatch_params.transfer_mode = debugEnv().moe_expert_overlay.dense_transfer
                                                        ? MoEExpertTransferMode::DenseFullSequence
                                                        : MoEExpertTransferMode::Auto;
                    dispatch_params.placement = *overlay_placement;
                    dispatch_params.routed_tiers = overlay_plan->routed_tiers;
                    dispatch_params.owner_map = owner_map_lifetime;
                    dispatch_params.residency_authority =
                        config_.moe.expert_overlay_residency_authority;
                    dispatch_params.cpu_current_batch_llep_state =
                        cpu_llep_state;
                    if (dispatch_ticket_storage && config_.moe.decode_histogram &&
                        (!mapped_intermediate_shards ||
                         config_.moe.rebalance_config.mode != MoERebalanceRuntimeMode::Off))
                    {
                        // This declared heterogeneous boundary already owns
                        // exact routes and live row geometry. One ticket is
                        // one executed batch, including rejected MTP work;
                        // acceptance never changes these service-cost facts.
                        dispatch_params.routing_evidence_publication =
                            MoEExpertDispatchStage::RoutingEvidencePublication{
                                .histogram = config_.moe.decode_histogram,
                                // Replicated TP routers describe one logical transaction.
                                // The frozen owner map already authenticated both IDs;
                                // only its continuation root publishes shared history.
                                // Every peer still constructs and executes its own
                                // original CPU/GPU expert dispatch and return work.
                                .producer_role = captured_overlay_continuation
                                    ? selectMoEContinuationHistogramProducerRole(
                                          graph_local_continuation_participant,
                                          continuation_root_participant)
                                    : MoERuntimeHistogramProducerRole::ProductionDecode,
                                .source = routed_histogram_source,
                            };
                    }
                    dispatch_params.output_lifetime = dispatch_output_lifetime;

                    if (sparse_graph_contract.ownsDispatchAuthority())
                    {
                        graph.addNode(
                            dispatch_name,
                            ComputeStageFactory::createMoEExpertDispatch(
                                dispatch_params),
                            DeviceId::cpu());
                        graph.addDependency(
                            dispatch_name,
                            dispatch_ticket_publish_name.empty()
                                ? prefix + "moe_routing"
                                : dispatch_ticket_publish_name);
                        dispatch_dependency = dispatch_name;
                        if (captured_overlay_continuation &&
                            !captured_local_expert_compute_node.empty())
                        {
                            /*
                             * Ticket publication is the immutable fork point
                             * for continuation-local GPU work. The GPU branch
                             * must never wait for host parsing: it consumes the
                             * already-published device bytes directly.
                             */
                            if (mapped_cuda_retained_m1)
                            {
                                if (dispatch_ticket_publish_name.empty())
                                    throw std::logic_error("GPU mapped retained QFN requires the original device ticket publisher");
                                // The local fusion retains the packet as an explicit reader.
                                // Native capture substitutes its original early router edge.
                                graph.addDependency(dispatch_ticket_publish_name, captured_local_expert_compute_node);
                                retained_pre_cpu_publications.push_back(dispatch_ticket_publish_name);
                            }
                            else
                                graph.addDependency(
                                    captured_local_expert_compute_node,
                                    dispatch_ticket_publish_name.empty()
                                        ? dispatch_name
                                        : dispatch_ticket_publish_name);

                            if (!dispatch_ticket_publish_name.empty() &&
                                requiresHeterogeneousTicketSegmentation(
                                    graph.nativeCaptureEnvelope()))
                            {
                                /*
                                 * A retained heterogeneous transaction has one
                                 * manual CPU unit per layer. Materializing the
                                 * host route table before the captured local
                                 * expert producer created a second host gap:
                                 * captured ticket -> host parse -> captured
                                 * local expert -> host sparse transaction.
                                 *
                                 * Make host parsing consume the completed local
                                 * producer instead. The downstream sparse CPU
                                 * stages already depend on this dispatch node,
                                 * so all host work becomes one adjacent manual
                                 * unit after the typed pre-CPU cutpoint. This
                                 * preserves device/host independence (the GPU
                                 * never waits on the host), removes a redundant
                                 * retained launch, and gives LocalTP authority
                                 * and followers one symmetric capture boundary.
                                 */
                                graph.addDependency(
                                    dispatch_name,
                                    captured_local_expert_compute_node);
                            }
                        }
                    }
                    if (ordinary_prefill_graph && has_shared_expert_branch &&
                        planned_shared_device == device &&
                        requiresHeterogeneousTicketSegmentation(graph.nativeCaptureEnvelope()) &&
                        !dispatch_ticket_publish_name.empty() &&
                        !captured_local_expert_compute_node.empty() &&
                        std::any_of(owner_map_lifetime->participants().begin(),
                            owner_map_lifetime->participants().end(),
                            [](const auto &participant) { return participant.device.is_cpu(); }))
                    {
                        // The complete ticket owns normalized rows and routes
                        // before either expert branch runs. CPU service already
                        // waits on this publication in its persistent worker;
                        // it need not wait for the unrelated shared GPU FFN.
                        shared_prefill_cpu_input_fork = SharedPrefillCPUInputFork{
                            .publication_node = dispatch_ticket_publish_name,
                            .routed_expert_node = captured_local_expert_compute_node};
                    }
                    host_dispatch_path_materialized = true;
                };

                struct OverlayTierTargets
                {
                    std::vector<int> participants;
                    bool graph_local_apportioned = false;
                };
                const auto resolve_overlay_tier_targets =
                    [&](size_t tier_index) -> OverlayTierTargets
                {
                    const auto &tier =
                        overlay_plan->routed_tiers[tier_index];
                    OverlayTierTargets result;
                    /*
                     * A heterogeneous GPU continuation executes its own tier
                     * directly inside each captured device graph. Only the
                     * exact logical root enters the rank-level sparse protocol,
                     * and that protocol addresses remote tiers only. This also
                     * prevents LocalTP siblings from submitting duplicate
                     * cross-rank collectives.
                     */
                    if (captured_overlay_continuation &&
                        static_cast<int>(tier_index) ==
                            continuation_tier_index)
                    {
                        return result;
                    }
                    if (captured_overlay_continuation &&
                        distributed_overlay &&
                        !sparse_graph_contract
                             .participatesInRankBatchProtocol())
                    {
                        return result;
                    }
                    result.participants =
                        owner_map_lifetime->participantIdsForTier(
                            static_cast<int>(tier_index));
                    const bool local_tp_apportioned =
                        isLocalTPExpertIdApportionedTier(
                            *overlay_plan,
                            tier);
                    const auto graph_scope = local_tp_apportioned
                        ? std::optional{MoEExpertOverlayGraphScope::resolveForTPParticipant(
                              *overlay_runtime_plan, device, config_.tp_ctx, config_.tp_device_idx)}
                        : std::nullopt;
                    const int graph_local_participant =
                        graph_scope
                            ? participantIdForTierDevice(
                                  *owner_map_lifetime,
                                  static_cast<int>(tier_index),
                                  device,
                                  &*graph_scope)
                            : -1;
                    result.graph_local_apportioned =
                        graph_local_participant >= 0 &&
                        (!distributed_overlay ||
                         owns_overlay_participant(
                             graph_local_participant));
                    if (result.graph_local_apportioned)
                        result.participants = {graph_local_participant};
                    return result;
                };

                /*
                 * A same-node distributed GPU continuation lowers directly to
                 * mapped activation epochs only when every remote participant
                 * in this graph family is physically node-local. Mixing a
                 * portable inter-node message into the retained mapped parent
                 * would reintroduce a host-owned transaction boundary, so a
                 * mixed physical topology remains one explicit MPI protocol.
                 * Participant lanes come from the frozen topology rather than
                 * the initial expert masks: an initially empty endpoint must
                 * remain addressable after a later placement-bank migration.
                 */
                const size_t mapped_activation_graph_family_ordinal =
                    mtp_sidecar_context
                        ? static_cast<size_t>(
                              std::max(0, mtp_depth_idx) + 1)
                        : 0u;
                std::vector<int> mapped_activation_remote_participants;
                std::vector<int> mapped_activation_local_participants;
                std::vector<std::string>
                    mapped_activation_dispatch_wave_identities;
                std::vector<std::string>
                    mapped_activation_return_wave_identities;
                const auto mapped_dispatch_wave_identity =
                    [&](int participant)
                {
                    return prefix +
                           "moe_overlay_mapped_dispatch_capture_family" +
                           std::to_string(
                               mapped_activation_graph_family_ordinal) +
                           "_p" + std::to_string(participant);
                };
                const auto mapped_return_wave_identity =
                    [&](int participant)
                {
                    return prefix +
                           "moe_overlay_mapped_return_capture_family" +
                           std::to_string(
                               mapped_activation_graph_family_ordinal) +
                           "_p" + std::to_string(participant);
                };
                bool use_mapped_activation_parent = false;
                MoEOverlayRoutePlacementDeviceBinding
                    pinned_route_placement;
                MoEDomainRouteAssignmentLedger
                    pinned_domain_route_assignment{};
                MoERuntimeRouteWeightBinding
                    pinned_runtime_route_weights{};
                if (captured_overlay_continuation &&
                    device.is_gpu())
                {
                    const auto *const source_descriptor =
                        owner_map_lifetime->participantForId(
                            continuation_root_participant);
                    if (!source_descriptor ||
                        !source_descriptor->world_rank_known ||
                        source_descriptor->world_rank !=
                            config_.moe.overlay_mpi_ctx->rank())
                    {
                        throw std::logic_error(
                            "Qwen35 MoE mapped activation parent cannot resolve its continuation rank");
                    }

                    // Planning reserves every authored service endpoint. Capture
                    // uses that same topology after frozen quotas prove which
                    // tiers can ever contribute; a mutable empty mask proves nothing.
                    const auto hostService=MoEOverlayHostServiceContract::resolveAdmitted(*overlay_plan,
                        *owner_map_lifetime,*config_.moe.overlay_mpi_ctx);
                    if(hostService.continuationTier()!=continuation_tier_index)
                        throw std::logic_error("Routed host service changed original continuation tier");
                    mapped_activation_local_participants.assign(hostService.localParticipants().begin(),hostService.localParticipants().end());
                    mapped_activation_remote_participants.assign(hostService.externalParticipants().begin(),hostService.externalParticipants().end());
                    const bool mapped_activation_topology=hostService.mappedActivationTopology();
                    // The original admitted endpoint contract owns this decision.
                    // No host boundary means one native parent; otherwise only
                    // the declared logical root owns the ticket service.
                    graph.setNativeCaptureEnvelope(hostService.captureEnvelope(
                        sparse_graph_contract.ownsDispatchAuthority()));
                    if (mapped_activation_topology)
                    {
                        mapped_activation_dispatch_wave_identities.reserve(
                            mapped_activation_remote_participants.size());
                        mapped_activation_return_wave_identities.reserve(
                            mapped_activation_remote_participants.size());
                        for (const int participant :
                             mapped_activation_remote_participants)
                        {
                            mapped_activation_dispatch_wave_identities.push_back(
                                mapped_dispatch_wave_identity(participant));
                            mapped_activation_return_wave_identities.push_back(
                                mapped_return_wave_identity(participant));
                        }
                    }
                    use_mapped_activation_parent =
                        sparse_graph_contract.ownsDispatchAuthority() &&
                        sparse_graph_contract
                            .participatesInRankBatchProtocol() &&
                        mapped_activation_topology;
                    if (sparse_graph_contract.ownsDispatchAuthority())
                    {
                        /*
                         * Transport choice does not own routing truth. Mapped
                         * GPU lanes, a colocated CPU ticket, and portable MPI
                         * packets all consume the same device-authored route
                         * ledger and placement epoch. Bind those values for
                         * every continuation authority so the final reducer
                         * can publish exact evidence even when this topology
                         * has no remote mapped GPU participant.
                         */
                        if (!moe_runtime_table)
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE captured overlay continuation requires the canonical device runtime placement table");
                        }
                        pinned_route_placement =
                            moe_runtime_table->overlayRoutePlacementBinding(
                                layer_idx);
                        if (!pinned_route_placement.valid())
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE captured overlay continuation received an incomplete placement-bank binding");
                        }
                        const auto &runtime_layer =
                            moe_runtime_table->hostLayerState(layer_idx);
                        const std::uint64_t required_route_slots =
                            static_cast<std::uint64_t>(total_tokens) *
                            static_cast<std::uint64_t>(config_.moe.top_k);
                        if (!runtime_layer.route_participant_ids ||
                            required_route_slots == 0u ||
                            required_route_slots >
                                static_cast<std::uint64_t>(
                                    runtime_layer.prefill_route_capacity))
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE captured overlay continuation has no complete final device route assignment ledger");
                        }
                        pinned_domain_route_assignment = {
                            .participant_ids =
                                runtime_layer.route_participant_ids,
                            .capacity =
                                runtime_layer.prefill_route_capacity,
                        };
                        pinned_runtime_route_weights =
                            bindMoERuntimeRouteWeights(
                                moe_runtime_table->deviceLayerState(
                                    layer_idx),
                                runtime_layer,
                                captured_overlay_route_weight_projection);
                        if (!pinned_runtime_route_weights.validFor(
                                static_cast<std::uint32_t>(total_tokens),
                                static_cast<std::uint32_t>(
                                    config_.moe.top_k)))
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE captured overlay continuation has no workload-correct final device route-weight publication");
                        }
                    }
                    else if (!mapped_cuda_retained_m1 && mapped_activation_topology &&
                             !sparse_graph_contract
                                  .ownsDispatchAuthority())
                    {
                        /*
                         * The logical root captures one packet publisher per
                         * remote lane before both continuation GPUs enter the
                         * common local-expert wave. A sibling with no packet
                         * work must still join those capture lifecycles in the
                         * same order; passive waves express that absence
                         * without launching an empty graph or routing traffic
                         * through the host.
                         */
                        graph.setGraphCaptureWaveContract(
                            prefix + "moe_routing",
                            GraphCaptureWaveContract{
                                .identity = captured_overlay_ticket_wave,
                                .passive_following_identities =
                                    mapped_activation_dispatch_wave_identities,
                            });
                    }
                }

                if (sparse_graph_contract.ownsDispatchAuthority() &&
                    (!captured_overlay_continuation ||
                     requiresHeterogeneousTicketSegmentation(graph.nativeCaptureEnvelope())))
                {
                    // The admitted endpoint contract is authoritative. An absent
                    // mapped remote lane does not imply that CPU dispatch exists.
                    ensure_host_dispatch_path();
                }

                /**
                 * Exact endpoint whose ordered return closes host dispatch.
                 *
                 * Rank-batched remote groups are lowered before rank-local
                 * endpoints. Keeping this as one typed endpoint prevents a
                 * remote batch and a later loopback return from both claiming
                 * the same residency lease terminal.
                 */
                struct OrderedHostSparseReturnEndpoint
                {
                    size_t tier_index = 0;
                    int participant_id = -1;

                    [[nodiscard]] bool matches(
                        size_t tier,
                        int participant) const noexcept
                    {
                        return tier_index == tier &&
                               participant_id == participant;
                    }
                };
                std::optional<OrderedHostSparseReturnEndpoint>
                    final_overlay_participant;
                std::optional<OrderedHostSparseReturnEndpoint>
                    final_rank_local_overlay_participant;
                for (size_t tier_index = 0;
                     tier_index < overlay_plan->routed_tiers.size();
                     ++tier_index)
                {
                    const auto tier_mask = expertMaskForTier(
                        *overlay_placement,
                        config_.moe.num_experts,
                        static_cast<int>(tier_index));
                    if (!hasActiveExpertMask(tier_mask))
                        continue;
                    const auto targets =
                        resolve_overlay_tier_targets(tier_index);
                    for (const int target_participant :
                         targets.participants)
                    {
                        const auto participant_mask =
                            owner_map_lifetime->expertMaskForParticipant(
                                layer_idx,
                                target_participant,
                                config_.moe.num_experts);
                        if (hasActiveExpertMask(participant_mask))
                        {
                            final_overlay_participant =
                                OrderedHostSparseReturnEndpoint{
                                    .tier_index = tier_index,
                                    .participant_id = target_participant,
                                };
                        }
                        const auto *const participant_descriptor =
                            owner_map_lifetime->participantForId(
                                target_participant);
                        if (distributed_overlay &&
                            sparse_graph_contract
                                .ownsDispatchAuthority() &&
                            participant_descriptor &&
                            participant_descriptor->world_rank_known &&
                            participant_descriptor->world_rank ==
                                config_.moe.overlay_mpi_ctx->rank() &&
                            (hasActiveExpertMask(participant_mask) ||
                             use_mapped_activation_parent))
                        {
                            /* The rank-local ticket contains only colocated
                             * lower-tier returns, so its completion frontier
                             * is independent of mapped remote lanes. */
                            final_rank_local_overlay_participant =
                                OrderedHostSparseReturnEndpoint{
                                    .tier_index = tier_index,
                                    .participant_id = target_participant,
                                };
                        }
                    }
                }
                const std::optional<OrderedHostSparseReturnEndpoint>
                    final_host_sparse_return =
                        final_rank_local_overlay_participant
                            ? final_rank_local_overlay_participant
                            : final_overlay_participant;

                const auto *continuation_descriptor = owner_map_lifetime->participantForId(
                    continuation_root_participant);
                if (!continuation_descriptor)
                    throw std::logic_error("Host sparse return has no continuation participant identity");
                const auto host_return_layout = continuation_descriptor->device.is_cpu()
                    ? MoEOverlayReturnLayout::CanonicalExpertRoutes
                    : MoEOverlayReturnLayout::ParticipantTokenPartials;
                std::vector<std::shared_ptr<MoEOverlayCollectiveWorkspace>> participant_workspaces(
                    static_cast<size_t>(participant_count));
                for (int participant = 0;
                     participant < participant_count;
                     ++participant)
                {
                    participant_workspaces[static_cast<size_t>(participant)] =
                        overlayProtocolWorkspaceForParticipant(
                            device, participant, host_return_layout);
                }

                std::shared_ptr<IMoEOverlaySparseCollectiveContext>
                    collective_context_lifetime;
                std::shared_ptr<IMoEOverlaySparseCollectiveContext>
                    rank_local_collective_context_lifetime;
                if (!distributed_overlay)
                {
                    MoEOverlayLocalSparseCollectiveContext::Config
                        collective_config;
                    collective_config.participant_count =
                        participant_count;
                    collective_config.slot_count = std::max<size_t>(
                        8,
                        overlay_plan->routed_tiers.size() *
                                static_cast<size_t>(participant_count) * 4u +
                            8u);
                    collective_context_lifetime =
                        std::make_shared<
                            MoEOverlayLocalSparseCollectiveContext>(
                            collective_config);
                }

                const auto ensure_rank_local_collective_context = [&]()
                {
                    if (rank_local_collective_context_lifetime)
                        return;
                    MoEOverlayRankLocalSparseCollectiveContext::Config
                        rank_local_config;
                    rank_local_config.slot_count = std::max<size_t>(
                        8,
                        overlay_plan->routed_tiers.size() *
                                static_cast<size_t>(participant_count) * 4u +
                            8u);
                    rank_local_collective_context_lifetime =
                        std::make_shared<
                            MoEOverlayRankLocalSparseCollectiveContext>(
                                rank_local_config);
                };

                if (captured_overlay_continuation && !distributed_overlay)
                {
                    /* One continuation GPU and colocated lower tiers form
                     * direct point-to-point sparse edges. They must not mimic
                     * a participant-count collective merely because MPI has a
                     * single rank. */
                    ensure_rank_local_collective_context();
                    collective_context_lifetime =
                        rank_local_collective_context_lifetime;
                }

                std::string last_return_reduce;
                bool first_return_scatter = true;
                std::vector<std::string>
                    distributed_rank_batch_dispatch_nodes;
                std::vector<std::string>
                    mapped_activation_dispatch_nodes;
                struct MappedActivationReturnBinding
                {
                    std::string dispatch_node_name;
                    std::string node_name;
                    MoEOverlayMappedActivationDeviceLane lane;
                    std::uint32_t stage_ordinal = 0u;
                };
                std::vector<MappedActivationReturnBinding>
                    mapped_activation_return_bindings;
                std::vector<std::string>
                    mapped_activation_return_nodes;
                bool has_rank_local_canonical_ticket_returns = false;
                std::string local_canonical_ticket_terminal;
                struct PendingCanonicalTicketConsume
                {
                    std::string participant_suffix;
                    std::shared_ptr<
                        MoEOverlayCanonicalRouteReturnTicketStorage>
                        storage;
                };
                std::vector<PendingCanonicalTicketConsume>
                    pending_canonical_ticket_consumes;

                const auto routed_domain_ordinal_for_tier =
                    [&](const RoutedExpertTier &tier) -> int
                {
                    for (size_t candidate = 0;
                         candidate < overlay_plan->domains.size();
                         ++candidate)
                    {
                        if (overlay_plan->domains[candidate].name ==
                            tier.domain)
                        {
                            return static_cast<int>(candidate);
                        }
                    }
                    throw std::logic_error(
                        "Qwen35 MoE rank-batch tier has no stable routed-domain ordinal");
                };

                /**
                 * Build one CPU expert endpoint inside a full NodeTP follower
                 * graph. The caller owns the rank-batch receive/send stages;
                 * this helper owns only prepared expert state and the exact
                 * dispatch-to-compute edge. Keeping that boundary explicit
                 * prevents the follower from acquiring route or publication
                 * authority merely because it also carries dense TP layers.
                 */
                const auto add_rank_batch_target_local_expert =
                    [&](size_t tier_index,
                        int target_participant,
                        const std::shared_ptr<MoEOverlaySparseRows> &inbound,
                        const std::shared_ptr<MoEOverlayReturnRows> &outbound,
                        const std::string &dispatch_node) -> std::string
                {
                    const auto &tier =
                        overlay_plan->routed_tiers[tier_index];
                    const auto *participant =
                        owner_map_lifetime->participantForId(
                            target_participant);
                    if (!participant || !participant->world_rank_known ||
                        participant->world_rank !=
                            config_.moe.overlay_mpi_ctx->rank())
                    {
                        throw std::logic_error(
                            "Qwen35 MoE rank-batch target does not own its declared participant");
                    }

                    const DeviceId target_device =
                        participantDeviceForGraphNativeOverlay(
                            *owner_map_lifetime,
                            target_participant);
                    if (!device.is_cpu() || !target_device.is_cpu())
                    {
                        throw std::logic_error(
                            "Qwen35 MoE full-model rank-batch follower is the CPU NodeTP protocol; captured GPU peers require the retained device transaction role");
                    }

                    const bool joint_host=config_.moe.expert_overlay_participant_residency &&
                        config_.moe.expert_overlay_participant_residency->isJointParticipant(target_participant);
                    std::unique_ptr<MoECPUExpertShardRouteStage> joint_collective;
                    auto participant_mask =
                        owner_map_lifetime->expertMaskForParticipant(
                            layer_idx,
                            target_participant,
                            config_.moe.num_experts);
                    if (!hasActiveExpertMask(participant_mask))
                    {
                        throw std::logic_error(
                            "Qwen35 MoE rank-batch follower was asked to build an inactive participant endpoint");
                    }

                    MoELocalExpertStage::Params local_params;
                    local_params.device_id = target_device;
                    local_params.input_rows_lifetime = inbound;
                    local_params.output_rows_lifetime = outbound;
                    local_params.return_layout = host_return_layout;
                    local_params.workspace_lifetime =
                        participant_workspaces[static_cast<size_t>(
                            target_participant)];
                    local_params.gate_exps = layer.moe_gate_exps;
                    local_params.up_exps = layer.moe_up_exps;
                    local_params.down_exps = layer.moe_down_exps;
                    local_params.num_experts = config_.moe.num_experts;
                    local_params.top_k = config_.moe.top_k;
                    local_params.d_model = config_.d_model;
                    local_params.expert_intermediate = expert_intermediate;
                    local_params.layer_idx = layer_idx;
                    local_params.expert_mask = std::move(participant_mask);
                    local_params.prepared_store = prepared_weight_store_;
                    local_params.physical_memory_authority = prepared_weight_store_
                        ? prepared_weight_store_->physicalMemoryAuthority() : nullptr;
                    local_params.runtime_participant_index =
                        target_participant;
                    const size_t compact_row_capacity =
                        static_cast<size_t>(
                            overlaySparseGraphRowCapacity(
                                config_,
                                target_device,
                                total_tokens));
                    local_params.graph_row_capacity = compact_row_capacity;
                    local_params.serial_compact_buffer_arena =
                        localExpertSerialBufferArenaForParticipant(
                            target_device,
                            target_participant,
                            std::max<size_t>(compact_row_capacity, 1u));
                    local_params.expert_weight_resolution_policy =
                        MoELocalExpertStage::ExpertWeightResolutionPolicy::
                            RegistryOnly;

                    if (model_ctx_)
                    {
                        const auto weight_manager =
                            model_ctx_->concreteWeightManager();
                        if (weight_manager)
                        {
                            auto &registry =
                                weight_manager->expertGemmRegistry();
                            local_params.expert_registry = &registry;
                            if(joint_host) {
                                if(!prepared_weight_store_ || !config_.moe.expert_overlay_cpu_joint_context)
                                    throw std::logic_error("CPU joint continuation follower lost its original source/context");
                                const auto formats=originalCPUExpertWorkspaceSourceFormats(model_ctx_->concreteLoader(),
                                    layer_idx,config_.d_model,expert_intermediate,config_.moe.num_experts);
                                joint_collective=CPUJointHostGraphEndpoint{
                                    *config_.moe.expert_overlay_participant_residency,registry,prepared_weight_store_->modelId(),
                                    config_.moe.expert_overlay_cpu_joint_context,
                                    {.gate=formats[0],.up=formats[1],.down=formats[2],
                                     .workers=omp_in_parallel()?omp_get_num_threads():omp_get_max_threads(),.execution=CPUExecutionGeometry::local()},
                                    total_tokens,prefix+"moe_cpu_joint_host_return_p"+std::to_string(target_participant)}.bind(local_params);
                            }
                            else (void)registry
                                .populateExpertEnginesForParticipant(
                                    tier.domain,
                                    target_device,
                                    participant->world_rank,
                                    participant
                                        ->domain_participant_index,
                                    layer_idx,
                                    config_.moe.num_experts,
                                    local_params.prepared_gate_gemm,
                                    local_params.prepared_up_gemm,
                                    local_params.prepared_down_gemm);
                        }
                    }
                    if (!local_params.expert_registry ||
                        (!joint_host && !MoELocalExpertStage::prepareExpertGemmEngines(
                            local_params)))
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE rank-batch follower has incomplete prepared expert engines for layer " +
                            std::to_string(layer_idx) + " participant " +
                            std::to_string(target_participant));
                    }

                    if (!joint_host && config_.moe.expert_overlay_participant_residency)
                    {
                        auto endpoint = config_.moe
                                            .expert_overlay_participant_residency
                                            ->endpoint(target_participant);
                        if (!endpoint)
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE rank-batch follower has no prepared residency endpoint for participant " +
                                std::to_string(target_participant));
                        }
                        std::vector<MoEOverlayPreparedExpertPayload>
                            prepared_payloads;
                        std::string residency_error;
                        if (!resolveMoEOverlayPreparedExpertPayloads(
                                *local_params.expert_registry,
                                *participant,
                                layer_idx,
                                config_.moe.num_experts,
                                local_params.expert_mask,
                                prepared_payloads,
                                &residency_error) ||
                            !config_.moe
                                 .expert_overlay_participant_residency
                                 ->registerInitialLayer(
                                     target_participant,
                                     layer_idx,
                                     local_params.expert_mask,
                                     prepared_payloads,
                                     &residency_error))
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE rank-batch follower could not publish prepared residency for layer " +
                                std::to_string(layer_idx) + " participant " +
                                std::to_string(target_participant) + ": " +
                                residency_error);
                        }
                        local_params.overlay_participant_residency =
                            std::move(endpoint);
                    }

                    const std::string local_name =
                        prefix + "moe_rank_batch_target_" +
                        nodeSuffixForTier(
                            tier,
                            static_cast<int>(tier_index)) +
                        "_p" + std::to_string(target_participant) +
                        "_local_expert";
                    graph.addNode(
                        local_name,
                        ComputeStageFactory::createMoELocalExpert(
                            local_params),
                        target_device);
                    graph.addDependency(local_name, dispatch_node);
                    if(joint_collective) {
                        const auto collective_name=local_name+"_rooted_fragment_combine";
                        graph.addNode(collective_name,std::move(joint_collective),DeviceId::cpu());
                        graph.addDependency(collective_name,local_name);
                        return collective_name;
                    }
                    return local_name;
                };

                for (size_t tier_index = 0; tier_index < overlay_plan->routed_tiers.size(); ++tier_index)
                {
                    const std::string rank_batch_entry = last_return_reduce;
                    std::vector<MoEOverlayRankBatchGraphLane> rank_batch_lanes;
                    std::vector<MoEOverlayLocalExpertGraphLane> rank_local_lanes;
                    const auto &tier = overlay_plan->routed_tiers[tier_index];
                    auto tier_mask = expertMaskForTier(*overlay_placement,
                                                       config_.moe.num_experts,
                                                       static_cast<int>(tier_index));
                    if (!hasActiveExpertMask(tier_mask) &&
                        !use_mapped_activation_parent)
                    {
                        LOG_DEBUG("[Qwen35MoEGraph] Skipping inactive MoE expert overlay tier "
                                  << tier.name << " for layer " << layer_idx);
                        continue;
                    }

                    const auto tier_targets =
                        resolve_overlay_tier_targets(tier_index);
                    std::vector<int> target_participants =
                        tier_targets.participants;
                    const bool compute_apportioned_tier_on_graph_local_participant =
                        tier_targets.graph_local_apportioned;

                    if (distributed_overlay && !target_participants.empty())
                    {
                        const auto *continuation_root_descriptor =
                            owner_map_lifetime->participantForId(
                                continuation_root_participant);
                        if (!continuation_root_descriptor ||
                            !continuation_root_descriptor->world_rank_known)
                        {
                            throw std::logic_error(
                                "Qwen35 MoE rank-batch graph cannot resolve its authenticated continuation participant");
                        }
                        const int source_world_rank =
                            continuation_root_descriptor->world_rank;
                        const int domain_ordinal =
                            routed_domain_ordinal_for_tier(tier);

                        if (sparse_graph_contract.isRankBatchTarget())
                        {
                            if (cpu_llep_state)
                            {
                                throw std::logic_error(
                                    "Qwen35 MoE full-model rank-batch followers cannot enter the unfinished rank-local CPU LLEP child protocol");
                            }
                            const int target_world_rank =
                                config_.moe.overlay_mpi_ctx->rank();
                            std::vector<int> local_participants;
                            std::sort(
                                target_participants.begin(),
                                target_participants.end());
                            for (const int participant_id :
                                 target_participants)
                            {
                                const auto *participant =
                                    owner_map_lifetime->participantForId(
                                        participant_id);
                                const auto participant_mask =
                                    owner_map_lifetime
                                        ->expertMaskForParticipant(
                                            layer_idx,
                                            participant_id,
                                            config_.moe.num_experts);
                                if (participant &&
                                    participant->world_rank_known &&
                                    participant->world_rank ==
                                        target_world_rank &&
                                    hasActiveExpertMask(participant_mask))
                                {
                                    local_participants.push_back(
                                        participant_id);
                                }
                            }
                            if (local_participants.empty())
                                continue;

                            auto transport =
                                overlayRankBatchTransportForGroup(
                                    device,
                                    static_cast<int>(tier_index),
                                    domain_ordinal,
                                    source_world_rank,
                                    target_world_rank,
                                    local_participants,
                                    *owner_map_lifetime,
                                    continuation_root_participant);
                            const auto make_rank_batch_key =
                                [&](MoEOverlayCollectiveDirection direction)
                            {
                                if (mtp_sidecar_context)
                                {
                                    return makeMTPMoEOverlayRankBatchKey(
                                        1,
                                        0,
                                        mtp_depth_idx,
                                        layer_idx,
                                        static_cast<int>(tier_index),
                                        domain_ordinal,
                                        source_world_rank,
                                        target_world_rank,
                                        direction);
                                }
                                return makeMoEOverlayRankBatchKey(
                                    1,
                                    0,
                                    ExpertHistogramSource::DecodeToken,
                                    layer_idx,
                                    static_cast<int>(tier_index),
                                    domain_ordinal,
                                    source_world_rank,
                                    target_world_rank,
                                    direction);
                            };

                            std::vector<std::shared_ptr<
                                MoEOverlaySparseRows>> dispatch_inbound;
                            std::vector<std::shared_ptr<
                                MoEOverlayReturnRows>> local_outputs;
                            dispatch_inbound.reserve(
                                local_participants.size());
                            local_outputs.reserve(
                                local_participants.size());
                            for (const int participant_id :
                                 local_participants)
                            {
                                const auto &workspace =
                                    participant_workspaces
                                        [static_cast<size_t>(
                                            participant_id)];
                                dispatch_inbound.push_back(
                                    std::make_shared<
                                        MoEOverlaySparseRows>(
                                        transport->hasSharedRowStorage()
                                            ? transport
                                                  ->sharedDispatchRows(
                                                      participant_id)
                                            : workspace->dispatchReceive(
                                                  layer_idx,
                                                  static_cast<int>(
                                                      tier_index))));
                                local_outputs.push_back(
                                    std::make_shared<
                                        MoEOverlayReturnRows>(
                                        transport->hasSharedRowStorage()
                                            ? transport
                                                  ->sharedReturnRows(
                                                      participant_id)
                                            : workspace->localExpertOutput(
                                                  layer_idx,
                                                  static_cast<int>(
                                                      tier_index))));
                            }

                            MoERankBatchDispatchStage::Params
                                follower_dispatch;
                            follower_dispatch.device_id = DeviceId::cpu();
                            follower_dispatch.endpoint_role =
                                MoERankBatchEndpointRole::RemoteTarget;
                            follower_dispatch.transport = transport;
                            follower_dispatch.key = make_rank_batch_key(
                                MoEOverlayCollectiveDirection::Dispatch);
                            follower_dispatch.source_participant =
                                continuation_root_participant;
                            follower_dispatch.participant_ids =
                                local_participants;
                            follower_dispatch.inbound_rows =
                                dispatch_inbound;
                            follower_dispatch.seq_len = total_tokens;
                            follower_dispatch.top_k = config_.moe.top_k;
                            follower_dispatch.d_model = config_.d_model;
                            follower_dispatch.tier_index =
                                static_cast<int>(tier_index);

                            const std::string group_suffix =
                                nodeSuffixForTier(
                                    tier,
                                    static_cast<int>(tier_index)) +
                                "_rank" +
                                std::to_string(target_world_rank);
                            const std::string dispatch_name =
                                prefix +
                                "moe_rank_batch_target_dispatch_" +
                                group_suffix;
                            graph.addNode(
                                dispatch_name,
                                ComputeStageFactory::
                                    createMoERankBatchDispatch(
                                        follower_dispatch),
                                DeviceId::cpu());
                            graph.addDependency(
                                dispatch_name,
                                last_return_reduce.empty()
                                    ? dispatch_dependency
                                    : last_return_reduce);

                            std::vector<std::string> local_nodes;
                            local_nodes.reserve(
                                local_participants.size());
                            for (size_t participant_index = 0;
                                 participant_index <
                                 local_participants.size();
                                 ++participant_index)
                            {
                                local_nodes.push_back(
                                    add_rank_batch_target_local_expert(
                                        tier_index,
                                        local_participants[
                                            participant_index],
                                        dispatch_inbound[
                                            participant_index],
                                        local_outputs[
                                            participant_index],
                                        dispatch_name));
                            }

                            std::vector<std::shared_ptr<const
                                MoEOverlayReturnRows>> outbound_rows;
                            outbound_rows.reserve(local_outputs.size());
                            for (const auto &rows : local_outputs)
                                outbound_rows.push_back(rows);

                            MoERankBatchReturnReduceStage::Params
                                follower_return;
                            follower_return.device_id = DeviceId::cpu();
                            follower_return.endpoint_role =
                                MoERankBatchEndpointRole::RemoteTarget;
                            follower_return.transport = transport;
                            follower_return.key = make_rank_batch_key(
                                MoEOverlayCollectiveDirection::ReturnReduce);
                            follower_return.continuation_participant =
                                continuation_root_participant;
                            follower_return.participant_ids =
                                local_participants;
                            follower_return.outbound_rows =
                                std::move(outbound_rows);
                            follower_return.return_layout = host_return_layout;
                            follower_return.seq_len = total_tokens;
                            follower_return.d_model = config_.d_model;

                            const std::string return_name =
                                prefix +
                                "moe_rank_batch_target_return_" +
                                group_suffix;
                            graph.addNode(
                                return_name,
                                ComputeStageFactory::
                                    createMoERankBatchReturnReduce(
                                        follower_return),
                                DeviceId::cpu());
                            for (const auto &local_node : local_nodes)
                            {
                                graph.addDependency(
                                    return_name,
                                    local_node);
                            }
                            last_return_reduce = return_name;
                            continue;
                        }

                        if (!sparse_graph_contract
                                 .ownsDispatchAuthority() ||
                            source_world_rank !=
                                config_.moe.overlay_mpi_ctx->rank())
                        {
                            throw std::logic_error(
                                "Qwen35 MoE rank-batch source role does not own the authenticated continuation participant");
                        }
                        if (cpu_llep_state)
                        {
                            throw std::logic_error(
                                "Qwen35 MoE rank-batch source cannot be combined with the unfinished rank-local CPU LLEP child protocol");
                        }

                        /*
                         * One remote rank may own several logical GPU/CPU
                         * participants. Keep those participant packets
                         * separate, but move them in one authenticated message.
                         * Portable rank batches omit empty initial masks
                         * symmetrically. Mapped retained epochs bind every
                         * topology-declared lane so later migration can make an
                         * initially empty participant live without recapture.
                         */
                        std::map<int, std::vector<int>> participants_by_rank;
                        std::vector<int> canonical_remote_participants;
                        std::vector<int> local_rank_participants;
                        std::sort(
                            target_participants.begin(),
                            target_participants.end());
                        for (const int target_participant : target_participants)
                        {
                            const auto participant_mask =
                                owner_map_lifetime->expertMaskForParticipant(
                                    layer_idx,
                                    target_participant,
                                    config_.moe.num_experts);
                            if (!use_mapped_activation_parent &&
                                !hasActiveExpertMask(participant_mask))
                                continue;
                            const auto *participant =
                                owner_map_lifetime->participantForId(
                                    target_participant);
                            if (!participant ||
                                !participant->world_rank_known)
                            {
                                throw std::logic_error(
                                    "Qwen35 MoE distributed sparse tier contains an unresolved participant endpoint");
                            }
                            if (participant->world_rank == source_world_rank && participant->device.is_cpu())
                            {
                                /* CPU service stays in the original local sparse
                                 * boundary. Same-rank foreign GPUs instead use
                                 * their real retained endpoint channel; they
                                 * never become continuation TP participants. */
                                local_rank_participants.push_back(
                                    target_participant);
                                continue;
                            }
                            participants_by_rank[participant->world_rank]
                                .push_back(target_participant);
                            canonical_remote_participants.push_back(
                                target_participant);
                        }

                        std::vector<std::pair<int, std::vector<int>>> rank_groups;
                        rank_groups.reserve(participants_by_rank.size());
                        for (auto &[target_rank, participants] :
                             participants_by_rank)
                        {
                            std::sort(participants.begin(), participants.end());
                            rank_groups.emplace_back(
                                target_rank, std::move(participants));
                        }
                        std::sort(
                            rank_groups.begin(),
                            rank_groups.end(),
                            [](const auto &lhs, const auto &rhs)
                            {
                                return lhs.second.front() < rhs.second.front();
                            });
                        std::vector<int> batched_participant_order;
                        for (const auto &group : rank_groups)
                        {
                            batched_participant_order.insert(
                                batched_participant_order.end(),
                                group.second.begin(),
                                group.second.end());
                        }
                        if (batched_participant_order !=
                            canonical_remote_participants)
                        {
                            throw std::logic_error(
                                "Qwen35 MoE cannot rank-batch an interleaved participant order without changing canonical FP32 accumulation order");
                        }

                        // Previous tiers may own the shared return destination.
                        // Freeze that entry edge: a later peer in this tier must
                        // never inherit the preceding peer's blocking return.
                        for (const auto &[target_world_rank, participants] :
                             rank_groups)
                        {
                            auto transport =
                                overlayRankBatchTransportForGroup(
                                    device,
                                    static_cast<int>(tier_index),
                                    domain_ordinal,
                                    source_world_rank,
                                    target_world_rank,
                                    participants,
                                    *owner_map_lifetime,
                                    continuation_root_participant);
                            const std::string group_suffix =
                                nodeSuffixForTier(
                                    tier,
                                    static_cast<int>(tier_index)) +
                                "_rank" +
                                std::to_string(target_world_rank);

                            if (use_mapped_activation_parent)
                            {
                                auto *const mapped_transport = dynamic_cast<
                                    IMoEOverlayMappedActivationTransport *>(
                                    transport.get());
                                if (!mapped_transport ||
                                    transport->kind() !=
                                        MoEOverlayRankBatchTransportKind::
                                            NodeLocalSharedRows ||
                                    mapped_activation_graph_family_ordinal >=
                                        mapped_transport
                                            ->activationGraphFamilyCount())
                                {
                                    throw std::runtime_error(
                                        "Qwen35 MoE node-local activation plan did not produce its required mapped graph capability");
                                }

                                const std::uint32_t stage_ordinal =
                                    mapped_transport->activationStageOrdinal(
                                        mapped_activation_graph_family_ordinal,
                                        layer_idx);
                                for (const int participant : participants)
                                {
                                    auto lane =
                                        mapped_transport->activationDeviceLane(
                                            participant,
                                            mapped_activation_graph_family_ordinal,
                                            device);
                                    if (!lane.valid() ||
                                        lane.target_participant_id !=
                                            participant ||
                                        lane.graph_family_ordinal !=
                                            mapped_activation_graph_family_ordinal)
                                    {
                                        throw std::runtime_error(
                                            "Qwen35 MoE mapped activation lane diverged from the planner participant or graph family");
                                    }

                                    const std::string packet_name =
                                        prefix +
                                        "moe_overlay_activation_dispatch_pack_" +
                                        group_suffix + "_p" +
                                        std::to_string(participant);
                                    mapped_activation_return_bindings.push_back(
                                        MappedActivationReturnBinding{
                                            .dispatch_node_name = packet_name,
                                            .node_name =
                                                prefix +
                                                "moe_overlay_activation_return_consume_" +
                                                group_suffix + "_p" +
                                                std::to_string(participant),
                                            .lane = std::move(lane),
                                            .stage_ordinal = stage_ordinal,
                                        });
                                }
                                continue;
                            }

                            ensure_host_dispatch_path();
                            std::vector<std::shared_ptr<
                                MoEOverlayCollectiveWorkspace>>
                                group_workspaces;
                            group_workspaces.reserve(participants.size());
                            for (const int participant : participants)
                            {
                                group_workspaces.push_back(
                                    participant_workspaces
                                        [static_cast<size_t>(participant)]);
                            }

                            const auto make_rank_batch_key =
                                [&](MoEOverlayCollectiveDirection direction)
                            {
                                if (mtp_sidecar_context)
                                {
                                    return makeMTPMoEOverlayRankBatchKey(
                                        1,
                                        0,
                                        mtp_depth_idx,
                                        layer_idx,
                                        static_cast<int>(tier_index),
                                        domain_ordinal,
                                        source_world_rank,
                                        target_world_rank,
                                        direction);
                                }
                                return makeMoEOverlayRankBatchKey(
                                    1,
                                    0,
                                    ExpertHistogramSource::DecodeToken,
                                    layer_idx,
                                    static_cast<int>(tier_index),
                                    domain_ordinal,
                                    source_world_rank,
                                    target_world_rank,
                                    direction);
                            };

                            MoERankBatchDispatchStage::Params batch_dispatch;
                            batch_dispatch.device_id = DeviceId::cpu();
                            batch_dispatch.endpoint_role =
                                MoERankBatchEndpointRole::ContinuationSource;
                            batch_dispatch.transport = transport;
                            batch_dispatch.key = make_rank_batch_key(
                                MoEOverlayCollectiveDirection::Dispatch);
                            batch_dispatch.source_participant =
                                continuation_root_participant;
                            batch_dispatch.participant_ids = participants;
                            if (!transport->hasSharedRowStorage())
                            {
                                batch_dispatch.participant_workspaces =
                                    group_workspaces;
                            }
                            batch_dispatch.dispatch_output =
                                dispatch_output_lifetime;
                            batch_dispatch.ticket_storage =
                                dispatch_ticket_storage;
                            if (!dispatch_ticket_storage)
                            {
                                batch_dispatch.hidden = buffers.normalized;
                                batch_dispatch.routing_indices =
                                    routing_indices;
                                batch_dispatch.routing_weights =
                                    routing_weights;
                                batch_dispatch.hidden_buffer_id =
                                    buffers.idFor(BufferId::NORMALIZED);
                                batch_dispatch.routing_indices_buffer_id =
                                    buffers.idFor(
                                        BufferId::MOE_EXPERT_INDICES);
                                batch_dispatch.routing_weights_buffer_id =
                                    buffers.idFor(
                                        BufferId::MOE_EXPERT_WEIGHTS);
                            }
                            batch_dispatch.seq_len = total_tokens;
                            batch_dispatch.top_k = config_.moe.top_k;
                            batch_dispatch.d_model = config_.d_model;
                            batch_dispatch.tier_index =
                                static_cast<int>(tier_index);

                            const std::string batch_dispatch_name =
                                prefix + "moe_rank_batch_dispatch_" +
                                group_suffix;
                            graph.addNode(
                                batch_dispatch_name,
                                ComputeStageFactory::
                                    createMoERankBatchDispatch(
                                        batch_dispatch),
                                DeviceId::cpu());
                            graph.addDependency(
                                batch_dispatch_name,
                                dispatch_dependency);
                            distributed_rank_batch_dispatch_nodes.push_back(
                                batch_dispatch_name);
                            if (!rank_batch_entry.empty())
                            {
                                graph.addDependency(
                                    batch_dispatch_name,
                                    rank_batch_entry);
                            }

                            std::vector<std::shared_ptr<
                                MoEOverlayReturnRows>>
                                return_inbound;
                            return_inbound.reserve(participants.size());
                            for (size_t participant_index = 0;
                                 participant_index < participants.size();
                                 ++participant_index)
                            {
                                const int participant =
                                    participants[participant_index];
                                return_inbound.push_back(
                                    std::make_shared<MoEOverlayReturnRows>(
                                        transport->hasSharedRowStorage()
                                            ? transport->sharedReturnRows(
                                                  participant)
                                            : group_workspaces
                                                  [participant_index]
                                                      ->returnReceive(
                                                          layer_idx,
                                                          static_cast<int>(
                                                              tier_index))));
                            }

                            const bool is_final_ordered_return =
                                final_host_sparse_return.has_value() &&
                                final_host_sparse_return->tier_index ==
                                    tier_index &&
                                std::find(
                                    participants.begin(),
                                    participants.end(),
                                    final_host_sparse_return
                                        ->participant_id) !=
                                    participants.end();
                            MoERankBatchReturnReduceStage::Params batch_return;
                            batch_return.device_id = DeviceId::cpu();
                            batch_return.endpoint_role =
                                MoERankBatchEndpointRole::ContinuationSource;
                            batch_return.transport = transport;
                            batch_return.key = make_rank_batch_key(
                                MoEOverlayCollectiveDirection::ReturnReduce);
                            batch_return.continuation_participant =
                                continuation_root_participant;
                            batch_return.participant_ids = participants;
                            batch_return.inbound_rows =
                                std::move(return_inbound);
                            batch_return.ticket_storage =
                                dispatch_ticket_storage;
                            batch_return.return_layout = host_return_layout;
                            if (!dispatch_ticket_storage)
                            {
                                batch_return.dense_output = host_return_layout == MoEOverlayReturnLayout::CanonicalExpertRoutes
                                    ? canonical_route_contributions : moe_output;
                                batch_return.dense_output_buffer_id =
                                    buffers.idFor(
                                        host_return_layout == MoEOverlayReturnLayout::CanonicalExpertRoutes
                                            ? BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS : BufferId::MOE_COMBINED_OUTPUT);
                            }
                            batch_return.seq_len = total_tokens;
                            batch_return.d_model = config_.d_model;
                            batch_return.clear_output_before_scatter =
                                first_return_scatter;
                            batch_return.publish_ticket_completion =
                                dispatch_ticket_storage &&
                                is_final_ordered_return;
                            batch_return.dispatch_output =
                                dispatch_output_lifetime;
                            batch_return.residency_lease_terminal =
                                moeOverlayHostDispatchLeaseTerminal(
                                    host_dispatch_lease_owner,
                                    is_final_ordered_return);

                            const std::string batch_return_name =
                                prefix + "moe_rank_batch_return_reduce_" +
                                group_suffix;
                            graph.addNode(
                                batch_return_name,
                                ComputeStageFactory::
                                    createMoERankBatchReturnReduce(
                                        batch_return),
                                DeviceId::cpu());
                            graph.addDependency(
                                batch_return_name,
                                batch_dispatch_name);
                            rank_batch_lanes.push_back({
                                .dispatch = batch_dispatch_name,
                                .returned = batch_return_name,
                            });

                            std::string tier_terminal = batch_return_name;
                            if (dispatch_ticket_storage &&
                                is_final_ordered_return)
                            {
                                MoEOverlayTicketConsumeStage::Params
                                    consume_params;
                                consume_params.device_id = device;
                                consume_params.output =
                                    captured_overlay_continuation
                                        ? buffers.attn_proj
                                        : moe_output;
                                consume_params.output_buffer_id =
                                    captured_overlay_continuation
                                        ? buffers.idFor(BufferId::ATTN_PROJ)
                                        : buffers.idFor(
                                              BufferId::MOE_COMBINED_OUTPUT);
                                consume_params.layer_idx = layer_idx;
                                consume_params.bucket_rows = total_tokens;
                                consume_params.d_model = config_.d_model;
                                consume_params.ticket_storage =
                                    dispatch_ticket_storage;

                                const std::string consume_name =
                                    prefix +
                                    "moe_overlay_ticket_consume_" +
                                    group_suffix;
                                graph.addNode(
                                    consume_name,
                                    ComputeStageFactory::
                                        createMoEOverlayTicketConsume(
                                            consume_params),
                                    device);
                                graph.addDependency(
                                    consume_name,
                                    batch_return_name);
                                if (captured_overlay_continuation)
                                {
                                    graph.setGraphCaptureWaveContract(
                                        consume_name,
                                        GraphCaptureWaveContract{
                                            .identity =
                                                captured_overlay_inbound_wave,
                                        });
                                }
                                tier_terminal = consume_name;
                            }

                            last_return_reduce = tier_terminal;
                            first_return_scatter = false;
                        }
                        if (local_rank_participants.empty())
                        {
                            if (!rank_batch_lanes.empty())
                                wireMoEOverlayRankBatchForkJoin(graph, rank_batch_lanes);
                            continue;
                        }

                        /*
                         * Remote rank batches and mapped lanes have now been
                         * declared. Fall through with only colocated endpoints
                         * and bind them to the allocation-free rank-local
                         * context. This includes the continuation's own
                         * participant loopback and colocated lower-tier peers.
                         * Their host dispatch branch shares the immutable
                         * ticket publication fork with continuation GPU work.
                         */
                        target_participants =
                            std::move(local_rank_participants);
                        ensure_host_dispatch_path();
                        ensure_rank_local_collective_context();
                        collective_context_lifetime =
                            rank_local_collective_context_lifetime;
                    }

                    for (const int target_participant : target_participants)
                    {
                        auto participant_mask =
                            owner_map_lifetime->expertMaskForParticipant(
                                layer_idx,
                                target_participant,
                                config_.moe.num_experts);
                        if (!hasActiveExpertMask(participant_mask) &&
                            !(distributed_overlay &&
                              use_mapped_activation_parent))
                            continue;

                        const DeviceId target_device = participantDeviceForGraphNativeOverlay(
                            *owner_map_lifetime,
                            target_participant);
                        const bool direct_rank_local_protocol =
                            rank_local_collective_context_lifetime &&
                            collective_context_lifetime.get() ==
                                rank_local_collective_context_lifetime.get();
                        if (!distributed_overlay &&
                            target_device.is_gpu() &&
                            target_device != device)
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE graph-native overlay would execute GPU expert participant " +
                                std::to_string(target_participant) + " on " +
                                target_device.to_string() + " from graph device " +
                                device.to_string() +
                                ". LocalTP expert-ID-apportioned GPU domains must lower only the graph-local participant; "
                                "other cross-device GPU expert execution requires a real participant/domain executor.");
                        }
                        const std::string participant_suffix =
                            nodeSuffixForTier(tier, static_cast<int>(tier_index)) +
                            "_p" + std::to_string(target_participant);

                        auto target_dispatch_inbound = std::make_shared<MoEOverlaySparseRows>(
                            direct_rank_local_protocol && target_participant == continuation_root_participant
                                ? participant_workspaces[static_cast<size_t>(target_participant)]->localExpertInput(
                                      layer_idx, static_cast<int>(tier_index))
                                : participant_workspaces[static_cast<size_t>(target_participant)]->dispatchReceive(
                                      layer_idx, static_cast<int>(tier_index)));
                        const MoEOverlayCollectiveKey dispatch_key = graphNativeMoEKey(
                            layer_idx,
                            static_cast<int>(tier_index),
                            target_participant,
                            MoEOverlayCollectiveDirection::Dispatch,
                            mtp_sidecar_context,
                            mtp_depth_idx);

                        std::string previous_dispatch_node;
                        std::string target_dispatch_node;
                        const std::vector<int> dispatch_sources =
                            distributed_overlay || direct_rank_local_protocol
                                ? std::vector<int>{
                                      continuation_root_participant}
                                : participantsWithLast(
                                      participant_count,
                                      target_participant);
                        for (const int source_participant : dispatch_sources)
                        {
                            /*
                             * In a distributed overlay every rank enters the
                             * same sparse collective, but only the rank that
                             * owns target_participant receives its packet.
                             * The receiving local-expert stage always reads
                             * target_dispatch_inbound, so the collective must
                             * publish into that target-owned view on every
                             * rank.  Publishing into the source workspace
                             * would leave remote owners with an empty input
                             * even though MPI delivered their routes.
                             *
                             * The single-process protocol retains its
                             * source-scoped scratch views because it models
                             * every logical participant in one graph.
                             */
                            auto inbound_lifetime =
                                distributed_overlay ||
                                        direct_rank_local_protocol ||
                                        source_participant == target_participant
                                    ? target_dispatch_inbound
                                    : std::make_shared<MoEOverlaySparseRows>(
                                          participant_workspaces
                                              [static_cast<size_t>(
                                                  source_participant)]
                                                  ->dispatchReceive(
                                                      layer_idx,
                                                      static_cast<int>(
                                                          tier_index)));

                            MoESparseDispatchStage::Params sparse_dispatch_params;
                            sparse_dispatch_params.device_id = DeviceId::cpu();
                            sparse_dispatch_params.collective_context_lifetime = collective_context_lifetime;
                            sparse_dispatch_params.workspace_lifetime = participant_workspaces[static_cast<size_t>(source_participant)];
                            sparse_dispatch_params.key = dispatch_key;
                            sparse_dispatch_params.source_participant = source_participant;
                            sparse_dispatch_params.target_participant = target_participant;
                            sparse_dispatch_params.seq_len = total_tokens;
                            sparse_dispatch_params.top_k = config_.moe.top_k;
                            sparse_dispatch_params.d_model = config_.d_model;
                            sparse_dispatch_params.tier_index = static_cast<int>(tier_index);
                            const bool publishes_dispatch_payload =
                                sparse_graph_contract
                                    .ownsDispatchAuthority() &&
                                source_participant ==
                                    continuation_root_participant;
                            sparse_dispatch_params.payload_publication_role =
                                publishes_dispatch_payload
                                    ? MoESparseDispatchStage::
                                          PayloadPublicationRole::
                                              PayloadAuthority
                                    : MoESparseDispatchStage::
                                          PayloadPublicationRole::
                                              EmptyCollectiveParticipant;
                            sparse_dispatch_params.replicated_hidden_export = true;
                            sparse_dispatch_params.logical_continuation_root_participant = continuation_root_participant;
                            sparse_dispatch_params.manual_boundary_requires_collective_completion =
                                distributed_overlay ||
                                direct_rank_local_protocol ||
                                source_participant == target_participant;
                            /*
                             * Independent rank graphs do not share stage-object
                             * lifetime. Require the runner's request/chunk
                             * identity instead of deriving a sparse MPI key
                             * from a local execution counter.
                             */
                            sparse_dispatch_params.require_explicit_transaction_identity =
                                distributed_overlay;
                            sparse_dispatch_params.require_explicit_execution_semantics =
                                config_.moe.expert_overlay_participant_residency != nullptr;
                            sparse_dispatch_params.inbound_rows_lifetime = inbound_lifetime;
                            if (publishes_dispatch_payload)
                            {
                                /*
                                 * Ticket ownership follows this exact sparse
                                 * edge, not the graph-wide authority.  A
                                 * rank-local heterogeneous graph models every
                                 * logical source in one ordered collective;
                                 * non-root sources publish authenticated empty
                                 * packets and must never observe or retain the
                                 * root's captured payload ticket.
                                 */
                                sparse_dispatch_params.ticket_storage =
                                    dispatch_ticket_storage;
                                if (!dispatch_ticket_storage)
                                {
                                    sparse_dispatch_params.hidden = buffers.normalized;
                                    sparse_dispatch_params.routing_indices = routing_indices;
                                    sparse_dispatch_params.routing_weights = routing_weights;
                                    sparse_dispatch_params.hidden_buffer_id = buffers.idFor(BufferId::NORMALIZED);
                                    sparse_dispatch_params.routing_indices_buffer_id = buffers.idFor(BufferId::MOE_EXPERT_INDICES);
                                    sparse_dispatch_params.routing_weights_buffer_id = buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS);
                                }
                                sparse_dispatch_params.dispatch_output_lifetime = dispatch_output_lifetime;
                                if (dispatch_ticket_storage)
                                {
                                    sparse_dispatch_params
                                        .ticket_observation_role =
                                        MoESparseDispatchStage::
                                            TicketObservationRole::
                                                MaterializedHostDispatch;
                                }
                            }

                            const std::string sparse_dispatch_name = prefix + "moe_sparse_dispatch_" +
                                                                     participant_suffix +
                                                                     "_from_p" + std::to_string(source_participant);
                            graph.addNode(sparse_dispatch_name,
                                          ComputeStageFactory::createMoESparseDispatch(sparse_dispatch_params),
                                          DeviceId::cpu());
                            if (cpu_llep_state)
                            {
                                cpu_llep_sparse_dispatch_nodes.push_back(
                                    sparse_dispatch_name);
                            }
                            graph.addDependency(sparse_dispatch_name,
                                                previous_dispatch_node.empty()
                                                    ? dispatch_dependency
                                                    : previous_dispatch_node);
                            // True collective lanes retain globally ordered
                            // dispatch/return pairing. A colocated endpoint next
                            // to independent rank-batch channels instead declares
                            // private work in the common fork. The shared helper
                            // joins all returns without serializing CPU sockets.
                            const auto &dispatch_predecessor =
                                direct_rank_local_protocol && !rank_batch_lanes.empty()
                                    ? rank_batch_entry : last_return_reduce;
                            if (!dispatch_predecessor.empty())
                                graph.addDependency(
                                    sparse_dispatch_name,
                                    dispatch_predecessor);
                            previous_dispatch_node = sparse_dispatch_name;
                            if (distributed_overlay ||
                                source_participant == target_participant)
                                target_dispatch_node = sparse_dispatch_name;
                        }

                        auto local_output_lifetime = std::make_shared<MoEOverlayReturnRows>(
                            participant_workspaces[static_cast<size_t>(target_participant)]->localExpertOutput(
                                layer_idx,
                                static_cast<int>(tier_index)));

                        std::shared_ptr<
                            MoEOverlayCanonicalRouteReturnTicketStorage>
                            canonical_route_ticket_storage;

                        const bool owns_target_participant =
                            !distributed_overlay ||
                            owns_overlay_participant(
                                target_participant);
                        std::string local_name;
                        std::string local_collective_name;
                        const bool joint_local_target = target_device.is_cpu() &&
                            config_.moe.expert_overlay_participant_residency &&
                            config_.moe.expert_overlay_participant_residency->isJointParticipant(target_participant);
                        std::unique_ptr<MoECPUExpertShardRouteStage> host_joint_collective;
                        if (owns_target_participant)
                        {
                            MoELocalExpertStage::Params local_params;
                            local_params.device_id = target_device;
                            local_params.input_rows_lifetime = target_dispatch_inbound;
                            local_params.output_rows_lifetime = local_output_lifetime;
                            local_params.return_layout = host_return_layout;
                            local_params.workspace_lifetime =
                                participant_workspaces[static_cast<size_t>(
                                    target_participant)];
                            local_params.gate_exps = layer.moe_gate_exps;
                            local_params.up_exps = layer.moe_up_exps;
                            local_params.down_exps = layer.moe_down_exps;
                            local_params.num_experts = config_.moe.num_experts;
                            local_params.top_k = config_.moe.top_k;
                            local_params.d_model = config_.d_model;
                            local_params.expert_intermediate = expert_intermediate;
                            local_params.layer_idx = layer_idx;
                            local_params.expert_mask = std::move(participant_mask);
                            local_params.prepared_store = prepared_weight_store_;
                            local_params.physical_memory_authority = prepared_weight_store_
                                ? prepared_weight_store_->physicalMemoryAuthority() : nullptr;
                            local_params.runtime_participant_index = target_participant;
                            if (target_device.is_gpu() &&
                                config_.moe.rebalance_config.mode ==
                                    MoERebalanceRuntimeMode::Dynamic)
                            {
                                /*
                                 * The retained sparse child executes from its
                                 * epoch-indexed prepared residency bank. Give
                                 * it only the canonical table's observation
                                 * capability: lending the complete runtime
                                 * table here would create a second placement
                                 * authority inside the child executor.
                                 */
                                if (target_device != device ||
                                    !moe_runtime_table)
                                {
                                    throw std::logic_error(
                                        "Qwen35 MoE Dynamic sparse GPU endpoint has no graph-local canonical runtime table for participant " +
                                        std::to_string(target_participant));
                                }
                                local_params.overlay_service_telemetry =
                                    moe_runtime_table
                                        ->deviceOverlayServiceTelemetryBinding(
                                            layer_idx);
                                /* Unsampled exact-equivalent layers deliberately
                                 * bind no marker pair. The shared economy catalog
                                 * pools the bounded stratified sample before
                                 * expanding its service cost during certification. */
                            }
                            /*
                             * This participant-local child belongs to one
                             * immutable graph family. Name its economic phase
                             * from that graph's semantic role, never from M:
                             * fixed-depth MTP drafts and prefix-restore
                             * conditions are one-row graphs but remain routed
                             * work inside an MTP transaction.
                             */
                            local_params.service_phase = routed_service_phase;
                            /*
                             * The host packet arena is capacity-wide and shared
                             * by the serial graph family. Compact device tensors
                             * must still select this graph's live row bucket;
                             * inheriting the host arena capacity would make a
                             * one-token decode graph pay the prefill allocation
                             * and coherence footprint.
                             */
                            const size_t compact_row_capacity =
                                static_cast<size_t>(
                                    overlaySparseGraphRowCapacity(
                                        config_,
                                        target_device,
                                        total_tokens));
                            local_params.graph_row_capacity =
                                compact_row_capacity;
                            bool joint_contributor = false;
                            if (joint_local_target && device.is_gpu())
                            {
                                const auto memory = prepared_weight_store_
                                    ? prepared_weight_store_->physicalMemoryAuthority() : nullptr;
                                if (!memory)
                                    throw std::logic_error("CPU joint graph lost its original physical authority");
                                const auto &admission = requireCPUSerialExpertStorageAdmission(
                                    config_.cpu_serial_expert_storage, {memory->worldRank(), target_device},
                                    target_participant, device);
                                const auto &joint = admission.envelope->admitted()->jointRoutes();
                                const auto &context = config_.moe.expert_overlay_cpu_joint_context;
                                if (!joint || !context || context->myIndex() < 0 ||
                                    joint->member() != static_cast<size_t>(context->myIndex()))
                                    throw std::logic_error("CPU joint graph lost its original rooted context");
                                joint_contributor = !joint->isRoot();
                            }
                            local_params.serial_compact_buffer_arena =
                                localExpertSerialBufferArenaForParticipant(
                                    target_device,
                                    target_participant,
                                    std::max<size_t>(
                                        compact_row_capacity,
                                        1u),
                                    captured_overlay_continuation &&
                                            direct_rank_local_protocol &&
                                            target_device.is_cpu() &&
                                            !joint_contributor
                                        ? std::optional<DeviceId>{device}
                                        : std::nullopt);
                            if (captured_overlay_continuation &&
                                direct_rank_local_protocol &&
                                target_device.is_cpu())
                            {
                                if (!canonical_route_contributions ||
                                    !device.is_gpu())
                                {
                                    throw std::logic_error(
                                        "Qwen35 MoE colocated CPU endpoint has no continuation canonical-route bank");
                                }
                                auto contribution_region = joint_contributor
                                    ? nullptr
                                    : local_params.serial_compact_buffer_arena->mappedCPUCanonicalRoutes(device);
                                canonical_route_ticket_storage =
                                    std::make_shared<
                                        MoEOverlayCanonicalRouteReturnTicketStorage>();
                                const auto ticket_intent = forward_input_construction_
                                    ? modelForwardInput().graph_submission_intent
                                    : ForwardGraphSubmissionIntent::Execute;
                                canonical_route_ticket_storage
                                    ->bindForGraphConstruction(
                                        ticket_intent, layer_idx,
                                        static_cast<size_t>(total_tokens) *
                                            static_cast<size_t>(
                                                config_.moe.top_k),
                                        config_.d_model,
                                        device,
                                        /*workspace_generation=*/1u,
                                        std::move(contribution_region),
                                        ticket_intent == ForwardGraphSubmissionIntent::WorkspaceDeclaration
                                            ? nullptr : mappedOverlayTicketArenaForDevice(device),
                                        joint_contributor ? MoEOverlayCanonicalRouteTicketPayload::ContributorCompletion
                                                          : MoEOverlayCanonicalRouteTicketPayload::CanonicalRows);
                                if (!joint_local_target) local_params
                                    .cpu_canonical_route_ticket_return =
                                    MoELocalExpertStage::
                                        CPUCanonicalRouteTicketReturnBinding{
                                            .storage =
                                                canonical_route_ticket_storage,
                                        };
                                has_rank_local_canonical_ticket_returns = true;
                            }
                            local_params.expert_weight_resolution_policy =
                                MoELocalExpertStage::ExpertWeightResolutionPolicy::
                                    RegistryOnly;
                            local_params.runtime_publication_stream =
                                target_device.is_gpu()
                                    ? device_state_publication_stream
                                    : nullptr;
                            if (model_ctx_)
                            {
                            auto weight_mgr = model_ctx_->concreteWeightManager();
                            if (weight_mgr)
                            {
                                auto &registry = weight_mgr->expertGemmRegistry();
                                local_params.expert_registry = &registry;
                                bool populated = false;
                                auto has_active_prepared_engines = [&]() -> bool
                                {
                                    if (local_params.prepared_gate_gemm.size() !=
                                            static_cast<size_t>(config_.moe.num_experts) ||
                                        local_params.prepared_up_gemm.size() !=
                                            static_cast<size_t>(config_.moe.num_experts) ||
                                        local_params.prepared_down_gemm.size() !=
                                            static_cast<size_t>(config_.moe.num_experts))
                                    {
                                        return false;
                                    }

                                    for (int expert = 0; expert < config_.moe.num_experts; ++expert)
                                    {
                                        if (!local_params.expert_mask[static_cast<size_t>(expert)])
                                            continue;
                                        if (!local_params.prepared_gate_gemm[static_cast<size_t>(expert)] ||
                                            !local_params.prepared_up_gemm[static_cast<size_t>(expert)] ||
                                            !local_params.prepared_down_gemm[static_cast<size_t>(expert)])
                                        {
                                            return false;
                                        }
                                        populated = true;
                                    }
                                    return populated;
                                };
                                if(joint_local_target && device.is_cpu()) {
                                    if(cpu_llep_state || !prepared_weight_store_ || !config_.moe.expert_overlay_cpu_joint_context)
                                        throw std::logic_error("CPU joint continuation root lost its original static source/context");
                                    const auto formats=originalCPUExpertWorkspaceSourceFormats(model_ctx_->concreteLoader(),
                                        layer_idx,config_.d_model,expert_intermediate,config_.moe.num_experts);
                                    host_joint_collective=CPUJointHostGraphEndpoint{
                                        *config_.moe.expert_overlay_participant_residency,registry,prepared_weight_store_->modelId(),
                                        config_.moe.expert_overlay_cpu_joint_context,
                                        {.gate=formats[0],.up=formats[1],.down=formats[2],
                                         .workers=omp_in_parallel()?omp_get_num_threads():omp_get_max_threads(),.execution=CPUExecutionGeometry::local()},
                                        total_tokens,prefix+"moe_cpu_joint_host_return_p"+std::to_string(target_participant)}.bind(local_params);
                                }
                                else if (joint_local_target)
                                {
                                    auto residency = config_.moe.expert_overlay_participant_residency;
                                    if (!prepared_weight_store_ || !config_.moe.expert_overlay_cpu_joint_context)
                                        throw std::runtime_error("Qwen35 colocated CPU fragment has no original prepared source/context");
                                    auto frame = adoptMoEOverlayJointGraphDeclaration(*residency,registry,
                                        prepared_weight_store_->modelId(),target_participant,layer_idx,
                                        config_.moe.num_experts,config_.d_model,expert_intermediate);
                                    const auto arena = local_params.serial_compact_buffer_arena;
                                    if (cpu_llep_state)
                                        throw std::logic_error("Qwen35 Static CPU joint fragments do not support current-batch LLEP");
                                    if (!frame || !frame->sourceSelection() ||
                                        !arena || !arena->cpuJointLayout() ||
                                        arena->cpuJointLayout()->isRoot() == joint_contributor ||
                                        !canonical_route_ticket_storage ||
                                        canonical_route_ticket_storage->metadataOnly() != joint_contributor)
                                        throw std::logic_error("Qwen35 colocated CPU fragment requires its original root or contributor return role");
                                    const auto &source = *frame->sourceSelection();
                                    local_params.cpu_joint_layer = frame;
                                    local_params.return_layout = MoEOverlayReturnLayout::ParticipantTokenPartials;
                                    local_params.gate_exps = nullptr;
                                    local_params.up_exps = nullptr;
                                    local_params.down_exps = nullptr;
                                    local_params.expert_intermediate = source.ownership()
                                        .projection(WeightRole::MoEExpertUp).rows;
                                    local_params.cpu_joint_route_producer = MoELocalExpertStage::CPUJointRouteProducerBinding{
                                        .packed_rows = arena->cpuJointPackedRecords(),
                                        .original_route_slots = target_dispatch_inbound->original_route_slots_host,
                                        .compact_route_slots = target_dispatch_inbound->compact_route_slots_host,
                                        .route_slot_capacity = canonical_route_ticket_storage->routeCapacity(),
                                        .deferred_canonical_ticket = canonical_route_ticket_storage};
                                    // The original loader routes main and shared-predictor
                                    // coordinates without a duplicated main-directory entry.
                                    const auto formats = originalCPUExpertWorkspaceSourceFormats(
                                        model_ctx_->concreteLoader(),layer_idx,config_.d_model,
                                        expert_intermediate,config_.moe.num_experts);
                                    local_params.cpu_workspace_source = MoELocalExpertStage::CPUExpertWorkspaceSource{
                                        .gate = formats[0], .up = formats[1], .down = formats[2],
                                        .workers = omp_in_parallel() ? omp_get_num_threads() : omp_get_max_threads(),
                                        .execution = CPUExecutionGeometry::local()};
                                    local_params.prepared_gate_gemm.assign(config_.moe.num_experts, nullptr);
                                    local_params.prepared_up_gemm.assign(config_.moe.num_experts, nullptr);
                                    local_params.prepared_down_gemm.assign(config_.moe.num_experts, nullptr);
                                    local_params.expert_mask.assign(config_.moe.num_experts,false);
                                }
                                if (!joint_local_target) if (const auto *participant = owner_map_lifetime->participantForId(target_participant))
                                {
                                    (void)registry.populateExpertEnginesForParticipant(
                                        tier.domain,
                                        target_device,
                                        participant->world_rank_known ? participant->world_rank : -1,
                                        participant->domain_participant_index,
                                        layer_idx,
                                        config_.moe.num_experts,
                                        local_params.prepared_gate_gemm,
                                        local_params.prepared_up_gemm,
                                        local_params.prepared_down_gemm);
                                    populated = has_active_prepared_engines();
                                }
                                if (!joint_local_target && !populated)
                                {
                                    (void)registry.populateExpertEnginesForDomain(
                                        tier.domain,
                                        target_device,
                                        layer_idx,
                                        config_.moe.num_experts,
                                        local_params.prepared_gate_gemm,
                                        local_params.prepared_up_gemm,
                                        local_params.prepared_down_gemm);
                                }
                            }
                        }
                        /*
                         * A mapped endpoint is graph topology even when the
                         * automatic capacity plan gives it no epoch-one live
                         * experts. Its residency bank and compact arena are
                         * already preallocated; a later migration publishes
                         * engines through that bank. Validate prepared engines
                         * only for experts that are live in the initial mask.
                         */
                        if (!joint_local_target && hasActiveExpertMask(local_params.expert_mask) &&
                            !MoELocalExpertStage::prepareExpertGemmEngines(
                                local_params))
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE graph failed to prepare participant-local "
                                "expert engines for layer " +
                                std::to_string(layer_idx) + " participant " +
                                std::to_string(target_participant) + " on " +
                                target_device.to_string());
                        }

                        if (!joint_local_target && config_.moe.expert_overlay_participant_residency)
                        {
                            if (!local_params.expert_registry)
                            {
                                throw std::runtime_error(
                                    "Qwen35 MoE production ExpertOverlay graph "
                                    "has no model-owned prepared engine registry");
                            }
                            const auto *participant =
                                owner_map_lifetime->participantForId(
                                    target_participant);
                            auto endpoint =
                                config_.moe
                                    .expert_overlay_participant_residency
                                    ->endpoint(target_participant);
                            if (!participant || !endpoint)
                            {
                                throw std::runtime_error(
                                    "Qwen35 MoE graph owns participant p" +
                                    std::to_string(target_participant) +
                                    " but its process-local residency endpoint is missing");
                            }

                            std::vector<
                                MoEOverlayPreparedExpertPayload>
                                prepared_payloads;
                            std::string residency_error;
                            if (!resolveMoEOverlayPreparedExpertPayloads(
                                    *local_params.expert_registry,
                                    *participant,
                                    layer_idx,
                                    config_.moe.num_experts,
                                    local_params.expert_mask,
                                    prepared_payloads,
                                    &residency_error) ||
                                !config_.moe
                                     .expert_overlay_participant_residency
                                     ->registerInitialLayer(
                                         target_participant,
                                         layer_idx,
                                         local_params.expert_mask,
                                         prepared_payloads,
                                         &residency_error))
                            {
                                throw std::runtime_error(
                                    "Qwen35 MoE graph could not install initial "
                                    "prepared residency bank for layer " +
                                    std::to_string(layer_idx) + " participant " +
                                    std::to_string(target_participant) + ": " +
                                    residency_error);
                            }
                            local_params.overlay_participant_residency =
                                std::move(endpoint);
                        }
                        local_params.cpu_current_batch_llep_state = joint_local_target ? nullptr : cpu_llep_state;

                        local_name = prefix + "moe_local_expert_" + participant_suffix;
                        auto local_stage =
                            ComputeStageFactory::createMoELocalExpert(
                                local_params);
                        auto *concrete_local_stage =
                            dynamic_cast<MoELocalExpertStage *>(
                                local_stage.get());
                        if (!concrete_local_stage)
                        {
                            throw std::logic_error(
                                "Qwen35 MoE local expert factory returned an incompatible stage");
                        }
                        if (cpu_llep_state)
                        {
                            if (cpu_llep_local_consumer)
                            {
                                throw std::logic_error(
                                    "Qwen35 MoE CPU LLEP rank graph owns more than one local sparse endpoint");
                            }
                            cpu_llep_local_consumer =
                                concrete_local_stage;
                        }
                        if (target_device.is_gpu())
                        {
                            if (!device_state_publication_stream)
                            {
                                throw std::runtime_error(
                                    "Qwen35 MoE graph-native GPU participant "
                                    "requires an exact model-lifetime stream");
                            }
                            local_stage->setGPUStream(
                                device_state_publication_stream);
                            if (!concrete_local_stage
                                     ->preparePersistentBuffers())
                            {
                                throw std::runtime_error(
                                    "Qwen35 MoE graph-native GPU participant "
                                    "failed persistent sparse-buffer preparation");
                            }
                        }
                        graph.addNode(
                            local_name,
                            std::move(local_stage),
                            target_device);
                        graph.addDependency(local_name,
                                            target_dispatch_node.empty() ? previous_dispatch_node : target_dispatch_node);
                        if (joint_local_target)
                        {
                            const auto arena = local_params.serial_compact_buffer_arena;
                            const auto &geometry = *arena->cpuJointLayout();
                            local_collective_name = local_name + "_rooted_fragment_combine";
                            if(host_joint_collective) graph.addNode(local_collective_name,std::move(host_joint_collective),DeviceId::cpu());
                            else graph.addNode(local_collective_name, std::make_unique<MoECPUExpertShardRouteStage>(
                                MoECPUExpertShardRouteStage::Params{
                                    .original_source = local_params.cpu_joint_layer->sourceSelection(),
                                    .tp = config_.moe.expert_overlay_cpu_joint_context.get(),
                                    .packed_records = arena->cpuJointPackedRecords().get(),
                                    .combined_routes = arena->cpuCanonicalRoutes().get(),
                                    .source_ranges = arena->cpuJointSourceRanges(),
                                    .participant_route_maps = arena->cpuJointRouteMaps(),
                                    .required_route_weights = arena->cpuJointRequiredRoutes(),
                                    .rows = total_tokens, .top_k = config_.moe.top_k, .columns = config_.d_model,
                                    .root_participant = static_cast<int>(geometry.rootMember()),
                                    .stage_name = local_collective_name,
                                    .original_return_rows = local_output_lifetime.get(),
                                    .original_return_route_capacity = canonical_route_ticket_storage->routeCapacity(),
                                    .original_sparse_routes = target_dispatch_inbound.get(),
                                    .original_prepared_layer = local_params.cpu_joint_layer,
                                    .original_tp_owner = config_.moe.expert_overlay_cpu_joint_context,
                                    .original_canonical_ticket = canonical_route_ticket_storage,
                                    .original_continuation_device = device}), DeviceId::cpu());
                            graph.addDependency(local_collective_name, local_name);
                        }
                        }

                        const MoEOverlayCollectiveKey return_key = graphNativeMoEKey(
                            layer_idx,
                            static_cast<int>(tier_index),
                            target_participant,
                            MoEOverlayCollectiveDirection::ReturnReduce,
                            mtp_sidecar_context,
                            mtp_depth_idx);
                        std::string previous_return_node = last_return_reduce;
                        std::string root_return_node;
                        const std::vector<int> return_sources =
                            distributed_overlay || direct_rank_local_protocol
                                ? std::vector<int>{target_participant}
                                : participantsWithLast(
                                      participant_count,
                                      continuation_root_participant);
                        for (const int source_participant : return_sources)
                        {
                            std::shared_ptr<const MoEOverlayReturnRows> outbound_lifetime;
                            if (distributed_overlay ||
                                direct_rank_local_protocol ||
                                source_participant == target_participant)
                            {
                                outbound_lifetime = local_output_lifetime;
                            }
                            else
                            {
                                outbound_lifetime = std::make_shared<MoEOverlayReturnRows>(
                                    participant_workspaces[static_cast<size_t>(source_participant)]->localExpertOutput(
                                        layer_idx,
                                        static_cast<int>(tier_index)));
                            }

                            // A rank-local edge already owns stable host rows.
                            // Alias that participant-private output until the
                            // ordered gather consumes it, rather than copying
                            // the entire route matrix to another local bank.
                            auto inbound_lifetime = direct_rank_local_protocol
                                ? local_output_lifetime
                                : std::make_shared<MoEOverlayReturnRows>(
                                      participant_workspaces[static_cast<size_t>(source_participant)]->returnReceive(
                                          layer_idx, static_cast<int>(tier_index)));

                            MoESparseReturnReduceStage::Params return_params;
                            return_params.device_id = DeviceId::cpu();
                            return_params.collective_context_lifetime = collective_context_lifetime;
                            return_params.workspace_lifetime = participant_workspaces[static_cast<size_t>(source_participant)];
                            return_params.key = return_key;
                            return_params.source_participant = source_participant;
                            return_params.target_participant = continuation_root_participant;
                            return_params.outbound_rows_lifetime = outbound_lifetime;
                            return_params.inbound_rows_lifetime = inbound_lifetime;
                            return_params.return_layout = host_return_layout;
                            if (canonical_route_ticket_storage)
                            {
                                return_params.inbound_consumer_role =
                                    MoESparseReturnReduceStage::
                                        InboundConsumerRole::
                                            CanonicalRouteTicketCompletion;
                                return_params
                                    .canonical_route_ticket_storage =
                                    canonical_route_ticket_storage;
                            }
                            else
                            {
                                return_params.ticket_storage =
                                    dispatch_ticket_storage;
                            }
                            if (!dispatch_ticket_storage &&
                                !canonical_route_ticket_storage)
                            {
                                return_params.dense_output = host_return_layout == MoEOverlayReturnLayout::CanonicalExpertRoutes
                                    ? canonical_route_contributions : moe_output;
                                return_params.dense_output_buffer_id =
                                    buffers.idFor(
                                        host_return_layout == MoEOverlayReturnLayout::CanonicalExpertRoutes
                                            ? BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS : BufferId::MOE_COMBINED_OUTPUT);
                            }
                            return_params.seq_len = total_tokens;
                            return_params.d_model = config_.d_model;
                            return_params.clear_output_before_scatter =
                                !canonical_route_ticket_storage &&
                                sparse_graph_contract
                                    .ownsDispatchAuthority() &&
                                first_return_scatter;
                            return_params.manual_boundary_requires_collective_completion =
                                distributed_overlay ||
                                direct_rank_local_protocol ||
                                source_participant ==
                                    continuation_root_participant;
                            return_params.require_explicit_transaction_identity =
                                distributed_overlay;
                            return_params.require_explicit_execution_semantics =
                                config_.moe.expert_overlay_participant_residency != nullptr;
                            const bool is_final_ordered_return =
                                sparse_graph_contract
                                    .ownsDispatchAuthority() &&
                                final_host_sparse_return.has_value() &&
                                final_host_sparse_return->matches(
                                    tier_index,
                                    target_participant) &&
                                !return_sources.empty() &&
                                source_participant == return_sources.back();
                            return_params.publish_ticket_completion =
                                dispatch_ticket_storage &&
                                !canonical_route_ticket_storage &&
                                is_final_ordered_return;
                            return_params.residency_lease_terminal =
                                moeOverlayHostDispatchLeaseTerminal(
                                    host_dispatch_lease_owner,
                                    is_final_ordered_return);
                            if (return_params.residency_lease_terminal ==
                                MoEOverlayHostDispatchLeaseTerminal::Release)
                            {
                                return_params.dispatch_output_lifetime =
                                    dispatch_output_lifetime;
                            }

                            const std::string return_name = prefix + "moe_sparse_return_reduce_" +
                                                            participant_suffix +
                                                            "_from_p" + std::to_string(source_participant);
                            graph.addNode(return_name,
                                          ComputeStageFactory::createMoESparseReturnReduce(return_params),
                                          DeviceId::cpu());
                            graph.addDependency(
                                return_name,
                                local_name.empty()
                                    ? (target_dispatch_node.empty()
                                           ? previous_dispatch_node
                                           : target_dispatch_node)
                                    : (local_collective_name.empty() ? local_name : local_collective_name));
                            if (!previous_return_node.empty())
                                graph.addDependency(return_name, previous_return_node);
                            previous_return_node = return_name;
                            if (distributed_overlay ||
                                direct_rank_local_protocol ||
                                source_participant ==
                                    continuation_root_participant)
                                root_return_node = return_name;
                        }

                        if (!root_return_node.empty())
                        {
                            if (direct_rank_local_protocol && !rank_batch_lanes.empty())
                                rank_local_lanes.push_back({
                                    .dispatch = target_dispatch_node,
                                    .compute = local_name,
                                    .returned = root_return_node,
                                    .collective = local_collective_name,
                                });
                            std::string tier_terminal = root_return_node;
                            const bool final_ticket_return =
                                final_host_sparse_return.has_value() &&
                                final_host_sparse_return->matches(
                                    tier_index,
                                    target_participant);
                            if (canonical_route_ticket_storage)
                            {
                                /* Defer every GPU ingress until all colocated
                                 * CPU endpoints have completed. This creates
                                 * one manual fan-in followed by one captured
                                 * ingress wave instead of alternating capture
                                 * and host execution per participant. */
                                pending_canonical_ticket_consumes.push_back(
                                    PendingCanonicalTicketConsume{
                                        .participant_suffix =
                                            participant_suffix,
                                        .storage =
                                            canonical_route_ticket_storage,
                                    });
                            }
                            else if (dispatch_ticket_storage &&
                                     final_ticket_return)
                            {
                                MoEOverlayTicketConsumeStage::Params
                                    consume_params;
                                consume_params.device_id = device;
                                consume_params.output =
                                    captured_overlay_continuation
                                        ? buffers.attn_proj
                                        : moe_output;
                                consume_params.output_buffer_id =
                                    captured_overlay_continuation
                                        ? buffers.idFor(BufferId::ATTN_PROJ)
                                        : buffers.idFor(
                                              BufferId::MOE_COMBINED_OUTPUT);
                                consume_params.layer_idx = layer_idx;
                                consume_params.bucket_rows = total_tokens;
                                consume_params.d_model = config_.d_model;
                                consume_params.ticket_storage =
                                    dispatch_ticket_storage;

                                const std::string consume_name =
                                    prefix +
                                    "moe_overlay_ticket_consume_" +
                                    participant_suffix;
                                graph.addNode(
                                    consume_name,
                                    ComputeStageFactory::
                                        createMoEOverlayTicketConsume(
                                            consume_params),
                                    device);
                                graph.addDependency(
                                    consume_name,
                                    root_return_node);
                                if (captured_overlay_continuation)
                                {
                                    graph.setGraphCaptureWaveContract(
                                        consume_name,
                                        GraphCaptureWaveContract{
                                            .identity =
                                                captured_overlay_inbound_wave,
                                        });
                                }
                                tier_terminal = consume_name;
                            }
                            if (compute_apportioned_tier_on_graph_local_participant && needsMoEParticipantAllreduce())
                            {
                                const size_t allreduce_count =
                                    static_cast<size_t>(total_tokens) * static_cast<size_t>(config_.d_model);
                                const std::string ar_name = prefix + "moe_sparse_return_reduce_" +
                                                            nodeSuffixForTier(tier, static_cast<int>(tier_index)) +
                                                            "_allreduce";
                                auto rebalance_sidebands = takeGraphRebalanceSidebandsForAllreduce();
                                auto allreduce_stage = createTPAllreduceStage(
                                    moe_output,
                                    allreduce_count,
                                    device,
                                    layer_idx,
                                    /*is_attention=*/false,
                                    ar_name,
                                    buffers.idFor(BufferId::MOE_COMBINED_OUTPUT),
                                    std::move(rebalance_sidebands));
                                if (allreduce_stage)
                                {
                                    graph.addNode(ar_name, std::move(allreduce_stage), device);
                                    graph.addDependency(ar_name, tier_terminal);
                                    tier_terminal = ar_name;
                                    maybeAddGraphRebalancePayloadStageAfterSideband(
                                        ar_name,
                                        tier_terminal);
                                }
                            }

                            last_return_reduce = tier_terminal;
                            first_return_scatter = false;
                        }
                    }
                    if (!rank_batch_lanes.empty())
                        wireMoEOverlayRankBatchForkJoin(graph, rank_batch_lanes, rank_local_lanes);
                }

                if (!pending_canonical_ticket_consumes.empty())
                {
                    if (last_return_reduce.empty())
                    {
                        throw std::logic_error(
                            "Qwen35 MoE canonical CPU ticket ingress has no completed manual frontier");
                    }
                    std::string ingress_dependency = last_return_reduce;
                    for (const auto &pending :
                         pending_canonical_ticket_consumes)
                    {
                        MoEOverlayTicketConsumeStage::Params consume_params;
                        consume_params.device_id = device;
                        if (!pending.storage->metadataOnly())
                        {
                            consume_params.output = canonical_route_contributions;
                            consume_params.output_buffer_id = buffers.idFor(
                                BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS);
                        }
                        consume_params.layer_idx = layer_idx;
                        consume_params.bucket_rows = total_tokens;
                        consume_params.top_k = config_.moe.top_k;
                        consume_params.d_model = config_.d_model;
                        consume_params.canonical_route_ticket_storage =
                            pending.storage;

                        const std::string consume_name =
                            prefix +
                            "moe_overlay_canonical_ticket_consume_" +
                            pending.participant_suffix;
                        graph.addNode(
                            consume_name,
                            ComputeStageFactory::
                                createMoEOverlayTicketConsume(
                                    consume_params),
                            device);
                        graph.addDependency(
                            consume_name, ingress_dependency);
                        /*
                         * Local GPU canonical publication writes every route
                         * slot, including zeroes for CPU-owned experts. CPU
                         * computation overlaps that work, but ingress replaces
                         * those zeroes and must therefore follow the local
                         * writer. A later reducer joining both branches cannot
                         * repair a write/write race in their shared route bank.
                         * This is a graph edge, not a host/stream wait; the CPU
                         * service and GPU expert compute remain independent.
                         */
                        if (!captured_local_expert_terminal.empty())
                        {
                            graph.addDependency(
                                consume_name, captured_local_expert_terminal);
                        }
                        graph.setGraphCaptureWaveContract(
                            consume_name,
                            GraphCaptureWaveContract{
                                .identity = captured_overlay_inbound_wave,
                            });
                        ingress_dependency = consume_name;
                    }
                    local_canonical_ticket_terminal = ingress_dependency;
                    last_return_reduce = ingress_dependency;
                }

                const std::string rank_local_ticket_return_terminal =
                    rank_local_collective_context_lifetime &&
                            !has_rank_local_canonical_ticket_returns
                        ? last_return_reduce
                        : std::string{};

                if (use_mapped_activation_parent)
                {
                    if (mapped_activation_return_bindings.empty() ||
                        captured_local_expert_terminal.empty())
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE mapped activation parent has an incomplete fanout or continuation-local overlap body");
                    }

                    /*
                     * Remote lanes are independent until their returned values
                     * enter the continuation accumulator. Exact one-row direct
                     * mappings use one topology-sized kernel grid. Wider rows
                     * use one typed fork/join transaction: every dispatch and
                     * return acquisition runs on a persistent auxiliary lane
                     * while continuation-local experts run on the public graph
                     * stream. The join folds lanes in this binding vector's
                     * planner-canonical order, so completion timing cannot
                     * perturb FP32 arithmetic.
                     */
                    const bool use_one_row_lane_batch =
                        total_tokens == 1 &&
                        mapped_activation_return_bindings.size() > 1u &&
                        std::all_of(
                            mapped_activation_return_bindings.begin(),
                            mapped_activation_return_bindings.end(),
                            [](const MappedActivationReturnBinding &binding)
                            {
                                const auto payload =
                                    binding.lane.dispatchPayload(1);
                                return payload.valid() &&
                                       payload.selection.usesCompactRows();
                            });

                    const bool use_asynchronous_lane_batch =
                        total_tokens > 1;
                    const bool use_lane_batch =
                        use_one_row_lane_batch ||
                        use_asynchronous_lane_batch;
                    std::shared_ptr<
                        MoEOverlayActivationLaneBatchState>
                        mapped_lane_transaction;
                    if (use_lane_batch)
                    {
                        std::vector<
                            MoEOverlayMappedActivationDeviceLane>
                            lanes;
                        lanes.reserve(
                            mapped_activation_return_bindings.size());
                        for (const auto &binding :
                             mapped_activation_return_bindings)
                        {
                            lanes.push_back(binding.lane);
                        }
                        if(!config_.moe.activation_batch_inventory)
                            throw std::logic_error("Original activation Batch graph requires its selected source/row inventory");
                        config_.moe.activation_batch_inventory->require(modelForwardInput(),identity,total_tokens,
                            mapped_activation_return_bindings.front().stage_ordinal);
                        mapped_lane_transaction = std::make_shared<
                            MoEOverlayActivationLaneBatchState>(
                            device,
                            std::move(lanes),
                            total_tokens,
                            mapped_activation_return_bindings.front()
                                .stage_ordinal,
                            layer_idx,
                            moeActivationBatchMemoryAuthority(device));
                    }

                    // Each CPU tensor shard consumes the original routed inputs.
                    // The acquired bank names its current whole/cohort target.
                    // Capture retains only the fixed original cohort/member geometry.
                    const auto joint_dispatch_target = [&](int target)
                    {
                        return jointRouteTargetFromMap(*owner_map_lifetime, layer_idx,
                                                       config_.moe.num_experts, target);
                    };

                    if (use_lane_batch)
                    {
                        MoEOverlayActivationDispatchPackBatchStage::Params
                            packet_params;
                        packet_params.device_id = device;
                        packet_params.transaction =
                            mapped_lane_transaction;
                        packet_params.hidden = buffers.normalized;
                        packet_params.routing_indices = routing_indices;
                        packet_params.routing_weights = routing_weights;
                        packet_params.placement = pinned_route_placement;
                        packet_params.joint_route_targets.reserve(
                            mapped_lane_transaction->lanes().size());
                        for (const auto &lane : mapped_lane_transaction->lanes())
                            packet_params.joint_route_targets.push_back(
                                joint_dispatch_target(lane.target_participant_id));
                        /* A sequence-length scalar is also the live-row count
                         * only for a padded single-request prefill bucket.
                         * During decode it contains the growing KV position,
                         * which can be hundreds of rows while this retained
                         * graph owns exactly one activation row. Binding it to
                         * decode would make every sparse packet fail its
                         * capacity contract after a non-empty prefill. */
                        packet_params.active_row_count_device =
                            batch_size == 1 && total_tokens > 1
                                ? routed_live_rows
                                : nullptr;
                        packet_params.hidden_buffer_id =
                            buffers.idFor(BufferId::NORMALIZED);
                        packet_params.routing_indices_buffer_id =
                            buffers.idFor(BufferId::MOE_EXPERT_INDICES);
                        packet_params.routing_weights_buffer_id =
                            buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS);
                        const std::string packet_name =
                            prefix +
                            "moe_overlay_activation_dispatch_pack_batch";
                        graph.addNode(
                            packet_name,
                            ComputeStageFactory::
                                createMoEOverlayActivationDispatchPackBatch(
                                    packet_params),
                            device);
                        GraphCaptureWaveContract packet_wave{
                            .identity = mapped_dispatch_wave_identity(
                                mapped_lane_transaction->lanes().front()
                                    .target_participant_id),
                        };
                        for (std::size_t lane = 1u;
                             lane < mapped_lane_transaction->lanes().size();
                             ++lane)
                        {
                            packet_wave.passive_following_identities.push_back(
                                mapped_dispatch_wave_identity(
                                    mapped_lane_transaction->lanes()[lane]
                                        .target_participant_id));
                        }
                        graph.setGraphCaptureWaveContract(
                            packet_name, std::move(packet_wave));
                        graph.addDependency(
                            packet_name, prefix + "moe_routing");
                        mapped_activation_dispatch_nodes.push_back(packet_name);
                    }
                    else
                    {
                        for (const auto &binding :
                             mapped_activation_return_bindings)
                        {
                            MoEOverlayActivationDispatchPackStage::Params
                                packet_params;
                            packet_params.device_id = device;
                            packet_params.lane = binding.lane;
                            packet_params.hidden = buffers.normalized;
                            packet_params.routing_indices = routing_indices;
                            packet_params.routing_weights = routing_weights;
                            packet_params.placement = pinned_route_placement;
                            packet_params.joint_route_target =
                                joint_dispatch_target(binding.lane.target_participant_id);
                            /* Decode has one live activation row regardless of
                             * the current sequence/KV length. Only padded
                             * prefill buckets may derive live rows from the
                             * sequence-length scalar. */
                            packet_params.active_row_count_device =
                                batch_size == 1 && total_tokens > 1
                                    ? routed_live_rows
                                    : nullptr;
                            packet_params.hidden_buffer_id =
                                buffers.idFor(BufferId::NORMALIZED);
                            packet_params.routing_indices_buffer_id =
                                buffers.idFor(
                                    BufferId::MOE_EXPERT_INDICES);
                            packet_params.routing_weights_buffer_id =
                                buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS);
                            packet_params.physical_rows = total_tokens;
                            packet_params.stage_ordinal =
                                binding.stage_ordinal;
                            packet_params.model_layer_index = layer_idx;

                            graph.addNode(
                                binding.dispatch_node_name,
                                ComputeStageFactory::
                                    createMoEOverlayActivationDispatchPack(
                                        packet_params),
                                device);
                            graph.setGraphCaptureWaveContract(
                                binding.dispatch_node_name,
                                GraphCaptureWaveContract{
                                    .identity =
                                        mapped_dispatch_wave_identity(
                                            binding.lane
                                                .target_participant_id),
                                });
                            graph.addDependency(
                                binding.dispatch_node_name,
                                mapped_activation_dispatch_nodes.empty()
                                    ? prefix + "moe_routing"
                                    : mapped_activation_dispatch_nodes.back());
                            mapped_activation_dispatch_nodes.push_back(
                                binding.dispatch_node_name);
                        }
                    }

                    if (mapped_activation_dispatch_nodes.empty())
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE mapped activation parent produced no dispatch stage");
                    }
                    const auto &cpu_packet = mapped_activation_dispatch_nodes.back();
                    if (mapped_cuda_retained_m1)
                    {
                        // Keep complete retirement at the ticket unit boundary.
                        // Native capture consumes the earlier router export for this packet.
                        graph.addDependency(cpu_packet, captured_local_expert_compute_node);
                        retained_pre_cpu_publications.push_back(cpu_packet);
                    }
                    else
                        graph.addDependency(captured_local_expert_compute_node, cpu_packet);
                    if ((mapped_decode_shared_overlap ||
                         (total_tokens > 1 && !shared_prefill_cpu_input_fork &&
                          requiresHeterogeneousTicketSegmentation(graph.nativeCaptureEnvelope()))) &&
                        planned_shared_device == device)
                        shared_decode_cpu_packet = cpu_packet;
                    if (total_tokens > 1 && planned_shared_device == device &&
                        !requiresHeterogeneousTicketSegmentation(graph.nativeCaptureEnvelope()))
                        shared_prefill_mapped_dispatch = cpu_packet;

                    if (use_lane_batch)
                    {
                        MoEOverlayActivationReturnConsumeBatchStage::Params
                            return_params;
                        return_params.device_id = device;
                        return_params.transaction =
                            mapped_lane_transaction;
                        return_params.canonical_route_contributions =
                            canonical_route_contributions;
                        return_params
                            .canonical_route_contributions_buffer_id =
                            buffers.idFor(
                                BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS);
                        const std::string return_name =
                            prefix +
                            "moe_overlay_activation_return_consume_batch";
                        graph.addNode(
                            return_name,
                            ComputeStageFactory::
                                createMoEOverlayActivationReturnConsumeBatch(
                                    return_params),
                            device);
                        GraphCaptureWaveContract return_wave{
                            .identity = mapped_return_wave_identity(
                                mapped_lane_transaction->lanes().front()
                                    .target_participant_id),
                        };
                        for (std::size_t lane = 1u;
                             lane < mapped_lane_transaction->lanes().size();
                             ++lane)
                        {
                            return_wave.passive_following_identities.push_back(
                                mapped_return_wave_identity(
                                    mapped_lane_transaction->lanes()[lane]
                                        .target_participant_id));
                        }
                        graph.setGraphCaptureWaveContract(
                            return_name, std::move(return_wave));
                        graph.addDependency(
                            return_name, captured_local_expert_terminal);
                        if (!local_canonical_ticket_terminal.empty())
                        {
                            graph.addDependency(
                                return_name,
                                local_canonical_ticket_terminal);
                        }
                        mapped_activation_return_nodes.push_back(
                            return_name);
                    }
                    else
                    {
                        for (const auto &binding :
                             mapped_activation_return_bindings)
                        {
                            MoEOverlayActivationReturnConsumeStage::Params
                                return_params;
                            return_params.device_id = device;
                            return_params.lane = binding.lane;
                            return_params.canonical_route_contributions =
                                canonical_route_contributions;
                            return_params
                                .canonical_route_contributions_buffer_id =
                                buffers.idFor(
                                    BufferId::
                                        MOE_CANONICAL_ROUTE_CONTRIBUTIONS);
                            return_params.physical_rows = total_tokens;
                            return_params.stage_ordinal =
                                binding.stage_ordinal;
                            return_params.model_layer_index = layer_idx;
                            graph.addNode(
                                binding.node_name,
                                ComputeStageFactory::
                                    createMoEOverlayActivationReturnConsume(
                                        return_params),
                                device);
                            graph.setGraphCaptureWaveContract(
                                binding.node_name,
                                GraphCaptureWaveContract{
                                    .identity =
                                        mapped_return_wave_identity(
                                            return_params.lane
                                                .target_participant_id),
                                });
                            graph.addDependency(
                                binding.node_name,
                                mapped_activation_return_nodes.empty()
                                    ? captured_local_expert_terminal
                                    : mapped_activation_return_nodes.back());
                            if (mapped_activation_return_nodes.empty() &&
                                !local_canonical_ticket_terminal.empty())
                            {
                                graph.addDependency(
                                    binding.node_name,
                                    local_canonical_ticket_terminal);
                            }
                            mapped_activation_return_nodes.push_back(
                                binding.node_name);
                        }
                    }
                    last_return_reduce =
                        mapped_activation_return_nodes.back();
                    first_return_scatter = false;

                }

                if (last_return_reduce.empty() &&
                    !captured_overlay_continuation)
                {
                    throw std::runtime_error("Qwen35 MoE graph-native overlay produced no local expert participants for layer " +
                                             std::to_string(layer_idx));
                }
                if (captured_overlay_continuation &&
                    captured_local_expert_terminal.empty())
                {
                    throw std::runtime_error(
                        "Qwen35 MoE distributed GPU continuation omitted its graph-local captured expert branch");
                }

                if (cpu_llep_state)
                {
                    if (!cpu_llep_overlay_domain || !global_tp_ctx ||
                        !cpu_llep_local_consumer ||
                        global_tp_ctx->degree() != participant_count ||
                        global_tp_ctx->myIndex() !=
                            cpu_llep_local_consumer
                                ->cpuCurrentBatchLLEPParticipantId())
                    {
                        throw std::logic_error(
                            "Qwen35 MoE CPU LLEP sparse graph has no exact local participant endpoint");
                    }

                    MoECPUCurrentBatchLLEPStage::Params begin_params;
                    begin_params.device_id = device;
                    begin_params.phase =
                        CPUCurrentBatchLLEPPhase::Begin;
                    begin_params.physical_executor =
                        cpu_current_batch_llep_executor_;
                    begin_params.residency_authority =
                        config_.moe.expert_overlay_residency_authority;
                    begin_params.residency_domain =
                        cpu_llep_overlay_domain->name;
                    begin_params.tp_ctx = global_tp_ctx;
                    begin_params.expert_consumer =
                        cpu_llep_local_consumer;
                    begin_params.state = cpu_llep_state;
                    begin_params.routing_indices = routing_indices;
                    begin_params.routing_weights = routing_weights;
                    begin_params.routing_indices_buffer_id =
                        buffers.idFor(BufferId::MOE_EXPERT_INDICES);
                    begin_params.routing_weights_buffer_id =
                        buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS);
                    begin_params.layer_idx = layer_idx;
                    begin_params.seq_len = total_tokens;
                    begin_params.top_k = config_.moe.top_k;
                    begin_params.num_experts = config_.moe.num_experts;
                    begin_params.participant_count = participant_count;
                    begin_params.participant_id =
                        global_tp_ctx->myIndex();
                    begin_params.planner_config.alpha_numerator =
                        std::max<uint32_t>(
                            1u,
                            config_.moe.routed_prefill_config
                                .llep_alpha_numerator);
                    begin_params.planner_config.alpha_denominator =
                        std::max<uint32_t>(
                            1u,
                            config_.moe.routed_prefill_config
                                .llep_alpha_denominator);
                    begin_params.planner_config.lambda_numerator =
                        std::max<uint32_t>(
                            1u,
                            config_.moe.routed_prefill_config
                                .llep_lambda_numerator);
                    begin_params.planner_config.lambda_denominator =
                        std::max<uint32_t>(
                            1u,
                            config_.moe.routed_prefill_config
                                .llep_lambda_denominator);
                    begin_params.planner_config.enable_balanced_skip =
                        config_.moe.routed_prefill_config
                            .llep_enable_balanced_skip;
                    begin_params.stage_name =
                        prefix + "moe_cpu_current_batch_llep_begin";
                    const std::string begin_name =
                        begin_params.stage_name;
                    graph.addNode(
                        begin_name,
                        ComputeStageFactory::createMoECPUCurrentBatchLLEP(
                            begin_params),
                        device);
                    graph.addDependency(
                        begin_name, prefix + "moe_routing");
                    if (sparse_graph_contract.ownsDispatchAuthority())
                        graph.addDependency(dispatch_name, begin_name);
                    for (const auto &sparse_dispatch_name :
                         cpu_llep_sparse_dispatch_nodes)
                    {
                        graph.addDependency(
                            sparse_dispatch_name, begin_name);
                    }

                    MoECPUCurrentBatchLLEPStage::Params restore_params;
                    restore_params.device_id = device;
                    restore_params.phase =
                        CPUCurrentBatchLLEPPhase::Restore;
                    restore_params.physical_executor =
                        cpu_current_batch_llep_executor_;
                    restore_params.residency_authority =
                        config_.moe.expert_overlay_residency_authority;
                    restore_params.residency_domain =
                        cpu_llep_overlay_domain->name;
                    restore_params.tp_ctx = global_tp_ctx;
                    restore_params.expert_consumer =
                        cpu_llep_local_consumer;
                    restore_params.state = cpu_llep_state;
                    restore_params.layer_idx = layer_idx;
                    restore_params.seq_len = total_tokens;
                    restore_params.top_k = config_.moe.top_k;
                    restore_params.num_experts = config_.moe.num_experts;
                    restore_params.participant_count = participant_count;
                    restore_params.participant_id =
                        global_tp_ctx->myIndex();
                    restore_params.stage_name =
                        prefix + "moe_cpu_current_batch_llep_restore";
                    const std::string restore_name =
                        restore_params.stage_name;
                    graph.addNode(
                        restore_name,
                        ComputeStageFactory::createMoECPUCurrentBatchLLEP(
                            restore_params),
                        device);
                    graph.addDependency(
                        restore_name, last_return_reduce);
                    last_return_reduce = restore_name;
                }

                /*
                 * Sparse return has one logical continuation owner: only that
                 * participant's MOE_COMBINED_OUTPUT contains the complete
                 * routed row.  A NodeTP continuation, however, resumes dense
                 * execution on every rank.  Publish the compact active prefix
                 * through the continuation domain before any rank consumes it;
                 * otherwise the non-root shards feed stale routed activations
                 * into the next layer and the first later TP collective mixes
                 * different hidden states.
                 *
                 * This is a production graph edge, not parity reconstruction.
                 * Every rank materializes the same rooted collective node after
                 * the globally ordered sparse protocol.  Single-participant
                 * heterogeneous overlays retain their root-local result and do
                 * not pay for a redundant broadcast.
                 */
                const auto merge_direct_ticket_routes_after =
                    [&](const std::string &mapped_return_terminal,
                        bool annotate_inbound_capture_wave)
                {
                    if (rank_local_ticket_return_terminal.empty())
                        return mapped_return_terminal;

                    /*
                     * Mapped remote lanes accumulate directly into the
                     * continuation output while the colocated CPU endpoint
                     * accumulates into the fixed ticket scratch. Join the two
                     * independent branches once, after both exact publication
                     * edges, without making either producer wait for the
                     * other.
                     */
                    ResidualAddStage::Params merge_params;
                    merge_params.device_id = device;
                    merge_params.input = buffers.attn_proj;
                    merge_params.residual = moe_output;
                    merge_params.output = moe_output;
                    merge_params.num_elements =
                        static_cast<size_t>(total_tokens) *
                        static_cast<size_t>(config_.d_model);
                    merge_params.input_buffer_id =
                        buffers.idFor(BufferId::ATTN_PROJ);
                    merge_params.residual_buffer_id =
                        buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                    merge_params.output_buffer_id =
                        buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                    const std::string merge_name =
                        prefix + "moe_overlay_colocated_routes_merge";
                    graph.addNode(
                        merge_name,
                        ComputeStageFactory::createResidualAdd(merge_params),
                        device);
                    graph.addDependency(
                        merge_name, mapped_return_terminal);
                    graph.addDependency(
                        merge_name, rank_local_ticket_return_terminal);
                    if (annotate_inbound_capture_wave)
                    {
                        graph.setGraphCaptureWaveContract(
                            merge_name,
                            GraphCaptureWaveContract{
                                .identity = captured_overlay_inbound_wave,
                            });
                    }
                    return merge_name;
                };

                if (host_return_layout == MoEOverlayReturnLayout::CanonicalExpertRoutes &&
                    sparse_graph_contract.ownsDispatchAuthority())
                {
                    if (!device.is_cpu() || !canonical_route_contributions || last_return_reduce.empty())
                        throw std::logic_error("Canonical host return has no complete root-owned publication bank");
                    // Reuse the LocalTP packed protocol and its coverage proof.
                    // Only this final node applies weights, in original router
                    // order, after every tier/participant has returned raw rows.
                    MoECanonicalRouteReduceStage::Params reduce;
                    reduce.device_id = device;
                    reduce.canonical_route_contributions = canonical_route_contributions;
                    reduce.routing_weights = routing_weights;
                    reduce.output = moe_output;
                    reduce.seq_len = total_tokens;
                    reduce.top_k = config_.moe.top_k;
                    reduce.d_model = config_.d_model;
                    reduce.canonical_route_arithmetic = MoECanonicalRouteArithmeticPolicy::UnweightedExpertRowThenOrderedFMA;
                    reduce.canonical_route_layout = MoECanonicalRoutePublicationLayout::PackedIndexedRouteRows;
                    reduce.reduction_role = MoECanonicalRouteReductionRole::RootOwner;
                    reduce.canonical_route_contributions_buffer_id = buffers.idFor(BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS);
                    reduce.routing_weights_buffer_id = buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS);
                    reduce.output_buffer_id = buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                    const std::string fold_name = prefix + "moe_host_routes_ordered_reduce";
                    graph.addNode(fold_name, ComputeStageFactory::createMoECanonicalRouteReduce(reduce), device);
                    graph.addDependency(fold_name, last_return_reduce);
                    last_return_reduce = fold_name;
                }

                const bool distributed_dense_continuation =
                    distributed_overlay &&
                    overlay_plan->continuation_domain_spec
                            .effectiveDensePolicy() ==
                        DenseParallelPolicy::TensorParallel;
                if (captured_local_tp_continuation)
                {
                    if (!local_tp_ctx || continuation_root_tp_index < 0 ||
                        continuation_root_tp_index >= local_tp_ctx->degree() ||
                        config_.tp_device_idx < 0 ||
                        config_.tp_device_idx >= local_tp_ctx->degree() ||
                        !canonical_route_contributions)
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE captured distributed LocalTP publication has an incomplete rooted collective contract");
                    }
                    const bool has_external_contributions = requiresCapturedSparseReturn(
                        *overlay_plan, *owner_map_lifetime, layer_idx, config_.moe.rebalance_config.mode,
                        sparse_graph_contract.ownsDispatchAuthority()
                            ? std::optional<std::string_view>{last_return_reduce} : std::nullopt);

                    const auto route_transport =
                        node_local_route_transport;
                    if (route_transport ==
                        MoEOverlayNodeLocalRouteTransport::Unresolved)
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE captured multi-GPU continuation has no resolved node-local route transport");
                    }
                    const bool use_mapped_sparse_routes =
                        route_transport ==
                        MoEOverlayNodeLocalRouteTransport::MappedSparse;
                    const auto &route_exchange =
                        config_.moe.node_local_route_exchange;
                    if (mapped_intermediate_shards &&
                        (!use_mapped_sparse_routes || !route_exchange))
                    {
                        throw std::logic_error(
                            "Mapped intermediate shards require the original continuation route exchange");
                    }
                    if (config_.moe.node_local_route_transport_policy
                                .requiresMappedExchange() !=
                        static_cast<bool>(route_exchange))
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE node-local route transport policy and mapped fabric ownership disagree");
                    }

                    MoEDomainRouteAssignmentLedger
                        domain_route_assignment{};
                    MoERuntimeRouteWeightBinding runtime_route_weights{};
                    MoEOverlayRoutePlacementDeviceBinding
                        overlay_route_placement{};
                    if (captured_distributed_overlay_runtime_table)
                    {
                        /*
                         * Route evidence follows the device-authored routing
                         * transaction, not the transport selected for canonical
                         * expert rows. Native NCCL/RCCL and mapped sparse lanes
                         * consume the same request-pinned placement epoch and
                         * final route schedule, so both must expose the same
                         * authenticated evidence at the root reducer boundary.
                         */
                        if (!moe_runtime_table || layer_idx < 0 ||
                            layer_idx >= moe_runtime_table->layerCount())
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE captured LocalTP continuation requires a complete device runtime table");
                        }
                        const auto &runtime_layer =
                            moe_runtime_table->hostLayerState(layer_idx);
                        const std::uint64_t required_route_slots =
                            static_cast<std::uint64_t>(total_tokens) *
                            static_cast<std::uint64_t>(config_.moe.top_k);
                        if (!runtime_layer.route_participant_ids ||
                            required_route_slots == 0u ||
                            required_route_slots >
                                static_cast<std::uint64_t>(
                                    runtime_layer.prefill_route_capacity))
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE captured LocalTP continuation has no complete final device assignment ledger");
                        }
                        domain_route_assignment = {
                            .participant_ids =
                                runtime_layer.route_participant_ids,
                            .capacity =
                                runtime_layer.prefill_route_capacity,
                        };
                        runtime_route_weights = bindMoERuntimeRouteWeights(
                            moe_runtime_table->deviceLayerState(layer_idx),
                            runtime_layer,
                            captured_overlay_route_weight_projection);
                        if (!runtime_route_weights.validFor(
                                static_cast<std::uint32_t>(total_tokens),
                                static_cast<std::uint32_t>(
                                    config_.moe.top_k)))
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE captured LocalTP continuation has no workload-correct final device route-weight publication");
                        }
                        overlay_route_placement =
                            moe_runtime_table->overlayRoutePlacementBinding(
                                layer_idx);
                        if (!overlay_route_placement.valid())
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE captured LocalTP continuation has no request-pinned global placement binding");
                        }
                    }
                    if (use_mapped_sparse_routes)
                    {
                        /*
                         * With no native P2P, each non-root publishes only the
                         * original route slots assigned to it. The root reads
                         * local VRAM or an exact mapped lane per slot and keeps
                         * the serial router order without transporting zeroes.
                         */
                        std::vector<MoENodeLocalRouteEndpoint> route_endpoints;
                        for (const auto &participant :
                             owner_map_lifetime->participants())
                        {
                            if (participant.tier_idx ==
                                    continuation_tier_index &&
                                continuation_expert_domain &&
                                participant.domain_name ==
                                    continuation_expert_domain->name)
                            {
                                route_endpoints.push_back({
                                    .participant_id =
                                        participant.domain_participant_index,
                                    .device = participant.device,
                                });
                            }
                        }
                        if (route_endpoints.size() !=
                            static_cast<std::size_t>(local_tp_ctx->degree()))
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE sparse continuation route endpoint count differs from its LocalTP device cell");
                        }
                        route_exchange->materialize(
                            std::move(route_endpoints),
                            continuation_root_tp_index,
                            static_cast<std::uint32_t>(
                                graphStableActivationRowCapacity(
                                    config_, device)),
                            static_cast<std::uint32_t>(config_.moe.top_k),
                            static_cast<std::uint32_t>(config_.d_model));
                        /* Assignment and placement evidence was bound above
                         * from the common runtime authority. The sparse fabric
                         * adds transport storage only; it does not redefine
                         * routing truth. */
                    }

                    std::string ordered_reduce_dependency =
                        captured_local_expert_terminal;
                    if (route_transport ==
                        MoEOverlayNodeLocalRouteTransport::NativeCollective)
                    {
                        /*
                         * A P2P-capable homogeneous domain keeps NCCL/RCCL in
                         * charge of topology. Reduce the live canonical route
                         * prefix to the fixed continuation root, then perform
                         * the same ordered FP32 fold used by the mapped path.
                         * Non-root reducers remain declarative passive nodes.
                         */
                        TPLocalRootedCollectiveStage::Params rooted_params;
                        rooted_params.device_id = device;
                        rooted_params.tp_ctx = local_tp_ctx;
                        rooted_params.tensor = canonical_route_contributions;
                        rooted_params.count =
                            static_cast<std::size_t>(total_tokens) *
                            static_cast<std::size_t>(config_.moe.top_k) *
                            static_cast<std::size_t>(config_.d_model);
                        rooted_params.dtype = CollectiveDataType::FLOAT32;
                        rooted_params.operation =
                            TPLocalRootedCollectiveOperation::ReduceSum;
                        rooted_params.root_device_index =
                            continuation_root_tp_index;
                        rooted_params.participant_device_index =
                            config_.tp_device_idx;
                        rooted_params.stage_name =
                            prefix +
                            "moe_overlay_continuation_routes_reduce_to_root";
                        rooted_params.tensor_buffer_id = buffers.idFor(
                            BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS);
                        graph.addNode(
                            rooted_params.stage_name,
                            ComputeStageFactory::createTPLocalRootedCollective(
                                rooted_params),
                            device);
                        graph.addDependency(
                            rooted_params.stage_name,
                            captured_local_expert_terminal);
                        ordered_reduce_dependency = rooted_params.stage_name;
                    }

                    // The native rooted reduction is still part of the GPU
                    // producer. Retain this frontier before CPU/follower
                    // ingress extends the ordered fold's dependencies below.
                    std::string captured_continuation_terminal =
                        ordered_reduce_dependency;
                    const bool ticket_segmented_envelope =
                        requiresHeterogeneousTicketSegmentation(
                            graph.nativeCaptureEnvelope());
                    if (!retained_pre_cpu_publications.empty())
                    {
                        if (ticket_segmented_envelope || !use_mapped_activation_parent)
                        {
                            if (captured_continuation_terminal == captured_local_expert_terminal)
                                captured_continuation_terminal = retained_pre_cpu_publications.back();
                            for (const auto& publication : retained_pre_cpu_publications)
                                if (publication != captured_continuation_terminal)
                                    graph.addDependency(captured_continuation_terminal, publication);
                        }
                        else if (!mapped_activation_return_nodes.empty())
                        {
                            // Pure GPU native full graph: routed ReduceSum only
                            // depends on local experts; attach dispatch packet
                            // completion to return ingress instead so LocalTP
                            // reduction runs immediately after local experts.
                            for (const auto& publication : retained_pre_cpu_publications)
                                graph.addDependency(mapped_activation_return_nodes.front(), publication);
                        }
                    }
                    for (const auto &pending : pending_canonical_ticket_consumes)
                    {
                        // The native reduction also writes the canonical bank.
                        // CPU rows may replace their slots only after every
                        // continuation writer, irrespective of launch ordering.
                        graph.addDependency(
                            prefix + "moe_overlay_canonical_ticket_consume_" +
                                pending.participant_suffix,
                            captured_continuation_terminal);
                    }

                    if (use_mapped_activation_parent)
                    {
                        if (mapped_activation_return_nodes.empty() ||
                            last_return_reduce.empty())
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE mapped activation parent has no canonical return materialization frontier");
                        }
                        /* Native LocalTP reduction, when selected, owns the
                         * continuation-domain slots first. Mapped followers
                         * then fill disjoint external-domain slots. Exactly one
                         * ordered reducer consumes the completed canonical bank. */
                        graph.addDependency(
                            mapped_activation_return_nodes.front(),
                            captured_continuation_terminal);
                        ordered_reduce_dependency = last_return_reduce;
                    }

                    const bool has_cpu_ticket_boundary =
                        requiresHeterogeneousTicketSegmentation(
                            graph.nativeCaptureEnvelope()) &&
                        (has_external_contributions || use_mapped_activation_parent);
                    if (has_cpu_ticket_boundary && !last_return_reduce.empty())
                    {
                        // Portable CPU packets and their ticket ingress form
                        // one service unit. Keep the final GPU fold in its
                        // captured suffix instead of interleaving a launch
                        // between host parsing and remote dispatch.
                        ordered_reduce_dependency = last_return_reduce;
                    }

                    MoECanonicalRouteReduceStage::Params reduce_params;
                    reduce_params.device_id = device;
                    reduce_params.canonical_route_contributions =
                        canonical_route_contributions;
                    reduce_params.output = moe_output;
                    reduce_params.seq_len = total_tokens;
                    reduce_params.top_k = config_.moe.top_k;
                    reduce_params.d_model = config_.d_model;
                    reduce_params.canonical_route_arithmetic =
                        MoECanonicalRouteArithmeticPolicy::
                            PreweightedContributionThenOrderedAdd;
                    reduce_params.canonical_route_layout =
                        MoECanonicalRoutePublicationLayout::
                            DenseOriginalRouteSlots;
                    reduce_params.reduction_role =
                        config_.tp_device_idx == continuation_root_tp_index
                            ? MoECanonicalRouteReductionRole::RootOwner
                            : MoECanonicalRouteReductionRole::
                                  NonRootParticipant;
                    /* The shared owner remains alive because ordinary prefill
                     * may embed it, but a native decode graph must not receive
                     * that capability. Stage behavior follows its phase's
                     * typed transport, never pointer presence elsewhere in
                     * the retained graph family. */
                    reduce_params.node_local_route_exchange =
                        use_mapped_sparse_routes ? route_exchange : nullptr;
                    const bool all_overlay_participants_gpu =
                        owner_map_lifetime &&
                        !owner_map_lifetime->participants().empty() &&
                        std::all_of(
                            owner_map_lifetime->participants().begin(),
                            owner_map_lifetime->participants().end(),
                            [](const auto &p) { return p.device.is_gpu(); });
                    reduce_params.node_local_route_precision =
                        (use_mapped_sparse_routes && ordinary_prefill_graph &&
                         !mtp_sidecar_context && all_overlay_participants_gpu &&
                         !has_rank_local_canonical_ticket_returns)
                            ? MoENodeLocalRoutePayloadPrecision::FixedPoint32Adaptive16
                            : MoENodeLocalRoutePayloadPrecision::FP32;
                    if (mapped_intermediate_shards)
                        reduce_params.node_local_route_ownership = MoENodeLocalRouteOwnership::ContinuationCohort;
                    reduce_params.domain_route_assignment =
                        domain_route_assignment;
                    reduce_params.external_route_source =
                        continuationExternalCanonicalRouteSource(
                            reduce_params.node_local_route_ownership,
                            use_mapped_activation_parent,
                            has_rank_local_canonical_ticket_returns);
                    reduce_params.runtime_route_weights =
                        runtime_route_weights;
                    reduce_params.overlay_route_placement =
                        overlay_route_placement;
                    reduce_params.route_participant_id =
                        config_.tp_device_idx;
                    reduce_params.canonical_route_contributions_buffer_id =
                        buffers.idFor(
                            BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS);
                    reduce_params.output_buffer_id =
                        buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);

                    const std::string ordered_reduce_name =
                        prefix +
                        "moe_overlay_continuation_routes_ordered_reduce";
                    std::string early_stage_sparse_name;
                    if (!has_cpu_ticket_boundary &&
                        use_mapped_sparse_routes &&
                        use_mapped_activation_parent &&
                        !mapped_activation_return_nodes.empty() &&
                        total_tokens > 1 &&
                        reduce_params.reduction_role ==
                            MoECanonicalRouteReductionRole::RootOwner)
                    {
                        /*
                         * Overlap LocalTP peer sparse-route staging (`CUDA:1 -> CUDA:0`)
                         * with the heterogeneous follower compute window instead of
                         * waiting until every follower return has materialized. The peer
                         * only publishes its own continuation-domain slots and writes
                         * into exchange-private root VRAM, completely disjoint from
                         * external follower return materialization.
                         */
                        auto shared_sparse_state = std::make_shared<
                            MoECanonicalRouteReduceStage::SharedNodeLocalSparseState>();
                        auto stage_params = reduce_params;
                        stage_params.node_local_sparse_phase =
                            MoENodeLocalSparsePhase::StageOnly;
                        stage_params.node_local_sparse_shared_state =
                            shared_sparse_state;
                        reduce_params.node_local_sparse_phase =
                            MoENodeLocalSparsePhase::FoldStaged;
                        reduce_params.node_local_sparse_shared_state =
                            std::move(shared_sparse_state);

                        early_stage_sparse_name =
                            prefix +
                            "moe_overlay_continuation_routes_stage_sparse";
                        graph.addNode(
                            early_stage_sparse_name,
                            ComputeStageFactory::createMoECanonicalRouteReduce(
                                stage_params),
                            device);
                        graph.addDependency(
                            early_stage_sparse_name,
                            captured_continuation_terminal);
                        graph.addDependency(
                            mapped_activation_return_nodes.front(),
                            early_stage_sparse_name);
                    }
                    graph.addNode(
                        ordered_reduce_name,
                        ComputeStageFactory::createMoECanonicalRouteReduce(
                            reduce_params),
                        device);
                    graph.addDependency(
                        ordered_reduce_name,
                        ordered_reduce_dependency);
                    if (!early_stage_sparse_name.empty())
                    {
                        graph.addDependency(
                            ordered_reduce_name,
                            early_stage_sparse_name);
                    }
                    if (ordered_reduce_dependency != captured_continuation_terminal)
                    {
                        graph.addDependency(
                            ordered_reduce_name,
                            captured_continuation_terminal);
                    }
                    if (!local_canonical_ticket_terminal.empty())
                    {
                        graph.addDependency(
                            ordered_reduce_name,
                            local_canonical_ticket_terminal);
                    }
                    captured_overlay_routed_unit_terminal =
                        ordered_reduce_name;
                    if (!has_cpu_ticket_boundary &&
                        route_transport == MoEOverlayNodeLocalRouteTransport::NativeCollective &&
                        !captured_continuation_terminal.empty())
                    {
                        captured_overlay_pre_return_terminal =
                            captured_continuation_terminal;
                        captured_overlay_first_return_node =
                            !mapped_activation_return_nodes.empty()
                                ? mapped_activation_return_nodes.front()
                                : ordered_reduce_name;
                    }
                    GraphCaptureWaveContract ordered_reduce_capture_wave{
                        .identity =
                            captured_overlay_outbound_wave,
                    };
                    if (!sparse_graph_contract
                             .ownsDispatchAuthority() &&
                        !mapped_activation_return_wave_identities.empty())
                    {
                        /* Root-only return consumers are captured after the
                         * common ordered reduction and before the common
                         * publication broadcast. Join their exact lane order
                         * passively so both LocalTP participants retain one
                         * capture-wave ordinal sequence. */
                        ordered_reduce_capture_wave
                            .passive_following_identities =
                            mapped_activation_return_wave_identities;
                    }
                    graph.setGraphCaptureWaveContract(
                        ordered_reduce_name,
                        std::move(ordered_reduce_capture_wave));
                    if (has_cpu_ticket_boundary)
                    {
                        if ((mapped_decode_shared_overlap ||
                             (total_tokens > 1 && !shared_prefill_cpu_input_fork)) &&
                            planned_shared_device == device)
                            shared_decode_ticket_cutpoint = captured_continuation_terminal;
                        /*
                         * Declaratively seal the pre-CPU captured ticket unit,
                         * early-trigger ordering, and HostSubmissionOrder edges.
                         * This rejects any wiring that places a GPU overlap
                         * branch ahead of early CPU wake-up, attaches manual
                         * work to a follower, or inverts post-CPU joins.
                         */
                        GraphComposer overlay_composer(graph);
                        std::vector<GraphNodeRef> rank_dispatch_refs;
                        if (sparse_graph_contract.ownsDispatchAuthority())
                        {
                            rank_dispatch_refs.reserve(distributed_rank_batch_dispatch_nodes.size());
                            for (const auto &rank_batch_dispatch_name : distributed_rank_batch_dispatch_nodes)
                                rank_dispatch_refs.push_back(overlay_composer.ref(rank_batch_dispatch_name));
                        }
                        std::optional<GraphNodeRef> early_trigger_ref;
                        if (sparse_graph_contract.ownsDispatchAuthority())
                        {
                            if (!dispatch_ticket_publish_name.empty())
                                early_trigger_ref = overlay_composer.ref(dispatch_ticket_publish_name);
                            else if (!mapped_activation_dispatch_nodes.empty())
                                early_trigger_ref = overlay_composer.ref(mapped_activation_dispatch_nodes.back());
                        }
                        sealHeterogeneousOverlayTicketSection(graph, HeterogeneousOverlayTicketSection{
                            .router = overlay_composer.ref(prefix + "moe_routing"),
                            .early_trigger = std::move(early_trigger_ref),
                            .pre_cpu_gpu_cutpoint = overlay_composer.ref(captured_continuation_terminal),
                            .ticket_unit_identity = prefix + "moe_overlay_pre_cpu_ticket_unit",
                            .pre_cpu_overlap_branches = {},
                            .manual_host_entry = (host_dispatch_path_materialized &&
                                                  sparse_graph_contract.ownsDispatchAuthority())
                                                     ? std::optional<GraphNodeRef>{overlay_composer.ref(dispatch_name)}
                                                     : std::nullopt,
                            .post_cpu_join = overlay_composer.ref(ordered_reduce_name),
                            .rank_batch_dispatch_entries = std::move(rank_dispatch_refs),
                        });
                    }
                    else if (sparse_graph_contract.ownsDispatchAuthority())
                    {
                        for (const auto &rank_batch_dispatch_name :
                             distributed_rank_batch_dispatch_nodes)
                        {
                            graph.addDependency(
                                rank_batch_dispatch_name,
                                ordered_reduce_name);
                        }
                    }

                    std::string publication_dependency =
                        ordered_reduce_name;
                    if (
                        sparse_graph_contract.ownsDispatchAuthority())
                    {
                        if (!has_external_contributions && last_return_reduce.empty())
                        {
                            // The same rooted GPU reduction already owns every expert route.
                            // ATTN_PROJ has no remote payload and must not be merged as one.
                        }
                        else if (use_mapped_activation_parent &&
                            rank_local_ticket_return_terminal.empty())
                        {
                            /* Mapped GPU returns are already represented in
                             * canonical route slots and were consumed by the
                             * sole ordered reducer above. */
                        }
                        else if (has_rank_local_canonical_ticket_returns &&
                                 distributed_rank_batch_dispatch_nodes.empty())
                        {
                            /* Colocated CPU rows already occupy authenticated
                             * original slots in the canonical bank consumed by
                             * the reducer. No dense residual exists to merge. */
                        }
                        else if (use_mapped_activation_parent)
                        {
                            publication_dependency =
                                merge_direct_ticket_routes_after(
                                    ordered_reduce_name,
                                    /*annotate_inbound_capture_wave=*/false);
                        }
                        else
                        {
                        /*
                         * The final sparse ticket owns the aggregate remote
                         * routed contribution in ATTN_PROJ. Merge it into the
                         * root's canonically folded continuation-local result;
                         * the later shared-expert combine overwrites ATTN_PROJ
                         * only after this dependency completes.
                         */
                        ResidualAddStage::Params merge_params;
                        merge_params.device_id = device;
                        merge_params.input = buffers.attn_proj;
                        merge_params.residual = moe_output;
                        merge_params.output = moe_output;
                        merge_params.num_elements =
                            static_cast<size_t>(total_tokens) *
                            static_cast<size_t>(config_.d_model);
                        merge_params.input_buffer_id =
                            buffers.idFor(BufferId::ATTN_PROJ);
                        merge_params.residual_buffer_id =
                            buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                        merge_params.output_buffer_id =
                            buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                        const std::string merge_name =
                            prefix +
                            "moe_overlay_remote_routes_merge";
                        graph.addNode(
                            merge_name,
                            ComputeStageFactory::createResidualAdd(
                                merge_params),
                            device);
                        graph.addDependency(
                            merge_name,
                            ordered_reduce_name);
                        graph.addDependency(
                            merge_name,
                            last_return_reduce);
                        publication_dependency = merge_name;
                        }
                    }

                    const bool can_defer_combined_publication =
                        shared_expert_requires_tp_allreduce &&
                        has_shared_expert_branch &&
                        layer.shared_expert_gate_inp &&
                        planned_shared_device == device;
                    if (can_defer_combined_publication)
                    {
                        /* The shared partial is already graph-independent and
                         * ready while remote routed experts execute. Defer this
                         * intermediate publication so the next collective can
                         * reduce shared evidence directly to the fixed root;
                         * one later broadcast will publish the final combined
                         * row instead of transmitting two full-width tensors. */
                        deferred_overlay_combined_publication =
                            DeferredOverlayCombinedPublication{
                                .root_device_index =
                                    continuation_root_tp_index,
                                .participant_count = local_tp_ctx->degree(),
                                .routed_terminal = publication_dependency,
                                .capture_wave_identity =
                                    captured_overlay_inbound_wave,
                            };
                        if (!deferred_overlay_combined_publication->valid())
                        {
                            throw std::logic_error(
                                "Qwen35 MoE deferred overlay publication has an incomplete rooted topology contract");
                        }
                        ffn_terminal = publication_dependency;
                    }
                    else
                    {
                        TPLocalRootedCollectiveStage::Params broadcast_params;
                        broadcast_params.device_id = device;
                        broadcast_params.tp_ctx = local_tp_ctx;
                        broadcast_params.tensor = moe_output;
                        broadcast_params.count =
                            static_cast<size_t>(total_tokens) *
                            static_cast<size_t>(config_.d_model);
                        broadcast_params.dtype = CollectiveDataType::FLOAT32;
                        broadcast_params.operation =
                            TPLocalRootedCollectiveOperation::Broadcast;
                        broadcast_params.root_device_index =
                            continuation_root_tp_index;
                        broadcast_params.participant_device_index =
                            config_.tp_device_idx;
                        broadcast_params.stage_name =
                            prefix +
                            "moe_overlay_continuation_broadcast";
                        broadcast_params.tensor_buffer_id =
                            buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                        if (shouldUseMoEOverlayMappedDensePublication(
                                route_transport,
                                broadcast_params.count * sizeof(float)))
                        {
                            broadcast_params
                                .mapped_dense_publication_exchange =
                                route_exchange;
                        }
                        const std::string publication_name =
                            broadcast_params.stage_name;
                        graph.addNode(
                            publication_name,
                            ComputeStageFactory::createTPLocalRootedCollective(
                                broadcast_params),
                            device);
                        graph.addDependency(
                            publication_name,
                            publication_dependency);
                        /*
                         * The root prepends return-ticket ingress and merge to
                         * this wave; the non-root begins at the broadcast. Both
                         * record the identical collective sequence.
                         */
                        graph.setGraphCaptureWaveContract(
                            publication_name,
                            GraphCaptureWaveContract{
                                .identity = captured_overlay_inbound_wave,
                            });
                        ffn_terminal = publication_name;
                    }
                }
                else if (captured_overlay_continuation)
                {
                    if (!sparse_graph_contract
                             .ownsDispatchAuthority() ||
                        captured_local_expert_terminal.empty())
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE captured single-device continuation has no graph-local publication authority");
                    }

                    std::string ordered_reduce_dependency =
                        captured_local_expert_terminal;
                    if (!last_return_reduce.empty() &&
                        use_mapped_activation_parent)
                    {
                        ordered_reduce_dependency = last_return_reduce;
                    }
                    else if (!local_canonical_ticket_terminal.empty())
                    {
                        ordered_reduce_dependency =
                            local_canonical_ticket_terminal;
                    }
                    else if (!last_return_reduce.empty())
                    {
                        /*
                         * Portable MPI returns use a separate dense buffer.
                         * Keep the local fold in the captured ingress suffix,
                         * after that return, so host ticket parsing and all
                         * rank-batch service remain one contiguous manual unit.
                         * An independent fold here would split host parsing
                         * from MPI dispatch with a needless GPU launch. Local
                         * expert computation still runs in the producer graph;
                         * this DAG edge introduces no stream synchronization.
                         */
                        ordered_reduce_dependency = last_return_reduce;
                    }

                    MoECanonicalRouteReduceStage::Params reduce_params;
                    reduce_params.device_id = device;
                    reduce_params.canonical_route_contributions =
                        canonical_route_contributions;
                    reduce_params.output = moe_output;
                    reduce_params.seq_len = total_tokens;
                    reduce_params.top_k = config_.moe.top_k;
                    reduce_params.d_model = config_.d_model;
                    reduce_params.canonical_route_arithmetic =
                        MoECanonicalRouteArithmeticPolicy::
                            PreweightedContributionThenOrderedAdd;
                    reduce_params.canonical_route_layout =
                        MoECanonicalRoutePublicationLayout::
                            DenseOriginalRouteSlots;
                    reduce_params.reduction_role =
                        MoECanonicalRouteReductionRole::RootOwner;
                    reduce_params.domain_route_assignment =
                        pinned_domain_route_assignment;
                    reduce_params.external_route_source =
                        use_mapped_activation_parent ||
                                has_rank_local_canonical_ticket_returns
                            ? MoEExternalCanonicalRouteSource::
                                  RootCanonicalRouteBank
                            : MoEExternalCanonicalRouteSource::
                                  DeferredDenseMerge;
                    reduce_params.runtime_route_weights =
                        pinned_runtime_route_weights;
                    reduce_params.overlay_route_placement =
                        pinned_route_placement;
                    reduce_params.canonical_route_contributions_buffer_id =
                        buffers.idFor(
                            BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS);
                    reduce_params.output_buffer_id =
                        buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);

                    const std::string ordered_reduce_name =
                        prefix +
                        "moe_overlay_continuation_routes_ordered_reduce";
                    graph.addNode(
                        ordered_reduce_name,
                        ComputeStageFactory::createMoECanonicalRouteReduce(
                            reduce_params),
                        device);
                    graph.addDependency(
                        ordered_reduce_name,
                        ordered_reduce_dependency);

                    /*
                     * A CPU ticket publishes only its own original route
                     * slots. Its readiness is not a completion edge for the
                     * concurrently executing continuation-local experts.
                     * Preserve both producers in the DAG, as LocalTP does:
                     * capture compilation may schedule independent units in
                     * parallel and must not rely on insertion order or the
                     * CPU branch normally taking longer than the GPU branch.
                     */
                    if (ordered_reduce_dependency !=
                        captured_local_expert_terminal)
                    {
                        graph.addDependency(
                            ordered_reduce_name,
                            captured_local_expert_terminal);
                    }

                    captured_overlay_routed_unit_terminal =
                        ordered_reduce_name;
                    if (requiresHeterogeneousTicketSegmentation(
                            graph.nativeCaptureEnvelope()))
                    {
                        /*
                         * The single-device continuation uses the same exact
                         * cutpoint as LocalTP: the local GPU expert terminal
                         * closes the producer unit immediately before the CPU
                         * ticket.  Ticket ingress and the ordered reduction
                         * belong to the following captured suffix; closing the
                         * unit on that reducer would add a redundant captured
                         * split after every CPU transaction.
                         */
                        graph.setHeterogeneousTicketUnitContract(
                            captured_local_expert_terminal,
                            GraphHeterogeneousTicketUnitContract{
                                .identity =
                                    prefix +
                                    "moe_overlay_pre_cpu_ticket_unit",
                            });

                        if (!rank_local_ticket_return_terminal.empty())
                        {
                            /*
                             * This is a submission-order edge, not a data
                             * dependency on the ticket publication itself.
                             * The host consumer reads the earlier authenticated
                             * publication record while the captured producer's
                             * remaining GPU work runs asynchronously.
                             */
                            graph.addDependency(
                                dispatch_name,
                                ordered_reduce_name);
                        }
                    }

                    std::string publication_dependency =
                        ordered_reduce_name;
                    if (use_mapped_activation_parent &&
                        !rank_local_ticket_return_terminal.empty())
                    {
                        publication_dependency =
                            merge_direct_ticket_routes_after(
                                ordered_reduce_name,
                                /*annotate_inbound_capture_wave=*/true);
                    }
                    else if (!last_return_reduce.empty() &&
                             !use_mapped_activation_parent &&
                             (!has_rank_local_canonical_ticket_returns ||
                              !distributed_rank_batch_dispatch_nodes.empty()))
                    {
                        /*
                         * A portable inter-node ticket arrives in ATTN_PROJ so
                         * it cannot overwrite the concurrently produced local
                         * contribution. Merge only after both asynchronous
                         * branches have published their exact completion edge.
                         */
                        ResidualAddStage::Params merge_params;
                        merge_params.device_id = device;
                        merge_params.input = buffers.attn_proj;
                        merge_params.residual = moe_output;
                        merge_params.output = moe_output;
                        merge_params.num_elements =
                            static_cast<size_t>(total_tokens) *
                            static_cast<size_t>(config_.d_model);
                        merge_params.input_buffer_id =
                            buffers.idFor(BufferId::ATTN_PROJ);
                        merge_params.residual_buffer_id =
                            buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                        merge_params.output_buffer_id =
                            buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                        const std::string merge_name =
                            prefix + "moe_overlay_remote_routes_merge";
                        graph.addNode(
                            merge_name,
                            ComputeStageFactory::createResidualAdd(
                                merge_params),
                            device);
                        graph.addDependency(
                            merge_name, captured_local_expert_terminal);
                        graph.addDependency(
                            merge_name, ordered_reduce_name);
                        graph.addDependency(
                            merge_name, last_return_reduce);
                        graph.setGraphCaptureWaveContract(
                            merge_name,
                            GraphCaptureWaveContract{
                                .identity = captured_overlay_inbound_wave,
                            });
                        publication_dependency = merge_name;
                    }
                    ffn_terminal = publication_dependency;
                }
                else if (distributed_dense_continuation)
                {
                    const auto &continuation_domain =
                        overlay_runtime_plan->continuationDomain();
                    if (!device.is_cpu() || !global_tp_ctx ||
                        global_tp_ctx->degree() <= 1 ||
                        global_tp_ctx->degree() !=
                            static_cast<int>(
                                continuation_domain.participants.size()) ||
                        global_tp_ctx->myIndex() != config_.tp_device_idx ||
                        continuation_root_participant < 0 ||
                        continuation_root_participant >=
                            global_tp_ctx->degree())
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE distributed dense continuation requires "
                            "one aligned CPU GlobalTP participant per continuation "
                            "domain entry for layer " +
                            std::to_string(layer_idx));
                    }
                    if (sparse_graph_contract.ownsDispatchAuthority() &&
                        last_return_reduce.empty())
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE distributed dense continuation source has no completed sparse return frontier");
                    }

                    /*
                     * A follower that owns no currently active expert may
                     * still be a dense TP participant. Its router node is the
                     * local readiness frontier; the rooted broadcast then
                     * waits for the source rank's complete sparse reduction.
                     * A follower with experts instead arrives here only after
                     * its authenticated return has been submitted.
                     */
                    const std::string publication_dependency =
                        last_return_reduce.empty()
                            ? prefix + "moe_routing"
                            : last_return_reduce;

                    MoECanonicalOutputBroadcastStage::Params
                        publication_params;
                    publication_params.device_id = device;
                    publication_params.tp_ctx = global_tp_ctx;
                    publication_params.output = moe_output;
                    publication_params.seq_len = total_tokens;
                    publication_params.d_model = config_.d_model;
                    publication_params.root_participant =
                        continuation_root_participant;
                    publication_params.stage_name =
                        prefix + "moe_overlay_continuation_broadcast";
                    publication_params.output_buffer_id =
                        buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);

                    const std::string publication_name =
                        publication_params.stage_name;
                    graph.addNode(
                        publication_name,
                        ComputeStageFactory::
                            createMoECanonicalOutputBroadcast(
                                publication_params),
                        device);
                    graph.addDependency(
                        publication_name,
                        publication_dependency);
                    ffn_terminal = publication_name;
                }
                else
                {
                    ffn_terminal = last_return_reduce;
                }
    }

    void Qwen35MoEGraph::buildFFNSharedAndFinalizeSection(FFNBuildContext &ctx)
    {
        using DeferredOverlayCombinedPublication = FFNBuildContext::DeferredOverlayCombinedPublication;

        const LayerWeights &layer = ctx.layer;
        ActivationBuffers &buffers = ctx.buffers;
        const int batch_size = ctx.batch_size;
        const DeviceId device = ctx.device;
        const int32_t *const sequence_lengths_device = ctx.sequence_lengths_device;
        const Qwen4ExpPredictorResidualBinding *const predictor = ctx.predictor;

        const int layer_idx = ctx.layer_idx;
        ComputeGraph &graph = ctx.graph;
        const bool mtp_sidecar_context = ctx.mtp_sidecar_context;
        const std::string &prefix = ctx.prefix;
        std::string &ffn_terminal = ctx.ffn_terminal;
        const int total_tokens = ctx.total_tokens;
        const int32_t *const routed_live_rows = ctx.routed_live_rows;
        const auto &overlay_runtime_plan = ctx.overlay_runtime_plan;
        const bool mapped_cuda_retained_m1 = ctx.mapped_cuda_retained_m1;
        const LayerWeightBindings &layer_bindings = ctx.layer_bindings;
        IMoERuntimeTable *const moe_runtime_table = ctx.moe_runtime_table;
        const MoEOverlayNodeLocalRouteTransport node_local_route_transport = ctx.node_local_route_transport;
        const bool last_device_rebalance_decode_layer = ctx.last_device_rebalance_decode_layer;
        ILocalTPContext *const local_tp_ctx = ctx.local_tp_ctx;
        const auto &env = debugEnv();
        std::string &current_batch_llep_plan_node = ctx.current_batch_llep_plan_node;
        std::string &current_batch_llep_apply_node = ctx.current_batch_llep_apply_node;
        std::string &current_batch_llep_expert_node = ctx.current_batch_llep_expert_node;
        auto &current_batch_llep_plan_params = ctx.current_batch_llep_plan_params;
        auto &current_batch_llep_apply_params = ctx.current_batch_llep_apply_params;
        bool &current_batch_llep_payload_sideband_taken = ctx.current_batch_llep_payload_sideband_taken;
        auto &residual_read = ctx.residual_read;
        const auto &routed_pipeline_kernel_owner = ctx.routed_pipeline_kernel_owner;
        TensorBase *const moe_output = ctx.moe_output;
        TensorBase *const canonical_route_contributions = ctx.canonical_route_contributions;
        TensorBase *const shared_output = ctx.shared_output;
        bool &moe_combined_output_ready = ctx.moe_combined_output_ready;
        std::string &shared_ffn_last = ctx.shared_ffn_last;
        auto &shared_prefill_cpu_input_fork = ctx.shared_prefill_cpu_input_fork;
        auto &shared_prefill_mapped_dispatch = ctx.shared_prefill_mapped_dispatch;
        auto &shared_decode_cpu_packet = ctx.shared_decode_cpu_packet;
        auto &shared_decode_ticket_cutpoint = ctx.shared_decode_ticket_cutpoint;
        bool &routed_overlay_has_distributed_sparse_protocol = ctx.routed_overlay_has_distributed_sparse_protocol;
        const DeviceId planned_shared_device = ctx.planned_shared_device;
        const bool shared_expert_requires_tp_allreduce = ctx.shared_expert_requires_tp_allreduce;
        auto &canonical_publication_lowering = ctx.canonical_publication_lowering;
        auto &deferred_overlay_combined_publication = ctx.deferred_overlay_combined_publication;
        std::string &captured_overlay_routed_unit_terminal = ctx.captured_overlay_routed_unit_terminal;
        std::string &captured_overlay_pre_return_terminal = ctx.captured_overlay_pre_return_terminal;
        std::string &captured_overlay_first_return_node = ctx.captured_overlay_first_return_node;

        const auto &forceDecodeEquivalentMoEVerifier = ctx.forceDecodeEquivalentMoEVerifier;
        const auto &forceGroupedSharedMoEVerifierPrefill = ctx.forceGroupedSharedMoEVerifierPrefill;
        const auto &graphRebalanceDomainKey = ctx.graphRebalanceDomainKey;
        const auto &takeCurrentBatchLLEPPayloadSideband = ctx.takeCurrentBatchLLEPPayloadSideband;
        const auto &appendRebalanceSidebands = ctx.appendRebalanceSidebands;
        const auto &takeGraphRebalanceSidebandsForAllreduce = ctx.takeGraphRebalanceSidebandsForAllreduce;
        const auto &maybeAddGraphRebalancePayloadStageAfterSideband = ctx.maybeAddGraphRebalancePayloadStageAfterSideband;
        (void)env;
        (void)current_batch_llep_plan_node;
        (void)current_batch_llep_apply_node;
        (void)current_batch_llep_expert_node;
        (void)current_batch_llep_payload_sideband_taken;
        (void)last_device_rebalance_decode_layer;
        (void)graphRebalanceDomainKey;
        (void)moe_runtime_table;

        // =====================================================================
        // Stage 4: Shared Expert FFN (always-active dense SwiGLU)
        // =====================================================================

        if (!moe_combined_output_ready &&
            layer.shared_expert_gate && layer.shared_expert_up && layer.shared_expert_down && shared_output)
        {
            DeviceId shared_device = planned_shared_device;
            /*
             * Shared experts are always-on dense FFNs even though their GGUF
             * names contain "expert". Phase-split decode therefore binds the
             * complete replicated shared-expert weights on every participant,
             * exactly like replicated attention and dense FFN weights. Only
             * the tensor-parallel prefill view produces a partial down row that
             * requires a collective. Keeping this policy in one named value
             * prevents the gate/combine lowering and the collective lowering
             * from disagreeing about whether the branch is already complete.
             */
            if (overlay_runtime_plan)
            {
                const auto &continuation_domain = overlay_runtime_plan->continuationDomain();
                const auto &shared_domain = overlay_runtime_plan->sharedExpertDomain();

                LOG_TRACE("[Qwen35MoEGraph] Layer " << layer_idx
                                                    << " shared expert uses domain " << shared_domain.name
                                                    << " on " << shared_device.to_string()
                                                    << "; final output returns to continuation domain "
                                                    << continuation_domain.name << " on " << device.to_string()
                                                    << " via MoE combine");
            }

            // Always use the actual weight dimensions (already sharded for TP).
            // config_.moe.shared_intermediate_size is the FULL size from metadata,
            // but when TP is active, gate/up weights have rows() == intermediate/tp_degree.
            int shared_intermediate = static_cast<int>(layer.shared_expert_gate->rows());

            SharedExpertFFNStage::Params shared_params;
            shared_params.device_id = shared_device;
            shared_params.input = buffers.normalized;
            shared_params.gate_w = layer.shared_expert_gate;
            shared_params.up_w = layer.shared_expert_up;
            shared_params.down_w = layer.shared_expert_down;
            shared_params.output = shared_output;
            shared_params.gate_scratch =
                buffers.get(buffers.idFor(BufferId::MOE_GATE_SCRATCH));
            shared_params.up_scratch =
                buffers.get(buffers.idFor(BufferId::MOE_UP_SCRATCH));
            shared_params.seq_len = total_tokens;
            // Shared and routed experts consume the same resident live prefix.
            // Capture capacity is not permission to evaluate padded rows.
            shared_params.active_row_count_device =
                shared_device.is_gpu() && batch_size == 1
                    ? routed_live_rows
                    : nullptr;
            shared_params.d_model = config_.d_model;
            shared_params.intermediate = shared_intermediate;
            shared_params.input_buffer_id = buffers.idFor(BufferId::NORMALIZED);
            shared_params.output_buffer_id = buffers.idFor(BufferId::MOE_SHARED_EXPERT_OUTPUT);
            shared_params.gate_scratch_buffer_id =
                buffers.idFor(BufferId::MOE_GATE_SCRATCH);
            shared_params.up_scratch_buffer_id =
                buffers.idFor(BufferId::MOE_UP_SCRATCH);
            /*
             * Shared-expert verifier rows must match ordinary serial decode,
             * not a separate dense GEMM oracle.  GPU serial decode uses the MoE
             * grouped table-decode helpers for the always-active shared expert,
             * so verifier-sized GPU batches use the sibling grouped table-prefill
             * route. This keeps runtime-M verifier execution economical while preserving the exact
             * descriptor family, split-K layout, and reduction order exercised
             * by production M=1 decode.
             */
            const bool shared_grouped_verifier_prefill =
                forceGroupedSharedMoEVerifierPrefill(shared_device);
            const bool shared_gpu_table_verifier_prefill =
                shared_grouped_verifier_prefill &&
                (shared_device.is_cuda() || shared_device.is_rocm());
            shared_params.force_grouped_verifier_prefill_for_decode =
                shared_gpu_table_verifier_prefill;
            /*
             * Router-row reuse is an optimization, not a prerequisite for the
             * grouped shared-expert verifier.  Deterministic execution disables
             * that reuse on both GPU backends so serial decode and grouped
             * verification quantize their inputs independently in the same
             * arithmetic regime.  Bind a required publication only when the
             * selected backend will actually produce it; otherwise the grouped
             * pipeline owns its normal graph-local quantization buffers.
             */
            const bool shared_router_q8_reuse_enabled =
                (shared_device.is_cuda() &&
                 debugEnv().gemm.cuda_moe_reuse_router_q8_hidden) ||
                (shared_device.is_rocm() &&
                 debugEnv().rocm.moe_reuse_router_q8_hidden);
            const bool shared_requires_router_q8_publication =
                shared_gpu_table_verifier_prefill &&
                shared_router_q8_reuse_enabled;
            if (shared_requires_router_q8_publication)
            {
                shared_params.required_router_q8_publication =
                    routed_pipeline_kernel_owner->router_q8_publication;
            }
            shared_params.force_decode_equivalent_verifier_prefill =
                (!shared_gpu_table_verifier_prefill &&
                 (!shared_grouped_verifier_prefill &&
                  forceDecodeEquivalentMoEVerifier(shared_device)));
            /*
             * Phase-split MTP sidecars execute against the replicated dense-decode
             * weight view rather than the ordinary per-rank TP decode view.  The
             * normal GPU grouped shared-expert decode shortcut is tuned for that
             * ordinary one-token decode path and owns a separate pointer-table
             * readiness contract.  MTP sidecar rows are verifier rows: they may
             * become live state, so their one-row shared-expert math must flow
             * through the decode-equivalent verifier path unless a grouped
             * shortcut has passed the same strict serial-decode proof.
             */
            shared_params.disable_grouped_decode_shortcut =
                mtp_sidecar_context &&
                (predictor ? predictor->layout().policy()==MTPShiftedKVHeadLayout::FullModelPerParticipant :
                    (config_.dense_tp_decode_replicated || config_.mtpUsesReplicatedDenseSidecarBinding()));
            shared_params.prepared_ref_gate = preparedRefForGraphWeight(
                layer_bindings.shared_expert_gate, shared_device);
            shared_params.prepared_ref_up = preparedRefForGraphWeight(
                layer_bindings.shared_expert_up, shared_device);
            shared_params.prepared_ref_down = preparedRefForGraphWeight(
                layer_bindings.shared_expert_down, shared_device);
            shared_params.prepared_store = prepared_weight_store_;
            if (shared_requires_router_q8_publication)
            {
                // Use the same typed prepared descriptor as the shared stage.
                // Floating triples do not consume Q8, even though their legacy
                // publication binding/dependency remains unchanged.
                DeviceMoEExpertDescriptor shared_descriptor;
                if (!prepared_weight_store_ || !shared_params.prepared_ref_gate ||
                    !shared_params.prepared_ref_up || !shared_params.prepared_ref_down ||
                    !exportDeviceMoEExpertWeightDescriptors(
                        prepared_weight_store_->gemmKernel(shared_params.prepared_ref_gate.value()),
                        prepared_weight_store_->gemmKernel(shared_params.prepared_ref_up.value()),
                        prepared_weight_store_->gemmKernel(shared_params.prepared_ref_down.value()),
                        config_.d_model, shared_intermediate, shared_descriptor))
                {
                    throw std::invalid_argument(
                        "shared expert required publication has no supported prepared descriptor triple");
                }
                if (!deviceMoEWeightFormatIsFloating(shared_descriptor.weight_format) &&
                    !shared_params.required_router_q8_publication->requireConsumerRows(
                        total_tokens, config_.d_model))
                {
                    throw std::invalid_argument(
                        "shared expert Q8 publication geometry conflicts with its routed graph");
                }
            }

            graph.addNode(prefix + "shared_expert_ffn",
                          ComputeStageFactory::createSharedExpertFFN(shared_params),
                          shared_device);
            graph.addDependency(prefix + "shared_expert_ffn", prefix + "ffn_norm");
            if (shared_prefill_cpu_input_fork)
            {
                // FIFO topological order otherwise makes this ffn_norm sibling
                // ready before routing's ticket child. Publish the same input
                // ticket first; retain the existing late CPU-result ingress and
                // all GPU scratch, arithmetic and collective dependencies.
                graph.addDependency(prefix + "shared_expert_ffn",
                    shared_prefill_cpu_input_fork->publication_node);
                // Preserve the former exact-stream shared-before-routed order;
                // only the CPU fork moves earlier, not either FFN's math or
                // the rank-symmetric collective sequence.
                graph.addDependency(shared_prefill_cpu_input_fork->routed_expert_node,
                    prefix + "shared_expert_ffn");
            }
            if (shared_prefill_mapped_dispatch)
            {
                // Order mapped-activation prefill dispatch ahead of the
                // independent shared_expert_ffn so remote GPU followers wake
                // immediately after moe_routing instead of waiting ~93 us/layer.
                graph.addDependency(prefix + "shared_expert_ffn",
                    *shared_prefill_mapped_dispatch);
            }
            if (shared_requires_router_q8_publication)
            {
                /*
                 * The GPU router quantizes the normalized verifier rows once and
                 * publishes those device-resident Q8 rows for both the routed and
                 * shared grouped pipelines.  The shared stage therefore consumes
                 * router-owned state even though its visible tensor input is still
                 * NORMALIZED.  Encode that hidden producer/consumer relationship in
                 * the graph: relying on the incidental insertion order of sibling
                 * nodes can let the shared stage reuse the previous layer's Q8
                 * publication when a fresh graph receives a different topological
                 * ordering. This edge applies equally to main-verifier and MTP
                 * sidecar graphs when reuse is enabled, and remains valid during
                 * whole-graph capture. Standalone quantization has no hidden
                 * producer and deliberately keeps the two branches parallel.
                 */
                graph.addDependency(
                    prefix + "shared_expert_ffn",
                    prefix + "moe_routing");
            }
            const bool main_verifier_rows =
                !mtp_sidecar_context &&
                config_.compute_all_position_logits &&
                total_tokens >= 1;
            /*
             * The accepted shared-verifier routes own their branch-local math:
             * CUDA and ROCm use the grouped table-prefill verifier route. Any route
             * outside those explicit grouped implementations keeps the
             * conservative dependency until it has its own branch-scoped
             * workspace proof.
             */
            const bool shared_verifier_owns_branch_local_math =
                main_verifier_rows &&
                (shared_gpu_table_verifier_prefill ||
                 shared_params.force_decode_equivalent_verifier_prefill ||
                 canonical_publication_lowering.usesRankBanks());
            bool shared_ffn_serialized_after_ffn_terminal = false;
            if (main_verifier_rows && !mapped_cuda_retained_m1 &&
                !shared_verifier_owns_branch_local_math &&
                !ffn_terminal.empty())
            {
                shared_ffn_serialized_after_ffn_terminal = true;
                /*
                 * A branch whose implementation shares mutable backend bridge
                 * state with routed execution is serialized by policy. The
                 * grouped GPU and canonical rank-bank implementations own
                 * distinct persistent workspaces and therefore remain parallel.
                 */
                graph.addDependency(prefix + "shared_expert_ffn", ffn_terminal);
            }
            if (!shared_ffn_serialized_after_ffn_terminal &&
                (shared_decode_cpu_packet || shared_decode_ticket_cutpoint))
            {
                orderMappedDecodeSharedFFN(graph, prefix + "shared_expert_ffn",
                    shared_decode_cpu_packet, shared_decode_ticket_cutpoint);
            }
            shared_ffn_last = prefix + "shared_expert_ffn";

            if (projection_arena_)
            {
                // Gate/up has packed its complete outgoing bytes before this
                // independent dense branch reuses scratch. Let native transport
                // run alongside that branch; import/down acquires the join.
                // No computation, reduction order or collective extent changes.
                overlapTPLocalAllGather(graph, prefix + "projection_intermediate_allgather",
                    prefix + "shared_expert_ffn");
            }

            if (canonical_publication_lowering.usesRankBanks())
            {
                if (!canonical_route_contributions || !buffers.normalized ||
                    !buffers.attn_proj || !moe_output ||
                    !layer.shared_expert_gate_inp || !local_tp_ctx ||
                    canonical_publication_lowering.root_participant < 0 ||
                    canonical_publication_lowering.participant_count <= 0 ||
                    canonical_publication_lowering.routed_producer.empty() ||
                    canonical_publication_lowering.rooted_reduce_node.empty() ||
                    canonical_publication_lowering.post_collective_terminal.empty())
                {
                    throw std::logic_error(
                        "Qwen35 MoE canonical rank-bank publication has an "
                        "incomplete graph contract for layer " +
                        std::to_string(layer_idx) + " on " +
                        device.to_string());
                }

                MoESharedExpertRankBankPublishStage::Params publish_params;
                publish_params.device_id = device;
                publish_params.shared_output = shared_output;
                publish_params.canonical_publication =
                    canonical_route_contributions;
                publish_params.seq_len = total_tokens;
                publish_params.top_k = config_.moe.top_k;
                publish_params.d_model = config_.d_model;
                publish_params.participant_device_index =
                    config_.tp_device_idx;
                publish_params.participant_count =
                    canonical_publication_lowering.participant_count;
                publish_params.active_row_count_device =
                    device.is_gpu() && batch_size == 1
                        ? routed_live_rows
                        : nullptr;
                publish_params.shared_output_buffer_id =
                    buffers.idFor(BufferId::MOE_SHARED_EXPERT_OUTPUT);
                publish_params.canonical_publication_buffer_id =
                    buffers.idFor(
                        BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS);

                const std::string publish_name =
                    prefix + "moe_shared_rank_bank_publish";
                graph.addNode(
                    publish_name,
                    ComputeStageFactory::
                        createMoESharedExpertRankBankPublish(
                            publish_params),
                    device);
                graph.addDependency(
                    publish_name,
                    prefix + "shared_expert_ffn");
                graph.addDependency(
                    publish_name,
                    canonical_publication_lowering.routed_producer);
                graph.addDependency(
                    canonical_publication_lowering.rooted_reduce_node,
                    publish_name);

                MoECanonicalPublicationFinalizeStage::Params finalize_params;
                finalize_params.device_id = device;
                finalize_params.input = buffers.normalized;
                finalize_params.gate_inp = layer.shared_expert_gate_inp;
                finalize_params.canonical_publication =
                    canonical_route_contributions;
                finalize_params.routed_output = moe_output;
                finalize_params.shared_output = shared_output;
                finalize_params.combined_output = buffers.attn_proj;
                finalize_params.seq_len = total_tokens;
                finalize_params.top_k = config_.moe.top_k;
                finalize_params.d_model = config_.d_model;
                finalize_params.participant_device_index =
                    config_.tp_device_idx;
                finalize_params.root_device_index =
                    canonical_publication_lowering.root_participant;
                finalize_params.participant_count =
                    canonical_publication_lowering.participant_count;
                finalize_params.active_row_count_device =
                    device.is_gpu() && batch_size == 1
                        ? routed_live_rows
                        : nullptr;
                finalize_params.input_buffer_id =
                    buffers.idFor(BufferId::NORMALIZED);
                finalize_params.canonical_publication_buffer_id =
                    buffers.idFor(
                        BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS);
                finalize_params.routed_output_buffer_id =
                    buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                finalize_params.shared_output_buffer_id =
                    buffers.idFor(BufferId::MOE_SHARED_EXPERT_OUTPUT);
                finalize_params.combined_output_buffer_id =
                    buffers.idFor(BufferId::ATTN_PROJ);

                const std::string finalize_name =
                    prefix + "moe_canonical_publication_finalize";
                graph.addNode(
                    finalize_name,
                    ComputeStageFactory::
                        createMoECanonicalPublicationFinalize(
                            finalize_params),
                    device);
                graph.addDependency(
                    finalize_name,
                    canonical_publication_lowering.post_collective_terminal);

                TPLocalRootedCollectiveStage::Params broadcast_params;
                broadcast_params.device_id = device;
                broadcast_params.tp_ctx = local_tp_ctx;
                broadcast_params.tensor = buffers.attn_proj;
                broadcast_params.count =
                    static_cast<size_t>(total_tokens) *
                    static_cast<size_t>(config_.d_model);
                broadcast_params.dtype = CollectiveDataType::FLOAT32;
                broadcast_params.operation =
                    TPLocalRootedCollectiveOperation::Broadcast;
                broadcast_params.root_device_index =
                    canonical_publication_lowering.root_participant;
                broadcast_params.participant_device_index =
                    config_.tp_device_idx;
                broadcast_params.stage_name =
                    prefix + "moe_canonical_publication_broadcast";
                broadcast_params.tensor_buffer_id =
                    buffers.idFor(BufferId::ATTN_PROJ);

                const std::string broadcast_name =
                    broadcast_params.stage_name;
                graph.addNode(
                    broadcast_name,
                    ComputeStageFactory::createTPLocalRootedCollective(
                        broadcast_params),
                    device);
                graph.addDependency(broadcast_name, finalize_name);

                moe_combined_output_ready = true;
                shared_ffn_last = broadcast_name;
                ffn_terminal = broadcast_name;

                LOG_TRACE(
                    "[Qwen35MoEGraph] Layer "
                    << layer_idx << " lowered "
                    << moeParticipantPublicationPolicyToString(
                           canonical_publication_lowering.policy)
                    << " participants="
                    << canonical_publication_lowering.participant_count
                    << " root="
                    << canonical_publication_lowering.root_participant
                    << " device=" << device.to_string());
            }

            /*
             * Input-parallel prefill shared-expert down rows are reduced before
             * the replicated sigmoid gate, preserving the serial LocalTP branch
             * arithmetic independently of routed-expert placement. Replicated
             * decode rows are already complete and must never be summed again.
             */
            if (shared_expert_requires_tp_allreduce &&
                !canonical_publication_lowering.usesRankBanks())
            {
                const size_t allreduce_count =
                    static_cast<size_t>(total_tokens) *
                    static_cast<size_t>(config_.d_model);
                const bool rooted_overlay_shared_reduction =
                    deferred_overlay_combined_publication.has_value();
                std::string ar_name =
                    prefix +
                    (rooted_overlay_shared_reduction
                         ? "shared_expert_reduce_to_overlay_root"
                         : "shared_expert_allreduce");
                std::vector<TPAllreduceSidebandWorkspaceBinding> rebalance_sidebands;
                if (shared_device == device)
                {
                    rebalance_sidebands = takeGraphRebalanceSidebandsForAllreduce();
                    appendRebalanceSidebands(
                        rebalance_sidebands,
                        takeCurrentBatchLLEPPayloadSideband());
                }
                // Shared partials are independent of routed down in both
                // prefill and decode. Decode keeps its original full-row sum,
                // gate and add; an event fork does not alter that arithmetic.
                // Control sidebands retain their separate ordered transaction.
                const bool projection_shared_overlap =
                    projection_arena_ &&
                    !rooted_overlay_shared_reduction && shared_device == device &&
                    rebalance_sidebands.empty() && !current_batch_llep_plan_params.has_value();
                std::unique_ptr<IComputeStage> allreduce_stage;
                bool installed_reduce_overlap = false;
                if (rooted_overlay_shared_reduction)
                {
                    const auto &publication =
                        *deferred_overlay_combined_publication;
                    if (!publication.valid() || !local_tp_ctx ||
                        config_.tp_device_idx < 0 ||
                        config_.tp_device_idx >= publication.participant_count ||
                        local_tp_ctx->degree() != publication.participant_count)
                    {
                        throw std::logic_error(
                            "Qwen35 MoE rooted overlay shared reduction has an incomplete participant contract");
                    }
                    TPLocalRootedCollectiveStage::Params reduce_params;
                    reduce_params.device_id = shared_device;
                    reduce_params.tp_ctx = local_tp_ctx;
                    reduce_params.tensor = shared_output;
                    reduce_params.count = allreduce_count;
                    reduce_params.dtype = CollectiveDataType::FLOAT32;
                    reduce_params.operation =
                        TPLocalRootedCollectiveOperation::ReduceSum;
                    reduce_params.root_device_index =
                        publication.root_device_index;
                    reduce_params.participant_device_index =
                        config_.tp_device_idx;
                    reduce_params.stage_name = ar_name;
                    reduce_params.tensor_buffer_id =
                        buffers.idFor(BufferId::MOE_SHARED_EXPERT_OUTPUT);
                    reduce_params.sideband_workspace_bindings =
                        std::move(rebalance_sidebands);
                    if (canonical_publication_lowering.usesMappedSparseRoutes() &&
                        publication.capture_rendezvous ==
                            DeferredOverlayCombinedPublication::CaptureRendezvous::NativeFullGraph &&
                        reduce_params.sideband_workspace_bindings.empty())
                    {
                        // The shared partial and routed expert rows are independent.
                        // Keep both event edges in this native executable; ticketed
                        // overlays and control sidebands retain their explicit order.
                        addTPLocalReduceOverlap(graph, std::move(reduce_params), {
                            .producer = prefix + "shared_expert_ffn",
                            .independent_compute = prefix + "moe_expert_ffn_overlay_fast",
                        });
                        graph.addDependency(publication.routed_terminal, ar_name);
                        installed_reduce_overlap = true;
                    }
                    else
                    {
                        allreduce_stage =
                            ComputeStageFactory::createTPLocalRootedCollective(
                                reduce_params);
                    }
                }
                else
                {
                    allreduce_stage = createTPAllreduceStage(
                        shared_output,
                        allreduce_count,
                        shared_device,
                        layer_idx,
                        /*is_attention=*/false,
                        ar_name,
                        buffers.idFor(BufferId::MOE_SHARED_EXPERT_OUTPUT),
                        std::move(rebalance_sidebands), std::nullopt,
                        predictor && shared_device.is_gpu() && predictor->rows().tokens().countOwner()
                            ? std::optional{NativeAllreduceRequestRows(1,total_tokens,config_.d_model,predictor->rows().tokens().countOwner())}
                            : std::nullopt,
                        predictor ? std::optional{ForwardExecutionPhase::Decode} : std::nullopt);
                    if (projection_shared_overlap && !forwardPhaseAllowsDecodeTopology() &&
                        layer.shared_expert_gate_inp)
                    {
                        const auto *native_sum = dynamic_cast<const TPAllreduceStage *>(allreduce_stage.get());
                        if (!native_sum)
                            throw std::logic_error("Projection shared-column publication requires its native sum declaration");
                        // Declare the final ownership boundary once. The
                        // reusable builder owns packing, fork/join, gating and
                        // final gather; this model owns neither event edges nor
                        // another buffer/precision policy.
                        ffn_terminal = finalizeMoEProjectionSharedColumns(graph, prefix + "projection_", {
                            .reduction = native_sum->params(),
                            .producer = prefix + "shared_expert_ffn",
                            .gate = layer.shared_expert_gate_inp,
                            .active_rows = batch_size == 1 ? routed_live_rows : nullptr,
                            .combined_output = {buffers.attn_proj, buffers.idFor(BufferId::ATTN_PROJ)}});
                        shared_ffn_last = ffn_terminal;
                        moe_combined_output_ready = true;
                        allreduce_stage.reset();
                    }
                }
                if (allreduce_stage || installed_reduce_overlap)
                {
                    if (allreduce_stage)
                        graph.addNode(ar_name, std::move(allreduce_stage), shared_device);
                    graph.addDependency(ar_name, prefix + "shared_expert_ffn");
                    if (projection_shared_overlap)
                    {
                        overlapMoEProjectionSharedAllreduce(graph, prefix + "projection_", ar_name);
                    }
                    if (rooted_overlay_shared_reduction &&
                        !shared_ffn_serialized_after_ffn_terminal &&
                        shared_device == device &&
                        !captured_overlay_pre_return_terminal.empty() &&
                        !requiresHeterogeneousTicketSegmentation(graph.nativeCaptureEnvelope()))
                    {
                        /*
                         * On a pure GPU native full graph without a manual CPU
                         * ticket cutpoint, both LocalTP participants run this
                         * rooted shared-expert reduction right after the routed
                         * LocalTP reduction and ahead of mapped follower return
                         * ingress, hiding the collective behind follower compute
                         * while preserving identical collective and wave order.
                         */
                        graph.addDependency(
                            ar_name,
                            captured_overlay_pre_return_terminal);
                        if (!captured_overlay_first_return_node.empty())
                        {
                            graph.addDependency(
                                captured_overlay_first_return_node,
                                ar_name);
                        }
                    }
                    else if (rooted_overlay_shared_reduction &&
                        !captured_overlay_routed_unit_terminal.empty())
                    {
                        /*
                         * The shared partial is ready before the heterogeneous
                         * ticket, but its rooted collective belongs to the
                         * post-ticket publication unit.  Making the collective
                         * consume the routed-unit terminal gives every LocalTP
                         * participant the same collective sequence in the same
                         * capture wave.  The reverse edge lets the authority
                         * enqueue an unmatched RCCL reduction before its manual
                         * CPU transaction while the peer waits at the next
                         * capture-wave rendezvous.  On devices whose collective
                         * wait kernel occupies the compute units, that cycle can
                         * also starve the ticket publication which would release
                         * the manual transaction.
                         */
                        graph.addDependency(
                            ar_name,
                            captured_overlay_routed_unit_terminal);
                    }
                    if (current_batch_llep_plan_params.has_value())
                    {
                        if (current_batch_llep_plan_node.empty() ||
                            current_batch_llep_apply_node.empty() ||
                            current_batch_llep_expert_node.empty() ||
                            !current_batch_llep_apply_params.has_value() ||
                            !current_batch_llep_payload_sideband_taken)
                        {
                            throw std::logic_error(
                                "Qwen35 MoE current-batch LLEP shared-collective binding is incomplete for layer " +
                                std::to_string(layer_idx));
                        }
                        graph.addDependency(
                            ar_name,
                            current_batch_llep_plan_node);
                        graph.addNode(
                            current_batch_llep_apply_node,
                            ComputeStageFactory::createMoEGPUCurrentBatchLLEP(
                                *current_batch_llep_apply_params),
                            device);
                        graph.addDependency(
                            current_batch_llep_apply_node,
                            ar_name);
                        graph.addDependency(
                            current_batch_llep_expert_node,
                            current_batch_llep_apply_node);
                    }
                    if (!rooted_overlay_shared_reduction &&
                        !projection_shared_overlap &&
                        (routed_overlay_has_distributed_sparse_protocol || projection_arena_) &&
                        !ffn_terminal.empty())
                    {
                        /*
                         * Both rank graphs must finish the same dispatch,
                         * participant compute, and return sequence before one
                         * of them enters this independent dense collective.
                         * The edge orders protocols without preventing the
                         * shared-expert GEMM from running while routed work is
                         * in flight.
                         */
                        graph.addDependency(ar_name, ffn_terminal);
                    }
                    shared_ffn_last = ar_name;
                    if (shared_device == device)
                    {
                        maybeAddGraphRebalancePayloadStageAfterSideband(
                            ar_name,
                            shared_ffn_last);
                    }
                }
            }

            // Stage 4b: Sigmoid gate on shared expert output
            if (!moe_combined_output_ready && layer.shared_expert_gate_inp &&
                !canonical_publication_lowering.usesRankBanks())
            {
                const bool rooted_overlay_finalize =
                    deferred_overlay_combined_publication.has_value();
                const bool can_fuse_gate_and_combine =
                    (!shared_expert_requires_tp_allreduce ||
                     rooted_overlay_finalize) &&
                    shared_device == device &&
                    moe_output && buffers.attn_proj;
                const bool gate_writes_combined_output =
                    can_fuse_gate_and_combine;

                SharedExpertGateStage::Params gate_params;
                gate_params.device_id = shared_device;
                gate_params.input = buffers.normalized;
                gate_params.gate_inp = layer.shared_expert_gate_inp;
                gate_params.shared_output = shared_output;
                gate_params.seq_len = total_tokens;
                gate_params.d_model = config_.d_model;
                gate_params.active_row_count_device =
                    shared_device.is_gpu() && batch_size == 1
                        ? routed_live_rows
                        : nullptr;
                if (rooted_overlay_finalize)
                {
                    const auto &publication =
                        *deferred_overlay_combined_publication;
                    gate_params.execution_role =
                        config_.tp_device_idx ==
                                publication.root_device_index
                            ? SharedExpertGateStage::ExecutionRole::RootOwner
                            : SharedExpertGateStage::ExecutionRole::
                                  NonRootObserver;
                }
                gate_params.input_buffer_id = buffers.idFor(BufferId::NORMALIZED);
                gate_params.output_buffer_id = buffers.idFor(BufferId::MOE_SHARED_EXPERT_OUTPUT);
                if (gate_writes_combined_output)
                {
                    gate_params.routed_residual = moe_output;
                    gate_params.combined_output = buffers.attn_proj;
                    gate_params.residual_buffer_id = buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                    gate_params.combined_output_buffer_id = buffers.idFor(BufferId::ATTN_PROJ);
                }

                graph.addNode(prefix + "shared_expert_gate",
                              ComputeStageFactory::createSharedExpertGate(gate_params),
                              shared_device);
                graph.addDependency(prefix + "shared_expert_gate", shared_ffn_last);
                if (gate_writes_combined_output)
                {
                    /* The root consumes both rooted shared bytes and its
                     * complete routed row. The non-root node is a typed no-op,
                     * but keeps the identical dependency and collective order
                     * before final publication. */
                    graph.addDependency(
                        prefix + "shared_expert_gate",
                        rooted_overlay_finalize
                            ? deferred_overlay_combined_publication
                                  ->routed_terminal
                            : ffn_terminal);
                    moe_combined_output_ready = true;
                }
                shared_ffn_last = prefix + "shared_expert_gate";
                if (moe_combined_output_ready)
                {
                    ffn_terminal = prefix + "shared_expert_gate";
                }

                if (rooted_overlay_finalize)
                {
                    const auto &publication =
                        *deferred_overlay_combined_publication;
                    if (!gate_writes_combined_output || !publication.valid() ||
                        !local_tp_ctx)
                    {
                        throw std::logic_error(
                            "Qwen35 MoE deferred overlay publication did not produce a root-owned combined row");
                    }

                    TPLocalRootedCollectiveStage::Params broadcast_params;
                    broadcast_params.device_id = device;
                    broadcast_params.tp_ctx = local_tp_ctx;
                    broadcast_params.tensor = buffers.attn_proj;
                    broadcast_params.count =
                        static_cast<size_t>(total_tokens) *
                        static_cast<size_t>(config_.d_model);
                    broadcast_params.dtype = CollectiveDataType::FLOAT32;
                    broadcast_params.operation =
                        TPLocalRootedCollectiveOperation::Broadcast;
                    broadcast_params.root_device_index =
                        publication.root_device_index;
                    broadcast_params.participant_device_index =
                        config_.tp_device_idx;
                    broadcast_params.stage_name =
                        prefix + "moe_overlay_combined_broadcast";
                    broadcast_params.tensor_buffer_id =
                        buffers.idFor(BufferId::ATTN_PROJ);
                    if (shouldUseMoEOverlayMappedDensePublication(
                            node_local_route_transport,
                            broadcast_params.count * sizeof(float)))
                    {
                        broadcast_params
                            .mapped_dense_publication_exchange =
                            config_.moe.node_local_route_exchange;
                    }
                    const std::string broadcast_name =
                        broadcast_params.stage_name;
                    graph.addNode(
                        broadcast_name,
                        ComputeStageFactory::createTPLocalRootedCollective(
                            broadcast_params),
                        device);
                    graph.addDependency(
                        broadcast_name, prefix + "shared_expert_gate");
                    if (publication.capture_rendezvous ==
                        DeferredOverlayCombinedPublication::
                            CaptureRendezvous::ExplicitWave)
                    {
                        graph.setGraphCaptureWaveContract(
                            broadcast_name,
                            GraphCaptureWaveContract{
                                .identity =
                                    publication.capture_wave_identity,
                            });
                    }
                    shared_ffn_last = broadcast_name;
                    ffn_terminal = broadcast_name;
                }
            }
        }

        // =====================================================================
        // Stage 5: Combine expert output + shared expert output → attn_proj
        // =====================================================================
        // The combined MoE output goes to attn_proj so that the next layer's
        // FusedResidualNormStage handles the residual add automatically.
        {
            if (moe_combined_output_ready)
            {
                // A fused shared gate or canonical rooted finalizer already
                // wrote ATTN_PROJ; no standalone residual add is permitted.
            }
            else if (!shared_ffn_last.empty() && shared_output)
            {
                // Add expert_output + shared_expert_output → attn_proj
                ResidualAddStage::Params add_params;
                add_params.device_id = device;
                add_params.input = shared_output;
                add_params.residual = moe_output;
                add_params.output = buffers.attn_proj;
                add_params.num_elements = static_cast<size_t>(total_tokens) * static_cast<size_t>(config_.d_model);
                add_params.input_buffer_id = buffers.idFor(BufferId::MOE_SHARED_EXPERT_OUTPUT);
                add_params.residual_buffer_id = buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                add_params.output_buffer_id = buffers.idFor(BufferId::ATTN_PROJ);

                graph.addNode(prefix + "moe_combine",
                              ComputeStageFactory::createResidualAdd(add_params),
                              device);
                graph.addDependency(prefix + "moe_combine", ffn_terminal);
                graph.addDependency(prefix + "moe_combine", shared_ffn_last);
                ffn_terminal = prefix + "moe_combine";
            }
            else
            {
                // No shared expert — copy expert output to attn_proj directly
                // (or if MOE_COMBINED_OUTPUT IS attn_proj, this is a no-op)
                if (moe_output != buffers.attn_proj)
                {
                    ResidualAddStage::Params copy_params;
                    copy_params.device_id = device;
                    copy_params.input = moe_output;
                    copy_params.residual = nullptr;
                    copy_params.output = buffers.attn_proj;
                    copy_params.num_elements = static_cast<size_t>(total_tokens) * static_cast<size_t>(config_.d_model);
                    copy_params.input_buffer_id = buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                    copy_params.output_buffer_id = buffers.idFor(BufferId::ATTN_PROJ);

                    graph.addNode(prefix + "moe_combine",
                                  ComputeStageFactory::createResidualAdd(copy_params),
                                  device);
                    graph.addDependency(prefix + "moe_combine", ffn_terminal);
                    ffn_terminal = prefix + "moe_combine";
                }
            }
        }

        // =====================================================================
        // Stage 6: Explicit residual (last layer only)
        // =====================================================================
        const bool skip_ffn_residual = (layer_idx < config_.pp_layer_offset + config_.n_layers - 1);
        if (residual_read)
        {
            ffn_terminal = addBranchBlockWrite(graph, std::move(*residual_read), buffers, ffn_terminal,
                predictor ? &predictor->ffnOutput() : nullptr);
        }
        else if (!skip_ffn_residual)
        {
            ResidualAddStage::Params res_params;
            res_params.device_id = device;
            res_params.input = buffers.attn_proj;
            res_params.residual = buffers.current_hidden;
            res_params.output = buffers.current_hidden;
            res_params.num_elements = static_cast<size_t>(total_tokens) * static_cast<size_t>(config_.d_model);
            res_params.input_buffer_id = buffers.idFor(BufferId::ATTN_PROJ);
            res_params.residual_buffer_id = buffers.idFor(BufferId::HIDDEN_STATE);
            res_params.output_buffer_id = buffers.idFor(BufferId::HIDDEN_STATE);

            graph.addNode(prefix + "ffn_residual",
                          ComputeStageFactory::createResidualAdd(res_params),
                          device);
            graph.addDependency(prefix + "ffn_residual", ffn_terminal);
            ffn_terminal = prefix + "ffn_residual";
        }

        if (last_device_rebalance_decode_layer &&
            !env.moe_rebalance.device_rebalance_maintenance_graph &&
            !ffn_terminal.empty())
        {
            const auto binding_it =
                moe_graph_rebalance_bindings_.find(graphRebalanceDomainKey());
            if (binding_it != moe_graph_rebalance_bindings_.end())
            {
                if (deviceMoERebalanceModeUsesTransferSlots(binding_it->second.transfer_mode) &&
                    binding_it->second.local_transfer_slot_count > 0)
                {
                    const auto transfer_directory_it =
                        moe_transfer_slot_directories_.find(binding_it->second.transfer_key);
                    const auto transfer_state_it =
                        moe_rebalance_transfer_states_.find(
                            binding_it->second.transfer_key + ":workspace=" +
                            binding_it->second.workspace_name);
                    if (transfer_directory_it == moe_transfer_slot_directories_.end() ||
                        !transfer_directory_it->second ||
                        transfer_state_it == moe_rebalance_transfer_states_.end() ||
                        !transfer_state_it->second)
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE graph-side rebalance lost its late transfer-join binding for layer " +
                            std::to_string(layer_idx) + " on " + device.to_string());
                    }

                    const std::string join_node =
                        prefix + "moe_device_rebalance_transfer_join";
                    MoEDeviceRebalanceStage::Params join_params;
                    join_params.device_id = device;
                    join_params.tp_ctx = local_tp_ctx;
                    join_params.moe_runtime_table = binding_it->second.moe_runtime_table;
                    join_params.tp_device_idx = config_.tp_device_idx;
                    join_params.config = binding_it->second.config;
                    join_params.local_transfer_slots =
                        transfer_directory_it->second->deviceEntries();
                    join_params.local_transfer_slot_count =
                        transfer_directory_it->second->slotCount();
                    join_params.collective_payload_slot_bytes =
                        binding_it->second.collective_payload_slot_bytes;
                    join_params.collective_payload_slot_capacity =
                        binding_it->second.collective_payload_slot_capacity;
                    join_params.stage_name = join_node;
                    join_params.workspace_name = binding_it->second.workspace_name;
                    join_params.phase = DeviceMoERebalanceStagePhase::JoinTransfer;
                    join_params.transfer_mode = binding_it->second.transfer_mode;
                    join_params.transfer_state = transfer_state_it->second;

                    graph.addNode(join_node,
                                  ComputeStageFactory::createMoEDeviceRebalance(join_params),
                                  device);
                    graph.addDependency(join_node, ffn_terminal);
                    ffn_terminal = join_node;
                }
            }
        }

        if (!mtp_sidecar_context &&
            device.is_gpu() &&
            mirroredLayerDiagnosticsEnabled() &&
            !ffn_terminal.empty())
        {
            /*
             * Preserve the two values needed to reconstruct a transformer
             * layer boundary in this graph family:
             *
             *  - current_hidden is the accumulated residual after attention.
             *  - attn_proj is the current layer's FFN delta.
             *
             * The next layer's fused residual norm combines those tensors.
             * Capturing both therefore distinguishes an attention/residual
             * divergence from a routed/shared-FFN divergence without copying a
             * complete activation matrix. HiddenStateRowSelectStage records
             * only the final logical row, so storage remains
             * O(graph variants * layers * d_model) even when a request
             * contains thousands of prefill or verifier rows.
             *
             * Grouped verifier graphs need their own bank. Accepted-state
             * publication can select the next terminal hidden row directly
             * from a grouped verifier result; retaining only the older prefill
             * bank would hide the producer that actually supplied the failed
             * mailbox payload.
             */
            const std::string graph_regime =
                config_.grouped_mtp_verifier
                    ? "grouped_verifier"
                    : config_.live_mtp_request_batch_condition
                    ? "request_batch_decode"
                    : total_tokens == 1
                          ? "serial_decode"
                          : "prefill";
            const std::string graph_shape =
                graph_regime + "_m" + std::to_string(total_tokens);

            auto add_row_checkpoint =
                [&](const std::string &boundary,
                    const ITensor *source,
                    BufferId source_buffer_id,
                    std::optional<int> fixed_row,
                    const std::string &dependency) -> std::string
            {
                if (!source)
                {
                    throw std::runtime_error(
                        "Qwen35 MoE mirrored-layer diagnostics lost source tensor for " +
                        boundary);
                }

                const std::string checkpoint_name =
                    graph_shape + "_layer" + std::to_string(layer_idx) +
                    "_" + boundary;
                auto &checkpoint = mirrored_layer_checkpoints_[checkpoint_name];
                if (!checkpoint)
                {
                    checkpoint = std::make_unique<FP32Tensor>(
                        std::vector<size_t>{
                            1u,
                            static_cast<size_t>(config_.d_model)},
                        device);
                    /*
                     * Diagnostic checkpoints are graph outputs, not host
                     * tensors awaiting upload. Prepare their model-lifetime
                     * device storage before the executor validates output
                     * placement. This allocation exists only when the
                     * opt-in mirrored-layer diagnostic changes graph topology;
                     * normal inference never creates these tensors.
                     */
                    if (!checkpoint->allocateOnDevice(device))
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE could not allocate mirrored-layer "
                            "diagnostic checkpoint " +
                            checkpoint_name + " on " + device.to_string());
                    }
                }

                const std::string node_name =
                    prefix + "mirror_checkpoint_" + graph_shape + "_" + boundary;
                HiddenStateRowSelectStage::Params checkpoint_params;
                checkpoint_params.device_id = device;
                checkpoint_params.input = source;
                checkpoint_params.output = checkpoint.get();
                checkpoint_params.input_buffer_id = source_buffer_id;
                checkpoint_params.seq_len = total_tokens;
                checkpoint_params.d_model = config_.d_model;
                if (fixed_row)
                {
                    checkpoint_params.selected_row_idx = *fixed_row;
                    checkpoint_params.selection_policy =
                        HiddenStateRowSelectStage::SelectionPolicy::
                            FixedDeviceRow;
                }
                else
                {
                    checkpoint_params.selected_row_idx = total_tokens - 1;
                    configureMirroredCheckpointRowOwnership(
                        checkpoint_params,
                        total_tokens,
                        device,
                        sequence_lengths_device);
                }

                graph.addNode(
                    node_name,
                    ComputeStageFactory::createHiddenStateRowSelect(
                        checkpoint_params),
                    device);
                graph.addDependency(node_name, dependency);
                return node_name;
            };

            const auto add_layer_boundary_bank =
                [&](const std::string &boundary,
                    const ITensor *source,
                    BufferId source_buffer_id,
                    const std::string &dependency) -> std::string
            {
                std::string checkpoint_dependency = add_row_checkpoint(
                    boundary,
                    source,
                    source_buffer_id,
                    std::nullopt,
                    dependency);
                if (!config_.grouped_mtp_verifier)
                    return checkpoint_dependency;

                /*
                 * A later transaction can fail in row zero even when the
                 * grouped graph's terminal/bonus row remains correct. Keep
                 * every physical verifier row under this opt-in diagnostic so
                 * the first divergent transformer boundary is observable in
                 * one run instead of requiring one recapture per layer/row.
                 */
                for (int row = 0; row < total_tokens; ++row)
                {
                    checkpoint_dependency = add_row_checkpoint(
                        boundary + "_row" + std::to_string(row),
                        source,
                        source_buffer_id,
                        row,
                        checkpoint_dependency);
                }
                return checkpoint_dependency;
            };

            const std::string hidden_checkpoint =
                add_layer_boundary_bank(
                    "attention_residual",
                    buffers.current_hidden,
                    BufferId::HIDDEN_STATE,
                    ffn_terminal);
            ffn_terminal =
                add_layer_boundary_bank(
                    "ffn_delta",
                    buffers.attn_proj,
                    BufferId::ATTN_PROJ,
                    hidden_checkpoint);
        }

    }
} // namespace llaminar2
